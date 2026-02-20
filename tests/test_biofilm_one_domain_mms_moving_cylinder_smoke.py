import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction as UFLTestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
)
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from examples.utils.biofilm.mms_moving_cylinder import BiofilmMovingCylinderMMS
from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms


def _tag_unit_square_boundaries(mesh: Mesh, *, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - 1.0) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - 1.0) <= tol,
        }
    )


def _as_float_time(fn):
    return lambda x, y, t: float(np.asarray(fn(np.asarray(x), np.asarray(y), float(t))).reshape(()))


def test_biofilm_one_domain_mms_moving_cylinder_smoke_chunk_transport_python() -> None:
    """
    MMS 3 smoke regression: rigid-chunk transport is solvable with u-extension=grad.

    This is a lightweight integration test:
    - runs 1 BE step on a tiny mesh (nx=2) with the moving-cylinder MMS forcing;
    - injects the MMS end-of-step state as the Newton initial guess for each step
      (so runtime stays manageable while still exercising time-dependent forcing);
    - checks the final-time L2 errors remain bounded.
    """
    nx = 2
    qdeg = 5
    qerr = 5
    dt_val = 0.02
    nsteps = 1

    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=int(nx), ny=int(nx), poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    _tag_unit_square_boundaries(mesh)

    me = MixedElement(
        mesh,
        field_specs={
            "v_x": 2,
            "v_y": 2,
            "p": 1,
            "vS_x": 2,
            "vS_y": 2,
            "u_x": 2,
            "u_y": 2,
            "phi": 1,
            "alpha": 1,
            "S": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    dvS = VectorTrialFunction(space=VS, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    dS = TrialFunction("S", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = UFLTestFunction("p", dof_handler=dh)
    phi_test = UFLTestFunction("phi", dof_handler=dh)
    alpha_test = UFLTestFunction("alpha", dof_handler=dh)
    S_test = UFLTestFunction("S", dof_handler=dh)

    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    vS_k = VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    vS_n = VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)

    mms = BiofilmMovingCylinderMMS(dt=float(dt_val), D_phi=0.0, gamma_phi=1.0, D_alpha=0.0)

    # Initial condition at t=0.
    t0 = 0.0
    v_n.set_values_from_function(lambda x, y: mms.v(x, y, t0))
    vS_n.set_values_from_function(lambda x, y: np.stack((0.0 * np.asarray(x, dtype=float), 0.0 * np.asarray(x, dtype=float)), axis=-1))
    u_n.set_values_from_function(lambda x, y: mms.u(x, y, t0))
    p_n.set_values_from_function(lambda x, y: float(mms.p(x, y, t0)))
    phi_n.set_values_from_function(lambda x, y: float(mms.phi(x, y, t0)))
    alpha_n.set_values_from_function(lambda x, y: float(mms.alpha(x, y, t0)))
    S_n.set_values_from_function(lambda x, y: float(mms.S(x, y, t0)))

    dt_c = Constant(float(dt_val))

    # Time-dependent forcing (mms.set_step_time is called from preproc_cb).
    f_v = Analytic(lambda x, y: mms.f_v(x, y), degree=10)
    f_u = Analytic(lambda x, y: mms.f_u(x, y), degree=10)
    s_v = Analytic(lambda x, y: mms.s_v(x, y), degree=6)
    f_phi = Analytic(lambda x, y: mms.f_phi(x, y), degree=10)
    f_alpha = Analytic(lambda x, y: mms.f_alpha(x, y), degree=10)
    f_S = Analytic(lambda x, y: mms.f_S(x, y), degree=6)

    forms = build_biofilm_one_domain_forms(
        v_k=v_k,
        p_k=p_k,
        vS_k=vS_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        S_k=S_k,
        v_n=v_n,
        p_n=p_n,
        vS_n=vS_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        S_n=S_n,
        dv=dv,
        dp=dp,
        dvS=dvS,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dS=dS,
        v_test=v_test,
        q_test=q_test,
        vS_test=vS_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        S_test=S_test,
        dx=dx(metadata={"q": int(qdeg)}),
        dt=dt_c,
        theta=1.0,
        rho_f=Constant(float(mms.rho_f)),
        mu_f=Constant(float(mms.mu_f)),
        kappa_inv=Constant(float(mms.kappa_inv)),
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        gamma_u=1.0,
        u_extension_mode="grad",
        D_phi=float(mms.D_phi),
        gamma_phi=float(mms.gamma_phi),
        D_alpha=float(mms.D_alpha),
        D_S=0.0,
        mu_max=0.0,
        K_S=1.0,
        k_g=0.0,
        k_d=0.0,
        Y=1.0,
        k_det=0.0,
        mu_b_model="mu",
        f_v=f_v,
        f_u=f_u,
        s_v=s_v,
        f_phi=f_phi,
        f_alpha=f_alpha,
        f_S=f_S,
    )

    bcs = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("v_x", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.v(x, y, t)[..., 0])),
                BoundaryCondition("v_y", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.v(x, y, t)[..., 1])),
                BoundaryCondition("p", "dirichlet", tag, _as_float_time(mms.p)),
                BoundaryCondition("vS_x", "dirichlet", tag, _as_float_time(lambda x, y, t: float(mms.vS(float(t))[0]))),
                BoundaryCondition("vS_y", "dirichlet", tag, _as_float_time(lambda x, y, t: float(mms.vS(float(t))[1]))),
                BoundaryCondition("u_x", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.u(x, y, t)[..., 0])),
                BoundaryCondition("u_y", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.u(x, y, t)[..., 1])),
                BoundaryCondition("phi", "dirichlet", tag, _as_float_time(mms.phi)),
                BoundaryCondition("alpha", "dirichlet", tag, _as_float_time(mms.alpha)),
                BoundaryCondition("S", "dirichlet", tag, _as_float_time(mms.S)),
            ]
        )
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0)) for b in bcs]

    solver_ref = {}
    last_init_step = {"step_no": None}

    def _preproc_cb(_funcs):
        solver = solver_ref.get("solver")
        if solver is None:
            return
        mms.set_step_time(t_n=float(solver._current_t), dt=float(solver._current_dt))

        step_no = int(getattr(solver, "_current_step_no", -1))
        if last_init_step["step_no"] == step_no:
            return

        t_guess = float(solver._current_t + solver._current_dt)  # theta=1
        v_k.set_values_from_function(lambda x, y: mms.v(x, y, t_guess))
        vS_guess = np.asarray(mms.vS(t_guess), dtype=float).reshape((2,))
        vS_k.set_values_from_function(
            lambda x, y, vS_guess=vS_guess: np.stack(
                (
                    0.0 * np.asarray(x, dtype=float) + float(vS_guess[0]),
                    0.0 * np.asarray(x, dtype=float) + float(vS_guess[1]),
                ),
                axis=-1,
            )
        )
        u_k.set_values_from_function(lambda x, y: mms.u(x, y, t_guess))
        p_k.set_values_from_function(lambda x, y: float(mms.p(x, y, t_guess)))
        phi_k.set_values_from_function(lambda x, y: float(mms.phi(x, y, t_guess)))
        alpha_k.set_values_from_function(lambda x, y: float(mms.alpha(x, y, t_guess)))
        S_k.set_values_from_function(lambda x, y: float(mms.S(x, y, t_guess)))
        last_init_step["step_no"] = step_no

    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=1.0e-6, max_newton_iter=4, line_search=False),
        quad_order=int(qdeg),
        backend="python",
        preproc_cb=_preproc_cb,
    )
    solver_ref["solver"] = solver

    _last_delta, n_done, _elapsed = solver.solve_time_interval(
        functions=[v_k, p_k, vS_k, u_k, phi_k, alpha_k, S_k],
        prev_functions=[v_n, p_n, vS_n, u_n, phi_n, alpha_n, S_n],
        aux_functions={"dt": dt_c},
        time_params=TimeStepperParameters(dt=float(dt_val), final_time=float(nsteps) * float(dt_val), max_steps=int(nsteps), theta=1.0, t0=t0),
    )
    assert int(n_done) == int(nsteps)

    t_err = float(nsteps) * float(dt_val)
    err_alpha = dh.l2_error(alpha_k, exact={"alpha": lambda x, y: mms.alpha(x, y, t_err)}, fields=["alpha"], quad_order=int(qerr), relative=False)
    err_u = dh.l2_error(
        u_k,
        exact={"u_x": lambda x, y: mms.u(x, y, t_err)[..., 0], "u_y": lambda x, y: mms.u(x, y, t_err)[..., 1]},
        fields=["u_x", "u_y"],
        quad_order=int(qerr),
        relative=False,
    )

    assert float(err_u) < 1.0e-3
    assert float(err_alpha) < 0.3
