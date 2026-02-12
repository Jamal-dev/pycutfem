import math

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

from examples.utils.biofilm.mms_one_domain_convergence import build_biofilm_one_domain_mms_trig_step
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


def _eoc(prev_h: float, h: float, prev_err: float, err: float) -> float:
    if not (prev_h > 0.0 and h > 0.0 and prev_err > 0.0 and err > 0.0):
        return float("nan")
    return float(math.log(prev_err / err) / math.log(prev_h / h))


def _solve_one(*, nx: int, qdeg: int, qerr: int, dt_val: float, theta: float, newton_tol: float, max_it: int, k_det: float, mms):
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
            "u_x": 2,
            "u_y": 2,
            "phi": 1,
            "alpha": 1,
            "S": 1,
            "X": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    dS = TrialFunction("S", dof_handler=dh)
    dX = TrialFunction("X", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = UFLTestFunction("p", dof_handler=dh)
    phi_test = UFLTestFunction("phi", dof_handler=dh)
    alpha_test = UFLTestFunction("alpha", dof_handler=dh)
    S_test = UFLTestFunction("S", dof_handler=dh)
    X_test = UFLTestFunction("X", dof_handler=dh)

    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh)
    X_k = Function("X_k", "X", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)
    X_n = Function("X_n", "X", dof_handler=dh)

    v_n.set_values_from_function(lambda x, y: mms.v(x, y, mms.t_n))
    u_n.set_values_from_function(lambda x, y: mms.u(x, y, mms.t_n))
    p_n.set_values_from_function(lambda x, y: float(mms.p(x, y, mms.t_n)))
    phi_n.set_values_from_function(lambda x, y: float(mms.phi(x, y, mms.t_n)))
    alpha_n.set_values_from_function(lambda x, y: float(mms.alpha(x, y, mms.t_n)))
    S_n.set_values_from_function(lambda x, y: float(mms.S(x, y, mms.t_n)))
    X_n.set_values_from_function(lambda x, y: float(mms.X(x, y, mms.t_n)))

    dt_c = Constant(float(dt_val))

    f_v = Analytic(lambda x, y: mms.f_v(x, y), degree=6)
    f_u = Analytic(lambda x, y: mms.f_u(x, y), degree=6)
    s_v = Analytic(lambda x, y: mms.s_v(x, y), degree=6)
    f_phi = Analytic(lambda x, y: mms.f_phi(x, y), degree=6)
    f_alpha = Analytic(lambda x, y: mms.f_alpha(x, y), degree=6)
    f_S = Analytic(lambda x, y: mms.f_S(x, y), degree=6)
    D_det_prev = Analytic(lambda x, y: mms.D_det_prev(x, y), degree=6)
    f_X = Analytic(lambda x, y: mms.f_X(x, y), degree=6)

    forms = build_biofilm_one_domain_forms(
        v_k=v_k,
        p_k=p_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        S_k=S_k,
        X_k=X_k,
        v_n=v_n,
        p_n=p_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        S_n=S_n,
        X_n=X_n,
        dv=dv,
        dp=dp,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dS=dS,
        dX=dX,
        v_test=v_test,
        q_test=q_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        S_test=S_test,
        X_test=X_test,
        dx=dx(metadata={"q": int(qdeg)}),
        dt=dt_c,
        theta=float(theta),
        rho_f=Constant(1.0),
        mu_f=Constant(1.0e-2),
        kappa_inv=Constant(10.0),
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        D_phi=0.1,
        gamma_phi=1.0,
        D_alpha=0.1,
        D_S=0.1,
        D_X=0.1,
        mu_max=0.4,
        K_S=0.3,
        k_g=0.5,
        k_d=0.1,
        Y=0.8,
        k_det=float(k_det),
        f_v=f_v,
        f_u=f_u,
        s_v=s_v,
        f_phi=f_phi,
        f_alpha=f_alpha,
        f_S=f_S,
        f_X=f_X,
        D_det_prev=D_det_prev,
    )

    bcs = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("v_x", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.v(x, y, t)[..., 0])),
                BoundaryCondition("v_y", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.v(x, y, t)[..., 1])),
                BoundaryCondition("p", "dirichlet", tag, _as_float_time(mms.p)),
                BoundaryCondition("u_x", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.u(x, y, t)[..., 0])),
                BoundaryCondition("u_y", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.u(x, y, t)[..., 1])),
                BoundaryCondition("phi", "dirichlet", tag, _as_float_time(mms.phi)),
                BoundaryCondition("alpha", "dirichlet", tag, _as_float_time(mms.alpha)),
                BoundaryCondition("S", "dirichlet", tag, _as_float_time(mms.S)),
                BoundaryCondition("X", "dirichlet", tag, _as_float_time(mms.X)),
            ]
        )
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0)) for b in bcs]

    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=float(newton_tol), max_newton_iter=int(max_it)),
        quad_order=int(qdeg),
        backend="cpp",
    )

    solver.solve_time_interval(
        functions=[v_k, p_k, u_k, phi_k, alpha_k, S_k, X_k],
        prev_functions=[v_n, p_n, u_n, phi_n, alpha_n, S_n, X_n],
        aux_functions={"dt": dt_c},
        time_params=TimeStepperParameters(dt=float(dt_val), final_time=float(dt_val), max_steps=1, theta=float(theta)),
    )

    t_err = float(mms.t_k)
    err_v = dh.l2_error(
        v_k,
        exact={"v_x": lambda x, y: mms.v(x, y, t_err)[..., 0], "v_y": lambda x, y: mms.v(x, y, t_err)[..., 1]},
        fields=["v_x", "v_y"],
        quad_order=int(qerr),
        relative=False,
    )
    err_p = dh.l2_error(p_k, exact={"p": lambda x, y: mms.p(x, y, t_err)}, fields=["p"], quad_order=int(qerr), relative=False)
    err_alpha = dh.l2_error(alpha_k, exact={"alpha": lambda x, y: mms.alpha(x, y, t_err)}, fields=["alpha"], quad_order=int(qerr), relative=False)

    return {
        "h": 1.0 / float(nx),
        "err_v": float(err_v),
        "err_p": float(err_p),
        "err_alpha": float(err_alpha),
    }


def test_biofilm_one_domain_mms_trig_convergence_cpp():
    # Keep the mesh sizes small: we only want to catch regressions where the MMS
    # forcing is inconsistent with `build_biofilm_one_domain_forms` (EOC ~ 0).
    dt_val = 0.05
    theta = 1.0
    k_det = 0.2
    qdeg = 8
    qerr = 8
    newton_tol = 1.0e-10
    max_it = 30

    mms = build_biofilm_one_domain_mms_trig_step(dt_val=float(dt_val), theta=float(theta), k_det=float(k_det))

    r4 = _solve_one(nx=4, qdeg=qdeg, qerr=qerr, dt_val=dt_val, theta=theta, newton_tol=newton_tol, max_it=max_it, k_det=k_det, mms=mms)
    r8 = _solve_one(nx=8, qdeg=qdeg, qerr=qerr, dt_val=dt_val, theta=theta, newton_tol=newton_tol, max_it=max_it, k_det=k_det, mms=mms)

    assert r8["err_v"] < r4["err_v"]
    assert r8["err_p"] < r4["err_p"]
    assert r8["err_alpha"] < r4["err_alpha"]

    eoc_v = _eoc(r4["h"], r8["h"], r4["err_v"], r8["err_v"])
    eoc_p = _eoc(r4["h"], r8["h"], r4["err_p"], r8["err_p"])
    eoc_alpha = _eoc(r4["h"], r8["h"], r4["err_alpha"], r8["err_alpha"])

    assert eoc_v > 1.7
    assert eoc_p > 1.4
    assert eoc_alpha > 1.7
