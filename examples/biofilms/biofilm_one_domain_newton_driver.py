import argparse

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
)
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.measures import dx
from examples.utils.biofilm.mms_one_domain_convergence import build_biofilm_one_domain_mms_trig_step
from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms
from pycutfem.utils.meshgen import structured_quad


def _tag_unit_square_boundaries(mesh: Mesh, *, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - 1.0) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - 1.0) <= tol,
        }
    )


def _as_float(fn):
    return lambda x, y: float(np.asarray(fn(np.asarray(x), np.asarray(y))).reshape(()))


def _as_float_time(fn):
    return lambda x, y, t: float(np.asarray(fn(np.asarray(x), np.asarray(y), float(t))).reshape(()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Newton time-step driver for the one-domain biofilm MMS problem.")
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--q", type=int, default=8, help="Quadrature order (dx metadata + NewtonSolver quad_order).")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--theta", type=float, default=1.0, help="One-step theta (1.0=BE, 0.5=CN).")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--newton-tol", type=float, default=1.0e-10)
    ap.add_argument("--max-it", type=int, default=30)
    ap.add_argument("--no-detachment", action="store_true", help="Set k_det=0 in the MMS/forcing.")
    args = ap.parse_args()

    dt_val = float(args.dt)
    theta = float(args.theta)
    qdeg = int(args.q)

    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=int(args.nx), ny=int(args.ny), poly_order=2)
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

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    S_test = TestFunction("S", dof_handler=dh)

    # Unknowns (k) and previous state (n)
    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)

    # MMS forcing for a single step t_n=0 -> t_k=dt
    k_det = 0.0 if bool(args.no_detachment) else 0.2
    mms = build_biofilm_one_domain_mms_trig_step(dt_val=dt_val, theta=theta, k_det=k_det)

    # Previous-step initial conditions
    v_n.set_values_from_function(lambda x, y: mms.v(x, y, mms.t_n))
    u_n.set_values_from_function(lambda x, y: mms.u(x, y, mms.t_n))
    p_n.set_values_from_function(lambda x, y: float(mms.p(x, y, mms.t_n)))
    phi_n.set_values_from_function(lambda x, y: float(mms.phi(x, y, mms.t_n)))
    alpha_n.set_values_from_function(lambda x, y: float(mms.alpha(x, y, mms.t_n)))
    S_n.set_values_from_function(lambda x, y: float(mms.S(x, y, mms.t_n)))

    dt_c = Constant(dt_val)

    # Forcing terms (discrete θ-step strong-form RHS)
    f_v = Analytic(lambda x, y: mms.f_v(x, y), degree=6)
    f_u = Analytic(lambda x, y: mms.f_u(x, y), degree=6)
    s_v = Analytic(lambda x, y: mms.s_v(x, y), degree=6)
    f_phi = Analytic(lambda x, y: mms.f_phi(x, y), degree=6)
    f_alpha = Analytic(lambda x, y: mms.f_alpha(x, y), degree=6)
    f_S = Analytic(lambda x, y: mms.f_S(x, y), degree=6)
    D_det_prev = Analytic(lambda x, y: mms.D_det_prev(x, y), degree=6)

    forms = build_biofilm_one_domain_forms(
        v_k=v_k,
        p_k=p_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        S_k=S_k,
        v_n=v_n,
        p_n=p_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        S_n=S_n,
        dv=dv,
        dp=dp,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dS=dS,
        v_test=v_test,
        q_test=q_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        S_test=S_test,
        dx=dx(metadata={"q": int(qdeg)}),
        dt=dt_c,
        theta=theta,
        rho_f=Constant(1.0),
        mu_f=Constant(1.0e-2),
        kappa_inv=Constant(10.0),
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        D_phi=0.1,
        gamma_phi=1.0,
        D_alpha=0.1,
        D_S=0.1,
        mu_max=0.4,
        K_S=0.3,
        k_g=0.5,
        k_d=0.1,
        Y=0.8,
        k_det=k_det,
        f_v=f_v,
        f_u=f_u,
        s_v=s_v,
        f_phi=f_phi,
        f_alpha=f_alpha,
        f_S=f_S,
        D_det_prev=D_det_prev,
    )

    # Time-dependent Dirichlet BCs are frozen at t_bc = t_n + θ*dt by the solver.
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
            ]
        )
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, _as_float(lambda x, y: 0.0)) for b in bcs]

    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=float(args.newton_tol), max_newton_iter=int(args.max_it)),
        quad_order=qdeg,
        backend=str(args.backend),
    )

    solver.solve_time_interval(
        functions=[v_k, p_k, u_k, phi_k, alpha_k, S_k],
        prev_functions=[v_n, p_n, u_n, phi_n, alpha_n, S_n],
        aux_functions={"dt": dt_c},
        time_params=TimeStepperParameters(dt=dt_val, final_time=dt_val, max_steps=1, theta=theta),
    )

    # L2 errors at t_k
    t_err = mms.t_k
    exact = {
        "v_x": lambda x, y: mms.v(x, y, t_err)[..., 0],
        "v_y": lambda x, y: mms.v(x, y, t_err)[..., 1],
    }
    err_v = dh.l2_error(v_k, exact=exact, fields=["v_x", "v_y"], quad_order=qdeg, relative=False)
    err_p = dh.l2_error(p_k, exact={"p": lambda x, y: mms.p(x, y, t_err)}, fields=["p"], quad_order=qdeg, relative=False)
    err_u = dh.l2_error(
        u_k,
        exact={"u_x": lambda x, y: mms.u(x, y, t_err)[..., 0], "u_y": lambda x, y: mms.u(x, y, t_err)[..., 1]},
        fields=["u_x", "u_y"],
        quad_order=qdeg,
        relative=False,
    )
    err_phi = dh.l2_error(phi_k, exact={"phi": lambda x, y: mms.phi(x, y, t_err)}, fields=["phi"], quad_order=qdeg, relative=False)
    err_alpha = dh.l2_error(
        alpha_k, exact={"alpha": lambda x, y: mms.alpha(x, y, t_err)}, fields=["alpha"], quad_order=qdeg, relative=False
    )
    err_S = dh.l2_error(S_k, exact={"S": lambda x, y: mms.S(x, y, t_err)}, fields=["S"], quad_order=qdeg, relative=False)

    print(f"backend={args.backend}  nx={args.nx}  dt={dt_val:g}  theta={theta:g}  q={qdeg}")
    print(f"  L2 errors: |e(v)|={err_v:.3e}  |e(p)|={err_p:.3e}  |e(u)|={err_u:.3e}  |e(phi)|={err_phi:.3e}  |e(alpha)|={err_alpha:.3e}  |e(S)|={err_S:.3e}")


if __name__ == "__main__":
    main()

