import argparse
import math
import os

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
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
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dx
from pycutfem.utils.biofilm_mms_moving_interface import BiofilmMovingInterfaceMMS
from pycutfem.utils.biofilm_one_domain import build_biofilm_one_domain_forms
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


def _as_float_time(fn):
    return lambda x, y, t: float(np.asarray(fn(np.asarray(x), np.asarray(y), float(t))).reshape(()))


def _sqrt(x: float) -> float:
    return float(math.sqrt(float(x)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Time-dependent MMS: moving diffuse interface (detachment-like) for the one-domain biofilm model.")
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--ny", type=int, default=16)
    ap.add_argument("--q", type=int, default=8, help="Assembly quadrature order.")
    ap.add_argument("--q-error", type=int, default=12, help="Quadrature order for error integrals.")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--final-time", type=float, default=0.1)
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--newton-tol", type=float, default=1.0e-10)
    ap.add_argument("--max-it", type=int, default=30)
    ap.add_argument("--outdir", type=str, default="examples/biofilms/results/mms_moving_interface", help="Directory for writing VTK output.")
    ap.add_argument("--vtk-every", type=int, default=0, help="Write VTK every N accepted steps (0 disables).")

    # MMS parameters
    ap.add_argument("--h0", type=float, default=0.4)
    ap.add_argument("--V-det", type=float, default=0.2)
    ap.add_argument("--phi-b", type=float, default=0.6)
    ap.add_argument("--S0", type=float, default=0.5)
    ap.add_argument("--a", type=float, default=0.2)
    ap.add_argument("--omega", type=float, default=2.0 * math.pi)
    ap.add_argument("--eps", type=float, default=None, help="Interface thickness epsilon. If omitted, eps = eps-factor*h.")
    ap.add_argument("--eps-factor", type=float, default=2.0)

    # Model knobs used by the forcing
    ap.add_argument("--rho-f", type=float, default=1.0)
    ap.add_argument("--mu-f", type=float, default=1.0e-2)
    ap.add_argument("--kappa-inv", type=float, default=10.0)
    ap.add_argument("--D-phi", type=float, default=0.1)
    ap.add_argument("--gamma-phi", type=float, default=1.0)
    ap.add_argument("--D-alpha", type=float, default=0.1)
    ap.add_argument("--k-det", type=float, default=0.2)
    ap.add_argument("--eta-n", type=float, default=1.0e-12)
    args = ap.parse_args()

    if float(args.dt) <= 0.0:
        raise ValueError("--dt must be positive.")
    if float(args.final_time) <= 0.0:
        raise ValueError("--final-time must be positive.")

    theta = 1.0  # This MMS is intended for BE usage.
    vtk_every = int(getattr(args, "vtk_every", 0) or 0)
    vtk_dir = os.path.join(str(args.outdir), "vtk")
    if vtk_every > 0:
        os.makedirs(vtk_dir, exist_ok=True)

    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=int(args.nx), ny=int(args.ny), poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    _tag_unit_square_boundaries(mesh)

    h = 1.0 / float(args.nx)
    eps = float(args.eps) if args.eps is not None else float(args.eps_factor) * h

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

    mms = BiofilmMovingInterfaceMMS(
        h0=float(args.h0),
        V_det=float(args.V_det),
        eps=float(eps),
        phi_b=float(args.phi_b),
        a=float(args.a),
        omega=float(args.omega),
        S0=float(args.S0),
        rho_f=float(args.rho_f),
        mu_f=float(args.mu_f),
        kappa_inv=float(args.kappa_inv),
        D_phi=float(args.D_phi),
        gamma_phi=float(args.gamma_phi),
        D_alpha=float(args.D_alpha),
        k_det=float(args.k_det),
        eta_n=float(args.eta_n),
        t_n=0.0,
        dt=float(args.dt),
    )

    # Initial condition at t=0
    t0 = 0.0
    v_n.set_values_from_function(lambda x, y: mms.v(x, y, t0))
    u_n.set_values_from_function(lambda x, y: mms.u(x, y, t0))
    p_n.set_values_from_function(lambda x, y: float(mms.p(x, y, t0)))
    phi_n.set_values_from_function(lambda x, y: float(mms.phi(x, y, t0)))
    alpha_n.set_values_from_function(lambda x, y: float(mms.alpha(x, y, t0)))
    S_n.set_values_from_function(lambda x, y: float(mms.S(x, y, t0)))

    dt_c = Constant(float(args.dt))

    # Time-dependent forcing: updated via NewtonSolver.preproc_cb.
    f_v = Analytic(lambda x, y: mms.f_v(x, y), degree=8)
    f_u = Analytic(lambda x, y: mms.f_u(x, y), degree=8)
    s_v = Analytic(lambda x, y: mms.s_v(x, y), degree=4)
    f_phi = Analytic(lambda x, y: mms.f_phi(x, y), degree=8)
    f_alpha = Analytic(lambda x, y: mms.f_alpha(x, y), degree=8)
    f_S = Analytic(lambda x, y: mms.f_S(x, y), degree=6)
    D_det_prev = Analytic(lambda x, y: mms.D_det_prev(x, y), degree=8)

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
        dx=dx(metadata={"q": int(args.q)}),
        dt=dt_c,
        theta=theta,
        rho_f=Constant(float(args.rho_f)),
        mu_f=Constant(float(args.mu_f)),
        kappa_inv=Constant(float(args.kappa_inv)),
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        D_phi=float(args.D_phi),
        gamma_phi=float(args.gamma_phi),
        D_alpha=float(args.D_alpha),
        D_S=0.0,
        mu_max=0.0,
        K_S=1.0,
        k_g=0.0,
        k_d=0.0,
        Y=1.0,
        k_det=float(args.k_det),
        f_v=f_v,
        f_u=f_u,
        s_v=s_v,
        f_phi=f_phi,
        f_alpha=f_alpha,
        f_S=f_S,
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
            ]
        )
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0)) for b in bcs]

    # Collect max-in-time and L2-in-time errors for the full time run.
    step_rows = []

    solver_ref = {}

    def _preproc_cb(_funcs):
        solver = solver_ref.get("solver")
        if solver is None:
            return
        mms.set_step_time(t_n=float(solver._current_t), dt=float(solver._current_dt))

    def _post_timeloop_cb(_funcs):
        solver = solver_ref.get("solver")
        if solver is None:
            return
        step_no = len(step_rows) + 1
        t_err = float(solver._current_t + solver._current_dt)

        err_v = dh.l2_error(
            v_k,
            exact={"v_x": lambda x, y: mms.v(x, y, t_err)[..., 0], "v_y": lambda x, y: mms.v(x, y, t_err)[..., 1]},
            fields=["v_x", "v_y"],
            quad_order=int(args.q_error),
            relative=False,
        )
        err_p = dh.l2_error(p_k, exact={"p": lambda x, y: mms.p(x, y, t_err)}, fields=["p"], quad_order=int(args.q_error), relative=False)
        err_u = dh.l2_error(
            u_k,
            exact={"u_x": lambda x, y: mms.u(x, y, t_err)[..., 0], "u_y": lambda x, y: mms.u(x, y, t_err)[..., 1]},
            fields=["u_x", "u_y"],
            quad_order=int(args.q_error),
            relative=False,
        )
        err_phi = dh.l2_error(phi_k, exact={"phi": lambda x, y: mms.phi(x, y, t_err)}, fields=["phi"], quad_order=int(args.q_error), relative=False)
        err_alpha = dh.l2_error(
            alpha_k, exact={"alpha": lambda x, y: mms.alpha(x, y, t_err)}, fields=["alpha"], quad_order=int(args.q_error), relative=False
        )
        err_S = dh.l2_error(S_k, exact={"S": lambda x, y: mms.S(x, y, t_err)}, fields=["S"], quad_order=int(args.q_error), relative=False)
        step_rows.append(
            {
                "t": float(t_err),
                "dt": float(solver._current_dt),
                "err_v": float(err_v),
                "err_p": float(err_p),
                "err_u": float(err_u),
                "err_phi": float(err_phi),
                "err_alpha": float(err_alpha),
                "err_S": float(err_S),
            }
        )
        if vtk_every > 0 and (step_no % vtk_every == 0):
            export_vtk(
                os.path.join(vtk_dir, f"mms_moving_interface_step={step_no:04d}.vtu"),
                mesh,
                dh,
                {"v": v_k, "p": p_k, "u": u_k, "phi": phi_k, "alpha": alpha_k, "S": S_k},
            )

    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=float(args.newton_tol), max_newton_iter=int(args.max_it)),
        quad_order=int(args.q),
        backend=str(args.backend),
        preproc_cb=_preproc_cb,
        postproc_timeloop_cb=_post_timeloop_cb,
    )
    solver_ref["solver"] = solver

    solver.solve_time_interval(
        functions=[v_k, p_k, u_k, phi_k, alpha_k, S_k],
        prev_functions=[v_n, p_n, u_n, phi_n, alpha_n, S_n],
        aux_functions={"dt": dt_c},
        time_params=TimeStepperParameters(dt=float(args.dt), final_time=float(args.final_time), max_steps=int(1.0e9), theta=theta, t0=t0),
    )

    if not step_rows:
        raise RuntimeError("No time steps executed (check --final-time and --dt).")

    def _agg_max(key: str) -> float:
        return float(max(r[key] for r in step_rows))

    def _agg_l2(key: str) -> float:
        return _sqrt(sum(float(r["dt"]) * float(r[key]) * float(r[key]) for r in step_rows))

    print(
        f"\nMoving-interface MMS time run | backend={args.backend} | nx={args.nx} | dt={float(args.dt):g} | final={float(args.final_time):g}\n"
        f"  eps={eps:.3e} (h={h:.3e}, eps_factor={float(args.eps_factor):g})  h0={float(args.h0):g}  V_det={float(args.V_det):g}\n"
        f"  steps={len(step_rows)}  t_end={step_rows[-1]['t']:.6g}"
    )
    print("  Max-in-time L2 errors:")
    print(
        f"    v={_agg_max('err_v'):.3e}  p={_agg_max('err_p'):.3e}  u={_agg_max('err_u'):.3e}  "
        f"phi={_agg_max('err_phi'):.3e}  alpha={_agg_max('err_alpha'):.3e}  S={_agg_max('err_S'):.3e}"
    )
    print("  L2-in-time(L2-in-space) errors:")
    print(
        f"    v={_agg_l2('err_v'):.3e}  p={_agg_l2('err_p'):.3e}  u={_agg_l2('err_u'):.3e}  "
        f"phi={_agg_l2('err_phi'):.3e}  alpha={_agg_l2('err_alpha'):.3e}  S={_agg_l2('err_S'):.3e}"
    )


if __name__ == "__main__":
    main()
