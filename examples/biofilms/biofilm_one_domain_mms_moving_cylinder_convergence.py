import argparse
import math
import os

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
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dx
from pycutfem.utils.biofilm_mms_moving_cylinder import BiofilmMovingCylinderMMS
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


def _eoc(h0: float, h1: float, e0: float, e1: float) -> float:
    if not (h0 > 0.0 and h1 > 0.0 and e0 > 0.0 and e1 > 0.0):
        return float("nan")
    return float(math.log(e1 / e0) / math.log(h1 / h0))


def _run_one(
    *,
    nx: int,
    ny: int,
    qdeg: int,
    qerr: int,
    dt_val: float,
    nsteps: int,
    backend: str,
    newton_tol: float,
    max_it: int,
    # MMS parameters
    x0: float,
    y0: float,
    Ax: float,
    Ay: float,
    omega: float,
    R: float,
    eps: float,
    phi_b: float,
    Omega0: float,
    sigma: float,
    # model/forcing parameters
    rho_f: float,
    mu_f: float,
    kappa_inv: float,
    D_phi: float,
    gamma_phi: float,
    D_alpha: float,
):
    if dt_val <= 0.0:
        raise ValueError("dt must be positive.")
    if nsteps <= 0:
        raise ValueError("nsteps must be positive.")

    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=int(nx), ny=int(ny), poly_order=2)
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
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    S_test = TestFunction("S", dof_handler=dh)
    X_test = TestFunction("X", dof_handler=dh)

    # Unknowns (k) and previous state (n)
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

    mms = BiofilmMovingCylinderMMS(
        x0=float(x0),
        y0=float(y0),
        Ax=float(Ax),
        Ay=float(Ay),
        omega=float(omega),
        R=float(R),
        eps=float(eps),
        phi_b=float(phi_b),
        Omega0=float(Omega0),
        sigma=float(sigma),
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        kappa_inv=float(kappa_inv),
        D_phi=float(D_phi),
        gamma_phi=float(gamma_phi),
        D_alpha=float(D_alpha),
        t_n=0.0,
        dt=float(dt_val),
    )

    # Initial condition at t=0
    t0 = 0.0
    v_n.set_values_from_function(lambda x, y: mms.v(x, y, t0))
    u_n.set_values_from_function(lambda x, y: mms.u(x, y, t0))
    p_n.set_values_from_function(lambda x, y: float(mms.p(x, y, t0)))
    phi_n.set_values_from_function(lambda x, y: float(mms.phi(x, y, t0)))
    alpha_n.set_values_from_function(lambda x, y: float(mms.alpha(x, y, t0)))
    S_n.set_values_from_function(lambda x, y: float(mms.S(x, y, t0)))
    X_n.set_values_from_function(lambda x, y: float(mms.X(x, y, t0)))

    dt_c = Constant(float(dt_val))

    # Time-dependent forcing: updated via NewtonSolver.preproc_cb.
    f_v = Analytic(lambda x, y: mms.f_v(x, y), degree=10)
    f_u = Analytic(lambda x, y: mms.f_u(x, y), degree=10)
    s_v = Analytic(lambda x, y: mms.s_v(x, y), degree=6)
    f_phi = Analytic(lambda x, y: mms.f_phi(x, y), degree=10)
    f_alpha = Analytic(lambda x, y: mms.f_alpha(x, y), degree=10)
    f_S = Analytic(lambda x, y: mms.f_S(x, y), degree=6)
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
        theta=1.0,
        rho_f=Constant(float(rho_f)),
        mu_f=Constant(float(mu_f)),
        kappa_inv=Constant(float(kappa_inv)),
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        # No extra u-regularization needed for MMS (u is Dirichlet everywhere).
        gamma_u=0.0,
        D_phi=float(D_phi),
        gamma_phi=float(gamma_phi),
        D_alpha=float(D_alpha),
        D_S=0.0,
        D_X=0.0,
        # No growth/detachment for this MMS.
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
        f_X=f_X,
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
        t_err = float(solver._current_t + solver._current_dt)
        err_v = dh.l2_error(
            v_k,
            exact={"v_x": lambda x, y: mms.v(x, y, t_err)[..., 0], "v_y": lambda x, y: mms.v(x, y, t_err)[..., 1]},
            fields=["v_x", "v_y"],
            quad_order=int(qerr),
            relative=False,
        )
        err_phi = dh.l2_error(phi_k, exact={"phi": lambda x, y: mms.phi(x, y, t_err)}, fields=["phi"], quad_order=int(qerr), relative=False)
        err_alpha = dh.l2_error(alpha_k, exact={"alpha": lambda x, y: mms.alpha(x, y, t_err)}, fields=["alpha"], quad_order=int(qerr), relative=False)
        err_u = dh.l2_error(
            u_k,
            exact={"u_x": lambda x, y: mms.u(x, y, t_err)[..., 0], "u_y": lambda x, y: mms.u(x, y, t_err)[..., 1]},
            fields=["u_x", "u_y"],
            quad_order=int(qerr),
            relative=False,
        )
        step_rows.append({"dt": float(solver._current_dt), "err_v": float(err_v), "err_u": float(err_u), "err_phi": float(err_phi), "err_alpha": float(err_alpha)})

    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=float(newton_tol), max_newton_iter=int(max_it)),
        quad_order=int(qdeg),
        backend=str(backend),
        preproc_cb=_preproc_cb,
        postproc_timeloop_cb=_post_timeloop_cb,
    )
    solver_ref["solver"] = solver

    solver.solve_time_interval(
        functions=[v_k, p_k, u_k, phi_k, alpha_k, S_k, X_k],
        prev_functions=[v_n, p_n, u_n, phi_n, alpha_n, S_n, X_n],
        aux_functions={"dt": dt_c},
        time_params=TimeStepperParameters(dt=float(dt_val), final_time=float(nsteps) * float(dt_val), max_steps=int(nsteps), theta=1.0, t0=t0),
    )

    if not step_rows:
        raise RuntimeError("No time steps executed.")

    def _agg_max(key: str) -> float:
        return float(max(r[key] for r in step_rows))

    def _agg_l2(key: str) -> float:
        return float(math.sqrt(sum(float(r["dt"]) * float(r[key]) * float(r[key]) for r in step_rows)))

    h = 1.0 / float(nx)
    return {
        "nx": int(nx),
        "h": float(h),
        "eps": float(eps),
        "ev_max": _agg_max("err_v"),
        "ev_l2t": _agg_l2("err_v"),
        "eu_max": _agg_max("err_u"),
        "eu_l2t": _agg_l2("err_u"),
        "ephi_max": _agg_max("err_phi"),
        "ephi_l2t": _agg_l2("err_phi"),
        "ealpha_max": _agg_max("err_alpha"),
        "ealpha_l2t": _agg_l2("err_alpha"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="h-convergence for the moving-cylinder MMS time test (rigid-chunk motion).")
    ap.add_argument("--nx-list", type=str, default="8,16,32", help="Comma-separated nx values (ny=nx).")
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--q-error", type=int, default=12)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--nsteps", type=int, default=5)
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--newton-tol", type=float, default=1.0e-10)
    ap.add_argument("--max-it", type=int, default=30)
    ap.add_argument("--outdir", type=str, default="examples/biofilms/results/moving_cylinder", help="Directory for saving CSV/LaTeX tables.")

    # MMS parameters
    ap.add_argument("--x0", type=float, default=0.5)
    ap.add_argument("--y0", type=float, default=0.5)
    ap.add_argument("--Ax", type=float, default=0.15)
    ap.add_argument("--Ay", type=float, default=0.0)
    ap.add_argument("--omega", type=float, default=2.0 * math.pi)
    ap.add_argument("--R", type=float, default=0.18)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--phi-b", type=float, default=0.6)
    ap.add_argument("--Omega0", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.25)

    # model/forcing parameters
    ap.add_argument("--rho-f", type=float, default=1.0)
    ap.add_argument("--mu-f", type=float, default=1.0e-2)
    ap.add_argument("--kappa-inv", type=float, default=10.0)
    ap.add_argument("--D-phi", type=float, default=0.0)
    ap.add_argument("--gamma-phi", type=float, default=0.0)
    ap.add_argument("--D-alpha", type=float, default=0.0)
    args = ap.parse_args()

    nx_list = [int(s.strip()) for s in str(args.nx_list).split(",") if s.strip()]
    if not nx_list:
        raise ValueError("Empty --nx-list.")

    rows = []
    for nx in nx_list:
        rows.append(
            _run_one(
                nx=int(nx),
                ny=int(nx),
                qdeg=int(args.q),
                qerr=int(args.q_error),
                dt_val=float(args.dt),
                nsteps=int(args.nsteps),
                backend=str(args.backend),
                newton_tol=float(args.newton_tol),
                max_it=int(args.max_it),
                x0=float(args.x0),
                y0=float(args.y0),
                Ax=float(args.Ax),
                Ay=float(args.Ay),
                omega=float(args.omega),
                R=float(args.R),
                eps=float(args.eps),
                phi_b=float(args.phi_b),
                Omega0=float(args.Omega0),
                sigma=float(args.sigma),
                rho_f=float(args.rho_f),
                mu_f=float(args.mu_f),
                kappa_inv=float(args.kappa_inv),
                D_phi=float(args.D_phi),
                gamma_phi=float(args.gamma_phi),
                D_alpha=float(args.D_alpha),
            )
        )

    # Post-process: EOC table.
    table_rows = []
    prev = None
    for r in rows:
        row = dict(r)
        row["eoc_ev_l2t"] = float("nan") if prev is None else _eoc(prev["h"], r["h"], prev["ev_l2t"], r["ev_l2t"])
        row["eoc_ealpha_l2t"] = float("nan") if prev is None else _eoc(prev["h"], r["h"], prev["ealpha_l2t"], r["ealpha_l2t"])
        table_rows.append(row)
        prev = r

    import pandas as pd

    df = pd.DataFrame(table_rows)
    os.makedirs(str(args.outdir), exist_ok=True)
    stem = f"biofilm_mms_moving_cylinder_backend={args.backend}_dt={args.dt}_nsteps={args.nsteps}"
    csv_path = os.path.join(str(args.outdir), f"{stem}.csv")
    tex_path = os.path.join(str(args.outdir), f"{stem}.tex")
    df.to_csv(csv_path, index=False)
    df.to_latex(tex_path, index=False, float_format=lambda x: f"{x:.3e}")

    print(df)
    print()
    print(df.to_latex(index=False, float_format=lambda x: f"{x:.3e}"))


if __name__ == "__main__":
    main()
