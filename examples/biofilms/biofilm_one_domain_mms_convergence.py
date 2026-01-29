import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

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
from pycutfem.utils.biofilm_mms_one_domain_convergence import build_biofilm_one_domain_mms_trig_step
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


def _eoc(prev_h: float, h: float, prev_err: float, err: float) -> float:
    if not (prev_h > 0.0 and h > 0.0 and prev_err > 0.0 and err > 0.0):
        return float("nan")
    return float(math.log(prev_err / err) / math.log(prev_h / h))


def _run_one(
    *,
    nx: int,
    ny: int,
    qdeg: int,
    qerr: int,
    dt_val: float,
    theta: float,
    backend: str,
    newton_tol: float,
    max_it: int,
    k_det: float,
):
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

    mms = build_biofilm_one_domain_mms_trig_step(dt_val=dt_val, theta=theta, k_det=k_det)

    v_n.set_values_from_function(lambda x, y: mms.v(x, y, mms.t_n))
    u_n.set_values_from_function(lambda x, y: mms.u(x, y, mms.t_n))
    p_n.set_values_from_function(lambda x, y: float(mms.p(x, y, mms.t_n)))
    phi_n.set_values_from_function(lambda x, y: float(mms.phi(x, y, mms.t_n)))
    alpha_n.set_values_from_function(lambda x, y: float(mms.alpha(x, y, mms.t_n)))
    S_n.set_values_from_function(lambda x, y: float(mms.S(x, y, mms.t_n)))

    dt_c = Constant(dt_val)

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
    )

    solver.solve_time_interval(
        functions=[v_k, p_k, u_k, phi_k, alpha_k, S_k],
        prev_functions=[v_n, p_n, u_n, phi_n, alpha_n, S_n],
        aux_functions={"dt": dt_c},
        time_params=TimeStepperParameters(dt=dt_val, final_time=dt_val, max_steps=1, theta=theta),
    )

    t_err = mms.t_k
    err_v = dh.l2_error(
        v_k,
        exact={"v_x": lambda x, y: mms.v(x, y, t_err)[..., 0], "v_y": lambda x, y: mms.v(x, y, t_err)[..., 1]},
        fields=["v_x", "v_y"],
        quad_order=int(qerr),
        relative=False,
    )
    err_p = dh.l2_error(p_k, exact={"p": lambda x, y: mms.p(x, y, t_err)}, fields=["p"], quad_order=int(qerr), relative=False)
    err_u = dh.l2_error(
        u_k,
        exact={"u_x": lambda x, y: mms.u(x, y, t_err)[..., 0], "u_y": lambda x, y: mms.u(x, y, t_err)[..., 1]},
        fields=["u_x", "u_y"],
        quad_order=int(qerr),
        relative=False,
    )
    err_phi = dh.l2_error(phi_k, exact={"phi": lambda x, y: mms.phi(x, y, t_err)}, fields=["phi"], quad_order=int(qerr), relative=False)
    err_alpha = dh.l2_error(
        alpha_k, exact={"alpha": lambda x, y: mms.alpha(x, y, t_err)}, fields=["alpha"], quad_order=int(qerr), relative=False
    )
    err_S = dh.l2_error(S_k, exact={"S": lambda x, y: mms.S(x, y, t_err)}, fields=["S"], quad_order=int(qerr), relative=False)

    h = 1.0 / float(nx)
    return {
        "nx": int(nx),
        "h": float(h),
        "err_v": float(err_v),
        "err_p": float(err_p),
        "err_u": float(err_u),
        "err_phi": float(err_phi),
        "err_alpha": float(err_alpha),
        "err_S": float(err_S),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="h-convergence MMS study for the one-domain biofilm model (Newton + BE/CN step).")
    ap.add_argument("--nx-list", type=str, default="4,8,16", help="Comma-separated nx values (ny=nx).")
    ap.add_argument("--q", type=int, default=8, help="Assembly quadrature order (dx metadata + NewtonSolver quad_order).")
    ap.add_argument("--q-error", type=int, default=12, help="Quadrature order for error integrals.")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--newton-tol", type=float, default=1.0e-10)
    ap.add_argument("--max-it", type=int, default=30)
    ap.add_argument("--no-detachment", action="store_true", help="Set k_det=0 in the MMS/forcing.")
    ap.add_argument("--convergence", action="store_true", help="Save log-log convergence plot as a PNG.")
    ap.add_argument("--outdir", type=str, default="comparison_outputs", help="Directory for saving CSV/LaTeX tables (and plots).")
    args = ap.parse_args()

    nx_list = [int(s.strip()) for s in str(args.nx_list).split(",") if s.strip()]
    if not nx_list:
        raise ValueError("Empty --nx-list.")

    k_det = 0.0 if bool(args.no_detachment) else 0.2
    rows = [
        _run_one(
            nx=nx,
            ny=nx,
            qdeg=int(args.q),
            qerr=int(args.q_error),
            dt_val=float(args.dt),
            theta=float(args.theta),
            backend=str(args.backend),
            newton_tol=float(args.newton_tol),
            max_it=int(args.max_it),
            k_det=float(k_det),
        )
        for nx in nx_list
    ]

    print(f"\nOne-domain biofilm MMS convergence | backend={args.backend} | theta={float(args.theta):g} | dt={float(args.dt):g}")
    recs: list[dict] = []
    for i, r in enumerate(rows):
        prev = rows[i - 1] if i > 0 else None
        recs.append(
            {
                "nx": int(r["nx"]),
                "h": float(r["h"]),
                "err_v": float(r["err_v"]),
                "eoc_v": float("nan") if prev is None else _eoc(prev["h"], r["h"], prev["err_v"], r["err_v"]),
                "err_p": float(r["err_p"]),
                "eoc_p": float("nan") if prev is None else _eoc(prev["h"], r["h"], prev["err_p"], r["err_p"]),
                "err_u": float(r["err_u"]),
                "eoc_u": float("nan") if prev is None else _eoc(prev["h"], r["h"], prev["err_u"], r["err_u"]),
                "err_phi": float(r["err_phi"]),
                "eoc_phi": float("nan") if prev is None else _eoc(prev["h"], r["h"], prev["err_phi"], r["err_phi"]),
                "err_alpha": float(r["err_alpha"]),
                "eoc_alpha": float("nan") if prev is None else _eoc(prev["h"], r["h"], prev["err_alpha"], r["err_alpha"]),
                "err_S": float(r["err_S"]),
                "eoc_S": float("nan") if prev is None else _eoc(prev["h"], r["h"], prev["err_S"], r["err_S"]),
            }
        )

    df = pd.DataFrame.from_records(recs)
    print("\nConvergence table (pandas):")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 160):
        print(df)

    latex = df.to_latex(index=False, float_format="%.3e", na_rep="-")
    print("\nLaTeX table (copy/paste):\n")
    print(latex)

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"biofilm_mms_trig_backend={args.backend}_theta={float(args.theta):g}_dt={float(args.dt):g}"
    out_csv = outdir / f"{stem}.csv"
    out_tex = outdir / f"{stem}.tex"
    df.to_csv(out_csv, index=False)
    out_tex.write_text(latex, encoding="utf-8")
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_tex}")

    if bool(args.convergence):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] --convergence requested but matplotlib is unavailable: {exc}")
            return

        hs = np.asarray([r["h"] for r in rows], dtype=float)
        series = [
            ("v", np.asarray([r["err_v"] for r in rows], dtype=float)),
            ("p", np.asarray([r["err_p"] for r in rows], dtype=float)),
            ("u", np.asarray([r["err_u"] for r in rows], dtype=float)),
            ("phi", np.asarray([r["err_phi"] for r in rows], dtype=float)),
            ("alpha", np.asarray([r["err_alpha"] for r in rows], dtype=float)),
            ("S", np.asarray([r["err_S"] for r in rows], dtype=float)),
        ]

        fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)

        for name, errs in series:
            label = f"e({name})"
            if hs.size >= 2 and np.all(errs > 0.0):
                slope = float(np.polyfit(np.log(hs), np.log(errs), 1)[0])
                label = f"{label} (p≈{slope:.2f})"
            ax.loglog(hs, errs, marker="o", linewidth=1.5, label=label)

        ax.set_xlabel("h")
        ax.set_ylabel("L2 error at t_k")
        ax.grid(True, which="both", linestyle=":", linewidth=0.7)
        ax.legend(loc="best", fontsize=9)
        ax.set_title(f"Biofilm MMS convergence (backend={args.backend}, theta={float(args.theta):g}, dt={float(args.dt):g})")

        out = outdir / f"{stem}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"\nSaved convergence plot to {out}")


if __name__ == "__main__":
    main()
