import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _k in (
    "NUMBA_DEBUG",
    "NUMBA_DUMP_BYTECODE",
    "NUMBA_DUMP_IR",
    "NUMBA_DUMP_SSA",
    "NUMBA_DEBUG_ARRAY_OPT",
):
    os.environ[_k] = "0"

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, NewtonSolver
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from examples.utils.biofilm.deformation_only import build_deformation_only_forms
from examples.utils.biofilm.mms_deformation_only import (
    DeformationOnlyMMS,
    build_deformation_only_mms_shear,
    build_deformation_only_mms_static,
    build_deformation_only_mms_translation,
)


FULL_DOMAIN_LS = AffineLevelSet(1.0, 0.0, -2.0)
ALPHA_TRANSPORT_VELOCITY = "biofilm_volume"
ALPHA_TRANSPORT_FORM = "conservative_weak"


def _serialize_param(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


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


def _build_case(case: str, *, dt_val: float, theta: float) -> DeformationOnlyMMS:
    key = str(case).strip().lower()
    if key == "static":
        return build_deformation_only_mms_static(dt_val=dt_val, theta=theta)
    if key == "translation":
        return build_deformation_only_mms_translation(dt_val=dt_val, theta=theta)
    if key == "shear":
        return build_deformation_only_mms_shear(dt_val=dt_val, theta=theta)
    raise ValueError(f"Unknown case {case!r}.")


def _create_problem(nx: int):
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
            "alpha": 1,
            "mu_alpha": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    problem = {
        "mesh": mesh,
        "me": me,
        "dh": dh,
        "dv": VectorTrialFunction(space=V, dof_handler=dh),
        "dvS": VectorTrialFunction(space=VS, dof_handler=dh),
        "du": VectorTrialFunction(space=U, dof_handler=dh),
        "dp": TrialFunction("p", dof_handler=dh),
        "dalpha": TrialFunction("alpha", dof_handler=dh),
        "dmu": TrialFunction("mu_alpha", dof_handler=dh),
        "v_test": VectorTestFunction(space=V, dof_handler=dh),
        "vS_test": VectorTestFunction(space=VS, dof_handler=dh),
        "u_test": VectorTestFunction(space=U, dof_handler=dh),
        "q_test": TestFunction("p", dof_handler=dh),
        "alpha_test": TestFunction("alpha", dof_handler=dh),
        "mu_test": TestFunction("mu_alpha", dof_handler=dh),
        "v_k": VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh),
        "p_k": Function("p_k", "p", dof_handler=dh),
        "vS_k": VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh),
        "u_k": VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh),
        "alpha_k": Function("alpha_k", "alpha", dof_handler=dh),
        "mu_k": Function("mu_k", "mu_alpha", dof_handler=dh),
        "v_n": VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh),
        "p_n": Function("p_n", "p", dof_handler=dh),
        "vS_n": VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh),
        "u_n": VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh),
        "alpha_n": Function("alpha_n", "alpha", dof_handler=dh),
        "mu_n": Function("mu_n", "mu_alpha", dof_handler=dh),
    }
    return problem


def _set_snapshots(problem, mms: DeformationOnlyMMS) -> None:
    problem["v_n"].set_values_from_function(lambda x, y: mms.v_n(x, y))
    problem["p_n"].set_values_from_function(lambda x, y: float(mms.p_n(x, y)))
    problem["vS_n"].set_values_from_function(lambda x, y: mms.vS_n(x, y))
    problem["u_n"].set_values_from_function(lambda x, y: mms.u_n(x, y))
    problem["alpha_n"].set_values_from_function(lambda x, y: float(mms.alpha_n(x, y)))
    problem["mu_n"].set_values_from_function(lambda x, y: float(mms.mu_alpha_n(x, y)))

    # Use the exact end-of-step state as the Newton initial guess for robustness.
    problem["v_k"].set_values_from_function(lambda x, y: mms.v_k(x, y))
    problem["p_k"].set_values_from_function(lambda x, y: float(mms.p_k(x, y)))
    problem["vS_k"].set_values_from_function(lambda x, y: mms.vS_k(x, y))
    problem["u_k"].set_values_from_function(lambda x, y: mms.u_k(x, y))
    problem["alpha_k"].set_values_from_function(lambda x, y: float(mms.alpha_k(x, y)))
    problem["mu_k"].set_values_from_function(lambda x, y: float(mms.mu_alpha_k(x, y)))


def _build_forms(problem, mms: DeformationOnlyMMS, *, qdeg: int):
    params = mms.params
    return build_deformation_only_forms(
        v_k=problem["v_k"],
        p_k=problem["p_k"],
        vS_k=problem["vS_k"],
        u_k=problem["u_k"],
        alpha_k=problem["alpha_k"],
        mu_alpha_k=problem["mu_k"],
        v_n=problem["v_n"],
        p_n=problem["p_n"],
        vS_n=problem["vS_n"],
        u_n=problem["u_n"],
        alpha_n=problem["alpha_n"],
        mu_alpha_n=problem["mu_n"],
        dv=problem["dv"],
        dp=problem["dp"],
        dvS=problem["dvS"],
        du=problem["du"],
        dalpha=problem["dalpha"],
        dmu_alpha=problem["dmu"],
        v_test=problem["v_test"],
        q_test=problem["q_test"],
        vS_test=problem["vS_test"],
        u_test=problem["u_test"],
        alpha_test=problem["alpha_test"],
        mu_alpha_test=problem["mu_test"],
        dx=dx(metadata={"q": int(qdeg)}),
        dt=Constant(float(mms.dt)),
        theta=float(mms.theta),
        rho_f=Constant(float(params["rho_f"])),
        mu_f=Constant(float(params["mu_f"])),
        mu_b=Constant(float(params["mu_b"])),
        kappa_inv=Constant(float(params["kappa_inv"])),
        mu_s=Constant(float(params["mu_s"])),
        lambda_s=Constant(float(params["lambda_s"])),
        phi_b=float(params["phi_b"]),
        M_alpha=float(params["M_alpha"]),
        gamma_alpha=float(params["gamma_alpha"]),
        eps_alpha=float(params["eps_alpha"]),
        support_physics="internal_conversion",
        alpha_advect_with=ALPHA_TRANSPORT_VELOCITY,
        alpha_advection_form=ALPHA_TRANSPORT_FORM,
        f_v=Analytic(lambda x, y: mms.f_v(x, y), degree=10),
        f_u=Analytic(lambda x, y: mms.f_u(x, y), degree=10),
        f_alpha=Analytic(lambda x, y: mms.f_alpha(x, y), degree=10),
    )


def _build_bcs(mms: DeformationOnlyMMS):
    bcs = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("v_x", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.v(x, y, t)[..., 0])),
                BoundaryCondition("v_y", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.v(x, y, t)[..., 1])),
                BoundaryCondition("p", "dirichlet", tag, _as_float_time(mms.p)),
                BoundaryCondition("vS_x", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.vS(x, y, t)[..., 0])),
                BoundaryCondition("vS_y", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.vS(x, y, t)[..., 1])),
                BoundaryCondition("u_x", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.u(x, y, t)[..., 0])),
                BoundaryCondition("u_y", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.u(x, y, t)[..., 1])),
                BoundaryCondition("alpha", "dirichlet", tag, _as_float_time(mms.alpha)),
                BoundaryCondition("mu_alpha", "dirichlet", tag, _as_float_time(mms.mu_alpha)),
            ]
        )
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0)) for b in bcs]
    return bcs, bcs_homog


def _solve_one(
    *,
    case: str,
    nx: int,
    qdeg: int,
    qerr: int,
    dt_val: float,
    theta: float,
    backend: str,
    error_backend: str | None,
    newton_tol: float,
    max_it: int,
):
    mms = _build_case(case, dt_val=dt_val, theta=theta)
    problem = _create_problem(nx)
    _set_snapshots(problem, mms)
    forms = _build_forms(problem, mms, qdeg=qdeg)
    bcs, bcs_homog = _build_bcs(mms)

    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=problem["dh"],
        mixed_element=problem["me"],
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=float(newton_tol),
            max_newton_iter=int(max_it),
            print_level=1,
        ),
        lin_params=LinearSolverParameters(backend="scipy", tol=1.0e-12, maxit=10000),
        quad_order=int(qdeg),
        backend=str(backend),
    )

    t_start = time.perf_counter()
    solver._current_step_no = 1
    solver._current_t = float(mms.t_n)
    solver._current_dt = float(mms.dt)
    aux_functions = {"dt": Constant(float(mms.dt))}
    bcs_now = solver._freeze_bcs(solver.bcs, float(mms.t_k))
    problem["dh"].apply_bcs(bcs_now, problem["v_k"], problem["p_k"], problem["vS_k"], problem["u_k"], problem["alpha_k"], problem["mu_k"])
    _, converged, n_iters = solver._newton_loop(
        [problem["v_k"], problem["p_k"], problem["vS_k"], problem["u_k"], problem["alpha_k"], problem["mu_k"]],
        [problem["v_n"], problem["p_n"], problem["vS_n"], problem["u_n"], problem["alpha_n"], problem["mu_n"]],
        aux_functions,
        bcs_now,
    )
    elapsed = time.perf_counter() - t_start
    if not converged:
        raise RuntimeError(f"Newton did not converge for case={case}, nx={nx}.")

    dh = problem["dh"]
    t_err = float(mms.t_k)
    err = {
        "v_l2": float(
            dh.l2_error(
                problem["v_k"],
                exact={"v_x": lambda x, y: mms.v(x, y, t_err)[..., 0], "v_y": lambda x, y: mms.v(x, y, t_err)[..., 1]},
                fields=["v_x", "v_y"],
                quad_order=int(qerr),
                relative=False,
            )
        ),
        "p_l2": float(dh.l2_error(problem["p_k"], exact={"p": lambda x, y: mms.p(x, y, t_err)}, fields=["p"], quad_order=int(qerr), relative=False)),
        "vS_l2": float(
            dh.l2_error(
                problem["vS_k"],
                exact={"vS_x": lambda x, y: mms.vS(x, y, t_err)[..., 0], "vS_y": lambda x, y: mms.vS(x, y, t_err)[..., 1]},
                fields=["vS_x", "vS_y"],
                quad_order=int(qerr),
                relative=False,
            )
        ),
        "u_l2": float(
            dh.l2_error(
                problem["u_k"],
                exact={"u_x": lambda x, y: mms.u(x, y, t_err)[..., 0], "u_y": lambda x, y: mms.u(x, y, t_err)[..., 1]},
                fields=["u_x", "u_y"],
                quad_order=int(qerr),
                relative=False,
            )
        ),
        "alpha_l2": float(
            dh.l2_error(
                problem["alpha_k"],
                exact={"alpha": lambda x, y: mms.alpha(x, y, t_err)},
                fields=["alpha"],
                quad_order=int(qerr),
                relative=False,
            )
        ),
        "mu_alpha_l2": float(
            dh.l2_error(
                problem["mu_k"],
                exact={"mu_alpha": lambda x, y: mms.mu_alpha(x, y, t_err)},
                fields=["mu_alpha"],
                quad_order=int(qerr),
                relative=False,
            )
        ),
    }

    error_backend_key = str(error_backend or "").strip().lower()
    if error_backend_key in {"", "auto"}:
        error_backend_key = "cpp" if str(backend).strip().lower() == "python" else str(backend).strip().lower()
    if error_backend_key == "c++":
        error_backend_key = "cpp"

    if error_backend_key in {"jit", "cpp"}:
        err["v_h1"] = float(
            dh.h1_error_vector_on_side_compiled(
                problem["v_k"],
                lambda x, y: mms.grad_v(x, y, t_err),
                FULL_DOMAIN_LS,
                side="-",
                fields=["v_x", "v_y"],
                relative=False,
                quad_order=int(qerr),
                backend=error_backend_key,
            )
        )
        err["vS_h1"] = float(
            dh.h1_error_vector_on_side_compiled(
                problem["vS_k"],
                lambda x, y: mms.grad_vS(x, y, t_err),
                FULL_DOMAIN_LS,
                side="-",
                fields=["vS_x", "vS_y"],
                relative=False,
                quad_order=int(qerr),
                backend=error_backend_key,
            )
        )
        err["u_h1"] = float(
            dh.h1_error_vector_on_side_compiled(
                problem["u_k"],
                lambda x, y: mms.grad_u(x, y, t_err),
                FULL_DOMAIN_LS,
                side="-",
                fields=["u_x", "u_y"],
                relative=False,
                quad_order=int(qerr),
                backend=error_backend_key,
            )
        )
        err["alpha_h1"] = float(
            dh.h1_error_scalar_on_side_compiled(
                problem["alpha_k"],
                lambda x, y: mms.grad_alpha(x, y, t_err),
                FULL_DOMAIN_LS,
                side="-",
                field="alpha",
                relative=False,
                quad_order=int(qerr),
                backend=error_backend_key,
            )
        )
        err["mu_alpha_h1"] = float(
            dh.h1_error_scalar_on_side_compiled(
                problem["mu_k"],
                lambda x, y: mms.grad_mu_alpha(x, y, t_err),
                FULL_DOMAIN_LS,
                side="-",
                field="mu_alpha",
                relative=False,
                quad_order=int(qerr),
                backend=error_backend_key,
            )
        )
    else:
        err["v_h1"] = float(
            dh.h1_error_vector_on_side(
                problem["v_k"],
                lambda x, y: mms.grad_v(x, y, t_err),
                FULL_DOMAIN_LS,
                side="-",
                fields=["v_x", "v_y"],
                relative=False,
                quad_order=int(qerr),
            )
        )
        err["vS_h1"] = float(
            dh.h1_error_vector_on_side(
                problem["vS_k"],
                lambda x, y: mms.grad_vS(x, y, t_err),
                FULL_DOMAIN_LS,
                side="-",
                fields=["vS_x", "vS_y"],
                relative=False,
                quad_order=int(qerr),
            )
        )
        err["u_h1"] = float(
            dh.h1_error_vector_on_side(
                problem["u_k"],
                lambda x, y: mms.grad_u(x, y, t_err),
                FULL_DOMAIN_LS,
                side="-",
                fields=["u_x", "u_y"],
                relative=False,
                quad_order=int(qerr),
            )
        )
        err["alpha_h1"] = float(
            dh.h1_error_scalar_on_side(
                problem["alpha_k"],
                lambda x, y: mms.grad_alpha(x, y, t_err),
                FULL_DOMAIN_LS,
                side="-",
                field="alpha",
                relative=False,
            )
        )
        err["mu_alpha_h1"] = float(
            dh.h1_error_scalar_on_side(
                problem["mu_k"],
                lambda x, y: mms.grad_mu_alpha(x, y, t_err),
                FULL_DOMAIN_LS,
                side="-",
                field="mu_alpha",
                relative=False,
            )
        )

    row = {
        "case": str(case),
        "nx": int(nx),
        "h": 1.0 / float(nx),
        "dt": float(dt_val),
        "theta": float(theta),
        "newton_iters": int(n_iters),
        "solve_seconds": float(elapsed),
        "alpha_transport_velocity": ALPHA_TRANSPORT_VELOCITY,
        "alpha_transport_form": ALPHA_TRANSPORT_FORM,
    }
    row.update(err)
    row.update({f"param_{k}": _serialize_param(v) for k, v in mms.params.items()})
    return row


def _add_eocs(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    metrics = [
        "v_l2",
        "p_l2",
        "vS_l2",
        "u_l2",
        "alpha_l2",
        "mu_alpha_l2",
        "v_h1",
        "vS_h1",
        "u_h1",
        "alpha_h1",
        "mu_alpha_h1",
    ]
    out = []
    prev = None
    for row in rows:
        cur = dict(row)
        for metric in metrics:
            cur[f"eoc_{metric}"] = float("nan") if prev is None else _eoc(prev["h"], cur["h"], prev[metric], cur[metric])
        out.append(cur)
        prev = cur
    return out


def _write_outputs(case: str, rows: list[dict[str, float]], *, outdir: Path, save_plot: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    case_key = str(case).strip().lower()
    stem = f"deformation_only_mms_{case_key}"

    csv_path = outdir / f"{stem}.csv"
    tex_path = outdir / f"{stem}.tex"
    json_path = outdir / f"{stem}.json"
    md_path = outdir / f"{stem}.md"

    df.to_csv(csv_path, index=False)
    latex = df.to_latex(index=False, float_format=lambda x: f"{x:.3e}", na_rep="-")
    tex_path.write_text(latex, encoding="utf-8")
    summary = {
        "case": case_key,
        "alpha_transport_velocity": ALPHA_TRANSPORT_VELOCITY,
        "alpha_transport_form": ALPHA_TRANSPORT_FORM,
        "rows": rows,
        "best_row": rows[-1] if rows else None,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        f"# Reduced deformation-only MMS: {case_key}",
        "",
        f"- alpha transport velocity: `{ALPHA_TRANSPORT_VELOCITY}`",
        f"- alpha transport form: `{ALPHA_TRANSPORT_FORM}`",
        f"- rows: {len(rows)}",
        f"- finest nx: {rows[-1]['nx'] if rows else '-'}",
        f"- finest Newton iterations: {rows[-1]['newton_iters'] if rows else '-'}",
        "",
        "## Table",
        "",
        _markdown_table(df),
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    if not save_plot:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable; skipping convergence plot: {exc}")
        return

    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    h = df["h"].to_numpy(dtype=float)
    for key, label in (
        ("v_l2", r"$\|e_v\|_{L^2}$"),
        ("u_l2", r"$\|e_u\|_{L^2}$"),
        ("alpha_l2", r"$\|e_\alpha\|_{L^2}$"),
        ("mu_alpha_l2", r"$\|e_{\mu_\alpha}\|_{L^2}$"),
    ):
        ax.loglog(h, df[key].to_numpy(dtype=float), marker="o", linewidth=1.5, label=label)
    ax.set_xlabel(r"$h$")
    ax.set_ylabel("error")
    ax.set_title(f"Reduced deformation-only MMS convergence: {case_key}")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend()
    fig.savefig(outdir / f"{stem}_convergence.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reduced deformation-only MMS h-convergence study with support-preserving alpha transport."
    )
    ap.add_argument("--case", type=str, default="static", choices=("static", "translation", "shear"))
    ap.add_argument("--nx-list", type=str, default="4,8,12")
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--q-error", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--error-backend", type=str, default="auto", choices=("auto", "python", "jit", "cpp"))
    ap.add_argument("--newton-tol", type=float, default=1.0e-10)
    ap.add_argument("--max-it", type=int, default=20)
    ap.add_argument("--convergence", action="store_true", help="Save a log-log convergence plot.")
    ap.add_argument("--outdir", type=str, default="examples/biofilms/results/deformation_only_mms")
    args = ap.parse_args()

    nx_list = [int(s.strip()) for s in str(args.nx_list).split(",") if s.strip()]
    if len(nx_list) < 2:
        raise ValueError("Provide at least two mesh sizes in --nx-list.")

    case_outdir = Path(str(args.outdir)) / str(args.case)
    rows = [
        _solve_one(
            case=str(args.case),
            nx=int(nx),
            qdeg=int(args.q),
            qerr=int(args.q_error),
            dt_val=float(args.dt),
            theta=float(args.theta),
            backend=str(args.backend),
            error_backend=str(args.error_backend),
            newton_tol=float(args.newton_tol),
            max_it=int(args.max_it),
        )
        for nx in nx_list
    ]
    rows = _add_eocs(rows)

    pd.set_option("display.max_columns", None)
    df = pd.DataFrame(rows)
    print(df)
    print()
    print(df.to_latex(index=False, float_format=lambda x: f"{x:.3e}", na_rep="-"))

    _write_outputs(str(args.case), rows, outdir=case_outdir, save_plot=bool(args.convergence))


def _markdown_table(frame: pd.DataFrame) -> str:
    cols = [str(c) for c in frame.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, rec in frame.iterrows():
        vals = []
        for col in cols:
            val = rec[col]
            if isinstance(val, float):
                vals.append("-" if not (val == val) else f"{val:.6e}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
