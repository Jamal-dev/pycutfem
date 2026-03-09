#!/usr/bin/env python3
"""Paper 1 Benchmark 5: Jonas-inspired exact shear FSI benchmark.

This benchmark verifies the reduced deformation-only one-domain model with:

  - active conservative Cahn--Hilliard transport for alpha,
  - an explicit tangential interface-traction transfer term on y = y_interface,
  - a closed-form shear-flow / elastic-layer exact solution,
  - exact volume forcing derived from the reduced Paper-1 equations.

The benchmark geometry is a unit square with a flat interface. It is
Jonas-inspired in the sense that tangential fluid shear drives a stationary
elastic deformation, but the geometry is tailored to the Paper-1 one-domain
setup so that the exact alpha field remains a true stationary CH equilibrium.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional in lightweight envs
    matplotlib = None
    plt = None

for _k in (
    "NUMBA_DEBUG",
    "NUMBA_DUMP_BYTECODE",
    "NUMBA_DUMP_IR",
    "NUMBA_DUMP_SSA",
    "NUMBA_DEBUG_ARRAY_OPT",
):
    os.environ[_k] = "0"

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, NewtonSolver
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from examples.utils.biofilm.benchmark5_jonas_shear_exact import JonasShearBenchmark, build_jonas_shear_benchmark
from examples.utils.biofilm.deformation_only import build_deformation_only_forms


FULL_DOMAIN_LS = AffineLevelSet(1.0, 0.0, -2.0)


@dataclass(frozen=True)
class CaseResult:
    row: dict[str, float]
    profile_samples: dict[str, np.ndarray]
    vtk_path: str | None


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
    return lambda x, y, t: float(np.asarray(fn(np.asarray(x), np.asarray(y))).reshape(()))


def _eval_scalar_at_point(dh: DofHandler, mesh: Mesh, f_scalar: Function, point: tuple[float, float]) -> float:
    from pycutfem.fem import transform

    xy = np.asarray(point, dtype=float)
    for elem in mesh.elements_list:
        verts = mesh.nodes_x_y_pos[list(elem.nodes)]
        if not (
            verts[:, 0].min() - 1.0e-12 <= xy[0] <= verts[:, 0].max() + 1.0e-12
            and verts[:, 1].min() - 1.0e-12 <= xy[1] <= verts[:, 1].max() + 1.0e-12
        ):
            continue
        try:
            xi, eta = transform.inverse_mapping(mesh, elem.id, xy)
        except Exception:
            continue
        if not (-1.0001 <= xi <= 1.0001 and -1.0001 <= eta <= 1.0001):
            continue
        me = dh.mixed_element
        fld = f_scalar.field_name
        phi = me.basis(fld, float(xi), float(eta))[me.slice(fld)]
        gdofs = dh.element_maps[fld][elem.id]
        vals = f_scalar.get_nodal_values(gdofs)
        return float(phi @ vals)
    return float("nan")


def _eval_vector_at_point(dh: DofHandler, mesh: Mesh, f_vec: VectorFunction, point: tuple[float, float]) -> np.ndarray:
    return np.asarray([_eval_scalar_at_point(dh, mesh, comp, point) for comp in f_vec.components], dtype=float)


def _eoc(prev_h: float, h: float, prev_err: float, err: float) -> float:
    if not (prev_h > 0.0 and h > 0.0 and prev_err > 0.0 and err > 0.0):
        return float("nan")
    return float(math.log(prev_err / err) / math.log(prev_h / h))


def _create_problem(nx: int) -> dict[str, object]:
    if int(nx) % 2 != 0:
        raise ValueError("Benchmark 5 requires even nx so the flat interface aligns with the mesh.")
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

    return {
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


def _set_snapshots(problem: dict[str, object], bench: JonasShearBenchmark) -> None:
    problem["v_n"].set_values_from_function(lambda x, y: bench.v_n(x, y))
    problem["p_n"].set_values_from_function(lambda x, y: float(bench.p_n(x, y)))
    problem["vS_n"].set_values_from_function(lambda x, y: bench.vS_n(x, y))
    problem["u_n"].set_values_from_function(lambda x, y: bench.u_n(x, y))
    problem["alpha_n"].set_values_from_function(lambda x, y: float(bench.alpha_n(x, y)))
    problem["mu_n"].set_values_from_function(lambda x, y: float(bench.mu_alpha_n(x, y)))

    problem["v_k"].set_values_from_function(lambda x, y: bench.v_k(x, y))
    problem["p_k"].set_values_from_function(lambda x, y: float(bench.p_k(x, y)))
    problem["vS_k"].set_values_from_function(lambda x, y: bench.vS_k(x, y))
    problem["u_k"].set_values_from_function(lambda x, y: bench.u_k(x, y))
    problem["alpha_k"].set_values_from_function(lambda x, y: float(bench.alpha_k(x, y)))
    problem["mu_k"].set_values_from_function(lambda x, y: float(bench.mu_alpha_k(x, y)))


def _build_forms(problem: dict[str, object], bench: JonasShearBenchmark, *, qdeg: int):
    params = bench.params
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
        dt=Constant(float(bench.dt)),
        theta=float(bench.theta),
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
        g_t_k=Analytic(lambda x, y: bench.g_t(x, y), degree=8),
        g_t_n=Analytic(lambda x, y: bench.g_t(x, y), degree=8),
        traction_weight_k=Analytic(lambda x, y: bench.traction_weight(x, y), degree=8),
        traction_weight_n=Analytic(lambda x, y: bench.traction_weight(x, y), degree=8),
        f_v=Analytic(lambda x, y: bench.f_v(x, y), degree=10),
        f_u=Analytic(lambda x, y: bench.f_u(x, y), degree=10),
        f_alpha=Analytic(lambda x, y: bench.f_alpha(x, y), degree=10),
    )


def _build_bcs(bench: JonasShearBenchmark):
    bcs = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("v_x", "dirichlet", tag, _as_float_time(lambda x, y: bench.v(x, y)[..., 0])),
                BoundaryCondition("v_y", "dirichlet", tag, _as_float_time(lambda x, y: bench.v(x, y)[..., 1])),
                BoundaryCondition("p", "dirichlet", tag, _as_float_time(bench.p)),
                BoundaryCondition("vS_x", "dirichlet", tag, _as_float_time(lambda x, y: bench.vS(x, y)[..., 0])),
                BoundaryCondition("vS_y", "dirichlet", tag, _as_float_time(lambda x, y: bench.vS(x, y)[..., 1])),
                BoundaryCondition("u_x", "dirichlet", tag, _as_float_time(lambda x, y: bench.u(x, y)[..., 0])),
                BoundaryCondition("u_y", "dirichlet", tag, _as_float_time(lambda x, y: bench.u(x, y)[..., 1])),
                BoundaryCondition("alpha", "dirichlet", tag, _as_float_time(bench.alpha)),
                BoundaryCondition("mu_alpha", "dirichlet", tag, _as_float_time(bench.mu_alpha)),
            ]
        )
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0)) for b in bcs]
    return bcs, bcs_homog


def _sample_profiles(problem: dict[str, object], bench: JonasShearBenchmark, *, n_samples: int) -> dict[str, np.ndarray]:
    x0 = 0.5
    y = np.linspace(0.0, 1.0, int(n_samples), dtype=float)
    v_num = np.asarray([_eval_vector_at_point(problem["dh"], problem["mesh"], problem["v_k"], (x0, float(yy)))[0] for yy in y], dtype=float)
    u_num = np.asarray([_eval_vector_at_point(problem["dh"], problem["mesh"], problem["u_k"], (x0, float(yy)))[0] for yy in y], dtype=float)
    alpha_num = np.asarray([_eval_scalar_at_point(problem["dh"], problem["mesh"], problem["alpha_k"], (x0, float(yy))) for yy in y], dtype=float)
    return {
        "y": y,
        "v_num": v_num,
        "u_num": u_num,
        "alpha_num": alpha_num,
        "v_exact": np.asarray(bench.v(x0 * np.ones_like(y), y), dtype=float)[:, 0],
        "u_exact": np.asarray(bench.u(x0 * np.ones_like(y), y), dtype=float)[:, 0],
        "alpha_exact": np.asarray(bench.alpha(x0 * np.ones_like(y), y), dtype=float),
    }


def _solve_one(
    *,
    nx: int,
    qdeg: int,
    qerr: int,
    backend: str,
    error_backend: str | None,
    newton_tol: float,
    max_it: int,
    profile_samples: int,
    vtk_dir: Path | None,
) -> CaseResult:
    bench = build_jonas_shear_benchmark()
    problem = _create_problem(nx)
    _set_snapshots(problem, bench)
    forms = _build_forms(problem, bench, qdeg=qdeg)
    bcs, bcs_homog = _build_bcs(bench)

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
    solver._current_t = 0.0
    solver._current_dt = float(bench.dt)
    aux_functions = {"dt": Constant(float(bench.dt))}
    bcs_now = solver._freeze_bcs(solver.bcs, float(bench.dt))
    problem["dh"].apply_bcs(bcs_now, problem["v_k"], problem["p_k"], problem["vS_k"], problem["u_k"], problem["alpha_k"], problem["mu_k"])
    _, converged, n_iters = solver._newton_loop(
        [problem["v_k"], problem["p_k"], problem["vS_k"], problem["u_k"], problem["alpha_k"], problem["mu_k"]],
        [problem["v_n"], problem["p_n"], problem["vS_n"], problem["u_n"], problem["alpha_n"], problem["mu_n"]],
        aux_functions,
        bcs_now,
    )
    elapsed = time.perf_counter() - t_start
    if not converged:
        raise RuntimeError(f"Newton did not converge for Benchmark 5, nx={nx}.")

    dh = problem["dh"]
    err = {
        "v_l2": float(
            dh.l2_error(
                problem["v_k"],
                exact={"v_x": lambda x, y: bench.v(x, y)[..., 0], "v_y": lambda x, y: bench.v(x, y)[..., 1]},
                fields=["v_x", "v_y"],
                quad_order=int(qerr),
                relative=False,
            )
        ),
        "p_l2": float(dh.l2_error(problem["p_k"], exact={"p": lambda x, y: bench.p(x, y)}, fields=["p"], quad_order=int(qerr), relative=False)),
        "vS_l2": float(
            dh.l2_error(
                problem["vS_k"],
                exact={"vS_x": lambda x, y: bench.vS(x, y)[..., 0], "vS_y": lambda x, y: bench.vS(x, y)[..., 1]},
                fields=["vS_x", "vS_y"],
                quad_order=int(qerr),
                relative=False,
            )
        ),
        "u_l2": float(
            dh.l2_error(
                problem["u_k"],
                exact={"u_x": lambda x, y: bench.u(x, y)[..., 0], "u_y": lambda x, y: bench.u(x, y)[..., 1]},
                fields=["u_x", "u_y"],
                quad_order=int(qerr),
                relative=False,
            )
        ),
        "alpha_l2": float(
            dh.l2_error(
                problem["alpha_k"],
                exact={"alpha": lambda x, y: bench.alpha(x, y)},
                fields=["alpha"],
                quad_order=int(qerr),
                relative=False,
            )
        ),
        "mu_alpha_l2": float(
            dh.l2_error(
                problem["mu_k"],
                exact={"mu_alpha": lambda x, y: bench.mu_alpha(x, y)},
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
                lambda x, y: bench.grad_v(x, y),
                FULL_DOMAIN_LS,
                side="-",
                fields=["v_x", "v_y"],
                relative=False,
                quad_order=int(qerr),
                backend=error_backend_key,
            )
        )
        err["u_h1"] = float(
            dh.h1_error_vector_on_side_compiled(
                problem["u_k"],
                lambda x, y: bench.grad_u(x, y),
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
                lambda x, y: bench.grad_alpha(x, y),
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
                lambda x, y: bench.grad_mu_alpha(x, y),
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
                lambda x, y: bench.grad_v(x, y),
                FULL_DOMAIN_LS,
                side="-",
                fields=["v_x", "v_y"],
                relative=False,
                quad_order=int(qerr),
            )
        )
        err["u_h1"] = float(
            dh.h1_error_vector_on_side(
                problem["u_k"],
                lambda x, y: bench.grad_u(x, y),
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
                lambda x, y: bench.grad_alpha(x, y),
                FULL_DOMAIN_LS,
                side="-",
                field="alpha",
                relative=False,
            )
        )
        err["mu_alpha_h1"] = float(
            dh.h1_error_scalar_on_side(
                problem["mu_k"],
                lambda x, y: bench.grad_mu_alpha(x, y),
                FULL_DOMAIN_LS,
                side="-",
                field="mu_alpha",
                relative=False,
            )
        )

    interface_point = (0.5, float(bench.interface_y))
    u_interface_num = float(_eval_vector_at_point(dh, problem["mesh"], problem["u_k"], interface_point)[0])
    u_interface_exact = float(np.asarray(bench.u(*interface_point), dtype=float).reshape(2)[0])
    alpha_interface_num = float(_eval_scalar_at_point(dh, problem["mesh"], problem["alpha_k"], interface_point))
    alpha_interface_exact = float(np.asarray(bench.alpha(*interface_point), dtype=float).reshape(()))

    row = {
        "nx": int(nx),
        "h": 1.0 / float(nx),
        "dt": float(bench.dt),
        "theta_time": float(bench.theta),
        "newton_iters": int(n_iters),
        "solve_seconds": float(elapsed),
        "u_interface_error": abs(u_interface_num - u_interface_exact),
        "alpha_interface_error": abs(alpha_interface_num - alpha_interface_exact),
        "tau_interface_exact": float(np.asarray(bench.g_t(*interface_point), dtype=float).reshape(2)[0]),
    }
    row.update(err)
    row.update({f"param_{k}": float(v) for k, v in bench.params.items()})

    vtk_path = None
    if vtk_dir is not None:
        vtk_dir.mkdir(parents=True, exist_ok=True)
        vtk_path = str(vtk_dir / f"benchmark5_jonas_shear_nx{int(nx):03d}.vtu")
        export_vtk(
            vtk_path,
            mesh=problem["mesh"],
            dof_handler=dh,
            functions={
                "v": problem["v_k"],
                "u": problem["u_k"],
                "alpha": problem["alpha_k"],
                "mu_alpha": problem["mu_k"],
            },
        )

    return CaseResult(
        row=row,
        profile_samples=_sample_profiles(problem, bench, n_samples=profile_samples),
        vtk_path=vtk_path,
    )


def _add_eocs(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    metrics = [
        "v_l2",
        "p_l2",
        "vS_l2",
        "u_l2",
        "alpha_l2",
        "mu_alpha_l2",
        "v_h1",
        "u_h1",
        "alpha_h1",
        "mu_alpha_h1",
        "u_interface_error",
        "alpha_interface_error",
    ]
    out: list[dict[str, float]] = []
    prev = None
    for row in rows:
        cur = dict(row)
        for metric in metrics:
            cur[f"eoc_{metric}"] = float("nan") if prev is None else _eoc(prev["h"], cur["h"], prev[metric], cur[metric])
        out.append(cur)
        prev = cur
    return out


def _write_profiles_csv(path: Path, samples: dict[str, np.ndarray]) -> None:
    keys = ["y", "v_num", "v_exact", "u_num", "u_exact", "alpha_num", "alpha_exact"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(keys)
        for i in range(len(samples["y"])):
            writer.writerow([f"{float(samples[key][i]):.16e}" for key in keys])


def _write_outputs(results: list[CaseResult], *, outdir: Path, save_plot: bool, png_dpi: int) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = _add_eocs([result.row for result in results])
    df = pd.DataFrame(rows)

    csv_path = outdir / "benchmark5_jonas_shear_summary.csv"
    tex_path = outdir / "benchmark5_jonas_shear_table.tex"
    json_path = outdir / "benchmark5_jonas_shear_summary.json"
    md_path = outdir / "benchmark5_jonas_shear_summary.md"
    profiles_csv = outdir / f"benchmark5_jonas_shear_profiles_nx{int(rows[-1]['nx']):03d}.csv"

    df.to_csv(csv_path, index=False)
    tex_path.write_text(df.to_latex(index=False, float_format=lambda x: f"{x:.3e}", na_rep="-"), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "rows": rows,
                "best_row": rows[-1] if rows else None,
                "profiles_csv": str(profiles_csv),
                "vtk_path": results[-1].vtk_path,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    md_path.write_text(_markdown_table(df), encoding="utf-8")
    _write_profiles_csv(profiles_csv, results[-1].profile_samples)

    outputs = {
        "summary_csv": str(csv_path),
        "summary_json": str(json_path),
        "table_tex": str(tex_path),
        "profiles_csv": str(profiles_csv),
    }

    if not save_plot or plt is None:
        return outputs

    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    h = df["h"].to_numpy(dtype=float)
    for key, label in (
        ("v_l2", r"$\|e_v\|_{L^2}$"),
        ("u_l2", r"$\|e_u\|_{L^2}$"),
        ("alpha_l2", r"$\|e_\alpha\|_{L^2}$"),
        ("u_interface_error", r"$|e_{u,\Gamma}|$"),
    ):
        ax.loglog(h, df[key].to_numpy(dtype=float), marker="o", linewidth=1.7, label=label)
    ax.set_xlabel(r"$h$")
    ax.set_ylabel("error")
    ax.set_title("Benchmark 5 convergence")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend()
    conv_png = outdir / "benchmark5_jonas_shear_convergence.png"
    fig.savefig(conv_png, dpi=int(png_dpi))
    plt.close(fig)

    prof = results[-1].profile_samples
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    axes[0].plot(prof["v_exact"], prof["y"], color="#111827", lw=2.0, label="exact")
    axes[0].plot(prof["v_num"], prof["y"], color="#1D4ED8", lw=1.7, linestyle="--", label="numerical")
    axes[0].set_xlabel(r"$v_x$")
    axes[0].set_ylabel(r"$y$")
    axes[0].set_title("Tangential velocity")
    axes[0].grid(True, linestyle=":", linewidth=0.5)
    axes[0].legend()

    axes[1].plot(prof["u_exact"], prof["y"], color="#111827", lw=2.0, label="exact")
    axes[1].plot(prof["u_num"], prof["y"], color="#B45309", lw=1.7, linestyle="--", label="numerical")
    axes[1].set_xlabel(r"$u_x$")
    axes[1].set_ylabel(r"$y$")
    axes[1].set_title("Tangential displacement")
    axes[1].grid(True, linestyle=":", linewidth=0.5)

    axes[2].plot(prof["alpha_exact"], prof["y"], color="#111827", lw=2.0, label="exact")
    axes[2].plot(prof["alpha_num"], prof["y"], color="#047857", lw=1.7, linestyle="--", label="numerical")
    axes[2].set_xlabel(r"$\alpha$")
    axes[2].set_ylabel(r"$y$")
    axes[2].set_title("Conserved CH profile")
    axes[2].grid(True, linestyle=":", linewidth=0.5)

    profiles_png = outdir / "benchmark5_jonas_shear_profiles.png"
    fig.savefig(profiles_png, dpi=int(png_dpi))
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    extent = [0.0, 1.0, 0.0, 1.0]
    y = prof["y"]
    X = np.linspace(0.0, 1.0, 48, dtype=float)
    Y = np.asarray(y, dtype=float)
    VV = np.tile(prof["v_num"].reshape(1, -1), (X.size, 1))
    UU = np.tile(prof["u_num"].reshape(1, -1), (X.size, 1))
    AA = np.tile(prof["alpha_num"].reshape(1, -1), (X.size, 1))
    for ax, field, title, cmap in (
        (axes[0], VV, r"$v_x$", "coolwarm"),
        (axes[1], UU, r"$u_x$", "cividis"),
        (axes[2], AA, r"$\alpha$", "viridis"),
    ):
        im = ax.imshow(field.T, origin="lower", extent=extent, aspect="auto", cmap=cmap)
        ax.axhline(0.5, color="white", lw=1.4, linestyle="--")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.82)
    fields_png = outdir / "benchmark5_jonas_shear_fields.png"
    fig.savefig(fields_png, dpi=int(png_dpi))
    plt.close(fig)

    outputs.update(
        {
            "convergence_png": str(conv_png),
            "profiles_png": str(profiles_png),
            "fields_png": str(fields_png),
        }
    )
    return outputs


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper 1 Benchmark 5: Jonas-inspired exact shear FSI benchmark.")
    ap.add_argument("--nx-list", type=str, default="8,16,32")
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--q-error", type=int, default=10)
    ap.add_argument("--backend", type=str, default="python", choices=("python", "jit", "cpp"))
    ap.add_argument("--error-backend", type=str, default="auto", choices=("auto", "python", "jit", "cpp"))
    ap.add_argument("--newton-tol", type=float, default=1.0e-10)
    ap.add_argument("--max-it", type=int, default=20)
    ap.add_argument("--profile-samples", type=int, default=401)
    ap.add_argument("--png-dpi", type=int, default=220)
    ap.add_argument("--vtk", action="store_true")
    ap.add_argument("--convergence", action="store_true", help="Save convergence and profile plots.")
    ap.add_argument("--outdir", type=str, default="examples/biofilms/benchmarks/FSI/results/paper1_benchmark5_jonas_shear")
    args = ap.parse_args()

    nx_list = [int(s.strip()) for s in str(args.nx_list).split(",") if s.strip()]
    if len(nx_list) < 2:
        raise ValueError("Provide at least two mesh sizes in --nx-list.")

    outdir = Path(str(args.outdir))
    vtk_dir = outdir / "vtk" if bool(args.vtk) else None
    results = [
        _solve_one(
            nx=int(nx),
            qdeg=int(args.q),
            qerr=int(args.q_error),
            backend=str(args.backend),
            error_backend=str(args.error_backend),
            newton_tol=float(args.newton_tol),
            max_it=int(args.max_it),
            profile_samples=int(args.profile_samples),
            vtk_dir=vtk_dir if int(nx) == int(nx_list[-1]) else None,
        )
        for nx in nx_list
    ]
    outputs = _write_outputs(results, outdir=outdir, save_plot=bool(args.convergence), png_dpi=int(args.png_dpi))

    print(pd.DataFrame(_add_eocs([result.row for result in results])))
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
