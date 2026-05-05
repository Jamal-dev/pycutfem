#!/usr/bin/env python3
"""Physical C++/PDAS drivers for three-constituent Benchmarks 2--5.

These drivers are intentionally small production representatives.  They build
the canonical nine-field residual and use the bounded PDAS solver, but each
benchmark activates only the fields that are physically meaningful in that
limit problem.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from examples.biofilms.benchmarks.three_constituent.seboldt_physical import (
    _field_stats,
    _make_homogeneous_bcs,
    _make_spaces,
    _make_state,
    _make_trial_test,
    _set_scalar_values_from_function,
    _tag_rectangle_boundaries,
)
from examples.biofilms.benchmarks.three_constituent.stoter_physical import (
    run_physical_stoter_three_constituent,
)
from examples.utils.biofilm.three_constituent_one_domain import (
    _named_c,
    build_three_constituent_one_domain_forms,
    build_three_constituent_pdas_solver,
)
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, TimeStepperParameters, VIParameters
from pycutfem.ufl.expressions import Function
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


@dataclass(frozen=True)
class PhysicalBenchmarkResult:
    case_id: str
    passed: bool
    outdir: Path
    summary: dict[str, object]


def _zero_scalar(x, y, t=0.0) -> float:
    del x, y, t
    return 0.0


def _one_scalar(x, y, t=0.0) -> float:
    del x, y, t
    return 1.0


def _constant_scalar(value: float):
    def _fn(x, y, t=0.0) -> float:
        del x, y, t
        return float(value)

    return _fn


def _constant_vector(xval: float, yval: float):
    def _fn(x, y):
        shape = np.broadcast(np.asarray(x, dtype=float), np.asarray(y, dtype=float)).shape
        return np.stack(
            [
                np.full(shape, float(xval), dtype=float),
                np.full(shape, float(yval), dtype=float),
            ],
            axis=-1,
        )

    return _fn


def _build_problem(
    *,
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    poly_order: int,
    pressure_order: int,
    scalar_order: int,
):
    nodes, elems, _, corners = structured_quad(float(Lx), float(Ly), nx=int(nx), ny=int(ny), poly_order=int(poly_order))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=int(poly_order),
    )
    _tag_rectangle_boundaries(mesh, Lx=float(Lx), Ly=float(Ly))
    me = MixedElement(
        mesh,
        field_specs={
            "vf_x": int(poly_order),
            "vf_y": int(poly_order),
            "pf": int(pressure_order),
            "vp_x": int(poly_order),
            "vp_y": int(poly_order),
            "pp": int(pressure_order),
            "vs_x": int(poly_order),
            "vs_y": int(poly_order),
            "us_x": int(poly_order),
            "us_y": int(poly_order),
            "alpha": int(scalar_order),
            "phi": int(scalar_order),
            "Gamma": int(scalar_order),
        },
    )
    dh = DofHandler(me, method="cg")
    spaces = _make_spaces(dh)
    trial, test = _make_trial_test(dh, spaces)
    state = _make_state(dh)
    return mesh, me, dh, trial, test, state


def _set_inactive_fields(dh: DofHandler, fields: tuple[str, ...]) -> None:
    inactive = set(int(d) for d in getattr(dh, "dof_tags", {}).get("inactive", set()) or set())
    for field in fields:
        inactive.update(int(d) for d in np.asarray(dh.get_field_slice(field), dtype=int).ravel().tolist())
    dh.dof_tags["inactive"] = inactive


def _solver(
    forms,
    *,
    dh: DofHandler,
    me: MixedElement,
    bcs: list[BoundaryCondition],
    backend: str,
    linear_backend: str,
    quad_order: int,
    newton_tol: float,
    max_newton_iter: int,
    pore_pressure_bounds: tuple[float | None, float | None] = (None, None),
):
    return build_three_constituent_pdas_solver(
        forms,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=_make_homogeneous_bcs(bcs),
        newton_params=NewtonParameters(
            newton_tol=float(newton_tol),
            newton_rtol=0.0,
            max_newton_iter=int(max_newton_iter),
            print_level=1,
            line_search=True,
            ls_max_iter=16,
        ),
        vi_params=VIParameters(
            c=1.0,
            c_by_field={"alpha": 1.0, "phi": 1.0, "pp": 1.0},
            project_initial_guess=True,
            project_each_iteration=False,
            active_set_persistence=1,
            inactive_reg_lambda0=1.0e-10,
            inactive_reg_lambda_max=1.0e4,
        ),
        lin_params=LinearSolverParameters(backend=str(linear_backend), tol=1.0e-10, maxit=10000),
        backend=str(backend),
        quad_order=int(quad_order),
        alpha_bounds=(0.0, 1.0),
        phi_bounds=(0.0, 1.0),
        pore_pressure_bounds=pore_pressure_bounds,
    )


def _functions_and_prev(dh: DofHandler, state: dict[str, object]):
    p_f_n = Function("p_f_n", "pf", dof_handler=dh)
    p_p_n = Function("p_p_n", "pp", dof_handler=dh)
    p_f_n.nodal_values[:] = state["p_f_k"].nodal_values[:]
    p_p_n.nodal_values[:] = state["p_p_k"].nodal_values[:]
    return (
        [
            state["v_f_k"],
            state["p_f_k"],
            state["v_p_k"],
            state["p_p_k"],
            state["v_s_k"],
            state["u_s_k"],
            state["alpha_k"],
            state["phi_k"],
            state["Gamma_k"],
        ],
        [
            state["v_f_n"],
            p_f_n,
            state["v_p_n"],
            p_p_n,
            state["v_s_n"],
            state["u_s_n"],
            state["alpha_n"],
            state["phi_n"],
            state["Gamma_n"],
        ],
    )


def _run_one_step(solver, *, functions: list[object], prev_functions: list[object], dt: float):
    return solver.solve_time_interval(
        functions=functions,
        prev_functions=prev_functions,
        time_params=TimeStepperParameters(
            dt=float(dt),
            final_time=float(dt),
            max_steps=1,
            stop_on_steady=False,
            allow_dt_reduction=False,
            predictor="prev",
        ),
    )


def _write_summary(outdir: Path, summary: dict[str, object]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    flat = {
        key: value
        for key, value in summary.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    metrics = summary.get("metrics")
    if isinstance(metrics, dict):
        flat.update({str(key): value for key, value in metrics.items() if isinstance(value, (str, int, float, bool))})
    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)


def run_physical_benchmark2_darcy_column(
    *,
    outdir: Path,
    nx: int = 4,
    ny: int = 2,
    dt: float = 0.05,
    phi0: float = 0.62,
    rho_p: float = 1.0,
    R_ps: float = 4.5,
    body_force_x: float = 1.0e-3,
    backend: str = "cpp",
    linear_backend: str = "scipy",
    quad_order: int = 6,
) -> PhysicalBenchmarkResult:
    outdir = Path(outdir).resolve()
    mesh, me, dh, trial, test, state = _build_problem(
        Lx=1.0,
        Ly=0.25,
        nx=int(nx),
        ny=int(ny),
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
    )
    del mesh
    v_exact = float(rho_p) * float(body_force_x) / (float(phi0) * float(R_ps))
    zero_v = _constant_vector(0.0, 0.0)
    vp = _constant_vector(v_exact, 0.0)
    for key in ("v_f_k", "v_f_n", "v_s_k", "v_s_n", "u_s_k", "u_s_n"):
        state[key].set_values_from_function(zero_v)
    for key in ("v_p_k", "v_p_n"):
        state[key].set_values_from_function(vp)
    for key in ("p_f_k", "p_p_k", "Gamma_k", "Gamma_n"):
        state[key].nodal_values[:] = 0.0
    for key in ("alpha_k", "alpha_n"):
        _set_scalar_values_from_function(dh, state[key], lambda x, y: 1.0)
    for key in ("phi_k", "phi_n"):
        _set_scalar_values_from_function(dh, state[key], lambda x, y: float(phi0))

    _set_inactive_fields(dh, ("vf_x", "vf_y", "pf", "pp", "vs_x", "vs_y", "us_x", "us_y"))
    forms = build_three_constituent_one_domain_forms(
        **state,
        **trial,
        **test,
        dx=dx(metadata={"q": int(quad_order)}),
        dt=_named_c("tc_b2_dt", float(dt)),
        rho_f=1.0,
        rho_p=rho_p,
        rho_s=1.0,
        mu_f=0.0,
        mu_p=0.0,
        mu_s=0.0,
        lambda_s=0.0,
        R_fp=0.0,
        R_fs=0.0,
        R_ps=R_ps,
        R_pair_cholesky=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, math.sqrt(float(R_ps)))),
        pair_weight_epsilon=1.0e-12,
        ell_Gamma=0.0,
        gamma_mobility="off",
        b_p=(float(body_force_x), 0.0),
    )
    bcs = [
        BoundaryCondition("vp_y", "dirichlet", tag, _zero_scalar)
        for tag in ("left", "right", "bottom", "top")
    ]
    bcs.extend(
        [
            BoundaryCondition("alpha", "dirichlet", "left", _one_scalar),
            BoundaryCondition("phi", "dirichlet", "left", _constant_scalar(phi0)),
        ]
    )
    solver = _solver(
        forms,
        dh=dh,
        me=me,
        bcs=bcs,
        backend=backend,
        linear_backend=linear_backend,
        quad_order=quad_order,
        newton_tol=1.0e-9,
        max_newton_iter=10,
    )
    functions, prev_functions = _functions_and_prev(dh, state)
    t0 = time.perf_counter()
    passed = False
    error = ""
    try:
        _run_one_step(solver, functions=functions, prev_functions=prev_functions, dt=float(dt))
        passed = True
    except Exception as exc:  # noqa: PERF203
        error = f"{type(exc).__name__}: {exc}"
    vp_vals = np.asarray(state["v_p_k"].components[0].nodal_values, dtype=float)
    flux = float(phi0) * float(np.mean(vp_vals))
    metrics = {
        **_field_stats(state),
        "v_p_x_mean": float(np.mean(vp_vals)),
        "darcy_velocity_exact": float(v_exact),
        "darcy_velocity_error": float(np.max(np.abs(vp_vals - v_exact))),
        "darcy_flux": flux,
        "darcy_flux_exact": float(rho_p) * float(body_force_x) / float(R_ps),
        "darcy_flux_error": flux - float(rho_p) * float(body_force_x) / float(R_ps),
    }
    passed = bool(
        passed
        and abs(metrics["darcy_flux_error"]) <= 1.0e-7
        and abs(metrics["darcy_velocity_error"]) <= 1.0e-5
        and metrics["alpha_lower_violation"] == 0.0
        and metrics["phi_lower_violation"] == 0.0
    )
    summary = {
        "case_id": "benchmark2_physical_fixed_porous_darcy_column",
        "passed": passed,
        "error": error,
        "elapsed_s": float(time.perf_counter() - t0),
        "mesh": {"nx": int(nx), "ny": int(ny), "total_dofs": int(dh.total_dofs), "active_dofs": int(np.asarray(solver.active_dofs, dtype=int).size)},
        "parameters": {"dt": float(dt), "phi0": float(phi0), "rho_p": float(rho_p), "R_ps": float(R_ps), "body_force_x": float(body_force_x)},
        "metrics": metrics,
    }
    _write_summary(outdir, summary)
    return PhysicalBenchmarkResult(str(summary["case_id"]), passed, outdir, summary)


def run_physical_benchmark3_drag_relaxation(
    *,
    outdir: Path,
    dt: float = 0.2,
    phi0: float = 0.48,
    rho_p: float = 1.0,
    rho_s: float = 1.35,
    R_ps: float = 3.2,
    v_p0: float = 1.0e-4,
    v_s0: float = -2.0e-5,
    backend: str = "cpp",
    linear_backend: str = "scipy",
    quad_order: int = 4,
) -> PhysicalBenchmarkResult:
    outdir = Path(outdir).resolve()
    _, me, dh, trial, test, state = _build_problem(
        Lx=1.0,
        Ly=1.0,
        nx=1,
        ny=1,
        poly_order=1,
        pressure_order=1,
        scalar_order=1,
    )
    for key in ("v_f_k", "v_f_n", "u_s_k", "u_s_n"):
        state[key].set_values_from_function(_constant_vector(0.0, 0.0))
    state["v_p_n"].set_values_from_function(_constant_vector(v_p0, 0.0))
    state["v_s_n"].set_values_from_function(_constant_vector(v_s0, 0.0))
    P = float(phi0)
    B = 1.0 - P
    r_p = P * float(rho_p)
    r_s = B * float(rho_s)
    chi = P * P * float(R_ps)
    rate_ie = 1.0 + float(dt) * chi * (1.0 / r_p + 1.0 / r_s)
    w = (float(v_p0) - float(v_s0)) / rate_ie
    momentum = r_p * float(v_p0) + r_s * float(v_s0)
    v_bar = momentum / (r_p + r_s)
    v_p_exact = v_bar + (r_s / (r_p + r_s)) * w
    v_s_exact = v_bar - (r_p / (r_p + r_s)) * w
    state["v_p_k"].set_values_from_function(_constant_vector(v_p_exact, 0.0))
    state["v_s_k"].set_values_from_function(_constant_vector(v_s_exact, 0.0))
    state["u_s_k"].set_values_from_function(_constant_vector(float(dt) * v_s_exact, 0.0))
    for key in ("p_f_k", "p_p_k", "Gamma_k", "Gamma_n"):
        state[key].nodal_values[:] = 0.0
    for key in ("alpha_k", "alpha_n"):
        _set_scalar_values_from_function(dh, state[key], lambda x, y: 1.0)
    for key in ("phi_k", "phi_n"):
        _set_scalar_values_from_function(dh, state[key], lambda x, y: float(phi0))
    _set_inactive_fields(
        dh,
        (
            "vf_x",
            "vf_y",
            "pf",
            "pp",
            "vp_y",
            "vs_y",
            "us_y",
            "alpha",
            "phi",
            "Gamma",
        ),
    )
    forms = build_three_constituent_one_domain_forms(
        **state,
        **trial,
        **test,
        dx=dx(metadata={"q": int(quad_order)}),
        dt=_named_c("tc_b3_dt", float(dt)),
        rho_f=1.0,
        rho_p=rho_p,
        rho_s=rho_s,
        mu_f=0.0,
        mu_p=0.0,
        mu_s=0.0,
        lambda_s=0.0,
        R_fp=0.0,
        R_fs=0.0,
        R_ps=R_ps,
        R_pair_cholesky=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, math.sqrt(float(R_ps)))),
        pair_weight_epsilon=1.0e-12,
        ell_Gamma=0.0,
        gamma_mobility="off",
    )
    solver = _solver(
        forms,
        dh=dh,
        me=me,
        bcs=[],
        backend=backend,
        linear_backend=linear_backend,
        quad_order=quad_order,
        newton_tol=1.0e-10,
        max_newton_iter=10,
    )
    functions, prev_functions = _functions_and_prev(dh, state)
    t0 = time.perf_counter()
    passed = False
    error = ""
    try:
        _run_one_step(solver, functions=functions, prev_functions=prev_functions, dt=float(dt))
        passed = True
    except Exception as exc:  # noqa: PERF203
        error = f"{type(exc).__name__}: {exc}"
    vp_vals = np.asarray(state["v_p_k"].components[0].nodal_values, dtype=float)
    vs_vals = np.asarray(state["v_s_k"].components[0].nodal_values, dtype=float)
    mom_vals = r_p * vp_vals + r_s * vs_vals
    metrics = {
        **_field_stats(state),
        "v_p_x_exact": float(v_p_exact),
        "v_s_x_exact": float(v_s_exact),
        "v_p_x_error": float(np.max(np.abs(vp_vals - v_p_exact))),
        "v_s_x_error": float(np.max(np.abs(vs_vals - v_s_exact))),
        "momentum_error": float(np.max(np.abs(mom_vals - momentum))),
        "relative_velocity_decay_factor": float((np.mean(vp_vals) - np.mean(vs_vals)) / (float(v_p0) - float(v_s0))),
    }
    passed = bool(passed and metrics["v_p_x_error"] <= 1.0e-8 and metrics["v_s_x_error"] <= 1.0e-8 and metrics["momentum_error"] <= 1.0e-8)
    summary = {
        "case_id": "benchmark3_physical_pore_solid_drag_relaxation",
        "passed": passed,
        "error": error,
        "elapsed_s": float(time.perf_counter() - t0),
        "mesh": {"nx": 1, "ny": 1, "total_dofs": int(dh.total_dofs), "active_dofs": int(np.asarray(solver.active_dofs, dtype=int).size)},
        "parameters": {"dt": float(dt), "phi0": float(phi0), "rho_p": float(rho_p), "rho_s": float(rho_s), "R_ps": float(R_ps), "v_p0": float(v_p0), "v_s0": float(v_s0)},
        "metrics": metrics,
    }
    _write_summary(outdir, summary)
    return PhysicalBenchmarkResult(str(summary["case_id"]), passed, outdir, summary)


def _moving_tanh_alpha(x, t: float, *, center0: float, speed: float, eps: float):
    return 0.5 * (1.0 + np.tanh((np.asarray(x, dtype=float) - (float(center0) + float(speed) * float(t))) / float(eps)))


def run_physical_benchmark4_moving_tanh_body(
    *,
    outdir: Path,
    nx: int = 24,
    ny: int = 4,
    dt: float = 1.0e-3,
    center0: float = 0.45,
    speed: float = 0.1,
    eps_alpha: float = 0.06,
    phi0: float = 0.55,
    backend: str = "cpp",
    linear_backend: str = "scipy",
    quad_order: int = 6,
) -> PhysicalBenchmarkResult:
    outdir = Path(outdir).resolve()
    _, me, dh, trial, test, state = _build_problem(
        Lx=1.0,
        Ly=0.2,
        nx=int(nx),
        ny=int(ny),
        poly_order=2,
        pressure_order=1,
        scalar_order=2,
    )
    zero_v = _constant_vector(0.0, 0.0)
    vs = _constant_vector(speed, 0.0)
    for key in ("v_f_k", "v_f_n", "v_p_k", "v_p_n", "u_s_k", "u_s_n"):
        state[key].set_values_from_function(zero_v)
    for key in ("v_s_k", "v_s_n"):
        state[key].set_values_from_function(vs)
    for key in ("p_f_k", "p_p_k", "Gamma_k", "Gamma_n"):
        state[key].nodal_values[:] = 0.0
    alpha_n = lambda x, y: float(_moving_tanh_alpha(x, 0.0, center0=center0, speed=speed, eps=eps_alpha))
    alpha_guess = lambda x, y: float(_moving_tanh_alpha(x, dt, center0=center0, speed=speed, eps=eps_alpha))
    for key, fn in (("alpha_n", alpha_n), ("alpha_k", alpha_guess)):
        _set_scalar_values_from_function(dh, state[key], fn)
    for key in ("phi_k", "phi_n"):
        _set_scalar_values_from_function(dh, state[key], lambda x, y: float(phi0))
    _set_inactive_fields(
        dh,
        (
            "vf_x",
            "vf_y",
            "pf",
            "vp_x",
            "vp_y",
            "pp",
            "vs_x",
            "vs_y",
            "us_x",
            "us_y",
            "phi",
            "Gamma",
        ),
    )
    forms = build_three_constituent_one_domain_forms(
        **state,
        **trial,
        **test,
        dx=dx(metadata={"q": int(quad_order)}),
        dt=_named_c("tc_b4_dt", float(dt)),
        rho_f=1.0,
        rho_p=1.0,
        rho_s=1.0,
        mu_f=0.0,
        mu_p=0.0,
        mu_s=0.0,
        lambda_s=0.0,
        R_fp=0.0,
        R_fs=0.0,
        R_ps=0.0,
        ell_Gamma=0.0,
        gamma_mobility="off",
    )
    alpha_bc = lambda x, y, t: float(_moving_tanh_alpha(x, dt, center0=center0, speed=speed, eps=eps_alpha))
    bcs = [BoundaryCondition("alpha", "dirichlet", tag, alpha_bc) for tag in ("left", "right", "bottom", "top")]
    solver = _solver(
        forms,
        dh=dh,
        me=me,
        bcs=bcs,
        backend=backend,
        linear_backend=linear_backend,
        quad_order=quad_order,
        newton_tol=1.0e-8,
        max_newton_iter=12,
    )
    functions, prev_functions = _functions_and_prev(dh, state)
    t0 = time.perf_counter()
    passed = False
    error = ""
    try:
        _run_one_step(solver, functions=functions, prev_functions=prev_functions, dt=float(dt))
        passed = True
    except Exception as exc:  # noqa: PERF203
        error = f"{type(exc).__name__}: {exc}"
    coords = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    alpha_vals = np.asarray(state["alpha_k"].nodal_values, dtype=float)
    alpha_exact = _moving_tanh_alpha(coords[:, 0], dt, center0=center0, speed=speed, eps=eps_alpha)
    center_target = float(center0) + float(speed) * float(dt)
    center_idx = int(np.argmin(np.abs(alpha_vals - 0.5))) if alpha_vals.size else 0
    center_num = float(coords[center_idx, 0]) if coords.size else float("nan")
    metrics = {
        **_field_stats(state),
        "alpha_linf_error_to_translated_profile": float(np.max(np.abs(alpha_vals - alpha_exact))),
        "interface_center_numeric": center_num,
        "interface_center_exact": center_target,
        "interface_center_error": center_num - center_target,
    }
    passed = bool(passed and metrics["alpha_lower_violation"] == 0.0 and metrics["alpha_upper_violation"] <= 1.0e-10 and abs(metrics["interface_center_error"]) <= 2.5 / max(int(nx), 1))
    summary = {
        "case_id": "benchmark4_physical_moving_tanh_porous_body",
        "passed": passed,
        "error": error,
        "elapsed_s": float(time.perf_counter() - t0),
        "mesh": {"nx": int(nx), "ny": int(ny), "total_dofs": int(dh.total_dofs), "active_dofs": int(np.asarray(solver.active_dofs, dtype=int).size)},
        "parameters": {"dt": float(dt), "center0": float(center0), "speed": float(speed), "eps_alpha": float(eps_alpha), "phi0": float(phi0)},
        "metrics": metrics,
    }
    _write_summary(outdir, summary)
    return PhysicalBenchmarkResult(str(summary["case_id"]), passed, outdir, summary)


def run_physical_benchmark5_fixed_porous_bed(
    *,
    outdir: Path,
    nx: int = 16,
    ny: int = 20,
    eps: float = 10.0,
    dt: float = 1.0e-3,
    final_time: float = 2.0e-3,
    backend: str = "cpp",
    linear_backend: str = "scipy",
    quad_order: int = 6,
) -> PhysicalBenchmarkResult:
    result = run_physical_stoter_three_constituent(
        outdir=Path(outdir),
        nx=int(nx),
        ny=int(ny),
        eps=float(eps),
        phi0=1.0,
        u_max=0.05,
        dt=float(dt),
        final_time=float(final_time),
        poly_order=1,
        pressure_order=1,
        scalar_order=1,
        kappa=5000.0,
        hydraulic_conductivity=5000.0,
        friction_alpha=1.0,
        bjs_factor=1.0,
        formulation="stoter_mixed_limit",
        gamma_mobility="off",
        resistance_model="full_cholesky",
        backend=str(backend),
        linear_backend=str(linear_backend),
        quad_order=int(quad_order),
        max_newton_iter=12,
        newton_tol=1.0e-7,
        pore_pressure_lower_bound=None,
    )
    summary = dict(result.summary)
    summary["case_id"] = "benchmark5_physical_free_flow_over_fixed_porous_bed"
    summary["benchmark5_adapter"] = "physical Stoter fixed-rigid bed with BJS law"
    summary["passed"] = bool(result.passed)
    _write_summary(Path(outdir), summary)
    return PhysicalBenchmarkResult(str(summary["case_id"]), bool(result.passed), Path(outdir), summary)


def run_physical_benchmarks_2_to_5(
    *,
    outdir: Path,
    cases: tuple[int, ...] = (2, 3, 4, 5),
    backend: str = "cpp",
    linear_backend: str = "scipy",
) -> list[PhysicalBenchmarkResult]:
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    results: list[PhysicalBenchmarkResult] = []
    for case in cases:
        if int(case) == 2:
            results.append(run_physical_benchmark2_darcy_column(outdir=outdir / "benchmark2_darcy_column", backend=backend, linear_backend=linear_backend))
        elif int(case) == 3:
            results.append(run_physical_benchmark3_drag_relaxation(outdir=outdir / "benchmark3_drag_relaxation", backend=backend, linear_backend=linear_backend))
        elif int(case) == 4:
            results.append(run_physical_benchmark4_moving_tanh_body(outdir=outdir / "benchmark4_moving_tanh_body", backend=backend, linear_backend=linear_backend))
        elif int(case) == 5:
            results.append(run_physical_benchmark5_fixed_porous_bed(outdir=outdir / "benchmark5_fixed_porous_bed", backend=backend, linear_backend=linear_backend))
        else:
            raise ValueError(f"Unsupported physical benchmark case {case!r}; expected 2, 3, 4, or 5.")
    summary = [
        {
            "case_id": result.case_id,
            "passed": bool(result.passed),
            "summary_path": str(result.outdir / "summary.json"),
        }
        for result in results
    ]
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("out/three_constituent_physical_benchmarks_2_to_5"))
    parser.add_argument("--cases", type=str, default="2,3,4,5")
    parser.add_argument("--backend", type=str, default="cpp")
    parser.add_argument("--linear-backend", type=str, default="scipy")
    args = parser.parse_args(argv)
    cases = tuple(int(v.strip()) for v in str(args.cases).split(",") if v.strip())
    results = run_physical_benchmarks_2_to_5(
        outdir=args.outdir,
        cases=cases,
        backend=str(args.backend),
        linear_backend=str(args.linear_backend),
    )
    print(json.dumps([result.summary for result in results], indent=2, sort_keys=True))
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PhysicalBenchmarkResult",
    "run_physical_benchmark2_darcy_column",
    "run_physical_benchmark3_drag_relaxation",
    "run_physical_benchmark4_moving_tanh_body",
    "run_physical_benchmark5_fixed_porous_bed",
    "run_physical_benchmarks_2_to_5",
]
