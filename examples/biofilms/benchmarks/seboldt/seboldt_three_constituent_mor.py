#!/usr/bin/env python3
"""Seboldt three-constituent native MOR benchmark.

This driver is intentionally local to the Seboldt benchmark directory.  It uses
the physical three-constituent one-domain model, resolves the diffuse interface
with a requested number of cells, collects FOM snapshots, and compares a
held-out full-order step against the native C++ bound-constrained QDEIM MOR
online solve.  The target can be a true reduced Galerkin projection, full-row
LSPG, sampled LSPG, or sampled GNAT/QDEIM so branch correctness can be checked
before accepting sparse hyper-reduction.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from examples.biofilms.benchmarks.three_constituent.seboldt_physical import (
    _alpha_profile,
    _bottom_inlet_y,
    _build_mesh,
    _cosine_ramp_value,
    _field_stats,
    _initialize_state,
    _make_homogeneous_bcs,
    _make_spaces,
    _make_state,
    _make_trial_test,
    _zero_scalar,
)
from examples.biofilms.benchmarks.three_constituent.three_constituent_mor import (
    CURRENT_STATE_KEYS,
    FIELD_SPECS,
    LOGICAL_FIELD_GROUPS,
    _as_single_volume_integral,
    _bound_stats,
    _compile_native_pair,
    _cross_validated_mode_count,
    _current_field_objects,
    _fit_trial_basis,
    _json_finite,
    _previous_field_objects,
    _relative_errors_by_field,
    _state_vector,
    _write_artifact,
)
from examples.utils.biofilm.three_constituent_one_domain import (
    _named_c,
    build_three_constituent_one_domain_forms,
    build_three_constituent_pdas_solver,
)
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.mor import (
    DWRReducedTrajectory,
    NativeAdjointDWRSpec,
    apply_affine_updates_to_static_args,
    assemble_qoi_gradient,
    bound_constraints_from_fields,
    build_block_balanced_gnat_sampling,
    build_block_row_weights,
    build_dirichlet_lift_state_updates,
    certify_dual_weighted_residual,
    certify_dual_weighted_residual_from_artifact_trajectory,
    evaluate_qoi_functional,
    field_row_blocks,
    fit_fieldwise_pod_basis,
    fit_time_parameterized_predictor,
    native_kernel_metadata_from_runner,
    native_online_convergence_status,
    project_reduced_coefficients_to_bounds,
    ReferencePolicy,
    reduced_qoi_gradient_from_full,
    reduced_target_from_native_kernel_pair,
    select_coordinate_band_elements,
    solve_with_branch_backtracking,
)
from pycutfem.mor.online_gauss_newton import (
    solve_native_bound_constrained_deim_online_gauss_newton,
    solve_native_bound_constrained_galerkin_online_gauss_newton,
    solve_native_bound_constrained_online_gauss_newton,
    solve_native_deim_online_gauss_newton,
    solve_native_online_gauss_newton,
)
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, TimeStepperParameters, VIParameters
from pycutfem.ufl.autodiff import linearize_form
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Function, TestFunction
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dx


@dataclass(frozen=True)
class SeboldtMORResult:
    passed: bool
    summary: dict[str, Any]
    outdir: Path


@dataclass(frozen=True)
class SeboldtMORConfig:
    outdir: Path
    nx: int = 4
    ny: int = 6
    poly_order: int = 1
    pressure_order: int = 1
    scalar_order: int = 1
    interface_cells: float = 7.0
    min_interface_cells: float = 6.0
    eps_alpha: float = 0.05
    dt: float = 1.0e-4
    train_steps: int = 10
    heldout_steps: int = 1
    final_time: float | None = None
    trajectory_cache: Path | None = None
    reuse_trajectory_cache: bool = False
    validate_training_trajectory: bool = False
    v_in: float = 0.05
    t_ramp: float = 0.0
    phi_b: float = 0.18
    qdeg: int = 3
    max_modes: int = 10
    basis_strategy: str = "fieldwise"
    field_max_modes: int | None = None
    sample_rows: int = 64
    min_rows_per_block: int = 4
    sample_interface_band_only: bool = False
    target: str = "sampled_gnat"
    row_weight_max: float = 1.0
    cv_tolerance: float = 2.0e-3
    newton_tol: float = 1.0e-7
    max_newton_iter: int = 10
    native_residual_tol: float = 1.0e-2
    native_optimality_tol: float | None = None
    native_step_tol: float = 1.0e-8
    native_feasibility_tol: float = 1.0e-8
    native_damping: float = 0.0
    native_max_step_norm: float | None = None
    native_max_step_norm_factor: float | None = None
    native_branch_radius: float | None = None
    native_branch_radius_factor: float | None = None
    native_branch_backtracking: bool = False
    native_branch_radius_schedule: str | None = None
    native_max_step_norm_schedule: str | None = None
    native_reference_weight: float = 0.0
    native_state_merit_weight: float = 0.0
    native_require_residual_convergence: bool = False
    native_min_substeps: int = 1
    native_max_substeps: int = 1
    native_substep_factor: int = 2
    online_predictor: str = "previous"
    reference_predictor_degree: int = 24
    reference_predictor_ridge: float = 1.0e-12
    audit_targets: bool = False
    dwr_previous_state_tangents: bool = False
    bound_violation_tolerance: float = 1.0e-8
    error_tolerance: float = 1.5e-1
    projection_error_tolerance: float = 5.0e-2
    max_native_iter: int = 12
    native_line_search: bool = False
    native_adaptive_damping: bool = False
    constraint_method: str = "ipm"
    native_accept_failed_steps: bool = False
    backend: str = "cpp"
    linear_backend: str = "scipy"


_TARGET_ALIASES = {
    "galerkin": "true_galerkin",
    "true_galerkin": "true_galerkin",
    "full_galerkin": "true_galerkin",
    "full_row_lspg": "full_row_lspg",
    "full_lspg": "full_row_lspg",
    "sampled": "sampled_lspg",
    "sampled_lspg": "sampled_lspg",
    "gnat": "sampled_gnat",
    "sampled_gnat": "sampled_gnat",
    "qdeim": "sampled_gnat",
}


def _parse_optional_float_schedule(raw: str | None) -> tuple[float | None, ...]:
    if raw is None or not str(raw).strip():
        return ()
    values: list[float | None] = []
    for token in str(raw).split(","):
        item = token.strip().lower()
        if not item:
            continue
        if item in {"default", "none"}:
            values.append(None)
            continue
        if item in {"inf", "+inf", "infinity", "+infinity"}:
            values.append(float("inf"))
            continue
        value = float(item)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("native branch/trust schedules must contain finite nonnegative values, inf, or none.")
        values.append(value)
    return tuple(values)


def _schedule_with_default(schedule: tuple[float | None, ...], default: float | None) -> tuple[float | None, ...]:
    if schedule:
        return tuple(None if value is not None and np.isinf(value) else value for value in schedule)
    return (default,)


def _native_online_status_ok(
    result: Any,
    residual_tol: float,
    optimality_tol: float | None = None,
    step_tol: float | None = None,
) -> bool:
    status = native_online_convergence_status(
        result,
        residual_tol=float(residual_tol),
        optimality_tol=optimality_tol,
        step_tol=step_tol,
        accept_stationary=False,
    )
    return bool(status.ok)


def _predict_online_coefficients(
    current: np.ndarray,
    previous: np.ndarray | None,
    *,
    predictor: str,
) -> np.ndarray:
    strategy = str(predictor).strip().lower()
    q = np.asarray(current, dtype=np.float64).reshape(-1)
    if strategy in {"previous", "constant", "prev", "none"} or previous is None:
        return np.ascontiguousarray(q.copy(), dtype=np.float64)
    q_prev = np.asarray(previous, dtype=np.float64).reshape(-1)
    if q_prev.shape != q.shape:
        raise ValueError("online coefficient predictor history has inconsistent dimensions.")
    if strategy in {"linear", "secant", "extrapolate"}:
        return np.ascontiguousarray(q + (q - q_prev), dtype=np.float64)
    if strategy in {"trajectory", "offline_trajectory", "training_projection", "snapshot_projection"}:
        return np.ascontiguousarray(q.copy(), dtype=np.float64)
    raise ValueError("online predictor must be 'previous', 'linear', or 'trajectory'.")


def _clip_decoded_predictor_distance(
    candidate: np.ndarray,
    reference: np.ndarray,
    trial_basis: np.ndarray,
    max_distance: float | None,
) -> tuple[np.ndarray, bool, float, float]:
    """Clip a reduced predictor by decoded-state distance from a reference."""

    q_ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    q_candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if q_candidate.shape != q_ref.shape:
        raise ValueError("predictor candidate and reference have inconsistent dimensions.")
    if not np.all(np.isfinite(q_candidate)):
        return np.ascontiguousarray(q_ref.copy(), dtype=np.float64), True, float("inf"), 0.0
    decoded_delta = np.asarray(trial_basis, dtype=np.float64) @ (q_candidate - q_ref)
    distance_before = float(np.linalg.norm(decoded_delta))
    if max_distance is None:
        return np.ascontiguousarray(q_candidate.copy(), dtype=np.float64), False, distance_before, distance_before
    limit = float(max_distance)
    if not np.isfinite(limit) or limit <= 0.0 or distance_before <= limit:
        return np.ascontiguousarray(q_candidate.copy(), dtype=np.float64), False, distance_before, distance_before
    clipped = q_ref + (limit / max(distance_before, 1.0e-300)) * (q_candidate - q_ref)
    return np.ascontiguousarray(clipped, dtype=np.float64), True, distance_before, limit


def _homogenize_basis_on_fixed_rows(
    trial_basis: np.ndarray,
    fixed_rows: np.ndarray,
    *,
    relative_tol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Zero fixed Dirichlet rows so reduced coefficients own only free DOFs."""

    basis = np.asarray(trial_basis, dtype=np.float64)
    fixed = np.asarray(fixed_rows, dtype=np.int64).reshape(-1)
    if basis.ndim != 2:
        raise ValueError("trial_basis must be rank-2.")
    homogeneous = np.ascontiguousarray(basis.copy(), dtype=np.float64)
    if fixed.size:
        if np.any(fixed < 0) or int(np.max(fixed)) >= homogeneous.shape[0]:
            raise ValueError("fixed_rows contains entries outside the trial basis.")
        homogeneous[fixed, :] = 0.0
    column_norms = np.linalg.norm(homogeneous, axis=0)
    scale = float(np.max(column_norms)) if column_norms.size else 0.0
    threshold = max(float(relative_tol) * max(scale, 1.0), 1.0e-14)
    keep = column_norms > threshold
    if keep.size and not np.any(keep):
        raise RuntimeError("all reduced basis columns vanish after Dirichlet homogenization.")
    out = np.ascontiguousarray(homogeneous[:, keep], dtype=np.float64)
    metadata = {
        "dirichlet_homogenized": bool(fixed.size),
        "fixed_rows": int(fixed.size),
        "raw_modes": int(basis.shape[1]),
        "online_modes": int(out.shape[1]),
        "dropped_fixed_only_modes": int(np.count_nonzero(~keep)),
        "drop_threshold": float(threshold),
    }
    return out, np.nonzero(keep)[0].astype(np.int64), metadata


def _project_online_coefficients(
    trial_basis: np.ndarray,
    offset: np.ndarray,
    state: np.ndarray,
    free_rows: np.ndarray,
) -> np.ndarray:
    """Least-squares projection using only rows owned by the reduced state."""

    rows = np.asarray(free_rows, dtype=np.int64).reshape(-1)
    basis = np.asarray(trial_basis, dtype=np.float64)
    base = np.asarray(offset, dtype=np.float64).reshape(-1)
    vec = np.asarray(state, dtype=np.float64).reshape(-1)
    if basis.ndim != 2 or base.size != basis.shape[0] or vec.size != basis.shape[0]:
        raise ValueError("incompatible basis, offset, and state shapes for online projection.")
    coeffs, *_ = np.linalg.lstsq(basis[rows, :], vec[rows] - base[rows], rcond=None)
    return np.ascontiguousarray(np.asarray(coeffs, dtype=np.float64).reshape(-1), dtype=np.float64)


def _decode_online_state(
    trial_basis: np.ndarray,
    offset: np.ndarray,
    coefficients: np.ndarray,
    *,
    fixed_rows: np.ndarray,
    lift_values: np.ndarray,
) -> np.ndarray:
    """Decode a reduced state and overwrite fixed rows by the known lift."""

    vec = np.ascontiguousarray(
        np.asarray(offset, dtype=np.float64).reshape(-1)
        + np.asarray(trial_basis, dtype=np.float64) @ np.asarray(coefficients, dtype=np.float64).reshape(-1),
        dtype=np.float64,
    )
    fixed = np.asarray(fixed_rows, dtype=np.int64).reshape(-1)
    if fixed.size:
        lift = np.asarray(lift_values, dtype=np.float64).reshape(-1)
        if lift.size != vec.size:
            raise ValueError("lift_values must have the same size as the decoded state.")
        vec[fixed] = lift[fixed]
    return vec


def _continuation_attempts_to_json(attempts: tuple[Any, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        options = dict(getattr(attempt, "options", {}) or {})
        rows.append(
            {
                "attempt": int(getattr(attempt, "attempt", len(rows) + 1)),
                "accepted": bool(getattr(attempt, "accepted", False)),
                "residual_norm": float(getattr(attempt, "residual_norm", np.inf)),
                "branch_distance": float(getattr(attempt, "branch_distance", np.nan)),
                "max_reference_distance": None
                if options.get("max_reference_distance") is None
                else float(options["max_reference_distance"]),
                "max_step_norm": None if options.get("max_step_norm") is None else float(options["max_step_norm"]),
                "message": str(getattr(attempt, "message", "")),
            }
        )
    return rows


def _set_native_dt_args(*args_maps: dict[str, Any], dt: float) -> tuple[str, ...]:
    changed: list[str] = []
    for args in args_maps:
        for name, value in args.items():
            key = str(name).lower()
            if key != "tc_dt" and not key.endswith("_dt"):
                continue
            if isinstance(value, np.ndarray):
                value[...] = float(dt)
            elif hasattr(value, "value"):
                value.value = float(dt)
            else:
                args[name] = float(dt)
            changed.append(str(name))
    return tuple(dict.fromkeys(changed))


def _substep_counts(max_substeps: int, factor: int, *, min_substeps: int = 1) -> tuple[int, ...]:
    max_count = max(1, int(max_substeps))
    min_count = min(max_count, max(1, int(min_substeps)))
    growth = max(2, int(factor))
    counts = [1]
    while counts[-1] < max_count:
        counts.append(min(max_count, counts[-1] * growth))
        if counts[-1] == counts[-2]:
            break
    filtered = [count for count in counts if count >= min_count]
    if not filtered or filtered[0] != min_count:
        filtered.insert(0, min_count)
    return tuple(dict.fromkeys(filtered))


def _reduced_target_metrics(residual: np.ndarray, jacobian: np.ndarray) -> dict[str, Any]:
    residual_vec = np.asarray(residual, dtype=float).reshape(-1)
    jac = np.asarray(jacobian, dtype=float)
    if jac.ndim != 2:
        raise ValueError("reduced target Jacobian must be rank-2.")
    residual_norm = float(np.linalg.norm(residual_vec))
    optimality = jac.T @ residual_vec if jac.size else np.zeros(jac.shape[1], dtype=float)
    singular = np.linalg.svd(jac, compute_uv=False) if jac.size else np.zeros(0, dtype=float)
    positive = singular[singular > 1.0e-13] if singular.size else singular
    condition = float(singular[0] / positive[-1]) if positive.size else float("inf")
    return {
        "residual_norm": residual_norm,
        "objective": float(0.5 * residual_norm * residual_norm),
        "optimality_norm": float(np.linalg.norm(optimality)),
        "jacobian_rank": int(positive.size),
        "jacobian_condition_estimate": condition,
        "target_rows": int(residual_vec.size),
        "reduced_cols": int(jac.shape[1]),
    }


def _current_functions(state: dict[str, object]) -> list[object]:
    return [state[key] for key in CURRENT_STATE_KEYS]


def _previous_functions(state: dict[str, object], p_f_n: Function, p_p_n: Function) -> list[object]:
    return [
        state["v_f_n"],
        p_f_n,
        state["v_p_n"],
        p_p_n,
        state["v_s_n"],
        state["u_s_n"],
        state["alpha_n"],
        state["phi_n"],
        state["Gamma_n"],
    ]


def _assign_state_vector(dh: DofHandler, field_objects: dict[str, object], values: np.ndarray) -> None:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != int(dh.total_dofs):
        raise ValueError("state vector size does not match dof handler.")
    for field in FIELD_SPECS:
        rows = np.asarray(dh.get_field_slice(field), dtype=np.int64)
        field_objects[field].set_nodal_values(rows, vector[rows])


def _is_previous_kernel_arg(name: str) -> bool:
    if not (name.startswith("u_") and name.endswith("_loc")):
        return False
    stem = name[2 : -len("_loc")]
    return stem.endswith(("_n", "_prev", "_previous", "_old"))


def _refresh_previous_state_static_args(args: dict[str, Any], previous_state: np.ndarray) -> tuple[str, ...]:
    """Refresh generated-kernel local arrays for the previous time level."""

    gdofs = np.asarray(args.get("gdofs_map"), dtype=np.int64)
    if gdofs.ndim != 2:
        raise ValueError("native static args must contain a rank-2 gdofs_map.")
    vector = np.asarray(previous_state, dtype=float).reshape(-1)
    if vector.size == 0 or int(np.max(gdofs)) >= vector.size:
        raise ValueError("previous_state is incompatible with gdofs_map.")
    safe = np.where(gdofs >= 0, gdofs, 0)
    gathered = np.ascontiguousarray(vector[safe], dtype=np.float64)
    gathered[gdofs < 0] = 0.0
    updated: list[str] = []
    for name, value in args.items():
        if not _is_previous_kernel_arg(str(name)):
            continue
        if not isinstance(value, np.ndarray) or value.shape != gdofs.shape:
            continue
        np.copyto(value, gathered, casting="unsafe")
        updated.append(str(name))
    return tuple(updated)


def _compile_native_previous_tangent(
    ctx: dict[str, Any],
    *,
    qdeg: int,
    element_ids: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any], tuple[str, ...], float]:
    """Compile ``dR_n/dx_{n-1}`` for reduced multi-step DWR certificates."""

    if element_ids is None:
        element_ids = np.arange(int(ctx["mesh"].n_elements), dtype=np.int32)
    else:
        element_ids = np.ascontiguousarray(np.asarray(element_ids, dtype=np.int32).reshape(-1))
    state = ctx["state"]
    previous_coefficients = [
        state["v_f_n"],
        ctx["p_f_n"],
        state["v_p_n"],
        ctx["p_p_n"],
        state["v_s_n"],
        state["u_s_n"],
        state["alpha_n"],
        state["phi_n"],
        state["Gamma_n"],
    ]
    trial = ctx["trial"]
    previous_directions = [
        trial["dv_f"],
        trial["dp_f"],
        trial["dv_p"],
        trial["dp_p"],
        trial["dv_s"],
        trial["du_s"],
        trial["dalpha"],
        trial["dphi"],
        trial["dGamma"],
    ]
    previous_tangent_form = linearize_form(ctx["forms"].residual_form, previous_coefficients, previous_directions)
    compiler = FormCompiler(ctx["dof_handler"], quadrature_order=int(qdeg), backend="cpp")
    t0 = time.perf_counter()
    runner, funcs, args, _ = compiler._prepare_volume_jit_kernel(
        _as_single_volume_integral(previous_tangent_form),
        element_ids=element_ids,
        full_local_layout=True,
    )
    runner(funcs, args)
    setup_s = time.perf_counter() - t0
    return runner, args, tuple(runner.param_order), setup_s


def _dirichlet_free_rows(dh: DofHandler, bcs: list[BoundaryCondition]) -> tuple[np.ndarray, np.ndarray]:
    fixed = np.asarray(sorted(int(row) for row in dh.get_dirichlet_data(bcs).keys()), dtype=np.int64)
    all_rows = np.arange(int(dh.total_dofs), dtype=np.int64)
    return np.setdiff1d(all_rows, fixed, assume_unique=False), fixed


def _field_mean_history(dh: DofHandler, snapshots: np.ndarray, field: str) -> np.ndarray:
    rows = np.asarray(dh.get_field_slice(field), dtype=np.int64).reshape(-1)
    if rows.size == 0:
        return np.zeros(int(snapshots.shape[1]), dtype=float)
    return np.mean(np.asarray(snapshots, dtype=float)[rows, :], axis=0)


def _save_trajectory_cache(path: Path, trajectory: dict[str, Any], *, n_steps: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        snapshots=np.asarray(trajectory["snapshots"], dtype=np.float64),
        step_times_s=np.asarray(trajectory["step_times_s"], dtype=np.float64),
        elapsed_s=np.asarray(float(trajectory["elapsed_s"]), dtype=np.float64),
        wall_s=np.asarray(float(trajectory["wall_s"]), dtype=np.float64),
        accepted_steps=np.asarray(int(trajectory["accepted_steps"]), dtype=np.int64),
        n_steps=np.asarray(int(n_steps), dtype=np.int64),
        step_metrics_json=np.asarray(json.dumps(_json_finite(trajectory["step_metrics"]), sort_keys=True)),
    )


def _load_trajectory_cache(path: Path, *, expected_steps: int) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        n_steps = int(np.asarray(data["n_steps"]).reshape(()))
        snapshots = np.asarray(data["snapshots"], dtype=np.float64)
        step_times_s = np.asarray(data["step_times_s"], dtype=np.float64).reshape(-1)
        if n_steps != int(expected_steps):
            raise ValueError(
                f"trajectory cache {path} has {n_steps} steps, expected {int(expected_steps)}."
            )
        if snapshots.ndim != 2 or snapshots.shape[1] != n_steps + 1:
            raise ValueError(f"trajectory cache {path} has inconsistent snapshot dimensions.")
        if step_times_s.size != n_steps:
            raise ValueError(f"trajectory cache {path} has inconsistent step timing dimensions.")
        return {
            "snapshots": snapshots,
            "step_times_s": step_times_s.tolist(),
            "elapsed_s": float(np.asarray(data["elapsed_s"]).reshape(())),
            "wall_s": float(np.asarray(data["wall_s"]).reshape(())),
            "accepted_steps": int(np.asarray(data["accepted_steps"]).reshape(())),
            "step_metrics": json.loads(str(np.asarray(data["step_metrics_json"]).reshape(()))),
        }


def _logical_rows(dh: DofHandler, fields: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([np.asarray(dh.get_field_slice(field), dtype=np.int64).reshape(-1) for field in fields])


def _logical_field_blocks(dh: DofHandler) -> tuple[tuple[np.ndarray, str], ...]:
    return tuple((_logical_rows(dh, fields), name) for name, fields in LOGICAL_FIELD_GROUPS.items())


def _fieldwise_cross_validated_mode_map(
    dh: DofHandler,
    snapshots: np.ndarray,
    *,
    max_modes: int | None,
    tolerance: float,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    mode_map: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for field_name, fields in LOGICAL_FIELD_GROUPS.items():
        block_rows = _logical_rows(dh, fields)
        modes, cv_rows = _cross_validated_mode_count(
            np.asarray(snapshots, dtype=float)[block_rows, :],
            max_modes=max_modes,
            tolerance=float(tolerance),
        )
        mode_map[field_name] = int(modes)
        rows.append({"field": field_name, "chosen_modes": int(modes), "cross_validation": cv_rows})
    return mode_map, rows


def _trajectory_branch_certificate(
    dh: DofHandler,
    *,
    snapshots: np.ndarray,
    train_snapshots: np.ndarray,
    trial_basis: np.ndarray,
    offset: np.ndarray,
    fixed_rows: np.ndarray | None = None,
    free_rows: np.ndarray | None = None,
    step_metrics: list[dict[str, float]],
    bound_tolerance: float,
) -> dict[str, Any]:
    bounds = [_bound_stats(dh, np.asarray(snapshots[:, i], dtype=float)) for i in range(int(snapshots.shape[1]))]
    max_bound_violation = float(max((row["max_violation"] for row in bounds), default=0.0))
    pressure_means = {
        field: _field_mean_history(dh, snapshots, field)
        for field in ("pf", "pp")
        if field in getattr(dh, "field_names", ()) or hasattr(dh, "get_field_slice")
    }
    pressure_max_abs_mean = {
        field: float(np.max(np.abs(values))) if values.size else 0.0
        for field, values in pressure_means.items()
    }
    train = np.asarray(train_snapshots, dtype=float)
    fixed = np.asarray((), dtype=np.int64) if fixed_rows is None else np.asarray(fixed_rows, dtype=np.int64).reshape(-1)
    if free_rows is None:
        free = np.setdiff1d(np.arange(int(train.shape[0]), dtype=np.int64), fixed, assume_unique=False)
    else:
        free = np.asarray(free_rows, dtype=np.int64).reshape(-1)
    coeffs = np.column_stack(
        [_project_online_coefficients(trial_basis, offset, train[:, i], free) for i in range(int(train.shape[1]))]
    )
    projected = np.asarray(offset, dtype=float).reshape(-1, 1) + np.asarray(trial_basis, dtype=float) @ coeffs
    if fixed.size:
        projected[fixed, :] = train[fixed, :]
    denom = np.maximum(np.linalg.norm(np.asarray(train_snapshots, dtype=float), axis=0), 1.0e-14)
    projection_errors = np.linalg.norm(projected - np.asarray(train_snapshots, dtype=float), axis=0) / denom
    nonlinear_ok = bool(all(np.isfinite(float(row.get("residual_norm", 0.0))) for row in step_metrics))
    return {
        "passed": bool(max_bound_violation <= float(bound_tolerance) and nonlinear_ok),
        "max_bound_violation": max_bound_violation,
        "pressure_mean_history": pressure_means,
        "pressure_max_abs_mean": pressure_max_abs_mean,
        "train_projection_relative_max": float(np.max(projection_errors)) if projection_errors.size else 0.0,
        "train_projection_relative_mean": float(np.mean(projection_errors)) if projection_errors.size else 0.0,
        "accepted_step_count": int(len(step_metrics)),
        "nonlinear_metrics_finite": nonlinear_ok,
    }


def _make_bcs(*, Lx: float, y_interface: float, eps_alpha: float, phi_b: float, v_in: float, t_ramp: float) -> list[BoundaryCondition]:
    alpha_bc = lambda x, y, t=0.0: _alpha_profile(x, y, y_interface=y_interface, eps_alpha=eps_alpha)
    phi_bc = lambda x, y, t=0.0: float(phi_b)
    inlet_y = lambda x, y, t=0.0: _bottom_inlet_y(
        x,
        Lx=Lx,
        v_in=v_in,
        ramp=_cosine_ramp_value(float(t), float(t_ramp)),
    )
    bcs: list[BoundaryCondition] = []
    for tag in ("left", "right"):
        bcs.extend(
            [
                BoundaryCondition("vf_x", "dirichlet", tag, _zero_scalar),
                BoundaryCondition("vf_y", "dirichlet", tag, _zero_scalar),
                BoundaryCondition("vp_x", "dirichlet", tag, _zero_scalar),
                BoundaryCondition("vs_x", "dirichlet", tag, _zero_scalar),
                BoundaryCondition("vs_y", "dirichlet", tag, _zero_scalar),
                BoundaryCondition("us_x", "dirichlet", tag, _zero_scalar),
                BoundaryCondition("us_y", "dirichlet", tag, _zero_scalar),
                BoundaryCondition("alpha", "dirichlet", tag, alpha_bc),
                BoundaryCondition("phi", "dirichlet", tag, phi_bc),
            ]
        )
    bcs.extend(
        [
            BoundaryCondition("vf_x", "dirichlet", "bottom", _zero_scalar),
            BoundaryCondition("vf_y", "dirichlet", "bottom", inlet_y),
            BoundaryCondition("vp_x", "dirichlet", "bottom", _zero_scalar),
            BoundaryCondition("vp_y", "dirichlet", "bottom", _zero_scalar),
            BoundaryCondition("vs_x", "dirichlet", "bottom", _zero_scalar),
            BoundaryCondition("vs_y", "dirichlet", "bottom", _zero_scalar),
            BoundaryCondition("us_x", "dirichlet", "bottom", _zero_scalar),
            BoundaryCondition("us_y", "dirichlet", "bottom", _zero_scalar),
            BoundaryCondition("alpha", "dirichlet", "bottom", alpha_bc),
            BoundaryCondition("phi", "dirichlet", "bottom", phi_bc),
            BoundaryCondition("pf", "dirichlet", "top", _zero_scalar),
            BoundaryCondition("pp", "dirichlet", "top", _zero_scalar),
            BoundaryCondition("alpha", "dirichlet", "top", alpha_bc),
            BoundaryCondition("phi", "dirichlet", "top", phi_bc),
        ]
    )
    return bcs


def _build_context(cfg: SeboldtMORConfig) -> dict[str, Any]:
    Lx = 1.0
    Ly = 1.5
    y_interface = 1.0
    mesh, mesh_meta = _build_mesh(
        Lx=Lx,
        Ly=Ly,
        nx=int(cfg.nx),
        ny=int(cfg.ny),
        poly_order=int(cfg.poly_order),
        y_interface=y_interface,
        eps_alpha=float(cfg.eps_alpha),
        adaptive_interface_target_cells=float(cfg.interface_cells),
        adaptive_interface_band_halfwidth_factor=1.0,
        adaptive_interface_max_ref=0,
    )
    min_cells = float(mesh_meta.get("interface_cells_across_2eps_min", float("nan")))
    if not np.isfinite(min_cells) or min_cells < float(cfg.min_interface_cells):
        raise RuntimeError(
            "Seboldt MOR snapshots require at least "
            f"{float(cfg.min_interface_cells):.1f} elements across 2*eps_alpha; got {min_cells:.3g}."
        )
    me = MixedElement(
        mesh,
        field_specs={
            "vf_x": int(cfg.poly_order),
            "vf_y": int(cfg.poly_order),
            "pf": int(cfg.pressure_order),
            "vp_x": int(cfg.poly_order),
            "vp_y": int(cfg.poly_order),
            "pp": int(cfg.pressure_order),
            "vs_x": int(cfg.poly_order),
            "vs_y": int(cfg.poly_order),
            "us_x": int(cfg.poly_order),
            "us_y": int(cfg.poly_order),
            "alpha": int(cfg.scalar_order),
            "phi": int(cfg.scalar_order),
            "Gamma": int(cfg.scalar_order),
        },
    )
    dh = DofHandler(me, method="cg")
    spaces = _make_spaces(dh)
    trial, test = _make_trial_test(dh, spaces)
    state = _make_state(dh)
    _initialize_state(
        dh,
        state,
        Lx=Lx,
        y_interface=y_interface,
        eps_alpha=float(cfg.eps_alpha),
        phi_b=float(cfg.phi_b),
        v_in=float(cfg.v_in),
        ramp=_cosine_ramp_value(float(cfg.dt), float(cfg.t_ramp)),
    )
    p_f_n = Function("p_f_n", "pf", dof_handler=dh)
    p_p_n = Function("p_p_n", "pp", dof_handler=dh)
    p_f_n.nodal_values[:] = state["p_f_k"].nodal_values[:]
    p_p_n.nodal_values[:] = state["p_p_k"].nodal_values[:]

    dt_c = _named_c("tc_seboldt_mor_dt", float(cfg.dt))
    kappa = 1.0e-3
    mu_f = 0.035
    mu_p = 0.035
    R_ps = mu_f / kappa
    forms = build_three_constituent_one_domain_forms(
        **state,
        **trial,
        **test,
        dx=dx(metadata={"q": int(cfg.qdeg)}),
        dt=dt_c,
        rho_f=1.0,
        rho_p=1.0,
        rho_s=1.0,
        mu_f=mu_f,
        mu_p=mu_p,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        R_fp=R_ps,
        R_fs=R_ps,
        R_ps=R_ps,
        theta_fp=0.5,
        ell_Gamma=R_ps,
        gamma_mobility="interface_delta",
        gamma_delta_epsilon=1.0e-12,
        transfer_velocity="free",
        lag_alpha_in_constitutive_laws=True,
        inactive_velocity_extension_factor=0.0,
        inactive_pressure_extension_factor=0.0,
        inactive_phi_extension_factor=0.0,
        inactive_displacement_extension_factor=0.0,
        phi_extension_value=float(cfg.phi_b),
    )
    bcs = _make_bcs(
        Lx=Lx,
        y_interface=y_interface,
        eps_alpha=float(cfg.eps_alpha),
        phi_b=float(cfg.phi_b),
        v_in=float(cfg.v_in),
        t_ramp=float(cfg.t_ramp),
    )
    solver = build_three_constituent_pdas_solver(
        forms,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=_make_homogeneous_bcs(bcs),
        newton_params=NewtonParameters(
            newton_tol=float(cfg.newton_tol),
            newton_rtol=0.0,
            max_newton_iter=int(cfg.max_newton_iter),
            print_level=1,
            line_search=True,
            ls_max_iter=16,
            ls_reduction=0.5,
        ),
        vi_params=VIParameters(
            c=1.0,
            c_by_field={"alpha": 1.0, "phi": 1.0},
            project_initial_guess=True,
            project_each_iteration=False,
            active_set_persistence=1,
            inactive_reg_lambda0=1.0e-10,
            inactive_reg_lambda_max=1.0e4,
        ),
        lin_params=LinearSolverParameters(backend=str(cfg.linear_backend), tol=1.0e-10, maxit=10000),
        backend=str(cfg.backend),
        quad_order=int(cfg.qdeg),
        alpha_bounds=(0.0, 1.0),
        phi_bounds=(0.0, 1.0),
        pore_pressure_bounds=(None, None),
    )
    return {
        "mesh": mesh,
        "mesh_meta": mesh_meta,
        "mixed_element": me,
        "dof_handler": dh,
        "trial": trial,
        "test": test,
        "state": state,
        "p_f_n": p_f_n,
        "p_p_n": p_p_n,
        "forms": forms,
        "bcs": bcs,
        "solver": solver,
        "functions": _current_functions(state),
        "prev_functions": _previous_functions(state, p_f_n, p_p_n),
    }


def _run_trajectory(cfg: SeboldtMORConfig, *, n_steps: int) -> tuple[dict[str, Any], dict[str, Any]]:
    ctx = _build_context(cfg)
    dh = ctx["dof_handler"]
    state = ctx["state"]
    p_f_n = ctx["p_f_n"]
    p_p_n = ctx["p_p_n"]
    snapshots = [_state_vector(dh, _previous_field_objects(state, p_f_n, p_p_n))]
    step_metrics: list[dict[str, float]] = []
    step_times: list[float] = []
    timer_state = {"last": time.perf_counter()}

    def _post_step(_funcs) -> None:
        now = time.perf_counter()
        step_times.append(float(now - float(timer_state["last"])))
        timer_state["last"] = now
        snapshots.append(_state_vector(dh, _current_field_objects(state)))
        metrics = _field_stats(state)
        step_metrics.append({key: float(val) for key, val in metrics.items() if isinstance(val, (int, float, np.floating))})

    ctx["solver"].post_timeloop_cb = _post_step
    t0 = time.perf_counter()
    _, accepted_steps, elapsed = ctx["solver"].solve_time_interval(
        functions=ctx["functions"],
        prev_functions=ctx["prev_functions"],
        time_params=TimeStepperParameters(
            dt=float(cfg.dt),
            final_time=float(cfg.dt) * int(n_steps),
            max_steps=int(n_steps),
            stop_on_steady=False,
            allow_dt_reduction=False,
            predictor="prev",
        ),
    )
    wall = time.perf_counter() - t0
    return ctx, {
        "snapshots": np.column_stack(snapshots),
        "step_metrics": step_metrics,
        "accepted_steps": int(accepted_steps),
        "elapsed_s": float(elapsed if elapsed is not None else wall),
        "wall_s": float(wall),
        "step_times_s": step_times,
    }


def _run_one_fom_step(cfg: SeboldtMORConfig, previous_state: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    ctx = _build_context(cfg)
    dh = ctx["dof_handler"]
    _assign_state_vector(dh, _current_field_objects(ctx["state"]), previous_state)
    _assign_state_vector(dh, _previous_field_objects(ctx["state"], ctx["p_f_n"], ctx["p_p_n"]), previous_state)
    t0 = time.perf_counter()
    _, accepted_steps, elapsed = ctx["solver"].solve_time_interval(
        functions=ctx["functions"],
        prev_functions=ctx["prev_functions"],
        time_params=TimeStepperParameters(
            dt=float(cfg.dt),
            final_time=float(cfg.dt),
            max_steps=1,
            stop_on_steady=False,
            allow_dt_reduction=False,
            predictor="prev",
        ),
    )
    wall = time.perf_counter() - t0
    return ctx, {
        "vector": _state_vector(dh, _current_field_objects(ctx["state"])),
        "accepted_steps": int(accepted_steps),
        "elapsed_s": float(elapsed if elapsed is not None else wall),
        "wall_s": float(wall),
        "active_dofs": int(np.asarray(getattr(ctx["solver"], "active_dofs", np.arange(int(dh.total_dofs))), dtype=int).size),
        "total_dofs": int(dh.total_dofs),
    }


def run_seboldt_mor_validation(cfg: SeboldtMORConfig) -> SeboldtMORResult:
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.validate_training_trajectory):
        n_total_steps = int(cfg.train_steps)
        if n_total_steps < 1:
            raise ValueError("train_steps must be positive when validate_training_trajectory is enabled.")
        validation_start_step = 0
        validation_steps = int(cfg.train_steps)
    else:
        n_total_steps = int(cfg.train_steps) + int(cfg.heldout_steps)
        if n_total_steps < 2:
            raise ValueError("train_steps + heldout_steps must be at least 2.")
        if int(cfg.heldout_steps) < 1:
            raise ValueError("heldout_steps must be positive unless validate_training_trajectory is enabled.")
        validation_start_step = int(cfg.train_steps)
        validation_steps = int(cfg.heldout_steps)

    trajectory_cache = None if cfg.trajectory_cache is None else Path(cfg.trajectory_cache)
    if bool(cfg.reuse_trajectory_cache) and trajectory_cache is not None and trajectory_cache.exists():
        trajectory_ctx = _build_context(cfg)
        trajectory = _load_trajectory_cache(trajectory_cache, expected_steps=n_total_steps)
    else:
        trajectory_ctx, trajectory = _run_trajectory(cfg, n_steps=n_total_steps)
        if trajectory_cache is not None:
            _save_trajectory_cache(trajectory_cache, trajectory, n_steps=n_total_steps)
    snapshots = np.asarray(trajectory["snapshots"], dtype=float)
    train_snapshots = snapshots[:, : int(cfg.train_steps) + 1]
    previous_state = snapshots[:, validation_start_step]
    heldout_targets = snapshots[:, validation_start_step + 1 : validation_start_step + validation_steps + 1]
    if heldout_targets.shape[1] != validation_steps:
        raise RuntimeError("trajectory did not produce the requested validation snapshots.")

    dh = trajectory_ctx["dof_handler"]
    step_times = list(float(v) for v in trajectory["step_times_s"])
    heldout_step_times = step_times[validation_start_step : validation_start_step + validation_steps]
    if len(heldout_step_times) != validation_steps:
        raise RuntimeError("trajectory did not produce the requested validation steps.")
    fom_heldout_s = float(np.sum(np.asarray(heldout_step_times, dtype=float)))

    basis_strategy = str(cfg.basis_strategy).strip().lower()
    if basis_strategy in {"fieldwise", "fieldwise_pod", "block", "block_pod"}:
        field_max_modes = int(cfg.field_max_modes) if cfg.field_max_modes is not None else int(cfg.max_modes)
        mode_map, cv_rows = _fieldwise_cross_validated_mode_map(
            dh,
            train_snapshots,
            max_modes=field_max_modes,
            tolerance=float(cfg.cv_tolerance),
        )
        fieldwise_basis = fit_fieldwise_pod_basis(
            train_snapshots,
            _logical_field_blocks(dh),
            total_dofs=int(dh.total_dofs),
            n_modes_per_block=mode_map,
            center=True,
        )
        trial_basis = fieldwise_basis.basis
        offset = fieldwise_basis.offset
        singular_values = dict(fieldwise_basis.singular_values)
        mode_count = int(trial_basis.shape[1])
        basis_metadata = dict(fieldwise_basis.metadata, mode_map=mode_map, cross_validation=cv_rows)
    elif basis_strategy in {"global", "global_pod"}:
        mode_count, cv_rows = _cross_validated_mode_count(
            train_snapshots,
            max_modes=int(cfg.max_modes),
            tolerance=float(cfg.cv_tolerance),
        )
        trial_basis, offset, singular_values = _fit_trial_basis(train_snapshots, n_modes=mode_count)
        basis_metadata = {"strategy": "global_pod", "total_modes": int(mode_count), "cross_validation": cv_rows}
    else:
        raise ValueError("basis_strategy must be 'fieldwise' or 'global'.")

    requested_target = str(cfg.target).strip().lower()
    if requested_target not in _TARGET_ALIASES:
        raise ValueError(f"unsupported Seboldt MOR target {cfg.target!r}; choose one of {tuple(sorted(_TARGET_ALIASES))}.")
    target = _TARGET_ALIASES[requested_target]
    constraint_method = str(cfg.constraint_method).strip().lower()
    if constraint_method not in {"none", "pdas", "ipm"}:
        raise ValueError("constraint_method must be 'none', 'pdas', or 'ipm'.")
    if constraint_method == "none" and target == "true_galerkin":
        raise ValueError("constraint_method='none' is not available for true Galerkin targets.")
    free_rows, fixed_rows = _dirichlet_free_rows(dh, trajectory_ctx["bcs"])
    trial_basis, kept_mode_indices, online_basis_metadata = _homogenize_basis_on_fixed_rows(trial_basis, fixed_rows)
    mode_count = int(trial_basis.shape[1])
    basis_metadata = {
        **dict(basis_metadata),
        **online_basis_metadata,
        "kept_mode_indices": kept_mode_indices,
        "total_modes": int(mode_count),
    }
    initial_coefficients = _project_online_coefficients(trial_basis, offset, previous_state, free_rows)
    train_coefficients = np.column_stack(
        [
            _project_online_coefficients(trial_basis, offset, train_snapshots[:, j], free_rows)
            for j in range(int(train_snapshots.shape[1]))
        ]
    )
    train_times = np.arange(int(train_coefficients.shape[1]), dtype=np.float64) * float(cfg.dt)
    predictor_key_config = str(cfg.online_predictor).strip().lower()
    learned_reference_predictor = None
    reference_policy = None
    if predictor_key_config in {"time_regression", "time_regressor", "learned", "reference_model"}:
        learned_reference_predictor = fit_time_parameterized_predictor(
            train_coefficients.T,
            train_times,
            degree=int(cfg.reference_predictor_degree),
            ridge=float(cfg.reference_predictor_ridge),
            metadata={"source": "seboldt_train_coefficients", "snapshot_count": int(train_coefficients.shape[1])},
        )
    train_coefficient_deltas = np.diff(np.asarray(train_coefficients, dtype=float), axis=1)
    train_decoded_deltas = np.asarray(trial_basis, dtype=float) @ train_coefficient_deltas
    train_step_norms = np.linalg.norm(train_decoded_deltas, axis=0) if train_decoded_deltas.size else np.asarray(())
    positive_train_steps = train_step_norms[train_step_norms > 0.0]
    train_step_scale = (
        float(np.median(positive_train_steps))
        if positive_train_steps.size
        else float(np.max(train_step_norms)) if train_step_norms.size else 0.0
    )
    effective_branch_radius = cfg.native_branch_radius
    if effective_branch_radius is None and cfg.native_branch_radius_factor is not None and train_step_scale > 0.0:
        effective_branch_radius = float(cfg.native_branch_radius_factor) * train_step_scale
    effective_max_step_norm = cfg.native_max_step_norm
    if effective_max_step_norm is None and cfg.native_max_step_norm_factor is not None and train_step_scale > 0.0:
        effective_max_step_norm = float(cfg.native_max_step_norm_factor) * train_step_scale
    if learned_reference_predictor is not None:
        reference_policy = ReferencePolicy(
            predictor=learned_reference_predictor,
            reference_weight=float(cfg.native_reference_weight),
            max_reference_distance=effective_branch_radius,
            max_step_norm=effective_max_step_norm,
            metric_basis=trial_basis,
            clip_reference=True,
            metadata={"benchmark": "seboldt_three_constituent"},
        )
    training_step_scale_metadata = {
        "decoded_step_norm_min": float(np.min(train_step_norms)) if train_step_norms.size else 0.0,
        "decoded_step_norm_median": float(np.median(train_step_norms)) if train_step_norms.size else 0.0,
        "decoded_step_norm_max": float(np.max(train_step_norms)) if train_step_norms.size else 0.0,
        "decoded_step_norm_last": float(train_step_norms[-1]) if train_step_norms.size else 0.0,
        "selected_scale": float(train_step_scale),
        "branch_radius_factor": None
        if cfg.native_branch_radius_factor is None
        else float(cfg.native_branch_radius_factor),
        "max_step_norm_factor": None
        if cfg.native_max_step_norm_factor is None
        else float(cfg.native_max_step_norm_factor),
        "effective_branch_radius": None if effective_branch_radius is None else float(effective_branch_radius),
        "effective_max_step_norm": None if effective_max_step_norm is None else float(effective_max_step_norm),
    }
    fom_branch_certificate = _trajectory_branch_certificate(
        dh,
        snapshots=snapshots,
        train_snapshots=train_snapshots,
        trial_basis=trial_basis,
        offset=offset,
        fixed_rows=fixed_rows,
        free_rows=free_rows,
        step_metrics=trajectory["step_metrics"],
        bound_tolerance=float(cfg.bound_violation_tolerance),
    )
    all_element_ids = np.arange(int(trajectory_ctx["mesh"].n_elements), dtype=np.int32)
    interface_element_ids = select_coordinate_band_elements(
        trajectory_ctx["mesh"],
        axis=1,
        center=1.0,
        half_width=float(cfg.eps_alpha),
    )
    sampling = None
    if target in {"sampled_gnat", "sampled_lspg"}:
        sampling = build_block_balanced_gnat_sampling(
            dh,
            trial_basis,
            snapshot_matrix=train_snapshots,
            free_rows=free_rows,
            row_blocks=field_row_blocks(dh),
            sample_rows=int(cfg.sample_rows),
            candidate_element_ids=interface_element_ids if bool(cfg.sample_interface_band_only) else None,
            mandatory_element_ids=interface_element_ids,
            min_rows_per_block=int(cfg.min_rows_per_block),
            row_weight_max=float(cfg.row_weight_max),
            rcond=1.0e-12,
        )
        row_dofs = sampling.row_dofs
        selected_basis = sampling.selected_basis if target == "sampled_gnat" else None
        residual_terms = sampling.residual_terms if target == "sampled_gnat" else None
        row_weights = sampling.row_weights
        sampled_element_ids = sampling.element_ids
        target_objective = "qdeim" if target == "sampled_gnat" else "sampled_lspg"
        sampling_metadata = dict(sampling.metadata, fixed_rows=int(fixed_rows.size))
    else:
        row_dofs = np.ascontiguousarray(free_rows, dtype=np.int64)
        selected_basis = None
        residual_terms = None
        full_row_weights = build_block_row_weights(
            train_snapshots,
            field_row_blocks(dh),
            max_weight=float(cfg.row_weight_max),
        )
        row_weights = np.ascontiguousarray(full_row_weights[row_dofs], dtype=np.float64)
        sampled_element_ids = all_element_ids
        target_objective = target
        sampling_metadata = {
            "strategy": target,
            "free_rows": int(free_rows.size),
            "fixed_rows": int(fixed_rows.size),
            "mandatory_interface_element_count": int(interface_element_ids.size),
            "uses_all_elements": True,
            "row_weight_min": float(np.min(row_weights)) if row_weights.size else 1.0,
            "row_weight_max": float(np.max(row_weights)) if row_weights.size else 1.0,
        }
    native_ctx = _build_context(cfg)
    _assign_state_vector(native_ctx["dof_handler"], _current_field_objects(native_ctx["state"]), previous_state)
    _assign_state_vector(
        native_ctx["dof_handler"],
        _previous_field_objects(native_ctx["state"], native_ctx["p_f_n"], native_ctx["p_p_n"]),
        previous_state,
    )
    residual_runner, residual_args, tangent_runner, tangent_args, coeff_args, native_setup_s = _compile_native_pair(
        native_ctx,
        qdeg=int(cfg.qdeg),
        element_ids=sampled_element_ids,
    )
    previous_tangent_runner = None
    previous_tangent_args: dict[str, Any] | None = None
    previous_tangent_param_order: tuple[str, ...] = ()
    previous_tangent_setup_s = 0.0
    if bool(cfg.dwr_previous_state_tangents):
        previous_tangent_runner, previous_tangent_args, previous_tangent_param_order, previous_tangent_setup_s = (
            _compile_native_previous_tangent(
                native_ctx,
                qdeg=int(cfg.qdeg),
                element_ids=sampled_element_ids,
            )
        )
    audit_targets_enabled = bool(cfg.audit_targets)
    audit_full_runner = None
    audit_full_args: dict[str, Any] | None = None
    audit_full_tangent_runner = None
    audit_full_tangent_args: dict[str, Any] | None = None
    audit_full_coeff_args: tuple[str, ...] = ()
    audit_full_setup_s = 0.0
    if audit_targets_enabled:
        (
            audit_full_runner,
            audit_full_args,
            audit_full_tangent_runner,
            audit_full_tangent_args,
            audit_full_coeff_args,
            audit_full_setup_s,
        ) = _compile_native_pair(
            native_ctx,
            qdeg=int(cfg.qdeg),
            element_ids=all_element_ids,
        )
    bound_constraints = bound_constraints_from_fields(
        dh,
        {"alpha": (0.0, 1.0), "phi": (0.0, 1.0)},
        metadata={"benchmark": "seboldt_three_constituent"},
    )
    reduced_bound_constraints = bound_constraints.reduce(trial_basis=trial_basis, offset=offset)
    initial_bound_violation_before_projection = float(reduced_bound_constraints.max_violation(initial_coefficients))
    initial_coefficients = project_reduced_coefficients_to_bounds(
        reduced_bound_constraints,
        initial_coefficients,
        tolerance=float(cfg.native_feasibility_tol),
        max_iterations=500,
    )
    initial_bound_violation_after_projection = float(reduced_bound_constraints.max_violation(initial_coefficients))
    from pycutfem.mor.cpp_backend.online_gauss_newton import module as _online_gauss_newton_module

    _online_gauss_newton_module()
    artifact_path = outdir / "seboldt_three_constituent_native_mor_artifact.npz"
    _write_artifact(
        artifact_path,
        trial_basis=trial_basis,
        offset=offset,
        residual_runner=residual_runner,
        tangent_runner=tangent_runner,
        row_dofs=row_dofs,
        element_ids=sampled_element_ids,
        row_weights=row_weights,
        selected_basis=selected_basis,
        residual_terms=residual_terms,
        bound_constraints=bound_constraints,
        coefficient_arg_names=coeff_args,
        solver_options={
            "target": target,
            "constraint_method": constraint_method,
            "max_iterations": int(cfg.max_native_iter),
            "residual_tol": float(cfg.native_residual_tol),
            "optimality_tol": None if cfg.native_optimality_tol is None else float(cfg.native_optimality_tol),
            "step_tol": float(cfg.native_step_tol),
            "feasibility_tol": float(cfg.native_feasibility_tol),
            "damping": float(cfg.native_damping),
            "residual_tol": float(cfg.native_residual_tol),
            "optimality_tol": None if cfg.native_optimality_tol is None else float(cfg.native_optimality_tol),
            "max_step_norm": None if effective_max_step_norm is None else float(effective_max_step_norm),
            "line_search": bool(cfg.native_line_search),
            "adaptive_damping": bool(cfg.native_adaptive_damping),
            "dirichlet_lift_state_updates": True,
        },
        metadata={
            "benchmark": "seboldt_three_constituent_mor",
            "target": target,
            "requested_target": requested_target,
            "sampling": sampling_metadata,
        },
        adjoint_dwr=NativeAdjointDWRSpec(
            qoi_name="alpha_mass",
            solver_options={"backend": "cpp", "rcond": 1.0e-12},
            estimator_options={"sign": -1.0},
            certification={"passed": False, "reason": "not_run"},
            metadata={
                "aggregation": "final",
                "description": "final-time integral of the alpha constituent",
            },
        ),
        reference_policy=reference_policy,
        target_objective=target_objective,
    )

    residual_metadata = native_kernel_metadata_from_runner(residual_runner)
    tangent_metadata = native_kernel_metadata_from_runner(tangent_runner)
    previous_tangent_metadata = (
        native_kernel_metadata_from_runner(previous_tangent_runner)
        if previous_tangent_runner is not None
        else None
    )
    audit_full_residual_metadata = (
        native_kernel_metadata_from_runner(audit_full_runner)
        if audit_full_runner is not None
        else None
    )
    audit_full_tangent_metadata = (
        native_kernel_metadata_from_runner(audit_full_tangent_runner)
        if audit_full_tangent_runner is not None
        else None
    )
    q = np.ascontiguousarray(np.asarray(initial_coefficients, dtype=np.float64).reshape(-1))
    q_previous_for_predictor: np.ndarray | None = None
    rom_previous_state = np.ascontiguousarray(np.asarray(previous_state, dtype=np.float64).reshape(-1))
    native_step_summaries: list[dict[str, Any]] = []
    native_results: list[Any] = []
    rom_vectors: list[np.ndarray] = []
    projection_vectors: list[np.ndarray] = []
    relative_error_history: list[float] = []
    projection_error_history: list[float] = []
    bound_violation_history: list[float] = []
    native_online_times: list[float] = []
    updated_previous_arg_names: set[str] = set()
    dirichlet_lift_arg_names: set[str] = set()
    branch_radius_schedule = _schedule_with_default(
        _parse_optional_float_schedule(cfg.native_branch_radius_schedule),
        effective_branch_radius,
    )
    max_step_norm_schedule = _schedule_with_default(
        _parse_optional_float_schedule(cfg.native_max_step_norm_schedule),
        effective_max_step_norm,
    )
    native_dt_arg_names: set[str] = set()
    dwr_residual_history: list[np.ndarray] = []
    dwr_jacobian_history: list[np.ndarray] = []
    dwr_previous_jacobian_history: list[np.ndarray] = []
    target_audit_history: list[dict[str, Any]] = []

    def _audit_target_state(
        *,
        label: str,
        target_name: str,
        coefficients: np.ndarray,
        previous_vector: np.ndarray,
        lift_values: np.ndarray,
        residual_runner_obj: Any,
        residual_metadata_obj: Any,
        residual_args_obj: dict[str, Any],
        tangent_runner_obj: Any,
        tangent_metadata_obj: Any,
        tangent_args_obj: dict[str, Any],
        coefficient_arg_names: tuple[str, ...],
        audit_row_dofs: np.ndarray,
        audit_selected_basis: np.ndarray | None = None,
        audit_residual_terms: np.ndarray | None = None,
        audit_row_weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        _set_native_dt_args(residual_args_obj, tangent_args_obj, dt=float(cfg.dt))
        _refresh_previous_state_static_args(residual_args_obj, previous_vector)
        _refresh_previous_state_static_args(tangent_args_obj, previous_vector)
        residual_updates = build_dirichlet_lift_state_updates(
            residual_args_obj,
            coefficient_arg_names,
            trial_basis=trial_basis,
            offset=offset,
            fixed_rows=fixed_rows,
            lift_values=lift_values,
        )
        tangent_updates = build_dirichlet_lift_state_updates(
            tangent_args_obj,
            coefficient_arg_names,
            trial_basis=trial_basis,
            offset=offset,
            fixed_rows=fixed_rows,
            lift_values=lift_values,
        )
        q_eval = np.ascontiguousarray(np.asarray(coefficients, dtype=np.float64).reshape(-1))
        apply_affine_updates_to_static_args(residual_args_obj, residual_updates, q_eval)
        apply_affine_updates_to_static_args(tangent_args_obj, tangent_updates, q_eval)
        residual, jacobian = reduced_target_from_native_kernel_pair(
            residual_metadata_capsule=residual_metadata_obj,
            residual_param_order=residual_runner_obj.param_order,
            residual_static_args=residual_args_obj,
            tangent_metadata_capsule=tangent_metadata_obj,
            tangent_param_order=tangent_runner_obj.param_order,
            tangent_static_args=tangent_args_obj,
            trial_basis=trial_basis,
            row_dofs=audit_row_dofs,
            target=target_name,
            selected_basis=audit_selected_basis,
            residual_terms=audit_residual_terms,
            row_weights=audit_row_weights,
        )
        return {
            "state": label,
            "target": target_name,
            **_reduced_target_metrics(residual, jacobian),
        }

    def _audit_step_targets(
        *,
        step: int,
        rom_coefficients: np.ndarray,
        projection_coefficients: np.ndarray,
        rom_previous_vector: np.ndarray,
        fom_previous_vector: np.ndarray,
        fom_current_vector: np.ndarray,
    ) -> dict[str, Any]:
        if not audit_targets_enabled:
            return {"enabled": False}
        if (
            audit_full_runner is None
            or audit_full_tangent_runner is None
            or audit_full_args is None
            or audit_full_tangent_args is None
            or audit_full_residual_metadata is None
            or audit_full_tangent_metadata is None
        ):
            raise RuntimeError("target audit was enabled but full native audit kernels were not compiled.")
        rows: list[dict[str, Any]] = []
        previous_cases = (
            ("online_previous", rom_previous_vector),
            ("fom_previous", fom_previous_vector),
        )
        for previous_label, previous_vector in previous_cases:
            for state_label, coeffs in (
                ("projection", projection_coefficients),
                ("rom", rom_coefficients),
            ):
                for audit_target in ("true_galerkin", "full_row_lspg"):
                    rows.append(
                        {
                            "previous": previous_label,
                            **_audit_target_state(
                                label=state_label,
                                target_name=audit_target,
                                coefficients=coeffs,
                                previous_vector=previous_vector,
                                lift_values=fom_current_vector,
                                residual_runner_obj=audit_full_runner,
                                residual_metadata_obj=audit_full_residual_metadata,
                                residual_args_obj=audit_full_args,
                                tangent_runner_obj=audit_full_tangent_runner,
                                tangent_metadata_obj=audit_full_tangent_metadata,
                                tangent_args_obj=audit_full_tangent_args,
                                coefficient_arg_names=audit_full_coeff_args,
                                audit_row_dofs=free_rows,
                            ),
                        }
                    )
                if sampling is not None:
                    rows.append(
                        {
                            "previous": previous_label,
                            **_audit_target_state(
                                label=state_label,
                                target_name="sampled_lspg",
                                coefficients=coeffs,
                                previous_vector=previous_vector,
                                lift_values=fom_current_vector,
                                residual_runner_obj=residual_runner,
                                residual_metadata_obj=residual_metadata,
                                residual_args_obj=residual_args,
                                tangent_runner_obj=tangent_runner,
                                tangent_metadata_obj=tangent_metadata,
                                tangent_args_obj=tangent_args,
                                coefficient_arg_names=coeff_args,
                                audit_row_dofs=row_dofs,
                                audit_row_weights=row_weights,
                            ),
                        }
                    )
                    if selected_basis is not None and residual_terms is not None:
                        rows.append(
                            {
                                "previous": previous_label,
                                **_audit_target_state(
                                    label=state_label,
                                    target_name="sampled_gnat",
                                    coefficients=coeffs,
                                    previous_vector=previous_vector,
                                    lift_values=fom_current_vector,
                                    residual_runner_obj=residual_runner,
                                    residual_metadata_obj=residual_metadata,
                                    residual_args_obj=residual_args,
                                    tangent_runner_obj=tangent_runner,
                                    tangent_metadata_obj=tangent_metadata,
                                    tangent_args_obj=tangent_args,
                                    coefficient_arg_names=coeff_args,
                                    audit_row_dofs=row_dofs,
                                    audit_selected_basis=selected_basis,
                                    audit_residual_terms=residual_terms,
                                    audit_row_weights=row_weights,
                                ),
                            }
                        )
        comparisons: list[dict[str, Any]] = []
        for previous_label, _ in previous_cases:
            for audit_target in ("true_galerkin", "full_row_lspg", "sampled_lspg", "sampled_gnat"):
                target_rows = [
                    row for row in rows
                    if row["previous"] == previous_label and row["target"] == audit_target
                ]
                by_state = {row["state"]: row for row in target_rows}
                if "projection" not in by_state or "rom" not in by_state:
                    continue
                projection_norm = float(by_state["projection"]["residual_norm"])
                rom_norm = float(by_state["rom"]["residual_norm"])
                comparisons.append(
                    {
                        "previous": previous_label,
                        "target": audit_target,
                        "rom_over_projection_residual": float(rom_norm / max(projection_norm, 1.0e-300)),
                        "projection_residual_norm": projection_norm,
                        "rom_residual_norm": rom_norm,
                    }
                )
        return {
            "enabled": True,
            "step": int(step),
            "rows": rows,
            "comparisons": comparisons,
        }

    for heldout_idx in range(validation_steps):
        fom_step_vector = np.asarray(heldout_targets[:, heldout_idx], dtype=np.float64).reshape(-1)
        step_reference_q = np.ascontiguousarray(q.copy(), dtype=np.float64)
        predictor_source = str(cfg.online_predictor)
        predictor_key = str(cfg.online_predictor).strip().lower()
        if predictor_key in {"time_regression", "time_regressor", "learned", "reference_model"}:
            if reference_policy is None:
                raise RuntimeError("time-regression online predictor was requested but no reference policy was fitted.")
            prediction = reference_policy.predict(
                time=float(validation_start_step + heldout_idx + 1) * float(cfg.dt),
                dt=float(cfg.dt),
                q_current=step_reference_q,
                q_previous=q_previous_for_predictor,
            )
            step_start_q = np.ascontiguousarray(prediction.coefficients, dtype=np.float64)
            predictor_source = str(prediction.metadata.get("predictor_kind", "time_parameterized"))
            learned_prediction_metadata = dict(prediction.metadata or {})
        else:
            step_start_q = _predict_online_coefficients(
                step_reference_q,
                q_previous_for_predictor,
                predictor=str(cfg.online_predictor),
            )
            learned_prediction_metadata = {}
        training_projection_index = int(validation_start_step + heldout_idx + 1)
        if predictor_key in {"trajectory", "offline_trajectory", "training_projection", "snapshot_projection"}:
            if 0 <= training_projection_index < int(train_coefficients.shape[1]):
                step_start_q = np.ascontiguousarray(
                    np.asarray(train_coefficients[:, training_projection_index], dtype=np.float64).reshape(-1),
                    dtype=np.float64,
                )
                predictor_source = "training_projection"
            else:
                predictor_source = "linear_fallback"
        (
            step_start_q,
            predictor_clipped,
            predictor_decoded_distance_before_clip,
            predictor_decoded_distance_after_clip,
        ) = _clip_decoded_predictor_distance(
            step_start_q,
            step_reference_q,
            trial_basis,
            effective_branch_radius,
        )
        step_start_q = project_reduced_coefficients_to_bounds(
            reduced_bound_constraints,
            step_start_q,
            tolerance=float(cfg.native_feasibility_tol),
            max_iterations=500,
        )
        predictor_coeff_distance = float(np.linalg.norm(step_start_q - step_reference_q))
        predictor_decoded_distance = float(np.linalg.norm(trial_basis @ (step_start_q - step_reference_q)))
        step_start_previous_state = np.ascontiguousarray(rom_previous_state.copy(), dtype=np.float64)

        def _run_reduced_substep(
            *,
            current_q: np.ndarray,
            current_previous_state: np.ndarray,
            lift_values: np.ndarray,
            substep_dt: float,
        ) -> dict[str, Any]:
            native_dt_arg_names.update(_set_native_dt_args(residual_args, tangent_args, dt=substep_dt))
            updated_previous_arg_names.update(_refresh_previous_state_static_args(residual_args, current_previous_state))
            updated_previous_arg_names.update(_refresh_previous_state_static_args(tangent_args, current_previous_state))
            residual_lift_updates = build_dirichlet_lift_state_updates(
                residual_args,
                coeff_args,
                trial_basis=trial_basis,
                offset=offset,
                fixed_rows=fixed_rows,
                lift_values=lift_values,
            )
            tangent_lift_updates = build_dirichlet_lift_state_updates(
                tangent_args,
                coeff_args,
                trial_basis=trial_basis,
                offset=offset,
                fixed_rows=fixed_rows,
                lift_values=lift_values,
            )
            dirichlet_lift_arg_names.update(update.name for update in residual_lift_updates)
            dirichlet_lift_arg_names.update(update.name for update in tangent_lift_updates)

            native_common = dict(
                residual_metadata_capsule=residual_metadata,
                residual_param_order=residual_runner.param_order,
                residual_static_args=residual_args,
                tangent_metadata_capsule=tangent_metadata,
                tangent_param_order=tangent_runner.param_order,
                tangent_static_args=tangent_args,
                trial_basis=trial_basis,
                offset=offset,
                initial_coefficients=current_q,
                row_dofs=row_dofs,
                bound_constraints=bound_constraints,
                coefficient_arg_names=coeff_args,
                row_weights=row_weights,
                residual_state_updates=residual_lift_updates,
                tangent_state_updates=tangent_lift_updates,
                max_iterations=int(cfg.max_native_iter),
                residual_tol=float(cfg.native_residual_tol),
                optimality_tol=None if cfg.native_optimality_tol is None else float(cfg.native_optimality_tol),
                step_tol=float(cfg.native_step_tol),
                damping=float(cfg.native_damping),
                adaptive_damping=bool(cfg.native_adaptive_damping),
                line_search=bool(cfg.native_line_search),
                max_line_search=10,
                active_tol=1.0e-10,
                feasibility_tol=float(cfg.native_feasibility_tol),
                max_step_norm=effective_max_step_norm,
                reference_coefficients=current_q,
                reference_weight=float(cfg.native_reference_weight),
                max_reference_distance=effective_branch_radius,
                state_merit_weight=float(cfg.native_state_merit_weight),
                require_residual_convergence=bool(cfg.native_require_residual_convergence),
                rcond=1.0e-12,
            )

            def _solve_native_once(**options: Any) -> Any:
                if constraint_method == "none":
                    unconstrained_options = dict(options)
                    for key in (
                        "bound_constraints",
                        "constraint_method",
                        "active_tol",
                        "feasibility_tol",
                    ):
                        unconstrained_options.pop(key, None)
                    if target == "sampled_gnat":
                        return solve_native_deim_online_gauss_newton(
                            **unconstrained_options,
                            selected_basis=selected_basis,
                            residual_terms=residual_terms,
                        )
                    if target == "true_galerkin":
                        raise ValueError("constraint_method='none' is not available for true Galerkin targets.")
                    return solve_native_online_gauss_newton(**unconstrained_options)
                if target == "sampled_gnat":
                    constrained_options = dict(options)
                    constrained_options.pop("reference_weight", None)
                    return solve_native_bound_constrained_deim_online_gauss_newton(
                        **constrained_options,
                        constraint_method=constraint_method,
                        selected_basis=selected_basis,
                        residual_terms=residual_terms,
                    )
                if target == "true_galerkin":
                    constrained_options = dict(options)
                    constrained_options.pop("reference_weight", None)
                    return solve_native_bound_constrained_galerkin_online_gauss_newton(
                        **constrained_options,
                        constraint_method=constraint_method,
                    )
                constrained_options = dict(options)
                constrained_options.pop("reference_weight", None)
                return solve_native_bound_constrained_online_gauss_newton(
                    **constrained_options,
                    constraint_method=constraint_method,
                )

            native_t0 = time.perf_counter()
            continuation_attempts: list[dict[str, Any]] = []
            continuation_accepted = True
            continuation_options: dict[str, Any] = {
                "max_reference_distance": None if effective_branch_radius is None else float(effective_branch_radius),
                "max_step_norm": None if effective_max_step_norm is None else float(effective_max_step_norm),
            }
            if bool(cfg.native_branch_backtracking):
                continuation = solve_with_branch_backtracking(
                    _solve_native_once,
                    base_options=native_common,
                    branch_radii=branch_radius_schedule,
                    trust_radii=max_step_norm_schedule,
                    accept=lambda result: _native_online_status_ok(
                        result,
                        float(cfg.native_residual_tol),
                        cfg.native_optimality_tol,
                        float(cfg.native_step_tol),
                    ),
                )
                if isinstance(continuation.result, Exception):
                    raise continuation.result
                native_result = continuation.result
                continuation_accepted = bool(continuation.accepted)
                continuation_attempts = _continuation_attempts_to_json(continuation.attempts)
                continuation_options = {
                    "max_reference_distance": None
                    if continuation.options.get("max_reference_distance") is None
                    else float(continuation.options["max_reference_distance"]),
                    "max_step_norm": None
                    if continuation.options.get("max_step_norm") is None
                    else float(continuation.options["max_step_norm"]),
                }
            else:
                native_result = _solve_native_once(**native_common)
            elapsed = time.perf_counter() - native_t0
            q_next = np.ascontiguousarray(np.asarray(native_result.coefficients, dtype=np.float64).reshape(-1))
            rom_next = _decode_online_state(
                trial_basis,
                offset,
                q_next,
                fixed_rows=fixed_rows,
                lift_values=lift_values,
            )
            return {
                "native": native_result,
                "online_s": float(elapsed),
                "q": q_next,
                "rom": rom_next,
                "continuation_accepted": bool(continuation_accepted),
                "continuation_attempts": continuation_attempts,
                "continuation_options": continuation_options,
                "residual_lift_updates": residual_lift_updates,
                "tangent_lift_updates": tangent_lift_updates,
            }

        selected_step: dict[str, Any] | None = None
        substep_records: list[dict[str, Any]] = []
        for n_substeps in _substep_counts(
            int(cfg.native_max_substeps),
            int(cfg.native_substep_factor),
            min_substeps=int(cfg.native_min_substeps),
        ):
            q_attempt = np.ascontiguousarray(step_start_q.copy(), dtype=np.float64)
            previous_attempt = np.ascontiguousarray(step_start_previous_state.copy(), dtype=np.float64)
            substep_records = []
            ok = True
            last_substep: dict[str, Any] | None = None
            for sub_idx in range(int(n_substeps)):
                theta = float(sub_idx + 1) / float(n_substeps)
                lift_values = np.ascontiguousarray(
                    step_start_previous_state + theta * (fom_step_vector - step_start_previous_state),
                    dtype=np.float64,
                )
                try:
                    last_substep = _run_reduced_substep(
                        current_q=q_attempt,
                        current_previous_state=previous_attempt,
                        lift_values=lift_values,
                        substep_dt=float(cfg.dt) / float(n_substeps),
                    )
                except Exception as exc:
                    substep_records.append(
                        {
                            "substep": int(sub_idx + 1),
                            "substeps": int(n_substeps),
                            "accepted": False,
                            "message": str(exc),
                        }
                    )
                    ok = False
                    break
                native_sub = last_substep["native"]
                sub_ok = bool(
                    last_substep["continuation_accepted"]
                    and _native_online_status_ok(
                        native_sub,
                        float(cfg.native_residual_tol),
                        cfg.native_optimality_tol,
                        float(cfg.native_step_tol),
                    )
                )
                substep_records.append(
                    {
                        "substep": int(sub_idx + 1),
                        "substeps": int(n_substeps),
                        "accepted": bool(sub_ok),
                        "online_s": float(last_substep["online_s"]),
                        "residual_norm": float(native_sub.residual_norm),
                        "optimality_norm": float(getattr(native_sub, "optimality_norm", float("nan"))),
                        "iterations": int(native_sub.iterations),
                        "continuation_accepted": bool(last_substep["continuation_accepted"]),
                        "continuation_attempts": last_substep["continuation_attempts"],
                    }
                )
                q_attempt = last_substep["q"]
                previous_attempt = last_substep["rom"]
                if not sub_ok:
                    ok = False
                    break
            if ok and last_substep is not None:
                selected_step = {
                    **last_substep,
                    "substeps_used": int(n_substeps),
                    "substep_records": substep_records,
                    "online_s": float(sum(float(row.get("online_s", 0.0)) for row in substep_records)),
                }
                break
        if selected_step is None:
            details = json.dumps(_json_finite(substep_records), sort_keys=True)
            if not bool(cfg.native_accept_failed_steps):
                raise RuntimeError(
                    f"native reduced step {heldout_idx + 1} failed all convergence checks; "
                    f"refusing to advance a failed reduced state. Attempts: {details}"
                )
            if last_substep is None:
                raise RuntimeError("native reduced substepping did not produce a candidate step: " + details)
            selected_step = {
                **last_substep,
                "substeps_used": int(substep_records[-1]["substeps"]) if substep_records else 1,
                "substep_records": substep_records,
                "online_s": float(sum(float(row.get("online_s", 0.0)) for row in substep_records)),
            }

        native = selected_step["native"]
        native_step_online_s = float(selected_step["online_s"])
        native_continuation_accepted_i = bool(selected_step["continuation_accepted"])
        native_continuation_attempts_i = list(selected_step["continuation_attempts"])
        native_continuation_options_i = dict(selected_step["continuation_options"])
        substeps_used_i = int(selected_step["substeps_used"])
        substep_records_i = list(selected_step["substep_records"])
        residual_lift_updates = selected_step["residual_lift_updates"]
        tangent_lift_updates = selected_step["tangent_lift_updates"]
        q_previous_for_predictor = step_reference_q
        q = np.ascontiguousarray(np.asarray(native.coefficients, dtype=np.float64).reshape(-1))
        rom_step_vector = _decode_online_state(
            trial_basis,
            offset,
            q,
            fixed_rows=fixed_rows,
            lift_values=fom_step_vector,
        )
        apply_affine_updates_to_static_args(residual_args, residual_lift_updates, q)
        apply_affine_updates_to_static_args(tangent_args, tangent_lift_updates, q)
        dwr_residual_i, dwr_jacobian_i = reduced_target_from_native_kernel_pair(
            residual_metadata_capsule=residual_metadata,
            residual_param_order=residual_runner.param_order,
            residual_static_args=residual_args,
            tangent_metadata_capsule=tangent_metadata,
            tangent_param_order=tangent_runner.param_order,
            tangent_static_args=tangent_args,
            trial_basis=trial_basis,
            row_dofs=row_dofs,
            target=target,
            selected_basis=selected_basis,
            residual_terms=residual_terms,
            row_weights=row_weights,
        )
        dwr_residual_history.append(dwr_residual_i)
        dwr_jacobian_history.append(dwr_jacobian_i)
        if (
            previous_tangent_runner is not None
            and previous_tangent_args is not None
            and previous_tangent_metadata is not None
        ):
            native_dt_arg_names.update(_set_native_dt_args(previous_tangent_args, dt=float(cfg.dt)))
            updated_previous_arg_names.update(
                _refresh_previous_state_static_args(previous_tangent_args, step_start_previous_state)
            )
            previous_tangent_lift_updates = build_dirichlet_lift_state_updates(
                previous_tangent_args,
                coeff_args,
                trial_basis=trial_basis,
                offset=offset,
                fixed_rows=fixed_rows,
                lift_values=fom_step_vector,
            )
            dirichlet_lift_arg_names.update(update.name for update in previous_tangent_lift_updates)
            apply_affine_updates_to_static_args(previous_tangent_args, previous_tangent_lift_updates, q)
            _, dwr_previous_jacobian_i = reduced_target_from_native_kernel_pair(
                residual_metadata_capsule=residual_metadata,
                residual_param_order=residual_runner.param_order,
                residual_static_args=residual_args,
                tangent_metadata_capsule=previous_tangent_metadata,
                tangent_param_order=previous_tangent_param_order,
                tangent_static_args=previous_tangent_args,
                trial_basis=trial_basis,
                row_dofs=row_dofs,
                target=target,
                selected_basis=selected_basis,
                residual_terms=residual_terms,
                row_weights=row_weights,
            )
            dwr_previous_jacobian_history.append(dwr_previous_jacobian_i)
        projection_coeffs = _project_online_coefficients(trial_basis, offset, fom_step_vector, free_rows)
        projection_step_vector = _decode_online_state(
            trial_basis,
            offset,
            projection_coeffs,
            fixed_rows=fixed_rows,
            lift_values=fom_step_vector,
        )
        audit_i = _audit_step_targets(
            step=heldout_idx + 1,
            rom_coefficients=q,
            projection_coefficients=projection_coeffs,
            rom_previous_vector=step_start_previous_state,
            fom_previous_vector=np.asarray(snapshots[:, validation_start_step + heldout_idx], dtype=np.float64).reshape(-1),
            fom_current_vector=fom_step_vector,
        )
        if bool(audit_i.get("enabled", False)):
            target_audit_history.append(audit_i)
        rel_error = float(np.linalg.norm(rom_step_vector - fom_step_vector) / max(np.linalg.norm(fom_step_vector), 1.0e-14))
        proj_error = float(
            np.linalg.norm(projection_step_vector - fom_step_vector) / max(np.linalg.norm(fom_step_vector), 1.0e-14)
        )
        rom_step_bounds = _bound_stats(dh, rom_step_vector)
        last_step_norm_i = (
            float(native.step_norm_history[-1])
            if np.asarray(native.step_norm_history, dtype=float).size
            else float("inf")
        )
        normalized_last_step_i = last_step_norm_i / max(float(np.linalg.norm(native.coefficients)), 1.0)
        stationary_i = bool(normalized_last_step_i <= float(cfg.native_step_tol))
        status_i = native_online_convergence_status(
            native,
            residual_tol=float(cfg.native_residual_tol),
            optimality_tol=cfg.native_optimality_tol,
            step_tol=float(cfg.native_step_tol),
            accept_stationary=False,
        )
        optimality_norm_i = float(status_i.optimality_norm)
        optimality_ok_i = bool(status_i.optimality_ok)
        status_ok_i = bool(status_i.ok)
        step_passed_i = bool(
            status_ok_i
            and native_continuation_accepted_i
            and rom_step_bounds["max_violation"] <= float(cfg.bound_violation_tolerance)
            and rel_error <= float(cfg.error_tolerance)
            and proj_error <= float(cfg.projection_error_tolerance)
        )
        native_step_summaries.append(
            {
                "step": int(heldout_idx + 1),
                "time": float(validation_start_step + heldout_idx + 1) * float(cfg.dt),
                "online_s": float(native_step_online_s),
                "converged": bool(native.converged),
                "iterations": int(native.iterations),
                "residual_norm": float(native.residual_norm),
                "optimality_norm": optimality_norm_i,
                "stationary": bool(stationary_i),
                "optimality_ok": bool(optimality_ok_i),
                "status_ok": bool(status_ok_i),
                "continuation_accepted": bool(native_continuation_accepted_i),
                "continuation_options": native_continuation_options_i,
                "continuation_attempts": native_continuation_attempts_i,
                "online_predictor": str(cfg.online_predictor),
                "predictor_source": predictor_source,
                "learned_prediction": learned_prediction_metadata,
                "predictor_clipped": bool(predictor_clipped),
                "predictor_decoded_distance_before_clip": float(predictor_decoded_distance_before_clip),
                "predictor_decoded_distance_after_clip": float(predictor_decoded_distance_after_clip),
                "predictor_coeff_distance": float(predictor_coeff_distance),
                "predictor_decoded_distance": float(predictor_decoded_distance),
                "substeps_used": int(substeps_used_i),
                "substeps": substep_records_i,
                "last_step_norm": float(last_step_norm_i),
                "normalized_last_step": float(normalized_last_step_i),
                "relative_state_vs_fom": float(rel_error),
                "projection_relative_state_vs_fom": float(proj_error),
                "bound_violation": float(rom_step_bounds["max_violation"]),
                "target_audit": audit_i,
                "passed": bool(step_passed_i),
            }
        )
        native_results.append(native)
        native_online_times.append(float(native_step_online_s))
        rom_vectors.append(rom_step_vector)
        projection_vectors.append(projection_step_vector)
        relative_error_history.append(float(rel_error))
        projection_error_history.append(float(proj_error))
        bound_violation_history.append(float(rom_step_bounds["max_violation"]))
        rom_previous_state = rom_step_vector

    if not native_results:
        raise RuntimeError("native reduced trajectory did not run any held-out steps.")
    native = native_results[-1]
    rom_vector = rom_vectors[-1]
    fom_vector = np.asarray(heldout_targets[:, -1], dtype=np.float64).reshape(-1)
    projection_vector = projection_vectors[-1]
    native_online_s = float(np.sum(np.asarray(native_online_times, dtype=float)))
    relative_state_error = float(np.max(np.asarray(relative_error_history, dtype=float)))
    projection_state_error = float(np.max(np.asarray(projection_error_history, dtype=float)))
    rom_bounds = _bound_stats(dh, rom_vector)
    max_bound_violation = float(np.max(np.asarray(bound_violation_history, dtype=float)))
    last_step_norm = (
        float(native.step_norm_history[-1])
        if np.asarray(native.step_norm_history, dtype=float).size
        else float("inf")
    )
    normalized_last_step = last_step_norm / max(float(np.linalg.norm(native.coefficients)), 1.0)
    native_stationary = bool(normalized_last_step <= float(cfg.native_step_tol))
    native_status_ok = bool(all(bool(row["status_ok"]) for row in native_step_summaries))
    passed = bool(
        bool(fom_branch_certificate["passed"])
        and native_status_ok
        and max_bound_violation <= float(cfg.bound_violation_tolerance)
        and relative_state_error <= float(cfg.error_tolerance)
        and projection_state_error <= float(cfg.projection_error_tolerance)
    )
    replay_validation = bool(
        cfg.validate_training_trajectory
        and str(cfg.online_predictor).strip().lower()
        in {"trajectory", "offline_trajectory", "training_projection", "snapshot_projection"}
    )
    predictive_validation_passed = bool(passed and not replay_validation)

    heldout_step_times_arr = np.asarray(heldout_step_times, dtype=float)
    fom_median_s = float(np.median(heldout_step_times_arr)) if heldout_step_times_arr.size else fom_heldout_s
    fom_mean_s = float(np.mean(heldout_step_times_arr)) if heldout_step_times_arr.size else fom_heldout_s
    speedup = fom_heldout_s / max(float(native_online_s), 1.0e-14)
    median_speedup = fom_median_s / max(float(np.median(np.asarray(native_online_times, dtype=float))), 1.0e-14)
    mean_speedup = fom_mean_s / max(float(np.mean(np.asarray(native_online_times, dtype=float))), 1.0e-14)
    validated_speedup = speedup if passed else None
    validated_median_speedup = median_speedup if passed else None
    validated_mean_speedup = mean_speedup if passed else None
    audit_projection_full_row_norms = [
        float(row["residual_norm"])
        for step in target_audit_history
        for row in step.get("rows", ())
        if row.get("previous") == "fom_previous"
        and row.get("state") == "projection"
        and row.get("target") == "full_row_lspg"
    ]
    audit_projection_sampled_gnat_norms = [
        float(row["residual_norm"])
        for step in target_audit_history
        for row in step.get("rows", ())
        if row.get("previous") == "fom_previous"
        and row.get("state") == "projection"
        and row.get("target") == "sampled_gnat"
    ]
    audit_rom_full_row_norms = [
        float(row["residual_norm"])
        for step in target_audit_history
        for row in step.get("rows", ())
        if row.get("previous") == "online_previous"
        and row.get("state") == "rom"
        and row.get("target") == "full_row_lspg"
    ]
    audit_rom_sampled_gnat_norms = [
        float(row["residual_norm"])
        for step in target_audit_history
        for row in step.get("rows", ())
        if row.get("previous") == "online_previous"
        and row.get("state") == "rom"
        and row.get("target") == "sampled_gnat"
    ]

    alpha_qoi_form = native_ctx["state"]["alpha_k"] * dx(metadata={"q": int(cfg.qdeg)})
    alpha_test = TestFunction("alpha", dof_handler=native_ctx["dof_handler"])
    _assign_state_vector(native_ctx["dof_handler"], _current_field_objects(native_ctx["state"]), rom_vector)
    rom_alpha_mass = evaluate_qoi_functional(
        alpha_qoi_form,
        dof_handler=native_ctx["dof_handler"],
        backend="cpp",
        quad_order=int(cfg.qdeg),
        name="alpha_mass",
    )
    alpha_gradient_full = assemble_qoi_gradient(
        alpha_qoi_form,
        native_ctx["state"]["alpha_k"],
        alpha_test,
        dof_handler=native_ctx["dof_handler"],
        backend="cpp",
        quad_order=int(cfg.qdeg),
    )
    alpha_gradient_reduced = reduced_qoi_gradient_from_full(alpha_gradient_full, trial_basis)
    _assign_state_vector(native_ctx["dof_handler"], _current_field_objects(native_ctx["state"]), fom_vector)
    fom_alpha_mass = evaluate_qoi_functional(
        alpha_qoi_form,
        dof_handler=native_ctx["dof_handler"],
        backend="cpp",
        quad_order=int(cfg.qdeg),
        name="alpha_mass",
    )
    _assign_state_vector(native_ctx["dof_handler"], _current_field_objects(native_ctx["state"]), rom_vector)

    dwr_residuals = np.stack(dwr_residual_history, axis=0)
    dwr_jacobians = np.stack(dwr_jacobian_history, axis=0)
    dwr_previous_jacobians = (
        np.stack(dwr_previous_jacobian_history, axis=0)
        if dwr_previous_jacobian_history
        else None
    )
    dwr_qoi_gradients = np.zeros((dwr_residuals.shape[0], dwr_jacobians.shape[2]), dtype=np.float64)
    dwr_qoi_gradients[-1, :] = alpha_gradient_reduced
    alpha_qoi_error = float(fom_alpha_mass - rom_alpha_mass)
    previous_state_jacobian_status = (
        "native_generated_reduced"
        if dwr_previous_jacobians is not None
        else "not_recorded"
    )
    dwr_trajectory = DWRReducedTrajectory(
        residuals=dwr_residuals,
        jacobians=dwr_jacobians,
        qoi_gradients=dwr_qoi_gradients,
        previous_state_jacobians=dwr_previous_jacobians,
        reference_qoi_error=alpha_qoi_error,
        metadata={
            "benchmark": "seboldt_three_constituent_mor",
            "qoi_name": "alpha_mass",
            "aggregation": "final",
            "target": target,
            "previous_state_jacobians": previous_state_jacobian_status,
            "previous_state_jacobian_note": (
                "Native generated reduced previous-state tangents are included."
                if dwr_previous_jacobians is not None
                else "Final-step certificate is exact for one held-out step. Multi-step certification must add reduced previous-state tangents."
            ),
        },
    )
    dwr_trajectory_path = outdir / "seboldt_three_constituent_alpha_mass_dwr_trajectory.npz"
    dwr_trajectory.save(dwr_trajectory_path)
    dwr_backend_fallback: str | None = None
    try:
        dwr_certificate = certify_dual_weighted_residual_from_artifact_trajectory(
            artifact_path,
            dwr_trajectory,
            reference_qoi_error=alpha_qoi_error,
            metadata={"trajectory_source": str(dwr_trajectory_path)},
        )
    except Exception as exc:
        dwr_backend_fallback = str(exc)
        dwr_certificate = certify_dual_weighted_residual(
            tuple(dwr_trajectory.residuals[i, :] for i in range(dwr_trajectory.n_steps)),
            tuple(dwr_trajectory.jacobians[i, :, :] for i in range(dwr_trajectory.n_steps)),
            tuple(dwr_trajectory.qoi_gradients[i, :] for i in range(dwr_trajectory.n_steps)),
            previous_state_jacobians=(
                None
                if dwr_trajectory.previous_state_jacobians is None
                else tuple(dwr_trajectory.previous_state_jacobians[i, :, :] for i in range(dwr_trajectory.n_steps))
            ),
            adjoint_dwr=NativeAdjointDWRSpec(
                qoi_name="alpha_mass",
                solver_options={"backend": "python", "rcond": 1.0e-12},
                estimator_options={"sign": -1.0},
            ),
            reference_qoi_error=alpha_qoi_error,
            metadata={
                "trajectory_source": str(dwr_trajectory_path),
                "cpp_backend_error": dwr_backend_fallback,
            },
        )

    summary: dict[str, Any] = {
        "passed": passed,
        "problem": {
            "name": "seboldt_three_constituent_physical_mor",
            "logical_fields": tuple(LOGICAL_FIELD_GROUPS),
            "nx": int(cfg.nx),
            "ny": int(cfg.ny),
            "poly_order": int(cfg.poly_order),
            "pressure_order": int(cfg.pressure_order),
            "scalar_order": int(cfg.scalar_order),
            "total_dofs": int(dh.total_dofs),
            "active_dofs": int(np.asarray(getattr(trajectory_ctx["solver"], "active_dofs", np.arange(int(dh.total_dofs))), dtype=int).size),
            "dt": float(cfg.dt),
            "train_steps": int(cfg.train_steps),
            "heldout_steps": int(cfg.heldout_steps),
            "validation_start_step": int(validation_start_step),
            "validation_steps": int(validation_steps),
            "validate_training_trajectory": bool(cfg.validate_training_trajectory),
            "final_time": None if cfg.final_time is None else float(cfg.final_time),
            "trajectory_cache": None if cfg.trajectory_cache is None else str(cfg.trajectory_cache),
            "reuse_trajectory_cache": bool(cfg.reuse_trajectory_cache),
            "simulated_time": float(n_total_steps) * float(cfg.dt),
            "v_in": float(cfg.v_in),
            "t_ramp": float(cfg.t_ramp),
        },
        "mesh": trajectory_ctx["mesh_meta"],
        "offline": {
            "trajectory_elapsed_s": float(trajectory["elapsed_s"]),
            "snapshot_count": int(train_snapshots.shape[1]),
            "training_step_scales": training_step_scale_metadata,
            "modes": int(mode_count),
            "basis": basis_metadata,
            "target": target,
            "requested_target": requested_target,
            "target_objective": target_objective,
            "requested_sample_rows": int(cfg.sample_rows),
            "min_rows_per_block": int(cfg.min_rows_per_block),
            "sample_interface_band_only": bool(cfg.sample_interface_band_only),
            "row_weight_max": float(cfg.row_weight_max),
            "singular_values": singular_values,
            "cross_validation": cv_rows,
            "sampling": sampling_metadata,
            "qdeim": sampling_metadata if target == "sampled_gnat" else {},
            "reference_predictor": None
            if learned_reference_predictor is None
            else {
                "kind": learned_reference_predictor.kind,
                "degree": int(learned_reference_predictor.degree),
                "ridge": float(learned_reference_predictor.ridge),
                "training_error_max": float(learned_reference_predictor.training_error_max),
                "training_error_mean": float(learned_reference_predictor.training_error_mean),
            },
            "mandatory_interface_element_ids": interface_element_ids,
            "sampled_element_count": int(sampled_element_ids.size),
            "sampled_element_ids": sampled_element_ids,
            "total_element_count": int(native_ctx["mesh"].n_elements),
            "native_kernel_setup_s": float(native_setup_s),
            "previous_tangent_kernel_setup_s": float(previous_tangent_setup_s),
            "dwr_previous_state_tangents": bool(dwr_previous_jacobians is not None),
            "audit_full_kernel_setup_s": float(audit_full_setup_s),
            "coefficient_arg_names": coeff_args,
            "initial_bound_violation_before_projection": initial_bound_violation_before_projection,
            "initial_bound_violation_after_projection": initial_bound_violation_after_projection,
            "previous_state_kernel_arg_names": tuple(sorted(updated_previous_arg_names)),
            "dirichlet_lift_kernel_arg_names": tuple(sorted(dirichlet_lift_arg_names)),
            "artifact": str(artifact_path),
            "artifact_size_bytes": int(artifact_path.stat().st_size),
        },
        "full_order": {
            "heldout_elapsed_s": fom_heldout_s,
            "heldout_step_elapsed_s": fom_heldout_s,
            "heldout_step_times_s": heldout_step_times,
            "validation_elapsed_s": fom_heldout_s,
            "validation_step_times_s": heldout_step_times,
            "trajectory_elapsed_s": float(trajectory["elapsed_s"]),
            "trajectory_wall_s": float(trajectory["wall_s"]),
            "accepted_steps": int(trajectory["accepted_steps"]),
            "step_times_s": step_times,
            "active_dofs": int(np.asarray(getattr(trajectory_ctx["solver"], "active_dofs", np.arange(int(dh.total_dofs))), dtype=int).size),
            "total_dofs": int(dh.total_dofs),
        },
        "fom_branch_certificate": fom_branch_certificate,
        "native_reduced": {
            "online_s": float(native_online_s),
            "step_online_times_s": native_online_times,
            "trajectory": native_step_summaries,
            "backend": native.backend,
            "converged": bool(native.converged),
            "iterations": int(native.iterations),
            "residual_norm": float(native.residual_norm),
            "optimality_norm": float(getattr(native, "optimality_norm", float("nan"))),
            "stationary": bool(native_stationary),
            "status_ok": bool(native_status_ok),
            "last_step_norm": float(last_step_norm),
            "normalized_last_step": float(normalized_last_step),
            "residual_norm_history": native.residual_norm_history,
            "optimality_norm_history": getattr(native, "optimality_norm_history", np.asarray((), dtype=float)),
            "step_norm_history": native.step_norm_history,
            "line_search_alpha_history": native.line_search_alpha_history,
            "damping_history": native.damping_history,
            "rejected_step_count": int(native.rejected_step_count),
            "linear_solver": native.linear_solver,
            "line_search": bool(cfg.native_line_search),
            "damping": float(cfg.native_damping),
            "max_step_norm": None if effective_max_step_norm is None else float(effective_max_step_norm),
            "branch_radius": None if effective_branch_radius is None else float(effective_branch_radius),
            "branch_backtracking": bool(cfg.native_branch_backtracking),
            "min_substeps": int(cfg.native_min_substeps),
            "max_substeps": int(cfg.native_max_substeps),
            "substep_factor": int(cfg.native_substep_factor),
            "online_predictor": str(cfg.online_predictor),
            "reference_predictor": None
            if learned_reference_predictor is None
            else {
                "kind": learned_reference_predictor.kind,
                "degree": int(learned_reference_predictor.degree),
                "ridge": float(learned_reference_predictor.ridge),
                "training_error_max": float(learned_reference_predictor.training_error_max),
                "training_error_mean": float(learned_reference_predictor.training_error_mean),
            },
            "branch_radius_schedule": [
                None if value is None else float(value)
                for value in branch_radius_schedule
            ],
            "max_step_norm_schedule": [
                None if value is None else float(value)
                for value in max_step_norm_schedule
            ],
            "requested_max_step_norm": None
            if cfg.native_max_step_norm is None
            else float(cfg.native_max_step_norm),
            "requested_branch_radius": None
            if cfg.native_branch_radius is None
            else float(cfg.native_branch_radius),
            "training_step_scales": training_step_scale_metadata,
            "reference_weight": float(cfg.native_reference_weight),
            "state_merit_weight": float(cfg.native_state_merit_weight),
            "require_residual_convergence": bool(cfg.native_require_residual_convergence),
            "feasibility_tol": float(cfg.native_feasibility_tol),
            "adaptive_damping": bool(cfg.native_adaptive_damping),
            "constraint_method": constraint_method,
            "timing_counters": native.timing_counters,
            "dt_kernel_arg_names": tuple(sorted(native_dt_arg_names)),
        },
        "speedup": {
            "fom_online_s": fom_heldout_s,
            "fom_median_step_s": fom_median_s,
            "fom_mean_step_s": fom_mean_s,
            "native_online_s": float(native_online_s),
            "native_median_step_s": float(np.median(np.asarray(native_online_times, dtype=float))),
            "native_mean_step_s": float(np.mean(np.asarray(native_online_times, dtype=float))),
            "validation_passed": bool(passed),
            "predictive_validation_passed": bool(predictive_validation_passed),
            "replay_validation": bool(replay_validation),
            "factor": speedup,
            "median_factor": median_speedup,
            "mean_factor": mean_speedup,
            "validated_factor": validated_speedup,
            "validated_median_factor": validated_median_speedup,
            "validated_mean_factor": validated_mean_speedup,
            "predictive_validated_factor": speedup if predictive_validation_passed else None,
            "predictive_validated_median_factor": median_speedup if predictive_validation_passed else None,
            "predictive_validated_mean_factor": mean_speedup if predictive_validation_passed else None,
        },
        "errors": {
            "relative_state_vs_fom": relative_state_error,
            "projection_relative_state_vs_fom": projection_state_error,
            "final_relative_state_vs_fom": float(relative_error_history[-1]),
            "final_projection_relative_state_vs_fom": float(projection_error_history[-1]),
            "relative_state_history": relative_error_history,
            "projection_relative_state_history": projection_error_history,
            "relative_error_tolerance": float(cfg.error_tolerance),
            "projection_error_tolerance": float(cfg.projection_error_tolerance),
            "bound_violation_tolerance": float(cfg.bound_violation_tolerance),
            "per_field_vs_fom": _relative_errors_by_field(dh, rom_vector, fom_vector),
            "projection_per_field_vs_fom": _relative_errors_by_field(dh, projection_vector, fom_vector),
        },
        "bounds": {
            "rom": rom_bounds,
            "fom": _bound_stats(dh, fom_vector),
            "rom_max_violation_over_heldout": max_bound_violation,
            "rom_violation_history": bound_violation_history,
        },
        "target_audit": {
            "enabled": bool(cfg.audit_targets),
            "steps": target_audit_history,
            "max_projected_fom_full_row_residual": float(np.max(audit_projection_full_row_norms))
            if audit_projection_full_row_norms
            else None,
            "max_projected_fom_sampled_gnat_residual": float(np.max(audit_projection_sampled_gnat_norms))
            if audit_projection_sampled_gnat_norms
            else None,
            "max_rom_full_row_residual": float(np.max(audit_rom_full_row_norms))
            if audit_rom_full_row_norms
            else None,
            "max_rom_sampled_gnat_residual": float(np.max(audit_rom_sampled_gnat_norms))
            if audit_rom_sampled_gnat_norms
            else None,
        },
        "dwr": {
            "qoi_name": "alpha_mass",
            "aggregation": "final",
            "rom_qoi": float(rom_alpha_mass),
            "fom_qoi": float(fom_alpha_mass),
            "reference_qoi_error": float(alpha_qoi_error),
            "full_gradient_norm": float(np.linalg.norm(alpha_gradient_full)),
            "reduced_gradient_norm": float(np.linalg.norm(alpha_gradient_reduced)),
            "trajectory": str(dwr_trajectory_path),
            "trajectory_size_bytes": int(dwr_trajectory_path.stat().st_size),
            "certificate": dwr_certificate.to_dict(),
            "cpp_backend_fallback_error": dwr_backend_fallback,
        },
        "fom_final_metrics": trajectory["step_metrics"][-1] if trajectory["step_metrics"] else {},
    }
    summary_path = outdir / "seboldt_three_constituent_mor_summary.json"
    summary_path.write_text(json.dumps(_json_finite(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SeboldtMORResult(passed=passed, summary=summary, outdir=outdir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("out/seboldt_three_constituent_mor"))
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--poly-order", type=int, default=1)
    parser.add_argument("--pressure-order", type=int, default=1)
    parser.add_argument("--scalar-order", type=int, default=1)
    parser.add_argument("--interface-cells", type=float, default=7.0)
    parser.add_argument("--min-interface-cells", type=float, default=6.0)
    parser.add_argument("--eps-alpha", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=5.0e-5)
    parser.add_argument("--final-time", type=float, default=None)
    parser.add_argument("--trajectory-cache", type=Path, default=None)
    parser.add_argument("--reuse-trajectory-cache", dest="reuse_trajectory_cache", action="store_true")
    parser.add_argument("--no-reuse-trajectory-cache", dest="reuse_trajectory_cache", action="store_false")
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--heldout-steps", type=int, default=1)
    parser.add_argument("--validate-training-trajectory", action="store_true")
    parser.add_argument("--v-in", type=float, default=0.05)
    parser.add_argument("--t-ramp", type=float, default=0.0)
    parser.add_argument("--phi-b", type=float, default=0.18)
    parser.add_argument("--qdeg", type=int, default=3)
    parser.add_argument("--max-modes", type=int, default=10)
    parser.add_argument("--basis-strategy", default="fieldwise", choices=("fieldwise", "global"))
    parser.add_argument("--field-max-modes", type=int, default=None)
    parser.add_argument("--sample-rows", type=int, default=64)
    parser.add_argument("--min-rows-per-block", type=int, default=4)
    parser.add_argument("--sample-interface-band-only", dest="sample_interface_band_only", action="store_true")
    parser.add_argument("--sample-all-candidate-rows", dest="sample_interface_band_only", action="store_false")
    parser.add_argument(
        "--target",
        default="sampled_gnat",
        choices=tuple(sorted(_TARGET_ALIASES)),
        help="Native reduced target: true Galerkin, full-row LSPG, sampled LSPG, or sampled GNAT/QDEIM.",
    )
    parser.add_argument("--row-weight-max", type=float, default=1.0)
    parser.add_argument("--cv-tolerance", type=float, default=5.0e-2)
    parser.add_argument("--newton-tol", type=float, default=1.0e-7)
    parser.add_argument("--max-newton-iter", type=int, default=10)
    parser.add_argument("--native-residual-tol", type=float, default=1.0e-2)
    parser.add_argument(
        "--native-optimality-tol",
        type=float,
        default=None,
        help="Absolute ||J^T r|| convergence gate for LSPG/GNAT; defaults to --native-residual-tol.",
    )
    parser.add_argument("--native-step-tol", type=float, default=1.0e-8)
    parser.add_argument("--native-feasibility-tol", type=float, default=1.0e-8)
    parser.add_argument("--native-damping", type=float, default=0.0)
    parser.add_argument("--native-max-step-norm", type=float, default=None)
    parser.add_argument("--native-max-step-norm-factor", type=float, default=None)
    parser.add_argument("--native-branch-radius", type=float, default=None)
    parser.add_argument("--native-branch-radius-factor", type=float, default=None)
    parser.add_argument("--native-branch-backtracking", dest="native_branch_backtracking", action="store_true")
    parser.add_argument("--no-native-branch-backtracking", dest="native_branch_backtracking", action="store_false")
    parser.add_argument("--native-branch-radius-schedule", type=str, default=None)
    parser.add_argument("--native-max-step-norm-schedule", type=str, default=None)
    parser.add_argument("--native-reference-weight", type=float, default=0.0)
    parser.add_argument("--native-state-merit-weight", type=float, default=0.0)
    parser.add_argument("--native-require-residual-convergence", action="store_true")
    parser.add_argument("--native-min-substeps", type=int, default=1)
    parser.add_argument("--native-max-substeps", type=int, default=1)
    parser.add_argument("--native-substep-factor", type=int, default=2)
    parser.add_argument(
        "--online-predictor",
        default="previous",
        choices=("previous", "linear", "trajectory", "time_regression"),
    )
    parser.add_argument("--reference-predictor-degree", type=int, default=24)
    parser.add_argument("--reference-predictor-ridge", type=float, default=1.0e-12)
    parser.add_argument("--audit-targets", dest="audit_targets", action="store_true")
    parser.add_argument("--no-audit-targets", dest="audit_targets", action="store_false")
    parser.add_argument("--dwr-previous-state-tangents", dest="dwr_previous_state_tangents", action="store_true")
    parser.add_argument("--no-dwr-previous-state-tangents", dest="dwr_previous_state_tangents", action="store_false")
    parser.add_argument("--bound-violation-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--error-tolerance", type=float, default=1.5e-1)
    parser.add_argument("--projection-error-tolerance", type=float, default=5.0e-2)
    parser.add_argument("--max-native-iter", type=int, default=12)
    parser.add_argument("--native-line-search", dest="native_line_search", action="store_true")
    parser.add_argument("--no-native-line-search", dest="native_line_search", action="store_false")
    parser.add_argument("--native-adaptive-damping", dest="native_adaptive_damping", action="store_true")
    parser.add_argument("--no-native-adaptive-damping", dest="native_adaptive_damping", action="store_false")
    parser.set_defaults(
        native_line_search=False,
        native_adaptive_damping=False,
        sample_interface_band_only=False,
        native_branch_backtracking=False,
        audit_targets=False,
        dwr_previous_state_tangents=False,
        reuse_trajectory_cache=False,
    )
    parser.add_argument("--constraint-method", default="ipm", choices=("none", "pdas", "ipm"))
    parser.add_argument("--native-accept-failed-steps", action="store_true")
    parser.add_argument("--backend", default="cpp")
    parser.add_argument("--linear-backend", default="scipy")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    heldout_steps = int(args.heldout_steps)
    train_steps = 10 if args.train_steps is None else int(args.train_steps)
    if args.final_time is not None:
        if float(args.final_time) <= 0.0:
            raise ValueError("--final-time must be positive.")
        if float(args.dt) <= 0.0:
            raise ValueError("--dt must be positive when --final-time is used.")
        total_steps = max(2, int(math.ceil(float(args.final_time) / float(args.dt))))
        if bool(args.validate_training_trajectory) and args.train_steps is None:
            train_steps = total_steps
            heldout_steps = 0
        elif args.train_steps is None:
            train_steps = max(1, total_steps - heldout_steps)
        elif train_steps + heldout_steps < total_steps:
            heldout_steps = max(1, total_steps - train_steps)
    if bool(args.validate_training_trajectory):
        heldout_steps = 0
    cfg = SeboldtMORConfig(
        outdir=args.outdir,
        nx=args.nx,
        ny=args.ny,
        poly_order=args.poly_order,
        pressure_order=args.pressure_order,
        scalar_order=args.scalar_order,
        interface_cells=args.interface_cells,
        min_interface_cells=args.min_interface_cells,
        eps_alpha=args.eps_alpha,
        dt=args.dt,
        train_steps=train_steps,
        heldout_steps=heldout_steps,
        final_time=args.final_time,
        trajectory_cache=args.trajectory_cache,
        reuse_trajectory_cache=bool(args.reuse_trajectory_cache),
        validate_training_trajectory=bool(args.validate_training_trajectory),
        v_in=args.v_in,
        t_ramp=args.t_ramp,
        phi_b=args.phi_b,
        qdeg=args.qdeg,
        max_modes=args.max_modes,
        basis_strategy=args.basis_strategy,
        field_max_modes=args.field_max_modes,
        sample_rows=args.sample_rows,
        min_rows_per_block=args.min_rows_per_block,
        sample_interface_band_only=bool(args.sample_interface_band_only),
        target=args.target,
        row_weight_max=args.row_weight_max,
        cv_tolerance=args.cv_tolerance,
        newton_tol=args.newton_tol,
        max_newton_iter=args.max_newton_iter,
        native_residual_tol=args.native_residual_tol,
        native_optimality_tol=args.native_optimality_tol,
        native_step_tol=args.native_step_tol,
        native_feasibility_tol=args.native_feasibility_tol,
        native_damping=args.native_damping,
        native_max_step_norm=args.native_max_step_norm,
        native_max_step_norm_factor=args.native_max_step_norm_factor,
        native_branch_radius=args.native_branch_radius,
        native_branch_radius_factor=args.native_branch_radius_factor,
        native_branch_backtracking=bool(args.native_branch_backtracking),
        native_branch_radius_schedule=args.native_branch_radius_schedule,
        native_max_step_norm_schedule=args.native_max_step_norm_schedule,
        native_reference_weight=args.native_reference_weight,
        native_state_merit_weight=args.native_state_merit_weight,
        native_require_residual_convergence=bool(args.native_require_residual_convergence),
        native_min_substeps=args.native_min_substeps,
        native_max_substeps=args.native_max_substeps,
        native_substep_factor=args.native_substep_factor,
        online_predictor=args.online_predictor,
        reference_predictor_degree=args.reference_predictor_degree,
        reference_predictor_ridge=args.reference_predictor_ridge,
        audit_targets=bool(args.audit_targets),
        dwr_previous_state_tangents=bool(args.dwr_previous_state_tangents),
        bound_violation_tolerance=args.bound_violation_tolerance,
        error_tolerance=args.error_tolerance,
        projection_error_tolerance=args.projection_error_tolerance,
        max_native_iter=args.max_native_iter,
        native_line_search=bool(args.native_line_search),
        native_adaptive_damping=bool(args.native_adaptive_damping),
        constraint_method=args.constraint_method,
        native_accept_failed_steps=bool(args.native_accept_failed_steps),
        backend=args.backend,
        linear_backend=args.linear_backend,
    )
    result = run_seboldt_mor_validation(cfg)
    print(json.dumps(_json_finite(result.summary), indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
