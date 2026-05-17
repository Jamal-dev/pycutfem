#!/usr/bin/env python3
"""Native MOR validation for the nine-field three-constituent model.

The driver uses the canonical one-domain FSPI/FPSI form, collects snapshots for
all nine logical fields, builds a cross-validated POD/QDEIM reduced model, and
solves the held-out nonlinear step with the C++ native bound-constrained online
loop.  Python is used for setup/offline work only.
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

from examples.biofilms.benchmarks.three_constituent.paper1_benchmark1_mms import (
    _as_analytic,
    _bc_scalar,
    _build_manufactured_problem,
    _tag_unit_square_boundaries,
)
from examples.biofilms.benchmarks.three_constituent.seboldt_physical import (
    _make_homogeneous_bcs,
    _make_spaces,
    _make_state,
    _make_trial_test,
)
from examples.utils.biofilm.three_constituent_one_domain import (
    build_three_constituent_one_domain_forms,
    build_three_constituent_pdas_solver,
)
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.mor import (
    BoundConstraintSpec,
    NativeAdjointDWRSpec,
    NativeGnatTargetSpec,
    NativeKernelReference,
    NativeReducedArtifact,
    bound_constraints_from_fields,
    build_qdeim_interpolation_rule,
    load_native_reduced_artifact,
    native_kernel_metadata_from_runner,
    solve_native_bound_constrained_deim_online_gauss_newton,
)
from pycutfem.mor.mixed_reduction import build_block_row_weights
from pycutfem.mor.pod import fit_pod
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, TimeStepperParameters, VIParameters
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Function
from pycutfem.ufl.forms import BoundaryCondition, Integral
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


FIELD_SPECS = {
    "vf_x": 2,
    "vf_y": 2,
    "pf": 1,
    "vp_x": 2,
    "vp_y": 2,
    "pp": 1,
    "vs_x": 2,
    "vs_y": 2,
    "us_x": 2,
    "us_y": 2,
    "alpha": 1,
    "phi": 1,
    "Gamma": 1,
}

LOGICAL_FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "v_f": ("vf_x", "vf_y"),
    "p_f": ("pf",),
    "v_p": ("vp_x", "vp_y"),
    "p_p": ("pp",),
    "v_s": ("vs_x", "vs_y"),
    "u_s": ("us_x", "us_y"),
    "alpha": ("alpha",),
    "phi": ("phi",),
    "Gamma": ("Gamma",),
}

CURRENT_STATE_KEYS = (
    "v_f_k",
    "p_f_k",
    "v_p_k",
    "p_p_k",
    "v_s_k",
    "u_s_k",
    "alpha_k",
    "phi_k",
    "Gamma_k",
)

CURRENT_COEFFICIENT_STEMS = CURRENT_STATE_KEYS


@dataclass(frozen=True)
class ThreeConstituentMORResult:
    passed: bool
    summary: dict[str, Any]
    outdir: Path


def _json_finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_finite(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _field_rows(dh: DofHandler, fields: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([np.asarray(dh.get_field_slice(field), dtype=np.int64).reshape(-1) for field in fields])


def _field_blocks(dh: DofHandler, *, row_subset: np.ndarray | None = None) -> list[tuple[np.ndarray, str]]:
    if row_subset is None:
        return [(_field_rows(dh, fields), name) for name, fields in LOGICAL_FIELD_GROUPS.items()]
    subset = np.asarray(row_subset, dtype=np.int64).reshape(-1)
    lookup = {int(gdof): idx for idx, gdof in enumerate(subset)}
    blocks: list[tuple[np.ndarray, str]] = []
    for name, fields in LOGICAL_FIELD_GROUPS.items():
        rows = [lookup[int(row)] for row in _field_rows(dh, fields) if int(row) in lookup]
        if rows:
            blocks.append((np.asarray(rows, dtype=np.int64), name))
    return blocks


def _current_field_objects(state: dict[str, object]) -> dict[str, object]:
    return {
        "vf_x": state["v_f_k"],
        "vf_y": state["v_f_k"],
        "pf": state["p_f_k"],
        "vp_x": state["v_p_k"],
        "vp_y": state["v_p_k"],
        "pp": state["p_p_k"],
        "vs_x": state["v_s_k"],
        "vs_y": state["v_s_k"],
        "us_x": state["u_s_k"],
        "us_y": state["u_s_k"],
        "alpha": state["alpha_k"],
        "phi": state["phi_k"],
        "Gamma": state["Gamma_k"],
    }


def _previous_field_objects(state: dict[str, object], p_f_n: Function, p_p_n: Function) -> dict[str, object]:
    return {
        "vf_x": state["v_f_n"],
        "vf_y": state["v_f_n"],
        "pf": p_f_n,
        "vp_x": state["v_p_n"],
        "vp_y": state["v_p_n"],
        "pp": p_p_n,
        "vs_x": state["v_s_n"],
        "vs_y": state["v_s_n"],
        "us_x": state["u_s_n"],
        "us_y": state["u_s_n"],
        "alpha": state["alpha_n"],
        "phi": state["phi_n"],
        "Gamma": state["Gamma_n"],
    }


def _state_vector(dh: DofHandler, field_objects: dict[str, object]) -> np.ndarray:
    values = np.zeros(int(dh.total_dofs), dtype=np.float64)
    for field in FIELD_SPECS:
        rows = np.asarray(dh.get_field_slice(field), dtype=np.int64)
        values[rows] = np.asarray(field_objects[field].get_nodal_values(rows), dtype=float)
    return values


def _set_exact_current_state(state: dict[str, object], callables: dict[str, Any]) -> None:
    state["v_f_k"].set_values_from_function(callables["v_f"])
    state["v_p_k"].set_values_from_function(callables["v_p"])
    state["v_s_k"].set_values_from_function(callables["v_s"])
    state["u_s_k"].set_values_from_function(callables["u_s"])
    state["p_f_k"].set_values_from_function(callables["p_f"])
    state["p_p_k"].set_values_from_function(callables["p_p"])
    state["alpha_k"].set_values_from_function(callables["alpha"])
    state["phi_k"].set_values_from_function(callables["phi"])
    state["Gamma_k"].set_values_from_function(callables["Gamma"])


def _set_exact_previous_state(state: dict[str, object], p_f_n: Function, p_p_n: Function, callables: dict[str, Any]) -> None:
    state["v_f_n"].set_values_from_function(callables["v_f"])
    state["v_p_n"].set_values_from_function(callables["v_p"])
    state["v_s_n"].set_values_from_function(callables["v_s"])
    state["u_s_n"].set_values_from_function(callables["u_s"])
    state["alpha_n"].set_values_from_function(callables["alpha"])
    state["phi_n"].set_values_from_function(callables["phi"])
    state["Gamma_n"].set_values_from_function(callables["Gamma"])
    p_f_n.set_values_from_function(callables["p_f"])
    p_p_n.set_values_from_function(callables["p_p"])


def _make_boundary_conditions(exact: dict[str, Any]) -> list[BoundaryCondition]:
    c = exact["callables"]
    bcs: list[BoundaryCondition] = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("vf_x", "dirichlet", tag, _bc_scalar(lambda x, y, f=c["v_f"]: f(x, y)[..., 0])),
                BoundaryCondition("vf_y", "dirichlet", tag, _bc_scalar(lambda x, y, f=c["v_f"]: f(x, y)[..., 1])),
                BoundaryCondition("pf", "dirichlet", tag, _bc_scalar(c["p_f"])),
                BoundaryCondition("vp_x", "dirichlet", tag, _bc_scalar(lambda x, y, f=c["v_p"]: f(x, y)[..., 0])),
                BoundaryCondition("vp_y", "dirichlet", tag, _bc_scalar(lambda x, y, f=c["v_p"]: f(x, y)[..., 1])),
                BoundaryCondition("pp", "dirichlet", tag, _bc_scalar(c["p_p"])),
                BoundaryCondition("vs_x", "dirichlet", tag, _bc_scalar(lambda x, y, f=c["v_s"]: f(x, y)[..., 0])),
                BoundaryCondition("vs_y", "dirichlet", tag, _bc_scalar(lambda x, y, f=c["v_s"]: f(x, y)[..., 1])),
                BoundaryCondition("us_x", "dirichlet", tag, _bc_scalar(lambda x, y, f=c["u_s"]: f(x, y)[..., 0])),
                BoundaryCondition("us_y", "dirichlet", tag, _bc_scalar(lambda x, y, f=c["u_s"]: f(x, y)[..., 1])),
                BoundaryCondition("alpha", "dirichlet", tag, _bc_scalar(c["alpha"])),
                BoundaryCondition("phi", "dirichlet", tag, _bc_scalar(c["phi"])),
                BoundaryCondition("Gamma", "dirichlet", tag, _bc_scalar(c["Gamma"])),
            ]
        )
    return bcs


def _build_context(
    *,
    nx: int,
    dt: float,
    qdeg: int,
    exact: dict[str, Any],
    mesh_order: int,
) -> dict[str, Any]:
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=int(nx), ny=int(nx), poly_order=int(mesh_order))
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=int(mesh_order))
    _tag_unit_square_boundaries(mesh)
    me = MixedElement(mesh, field_specs=FIELD_SPECS)
    dh = DofHandler(me, method="cg")
    spaces = _make_spaces(dh)
    trial, test = _make_trial_test(dh, spaces)
    state = _make_state(dh)
    p_f_n = Function("p_f_n", "pf", dof_handler=dh)
    p_p_n = Function("p_p_n", "pp", dof_handler=dh)

    _set_exact_current_state(state, exact["callables"])
    _set_exact_previous_state(state, p_f_n, p_p_n, exact["callables"]["n"])

    sources = {key: _as_analytic(fn) for key, fn in exact["source_callables"].items()}
    forms = build_three_constituent_one_domain_forms(
        **state,
        **trial,
        **test,
        dx=dx(metadata={"q": int(qdeg)}),
        **exact["params"],
        **sources,
    )
    bcs = _make_boundary_conditions(exact)
    return {
        "mesh": mesh,
        "mixed_element": me,
        "dof_handler": dh,
        "state": state,
        "p_f_n": p_f_n,
        "p_p_n": p_p_n,
        "forms": forms,
        "bcs": bcs,
        "exact": exact,
        "dt": float(dt),
        "qdeg": int(qdeg),
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


def _run_full_order_solve(
    *,
    nx: int,
    dt: float,
    qdeg: int,
    exact: dict[str, Any],
    mesh_order: int,
    backend: str,
    linear_backend: str,
    newton_tol: float,
    max_newton_iter: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ctx = _build_context(nx=nx, dt=dt, qdeg=qdeg, exact=exact, mesh_order=mesh_order)
    dh = ctx["dof_handler"]
    state = ctx["state"]
    exact_vector = _state_vector(dh, _current_field_objects(state))
    solver = build_three_constituent_pdas_solver(
        ctx["forms"],
        dof_handler=dh,
        mixed_element=ctx["mixed_element"],
        bcs=ctx["bcs"],
        bcs_homog=_make_homogeneous_bcs(ctx["bcs"]),
        newton_params=NewtonParameters(
            newton_tol=float(newton_tol),
            newton_rtol=0.0,
            max_newton_iter=int(max_newton_iter),
            print_level=0,
            line_search=True,
            ls_max_iter=12,
        ),
        vi_params=VIParameters(c=1.0, project_initial_guess=True, active_set_persistence=1),
        lin_params=LinearSolverParameters(backend=str(linear_backend), tol=1.0e-11, maxit=10000),
        backend=str(backend),
        quad_order=int(qdeg),
        alpha_bounds=(0.0, 1.0),
        phi_bounds=(0.0, 1.0),
    )
    t0 = time.perf_counter()
    _, n_steps, elapsed = solver.solve_time_interval(
        functions=_current_functions(state),
        prev_functions=_previous_functions(state, ctx["p_f_n"], ctx["p_p_n"]),
        time_params=TimeStepperParameters(dt=float(dt), final_time=float(dt), max_steps=1, stop_on_steady=False),
    )
    wall = time.perf_counter() - t0
    timing = {
        "elapsed_s": float(elapsed if elapsed is not None else wall),
        "wall_s": float(wall),
        "n_steps": int(n_steps),
        "total_dofs": int(dh.total_dofs),
        "active_dofs": int(np.asarray(getattr(solver, "active_dofs", np.arange(int(dh.total_dofs))), dtype=int).size),
    }
    vector = _state_vector(dh, _current_field_objects(state))
    return ctx, {"vector": vector, "exact_vector": exact_vector, "timing": timing}


def _collect_snapshot_matrix(
    *,
    nx: int,
    train_dts: tuple[float, ...],
    qdeg: int,
    gamma_delta_epsilon: float,
    mesh_order: int,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    snapshots: list[np.ndarray] = []
    dh0: DofHandler | None = None
    previous_snapshot: np.ndarray | None = None
    for dt in train_dts:
        exact = _build_manufactured_problem(dt=float(dt), gamma_delta_epsilon=float(gamma_delta_epsilon))
        ctx = _build_context(nx=nx, dt=float(dt), qdeg=qdeg, exact=exact, mesh_order=mesh_order)
        dh = ctx["dof_handler"]
        if dh0 is None:
            dh0 = dh
            previous_snapshot = _state_vector(dh, _previous_field_objects(ctx["state"], ctx["p_f_n"], ctx["p_p_n"]))
        snapshots.append(_state_vector(dh, _current_field_objects(ctx["state"])))
    if dh0 is None or previous_snapshot is None:
        raise ValueError("at least one training dt is required.")
    matrix = np.column_stack([previous_snapshot, *snapshots])
    return matrix, {"dof_handler": dh0, "train_dts": tuple(float(v) for v in train_dts)}, previous_snapshot


def _cross_validated_mode_count(
    snapshots: np.ndarray,
    *,
    max_modes: int | None,
    tolerance: float,
) -> tuple[int, list[dict[str, float]]]:
    n_snapshots = int(snapshots.shape[1])
    max_rank = max(1, n_snapshots - 1)
    if max_modes is not None:
        max_rank = min(max_rank, int(max_modes))
    rows: list[dict[str, float]] = []
    snapshot_norm_scale = max(float(np.max(np.linalg.norm(snapshots, axis=0))), 1.0e-14)
    for modes in range(1, max_rank + 1):
        errors: list[float] = []
        for heldout in range(n_snapshots):
            train = np.delete(snapshots, heldout, axis=1)
            local_modes = min(modes, max(1, train.shape[1] - 1))
            pod = fit_pod(train, n_modes=local_modes, center=True)
            rhs = snapshots[:, heldout] - np.asarray(pod.mean, dtype=float).reshape(-1)
            coeffs, *_ = np.linalg.lstsq(np.asarray(pod.basis, dtype=float), rhs, rcond=None)
            recon = np.asarray(pod.mean, dtype=float).reshape(-1) + np.asarray(pod.basis, dtype=float) @ coeffs
            denom = max(float(np.linalg.norm(snapshots[:, heldout])), 1.0e-10 * snapshot_norm_scale, 1.0e-14)
            errors.append(float(np.linalg.norm(recon - snapshots[:, heldout]) / denom))
        rows.append({"modes": float(modes), "mean_relative_error": float(np.mean(errors)), "max_relative_error": float(np.max(errors))})
    best = min(rows, key=lambda item: (item["mean_relative_error"], item["modes"]))
    chosen = int(best["modes"])
    for row in rows:
        if row["mean_relative_error"] <= float(tolerance):
            chosen = int(row["modes"])
            break
    return chosen, rows


def _fit_trial_basis(
    snapshots: np.ndarray,
    *,
    n_modes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pod = fit_pod(snapshots, n_modes=int(n_modes), center=True)
    basis = np.ascontiguousarray(np.asarray(pod.basis, dtype=float))
    offset = np.ascontiguousarray(np.asarray(pod.mean, dtype=float).reshape(-1))
    singular_values = np.ascontiguousarray(np.asarray(pod.singular_values, dtype=float).reshape(-1))
    return basis, offset, singular_values


def _dirichlet_free_rows(dh: DofHandler, bcs: list[BoundaryCondition]) -> tuple[np.ndarray, np.ndarray]:
    fixed = np.asarray(sorted(int(row) for row in dh.get_dirichlet_data(bcs).keys()), dtype=np.int64)
    all_rows = np.arange(int(dh.total_dofs), dtype=np.int64)
    free = np.setdiff1d(all_rows, fixed, assume_unique=False)
    return free, fixed


def _build_qdeim_target(
    dh: DofHandler,
    bcs: list[BoundaryCondition],
    trial_basis: np.ndarray,
    snapshot_matrix: np.ndarray,
    *,
    rcond: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    free_rows, fixed_rows = _dirichlet_free_rows(dh, bcs)
    if free_rows.size < trial_basis.shape[1]:
        free_rows = np.arange(int(dh.total_dofs), dtype=np.int64)
    centered = snapshot_matrix - np.mean(snapshot_matrix, axis=1, keepdims=True)
    reference = centered[free_rows, :]
    row_weights = build_block_row_weights(reference, _field_blocks(dh, row_subset=free_rows), max_weight=1.0e6)
    weighted_basis = np.sqrt(row_weights)[:, None] * np.asarray(trial_basis[free_rows, :], dtype=float)
    rule = build_qdeim_interpolation_rule(weighted_basis, rcond=rcond)
    row_dofs = np.ascontiguousarray(free_rows[rule.rows], dtype=np.int64)
    sampled_weights = np.ascontiguousarray(row_weights[rule.rows], dtype=np.float64)
    residual_terms = np.ascontiguousarray(weighted_basis.T @ weighted_basis, dtype=np.float64)
    metadata = {
        "free_rows": int(free_rows.size),
        "fixed_rows": int(fixed_rows.size),
        "selected_rows": int(row_dofs.size),
        "row_weight_min": float(np.min(sampled_weights)),
        "row_weight_max": float(np.max(sampled_weights)),
        "selected_basis_condition": float(np.linalg.cond(np.asarray(rule.selected_basis, dtype=float))),
    }
    return row_dofs, np.asarray(rule.selected_basis, dtype=np.float64), residual_terms, sampled_weights, metadata


def _infer_current_coefficient_args(*param_orders: Any) -> tuple[str, ...]:
    names: list[str] = []
    for order in param_orders:
        for raw in order:
            name = str(raw)
            if not name.endswith("_loc"):
                continue
            if any(stem in name for stem in CURRENT_COEFFICIENT_STEMS):
                names.append(name)
    return tuple(dict.fromkeys(names))


def _as_single_volume_integral(form: Any) -> Any:
    integrals = list(getattr(form, "integrals", ()))
    if not integrals:
        return form
    if len(integrals) == 1:
        return integrals[0]
    measure = integrals[0].measure
    integrand = integrals[0].integrand
    for integral in integrals[1:]:
        if integral.measure.on_facet != measure.on_facet:
            raise ValueError("native online benchmark expects one volume measure family.")
        integrand = integrand + integral.integrand
    return Integral(integrand, measure)


def _support_element_ids(dh: DofHandler, row_dofs: np.ndarray) -> np.ndarray:
    selected = set(int(row) for row in np.asarray(row_dofs, dtype=np.int64).reshape(-1))
    if not selected:
        return np.arange(int(dh.mixed_element.mesh.n_elements), dtype=np.int32)
    elements: set[int] = set()
    n_elements = int(dh.mixed_element.mesh.n_elements)
    for eid in range(n_elements):
        for field in FIELD_SPECS:
            if selected.intersection(int(row) for row in dh.element_maps[field][eid]):
                elements.add(int(eid))
                break
    if not elements:
        return np.arange(n_elements, dtype=np.int32)
    return np.asarray(sorted(elements), dtype=np.int32)


def _compile_native_pair(
    ctx: dict[str, Any],
    *,
    qdeg: int,
    element_ids: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any], Any, dict[str, Any], tuple[str, ...], float]:
    if element_ids is None:
        element_ids = np.arange(int(ctx["mesh"].n_elements), dtype=np.int32)
    else:
        element_ids = np.ascontiguousarray(np.asarray(element_ids, dtype=np.int32).reshape(-1))
    compiler = FormCompiler(ctx["dof_handler"], quadrature_order=int(qdeg), backend="cpp")
    t0 = time.perf_counter()
    residual_runner, residual_funcs, residual_args, _ = compiler._prepare_volume_jit_kernel(
        _as_single_volume_integral(ctx["forms"].residual_form),
        element_ids=element_ids,
        full_local_layout=True,
    )
    tangent_runner, tangent_funcs, tangent_args, _ = compiler._prepare_volume_jit_kernel(
        _as_single_volume_integral(ctx["forms"].jacobian_form),
        element_ids=element_ids,
        full_local_layout=True,
    )
    residual_runner(residual_funcs, residual_args)
    tangent_runner(tangent_funcs, tangent_args)
    setup_s = time.perf_counter() - t0
    coefficient_args = _infer_current_coefficient_args(residual_runner.param_order, tangent_runner.param_order)
    if not coefficient_args:
        raise RuntimeError(
            "Could not infer current-state coefficient arrays from generated native kernel parameters: "
            f"{tuple(residual_runner.param_order)} / {tuple(tangent_runner.param_order)}"
        )
    return residual_runner, residual_args, tangent_runner, tangent_args, coefficient_args, setup_s


def _relative_errors_by_field(dh: DofHandler, values: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, fields in LOGICAL_FIELD_GROUPS.items():
        rows = _field_rows(dh, fields)
        denom = max(float(np.linalg.norm(reference[rows])), 1.0e-14)
        out[name] = float(np.linalg.norm(values[rows] - reference[rows]) / denom)
    return out


def _bound_stats(dh: DofHandler, values: np.ndarray) -> dict[str, float]:
    alpha = values[np.asarray(dh.get_field_slice("alpha"), dtype=np.int64)]
    phi = values[np.asarray(dh.get_field_slice("phi"), dtype=np.int64)]
    violation = max(
        0.0,
        float(np.max(-alpha)),
        float(np.max(alpha - 1.0)),
        float(np.max(-phi)),
        float(np.max(phi - 1.0)),
    )
    return {
        "alpha_min": float(np.min(alpha)),
        "alpha_max": float(np.max(alpha)),
        "phi_min": float(np.min(phi)),
        "phi_max": float(np.max(phi)),
        "max_violation": violation,
    }


def _write_artifact(
    path: Path,
    *,
    trial_basis: np.ndarray,
    offset: np.ndarray,
    residual_runner: Any,
    tangent_runner: Any,
    row_dofs: np.ndarray,
    element_ids: np.ndarray,
    row_weights: np.ndarray | None,
    selected_basis: np.ndarray | None,
    residual_terms: np.ndarray | None,
    bound_constraints: BoundConstraintSpec,
    coefficient_arg_names: tuple[str, ...],
    solver_options: dict[str, Any],
    metadata: dict[str, Any],
    adjoint_dwr: NativeAdjointDWRSpec | None = None,
    reference_policy: Any | None = None,
    target_objective: str = "qdeim",
) -> None:
    artifact = NativeReducedArtifact(
        problem_id="three_constituent_one_domain_mor",
        trial_basis=trial_basis,
        offset=offset,
        residual_kernel=NativeKernelReference(
            kernel_id="three_constituent_residual",
            abi="native-kernel-v1",
            param_order=tuple(residual_runner.param_order),
        ),
        tangent_kernel=NativeKernelReference(
            kernel_id="three_constituent_tangent",
            abi="native-kernel-v1",
            param_order=tuple(tangent_runner.param_order),
        ),
        target=NativeGnatTargetSpec(
            row_dofs=row_dofs,
            element_ids=element_ids,
            row_weights=row_weights,
            selected_basis=selected_basis,
            residual_terms=residual_terms,
            objective=str(target_objective),
            metadata={"logical_fields": tuple(LOGICAL_FIELD_GROUPS)},
        ),
        bound_constraints=bound_constraints,
        adjoint_dwr=adjoint_dwr,
        reference_policy=reference_policy,
        solver_options=solver_options,
        metadata={**metadata, "coefficient_arg_names": coefficient_arg_names},
    )
    artifact.save(path)
    loaded = load_native_reduced_artifact(path)
    if loaded.trial_basis.shape != artifact.trial_basis.shape:
        raise RuntimeError("native MOR artifact round trip changed the trial-basis shape.")


def run_validation(
    *,
    outdir: Path,
    nx: int = 2,
    heldout_dt: float = 0.060,
    train_dts: tuple[float, ...] = (0.035, 0.045, 0.055, 0.070, 0.080),
    gamma_delta_epsilon: float = 1.0e-3,
    qdeg: int = 4,
    mesh_order: int = 2,
    max_modes: int | None = 5,
    cv_tolerance: float = 2.5e-5,
    newton_tol: float = 1.0e-9,
    max_newton_iter: int = 12,
    native_residual_tol: float = 1.0e-8,
    max_native_iter: int = 16,
    backend: str = "cpp",
    linear_backend: str = "scipy",
    warmup: bool = True,
) -> ThreeConstituentMORResult:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_dts = tuple(float(v) for v in train_dts)
    heldout_exact = _build_manufactured_problem(dt=float(heldout_dt), gamma_delta_epsilon=float(gamma_delta_epsilon))

    snapshot_t0 = time.perf_counter()
    snapshots, snapshot_info, previous_snapshot = _collect_snapshot_matrix(
        nx=int(nx),
        train_dts=train_dts,
        qdeg=int(qdeg),
        gamma_delta_epsilon=float(gamma_delta_epsilon),
        mesh_order=int(mesh_order),
    )
    snapshot_s = time.perf_counter() - snapshot_t0
    mode_count, cv_rows = _cross_validated_mode_count(snapshots, max_modes=max_modes, tolerance=float(cv_tolerance))
    trial_basis, offset, singular_values = _fit_trial_basis(snapshots, n_modes=mode_count)

    if warmup:
        _run_full_order_solve(
            nx=int(nx),
            dt=float(heldout_dt),
            qdeg=int(qdeg),
            exact=heldout_exact,
            mesh_order=int(mesh_order),
            backend=backend,
            linear_backend=linear_backend,
            newton_tol=newton_tol,
            max_newton_iter=max_newton_iter,
        )
    fom_ctx, fom = _run_full_order_solve(
        nx=int(nx),
        dt=float(heldout_dt),
        qdeg=int(qdeg),
        exact=heldout_exact,
        mesh_order=int(mesh_order),
        backend=backend,
        linear_backend=linear_backend,
        newton_tol=newton_tol,
        max_newton_iter=max_newton_iter,
    )
    dh = fom_ctx["dof_handler"]
    exact_vector = np.asarray(fom["exact_vector"], dtype=float)
    fom_vector = np.asarray(fom["vector"], dtype=float)

    row_dofs, selected_basis, residual_terms, row_weights, target_meta = _build_qdeim_target(
        dh,
        fom_ctx["bcs"],
        trial_basis,
        snapshots,
        rcond=1.0e-12,
    )
    bound_constraints = bound_constraints_from_fields(
        dh,
        {"alpha": (0.0, 1.0), "phi": (0.0, 1.0)},
        metadata={"logical_fields": ("alpha", "phi")},
    )

    native_ctx = _build_context(nx=int(nx), dt=float(heldout_dt), qdeg=int(qdeg), exact=heldout_exact, mesh_order=int(mesh_order))
    sampled_element_ids = _support_element_ids(native_ctx["dof_handler"], row_dofs)
    residual_runner, residual_args, tangent_runner, tangent_args, coeff_args, native_setup_s = _compile_native_pair(
        native_ctx,
        qdeg=int(qdeg),
        element_ids=sampled_element_ids,
    )
    initial_coefficients = np.zeros(int(trial_basis.shape[1]), dtype=np.float64)
    solver_options = {
        "constraint_method": "pdas",
        "max_iterations": int(max_native_iter),
        "residual_tol": float(native_residual_tol),
        "step_tol": 1.0e-11,
        "adaptive_damping": True,
        "line_search": True,
        "max_line_search": 10,
        "active_tol": 1.0e-10,
        "feasibility_tol": 1.0e-10,
        "rcond": 1.0e-12,
    }
    from pycutfem.mor.cpp_backend.online_gauss_newton import module as _online_gauss_newton_module

    _online_gauss_newton_module()
    native_t0 = time.perf_counter()
    native = solve_native_bound_constrained_deim_online_gauss_newton(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_param_order=residual_runner.param_order,
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_param_order=tangent_runner.param_order,
        tangent_static_args=tangent_args,
        trial_basis=trial_basis,
        offset=offset,
        initial_coefficients=initial_coefficients,
        row_dofs=row_dofs,
        selected_basis=selected_basis,
        residual_terms=residual_terms,
        bound_constraints=bound_constraints,
        coefficient_arg_names=coeff_args,
        row_weights=row_weights,
        **solver_options,
    )
    native_online_s = time.perf_counter() - native_t0
    rom_vector = offset + trial_basis @ native.coefficients
    projection_coeffs, *_ = np.linalg.lstsq(trial_basis, fom_vector - offset, rcond=None)
    projection_vector = offset + trial_basis @ projection_coeffs

    artifact_path = outdir / "three_constituent_native_mor_artifact.npz"
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
        solver_options=solver_options,
        metadata={
            "nx": int(nx),
            "heldout_dt": float(heldout_dt),
            "train_dts": train_dts,
            "mesh_order": int(mesh_order),
            "qdeg": int(qdeg),
        },
    )

    fom_online_s = float(fom["timing"]["elapsed_s"])
    speedup = fom_online_s / max(float(native_online_s), 1.0e-14)
    errors_vs_fom = _relative_errors_by_field(dh, rom_vector, fom_vector)
    errors_vs_exact = _relative_errors_by_field(dh, rom_vector, exact_vector)
    projection_errors_vs_fom = _relative_errors_by_field(dh, projection_vector, fom_vector)
    rom_bound_stats = _bound_stats(dh, rom_vector)
    fom_bound_stats = _bound_stats(dh, fom_vector)
    relative_state_error = float(np.linalg.norm(rom_vector - fom_vector) / max(np.linalg.norm(fom_vector), 1.0e-14))
    projection_state_error = float(np.linalg.norm(projection_vector - fom_vector) / max(np.linalg.norm(fom_vector), 1.0e-14))

    passed = bool(
        native.converged
        and rom_bound_stats["max_violation"] <= 1.0e-9
        and relative_state_error <= max(1.0e-2, 25.0 * projection_state_error + 1.0e-10)
    )
    summary: dict[str, Any] = {
        "passed": passed,
        "problem": {
            "name": "three_constituent_one_domain_mms",
            "logical_fields": tuple(LOGICAL_FIELD_GROUPS),
            "nx": int(nx),
            "mesh_order": int(mesh_order),
            "qdeg": int(qdeg),
            "heldout_dt": float(heldout_dt),
            "train_dts": train_dts,
            "total_dofs": int(dh.total_dofs),
            "previous_snapshot_norm": float(np.linalg.norm(previous_snapshot)),
        },
        "offline": {
            "snapshot_collection_s": float(snapshot_s),
            "snapshot_count": int(snapshots.shape[1]),
            "modes": int(mode_count),
            "singular_values": singular_values,
            "cross_validation": cv_rows,
            "qdeim": target_meta,
            "sampled_element_ids": sampled_element_ids,
            "sampled_element_count": int(sampled_element_ids.size),
            "total_element_count": int(native_ctx["mesh"].n_elements),
            "native_kernel_setup_s": float(native_setup_s),
            "coefficient_arg_names": coeff_args,
            "artifact": str(artifact_path),
            "artifact_size_bytes": int(artifact_path.stat().st_size),
        },
        "full_order": fom["timing"],
        "native_reduced": {
            "online_s": float(native_online_s),
            "backend": native.backend,
            "converged": bool(native.converged),
            "iterations": int(native.iterations),
            "residual_norm": float(native.residual_norm),
            "linear_solver": native.linear_solver,
            "timing_counters": native.timing_counters,
        },
        "speedup": {
            "fom_online_s": fom_online_s,
            "native_online_s": float(native_online_s),
            "factor": float(speedup),
        },
        "errors": {
            "relative_state_vs_fom": relative_state_error,
            "projection_relative_state_vs_fom": projection_state_error,
            "per_field_vs_fom": errors_vs_fom,
            "per_field_vs_exact": errors_vs_exact,
            "projection_per_field_vs_fom": projection_errors_vs_fom,
        },
        "bounds": {
            "rom": rom_bound_stats,
            "fom": fom_bound_stats,
        },
    }
    summary_path = outdir / "three_constituent_mor_summary.json"
    summary_path.write_text(json.dumps(_json_finite(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ThreeConstituentMORResult(passed=passed, summary=summary, outdir=outdir)


def _parse_train_dts(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("out/three_constituent_mor"))
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--heldout-dt", type=float, default=0.060)
    parser.add_argument("--train-dts", type=_parse_train_dts, default=(0.035, 0.045, 0.055, 0.070, 0.080))
    parser.add_argument("--gamma-delta-epsilon", type=float, default=1.0e-3)
    parser.add_argument("--qdeg", type=int, default=4)
    parser.add_argument("--mesh-order", type=int, default=2)
    parser.add_argument("--max-modes", type=int, default=5)
    parser.add_argument("--cv-tolerance", type=float, default=2.5e-5)
    parser.add_argument("--newton-tol", type=float, default=1.0e-9)
    parser.add_argument("--max-newton-iter", type=int, default=12)
    parser.add_argument("--native-residual-tol", type=float, default=1.0e-8)
    parser.add_argument("--max-native-iter", type=int, default=16)
    parser.add_argument("--backend", default="cpp")
    parser.add_argument("--linear-backend", default="scipy")
    parser.add_argument("--no-warmup", action="store_true", help="Skip the cache-warming FOM solve.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_validation(
        outdir=args.outdir,
        nx=args.nx,
        heldout_dt=args.heldout_dt,
        train_dts=tuple(args.train_dts),
        gamma_delta_epsilon=args.gamma_delta_epsilon,
        qdeg=args.qdeg,
        mesh_order=args.mesh_order,
        max_modes=args.max_modes,
        cv_tolerance=args.cv_tolerance,
        newton_tol=args.newton_tol,
        max_newton_iter=args.max_newton_iter,
        native_residual_tol=args.native_residual_tol,
        max_native_iter=args.max_native_iter,
        backend=args.backend,
        linear_backend=args.linear_backend,
        warmup=not args.no_warmup,
    )
    summary = _json_finite(result.summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
