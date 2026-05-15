"""Native online Gauss-Newton solve driver for generated UFL kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .native_assembly import native_kernel_metadata_from_runner
from .sparse import NativeSparseMatrix, is_sparse_matrix_like


@dataclass(frozen=True)
class NativeOnlineGaussNewtonResult:
    """Result returned by the native C++ online Gauss-Newton driver."""

    coefficients: np.ndarray
    converged: bool
    iterations: int
    residual_norm: float
    linear_solver: str
    residual_norm_history: np.ndarray
    step_norm_history: np.ndarray
    line_search_alpha_history: np.ndarray
    damping_history: np.ndarray
    rejected_step_count: int
    timing_counters: dict[str, float | int]
    backend: str


def _normalize_static_args(
    static_args: Mapping[str, Any],
    param_order: Sequence[str],
    coefficient_arg_names: Sequence[str],
) -> dict[str, Any]:
    args = dict(static_args)
    if "gdofs_map" not in args:
        raise ValueError("native online kernels require a gdofs_map static argument.")
    gdofs_map = np.asarray(args["gdofs_map"], dtype=np.int32)
    if gdofs_map.ndim != 2:
        raise ValueError("gdofs_map must be a rank-2 array.")
    args["gdofs_map"] = np.ascontiguousarray(gdofs_map)

    for name in param_order:
        if name in args:
            continue
        if name in coefficient_arg_names:
            args[name] = np.empty(gdofs_map.shape, dtype=np.float64)
            continue
        raise KeyError(f"Missing native kernel static argument {name!r}.")
    return args


def _infer_coefficient_arg_names(
    residual_param_order: Sequence[str],
    tangent_param_order: Sequence[str],
) -> tuple[str, ...]:
    names = []
    for raw in tuple(residual_param_order) + tuple(tangent_param_order):
        name = str(raw)
        if not name.endswith("_loc"):
            continue
        stem = name[: -len("_loc")]
        if stem.endswith(("_n", "_prev", "_previous", "_old")):
            continue
        if stem.endswith(("_k", "_current")) or name.startswith("u_"):
            names.append(name)
    return tuple(dict.fromkeys(names))


def _coerce_affine_updates(updates: Sequence[Mapping[str, Any]] | None) -> tuple[dict[str, Any], ...]:
    if updates is None:
        return ()
    out: list[dict[str, Any]] = []
    for item in updates:
        if hasattr(item, "to_native_dict"):
            item = item.to_native_dict()
        out.append(
            {
                "name": str(item["name"]),
                "basis": np.ascontiguousarray(np.asarray(item["basis"], dtype=float)),
                "offset": np.ascontiguousarray(np.asarray(item["offset"], dtype=float).reshape(-1)),
            }
        )
    return tuple(out)


def _coerce_symbolic_updates(updates: Sequence[Mapping[str, Any]] | None) -> tuple[dict[str, Any], ...]:
    if updates is None:
        return ()
    out: list[dict[str, Any]] = []
    for item in updates:
        if hasattr(item, "to_online_dict"):
            item = item.to_online_dict()
        out.append(
            {
                "metadata_capsule": item["metadata_capsule"],
                "param_order": list(str(name) for name in item["param_order"]),
                "static_args": dict(item["static_args"]),
                "target_name": str(item["target_name"]),
                "scale": float(item.get("scale", 1.0)),
                "offset": float(item.get("offset", 0.0)),
            }
        )
    return tuple(out)


def _coerce_gnat_lift(gnat_lift: Any) -> Any:
    if gnat_lift is None:
        return None
    if is_sparse_matrix_like(gnat_lift):
        return NativeSparseMatrix.coerce(gnat_lift).to_native_dict()
    return np.ascontiguousarray(np.asarray(gnat_lift, dtype=float))


def _result_from_raw(raw: Mapping[str, Any]) -> NativeOnlineGaussNewtonResult:
    return NativeOnlineGaussNewtonResult(
        coefficients=np.asarray(raw["coefficients"], dtype=float).reshape(-1),
        converged=bool(raw["converged"]),
        iterations=int(raw["iterations"]),
        residual_norm=float(raw["residual_norm"]),
        linear_solver=str(raw["linear_solver"]),
        residual_norm_history=np.asarray(raw["residual_norm_history"], dtype=float).reshape(-1),
        step_norm_history=np.asarray(raw["step_norm_history"], dtype=float).reshape(-1),
        line_search_alpha_history=np.asarray(raw["line_search_alpha_history"], dtype=float).reshape(-1),
        damping_history=np.asarray(raw["damping_history"], dtype=float).reshape(-1),
        rejected_step_count=int(raw["rejected_step_count"]),
        timing_counters=dict(raw["timing_counters"]),
        backend=str(raw["backend"]),
    )


def solve_native_online_gauss_newton(
    *,
    residual_metadata_capsule: Any,
    residual_param_order: Sequence[str],
    residual_static_args: Mapping[str, Any],
    tangent_metadata_capsule: Any,
    tangent_param_order: Sequence[str],
    tangent_static_args: Mapping[str, Any],
    trial_basis: np.ndarray,
    offset: np.ndarray,
    initial_coefficients: np.ndarray,
    row_dofs: np.ndarray,
    coefficient_arg_names: Sequence[str] | None = None,
    element_weights: np.ndarray | None = None,
    row_weights: np.ndarray | None = None,
    gnat_lift: np.ndarray | NativeSparseMatrix | Mapping[str, Any] | Any | None = None,
    residual_state_updates: Sequence[Mapping[str, Any]] | None = None,
    tangent_state_updates: Sequence[Mapping[str, Any]] | None = None,
    residual_symbolic_state_updates: Sequence[Mapping[str, Any]] | None = None,
    tangent_symbolic_state_updates: Sequence[Mapping[str, Any]] | None = None,
    max_iterations: int = 8,
    residual_tol: float = 1.0e-10,
    step_tol: float = 1.0e-12,
    damping: float = 0.0,
    adaptive_damping: bool = False,
    max_damping_retries: int = 4,
    damping_increase: float = 10.0,
    damping_decrease: float = 0.25,
    line_search: bool = False,
    max_line_search: int = 6,
    sufficient_decrease: float = 1.0e-4,
    rcond: float | None = None,
) -> NativeOnlineGaussNewtonResult:
    """Solve a reduced nonlinear system without Python calls inside the online loop.

    The caller still owns the offline/setup phase: compile residual and tangent UFL
    forms with the C++ backend, prepare their static argument dictionaries, and
    pass the native metadata capsules exposed by the generated kernels.  Once this
    function enters the extension module, all reduced iterations run in C++.
    """

    residual_param_names = tuple(str(name) for name in residual_param_order)
    tangent_param_names = tuple(str(name) for name in tangent_param_order)
    coeff_names = (
        tuple(str(name) for name in coefficient_arg_names)
        if coefficient_arg_names is not None
        else _infer_coefficient_arg_names(residual_param_names, tangent_param_names)
    )
    residual_args = _normalize_static_args(residual_static_args, residual_param_names, coeff_names)
    tangent_args = _normalize_static_args(tangent_static_args, tangent_param_names, coeff_names)

    from .cpp_backend.online_gauss_newton import module as _online_gauss_newton_module

    raw = _online_gauss_newton_module().solve_online_gauss_newton(
        residual_metadata_capsule,
        list(residual_param_names),
        residual_args,
        tangent_metadata_capsule,
        list(tangent_param_names),
        tangent_args,
        np.ascontiguousarray(np.asarray(trial_basis, dtype=float)),
        np.ascontiguousarray(np.asarray(offset, dtype=float).reshape(-1)),
        np.ascontiguousarray(np.asarray(initial_coefficients, dtype=float).reshape(-1)),
        np.ascontiguousarray(np.asarray(row_dofs, dtype=np.int64).reshape(-1)),
        list(coeff_names),
        None if element_weights is None else np.ascontiguousarray(np.asarray(element_weights, dtype=float).reshape(-1)),
        None if row_weights is None else np.ascontiguousarray(np.asarray(row_weights, dtype=float).reshape(-1)),
        _coerce_gnat_lift(gnat_lift),
        _coerce_affine_updates(residual_state_updates),
        _coerce_affine_updates(tangent_state_updates),
        _coerce_symbolic_updates(residual_symbolic_state_updates),
        _coerce_symbolic_updates(tangent_symbolic_state_updates),
        int(max_iterations),
        float(residual_tol),
        float(step_tol),
        float(damping),
        bool(adaptive_damping),
        int(max_damping_retries),
        float(damping_increase),
        float(damping_decrease),
        bool(line_search),
        int(max_line_search),
        float(sufficient_decrease),
        -1.0 if rcond is None else float(rcond),
    )
    return _result_from_raw(raw)


def solve_native_deim_online_gauss_newton(
    *,
    residual_metadata_capsule: Any,
    residual_param_order: Sequence[str],
    residual_static_args: Mapping[str, Any],
    tangent_metadata_capsule: Any,
    tangent_param_order: Sequence[str],
    tangent_static_args: Mapping[str, Any],
    trial_basis: np.ndarray,
    offset: np.ndarray,
    initial_coefficients: np.ndarray,
    row_dofs: np.ndarray,
    selected_basis: np.ndarray,
    residual_terms: np.ndarray,
    coefficient_arg_names: Sequence[str] | None = None,
    element_weights: np.ndarray | None = None,
    row_weights: np.ndarray | None = None,
    gnat_lift: np.ndarray | NativeSparseMatrix | Mapping[str, Any] | Any | None = None,
    residual_state_updates: Sequence[Mapping[str, Any]] | None = None,
    tangent_state_updates: Sequence[Mapping[str, Any]] | None = None,
    residual_symbolic_state_updates: Sequence[Mapping[str, Any]] | None = None,
    tangent_symbolic_state_updates: Sequence[Mapping[str, Any]] | None = None,
    max_iterations: int = 8,
    residual_tol: float = 1.0e-10,
    step_tol: float = 1.0e-12,
    damping: float = 0.0,
    adaptive_damping: bool = False,
    max_damping_retries: int = 4,
    damping_increase: float = 10.0,
    damping_decrease: float = 0.25,
    line_search: bool = False,
    max_line_search: int = 6,
    sufficient_decrease: float = 1.0e-4,
    rcond: float | None = None,
) -> NativeOnlineGaussNewtonResult:
    """Solve a DEIM/QDEIM-composed reduced nonlinear system in the native loop.

    ``row_dofs`` identify the selected nonlinear residual features
    ``P^T R(q)``.  ``selected_basis`` is ``P^T U_c`` and
    ``residual_terms`` stores the offline reduced residual contribution per
    collateral mode.  C++ evaluates selected UFL rows, applies the fixed
    interpolation operator, composes the reduced residual/Jacobian, and runs
    Gauss-Newton without Python work inside the iteration.
    """

    residual_param_names = tuple(str(name) for name in residual_param_order)
    tangent_param_names = tuple(str(name) for name in tangent_param_order)
    coeff_names = (
        tuple(str(name) for name in coefficient_arg_names)
        if coefficient_arg_names is not None
        else _infer_coefficient_arg_names(residual_param_names, tangent_param_names)
    )
    residual_args = _normalize_static_args(residual_static_args, residual_param_names, coeff_names)
    tangent_args = _normalize_static_args(tangent_static_args, tangent_param_names, coeff_names)

    from .cpp_backend.online_gauss_newton import module as _online_gauss_newton_module

    raw = _online_gauss_newton_module().solve_deim_online_gauss_newton(
        residual_metadata_capsule,
        list(residual_param_names),
        residual_args,
        tangent_metadata_capsule,
        list(tangent_param_names),
        tangent_args,
        np.ascontiguousarray(np.asarray(trial_basis, dtype=float)),
        np.ascontiguousarray(np.asarray(offset, dtype=float).reshape(-1)),
        np.ascontiguousarray(np.asarray(initial_coefficients, dtype=float).reshape(-1)),
        np.ascontiguousarray(np.asarray(row_dofs, dtype=np.int64).reshape(-1)),
        list(coeff_names),
        np.ascontiguousarray(np.asarray(selected_basis, dtype=float)),
        np.ascontiguousarray(np.asarray(residual_terms, dtype=float)),
        None if element_weights is None else np.ascontiguousarray(np.asarray(element_weights, dtype=float).reshape(-1)),
        None if row_weights is None else np.ascontiguousarray(np.asarray(row_weights, dtype=float).reshape(-1)),
        _coerce_gnat_lift(gnat_lift),
        _coerce_affine_updates(residual_state_updates),
        _coerce_affine_updates(tangent_state_updates),
        _coerce_symbolic_updates(residual_symbolic_state_updates),
        _coerce_symbolic_updates(tangent_symbolic_state_updates),
        int(max_iterations),
        float(residual_tol),
        float(step_tol),
        float(damping),
        bool(adaptive_damping),
        int(max_damping_retries),
        float(damping_increase),
        float(damping_decrease),
        bool(line_search),
        int(max_line_search),
        float(sufficient_decrease),
        -1.0 if rcond is None else float(rcond),
    )
    return _result_from_raw(raw)


__all__ = [
    "NativeOnlineGaussNewtonResult",
    "native_kernel_metadata_from_runner",
    "solve_native_deim_online_gauss_newton",
    "solve_native_online_gauss_newton",
]
