"""Decoded-state bound constraints for reduced MOR solves.

The reduced state used by the native online solvers is usually decoded as

    x(q) = offset + V q.

Physical bounds such as ``0 <= alpha <= 1`` therefore become linear
inequalities in the reduced coefficients, not simple coefficient box bounds.
This module stores those decoded-state bounds in a problem-generic form and
builds the active equality rows used by native PDAS/IPM Gauss-Newton drivers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .sparse import NativeSparseMatrix, is_sparse_matrix_like


def _finite_matrix(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2-D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _finite_vector(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _bound_vector(value: Any, n_rows: int, label: str) -> np.ndarray:
    if value is None:
        default = -np.inf if "lower" in label else np.inf
        return np.full(int(n_rows), default, dtype=np.float64)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(int(n_rows), float(arr), dtype=np.float64)
    else:
        arr = arr.reshape(-1)
    if arr.size != int(n_rows):
        raise ValueError(f"{label} must be scalar or have one entry per constrained row.")
    if np.any(np.isnan(arr)):
        raise ValueError(f"{label} must not contain NaN.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _positive_scale(value: Any, n_rows: int, label: str) -> np.ndarray:
    if value is None:
        return np.ones(int(n_rows), dtype=np.float64)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(int(n_rows), float(arr), dtype=np.float64)
    else:
        arr = arr.reshape(-1)
    if arr.size != int(n_rows):
        raise ValueError(f"{label} must be scalar or have one entry per constrained row.")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{label} must contain finite positive values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _row_vector(value: Any, label: str = "rows") -> np.ndarray:
    rows = np.asarray(value, dtype=np.int64).reshape(-1)
    if rows.size == 0:
        raise ValueError(f"{label} must contain at least one row.")
    if np.any(rows < 0):
        raise ValueError(f"{label} must be nonnegative.")
    if np.unique(rows).size != rows.size:
        raise ValueError(f"{label} must not contain duplicate rows.")
    return np.ascontiguousarray(rows, dtype=np.int64)


def _constraint_matrix(value: Any, label: str) -> np.ndarray | NativeSparseMatrix:
    if is_sparse_matrix_like(value):
        return NativeSparseMatrix.coerce(value)
    return _finite_matrix(value, label)


def _constraint_shape(matrix: np.ndarray | NativeSparseMatrix) -> tuple[int, int]:
    if isinstance(matrix, NativeSparseMatrix):
        return matrix.shape
    return (int(matrix.shape[0]), int(matrix.shape[1]))


def _constraint_matvec(matrix: np.ndarray | NativeSparseMatrix, vector: Any) -> np.ndarray:
    if isinstance(matrix, NativeSparseMatrix):
        return matrix.matvec(vector)
    return np.ascontiguousarray(matrix @ np.asarray(vector, dtype=float).reshape(-1), dtype=np.float64)


def _constraint_active_rows(matrix: np.ndarray | NativeSparseMatrix, rows: np.ndarray) -> np.ndarray:
    row_ids = np.asarray(rows, dtype=np.int64).reshape(-1)
    if isinstance(matrix, NativeSparseMatrix):
        dense = matrix.to_dense()[row_ids, :]
        return np.ascontiguousarray(dense, dtype=np.float64)
    return np.ascontiguousarray(matrix[row_ids, :], dtype=np.float64)


@dataclass(frozen=True)
class BoundActivity:
    """Active and violated decoded-bound rows at one reduced coefficient vector."""

    lower_active: np.ndarray
    upper_active: np.ndarray
    inactive: np.ndarray
    lower_violation: np.ndarray
    upper_violation: np.ndarray
    max_violation: float


@dataclass(frozen=True)
class ActiveBoundEquations:
    """Equality rows produced by active decoded bounds.

    The equality is ``constraint_matrix @ step = rhs`` for the current
    Gauss-Newton increment.
    """

    constraint_matrix: np.ndarray
    rhs: np.ndarray
    lower_active: np.ndarray
    upper_active: np.ndarray
    rows: np.ndarray


@dataclass(frozen=True)
class ReducedBoundConstraintSpec:
    """Decoded full-state bounds expressed in reduced coordinates."""

    constraint_matrix: np.ndarray | NativeSparseMatrix
    offset: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    rows: np.ndarray
    row_scaling: np.ndarray
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        A = _constraint_matrix(self.constraint_matrix, "constraint_matrix")
        n_rows, n_cols = _constraint_shape(A)
        offset = _finite_vector(self.offset, "offset")
        if n_rows != offset.size:
            raise ValueError("constraint_matrix row count must match offset size.")
        lower = _bound_vector(self.lower, n_rows, "lower")
        upper = _bound_vector(self.upper, n_rows, "upper")
        if np.any(lower > upper):
            raise ValueError("lower bounds must be less than or equal to upper bounds.")
        rows = _row_vector(self.rows)
        if rows.size != n_rows:
            raise ValueError("rows size must match constraint_matrix row count.")
        scale = _positive_scale(self.row_scaling, n_rows, "row_scaling")
        object.__setattr__(self, "constraint_matrix", A)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "row_scaling", scale)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_constraints(self) -> int:
        return int(_constraint_shape(self.constraint_matrix)[0])

    @property
    def n_coefficients(self) -> int:
        return int(_constraint_shape(self.constraint_matrix)[1])

    def decoded_values(self, coefficients: Any) -> np.ndarray:
        q = _finite_vector(coefficients, "coefficients")
        if q.size != self.n_coefficients:
            raise ValueError("coefficients size must match reduced constraint columns.")
        return np.ascontiguousarray(self.offset + _constraint_matvec(self.constraint_matrix, q), dtype=np.float64)

    def violations(self, coefficients: Any) -> tuple[np.ndarray, np.ndarray]:
        values = self.decoded_values(coefficients)
        lower_violation = np.maximum(self.lower - values, 0.0)
        upper_violation = np.maximum(values - self.upper, 0.0)
        lower_violation[~np.isfinite(self.lower)] = 0.0
        upper_violation[~np.isfinite(self.upper)] = 0.0
        return (
            np.ascontiguousarray(lower_violation, dtype=np.float64),
            np.ascontiguousarray(upper_violation, dtype=np.float64),
        )

    def max_violation(self, coefficients: Any) -> float:
        lower_violation, upper_violation = self.violations(coefficients)
        if lower_violation.size == 0:
            return 0.0
        return float(max(float(np.max(lower_violation)), float(np.max(upper_violation))))

    def activity(self, coefficients: Any, *, active_tol: float = 1.0e-10) -> BoundActivity:
        values = self.decoded_values(coefficients)
        tol = float(active_tol)
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("active_tol must be finite and nonnegative.")
        lower_finite = np.isfinite(self.lower)
        upper_finite = np.isfinite(self.upper)
        lower_active = lower_finite & (values <= self.lower + tol)
        upper_active = upper_finite & (values >= self.upper - tol)
        # Equal lower/upper bounds should be represented once as a lower row.
        upper_active = upper_active & ~lower_active
        inactive = ~(lower_active | upper_active)
        lower_violation, upper_violation = self.violations(coefficients)
        return BoundActivity(
            lower_active=np.ascontiguousarray(np.nonzero(lower_active)[0], dtype=np.int64),
            upper_active=np.ascontiguousarray(np.nonzero(upper_active)[0], dtype=np.int64),
            inactive=np.ascontiguousarray(np.nonzero(inactive)[0], dtype=np.int64),
            lower_violation=lower_violation,
            upper_violation=upper_violation,
            max_violation=float(max(float(np.max(lower_violation)), float(np.max(upper_violation)))),
        )

    def active_equations(self, coefficients: Any, *, active_tol: float = 1.0e-10) -> ActiveBoundEquations:
        q = _finite_vector(coefficients, "coefficients")
        values = self.decoded_values(q)
        activity = self.activity(q, active_tol=active_tol)
        active = np.concatenate((activity.lower_active, activity.upper_active)).astype(np.int64, copy=False)
        if active.size == 0:
            return ActiveBoundEquations(
                constraint_matrix=np.zeros((0, self.n_coefficients), dtype=np.float64),
                rhs=np.zeros(0, dtype=np.float64),
                lower_active=activity.lower_active,
                upper_active=activity.upper_active,
                rows=np.zeros(0, dtype=np.int64),
            )
        rhs_lower = self.lower[activity.lower_active] - values[activity.lower_active]
        rhs_upper = self.upper[activity.upper_active] - values[activity.upper_active]
        rhs = np.concatenate((rhs_lower, rhs_upper)).astype(np.float64, copy=False)
        scale = self.row_scaling[active]
        C = _constraint_active_rows(self.constraint_matrix, active) * scale[:, None]
        return ActiveBoundEquations(
            constraint_matrix=np.ascontiguousarray(C, dtype=np.float64),
            rhs=np.ascontiguousarray(rhs * scale, dtype=np.float64),
            lower_active=activity.lower_active,
            upper_active=activity.upper_active,
            rows=np.ascontiguousarray(self.rows[active], dtype=np.int64),
        )

    def to_native_dict(self) -> dict[str, Any]:
        matrix = self.constraint_matrix.to_native_dict() if isinstance(self.constraint_matrix, NativeSparseMatrix) else self.constraint_matrix
        return {
            "kind": "reduced_bound_constraints",
            "constraint_matrix": matrix,
            "offset": self.offset,
            "lower": self.lower,
            "upper": self.upper,
            "rows": self.rows,
            "row_scaling": self.row_scaling,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_native_dict(cls, payload: Mapping[str, Any]) -> "ReducedBoundConstraintSpec":
        return cls(
            constraint_matrix=payload["constraint_matrix"],
            offset=payload["offset"],
            lower=payload["lower"],
            upper=payload["upper"],
            rows=payload["rows"],
            row_scaling=payload.get("row_scaling", 1.0),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class BoundConstraintSpec:
    """Full-state decoded bounds before reduction by ``offset + V q``."""

    rows: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    row_scaling: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        rows = _row_vector(self.rows)
        lower = _bound_vector(self.lower, rows.size, "lower")
        upper = _bound_vector(self.upper, rows.size, "upper")
        if np.any(lower > upper):
            raise ValueError("lower bounds must be less than or equal to upper bounds.")
        scale = _positive_scale(self.row_scaling, rows.size, "row_scaling")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "row_scaling", scale)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def reduce(self, *, trial_basis: Any, offset: Any, sparse: bool = False, drop_tol: float = 0.0) -> ReducedBoundConstraintSpec:
        if isinstance(trial_basis, NativeSparseMatrix) or isinstance(trial_basis, Mapping):
            V: Any = NativeSparseMatrix.coerce(trial_basis)
        elif is_sparse_matrix_like(trial_basis):
            V = trial_basis
        else:
            V = _finite_matrix(trial_basis, "trial_basis")
        x0 = _finite_vector(offset, "offset")
        shape = V.shape
        if int(shape[0]) != x0.size:
            raise ValueError("trial_basis rows must match offset size.")
        if np.any(self.rows >= x0.size):
            raise ValueError("constraint rows are outside the decoded state size.")
        if is_sparse_matrix_like(V):
            try:
                import scipy.sparse as sp
            except Exception:  # pragma: no cover - scipy is normally present
                sp = None
            if isinstance(V, NativeSparseMatrix):
                reduced_matrix: np.ndarray | NativeSparseMatrix = V.row_subset(self.rows)
            elif sp is not None and sp.issparse(V):
                reduced_matrix = NativeSparseMatrix.from_scipy(V.tocsr()[self.rows, :])
            else:
                reduced_matrix = NativeSparseMatrix.coerce(V).row_subset(self.rows)
        else:
            sliced = np.asarray(V[self.rows, :], dtype=float)
            reduced_matrix = NativeSparseMatrix.from_dense(sliced, drop_tol=drop_tol) if bool(sparse) else sliced
        return ReducedBoundConstraintSpec(
            constraint_matrix=reduced_matrix,
            offset=x0[self.rows],
            lower=self.lower,
            upper=self.upper,
            rows=self.rows,
            row_scaling=self.row_scaling,
            metadata=dict(self.metadata or {}),
        )

    def to_native_dict(self) -> dict[str, Any]:
        return {
            "kind": "bound_constraints",
            "rows": self.rows,
            "lower": self.lower,
            "upper": self.upper,
            "row_scaling": self.row_scaling,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_native_dict(cls, payload: Mapping[str, Any]) -> "BoundConstraintSpec":
        return cls(
            rows=payload["rows"],
            lower=payload["lower"],
            upper=payload["upper"],
            row_scaling=payload.get("row_scaling", None),
            metadata=payload.get("metadata", {}),
        )


def bound_constraints_from_fields(
    dof_handler: Any,
    field_bounds: Mapping[str, tuple[float | None, float | None]],
    *,
    row_scaling: float | Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BoundConstraintSpec:
    """Build decoded bounds from field names and a dof handler.

    ``field_bounds`` maps each field name to ``(lower, upper)``.  ``None`` means
    an infinite one-sided bound.
    """

    rows: list[np.ndarray] = []
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    field_names: list[str] = []
    for field, (lo, hi) in field_bounds.items():
        field_rows = np.asarray(dof_handler.get_field_slice(str(field)), dtype=np.int64).reshape(-1)
        if field_rows.size == 0:
            continue
        rows.append(field_rows)
        lower_value = -np.inf if lo is None else float(lo)
        upper_value = np.inf if hi is None else float(hi)
        lower.append(np.full(field_rows.size, lower_value, dtype=np.float64))
        upper.append(np.full(field_rows.size, upper_value, dtype=np.float64))
        if isinstance(row_scaling, Mapping):
            scale_value = float(row_scaling.get(str(field), 1.0))
        elif row_scaling is None:
            scale_value = 1.0
        else:
            scale_value = float(row_scaling)
        scales.append(np.full(field_rows.size, scale_value, dtype=np.float64))
        field_names.extend([str(field)] * int(field_rows.size))
    if not rows:
        raise ValueError("field_bounds did not produce any constrained rows.")
    meta = dict(metadata or {})
    meta.setdefault("fields", tuple(str(name) for name in field_bounds))
    meta.setdefault("field_per_row", tuple(field_names))
    return BoundConstraintSpec(
        rows=np.concatenate(rows),
        lower=np.concatenate(lower),
        upper=np.concatenate(upper),
        row_scaling=np.concatenate(scales),
        metadata=meta,
    )


def project_reduced_coefficients_to_bounds(
    constraints: ReducedBoundConstraintSpec | Mapping[str, Any],
    coefficients: Any,
    *,
    max_iterations: int = 200,
    tolerance: float = 1.0e-10,
    relaxation: float = 1.0,
) -> np.ndarray:
    """Project reduced coefficients onto decoded linear bound constraints.

    The projection uses cyclic half-space projections in coefficient space.
    It is intended for offline/setup and predictor repair before entering the
    native online loop, so the C++ nonlinear iteration can start from a
    feasible decoded state.
    """

    spec = constraints if isinstance(constraints, ReducedBoundConstraintSpec) else ReducedBoundConstraintSpec.from_native_dict(constraints)
    q = _finite_vector(coefficients, "coefficients").copy()
    if q.size != spec.n_coefficients:
        raise ValueError("coefficients size must match reduced constraint columns.")
    max_iter = max(1, int(max_iterations))
    tol = float(tolerance)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    omega = float(relaxation)
    if not np.isfinite(omega) or omega <= 0.0 or omega > 2.0:
        raise ValueError("relaxation must be in the interval (0, 2].")
    if isinstance(spec.constraint_matrix, NativeSparseMatrix):
        A = spec.constraint_matrix.to_dense()
    else:
        A = np.asarray(spec.constraint_matrix, dtype=float)
    row_norm_sq = np.sum(A * A, axis=1)
    active = row_norm_sq > 0.0
    if not np.any(active):
        return np.ascontiguousarray(q, dtype=np.float64)
    for _iteration in range(max_iter):
        max_violation = 0.0
        values = spec.offset + A @ q
        for i in range(A.shape[0]):
            if not active[i]:
                continue
            if np.isfinite(spec.lower[i]) and values[i] < spec.lower[i] - tol:
                violation = float(spec.lower[i] - values[i])
                q += omega * violation / row_norm_sq[i] * A[i, :]
                max_violation = max(max_violation, violation)
            elif np.isfinite(spec.upper[i]) and values[i] > spec.upper[i] + tol:
                violation = float(values[i] - spec.upper[i])
                q -= omega * violation / row_norm_sq[i] * A[i, :]
                max_violation = max(max_violation, violation)
        if max_violation <= tol or spec.max_violation(q) <= tol:
            break
    return np.ascontiguousarray(q, dtype=np.float64)


__all__ = [
    "ActiveBoundEquations",
    "BoundActivity",
    "BoundConstraintSpec",
    "ReducedBoundConstraintSpec",
    "bound_constraints_from_fields",
    "project_reduced_coefficients_to_bounds",
]
