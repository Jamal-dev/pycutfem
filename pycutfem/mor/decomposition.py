"""Problem-generic non-affine MOR decomposition helpers.

The objects in this module describe offline/online decompositions such as
DEIM/QDEIM without tying them to a particular PDE or example.  Python owns the
offline training and artifact inspection; the small online solves/compositions
can be dispatched to C++ for native reduced drivers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _as_finite_matrix(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2-D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _as_finite_vector(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _normalize_backend(backend: str) -> str:
    name = str(backend).strip().lower()
    if name in {"python", "numpy"}:
        return "python"
    if name in {"cpp", "c++", "native"}:
        return "cpp"
    raise ValueError(f"Unsupported MOR decomposition backend {backend!r}.")


@dataclass(frozen=True)
class CollateralBasis:
    """Collateral basis for non-affine nonlinear features or residuals."""

    basis: np.ndarray
    singular_values: np.ndarray | None = None
    snapshot_mean: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        basis = _as_finite_matrix(self.basis, "collateral basis")
        if basis.shape[1] == 0:
            raise ValueError("collateral basis must contain at least one mode.")
        singular_values = None
        if self.singular_values is not None:
            singular_values = _as_finite_vector(self.singular_values, "collateral singular values")
            if singular_values.size < basis.shape[1]:
                raise ValueError("collateral singular values must cover every retained mode.")
        snapshot_mean = None
        if self.snapshot_mean is not None:
            snapshot_mean = _as_finite_vector(self.snapshot_mean, "collateral snapshot mean")
            if snapshot_mean.size != basis.shape[0]:
                raise ValueError("collateral snapshot mean size must match basis rows.")
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "singular_values", singular_values)
        object.__setattr__(self, "snapshot_mean", snapshot_mean)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_features(self) -> int:
        return int(self.basis.shape[0])

    @property
    def n_modes(self) -> int:
        return int(self.basis.shape[1])

    def to_native_dict(self) -> dict[str, Any]:
        return {
            "basis": self.basis,
            "singular_values": np.zeros(0, dtype=float) if self.singular_values is None else self.singular_values,
            "snapshot_mean": np.zeros(0, dtype=float) if self.snapshot_mean is None else self.snapshot_mean,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class InterpolationRule:
    """Rows and solve data for DEIM/QDEIM/EIM-style online interpolation."""

    method: str
    rows: np.ndarray
    basis: np.ndarray
    selected_basis: np.ndarray | None = None
    solve_mode: str = "auto"
    rcond: float | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        basis = _as_finite_matrix(self.basis, "interpolation basis")
        rows = np.asarray(self.rows, dtype=np.int64).reshape(-1)
        if rows.size < basis.shape[1]:
            raise ValueError("interpolation rule must select at least n_modes rows.")
        if np.any(rows < 0) or np.any(rows >= basis.shape[0]):
            raise ValueError("interpolation rows are out of range.")
        if np.unique(rows).size != rows.size:
            raise ValueError("interpolation rows must be unique.")
        selected = (
            _as_finite_matrix(self.selected_basis, "selected interpolation basis")
            if self.selected_basis is not None
            else np.ascontiguousarray(basis[rows, :], dtype=np.float64)
        )
        if selected.shape != (rows.size, basis.shape[1]):
            raise ValueError("selected interpolation basis shape is incompatible.")
        rcond = None if self.rcond is None else float(self.rcond)
        if rcond is not None and (not np.isfinite(rcond) or rcond < 0.0):
            raise ValueError("interpolation rcond must be finite and nonnegative.")
        object.__setattr__(self, "method", str(self.method).lower())
        object.__setattr__(self, "rows", np.ascontiguousarray(rows, dtype=np.int64))
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "selected_basis", selected)
        object.__setattr__(self, "rcond", rcond)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_modes(self) -> int:
        return int(self.basis.shape[1])

    @property
    def n_selected(self) -> int:
        return int(self.rows.size)

    def to_native_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "rows": self.rows,
            "basis": self.basis,
            "selected_basis": self.selected_basis,
            "solve_mode": self.solve_mode,
            "rcond": -1.0 if self.rcond is None else float(self.rcond),
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_native_dict(cls, payload: Mapping[str, Any]) -> "InterpolationRule":
        return cls(
            method=str(payload["method"]),
            rows=payload["rows"],
            basis=payload["basis"],
            selected_basis=payload.get("selected_basis"),
            solve_mode=str(payload.get("solve_mode", "auto")),
            rcond=None if float(payload.get("rcond", -1.0)) < 0.0 else float(payload["rcond"]),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class EmpiricalCubatureRule:
    """Element/entity samples and weights for empirical-cubature targets."""

    entity_ids: np.ndarray
    weights: np.ndarray
    entity_kind: str = "cell"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        entity_ids = np.asarray(self.entity_ids, dtype=np.int64).reshape(-1)
        weights = _as_finite_vector(self.weights, "empirical cubature weights")
        if entity_ids.size != weights.size:
            raise ValueError("empirical cubature entity_ids and weights must have matching sizes.")
        if np.any(entity_ids < 0):
            raise ValueError("empirical cubature entity ids must be nonnegative.")
        if np.any(weights < 0.0):
            raise ValueError("empirical cubature weights must be nonnegative.")
        object.__setattr__(self, "entity_ids", np.ascontiguousarray(entity_ids, dtype=np.int64))
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "entity_kind", str(self.entity_kind))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_native_dict(self) -> dict[str, Any]:
        return {
            "entity_ids": self.entity_ids,
            "weights": self.weights,
            "entity_kind": self.entity_kind,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class ReducedOperatorTerm:
    """Offline reduced term block used by a native non-affine evaluator."""

    term_id: str
    residual_block: np.ndarray
    jacobian_block: np.ndarray | None = None
    role: str = "residual"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        residual = _as_finite_vector(self.residual_block, "reduced residual term")
        jacobian = None
        if self.jacobian_block is not None:
            jacobian = _as_finite_matrix(self.jacobian_block, "reduced jacobian term")
        object.__setattr__(self, "term_id", str(self.term_id))
        object.__setattr__(self, "residual_block", residual)
        object.__setattr__(self, "jacobian_block", jacobian)
        object.__setattr__(self, "role", str(self.role))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class NativeReducedEvaluationGraph:
    """Serializable online graph for native non-affine reduced evaluation."""

    interpolation_rule: InterpolationRule | None = None
    cubature_rule: EmpiricalCubatureRule | None = None
    operator_terms: tuple[ReducedOperatorTerm, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator_terms", tuple(self.operator_terms))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_native_dict(self) -> dict[str, Any]:
        return {
            "interpolation_rule": None if self.interpolation_rule is None else self.interpolation_rule.to_native_dict(),
            "cubature_rule": None if self.cubature_rule is None else self.cubature_rule.to_native_dict(),
            "operator_term_ids": tuple(term.term_id for term in self.operator_terms),
            "metadata": dict(self.metadata or {}),
        }


def fit_collateral_basis(
    snapshots: Any,
    *,
    n_modes: int | None = None,
    energy: float | None = None,
    center: bool = False,
) -> CollateralBasis:
    """Fit a POD/collateral basis from feature snapshots.

    ``snapshots`` uses shape ``(n_features, n_snapshots)``.  ``energy`` is the
    retained squared-singular-value fraction.
    """

    matrix = _as_finite_matrix(snapshots, "collateral snapshots")
    if center:
        mean = np.mean(matrix, axis=1)
        matrix = matrix - mean[:, None]
    else:
        mean = None
    if matrix.shape[1] == 0:
        raise ValueError("collateral snapshots must include at least one snapshot.")
    u, s, _vh = np.linalg.svd(matrix, full_matrices=False)
    if energy is not None:
        target = float(energy)
        if not np.isfinite(target) or target <= 0.0 or target > 1.0:
            raise ValueError("energy must be in the interval (0, 1].")
        cumulative = np.cumsum(s * s)
        total = float(cumulative[-1]) if cumulative.size else 0.0
        energy_modes = 1 if total == 0.0 else int(np.searchsorted(cumulative / total, target, side="left") + 1)
        n_modes = energy_modes if n_modes is None else min(int(n_modes), energy_modes)
    if n_modes is None:
        n_modes = min(matrix.shape)
    n = int(n_modes)
    if n <= 0 or n > u.shape[1]:
        raise ValueError("n_modes must be between 1 and the snapshot rank.")
    return CollateralBasis(
        basis=u[:, :n],
        singular_values=s[:n],
        snapshot_mean=mean,
        metadata={"centered": bool(center), "snapshot_count": int(matrix.shape[1])},
    )


def select_deim_rows(basis: Any) -> np.ndarray:
    """Select interpolation rows using the classical DEIM greedy rule."""

    u = _as_finite_matrix(basis, "DEIM basis")
    n_features, n_modes = u.shape
    if n_modes == 0 or n_modes > n_features:
        raise ValueError("DEIM basis must have 1 <= n_modes <= n_features.")
    rows: list[int] = [int(np.argmax(np.abs(u[:, 0])))]
    for mode in range(1, n_modes):
        selected = np.asarray(rows, dtype=np.int64)
        coeffs, *_ = np.linalg.lstsq(u[selected, :mode], u[selected, mode], rcond=None)
        residual = u[:, mode] - u[:, :mode] @ coeffs
        order = np.argsort(-np.abs(residual))
        for candidate in order:
            row = int(candidate)
            if row not in rows:
                rows.append(row)
                break
        else:  # pragma: no cover - impossible unless n_modes > n_features
            raise ValueError("DEIM could not find a unique interpolation row.")
    return np.ascontiguousarray(np.asarray(rows, dtype=np.int64))


def select_qdeim_rows(basis: Any) -> np.ndarray:
    """Select interpolation rows using QDEIM pivoted QR when SciPy is present."""

    u = _as_finite_matrix(basis, "QDEIM basis")
    n_features, n_modes = u.shape
    if n_modes == 0 or n_modes > n_features:
        raise ValueError("QDEIM basis must have 1 <= n_modes <= n_features.")
    try:
        from scipy.linalg import qr  # type: ignore

        _q, _r, pivots = qr(u.T, pivoting=True, mode="economic")
        rows = np.asarray(pivots[:n_modes], dtype=np.int64)
    except Exception:
        rows = select_deim_rows(u)
    if np.unique(rows).size != rows.size:
        raise ValueError("QDEIM selected duplicate interpolation rows.")
    return np.ascontiguousarray(rows, dtype=np.int64)


def build_interpolation_rule(
    collateral_basis: CollateralBasis | Any,
    *,
    method: str = "deim",
    rows: Any | None = None,
    rcond: float | None = None,
) -> InterpolationRule:
    basis = collateral_basis.basis if isinstance(collateral_basis, CollateralBasis) else _as_finite_matrix(collateral_basis, "basis")
    name = str(method).lower()
    if rows is None:
        if name == "deim":
            rows = select_deim_rows(basis)
        elif name == "qdeim":
            rows = select_qdeim_rows(basis)
        else:
            raise ValueError("rows must be provided for custom interpolation methods.")
    return InterpolationRule(method=name, rows=rows, basis=basis, rcond=rcond)


def build_deim_interpolation_rule(collateral_basis: CollateralBasis | Any, *, rcond: float | None = None) -> InterpolationRule:
    return build_interpolation_rule(collateral_basis, method="deim", rcond=rcond)


def build_qdeim_interpolation_rule(collateral_basis: CollateralBasis | Any, *, rcond: float | None = None) -> InterpolationRule:
    return build_interpolation_rule(collateral_basis, method="qdeim", rcond=rcond)


def interpolation_coefficients(
    rule: InterpolationRule,
    selected_values: Any,
    *,
    backend: str = "python",
    rcond: float | None = None,
) -> np.ndarray:
    """Solve ``P^T U c = P^T f`` for DEIM/QDEIM coefficients."""

    values = np.asarray(selected_values, dtype=float)
    if values.ndim == 1:
        rhs = np.ascontiguousarray(values.reshape(-1), dtype=np.float64)
        if rhs.size != rule.n_selected:
            raise ValueError("selected values size must match interpolation rows.")
    elif values.ndim == 2:
        rhs = np.ascontiguousarray(values, dtype=np.float64)
        if rhs.shape[0] != rule.n_selected:
            raise ValueError("selected values row count must match interpolation rows.")
    else:
        raise ValueError("selected values must be a vector or matrix.")
    if not np.all(np.isfinite(rhs)):
        raise ValueError("selected values must be finite.")

    effective_rcond = rule.rcond if rcond is None else rcond
    if _normalize_backend(backend) == "cpp":
        from .cpp_backend.deim_online import module as _deim_online_module

        raw = _deim_online_module().solve_interpolation(rule.selected_basis, rhs, -1.0 if effective_rcond is None else float(effective_rcond))
        coeffs = np.asarray(raw["coefficients"], dtype=float)
        return coeffs.reshape(-1) if values.ndim == 1 else coeffs

    selected_basis = np.asarray(rule.selected_basis, dtype=float)
    if selected_basis.shape[0] == selected_basis.shape[1]:
        try:
            return np.linalg.solve(selected_basis, rhs)
        except np.linalg.LinAlgError:
            pass
    coeffs, *_ = np.linalg.lstsq(selected_basis, rhs, rcond=effective_rcond)
    return np.asarray(coeffs, dtype=float)


def reconstruct_from_interpolation(
    rule: InterpolationRule,
    selected_values: Any,
    *,
    backend: str = "python",
    rcond: float | None = None,
) -> np.ndarray:
    coeffs = interpolation_coefficients(rule, selected_values, backend=backend, rcond=rcond)
    return np.asarray(rule.basis @ coeffs, dtype=float)


def compose_reduced_operator(
    coefficients: Any,
    residual_terms: Any,
    jacobian_terms: Any | None = None,
    *,
    backend: str = "python",
) -> dict[str, np.ndarray]:
    """Compose reduced residual/Jacobian blocks from interpolation coefficients.

    ``residual_terms`` has shape ``(n_terms, n_residual)`` and
    ``jacobian_terms`` has shape ``(n_terms, n_residual, n_modes)``.
    """

    coeffs = _as_finite_vector(coefficients, "interpolation coefficients")
    residual = _as_finite_matrix(residual_terms, "reduced residual terms")
    if residual.shape[0] != coeffs.size:
        raise ValueError("residual_terms first dimension must match coefficient count.")
    jacobian = None
    if jacobian_terms is not None:
        jacobian = np.asarray(jacobian_terms, dtype=float)
        if jacobian.ndim != 3:
            raise ValueError("jacobian_terms must be a 3-D array.")
        if jacobian.shape[0] != coeffs.size:
            raise ValueError("jacobian_terms first dimension must match coefficient count.")
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("jacobian_terms must contain only finite values.")
        jacobian = np.ascontiguousarray(jacobian, dtype=np.float64)

    if _normalize_backend(backend) == "cpp":
        from .cpp_backend.deim_online import module as _deim_online_module

        raw = _deim_online_module().compose_reduced_system(
            coeffs,
            residual,
            None if jacobian is None else jacobian,
        )
        out = {"residual": np.asarray(raw["residual"], dtype=float).reshape(-1)}
        if "jacobian" in raw:
            out["jacobian"] = np.asarray(raw["jacobian"], dtype=float)
        return out

    out = {"residual": np.asarray(coeffs @ residual, dtype=float).reshape(-1)}
    if jacobian is not None:
        out["jacobian"] = np.asarray(np.tensordot(coeffs, jacobian, axes=(0, 0)), dtype=float)
    return out


__all__ = [
    "CollateralBasis",
    "EmpiricalCubatureRule",
    "InterpolationRule",
    "NativeReducedEvaluationGraph",
    "ReducedOperatorTerm",
    "build_deim_interpolation_rule",
    "build_interpolation_rule",
    "build_qdeim_interpolation_rule",
    "compose_reduced_operator",
    "fit_collateral_basis",
    "interpolation_coefficients",
    "reconstruct_from_interpolation",
    "select_deim_rows",
    "select_qdeim_rows",
]
