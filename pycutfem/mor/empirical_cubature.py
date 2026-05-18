"""Positive empirical cubature fitting for hyper-reduced operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .decomposition import EmpiricalCubatureRule


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


def _index_vector(value: Any, label: str, upper: int | None = None, *, unique: bool = True) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64).reshape(-1)
    if np.any(arr < 0):
        raise ValueError(f"{label} must contain only nonnegative ids.")
    if upper is not None and arr.size and int(np.max(arr)) >= int(upper):
        raise ValueError(f"{label} contains ids outside the available range.")
    if unique:
        arr = np.unique(arr)
    return np.ascontiguousarray(arr, dtype=np.int64)


def _solve_nonnegative_least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import nnls

        weights, _ = nnls(A, b)
        return np.asarray(weights, dtype=float)
    except Exception:
        active = np.arange(A.shape[1], dtype=int)
        weights = np.zeros(A.shape[1], dtype=float)
        for _ in range(max(1, A.shape[1])):
            if active.size == 0:
                break
            local, *_ = np.linalg.lstsq(A[:, active], b, rcond=None)
            local = np.asarray(local, dtype=float).reshape(-1)
            keep = local > 0.0
            if np.all(keep):
                weights[active] = local
                break
            active = active[keep]
        return np.maximum(weights, 0.0)


@dataclass(frozen=True)
class EmpiricalCubatureFit:
    """Result of a positive empirical-cubature fit."""

    rule: EmpiricalCubatureRule
    relative_error: float
    residual_norm: float
    target_norm: float
    iterations: int
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_error", float(self.relative_error))
        object.__setattr__(self, "residual_norm", float(self.residual_norm))
        object.__setattr__(self, "target_norm", float(self.target_norm))
        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def passed(self) -> bool:
        return bool(self.metadata.get("passed", False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule.to_native_dict(),
            "relative_error": float(self.relative_error),
            "residual_norm": float(self.residual_norm),
            "target_norm": float(self.target_norm),
            "iterations": int(self.iterations),
            "metadata": dict(self.metadata or {}),
        }


def apply_empirical_cubature(contributions: Any, rule: EmpiricalCubatureRule) -> np.ndarray:
    """Apply a cubature rule to entity contribution rows."""

    matrix = _finite_matrix(contributions, "cubature contributions")
    ids = _index_vector(rule.entity_ids, "cubature entity ids", matrix.shape[0])
    weights = _finite_vector(rule.weights, "cubature weights")
    if weights.size != ids.size:
        raise ValueError("cubature weights must match entity ids.")
    return np.asarray(weights @ matrix[ids, :], dtype=float).reshape(-1)


def fit_positive_empirical_cubature(
    contributions: Any,
    *,
    target: Any | None = None,
    candidate_ids: Any | None = None,
    mandatory_ids: Any | None = None,
    max_entities: int | None = None,
    tolerance: float = 1.0e-8,
    normalize_targets: bool = True,
    prune_tolerance: float = 1.0e-12,
    entity_kind: str = "cell",
    metadata: Mapping[str, Any] | None = None,
) -> EmpiricalCubatureFit:
    """Fit nonnegative entity weights for empirical cubature.

    ``contributions`` has shape ``(n_entities, n_targets)``.  The fitted rule
    approximates ``target``; when ``target`` is omitted it uses the full sum of
    all entity contributions.
    """

    matrix = _finite_matrix(contributions, "cubature contributions")
    n_entities, n_targets = matrix.shape
    if n_entities == 0 or n_targets == 0:
        raise ValueError("cubature contributions must be nonempty.")
    b = np.sum(matrix, axis=0) if target is None else _finite_vector(target, "cubature target")
    if b.size != n_targets:
        raise ValueError("cubature target size must match contribution columns.")

    candidates = (
        np.arange(n_entities, dtype=np.int64)
        if candidate_ids is None
        else _index_vector(candidate_ids, "cubature candidate_ids", n_entities)
    )
    mandatory = (
        np.zeros(0, dtype=np.int64)
        if mandatory_ids is None
        else _index_vector(mandatory_ids, "cubature mandatory_ids", n_entities)
    )
    selected: list[int] = [int(v) for v in mandatory]
    selected_set = set(selected)
    max_count = n_entities if max_entities is None else max(0, int(max_entities))
    max_count = max(max_count, len(selected))
    tol = float(tolerance)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("cubature tolerance must be finite and nonnegative.")

    A = matrix.T
    if normalize_targets:
        scale = 1.0 / np.maximum(np.abs(b), np.linalg.norm(A, axis=1))
        scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
        A_fit = scale[:, None] * A
        b_fit = scale * b
    else:
        A_fit = A
        b_fit = b

    target_norm = float(np.linalg.norm(b))
    fit_weights = np.zeros(0, dtype=float)
    residual = b_fit.copy()
    rel_error = float("inf")
    iterations = 0
    candidate_set = set(int(v) for v in candidates)

    while True:
        iterations += 1
        if selected:
            ids = np.asarray(selected, dtype=np.int64)
            fit_weights = _solve_nonnegative_least_squares(A_fit[:, ids], b_fit)
            approx = A_fit[:, ids] @ fit_weights
            residual = b_fit - approx
            unscaled_residual = b - A[:, ids] @ fit_weights
        else:
            unscaled_residual = b.copy()
        residual_norm = float(np.linalg.norm(unscaled_residual))
        rel_error = residual_norm / max(target_norm, 1.0e-300)
        if rel_error <= tol or len(selected) >= max_count:
            break

        available = np.asarray(
            [idx for idx in candidates if int(idx) not in selected_set],
            dtype=np.int64,
        )
        if available.size == 0:
            break
        correlations = A_fit[:, available].T @ residual
        best = int(available[int(np.argmax(np.abs(correlations)))])
        if best not in candidate_set:
            break
        selected.append(best)
        selected_set.add(best)

    ids = np.asarray(selected, dtype=np.int64)
    weights = np.asarray(fit_weights, dtype=float).reshape(-1)
    if ids.size != weights.size:
        weights = np.zeros(ids.size, dtype=float)
    keep = weights > float(prune_tolerance)
    if mandatory.size:
        keep = keep | np.isin(ids, mandatory)
    ids = ids[keep]
    weights = weights[keep]
    if ids.size == 0:
        ids = np.zeros(0, dtype=np.int64)
        weights = np.zeros(0, dtype=float)
    order = np.argsort(ids)
    ids = np.ascontiguousarray(ids[order], dtype=np.int64)
    weights = np.ascontiguousarray(weights[order], dtype=np.float64)
    final = apply_empirical_cubature(matrix, EmpiricalCubatureRule(ids, weights, entity_kind=entity_kind))
    residual_norm = float(np.linalg.norm(b - final))
    rel_error = residual_norm / max(target_norm, 1.0e-300)
    return EmpiricalCubatureFit(
        rule=EmpiricalCubatureRule(ids, weights, entity_kind=entity_kind, metadata=dict(metadata or {})),
        relative_error=rel_error,
        residual_norm=residual_norm,
        target_norm=target_norm,
        iterations=iterations,
        metadata={
            "passed": bool(rel_error <= tol),
            "tolerance": tol,
            "candidate_count": int(candidates.size),
            "mandatory_count": int(mandatory.size),
            "max_entities": int(max_count),
            **dict(metadata or {}),
        },
    )


__all__ = [
    "EmpiricalCubatureFit",
    "apply_empirical_cubature",
    "fit_positive_empirical_cubature",
]
