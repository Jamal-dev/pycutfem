"""Reduced-reference policies for nonlinear ROM branch selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .predictors import (
    LinearHistoryReducedPredictor,
    ReducedReferencePrediction,
    ReducedReferencePredictor,
    predictor_from_native_dict,
)


def _finite_vector(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def clip_reference_distance(
    reference: Any,
    current: Any,
    *,
    metric_basis: np.ndarray | None = None,
    max_distance: float | None = None,
) -> tuple[np.ndarray, bool, float, float]:
    """Clip a reference by decoded or coefficient distance from ``current``."""

    q_ref = _finite_vector(reference, "reference")
    q_cur = _finite_vector(current, "current")
    if q_ref.shape != q_cur.shape:
        raise ValueError("reference and current coefficient vectors must have the same shape.")
    delta = q_ref - q_cur
    if metric_basis is None:
        metric_delta = delta
    else:
        basis = np.asarray(metric_basis, dtype=float)
        if basis.ndim != 2 or basis.shape[1] != delta.size:
            raise ValueError("metric_basis must have shape (n_decoded, n_coefficients).")
        metric_delta = basis @ delta
    before = float(np.linalg.norm(metric_delta))
    if max_distance is None:
        return q_ref, False, before, before
    limit = float(max_distance)
    if not np.isfinite(limit) or limit <= 0.0 or before <= limit:
        return q_ref, False, before, before
    clipped = q_cur + (limit / max(before, 1.0e-300)) * delta
    return np.ascontiguousarray(clipped, dtype=np.float64), True, before, limit


@dataclass(frozen=True)
class ReferencePolicyResult:
    """Reference state and native solver options for one ROM time step."""

    coefficients: np.ndarray
    reference_weight: float
    max_reference_distance: float | None
    max_step_norm: float | None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", _finite_vector(self.coefficients, "reference coefficients"))
        object.__setattr__(self, "reference_weight", float(self.reference_weight))
        object.__setattr__(self, "max_reference_distance", None if self.max_reference_distance is None else float(self.max_reference_distance))
        object.__setattr__(self, "max_step_norm", None if self.max_step_norm is None else float(self.max_step_norm))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def native_options(self) -> dict[str, Any]:
        return {
            "reference_coefficients": self.coefficients,
            "reference_weight": float(self.reference_weight),
            "max_reference_distance": self.max_reference_distance,
            "max_step_norm": self.max_step_norm,
        }


@dataclass(frozen=True)
class ReferencePolicy:
    """Problem-generic branch reference policy.

    The policy owns no PDE-specific data.  It combines a reduced-coordinate
    predictor with scalar globalization controls used by native LSPG/GNAT
    solvers.
    """

    predictor: ReducedReferencePredictor | Mapping[str, Any] | None = None
    reference_weight: float = 0.0
    max_reference_distance: float | None = None
    max_step_norm: float | None = None
    metric_basis: np.ndarray | None = None
    clip_reference: bool = True
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        predictor = self.predictor
        if predictor is None:
            predictor = LinearHistoryReducedPredictor()
        elif isinstance(predictor, Mapping):
            predictor = predictor_from_native_dict(predictor)
        if self.metric_basis is not None:
            basis = np.asarray(self.metric_basis, dtype=float)
            if basis.ndim != 2 or not np.all(np.isfinite(basis)):
                raise ValueError("metric_basis must be a finite 2-D array.")
            object.__setattr__(self, "metric_basis", np.ascontiguousarray(basis, dtype=np.float64))
        object.__setattr__(self, "predictor", predictor)
        object.__setattr__(self, "reference_weight", float(self.reference_weight))
        object.__setattr__(self, "max_reference_distance", None if self.max_reference_distance is None else float(self.max_reference_distance))
        object.__setattr__(self, "max_step_norm", None if self.max_step_norm is None else float(self.max_step_norm))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def predict(
        self,
        *,
        time: float,
        dt: float | None = None,
        q_current: Any,
        q_previous: Any | None = None,
        parameters: Mapping[str, Any] | np.ndarray | None = None,
    ) -> ReferencePolicyResult:
        prediction: ReducedReferencePrediction = self.predictor.predict(
            time=float(time),
            dt=dt,
            q_current=q_current,
            q_previous=q_previous,
            parameters=parameters,
        )
        q = prediction.coefficients
        clipped = False
        before = float("nan")
        after = float("nan")
        if self.clip_reference:
            q, clipped, before, after = clip_reference_distance(
                q,
                q_current,
                metric_basis=self.metric_basis,
                max_distance=self.max_reference_distance,
            )
        metadata = {
            "predictor_kind": prediction.predictor_kind,
            "predictor": dict(prediction.metadata or {}),
            "reference_clipped": bool(clipped),
            "reference_distance_before_clip": float(before),
            "reference_distance_after_clip": float(after),
            **dict(self.metadata or {}),
        }
        return ReferencePolicyResult(
            coefficients=q,
            reference_weight=float(self.reference_weight),
            max_reference_distance=self.max_reference_distance,
            max_step_norm=self.max_step_norm,
            metadata=metadata,
        )

    def to_native_dict(self) -> dict[str, Any]:
        return {
            "predictor": self.predictor.to_native_dict(),
            "reference_weight": float(self.reference_weight),
            "max_reference_distance": self.max_reference_distance,
            "max_step_norm": self.max_step_norm,
            "metric_basis": self.metric_basis,
            "clip_reference": bool(self.clip_reference),
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_native_dict(cls, payload: Mapping[str, Any]) -> "ReferencePolicy":
        return cls(
            predictor=payload.get("predictor"),
            reference_weight=float(payload.get("reference_weight", 0.0)),
            max_reference_distance=payload.get("max_reference_distance"),
            max_step_norm=payload.get("max_step_norm"),
            metric_basis=payload.get("metric_basis"),
            clip_reference=bool(payload.get("clip_reference", True)),
            metadata=payload.get("metadata", {}),
        )


__all__ = [
    "ReferencePolicy",
    "ReferencePolicyResult",
    "clip_reference_distance",
]
