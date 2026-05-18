"""Validation helpers for nonlinear-regime atlases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .data import RegimeAtlas, RegimeDataset, RegimeValidationSplit, make_validation_split


@dataclass(frozen=True)
class RegimeValidationReport:
    """Validation result for one region or one whole atlas."""

    region_id: str
    error: float
    passed: bool
    train_indices: np.ndarray | None = None
    validation_indices: np.ndarray | None = None
    metrics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        error = float(self.error)
        if not np.isfinite(error) or error < 0.0:
            raise ValueError("validation error must be finite and nonnegative.")
        object.__setattr__(self, "region_id", str(self.region_id))
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(
            self,
            "train_indices",
            None if self.train_indices is None else np.asarray(self.train_indices, dtype=int).reshape(-1),
        )
        object.__setattr__(
            self,
            "validation_indices",
            None if self.validation_indices is None else np.asarray(self.validation_indices, dtype=int).reshape(-1),
        )
        object.__setattr__(self, "metrics", dict(self.metrics or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "error": float(self.error),
            "passed": bool(self.passed),
            "train_indices": None if self.train_indices is None else self.train_indices.astype(int).tolist(),
            "validation_indices": None
            if self.validation_indices is None
            else self.validation_indices.astype(int).tolist(),
            "metrics": dict(self.metrics or {}),
        }


@dataclass(frozen=True)
class RegimeValidationSummary:
    """Aggregate validation reports for an atlas."""

    reports: tuple[RegimeValidationReport, ...]
    boundary_error: float = 0.0
    fallback_rate: float = 0.0
    metadata: Mapping[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return bool(all(report.passed for report in self.reports))

    @property
    def max_error(self) -> float:
        return 0.0 if not self.reports else float(max(report.error for report in self.reports))

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "max_error": float(self.max_error),
            "boundary_error": float(self.boundary_error),
            "fallback_rate": float(self.fallback_rate),
            "reports": [report.to_dict() for report in self.reports],
            "metadata": dict(self.metadata or {}),
        }


def summarize_region_errors(
    atlas: RegimeAtlas,
    errors: Mapping[str, float] | Sequence[float],
    *,
    tolerance: float,
) -> RegimeValidationSummary:
    """Create a summary from precomputed region errors."""

    if isinstance(errors, Mapping):
        error_map = {str(key): float(value) for key, value in errors.items()}
        reports = tuple(
            RegimeValidationReport(
                region_id=region.region_id,
                error=float(error_map.get(region.region_id, float("inf"))),
                passed=float(error_map.get(region.region_id, float("inf"))) <= float(tolerance),
                validation_indices=region.sample_indices,
            )
            for region in atlas.regions
        )
    else:
        values = np.asarray(errors, dtype=float).reshape(-1)
        if values.size != atlas.n_regions:
            raise ValueError("error sequence length must match atlas.n_regions.")
        reports = tuple(
            RegimeValidationReport(
                region_id=region.region_id,
                error=float(values[i]),
                passed=float(values[i]) <= float(tolerance),
                validation_indices=region.sample_indices,
            )
            for i, region in enumerate(atlas.regions)
        )
    return RegimeValidationSummary(reports=reports)


def boundary_halo_score(atlas: RegimeAtlas, dataset: RegimeDataset, *, quantile: float = 0.1) -> float:
    """Return a simple feature-boundary score based on nearest competing region."""

    if atlas.n_regions <= 1:
        return 0.0
    values: list[float] = []
    for i, feature in enumerate(dataset.features):
        label = int(atlas.labels[i])
        if label == int(atlas.outlier_label) or label >= atlas.n_regions:
            continue
        distances = np.asarray([region.distance(feature) for region in atlas.regions], dtype=float)
        own = float(distances[label])
        distances[label] = float("inf")
        other = float(np.min(distances))
        if np.isfinite(other):
            values.append(other - own)
    if not values:
        return 0.0
    margins = np.asarray(values, dtype=float)
    q = float(np.clip(float(quantile), 0.0, 1.0))
    return float(np.quantile(margins, q))


__all__ = [
    "RegimeValidationReport",
    "RegimeValidationSplit",
    "RegimeValidationSummary",
    "boundary_halo_score",
    "make_validation_split",
    "summarize_region_errors",
]
