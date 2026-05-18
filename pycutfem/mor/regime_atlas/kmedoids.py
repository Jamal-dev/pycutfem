"""K-medoids partitioner for nonlinear-regime atlases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .data import RegimeAtlas, RegimeDataset
from .features import fit_k_medoids, robust_feature_center_scale, scale_feature_matrix
from .partitioners import coerce_regime_dataset, labels_to_atlas


@dataclass(frozen=True)
class KMedoidsPartitioner:
    """Deterministic weighted k-medoids regime partitioner."""

    n_regions: int
    radius_quantile: float = 1.0
    radius_safety_factor: float = 1.05
    max_iterations: int = 100
    metadata: Mapping[str, Any] | None = None

    def fit(self, dataset: RegimeDataset | np.ndarray) -> RegimeAtlas:
        ds = coerce_regime_dataset(dataset)
        center, scale = robust_feature_center_scale(ds.features)
        scaled = scale_feature_matrix(ds.features, center=center, scale=scale)
        result = fit_k_medoids(
            scaled,
            n_clusters=int(self.n_regions),
            sample_weights=ds.weights,
            max_iterations=int(self.max_iterations),
        )
        atlas = labels_to_atlas(
            ds,
            result.labels,
            radius_quantile=float(self.radius_quantile),
            radius_safety_factor=float(self.radius_safety_factor),
            metadata={
                "partitioner": "kmedoids",
                "inertia": float(result.inertia),
                "iterations": int(result.iterations),
                "medoid_indices": result.medoid_indices.astype(int).tolist(),
                **dict(self.metadata or {}),
            },
        )
        return atlas


def fit_kmedoids_regime_atlas(
    features: np.ndarray,
    *,
    n_regions: int,
    feature_names: Sequence[str] | None = None,
    **kwargs: Any,
) -> RegimeAtlas:
    dataset = RegimeDataset(features=features, feature_names=None if feature_names is None else tuple(feature_names))
    return KMedoidsPartitioner(n_regions=int(n_regions), **kwargs).fit(dataset)


__all__ = [
    "KMedoidsPartitioner",
    "fit_kmedoids_regime_atlas",
]
