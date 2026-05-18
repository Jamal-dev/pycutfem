"""Dependency-free agglomerative partitioning for regime atlases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .data import RegimeAtlas, RegimeDataset
from .features import robust_feature_center_scale, scale_feature_matrix
from .partitioners import coerce_regime_dataset, labels_to_atlas


def _cluster_distance(matrix: np.ndarray, left: list[int], right: list[int], linkage: str) -> float:
    diff = matrix[np.asarray(left), None, :] - matrix[None, np.asarray(right), :]
    distances = np.sqrt(np.einsum("ijd,ijd->ij", diff, diff))
    if linkage == "single":
        return float(np.min(distances))
    if linkage == "complete":
        return float(np.max(distances))
    if linkage == "average":
        return float(np.mean(distances))
    raise ValueError(f"unsupported linkage: {linkage!r}")


@dataclass(frozen=True)
class HierarchicalPartitioner:
    """Small agglomerative partitioner for offline regime discovery."""

    n_regions: int
    linkage: str = "average"
    radius_quantile: float = 1.0
    radius_safety_factor: float = 1.05
    metadata: Mapping[str, Any] | None = None

    def fit(self, dataset: RegimeDataset | np.ndarray) -> RegimeAtlas:
        ds = coerce_regime_dataset(dataset)
        k = int(self.n_regions)
        if k < 1 or k > ds.n_samples:
            raise ValueError("n_regions must lie between one and the number of samples.")
        linkage = str(self.linkage).lower()
        if linkage not in {"single", "complete", "average"}:
            raise ValueError("linkage must be 'single', 'complete', or 'average'.")
        center, scale = robust_feature_center_scale(ds.features)
        scaled = scale_feature_matrix(ds.features, center=center, scale=scale)
        clusters: list[list[int]] = [[i] for i in range(ds.n_samples)]
        while len(clusters) > k:
            best_pair = (0, 1)
            best_distance = float("inf")
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distance = _cluster_distance(scaled, clusters[i], clusters[j], linkage)
                    if distance < best_distance:
                        best_distance = distance
                        best_pair = (i, j)
            i, j = best_pair
            clusters[i] = clusters[i] + clusters[j]
            del clusters[j]
        labels = np.empty(ds.n_samples, dtype=int)
        for label, indices in enumerate(clusters):
            labels[np.asarray(indices, dtype=int)] = label
        return labels_to_atlas(
            ds,
            labels,
            radius_quantile=float(self.radius_quantile),
            radius_safety_factor=float(self.radius_safety_factor),
            metadata={"partitioner": "hierarchical", "linkage": linkage, **dict(self.metadata or {})},
        )


__all__ = ["HierarchicalPartitioner"]
