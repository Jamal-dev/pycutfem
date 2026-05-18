"""Density/outlier partitioning for regime atlases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .data import RegimeAtlas, RegimeDataset
from .features import robust_feature_center_scale, scale_feature_matrix
from .partitioners import coerce_regime_dataset, labels_to_atlas


@dataclass(frozen=True)
class DensityPartitioner:
    """Small DBSCAN-style density partitioner with outlier label ``-1``."""

    eps: float
    min_samples: int = 5
    radius_quantile: float = 1.0
    radius_safety_factor: float = 1.05
    metadata: Mapping[str, Any] | None = None

    def fit(self, dataset: RegimeDataset | np.ndarray) -> RegimeAtlas:
        ds = coerce_regime_dataset(dataset)
        eps = float(self.eps)
        min_samples = int(self.min_samples)
        if not np.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be finite and positive.")
        if min_samples < 1:
            raise ValueError("min_samples must be at least one.")
        center, scale = robust_feature_center_scale(ds.features)
        scaled = scale_feature_matrix(ds.features, center=center, scale=scale)
        diff = scaled[:, None, :] - scaled[None, :, :]
        distances = np.sqrt(np.einsum("ijd,ijd->ij", diff, diff))
        neighbors = [np.flatnonzero(distances[i, :] <= eps).astype(int) for i in range(ds.n_samples)]
        labels = np.full(ds.n_samples, -1, dtype=int)
        visited = np.zeros(ds.n_samples, dtype=bool)
        cluster_id = 0
        for i in range(ds.n_samples):
            if visited[i]:
                continue
            visited[i] = True
            if neighbors[i].size < min_samples:
                continue
            labels[i] = cluster_id
            seeds = list(int(value) for value in neighbors[i].tolist())
            cursor = 0
            while cursor < len(seeds):
                j = seeds[cursor]
                cursor += 1
                if not visited[j]:
                    visited[j] = True
                    if neighbors[j].size >= min_samples:
                        for value in neighbors[j].tolist():
                            if int(value) not in seeds:
                                seeds.append(int(value))
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1
        return labels_to_atlas(
            ds,
            labels,
            radius_quantile=float(self.radius_quantile),
            radius_safety_factor=float(self.radius_safety_factor),
            metadata={
                "partitioner": "density",
                "eps": eps,
                "min_samples": min_samples,
                "outlier_count": int(np.count_nonzero(labels == -1)),
                **dict(self.metadata or {}),
            },
        )


__all__ = ["DensityPartitioner"]
