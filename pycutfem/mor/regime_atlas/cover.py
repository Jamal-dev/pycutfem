"""Epsilon-cover partitioner for nonlinear-regime atlases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .data import RegimeAtlas, RegimeDataset
from .features import robust_feature_center_scale, scale_feature_matrix
from .partitioners import coerce_regime_dataset, labels_to_atlas


@dataclass(frozen=True)
class EpsilonCoverPartitioner:
    """Greedy farthest-point cover with an explicit feature radius."""

    epsilon: float
    max_regions: int | None = None
    radius_safety_factor: float = 1.0
    metadata: Mapping[str, Any] | None = None

    def fit(self, dataset: RegimeDataset | np.ndarray) -> RegimeAtlas:
        ds = coerce_regime_dataset(dataset)
        epsilon = float(self.epsilon)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive.")
        max_regions = ds.n_samples if self.max_regions is None else int(self.max_regions)
        if max_regions < 1:
            raise ValueError("max_regions must be at least one.")
        center, scale = robust_feature_center_scale(ds.features)
        scaled = scale_feature_matrix(ds.features, center=center, scale=scale)

        first = int(np.argmin(np.sum((scaled - scaled.mean(axis=0)[None, :]) ** 2, axis=1)))
        centers = [first]
        labels = np.zeros(ds.n_samples, dtype=int)
        distances = np.linalg.norm(scaled - scaled[first, :][None, :], axis=1)
        while float(np.max(distances)) > epsilon and len(centers) < max_regions:
            next_index = int(np.argmax(distances))
            centers.append(next_index)
            all_distances = np.column_stack(
                [np.linalg.norm(scaled - scaled[index, :][None, :], axis=1) for index in centers]
            )
            labels = np.argmin(all_distances, axis=1).astype(int)
            distances = all_distances[np.arange(ds.n_samples), labels]

        all_distances = np.column_stack(
            [np.linalg.norm(scaled - scaled[index, :][None, :], axis=1) for index in centers]
        )
        labels = np.argmin(all_distances, axis=1).astype(int)
        radius = float(epsilon * float(self.radius_safety_factor))
        return labels_to_atlas(
            ds,
            labels,
            max_feature_distances=[radius for _ in centers],
            metadata={
                "partitioner": "epsilon_cover",
                "epsilon": epsilon,
                "center_indices": [int(index) for index in centers],
                **dict(self.metadata or {}),
            },
        )


__all__ = ["EpsilonCoverPartitioner"]
