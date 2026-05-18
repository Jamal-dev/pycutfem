"""Diagonal Gaussian-mixture partitioner for regime atlases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .data import RegimeAtlas, RegimeDataset
from .features import fit_k_medoids, robust_feature_center_scale, scale_feature_matrix
from .partitioners import coerce_regime_dataset, labels_to_atlas


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    vmax = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(vmax, axis=axis) + np.log(np.sum(np.exp(values - vmax), axis=axis))


@dataclass(frozen=True)
class MixturePartitioner:
    """Small diagonal Gaussian mixture with uncertainty-aware outliers."""

    n_components: int
    max_iterations: int = 100
    covariance_floor: float = 1.0e-8
    min_probability: float = 0.0
    radius_quantile: float = 1.0
    radius_safety_factor: float = 1.05
    metadata: Mapping[str, Any] | None = None

    def fit(self, dataset: RegimeDataset | np.ndarray) -> RegimeAtlas:
        ds = coerce_regime_dataset(dataset)
        k = int(self.n_components)
        if k < 1 or k > ds.n_samples:
            raise ValueError("n_components must lie between one and the number of samples.")
        cov_floor = max(float(self.covariance_floor), 1.0e-300)
        min_probability = float(np.clip(float(self.min_probability), 0.0, 1.0))
        center, scale = robust_feature_center_scale(ds.features)
        scaled = scale_feature_matrix(ds.features, center=center, scale=scale)
        init = fit_k_medoids(scaled, n_clusters=k, sample_weights=ds.weights)
        means = scaled[init.medoid_indices, :].copy()
        variances = np.tile(np.var(scaled, axis=0) + cov_floor, (k, 1))
        weights = np.full(k, 1.0 / k, dtype=float)
        responsibilities = np.full((ds.n_samples, k), 1.0 / k, dtype=float)
        for _ in range(max(1, int(self.max_iterations))):
            log_prob = np.empty((ds.n_samples, k), dtype=float)
            for j in range(k):
                var = np.maximum(variances[j, :], cov_floor)
                diff = scaled - means[j, :][None, :]
                log_det = float(np.sum(np.log(2.0 * np.pi * var)))
                log_prob[:, j] = np.log(max(weights[j], 1.0e-300)) - 0.5 * (
                    np.sum((diff * diff) / var[None, :], axis=1) + log_det
                )
            normalizer = _logsumexp(log_prob, axis=1)
            new_resp = np.exp(log_prob - normalizer[:, None])
            counts = np.maximum(new_resp.sum(axis=0), 1.0e-300)
            weights = counts / float(ds.n_samples)
            means = (new_resp.T @ scaled) / counts[:, None]
            for j in range(k):
                diff = scaled - means[j, :][None, :]
                variances[j, :] = (new_resp[:, j][:, None] * diff * diff).sum(axis=0) / counts[j]
            if np.linalg.norm(new_resp - responsibilities) <= 1.0e-10:
                responsibilities = new_resp
                break
            responsibilities = new_resp
        probabilities = np.max(responsibilities, axis=1)
        labels = np.argmax(responsibilities, axis=1).astype(int)
        labels[probabilities < min_probability] = -1
        return labels_to_atlas(
            ds,
            labels,
            radius_quantile=float(self.radius_quantile),
            radius_safety_factor=float(self.radius_safety_factor),
            metadata={
                "partitioner": "mixture",
                "weights": weights.astype(float).tolist(),
                "means": means.astype(float).tolist(),
                "variances": variances.astype(float).tolist(),
                "min_probability": min_probability,
                "min_assigned_probability": float(np.min(probabilities)) if probabilities.size else 0.0,
                **dict(self.metadata or {}),
            },
        )


__all__ = ["MixturePartitioner"]
