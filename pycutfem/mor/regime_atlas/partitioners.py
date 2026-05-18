"""Common partitioner contracts and helpers for regime atlases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .data import RegimeAtlas, RegimeDataset, RegimeRegion, as_feature_matrix
from .features import robust_feature_center_scale, scale_feature_matrix


class RegimePartitioner(Protocol):
    """Protocol implemented by all generic regime partitioners."""

    def fit(self, dataset: RegimeDataset | np.ndarray) -> RegimeAtlas:
        """Fit a regime atlas from a dataset or row-major feature matrix."""


@dataclass(frozen=True)
class RegimePartitionerConfig:
    """Small serializable partitioner factory configuration."""

    kind: str
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RegimePartitionerConfig":
        values = dict(mapping)
        kind = str(values.pop("kind"))
        options = dict(values.pop("options", {}))
        options.update(values)
        return cls(kind=kind, options=options)


def coerce_regime_dataset(data: RegimeDataset | np.ndarray) -> RegimeDataset:
    if isinstance(data, RegimeDataset):
        return data
    return RegimeDataset(features=as_feature_matrix(np.asarray(data, dtype=float)))


def normalize_region_labels(labels: Sequence[int] | np.ndarray, *, outlier_label: int = -1) -> np.ndarray:
    raw = np.asarray(labels, dtype=int).reshape(-1)
    out = np.full(raw.shape, int(outlier_label), dtype=int)
    next_label = 0
    for label in sorted(set(raw.tolist())):
        if int(label) == int(outlier_label):
            continue
        out[raw == int(label)] = next_label
        next_label += 1
    return out


def labels_to_atlas(
    dataset: RegimeDataset | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    outlier_label: int = -1,
    radius_quantile: float = 1.0,
    radius_safety_factor: float = 1.05,
    max_feature_distances: Mapping[int, float] | Sequence[float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RegimeAtlas:
    """Build a :class:`RegimeAtlas` from sample labels."""

    ds = coerce_regime_dataset(dataset)
    raw_labels = np.asarray(labels, dtype=int).reshape(-1)
    if raw_labels.size != ds.n_samples:
        raise ValueError("labels length must match the number of samples.")
    labels_norm = normalize_region_labels(raw_labels, outlier_label=outlier_label)
    if not 0.0 < float(radius_quantile) <= 1.0:
        raise ValueError("radius_quantile must lie in (0, 1].")
    if float(radius_safety_factor) <= 0.0:
        raise ValueError("radius_safety_factor must be positive.")
    global_center, global_scale = robust_feature_center_scale(ds.features)
    regions: list[RegimeRegion] = []
    if isinstance(max_feature_distances, Mapping):
        radius_map = {int(k): float(v) for k, v in max_feature_distances.items()}
    elif max_feature_distances is None:
        radius_map = {}
    else:
        radius_map = {i: float(value) for i, value in enumerate(max_feature_distances)}

    for label in sorted(set(labels_norm.tolist())):
        if int(label) == int(outlier_label):
            continue
        indices = np.flatnonzero(labels_norm == int(label))
        if indices.size == 0:
            continue
        center = ds.features[indices, :].mean(axis=0)
        distances = np.linalg.norm(
            scale_feature_matrix(ds.features[indices, :], center=center, scale=global_scale),
            axis=1,
        )
        radius = radius_map.get(
            int(label),
            float(np.quantile(distances, float(radius_quantile)) * float(radius_safety_factor)),
        )
        radius = max(float(radius), 1.0e-12)
        regions.append(
            RegimeRegion(
                region_id=f"region_{int(label):03d}",
                index=int(label),
                sample_indices=indices.astype(int),
                feature_center=np.asarray(center, dtype=float),
                feature_scale=np.asarray(global_scale, dtype=float),
                max_feature_distance=radius,
                metadata={
                    "mean_feature_distance": float(np.mean(distances)) if distances.size else 0.0,
                    "max_training_feature_distance": float(np.max(distances)) if distances.size else 0.0,
                },
            )
        )
    return RegimeAtlas(
        regions=tuple(regions),
        labels=labels_norm,
        feature_names=tuple(ds.feature_names or ()),
        global_center=global_center,
        global_scale=global_scale,
        outlier_label=int(outlier_label),
        metadata=dict(metadata or {}),
    )


def make_regime_partitioner(config: RegimePartitionerConfig | Mapping[str, Any] | str) -> RegimePartitioner:
    """Instantiate a partitioner from a small serializable configuration."""

    if isinstance(config, str):
        cfg = RegimePartitionerConfig(kind=config)
    elif isinstance(config, RegimePartitionerConfig):
        cfg = config
    else:
        cfg = RegimePartitionerConfig.from_mapping(config)
    kind = str(cfg.kind).strip().lower().replace("-", "_")
    options = dict(cfg.options)

    if kind in {"kmedoids", "k_medoids", "feature_atlas"}:
        from .kmedoids import KMedoidsPartitioner

        return KMedoidsPartitioner(**options)
    if kind in {"epsilon_cover", "cover", "greedy_cover"}:
        from .cover import EpsilonCoverPartitioner

        return EpsilonCoverPartitioner(**options)
    if kind in {"hierarchical", "agglomerative"}:
        from .hierarchical import HierarchicalPartitioner

        return HierarchicalPartitioner(**options)
    if kind in {"density", "dbscan"}:
        from .density import DensityPartitioner

        return DensityPartitioner(**options)
    if kind in {"mixture", "gmm", "gaussian_mixture"}:
        from .mixture import MixturePartitioner

        return MixturePartitioner(**options)
    raise ValueError(f"unsupported regime partitioner kind: {cfg.kind!r}")


__all__ = [
    "RegimePartitioner",
    "RegimePartitionerConfig",
    "coerce_regime_dataset",
    "labels_to_atlas",
    "make_regime_partitioner",
    "normalize_region_labels",
]
