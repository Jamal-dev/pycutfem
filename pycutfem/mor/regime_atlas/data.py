"""Core data contracts for nonlinear-regime atlas construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


def as_feature_matrix(features: np.ndarray, *, name: str = "features") -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a row-major 1D or 2D feature matrix.")
    if matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ValueError(f"{name} must contain at least one sample and one feature.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.asarray(matrix, dtype=float)


def as_index_vector(values: Sequence[int] | np.ndarray, *, name: str = "indices") -> np.ndarray:
    indices = np.asarray(values, dtype=int).reshape(-1)
    if indices.size == 0:
        raise ValueError(f"{name} must contain at least one index.")
    return indices


def _optional_sample_vector(values: Any | None, n_samples: int, *, name: str, dtype: Any) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if arr.size != int(n_samples):
        raise ValueError(f"{name} length must match the number of samples.")
    return arr


@dataclass(frozen=True)
class RegimeDataset:
    """Problem-generic row-major feature dataset for regime discovery."""

    features: np.ndarray
    feature_names: tuple[str, ...] | None = None
    sample_ids: np.ndarray | None = None
    groups: np.ndarray | None = None
    steps: np.ndarray | None = None
    weights: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        features = as_feature_matrix(self.features)
        object.__setattr__(self, "features", features)
        n_samples, n_features = features.shape
        if self.feature_names is None:
            names = tuple(f"feature_{i}" for i in range(n_features))
        else:
            names = tuple(str(name) for name in self.feature_names)
            if len(names) != n_features:
                raise ValueError("feature_names length must match the number of features.")
        object.__setattr__(self, "feature_names", names)
        sample_ids = (
            np.arange(n_samples, dtype=int)
            if self.sample_ids is None
            else _optional_sample_vector(self.sample_ids, n_samples, name="sample_ids", dtype=int)
        )
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(
            self,
            "groups",
            _optional_sample_vector(self.groups, n_samples, name="groups", dtype=object),
        )
        object.__setattr__(
            self,
            "steps",
            _optional_sample_vector(self.steps, n_samples, name="steps", dtype=int),
        )
        weights = _optional_sample_vector(self.weights, n_samples, name="weights", dtype=float)
        if weights is not None:
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("weights must be finite and nonnegative.")
            if float(weights.sum()) <= 0.0:
                raise ValueError("at least one sample weight must be positive.")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    def subset(self, indices: Sequence[int] | np.ndarray) -> "RegimeDataset":
        idx = np.asarray(indices, dtype=int).reshape(-1)
        return RegimeDataset(
            features=self.features[idx, :],
            feature_names=self.feature_names,
            sample_ids=self.sample_ids[idx],
            groups=None if self.groups is None else self.groups[idx],
            steps=None if self.steps is None else self.steps[idx],
            weights=None if self.weights is None else self.weights[idx],
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class RegimeValidationSplit:
    """Train/validation split for atlas validation."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    group_held_out: bool = False
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        train = np.asarray(self.train_indices, dtype=int).reshape(-1)
        validation = np.asarray(self.validation_indices, dtype=int).reshape(-1)
        if train.size == 0 or validation.size == 0:
            raise ValueError("train_indices and validation_indices must be nonempty.")
        if set(train.tolist()).intersection(validation.tolist()):
            raise ValueError("train and validation indices must be disjoint.")
        object.__setattr__(self, "train_indices", train)
        object.__setattr__(self, "validation_indices", validation)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class RegimeRegion:
    """One generic regime region in feature space."""

    region_id: str
    index: int
    sample_indices: np.ndarray
    feature_center: np.ndarray
    feature_scale: np.ndarray
    max_feature_distance: float
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        indices = np.asarray(self.sample_indices, dtype=int).reshape(-1)
        if indices.size == 0:
            raise ValueError("region sample_indices must be nonempty.")
        center = np.asarray(self.feature_center, dtype=float).reshape(-1)
        scale = np.asarray(self.feature_scale, dtype=float).reshape(-1)
        if center.size == 0 or scale.size != center.size:
            raise ValueError("feature_center and feature_scale have incompatible sizes.")
        if not np.all(np.isfinite(center)):
            raise ValueError("feature_center must be finite.")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("feature_scale must be finite and positive.")
        radius = float(self.max_feature_distance)
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("max_feature_distance must be finite and nonnegative.")
        object.__setattr__(self, "region_id", str(self.region_id))
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "sample_indices", indices)
        object.__setattr__(self, "feature_center", center)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "max_feature_distance", radius)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def support_count(self) -> int:
        return int(self.sample_indices.size)

    def distance(self, feature: Sequence[float] | np.ndarray) -> float:
        values = np.asarray(feature, dtype=float).reshape(-1)
        if values.size != self.feature_center.size:
            return float("inf")
        return float(np.linalg.norm((values - self.feature_center) / self.feature_scale))

    def contains(self, feature: Sequence[float] | np.ndarray) -> bool:
        return bool(self.distance(feature) <= float(self.max_feature_distance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.region_id,
            "index": int(self.index),
            "sample_indices": self.sample_indices.astype(int).tolist(),
            "support_count": int(self.support_count),
            "feature_center": self.feature_center.astype(float).tolist(),
            "feature_scale": self.feature_scale.astype(float).tolist(),
            "max_feature_distance": float(self.max_feature_distance),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class RegimeAtlas:
    """Generic atlas of certified or candidate nonlinear regimes."""

    regions: tuple[RegimeRegion, ...]
    labels: np.ndarray
    feature_names: tuple[str, ...]
    global_center: np.ndarray
    global_scale: np.ndarray
    outlier_label: int = -1
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels, dtype=int).reshape(-1)
        if labels.size == 0:
            raise ValueError("labels must be nonempty.")
        names = tuple(str(name) for name in self.feature_names)
        center = np.asarray(self.global_center, dtype=float).reshape(-1)
        scale = np.asarray(self.global_scale, dtype=float).reshape(-1)
        if center.size != len(names) or scale.size != len(names):
            raise ValueError("global_center/global_scale sizes must match feature_names.")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("global_scale must be finite and positive.")
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "global_center", center)
        object.__setattr__(self, "global_scale", scale)
        object.__setattr__(self, "outlier_label", int(self.outlier_label))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_regions(self) -> int:
        return int(len(self.regions))

    @property
    def outlier_indices(self) -> np.ndarray:
        return np.flatnonzero(self.labels == int(self.outlier_label)).astype(int)

    @property
    def support_counts(self) -> np.ndarray:
        return np.asarray([region.support_count for region in self.regions], dtype=int)

    @property
    def coverage(self) -> float:
        if self.labels.size == 0:
            return 0.0
        return float(np.count_nonzero(self.labels != int(self.outlier_label)) / self.labels.size)

    def region_for_feature(self, feature: Sequence[float] | np.ndarray) -> tuple[RegimeRegion, float] | None:
        if not self.regions:
            return None
        distances = np.asarray([region.distance(feature) for region in self.regions], dtype=float)
        index = int(np.argmin(distances))
        if not np.isfinite(distances[index]):
            return None
        region = self.regions[index]
        if distances[index] > float(region.max_feature_distance):
            return None
        return region, float(distances[index])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "n_regions": int(self.n_regions),
            "feature_names": list(self.feature_names),
            "global_center": self.global_center.astype(float).tolist(),
            "global_scale": self.global_scale.astype(float).tolist(),
            "outlier_label": int(self.outlier_label),
            "coverage": float(self.coverage),
            "labels": self.labels.astype(int).tolist(),
            "regions": [region.to_dict() for region in self.regions],
            "metadata": dict(self.metadata or {}),
        }


def make_validation_split(
    dataset: RegimeDataset,
    *,
    test_fraction: float = 0.2,
    random_state: int = 0,
    groups: Sequence[Any] | np.ndarray | None = None,
) -> RegimeValidationSplit:
    """Create a random sample split or grouped holdout split."""

    if not 0.0 < float(test_fraction) < 1.0:
        raise ValueError("test_fraction must lie in (0, 1).")
    rng = np.random.default_rng(int(random_state))
    group_values = dataset.groups if groups is None else np.asarray(groups, dtype=object).reshape(-1)
    if group_values is not None:
        if group_values.size != dataset.n_samples:
            raise ValueError("groups length must match the number of samples.")
        unique = np.asarray(sorted(set(group_values.tolist())), dtype=object)
        rng.shuffle(unique)
        n_validation_groups = max(1, int(round(float(test_fraction) * unique.size)))
        validation_groups = set(unique[:n_validation_groups].tolist())
        validation = np.asarray(
            [i for i, value in enumerate(group_values.tolist()) if value in validation_groups],
            dtype=int,
        )
        train = np.asarray(
            [i for i, value in enumerate(group_values.tolist()) if value not in validation_groups],
            dtype=int,
        )
        if train.size == 0 or validation.size == 0:
            raise ValueError("grouped validation split produced an empty side.")
        return RegimeValidationSplit(
            train_indices=train,
            validation_indices=validation,
            group_held_out=True,
            metadata={"validation_groups": sorted(str(value) for value in validation_groups)},
        )

    permutation = rng.permutation(dataset.n_samples)
    n_validation = max(1, int(round(float(test_fraction) * dataset.n_samples)))
    validation = permutation[:n_validation]
    train = permutation[n_validation:]
    if train.size == 0:
        raise ValueError("validation split left no training samples.")
    return RegimeValidationSplit(train_indices=train, validation_indices=validation, group_held_out=False)


__all__ = [
    "RegimeAtlas",
    "RegimeDataset",
    "RegimeRegion",
    "RegimeValidationSplit",
    "as_feature_matrix",
    "as_index_vector",
    "make_validation_split",
]
