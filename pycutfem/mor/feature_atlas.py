"""Feature-based local reduced-model atlas utilities.

The atlas is an offline design tool for nonlinear ROM/HROM deployments.  It
groups sampled stages by cheap regime features instead of arbitrary time
intervals, then emits local-bank manifest metadata that the online driver can
use through :mod:`pycutfem.mor.local_banks`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .local_banks import LocalReducedModelBankEntry


def _as_feature_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("features must be a 1D or 2D array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must contain at least one sample and one feature")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must be finite")
    return matrix


def _as_steps(steps: Sequence[int] | np.ndarray | None, n_samples: int) -> np.ndarray | None:
    if steps is None:
        return None
    values = np.asarray(steps, dtype=int).reshape(-1)
    if values.size != n_samples:
        raise ValueError("steps length must match the number of feature samples")
    return values


def robust_feature_center_scale(
    features: np.ndarray,
    *,
    eps: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return median/IQR feature scaling with safe fallbacks.

    The returned arrays have shape ``(n_features,)`` and operate on row-major
    feature matrices: ``(features - center) / scale``.
    """

    matrix = _as_feature_matrix(features)
    center = np.median(matrix, axis=0)
    q75 = np.percentile(matrix, 75.0, axis=0)
    q25 = np.percentile(matrix, 25.0, axis=0)
    scale = q75 - q25
    std = matrix.std(axis=0)
    scale = np.where(scale > eps, scale, std)
    scale = np.where(scale > eps, scale, 1.0)
    return center.astype(float), scale.astype(float)


def scale_feature_matrix(
    features: np.ndarray,
    *,
    center: Sequence[float] | np.ndarray,
    scale: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Scale a row-major feature matrix with finite positive scales."""

    matrix = _as_feature_matrix(features)
    center_arr = np.asarray(center, dtype=float).reshape(-1)
    scale_arr = np.asarray(scale, dtype=float).reshape(-1)
    if center_arr.size != matrix.shape[1] or scale_arr.size != matrix.shape[1]:
        raise ValueError("center and scale lengths must match the number of features")
    if not np.all(np.isfinite(center_arr)):
        raise ValueError("feature center must be finite")
    if not np.all(np.isfinite(scale_arr)) or np.any(scale_arr <= 0.0):
        raise ValueError("feature scale must be finite and positive")
    return (matrix - center_arr[None, :]) / np.maximum(scale_arr[None, :], 1.0e-300)


@dataclass(frozen=True)
class KMedoidsResult:
    """Result of deterministic weighted k-medoids clustering."""

    labels: np.ndarray
    medoid_indices: np.ndarray
    inertia: float
    iterations: int

    @property
    def n_clusters(self) -> int:
        return int(self.medoid_indices.size)


def _squared_distances_to_medoids(features: np.ndarray, medoid_indices: np.ndarray) -> np.ndarray:
    medoids = features[np.asarray(medoid_indices, dtype=int), :]
    diff = features[:, None, :] - medoids[None, :, :]
    return np.einsum("nkd,nkd->nk", diff, diff)


def _initial_medoids(features: np.ndarray, n_clusters: int) -> np.ndarray:
    mean = features.mean(axis=0)
    first = int(np.argmin(np.sum((features - mean[None, :]) ** 2, axis=1)))
    medoids = [first]
    nearest = np.sum((features - features[first, :][None, :]) ** 2, axis=1)
    for _ in range(1, int(n_clusters)):
        nearest[np.asarray(medoids, dtype=int)] = -1.0
        next_idx = int(np.argmax(nearest))
        medoids.append(next_idx)
        candidate = np.sum((features - features[next_idx, :][None, :]) ** 2, axis=1)
        nearest = np.minimum(nearest, candidate)
    return np.asarray(medoids, dtype=int)


def fit_k_medoids(
    features: np.ndarray,
    *,
    n_clusters: int,
    sample_weights: Sequence[float] | np.ndarray | None = None,
    max_iterations: int = 100,
) -> KMedoidsResult:
    """Cluster row-major features with deterministic k-medoids.

    This implementation is deliberately dependency-free.  It is intended for
    offline ROM regime discovery where sample counts are usually in the
    hundreds or low thousands.
    """

    matrix = _as_feature_matrix(features)
    n_samples = matrix.shape[0]
    k = int(n_clusters)
    if k < 1:
        raise ValueError("n_clusters must be at least one")
    if k > n_samples:
        raise ValueError("n_clusters cannot exceed the number of samples")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least one")
    if sample_weights is None:
        weights = np.ones(n_samples, dtype=float)
    else:
        weights = np.asarray(sample_weights, dtype=float).reshape(-1)
        if weights.size != n_samples:
            raise ValueError("sample_weights length must match the number of samples")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("sample_weights must be finite and nonnegative")
        if float(weights.sum()) <= 0.0:
            raise ValueError("at least one sample weight must be positive")

    medoids = _initial_medoids(matrix, k)
    labels = np.zeros(n_samples, dtype=int)
    inertia = float("inf")
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        distances = _squared_distances_to_medoids(matrix, medoids)
        labels = np.argmin(distances, axis=1).astype(int)
        nearest = distances[np.arange(n_samples), labels]
        inertia = float(np.sum(weights * nearest))
        updated = medoids.copy()
        for cluster in range(k):
            cluster_indices = np.flatnonzero(labels == cluster)
            if cluster_indices.size == 0:
                farthest = int(np.argmax(nearest))
                updated[cluster] = farthest
                continue
            cluster_features = matrix[cluster_indices, :]
            pair_diff = cluster_features[:, None, :] - cluster_features[None, :, :]
            pair_dist = np.einsum("ijd,ijd->ij", pair_diff, pair_diff)
            cluster_weights = weights[cluster_indices]
            costs = pair_dist @ cluster_weights
            updated[cluster] = int(cluster_indices[int(np.argmin(costs))])
        if np.array_equal(updated, medoids):
            break
        medoids = updated

    distances = _squared_distances_to_medoids(matrix, medoids)
    labels = np.argmin(distances, axis=1).astype(int)
    nearest = distances[np.arange(n_samples), labels]
    inertia = float(np.sum(weights * nearest))
    return KMedoidsResult(labels=labels, medoid_indices=medoids, inertia=inertia, iterations=iterations)


@dataclass(frozen=True)
class FeatureAtlasRegion:
    """One local regime in a feature-based reduced-model atlas."""

    region_id: str
    index: int
    sample_indices: np.ndarray
    medoid_index: int
    feature_center: np.ndarray
    feature_scale: np.ndarray
    max_feature_distance: float
    mean_distance: float
    max_training_distance: float
    step_start: int | None = None
    step_end: int | None = None
    metadata: Mapping[str, Any] | None = None

    @property
    def support_count(self) -> int:
        return int(np.asarray(self.sample_indices, dtype=int).size)

    def distance(self, feature: Sequence[float] | np.ndarray) -> float:
        values = np.asarray(feature, dtype=float).reshape(-1)
        center = np.asarray(self.feature_center, dtype=float).reshape(-1)
        scale = np.asarray(self.feature_scale, dtype=float).reshape(-1)
        if values.size != center.size:
            return float("inf")
        return float(np.linalg.norm((values - center) / np.maximum(scale, 1.0e-300)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.region_id),
            "index": int(self.index),
            "support_count": int(self.support_count),
            "sample_indices": np.asarray(self.sample_indices, dtype=int).tolist(),
            "medoid_index": int(self.medoid_index),
            "feature_center": np.asarray(self.feature_center, dtype=float).tolist(),
            "feature_scale": np.asarray(self.feature_scale, dtype=float).tolist(),
            "max_feature_distance": float(self.max_feature_distance),
            "mean_distance": float(self.mean_distance),
            "max_training_distance": float(self.max_training_distance),
            "step_start": None if self.step_start is None else int(self.step_start),
            "step_end": None if self.step_end is None else int(self.step_end),
            "metadata": dict(self.metadata or {}),
        }

    def to_bank_entry(
        self,
        *,
        path: str | Path,
        priority: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LocalReducedModelBankEntry:
        payload = dict(self.metadata or {})
        payload.update(dict(metadata or {}))
        payload.update(
            {
                "atlas_region_index": int(self.index),
                "atlas_support_count": int(self.support_count),
                "atlas_mean_feature_distance": float(self.mean_distance),
                "atlas_max_training_feature_distance": float(self.max_training_distance),
            }
        )
        return LocalReducedModelBankEntry(
            model_id=str(self.region_id),
            path=Path(path),
            step_start=1 if self.step_start is None else int(self.step_start),
            step_end=self.step_end,
            priority=int(self.support_count if priority is None else priority),
            feature_center=np.asarray(self.feature_center, dtype=float),
            feature_scale=np.asarray(self.feature_scale, dtype=float),
            max_feature_distance=float(self.max_feature_distance),
            metadata=payload,
        )


@dataclass(frozen=True)
class FeatureAtlasFit:
    """Fitted feature atlas with local regime regions."""

    regions: tuple[FeatureAtlasRegion, ...]
    labels: np.ndarray
    feature_names: tuple[str, ...]
    global_center: np.ndarray
    global_scale: np.ndarray
    inertia: float
    silhouette: float
    coverage: float
    radius_quantile: float
    radius_safety_factor: float

    @property
    def n_regions(self) -> int:
        return int(len(self.regions))

    @property
    def support_counts(self) -> np.ndarray:
        return np.asarray([region.support_count for region in self.regions], dtype=int)

    @property
    def max_region_radius(self) -> float:
        if not self.regions:
            return float("nan")
        return float(max(region.max_feature_distance for region in self.regions))

    def region_for_feature(self, feature: Sequence[float] | np.ndarray) -> tuple[FeatureAtlasRegion, float] | None:
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
            "global_center": np.asarray(self.global_center, dtype=float).tolist(),
            "global_scale": np.asarray(self.global_scale, dtype=float).tolist(),
            "inertia": float(self.inertia),
            "silhouette": float(self.silhouette),
            "coverage": float(self.coverage),
            "radius_quantile": float(self.radius_quantile),
            "radius_safety_factor": float(self.radius_safety_factor),
            "regions": [region.to_dict() for region in self.regions],
        }


def _mean_silhouette(features: np.ndarray, labels: np.ndarray, *, max_samples: int = 2000) -> float:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    n_samples = features.shape[0]
    if len(set(labels.tolist())) < 2 or n_samples < 3:
        return 0.0
    if n_samples > max_samples:
        indices = np.linspace(0, n_samples - 1, num=max_samples, dtype=int)
        matrix = features[indices, :]
        labels_eval = labels[indices]
    else:
        matrix = features
        labels_eval = labels
    diff = matrix[:, None, :] - matrix[None, :, :]
    distances = np.sqrt(np.einsum("ijd,ijd->ij", diff, diff))
    values: list[float] = []
    for i in range(matrix.shape[0]):
        same = labels_eval == labels_eval[i]
        same[i] = False
        a = float(distances[i, same].mean()) if np.any(same) else 0.0
        b = float("inf")
        for label in sorted(set(labels_eval.tolist())):
            if label == int(labels_eval[i]):
                continue
            other = labels_eval == label
            if np.any(other):
                b = min(b, float(distances[i, other].mean()))
        if not np.isfinite(b):
            continue
        denom = max(a, b)
        values.append(0.0 if denom <= 0.0 else (b - a) / denom)
    return float(np.mean(values)) if values else 0.0


def fit_feature_atlas(
    features: np.ndarray,
    *,
    n_regions: int,
    feature_names: Sequence[str] | None = None,
    steps: Sequence[int] | np.ndarray | None = None,
    sample_weights: Sequence[float] | np.ndarray | None = None,
    min_step_span: bool = True,
    radius_quantile: float = 0.98,
    radius_safety_factor: float = 1.10,
    max_iterations: int = 100,
) -> FeatureAtlasFit:
    """Fit a feature-based local reduced-model atlas with k-medoids."""

    matrix = _as_feature_matrix(features)
    n_samples, n_features = matrix.shape
    if feature_names is None:
        names = tuple(f"feature_{i}" for i in range(n_features))
    else:
        names = tuple(str(name) for name in feature_names)
        if len(names) != n_features:
            raise ValueError("feature_names length must match the number of features")
    step_values = _as_steps(steps, n_samples)
    if not 0.0 < float(radius_quantile) <= 1.0:
        raise ValueError("radius_quantile must lie in (0, 1]")
    if float(radius_safety_factor) <= 0.0:
        raise ValueError("radius_safety_factor must be positive")

    global_center, global_scale = robust_feature_center_scale(matrix)
    scaled = scale_feature_matrix(matrix, center=global_center, scale=global_scale)
    cluster = fit_k_medoids(
        scaled,
        n_clusters=int(n_regions),
        sample_weights=sample_weights,
        max_iterations=max_iterations,
    )
    regions: list[FeatureAtlasRegion] = []
    assigned_distances = np.empty(n_samples, dtype=float)
    for region_index in range(int(n_regions)):
        sample_indices = np.flatnonzero(cluster.labels == region_index)
        if sample_indices.size == 0:
            continue
        center = matrix[sample_indices, :].mean(axis=0)
        distances = np.linalg.norm(
            scale_feature_matrix(matrix[sample_indices, :], center=center, scale=global_scale),
            axis=1,
        )
        assigned_distances[sample_indices] = distances
        radius = float(np.quantile(distances, float(radius_quantile)) * float(radius_safety_factor))
        radius = max(radius, 1.0e-12)
        step_start = None
        step_end = None
        if min_step_span and step_values is not None:
            step_start = int(step_values[sample_indices].min())
            step_end = int(step_values[sample_indices].max())
        regions.append(
            FeatureAtlasRegion(
                region_id=f"region_{region_index:03d}",
                index=int(region_index),
                sample_indices=sample_indices.astype(int),
                medoid_index=int(cluster.medoid_indices[region_index]),
                feature_center=center.astype(float),
                feature_scale=global_scale.astype(float),
                max_feature_distance=radius,
                mean_distance=float(distances.mean()),
                max_training_distance=float(distances.max()),
                step_start=step_start,
                step_end=step_end,
                metadata={
                    "feature_names": list(names),
                    "radius_quantile": float(radius_quantile),
                    "radius_safety_factor": float(radius_safety_factor),
                },
            )
        )

    accepted = 0
    for i in range(n_samples):
        region = regions[int(cluster.labels[i])]
        if assigned_distances[i] <= float(region.max_feature_distance):
            accepted += 1
    return FeatureAtlasFit(
        regions=tuple(regions),
        labels=cluster.labels.astype(int),
        feature_names=names,
        global_center=global_center,
        global_scale=global_scale,
        inertia=float(cluster.inertia),
        silhouette=_mean_silhouette(scaled, cluster.labels),
        coverage=float(accepted / n_samples),
        radius_quantile=float(radius_quantile),
        radius_safety_factor=float(radius_safety_factor),
    )


@dataclass(frozen=True)
class FeatureAtlasDiagnostics:
    """Simple over/under-representation diagnostics for an atlas."""

    passed: bool
    coverage: float
    bankable_coverage: float
    min_support: int
    max_radius: float
    overfit_region_ids: tuple[str, ...]
    underfit_region_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "coverage": float(self.coverage),
            "bankable_coverage": float(self.bankable_coverage),
            "min_support": int(self.min_support),
            "max_radius": float(self.max_radius),
            "overfit_region_ids": list(self.overfit_region_ids),
            "underfit_region_ids": list(self.underfit_region_ids),
            "reasons": list(self.reasons),
        }


def diagnose_feature_atlas(
    atlas: FeatureAtlasFit,
    *,
    min_support: int = 50,
    target_coverage: float = 0.98,
    max_radius: float | None = None,
) -> FeatureAtlasDiagnostics:
    """Flag atlas regions that are too narrow or too broad."""

    min_support_value = int(min_support)
    if min_support_value < 1:
        raise ValueError("min_support must be at least one")
    if not 0.0 < float(target_coverage) <= 1.0:
        raise ValueError("target_coverage must lie in (0, 1]")
    max_radius_value = float("inf") if max_radius is None else float(max_radius)
    if max_radius_value <= 0.0:
        raise ValueError("max_radius must be positive when provided")

    overfit = tuple(
        region.region_id for region in atlas.regions if int(region.support_count) < min_support_value
    )
    underfit = tuple(
        region.region_id for region in atlas.regions if float(region.max_feature_distance) > max_radius_value
    )
    total_support = int(sum(region.support_count for region in atlas.regions))
    bankable_support = int(
        sum(
            region.support_count
            for region in atlas.regions
            if int(region.support_count) >= min_support_value
            and float(region.max_feature_distance) <= max_radius_value
        )
    )
    bankable_coverage = 0.0 if total_support <= 0 else float(bankable_support / total_support)
    bankable_coverage = min(float(atlas.coverage), bankable_coverage)
    reasons: list[str] = []
    if float(bankable_coverage) < float(target_coverage):
        reasons.append(
            f"bankable coverage {float(bankable_coverage):.6g} below target {float(target_coverage):.6g}"
        )
    return FeatureAtlasDiagnostics(
        passed=not reasons,
        coverage=float(atlas.coverage),
        bankable_coverage=float(bankable_coverage),
        min_support=int(atlas.support_counts.min()) if atlas.regions else 0,
        max_radius=float(atlas.max_region_radius),
        overfit_region_ids=overfit,
        underfit_region_ids=underfit,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class FeatureAtlasSizeSelection:
    """Result of choosing the number of atlas regions by validation gates."""

    selected: FeatureAtlasFit
    diagnostics: FeatureAtlasDiagnostics
    candidates: tuple[tuple[FeatureAtlasFit, FeatureAtlasDiagnostics], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "candidates": [
                {
                    "atlas": atlas.to_dict(),
                    "diagnostics": diagnostics.to_dict(),
                }
                for atlas, diagnostics in self.candidates
            ],
        }


def select_feature_atlas_size(
    features: np.ndarray,
    *,
    k_values: Sequence[int] | range,
    feature_names: Sequence[str] | None = None,
    steps: Sequence[int] | np.ndarray | None = None,
    sample_weights: Sequence[float] | np.ndarray | None = None,
    min_support: int = 50,
    target_coverage: float = 0.98,
    max_radius: float | None = None,
    radius_quantile: float = 0.98,
    radius_safety_factor: float = 1.10,
) -> FeatureAtlasSizeSelection:
    """Choose the smallest atlas that passes coverage/support/radius gates."""

    candidates: list[tuple[FeatureAtlasFit, FeatureAtlasDiagnostics]] = []
    for k in k_values:
        atlas = fit_feature_atlas(
            features,
            n_regions=int(k),
            feature_names=feature_names,
            steps=steps,
            sample_weights=sample_weights,
            radius_quantile=radius_quantile,
            radius_safety_factor=radius_safety_factor,
        )
        diagnostics = diagnose_feature_atlas(
            atlas,
            min_support=int(min_support),
            target_coverage=float(target_coverage),
            max_radius=max_radius,
        )
        candidates.append((atlas, diagnostics))
    if not candidates:
        raise ValueError("k_values must contain at least one candidate")
    passing = [(atlas, diagnostics) for atlas, diagnostics in candidates if diagnostics.passed]
    if passing:
        selected, selected_diag = min(passing, key=lambda item: item[0].n_regions)
    else:
        selected, selected_diag = max(
            candidates,
            key=lambda item: (
                float(item[1].coverage),
                int(item[1].min_support),
                -float(item[1].max_radius),
                -int(item[0].n_regions),
            ),
        )
    return FeatureAtlasSizeSelection(
        selected=selected,
        diagnostics=selected_diag,
        candidates=tuple(candidates),
    )


def subspace_principal_angles(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    *,
    degrees: bool = False,
) -> np.ndarray:
    """Compute principal angles between two column spaces."""

    a = np.asarray(basis_a, dtype=float)
    b = np.asarray(basis_b, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("basis_a and basis_b must be two-dimensional")
    if a.shape[0] != b.shape[0]:
        raise ValueError("basis_a and basis_b must have the same row count")
    if a.shape[1] == 0 or b.shape[1] == 0:
        raise ValueError("basis_a and basis_b must contain at least one vector")
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    singular_values = np.linalg.svd(qa.T @ qb, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    angles = np.arccos(singular_values)
    return np.degrees(angles) if degrees else angles


def subspace_chordal_distance(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Return the chordal distance between two basis column spaces."""

    angles = subspace_principal_angles(basis_a, basis_b, degrees=False)
    return float(np.linalg.norm(np.sin(angles)))


def feature_atlas_to_bank_manifest(
    atlas: FeatureAtlasFit,
    *,
    model_path_template: str = "{region_id}.npz",
    description: str | None = None,
    priority_by_support: bool = True,
    min_support: int = 1,
    max_radius: float | None = None,
    include_unbankable: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a JSON-serializable local-bank manifest from an atlas."""

    banks: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    max_radius_value = float("inf") if max_radius is None else float(max_radius)
    for region in atlas.regions:
        bankable = (
            int(region.support_count) >= int(min_support)
            and float(region.max_feature_distance) <= max_radius_value
        )
        if not bankable and not include_unbankable:
            skipped.append(
                {
                    "id": str(region.region_id),
                    "support_count": int(region.support_count),
                    "max_feature_distance": float(region.max_feature_distance),
                    "reason": "below_min_support_or_above_max_radius",
                }
            )
            continue
        path = model_path_template.format(
            region_id=region.region_id,
            region_index=region.index,
            support_count=region.support_count,
        )
        priority = int(region.support_count) if priority_by_support else int(atlas.n_regions - region.index)
        metadata = dict(extra_metadata or {})
        entry = region.to_bank_entry(path=path, priority=priority, metadata=metadata)
        banks.append(entry.to_dict())
    return {
        "schema_version": 1,
        "description": description
        or "Feature-based local reduced-model bank manifest generated from a fitted atlas.",
        "feature_names": list(atlas.feature_names),
        "atlas": {
            "n_regions": int(atlas.n_regions),
            "bank_count": int(len(banks)),
            "skipped_region_count": int(len(skipped)),
            "coverage": float(atlas.coverage),
            "silhouette": float(atlas.silhouette),
            "inertia": float(atlas.inertia),
            "radius_quantile": float(atlas.radius_quantile),
            "radius_safety_factor": float(atlas.radius_safety_factor),
        },
        "skipped_regions": skipped,
        "banks": banks,
    }


__all__ = [
    "FeatureAtlasDiagnostics",
    "FeatureAtlasFit",
    "FeatureAtlasRegion",
    "FeatureAtlasSizeSelection",
    "KMedoidsResult",
    "diagnose_feature_atlas",
    "feature_atlas_to_bank_manifest",
    "fit_feature_atlas",
    "fit_k_medoids",
    "robust_feature_center_scale",
    "scale_feature_matrix",
    "select_feature_atlas_size",
    "subspace_chordal_distance",
    "subspace_principal_angles",
]
