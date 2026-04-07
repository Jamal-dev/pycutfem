from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pod import PODBasis, fit_pod, project_to_basis


def _as_snapshot_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("expected a 1D or 2D array")
    return matrix


@dataclass(frozen=True)
class QuadraticFeatureMap:
    """Deterministic upper-triangular quadratic features."""

    rank: int
    pairs: tuple[tuple[int, int], ...] = field(init=False)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be positive")
        object.__setattr__(
            self,
            "pairs",
            tuple((i, j) for i in range(self.rank) for j in range(i, self.rank)),
        )

    @property
    def n_terms(self) -> int:
        return len(self.pairs)

    def transform(self, reduced_coords: np.ndarray) -> np.ndarray:
        coords = _as_snapshot_matrix(reduced_coords)
        if coords.shape[0] != self.rank:
            raise ValueError("reduced coordinate rank does not match feature map rank")
        features = np.empty((self.n_terms, coords.shape[1]), dtype=float)
        for index, (i, j) in enumerate(self.pairs):
            features[index, :] = coords[i, :] * coords[j, :]
        return features


def quadratic_feature_matrix(reduced_coords: np.ndarray, *, rank: int | None = None) -> np.ndarray:
    coords = _as_snapshot_matrix(reduced_coords)
    feature_map = QuadraticFeatureMap(rank=coords.shape[0] if rank is None else rank)
    return feature_map.transform(coords)


@dataclass
class QuadraticManifoldDecoder:
    """Linear POD decoder enriched by quadratic correction terms."""

    linear_basis: np.ndarray
    quadratic_basis: np.ndarray
    mean: np.ndarray | None = None
    feature_map: QuadraticFeatureMap | None = None

    def __post_init__(self) -> None:
        self.linear_basis = np.asarray(self.linear_basis, dtype=float)
        self.quadratic_basis = np.asarray(self.quadratic_basis, dtype=float)
        if self.feature_map is None:
            self.feature_map = QuadraticFeatureMap(rank=self.linear_basis.shape[1])

    @property
    def n_linear_modes(self) -> int:
        return int(self.linear_basis.shape[1])

    def decode(self, reduced_coords: np.ndarray) -> np.ndarray:
        coords = _as_snapshot_matrix(reduced_coords)
        quadratic_features = self.feature_map.transform(coords)
        reconstruction = self.linear_basis @ coords + self.quadratic_basis @ quadratic_features
        if self.mean is not None:
            reconstruction = reconstruction + np.asarray(self.mean, dtype=float)
        return reconstruction

    def restricted(self, restriction_matrix: np.ndarray) -> "QuadraticManifoldDecoder":
        matrix = np.asarray(restriction_matrix, dtype=float)
        mean = None if self.mean is None else matrix @ self.mean
        return QuadraticManifoldDecoder(
            linear_basis=matrix @ self.linear_basis,
            quadratic_basis=matrix @ self.quadratic_basis,
            mean=mean,
            feature_map=self.feature_map,
        )


def fit_quadratic_manifold(
    snapshots: np.ndarray,
    reduced_coords: np.ndarray,
    basis: np.ndarray,
    *,
    mean: np.ndarray | None = None,
) -> np.ndarray:
    """Fit Eq. (17) for a fixed linear basis and reduced coordinates."""

    snapshot_matrix = _as_snapshot_matrix(snapshots)
    coords = _as_snapshot_matrix(reduced_coords)
    linear_basis = np.asarray(basis, dtype=float)
    centered = snapshot_matrix if mean is None else snapshot_matrix - np.asarray(mean, dtype=float)
    residual = centered - linear_basis @ coords
    features = QuadraticFeatureMap(rank=coords.shape[0]).transform(coords)
    quadratic_basis = np.linalg.lstsq(features.T, residual.T, rcond=None)[0].T
    # Numerically enforce the orthogonality constraint from Eq. (17).
    quadratic_basis = quadratic_basis - linear_basis @ (linear_basis.T @ quadratic_basis)
    return quadratic_basis


def fit_quadratic_decoder(
    snapshots: np.ndarray,
    *,
    basis: np.ndarray | None = None,
    pod: PODBasis | None = None,
    n_modes: int | None = None,
    energy: float | None = None,
    center: bool = True,
) -> QuadraticManifoldDecoder:
    """Fit a POD basis and the quadratic correction used in Eqs. (14)-(17)."""

    snapshot_matrix = _as_snapshot_matrix(snapshots)
    if pod is None:
        if basis is not None:
            pod = PODBasis(
                basis=np.asarray(basis, dtype=float),
                singular_values=np.array([], dtype=float),
                energy_fraction=np.array([], dtype=float),
                mean=snapshot_matrix.mean(axis=1, keepdims=True) if center else None,
            )
        else:
            pod = fit_pod(snapshot_matrix, n_modes=n_modes, energy=energy, center=center)

    reduced_coords = project_to_basis(snapshot_matrix, pod.basis, pod.mean)
    quadratic_basis = fit_quadratic_manifold(
        snapshot_matrix,
        reduced_coords,
        pod.basis,
        mean=pod.mean,
    )
    return QuadraticManifoldDecoder(
        linear_basis=pod.basis,
        quadratic_basis=quadratic_basis,
        mean=pod.mean,
        feature_map=QuadraticFeatureMap(rank=pod.basis.shape[1]),
    )
