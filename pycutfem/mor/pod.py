from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scaling import MeanCenterer


def _as_snapshot_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("expected a 1D or 2D snapshot matrix")
    return matrix


def _select_rank(singular_values: np.ndarray, n_modes: int | None, energy: float | None) -> int:
    if n_modes is not None and energy is not None:
        raise ValueError("specify either n_modes or energy, not both")
    if n_modes is not None:
        if n_modes < 1:
            raise ValueError("n_modes must be positive")
        return min(int(n_modes), singular_values.size)
    if energy is None:
        return singular_values.size
    if not 0.0 < energy <= 1.0:
        raise ValueError("energy must lie in (0, 1]")
    squared = singular_values**2
    cumulative = np.cumsum(squared) / squared.sum()
    return int(np.searchsorted(cumulative, energy, side="left") + 1)


@dataclass
class PODBasis:
    basis: np.ndarray
    singular_values: np.ndarray
    energy_fraction: np.ndarray
    mean: np.ndarray | None = None

    @property
    def n_modes(self) -> int:
        return int(self.basis.shape[1])

    def project(self, values: np.ndarray) -> np.ndarray:
        return project_to_basis(values, self.basis, self.mean)

    def reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        return reconstruct_from_basis(coeffs, self.basis, self.mean)


def fit_pod(
    values: np.ndarray,
    *,
    n_modes: int | None = None,
    energy: float | None = None,
    center: bool = True,
) -> PODBasis:
    """Fit a POD basis on a feature-major snapshot matrix."""

    matrix = _as_snapshot_matrix(values)
    centerer = MeanCenterer()
    mean = None
    if center:
        matrix = centerer.fit_transform(matrix)
        mean = centerer.mean_

    left_singular_vectors, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = _select_rank(singular_values, n_modes=n_modes, energy=energy)
    squared = singular_values**2
    total_energy = squared.sum()
    energy_fraction = np.cumsum(squared) / total_energy if total_energy else np.zeros_like(squared)
    return PODBasis(
        basis=left_singular_vectors[:, :rank],
        singular_values=singular_values[:rank],
        energy_fraction=energy_fraction[:rank],
        mean=mean,
    )


def project_to_basis(values: np.ndarray, basis: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
    matrix = _as_snapshot_matrix(values)
    if mean is not None:
        matrix = matrix - np.asarray(mean, dtype=float)
    return np.asarray(basis, dtype=float).T @ matrix


def reconstruct_from_basis(
    coeffs: np.ndarray,
    basis: np.ndarray,
    mean: np.ndarray | None = None,
) -> np.ndarray:
    reduced = _as_snapshot_matrix(coeffs)
    reconstruction = np.asarray(basis, dtype=float) @ reduced
    if mean is not None:
        reconstruction = reconstruction + np.asarray(mean, dtype=float)
    return reconstruction


def project(values: np.ndarray, basis: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
    return project_to_basis(values, basis, mean)


def reconstruct(coeffs: np.ndarray, basis: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
    return reconstruct_from_basis(coeffs, basis, mean)
