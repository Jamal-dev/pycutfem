"""Subspace/Grassmann partitioning utilities for regime atlases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .features import subspace_chordal_distance, subspace_principal_angles


def _basis_list(bases: Sequence[np.ndarray]) -> list[np.ndarray]:
    out = [np.asarray(basis, dtype=float) for basis in bases]
    if not out:
        raise ValueError("at least one basis is required.")
    row_count = out[0].shape[0]
    for basis in out:
        if basis.ndim != 2 or basis.shape[0] != row_count or basis.shape[1] == 0:
            raise ValueError("all bases must be 2D with a shared row count and at least one mode.")
    return out


@dataclass(frozen=True)
class SubspacePartition:
    labels: np.ndarray
    medoid_indices: np.ndarray
    distance_matrix: np.ndarray


def subspace_distance_matrix(bases: Sequence[np.ndarray]) -> np.ndarray:
    basis_list = _basis_list(bases)
    n = len(basis_list)
    distances = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            value = subspace_chordal_distance(basis_list[i], basis_list[j])
            distances[i, j] = distances[j, i] = value
    return distances


@dataclass(frozen=True)
class SubspacePartitioner:
    """K-medoids-like partitioner using chordal subspace distances."""

    n_regions: int
    max_iterations: int = 100

    def fit(self, bases: Sequence[np.ndarray]) -> SubspacePartition:
        basis_list = _basis_list(bases)
        distances = subspace_distance_matrix(basis_list)
        k = int(self.n_regions)
        if k < 1 or k > len(basis_list):
            raise ValueError("n_regions must lie between one and the number of bases.")
        first = int(np.argmin(np.sum(distances, axis=1)))
        medoids = [first]
        nearest = distances[:, first].copy()
        for _ in range(1, k):
            nearest[np.asarray(medoids, dtype=int)] = -1.0
            medoids.append(int(np.argmax(nearest)))
            nearest = np.minimum(nearest, distances[:, medoids[-1]])
        medoids_arr = np.asarray(medoids, dtype=int)
        labels = np.zeros(len(basis_list), dtype=int)
        for _ in range(max(1, int(self.max_iterations))):
            labels = np.argmin(distances[:, medoids_arr], axis=1).astype(int)
            updated = medoids_arr.copy()
            for region in range(k):
                indices = np.flatnonzero(labels == region)
                if indices.size == 0:
                    updated[region] = int(np.argmax(np.min(distances[:, medoids_arr], axis=1)))
                    continue
                costs = distances[np.ix_(indices, indices)].sum(axis=1)
                updated[region] = int(indices[int(np.argmin(costs))])
            if np.array_equal(updated, medoids_arr):
                break
            medoids_arr = updated
        labels = np.argmin(distances[:, medoids_arr], axis=1).astype(int)
        return SubspacePartition(labels=labels, medoid_indices=medoids_arr, distance_matrix=distances)


__all__ = [
    "SubspacePartition",
    "SubspacePartitioner",
    "subspace_chordal_distance",
    "subspace_distance_matrix",
    "subspace_principal_angles",
]
