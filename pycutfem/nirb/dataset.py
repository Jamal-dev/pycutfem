from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pycutfem.mor.snapshots import SnapshotBatch


def _as_snapshot_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("expected a feature-major snapshot matrix")
    return matrix


@dataclass(frozen=True)
class DatasetSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray


@dataclass
class OfflineDataset:
    forces: np.ndarray
    displacements: np.ndarray
    parameters: np.ndarray | None = None
    times: np.ndarray | None = None
    subiterations: np.ndarray | None = None
    converged: np.ndarray | None = None
    solid_times: np.ndarray | None = None
    fluid_times: np.ndarray | None = None
    interface_indices: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.forces = _as_snapshot_matrix(self.forces)
        self.displacements = _as_snapshot_matrix(self.displacements)
        if self.forces.shape[1] != self.displacements.shape[1]:
            raise ValueError("force and displacement snapshot counts must match")
        self.parameters = None if self.parameters is None else np.asarray(self.parameters, dtype=float)
        self.times = None if self.times is None else np.asarray(self.times, dtype=float)
        self.subiterations = None if self.subiterations is None else np.asarray(self.subiterations, dtype=int)
        self.converged = None if self.converged is None else np.asarray(self.converged, dtype=bool)
        self.solid_times = None if self.solid_times is None else np.asarray(self.solid_times, dtype=float)
        self.fluid_times = None if self.fluid_times is None else np.asarray(self.fluid_times, dtype=float)
        self.interface_indices = (
            None if self.interface_indices is None else np.asarray(self.interface_indices, dtype=int)
        )

    @property
    def n_snapshots(self) -> int:
        return int(self.forces.shape[1])

    @classmethod
    def from_snapshot_batch(
        cls,
        batch: SnapshotBatch,
        *,
        interface_indices: np.ndarray | None = None,
    ) -> "OfflineDataset":
        return cls(
            forces=batch.interface_forces,
            displacements=batch.full_displacements,
            parameters=batch.parameters,
            times=batch.times,
            subiterations=batch.subiterations,
            converged=batch.converged,
            solid_times=batch.solid_times,
            fluid_times=batch.fluid_times,
            interface_indices=interface_indices,
            metadata=batch.metadata,
        )

    def subset(self, indices: np.ndarray) -> "OfflineDataset":
        chosen = np.asarray(indices, dtype=int)
        return OfflineDataset(
            forces=self.forces[:, chosen],
            displacements=self.displacements[:, chosen],
            parameters=None if self.parameters is None else self.parameters[chosen],
            times=None if self.times is None else self.times[chosen],
            subiterations=None if self.subiterations is None else self.subiterations[chosen],
            converged=None if self.converged is None else self.converged[chosen],
            solid_times=None if self.solid_times is None else self.solid_times[chosen],
            fluid_times=None if self.fluid_times is None else self.fluid_times[chosen],
            interface_indices=self.interface_indices,
            metadata=dict(self.metadata),
        )

    def split(self, *, test_fraction: float = 0.2, random_state: int = 0) -> DatasetSplit:
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must lie in (0, 1)")
        permutation = np.random.default_rng(random_state).permutation(self.n_snapshots)
        n_validation = max(1, int(round(test_fraction * self.n_snapshots)))
        return DatasetSplit(
            train_indices=permutation[n_validation:],
            validation_indices=permutation[:n_validation],
        )


def load_dataset(path: str | Path, *, interface_indices: np.ndarray | None = None) -> OfflineDataset:
    return OfflineDataset.from_snapshot_batch(
        SnapshotBatch.load(path),
        interface_indices=interface_indices,
    )
