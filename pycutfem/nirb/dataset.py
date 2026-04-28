from __future__ import annotations

import csv
import json
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


def _array_path(co_sim_dir: Path, key: str) -> Path:
    name = key if key.endswith(".npy") else f"{key}.npy"
    return co_sim_dir / name


def _resolve_co_sim_dir(path: str | Path) -> tuple[Path, Path]:
    root = Path(path)
    if (root / "coSimData").is_dir():
        return root / "coSimData", root
    return root, root.parent


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_snapshot_metadata(path: Path, n_snapshots: int) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if not path.exists():
        return None, None, None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != n_snapshots:
        return None, None, None
    times = np.asarray([float(row.get("time_s", 0.0) or 0.0) for row in rows], dtype=float)
    subiterations = np.asarray([int(row.get("coupling_iter", 0) or 0) for row in rows], dtype=int)
    converged = np.asarray(
        [str(row.get("converged", "")).strip().lower() in {"1", "true", "yes"} for row in rows],
        dtype=bool,
    )
    return times, subiterations, converged


def load_cosim_snapshot_batch(
    path: str | Path,
    *,
    force_key: str = "load_guess_data",
    displacement_key: str = "disp_data",
) -> SnapshotBatch:
    """Load Example-2-style ``coSimData`` arrays as a NIRB snapshot batch.

    ``force_key`` defaults to ``load_guess_data`` because that is the force
    vector consumed by the structural solve at each fixed-point iteration. If
    an older dataset does not contain it, the loader falls back to
    ``load_data`` for compatibility with the paper artifact layout.
    """

    co_sim_dir, run_root = _resolve_co_sim_dir(path)
    force_path = _array_path(co_sim_dir, force_key)
    if not force_path.exists() and force_key == "load_guess_data":
        force_path = _array_path(co_sim_dir, "load_data")
        force_key = "load_data"
    displacement_path = _array_path(co_sim_dir, displacement_key)
    if not force_path.exists():
        raise FileNotFoundError(force_path)
    if not displacement_path.exists():
        raise FileNotFoundError(displacement_path)

    forces = _as_snapshot_matrix(np.load(force_path))
    displacements = _as_snapshot_matrix(np.load(displacement_path))
    if forces.shape[1] != displacements.shape[1]:
        raise ValueError(
            "coSimData force and displacement snapshot counts differ: "
            f"{forces.shape[1]} != {displacements.shape[1]}"
        )

    n_snapshots = int(forces.shape[1])
    times, subiterations, converged = _load_snapshot_metadata(run_root / "snapshot_metadata.csv", n_snapshots)
    solid_times = None
    fluid_times = None
    solid_time_path = co_sim_dir / "structure_time.npy"
    fluid_time_path = co_sim_dir / "fluid_time.npy"
    if solid_time_path.exists():
        values = np.asarray(np.load(solid_time_path), dtype=float).reshape(-1)
        if values.size == n_snapshots:
            solid_times = values
    if fluid_time_path.exists():
        values = np.asarray(np.load(fluid_time_path), dtype=float).reshape(-1)
        if values.size == n_snapshots:
            fluid_times = values

    summary = _load_json_if_present(run_root / "summary.json")
    parameters = None
    if "reynolds" in summary:
        parameters = np.full((n_snapshots, 1), float(summary["reynolds"]), dtype=float)

    return SnapshotBatch(
        interface_forces=forces,
        full_displacements=displacements,
        parameters=parameters,
        times=times,
        subiterations=subiterations,
        converged=converged,
        solid_times=solid_times,
        fluid_times=fluid_times,
        metadata={
            "source": "coSimData",
            "co_sim_dir": str(co_sim_dir),
            "run_root": str(run_root),
            "force_key": force_key,
            "displacement_key": displacement_key,
            "force_path": str(force_path),
            "displacement_path": str(displacement_path),
            "summary": summary,
        },
    )


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


def load_dataset(
    path: str | Path,
    *,
    interface_indices: np.ndarray | None = None,
    force_key: str = "load_guess_data",
    displacement_key: str = "disp_data",
) -> OfflineDataset:
    source = Path(path)
    batch = (
        load_cosim_snapshot_batch(source, force_key=force_key, displacement_key=displacement_key)
        if source.is_dir()
        else SnapshotBatch.load(source)
    )
    return OfflineDataset.from_snapshot_batch(
        batch,
        interface_indices=interface_indices,
    )
