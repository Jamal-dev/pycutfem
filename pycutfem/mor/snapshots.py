from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _as_snapshot_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("expected a 1D or 2D snapshot matrix")
    return matrix


@dataclass
class SnapshotBatch:
    interface_forces: np.ndarray
    full_displacements: np.ndarray
    parameters: np.ndarray | None = None
    times: np.ndarray | None = None
    subiterations: np.ndarray | None = None
    converged: np.ndarray | None = None
    solid_times: np.ndarray | None = None
    fluid_times: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.interface_forces = _as_snapshot_matrix(self.interface_forces)
        self.full_displacements = _as_snapshot_matrix(self.full_displacements)
        if self.interface_forces.shape[1] != self.full_displacements.shape[1]:
            raise ValueError("force and displacement snapshot counts must match")
        self.parameters = None if self.parameters is None else np.asarray(self.parameters, dtype=float)
        self.times = None if self.times is None else np.asarray(self.times, dtype=float)
        self.subiterations = None if self.subiterations is None else np.asarray(self.subiterations, dtype=int)
        self.converged = None if self.converged is None else np.asarray(self.converged, dtype=bool)
        self.solid_times = None if self.solid_times is None else np.asarray(self.solid_times, dtype=float)
        self.fluid_times = None if self.fluid_times is None else np.asarray(self.fluid_times, dtype=float)
        for field_name in (
            "times",
            "subiterations",
            "converged",
            "solid_times",
            "fluid_times",
        ):
            values = getattr(self, field_name)
            if values is not None and values.shape[0] != self.n_snapshots:
                raise ValueError(f"{field_name} length must match snapshot count")
        if self.parameters is not None and self.parameters.shape[0] != self.n_snapshots:
            raise ValueError("parameters must be sample-major with one row per snapshot")

    @property
    def n_snapshots(self) -> int:
        return int(self.interface_forces.shape[1])

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            interface_forces=self.interface_forces,
            full_displacements=self.full_displacements,
            parameters=self.parameters,
            times=self.times,
            subiterations=self.subiterations,
            converged=self.converged,
            solid_times=self.solid_times,
            fluid_times=self.fluid_times,
            metadata=np.array(json.dumps(self.metadata), dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SnapshotBatch":
        with np.load(Path(path), allow_pickle=True) as data:
            metadata = {}
            if "metadata" in data and data["metadata"].shape == ():
                metadata = json.loads(str(data["metadata"].item()))
            return cls(
                interface_forces=data["interface_forces"],
                full_displacements=data["full_displacements"],
                parameters=data["parameters"] if "parameters" in data and data["parameters"].shape != () else None,
                times=data["times"] if "times" in data and data["times"].shape != () else None,
                subiterations=data["subiterations"]
                if "subiterations" in data and data["subiterations"].shape != ()
                else None,
                converged=data["converged"] if "converged" in data and data["converged"].shape != () else None,
                solid_times=data["solid_times"]
                if "solid_times" in data and data["solid_times"].shape != ()
                else None,
                fluid_times=data["fluid_times"]
                if "fluid_times" in data and data["fluid_times"].shape != ()
                else None,
                metadata=metadata,
            )


@dataclass
class SnapshotWriter:
    _forces: list[np.ndarray] = field(default_factory=list)
    _displacements: list[np.ndarray] = field(default_factory=list)
    _parameters: list[np.ndarray | None] = field(default_factory=list)
    _times: list[float] = field(default_factory=list)
    _subiterations: list[int] = field(default_factory=list)
    _converged: list[bool] = field(default_factory=list)
    _solid_times: list[float] = field(default_factory=list)
    _fluid_times: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(
        self,
        *,
        interface_force: np.ndarray,
        full_displacement: np.ndarray,
        parameter: np.ndarray | None = None,
        time: float = 0.0,
        subiteration: int = 0,
        converged: bool = False,
        solid_time: float = 0.0,
        fluid_time: float = 0.0,
    ) -> None:
        self._forces.append(np.asarray(interface_force, dtype=float).reshape(-1))
        self._displacements.append(np.asarray(full_displacement, dtype=float).reshape(-1))
        self._parameters.append(None if parameter is None else np.asarray(parameter, dtype=float).reshape(-1))
        self._times.append(float(time))
        self._subiterations.append(int(subiteration))
        self._converged.append(bool(converged))
        self._solid_times.append(float(solid_time))
        self._fluid_times.append(float(fluid_time))

    def to_batch(self) -> SnapshotBatch:
        parameters = None
        if any(parameter is not None for parameter in self._parameters):
            param_dim = max(0 if parameter is None else parameter.size for parameter in self._parameters)
            parameters = np.full((len(self._parameters), param_dim), np.nan, dtype=float)
            for row, parameter in enumerate(self._parameters):
                if parameter is not None:
                    parameters[row, : parameter.size] = parameter
        return SnapshotBatch(
            interface_forces=np.column_stack(self._forces),
            full_displacements=np.column_stack(self._displacements),
            parameters=parameters,
            times=np.asarray(self._times, dtype=float),
            subiterations=np.asarray(self._subiterations, dtype=int),
            converged=np.asarray(self._converged, dtype=bool),
            solid_times=np.asarray(self._solid_times, dtype=float),
            fluid_times=np.asarray(self._fluid_times, dtype=float),
            metadata=self.metadata,
        )

    def save(self, path: str | Path) -> None:
        self.to_batch().save(path)


@dataclass
class SnapshotReader:
    path: str | Path

    def load(self) -> SnapshotBatch:
        return SnapshotBatch.load(self.path)
