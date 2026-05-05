from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import precice  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    precice = None


def _as_point_cloud(coordinates: np.ndarray, *, mesh_dim: int | None = None) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2:
        raise ValueError("Coupling coordinates must have shape (npoints, ndim).")
    if coords.shape[0] == 0:
        ndim = int(mesh_dim) if mesh_dim is not None else int(coords.shape[1] if coords.shape[1] > 0 else 0)
        if ndim <= 0:
            raise ValueError("Cannot infer the mesh dimension from an empty coordinate array.")
        return np.zeros((0, ndim), dtype=float)
    if mesh_dim is not None and int(coords.shape[1]) != int(mesh_dim):
        raise ValueError(
            f"Coordinate dimension mismatch: expected {int(mesh_dim)}, got {int(coords.shape[1])}."
        )
    return np.asarray(coords, dtype=float).copy()


def _reshape_field_values(values: np.ndarray, *, npoints: int, data_dim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if int(data_dim) <= 0:
        raise ValueError(f"Invalid preCICE data dimension {int(data_dim)}.")
    if int(data_dim) == 1:
        if arr.ndim == 0:
            if int(npoints) != 1:
                raise ValueError("A scalar value can only be used for a single coupling vertex.")
            return np.asarray([float(arr)], dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        flat = np.asarray(arr, dtype=float).reshape(-1)
        if int(flat.size) != int(npoints):
            raise ValueError(f"Expected {int(npoints)} scalar values, got {int(flat.size)}.")
        return flat.copy()
    if arr.ndim == 1:
        if int(arr.size) != int(npoints) * int(data_dim):
            raise ValueError(
                f"Expected {int(npoints) * int(data_dim)} entries for vector data, got {int(arr.size)}."
            )
        arr = arr.reshape((int(npoints), int(data_dim)))
    if arr.ndim != 2 or arr.shape != (int(npoints), int(data_dim)):
        raise ValueError(
            f"Expected values with shape ({int(npoints)}, {int(data_dim)}), got {tuple(arr.shape)}."
        )
    return np.asarray(arr, dtype=float).copy()


def _make_zero_values(*, npoints: int, data_dim: int) -> np.ndarray:
    if int(data_dim) == 1:
        return np.zeros((int(npoints),), dtype=float)
    return np.zeros((int(npoints), int(data_dim)), dtype=float)


@dataclass
class CouplingCheckpoint:
    payload: Any
    time: float
    time_window: int


class PreCICEPointParticipant:
    """Small pyprecice wrapper for point-cloud coupling meshes."""

    def __init__(
        self,
        *,
        participant_name: str,
        config_file: str | Path,
        mesh_name: str,
        coordinates: np.ndarray,
        read_fields: tuple[str, ...] | list[str] = (),
        write_fields: tuple[str, ...] | list[str] = (),
        rank: int = 0,
        size: int = 1,
    ) -> None:
        if precice is None:
            raise RuntimeError(
                "The Python preCICE bindings are not available. Install `precice` / `pyprecice` in the runtime environment."
            )
        self.participant_name = str(participant_name)
        self.config_file = str(Path(config_file).resolve())
        self.mesh_name = str(mesh_name)
        self.read_fields = tuple(str(name) for name in tuple(read_fields or ()))
        self.write_fields = tuple(str(name) for name in tuple(write_fields or ()))
        self._participant = precice.Participant(self.participant_name, self.config_file, int(rank), int(size))
        self.mesh_dim = int(self._participant.get_mesh_dimensions(self.mesh_name))
        self.coordinates = _as_point_cloud(coordinates, mesh_dim=self.mesh_dim)
        self.vertex_ids = self._participant.set_mesh_vertices(self.mesh_name, self.coordinates)
        self._data_dims = {
            name: int(self._participant.get_data_dimensions(self.mesh_name, name))
            for name in sorted(set(self.read_fields + self.write_fields))
        }
        self._checkpoint: CouplingCheckpoint | None = None

    @property
    def vertex_count(self) -> int:
        return int(self.coordinates.shape[0])

    def initialize(self, *, initial_write_data: dict[str, np.ndarray] | None = None) -> float:
        init = dict(initial_write_data or {})
        if bool(self._participant.requires_initial_data()):
            for field in self.write_fields:
                values = init.get(field)
                if values is None:
                    values = _make_zero_values(
                        npoints=self.vertex_count,
                        data_dim=int(self._data_dims[field]),
                    )
                self.write(field, values)
        self._participant.initialize()
        return float(self._participant.get_max_time_step_size())

    def finalize(self) -> None:
        self._participant.finalize()

    def get_max_time_step_size(self) -> float:
        return float(self._participant.get_max_time_step_size())

    def is_coupling_ongoing(self) -> bool:
        return bool(self._participant.is_coupling_ongoing())

    def is_time_window_complete(self) -> bool:
        return bool(self._participant.is_time_window_complete())

    def requires_writing_checkpoint(self) -> bool:
        return bool(self._participant.requires_writing_checkpoint())

    def requires_reading_checkpoint(self) -> bool:
        return bool(self._participant.requires_reading_checkpoint())

    def advance(self, dt: float) -> float:
        self._participant.advance(float(dt))
        return float(self._participant.get_max_time_step_size())

    def read(self, field_name: str, relative_read_time: float) -> np.ndarray:
        name = str(field_name)
        if name not in self.read_fields:
            raise KeyError(f"{name!r} is not configured as a read field on {self.mesh_name!r}.")
        raw = self._participant.read_data(
            self.mesh_name,
            name,
            self.vertex_ids,
            float(relative_read_time),
        )
        return _reshape_field_values(
            raw,
            npoints=self.vertex_count,
            data_dim=int(self._data_dims[name]),
        )

    def write(self, field_name: str, values: np.ndarray) -> None:
        name = str(field_name)
        if name not in self.write_fields:
            raise KeyError(f"{name!r} is not configured as a write field on {self.mesh_name!r}.")
        payload = _reshape_field_values(
            values,
            npoints=self.vertex_count,
            data_dim=int(self._data_dims[name]),
        )
        self._participant.write_data(self.mesh_name, name, self.vertex_ids, payload)

    def store_checkpoint(self, payload: Any, *, time: float, time_window: int) -> None:
        self._checkpoint = CouplingCheckpoint(
            payload=copy.deepcopy(payload),
            time=float(time),
            time_window=int(time_window),
        )

    def retrieve_checkpoint(self) -> CouplingCheckpoint:
        if self._checkpoint is None:
            raise RuntimeError("No coupling checkpoint has been stored.")
        return CouplingCheckpoint(
            payload=copy.deepcopy(self._checkpoint.payload),
            time=float(self._checkpoint.time),
            time_window=int(self._checkpoint.time_window),
        )
