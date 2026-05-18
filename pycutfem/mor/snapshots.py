from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _as_snapshot_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("expected a 1D or 2D snapshot matrix")
    return matrix


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _optional_sample_vector(values: Any | None, n_snapshots: int, label: str, dtype: Any) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if arr.shape[0] != int(n_snapshots):
        raise ValueError(f"{label} length must match snapshot count")
    return arr


@dataclass
class NamedSnapshotBatch:
    """Generic named snapshot container for problem-independent MOR data.

    Each entry in ``fields`` is stored as a feature-major matrix with shape
    ``(n_features, n_snapshots)``.  Field names are deliberately generic:
    examples can use names such as ``state``, ``residual``, ``jacobian_action``,
    ``qoi_gradient``, ``interface_force``, or any other problem-level quantity.
    """

    fields: Mapping[str, np.ndarray]
    parameters: np.ndarray | None = None
    times: np.ndarray | None = None
    converged: np.ndarray | None = None
    sample_metadata: tuple[Mapping[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("NamedSnapshotBatch requires at least one named field")
        normalized: dict[str, np.ndarray] = {}
        n_snapshots: int | None = None
        for raw_name, values in self.fields.items():
            name = str(raw_name)
            if not name:
                raise ValueError("snapshot field names must be nonempty")
            if name in normalized:
                raise ValueError(f"duplicate snapshot field name {name!r}")
            matrix = _as_snapshot_matrix(values)
            if n_snapshots is None:
                n_snapshots = int(matrix.shape[1])
            elif matrix.shape[1] != n_snapshots:
                raise ValueError("all named snapshot fields must have the same snapshot count")
            normalized[name] = matrix
        assert n_snapshots is not None
        self.fields = normalized
        self.parameters = None if self.parameters is None else np.asarray(self.parameters, dtype=float)
        if self.parameters is not None and self.parameters.shape[0] != n_snapshots:
            raise ValueError("parameters must be sample-major with one row per snapshot")
        self.times = _optional_sample_vector(self.times, n_snapshots, "times", float)
        self.converged = _optional_sample_vector(self.converged, n_snapshots, "converged", bool)
        self.sample_metadata = tuple(dict(item) for item in self.sample_metadata)
        if self.sample_metadata and len(self.sample_metadata) != n_snapshots:
            raise ValueError("sample_metadata length must match snapshot count")
        self.metadata = dict(self.metadata)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.fields.keys())

    @property
    def n_snapshots(self) -> int:
        return int(next(iter(self.fields.values())).shape[1])

    def field(self, name: str) -> np.ndarray:
        return self.fields[str(name)]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.field(name)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        field_names = self.field_names
        for index, name in enumerate(field_names):
            arrays[f"field_{index}"] = self.fields[name]
        if self.parameters is not None:
            arrays["parameters"] = self.parameters
        if self.times is not None:
            arrays["times"] = self.times
        if self.converged is not None:
            arrays["converged"] = self.converged
        manifest = {
            "schema_version": 1,
            "field_names": field_names,
            "metadata": self.metadata,
            "sample_metadata": self.sample_metadata,
            "has_parameters": self.parameters is not None,
            "has_times": self.times is not None,
            "has_converged": self.converged is not None,
        }
        arrays["manifest_json"] = np.asarray(json.dumps(manifest, default=_json_default))
        np.savez_compressed(target, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "NamedSnapshotBatch":
        with np.load(Path(path), allow_pickle=False) as data:
            manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
            field_names = tuple(str(name) for name in manifest["field_names"])
            fields = {name: data[f"field_{index}"] for index, name in enumerate(field_names)}
            return cls(
                fields=fields,
                parameters=data["parameters"] if bool(manifest.get("has_parameters", False)) else None,
                times=data["times"] if bool(manifest.get("has_times", False)) else None,
                converged=data["converged"] if bool(manifest.get("has_converged", False)) else None,
                sample_metadata=tuple(dict(item) for item in manifest.get("sample_metadata", ())),
                metadata=dict(manifest.get("metadata", {})),
            )


@dataclass
class NamedSnapshotWriter:
    """Append generic named snapshots and materialize a ``NamedSnapshotBatch``."""

    metadata: dict[str, Any] = field(default_factory=dict)
    _fields: dict[str, list[np.ndarray]] = field(default_factory=dict)
    _field_sizes: dict[str, int] = field(default_factory=dict)
    _field_names: tuple[str, ...] | None = None
    _parameters: list[np.ndarray | None] = field(default_factory=list)
    _times: list[float] = field(default_factory=list)
    _converged: list[bool] = field(default_factory=list)
    _sample_metadata: list[Mapping[str, Any]] = field(default_factory=list)

    def append(
        self,
        fields: Mapping[str, Any],
        *,
        parameter: np.ndarray | None = None,
        time: float = 0.0,
        converged: bool = False,
        sample_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        names = tuple(str(name) for name in fields.keys())
        if not names:
            raise ValueError("append requires at least one named snapshot field")
        if self._field_names is None:
            if len(set(names)) != len(names):
                raise ValueError("snapshot field names must be unique")
            self._field_names = names
            self._fields = {name: [] for name in names}
        elif set(names) != set(self._field_names):
            raise ValueError("all appended samples must provide the same field names")

        assert self._field_names is not None
        for name in self._field_names:
            values = np.asarray(fields[name], dtype=float).reshape(-1)
            previous_size = self._field_sizes.setdefault(name, int(values.size))
            if values.size != previous_size:
                raise ValueError(f"snapshot field {name!r} changed size")
            self._fields[name].append(values)
        self._parameters.append(None if parameter is None else np.asarray(parameter, dtype=float).reshape(-1))
        self._times.append(float(time))
        self._converged.append(bool(converged))
        self._sample_metadata.append(dict(sample_metadata or {}))

    def to_batch(self) -> NamedSnapshotBatch:
        if self._field_names is None or not self._field_names:
            raise ValueError("cannot build a NamedSnapshotBatch without samples")
        parameters = None
        if any(parameter is not None for parameter in self._parameters):
            param_dim = max(0 if parameter is None else parameter.size for parameter in self._parameters)
            parameters = np.full((len(self._parameters), param_dim), np.nan, dtype=float)
            for row, parameter in enumerate(self._parameters):
                if parameter is not None:
                    parameters[row, : parameter.size] = parameter
        return NamedSnapshotBatch(
            fields={name: np.column_stack(self._fields[name]) for name in self._field_names},
            parameters=parameters,
            times=np.asarray(self._times, dtype=float),
            converged=np.asarray(self._converged, dtype=bool),
            sample_metadata=tuple(self._sample_metadata),
            metadata=dict(self.metadata),
        )

    def save(self, path: str | Path) -> None:
        self.to_batch().save(path)


@dataclass
class NamedSnapshotReader:
    path: str | Path

    def load(self) -> NamedSnapshotBatch:
        return NamedSnapshotBatch.load(self.path)


@dataclass
class SnapshotBatch:
    """NIRB/FSI force-displacement compatibility snapshot container.

    New generic MOR code should prefer ``NamedSnapshotBatch``.  This class is
    kept for existing NIRB datasets that are explicitly force/displacement
    pairs.
    """

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


__all__ = [
    "NamedSnapshotBatch",
    "NamedSnapshotReader",
    "NamedSnapshotWriter",
    "SnapshotBatch",
    "SnapshotReader",
    "SnapshotWriter",
]
