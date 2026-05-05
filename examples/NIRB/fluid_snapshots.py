from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from examples.NIRB.fluid_fom_operator import FluidFOMOperator
from examples.NIRB.fluid_lspg import pack_fluid_state, write_fluid_state
from examples.NIRB.run_example2_local import CoordinateLookup


FLUID_STAGE_SNAPSHOT_SCHEMA_VERSION = 1


_FIELD_KEYS = (
    "u",
    "p",
    "a",
    "u_prev",
    "p_prev",
    "a_prev",
    "d_mesh",
    "d_prev",
    "d_prev2",
    "w_mesh",
    "w_mesh_prev",
    "a_mesh",
    "a_mesh_prev",
)

_PROB_FIELD_NAMES = {
    "u": "u_k",
    "p": "p_k",
    "a": "a_k",
    "u_prev": "u_prev",
    "p_prev": "p_prev",
    "a_prev": "a_prev",
    "d_mesh": "d_mesh",
    "d_prev": "d_prev",
    "d_prev2": "d_prev2",
    "w_mesh": "w_mesh_k",
    "w_mesh_prev": "w_mesh_prev",
    "a_mesh": "a_mesh_k",
    "a_mesh_prev": "a_mesh_prev",
}


def _copy_prob_values(operator: FluidFOMOperator, logical_name: str) -> np.ndarray:
    prob_name = _PROB_FIELD_NAMES[logical_name]
    values = operator.prob.get(prob_name)
    if values is None:
        return np.zeros(0, dtype=float)
    return np.asarray(values.nodal_values, dtype=float).copy()


def _as_feature_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D feature-major matrix.")
    return matrix


@dataclass(frozen=True)
class FluidStageRecord:
    """One exact Example 2 fluid stage, including hidden DVMS and BC metadata."""

    state: np.ndarray
    u: np.ndarray
    p: np.ndarray
    a: np.ndarray
    u_prev: np.ndarray
    p_prev: np.ndarray
    a_prev: np.ndarray
    d_mesh: np.ndarray
    d_prev: np.ndarray
    d_prev2: np.ndarray
    w_mesh: np.ndarray
    w_mesh_prev: np.ndarray
    a_mesh: np.ndarray
    a_mesh_prev: np.ndarray
    free_dofs: np.ndarray
    fixed_dofs: np.ndarray
    fixed_values: np.ndarray
    dvms: dict[str, np.ndarray] = field(default_factory=dict)
    reaction_coords: np.ndarray | None = None
    reaction_values: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", np.asarray(self.state, dtype=float).reshape(-1))
        for key in _FIELD_KEYS:
            object.__setattr__(self, key, np.asarray(getattr(self, key), dtype=float).reshape(-1))
        free = np.asarray(self.free_dofs, dtype=int).reshape(-1)
        fixed = np.asarray(self.fixed_dofs, dtype=int).reshape(-1)
        fixed_values = np.asarray(self.fixed_values, dtype=float).reshape(-1)
        if np.unique(free).size != free.size:
            raise ValueError("FluidStageRecord.free_dofs must be unique.")
        if np.unique(fixed).size != fixed.size:
            raise ValueError("FluidStageRecord.fixed_dofs must be unique.")
        if fixed_values.size != fixed.size:
            raise ValueError("FluidStageRecord.fixed_values size must match fixed_dofs.")
        if np.intersect1d(free, fixed).size:
            raise ValueError("FluidStageRecord free and fixed DOFs must be disjoint.")
        object.__setattr__(self, "free_dofs", free)
        object.__setattr__(self, "fixed_dofs", fixed)
        object.__setattr__(self, "fixed_values", fixed_values)
        object.__setattr__(
            self,
            "dvms",
            {str(key): np.asarray(value, dtype=float).copy() for key, value in dict(self.dvms).items()},
        )
        if self.reaction_coords is not None:
            coords = np.asarray(self.reaction_coords, dtype=float)
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError("reaction_coords must have shape (n_points, 2).")
            object.__setattr__(self, "reaction_coords", coords.copy())
        if self.reaction_values is not None:
            values = np.asarray(self.reaction_values, dtype=float)
            if values.ndim == 1:
                values = values.reshape(-1, 2)
            if values.ndim != 2 or values.shape[1] != 2:
                raise ValueError("reaction_values must have shape (n_points, 2).")
            if self.reaction_coords is not None and values.shape[0] != self.reaction_coords.shape[0]:
                raise ValueError("reaction_values point count must match reaction_coords.")
            object.__setattr__(self, "reaction_values", values.copy())


def capture_fluid_stage(
    operator: FluidFOMOperator,
    *,
    reaction_loads: CoordinateLookup | None = None,
    include_reaction: bool = False,
    metadata: dict[str, Any] | None = None,
) -> FluidStageRecord:
    """Capture the exact state needed for later full-mesh LSPG replay."""

    bcs = operator.prob.get("_current_bcs") or []
    bc_map = operator.dh.get_dirichlet_data(bcs) or {}
    fixed_items = sorted((int(gdof), float(value)) for gdof, value in bc_map.items())
    fixed_dofs = np.asarray([gdof for gdof, _value in fixed_items], dtype=int)
    fixed_values = np.asarray([value for _gdof, value in fixed_items], dtype=float)
    if reaction_loads is None and bool(include_reaction):
        reaction_loads = operator.reaction_loads(refresh_state=False)
    dvms_snapshot = operator.snapshot_history() or {}
    return FluidStageRecord(
        state=pack_fluid_state(operator),
        u=_copy_prob_values(operator, "u"),
        p=_copy_prob_values(operator, "p"),
        a=_copy_prob_values(operator, "a"),
        u_prev=_copy_prob_values(operator, "u_prev"),
        p_prev=_copy_prob_values(operator, "p_prev"),
        a_prev=_copy_prob_values(operator, "a_prev"),
        d_mesh=_copy_prob_values(operator, "d_mesh"),
        d_prev=_copy_prob_values(operator, "d_prev"),
        d_prev2=_copy_prob_values(operator, "d_prev2"),
        w_mesh=_copy_prob_values(operator, "w_mesh"),
        w_mesh_prev=_copy_prob_values(operator, "w_mesh_prev"),
        a_mesh=_copy_prob_values(operator, "a_mesh"),
        a_mesh_prev=_copy_prob_values(operator, "a_mesh_prev"),
        free_dofs=operator.free_fluid_dofs(),
        fixed_dofs=fixed_dofs,
        fixed_values=fixed_values,
        dvms=dvms_snapshot,
        reaction_coords=None if reaction_loads is None else np.asarray(reaction_loads.coords, dtype=float),
        reaction_values=None if reaction_loads is None else np.asarray(reaction_loads.values, dtype=float),
        metadata={} if metadata is None else dict(metadata),
    )


def restore_fluid_stage(operator: FluidFOMOperator, record: FluidStageRecord) -> None:
    """Restore a captured stage into the Example 2 fluid problem."""

    write_fluid_state(operator, np.asarray(record.state, dtype=float))
    for logical_name in _FIELD_KEYS:
        prob_name = _PROB_FIELD_NAMES[logical_name]
        field = operator.prob.get(prob_name)
        values = np.asarray(getattr(record, logical_name), dtype=float)
        if field is not None and values.size:
            field.nodal_values[:] = values
    operator.restore_history(record.dvms)


@dataclass(frozen=True)
class FluidStageSnapshotBatch:
    state: np.ndarray
    u: np.ndarray
    p: np.ndarray
    a: np.ndarray
    u_prev: np.ndarray
    p_prev: np.ndarray
    a_prev: np.ndarray
    d_mesh: np.ndarray
    d_prev: np.ndarray
    d_prev2: np.ndarray
    w_mesh: np.ndarray
    w_mesh_prev: np.ndarray
    a_mesh: np.ndarray
    a_mesh_prev: np.ndarray
    free_dofs: np.ndarray
    fixed_dofs: np.ndarray
    fixed_values: np.ndarray
    dvms: dict[str, np.ndarray] = field(default_factory=dict)
    dvms_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    reaction_coords: np.ndarray | None = None
    reaction_values: np.ndarray | None = None
    metadata: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = FLUID_STAGE_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", _as_feature_matrix(self.state, name="state"))
        n_snapshots = int(self.state.shape[1])
        for key in _FIELD_KEYS:
            matrix = _as_feature_matrix(getattr(self, key), name=key)
            if matrix.shape[1] != n_snapshots:
                raise ValueError(f"{key} snapshot count must match state.")
            object.__setattr__(self, key, matrix)
        free = np.asarray(self.free_dofs, dtype=int).reshape(-1)
        fixed = np.asarray(self.fixed_dofs, dtype=int).reshape(-1)
        fixed_values = _as_feature_matrix(self.fixed_values, name="fixed_values")
        if fixed_values.shape != (fixed.size, n_snapshots):
            raise ValueError("fixed_values must have shape (n_fixed_dofs, n_snapshots).")
        if np.unique(free).size != free.size:
            raise ValueError("free_dofs must be unique.")
        if np.unique(fixed).size != fixed.size:
            raise ValueError("fixed_dofs must be unique.")
        if np.intersect1d(free, fixed).size:
            raise ValueError("free and fixed DOFs must be disjoint.")
        object.__setattr__(self, "free_dofs", free)
        object.__setattr__(self, "fixed_dofs", fixed)
        object.__setattr__(self, "fixed_values", fixed_values)
        dvms = {str(key): _as_feature_matrix(value, name=f"dvms[{key}]") for key, value in dict(self.dvms).items()}
        shapes = {str(key): tuple(int(v) for v in shape) for key, shape in dict(self.dvms_shapes).items()}
        for key, value in dvms.items():
            if value.shape[1] != n_snapshots:
                raise ValueError(f"DVMS field {key!r} snapshot count must match state.")
            if key not in shapes:
                shapes[key] = (int(value.shape[0]),)
            if int(np.prod(shapes[key], dtype=int)) != int(value.shape[0]):
                raise ValueError(f"DVMS field {key!r} shape is incompatible with its flattened matrix.")
        object.__setattr__(self, "dvms", dvms)
        object.__setattr__(self, "dvms_shapes", shapes)
        if self.reaction_coords is not None:
            coords = np.asarray(self.reaction_coords, dtype=float)
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError("reaction_coords must have shape (n_points, 2).")
            object.__setattr__(self, "reaction_coords", coords.copy())
        if self.reaction_values is not None:
            values = _as_feature_matrix(self.reaction_values, name="reaction_values")
            if values.shape[1] != n_snapshots:
                raise ValueError("reaction_values snapshot count must match state.")
            if self.reaction_coords is not None and values.shape[0] != 2 * self.reaction_coords.shape[0]:
                raise ValueError("reaction_values row count must match flattened reaction_coords.")
            object.__setattr__(self, "reaction_values", values)
        metadata = [{} for _ in range(n_snapshots)] if not self.metadata else list(self.metadata)
        if len(metadata) != n_snapshots:
            raise ValueError("metadata length must match snapshot count.")
        object.__setattr__(self, "metadata", [dict(item) for item in metadata])

    @property
    def n_snapshots(self) -> int:
        return int(self.state.shape[1])

    def record(self, index: int) -> FluidStageRecord:
        idx = int(index)
        if idx < 0 or idx >= self.n_snapshots:
            raise IndexError(idx)
        reaction_values = None
        if self.reaction_values is not None:
            reaction_values = self.reaction_values[:, idx].reshape(-1, 2)
        return FluidStageRecord(
            state=self.state[:, idx],
            u=self.u[:, idx],
            p=self.p[:, idx],
            a=self.a[:, idx],
            u_prev=self.u_prev[:, idx],
            p_prev=self.p_prev[:, idx],
            a_prev=self.a_prev[:, idx],
            d_mesh=self.d_mesh[:, idx],
            d_prev=self.d_prev[:, idx],
            d_prev2=self.d_prev2[:, idx],
            w_mesh=self.w_mesh[:, idx],
            w_mesh_prev=self.w_mesh_prev[:, idx],
            a_mesh=self.a_mesh[:, idx],
            a_mesh_prev=self.a_mesh_prev[:, idx],
            free_dofs=self.free_dofs,
            fixed_dofs=self.fixed_dofs,
            fixed_values=self.fixed_values[:, idx],
            dvms={key: values[:, idx].reshape(self.dvms_shapes[key]) for key, values in self.dvms.items()},
            reaction_coords=None if self.reaction_coords is None else self.reaction_coords,
            reaction_values=reaction_values,
            metadata=self.metadata[idx],
        )

    def subset(self, indices: np.ndarray | list[int] | tuple[int, ...]) -> "FluidStageSnapshotBatch":
        idx = np.asarray(indices, dtype=int).reshape(-1)
        if idx.size == 0:
            raise ValueError("Cannot build an empty FluidStageSnapshotBatch subset.")
        if np.any(idx < 0) or np.any(idx >= self.n_snapshots):
            raise IndexError("FluidStageSnapshotBatch subset index out of range.")
        field_matrices = {key: np.asarray(getattr(self, key), dtype=float)[:, idx] for key in _FIELD_KEYS}
        return FluidStageSnapshotBatch(
            state=self.state[:, idx],
            free_dofs=self.free_dofs,
            fixed_dofs=self.fixed_dofs,
            fixed_values=self.fixed_values[:, idx],
            dvms={key: values[:, idx] for key, values in self.dvms.items()},
            dvms_shapes=self.dvms_shapes,
            reaction_coords=None if self.reaction_coords is None else self.reaction_coords,
            reaction_values=None if self.reaction_values is None else self.reaction_values[:, idx],
            metadata=[dict(self.metadata[int(i)]) for i in idx],
            schema_version=int(self.schema_version),
            **field_matrices,
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "schema_version": np.asarray(int(self.schema_version), dtype=int),
            "state": self.state,
            "free_dofs": self.free_dofs,
            "fixed_dofs": self.fixed_dofs,
            "fixed_values": self.fixed_values,
            "metadata_json": np.asarray(json.dumps(self.metadata)),
            "dvms_keys_json": np.asarray(json.dumps(sorted(self.dvms.keys()))),
            "dvms_shapes_json": np.asarray(
                json.dumps({key: list(self.dvms_shapes[key]) for key in sorted(self.dvms.keys())})
            ),
        }
        for key in _FIELD_KEYS:
            payload[key] = np.asarray(getattr(self, key), dtype=float)
        for idx, key in enumerate(sorted(self.dvms.keys())):
            payload[f"dvms_{idx}"] = np.asarray(self.dvms[key], dtype=float)
        if self.reaction_coords is not None:
            payload["reaction_coords"] = np.asarray(self.reaction_coords, dtype=float)
        if self.reaction_values is not None:
            payload["reaction_values"] = np.asarray(self.reaction_values, dtype=float)
        np.savez_compressed(target, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "FluidStageSnapshotBatch":
        with np.load(Path(path), allow_pickle=False) as data:
            keys = json.loads(str(data["dvms_keys_json"].item())) if "dvms_keys_json" in data else []
            shapes = (
                json.loads(str(data["dvms_shapes_json"].item()))
                if "dvms_shapes_json" in data
                else {str(key): [int(np.asarray(data[f"dvms_{idx}"]).shape[0])] for idx, key in enumerate(keys)}
            )
            dvms = {str(key): np.asarray(data[f"dvms_{idx}"], dtype=float) for idx, key in enumerate(keys)}
            metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else []
            kwargs = {key: np.asarray(data[key], dtype=float) for key in _FIELD_KEYS}
            return cls(
                state=np.asarray(data["state"], dtype=float),
                free_dofs=np.asarray(data["free_dofs"], dtype=int),
                fixed_dofs=np.asarray(data["fixed_dofs"], dtype=int),
                fixed_values=np.asarray(data["fixed_values"], dtype=float),
                dvms=dvms,
                dvms_shapes={str(key): tuple(int(v) for v in values) for key, values in dict(shapes).items()},
                reaction_coords=np.asarray(data["reaction_coords"], dtype=float) if "reaction_coords" in data else None,
                reaction_values=np.asarray(data["reaction_values"], dtype=float) if "reaction_values" in data else None,
                metadata=list(metadata),
                schema_version=int(np.asarray(data["schema_version"], dtype=int).reshape(-1)[0])
                if "schema_version" in data
                else FLUID_STAGE_SNAPSHOT_SCHEMA_VERSION,
                **kwargs,
            )


@dataclass
class FluidStageSnapshotWriter:
    records: list[FluidStageRecord] = field(default_factory=list)

    def append(self, record: FluidStageRecord) -> None:
        self.records.append(record)

    def append_from_operator(
        self,
        operator: FluidFOMOperator,
        *,
        reaction_loads: CoordinateLookup | None = None,
        include_reaction: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> FluidStageRecord:
        record = capture_fluid_stage(
            operator,
            reaction_loads=reaction_loads,
            include_reaction=include_reaction,
            metadata=metadata,
        )
        self.append(record)
        return record

    def to_batch(self) -> FluidStageSnapshotBatch:
        if not self.records:
            raise ValueError("Cannot build a FluidStageSnapshotBatch without records.")
        free_dofs = np.asarray(self.records[0].free_dofs, dtype=int)
        fixed_dofs = np.asarray(self.records[0].fixed_dofs, dtype=int)
        dvms_keys = sorted(self.records[0].dvms.keys())
        dvms_shapes = {
            key: tuple(int(v) for v in np.asarray(self.records[0].dvms[key], dtype=float).shape) for key in dvms_keys
        }
        have_reaction = self.records[0].reaction_coords is not None and self.records[0].reaction_values is not None
        reaction_coords = None if not have_reaction else np.asarray(self.records[0].reaction_coords, dtype=float)
        for record in self.records:
            if not np.array_equal(record.free_dofs, free_dofs):
                raise ValueError("All fluid-stage records must use the same free_dofs.")
            if not np.array_equal(record.fixed_dofs, fixed_dofs):
                raise ValueError("All fluid-stage records must use the same fixed_dofs.")
            if sorted(record.dvms.keys()) != dvms_keys:
                raise ValueError("All fluid-stage records must carry the same DVMS fields.")
            for key in dvms_keys:
                if tuple(np.asarray(record.dvms[key], dtype=float).shape) != dvms_shapes[key]:
                    raise ValueError("All fluid-stage records must carry matching DVMS field shapes.")
            record_has_reaction = record.reaction_coords is not None and record.reaction_values is not None
            if record_has_reaction != have_reaction:
                raise ValueError("Either all or no fluid-stage records must carry reaction loads.")
            if have_reaction and not np.allclose(np.asarray(record.reaction_coords, dtype=float), reaction_coords):
                raise ValueError("All reaction snapshots must use the same interface coordinates.")
        field_matrices = {
            key: np.column_stack([np.asarray(getattr(record, key), dtype=float) for record in self.records])
            for key in _FIELD_KEYS
        }
        dvms = {
            key: np.column_stack([np.asarray(record.dvms[key], dtype=float).reshape(-1) for record in self.records])
            for key in dvms_keys
        }
        return FluidStageSnapshotBatch(
            state=np.column_stack([record.state for record in self.records]),
            free_dofs=free_dofs,
            fixed_dofs=fixed_dofs,
            fixed_values=np.column_stack([record.fixed_values for record in self.records]),
            dvms=dvms,
            dvms_shapes=dvms_shapes,
            reaction_coords=reaction_coords,
            reaction_values=None
            if not have_reaction
            else np.column_stack([np.asarray(record.reaction_values, dtype=float).reshape(-1) for record in self.records]),
            metadata=[dict(record.metadata) for record in self.records],
            **field_matrices,
        )

    def save(self, path: str | Path) -> None:
        self.to_batch().save(path)


__all__ = [
    "FLUID_STAGE_SNAPSHOT_SCHEMA_VERSION",
    "FluidStageRecord",
    "FluidStageSnapshotBatch",
    "FluidStageSnapshotWriter",
    "capture_fluid_stage",
    "restore_fluid_stage",
]
