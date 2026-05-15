"""Native state-update specifications for reduced online solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _finite_matrix(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2-D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _finite_vector(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


@dataclass(frozen=True)
class NativeStateArraySpec:
    """Named mutable array exposed to a native online solve."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "float64"
    role: str = "state"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        shape = tuple(int(v) for v in self.shape)
        if any(v < 0 for v in shape):
            raise ValueError("state array shape entries must be nonnegative.")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", str(self.dtype))
        object.__setattr__(self, "role", str(self.role))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_native_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": np.asarray(self.shape, dtype=np.int64),
            "dtype": self.dtype,
            "role": self.role,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class AffineStateUpdateSpec:
    """C++-executable affine state update ``target = offset + basis @ q``."""

    name: str
    basis: np.ndarray
    offset: np.ndarray
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        basis = _finite_matrix(self.basis, "affine state update basis")
        offset = _finite_vector(self.offset, "affine state update offset")
        if basis.shape[0] != offset.size:
            raise ValueError("affine state update basis rows must match offset size.")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_native_dict(self) -> dict[str, Any]:
        return {
            "kind": "affine",
            "name": self.name,
            "basis": self.basis,
            "offset": self.offset,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_native_dict(cls, payload: Mapping[str, Any]) -> "AffineStateUpdateSpec":
        return cls(
            name=str(payload["name"]),
            basis=payload["basis"],
            offset=payload["offset"],
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class SymbolicStateUpdateKernelSpec:
    """Reference to a generated native kernel that refreshes nonlinear state.

    This is the problem-generic contract used by artifacts and future native
    drivers for DVMS/history/quadrature updates.  The current online driver can
    execute affine updates directly and can persist symbolic kernel references;
    generated symbolic update code plugs into this same contract.
    """

    name: str
    kernel_id: str
    abi: str
    param_order: tuple[str, ...]
    target_names: tuple[str, ...]
    argument_map: Mapping[str, str] | None = None
    stage: str = "pre_residual"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        param_order = tuple(str(v) for v in self.param_order)
        target_names = tuple(str(v) for v in self.target_names)
        if not param_order:
            raise ValueError("symbolic state update kernel param_order must not be empty.")
        if not target_names:
            raise ValueError("symbolic state update kernel must expose at least one target.")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "kernel_id", str(self.kernel_id))
        object.__setattr__(self, "abi", str(self.abi))
        object.__setattr__(self, "param_order", param_order)
        object.__setattr__(self, "target_names", target_names)
        object.__setattr__(self, "argument_map", dict(self.argument_map or {}))
        object.__setattr__(self, "stage", str(self.stage))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_native_dict(self) -> dict[str, Any]:
        return {
            "kind": "symbolic_kernel",
            "name": self.name,
            "kernel_id": self.kernel_id,
            "abi": self.abi,
            "param_order": tuple(self.param_order),
            "target_names": tuple(self.target_names),
            "argument_map": dict(self.argument_map or {}),
            "stage": self.stage,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class NativeStateUpdateKernelCall:
    """Runtime call descriptor for a generated native state-update kernel."""

    metadata_capsule: Any
    param_order: tuple[str, ...]
    static_args: Mapping[str, Any]
    target_name: str
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        param_order = tuple(str(v) for v in self.param_order)
        if not param_order:
            raise ValueError("native state-update kernel call param_order must not be empty.")
        if not str(self.target_name):
            raise ValueError("native state-update kernel call target_name must not be empty.")
        scale = float(self.scale)
        offset = float(self.offset)
        if not np.isfinite(scale) or not np.isfinite(offset):
            raise ValueError("native state-update kernel call scale/offset must be finite.")
        object.__setattr__(self, "param_order", param_order)
        object.__setattr__(self, "static_args", dict(self.static_args))
        object.__setattr__(self, "target_name", str(self.target_name))
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", offset)

    def to_online_dict(self) -> dict[str, Any]:
        return {
            "metadata_capsule": self.metadata_capsule,
            "param_order": tuple(self.param_order),
            "static_args": dict(self.static_args),
            "target_name": self.target_name,
            "scale": float(self.scale),
            "offset": float(self.offset),
        }


@dataclass(frozen=True)
class StateTransactionSpec:
    """Transaction boundaries for line-search/trust-region trial states."""

    state_arrays: tuple[NativeStateArraySpec, ...] = ()
    affine_updates: tuple[AffineStateUpdateSpec, ...] = ()
    symbolic_updates: tuple[SymbolicStateUpdateKernelSpec, ...] = ()
    restore_on_reject: bool = True
    commit_on_accept: bool = True
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_arrays", tuple(self.state_arrays))
        object.__setattr__(self, "affine_updates", tuple(self.affine_updates))
        object.__setattr__(self, "symbolic_updates", tuple(self.symbolic_updates))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_native_dict(self) -> dict[str, Any]:
        return {
            "state_arrays": tuple(spec.to_native_dict() for spec in self.state_arrays),
            "affine_updates": tuple(spec.to_native_dict() for spec in self.affine_updates),
            "symbolic_updates": tuple(spec.to_native_dict() for spec in self.symbolic_updates),
            "restore_on_reject": bool(self.restore_on_reject),
            "commit_on_accept": bool(self.commit_on_accept),
            "metadata": dict(self.metadata or {}),
        }


def coerce_affine_state_update(update: Mapping[str, Any] | AffineStateUpdateSpec) -> AffineStateUpdateSpec:
    if isinstance(update, AffineStateUpdateSpec):
        return update
    return AffineStateUpdateSpec.from_native_dict(update)


def coerce_affine_state_updates(
    updates: Sequence[Mapping[str, Any] | AffineStateUpdateSpec] | None,
) -> tuple[AffineStateUpdateSpec, ...]:
    if updates is None:
        return ()
    return tuple(coerce_affine_state_update(item) for item in updates)


def apply_affine_state_updates(
    updates: Sequence[Mapping[str, Any] | AffineStateUpdateSpec],
    coefficients: Any,
) -> dict[str, np.ndarray]:
    """Evaluate affine state updates in Python for verification/offline tooling."""

    q = _finite_vector(coefficients, "reduced coefficients")
    out: dict[str, np.ndarray] = {}
    for update in coerce_affine_state_updates(updates):
        if update.basis.shape[1] != q.size:
            raise ValueError("affine state update basis columns must match coefficient size.")
        out[update.name] = np.asarray(update.offset + update.basis @ q, dtype=float).reshape(-1)
    return out


__all__ = [
    "AffineStateUpdateSpec",
    "NativeStateUpdateKernelCall",
    "NativeStateArraySpec",
    "StateTransactionSpec",
    "SymbolicStateUpdateKernelSpec",
    "apply_affine_state_updates",
    "coerce_affine_state_update",
    "coerce_affine_state_updates",
]
