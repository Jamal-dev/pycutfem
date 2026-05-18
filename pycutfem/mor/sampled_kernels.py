"""Sampled native-kernel argument bundles for MOR online solves.

The utilities here are deliberately problem-neutral.  A PDE/example owns the
UFL forms, fields, boundary conditions, and coefficient names.  This module
owns the common array bookkeeping needed to run a generated native kernel on a
selected entity set without hand-written slicing in each example.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _as_index_vector(values: Any, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if np.any(arr < 0):
        raise ValueError(f"{label} must contain only nonnegative ids.")
    if np.unique(arr).size != arr.size:
        raise ValueError(f"{label} must not contain duplicate ids.")
    return np.ascontiguousarray(arr, dtype=np.int64)


def _infer_entity_count(static_args: Mapping[str, Any], entity_axis_arg: str = "gdofs_map") -> int:
    if entity_axis_arg in static_args:
        arr = np.asarray(static_args[entity_axis_arg])
        if arr.ndim == 0:
            raise ValueError(f"{entity_axis_arg!r} must have an entity axis.")
        return int(arr.shape[0])
    counts: list[int] = []
    for value in static_args.values():
        arr = np.asarray(value)
        if arr.ndim > 0:
            counts.append(int(arr.shape[0]))
    if not counts:
        raise ValueError("could not infer entity count from static_args.")
    return max(set(counts), key=counts.count)


def _slice_entity_array(value: Any, element_ids: np.ndarray, entity_count: int) -> Any:
    arr = np.asarray(value)
    if arr.ndim == 0 or int(arr.shape[0]) != int(entity_count):
        return value
    return np.ascontiguousarray(arr[element_ids, ...])


@dataclass(frozen=True)
class NativeSampledKernelBundle:
    """Static arguments for evaluating a native kernel on sampled entities."""

    param_order: tuple[str, ...]
    static_args: Mapping[str, Any]
    element_ids: np.ndarray
    gdofs_map: np.ndarray
    coefficient_arg_names: tuple[str, ...] = ()
    metadata_capsule: Any | None = None
    source_entity_count: int | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        param_order = tuple(str(name) for name in self.param_order)
        if not param_order:
            raise ValueError("sampled kernel bundle param_order must not be empty.")
        element_ids = _as_index_vector(self.element_ids, "sampled element_ids")
        gdofs = np.asarray(self.gdofs_map, dtype=np.int32)
        if gdofs.ndim != 2:
            raise ValueError("sampled kernel bundle gdofs_map must have shape (n_entities, n_local_dofs).")
        if int(gdofs.shape[0]) != int(element_ids.size):
            raise ValueError("sampled kernel bundle gdofs_map row count must match element_ids.")
        args = dict(self.static_args)
        args["gdofs_map"] = np.ascontiguousarray(gdofs, dtype=np.int32)
        for name in param_order:
            if name not in args:
                raise KeyError(f"sampled kernel bundle is missing static argument {name!r}.")
        object.__setattr__(self, "param_order", param_order)
        object.__setattr__(self, "static_args", args)
        object.__setattr__(self, "element_ids", element_ids)
        object.__setattr__(self, "gdofs_map", args["gdofs_map"])
        object.__setattr__(self, "coefficient_arg_names", tuple(str(v) for v in self.coefficient_arg_names))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


def build_sampled_static_args(
    static_args: Mapping[str, Any],
    *,
    element_ids: Any,
    param_order: Sequence[str] | None = None,
    coefficient_arg_names: Sequence[str] | None = None,
    entity_count: int | None = None,
    entity_axis_arg: str = "gdofs_map",
    allocate_missing_coefficients: bool = True,
) -> dict[str, Any]:
    """Return static args restricted to ``element_ids``.

    Arrays whose first axis equals the source entity count are sliced.  Scalars
    and arrays without an entity axis are passed through unchanged.  Missing
    coefficient-local arrays can be allocated with the sampled ``gdofs_map``
    shape, matching the contract used by the native online Gauss-Newton driver.
    """

    ids = _as_index_vector(element_ids, "sampled element_ids")
    n_entities = _infer_entity_count(static_args, entity_axis_arg) if entity_count is None else int(entity_count)
    if n_entities < 0:
        raise ValueError("entity_count must be nonnegative.")
    if ids.size and int(ids[-1]) >= n_entities:
        raise ValueError("sampled element_ids contain ids outside the source entity count.")

    out = {
        str(name): _slice_entity_array(value, ids, n_entities)
        for name, value in dict(static_args).items()
    }
    if entity_axis_arg not in out:
        raise ValueError(f"static_args must contain {entity_axis_arg!r} to build sampled native args.")
    gdofs = np.asarray(out[entity_axis_arg], dtype=np.int32)
    if gdofs.ndim != 2:
        raise ValueError(f"{entity_axis_arg!r} must have shape (n_entities, n_local_dofs).")
    if entity_axis_arg != "gdofs_map":
        out["gdofs_map"] = gdofs
    else:
        out["gdofs_map"] = np.ascontiguousarray(gdofs, dtype=np.int32)

    if allocate_missing_coefficients:
        for name in tuple(str(v) for v in (coefficient_arg_names or ())):
            if name not in out:
                out[name] = np.empty_like(out["gdofs_map"], dtype=np.float64)

    if param_order is not None:
        for name in tuple(str(v) for v in param_order):
            if name not in out:
                raise KeyError(f"sampled static args are missing required parameter {name!r}.")
    return out


def build_sampled_native_kernel_bundle(
    *,
    param_order: Sequence[str],
    static_args: Mapping[str, Any],
    element_ids: Any,
    coefficient_arg_names: Sequence[str] | None = None,
    metadata_capsule: Any | None = None,
    entity_count: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> NativeSampledKernelBundle:
    """Build a validated sampled native-kernel bundle."""

    sampled_args = build_sampled_static_args(
        static_args,
        element_ids=element_ids,
        param_order=param_order,
        coefficient_arg_names=coefficient_arg_names,
        entity_count=entity_count,
    )
    return NativeSampledKernelBundle(
        param_order=tuple(str(v) for v in param_order),
        static_args=sampled_args,
        element_ids=element_ids,
        gdofs_map=sampled_args["gdofs_map"],
        coefficient_arg_names=tuple(str(v) for v in (coefficient_arg_names or ())),
        metadata_capsule=metadata_capsule,
        source_entity_count=entity_count,
        metadata=metadata,
    )


def build_sampled_native_kernel_bundle_from_runner(
    runner: Any,
    static_args: Mapping[str, Any],
    *,
    element_ids: Any,
    coefficient_arg_names: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> NativeSampledKernelBundle:
    """Build a sampled bundle from a generated C++/JIT kernel runner."""

    from .native_assembly import native_kernel_metadata_from_runner

    return build_sampled_native_kernel_bundle(
        param_order=tuple(str(v) for v in getattr(runner, "param_order")),
        static_args=static_args,
        element_ids=element_ids,
        coefficient_arg_names=coefficient_arg_names,
        metadata_capsule=native_kernel_metadata_from_runner(runner),
        metadata=metadata,
    )


__all__ = [
    "NativeSampledKernelBundle",
    "build_sampled_native_kernel_bundle",
    "build_sampled_native_kernel_bundle_from_runner",
    "build_sampled_static_args",
]
