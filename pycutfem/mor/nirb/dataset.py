from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pycutfem.mor.snapshots import NamedSnapshotBatch


def _as_snapshot_matrix(values: np.ndarray, *, name: str = "snapshots") -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a feature-major snapshot matrix")
    return matrix


def _sample_vector(values: Any | None, n_snapshots: int, *, name: str, dtype: Any) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if arr.size != int(n_snapshots):
        raise ValueError(f"{name} length must match the snapshot count")
    return arr


def _context_vector(values: Any, n_snapshots: int, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.full((int(n_snapshots),), float(arr), dtype=float)
    arr = arr.reshape(-1)
    if arr.size != int(n_snapshots):
        raise ValueError(f"context feature {name!r} length must match the snapshot count")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"context feature {name!r} contains non-finite values")
    return arr


def _load_json_scalar(value: np.ndarray) -> dict[str, Any]:
    if value.shape != ():
        return {}
    try:
        return json.loads(str(value.item()))
    except Exception:
        return {}


@dataclass(frozen=True)
class DatasetSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray


@dataclass
class NIRBDataset:
    """Input-output snapshot data for non-intrusive reduced-basis training.

    The dataset stores paired columns ``x_j`` and ``y_j`` with shapes
    ``(n_input_dofs, n_snapshots)`` and ``(n_output_dofs, n_snapshots)``.  The
    names are intentionally problem-independent: examples may interpret the
    input as parameters, loads, boundary data, sensor values, or reduced
    coordinates, and may interpret the output as a state, correction, QoI, or
    interface quantity.
    """

    input_snapshots: np.ndarray
    output_snapshots: np.ndarray
    parameters: np.ndarray | None = None
    times: np.ndarray | None = None
    converged: np.ndarray | None = None
    context_features: Mapping[str, np.ndarray] = field(default_factory=dict)
    output_indices: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.input_snapshots = _as_snapshot_matrix(self.input_snapshots, name="input_snapshots")
        self.output_snapshots = _as_snapshot_matrix(self.output_snapshots, name="output_snapshots")
        if self.input_snapshots.shape[1] != self.output_snapshots.shape[1]:
            raise ValueError("input and output snapshot counts must match")
        n_snapshots = self.n_snapshots
        self.parameters = None if self.parameters is None else np.asarray(self.parameters, dtype=float)
        if self.parameters is not None and self.parameters.shape[0] != n_snapshots:
            raise ValueError("parameters must be sample-major with one row per snapshot")
        self.times = _sample_vector(self.times, n_snapshots, name="times", dtype=float)
        self.converged = _sample_vector(self.converged, n_snapshots, name="converged", dtype=bool)
        self.context_features = {
            str(name): _context_vector(values, n_snapshots, name=str(name))
            for name, values in dict(self.context_features).items()
        }
        self.output_indices = None if self.output_indices is None else np.asarray(self.output_indices, dtype=int)
        self.metadata = dict(self.metadata)

    @property
    def n_snapshots(self) -> int:
        return int(self.input_snapshots.shape[1])

    def context(self, name: str) -> np.ndarray:
        key = str(name)
        if key in self.context_features:
            return np.asarray(self.context_features[key], dtype=float)
        if key == "time" and self.times is not None:
            return np.asarray(self.times, dtype=float)
        if key.startswith("parameter_") and self.parameters is not None:
            index = int(key.removeprefix("parameter_"))
            return np.asarray(self.parameters[:, index], dtype=float)
        raise KeyError(f"context feature {name!r} is not available")

    def subset(self, indices: np.ndarray) -> "NIRBDataset":
        chosen = np.asarray(indices, dtype=int)
        return NIRBDataset(
            input_snapshots=self.input_snapshots[:, chosen],
            output_snapshots=self.output_snapshots[:, chosen],
            parameters=None if self.parameters is None else self.parameters[chosen],
            times=None if self.times is None else self.times[chosen],
            converged=None if self.converged is None else self.converged[chosen],
            context_features={name: values[chosen] for name, values in self.context_features.items()},
            output_indices=self.output_indices,
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


def dataset_from_named_snapshot_batch(
    batch: NamedSnapshotBatch,
    *,
    input_field: str = "input",
    output_field: str = "output",
    output_indices: np.ndarray | None = None,
) -> NIRBDataset:
    input_name = str(input_field)
    output_name = str(output_field)
    if input_name not in batch.fields or output_name not in batch.fields:
        names = ", ".join(batch.field_names)
        raise KeyError(
            f"NamedSnapshotBatch must contain fields {input_name!r} and {output_name!r}; available fields: {names}"
        )
    context_features: dict[str, np.ndarray] = {}
    if batch.times is not None:
        context_features["time"] = np.asarray(batch.times, dtype=float)
    if batch.parameters is not None:
        for index in range(batch.parameters.shape[1]):
            context_features[f"parameter_{index}"] = np.asarray(batch.parameters[:, index], dtype=float)
    if batch.sample_metadata:
        keys = set().union(*(dict(item).keys() for item in batch.sample_metadata))
        for key in sorted(str(item) for item in keys):
            values = [dict(item).get(key) for item in batch.sample_metadata]
            if all(value is not None for value in values):
                try:
                    context_features[key] = np.asarray(values, dtype=float)
                except (TypeError, ValueError):
                    pass
    return NIRBDataset(
        input_snapshots=batch[input_name],
        output_snapshots=batch[output_name],
        parameters=batch.parameters,
        times=batch.times,
        converged=batch.converged,
        context_features=context_features,
        output_indices=output_indices,
        metadata={
            "source": "NamedSnapshotBatch",
            "input_field": input_name,
            "output_field": output_name,
            **dict(batch.metadata),
        },
    )


def _load_npz_dataset(
    path: Path,
    *,
    input_field: str,
    output_field: str,
    output_indices: np.ndarray | None,
) -> NIRBDataset:
    with np.load(path, allow_pickle=True) as data:
        if "manifest_json" in data:
            batch = NamedSnapshotBatch.load(path)
            return dataset_from_named_snapshot_batch(
                batch,
                input_field=input_field,
                output_field=output_field,
                output_indices=output_indices,
            )

        if "input_snapshots" in data and "output_snapshots" in data:
            input_snapshots = data["input_snapshots"]
            output_snapshots = data["output_snapshots"]
        elif "input" in data and "output" in data:
            input_snapshots = data["input"]
            output_snapshots = data["output"]
        else:
            raise ValueError(
                "NIRB datasets must be a NamedSnapshotBatch or contain "
                "'input_snapshots'/'output_snapshots' arrays."
            )

        metadata = _load_json_scalar(data["metadata"]) if "metadata" in data else {}
        parameters = data["parameters"] if "parameters" in data and data["parameters"].shape != () else None
        times = data["times"] if "times" in data and data["times"].shape != () else None
        converged = data["converged"] if "converged" in data and data["converged"].shape != () else None
        context_features: dict[str, np.ndarray] = {}
        if times is not None:
            context_features["time"] = np.asarray(times, dtype=float)
        if "subiterations" in data and data["subiterations"].shape != ():
            context_features["subiteration"] = np.asarray(data["subiterations"], dtype=float)
            context_features["coupling_iter"] = np.asarray(data["subiterations"], dtype=float)
        if parameters is not None:
            for index in range(parameters.shape[1]):
                context_features[f"parameter_{index}"] = np.asarray(parameters[:, index], dtype=float)

    return NIRBDataset(
        input_snapshots=input_snapshots,
        output_snapshots=output_snapshots,
        parameters=parameters,
        times=times,
        converged=converged,
        context_features=context_features,
        output_indices=output_indices,
        metadata={"source": "npz", "path": str(path), **metadata},
    )


def load_dataset(
    path: str | Path,
    *,
    input_field: str = "input",
    output_field: str = "output",
    output_indices: np.ndarray | None = None,
) -> NIRBDataset:
    source = Path(path)
    if source.is_dir():
        raise ValueError(
            "pycutfem.mor.nirb.load_dataset expects a generic snapshot file. "
            "Use an example-level loader to convert solver-specific directories into a NamedSnapshotBatch."
        )
    if source.suffix != ".npz":
        raise ValueError(f"unsupported NIRB dataset suffix: {source.suffix}")
    return _load_npz_dataset(
        source,
        input_field=input_field,
        output_field=output_field,
        output_indices=output_indices,
    )


__all__ = [
    "DatasetSplit",
    "NIRBDataset",
    "dataset_from_named_snapshot_batch",
    "load_dataset",
]
