from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pycutfem.mor.io import save_results
from pycutfem.mor.metrics import (
    accumulated_iteration_overhead,
    max_online_relative_displacement_error,
    mean_sample_l2_error,
    online_relative_displacement_error,
    snapshot_l2_error,
)
from pycutfem.mor.timing import build_speedup_report


def _load_snapshot_matrix(path: str | Path, preferred_keys: tuple[str, ...]) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".npy":
        matrix = np.asarray(np.load(source), dtype=float)
    elif source.suffix == ".npz":
        with np.load(source) as data:
            for key in preferred_keys:
                if key in data:
                    matrix = np.asarray(data[key], dtype=float)
                    break
            else:
                raise ValueError(f"could not find any of {preferred_keys} in {source}")
    else:
        raise ValueError(f"unsupported array suffix: {source.suffix}")
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    return matrix


@dataclass
class ValidationConfig:
    reference_path: str
    prediction_path: str
    metrics_path: str
    thresholds: dict[str, float] = field(default_factory=dict)
    fom_iterations: list[float] | None = None
    rom_iterations: list[float] | None = None
    fom_model_time: float | None = None
    rom_model_time: float | None = None
    fom_total_time: float | None = None
    rom_total_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "ValidationConfig":
        values = dict(mapping)
        if "fom_solid_time" in values and "fom_model_time" not in values:
            values["fom_model_time"] = values.pop("fom_solid_time")
        if "rom_solid_time" in values and "rom_model_time" not in values:
            values["rom_model_time"] = values.pop("rom_solid_time")
        return cls(**values)


def validate_rom(config: ValidationConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        config = ValidationConfig.from_mapping(config)

    reference = _load_snapshot_matrix(
        config.reference_path,
        preferred_keys=("reference", "output", "output_snapshots", "predictions"),
    )
    prediction = _load_snapshot_matrix(
        config.prediction_path,
        preferred_keys=("predictions", "output", "output_snapshots"),
    )

    metrics = {
        "mean_sample_l2_error": mean_sample_l2_error(reference, prediction),
        "snapshot_l2_error": snapshot_l2_error(reference, prediction),
        "max_online_relative_output_error": max_online_relative_displacement_error(reference, prediction),
        "online_relative_output_error": online_relative_displacement_error(reference, prediction).tolist(),
    }
    metrics["max_online_relative_displacement_error"] = metrics["max_online_relative_output_error"]
    metrics["online_relative_displacement_error"] = metrics["online_relative_output_error"]

    if config.fom_iterations is not None and config.rom_iterations is not None:
        metrics["iteration_overhead"] = accumulated_iteration_overhead(
            np.asarray(config.fom_iterations, dtype=float),
            np.asarray(config.rom_iterations, dtype=float),
        )

    if config.fom_model_time is not None and config.rom_model_time is not None:
        metrics.update(
            build_speedup_report(
                fom_solid_time=config.fom_model_time,
                rom_solid_time=config.rom_model_time,
                fom_total_time=config.fom_total_time,
                rom_total_time=config.rom_total_time,
            )
        )
        metrics["model_speedup"] = metrics["solid_speedup"]

    passes = {}
    for name, threshold in config.thresholds.items():
        if name not in metrics:
            raise KeyError(f"threshold specified for unknown metric: {name}")
        passes[name] = bool(metrics[name] <= threshold)

    result = {"metrics": metrics, "passes": passes, "metadata": config.metadata}
    save_results(result, config.metrics_path)
    return result
