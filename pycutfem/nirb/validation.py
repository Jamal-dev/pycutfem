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
    fom_solid_time: float | None = None
    rom_solid_time: float | None = None
    fom_total_time: float | None = None
    rom_total_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "ValidationConfig":
        return cls(**mapping)


def validate_rom(config: ValidationConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        config = ValidationConfig.from_mapping(config)

    reference = _load_snapshot_matrix(
        config.reference_path,
        preferred_keys=("reference", "displacements", "full_displacements", "predictions"),
    )
    prediction = _load_snapshot_matrix(
        config.prediction_path,
        preferred_keys=("predictions", "displacements", "full_displacements"),
    )

    metrics = {
        "mean_sample_l2_error": mean_sample_l2_error(reference, prediction),
        "snapshot_l2_error": snapshot_l2_error(reference, prediction),
        "max_online_relative_displacement_error": max_online_relative_displacement_error(reference, prediction),
        "online_relative_displacement_error": online_relative_displacement_error(reference, prediction).tolist(),
    }

    if config.fom_iterations is not None and config.rom_iterations is not None:
        metrics["iteration_overhead"] = accumulated_iteration_overhead(
            np.asarray(config.fom_iterations, dtype=float),
            np.asarray(config.rom_iterations, dtype=float),
        )

    if config.fom_solid_time is not None and config.rom_solid_time is not None:
        metrics.update(
            build_speedup_report(
                fom_solid_time=config.fom_solid_time,
                rom_solid_time=config.rom_solid_time,
                fom_total_time=config.fom_total_time,
                rom_total_time=config.rom_total_time,
            )
        )

    passes = {}
    for name, threshold in config.thresholds.items():
        if name not in metrics:
            raise KeyError(f"threshold specified for unknown metric: {name}")
        passes[name] = bool(metrics[name] <= threshold)

    result = {"metrics": metrics, "passes": passes, "metadata": config.metadata}
    save_results(result, config.metrics_path)
    return result
