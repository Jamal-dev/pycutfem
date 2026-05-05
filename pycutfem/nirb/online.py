from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pycutfem.mor.io import load_model

from .offline import TrainedNIRBModel


def _as_snapshot_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("expected a feature-major snapshot matrix")
    return matrix


@dataclass
class OnlineConfig:
    model_path: str
    forces_path: str
    predictions_path: str
    interface_only: bool = False

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "OnlineConfig":
        return cls(**mapping)


def load_force_matrix(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".npy":
        return _as_snapshot_matrix(np.load(source))
    if source.suffix == ".npz":
        with np.load(source) as data:
            if "forces" in data:
                return _as_snapshot_matrix(data["forces"])
            if "interface_forces" in data:
                return _as_snapshot_matrix(data["interface_forces"])
        raise ValueError(f"could not find a force matrix in {source}")
    raise ValueError(f"unsupported force file suffix: {source.suffix}")


def run_online_pipeline(config: OnlineConfig | dict[str, Any]) -> np.ndarray:
    if isinstance(config, dict):
        config = OnlineConfig.from_mapping(config)

    model: TrainedNIRBModel = load_model(config.model_path)
    forces = load_force_matrix(config.forces_path)
    predictions = model.predict_interface(forces) if config.interface_only else model.predict_full(forces)

    target = Path(config.predictions_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        predictions=predictions,
        interface_only=np.array(config.interface_only, dtype=bool),
    )
    return predictions
