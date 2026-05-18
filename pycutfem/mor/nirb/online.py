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
    input_path: str
    predictions_path: str
    restricted_output: bool = False

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "OnlineConfig":
        return cls(**dict(mapping))


def load_input_matrix(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".npy":
        return _as_snapshot_matrix(np.load(source))
    if source.suffix == ".npz":
        with np.load(source) as data:
            for key in ("input", "input_snapshots"):
                if key in data:
                    return _as_snapshot_matrix(data[key])
        raise ValueError(f"could not find an input matrix in {source}")
    raise ValueError(f"unsupported input file suffix: {source.suffix}")


def run_online_pipeline(config: OnlineConfig | dict[str, Any]) -> np.ndarray:
    if isinstance(config, dict):
        config = OnlineConfig.from_mapping(config)

    model: TrainedNIRBModel = load_model(config.model_path)
    input_values = load_input_matrix(config.input_path)
    predictions = (
        model.predict_restricted(input_values)
        if bool(config.restricted_output)
        else model.predict(input_values)
    )

    target = Path(config.predictions_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        predictions=predictions,
        restricted_output=np.array(config.restricted_output, dtype=bool),
    )
    return predictions


__all__ = [
    "OnlineConfig",
    "load_input_matrix",
    "run_online_pipeline",
]
