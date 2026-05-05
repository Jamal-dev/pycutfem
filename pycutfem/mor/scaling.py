from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_snapshot_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("expected a 1D or 2D array of snapshots")
    return matrix


@dataclass
class MeanCenterer:
    """Mean subtraction for feature-major snapshot matrices."""

    mean_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "MeanCenterer":
        matrix = _as_snapshot_matrix(values)
        self.mean_ = matrix.mean(axis=1, keepdims=True)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("MeanCenterer must be fit before transform")
        matrix = _as_snapshot_matrix(values)
        return matrix - self.mean_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("MeanCenterer must be fit before inverse_transform")
        matrix = _as_snapshot_matrix(values)
        return matrix + self.mean_

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)


@dataclass
class StandardScaler:
    """Mean subtraction and variance scaling for feature-major matrices."""

    eps: float = 1.0e-12
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "StandardScaler":
        matrix = _as_snapshot_matrix(values)
        self.mean_ = matrix.mean(axis=1, keepdims=True)
        scale = matrix.std(axis=1, keepdims=True)
        self.scale_ = np.where(scale < self.eps, 1.0, scale)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fit before transform")
        matrix = _as_snapshot_matrix(values)
        return (matrix - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fit before inverse_transform")
        matrix = _as_snapshot_matrix(values)
        return matrix * self.scale_ + self.mean_

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)
