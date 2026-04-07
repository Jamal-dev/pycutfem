from __future__ import annotations

import numpy as np


def _as_snapshot_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("expected a 1D or 2D array")
    return matrix


def mean_sample_l2_error(reference: np.ndarray, prediction: np.ndarray) -> float:
    ref = _as_snapshot_matrix(reference)
    pred = _as_snapshot_matrix(prediction)
    if ref.shape != pred.shape:
        raise ValueError("reference and prediction must have matching shapes")
    return float(np.mean(np.linalg.norm(ref - pred, axis=0)))


def snapshot_l2_error(reference: np.ndarray, prediction: np.ndarray) -> float:
    ref = _as_snapshot_matrix(reference)
    pred = _as_snapshot_matrix(prediction)
    if ref.shape != pred.shape:
        raise ValueError("reference and prediction must have matching shapes")
    return float(np.linalg.norm(ref - pred))


def reduced_regression_error(reference: np.ndarray, prediction: np.ndarray) -> float:
    ref = _as_snapshot_matrix(reference)
    pred = _as_snapshot_matrix(prediction)
    if ref.shape != pred.shape:
        raise ValueError("reference and prediction must have matching shapes")
    return float(np.linalg.norm(ref - pred))


def online_relative_displacement_error(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    ref = _as_snapshot_matrix(reference)
    pred = _as_snapshot_matrix(prediction)
    if ref.shape != pred.shape:
        raise ValueError("reference and prediction must have matching shapes")
    denominator = np.mean(np.linalg.norm(ref, axis=0))
    if denominator <= 0.0:
        raise ValueError("reference norm mean must be positive")
    return np.linalg.norm(ref - pred, axis=0) / denominator


def max_online_relative_displacement_error(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.max(online_relative_displacement_error(reference, prediction)))


def accumulated_iteration_overhead(fom_iterations: np.ndarray, rom_iterations: np.ndarray) -> float:
    fom = np.asarray(fom_iterations, dtype=float).ravel()
    rom = np.asarray(rom_iterations, dtype=float).ravel()
    if fom.shape != rom.shape:
        raise ValueError("iteration arrays must have matching shapes")
    baseline = fom.sum()
    if baseline <= 0.0:
        raise ValueError("sum of FOM iterations must be positive")
    return float((rom.sum() - baseline) / baseline)


def speedup(fom_time: float, rom_time: float) -> float:
    if rom_time <= 0.0:
        raise ValueError("rom_time must be positive")
    return float(fom_time / rom_time)
