"""Problem-generic reduced-coordinate reference predictors.

The predictors in this module operate only on reduced coefficients, time,
optional parameter vectors, and history.  They intentionally do not know about
PDE fields, meshes, or example-specific state layouts, so the same artifacts can
be reused by native LSPG/GNAT drivers for different nonlinear systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np


REDUCED_REFERENCE_PREDICTOR_SCHEMA_VERSION = 1


def _finite_vector(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _finite_matrix(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2-D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _coerce_snapshot_matrix(q_snapshots: Any) -> np.ndarray:
    arr = np.asarray(q_snapshots, dtype=float)
    if arr.ndim == 2:
        pass
    elif arr.ndim == 3:
        # (n_trajectories, n_steps, n_modes)
        if arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] == 0:
            raise ValueError("q_snapshots must not have empty trajectory, step, or mode axes.")
        arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    else:
        raise ValueError("q_snapshots must have shape (n_steps, n_modes), (n_modes, n_steps), or (n_trajectories, n_steps, n_modes).")
    return _finite_matrix(arr, "q_snapshots")


def _normalize_times(times: Any) -> tuple[np.ndarray, float, float]:
    t = _finite_vector(times, "times")
    if t.size == 0:
        raise ValueError("times must not be empty.")
    center = float(0.5 * (float(np.min(t)) + float(np.max(t))))
    half_width = float(0.5 * (float(np.max(t)) - float(np.min(t))))
    if half_width <= 0.0:
        half_width = 1.0
    tau = np.clip((t - center) / half_width, -1.0, 1.0)
    return np.ascontiguousarray(tau, dtype=np.float64), center, half_width


def _chebyshev_features(tau: np.ndarray, degree: int) -> np.ndarray:
    if degree < 0:
        raise ValueError("degree must be nonnegative.")
    x = np.asarray(tau, dtype=float).reshape(-1)
    out = np.empty((x.size, int(degree) + 1), dtype=np.float64)
    out[:, 0] = 1.0
    if degree >= 1:
        out[:, 1] = x
    for k in range(2, int(degree) + 1):
        out[:, k] = 2.0 * x * out[:, k - 1] - out[:, k - 2]
    return np.ascontiguousarray(out, dtype=np.float64)


def _parameter_matrix(parameters: Mapping[str, Any] | np.ndarray | None, n_samples: int) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    if parameters is None:
        return (
            np.ones((n_samples, 1), dtype=np.float64),
            (),
            np.zeros(0, dtype=np.float64),
            np.ones(0, dtype=np.float64),
        )
    if isinstance(parameters, Mapping):
        names = tuple(sorted(str(name) for name in parameters))
        columns = []
        for name in names:
            value = np.asarray(parameters[name], dtype=float)
            if value.ndim == 0:
                value = np.full(n_samples, float(value), dtype=float)
            value = value.reshape(-1)
            if value.size == 1 and n_samples > 1:
                value = np.full(n_samples, float(value[0]), dtype=float)
            if value.size != n_samples:
                raise ValueError(f"parameter {name!r} must have {n_samples} entries.")
            columns.append(value)
        raw = np.column_stack(columns) if columns else np.empty((n_samples, 0), dtype=np.float64)
    else:
        raw = np.asarray(parameters, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        if raw.ndim != 2:
            raise ValueError("parameters must be a mapping, vector, matrix, or None.")
        if raw.shape[0] == 1 and n_samples > 1:
            raw = np.repeat(raw, n_samples, axis=0)
        if raw.shape[0] != n_samples:
            raise ValueError(f"parameters must have {n_samples} rows.")
        names = tuple(f"p{i}" for i in range(raw.shape[1]))
    if not np.all(np.isfinite(raw)):
        raise ValueError("parameters must contain only finite values.")
    if raw.shape[1] == 0:
        return (
            np.ones((n_samples, 1), dtype=np.float64),
            (),
            np.zeros(0, dtype=np.float64),
            np.ones(0, dtype=np.float64),
        )
    center = np.mean(raw, axis=0)
    scale = np.std(raw, axis=0)
    scale[scale <= 0.0] = 1.0
    normalized = (raw - center) / scale
    features = np.column_stack([np.ones(n_samples, dtype=np.float64), normalized])
    return (
        np.ascontiguousarray(features, dtype=np.float64),
        names,
        np.ascontiguousarray(center, dtype=np.float64),
        np.ascontiguousarray(scale, dtype=np.float64),
    )


def _compose_features(time_features: np.ndarray, parameter_features: np.ndarray) -> np.ndarray:
    if parameter_features.shape[1] == 1:
        return np.ascontiguousarray(time_features, dtype=np.float64)
    rows = []
    for i in range(time_features.shape[0]):
        rows.append(np.kron(parameter_features[i, :], time_features[i, :]))
    return np.ascontiguousarray(np.vstack(rows), dtype=np.float64)


@dataclass(frozen=True)
class ReducedReferencePrediction:
    """A predicted reduced reference coefficient vector."""

    coefficients: np.ndarray
    predictor_kind: str
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", _finite_vector(self.coefficients, "prediction coefficients"))
        object.__setattr__(self, "predictor_kind", str(self.predictor_kind))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


class ReducedReferencePredictor(Protocol):
    """Protocol implemented by reduced-coordinate reference predictors."""

    kind: str

    def predict(
        self,
        *,
        time: float,
        dt: float | None = None,
        q_current: Any | None = None,
        q_previous: Any | None = None,
        parameters: Mapping[str, Any] | np.ndarray | None = None,
    ) -> ReducedReferencePrediction:
        ...

    def to_native_dict(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class ConstantReducedPredictor:
    """Use the current reduced state as the next reference."""

    kind: str = "constant"

    def predict(
        self,
        *,
        time: float,
        dt: float | None = None,
        q_current: Any | None = None,
        q_previous: Any | None = None,
        parameters: Mapping[str, Any] | np.ndarray | None = None,
    ) -> ReducedReferencePrediction:
        if q_current is None:
            raise ValueError("ConstantReducedPredictor requires q_current.")
        return ReducedReferencePrediction(_finite_vector(q_current, "q_current"), self.kind, {"time": float(time)})

    def to_native_dict(self) -> dict[str, Any]:
        return {"schema_version": REDUCED_REFERENCE_PREDICTOR_SCHEMA_VERSION, "kind": self.kind}


@dataclass(frozen=True)
class LinearHistoryReducedPredictor:
    """Use a secant extrapolation from the previous two reduced states."""

    kind: str = "linear_history"

    def predict(
        self,
        *,
        time: float,
        dt: float | None = None,
        q_current: Any | None = None,
        q_previous: Any | None = None,
        parameters: Mapping[str, Any] | np.ndarray | None = None,
    ) -> ReducedReferencePrediction:
        if q_current is None:
            raise ValueError("LinearHistoryReducedPredictor requires q_current.")
        q = _finite_vector(q_current, "q_current")
        if q_previous is None:
            return ReducedReferencePrediction(q, self.kind, {"time": float(time), "fallback": "constant"})
        qp = _finite_vector(q_previous, "q_previous")
        if qp.shape != q.shape:
            raise ValueError("q_previous shape must match q_current.")
        return ReducedReferencePrediction(q + (q - qp), self.kind, {"time": float(time), "fallback": None})

    def to_native_dict(self) -> dict[str, Any]:
        return {"schema_version": REDUCED_REFERENCE_PREDICTOR_SCHEMA_VERSION, "kind": self.kind}


@dataclass(frozen=True)
class TimeParameterizedReducedPredictor:
    """Chebyshev time/parameter regressor for reduced reference states."""

    coefficients: np.ndarray
    time_center: float
    time_half_width: float
    degree: int
    parameter_names: tuple[str, ...] = ()
    parameter_center: np.ndarray | None = None
    parameter_scale: np.ndarray | None = None
    ridge: float = 0.0
    training_error_max: float = 0.0
    training_error_mean: float = 0.0
    metadata: Mapping[str, Any] | None = None
    kind: str = "time_parameterized"

    def __post_init__(self) -> None:
        coeffs = _finite_matrix(self.coefficients, "predictor coefficients")
        names = tuple(str(name) for name in self.parameter_names)
        p_center = np.zeros(len(names), dtype=np.float64) if self.parameter_center is None else _finite_vector(self.parameter_center, "parameter_center")
        p_scale = np.ones(len(names), dtype=np.float64) if self.parameter_scale is None else _finite_vector(self.parameter_scale, "parameter_scale")
        if p_center.size != len(names) or p_scale.size != len(names):
            raise ValueError("parameter_center and parameter_scale sizes must match parameter_names.")
        if int(self.degree) < 0:
            raise ValueError("degree must be nonnegative.")
        object.__setattr__(self, "coefficients", coeffs)
        object.__setattr__(self, "time_center", float(self.time_center))
        object.__setattr__(self, "time_half_width", float(self.time_half_width) if float(self.time_half_width) > 0.0 else 1.0)
        object.__setattr__(self, "degree", int(self.degree))
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "parameter_center", np.ascontiguousarray(p_center, dtype=np.float64))
        object.__setattr__(self, "parameter_scale", np.ascontiguousarray(p_scale, dtype=np.float64))
        object.__setattr__(self, "ridge", float(self.ridge))
        object.__setattr__(self, "training_error_max", float(self.training_error_max))
        object.__setattr__(self, "training_error_mean", float(self.training_error_mean))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_modes(self) -> int:
        return int(self.coefficients.shape[1])

    def _feature_row(self, time: float, parameters: Mapping[str, Any] | np.ndarray | None = None) -> np.ndarray:
        tau = np.asarray([(float(time) - self.time_center) / self.time_half_width], dtype=np.float64)
        time_features = _chebyshev_features(tau, self.degree)
        if not self.parameter_names:
            return np.ascontiguousarray(time_features[0], dtype=np.float64)
        if parameters is None:
            raw = np.asarray(self.parameter_center, dtype=np.float64)
        elif isinstance(parameters, Mapping):
            raw = np.asarray([float(np.asarray(parameters[name], dtype=float).reshape(-1)[0]) for name in self.parameter_names], dtype=np.float64)
        else:
            raw = np.asarray(parameters, dtype=float).reshape(-1)
        if raw.size != len(self.parameter_names):
            raise ValueError("prediction parameters size must match fitted parameter count.")
        param = np.concatenate(([1.0], (raw - self.parameter_center) / self.parameter_scale))
        return np.ascontiguousarray(np.kron(param, time_features[0]), dtype=np.float64)

    def predict(
        self,
        *,
        time: float,
        dt: float | None = None,
        q_current: Any | None = None,
        q_previous: Any | None = None,
        parameters: Mapping[str, Any] | np.ndarray | None = None,
    ) -> ReducedReferencePrediction:
        row = self._feature_row(float(time), parameters)
        if row.size != self.coefficients.shape[0]:
            raise ValueError("predictor feature size does not match fitted coefficient rows.")
        q = row @ self.coefficients
        return ReducedReferencePrediction(
            q,
            self.kind,
            {
                "time": float(time),
                "dt": None if dt is None else float(dt),
                "training_error_max": float(self.training_error_max),
                "training_error_mean": float(self.training_error_mean),
            },
        )

    def to_native_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REDUCED_REFERENCE_PREDICTOR_SCHEMA_VERSION,
            "kind": self.kind,
            "coefficients": self.coefficients,
            "time_center": float(self.time_center),
            "time_half_width": float(self.time_half_width),
            "degree": int(self.degree),
            "parameter_names": tuple(self.parameter_names),
            "parameter_center": self.parameter_center,
            "parameter_scale": self.parameter_scale,
            "ridge": float(self.ridge),
            "training_error_max": float(self.training_error_max),
            "training_error_mean": float(self.training_error_mean),
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_native_dict(cls, payload: Mapping[str, Any]) -> "TimeParameterizedReducedPredictor":
        return cls(
            coefficients=payload["coefficients"],
            time_center=float(payload["time_center"]),
            time_half_width=float(payload["time_half_width"]),
            degree=int(payload["degree"]),
            parameter_names=tuple(payload.get("parameter_names", ())),
            parameter_center=payload.get("parameter_center"),
            parameter_scale=payload.get("parameter_scale"),
            ridge=float(payload.get("ridge", 0.0)),
            training_error_max=float(payload.get("training_error_max", 0.0)),
            training_error_mean=float(payload.get("training_error_mean", 0.0)),
            metadata=payload.get("metadata", {}),
        )


def fit_time_parameterized_predictor(
    q_snapshots: Any,
    times: Any,
    *,
    parameters: Mapping[str, Any] | np.ndarray | None = None,
    degree: int = 12,
    ridge: float = 1.0e-10,
    metadata: Mapping[str, Any] | None = None,
) -> TimeParameterizedReducedPredictor:
    """Fit a time/parameter reduced reference predictor.

    ``q_snapshots`` is problem-generic reduced data.  It may be supplied as
    ``(n_steps, n_modes)``, ``(n_modes, n_steps)``, or
    ``(n_trajectories, n_steps, n_modes)``.  The returned predictor stores only
    dense arrays and scalar metadata, so it is suitable for native artifacts.
    """

    q = _coerce_snapshot_matrix(q_snapshots)
    t_raw = np.asarray(times, dtype=float)
    if t_raw.ndim == 2:
        t = t_raw.reshape(-1)
    else:
        t = t_raw.reshape(-1)
    if t.size != q.shape[0] and q.ndim == 2 and t.size == q.shape[1]:
        q = np.ascontiguousarray(q.T, dtype=np.float64)
    if t.size != q.shape[0]:
        raise ValueError("times must provide one value per reduced snapshot.")
    tau, t_center, t_half_width = _normalize_times(t)
    time_features = _chebyshev_features(tau, int(degree))
    param_features, names, p_center, p_scale = _parameter_matrix(parameters, int(q.shape[0]))
    X = _compose_features(time_features, param_features)
    lam = float(ridge)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError("ridge must be finite and nonnegative.")
    if lam > 0.0:
        normal = X.T @ X
        rhs = X.T @ q
        coeffs = np.linalg.solve(normal + lam * np.eye(normal.shape[0], dtype=np.float64), rhs)
    else:
        coeffs = np.linalg.lstsq(X, q, rcond=None)[0]
    fitted = X @ coeffs
    denom = np.maximum(np.linalg.norm(q, axis=1), 1.0e-300)
    rel = np.linalg.norm(fitted - q, axis=1) / denom
    return TimeParameterizedReducedPredictor(
        coefficients=np.ascontiguousarray(coeffs, dtype=np.float64),
        time_center=t_center,
        time_half_width=t_half_width,
        degree=int(degree),
        parameter_names=names,
        parameter_center=p_center,
        parameter_scale=p_scale,
        ridge=lam,
        training_error_max=float(np.max(rel)) if rel.size else 0.0,
        training_error_mean=float(np.mean(rel)) if rel.size else 0.0,
        metadata=metadata,
    )


def predictor_from_native_dict(payload: Mapping[str, Any]) -> ReducedReferencePredictor:
    kind = str(payload.get("kind", "")).lower()
    if kind in {"constant", "previous"}:
        return ConstantReducedPredictor()
    if kind in {"linear_history", "linear", "secant"}:
        return LinearHistoryReducedPredictor()
    if kind in {"time_parameterized", "time_regression", "time_regressor", "learned"}:
        return TimeParameterizedReducedPredictor.from_native_dict(payload)
    raise ValueError(f"unknown reduced reference predictor kind {kind!r}.")


__all__ = [
    "REDUCED_REFERENCE_PREDICTOR_SCHEMA_VERSION",
    "ConstantReducedPredictor",
    "LinearHistoryReducedPredictor",
    "ReducedReferencePrediction",
    "ReducedReferencePredictor",
    "TimeParameterizedReducedPredictor",
    "fit_time_parameterized_predictor",
    "predictor_from_native_dict",
]
