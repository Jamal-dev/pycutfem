from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


def _as_flat_vector(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1).copy()


def _solve_linear_system(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def _regularized_lstsq(matrix: np.ndarray, rhs: np.ndarray, regularization: float) -> np.ndarray:
    reg = max(float(regularization), 0.0)
    if reg > 0.0:
        n_cols = int(matrix.shape[1])
        matrix_aug = np.vstack([matrix, math.sqrt(reg) * np.eye(n_cols, dtype=float)])
        rhs_aug = np.concatenate([rhs, np.zeros((n_cols,), dtype=float)])
        return np.linalg.lstsq(matrix_aug, rhs_aug, rcond=None)[0]
    return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def _clip_relaxation(value: float, lower: float, upper: float) -> float:
    lo = float(lower)
    hi = float(upper)
    if lo > hi:
        raise ValueError(f"Invalid relaxation bounds: min={lo} exceeds max={hi}.")
    return float(np.clip(float(value), lo, hi))


@dataclass(frozen=True)
class CouplingAccelerationStep:
    next_iterate: np.ndarray
    delta: np.ndarray
    relaxation: float
    used_history: bool
    method: str


class CouplingAccelerator:
    def initialize_solution_step(self) -> None:
        pass

    def finalize_solution_step(self, *, accepted: bool = True) -> None:
        pass

    def compute_next_iterate(self, *, x_curr: np.ndarray, residual_curr: np.ndarray) -> CouplingAccelerationStep:
        raise NotImplementedError


class ConstantRelaxationCouplingAccelerator(CouplingAccelerator):
    def __init__(self, *, relaxation: float) -> None:
        self.relaxation = float(relaxation)

    def compute_next_iterate(self, *, x_curr: np.ndarray, residual_curr: np.ndarray) -> CouplingAccelerationStep:
        x_vec = _as_flat_vector(x_curr)
        r_vec = _as_flat_vector(residual_curr)
        if x_vec.size != r_vec.size:
            raise ValueError("Constant relaxation requires x_curr and residual_curr with the same size.")
        delta = self.relaxation * r_vec
        return CouplingAccelerationStep(
            next_iterate=x_vec + delta,
            delta=delta,
            relaxation=float(self.relaxation),
            used_history=False,
            method="constant",
        )


class AitkenCouplingAccelerator(CouplingAccelerator):
    def __init__(
        self,
        *,
        init_relaxation: float,
        relaxation_min: float,
        relaxation_max: float,
        init_relaxation_max: Optional[float] = None,
    ) -> None:
        self.relaxation_min = float(relaxation_min)
        self.relaxation_max = float(relaxation_max)
        self.init_relaxation_max = (
            float(relaxation_max) if init_relaxation_max is None else float(init_relaxation_max)
        )
        self.alpha_old = _clip_relaxation(float(init_relaxation), self.relaxation_min, self.relaxation_max)
        self._prev_residual: Optional[np.ndarray] = None
        self._initial_iteration = True

    def initialize_solution_step(self) -> None:
        self._prev_residual = None
        self._initial_iteration = True

    def compute_next_iterate(self, *, x_curr: np.ndarray, residual_curr: np.ndarray) -> CouplingAccelerationStep:
        x_vec = _as_flat_vector(x_curr)
        r_vec = _as_flat_vector(residual_curr)
        if x_vec.size != r_vec.size:
            raise ValueError("Aitken relaxation requires x_curr and residual_curr with the same size.")

        used_history = not self._initial_iteration
        if self._initial_iteration:
            alpha = _clip_relaxation(
                min(float(self.alpha_old), float(self.init_relaxation_max)),
                self.relaxation_min,
                self.relaxation_max,
            )
            self._initial_iteration = False
        else:
            alpha = float(self.alpha_old)
            if self._prev_residual is not None:
                delta_r = r_vec - self._prev_residual
                denom = float(np.dot(delta_r, delta_r))
                if denom > 1.0e-30 and np.isfinite(denom):
                    alpha_new = -alpha * float(np.dot(self._prev_residual, delta_r)) / denom
                    if np.isfinite(alpha_new):
                        alpha = _clip_relaxation(alpha_new, self.relaxation_min, self.relaxation_max)
            self.alpha_old = float(alpha)

        self._prev_residual = r_vec.copy()
        delta = alpha * r_vec
        return CouplingAccelerationStep(
            next_iterate=x_vec + delta,
            delta=delta,
            relaxation=float(alpha),
            used_history=used_history,
            method="aitken",
        )


class IQNILSCouplingAccelerator(CouplingAccelerator):
    def __init__(
        self,
        *,
        iteration_horizon: int,
        timestep_horizon: int,
        alpha: float,
        fallback_alpha: Optional[float] = None,
        regularization: float = 0.0,
        method_name: str = "iqn_ils",
    ) -> None:
        self.iteration_horizon = max(int(iteration_horizon), 1)
        self.timestep_horizon = max(int(timestep_horizon), 1)
        self.alpha = float(alpha)
        self.fallback_alpha = float(alpha if fallback_alpha is None else fallback_alpha)
        self.regularization = max(float(regularization), 0.0)
        self.method_name = str(method_name)
        self._residual_history: deque[np.ndarray] = deque(maxlen=self.iteration_horizon)
        self._prediction_history: deque[np.ndarray] = deque(maxlen=self.iteration_horizon)
        reuse_count = max(self.timestep_horizon - 1, 0)
        self._v_old_mats: deque[np.ndarray] = deque(maxlen=reuse_count)
        self._w_old_mats: deque[np.ndarray] = deque(maxlen=reuse_count)
        self._v_new: Optional[np.ndarray] = None
        self._w_new: Optional[np.ndarray] = None

    def initialize_solution_step(self) -> None:
        self._residual_history.clear()
        self._prediction_history.clear()
        self._v_new = None
        self._w_new = None

    def finalize_solution_step(self, *, accepted: bool = True) -> None:
        if accepted and self.timestep_horizon > 1 and self._v_new is not None and self._w_new is not None:
            self._v_old_mats.appendleft(np.asarray(self._v_new, dtype=float).copy())
            self._w_old_mats.appendleft(np.asarray(self._w_new, dtype=float).copy())
        self.initialize_solution_step()

    def _old_matrices(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._v_old_mats or not self._w_old_mats:
            return None, None
        v_old = np.concatenate([np.asarray(block, dtype=float) for block in self._v_old_mats], axis=1)
        w_old = np.concatenate([np.asarray(block, dtype=float) for block in self._w_old_mats], axis=1)
        return v_old, w_old

    def compute_next_iterate(self, *, x_curr: np.ndarray, residual_curr: np.ndarray) -> CouplingAccelerationStep:
        x_vec = _as_flat_vector(x_curr)
        r_vec = _as_flat_vector(residual_curr)
        if x_vec.size != r_vec.size:
            raise ValueError("IQN-ILS requires x_curr and residual_curr with the same size.")

        self._residual_history.appendleft(r_vec.copy())
        self._prediction_history.appendleft((x_vec + r_vec).copy())

        k = len(self._residual_history) - 1
        v_old, w_old = self._old_matrices()
        has_old = (
            v_old is not None
            and w_old is not None
            and int(v_old.size) > 0
            and int(w_old.size) > 0
        )

        if (not has_old) and k == 0:
            delta = self.fallback_alpha * r_vec
            return CouplingAccelerationStep(
                next_iterate=x_vec + delta,
                delta=delta,
                relaxation=float(self.fallback_alpha),
                used_history=False,
                method=self.method_name,
            )

        if k > 0:
            self._v_new = np.column_stack(
                [self._residual_history[i] - self._residual_history[i + 1] for i in range(k)]
            )
            self._w_new = np.column_stack(
                [self._prediction_history[i] - self._prediction_history[i + 1] for i in range(k)]
            )
        else:
            self._v_new = None
            self._w_new = None

        if has_old:
            v_mat = v_old if self._v_new is None else np.hstack((self._v_new, v_old))
            w_mat = w_old if self._w_new is None else np.hstack((self._w_new, w_old))
        else:
            v_mat = self._v_new
            w_mat = self._w_new

        if v_mat is None or w_mat is None or int(v_mat.size) == 0 or int(w_mat.size) == 0:
            delta = self.fallback_alpha * r_vec
            return CouplingAccelerationStep(
                next_iterate=x_vec + delta,
                delta=delta,
                relaxation=float(self.fallback_alpha),
                used_history=False,
                method=self.method_name,
            )

        delta_r = -self._residual_history[0]
        coeffs = _regularized_lstsq(v_mat, delta_r, self.regularization)
        delta = w_mat @ coeffs - delta_r
        if not np.all(np.isfinite(delta)):
            delta = self.fallback_alpha * r_vec
            used_history = False
        else:
            used_history = True
        return CouplingAccelerationStep(
            next_iterate=x_vec + delta,
            delta=delta,
            relaxation=float(self.fallback_alpha),
            used_history=used_history,
            method=self.method_name,
        )


class MVQNCouplingAccelerator(CouplingAccelerator):
    def __init__(self, *, horizon: int, alpha: float) -> None:
        self.horizon = max(int(horizon), 1)
        self.alpha = float(alpha)
        if self.alpha <= 0.0:
            raise ValueError("MVQN requires a strictly positive fallback relaxation alpha.")
        self._residual_history: deque[np.ndarray] = deque(maxlen=self.horizon)
        self._iterate_history: deque[np.ndarray] = deque(maxlen=self.horizon)
        self._jacobian: Optional[np.ndarray] = None
        self._jacobian_hat: Optional[np.ndarray] = None

    def initialize_solution_step(self) -> None:
        self._residual_history.clear()
        self._iterate_history.clear()
        self._jacobian_hat = None

    def finalize_solution_step(self, *, accepted: bool = True) -> None:
        if accepted and self._jacobian_hat is not None:
            self._jacobian = np.asarray(self._jacobian_hat, dtype=float).copy()
        self.initialize_solution_step()

    def compute_next_iterate(self, *, x_curr: np.ndarray, residual_curr: np.ndarray) -> CouplingAccelerationStep:
        x_vec = _as_flat_vector(x_curr)
        r_vec = _as_flat_vector(residual_curr)
        if x_vec.size != r_vec.size:
            raise ValueError("MVQN requires x_curr and residual_curr with the same size.")

        self._residual_history.appendleft(r_vec.copy())
        self._iterate_history.appendleft(x_vec.copy())
        k = len(self._residual_history) - 1
        row = int(r_vec.size)

        if k == 0:
            if self._jacobian is None:
                delta = self.alpha * r_vec
                return CouplingAccelerationStep(
                    next_iterate=x_vec + delta,
                    delta=delta,
                    relaxation=float(self.alpha),
                    used_history=False,
                    method="mvqn",
                )
            delta = _solve_linear_system(self._jacobian, -r_vec)
            if not np.all(np.isfinite(delta)):
                delta = self.alpha * r_vec
            return CouplingAccelerationStep(
                next_iterate=x_vec + delta,
                delta=delta,
                relaxation=float(self.alpha),
                used_history=True,
                method="mvqn",
            )

        if self._jacobian is None:
            self._jacobian = -np.identity(row, dtype=float) / self.alpha

        v_mat = np.column_stack(
            [self._residual_history[i] - self._residual_history[i + 1] for i in range(k)]
        )
        w_mat = np.column_stack(
            [self._iterate_history[i] - self._iterate_history[i + 1] for i in range(k)]
        )
        rhs = v_mat - self._jacobian @ w_mat
        w_right_inverse = np.linalg.lstsq(w_mat, np.identity(row, dtype=float), rcond=None)[0]
        self._jacobian_hat = self._jacobian + rhs @ w_right_inverse
        delta = _solve_linear_system(self._jacobian_hat, -self._residual_history[0])
        if not np.all(np.isfinite(delta)):
            delta = self.alpha * r_vec
            used_history = False
        else:
            used_history = True
        return CouplingAccelerationStep(
            next_iterate=x_vec + delta,
            delta=delta,
            relaxation=float(self.alpha),
            used_history=used_history,
            method="mvqn",
        )


def create_coupling_accelerator(
    kind: str,
    *,
    relaxation: float,
    relaxation_min: Optional[float] = None,
    relaxation_max: Optional[float] = None,
    init_relaxation_max: Optional[float] = None,
    history: int = 6,
    regularization: float = 0.0,
    timestep_horizon: int = 1,
) -> CouplingAccelerator:
    key = str(kind).strip().lower()
    if key == "constant":
        return ConstantRelaxationCouplingAccelerator(relaxation=float(relaxation))
    if key == "aitken":
        if relaxation_min is None or relaxation_max is None:
            raise ValueError("Aitken acceleration requires relaxation_min and relaxation_max.")
        return AitkenCouplingAccelerator(
            init_relaxation=float(relaxation),
            relaxation_min=float(relaxation_min),
            relaxation_max=float(relaxation_max),
            init_relaxation_max=init_relaxation_max,
        )
    if key == "iqn_ils":
        return IQNILSCouplingAccelerator(
            iteration_horizon=int(history),
            timestep_horizon=int(timestep_horizon),
            alpha=float(relaxation),
            fallback_alpha=float(relaxation),
            regularization=float(regularization),
            method_name="iqn_ils",
        )
    if key == "iqln":
        return IQNILSCouplingAccelerator(
            iteration_horizon=int(history),
            timestep_horizon=int(timestep_horizon),
            alpha=0.0,
            fallback_alpha=float(relaxation),
            regularization=float(regularization),
            method_name="iqln",
        )
    if key == "mvqn":
        return MVQNCouplingAccelerator(horizon=int(history), alpha=float(relaxation))
    raise ValueError(f"Unknown coupling accelerator '{kind}'.")
