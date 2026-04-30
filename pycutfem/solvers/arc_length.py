"""Ramm arc-length continuation utilities.

The implementation mirrors the strategy used by Kratos Poromechanics while
remaining independent of any element type. Callers provide callbacks for the
tangent matrix, residual vector, and reference load vector, so the same
continuation step can drive UFL, runtime-operator, or assembled external
systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la


Array = np.ndarray
LinearSolve = Callable[[Array, Array], Array]
TangentCallback = Callable[[Array, float], Array]
ResidualCallback = Callable[[Array, float], Array]
ConvergenceCallback = Callable[["RammArcLengthIteration"], bool]


@dataclass(frozen=True)
class RammArcLengthParameters:
    """Numerical controls for one Ramm arc-length continuation step."""

    desired_iterations: int = 4
    max_iterations: int = 30
    max_radius_factor: float = 10.0
    min_radius_factor: float = 1.0
    residual_tolerance: float = 1.0e-8
    residual_relative_tolerance: float = 0.0
    update_tolerance: float = 0.0
    update_relative_tolerance: float = 0.0
    divergence_factor: float = 1.0e3
    equilibrium_norm_floor: float = 1.0e-10

    def __post_init__(self) -> None:
        if int(self.desired_iterations) <= 0:
            raise ValueError("desired_iterations must be positive.")
        if int(self.max_iterations) <= 0:
            raise ValueError("max_iterations must be positive.")
        if float(self.max_radius_factor) <= 0.0:
            raise ValueError("max_radius_factor must be positive.")
        if float(self.min_radius_factor) <= 0.0:
            raise ValueError("min_radius_factor must be positive.")
        if float(self.min_radius_factor) > float(self.max_radius_factor):
            raise ValueError("min_radius_factor cannot exceed max_radius_factor.")
        if float(self.residual_tolerance) < 0.0:
            raise ValueError("residual_tolerance cannot be negative.")
        if float(self.residual_relative_tolerance) < 0.0:
            raise ValueError("residual_relative_tolerance cannot be negative.")
        if float(self.update_tolerance) < 0.0:
            raise ValueError("update_tolerance cannot be negative.")
        if float(self.update_relative_tolerance) < 0.0:
            raise ValueError("update_relative_tolerance cannot be negative.")
        if float(self.divergence_factor) <= 0.0:
            raise ValueError("divergence_factor must be positive.")


@dataclass
class RammArcLengthState:
    """Persistent state carried between Ramm arc-length load steps."""

    radius_0: float
    radius: float
    lambda_value: float = 0.0
    lambda_old: float = 1.0
    norm_x_equilibrium: float = 0.0

    def __post_init__(self) -> None:
        if float(self.radius_0) <= 0.0:
            raise ValueError("radius_0 must be positive.")
        if float(self.radius) <= 0.0:
            raise ValueError("radius must be positive.")


@dataclass(frozen=True)
class RammArcLengthIteration:
    """Diagnostic record for a predictor or corrector iteration."""

    iteration: int
    lambda_value: float
    delta_lambda: float
    delta_lambda_step: float
    residual_norm: float
    residual_ratio: float
    update_norm: float
    update_ratio: float = 0.0


@dataclass(frozen=True)
class RammArcLengthStepResult:
    """Result returned after one arc-length step."""

    x: Array
    state: RammArcLengthState
    converged: bool
    iterations: int
    dx_step: Array
    delta_lambda_step: float
    history: tuple[RammArcLengthIteration, ...] = field(default_factory=tuple)


def initialize_ramm_arc_length_state(
    tangent: Array,
    reference_load: Array,
    *,
    linear_solver: LinearSolve | None = None,
) -> RammArcLengthState:
    """Create the initial Ramm state from ``K dx = f_ref``.

    Kratos sets the initial arc-length radius to ``||K^{-1} f_ref||_2``. This
    helper exposes that rule directly so callers can initialize the continuation
    state with any assembled tangent and load vector.
    """

    dxf = _solve(tangent, reference_load, linear_solver)
    radius_0 = float(np.linalg.norm(dxf, ord=2))
    if radius_0 <= 0.0:
        raise ValueError("Initial arc-length radius is zero; reference_load is not active in the tangent space.")
    return RammArcLengthState(radius_0=radius_0, radius=radius_0)


def ramm_arc_length_step(
    x: Array,
    state: RammArcLengthState,
    *,
    tangent_callback: TangentCallback,
    residual_callback: ResidualCallback,
    reference_load: Array,
    params: RammArcLengthParameters | None = None,
    linear_solver: LinearSolve | None = None,
    convergence_callback: ConvergenceCallback | None = None,
) -> RammArcLengthStepResult:
    """Advance one generic Ramm arc-length step.

    The callbacks are evaluated at ``(x, lambda_value)`` and must return the
    tangent matrix and residual RHS in the same sign convention as the Newton
    solve ``K dx = residual``. The load factor ``lambda_value`` is explicit, so
    callers can scale external loads without mutating global state.
    """

    p = params or RammArcLengthParameters()
    x0 = np.asarray(x, dtype=float).reshape(-1)
    f_ref = np.asarray(reference_load, dtype=float).reshape(-1)
    if x0.shape != f_ref.shape:
        raise ValueError(f"x and reference_load must have the same shape, got {x0.shape} and {f_ref.shape}.")

    K = tangent_callback(x0.copy(), float(state.lambda_value))
    dxf = _solve(K, f_ref, linear_solver)
    dxf_norm = float(np.linalg.norm(dxf, ord=2))
    if dxf_norm <= 0.0:
        raise RuntimeError("Arc-length predictor has zero reference-load direction.")

    delta_lambda = float(state.radius) / dxf_norm
    delta_lambda_step = delta_lambda
    dx_pred = delta_lambda * dxf
    dx_step = dx_pred.copy()
    x_trial = x0 + dx_pred
    lambda_trial = float(state.lambda_value) + delta_lambda

    initial_residual = np.asarray(residual_callback(x0.copy(), float(state.lambda_value)), dtype=float).reshape(-1)
    initial_norm = max(float(np.linalg.norm(initial_residual, ord=2)), np.finfo(float).tiny)
    history: list[RammArcLengthIteration] = []

    residual = np.asarray(residual_callback(x_trial.copy(), lambda_trial), dtype=float).reshape(-1)
    residual_norm = float(np.linalg.norm(residual, ord=2))
    residual_ratio = residual_norm / initial_norm
    update_denominator = _update_denominator(x_trial, float(state.norm_x_equilibrium), p)
    iteration = RammArcLengthIteration(
        iteration=1,
        lambda_value=lambda_trial,
        delta_lambda=delta_lambda,
        delta_lambda_step=delta_lambda_step,
        residual_norm=residual_norm,
        residual_ratio=residual_ratio,
        update_norm=float(np.linalg.norm(dx_pred, ord=2)),
        update_ratio=float(np.linalg.norm(dx_pred, ord=2)) / update_denominator,
    )
    history.append(iteration)
    converged = _is_converged(iteration, p, convergence_callback)

    iteration_number = 1
    while not converged and iteration_number < int(p.max_iterations):
        iteration_number += 1
        K = tangent_callback(x_trial.copy(), lambda_trial)
        dxf = _solve(K, f_ref, linear_solver)
        dxb = _solve(K, residual, linear_solver)
        denom = float(np.dot(dx_pred, dxf))
        if abs(denom) <= np.finfo(float).tiny:
            raise RuntimeError("Arc-length correction denominator is zero.")
        delta_lambda = -float(np.dot(dx_pred, dxb)) / denom
        dx = dxb + delta_lambda * dxf

        if float(state.norm_x_equilibrium) > float(p.equilibrium_norm_floor):
            if float(np.linalg.norm(dx, ord=2)) / float(state.norm_x_equilibrium) > float(p.divergence_factor):
                break
            base_lambda = max(abs(float(state.lambda_value)), np.finfo(float).tiny)
            if abs(delta_lambda) / base_lambda > float(p.divergence_factor):
                break

        delta_lambda_step += delta_lambda
        lambda_trial += delta_lambda
        dx_step += dx
        x_trial = x_trial + dx
        residual = np.asarray(residual_callback(x_trial.copy(), lambda_trial), dtype=float).reshape(-1)
        residual_norm = float(np.linalg.norm(residual, ord=2))
        residual_ratio = residual_norm / initial_norm
        update_norm = float(np.linalg.norm(dx, ord=2))
        update_denominator = _update_denominator(x_trial, float(state.norm_x_equilibrium), p)
        iteration = RammArcLengthIteration(
            iteration=iteration_number,
            lambda_value=lambda_trial,
            delta_lambda=delta_lambda,
            delta_lambda_step=delta_lambda_step,
            residual_norm=residual_norm,
            residual_ratio=residual_ratio,
            update_norm=update_norm,
            update_ratio=update_norm / update_denominator,
        )
        history.append(iteration)
        converged = _is_converged(iteration, p, convergence_callback)

    radius = float(state.radius) * np.sqrt(float(p.desired_iterations) / float(iteration_number))
    if converged:
        radius = min(radius, float(p.max_radius_factor) * float(state.radius_0))
        radius = max(radius, float(p.min_radius_factor) * float(state.radius_0))
        next_state = RammArcLengthState(
            radius_0=float(state.radius_0),
            radius=radius,
            lambda_value=lambda_trial,
            lambda_old=lambda_trial,
            norm_x_equilibrium=float(np.linalg.norm(x_trial, ord=2)),
        )
        x_out = x_trial
    else:
        next_state = RammArcLengthState(
            radius_0=float(state.radius_0),
            radius=radius,
            lambda_value=float(state.lambda_value),
            lambda_old=float(state.lambda_old),
            norm_x_equilibrium=float(state.norm_x_equilibrium),
        )
        x_out = x0

    return RammArcLengthStepResult(
        x=np.asarray(x_out, dtype=float),
        state=next_state,
        converged=bool(converged),
        iterations=int(iteration_number),
        dx_step=np.asarray(dx_step, dtype=float),
        delta_lambda_step=float(delta_lambda_step),
        history=tuple(history),
    )


def _solve(matrix: Array, rhs: Array, linear_solver: LinearSolve | None) -> Array:
    A = matrix if sp.issparse(matrix) else np.asarray(matrix, dtype=float)
    b = np.asarray(rhs, dtype=float).reshape(-1)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"matrix must be square, got shape {A.shape}.")
    if A.shape[0] != b.shape[0]:
        raise ValueError(f"matrix/rhs size mismatch: {A.shape} vs {b.shape}.")
    if linear_solver is not None:
        out = linear_solver(A, b)
    elif sp.issparse(A):
        out = sp_la.spsolve(A.tocsc(), b)
    else:
        out = np.linalg.solve(A, b)
    return np.asarray(out, dtype=float).reshape(-1)


def _is_converged(
    iteration: RammArcLengthIteration,
    params: RammArcLengthParameters,
    convergence_callback: ConvergenceCallback | None,
) -> bool:
    if convergence_callback is not None:
        return bool(convergence_callback(iteration))
    residual_tolerances_enabled = (
        float(params.residual_tolerance) > 0.0 or float(params.residual_relative_tolerance) > 0.0
    )
    residual_ok = not residual_tolerances_enabled
    if float(params.residual_tolerance) > 0.0 and iteration.residual_norm <= float(params.residual_tolerance):
        residual_ok = True
    if (
        float(params.residual_relative_tolerance) > 0.0
        and iteration.residual_ratio <= float(params.residual_relative_tolerance)
    ):
        residual_ok = True

    update_tolerances_enabled = float(params.update_tolerance) > 0.0 or float(params.update_relative_tolerance) > 0.0
    if not update_tolerances_enabled:
        return residual_ok

    update_ok = False
    if float(params.update_tolerance) > 0.0 and iteration.update_norm <= float(params.update_tolerance):
        update_ok = True
    if float(params.update_relative_tolerance) > 0.0 and iteration.update_ratio <= float(params.update_relative_tolerance):
        update_ok = True
    return residual_ok and update_ok


def _update_denominator(x_trial: Array, equilibrium_norm: float, params: RammArcLengthParameters) -> float:
    norm_trial = float(np.linalg.norm(np.asarray(x_trial, dtype=float).reshape(-1), ord=2))
    return max(norm_trial, float(equilibrium_norm), float(params.equilibrium_norm_floor), np.finfo(float).tiny)
