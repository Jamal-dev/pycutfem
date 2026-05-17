"""Backend-agnostic nonlinear globalization helpers for reduced solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BranchGlobalizationSpec:
    """State-distance guard used to keep a reduced solve near a predictor."""

    reference_coefficients: np.ndarray
    trial_basis: np.ndarray | None = None
    max_reference_distance: float | None = None
    state_merit_weight: float = 0.0
    require_residual_convergence: bool = False
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        ref = np.asarray(self.reference_coefficients, dtype=float).reshape(-1)
        if not np.all(np.isfinite(ref)):
            raise ValueError("reference_coefficients must contain only finite values.")
        basis = None if self.trial_basis is None else np.asarray(self.trial_basis, dtype=float)
        if basis is not None:
            if basis.ndim != 2 or basis.shape[1] != ref.size:
                raise ValueError("trial_basis must be rank-2 with one column per reference coefficient.")
            if not np.all(np.isfinite(basis)):
                raise ValueError("trial_basis must contain only finite values.")
            basis = np.ascontiguousarray(basis, dtype=np.float64)
        radius = None if self.max_reference_distance is None else float(self.max_reference_distance)
        if radius is not None and (not np.isfinite(radius) or radius < 0.0):
            raise ValueError("max_reference_distance must be finite nonnegative or None.")
        if not np.isfinite(self.state_merit_weight) or float(self.state_merit_weight) < 0.0:
            raise ValueError("state_merit_weight must be finite and nonnegative.")
        object.__setattr__(self, "reference_coefficients", np.ascontiguousarray(ref, dtype=np.float64))
        object.__setattr__(self, "trial_basis", basis)
        object.__setattr__(self, "max_reference_distance", radius)
        object.__setattr__(self, "state_merit_weight", float(self.state_merit_weight))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def decoded_distance(self, coefficients: Any) -> float:
        q = np.asarray(coefficients, dtype=float).reshape(-1)
        if q.size != self.reference_coefficients.size:
            raise ValueError("coefficients size must match reference_coefficients.")
        delta = q - self.reference_coefficients
        if self.trial_basis is not None:
            delta = self.trial_basis @ delta
        return float(np.linalg.norm(delta))

    def within_radius(self, coefficients: Any, *, tolerance: float = 1.0e-14) -> bool:
        if self.max_reference_distance is None:
            return True
        return self.decoded_distance(coefficients) <= float(self.max_reference_distance) + float(tolerance)

    def state_penalty(self, coefficients: Any) -> float:
        if self.state_merit_weight <= 0.0:
            return 0.0
        distance = self.decoded_distance(coefficients)
        scale = (
            float(self.max_reference_distance)
            if self.max_reference_distance is not None and self.max_reference_distance > 0.0
            else max(1.0, float(np.linalg.norm(self.reference_coefficients)))
        )
        normalized = distance / max(scale, 1.0e-300)
        return float(self.state_merit_weight) * normalized * normalized

    def merit(self, residual_norm: float, coefficients: Any) -> float:
        if not np.isfinite(residual_norm) or float(residual_norm) < 0.0:
            raise ValueError("residual_norm must be finite and nonnegative.")
        return float(residual_norm) + self.state_penalty(coefficients)

    def to_native_options(self) -> dict[str, Any]:
        return {
            "reference_coefficients": self.reference_coefficients,
            "max_reference_distance": self.max_reference_distance,
            "state_merit_weight": self.state_merit_weight,
            "require_residual_convergence": bool(self.require_residual_convergence),
        }


@dataclass(frozen=True)
class ContinuationAttempt:
    """One branch-continuation/backtracking attempt."""

    attempt: int
    options: Mapping[str, Any]
    accepted: bool
    residual_norm: float
    branch_distance: float
    message: str = ""


@dataclass(frozen=True)
class ContinuationResult:
    """Result of a branch/backtracking solve policy."""

    result: Any
    accepted: bool
    attempts: tuple[ContinuationAttempt, ...]
    options: Mapping[str, Any]


def clip_step_to_trust_region(step: Any, max_step_norm: float | None) -> tuple[np.ndarray, bool]:
    """Return a reduced step clipped to a Euclidean trust-region radius."""

    raw = np.asarray(step, dtype=float).reshape(-1)
    if not np.all(np.isfinite(raw)):
        raise ValueError("step must contain only finite values.")
    if max_step_norm is None:
        return np.ascontiguousarray(raw, dtype=np.float64), False
    radius = float(max_step_norm)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("max_step_norm must be finite nonnegative or None.")
    norm = float(np.linalg.norm(raw))
    if radius == 0.0:
        return np.zeros_like(raw, dtype=np.float64), bool(norm > 0.0)
    if norm <= radius:
        return np.ascontiguousarray(raw, dtype=np.float64), False
    return np.ascontiguousarray((radius / norm) * raw, dtype=np.float64), True


def step_alpha_to_branch_radius(
    coefficients: Any,
    step: Any,
    spec: BranchGlobalizationSpec,
    *,
    max_bisections: int = 60,
) -> float:
    """Largest bisection alpha in ``[0, 1]`` that stays inside the branch radius."""

    q = np.asarray(coefficients, dtype=float).reshape(-1)
    p = np.asarray(step, dtype=float).reshape(-1)
    if q.size != p.size:
        raise ValueError("coefficients and step must have the same size.")
    if spec.within_radius(q + p):
        return 1.0
    lo = 0.0
    hi = 1.0
    for _ in range(max(1, int(max_bisections))):
        mid = 0.5 * (lo + hi)
        if spec.within_radius(q + mid * p):
            lo = mid
        else:
            hi = mid
    return float(lo)


def solve_with_branch_backtracking(
    solve_once: Callable[..., Any],
    *,
    base_options: Mapping[str, Any] | None = None,
    branch_radii: Sequence[float | None] = (None,),
    trust_radii: Sequence[float | None] = (None,),
    accept: Callable[[Any], bool] | None = None,
) -> ContinuationResult:
    """Retry a reduced solve over branch/trust-radius schedules.

    ``solve_once`` is called with ``base_options`` plus
    ``max_reference_distance`` and ``max_step_norm``.  The default acceptance
    accepts objects with ``converged=True``.  This is intentionally algebraic so
    transient drivers can refresh residual kernels, previous states, or time-step
    data around it without baking one PDE into the MOR layer.
    """

    options0 = dict(base_options or {})
    attempts: list[ContinuationAttempt] = []
    last_result: Any = None
    last_options: dict[str, Any] = {}

    def _accepted(result: Any) -> bool:
        if accept is not None:
            return bool(accept(result))
        return bool(getattr(result, "converged", False))

    for radius in tuple(branch_radii):
        for trust in tuple(trust_radii):
            opts = dict(options0)
            if radius is not None:
                opts["max_reference_distance"] = float(radius)
            if trust is not None:
                opts["max_step_norm"] = float(trust)
            last_options = opts
            try:
                result = solve_once(**opts)
                last_result = result
                ok = _accepted(result)
                timings = dict(getattr(result, "timing_counters", {}) or {})
                attempts.append(
                    ContinuationAttempt(
                        attempt=len(attempts) + 1,
                        options=dict(opts),
                        accepted=ok,
                        residual_norm=float(getattr(result, "residual_norm", np.inf)),
                        branch_distance=float(timings.get("final_branch_distance", np.nan)),
                    )
                )
                if ok:
                    return ContinuationResult(result=result, accepted=True, attempts=tuple(attempts), options=dict(opts))
            except Exception as exc:
                attempts.append(
                    ContinuationAttempt(
                        attempt=len(attempts) + 1,
                        options=dict(opts),
                        accepted=False,
                        residual_norm=float("inf"),
                        branch_distance=float("nan"),
                        message=str(exc),
                    )
                )
                last_result = exc
    return ContinuationResult(result=last_result, accepted=False, attempts=tuple(attempts), options=dict(last_options))


__all__ = [
    "BranchGlobalizationSpec",
    "ContinuationAttempt",
    "ContinuationResult",
    "clip_step_to_trust_region",
    "solve_with_branch_backtracking",
    "step_alpha_to_branch_radius",
]
