from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DtControllerParams:
    """
    Heuristic dt controller based on Newton convergence behaviour.

    - On Newton failure: reject and retry with `decrease_factor_fail`.
    - On Newton success:
        - if iters > iters_decrease_threshold for `slow_steps_before_decrease`
          consecutive steps: decrease dt for the *next* step by
          `decrease_factor_slow`.
        - if iters <= iters_increase_threshold for `easy_steps_before_increase`
          consecutive steps: increase dt for the next step by `increase_factor`,
          capped by `dt_max` (defaults to the initial dt).
    """

    dt_min: float = 0.0
    dt_max: Optional[float] = None

    iters_increase_threshold: int = 8
    iters_decrease_threshold: int = 20

    easy_steps_before_increase: int = 2
    slow_steps_before_decrease: int = 1

    increase_factor: float = 1.1
    decrease_factor_slow: float = 0.9
    decrease_factor_fail: float = 0.5

    reject_on_slow: bool = False


@dataclass(frozen=True)
class DtDecision:
    dt: float
    retry_step: bool
    reason: str


class AdaptiveDtController:
    def __init__(self, *, dt0: float, params: DtControllerParams) -> None:
        if dt0 <= 0.0:
            raise ValueError(f"dt0 must be positive, got {dt0!r}")
        self._params = params
        self._dt_max = float(params.dt_max) if params.dt_max is not None else float(dt0)
        self._easy_steps = 0
        self._slow_steps = 0

    @property
    def dt_max(self) -> float:
        return self._dt_max

    def on_success(self, *, dt: float, n_iters: int) -> DtDecision:
        p = self._params
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        if n_iters < 0:
            raise ValueError(f"n_iters must be non-negative, got {n_iters!r}")

        if n_iters <= int(p.iters_increase_threshold):
            self._easy_steps += 1
            self._slow_steps = 0
        elif n_iters > int(p.iters_decrease_threshold):
            self._slow_steps += 1
            self._easy_steps = 0
        else:
            self._easy_steps = 0
            self._slow_steps = 0

        # Slow convergence: decrease dt (for next step) after N consecutive slow steps.
        if (
            self._slow_steps >= int(p.slow_steps_before_decrease)
            and p.decrease_factor_slow < 1.0
            and dt > max(float(p.dt_min), 0.0)
        ):
            new_dt = max(float(p.dt_min), dt * float(p.decrease_factor_slow))
            new_dt = min(new_dt, self._dt_max)
            self._slow_steps = 0
            if new_dt < dt:
                return DtDecision(
                    dt=new_dt, retry_step=bool(p.reject_on_slow), reason="slow_newton"
                )

        # Easy steps: cautiously increase dt (for next step) after N consecutive easy steps.
        if (
            self._easy_steps >= int(p.easy_steps_before_increase)
            and p.increase_factor > 1.0
            and dt < self._dt_max
        ):
            new_dt = min(self._dt_max, dt * float(p.increase_factor))
            self._easy_steps = 0
            if new_dt > dt:
                return DtDecision(dt=new_dt, retry_step=False, reason="fast_newton")

        return DtDecision(dt=dt, retry_step=False, reason="keep")

    def on_failure(self, *, dt: float, reason: str) -> DtDecision:
        p = self._params
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        self._easy_steps = 0
        self._slow_steps = 0

        factor = float(p.decrease_factor_fail)
        if factor <= 0.0 or factor >= 1.0:
            raise ValueError(f"decrease_factor_fail must be in (0, 1), got {factor!r}")

        new_dt = max(float(p.dt_min), dt * factor)
        return DtDecision(dt=new_dt, retry_step=True, reason=f"newton_failed:{reason}")

