from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from .metrics import speedup


@dataclass
class Timer:
    elapsed: float | None = None
    _start: float | None = None

    def __enter__(self) -> "Timer":
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._start is not None:
            self.elapsed = perf_counter() - self._start


@dataclass
class TimingAccumulator:
    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, elapsed: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + float(elapsed)
        self.counts[name] = self.counts.get(name, 0) + 1

    @contextmanager
    def track(self, name: str):
        with Timer() as timer:
            yield timer
        self.add(name, timer.elapsed or 0.0)

    def average(self, name: str) -> float:
        if name not in self.totals:
            raise KeyError(name)
        return self.totals[name] / self.counts[name]

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "total": total,
                "count": float(self.counts[name]),
                "average": total / self.counts[name],
            }
            for name, total in self.totals.items()
        }


def build_speedup_report(
    *,
    fom_solid_time: float,
    rom_solid_time: float,
    fom_total_time: float | None = None,
    rom_total_time: float | None = None,
) -> dict[str, float]:
    report = {"solid_speedup": speedup(fom_solid_time, rom_solid_time)}
    if fom_total_time is not None and rom_total_time is not None:
        report["overall_speedup"] = speedup(fom_total_time, rom_total_time)
    return report


@dataclass(frozen=True)
class StageBreakEvenCertificate:
    """Timing certificate for deciding whether a reduced stage pays off."""

    passed: bool
    exact_mean_time: float
    reduced_mean_time: float
    speedup: float
    required_speedup: float
    exact_count: int
    reduced_count: int
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "exact_mean_time", float(self.exact_mean_time))
        object.__setattr__(self, "reduced_mean_time", float(self.reduced_mean_time))
        object.__setattr__(self, "speedup", float(self.speedup))
        object.__setattr__(self, "required_speedup", float(self.required_speedup))
        object.__setattr__(self, "exact_count", int(self.exact_count))
        object.__setattr__(self, "reduced_count", int(self.reduced_count))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "exact_mean_time": float(self.exact_mean_time),
            "reduced_mean_time": float(self.reduced_mean_time),
            "speedup": float(self.speedup),
            "required_speedup": float(self.required_speedup),
            "exact_count": int(self.exact_count),
            "reduced_count": int(self.reduced_count),
            "metadata": dict(self.metadata or {}),
        }


def build_stage_break_even_certificate(
    *,
    exact_stage_times: Any,
    reduced_stage_times: Any,
    required_speedup: float = 1.0,
    min_samples: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> StageBreakEvenCertificate:
    """Build a cost gate for accepting a reduced online stage."""

    exact = np.asarray(exact_stage_times, dtype=float).reshape(-1)
    reduced = np.asarray(reduced_stage_times, dtype=float).reshape(-1)
    exact = exact[np.isfinite(exact) & (exact >= 0.0)]
    reduced = reduced[np.isfinite(reduced) & (reduced >= 0.0)]
    if exact.size == 0 or reduced.size == 0:
        raise ValueError("exact_stage_times and reduced_stage_times must contain nonnegative finite samples.")
    exact_mean = float(np.mean(exact))
    reduced_mean = float(np.mean(reduced))
    ratio = speedup(exact_mean, reduced_mean)
    required = float(required_speedup)
    if not np.isfinite(required) or required <= 0.0:
        raise ValueError("required_speedup must be finite and positive.")
    enough_samples = int(exact.size) >= int(min_samples) and int(reduced.size) >= int(min_samples)
    return StageBreakEvenCertificate(
        passed=bool(enough_samples and ratio >= required),
        exact_mean_time=exact_mean,
        reduced_mean_time=reduced_mean,
        speedup=ratio,
        required_speedup=required,
        exact_count=int(exact.size),
        reduced_count=int(reduced.size),
        metadata={
            "min_samples": int(min_samples),
            "enough_samples": bool(enough_samples),
            **dict(metadata or {}),
        },
    )
