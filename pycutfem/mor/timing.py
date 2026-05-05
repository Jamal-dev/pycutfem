from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter

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
