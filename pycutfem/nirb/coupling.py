from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CouplingIterationRecord:
    time: float
    subiteration: int
    converged: bool
    solid_time: float
    fluid_time: float
    reduced_displacement: np.ndarray


@dataclass
class CouplingTrace:
    records: list[CouplingIterationRecord] = field(default_factory=list)

    def append(self, record: CouplingIterationRecord) -> None:
        self.records.append(record)

    def accumulated_iterations(self) -> int:
        return len(self.records)

    def reduced_history(self) -> np.ndarray:
        if not self.records:
            return np.zeros((0, 0), dtype=float)
        return np.column_stack([record.reduced_displacement for record in self.records])
