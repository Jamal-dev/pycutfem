"""Error-driven atlas selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .data import RegimeAtlas
from .validation import RegimeValidationSummary


@dataclass(frozen=True)
class RegimeAtlasCandidate:
    atlas: RegimeAtlas
    validation: RegimeValidationSummary
    complexity: float = 0.0
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RegimeAtlasSelection:
    selected: RegimeAtlasCandidate
    candidates: tuple[RegimeAtlasCandidate, ...]
    scores: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        selected_index = next(i for i, candidate in enumerate(self.candidates) if candidate is self.selected)
        return {
            "selected_index": int(selected_index),
            "selected_score": float(self.scores[selected_index]),
            "scores": [float(value) for value in self.scores],
            "selected": {
                "atlas": self.selected.atlas.to_dict(),
                "validation": self.selected.validation.to_dict(),
                "complexity": float(self.selected.complexity),
                "metadata": dict(self.selected.metadata or {}),
            },
        }


@dataclass(frozen=True)
class RegimeAtlasSelector:
    """Penalized selector for candidate atlases."""

    max_validation_error: float | None = None
    complexity_weight: float = 0.0
    region_penalty: float = 0.0
    boundary_weight: float = 0.0
    fallback_weight: float = 0.0

    def score(self, candidate: RegimeAtlasCandidate) -> float:
        validation = candidate.validation
        return float(
            validation.max_error
            + float(self.complexity_weight) * float(candidate.complexity)
            + float(self.region_penalty) * float(candidate.atlas.n_regions)
            + float(self.boundary_weight) * float(validation.boundary_error)
            + float(self.fallback_weight) * float(validation.fallback_rate)
        )

    def select(self, candidates: Sequence[RegimeAtlasCandidate]) -> RegimeAtlasSelection:
        items = tuple(candidates)
        if not items:
            raise ValueError("at least one candidate is required.")
        scores = tuple(float(self.score(candidate)) for candidate in items)
        eligible = list(range(len(items)))
        if self.max_validation_error is not None:
            threshold = float(self.max_validation_error)
            passing = [i for i, item in enumerate(items) if item.validation.max_error <= threshold]
            if passing:
                eligible = passing
        selected_index = min(eligible, key=lambda i: (scores[i], items[i].atlas.n_regions))
        return RegimeAtlasSelection(selected=items[selected_index], candidates=items, scores=scores)


__all__ = [
    "RegimeAtlasCandidate",
    "RegimeAtlasSelection",
    "RegimeAtlasSelector",
]
