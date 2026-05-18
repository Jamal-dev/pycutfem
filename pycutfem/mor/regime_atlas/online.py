"""Online novelty and local-regime bank selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .banking import LocalReducedModelBankEntry, load_regime_bank_manifest, select_local_reduced_model_bank
from .data import RegimeAtlas


@dataclass(frozen=True)
class RegimeNoveltyDecision:
    """Online decision for local regime selection or fallback."""

    selected: bool
    reason: str
    region_id: str = ""
    distance: float = float("nan")
    uncertainty: float = float("nan")
    entry: LocalReducedModelBankEntry | None = None
    metadata: Mapping[str, Any] | None = None

    @property
    def model_id(self) -> str:
        if self.entry is not None:
            return self.entry.model_id
        return self.region_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": bool(self.selected),
            "reason": str(self.reason),
            "region_id": str(self.region_id),
            "model_id": self.model_id,
            "distance": float(self.distance),
            "uncertainty": float(self.uncertainty),
            "entry": None if self.entry is None else self.entry.to_dict(),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class RegimeOnlineSelector:
    """Select a certified region or bank entry for an online feature."""

    atlas: RegimeAtlas | None = None
    entries: tuple[LocalReducedModelBankEntry, ...] = ()
    fallback_policy: Mapping[str, Any] | None = None

    @classmethod
    def from_manifest(cls, path: str | Path) -> "RegimeOnlineSelector":
        manifest = load_regime_bank_manifest(path)
        return cls(entries=manifest.entries, fallback_policy=manifest.fallback_policy)

    def select(
        self,
        *,
        feature: Sequence[float] | np.ndarray,
        step: int = 1,
    ) -> RegimeNoveltyDecision:
        values = np.asarray(feature, dtype=float).reshape(-1)
        if self.entries:
            selection = select_local_reduced_model_bank(self.entries, step=int(step), feature=values)
            if selection.selected and selection.entry is not None:
                return RegimeNoveltyDecision(
                    selected=True,
                    reason=selection.reason,
                    region_id=selection.entry.model_id,
                    distance=float(selection.distance),
                    entry=selection.entry,
                )
            return RegimeNoveltyDecision(
                selected=False,
                reason=selection.reason,
                distance=float(selection.distance),
                metadata={"fallback_policy": dict(self.fallback_policy or {})},
            )
        if self.atlas is None:
            return RegimeNoveltyDecision(selected=False, reason="no_atlas_or_bank_entries")
        result = self.atlas.region_for_feature(values)
        if result is None:
            return RegimeNoveltyDecision(
                selected=False,
                reason="outside_certified_region",
                metadata={"fallback_policy": dict(self.fallback_policy or {})},
            )
        region, distance = result
        uncertainty = 0.0
        if self.atlas.n_regions > 1:
            distances = np.asarray([item.distance(values) for item in self.atlas.regions], dtype=float)
            distances_sorted = np.sort(distances[np.isfinite(distances)])
            if distances_sorted.size >= 2:
                uncertainty = float(distances_sorted[0] / max(distances_sorted[1], 1.0e-300))
        return RegimeNoveltyDecision(
            selected=True,
            reason="inside_certified_region",
            region_id=region.region_id,
            distance=float(distance),
            uncertainty=float(uncertainty),
        )


__all__ = [
    "RegimeNoveltyDecision",
    "RegimeOnlineSelector",
]
