"""Local reduced-model bank selection utilities.

Nonlinear HROMs often need several local trial/sample spaces instead of one
global ``V`` and one global sampling matrix.  This module keeps the selection
logic algebraic and problem-independent: a project-specific driver attaches the
loaded model objects to the selected entries.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LocalReducedModelBankEntry:
    """One local reduced model in a deployable bank manifest."""

    model_id: str
    path: Path
    step_start: int = 1
    step_end: int | None = None
    priority: int = 0
    feature_center: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    max_feature_distance: float | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        start = max(1, int(self.step_start))
        end = None if self.step_end is None else int(self.step_end)
        if end is not None and end < start:
            raise ValueError(f"Bank entry {self.model_id!r} has step_end < step_start.")
        object.__setattr__(self, "model_id", str(self.model_id))
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "step_start", start)
        object.__setattr__(self, "step_end", end)
        object.__setattr__(self, "priority", int(self.priority))
        center = None if self.feature_center is None else np.asarray(self.feature_center, dtype=float).reshape(-1)
        scale = None if self.feature_scale is None else np.asarray(self.feature_scale, dtype=float).reshape(-1)
        if center is not None:
            if not np.all(np.isfinite(center)):
                raise ValueError(f"Bank entry {self.model_id!r} feature_center must be finite.")
            if scale is None:
                scale = np.ones_like(center)
            if scale.size != center.size:
                raise ValueError(f"Bank entry {self.model_id!r} feature_scale size must match feature_center.")
            if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
                raise ValueError(f"Bank entry {self.model_id!r} feature_scale must be finite and positive.")
        max_distance = None
        if self.max_feature_distance is not None:
            max_distance = float(self.max_feature_distance)
            if not np.isfinite(max_distance) or max_distance < 0.0:
                raise ValueError(
                    f"Bank entry {self.model_id!r} max_feature_distance must be finite and nonnegative."
                )
        object.__setattr__(self, "feature_center", center)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "max_feature_distance", max_distance)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def active_for_step(self, step: int) -> bool:
        value = int(step)
        if value < int(self.step_start):
            return False
        return self.step_end is None or value <= int(self.step_end)

    def feature_distance(self, feature: Sequence[float] | np.ndarray | None) -> float:
        if feature is None or self.feature_center is None:
            return 0.0
        values = np.asarray(feature, dtype=float).reshape(-1)
        center = np.asarray(self.feature_center, dtype=float).reshape(-1)
        scale = np.asarray(self.feature_scale, dtype=float).reshape(-1)
        if values.size != center.size:
            return float("inf")
        return float(np.linalg.norm((values - center) / np.maximum(scale, 1.0e-300)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "path": str(self.path),
            "step_start": int(self.step_start),
            "step_end": None if self.step_end is None else int(self.step_end),
            "priority": int(self.priority),
            "feature_center": None
            if self.feature_center is None
            else np.asarray(self.feature_center, dtype=float).tolist(),
            "feature_scale": None
            if self.feature_scale is None
            else np.asarray(self.feature_scale, dtype=float).tolist(),
            "max_feature_distance": (
                None if self.max_feature_distance is None else float(self.max_feature_distance)
            ),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class LocalReducedModelSelection:
    """Selected local reduced-model bank entry."""

    entry: LocalReducedModelBankEntry | None
    reason: str
    distance: float = float("nan")

    @property
    def selected(self) -> bool:
        return self.entry is not None

    @property
    def model_id(self) -> str:
        return "" if self.entry is None else str(self.entry.model_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": bool(self.selected),
            "model_id": self.model_id,
            "reason": str(self.reason),
            "distance": float(self.distance),
            "entry": None if self.entry is None else self.entry.to_dict(),
        }


def _entry_from_mapping(item: Mapping[str, Any], *, base_dir: Path) -> LocalReducedModelBankEntry:
    raw_path = Path(str(item.get("path", "")))
    if not raw_path.is_absolute():
        raw_path = base_dir / raw_path
    model_id = str(item.get("id", raw_path.stem))
    step_end_raw = item.get("step_end", item.get("end_step", None))
    step_end = None if step_end_raw in {None, "", "none", "None"} else int(step_end_raw)
    metadata = {k: v for k, v in dict(item).items() if k not in {
        "id",
        "path",
        "step_start",
        "start_step",
        "step_end",
        "end_step",
        "priority",
        "feature_center",
        "feature_scale",
        "max_feature_distance",
        "metadata",
    }}
    nested_metadata = item.get("metadata", None)
    if isinstance(nested_metadata, Mapping):
        metadata.update(dict(nested_metadata))
    return LocalReducedModelBankEntry(
        model_id=model_id,
        path=raw_path,
        step_start=int(item.get("step_start", item.get("start_step", 1))),
        step_end=step_end,
        priority=int(item.get("priority", 0)),
        feature_center=item.get("feature_center"),
        feature_scale=item.get("feature_scale"),
        max_feature_distance=item.get("max_feature_distance"),
        metadata=metadata,
    )


def load_local_reduced_model_bank_manifest(path: str | Path) -> tuple[LocalReducedModelBankEntry, ...]:
    """Load a JSON manifest describing local reduced-model banks.

    Expected format::

        {
          "schema_version": 1,
          "banks": [
            {"id": "early", "path": "early.npz", "step_start": 1, "step_end": 120},
            {"id": "late", "path": "late.npz", "step_start": 121}
          ]
        }
    """

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        raw_banks = payload
    else:
        version = int(payload.get("schema_version", 1))
        if version != 1:
            raise ValueError(f"Unsupported local reduced-model bank schema_version={version}.")
        raw_banks = payload.get("banks", [])
    if not isinstance(raw_banks, list) or not raw_banks:
        raise ValueError("Local reduced-model bank manifest must contain a nonempty 'banks' list.")
    entries = tuple(_entry_from_mapping(item, base_dir=manifest_path.parent) for item in raw_banks)
    ids = [entry.model_id for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("Local reduced-model bank ids must be unique.")
    return entries


def select_local_reduced_model_bank(
    entries: Sequence[LocalReducedModelBankEntry],
    *,
    step: int,
    feature: Sequence[float] | np.ndarray | None = None,
) -> LocalReducedModelSelection:
    """Select the active bank entry by step interval, priority, and feature distance."""

    active = [entry for entry in entries if entry.active_for_step(int(step))]
    if not active:
        return LocalReducedModelSelection(None, reason="no_active_step_interval")
    scored = []
    for entry in active:
        distance = float(entry.feature_distance(feature))
        if (
            entry.max_feature_distance is not None
            and (not np.isfinite(distance) or distance > float(entry.max_feature_distance))
        ):
            continue
        scored.append(
            (
                -int(entry.priority),
                distance,
                int(entry.step_start),
                entry.model_id,
                entry,
            )
        )
    if not scored:
        return LocalReducedModelSelection(None, reason="no_active_feature_radius")
    scored.sort(key=lambda item: (item[0], item[1], -item[2], item[3]))
    selected = scored[0][-1]
    distance = float(scored[0][1])
    reason = "priority_feature" if selected.feature_center is not None and feature is not None else "step_priority"
    return LocalReducedModelSelection(selected, reason=reason, distance=distance)


__all__ = [
    "LocalReducedModelBankEntry",
    "LocalReducedModelSelection",
    "load_local_reduced_model_bank_manifest",
    "select_local_reduced_model_bank",
]
