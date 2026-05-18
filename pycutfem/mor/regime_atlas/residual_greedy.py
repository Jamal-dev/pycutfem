"""Residual-error greedy splitting for nonlinear-regime atlases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .data import RegimeAtlas, RegimeDataset
from .kmedoids import KMedoidsPartitioner
from .partitioners import coerce_regime_dataset, labels_to_atlas
from .validation import RegimeValidationReport, RegimeValidationSummary


TrainRegionModel = Callable[[np.ndarray, RegimeDataset], Any]
EvaluateRegionModel = Callable[[Any, np.ndarray, RegimeDataset], RegimeValidationReport | Mapping[str, Any] | float]
CandidateSplitter = Callable[[np.ndarray, RegimeDataset], np.ndarray]


@dataclass(frozen=True)
class ResidualGreedyConfig:
    max_regions: int = 8
    min_support: int = 20
    validation_tolerance: float = 1.0e-2
    improvement_margin: float = 1.0e-4
    region_penalty: float = 0.0
    max_iterations: int = 50


@dataclass(frozen=True)
class ResidualGreedySplitEvent:
    parent_label: int
    accepted: bool
    before_score: float
    after_score: float
    reason: str
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_label": int(self.parent_label),
            "accepted": bool(self.accepted),
            "before_score": float(self.before_score),
            "after_score": float(self.after_score),
            "reason": str(self.reason),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class ResidualGreedyResult:
    atlas: RegimeAtlas
    validation: RegimeValidationSummary
    events: tuple[ResidualGreedySplitEvent, ...]

    @property
    def accepted_splits(self) -> int:
        return int(sum(1 for event in self.events if event.accepted))

    @property
    def rejected_splits(self) -> int:
        return int(sum(1 for event in self.events if not event.accepted))


def _coerce_report(
    value: RegimeValidationReport | Mapping[str, Any] | float,
    *,
    region_id: str,
    indices: np.ndarray,
    tolerance: float,
) -> RegimeValidationReport:
    if isinstance(value, RegimeValidationReport):
        return value
    if isinstance(value, Mapping):
        error = float(value.get("error", value.get("validation_error", 0.0)))
        passed = bool(value.get("passed", error <= float(tolerance)))
        return RegimeValidationReport(
            region_id=region_id,
            error=error,
            passed=passed,
            validation_indices=indices,
            metrics=dict(value),
        )
    error = float(value)
    return RegimeValidationReport(
        region_id=region_id,
        error=error,
        passed=error <= float(tolerance),
        validation_indices=indices,
    )


def _default_two_way_split(indices: np.ndarray, dataset: RegimeDataset) -> np.ndarray:
    local = dataset.subset(indices)
    if local.n_samples < 2:
        return np.zeros(local.n_samples, dtype=int)
    atlas = KMedoidsPartitioner(n_regions=2, radius_quantile=1.0).fit(local)
    return atlas.labels


@dataclass
class ResidualGreedySplitter:
    """Adaptively split only regions that fail validation."""

    config: ResidualGreedyConfig = ResidualGreedyConfig()
    candidate_splitters: Sequence[CandidateSplitter] | None = None

    def _evaluate(
        self,
        *,
        labels: np.ndarray,
        dataset: RegimeDataset,
        train_model: TrainRegionModel,
        evaluate_model: EvaluateRegionModel,
    ) -> tuple[RegimeValidationSummary, dict[int, Any]]:
        reports: list[RegimeValidationReport] = []
        models: dict[int, Any] = {}
        for label in sorted(set(labels.tolist())):
            indices = np.flatnonzero(labels == int(label)).astype(int)
            if indices.size == 0:
                continue
            region_id = f"region_{int(label):03d}"
            model = train_model(indices, dataset)
            models[int(label)] = model
            raw_report = evaluate_model(model, indices, dataset)
            reports.append(
                _coerce_report(
                    raw_report,
                    region_id=region_id,
                    indices=indices,
                    tolerance=float(self.config.validation_tolerance),
                )
            )
        return RegimeValidationSummary(reports=tuple(reports)), models

    def _objective(self, summary: RegimeValidationSummary, *, n_regions: int) -> float:
        return float(summary.max_error + float(self.config.region_penalty) * int(n_regions))

    def fit(
        self,
        dataset: RegimeDataset | np.ndarray,
        *,
        train_model: TrainRegionModel,
        evaluate_model: EvaluateRegionModel,
    ) -> ResidualGreedyResult:
        ds = coerce_regime_dataset(dataset)
        labels = np.zeros(ds.n_samples, dtype=int)
        splitters = tuple(self.candidate_splitters or (_default_two_way_split,))
        events: list[ResidualGreedySplitEvent] = []

        summary, _ = self._evaluate(
            labels=labels,
            dataset=ds,
            train_model=train_model,
            evaluate_model=evaluate_model,
        )
        for _ in range(max(1, int(self.config.max_iterations))):
            if summary.passed or len(set(labels.tolist())) >= int(self.config.max_regions):
                break
            failing = [report for report in summary.reports if not report.passed]
            if not failing:
                break
            worst = max(failing, key=lambda report: report.error)
            parent_label = int(worst.region_id.removeprefix("region_"))
            parent_indices = np.flatnonzero(labels == parent_label).astype(int)
            before_score = self._objective(summary, n_regions=len(set(labels.tolist())))
            if parent_indices.size < 2 * int(self.config.min_support):
                events.append(
                    ResidualGreedySplitEvent(
                        parent_label=parent_label,
                        accepted=False,
                        before_score=before_score,
                        after_score=before_score,
                        reason="below_min_support",
                    )
                )
                break

            best_labels = None
            best_summary = None
            best_score = float("inf")
            best_metadata: dict[str, Any] = {}
            for splitter_index, splitter in enumerate(splitters):
                local_labels = np.asarray(splitter(parent_indices, ds), dtype=int).reshape(-1)
                if local_labels.size != parent_indices.size or len(set(local_labels.tolist())) < 2:
                    continue
                counts = [np.count_nonzero(local_labels == label) for label in sorted(set(local_labels.tolist()))]
                if min(counts) < int(self.config.min_support):
                    continue
                candidate = labels.copy()
                new_label = int(labels.max()) + 1
                first_local = sorted(set(local_labels.tolist()))[0]
                for local_label in sorted(set(local_labels.tolist())):
                    mask = local_labels == int(local_label)
                    candidate[parent_indices[mask]] = parent_label if local_label == first_local else new_label
                    if local_label != first_local:
                        new_label += 1
                candidate_summary, _ = self._evaluate(
                    labels=candidate,
                    dataset=ds,
                    train_model=train_model,
                    evaluate_model=evaluate_model,
                )
                score = self._objective(candidate_summary, n_regions=len(set(candidate.tolist())))
                if score < best_score:
                    best_score = score
                    best_labels = candidate
                    best_summary = candidate_summary
                    best_metadata = {"splitter_index": int(splitter_index), "child_counts": [int(v) for v in counts]}
            if best_labels is None or best_summary is None:
                events.append(
                    ResidualGreedySplitEvent(
                        parent_label=parent_label,
                        accepted=False,
                        before_score=before_score,
                        after_score=before_score,
                        reason="no_valid_candidate_split",
                    )
                )
                break
            improvement = before_score - best_score
            if improvement <= float(self.config.improvement_margin):
                events.append(
                    ResidualGreedySplitEvent(
                        parent_label=parent_label,
                        accepted=False,
                        before_score=before_score,
                        after_score=best_score,
                        reason="insufficient_validation_improvement",
                        metadata={**best_metadata, "improvement": float(improvement)},
                    )
                )
                break
            labels = best_labels
            summary = best_summary
            events.append(
                ResidualGreedySplitEvent(
                    parent_label=parent_label,
                    accepted=True,
                    before_score=before_score,
                    after_score=best_score,
                    reason="accepted",
                    metadata={**best_metadata, "improvement": float(improvement)},
                )
            )

        atlas = labels_to_atlas(
            ds,
            labels,
            radius_quantile=1.0,
            radius_safety_factor=1.05,
            metadata={"partitioner": "residual_greedy"},
        )
        return ResidualGreedyResult(atlas=atlas, validation=summary, events=tuple(events))


__all__ = [
    "CandidateSplitter",
    "EvaluateRegionModel",
    "ResidualGreedyConfig",
    "ResidualGreedyResult",
    "ResidualGreedySplitEvent",
    "ResidualGreedySplitter",
    "TrainRegionModel",
]
