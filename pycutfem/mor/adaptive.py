"""Certified adaptive enrichment helpers for MOR artifacts.

The routines here do not know any specific PDE.  They consume algebraic
certificates produced by the mixed-basis, GNAT, branch, and DWR layers and
return deterministic enrichment actions that an offline driver can execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .dwr import DWREstimate, dominant_dwr_contributions


@dataclass(frozen=True)
class AdaptiveEnrichmentAction:
    """One requested enrichment of a certified reduced model."""

    kind: str
    target: str
    priority: float
    reason: str
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.priority)):
            raise ValueError("adaptive enrichment action priority must be finite.")
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "target", str(self.target))
        object.__setattr__(self, "priority", float(self.priority))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "priority": float(self.priority),
            "reason": self.reason,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class AdaptiveMORDecision:
    """Certified adaptive-loop decision for one ROM attempt."""

    accepted: bool
    actions: tuple[AdaptiveEnrichmentAction, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        actions = tuple(sorted(self.actions, key=lambda item: item.priority, reverse=True))
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "actions": tuple(action.to_dict() for action in self.actions),
            "metadata": dict(self.metadata or {}),
        }


def _field_error_items(field_errors: Mapping[str, Any] | None) -> list[tuple[str, float, bool]]:
    out: list[tuple[str, float, bool]] = []
    for name, value in dict(field_errors or {}).items():
        if hasattr(value, "max_relative_error"):
            err = float(getattr(value, "max_relative_error"))
            passed = bool(getattr(value, "passed", False))
        elif isinstance(value, Mapping):
            err = float(value.get("max_relative_error", value.get("relative_error", 0.0)))
            passed = bool(value.get("passed", err == 0.0))
        else:
            err = float(value)
            passed = False
        out.append((str(name), err, passed))
    return out


def select_certified_adaptive_enrichment_actions(
    *,
    field_errors: Mapping[str, Any] | None = None,
    dwr_estimate: DWREstimate | Mapping[str, Any] | None = None,
    norm_equivalence_certificate: Any | Mapping[str, Any] | None = None,
    branch_certificate: Mapping[str, Any] | None = None,
    mixed_stability_certificate: Any | Mapping[str, Any] | None = None,
    projection_tolerance: float = 5.0e-2,
    max_effectivity: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AdaptiveMORDecision:
    """Select primal/adjoint/collateral/GNAT enrichment actions from gates."""

    actions: list[AdaptiveEnrichmentAction] = []
    for name, err, passed in _field_error_items(field_errors):
        if (not passed) or err > float(projection_tolerance):
            actions.append(
                AdaptiveEnrichmentAction(
                    kind="primal_basis",
                    target=name,
                    priority=err,
                    reason="field_projection_error",
                    metadata={"max_relative_error": err, "tolerance": float(projection_tolerance)},
                )
            )

    if dwr_estimate is not None:
        if isinstance(dwr_estimate, DWREstimate):
            estimate_abs = float(dwr_estimate.absolute_estimate)
            effectivity = dwr_estimate.effectivity
            dominant = dominant_dwr_contributions(dwr_estimate, max_entries=3)
        else:
            estimate_abs = float(dwr_estimate.get("absolute_estimate", abs(float(dwr_estimate.get("estimate", 0.0)))))
            effectivity = dwr_estimate.get("effectivity")
            dominant = {}
        if max_effectivity is not None and effectivity is not None and float(effectivity) > float(max_effectivity):
            actions.append(
                AdaptiveEnrichmentAction(
                    kind="adjoint_basis",
                    target="qoi",
                    priority=float(effectivity),
                    reason="dwr_effectivity_overconservative",
                    metadata={"effectivity": float(effectivity), "max_effectivity": float(max_effectivity)},
                )
            )
        if estimate_abs > 0.0:
            actions.append(
                AdaptiveEnrichmentAction(
                    kind="adjoint_basis",
                    target="qoi",
                    priority=estimate_abs,
                    reason="dwr_qoi_indicator",
                    metadata={"dominant": dominant},
                )
            )

    if norm_equivalence_certificate is not None:
        if isinstance(norm_equivalence_certificate, Mapping):
            norm_passed = bool(norm_equivalence_certificate.get("passed", False))
            lower = float(norm_equivalence_certificate.get("lower_constant", 0.0))
        else:
            norm_passed = bool(getattr(norm_equivalence_certificate, "passed", False))
            lower = float(getattr(norm_equivalence_certificate, "lower_constant", 0.0))
        if not norm_passed:
            actions.append(
                AdaptiveEnrichmentAction(
                    kind="gnat_rows",
                    target="sampled_residual",
                    priority=1.0 / max(lower, 1.0e-300),
                    reason="norm_equivalence_failed",
                    metadata={"lower_constant": lower},
                )
            )

    if branch_certificate is not None and not bool(branch_certificate.get("passed", False)):
        actions.append(
            AdaptiveEnrichmentAction(
                kind="branch_reference",
                target="trajectory_predictor",
                priority=float(branch_certificate.get("max_branch_distance", 1.0)),
                reason="branch_certificate_failed",
                metadata=dict(branch_certificate),
            )
        )

    if mixed_stability_certificate is not None:
        if isinstance(mixed_stability_certificate, Mapping):
            mixed_passed = bool(mixed_stability_certificate.get("passed", False))
            ranks = mixed_stability_certificate.get("coupling_ranks", {})
        else:
            mixed_passed = bool(getattr(mixed_stability_certificate, "passed", False))
            ranks = getattr(mixed_stability_certificate, "coupling_ranks", {})
        if not mixed_passed:
            for name, cert in dict(ranks).items():
                passed = bool(cert.get("passed", False) if isinstance(cert, Mapping) else getattr(cert, "passed", False))
                if not passed:
                    actions.append(
                        AdaptiveEnrichmentAction(
                            kind="lift_enrichment",
                            target=str(name),
                            priority=1.0,
                            reason="mixed_coupling_rank_failed",
                            metadata=cert if isinstance(cert, Mapping) else cert.to_dict(),
                        )
                    )

    accepted = len(actions) == 0
    return AdaptiveMORDecision(
        accepted=accepted,
        actions=tuple(actions),
        metadata=dict(metadata or {}),
    )


def augment_rows_from_dwr_localization(
    residuals: Any,
    adjoints: Any,
    row_dofs: Any,
    *,
    max_new_rows: int,
    mandatory_rows: Any | None = None,
) -> np.ndarray:
    """Return sampled rows augmented by largest unsampled DWR row indicators."""

    R = np.asarray(residuals, dtype=float)
    Z = np.asarray(adjoints, dtype=float)
    if R.ndim == 1:
        R = R[None, :]
    if Z.ndim == 1:
        Z = Z[None, :]
    if R.shape != Z.shape:
        raise ValueError("residuals and adjoints must have matching shape.")
    rows = np.unique(np.asarray(row_dofs, dtype=np.int64).reshape(-1))
    mandatory = (
        np.zeros(0, dtype=np.int64)
        if mandatory_rows is None
        else np.unique(np.asarray(mandatory_rows, dtype=np.int64).reshape(-1))
    )
    if np.any(rows < 0) or np.any(mandatory < 0) or np.any(rows >= R.shape[1]) or np.any(mandatory >= R.shape[1]):
        raise ValueError("row ids are outside the residual width.")
    selected = np.union1d(rows, mandatory).astype(np.int64, copy=False)
    count = max(0, int(max_new_rows))
    if count == 0:
        return np.ascontiguousarray(selected, dtype=np.int64)
    scores = np.sum(np.abs(R * Z), axis=0)
    available = np.setdiff1d(np.arange(R.shape[1], dtype=np.int64), selected, assume_unique=False)
    if available.size == 0:
        return np.ascontiguousarray(selected, dtype=np.int64)
    order = available[np.argsort(scores[available])[::-1]]
    add = order[: min(count, order.size)]
    return np.ascontiguousarray(np.union1d(selected, add).astype(np.int64, copy=False), dtype=np.int64)


__all__ = [
    "AdaptiveEnrichmentAction",
    "AdaptiveMORDecision",
    "augment_rows_from_dwr_localization",
    "select_certified_adaptive_enrichment_actions",
]
