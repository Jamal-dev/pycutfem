"""Readiness gates for moving a certified MOR stack to a harder model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MORReadinessGate:
    """One readiness decision with the value that was checked."""

    name: str
    passed: bool
    value: Any = None
    threshold: Any = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "value": self.value,
            "threshold": self.threshold,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MORReadinessCriteria:
    """Thresholds for a certified native MOR readiness decision."""

    max_state_error: float = 2.0e-2
    max_projection_error: float = 5.0e-2
    min_validated_speedup: float = 1.0
    max_bound_violation: float = 1.0e-8
    require_predictive_validation: bool = True
    require_dwr: bool = True
    require_interface_complete: bool = True
    require_artifact_metadata: bool = True
    require_field_error_metadata: bool = True
    required_milestones: tuple[int, ...] = (24, 33, 34, 35, 36, 37, 38, 39)

    def __post_init__(self) -> None:
        for name in ("max_state_error", "max_projection_error", "min_validated_speedup", "max_bound_violation"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "required_milestones", tuple(int(v) for v in self.required_milestones))


@dataclass(frozen=True)
class MORReadinessCertificate:
    """Aggregate readiness result for moving to the next model."""

    passed: bool
    gates: tuple[MORReadinessGate, ...]
    recommended_next_fields: tuple[str, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "gates", tuple(self.gates))
        object.__setattr__(self, "recommended_next_fields", tuple(str(v) for v in self.recommended_next_fields))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def failed_gates(self) -> tuple[MORReadinessGate, ...]:
        return tuple(gate for gate in self.gates if not gate.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "gates": tuple(gate.to_dict() for gate in self.gates),
            "recommended_next_fields": tuple(self.recommended_next_fields),
            "metadata": dict(self.metadata),
        }


def _nested(mapping: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "passed", "completed"}
    return bool(value)


def _status_completed(value: Any) -> bool:
    if isinstance(value, Mapping):
        return _status_completed(value.get("status", value.get("passed", False)))
    if isinstance(value, str):
        return "completed" in value.strip().lower() or value.strip().lower() in {"done", "passed"}
    return bool(value)


def _finite_float(value: Any, default: float = float("inf")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _artifact_metadata_gate(artifact: Any | Mapping[str, Any] | None) -> MORReadinessGate:
    if artifact is None:
        return MORReadinessGate("artifact_metadata", False, value="missing", threshold="native reduced artifact")
    if isinstance(artifact, Mapping):
        target = artifact.get("target")
        tangent = artifact.get("tangent_kernel")
        adjoint = artifact.get("adjoint_dwr")
        reference = artifact.get("reference_policy")
        problem_id = artifact.get("problem_id")
    else:
        target = getattr(artifact, "target", None)
        tangent = getattr(artifact, "tangent_kernel", None)
        adjoint = getattr(artifact, "adjoint_dwr", None)
        reference = getattr(artifact, "reference_policy", None)
        problem_id = getattr(artifact, "problem_id", None)
    missing = [
        name
        for name, value in (
            ("problem_id", problem_id),
            ("target", target),
            ("tangent_kernel", tangent),
            ("adjoint_dwr", adjoint),
            ("reference_policy", reference),
        )
        if value is None
    ]
    return MORReadinessGate(
        "artifact_metadata",
        not missing,
        value={"problem_id": problem_id, "missing": tuple(missing)},
        threshold="problem_id,target,tangent_kernel,adjoint_dwr,reference_policy",
    )


def _field_error_gate(summary: Mapping[str, Any], extra_field_errors: Mapping[str, Any] | None) -> MORReadinessGate:
    candidates = (
        extra_field_errors,
        _nested(summary, ("errors", "field_errors")),
        _nested(summary, ("offline", "field_errors")),
        _nested(summary, ("mixed_stability", "field_errors")),
    )
    field_errors = next((dict(value) for value in candidates if isinstance(value, Mapping) and value), {})
    if not field_errors:
        return MORReadinessGate("field_error_metadata", False, value="missing", threshold="per-field error map")
    worst = 0.0
    failing: list[str] = []
    for name, value in field_errors.items():
        if isinstance(value, Mapping):
            err = _finite_float(value.get("max_relative_error", value.get("relative_error", 0.0)), 0.0)
            passed = _as_bool(value.get("passed", True))
        else:
            err = _finite_float(value, 0.0)
            passed = True
        worst = max(worst, err)
        if not passed:
            failing.append(str(name))
    return MORReadinessGate(
        "field_error_metadata",
        len(failing) == 0,
        value={"field_count": len(field_errors), "worst_relative_error": worst, "failing": tuple(failing)},
        threshold="all field gates passed",
    )


def certify_mor_readiness(
    validation_summary: Mapping[str, Any],
    *,
    artifact: Any | Mapping[str, Any] | None = None,
    criteria: MORReadinessCriteria | None = None,
    milestone_statuses: Mapping[int | str, Any] | None = None,
    field_errors: Mapping[str, Any] | None = None,
    next_fields: Sequence[str] = ("nutrient", "growth", "damage", "detachment"),
    metadata: Mapping[str, Any] | None = None,
) -> MORReadinessCertificate:
    """Certify readiness to move a MOR stack to a harder physical model.

    The function is intentionally summary-driven: examples and CI jobs can feed
    JSON output from a certified validation run plus an optional native reduced
    artifact.  The resulting certificate is explicit about which gate failed.
    """

    summary = dict(validation_summary)
    cfg = MORReadinessCriteria() if criteria is None else criteria
    gates: list[MORReadinessGate] = []

    statuses = {int(k): v for k, v in dict(milestone_statuses or {}).items()}
    if statuses:
        missing = tuple(m for m in cfg.required_milestones if not _status_completed(statuses.get(m, False)))
        gates.append(
            MORReadinessGate(
                "required_milestones",
                len(missing) == 0,
                value={"missing": missing},
                threshold=cfg.required_milestones,
            )
        )

    passed_summary = _as_bool(summary.get("passed", False))
    gates.append(MORReadinessGate("validation_summary_passed", passed_summary, value=passed_summary, threshold=True))

    state_error = _finite_float(_nested(summary, ("errors", "relative_state_vs_fom")), float("inf"))
    gates.append(
        MORReadinessGate(
            "state_error",
            state_error <= cfg.max_state_error,
            value=state_error,
            threshold=cfg.max_state_error,
        )
    )
    projection_error = _finite_float(_nested(summary, ("errors", "projection_relative_state_vs_fom")), float("inf"))
    gates.append(
        MORReadinessGate(
            "projection_error",
            projection_error <= cfg.max_projection_error,
            value=projection_error,
            threshold=cfg.max_projection_error,
        )
    )
    speed = _finite_float(
        _nested(summary, ("speedup", "predictive_validated_factor"), _nested(summary, ("speedup", "validated_factor"))),
        0.0,
    )
    gates.append(
        MORReadinessGate(
            "validated_speedup",
            speed >= cfg.min_validated_speedup,
            value=speed,
            threshold=cfg.min_validated_speedup,
        )
    )
    if cfg.require_predictive_validation:
        predictive = _as_bool(_nested(summary, ("speedup", "predictive_validation_passed"), False))
        replay = _as_bool(_nested(summary, ("speedup", "replay_validation"), True))
        gates.append(
            MORReadinessGate(
                "predictive_validation",
                predictive and not replay,
                value={"predictive_validation_passed": predictive, "replay_validation": replay},
                threshold={"predictive_validation_passed": True, "replay_validation": False},
            )
        )
    bound_violation = _finite_float(
        _nested(summary, ("errors", "max_bound_violation"), _nested(summary, ("bounds", "max_violation"), 0.0)),
        0.0,
    )
    gates.append(
        MORReadinessGate(
            "decoded_bounds",
            bound_violation <= cfg.max_bound_violation,
            value=bound_violation,
            threshold=cfg.max_bound_violation,
        )
    )
    if cfg.require_dwr:
        dwr_passed = _as_bool(_nested(summary, ("dwr", "certificate", "passed"), False))
        effectivity = _finite_float(_nested(summary, ("dwr", "certificate", "estimate", "effectivity")), float("inf"))
        gates.append(
            MORReadinessGate(
                "dwr_certificate",
                dwr_passed,
                value={"passed": dwr_passed, "effectivity": effectivity},
                threshold=True,
            )
        )
    if cfg.require_interface_complete:
        missing = int(_finite_float(_nested(summary, ("offline", "sampling", "missing_mandatory_element_count"), 0), 0))
        complete = _as_bool(_nested(summary, ("offline", "sampling", "interface_complete"), missing == 0))
        gates.append(
            MORReadinessGate(
                "interface_sampling",
                complete and missing == 0,
                value={"interface_complete": complete, "missing_mandatory_element_count": missing},
                threshold={"interface_complete": True, "missing_mandatory_element_count": 0},
            )
        )
    if cfg.require_artifact_metadata:
        gates.append(_artifact_metadata_gate(artifact if artifact is not None else summary.get("artifact")))
    if cfg.require_field_error_metadata:
        gates.append(_field_error_gate(summary, field_errors))

    passed = all(gate.passed for gate in gates)
    return MORReadinessCertificate(
        passed=passed,
        gates=tuple(gates),
        recommended_next_fields=tuple(next_fields),
        metadata={
            "summary_name": summary.get("name", summary.get("benchmark")),
            "criteria": {
                "max_state_error": cfg.max_state_error,
                "max_projection_error": cfg.max_projection_error,
                "min_validated_speedup": cfg.min_validated_speedup,
                "max_bound_violation": cfg.max_bound_violation,
            },
            **dict(metadata or {}),
        },
    )


__all__ = [
    "MORReadinessCertificate",
    "MORReadinessCriteria",
    "MORReadinessGate",
    "certify_mor_readiness",
]
