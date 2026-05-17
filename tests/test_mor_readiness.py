from __future__ import annotations

import numpy as np

from pycutfem.mor import (
    MORReadinessCriteria,
    NativeAdjointDWRSpec,
    NativeGnatTargetSpec,
    NativeKernelReference,
    NativeReducedArtifact,
    ReferencePolicy,
    TimeParameterizedReducedPredictor,
    certify_mor_readiness,
)


def _summary() -> dict:
    return {
        "benchmark": "seboldt_three_constituent",
        "passed": True,
        "errors": {
            "relative_state_vs_fom": 1.1066e-3,
            "projection_relative_state_vs_fom": 1.557e-5,
            "max_bound_violation": 0.0,
            "field_errors": {
                "alpha": {"max_relative_error": 1.0e-3, "passed": True},
                "phi": {"max_relative_error": 2.0e-3, "passed": True},
            },
        },
        "speedup": {
            "predictive_validated_factor": 3.79,
            "predictive_validation_passed": True,
            "replay_validation": False,
        },
        "offline": {
            "sampling": {
                "interface_complete": True,
                "missing_mandatory_element_count": 0,
            }
        },
        "dwr": {
            "certificate": {
                "passed": True,
                "estimate": {"effectivity": 14.21},
            }
        },
    }


def _artifact() -> NativeReducedArtifact:
    predictor = TimeParameterizedReducedPredictor(
        coefficients=np.zeros((2, 2), dtype=float),
        time_center=0.5,
        time_half_width=0.5,
        degree=1,
        ridge=0.0,
    )
    return NativeReducedArtifact(
        problem_id="readiness",
        trial_basis=np.eye(2, dtype=float),
        offset=np.zeros(2, dtype=float),
        residual_kernel=NativeKernelReference(
            kernel_id="residual",
            abi="native-kernel-v1",
            param_order=("gdofs_map", "u_loc"),
        ),
        tangent_kernel=NativeKernelReference(
            kernel_id="tangent",
            abi="native-kernel-v1",
            param_order=("gdofs_map", "u_loc"),
        ),
        target=NativeGnatTargetSpec(
            row_dofs=np.array([0, 1], dtype=np.int64),
            objective="sampled_gnat",
            metadata={"interface_complete": True},
        ),
        adjoint_dwr=NativeAdjointDWRSpec(qoi_name="alpha_mass"),
        reference_policy=ReferencePolicy(
            predictor=predictor,
            reference_weight=1.0e-4,
            max_reference_distance=1.0,
        ),
    )


def test_mor_readiness_accepts_certified_predictive_seboldt_summary() -> None:
    certificate = certify_mor_readiness(
        _summary(),
        artifact=_artifact(),
        milestone_statuses={24: "Completed", 33: "Completed", 34: "Completed", 35: "Completed", 36: "Completed", 37: "Completed", 38: "Completed", 39: "Completed"},
    )

    assert certificate.passed
    assert not certificate.failed_gates
    assert "nutrient" in certificate.recommended_next_fields


def test_mor_readiness_rejects_replay_or_missing_artifact_metadata() -> None:
    summary = _summary()
    summary["speedup"]["replay_validation"] = True
    artifact = _artifact().to_dict()
    artifact["reference_policy"] = None

    certificate = certify_mor_readiness(
        summary,
        artifact=artifact,
        criteria=MORReadinessCriteria(require_field_error_metadata=False),
        milestone_statuses={24: True, 33: True, 34: True, 35: True, 36: True, 37: True, 38: True, 39: True},
    )

    failed = {gate.name for gate in certificate.failed_gates}
    assert "predictive_validation" in failed
    assert "artifact_metadata" in failed
