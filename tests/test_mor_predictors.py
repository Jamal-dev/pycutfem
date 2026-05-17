from __future__ import annotations

import numpy as np

from pycutfem.mor import (
    NativeKernelReference,
    NativeReducedArtifact,
    ReferencePolicy,
    fit_time_parameterized_predictor,
    load_native_reduced_artifact,
    predictor_from_native_dict,
    save_native_reduced_artifact,
)
from pycutfem.mor.predictors import ConstantReducedPredictor, LinearHistoryReducedPredictor


def test_time_parameterized_predictor_fits_and_roundtrips() -> None:
    times = np.linspace(0.0, 1.0, 41)
    q = np.column_stack(
        [
            np.sin(0.5 * np.pi * times),
            np.cos(0.5 * np.pi * times),
            times * (1.0 - times),
        ]
    )

    predictor = fit_time_parameterized_predictor(q, times, degree=12, ridge=1.0e-14)
    predicted = np.vstack([predictor.predict(time=float(t)).coefficients for t in times])

    assert predictor.training_error_max < 1.0e-8
    np.testing.assert_allclose(predicted, q, rtol=1.0e-8, atol=1.0e-8)

    restored = predictor_from_native_dict(predictor.to_native_dict())
    np.testing.assert_allclose(restored.predict(time=0.37).coefficients, predictor.predict(time=0.37).coefficients)


def test_history_predictors_are_problem_generic() -> None:
    q0 = np.array([1.0, -2.0, 0.5])
    q1 = np.array([1.25, -1.5, 0.25])

    constant = ConstantReducedPredictor().predict(time=1.0, q_current=q1)
    linear = LinearHistoryReducedPredictor().predict(time=1.0, q_current=q1, q_previous=q0)

    np.testing.assert_allclose(constant.coefficients, q1)
    np.testing.assert_allclose(linear.coefficients, q1 + (q1 - q0))


def test_reference_policy_clips_in_decoded_metric_and_artifact_roundtrips(tmp_path) -> None:
    times = np.linspace(0.0, 1.0, 8)
    q = np.column_stack([times, times**2])
    predictor = fit_time_parameterized_predictor(q, times, degree=4, ridge=1.0e-14)
    metric_basis = np.array([[2.0, 0.0], [0.0, 1.0]])
    policy = ReferencePolicy(
        predictor=predictor,
        reference_weight=1.0e-3,
        max_reference_distance=0.25,
        max_step_norm=0.5,
        metric_basis=metric_basis,
    )

    result = policy.predict(time=1.0, q_current=np.zeros(2))

    assert result.metadata["reference_clipped"]
    assert result.metadata["reference_distance_after_clip"] == 0.25
    assert np.linalg.norm(metric_basis @ result.coefficients) <= 0.25 + 1.0e-12

    artifact = NativeReducedArtifact(
        problem_id="predictor_roundtrip",
        trial_basis=np.eye(2),
        offset=np.zeros(2),
        residual_kernel=NativeKernelReference(kernel_id="r", abi="test", param_order=("gdofs_map",)),
        reference_policy=policy,
    )
    path = tmp_path / "artifact.npz"
    save_native_reduced_artifact(artifact, path)
    loaded = load_native_reduced_artifact(path)

    assert loaded.reference_policy is not None
    loaded_result = loaded.reference_policy.predict(time=1.0, q_current=np.zeros(2))
    np.testing.assert_allclose(loaded_result.coefficients, result.coefficients)
