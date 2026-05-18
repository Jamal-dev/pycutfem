from __future__ import annotations

import numpy as np

from pycutfem.mor import (
    LocalReducedModelBankEntry,
    OnlineErrorCalibrator,
    SampleStateTransaction,
    apply_empirical_cubature,
    build_sampled_native_kernel_bundle,
    build_stage_break_even_certificate,
    fit_gappy_pod_quantity_operator,
    fit_positive_empirical_cubature,
    select_local_reduced_model_bank,
)


def test_sampled_native_kernel_bundle_slices_entity_arrays_and_allocates_coefficients() -> None:
    static_args = {
        "gdofs_map": np.arange(20, dtype=np.int32).reshape(4, 5),
        "detJ": np.array([1.0, 2.0, 3.0, 4.0]),
        "constant": np.array([9.0]),
    }

    bundle = build_sampled_native_kernel_bundle(
        param_order=("gdofs_map", "detJ", "constant", "u_loc"),
        static_args=static_args,
        element_ids=np.array([1, 3], dtype=np.int64),
        coefficient_arg_names=("u_loc",),
    )

    np.testing.assert_array_equal(bundle.element_ids, np.array([1, 3], dtype=np.int64))
    np.testing.assert_array_equal(bundle.static_args["gdofs_map"], static_args["gdofs_map"][[1, 3], :])
    np.testing.assert_allclose(bundle.static_args["detJ"], np.array([2.0, 4.0]))
    np.testing.assert_allclose(bundle.static_args["constant"], np.array([9.0]))
    assert bundle.static_args["u_loc"].shape == (2, 5)
    assert bundle.static_args["u_loc"].dtype == np.float64


def test_gappy_pod_quantity_operator_reconstructs_from_sample_rows() -> None:
    basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    offset = np.array([0.5, -0.25, 1.0])
    operator = fit_gappy_pod_quantity_operator(
        basis=basis,
        sample_rows=np.array([0, 1], dtype=np.int64),
        offset=offset,
        name="interface_load",
    )
    reference = offset + basis @ np.array([2.0, -1.0])

    reconstructed = operator.reconstruct_from_samples(operator.sample(reference))

    np.testing.assert_allclose(reconstructed, reference)
    assert operator.relative_error(reference) <= 1.0e-14
    assert operator.name == "interface_load"


def test_gappy_pod_quantity_operator_accepts_stored_reconstruction_map() -> None:
    basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    sample_rows = np.array([0, 1], dtype=np.int64)
    sample_to_coefficients = np.linalg.pinv(basis[sample_rows, :])
    operator = fit_gappy_pod_quantity_operator(
        basis=basis,
        sample_rows=sample_rows,
        sample_to_coefficients=sample_to_coefficients,
    )
    reference = basis @ np.array([2.0, -1.0])

    np.testing.assert_allclose(operator.reconstruct_from_samples(operator.sample(reference)), reference)
    np.testing.assert_allclose(operator.sample_to_coefficients, sample_to_coefficients)


def test_positive_empirical_cubature_fits_sparse_nonnegative_rule() -> None:
    contributions = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    fit = fit_positive_empirical_cubature(contributions, max_entities=1, tolerance=1.0e-12)

    assert fit.passed
    np.testing.assert_array_equal(fit.rule.entity_ids, np.array([2], dtype=np.int64))
    np.testing.assert_allclose(fit.rule.weights, np.array([2.0]))
    np.testing.assert_allclose(apply_empirical_cubature(contributions, fit.rule), contributions.sum(axis=0))


def test_sample_state_transaction_restores_rejected_trial_and_keeps_accepted_trial() -> None:
    state = {"hidden": np.array([1.0, 2.0, 3.0])}
    transaction = SampleStateTransaction(state)

    with transaction.trial() as trial:
        state["hidden"][:] = 0.0
        assert not trial.accepted
    np.testing.assert_allclose(state["hidden"], np.array([1.0, 2.0, 3.0]))

    with transaction.trial() as trial:
        state["hidden"][:] = 5.0
        trial.accept()
    np.testing.assert_allclose(state["hidden"], np.array([5.0, 5.0, 5.0]))


def test_stage_break_even_certificate_reports_cost_gate() -> None:
    cert = build_stage_break_even_certificate(
        exact_stage_times=np.array([1.0, 1.2, 0.8]),
        reduced_stage_times=np.array([0.4, 0.5]),
        required_speedup=1.5,
        min_samples=2,
    )

    assert cert.passed
    assert cert.speedup > 2.0
    assert cert.exact_count == 3
    assert cert.reduced_count == 2


def test_online_error_calibrator_learns_reliability_factor() -> None:
    calibrator = OnlineErrorCalibrator(
        tolerance=0.02,
        max_estimate_tolerance=0.08,
        quantile=0.5,
        safety_factor=1.0,
        min_samples=2,
    )

    assert not calibrator.evaluate(0.03).accepted
    calibrator.record(estimate=0.04, true_error=0.01)
    decision = calibrator.record(estimate=0.06, true_error=0.015)

    assert decision.accepted
    assert calibrator.sample_count == 2
    assert calibrator.reliability_factor < 1.0
    assert calibrator.effective_estimate_tolerance > 0.02


def test_local_reduced_model_bank_selects_active_priority_and_feature() -> None:
    early = LocalReducedModelBankEntry(
        model_id="early",
        path="early.npz",
        step_start=1,
        step_end=50,
        priority=0,
    )
    late_far = LocalReducedModelBankEntry(
        model_id="late_far",
        path="late_far.npz",
        step_start=40,
        step_end=None,
        priority=1,
        feature_center=np.array([10.0, 0.0]),
        feature_scale=np.array([1.0, 1.0]),
        max_feature_distance=5.0,
    )
    late_near = LocalReducedModelBankEntry(
        model_id="late_near",
        path="late_near.npz",
        step_start=40,
        step_end=None,
        priority=1,
        feature_center=np.array([0.0, 0.0]),
        feature_scale=np.array([1.0, 1.0]),
    )

    selected = select_local_reduced_model_bank(
        [early, late_far, late_near],
        step=45,
        feature=np.array([0.2, 0.0]),
    )

    assert selected.selected
    assert selected.model_id == "late_near"
    assert selected.distance < 1.0
    no_radius = select_local_reduced_model_bank(
        [late_far],
        step=45,
        feature=np.array([0.2, 0.0]),
    )
    assert not no_radius.selected
    assert no_radius.reason == "no_active_feature_radius"
    assert not select_local_reduced_model_bank([early], step=60).selected
