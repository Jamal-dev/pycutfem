import numpy as np
import pytest

from pycutfem.solvers import (
    RammArcLengthParameters,
    RammArcLengthState,
    initialize_ramm_arc_length_state,
    ramm_arc_length_step,
)


def test_ramm_arc_length_initial_radius_matches_kratos_rule():
    tangent = np.asarray([[2.0]], dtype=float)
    reference_load = np.asarray([4.0], dtype=float)

    state = initialize_ramm_arc_length_state(tangent, reference_load)

    assert state.radius_0 == pytest.approx(2.0)
    assert state.radius == pytest.approx(2.0)
    assert state.lambda_value == pytest.approx(0.0)
    assert state.lambda_old == pytest.approx(1.0)


def test_ramm_arc_length_linear_predictor_matches_kratos_update_and_radius_clamp():
    tangent = np.asarray([[2.0]], dtype=float)
    reference_load = np.asarray([4.0], dtype=float)
    state = initialize_ramm_arc_length_state(tangent, reference_load)
    params = RammArcLengthParameters(
        desired_iterations=16,
        max_radius_factor=3.0,
        min_radius_factor=1.0,
        residual_tolerance=1.0e-12,
    )

    def K(_x, _lam):
        return tangent

    def residual(x, lam):
        return lam * reference_load - tangent @ x

    result = ramm_arc_length_step(
        np.zeros(1),
        state,
        tangent_callback=K,
        residual_callback=residual,
        reference_load=reference_load,
        params=params,
    )

    assert result.converged
    assert result.iterations == 1
    np.testing.assert_allclose(result.x, [2.0], rtol=0.0, atol=1.0e-14)
    assert result.state.lambda_value == pytest.approx(1.0)
    assert result.delta_lambda_step == pytest.approx(1.0)
    assert result.state.radius == pytest.approx(6.0)


def test_ramm_arc_length_can_require_residual_and_update_convergence():
    tangent = np.asarray([[2.0]], dtype=float)
    reference_load = np.asarray([4.0], dtype=float)
    state = initialize_ramm_arc_length_state(tangent, reference_load)
    params = RammArcLengthParameters(
        desired_iterations=4,
        residual_tolerance=1.0e-12,
        update_tolerance=1.0e-12,
    )

    def K(_x, _lam):
        return tangent

    def residual(x, lam):
        return lam * reference_load - tangent @ x

    result = ramm_arc_length_step(
        np.zeros(1),
        state,
        tangent_callback=K,
        residual_callback=residual,
        reference_load=reference_load,
        params=params,
    )

    assert result.converged
    assert result.iterations == 2
    np.testing.assert_allclose(result.x, [2.0], rtol=0.0, atol=1.0e-14)
    assert result.history[0].update_norm > params.update_tolerance
    assert result.history[1].update_norm <= params.update_tolerance
    assert result.state.lambda_value == pytest.approx(1.0)
    assert result.state.radius == pytest.approx(2.0 * np.sqrt(2.0))


def test_ramm_arc_length_correction_formula_matches_kratos_header():
    reference_load = np.asarray([2.0, 0.0], dtype=float)
    state = RammArcLengthState(radius_0=2.0, radius=2.0)
    params = RammArcLengthParameters(
        desired_iterations=4,
        max_iterations=3,
        residual_tolerance=1.0e-12,
    )

    def K(_x, _lam):
        return np.eye(2)

    def residual(x, lam):
        del lam
        if x[1] >= 1.0 - 1.0e-14:
            return np.zeros(2)
        return np.asarray([0.0, 1.0], dtype=float)

    result = ramm_arc_length_step(
        np.zeros(2),
        state,
        tangent_callback=K,
        residual_callback=residual,
        reference_load=reference_load,
        params=params,
    )

    assert result.converged
    assert result.iterations == 2
    np.testing.assert_allclose(result.x, [2.0, 1.0], rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.dx_step, [2.0, 1.0], rtol=0.0, atol=1.0e-14)
    assert result.history[0].delta_lambda == pytest.approx(1.0)
    assert result.history[1].delta_lambda == pytest.approx(0.0)
    assert result.state.lambda_value == pytest.approx(1.0)


def test_ramm_arc_length_rejects_zero_correction_denominator():
    reference_load = np.asarray([0.0, 1.0], dtype=float)
    state = RammArcLengthState(radius_0=1.0, radius=1.0)

    def K(x, _lam):
        if np.linalg.norm(x) > 0.0:
            return np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        return np.eye(2)

    def residual(_x, _lam):
        return np.asarray([1.0, 0.0], dtype=float)

    with pytest.raises(RuntimeError, match="denominator is zero"):
        ramm_arc_length_step(
            np.zeros(2),
            state,
            tangent_callback=K,
            residual_callback=residual,
            reference_load=reference_load,
            params=RammArcLengthParameters(max_iterations=2, residual_tolerance=1.0e-16),
        )
