from __future__ import annotations

import numpy as np

from pycutfem.solvers.coupling_acceleration import (
    AitkenCouplingAccelerator,
    IQNILSCouplingAccelerator,
    MVQNCouplingAccelerator,
    create_coupling_accelerator,
)


def test_aitken_accelerator_clips_and_uses_history_after_first_iteration() -> None:
    accel = AitkenCouplingAccelerator(
        init_relaxation=0.5,
        relaxation_min=1.0e-3,
        relaxation_max=1.0,
    )
    accel.initialize_solution_step()

    first = accel.compute_next_iterate(
        x_curr=np.array([0.0, 0.0]),
        residual_curr=np.array([1.0, 0.0]),
    )
    second = accel.compute_next_iterate(
        x_curr=np.array([0.5, 0.0]),
        residual_curr=np.array([0.2, 0.0]),
    )

    assert first.used_history is False
    assert np.allclose(first.next_iterate, np.array([0.5, 0.0]))
    assert second.used_history is True
    assert 1.0e-3 <= second.relaxation <= 1.0


def test_iqnils_falls_back_to_relaxed_picard_without_history() -> None:
    accel = IQNILSCouplingAccelerator(
        iteration_horizon=3,
        timestep_horizon=1,
        alpha=0.5,
        regularization=1.0e-10,
    )
    accel.initialize_solution_step()

    step = accel.compute_next_iterate(
        x_curr=np.array([[1.0, 2.0]]),
        residual_curr=np.array([[2.0, 4.0]]),
    )

    assert step.used_history is False
    assert np.allclose(step.next_iterate, np.array([2.0, 4.0]))


def test_iqnils_reuses_previous_step_matrices_on_first_iteration() -> None:
    accel = IQNILSCouplingAccelerator(
        iteration_horizon=4,
        timestep_horizon=2,
        alpha=0.5,
        regularization=0.0,
    )

    x0 = np.array([0.0, 0.0])
    r0 = np.array([0.4, 0.3])
    x1 = np.array([0.8, 0.1])
    r1 = np.array([0.2, 0.5])

    accel.initialize_solution_step()
    accel.compute_next_iterate(x_curr=x0, residual_curr=r0)
    accel.compute_next_iterate(x_curr=x1, residual_curr=r1)
    accel.finalize_solution_step(accepted=True)

    x2 = np.array([1.2, 0.4])
    r2 = np.array([0.1, 0.4])

    accel.initialize_solution_step()
    step = accel.compute_next_iterate(x_curr=x2, residual_curr=r2)

    v_old = (r1 - r0).reshape(-1, 1)
    g0 = x0 + r0
    g1 = x1 + r1
    w_old = (g1 - g0).reshape(-1, 1)
    coeff = np.linalg.lstsq(v_old, -r2, rcond=None)[0]
    expected_delta = (w_old @ coeff).reshape(-1) + r2

    assert step.used_history is True
    assert np.allclose(step.delta, expected_delta)


def test_mvqn_matches_kratos_style_update_formula() -> None:
    accel = MVQNCouplingAccelerator(horizon=4, alpha=0.5)
    accel.initialize_solution_step()

    x0 = np.array([0.0, 0.0])
    r0 = np.array([1.0, 1.0])
    first = accel.compute_next_iterate(x_curr=x0, residual_curr=r0)
    assert np.allclose(first.next_iterate, np.array([0.5, 0.5]))

    x1 = np.array([0.5, 0.5])
    r1 = np.array([0.25, 0.75])
    second = accel.compute_next_iterate(x_curr=x1, residual_curr=r1)

    jacobian = -np.identity(2, dtype=float) / 0.5
    v_mat = (r1 - r0).reshape(-1, 1)
    w_mat = (x1 - x0).reshape(-1, 1)
    rhs = v_mat - jacobian @ w_mat
    w_right_inverse = np.linalg.lstsq(w_mat, np.identity(2, dtype=float), rcond=None)[0]
    jacobian_hat = jacobian + rhs @ w_right_inverse
    expected_delta = np.linalg.solve(jacobian_hat, -r1)

    assert second.used_history is True
    assert np.allclose(second.delta, expected_delta)


def test_mvqn_reuses_accepted_jacobian_on_next_step() -> None:
    accel = MVQNCouplingAccelerator(horizon=4, alpha=0.5)

    x0 = np.array([0.0, 0.0])
    r0 = np.array([1.0, 1.0])
    x1 = np.array([0.5, 0.5])
    r1 = np.array([0.25, 0.75])

    accel.initialize_solution_step()
    accel.compute_next_iterate(x_curr=x0, residual_curr=r0)
    accel.compute_next_iterate(x_curr=x1, residual_curr=r1)
    accel.finalize_solution_step(accepted=True)

    r2 = np.array([0.2, -0.1])
    accel.initialize_solution_step()
    step = accel.compute_next_iterate(x_curr=np.array([0.7, 0.3]), residual_curr=r2)

    jacobian = -np.identity(2, dtype=float) / 0.5
    v_mat = (r1 - r0).reshape(-1, 1)
    w_mat = (x1 - x0).reshape(-1, 1)
    rhs = v_mat - jacobian @ w_mat
    w_right_inverse = np.linalg.lstsq(w_mat, np.identity(2, dtype=float), rcond=None)[0]
    jacobian_hat = jacobian + rhs @ w_right_inverse
    expected_delta = np.linalg.solve(jacobian_hat, -r2)

    assert step.used_history is True
    assert np.allclose(step.delta, expected_delta)


def test_factory_creates_mvqn_accelerator() -> None:
    accel = create_coupling_accelerator(
        "mvqn",
        relaxation=0.5,
        history=3,
        regularization=0.0,
        timestep_horizon=1,
    )

    assert isinstance(accel, MVQNCouplingAccelerator)
