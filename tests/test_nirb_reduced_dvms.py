from __future__ import annotations

import numpy as np
import pytest

from examples.NIRB.reduced_dvms import ReducedDVMSState, update_predicted_subscale_local


def _zeros(n_elements: int = 2, n_q: int = 3) -> np.ndarray:
    return np.zeros((n_elements, n_q, 2), dtype=float)


def _base_kwargs() -> dict[str, object]:
    n_elements = 2
    n_q = 3
    u_k_q = np.array(
        [
            [[0.20, -0.10], [0.25, -0.05], [0.30, 0.00]],
            [[-0.10, 0.15], [-0.15, 0.20], [-0.20, 0.25]],
        ],
        dtype=float,
    )
    return {
        "predicted_subscale_velocity": _zeros(n_elements, n_q),
        "old_subscale_velocity": np.full((n_elements, n_q, 2), [0.02, -0.01], dtype=float),
        "u_k_q": u_k_q,
        "u_prev_q": u_k_q - 0.01,
        "a_prev_q": _zeros(n_elements, n_q),
        "p_grad_phys": np.array([[0.03, -0.02], [-0.01, 0.04]], dtype=float),
        "grad_u_phys": np.array(
            [
                [[0.08, 0.01], [-0.02, 0.05]],
                [[0.04, -0.03], [0.02, 0.06]],
            ],
            dtype=float,
        ),
        "resolved_conv_velocity": u_k_q.copy(),
        "h_e": np.array([0.5, 0.7], dtype=float),
        "rho_f": 1.3,
        "mu_f": 0.08,
        "dt": 0.2,
        "bossak_alpha": -0.2,
    }


def test_reduced_dvms_state_validates_and_selects_elements() -> None:
    state = ReducedDVMSState(
        predicted_subscale_velocity=np.arange(12, dtype=float).reshape(2, 3, 2),
        old_subscale_velocity=_zeros(),
        momentum_projection=_zeros(),
        mass_projection=np.zeros((2, 3), dtype=float),
        old_mass_residual=np.ones((2, 3), dtype=float),
    )

    selected = state.select([1])

    assert selected.n_elements == 1
    assert selected.n_q == 3
    np.testing.assert_allclose(selected.predicted_subscale_velocity, state.predicted_subscale_velocity[1:2])


def test_reduced_dvms_state_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="old_subscale_velocity"):
        ReducedDVMSState(
            predicted_subscale_velocity=_zeros(),
            old_subscale_velocity=np.zeros((2, 2, 2), dtype=float),
            momentum_projection=_zeros(),
            mass_projection=np.zeros((2, 3), dtype=float),
            old_mass_residual=np.zeros((2, 3), dtype=float),
        )


def test_update_predicted_subscale_local_converges_to_small_fixed_point_residual() -> None:
    kwargs = _base_kwargs()

    out = update_predicted_subscale_local(**kwargs, max_iterations=20, rel_tol=1.0e-12, abs_tol=1.0e-20)

    assert out.shape == (2, 3, 2)
    assert np.all(np.isfinite(out))
    assert np.linalg.norm(out) > 0.0

    dt = float(kwargs["dt"])
    rho = float(kwargs["rho_f"])
    mu = float(kwargs["mu_f"])
    h = np.asarray(kwargs["h_e"], dtype=float)[:, None]
    grad_u = np.broadcast_to(np.asarray(kwargs["grad_u_phys"], dtype=float)[:, None, :, :], (2, 3, 2, 2))
    static_residual = -(
        rho
        * (
            (1.0 / ((0.5 - float(kwargs["bossak_alpha"])) * dt))
            * (np.asarray(kwargs["u_k_q"]) - np.asarray(kwargs["u_prev_q"]))
            + ((-1.0 + (0.5 - float(kwargs["bossak_alpha"]))) / (0.5 - float(kwargs["bossak_alpha"])))
            * np.asarray(kwargs["a_prev_q"])
        )
        + rho * np.einsum("eqij,eqj->eqi", grad_u, np.asarray(kwargs["resolved_conv_velocity"]), optimize=True)
        + np.asarray(kwargs["p_grad_phys"], dtype=float)[:, None, :]
    ) + (rho / dt) * np.asarray(kwargs["old_subscale_velocity"], dtype=float)
    inv_tau = 8.0 * mu / (h * h) + rho * (
        1.0 / dt + 2.0 * np.linalg.norm(np.asarray(kwargs["resolved_conv_velocity"]) + out, axis=2) / h
    )
    linearization = rho * grad_u.copy()
    linearization[:, :, 0, 0] += inv_tau
    linearization[:, :, 1, 1] += inv_tau
    residual = static_residual - np.einsum("eqij,eqj->eqi", linearization, out, optimize=True)

    assert np.max(np.linalg.norm(residual, axis=2)) < 1.0e-8


def test_update_predicted_subscale_local_returns_finite_output() -> None:
    kwargs = _base_kwargs()
    kwargs["momentum_projection"] = np.full((2, 3, 2), [0.1, -0.2], dtype=float)

    out = update_predicted_subscale_local(**kwargs)

    assert out.shape == (2, 3, 2)
    assert np.all(np.isfinite(out))


def test_update_predicted_subscale_local_zeros_singular_or_invalid_solves() -> None:
    kwargs = _base_kwargs()
    kwargs.update(
        {
            "predicted_subscale_velocity": np.ones((2, 3, 2), dtype=float),
            "old_subscale_velocity": _zeros(),
            "grad_u_phys": np.zeros((2, 2, 2), dtype=float),
            "p_grad_phys": np.ones((2, 2), dtype=float),
            "rho_f": 0.0,
            "mu_f": 0.0,
        }
    )

    out = update_predicted_subscale_local(**kwargs)

    np.testing.assert_allclose(out, 0.0, atol=0.0)


def test_update_predicted_subscale_local_validates_shapes() -> None:
    kwargs = _base_kwargs()
    kwargs["u_prev_q"] = np.zeros((2, 2, 2), dtype=float)

    with pytest.raises(ValueError, match="u_prev_q"):
        update_predicted_subscale_local(**kwargs)
