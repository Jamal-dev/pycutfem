import numpy as np

from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import (
    _latent_forward_array,
    _latent_mass_return_shift,
)


def test_latent_mass_return_shift_hits_target_mass_sigmoid():
    z0 = np.asarray([-2.5, -0.5, 0.25, 1.5, 3.0], dtype=float)
    weights = np.asarray([0.2, 0.6, 0.9, 0.8, 0.5], dtype=float)
    base_mass = float(weights @ _latent_forward_array(z0, map_kind="sigmoid"))
    target_mass = 0.87 * base_mass

    result = _latent_mass_return_shift(
        z0,
        local_weights=weights,
        target_mass=target_mass,
        map_kind="sigmoid",
        tol_rel=1.0e-12,
        max_it=80,
    )

    assert bool(result["converged"])
    z_new = np.asarray(result["latent_values"], dtype=float)
    mass_new = float(weights @ _latent_forward_array(z_new, map_kind="sigmoid"))
    assert abs(mass_new - target_mass) <= 1.0e-10 * max(abs(target_mass), 1.0)


def test_latent_mass_return_shift_respects_free_mask():
    z0 = np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=float)
    weights = np.asarray([0.4, 0.7, 0.6, 0.3], dtype=float)
    free_mask = np.asarray([False, True, True, False], dtype=bool)
    target_mass = float(weights @ _latent_forward_array(z0, map_kind="algebraic")) * 1.05

    result = _latent_mass_return_shift(
        z0,
        local_weights=weights,
        target_mass=target_mass,
        map_kind="algebraic",
        free_mask=free_mask,
        tol_rel=1.0e-12,
        max_it=80,
    )

    assert bool(result["converged"])
    z_new = np.asarray(result["latent_values"], dtype=float)
    assert z_new[0] == z0[0]
    assert z_new[3] == z0[3]
    mass_new = float(weights @ _latent_forward_array(z_new, map_kind="algebraic"))
    assert abs(mass_new - target_mass) <= 1.0e-10 * max(abs(target_mass), 1.0)
