import numpy as np
import pytest

from examples.navier_stokes_stabilized import solve_oseen, convergence_study


@pytest.mark.parametrize("stabilization", ["none", "grad_div", "pressure_stab"])
def test_navier_stokes_stabilizations(stabilization):
    u_err, p_err = solve_oseen(
        stabilization=stabilization,
        nx=2,
        ny=2,
        nu=1.0e-2,
        quad_order=4,
        backend="python",
    )
    assert np.isfinite(u_err)
    assert np.isfinite(p_err)
    assert u_err < 1.0


def test_navier_stokes_dg_pressure():
    u_err, p_err = solve_oseen(
        stabilization="pressure_stab",
        pressure_method="dg",
        nx=2,
        ny=2,
        nu=1.0e-2,
        quad_order=4,
        backend="python",
    )
    assert np.isfinite(u_err)
    assert np.isfinite(p_err)
    assert u_err < 1.0
    assert p_err < 0.2


@pytest.mark.parametrize("diffusion_flux", ["symmetric", "nonsymmetric"])
def test_navier_stokes_dg_velocity_flux(diffusion_flux):
    u_err, p_err = solve_oseen(
        stabilization="grad_div",
        velocity_method="dg",
        pressure_method="cg",
        diffusion_flux=diffusion_flux,
        nx=2,
        ny=2,
        nu=1.0e-2,
        quad_order=4,
        backend="python",
    )
    assert np.isfinite(u_err)
    assert np.isfinite(p_err)
    assert u_err < 1.0


def test_navier_stokes_convergence_study():
    result = convergence_study(
        base_n=2,
        levels=2,
        stabilization="grad_div",
        nu=1.0e-2,
        quad_order=4,
        backend="python",
    )
    assert len(result["ns"]) == 2
    assert all(np.isfinite(val) for val in result["u_errs"])
    assert all(np.isfinite(val) for val in result["p_errs"])
