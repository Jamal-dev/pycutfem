import numpy as np

from examples.advection_diffusion_supg import solve_advection_diffusion_supg


def test_advection_diffusion_supg():
    err = solve_advection_diffusion_supg(
        nx=4,
        ny=4,
        poly_order=1,
        epsilon=1.0e-2,
        beta_vec=(1.0, 0.5),
        quad_order=4,
        backend="python",
    )
    assert err < 0.25


def test_advection_diffusion_supg_dg_backends():
    settings = dict(
        nx=4,
        ny=4,
        poly_order=1,
        epsilon=1.0e-2,
        beta_vec=(1.0, 0.5),
        quad_order=4,
        method="dg",
    )
    err_py = solve_advection_diffusion_supg(backend="python", **settings)
    err_jit = solve_advection_diffusion_supg(backend="jit", **settings)
    err_cpp = solve_advection_diffusion_supg(backend="cpp", **settings)

    assert np.isfinite(err_py)
    assert np.isfinite(err_jit)
    assert np.isfinite(err_cpp)
    assert err_py < 1.5
    assert abs(err_py - err_jit) < 1.0e-6
    assert abs(err_py - err_cpp) < 1.0e-6
