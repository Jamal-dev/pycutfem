import numpy as np

from examples.dg_solid_body_rotation import solve_solid_body_rotation, convergence_study


def test_dg_solid_body_rotation_backends_match():
    settings = dict(
        nx=4,
        ny=4,
        poly_order=1,
        cfl=0.4,
        quad_order=4,
    )
    errs = {
        backend: solve_solid_body_rotation(backend=backend, **settings)
        for backend in ("python", "jit", "cpp")
    }
    assert all(np.isfinite(val) for val in errs.values())
    assert all(val < 0.25 for val in errs.values())

    err_py = errs["python"]
    assert abs(err_py - errs["jit"]) < 1.0e-6
    assert abs(err_py - errs["cpp"]) < 1.0e-6


def test_dg_solid_body_rotation_convergence():
    result = convergence_study(
        base_n=4,
        levels=2,
        poly_order=1,
        cfl=0.4,
        quad_order=4,
        backend="python",
    )
    errors = result["errors"]
    ooa = result["ooa"]
    assert len(errors) == 2
    assert len(ooa) == 1
    assert all(np.isfinite(val) for val in errors)
    assert all(np.isfinite(val) for val in ooa)
    assert errors[0] < 0.25
    assert errors[1] < errors[0] * 0.8
    assert ooa[0] > 0.5
