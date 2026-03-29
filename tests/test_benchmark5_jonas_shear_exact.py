from __future__ import annotations

import math

import numpy as np

from examples.utils.biofilm.benchmark5_jonas_shear_exact import build_jonas_shear_benchmark

_TRAPEZOID = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def test_benchmark5_jonas_shear_exact_properties() -> None:
    bench = build_jonas_shear_benchmark()

    x_mid = 0.5
    y_mid = float(bench.interface_y)

    alpha_mid = float(np.asarray(bench.alpha(x_mid, y_mid), dtype=float).reshape(()))
    B_mid = float(np.asarray(bench.B(x_mid, y_mid), dtype=float).reshape(()))
    mu_mid = float(np.asarray(bench.mu_alpha(x_mid, y_mid), dtype=float).reshape(()))
    g_t = np.asarray(bench.g_t(x_mid, y_mid), dtype=float).reshape(2)

    assert math.isclose(alpha_mid, 0.5, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(B_mid, 0.25, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(mu_mid, 0.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert g_t.shape == (2,)
    assert math.isclose(g_t[0], -0.2, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(g_t[1], 0.0, rel_tol=0.0, abs_tol=1.0e-12)

    y = np.linspace(0.0, 1.0, 4001, dtype=float)
    x = np.full_like(y, x_mid)
    weight = np.asarray(bench.traction_weight(x, y), dtype=float)
    integral = float(_TRAPEZOID(weight, y))
    assert 0.9999 <= integral <= 1.0001

    for yy in (0.15, 0.5, 0.85):
        f_alpha = float(np.asarray(bench.f_alpha(x_mid, yy), dtype=float).reshape(()))
        f_B = float(np.asarray(bench.f_B(x_mid, yy), dtype=float).reshape(()))
        assert math.isclose(f_alpha, 0.0, rel_tol=0.0, abs_tol=1.0e-12)
        assert math.isclose(f_B, 0.0, rel_tol=0.0, abs_tol=1.0e-12)
