from __future__ import annotations

import numpy as np

from examples.biofilms.benchmarks.dadu.paper1_benchmark3_duddu2007_growth import (
    _interp_rectilinear,
    _interp_time_series,
    _mask_metrics,
)


def test_interp_time_series_is_linear_between_snapshots() -> None:
    t = np.asarray([0.0, 1.0], dtype=float)
    values = np.asarray([[0.0, 2.0], [2.0, 4.0]], dtype=float)
    got = _interp_time_series(t, values, 0.25)
    assert np.allclose(got, np.asarray([0.5, 2.5], dtype=float))


def test_interp_rectilinear_preserves_constant_field() -> None:
    xs_src = np.asarray([0.0, 1.0], dtype=float)
    ys_src = np.asarray([0.0, 1.0], dtype=float)
    field = np.full((2, 2), 3.5, dtype=float)
    xs_dst = np.linspace(0.1, 0.9, 4)
    ys_dst = np.linspace(0.2, 0.8, 3)
    got = _interp_rectilinear(xs_src, ys_src, field, xs_dst, ys_dst)
    assert got.shape == (3, 4)
    assert np.allclose(got, 3.5)


def test_interp_rectilinear_preserves_bilinear_field() -> None:
    xs_src = np.asarray([0.0, 0.5, 1.0], dtype=float)
    ys_src = np.asarray([0.0, 1.0, 2.0], dtype=float)
    xx, yy = np.meshgrid(xs_src, ys_src)
    field = 2.0 * xx - 0.5 * yy + 1.25
    xs_dst = np.asarray([0.1, 0.4, 0.7, 0.9], dtype=float)
    ys_dst = np.asarray([0.2, 0.8, 1.6], dtype=float)
    got = _interp_rectilinear(xs_src, ys_src, field, xs_dst, ys_dst)
    xx_dst, yy_dst = np.meshgrid(xs_dst, ys_dst)
    expect = 2.0 * xx_dst - 0.5 * yy_dst + 1.25
    assert np.allclose(got, expect)


def test_mask_metrics_vanish_for_identical_masks() -> None:
    xs = np.asarray([0.25, 0.75], dtype=float)
    ys = np.asarray([0.25, 0.75], dtype=float)
    mask = np.asarray([[True, False], [True, True]], dtype=bool)
    metrics = _mask_metrics(mask, mask.copy(), xs, ys)
    assert metrics["area_abs_err_mm2"] == 0.0
    assert metrics["shape_mismatch"] == 0.0
    assert metrics["centroid_err_mm"] == 0.0
    assert metrics["profile_mae_mm"] == 0.0
