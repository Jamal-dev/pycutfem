import math

import numpy as np

from pycutfem.core.levelset import (
    AffineLevelSet,
    MaxLevelSet,
    MinLevelSet,
    RotatedBoxLevelSet,
    ScaledLevelSet,
)


def test_rotated_box_value_and_gradient_axis_aligned():
    ls = RotatedBoxLevelSet(center=(0.0, 0.0), hx=1.0, hy=2.0, angle=0.0)

    assert np.isclose(ls(np.array([0.0, 0.0])), -1.0)
    assert np.isclose(ls(np.array([1.0, 0.0])), 0.0)
    assert np.isclose(ls(np.array([2.0, 0.0])), 1.0)

    g_right = ls.gradient(np.array([0.9, 0.0]))
    assert np.allclose(g_right, np.array([1.0, 0.0]), atol=1.0e-12)

    g_top = ls.gradient(np.array([0.0, 1.9]))
    assert np.allclose(g_top, np.array([0.0, 1.0]), atol=1.0e-12)


def test_rotated_box_gradient_rotates():
    ls = RotatedBoxLevelSet(center=(0.0, 0.0), hx=1.0, hy=2.0, angle=math.pi / 2.0)
    # Local x' aligns with global +y when angle=pi/2, so the "right face" normal is +y.
    g = ls.gradient(np.array([0.0, 0.9]))
    assert np.allclose(g, np.array([0.0, 1.0]), atol=1.0e-12)


def test_scaled_levelset_flips_gradient_on_negative_scale():
    base = AffineLevelSet(a=1.0, b=0.0, c=0.0).normalised()
    ls_pos = ScaledLevelSet(2.0, base)
    ls_neg = ScaledLevelSet(-1.0, base)

    x = np.array([0.25, -0.1])
    assert np.isclose(ls_pos(x), 2.0 * base(x))
    assert np.isclose(ls_neg(x), -1.0 * base(x))

    g0 = base.gradient(x)
    assert np.allclose(ls_pos.gradient(x), g0, atol=1.0e-12)
    assert np.allclose(ls_neg.gradient(x), -g0, atol=1.0e-12)


def test_min_max_levelset_selects_values_and_gradients():
    l1 = AffineLevelSet(a=1.0, b=0.0, c=0.0).normalised()  # phi = x
    l2 = AffineLevelSet(a=-1.0, b=0.0, c=0.0).normalised()  # phi = -x
    mn = MinLevelSet(l1, l2)
    mx = MaxLevelSet(l1, l2)

    x = np.array([0.3, 0.0])
    assert np.isclose(mn(x), min(float(l1(x)), float(l2(x))))
    assert np.isclose(mx(x), max(float(l1(x)), float(l2(x))))

    # For x>0: l2(x)=-x is the min, l1(x)=+x is the max.
    assert np.allclose(mn.gradient(x), l2.gradient(x), atol=1.0e-12)
    assert np.allclose(mx.gradient(x), l1.gradient(x), atol=1.0e-12)

