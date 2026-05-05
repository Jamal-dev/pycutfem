from __future__ import annotations

from examples.biofilms.benchmarks.blauert.blauert_biofilm_deformation_one_domain import _cosine_ramp_value


def test_cosine_ramp_value_basic_behavior() -> None:
    assert _cosine_ramp_value(0.0, 2.0) == 0.0
    assert _cosine_ramp_value(-1.0, 2.0) == 0.0
    assert abs(_cosine_ramp_value(1.0, 2.0) - 0.5) < 1.0e-12
    assert _cosine_ramp_value(2.0, 2.0) == 1.0
    assert _cosine_ramp_value(3.0, 2.0) == 1.0


def test_cosine_ramp_value_degenerate_ramp_is_identity() -> None:
    assert _cosine_ramp_value(0.0, 0.0) == 1.0
    assert _cosine_ramp_value(1.0, -1.0) == 1.0
