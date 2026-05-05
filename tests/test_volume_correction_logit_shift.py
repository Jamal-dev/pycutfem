import numpy as np

from examples.utils.shared.volume_correction import logit_shift_to_match_integral


def test_logit_shift_matches_target_integral_and_keeps_bounds():
    rng = np.random.default_rng(0)
    n = 200
    # Diffuse indicator-like values (cluster near 0/1 but not exactly).
    a = rng.beta(0.3, 0.3, size=n)
    w = rng.random(size=n) + 1.0e-3
    m0 = float(np.dot(w, a))

    # Increase mass by 20% but keep it below the max attainable (sum(weights)).
    target = min(0.95 * float(np.dot(w, np.ones_like(w))), 1.2 * m0)
    res = logit_shift_to_match_integral(a, weights=w, target_mass=target)

    assert np.all(np.isfinite(res.values))
    assert float(res.values.min()) >= 0.0
    assert float(res.values.max()) <= 1.0
    assert abs(res.mass - target) <= 5.0e-12 * max(1.0, abs(target))
    assert res.shift > 0.0


def test_logit_shift_handles_extreme_targets():
    rng = np.random.default_rng(1)
    n = 50
    a = rng.random(size=n)
    w = rng.random(size=n) + 1.0e-6
    total = float(np.dot(w, np.ones_like(w)))

    res0 = logit_shift_to_match_integral(a, weights=w, target_mass=0.0)
    assert np.allclose(res0.values, 0.0)
    assert res0.mass == 0.0

    res1 = logit_shift_to_match_integral(a, weights=w, target_mass=total)
    assert np.allclose(res1.values, 1.0)
    assert abs(res1.mass - total) <= 1.0e-14 * max(1.0, abs(total))


def test_logit_shift_is_noop_when_target_matches_current_mass():
    rng = np.random.default_rng(2)
    n = 80
    a = rng.random(size=n)
    w = rng.random(size=n) + 1.0e-3
    target = float(np.dot(w, np.clip(a, 0.0, 1.0)))

    res = logit_shift_to_match_integral(a, weights=w, target_mass=target)
    assert res.iterations == 0
    assert res.shift == 0.0
    assert abs(res.mass - target) <= 1.0e-14 * max(1.0, abs(target))

