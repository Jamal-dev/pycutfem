from __future__ import annotations

from pycutfem.solvers.dt_controller import AdaptiveDtController, DtControllerParams


def test_dt_controller_keeps_dt_in_normal_range() -> None:
    ctrl = AdaptiveDtController(dt0=0.01, params=DtControllerParams())
    dec = ctrl.on_success(dt=0.01, n_iters=10)
    assert dec.dt == 0.01
    assert dec.retry_step is False
    assert dec.reason == "keep"


def test_dt_controller_decreases_dt_on_slow_newton() -> None:
    ctrl = AdaptiveDtController(dt0=0.01, params=DtControllerParams())
    dec = ctrl.on_success(dt=0.01, n_iters=21)
    assert dec.dt == 0.01 * 0.9
    assert dec.retry_step is False
    assert dec.reason == "slow_newton"


def test_dt_controller_increases_dt_after_consecutive_easy_steps() -> None:
    ctrl = AdaptiveDtController(dt0=0.01, params=DtControllerParams())

    # First force a decrease.
    dec = ctrl.on_success(dt=0.01, n_iters=21)
    dt = dec.dt

    # Two easy steps trigger an increase (capped by dt_max=dt0).
    dec1 = ctrl.on_success(dt=dt, n_iters=5)
    assert dec1.dt == dt
    assert dec1.reason == "keep"

    dec2 = ctrl.on_success(dt=dt, n_iters=5)
    assert dec2.dt > dt
    assert dec2.dt <= 0.01
    assert dec2.reason == "fast_newton"


def test_dt_controller_reduces_dt_on_failure_and_retries() -> None:
    ctrl = AdaptiveDtController(dt0=0.01, params=DtControllerParams())
    dec = ctrl.on_failure(dt=0.01, reason="line_search")
    assert dec.dt == 0.005
    assert dec.retry_step is True
    assert dec.reason == "newton_failed:line_search"


def test_dt_controller_respects_dt_min_on_failure() -> None:
    ctrl = AdaptiveDtController(
        dt0=0.01,
        params=DtControllerParams(dt_min=0.004),
    )
    dec = ctrl.on_failure(dt=0.005, reason="max_iter")
    assert dec.dt == 0.004
    assert dec.retry_step is True


def test_dt_controller_can_reject_on_slow_convergence() -> None:
    ctrl = AdaptiveDtController(
        dt0=0.01,
        params=DtControllerParams(reject_on_slow=True),
    )
    dec = ctrl.on_success(dt=0.01, n_iters=21)
    assert dec.reason == "slow_newton"
    assert dec.retry_step is True

