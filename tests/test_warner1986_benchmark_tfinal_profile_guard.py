from __future__ import annotations


def test_warner1986_benchmark_does_not_request_t6_profiles_when_tfinal_lt_6() -> None:
    # Regression test: the benchmark script used to unconditionally request a
    # t=6d profile plot for case 1, which crashes if the simulation horizon is
    # shorter (e.g. t_final=0.5d).
    from examples.biofilms.benchmarks.warner import warner1986_benchmark as wb

    assert wb._needs_t6_profiles(case_id=1, t_final=0.5) is False
    assert wb._needs_t6_profiles(case_id=4, t_final=0.5) is False
    assert wb._needs_t6_profiles(case_id=1, t_final=6.0) is True
    assert wb._needs_t6_profiles(case_id=4, t_final=10.0) is True


def test_warner1986_case4_slough_window_is_not_skipped() -> None:
    # Regression test: without segmenting the integration, a stiff BDF step may
    # jump over the short sloughing window and miss the thickness drop entirely.
    import numpy as np

    from examples.biofilms.benchmarks.warner import warner1986_benchmark as wb

    params = wb.Warner1986Params()
    rhs = wb._rhs_python(4, params)

    t_eval = np.asarray([0.0, params.slough_t0, params.slough_t1, 6.0], dtype=float)
    sol = wb._solve_case(case_id=4, params=params, rhs=rhs, t_final=6.0, t_eval=t_eval, rtol=1.0e-8, atol=1.0e-10)

    # Extract thickness at window endpoints.
    t = np.asarray(sol.t, dtype=float)
    y = np.asarray(sol.y, dtype=float)
    idx0 = int(np.where(np.isclose(t, float(params.slough_t0)))[0][0])
    idx1 = int(np.where(np.isclose(t, float(params.slough_t1)))[0][0])

    L0 = float(y[0, idx0])
    L1 = float(y[0, idx1])
    dL_um = 1.0e6 * (L1 - L0)

    assert dL_um < -450.0
    assert dL_um > -550.0
