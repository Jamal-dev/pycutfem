import numpy as np

from examples.debug.arctan_plateau_benchmark import (
    anderson_solve,
    fixed_point_solve,
    newton_solve,
    ptc_solve,
)


def test_arctan_plateau_newton_diverges_from_plateau_start() -> None:
    report = newton_solve(np.array([3.0, 3.0], dtype=float), max_iter=4)

    assert not report.converged
    assert report.residual_inf > 1.0
    assert max(abs(v) for v in report.state) > 1.0e6


def test_arctan_plateau_ptc_requires_aligned_mass_operator() -> None:
    identity_report = ptc_solve(
        np.array([3.0, 3.0], dtype=float),
        mass="identity",
        sign=1.0,
        max_iter=30,
    )
    root_report = ptc_solve(
        np.array([3.0, 3.0], dtype=float),
        mass="root",
        sign=1.0,
        max_iter=30,
    )

    assert not identity_report.converged
    assert identity_report.residual_inf > 1.0e-1
    assert root_report.converged
    assert root_report.residual_inf < 1.0e-8


def test_arctan_plateau_wrong_sign_ptc_breaks_the_continuation() -> None:
    report = ptc_solve(
        np.array([3.0, 3.0], dtype=float),
        mass="root",
        sign=-1.0,
        max_iter=10,
    )

    assert not report.converged
    assert "failed" in report.note.lower()


def test_arctan_plateau_anderson_state_mixing_beats_plain_fixed_point() -> None:
    base = fixed_point_solve(np.array([3.0, 3.0], dtype=float), omega=0.5, max_iter=50)
    aa_state = anderson_solve(
        np.array([3.0, 3.0], dtype=float),
        omega=0.5,
        mode="state",
        max_iter=50,
    )

    assert base.converged
    assert aa_state.converged
    assert aa_state.iterations < base.iterations
    assert aa_state.residual_inf < 1.0e-8


def test_arctan_plateau_increment_mixing_is_not_a_safe_substitute() -> None:
    aa_state = anderson_solve(
        np.array([3.0, 3.0], dtype=float),
        omega=0.5,
        mode="state",
        max_iter=40,
    )
    aa_increment = anderson_solve(
        np.array([3.0, 3.0], dtype=float),
        omega=0.5,
        mode="increment",
        max_iter=40,
    )

    assert aa_state.converged
    assert not aa_increment.converged
    assert aa_increment.residual_inf > 1.0e-2
