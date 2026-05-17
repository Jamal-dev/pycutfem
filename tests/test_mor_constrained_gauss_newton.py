from __future__ import annotations

import numpy as np
import pytest

from pycutfem.mor import (
    BoundConstraintSpec,
    EqualityConstrainedGaussNewtonStepResult,
    equality_constrained_gauss_newton_step,
    gauss_newton_step,
)


def _have_cpp_backend() -> bool:
    try:
        import pybind11  # noqa: F401

        return True
    except Exception:
        return False


BACKENDS = ["python"] + (["cpp"] if _have_cpp_backend() else [])


def _augmented_system(
    J: np.ndarray,
    r: np.ndarray,
    weights: np.ndarray | None = None,
    damping: float = 0.0,
    damping_diagonal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    w = np.ones(r.size, dtype=float) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    diag = np.ones(J.shape[1], dtype=float) if damping_diagonal is None else np.asarray(damping_diagonal, dtype=float)
    sqrt_w = np.sqrt(w)
    A = J * sqrt_w[:, None]
    b = -r * sqrt_w
    if damping > 0.0 and J.shape[1] > 0:
        A = np.vstack((A, np.sqrt(damping) * np.diag(diag)))
        b = np.concatenate((b, np.zeros(J.shape[1], dtype=float)))
    return A, b


def _kkt_reference(A: np.ndarray, b: np.ndarray, C: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H = A.T @ A
    rhs = A.T @ b
    K = np.block(
        [
            [H, C.T],
            [C, np.zeros((C.shape[0], C.shape[0]), dtype=float)],
        ]
    )
    sol = np.linalg.solve(K, np.concatenate((rhs, h)))
    return sol[: A.shape[1]], sol[A.shape[1] :]


@pytest.mark.parametrize("backend", BACKENDS)
def test_equality_constrained_step_matches_kkt_reference(backend: str, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"mor_constrained_gn_{backend}"))
    J = np.array(
        [
            [2.0, -1.0, 0.5],
            [0.5, 3.0, -0.25],
            [-1.0, 0.25, 2.0],
            [4.0, 1.5, 0.75],
        ],
        dtype=float,
    )
    r = np.array([1.0, -2.0, 0.5, 3.0], dtype=float)
    weights = np.array([1.0, 0.5, 2.0, 0.75], dtype=float)
    damping = 0.2
    damping_diagonal = np.array([1.0, 0.5, 2.0], dtype=float)
    C = np.array([[1.0, -0.5, 0.25]], dtype=float)
    h = np.array([0.1], dtype=float)
    A, b = _augmented_system(J, r, weights, damping, damping_diagonal)
    expected_step, _expected_multipliers = _kkt_reference(A, b, C, h)

    result = equality_constrained_gauss_newton_step(
        J,
        r,
        constraint_matrix=C,
        constraint_rhs=h,
        weights=weights,
        damping=damping,
        damping_diagonal=damping_diagonal,
        method="kkt",
        backend=backend,
    )

    assert isinstance(result, EqualityConstrainedGaussNewtonStepResult)
    np.testing.assert_allclose(result.step, expected_step, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(C @ result.step, h, rtol=1.0e-12, atol=1.0e-12)
    assert result.constraint_rank == 1
    assert result.converged


@pytest.mark.parametrize("backend", BACKENDS)
def test_equality_constrained_step_handles_dependent_active_rows(backend: str, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"mor_constrained_dependent_{backend}"))
    J = np.array([[1.0, 0.5], [0.25, 2.0], [-1.0, 1.0]], dtype=float)
    r = np.array([0.5, -0.25, 1.5], dtype=float)
    C = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=float)
    h = np.array([0.2, 0.4], dtype=float)

    result = equality_constrained_gauss_newton_step(
        J,
        r,
        constraint_matrix=C,
        constraint_rhs=h,
        method="nullspace",
        backend=backend,
        rcond=1.0e-12,
    )

    assert result.constraint_rank == 1
    np.testing.assert_allclose(C @ result.step, h, rtol=1.0e-11, atol=1.0e-11)
    assert result.converged


@pytest.mark.parametrize("backend", BACKENDS)
def test_equality_constrained_empty_constraints_match_unconstrained(backend: str, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"mor_constrained_empty_{backend}"))
    J = np.array([[1.0, -0.5], [2.0, 1.0], [0.25, 3.0]], dtype=float)
    r = np.array([1.0, -2.0, 0.25], dtype=float)

    constrained = equality_constrained_gauss_newton_step(J, r, backend=backend)
    unconstrained = gauss_newton_step(J, r, backend=backend)

    np.testing.assert_allclose(constrained.step, unconstrained.step, rtol=1.0e-12, atol=1.0e-12)
    assert constrained.constraint_rank == 0
    assert constrained.constraint_violation_norm == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_active_bound_equations_feed_constrained_step(backend: str, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"mor_constrained_bounds_{backend}"))
    J = np.array([[1.0, 0.0], [0.5, 2.0], [-1.0, 1.0]], dtype=float)
    r = np.array([0.25, -1.0, 0.5], dtype=float)
    trial_basis = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=float)
    offset = np.zeros(3, dtype=float)
    q = np.array([-0.1, 0.2], dtype=float)
    bounds = BoundConstraintSpec(rows=np.array([0, 1]), lower=np.array([0.0, -1.0]), upper=np.array([1.0, 1.0]))
    active = bounds.reduce(trial_basis=trial_basis, offset=offset).active_equations(q)

    result = equality_constrained_gauss_newton_step(
        J,
        r,
        constraint_matrix=active.constraint_matrix,
        constraint_rhs=active.rhs,
        backend=backend,
    )

    q_trial = q + result.step
    decoded = offset + trial_basis @ q_trial
    np.testing.assert_allclose(decoded[0], 0.0, atol=1.0e-12)
    assert result.converged


@pytest.mark.parametrize("backend", BACKENDS)
def test_equality_constrained_step_rejects_inconsistent_constraints(backend: str, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"mor_constrained_bad_{backend}"))
    J = np.eye(2, dtype=float)
    r = np.array([1.0, 2.0], dtype=float)
    C = np.array([[0.0, 0.0]], dtype=float)
    h = np.array([1.0], dtype=float)

    with pytest.raises(ValueError, match="inconsistent"):
        equality_constrained_gauss_newton_step(
            J,
            r,
            constraint_matrix=C,
            constraint_rhs=h,
            backend=backend,
        )
