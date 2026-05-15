from __future__ import annotations

import numpy as np
import pytest

from pycutfem.mor.gauss_newton import (
    GaussNewtonNormalEquations,
    GaussNewtonStepResult,
    form_normal_equations,
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_gauss_newton_step_matches_dense_least_squares(backend: str, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"mor_gauss_newton_{backend}"))
    J = np.array(
        [
            [2.0, -1.0],
            [0.5, 3.0],
            [-1.0, 0.25],
            [4.0, 1.5],
        ],
        dtype=float,
    )
    r = np.array([1.0, -2.0, 0.5, 3.0], dtype=float)

    expected, *_ = np.linalg.lstsq(J, -r, rcond=None)
    result = gauss_newton_step(J, r, backend=backend, method="auto")

    assert isinstance(result, GaussNewtonStepResult)
    np.testing.assert_allclose(result.step, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(result.weighted_residual_norm, np.linalg.norm(r), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        result.linearized_residual_norm,
        np.linalg.norm(J @ expected + r),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert result.rank == J.shape[1]
    assert result.converged


@pytest.mark.parametrize("backend", BACKENDS)
def test_weighted_damped_normal_equations_match_reference(backend: str, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"mor_gauss_newton_normal_{backend}"))
    J = np.array(
        [
            [1.0, 0.25, -0.5],
            [2.0, -1.0, 0.75],
            [-0.5, 3.0, 1.0],
            [0.25, 1.5, -2.0],
        ],
        dtype=float,
    )
    r = np.array([0.5, -1.5, 2.0, -0.25], dtype=float)
    weights = np.array([0.25, 2.0, 1.5, 0.75], dtype=float)
    damping = 0.125
    damping_diagonal = np.array([1.0, 0.5, 2.0], dtype=float)

    normal = form_normal_equations(
        J,
        r,
        weights=weights,
        damping=damping,
        damping_diagonal=damping_diagonal,
        backend=backend,
    )

    expected_matrix = J.T @ (weights[:, None] * J) + damping * np.diag(damping_diagonal**2)
    expected_rhs = -(J.T @ (weights * r))
    expected_gradient = -expected_rhs

    assert isinstance(normal, GaussNewtonNormalEquations)
    np.testing.assert_allclose(normal.normal_matrix, expected_matrix, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(normal.normal_rhs, expected_rhs, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(normal.gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        normal.weighted_residual_norm,
        np.sqrt(np.dot(weights, r * r)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_weighted_damped_step_matches_augmented_lstsq(backend: str, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"mor_gauss_newton_lm_{backend}"))
    J = np.array(
        [
            [1.5, -0.25],
            [-0.75, 2.0],
            [3.0, 1.0],
            [0.5, -1.5],
        ],
        dtype=float,
    )
    r = np.array([2.0, -1.0, 0.25, 1.5], dtype=float)
    weights = np.array([1.0, 0.5, 4.0, 0.25], dtype=float)
    damping = 0.2
    damping_diagonal = np.array([1.0, 3.0], dtype=float)
    A, b = _augmented_system(J, r, weights, damping, damping_diagonal)

    expected, *_ = np.linalg.lstsq(A, b, rcond=None)
    result = gauss_newton_step(
        J,
        r,
        weights=weights,
        damping=damping,
        damping_diagonal=damping_diagonal,
        backend=backend,
        method="svd",
    )

    np.testing.assert_allclose(result.step, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(result.linearized_residual_norm, np.linalg.norm(A @ expected - b), rtol=1.0e-12)
    assert result.rank == J.shape[1]


@pytest.mark.parametrize("backend", BACKENDS)
def test_normal_method_falls_back_for_rank_deficient_system(backend: str, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"mor_gauss_newton_rank_{backend}"))
    J = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ],
        dtype=float,
    )
    r = np.array([-1.0, 0.5, 2.0], dtype=float)

    expected, *_ = np.linalg.lstsq(J, -r, rcond=1.0e-12)
    result = gauss_newton_step(J, r, backend=backend, method="normal", rcond=1.0e-12)

    np.testing.assert_allclose(result.step, expected, rtol=1.0e-12, atol=1.0e-12)
    assert result.rank == 1
    assert result.method == "normal_svd_fallback"
    assert result.converged


def test_gauss_newton_input_validation() -> None:
    J = np.ones((2, 2), dtype=float)
    r = np.ones(2, dtype=float)

    with pytest.raises(ValueError, match="row count"):
        gauss_newton_step(J, np.ones(3), backend="python")
    with pytest.raises(ValueError, match="nonnegative"):
        gauss_newton_step(J, r, weights=np.array([1.0, -1.0]), backend="python")
    with pytest.raises(ValueError, match="damping must"):
        gauss_newton_step(J, r, damping=-1.0, backend="python")
    with pytest.raises(ValueError, match="damping_diagonal size"):
        gauss_newton_step(J, r, damping_diagonal=np.ones(3), backend="python")
    with pytest.raises(ValueError, match="method must"):
        gauss_newton_step(J, r, method="bad", backend="python")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported"):
        gauss_newton_step(J, r, backend="jit")  # type: ignore[arg-type]
