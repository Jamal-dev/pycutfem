"""Dense Gauss-Newton least-squares kernels for MOR online solvers.

This module provides a backend-neutral reduced least-squares step:

    min_delta ||sqrt(W) (r + J delta)||_2^2 + damping ||D delta||_2^2

The Python backend is the reference implementation.  The C++ backend uses the
same contract and is intended as the dense algebra foundation for the future
fully native online reduced nonlinear driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


GaussNewtonBackend = Literal["python", "cpp", "c++"]
GaussNewtonMethod = Literal["auto", "qr", "svd", "normal"]


@dataclass(frozen=True)
class GaussNewtonNormalEquations:
    """Weighted normal equations for one reduced Gauss-Newton linearization."""

    normal_matrix: np.ndarray
    normal_rhs: np.ndarray
    gradient: np.ndarray
    weighted_residual_norm: float


@dataclass(frozen=True)
class GaussNewtonStepResult:
    """Result of one dense reduced Gauss-Newton step."""

    step: np.ndarray
    rank: int
    method: str
    weighted_residual_norm: float
    linearized_residual_norm: float
    gradient_norm: float
    damping: float
    converged: bool
    condition_estimate: float = float("nan")


def _normalize_backend_name(backend: str | None) -> str:
    name = str(backend or "python").strip().lower()
    if name in {"c++", "cpp"}:
        return "cpp"
    if name != "python":
        raise ValueError(f"Unsupported Gauss-Newton backend {backend!r}.")
    return name


def _normalize_method(method: str | None) -> str:
    name = str(method or "auto").strip().lower()
    if name not in {"auto", "qr", "svd", "normal"}:
        raise ValueError("method must be one of 'auto', 'qr', 'svd', or 'normal'.")
    return name


def _validate_inputs(
    jacobian: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray | None,
    damping: float,
    damping_diagonal: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    J = np.asarray(jacobian, dtype=float)
    r = np.asarray(residual, dtype=float).reshape(-1)
    if J.ndim != 2:
        raise ValueError("jacobian must be a rank-2 array.")
    if int(J.shape[0]) != int(r.size):
        raise ValueError("jacobian row count must match residual size.")
    if not np.all(np.isfinite(J)) or not np.all(np.isfinite(r)):
        raise ValueError("jacobian and residual must contain only finite values.")

    if weights is None:
        w = np.ones(int(r.size), dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if int(w.size) != int(r.size):
            raise ValueError("weights size must match residual size.")
        if np.any(w < 0.0) or not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite and nonnegative.")

    damping_value = float(damping)
    if not np.isfinite(damping_value) or damping_value < 0.0:
        raise ValueError("damping must be finite and nonnegative.")

    n_cols = int(J.shape[1])
    if damping_diagonal is None:
        diag = np.ones(n_cols, dtype=float)
    else:
        diag = np.asarray(damping_diagonal, dtype=float).reshape(-1)
        if int(diag.size) != n_cols:
            raise ValueError("damping_diagonal size must match the number of columns.")
        if np.any(diag < 0.0) or not np.all(np.isfinite(diag)):
            raise ValueError("damping_diagonal must be finite and nonnegative.")

    return J, r, w, damping_value, diag


def _weighted_augmented_system(
    J: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    damping: float,
    damping_diagonal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    sqrt_w = np.sqrt(w)
    A = J * sqrt_w[:, None]
    b = -r * sqrt_w
    weighted_residual_norm = float(np.sqrt(np.dot(w, r * r)))
    if damping > 0.0 and int(J.shape[1]) > 0:
        damp = np.sqrt(damping) * np.diag(damping_diagonal)
        A = np.vstack((A, damp))
        b = np.concatenate((b, np.zeros(int(J.shape[1]), dtype=float)))
    return A, b, weighted_residual_norm


def _condition_estimate(A: np.ndarray) -> float:
    if A.size == 0:
        return float("nan")
    singular_values = np.linalg.svd(A, compute_uv=False)
    if singular_values.size == 0:
        return float("nan")
    smax = float(singular_values[0])
    positive = singular_values[singular_values > 0.0]
    if smax <= 0.0 or positive.size == 0:
        return float("inf")
    return float(smax / positive[-1])


def _normal_rank_threshold(values: np.ndarray, rcond: float | None) -> float:
    if values.size == 0:
        return 0.0
    scale = float(np.max(np.abs(values)))
    if scale <= 0.0:
        return 0.0
    if rcond is None or rcond <= 0.0:
        return float(np.finfo(float).eps * max(values.shape) * scale)
    return float(rcond * scale)


def _solve_python(
    A: np.ndarray,
    b: np.ndarray,
    normal_matrix: np.ndarray,
    normal_rhs: np.ndarray,
    *,
    method: str,
    rcond: float | None,
) -> tuple[np.ndarray, int, str, float]:
    n_cols = int(A.shape[1])
    if n_cols == 0:
        return np.zeros(0, dtype=float), 0, "empty", float("nan")

    lstsq_rcond = None if rcond is None or rcond <= 0.0 else float(rcond)
    if method == "normal":
        try:
            if normal_matrix.size:
                eigvals = np.linalg.eigvalsh(normal_matrix)
                threshold = _normal_rank_threshold(eigvals, lstsq_rcond)
                if np.count_nonzero(eigvals > threshold) < n_cols:
                    raise np.linalg.LinAlgError("rank-deficient normal equations")
            step = np.linalg.solve(normal_matrix, normal_rhs)
            if not np.all(np.isfinite(step)):
                raise np.linalg.LinAlgError("non-finite normal-equation solution")
            return np.asarray(step, dtype=float).reshape(-1), n_cols, "normal_solve", _condition_estimate(A)
        except np.linalg.LinAlgError:
            step, _residuals, rank, _singular_values = np.linalg.lstsq(A, b, rcond=lstsq_rcond)
            return np.asarray(step, dtype=float).reshape(-1), int(rank), "normal_svd_fallback", _condition_estimate(A)

    step, _residuals, rank, _singular_values = np.linalg.lstsq(A, b, rcond=lstsq_rcond)
    method_used = "svd" if method == "svd" else method
    if method == "auto":
        method_used = "auto_lstsq"
    if method == "qr":
        method_used = "qr_lstsq"
    return np.asarray(step, dtype=float).reshape(-1), int(rank), method_used, _condition_estimate(A)


def _normal_equations_from_arrays(
    J: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    damping: float,
    damping_diagonal: np.ndarray,
) -> GaussNewtonNormalEquations:
    A, b, weighted_residual_norm = _weighted_augmented_system(J, r, w, damping, damping_diagonal)
    normal_matrix = A.T @ A
    normal_rhs = A.T @ b
    gradient = -normal_rhs
    return GaussNewtonNormalEquations(
        normal_matrix=np.asarray(normal_matrix, dtype=float),
        normal_rhs=np.asarray(normal_rhs, dtype=float).reshape(-1),
        gradient=np.asarray(gradient, dtype=float).reshape(-1),
        weighted_residual_norm=weighted_residual_norm,
    )


def _normal_equations_from_raw(raw: dict) -> GaussNewtonNormalEquations:
    return GaussNewtonNormalEquations(
        normal_matrix=np.asarray(raw["normal_matrix"], dtype=float),
        normal_rhs=np.asarray(raw["normal_rhs"], dtype=float).reshape(-1),
        gradient=np.asarray(raw["gradient"], dtype=float).reshape(-1),
        weighted_residual_norm=float(raw["weighted_residual_norm"]),
    )


def _step_result_from_raw(raw: dict) -> GaussNewtonStepResult:
    return GaussNewtonStepResult(
        step=np.asarray(raw["step"], dtype=float).reshape(-1),
        rank=int(raw["rank"]),
        method=str(raw["method"]),
        weighted_residual_norm=float(raw["weighted_residual_norm"]),
        linearized_residual_norm=float(raw["linearized_residual_norm"]),
        gradient_norm=float(raw["gradient_norm"]),
        damping=float(raw["damping"]),
        converged=bool(raw["converged"]),
        condition_estimate=float(raw["condition_estimate"]),
    )


def form_normal_equations(
    jacobian: np.ndarray,
    residual: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    damping: float = 0.0,
    damping_diagonal: np.ndarray | None = None,
    backend: GaussNewtonBackend = "python",
) -> GaussNewtonNormalEquations:
    """Build weighted normal equations for one reduced linearized residual.

    The returned system solves ``normal_matrix @ step = normal_rhs`` where
    ``normal_rhs = -J.T @ W @ residual`` plus optional Levenberg-Marquardt
    damping on the left-hand side.
    """

    backend_name = _normalize_backend_name(backend)
    J, r, w, damping_value, diag = _validate_inputs(jacobian, residual, weights, damping, damping_diagonal)
    if backend_name == "cpp":
        from .cpp_backend.gauss_newton import module as _gauss_newton_cpp_module

        raw = _gauss_newton_cpp_module().form_normal_equations(J, r, w, damping_value, diag)
        return _normal_equations_from_raw(raw)
    return _normal_equations_from_arrays(J, r, w, damping_value, diag)


def gauss_newton_step(
    jacobian: np.ndarray,
    residual: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    damping: float = 0.0,
    damping_diagonal: np.ndarray | None = None,
    method: GaussNewtonMethod = "auto",
    rcond: float | None = None,
    backend: GaussNewtonBackend = "python",
) -> GaussNewtonStepResult:
    """Compute one dense reduced Gauss-Newton step.

    Parameters follow the objective documented in this module.  ``backend="cpp"``
    dispatches to the compiled Eigen implementation and keeps the same return
    object as the Python reference path.
    """

    backend_name = _normalize_backend_name(backend)
    method_name = _normalize_method(method)
    J, r, w, damping_value, diag = _validate_inputs(jacobian, residual, weights, damping, damping_diagonal)
    rcond_value = -1.0 if rcond is None else float(rcond)
    if not np.isfinite(rcond_value):
        raise ValueError("rcond must be finite when provided.")

    if backend_name == "cpp":
        from .cpp_backend.gauss_newton import module as _gauss_newton_cpp_module

        raw = _gauss_newton_cpp_module().gauss_newton_step(J, r, w, damping_value, diag, method_name, rcond_value)
        return _step_result_from_raw(raw)

    normal = _normal_equations_from_arrays(J, r, w, damping_value, diag)
    A, b, _weighted_residual_norm = _weighted_augmented_system(J, r, w, damping_value, diag)
    step, rank, method_used, condition_estimate = _solve_python(
        A,
        b,
        normal.normal_matrix,
        normal.normal_rhs,
        method=method_name,
        rcond=None if rcond is None else rcond_value,
    )
    linearized_residual_norm = float(np.linalg.norm(A @ step - b))
    return GaussNewtonStepResult(
        step=step,
        rank=rank,
        method=method_used,
        weighted_residual_norm=normal.weighted_residual_norm,
        linearized_residual_norm=linearized_residual_norm,
        gradient_norm=float(np.linalg.norm(normal.gradient)),
        damping=damping_value,
        converged=bool(np.all(np.isfinite(step))),
        condition_estimate=condition_estimate,
    )


__all__ = [
    "GaussNewtonBackend",
    "GaussNewtonMethod",
    "GaussNewtonNormalEquations",
    "GaussNewtonStepResult",
    "form_normal_equations",
    "gauss_newton_step",
]
