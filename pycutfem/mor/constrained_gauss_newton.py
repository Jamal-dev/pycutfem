"""Equality-constrained reduced Gauss-Newton step kernels.

This module is the algebraic foundation for native PDAS/IPM reduced solves.  A
PDAS iteration turns active decoded bounds into linear equalities in the
Gauss-Newton increment,

    C step = h,

and then solves the constrained least-squares subproblem.  The implementation
is backend-neutral: the Python path is the reference implementation and the C++
path uses the native Eigen backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .gauss_newton import (
    _condition_estimate,
    _normalize_backend_name,
    _validate_inputs,
    _weighted_augmented_system,
    gauss_newton_step,
)


ConstrainedGaussNewtonBackend = Literal["python", "cpp", "c++"]
ConstrainedGaussNewtonMethod = Literal["auto", "nullspace", "svd", "kkt"]


@dataclass(frozen=True)
class EqualityConstrainedGaussNewtonStepResult:
    """Result of one equality-constrained reduced Gauss-Newton step."""

    step: np.ndarray
    multipliers: np.ndarray
    rank: int
    constraint_rank: int
    method: str
    weighted_residual_norm: float
    linearized_residual_norm: float
    constraint_violation_norm: float
    gradient_norm: float
    damping: float
    converged: bool
    condition_estimate: float = float("nan")


def _normalize_method(method: str | None) -> str:
    name = str(method or "auto").strip().lower()
    if name not in {"auto", "nullspace", "svd", "kkt"}:
        raise ValueError("method must be one of 'auto', 'nullspace', 'svd', or 'kkt'.")
    return name


def _validate_constraints(
    constraint_matrix: np.ndarray | None,
    constraint_rhs: np.ndarray | None,
    n_coefficients: int,
) -> tuple[np.ndarray, np.ndarray]:
    if constraint_matrix is None:
        C = np.zeros((0, int(n_coefficients)), dtype=np.float64)
    else:
        C = np.asarray(constraint_matrix, dtype=float)
        if C.ndim != 2:
            raise ValueError("constraint_matrix must be a rank-2 array.")
        if C.shape[1] != int(n_coefficients):
            raise ValueError("constraint_matrix column count must match the reduced dimension.")
        if not np.all(np.isfinite(C)):
            raise ValueError("constraint_matrix must contain only finite values.")
        C = np.ascontiguousarray(C, dtype=np.float64)
    if constraint_rhs is None:
        h = np.zeros(C.shape[0], dtype=np.float64)
    else:
        h = np.asarray(constraint_rhs, dtype=float).reshape(-1)
        if h.size != C.shape[0]:
            raise ValueError("constraint_rhs size must match constraint_matrix row count.")
        if not np.all(np.isfinite(h)):
            raise ValueError("constraint_rhs must contain only finite values.")
        h = np.ascontiguousarray(h, dtype=np.float64)
    return C, h


def _svd_threshold(singular_values: np.ndarray, rcond: float | None) -> float:
    if singular_values.size == 0:
        return 0.0
    scale = float(np.max(np.abs(singular_values)))
    if scale <= 0.0:
        return 0.0
    if rcond is None or rcond <= 0.0:
        return float(np.finfo(float).eps * max(singular_values.shape) * scale)
    return float(rcond * scale)


def _constraint_particular_and_nullspace(
    C: np.ndarray,
    h: np.ndarray,
    *,
    rcond: float | None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    n = int(C.shape[1])
    if C.shape[0] == 0:
        return np.zeros(n, dtype=np.float64), np.eye(n, dtype=np.float64), 0, 0.0

    U, s, Vt = np.linalg.svd(C, full_matrices=True)
    threshold = _svd_threshold(s, rcond)
    rank = int(np.count_nonzero(s > threshold))
    if rank == 0:
        particular = np.zeros(n, dtype=np.float64)
    else:
        particular = Vt[:rank, :].T @ ((U[:, :rank].T @ h) / s[:rank])
    nullspace = Vt[rank:, :].T.copy()
    violation = float(np.linalg.norm(C @ particular - h))
    return (
        np.ascontiguousarray(particular, dtype=np.float64),
        np.ascontiguousarray(nullspace, dtype=np.float64),
        rank,
        violation,
    )


def _multipliers_from_stationarity(C: np.ndarray, stationarity_rhs: np.ndarray, rcond: float | None) -> np.ndarray:
    if C.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(np.linalg.lstsq(C.T, stationarity_rhs, rcond=None if rcond is None or rcond <= 0.0 else rcond)[0], dtype=float)


def _solve_python_nullspace(
    A: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    h: np.ndarray,
    normal_matrix: np.ndarray,
    normal_rhs: np.ndarray,
    *,
    rcond: float | None,
) -> tuple[np.ndarray, np.ndarray, int, int, str, float, float]:
    particular, nullspace, constraint_rank, constraint_violation = _constraint_particular_and_nullspace(C, h, rcond=rcond)
    tolerance = 100.0 * np.finfo(float).eps * max(1.0, float(np.linalg.norm(C)), float(np.linalg.norm(h)))
    if constraint_violation > max(tolerance, 1.0e-12):
        raise ValueError("equality constraints are inconsistent.")

    if nullspace.shape[1] == 0:
        step = particular
        rank = 0
        condition = float("nan")
    else:
        A_red = A @ nullspace
        b_red = b - A @ particular
        z, _residuals, rank, _singular_values = np.linalg.lstsq(
            A_red,
            b_red,
            rcond=None if rcond is None or rcond <= 0.0 else rcond,
        )
        step = particular + nullspace @ z
        condition = _condition_estimate(A_red)
    stationarity_rhs = normal_rhs - normal_matrix @ step
    multipliers = _multipliers_from_stationarity(C, stationarity_rhs, rcond)
    return (
        np.ascontiguousarray(step, dtype=np.float64),
        np.ascontiguousarray(multipliers, dtype=np.float64),
        int(rank),
        int(constraint_rank),
        "nullspace_svd",
        float(condition),
        float(np.linalg.norm(C @ step - h)),
    )


def _solve_python_kkt(
    A: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    h: np.ndarray,
    normal_matrix: np.ndarray,
    normal_rhs: np.ndarray,
    *,
    rcond: float | None,
) -> tuple[np.ndarray, np.ndarray, int, int, str, float, float]:
    if C.shape[0] == 0:
        result = gauss_newton_step(A, -b, backend="python", method="svd", rcond=rcond)
        return result.step, np.zeros(0, dtype=np.float64), result.rank, 0, "kkt_unconstrained", result.condition_estimate, 0.0

    n = int(normal_matrix.shape[0])
    m = int(C.shape[0])
    K = np.block(
        [
            [normal_matrix, C.T],
            [C, np.zeros((m, m), dtype=np.float64)],
        ]
    )
    rhs = np.concatenate((normal_rhs, h))
    try:
        sol = np.linalg.solve(K, rhs)
        step = np.asarray(sol[:n], dtype=float)
        multipliers = np.asarray(sol[n:], dtype=float)
        if not np.all(np.isfinite(step)) or not np.all(np.isfinite(multipliers)):
            raise np.linalg.LinAlgError("non-finite KKT solution")
        s = np.linalg.svd(C, compute_uv=False)
        constraint_rank = int(np.count_nonzero(s > _svd_threshold(s, rcond)))
        rank = int(np.linalg.matrix_rank(A, tol=None if rcond is None or rcond <= 0.0 else rcond))
        return (
            np.ascontiguousarray(step, dtype=np.float64),
            np.ascontiguousarray(multipliers, dtype=np.float64),
            rank,
            constraint_rank,
            "kkt_solve",
            _condition_estimate(K),
            float(np.linalg.norm(C @ step - h)),
        )
    except np.linalg.LinAlgError:
        out = _solve_python_nullspace(A, b, C, h, normal_matrix, normal_rhs, rcond=rcond)
        step, multipliers, rank, constraint_rank, _method, condition, violation = out
        return step, multipliers, rank, constraint_rank, "kkt_nullspace_fallback", condition, violation


def _result_from_raw(raw: dict) -> EqualityConstrainedGaussNewtonStepResult:
    return EqualityConstrainedGaussNewtonStepResult(
        step=np.asarray(raw["step"], dtype=float).reshape(-1),
        multipliers=np.asarray(raw["multipliers"], dtype=float).reshape(-1),
        rank=int(raw["rank"]),
        constraint_rank=int(raw["constraint_rank"]),
        method=str(raw["method"]),
        weighted_residual_norm=float(raw["weighted_residual_norm"]),
        linearized_residual_norm=float(raw["linearized_residual_norm"]),
        constraint_violation_norm=float(raw["constraint_violation_norm"]),
        gradient_norm=float(raw["gradient_norm"]),
        damping=float(raw["damping"]),
        converged=bool(raw["converged"]),
        condition_estimate=float(raw["condition_estimate"]),
    )


def equality_constrained_gauss_newton_step(
    jacobian: np.ndarray,
    residual: np.ndarray,
    *,
    constraint_matrix: np.ndarray | None = None,
    constraint_rhs: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    damping: float = 0.0,
    damping_diagonal: np.ndarray | None = None,
    method: ConstrainedGaussNewtonMethod = "auto",
    rcond: float | None = None,
    backend: ConstrainedGaussNewtonBackend = "python",
) -> EqualityConstrainedGaussNewtonStepResult:
    """Compute one equality-constrained reduced Gauss-Newton step.

    The subproblem is

    ``min_step ||sqrt(W) (r + J step)||^2 + damping ||D step||^2``

    subject to ``constraint_matrix @ step = constraint_rhs``.
    """

    backend_name = _normalize_backend_name(backend)
    method_name = _normalize_method(method)
    J, r, w, damping_value, diag = _validate_inputs(jacobian, residual, weights, damping, damping_diagonal)
    C, h = _validate_constraints(constraint_matrix, constraint_rhs, J.shape[1])
    rcond_value = -1.0 if rcond is None else float(rcond)
    if not np.isfinite(rcond_value):
        raise ValueError("rcond must be finite when provided.")

    if C.shape[0] == 0:
        result = gauss_newton_step(
            J,
            r,
            weights=w,
            damping=damping_value,
            damping_diagonal=diag,
            method="auto" if method_name == "auto" else "svd",
            rcond=rcond,
            backend=backend_name,
        )
        return EqualityConstrainedGaussNewtonStepResult(
            step=result.step,
            multipliers=np.zeros(0, dtype=np.float64),
            rank=result.rank,
            constraint_rank=0,
            method=f"unconstrained_{result.method}",
            weighted_residual_norm=result.weighted_residual_norm,
            linearized_residual_norm=result.linearized_residual_norm,
            constraint_violation_norm=0.0,
            gradient_norm=result.gradient_norm,
            damping=result.damping,
            converged=result.converged,
            condition_estimate=result.condition_estimate,
        )

    if backend_name == "cpp":
        from .cpp_backend.constrained_gauss_newton import module as _constrained_cpp_module

        raw = _constrained_cpp_module().equality_constrained_gauss_newton_step(
            J,
            r,
            C,
            h,
            w,
            damping_value,
            diag,
            method_name,
            rcond_value,
        )
        return _result_from_raw(raw)

    A, b, weighted_residual_norm = _weighted_augmented_system(J, r, w, damping_value, diag)
    normal_matrix = A.T @ A
    normal_rhs = A.T @ b
    if method_name in {"auto", "nullspace", "svd"}:
        step, multipliers, rank, constraint_rank, method_used, condition, violation = _solve_python_nullspace(
            A,
            b,
            C,
            h,
            normal_matrix,
            normal_rhs,
            rcond=None if rcond is None else rcond_value,
        )
        if method_name == "auto":
            method_used = "auto_nullspace_svd"
    else:
        step, multipliers, rank, constraint_rank, method_used, condition, violation = _solve_python_kkt(
            A,
            b,
            C,
            h,
            normal_matrix,
            normal_rhs,
            rcond=None if rcond is None else rcond_value,
        )
    linearized = float(np.linalg.norm(A @ step - b))
    return EqualityConstrainedGaussNewtonStepResult(
        step=np.ascontiguousarray(step, dtype=np.float64),
        multipliers=np.ascontiguousarray(multipliers, dtype=np.float64),
        rank=int(rank),
        constraint_rank=int(constraint_rank),
        method=method_used,
        weighted_residual_norm=weighted_residual_norm,
        linearized_residual_norm=linearized,
        constraint_violation_norm=float(violation),
        gradient_norm=float(np.linalg.norm(-normal_rhs)),
        damping=damping_value,
        converged=bool(np.all(np.isfinite(step)) and violation <= 1.0e-10),
        condition_estimate=float(condition),
    )


__all__ = [
    "ConstrainedGaussNewtonBackend",
    "ConstrainedGaussNewtonMethod",
    "EqualityConstrainedGaussNewtonStepResult",
    "equality_constrained_gauss_newton_step",
]
