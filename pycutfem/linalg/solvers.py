from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from .block import BlockLinearSystem


@dataclass(frozen=True)
class LinearSolveReport:
    method: str
    converged: bool
    iterations: int
    residual_norm: float
    residual_history: tuple[float, ...]
    info: int = 0


@dataclass(frozen=True)
class KrylovOptions:
    method: str = "gmres"
    rtol: float = 1.0e-8
    atol: float = 0.0
    maxiter: int | None = 200
    restart: int | None = None
    inner_m: int = 30
    outer_k: int = 3
    callback_type: str | None = "pr_norm"


@dataclass(frozen=True)
class UzawaOptions:
    rtol: float = 1.0e-8
    maxiter: int = 200
    relaxation: float = 1.0


def _preconditioner_operator(preconditioner, shape: tuple[int, int]) -> spla.LinearOperator | None:
    if preconditioner is None:
        return None
    if isinstance(preconditioner, spla.LinearOperator):
        return preconditioner
    if hasattr(preconditioner, "as_linear_operator"):
        return preconditioner.as_linear_operator()
    if callable(preconditioner):
        return spla.LinearOperator(shape, matvec=preconditioner, dtype=float)
    raise TypeError(f"Unsupported preconditioner type {type(preconditioner)!r}.")


class ScipyKrylovSolver:
    def __init__(self, options: KrylovOptions | None = None) -> None:
        self.options = options or KrylovOptions()
        self.last_report: LinearSolveReport | None = None

    def solve(self, matrix, rhs, *, preconditioner=None, x0=None) -> tuple[np.ndarray, LinearSolveReport]:
        a_op = spla.aslinearoperator(matrix)
        b = np.asarray(rhs, dtype=float).reshape(-1)
        residual_history: list[float] = []

        def _callback(arg) -> None:
            if np.isscalar(arg):
                residual_history.append(float(abs(arg)))
                return
            vec = np.asarray(arg, dtype=float).reshape(-1)
            try:
                res = np.asarray(a_op @ vec, dtype=float).reshape(-1) - b
                residual_history.append(float(np.linalg.norm(res, ord=np.inf)))
            except Exception:
                residual_history.append(float(np.linalg.norm(vec, ord=np.inf)))

        method = str(self.options.method or "gmres").strip().lower()
        m_op = _preconditioner_operator(preconditioner, a_op.shape)
        solver_map = {
            "gmres": spla.gmres,
            "lgmres": spla.lgmres,
            "bicgstab": spla.bicgstab,
            "minres": spla.minres,
        }
        if method not in solver_map:
            raise ValueError(f"Unsupported Krylov method '{self.options.method}'.")

        kwargs = {
            "x0": None if x0 is None else np.asarray(x0, dtype=float).reshape(-1),
            "rtol": float(self.options.rtol),
            "maxiter": self.options.maxiter,
            "M": m_op,
            "callback": _callback,
        }
        if method != "minres":
            kwargs["atol"] = float(self.options.atol)
        if method == "gmres":
            kwargs["restart"] = self.options.restart
            kwargs["callback_type"] = self.options.callback_type
        elif method == "lgmres":
            kwargs["inner_m"] = int(self.options.inner_m)
            kwargs["outer_k"] = int(self.options.outer_k)

        x, info = solver_map[method](a_op, b, **kwargs)
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        res = np.asarray(a_op @ x_arr, dtype=float).reshape(-1) - b
        res_norm = float(np.linalg.norm(res, ord=np.inf)) if res.size else 0.0
        converged = int(info) == 0
        report = LinearSolveReport(
            method=method,
            converged=bool(converged),
            iterations=len(residual_history),
            residual_norm=res_norm,
            residual_history=tuple(float(v) for v in residual_history),
            info=int(info),
        )
        self.last_report = report
        if not converged:
            raise RuntimeError(
                f"{method} did not converge (info={int(info)}, |Ax-b|_inf={res_norm:.3e})."
            )
        return x_arr, report


class UzawaSolver:
    """Classical Uzawa iteration for 2x2 saddle-point systems."""

    def __init__(
        self,
        system: BlockLinearSystem,
        *,
        primal_solver,
        schur_solver,
        primal_block: int | str = 0,
        multiplier_block: int | str = 1,
        options: UzawaOptions | None = None,
    ) -> None:
        if int(system.nblocks) != 2:
            raise ValueError(f"UzawaSolver requires a 2-block system, got {system.nblocks}.")
        self.system = system
        self.primal_solver = primal_solver
        self.schur_solver = schur_solver
        self.primal_block = primal_block
        self.multiplier_block = multiplier_block
        self.options = options or UzawaOptions()
        self.last_report: LinearSolveReport | None = None

    def solve(self, x0=None) -> tuple[np.ndarray, LinearSolveReport]:
        rhs_parts = list(self.system.split_rhs())
        primal_idx = self.system.layout._normalize_key(self.primal_block)
        multiplier_idx = self.system.layout._normalize_key(self.multiplier_block)
        rhs_u = rhs_parts[primal_idx]
        rhs_p = rhs_parts[multiplier_idx]
        if x0 is None:
            x_parts = [np.zeros_like(part) for part in rhs_parts]
        else:
            x_parts = [np.asarray(part, dtype=float) for part in self.system.layout.split_vector(x0)]
        x_u = np.asarray(x_parts[primal_idx], dtype=float).reshape(-1)
        x_p = np.asarray(x_parts[multiplier_idx], dtype=float).reshape(-1)
        a12 = self.system.block(self.primal_block, self.multiplier_block)
        a21 = self.system.block(self.multiplier_block, self.primal_block)
        a22 = self.system.block(self.multiplier_block, self.multiplier_block)
        rhs_scale = max(float(np.linalg.norm(self.system.rhs, ord=np.inf)), 1.0)
        history: list[float] = []

        for _it in range(1, int(self.options.maxiter) + 1):
            rhs_u_eff = np.asarray(rhs_u, dtype=float) - np.asarray(a12 @ x_p, dtype=float).reshape(-1)
            x_u = np.asarray(self.primal_solver.solve(rhs_u_eff), dtype=float).reshape(-1)
            res_p = (
                np.asarray(rhs_p, dtype=float)
                - np.asarray(a21 @ x_u, dtype=float).reshape(-1)
                - np.asarray(a22 @ x_p, dtype=float).reshape(-1)
            )
            corr_p = np.asarray(self.schur_solver.solve(res_p), dtype=float).reshape(-1)
            x_p = x_p - float(self.options.relaxation) * corr_p
            x_parts[primal_idx] = x_u
            x_parts[multiplier_idx] = x_p
            x_full = self.system.assemble_vector(x_parts)
            res_full = np.asarray(self.system.matrix @ x_full, dtype=float).reshape(-1) - self.system.rhs
            res_norm = float(np.linalg.norm(res_full, ord=np.inf)) if res_full.size else 0.0
            history.append(res_norm)
            if res_norm <= float(self.options.rtol) * rhs_scale:
                report = LinearSolveReport(
                    method="uzawa",
                    converged=True,
                    iterations=_it,
                    residual_norm=res_norm,
                    residual_history=tuple(history),
                    info=0,
                )
                self.last_report = report
                return x_full, report

        x_parts[primal_idx] = x_u
        x_parts[multiplier_idx] = x_p
        x_full = self.system.assemble_vector(x_parts)
        res_full = np.asarray(self.system.matrix @ x_full, dtype=float).reshape(-1) - self.system.rhs
        res_norm = float(np.linalg.norm(res_full, ord=np.inf)) if res_full.size else 0.0
        report = LinearSolveReport(
            method="uzawa",
            converged=False,
            iterations=int(self.options.maxiter),
            residual_norm=res_norm,
            residual_history=tuple(history),
            info=int(self.options.maxiter),
        )
        self.last_report = report
        raise RuntimeError(
            f"Uzawa iteration did not converge within {int(self.options.maxiter)} steps "
            f"(|Ax-b|_inf={res_norm:.3e})."
        )
