from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .cpp_backend.amgcl import module as _amgcl_module
from .solvers import LinearSolveReport


@dataclass(frozen=True)
class AMGCLSettings:
    preconditioner_type: str = "amg"
    smoother_type: str = "ilu0"
    krylov_type: str = "gmres"
    coarsening_type: str = "aggregation"
    tolerance: float = 1.0e-6
    max_iteration: int = 100
    gmres_krylov_space_dimension: int = 100
    verbosity: int = 1
    scaling: bool = False
    block_size: int = 1
    use_block_matrices_if_possible: bool = True
    coarse_enough: int = 1000
    max_levels: int = -1
    pre_sweeps: int = 1
    post_sweeps: int = 1
    preserve_explicit_zeros: bool = False


def _canonical_csr_matrix(matrix, *, preserve_explicit_zeros: bool = False) -> sp.csr_matrix:
    if sp.issparse(matrix):
        csr = matrix.tocsr(copy=True)
    else:
        csr = sp.csr_matrix(np.asarray(matrix, dtype=float))
    csr = csr.astype(float, copy=False)
    try:
        csr.sum_duplicates()
    except Exception:
        pass
    if not bool(preserve_explicit_zeros):
        try:
            csr.eliminate_zeros()
        except Exception:
            pass
    try:
        csr.sort_indices()
    except Exception:
        pass
    return csr


def _settings_kwargs(cfg: AMGCLSettings) -> dict[str, object]:
    return {
        "preconditioner_type": str(cfg.preconditioner_type),
        "smoother_type": str(cfg.smoother_type),
        "krylov_type": str(cfg.krylov_type),
        "coarsening_type": str(cfg.coarsening_type),
        "tolerance": float(cfg.tolerance),
        "max_iteration": int(cfg.max_iteration),
        "gmres_krylov_space_dimension": int(cfg.gmres_krylov_space_dimension),
        "verbosity": int(cfg.verbosity),
        "scaling": bool(cfg.scaling),
        "block_size": int(cfg.block_size),
        "use_block_matrices_if_possible": bool(cfg.use_block_matrices_if_possible),
        "coarse_enough": int(cfg.coarse_enough),
        "max_levels": int(cfg.max_levels),
        "pre_sweeps": int(cfg.pre_sweeps),
        "post_sweeps": int(cfg.post_sweeps),
    }


def _report_from_raw(cfg: AMGCLSettings, raw: dict[str, object]) -> LinearSolveReport:
    return LinearSolveReport(
        method=f"amgcl/{cfg.krylov_type}",
        converged=bool(raw["converged"]),
        iterations=int(raw["iterations"]),
        residual_norm=float(raw["residual_norm"]),
        residual_history=(),
        info=0 if bool(raw["converged"]) else int(raw["iterations"]),
    )


def solve_sparse_amgcl(
    matrix,
    rhs,
    *,
    params: AMGCLSettings | None = None,
    x0=None,
) -> tuple[np.ndarray, LinearSolveReport]:
    cfg = params or AMGCLSettings()
    csr = _canonical_csr_matrix(matrix, preserve_explicit_zeros=bool(cfg.preserve_explicit_zeros))
    rhs_arr = np.asarray(rhs, dtype=float).reshape(-1)
    if rhs_arr.size != int(csr.shape[0]):
        raise ValueError(
            "RHS size does not match the AMGCL system: "
            f"{rhs_arr.size} != {int(csr.shape[0])}."
        )
    guess = None if x0 is None else np.asarray(x0, dtype=float).reshape(-1).copy()
    if guess is not None and guess.size != rhs_arr.size:
        raise ValueError(
            "Initial guess size does not match the AMGCL system: "
            f"{guess.size} != {rhs_arr.size}."
        )

    raw = _amgcl_module().solve_csr(
        csr.indptr.astype(np.int64, copy=False),
        csr.indices.astype(np.int64, copy=False),
        np.asarray(csr.data, dtype=float),
        rhs_arr,
        guess,
        **_settings_kwargs(cfg),
    )

    solution = np.asarray(raw["solution"], dtype=float).reshape(-1)
    report = _report_from_raw(cfg, raw)
    return solution, report


class AMGCLSubsolver:
    def __init__(self, matrix, params: AMGCLSettings | None = None) -> None:
        self.params = params or AMGCLSettings()
        self.matrix = _canonical_csr_matrix(
            matrix,
            preserve_explicit_zeros=bool(self.params.preserve_explicit_zeros),
        )
        self.last_report: LinearSolveReport | None = None
        self.shape = tuple(map(int, self.matrix.shape))
        self._handle = _amgcl_module().AMGCLSolverHandle(
            self.matrix.indptr.astype(np.int64, copy=False),
            self.matrix.indices.astype(np.int64, copy=False),
            np.asarray(self.matrix.data, dtype=float),
            **_settings_kwargs(self.params),
        )

    def solve(self, rhs: np.ndarray, *, x0=None) -> np.ndarray:
        rhs_arr = np.asarray(rhs, dtype=float).reshape(-1)
        guess = None if x0 is None else np.asarray(x0, dtype=float).reshape(-1).copy()
        raw = self._handle.solve(rhs_arr, guess)
        solution = np.asarray(raw["solution"], dtype=float).reshape(-1)
        report = _report_from_raw(self.params, raw)
        self.last_report = report
        if not report.converged:
            raise RuntimeError(
                "AMGCL did not converge "
                f"(iterations={report.iterations}, residual={report.residual_norm:.3e}, "
                f"tol={self.params.tolerance:.3e})."
            )
        return solution

    def as_linear_operator(self) -> spla.LinearOperator:
        return spla.LinearOperator(self.shape, matvec=self.solve, dtype=float)
