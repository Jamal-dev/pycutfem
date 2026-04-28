from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .cpp_backend.eigen_sparseqr import module as _eigen_sparseqr_module
from .solvers import LinearSolveReport


def _canonical_csr_matrix(matrix) -> sp.csr_matrix:
    if sp.issparse(matrix):
        csr = matrix.tocsr(copy=True)
    else:
        csr = sp.csr_matrix(np.asarray(matrix, dtype=float))
    csr = csr.astype(float, copy=False)
    try:
        csr.sum_duplicates()
    except Exception:
        pass
    try:
        csr.eliminate_zeros()
    except Exception:
        pass
    try:
        csr.sort_indices()
    except Exception:
        pass
    return csr


@dataclass(frozen=True)
class EigenSparseQRSettings:
    ordering: str = "colamd"


def _report_from_raw(raw: dict[str, object]) -> LinearSolveReport:
    return LinearSolveReport(
        method="eigen_sparseqr",
        converged=bool(raw["converged"]),
        iterations=1,
        residual_norm=float("nan"),
        residual_history=(),
        info=0 if bool(raw["converged"]) else 1,
    )


def solve_sparse_eigen_sparseqr(matrix, rhs) -> tuple[np.ndarray, LinearSolveReport]:
    csr = _canonical_csr_matrix(matrix)
    rhs_arr = np.asarray(rhs, dtype=float).reshape(-1)
    if rhs_arr.size != int(csr.shape[0]):
        raise ValueError(
            "RHS size does not match the Eigen SparseQR system: "
            f"{rhs_arr.size} != {int(csr.shape[0])}."
        )
    raw = _eigen_sparseqr_module().solve_csr(
        csr.indptr.astype(np.int64, copy=False),
        csr.indices.astype(np.int64, copy=False),
        np.asarray(csr.data, dtype=float),
        rhs_arr,
    )
    solution = np.asarray(raw["solution"], dtype=float).reshape(-1)
    report = _report_from_raw(raw)
    return solution, report


class EigenSparseQRSubsolver:
    def __init__(self, matrix) -> None:
        self.matrix = _canonical_csr_matrix(matrix)
        self.shape = tuple(map(int, self.matrix.shape))
        self.last_report: LinearSolveReport | None = None
        self._handle = _eigen_sparseqr_module().EigenSparseQRHandle(
            self.matrix.indptr.astype(np.int64, copy=False),
            self.matrix.indices.astype(np.int64, copy=False),
            np.asarray(self.matrix.data, dtype=float),
        )

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs_arr = np.asarray(rhs, dtype=float).reshape(-1)
        raw = self._handle.solve(rhs_arr)
        solution = np.asarray(raw["solution"], dtype=float).reshape(-1)
        report = _report_from_raw(raw)
        self.last_report = report
        if not report.converged:
            raise RuntimeError("Eigen SparseQR did not converge.")
        return solution
