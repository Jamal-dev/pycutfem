from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .block import BlockLinearSystem


def _canonical_sparse_matrix(matrix, *, fmt: str = "csc", shift: float = 0.0) -> sp.spmatrix:
    if not sp.issparse(matrix):
        out = sp.csc_matrix(np.asarray(matrix, dtype=float))
    else:
        out = matrix.copy()
        out = out.asformat(fmt)
    if shift:
        out = out + shift * sp.eye(out.shape[0], out.shape[1], format=fmt)
    try:
        out.sum_duplicates()
    except Exception:
        pass
    try:
        out.eliminate_zeros()
    except Exception:
        pass
    try:
        out.sort_indices()
    except Exception:
        pass
    return out


def _safe_diagonal(diagonal, *, rel_floor: float = 1.0e-12) -> np.ndarray:
    diag = np.asarray(diagonal, dtype=float).reshape(-1).copy()
    if diag.size == 0:
        return diag
    scale = float(np.max(np.abs(diag))) if diag.size else 1.0
    floor = max(float(rel_floor) * max(scale, 1.0), 1.0e-15)
    bad = (~np.isfinite(diag)) | (np.abs(diag) < floor)
    if np.any(bad):
        signs = np.sign(diag)
        signs[(~np.isfinite(signs)) | (signs == 0.0)] = 1.0
        diag[bad] = signs[bad] * floor
    return diag


@dataclass(frozen=True)
class SparseSubsolverSpec:
    kind: str = "ilu"
    shift: float = 0.0
    drop_tol: float = 1.0e-4
    fill_factor: float = 10.0
    diag_pivot_thresh: float | None = None
    diag_rel_floor: float = 1.0e-12


class IdentitySubsolver:
    def __init__(self, size: int) -> None:
        self.shape = (int(size), int(size))

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return np.asarray(rhs, dtype=float).reshape(-1).copy()


class DiagonalSubsolver:
    def __init__(self, diagonal, *, rel_floor: float = 1.0e-12) -> None:
        self.diagonal = _safe_diagonal(diagonal, rel_floor=rel_floor)
        self.shape = (int(self.diagonal.size), int(self.diagonal.size))

    @classmethod
    def from_matrix(cls, matrix, *, rel_floor: float = 1.0e-12) -> "DiagonalSubsolver":
        if sp.issparse(matrix):
            diagonal = np.asarray(matrix.diagonal(), dtype=float).reshape(-1)
        else:
            diagonal = np.asarray(np.diag(np.asarray(matrix, dtype=float)), dtype=float).reshape(-1)
        return cls(diagonal, rel_floor=rel_floor)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs_arr = np.asarray(rhs, dtype=float).reshape(-1)
        if int(rhs_arr.size) != self.shape[0]:
            raise ValueError(
                f"DiagonalSubsolver expected rhs of size {self.shape[0]} but received {int(rhs_arr.size)}."
            )
        return rhs_arr / self.diagonal


class DirectSubsolver:
    def __init__(self, matrix, *, shift: float = 0.0) -> None:
        csc = _canonical_sparse_matrix(matrix, fmt="csc", shift=shift)
        self.shape = tuple(map(int, csc.shape))
        self._factor = spla.splu(csc)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs_arr = np.asarray(rhs, dtype=float)
        out = self._factor.solve(rhs_arr)
        return np.asarray(out, dtype=float).reshape(-1)


class ILUSubsolver:
    def __init__(
        self,
        matrix,
        *,
        shift: float = 0.0,
        drop_tol: float = 1.0e-4,
        fill_factor: float = 10.0,
        diag_pivot_thresh: float | None = None,
    ) -> None:
        csc = _canonical_sparse_matrix(matrix, fmt="csc", shift=shift)
        self.shape = tuple(map(int, csc.shape))
        kwargs = {
            "drop_tol": float(drop_tol),
            "fill_factor": float(fill_factor),
        }
        if diag_pivot_thresh is not None:
            kwargs["diag_pivot_thresh"] = float(diag_pivot_thresh)
        self._factor = spla.spilu(csc, **kwargs)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs_arr = np.asarray(rhs, dtype=float).reshape(-1)
        out = self._factor.solve(rhs_arr)
        return np.asarray(out, dtype=float).reshape(-1)


def build_subsolver(matrix, spec) -> object:
    if hasattr(spec, "solve") and callable(getattr(spec, "solve")):
        return spec
    if isinstance(spec, str):
        spec_obj = SparseSubsolverSpec(kind=str(spec))
    elif isinstance(spec, dict):
        spec_obj = SparseSubsolverSpec(**spec)
    elif isinstance(spec, SparseSubsolverSpec):
        spec_obj = spec
    else:
        raise TypeError(f"Unsupported subsolver specification type: {type(spec)!r}")

    kind = str(spec_obj.kind or "ilu").strip().lower().replace("-", "_")
    size = int(matrix.shape[0]) if hasattr(matrix, "shape") else int(matrix)
    if kind in {"identity", "none"}:
        return IdentitySubsolver(size=size)
    if kind in {"diag", "diagonal"}:
        return DiagonalSubsolver.from_matrix(matrix, rel_floor=spec_obj.diag_rel_floor)
    if kind in {"direct", "lu"}:
        return DirectSubsolver(matrix, shift=spec_obj.shift)
    if kind == "ilu":
        return ILUSubsolver(
            matrix,
            shift=spec_obj.shift,
            drop_tol=spec_obj.drop_tol,
            fill_factor=spec_obj.fill_factor,
            diag_pivot_thresh=spec_obj.diag_pivot_thresh,
        )
    raise ValueError(f"Unsupported sparse subsolver kind '{spec_obj.kind}'.")


def lumped_schur_complement(
    system: BlockLinearSystem,
    *,
    primal_block: int | str,
    multiplier_block: int | str,
    shift: float = 0.0,
    diag_rel_floor: float = 1.0e-12,
    add_constraint_diagonal: bool = True,
) -> sp.csr_matrix:
    a_block = system.block(primal_block, primal_block)
    b_block = system.block(multiplier_block, primal_block)
    bt_block = system.block(primal_block, multiplier_block)
    inv_diag = 1.0 / np.abs(_safe_diagonal(a_block.diagonal(), rel_floor=diag_rel_floor))
    schur = (b_block @ sp.diags(inv_diag, format="csr") @ bt_block).tocsr()
    if add_constraint_diagonal:
        c_block = system.block(multiplier_block, multiplier_block)
        if int(c_block.shape[0]) and int(c_block.shape[1]):
            c_diag = np.abs(np.asarray(c_block.diagonal(), dtype=float).reshape(-1))
            if c_diag.size:
                schur = schur + sp.diags(c_diag, format="csr")
    if shift:
        schur = schur + float(shift) * sp.eye(schur.shape[0], schur.shape[1], format="csr")
    return schur.tocsr()


class BlockDiagonalPreconditioner:
    def __init__(self, system: BlockLinearSystem, block_solvers) -> None:
        self.system = system
        self.block_solvers = tuple(block_solvers)
        if len(self.block_solvers) != self.system.nblocks:
            raise ValueError(
                f"Expected {self.system.nblocks} block solvers but received {len(self.block_solvers)}."
            )

    def apply(self, x: np.ndarray) -> np.ndarray:
        rhs_parts = self.system.layout.split_vector(x)
        out_parts = [solver.solve(rhs) for solver, rhs in zip(self.block_solvers, rhs_parts)]
        return self.system.assemble_vector(out_parts)

    def as_linear_operator(self) -> spla.LinearOperator:
        shape = self.system.shape
        return spla.LinearOperator(shape, matvec=self.apply, dtype=float)


class BlockTriangularPreconditioner:
    def __init__(self, system: BlockLinearSystem, block_solvers, *, lower: bool = True) -> None:
        self.system = system
        self.block_solvers = tuple(block_solvers)
        self.lower = bool(lower)
        if len(self.block_solvers) != self.system.nblocks:
            raise ValueError(
                f"Expected {self.system.nblocks} block solvers but received {len(self.block_solvers)}."
            )

    def apply(self, x: np.ndarray) -> np.ndarray:
        rhs_parts = [np.asarray(part, dtype=float).copy() for part in self.system.layout.split_vector(x)]
        sol_parts = [np.zeros((len(part),), dtype=float) for part in rhs_parts]
        if self.lower:
            order = range(self.system.nblocks)
            coupling = lambda i: range(i)
        else:
            order = range(self.system.nblocks - 1, -1, -1)
            coupling = lambda i: range(i + 1, self.system.nblocks)

        for i in order:
            rhs_i = rhs_parts[i].copy()
            for j in coupling(i):
                block_ij = self.system.block(i, j)
                if block_ij.nnz:
                    rhs_i -= np.asarray(block_ij @ sol_parts[j], dtype=float).reshape(-1)
            sol_parts[i] = np.asarray(self.block_solvers[i].solve(rhs_i), dtype=float).reshape(-1)
        return self.system.assemble_vector(sol_parts)

    def as_linear_operator(self) -> spla.LinearOperator:
        shape = self.system.shape
        return spla.LinearOperator(shape, matvec=self.apply, dtype=float)


class UzawaPreconditioner:
    def __init__(
        self,
        system: BlockLinearSystem,
        *,
        primal_solver,
        schur_solver,
        primal_block: int | str = 0,
        multiplier_block: int | str = 1,
        relaxation: float = 1.0,
    ) -> None:
        if int(system.nblocks) != 2:
            raise ValueError(
                f"UzawaPreconditioner requires a 2-block system, got {system.nblocks} blocks."
            )
        self.system = system
        self.primal_solver = primal_solver
        self.schur_solver = schur_solver
        self.primal_block = primal_block
        self.multiplier_block = multiplier_block
        self.relaxation = float(relaxation)

    def apply(self, x: np.ndarray) -> np.ndarray:
        parts = list(self.system.layout.split_vector(x))
        primal_idx = self.system.layout._normalize_key(self.primal_block)
        multiplier_idx = self.system.layout._normalize_key(self.multiplier_block)
        ru = parts[primal_idx]
        rp = parts[multiplier_idx]
        b_block = self.system.block(self.multiplier_block, self.primal_block)
        zu = np.asarray(self.primal_solver.solve(ru), dtype=float).reshape(-1)
        rp_corr = np.asarray(rp, dtype=float).reshape(-1) - np.asarray(b_block @ zu, dtype=float).reshape(-1)
        zp = -self.relaxation * np.asarray(self.schur_solver.solve(rp_corr), dtype=float).reshape(-1)
        out_parts = [np.zeros_like(part) for part in parts]
        out_parts[primal_idx] = zu
        out_parts[multiplier_idx] = zp
        return self.system.assemble_vector(out_parts)

    def as_linear_operator(self) -> spla.LinearOperator:
        return spla.LinearOperator(self.system.shape, matvec=self.apply, dtype=float)
