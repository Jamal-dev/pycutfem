from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .layout import FieldBlockLayout


@dataclass
class BlockLinearSystem:
    """Sparse linear system together with a reusable block layout."""

    matrix: sp.spmatrix
    rhs: np.ndarray
    layout: FieldBlockLayout
    _block_cache: dict[tuple[int, int], sp.csr_matrix] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not sp.issparse(self.matrix):
            self.matrix = sp.csr_matrix(np.asarray(self.matrix, dtype=float))
        else:
            self.matrix = self.matrix.tocsr()
        self.rhs = np.asarray(self.rhs, dtype=float).reshape(-1)
        nrow, ncol = map(int, self.matrix.shape)
        if nrow != ncol:
            raise ValueError(f"BlockLinearSystem expects a square matrix, got shape {self.matrix.shape}.")
        if int(self.rhs.size) != nrow:
            raise ValueError(
                f"Right-hand side has size {int(self.rhs.size)} but matrix has {nrow} rows."
            )
        if self.layout.size != nrow:
            raise ValueError(
                f"Layout size {self.layout.size} does not match matrix size {nrow}."
            )

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(map(int, self.matrix.shape))

    @property
    def nblocks(self) -> int:
        return int(self.layout.nblocks)

    def block(self, row_block: int | str, col_block: int | str) -> sp.csr_matrix:
        rkey = self.layout._normalize_key(row_block)
        ckey = self.layout._normalize_key(col_block)
        key = (rkey, ckey)
        cached = self._block_cache.get(key)
        if cached is None:
            cached = self.layout.submatrix(self.matrix, rkey, ckey).tocsr()
            self._block_cache[key] = cached
        return cached

    def diagonal_block(self, block: int | str) -> sp.csr_matrix:
        return self.block(block, block)

    def split_rhs(self) -> tuple[np.ndarray, ...]:
        return self.layout.split_vector(self.rhs)

    def assemble_vector(self, parts) -> np.ndarray:
        return self.layout.assemble_vector(parts)

    def as_linear_operator(self) -> spla.LinearOperator:
        return spla.aslinearoperator(self.matrix)
