from __future__ import annotations

from typing import Mapping

import numpy as np
import scipy.sparse as sp

from pycutfem.core.dofhandler import DofHandler


def coupled_dirichlet_data(
    *,
    dh_pos: DofHandler,
    bcs_pos,
    dh_neg: DofHandler,
    bcs_neg,
    neg_offset: int,
) -> dict[int, float]:
    """Build a global (row->value) Dirichlet map for a 2-block coupled system."""
    data: dict[int, float] = {}
    data.update({int(k): float(v) for k, v in (dh_pos.get_dirichlet_data(bcs_pos) or {}).items()})
    data_neg = dh_neg.get_dirichlet_data(bcs_neg) or {}
    for k, v in data_neg.items():
        data[int(neg_offset) + int(k)] = float(v)
    return data


def apply_dirichlet_data(
    K: sp.spmatrix,
    F: np.ndarray,
    data: Mapping[int, float],
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Apply strong Dirichlet constraints to (K,F) via row/col elimination.

    This mirrors `FormCompiler._apply_bcs` but operates on already-assembled
    coupled systems.
    """
    if not data:
        return K.tocsr(), np.asarray(F, dtype=float)

    rows = np.fromiter((int(k) for k in data.keys()), dtype=int)
    vals = np.fromiter((float(v) for v in data.values()), dtype=float)
    F = np.asarray(F, dtype=float).copy()

    # Shift RHS by prescribed values
    bc_vec = np.bincount(rows, weights=vals, minlength=F.size)
    F -= K @ bc_vec

    # Eliminate rows/cols
    K_lil = K.tolil()
    K_lil[rows, :] = 0.0
    K_lil[:, rows] = 0.0
    K_lil[rows, rows] = 1.0
    K_csr = K_lil.tocsr()

    F[rows] = vals
    return K_csr, F

