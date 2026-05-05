
"""pycutfem.assembly.boundary_conditions
Proper Dirichlet elimination with RHS correction.
"""
import numpy as np
import scipy.sparse as sp

def _zero_rows_cols_identity_csr(K: sp.spmatrix, rows: np.ndarray) -> sp.csr_matrix:
    rows = np.unique(np.asarray(rows, dtype=int).ravel())
    if rows.size == 0:
        return K.tocsr(copy=True) if hasattr(K, "tocsr") else sp.csr_matrix(K)

    K_csc = K.tocsc(copy=True) if hasattr(K, "tocsc") else sp.csc_matrix(K)
    n_rows, n_cols = K_csc.shape
    keep = (rows >= 0) & (rows < min(n_rows, n_cols))
    rows = rows[keep]
    if rows.size == 0:
        return K_csc.tocsr()

    c_indptr = K_csc.indptr
    c_data = K_csc.data
    for rr in rows.tolist():
        start = int(c_indptr[rr])
        end = int(c_indptr[rr + 1])
        if end > start:
            c_data[start:end] = 0.0

    K_csr = K_csc.tocsr()
    indptr = K_csr.indptr
    indices = K_csr.indices
    data = K_csr.data
    missing_diag: list[int] = []
    for rr in rows.tolist():
        start = int(indptr[rr])
        end = int(indptr[rr + 1])
        if end <= start:
            missing_diag.append(rr)
            continue
        data[start:end] = 0.0
        hit = np.nonzero(indices[start:end] == rr)[0]
        if hit.size:
            data[start + int(hit[0])] = 1.0
        else:
            missing_diag.append(rr)
    if missing_diag:
        diag_rows = np.asarray(missing_diag, dtype=int)
        K_csr = (
            K_csr
            + sp.csr_matrix(
                (np.ones(diag_rows.size, dtype=float), (diag_rows, diag_rows)),
                shape=K_csr.shape,
            )
        ).tocsr()
    K_csr.eliminate_zeros()
    return K_csr

def apply_dirichlet(K, F, dbc):
    """Return (K_bc, F_bc).

    Parameters
    ----------
    dbc : dict {dof: value}
        Dirichlet boundary values.
    """
    K = K.tocsr(copy=True) if hasattr(K, "tocsr") else sp.csr_matrix(K)
    F = F.copy()

    if not dbc:
        return K, F

    rows = np.fromiter(dbc.keys(), dtype=int)
    vals = np.fromiter(dbc.values(), dtype=float)
    if rows.size == 0:
        return K, F

    F -= K @ np.bincount(rows, weights=vals, minlength=K.shape[0])
    K = _zero_rows_cols_identity_csr(K, rows)
    F[rows] = vals
    return K, F
