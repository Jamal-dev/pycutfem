import types

import numpy as np
import scipy.sparse as sp


def test_prune_drop_tol_is_not_scaled_by_global_max(monkeypatch):
    """
    Regression test for `_prune_decoupled_rows_cols`:

    A single huge matrix entry (e.g. from a penalty block) must NOT cause
    unrelated moderate rows/cols to be pruned when `PYCUTFEM_DROP_TOL` is set.
    """
    monkeypatch.setenv("PYCUTFEM_DROP_TOL", "1e-12")

    from pycutfem.solvers.nonlinear_solver import NewtonSolver, _ActiveReducer

    n = 6
    diag = np.ones(n, dtype=float)
    diag[0] = 1.0e14  # outlier
    A_full = sp.diags(diag, format="csr")
    R_full = np.zeros(n, dtype=float)

    dh = types.SimpleNamespace(total_dofs=n)

    dummy = types.SimpleNamespace()
    dummy.dh = dh
    dummy.constraints = None
    dummy.active_dofs = np.arange(n, dtype=int)
    dummy.full_to_red = np.arange(n, dtype=int)
    dummy.restrictor = _ActiveReducer(dh, dummy.active_dofs, constraint=None)
    dummy._pattern_stale = False

    # Called only if pruning happens. Return a consistent reduced view.
    def _assemble_system_reduced(_coeffs, *, need_matrix: bool = True):
        aid = np.asarray(dummy.active_dofs, dtype=int)
        A = A_full[np.ix_(aid, aid)] if need_matrix else None
        R = R_full[aid]
        return A, R

    dummy._assemble_system_reduced = _assemble_system_reduced
    dummy._build_reduced_pattern = lambda: None

    A_red, R_red, pruned, _extra = NewtonSolver._prune_decoupled_rows_cols(  # type: ignore[attr-defined]
        dummy, {}, A_full, R_full, n
    )

    assert pruned is False
    assert np.asarray(dummy.active_dofs, dtype=int).size == n
    assert A_red.shape == (n, n)
    assert R_red.shape == (n,)
