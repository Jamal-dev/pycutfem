from __future__ import annotations

import numpy as np
import scipy.sparse as sp  # type: ignore

from examples.debug.parity.compare_vi_linearized_dump import _build_augmented_matrix


def test_vi_dump_build_augmented_matrix_drops_equality_coupling_on_active_rows() -> None:
    A = sp.csr_matrix(np.array([[2.0, 0.5], [0.25, 3.0]], dtype=float))
    B = sp.csr_matrix(np.array([[4.0, 5.0]], dtype=float))
    rhs_base = np.array([1.0, -2.0], dtype=float)
    lo = np.array([-np.inf, -np.inf], dtype=float)
    hi = np.array([np.inf, 1.0], dtype=float)
    b_eff = np.array([0.0], dtype=float)
    state = np.array([0, -1], dtype=np.int8)

    M, rhs = _build_augmented_matrix(
        A=A,
        B=B,
        rhs_base=rhs_base,
        state=state,
        lo=lo,
        hi=hi,
        b_eff=b_eff,
        shift_factor=0.0,
    )

    M = M.toarray()
    assert np.isclose(M[1, 1], 1.0)
    assert np.isclose(M[1, 2], 0.0)
    assert np.isclose(rhs[1], 1.0)


def test_vi_dump_build_augmented_matrix_uses_base_diagonal_for_shift() -> None:
    A = sp.csr_matrix(np.array([[2.0, 100.0], [50.0, 3.0]], dtype=float))
    B = sp.csr_matrix(np.array([[1.0, 0.0]], dtype=float))
    rhs_base = np.zeros((2,), dtype=float)
    lo = np.full((2,), -np.inf, dtype=float)
    hi = np.full((2,), np.inf, dtype=float)
    b_eff = np.zeros((1,), dtype=float)
    state = np.zeros((2,), dtype=np.int8)

    M, _ = _build_augmented_matrix(
        A=A,
        B=B,
        rhs_base=rhs_base,
        state=state,
        lo=lo,
        hi=hi,
        b_eff=b_eff,
        shift_factor=1.0e-1,
    )

    M = M.toarray()
    # The added regularization is lambda * abs(diag(A_base)), not a row-sum scale.
    assert np.isclose(M[0, 0], 2.0 + 0.1 * 2.0)
    assert np.isclose(M[1, 1], 3.0 + 0.1 * 3.0)
