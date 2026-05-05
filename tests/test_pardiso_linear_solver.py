import types

import numpy as np
import pytest
import scipy.sparse as sp

from pycutfem.solvers import nonlinear_solver as nls
from pycutfem.solvers.nonlinear_solver import NewtonSolver


def _make_solver(backend: str) -> NewtonSolver:
    solver = NewtonSolver.__new__(NewtonSolver)
    solver.lp = types.SimpleNamespace(backend=backend)
    return solver


def test_pardiso_alias_matches_direct_backend() -> None:
    if not nls.HAS_PYPARDISO:
        pytest.skip("pypardiso not available")

    A = sp.csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64))
    rhs = np.array([1.0, 2.0], dtype=np.float64)
    ref = np.linalg.solve(A.toarray(), rhs)

    x_alias = _make_solver("pypardiso")._solve_linear_system(A, rhs)
    x_named = _make_solver("pardiso")._solve_linear_system(A, rhs)

    assert np.allclose(x_alias, ref, atol=1.0e-12, rtol=0.0)
    assert np.allclose(x_named, ref, atol=1.0e-12, rtol=0.0)


def test_pardiso_backend_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    solver = _make_solver("pardiso")
    A = sp.eye(2, format="csr", dtype=np.float64)
    rhs = np.array([1.0, 2.0], dtype=np.float64)

    monkeypatch.setattr(nls, "HAS_PYPARDISO", False)
    monkeypatch.setattr(nls, "PyPardisoSolver", None)

    with pytest.raises(RuntimeError, match="pypardiso is not available"):
        solver._solve_linear_system(A, rhs)


def test_pardiso_linear_solver_resets_cache_on_nonfinite() -> None:
    if not nls.HAS_PYPARDISO:
        pytest.skip("pypardiso not available")

    solver = _make_solver("pardiso")
    A_good = sp.eye(2, format="csr", dtype=np.float64)
    rhs = np.array([1.0, 2.0], dtype=np.float64)

    x0 = solver._solve_linear_system_pardiso(A_good, rhs)
    assert np.allclose(x0, rhs, atol=0.0, rtol=0.0)

    A_bad = A_good.copy()
    A_bad.data[0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        solver._solve_linear_system_pardiso(A_bad, rhs)

    x1 = solver._solve_linear_system_pardiso(A_good, rhs)
    assert np.allclose(x1, rhs, atol=0.0, rtol=0.0)
