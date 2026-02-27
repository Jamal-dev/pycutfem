import numpy as np
import pytest
import scipy.sparse as sp

from pycutfem.solvers import nonlinear_solver as nls
from pycutfem.solvers.nonlinear_solver import NewtonSolver


pytestmark = pytest.mark.skipif(
    not nls.HAS_PETSC, reason="petsc4py not available"
)


def test_petsc_linear_solver_resets_cache_on_nonfinite():
    """
    Regression test for a PETSc "wrong state / Vec locked" failure mode.

    If a Newton iteration produces a non-finite linear system (NaN/Inf entries),
    the PETSc solve must fail fast and leave the cached PETSc objects usable for
    subsequent retries (e.g. adaptive-Δt).
    """
    solver = NewtonSolver.__new__(NewtonSolver)
    if hasattr(solver, "_petsc_linear_cache"):
        delattr(solver, "_petsc_linear_cache")

    A_good = sp.eye(2, format="csr", dtype=np.float64)
    rhs = np.array([1.0, 2.0], dtype=np.float64)

    x0 = solver._solve_linear_system_petsc(A_good, rhs)
    assert np.allclose(x0, rhs, atol=0.0, rtol=0.0)

    A_bad = A_good.copy()
    A_bad.data[0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        solver._solve_linear_system_petsc(A_bad, rhs)

    x1 = solver._solve_linear_system_petsc(A_good, rhs)
    assert np.allclose(x1, rhs, atol=0.0, rtol=0.0)

