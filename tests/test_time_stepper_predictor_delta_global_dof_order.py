from __future__ import annotations

import numpy as np

from pycutfem.solvers.nonlinear_solver import NewtonSolver


class _DummyDH:
    def __init__(self, total_dofs: int) -> None:
        self.total_dofs = int(total_dofs)


class _DummyFunc:
    def __init__(self, name: str, g_dofs: list[int], values: list[float]) -> None:
        self.name = str(name)
        self._g_dofs = np.asarray(g_dofs, dtype=int)
        self.nodal_values = np.asarray(values, dtype=float)


def test_predictor_delta_uses_global_dof_order_for_last_success_increment() -> None:
    """
    Regression test for the "delta" predictor bookkeeping.

    When mixed fields do not have contiguous global DOF numbering, the per-field
    concatenation `np.hstack([f.nodal_values ...])` does *not* represent a global
    DOF-ordered vector. Time-stepper predictors must therefore build the last-step
    increment in global DOF order via each function's `_g_dofs` map.
    """

    solver = NewtonSolver.__new__(NewtonSolver)  # bypass heavy __init__
    solver.dh = _DummyDH(total_dofs=4)

    # Non-contiguous global DOF numbering across two fields:
    #   f1 owns DOFs [0,2], f2 owns DOFs [1,3].
    f1_prev = _DummyFunc("f1", [0, 2], [10.0, 30.0])
    f2_prev = _DummyFunc("f2", [1, 3], [20.0, 40.0])

    # Apply different per-DOF increments so ordering mistakes are detectable.
    f1 = _DummyFunc("f1", [0, 2], [11.0, 32.0])  # +1 at dof0, +2 at dof2
    f2 = _DummyFunc("f2", [1, 3], [23.0, 44.0])  # +3 at dof1, +4 at dof3

    prev_full = solver._gather_full_iterate([f1_prev, f2_prev])
    cur_full = solver._gather_full_iterate([f1, f2])
    delta_full = cur_full - prev_full

    assert np.allclose(prev_full, np.array([10.0, 20.0, 30.0, 40.0]))
    assert np.allclose(cur_full, np.array([11.0, 23.0, 32.0, 44.0]))
    assert np.allclose(delta_full, np.array([1.0, 3.0, 2.0, 4.0]))

    # The naive concatenation would yield a *different ordering* ([0,2,1,3]).
    delta_hstack = np.hstack([f1.nodal_values, f2.nodal_values]) - np.hstack([f1_prev.nodal_values, f2_prev.nodal_values])
    assert np.allclose(delta_hstack, np.array([1.0, 2.0, 3.0, 4.0]))
    assert not np.allclose(delta_hstack, delta_full)

