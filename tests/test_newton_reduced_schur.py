from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from pycutfem.solvers.nonlinear_solver import NewtonSolver


class _DummyDH:
    def __init__(self) -> None:
        self.field_names = ["u_x", "u_y", "p", "p_mean"]
        self.total_dofs = 6
        self._slices = {
            "u_x": [0, 1],
            "u_y": [2, 3],
            "p": [4],
            "p_mean": [5],
        }

    def get_field_slice(self, field: str):
        return list(self._slices[field])


def _make_dummy_solver() -> NewtonSolver:
    solver = NewtonSolver.__new__(NewtonSolver)
    solver.dh = _DummyDH()
    solver.constraints = None
    solver.active_dofs = np.arange(6, dtype=int)
    solver.full_to_red = np.arange(6, dtype=int)
    solver._reduced_field_names = np.asarray(["u_x", "u_x", "u_y", "u_y", "p", "p_mean"], dtype=object)
    solver._reduced_schur_enabled = True
    solver._reduced_schur_pressure_fields = ("p", "p_mean")
    solver._reduced_schur_diag_only = False
    solver._reduced_schur_shift_rel = 1.0e-12
    solver._reduced_schur_pressure_scale_mode = "none"
    solver._reduced_schur_pressure_scale_value = 1.0
    solver._reduced_schur_trace = False
    solver._reduced_schur_last_info = {}
    solver.lp = SimpleNamespace(backend="scipy")
    solver.np = SimpleNamespace(line_search=True, globalization="line_search")
    return solver


def test_reduced_schur_solve_solves_saddle_block():
    solver = _make_dummy_solver()
    K = np.array(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.5, 0.2, 0.0],
            [0.0, 0.2, 2.8, 0.7],
            [0.0, 0.0, 0.7, 2.4],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [1.0, -0.5, 0.3, 0.1],
            [0.2, 0.1, -0.4, 0.9],
        ],
        dtype=float,
    )
    A = np.block([[K, B.T], [B, np.zeros((2, 2), dtype=float)]])
    A_csr = sp.csr_matrix(A)
    rhs = np.array([1.0, -2.0, 0.5, 1.5, -0.25, 0.75], dtype=float)

    sol = solver._solve_linear_system_reduced_schur(A_csr, rhs)
    res = np.asarray(A_csr @ sol - rhs, dtype=float).ravel()

    assert float(np.linalg.norm(res, ord=np.inf)) < 1.0e-9
    assert int(solver._reduced_schur_last_info.get("p_dofs", 0)) == 2
    assert int(solver._reduced_schur_last_info.get("u_dofs", 0)) == 4
    assert str(solver._reduced_schur_last_info.get("pressure_scale_mode", "")) == "none"


def test_reduced_schur_pressure_scaling_keeps_solution():
    solver = _make_dummy_solver()
    solver._reduced_schur_pressure_scale_mode = "drag"
    solver._reduced_schur_pressure_scale_value = 35.0
    K = np.array(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.5, 0.2, 0.0],
            [0.0, 0.2, 2.8, 0.7],
            [0.0, 0.0, 0.7, 2.4],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [1.0, -0.5, 0.3, 0.1],
            [0.2, 0.1, -0.4, 0.9],
        ],
        dtype=float,
    )
    A = np.block([[K, B.T], [B, np.zeros((2, 2), dtype=float)]])
    A_csr = sp.csr_matrix(A)
    rhs = np.array([1.0, -2.0, 0.5, 1.5, -0.25, 0.75], dtype=float)

    sol = solver._solve_linear_system_reduced_schur(A_csr, rhs)
    res = np.asarray(A_csr @ sol - rhs, dtype=float).ravel()

    assert float(np.linalg.norm(res, ord=np.inf)) < 1.0e-9
    assert str(solver._reduced_schur_last_info.get("pressure_scale_mode", "")) == "drag"
    assert abs(float(solver._reduced_schur_last_info.get("pressure_scale", 0.0)) - 35.0) < 1.0e-12


def test_reduced_schur_solve_rejects_pi_s_split_branch():
    solver = _make_dummy_solver()
    solver._reduced_field_names = np.asarray(["u_x", "u_x", "u_y", "u_y", "pi_s", "p_mean"], dtype=object)
    A = sp.eye(6, format="csr")
    rhs = np.ones((6,), dtype=float)

    try:
        solver._solve_linear_system_reduced_schur(A, rhs)
    except RuntimeError as exc:
        assert "pi_s split branch" in str(exc)
    else:
        raise AssertionError("Expected the reduced Schur solve to refuse the pi_s branch.")


def test_globalized_newton_usable_direct_solve_quality_tracks_newton_tolerance():
    solver = _make_dummy_solver()
    solver.np = SimpleNamespace(
        line_search=True,
        globalization="line_search",
        newton_tol=1.0e-6,
        newton_rtol=1.0e-6,
    )

    assert solver._direct_solve_quality_passes(5.6e-5, 4.1e-5)

    solver.np = SimpleNamespace(
        line_search=True,
        globalization="line_search",
        newton_tol=1.0e-8,
        newton_rtol=0.0,
    )

    assert not solver._direct_solve_quality_passes(5.6e-5, 4.1e-5)


def test_newton_selected_field_mask_uses_reduced_field_names() -> None:
    solver = _make_dummy_solver()

    mask = solver._newton_selected_field_mask(("u_x", "p"))
    np.testing.assert_array_equal(mask, np.array([True, True, False, False, True, False], dtype=bool))


def test_newton_selected_field_mask_defaults_to_all_reduced_dofs() -> None:
    solver = _make_dummy_solver()

    mask = solver._newton_selected_field_mask(())
    np.testing.assert_array_equal(mask, np.ones((6,), dtype=bool))


def test_newton_ptc_adds_row_normalized_selected_block_by_default() -> None:
    solver = _make_dummy_solver()
    solver.np = SimpleNamespace(
        ptc_operator_mode="row_normalized",
        ptc_min_diag=1.0e-12,
    )
    dense = np.eye(6, dtype=float)
    mask = solver._newton_selected_field_mask(("u_x", "u_y"))
    idx = np.flatnonzero(mask)
    block = np.eye(idx.size, dtype=float)
    block[0, :2] = np.array([2.0, -4.0], dtype=float)
    block[1, :2] = np.array([1.0, 3.0], dtype=float)
    dense[np.ix_(idx, idx)] = block
    A = sp.csr_matrix(dense)

    A_ptc, add = solver._newton_apply_ptc_regularization(A, A, mask, 3.0)

    added = np.asarray((A_ptc - A).todense(), dtype=float)
    expected_block = np.zeros((idx.size, idx.size), dtype=float)
    expected_block[0, :2] = 3.0 * np.array([0.5, -1.0], dtype=float)
    expected_block[1, :2] = 3.0 * np.array([1.0 / 3.0, 1.0], dtype=float)
    expected_block[2, 2] = 3.0
    expected_block[3, 3] = 3.0
    np.testing.assert_allclose(added[np.ix_(idx, idx)], expected_block)
    np.testing.assert_allclose(added[~mask, :], 0.0)
    np.testing.assert_allclose(added[:, ~mask], 0.0)
    expected_add = np.zeros((6,), dtype=float)
    expected_add[idx] = np.array([1.5, 3.0, 3.0, 3.0], dtype=float)
    np.testing.assert_allclose(add, expected_add)


def test_newton_ptc_switches_to_late_fields_and_mode() -> None:
    solver = _make_dummy_solver()
    solver.np = SimpleNamespace(
        ptc_fields=("u_x", "u_y"),
        ptc_operator_mode="row_normalized",
        ptc_late_fields=("p", "p_mean"),
        ptc_late_switch_residual=1.0e-3,
        ptc_late_operator_mode="diag",
    )

    assert not solver._newton_ptc_late_phase_active(residual_norm=5.0e-3)
    assert solver._newton_ptc_active_fields(residual_norm=5.0e-3) == ("u_x", "u_y")
    assert solver._newton_ptc_operator_mode(residual_norm=5.0e-3) == "row_normalized"

    assert solver._newton_ptc_late_phase_active(residual_norm=5.0e-4)
    assert solver._newton_ptc_active_fields(residual_norm=5.0e-4) == ("p", "p_mean")
    assert solver._newton_ptc_operator_mode(residual_norm=5.0e-4) == "diag"


def test_newton_ptc_diag_mode_adds_selected_diagonal_metric() -> None:
    solver = _make_dummy_solver()
    solver.np = SimpleNamespace(
        ptc_operator_mode="diag",
        ptc_min_diag=1.0e-12,
    )
    dense = np.array(
        [
            [4.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 5.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 7.0],
            [0.0, 0.0, 0.0, 0.0, 7.0, 9.0],
        ],
        dtype=float,
    )
    A = sp.csr_matrix(dense)
    mask = solver._newton_selected_field_mask(("p", "p_mean"))

    A_ptc, add = solver._newton_apply_ptc_regularization(A, A, mask, 2.0)

    added = np.asarray((A_ptc - A).todense(), dtype=float)
    expected_add = np.zeros((6,), dtype=float)
    expected_add[4] = 14.0
    expected_add[5] = 18.0
    np.testing.assert_allclose(np.diag(added), expected_add)
    np.testing.assert_allclose(added - np.diag(np.diag(added)), 0.0)
    np.testing.assert_allclose(add, expected_add)


def test_newton_selected_field_freeze_zeroes_complement_rhs() -> None:
    solver = _make_dummy_solver()
    A = sp.csr_matrix(np.arange(1, 37, dtype=float).reshape((6, 6)))
    rhs = np.arange(1, 7, dtype=float)
    mask = solver._newton_selected_field_mask(("u_x",))

    A_frozen, rhs_frozen = solver._newton_apply_selected_field_freeze(A, rhs, mask)

    frozen = np.flatnonzero(~mask)
    np.testing.assert_allclose(rhs_frozen[frozen], 0.0)
    np.testing.assert_allclose(np.asarray(A_frozen[frozen, :].todense()), np.eye(6, dtype=float)[frozen, :])
