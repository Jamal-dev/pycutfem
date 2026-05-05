from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from pycutfem.linalg import (
    BlockLinearSystem,
    BlockTriangularPreconditioner,
    DirectSubsolver,
    FieldBlockLayout,
    KrylovOptions,
    ScipyKrylovSolver,
    UzawaOptions,
    UzawaSolver,
    build_subsolver,
    lumped_schur_complement,
)
from pycutfem.solvers.nonlinear_solver import NewtonSolver


def _make_saddle_matrix(*, n_primal: int, n_constraints: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n_primal, n_primal))
    a = m.T @ m + 2.0 * np.eye(n_primal)
    b = rng.standard_normal((n_constraints, n_primal))
    z = np.zeros((n_constraints, n_constraints), dtype=float)
    mat = np.block([[a, b.T], [b, z]])
    rhs = rng.standard_normal((n_primal + n_constraints,), dtype=float)
    return a, b, sp.csr_matrix(mat), rhs


def test_field_block_layout_groups_fields_and_remainder():
    field_names = np.asarray(["u", "u", "p", "lam0", "lam1", "aux"], dtype=object)
    layout = FieldBlockLayout.from_field_names(
        field_names,
        [
            ("flow", ("u", "p")),
            ("constraints", ("lam0", "lam1")),
        ],
        include_remaining=True,
        remainder_name="other",
    )

    assert layout.block_names == ("flow", "constraints", "other")
    np.testing.assert_array_equal(layout.indices("flow"), np.array([0, 1, 2], dtype=int))
    np.testing.assert_array_equal(layout.indices("constraints"), np.array([3, 4], dtype=int))
    np.testing.assert_array_equal(layout.indices("other"), np.array([5], dtype=int))


def test_block_triangular_gmres_solves_saddle_system_with_ilu():
    _, _, mat, rhs = _make_saddle_matrix(n_primal=12, n_constraints=3, seed=2)
    field_names = np.asarray(["u"] * 12 + ["lam"] * 3, dtype=object)
    layout = FieldBlockLayout.from_field_names(
        field_names,
        [("primal", ("u",)), ("lambda", ("lam",))],
        include_remaining=False,
    )
    system = BlockLinearSystem(mat, rhs, layout)
    schur = lumped_schur_complement(system, primal_block="primal", multiplier_block="lambda", shift=1.0e-8)
    block_solvers = (
        build_subsolver(system.diagonal_block("primal"), {"kind": "ilu", "drop_tol": 1.0e-6, "fill_factor": 25.0}),
        build_subsolver(schur, {"kind": "direct"}),
    )
    preconditioner = BlockTriangularPreconditioner(system, block_solvers, lower=True)
    solver = ScipyKrylovSolver(KrylovOptions(method="gmres", rtol=1.0e-10, maxiter=200, restart=50))

    sol, report = solver.solve(system.matrix, system.rhs, preconditioner=preconditioner)
    ref = np.linalg.solve(system.matrix.toarray(), rhs)

    assert report.converged
    assert report.iterations > 0
    np.testing.assert_allclose(sol, ref, atol=1.0e-8, rtol=1.0e-8)


def test_uzawa_solver_solves_two_block_saddle_system():
    a_dense, b_dense, mat, rhs = _make_saddle_matrix(n_primal=10, n_constraints=2, seed=4)
    field_names = np.asarray(["u"] * 10 + ["lam"] * 2, dtype=object)
    layout = FieldBlockLayout.from_field_names(
        field_names,
        [("primal", ("u",)), ("lambda", ("lam",))],
        include_remaining=False,
    )
    system = BlockLinearSystem(mat, rhs, layout)
    schur_exact = sp.csr_matrix(b_dense @ np.linalg.solve(a_dense, b_dense.T))
    uzawa = UzawaSolver(
        system,
        primal_solver=DirectSubsolver(system.diagonal_block("primal")),
        schur_solver=DirectSubsolver(schur_exact, shift=1.0e-12),
        options=UzawaOptions(rtol=1.0e-10, maxiter=10, relaxation=1.0),
    )

    sol, report = uzawa.solve()
    ref = np.linalg.solve(system.matrix.toarray(), rhs)

    assert report.converged
    np.testing.assert_allclose(sol, ref, atol=1.0e-9, rtol=1.0e-9)


def test_block_layer_handles_eight_constraint_blocks():
    _, _, mat, rhs = _make_saddle_matrix(n_primal=16, n_constraints=8, seed=6)
    field_names = np.asarray(["u"] * 16 + [f"lam{i}" for i in range(8)], dtype=object)
    layout = FieldBlockLayout.from_field_names(
        field_names,
        [("primal", ("u",))] + [(f"lam{i}", (f"lam{i}",)) for i in range(8)],
        include_remaining=False,
    )
    system = BlockLinearSystem(mat, rhs, layout)
    block_solvers = [
        build_subsolver(system.diagonal_block("primal"), {"kind": "ilu", "drop_tol": 1.0e-6, "fill_factor": 30.0})
    ]
    for idx in range(8):
        block_name = f"lam{idx}"
        schur = lumped_schur_complement(system, primal_block="primal", multiplier_block=block_name, shift=1.0e-8)
        block_solvers.append(build_subsolver(schur, {"kind": "direct"}))

    preconditioner = BlockTriangularPreconditioner(system, tuple(block_solvers), lower=True)
    solver = ScipyKrylovSolver(KrylovOptions(method="gmres", rtol=1.0e-10, maxiter=300, restart=80))
    sol, report = solver.solve(system.matrix, system.rhs, preconditioner=preconditioner)
    residual = np.asarray(system.matrix @ sol - rhs, dtype=float).reshape(-1)

    assert report.converged
    assert float(np.linalg.norm(residual, ord=np.inf)) < 1.0e-8


def test_newton_block_backend_uses_field_layout_for_multi_constraint_system():
    _, _, mat, rhs = _make_saddle_matrix(n_primal=14, n_constraints=8, seed=8)
    solver = NewtonSolver.__new__(NewtonSolver)
    solver.active_dofs = np.arange(mat.shape[0], dtype=int)
    solver._reduced_field_names = np.asarray(["u"] * 14 + [f"lam{i}" for i in range(8)], dtype=object)
    solver.lp = SimpleNamespace(backend="block", tol=1.0e-10, maxit=300)

    block_subsolvers = {"primal": {"kind": "ilu", "drop_tol": 1.0e-6, "fill_factor": 30.0}}
    block_approximations = {
        f"lam{i}": (lambda system, name=f"lam{i}": lumped_schur_complement(
            system,
            primal_block="primal",
            multiplier_block=name,
            shift=1.0e-8,
        ))
        for i in range(8)
    }
    for idx in range(8):
        block_subsolvers[f"lam{idx}"] = {"kind": "direct"}

    solver.set_block_linear_solver(
        blocks=[("primal", ("u",))] + [(f"lam{i}", (f"lam{i}",)) for i in range(8)],
        method="gmres",
        preconditioner="lower_triangular",
        include_remaining=False,
        block_subsolvers=block_subsolvers,
        block_approximations=block_approximations,
    )

    sol = solver._solve_linear_system(mat, rhs)
    ref = np.linalg.solve(mat.toarray(), rhs)

    np.testing.assert_allclose(sol, ref, atol=1.0e-8, rtol=1.0e-8)
    assert tuple(solver._block_linear_last_report.get("layout_blocks", ())) == tuple(
        ["primal"] + [f"lam{i}" for i in range(8)]
    )


def test_newton_block_backend_falls_back_when_pruning_drops_a_configured_block():
    _, _, mat, rhs = _make_saddle_matrix(n_primal=10, n_constraints=2, seed=10)
    solver = NewtonSolver.__new__(NewtonSolver)
    solver.active_dofs = np.arange(10, dtype=int)
    solver._reduced_field_names = np.asarray(["u"] * 10, dtype=object)
    solver.lp = SimpleNamespace(backend="scipy", tol=1.0e-10, maxit=300)

    fallback_called = {"value": False}

    def _fake_scipy(A, b, *, allow_shift=True):
        fallback_called["value"] = True
        return np.linalg.solve(A.toarray(), b)

    solver._solve_linear_system_scipy = _fake_scipy

    solver.set_block_linear_solver(
        blocks=[("primal", ("u",)), ("multiplier", ("lam",))],
        method="gmres",
        preconditioner="lower_triangular",
        include_remaining=False,
        trace=True,
    )

    sol = solver._solve_linear_system(mat[:10, :10].tocsr(), rhs[:10])
    ref = np.linalg.solve(mat[:10, :10].toarray(), rhs[:10])

    assert bool(fallback_called["value"])
    np.testing.assert_allclose(sol, ref, atol=1.0e-10, rtol=1.0e-10)
    assert str(solver._block_linear_last_report.get("method", "")) == "fallback_scipy"
    assert tuple(solver._block_linear_last_report.get("dropped_blocks", ())) == ("multiplier",)
