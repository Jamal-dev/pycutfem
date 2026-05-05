from __future__ import annotations

import numpy as np
import pytest

from examples.biofilms.benchmarks.FSI.paper1_benchmark5_jonas_shear import (
    _build_bcs,
    _build_forms,
    _create_problem,
    _set_snapshots,
    build_jonas_shear_benchmark,
)
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, NewtonSolver
from pycutfem.ufl.expressions import Constant


def test_benchmark5_jonas_shear_smoke_assemble(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "jit_cache_cpp"))

    bench = build_jonas_shear_benchmark()
    problem = _create_problem(2)
    _set_snapshots(problem, bench)

    forms = _build_forms(problem, bench, qdeg=6)
    bcs, bcs_homog = _build_bcs(bench)
    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=problem["dh"],
        mixed_element=problem["me"],
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=1.0e-8, max_newton_iter=1, print_level=0),
        lin_params=LinearSolverParameters(backend="scipy", tol=1.0e-12, maxit=5000),
        quad_order=6,
        backend="cpp",
    )

    dt_c = Constant(float(bench.dt))
    aux_functions = {"dt": dt_c}
    bcs_now = solver._freeze_bcs(solver.bcs, float(bench.dt))
    problem["dh"].apply_bcs(
        bcs_now,
        problem["v_k"],
        problem["p_k"],
        problem["vS_k"],
        problem["u_k"],
        problem["alpha_k"],
        problem["B_k"],
        problem["mu_k"],
    )

    coeffs = {
        problem["v_k"].name: problem["v_k"],
        problem["p_k"].name: problem["p_k"],
        problem["vS_k"].name: problem["vS_k"],
        problem["u_k"].name: problem["u_k"],
        problem["alpha_k"].name: problem["alpha_k"],
        problem["B_k"].name: problem["B_k"],
        problem["mu_k"].name: problem["mu_k"],
        problem["v_n"].name: problem["v_n"],
        problem["p_n"].name: problem["p_n"],
        problem["vS_n"].name: problem["vS_n"],
        problem["u_n"].name: problem["u_n"],
        problem["alpha_n"].name: problem["alpha_n"],
        problem["B_n"].name: problem["B_n"],
        problem["mu_n"].name: problem["mu_n"],
        "dt": aux_functions["dt"],
    }

    A_red, R_red = solver._assemble_system_reduced(coeffs, need_matrix=True)
    A = A_red.toarray()
    R = np.asarray(R_red, dtype=float)

    assert A.shape[0] == A.shape[1] > 0
    assert R.shape == (A.shape[0],)
    assert np.isfinite(A).all()
    assert np.isfinite(R).all()
    assert np.linalg.norm(A, ord=np.inf) > 0.0
    assert np.linalg.norm(R, ord=np.inf) > 0.0
