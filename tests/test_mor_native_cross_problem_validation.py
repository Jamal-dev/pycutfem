from __future__ import annotations

import numpy as np
import pytest

from tests.test_mor_online_gauss_newton import _compile_native_pair, _scalar_problem

from pycutfem.mor import (
    NativeGnatTargetSpec,
    NativeKernelReference,
    NativeReducedArtifact,
    load_native_reduced_artifact,
    native_kernel_metadata_from_runner,
    solve_native_deim_online_gauss_newton,
)
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction
from pycutfem.ufl.measures import dx


def _have_cpp_backend() -> bool:
    try:
        import pybind11  # noqa: F401

        return True
    except Exception:
        return False


def _cubic_native_pair():
    mesh, dh = _scalar_problem()
    uh = Function("uh", "u", dh)
    uh.nodal_values[:] = 1.2
    v = TestFunction("u", dh)
    du = TrialFunction("u", dh)
    residual = ((uh * uh * uh) - Constant(8.0)) * v * dx(metadata={"q": 4})
    tangent = Constant(3.0) * uh * uh * du * v * dx(metadata={"q": 4})
    return mesh, dh, *_compile_native_pair(dh, residual, tangent)


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_loaded_native_reduced_artifact_solves_independent_cubic_problem(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_cross_cubic_artifact"))
    _mesh, dh, residual_runner, residual_args, tangent_runner, tangent_args = _cubic_native_pair()
    n_dofs = int(dh.total_dofs)
    artifact = NativeReducedArtifact(
        problem_id="cubic_reaction_smoke",
        trial_basis=np.ones((n_dofs, 1), dtype=float),
        offset=np.zeros(n_dofs, dtype=float),
        residual_kernel=NativeKernelReference(
            kernel_id="cubic_residual",
            abi="native-kernel-v1",
            param_order=tuple(residual_runner.param_order),
        ),
        tangent_kernel=NativeKernelReference(
            kernel_id="cubic_tangent",
            abi="native-kernel-v1",
            param_order=tuple(tangent_runner.param_order),
        ),
        target=NativeGnatTargetSpec(row_dofs=np.arange(n_dofs, dtype=np.int64), objective="sampled_lspg"),
        solver_options={"max_iterations": 12, "residual_tol": 1.0e-11, "line_search": True, "adaptive_damping": True},
        metadata={"coefficient_arg_names": ("u_uh_loc",)},
    )
    path = tmp_path / "cubic_native_artifact.npz"
    artifact.save(path)
    loaded = load_native_reduced_artifact(path)
    runtime = loaded.instantiate(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_static_args=tangent_args,
    )

    result = runtime.solve(np.array([1.2], dtype=float))

    np.testing.assert_allclose(result.coefficients, np.array([2.0]), rtol=1.0e-11, atol=1.0e-11)
    assert result.converged
    assert result.backend == "cpp_native_online"
    assert result.residual_norm <= 1.0e-11


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_native_deim_online_path_solves_independent_cubic_problem(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_cross_cubic_deim"))
    _mesh, dh, residual_runner, residual_args, tangent_runner, tangent_args = _cubic_native_pair()
    n_dofs = int(dh.total_dofs)
    selected_basis = np.eye(n_dofs, dtype=float)
    selected_basis[0, 0] = 1.75
    if n_dofs > 1:
        selected_basis[0, 1] = -0.2
        selected_basis[1, 1] = 1.25
    residual_terms = selected_basis.T

    result = solve_native_deim_online_gauss_newton(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_param_order=residual_runner.param_order,
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_param_order=tangent_runner.param_order,
        tangent_static_args=tangent_args,
        trial_basis=np.ones((n_dofs, 1), dtype=float),
        offset=np.zeros(n_dofs, dtype=float),
        initial_coefficients=np.array([1.2], dtype=float),
        row_dofs=np.arange(n_dofs, dtype=np.int64),
        selected_basis=selected_basis,
        residual_terms=residual_terms,
        coefficient_arg_names=("u_uh_loc",),
        max_iterations=12,
        residual_tol=1.0e-11,
        line_search=True,
        adaptive_damping=True,
    )

    np.testing.assert_allclose(result.coefficients, np.array([2.0]), rtol=1.0e-11, atol=1.0e-11)
    assert result.converged
    assert result.backend == "cpp_native_deim_online"
    assert int(result.timing_counters["deim_interpolation_applications"]) >= int(result.iterations)
