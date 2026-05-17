from __future__ import annotations

import numpy as np
import pytest

from examples.NIRB.fluid_gnat import FluidGNATSystem
from examples.NIRB.reduced_fluid import ReducedFluidNativeOnlineSpec
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.integration.quadrature import volume
from pycutfem.mor import (
    BoundConstraintSpec,
    NativeStateUpdateKernelCall,
    NativeSparseMatrix,
    native_kernel_metadata_from_runner,
    solve_native_bound_constrained_deim_online_gauss_newton,
    solve_native_bound_constrained_galerkin_online_gauss_newton,
    solve_native_bound_constrained_online_gauss_newton,
    solve_native_deim_online_gauss_newton,
    solve_native_online_gauss_newton,
)
from pycutfem.mor.online_gauss_newton import _infer_coefficient_arg_names, native_online_convergence_status
from pycutfem.state import QuadratureLayout, StateRegistry
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_triangles


def _have_cpp_backend() -> bool:
    try:
        import pybind11  # noqa: F401

        return True
    except Exception:
        return False


def test_default_coefficient_inference_supports_mixed_current_fields() -> None:
    names = _infer_coefficient_arg_names(
        ("gdofs_map", "u_k_loc", "p_k_loc", "u_n_loc"),
        ("p_k_loc", "u_k_loc", "p_n_loc"),
    )

    assert names == ("u_k_loc", "p_k_loc")


def test_native_online_convergence_status_separates_lspg_optimality_from_residual() -> None:
    result = type(
        "Result",
        (),
        {
            "converged": False,
            "residual_norm": 5.0e-4,
            "optimality_norm": 1.5e-4,
            "coefficients": np.array([1.0, -2.0]),
            "step_norm_history": np.array([1.0e-12]),
        },
    )()

    strict = native_online_convergence_status(result, residual_tol=1.0e-4)
    relaxed_optimality = native_online_convergence_status(
        result,
        residual_tol=1.0e-4,
        optimality_tol=2.0e-4,
    )

    assert not strict.ok
    assert not strict.residual_ok
    assert not strict.optimality_ok
    assert relaxed_optimality.ok
    assert not relaxed_optimality.residual_ok
    assert relaxed_optimality.optimality_ok


def test_native_online_convergence_status_requires_requested_optimality_gate() -> None:
    result = type(
        "Result",
        (),
        {
            "converged": False,
            "residual_norm": 1.0e-8,
            "optimality_norm": 1.0e-2,
            "coefficients": np.array([1.0]),
            "step_norm_history": np.array([0.0]),
        },
    )()

    status = native_online_convergence_status(
        result,
        residual_tol=1.0e-4,
        optimality_tol=1.0e-4,
    )

    assert status.residual_ok
    assert not status.optimality_ok
    assert not status.ok


def _scalar_problem():
    nodes, elems, edges, corners = structured_triangles(
        1.0,
        1.0,
        nx_quads=1,
        ny_quads=1,
        poly_order=1,
    )
    mesh = Mesh(nodes, elems, edges, corners, element_type="tri", poly_order=1)
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")
    return mesh, dh


def _compile_native_pair(dh, residual_integral, tangent_integral):
    element_ids = np.arange(int(dh.mixed_element.mesh.n_elements), dtype=np.int32)
    compiler = FormCompiler(dh, quadrature_order=4, backend="cpp")
    residual_runner, residual_funcs, residual_args, _gdofs = compiler._prepare_volume_jit_kernel(
        residual_integral,
        element_ids=element_ids,
        full_local_layout=True,
    )
    tangent_runner, tangent_funcs, tangent_args, _ = compiler._prepare_volume_jit_kernel(
        tangent_integral,
        element_ids=element_ids,
        full_local_layout=True,
    )
    # Setup-only pybind calls seed live coefficient arrays. The native solve loop
    # then mutates those arrays directly and does not call Python again.
    residual_runner(residual_funcs, residual_args)
    tangent_runner(tangent_funcs, tangent_args)
    return residual_runner, residual_args, tangent_runner, tangent_args


def _nonlinear_native_pair():
    mesh, dh = _scalar_problem()
    uh = Function("uh", "u", dh)
    uh.nodal_values[:] = 1.0
    v = TestFunction("u", dh)
    du = TrialFunction("u", dh)
    residual = ((uh * uh) - Constant(4.0)) * v * dx(metadata={"q": 4})
    tangent = Constant(2.0) * uh * du * v * dx(metadata={"q": 4})
    return mesh, dh, *_compile_native_pair(dh, residual, tangent)


def _linear_overdetermined_native_pair():
    mesh, dh = _scalar_problem()
    uh = Function("uh", "u", dh)
    uh.nodal_values[:] = 0.0
    source = Function("source", "u", dh)
    source.nodal_values[:] = np.linspace(1.0, 2.0, int(dh.total_dofs))
    v = TestFunction("u", dh)
    du = TrialFunction("u", dh)
    residual = (uh - source) * v * dx(metadata={"q": 4})
    tangent = du * v * dx(metadata={"q": 4})
    return mesh, dh, *_compile_native_pair(dh, residual, tangent)


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_native_online_gauss_newton_solves_generated_ufl_nonlinear_loop(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_online_nonlinear"))
    _mesh, dh, residual_runner, residual_args, tangent_runner, tangent_args = _nonlinear_native_pair()

    result = solve_native_online_gauss_newton(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_param_order=residual_runner.param_order,
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_param_order=tangent_runner.param_order,
        tangent_static_args=tangent_args,
        trial_basis=np.ones((int(dh.total_dofs), 1), dtype=float),
        offset=np.zeros(int(dh.total_dofs), dtype=float),
        initial_coefficients=np.array([1.0], dtype=float),
        row_dofs=np.arange(int(dh.total_dofs), dtype=np.int64),
        coefficient_arg_names=("u_uh_loc",),
        max_iterations=8,
        residual_tol=1.0e-11,
        line_search=True,
        adaptive_damping=True,
    )

    np.testing.assert_allclose(result.coefficients, np.array([2.0]), rtol=1.0e-12, atol=1.0e-12)
    assert result.converged
    assert result.backend == "cpp_native_online"
    assert result.linear_solver in {"qr", "svd"}
    assert result.residual_norm <= 1.0e-11
    assert result.optimality_norm <= 1.0e-11
    assert result.step_norm_history.shape == result.line_search_alpha_history.shape
    assert result.step_norm_history.shape == result.damping_history.shape
    assert result.optimality_norm_history.size >= result.iterations
    assert int(result.timing_counters["assemblies"]) >= int(result.iterations)
    assert int(result.timing_counters["kernel_calls"]) == 2 * int(result.timing_counters["assemblies"])
    np.testing.assert_allclose(residual_args["u_uh_loc"], 2.0, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(tangent_args["u_uh_loc"], 2.0, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_native_online_gauss_newton_converges_by_lspg_optimality_with_nonzero_residual(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_online_lspg_optimality"))
    _mesh, dh, residual_runner, residual_args, tangent_runner, tangent_args = _linear_overdetermined_native_pair()

    result = solve_native_online_gauss_newton(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_param_order=residual_runner.param_order,
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_param_order=tangent_runner.param_order,
        tangent_static_args=tangent_args,
        trial_basis=np.ones((int(dh.total_dofs), 1), dtype=float),
        offset=np.zeros(int(dh.total_dofs), dtype=float),
        initial_coefficients=np.array([0.0], dtype=float),
        row_dofs=np.arange(int(dh.total_dofs), dtype=np.int64),
        coefficient_arg_names=("u_uh_loc",),
        max_iterations=4,
        residual_tol=1.0e-10,
        line_search=True,
    )

    assert result.converged
    assert result.residual_norm > 1.0e-4
    assert result.optimality_norm <= 1.0e-10 * max(1.0, result.residual_norm)


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_native_online_gauss_newton_reference_regularization_biases_branch(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_online_reference_regularization"))
    _mesh, dh, residual_runner, residual_args, tangent_runner, tangent_args = _linear_overdetermined_native_pair()
    n_dofs = int(dh.total_dofs)

    common = dict(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_param_order=residual_runner.param_order,
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_param_order=tangent_runner.param_order,
        tangent_static_args=tangent_args,
        trial_basis=np.ones((n_dofs, 1), dtype=float),
        offset=np.zeros(n_dofs, dtype=float),
        initial_coefficients=np.array([0.0], dtype=float),
        row_dofs=np.arange(n_dofs, dtype=np.int64),
        coefficient_arg_names=("u_uh_loc",),
        max_iterations=8,
        residual_tol=1.0e-12,
        optimality_tol=1.0e-10,
        line_search=True,
    )

    unregularized = solve_native_online_gauss_newton(**common)
    regularized = solve_native_online_gauss_newton(
        **common,
        reference_coefficients=np.array([0.25], dtype=float),
        reference_weight=100.0,
    )

    assert unregularized.converged
    assert regularized.converged
    assert regularized.coefficients[0] < unregularized.coefficients[0]
    assert abs(regularized.coefficients[0] - 0.25) < abs(unregularized.coefficients[0] - 0.25)


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_native_online_gauss_newton_unconstrained_trust_region_limits_state_jump(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_online_unconstrained_trust"))
    _mesh, dh, residual_runner, residual_args, tangent_runner, tangent_args = _nonlinear_native_pair()
    n_dofs = int(dh.total_dofs)
    initial = np.array([1.0], dtype=float)

    result = solve_native_online_gauss_newton(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_param_order=residual_runner.param_order,
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_param_order=tangent_runner.param_order,
        tangent_static_args=tangent_args,
        trial_basis=np.ones((n_dofs, 1), dtype=float),
        offset=np.zeros(n_dofs, dtype=float),
        initial_coefficients=initial,
        row_dofs=np.arange(n_dofs, dtype=np.int64),
        coefficient_arg_names=("u_uh_loc",),
        max_iterations=4,
        residual_tol=1.0e-14,
        optimality_tol=1.0e-14,
        step_tol=1.0e-14,
        line_search=True,
        adaptive_damping=False,
        reference_coefficients=initial,
        max_step_norm=0.25,
        max_reference_distance=0.25,
        state_merit_weight=1.0,
        require_residual_convergence=True,
    )

    assert not result.converged
    assert float(result.timing_counters["final_branch_distance"]) <= 0.25 + 1.0e-12
    assert int(result.timing_counters["trust_region_clipped_steps"]) > 0
    assert result.residual_norm > 1.0e-3


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_native_online_gauss_newton_supports_gnat_lift_and_weights(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_online_gnat"))
    _mesh, dh, residual_runner, residual_args, tangent_runner, tangent_args = _nonlinear_native_pair()

    result = solve_native_online_gauss_newton(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_param_order=residual_runner.param_order,
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_param_order=tangent_runner.param_order,
        tangent_static_args=tangent_args,
        trial_basis=np.ones((int(dh.total_dofs), 1), dtype=float),
        offset=np.zeros(int(dh.total_dofs), dtype=float),
        initial_coefficients=np.array([1.1], dtype=float),
        row_dofs=np.array([0, 2, 3], dtype=np.int64),
        coefficient_arg_names=("u_uh_loc",),
        element_weights=np.array([0.75, 1.25], dtype=float),
        row_weights=np.array([1.0, 0.5, 2.0], dtype=float),
        gnat_lift=np.array([[1.0, -0.25, 0.5], [0.2, 1.0, -0.75]], dtype=float),
        damping=1.0e-10,
        max_iterations=10,
        residual_tol=1.0e-11,
        line_search=True,
    )

    np.testing.assert_allclose(result.coefficients, np.array([2.0]), rtol=1.0e-12, atol=1.0e-12)
    assert result.converged
    assert result.residual_norm <= 1.0e-11


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_native_online_gauss_newton_accepts_sparse_gnat_lift(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_online_sparse_gnat"))
    _mesh, dh, residual_runner, residual_args, tangent_runner, tangent_args = _nonlinear_native_pair()
    dense_lift = np.array([[1.0, -0.25, 0.5], [0.2, 1.0, -0.75]], dtype=float)

    result = solve_native_online_gauss_newton(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_param_order=residual_runner.param_order,
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_param_order=tangent_runner.param_order,
        tangent_static_args=tangent_args,
        trial_basis=np.ones((int(dh.total_dofs), 1), dtype=float),
        offset=np.zeros(int(dh.total_dofs), dtype=float),
        initial_coefficients=np.array([1.1], dtype=float),
        row_dofs=np.array([0, 2, 3], dtype=np.int64),
        coefficient_arg_names=("u_uh_loc",),
        element_weights=np.array([0.75, 1.25], dtype=float),
        row_weights=np.array([1.0, 0.5, 2.0], dtype=float),
        gnat_lift=NativeSparseMatrix.from_dense(dense_lift),
        damping=1.0e-10,
        max_iterations=10,
        residual_tol=1.0e-11,
        line_search=True,
    )

    np.testing.assert_allclose(result.coefficients, np.array([2.0]), rtol=1.0e-12, atol=1.0e-12)
    assert result.converged
    assert result.residual_norm <= 1.0e-11
    assert int(result.timing_counters["sparse_lift_nonzeros"]) == int(np.count_nonzero(dense_lift))
    assert int(result.timing_counters["sparse_lift_applications"]) >= int(result.iterations)


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
@pytest.mark.parametrize("constraint_method", ["pdas", "ipm"])
def test_native_bound_constrained_online_gauss_newton_enforces_decoded_bounds(
    constraint_method: str,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"mor_online_constrained_{constraint_method}"))
    _mesh, dh, residual_runner, residual_args, tangent_runner, tangent_args = _nonlinear_native_pair()
    n_dofs = int(dh.total_dofs)
    upper = 1.5
    bounds = BoundConstraintSpec(
        rows=np.arange(n_dofs, dtype=np.int64),
        lower=np.zeros(n_dofs, dtype=float),
        upper=np.full(n_dofs, upper, dtype=float),
    )

    result = solve_native_bound_constrained_online_gauss_newton(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_param_order=residual_runner.param_order,
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_param_order=tangent_runner.param_order,
        tangent_static_args=tangent_args,
        trial_basis=np.ones((n_dofs, 1), dtype=float),
        offset=np.zeros(n_dofs, dtype=float),
        initial_coefficients=np.array([1.25], dtype=float),
        row_dofs=np.arange(n_dofs, dtype=np.int64),
        bound_constraints=bounds,
        constraint_method=constraint_method,
        coefficient_arg_names=("u_uh_loc",),
        max_iterations=30,
        residual_tol=1.0e-14,
        step_tol=1.0e-8 if constraint_method == "ipm" else 1.0e-12,
        line_search=True,
        adaptive_damping=True,
        barrier_initial=1.0e-3,
        barrier_decay=0.2,
        feasibility_tol=1.0e-10,
    )

    assert result.converged
    assert result.backend == f"cpp_native_{constraint_method}_online"
    assert result.coefficients[0] <= upper + 1.0e-10
    assert result.coefficients[0] > 1.45
    if constraint_method == "pdas":
        np.testing.assert_allclose(result.coefficients, np.array([upper]), rtol=1.0e-10, atol=1.0e-10)
    assert float(result.timing_counters["bound_violation"]) <= 1.0e-10


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_native_bound_constrained_online_gauss_newton_accepts_sparse_constraint_matrix(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_online_sparse_constraints"))
    _mesh, dh, residual_runner, residual_args, tangent_runner, tangent_args = _nonlinear_native_pair()
    n_dofs = int(dh.total_dofs)
    trial_basis = np.ones((n_dofs, 1), dtype=float)
    upper = 1.5
    bounds = BoundConstraintSpec(
        rows=np.arange(n_dofs, dtype=np.int64),
        lower=np.zeros(n_dofs, dtype=float),
        upper=np.full(n_dofs, upper, dtype=float),
    ).reduce(trial_basis=trial_basis, offset=np.zeros(n_dofs, dtype=float), sparse=True)

    result = solve_native_bound_constrained_online_gauss_newton(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_param_order=residual_runner.param_order,
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_param_order=tangent_runner.param_order,
        tangent_static_args=tangent_args,
        trial_basis=trial_basis,
        offset=np.zeros(n_dofs, dtype=float),
        initial_coefficients=np.array([1.25], dtype=float),
        row_dofs=np.arange(n_dofs, dtype=np.int64),
        bound_constraints=bounds,
        constraint_method="pdas",
        coefficient_arg_names=("u_uh_loc",),
        max_iterations=20,
        residual_tol=1.0e-14,
        line_search=True,
        adaptive_damping=True,
    )

    assert result.converged
    np.testing.assert_allclose(result.coefficients, np.array([upper]), rtol=1.0e-10, atol=1.0e-10)
    assert int(result.timing_counters["bound_constraint_nonzeros"]) == n_dofs
    assert int(result.timing_counters["bound_constraint_rows"]) == n_dofs
    assert float(result.timing_counters["bound_violation"]) <= 1.0e-10


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_native_bound_constrained_online_gauss_newton_branch_guard_limits_state_jump(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_online_branch_guard"))
    _mesh, dh, residual_runner, residual_args, tangent_runner, tangent_args = _nonlinear_native_pair()
    n_dofs = int(dh.total_dofs)
    bounds = BoundConstraintSpec(
        rows=np.arange(n_dofs, dtype=np.int64),
        lower=np.zeros(n_dofs, dtype=float),
        upper=np.full(n_dofs, 3.0, dtype=float),
    )
    initial = np.array([1.0], dtype=float)

    result = solve_native_bound_constrained_online_gauss_newton(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_param_order=residual_runner.param_order,
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_param_order=tangent_runner.param_order,
        tangent_static_args=tangent_args,
        trial_basis=np.ones((n_dofs, 1), dtype=float),
        offset=np.zeros(n_dofs, dtype=float),
        initial_coefficients=initial,
        row_dofs=np.arange(n_dofs, dtype=np.int64),
        bound_constraints=bounds,
        constraint_method="pdas",
        coefficient_arg_names=("u_uh_loc",),
        max_iterations=4,
        residual_tol=1.0e-14,
        step_tol=1.0e-14,
        line_search=True,
        adaptive_damping=False,
        max_line_search=8,
        reference_coefficients=initial,
        max_reference_distance=0.5,
        state_merit_weight=1.0,
        require_residual_convergence=True,
    )

    assert not result.converged
    assert float(result.timing_counters["final_branch_distance"]) <= 0.5 + 1.0e-12
    assert int(result.timing_counters["branch_guard_rejections"]) > 0
    assert result.residual_norm > 1.0e-3


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_native_bound_constrained_galerkin_online_gauss_newton_solves_full_free_projection(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_online_galerkin_constrained"))
    _mesh, dh, residual_runner, residual_args, tangent_runner, tangent_args = _nonlinear_native_pair()
    n_dofs = int(dh.total_dofs)
    trial_basis = np.eye(n_dofs, dtype=float)
    bounds = BoundConstraintSpec(
        rows=np.arange(n_dofs, dtype=np.int64),
        lower=np.zeros(n_dofs, dtype=float),
        upper=np.full(n_dofs, 3.0, dtype=float),
    )

    result = solve_native_bound_constrained_galerkin_online_gauss_newton(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_param_order=residual_runner.param_order,
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_param_order=tangent_runner.param_order,
        tangent_static_args=tangent_args,
        trial_basis=trial_basis,
        offset=np.zeros(n_dofs, dtype=float),
        initial_coefficients=np.ones(n_dofs, dtype=float),
        row_dofs=np.arange(n_dofs, dtype=np.int64),
        bound_constraints=bounds,
        constraint_method="pdas",
        coefficient_arg_names=("u_uh_loc",),
        max_iterations=10,
        residual_tol=1.0e-11,
        line_search=True,
        adaptive_damping=True,
    )

    assert result.converged
    assert result.backend == "cpp_native_galerkin_pdas_online"
    np.testing.assert_allclose(result.coefficients, np.full(n_dofs, 2.0), rtol=1.0e-10, atol=1.0e-10)
    assert float(result.timing_counters["bound_violation"]) <= 1.0e-10
    assert int(result.timing_counters["assemblies"]) >= int(result.iterations)


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_native_deim_online_gauss_newton_solves_generated_ufl_loop(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_online_deim"))
    _mesh, dh, residual_runner, residual_args, tangent_runner, tangent_args = _nonlinear_native_pair()
    n_rows = int(dh.total_dofs)
    selected_basis = np.eye(n_rows, dtype=float)
    selected_basis[0, 0] = 2.0
    if n_rows > 1:
        selected_basis[0, 1] = 0.25
        selected_basis[1, 1] = 1.5
    # residual_terms.T maps interpolation coefficients back to the sampled
    # residual target.  With residual_terms=A.T this path is algebraically the
    # same as sampled LSPG, but it exercises the native DEIM solve/composition
    # and the selected UFL feature/Jacobian evaluation inside C++.
    residual_terms = selected_basis.T

    result = solve_native_deim_online_gauss_newton(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_param_order=residual_runner.param_order,
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_param_order=tangent_runner.param_order,
        tangent_static_args=tangent_args,
        trial_basis=np.ones((n_rows, 1), dtype=float),
        offset=np.zeros(n_rows, dtype=float),
        initial_coefficients=np.array([1.0], dtype=float),
        row_dofs=np.arange(n_rows, dtype=np.int64),
        selected_basis=selected_basis,
        residual_terms=residual_terms,
        coefficient_arg_names=("u_uh_loc",),
        max_iterations=8,
        residual_tol=1.0e-11,
        line_search=True,
        adaptive_damping=True,
    )

    np.testing.assert_allclose(result.coefficients, np.array([2.0]), rtol=1.0e-12, atol=1.0e-12)
    assert result.converged
    assert result.backend == "cpp_native_deim_online"
    assert result.residual_norm <= 1.0e-11
    assert int(result.timing_counters["deim_interpolation_applications"]) >= int(result.iterations)
    assert int(result.timing_counters["deim_composition_applications"]) >= int(result.iterations)


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
@pytest.mark.parametrize("constraint_method", ["pdas", "ipm"])
def test_native_bound_constrained_deim_online_gauss_newton_enforces_decoded_bounds(
    constraint_method: str,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"mor_online_deim_constrained_{constraint_method}"))
    _mesh, dh, residual_runner, residual_args, tangent_runner, tangent_args = _nonlinear_native_pair()
    n_rows = int(dh.total_dofs)
    selected_basis = np.eye(n_rows, dtype=float)
    selected_basis[0, 0] = 2.0
    if n_rows > 1:
        selected_basis[0, 1] = 0.25
        selected_basis[1, 1] = 1.5
    residual_terms = selected_basis.T
    upper = 1.5
    bounds = BoundConstraintSpec(
        rows=np.arange(n_rows, dtype=np.int64),
        lower=np.zeros(n_rows, dtype=float),
        upper=np.full(n_rows, upper, dtype=float),
    )

    result = solve_native_bound_constrained_deim_online_gauss_newton(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_param_order=residual_runner.param_order,
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_param_order=tangent_runner.param_order,
        tangent_static_args=tangent_args,
        trial_basis=np.ones((n_rows, 1), dtype=float),
        offset=np.zeros(n_rows, dtype=float),
        initial_coefficients=np.array([1.25], dtype=float),
        row_dofs=np.arange(n_rows, dtype=np.int64),
        selected_basis=selected_basis,
        residual_terms=residual_terms,
        bound_constraints=bounds,
        constraint_method=constraint_method,
        coefficient_arg_names=("u_uh_loc",),
        max_iterations=30,
        residual_tol=1.0e-14,
        step_tol=1.0e-8 if constraint_method == "ipm" else 1.0e-12,
        line_search=True,
        adaptive_damping=True,
        barrier_initial=1.0e-3,
        barrier_decay=0.2,
        feasibility_tol=1.0e-10,
    )

    assert result.converged
    assert result.backend == f"cpp_native_deim_{constraint_method}_online"
    assert result.coefficients[0] <= upper + 1.0e-10
    assert result.coefficients[0] > 1.45
    assert int(result.timing_counters["deim_interpolation_applications"]) >= int(result.iterations)
    assert int(result.timing_counters["deim_composition_applications"]) >= int(result.iterations)
    assert float(result.timing_counters["bound_violation"]) <= 1.0e-10


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_native_online_gauss_newton_updates_quadrature_state_after_rejected_line_search(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_online_qstate"))
    mesh, dh = _scalar_problem()
    uh = Function("uh", "u", dh)
    uh.nodal_values[:] = 0.25
    v = TestFunction("u", dh)
    du = TrialFunction("u", dh)
    qref, qw = volume(mesh.element_type, 4)
    layout = QuadratureLayout(
        entity_kind="volume_cell",
        cell_type=mesh.element_type,
        quadrature_order=4,
        reference_points=np.asarray(qref, dtype=float),
        reference_weights=np.asarray(qw, dtype=float),
    )
    registry = StateRegistry()
    qfield = registry.register_quadrature("source_q", layout=layout, n_entities=int(mesh.n_elements))
    qsource = qfield.coefficient(jit_name="online_qsource")
    residual = ((uh * uh) - Constant(4.0) + Constant(1.0e-12) * qsource) * v * dx(metadata={"q": 4})
    tangent = Constant(2.0) * uh * du * v * dx(metadata={"q": 4})
    residual_runner, residual_args, tangent_runner, tangent_args = _compile_native_pair(dh, residual, tangent)

    result = solve_native_online_gauss_newton(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_param_order=residual_runner.param_order,
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_param_order=tangent_runner.param_order,
        tangent_static_args=tangent_args,
        trial_basis=np.ones((int(dh.total_dofs), 1), dtype=float),
        offset=np.zeros(int(dh.total_dofs), dtype=float),
        initial_coefficients=np.array([0.25], dtype=float),
        row_dofs=np.arange(int(dh.total_dofs), dtype=np.int64),
        coefficient_arg_names=("u_uh_loc",),
        residual_state_updates=(
            {
                "name": "online_qsource",
                "basis": np.ones((int(mesh.n_elements) * int(layout.n_qp), 1), dtype=float),
                "offset": np.zeros(int(mesh.n_elements) * int(layout.n_qp), dtype=float),
            },
        ),
        max_iterations=12,
        residual_tol=1.0e-10,
        line_search=True,
        max_line_search=8,
    )

    assert result.converged
    assert np.any((result.line_search_alpha_history > 0.0) & (result.line_search_alpha_history < 1.0))
    np.testing.assert_allclose(result.coefficients, np.array([2.0]), rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(
        residual_args["online_qsource"],
        float(result.coefficients[0]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_native_online_gauss_newton_executes_symbolic_state_update_kernel(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_online_symbolic_state"))
    mesh, dh = _scalar_problem()
    uh = Function("uh", "u", dh)
    uh.nodal_values[:] = 1.0
    v = TestFunction("u", dh)
    du = TrialFunction("u", dh)
    registry = StateRegistry()
    cell_source = registry.register_cell("source", n_cells=int(mesh.n_elements))
    source = cell_source.coefficient(jit_name="online_cell_source")
    residual = ((uh * uh) - Constant(4.0) + Constant(1.0e-12) * source) * v * dx(metadata={"q": 4})
    tangent = Constant(2.0) * uh * du * v * dx(metadata={"q": 4})
    residual_runner, residual_args, tangent_runner, tangent_args = _compile_native_pair(dh, residual, tangent)

    element_ids = np.arange(int(mesh.n_elements), dtype=np.int32)
    compiler = FormCompiler(dh, quadrature_order=4, backend="cpp")
    update_integral = Constant(2.0) * uh * dx(metadata={"q": 4})
    update_runner, update_funcs, update_args, _gdofs = compiler._prepare_volume_jit_kernel(
        update_integral,
        element_ids=element_ids,
        full_local_layout=True,
    )
    update_runner(update_funcs, update_args)

    result = solve_native_online_gauss_newton(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_param_order=residual_runner.param_order,
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_param_order=tangent_runner.param_order,
        tangent_static_args=tangent_args,
        trial_basis=np.ones((int(dh.total_dofs), 1), dtype=float),
        offset=np.zeros(int(dh.total_dofs), dtype=float),
        initial_coefficients=np.array([1.0], dtype=float),
        row_dofs=np.arange(int(dh.total_dofs), dtype=np.int64),
        coefficient_arg_names=("u_uh_loc",),
        residual_symbolic_state_updates=(
            NativeStateUpdateKernelCall(
                metadata_capsule=native_kernel_metadata_from_runner(update_runner),
                param_order=tuple(update_runner.param_order),
                static_args=update_args,
                target_name="online_cell_source",
            ),
        ),
        max_iterations=8,
        residual_tol=1.0e-11,
        line_search=True,
    )

    np.testing.assert_allclose(result.coefficients, np.array([2.0]), rtol=1.0e-10, atol=1.0e-10)
    assert result.converged
    assert int(result.timing_counters["symbolic_state_update_calls"]) >= int(result.iterations)
    np.testing.assert_allclose(residual_args["online_cell_source"], 2.0, rtol=1.0e-10, atol=1.0e-10)


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_nirb_native_online_spec_maps_mor_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_online_nirb_spec"))
    _mesh, dh, residual_runner, residual_args, tangent_runner, tangent_args = _nonlinear_native_pair()
    spec = ReducedFluidNativeOnlineSpec(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_param_order=tuple(residual_runner.param_order),
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_param_order=tuple(tangent_runner.param_order),
        tangent_static_args=tangent_args,
        trial_basis=np.ones((int(dh.total_dofs), 1), dtype=float),
        offset=np.zeros(int(dh.total_dofs), dtype=float),
        row_dofs=np.arange(int(dh.total_dofs), dtype=np.int64),
        coefficient_arg_names=("u_uh_loc",),
        objective="sampled_lspg",
    )

    result = spec.solve(
        np.array([1.0], dtype=float),
        max_iterations=8,
        residual_tol=1.0e-11,
        line_search=True,
    )

    np.testing.assert_allclose(result.coefficients, np.array([2.0]), rtol=1.0e-12, atol=1.0e-12)
    assert result.converged
    assert result.metadata is not None
    assert result.metadata["backend"] == "cpp_native_online"
    assert "timing_counters" in result.metadata
    assert len(result.trajectory) >= 1


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_fluid_gnat_system_cpp_step_matches_numpy(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_online_gnat_step"))
    J = np.array([[2.0, -1.0], [0.5, 3.0], [4.0, 1.5]], dtype=float)
    r = np.array([1.0, -2.0, 0.25], dtype=float)
    expected, *_ = np.linalg.lstsq(J, -r, rcond=None)
    system = FluidGNATSystem(
        coefficients=np.zeros(2, dtype=float),
        sampled_residual=r.copy(),
        sampled_trial_jacobian=J.copy(),
        residual_coefficients=r.copy(),
        gnat_trial_jacobian=J.copy(),
        normal_matrix=J.T @ J,
        normal_rhs=-(J.T @ r),
        estimated_residual_norm=float(np.linalg.norm(r)),
        row_dofs=np.arange(3, dtype=int),
        element_ids=np.arange(1, dtype=int),
    )

    np.testing.assert_allclose(system.gauss_newton_step(backend="cpp"), expected, rtol=1.0e-12, atol=1.0e-12)
