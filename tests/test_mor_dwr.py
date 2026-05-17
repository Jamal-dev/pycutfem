from __future__ import annotations

import numpy as np
import pytest

from pycutfem.mor import (
    DWRReducedTrajectory,
    NativeAdjointDWRSpec,
    NativeKernelReference,
    NativeReducedArtifact,
    TransientResidualDependencySpec,
    QoIFunctionalSpec,
    assemble_qoi_gradient,
    certify_dual_weighted_residual,
    certify_dual_weighted_residual_from_artifact_trajectory,
    check_qoi_gradient,
    dominant_dwr_contributions,
    dual_weighted_residual_estimate,
    dwr_certification_guard,
    evaluate_qoi_functional,
    finite_difference_gradient,
    load_dwr_reduced_trajectory,
    load_native_reduced_artifact,
    reduced_qoi_gradient_from_full,
    save_dwr_reduced_trajectory,
    solve_discrete_adjoint,
    solve_reduced_discrete_adjoint,
    solve_transpose_system,
)


def _have_cpp_backend() -> bool:
    try:
        import pybind11  # noqa: F401

        return True
    except Exception:
        return False


def test_discrete_adjoint_backward_sweep_matches_manual_solution() -> None:
    A0 = np.array([[3.0, 0.25], [0.5, 2.0]], dtype=float)
    A1 = np.array([[2.0, -0.1], [0.3, 1.5]], dtype=float)
    M1 = np.array([[-0.7, 0.1], [0.2, -0.4]], dtype=float)
    grad0 = np.array([0.0, 0.0], dtype=float)
    grad1 = np.array([1.0, -0.5], dtype=float)

    result = solve_discrete_adjoint(
        [A0, A1],
        [grad0, grad1],
        previous_state_jacobians=[np.zeros_like(A0), M1],
    )

    z1 = np.linalg.solve(A1.T, grad1)
    z0 = np.linalg.solve(A0.T, grad0 - M1.T @ z1)
    np.testing.assert_allclose(result.adjoints[1], z1, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(result.adjoints[0], z0, rtol=1.0e-12, atol=1.0e-12)
    assert result.backend == "python_adjoint"
    assert np.all(result.residual_norm_history <= 1.0e-12)


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_native_discrete_adjoint_matches_python_backend(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_adjoint_cpp"))
    jacobians = [
        np.array([[4.0, 1.0], [0.5, 3.0]], dtype=float),
        np.array([[2.0, -0.25], [0.75, 2.5]], dtype=float),
    ]
    previous = [
        np.zeros((2, 2), dtype=float),
        np.array([[-1.0, 0.2], [0.1, -0.8]], dtype=float),
    ]
    gradients = [np.array([0.25, 0.0], dtype=float), np.array([1.0, 0.5], dtype=float)]

    py_result = solve_discrete_adjoint(jacobians, gradients, previous_state_jacobians=previous, backend="python")
    cpp_result = solve_discrete_adjoint(jacobians, gradients, previous_state_jacobians=previous, backend="cpp")

    assert cpp_result.backend == "cpp_native_adjoint"
    for z_cpp, z_py in zip(cpp_result.adjoints, py_result.adjoints, strict=True):
        np.testing.assert_allclose(z_cpp, z_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_array_equal(cpp_result.rank_history, py_result.rank_history)


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_native_transpose_solve_matches_numpy(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "mor_adjoint_transpose"))
    J = np.array([[2.0, -1.0], [0.5, 3.0], [4.0, 1.5]], dtype=float)
    rhs = np.array([1.0, -2.0], dtype=float)
    z, metadata = solve_transpose_system(J, rhs, backend="cpp")
    expected, *_ = np.linalg.lstsq(J.T, rhs, rcond=None)

    np.testing.assert_allclose(z, expected, rtol=1.0e-12, atol=1.0e-12)
    assert metadata["backend"] == "cpp_native_adjoint"
    assert metadata["rank"] == 2


def test_dwr_estimate_is_exact_for_linear_residual_error_identity() -> None:
    A = np.array([[3.0, 0.5], [0.25, 2.0]], dtype=float)
    b = np.array([1.0, -0.5], dtype=float)
    c = np.array([2.0, -1.0], dtype=float)
    x_exact = np.linalg.solve(A, b)
    x_rom = np.array([0.1, -0.2], dtype=float)
    residual = A @ x_rom - b
    adjoint = np.linalg.solve(A.T, c)
    qoi_error = float(c @ (x_exact - x_rom))

    estimate = dual_weighted_residual_estimate(
        [residual],
        [adjoint],
        reference_qoi_error=qoi_error,
        effectivity_bounds=(0.999999, 1.000001),
    )

    np.testing.assert_allclose(estimate.estimate, qoi_error, rtol=1.0e-12, atol=1.0e-12)
    assert estimate.passed
    assert estimate.effectivity == pytest.approx(1.0)


def test_artifact_driven_dwr_certification_uses_adjoint_metadata() -> None:
    A = np.array([[3.0, 0.5], [0.25, 2.0]], dtype=float)
    b = np.array([1.0, -0.5], dtype=float)
    c = np.array([2.0, -1.0], dtype=float)
    x_exact = np.linalg.solve(A, b)
    x_rom = np.array([0.1, -0.2], dtype=float)
    residual = A @ x_rom - b
    qoi_error = float(c @ (x_exact - x_rom))
    artifact = NativeReducedArtifact(
        problem_id="linear_certification",
        trial_basis=np.eye(2, dtype=float),
        offset=np.zeros(2, dtype=float),
        residual_kernel=NativeKernelReference(
            kernel_id="residual",
            abi="native-kernel-v1",
            param_order=("gdofs_map", "u_loc"),
        ),
        tangent_kernel=NativeKernelReference(
            kernel_id="tangent",
            abi="native-kernel-v1",
            param_order=("gdofs_map", "u_loc"),
        ),
        adjoint_dwr=NativeAdjointDWRSpec(
            qoi_name="linear_output",
            solver_options={"backend": "python", "rcond": 1.0e-12},
            estimator_options={"effectivity_bounds": (0.999999, 1.000001), "sign": -1.0},
        ),
    )

    certificate = certify_dual_weighted_residual(
        [residual],
        [A],
        [c],
        artifact=artifact,
        reference_qoi_error=qoi_error,
    )

    assert certificate.passed
    assert certificate.qoi_name == "linear_output"
    np.testing.assert_allclose(certificate.estimate.estimate, qoi_error, rtol=1.0e-12, atol=1.0e-12)
    assert certificate.estimate.effectivity == pytest.approx(1.0)
    assert certificate.metadata["artifact_problem_id"] == "linear_certification"
    assert certificate.metadata["rank_ok"] is True


def test_saved_artifact_trajectory_dwr_certification_roundtrip(tmp_path) -> None:
    A = np.array([[3.0, 0.5], [0.25, 2.0]], dtype=float)
    b = np.array([1.0, -0.5], dtype=float)
    c = np.array([2.0, -1.0], dtype=float)
    x_exact = np.linalg.solve(A, b)
    x_rom = np.array([0.1, -0.2], dtype=float)
    residual = A @ x_rom - b
    qoi_error = float(c @ (x_exact - x_rom))
    artifact = NativeReducedArtifact(
        problem_id="saved_linear_certification",
        trial_basis=np.eye(2, dtype=float),
        offset=np.zeros(2, dtype=float),
        residual_kernel=NativeKernelReference(
            kernel_id="residual",
            abi="native-kernel-v1",
            param_order=("gdofs_map", "u_loc"),
        ),
        tangent_kernel=NativeKernelReference(
            kernel_id="tangent",
            abi="native-kernel-v1",
            param_order=("gdofs_map", "u_loc"),
        ),
        adjoint_dwr=NativeAdjointDWRSpec(
            qoi_name="saved_linear_output",
            solver_options={"backend": "python"},
            estimator_options={"effectivity_bounds": (0.999999, 1.000001)},
        ),
    )
    artifact_path = tmp_path / "artifact.npz"
    trajectory_path = tmp_path / "trajectory.npz"
    artifact.save(artifact_path)
    np.savez(
        trajectory_path,
        residuals=residual.reshape(1, -1),
        jacobians=A.reshape(1, *A.shape),
        qoi_gradients=c.reshape(1, -1),
        reference_qoi_error=np.array([qoi_error], dtype=float),
    )

    certificate = certify_dual_weighted_residual_from_artifact_trajectory(
        artifact_path,
        trajectory_path,
    )

    assert certificate.passed
    assert certificate.qoi_name == "saved_linear_output"
    np.testing.assert_allclose(certificate.estimate.estimate, qoi_error, rtol=1.0e-12, atol=1.0e-12)
    assert certificate.metadata["artifact_problem_id"] == "saved_linear_certification"
    assert certificate.metadata["trajectory_source"] == str(trajectory_path)


def test_dwr_reduced_trajectory_save_load_roundtrip(tmp_path) -> None:
    trajectory = DWRReducedTrajectory(
        residuals=np.array([[1.0, -2.0]], dtype=float),
        jacobians=np.array([[[3.0, 0.5], [0.25, 2.0]]], dtype=float),
        qoi_gradients=np.array([[2.0, -1.0]], dtype=float),
        row_weights=np.array([[1.0, 0.5]], dtype=float),
        reference_qoi_error=0.25,
        metadata={"case": "unit"},
    )
    path = tmp_path / "trajectory.npz"

    save_dwr_reduced_trajectory(trajectory, path)
    loaded = load_dwr_reduced_trajectory(path)

    np.testing.assert_allclose(loaded.residuals, trajectory.residuals)
    np.testing.assert_allclose(loaded.jacobians, trajectory.jacobians)
    np.testing.assert_allclose(loaded.qoi_gradients, trajectory.qoi_gradients)
    np.testing.assert_allclose(loaded.row_weights, trajectory.row_weights)
    assert loaded.reference_qoi_error == pytest.approx(0.25)
    assert loaded.metadata["case"] == "unit"


def test_ufl_qoi_value_gradient_and_reduced_projection() -> None:
    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.core.mesh import Mesh
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.ufl.expressions import Function, TestFunction
    from pycutfem.ufl.measures import dx
    from pycutfem.utils.meshgen import structured_quad

    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")
    u = Function("u_k", "u", dof_handler=dh)
    du = TestFunction("u", dof_handler=dh)
    coords = np.asarray(dh.get_dof_coords("u"), dtype=float)
    u.nodal_values[:] = 1.0 + coords[:, 0]
    qoi = u * dx(metadata={"q": 3})

    value = evaluate_qoi_functional(qoi, dof_handler=dh, backend="cpp", name="mass")
    gradient = assemble_qoi_gradient(qoi, u, du, dof_handler=dh, backend="cpp", quad_order=3)

    assert value == pytest.approx(1.5)
    np.testing.assert_allclose(gradient, np.full(int(dh.total_dofs), 0.25), rtol=1.0e-12, atol=1.0e-12)
    basis = np.column_stack([np.ones(int(dh.total_dofs)), np.arange(int(dh.total_dofs), dtype=float)])
    reduced = reduced_qoi_gradient_from_full(gradient, basis)
    np.testing.assert_allclose(reduced, basis.T @ gradient)


def test_dwr_block_localization_identifies_dominant_missing_component() -> None:
    residual = np.array([0.01, -0.02, 10.0, -5.0], dtype=float)
    adjoint = np.array([1.0, 1.0, 0.5, -0.5], dtype=float)
    estimate = dual_weighted_residual_estimate(
        [residual],
        [adjoint],
        row_blocks=[
            {"name": "small", "rows": np.array([0, 1], dtype=np.int64)},
            {"name": "interface", "rows": np.array([2, 3], dtype=np.int64)},
        ],
    )

    assert abs(estimate.block_contributions["interface"]) > 100.0 * abs(estimate.block_contributions["small"])


def test_finite_difference_gradient_and_qoi_spec() -> None:
    spec = QoIFunctionalSpec(name="mass", aggregation="time_integral", fields=("alpha",), tolerance=1.0e-3)
    assert spec.name == "mass"
    x = np.array([0.5, -1.0, 2.0], dtype=float)
    grad = finite_difference_gradient(lambda y: float(y @ y), x, step=1.0e-7)
    np.testing.assert_allclose(grad, 2.0 * x, rtol=1.0e-7, atol=1.0e-7)


def test_reduced_adjoint_and_qoi_gradient_guard_helpers() -> None:
    A = np.array([[2.0, 0.25], [0.1, 3.0]], dtype=float)
    c = np.array([1.0, -0.5], dtype=float)
    basis = np.eye(2)

    full = solve_discrete_adjoint([A], [c])
    reduced = solve_reduced_discrete_adjoint([A], [c], basis)

    np.testing.assert_allclose(reduced.adjoints[0], full.adjoints[0], rtol=1.0e-12, atol=1.0e-12)
    assert reduced.backend == "python_reduced_adjoint"

    x = np.array([0.25, -0.75], dtype=float)
    check = check_qoi_gradient(lambda y: float(y @ y), x, 2.0 * x, step=1.0e-7, tolerance=1.0e-6)
    assert check.passed

    estimate = dual_weighted_residual_estimate(
        [np.array([0.1, -0.2], dtype=float)],
        [np.array([1.0, 2.0], dtype=float)],
        row_blocks=[{"name": "state", "rows": np.array([0, 1], dtype=np.int64)}],
    )
    dominant = dominant_dwr_contributions(estimate)
    assert dominant["steps"][0]["index"] == 0
    guarded = dwr_certification_guard(
        estimate,
        branch_certificate={"passed": True},
        norm_equivalence_certificate={"passed": True},
        gauge_certificate={"passed": True},
        require_norm_equivalence=True,
        require_gauge=True,
        safety_factor=2.0,
    )
    assert guarded.passed
    assert guarded.certified_bound == pytest.approx(2.0 * estimate.absolute_estimate)


def test_artifact_adjoint_dwr_schema_carries_certification_metadata(tmp_path) -> None:
    dependency = TransientResidualDependencySpec(
        current_state=True,
        previous_state=True,
        history_width=1,
        parameter_names=("mu",),
    )
    artifact = NativeReducedArtifact(
        problem_id="adjoint_schema",
        trial_basis=np.eye(2, dtype=float),
        offset=np.zeros(2, dtype=float),
        residual_kernel=NativeKernelReference(
            kernel_id="residual",
            abi="native-kernel-v1",
            param_order=("gdofs_map", "u_loc"),
        ),
        tangent_kernel=NativeKernelReference(
            kernel_id="tangent",
            abi="native-kernel-v1",
            param_order=("gdofs_map", "u_loc"),
        ),
        adjoint_dwr=NativeAdjointDWRSpec(
            qoi_name="mass",
            transient_dependency=dependency.to_dict(),
            checkpoint_policy={"kind": "dense_reduced", "stride": 1},
            field_layout_signature={"fields": ("u", "p")},
            pressure_gauge={"blocks": ({"name": "p", "rows": (1,)},)},
            norm_equivalence_certificate={"passed": True, "lower_constant": 0.5},
            solver_options={"backend": "cpp"},
            estimator_options={"effectivity_bounds": (0.2, 5.0)},
        ),
    )
    payload = artifact.to_dict()
    loaded = NativeReducedArtifact.from_dict(payload)

    assert loaded.adjoint_dwr is not None
    assert loaded.adjoint_dwr.transient_dependency["previous_state"] is True
    assert loaded.adjoint_dwr.checkpoint_policy["stride"] == 1
    assert loaded.adjoint_dwr.norm_equivalence_certificate["passed"] is True

    path = tmp_path / "adjoint_schema.npz"
    artifact.save(path)
    reloaded = load_native_reduced_artifact(path)
    assert reloaded.adjoint_dwr is not None
    assert reloaded.adjoint_dwr.field_layout_signature["fields"] == ("u", "p")
    assert reloaded.adjoint_dwr.pressure_gauge["blocks"][0]["name"] == "p"
