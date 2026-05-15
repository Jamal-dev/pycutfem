import numpy as np

from examples.NIRB.reduced_fluid import ReducedFluidDVMSOperator, ReducedFluidSystem
from examples.NIRB.reduced_mesh import (
    ReducedMeshDisplacementMap,
    ReducedMeshMotionOperator,
    ReducedMeshMotionState,
    ReducedMeshSampleEvaluator,
    bossak_displacement_kinematics,
    fit_reduced_mesh_displacement_map,
)


def test_reduced_mesh_motion_solves_small_ale_system_and_updates_bossak() -> None:
    stiffness = np.array([[4.0, 1.0], [1.0, 3.0]])
    coupling = np.array([[2.0], [-1.0]])
    operator = ReducedMeshMotionOperator(stiffness=stiffness, interface_coupling=coupling, dt=0.1, bossak_alpha=-0.3)
    history = ReducedMeshMotionState.zeros(2)

    result = operator.solve(np.array([0.2]), history)
    expected_q = np.linalg.solve(stiffness, -(coupling @ np.array([0.2])))
    expected_v, expected_a = bossak_displacement_kinematics(
        q_curr=expected_q,
        q_prev=np.zeros(2),
        v_prev=np.zeros(2),
        a_prev=np.zeros(2),
        dt=0.1,
        alpha=-0.3,
    )

    np.testing.assert_allclose(result.q, expected_q)
    np.testing.assert_allclose(result.v, expected_v)
    np.testing.assert_allclose(result.a, expected_a)
    assert result.residual_norm < 1.0e-14


def test_reduced_mesh_sample_evaluator_decodes_values_and_gradients() -> None:
    value_basis = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [1.0, -1.0]],
        ]
    )
    grad_basis = np.stack([2.0 * value_basis, -value_basis], axis=2)
    evaluator = ReducedMeshSampleEvaluator(value_basis=value_basis, grad_basis=grad_basis)
    coeffs = np.array([0.25, -0.5])

    np.testing.assert_allclose(evaluator.values(coeffs), np.tensordot(value_basis, coeffs, axes=([-1], [0])))
    np.testing.assert_allclose(evaluator.gradients(coeffs), np.tensordot(grad_basis, coeffs, axes=([-1], [0])))


def test_reduced_mesh_displacement_map_fits_and_round_trips(tmp_path) -> None:
    interface_basis = np.eye(2)
    mesh_basis = np.eye(4, 3)
    interface_snapshots = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.5, -0.5, -1.0],
        ],
        dtype=float,
    )
    true_linear = np.array(
        [
            [2.0, -1.0],
            [0.5, 3.0],
            [-1.0, 0.25],
        ],
        dtype=float,
    )
    true_bias = np.array([0.2, -0.1, 0.3], dtype=float)
    mesh_coeffs = true_bias[:, None] + true_linear @ interface_snapshots
    mesh_snapshots = mesh_basis @ mesh_coeffs

    fitted = fit_reduced_mesh_displacement_map(
        interface_basis=interface_basis,
        interface_mean=None,
        interface_snapshots=interface_snapshots,
        mesh_basis=mesh_basis,
        mesh_mean=None,
        mesh_snapshots=mesh_snapshots,
        fluid_coords_ref=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        interface_coords_ref=np.array([[0.0, 1.0]], dtype=float),
        training_steps=np.array([1, 2, 3, 4], dtype=int),
    )
    assert np.max(fitted.training_relative_errors) < 1.0e-12

    predicted = fitted.predict_nodal_displacement(interface_snapshots[:, 2])
    np.testing.assert_allclose(predicted.reshape(-1), mesh_snapshots[:, 2], atol=1.0e-12)

    path = tmp_path / "mesh_map.npz"
    fitted.save(path)
    loaded = ReducedMeshDisplacementMap.load(path)
    np.testing.assert_allclose(loaded.predict_nodal_displacement(interface_snapshots[:, 3]), mesh_snapshots[:, 3].reshape(2, 2))


def test_reduced_fluid_operator_solves_coefficient_system_without_global_state() -> None:
    target = np.array([0.25, -0.75])
    calls: list[np.ndarray] = []

    def assembler(coefficients: np.ndarray) -> ReducedFluidSystem:
        calls.append(coefficients.copy())
        residual = coefficients - target
        tangent = np.eye(2)
        return ReducedFluidSystem(
            coefficients=coefficients.copy(),
            residual=residual,
            tangent=tangent,
            residual_norm=float(np.linalg.norm(residual)),
            metadata={"backend": "synthetic"},
        )

    operator = ReducedFluidDVMSOperator(
        n_modes=2,
        assembler=assembler,
        reaction_evaluator=lambda coeffs: np.array([coeffs[0] + 2.0 * coeffs[1]]),
        max_iterations=4,
        residual_tol=1.0e-12,
    )

    result = operator.solve(np.zeros(2))

    np.testing.assert_allclose(result.coefficients, target, atol=1.0e-12)
    np.testing.assert_allclose(result.reaction_coefficients, [target[0] + 2.0 * target[1]])
    assert result.converged
    assert len(calls) >= 2
