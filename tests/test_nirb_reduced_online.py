import numpy as np

from examples.NIRB.reduced_fluid import ReducedFluidSolveResult
from examples.NIRB.reduced_mesh import ReducedMeshMotionOperator, ReducedMeshMotionState
from examples.NIRB.reduced_online import ReducedFSIState, ReducedOnlineFSISolver
from pycutfem.mor.nirb.reduced_spaces import ReducedIQNILS, ReducedSpace


def test_reduced_online_fsi_solver_converges_without_full_reconstruction() -> None:
    load_space = ReducedSpace(basis=np.eye(1), name="load")
    displacement_space = ReducedSpace(basis=np.eye(1), name="disp")
    mesh_operator = ReducedMeshMotionOperator(
        stiffness=np.eye(1),
        interface_coupling=-np.eye(1),
        dt=0.1,
        bossak_alpha=-0.3,
    )

    def solid_solve(load_coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        displacement = 0.5 * np.asarray(load_coefficients, dtype=float).reshape(-1)
        return displacement.copy(), displacement.copy()

    def fluid_solve(load_coefficients: np.ndarray, mesh_result) -> ReducedFluidSolveResult:
        del load_coefficients
        reaction = 2.0 * np.asarray(mesh_result.q, dtype=float).reshape(-1)
        return ReducedFluidSolveResult(
            coefficients=reaction.copy(),
            residual_norm=0.0,
            iterations=1,
            converged=True,
            trajectory=({"iteration": 1.0, "residual_norm": 0.0, "step_norm": 0.0},),
            reaction_coefficients=reaction,
        )

    initial_state = ReducedFSIState(
        load=np.array([1.0]),
        solid=np.zeros(1),
        interface_displacement=np.zeros(1),
        mesh=np.zeros(1),
        mesh_velocity=np.zeros(1),
        mesh_acceleration=np.zeros(1),
        fluid=np.zeros(1),
        reaction_load=np.zeros(1),
        mesh_history=ReducedMeshMotionState.zeros(1),
    )
    solver = ReducedOnlineFSISolver(
        load_space=load_space,
        displacement_space=displacement_space,
        solid_solve=solid_solve,
        mesh_operator=mesh_operator,
        fluid_solve=fluid_solve,
        iqn=ReducedIQNILS(omega=1.0),
        coupling_abs_tol=1.0e-12,
        coupling_rel_tol=1.0e-12,
        max_coupling_iterations=5,
    )

    result = solver.run_step(initial_state)

    assert result.converged
    assert len(result.iterations) == 2
    assert result.timers["forbidden_full_call_count"] == 0
    assert result.timers["full_reconstruction_s"] == 0.0
    np.testing.assert_allclose(result.state.load, [1.0], atol=1.0e-12)
