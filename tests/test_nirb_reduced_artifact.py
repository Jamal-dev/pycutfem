import numpy as np
import pytest

from examples.NIRB.reduced_artifact import FullyReducedNIRBArtifact


def _artifact() -> FullyReducedNIRBArtifact:
    return FullyReducedNIRBArtifact(
        interface_load_basis=np.eye(2),
        interface_disp_basis=np.eye(3, 2),
        interface_mass=np.ones(3),
        solid_to_interface_linear=np.ones((2, 2)),
        solid_to_interface_quadratic=np.ones((2, 3)),
        solid_to_interface_bias=np.zeros(2),
        mesh_stiffness=np.eye(4),
        mesh_interface_coupling=np.ones((4, 2)),
        fluid_basis=np.eye(5, 3),
        fluid_sample_elements=np.array([0, 2, 4], dtype=int),
        fluid_sample_rows=np.array([1, 3], dtype=int),
        fluid_element_weights=np.ones(3),
        fluid_row_weights=np.ones(2),
        reaction_matrix=np.ones((2, 3)),
        dvms_sample_layout=np.array([[0, 0], [2, 1]], dtype=int),
        validation_training_steps=np.array([226, 227], dtype=int),
        validation_heldout_steps=np.array([236], dtype=int),
        validation_error_budget=np.array([1.0e-2, 5.0e-3]),
    )


def test_fully_reduced_artifact_round_trips_npz(tmp_path) -> None:
    path = tmp_path / "fully_reduced.npz"
    original = _artifact()
    original.save(path)

    loaded = FullyReducedNIRBArtifact.load(path)

    for key, value in original.arrays().items():
        np.testing.assert_allclose(loaded.arrays()[key], value)


def test_fully_reduced_artifact_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="reaction matrix rows"):
        FullyReducedNIRBArtifact(
            interface_load_basis=np.eye(2),
            interface_disp_basis=np.eye(3, 2),
            interface_mass=np.ones(3),
            solid_to_interface_linear=np.ones((2, 2)),
            solid_to_interface_quadratic=np.ones((2, 3)),
            solid_to_interface_bias=np.zeros(2),
            mesh_stiffness=np.eye(4),
            mesh_interface_coupling=np.ones((4, 2)),
            fluid_basis=np.eye(5, 3),
            fluid_sample_elements=np.array([0], dtype=int),
            fluid_sample_rows=np.array([1], dtype=int),
            fluid_element_weights=np.ones(1),
            fluid_row_weights=np.ones(1),
            reaction_matrix=np.ones((3, 3)),
        )
