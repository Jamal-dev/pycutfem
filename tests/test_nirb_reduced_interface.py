from pathlib import Path

import numpy as np

from pycutfem.mor.snapshots import SnapshotBatch
from pycutfem.nirb.coupling import NIRBSolidPredictor
from pycutfem.nirb.offline import OfflineConfig, RegressionConfig, run_offline_pipeline
from pycutfem.nirb.reduced_interface import (
    ReducedIQNILS,
    ReducedInterfaceDecoder,
    ReducedInterfaceSpace,
    ReducedTransfer,
)


def test_reduced_interface_space_projects_with_mass_matrix() -> None:
    basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 0.0],
        ]
    )
    mass = np.array([2.0, 0.5, 1.0])
    space = ReducedInterfaceSpace(basis=basis, mass=mass)
    coefficients = np.array([0.25, -0.5])
    values = space.reconstruct(coefficients)

    np.testing.assert_allclose(space.project(values), coefficients, atol=1.0e-12)
    assert space.norm(coefficients) == space.norm(values)


def test_reduced_transfer_matches_projected_full_transfer() -> None:
    source = ReducedInterfaceSpace(
        basis=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        name="source",
    )
    target = ReducedInterfaceSpace(
        basis=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        name="target",
    )
    full_transfer = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
        ]
    )
    reduced_transfer = ReducedTransfer.from_full_transfer(
        source=source,
        target=target,
        full_transfer=full_transfer,
    )
    source_coeffs = np.array([0.3, -0.2])
    expected = target.project(full_transfer @ source.reconstruct(source_coeffs))

    np.testing.assert_allclose(reduced_transfer.apply(source_coeffs), expected, atol=1.0e-12)


def test_nirb_predictor_can_return_reduced_interface_without_full_decode(tmp_path: Path) -> None:
    reduced_forces = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    displacements = np.vstack(
        [
            reduced_forces[0] + 2.0 * reduced_forces[1],
            -reduced_forces[0],
            0.5 * reduced_forces[0] - reduced_forces[1],
            2.0 * reduced_forces[1],
        ]
    )

    dataset_path = tmp_path / "dataset.npz"
    model_path = tmp_path / "model.pkl"
    SnapshotBatch(interface_forces=reduced_forces, full_displacements=displacements).save(dataset_path)
    model = run_offline_pipeline(
        OfflineConfig(
            dataset_path=str(dataset_path),
            model_path=str(model_path),
            force_modes=2,
            displacement_modes=2,
            regression=RegressionConfig(kind="tps_rbf"),
            use_quadratic_decoder=False,
        )
    )

    restriction = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    interface_space = ReducedInterfaceSpace(basis=np.eye(2), name="solid-interface")
    reduced_decoder = ReducedInterfaceDecoder.from_full_decoder(
        linear_basis=model.decoder.linear_basis,
        quadratic_basis=model.decoder.quadratic_basis,
        mean=model.decoder.mean,
        output_space=interface_space,
        restriction_matrix=restriction,
        feature_map=model.decoder.feature_map,
    )
    predictor = NIRBSolidPredictor(model=model, reduced_interface_decoder=reduced_decoder)

    force_coeffs = model.force_basis.project(reduced_forces[:, 2].reshape(-1, 1))[:, 0]
    prediction = predictor.predict_reduced_interface_from_force_coefficients(force_coeffs)
    full_prediction = model.predict_full(reduced_forces[:, 2])[:, 0]

    assert prediction.full_displacement is None
    assert prediction.interface_displacement is None
    assert prediction.reduced_displacement is not None
    assert prediction.reduced_interface_displacement is not None
    np.testing.assert_allclose(
        interface_space.reconstruct(prediction.reduced_interface_displacement),
        restriction @ full_prediction,
        atol=1.0e-8,
    )


def test_reduced_iqnils_first_step_is_relaxed_picard() -> None:
    update = ReducedIQNILS(omega=0.25)
    current = np.array([1.0, 2.0])
    returned = np.array([3.0, 0.0])

    next_values = update.next(current, returned)

    np.testing.assert_allclose(next_values, current + 0.25 * (returned - current))
    assert len(update.x_history) == 1
    update.finalize_step()
    assert len(update.x_history) == 0
