from pathlib import Path

import numpy as np

from pycutfem.mor.snapshots import SnapshotBatch
from pycutfem.nirb.offline import OfflineConfig, RegressionConfig, run_offline_pipeline


def test_offline_pipeline_accepts_full_interface_restriction_matrix(tmp_path: Path):
    reduced_forces = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    forces = reduced_forces.copy()
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
    interface_matrix_path = tmp_path / "interface.npy"
    SnapshotBatch(interface_forces=forces, full_displacements=displacements).save(dataset_path)

    interface_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    np.save(interface_matrix_path, interface_matrix)

    model = run_offline_pipeline(
        OfflineConfig(
            dataset_path=str(dataset_path),
            model_path=str(model_path),
            force_modes=2,
            displacement_modes=2,
            regression=RegressionConfig(kind="tps_rbf"),
            use_quadratic_decoder=False,
            interface_matrix_path=str(interface_matrix_path),
        )
    )

    predicted_full = model.predict_full(forces)
    predicted_interface = model.predict_interface(forces)
    assert np.allclose(predicted_interface, interface_matrix @ predicted_full, atol=1.0e-8)
