from pathlib import Path

import numpy as np

from examples.utils.nirb import NIRBSolidPredictor
from examples.utils.nirb import load_cosim_snapshot_batch
from pycutfem.mor.snapshots import NamedSnapshotBatch
from pycutfem.mor.nirb.dataset import load_dataset
from pycutfem.mor.nirb.offline import OfflineConfig, RegressionConfig, run_offline_pipeline


def test_load_cosim_snapshot_batch_uses_structure_input_loads(tmp_path: Path):
    co_sim = tmp_path / "coSimData"
    co_sim.mkdir()
    np.save(co_sim / "load_guess_data.npy", np.ones((2, 3)))
    np.save(co_sim / "load_data.npy", 2.0 * np.ones((2, 3)))
    np.save(co_sim / "disp_data.npy", np.arange(12, dtype=float).reshape(4, 3))
    np.save(co_sim / "structure_time.npy", np.array([0.1, 0.2, 0.3]))
    (tmp_path / "snapshot_metadata.csv").write_text(
        "step,time_s,coupling_iter,converged\n"
        "1,0.1,1,False\n"
        "1,0.1,2,True\n"
        "2,0.2,1,True\n",
        encoding="utf-8",
    )

    batch = load_cosim_snapshot_batch(tmp_path)
    dataset_path = tmp_path / "generic_snapshots.npz"
    batch.save(dataset_path)
    dataset = load_dataset(dataset_path, input_field="interface_load", output_field="solid_displacement")

    assert np.allclose(batch["interface_load"], 1.0)
    assert np.allclose(dataset.input_snapshots, 1.0)
    assert [row["subiteration"] for row in batch.sample_metadata] == [1, 2, 1]
    assert np.array_equal(batch.converged, np.array([False, True, True]))
    assert [row["solid_time"] for row in batch.sample_metadata] == [0.1, 0.2, 0.3]


def test_nirb_solid_predictor_loads_model_and_checks_shape(tmp_path: Path):
    forces = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0]])
    displacements = np.vstack([forces[0] + forces[1], 2.0 * forces[0]])
    dataset_path = tmp_path / "dataset.npz"
    model_path = tmp_path / "model.pkl"
    NamedSnapshotBatch(fields={"input": forces, "output": displacements}).save(dataset_path)

    run_offline_pipeline(
        OfflineConfig(
            dataset_path=str(dataset_path),
            model_path=str(model_path),
            input_modes=2,
            output_modes=2,
            regression=RegressionConfig(kind="tps_rbf"),
            use_quadratic_decoder=False,
        )
    )

    predictor = NIRBSolidPredictor.from_path(model_path, full_shape=(1, 2))
    prediction = predictor.predict(forces[:, 1])
    assert prediction.full_displacement.shape == (2,)
    assert prediction.elapsed_s >= 0.0
