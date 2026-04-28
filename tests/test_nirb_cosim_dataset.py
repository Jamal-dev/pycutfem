from pathlib import Path

import numpy as np

from pycutfem.nirb.coupling import NIRBSolidPredictor
from pycutfem.nirb.dataset import load_cosim_snapshot_batch, load_dataset
from pycutfem.nirb.offline import OfflineConfig, RegressionConfig, run_offline_pipeline


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
    dataset = load_dataset(tmp_path)

    assert np.allclose(batch.interface_forces, 1.0)
    assert np.allclose(dataset.forces, 1.0)
    assert np.array_equal(batch.subiterations, np.array([1, 2, 1]))
    assert np.array_equal(batch.converged, np.array([False, True, True]))
    assert np.allclose(batch.solid_times, np.array([0.1, 0.2, 0.3]))


def test_nirb_solid_predictor_loads_model_and_checks_shape(tmp_path: Path):
    forces = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0]])
    displacements = np.vstack([forces[0] + forces[1], 2.0 * forces[0]])
    dataset_path = tmp_path / "dataset.npz"
    model_path = tmp_path / "model.pkl"
    np.savez_compressed(
        dataset_path,
        interface_forces=forces,
        full_displacements=displacements,
        parameters=None,
        times=None,
        subiterations=None,
        converged=None,
        solid_times=None,
        fluid_times=None,
        metadata=np.array("{}", dtype=object),
    )

    run_offline_pipeline(
        OfflineConfig(
            dataset_path=str(dataset_path),
            model_path=str(model_path),
            force_modes=2,
            displacement_modes=2,
            regression=RegressionConfig(kind="tps_rbf"),
            use_quadratic_decoder=False,
        )
    )

    predictor = NIRBSolidPredictor.from_path(model_path, full_shape=(1, 2))
    prediction = predictor.predict(forces[:, 1])
    assert prediction.full_displacement.shape == (2,)
    assert prediction.elapsed_s >= 0.0
