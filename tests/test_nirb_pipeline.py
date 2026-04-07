import json
from pathlib import Path

import numpy as np

from pycutfem.mor.snapshots import SnapshotBatch
from pycutfem.nirb.offline import OfflineConfig, RegressionConfig, run_offline_pipeline
from pycutfem.nirb.online import OnlineConfig, run_online_pipeline
from pycutfem.nirb.validation import ValidationConfig, validate_rom


def test_synthetic_nirb_pipeline_round_trips_through_training_prediction_and_validation(tmp_path: Path):
    rng = np.random.default_rng(7)
    reduced_forces = rng.normal(size=(2, 80))

    force_mean = np.array([[0.25], [-0.1], [0.0], [0.0]])
    force_basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    forces = force_mean + force_basis @ reduced_forces

    reduced_displacements = np.vstack(
        [
            0.5 + 1.5 * reduced_forces[0] - 0.25 * reduced_forces[1] + 0.75 * reduced_forces[0] * reduced_forces[1],
            -0.2 + 0.5 * reduced_forces[1] ** 2,
        ]
    )
    displacement_mean = np.array([[0.1], [0.2], [0.0], [0.0]])
    displacement_basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    displacements = displacement_mean + displacement_basis @ reduced_displacements

    dataset_path = tmp_path / "synthetic_snapshots.npz"
    SnapshotBatch(
        interface_forces=forces,
        full_displacements=displacements,
        metadata={"benchmark": "synthetic"},
    ).save(dataset_path)

    model_path = tmp_path / "model.pkl"
    predictions_path = tmp_path / "predictions.npz"
    reference_path = tmp_path / "reference.npz"
    metrics_path = tmp_path / "metrics.json"

    model = run_offline_pipeline(
        OfflineConfig(
            dataset_path=str(dataset_path),
            model_path=str(model_path),
            force_modes=2,
            displacement_modes=2,
            regression=RegressionConfig(kind="poly_lasso", standardize_inputs=False),
            use_quadratic_decoder=False,
        )
    )
    full_predictions = model.predict_full(forces)

    assert np.allclose(full_predictions, displacements, atol=1.0e-8)

    np.savez_compressed(tmp_path / "forces.npz", forces=forces)
    online_predictions = run_online_pipeline(
        OnlineConfig(
            model_path=str(model_path),
            forces_path=str(tmp_path / "forces.npz"),
            predictions_path=str(predictions_path),
            interface_only=False,
        )
    )
    np.savez_compressed(reference_path, reference=displacements)

    result = validate_rom(
        ValidationConfig(
            reference_path=str(reference_path),
            prediction_path=str(predictions_path),
            metrics_path=str(metrics_path),
            thresholds={"mean_sample_l2_error": 1.0e-8, "max_online_relative_displacement_error": 1.0e-8},
            fom_iterations=[10, 12, 11],
            rom_iterations=[10, 12, 11],
            fom_solid_time=10.0,
            rom_solid_time=0.1,
            fom_total_time=50.0,
            rom_total_time=10.0,
        )
    )

    assert np.allclose(online_predictions, displacements, atol=1.0e-8)
    assert result["passes"] == {
        "mean_sample_l2_error": True,
        "max_online_relative_displacement_error": True,
    }

    serialized = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert serialized["metrics"]["solid_speedup"] == 100.0
    assert serialized["metrics"]["overall_speedup"] == 5.0
