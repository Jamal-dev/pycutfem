from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pycutfem.mor.cross_validation import run_mode_cross_validation
from pycutfem.mor.io import save_config
from pycutfem.mor.metrics import (
    accumulated_iteration_overhead,
    max_online_relative_displacement_error,
    mean_sample_l2_error,
)
from pycutfem.mor.regressors import ThinPlateSplineRBF
from pycutfem.mor.snapshots import NamedSnapshotBatch
from pycutfem.mor.nirb.offline import OfflineConfig, RegressionConfig, run_offline_pipeline

from examples.NIRB.common import (
    ARTERIAL_REPO_URL,
    compute_tube_stress_strain,
    default_artifacts_root,
    default_cache_root,
    dump_json,
    ensure_git_clone,
    plot_cumulative_iterations,
    plot_heatmap,
    plot_xy,
    read_arterial_vtk_series,
    repo_root,
)


EXAMPLE1_FORCE_MODES = 3
EXAMPLE1_DISPLACEMENT_MODES = 9


def _repo_relative(path: Path) -> str:
    return str(path.relative_to(repo_root()))


def prepare_example1_artifacts(cache_root: Path, artifacts_root: Path) -> dict[str, Path]:
    repo = ensure_git_clone(ARTERIAL_REPO_URL, cache_root / "ArterialWallROM")
    example_root = artifacts_root / "example1"
    example_root.mkdir(parents=True, exist_ok=True)

    train_dataset_path = example_root / "train_snapshots.npz"
    NamedSnapshotBatch(
        fields={
            "input": np.load(repo / "FOM_DATA" / "first_mu" / "pres_TRAIN_DATA.npy"),
            "output": np.load(repo / "FOM_DATA" / "first_mu" / "sec_TRAIN_DATA.npy"),
        },
        metadata={"benchmark": "example1", "paper_modes": [EXAMPLE1_FORCE_MODES, EXAMPLE1_DISPLACEMENT_MODES]},
    ).save(train_dataset_path)

    same_forces_path = example_root / "same_mu_forces.npy"
    same_reference_path = example_root / "same_mu_reference.npy"
    other_forces_path = example_root / "other_mu_forces.npy"
    other_reference_path = example_root / "other_mu_reference.npy"
    np.save(same_forces_path, np.load(repo / "FOM_DATA" / "first_mu" / "pressure.npy"))
    np.save(same_reference_path, np.load(repo / "FOM_DATA" / "first_mu" / "diameter.npy"))
    np.save(other_forces_path, np.load(repo / "FOM_DATA" / "second_mu" / "pressure.npy"))
    np.save(other_reference_path, np.load(repo / "FOM_DATA" / "second_mu" / "diameter.npy"))

    same_coupled_path = example_root / "same_mu_coupled_rom_fom.npz"
    other_coupled_path = example_root / "other_mu_coupled_rom_fom.npz"
    if not same_coupled_path.exists():
        np.savez_compressed(same_coupled_path, **read_arterial_vtk_series(repo / "output_first_mu_rom"))
    if not other_coupled_path.exists():
        np.savez_compressed(other_coupled_path, **read_arterial_vtk_series(repo / "output_second_mu_rom"))

    config_path = Path(__file__).resolve().parent / "configs" / "example1.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    save_config(
        {
            "dataset_path": _repo_relative(train_dataset_path),
            "model_path": _repo_relative(example_root / "model.pkl"),
            "input_modes": EXAMPLE1_FORCE_MODES,
            "output_modes": EXAMPLE1_DISPLACEMENT_MODES,
            "regression": {"kind": "tps_rbf", "smoothing": 0.0},
            "use_quadratic_decoder": False,
            "metadata": {"benchmark": "example1"},
        },
        config_path,
    )

    return {
        "repo": repo,
        "train_dataset": train_dataset_path,
        "same_forces": same_forces_path,
        "same_reference": same_reference_path,
        "other_forces": other_forces_path,
        "other_reference": other_reference_path,
        "same_coupled": same_coupled_path,
        "other_coupled": other_coupled_path,
        "model": example_root / "model.pkl",
        "results": example_root / "results.json",
    }


def run_example1(
    *,
    cache_root: Path | None = None,
    artifacts_root: Path | None = None,
) -> dict[str, Any]:
    cache_root = default_cache_root() if cache_root is None else Path(cache_root)
    artifacts_root = default_artifacts_root() if artifacts_root is None else Path(artifacts_root)
    paths = prepare_example1_artifacts(cache_root=cache_root, artifacts_root=artifacts_root)

    train_snapshots = NamedSnapshotBatch.load(paths["train_dataset"])
    train_forces = np.asarray(train_snapshots["input"], dtype=float)
    train_displacements = np.asarray(train_snapshots["output"], dtype=float)

    mode_sweep = run_mode_cross_validation(
        train_forces,
        train_displacements,
        input_modes=list(range(1, 7)),
        output_modes=list(range(1, 13)),
        regressor_factory=lambda: ThinPlateSplineRBF(),
        test_fraction=0.2,
        random_state=0,
        use_quadratic_decoder=False,
    )
    dump_json(mode_sweep.entries, paths["results"].with_name("mode_sweep.json"))
    plot_heatmap(
        mode_sweep.entries,
        x_attr="input_modes",
        y_attr="output_modes",
        value_attr="validation_error",
        path=paths["results"].with_name("mode_sweep_heatmap.png"),
        title="Example 1 Held-Out Error",
        xlabel="Pressure modes",
        ylabel="Section modes",
    )

    model = run_offline_pipeline(
        OfflineConfig(
            dataset_path=str(paths["train_dataset"]),
            model_path=str(paths["model"]),
            input_modes=EXAMPLE1_FORCE_MODES,
            output_modes=EXAMPLE1_DISPLACEMENT_MODES,
            regression=RegressionConfig(kind="tps_rbf"),
            use_quadratic_decoder=False,
            metadata={"benchmark": "example1"},
        )
    )

    same_forces = np.load(paths["same_forces"])
    same_reference = np.load(paths["same_reference"])
    other_forces = np.load(paths["other_forces"])
    other_reference = np.load(paths["other_reference"])

    started = time.perf_counter()
    same_prediction = model.predict(same_forces)
    same_elapsed = time.perf_counter() - started
    started = time.perf_counter()
    other_prediction = model.predict(other_forces)
    other_elapsed = time.perf_counter() - started

    np.savez_compressed(paths["results"].with_name("same_mu_prediction.npz"), predictions=same_prediction)
    np.savez_compressed(paths["results"].with_name("other_mu_prediction.npz"), predictions=other_prediction)

    same_coupled = np.load(paths["same_coupled"])
    other_coupled = np.load(paths["other_coupled"])

    coupled_same_pressure = np.asarray(same_coupled["pressure"], dtype=float)
    coupled_same_diameter = np.asarray(same_coupled["diameter"], dtype=float)
    coupled_same_iters = np.asarray(same_coupled["iters"], dtype=float)
    coupled_other_pressure = np.asarray(other_coupled["pressure"], dtype=float)
    coupled_other_diameter = np.asarray(other_coupled["diameter"], dtype=float)
    coupled_other_iters = np.asarray(other_coupled["iters"], dtype=float)

    plot_xy(
        np.arange(1, same_reference.shape[1] + 1, dtype=float) * 0.1,
        [
            ("FOM-FOM", same_forces[0, :]),
            ("ROM-FOM", coupled_same_pressure[0, :]),
        ],
        path=paths["results"].with_name("same_mu_inlet_pressure.png"),
        title="Example 1 Same-Parameter Inlet Pressure",
        xlabel="Time [s]",
        ylabel="Pressure [Pa]",
    )
    plot_xy(
        np.arange(1, other_reference.shape[1] + 1, dtype=float) * 0.1,
        [
            ("FOM-FOM", other_forces[0, :]),
            ("ROM-FOM", coupled_other_pressure[0, :]),
        ],
        path=paths["results"].with_name("other_mu_inlet_pressure.png"),
        title="Example 1 Extrapolated Inlet Pressure",
        xlabel="Time [s]",
        ylabel="Pressure [Pa]",
    )
    plot_cumulative_iterations(
        np.load(paths["repo"] / "FOM_DATA" / "first_mu" / "iters.npy"),
        coupled_same_iters,
        path=paths["results"].with_name("same_mu_iterations.png"),
        title="Example 1 Cumulative Coupling Iterations",
    )

    fom_strain, fom_stress = compute_tube_stress_strain(same_reference, same_forces)
    coupled_strain, coupled_stress = compute_tube_stress_strain(coupled_same_diameter, coupled_same_pressure)
    order = np.argsort(fom_strain.ravel())
    order_coupled = np.argsort(coupled_strain.ravel())
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(fom_strain.ravel()[order], fom_stress.ravel()[order], label="FOM-FOM")
    axis.plot(
        coupled_strain.ravel()[order_coupled],
        coupled_stress.ravel()[order_coupled],
        label="ROM-FOM",
    )
    axis.set_title("Example 1 Stress-Strain Reconstruction")
    axis.set_xlabel("Strain")
    axis.set_ylabel("Stress [Pa]")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["results"].with_name("stress_strain.png"), dpi=200)
    plt.close(figure)

    result = {
        "benchmark": "example1",
        "paper_selected_modes": {
            "pressure_modes": EXAMPLE1_FORCE_MODES,
            "section_modes": EXAMPLE1_DISPLACEMENT_MODES,
        },
        "mode_sweep": {
            "best_validation_error": min(entry.validation_error for entry in mode_sweep.entries),
            "best_entry": min(mode_sweep.entries, key=lambda entry: entry.validation_error),
        },
        "structural_rom_against_fom": {
            "same_mu_mean_sample_l2_error": mean_sample_l2_error(same_reference, same_prediction),
            "same_mu_max_relative_error": max_online_relative_displacement_error(same_reference, same_prediction),
            "other_mu_mean_sample_l2_error": mean_sample_l2_error(other_reference, other_prediction),
            "other_mu_max_relative_error": max_online_relative_displacement_error(other_reference, other_prediction),
            "same_mu_mean_inference_time_per_snapshot": same_elapsed / same_reference.shape[1],
            "other_mu_mean_inference_time_per_snapshot": other_elapsed / other_reference.shape[1],
        },
        "published_coupled_results": {
            "same_mu_pressure_mean_sample_l2_error": mean_sample_l2_error(same_forces, coupled_same_pressure),
            "same_mu_diameter_mean_sample_l2_error": mean_sample_l2_error(same_reference, coupled_same_diameter),
            "same_mu_pressure_max_relative_error": max_online_relative_displacement_error(same_forces, coupled_same_pressure),
            "same_mu_diameter_max_relative_error": max_online_relative_displacement_error(same_reference, coupled_same_diameter),
            "same_mu_iteration_overhead": accumulated_iteration_overhead(
                np.load(paths["repo"] / "FOM_DATA" / "first_mu" / "iters.npy"),
                coupled_same_iters,
            ),
            "other_mu_pressure_mean_sample_l2_error": mean_sample_l2_error(other_forces, coupled_other_pressure),
            "other_mu_diameter_mean_sample_l2_error": mean_sample_l2_error(other_reference, coupled_other_diameter),
            "other_mu_pressure_max_relative_error": max_online_relative_displacement_error(other_forces, coupled_other_pressure),
            "other_mu_diameter_max_relative_error": max_online_relative_displacement_error(other_reference, coupled_other_diameter),
            "other_mu_iteration_overhead": accumulated_iteration_overhead(
                np.load(paths["repo"] / "FOM_DATA" / "second_mu" / "iters.npy"),
                coupled_other_iters,
            ),
        },
        "artifacts": {name: str(path) for name, path in paths.items() if name != "repo"},
    }
    dump_json(result, paths["results"])
    return result
