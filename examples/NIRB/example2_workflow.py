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
    online_relative_displacement_error,
    reduced_regression_error,
    snapshot_l2_error,
)
from pycutfem.mor.pod import fit_pod, project_to_basis
from pycutfem.mor.quadratic_manifold import fit_quadratic_decoder
from pycutfem.mor.regressors import PolynomialLassoRegressor
from pycutfem.mor.snapshots import NamedSnapshotBatch
from pycutfem.mor.nirb.offline import OfflineConfig, RegressionConfig, run_offline_pipeline

from examples.NIRB.common import (
    DOUBLE_FLAP_REPO_URL,
    default_artifacts_root,
    default_cache_root,
    download_public_lfs_files,
    dump_json,
    ensure_git_clone,
    list_numeric_files,
    plot_cumulative_iterations,
    plot_xy,
    read_structural_vtk_series,
    reject_outliers,
    reynolds_from_case,
    repo_root,
)


TRAINING_CASES = ("14", "12", "10")
COUPLED_CASES = ("08.3", "10", "12", "13", "14")
EXAMPLE2_FORCE_MODES = 45
EXAMPLE2_DISPLACEMENT_MODES = 9


def _repo_relative(path: Path) -> str:
    return str(path.relative_to(repo_root()))


def _prediction_start(case_name: str) -> int:
    return 100 if case_name in {"08.3", "13"} else 425


def _double_flap_pointer_paths(repo: Path, *, include_online_vtk: bool) -> list[Path]:
    pointers: list[Path] = [
        repo / "coSimData" / "map_used.npy",
        repo / "coSimData" / "coords_interf.npy",
    ]
    for case_name in TRAINING_CASES:
        pointers.extend(
            [
                repo / "Double_E10" / case_name / "coSimData" / "disp_data.npy",
                repo / "Double_E10" / case_name / "coSimData" / "load_data.npy",
                repo / "Double_E10" / case_name / "coSimData" / "iters.npy",
            ]
        )

    timing_files = (
        "fluid_time.npy",
        "increment_time.npy",
        "iters.npy",
        "structure_time.npy",
        "total_solving_time.npy",
        "structure_recons_time.npy",
    )
    for case_name in COUPLED_CASES:
        for file_name in timing_files:
            fom_path = repo / "Double_E10" / case_name / "coSimData" / file_name
            rom_path = repo / "Double_ROM_E10" / f"{case_name}_LASSO" / "coSimData" / file_name
            if fom_path.exists():
                pointers.append(fom_path)
            if rom_path.exists():
                pointers.append(rom_path)

    if include_online_vtk:
        pointers.extend(
            list_numeric_files(repo / "Double_E10" / "08.3" / "vtk_data" / "vtk_output_fsi_csm", "Structure_0_*.vtk")
        )
        pointers.extend(
            list_numeric_files(
                repo / "Double_ROM_E10" / "08.3_LASSO" / "vtk_data" / "vtk_output_fsi_csm",
                "Structure_0_*.vtk",
            )
        )
    return pointers


def _materialize_double_flap_payloads(cache_root: Path, *, include_online_vtk: bool) -> tuple[Path, Path]:
    repo = ensure_git_clone(DOUBLE_FLAP_REPO_URL, cache_root / "DoubleFlap")
    payload_root = cache_root / "DoubleFlap_lfs"
    download_public_lfs_files(
        DOUBLE_FLAP_REPO_URL,
        _double_flap_pointer_paths(repo, include_online_vtk=include_online_vtk),
        source_root=repo,
        destination_root=payload_root,
    )
    return repo, payload_root


def _build_training_snapshots(payload_root: Path) -> tuple[np.ndarray, np.ndarray]:
    force_snapshots: list[np.ndarray] = []
    displacement_snapshots: list[np.ndarray] = []
    for case_name in TRAINING_CASES:
        iterations = np.load(payload_root / "Double_E10" / case_name / "coSimData" / "iters.npy").copy()
        iterations[0] -= 1
        converged = (iterations[:425].cumsum() - 1).astype(int)
        final_snapshot = int(converged[-1] + 1)
        force_snapshots.append(
            np.load(payload_root / "Double_E10" / case_name / "coSimData" / "load_data.npy")[:, :final_snapshot]
        )
        displacement_snapshots.append(
            np.load(payload_root / "Double_E10" / case_name / "coSimData" / "disp_data.npy")[:, :final_snapshot]
        )
    return np.concatenate(force_snapshots, axis=1), np.concatenate(displacement_snapshots, axis=1)


def _evaluate_offline_rom4(
    forces: np.ndarray,
    displacements: np.ndarray,
    *,
    random_state: int = 0,
) -> dict[str, float]:
    ids = np.arange(forces.shape[1])
    rng = np.random.default_rng(random_state)
    train_ids = np.sort(rng.choice(ids, size=int(0.95 * ids.size), replace=False))
    test_ids = np.setdiff1d(ids, train_ids)

    f_train = forces[:, train_ids]
    f_test = forces[:, test_ids]
    u_train = displacements[:, train_ids]
    u_test = displacements[:, test_ids]

    force_basis = fit_pod(f_train, n_modes=EXAMPLE2_FORCE_MODES, center=True)
    decoder = fit_quadratic_decoder(u_train, n_modes=EXAMPLE2_DISPLACEMENT_MODES, center=True)
    reduced_force_train = force_basis.project(f_train).T
    reduced_force_test = force_basis.project(f_test).T
    reduced_disp_train = project_to_basis(u_train, decoder.linear_basis, decoder.mean).T

    regressor = PolynomialLassoRegressor(degree=2, criterion="bic", standardize_inputs=False)
    regressor.fit(reduced_force_train, reduced_disp_train)

    reduced_disp_pred_train = regressor.predict(reduced_force_train).T
    reduced_disp_pred_test = regressor.predict(reduced_force_test).T
    full_pred_test = decoder.decode(reduced_disp_pred_test)

    return {
        "validation_snapshot_l2_error": snapshot_l2_error(u_test, full_pred_test),
        "validation_mean_sample_l2_error": mean_sample_l2_error(u_test, full_pred_test),
        "validation_max_relative_error": max_online_relative_displacement_error(u_test, full_pred_test),
        "training_reduced_regression_l2_error": reduced_regression_error(
            reduced_disp_train.T,
            reduced_disp_pred_train,
        ),
    }


def _load_case_co_sim(payload_root: Path, case_name: str, *, rom: bool) -> Path:
    if rom:
        return payload_root / "Double_ROM_E10" / f"{case_name}_LASSO" / "coSimData"
    return payload_root / "Double_E10" / case_name / "coSimData"


def _compute_speed_metrics(payload_root: Path, case_name: str) -> dict[str, float]:
    fom_root = _load_case_co_sim(payload_root, case_name, rom=False)
    rom_root = _load_case_co_sim(payload_root, case_name, rom=True)
    start_increment = _prediction_start(case_name)

    fom_iterations = np.load(fom_root / "iters.npy")
    rom_iterations = np.load(rom_root / "iters.npy")
    fom_iteration_offset = int(np.cumsum(fom_iterations)[start_increment])
    rom_iteration_offset = int(np.cumsum(rom_iterations)[start_increment])

    fom_structure_times = reject_outliers(np.load(fom_root / "structure_time.npy")[fom_iteration_offset:])
    rom_structure_times = reject_outliers(np.load(rom_root / "structure_time.npy")[rom_iteration_offset:])
    solid_speedup = float(np.mean(fom_structure_times) / np.mean(rom_structure_times))

    reconstruction_time = 0.0
    reconstruction_path = rom_root / "structure_recons_time.npy"
    if reconstruction_path.exists():
        reconstruction_values = np.load(reconstruction_path)
        reconstruction_time = float(np.sum(reconstruction_values))

    fom_total_time = float(np.load(fom_root / "total_solving_time.npy")) - float(
        np.load(fom_root / "increment_time.npy")[:start_increment].sum()
    )
    rom_total_time = float(np.load(rom_root / "total_solving_time.npy")) - float(
        np.load(rom_root / "increment_time.npy")[:start_increment].sum()
    )
    rom_total_time = rom_total_time + reconstruction_time
    if case_name == "12":
        # This mirrors the authors' notebook correction used for the published table.
        rom_total_time -= 14.0

    return {
        "solid_speedup": solid_speedup,
        "overall_speedup": float(fom_total_time / rom_total_time),
        "iteration_overhead": accumulated_iteration_overhead(
            fom_iterations[start_increment:],
            rom_iterations[start_increment:],
        ),
    }


def _parse_online_displacements(payload_root: Path, example_root: Path) -> tuple[Path, Path]:
    fom_path = example_root / "online_08p3_fom.npz"
    rom_path = example_root / "online_08p3_rom.npz"
    if not fom_path.exists():
        np.savez_compressed(
            fom_path,
            **read_structural_vtk_series(
                payload_root / "Double_E10" / "08.3" / "vtk_data" / "vtk_output_fsi_csm"
            ),
        )
    if not rom_path.exists():
        np.savez_compressed(
            rom_path,
            **read_structural_vtk_series(
                payload_root / "Double_ROM_E10" / "08.3_LASSO" / "vtk_data" / "vtk_output_fsi_csm"
            ),
        )
    return fom_path, rom_path


def prepare_example2_artifacts(
    cache_root: Path,
    artifacts_root: Path,
    *,
    include_online_vtk: bool,
) -> dict[str, Path]:
    _, payload_root = _materialize_double_flap_payloads(cache_root, include_online_vtk=include_online_vtk)

    example_root = artifacts_root / "example2"
    example_root.mkdir(parents=True, exist_ok=True)

    forces, displacements = _build_training_snapshots(payload_root)
    train_dataset_path = example_root / "rom4_train_snapshots.npz"
    NamedSnapshotBatch(
        fields={"input": forces, "output": displacements},
        metadata={
            "benchmark": "example2",
            "training_cases": [reynolds_from_case(case) for case in TRAINING_CASES],
        },
    ).save(train_dataset_path)

    interface_matrix_path = example_root / "interface_matrix.npy"
    np.save(interface_matrix_path, np.load(payload_root / "coSimData" / "map_used.npy").T)
    np.save(example_root / "interface_coordinates.npy", np.load(payload_root / "coSimData" / "coords_interf.npy"))

    config_path = Path(__file__).resolve().parent / "configs" / "example2_rom4.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    save_config(
        {
            "dataset_path": _repo_relative(train_dataset_path),
            "model_path": _repo_relative(example_root / "rom4_model.pkl"),
            "input_modes": EXAMPLE2_FORCE_MODES,
            "output_modes": EXAMPLE2_DISPLACEMENT_MODES,
            "regression": {
                "kind": "poly_lasso",
                "degree": 2,
                "criterion": "bic",
                "standardize_inputs": False,
            },
            "use_quadratic_decoder": True,
            "output_matrix_path": _repo_relative(interface_matrix_path),
            "metadata": {"benchmark": "example2", "rom": "ROM4"},
        },
        config_path,
    )

    return {
        "payload_root": payload_root,
        "train_dataset": train_dataset_path,
        "interface_matrix": interface_matrix_path,
        "model": example_root / "rom4_model.pkl",
        "results": example_root / "results.json",
        "root": example_root,
    }


def run_example2(
    *,
    cache_root: Path | None = None,
    artifacts_root: Path | None = None,
    include_online_vtk: bool = True,
) -> dict[str, Any]:
    cache_root = default_cache_root() if cache_root is None else Path(cache_root)
    artifacts_root = default_artifacts_root() if artifacts_root is None else Path(artifacts_root)
    paths = prepare_example2_artifacts(
        cache_root=cache_root,
        artifacts_root=artifacts_root,
        include_online_vtk=include_online_vtk,
    )

    train_snapshots = NamedSnapshotBatch.load(paths["train_dataset"])
    train_forces = np.asarray(train_snapshots["input"], dtype=float)
    train_displacements = np.asarray(train_snapshots["output"], dtype=float)

    mode_sweep = run_mode_cross_validation(
        train_forces,
        train_displacements,
        input_modes=list(range(15, 61, 5)),
        output_modes=[EXAMPLE2_DISPLACEMENT_MODES],
        regressor_factory=lambda: PolynomialLassoRegressor(
            degree=2,
            criterion="bic",
            standardize_inputs=False,
        ),
        test_fraction=0.05,
        random_state=0,
        use_quadratic_decoder=True,
    )
    dump_json(mode_sweep.entries, paths["root"] / "mode_sweep.json")

    sorted_entries = sorted(mode_sweep.entries, key=lambda entry: entry.input_modes)
    figure, axis_left = plt.subplots(figsize=(8, 4.5))
    axis_left.semilogy(
        [entry.input_modes for entry in sorted_entries],
        [entry.validation_error for entry in sorted_entries],
        marker="o",
        color="black",
        label="Validation L2 error",
    )
    axis_left.set_xlabel("Force modes")
    axis_left.set_ylabel("Validation L2 error")
    axis_left.grid(True, alpha=0.3)
    axis_right = axis_left.twinx()
    axis_right.semilogy(
        [entry.input_modes for entry in sorted_entries],
        [entry.regression_error for entry in sorted_entries],
        marker="s",
        color="red",
        label="Training reduced regression error",
    )
    axis_right.set_ylabel("Reduced regression L2 error")
    lines = axis_left.get_lines() + axis_right.get_lines()
    axis_left.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout()
    figure.savefig(paths["root"] / "mode_sweep.png", dpi=200)
    plt.close(figure)

    model = run_offline_pipeline(
        OfflineConfig(
            dataset_path=str(paths["train_dataset"]),
            model_path=str(paths["model"]),
            input_modes=EXAMPLE2_FORCE_MODES,
            output_modes=EXAMPLE2_DISPLACEMENT_MODES,
            regression=RegressionConfig(
                kind="poly_lasso",
                degree=2,
                criterion="bic",
                standardize_inputs=False,
            ),
            use_quadratic_decoder=True,
            output_matrix_path=str(paths["interface_matrix"]),
            metadata={"benchmark": "example2", "rom": "ROM4"},
        )
    )

    started = time.perf_counter()
    _ = model.predict_restricted(train_forces)
    interface_inference_elapsed = time.perf_counter() - started

    offline_metrics = _evaluate_offline_rom4(train_forces, train_displacements, random_state=0)

    speed_metrics = {
        case_name: _compute_speed_metrics(paths["payload_root"], case_name)
        for case_name in COUPLED_CASES
    }

    online_metrics: dict[str, Any] = {}
    if include_online_vtk:
        fom_path, rom_path = _parse_online_displacements(paths["payload_root"], paths["root"])
        fom_series = np.load(fom_path)
        rom_series = np.load(rom_path)
        start_index = _prediction_start("08.3")
        fom_displacements = np.asarray(fom_series["displacements"], dtype=float)[:, start_index:]
        rom_displacements = np.asarray(rom_series["displacements"], dtype=float)[:, start_index:]
        error_curve = online_relative_displacement_error(fom_displacements, rom_displacements)
        online_metrics = {
            "reynolds_number": reynolds_from_case("08.3"),
            "start_increment": start_index,
            "mean_sample_l2_error": mean_sample_l2_error(fom_displacements, rom_displacements),
            "max_relative_displacement_error": max_online_relative_displacement_error(
                fom_displacements, rom_displacements
            ),
            "iteration_overhead": speed_metrics["08.3"]["iteration_overhead"],
            "solid_speedup": speed_metrics["08.3"]["solid_speedup"],
            "overall_speedup": speed_metrics["08.3"]["overall_speedup"],
        }

        time_values = np.arange(fom_displacements.shape[1], dtype=float) * 0.008 + 0.008 * start_index
        left_tip_x_index = 2 * 1488
        plot_xy(
            time_values,
            [
                ("FOM-FOM tip displacement", fom_displacements[left_tip_x_index, :]),
                ("ROM-FOM tip displacement", rom_displacements[left_tip_x_index, :]),
            ],
            path=paths["root"] / "tip_displacement_re300.png",
            title="Example 2 Left-Tip X-Displacement at Re ~ 300",
            xlabel="Time [s]",
            ylabel="Displacement [m]",
        )
        plot_xy(
            time_values,
            [("Relative displacement error", error_curve)],
            path=paths["root"] / "relative_error_re300.png",
            title="Example 2 Online Relative Error at Re ~ 300",
            xlabel="Time [s]",
            ylabel="Relative error",
        )

    plot_cumulative_iterations(
        np.load(_load_case_co_sim(paths["payload_root"], "08.3", rom=False) / "iters.npy"),
        np.load(_load_case_co_sim(paths["payload_root"], "08.3", rom=True) / "iters.npy"),
        path=paths["root"] / "iterations_re300.png",
        title="Example 2 Cumulative Iterations at Re ~ 300",
    )

    result = {
        "benchmark": "example2",
        "paper_selected_modes": {
            "input_modes": EXAMPLE2_FORCE_MODES,
            "output_modes": EXAMPLE2_DISPLACEMENT_MODES,
        },
        "training_cases_reynolds": [reynolds_from_case(case_name) for case_name in TRAINING_CASES],
        "mode_sweep": {
            "best_entry": min(mode_sweep.entries, key=lambda entry: entry.validation_error),
            "entries": mode_sweep.entries,
        },
        "offline_validation": offline_metrics,
        "interface_inference_time_per_snapshot": interface_inference_elapsed / train_forces.shape[1],
        "speed_metrics": speed_metrics,
        "online_re300": online_metrics,
        "artifacts": {name: str(path) for name, path in paths.items() if name != "payload_root"},
    }
    dump_json(result, paths["results"])
    return result
