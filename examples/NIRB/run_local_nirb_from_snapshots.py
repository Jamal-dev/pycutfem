from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from examples.NIRB.common import dump_json
from pycutfem.mor.metrics import (
    max_online_relative_displacement_error,
    mean_sample_l2_error,
    snapshot_l2_error,
    speedup,
)
from examples.utils.nirb import load_cosim_snapshot_batch
from pycutfem.mor.nirb.offline import OfflineConfig, RegressionConfig, run_offline_pipeline


def _load_scalar(path: Path) -> float | None:
    if not path.exists():
        return None
    return float(np.asarray(np.load(path), dtype=float).reshape(-1)[0])


def _context_from_batch(batch, n_samples: int, input_features: tuple[str, ...]) -> dict[str, np.ndarray] | None:
    if not input_features:
        return None
    context: dict[str, np.ndarray] = {}
    for name in input_features:
        if name == "time":
            if batch.times is None:
                raise ValueError("--input-features includes time, but snapshot metadata has no time_s column")
            context["time"] = np.asarray(batch.times[:n_samples], dtype=float)
        elif name == "coupling_iter":
            values = [dict(item).get("coupling_iter") for item in batch.sample_metadata[:n_samples]]
            if any(value is None for value in values):
                raise ValueError(
                    "--input-features includes coupling_iter, but snapshot metadata has no coupling_iter column"
                )
            context["coupling_iter"] = np.asarray(values, dtype=float)
        else:
            raise ValueError(f"Unsupported input feature {name!r}")
    return context


def _prediction_loop(model, forces: np.ndarray, context: dict[str, np.ndarray] | None = None) -> np.ndarray:
    columns = []
    for index in range(forces.shape[1]):
        sample_context = None
        if context is not None:
            sample_context = {key: np.asarray(value, dtype=float).reshape(-1)[index] for key, value in context.items()}
        columns.append(model.predict(forces[:, index], context=sample_context)[:, 0])
    return np.column_stack(columns)


def _time_prediction(
    model,
    forces: np.ndarray,
    *,
    repeats: int,
    context: dict[str, np.ndarray] | None = None,
) -> dict[str, float | np.ndarray]:
    repeats = max(1, int(repeats))

    batch_times: list[float] = []
    batch_prediction = None
    for _ in range(repeats):
        started = time.perf_counter()
        batch_prediction = model.predict(forces, context=context)
        batch_times.append(time.perf_counter() - started)

    loop_times: list[float] = []
    loop_prediction = None
    for _ in range(repeats):
        started = time.perf_counter()
        loop_prediction = _prediction_loop(model, forces, context=context)
        loop_times.append(time.perf_counter() - started)

    if batch_prediction is None or loop_prediction is None:
        raise RuntimeError("prediction timing failed")

    return {
        "batch_prediction": np.asarray(batch_prediction, dtype=float),
        "loop_prediction": np.asarray(loop_prediction, dtype=float),
        "batch_time_s": float(min(batch_times)),
        "loop_time_s": float(min(loop_times)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and time an Example 2 NIRB model from local coSimData.")
    parser.add_argument("--snapshot-dir", type=Path, required=True, help="Run directory or coSimData directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--force-key", default="load_guess_data")
    parser.add_argument("--displacement-key", default="disp_data")
    parser.add_argument("--force-modes", type=int, default=45)
    parser.add_argument("--displacement-modes", type=int, default=9)
    parser.add_argument("--zero-anchor-weight", type=int, default=0)
    parser.add_argument(
        "--regression-kind",
        choices=("poly_lasso", "poly_ls", "poly_ridge", "tps_rbf", "knn"),
        default="poly_lasso",
    )
    parser.add_argument("--regularization", type=float, default=0.0)
    parser.add_argument("--knn-neighbors", type=int, default=8)
    parser.add_argument("--knn-power", type=float, default=2.0)
    parser.add_argument(
        "--input-features",
        default="",
        help=(
            "Comma-separated context features appended to reduced force coordinates. "
            "Supported: time,coupling_iter."
        ),
    )
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--max-eval-snapshots", type=int, default=None)
    parser.add_argument(
        "--interface-matrix-path",
        type=Path,
        default=None,
        help="Optional solid-full to solid-interface restriction matrix. Defaults to coSimData/map_used.npy when present.",
    )
    parser.add_argument(
        "--no-interface-matrix",
        action="store_true",
        help="Do not attach an interface restriction, useful when the displacement snapshots are already interface-only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path) if args.model_path is not None else output_dir / "nirb_model.pkl"

    batch = load_cosim_snapshot_batch(
        args.snapshot_dir,
        force_key=str(args.force_key),
        displacement_key=str(args.displacement_key),
    )
    dataset_path = output_dir / "generic_nirb_snapshots.npz"
    batch.save(dataset_path)
    forces = np.asarray(batch["interface_load"], dtype=float)
    reference = np.asarray(batch["solid_displacement"], dtype=float)
    co_sim_dir = Path(batch.metadata["co_sim_dir"])
    interface_matrix_path = Path(args.interface_matrix_path) if args.interface_matrix_path is not None else None
    if interface_matrix_path is None and not bool(args.no_interface_matrix):
        candidate = co_sim_dir / "map_used.npy"
        if candidate.exists():
            interface_matrix_path = candidate
    if interface_matrix_path is not None:
        interface_matrix = np.asarray(np.load(interface_matrix_path), dtype=float)
        if interface_matrix.ndim != 2:
            raise ValueError(f"Interface matrix must be 2D: {interface_matrix_path}")
        if interface_matrix.shape[1] != reference.shape[0]:
            if interface_matrix.shape[0] == reference.shape[0]:
                reoriented_path = output_dir / "interface_matrix_reoriented.npy"
                np.save(reoriented_path, interface_matrix.T)
                interface_matrix_path = reoriented_path
            else:
                raise ValueError(
                    "Interface matrix is incompatible with displacement snapshots: "
                    f"{interface_matrix.shape} vs {reference.shape[0]} full dofs"
                )
    if args.max_eval_snapshots is not None:
        n_eval = min(int(args.max_eval_snapshots), forces.shape[1])
        forces_eval = forces[:, :n_eval]
        reference_eval = reference[:, :n_eval]
    else:
        forces_eval = forces
        reference_eval = reference
    input_features = tuple(
        item.strip()
        for item in str(args.input_features).replace(";", ",").split(",")
        if item.strip()
    )
    input_features = tuple("coupling_iter" if item == "subiteration" else item for item in input_features)
    context_eval = _context_from_batch(batch, forces_eval.shape[1], input_features)

    config = OfflineConfig(
        dataset_path=str(dataset_path),
        model_path=str(model_path),
        input_modes=int(args.force_modes),
        output_modes=int(args.displacement_modes),
        regression=RegressionConfig(
            kind=str(args.regression_kind),
            degree=2,
            criterion="bic",
            standardize_inputs=bool(str(args.regression_kind) == "knn"),
            regularization=float(args.regularization),
            n_neighbors=int(args.knn_neighbors),
            power=float(args.knn_power),
        ),
        use_quadratic_decoder=True,
        dataset_input_field="interface_load",
        dataset_output_field="solid_displacement",
        zero_anchor_weight=int(args.zero_anchor_weight),
        output_matrix_path=None if interface_matrix_path is None else str(interface_matrix_path),
        context_feature_names=input_features,
        metadata={"benchmark": "example2", "source": "local_pycutfem_cosim"},
    )

    train_start = time.perf_counter()
    model = run_offline_pipeline(config)
    train_time_s = time.perf_counter() - train_start

    timing = _time_prediction(model, forces_eval, repeats=int(args.timing_repeats), context=context_eval)
    prediction = np.asarray(timing["batch_prediction"], dtype=float)
    prediction_path = output_dir / "nirb_full_prediction.npy"
    np.save(prediction_path, prediction)

    run_root = Path(batch.metadata["run_root"])
    solid_times = np.asarray(
        [dict(item).get("solid_time", np.nan) for item in batch.sample_metadata],
        dtype=float,
    )
    fom_structure_time_s = float(np.nansum(solid_times)) if np.all(np.isfinite(solid_times)) else None
    if fom_structure_time_s is not None and args.max_eval_snapshots is not None:
        fom_structure_time_s = float(np.sum(solid_times[: forces_eval.shape[1]]))
    total_time = _load_scalar(co_sim_dir / "total_solving_time.npy")
    if total_time is None:
        summary = batch.metadata.get("summary", {})
        total_time = float(summary["total_wall_time_s"]) if "total_wall_time_s" in summary else None

    loop_time_s = float(timing["loop_time_s"])
    speed_report: dict[str, float] = {}
    if fom_structure_time_s is not None:
        speed_report["solid_speedup_loop"] = speedup(fom_structure_time_s, loop_time_s)
        speed_report["solid_speedup_batch"] = speedup(fom_structure_time_s, float(timing["batch_time_s"]))
    if fom_structure_time_s is not None and total_time is not None:
        rom_total_estimate = max(0.0, float(total_time) - fom_structure_time_s) + loop_time_s
        speed_report["overall_speedup_estimate"] = speedup(float(total_time), rom_total_estimate)
        speed_report["rom_total_time_estimate_s"] = float(rom_total_estimate)

    result = {
        "snapshot_dir": str(args.snapshot_dir),
        "co_sim_dir": str(co_sim_dir),
        "run_root": str(run_root),
        "model_path": str(model_path),
        "prediction_path": str(prediction_path),
        "force_key": str(args.force_key),
        "displacement_key": str(args.displacement_key),
        "force_modes": int(args.force_modes),
        "displacement_modes": int(args.displacement_modes),
        "zero_anchor_weight": int(args.zero_anchor_weight),
        "regression_kind": str(args.regression_kind),
        "regularization": float(args.regularization),
        "knn_neighbors": int(args.knn_neighbors),
        "knn_power": float(args.knn_power),
        "input_features": list(input_features),
        "interface_matrix_path": None if interface_matrix_path is None else str(interface_matrix_path),
        "snapshots_trained": int(forces.shape[1]),
        "snapshots_evaluated": int(forces_eval.shape[1]),
        "train_time_s": float(train_time_s),
        "prediction_batch_time_s": float(timing["batch_time_s"]),
        "prediction_loop_time_s": loop_time_s,
        "fom_structure_time_s": fom_structure_time_s,
        "fom_total_time_s": total_time,
        "speedup": speed_report,
        "metrics": {
            "mean_sample_l2_error": mean_sample_l2_error(reference_eval, prediction),
            "snapshot_l2_error": snapshot_l2_error(reference_eval, prediction),
            "max_online_relative_displacement_error": max_online_relative_displacement_error(
                reference_eval,
                prediction,
            ),
        },
    }
    dump_json(result, output_dir / "nirb_report.json")
    print(output_dir / "nirb_report.json")


if __name__ == "__main__":
    main()
