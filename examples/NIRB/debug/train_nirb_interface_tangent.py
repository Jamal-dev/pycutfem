from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.NIRB.common import dump_json


def _load_cosim_dir(run_root: Path) -> Path:
    root = Path(run_root)
    if (root / "coSimData").is_dir():
        return root / "coSimData"
    if root.name == "coSimData" and root.is_dir():
        return root
    raise FileNotFoundError(f"Could not find coSimData under {root}")


def _pair_indices_from_iters(iters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    previous: list[int] = []
    current: list[int] = []
    offset = 0
    for count in np.asarray(iters, dtype=int).reshape(-1):
        if count > 1:
            local = np.arange(offset, offset + int(count), dtype=int)
            previous.extend(local[:-1].tolist())
            current.extend(local[1:].tolist())
        offset += int(count)
    return np.asarray(previous, dtype=int), np.asarray(current, dtype=int)


def _fit_secant_map(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge_relative: float,
) -> np.ndarray:
    x_mat = np.asarray(x, dtype=float)
    y_mat = np.asarray(y, dtype=float)
    if x_mat.ndim != 2 or y_mat.ndim != 2:
        raise ValueError("x and y must be feature-major matrices")
    if x_mat.shape[1] != y_mat.shape[1]:
        raise ValueError("x and y sample counts must match")
    gram = x_mat @ x_mat.T
    ridge = float(ridge_relative) * (float(np.trace(gram)) / max(int(gram.shape[0]), 1))
    lhs = gram + ridge * np.eye(gram.shape[0], dtype=float)
    rhs = x_mat @ y_mat.T
    try:
        tangent_t = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        tangent_t = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return np.asarray(tangent_t.T, dtype=float)


def _relative_column_errors(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    denom = np.linalg.norm(ref, axis=0)
    denom = np.maximum(denom, 1.0e-14)
    return np.linalg.norm(pred - ref, axis=0) / denom


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a whole-history secant interface-compliance operator for the NIRB solid surrogate. "
            "The operator maps consecutive interface-load increments to consecutive solid displacement increments."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--load-key", default="load_guess_data")
    parser.add_argument("--displacement-key", default="disp_data")
    parser.add_argument("--iters-key", default="iters")
    parser.add_argument("--min-load-increment-norm", type=float, default=1.0e-14)
    parser.add_argument("--ridge-relative", type=float, default=1.0e-10)
    parser.add_argument("--fit-full", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-interface", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    co_sim_dir = _load_cosim_dir(Path(args.run_root))
    load_path = co_sim_dir / f"{args.load_key}.npy"
    disp_path = co_sim_dir / f"{args.displacement_key}.npy"
    iters_path = co_sim_dir / f"{args.iters_key}.npy"
    map_path = co_sim_dir / "map_used.npy"
    coords_path = co_sim_dir / "coords_interf.npy"
    for path in (load_path, disp_path, iters_path, map_path, coords_path):
        if not path.exists():
            raise FileNotFoundError(path)

    loads = np.asarray(np.load(load_path), dtype=float)
    displacements = np.asarray(np.load(disp_path), dtype=float)
    iters = np.asarray(np.load(iters_path), dtype=int)
    interface_matrix = np.asarray(np.load(map_path), dtype=float)
    interface_coords = np.asarray(np.load(coords_path), dtype=float)
    if loads.ndim != 2 or loads.shape[0] != interface_coords.shape[0] * 2:
        raise ValueError(f"load matrix shape {loads.shape} does not match interface coords {interface_coords.shape}")
    if displacements.ndim != 2 or interface_matrix.shape[1] != displacements.shape[0]:
        raise ValueError(
            f"displacement matrix shape {displacements.shape} is incompatible with map_used {interface_matrix.shape}"
        )

    prev_idx, curr_idx = _pair_indices_from_iters(iters)
    d_load = loads[:, curr_idx] - loads[:, prev_idx]
    load_norms = np.linalg.norm(d_load, axis=0)
    keep = load_norms > float(args.min_load_increment_norm)
    prev_idx = prev_idx[keep]
    curr_idx = curr_idx[keep]
    d_load = d_load[:, keep]
    if d_load.shape[1] == 0:
        raise RuntimeError("No usable consecutive coupling-state load increments found.")

    d_full = displacements[:, curr_idx] - displacements[:, prev_idx]
    d_interface = interface_matrix @ d_full

    full_tangent = None
    interface_tangent = None
    full_error = np.zeros((0,), dtype=float)
    interface_error = np.zeros((0,), dtype=float)
    if bool(args.fit_full):
        full_tangent = _fit_secant_map(d_load, d_full, ridge_relative=float(args.ridge_relative))
        full_error = _relative_column_errors(d_full, full_tangent @ d_load)
    if bool(args.fit_interface):
        interface_tangent = _fit_secant_map(d_load, d_interface, ridge_relative=float(args.ridge_relative))
        interface_error = _relative_column_errors(d_interface, interface_tangent @ d_load)

    metadata = {
        "source_run_root": str(Path(args.run_root)),
        "co_sim_dir": str(co_sim_dir),
        "load_key": str(args.load_key),
        "displacement_key": str(args.displacement_key),
        "snapshot_count": int(loads.shape[1]),
        "pair_count": int(d_load.shape[1]),
        "step_count": int(iters.size),
        "ridge_relative": float(args.ridge_relative),
        "min_load_increment_norm": float(args.min_load_increment_norm),
        "load_dofs": int(loads.shape[0]),
        "full_displacement_dofs": int(displacements.shape[0]),
        "interface_displacement_dofs": int(interface_matrix.shape[0]),
        "full_error_mean": float(np.mean(full_error)) if full_error.size else None,
        "full_error_max": float(np.max(full_error)) if full_error.size else None,
        "interface_error_mean": float(np.mean(interface_error)) if interface_error.size else None,
        "interface_error_max": float(np.max(interface_error)) if interface_error.size else None,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "load_coords": interface_coords,
        "interface_coords": interface_coords,
        "metadata_json": np.asarray(json.dumps(metadata)),
    }
    if full_tangent is not None:
        payload["full_tangent"] = np.asarray(full_tangent, dtype=float)
    if interface_tangent is not None:
        payload["interface_tangent"] = np.asarray(interface_tangent, dtype=float)
    np.savez_compressed(output_path, **payload)

    summary_path = Path(args.summary_path) if args.summary_path is not None else output_path.with_suffix(".json")
    dump_json(metadata, summary_path)
    print(summary_path)


if __name__ == "__main__":
    main()
