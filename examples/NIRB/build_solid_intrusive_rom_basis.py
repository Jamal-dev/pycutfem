from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _load_snapshot_matrix(path: Path) -> np.ndarray:
    matrix = np.asarray(np.load(path), dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D snapshot matrix in {path}, got shape {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Snapshot matrix contains non-finite values: {path}")
    return matrix


def _select_columns(matrix: np.ndarray, *, max_snapshots: int | None, stride: int) -> np.ndarray:
    stride_value = max(1, int(stride))
    selected = np.asarray(matrix[:, ::stride_value], dtype=float)
    if max_snapshots is not None and int(max_snapshots) > 0:
        selected = selected[:, : int(max_snapshots)]
    if int(selected.shape[1]) == 0:
        raise ValueError("No snapshots selected for solid ROM basis training.")
    return selected


def _fit_origin_pod(matrix: np.ndarray, n_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # The structural clamp is homogeneous; the ROM must be a linear displacement
    # subspace through zero, not a mean-shifted affine map.
    left, singular_values, _ = np.linalg.svd(np.asarray(matrix, dtype=float), full_matrices=False)
    keep = min(max(1, int(n_modes)), int(left.shape[1]))
    singular_kept = np.asarray(singular_values[:keep], dtype=float)
    squared = np.asarray(singular_values, dtype=float) ** 2
    total = float(np.sum(squared))
    energy = np.cumsum(squared) / total if total > 0.0 else np.zeros_like(squared)
    return np.asarray(left[:, :keep], dtype=float), singular_kept, np.asarray(energy[:keep], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an intrusive solid ROM POD basis for NIRB Example 2.")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        required=True,
        help="Run directory or coSimData directory containing disp_data.npy.",
    )
    parser.add_argument("--output-path", type=Path, required=True, help="Output .npz basis path.")
    parser.add_argument("--displacement-key", default="disp_data")
    parser.add_argument("--modes", type=int, default=120)
    parser.add_argument("--max-snapshots", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--append-zero-snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append one zero snapshot so the origin remains explicitly represented.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_root = Path(args.snapshot_dir)
    co_sim_dir = snapshot_root / "coSimData" if (snapshot_root / "coSimData").exists() else snapshot_root
    snapshot_path = co_sim_dir / str(args.displacement_key)
    if not snapshot_path.exists():
        snapshot_path = co_sim_dir / f"{args.displacement_key}.npy"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Could not find displacement snapshot matrix: {snapshot_path}")

    train_start = time.perf_counter()
    snapshots = _load_snapshot_matrix(snapshot_path)
    selected = _select_columns(
        snapshots,
        max_snapshots=args.max_snapshots,
        stride=int(args.stride),
    )
    if bool(args.append_zero_snapshot):
        selected = np.column_stack([np.zeros((int(selected.shape[0]),), dtype=float), selected])
    basis, singular_values, energy_fraction = _fit_origin_pod(selected, int(args.modes))
    train_time = time.perf_counter() - train_start

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "snapshot_path": str(snapshot_path),
        "snapshot_shape": list(map(int, snapshots.shape)),
        "selected_shape": list(map(int, selected.shape)),
        "modes": int(basis.shape[1]),
        "centered": False,
        "append_zero_snapshot": bool(args.append_zero_snapshot),
        "stride": int(args.stride),
        "max_snapshots": None if args.max_snapshots is None else int(args.max_snapshots),
        "train_time_s": float(train_time),
        "energy_fraction_last": float(energy_fraction[-1]) if energy_fraction.size else 0.0,
    }
    np.savez_compressed(
        output_path,
        basis=np.asarray(basis, dtype=float),
        mean=np.zeros((int(basis.shape[0]),), dtype=float),
        singular_values=np.asarray(singular_values, dtype=float),
        energy_fraction=np.asarray(energy_fraction, dtype=float),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"basis_path": str(output_path), **metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
