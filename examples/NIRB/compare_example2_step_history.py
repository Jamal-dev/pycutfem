from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _coord_key(x: float, y: float, ndigits: int = 12) -> tuple[float, float]:
    return (round(float(x), ndigits), round(float(y), ndigits))


def _dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported JSON type: {type(value).__name__}")

    path.write_text(json.dumps(data, indent=2, default=default), encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class PointLookup:
    def __init__(self, coords: np.ndarray, values: np.ndarray) -> None:
        coords_arr = np.asarray(coords, dtype=float)
        values_arr = np.asarray(values, dtype=float)
        if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
            raise ValueError("coords must have shape (n, 2)")
        if values_arr.ndim == 1:
            values_arr = values_arr.reshape(-1, 1)
        if values_arr.ndim != 2 or values_arr.shape[0] != coords_arr.shape[0]:
            raise ValueError("values must have shape (n, m)")
        self.coords = coords_arr
        self.values = values_arr
        self._exact = {_coord_key(x, y): values_arr[i].copy() for i, (x, y) in enumerate(coords_arr)}

    def sample(self, target_coords: np.ndarray) -> np.ndarray:
        target = np.asarray(target_coords, dtype=float)
        out = np.empty((target.shape[0], self.values.shape[1]), dtype=float)
        for i, (x, y) in enumerate(target):
            hit = self._exact.get(_coord_key(x, y))
            if hit is not None:
                out[i, :] = hit
                continue
            dist2 = np.sum((self.coords - target[i][None, :]) ** 2, axis=1)
            out[i, :] = self.values[int(np.argmin(dist2)), :]
        return out


def _compare_fields(
    reference_coords: np.ndarray,
    reference_values: np.ndarray,
    local_coords: np.ndarray,
    local_values: np.ndarray,
    *,
    reference_scale: float = 1.0,
) -> dict[str, float]:
    ref = float(reference_scale) * np.asarray(reference_values, dtype=float)
    loc = PointLookup(local_coords, local_values).sample(reference_coords)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if loc.ndim == 1:
        loc = loc.reshape(-1, 1)
    diff = loc - ref
    ref_flat = ref.reshape(-1)
    loc_flat = loc.reshape(-1)
    diff_flat = diff.reshape(-1)
    denom = max(float(np.linalg.norm(ref_flat)), 1.0e-15)
    cosine_denom = max(float(np.linalg.norm(loc_flat) * np.linalg.norm(ref_flat)), 1.0e-15)
    return {
        "abs_rms": float(np.linalg.norm(diff_flat) / np.sqrt(max(diff_flat.size, 1))),
        "abs_max": float(np.max(np.abs(diff_flat))),
        "rel_l2": float(np.linalg.norm(diff_flat) / denom),
        "cosine": float(np.dot(loc_flat, ref_flat) / cosine_denom),
        "reference_max_norm": float(np.max(np.linalg.norm(ref, axis=1))),
        "local_max_norm": float(np.max(np.linalg.norm(loc, axis=1))),
    }


def _resolve_step_history_dir(path: Path) -> Path:
    path = Path(path).resolve()
    if path.is_dir() and (path / "step_history").is_dir():
        return path / "step_history"
    return path


def _load_step_files(step_history_dir: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted(step_history_dir.glob("step*.npz")):
        stem = path.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if not digits:
            continue
        files[int(digits)] = path
    return files


def _npz_arr(data, key: str) -> np.ndarray | None:
    if key not in data:
        return None
    return np.asarray(data[key], dtype=float)


def _first_existing_arr(data, *keys: str) -> np.ndarray | None:
    for key in keys:
        arr = _npz_arr(data, key)
        if arr is not None:
            return arr
    return None


def _first_matching_arr(data, target_rows: int, *keys: str) -> np.ndarray | None:
    for key in keys:
        arr = _npz_arr(data, key)
        if arr is None:
            continue
        if arr.ndim == 0:
            continue
        if arr.ndim == 1:
            if arr.shape[0] == int(target_rows):
                return arr
            continue
        if arr.shape[0] == int(target_rows):
            return arr
    return None


def _first_divergence(rows: list[dict[str, Any]], metric_key: str, threshold: float) -> int | None:
    for row in rows:
        value = row.get(metric_key)
        if value is None:
            continue
        if float(value) > float(threshold):
            return int(row["step"])
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local and Kratos Example 2 accepted-step histories.")
    parser.add_argument("--local-step-dir", type=Path, required=True, help="Local output dir or local step_history dir.")
    parser.add_argument("--kratos-step-dir", type=Path, required=True, help="Kratos run dir or Kratos step_history dir.")
    parser.add_argument("--divergence-threshold", type=float, default=5.0e-3)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_step_dir = _resolve_step_history_dir(Path(args.local_step_dir))
    kratos_step_dir = _resolve_step_history_dir(Path(args.kratos_step_dir))
    output_json = (
        Path(args.output_json).resolve()
        if args.output_json is not None
        else local_step_dir.parent / "comparison_step_history.json"
    )
    output_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv is not None
        else local_step_dir.parent / "comparison_step_history.csv"
    )

    local_steps = _load_step_files(local_step_dir)
    kratos_steps = _load_step_files(kratos_step_dir)
    common_steps = sorted(set(local_steps).intersection(kratos_steps))
    rows: list[dict[str, Any]] = []

    field_specs = (
        ("fluid_velocity", ("fluid_coords_ref",), ("fluid_velocity_nodal_values", "fluid_velocity_values"), 1.0),
        ("fluid_pressure", ("fluid_coords_ref",), ("fluid_pressure_nodal_values", "fluid_pressure_values"), 1.0),
        ("fluid_mesh_displacement", ("fluid_coords_ref",), ("fluid_mesh_displacement_nodal_values", "fluid_mesh_displacement_values"), 1.0),
        ("fluid_mesh_velocity", ("fluid_coords_ref",), ("fluid_mesh_velocity_nodal_values", "fluid_mesh_velocity_values"), 1.0),
        ("structure_displacement", ("structure_coords_ref",), ("structure_displacement_nodal_values", "structure_displacement_values"), 1.0),
        # Local accepted-step interface_load_values are structure-side returned loads.
        # Kratos accepted-step interface_load_values are fluid-side loads on the same interface,
        # so they must be sign-flipped for a like-for-like comparison.
        ("interface_load", ("interface_load_coords_ref",), ("interface_load_values",), -1.0),
        ("interface_disp", ("interface_disp_coords_ref",), ("interface_disp_values",), 1.0),
        ("interface_velocity", ("interface_velocity_coords_ref",), ("interface_velocity_values",), 1.0),
    )

    for step in common_steps:
        row: dict[str, Any] = {"step": int(step)}
        with np.load(local_steps[step]) as local_data, np.load(kratos_steps[step]) as kratos_data:
            row["local_time_s"] = float(np.asarray(local_data["time_s"], dtype=float).reshape(-1)[0]) if "time_s" in local_data else None
            row["kratos_time_s"] = float(np.asarray(kratos_data["time_s"], dtype=float).reshape(-1)[0]) if "time_s" in kratos_data else None
            for label, coords_keys, values_keys, reference_scale in field_specs:
                ref_coords = _first_existing_arr(kratos_data, *coords_keys)
                local_coords = _first_existing_arr(local_data, *coords_keys)
                if ref_coords is None or local_coords is None:
                    continue
                ref_values = _first_matching_arr(kratos_data, int(ref_coords.shape[0]), *values_keys)
                local_values = _first_matching_arr(local_data, int(local_coords.shape[0]), *values_keys)
                if ref_values is None or local_values is None:
                    continue
                metrics = _compare_fields(
                    ref_coords,
                    ref_values,
                    local_coords,
                    local_values,
                    reference_scale=float(reference_scale),
                )
                for metric_name, value in metrics.items():
                    row[f"{label}_{metric_name}"] = float(value)
        rel_keys = [key for key in row.keys() if str(key).endswith("_rel_l2")]
        if rel_keys:
            worst_key = max(rel_keys, key=lambda key: float(row[key]))
            row["worst_rel_l2_key"] = str(worst_key)
            row["worst_rel_l2_value"] = float(row[worst_key])
        rows.append(row)

    first_divergence_by_field: dict[str, int | None] = {}
    for label, _coords_keys, _values_keys, _reference_scale in field_specs:
        first_divergence_by_field[label] = _first_divergence(rows, f"{label}_rel_l2", float(args.divergence_threshold))

    overall_candidates = [step for step in first_divergence_by_field.values() if step is not None]
    summary = {
        "local_step_dir": str(local_step_dir),
        "kratos_step_dir": str(kratos_step_dir),
        "steps_compared": [int(step) for step in common_steps],
        "divergence_threshold": float(args.divergence_threshold),
        "first_divergence_by_field": first_divergence_by_field,
        "first_divergence_step": int(min(overall_candidates)) if overall_candidates else None,
        "rows": rows,
    }
    if rows:
        summary["last_compared_step"] = int(rows[-1]["step"])

    _dump_json(summary, output_json)
    _write_csv(rows, output_csv)
    print(f"comparison_json: {output_json}")
    print(f"comparison_csv: {output_csv}")


if __name__ == "__main__":
    main()
