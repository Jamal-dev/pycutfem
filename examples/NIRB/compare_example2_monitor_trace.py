from __future__ import annotations

import argparse
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


class PointLookup:
    def __init__(self, coords: np.ndarray, values: np.ndarray) -> None:
        coords_arr = np.asarray(coords, dtype=float)
        values_arr = np.asarray(values, dtype=float)
        if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
            raise ValueError("coords must have shape (n, 2)")
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


def _rms(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=float).reshape(-1)
    return float(np.linalg.norm(flat) / np.sqrt(max(flat.size, 1)))


def _compare_fields(reference_coords: np.ndarray, reference_values: np.ndarray, local_coords: np.ndarray, local_values: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference_values, dtype=float)
    loc = PointLookup(local_coords, local_values).sample(reference_coords)
    diff = loc - ref
    ref_flat = ref.reshape(-1)
    loc_flat = loc.reshape(-1)
    denom = max(float(np.linalg.norm(ref_flat)), 1.0e-15)
    cosine = float(np.dot(loc_flat, ref_flat) / max(float(np.linalg.norm(loc_flat) * np.linalg.norm(ref_flat)), 1.0e-15))
    return {
        "abs_rms": _rms(diff),
        "abs_max": float(np.max(np.abs(diff))),
        "rel_l2": float(np.linalg.norm(diff.reshape(-1)) / denom),
        "cosine": cosine,
        "reference_max_norm": float(np.max(np.linalg.norm(ref, axis=1))),
        "local_max_norm": float(np.max(np.linalg.norm(loc, axis=1))),
    }


def _load_matrix(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    matrix = np.load(path)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    return np.asarray(matrix, dtype=float)


def _load_local_snapshot(matrix: np.ndarray | None, coords: np.ndarray, iteration: int) -> tuple[np.ndarray, np.ndarray] | None:
    if matrix is None or matrix.shape[1] < iteration:
        return None
    values = np.asarray(matrix[:, iteration - 1], dtype=float).reshape(-1, 2)
    return coords, values


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("records", []))


def _find_npz(
    manifest: list[dict[str, Any]],
    *,
    stage: str,
    iteration: int,
    monitor_dir: Path,
    step: int | None = None,
) -> Path | None:
    for record in manifest:
        if step is not None and int(record.get("step", -1)) != int(step):
            continue
        if str(record.get("stage")) != str(stage):
            continue
        if int(record.get("iteration", -1)) != int(iteration):
            continue
        stem = f"step{int(record.get('step', 0)):04d}_iter{int(record['iteration']):04d}_{record['stage']}"
        candidate = monitor_dir / f"{stem}.npz"
        if candidate.exists():
            return candidate
    return None


def _npz_field(npz_path: Path, *, solver: str, field: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path) as data:
        coords = np.asarray(data[f"{solver}_{field}_coords_ref"], dtype=float)
        values = np.asarray(data[f"{solver}_{field}_values"], dtype=float)
    return coords, values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local Example 2 interface traces against a monitored Kratos coupling run.")
    parser.add_argument("--local-output-dir", type=Path, required=True)
    parser.add_argument("--kratos-monitor-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--structure-stage", type=str, default="after_sync_output_structure")
    parser.add_argument("--fluid-stage", type=str, default="after_sync_output_fluid")
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_output_dir = Path(args.local_output_dir).resolve()
    kratos_monitor_dir = Path(args.kratos_monitor_dir).resolve()
    co_sim_dir = local_output_dir / "coSimData"
    output_path = (
        Path(args.output_path).resolve()
        if args.output_path is not None
        else local_output_dir / "comparison_monitor_trace.json"
    )

    manifest = _load_manifest(kratos_monitor_dir / "manifest.json")
    local_coords = np.load(co_sim_dir / "coords_interf.npy")
    local_coords_fluid = (
        np.load(co_sim_dir / "coords_interf_fluid.npy")
        if (co_sim_dir / "coords_interf_fluid.npy").exists()
        else local_coords
    )
    local_disp = _load_matrix(co_sim_dir / "interface_disp_data.npy")
    local_vel = _load_matrix(co_sim_dir / "interface_velocity_data.npy")
    local_load_return = _load_matrix(co_sim_dir / "load_return_data.npy")
    if local_load_return is None:
        local_load_return = _load_matrix(co_sim_dir / "load_data.npy")
    local_load_guess = _load_matrix(co_sim_dir / "load_guess_data.npy")
    local_fluid_load_guess = _load_matrix(co_sim_dir / "fluid_load_guess_data.npy")
    local_fluid_load_return = _load_matrix(co_sim_dir / "fluid_load_return_data.npy")
    local_reaction = _load_matrix(co_sim_dir / "load_reaction_data.npy")
    local_stress = _load_matrix(co_sim_dir / "load_stress_data.npy")

    iterations = sorted(
        {
            int(record["iteration"])
            for record in manifest
            if int(record.get("iteration", 0)) > 0 and int(record.get("step", -1)) == int(args.step)
        }
    )
    rows: list[dict[str, Any]] = []

    for iteration in iterations:
        row: dict[str, Any] = {"iteration": int(iteration)}

        struct_npz = _find_npz(
            manifest,
            stage=str(args.structure_stage),
            iteration=iteration,
            monitor_dir=kratos_monitor_dir,
            step=int(args.step),
        )
        fluid_npz = _find_npz(
            manifest,
            stage=str(args.fluid_stage),
            iteration=iteration,
            monitor_dir=kratos_monitor_dir,
            step=int(args.step),
        )

        if struct_npz is not None:
            try:
                ref_coords, ref_disp = _npz_field(struct_npz, solver="structure", field="disp")
                local_disp_snapshot = _load_local_snapshot(local_disp, local_coords, iteration)
                if local_disp_snapshot is not None:
                    _, local_disp_values = local_disp_snapshot
                    for key, value in _compare_fields(ref_coords, ref_disp, local_coords, local_disp_values).items():
                        row[f"structure_disp_{key}"] = value
            except KeyError:
                pass

        if fluid_npz is not None:
            try:
                ref_coords, ref_velocity = _npz_field(fluid_npz, solver="fluid", field="velocity")
                local_vel_snapshot = _load_local_snapshot(local_vel, local_coords, iteration)
                if local_vel_snapshot is not None:
                    _, local_vel_values = local_vel_snapshot
                    for key, value in _compare_fields(ref_coords, ref_velocity, local_coords, local_vel_values).items():
                        row[f"fluid_velocity_{key}"] = value
            except KeyError:
                pass
        if struct_npz is not None:
            try:
                ref_coords_guess, ref_load_guess = _npz_field(struct_npz, solver="fluid", field="load")
                local_snapshot = _load_local_snapshot(
                    local_fluid_load_guess if local_fluid_load_guess is not None else local_load_guess,
                    local_coords_fluid if local_fluid_load_guess is not None else local_coords,
                    iteration,
                )
                if local_snapshot is not None:
                    local_guess_coords, local_values = local_snapshot
                    for key, value in _compare_fields(
                        ref_coords_guess,
                        ref_load_guess,
                        local_guess_coords,
                        local_values,
                    ).items():
                        row[f"fluid_load_guess_{key}"] = value
            except KeyError:
                pass

        if fluid_npz is not None:
            try:
                ref_coords_return, ref_load_return = _npz_field(fluid_npz, solver="fluid", field="load")
                # Prefer the local fluid-side traces when available. The
                # structure-side traces are still useful diagnostics, but they
                # include the extra fluid->structure mapper path and are not a
                # like-for-like compare to the monitored Kratos fluid load.
                fluid_return_snapshot = _load_local_snapshot(
                    local_fluid_load_return if local_fluid_load_return is not None else local_load_return,
                    local_coords_fluid if local_fluid_load_return is not None else local_coords,
                    iteration,
                )
                if fluid_return_snapshot is not None:
                    local_return_coords, local_values = fluid_return_snapshot
                    for key, value in _compare_fields(
                        ref_coords_return,
                        ref_load_return,
                        local_return_coords,
                        local_values,
                    ).items():
                        row["fluid_load_return_" + key] = value

                for label, matrix in (
                    ("fluid_load_reaction", local_reaction),
                    ("fluid_load_stress", local_stress),
                ):
                    local_snapshot = _load_local_snapshot(matrix, local_coords, iteration)
                    if local_snapshot is None:
                        continue
                    local_struct_coords, local_values = local_snapshot
                    for key, value in _compare_fields(
                        ref_coords_return,
                        ref_load_return,
                        local_struct_coords,
                        -1.0 * local_values,
                    ).items():
                        row[f"{label}_{key}"] = value
            except KeyError:
                pass

        rows.append(row)

    summary = {
        "local_output_dir": str(local_output_dir),
        "kratos_monitor_dir": str(kratos_monitor_dir),
        "step": int(args.step),
        "structure_stage": str(args.structure_stage),
        "fluid_stage": str(args.fluid_stage),
        "iterations": rows,
    }

    if rows:
        first = rows[0]
        keys = [key for key in first.keys() if key.endswith("_rel_l2")]
        if keys:
            summary["first_iteration_largest_rel_l2_key"] = max(keys, key=lambda key: float(first.get(key, -1.0)))
            summary["first_iteration_largest_rel_l2_value"] = float(first[summary["first_iteration_largest_rel_l2_key"]])

    _dump_json(summary, output_path)
    print(f"comparison: {output_path}")


if __name__ == "__main__":
    main()
