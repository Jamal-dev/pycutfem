from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from examples.NIRB.common import dump_json
from examples.NIRB.run_example2_local import CoordinateLookup, _negate_lookup, _resample_lookup_to_coords


def _compare_point_fields(
    *,
    ref_coords: np.ndarray,
    ref_values: np.ndarray,
    local_lookup: CoordinateLookup,
) -> dict[str, float]:
    ref = np.asarray(ref_values, dtype=float)
    loc = np.asarray(local_lookup(ref_coords[:, 0], ref_coords[:, 1]), dtype=float).reshape(ref.shape)
    diff = loc - ref
    ref_flat = ref.reshape(-1)
    loc_flat = loc.reshape(-1)
    denom = max(float(np.linalg.norm(ref_flat)), 1.0e-15)
    loc_norm = float(np.linalg.norm(loc_flat))
    ref_norm = float(np.linalg.norm(ref_flat))
    cosine = float(np.dot(loc_flat, ref_flat) / max(loc_norm * ref_norm, 1.0e-15))
    return {
        "rel_l2": float(np.linalg.norm(diff.reshape(-1)) / denom),
        "abs_rms": float(np.linalg.norm(diff.reshape(-1)) / np.sqrt(max(diff.size, 1))),
        "abs_max": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "cosine": cosine,
        "reference_max_norm": float(np.max(np.linalg.norm(ref, axis=1))) if ref.size else 0.0,
        "local_max_norm": float(np.max(np.linalg.norm(loc, axis=1))) if loc.size else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the local Example 2 interface transfer against a monitored Kratos coupling stage.")
    parser.add_argument("--output", type=Path, default=Path("examples/NIRB/artifacts/interface_gate_check.json"))
    parser.add_argument(
        "--kratos-monitor",
        type=Path,
        default=Path("examples/NIRB/artifacts/kratos_example2_monitor_20260407/coupling_monitor/step0001_iter0002_after_sync_output_structure.npz"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(Path(args.kratos_monitor).resolve()) as data:
        fluid_disp_coords = np.asarray(data["fluid_disp_coords_ref"], dtype=float)
        fluid_disp_values = np.asarray(data["fluid_disp_values"], dtype=float)
        structure_disp_coords = np.asarray(data["structure_disp_coords_ref"], dtype=float)
        structure_disp_values = np.asarray(data["structure_disp_values"], dtype=float)
        fluid_load_coords = np.asarray(data["fluid_load_coords_ref"], dtype=float)
        fluid_load_values = np.asarray(data["fluid_load_values"], dtype=float)
        structure_load_coords = np.asarray(data["structure_load_coords_ref"], dtype=float)
        structure_load_values = np.asarray(data["structure_load_values"], dtype=float)

    disp_lookup = _resample_lookup_to_coords(
        CoordinateLookup(fluid_disp_coords, fluid_disp_values, dim=2),
        structure_disp_coords,
    )
    load_lookup = _resample_lookup_to_coords(
        _negate_lookup(CoordinateLookup(fluid_load_coords, fluid_load_values, dim=2)),
        structure_load_coords,
    )

    disp_cmp = _compare_point_fields(
        ref_coords=structure_disp_coords,
        ref_values=structure_disp_values,
        local_lookup=disp_lookup,
    )
    load_cmp = _compare_point_fields(
        ref_coords=structure_load_coords,
        ref_values=structure_load_values,
        local_lookup=load_lookup,
    )

    summary = {
        "kratos_monitor": str(Path(args.kratos_monitor).resolve()),
        "fluid_to_structure_disp": disp_cmp,
        "fluid_to_structure_load": load_cmp,
        "criteria": {
            "disp_rel_l2_le_1e-12": bool(float(disp_cmp["rel_l2"]) <= 1.0e-12),
            "load_rel_l2_le_1e-12": bool(float(load_cmp["rel_l2"]) <= 1.0e-12),
            "load_cosine_ge_0.999999999999": bool(float(load_cmp["cosine"]) >= 0.999999999999),
        },
    }
    summary["criteria_all_pass"] = bool(all(summary["criteria"].values()))
    dump_json(summary, Path(args.output))
    print(f"interface gate: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
