#!/usr/bin/env python3
"""Compare one-domain simulation output against the Christan Biofilm I contours."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.biofilms.benchmarks.christan.prepare_biofilm_I_geometry import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_GEOM_DIR,
    ensure_geometry_artifacts,
    front_x_mm,
)


def _read_plain_contour_mm(path: Path) -> np.ndarray:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError(f"Empty contour CSV: {path}")
    pts = np.asarray([(float(row["x_mm"]), float(row["y_mm"])) for row in rows], dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected x_mm,y_mm columns in {path}")
    if not np.allclose(pts[0], pts[-1], rtol=0.0, atol=1.0e-12):
        pts = np.vstack([pts, pts[0]])
    return pts


def _read_snapshot_contours_mm(path: Path) -> list[np.ndarray]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    groups: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        cid = int(row["contour_id"])
        xx = 1.0e3 * float(row["x_m"])
        yy = 1.0e3 * float(row["y_m"])
        groups.setdefault(cid, []).append((xx, yy))
    out = [np.asarray(groups[k], dtype=float) for k in sorted(groups)]
    if not out:
        raise ValueError(f"No contours found in {path}")
    return out


def find_snapshot(out_dir: Path, t_s: float) -> Path:
    snap_dir = Path(out_dir) / "snapshots"
    matches = sorted(snap_dir.glob(f"*t{float(t_s):06.3f}_alpha05.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Missing snapshot at t={float(t_s):.3f} s in {snap_dir}. "
            "Re-run the simulation with --snapshot-times including this time."
        )
    return matches[0]


def _read_timeseries_row(out_dir: Path, t_s: float) -> dict[str, str]:
    ts_path = Path(out_dir) / "timeseries.csv"
    rows = list(csv.DictReader(ts_path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError(f"Empty timeseries CSV: {ts_path}")
    best_row = None
    best_err = float("inf")
    for row in rows:
        try:
            row_t = float(row.get("t_s", float("nan")))
        except Exception:
            row_t = float("nan")
        if not np.isfinite(row_t):
            continue
        err = abs(float(row_t) - float(t_s))
        if err < best_err:
            best_err = err
            best_row = row
    if best_row is None:
        raise ValueError(f"No finite t_s rows found in {ts_path}")
    return best_row


def _contour_bbox(points_list: list[np.ndarray]) -> tuple[float, float, float, float]:
    pts = np.vstack([np.asarray(points, dtype=float) for points in points_list])
    return (
        float(np.min(pts[:, 0])),
        float(np.max(pts[:, 0])),
        float(np.min(pts[:, 1])),
        float(np.max(pts[:, 1])),
    )


def _front_x_points_list(points_list: list[np.ndarray], y_sample_mm: float) -> float:
    xs: list[float] = []
    for points in points_list:
        val = front_x_mm(np.asarray(points, dtype=float), float(y_sample_mm))
        if np.isfinite(val):
            xs.append(float(val))
    if not xs:
        return float("nan")
    return float(min(xs))


def _front_profile(points_list: list[np.ndarray], y_samples_mm: np.ndarray) -> np.ndarray:
    ys = np.asarray(y_samples_mm, dtype=float).ravel()
    return np.asarray([_front_x_points_list(points_list, float(y_mm)) for y_mm in ys], dtype=float)


def _contour_profile_metrics(
    exp_points: list[np.ndarray],
    sim_points: list[np.ndarray],
    *,
    n_samples: int = 256,
) -> dict[str, float]:
    _, _, y_exp_min, y_exp_max = _contour_bbox(exp_points)
    _, _, y_sim_min, y_sim_max = _contour_bbox(sim_points)
    y_min = max(float(y_exp_min), float(y_sim_min))
    y_max = min(float(y_exp_max), float(y_sim_max))
    if not (np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min):
        return {
            "y_min_mm": float("nan"),
            "y_max_mm": float("nan"),
            "rmse_um": float("nan"),
            "mae_um": float("nan"),
            "max_um": float("nan"),
        }
    ys = np.linspace(y_min, y_max, int(max(16, n_samples)), dtype=float)
    x_exp = _front_profile(exp_points, ys)
    x_sim = _front_profile(sim_points, ys)
    good = np.isfinite(x_exp) & np.isfinite(x_sim)
    if not np.any(good):
        return {
            "y_min_mm": float(y_min),
            "y_max_mm": float(y_max),
            "rmse_um": float("nan"),
            "mae_um": float("nan"),
            "max_um": float("nan"),
        }
    err_um = 1.0e3 * (x_sim[good] - x_exp[good])
    return {
        "y_min_mm": float(y_min),
        "y_max_mm": float(y_max),
        "rmse_um": float(np.sqrt(np.mean(err_um**2))),
        "mae_um": float(np.mean(np.abs(err_um))),
        "max_um": float(np.max(np.abs(err_um))),
    }


def _nearest_distance_metrics(exp_points: list[np.ndarray], sim_points: list[np.ndarray]) -> dict[str, float]:
    exp = np.vstack([np.asarray(points, dtype=float) for points in exp_points])
    sim = np.vstack([np.asarray(points, dtype=float) for points in sim_points])
    if exp.size == 0 or sim.size == 0:
        return {"mean_um": float("nan"), "rmse_um": float("nan"), "max_um": float("nan")}
    diff_es = exp[:, None, :] - sim[None, :, :]
    d_es = np.sqrt(np.min(np.sum(diff_es * diff_es, axis=2), axis=1))
    diff_se = sim[:, None, :] - exp[None, :, :]
    d_se = np.sqrt(np.min(np.sum(diff_se * diff_se, axis=2), axis=1))
    both_um = 1.0e3 * np.concatenate([d_es, d_se])
    return {
        "mean_um": float(np.mean(both_um)),
        "rmse_um": float(np.sqrt(np.mean(both_um**2))),
        "max_um": float(np.max(both_um)),
    }


def _target_metrics(
    *,
    target_name: str,
    initial_exp: np.ndarray,
    final_exp: np.ndarray,
    sim_initial: list[np.ndarray],
    sim_final: list[np.ndarray],
    y_levels_um: list[int],
) -> dict[str, object]:
    profile = _contour_profile_metrics([final_exp], sim_final)
    nearest = _nearest_distance_metrics([final_exp], sim_final)
    per_y: dict[str, dict[str, float]] = {}
    front_errors: list[float] = []
    dx_errors: list[float] = []
    for y_um in y_levels_um:
        y_mm = 1.0e-3 * float(y_um)
        exp_initial_front = front_x_mm(initial_exp, y_mm)
        exp_final_front = front_x_mm(final_exp, y_mm)
        sim_initial_front = _front_x_points_list(sim_initial, y_mm)
        sim_final_front = _front_x_points_list(sim_final, y_mm)
        front_err_um = (
            1.0e3 * (sim_final_front - exp_final_front)
            if np.isfinite(sim_final_front) and np.isfinite(exp_final_front)
            else float("nan")
        )
        exp_dx_um = (
            1.0e3 * (exp_final_front - exp_initial_front)
            if np.isfinite(exp_final_front) and np.isfinite(exp_initial_front)
            else float("nan")
        )
        sim_dx_um = (
            1.0e3 * (sim_final_front - sim_initial_front)
            if np.isfinite(sim_final_front) and np.isfinite(sim_initial_front)
            else float("nan")
        )
        dx_err_um = float(sim_dx_um - exp_dx_um) if np.isfinite(sim_dx_um) and np.isfinite(exp_dx_um) else float("nan")
        if np.isfinite(front_err_um):
            front_errors.append(abs(float(front_err_um)))
        if np.isfinite(dx_err_um):
            dx_errors.append(abs(float(dx_err_um)))
        per_y[str(int(y_um))] = {
            "exp_initial_front_x_mm": float(exp_initial_front),
            "exp_final_front_x_mm": float(exp_final_front),
            "exp_dx_front_um": float(exp_dx_um),
            "sim_initial_front_x_mm": float(sim_initial_front),
            "sim_final_front_x_mm": float(sim_final_front),
            "sim_dx_front_um": float(sim_dx_um),
            "front_error_um": float(front_err_um),
            "dx_error_um": float(dx_err_um),
        }
    return {
        "target": str(target_name),
        "profile": profile,
        "nearest": nearest,
        "mean_front_abs_error_um": float(np.mean(front_errors)) if front_errors else float("nan"),
        "mean_dx_abs_error_um": float(np.mean(dx_errors)) if dx_errors else float("nan"),
        "per_y": per_y,
    }


def _target_metrics_from_timeseries_row(
    *,
    target_name: str,
    initial_exp: np.ndarray,
    final_exp: np.ndarray,
    sim_row: dict[str, str],
    y_levels_um: list[int],
) -> dict[str, object]:
    per_y: dict[str, dict[str, float]] = {}
    front_errors: list[float] = []
    dx_errors: list[float] = []
    for y_um in y_levels_um:
        y_mm = 1.0e-3 * float(y_um)
        exp_initial_front = front_x_mm(initial_exp, y_mm)
        exp_final_front = front_x_mm(final_exp, y_mm)
        sim_final_front = float(sim_row.get(f"x_front_y{int(y_um)}um", float("nan")))
        sim_dx_um = float(sim_row.get(f"dx_front_y{int(y_um)}um", float("nan")))
        sim_initial_front = (
            float(sim_final_front - 1.0e-3 * sim_dx_um) if np.isfinite(sim_final_front) and np.isfinite(sim_dx_um) else float("nan")
        )
        front_err_um = (
            1.0e3 * (sim_final_front - exp_final_front)
            if np.isfinite(sim_final_front) and np.isfinite(exp_final_front)
            else float("nan")
        )
        exp_dx_um = (
            1.0e3 * (exp_final_front - exp_initial_front)
            if np.isfinite(exp_final_front) and np.isfinite(exp_initial_front)
            else float("nan")
        )
        dx_err_um = float(sim_dx_um - exp_dx_um) if np.isfinite(sim_dx_um) and np.isfinite(exp_dx_um) else float("nan")
        if np.isfinite(front_err_um):
            front_errors.append(abs(float(front_err_um)))
        if np.isfinite(dx_err_um):
            dx_errors.append(abs(float(dx_err_um)))
        per_y[str(int(y_um))] = {
            "exp_initial_front_x_mm": float(exp_initial_front),
            "exp_final_front_x_mm": float(exp_final_front),
            "exp_dx_front_um": float(exp_dx_um),
            "sim_initial_front_x_mm": float(sim_initial_front),
            "sim_final_front_x_mm": float(sim_final_front),
            "sim_dx_front_um": float(sim_dx_um),
            "front_error_um": float(front_err_um),
            "dx_error_um": float(dx_err_um),
        }
    return {
        "target": str(target_name),
        "profile": {
            "y_min_mm": float("nan"),
            "y_max_mm": float("nan"),
            "rmse_um": float("nan"),
            "mae_um": float("nan"),
            "max_um": float("nan"),
        },
        "nearest": {"mean_um": float("nan"), "rmse_um": float("nan"), "max_um": float("nan")},
        "mean_front_abs_error_um": float(np.mean(front_errors)) if front_errors else float("nan"),
        "mean_dx_abs_error_um": float(np.mean(dx_errors)) if dx_errors else float("nan"),
        "per_y": per_y,
    }


def _nanmean_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def compare_case(
    *,
    out_dir: Path,
    target_time: float,
    initial_time: float = 0.0,
    initial_csv: Path | None = None,
    final_csv: Path | None = None,
    final_alt_csv: Path | None = None,
    y_levels_um: list[int] | None = None,
    geometry_dir: Path = DEFAULT_GEOM_DIR,
) -> dict[str, object]:
    if y_levels_um is None:
        y_levels_um = [150, 250, 350]

    geom = ensure_geometry_artifacts(force=False, out_dir=Path(geometry_dir))
    initial_path = Path(str(initial_csv or geom["contour_files"]["initial"]))
    final_path = Path(str(final_csv or geom["contour_files"]["final_primary"]))
    final_alt_path = Path(str(final_alt_csv or geom["contour_files"]["final_alternative"]))

    initial_exp = _read_plain_contour_mm(initial_path)
    final_exp = _read_plain_contour_mm(final_path)
    final_alt = _read_plain_contour_mm(final_alt_path)
    comparison_mode = "contour"
    try:
        sim_initial = _read_snapshot_contours_mm(find_snapshot(Path(out_dir), float(initial_time)))
        sim_final = _read_snapshot_contours_mm(find_snapshot(Path(out_dir), float(target_time)))
        primary = _target_metrics(
            target_name="final_primary",
            initial_exp=initial_exp,
            final_exp=final_exp,
            sim_initial=sim_initial,
            sim_final=sim_final,
            y_levels_um=list(y_levels_um),
        )
        alternate = _target_metrics(
            target_name="final_alternative",
            initial_exp=initial_exp,
            final_exp=final_alt,
            sim_initial=sim_initial,
            sim_final=sim_final,
            y_levels_um=list(y_levels_um),
        )
    except (FileNotFoundError, ValueError):
        comparison_mode = "front_only_timeseries"
        sim_row = _read_timeseries_row(Path(out_dir), float(target_time))
        primary = _target_metrics_from_timeseries_row(
            target_name="final_primary",
            initial_exp=initial_exp,
            final_exp=final_exp,
            sim_row=sim_row,
            y_levels_um=list(y_levels_um),
        )
        alternate = _target_metrics_from_timeseries_row(
            target_name="final_alternative",
            initial_exp=initial_exp,
            final_exp=final_alt,
            sim_row=sim_row,
            y_levels_um=list(y_levels_um),
        )

    combined_profile_rmse = _nanmean_or_nan(
        [
            float(primary["profile"]["rmse_um"]),
            float(alternate["profile"]["rmse_um"]),
        ]
    )
    combined_mean_dx_abs = _nanmean_or_nan(
        [
            float(primary["mean_dx_abs_error_um"]),
            float(alternate["mean_dx_abs_error_um"]),
        ]
    )
    combined_front_abs = _nanmean_or_nan(
        [
            float(primary["mean_front_abs_error_um"]),
            float(alternate["mean_front_abs_error_um"]),
        ]
    )
    combined_nearest_mean = _nanmean_or_nan(
        [
            float(primary["nearest"]["mean_um"]),
            float(alternate["nearest"]["mean_um"]),
        ]
    )
    combined_nearest_max = _nanmean_or_nan(
        [
            float(primary["nearest"]["max_um"]),
            float(alternate["nearest"]["max_um"]),
        ]
    )
    score_num = 0.0
    score_den = 0.0
    if np.isfinite(combined_profile_rmse):
        score_num += 0.7 * float(combined_profile_rmse)
        score_den += 0.7
    if np.isfinite(combined_mean_dx_abs):
        score_num += 0.3 * float(combined_mean_dx_abs)
        score_den += 0.3
    combined_score = float(score_num / score_den) if score_den > 0.0 else float("nan")

    return {
        "out_dir": str(Path(out_dir).resolve()),
        "target_time_s": float(target_time),
        "initial_time_s": float(initial_time),
        "y_levels_um": [int(v) for v in y_levels_um],
        "geometry_dir": str(Path(geometry_dir).resolve()),
        "comparison_mode": str(comparison_mode),
        "primary": primary,
        "alternate": alternate,
        "combined": {
            "profile_rmse_um": float(combined_profile_rmse),
            "mean_dx_abs_error_um": float(combined_mean_dx_abs),
            "mean_front_abs_error_um": float(combined_front_abs),
            "nearest_mean_um": float(combined_nearest_mean),
            "nearest_max_um": float(combined_nearest_max),
            "score": float(combined_score),
        },
    }


def _parse_y_levels(raw: str) -> list[int]:
    out: list[int] = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        out.append(int(float(text)))
    if not out:
        raise ValueError("Expected at least one y-level.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--target-time", type=float, required=True)
    ap.add_argument("--initial-time", type=float, default=0.0)
    ap.add_argument("--geometry-dir", type=str, default=str(DEFAULT_GEOM_DIR))
    ap.add_argument("--initial-csv", type=str, default="")
    ap.add_argument("--final-csv", type=str, default="")
    ap.add_argument("--final-alt-csv", type=str, default="")
    ap.add_argument("--y-levels-um", type=str, default="150,250,350")
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    payload = compare_case(
        out_dir=Path(str(args.out_dir)),
        target_time=float(args.target_time),
        initial_time=float(args.initial_time),
        initial_csv=Path(str(args.initial_csv)) if str(args.initial_csv).strip() else None,
        final_csv=Path(str(args.final_csv)) if str(args.final_csv).strip() else None,
        final_alt_csv=Path(str(args.final_alt_csv)) if str(args.final_alt_csv).strip() else None,
        y_levels_um=_parse_y_levels(str(args.y_levels_um)),
        geometry_dir=Path(str(args.geometry_dir)),
    )
    if str(args.json_out).strip():
        Path(str(args.json_out)).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
