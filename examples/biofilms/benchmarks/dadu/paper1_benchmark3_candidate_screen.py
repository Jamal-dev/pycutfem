#!/usr/bin/env python3
"""Screen archived one-domain Duddu Example 2 runs for Paper 1 Benchmark 3."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from examples.biofilms.benchmarks.dadu.paper1_benchmark3_duddu2007_growth import (
    DEFAULT_XFEM_DIR,
    REPO_ROOT,
    TARGET_TIMES,
    _alpha_grid_from_map,
    _fmt_float,
    _interp_rectilinear,
    _interp_time_series,
    _load_alpha_grid_config,
    _mask_metrics,
    _read_timeseries,
    _rectilinear_interp_plan,
)


def _resolve_path(raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _candidate_dirs(results_root: Path, *, required_tags: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for path in sorted(results_root.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if "28p6" not in name:
            continue
        if not all((path / fname).exists() for fname in ("summary.json", "snaps_alpha.npz", "y_top_timeseries.csv")):
            continue
        if not any(tag in name for tag in required_tags):
            continue
        out.append(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Screen archived one-domain Duddu Example 2 runs against the XFEM reference.")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument(
        "--results-root",
        type=str,
        default="examples/biofilms/benchmarks/dadu/results",
    )
    ap.add_argument("--xfem-dir", type=str, default=str(DEFAULT_XFEM_DIR))
    ap.add_argument("--eval-grid", type=int, default=80)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--require-full-targets", action="store_true")
    args = ap.parse_args()

    outdir = _resolve_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results_root = _resolve_path(args.results_root)
    xfem_dir = _resolve_path(args.xfem_dir)

    xfem_snaps = np.load(xfem_dir / "snaps_phi.npz")
    t_ref = np.asarray(xfem_snaps["t_days"], dtype=float).ravel()
    phi_ref = np.asarray(xfem_snaps["phi"], dtype=float)
    grid_x_ref = np.asarray(xfem_snaps["grid_x"], dtype=float).ravel()
    grid_y_ref = np.asarray(xfem_snaps["grid_y"], dtype=float).ravel()
    t_y_ref, y_ref = _read_timeseries(xfem_dir / "y_top_timeseries.csv")

    eval_grid = int(args.eval_grid)
    x_cell = np.linspace(
        float(grid_x_ref[0]) + 0.5 * (float(grid_x_ref[-1] - grid_x_ref[0]) / float(eval_grid)),
        float(grid_x_ref[-1]) - 0.5 * (float(grid_x_ref[-1] - grid_x_ref[0]) / float(eval_grid)),
        int(eval_grid),
    )
    y_cell = np.linspace(
        float(grid_y_ref[0]) + 0.5 * (float(grid_y_ref[-1] - grid_y_ref[0]) / float(eval_grid)),
        float(grid_y_ref[-1]) - 0.5 * (float(grid_y_ref[-1] - grid_y_ref[0]) / float(eval_grid)),
        int(eval_grid),
    )
    ref_plan = _rectilinear_interp_plan(grid_x_ref, grid_y_ref, x_cell, y_cell)
    ref_idx = [int(np.argmin(np.abs(t_ref - float(target)))) for target in TARGET_TIMES.tolist()]

    rows: list[dict[str, object]] = []
    for path in _candidate_dirs(
        results_root,
        required_tags=("oneDom", "one_domain", "fig6_example2", "benchmark3"),
    ):
        t_cmp_y, y_cmp = _read_timeseries(path / "y_top_timeseries.csv")
        t_min = max(float(t_y_ref[0]), float(t_cmp_y[0]))
        t_max = min(float(t_y_ref[-1]), float(t_cmp_y[-1]))
        targets = TARGET_TIMES[(TARGET_TIMES >= t_min - 1.0e-12) & (TARGET_TIMES <= t_max + 1.0e-12)]
        if int(args.require_full_targets) and int(targets.size) != int(TARGET_TIMES.size):
            continue

        y_ref_t = np.interp(targets, t_y_ref, y_ref)
        y_cmp_t = np.interp(targets, t_cmp_y, y_cmp)
        y_err = np.abs(y_cmp_t - y_ref_t)

        one_snaps = np.load(path / "snaps_alpha.npz")
        t_cmp = np.asarray(one_snaps["t_days"], dtype=float).ravel()
        alpha_cmp = np.asarray(one_snaps["alpha"], dtype=float)
        xs_alpha, ys_alpha, ii_alpha, jj_alpha = _load_alpha_grid_config(path / "summary.json")
        alpha_plan = _rectilinear_interp_plan(xs_alpha, ys_alpha, x_cell, y_cell)

        area_abs: list[float] = []
        centroid: list[float] = []
        shape: list[float] = []
        profile: list[float] = []
        active_targets: list[float] = []
        for target, j_ref in zip(TARGET_TIMES.tolist(), ref_idx):
            if target < float(t_cmp[0]) - 1.0e-12 or target > float(t_cmp[-1]) + 1.0e-12:
                continue
            phi_grid = np.asarray(phi_ref[j_ref], dtype=float)
            alpha_dof = _interp_time_series(t_cmp, alpha_cmp, float(target))
            alpha_grid = _alpha_grid_from_map(
                xs=xs_alpha,
                ys=ys_alpha,
                ii=ii_alpha,
                jj=jj_alpha,
                alpha_dof=alpha_dof,
            )
            phi_eval = _interp_rectilinear(
                grid_x_ref,
                grid_y_ref,
                phi_grid,
                x_cell,
                y_cell,
                plan=ref_plan,
            )
            alpha_eval = _interp_rectilinear(
                xs_alpha,
                ys_alpha,
                alpha_grid,
                x_cell,
                y_cell,
                plan=alpha_plan,
            )
            metrics = _mask_metrics(phi_eval <= 0.0, alpha_eval >= 0.5, x_cell, y_cell)
            active_targets.append(float(target))
            area_abs.append(float(metrics["area_abs_err_mm2"]))
            centroid.append(float(metrics["centroid_err_mm"]))
            shape.append(float(metrics["shape_mismatch"]))
            profile.append(float(metrics["profile_mae_mm"]))

        rows.append(
            {
                "candidate": path.name,
                "n_targets": int(len(active_targets)),
                "y_top_mae_mm": float(np.mean(y_err)),
                "y_top_final_mm": float(y_err[-1]),
                "area_mae_mm2": float(np.mean(area_abs)),
                "centroid_max_mm": float(np.nanmax(np.asarray(centroid, dtype=float))),
                "shape_mae": float(np.mean(shape)),
                "shape_final": float(shape[-1]),
                "profile_final_mm": float(profile[-1]),
            }
        )

    rows.sort(
        key=lambda row: (
            float(row["shape_final"]),
            float(row["shape_mae"]),
            float(row["y_top_mae_mm"]),
        )
    )

    csv_path = outdir / "benchmark3_candidate_screen.csv"
    json_path = outdir / "benchmark3_candidate_screen.json"
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")

    print(f"- Wrote {csv_path}")
    print(f"- Wrote {json_path}")
    for row in rows[: max(1, int(args.top_k))]:
        print(
            "[screen] "
            f"shape(T)={_fmt_float(float(row['shape_final']))} "
            f"shape(MAE)={_fmt_float(float(row['shape_mae']))} "
            f"y_top(MAE)={_fmt_float(float(row['y_top_mae_mm']))} "
            f"targets={int(row['n_targets']):02d} "
            f"{row['candidate']}"
        )


if __name__ == "__main__":
    main()
