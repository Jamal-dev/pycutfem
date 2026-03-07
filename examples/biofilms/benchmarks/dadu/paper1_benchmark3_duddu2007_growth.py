#!/usr/bin/env python3
"""Paper 1 Benchmark 3: two-domain vs one-domain Duddu (2007) growth comparison.

This driver assembles the publishable comparison package for the three-colony
growth benchmark:

- the two-domain XFEM + level-set reference reproduction,
- the one-domain diffuse-interface reproductions on a small mesh ladder,
- manuscript-grade aggregate metrics and figures.

By default the script reuses the validated raw runs already stored under
``examples/biofilms/benchmarks/dadu/results/``. With ``--raw-policy run`` it can
also regenerate those raw runs in place.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np

from examples.biofilms.benchmarks.dadu.plot_one_domain_interface_from_snaps import (
    _alpha_grid_from_map,
    _build_alpha_grid_map,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
TARGET_TIMES = np.asarray(
    [0.0, 1.1, 2.7, 4.6, 6.6, 8.7, 10.7, 12.7, 14.7, 16.7, 18.6, 20.6, 22.5, 24.5, 26.5, 28.6],
    dtype=float,
)
DEFAULT_XFEM_DIR = Path(
    "examples/biofilms/benchmarks/dadu/results/paper1_benchmark3_xfem_reference"
)
DEFAULT_ONE_DOMAIN_DIRS = {
    60: Path(
        "examples/biofilms/benchmarks/dadu/results/paper1_benchmark3_one_domain_nx060"
    ),
    80: Path(
        "examples/biofilms/benchmarks/dadu/results/paper1_benchmark3_one_domain_nx080"
    ),
}
NUMBA_DEBUG_ENV_KEYS = (
    "NUMBA_DEBUG",
    "NUMBA_DUMP_BYTECODE",
    "NUMBA_DUMP_IR",
    "NUMBA_DUMP_SSA",
    "NUMBA_DEBUG_ARRAY_OPT",
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_timeseries(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_csv_rows(path)
    t = np.asarray([float(r["t_days"]) for r in rows], dtype=float)
    y = np.asarray([float(r["y_top_mm"]) for r in rows], dtype=float)
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    keep = np.ones(t.size, dtype=bool)
    if t.size > 1:
        keep[:-1] = t[1:] != t[:-1]
    return t[keep], y[keep]


def _fmt_float(val: float, digits: int = 3) -> str:
    return f"{float(val):.{digits}e}"


def _conda_python(env_name: str, script_rel: str, *args: str) -> list[str]:
    script = str((REPO_ROOT / script_rel).resolve())
    current = str(os.environ.get("CONDA_DEFAULT_ENV", "")).strip()
    if current == str(env_name).strip():
        return [sys.executable, "-u", script, *[str(a) for a in args]]
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        str(env_name),
        "python",
        "-u",
        script,
        *[str(a) for a in args],
    ]


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in NUMBA_DEBUG_ENV_KEYS:
        env[key] = "0"
    return env


def _run_logged(cmd: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(
        [part for part in cmd if part],
        cwd=str(cwd),
        env=_clean_env(),
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with rc={proc.returncode}: {shlex.join(cmd)}")


def _parse_nx_list(raw: str) -> list[int]:
    out: list[int] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError("No one-domain meshes selected.")
    return sorted(set(out))


def _resolve_path(raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _parse_one_domain_dirs(raw_items: list[str], *, nx_list: list[int]) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for item in raw_items:
        if "=" not in str(item):
            raise ValueError(f"Expected NX=PATH for --one-domain-dir, got {item!r}.")
        nx_raw, path_raw = str(item).split("=", 1)
        parsed[int(nx_raw)] = _resolve_path(path_raw)
    for nx in nx_list:
        if nx in parsed:
            continue
        default = DEFAULT_ONE_DOMAIN_DIRS.get(int(nx))
        if default is None:
            raise ValueError(f"No default raw directory is known for nx={nx}.")
        parsed[int(nx)] = _resolve_path(default)
    return parsed


def _load_alpha_grid_config(summary_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    xs, ys, ii, jj = _build_alpha_grid_map(
        L=float(summary["L_mm"]),
        H=float(summary["H_mm"]),
        nx=int(summary["nx"]),
        ny=int(summary["ny"]),
    )
    return xs, ys, ii, jj


def _interp_time_series(t: np.ndarray, values: np.ndarray, target: float) -> np.ndarray:
    if target <= float(t[0]) + 1.0e-14:
        return np.asarray(values[0], dtype=float)
    if target >= float(t[-1]) - 1.0e-14:
        return np.asarray(values[-1], dtype=float)
    hi = int(np.searchsorted(t, float(target), side="right"))
    lo = hi - 1
    t0 = float(t[lo])
    t1 = float(t[hi])
    if abs(t1 - t0) <= 1.0e-14:
        return np.asarray(values[hi], dtype=float)
    w = (float(target) - t0) / (t1 - t0)
    return (1.0 - w) * np.asarray(values[lo], dtype=float) + w * np.asarray(values[hi], dtype=float)


def _axis_interp_plan(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=float).ravel()
    dst = np.asarray(dst, dtype=float).ravel()
    if src.size < 2:
        raise ValueError("Need at least two source coordinates for interpolation.")

    below = dst <= float(src[0]) + 1.0e-14
    above = dst >= float(src[-1]) - 1.0e-14
    hi = np.searchsorted(src, dst, side="right")
    hi = np.clip(hi, 1, src.size - 1)
    lo = hi - 1

    lo[below] = 0
    hi[below] = 0
    lo[above] = src.size - 1
    hi[above] = src.size - 1

    denom = src[hi] - src[lo]
    w = np.zeros(dst.shape, dtype=float)
    valid = (~below) & (~above) & (np.abs(denom) > 1.0e-14)
    w[valid] = (dst[valid] - src[lo[valid]]) / denom[valid]
    return lo.astype(int), hi.astype(int), w


def _rectilinear_interp_plan(
    xs_src: np.ndarray,
    ys_src: np.ndarray,
    xs_dst: np.ndarray,
    ys_dst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ix0, ix1, wx = _axis_interp_plan(xs_src, xs_dst)
    iy0, iy1, wy = _axis_interp_plan(ys_src, ys_dst)
    return ix0, ix1, wx, iy0, iy1, wy


def _interp_rectilinear(
    xs_src: np.ndarray,
    ys_src: np.ndarray,
    values: np.ndarray,
    xs_dst: np.ndarray,
    ys_dst: np.ndarray,
    *,
    plan: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    src = np.asarray(values, dtype=float)
    if src.ndim != 2:
        raise ValueError("Rectilinear interpolation expects a 2D field.")
    if plan is None:
        plan = _rectilinear_interp_plan(xs_src, ys_src, xs_dst, ys_dst)
    ix0, ix1, wx, iy0, iy1, wy = plan
    if src.shape != (int(np.asarray(ys_src).size), int(np.asarray(xs_src).size)):
        raise ValueError("Source field shape does not match the source grid.")

    v00 = src[iy0[:, None], ix0[None, :]]
    v01 = src[iy0[:, None], ix1[None, :]]
    v10 = src[iy1[:, None], ix0[None, :]]
    v11 = src[iy1[:, None], ix1[None, :]]

    wx2 = wx.reshape(1, -1)
    wy2 = wy.reshape(-1, 1)
    top = (1.0 - wx2) * v00 + wx2 * v01
    bot = (1.0 - wx2) * v10 + wx2 * v11
    return (1.0 - wy2) * top + wy2 * bot


def _mask_metrics(mask_ref: np.ndarray, mask_cmp: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> dict[str, float]:
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    area_ref = float(np.sum(mask_ref) * dx * dy)
    area_cmp = float(np.sum(mask_cmp) * dx * dy)
    area_abs_err = abs(area_cmp - area_ref)
    symdiff = float(np.sum(np.logical_xor(mask_ref, mask_cmp)) * dx * dy)
    union_area = float(np.sum(np.logical_or(mask_ref, mask_cmp)) * dx * dy)
    shape_mismatch = symdiff / max(union_area, 1.0e-14)

    def _centroid(mask: np.ndarray) -> tuple[float, float]:
        idx = np.argwhere(mask)
        if idx.size == 0:
            return float("nan"), float("nan")
        xx = xs[idx[:, 1]]
        yy = ys[idx[:, 0]]
        return float(np.mean(xx)), float(np.mean(yy))

    cx_ref, cy_ref = _centroid(mask_ref)
    cx_cmp, cy_cmp = _centroid(mask_cmp)
    if math.isfinite(cx_ref) and math.isfinite(cx_cmp) and math.isfinite(cy_ref) and math.isfinite(cy_cmp):
        centroid_err = math.hypot(cx_cmp - cx_ref, cy_cmp - cy_ref)
    else:
        centroid_err = float("nan")

    def _top_profile(mask: np.ndarray) -> np.ndarray:
        top = np.zeros(mask.shape[1], dtype=float)
        for i in range(mask.shape[1]):
            col = np.flatnonzero(mask[:, i])
            top[i] = float(ys[col[-1]]) if col.size else 0.0
        return top

    top_ref = _top_profile(mask_ref)
    top_cmp = _top_profile(mask_cmp)
    profile_mae = float(np.mean(np.abs(top_cmp - top_ref)))

    return {
        "area_ref_mm2": area_ref,
        "area_cmp_mm2": area_cmp,
        "area_abs_err_mm2": area_abs_err,
        "centroid_err_mm": float(centroid_err),
        "shape_mismatch": float(shape_mismatch),
        "profile_mae_mm": profile_mae,
    }


def _ensure_xfem_outputs(
    *,
    xfem_dir: Path,
    raw_policy: str,
    conda_env: str,
) -> None:
    needed = [
        xfem_dir / "y_top_timeseries.csv",
        xfem_dir / "summary.json",
        xfem_dir / "fig6a_interface.png",
        xfem_dir / "fig6b_S.png",
        xfem_dir / "fig6c_Phi.png",
        xfem_dir / "snaps_phi.npz",
    ]
    if raw_policy == "existing" and all(path.exists() for path in needed):
        return
    if raw_policy == "existing":
        missing = ", ".join(str(path) for path in needed if not path.exists())
        raise FileNotFoundError(f"Missing XFEM benchmark outputs: {missing}")
    if raw_policy == "auto" and all(path.exists() for path in needed):
        return

    xfem_dir.mkdir(parents=True, exist_ok=True)
    cmd = _conda_python(
        conda_env,
        "examples/biofilms/benchmarks/dadu/duddu2007_growth_2d_fig6_example2.py",
        "--backend",
        "cpp",
        "--linear-solver",
        "petsc",
        "--mesh-nx",
        "20",
        "--mesh-ny",
        "20",
        "--grid-nx",
        "200",
        "--grid-ny",
        "200",
        "--q",
        "2",
        "--speed-mode",
        "qp",
        "--substrate-bc",
        "moving",
        "--Ls",
        "0.1",
        "--S-penalty",
        "1e6",
        "--t-final",
        "28.6",
        "--reinit-every",
        "1",
        "--outdir",
        str(xfem_dir),
    )
    _run_logged(cmd, cwd=REPO_ROOT)


def _ensure_one_domain_outputs(
    *,
    nx: int,
    run_dir: Path,
    xfem_dir: Path,
    need_panels: bool,
    raw_policy: str,
    conda_env: str,
) -> None:
    raw_core = [
        run_dir / "summary.json",
        run_dir / "snaps_alpha.npz",
        run_dir / "y_top_timeseries.csv",
    ]
    core_needed = list(raw_core) + [run_dir / "y_top_compare.csv"]
    panel_needed = [
        run_dir / "fig6a_interface.png",
        run_dir / "fig6b_S.png",
        run_dir / "fig6c_Phi.png",
        run_dir / "y_top_compare.png",
    ]
    needed = list(core_needed) + (panel_needed if need_panels else [])
    if raw_policy == "existing" and all(path.exists() for path in needed):
        return
    if all(path.exists() for path in raw_core):
        if not (run_dir / "y_top_compare.csv").exists():
            _run_logged(
                _conda_python(
                    conda_env,
                    "examples/biofilms/benchmarks/dadu/compare_duddu2007_fig6_y_top.py",
                    "--a",
                    str(xfem_dir),
                    "--b",
                    str(run_dir),
                    "--label-a",
                    "XFEM",
                    "--label-b",
                    "one-domain",
                    "--outdir",
                    str(run_dir),
                ),
                cwd=REPO_ROOT,
            )
        if need_panels and not (run_dir / "fig6a_interface.png").exists():
            _run_logged(
                _conda_python(
                    conda_env,
                    "examples/biofilms/benchmarks/dadu/plot_one_domain_interface_from_snaps.py",
                    "--results-dir",
                    str(run_dir),
                    "--paper-times",
                    "--out",
                    str(run_dir / "fig6a_interface.png"),
                ),
                cwd=REPO_ROOT,
            )
        if need_panels and (
            (not (run_dir / "fig6b_S.png").exists())
            or (not (run_dir / "fig6c_Phi.png").exists())
        ):
            if (run_dir / "final_fields.npz").exists():
                _run_logged(
                    _conda_python(
                        conda_env,
                        "examples/biofilms/benchmarks/dadu/plot_one_domain_fig6_panels_from_npz.py",
                        "--results-dir",
                        str(run_dir),
                    ),
                    cwd=REPO_ROOT,
                )
        if all(path.exists() for path in needed):
            return

    if raw_policy == "existing":
        missing = ", ".join(str(path) for path in needed if not path.exists())
        raise FileNotFoundError(f"Missing one-domain benchmark outputs for nx={nx}: {missing}")
    if raw_policy != "run" and all(path.exists() for path in needed):
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = _conda_python(
        conda_env,
        "examples/biofilms/benchmarks/dadu/duddu2007_one_domain_growth_2d_fig6_example2.py",
        "--backend",
        "cpp",
        "--linear-solver",
        "petsc",
        "--substrate-solver",
        "newton",
        "--substrate-advection",
        "off",
        "--alpha-advect-with",
        "mix",
        "--phi-update",
        "mix",
        "--D-S",
        "120",
        "--gamma-vS",
        "0.1",
        "--vS-ext-mode",
        "l2",
        "--D-alpha",
        "0",
        "--ac-M",
        "1",
        "--ac-gamma",
        "5e-4",
        "--ac-mobility",
        "degenerate",
        "--ac-mobility-floor",
        "0.1",
        "--q",
        "2",
        "--nx",
        str(int(nx)),
        "--ny",
        str(int(nx)),
        "--dt",
        "0.2",
        "--t-final",
        "28.6",
        "--newton-tol",
        "1e-8",
        "--max-it",
        "20",
        "--progress-every",
        "20",
        "--write-every",
        "1",
        "--flush-snaps-every",
        "20",
        "--skip-plots",
        "--outdir",
        str(run_dir),
    )
    _run_logged(cmd, cwd=REPO_ROOT)

    _run_logged(
        _conda_python(
            conda_env,
            "examples/biofilms/benchmarks/dadu/plot_one_domain_interface_from_snaps.py",
            "--results-dir",
            str(run_dir),
            "--paper-times",
            "--out",
            str(run_dir / "fig6a_interface.png"),
        ),
        cwd=REPO_ROOT,
    )
    _run_logged(
        _conda_python(
            conda_env,
            "examples/biofilms/benchmarks/dadu/compare_duddu2007_fig6_y_top.py",
            "--a",
            str(xfem_dir),
            "--b",
            str(run_dir),
            "--label-a",
            "XFEM",
            "--label-b",
            "one-domain",
            "--outdir",
            str(run_dir),
        ),
        cwd=REPO_ROOT,
    )
    if need_panels:
        _run_logged(
            _conda_python(
                conda_env,
                "examples/biofilms/benchmarks/dadu/plot_one_domain_fig6_panels_from_npz.py",
                "--results-dir",
                str(run_dir),
            ),
            cwd=REPO_ROOT,
        )


def _maybe_make_compare_panel(*, xfem_dir: Path, one_domain_dir: Path, conda_env: str) -> Path:
    out = xfem_dir / "compare_fig6.png"
    if out.exists():
        return out
    _run_logged(
        _conda_python(
            conda_env,
            "examples/biofilms/benchmarks/dadu/compare_duddu2007_fig6.py",
            "--our-dir",
            str(xfem_dir),
            "--our-label",
            "XFEM",
            "--extra-dir",
            str(one_domain_dir),
            "--extra-label",
            "one-domain",
            "--paper-page",
            "page-17.png",
        ),
        cwd=REPO_ROOT,
    )
    return out


def _compute_target_rows(
    *,
    nx: int,
    xfem_dir: Path,
    one_domain_dir: Path,
    eval_grid: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    xfem_snaps = np.load(xfem_dir / "snaps_phi.npz")
    t_ref = np.asarray(xfem_snaps["t_days"], dtype=float).ravel()
    phi_ref = np.asarray(xfem_snaps["phi"], dtype=float)
    grid_x_ref = np.asarray(xfem_snaps["grid_x"], dtype=float).ravel()
    grid_y_ref = np.asarray(xfem_snaps["grid_y"], dtype=float).ravel()

    one_snaps = np.load(one_domain_dir / "snaps_alpha.npz")
    t_cmp = np.asarray(one_snaps["t_days"], dtype=float).ravel()
    alpha_cmp = np.asarray(one_snaps["alpha"], dtype=float)

    one_summary_path = one_domain_dir / "summary.json"
    xs_alpha, ys_alpha, ii_alpha, jj_alpha = _load_alpha_grid_config(one_summary_path)

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
    alpha_plan = _rectilinear_interp_plan(xs_alpha, ys_alpha, x_cell, y_cell)

    y_top_rows = _read_csv_rows(one_domain_dir / "y_top_compare.csv")
    y_top_by_time = {round(float(r["t_days"]), 10): r for r in y_top_rows}
    y_top_col = [k for k in y_top_rows[0] if k.startswith("y_top_") and "XFEM" not in k][0]
    ref_idx = [int(np.argmin(np.abs(t_ref - float(target)))) for target in TARGET_TIMES.tolist()]

    rows: list[dict[str, object]] = []
    for target, j_ref in zip(TARGET_TIMES.tolist(), ref_idx):
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

        mask_ref = phi_eval <= 0.0
        mask_cmp = alpha_eval >= 0.5
        metrics = _mask_metrics(mask_ref, mask_cmp, x_cell, y_cell)

        key = round(float(target), 10)
        y_top_row = y_top_by_time[key]
        t_left = float(t_cmp[max(0, int(np.searchsorted(t_cmp, float(target), side="right")) - 1)])
        t_right = float(t_cmp[min(t_cmp.size - 1, int(np.searchsorted(t_cmp, float(target), side="right")))])
        interp_gap = min(abs(float(target) - t_left), abs(float(target) - t_right))

        rows.append(
            {
                "nx": int(nx),
                "t_days": float(target),
                "xfem_snapshot_days": float(t_ref[j_ref]),
                "one_domain_interp_gap_days": float(interp_gap),
                "y_top_xfem_mm": float(y_top_row["y_top_XFEM_mm"]),
                "y_top_one_domain_mm": float(y_top_row[y_top_col]),
                "y_top_abs_err_mm": float(y_top_row["abs_err_mm"]),
                **metrics,
            }
        )

    y_top_abs = np.asarray([float(r["y_top_abs_err_mm"]) for r in rows], dtype=float)
    area_abs = np.asarray([float(r["area_abs_err_mm2"]) for r in rows], dtype=float)
    centroid = np.asarray([float(r["centroid_err_mm"]) for r in rows], dtype=float)
    shape = np.asarray([float(r["shape_mismatch"]) for r in rows], dtype=float)
    profile = np.asarray([float(r["profile_mae_mm"]) for r in rows], dtype=float)
    interp_gap = np.asarray([float(r["one_domain_interp_gap_days"]) for r in rows], dtype=float)

    summary_row = {
        "nx": int(nx),
        "y_top_mae_mm": float(np.mean(y_top_abs)),
        "y_top_max_mm": float(np.max(y_top_abs)),
        "y_top_final_mm": float(y_top_abs[-1]),
        "area_mae_mm2": float(np.mean(area_abs)),
        "area_max_mm2": float(np.max(area_abs)),
        "area_final_mm2": float(area_abs[-1]),
        "centroid_max_mm": float(np.nanmax(centroid)),
        "centroid_final_mm": float(centroid[-1]),
        "shape_mae": float(np.mean(shape)),
        "shape_max": float(np.max(shape)),
        "shape_final": float(shape[-1]),
        "profile_mae_mm": float(np.mean(profile)),
        "profile_final_mm": float(profile[-1]),
        "interp_gap_max_days": float(np.max(interp_gap)),
        "raw_dir": str(one_domain_dir.relative_to(REPO_ROOT)),
    }
    return rows, summary_row


def _plot_y_top_mesh(
    *,
    xfem_dir: Path,
    one_domain_dirs: dict[int, Path],
    out: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    t_ref, y_ref = _read_timeseries(xfem_dir / "y_top_timeseries.csv")
    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    ax.plot(t_ref, y_ref, color="black", lw=2.0, label="two-domain XFEM")
    for nx, run_dir in sorted(one_domain_dirs.items()):
        t_cmp, y_cmp = _read_timeseries(run_dir / "y_top_timeseries.csv")
        ax.plot(t_cmp, y_cmp, lw=1.6, label=f"one-domain nx={nx}")
    ax.set_xlabel("t (days)")
    ax.set_ylabel(r"$y_{top}$ (mm)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_finest_geometry(rows: list[dict[str, object]], *, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    t = np.asarray([float(r["t_days"]) for r in rows], dtype=float)
    area = np.asarray([float(r["area_abs_err_mm2"]) for r in rows], dtype=float)
    centroid = np.asarray([float(r["centroid_err_mm"]) for r in rows], dtype=float)
    shape = np.asarray([float(r["shape_mismatch"]) for r in rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(6.2, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(t, area, marker="o", ms=3.5, lw=1.4)
    axes[0].set_ylabel(r"$|\Delta A|$ (mm$^2$)")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t, centroid, marker="o", ms=3.5, lw=1.4)
    axes[1].set_ylabel(r"$e_c$ (mm)")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(t, shape, marker="o", ms=3.5, lw=1.4)
    axes[2].set_ylabel(r"$e_{shape}$")
    axes[2].set_xlabel("t (days)")
    axes[2].grid(True, alpha=0.25)

    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper 1 Benchmark 3: Duddu two-domain vs one-domain comparison.")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--conda-env", type=str, default="fenicsx")
    ap.add_argument("--raw-policy", choices=("auto", "existing", "run"), default="auto")
    ap.add_argument("--one-domain-nx-list", type=str, default="60,80")
    ap.add_argument("--eval-grid", type=int, default=400)
    ap.add_argument("--xfem-dir", type=str, default=str(DEFAULT_XFEM_DIR))
    ap.add_argument(
        "--one-domain-dir",
        type=str,
        action="append",
        default=[],
        help="Raw one-domain result directory as NX=PATH. Repeat once per mesh.",
    )
    args = ap.parse_args()

    outdir = _resolve_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nx_list = _parse_nx_list(args.one_domain_nx_list)
    xfem_dir = _resolve_path(args.xfem_dir)
    one_domain_dirs = _parse_one_domain_dirs(args.one_domain_dir, nx_list=nx_list)

    _ensure_xfem_outputs(xfem_dir=xfem_dir, raw_policy=str(args.raw_policy), conda_env=str(args.conda_env))
    finest_nx = max(nx_list)
    for nx, run_dir in sorted(one_domain_dirs.items()):
        _ensure_one_domain_outputs(
            nx=int(nx),
            run_dir=run_dir,
            xfem_dir=xfem_dir,
            need_panels=int(nx) == int(finest_nx),
            raw_policy=str(args.raw_policy),
            conda_env=str(args.conda_env),
        )

    compare_fig = _maybe_make_compare_panel(
        xfem_dir=xfem_dir,
        one_domain_dir=one_domain_dirs[finest_nx],
        conda_env=str(args.conda_env),
    )

    all_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    finest_rows: list[dict[str, object]] = []
    for nx, run_dir in sorted(one_domain_dirs.items()):
        rows, summary_row = _compute_target_rows(
            nx=int(nx),
            xfem_dir=xfem_dir,
            one_domain_dir=run_dir,
            eval_grid=int(args.eval_grid),
        )
        all_rows.extend(rows)
        summary_rows.append(summary_row)
        if int(nx) == finest_nx:
            finest_rows = rows

    summary_rows.sort(key=lambda row: int(row["nx"]))
    finest_rows.sort(key=lambda row: float(row["t_days"]))

    summary_csv = outdir / "benchmark3_duddu_growth_summary.csv"
    target_csv = outdir / "benchmark3_duddu_growth_target_metrics.csv"
    _write_csv(summary_csv, summary_rows)
    _write_csv(target_csv, finest_rows)

    _plot_y_top_mesh(
        xfem_dir=xfem_dir,
        one_domain_dirs=one_domain_dirs,
        out=outdir / "benchmark3_duddu_growth_y_top_mesh.png",
    )
    _plot_finest_geometry(
        finest_rows,
        out=outdir / "benchmark3_duddu_growth_geometry_timeseries.png",
    )

    y_top_compare_png = one_domain_dirs[finest_nx] / "y_top_compare.png"
    if y_top_compare_png.exists():
        shutil.copy2(y_top_compare_png, outdir / "benchmark3_duddu_growth_y_top_finest.png")
    if compare_fig.exists():
        shutil.copy2(compare_fig, outdir / "benchmark3_duddu_growth_compare_fig6.png")

    payload = {
        "benchmark": "duddu2007_growth_compare",
        "xfem_dir": str(xfem_dir.relative_to(REPO_ROOT)),
        "one_domain_dirs": {str(nx): str(path.relative_to(REPO_ROOT)) for nx, path in sorted(one_domain_dirs.items())},
        "summary_rows": summary_rows,
        "finest_rows": finest_rows,
    }
    (outdir / "benchmark3_duddu_growth_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"- Wrote {summary_csv}")
    print(f"- Wrote {target_csv}")
    print(f"- Wrote {outdir/'benchmark3_duddu_growth_summary.json'}")
    print(
        "[benchmark3] finest metrics: "
        f"y_top MAE={_fmt_float(float(summary_rows[-1]['y_top_mae_mm']))} mm, "
        f"final y_top={_fmt_float(float(summary_rows[-1]['y_top_final_mm']))} mm, "
        f"area MAE={_fmt_float(float(summary_rows[-1]['area_mae_mm2']))} mm^2, "
        f"shape(T)={_fmt_float(float(summary_rows[-1]['shape_final']))}"
    )


if __name__ == "__main__":
    main()
