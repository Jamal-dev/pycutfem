"""
Compare one-domain simulation output against Blauert video-extracted front displacement.

Inputs
------
Experimental CSV (video extractor):
  examples/biofilms/benchmarks/blauert/exp_front_displacement_from_video.csv

Simulation CSV (one-domain driver):
  <out_dir>/timeseries.csv

The experimental file stores displacements in microns; the simulation file stores
displacements in meters. This script converts simulation values to microns and
computes simple error metrics (MAE/RMSE/max).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


def _read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        cols: dict[str, list[float]] = {str(name): [] for name in reader.fieldnames}
        for row in reader:
            for k in cols:
                v = row.get(k, "")
                try:
                    cols[k].append(float(v))
                except Exception:
                    cols[k].append(float("nan"))
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def _interp_1d(t_src: np.ndarray, y_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    t_src = np.asarray(t_src, dtype=float).ravel()
    y_src = np.asarray(y_src, dtype=float).ravel()
    t_tgt = np.asarray(t_tgt, dtype=float).ravel()
    mask = np.isfinite(t_src) & np.isfinite(y_src)
    t_src = t_src[mask]
    y_src = y_src[mask]
    if t_src.size < 2:
        return np.full_like(t_tgt, float("nan"), dtype=float)
    order = np.argsort(t_src)
    t_src = t_src[order]
    y_src = y_src[order]
    # Keep only unique times (np.interp requires increasing x).
    keep = np.r_[True, np.diff(t_src) > 0.0]
    t_src = t_src[keep]
    y_src = y_src[keep]
    if t_src.size < 2:
        return np.full_like(t_tgt, float("nan"), dtype=float)

    t0 = float(t_src[0])
    t1 = float(t_src[-1])
    out = np.full_like(t_tgt, float("nan"), dtype=float)
    in_range = (t_tgt >= t0) & (t_tgt <= t1)
    if not np.any(in_range):
        return out
    out[in_range] = np.interp(t_tgt[in_range], t_src, y_src)
    return out


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "max_abs": float("nan"), "bias": float("nan")}
    err = y_pred[m] - y_true[m]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    max_abs = float(np.max(np.abs(err)))
    bias = float(np.mean(err))
    return {"n": int(np.sum(m)), "mae": mae, "rmse": rmse, "max_abs": max_abs, "bias": bias}


def _select_time_window(t: np.ndarray, *, t_min: float | None, t_max: float | None) -> np.ndarray:
    t = np.asarray(t, dtype=float).ravel()
    m = np.isfinite(t)
    if t_min is not None:
        m &= t >= float(t_min)
    if t_max is not None:
        m &= t <= float(t_max)
    return np.nonzero(m)[0]


def _parse_y_levels_from_columns(cols: list[str], *, kind: str) -> dict[int, str]:
    """
    Return mapping y_um -> column name for dx-front columns.

    kind: "exp" expects `dx_front_y{N}um_um`
          "sim" expects `dx_front_y{N}um`
    """
    if kind == "exp":
        pat = re.compile(r"^dx_front_y(\d+)um_um$")
    elif kind == "sim":
        pat = re.compile(r"^dx_front_y(\d+)um$")
    else:
        raise ValueError(kind)
    out: dict[int, str] = {}
    for c in cols:
        m = pat.match(str(c))
        if not m:
            continue
        out[int(m.group(1))] = str(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare one-domain Blauert simulation dx_front(t) to video extractor CSV.")
    ap.add_argument(
        "--exp-csv",
        type=str,
        default="examples/biofilms/benchmarks/blauert/exp_front_displacement_from_video.csv",
        help="Experimental CSV (video extractor).",
    )
    ap.add_argument("--out-dir", type=str, default=None, help="Simulation out-dir containing timeseries.csv.")
    ap.add_argument("--sim-csv", type=str, default=None, help="Simulation CSV (timeseries.csv). Overrides --out-dir.")
    ap.add_argument("--t-min", type=float, default=None, help="Min time [s] to compare.")
    ap.add_argument("--t-max", type=float, default=None, help="Max time [s] to compare.")
    ap.add_argument(
        "--sample-on",
        type=str,
        default="exp",
        choices=("exp", "sim"),
        help="Which time grid to sample on (other series is interpolated).",
    )
    ap.add_argument(
        "--compare",
        type=str,
        default="global",
        choices=("global", "per-y", "all"),
        help="Which displacement series to compare.",
    )
    ap.add_argument("--y-tol-um", type=float, default=30.0, help="Warn if exp y is more than this far from sim y [um].")
    ap.add_argument("--json-out", type=str, default=None, help="Optional path to write metrics as JSON.")
    args = ap.parse_args()

    exp_path = Path(str(args.exp_csv))
    if not exp_path.exists():
        raise FileNotFoundError(exp_path)

    if args.sim_csv is not None:
        sim_path = Path(str(args.sim_csv))
    else:
        if args.out_dir is None:
            raise SystemExit("Provide --sim-csv or --out-dir.")
        sim_path = Path(str(args.out_dir)) / "timeseries.csv"
    if not sim_path.exists():
        raise FileNotFoundError(sim_path)

    exp = _read_csv_columns(exp_path)
    sim = _read_csv_columns(sim_path)

    if "t_s" not in exp:
        raise ValueError(f"Experimental CSV missing required column t_s: {exp_path}")
    if "t_s" not in sim:
        raise ValueError(f"Simulation CSV missing required column t_s: {sim_path}")

    t_exp = np.asarray(exp["t_s"], dtype=float).ravel()
    t_sim = np.asarray(sim["t_s"], dtype=float).ravel()

    # Choose target sampling grid.
    # (performed per-series below)

    out: dict[str, object] = {
        "exp_csv": str(exp_path),
        "sim_csv": str(sim_path),
        "t_min": float(args.t_min) if args.t_min is not None else None,
        "t_max": float(args.t_max) if args.t_max is not None else None,
        "sample_on": str(args.sample_on),
        "global": {},
        "per_y": {},
    }

    if str(args.compare) in {"global", "all"}:
        if "dx_front_um" not in exp:
            raise ValueError(f"Experimental CSV missing required column dx_front_um for global compare: {exp_path}")
        if "dx_front_global" not in sim:
            raise ValueError(
                f"Simulation CSV missing required column dx_front_global for global compare: {sim_path}. "
                "Re-run the simulation with the current Blauert driver or use --compare per-y."
            )
        dx_exp_um = np.asarray(exp["dx_front_um"], dtype=float).ravel()
        dx_sim_um = 1.0e6 * np.asarray(sim["dx_front_global"], dtype=float).ravel()

        if str(args.sample_on) == "sim":
            idx = _select_time_window(t_sim, t_min=args.t_min, t_max=args.t_max)
            t_ref = t_sim[idx]
            y_true = _interp_1d(t_exp, dx_exp_um, t_ref)
            y_pred = dx_sim_um[idx]
        else:
            idx = _select_time_window(t_exp, t_min=args.t_min, t_max=args.t_max)
            t_ref = t_exp[idx]
            y_true = dx_exp_um[idx]
            y_pred = _interp_1d(t_sim, dx_sim_um, t_ref)
        m_comp = np.isfinite(t_ref) & np.isfinite(y_true) & np.isfinite(y_pred)
        if np.any(m_comp):
            t0, t1 = float(np.min(t_ref[m_comp])), float(np.max(t_ref[m_comp]))
        else:
            t0, t1 = float("nan"), float("nan")

        m = _metrics(y_true, y_pred)
        out["global"] = {"window_s": [t0, t1], **m}
        print(f"[global] window t=[{t0:.3g}, {t1:.3g}] s; n={m['n']}")
        print(f"  MAE   = {m['mae']:.3g} um")
        print(f"  RMSE  = {m['rmse']:.3g} um")
        print(f"  Max   = {m['max_abs']:.3g} um")
        print(f"  Bias  = {m['bias']:.3g} um")

    if str(args.compare) in {"per-y", "all"}:
        exp_map = _parse_y_levels_from_columns(list(exp.keys()), kind="exp")
        sim_map = _parse_y_levels_from_columns(list(sim.keys()), kind="sim")
        if not exp_map:
            print("[per-y] No exp dx_front_y* columns found; skipping.")
        elif not sim_map:
            print("[per-y] No sim dx_front_y* columns found; skipping.")
        else:
            sim_levels = np.array(sorted(sim_map.keys()), dtype=float)
            per_y: dict[str, object] = {}
            for y_um, exp_col in sorted(exp_map.items()):
                j = int(np.argmin(np.abs(sim_levels - float(y_um))))
                y_sim_um = int(sim_levels[j])
                sim_col = sim_map.get(y_sim_um)
                if sim_col is None:
                    continue
                if abs(float(y_sim_um) - float(y_um)) > float(args.y_tol_um):
                    print(f"[per-y] warning: exp y={y_um}um mapped to sim y={y_sim_um}um (|dy|>{args.y_tol_um}um)")

                dx_exp_y_um = np.asarray(exp[exp_col], dtype=float).ravel()
                dx_sim_y_um = 1.0e6 * np.asarray(sim[sim_col], dtype=float).ravel()

                if str(args.sample_on) == "sim":
                    idx = _select_time_window(t_sim, t_min=args.t_min, t_max=args.t_max)
                    t_ref = t_sim[idx]
                    y_true = _interp_1d(t_exp, dx_exp_y_um, t_ref)
                    y_pred = dx_sim_y_um[idx]
                else:
                    idx = _select_time_window(t_exp, t_min=args.t_min, t_max=args.t_max)
                    t_ref = t_exp[idx]
                    y_true = dx_exp_y_um[idx]
                    y_pred = _interp_1d(t_sim, dx_sim_y_um, t_ref)
                m_comp = np.isfinite(t_ref) & np.isfinite(y_true) & np.isfinite(y_pred)
                if np.any(m_comp):
                    t0, t1 = float(np.min(t_ref[m_comp])), float(np.max(t_ref[m_comp]))
                else:
                    t0, t1 = float("nan"), float("nan")

                m = _metrics(y_true, y_pred)
                key = f"y{int(y_um)}um"
                per_y[key] = {"exp_y_um": int(y_um), "sim_y_um": int(y_sim_um), "window_s": [t0, t1], **m}
                print(f"[per-y] y={int(y_um)}um (sim y={int(y_sim_um)}um): RMSE={m['rmse']:.3g} um (n={m['n']})")
            out["per_y"] = per_y

    if args.json_out is not None:
        out_path = Path(str(args.json_out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[write] {out_path}")

    # Non-zero exit on no overlap or no valid points
    if str(args.compare) in {"global", "all"}:
        n = int(out.get("global", {}).get("n", 0)) if isinstance(out.get("global", {}), dict) else 0
        if n <= 0:
            raise SystemExit("No valid comparison points (global). Check --t-min/--t-max and input CSVs.")
    if str(args.compare) in {"per-y"}:
        per_y = out.get("per_y", {})
        if isinstance(per_y, dict) and per_y:
            # at least one series should have points
            if not any(int(v.get("n", 0)) > 0 for v in per_y.values() if isinstance(v, dict)):
                raise SystemExit("No valid comparison points (per-y). Check --t-min/--t-max and input CSVs.")


if __name__ == "__main__":
    main()
