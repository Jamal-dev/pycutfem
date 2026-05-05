#!/usr/bin/env python
"""
Pandas table for Warner (1986) case 4 sloughing comparison.

This script compares the FD ζ-model reference (Warner benchmark) against a
one-domain run CSV at key times around the sloughing event:
  - t = 5.984 d  (just before the window)
  - t = 5.994 d  (end of the window; after sloughing)
  - t = 6.000 d  (paper plots use 6 d)

Run:
  conda run --no-capture-output -n fenicsx \
    python -u examples/biofilms/benchmarks/warner/case4_slough_table.py \
      --backend cpp --run-tag case4_shift_smoke_ny120_dt0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

REF_OUTDIR = REPO / "examples" / "biofilms" / "results" / "warner1986"
ONE_OUTDIR = REPO / "examples" / "biofilms" / "results" / "warner1986_one_domain"


def _interp(df: pd.DataFrame, *, t_col: str, y_col: str, t: float) -> float:
    tt = df[t_col].to_numpy(dtype=float)
    yy = df[y_col].to_numpy(dtype=float)
    tq = float(t)
    if tt.size == 0:
        return float("nan")
    if tq <= float(tt[0]):
        return float(yy[0])
    if tq >= float(tt[-1]):
        return float(yy[-1])
    return float(np.interp(tq, tt, yy))


def main() -> None:
    ap = argparse.ArgumentParser(description="Case 4 sloughing table (Warner FD vs one-domain).")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--run-tag", type=str, required=True, help="Suffix tag used in one-domain output filename.")
    ap.add_argument(
        "--times",
        type=str,
        default="5.984,5.994,6.0",
        help="Comma-separated list of times (days) to include (default: 5.984,5.994,6.0).",
    )
    args = ap.parse_args()

    times = [float(x.strip()) for x in str(args.times).split(",") if x.strip()]
    if not times:
        raise ValueError("--times must be non-empty.")

    ref_path = REF_OUTDIR / "case4_backend=cpp_timeseries.csv"
    one_path = ONE_OUTDIR / f"one_domain_strip_case4_backend={str(args.backend)}_{str(args.run_tag).strip()}_timeseries.csv"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing Warner reference CSV: {ref_path}")
    if not one_path.exists():
        raise FileNotFoundError(f"Missing one-domain CSV: {one_path}")

    ref = pd.read_csv(ref_path)
    one = pd.read_csv(one_path)

    rows: list[dict[str, float]] = []
    for t in times:
        rows.append(
            {
                "t_days": float(t),
                "L_ref_um": _interp(ref, t_col="t_days", y_col="L_um", t=t),
                "L_one_um": _interp(one, t_col="t_days", y_col="L_thickness_um", t=t),
                "removal_ref": _interp(ref, t_col="t_days", y_col="jL_1", t=t) * (-1.0),
                "removal_one": _interp(one, t_col="t_days", y_col="removal", t=t),
            }
        )

    df = pd.DataFrame(rows)
    df["L_rel_err"] = (df["L_one_um"] - df["L_ref_um"]).abs() / df["L_ref_um"].abs().clip(lower=1.0e-12)
    df["removal_rel_err"] = (df["removal_one"] - df["removal_ref"]).abs() / df["removal_ref"].abs().clip(lower=1.0e-12)

    # Sloughing drop between the first two times (if provided).
    if len(times) >= 2:
        t0 = float(times[0])
        t1 = float(times[1])
        dL_ref = float(df.loc[df["t_days"] == t1, "L_ref_um"].iloc[0] - df.loc[df["t_days"] == t0, "L_ref_um"].iloc[0])
        dL_one = float(df.loc[df["t_days"] == t1, "L_one_um"].iloc[0] - df.loc[df["t_days"] == t0, "L_one_um"].iloc[0])
        print(df.to_string(index=False))
        print()
        print(f"ΔL (ref) {t0:g}→{t1:g} d: {dL_ref:.6g} µm")
        print(f"ΔL (one) {t0:g}→{t1:g} d: {dL_one:.6g} µm")
        return

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
