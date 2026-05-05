#!/usr/bin/env python
"""
Pandas convergence/parameter table for Warner (1986) case 4 sloughing.

This script compares the FD ζ-model reference (Warner benchmark) against multiple
one-domain run CSVs at user-selected times.

Run:
  conda run --no-capture-output -n fenicsx \
    python -u examples/biofilms/benchmarks/warner/case4_slough_convergence_table.py \
      --backend cpp \
      --run-tags tagA,tagB,tagC
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
    ap = argparse.ArgumentParser(description="Case 4 sloughing convergence table (Warner FD vs one-domain).")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument(
        "--run-tags",
        type=str,
        required=True,
        help="Comma-separated list of one-domain run tags (suffixes used in output filenames).",
    )
    ap.add_argument(
        "--times",
        type=str,
        default="5.984,5.994,6.0",
        help="Comma-separated list of times (days) to include (default: 5.984,5.994,6.0).",
    )
    ap.add_argument("--out-csv", type=str, default="", help="Optional output CSV path.")
    args = ap.parse_args()

    times = [float(x.strip()) for x in str(args.times).split(",") if x.strip()]
    if not times:
        raise ValueError("--times must be non-empty.")

    run_tags = [x.strip() for x in str(args.run_tags).split(",") if x.strip()]
    if not run_tags:
        raise ValueError("--run-tags must be non-empty.")

    ref_path = REF_OUTDIR / "case4_backend=cpp_timeseries.csv"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing Warner reference CSV: {ref_path}")
    ref = pd.read_csv(ref_path)

    rows: list[dict[str, object]] = []
    for tag in run_tags:
        one_path = ONE_OUTDIR / f"one_domain_strip_case4_backend={str(args.backend)}_{tag}_timeseries.csv"
        if not one_path.exists():
            raise FileNotFoundError(f"Missing one-domain CSV: {one_path}")
        one = pd.read_csv(one_path)

        for t in times:
            L_ref = _interp(ref, t_col="t_days", y_col="L_um", t=t)
            R_ref = _interp(ref, t_col="t_days", y_col="jL_1", t=t) * (-1.0)
            L_one = _interp(one, t_col="t_days", y_col="L_thickness_um", t=t)
            R_one = _interp(one, t_col="t_days", y_col="removal", t=t)
            rows.append(
                {
                    "run_tag": tag,
                    "t_days": float(t),
                    "L_ref_um": float(L_ref),
                    "L_one_um": float(L_one),
                    "L_rel_err": float(abs(L_one - L_ref) / max(abs(L_ref), 1.0e-12)),
                    "removal_ref": float(R_ref),
                    "removal_one": float(R_one),
                    "removal_rel_err": float(abs(R_one - R_ref) / max(abs(R_ref), 1.0e-12)),
                }
            )

    df = pd.DataFrame(rows).sort_values(["run_tag", "t_days"], ignore_index=True)
    print(df.to_string(index=False))

    if len(times) >= 2:
        t0 = float(times[0])
        t1 = float(times[1])
        drops: list[dict[str, object]] = []
        for tag in run_tags:
            dfi = df[df["run_tag"] == tag].set_index("t_days")
            dL_ref = float(dfi.loc[t1, "L_ref_um"] - dfi.loc[t0, "L_ref_um"])
            dL_one = float(dfi.loc[t1, "L_one_um"] - dfi.loc[t0, "L_one_um"])
            drops.append(
                {
                    "run_tag": tag,
                    f"ΔL_ref_{t0:g}→{t1:g}_um": dL_ref,
                    f"ΔL_one_{t0:g}→{t1:g}_um": dL_one,
                    "ΔL_abs_err_um": float(abs(dL_one - dL_ref)),
                }
            )
        print()
        print(pd.DataFrame(drops).to_string(index=False))

    out_csv = str(args.out_csv).strip()
    if out_csv:
        out_path = Path(out_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print()
        print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()

