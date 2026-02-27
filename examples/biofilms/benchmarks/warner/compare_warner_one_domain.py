#!/usr/bin/env python
"""
Compare one-domain benchmark outputs against the Warner (1986) FD reference.

This is a lightweight post-processing helper intended for quick iteration while
calibrating cases 1–4 in `warner1986_one_domain.py`.

Run (example):
  conda run --no-capture-output -n fenicsx \
    python -u examples/biofilms/benchmarks/warner/compare_warner_one_domain.py \
      --cases 2,3,4 --backend cpp --run-tag my_run
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


def _interp(t_src: np.ndarray, y_src: np.ndarray, t_query: float) -> float:
    t_src = np.asarray(t_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    tq = float(t_query)
    if t_src.size == 0:
        return float("nan")
    if tq <= float(t_src[0]):
        return float(y_src[0])
    if tq >= float(t_src[-1]):
        return float(y_src[-1])
    return float(np.interp(tq, t_src, y_src))


def _load_ref(case_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = REF_OUTDIR / f"case{int(case_id)}_backend=cpp_timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Warner reference CSV: {path}")
    df = pd.read_csv(path)
    t = df["t_days"].to_numpy(dtype=float)
    L_um = df["L_um"].to_numpy(dtype=float)
    removal = (-df["jL_1"]).to_numpy(dtype=float)  # FD stores uptake as negative
    return t, L_um, removal


def _load_one(case_id: int, *, backend: str, run_tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tag = str(run_tag or "").strip()
    suffix = f"_{tag}" if tag else ""
    path = ONE_OUTDIR / f"one_domain_strip_case{int(case_id)}_backend={str(backend)}{suffix}_timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing one-domain CSV: {path}")
    df = pd.read_csv(path)
    t = df["t_days"].to_numpy(dtype=float)
    L_um = df["L_thickness_um"].to_numpy(dtype=float)
    removal = df["removal"].to_numpy(dtype=float)
    return t, L_um, removal


def _row(case_id: int, t_days: float, ref, one) -> dict[str, float]:
    t_ref, L_ref, R_ref = ref
    t_one, L_one, R_one = one
    t = float(t_days)
    Lr = _interp(t_ref, L_ref, t)
    Lo = _interp(t_one, L_one, t)
    Rr = _interp(t_ref, R_ref, t)
    Ro = _interp(t_one, R_one, t)
    return {
        "case_id": int(case_id),
        "t_days": float(t),
        "L_ref_um": float(Lr),
        "L_one_um": float(Lo),
        "L_rel_err": float(abs(Lo - Lr) / max(abs(Lr), 1.0e-12)),
        "removal_ref": float(Rr),
        "removal_one": float(Ro),
        "removal_rel_err": float(abs(Ro - Rr) / max(abs(Rr), 1.0e-12)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare one-domain vs Warner FD outputs (cases 1–4).")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--run-tag", type=str, default="", help="Run tag suffix used in one-domain output filenames.")
    ap.add_argument(
        "--cases",
        type=str,
        default="1,2,3,4",
        help="Comma-separated subset of {1,2,3,4} to compare (default: all).",
    )
    ap.add_argument(
        "--times",
        type=str,
        default="",
        help="Optional comma-separated time list in days (overrides per-case defaults).",
    )
    ap.add_argument(
        "--allow-extrapolation",
        action="store_true",
        help="Allow comparing at times beyond the last one-domain output time by holding the last value.",
    )
    args = ap.parse_args()

    cases: list[int] = []
    for part in str(args.cases).split(","):
        part = part.strip()
        if not part:
            continue
        cid = int(part)
        if cid not in {1, 2, 3, 4}:
            raise ValueError(f"--cases must be subset of {{1,2,3,4}}; got {cid}.")
        cases.append(cid)
    cases = sorted(set(cases)) or [1, 2, 3, 4]

    custom_times: list[float] | None = None
    if str(args.times).strip():
        custom_times = [float(x.strip()) for x in str(args.times).split(",") if x.strip()]

    times_by_case: dict[int, list[float]] = {
        1: [0.5, 1.0, 4.0, 6.0, 10.0],
        2: [1.0, 4.0, 5.5, 6.0, 7.5, 10.0],
        3: [1.0, 4.0, 6.0, 10.0],
        4: [1.0, 4.0, 5.984, 5.994, 6.0, 6.5, 10.0],
    }

    rows: list[dict[str, float]] = []
    for cid in cases:
        ref = _load_ref(cid)
        one = _load_one(cid, backend=str(args.backend), run_tag=str(args.run_tag))
        times = custom_times if custom_times is not None else times_by_case[int(cid)]
        if (not bool(args.allow_extrapolation)) and times:
            t_last = float(one[0][-1]) if one[0].size else float("nan")
            t_req = float(max(times))
            if np.isfinite(t_last) and t_last + 1.0e-12 < t_req:
                raise RuntimeError(
                    f"One-domain CSV for case {cid} ends at t={t_last:g} d but comparison requested t={t_req:g} d. "
                    "Re-run with a larger --t-final or pass --allow-extrapolation."
                )
        for t in times:
            rows.append(_row(cid, t, ref, one))

    df = pd.DataFrame(rows).sort_values(["case_id", "t_days"], ignore_index=True)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
