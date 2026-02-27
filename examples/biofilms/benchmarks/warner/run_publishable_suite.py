#!/usr/bin/env python
"""
Reproducible runner for the Warner (1986) benchmark vs the one-domain biofilm model.

This script lives next to:
  - `warner1986_benchmark.py`   (FD / method-of-lines ζ-model reference)
  - `warner1986_one_domain.py` (reduced one-domain PDE comparison)

It provides:
  - a single entry point to generate reference CSVs (cases 1–4),
  - a “publishable” one-domain parameter preset (from `investigation.md`),
  - a mesh/interface-width convergence sweep for case 1,
  - a compact CSV/Markdown summary of thickness + removal errors at selected times.

Run inside the fenicsx env (recommended):
  conda run --no-capture-output -n fenicsx python -u examples/biofilms/benchmarks/warner/run_publishable_suite.py
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

REF_OUTDIR = REPO / "examples" / "biofilms" / "results" / "warner1986"
ONE_OUTDIR = REPO / "examples" / "biofilms" / "results" / "warner1986_one_domain"


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _load_ref(case_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = REF_OUTDIR / f"case{int(case_id)}_backend=cpp_timeseries.csv"
    data = np.genfromtxt(str(path), delimiter=",", names=True, dtype=float, encoding=None)
    if getattr(data, "shape", ()) == ():
        data = np.array([data], dtype=data.dtype)
    t = np.asarray(data["t_days"], dtype=float)
    L_um = np.asarray(data["L_um"], dtype=float)
    removal = -np.asarray(data["jL_1"], dtype=float)  # Warner stores uptake as negative
    return t, L_um, removal


def _load_one(case_id: int, *, backend: str, run_tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    suffix = f"_{str(run_tag).strip()}" if str(run_tag).strip() else ""
    path = ONE_OUTDIR / f"one_domain_strip_case{int(case_id)}_backend={backend}{suffix}_timeseries.csv"
    data = np.genfromtxt(str(path), delimiter=",", names=True, dtype=float, encoding=None)
    if getattr(data, "shape", ()) == ():
        data = np.array([data], dtype=data.dtype)
    t = np.asarray(data["t_days"], dtype=float)
    L_um = np.asarray(data["L_thickness_um"], dtype=float)
    removal = np.asarray(data["removal"], dtype=float)
    return t, L_um, removal


def _interp(t_src: np.ndarray, y_src: np.ndarray, t_query: float) -> float:
    t_src = np.asarray(t_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    tq = float(t_query)
    if tq <= float(t_src[0]):
        return float(y_src[0])
    if tq >= float(t_src[-1]):
        return float(y_src[-1])
    return float(np.interp(tq, t_src, y_src))


def _error_row(*, case_id: int, t: float, ref, one) -> dict[str, float]:
    t_ref, L_ref, R_ref = ref
    t_one, L_one, R_one = one
    Lr = _interp(t_ref, L_ref, t)
    Lo = _interp(t_one, L_one, t)
    Rr = _interp(t_ref, R_ref, t)
    Ro = _interp(t_one, R_one, t)
    L_rel = float(abs(Lo - Lr) / max(abs(Lr), 1.0e-12))
    R_rel = float(abs(Ro - Rr) / max(abs(Rr), 1.0e-12))
    return {
        "case_id": int(case_id),
        "t_days": float(t),
        "L_ref_um": float(Lr),
        "L_one_um": float(Lo),
        "L_rel_err": float(L_rel),
        "removal_ref": float(Rr),
        "removal_one": float(Ro),
        "removal_rel_err": float(R_rel),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Warner (1986) publishable comparison suite.")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--run-tag", type=str, default="base", help="Suffix tag used for one-domain output filenames.")
    ap.add_argument(
        "--cases",
        type=str,
        default="1,2,3,4",
        help="Comma-separated subset of {1,2,3,4} to run/compare (default: all).",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-ref", action="store_true", help="Skip generating Warner FD reference CSVs.")
    ap.add_argument("--skip-one", action="store_true", help="Skip running the one-domain comparisons.")
    ap.add_argument("--skip-convergence", action="store_true", help="Skip the case 1 mesh/eps sweep.")
    ap.add_argument("--summary-only", action="store_true", help="Only write summary from existing CSVs.")
    args = ap.parse_args()

    backend = str(args.backend)
    dry = bool(args.dry_run)
    run_tag = str(args.run_tag).strip() or "base"

    cases: list[int] = []
    for part in str(args.cases).split(","):
        part = part.strip()
        if not part:
            continue
        cid = int(part)
        if cid not in {1, 2, 3, 4}:
            raise ValueError(f"--cases must be a subset of {{1,2,3,4}}; got {cid}.")
        cases.append(cid)
    cases = sorted(set(cases))
    if not cases:
        cases = [1, 2, 3, 4]

    if dry and not bool(args.summary_only):
        # Dry-run is intended to be copy/paste friendly; don't attempt to read/write
        # summary outputs when we haven't executed anything.
        print("[dry-run] commands printed; skipping execution and summary.", flush=True)
        # Still show the commands users would run.

    if not args.summary_only and not args.skip_ref:
        _run(
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "fenicsx",
                "python",
                "-u",
                str(HERE / "warner1986_benchmark.py"),
                "--case",
                "all",
                "--backend",
                "cpp",
                "--no-plots",
                "--outdir",
                str(REF_OUTDIR),
            ],
            dry_run=dry,
        )

    # “Publishable” preset (strip, full mechanics + strong drag).
    #
    # Notes (important for stability / reproducibility):
    # - Use split solves for robustness with stiff substrate kinetics.
    # - Keep dt fixed unless you are actively tuning an adaptive strategy (dt growth can skip short events).
    base_one = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "fenicsx",
        "python",
        "-u",
        str(HERE / "warner1986_one_domain.py"),
        "--backend",
        backend,
        "--mode",
        "strip",
        "--mechanics",
        "full",
        "--solve-strategy",
        "split",
        "--freeze-phi",
        "--freeze-u",
        "--freeze-vSx",
        "--s-v-mode",
        "mu",
        "--kappa-inv",
        "1e10",
        "--bulk-mode",
        "well_mixed",
        "--bulk-gamma",
        "1000",
        "--bulk-alpha-power",
        "16",
        "--surface-gamma",
        "0",
        "--removal-metric",
        "warner_stencil",
        "--thickness-metric",
        "half",
        "--L-ref-m",
        "1e-3",
        "--Lx",
        "0.2",
        "--nx",
        "1",
        "--rho-s-effective",
        "5000",
        "--phi-b",
        "0.3",
        "--D-alpha",
        "0.0",
        "--alpha-advection-form",
        "advective",
        "--k-g",
        "0.0",
        "--mu-max",
        "4.8",
        "--k-d",
        "0.2",
        "--K-S",
        "5.0",
        "--Y",
        "0.4",
        "--dt",
        "0.05",
        "--dt-max",
        "0.05",
        "--dt-increase-factor",
        "1.0",
        "--dt-after-event",
        "0.01",
        "--q",
        "2",
        "--vi-c",
        "1.0",
        "--no-line-search",
        "--outdir",
        str(ONE_OUTDIR),
        "--no-plots",
    ]

    if not args.summary_only and not args.skip_one:
        # Case 1 (long run; used for base comparisons).
        if 1 in cases:
            _run(
                base_one
                + [
                    "--case",
                    "1",
                    "--run-tag",
                    run_tag,
                    "--Hy",
                    "1.5",
                    "--h0",
                    "0.005",
                    "--eps0",
                    "0.003",
                    "--ny",
                    "480",
                    "--t-final",
                    "10.0",
                ],
                dry_run=dry,
            )

        # Cases 2–4 (long runs; discontinuities handled internally by the driver).
        # Case 2: enable the reduced "autotroph" volumetric expansion surrogate after t>=6d.
        if 2 in cases:
            _run(
                base_one
                + [
                    "--case",
                    "2",
                    "--run-tag",
                    run_tag,
                    "--case2-autotroph",
                    "--case2-autotroph-start",
                    "6.0",
                    "--case2-autotroph-mu0",
                    "0.173",
                    "--case2-autotroph-mu1",
                    "0.249",
                    "--case2-autotroph-tau",
                    "3.0",
                    "--Hy",
                    "0.9",
                    "--h0",
                    "0.005",
                    "--eps0",
                    "0.003",
                    "--ny",
                    "300",
                    "--t-final",
                    "10.0",
                ],
                dry_run=dry,
            )
        if 3 in cases:
            _run(
                base_one
                + [
                    "--case",
                    "3",
                    "--run-tag",
                    run_tag,
                    "--lambda-shear",
                    "750.0",
                    "--shear-mode",
                    "truncate",
                    "--Hy",
                    "0.7",
                    "--h0",
                    "0.005",
                    "--eps0",
                    "0.003",
                    "--ny",
                    "240",
                    "--t-final",
                    "10.0",
                ],
                dry_run=dry,
            )
        if 4 in cases:
            _run(
                base_one
                + [
                    "--case",
                    "4",
                    "--run-tag",
                    run_tag,
                    "--slough-mode",
                    "shift_window",
                    "--Hy",
                    "1.1",
                    "--h0",
                    "0.005",
                    "--eps0",
                    "0.003",
                    "--ny",
                    "400",
                    "--t-final",
                    "10.0",
                ],
                dry_run=dry,
            )

    if not args.summary_only and not args.skip_convergence:
        # Mesh/interface-width sweep for case 1 (eps0 shrinks with h to approach a sharper interface).
        sweep = [
            (120, 0.006),
            (240, 0.003),
            (480, 0.0015),
        ]
        for ny, eps0 in sweep:
            tag = f"conv_ny{int(ny)}_eps{eps0:g}"
            _run(
                base_one
                + [
                    "--case",
                    "1",
                    "--run-tag",
                    tag,
                    "--t-final",
                    "0.5",
                    "--Hy",
                    "0.2",
                    "--h0",
                    "0.005",
                    "--eps0",
                    str(eps0),
                    "--ny",
                    str(ny),
                ],
                dry_run=dry,
            )

    if dry and not bool(args.summary_only):
        return

    # Summary (requires CSVs to exist).
    rows: list[dict[str, float]] = []
    times_by_case = {
        1: [0.5, 6.0, 10.0],
        2: [5.5, 6.0, 7.5, 10.0],
        3: [4.0, 6.0, 10.0],
        4: [5.984, 5.994, 6.0, 6.5, 10.0],
    }
    for cid in cases:
        times = times_by_case[int(cid)]
        ref = _load_ref(cid)
        one = _load_one(cid, backend=backend, run_tag=run_tag)
        for t in times:
            rows.append(_error_row(case_id=cid, t=t, ref=ref, one=one))

    df = pd.DataFrame(rows).sort_values(["case_id", "t_days"], ignore_index=True)
    out_csv = ONE_OUTDIR / f"publishable_suite_summary_backend={backend}_{run_tag}.csv"
    df.to_csv(out_csv, index=False)
    print("\n=== Warner (1986) vs one-domain (base) ===")
    print(df.to_string(index=False))
    print(f"[summary] wrote: {out_csv}")

    # Convergence table (case 1, t=0.5d).
    if not bool(args.skip_convergence):
        ref1 = _load_ref(1)
        conv_rows: list[dict[str, float]] = []
        for ny, eps0 in [(120, 0.006), (240, 0.003), (480, 0.0015)]:
            tag = f"conv_ny{int(ny)}_eps{eps0:g}"
            one = _load_one(1, backend=backend, run_tag=tag)
            row = _error_row(case_id=1, t=0.5, ref=ref1, one=one)
            row.update({"ny": int(ny), "eps0": float(eps0), "run_tag": tag})
            conv_rows.append(row)
        df_conv = pd.DataFrame(conv_rows).sort_values(["ny"], ignore_index=True)
        out_conv = ONE_OUTDIR / f"convergence_case1_t=0.5_backend={backend}.csv"
        df_conv.to_csv(out_conv, index=False)
        print("\n=== Convergence (case 1 @ t=0.5d) ===")
        print(df_conv.to_string(index=False))
        print(f"[convergence] wrote: {out_conv}")


if __name__ == "__main__":
    main()
