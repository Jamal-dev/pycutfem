#!/usr/bin/env python3
"""Run the three-constituent benchmark suite used by the manuscript.

The ``paper`` suite launches all cases discussed in the paper.  The Seboldt
case is expensive, so an existing completed Seboldt directory can be imported
with ``--reuse-seboldt`` when regenerating tables and manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from examples.biofilms.benchmarks.three_constituent.paper1_benchmark1_mms import run_benchmark1_mms
from examples.biofilms.benchmarks.three_constituent.paper1_benchmark3_moving_support_mms import (
    run_benchmark3_moving_support_mms,
)
from examples.biofilms.benchmarks.three_constituent.paper1_physical_benchmarks_2_to_5 import (
    run_physical_benchmarks_2_to_5,
)
from examples.biofilms.benchmarks.three_constituent.paper1_three_constituent_benchmark_suite import (
    run_all_benchmarks,
    write_benchmark_outputs,
)
from examples.biofilms.benchmarks.three_constituent.seboldt_physical import (
    run_physical_seboldt_three_constituent,
)
from examples.biofilms.benchmarks.three_constituent.stoter_physical import (
    run_physical_stoter_three_constituent,
)


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None
    return out or None


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _case_record(case_id: str, passed: bool, summary_path: Path, *, reused: bool = False) -> dict[str, Any]:
    return {
        "case_id": str(case_id),
        "passed": bool(passed),
        "reused": bool(reused),
        "summary_path": str(summary_path),
    }


def _run_or_reuse_seboldt(
    *,
    outdir: Path,
    suite: str,
    reuse_seboldt: Path | None,
    backend: str,
    linear_backend: str,
    write_vtk: bool,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    if reuse_seboldt is not None:
        src = Path(reuse_seboldt).resolve()
        summary_path = src / "benchmark7_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing Seboldt summary: {summary_path}")
        summary = _read_json(summary_path)
        if not isinstance(summary, dict):
            raise TypeError(f"Unexpected Seboldt summary type in {summary_path}")
        for name in (
            "benchmark7_summary.json",
            "benchmark7_summary.csv",
            "benchmark7_history.csv",
            "benchmark7_extrema_history.csv",
            "profile_final.csv",
            "benchmark7_seboldt_profiles.png",
            "top_uy_history.png",
            "top_uy_animation.gif",
        ):
            _copy_if_exists(src / name, outdir / name)
        return _case_record("benchmark7_seboldt_physical", bool(summary.get("passed", False)), outdir / "benchmark7_summary.json", reused=True)

    if suite == "smoke":
        result = run_physical_seboldt_three_constituent(
            outdir=outdir,
            nx=12,
            ny=18,
            eps_alpha=0.05,
            adaptive_interface_target_cells=4.0,
            t_ramp=0.02,
            dt=1.0e-3,
            final_time=2.0e-2,
            max_steps=20,
            inactive_domain_closure=True,
            inactive_alpha_low=5.0e-4,
            inactive_alpha_high=9.995e-1,
            gamma_mobility="interface_delta",
            ell_gamma_factor=0.05,
            transfer_velocity="free",
            lag_alpha_in_constitutive_laws=True,
            pore_pressure_lower_bound=0.0,
            pore_momentum_outflow="conservative",
            backend=backend,
            linear_backend=linear_backend,
            vtk_every=0,
        )
        return _case_record(result.summary.get("case_id", "benchmark7_seboldt_physical"), result.passed, outdir / "benchmark7_summary.json")

    result = run_physical_seboldt_three_constituent(
        outdir=outdir,
        nx=32,
        ny=48,
        eps_alpha=0.05,
        adaptive_interface_target_cells=8.0,
        t_ramp=1.0,
        dt=5.0e-4,
        final_time=3.0,
        inactive_domain_closure=True,
        inactive_alpha_low=5.0e-4,
        inactive_alpha_high=9.995e-1,
        gamma_mobility="interface_delta",
        ell_gamma_factor=0.05,
        transfer_velocity="free",
        lag_alpha_in_constitutive_laws=True,
        pore_pressure_lower_bound=0.0,
        pore_momentum_outflow="conservative",
        backend=backend,
        linear_backend=linear_backend,
        history_stride=5,
        vtk_every=20 if write_vtk else 0,
    )
    return _case_record(result.summary.get("case_id", "benchmark7_seboldt_physical"), result.passed, outdir / "benchmark7_summary.json")


def _write_summary_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "passed", "reused", "summary_path"])
        writer.writeheader()
        for row in cases:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})


def run_paper_benchmarks(
    *,
    outdir: Path,
    suite: str,
    backend: str,
    linear_backend: str,
    reuse_seboldt: Path | None,
    skip_seboldt: bool,
    write_vtk: bool,
) -> dict[str, Any]:
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []

    gate_dir = outdir / "analytic_gates"
    gate_results = run_all_benchmarks()
    gate_summary = write_benchmark_outputs(gate_results, gate_dir)
    cases.append(_case_record("analytic_constituent_gates", all(r.passed for r in gate_results), gate_summary))

    mms_dir = outdir / "benchmark1_mms"
    mms = run_benchmark1_mms(
        outdir=mms_dir,
        nx_list=(4, 8) if suite == "smoke" else (4, 8, 16, 24),
        dt=2.0e-2,
        backend=backend,
        linear_backend=linear_backend,
    )
    cases.append(_case_record(mms.summary.get("case_id", "benchmark1_mms"), mms.passed, mms_dir / "summary.json"))

    moving_dir = outdir / "benchmark3_moving_support_mms"
    moving = run_benchmark3_moving_support_mms(
        outdir=moving_dir,
        nx_list=(8, 16) if suite == "smoke" else (8, 12, 16, 24),
        t1=2.0e-2,
        backend=backend,
        linear_backend=linear_backend,
        make_figures=True,
    )
    cases.append(_case_record(moving.summary.get("case_id", "benchmark3_moving_support_full_mms"), moving.passed, moving_dir / "summary.json"))

    physical_dir = outdir / "benchmarks2_to_5"
    phys = run_physical_benchmarks_2_to_5(outdir=physical_dir, cases=(2, 3, 4, 5), backend=backend, linear_backend=linear_backend)
    for result in phys:
        cases.append(_case_record(result.case_id, result.passed, result.outdir / "summary.json"))

    stoter_dir = outdir / "benchmark6_stoter"
    if suite == "smoke":
        stoter = run_physical_stoter_three_constituent(
            outdir=stoter_dir,
            nx=16,
            ny=20,
            eps=5.0,
            u_max=0.05,
            dt=1.0e-3,
            final_time=2.0e-3,
            backend=backend,
            linear_backend=linear_backend,
        )
    else:
        stoter = run_physical_stoter_three_constituent(
            outdir=stoter_dir,
            nx=32,
            ny=40,
            eps=10.0,
            u_max=0.05,
            dt=1.0e-3,
            final_time=2.0e-2,
            backend=backend,
            linear_backend=linear_backend,
        )
    cases.append(_case_record(stoter.summary.get("case_id", "benchmark6_stoter"), stoter.passed, stoter_dir / "summary.json"))

    if not skip_seboldt:
        cases.append(
            _run_or_reuse_seboldt(
                outdir=outdir / "benchmark7_seboldt",
                suite=suite,
                reuse_seboldt=reuse_seboldt,
                backend=backend,
                linear_backend=linear_backend,
                write_vtk=write_vtk,
            )
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "suite": suite,
        "backend": backend,
        "linear_backend": linear_backend,
        "cases": cases,
        "passed": all(bool(row["passed"]) for row in cases),
    }
    _write_json(outdir / "manifest.json", manifest)
    _write_summary_csv(outdir / "summary.csv", cases)
    _write_json(outdir / "summary.json", cases)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--outdir", type=Path, default=Path("out/three_constituent_paper_benchmarks"))
    parser.add_argument("--backend", default="cpp")
    parser.add_argument("--linear-backend", default="scipy")
    parser.add_argument("--reuse-seboldt", type=Path, default=None)
    parser.add_argument("--skip-seboldt", action="store_true")
    parser.add_argument("--write-vtk", action="store_true")
    args = parser.parse_args(argv)

    manifest = run_paper_benchmarks(
        outdir=args.outdir,
        suite=str(args.suite),
        backend=str(args.backend),
        linear_backend=str(args.linear_backend),
        reuse_seboldt=args.reuse_seboldt,
        skip_seboldt=bool(args.skip_seboldt),
        write_vtk=bool(args.write_vtk),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if bool(manifest["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
