"""
Compare a CutFEM cylinder run against the bundled FeatFlow reference data.

Usage
-----
Run the benchmark first so it writes `functionals.csv`, then:

`python examples/turek_cylinder/compare_featflow.py --level 6 --sim examples/turek_cylinder/turek_results/functionals.csv`
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pycutfem.benchmarks.featflow import compare_timeseries


def _read_sim_csv(path: Path) -> dict[str, np.ndarray]:
    with Path(path).open("r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise ValueError(f"No rows found in {path}.")

    def col(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype=float)

    return {
        "time": col("time"),
        "Cd": col("Cd_surf"),
        "Cl": col("Cl_surf"),
        "dp": col("dp"),
    }


def _read_featflow_forces(path: Path) -> dict[str, np.ndarray]:
    time: list[float] = []
    cd: list[float] = []
    cl: list[float] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            time.append(float(parts[1]))
            cd.append(float(parts[3]))
            cl.append(float(parts[4]))
    return {"time": np.asarray(time, float), "Cd": np.asarray(cd, float), "Cl": np.asarray(cl, float)}


def _read_featflow_pressure(path: Path) -> dict[str, np.ndarray]:
    time: list[float] = []
    pA: list[float] = []
    pB: list[float] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 12:
                continue
            time.append(float(parts[1]))
            pA.append(float(parts[6]))
            pB.append(float(parts[11]))
    t = np.asarray(time, float)
    pa = np.asarray(pA, float)
    pb = np.asarray(pB, float)
    return {"time": t, "dp": pa - pb}


def main() -> int:
    p = argparse.ArgumentParser(description="Compare pycutfem cylinder run to FeatFlow reference.")
    p.add_argument("--level", type=int, default=6, choices=range(1, 7), help="FeatFlow level (1..6).")
    p.add_argument(
        "--sim",
        type=Path,
        default=None,
        help="Path to simulation functionals.csv.",
    )
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    if args.sim is None:
        candidate = here / f"turek_results_lv{int(args.level)}" / "functionals.csv"
        args.sim = candidate if candidate.is_file() else (here / "turek_results" / "functionals.csv")
    ref_dir = here / "data" / "featflow"
    bdforces = ref_dir / f"bdforces_lv{int(args.level)}"
    pointvals = ref_dir / f"pointvalues_lv{int(args.level)}"

    if not bdforces.is_file():
        raise SystemExit(f"Missing reference file: {bdforces}")
    if not pointvals.is_file():
        raise SystemExit(f"Missing reference file: {pointvals}")

    sim = _read_sim_csv(Path(args.sim))
    ref_f = _read_featflow_forces(bdforces)
    ref_p = _read_featflow_pressure(pointvals)

    # Compare on the overlapping window via interpolation of sim onto ref time.
    for name in ("Cd", "Cl"):
        stats = compare_timeseries(
            ref_time=ref_f["time"],
            ref_val=ref_f[name],
            sim_time=sim["time"],
            sim_val=sim[name],
        )
        print(
            f"{name}: rms={stats['rms']:.6e} max_abs={stats['max_abs']:.6e}  "
            f"(t∈[{stats['t0']:.3g},{stats['t1']:.3g}], n={stats['n']})"
        )

    stats = compare_timeseries(
        ref_time=ref_p["time"],
        ref_val=ref_p["dp"],
        sim_time=sim["time"],
        sim_val=sim["dp"],
    )
    print(f"dp: rms={stats['rms']:.6e} max_abs={stats['max_abs']:.6e}  (t∈[{stats['t0']:.3g},{stats['t1']:.3g}], n={stats['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
