#!/usr/bin/env python3
"""
Convergence analysis and cross-check (FEniCS vs pycutfem) for the poroelastic
consolidation benchmark.

This script *orchestrates* runs in both conda envs via `conda run`:
  - `fenics`  : dolfin monolithic reference
  - `fenicsx` : pycutfem implementation

It uses a deterministic triangulation (`structured` by default) and a shared
`--mesh-file` per level to guarantee both solvers run on identical meshes.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np


def _nx_ny_levels(nx0: int, ny0: int, nlevels: int) -> list[tuple[int, int]]:
    nx0 = int(nx0)
    ny0 = int(ny0)
    if nx0 < 2 or ny0 < 2:
        raise ValueError("nx0 and ny0 must be >= 2")
    if nlevels < 1:
        raise ValueError("nlevels must be >= 1")
    out = []
    for lev in range(nlevels):
        factor = 2**lev
        out.append(((nx0 - 1) * factor + 1, (ny0 - 1) * factor + 1))
    return out


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx0", type=int, default=9)
    ap.add_argument("--ny0", type=int, default=5)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--triangulation", choices=("structured", "delaunay"), default="structured")
    ap.add_argument("--final-time", type=float, default=0.3)
    ap.add_argument("--num-time-steps", type=int, default=1)
    ap.add_argument("--t1", type=float, default=3.6)
    ap.add_argument("--pressure-region", type=float, default=5.0)
    ap.add_argument("--out-dir", default="examples/poroelasticity/convergence_out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = out_dir / "mesh_files"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    levels = _nx_ny_levels(args.nx0, args.ny0, args.levels)
    pmax_fen: list[float] = []
    pmax_pyc: list[float] = []

    for lev, (nx, ny) in enumerate(levels):
        tag = f"nx{nx}_ny{ny}"
        mesh_file = mesh_dir / f"mesh_{tag}_{args.triangulation}.npz"
        fen_out = out_dir / f"fenics_{tag}"
        pyc_out = out_dir / f"pycutfem_{tag}"

        _run(
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "fenics",
                "python",
                "examples/poroelasticity/consolidation_fenics_monolithic.py",
                "--output-dir",
                str(fen_out),
                "--mesh-file",
                str(mesh_file),
                "--triangulation",
                args.triangulation,
                "--nx",
                str(nx),
                "--ny",
                str(ny),
                "--final-time",
                str(args.final_time),
                "--num-time-steps",
                str(args.num_time_steps),
                "--t1",
                str(args.t1),
                "--pressure-region",
                str(args.pressure_region),
            ]
        )
        _run(
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "fenicsx",
                "python",
                "examples/poroelasticity/consolidation_pycutfem.py",
                "--output-dir",
                str(pyc_out),
                "--mesh-file",
                str(mesh_file),
                "--triangulation",
                args.triangulation,
                "--nx",
                str(nx),
                "--ny",
                str(ny),
                "--final-time",
                str(args.final_time),
                "--num-time-steps",
                str(args.num_time_steps),
                "--t1",
                str(args.t1),
                "--pressure-region",
                str(args.pressure_region),
                "--backend",
                "jit",
            ]
        )

        import pandas as pd

        df_f = pd.read_csv(fen_out / "results" / "nonIncremental_test_results.csv")
        df_p = pd.read_csv(pyc_out / "results" / "nonIncremental_test_results.csv")
        t_end = float(df_f["time"].iloc[-1])
        p_f = float(df_f["p_w_max"].iloc[-1])
        p_p = float(df_p["p_w_max"].iloc[-1])
        pmax_fen.append(p_f)
        pmax_pyc.append(p_p)

        print(f"[level {lev}] {tag}  t_end={t_end:.6g}  pmax_fen={p_f:.6e}  pmax_py={p_p:.6e}  Δ={p_p - p_f:.3e}")

    # Convergence of pmax to finest (discrete 'reference').
    p_ref_f = pmax_fen[-1]
    p_ref_p = pmax_pyc[-1]
    hs = np.array([1.0 / (nx - 1) for nx, _ in levels], dtype=float)
    err_f = np.abs(np.array(pmax_fen) - p_ref_f)
    err_p = np.abs(np.array(pmax_pyc) - p_ref_p)

    def est_orders(err: np.ndarray) -> list[float]:
        out = [float("nan")]
        for i in range(1, len(err)):
            if err[i] == 0.0 or err[i - 1] == 0.0:
                out.append(float("nan"))
            else:
                out.append(float(np.log(err[i - 1] / err[i]) / np.log(hs[i - 1] / hs[i])))
        return out

    ord_f = est_orders(err_f)
    ord_p = est_orders(err_p)

    print("\nlevel  h        |pmax_fen-p*|   order   |pmax_py-p*|    order")
    for lev, (h, e1, o1, e2, o2) in enumerate(zip(hs, err_f, ord_f, err_p, ord_p)):
        print(f"{lev:5d}  {h:1.3e}  {e1:1.3e}  {o1:6.3f}  {e2:1.3e}  {o2:6.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

