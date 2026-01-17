#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _run_pycutfem(*, cycles: int, poly_order: int, mark_fraction: float) -> list[dict[str, Any]]:
    import runpy

    root = Path(__file__).resolve().parents[2]
    mod = runpy.run_path(str(root / "examples" / "lshape_poisson_amr.py"))

    initial_mesh = mod["initial_lshape_mesh"]
    solve_once = mod["solve_once"]
    refine_marked = mod["refine_marked"]

    mesh = initial_mesh(int(poly_order))
    out: list[dict[str, Any]] = []

    for cycle in range(int(cycles)):
        result = solve_once(mesh, int(poly_order))
        constraints = result.dof_handler.build_hanging_node_constraints()
        ndof_full = int(result.dof_handler.total_dofs)
        ndof_master = int(constraints.n_master) if constraints is not None else ndof_full
        out.append(
            {
                "cycle": cycle,
                "n_elements": int(mesh.n_elements),
                "ndof_full": ndof_full,
                "ndof_master": ndof_master,
                "l2_error": float(result.l2_error),
                "continuity_jump": float(result.continuity_jump),
                "n_slaves": int(constraints.slaves.size) if constraints is not None else 0,
            }
        )

        if cycle == int(cycles) - 1:
            break

        indicators = np.asarray(result.indicators, dtype=float)
        n_mark = max(1, int(float(mark_fraction) * indicators.size))
        marked = set(np.argsort(indicators)[-n_mark:])
        mesh = refine_marked(mesh, marked)

    return out


def _run_ngsolve(*, levels: int, order: int, maxh: float, conda_env: str) -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parents[2]
    script = root / "examples" / "debug" / "lshape_poisson_ngsolve_reference.py"
    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        str(script),
        "--format",
        "json",
        "--levels",
        str(int(levels)),
        "--order",
        str(int(order)),
        "--maxh",
        str(float(maxh)),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare PyCutFEM L-shape Poisson AMR against an NGSolve reference run.")
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--poly-order", type=int, default=2)
    ap.add_argument("--mark-fraction", type=float, default=0.25)
    ap.add_argument("--ngsolve-env", type=str, default="ngsolve-dev")
    ap.add_argument("--ngsolve-maxh", type=float, default=0.5)
    ap.add_argument("--ngsolve-order", type=int, default=None)
    args = ap.parse_args()

    py_rows = _run_pycutfem(cycles=args.cycles, poly_order=args.poly_order, mark_fraction=args.mark_fraction)
    try:
        ng_rows = _run_ngsolve(
            levels=args.cycles,
            order=int(args.ngsolve_order if args.ngsolve_order is not None else args.poly_order),
            maxh=float(args.ngsolve_maxh),
            conda_env=str(args.ngsolve_env),
        )
    except Exception as e:
        print(f"[compare] NGSolve run failed ({e}). Showing PyCutFEM results only.", file=sys.stderr)
        ng_rows = []

    print(f"{'i':>3}  {'py ndof(master)':>14}  {'py L2':>12}  {'ng ndof':>10}  {'ng L2':>12}")
    for i in range(len(py_rows)):
        py = py_rows[i]
        ng = ng_rows[i] if i < len(ng_rows) else None
        ng_ndof = f"{int(ng['ndof']):10d}" if ng is not None else f"{'-':>10}"
        ng_l2 = f"{float(ng['l2_error']):12.5e}" if ng is not None else f"{'-':>12}"
        print(f"{i:3d}  {py['ndof_master']:14d}  {py['l2_error']:12.5e}  {ng_ndof}  {ng_l2}")


if __name__ == "__main__":
    main()

