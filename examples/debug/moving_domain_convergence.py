#!/usr/bin/env python
"""
Convergence driver for the moving-domain CutFEM demo.

Runs the pycutfem example for a sequence of mesh/time refinements, writes
per-level errors to CSV, and (optionally) calls the xfem runner to collect
matching results for comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Any

from examples.moving_domain import run_moving_domain, build_gmsh_mesh


def _rate(prev: float, curr: float) -> float | None:
    if prev is None or curr is None or prev <= 0.0 or curr <= 0.0:
        return None
    return math.log(prev / curr, 2.0)


def run_pycutfem_levels(args) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    mesh_dir = Path(args.mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.levels):
        Lx = args.base_Lx + i
        Lt = args.base_Lt + i if args.lock_Lt_to_Lx else args.base_Lt
        mesh_file = mesh_dir / f"moving_domain_rect_L{Lx}.msh"
        md_args = SimpleNamespace(
            mesh_file=mesh_file,
            rebuild_mesh=True,
            h0=args.h0,
            Lx=Lx,
            mesh_order=args.mesh_order,
            gmsh_element=args.gmsh_element,
            poly_order=args.poly_order,
            t0=args.t0,
            Lt=Lt,
            T_end=args.T_end,
            backend=args.backend,
            save_vtk=False,
            vtk_every=9999,
            output_dir="moving_domain_results",
            view_gmsh=False,
            return_data=True,
        )
        res = run_moving_domain(md_args, return_data=True)
        row = res.as_dict()
        rows.append(row)
    # compute rates
    for i, row in enumerate(rows):
        if i == 0:
            row["rate_L2"] = None
            row["rate_H1"] = None
            continue
        row["rate_L2"] = _rate(rows[i - 1]["err_l2l2"], row["err_l2l2"])
        row["rate_H1"] = _rate(rows[i - 1]["err_l2h1"], row["err_l2h1"])
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]], field_order: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in field_order})
    print(f"Wrote {len(rows)} rows to {path}")


def run_xfem_levels(args) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(args.levels):
        Lx = args.base_Lx + i
        Lt = args.base_Lt + i if args.lock_Lt_to_Lx else args.base_Lt
        mesh_file = Path(args.mesh_dir) / f"moving_domain_rect_L{Lx}.msh"
        cmd = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            args.xfem_env,
            "python",
            str(args.xfem_runner),
            "--mesh-file",
            str(mesh_file),
            "--h0",
            str(args.h0),
            "--Lx",
            str(Lx),
            "--Lt",
            str(Lt),
            "--t0",
            str(args.t0),
            "--T-end",
            str(args.T_end),
            "--poly-order",
            str(args.poly_order),
            "--output-json",
        ]
        print(f"[xfem] running: {' '.join(cmd)}")
        run = subprocess.run(cmd, check=True, text=True, capture_output=True)
        json_line = next((ln for ln in run.stdout.splitlines() if ln.startswith("RESULT_JSON")), None)
        if json_line is None:
            raise RuntimeError("xfem runner did not emit RESULT_JSON line.")
        payload = json.loads(json_line.split("RESULT_JSON", 1)[1].strip())
        rows.append(payload)
    for i, row in enumerate(rows):
        if i == 0:
            row["rate_L2"] = None
            row["rate_H1"] = None
            continue
        row["rate_L2"] = _rate(rows[i - 1]["err_l2l2"], row["err_l2l2"])
        row["rate_H1"] = _rate(rows[i - 1]["err_l2h1"], row["err_l2h1"])
    return rows


def main():
    ap = argparse.ArgumentParser(description="Run moving-domain convergence for pycutfem (and optional xfem).")
    ap.add_argument("--levels", type=int, default=4, help="Number of refinement levels to run.")
    ap.add_argument("--base-Lx", type=int, default=0, help="Starting spatial refinement index.")
    ap.add_argument("--base-Lt", type=int, default=0, help="Starting temporal refinement index.")
    lock_group = ap.add_mutually_exclusive_group()
    lock_group.add_argument("--lock-Lt-to-Lx", action="store_true", dest="lock_Lt_to_Lx", help="Advance Lt together with Lx.")
    lock_group.add_argument("--unlock-Lt", action="store_false", dest="lock_Lt_to_Lx", help="Keep Lt fixed (do not tie it to Lx).")
    ap.set_defaults(lock_Lt_to_Lx=False)
    ap.add_argument("--h0", type=float, default=0.2, help="Base mesh size before refinements.")
    ap.add_argument("--mesh-order", type=int, default=1, choices=(1, 2))
    ap.add_argument("--gmsh-element", type=str, choices=("tri", "quad"), default="quad")
    ap.add_argument("--poly-order", type=int, default=1)
    ap.add_argument("--t0", type=float, default=0.1)
    ap.add_argument("--T-end", type=float, default=0.4, dest="T_end")
    ap.add_argument("--backend", type=str, default="jit", choices=("jit", "python"))
    ap.add_argument("--mesh-dir", type=Path, default=Path("examples/meshes"))
    ap.add_argument("--pycutfem-csv", type=Path, default=Path("examples/debug/moving_domain_convergence_pycutfem.csv"))
    ap.add_argument("--xfem-runner", type=Path, default=Path("examples/debug/moving_domain_xfem_runner.py"))
    ap.add_argument("--xfem-env", type=str, default="xfemcustom")
    ap.add_argument("--run-xfem", action="store_true", help="Also run the xfem runner and write a second CSV.")
    ap.add_argument("--xfem-csv", type=Path, default=Path("examples/debug/moving_domain_convergence_xfem.csv"))
    ap.add_argument("--comparison-csv", type=Path, default=Path("examples/debug/moving_domain_convergence_compare.csv"))
    args = ap.parse_args()

    py_rows = run_pycutfem_levels(args)
    py_fields = ["Lx", "Lt", "h_max", "dt", "err_l2l2", "err_l2h1", "linf_l2", "rate_L2", "rate_H1", "gamma_s", "delta", "mesh_file"]
    _write_csv(Path(args.pycutfem_csv), py_rows, py_fields)

    xf_rows: List[Dict[str, Any]] = []
    if args.run_xfem:
        xf_rows = run_xfem_levels(args)
        _write_csv(Path(args.xfem_csv), xf_rows, py_fields)

    if args.run_xfem and xf_rows:
        merged = []
        xf_by_Lx = {int(r["Lx"]): r for r in xf_rows}
        for r in py_rows:
            lx = int(r["Lx"])
            if lx not in xf_by_Lx:
                continue
            rr = xf_by_Lx[lx]
            merged.append({
                "Lx": lx,
                "Lt_py": r["Lt"],
                "Lt_xf": rr.get("Lt", rr.get("Lt_py", r["Lt"])),
                "h_max_py": r["h_max"],
                "h_max_xf": rr.get("h_max", None),
                "dt_py": r["dt"],
                "dt_xf": rr.get("dt", None),
                "err_l2l2_py": r["err_l2l2"],
                "err_l2l2_xf": rr.get("err_l2l2"),
                "err_l2h1_py": r["err_l2h1"],
                "err_l2h1_xf": rr.get("err_l2h1"),
            })
        comp_fields = ["Lx", "Lt_py", "Lt_xf", "h_max_py", "h_max_xf", "dt_py", "dt_xf", "err_l2l2_py", "err_l2l2_xf", "err_l2h1_py", "err_l2h1_xf"]
        _write_csv(Path(args.comparison_csv), merged, comp_fields)


if __name__ == "__main__":
    main()
