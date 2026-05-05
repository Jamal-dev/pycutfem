#!/usr/bin/env python3
"""
Export VTK snapshots for the Duddu (2007) Fig.6 one-domain growth reproduction.

Why this exists
--------------
The one-domain Fig.6 driver stores lightweight NumPy snapshots (`snaps_alpha.npz`)
and final fields (`final_fields.npz`) for plotting. Review/inspection in ParaView
is often easier with VTK files, so this script converts those NPZ snapshots into
`.vtu` files without rerunning the simulation.

Outputs
-------
Creates `<run_dir>/vtk/` containing:
  - `step=XXXX.vtu`        : alpha snapshots (subsampled by --every)
  - `alpha_series.pvd`     : ParaView time-series collection for the alpha snapshots
  - `final_fields.vtu`     : final (alpha, S, p) fields if final_fields.npz is present
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.ufl.expressions import Function, VectorFunction
from pycutfem.utils.meshgen import structured_quad


def _write_pvd(*, out_pvd: Path, datasets: list[tuple[float, str]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for t, rel in datasets:
        lines.append(f'    <DataSet timestep="{float(t):.12e}" group="" part="0" file="{rel}"/>')
    lines += ["  </Collection>", "</VTKFile>", ""]
    out_pvd.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Duddu2007 Fig6 one-domain NPZ snapshots to VTK (.vtu).")
    ap.add_argument(
        "--run-dir",
        type=str,
        default=(
            "examples/biofilms/benchmarks/dadu/results/"
            "_tune_oneDom_phiMix_alphaMix_DS120_vSextL2_gvS0p1_acM1_g5e-4_mobdeg_floor0p1_nx80_dt0p2_q2_t28p6"
        ),
        help="One-domain Fig.6 output directory containing snaps_alpha.npz and summary.json.",
    )
    ap.add_argument("--every", type=int, default=10, help="Export every Nth stored snapshot (1 exports all).")
    ap.add_argument("--vtk-dir", type=str, default="", help="Optional VTK output directory (default: <run_dir>/vtk).")
    args = ap.parse_args()

    run_dir = Path(str(args.run_dir)).expanduser().resolve()
    snaps_npz = run_dir / "snaps_alpha.npz"
    summary_json = run_dir / "summary.json"
    if not snaps_npz.exists():
        raise FileNotFoundError(f"Missing {snaps_npz}")
    if not summary_json.exists():
        raise FileNotFoundError(f"Missing {summary_json}")

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    L = float(summary["L_mm"])
    H = float(summary["H_mm"])
    nx = int(summary["nx"])
    ny = int(summary["ny"])

    out_vtk = Path(str(args.vtk_dir)).expanduser() if str(args.vtk_dir).strip() else (run_dir / "vtk")
    out_vtk.mkdir(parents=True, exist_ok=True)

    # Rebuild the same mesh/dofhandler used by the driver (Q2 geometry, Q1 fields).
    nodes, elems, _edges, corners = structured_quad(L, H, nx=nx, ny=ny, poly_order=2, offset=(0.0, 0.0))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )

    field_specs: dict[str, object] = {"alpha": 1, "S": 1, "p": 1, "vS_x": 2, "vS_y": 2}
    me = MixedElement(mesh, field_specs=field_specs)
    dh = DofHandler(me, method="cg")

    alpha = Function("alpha", "alpha", dof_handler=dh)
    S = Function("S", "S", dof_handler=dh)
    p = Function("p", "p", dof_handler=dh)
    vS = VectorFunction("vS", ["vS_x", "vS_y"], dof_handler=dh)

    with np.load(str(snaps_npz)) as z:
        t_days = np.asarray(z["t_days"], dtype=float).ravel()
        a_all = np.asarray(z["alpha"], dtype=float)

    if a_all.ndim != 2:
        raise ValueError(f"Expected alpha snapshots as a 2D array; got shape={a_all.shape}")
    if a_all.shape[0] != t_days.size:
        raise ValueError(f"Snapshot size mismatch: t_days={t_days.size} vs alpha[0]={a_all.shape[0]}")

    every = int(max(1, int(args.every)))
    datasets: list[tuple[float, str]] = []
    for i in range(0, int(t_days.size), every):
        alpha.nodal_values[:] = np.asarray(a_all[i, :], dtype=float)
        out = out_vtk / f"step={i:04d}.vtu"
        export_vtk(str(out), mesh, dh, {"alpha": alpha})
        datasets.append((float(t_days[i]), out.name))

    _write_pvd(out_pvd=out_vtk / "alpha_series.pvd", datasets=datasets)
    print(f"[ok] wrote {len(datasets)} alpha snapshots under {out_vtk}")

    # Optional: final fields.
    final_npz = run_dir / "final_fields.npz"
    if final_npz.exists():
        with np.load(str(final_npz)) as z:
            alpha.nodal_values[:] = np.asarray(z["alpha"], dtype=float).ravel()
            S.nodal_values[:] = np.asarray(z["S"], dtype=float).ravel()
            p.nodal_values[:] = np.asarray(z["p"], dtype=float).ravel()
            if "vS" in z:
                vS.nodal_values[:] = np.asarray(z["vS"], dtype=float).ravel()
        payload = {"alpha": alpha, "S": S, "p": p}
        if np.any(np.asarray(vS.nodal_values, dtype=float)):
            payload["vS"] = vS
        export_vtk(str(out_vtk / "final_fields.vtu"), mesh, dh, payload)
        print(f"[ok] wrote {out_vtk/'final_fields.vtu'}")


if __name__ == "__main__":
    main()
