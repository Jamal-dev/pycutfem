#!/usr/bin/env python3
"""
Postprocess XFEM Duddu (2007) Example 1 VTK snapshots into colored PNG panels.

Some long XFEM runs may be interrupted after writing `vtk/step=XXXX.vtu` files but
before producing the final Matplotlib figures. This script rebuilds:

  - fig5a_interface.png : interface motion (phi_ls=0) colored by time
  - fig5b_S.png         : substrate S at final snapshot (color) + interface overlay
  - fig5c_Phi.png       : velocity potential Phi at final snapshot (color) + interface overlay
  - vtk/series.pvd      : ParaView time-series collection (optional)

Run
---
conda run --no-capture-output -n fenicsx python \
  examples/biofilms/benchmarks/dadu/plot_xfem_fig5_from_vtk.py \
  --vtk-dir examples/biofilms/benchmarks/dadu/results/<RUN>/vtk
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


def _step_index(path: Path) -> int:
    m = re.search(r"step=(\d+)\.vtu$", path.name)
    if not m:
        raise ValueError(f"Unexpected VTK snapshot name: {path.name}")
    return int(m.group(1))


def _read_times(y_top_csv: Path) -> dict[int, float]:
    if not y_top_csv.exists():
        return {}
    out: dict[int, float] = {}
    with y_top_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row:
                continue
            step = int(float(row.get("step", "0")))
            t = float(row.get("t_days", "0.0"))
            out[step] = t
    return out


def _triangles_from_meshio(mesh) -> np.ndarray:
    tri = None
    quad = None
    for cb in getattr(mesh, "cells", []):
        if cb.type in {"triangle", "tri"}:
            tri = np.asarray(cb.data, dtype=int)
            break
        if cb.type in {"quad", "quadrilateral"}:
            quad = np.asarray(cb.data, dtype=int)
    if tri is not None:
        return tri
    if quad is None:
        raise RuntimeError("No triangle/quad cells found in VTK file.")
    return np.vstack([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]]).astype(int, copy=False)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtk-dir", type=str, required=True, help="Directory containing step=XXXX.vtu snapshots.")
    ap.add_argument("--y-top-csv", type=str, default="", help="Optional CSV with (t_days,step,...) for timesteps.")
    ap.add_argument("--outdir", type=str, default="", help="Output directory for PNGs (default: <vtk-dir>/..).")
    ap.add_argument("--cmap-interface", type=str, default="viridis")
    ap.add_argument("--cmap-S", type=str, default="viridis")
    ap.add_argument("--cmap-Phi", type=str, default="RdBu_r")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--write-pvd", action="store_true", help="Write <vtk-dir>/series.pvd using y-top times if available.")
    args = ap.parse_args()

    vtk_dir = Path(str(args.vtk_dir)).expanduser().resolve()
    if not vtk_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {vtk_dir}")
    vtu_files = sorted(vtk_dir.glob("step=*.vtu"), key=_step_index)
    if not vtu_files:
        raise FileNotFoundError(f"No step=*.vtu files found under {vtk_dir}")

    y_top_csv = Path(str(args.y_top_csv)).expanduser().resolve() if str(args.y_top_csv).strip() else (vtk_dir.parent / "y_top_timeseries.csv")
    times_by_step = _read_times(y_top_csv)

    # Lazy imports so the script can be inspected outside the FEniCS env.
    import meshio  # type: ignore

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import Normalize

    # --- load snapshots ----------------------------------------------------
    snaps: list[dict[str, object]] = []
    datasets: list[tuple[float, str]] = []
    for f in vtu_files:
        step = _step_index(f)
        t = float(times_by_step.get(step, float(step)))
        datasets.append((t, f.name))

        msh = meshio.read(str(f))
        pts = np.asarray(msh.points, dtype=float)[:, :2]
        tris = _triangles_from_meshio(msh)
        pd = getattr(msh, "point_data", {}) or {}
        if "phi_ls" not in pd:
            raise KeyError(f"Missing point_data['phi_ls'] in {f}")
        phi = np.asarray(pd["phi_ls"], dtype=float).ravel()
        S = np.asarray(pd.get("S", np.full_like(phi, np.nan)), dtype=float).ravel()
        Phi = np.asarray(pd.get("Phi", np.full_like(phi, np.nan)), dtype=float).ravel()

        snaps.append({"t": t, "step": step, "pts": pts, "tri": tris, "phi": phi, "S": S, "Phi": Phi})

    snaps_sorted = sorted(snaps, key=lambda d: float(d["t"]))
    t0 = float(snaps_sorted[0]["t"])
    t1 = float(snaps_sorted[-1]["t"])
    norm = Normalize(vmin=t0, vmax=t1 if t1 > t0 else (t0 + 1.0))
    cmap_iface = plt.get_cmap(str(args.cmap_interface))

    # --- interface motion --------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.0), constrained_layout=True)
    for s in snaps_sorted:
        pts = np.asarray(s["pts"], float)
        tri = np.asarray(s["tri"], int)
        phi = np.asarray(s["phi"], float)
        triang = mtri.Triangulation(pts[:, 0], pts[:, 1], triangles=tri)
        color = cmap_iface(norm(float(s["t"])))
        try:
            ax.tricontour(triang, phi, levels=[0.0], colors=[color], linewidths=1.2)
        except Exception:
            # If the contour is not present due to numerical issues, skip.
            continue
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_iface)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("t (days)")
    ax.set_title("Example 1: interface motion (XFEM, colored by time)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    outdir = Path(str(args.outdir)).expanduser().resolve() if str(args.outdir).strip() else vtk_dir.parent
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "fig5a_interface.png", dpi=int(args.dpi))
    plt.close(fig)

    # --- final scalar panels ----------------------------------------------
    last = snaps_sorted[-1]
    pts = np.asarray(last["pts"], float)
    tri = np.asarray(last["tri"], int)
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], triangles=tri)
    phi_last = np.asarray(last["phi"], float)

    def _panel(*, values: np.ndarray, title: str, cmap: str, outname: str) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.0), constrained_layout=True)
        tpc = ax.tripcolor(triang, np.asarray(values, float), shading="gouraud", cmap=str(cmap))
        fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
        ax.tricontour(triang, phi_last, levels=[0.0], colors="k", linewidths=1.2)
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        fig.savefig(outdir / outname, dpi=int(args.dpi))
        plt.close(fig)

    _panel(
        values=np.asarray(last["S"], float),
        title=f"Example 1: substrate S at t={float(last['t']):.1f} days (XFEM)",
        cmap=str(args.cmap_S),
        outname="fig5b_S.png",
    )
    _panel(
        values=np.asarray(last["Phi"], float),
        title=f"Example 1: velocity potential $\\Phi$ at t={float(last['t']):.1f} days (XFEM)",
        cmap=str(args.cmap_Phi),
        outname="fig5c_Phi.png",
    )

    if bool(args.write_pvd) and datasets:
        _write_pvd(out_pvd=vtk_dir / "series.pvd", datasets=sorted(datasets, key=lambda x: float(x[0])))

    print(f"- Wrote {outdir/'fig5a_interface.png'}")
    print(f"- Wrote {outdir/'fig5b_S.png'}")
    print(f"- Wrote {outdir/'fig5c_Phi.png'}")
    if bool(args.write_pvd):
        print(f"- Wrote {vtk_dir/'series.pvd'}")


if __name__ == "__main__":
    main()
