#!/usr/bin/env python3
"""
Extract y_top(t) from XFEM VTK snapshots (phi_ls iso-contour) for Duddu (2007) runs.

Why this exists
--------------
Some XFEM benchmark drivers write VTK snapshots (`vtk/step=XXXX.vtu`) at paper target
times and only write `y_top_timeseries.csv` at the very end. This helper computes
`y_top_mm = max_y( phi_ls = 0 )` directly from the VTK files so partial runs can be
compared and visualized without rerunning.

Assumptions
-----------
- Each `.vtu` contains point data named `phi_ls` (signed level set).
- Filenames follow `step=XXXX.vtu` where XXXX is a 0-based index into the target
  time list used by the driver.

Run
---
conda run --no-capture-output -n fenicsx python \
  examples/biofilms/benchmarks/dadu/extract_y_top_from_xfem_vtk.py \
  --vtk-dir examples/biofilms/benchmarks/dadu/results/<RUN>/vtk \
  --targets "0,1,2.3,..." \
  --out examples/biofilms/benchmarks/dadu/results/<RUN>/y_top_timeseries.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


def _parse_targets_csv(s: str) -> list[float]:
    if not str(s).strip():
        return []
    out: list[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    # sorted unique
    uniq: list[float] = []
    for v in sorted(out):
        if not uniq or abs(v - uniq[-1]) > 1.0e-12:
            uniq.append(float(v))
    return uniq


def _step_index(path: Path) -> int:
    m = re.search(r"step=(\d+)\.vtu$", path.name)
    if not m:
        raise ValueError(f"Unexpected VTK filename (expected step=XXXX.vtu): {path.name}")
    return int(m.group(1))


def _unique_edges_from_triangles(tris: np.ndarray) -> np.ndarray:
    """
    Return unique undirected edges from a (ntri,3) triangle connectivity array.
    """
    tris = np.asarray(tris, dtype=np.int64)
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError(f"Expected triangles (n,3); got shape={tris.shape}")
    e01 = tris[:, [0, 1]]
    e12 = tris[:, [1, 2]]
    e20 = tris[:, [2, 0]]
    edges = np.vstack([e01, e12, e20])
    edges.sort(axis=1)
    # unique rows
    edges = np.unique(edges, axis=0)
    return edges


def _y_top_from_phi_edges(*, points_xy: np.ndarray, phi: np.ndarray, edges: np.ndarray) -> tuple[float, int]:
    """
    Compute y_top as the max y-coordinate on the phi=0 isocontour, using
    linear interpolation on edges where phi changes sign.
    Returns (y_top, n_intersections).
    """
    pts = np.asarray(points_xy, dtype=float)
    phi = np.asarray(phi, dtype=float).ravel()
    edges = np.asarray(edges, dtype=np.int64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected points (n,2); got shape={pts.shape}")
    if phi.size != pts.shape[0]:
        raise ValueError("phi size mismatch with points")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Expected edges (m,2); got shape={edges.shape}")

    y_top = 0.0
    n_int = 0
    for i0, i1 in edges.tolist():
        a = float(phi[i0])
        b = float(phi[i1])
        if (a == 0.0) and (b == 0.0):
            # Degenerate: entire edge on interface. Use both endpoints.
            y_top = max(y_top, float(pts[i0, 1]), float(pts[i1, 1]))
            n_int += 2
            continue
        if a == 0.0:
            y_top = max(y_top, float(pts[i0, 1]))
            n_int += 1
            continue
        if b == 0.0:
            y_top = max(y_top, float(pts[i1, 1]))
            n_int += 1
            continue
        if (a < 0.0 and b > 0.0) or (a > 0.0 and b < 0.0):
            t01 = a / (a - b)
            y = float(pts[i0, 1] + t01 * (pts[i1, 1] - pts[i0, 1]))
            y_top = max(y_top, y)
            n_int += 1
    return float(y_top), int(n_int)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract y_top(t) from XFEM VTK snapshots (phi_ls=0 contour).")
    ap.add_argument("--vtk-dir", type=str, required=True, help="Directory containing step=XXXX.vtu snapshots.")
    ap.add_argument("--targets", type=str, default="", help="Comma-separated target times (days) matching the step indices.")
    ap.add_argument("--out", type=str, default="", help="Output CSV path (default: <vtk-dir>/../y_top_timeseries.csv).")
    args = ap.parse_args()

    vtk_dir = Path(str(args.vtk_dir)).expanduser().resolve()
    if vtk_dir.is_dir():
        vtu_files = sorted(vtk_dir.glob("step=*.vtu"), key=_step_index)
    else:
        raise FileNotFoundError(f"Not a directory: {vtk_dir}")
    if not vtu_files:
        raise FileNotFoundError(f"No step=*.vtu files found under {vtk_dir}")

    targets = _parse_targets_csv(str(args.targets))
    max_step = max(_step_index(p) for p in vtu_files)
    if targets and max_step >= len(targets):
        raise ValueError(f"Need targets for step indices up to {max_step} (got {len(targets)} target(s)).")

    out_csv = Path(str(args.out)).expanduser() if str(args.out).strip() else (vtk_dir.parent / "y_top_timeseries.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Import meshio lazily so this script can be inspected without the conda env.
    import meshio  # type: ignore

    rows: list[dict[str, object]] = []
    for vtu in vtu_files:
        step = _step_index(vtu)
        mesh = meshio.read(str(vtu))
        if "phi_ls" not in mesh.point_data:
            raise KeyError(f"Missing point_data['phi_ls'] in {vtu.name}; found keys={list(mesh.point_data.keys())}")
        phi = np.asarray(mesh.point_data["phi_ls"], dtype=float).ravel()
        pts = np.asarray(mesh.points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise ValueError(f"Unexpected points array shape in {vtu.name}: {pts.shape}")
        pts_xy = pts[:, :2]

        # Collect triangle connectivity (ignore any other cell types).
        tris_list: list[np.ndarray] = []
        for cell_block in mesh.cells:
            if cell_block.type == "triangle":
                tris_list.append(np.asarray(cell_block.data, dtype=np.int64))
        if not tris_list:
            raise ValueError(f"No triangle cells found in {vtu.name}; cell types={[c.type for c in mesh.cells]}")
        tris = np.vstack(tris_list)
        edges = _unique_edges_from_triangles(tris)
        y_top, n_int = _y_top_from_phi_edges(points_xy=pts_xy, phi=phi, edges=edges)

        if targets:
            t_days = float(targets[step])
        else:
            t_days = float(step)

        rows.append({"t_days": t_days, "step": int(step), "y_top_mm": float(y_top), "n_intersections": int(n_int)})

    _write_csv(out_csv, rows)
    print(f"[ok] wrote {out_csv} ({len(rows)} row(s))")


if __name__ == "__main__":
    main()
