#!/usr/bin/env python3
"""
Postprocess a Lie benchmark simulation: compute dx(t) at SVG mark points.

This is meant to pair with the experimental DIC extractor:
  examples/biofilms/benchmarks/lie/extract_deformation_timeseries_dic_video_s1.py --mode marks

It reads VTU files in <sim-dir>/vtk/step=XXXX.vtu, interpolates the Eulerian
displacement field `u(x,t)` to 3 reference points defined by SVG marks in
frame_0000.svg, and computes a DIC-like displacement:

    x(t) = x_ref + u(x(t), t)   (fixed-point iteration)
    dx(t) = x(t) - x_ref

Output CSV is compatible with:
  examples/biofilms/benchmarks/lie/compare_exp_sim_timeseries_dx.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import meshio
import numpy as np
from scipy.spatial import cKDTree

from examples.biofilms.benchmarks.lie.svg_trace_utils import extract_svg_frame_geometry, extract_svg_mark_points_px


def _vtk_step_files(vtk_dir: Path) -> list[tuple[int, Path]]:
    step_re = re.compile(r"step=(\d+)\.vtu$")
    out: list[tuple[int, Path]] = []
    for p in sorted(vtk_dir.glob("step=*.vtu")):
        m = step_re.search(p.name)
        if not m:
            continue
        out.append((int(m.group(1)), p))
    if not out:
        raise FileNotFoundError(f"No VTK files found in {vtk_dir} (expected step=XXXX.vtu).")
    return out


def _read_timeseries_t(ts_path: Path) -> np.ndarray:
    arr = np.genfromtxt(str(ts_path), delimiter=",", skip_header=1, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 1:
        raise ValueError(f"Expected at least 1 column in {ts_path}; got shape={arr.shape}")
    return np.asarray(arr[:, 0], dtype=float)


def _select_three_mark_points_px(marks_px: np.ndarray) -> np.ndarray:
    pts = np.asarray(marks_px, dtype=float).reshape((-1, 2))
    if pts.shape[0] < 3:
        raise RuntimeError(f"Need ≥3 mark points; got {pts.shape[0]}")

    y = pts[:, 1]
    i_top = int(np.argmin(y))
    i_bot = int(np.argmax(y))
    y_med = float(np.median(y))
    i_mid = int(np.argmin(np.abs(y - y_med)))
    idx: list[int] = []
    for i in (i_top, i_mid, i_bot):
        if int(i) not in idx:
            idx.append(int(i))
    if len(idx) < 3:
        rest = [int(i) for i in np.argsort(y).tolist() if int(i) not in idx]
        idx.extend(rest[: (3 - len(idx))])
    idx = idx[:3]
    # Order top->bottom in image coords.
    idx = sorted(idx, key=lambda j: float(pts[int(j), 1]))
    return np.asarray(pts[idx, :], dtype=float)


def _infer_block_x0(points_xy: np.ndarray, *, block_w: float) -> float:
    xy = np.asarray(points_xy, dtype=float)
    x_min = float(np.min(xy[:, 0]))
    x_max = float(np.max(xy[:, 0]))
    xc = 0.5 * (x_min + x_max)
    return float(xc - 0.5 * float(block_w))


def _interp_displacement_idw(
    *,
    tree: cKDTree,
    u_nodes: np.ndarray,
    xq: np.ndarray,
    k: int = 8,
    power: float = 2.0,
) -> np.ndarray:
    xq = np.asarray(xq, dtype=float)
    if xq.ndim == 1:
        xq = xq.reshape(1, 2)
    if xq.ndim != 2 or xq.shape[1] != 2:
        raise ValueError("xq must have shape (N,2)")
    k = int(max(1, int(k)))

    dist, idx = tree.query(xq, k=k)
    if k == 1:
        dist = dist.reshape((-1, 1))
        idx = idx.reshape((-1, 1))

    hit = dist[:, 0] <= 1.0e-14
    u_out = np.empty((xq.shape[0], 2), dtype=float)
    if np.any(hit):
        u_out[hit, :] = u_nodes[idx[hit, 0], :]

    miss = ~hit
    if np.any(miss):
        d = np.maximum(dist[miss, :], 1.0e-14)
        w = 1.0 / (d**float(power))
        w_sum = np.sum(w, axis=1, keepdims=True)
        w = w / np.maximum(w_sum, 1.0e-30)
        u_out[miss, :] = np.sum(u_nodes[idx[miss, :], :] * w[:, :, None], axis=1)
    return np.asarray(u_out, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute dx(t) at SVG mark points from simulation VTU files.")
    ap.add_argument("--sim-dir", type=str, required=True, help="Simulation output directory containing vtk/step=XXXX.vtu and timeseries.csv.")
    ap.add_argument("--out-csv", type=str, default="", help="Output CSV. Default: <sim-dir>/timeseries_svg_marks.csv")

    ap.add_argument("--svg-frame0", type=str, default="examples/biofilms/benchmarks/lie/svg_fles/frame_0000.svg", help="Frame-0 SVG containing mark paths.")
    ap.add_argument("--block-w-mm", type=float, default=1.0, help="Support width for SVG px->mm scaling (mm).")
    ap.add_argument("--cubic-samples", type=int, default=20, help="SVG cubic Bezier sampling.")
    ap.add_argument("--join-tol-px", type=float, default=5.0, help="SVG segment join tolerance (px).")

    ap.add_argument("--block-w", type=float, default=1.0e-3, help="Support width in simulation coordinates (m).")
    ap.add_argument("--block-h", type=float, default=3.0e-3, help="Support height in simulation coordinates (m).")
    ap.add_argument("--block-x0", type=float, default=float("nan"), help="Support left x in simulation coordinates (m). Default: infer from VTU x-extent and --block-w.")

    ap.add_argument("--fixed-point-iters", type=int, default=7, help="Fixed-point iterations for x = x_ref + u(x,t).")
    ap.add_argument("--idw-k", type=int, default=8, help="kNN used in IDW interpolation.")
    ap.add_argument("--idw-power", type=float, default=2.0, help="IDW power.")
    args = ap.parse_args()

    sim_dir = Path(str(args.sim_dir))
    vtk_dir = sim_dir / "vtk"
    ts_path = sim_dir / "timeseries.csv"
    if not vtk_dir.exists():
        raise FileNotFoundError(f"Missing VTK directory: {vtk_dir}")
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing timeseries.csv: {ts_path}")

    step_files = _vtk_step_files(vtk_dir)
    t_s = _read_timeseries_t(ts_path)
    if t_s.size != len(step_files):
        raise RuntimeError(
            f"timeseries.csv rows ({t_s.size}) != VTK steps ({len(step_files)}). "
            "This postprocessor assumes 1:1 time/VTK export (vtk-every=1, no dropped steps)."
        )

    # Load mesh coordinates from step 0 and build a KDTree.
    _step0, vtu0 = step_files[0]
    m0 = meshio.read(str(vtu0))
    xy = np.asarray(m0.points[:, :2], dtype=float)
    tree = cKDTree(xy)

    block_w = float(args.block_w)
    block_h = float(args.block_h)
    block_x0 = float(args.block_x0)
    if not np.isfinite(block_x0):
        block_x0 = _infer_block_x0(xy, block_w=block_w)

    # Extract marks from SVG frame 0 -> local mm coords (base-left anchored, y up).
    svg0 = Path(str(args.svg_frame0))
    if not svg0.exists():
        raise FileNotFoundError(f"SVG frame0 not found: {svg0}")
    geom0 = extract_svg_frame_geometry(
        svg0,
        block_w_mm=float(args.block_w_mm),
        cubic_samples=int(args.cubic_samples),
        join_tol_px=float(args.join_tol_px),
    )
    marks_px = extract_svg_mark_points_px(svg0, cubic_samples=int(args.cubic_samples), join_tol_px=float(args.join_tol_px))
    marks_px = np.asarray(marks_px, dtype=float).reshape((-1, 2))
    marks3_px = _select_three_mark_points_px(marks_px)
    marks3_mm = geom0.px_to_mm(marks3_px)

    # Convert to simulation global meters, order top->bottom by y (y up).
    marks3_m_local = marks3_mm * 1.0e-3
    pts_ref = np.column_stack([float(block_x0) + marks3_m_local[:, 0], float(block_h) + marks3_m_local[:, 1]])
    order = np.argsort(-pts_ref[:, 1])  # descending y
    pts_ref = np.asarray(pts_ref[order, :], dtype=float)

    out_csv = Path(str(args.out_csv)) if str(args.out_csv).strip() else (sim_dir / "timeseries_svg_marks.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fp = int(max(0, int(args.fixed_point_iters)))
    k = int(args.idw_k)
    power = float(args.idw_power)

    with out_csv.open("w", encoding="utf-8") as f:
        f.write("t_s,dx_line1_m,dx_line2_m,dx_line3_m\n")
        for (t, (_step, vtu)) in zip(t_s.tolist(), step_files):
            mesh = meshio.read(str(vtu))
            if "u" not in mesh.point_data:
                raise KeyError(f"Missing point_data['u'] in {vtu}")
            u = np.asarray(mesh.point_data["u"][:, :2], dtype=float)

            xy_ref = np.asarray(pts_ref, dtype=float)
            xy_cur = xy_ref.copy()
            if fp > 0:
                for _ in range(fp):
                    u_val = _interp_displacement_idw(tree=tree, u_nodes=u, xq=xy_cur, k=k, power=power)
                    xy_cur = xy_ref + u_val

            dx = xy_cur[:, 0] - xy_ref[:, 0]
            f.write(f"{float(t):.12e},{float(dx[0]):.12e},{float(dx[1]):.12e},{float(dx[2]):.12e}\n")

    print(f"[ok] wrote {out_csv} ({t_s.size} steps)")
    print(f"[info] block_x0={block_x0:.6e} m, block_h={block_h:.6e} m (block_w={block_w:.3e} m)")
    print(f"[info] svg marks (local mm, top->bottom):\\n{marks3_mm[order]}")

if __name__ == "__main__":
    main()
