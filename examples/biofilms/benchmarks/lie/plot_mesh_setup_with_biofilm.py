#!/usr/bin/env python3
"""
Plot the Lie benchmark mesh setup with a marked biofilm footprint.

This is a lightweight visualization helper (no FEniCS runtime required). It
plots the rectangular channel mesh (structured quads) and overlays the biofilm
polygon (typically extracted from Video S3) after scaling/translation.

The experiment uses a cylindrical support (1 mm diameter × 3 mm height). In 2D,
we represent it as a rectangular block (width=1 mm, height=3 mm) glued to the
bottom wall at the channel center; the fluid domain is the channel minus this
block, and the biofilm is attached to the block top.

Example
-------
python examples/biofilms/benchmarks/lie/plot_mesh_setup_with_biofilm.py \\
  --L 15e-3 --H 10e-3 --nx 200 --ny 80 \\
  --poly-csv examples/biofilms/benchmarks/lie/biofilm_v3_frame0_polygon_mm_simpl2.csv \\
  --poly-scale 1e-3 --align-poly-to-block \\
  --out-dir out/_lie_setup
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _read_polygon_csv(path: str | Path) -> np.ndarray:
    arr = np.genfromtxt(str(path), delimiter=",", skip_header=1, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    arr = np.asarray(arr, dtype=float)
    if arr.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in polygon csv; got shape={arr.shape}")
    return arr[:, :2]


def _mesh_segments_mm(*, L_m: float, H_m: float, nx: int, ny: int, stride: int) -> list[list[tuple[float, float]]]:
    L_mm = float(L_m) * 1.0e3
    H_mm = float(H_m) * 1.0e3
    nx = int(nx)
    ny = int(ny)
    stride = max(1, int(stride))
    xs = np.linspace(0.0, L_mm, nx + 1)
    ys = np.linspace(0.0, H_mm, ny + 1)

    segs: list[list[tuple[float, float]]] = []
    for i in range(0, nx + 1, stride):
        x = float(xs[i])
        segs.append([(x, 0.0), (x, H_mm)])
    for j in range(0, ny + 1, stride):
        y = float(ys[j])
        segs.append([(0.0, y), (L_mm, y)])
    return segs


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot mesh setup with biofilm polygon overlay (Lie benchmark).")
    ap.add_argument("--L", type=float, default=15.0e-3, help="Channel length [m].")
    ap.add_argument("--H", type=float, default=10.0e-3, help="Channel height [m].")
    ap.add_argument("--nx", type=int, default=200)
    ap.add_argument("--ny", type=int, default=80)
    ap.add_argument("--block-w", type=float, default=1.0e-3, help="Support (block) width [m].")
    ap.add_argument("--block-h", type=float, default=3.0e-3, help="Support (block) height [m].")
    ap.add_argument(
        "--block-xc",
        type=float,
        default=float("nan"),
        help="Support center x-coordinate [m]. Default: L/2.",
    )
    ap.add_argument("--poly-csv", type=str, required=True, help="Polygon CSV with columns x_mm,y_mm (header allowed).")
    ap.add_argument("--poly-scale", type=float, default=1.0e-3, help="Scale to apply to polygon coordinates (mm->m: 1e-3).")
    ap.add_argument("--poly-tx", type=float, default=0.0, help="Translate polygon x by tx [m].")
    ap.add_argument("--poly-ty", type=float, default=0.0, help="Translate polygon y by ty [m].")
    ap.add_argument("--align-ymin-to-zero", action="store_true", help="Translate polygon ymin to y=0 (legacy helper).")
    ap.add_argument(
        "--align-poly-to-block",
        action="store_true",
        help="Translate polygon so that (x_center,y_min) aligns to (block_xc, block_h).",
    )
    ap.add_argument("--mesh-stride-full", type=int, default=5, help="Plot every N-th mesh line in the full view.")
    ap.add_argument("--mesh-stride-zoom", type=int, default=1, help="Plot every N-th mesh line in the zoom view.")
    ap.add_argument("--zoom-pad-mm", type=float, default=0.6, help="Padding around polygon bbox for zoom plot [mm].")
    ap.add_argument("--out-dir", type=str, default="out/_lie_setup")
    args = ap.parse_args()

    # Headless-safe backend (must be set before importing pyplot).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle

    poly_mm = _read_polygon_csv(args.poly_csv)
    poly_m = poly_mm * float(args.poly_scale) + np.array([float(args.poly_tx), float(args.poly_ty)], dtype=float)
    if args.align_ymin_to_zero:
        poly_m = poly_m + np.array([0.0, -float(np.min(poly_m[:, 1]))], dtype=float)

    L_m = float(args.L)
    H_m = float(args.H)
    block_w = float(args.block_w)
    block_h = float(args.block_h)
    block_xc = float(args.block_xc)
    if not np.isfinite(block_xc):
        block_xc = 0.5 * L_m
    block_x0 = block_xc - 0.5 * block_w
    block_x1 = block_xc + 0.5 * block_w

    if args.align_poly_to_block:
        poly_xc = 0.5 * (float(np.min(poly_m[:, 0])) + float(np.max(poly_m[:, 0])))
        poly_ymin = float(np.min(poly_m[:, 1]))
        poly_m = poly_m + np.array([block_xc - poly_xc, block_h - poly_ymin], dtype=float)
    poly_plot_mm = poly_m * 1.0e3

    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_full = out_dir / "mesh_setup_full.png"
    out_zoom = out_dir / "mesh_setup_zoom.png"

    def plot_one(*, zoom: bool) -> None:
        fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=200)
        stride = int(args.mesh_stride_zoom if zoom else args.mesh_stride_full)
        segs = _mesh_segments_mm(L_m=L_m, H_m=H_m, nx=int(args.nx), ny=int(args.ny), stride=stride)
        lc = LineCollection(segs, colors="0.85", linewidths=0.6)
        ax.add_collection(lc)

        # Domain rectangle
        L_mm = L_m * 1.0e3
        H_mm = H_m * 1.0e3
        ax.plot([0, L_mm, L_mm, 0, 0], [0, 0, H_mm, H_mm, 0], color="0.1", linewidth=1.1)

        # Support block (paint it after mesh lines to "cut out" the mesh visually).
        bx0_mm = block_x0 * 1.0e3
        bw_mm = block_w * 1.0e3
        bh_mm = block_h * 1.0e3
        ax.add_patch(Rectangle((bx0_mm, 0.0), bw_mm, bh_mm, facecolor="white", edgecolor="0.0", linewidth=1.6, zorder=5))

        # Emphasize the bottom wall segments outside the block.
        ax.plot([0.0, bx0_mm], [0.0, 0.0], color="0.0", linewidth=2.0)
        ax.plot([bx0_mm + bw_mm, L_mm], [0.0, 0.0], color="0.0", linewidth=2.0)

        # Biofilm polygon
        ax.plot(poly_plot_mm[:, 0], poly_plot_mm[:, 1], color="tab:green", linewidth=2.0, label="biofilm")
        ax.fill(poly_plot_mm[:, 0], poly_plot_mm[:, 1], color="tab:green", alpha=0.18)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

        if zoom:
            pad = float(args.zoom_pad_mm)
            xmin, ymin = np.min(poly_plot_mm, axis=0)
            xmax, ymax = np.max(poly_plot_mm, axis=0)
            ax.set_xlim(max(0.0, xmin - pad), min(L_mm, xmax + pad))
            ax.set_ylim(max(0.0, ymin - pad), min(H_mm, ymax + pad))
            ax.set_title("Lie benchmark: mesh + biofilm (zoom)")
        else:
            ax.set_xlim(0.0, L_mm)
            ax.set_ylim(0.0, H_mm)
            ax.set_title("Lie benchmark: mesh + biofilm (full domain)")

        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        fig.savefig(str(out_zoom if zoom else out_full))
        plt.close(fig)

    plot_one(zoom=False)
    plot_one(zoom=True)

    print(f"[ok] wrote {out_full}")
    print(f"[ok] wrote {out_zoom}")


if __name__ == "__main__":
    main()
