#!/usr/bin/env python3
"""
Visualize the Lie benchmark *background mesh* (channel minus support block).

This reproduces the mesh construction used in
`examples/biofilms/benchmarks/lie/lie_synthetic_deformation_one_domain.py`:
we do **not** boolean-subtract geometry in a CAD sense. Instead, we build a
conforming structured quad mesh by meshing the five rectangles *around* the
support block and then merging coincident nodes.

The removed block corresponds to the "slit"/notch:
    [block_x0, block_x1] × [0, block_h]

The plot is produced using:
    `pycutfem/io/visualization.py::plot_mesh_2`

Example
-------
python -u examples/biofilms/benchmarks/lie/plot_background_mesh_slit_plot2.py \\
  --L 15e-3 --H 10e-3 --block-w 1e-3 --block-h 3e-3 \\
  --nx 120 --ny 80 --poly-order 2 \\
  --out-dir out/_lie_mesh_plot2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node
from pycutfem.io.visualization import plot_mesh_2
from pycutfem.utils.meshgen import structured_quad


def _auto_split_counts(total: int, parts: tuple[float, ...], *, min_each: int = 1) -> tuple[int, ...]:
    total = int(total)
    if total <= 0:
        raise ValueError("total must be > 0")
    w = np.asarray(parts, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("parts must be non-empty")
    if np.any(w < 0.0):
        raise ValueError("parts must be non-negative")
    if float(np.sum(w)) <= 0.0:
        raise ValueError("parts sum must be positive")

    weights = w / float(np.sum(w))
    counts = np.floor(weights * float(total)).astype(int)
    for i in range(counts.size):
        if w[i] > 0.0 and counts[i] < int(min_each):
            counts[i] = int(min_each)

    # Distribute remainder to the largest-weight bins.
    while int(np.sum(counts)) < total:
        i = int(np.argmax(weights))
        counts[i] += 1
    while int(np.sum(counts)) > total:
        cand = [i for i in range(counts.size) if counts[i] > int(min_each)]
        if not cand:
            break
        i = max(cand, key=lambda j: int(counts[j]))
        counts[i] -= 1
    return tuple(int(c) for c in counts.tolist())


def _merge_mesh_parts(parts: list[tuple[list[Node], np.ndarray, np.ndarray]]) -> tuple[list[Node], np.ndarray, np.ndarray]:
    lookup: dict[tuple[float, float], int] = {}
    nodes_out: list[Node] = []
    elems_out: list[np.ndarray] = []
    corners_out: list[np.ndarray] = []
    ndigits = 12

    for nodes, elems, corners in parts:
        if not nodes:
            continue
        local2global = np.empty((len(nodes),), dtype=int)
        for n in nodes:
            key = (round(float(n.x), ndigits), round(float(n.y), ndigits))
            gid = lookup.get(key)
            if gid is None:
                gid = len(nodes_out)
                lookup[key] = gid
                nodes_out.append(Node(gid, float(key[0]), float(key[1])))
            local2global[int(n.id)] = int(gid)

        elems_out.append(local2global[np.asarray(elems, dtype=int)])
        corners_out.append(local2global[np.asarray(corners, dtype=int)])

    if not elems_out:
        raise RuntimeError("No mesh parts were generated.")
    return nodes_out, np.vstack(elems_out), np.vstack(corners_out)


def _channel_minus_block_mesh(
    *,
    L: float,
    H: float,
    block_x0: float,
    block_x1: float,
    block_h: float,
    nx_left: int,
    nx_mid: int,
    nx_right: int,
    ny_bottom: int,
    ny_top: int,
    poly_order: int,
) -> tuple[list[Node], np.ndarray, np.ndarray]:
    L = float(L)
    H = float(H)
    block_x0 = float(block_x0)
    block_x1 = float(block_x1)
    block_h = float(block_h)
    if not (0.0 < block_x0 < block_x1 < L):
        raise ValueError("Block must lie strictly inside the channel in x.")
    if not (0.0 < block_h < H):
        raise ValueError("Block height must satisfy 0 < block_h < H.")

    w_left = block_x0
    w_mid = block_x1 - block_x0
    w_right = L - block_x1
    h_bot = block_h
    h_top = H - block_h

    parts: list[tuple[list[Node], np.ndarray, np.ndarray]] = []

    def add_part(*, w: float, h: float, nx: int, ny: int, off: tuple[float, float]) -> None:
        if float(w) <= 0.0 or float(h) <= 0.0 or int(nx) <= 0 or int(ny) <= 0:
            return
        nodes, elems, _edges, corners = structured_quad(float(w), float(h), nx=int(nx), ny=int(ny), poly_order=int(poly_order), offset=off)
        parts.append((nodes, np.asarray(elems, dtype=int), np.asarray(corners, dtype=int)))

    # We mesh the 5 rectangles around the block and skip the mid-bottom rectangle
    # (which would be the block region itself).
    add_part(w=w_left, h=h_bot, nx=nx_left, ny=ny_bottom, off=(0.0, 0.0))  # left-bottom
    add_part(w=w_left, h=h_top, nx=nx_left, ny=ny_top, off=(0.0, block_h))  # left-top
    add_part(w=w_mid, h=h_top, nx=nx_mid, ny=ny_top, off=(block_x0, block_h))  # mid-top
    add_part(w=w_right, h=h_bot, nx=nx_right, ny=ny_bottom, off=(block_x1, 0.0))  # right-bottom
    add_part(w=w_right, h=h_top, nx=nx_right, ny=ny_top, off=(block_x1, block_h))  # right-top

    return _merge_mesh_parts(parts)


def _tag_channel_with_block_boundaries(
    mesh: Mesh,
    *,
    L: float,
    H: float,
    block_x0: float,
    block_x1: float,
    block_h: float,
    tol: float = 1.0e-12,
) -> None:
    L = float(L)
    H = float(H)
    block_x0 = float(block_x0)
    block_x1 = float(block_x1)
    block_h = float(block_h)

    def _in_block_x(x: float) -> bool:
        return (block_x0 - tol) <= float(x) <= (block_x1 + tol)

    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - L) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - H) <= tol,
            # Support block surfaces (these become boundary edges because the block is removed from the mesh).
            "block_top": lambda x, y: abs(y - block_h) <= tol and _in_block_x(x),
            "block_left": lambda x, y: abs(x - block_x0) <= tol and (y <= block_h + tol),
            "block_right": lambda x, y: abs(x - block_x1) <= tol and (y <= block_h + tol),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot the channel mesh with a 1×3 mm support-block notch (slit).")
    ap.add_argument("--L", type=float, default=15.0e-3, help="Channel length [m].")
    ap.add_argument("--H", type=float, default=10.0e-3, help="Channel height [m].")
    ap.add_argument("--block-w", type=float, default=1.0e-3, help="Support width [m].")
    ap.add_argument("--block-h", type=float, default=3.0e-3, help="Support height [m].")
    ap.add_argument("--block-xc", type=float, default=float("nan"), help="Support center x [m]. Default: L/2.")
    ap.add_argument("--nx", type=int, default=120, help="Total cells in x (split across left/mid/right).")
    ap.add_argument("--ny", type=int, default=80, help="Total cells in y (split across bottom/top).")
    ap.add_argument("--poly-order", type=int, default=2, choices=(1, 2), help="Quad polynomial order for the mesh.")
    ap.add_argument("--out-dir", type=str, default="out/_lie_mesh_plot2")
    ap.add_argument("--zoom-pad-mm", type=float, default=1.0, help="Padding around the block for the zoom plot [mm].")
    ap.add_argument("--plot-nodes", action="store_true", help="Also plot mesh nodes (can be cluttered).")
    ap.add_argument(
        "--biofilm-poly-mm",
        type=str,
        default="examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm.csv",
        help="Biofilm polygon CSV in mm with local coords (base at y=0, centered at x=0). Use '' to disable overlay.",
    )
    args = ap.parse_args()

    L = float(args.L)
    H = float(args.H)
    block_w = float(args.block_w)
    block_h = float(args.block_h)
    block_xc = float(args.block_xc)
    if not np.isfinite(block_xc):
        block_xc = 0.5 * L
    block_x0 = block_xc - 0.5 * block_w
    block_x1 = block_xc + 0.5 * block_w

    nx_left, nx_mid, nx_right = _auto_split_counts(int(args.nx), (block_x0, block_w, L - block_x1), min_each=2)
    ny_bottom, ny_top = _auto_split_counts(int(args.ny), (block_h, H - block_h), min_each=2)

    nodes, elems, corners = _channel_minus_block_mesh(
        L=L,
        H=H,
        block_x0=block_x0,
        block_x1=block_x1,
        block_h=block_h,
        nx_left=nx_left,
        nx_mid=nx_mid,
        nx_right=nx_right,
        ny_bottom=ny_bottom,
        ny_top=ny_top,
        poly_order=int(args.poly_order),
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=int(args.poly_order),
    )
    for e in mesh.elements_list:
        e.tag = "fluid"
    _tag_channel_with_block_boundaries(mesh, L=L, H=H, block_x0=block_x0, block_x1=block_x1, block_h=block_h)

    # Plot in mm for readability.
    mesh.nodes_x_y_pos[:] *= 1.0e3
    L_mm = L * 1.0e3
    H_mm = H * 1.0e3
    bx0_mm = block_x0 * 1.0e3
    bw_mm = block_w * 1.0e3
    bh_mm = block_h * 1.0e3

    biofilm_poly = None
    if str(args.biofilm_poly_mm).strip():
        poly = np.genfromtxt(str(args.biofilm_poly_mm), delimiter=",", skip_header=1, dtype=float)
        if poly.ndim == 1:
            poly = poly.reshape(-1, 2)
        poly = np.asarray(poly, dtype=float)[:, :2]
        if poly.shape[0] < 3:
            raise ValueError(f"Biofilm polygon must have >=3 points; got {poly.shape[0]}")
        if not np.allclose(poly[0], poly[-1], rtol=0.0, atol=1.0e-12):
            poly = np.vstack([poly, poly[0]])
        # Local -> global: (x,y)=(0,0) sits at (block_xc, block_h).
        poly[:, 0] += float(block_xc * 1.0e3)
        poly[:, 1] += float(block_h * 1.0e3)
        biofilm_poly = poly

    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    def _draw(ax, *, zoom: bool) -> None:
        plot_mesh_2(
            mesh,
            show=False,
            ax=ax,
            plot_nodes=bool(args.plot_nodes),
            plot_edges=True,
            elem_tags=False,
            edge_colors=True,
            plot_interface=False,
            fluid_solid_overlay=False,
        )

        # Overlay the removed block as a white patch (purely for readability).
        ax.add_patch(Rectangle((bx0_mm, 0.0), bw_mm, bh_mm, facecolor="white", edgecolor="black", linewidth=2.2, zorder=10))
        # Mark the attachment line (top of the block).
        ax.plot([bx0_mm, bx0_mm + bw_mm], [bh_mm, bh_mm], color="red", linewidth=3.0, zorder=11, label="block top (biofilm clamp)")

        if biofilm_poly is not None:
            ax.fill(
                biofilm_poly[:, 0],
                biofilm_poly[:, 1],
                facecolor=(0.2, 0.8, 0.2, 0.18),
                edgecolor="none",
                zorder=12,
            )
            ax.plot(
                biofilm_poly[:, 0],
                biofilm_poly[:, 1],
                color=(0.1, 0.6, 0.1),
                linewidth=2.6,
                zorder=13,
                label="biofilm (exp frame 0)",
            )

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        if zoom:
            pad = float(args.zoom_pad_mm)
            ax.set_xlim(bx0_mm - pad, bx0_mm + bw_mm + pad)
            ax.set_ylim(0.0, bh_mm + 3.5 * pad)
            ax.set_title("Lie benchmark background mesh (zoom on 1×3 mm slit)")
        else:
            ax.set_xlim(0.0, L_mm)
            ax.set_ylim(0.0, H_mm)
            ax.set_title("Lie benchmark background mesh (full domain)")
        ax.set_aspect("equal", "box")
        ax.legend(loc="upper right", frameon=True)

    fig, ax = plt.subplots(figsize=(10.0, 5.2), dpi=200)
    _draw(ax, zoom=False)
    fig.tight_layout()
    out_full = out_dir / "mesh_slit_full_plot_mesh_2_with_biofilm.png" if biofilm_poly is not None else out_dir / "mesh_slit_full_plot_mesh_2.png"
    fig.savefig(out_full)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=220)
    _draw(ax, zoom=True)
    fig.tight_layout()
    out_zoom = out_dir / "mesh_slit_zoom_plot_mesh_2_with_biofilm.png" if biofilm_poly is not None else out_dir / "mesh_slit_zoom_plot_mesh_2.png"
    fig.savefig(out_zoom)
    plt.close(fig)

    print(f"[ok] wrote {out_full}")
    print(f"[ok] wrote {out_zoom}")


if __name__ == "__main__":
    main()
