"""Example 4.1 (paper) geometry: two-mesh setup (outer fluid mesh + inner porous mesh).

This script only builds the *two meshes* and the associated CutFEM tags needed
to reproduce the paper's Fig. 3 setup:

- Ω^F: outer square (size 1) rotated by 45°, truncated by a vertical cut x=x0.
- Ω^P: inner square (size 0.5) rotated by 30°.

The intended workflow is:
1) Verify boundary tagging/DOF activity for both meshes.
2) Use Ω^P geometry to tag Ω^F "inactive" nodes/elements inside the porous solid.
3) Couple the two discretizations via non-matching Nitsche on Γ^FP.

For now this file focuses on steps (1)-(2): robust mesh generation + diagnostics.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from pycutfem.core.levelset import AffineLevelSet, MinLevelSet, RotatedBoxLevelSet, ScaledLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.io.visualization import plot_mesh_2
from pycutfem.utils.meshgen import structured_quad


def build_two_meshes(*, nx_f: int, nx_p: int, poly_order: int, x0: float = -0.45):
    # --- geometry level sets (paper sec. 4.1) ---
    poro_ls = RotatedBoxLevelSet(center=(0.0, 0.0), hx=0.25, hy=0.25, angle=math.pi / 6.0)  # Ω^P (30°)

    # Fluid mesh: rotated 45° square mesh (size 1) like paper Fig. 3.
    nodes_f, elems_f, edges_f, corners_f = structured_quad(
        1.0,
        1.0,
        nx=nx_f,
        ny=nx_f,
        poly_order=poly_order,
        offset=(-0.5, -0.5),
        rotation=math.pi / 4.0,
        rotation_center=(0.0, 0.0),
    )
    mesh_f = Mesh(
        nodes=nodes_f,
        element_connectivity=elems_f,
        edges_connectivity=edges_f,
        elements_corner_nodes=corners_f,
        element_type="quad",
        poly_order=poly_order,
    )

    # Truncation: keep {x >= x0}. We use the sign convention from existing Example 4.1 code:
    # `cut_ls` is negative on the physical side, and `cut_pos` is positive on {x >= x0}.
    cut_ls = AffineLevelSet(-1.0, 0.0, float(x0))  # φ = x0 - x
    cut_pos = ScaledLevelSet(-1.0, cut_ls)

    # Fluid activity level set: positive in Ω^F, negative in removed or porous region.
    fluid_ls = MinLevelSet(poro_ls, cut_pos)

    # Tag CutFEM interface segments on the fluid mesh (Γ^FP and the inlet cut are both inside).
    mesh_f.classify_elements(fluid_ls)
    mesh_f.classify_edges(fluid_ls)
    mesh_f.build_interface_segments(fluid_ls)

    # Porous mesh: rotated 30° square mesh (size 0.5) centered at origin.
    nodes_p, elems_p, edges_p, corners_p = structured_quad(
        0.5,
        0.5,
        nx=nx_p,
        ny=nx_p,
        poly_order=poly_order,
        offset=(-0.25, -0.25),
        rotation=math.pi / 6.0,
        rotation_center=(0.0, 0.0),
    )
    mesh_p = Mesh(
        nodes=nodes_p,
        element_connectivity=elems_p,
        edges_connectivity=edges_p,
        elements_corner_nodes=corners_p,
        element_type="quad",
        poly_order=poly_order,
    )

    # Boundary tags on the porous mesh: mark all outer edges as "interface".
    # (The poro mesh is *exactly* Ω^P, so its whole boundary participates in Γ^FP.)
    mesh_p.tag_boundary_edges({"interface": lambda x, y: True})

    # Identify fluid nodes strictly inside Ω^P (these should be made inactive for fluid unknowns).
    phi_poro_on_f_nodes = np.asarray(poro_ls(mesh_f.nodes_x_y_pos), dtype=float)
    inside_poro_nodes = np.flatnonzero(phi_poro_on_f_nodes < 0.0)

    return dict(
        mesh_f=mesh_f,
        mesh_p=mesh_p,
        poro_ls=poro_ls,
        cut_ls=cut_ls,
        fluid_ls=fluid_ls,
        inside_poro_nodes=inside_poro_nodes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx-f", type=int, default=16, help="Fluid mesh resolution (outer square).")
    parser.add_argument("--nx-p", type=int, default=0, help="Poro mesh resolution (inner square). Default: nx_f//2.")
    parser.add_argument("--p", type=int, default=1, help="Polynomial order of the geometry mesh.")
    parser.add_argument("--x0", type=float, default=-0.45, help="Vertical cut x=x0 (paper: -0.45).")
    parser.add_argument("--plot", action="store_true", help="Save mesh plots.")
    parser.add_argument("--outdir", type=str, default="examples/FPI/_two_mesh_setup")
    args = parser.parse_args()

    nx_f = int(args.nx_f)
    nx_p = int(args.nx_p) if int(args.nx_p) > 0 else max(1, nx_f // 2)
    poly = int(args.p)

    prob = build_two_meshes(nx_f=nx_f, nx_p=nx_p, poly_order=poly, x0=float(args.x0))
    mesh_f: Mesh = prob["mesh_f"]
    mesh_p: Mesh = prob["mesh_p"]
    poro_ls = prob["poro_ls"]
    cut_ls = prob["cut_ls"]
    fluid_ls = prob["fluid_ls"]
    inside = np.asarray(prob["inside_poro_nodes"], dtype=int)

    # --- summary stats ---
    n_cut = sum(1 for e in mesh_f.elements_list if getattr(e, "tag", "") == "cut")
    n_inside = sum(1 for e in mesh_f.elements_list if getattr(e, "tag", "") == "inside")
    n_outside = sum(1 for e in mesh_f.elements_list if getattr(e, "tag", "") == "outside")

    # Count interface segments on the fluid mesh (per-element list).
    n_seg = 0
    for e in mesh_f.elements_list:
        segs = getattr(e, "interface_segments", None)
        if segs:
            n_seg += len(segs)

    print("Two-mesh Example 4.1 setup")
    print(f"  fluid mesh:  nx_f={nx_f}, p={poly}, n_elem={mesh_f.n_elements}, n_nodes={mesh_f.nodes_x_y_pos.shape[0]}")
    print(f"    tags: inside={n_inside}, outside={n_outside}, cut={n_cut}")
    print(f"    interface segments (fluid_ls): {n_seg}")
    print(f"    fluid nodes inside Ω^P (to be deactivated): {inside.size}")
    print(f"  poro mesh:   nx_p={nx_p}, p={poly}, n_elem={mesh_p.n_elements}, n_nodes={mesh_p.nodes_x_y_pos.shape[0]}")

    if args.plot:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Fluid mesh plot: show both poro boundary (fluid_ls=0) and inlet cut (cut_ls=0).
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 9))
        plot_mesh_2(mesh_f, level_set=fluid_ls, show=False, ax=ax, plot_nodes=False, plot_interface=True)
        ax.set_title("Fluid mesh (rot 45°), with fluid_ls=0 (poro boundary + inlet)")
        fig.savefig(outdir / f"mesh_fluid_nx{nx_f}_p{poly}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 8))
        plot_mesh_2(mesh_p, level_set=poro_ls, show=False, ax=ax, plot_nodes=False, plot_interface=False)
        ax.set_title("Porous mesh (rot 30°), with poro_ls=0")
        fig.savefig(outdir / f"mesh_poro_nx{nx_p}_p{poly}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Combined overlay (lines only) for quick visual confirmation.
        fig, ax = plt.subplots(figsize=(10, 9))
        plot_mesh_2(mesh_f, show=False, ax=ax, plot_nodes=False, elem_tags=False, edge_colors=False, plot_interface=False)
        plot_mesh_2(mesh_p, show=False, ax=ax, plot_nodes=False, elem_tags=False, edge_colors=False, plot_interface=False)
        ax.set_aspect("equal")
        ax.set_title("Overlay: fluid mesh (45°) + poro mesh (30°)")
        fig.savefig(outdir / f"mesh_overlay_nxF{nx_f}_nxP{nx_p}_p{poly}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()

