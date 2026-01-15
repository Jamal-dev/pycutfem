"""
Showcase: what do `dCut`, `dCutSkeleton`, `dInterface`, and `dS` measure?

We work with a background mesh T_h on a rectangle Ω ⊂ R² and a level set φ(x).
The level set induces the physical subdomains

    Ω⁻ = { x ∈ Ω : φ(x) < 0 },   Ω⁺ = { x ∈ Ω : φ(x) > 0 },   Γ = { x ∈ Ω : φ(x) = 0 }.

CutFEM uses different integration measures depending on what geometric object the
weak form targets:

1) "dCut"  (volume, cut by φ):
   ∫_{Ω⁻} f dx   or   ∫_{Ω⁺} f dx

   In PyCutFEM this is expressed as `dx(level_set=φ, metadata={'side': '-', ...})`
   (or `side='+'`). Mathematically this is the standard Lebesgue integral over
   the *physical* subdomain Ω±, implemented by integrating over cut cells K∩Ω±.

2) `dInterface`  (codimension-1 interface integral on Γ):
   ∫_{Γ} g ds

   This is used for interface conditions (Nitsche coupling, surface tension,
   transmission jumps, etc.). The geometry is the zero level set Γ itself.

3) `dS`  (boundary integral on ∂Ω):
   ∫_{∂Ω} h ds

   This is a standard boundary-edge measure (Neumann data, fluxes, etc.).

4) `dCutSkeleton`  (interior-facet measure restricted by φ):
   ∫_{F_h ∩ Ω±} r ds

   where F_h is (a selected subset of) the mesh skeleton (interior facets).
   This is *not* the physical interface Γ. It is a facet-based measure used by
   stabilizations such as continuous interior penalty (CIP) / ghost penalties:

   - The stabilized bilinear form penalizes jumps of (possibly higher) normal
     derivatives across interior facets to control cut-cell ill-conditioning.
   - Restricting to Ω± (via φ) keeps the stabilization consistent with the
     physical subdomain and avoids penalizing facets on the opposite side.

This script prints simple "unit" integrals (integrand = 1) to make the geometric
meaning visible. In 2D:
  - dCut integrals return *areas* (units: length²)
  - dInterface, dS, dCutSkeleton return *lengths* (units: length)
"""

from __future__ import annotations

import argparse
import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad

from pycutfem.ufl.expressions import Constant
from pycutfem.ufl.measures import dx, dInterface, dCutSkeleton, dS
from pycutfem.ufl.forms import assemble_form, Equation


def _as_float(x) -> float:
    arr = np.asarray(x, dtype=float)
    return float(arr.sum()) if arr.shape else float(arr)


def assemble_scalar(dh: DofHandler, integral, *, name: str, backend: str) -> float:
    res = assemble_form(
        Equation(None, integral),
        dof_handler=dh,
        assembler_hooks={integral.integrand: {"name": name}},
        backend=backend,
    )
    return _as_float(res[name])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["python", "jit", "cpp"], default="cpp")
    ap.add_argument("--nx", type=int, default=40)
    ap.add_argument("--ny", type=int, default=40)
    ap.add_argument("--Lx", type=float, default=2.0)
    ap.add_argument("--Ly", type=float, default=2.0)
    ap.add_argument("--geom-order", type=int, default=1)
    ap.add_argument("--q", type=int, default=8, help="Quadrature order (per measure).")
    ap.add_argument("--radius", type=float, default=0.7)
    args = ap.parse_args()

    # --- background mesh Ω ----------------------------------------------------
    nodes, elems, _, corners = structured_quad(
        args.Lx,
        args.Ly,
        nx=args.nx,
        ny=args.ny,
        poly_order=args.geom_order,
        offset=[-args.Lx / 2.0, -args.Ly / 2.0],
    )
    mesh = Mesh(
        nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=args.geom_order,
    )

    # Boundary tagging for dS
    tol = 1e-12
    bc_tags = {
        "left": lambda x, y: abs(x + args.Lx / 2.0) <= tol,
        "right": lambda x, y: abs(x - args.Lx / 2.0) <= tol,
        "bottom": lambda x, y: abs(y + args.Ly / 2.0) <= tol,
        "top": lambda x, y: abs(y - args.Ly / 2.0) <= tol,
    }
    mesh.tag_boundary_edges(bc_tags)
    boundary_bs = mesh.edge_bitset("left") | mesh.edge_bitset("right") | mesh.edge_bitset("bottom") | mesh.edge_bitset("top")

    # --- level set φ and CutFEM classification -------------------------------
    level_set = CircleLevelSet(center=(0.0, 0.0), radius=float(args.radius))
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    # A dummy FE space is enough for scalar functionals
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    # --- Measures -------------------------------------------------------------
    q_meta = {"q": int(args.q)}

    # "dCut": integrate over Ω±
    dx_neg = dx(level_set=level_set, metadata={**q_meta, "side": "-"})
    dx_pos = dx(level_set=level_set, metadata={**q_meta, "side": "+"})

    # Interface Γ
    d_gamma = dInterface(level_set=level_set, metadata=q_meta)

    # Boundary ∂Ω
    d_boundary = dS(defined_on=boundary_bs, metadata=q_meta)

    # Skeleton facets near the cut (use the "ghost" interior facets as a typical stabilization set)
    ghost_bs = mesh.edge_bitset("ghost")
    d_skel_neg = dCutSkeleton(defined_on=ghost_bs, level_set=level_set, metadata={**q_meta, "side": "-"})
    d_skel_pos = dCutSkeleton(defined_on=ghost_bs, level_set=level_set, metadata={**q_meta, "side": "+"})

    # --- Assemble "unit" integrals -------------------------------------------
    area_total = assemble_scalar(dh, Constant(1.0) * dx(metadata=q_meta), name="area", backend=args.backend)
    area_neg = assemble_scalar(dh, Constant(1.0) * dx_neg, name="area_neg", backend=args.backend)
    area_pos = assemble_scalar(dh, Constant(1.0) * dx_pos, name="area_pos", backend=args.backend)
    len_ifc = assemble_scalar(dh, Constant(1.0) * d_gamma, name="len_ifc", backend=args.backend)
    len_bnd = assemble_scalar(dh, Constant(1.0) * d_boundary, name="len_bnd", backend=args.backend)
    len_skel_neg = assemble_scalar(dh, Constant(1.0) * d_skel_neg, name="len_skel_neg", backend=args.backend)
    len_skel_pos = assemble_scalar(dh, Constant(1.0) * d_skel_pos, name="len_skel_pos", backend=args.backend)

    # --- Print ---------------------------------------------------------------
    print("=== CutFEM measure showcase (2D) ===")
    print(f"backend        : {args.backend}")
    print(f"mesh           : structured quad {args.nx}×{args.ny}, geom_order={args.geom_order}")
    print(f"Ω              : [{-args.Lx/2:g},{args.Lx/2:g}]×[{-args.Ly/2:g},{args.Ly/2:g}]")
    print(f"φ              : circle(center=(0,0), radius={args.radius:g})")
    print(f"quadrature q   : {args.q}")
    print()
    print("Volume measures (area):")
    print(f"  ∫_Ω 1 dx                       = {area_total:.16e}")
    print(f"  ∫_{'{'}φ<0{'}'} 1 dCut (Ω⁻)      = {area_neg:.16e}")
    print(f"  ∫_{'{'}φ>0{'}'} 1 dCut (Ω⁺)      = {area_pos:.16e}")
    print(f"  (Ω⁻ + Ω⁺) / Ω                  = {(area_neg + area_pos) / max(area_total, 1e-30):.16e}")
    print()
    print("Codimension-1 measures (length):")
    print(f"  ∫_Γ 1 dInterface               = {len_ifc:.16e}    (≈ 2πR = {2*np.pi*args.radius:.16e})")
    print(f"  ∫_∂Ω 1 dS                      = {len_bnd:.16e}    (exact = {2*(args.Lx + args.Ly):.16e})")
    print()
    print("Skeleton measure (length on selected interior facets):")
    print(f"  ∫_{'{' }F_h∩Ω⁻{'}'} 1 dCutSkeleton = {len_skel_neg:.16e}")
    print(f"  ∫_{'{' }F_h∩Ω⁺{'}'} 1 dCutSkeleton = {len_skel_pos:.16e}")
    print()
    print("Notes:")
    print("  - dInterface integrates on the *physical* interface Γ (φ=0).")
    print("  - dCutSkeleton integrates on interior facets F_h (a mesh object),")
    print("    optionally restricted to a stabilization set (here: 'ghost' facets).")


if __name__ == "__main__":
    main()
