#!/usr/bin/env python3
"""
Compare PyCutFEM `dFacetPatch` against NGSolve/XFEM `dFacetPatch`.

Why:
  Most existing comparisons use constant probe fields, for which facet-patch
  jump terms can be ~0 (global low-order polynomials are in the nullspace).
  This script uses smooth non-polynomial probes so the facet-patch energies are
  non-zero and discrepancies show up.

Run (requires ngsolve-dev env):
  conda run --no-capture-output -n ngsolve-dev \\
    BACKEND=cpp python examples/debug/compare_facet_patch_to_ngsolve.py

Notes:
  - We split facet-patch measures by side (ghost_neg / ghost_pos) to validate
    per-side selection and sign/ordering robustness.

Important:
  Comparing facet-patch energies requires the *same discrete function* in both
  libraries. NGSolve's `H1` uses a hierarchical/modal basis while PyCutFEM uses
  nodal Lagrange bases. For a general smooth probe, the two libraries embed the
  analytic function into their spaces differently (different DOF functionals),
  which can show up as ~O(1%) differences even if the integral itself is correct.

  To eliminate this effect, the default probe is a piecewise polynomial that is
  exactly representable in both spaces (hence energies should match to roundoff).
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

# ------------------------- PyCutFEM -------------------------
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.core.mesh import Mesh as PCMesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import VectorTestFunction, VectorTrialFunction, grad, inner, jump
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dFacetPatch
from pycutfem.utils.meshgen import structured_quad

# ------------------------- NGSolve/XFEM -------------------------
from netgen.meshing import Element1D, Element2D, FaceDescriptor, Mesh as NetgenMesh, MeshPoint, Pnt
from ngsolve import (  # noqa: F401
    BilinearForm,
    CoefficientFunction,
    Compress,
    FESpace,
    Grad,
    GridFunction,
    H1,
    IfPos,
    InnerProduct,
    Mesh,
    TaskManager,
    VectorH1,
    cos,
    sin,
    x,
    y,
)
from xfem import (  # noqa: F401
    CutInfo,
    GetDofsOfElements,
    GetFacetsWithNeighborTypes,
    HASNEG,
    HASPOS,
    IF,
    NEG,
    POS,
    dCut,
    dFacetPatch as ng_dFacetPatch,
)
from xfem.lsetcurv import InterpolateToP1  # noqa: F401


def _err_abs_rel(a: float, b: float) -> tuple[float, float]:
    abs_err = float(abs(a - b))
    scale = max(float(abs(a)), float(abs(b)))
    rel_err = abs_err / scale if scale > 0.0 else 0.0
    return abs_err, rel_err


def assemble_and_energy_pc(form, dh: DofHandler, u_vec: np.ndarray, v_vec: np.ndarray, *, backend: str) -> float:
    K, _ = assemble_form(Equation(form, None), dof_handler=dh, backend=backend)
    return float(v_vec @ (K @ u_vec))


def assemble_and_energy_ng(bf, gfu, gfv) -> float:
    with TaskManager():
        bf.Assemble()
    tmp = gfv.vec.CreateVector()
    bf.Apply(gfu.vec, tmp)
    return float(InnerProduct(gfv.vec, tmp))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, default=2, help="Polynomial order (velocity)")
    parser.add_argument("--q", type=int, default=8, help="Facet-patch quadrature order")
    parser.add_argument("--nx", type=int, default=16, help="PyCutFEM structured mesh cells in x/y")
    parser.add_argument("--L", type=float, default=2.0, help="Domain size in x ([-L/2, L/2])")
    parser.add_argument("--H", type=float, default=2.0, help="Domain size in y ([-H/2, H/2])")
    parser.add_argument("--R", type=float, default=2.0 / 3.0, help="Circle radius for the level set")
    parser.add_argument(
        "--probe",
        choices=("piecewise_poly", "analytic"),
        default="piecewise_poly",
        help="Probe type: 'piecewise_poly' matches to roundoff; 'analytic' may differ ~O(1%) due to projection.",
    )
    parser.add_argument("--rtol", type=float, default=5.0e-12, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1.0e-10, help="Absolute tolerance")
    args = parser.parse_args()

    backend = os.getenv("BACKEND", "python").lower()
    if backend not in {"python", "jit", "cpp", "c++"}:
        raise ValueError(f"Unsupported BACKEND='{backend}'. Use python/jit/cpp.")

    order = int(args.order)
    q = int(args.q)
    L = float(args.L)
    H = float(args.H)
    R = float(args.R)

    # ------------------------------------------------------------
    # PyCutFEM setup (structured quad mesh)
    # ------------------------------------------------------------
    nodes, elems, edges, corners = structured_quad(
        L,
        H,
        nx=int(args.nx),
        ny=int(args.nx),
        poly_order=1,
        offset=(-L / 2.0, -H / 2.0),
    )
    pc_mesh = PCMesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    me = MixedElement(
        pc_mesh,
        field_specs={
            "u_pos_x": order,
            "u_pos_y": order,
            "u_neg_x": order,
            "u_neg_y": order,
            "lm": ":number:",
        },
    )
    dh = DofHandler(me, method="cg")

    ls = CircleLevelSet(center=(0.0, 0.0), radius=R)
    dh.classify_from_levelset(ls)

    ghost_pos = pc_mesh.edge_bitset("ghost_pos")
    ghost_neg = pc_mesh.edge_bitset("ghost_neg")
    ghost_both = pc_mesh.edge_bitset("ghost_both")
    if ghost_pos.cardinality() == 0 or ghost_neg.cardinality() == 0:
        raise RuntimeError(
            f"Empty ghost sets: ghost_pos={ghost_pos.cardinality()} ghost_neg={ghost_neg.cardinality()}. "
            "Try different nx/R."
        )
    print(
        "[pc] ghost edges: "
        f"ghost_pos={ghost_pos.cardinality()} ghost_neg={ghost_neg.cardinality()} ghost_both={ghost_both.cardinality()}"
    )

    Vpos = FunctionSpace("Vpos", ["u_pos_x", "u_pos_y"], side="+")
    Vneg = FunctionSpace("Vneg", ["u_neg_x", "u_neg_y"], side="-")
    up = VectorTrialFunction(Vpos, dh, side="+")
    vp = VectorTestFunction(Vpos, dh, side="+")
    un = VectorTrialFunction(Vneg, dh, side="-")
    vn = VectorTestFunction(Vneg, dh, side="-")

    # Match NGSolve's HASPOS/HASNEG facet-patch sets:
    # - (+) side uses {POS-IF facets} ∪ {IF-IF facets}  -> ghost_pos ∪ ghost_both
    # - (-) side uses {NEG-IF facets} ∪ {IF-IF facets}  -> ghost_neg ∪ ghost_both
    dW_pos = dFacetPatch(defined_on=(ghost_pos | ghost_both), level_set=ls, metadata={"q": q})
    dW_neg = dFacetPatch(defined_on=(ghost_neg | ghost_both), level_set=ls, metadata={"q": q})

    a_mass_pc = inner(jump(un), jump(vn)) * dW_neg + inner(jump(up), jump(vp)) * dW_pos
    a_diff_pc = inner(grad(jump(un)), grad(jump(vn))) * dW_neg + inner(grad(jump(up)), grad(jump(vp))) * dW_pos

    # Probe fields:
    # - piecewise_poly: piecewise quadratic, continuous across x=x_split.
    #                  Exactly representable in both spaces (preferred).
    # - analytic      : smooth non-polynomial; representations differ between bases.
    x_split = (-L / 2.0) + (L / float(int(args.nx))) * float(int(args.nx) // 2)

    def u_pos_exact(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x0 = xy[:, 0]
        y0 = xy[:, 1]
        if args.probe == "analytic":
            ux = np.sin(2.0 * math.pi * x0) + 0.25 * np.cos(math.pi * y0) + 0.1 * x0 * y0
            uy = np.cos(2.0 * math.pi * y0) - 0.15 * np.sin(math.pi * x0) + 0.05 * x0 * x0
            return ux, uy

        base_x = 1.0 + 0.5 * x0 + 0.25 * y0 + 0.75 * x0 * y0
        base_y = -0.25 + 0.15 * x0 - 0.6 * y0 + 0.4 * x0 * x0 + 0.2 * y0 * y0
        dx = x0 - x_split
        # Right-side variation (keeps continuity at x=x_split)
        ux = base_x + np.where(dx >= 0.0, 7.0 * dx * (x0 + 0.5 * y0), 0.0)
        uy = base_y + np.where(dx >= 0.0, -5.0 * dx * (0.25 * x0 - y0), 0.0)
        return ux, uy

    def u_neg_exact(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x0 = xy[:, 0]
        y0 = xy[:, 1]
        if args.probe == "analytic":
            ux = -0.7 * np.cos(2.0 * math.pi * x0) + 0.2 * y0 + 0.05 * x0 * y0
            uy = 0.4 * np.sin(2.0 * math.pi * y0) + 0.1 * x0 - 0.07 * np.cos(math.pi * y0)
            return ux, uy

        base_x = -0.4 + 0.3 * x0 - 0.2 * y0 + 0.1 * x0 * x0 - 0.15 * x0 * y0
        base_y = 0.6 + 0.25 * x0 + 0.35 * y0 - 0.05 * y0 * y0 + 0.2 * x0 * y0
        dx = x0 - x_split
        ux = base_x + np.where(dx <= 0.0, 6.0 * dx * (x0 - 0.25 * y0), 0.0)
        uy = base_y + np.where(dx <= 0.0, 4.0 * dx * (0.5 * x0 + y0), 0.0)
        return ux, uy

    u_vec = np.zeros(dh.total_dofs, dtype=float)
    v_vec = np.zeros(dh.total_dofs, dtype=float)

    for field, vals_fn in (
        ("u_pos_x", lambda xy: u_pos_exact(xy)[0]),
        ("u_pos_y", lambda xy: u_pos_exact(xy)[1]),
        ("u_neg_x", lambda xy: u_neg_exact(xy)[0]),
        ("u_neg_y", lambda xy: u_neg_exact(xy)[1]),
    ):
        idx = np.asarray(dh.get_field_slice(field), dtype=int)
        xy = np.asarray(dh.get_field_dof_coords(field), dtype=float)
        vals = np.asarray(vals_fn(xy), dtype=float)
        u_vec[idx] = vals
        v_vec[idx] = vals

    # ------------------------------------------------------------
    # NGSolve setup (same structured mesh as PyCutFEM)
    # ------------------------------------------------------------
    ng_netgen = NetgenMesh(dim=2)
    dom = ng_netgen.AddRegion("dom", 2)
    ng_netgen.Add(FaceDescriptor(surfnr=1, domin=dom, domout=0, bc=0))

    node_xy = np.asarray([(float(n.x), float(n.y)) for n in nodes], dtype=float)
    point_ids = [ng_netgen.Add(MeshPoint(Pnt(float(x0), float(y0), 0.0))) for (x0, y0) in node_xy]
    for conn in np.asarray(elems, dtype=int):
        ids = conn.tolist()
        # PyCutFEM structured quads are ordered [bl, br, tl, tr]; Netgen expects CCW [bl, br, tr, tl].
        if len(ids) == 4:
            ids = [ids[0], ids[1], ids[3], ids[2]]
        verts = [point_ids[int(i)] for i in ids]
        ng_netgen.Add(Element2D(1, verts))

    # Add boundary segments so the Netgen mesh is well-formed for downstream XFEM routines.
    xmin, xmax = -L / 2.0, L / 2.0
    ymin, ymax = -H / 2.0, H / 2.0
    tol = 1.0e-12
    for n0, n1 in np.asarray(edges, dtype=int):
        x0, y0 = node_xy[int(n0)]
        x1, y1 = node_xy[int(n1)]
        bc = None
        if abs(x0 - xmin) < tol and abs(x1 - xmin) < tol:
            bc = 1  # left
        elif abs(x0 - xmax) < tol and abs(x1 - xmax) < tol:
            bc = 2  # right
        elif abs(y0 - ymin) < tol and abs(y1 - ymin) < tol:
            bc = 3  # bottom
        elif abs(y0 - ymax) < tol and abs(y1 - ymax) < tol:
            bc = 4  # top
        if bc is None:
            continue
        ng_netgen.Add(Element1D([point_ids[int(n0)], point_ids[int(n1)]], index=int(bc)))
    ng_netgen.Update()

    ng_mesh = Mesh(ng_netgen)

    levelset_cf = (x * x + y * y) ** 0.5 - R
    lsetp1 = GridFunction(H1(ng_mesh, order=1))
    InterpolateToP1(levelset_cf, lsetp1)
    ci = CutInfo(ng_mesh, lsetp1)

    ba_facets = [
        GetFacetsWithNeighborTypes(ng_mesh, a=ci.GetElementsOfType(HASNEG), b=ci.GetElementsOfType(IF)),
        GetFacetsWithNeighborTypes(ng_mesh, a=ci.GetElementsOfType(HASPOS), b=ci.GetElementsOfType(IF)),
    ]
    dw_neg = ng_dFacetPatch(definedonelements=ba_facets[0])
    dw_pos = ng_dFacetPatch(definedonelements=ba_facets[1])
    print(f"[ng] facet sets: neg={ba_facets[0].NumSet()} pos={ba_facets[1].NumSet()}")

    Vhbase = VectorH1(ng_mesh, order=order)
    Vhneg = Compress(Vhbase, GetDofsOfElements(Vhbase, ci.GetElementsOfType(HASNEG)))
    Vhpos = Compress(Vhbase, GetDofsOfElements(Vhbase, ci.GetElementsOfType(HASPOS)))
    Wh = FESpace([Vhneg * Vhpos], dgjumps=True)
    u_pair = Wh.TrialFunction()[0]
    v_pair = Wh.TestFunction()[0]
    gfu = GridFunction(Wh)
    gfv = GridFunction(Wh)

    if args.probe == "analytic":
        u_pos_cf = CoefficientFunction(
            (
                sin(2.0 * math.pi * x) + 0.25 * cos(math.pi * y) + 0.1 * x * y,
                cos(2.0 * math.pi * y) - 0.15 * sin(math.pi * x) + 0.05 * x * x,
            )
        )
        u_neg_cf = CoefficientFunction(
            (
                -0.7 * cos(2.0 * math.pi * x) + 0.2 * y + 0.05 * x * y,
                0.4 * sin(2.0 * math.pi * y) + 0.1 * x - 0.07 * cos(math.pi * y),
            )
        )
    else:
        dx_cf = x - x_split
        u_pos_cf = CoefficientFunction(
            (
                (1.0 + 0.5 * x + 0.25 * y + 0.75 * x * y) + IfPos(dx_cf, 7.0 * dx_cf * (x + 0.5 * y), 0.0),
                (-0.25 + 0.15 * x - 0.6 * y + 0.4 * x * x + 0.2 * y * y) + IfPos(dx_cf, -5.0 * dx_cf * (0.25 * x - y), 0.0),
            )
        )
        u_neg_cf = CoefficientFunction(
            (
                (-0.4 + 0.3 * x - 0.2 * y + 0.1 * x * x - 0.15 * x * y) + IfPos(-dx_cf, 6.0 * dx_cf * (x - 0.25 * y), 0.0),
                (0.6 + 0.25 * x + 0.35 * y - 0.05 * y * y + 0.2 * x * y) + IfPos(-dx_cf, 4.0 * dx_cf * (0.5 * x + y), 0.0),
            )
        )

    gfu.vec[:] = 0.0
    gfv.vec[:] = 0.0
    with TaskManager():
        gfu.components[0].components[0].Set(u_neg_cf)
        gfu.components[0].components[1].Set(u_pos_cf)
        gfv.components[0].components[0].Set(u_neg_cf)
        gfv.components[0].components[1].Set(u_pos_cf)

    a_mass_ng = (
        InnerProduct(u_pair[0] - u_pair[0].Other(), v_pair[0] - v_pair[0].Other()) * dw_neg
        + InnerProduct(u_pair[1] - u_pair[1].Other(), v_pair[1] - v_pair[1].Other()) * dw_pos
    )
    a_diff_ng = (
        InnerProduct(Grad(u_pair[0]) - Grad(u_pair[0].Other()), Grad(v_pair[0]) - Grad(v_pair[0].Other())) * dw_neg
        + InnerProduct(Grad(u_pair[1]) - Grad(u_pair[1].Other()), Grad(v_pair[1]) - Grad(v_pair[1].Other())) * dw_pos
    )

    # ------------------------------------------------------------
    # Compare energies (+ patch volumes for normalization/debug)
    # ------------------------------------------------------------
    E_mass_pc = assemble_and_energy_pc(a_mass_pc, dh, u_vec, v_vec, backend=backend)
    E_diff_pc = assemble_and_energy_pc(a_diff_pc, dh, u_vec, v_vec, backend=backend)

    bf_mass = BilinearForm(Wh)
    bf_mass += a_mass_ng
    E_mass_ng = assemble_and_energy_ng(bf_mass, gfu, gfv)

    bf_diff = BilinearForm(Wh)
    bf_diff += a_diff_ng
    E_diff_ng = assemble_and_energy_ng(bf_diff, gfu, gfv)

    atol = float(args.atol)
    rtol = float(args.rtol)

    def _report(label: str, a: float, b: float) -> None:
        abs_err, rel_err = _err_abs_rel(a, b)
        ok = abs_err <= (atol + rtol * max(abs(a), abs(b)))
        mark = "OK" if ok else "FAIL"
        print(f"{label:24s} PC={a:+.12e}  NG={b:+.12e}  abs={abs_err:.3e} rel={rel_err:.3e}  {mark}")

    print("\n--- dFacetPatch comparison (PyCutFEM vs NGSolve) ---")
    print(f"backend={backend} order={order} q={q}  (rtol={rtol:g} atol={atol:g})")
    _report("patch_mass_energy", E_mass_pc, E_mass_ng)
    _report("patch_diff_energy", E_diff_pc, E_diff_ng)


if __name__ == "__main__":
    main()
