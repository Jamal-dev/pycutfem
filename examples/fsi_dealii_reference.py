#!/usr/bin/env python
"""
Deal.II FSI benchmark (Turek–Hron) reproduced in pycutfem.

- Geometry and mesh: use the reference `fsi.inp` UCD mesh from the deal.II
  code (channel 2.5 x 0.41 with a rigid cylinder and an attached beam).
- Formulation: conforming ALE Navier–Stokes + compressible Neo-Hookean solid,
  monolithic theta-scheme as in `fsi_ale_conforming.py`.
- Parameters: by default mirrors `step-fsi.prm` (FSI-1 / Re≈20). Use
  `--turek-case fsi2` (Re≈100) to obtain the oscillatory benchmark.

The goal is to mirror the established deal.II example and verify that the
pycutfem formulation converges on the same data set.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pycutfem.core.topology import Node
from pycutfem.core.mesh import Mesh
from pycutfem.utils.bitset import BitSet
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    Identity,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    dot,
    div,
    grad,
    inner,
    det,
    inv,
    trace,
    cof, # cofactor
    FacetNormal,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dx, dS
from pycutfem.solvers.nonlinear_solver import (
    NewtonParameters,
    NewtonSolver,
    PetscSnesNewtonSolver,
    LinearSolverParameters,
    TimeStepperParameters,
)
from pycutfem.utils.gmsh_loader import mesh_from_gmsh
from pycutfem.io.visualization import plot_mesh_2

# ----------------------------------------------------------------------------- 
# Geometry helpers (Turek–Hron benchmark)
# -----------------------------------------------------------------------------
H = 0.41
L = 2.5
RADIUS = 0.05
CENTER = (0.2, 0.2)
BEAM_LENGTH = 0.35
BEAM_HEIGHT = 0.02
BEAM_X0 = CENTER[0] + RADIUS
BEAM_Y0 = CENTER[1] - 0.5 * BEAM_HEIGHT
BEAM_Y1 = CENTER[1] + 0.5 * BEAM_HEIGHT


def load_ucd_mesh(path: Path, poly_order: int = 1) -> Tuple[Mesh, BitSet, BitSet]:
    """
    Minimal UCD reader for the deal.II `fsi.inp` mesh.
    Returns the Mesh plus BitSets for fluid and solid elements.
    """
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().split()
        if len(header) < 2:
            raise RuntimeError(f"Unexpected UCD header in {path}")
        n_nodes, n_cells = int(header[0]), int(header[1])

        nodes: List[Node] = []
        node_id_map: Dict[int, int] = {}
        for _ in range(n_nodes):
            nid, xs, ys, *_ = f.readline().split()
            old_id = int(nid)
            new_id = len(nodes)
            node_id_map[old_id] = new_id
            nodes.append(Node(new_id, float(xs), float(ys)))

        coords = np.array([[n.x, n.y] for n in nodes], dtype=float)

        def _signed_area_quad(node_ids: List[int]) -> float:
            pts = coords[np.asarray(node_ids, dtype=int)]
            x = pts[:, 0]
            y = pts[:, 1]
            return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

        def _order_corners_perimeter_ccw(node_ids: List[int]) -> List[int]:
            """
            Return the 4 corner node ids in a CCW perimeter order starting from the
            geometrically bottom-left corner (min x, then min y).

            This yields a stable corner ordering for Mesh corner_connectivity, while
            allowing element_connectivity to use the row-major lattice ordering
            required by higher-order routines.
            """
            pts = coords[np.asarray(node_ids, dtype=int)]
            centroid = pts.mean(axis=0)
            angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
            order = np.argsort(angles)
            perim = [node_ids[int(i)] for i in order]
            if _signed_area_quad(perim) < 0.0:
                perim = list(reversed(perim))
            pts_perim = coords[np.asarray(perim, dtype=int)]
            start = int(np.lexsort((pts_perim[:, 1], pts_perim[:, 0]))[0])
            return perim[start:] + perim[:start]

        elem_conn: List[List[int]] = []
        corner_conn: List[List[int]] = []
        elem_tags: List[str] = []
        boundary_segments: List[Tuple[int, int, int]] = []
        for _ in range(n_cells):
            parts = f.readline().split()
            if len(parts) < 3:
                continue
            _, mat_id, cell_type, *conn = parts
            ctype = cell_type.lower()
            if ctype == "quad":
                conn_int = [node_id_map[int(n)] for n in conn]
                corn = conn_int[:4]
                # Mesh expects:
                # - corner_connectivity: perimeter order (BL, BR, TR, TL) up to rotation
                # - element_connectivity: row-major lattice order (BL, BR, TL, TR) for FE_Q
                perim = _order_corners_perimeter_ccw(corn)
                lattice = [perim[0], perim[1], perim[3], perim[2]]
                elem_conn.append(lattice)
                corner_conn.append(perim)
                elem_tags.append("solid" if int(mat_id) == 1 else "fluid")
            elif ctype == "line" and len(conn) >= 2:
                n1, n2 = (node_id_map[int(conn[0])], node_id_map[int(conn[1])])
                boundary_segments.append((int(mat_id), int(n1), int(n2)))
            else:
                continue

    mesh = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(elem_conn, dtype=int),
        elements_corner_nodes=np.asarray(corner_conn, dtype=int),
        element_type="quad",
        poly_order=poly_order,
    )

    # Cache element bitsets
    fluid_mask = np.fromiter((tag == "fluid" for tag in elem_tags), bool)
    solid_mask = ~fluid_mask
    for el, tag in zip(mesh.elements_list, elem_tags):
        el.tag = tag
    mesh._elem_bitsets = {
        "fluid": BitSet(fluid_mask),
        "solid": BitSet(solid_mask),
    }

    # Transfer boundary ids from UCD "line" elements onto the matching Mesh edges.
    # deal.II boundary ids: 0=inlet, 1=outlet, 2=walls, 80=cylinder, 81=beam_root arc.
    ucd_id_to_tag = {
        0: "inlet",
        1: "outlet",
        2: "walls",
        80: "cylinder",
        81: "beam_root",
    }
    edge_by_key = {tuple(sorted(map(int, e.nodes))): e for e in mesh.edges_list}
    hit = miss = 0
    for mat_id, n1, n2 in boundary_segments:
        tag = ucd_id_to_tag.get(int(mat_id))
        if not tag:
            continue
        key = (min(int(n1), int(n2)), max(int(n1), int(n2)))
        edge = edge_by_key.get(key)
        if edge is None:
            miss += 1
            continue
        edge.tag = tag
        hit += 1
    if miss:
        print(f"[warn] UCD boundary tagging: {miss} line segments did not match any mesh edge.")
    if hit:
        mesh.rebuild_edge_bitsets()
    return mesh, mesh.element_bitset("fluid"), mesh.element_bitset("solid")


def _adaptive_locators(mesh: Mesh, tol: float):
    """
    Build locator functions for inlet/outlet/walls using actual boundary extents,
    plus geometric tests for cylinder/beam.
    """
    cx, cy = CENTER
    r2 = RADIUS * RADIUS
    beam_x1 = BEAM_X0 + BEAM_LENGTH

    coords = np.asarray(getattr(mesh, "nodes_x_y_pos", []), float)
    xmin_raw, ymin_raw, xmax_raw, ymax_raw = 0.0, 0.0, L, H
    if coords.size:
        xmin_raw, ymin_raw = coords.min(axis=0)
        xmax_raw, ymax_raw = coords.max(axis=0)

    mids = []
    for e in mesh.edges_list:
        if e.right is None:
            mids.append(mesh.nodes_x_y_pos[list(e.nodes)].mean(axis=0))
    if mids:
        mids = np.asarray(mids, float)
        xmin_raw = min(xmin_raw, float(mids[:, 0].min()))
        ymin_raw = min(ymin_raw, float(mids[:, 1].min()))
        xmax_raw = max(xmax_raw, float(mids[:, 0].max()))
        ymax_raw = max(ymax_raw, float(mids[:, 1].max()))

    xmin = max(0.0, xmin_raw)
    xmax = min(L, xmax_raw)
    ymin = max(0.0, ymin_raw)
    ymax = min(H, ymax_raw)
    span = float(max(xmax - xmin, ymax - ymin, 1.0))
    tol_loc = max(tol, 1e-4 * span)
    tol_cyl = max(tol_loc, 2e-3 * span)       # allow slight curvature drift
    tol_root = max(tol_loc, 4.5e-3 * span)      # beam root points sit on the curved cylinder/beam junction
    tol_y = max(tol_loc, 8e-4 * span)
    tol_x_root = max(tol_loc, 1.5e-3 * span)

    def on_cylinder(x: float, y: float) -> bool:
        return abs((x - cx) ** 2 + (y - cy) ** 2 - r2) < tol_cyl

    def on_beam_outer(x: float, y: float) -> bool:
        on_x0 = abs(x - BEAM_X0) < tol_loc and BEAM_Y0 - tol_loc <= y <= BEAM_Y1 + tol_loc
        on_x1 = abs(x - beam_x1) < tol_loc and BEAM_Y0 - tol_loc <= y <= BEAM_Y1 + tol_loc
        on_y0 = abs(y - BEAM_Y0) < tol_loc and BEAM_X0 - tol_loc <= x <= beam_x1 + tol_loc
        on_y1 = abs(y - BEAM_Y1) < tol_loc and BEAM_X0 - tol_loc <= x <= beam_x1 + tol_loc
        return on_x0 or on_x1 or on_y0 or on_y1

    def on_beam_root(x: float, y: float) -> bool:
        # Straight portion plus small curved interface near the cylinder/beam junction.
        on_vertical = abs(x - BEAM_X0) < tol_x_root and (BEAM_Y0 - tol_y) <= y <= (BEAM_Y1 + tol_y)
        on_arc = (
            abs((x - cx) ** 2 + (y - cy) ** 2 - r2) < tol_root
            and abs(x - BEAM_X0) < tol_x_root
            and (BEAM_Y0 - tol_y) <= y <= (BEAM_Y1 + tol_y)
            and x >= cx
        )
        return on_vertical or on_arc

    return {
        "inlet": lambda x, y: abs(x - xmin) < tol_loc,
        "outlet": lambda x, y: abs(x - xmax) < tol_loc,
        "walls": lambda x, y: abs(y - ymin) < tol_loc or abs(y - ymax) < tol_loc,
        "cylinder": on_cylinder,
        "beam_outer": on_beam_outer,
        "beam_root": on_beam_root,
        # Combined locator for cylinder + full beam perimeter
        "obstacle_set": lambda x, y: on_cylinder(x, y) or on_beam_outer(x, y),
    }


def retag_boundaries(mesh: Mesh, tol: float = 1.0e-8, *, overwrite: bool = True) -> None:
    """
    Geometric tagging of boundary edges to mirror the deal.II boundary ids on the
    canonical channel (x in [0,L], y in [0,H]).
    - cylinder (circle at CENTER, radius RADIUS)
    - beam_outer (outer beam box except the circular interface)
    - beam_root (left edge of the beam)
    """
    locators = _adaptive_locators(mesh, tol)
    try:
        mesh.tag_boundary_edges(locators, overwrite=overwrite)
    except TypeError:
        mesh.tag_boundary_edges(locators)
    # Drop any boundary tags whose midpoints fail their own locator (guards against padded meshes)
    for e in mesh.edges_list:
        if e.right is not None:
            continue
        tag = getattr(e, "tag", "") or ""
        if not tag:
            continue
        if tag not in locators:
            continue
        mpx, mpy = mesh.nodes_x_y_pos[list(e.nodes)].mean(axis=0)
        if not locators[tag](mpx, mpy):
            e.tag = ""
    mesh.rebuild_edge_bitsets()


def tag_nodes_from_edges(mesh: Mesh) -> Dict[str, int]:
    """
    Populate Node.tag with a comma-separated list of incident boundary tags.
    Returns a count per tag for quick diagnostics.
    """
    tag_to_nodes: Dict[str, set[int]] = {}
    for edge in mesh.edges_list:
        if edge.right is not None:
            continue
        tag = getattr(edge, "tag", "") or ""
        if not tag:
            continue
        nodes = edge.all_nodes if edge.all_nodes else edge.nodes
        tag_to_nodes.setdefault(tag, set()).update(int(n) for n in nodes)

    for nid in range(len(mesh.nodes_list)):
        mesh.nodes_list[nid].tag = ""
    for tag, nodes in tag_to_nodes.items():
        for nid in nodes:
            node = mesh.nodes_list[int(nid)]
            pieces = set(node.tag.split(",")) if node.tag else set()
            pieces.add(tag)
            node.tag = ",".join(sorted(p for p in pieces if p))

    return {tag: len(nodes) for tag, nodes in tag_to_nodes.items()}


def classify_fluid_solid(mesh: Mesh, tol: float = 1.0e-9) -> Tuple[BitSet, BitSet]:
    """
    Ensure the mesh has fluid/solid element bitsets. If tags already exist, reuse them;
    otherwise classify geometrically (beam box minus the circular hole is 'solid').
    """
    cached = getattr(mesh, "_elem_bitsets", {})
    if "fluid" in cached and "solid" in cached:
        return mesh.element_bitset("fluid"), mesh.element_bitset("solid")

    beam_x0 = CENTER[0] + RADIUS
    beam_x1 = beam_x0 + BEAM_LENGTH
    beam_y0 = CENTER[1] - 0.5 * BEAM_HEIGHT
    beam_y1 = CENTER[1] + 0.5 * BEAM_HEIGHT
    rad_tol = RADIUS + tol

    coords = mesh.nodes_x_y_pos[mesh.corner_connectivity].mean(axis=1).T
    cx, cy = coords[0], coords[1]
    inside_mask = (
        (cx >= beam_x0 - tol)
        & (cx <= beam_x1 + tol)
        & (cy >= beam_y0 - tol)
        & (cy <= beam_y1 + tol)
        & (np.hypot(cx - CENTER[0], cy - CENTER[1]) >= rad_tol)
    )
    tags = np.where(inside_mask, "solid", "fluid")
    for el, tag in zip(mesh.elements_list, tags):
        el.tag = str(tag)
    mesh._elem_bitsets = {
        "fluid": BitSet(tags == "fluid"),
        "solid": BitSet(tags == "solid"),
    }
    return mesh._elem_bitsets["fluid"], mesh._elem_bitsets["solid"]


def ensure_boundary_tags(mesh: Mesh, tol: float = 1.0e-6, *, force_geometric: bool = True, mesh_size: float | None = None) -> Dict[str, Tuple[int, int]]:
    """
    Ensure standard boundary tags exist. By default, re-tag geometrically to
    match the reference Turek–Hron channel (x∈[0,L], y∈[0,H]) regardless of
    what the .msh provides. Returns counts before/after.
    """
    tags = ("inlet", "outlet", "walls", "cylinder", "beam_outer", "beam_root")
    counts_before = {t: mesh.edge_bitset(t).cardinality() for t in tags}

    did_geometric = False
    if force_geometric:
        # Always enforce canonical tags; overwrite existing ones.
        tol_eff = tol
        if mesh_size is not None:
            tol_eff = max(tol, 1e-4 * mesh_size)
        retag_boundaries(mesh, tol=tol_eff, overwrite=True)
        did_geometric = True
    else:
        need_retag = counts_before.get("outlet", 0) == 0 or counts_before.get("inlet", 0) == 0
        if need_retag:
            retag_boundaries(mesh, tol=tol, overwrite=False)
            did_geometric = True

    # Only prune when tags were produced geometrically; mesh-provided tags (UCD/Gmsh)
    # should be trusted and may not match our simplistic locators.
    if did_geometric:
        locators = _adaptive_locators(mesh, tol)
        for e in mesh.edges_list:
            if e.right is not None:
                continue
            tag = getattr(e, "tag", "") or ""
            if tag and tag in locators:
                mpx, mpy = mesh.nodes_x_y_pos[list(e.nodes)].mean(axis=0)
                if not locators[tag](mpx, mpy):
                    e.tag = ""
        mesh.rebuild_edge_bitsets()

    counts_after = {t: mesh.edge_bitset(t).cardinality() for t in tags}

    # If outlet is still empty, force-tag boundary edges near x=L.
    if counts_after.get("outlet", 0) == 0:
        for edge in mesh.edges_list:
            if edge.right is not None:
                continue
            mpx, mpy = mesh.nodes_x_y_pos[list(edge.nodes)].mean(axis=0)
            if abs(mpx - L) < max(tol, 1e-6) and not getattr(edge, "tag", None):
                edge.tag = "outlet"
        mesh.rebuild_edge_bitsets()
        counts_after["outlet"] = int(mesh.edge_bitset("outlet").cardinality())

    return {t: (counts_before.get(t, 0), counts_after.get(t, 0)) for t in tags}


def tag_fluid_solid_interface_edges(mesh: Mesh, *, tag: str = "beam_outer") -> int:
    """
    Tag conforming fluid–solid interface edges (interior edges where element tags differ).

    For drag/lift, we integrate the *fluid* traction and need the outward normal from
    the fluid cell. Enforce `edge.left` to be the fluid owner for these edges.
    """
    count = 0
    for e in mesh.edges_list:
        if e.left is None or e.right is None:
            continue
        left_tag = str(getattr(mesh.elements_list[int(e.left)], "tag", "") or "")
        right_tag = str(getattr(mesh.elements_list[int(e.right)], "tag", "") or "")
        if {left_tag, right_tag} != {"fluid", "solid"}:
            continue
        if left_tag != "fluid":
            e.left, e.right = e.right, e.left
            e.lid, e.right_lid = e.right_lid, e.lid
            e.left_nodes, e.right_nodes = e.right_nodes, e.left_nodes
            if getattr(e, "normal", None) is not None:
                e.normal = -np.asarray(e.normal, dtype=float)
        e.tag = tag
        count += 1
    mesh.rebuild_edge_bitsets()
    return count


def ensure_obstacle_set(mesh: Mesh, *, name: str = "obstacle_set") -> BitSet:
    """
    Ensure a combined obstacle edge BitSet exists even though edges carry only one tag.
    """
    obstacle = None
    for t in ("cylinder", "beam_outer", "beam_root"):
        bs = mesh.edge_bitset(t)
        if bs is None or bs.cardinality() == 0:
            continue
        obstacle = bs if obstacle is None else (obstacle | bs)
    if obstacle is None:
        obstacle = BitSet(np.zeros(len(mesh.edges_list), bool))
    if not hasattr(mesh, "_edge_bitsets"):
        mesh._edge_bitsets = {}
    mesh._edge_bitsets[name] = obstacle
    return obstacle


def boundary_nodes_by_tag(mesh: Mesh) -> Dict[str, set[int]]:
    """
    Collect boundary nodes for each edge tag, using all_nodes to capture
    higher-order edge nodes.
    """
    out: Dict[str, set[int]] = {}
    for e in mesh.edges_list:
        if e.right is not None:
            continue
        tag = getattr(e, "tag", "") or ""
        if not tag:
            continue
        nodes = e.all_nodes if e.all_nodes else e.nodes
        out.setdefault(tag, set()).update(int(n) for n in nodes)
    return out


def find_hanging_edges(mesh: Mesh, tol: float = 1.0e-8) -> List[Tuple[int, float, float]]:
    """
    Detect single-owner edges whose midpoints do not belong to any canonical
    boundary locator (inlet, outlet, walls, cylinder, beam box). These edges
    indicate hanging nodes / non-conforming interfaces.
    Returns a list of (edge_id, mid_x, mid_y).
    """
    locators = _adaptive_locators(mesh, tol)
    coords = mesh.nodes_x_y_pos
    hanging: List[Tuple[int, float, float]] = []
    for edge in mesh.edges_list:
        if (edge.left is None) or (edge.right is None):
            mid = coords[list(edge.nodes)].mean(axis=0)
            if not any(loc(float(mid[0]), float(mid[1])) for loc in locators.values()):
                hanging.append((int(edge.gid), float(mid[0]), float(mid[1])))
    return hanging


def symgrad(u):
    return 0.5 * (grad(u) + grad(u).T)
def transpose(A):
    return A.T  
def _is_zero(expr) -> bool:
    """
    Cheap zero check for scalars/vectors/matrices represented as Constant or numbers.
    Avoids triggering tensor algebra on obvious zeros when linearizing terms.
    """
    if isinstance(expr, Constant):
        arr = np.asarray(expr.value)
        return np.allclose(arr, 0.0)
    if isinstance(expr, (int, float, np.floating)):
        return abs(expr) < 1.0e-14
    return False
class ALE_Helpers:
    """
    ALE kinematic helpers, mirroring the C++ ALE_Transformations namespace.
    All tensor products use `dot` (last index of first tensor with
    first index of second tensor). Cofactors, det, inv as in pycutfem UFL.
    """
    @staticmethod
    def get_F(grad_d):
        return Identity(2) + grad_d

    @staticmethod
    def get_J(F):
        return det(F)

    @staticmethod
    def get_F_inv(F):
        return inv(F)

    @staticmethod
    def get_cof_F(F):
        return cof(F)  # J * F^{-T}

    @staticmethod
    def get_J_LinU(F, grad_dd):
        r"""
        Linearization of J with respect to displacement d.

        δJ = cof(F) : ∇δd = inner(cof(F), grad(δd)).
        """
        cof_F = cof(F)
        return inner(cof_F, grad_dd)


    @staticmethod
    def get_F_inv_LinU(F_inv, grad_dd):
        r"""
        Linearization of F^{-1} with respect to d.

        δ(F^{-1}) = - F^{-1} (δF) F^{-1},  δF = ∇δd.
        """
        return -dot(F_inv, dot(grad_dd, F_inv))

    @staticmethod
    def get_cof_F_LinU(F, F_inv, grad_dd):
        r"""
        Linearization of cof(F) = J F^{-T} with respect to d.

        δ(cof(F)) = δ(J F^{-T}) = δJ F^{-T} + J δ(F^{-T}),
        δJ  = cof(F) : ∇δd,
        δF^{-T} = (δF^{-1})^T = ( -F^{-1} δF F^{-1} )^T.
        """
        J = det(F)
        cof_F = cof(F)

        J_LinU = inner(cof_F, grad_dd)
        F_inv_LinU = -dot(F_inv, dot(grad_dd, F_inv))
        dF_inv_T = F_inv_LinU.T

        return J_LinU * F_inv.T + J * dF_inv_T
class NSE_ALE:
    """
    Fluid NSE terms in ALE formulation.

    All tensor products use `dot` (matrix multiplication / tensor contraction),
    and `inner` is the Frobenius product (double contraction).
    """
    @staticmethod
    def get_stress_fluid_ALE(mu_f, p, grad_v, F_inv):
        r"""
        Cauchy stress in ALE:

        σ = -p I + μ (∇v F^{-1} + F^{-T} ∇v^T).
        """
        I = Identity(2)
        grad_v_T = grad_v.T
        F_inv_T = F_inv.T

        sigma_visc = mu_f * (dot(grad_v, F_inv) + dot(F_inv_T, grad_v_T))
        return -p * I + sigma_visc
    @staticmethod
    def get_stress_fluid_ALE_direct(mu_f, pI, grad_v, F_inv, grad_v_T, F_inv_T):
        r"""
        Cauchy stress in ALE:

        σ = -p I + μ (∇v F^{-1} + F^{-T} ∇v^T).
        """
        sigma_visc = mu_f * (dot(grad_v, F_inv) + dot(F_inv_T, grad_v_T))
        return -pI + sigma_visc

    @staticmethod
    def get_stress_fluid_except_pressure_ALE(mu_f, grad_v, F_inv):
        grad_v_T = grad_v.T
        F_inv_T = F_inv.T
        return  mu_f * (dot(grad_v, F_inv) + dot(F_inv_T, grad_v_T))

    # ------------------------------------------------------------------ #
    # Linearization of stress                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_stress_fluid_ALE_1st_term_LinAll_short(
        pI,
        F_inv_T,
        J_F_inv_T_LinU_trial,
        pI_LinP,
        J):
        return (-J * dot(pI_LinP , F_inv_T) - dot(pI, J_F_inv_T_LinU_trial)) 
    
    @staticmethod
    def get_stress_fluid_ALE_2nd_term_LinAll_short(
        J_F_inv_T_LinU,
        stress_fluid_ALE,
        grad_v,
        grad_v_LinV,
        F_inv,
        F_inv_LinU,
        J,
        mu_f,
    ):
        r"""
        C++: get_stress_fluid_ALE_2nd_term_LinAll_short

        Returns (without density factor):

        μ [ J (σ_LinV + σ_LinU) F^{-T} + σ J_F^{-T}_LinU ],

        where

        σ_LinV = ∇(δv) F^{-1} + F^{-T} ∇(δv)^T,
        σ_LinU = ∇v δF^{-1} + (δF^{-1})^T ∇v^T.
        """
        F_inv_T = F_inv.T

        sigma_terms = []

        # σ_LinV
        if not _is_zero(grad_v_LinV):
            sigma_terms.append(
                dot(grad_v_LinV, F_inv) + dot(F_inv_T, grad_v_LinV.T)
            )

        # σ_LinU
        if not _is_zero(F_inv_LinU):
            sigma_terms.append(
                dot(grad_v, F_inv_LinU) + dot(F_inv_LinU.T, grad_v.T)
            )

        pieces = []

        if sigma_terms:
            sigma_sum = sigma_terms[0]
            for term in sigma_terms[1:]:
                sigma_sum = sigma_sum + term
            # J (σ_LinV + σ_LinU) F^{-T}
            pieces.append(J * dot(sigma_sum, F_inv_T))

        # σ J_F^{-T}_LinU
        if not _is_zero(J_F_inv_T_LinU):
            pieces.append(dot(stress_fluid_ALE, J_F_inv_T_LinU))

        if not pieces:
            return Constant(0.0)

        total = pieces[0]
        for term in pieces[1:]:
            total = total + term

        return mu_f * total

    @staticmethod
    def get_stress_fluid_ALE_3rd_term_LinAll_short(
        F_inv,
        F_inv_LinU_trial,
        grad_v,
        grad_v_LinV_trial,
        mu_f,
        J,
        J_F_inv_T_LinU_trial,
    ):
        r"""
        C++: get_stress_fluid_ALE_3rd_term_LinAll_short

        Returns (without density factor):

        μ [ J_F^{-T}_LinU ∇v^T F^{-T}
          + J F^{-T} ∇(δv)^T F^{-T}
          + J F^{-T} ∇v^T (δF^{-1})^T ].
        """
        F_inv_T = F_inv.T

        term1 = dot(dot(J_F_inv_T_LinU_trial, grad_v.T), F_inv_T)
        term2 = J * dot(dot(F_inv_T, grad_v_LinV_trial.T), F_inv_T)
        term3 = J * dot(dot(F_inv_T, grad_v.T), F_inv_LinU_trial.T)

        return mu_f * (term1 + term2 + term3)
    # ------------------------------------------------------------------ #
    # Incompressibility                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_Incompressibility_ALE(v, F):
        r"""
        g = J F^{-1} : ∇v = cof(F) : ∇v.

        This compact expression is algebraically equivalent to the explicit
        polynomial used in the C++ code.
        """
        return inner(cof(F), grad(v))
    @staticmethod
    def get_Incompressibility_ALE_LinV_optimized(grad_v, grad_v_trial, F, grad_dd_trial):
        r"""
        2D Optimized Linearization of g = cof(F) : ∇v
        
        This bypasses calculating determinants and inverses for the linearization
        of the geometry term, valid ONLY for 2D.
        """
        cof_F = cof(F)
        term_v = inner(cof_F, grad_v_trial)
        delta_cof_F = cof(grad_dd_trial)
        term_geom = inner(delta_cof_F, grad_v)
        return term_v + term_geom
    @staticmethod
    def get_Incompressibility_ALE_LinAll(v, v_trial, F, F_inv, grad_dd):
        r"""
        Linearization of g = cof(F) : ∇v w.r.t. both v and d.

        Using cof(F) = J F^{-T}:

        δg = cof(F) : ∇(δv) + δ(cof(F)) : ∇v
            = cof(F) : ∇(δv) + [δJ F^{-T} + J δF^{-T}] : ∇v.

        We implement this in terms of F, F^{-1}, and δF = ∇δd.
        """
        cof_F = cof(F)

        # First part: cof(F) : ∇(δv)
        term_v = inner(cof_F, grad(v_trial))

        # δJ = cof(F) : δF
        delta_J = inner(cof_F, grad_dd)

        # δF^{-1} = -F^{-1} δF F^{-1}
        F_inv_LinU = -dot(F_inv, dot(grad_dd, F_inv))
        delta_F_inv_T = F_inv_LinU.T

        J = det(F)
        F_inv_T = F_inv.T

        # δcof(F) = δ(J F^{-T}) = δJ F^{-T} + J δF^{-T}
        delta_cof_F = delta_J * F_inv_T + J * delta_F_inv_T

        term_geom = inner(delta_cof_F, grad(v))

        return term_v + term_geom
    
    # ------------------------------------------------------------------ #
    # Convection                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_Convection_LinAll_short(
        phi_grad_v_trial,
        phi_v_trial,
        J,
        J_LinU,
        F_inv,
        F_inv_LinU,
        v,
        grad_v,
        density,
    ):
        r"""
        C++: get_Convection_LinAll_short

        For c = ρ J ∇v F^{-1} v we have

        δc = ρ [ δJ ∇v F^{-1} v
                + J ∇v δF^{-1} v
                + J ∇(δv) F^{-1} v
                + J ∇v F^{-1} δv ].
        """
        grad_v_Finv = dot(grad_v, F_inv)

        conv_LinU = None
        if not _is_zero(J_LinU):
            if conv_LinU is None:
                conv_LinU = J_LinU * dot(grad_v_Finv, v)
            else:
                conv_LinU += J_LinU * dot(grad_v_Finv, v)
        if not _is_zero(F_inv_LinU):
            if conv_LinU is None:
                conv_LinU = J * dot(dot(grad_v, F_inv_LinU), v)
            else:
                conv_LinU += J * dot(dot(grad_v, F_inv_LinU), v)

        conv_LinV = None
        if not _is_zero(phi_grad_v_trial):
            if conv_LinV is None:
                conv_LinV = J * dot(dot(phi_grad_v_trial, F_inv), v)
            else:
                conv_LinV += J * dot(dot(phi_grad_v_trial, F_inv), v)
        if not _is_zero(phi_v_trial):
            if conv_LinV is None:
                conv_LinV = J * dot(grad_v_Finv, phi_v_trial)
            else:
                conv_LinV += J * dot(grad_v_Finv, phi_v_trial)

        if conv_LinU is None and conv_LinV is not None:
            return density * conv_LinV
        elif conv_LinV is None and conv_LinU is not None:
            return density * conv_LinU
        elif conv_LinU is not None and conv_LinV is not None:
            return density * (conv_LinU + conv_LinV)
        else:
            return Constant(0.0)
            

    @staticmethod
    def get_Convection_u_LinAll_short(phi_grad_v_trial, phi_u_disp_trial, J, J_LinU,
                                      F_inv, F_inv_LinU, u_disp, grad_v,
                                      density):
        """
        Same as above, but with u instead of v.
        """
        grad_v_Finv = dot(grad_v, F_inv)
        grad_v_Finv_LinU = dot(grad_v, F_inv_LinU)
        conv_LinU = (J_LinU * dot(grad_v_Finv, u_disp)
                        + J * dot(grad_v_Finv_LinU, u_disp)
                        + J * dot(grad_v_Finv, phi_u_disp_trial)
            )
        conv_LinV = J * dot(phi_grad_v_trial, dot(F_inv, u_disp))
        return density * (conv_LinU + conv_LinV)

    @staticmethod
    def get_Convection_u_old_LinAll_short(phi_grad_v_trial, J, J_LinU,
                                          F_inv, F_inv_LinU,
                                          u_old_disp, grad_v, density):
        """
        Same structure, using old timestep quantities.
        """
        grad_v_Finv = dot(grad_v, F_inv)
        grad_v_Finv_LinU = dot(grad_v, F_inv_LinU)
        conv_LinU = (J_LinU * dot(grad_v_Finv, u_old_disp)
                     + J * dot(grad_v_Finv_LinU, u_old_disp)
            )
        
        F_inv_u_old = dot(F_inv, u_old_disp)
        conv_LinV = J * dot(phi_grad_v_trial, F_inv_u_old)
        return density * (conv_LinU + conv_LinV)

    # ------------------------------------------------------------------ #
    # Acceleration                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_acceleration_term_LinAll(J, J_old, J_LinU_trial,
                                     v, v_old, v_trial, density):
        r"""
        C++ idea:

        ρ/2 [ (J + J_old) (v - v_old) ].

        Linearized:

        δa = ρ/2 [ J_LinU (v - v_old) + (J + J_old) δv ].
        """
        term_geom = (J_LinU_trial * (v - v_old)) 
        term_vel  = (J + J_old) * v_trial 
        return 0.5 * density * (term_geom + term_vel)





class Structure_Terms:
    """
    STVK material terms.
    """
    @staticmethod
    def get_C(F):
        return dot(F.T, F)
    @staticmethod
    def get_E(F):
        I = Identity(2)
        C = Structure_Terms.get_C(F)
        return 0.5 * (C - I)

    @staticmethod
    def get_S(E, mu_s, lambda_s):
        r"""
        Second Piola–Kirchhoff stress (STVK):

        S = λ tr(E) I + 2 μ E.
        """
        I = Identity(2)
        trE = trace(E)
        return lambda_s * trE * I + 2.0 * mu_s * E
    @staticmethod
    def get_Cauchy_stress(F, mu_s, lambda_s):
        r"""
        Cauchy stress (STVK):

        E = 1/2 (F^T F - I),

        S = λ tr(E) I + 2 μ E,

        σ = 1/J F S F^T.
        """
        J = det(F)
        E = Structure_Terms.get_E(F)
        S = Structure_Terms.get_S(E, mu_s, lambda_s)
        return (1.0 / J) * dot(dot(F, S), F.T)
    @staticmethod
    def get_F_LinU(grad_dd):
        r"""
        Linearization of F w.r.t. displacement d.

        F = I + ∇d, δF = ∇δd.
        """
        return grad_dd
    @staticmethod
    def get_E_LinU(F, grad_dd):
        r"""
        Linearization of E w.r.t. displacement d.

        F = I + ∇d, δF = ∇δd.

        E = 1/2 (F^T F - I),
        δE = 1/2 (δF^T F + F^T δF).
        """
        delta_F = grad_dd
        delta_E = 0.5 * (dot(delta_F.T, F) + dot(F.T, delta_F))
        return delta_E
    @staticmethod
    def get_Piola_Kirchhoff_1st_LinAll(F, grad_dd, mu_s, lambda_s):
        r"""
        Linearization of 1st Piola–Kirchhoff stress P w.r.t. displacement d.

        F = I + ∇d, δF = ∇δd.

        E = 1/2 (F^T F - I),
        δE = 1/2 (δF^T F + F^T δF),

        S = λ tr(E) I + 2 μ E,
        δS = λ tr(δE) I + 2 μ δE,

        P = F S,
        δP = δF S + F δS.
        """
        delta_F = grad_dd
        E = Structure_Terms.get_E(F)
        S = Structure_Terms.get_S(E, mu_s, lambda_s)

        delta_E = 0.5 * (dot(delta_F.T, F) + dot(F.T, delta_F))
        tr_delta_E = trace(delta_E)
        delta_S = lambda_s * tr_delta_E * Identity(2) + 2.0 * mu_s * delta_E

        delta_P = dot(delta_F, S) + dot(F, delta_S)
        return delta_P

    @staticmethod
    def get_S_LinU(F, grad_dd, mu_s, lambda_s):
        r"""
        Linearization of S and E w.r.t. displacement d.

        F = I + ∇d, δF = ∇δd.

        E = 1/2 (F^T F - I),
        δE = 1/2 (δF^T F + F^T δF),

        S = λ tr(E) I + 2 μ E,
        δS = λ tr(δE) I + 2 μ δE.
        """
        delta_F = grad_dd
        E = Structure_Terms.get_E(F)

        delta_E = 0.5 * (dot(delta_F.T, F) + dot(F.T, delta_F))
        tr_delta_E = trace(delta_E)

        I = Identity(2)
        delta_S = lambda_s * tr_delta_E * I + 2.0 * mu_s * delta_E

        return delta_S, delta_E


def build_jac(
    *,
    uk, u_prev,       # Fluid Velocity (current, old)
    dk, d_prev,       # Displacement (current, old)
    pk, p_prev,       # Pressure (current, old)
    du, dd, dp,       # Trial functions (δv, δd, δp)
    test_v, test_w, test_q,          # Test functions (ψ_v, ψ_d, ψ_p)
    timestep: Constant,
    theta: Constant,
    rho_f: Constant,
    mu_f: Constant,
    rho_s: Constant,
    lambda_s: Constant,
    mu_s: Constant,
    alpha_u: Constant,
    stab_eps: Constant,
    p_gauge: Constant | None = None,
    fluid_bs,
    solid_bs,
    outlet_bs,
    quad_order: int,
):
    # --- Integration Measures ---
    dx_f = dx(defined_on=fluid_bs, metadata={"q": quad_order})
    dx_s = dx(defined_on=solid_bs, metadata={"q": quad_order})
    dS_outlet = dS(defined_on=outlet_bs, metadata={"q": quad_order})
    n =  FacetNormal()
    
    # --- Geometric State (Current Newton Iteration) ---
    I = Identity(2)
    F = ALE_Helpers.get_F(grad(dk))
    J = ALE_Helpers.get_J(F)
    Finv = ALE_Helpers.get_F_inv(F)
    F_inv_T = Finv.T
    cof_F = ALE_Helpers.get_cof_F(F)  # J * Finv.T
    pI = pk * Identity(2)
    pI_LinP_trial = dp * Identity(2)

    # --- Geometric State (Previous Timestep) ---
    F_old = ALE_Helpers.get_F(grad(d_prev))
    J_old = ALE_Helpers.get_J(F_old)
    
    # --- Geometric Linearization (Shape Derivatives) ---
    # These represent the variation of geometric terms w.r.t displacement trial function (dd)
    grad_dd = grad(dd)
    J_F_inv_T_LinU = cof(grad_dd)  # δ(J F^{-T}) # trial of displacement
    J_LinU = ALE_Helpers.get_J_LinU(F, grad_dd) # trial of displacement
    Finv_LinU = ALE_Helpers.get_F_inv_LinU(Finv, grad_dd) # trial of displacement
    cof_F_LinU = ALE_Helpers.get_cof_F_LinU(F, Finv, grad_dd) # δ(J F^{-T}) # trial of displacement

    # test gradients
    test_grad_v = grad(test_v)
    test_grad_w = grad(test_w)

    # ========================================================================
    # 1. FLUID RESIDUAL (ALE Navier-Stokes)
    # ========================================================================
    
    # --- Velocity & Gradients ---
    grad_uk = grad(uk)
    grad_uk_T = grad_uk.T
    grad_u_prev = grad(u_prev)

    sigma_ALE = NSE_ALE.get_stress_fluid_ALE_direct(
        mu_f, pI, grad_uk, Finv, grad_uk_T, F_inv_T
    )
    # ---  Mass / Acceleration Term ---
    acc_term_jac = NSE_ALE.get_acceleration_term_LinAll(
        J, J_old, J_LinU,
        uk, u_prev, du,
        rho_f
    )  
    # ---  Convection Term ---
    convection_fluid_v = NSE_ALE.get_Convection_LinAll_short(
        grad(du), du, J, J_LinU,
        Finv, Finv_LinU, uk, grad_uk,
        rho_f
    )  
    convection_fluid_d = NSE_ALE.get_Convection_u_LinAll_short(
        grad(du), dd, J, J_LinU,
        Finv, Finv_LinU, dk, grad_uk,
        rho_f
    )
    convection_fluid_u_old = NSE_ALE.get_Convection_u_old_LinAll_short(
        grad(du), J, J_LinU,
        Finv, Finv_LinU, d_prev, grad_uk,
        rho_f
    )
    # ---  Diffusion Term ---
    stress_fluid_term_1 = NSE_ALE.get_stress_fluid_ALE_1st_term_LinAll_short(
        pI, F_inv_T,
        J_F_inv_T_LinU,
        pI_LinP_trial,
        J
    )
    stress_visc_no_mu = dot(grad_uk, Finv) + dot(F_inv_T, grad_uk_T)  # no pressure, no mu

    stress_fluid_term_2 = NSE_ALE.get_stress_fluid_ALE_2nd_term_LinAll_short(
        J_F_inv_T_LinU,
        stress_visc_no_mu,#sigma_ALE,
        grad_uk,
        grad(du),
        Finv, Finv_LinU,
        J,
        mu_f,
    )
    jac_mass_du = dot(acc_term_jac, test_v) 
    jac_convection_du = timestep * theta * dot(convection_fluid_v, test_v)
    jac_convection_du += -dot(convection_fluid_d, test_v)
    jac_convection_du += dot(convection_fluid_u_old, test_v)
    jac_diffusion_du = timestep * inner(stress_fluid_term_1, test_grad_v)
    jac_diffusion_du += timestep * theta * inner(stress_fluid_term_2, test_grad_v)
    # -------- Biharmonic equation ----------
    jac_biharmonic_dd = (-alpha_u/(J*J) * J_LinU * inner(grad(dk), grad(test_w))
                         + alpha_u/J * inner(grad(dd), grad(test_w)))
    # ---------Incompressibility  ----------
    incompressility_ALE_LinALl = NSE_ALE.get_Incompressibility_ALE_LinV_optimized(
        grad_uk, grad(du), F, grad_dd
    )
    jac_incompressibility_dp = incompressility_ALE_LinALl * test_q

    volume_terms_fluid = (jac_mass_du
            + jac_convection_du
            + jac_diffusion_du
            + jac_biharmonic_dd
            + jac_incompressibility_dp
            ) * dx_f
    if p_gauge is not None:
        volume_terms_fluid += (p_gauge * dp * test_q) * dx_f
    # -----------------------------------------------------------------------
    # ----------------- do-nothing bc ---------------------------------------
    neuman_term = NSE_ALE.get_stress_fluid_ALE_3rd_term_LinAll_short(
        Finv,
        Finv_LinU,
        grad_uk,
        grad(du),
        mu_f,
        J,
        J_F_inv_T_LinU,
    )
    neuman_flux = dot(neuman_term, n)

    out_flow_jac = - timestep * theta * dot(neuman_flux, test_v) * dS_outlet

    #-----------------------------------------------------------------------
    #---------------- Solid terms ---------------------------------------
    #-----------------------------------------------------------------------
    solid_stress_LinU = Structure_Terms.get_Piola_Kirchhoff_1st_LinAll(
        F, grad_dd, mu_s, lambda_s
    )
    jac_solid = (
        rho_s * dot(du, test_v)
        + timestep * theta * inner(solid_stress_LinU, grad(test_v))
        + rho_s * dot(dd, test_w)
        - rho_s * timestep * theta * dot(du, test_w)
        + dp * test_q
    ) * dx_s
    if p_gauge is not None:
        jac_solid += (p_gauge * dp * test_q) * dx_s


    return volume_terms_fluid + out_flow_jac + jac_solid
    # return volume_terms_fluid  + jac_solid

def build_residual(
    *,
    uk, u_prev,       # Fluid Velocity (current, old)
    dk, d_prev,       # Displacement (current, old)
    pk, p_prev,       # Pressure (current, old)
    v_test, w_test, q_test,          # Test functions (ψ_v, ψ_d, ψ_p)
    dt: Constant,
    theta: Constant,
    rho_f: Constant,
    mu_f: Constant,
    rho_s: Constant,
    lambda_s: Constant,
    mu_s: Constant,
    alpha_u: Constant,
    stab_eps: Constant,
    p_gauge: Constant | None = None,
    fluid_bs,
    solid_bs,
    outlet_bs,
    quad_order: int,
):
    # --- Integration Measures ---
    dx_f = dx(defined_on=fluid_bs, metadata={"q": quad_order})
    dx_s = dx(defined_on=solid_bs, metadata={"q": quad_order})
    dS_outlet = dS(defined_on=outlet_bs, metadata={"q": quad_order})
    I = Identity(2)
    n =  FacetNormal()
    grad_v = grad(uk)
    grad_d = grad(dk)
    grad_v_old = grad(u_prev)
    grad_d_old = grad(d_prev)
    F = ALE_Helpers.get_F(grad_d)
    Finv = ALE_Helpers.get_F_inv(F)
    J = ALE_Helpers.get_J(F)
    F_old = ALE_Helpers.get_F(grad_d_old)
    Finv_old = ALE_Helpers.get_F_inv(F_old)
    J_old = ALE_Helpers.get_J(F_old)
    pI = pk * Identity(2)
    J_theta = theta * J + (1.0 - theta) * J_old

    # acceleration term *DT
    acc_term = rho_f * J_theta * inner((uk - u_prev), v_test)
    # convection term
    convection_fluid = rho_f * J * dot(dot(grad_v, Finv), uk)
    convection_fluid_with_u = rho_f * J * dot(dot(grad_v, Finv), dk)
    convection_fluid_with_u_old = rho_f * J * dot(dot(grad_v, Finv), d_prev)
    old_convection_fluid = rho_f * J_old * dot(dot(grad_v_old, Finv_old), u_prev)
    convec_term = (
        dt * theta * dot(convection_fluid, v_test)
        + dt * (1.0-theta) * dot(old_convection_fluid, v_test)
        - dot(convection_fluid_with_u - convection_fluid_with_u_old, v_test)
    ) 
    # incompressibility term
    fluid_pressure = -(J * dot(pI, Finv.T))
    pressure_term = dt * inner(fluid_pressure, grad(v_test))
    # stress terms
    sigma_ALE = NSE_ALE.get_stress_fluid_except_pressure_ALE(mu_f, grad_v, Finv)
    sigma_ALE_old = NSE_ALE.get_stress_fluid_except_pressure_ALE(mu_f, grad_v_old, Finv_old)
    stress_fluid_viscous = J * dot(sigma_ALE, Finv.T)
    stress_fluid_viscous_old = J_old * dot(sigma_ALE_old, Finv_old.T)
    stress_term = dt * theta * inner(stress_fluid_viscous, grad(v_test))
    stress_term += dt * (1.0-theta) * inner(stress_fluid_viscous_old, grad(v_test))
    # biharmonic stabilization
    biharmonic_term = (
        alpha_u / J * inner(grad(dk), grad(w_test))
    )
    # incompressibility
    incompressibility_fluid = NSE_ALE.get_Incompressibility_ALE(uk, F)
    incompressibility_term = incompressibility_fluid * q_test

    residual_fluid = (
        acc_term
        + convec_term
        + pressure_term
        + stress_term
        + biharmonic_term
        + incompressibility_term
    ) * dx_f
    if p_gauge is not None:
        residual_fluid += (p_gauge * pk * q_test) * dx_f
    # do-nothing BC at outlet
    sigma_ALE_tilde = mu_f * dot(Finv.T, grad_v.T) 
    sigma_ALE_tilde_old = mu_f * dot(Finv_old.T, grad_v_old.T)
    stress_fluid_transpose = J * dot(sigma_ALE_tilde, Finv.T)
    stress_fluid_transpose_old = J_old * dot(sigma_ALE_tilde_old, Finv_old.T)
    neuman_flux = dot(stress_fluid_transpose, n)
    neuman_flux_old = dot(stress_fluid_transpose_old, n)
    out_flow = (- dt * theta * dot(neuman_flux, v_test) 
                - dt * (1.0-theta) * dot(neuman_flux_old, v_test)
    ) 
    residual_outlet = out_flow * dS_outlet

    #-----------------------------------------------------------------------
    #---------------- Solid terms ---------------------------------------
    #-----------------------------------------------------------------------
    solid_stress = Structure_Terms.get_Cauchy_stress(F, mu_s, lambda_s)
    solid_stress_old = Structure_Terms.get_Cauchy_stress(F_old, mu_s, lambda_s)
    solid_stress_transfomed = J * dot(solid_stress, Finv.T)
    solid_stress_transfomed_old = J_old * dot(solid_stress_old, Finv_old.T)
    residual_solid = (
        rho_s * inner(uk - u_prev, v_test)
        + dt * theta * inner(solid_stress_transfomed, grad(v_test))
        + dt * (1.0 - theta) * inner(solid_stress_transfomed_old, grad(v_test))
        + rho_s * inner(dk - d_prev, w_test)
        - rho_s * dt * theta * inner(uk, w_test)
        - rho_s * dt * (1.0 - theta) * inner(u_prev, w_test)
        + pk * q_test
    ) * dx_s
    if p_gauge is not None:
        residual_solid += (p_gauge * pk * q_test) * dx_s

    
    return residual_fluid + residual_outlet + residual_solid
    # return residual_fluid  + residual_solid


def finite_difference_jacobian_check(
    *,
    dh: DofHandler,
    res_form,
    jac_form,
    funcs: list,
    bcs,
    bcs_homog,
    quad_order: int,
    backend: str,
    eps: float = 1.0e-6,
    seed: int = 0,
):
    """
    Compare assembled Jacobian action against a finite-difference residual.
    Returns basic diagnostics as a dict.
    """
    rng = np.random.default_rng(seed)
    eq_res = Equation(None, res_form)
    eq_jac = Equation(jac_form, None)

    # Use homogeneous BCs consistently for both residual and Jacobian assembly.
    bc_active = bcs_homog or bcs
    bc_map = dh.get_dirichlet_data(bc_active)
    bc_rows = np.fromiter(bc_map.keys(), dtype=int) if bc_map else np.array([], dtype=int)
    inactive = np.asarray(sorted(dh.dof_tags.get("inactive", [])), dtype=int) if getattr(dh, "dof_tags", None) else np.array([], dtype=int)
    frozen = np.unique(np.concatenate([bc_rows, inactive])) if (bc_rows.size or inactive.size) else np.array([], dtype=int)
    mask = np.ones(dh.total_dofs, dtype=bool)
    if frozen.size:
        mask[frozen] = False

    direction = rng.standard_normal(dh.total_dofs)
    direction[~mask] = 0.0
    max_dir = float(np.linalg.norm(direction, ord=np.inf))
    if max_dir == 0.0:
        raise RuntimeError("All DOFs are constrained; cannot run finite-difference check.")
    delta = (eps / max_dir) * direction

    K, _ = assemble_form(eq_jac, dof_handler=dh, bcs=bcs_homog, quad_degree=quad_order, backend=backend)
    _, R0 = assemble_form(eq_res, dof_handler=dh, bcs=bcs_homog, quad_degree=quad_order, backend=backend)

    snap = [f.nodal_values.copy() for f in funcs]
    dh.add_to_functions(delta, funcs)
    dh.apply_bcs(bc_active, *funcs)
    _, R1 = assemble_form(eq_res, dof_handler=dh, bcs=bc_active, quad_degree=quad_order, backend=backend)
    for f, buf in zip(funcs, snap):
        f.nodal_values[:] = buf
    dh.apply_bcs(bc_active, *funcs)

    Jd = K @ delta
    fd = (R1 - R0) / eps

    active = np.flatnonzero(mask)
    err = Jd - fd
    max_abs = float(np.max(np.abs(err[active]))) if active.size else 0.0
    fd_norm = float(np.linalg.norm(fd[active], ord=np.inf)) if active.size else 0.0
    Jd_norm = float(np.linalg.norm(Jd[active], ord=np.inf)) if active.size else 0.0
    worst_gdof = int(active[np.argmax(np.abs(err[active]))]) if active.size else -1

    return {
        "max_abs_diff": max_abs,
        "fd_norm_inf": fd_norm,
        "Jd_norm_inf": Jd_norm,
        "worst_gdof": worst_gdof,
        "delta_norm_inf": float(np.linalg.norm(delta, ord=np.inf)),
    }



def _collect_boundary_edges(mesh, tags: tuple[str, ...]):
    bs = None
    missing = []
    for t in tags:
        try:
            bst = mesh.edge_bitset(t)
        except KeyError:
            missing.append(t)
            continue
        if bst is None or bst.cardinality() == 0:
            missing.append(t)
            continue
        bs = bst if bs is None else (bs | bst)
    return bs, missing


def compute_drag_lift(
    dh: DofHandler,
    mesh,
    u: VectorFunction,
    d: VectorFunction,
    p: Function,
    rho: float,
    mu: float,
    tags: tuple[str, ...] = ("cylinder", "beam_outer", "beam_root", "obstacle_set"),
):
    """
    Compute drag/lift on the perimeter of the cylinder **and** beam (including beam root).
    Falls back to the combined ``obstacle_set`` tag if explicit beam tags are absent.
    """
    edge_set, missing = _collect_boundary_edges(mesh, tags)
    if edge_set is None or edge_set.cardinality() == 0:
        raise ValueError(f"No boundary edges found for tags {tags}; missing={missing}")
    if missing:
        print(f"[warn] Drag/Lift: missing boundary tags with zero edges: {missing}")

    n = FacetNormal()
    I = Identity(2)
    F = I + grad(d)
    Finv = inv(F)
    J = det(F)
    #grad_u = dot(grad(u), Finv)
    grad_u = grad(u)  # NOT dot(grad(u), Finv)

    pI = p * I
    sigma_viscous_ALE = NSE_ALE.get_stress_fluid_except_pressure_ALE(mu, grad_u, Finv)
    stress_viscous = J * dot(sigma_viscous_ALE, Finv.T)
    fluid_pressure = -(J * dot(pI, Finv.T))
    # deal.II reference (step-fsi.cc): integrate the force on the obstacle,
    # i.e. -(σ·n) with n the outward normal from the fluid cell.
    traction = -dot(stress_viscous + fluid_pressure, n)
    ex = Constant([1.0, 0.0], dim=1)
    ey = Constant([0.0, 1.0], dim=1)
    measure = dS(defined_on=edge_set)
    drag = dot(traction, ex) * measure
    lift = dot(traction, ey) * measure
    hooks = {
        drag.integrand: {"name": "drag"},
        lift.integrand: {"name": "lift"},
    }
    res = assemble_form(
        Equation(None, drag + lift),
        dof_handler=dh,
        bcs=[],
        assembler_hooks=hooks,
        backend="python",
    )
    return float(res["drag"]), float(res["lift"])


def tip_displacement(dh: DofHandler, d: VectorFunction):
    coords = dh.get_dof_coords("dx")
    tip = np.array([BEAM_X0 + BEAM_LENGTH, CENTER[1]])
    diffs = coords - tip
    idx = int(np.argmin(np.einsum("ij,ij->i", diffs, diffs)))
    return float(d.nodal_values[idx]), float(d.nodal_values[idx + len(coords)])


def _vector_on_mesh_nodes(dh: DofHandler, mesh: Mesh, vf: VectorFunction) -> np.ndarray:
    """
    Extract a (n_nodes, 2) array by taking the vector components at mesh nodes.

    For higher-order fields (e.g. Q2), this returns the values on the *geometry*
    nodes only (corner nodes for Q1 geometry). This is suitable for lightweight
    VTK/Matplotlib visualization and avoids corrupting node data with interior DOFs.
    """
    n_nodes = len(mesh.nodes_list)
    out = np.zeros((n_nodes, 2), dtype=float)
    comps = list(getattr(vf, "components", ()))[:2]
    for comp_idx, comp in enumerate(comps):
        for gdof, lidx in comp._g2l.items():
            _field, node_id = dh._dof_to_node_map.get(int(gdof), ("", None))
            if node_id is None:
                continue
            out[int(node_id), comp_idx] = float(comp.nodal_values[lidx])
    return out


def _split_corner_cells_to_tris(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (triangles, tri_elem_ids) for Matplotlib Triangulation on mesh corners.

    - triangles: (n_tri, 3) int array of node indices
    - tri_elem_ids: (n_tri,) int array mapping each triangle to its parent element id
    """
    corners = np.asarray(mesh.corner_connectivity, dtype=int)
    n_elem = corners.shape[0]
    if mesh.element_type == "tri" or corners.shape[1] == 3:
        tris = corners[:, :3].copy()
        elem_ids = np.arange(n_elem, dtype=int)
        return tris, elem_ids
    if mesh.element_type == "quad" or corners.shape[1] == 4:
        tris = np.vstack((corners[:, [0, 1, 2]], corners[:, [0, 2, 3]])).astype(int, copy=False)
        elem_ids = np.repeat(np.arange(n_elem, dtype=int), 2)
        return tris, elem_ids
    raise ValueError(f"Unsupported element type for triangulation: {mesh.element_type} ({corners.shape[1]} corners)")


def _cellwise_mean_pressure(dh: DofHandler, p: Function, *, fluid_bs: BitSet | None = None) -> np.ndarray:
    """Compute a simple per-cell pressure value for VTK cell data."""
    n_cells = len(dh.mixed_element.mesh.corner_connectivity)
    out = np.full(n_cells, np.nan, dtype=float)
    for eid in range(n_cells):
        if fluid_bs is not None and not bool(fluid_bs.mask[int(eid)]):
            continue
        gdofs = np.asarray(dh.element_maps["p"][int(eid)], dtype=int)
        vals = p.get_nodal_values(gdofs)
        out[int(eid)] = float(np.mean(vals)) if vals.size else np.nan
    return out


def save_fsi_frame(
    *,
    case_label: str = "FSI",
    outdir: Path,
    mesh: Mesh,
    dh: DofHandler,
    u: VectorFunction,
    d: VectorFunction,
    fluid_bs: BitSet,
    step: int,
    time: float,
    cd: float,
    cl: float,
    tip_dx: float,
    tip_dy: float,
    scale: float = 1.0,
    dpi: int = 150,
) -> Path:
    """
    Save a single PNG snapshot showing fluid velocity magnitude + deformed beam outline.

    The plot uses a lightweight corner triangulation (Q1 geometry) and warps the
    coordinates by `scale * d` for visualization.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.tri import Triangulation

    frames_dir = outdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    path = frames_dir / f"frame_{step:04d}.png"

    u_nodes = _vector_on_mesh_nodes(dh, mesh, u)
    d_nodes = _vector_on_mesh_nodes(dh, mesh, d)

    coords = np.asarray(mesh.nodes_x_y_pos, dtype=float)
    x_def = coords[:, 0] + float(scale) * d_nodes[:, 0]
    y_def = coords[:, 1] + float(scale) * d_nodes[:, 1]

    tris, tri_elem = _split_corner_cells_to_tris(mesh)
    tri = Triangulation(x_def, y_def, tris)
    tri.set_mask(~np.asarray(fluid_bs.mask, dtype=bool)[tri_elem])

    u_mag = np.linalg.norm(u_nodes, axis=1)
    u_mag = np.nan_to_num(u_mag, nan=0.0)

    fig, ax = plt.subplots(figsize=(12, 3.6))
    pcm = ax.tripcolor(tri, u_mag, shading="gouraud", cmap="turbo")
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("|u| (velocity magnitude)")

    # Draw all outer boundaries (channel) lightly, then obstacle boundary thicker.
    segs_outer: list[np.ndarray] = []
    segs_obs: list[np.ndarray] = []
    for edge in mesh.edges_list:
        nids = list(edge.all_nodes) if getattr(edge, "all_nodes", ()) else list(edge.nodes)
        if len(nids) < 2:
            continue
        pts = np.c_[x_def[nids], y_def[nids]]
        out = segs_outer if edge.right is None else None
        obs = segs_obs if getattr(edge, "tag", "") in {"cylinder", "beam_outer", "beam_root", "obstacle_set"} else None
        for i in range(len(nids) - 1):
            if out is not None:
                out.append(pts[i : i + 2])
            if obs is not None:
                obs.append(pts[i : i + 2])

    if segs_outer:
        ax.add_collection(LineCollection(segs_outer, colors="black", linewidths=0.6, alpha=0.65, zorder=5))
    if segs_obs:
        ax.add_collection(LineCollection(segs_obs, colors="black", linewidths=2.2, zorder=6))

    tip_ref = np.array([BEAM_X0 + BEAM_LENGTH, CENTER[1]], dtype=float)
    tip_xy = tip_ref + float(scale) * np.array([tip_dx, tip_dy], dtype=float)
    ax.plot([tip_xy[0]], [tip_xy[1]], marker="o", markersize=4, color="crimson", zorder=7)

    ax.set_aspect("equal", "box")
    xmin, ymin = float(np.min(x_def)), float(np.min(y_def))
    xmax, ymax = float(np.max(x_def)), float(np.max(y_def))
    span = max(xmax - xmin, ymax - ymin, 1e-12)
    pad = 0.05 * span
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{case_label} | step={step}, t={time:.3f} | Cd={cd:.4e}, Cl={cl:.4e}")
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- 
# Boundary data (matches BoundaryParabola in step-fsi.cc)
# -----------------------------------------------------------------------------
def inlet_parabola(y: float, t: float, u_mean: float) -> float:
    scale = 4.0 / (H * H) * y * (H - y)
    if t < 2.0:
        return 1.5 * u_mean * 0.5 * (1.0 - math.cos(0.5 * math.pi * t)) * scale
    return 1.5 * u_mean * scale


def build_bcs(
    u_mean: float,
    theta: float,
) -> Tuple[List[BoundaryCondition], List[BoundaryCondition]]:
    zero = lambda x, y, t=0.0: 0.0

    def u_in(x, y, t=0.0):
        return inlet_parabola(y, t, u_mean)

    vel_bcs = [
        BoundaryCondition("ux", "dirichlet", "inlet", u_in),
        BoundaryCondition("uy", "dirichlet", "inlet", zero),
        BoundaryCondition("ux", "dirichlet", "walls", zero),
        BoundaryCondition("uy", "dirichlet", "walls", zero),
        BoundaryCondition("ux", "dirichlet", "cylinder", zero),
        BoundaryCondition("uy", "dirichlet", "cylinder", zero),
        # Clamp the beam root to the rigid cylinder (locator-based; no explicit edge tag)
        BoundaryCondition("ux", "dirichlet", "beam_root", zero),
        BoundaryCondition("uy", "dirichlet", "beam_root", zero),
    ]
    # Keep the ALE mesh fixed on the outer fluid boundary and the rigid cylinder.
    disp_bcs = [
        BoundaryCondition("dx", "dirichlet", "inlet", zero),
        BoundaryCondition("dy", "dirichlet", "inlet", zero),
        # Match step-fsi.cc: allow tangential mesh slip on the walls (fix normal component only).
        BoundaryCondition("dy", "dirichlet", "walls", zero),
        BoundaryCondition("dx", "dirichlet", "outlet", zero),
        BoundaryCondition("dy", "dirichlet", "outlet", zero),
        BoundaryCondition("dx", "dirichlet", "cylinder", zero),
        BoundaryCondition("dy", "dirichlet", "cylinder", zero),
        # Clamp beam root (no boundary edges; picked up via locator)
        BoundaryCondition("dx", "dirichlet", "beam_root", zero),
        BoundaryCondition("dy", "dirichlet", "beam_root", zero),
    ]
    # Anchor one pressure node to avoid the nullspace
    # p_bcs = [BoundaryCondition("p", "dirichlet", "outlet", zero)]
    bcs = vel_bcs + disp_bcs 
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, zero) for b in bcs]
    return bcs, bcs_homog


# ----------------------------------------------------------------------------- 
# Main driver
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deal.II FSI benchmark in pycutfem (ALE, conforming mesh).")
    ap.add_argument("--mesh", type=Path, default=Path("examples/meshes/fsi_conforming.msh"), help="Path to mesh (.msh from gmsh or .inp UCD).")
    ap.add_argument("--mesh-format", choices=("auto", "gmsh", "ucd"), default="auto", help="Override mesh loader; auto picks by file extension.")
    ap.add_argument("--mesh-size", type=float, default=None, help="Optional characteristic mesh size (used for tagging tolerances).")
    ap.add_argument("--poly-order", type=int, default=2, help="Polynomial order for velocity/displacement (Taylor–Hood).")
    ap.add_argument(
        "--turek-case",
        choices=("fsi1", "fsi2", "fsi3"),
        default="fsi1",
        help="Turek–Hron benchmark case preset: fsi1=steady (Re≈20), fsi2=periodic (Re≈100), fsi3=chaotic (Re≈200).",
    )
    ap.add_argument(
        "--u-mean",
        type=float,
        default=None,
        help="Mean inflow velocity U_mean (overrides --turek-case preset).",
    )
    ap.add_argument(
        "--rho-s",
        type=float,
        default=None,
        help="Solid density rho_s (overrides --turek-case preset).",
    )
    ap.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Time step size; defaults to the selected --turek-case preset (fsi1:1.0, fsi2/3:0.005).",
    )
    ap.add_argument(
        "--theta",
        type=float,
        default=None,
        help="Theta scheme parameter (1=BE, 0.5=CN); defaults to the selected --turek-case preset (fsi1:1.0, fsi2/3:0.5).",
    )
    ap.add_argument(
        "--allow-dt-reduction",
        action="store_true",
        help="Allow adaptive dt reduction when Newton updates blow up (requires re-JIT of dt-dependent kernels).",
    )
    ap.add_argument("--n-steps", type=int, default=2, help="Number of time steps (default small for quick verification).")
    ap.add_argument("--backend", choices=("jit", "python"), default="python", help="Form compiler backend.")
    ap.add_argument(
        "--nonlinear-solver",
        choices=("snes", "newton"),
        default="newton",
        help="Nonlinear solver: PETSc SNES (robust) or pure Python Newton.",
    )
    ap.add_argument("--newton-tol", type=float, default=1.0e-8, help="Newton convergence tolerance on ‖R‖∞.")
    ap.add_argument("--max-newton-iter", type=int, default=40, help="Maximum Newton iterations per time step.")
    ap.add_argument(
        "--linear-solver",
        choices=("scipy", "petsc"),
        default="scipy",
        help="Linear solver backend for --nonlinear-solver=newton.",
    )
    ap.add_argument(
        "--ls-mode",
        choices=("armijo", "dealii"),
        default="armijo",
        help="Line search: Armijo on ½‖R‖² (robust) or deal.II-style ‖R‖∞ decrease.",
    )
    ap.add_argument("--ls-max-iter", type=int, default=25, help="Maximum backtracking steps in the line search.")
    ap.add_argument("--ls-reduction", type=float, default=0.5, help="Backtracking reduction factor α←β·α.")
    ap.add_argument("--ls-c1", type=float, default=1.0e-4, help="Armijo sufficient decrease parameter c1.")
    ap.add_argument("--boundary-tol", type=float, default=1.0e-6, help="Tolerance for geometric boundary retagging.")
    ap.add_argument("--mesh-report", action="store_true", help="Print mesh diagnostics (boundary coverage, hanging nodes) and exit before assembly.")
    ap.add_argument("--assemble-only", action="store_true", help="Assemble residual/Jacobian once and exit (no Newton solve).")
    ap.add_argument("--plot-bcs", action="store_true", help="Save a PNG of Dirichlet DOFs using plot_mesh_2.")
    ap.add_argument("--plot-bc-bitsets", action="store_true", help="Plot BC DOF sets per boundary tag (diagnostic).")
    ap.add_argument(
        "--force-geometric-tags",
        action="store_true",
        default=False,
        help="Overwrite existing boundary tags with geometric retagging (useful if the mesh has no/incorrect tags).",
    )
    ap.add_argument("--n-refinements", type=int, default=0, help="Uniform refinement levels (deal.II: triangulation.refine_global(n)).")
    ap.add_argument("--save-vtk", dest="save_vtk", action="store_true", help="Enable VTK output.")
    ap.add_argument("--no-save-vtk", dest="save_vtk", action="store_false", help="Disable VTK output.")
    ap.set_defaults(save_vtk=False)
    ap.add_argument("--vtk-every", type=int, default=1, help="VTK output frequency (in time steps).")
    ap.add_argument("--output-dir", type=Path, default=Path("fsi_dealii_reference_results"), help="Directory for VTK output.")
    ap.add_argument("--save-frames", action="store_true", help="Save Matplotlib PNG frames (velocity + beam motion).")
    ap.add_argument("--frames-every", type=int, default=1, help="PNG frame output frequency (in time steps).")
    ap.add_argument("--frames-dpi", type=int, default=150, help="DPI for saved PNG frames.")
    ap.add_argument("--frames-scale", type=float, default=1.0, help="Scale factor applied to displacement for plotting (visual only).")
    ap.add_argument("--fd-check", action="store_true", help="Run a finite-difference Jacobian check and report the max discrepancy.")
    ap.add_argument("--fd-eps", type=float, default=1.0e-6, help="Step length for the finite-difference Jacobian check.")
    ap.add_argument(
        "--no-anchor-pressure",
        action="store_false",
        dest="anchor_pressure",
        help="Disable pinning one pressure DOF (gauge fixing).",
    )
    ap.set_defaults(anchor_pressure=True)
    return ap.parse_args()


def _pin_pressure_gauge(mesh: Mesh, dh: DofHandler, *, tag: str = "p_anchor") -> int | None:
    """
    Pin a single pressure DOF to remove the (near) constant-pressure nullspace.
    Returns the pinned global DOF id, or None if not found.
    """
    try:
        coords_p = dh.get_dof_coords("p")
        p_dofs = np.asarray(dh.get_field_slice("p"), dtype=int)
    except Exception:
        return None
    if coords_p is None or len(coords_p) == 0 or p_dofs.size == 0:
        return None

    # Prefer a DOF on the outlet boundary (x ≈ xmax), closest to the mid-height.
    try:
        x_max = float(np.max(coords_p[:, 0]))
        outlet = np.where(np.isclose(coords_p[:, 0], x_max, atol=1.0e-10))[0]
    except Exception:
        outlet = np.array([], dtype=int)
    if outlet.size == 0:
        outlet = np.arange(coords_p.shape[0], dtype=int)

    y_mid = 0.5 * H
    loc = int(outlet[np.argmin(np.abs(coords_p[outlet, 1] - y_mid))])
    gdof = int(p_dofs[loc])

    dh.dof_tags.setdefault(tag, set()).add(gdof)
    return gdof


def _collect_edge_dofs(mesh: Mesh, dh: DofHandler, tag: str, field: str) -> set[int]:
    """
    Return all DOFs of ``field`` that lie on edges carrying ``tag``.
    Uses DofHandler.edge_dofs so higher-order edge nodes are included.
    """
    try:
        edge_ids = mesh.edge_bitset(tag).to_indices()
    except Exception:
        return set()
    out: set[int] = set()
    for eid in edge_ids:
        try:
            out.update(int(d) for d in dh.edge_dofs(field, int(eid)))
        except Exception:
            continue
    return out


def _plot_dirichlet(mesh: Mesh, dh: DofHandler, bc_dofs: Dict[int, float], missing: set[int], extra: set[int], output_path: Path) -> None:
    """
    Save a diagnostic plot of Dirichlet DOFs per field using plot_mesh_2.
    """
    import matplotlib
    # matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_mesh_2(mesh, plot_nodes=False, plot_edges=True, elem_tags=False, edge_colors=True, show=False, ax=ax)

    if getattr(dh, "_dof_coords", None) is None:
        _ = dh.get_field_slice(dh.field_names[0])  # force coord build
    coords = dh._dof_coords

    palette = {"ux": "#d62728", "uy": "#2ca02c", "dx": "#1f77b4", "dy": "#ff7f0e", "p": "#9467bd"}
    grouped: Dict[str, list[int]] = {}
    for gd in bc_dofs.keys():
        fld, _ = dh._dof_to_node_map.get(int(gd), (None, None))
        if fld is None:
            continue
        grouped.setdefault(fld, []).append(int(gd))

    for fld, ids in grouped.items():
        pts = coords[ids]
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            "o",
            markersize=3.2,
            markerfacecolor=palette.get(fld, "k"),
            markeredgecolor="white",
            markeredgewidth=0.35,
            linestyle="None",
            label=f"{fld} BC",
        )

    if missing:
        pts = coords[list(missing)]
        ax.plot(pts[:, 0], pts[:, 1], "x", color="black", markersize=6, markeredgewidth=1.4, label="missing")
    if extra:
        pts = coords[list(extra)]
        ax.plot(pts[:, 0], pts[:, 1], "s", color="gold", markersize=4.5, markeredgecolor="black", label="extra")

    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.set_title("Dirichlet DOFs by field")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.show()
    # plt.close(fig)


def _plot_bc_bitsets(mesh: Mesh, dh: DofHandler, bcs: List[BoundaryCondition], output_path: Path) -> None:
    """
    Plot Dirichlet DOF sets per boundary tag on subplots using plot_mesh_2.
    """
    import math
    import matplotlib
    # matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect DOFs per tag by re-querying get_dirichlet_data for each tag separately.
    tag_to_bcs: Dict[str, List[BoundaryCondition]] = {}
    for bc in bcs:
        if getattr(bc, "method", "") != "dirichlet":
            continue
        tag_to_bcs.setdefault(bc.domain_tag, []).append(bc)

    tag_to_dofs: Dict[str, Dict[str, list[int]]] = {}
    for tag, bclist in tag_to_bcs.items():
        dofs = dh.get_dirichlet_data(bclist)
        by_field: Dict[str, list[int]] = {}
        for gd in dofs:
            field, _ = dh._dof_to_node_map.get(int(gd), (None, None))
            if field is None:
                continue
            by_field.setdefault(field, []).append(int(gd))
        tag_to_dofs[tag] = by_field

    if not tag_to_dofs:
        print("[plot-bc-bitsets] No Dirichlet DOFs found.")
        return

    palette = {"ux": "#d62728", "uy": "#2ca02c", "dx": "#1f77b4", "dy": "#ff7f0e", "p": "#9467bd"}
    n_tags = len(tag_to_dofs)
    ncols = min(3, n_tags)
    nrows = math.ceil(n_tags / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[n_tags:]:
        ax.axis("off")

    for ax, (tag, by_field) in zip(axes, tag_to_dofs.items()):
        plot_mesh_2(mesh, plot_nodes=False, plot_edges=True, elem_tags=False, edge_colors=True, show=False, ax=ax)
        for fld, ids in by_field.items():
            pts = dh._dof_coords[ids]
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                "o",
                markersize=2.5,
                markerfacecolor=palette.get(fld, "k"),
                markeredgecolor="white",
                markeredgewidth=0.25,
                linestyle="None",
                label=fld,
            )
        ax.set_title(tag)
        ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.show()
    # plt.close(fig)


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Turek–Hron benchmark presets
    # ------------------------------------------------------------------
    case_label_map = {"fsi1": "FSI-1", "fsi2": "FSI-2", "fsi3": "FSI-3"}
    case_defaults = {
        # Matches deal.II step-fsi / "FSI-1" steady benchmark (Re≈20).
        "fsi1": {"u_mean": 0.2, "rho_s": 1.0e3, "dt": 1.0, "theta": 1.0},
        # Oscillatory benchmarks (Re≈100/200). Keep rho_s aligned with the other
        # pycutfem Turek FSI examples.
        "fsi2": {"u_mean": 1.0, "rho_s": 1.0e4, "dt": 0.005, "theta": 0.5},
        "fsi3": {"u_mean": 2.0, "rho_s": 1.0e4, "dt": 0.005, "theta": 0.5},
    }

    case_label = case_label_map.get(str(args.turek_case), str(args.turek_case))
    preset = case_defaults.get(str(args.turek_case), case_defaults["fsi1"])
    u_mean_ref = float(args.u_mean) if args.u_mean is not None else float(preset["u_mean"])
    rho_s_val = float(args.rho_s) if args.rho_s is not None else float(preset["rho_s"])
    args.dt = float(args.dt) if args.dt is not None else float(preset["dt"])
    args.theta = float(args.theta) if args.theta is not None else float(preset["theta"])
    if not math.isclose(float(args.dt), float(preset["dt"])):
        print(
            f"[warn] dt={float(args.dt):g} differs from {float(preset['dt']):g} for {case_label}; "
            "large dt can suppress the expected oscillatory response."
        )
    if not math.isclose(float(args.theta), float(preset["theta"])):
        print(
            f"[warn] theta={float(args.theta):g} differs from {float(preset['theta']):g} for {case_label}; "
            "theta>0.5 adds numerical damping."
        )

    mesh_path = args.mesh.resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    use_gmsh = args.mesh_format == "gmsh" or (args.mesh_format == "auto" and mesh_path.suffix.lower() == ".msh")
    if use_gmsh:
        mesh = mesh_from_gmsh(mesh_path, apply_boundary_tags=True)
    else:
        mesh, _fluid, _solid = load_ucd_mesh(mesh_path, poly_order=1)
    fluid_bs, solid_bs = classify_fluid_solid(mesh)

    if args.n_refinements > 0:
        mesh = mesh.refine_uniform(int(args.n_refinements))
        fluid_bs, solid_bs = classify_fluid_solid(mesh)

    counts = ensure_boundary_tags(
        mesh,
        tol=args.boundary_tol,
        mesh_size=args.mesh_size,
        force_geometric=args.force_geometric_tags,
    )
    n_ifc = tag_fluid_solid_interface_edges(mesh, tag="beam_outer")
    ensure_obstacle_set(mesh)
    if n_ifc:
        print(f"Tagged fluid–solid interface edges: {n_ifc} ('beam_outer').")
    node_counts = tag_nodes_from_edges(mesh)
    outlet_bs = mesh.edge_bitset("outlet")
    if outlet_bs.cardinality() == 0:
        raise RuntimeError("Outlet boundary is empty after retagging; check mesh geometry.")
    counts_msg = ", ".join(f"{k}:{v[0]}->{v[1]}" for k, v in counts.items())
    nodes_msg = ", ".join(f"{k}:{v}" for k, v in node_counts.items())
    print(f"Boundary edges (before→after): {counts_msg}")
    print(f"Boundary nodes per tag: {nodes_msg}")

    hanging_edges = find_hanging_edges(mesh, tol=args.boundary_tol)
    if hanging_edges:
        xs = np.array([x for _, x, _ in hanging_edges], dtype=float)
        ys = np.array([y for _, _, y in hanging_edges], dtype=float)
        sample = [(eid, round(float(x), 4), round(float(y), 4)) for eid, x, y in hanging_edges[:8]]
        print(
            f"[warn] Hanging edges (single-owner off-boundary): {len(hanging_edges)} "
            f"(x≈{xs.min():.3f}…{xs.max():.3f}, y≈{ys.min():.3f}…{ys.max():.3f}, examples={sample})"
        )
    else:
        print("Hanging edges: none detected (all single-owner edges lie on boundaries).")

    # Quick orientation check
    coords = mesh.nodes_x_y_pos
    areas = []
    for el in mesh.elements_list:
        pts = coords[list(el.corner_nodes)]
        x = pts[:, 0]
        y = pts[:, 1]
        area = 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
        areas.append(abs(area))
    print(f"Element area stats: min={np.min(areas):.3e}, max={np.max(areas):.3e}")

    element = MixedElement(
        mesh,
        field_specs={"ux": args.poly_order, "uy": args.poly_order, "dx": args.poly_order, "dy": args.poly_order, "p": args.poly_order - 1},
    )
    # Match step-fsi.cc: CG velocity/displacement, DG pressure (FE_DGP).
    dh = DofHandler(element, method="cg", field_methods={"p": "dg"})

    vel_space = FunctionSpace(name="vel", field_names=["ux", "uy"], dim=1)
    disp_space = FunctionSpace(name="disp", field_names=["dx", "dy"], dim=1)

    du = VectorTrialFunction(space=vel_space, dof_handler=dh)
    dd = VectorTrialFunction(space=disp_space, dof_handler=dh)
    dp = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)

    v = VectorTestFunction(space=vel_space, dof_handler=dh)
    w = VectorTestFunction(space=disp_space, dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)

    uk = VectorFunction(name="u", field_names=["ux", "uy"], dof_handler=dh)
    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    dk = VectorFunction(name="d", field_names=["dx", "dy"], dof_handler=dh)
    d_prev = VectorFunction(name="d_prev", field_names=["dx", "dy"], dof_handler=dh)
    pk = Function(name="p", field_name="p", dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)
    for f in (uk, u_prev, dk, d_prev, pk, p_prev):
        f.nodal_values.fill(0.0)

    # Physical parameters from step-fsi.prm
    rho_f = Constant(1.0e3)
    # step-fsi.cc uses ν (kinematic viscosity) and multiplies by density in the stress:
    # μ = ρ ν, with ν = 1e-3 from step-fsi.prm.
    nu_f = Constant(1.0e-3)
    mu_f = rho_f * nu_f
    rho_s = Constant(float(rho_s_val))
    mu_s = Constant(0.5e6)
    nu_s = 0.4 # Poisson ratio
    E_s = 2.0 * float(mu_s.value) * (1.0 + nu_s)
    lambda_s = Constant(E_s * nu_s / ((1.0 + nu_s) * (1.0 - 2.0 * nu_s)))
    alpha_u = Constant(1.0e-8) # mesh control parameter
    stab_eps = Constant(1.0e-8)
    dt_const = Constant(args.dt)
    theta_const = Constant(args.theta)
    quad_order = 2 * args.poly_order + 4

    # Drop pressure DOFs living purely in the solid.
    # NOTE: step-fsi.cc keeps pressure unknowns in the solid and enforces p=0 there
    # via a mass term; leave them enabled here to mirror the reference.
    # dropped_p = dh.tag_dofs_from_element_bitset("inactive", "p", solid_bs, strict=True)
    # if dropped_p:
    #     print(f"Dropped {len(dropped_p)} pressure DOFs inside the solid.")

    # If the mesh lacks an explicit beam_root tag, derive it from cylinder edges.
    if mesh.edge_bitset("beam_root").cardinality() == 0:
        loc_map = _adaptive_locators(mesh, args.boundary_tol)
        beam_root_loc = loc_map.get("beam_root", lambda *_: False)
        cyl_edges = mesh.edge_bitset("cylinder").to_indices()
        if cyl_edges.size:
            # Use corner nodes on the selected cylinder edges to avoid sweeping high-order interior DOFs.
            for field in ("ux", "uy", "dx", "dy"):
                ids: set[int] = set()
                node2dof = dh.dof_map.get(field, {})
                for eid in cyl_edges:
                    try:
                        e_obj = mesh.edge(int(eid))
                    except Exception:
                        continue
                    for nid in (e_obj.all_nodes if e_obj.all_nodes else e_obj.nodes):
                        x, y = mesh.nodes_x_y_pos[int(nid)]
                        if beam_root_loc(float(x), float(y)):
                            gd = node2dof.get(int(nid))
                            if gd is not None:
                                ids.add(int(gd))
                if ids:
                    dh.dof_tags.setdefault("beam_root", set()).update(ids)

    pressure_is_dg = bool(getattr(dh, "_is_dg_field", lambda f: False)("p"))
    p_gauge = Constant(1.0e-6) if pressure_is_dg else None

    res_form = build_residual(
        uk=uk,
        u_prev=u_prev,
        dk=dk,
        d_prev=d_prev,
        pk=pk,
        p_prev=p_prev,
        v_test=v,   w_test=w, q_test=q,
        dt=dt_const,
        theta=theta_const,
        rho_f=rho_f,
        mu_f=mu_f,
        rho_s=rho_s,
        lambda_s=lambda_s,
        mu_s=mu_s,
        alpha_u=alpha_u,
        stab_eps=stab_eps,
        p_gauge=p_gauge,
        fluid_bs=fluid_bs,
        solid_bs=solid_bs,
        outlet_bs=outlet_bs,
        quad_order=quad_order,
    )
    # return assemble_form(Equation(None, res_form), dh, backend='python')

    jac_form = build_jac(
        uk=uk,
        u_prev=u_prev,
        dk=dk,
        d_prev=d_prev,
        pk=pk,
        p_prev=p_prev,
        du=du,
        dd=dd,
        dp=dp,
        test_v=v,
        test_w=w,
        test_q=q,
        timestep=dt_const,
        theta=theta_const,
        rho_f=rho_f,
        mu_f=mu_f,
        rho_s=rho_s,
        lambda_s=lambda_s,
        mu_s=mu_s,
        alpha_u=alpha_u,
        stab_eps=stab_eps,
        p_gauge=p_gauge,
        fluid_bs=fluid_bs,
        solid_bs=solid_bs,
        outlet_bs=outlet_bs,
        quad_order=quad_order,
    )
    # return assemble_form(Equation(jac_form, None), dh, backend='python')
    

    diameter = 2.0 * float(RADIUS)
    re_mean = float(u_mean_ref) * diameter / float(nu_f.value)
    print(
        f"Benchmark preset: {case_label} | U_mean={float(u_mean_ref):g}, rho_s={float(rho_s_val):g}, "
        f"dt={float(args.dt):g}, theta={float(args.theta):g} | Re_mean≈{re_mean:.1f}"
    )
    bcs, bcs_homog = build_bcs(u_mean=u_mean_ref, theta=args.theta)
    if args.anchor_pressure and not pressure_is_dg:
        pinned = _pin_pressure_gauge(mesh, dh, tag="p_anchor")
        if pinned is None:
            print("[warn] Could not pin a pressure DOF; continuing without gauge fixing.")
        else:
            zero = lambda x, y, t=0.0: 0.0
            bcs.append(BoundaryCondition("p", "dirichlet", "p_anchor", zero))
            bcs_homog.append(BoundaryCondition("p", "dirichlet", "p_anchor", zero))
            print(f"Pinned pressure DOF for gauge fixing: {pinned}")
    elif args.anchor_pressure and pressure_is_dg:
        print("[info] DG pressure uses a weak gauge term; skipping strong pinning.")

    dh.apply_bcs(bcs, uk, u_prev, dk, d_prev, pk, p_prev)
    print(f"Mesh elements: fluid={fluid_bs.cardinality()}, solid={solid_bs.cardinality()}")
    print(f"Total DOFs: {dh.total_dofs}")
    # Avoid locator-based sweep of interior DOFs; rely on tagged boundary edges.
    bc_dofs = dh.get_dirichlet_data(bcs, locators={})
    print(f"Dirichlet constraints: {len(bc_dofs)} DOFs")
    bc_dof_set = set(bc_dofs.keys())
    boundary_nodes = boundary_nodes_by_tag(mesh)
    bc_fields_by_tag: Dict[str, set[str]] = {}
    for bc in bcs:
        if getattr(bc, "method", "") != "dirichlet":
            continue
        bc_fields_by_tag.setdefault(bc.domain_tag, set()).add(bc.field)

    coverage_report = []
    locators = getattr(mesh, "_boundary_locators", {})
    field_sets = {f: set(dh.get_field_slice(f)) for f in dh.field_names}
    geom_nodes_union: set[int] = set()
    for tag, fields in bc_fields_by_tag.items():
        nodes = boundary_nodes.get(tag, set())
        geom_nodes_union |= nodes
        for field in fields:
            node2dof = dh.dof_map.get(field, {})
            expected = {node2dof[n] for n in nodes if n in node2dof}
            if tag in dh.dof_tags:
                expected |= (set(dh.dof_tags[tag]) & field_sets[field])
            missing = expected - bc_dof_set
            coverage_report.append((field, tag, len(expected), len(missing)))
    print(f"Boundary geometry nodes touched by BCs: {len(geom_nodes_union)}")
    for field, tag, exp, miss in coverage_report:
        msg = f"{field}@{tag}: expected {exp}, missing {miss}"
        if miss:
            msg += f" (first few missing: {sorted(list(missing))[:5]})"
        print("  " + msg)
    # Hard verification: all expected boundary DOFs must be present.
    expected_all: set[int] = set()
    for tag, fields in bc_fields_by_tag.items():
        nodes = boundary_nodes.get(tag, set())
        for field in fields:
            node2dof = dh.dof_map.get(field, {})
            expected_all |= {node2dof[n] for n in nodes if n in node2dof}
            # include higher-order edge DOFs that live on tagged edges
            expected_all |= _collect_edge_dofs(mesh, dh, tag, field)
            if tag in dh.dof_tags:
                expected_all |= (set(dh.dof_tags[tag]) & field_sets[field])
    missing_all = expected_all - bc_dof_set
    extra_all = bc_dof_set - expected_all
    if missing_all:
        msg = f"{len(missing_all)} boundary DOFs missing (examples: {sorted(list(missing_all))[:10]})."
        if args.mesh_report:
            print(f"[warn] Dirichlet coverage incomplete: {msg}")
        else:
            raise RuntimeError(f"Dirichlet coverage incomplete: {msg}")
    if extra_all:
        print(f"[warn] Dirichlet set has {len(extra_all)} DOFs not on tagged boundaries (examples: {sorted(list(extra_all))[:10]}).")
    elif not missing_all:
        print("All boundary DOFs covered by supplied Dirichlet tags.")

    if args.plot_bcs:
        out_path = Path("examples/plots/fsi_dirichlet_bcs.png")
        _plot_dirichlet(mesh, dh, bc_dofs, missing_all, extra_all, out_path)
        print(f"Saved Dirichlet BC plot to {out_path}")
    if args.plot_bc_bitsets:
        out_path = Path("examples/plots/fsi_dirichlet_bc_sets.png")
        _plot_bc_bitsets(mesh, dh, bcs, out_path)
        print(f"Saved Dirichlet BC bitsets to {out_path}")

    if args.mesh_report:
        print("Mesh diagnostics requested (--mesh-report); skipping assembly/solve.")
        return

    if args.fd_check:
        stats = finite_difference_jacobian_check(
            dh=dh,
            res_form=res_form,
            jac_form=jac_form,
            funcs=[uk, dk, pk],
            bcs=bcs,
            bcs_homog=bcs_homog,
            quad_order=quad_order,
            backend=args.backend,
            eps=args.fd_eps,
        )
        print(
            f"[fd-check] ||J·δ||_inf={stats['Jd_norm_inf']:.3e}, "
            f"||FD||_inf={stats['fd_norm_inf']:.3e}, "
            f"max|diff|={stats['max_abs_diff']:.3e} at gdof {stats['worst_gdof']}, "
            f"||δ||_inf={stats['delta_norm_inf']:.3e}"
        )
        if args.assemble_only:
            return

    if args.assemble_only:
        print("Assembling once with backend='{0}' for diagnostics...".format(args.backend))
        eq_res = Equation(None, res_form)
        eq_jac = Equation(jac_form, None)
        assemble_form(eq_res, dof_handler=dh, bcs=bcs, backend=args.backend)
        assemble_form(eq_jac, dof_handler=dh, bcs=bcs, backend=args.backend)
        print("Assembly completed; exiting early (--assemble-only).")
        return

    time_params = TimeStepperParameters(
        dt=args.dt,
        max_steps=args.n_steps,
        theta=float(args.theta),
        stop_on_steady=False,
        final_time=args.dt * args.n_steps,
        allow_dt_reduction=bool(args.allow_dt_reduction),
    )

    newton_params = NewtonParameters(
        newton_tol=float(args.newton_tol),
        max_newton_iter=int(args.max_newton_iter),
        line_search=True,
        ls_mode=str(args.ls_mode),
        ls_max_iter=int(args.ls_max_iter),
        ls_reduction=float(args.ls_reduction),
        ls_c1=float(args.ls_c1),
    )
    if args.nonlinear_solver == "snes":
        solver = PetscSnesNewtonSolver(
            residual_form=res_form,
            jacobian_form=jac_form,
            dof_handler=dh,
            mixed_element=element,
            bcs=bcs,
            bcs_homog=bcs_homog,
            newton_params=newton_params,
            quad_order=quad_order,
            backend=args.backend,
            petsc_options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "ksp_type": "preonly",
                "pc_type": "lu",
                # Let PETSc pick an available LU backend.
            },
        )
    else:
        solver = NewtonSolver(
            residual_form=res_form,
            jacobian_form=jac_form,
            dof_handler=dh,
            mixed_element=element,
            bcs=bcs,
            bcs_homog=bcs_homog,
            newton_params=newton_params,
            lin_params=LinearSolverParameters(backend=str(args.linear_solver)),
            quad_order=quad_order,
            backend=args.backend,
        )

    def _sync_dt(new_dt: float) -> None:
        if math.isclose(float(dt_const.value), float(new_dt)):
            return
        dt_const.value = float(new_dt)
        if solver.backend == "jit":
            solver._compile_all_kernels()

    time_params.on_dt_change = _sync_dt

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "fsi_benchmark.csv"
    csv_fieldnames = ["step", "time", "drag", "lift", "cd", "cl", "tip_dx", "tip_dy"]
    csv_file = None
    csv_writer = None
    try:
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        csv_writer.writeheader()
        csv_file.flush()
    except Exception as exc:
        print(f"[warn] Could not open benchmark CSV at {csv_path} for incremental writes: {exc}")
        csv_file = None
        csv_writer = None

    # Inlet BC ramps up until t=2.0 (see inlet_parabola). If the run ends before
    # that, the forcing is tiny and the tip motion will look one-signed/non-oscillatory.
    ramp_time = 2.0
    if float(time_params.final_time) < ramp_time:
        n_ramp = int(math.ceil(ramp_time / float(time_params.dt)))
        ramp_fac = 0.5 * (1.0 - math.cos(0.5 * math.pi * float(time_params.final_time)))
        u_max_now = 1.5 * float(u_mean_ref) * ramp_fac
        u_max_full = 1.5 * float(u_mean_ref)
        print(
            f"[note] final_time={float(time_params.final_time):.3f} < {ramp_time:.1f}s inflow ramp: "
            f"u_in,max≈{u_max_now:.3e} (full={u_max_full:.3e}). "
            f"Use --n-steps>={n_ramp} (dt={float(time_params.dt):g}) to reach full inflow."
        )

    rho_f_val = float(rho_f.value)
    mu_f_val = float(rho_f.value) * float(nu_f.value)
    diameter = 2.0 * float(RADIUS)
    coeff_scale = float("nan")
    if rho_f_val > 0.0 and u_mean_ref > 0.0 and diameter > 0.0:
        coeff_scale = 2.0 / (rho_f_val * (u_mean_ref**2) * diameter)

    history_records: list[dict[str, float | int]] = []
    post_state = {"step": 0, "time": 0.0}
    warned_drag_lift = False

    def _record_metrics(step_idx: int, time_val: float) -> dict[str, float | int]:
        nonlocal warned_drag_lift
        try:
            drag, lift = compute_drag_lift(dh, mesh, uk, dk, pk, rho_f_val, mu_f_val)
        except Exception as exc:
            drag = lift = float("nan")
            if not warned_drag_lift:
                warned_drag_lift = True
                print(f"Drag/lift computation skipped: {exc}")
        tip_dx, tip_dy = tip_displacement(dh, dk)
        cd = coeff_scale * drag
        cl = coeff_scale * lift
        record: dict[str, float | int] = {
            "step": int(step_idx),
            "time": float(time_val),
            "drag": float(drag),
            "lift": float(lift),
            "cd": float(cd),
            "cl": float(cl),
            "tip_dx": float(tip_dx),
            "tip_dy": float(tip_dy),
        }
        history_records.append(record)
        if csv_writer is not None and csv_file is not None:
            try:
                csv_writer.writerow(record)
                csv_file.flush()
            except Exception as exc:
                print(f"[warn] Failed to append to {csv_path}: {exc}")
        print(
            f"    step={step_idx:4d}, t={time_val:8.3f} | "
            f"Cd={cd: .6e}, Cl={cl: .6e} | tip=({tip_dx: .6e}, {tip_dy: .6e})"
        )
        return record

    def _dump(_step_idx: int) -> None:
        return

    subdomain = np.where(np.asarray(solid_bs.mask, dtype=bool), 1, 0).astype(np.int32, copy=False)

    if args.save_vtk:
        from pycutfem.io.vtk import export_vtk

        def _dump(step_idx: int) -> None:
            fname = outdir / f"fsi_{step_idx:04d}.vtu"
            u_nodes = _vector_on_mesh_nodes(dh, mesh, uk)
            d_nodes = _vector_on_mesh_nodes(dh, mesh, dk)
            export_vtk(
                str(fname),
                mesh=mesh,
                dof_handler=dh,
                functions={
                    "u": uk,
                    "d": dk,
                    "u_mag": np.linalg.norm(u_nodes, axis=1),
                    "d_mag": np.linalg.norm(d_nodes, axis=1),
                },
                cell_data={
                    "subdomain": subdomain,
                    "p_cell": _cellwise_mean_pressure(dh, pk, fluid_bs=fluid_bs),
                },
            )

        _dump(0)

    rec0 = _record_metrics(0, 0.0)
    if args.save_frames:
        save_fsi_frame(
            case_label=str(case_label),
            outdir=outdir,
            mesh=mesh,
            dh=dh,
            u=uk,
            d=dk,
            fluid_bs=fluid_bs,
            step=int(rec0["step"]),
            time=float(rec0["time"]),
            cd=float(rec0["cd"]),
            cl=float(rec0["cl"]),
            tip_dx=float(rec0["tip_dx"]),
            tip_dy=float(rec0["tip_dy"]),
            scale=float(args.frames_scale),
            dpi=int(args.frames_dpi),
        )

    def _post_step(_funcs) -> None:
        post_state["step"] += 1
        post_state["time"] += float(time_params.dt)
        k = int(post_state["step"])
        t = float(post_state["time"])
        if args.save_vtk and k % max(int(args.vtk_every), 1) == 0:
            _dump(k)
        rec = _record_metrics(k, t)
        if args.save_frames and k % max(int(args.frames_every), 1) == 0:
            save_fsi_frame(
                outdir=outdir,
                mesh=mesh,
                dh=dh,
                u=uk,
                d=dk,
                fluid_bs=fluid_bs,
                step=int(rec["step"]),
                time=float(rec["time"]),
                cd=float(rec["cd"]),
                cl=float(rec["cl"]),
                tip_dx=float(rec["tip_dx"]),
                tip_dy=float(rec["tip_dy"]),
                case_label=str(case_label),
                scale=float(args.frames_scale),
                dpi=int(args.frames_dpi),
            )

    solver.post_timeloop_cb = _post_step
    aux_funcs: Dict[str, object] = {
        "u_ux": uk.components[0],
        "u_uy": uk.components[1],
        "u_prev_ux": u_prev.components[0],
        "u_prev_uy": u_prev.components[1],
        "d_dx": dk.components[0],
        "d_dy": dk.components[1],
        "d_prev_dx": d_prev.components[0],
        "d_prev_dy": d_prev.components[1],
    }

    try:
        delta, steps, elapsed = solver.solve_time_interval(
            functions=[uk, dk, pk],
            prev_functions=[u_prev, d_prev, p_prev],
            aux_functions=aux_funcs,
            time_params=time_params,
        )
    finally:
        if csv_file is not None:
            try:
                csv_file.close()
            except Exception:
                pass

    final = history_records[-1] if history_records else {}
    print(
        f"Solved {steps} step(s) in {elapsed:.2f}s, ||ΔU||_inf={np.linalg.norm(delta, np.inf):.3e}, "
        f"Cd={float(final.get('cd', float('nan'))):.3e}, Cl={float(final.get('cl', float('nan'))):.3e}, "
        f"tip=({float(final.get('tip_dx', float('nan'))):.3e},{float(final.get('tip_dy', float('nan'))):.3e})"
    )
    print(f"Wrote benchmark time history (incremental) to {csv_path}")


if __name__ == "__main__":
    main()
