#!/usr/bin/env python
# coding: utf-8
"""
Monolithic CutFEM setup for the Turek–Hron FSI-2 benchmark.

- Geometry: channel with a *rigid* circular hole; the elastic beam is described
  by a level-set that is advected with the solid displacement.
- Mechanics: nonlinear St. Venant–Kirchhoff solid in a fully Eulerian frame
  (Cauchy stress, advective transport), incompressible Navier–Stokes for the fluid.
- The beam level set is updated every time step and the mesh is reclassified so
  curved deformations of the beam are captured.
- A small finite-difference Jacobian check is run to validate the assembled
  Jacobian against the residual.
"""
from __future__ import annotations

import math
import os
import sys
import argparse
import time
from typing import Dict, Iterable, Sequence

import numba
import numpy as np

from pycutfem.core.geometry import hansbo_cut_ratio
from pycutfem.core.levelset import BeamLevelSet, LevelSetGridFunction
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.bitset import BitSet
from pycutfem.utils.ogrid_meshgen import circular_hole_ogrid

from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    FacetNormal,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    Pos,
    Neg,
    ElementWiseConstant,
    restrict,
    grad,
    inner,
    dot,
    div,
    jump,
    Identity,
    Hessian,
    det,
    inv,
    trace,
    Jump,
)
from pycutfem.ufl.measures import dx, dGhost, dInterface
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.io.vtk import export_vtk
from pycutfem.io.visualization import plot_mesh_2
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.helpers import analyze_active_dofs
from pycutfem.core.topology import Node
from pycutfem.fem import transform
from pycutfem.solvers.nonlinear_solver import (
    NewtonParameters,
    NewtonSolver,
    TimeStepperParameters,
    _ActiveReducer,
)

# -----------------------------------------------------------------------------
# Numba configuration
# -----------------------------------------------------------------------------
try:
    num_cores = os.cpu_count()
    numba.set_num_threads(num_cores)
    print(f"Numba threads: {numba.get_num_threads()}")
except Exception:
    print("Numba not configured; continuing without thread pinning.")

_t0_global = time.perf_counter()
def _log_step(msg: str) -> None:
    t = time.perf_counter() - _t0_global
    print(f"[t={t:7.3f}s] {msg}")

# -----------------------------------------------------------------------------
# CLI / environment options
# -----------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(description="Monolithic CutFEM Turek–Hron FSI-2 (fully Eulerian solid).")
    parser.add_argument("--dt", type=float, default=float(os.getenv("DT", "0.005")), help="Time step size.")
    parser.add_argument("--poly-order", type=int, default=int(os.getenv("POLY_ORDER", "2")), help="Polynomial order for primary fields.")
    parser.add_argument("--mesh-size", type=float, default=float(os.getenv("MESH_SIZE", "0.025")), help="Target mesh size for structured O-grid.")
    parser.add_argument("--fd-backend", type=str, default=os.getenv("FD_BACKEND", "jit"), choices=["jit", "python"], help="Backend for FD Jacobian checks.")
    parser.add_argument("--output-dir", type=str, default=os.getenv("OUTPUT_DIR", "turek_results_fsi_ii_eulerian"), help="Directory for VTK output.")
    parser.add_argument("--save-vtk", dest="save_vtk", action="store_true", help="Enable VTK output.")
    parser.add_argument("--no-save-vtk", dest="save_vtk", action="store_false", help="Disable VTK output.")
    parser.add_argument("--run-fd-check", dest="run_fd_check", action="store_true", help="Run finite-difference Jacobian alignment check.")
    parser.add_argument("--no-run-fd-check", dest="run_fd_check", action="store_false", help="Skip finite-difference check.")
    parser.add_argument("--run-fd-terms", dest="run_fd_terms", action="store_true", help="Run per-term FD checks (requires --run-fd-check).")
    parser.add_argument("--no-run-fd-terms", dest="run_fd_terms", action="store_false", help="Skip per-term FD checks.")
    parser.add_argument("--run-time-stepping", dest="run_time_stepping", action="store_true", help="Run transient solve.")
    parser.add_argument("--no-run-time-stepping", dest="run_time_stepping", action="store_false", help="Skip transient solve.")
    parser.add_argument("--plot-mesh", dest="plot_mesh", action="store_true", help="Save mesh/ghost/level-set plot each step.")
    parser.add_argument("--no-plot-mesh", dest="plot_mesh", action="store_false", help="Disable mesh plotting.")
    parser.add_argument("--plot-mesh-every", type=int, default=int(os.getenv("PLOT_MESH_EVERY", "1")), help="Plot mesh every N steps when enabled.")
    parser.add_argument("--interactive-plot", dest="interactive_plot", action="store_true", help="Show interactive mesh plot with toggles.")
    parser.add_argument("--no-interactive-plot", dest="interactive_plot", action="store_false", help="Disable interactive mesh plot.")
    parser.add_argument("--plot-levelset", dest="plot_levelset", action="store_true", help="Overlay level-set zero contour on mesh plot.")
    parser.add_argument("--no-plot-levelset", dest="plot_levelset", action="store_false", help="Hide level-set on mesh plot.")
    parser.add_argument("--plot-interface-points", dest="plot_interface_points", action="store_true", help="Overlay interface points/segments on mesh plot.")
    parser.add_argument("--no-plot-interface-points", dest="plot_interface_points", action="store_false", help="Hide interface points on mesh plot.")
    parser.add_argument("--plot-show", dest="plot_show", action="store_true", help="Call plt.show() for mesh plots.")
    parser.add_argument("--no-plot-show", dest="plot_show", action="store_false", help="Do not show mesh plots (only save).")
    parser.add_argument("--plot-resolution", type=int, default=int(os.getenv("PLOT_RESOLUTION", "120")), help="Grid resolution for level-set contour in mesh plots.")
    parser.add_argument("--plot-only", dest="plot_only", action="store_true", help="Stop after plotting the initial mesh (skip JIT/solver setup).")
    parser.add_argument("--force-full-setup", dest="force_full_setup", action="store_true", help="Always build full solver even when only plotting.")
    parser.set_defaults(
        save_vtk=os.getenv("SAVE_VTK", "1") not in ("0", "false", "False"),
        run_fd_check=os.getenv("RUN_FD_CHECK", "0") != "0",
        run_fd_terms=os.getenv("RUN_FD_TERMS", "0") != "0",
        run_time_stepping=os.getenv("RUN_TIME_STEPPING", "1") != "0",
        plot_mesh=os.getenv("PLOT_MESH", "0") != "0",
        interactive_plot=os.getenv("INTERACTIVE_PLOT", "0") != "0",
        plot_levelset=os.getenv("PLOT_LEVELSET", "1") != "0",
        plot_interface_points=os.getenv("PLOT_INTERFACE_POINTS", "1") != "0",
        plot_show=os.getenv("PLOT_SHOW", "0") != "0",
        plot_only=os.getenv("PLOT_ONLY", "0") != "0",
        force_full_setup=os.getenv("FORCE_FULL_SETUP", "0") != "0",
    )
    return parser.parse_args()

ARGS = _parse_args()

# -----------------------------------------------------------------------------
# Problem parameters (Turek–Hron FSI-2)
# -----------------------------------------------------------------------------
H = 0.41
L = 2.2
RADIUS = 0.05
CENTER = (0.2, 0.2)

RHO_F = 1.0e3
MU_F = 1.0
U_MEAN = 1.0
U_MAX = 1.5 * U_MEAN

NU_S = 0.4
MU_S = 0.5e6
E_S = 2.0 * MU_S * (1.0 + NU_S)
RHO_S = 10.0e3
MU_S = E_S / (2.0 * (1.0 + NU_S))
LAMBDA_S = E_S * NU_S / ((1.0 + NU_S) * (1.0 - 2.0 * NU_S))

BEAM_LENGTH = 0.35
BEAM_HEIGHT = 0.02
BEAM_CENTER = (CENTER[0] + RADIUS + 0.5 * BEAM_LENGTH, CENTER[1])
POINT_B = (0.15, 0.2)
POINT_A_INITIAL = (0.6, 0.2) # Point A will change while Point B is fixed

BETA_PENALTY = 90.0 * MU_F
DT = float(ARGS.dt)
POLY_ORDER = int(ARGS.poly_order)
MESH_SIZE = float(ARGS.mesh_size)
FD_BACKEND = ARGS.fd_backend

# -----------------------------------------------------------------------------
# Mesh and boundary helpers
# -----------------------------------------------------------------------------


def _count_segments(width: float, mesh_size: float, *, min_cells: int = 1) -> int:
    if width <= 1.0e-12:
        return 0
    return max(min_cells, int(math.ceil(width / mesh_size)))


def build_structured_channel_mesh(mesh_size: float, poly_order: int) -> Mesh:
    """
    Structured O-grid mesh of the channel with a circular hole.
    """
    margin = max(2.5 * mesh_size, 0.015)
    half_x_cap = min(CENTER[0], L - CENTER[0]) - margin
    half_y_cap = min(CENTER[1], H - CENTER[1]) - margin
    if half_x_cap <= RADIUS or half_y_cap <= RADIUS:
        raise RuntimeError("O-grid collapsed: decrease mesh size or adjust parameters.")
    hx = half_x_cap
    hy = half_y_cap
    ring_thickness = min(hx, hy) - RADIUS
    if ring_thickness <= 0.0:
        raise RuntimeError("Ring thickness must be positive for the structured mesh.")

    x_inner_left = CENTER[0] - hx
    x_inner_right = CENTER[0] + hx
    y_inner_bottom = CENTER[1] - hy
    y_inner_top = CENTER[1] + hy

    nx_left = _count_segments(x_inner_left - 0.0, mesh_size, min_cells=1)
    nx_right = _count_segments(L - x_inner_right, mesh_size, min_cells=1)
    nx_mid = _count_segments(x_inner_right - x_inner_left, mesh_size, min_cells=4)
    if nx_mid % 2:
        nx_mid += 1

    ny_bottom = _count_segments(y_inner_bottom - 0.0, mesh_size, min_cells=1)
    ny_top = _count_segments(H - y_inner_top, mesh_size, min_cells=1)
    ny_mid = _count_segments(y_inner_top - y_inner_bottom, mesh_size, min_cells=4)
    if ny_mid % 2:
        ny_mid += 1

    n_radial_layers = max(2, _count_segments(ring_thickness, mesh_size, min_cells=2))

    nodes, elements, edges, corners = circular_hole_ogrid(
        L,
        H,
        circle_center=CENTER,
        circle_radius=RADIUS,
        ring_thickness=ring_thickness,
        n_radial_layers=n_radial_layers,
        nx_outer=(nx_left, nx_mid, nx_right),
        ny_outer=(ny_bottom, ny_mid, ny_top),
        poly_order=poly_order,
        outer_box_half_lengths=(hx, hy),
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elements,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )
    tag_channel_boundaries(mesh, mesh_size)
    return mesh


def tag_channel_boundaries(mesh: Mesh, mesh_size: float) -> None:
    """
    Tag inlet, outlet, walls, and the rigid cylinder boundary.
    """
    tol = 1.0e-9
    circle_tol = max(0.25 * mesh_size, 1.0e-4)
    rect_locators = {
        "inlet": lambda x, y: abs(x - 0.0) <= tol,
        "outlet": lambda x, y: abs(x - L) <= tol,
        "walls": lambda x, y: abs(y - 0.0) <= tol or abs(y - H) <= tol,
    }
    mesh.tag_boundary_edges(rect_locators)

    def on_circle(x: float, y: float) -> bool:
        return abs(math.hypot(x - CENTER[0], y - CENTER[1]) - RADIUS) <= circle_tol

    for edge in mesh.edges_list:
        if edge.right is not None:
            continue
        mpx, mpy = mesh.nodes_x_y_pos[list(edge.nodes)].mean(axis=0)
        if on_circle(mpx, mpy):
            edge.tag = "cylinder"
    # cache
    cyl_mask = np.fromiter((getattr(e, "tag", "") == "cylinder" for e in mesh.edges_list), bool)
    mesh._edge_bitsets = getattr(mesh, "_edge_bitsets", {})
    mesh._edge_bitsets["cylinder"] = BitSet(cyl_mask)
    loc_map = getattr(mesh, "_boundary_locators", {})
    loc_map["cylinder"] = on_circle
    mesh._boundary_locators = loc_map


# -----------------------------------------------------------------------------
# Localized asymmetric refinement around the beam (produces hanging nodes)
# -----------------------------------------------------------------------------
def _quad_corner_indices(p: int) -> tuple[int, int, int, int]:
    """Return (bl, br, tr, tl) local indices in lattice order (eta outer, xi inner)."""
    n = p + 1
    bl = 0
    br = p
    tr = p * n + p
    tl = p * n
    return bl, br, tr, tl


def _refine_element_quads(mesh: Mesh, eid: int, orientation: str, nodes, node_lookup) -> tuple[list[list[int]], list[list[int]]]:
    """
    Split one quad element into 2 children (vertical or horizontal).
    Returns (child_connectivity, child_corners).
    """
    p = mesh.poly_order
    t = np.linspace(-1.0, 1.0, p + 1)
    parent_conn = mesh.elements_connectivity[eid]

    def _parent_node(xi_p: float, eta_p: float) -> int | None:
        ix = np.where(np.isclose(t, xi_p, atol=1e-12))[0]
        iy = np.where(np.isclose(t, eta_p, atol=1e-12))[0]
        if ix.size and iy.size:
            idx = int(iy[0] * (p + 1) + ix[0])
            return int(parent_conn[idx])
        return None

    def _get_node(xi_p: float, eta_p: float) -> int:
        # Reuse parent nodes when possible; otherwise create/reuse global by coordinate.
        nid = _parent_node(xi_p, eta_p)
        if nid is not None:
            return nid
        x_phys = transform.x_mapping(mesh, eid, (float(xi_p), float(eta_p)))
        key = (float(round(x_phys[0], 14)), float(round(x_phys[1], 14)))
        nid = node_lookup.get(key)
        if nid is not None:
            return nid
        nid = len(nodes)
        node_lookup[key] = nid
        nodes.append(Node(nid, float(x_phys[0]), float(x_phys[1])))
        return nid

    def _child(refine_mode: str):
        conn = []
        xi_child = t
        eta_child = t
        for eta in eta_child:
            for xi in xi_child:
                if refine_mode == "left":
                    xi_p = 0.5 * (xi - 1.0)
                    eta_p = eta
                elif refine_mode == "right":
                    xi_p = 0.5 * (xi + 1.0)
                    eta_p = eta
                elif refine_mode == "bottom":
                    xi_p = xi
                    eta_p = 0.5 * (eta - 1.0)
                elif refine_mode == "top":
                    xi_p = xi
                    eta_p = 0.5 * (eta + 1.0)
                else:
                    raise ValueError(refine_mode)
                conn.append(_get_node(xi_p, eta_p))
        bl, br, tr, tl = _quad_corner_indices(p)
        corners = [conn[bl], conn[br], conn[tr], conn[tl]]
        return conn, corners

    if orientation == "vertical":
        c1, corners1 = _child("left")
        c2, corners2 = _child("right")
        return [c1, c2], [corners1, corners2]
    else:
        c1, corners1 = _child("bottom")
        c2, corners2 = _child("top")
        return [c1, c2], [corners1, corners2]


def asymmetric_refine_around_beam(mesh: Mesh, beam_ls: BeamLevelSet, levels: int = 2, band: float | None = None, splits_per_mark: int = 2) -> Mesh:
    """
    Refine quads touching the beam level set with orientation bias:
    left half → vertical split, right half → horizontal split.
    Produces hanging nodes that are handled by the constraint layer.
    """
    if mesh.element_type != "quad":
        return mesh

    band = band or max(3.0 * beam_ls.hy, 4.0 * MESH_SIZE)
    center_x = float(getattr(beam_ls, "cx", BEAM_CENTER[0]))
    beam_xmin = float(beam_ls.cx - beam_ls.hx)
    beam_xmax = float(beam_ls.cx + beam_ls.hx)
    beam_ymin = float(beam_ls.cy - beam_ls.hy)
    beam_ymax = float(beam_ls.cy + beam_ls.hy)

    marked = set()
    t0_mark = time.perf_counter()
    for elem in mesh.elements_list:
        corners = mesh.nodes_x_y_pos[list(elem.corner_nodes)]
        phi_corner = beam_ls(corners)
        hits_phi = np.any(phi_corner <= 0.0) or np.any(np.abs(phi_corner) <= band)
        ex_min, ey_min = corners.min(axis=0)
        ex_max, ey_max = corners.max(axis=0)
        hits_bbox = (ex_min <= beam_xmax + band and ex_max >= beam_xmin - band and ey_min <= beam_ymax + band and ey_max >= beam_ymin - band)
        if hits_phi or hits_bbox:
            marked.add(elem.id)
    print(f"[refine_beam] mark phase: {len(marked)} elems in {time.perf_counter()-t0_mark:.3f}s")
    for _ in range(max(0, levels - 1)):
        new = set()
        for eid in marked:
            for nb in mesh._neighbors[eid]:
                if nb is not None:
                    new.add(int(nb))
        marked |= new
    print(f"[refine_beam] neighbor expansion to {len(marked)} elems (levels={levels})")

    if not marked:
        return mesh

    # Cap splits_per_mark to avoid runaway refinement
    splits_per_mark = max(1, min(int(splits_per_mark), 4))

    nodes = list(mesh.nodes_list)
    node_lookup = {(round(nd.x, 14), round(nd.y, 14)): int(nd.id) for nd in nodes}
    new_elems = []
    new_corners = []

    t0_split = time.perf_counter()
    for eid, elem in enumerate(mesh.elements_list):
        if eid not in marked:
            new_elems.append(list(mesh.elements_connectivity[eid]))
            new_corners.append(list(mesh.corner_connectivity[eid]))
            continue
        cx, _ = elem.centroid()
        orient_primary = "vertical" if cx <= center_x else "horizontal"
        conns_lvl1, corners_lvl1 = _refine_element_quads(mesh, eid, orient_primary, nodes, node_lookup)

        if splits_per_mark <= 1:
            new_elems.extend(conns_lvl1)
            new_corners.extend(corners_lvl1)
            continue

        current = list(zip(conns_lvl1, corners_lvl1))
        for _ in range(1, splits_per_mark):
            next_level = []
            orient_secondary = "horizontal" if orient_primary == "vertical" else "vertical"
            bl, br, tr, tl = _quad_corner_indices(mesh.poly_order)
            for conn_child, _ in current:
                child_corners = [conn_child[bl], conn_child[br], conn_child[tr], conn_child[tl]]
                tmp_mesh = Mesh(
                    nodes=nodes,
                    element_connectivity=np.asarray([conn_child], dtype=int),
                    elements_corner_nodes=np.asarray([child_corners], dtype=int),
                    element_type="quad",
                    poly_order=mesh.poly_order,
                )
                tmp_conn, tmp_corners = _refine_element_quads(tmp_mesh, 0, orient_secondary, nodes, node_lookup)
                next_level.extend(zip(tmp_conn, tmp_corners))
            current = next_level
        for conn_child, corners_child in current:
            new_elems.append(conn_child)
            new_corners.append(corners_child)
    print(f"[refine_beam] split phase produced {len(new_elems)} elems, {len(nodes)} nodes in {time.perf_counter()-t0_split:.3f}s")

    new_mesh = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(new_elems, dtype=int),
        elements_corner_nodes=np.asarray(new_corners, dtype=int),
        element_type="quad",
        poly_order=mesh.poly_order,
    )
    tag_channel_boundaries(new_mesh, MESH_SIZE)
    print(f"[refine_beam] marked {len(marked)} elems → {len(new_elems)} elements, {len(nodes)} nodes (band={band:.4f})")
    return new_mesh


# -----------------------------------------------------------------------------
# Optimized asymmetric refinement (precomputed grids, no temporary Mesh)
# -----------------------------------------------------------------------------
def _precompute_child_parametric_fast(p: int) -> dict[str, np.ndarray]:
    t = np.linspace(-1.0, 1.0, p + 1)
    grids = {}
    for ref in ("left", "right", "bottom", "top"):
        xi_list = []
        eta_list = []
        for eta in t:
            for xi in t:
                if ref == "left":
                    xi_p = 0.5 * (xi - 1.0)
                    eta_p = eta
                elif ref == "right":
                    xi_p = 0.5 * (xi + 1.0)
                    eta_p = eta
                elif ref == "bottom":
                    xi_p = xi
                    eta_p = 0.5 * (eta - 1.0)
                elif ref == "top":
                    xi_p = xi
                    eta_p = 0.5 * (eta + 1.0)
                xi_list.append(xi_p)
                eta_list.append(eta_p)
        grids[ref] = np.column_stack([np.asarray(xi_list), np.asarray(eta_list)])
    return grids


def _apply_sequence_to_grid(base_grid: np.ndarray, sequence: list[str]) -> np.ndarray:
    xi_eta = base_grid.copy()
    for op in sequence:
        if op == "left":
            xi_eta[:, 0] = 0.5 * (xi_eta[:, 0] - 1.0)
        elif op == "right":
            xi_eta[:, 0] = 0.5 * (xi_eta[:, 0] + 1.0)
        elif op == "bottom":
            xi_eta[:, 1] = 0.5 * (xi_eta[:, 1] - 1.0)
        elif op == "top":
            xi_eta[:, 1] = 0.5 * (xi_eta[:, 1] + 1.0)
    return xi_eta


def _generate_sequences(primary_orientation: str, depth: int) -> list[list[str]]:
    if primary_orientation == "vertical":
        seqs = [["left"], ["right"]]
    else:
        seqs = [["bottom"], ["top"]]
    orient = primary_orientation
    for _ in range(1, depth):
        orient = "horizontal" if orient == "vertical" else "vertical"
        ops = ["bottom", "top"] if orient == "horizontal" else ["left", "right"]
        seqs = [s + [op] for s in seqs for op in ops]
    return seqs


def mark_elements_near_levelset(mesh: Mesh, level_set, band: float, levels: int = 1) -> set[int]:
    """Mark elements whose corner box intersects the level set band; expand to neighbor layers."""
    marked: set[int] = set()
    corners_all = mesh.nodes_x_y_pos[mesh.corner_connectivity]
    phi_corners = level_set(corners_all.reshape(-1, 2)).reshape(corners_all.shape[:-1])
    ex_min = corners_all[..., 0].min(axis=1)
    ex_max = corners_all[..., 0].max(axis=1)
    ey_min = corners_all[..., 1].min(axis=1)
    ey_max = corners_all[..., 1].max(axis=1)

    cx = getattr(level_set, "cx", None)
    hx = getattr(level_set, "hx", None)
    cy = getattr(level_set, "cy", None)
    hy = getattr(level_set, "hy", None)
    if cx is not None and hx is not None and cy is not None and hy is not None:
        beam_xmin = float(cx - hx)
        beam_xmax = float(cx + hx)
        beam_ymin = float(cy - hy)
        beam_ymax = float(cy + hy)
    else:
        beam_xmin = beam_xmax = beam_ymin = beam_ymax = 0.0

    for eid, (phi, xmin, xmax, ymin, ymax) in enumerate(zip(phi_corners, ex_min, ex_max, ey_min, ey_max)):
        hits_phi = np.any(phi <= 0.0) or np.any(np.abs(phi) <= band)
        hits_bbox = (xmin <= beam_xmax + band and xmax >= beam_xmin - band and ymin <= beam_ymax + band and ymax >= beam_ymin - band)
        if hits_phi or hits_bbox:
            marked.add(int(eid))

    for _ in range(max(0, levels - 1)):
        new = set()
        for eid in marked:
            for nb in mesh._neighbors[eid]:
                if nb is not None:
                    new.add(int(nb))
        marked |= new
    return marked


def _precompute_child_parametric_fast(p: int) -> dict[str, np.ndarray]:
    t = np.linspace(-1.0, 1.0, p + 1)
    grids = {}
    for ref in ("left", "right", "bottom", "top"):
        xi_list = []
        eta_list = []
        for eta in t:
            for xi in t:
                if ref == "left":
                    xi_p = 0.5 * (xi - 1.0)
                    eta_p = eta
                elif ref == "right":
                    xi_p = 0.5 * (xi + 1.0)
                    eta_p = eta
                elif ref == "bottom":
                    xi_p = xi
                    eta_p = 0.5 * (eta - 1.0)
                elif ref == "top":
                    xi_p = xi
                    eta_p = 0.5 * (eta + 1.0)
                xi_list.append(xi_p)
                eta_list.append(eta_p)
        grids[ref] = np.column_stack([np.asarray(xi_list), np.asarray(eta_list)])
    return grids


def _apply_sequence_to_grid(base_grid: np.ndarray, sequence: list[str]) -> np.ndarray:
    xi_eta = base_grid.copy()
    for op in sequence:
        if op == "left":
            xi_eta[:, 0] = 0.5 * (xi_eta[:, 0] - 1.0)
        elif op == "right":
            xi_eta[:, 0] = 0.5 * (xi_eta[:, 0] + 1.0)
        elif op == "bottom":
            xi_eta[:, 1] = 0.5 * (xi_eta[:, 1] - 1.0)
        elif op == "top":
            xi_eta[:, 1] = 0.5 * (xi_eta[:, 1] + 1.0)
    return xi_eta


def _generate_sequences(primary_orientation: str, depth: int) -> list[list[str]]:
    if primary_orientation == "vertical":
        seqs = [["left"], ["right"]]
    else:
        seqs = [["bottom"], ["top"]]
    orient = primary_orientation
    for _ in range(1, depth):
        orient = "horizontal" if orient == "vertical" else "vertical"
        ops = ["bottom", "top"] if orient == "horizontal" else ["left", "right"]
        seqs = [s + [op] for s in seqs for op in ops]
    return seqs


def _refine_element_with_sequences_fast(mesh: Mesh, eid: int, sequences: list[list[str]], nodes, node_lookup, base_grid: np.ndarray, xi_to_idx, bl, br, tr, tl) -> tuple[list[list[int]], list[list[int]]]:
    conns_out: list[list[int]] = []
    corners_out: list[list[int]] = []

    def _get_node(xi_p: float, eta_p: float) -> int:
        key_pe = (float(np.round(xi_p, 12)), float(np.round(eta_p, 12)))
        idx = xi_to_idx.get(key_pe)
        if idx is not None:
            return int(mesh.elements_connectivity[eid][idx])
        x_phys = transform.x_mapping(mesh, eid, (float(xi_p), float(eta_p)))
        key = (float(round(x_phys[0], 14)), float(round(x_phys[1], 14)))
        nid = node_lookup.get(key)
        if nid is not None:
            return nid
        nid = len(nodes)
        node_lookup[key] = nid
        nodes.append(Node(nid, float(x_phys[0]), float(x_phys[1])))
        return nid

    for seq in sequences:
        grid = _apply_sequence_to_grid(base_grid, seq)
        conn = [_get_node(float(xi), float(eta)) for xi, eta in grid]
        corners = [conn[bl], conn[br], conn[tr], conn[tl]]
        conns_out.append(conn)
        corners_out.append(corners)
    return conns_out, corners_out


def asymmetric_refine_around_beam_fast(mesh: Mesh, beam_ls: BeamLevelSet, levels: int = 2, band: float | None = None, splits_per_mark: int = 2) -> Mesh:
    """Optimized refinement around the beam using precomputed param grids."""
    if mesh.element_type != "quad":
        return mesh
    band = band or max(3.0 * beam_ls.hy, 4.0 * MESH_SIZE)
    center_x = float(getattr(beam_ls, "cx", BEAM_CENTER[0]))
    splits_per_mark = max(1, min(int(splits_per_mark), 3))

    marked = mark_elements_near_levelset(mesh, beam_ls, band=band, levels=levels)
    print(f"[refine_beam_fast] total marked {len(marked)} elems (band={band:.4f}, levels={levels})")
    if not marked:
        return mesh

    nodes = list(mesh.nodes_list)
    node_lookup = {(round(nd.x, 14), round(nd.y, 14)): int(nd.id) for nd in nodes}
    new_elems: list[list[int]] = []
    new_corners: list[list[int]] = []

    p = mesh.poly_order
    t = np.linspace(-1.0, 1.0, p + 1)
    xi_to_idx_template = {(float(xi), float(eta)): int(j * (p + 1) + i) for j, eta in enumerate(t) for i, xi in enumerate(t)}
    bl, br, tr, tl = _quad_corner_indices(p)
    base_grid_left = _precompute_child_parametric_fast(p)["left"]

    t0_split = time.perf_counter()
    for eid, elem in enumerate(mesh.elements_list):
        if eid not in marked:
            new_elems.append(list(mesh.elements_connectivity[eid]))
            new_corners.append(list(mesh.corner_connectivity[eid]))
            continue
        cx, _ = elem.centroid()
        primary = "vertical" if cx <= center_x else "horizontal"
        sequences = _generate_sequences(primary, splits_per_mark)
        conns, corners = _refine_element_with_sequences_fast(
            mesh, eid, sequences, nodes, node_lookup, base_grid_left, xi_to_idx_template, bl, br, tr, tl
        )
        new_elems.extend(conns)
        new_corners.extend(corners)
    print(f"[refine_beam_fast] split produced {len(new_elems)} elems, {len(nodes)} nodes in {time.perf_counter()-t0_split:.3f}s")

    new_mesh = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(new_elems, dtype=int),
        elements_corner_nodes=np.asarray(new_corners, dtype=int),
        element_type="quad",
        poly_order=mesh.poly_order,
    )
    tag_channel_boundaries(new_mesh, MESH_SIZE)
    print(f"[refine_beam_fast] final mesh: {len(new_mesh.elements_list)} elems, {len(new_mesh.nodes_list)} nodes")
    return new_mesh


# -----------------------------------------------------------------------------
# Level-set update utilities
# -----------------------------------------------------------------------------


def update_beam_levelset_from_displacement(
    ls_beam: LevelSetGridFunction, disp_vec: VectorFunction, beam_ref_ls: BeamLevelSet
) -> None:
    """
    Advect the beam LevelSetGridFunction with the current solid displacement:
    φ^{k+1}(x_node) = φ_ref(x_node - d^{k+1}(x_node)).
    """
    dh_ls = ls_beam.dh
    mesh = dh_ls.mixed_element.mesh
    disp_dh = disp_vec._dof_handler

    gphi = np.asarray(dh_ls.get_field_slice(ls_beam.field), dtype=int)
    node_ids = np.array([dh_ls._dof_to_node_map[int(gd)][1] for gd in gphi], dtype=int)

    disp_x = disp_vec.components[0]
    disp_y = disp_vec.components[1]
    phi_vals = ls_beam.nodal_values()

    for gd_phi, nid in zip(gphi, node_ids):
        x, y = mesh.nodes_x_y_pos[int(nid)]
        gd_dx = disp_dh.dof_map["d_neg_x"].get(int(nid), None)
        gd_dy = disp_dh.dof_map["d_neg_y"].get(int(nid), None)

        ux = disp_x.get_nodal_values(np.array([gd_dx], dtype=int))[0] if gd_dx is not None else 0.0
        uy = disp_y.get_nodal_values(np.array([gd_dy], dtype=int))[0] if gd_dy is not None else 0.0

        X_ref = np.array([x - ux, y - uy], float)
        phi_new = beam_ref_ls(X_ref)

        li = ls_beam._g2l[int(gd_phi)]
        phi_vals[int(li)] = float(phi_new)

    ls_beam.commit()


def _copy_bitset(bs: BitSet) -> BitSet:
    return BitSet(np.array(bs.mask, dtype=bool))


def make_domain_sets(mesh: Mesh) -> Dict[str, BitSet]:
    fluid = mesh.element_bitset("outside")
    solid = mesh.element_bitset("inside")
    cut = mesh.element_bitset("cut")
    fluid_ifc = fluid | cut
    solid_ifc = solid | cut
    has_pos = fluid | cut
    has_neg = solid | cut
    # By convention include the true interface edges in both ghost sets
    # (they are common to pos/neg ghost layers). ghost_both contains edges
    # that are shared by both ghost layers and should be present on each side.
    ghost_both = mesh.edge_bitset("ghost_both")
    solid_ghost = mesh.edge_bitset("ghost_neg") | ghost_both | mesh.edge_bitset("interface")
    fluid_ghost = mesh.edge_bitset("ghost_pos") | ghost_both | mesh.edge_bitset("interface")
    return {
        "fluid_domain": _copy_bitset(fluid),
        "solid_domain": _copy_bitset(solid),
        "cut_domain": _copy_bitset(cut),
        "fluid_interface": _copy_bitset(fluid_ifc),
        "solid_interface": _copy_bitset(solid_ifc),
        "has_pos": _copy_bitset(has_pos),
        "has_neg": _copy_bitset(has_neg),
        "solid_ghost": _copy_bitset(solid_ghost),
        "fluid_ghost": _copy_bitset(fluid_ghost),
    }


def _update_bs(target: BitSet, new_mask: np.ndarray) -> None:
    target.mask[...] = new_mask.astype(bool)


def refresh_domains(mesh: Mesh, domains: Dict[str, BitSet]) -> None:
    fluid = mesh.element_bitset("outside")
    solid = mesh.element_bitset("inside")
    cut = mesh.element_bitset("cut")
    _update_bs(domains["fluid_domain"], fluid.mask)
    _update_bs(domains["solid_domain"], solid.mask)
    _update_bs(domains["cut_domain"], cut.mask)
    _update_bs(domains["fluid_interface"], fluid.mask | cut.mask)
    _update_bs(domains["solid_interface"], solid.mask | cut.mask)
    _update_bs(domains["has_pos"], fluid.mask | cut.mask)
    _update_bs(domains["has_neg"], solid.mask | cut.mask)
    ghost_both = mesh.edge_bitset("ghost_both")
    solid_ghost = mesh.edge_bitset("ghost_neg") | ghost_both | mesh.edge_bitset("interface")
    fluid_ghost = mesh.edge_bitset("ghost_pos") | ghost_both | mesh.edge_bitset("interface")
    _update_bs(domains["solid_ghost"], solid_ghost.mask)
    _update_bs(domains["fluid_ghost"], fluid_ghost.mask)


def refresh_hansbo_kappa(
    mesh: Mesh,
    level_set,
    theta_pos_vals: np.ndarray,
    theta_neg_vals: np.ndarray,
    theta_min: float = 1.0e-3,
) -> None:
    theta_pos_vals[:] = np.clip(hansbo_cut_ratio(mesh, level_set, side="+"), theta_min, 1.0)
    theta_neg_vals[:] = np.clip(hansbo_cut_ratio(mesh, level_set, side="-"), theta_min, 1.0)


def retag_inactive(dh: DofHandler) -> None:
    dh.dof_tags["inactive"] = set()
    dh.tag_dofs_from_element_bitset("inactive", "u_pos_x", "inside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "u_pos_y", "inside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "p_pos_", "inside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "vs_neg_x", "outside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "vs_neg_y", "outside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "d_neg_x", "outside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "d_neg_y", "outside", strict=True)


def recompute_active_dofs(solver: NewtonSolver, bcs_active: Sequence[BoundaryCondition]) -> None:
    dh = solver.dh
    ndof = dh.total_dofs
    active_by_restr, has_restriction = analyze_active_dofs(solver.equation, dh, solver.me, bcs_active)
    bc_dofs = set(dh.get_dirichlet_data(bcs_active).keys())
    candidate = set(active_by_restr) if has_restriction else set(range(ndof))
    inactive = set(dh.dof_tags.get("inactive", set()))
    inactive_free = inactive - bc_dofs
    free = sorted((candidate - bc_dofs) - inactive_free)
    solver.active_dofs = np.asarray(free, dtype=int)
    solver.full_to_red = -np.ones(ndof, dtype=int)
    solver.full_to_red[solver.active_dofs] = np.arange(len(solver.active_dofs), dtype=int)
    solver.red_to_full = solver.active_dofs
    solver.use_reduced = len(solver.active_dofs) < ndof
    solver.restrictor = _ActiveReducer(dh, solver.active_dofs)
    solver._pattern_stale = True


# -----------------------------------------------------------------------------
# Finite-difference Jacobian check
# -----------------------------------------------------------------------------


def select_fd_dofs(dh: DofHandler, fields_to_probe: Dict[str, int], elem_tag: str = "cut") -> np.ndarray:
    selected: list[int] = []
    elems = dh.element_bitset(elem_tag).to_indices()
    probe_eid = int(elems[0]) if len(elems) else 0
    for field, count in fields_to_probe.items():
        try:
            local = dh.element_dofs(field, probe_eid)
        except Exception:
            local = []
        selected.extend(list(local[:count]))
    return np.array(sorted(set(selected)), dtype=int)


def finite_difference_check(
    jac_form,
    res_form,
    dh: DofHandler,
    bcs: Sequence[BoundaryCondition],
    functions: Dict[str, VectorFunction | Function],
    probe_dofs: Iterable[int],
    eps: float = 1.0e-6,
) -> None:
    backend = FD_BACKEND
    compiler = FormCompiler(dh, backend=backend)
    eq = Equation(jac_form, res_form)
    base_K, base_R = compiler.assemble(eq, bcs=bcs)
    if base_K is None or base_R is None:
        print("Skipping FD check: Jacobian or residual form missing.")
        return

    def perturb(field: str, gdof: int, new_value: float) -> float:
        func = functions[field]
        old = func.get_nodal_values(np.array([gdof], dtype=int))[0]
        func.set_nodal_values(np.array([gdof], dtype=int), np.array([new_value], dtype=float))
        return old

    rows = []
    bc_dofs = set(dh.get_dirichlet_data(bcs).keys())
    inactive = set(dh.dof_tags.get("inactive", set()))
    for gdof in probe_dofs:
        field, _ = dh._dof_to_node_map[int(gdof)]
        if field not in functions:
            continue
        if int(gdof) in bc_dofs:
            continue
        if int(gdof) in inactive:
            continue
        old_val = functions[field].get_nodal_values(np.array([gdof], dtype=int))[0]
        perturb(field, int(gdof), old_val + eps)
        K_plus, R_plus = compiler.assemble(eq, bcs=bcs)
        perturb(field, int(gdof), old_val - eps)
        K_minus, R_minus = compiler.assemble(eq, bcs=bcs)
        perturb(field, int(gdof), old_val)
        fd_col = (R_plus - R_minus) / (2 * eps)
        jac_col = base_K[:, int(gdof)].toarray().ravel()
        err_vec = fd_col - jac_col
        err = np.linalg.norm(err_vec, ord=np.inf)
        mag = np.linalg.norm(jac_col, ord=np.inf)
        rel = err / (mag + 1.0e-14)
        rows.append((gdof, field, err, mag, rel))
    print("Finite-difference Jacobian check (gdof, field, err, |J|, rel):")
    for gd, fld, err, mag, rel in rows:
        print(f"  {gd:5d}  {fld:10s}  err={err:9.3e}  |J|={mag:9.3e}  rel={rel:9.3e}")


# -----------------------------------------------------------------------------
# Main setup
# -----------------------------------------------------------------------------
print("--- Setting up the Turek–Hron FSI-2 benchmark ---")
Re = RHO_F * U_MAX * (2 * RADIUS) / MU_F
print(f"Reynolds number: {Re:.2f}")
_log_step("start setup")

# Mesh with rigid hole
mesh = build_structured_channel_mesh(MESH_SIZE, POLY_ORDER)
_log_step("built base mesh")

# Beam level set (reference configuration)
beam_ref_ls = BeamLevelSet(center=BEAM_CENTER, Lb=BEAM_LENGTH, Hb=BEAM_HEIGHT)
# Local refinement around the beam to capture the interface with hanging nodes
mesh = asymmetric_refine_around_beam_fast(mesh, beam_ref_ls, levels=2, band=None, splits_per_mark=2)
_log_step("refined mesh around beam")

# Use higher order for φ_beam to reduce geometric distortion of the zero set
ls_me = MixedElement(mesh, field_specs={"phi_beam": POLY_ORDER})
ls_dh = DofHandler(ls_me, method="cg")
ls_beam = LevelSetGridFunction(ls_dh, field="phi_beam")
ls_beam.interpolate(lambda x, y: beam_ref_ls(np.array([x, y])))
ls_beam.commit()
_log_step("interpolated/committed level set")

domains = make_domain_sets(mesh)
inside_elems = mesh.element_bitset("inside").to_indices()
if inside_elems.size:
    x_in = mesh.nodes_x_y_pos[np.unique(np.concatenate([mesh.corner_connectivity[i] for i in inside_elems]))][:, 0]
    print(f"Inside elements: {inside_elems.size}, x-span=({x_in.min():.3f},{x_in.max():.3f})")
else:
    print("Inside elements: 0")
print(
    f"Ghost edges: total={mesh.edge_bitset('ghost').cardinality()}, "
    f"pos={mesh.edge_bitset('ghost_pos').cardinality()}, "
    f"neg={mesh.edge_bitset('ghost_neg').cardinality()}, "
    f"both={mesh.edge_bitset('ghost_both').cardinality()}, "
    f"fluid_ghost(defined_on)={domains['fluid_ghost'].cardinality()}, "
    f"solid_ghost(defined_on)={domains['solid_ghost'].cardinality()}"
)
_log_step("built domain sets / ghost counts")

# Prepare output dir before any early exits
output_dir = ARGS.output_dir
os.makedirs(output_dir, exist_ok=True)

# Fast path: only produce an initial plot and exit (skip JIT/solver setup).
quick_plot_only = (
    ARGS.plot_mesh
    and (ARGS.plot_only or (not ARGS.run_time_stepping and not ARGS.run_fd_check and not ARGS.run_fd_terms))
    and not ARGS.force_full_setup
)
if quick_plot_only:
    import matplotlib.pyplot as plt
    from pycutfem.io.visualization import plot_mesh_2

    fig, ax = plt.subplots(figsize=(10, 8))
    level_set_for_plot = beam_ref_ls if ARGS.plot_levelset else None
    plot_mesh_2(
        mesh,
        level_set=level_set_for_plot,
        plot_nodes=True,
        plot_edges=True,
        elem_tags=True,
        edge_colors=True,
        show=False,
        ax=ax,
        resolution=max(20, int(ARGS.plot_resolution)),
    )
    ax.set_title("Initial mesh (plot-only)")
    fname = os.path.join(output_dir, f"mesh_{0:04d}.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"[plot-only] saved {fname} and exiting (skipped solver setup)")
    if ARGS.plot_show or ARGS.interactive_plot:
        plt.show()
    else:
        plt.close(fig)
    sys.exit(0)

# Mixed element for fluid/solid unknowns
mixed_element = MixedElement(
    mesh,
    field_specs={
        "u_pos_x": POLY_ORDER,
        "u_pos_y": POLY_ORDER,
        "p_pos_": POLY_ORDER - 1,
        "vs_neg_x": POLY_ORDER - 1,
        "vs_neg_y": POLY_ORDER - 1,
        "d_neg_x": POLY_ORDER - 1,
        "d_neg_y": POLY_ORDER - 1,
    },
)
dof_handler = DofHandler(mixed_element, method="cg")

# Boundary conditions
def parabolic_inflow(x, y, t=None):
    """
    Parabolic inflow ramped in time:
        v_in(t,0,y) = v_base(y) * 0.5*(1 - cos(pi/2 * t))  for t < 2
                    = v_base(y)                             otherwise
    """
    v_base = 1.5 * 4 * U_MEAN * y * (H - y) / (H**2)
    if t is None:
        return v_base
    if t < 2.0:
        return v_base * 0.5 * (1.0 - math.cos(0.5 * math.pi * t))
    return v_base


bcs: list[BoundaryCondition] = [
    BoundaryCondition("u_pos_x", "dirichlet", "inlet", parabolic_inflow),
    BoundaryCondition("u_pos_y", "dirichlet", "inlet", lambda x, y: 0.0),
    BoundaryCondition("u_pos_x", "dirichlet", "walls", lambda x, y: 0.0),
    BoundaryCondition("u_pos_y", "dirichlet", "walls", lambda x, y: 0.0),
    BoundaryCondition("u_pos_x", "dirichlet", "cylinder", lambda x, y: 0.0),
    BoundaryCondition("u_pos_y", "dirichlet", "cylinder", lambda x, y: 0.0),
]

# Pressure pin to remove nullspace
# pin_tag = "pressure_pin"
# dof_handler.tag_dof_by_locator(
#     pin_tag,
#     "p_pos_",
#     locator=lambda x, y: abs(x - L) <= 1.0e-9 and abs(y - 0.5 * H) <= 1.0e-3,
#     find_first=True,
# )
# bcs.append(BoundaryCondition("p_pos_", "dirichlet", pin_tag, lambda x, y: 0.0))

# Clamp beam at the circle interface
beam_clamp_tag = "beam_root"
beam_root_locator = lambda x, y: x <= CENTER[0] + RADIUS + 1.0e-6 and abs(y - CENTER[1]) <= 0.5 * BEAM_HEIGHT
for field in ["vs_neg_x", "vs_neg_y", "d_neg_x", "d_neg_y"]:
    dof_handler.tag_dof_by_locator(
        beam_clamp_tag,
        field,
        locator=beam_root_locator,
        find_first=False,
    )
    bcs.append(BoundaryCondition(field, "dirichlet", beam_clamp_tag, lambda x, y: 0.0))

bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

# Tag inactive DOFs according to current classification
retag_inactive(dof_handler)

print(f"Interface edges: {mesh.edge_bitset('interface').cardinality()}")
print(f"Cut elements:    {mesh.element_bitset('cut').cardinality()}")

# -----------------------------------------------------------------------------
# Function spaces and unknowns
# -----------------------------------------------------------------------------
velocity_fluid_space = FunctionSpace(name="velocity_fluid", field_names=["u_pos_x", "u_pos_y"], dim=1, side="+")
pressure_fluid_space = FunctionSpace(name="pressure_fluid", field_names=["p_pos_"], dim=0, side="+")
velocity_solid_space = FunctionSpace(name="velocity_solid", field_names=["vs_neg_x", "vs_neg_y"], dim=1, side="-")
displacement_space = FunctionSpace(name="displacement", field_names=["d_neg_x", "d_neg_y"], dim=1, side="-")

du_f = VectorTrialFunction(space=velocity_fluid_space, dof_handler=dof_handler)
dp_f = TrialFunction(name="trial_pressure_fluid", field_name="p_pos_", dof_handler=dof_handler, side="+")
du_s = VectorTrialFunction(space=velocity_solid_space, dof_handler=dof_handler)
ddisp_s = VectorTrialFunction(space=displacement_space, dof_handler=dof_handler)
test_vel_f = VectorTestFunction(space=velocity_fluid_space, dof_handler=dof_handler)
test_q_f = TestFunction(name="test_pressure_fluid", field_name="p_pos_", dof_handler=dof_handler, side="+")
test_vel_s = VectorTestFunction(space=velocity_solid_space, dof_handler=dof_handler)
test_disp_s = VectorTestFunction(space=displacement_space, dof_handler=dof_handler)

uf_k = VectorFunction(name="u_f_k", field_names=["u_pos_x", "u_pos_y"], dof_handler=dof_handler, side="+")
pf_k = Function(name="p_f_k", field_name="p_pos_", dof_handler=dof_handler, side="+")
uf_n = VectorFunction(name="u_f_n", field_names=["u_pos_x", "u_pos_y"], dof_handler=dof_handler, side="+")
pf_n = Function(name="p_f_n", field_name="p_pos_", dof_handler=dof_handler, side="+")
us_k = VectorFunction(name="u_s_k", field_names=["vs_neg_x", "vs_neg_y"], dof_handler=dof_handler, side="-")
us_n = VectorFunction(name="u_s_n", field_names=["vs_neg_x", "vs_neg_y"], dof_handler=dof_handler, side="-")
disp_k = VectorFunction(name="disp_k", field_names=["d_neg_x", "d_neg_y"], dof_handler=dof_handler, side="-")
disp_n = VectorFunction(name="disp_n", field_names=["d_neg_x", "d_neg_y"], dof_handler=dof_handler, side="-")

for func in [uf_k, pf_k, uf_n, pf_n, us_k, us_n, disp_k, disp_n]:
    func.nodal_values.fill(0.0)

dof_handler.apply_bcs(bcs, uf_k, pf_k, us_k, disp_k)
dof_handler.apply_bcs(bcs, uf_n, pf_n, us_n, disp_n)

# -----------------------------------------------------------------------------
# Measures and stabilization weights
# -----------------------------------------------------------------------------
qvol = 6
dx_fluid = dx(
    defined_on=domains["fluid_interface"],
    level_set=ls_beam,
    metadata={"q": qvol, "side": "+"},
)
# Solid integrals are evaluated in the current (Eulerian) frame because the
# level set is updated with the present displacement.
dx_solid = dx(
    defined_on=domains["solid_interface"],
    level_set=ls_beam,
    metadata={"q": qvol, "side": "-"},
)
dΓ = dInterface(
    defined_on=domains["cut_domain"],
    level_set=ls_beam,
    metadata={"q": qvol + 2, "derivs": {(0, 0), (0, 1), (1, 0)}},
)
dG_fluid = dGhost(
    defined_on=domains["fluid_ghost"],
    level_set=ls_beam,
    metadata={"q": qvol, "derivs": {(0, 1), (1, 0)}},
)
dG_solid = dGhost(
    defined_on=domains["solid_ghost"],
    level_set=ls_beam,
    metadata={"q": qvol, "derivs": {(0, 1), (1, 0)}},
)

cell_h = CellDiameter()
beta_N = Constant(BETA_PENALTY * POLY_ORDER * (POLY_ORDER + 1))

theta_min = 1.0e-3
theta_pos_vals = np.clip(hansbo_cut_ratio(mesh, ls_beam, side="+"), theta_min, 1.0)
theta_neg_vals = np.clip(hansbo_cut_ratio(mesh, ls_beam, side="-"), theta_min, 1.0)
kappa_pos = Pos(ElementWiseConstant(theta_pos_vals))
kappa_neg = Neg(ElementWiseConstant(theta_neg_vals))

use_restricted_forms = True
if use_restricted_forms:
    du_f_R = restrict(du_f, domains["has_pos"])
    dp_f_R = restrict(dp_f, domains["has_pos"])
    test_vel_f_R = restrict(test_vel_f, domains["has_pos"])
    test_q_f_R = restrict(test_q_f, domains["has_pos"])
    uf_k_R = restrict(uf_k, domains["has_pos"])
    uf_n_R = restrict(uf_n, domains["has_pos"])
    pf_k_R = restrict(pf_k, domains["has_pos"])
    pf_n_R = restrict(pf_n, domains["has_pos"])

    du_s_R = restrict(du_s, domains["has_neg"])
    ddisp_s_R = restrict(ddisp_s, domains["has_neg"])
    test_vel_s_R = restrict(test_vel_s, domains["has_neg"])
    test_disp_s_R = restrict(test_disp_s, domains["has_neg"])
    us_k_R = restrict(us_k, domains["has_neg"])
    us_n_R = restrict(us_n, domains["has_neg"])
    disp_k_R = restrict(disp_k, domains["has_neg"])
    disp_n_R = restrict(disp_n, domains["has_neg"])
else:
    du_f_R = du_f
    dp_f_R = dp_f
    test_vel_f_R = test_vel_f
    test_q_f_R = test_q_f
    uf_k_R = uf_k
    uf_n_R = uf_n
    pf_k_R = pf_k
    pf_n_R = pf_n
    du_s_R = du_s
    ddisp_s_R = ddisp_s
    test_vel_s_R = test_vel_s
    test_disp_s_R = test_disp_s
    us_k_R = us_k
    us_n_R = us_n
    disp_k_R = disp_k
    disp_n_R = disp_n

I2 = Identity(2)
n = FacetNormal()

# -----------------------------------------------------------------------------
# Constitutive helpers
# -----------------------------------------------------------------------------
def epsilon_f(u):
    return 0.5 * (grad(u) + grad(u).T)


def epsilon_s_linear_L(disp, disp_k):
    return 0.5 * (grad(disp) + grad(disp).T + dot(grad(disp).T, grad(disp_k)) + dot(grad(disp_k).T, grad(disp)))


def epsilon_s_linear_R(disp_k):
    return 0.5 * (grad(disp_k) + grad(disp_k).T + dot(grad(disp_k).T, grad(disp_k)))


def sigma_s_linear_weak_linear_a(ddisp, disp_k, grad_v_test):
    eps = epsilon_s_linear_L(ddisp, disp_k)
    return 2.0 * Constant(MU_S) * inner(eps, grad_v_test) + Constant(LAMBDA_S) * trace(eps) * trace(grad_v_test)


def sigma_s_linear_weak_nonlinear_residual(disp_k, grad_v_test):
    eps = epsilon_s_linear_R(disp_k)
    return 2.0 * Constant(MU_S) * inner(eps, grad_v_test) + Constant(LAMBDA_S) * trace(eps) * trace(grad_v_test)


def traction_fluid(u_vec, p_scal):
    return 2.0 * Constant(MU_F) * dot(epsilon_f(u_vec), n) - p_scal * n


def F_of(d):
    return I2 + grad(d)


def C_of(F):
    return dot(F.T, F)


def E_of(F):
    return 0.5 * (C_of(F) - I2)


def S_stvk(E):
    return Constant(LAMBDA_S) * trace(E) * I2 + Constant(2.0 * MU_S) * E


def sigma_s_nonlinear(d):
    F = F_of(d)
    E = E_of(F)
    S = S_stvk(E)
    J = det(F)
    return (1.0 / J) * dot(dot(F, S), F.T)

# ----------------------- Diagnostic helpers -------------------------------
def _interface_length(mesh: Mesh) -> float:
    """Total length of edges tagged as interface."""
    length = 0.0
    for gid in mesh.edge_bitset("interface").to_indices():
        e = mesh.edge(int(gid))
        p0, p1 = mesh.nodes_x_y_pos[list(e.nodes)]
        length += float(np.linalg.norm(p1 - p0))
    return length

def _eval_scalar_at_point(dh: DofHandler, mesh: Mesh, f_scalar: Function, point: tuple[float, float]) -> float:
    """
    Evaluate a scalar Function at a physical point using element search
    and basis evaluation (robust to mixed-order layouts).
    """
    xy = np.asarray(point, float)
    for e in mesh.elements_list:
        verts = mesh.nodes_x_y_pos[list(e.nodes)]
        if not (verts[:, 0].min() - 1e-12 <= xy[0] <= verts[:, 0].max() + 1e-12 and
                verts[:, 1].min() - 1e-12 <= xy[1] <= verts[:, 1].max() + 1e-12):
            continue
        try:
            xi, eta = transform.inverse_mapping(mesh, e.id, xy)
        except Exception:
            continue
        if not (-1.0001 <= xi <= 1.0001 and -1.0001 <= eta <= 1.0001):
            continue
        me = dh.mixed_element
        fld = f_scalar.field_name
        phi = me.basis(fld, float(xi), float(eta))[me.slice(fld)]
        gdofs = dh.element_maps[fld][e.id]
        vals = f_scalar.get_nodal_values(gdofs)
        return float(phi @ vals)
    return float("nan")

def _tip_position(dh: DofHandler, mesh: Mesh, disp: VectorFunction, ref_tip: np.ndarray) -> np.ndarray:
    """Current position of the beam tip: X_ref + u(X_ref)."""
    dx = _eval_scalar_at_point(dh, mesh, disp.components[0], tuple(ref_tip))
    dy = _eval_scalar_at_point(dh, mesh, disp.components[1], tuple(ref_tip))
    return np.asarray([ref_tip[0] + dx, ref_tip[1] + dy], float)


def dsigma_s(d_ref, delta_d):
    Fk = F_of(d_ref)
    Ek = E_of(Fk)
    Sk = S_stvk(Ek)
    dF = grad(delta_d)
    dE = 0.5 * (dot(dF.T, Fk) + dot(Fk.T, dF))
    dS = Constant(LAMBDA_S) * trace(dE) * I2 + Constant(2.0 * MU_S) * dE
    Jk = det(Fk)
    Finv = inv(Fk)
    dJ = Jk * trace(dot(Finv, dF))
    term = dot(dF, dot(Sk, Fk.T)) + dot(Fk, dot(dS, Fk.T)) + dot(Fk, dot(Sk, dF.T))
    return (1.0 / Jk) * term - (dJ / Jk) * sigma_s_nonlinear(d_ref)


def traction_solid_R(d):
    return dot(sigma_s_nonlinear(d), n)


def traction_solid_L(delta_d, d_ref):
    return dot(dsigma_s(d_ref, delta_d), n)


def delta_E_GreenLagrange(w, u_ref):
    F_ref = F_of(u_ref)
    grad_w = grad(w)
    return Constant(0.5) * (dot(grad_w.T, F_ref) + dot(F_ref.T, grad_w))


def grad_inner_jump(u, v):
    a = dot(jump(grad(u)), n)
    b = dot(jump(grad(v)), n)
    return inner(a, b)


# -----------------------------------------------------------------------------
# Weak forms
# -----------------------------------------------------------------------------
dt = Constant(DT)
theta = Constant(1.0)
rho_f_const = Constant(RHO_F)
rho_s_const = Constant(RHO_S)
mu_f_const = Constant(MU_F)
mu_s_const = Constant(MU_S)
lambda_s_const = Constant(LAMBDA_S)

jump_vel_trial = Jump(du_f, du_s)
jump_vel_test = Jump(test_vel_f, test_vel_s)
jump_vel_res = Jump(uf_k, us_k)

avg_flux_trial = kappa_pos * traction_fluid(Pos(du_f), Pos(dp_f)) - kappa_neg * traction_solid_L(Neg(ddisp_s), Neg(disp_k))
avg_flux_test = kappa_pos * traction_fluid(Pos(test_vel_f), -Pos(test_q_f)) - kappa_neg * traction_solid_L(Neg(test_disp_s), Neg(disp_k))
avg_flux_res = kappa_pos * traction_fluid(Pos(uf_k), Pos(pf_k)) - kappa_neg * traction_solid_R(Neg(disp_k))

J_int = (-dot(avg_flux_trial, jump_vel_test) - dot(avg_flux_test, jump_vel_trial) + (beta_N * mu_f_const / cell_h) * dot(jump_vel_trial, jump_vel_test)) * dΓ
R_int = (-dot(avg_flux_res, jump_vel_test) - dot(avg_flux_test, jump_vel_res) + (beta_N * mu_f_const / cell_h) * dot(jump_vel_res, jump_vel_test)) * dΓ

a_vol_f = (
    rho_f_const / dt * dot(du_f_R, test_vel_f_R)
    + theta * rho_f_const * dot(dot(grad(uf_k_R), du_f_R), test_vel_f_R)
    + theta * rho_f_const * dot(dot(grad(du_f_R), uf_k_R), test_vel_f_R)
    + theta * mu_f_const * inner(grad(du_f_R), grad(test_vel_f_R))
    - dp_f_R * div(test_vel_f_R)
    + test_q_f_R * div(du_f_R)
) * dx_fluid

r_vol_f = (
    rho_f_const * dot(uf_k_R - uf_n_R, test_vel_f_R) / dt
    + theta * rho_f_const * dot(dot(grad(uf_k_R), uf_k_R), test_vel_f_R)
    + (1 - theta) * rho_f_const * dot(dot(grad(uf_n_R), uf_n_R), test_vel_f_R)
    + theta * mu_f_const * inner(grad(uf_k_R), grad(test_vel_f_R))
    + (1 - theta) * mu_f_const * inner(grad(uf_n_R), grad(test_vel_f_R))
    - pf_k_R * div(test_vel_f_R)
    + test_q_f_R * div(uf_k_R)
) * dx_fluid

sigma_s_k = sigma_s_nonlinear(disp_k_R)  # Cauchy stress in current frame
sigma_s_n = sigma_s_nonlinear(disp_n_R)
dsigma_s_k = dsigma_s(disp_k_R, ddisp_s_R)

a_vol_s = (
    rho_s_const * dot(du_s_R, test_vel_s_R) / dt
    + theta * inner(dsigma_s_k, grad(test_vel_s_R))
    + rho_s_const * theta
      * (dot(dot(grad(us_k_R), du_s_R), test_vel_s_R) + 
         dot(dot(grad(du_s_R), us_k_R), test_vel_s_R))
) * dx_solid
r_vol_s = (
    rho_s_const * dot(us_k_R - us_n_R, test_vel_s_R) / dt
    + theta * inner(sigma_s_k, grad(test_vel_s_R))
    + (1 - theta) * inner(sigma_s_n, grad(test_vel_s_R))
    + rho_s_const
        * (
        theta * dot(dot(grad(us_k_R), us_k_R), test_vel_s_R)
        + (1 - theta) * dot(dot(grad(us_n_R), us_n_R), test_vel_s_R)
    )
) * dx_solid



a_svc = (dot(ddisp_s_R, test_disp_s_R) / dt - theta * dot(du_s_R, test_disp_s_R)
         + theta * (dot(dot(grad(ddisp_s_R), us_k_R), test_disp_s_R) 
                    + dot(dot(grad(disp_k_R), du_s_R), test_disp_s_R))
         ) * dx_solid
# Kinematic constraint with advected displacement in Eulerian frame
r_svc = (
    dot(disp_k_R - disp_n_R, test_disp_s_R) / dt
    - theta * dot(us_k_R, test_disp_s_R)
    - (1 - theta) * dot(us_n_R, test_disp_s_R)
    + theta * dot(dot(grad(disp_k_R), us_k_R), test_disp_s_R)
    + (1 - theta) * dot(dot(grad(disp_n_R), us_n_R), test_disp_s_R)
) * dx_solid

penalty_val = 1e-1
penalty_grad = 1e-1
gamma_v = Constant(penalty_val * POLY_ORDER**2)
gamma_v_grad = Constant(penalty_grad * POLY_ORDER**2)
gamma_p = Constant(penalty_val * POLY_ORDER)


def g_v_f(gamma, phi_1, phi_2):
    return gamma * (cell_h * grad_inner_jump(phi_1, phi_2))


def g_p(gamma, phi_1, phi_2):
    return gamma * (cell_h**3.0 * grad_inner_jump(phi_1, phi_2))


def g_v_s(gamma, phi_1, phi_2):
    return gamma * (cell_h**3.0 * grad_inner_jump(phi_1, phi_2))


def g_disp_s(gamma, phi_1, phi_2):
    return gamma * (cell_h * grad_inner_jump(phi_1, phi_2))


a_stab = (
    (Constant(2.0) * mu_f_const * g_v_f(gamma_v, du_f_R, test_vel_f_R) + g_p(gamma_p, dp_f_R, test_q_f_R)) * dG_fluid
    + (rho_s_const * g_v_s(gamma_v, du_s_R, test_vel_s_R) + Constant(2.0) * mu_s_const * g_disp_s(gamma_v_grad, ddisp_s_R, test_disp_s_R))
    * dG_solid
)
r_stab = (
    (Constant(2.0) * mu_f_const * g_v_f(gamma_v, uf_k_R, test_vel_f_R) + g_p(gamma_p, pf_k_R, test_q_f_R)) * dG_fluid
    + (rho_s_const * g_v_s(gamma_v, us_k_R, test_vel_s_R) + Constant(2.0) * mu_s_const * g_disp_s(gamma_v_grad, disp_k_R, test_disp_s_R))
    * dG_solid
)

jacobian_form = a_vol_f + J_int + a_vol_s + a_stab
residual_form = r_vol_f + R_int + r_vol_s + r_stab

# ----------------------------------------------------------------------------- 
# Diagnostics: tip displacement, drag/lift (avg traction), pressure drop, VTK
# -----------------------------------------------------------------------------
REF_TIP = np.array([CENTER[0] + RADIUS + BEAM_LENGTH, CENTER[1]], dtype=float)
PROBE_B = np.array([CENTER[0] - 0.05, CENTER[1]], dtype=float)
obs_history = {"time": [], "tip": [], "drag": [], "lift": [], "dp": []}
output_dir = ARGS.output_dir
os.makedirs(output_dir, exist_ok=True)
SAVE_VTK = bool(ARGS.save_vtk)
PLOT_MESH_EVERY = max(1, int(ARGS.plot_mesh_every))

def _save_vtk(step_idx: int) -> None:
    if not SAVE_VTK:
        return
    fname = os.path.join(output_dir, f"solution_{step_idx:04d}.vtu")
    export_vtk(
        filename=fname,
        mesh=mesh,
        dof_handler=dof_handler,
        functions={
            "uf": uf_k,
            "pf": pf_k,
            "us": us_k,
            "disp": disp_k,
        },
    )


def _plot_mesh(step_idx: int, title: str = "Mesh / Ghost / Level-set") -> None:
    if not ARGS.plot_mesh:
        return
    if step_idx % PLOT_MESH_EVERY != 0:
        return
    import matplotlib.pyplot as plt
    from matplotlib.widgets import CheckButtons
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot with the analytic reference LS to avoid FE warping in the visualization.
    level_set_for_plot = beam_ref_ls if ARGS.plot_levelset else None
    plot_mesh_2(
        mesh,
        level_set=level_set_for_plot,
        plot_nodes=True,
        plot_edges=True,
        elem_tags=True,
        edge_colors=True,
        show=False,
        ax=ax,
        resolution=max(20, int(ARGS.plot_resolution)),
    )
    interface_artist = None
    if ARGS.plot_interface_points:
        segments = []
        for elem in mesh.elements_list:
            pts = getattr(elem, "interface_pts", [])
            if len(pts) == 2:
                segments.append(np.asarray(pts, float))
        if segments:
            interface_artist = ax.add_collection(LineCollection(segments, colors="magenta", linewidths=1.5, zorder=6, label="Interface pts"))

    levelset_artists = []
    if level_set_for_plot is not None:
        xmin, ymin = mesh.nodes_x_y_pos.min(axis=0)
        xmax, ymax = mesh.nodes_x_y_pos.max(axis=0)
        padding = (xmax - xmin) * 0.05
        res = max(20, int(ARGS.plot_resolution))
        gx, gy = np.meshgrid(
            np.linspace(xmin - padding, xmax + padding, res),
            np.linspace(ymin - padding, ymax + padding, res),
        )
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        vals = np.apply_along_axis(level_set_for_plot, 1, pts).reshape(gx.shape)
        cs = ax.contour(gx, gy, vals, levels=[0.0], colors="green", linewidths=1.5, zorder=5)
        levelset_artists = cs.collections

    toggles = []
    labels = []
    artists = []
    if levelset_artists:
        labels.append("Level set")
        artists.append(levelset_artists)
        toggles.append(True)
    if interface_artist is not None:
        labels.append("Interface pts")
        artists.append([interface_artist])
        toggles.append(True)

    if ARGS.interactive_plot and labels:
        rax = fig.add_axes([0.82, 0.4, 0.15, 0.15])
        check = CheckButtons(rax, labels, toggles)

        def func(label):
            for lab, arts in zip(labels, artists):
                if lab == label:
                    new_vis = not arts[0].get_visible()
                    for art in arts:
                        art.set_visible(new_vis)
            fig.canvas.draw_idle()

        check.on_clicked(func)

    ax.set_title(f"{title} (step {step_idx})")
    fname = os.path.join(output_dir, f"mesh_{step_idx:04d}.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    if ARGS.plot_show or ARGS.interactive_plot:
        import matplotlib
        plt.show()
        # block = bool(ARGS.interactive_plot)
        # backend = matplotlib.get_backend().lower()
        # if "agg" in backend and block:
        #     print("Non-interactive backend detected; skipping blocking plt.show(). Image saved instead.")
        #     block = False
        # plt.show(block=block)
        # if not block:
        #     plt.pause(0.1)
        #     plt.close(fig)
    else:
        plt.close(fig)

# Plot initial mesh/level set before any expensive steps (FD checks/time stepping)
_plot_mesh(step_idx=0, title="Initial mesh")

def _compute_observables(step_idx: int, t_curr: float) -> None:
    dGamma_obs = dInterface(
        defined_on=domains["cut_domain"],
        level_set=ls_beam,
        metadata={"q": qvol + 2, "derivs": {(0, 0), (1, 0), (0, 1)}},
    )
    ex = Constant(np.array([1.0, 0.0]), dim=1)
    ey = Constant(np.array([0.0, 1.0]), dim=1)
    t_fluid = traction_fluid(Pos(uf_k), Pos(pf_k))
    t_solid = traction_solid_R(Neg(disp_k))
    t_avg = Constant(0.5) * (t_fluid + t_solid)
    drag_int = dot(t_avg, ex) * dGamma_obs
    lift_int = dot(t_avg, ey) * dGamma_obs
    hooks = {drag_int.integrand: {"name": "FD"}, lift_int.integrand: {"name": "FL"}}
    res = assemble_form(
        Equation(None, drag_int + lift_int),
        dof_handler=dof_handler,
        bcs=[],
        assembler_hooks=hooks,
        backend="jit",
    )
    F_D = float(res.get("FD", 0.0))
    F_L = float(res.get("FL", 0.0))
    tip_pos = _tip_position(dof_handler, mesh, disp_k, REF_TIP)
    pA = _eval_scalar_at_point(dof_handler, mesh, pf_k, tuple(tip_pos))
    pB = _eval_scalar_at_point(dof_handler, mesh, pf_k, tuple(PROBE_B))
    dp = pB - pA
    obs_history["time"].append(t_curr)
    obs_history["drag"].append(F_D)
    obs_history["lift"].append(F_L)
    obs_history["dp"].append(dp)
    obs_history["tip"].append(tip_pos.tolist())
    print(
        f"[obs {step_idx:04d}] t={t_curr:.3f}  FD={F_D:.4e}  FL={F_L:.4e}  Δp={dp:.4e}  "
        f"tip=({tip_pos[0]:.5f},{tip_pos[1]:.5f})  |Γ|≈{_interface_length(mesh):.5f}"
    )
    _save_vtk(step_idx)

# -----------------------------------------------------------------------------
# Finite-difference alignment check
# -----------------------------------------------------------------------------
if ARGS.run_fd_check:
    fd_fields = {
        "u_pos_x": uf_k,
        "u_pos_y": uf_k,
        "p_pos_": pf_k,
        "vs_neg_x": us_k,
        "vs_neg_y": us_k,
        "d_neg_x": disp_k,
        "d_neg_y": disp_k,
    }
    probe = select_fd_dofs(dof_handler, {"u_pos_x": 2, "u_pos_y": 2, "vs_neg_x": 2, "d_neg_x": 2}, elem_tag="cut")
    finite_difference_check(jacobian_form, residual_form, dof_handler, bcs_homog, fd_fields, probe, eps=1.0e-7)
    if ARGS.run_fd_terms:
        term_blocks = {
            "fluid_vol": (a_vol_f, r_vol_f),
            "solid_vol": (a_vol_s, r_vol_s),
            "solid_vel_constraint": (a_svc, r_svc),
            "interface": (J_int, R_int),
            "stab_fluid_vel": (
                (Constant(2.0) * mu_f_const * g_v_f(gamma_v, du_f_R, test_vel_f_R)) * dG_fluid,
                (Constant(2.0) * mu_f_const * g_v_f(gamma_v, uf_k_R, test_vel_f_R)) * dG_fluid,
            ),
            "stab_fluid_p": (
                g_p(gamma_p, dp_f_R, test_q_f_R) * dG_fluid,
                g_p(gamma_p, pf_k_R, test_q_f_R) * dG_fluid,
            ),
            "stab_solid_vel": (
                (rho_s_const * g_v_s(gamma_v, du_s_R, test_vel_s_R)) * dG_solid,
                (rho_s_const * g_v_s(gamma_v, us_k_R, test_vel_s_R)) * dG_solid,
            ),
            "stab_solid_disp": (
                (Constant(2.0) * mu_s_const * g_disp_s(gamma_v_grad, ddisp_s_R, test_disp_s_R)) * dG_solid,
                (Constant(2.0) * mu_s_const * g_disp_s(gamma_v_grad, disp_k_R, test_disp_s_R)) * dG_solid,
            ),
        }
        for name, (jf, rf) in term_blocks.items():
            print(f"\n[FD term] {name}")
            finite_difference_check(jf, rf, dof_handler, bcs_homog, fd_fields, probe, eps=1.0e-7)

# -----------------------------------------------------------------------------
# Time stepping
# -----------------------------------------------------------------------------
if ARGS.run_time_stepping:
    time_params = TimeStepperParameters(dt=DT, max_steps=50, stop_on_steady=True, steady_tol=1e-6, theta=theta.value)
    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dof_handler,
        mixed_element=mixed_element,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=1e-6, line_search=True),
    )

    step_idx = [0]  # mutable so the closure can update
    dt_val = DT.value if hasattr(DT, "value") else float(DT)

    def post_step_cb(funcs):
        update_beam_levelset_from_displacement(ls_beam, disp_k, beam_ref_ls)
        refresh_domains(mesh, domains)
        refresh_hansbo_kappa(mesh, ls_beam, theta_pos_vals, theta_neg_vals)
        retag_inactive(dof_handler)
        dof_handler.apply_bcs(bcs, uf_k, pf_k, us_k, disp_k)
        recompute_active_dofs(solver, solver.bcs_homog if solver.bcs_homog else solver.bcs)
        _compute_observables(step_idx[0], step_idx[0] * dt_val)
        _plot_mesh(step_idx[0], title="Mesh / level-set")
        step_idx[0] += 1

    solver.post_timeloop_cb = post_step_cb

    solver.solve_time_interval(
        functions=[uf_k, pf_k, us_k, disp_k],
        prev_functions=[uf_n, pf_n, us_n, disp_n],
        time_params=time_params,
    )
