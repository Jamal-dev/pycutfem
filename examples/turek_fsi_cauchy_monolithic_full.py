#!/usr/bin/env python
# coding: utf-8
"""
Monolithic CutFEM setup for the Turek–Hron FSI-2 benchmark.

- Geometry: channel with a *rigid* circular hole; the elastic beam is described
  by a level-set that is advected with the solid displacement.
- Mechanics: nonlinear St. Venant–Kirchhoff solid, incompressible Navier–Stokes
  for the fluid.
- The beam level set is updated every time step and the mesh is reclassified so
  curved deformations of the beam are captured.
- A small finite-difference Jacobian check is run to validate the assembled
  Jacobian against the residual.
"""
from __future__ import annotations

import math
import os
import time
import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, Iterable, Sequence

try:
    import gmsh  # type: ignore
except Exception:
    gmsh = None

import numba
import numpy as np

from pycutfem.core.geometry import hansbo_cut_ratio
from pycutfem.core.levelset import BeamLevelSet, LevelSetGridFunction
from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node
from pycutfem.fem import transform
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.bitset import BitSet
from pycutfem.utils.gmsh_loader import mesh_from_gmsh
from pycutfem.utils.ogrid_meshgen import circular_hole_ogrid

from pycutfem.ufl.spaces import FunctionSpace
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

# -----------------------------------------------------------------------------
# CLI / environment options
# -----------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(description="Monolithic CutFEM Turek–Hron FSI-2 (Cauchy solid).")
    parser.add_argument("--dt", type=float, default=float(os.getenv("DT", "0.005")), help="Time step size.")
    parser.add_argument("--poly-order", type=int, default=int(os.getenv("POLY_ORDER", "2")), help="Polynomial order for primary fields.")
    parser.add_argument("--mesh-size", type=float, default=float(os.getenv("MESH_SIZE", "0.025")), help="Target mesh size for structured O-grid.")
    parser.add_argument(
        "--mesh-backend",
        choices=("gmsh", "structured"),
        default=os.getenv("MESH_BACKEND", "gmsh"),
        help="Mesh generator: 'gmsh' (beam-aware blocked grid) or the legacy structured O-grid.",
    )
    parser.add_argument("--mesh-file", type=Path, default=None, help="Optional path to reuse/store the gmsh .msh file.")
    parser.add_argument("--rebuild-mesh", action="store_true", help="Force rebuilding the gmsh mesh instead of reusing an existing file.")
    parser.add_argument("--view-gmsh", action="store_true", help="Preview the gmsh model before meshing.")
    parser.add_argument(
        "--refine-initial",
        dest="refine_initial",
        action="store_true",
        help="Apply asymmetric refinement around the beam in the initial mesh.",
    )
    parser.add_argument(
        "--no-refine-initial",
        dest="refine_initial",
        action="store_false",
        help="Skip initial refinement around the beam.",
    )
    parser.add_argument("--refine-levels", type=int, default=int(os.getenv("BEAM_REFINE_LEVELS", "2")), help="Number of refinement expansion layers.")
    parser.add_argument("--refine-band", type=float, default=None, help="Optional refinement band half-width around the beam (override default heuristic).")
    parser.add_argument("--plot-mesh", dest="plot_mesh", action="store_true", help="Plot the initial mesh/level-set using plot_mesh_2.")
    parser.add_argument("--no-plot-mesh", dest="plot_mesh", action="store_false", help="Disable initial mesh plotting.")
    parser.add_argument("--plot-mesh-file", type=Path, default=None, help="If set, save the mesh plot to this path.")
    parser.add_argument("--plot-only", dest="plot_only", action="store_true", help="Plot the initial mesh and exit.")
    parser.set_defaults(
        refine_initial=os.getenv("REFINE_INITIAL", "1") != "0",
        plot_mesh=os.getenv("PLOT_MESH", "0") != "0",
        plot_only=os.getenv("PLOT_ONLY", "0") != "0",
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
BEAM_SHIFT_X = float(os.getenv("BEAM_SHIFT_X", "0.0"))
BEAM_REF_CENTER = (BEAM_CENTER[0] - BEAM_SHIFT_X, BEAM_CENTER[1])
BEAM_REF_LENGTH = BEAM_LENGTH + 2.0 * BEAM_SHIFT_X
BEAM_REF_HEIGHT = BEAM_HEIGHT
POINT_B = (0.15, 0.2)
POINT_A_INITIAL = (0.6, 0.2) # Point A will change while Point B is fixed

BETA_PENALTY = 90.0 * MU_F
DT = float(ARGS.dt)
POLY_ORDER = int(ARGS.poly_order)
MESH_SIZE = float(ARGS.mesh_size)
BEAM_ROOT_TOL = float(max(1.0e-6, 1.0e-3 * MESH_SIZE))
BEAM_ROOT_BIAS = float(max(1.0e-8, 1.0e-4 * MESH_SIZE))
BEAM_ROOT_INSET = float(os.getenv("BEAM_ROOT_INSET", str(max(5.0e-4, 0.04 * MESH_SIZE))))
BEAM_ROOT_DOF_TOL = float(os.getenv("BEAM_ROOT_DOF_TOL", str(max(0.2 * MESH_SIZE, 1.0e-3))))
PIN_PRESSURE = os.getenv("PIN_PRESSURE", "1") not in ("0", "false", "False")
SOLID_CUT_DROP = float(os.getenv("SOLID_CUT_DROP", "0.0"))

# -----------------------------------------------------------------------------
# Mesh and boundary helpers
# -----------------------------------------------------------------------------


class BeamArcRootLevelSet:
    """
    Beam level set with a curved root that follows the cylinder arc, so the
    beam attaches without leaving a vertical gap at x=beam_x0.
    """

    def __init__(
        self,
        *,
        beam_center: tuple[float, float],
        beam_length: float,
        beam_height: float,
        cyl_center: tuple[float, float],
        cyl_radius: float,
        root_inset: float,
        root_bias: float,
        root_tol: float,
    ):
        self.cx = float(beam_center[0])
        self.cy = float(beam_center[1])
        self.hx = 0.5 * float(beam_length)
        self.hy = 0.5 * float(beam_height)
        self._beam_x1 = self.cx + self.hx
        self._beam_y0 = self.cy - self.hy
        self._beam_y1 = self.cy + self.hy
        self._cyl_center = np.asarray(cyl_center, dtype=float)
        self._cyl_radius = float(cyl_radius)
        self._root_inset = float(root_inset)
        self._root_bias = float(root_bias)
        self._root_tol = float(root_tol)
        self.cache_token = (
            "beam_arc_root",
            float(beam_center[0]),
            float(beam_center[1]),
            float(beam_length),
            float(beam_height),
            float(cyl_radius),
            float(root_inset),
            float(root_bias),
            float(root_tol),
        )

    def _x_arc(self, y: np.ndarray) -> np.ndarray:
        dy = y - self._cyl_center[1]
        rad2 = self._cyl_radius * self._cyl_radius
        inside = np.maximum(rad2 - dy * dy, 0.0)
        return self._cyl_center[0] + np.sqrt(inside) - self._root_inset

    def __call__(self, x):
        x = np.asarray(x, float)
        x_coord = x[..., 0]
        y_coord = x[..., 1]

        x_arc = self._x_arc(y_coord)
        phi_left = x_arc - x_coord
        phi_right = x_coord - self._beam_x1
        phi_top = y_coord - self._beam_y1
        phi_bottom = self._beam_y0 - y_coord

        phi = np.max(np.stack((phi_left, phi_right, phi_top, phi_bottom), axis=-1), axis=-1)
        if self._root_bias > 0.0:
            if x.ndim == 1:
                on_root = (
                    (np.abs(phi_left) <= self._root_tol)
                    and (self._beam_y0 - self._root_tol <= y_coord <= self._beam_y1 + self._root_tol)
                    and (x_coord >= self._cyl_center[0])
                )
                if on_root:
                    phi = min(float(phi), -self._root_bias)
            else:
                on_root = (
                    (np.abs(phi_left) <= self._root_tol)
                    & (y_coord >= self._beam_y0 - self._root_tol)
                    & (y_coord <= self._beam_y1 + self._root_tol)
                    & (x_coord >= self._cyl_center[0])
                )
                phi = np.where(on_root, np.minimum(phi, -self._root_bias), phi)
        return phi

    def gradient(self, x):
        x = np.asarray(x, float)
        x_coord = x[..., 0]
        y_coord = x[..., 1]

        x_arc = self._x_arc(y_coord)
        phi_left = x_arc - x_coord
        phi_right = x_coord - self._beam_x1
        phi_top = y_coord - self._beam_y1
        phi_bottom = self._beam_y0 - y_coord
        phis = np.stack((phi_left, phi_right, phi_top, phi_bottom), axis=-1)
        idx = np.argmax(phis, axis=-1)

        dy = y_coord - self._cyl_center[1]
        rad2 = self._cyl_radius * self._cyl_radius
        denom = np.sqrt(np.maximum(rad2 - dy * dy, 1.0e-18))
        dx_arc_dy = -dy / denom

        grad_left = np.stack((-np.ones_like(x_coord), dx_arc_dy), axis=-1)
        grad_right = np.stack((np.ones_like(x_coord), np.zeros_like(x_coord)), axis=-1)
        grad_top = np.stack((np.zeros_like(x_coord), np.ones_like(x_coord)), axis=-1)
        grad_bottom = np.stack((np.zeros_like(x_coord), -np.ones_like(x_coord)), axis=-1)
        grads = np.stack((grad_left, grad_right, grad_top, grad_bottom), axis=-2)
        if x.ndim == 1:
            return grads[int(idx)]
        grad = np.take_along_axis(grads, idx[..., None, None], axis=-2).squeeze(-2)
        return grad


def _count_segments(width: float, mesh_size: float, *, min_cells: int = 1) -> int:
    if width <= 1.0e-12:
        return 0
    return max(min_cells, int(math.ceil(width / mesh_size)))


def _nodes_for_length(length: float, mesh_size: float, *, min_nodes: int = 3) -> int:
    segments = max(1, int(round(length / mesh_size)))
    return max(min_nodes, segments + 1)


def build_blocked_gmsh_mesh(
    path: Path,
    mesh_size: float,
    poly_order: int,
    *,
    view: bool = False,
    beam_center: tuple[float, float] | None = None,
    beam_length: float | None = None,
    beam_height: float | None = None,
) -> None:
    """
    Build a blocked, beam-aligned O-grid mesh with gmsh.
    """
    if gmsh is None:
        raise RuntimeError("Gmsh backend requested but the gmsh Python module is not available.")

    if beam_center is None:
        beam_center = BEAM_CENTER
    if beam_length is None:
        beam_length = BEAM_LENGTH
    if beam_height is None:
        beam_height = BEAM_HEIGHT

    gmsh.initialize()
    try:
        gmsh.model.add("turek_fsi_blocked")
        occ = gmsh.model.occ

        # Helper registries ---------------------------------------------------
        def _point_key(x: float, y: float) -> tuple[float, float]:
            return (round(float(x), 12), round(float(y), 12))

        point_lookup: dict[tuple[float, float], int] = {}
        point_coords: dict[int, tuple[float, float]] = {}

        def add_point(x: float, y: float) -> int:
            key = _point_key(x, y)
            if key in point_lookup:
                return point_lookup[key]
            tag = occ.addPoint(x, y, 0.0)
            point_lookup[key] = tag
            point_coords[tag] = (float(x), float(y))
            return tag

        line_lookup: dict[tuple[int, int], int] = {}
        line_lengths: dict[int, float] = {}
        line_target_nodes: dict[int, int] = {}
        line_meta: dict[int, tuple[int, int]] = {}

        def register_line(tag: int, start: int, end: int, length: float) -> None:
            line_lookup[(start, end)] = tag
            line_lengths[tag] = length

        def oriented_line(start: int, end: int) -> int:
            tag = line_lookup.get((start, end))
            if tag is not None:
                return tag
            tag = line_lookup.get((end, start))
            if tag is None:
                raise KeyError(f"No curve between points {start} and {end}.")
            return -tag

        boundary_edges: dict[str, list[int]] = {"inlet": [], "outlet": [], "walls": [], "cylinder": []}

        def add_line(start: int, end: int, boundary: str | None = None) -> int:
            tag = occ.addLine(start, end)
            x0, y0 = point_coords[start]
            x1, y1 = point_coords[end]
            length = math.hypot(x1 - x0, y1 - y0)
            register_line(tag, start, end, length)
            line_meta[tag] = (start, end)
            if boundary:
                boundary_edges[boundary].append(tag)
            line_target_nodes[tag] = line_target_nodes.get(tag, _nodes_for_length(length, mesh_size))
            return tag

        def _coord_index(seq: Sequence[float], value: float) -> int:
            for i, v in enumerate(seq):
                if abs(v - value) <= 1.0e-12:
                    return i
            raise ValueError(f"{value} not found in coordinate list.")

        def _arc_length(a_tag: int, b_tag: int) -> float:
            xa, ya = point_coords[a_tag]
            xb, yb = point_coords[b_tag]
            ang_a = math.atan2(ya - CENTER[1], xa - CENTER[0])
            ang_b = math.atan2(yb - CENTER[1], xb - CENTER[0])
            delta = ang_b - ang_a
            if delta <= 0.0:
                delta += 2.0 * math.pi
            return RADIUS * delta

        # Beam-aware square around the cylinder --------------------------------
        beam_x0 = beam_center[0] - 0.5 * beam_length
        beam_x1 = beam_center[0] + 0.5 * beam_length
        beam_y0 = beam_center[1] - 0.5 * beam_height
        beam_y1 = beam_center[1] + 0.5 * beam_height

        pad = max(0.6 * mesh_size, 0.008)
        hx = max(RADIUS + pad, min(CENTER[0] - pad, L - CENTER[0] - pad, 0.35))
        hy = max(RADIUS + pad, min(CENTER[1] - pad, H - CENTER[1] - pad, 0.35))
        if hx <= RADIUS or hy <= RADIUS:
            raise RuntimeError("Beam-aware O-grid collapsed; increase mesh size or adjust parameters.")

        square_left = CENTER[0] - hx
        square_right = CENTER[0] + hx
        square_bottom = CENTER[1] - hy
        square_top = CENTER[1] + hy

        beam_nodes = max(4, _nodes_for_length(beam_height, mesh_size, min_nodes=4))

        x_coords = sorted({0.0, square_left, square_right, beam_x0, beam_x1, L})
        y_coords = sorted({0.0, square_bottom, beam_y0, beam_center[1], beam_y1, square_top, H})

        x_interval_nodes = []
        for ix in range(len(x_coords) - 1):
            length = x_coords[ix + 1] - x_coords[ix]
            x_interval_nodes.append(_nodes_for_length(length, mesh_size))

        y_interval_nodes = []
        for iy in range(len(y_coords) - 1):
            length = y_coords[iy + 1] - y_coords[iy]
            count = _nodes_for_length(length, mesh_size)
            if y_coords[iy] >= square_bottom - 1.0e-12 and y_coords[iy + 1] <= square_top + 1.0e-12:
                count = max(count, beam_nodes)
            y_interval_nodes.append(count)

        grid_points: dict[tuple[int, int], int] = {}
        for ix, x in enumerate(x_coords):
            for iy, y in enumerate(y_coords):
                grid_points[(ix, iy)] = add_point(x, y)

        def _inside_inner(x_mid: float, y_mid: float) -> bool:
            return (square_left < x_mid < square_right) and (square_bottom < y_mid < square_top)

        horizontal_lines: dict[tuple[int, int], int] = {}
        for iy, y in enumerate(y_coords):
            for ix in range(len(x_coords) - 1):
                x0, x1 = x_coords[ix], x_coords[ix + 1]
                if _inside_inner(0.5 * (x0 + x1), y):
                    continue
                boundary = None
                if abs(y - 0.0) <= 1.0e-12 or abs(y - H) <= 1.0e-12:
                    boundary = "walls"
                tag = add_line(grid_points[(ix, iy)], grid_points[(ix + 1, iy)], boundary=boundary)
                horizontal_lines[(iy, ix)] = tag
                line_target_nodes[tag] = max(line_target_nodes.get(tag, 0), x_interval_nodes[ix])

        vertical_lines: dict[tuple[int, int], int] = {}
        for ix, x in enumerate(x_coords):
            for iy in range(len(y_coords) - 1):
                y0, y1 = y_coords[iy], y_coords[iy + 1]
                if _inside_inner(x, 0.5 * (y0 + y1)):
                    continue
                boundary = None
                if abs(x - 0.0) <= 1.0e-12:
                    boundary = "inlet"
                elif abs(x - L) <= 1.0e-12:
                    boundary = "outlet"
                tag = add_line(grid_points[(ix, iy)], grid_points[(ix, iy + 1)], boundary=boundary)
                vertical_lines[(ix, iy)] = tag
                line_target_nodes[tag] = max(line_target_nodes.get(tag, 0), y_interval_nodes[iy])

        fluid_surfaces: list[int] = []
        surface_loops: list[list[int]] = []
        for ix in range(len(x_coords) - 1):
            for iy in range(len(y_coords) - 1):
                xm = 0.5 * (x_coords[ix] + x_coords[ix + 1])
                ym = 0.5 * (y_coords[iy] + y_coords[iy + 1])
                if _inside_inner(xm, ym):
                    continue
                loop = [
                    horizontal_lines[(iy, ix)],
                    vertical_lines[(ix + 1, iy)],
                    -horizontal_lines[(iy + 1, ix)],
                    -vertical_lines[(ix, iy)],
                ]
                cloop = occ.addCurveLoop(loop)
                fluid_surfaces.append(occ.addPlaneSurface([cloop]))
                surface_loops.append(loop)

        # Beam-aware O-grid ring around the circle -----------------------------
        x_right_idx = _coord_index(x_coords, square_right)
        x_left_idx = _coord_index(x_coords, square_left)
        y_bot_idx = _coord_index(y_coords, square_bottom)
        y_top_idx = _coord_index(y_coords, square_top)
        y_beam_bot_idx = _coord_index(y_coords, beam_y0)
        y_beam_top_idx = _coord_index(y_coords, beam_y1)
        y_mid_idx = _coord_index(y_coords, CENTER[1])

        # Build square boundary points aligned with grid intersections to avoid T-junctions.
        y_seq = [y_bot_idx, y_beam_bot_idx, y_mid_idx, y_beam_top_idx, y_top_idx]
        top_x_indices = sorted(
            {idx for idx, xv in enumerate(x_coords) if square_left - 1e-12 <= xv <= square_right + 1e-12},
            reverse=True,
        )

        def _append_unique(coords_list: list[tuple[int, int]], coord: tuple[int, int]) -> None:
            if not coords_list or coords_list[-1] != coord:
                coords_list.append(coord)

        square_point_coords: list[tuple[int, int]] = []
        # right edge bottom→top
        for yi in y_seq:
            _append_unique(square_point_coords, (x_right_idx, yi))
        # top edge right→left
        for idx in top_x_indices[1:]:
            _append_unique(square_point_coords, (idx, y_top_idx))
        # left edge top→bottom (skip repeating top)
        for yi in reversed(y_seq[:-1]):
            _append_unique(square_point_coords, (x_left_idx, yi))
        # bottom edge left→right (skip left corner)
        bot_x_indices = sorted({idx for idx in top_x_indices}, reverse=False)
        for idx in bot_x_indices[1:]:
            _append_unique(square_point_coords, (idx, y_bot_idx))
        # close by omitting duplicate of starting point
        if square_point_coords and square_point_coords[-1] == square_point_coords[0]:
            square_point_coords.pop()

        square_points = [grid_points[(ix, iy)] for ix, iy in square_point_coords]

        center_pt = add_point(CENTER[0], CENTER[1])

        def _circle_point_through_square(nid_square: int) -> int:
            xs, ys = point_coords[nid_square]
            vec = np.array([xs - CENTER[0], ys - CENTER[1]], float)
            rlen = float(np.hypot(vec[0], vec[1]))
            if rlen <= 1.0e-14:
                raise RuntimeError("Square point coincides with circle center.")
            scale = RADIUS / rlen
            xc = CENTER[0] + scale * vec[0]
            yc = CENTER[1] + scale * vec[1]
            return add_point(xc, yc)

        circle_points = [_circle_point_through_square(pid) for pid in square_points]

        arc_lines: list[int] = []
        for i in range(len(circle_points)):
            start = circle_points[i]
            end = circle_points[(i + 1) % len(circle_points)]
            tag = occ.addCircleArc(start, center_pt, end)
            length = _arc_length(start, end)
            register_line(tag, start, end, length)
            line_target_nodes[tag] = max(line_target_nodes.get(tag, 0), _nodes_for_length(length, mesh_size))
            boundary_edges["cylinder"].append(tag)
            arc_lines.append(tag)

        square_segments: list[int] = []
        def _segment_target_nodes(a_tag: int, b_tag: int) -> int:
            xa, ya = point_coords[a_tag]
            xb, yb = point_coords[b_tag]
            if abs(ya - yb) <= 1.0e-12:
                ix0 = _coord_index(x_coords, min(xa, xb))
                return x_interval_nodes[ix0]
            if abs(xa - xb) <= 1.0e-12:
                iy0 = _coord_index(y_coords, min(ya, yb))
                return y_interval_nodes[iy0]
            return _nodes_for_length(math.hypot(xb - xa, yb - ya), mesh_size)

        for i in range(len(square_points)):
            a = square_points[i]
            b = square_points[(i + 1) % len(square_points)]
            try:
                seg = oriented_line(a, b)
            except KeyError:
                seg = add_line(a, b)
            square_segments.append(seg)
            seg_tag = abs(seg)
            line_target_nodes[seg_tag] = max(line_target_nodes.get(seg_tag, 0), _segment_target_nodes(a, b))

        radial_nodes = max(beam_nodes, _nodes_for_length(max(hx, hy) - RADIUS, mesh_size, min_nodes=4))
        radial_lines: list[int] = []
        for i, (c_pt, s_pt) in enumerate(zip(circle_points, square_points)):
            tag = add_line(c_pt, s_pt)
            default_nodes = max(radial_nodes, beam_nodes)
            line_target_nodes[tag] = max(default_nodes, line_target_nodes.get(tag, default_nodes))
            radial_lines.append(tag)

        for i, arc in enumerate(arc_lines):
            seg_tag = abs(square_segments[i])
            a = square_points[i]
            b = square_points[(i + 1) % len(square_points)]
            seg_nodes = _segment_target_nodes(a, b)
            arc_nodes = _nodes_for_length(line_lengths[arc], mesh_size)
            pair_count = max(seg_nodes, arc_nodes)
            xa, ya = point_coords[a]
            xb, yb = point_coords[b]
            if abs(ya - yb) <= 1.0e-12:
                ix0 = _coord_index(x_coords, min(xa, xb))
                x_interval_nodes[ix0] = max(x_interval_nodes[ix0], pair_count)
                pair_count = max(pair_count, x_interval_nodes[ix0])
            elif abs(xa - xb) <= 1.0e-12:
                iy0 = _coord_index(y_coords, min(ya, yb))
                y_interval_nodes[iy0] = max(y_interval_nodes[iy0], pair_count)
                pair_count = max(pair_count, y_interval_nodes[iy0])
            line_target_nodes[arc] = pair_count
            line_target_nodes[seg_tag] = pair_count

            ra = radial_lines[i]
            rb = radial_lines[(i + 1) % len(radial_lines)]
            radial_count = max(
                pair_count,
                line_target_nodes.get(ra, radial_nodes),
                line_target_nodes.get(rb, radial_nodes),
                radial_nodes,
            )
            line_target_nodes[ra] = radial_count
            line_target_nodes[rb] = radial_count

        for (iy, ix), tag in horizontal_lines.items():
            line_target_nodes[tag] = max(line_target_nodes.get(tag, 0), x_interval_nodes[ix])
        for (ix, iy), tag in vertical_lines.items():
            line_target_nodes[tag] = max(line_target_nodes.get(tag, 0), y_interval_nodes[iy])
        for i, seg in enumerate(square_segments):
            a = square_points[i]
            b = square_points[(i + 1) % len(square_points)]
            xa, ya = point_coords[a]
            xb, yb = point_coords[b]
            if abs(ya - yb) <= 1.0e-12:
                ix0 = _coord_index(x_coords, min(xa, xb))
                target = x_interval_nodes[ix0]
            elif abs(xa - xb) <= 1.0e-12:
                iy0 = _coord_index(y_coords, min(ya, yb))
                target = y_interval_nodes[iy0]
            else:
                target = line_target_nodes.get(abs(seg), _segment_target_nodes(a, b))
            line_target_nodes[abs(seg)] = max(line_target_nodes.get(abs(seg), 0), target)
            line_target_nodes[arc_lines[i]] = max(line_target_nodes.get(arc_lines[i], 0), target)

        for i in range(len(arc_lines)):
            pair = max(line_target_nodes.get(arc_lines[i], 0), line_target_nodes.get(abs(square_segments[i]), 0))
            line_target_nodes[arc_lines[i]] = pair
            line_target_nodes[abs(square_segments[i])] = pair
            rad_pair = max(
                line_target_nodes.get(radial_lines[i], 0),
                line_target_nodes.get(radial_lines[(i + 1) % len(radial_lines)], 0),
            )
            line_target_nodes[radial_lines[i]] = rad_pair
            line_target_nodes[radial_lines[(i + 1) % len(radial_lines)]] = rad_pair

        for i in range(len(circle_points)):
            next_i = (i + 1) % len(circle_points)
            loop = [
                arc_lines[i],
                radial_lines[next_i],
                -square_segments[i],
                -radial_lines[i],
            ]
            cloop = occ.addCurveLoop(loop)
            fluid_surfaces.append(occ.addPlaneSurface([cloop]))
            surface_loops.append(loop)

        for tag, loop in zip(fluid_surfaces, surface_loops):
            if len(loop) != 4:
                continue
            a = abs(loop[0])
            b = abs(loop[1])
            c = abs(loop[2])
            d = abs(loop[3])
            na = line_target_nodes.get(a)
            nc = line_target_nodes.get(c)
            nb = line_target_nodes.get(b)
            nd = line_target_nodes.get(d)
            if na != nc or nb != nd:
                def _edge_info(edge_tag: int) -> str:
                    pts = line_meta.get(abs(edge_tag))
                    if pts:
                        p0, p1 = pts
                        x0, y0 = point_coords[p0]
                        x1, y1 = point_coords[p1]
                        return f"{abs(edge_tag)}:({x0:.4f},{y0:.4f})->({x1:.4f},{y1:.4f})"
                    return str(abs(edge_tag))
                raise RuntimeError(
                    f"Transfinite mismatch on surface {tag}: "
                    f"edges {[ _edge_info(e) for e in loop ]} have node counts {(na, nc, nb, nd)}"
                )

        occ.synchronize()
        gmsh.model.mesh.setCompound(2, fluid_surfaces)

        gmsh.model.addPhysicalGroup(2, fluid_surfaces, tag=1)
        gmsh.model.setPhysicalName(2, 1, "fluid")
        boundary_tag_hints = {"inlet": 11, "outlet": 12, "walls": 13, "cylinder": 14}
        for name, tag_hint in boundary_tag_hints.items():
            edges = sorted(set(boundary_edges[name]))
            if not edges:
                continue
            tag = gmsh.model.addPhysicalGroup(1, edges, tag=tag_hint)
            gmsh.model.setPhysicalName(1, tag, name)

        radial_set = set(radial_lines)
        for tag, length in line_lengths.items():
            target_nodes = int(line_target_nodes.get(tag, _nodes_for_length(length, mesh_size)))
            progression = 1.0
            if tag in radial_set:
                progression = 1.12
            gmsh.model.mesh.setTransfiniteCurve(tag, target_nodes, "Progression", progression)

        for surf in fluid_surfaces:
            gmsh.model.mesh.setTransfiniteSurface(surf)
            gmsh.model.mesh.setRecombine(2, surf)

        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(int(poly_order))

        path.parent.mkdir(parents=True, exist_ok=True)
        if view:
            try:
                gmsh.fltk.initialize()
                gmsh.fltk.run()
            except Exception:
                print("Gmsh GUI not available; skipping mesh preview.")
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


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


def _load_gmsh_mesh(
    mesh_size: float,
    poly_order: int,
    *,
    mesh_file: Path | None,
    rebuild: bool,
    view: bool,
    beam_center: tuple[float, float] | None = None,
    beam_length: float | None = None,
    beam_height: float | None = None,
) -> tuple[Mesh, Path | None]:
    if gmsh is None:
        raise RuntimeError("Gmsh backend requested but gmsh is not available.")
    mesh_path = mesh_file.expanduser().resolve() if mesh_file is not None else None
    if mesh_path is not None:
        if rebuild or not mesh_path.exists():
            print(f"Generating gmsh blocked mesh at {mesh_path} (h={mesh_size}, Q{poly_order})")
            build_blocked_gmsh_mesh(
                mesh_path,
                mesh_size,
                poly_order,
                view=view,
                beam_center=beam_center,
                beam_length=beam_length,
                beam_height=beam_height,
            )
        else:
            print(f"Reusing gmsh mesh at {mesh_path}")
        return mesh_from_gmsh(mesh_path), mesh_path

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "turek_fsi_block.msh"
        print(f"Generating temporary gmsh blocked mesh (h={mesh_size}, Q{poly_order})")
        build_blocked_gmsh_mesh(
            tmp_path,
            mesh_size,
            poly_order,
            view=view,
            beam_center=beam_center,
            beam_length=beam_length,
            beam_height=beam_height,
        )
        mesh = mesh_from_gmsh(tmp_path)
    return mesh, None


def build_channel_mesh(
    mesh_size: float,
    poly_order: int,
    *,
    beam_center: tuple[float, float] | None = None,
    beam_length: float | None = None,
    beam_height: float | None = None,
) -> Mesh:
    """
    Select mesh backend (gmsh blocked grid or legacy structured O-grid) and
    validate the resulting mesh for Q{poly_order}.
    """
    backend = getattr(ARGS, "mesh_backend", "gmsh")
    if backend == "gmsh":
        mesh, _ = _load_gmsh_mesh(
            mesh_size,
            poly_order,
            mesh_file=getattr(ARGS, "mesh_file", None),
            rebuild=bool(getattr(ARGS, "rebuild_mesh", False)),
            view=bool(getattr(ARGS, "view_gmsh", False)),
            beam_center=beam_center,
            beam_length=beam_length,
            beam_height=beam_height,
        )
    else:
        mesh = build_structured_channel_mesh(mesh_size, poly_order)
    if mesh.element_type != "quad":
        raise RuntimeError(f"Expected a quadrilateral mesh, got {mesh.element_type}.")
    if int(mesh.poly_order) != int(poly_order):
        raise RuntimeError(f"Gmsh mesh order {mesh.poly_order} does not match requested Q{poly_order}.")
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


def _beam_root_locator(
    mesh: Mesh,
    beam_center: tuple[float, float],
    beam_length: float,
    beam_height: float,
    *,
    tol: float = 0.0,
):
    cx, cy = CENTER
    r2 = RADIUS * RADIUS
    beam_x0 = beam_center[0] - 0.5 * beam_length
    beam_y0 = beam_center[1] - 0.5 * beam_height
    beam_y1 = beam_center[1] + 0.5 * beam_height

    coords = np.asarray(getattr(mesh, "nodes_x_y_pos", []), float)
    xmin_raw, ymin_raw, xmax_raw, ymax_raw = 0.0, 0.0, L, H
    if coords.size:
        xmin_raw, ymin_raw = coords.min(axis=0)
        xmax_raw, ymax_raw = coords.max(axis=0)
    span = float(max(xmax_raw - xmin_raw, ymax_raw - ymin_raw, 1.0))
    tol_loc = max(tol, 1.0e-4 * span)
    tol_root = max(tol_loc, 4.5e-3 * span)
    tol_y = max(tol_loc, 8e-4 * span)
    tol_x_root = max(tol_loc, 1.5e-3 * span)

    def on_beam_root(x: float, y: float) -> bool:
        on_vertical = abs(x - beam_x0) < tol_x_root and (beam_y0 - tol_y) <= y <= (beam_y1 + tol_y)
        on_arc = (
            abs((x - cx) ** 2 + (y - cy) ** 2 - r2) < tol_root
            and (beam_y0 - tol_y) <= y <= (beam_y1 + tol_y)
            and x >= cx
        )
        return on_vertical or on_arc

    return on_beam_root


def _tag_beam_root_from_cylinder(
    dh: DofHandler,
    mesh: Mesh,
    locator,
    fields: Sequence[str],
    *,
    tag: str = "beam_root",
) -> None:
    loc_map = getattr(mesh, "_boundary_locators", {})
    loc_map[tag] = locator
    mesh._boundary_locators = loc_map

    edge_mask = np.zeros(len(mesh.edges_list), dtype=bool)
    try:
        candidates = mesh.edge_bitset("cylinder").to_indices()
    except Exception:
        candidates = np.arange(len(mesh.edges_list))
    for eid in candidates:
        try:
            e_obj = mesh.edge(int(eid))
        except Exception:
            continue
        if (e_obj.left is not None) and (e_obj.right is not None):
            continue
        mpx, mpy = mesh.nodes_x_y_pos[list(e_obj.nodes)].mean(axis=0)
        if locator(float(mpx), float(mpy)):
            edge_mask[int(eid)] = True

    if edge_mask.any():
        mesh._edge_bitsets = getattr(mesh, "_edge_bitsets", {})
        mesh._edge_bitsets[tag] = BitSet(edge_mask)
    else:
        return

    node_coords = mesh.nodes_x_y_pos
    for field in fields:
        node2dof = dh.dof_map.get(field, {})
        ids: set[int] = set()
        for eid in np.flatnonzero(edge_mask):
            e_obj = mesh.edge(int(eid))
            nodes = e_obj.all_nodes if getattr(e_obj, "all_nodes", ()) else e_obj.nodes
            for nid in nodes:
                x, y = node_coords[int(nid)]
                if locator(float(x), float(y)):
                    gd = node2dof.get(int(nid))
                    if gd is not None:
                        ids.add(int(gd))
        if ids:
            dh.dof_tags.setdefault(tag, set()).update(ids)


def _tag_beam_root_from_levelset(
    dh: DofHandler,
    beam_ls: "BeamArcRootLevelSet",
    fields: Sequence[str],
    *,
    tag: str = "beam_root",
    tol: float | None = None,
) -> int:
    if not hasattr(beam_ls, "_x_arc"):
        return 0
    if not fields:
        return 0
    _ = dh.get_field_slice(fields[0])
    coords = getattr(dh, "_dof_coords", None)
    if coords is None or not len(coords):
        return 0
    tol_x = float(BEAM_ROOT_DOF_TOL if tol is None else tol)
    tol_y = max(0.5 * tol_x, 1.0e-4)
    y = coords[:, 1]
    x_arc = beam_ls._x_arc(y)
    on_root = (
        (np.abs(coords[:, 0] - x_arc) <= tol_x)
        & (y >= beam_ls._beam_y0 - tol_y)
        & (y <= beam_ls._beam_y1 + tol_y)
        & (coords[:, 0] >= beam_ls._cyl_center[0] - tol_x)
    )
    added = 0
    for field in fields:
        ids = np.asarray(dh.get_field_slice(field), dtype=int)
        sel = ids[on_root[ids]]
        if sel.size:
            dh.dof_tags.setdefault(tag, set()).update(map(int, sel))
            added += int(sel.size)
    return added


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
    nloc = (p + 1) ** 2
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
        # refine_mode: 'left', 'right', 'bottom', 'top'
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


def asymmetric_refine_around_beam(mesh: Mesh, beam_ls: BeamLevelSet, levels: int = 2, band: float | None = None) -> Mesh:
    """
    Refine quads touching the beam level set with orientation bias:
    left half → vertical split, right half → horizontal split.
    Produces hanging nodes that are handled by the constraint layer.
    """
    if mesh.element_type != "quad":
        return mesh

    band = band or max(2.0 * beam_ls.hy, 3.0 * MESH_SIZE)
    center_x = float(getattr(beam_ls, "cx", BEAM_CENTER[0]))
    beam_xmin = float(beam_ls.cx - beam_ls.hx)
    beam_xmax = float(beam_ls.cx + beam_ls.hx)
    beam_ymin = float(beam_ls.cy - beam_ls.hy)
    beam_ymax = float(beam_ls.cy + beam_ls.hy)

    marked = set()
    for elem in mesh.elements_list:
        corners = mesh.nodes_x_y_pos[list(elem.corner_nodes)]
        phi_corner = beam_ls(corners)
        # Any corner inside/close to the beam → mark
        hits_phi = np.any(phi_corner <= 0.0) or np.any(np.abs(phi_corner) <= band)
        # Bounding box overlap (captures the beam body even if centroid/corners miss)
        ex_min, ey_min = corners.min(axis=0)
        ex_max, ey_max = corners.max(axis=0)
        hits_bbox = (ex_min <= beam_xmax + band and ex_max >= beam_xmin - band and ey_min <= beam_ymax + band and ey_max >= beam_ymin - band)
        if hits_phi or hits_bbox:
            marked.add(elem.id)
    # expand to neighbor layers
    for _ in range(max(0, levels - 1)):
        new = set()
        for eid in marked:
            for nb in mesh._neighbors[eid]:
                if nb is not None:
                    new.add(int(nb))
        marked |= new

    if not marked:
        return mesh

    nodes = list(mesh.nodes_list)
    node_lookup = {(round(nd.x, 14), round(nd.y, 14)): int(nd.id) for nd in nodes}
    new_elems = []
    new_corners = []

    for eid, elem in enumerate(mesh.elements_list):
        if eid not in marked:
            new_elems.append(list(mesh.elements_connectivity[eid]))
            new_corners.append(list(mesh.corner_connectivity[eid]))
            continue
        cx, _ = elem.centroid()
        orient = "vertical" if cx <= center_x else "horizontal"
        conns, corners = _refine_element_quads(mesh, eid, orient, nodes, node_lookup)
        new_elems.extend(conns)
        new_corners.extend(corners)

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
# Level-set update utilities
# -----------------------------------------------------------------------------


def update_beam_levelset_from_displacement(
    ls_beam: LevelSetGridFunction, disp_vec: VectorFunction, beam_ref_ls: Callable[[np.ndarray], float]
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
    # Rebuild interface classification/segments for aligned interface terms
    mesh.classify_elements(ls_beam)
    mesh.classify_edges(ls_beam)
    mesh.build_interface_segments(ls_beam)


def _copy_bitset(bs: BitSet) -> BitSet:
    return BitSet(np.array(bs.mask, dtype=bool))


def make_domain_sets(mesh: Mesh) -> Dict[str, BitSet]:
    fluid = mesh.element_bitset("outside")
    solid = mesh.element_bitset("inside")
    cut = mesh.element_bitset("cut")
    # These are not interface integrals; they are volume integrals
    fluid_ifc = fluid | cut   # instead of just cut
    solid_ifc = solid | cut
    has_pos = fluid | cut
    has_neg = solid | cut
    # Ghost penalties live on ghost edges only; interface edges are handled via dInterface.
    ghost_both = mesh.edge_bitset("ghost_both") if "ghost_both" in mesh._edge_bitsets else BitSet(np.zeros(len(mesh.edges_list), bool))
    solid_ghost = mesh.edge_bitset("ghost_neg") | ghost_both
    fluid_ghost = mesh.edge_bitset("ghost_pos") | ghost_both
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
    has_fluid = fluid | cut
    has_solid = solid | cut
    _update_bs(domains["fluid_domain"], fluid.mask)
    _update_bs(domains["solid_domain"], solid.mask)
    _update_bs(domains["cut_domain"], cut.mask)
    _update_bs(domains["fluid_interface"], has_fluid.mask)
    _update_bs(domains["solid_interface"], has_solid.mask)
    _update_bs(domains["has_pos"], fluid.mask | cut.mask)
    _update_bs(domains["has_neg"], solid.mask | cut.mask)
    ghost_both = mesh.edge_bitset("ghost_both") if "ghost_both" in mesh._edge_bitsets else BitSet(np.zeros(len(mesh.edges_list), bool))
    solid_ghost = mesh.edge_bitset("ghost_neg") | ghost_both
    fluid_ghost = mesh.edge_bitset("ghost_pos") | ghost_both
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


def retag_inactive(
    dh: DofHandler,
    *,
    theta_neg: np.ndarray | None = None,
    solid_cut_drop: float = 0.0,
) -> None:
    dh.dof_tags["inactive"] = set()
    dh.tag_dofs_from_element_bitset("inactive", "u_pos_x", "inside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "u_pos_y", "inside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "p_pos_", "inside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "vs_neg_x", "outside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "vs_neg_y", "outside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "d_neg_x", "outside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "d_neg_y", "outside", strict=True)
    if theta_neg is not None and solid_cut_drop > 0.0:
        cut_mask = mesh.element_bitset("cut").mask
        bad = cut_mask & (theta_neg < solid_cut_drop)
        if np.any(bad):
            for field in ("vs_neg_x", "vs_neg_y", "d_neg_x", "d_neg_y"):
                dh.tag_dofs_from_element_bitset("inactive", field, bad, strict=False)


def recompute_active_dofs(solver: NewtonSolver, bcs_active: Sequence[BoundaryCondition]) -> None:
    dh = solver.dh
    ndof_effective = solver.constraints.n_master if getattr(solver, "constraints", None) else dh.total_dofs
    map_to_master = (lambda ids: solver.constraints.to_master_set(ids)) if getattr(solver, "constraints", None) else (lambda ids: set(ids))
    active_by_restr, has_restriction = analyze_active_dofs(solver.equation, dh, solver.me, bcs_active)
    bc_dofs = set(dh.get_dirichlet_data(bcs_active).keys())
    candidate = map_to_master(active_by_restr) if has_restriction else set(range(ndof_effective))
    inactive = set(dh.dof_tags.get("inactive", set()))
    inactive_free = inactive - bc_dofs
    free = sorted((candidate - map_to_master(bc_dofs)) - map_to_master(inactive_free))
    solver.active_dofs = np.asarray(free, dtype=int)
    solver.full_to_red = -np.ones(ndof_effective, dtype=int)
    solver.full_to_red[solver.active_dofs] = np.arange(len(solver.active_dofs), dtype=int)
    solver.red_to_full = solver.active_dofs
    solver.use_reduced = len(solver.active_dofs) < ndof_effective
    solver.restrictor = _ActiveReducer(dh, solver.active_dofs, constraint=getattr(solver, "constraints", None))
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
    backend = os.getenv("FD_BACKEND", "jit")
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

# Beam level set (reference configuration)
beam_ref_center = BEAM_REF_CENTER
beam_ref_length = BEAM_REF_LENGTH
beam_ref_height = BEAM_REF_HEIGHT

# Mesh with rigid hole
mesh = build_channel_mesh(
    MESH_SIZE,
    POLY_ORDER,
    beam_center=beam_ref_center,
    beam_length=beam_ref_length,
    beam_height=beam_ref_height,
)

beam_ref_ls = BeamArcRootLevelSet(
    beam_center=beam_ref_center,
    beam_length=beam_ref_length,
    beam_height=beam_ref_height,
    cyl_center=CENTER,
    cyl_radius=RADIUS,
    root_inset=BEAM_ROOT_INSET,
    root_bias=BEAM_ROOT_BIAS,
    root_tol=BEAM_ROOT_TOL,
)

# Classify and extract interface geometry (required for aligned interface assembly)
mesh.classify_elements(beam_ref_ls)
mesh.classify_edges(beam_ref_ls)
mesh.build_interface_segments(beam_ref_ls)

# Optional asymmetric refinement around the beam to concentrate cells (with hanging nodes)
if ARGS.refine_initial:
    refine_levels = int(getattr(ARGS, "refine_levels", 2))
    refine_band = getattr(ARGS, "refine_band", None)
    mesh = asymmetric_refine_around_beam(mesh, beam_ref_ls, levels=refine_levels, band=refine_band)

# Re-classify after refinement to refresh cut/ghost/interface sets and segments
mesh.classify_elements(beam_ref_ls)
mesh.classify_edges(beam_ref_ls)
mesh.build_interface_segments(beam_ref_ls)

if ARGS.plot_mesh:
    ax = plot_mesh_2(mesh, level_set=beam_ref_ls, show=False)
    out_path = getattr(ARGS, "plot_mesh_file", None)
    if out_path:
        import matplotlib.pyplot as plt
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved mesh plot to {out_path}")
    else:
        import matplotlib.pyplot as plt
        plt.show()
    if getattr(ARGS, "plot_only", False):
        import sys
        sys.exit(0)

ls_me = MixedElement(mesh, field_specs={"phi_beam": 1})
ls_dh = DofHandler(ls_me, method="cg")
ls_beam = LevelSetGridFunction(ls_dh, field="phi_beam")
ls_beam.interpolate(lambda x, y: beam_ref_ls(np.array([x, y])))
ls_beam.commit()

domains = make_domain_sets(mesh)
print(
    f"Ghost edges: total={mesh.edge_bitset('ghost').cardinality()}, "
    f"pos + ghost_both={mesh.edge_bitset('ghost_pos').cardinality()}, "
    f"neg + ghost_both={mesh.edge_bitset('ghost_neg').cardinality()}, "
    f"both={mesh.edge_bitset('ghost_both').cardinality()}, "
    f"fluid_ghost(defined_on)={domains['fluid_ghost'].cardinality()}, "
    f"solid_ghost(defined_on)={domains['solid_ghost'].cardinality()}"
)

# Mixed element for fluid/solid unknowns
mixed_element = MixedElement(
    mesh,
    field_specs={
        "u_pos_x": POLY_ORDER,
        "u_pos_y": POLY_ORDER,
        "p_pos_": POLY_ORDER - 1,
        "vs_neg_x": POLY_ORDER ,
        "vs_neg_y": POLY_ORDER ,
        "d_neg_x": POLY_ORDER ,
        "d_neg_y": POLY_ORDER ,
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
if PIN_PRESSURE:
    pin_tag = "pressure_pin"
    dof_handler.tag_dof_by_locator(
        pin_tag,
        "p_pos_",
        locator=lambda x, y: abs(x - L) <= 1.0e-9 and abs(y - 0.5 * H) <= 1.0e-3,
        find_first=True,
    )
    bcs.append(BoundaryCondition("p_pos_", "dirichlet", pin_tag, lambda x, y: 0.0))

# Clamp beam at the circle interface
beam_clamp_tag = "beam_root"
beam_root_locator = _beam_root_locator(mesh, beam_ref_center, beam_ref_length, beam_ref_height)
beam_root_fields = ["vs_neg_x", "vs_neg_y", "d_neg_x", "d_neg_y"]
_tag_beam_root_from_cylinder(dof_handler, mesh, beam_root_locator, beam_root_fields, tag=beam_clamp_tag)
_tag_beam_root_from_levelset(dof_handler, beam_ref_ls, beam_root_fields, tag=beam_clamp_tag)
beam_root_dofs = len(dof_handler.dof_tags.get(beam_clamp_tag, set()))
if beam_root_dofs == 0:
    print("[warn] Beam root DOF tagging found no DOFs; check BEAM_ROOT_DOF_TOL or beam geometry.")
else:
    print(f"Beam root DOFs tagged: {beam_root_dofs}")
for field in beam_root_fields:
    bcs.append(BoundaryCondition(field, "dirichlet", beam_clamp_tag, lambda x, y: 0.0))

bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

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
retag_inactive(dof_handler, theta_neg=theta_neg_vals, solid_cut_drop=SOLID_CUT_DROP)

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

avg_flux_trial = kappa_pos * traction_fluid(Pos(du_f), Pos(dp_f)) + kappa_neg * traction_solid_L(Neg(ddisp_s), Neg(disp_k))
avg_flux_test = kappa_pos * traction_fluid(Pos(test_vel_f), -Pos(test_q_f)) + kappa_neg * traction_solid_L(Neg(test_disp_s), Neg(disp_k))
avg_flux_res = kappa_pos * traction_fluid(Pos(uf_k), Pos(pf_k)) + kappa_neg * traction_solid_R(Neg(disp_k))

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

sigma_s_k = sigma_s_nonlinear(disp_k_R)
sigma_s_n = sigma_s_nonlinear(disp_n_R)
dsigma_s_k = dsigma_s(disp_k_R, ddisp_s_R)

a_vol_s = (
    rho_s_const * dot(du_s_R, test_vel_s_R) / dt
    + theta * inner(dsigma_s_k, grad(test_vel_s_R))
    + rho_s_const * theta
      * (dot(dot(grad(us_k_R), du_s_R), test_vel_s_R) + dot(dot(grad(du_s_R), us_k_R), test_vel_s_R))
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

a_svc = (
    dot(ddisp_s_R, test_disp_s_R) / dt
    - theta * dot(du_s_R, test_disp_s_R)
    + theta * (dot(dot(grad(ddisp_s_R), us_k_R), test_disp_s_R) + dot(dot(grad(disp_k_R), du_s_R), test_disp_s_R))
) * dx_solid
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
solid_reg_eps = Constant(float(os.getenv("SOLID_REG_EPS", "1e-6")))


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

a_reg = solid_reg_eps * (dot(du_s_R, test_vel_s_R) + dot(ddisp_s_R, test_disp_s_R)) * dx_solid
r_reg = solid_reg_eps * (dot(us_k_R, test_vel_s_R) + dot(disp_k_R, test_disp_s_R)) * dx_solid

jacobian_form = a_vol_f + J_int + a_vol_s + a_svc + a_stab + a_reg
residual_form = r_vol_f + R_int + r_vol_s + r_svc + r_stab + r_reg

# ----------------------------------------------------------------------------- 
# Diagnostics: tip displacement, drag/lift (avg traction), pressure drop, VTK
# -----------------------------------------------------------------------------
REF_TIP = np.array([CENTER[0] + RADIUS + BEAM_LENGTH, CENTER[1]], dtype=float)
PROBE_B = np.array([CENTER[0] - 0.05, CENTER[1]], dtype=float)
obs_history = {"time": [], "tip": [], "drag": [], "lift": [], "dp": []}
output_dir = os.getenv("OUTPUT_DIR", "turek_results_fsi_ii_monolithic_full")
os.makedirs(output_dir, exist_ok=True)
SAVE_VTK = os.getenv("SAVE_VTK", "1") not in ("0", "false", "False")

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
if os.getenv("RUN_FD_CHECK", "0") != "0":
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
    if os.getenv("RUN_FD_TERMS", "0") != "0":
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
if os.getenv("RUN_TIME_STEPPING", "1") != "0":
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
        solver.refresh_levelset_kernels(ls_beam)
        retag_inactive(dof_handler, theta_neg=theta_neg_vals, solid_cut_drop=SOLID_CUT_DROP)
        dof_handler.apply_bcs(bcs, uf_k, pf_k, us_k, disp_k)
        recompute_active_dofs(solver, solver.bcs_homog if solver.bcs_homog else solver.bcs)
        _compute_observables(step_idx[0], step_idx[0] * dt_val)
        step_idx[0] += 1

    solver.post_timeloop_cb = post_step_cb

    solver.solve_time_interval(
        functions=[uf_k, pf_k, us_k, disp_k],
        prev_functions=[uf_n, pf_n, us_n, disp_n],
        time_params=time_params,
    )
