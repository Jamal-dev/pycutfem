#!/usr/bin/env python
"""
Conforming ALE FSI (Taylor–Hood) on the Turek–Hron channel with a Neo-Hookean solid.

- Geometry: channel with a circular hole and an attached beam meshed by a
  transfinite O-grid. The mesh is 100% quadrilateral and aligned with the beam
  box, so no CutFEM/cut elements appear. The circle is subtracted from the mesh.
- Formulation: ALE Navier–Stokes with mesh velocity from the displacement field,
  compressible Neo-Hookean solid, one theta-step solved monolithically.
- Diagnostics: optional finite-difference Jacobian check, drag/lift, and tip displacement.
- Warm start: solve a Stokes extension for the inflow profile and use it as the
  initial Newton guess.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Dict

import numpy as np

try:
    import gmsh  # type: ignore
except Exception:
    gmsh = None

from neo_hookean_verification import neo_hookean_pk1, neo_hookean_delta_P

from pycutfem.core.levelset import BeamLevelSet, CircleLevelSet, LevelSetFunction
from pycutfem.utils.bitset import BitSet
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.gmsh_loader import mesh_from_gmsh
from pycutfem.ufl.expressions import (
    TrialFunction,
    TestFunction,
    VectorTrialFunction,
    VectorTestFunction,
    VectorFunction,
    Function,
    Constant,
    Identity,
    grad,
    inner,
    trace,
    det,
    inv,
    dot,
    div,
)
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dx, dS
from pycutfem.ufl.forms import Equation, BoundaryCondition, assemble_form
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.solvers.nonlinear_solver import NewtonSolver, NewtonParameters, TimeStepperParameters

# Geometry / beam (Turek–Hron)
H = 0.41
L = 2.5
RADIUS = 0.05
CENTER = (0.2, 0.2)
BEAM_LENGTH = 0.35
BEAM_HEIGHT = 0.02
BEAM_CENTER = (CENTER[0] + RADIUS + 0.5 * BEAM_LENGTH, CENTER[1])

# Material parameters (from examples/debug/fsi_TaylorHood_xfem.py)
RHO_F = 1.0e3
NU_F = 1.0e-3  # kinematic viscosity
MU_F = NU_F    # dynamic viscosity (rho=1 reference)
RHO_S = 1.0e4
NU_S = 0.4
MU_S = 0.5e6
E_S = 2.0 * MU_S * (1.0 + NU_S)
LAMBDA_S = E_S * NU_S / ((1.0 + NU_S) * (1.0 - 2.0 * NU_S))


def symgrad(u):
    return 0.5 * (grad(u) + grad(u).T)


def Sym(u):
    return 0.5 * (u + u.T)


class BeamCircularRootLevelSet(LevelSetFunction):
    """
    Beam with a circular root: inside the rectangular beam box but outside
    the cylinder. The zero-set follows the circle on the left side.
    """

    def __init__(self, center, Lb, Hb, circle_center, radius, offset: float = 1e-6):
        self.beam = BeamLevelSet(center=center, Lb=Lb, Hb=Hb)
        self.circle = CircleLevelSet(center=circle_center, radius=radius)
        self.offset = float(offset)
        self.cache_token = ("beam_circ_root", self.beam.cache_token, float(radius), float(offset))

    def __call__(self, x):
        phi_beam = self.beam(x)
        phi_circ = self.circle(x)
        # Exclude the circle by shifting its signed distance slightly positive.
        return np.maximum(phi_beam, -phi_circ + self.offset)

    def gradient(self, x):
        # Pick the active branch.
        phi_beam = self.beam(x)
        phi_circ = self.circle(x)
        use_beam = phi_beam >= -phi_circ + self.offset
        g_beam = self.beam.gradient(x)
        g_circ = -self.circle.gradient(x)
        return np.where(np.expand_dims(use_beam, -1), g_beam, g_circ)


def classify_conforming(mesh, tol=1e-9):
    """
    Geometry-based classification: beam is the rectangular box attached to the
    circle, excluding the circular hole. This avoids cut elements for the
    conforming mesh.
    """
    # Prefer physical groups if present
    phys_map = getattr(mesh, "_element_physical_tags", None)
    if phys_map:
        for eid, tag in phys_map.items():
            mesh.elements_list[eid].tag = "solid" if tag == 2 else "fluid"
        tags_el = np.array([e.tag for e in mesh.elements_list])
        mesh._elem_bitsets = {t: BitSet(tags_el == t) for t in np.unique(tags_el)}
        return

    # Fallback: geometry check
    beam_x0 = CENTER[0] + RADIUS
    beam_x1 = beam_x0 + BEAM_LENGTH
    beam_y0 = CENTER[1] - 0.5 * BEAM_HEIGHT
    beam_y1 = CENTER[1] + 0.5 * BEAM_HEIGHT
    rad_tol = RADIUS + 1e-8
    cx, cy = mesh.nodes_x_y_pos[mesh.corner_connectivity].mean(axis=1).T  # element centroids
    inside_mask = (
        (cx >= beam_x0 - tol)
        & (cx <= beam_x1 + tol)
        & (cy >= beam_y0 - tol)
        & (cy <= beam_y1 + tol)
        & (np.hypot(cx - CENTER[0], cy - CENTER[1]) >= rad_tol)
    )
    outside_mask = ~inside_mask
    for eid in np.where(inside_mask)[0]:
        mesh.elements_list[eid].tag = "inside"
    for eid in np.where(outside_mask)[0]:
        mesh.elements_list[eid].tag = "outside"
    tags_el = np.array([e.tag for e in mesh.elements_list])
    mesh._elem_bitsets = {t: BitSet(tags_el == t) for t in np.unique(tags_el)}


def coverage_diagnostics(mesh, n_samples: int = 60, tol: float = 1.0e-6) -> dict[str, list[float]]:
    """
    Sample vertical and horizontal lines across the domain and report locations
    where the union of element intervals leaves a gap (potential holes).
    The circular hole is ignored on purpose.
    """

    def _scan(axis: str) -> list[float]:
        gaps: list[float] = []
        coords = mesh.nodes_x_y_pos
        if axis == "x":
            samples = np.linspace(coords[:, 0].min(), coords[:, 0].max(), n_samples)
            for x0 in samples:
                if abs(x0 - CENTER[0]) <= RADIUS + tol:
                    continue
                intervals = []
                for e in mesh.elements_list:
                    cn = coords[list(e.corner_nodes)]
                    xmin, xmax = cn[:, 0].min(), cn[:, 0].max()
                    if xmin - tol <= x0 <= xmax + tol:
                        ymin, ymax = cn[:, 1].min(), cn[:, 1].max()
                        intervals.append((ymin, ymax))
                if not intervals:
                    gaps.append(float(x0))
                    continue
                intervals.sort()
                low, high = intervals[0]
                for a, b in intervals[1:]:
                    if a > high + tol:
                        gaps.append(float(x0))
                        break
                    high = max(high, b)
                if low > coords[:, 1].min() + tol or high < coords[:, 1].max() - tol:
                    gaps.append(float(x0))
        else:
            samples = np.linspace(coords[:, 1].min(), coords[:, 1].max(), n_samples)
            for y0 in samples:
                if abs(y0 - CENTER[1]) <= RADIUS + tol:
                    continue
                intervals = []
                for e in mesh.elements_list:
                    cn = coords[list(e.corner_nodes)]
                    ymin, ymax = cn[:, 1].min(), cn[:, 1].max()
                    if ymin - tol <= y0 <= ymax + tol:
                        xmin, xmax = cn[:, 0].min(), cn[:, 0].max()
                        intervals.append((xmin, xmax))
                if not intervals:
                    gaps.append(float(y0))
                    continue
                intervals.sort()
                low, high = intervals[0]
                for a, b in intervals[1:]:
                    if a > high + tol:
                        gaps.append(float(y0))
                        break
                    high = max(high, b)
                if low > coords[:, 0].min() + tol or high < coords[:, 0].max() - tol:
                    gaps.append(float(y0))
        return gaps

    return {"gaps_x": _scan("x"), "gaps_y": _scan("y")}


def build_mesh(path: Path, h: float, order: int = 1, view: bool = False) -> Path:
    """
    Build a fully quadrilateral, block-structured mesh (O-grid around the
    cylinder with beam-aligned radial blocks). Every surface is 4-sided and
    transfinite so gmsh produces quads only.
    """
    if gmsh is None:
        raise RuntimeError("gmsh is required to build the mesh.")

    mesh_size = float(h)
    beam_x0 = CENTER[0] + RADIUS
    beam_x1 = beam_x0 + BEAM_LENGTH
    beam_y0 = CENTER[1] - 0.5 * BEAM_HEIGHT
    beam_y1 = CENTER[1] + 0.5 * BEAM_HEIGHT

    # Use the circle intersection with the beam top/bottom for meshing to align the curved root
    beam_x0_mesh = CENTER[0] + math.sqrt(max(RADIUS**2 - (0.5 * BEAM_HEIGHT) ** 2, 0.0))
    beam_x1_mesh = beam_x0_mesh + BEAM_LENGTH

    pad = max(0.6 * mesh_size, 0.008)
    hx = max(RADIUS + pad, min(CENTER[0] - pad, L - CENTER[0] - pad, 0.35))
    hy = max(RADIUS + pad, min(CENTER[1] - pad, H - CENTER[1] - pad, 0.35))
    if hx <= RADIUS or hy <= RADIUS:
        raise RuntimeError("O-grid collapsed; increase mesh size or adjust padding.")

    square_left = CENTER[0] - hx
    square_right = CENTER[0] + hx
    square_bottom = CENTER[1] - hy
    square_top = CENTER[1] + hy

    def _nodes_for_length(length: float, target: float, *, min_nodes: int = 3) -> int:
        segments = max(1, int(round(length / target)))
        return max(min_nodes, segments + 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.model.add("fsi_conforming_blocked")
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
            tag = occ.addPoint(float(x), float(y), 0.0)
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
            line_meta[tag] = (start, end)

        def oriented_line(start: int, end: int) -> int:
            tag = line_lookup.get((start, end))
            if tag is not None:
                return tag
            tag = line_lookup.get((end, start))
            if tag is None:
                raise KeyError(f"No curve between points {start} and {end}.")
            return -tag

        boundary_edges: dict[str, list[int]] = {
            "inlet": [],
            "outlet": [],
            "walls": [],
            "cylinder": [],
            "beam_root": [],
            "beam_outer": [],
        }

        def add_line(start: int, end: int, boundary: str | None = None) -> int:
            tag = occ.addLine(start, end)
            x0, y0 = point_coords[start]
            x1, y1 = point_coords[end]
            length = math.hypot(x1 - x0, y1 - y0)
            register_line(tag, start, end, length)
            if boundary:
                boundary_edges[boundary].append(tag)
            return tag

        beam_nodes = max(4, _nodes_for_length(BEAM_HEIGHT, mesh_size, min_nodes=4))
        beam_segments = max(2, _nodes_for_length(BEAM_LENGTH, mesh_size, min_nodes=3) - 1)
        beam_split_x = [
            beam_x0_mesh + i * BEAM_LENGTH / beam_segments for i in range(1, beam_segments)
        ]

        # Global Cartesian grid lines, including inner-square sides and beam box
        x_coords = sorted({0.0, square_left, square_right, beam_x0_mesh, beam_x1_mesh, L, *beam_split_x})
        y_coords = sorted({0.0, square_bottom, beam_y0, CENTER[1], beam_y1, square_top, H})

        x_interval_nodes = []
        for ix in range(len(x_coords) - 1):
            length = x_coords[ix + 1] - x_coords[ix]
            x_interval_nodes.append(_nodes_for_length(length, mesh_size))

        y_interval_nodes = []
        for iy in range(len(y_coords) - 1):
            length = y_coords[iy + 1] - y_coords[iy]
            count = _nodes_for_length(length, mesh_size)
            # Ensure at least beam_nodes along verticals that intersect the inner square
            if y_coords[iy] >= square_bottom - 1.0e-12 and y_coords[iy + 1] <= square_top + 1.0e-12:
                count = max(count, beam_nodes)
            y_interval_nodes.append(count)

        grid_points: dict[tuple[int, int], int] = {}
        for ix, x in enumerate(x_coords):
            for iy, y in enumerate(y_coords):
                grid_points[(ix, iy)] = add_point(x, y)

        def _inside_inner(x_mid: float, y_mid: float) -> bool:
            return (square_left < x_mid < square_right) and (square_bottom < y_mid < square_top)

        def _coord_index(seq: list[float], value: float) -> int:
            for i, v in enumerate(seq):
                if abs(v - value) <= 1.0e-12:
                    return i
            raise ValueError(f"{value} not found in coordinate list.")

        # Axis-aligned grid outside the inner square -------------------------
        horizontal_lines: dict[tuple[int, int], int] = {}
        for iy, y in enumerate(y_coords):
            for ix in range(len(x_coords) - 1):
                x0, x1 = x_coords[ix], x_coords[ix + 1]
                x_mid = 0.5 * (x0 + x1)
                
                # Standard inner-square check
                if _inside_inner(x_mid, y):
                    continue

                # --- FIX 1: Suppress horizontal lines running through the beam strip ---
                # Check if strictly inside beam height AND inside the beam extent (not the whole right domain).
                if (
                    beam_y0 + 1e-12 < y < beam_y1 - 1e-12
                    and square_right - 1e-12 < x_mid < beam_x1_mesh + 1e-12
                ):
                    continue
                # Suppress the midline beyond the beam to avoid a redundant row (hanging nodes).
                if abs(y - CENTER[1]) <= 1.0e-12 and x0 >= beam_x1_mesh - 1.0e-12:
                    continue
                # -----------------------------------------------------------------------

                boundary = None
                if abs(y - 0.0) <= 1.0e-12 or abs(y - H) <= 1.0e-12:
                    boundary = "walls"
                elif abs(y - beam_y0) <= 1.0e-12 or abs(y - beam_y1) <= 1.0e-12:
                    if max(x0, x1) >= beam_x0_mesh - 1.0e-12:
                        boundary = "beam_outer"
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
                elif abs(x - beam_x1_mesh) <= 1.0e-12 and (y0 >= beam_y0 - 1.0e-12) and (y1 <= beam_y1 + 1.0e-12):
                    boundary = "beam_outer"
                tag = add_line(grid_points[(ix, iy)], grid_points[(ix, iy + 1)], boundary=boundary)
                vertical_lines[(ix, iy)] = tag
                line_target_nodes[tag] = max(line_target_nodes.get(tag, 0), y_interval_nodes[iy])

        fluid_surfaces: list[int] = []
        solid_surfaces: list[int] = []
        all_surfaces: list[int] = []
        surface_loops: list[list[int]] = []
        tol = 1.0e-12

        # Create rectangular blocks outside the inner square
        for ix in range(len(x_coords) - 1):
            for iy in range(len(y_coords) - 1):
                xm = 0.5 * (x_coords[ix] + x_coords[ix + 1])
                ym = 0.5 * (y_coords[iy] + y_coords[iy + 1])
                if _inside_inner(xm, ym):
                    continue
                if (beam_x0_mesh - tol) <= xm <= (beam_x1_mesh + tol) and (beam_y0 - tol) <= ym <= (beam_y1 + tol):
                    continue  # beam strip handled separately (to align with curved root)
                
                # --- FIX: Ensure we don't try to use non-existent lines ---
                l_bot = horizontal_lines.get((iy, ix))
                l_right = vertical_lines.get((ix + 1, iy))
                l_top = horizontal_lines.get((iy + 1, ix))
                l_left = vertical_lines.get((ix, iy))
                
                if l_bot is None or l_right is None or l_top is None or l_left is None:
                    continue # Skip blocks where lines were suppressed
                
                loop = [l_bot, l_right, -l_top, -l_left]
                cloop = occ.addCurveLoop(loop)
                surf = occ.addPlaneSurface([cloop])
                all_surfaces.append(surf)
                surface_loops.append(loop)
                if (beam_x0 - tol) <= xm <= (beam_x1 + tol) and (beam_y0 - tol) <= ym <= (beam_y1 + tol):
                    solid_surfaces.append(surf)
                else:
                    fluid_surfaces.append(surf)

        # Beam-aware O-grid ring around the circle ---------------------------
        x_right_idx = _coord_index(x_coords, square_right)
        x_left_idx = _coord_index(x_coords, square_left)
        y_bot_idx = _coord_index(y_coords, square_bottom)
        y_top_idx = _coord_index(y_coords, square_top)
        y_beam_bot_idx = _coord_index(y_coords, beam_y0)
        y_beam_top_idx = _coord_index(y_coords, beam_y1)
        y_mid_idx = _coord_index(y_coords, CENTER[1])

        y_seq = [y_bot_idx, y_beam_bot_idx, y_mid_idx, y_beam_top_idx, y_top_idx]
        top_x_indices = sorted(
            {idx for idx, xv in enumerate(x_coords) if square_left - 1e-12 <= xv <= square_right + 1e-12},
            reverse=True,
        )

        def _append_unique(coords_list: list[tuple[int, int]], coord: tuple[int, int]) -> None:
            if not coords_list or coords_list[-1] != coord:
                coords_list.append(coord)

        square_point_coords: list[tuple[int, int]] = []
        for yi in y_seq:
            _append_unique(square_point_coords, (x_right_idx, yi))
        for idx in top_x_indices[1:]:
            _append_unique(square_point_coords, (idx, y_top_idx))
        for yi in reversed(y_seq[:-1]):
            _append_unique(square_point_coords, (x_left_idx, yi))
        bot_x_indices = sorted({idx for idx in top_x_indices}, reverse=False)
        for idx in bot_x_indices[1:]:
            _append_unique(square_point_coords, (idx, y_bot_idx))
        if square_point_coords and square_point_coords[-1] == square_point_coords[0]:
            square_point_coords.pop()

        square_points = [grid_points[(ix, iy)] for ix, iy in square_point_coords]
        center_pt = add_point(CENTER[0], CENTER[1])

        def _arc_length(a_tag: int, b_tag: int) -> float:
            xa, ya = point_coords[a_tag]
            xb, yb = point_coords[b_tag]
            ang_a = math.atan2(ya - CENTER[1], xa - CENTER[0])
            ang_b = math.atan2(yb - CENTER[1], xb - CENTER[0])
            delta = ang_b - ang_a
            if delta <= 0.0:
                delta += 2.0 * math.pi
            return RADIUS * delta

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

        def _circle_point_on_right_at_y(y_target: float) -> int:
            dx = math.sqrt(max(RADIUS**2 - (y_target - CENTER[1]) ** 2, 0.0))
            return add_point(CENTER[0] + dx, y_target)

        # Snap the beam-side circle points to the exact y-levels of the beam so
        # the transition quad shares its edges cleanly (avoids stray radials).
        beam_circle_bottom = _circle_point_on_right_at_y(beam_y0)
        beam_circle_top = _circle_point_on_right_at_y(beam_y1)
        for idx, pid in enumerate(square_points):
            xs, ys = point_coords[pid]
            if xs >= CENTER[0] - tol and abs(ys - beam_y0) <= tol:
                circle_points[idx] = beam_circle_bottom
            elif xs >= CENTER[0] - tol and abs(ys - beam_y1) <= tol:
                circle_points[idx] = beam_circle_top
        
        # --- FIX 2: Helper to identify segments inside or ON the beam boundary ---
        # We want to suppress small segments that are superseded by the large beam block lines.
        def _is_beam_span_segment(p1: int, p2: int) -> bool:
            y1 = point_coords[p1][1]
            y2 = point_coords[p2][1]
            y_mid = 0.5 * (y1 + y2)
            x_mid = 0.5 * (point_coords[p1][0] + point_coords[p2][0])
            # Only suppress the beam-height arcs on the beam (right) side; keep the left side intact.
            return (x_mid >= CENTER[0] - 1e-9) and (y_mid > beam_y0 + 1e-9) and (y_mid < beam_y1 - 1e-9)

        arc_lines: list[int] = []
        for i in range(len(circle_points)):
            start = circle_points[i]
            end = circle_points[(i + 1) % len(circle_points)]
            
            if _is_beam_span_segment(start, end):
                 arc_lines.append(0)
                 continue

            tag = occ.addCircleArc(start, center_pt, end)
            length = _arc_length(start, end)
            register_line(tag, start, end, length)
            arc_lines.append(tag)
            line_target_nodes[tag] = max(line_target_nodes.get(tag, 0), _nodes_for_length(length, mesh_size))
            x_mid = 0.5 * (point_coords[start][0] + point_coords[end][0])
            y_mid = 0.5 * (point_coords[start][1] + point_coords[end][1])
            on_root = (x_mid >= CENTER[0] - tol) and (beam_y0 - tol <= y_mid <= beam_y1 + tol)
            boundary_edges["beam_root" if on_root else "cylinder"].append(tag)
        
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
        
        for idx, (c_pt, s_pt) in enumerate(zip(circle_points, square_points)):
            tag = add_line(c_pt, s_pt)
            default_nodes = max(radial_nodes, beam_nodes)
            line_target_nodes[tag] = max(default_nodes, line_target_nodes.get(tag, default_nodes))
            radial_lines.append(tag)

        # Harmonize node counts
        for i, arc in enumerate(arc_lines):
            seg_tag_raw = square_segments[i]
            seg_tag = abs(seg_tag_raw) if seg_tag_raw != 0 else 0
            
            a = square_points[i]
            b = square_points[(i + 1) % len(square_points)]
            seg_nodes = _segment_target_nodes(a, b)
            
            if arc != 0:
                arc_nodes = _nodes_for_length(line_lengths[arc], mesh_size)
            else:
                arc_nodes = 0
            
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
            
            if arc != 0: line_target_nodes[arc] = pair_count
            if seg_tag != 0: line_target_nodes[seg_tag] = pair_count

            ra = radial_lines[i]
            rb = radial_lines[(i + 1) % len(radial_lines)]
            
            target_ra = line_target_nodes.get(ra, radial_nodes) if ra != 0 else radial_nodes
            target_rb = line_target_nodes.get(rb, radial_nodes) if rb != 0 else radial_nodes

            radial_count = max(pair_count, target_ra, target_rb, radial_nodes)
            if ra != 0: line_target_nodes[ra] = radial_count
            if rb != 0: line_target_nodes[rb] = radial_count

        # Propagate back to grid lines
        for (iy, ix), tag in horizontal_lines.items():
            line_target_nodes[tag] = max(line_target_nodes.get(tag, 0), x_interval_nodes[ix])
        for (ix, iy), tag in vertical_lines.items():
            line_target_nodes[tag] = max(line_target_nodes.get(tag, 0), y_interval_nodes[iy])
        for i, seg_raw in enumerate(square_segments):
            if seg_raw == 0: continue
            seg = abs(seg_raw)
            a = square_points[i]
            b = square_points[(i + 1) % len(square_points)]
            target = line_target_nodes.get(seg, _segment_target_nodes(a, b))
            line_target_nodes[seg] = max(line_target_nodes.get(seg, 0), target)

        n_pairs = min(len(arc_lines), len(square_segments), len(radial_lines))
        for i in range(n_pairs):
            arc_tag = arc_lines[i]
            seg_tag_raw = square_segments[i]
            seg_tag = abs(seg_tag_raw) if seg_tag_raw != 0 else 0

            target_arc = line_target_nodes.get(arc_tag, 0) if arc_tag != 0 else 0
            target_seg = line_target_nodes.get(seg_tag, 0) if seg_tag != 0 else 0
            pair = max(target_arc, target_seg)
            
            if arc_tag != 0: line_target_nodes[arc_tag] = pair
            if seg_tag != 0: line_target_nodes[seg_tag] = pair

            ra = radial_lines[i]
            rb = radial_lines[(i + 1) % len(radial_lines)]
            target_ra = line_target_nodes.get(ra, 0) if ra != 0 else 0
            target_rb = line_target_nodes.get(rb, 0) if rb != 0 else 0
            rad_pair = max(target_ra, target_rb, pair)
            if ra != 0: line_target_nodes[ra] = rad_pair
            if rb != 0: line_target_nodes[rb] = rad_pair

        # Beam root wedge (single quad with arc and vertical side)
        def _circle_point_at_y(y_target: float) -> int:
            for pid in circle_points:
                xx, yy = point_coords[pid]
                if abs(yy - y_target) <= 1.0e-12 and xx >= CENTER[0]:
                    return pid
            dx = math.sqrt(max(RADIUS**2 - (y_target - CENTER[1]) ** 2, 0.0))
            return add_point(CENTER[0] + dx, y_target)

        beam_bot_pt = _circle_point_at_y(beam_y0)
        beam_top_pt = _circle_point_at_y(beam_y1)
        beam_arc = occ.addCircleArc(beam_bot_pt, center_pt, beam_top_pt)
        register_line(beam_arc, beam_bot_pt, beam_top_pt, _arc_length(beam_bot_pt, beam_top_pt))
        line_target_nodes[beam_arc] = max(line_target_nodes.get(beam_arc, 0), beam_nodes)

        beam_right_bot = grid_points[(x_right_idx, y_beam_bot_idx)]
        beam_right_top = grid_points[(x_right_idx, y_beam_top_idx)]
        beam_vertical = add_line(beam_right_bot, beam_right_top, boundary="beam_outer")
        line_target_nodes[beam_vertical] = max(line_target_nodes.get(beam_vertical, 0), beam_nodes)

        try:
            beam_top_line = oriented_line(beam_top_pt, beam_right_top)
        except KeyError:
            beam_top_line = add_line(beam_top_pt, beam_right_top, boundary="beam_outer")
        try:
            beam_bot_line = oriented_line(beam_right_bot, beam_bot_pt)
        except KeyError:
            beam_bot_line = add_line(beam_right_bot, beam_bot_pt, boundary="beam_outer")
        span_nodes = max(
            _segment_target_nodes(beam_top_pt, beam_right_top),
            _segment_target_nodes(beam_right_bot, beam_bot_pt),
        )
        line_target_nodes[abs(beam_top_line)] = max(line_target_nodes.get(abs(beam_top_line), 0), span_nodes)
        line_target_nodes[abs(beam_bot_line)] = max(line_target_nodes.get(abs(beam_bot_line), 0), span_nodes)

        beam_root_loop = [beam_arc, beam_top_line, -beam_vertical, beam_bot_line]
        beam_root_cloop = occ.addCurveLoop(beam_root_loop)
        beam_root_surf = occ.addPlaneSurface([beam_root_cloop])
        all_surfaces.append(beam_root_surf)
        surface_loops.append(beam_root_loop)
        solid_surfaces.append(beam_root_surf)
        boundary_edges["beam_root"] = [beam_arc]
        boundary_edges["beam_outer"].extend([abs(beam_top_line), abs(beam_bot_line), abs(beam_vertical)])

        # Locate the right-side inner-square indices that sit on the beam edges.
        def _find_square_index_at_y(y_target: float) -> int:
            for idx, pid in enumerate(square_points):
                x, y = point_coords[pid]
                if abs(y - y_target) <= 1.0e-12 and abs(x - square_right) <= 1.0e-9:
                    return idx
            raise RuntimeError(f"No inner-square point found at y={y_target}.")

        beam_bottom_idx = _find_square_index_at_y(beam_y0)
        beam_top_idx = _find_square_index_at_y(beam_y1)
        beam_bottom_radial = radial_lines[beam_bottom_idx]
        beam_top_radial = radial_lines[beam_top_idx]

        def _segment_on_beam(idx: int) -> bool:
            a = square_points[idx]
            b = square_points[(idx + 1) % len(square_points)]
            xa, ya = point_coords[a]
            xb, yb = point_coords[b]
            x_mid = 0.5 * (xa + xb)
            y_min = min(ya, yb)
            y_max = max(ya, yb)
            return abs(x_mid - square_right) <= 1.0e-9 and (y_min >= beam_y0 - tol) and (y_max <= beam_y1 + tol)

        beam_segment_indices = [i for i in range(len(circle_points)) if _segment_on_beam(i)]

        for i in range(len(circle_points)):
            if i in beam_segment_indices:
                continue 
            
            arc = arc_lines[i]
            rad_curr = radial_lines[i]
            rad_next = radial_lines[(i + 1) % len(radial_lines)]
            seg = square_segments[i]

            if 0 in (arc, rad_curr, rad_next, seg):
                continue

            loop = [arc, rad_next, -seg, -rad_curr]
            cloop = occ.addCurveLoop(loop)
            surf = occ.addPlaneSurface([cloop])
            all_surfaces.append(surf)
            surface_loops.append(loop)
            if _segment_on_beam(i):
                solid_surfaces.append(surf)
            else:
                fluid_surfaces.append(surf)

        # Fill the beam-height band to the right of the beam when the centerline is suppressed
        def _edge_between(ix: int, iy0: int, iy1: int) -> int:
            a = grid_points[(ix, iy0)]
            b = grid_points[(ix, iy1)]
            try:
                return oriented_line(a, b)
            except KeyError:
                return add_line(a, b)

        start_ix = _coord_index(x_coords, beam_x1_mesh)
        for ix in range(start_ix, len(x_coords) - 1):
            bottom = horizontal_lines[(y_beam_bot_idx, ix)]
            top = horizontal_lines[(y_beam_top_idx, ix)]
            left = _edge_between(ix, y_beam_bot_idx, y_beam_top_idx)
            right = _edge_between(ix + 1, y_beam_bot_idx, y_beam_top_idx)

            # update node counts
            span_count = _segment_target_nodes(grid_points[(ix, y_beam_bot_idx)], grid_points[(ix, y_beam_top_idx)])
            line_target_nodes[abs(left)] = max(line_target_nodes.get(abs(left), 0), span_count)
            line_target_nodes[abs(right)] = max(line_target_nodes.get(abs(right), 0), span_count)

            loop = [bottom, right, -top, -left]
            cloop = occ.addCurveLoop(loop)
            surf = occ.addPlaneSurface([cloop])
            all_surfaces.append(surf)
            surface_loops.append(loop)
            fluid_surfaces.append(surf)

        # Beam strip
        beam_chain = [x for x in x_coords if x >= square_right - tol and x <= beam_x1_mesh + tol]
        beam_chain = sorted(beam_chain)
        beam_bottom_pts = [add_point(x, beam_y0) for x in beam_chain]
        beam_top_pts = [add_point(x, beam_y1) for x in beam_chain]
        
        for i in range(len(beam_chain) - 1):
            p0 = beam_bottom_pts[i]
            p1 = beam_bottom_pts[i + 1]
            p2 = beam_top_pts[i + 1]
            p3 = beam_top_pts[i]
            
            def _edge(a: int, b: int, boundary: str | None = None) -> int:
                try:
                    return oriented_line(a, b)
                except KeyError:
                    return add_line(a, b, boundary=boundary)

            bottom = _edge(p0, p1, boundary="beam_outer")
            right = _edge(p1, p2)
            top = _edge(p2, p3, boundary="beam_outer")
            
            # --- FIX 4: Reuse beam_vertical for the first block's left edge ---
            if i == 0:
                # The first block's left edge is p3(Top) -> p0(Bot).
                # beam_vertical is Bot -> Top. So we use -beam_vertical.
                left = -beam_vertical
            else:
                left = _edge(p3, p0)

            horiz_count = _segment_target_nodes(p0, p1)
            line_target_nodes[abs(bottom)] = max(line_target_nodes.get(abs(bottom), 0), horiz_count)
            line_target_nodes[abs(top)] = max(line_target_nodes.get(abs(top), 0), horiz_count)
            line_target_nodes[abs(left)] = max(line_target_nodes.get(abs(left), 0), beam_nodes)
            line_target_nodes[abs(right)] = max(line_target_nodes.get(abs(right), 0), beam_nodes)

            try:
                cloop = occ.addCurveLoop([bottom, right, top, left])
            except Exception:
                raise RuntimeError(f"Failed beam strip loop at i={i}")
            surf = occ.addPlaneSurface([cloop])
            all_surfaces.append(surf)
            surface_loops.append([bottom, right, top, left])
            solid_surfaces.append(surf)

            if i == len(beam_chain) - 2:
                boundary_edges["beam_outer"].append(abs(right))
            boundary_edges["beam_outer"].extend([abs(top), abs(bottom)])

        # (Optional) dangling radial cleanup could go here; keep radials to preserve coverage.

        # Sanity-check
        for loop in surface_loops:
            if len(loop) != 4: continue
            a, b, c, d = map(abs, loop)
            na = line_target_nodes.get(a, _nodes_for_length(line_lengths.get(a, mesh_size), mesh_size))
            nc = line_target_nodes.get(c, _nodes_for_length(line_lengths.get(c, mesh_size), mesh_size))
            nb = line_target_nodes.get(b, _nodes_for_length(line_lengths.get(b, mesh_size), mesh_size))
            nd = line_target_nodes.get(d, _nodes_for_length(line_lengths.get(d, mesh_size), mesh_size))
            pair_ac = max(na, nc)
            pair_bd = max(nb, nd)
            line_target_nodes[a] = pair_ac
            line_target_nodes[c] = pair_ac
            line_target_nodes[b] = pair_bd
            line_target_nodes[d] = pair_bd

        for tag, loop in zip(all_surfaces, surface_loops):
            if len(loop) != 4: continue
            a, b, c, d = map(abs, loop)
            na = line_target_nodes.get(a)
            nc = line_target_nodes.get(c)
            nb = line_target_nodes.get(b)
            nd = line_target_nodes.get(d)
            if na != nc or nb != nd:
                raise RuntimeError(f"Transfinite mismatch on surface {tag}")

        occ.synchronize()

        if not fluid_surfaces or not solid_surfaces:
            raise RuntimeError("Failed to build sets.")

        existing_curves = {t for dim, t in gmsh.model.getEntities(1)}
        for tag, length in line_lengths.items():
            if tag not in existing_curves or tag == 0:
                continue
            target_nodes = int(line_target_nodes.get(tag, _nodes_for_length(length, mesh_size)))
            progression = 1.12 if tag in set(radial_lines) else 1.0
            gmsh.model.mesh.setTransfiniteCurve(tag, target_nodes, "Progression", progression)

        for surf in all_surfaces:
            gmsh.model.mesh.setTransfiniteSurface(surf)
            gmsh.model.mesh.setRecombine(2, surf)

        gmsh.model.addPhysicalGroup(2, fluid_surfaces, tag=1)
        gmsh.model.setPhysicalName(2, 1, "fluid")
        gmsh.model.addPhysicalGroup(2, solid_surfaces, tag=2)
        gmsh.model.setPhysicalName(2, 2, "solid")

        existing_curves = {t for dim, t in gmsh.model.getEntities(1)}
        for key, edges in boundary_edges.items():
            boundary_edges[key] = [e for e in edges if abs(e) in existing_curves]

        boundary_tag_ids = {"inlet": 11, "walls": 12, "outlet": 13, "cylinder": 14, "beam_outer": 15, "beam_root": 16}
        for name, edges in boundary_edges.items():
            if not edges: continue
            phys_tag = boundary_tag_ids.get(name)
            tag = gmsh.model.addPhysicalGroup(1, sorted(set(edges)), tag=phys_tag)
            gmsh.model.setPhysicalName(1, tag, name)

        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(int(order))
        print(f"Debug: fluid surfaces={len(fluid_surfaces)}, solid surfaces={len(solid_surfaces)}, all={len(all_surfaces)}")

        if view:
            try:
                gmsh.fltk.initialize()
                gmsh.fltk.run()
            except Exception:
                print("Gmsh GUI not available.")
        gmsh.write(str(path))
        return path
    finally:
        gmsh.finalize()



def retag_boundaries(mesh):
    xmin, xmax = 0.0, L
    ymin, ymax = 0.0, H
    beam_x0 = CENTER[0] + RADIUS
    beam_x1 = beam_x0 + BEAM_LENGTH
    beam_y0 = CENTER[1] - 0.5 * BEAM_HEIGHT
    beam_y1 = beam_y0 + BEAM_HEIGHT
    tol = 1e-8
    mesh.tag_boundary_edges(
        {
            "inlet": lambda x, y: abs(x - xmin) < tol,
            "outlet": lambda x, y: abs(x - xmax) < tol,
            "walls": lambda x, y: abs(y - ymin) < tol or abs(y - ymax) < tol,
            "cylinder": lambda x, y: abs((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2 - RADIUS**2) < 1e-6,
            "beam_root": lambda x, y: abs(x - beam_x0) < tol and (beam_y0 - tol) <= y <= (beam_y1 + tol),
            "beam_outer": lambda x, y: (beam_x0 - tol) <= x <= (beam_x1 + tol) and (beam_y0 - tol) <= y <= (beam_y1 + tol),
        }
    )


def pick_probe_dofs(dh: DofHandler, bcs: Iterable[BoundaryCondition], per_field: int = 2) -> np.ndarray:
    bc_dofs = set(dh.get_dirichlet_data(bcs).keys())
    probes: list[int] = []
    for field in dh.mixed_element.field_names:
        taken = 0
        for gd in dh.get_field_slice(field):
            if int(gd) in bc_dofs:
                continue
            probes.append(int(gd))
            taken += 1
            if taken >= per_field:
                break
    return np.array(probes, dtype=int)


def finite_difference_check(
    jac_form,
    res_form,
    dh: DofHandler,
    bcs: Iterable[BoundaryCondition],
    functions: Dict[str, VectorFunction | Function],
    probe_dofs: Iterable[int],
    eps: float = 1.0e-6,
    backend: str = "jit",
) -> None:
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

    bc_dofs = set(dh.get_dirichlet_data(bcs).keys())
    rows = []
    for gdof in probe_dofs:
        field, _ = dh._dof_to_node_map[int(gdof)]
        if field not in functions or int(gdof) in bc_dofs:
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
        print(f"  {gd:5d}  {fld:6s}  err={err:9.3e}  |J|={mag:9.3e}  rel={rel:9.3e}")

def CalcStresses(A,I2 = Identity(2)):
    F = A + I2  
    C = dot(F.T, F)
    E = 0.5 * (C - I2)
    J = det(F)
    Finv = inv(F)
    return (F, C, E, J, Finv)
def build_forms(
    *,
    uk,
    u_prev,
    dk,
    d_prev,
    pk,
    p_prev,   # (kept for signature compatibility; not used here)
    du,
    dd,
    dp,
    v,
    w,
    q,
    dt_const: Constant,
    theta_const: Constant,
    rho_f: Constant,
    nu_f: Constant,
    rho_s: Constant,
    c10: Constant,
    kappa: Constant,
    mesh_reg: Constant,
    stab_eps: Constant,
    fluid_bs,
    solid_bs,
    quad_order: int,
):
    dx_f = dx(defined_on=fluid_bs, metadata={"q": quad_order})
    dx_s = dx(defined_on=solid_bs, metadata={"q": quad_order})

    I2 = Identity(2)
    one_minus_theta = Constant(1.0) - theta_const

    # --- Kinematics ---
    Fk = I2 + grad(dk)
    J = det(Fk)
    Finv = inv(Fk)

    F_old = I2 + grad(d_prev)
    J_old = det(F_old)
    Finv_old = inv(F_old)

    # Mesh velocity (ALE): depends on current dk
    mesh_vel = (dk - d_prev) / dt_const

    # --- Physical-domain gradients via pull-back ---
    grad_uk_phys = dot(grad(uk), Finv)
    grad_v_phys = dot(grad(v), Finv)

    eps_uk = Sym(grad_uk_phys)
    eps_v = Sym(grad_v_phys)

    div_uk = trace(grad_uk_phys)
    div_v = trace(grad_v_phys)

    # Old (previous time-step) physical gradients (only for old residual pieces)
    grad_u_old_phys = dot(grad(u_prev), Finv_old)
    eps_u_old = Sym(grad_u_old_phys)

    grad_v_old_phys = dot(grad(v), Finv_old)
    eps_v_old = Sym(grad_v_old_phys)

    # --- Shape derivatives wrt dd ---
    dF = grad(dd)
    dFinv = -dot(Finv, dot(dF, Finv))  # δ(F^{-1}) = -F^{-1} (δF) F^{-1}
    dJ = J * trace(dot(Finv, dF))        # δJ = J tr(F^{-1} δF)

    # Reusable "shape" variations of pulled-back grads/divs
    grad_uk_shape = dot(grad(uk), dFinv)   # δ(grad(u)*Finv) with u fixed
    grad_v_shape = dot(grad(v), dFinv)

    eps_uk_shape = Sym(grad_uk_shape)
    eps_v_shape = Sym(grad_v_shape)

    div_uk_shape = trace(grad_uk_shape)
    div_v_shape = trace(grad_v_shape)

    # --- RESIDUAL (R) ---
    # Fluid: mass (theta on J only, like CN)
    mass_res_f = (rho_f / dt_const) * (theta_const * J + one_minus_theta * J_old) * inner(uk - u_prev, v)

    # Fluid: viscosity (CN)
    sigma_uk = Constant(2.0) * rho_f * nu_f * eps_uk
    visc_res_f = (
        theta_const * J * inner(sigma_uk, eps_v)
        + one_minus_theta * J_old * inner(Constant(2.0) * rho_f * nu_f * eps_u_old, eps_v_old)
    )

    # Fluid: convection (CN), BUT mesh_vel uses current dk in BOTH parts
    conv_vel_new = uk - mesh_vel
    conv_vel_old = u_prev - mesh_vel
    adv_res_f = rho_f * (
        theta_const * J * dot(dot(grad_uk_phys, conv_vel_new), v)
        + one_minus_theta * J_old * dot(dot(grad_u_old_phys, conv_vel_old), v)
    )

    # Fluid: pressure/continuity (implicit Euler, no theta, no old term)
    pres_res_f = -J * (pk * div_v + q * div_uk) + stab_eps * pk * q

    residual_form = (mass_res_f + visc_res_f + adv_res_f + pres_res_f) * dx_f

    # If p DOFs exist in the solid region in your mixed space, this avoids singular rows
    residual_form += stab_eps * pk * q * dx_s

    # Solid: Neo-Hookean + time update
    P, cache = neo_hookean_pk1(Fk, c10=c10, kappa=kappa)
    P_old, _ = neo_hookean_pk1(F_old, c10=c10, kappa=kappa)

    solid_mass = (rho_s / dt_const) * inner(uk - u_prev, v)
    solid_stress = inner(theta_const * P + one_minus_theta * P_old, grad(v))
    solid_kin = inner(uk + u_prev - Constant(2.0) * (dk - d_prev) / dt_const, w)
    residual_form += (solid_mass + solid_stress + solid_kin) * dx_s

    # Mesh regularization in fluid
    residual_form += mesh_reg * inner(Sym(grad(dk)), Sym(grad(w))) * dx_f

    # --- JACOBIAN (K = dR/d(du,dp,dd)) ---
    # u-block
    grad_du_phys = dot(grad(du), Finv)
    eps_du = Sym(grad_du_phys)
    div_du = trace(grad_du_phys)

    mass_jac_u = (rho_f / dt_const) * (theta_const * J + one_minus_theta * J_old) * inner(du, v)
    const_visc_jac_u = Constant(2.0) * rho_f * nu_f * theta_const * J
    visc_jac_u = const_visc_jac_u * inner(eps_du, eps_v)

    conv_jac_u = rho_f * theta_const * J * (
        dot(dot(grad_du_phys, conv_vel_new), v)   # δ grad(u)
        + dot(dot(grad_uk_phys, du), v)            # δ conv velocity (u)
    )

    pres_jac_u = -J * q * div_du
    pres_jac_p = -J * dp * div_v + stab_eps * dp * q

    # d-block (shape derivatives)
    mass_jac_d = (rho_f / dt_const) * theta_const * dJ * dot(uk - u_prev, v)

    # inner(sigma_uk, eps_v) is equal to trace(dot(sigma_uk, eps_v.T)) but since it is symmetric
    term_b_dJ = trace(dot(sigma_uk, eps_v))
    visc_jac_d = theta_const * (
        dJ * term_b_dJ
        + J * inner(Constant(2.0) * rho_f * nu_f * eps_uk_shape, eps_v)
        + J * inner(sigma_uk, eps_v_shape)
    )

    # convection shape: new part (J, Finv in grad, mesh_vel) + old part (mesh_vel only)
    conv_jac_d = rho_f * (
        theta_const * (
            dJ * dot(dot(grad_uk_phys, conv_vel_new), v)
            + J * dot(dot(grad_uk_shape, conv_vel_new), v)
            + J * dot(dot(grad_uk_phys, -dd / dt_const), v)
        )
        + one_minus_theta * (
            # only mesh_vel depends on dk here
            J_old * dot(dot(grad_u_old_phys, -dd / dt_const), v)
        )
    )

    pres_jac_d = (
        -dJ * (pk * div_v + q * div_uk)
        -J * (pk * div_v_shape + q * div_uk_shape)
    )

    jacobian_form = (
        (mass_jac_u + visc_jac_u + conv_jac_u + pres_jac_u + pres_jac_p + mass_jac_d + visc_jac_d + conv_jac_d + pres_jac_d)
        * dx_f
    )

    # Same stabilization on solid for dp*q
    jacobian_form += stab_eps * dp * q * dx_s

    # Solid Jacobian
    deltaP = neo_hookean_delta_P(Fk, cache, grad(dd), c10=c10, kappa=kappa)
    solid_mass_jac = (rho_s / dt_const) * inner(du, v)
    solid_stress_jac = theta_const * inner(deltaP, grad(v))
    solid_kin_jac = inner(du, w) - Constant(2.0) * inner(dd, w) / dt_const
    jacobian_form += (solid_mass_jac + solid_stress_jac + solid_kin_jac) * dx_s

    # Mesh regularization Jacobian
    jacobian_form += mesh_reg * inner(Sym(grad(dd)), Sym(grad(w))) * dx_f

    return jacobian_form, residual_form



def compute_drag_lift(dh: DofHandler, mesh, u: VectorFunction, p: Function, mu: float):
    from pycutfem.ufl.expressions import FacetNormal

    n = FacetNormal()
    grad_u = grad(u)
    sigma = mu * (grad_u + grad_u.T) - p * Identity(2)
    traction = dot(sigma, n)
    drag = traction[0] * dS(defined_on=mesh.edge_bitset("cylinder"))
    lift = traction[1] * dS(defined_on=mesh.edge_bitset("cylinder"))
    eq_drag = Equation(drag, None)
    eq_lift = Equation(lift, None)
    comp = FormCompiler(dh, backend="python")
    drag_val = comp.assemble(eq_drag)[1].sum()
    lift_val = comp.assemble(eq_lift)[1].sum()
    return drag_val, lift_val


def tip_displacement(dh: DofHandler, d: VectorFunction):
    coords = dh.get_dof_coords("dx")
    tip = np.array([BEAM_CENTER[0] + BEAM_LENGTH, BEAM_CENTER[1]])
    diffs = coords - tip
    idx = int(np.argmin(np.einsum("ij,ij->i", diffs, diffs)))
    return float(d.nodal_values[idx]), float(d.nodal_values[idx + len(coords)])


def solve_initial_stokes(dh, vel_space, disp_space, p_space, vel_bcs, disp_bcs, p_bcs, fluid_bs, mu):
    """Compute a Stokes extension for the inflow profile as initial guess."""
    du = VectorTrialFunction(space=vel_space, dof_handler=dh)
    dd = VectorTrialFunction(space=disp_space, dof_handler=dh)
    dp = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    v = VectorTestFunction(space=vel_space, dof_handler=dh)
    w = VectorTestFunction(space=disp_space, dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)

    stab_eps = Constant(1e-8)
    kill = Constant(1e-8)
    dx_f = dx(defined_on=fluid_bs)
    dx_all = dx()
    a = (
        2 * mu * inner(symgrad(du), symgrad(v))
        - dp * div(v)
        + q * div(du)
        + stab_eps * dp * q
    ) * dx_f
    a += kill * (inner(du, v) + inner(dd, w) + dp * q) * dx_all
    r = kill * (inner(du, v) + inner(dd, w) + dp * q) * dx_all
    eq = Equation(a, r)
    bcs = vel_bcs + disp_bcs + p_bcs
    K, F = assemble_form(eq, dof_handler=dh, bcs=bcs, backend="jit")
    sol = np.linalg.solve(K.todense(), F)
    return sol


def main():
    ap = argparse.ArgumentParser(description="Conforming ALE FSI (Neo-Hookean solid, ALE fluid).")
    ap.add_argument("--dt", type=float, default=0.004)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--mesh-size", type=float, default=0.08)
    ap.add_argument("--poly-order", type=int, default=2)
    ap.add_argument("--backend", choices=("jit", "python"), default="jit")
    ap.add_argument("--mesh-file", type=Path, default=Path("examples/meshes/fsi_conforming.msh"))
    ap.add_argument("--rebuild-mesh", action="store_true")
    ap.add_argument("--view-gmsh", action="store_true", help="Preview mesh in gmsh when rebuilding.")
    ap.add_argument("--run-fd", action="store_true", help="Run finite-difference Jacobian check.", default=True)
    ap.add_argument("--no-fd", dest="run_fd", action="store_false", help="Skip FD Jacobian check.")
    ap.add_argument("--no-line-search", dest="line_search", action="store_false", help="Disable Newton line search.")
    ap.set_defaults(line_search=True)
    args = ap.parse_args()

    mesh_path = args.mesh_file
    if args.rebuild_mesh or not mesh_path.exists():
        print(f"Building conforming mesh at {mesh_path} (h={args.mesh_size}, P{args.poly_order})")
        build_mesh(mesh_path, h=args.mesh_size, order=args.poly_order, view=args.view_gmsh)
    mesh = mesh_from_gmsh(mesh_path)
    if not getattr(mesh, "boundary_tag_names", None):
        retag_boundaries(mesh)

    # Quick coverage diagnostic to spot missing cells/holes in the quad mesh
    coverage = coverage_diagnostics(mesh, n_samples=80)
    gaps_x, gaps_y = coverage["gaps_x"], coverage["gaps_y"]
    if gaps_x or gaps_y:
        def _preview(vals: list[float], max_items: int = 10) -> str:
            if not vals:
                return "[]"
            arr = np.array(vals, dtype=float)
            if len(arr) <= max_items:
                return np.array2string(arr, precision=4, separator=", ")
            head = np.array2string(arr[:max_items], precision=4, separator=", ")
            return f"{head} ... (total {len(arr)})"

        print(
            f"Coverage gaps detected: vertical samples={len(gaps_x)}, horizontal samples={len(gaps_y)}"
        )
        if gaps_x:
            print(f"  gaps_x (x-values): {_preview(gaps_x)}")
        if gaps_y:
            print(f"  gaps_y (y-values): {_preview(gaps_y)}")
    else:
        print("Coverage diagnostics: sampled vertical/horizontal lines fully covered (no gaps).")

    beam_ls = BeamCircularRootLevelSet(center=BEAM_CENTER, Lb=BEAM_LENGTH, Hb=BEAM_HEIGHT, circle_center=CENTER, radius=RADIUS, offset=1e-6)

    classify_conforming(mesh)
    fluid_bs = mesh.element_bitset("outside")
    solid_bs = mesh.element_bitset("inside")

    element = MixedElement(mesh, field_specs={"ux": args.poly_order, "uy": args.poly_order, "dx": args.poly_order, "dy": args.poly_order, "p": args.poly_order - 1})
    dh = DofHandler(element, method="cg")

    vel_space = FunctionSpace(name="vel", field_names=["ux", "uy"], dim=1)
    disp_space = FunctionSpace(name="disp", field_names=["dx", "dy"], dim=1)
    p_space = FunctionSpace(name="p", field_names=["p"], dim=0)

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
    for f in [uk, u_prev, dk, d_prev, pk, p_prev]:
        f.nodal_values.fill(0.0)

    dt_const = Constant(args.dt)
    theta_const = Constant(args.theta)
    rho_f = Constant(RHO_F)
    nu_f = Constant(NU_F)
    rho_s = Constant(RHO_S)
    c10 = Constant(0.5 * MU_S)
    kappa = Constant(LAMBDA_S + (2.0 / 3.0) * MU_S)
    mesh_reg = Constant(1e-6 * MU_S)
    stab_eps = Constant(1e-8)
    quad_order = 2 * args.poly_order + 4

    jac_form, res_form = build_forms(
        uk=uk,
        u_prev=u_prev,
        dk=dk,
        d_prev=d_prev,
        pk=pk,
        p_prev=p_prev,
        du=du,
        dd=dd,
        dp=dp,
        v=v,
        w=w,
        q=q,
        dt_const=dt_const,
        theta_const=theta_const,
        rho_f=rho_f,
        nu_f=nu_f,
        rho_s=rho_s,
        c10=c10,
        kappa=kappa,
        mesh_reg=mesh_reg,
        stab_eps=stab_eps,
        fluid_bs=fluid_bs,
        solid_bs=solid_bs,
        quad_order=quad_order,
    )

    def inlet_u(x, y, t=0.0):
        return 4.0 * 1.5 * y * (H - y) / (H * H)

    zero = lambda x, y, t=0.0: 0.0
    vel_bcs = [
        BoundaryCondition("ux", "dirichlet", "inlet", inlet_u),
        BoundaryCondition("uy", "dirichlet", "inlet", zero),
        BoundaryCondition("ux", "dirichlet", "walls", zero),
        BoundaryCondition("uy", "dirichlet", "walls", zero),
        BoundaryCondition("ux", "dirichlet", "cylinder", zero),
        BoundaryCondition("uy", "dirichlet", "cylinder", zero),
        BoundaryCondition("ux", "dirichlet", "beam_root", zero),
        BoundaryCondition("uy", "dirichlet", "beam_root", zero),
    ]
    disp_bcs = [
        BoundaryCondition("dx", "dirichlet", "inlet", zero),
        BoundaryCondition("dy", "dirichlet", "inlet", zero),
        BoundaryCondition("dx", "dirichlet", "outlet", zero),
        BoundaryCondition("dy", "dirichlet", "outlet", zero),
        BoundaryCondition("dx", "dirichlet", "walls", zero),
        BoundaryCondition("dy", "dirichlet", "walls", zero),
        BoundaryCondition("dx", "dirichlet", "cylinder", zero),
        BoundaryCondition("dy", "dirichlet", "cylinder", zero),
        BoundaryCondition("dx", "dirichlet", "beam_root", zero),
        BoundaryCondition("dy", "dirichlet", "beam_root", zero),
    ]
    p_bcs = [BoundaryCondition("p", "dirichlet", "outlet", zero)]
    bcs = vel_bcs + disp_bcs + p_bcs
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, zero) for b in bcs]

    stokes_sol = solve_initial_stokes(dh, vel_space, disp_space, p_space, vel_bcs, disp_bcs, p_bcs, fluid_bs, mu=MU_F)
    def _set(field, names):
        for nm in names:
            sl = dh.get_field_slice(nm)
            field.set_nodal_values(sl, stokes_sol[sl])
    _set(uk, ["ux", "uy"])
    _set(pk, ["p"])
    u_prev.nodal_values[:] = uk.nodal_values[:]
    p_prev.nodal_values[:] = pk.nodal_values[:]

    dh.apply_bcs(bcs, uk, u_prev, dk, d_prev, pk, p_prev)
    print(f"Total DOFs: {dh.total_dofs}")

    if args.run_fd:
        probe = pick_probe_dofs(dh, bcs_homog, per_field=2)
        fd_fields = {"ux": uk, "uy": uk, "dx": dk, "dy": dk, "p": pk}
        finite_difference_check(jac_form, res_form, dh, bcs_homog, fd_fields, probe, eps=1e-7, backend=args.backend)

    solver = NewtonSolver(
        residual_form=res_form,
        jacobian_form=jac_form,
        dof_handler=dh,
        mixed_element=element,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=1e-8, max_newton_iter=20, line_search=args.line_search),
        backend=args.backend,
    )
    time_params = TimeStepperParameters(dt=args.dt, max_steps=1, stop_on_steady=False)
    aux_funcs = {
        "u_ux": uk.components[0],
        "u_uy": uk.components[1],
        "u_prev_ux": u_prev.components[0],
        "u_prev_uy": u_prev.components[1],
        "d_dx": dk.components[0],
        "d_dy": dk.components[1],
        "d_prev_dx": d_prev.components[0],
        "d_prev_dy": d_prev.components[1],
    }
    delta, steps, elapsed = solver.solve_time_interval(
        functions=[uk, dk, pk],
        prev_functions=[u_prev, d_prev, p_prev],
        aux_functions=aux_funcs,
        time_params=time_params,
    )
    vel_norm = np.linalg.norm(uk.nodal_values, ord=np.inf)
    disp_norm = np.linalg.norm(dk.nodal_values, ord=np.inf)
    drag, lift = compute_drag_lift(dh, mesh, uk, pk, MU_F)
    tip_dx, tip_dy = tip_displacement(dh, dk)
    print(f"Solved {steps} step(s) in {elapsed:.2f}s: ||u||_inf={vel_norm:.3e}, ||d||_inf={disp_norm:.3e}")
    print(f"Drag={drag:.3e}, Lift={lift:.3e}, Tip disp=({tip_dx:.3e}, {tip_dy:.3e})")


if __name__ == "__main__":
    main()
