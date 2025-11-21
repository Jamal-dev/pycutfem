#!/usr/bin/env python
"""
Volume-only variant of the Turek–Schafer 2D-2 benchmark that uses a Gmsh mesh
with an explicit cylinder hole instead of CutFEM.

The script generates (or reuses) a triangular mesh of the channel with a
rectangular domain (2.2 x 0.41) and a cylindrical obstacle.  The mesh is
imported with :func:`pycutfem.utils.gmsh_loader.mesh_from_gmsh`, after which a
standard Taylor–Hood discretization with a one-step theta time scheme is
assembled.  Drag and lift coefficients are computed by integrating the fluid
traction over the cylinder boundary using only volume forms (no ghost terms).
"""
from __future__ import annotations

import argparse
import math
import os
import time
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory

try:  # Gmsh is optional when using the structured O-grid backend
    import gmsh
except Exception:  # pragma: no cover - handled at runtime
    gmsh = None

import numba
import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem import transform
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import (
    NewtonParameters,
    NewtonSolver,
    TimeStepperParameters,
)
from pycutfem.ufl.expressions import (
    Constant,
    FacetNormal,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    div,
    dot,
    grad,
    inner,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dS, dx
from pycutfem.utils.gmsh_loader import mesh_from_gmsh
from pycutfem.utils.ogrid_meshgen import circular_hole_ogrid
from pycutfem.utils.bitset import BitSet
import matplotlib.pyplot as plt
from pycutfem.io.visualization import visualize_boundary_dofs


# --------------------------------------------------------------------------- #
#                        Problem / geometry parameters
# --------------------------------------------------------------------------- #
H = 0.41   # channel height
L = 2.2    # channel length
D = 0.1    # cylinder diameter
CENTER = (0.2, 0.2)
RADIUS = D / 2.0
RHO = 1.0
MU = 1.0e-3
U_MEAN = 1.5
FE_ORDER = 2  # Taylor–Hood (Q2/Q1) equivalent on triangles


def _configure_numba():
    """Set numba to use all available threads."""
    try:
        numba.set_num_threads(os.cpu_count())
        print(f"Numba threads: {numba.get_num_threads()}")
    except (ImportError, AttributeError):
        print("Numba not configured; running in pure Python mode.")


# --------------------------------------------------------------------------- #
#                              Mesh generation
# --------------------------------------------------------------------------- #
def build_turek_channel_mesh(path: Path, mesh_size: float, cell_type: str = "tri", view_mesh: bool = False, mesh_order: int | None = None) -> None:
    """
    Build the Turek benchmark mesh with an O-grid type block structure.
    The mesh is written to ``path``.
    """
    if gmsh is None:
        raise RuntimeError("Gmsh is not available; cannot build a gmsh-based mesh.")
    if cell_type not in {"tri", "quad"}:
        raise ValueError("cell_type must be 'tri' or 'quad'")

    gmsh.initialize()
    try:
        gmsh.model.add("turek_channel_volume")
        occ = gmsh.model.occ

        # --- helper utilities ------------------------------------------------
        def _point_key(x, y):
            return (round(float(x), 12), round(float(y), 12))

        point_lookup: dict[tuple[float, float], int] = {}
        point_coords: dict[int, tuple[float, float]] = {}

        def add_point(x: float, y: float) -> int:
            key = _point_key(x, y)
            if key not in point_lookup:
                tag = occ.addPoint(x, y, 0.0)
                point_lookup[key] = tag
                point_coords[tag] = (x, y)
            return point_lookup[key]

        line_lookup: dict[tuple[int, int], int] = {}
        line_lengths: dict[int, float] = {}
        line_target_nodes: dict[int, int] = {}

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

        boundary_edges: dict[str, list[int]] = {
            "inlet": [],
            "outlet": [],
            "walls": [],
            "cylinder": [],
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

        # --- block layout ----------------------------------------------------
        buffer = max(2.5 * mesh_size, 0.01)
        inner_half_x = min(CENTER[0] - buffer, L - CENTER[0] - buffer, 0.35)
        inner_half_y = min(CENTER[1] - buffer, H - CENTER[1] - buffer, 0.35)
        if inner_half_x <= RADIUS or inner_half_y <= RADIUS:
            raise RuntimeError("Inner square collapsed; adjust mesh parameters.")
        inner_x0 = CENTER[0] - inner_half_x
        inner_x1 = CENTER[0] + inner_half_x
        inner_y0 = CENTER[1] - inner_half_y
        inner_y1 = CENTER[1] + inner_half_y

        x_coords = [0.0, inner_x0, CENTER[0], inner_x1, L]
        y_coords = [0.0, inner_y0, CENTER[1], inner_y1, H]
        nx = len(x_coords)
        ny = len(y_coords)

        x_counts = [
            max(3, int(round((x_coords[i + 1] - x_coords[i]) / mesh_size)) + 1)
            for i in range(nx - 1)
        ]
        y_counts = [
            max(3, int(round((y_coords[j + 1] - y_coords[j]) / mesh_size)) + 1)
            for j in range(ny - 1)
        ]

        grid_points: dict[tuple[int, int], int] = {}
        for ix, x in enumerate(x_coords):
            for iy, y in enumerate(y_coords):
                grid_points[(ix, iy)] = add_point(x, y)

        horizontal_lines: dict[tuple[int, int], int] = {}
        for iy in range(ny):
            for ix in range(nx - 1):
                start = grid_points[(ix, iy)]
                end = grid_points[(ix + 1, iy)]
                boundary = None
                if iy == 0 or iy == ny - 1:
                    boundary = "walls"
                line_tag = add_line(start, end, boundary=boundary)
                horizontal_lines[(iy, ix)] = line_tag
                line_target_nodes[line_tag] = x_counts[ix]

        vertical_lines: dict[tuple[int, int], int] = {}
        for ix in range(nx):
            for iy in range(ny - 1):
                start = grid_points[(ix, iy)]
                end = grid_points[(ix, iy + 1)]
                boundary = None
                if ix == 0:
                    boundary = "inlet"
                elif ix == nx - 1:
                    boundary = "outlet"
                line_tag = add_line(start, end, boundary=boundary)
                vertical_lines[(ix, iy)] = line_tag
                line_target_nodes[line_tag] = y_counts[iy]

        fluid_surfaces: list[int] = []

        def add_rect_surface(ix: int, iy: int) -> None:
            loop = [
                horizontal_lines[(iy, ix)],
                vertical_lines[(ix + 1, iy)],
                -horizontal_lines[(iy + 1, ix)],
                -vertical_lines[(ix, iy)],
            ]
            cloop = occ.addCurveLoop(loop)
            fluid_surfaces.append(occ.addPlaneSurface([cloop]))

        for ix in range(nx - 1):
            for iy in range(ny - 1):
                if 1 <= ix <= 2 and 1 <= iy <= 2:
                    continue  # hole handled separately
                add_rect_surface(ix, iy)

        # --- O-grid around the cylinder --------------------------------------
        center_pt = grid_points[(2, 2)]  # (CENTER)
        circle_angles = [i * math.pi / 4.0 for i in range(8)]
        circle_points: list[int] = []
        angle_lookup: dict[int, float] = {}
        for angle in circle_angles:
            x = CENTER[0] + RADIUS * math.cos(angle)
            y = CENTER[1] + RADIUS * math.sin(angle)
            tag = add_point(x, y)
            circle_points.append(tag)
            angle_lookup[tag] = angle

        square_point_indices = [
            (3, 2),  # right mid
            (3, 3),  # top right corner
            (2, 3),  # top mid
            (1, 3),  # top left corner
            (1, 2),  # left mid
            (1, 1),  # bottom left corner
            (2, 1),  # bottom mid
            (3, 1),  # bottom right corner
        ]
        square_points = [grid_points[idx] for idx in square_point_indices]

        radial_span = max(
            math.hypot(inner_half_x, inner_half_y) - RADIUS,
            inner_half_x - RADIUS,
            inner_half_y - RADIUS,
        )
        radial_node_target = max(6, int(round(radial_span / mesh_size)) + 1)

        radial_lines: list[int] = []
        for c_tag, s_tag in zip(circle_points, square_points):
            tag = add_line(c_tag, s_tag)
            line_target_nodes[tag] = radial_node_target
            radial_lines.append(tag)

        square_segments: list[int] = []
        for i in range(len(square_points)):
            a = square_points[i]
            b = square_points[(i + 1) % len(square_points)]
            square_segments.append(oriented_line(a, b))

        arc_lines: list[int] = []
        for i in range(len(circle_points)):
            start = circle_points[i]
            end = circle_points[(i + 1) % len(circle_points)]
            theta_start = angle_lookup[start]
            theta_end = angle_lookup[end]
            delta = theta_end - theta_start
            if delta <= 0.0:
                delta += 2.0 * math.pi
            tag = occ.addCircleArc(start, center_pt, end)
            register_line(tag, start, end, RADIUS * delta)
            boundary_edges["cylinder"].append(tag)
            matching_segment = abs(square_segments[i])
            line_target_nodes[tag] = line_target_nodes[matching_segment]
            arc_lines.append(tag)

        for i in range(len(circle_points)):
            # loop = [
            #     arc_lines[i],
            #     radial_lines[(i + 1) % len(circle_points)],
            #     -square_segments[i],
            #     -radial_lines[i],
            # ]
            next_i = (i + 1) % len(circle_points)
            
            loop = [
                radial_lines[i],            # Center -> Square (Outward)
                square_segments[i],         # Square -> Square Next (CCW along boundary)
                -radial_lines[next_i],      # Square Next -> Center Next (Inward)
                -arc_lines[i],              # Center Next -> Center (CW along hole)
            ]
            
            # NOTE: Traversing the hole boundary CW keeps the domain on the Left.
            # Traversing the outer square CCW keeps the domain on the Left.
            # This is the correct orientation for a valid surface with a hole.
            cloop = occ.addCurveLoop(loop)
            fluid_surfaces.append(occ.addPlaneSurface([cloop]))

        gmsh.model.occ.removeAllDuplicates()
        occ.synchronize()

        gmsh.model.addPhysicalGroup(2, fluid_surfaces, tag=1)
        gmsh.model.setPhysicalName(2, 1, "fluid")

        boundary_tag_hints = {"inlet": 11, "outlet": 12, "walls": 13, "cylinder": 14}
        for name, tag_hint in boundary_tag_hints.items():
            edges = sorted(set(boundary_edges[name]))
            if not edges:
                continue
            tag = gmsh.model.addPhysicalGroup(1, edges, tag=tag_hint)
            gmsh.model.setPhysicalName(1, tag, name)

        # --- transfinite meshing controls ------------------------------------
        def nodes_for_length(length: float, *, min_nodes: int = 3) -> int:
            segments = max(2, int(round(length / mesh_size)))
            return max(min_nodes, segments + 1)

        radial_set = set(radial_lines)
        arc_set = set(arc_lines)

        for tag, length in line_lengths.items():
            target_nodes = line_target_nodes.get(tag)
            progression = 1.0
            if tag in radial_set:
                progression = 1.2
            if target_nodes is None:
                target_nodes = nodes_for_length(length)
            gmsh.model.mesh.setTransfiniteCurve(
                tag,
                target_nodes,
                "Progression",
                progression,
            )

        for surf in fluid_surfaces:
            gmsh.model.mesh.setTransfiniteSurface(surf)
            if cell_type == "quad":
                gmsh.model.mesh.setRecombine(2, surf)

        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 1 if cell_type == "quad" else 0)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.model.mesh.generate(2)
        # Keep the geometric order consistent with the FE order unless explicitly overridden.
        gmsh.model.mesh.setOrder(int(mesh_order) if mesh_order is not None else FE_ORDER)

        path.parent.mkdir(parents=True, exist_ok=True)
        if view_mesh:
            try:
                gmsh.fltk.initialize()
                gmsh.fltk.run()
            except Exception:
                print("Gmsh GUI not available; skipping mesh preview.")
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def prepare_mesh(mesh_file: Path | None, mesh_size: float, rebuild: bool, cell_type: str, view_mesh: bool) -> tuple:
    """
    Generate (if needed) and load the Gmsh mesh.
    Returns the in-memory :class:`Mesh` and the path to the mesh file (if kept).
    """
    if gmsh is None:
        raise RuntimeError("Gmsh backend requested but the gmsh Python module is not available.")
    if mesh_file is not None:
        mesh_file = mesh_file.expanduser().resolve()
        if rebuild or not mesh_file.exists():
            print(f"Generating Gmsh mesh at {mesh_file} (h={mesh_size}, cell_type={cell_type})")
            build_turek_channel_mesh(mesh_file, mesh_size, cell_type, view_mesh=view_mesh)
        else:
            print(f"Reusing existing mesh at {mesh_file}")
        if view_mesh and mesh_file.exists():
            try:
                gmsh.initialize()
                gmsh.open(str(mesh_file))
                gmsh.fltk.run()
            except Exception:
                print("Gmsh GUI not available; skipping mesh preview.")
            finally:
                gmsh.finalize()
        return mesh_from_gmsh(mesh_file), mesh_file

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "turek_channel.msh"
        print(f"Generating temporary Gmsh mesh (h={mesh_size}, cell_type={cell_type})")
        build_turek_channel_mesh(tmp_path, mesh_size, cell_type, view_mesh=view_mesh)
        mesh = mesh_from_gmsh(tmp_path)
    # tmpdir is cleaned up here; mesh lives on in memory
    return mesh, None


def _count_segments(width: float, mesh_size: float, *, min_cells: int = 1) -> int:
    if width <= 1.0e-12:
        return 0
    return max(min_cells, int(math.ceil(width / mesh_size)))


def build_structured_channel_mesh(mesh_size: float, poly_order: int) -> Mesh:
    """
    Build a structured quadrilateral mesh using the internal O-grid generator.
    """
    margin = max(2.5 * mesh_size, 0.015)
    half_x_cap = min(CENTER[0], L - CENTER[0]) - margin
    half_y_cap = min(CENTER[1], H - CENTER[1]) - margin
    if half_x_cap <= RADIUS or half_y_cap <= RADIUS:
        raise RuntimeError(
            "Structured O-grid collapsed: decrease mesh size or adjust parameters."
        )
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
    _tag_structured_mesh_boundaries(mesh, mesh_size)
    return mesh


def _tag_structured_mesh_boundaries(mesh: Mesh, mesh_size: float) -> None:
    tol = 1.0e-9
    circle_tol = max(0.25 * mesh_size, 1.0e-4)
    rect_locators = OrderedDict(
        [
            ("inlet", lambda x, y: abs(x - 0.0) <= tol),
            ("outlet", lambda x, y: abs(x - L) <= tol),
            ("walls", lambda x, y: abs(y - 0.0) <= tol or abs(y - H) <= tol),
        ]
    )
    mesh.tag_boundary_edges(rect_locators)

    circle_locator = lambda x, y: abs(math.hypot(x - CENTER[0], y - CENTER[1]) - RADIUS) <= circle_tol
    circle_corner_nodes = {
        node.id
        for node in mesh.nodes_list
        if node.tag and "boundary_circle" in node.tag.split(",")
    }
    cyl_mask = np.zeros(len(mesh.edges_list), dtype=bool)
    for edge in mesh.edges_list:
        if edge.right is not None:
            continue
        node_ids = set(edge.nodes)
        if node_ids and node_ids.issubset(circle_corner_nodes):
            edge.tag = "cylinder"
            cyl_mask[edge.gid] = True
    if hasattr(mesh, "_edge_bitsets"):
        mesh._edge_bitsets["cylinder"] = BitSet(cyl_mask)
    # Cache locators including the circular boundary for later use in BC application
    loc_map = getattr(mesh, "_boundary_locators", {})
    loc_map = dict(loc_map, **rect_locators)
    loc_map["cylinder"] = circle_locator
    mesh._boundary_locators = loc_map


def check_positive_jacobians(mesh: Mesh, sample_density: int | None = None):
    """
    Verify that det(J) > 0 for all elements at a grid of reference points.
    Returns (ok_flag, min_det, failure_info).
    """
    if mesh.element_type == "quad":
        n = sample_density or max(3, mesh.poly_order + 2)
        coords = np.linspace(-1.0, 1.0, n)
        sample_pts = [(float(xi), float(eta)) for xi in coords for eta in coords]
    else:
        n = sample_density or max(3, mesh.poly_order + 2)
        sample_pts = []
        for i in range(1, n):
            for j in range(1, n - i):
                xi = i / n
                eta = j / n
                sample_pts.append((float(xi), float(eta)))
        if not sample_pts:
            sample_pts = [(1.0 / 3.0, 1.0 / 3.0)]

    min_det = float("inf")
    for elem in mesh.elements_list:
        for xi, eta in sample_pts:
            detJ = float(transform.det_jacobian(mesh, elem.id, (xi, eta)))
            min_det = min(min_det, detJ)
            if detJ <= 0.0:
                return False, detJ, (elem.id, xi, eta)
    return True, min_det, None


def generate_mesh(args):
    """
    Create a mesh using either gmsh (default) or the structured O-grid backend.
    """
    geometric_order = 2
    if args.mesh_backend == "structured":
        if args.mesh_type != "quad":
            raise ValueError("The structured mesh backend only supports quadrilateral elements.")
        mesh = build_structured_channel_mesh(args.mesh_size, poly_order=geometric_order)
        return mesh, None
    return prepare_mesh(args.mesh_file, args.mesh_size, args.rebuild_mesh, args.mesh_type, args.view_gmsh)


# --------------------------------------------------------------------------- #
#                        Helper evaluation utilities
# --------------------------------------------------------------------------- #
def locate_element(mesh, point: np.ndarray):
    """Locate the element containing ``point`` (returns (eid, xi, eta))."""
    xy = np.asarray(point, dtype=float)
    for elem in mesh.elements_list:
        node_ids = elem.nodes
        coords = mesh.nodes_x_y_pos[list(node_ids)]
        if not (
            coords[:, 0].min() - 1e-12 <= xy[0] <= coords[:, 0].max() + 1e-12
            and coords[:, 1].min() - 1e-12 <= xy[1] <= coords[:, 1].max() + 1e-12
        ):
            continue
        try:
            xi, eta = transform.inverse_mapping(mesh, elem.id, xy)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if -1.0001 <= xi <= 1.0001 and -1.0001 <= eta <= 1.0001:
            return elem.id, xi, eta
    return None, None, None


def evaluate_scalar_field(dof_handler, mesh, field_name: str, func: Function, point):
    """Evaluate a scalar Function at a physical point."""
    eid, xi, eta = locate_element(mesh, point)
    if eid is None:
        return math.nan
    me = dof_handler.mixed_element
    phi = me.basis(field_name, xi, eta)[me.slice(field_name)]
    gdofs = np.asarray(dof_handler.element_maps[field_name][eid], dtype=int)
    vals = func.get_nodal_values(gdofs)
    return float(phi @ vals)


def evaluate_vector_field(dof_handler, mesh, v_func: VectorFunction, point):
    """Evaluate the vector function at a point by evaluating each component."""
    return np.array(
        [evaluate_scalar_field(dof_handler, mesh, field.field_name, field, point) for field in v_func],
        dtype=float,
    )


def nearest_pressure_dof_value(dof_handler: DofHandler, p_func: Function, point: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Evaluate the pressure function at the DOF located closest to ``point``.
    Works robustly even if the requested location is outside the mesh (e.g. inside the cylinder).
    """
    coords = dof_handler.get_dof_coords("p")
    diffs = coords - point
    idx = int(np.argmin(np.einsum("ij,ij->i", diffs, diffs)))
    return float(p_func.nodal_values[idx]), coords[idx]


# --------------------------------------------------------------------------- #
#                            Weak form components
# --------------------------------------------------------------------------- #
def epsilon(u):
    """Symmetric gradient."""
    return 0.5 * (grad(u) + grad(u).T)


def build_volume_forms(
    u_trial,
    v_test,
    p_trial,
    q_test,
    u_k,
    u_n,
    p_k,
    p_n,
    rho_const,
    mu_const,
    dt_const,
    theta_const,
    dx_measure,
):
    """Return (jacobian_form, residual_form) for the standard theta-scheme."""
    a_vol = (
        rho_const * dot(u_trial, v_test) / dt_const
        + theta_const * rho_const * dot(dot(grad(u_k), u_trial), v_test)
        + theta_const * rho_const * dot(dot(grad(u_trial), u_k), v_test)
        + 2.0 * theta_const * mu_const * inner(epsilon(u_trial), epsilon(v_test))
        - p_trial * div(v_test)
        + q_test * div(u_trial)
    ) * dx_measure

    r_vol = (
        rho_const * dot(u_k - u_n, v_test) / dt_const
        + theta_const * rho_const * dot(dot(grad(u_k), u_k), v_test)
        + (1.0 - theta_const) * rho_const * dot(dot(grad(u_n), u_n), v_test)
        + 2.0 * theta_const * mu_const * inner(epsilon(u_k), epsilon(v_test))
        + 2.0 * (1.0 - theta_const) * mu_const * inner(epsilon(u_n), epsilon(v_test))
        - p_k * div(v_test)
        + q_test * div(u_k)
    ) * dx_measure

    return a_vol, r_vol


def traction_dot_direction(u_vec, p_scal, direction, mu_const):
    """Return ((σ(u,p)·n)·direction) on boundary facets."""
    n = FacetNormal()
    grad_u = grad(u_vec)
    sigma_n = mu_const * (dot(grad_u, n) + dot(grad_u.T, n)) - p_scal * n
    return dot(sigma_n, direction)


# --------------------------------------------------------------------------- #
#                            Main driver routine
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Volume-only Turek benchmark using a Gmsh mesh."
    )
    parser.add_argument("--backend", choices=("python", "jit"), default="jit", help="Assembly backend to use.")
    parser.add_argument("--mesh-backend", choices=("gmsh", "structured"), default="gmsh",
                        help="Mesh generator: 'gmsh' (default) or the built-in structured O-grid.")
    parser.add_argument("--mesh-size", type=float, default=0.02, help="Target edge size for the mesh generator.")
    parser.add_argument("--mesh-type", choices=("tri", "quad"), default="quad",
                        help="Element type generated by gmsh/structured backend (structured supports only quad).")
    parser.add_argument("--mesh-file", type=Path, help="Optional path to reuse/store the .msh file (gmsh backend).")
    parser.add_argument("--rebuild-mesh", action="store_true", help="Force rebuilding the gmsh mesh.")
    parser.add_argument("--view-gmsh", action="store_true", help="Preview the generated gmsh mesh.")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step size.")
    parser.add_argument("--theta", type=float, default=0.5, help="Theta parameter for the time-stepping scheme.")
    parser.add_argument("--max-steps", type=int, default=36, help="Maximum number of time steps.")
    parser.add_argument(
        "--save-vtk",
        dest="save_vtk",
        action="store_true",
        help="Write VTU files for each step (default: on).",
    )
    parser.add_argument(
        "--no-save-vtk",
        dest="save_vtk",
        action="store_false",
        help="Disable VTU output.",
    )
    parser.set_defaults(save_vtk=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("turek_volume_results"),
        help="Directory for VTU files / diagnostic plots.",
    )
    parser.add_argument("--stop-on-steady", action="store_true", help="Stop when reaching steady state.")
    parser.add_argument(
        "--check-det-only",
        action="store_true",
        help="Only build the mesh, report det(J) statistics, and exit.",
    )
    args = parser.parse_args()

    _configure_numba()

    mesh, persistent_mesh_path = generate_mesh(args)
    print(mesh)
    need_det_check = args.mesh_backend == "structured" or args.check_det_only
    if need_det_check:
        det_ok, det_min, failure = check_positive_jacobians(mesh)
        if not det_ok:
            eid, xi, eta = failure
            raise RuntimeError(
                f"Jacobian determinant non-positive at element {eid} for (xi, eta)=({xi:.3f}, {eta:.3f})"
            )
        print(f"Minimum det(J) across sampled points: {det_min:.6e}")
        if args.check_det_only:
            if persistent_mesh_path:
                print(f"Mesh stored at: {persistent_mesh_path}")
            return

    cylinder_edges = mesh.edge_bitset("cylinder")
    if cylinder_edges.cardinality() == 0:
        raise RuntimeError("Cylinder boundary tag not found in the imported mesh.")
    for name in ("inlet", "outlet", "walls", "cylinder"):
        print(f"Edges tagged '{name}': {mesh.edge_bitset(name).cardinality()}")

    # Mixed Taylor–Hood space (vector velocity + scalar pressure)
    mixed_element = MixedElement(
        mesh,
        field_specs={"ux": FE_ORDER, "uy": FE_ORDER, "p": FE_ORDER - 1},
    )
    dof_handler = DofHandler(mixed_element, method="cg")

    # Boundary conditions (no-slip on walls + cylinder, parabolic inflow)
    def parabolic_inflow(x, y):
        return 4.0 * U_MEAN * y * (H - y) / (H**2)

    bcs: list[BoundaryCondition] = [
        BoundaryCondition("ux", "dirichlet", "inlet", parabolic_inflow),
        BoundaryCondition("uy", "dirichlet", "inlet", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "walls", lambda x, y: 0.0),
        BoundaryCondition("uy", "dirichlet", "walls", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "cylinder", lambda x, y: 0.0),
        BoundaryCondition("uy", "dirichlet", "cylinder", lambda x, y: 0.0),
    ]

    bcs_homog = [
        BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs
    ]

    print(f"Dirichlet data: {dof_handler.dirichlet_stats(bcs)}")

    # --- Function spaces / functions ---
    velocity_space = FunctionSpace(name="velocity", field_names=["ux", "uy"], dim=1, side="+")
    pressure_space = FunctionSpace(name="pressure", field_names=["p"], dim=0, side="+")

    du = VectorTrialFunction(space=velocity_space, dof_handler=dof_handler, side="+")
    v = VectorTestFunction(space=velocity_space, dof_handler=dof_handler, side="+")
    dp = TrialFunction(name="trial_pressure", field_name="p", dof_handler=dof_handler, side="+")
    q = TestFunction(name="test_pressure", field_name="p", dof_handler=dof_handler, side="+")

    u_k = VectorFunction(name="u_k", field_names=["ux", "uy"], dof_handler=dof_handler, side="+")
    u_n = VectorFunction(name="u_n", field_names=["ux", "uy"], dof_handler=dof_handler, side="+")
    p_k = Function(name="p_k", field_name="p", dof_handler=dof_handler, side="+")
    p_n = Function(name="p_n", field_name="p", dof_handler=dof_handler, side="+")

    # Initialize
    for func in (u_k, u_n):
        func.nodal_values.fill(0.0)
    for func in (p_k, p_n):
        func.nodal_values.fill(0.0)
    dof_handler.apply_bcs(bcs, u_n, p_n)
    dof_handler.apply_bcs(bcs, u_k, p_k)

    rho_const = Constant(RHO)
    mu_const = Constant(MU)
    dt_const = Constant(args.dt)
    theta_const = Constant(args.theta)

    volume_quadrature = 2 * FE_ORDER + 2
    dx_vol = dx(metadata={"q": volume_quadrature})

    jacobian_form, residual_form = build_volume_forms(
        du,
        v,
        dp,
        q,
        u_k,
        u_n,
        p_k,
        p_n,
        rho_const,
        mu_const,
        dt_const,
        theta_const,
        dx_vol,
    )

    # --- Force diagnostics on the cylinder boundary ---
    boundary_quadrature = max(8, FE_ORDER * 3)
    d_gamma_cyl = dS(defined_on=cylinder_edges, metadata={"q": boundary_quadrature})

    e_x = Constant(np.array([1.0, 0.0]), dim=1)
    e_y = Constant(np.array([0.0, 1.0]), dim=1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_point = np.array([0.15, 0.2])
    histories: dict[str, list[float]] = {"time": [], "cd": [], "cl": [], "dp": []}

    probe_A = np.array([CENTER[0] - RADIUS - 0.01, CENTER[1]])
    probe_B = np.array([CENTER[0] + RADIUS + 0.01, CENTER[1]])

    visualize_boundary_dofs(
        mesh,
        tags=["cylinder", "inlet", "outlet", "walls"],
        dof_handler=dof_handler,
        fields=["ux", "uy"],
        annotate_nodes=True,
        annotate_dofs=True,
        title="Cylinder Dirichlet nodes",
    )
    def save_solution(funcs):
        """Callback executed after every converged time step."""
        velocity = funcs[0]  # VectorFunction
        pressure = funcs[1]  # Function

        # Optionally write VTK outputs
        step_id = len(histories["time"])
        if args.save_vtk:
            filename = output_dir / f"solution_{step_id:04d}.vtu"
            export_vtk(
                filename=str(filename),
                mesh=mesh,
                dof_handler=dof_handler,
                functions={"velocity": velocity, "pressure": pressure},
            )

        traction_drag = traction_dot_direction(velocity, pressure, e_x, mu_const) * d_gamma_cyl
        traction_lift = traction_dot_direction(velocity, pressure, e_y, mu_const) * d_gamma_cyl

        drag = assemble_form(
            Equation(None, traction_drag),
            dof_handler=dof_handler,
            assembler_hooks={traction_drag.integrand: {"name": "drag"}},
            backend="python",
        )["drag"]
        lift = assemble_form(
            Equation(None, traction_lift),
            dof_handler=dof_handler,
            assembler_hooks={traction_lift.integrand: {"name": "lift"}},
            backend="python",
        )["lift"]

        coeff = 2.0 / (RHO * U_MEAN**2 * D)
        c_d = coeff * drag
        c_l = coeff * lift

        # Pressure drop measurement (points upstream/downstream of cylinder)
        pA, _ = nearest_pressure_dof_value(dof_handler, pressure, probe_A)
        pB, _ = nearest_pressure_dof_value(dof_handler, pressure, probe_B)
        dp = pA - pB

        u_monitor = evaluate_vector_field(dof_handler, mesh, velocity, monitor_point)

        histories["time"].append(step_id * args.dt)
        histories["cd"].append(c_d)
        histories["cl"].append(c_l)
        histories["dp"].append(dp)

        print(
            f"[step {step_id:04d}] "
            f"Cd={c_d:.4f}  Cl={c_l:.4f}  Δp={dp:.4f}  "
            f"u(0.15,0.20)=({u_monitor[0]:.4f}, {u_monitor[1]:.4f})"
        )

    # Solver setup
    time_params = TimeStepperParameters(
        dt=args.dt,
        max_steps=args.max_steps,
        stop_on_steady=args.stop_on_steady,
        steady_tol=1e-6,
        theta=args.theta,
    )

    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dof_handler,
        mixed_element=mixed_element,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=1e-6, line_search=True),
        postproc_timeloop_cb=save_solution,
        backend=args.backend,
    )

    functions = [u_k, p_k]
    prev_functions = [u_n, p_n]

    t0 = time.time()
    try:
        solver.solve_time_interval(
            functions=functions,
            prev_functions=prev_functions,
            time_params=time_params,
        )
    except Exception as exc:  # pragma: no cover - diagnostic output
        print(f"Solver failed: {exc}")
        raise
    finally:
        print(f"Total runtime: {time.time() - t0:.1f} s")
        if persistent_mesh_path:
            print(f"Mesh stored at: {persistent_mesh_path}")
        if histories["time"]:
            print(
                f"Final Cd={histories['cd'][-1]:.4f}, "
                f"Cl={histories['cl'][-1]:.4f}, "
                f"Δp={histories['dp'][-1]:.4f}"
            )
            try:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
                axes[0].plot(histories["time"], histories["cd"], color="tab:blue")
                axes[0].set_ylabel("Cd")
                axes[0].set_title("Drag coefficient")
                axes[0].grid(True, linestyle=":", linewidth=0.5)

                axes[1].plot(histories["time"], histories["cl"], color="tab:green")
                axes[1].set_ylabel("Cl")
                axes[1].set_title("Lift coefficient")
                axes[1].grid(True, linestyle=":", linewidth=0.5)

                axes[2].plot(histories["time"], histories["dp"], color="tab:red")
                axes[2].set_ylabel("Δp")
                axes[2].set_xlabel("Time")
                axes[2].set_title("Pressure drop")
                axes[2].grid(True, linestyle=":", linewidth=0.5)

                fig.tight_layout()
                fig.savefig(output_dir / "diagnostics.png", dpi=200)
                plt.close(fig)
            except Exception as exc:  # pragma: no cover
                print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
