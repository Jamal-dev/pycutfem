from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import LevelSetGridFunction
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.bitset import BitSet
from pycutfem.utils.ogrid_meshgen import circular_hole_ogrid


# -----------------------------------------------------------------------------
# Turek–Hron FSI-2 geometry (channel + cylinder + beam level set)
# -----------------------------------------------------------------------------
H = 0.41
L = 2.2
RADIUS = 0.05
CENTER = (0.2, 0.2)

BEAM_LENGTH = 0.35
BEAM_HEIGHT = 0.02
BEAM_CENTER = (CENTER[0] + RADIUS + 0.5 * BEAM_LENGTH, CENTER[1])


def _count_segments(width: float, mesh_size: float, *, min_cells: int = 1) -> int:
    if float(width) <= 1.0e-12:
        return 0
    return max(int(min_cells), int(math.ceil(float(width) / float(mesh_size))))


def tag_channel_boundaries(mesh: Mesh, mesh_size: float) -> None:
    tol = 1.0e-9
    circle_tol = max(0.25 * float(mesh_size), 1.0e-4)
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
        if on_circle(float(mpx), float(mpy)):
            edge.tag = "cylinder"

    # cache the bitset + locator for downstream convenience
    cyl_mask = np.fromiter((getattr(e, "tag", "") == "cylinder" for e in mesh.edges_list), bool)
    mesh._edge_bitsets = getattr(mesh, "_edge_bitsets", {})
    mesh._edge_bitsets["cylinder"] = BitSet(cyl_mask)
    loc_map = getattr(mesh, "_boundary_locators", {})
    loc_map["cylinder"] = on_circle
    mesh._boundary_locators = loc_map


def build_structured_channel_mesh(mesh_size: float, poly_order: int) -> Mesh:
    """
    Structured O-grid mesh of the channel with a circular hole.

    This mirrors the "structured" backend in `examples/turek_fsi_fully_eulerian.py`
    but is safe to import from tests/benchmarks (no CLI side effects).
    """
    mesh_size = float(mesh_size)
    poly_order = int(poly_order)

    base_margin = max(2.5 * mesh_size, 0.015)
    min_half_cap = min(CENTER[0], L - CENTER[0], CENTER[1], H - CENTER[1])
    min_ring = max(0.5 * mesh_size, 0.005)
    max_margin = min_half_cap - (RADIUS + min_ring)
    if max_margin <= 0.0:
        raise RuntimeError("O-grid collapsed: circle too close to boundary; reduce radius or move center.")
    margin = min(base_margin, max_margin)

    half_x_cap = min(CENTER[0], L - CENTER[0]) - margin
    half_y_cap = min(CENTER[1], H - CENTER[1]) - margin
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


class BeamArcRootLevelSet:
    """
    Beam level set with a curved root that follows the cylinder arc.

    This mirrors the helper in `examples/turek_fsi_fully_eulerian.py`.
    """

    def __init__(
        self,
        *,
        beam_center: Tuple[float, float],
        beam_length: float,
        beam_height: float,
        cyl_center: Tuple[float, float],
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

        # Bias in a narrow band around the root to prefer inside on the beam side.
        root_band = (np.abs(x_coord - x_arc) <= self._root_tol) & (y_coord >= self._beam_y0) & (y_coord <= self._beam_y1)
        if np.any(root_band):
            phi = np.asarray(phi, float)
            phi[root_band] = phi[root_band] - self._root_bias
        return phi


@dataclass(frozen=True)
class TurekFSI2Setup:
    mesh: Mesh
    dof_handler: DofHandler
    level_set: LevelSetGridFunction
    edge_tol: float


def build_turek_fsi2_setup(
    *,
    mesh_size: float = 0.025,
    poly_order: int = 2,
    beam_shift_x: float = 0.0,
    root_inset: float | None = None,
    root_bias: float | None = None,
    root_tol: float | None = None,
    edge_tol: float | None = None,
) -> TurekFSI2Setup:
    """
    Build the structured Turek FSI-2 mesh and a FE-backed beam level set.

    Returns a `LevelSetGridFunction` committed/classified against the mesh.
    """
    mesh_size = float(mesh_size)
    poly_order = int(poly_order)

    beam_center = (BEAM_CENTER[0] - float(beam_shift_x), BEAM_CENTER[1])
    beam_length = BEAM_LENGTH + 2.0 * float(beam_shift_x)
    beam_height = BEAM_HEIGHT

    if root_inset is None:
        root_inset = max(5.0e-4, 0.04 * mesh_size)
    if root_bias is None:
        root_bias = max(1.0e-8, 1.0e-4 * mesh_size)
    if root_tol is None:
        root_tol = max(1.0e-6, 1.0e-3 * mesh_size)
    if edge_tol is None:
        edge_tol = max(1.0e-12, 1.0e-10 * mesh_size)

    mesh = build_structured_channel_mesh(mesh_size, poly_order)

    beam_ref_ls = BeamArcRootLevelSet(
        beam_center=beam_center,
        beam_length=beam_length,
        beam_height=beam_height,
        cyl_center=CENTER,
        cyl_radius=RADIUS,
        root_inset=float(root_inset),
        root_bias=float(root_bias),
        root_tol=float(root_tol),
    )

    # FE-backed level set on its own scalar space
    ls_me = MixedElement(mesh, field_specs={"phi_beam": poly_order})
    ls_dh = DofHandler(ls_me, method="cg")
    level_set = LevelSetGridFunction(ls_dh, field="phi_beam")
    level_set.interpolate(lambda x, y: float(beam_ref_ls(np.array([x, y], dtype=float))))
    level_set.commit(tol=float(edge_tol))

    # Main unknown handler (matches the example)
    me = MixedElement(
        mesh,
        field_specs={
            "u_pos_x": poly_order,
            "u_pos_y": poly_order,
            "p_pos_": poly_order,
            "vs_neg_x": poly_order,
            "vs_neg_y": poly_order,
            "d_neg_x": poly_order,
            "d_neg_y": poly_order,
        },
    )
    dh = DofHandler(me, method="cg")

    return TurekFSI2Setup(mesh=mesh, dof_handler=dh, level_set=level_set, edge_tol=float(edge_tol))

