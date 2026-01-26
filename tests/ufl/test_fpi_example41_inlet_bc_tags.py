import math

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet, MaxLevelSet, MinLevelSet, RotatedBoxLevelSet, ScaledLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.meshgen import structured_quad


def _build_example41_geometry(*, nx: int, poly_order: int):
    x0 = -0.45
    y0 = -0.75
    L = 1.5

    nodes, elems, edges, corners = structured_quad(L, L, nx=nx, ny=nx, poly_order=poly_order, offset=(x0, y0))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    poro_ls = RotatedBoxLevelSet(center=(0.0, 0.0), hx=0.25, hy=0.25, angle=math.pi / 6.0)  # Ω^P
    fluid_sq_ls = RotatedBoxLevelSet(center=(0.0, 0.0), hx=0.5, hy=0.5, angle=math.pi / 4.0)  # base square
    cut_ls = AffineLevelSet(-1.0, 0.0, x0)  # φ = x0 - x  (negative for x > x0)

    outer_std = MaxLevelSet(fluid_sq_ls, cut_ls)
    outer_pos = ScaledLevelSet(-1.0, outer_std)
    fluid_ls = MinLevelSet(poro_ls, outer_pos)  # positive inside Ω^F

    mesh.tag_boundary_edges(
        {
            "inlet": lambda x, y: (abs(x - x0) <= 1.0e-12)
            and (float(fluid_ls(np.array([x, y], dtype=float))) >= -1.0e-12),
        }
    )

    me = MixedElement(mesh, field_specs={"v_pos_x": poly_order, "v_pos_y": poly_order})
    dh = DofHandler(me, method="cg")

    def _inlet_dof_locator(x: float, y: float) -> bool:
        if abs(float(x) - float(x0)) > 1.0e-10:
            return False
        return float(fluid_sq_ls(np.array([float(x), float(y)], dtype=float))) <= 1.0e-12

    dh.tag_dofs_by_locator_map({"inlet_dofs": _inlet_dof_locator}, fields=["v_pos_x", "v_pos_y"])
    return mesh, dh, x0


def _expected_inlet_nodes(*, nx: int) -> int:
    # Rotated outer square (size 1, β=45°) truncated at x=x0=-0.45.
    # The inlet segment is y ∈ [-(√2/2 + x0), +(√2/2 + x0)].
    x0 = -0.45
    y_max = (math.sqrt(2.0) / 2.0) + x0
    assert y_max > 0.0

    y0 = -0.75
    L = 1.5
    h = L / float(nx)
    ys = np.linspace(y0, y0 + L, int(nx) + 1)
    inside = np.abs(ys) <= (y_max + 1.0e-12)
    return int(np.count_nonzero(inside))


def _expected_inlet_edges(*, nx: int) -> int:
    x0 = -0.45
    y_max = (math.sqrt(2.0) / 2.0) + x0
    y0 = -0.75
    L = 1.5
    h = L / float(nx)
    mids = y0 + (np.arange(nx, dtype=float) + 0.5) * h
    inside = np.abs(mids) <= (y_max + 1.0e-12)
    return int(np.count_nonzero(inside))


def test_example41_inlet_dofs_match_expected_nodes_p1():
    # With p=1, inlet DOFs coincide with mesh nodes on x=x0.
    for nx in (2, 4, 8, 16):
        mesh, dh, _x0 = _build_example41_geometry(nx=nx, poly_order=1)
        inlet = set(dh.dof_tags.get("inlet_dofs", set()))
        assert inlet, "Expected non-empty inlet DOF tag."

        expected = _expected_inlet_nodes(nx=nx)
        for field in ("v_pos_x", "v_pos_y"):
            field_ids = set(int(v) for v in dh.get_field_slice(field))
            assert len(inlet & field_ids) == expected

        # Edge tagging is midpoint-based; on coarse meshes it can be empty.
        exp_edges = _expected_inlet_edges(nx=nx)
        assert int(mesh.edge_bitset("inlet").cardinality()) == exp_edges

