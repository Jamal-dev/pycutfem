import math

import numpy as np
import pytest

from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.utils.ogrid_meshgen import circular_hole_ogrid
from pycutfem.fem import transform


def _build_test_mesh():
    Lx = 1.0
    Ly = 1.0
    center = (0.55, 0.45)
    radius = 0.2
    ring = 0.12
    nodes, elements, edges, corners = circular_hole_ogrid(
        Lx,
        Ly,
        circle_center=center,
        circle_radius=radius,
        ring_thickness=ring,
        n_radial_layers=2,
        nx_outer=(2, 4, 2),
        ny_outer=(2, 4, 2),
        poly_order=2,
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elements,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    tol = 1.0e-9
    circle_tol = 1.0e-8
    mesh.tag_boundary_edges(
        {
            "left": lambda x, _y: abs(x - 0.0) <= tol,
            "right": lambda x, _y: abs(x - Lx) <= tol,
            "bottom": lambda _x, y: abs(y - 0.0) <= tol,
            "top": lambda _x, y: abs(y - Ly) <= tol,
            "cylinder": lambda x, y: abs(math.hypot(x - center[0], y - center[1]) - radius)
            <= circle_tol,
        }
    )
    return mesh, center, radius


def test_corners_match_row_major_layout():
    _, elements, _, corners = circular_hole_ogrid(
        1.0,
        1.0,
        circle_center=(0.3, 0.35),
        circle_radius=0.15,
        ring_thickness=0.1,
        n_radial_layers=2,
        nx_outer=(2, 4, 2),
        ny_outer=(2, 4, 2),
        poly_order=2,
    )
    poly_order = 2
    nodes_per_edge = poly_order + 1
    for elem_nodes, elem_corners in zip(elements, corners):
        arr = np.asarray(elem_nodes, dtype=np.int64).reshape(nodes_per_edge, nodes_per_edge)
        assert int(arr[0, 0]) == elem_corners[0]
        assert int(arr[0, -1]) == elem_corners[1]
        assert int(arr[-1, -1]) == elem_corners[2]
        assert int(arr[-1, 0]) == elem_corners[3]


def test_cylinder_dirichlet_dofs_cover_boundary_nodes():
    mesh, center, radius = _build_test_mesh()
    circle_nodes = {
        node.id
        for node in mesh.nodes_list
        if node.tag and "boundary_circle" in node.tag.split(",")
    }
    assert circle_nodes, "Expected tagged circle nodes."

    mixed = MixedElement(mesh, field_specs={"u": 2})
    handler = DofHandler(mixed, method="cg")

    bc = BoundaryCondition("u", "dirichlet", "cylinder", lambda _x, _y: 0.0)
    bc_data = handler.get_dirichlet_data([bc])

    selected_nodes = {
        nid
        for gd in bc_data.keys()
        for field, nid in [handler._dof_to_node_map.get(gd, (None, None))]
        if field == "u" and nid is not None
    }
    missing = circle_nodes - selected_nodes
    assert not missing, f"Cylinder nodes missing from Dirichlet BC: {missing}"


def test_shared_edge_dofs_match_between_blocks():
    mesh, _, _ = _build_test_mesh()
    mixed = MixedElement(mesh, field_specs={"u": 2})
    handler = DofHandler(mixed, method="cg")

    element_maps = handler.element_maps["u"]
    node_to_dof = handler.dof_map["u"]

    for edge in mesh.edges_list:
        if edge.right is None:
            continue
        left = element_maps[edge.left]
        right = element_maps[edge.right]
        left_set = set(left)
        right_set = set(right)
        for nid in edge.all_nodes:
            gd = node_to_dof.get(int(nid))
            assert gd in left_set, f"Left element missing DOF for node {nid} on edge {edge.gid}"
            assert gd in right_set, f"Right element missing DOF for node {nid} on edge {edge.gid}"


def _build_mesh(Lx, Ly, params):
    nodes, elements, edges, corners = circular_hole_ogrid(
        Lx,
        Ly,
        circle_center=params["center"],
        circle_radius=params["radius"],
        ring_thickness=params["ring"],
        n_radial_layers=params["layers"],
        nx_outer=params["nx_outer"],
        ny_outer=params["ny_outer"],
        poly_order=params.get("order", 2),
    )
    return Mesh(
        nodes=nodes,
        element_connectivity=elements,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=params.get("order", 2),
    )


@pytest.mark.parametrize(
    "geom",
    [
        {
            "Lx": 0.6,
            "Ly": 0.5,
            "params": {
                "center": (0.25, 0.25),
                "radius": 0.07,
                "ring": 0.08,
                "layers": 2,
                "nx_outer": (2, 6, 3),
                "ny_outer": (2, 4, 2),
            },
        },
        {
            "Lx": 0.52,
            "Ly": 0.5,
            "params": {
                "center": (0.2, 0.28),
                "radius": 0.05,
                "ring": 0.06,
                "layers": 3,
                "nx_outer": (1, 7, 2),
                "ny_outer": (2, 6, 3),
            },
        },
    ],
)
def test_positive_jacobians_across_parameter_sets(geom):
    mesh = _build_mesh(geom["Lx"], geom["Ly"], geom["params"])
    sample_coords = (-0.75, -0.25, 0.25, 0.75)
    for elem in mesh.elements_list:
        for xi in sample_coords:
            for eta in sample_coords:
                det_j = transform.det_jacobian(mesh, elem.id, (xi, eta))
                assert det_j > 0.0, f"Negative detJ at element {elem.id}"


@pytest.mark.parametrize(
    "Lx,Ly,params",
    [
        (
            0.6,
            0.5,
            {
                "center": (0.25, 0.25),
                "radius": 0.07,
                "ring": 0.08,
                "layers": 2,
                "nx_outer": (2, 6, 3),
                "ny_outer": (2, 4, 2),
            },
        ),
        (
            0.52,
            0.5,
            {
                "center": (0.2, 0.28),
                "radius": 0.05,
                "ring": 0.06,
                "layers": 3,
                "nx_outer": (1, 7, 2),
                "ny_outer": (2, 6, 3),
            },
        ),
    ],
)
def test_total_area_matches_annulus(Lx, Ly, params):
    mesh = _build_mesh(Lx, Ly, params)
    hole_area = math.pi * params["radius"] ** 2
    expected = Lx * Ly - hole_area
    actual = float(mesh.areas().sum())
    assert math.isclose(actual, expected, rel_tol=2e-3, abs_tol=5e-4)
