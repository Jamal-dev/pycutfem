import math

import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node
from pycutfem.utils.meshgen import structured_quad
from pycutfem.utils.ogrid_meshgen import circular_hole_ogrid


def test_explicit_edges_q2_boundary_edges_expose_all_nodes():
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    mesh.tag_boundary_edges({"left": lambda x, _y: np.isclose(x, 0.0)})
    for edge_id in mesh.edge_bitset("left").to_indices():
        e = mesh.edge(int(edge_id))
        assert len(e.all_nodes) == 3
        coords = mesh.nodes_x_y_pos[list(e.all_nodes)]
        assert np.allclose(coords[:, 0], 0.0)


def test_explicit_edges_ogrid_cylinder_edges_expose_all_nodes_q2():
    Lx = 1.0
    Ly = 1.0
    center = (0.55, 0.45)
    radius = 0.2
    nodes, elements, edges, corners = circular_hole_ogrid(
        Lx,
        Ly,
        circle_center=center,
        circle_radius=radius,
        ring_thickness=0.12,
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

    circle_nodes = {
        node.id
        for node in mesh.nodes_list
        if node.tag and "boundary_circle" in node.tag.split(",")
    }
    assert circle_nodes, "Expected tagged circle nodes."

    circle_tol = 1.0e-8
    mesh.tag_boundary_edges(
        {
            "cylinder": lambda x, y: abs(math.hypot(x - center[0], y - center[1]) - radius)
            <= circle_tol
        }
    )
    cyl_edges = mesh.edge_bitset("cylinder").to_indices()
    assert cyl_edges.size > 0
    for edge_id in cyl_edges:
        e = mesh.edge(int(edge_id))
        assert len(e.all_nodes) == 3
        assert all(int(nid) in circle_nodes for nid in e.all_nodes)


def test_hanging_edge_inference_handles_curved_q2_edge():
    nodes = [
        Node(0, 0.0, 0.0),  # bl
        Node(1, 0.5, 0.1),  # bottom mid (off the chord -> curved)
        Node(2, 1.0, 0.0),  # br
        Node(3, 0.0, 0.5),  # left mid
        Node(4, 0.5, 0.5),  # center
        Node(5, 1.0, 0.5),  # right mid
        Node(6, 0.0, 1.0),  # tl
        Node(7, 0.5, 1.0),  # top mid
        Node(8, 1.0, 1.0),  # tr
    ]
    elems = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=int)
    corners = np.array([[0, 2, 8, 6]], dtype=int)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=None,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    # The curved bottom side should be represented by segments (0-1) and (1-2).
    keys = {tuple(sorted(e.nodes)) for e in mesh.edges_list}
    assert (0, 1) in keys
    assert (1, 2) in keys
