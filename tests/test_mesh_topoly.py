import pytest
import numpy as np
from numpy.testing import assert_equal
import matplotlib.pyplot as plt

# --- Imports from your project structure ---
from pycutfem.core.mesh import Mesh
from pycutfem.utils.meshgen import structured_quad

# --- Test Cases for Mesh Topology ---

class TestMeshTopology:
    def test_get_q1_corners(self):
        """
        Tests if _get_element_corner_global_indices correctly finds the 4 corner
        nodes for a simple Q1 element.
        """
        # A 2x2 grid of Q1 elements. Total nodes = (1*2+1) * (1*2+1) = 9 nodes.
        poly_order = 1
        nodes, elems, edge_connectvity, elem_connectivity_corner_nodes = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=poly_order)
        mesh = Mesh(nodes, elems, edge_connectvity, elem_connectivity_corner_nodes, element_type="quad", poly_order=poly_order)
        
        # Test element 0 (bottom-left)
        # Expected nodes for a 3x3 node grid:
        # 6--7--8
        # |  |  |
        # 3--4--5
        # |  |  |
        # 0--1--2
        # Element 0 corners are GIDs 0, 1, 4, 3
        corners_e0 = mesh.corner_connectivity[0]
        assert_equal(sorted(corners_e0), sorted([0, 1, 4, 3]),
                     err_msg="Incorrect corners for Q1 element 0")
        
        # Test element 3 (top-right)
        # Element 3 corners are GIDs 4, 5, 8, 7
        corners_e3 = mesh.corner_connectivity[3]
        assert_equal(sorted(corners_e3), sorted([4, 5, 8, 7]),
                     err_msg="Incorrect corners for Q1 element 3")

    def test_get_q2_corners(self):
        """
        Tests if _get_element_corner_global_indices correctly finds the 4 corner
        nodes for a higher-order Q2 element, ignoring the other 5 nodes.
        """
        # A single Q2 element. Total nodes = (2*1+1) * (2*1+1) = 9 nodes.
        poly_order = 2
        nodes, elems, edge_connectvity, elem_connectivity_corner_nodes = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=poly_order)
        mesh = Mesh(nodes, elems, edge_connectvity, elem_connectivity_corner_nodes, element_type="quad", poly_order=poly_order)
        
        # Node grid for a single Q2 element:
        # 6--7--8
        # |  |  |
        # 3--4--5
        # |  |  |
        # 0--1--2
        # Corner nodes are at GIDs 0, 2, 8, 6.
        expected_corners = [0, 2, 8, 6]
        actual_corners = mesh.corner_connectivity[0]
        
        assert_equal(sorted(actual_corners), sorted(expected_corners),
                     err_msg="Did not find correct corner nodes for a Q2 element.")

    def test_build_edges_q1(self):
        """
        Tests if _build_edges correctly identifies the number of interior
        and boundary edges for a simple multi-element Q1 mesh.
        """
        # A 2x1 mesh of Q1 elements (two quads side-by-side)
        nodes, elems, edge_connectvity, elem_connectivity_corner_nodes = structured_quad(2.0, 1.0, nx=2, ny=1, poly_order=1)
        mesh = Mesh(nodes, elems, edge_connectvity, elem_connectivity_corner_nodes, element_type="quad", poly_order=1)

        # Expected edges: 4 + 4 - 1 shared = 7 total edges
        assert len(mesh.edges_list) == 7, f"Expected 7 edges, but found {len(mesh.edges_list)}"
        
        num_interior = sum(1 for edge in mesh.edges_list if edge.right is not None)
        num_boundary = sum(1 for edge in mesh.edges_list if edge.right is None)
        
        assert num_interior == 1, f"Expected 1 interior edge, but found {num_interior}"
        assert num_boundary == 6, f"Expected 6 boundary edges, but found {num_boundary}"

    def test_build_edges_q2(self):
        """
        Tests if _build_edges correctly identifies edges for a higher-order Q2 mesh.
        """
        # A 2x1 mesh of Q2 elements
        nodes, elems, edge_connectvity, elem_connectivity_corner_nodes = structured_quad(2.0, 1.0, nx=2, ny=1, poly_order=2)
        mesh = Mesh(nodes, elems, edge_connectvity, elem_connectivity_corner_nodes, element_type="quad", poly_order=2)
        # x_list = [n.x for n in mesh.nodes_list]
        # y_list = [n.y for n in mesh.nodes_list]
        # edges = mesh.edges_list
        # for i, n in enumerate(mesh.nodes_list):
        #     plt.text(n.x, n.y, str(i), fontsize=12, ha='center', va='center')
        # plt.scatter(x_list, y_list, color='red', s=50, label='Nodes')
        # mid_xs =[]
        # mid_ys = []
        # for i, edge in enumerate(edges):
        #     n1, n2 = edge.nodes
        #     mid_x = (mesh.nodes_list[n1].x + mesh.nodes_list[n2].x) / 2
        #     mid_y = (mesh.nodes_list[n1].y + mesh.nodes_list[n2].y) / 2 +.02
        #     plt.text(mid_x, mid_y, f'Edge: {i}', fontsize=10, ha='center', va='center', color='blue')
        #     mid_xs.append(mid_x)
        #     mid_ys.append(mid_y)

        # plt.scatter(mid_xs, mid_ys, color='blue', s=50, label='Edges')
        # plt.show()
        # Expected edges: 4 + 4 - 1 shared = 7 total geometric edges
        assert len(mesh.edges_list) == 7, f"Expected 7 edges for Q2 mesh, but found {len(mesh.edges_list)}"
        
        num_interior = sum(1 for edge in mesh.edges_list if edge.right is not None)
        num_boundary = sum(1 for edge in mesh.edges_list if edge.right is None)
        
        assert num_interior == 1, f"Expected 1 interior edge for Q2 mesh, but found {num_interior}"
        assert num_boundary == 6, f"Expected 6 boundary edges for Q2 mesh, but found {num_boundary}"

        # Verify that the interior edge connects the correct corner nodes.
        # nx=2, ny=1, order=2. num_x_nodes = 2*2+1=5. num_y_nodes = 2*1+1=3
        # 10--11--12--13--14
        #  |   |   |   |   |
        #  5---6---7---8---9
        #  |   |   |   |   |
        #  0---1---2---3---4
        # Elem 0 corners: GIDs 0, 2, 12, 10
        # Elem 1 corners: GIDs 2, 4, 14, 12
        # The shared edge is between nodes 2 and 12.
        shared_edge_nodes = tuple(sorted((2, 12)))
        
        found_edge = False
        for edge in mesh.edges_list:
            if edge.right is not None:
                assert tuple(sorted(edge.nodes)) == shared_edge_nodes
                found_edge = True
        assert found_edge, "Interior edge nodes are not what was expected for Q2 mesh."

