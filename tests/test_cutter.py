import numpy as np
from pycutfem.core import Mesh
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.cutters import classify_elements, classify_edges
from pycutfem.core.topology import Node
from pycutfem.utils.meshgen import structured_quad, structured_triangles
from pycutfem.core.levelset import LevelSetFunction

class LineLevelSet(LevelSetFunction):
    """Defines a level set for a vertical line phi(x,y) = x - c."""
    def __init__(self,m, b: float = 0.0):
        self.m = m
        self.b = b
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Handles both single points (shape (2,)) and arrays of points (shape (N,2))
        return self.m *x[..., 0] - self.b
    def gradient(self, x: np.ndarray) -> np.ndarray:
        # Gradient is constant [1, 0]
        grad = self.m
        # If input is a batch of points, tile the gradient to match
        if x.ndim > 1:
            return np.tile(grad, (x.shape[0], 1))
        return grad

def test_element_and_edge_classification_simple():
    """
    Tests classification on a very simple 2-element mesh.
    This test is corrected to pass the level_set to classify_edges
    and to assert the correct 'ghost' tag.
    """
    # 1. Setup
    # Using Node class for clarity, though a raw numpy array would also work with Mesh.
    nodes_list, elements_np, edge_connectivity, corners_np = structured_triangles(1.0, 1.0, nx_quads=1, ny_quads=1, poly_order=1)
    
    # We don't need the edge connectivity table from a generator for this manual mesh.
    
    mesh = Mesh(nodes=nodes_list, element_connectivity=elements_np, 
                edges_connectivity=edge_connectivity,
                elements_corner_nodes=corners_np,
                element_type='tri', poly_order=1)
    # print('tri neighbours:',mesh.neighbors())
    # print('tri elements:',mesh.elements_list)
    # quad_nodes, elements_quads, edge_connectivity_quads, corners_quads = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=1)
    # mesh_quads = Mesh(nodes=quad_nodes, element_connectivity=elements_quads,
    #                   edges_connectivity=edge_connectivity_quads,
    #                   elements_corner_nodes=corners_quads,
    #                   element_type='quad', poly_order=1)
    # # print('quad neighbours:',mesh_quads.neighbors())
    # # print('quad elements:',mesh_quads.elements_list)
    # print('quad edges:',mesh_quads.edges_list)
    # print('tri edges:',mesh.edges_list)
    ls = CircleLevelSet(center=(0.5, 0.5), radius=0.3)

    # 2. Classify
    mesh.classify_elements( ls)
    
    # FIX: Pass the level_set object to classify_edges.
    mesh.classify_edges(ls)

    # 3. Assert
    # In this configuration, both elements are 'cut'.
    # The edge between two 'cut' elements should be tagged 'ghost'.
    print(mesh.elements_list)
    interior_edge_tags = {e.tag for e in mesh.edges_list if e.right is not None}
    assert 'ghost' in interior_edge_tags, "The edge between two 'cut' elements should be 'ghost'."


def test_full_interface_and_ghost_edge_creation():
    """
    Creates a mesh and a circle level set that reliably partitions it, ensuring
    all element and edge types are generated and correctly identified.
    """
    # 1. Setup a mesh and a circle level set.
    # Using a finer 8x8 grid ensures some elements are fully inside the circle.
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=8, ny=8, poly_order=1)
    
    ls = CircleLevelSet(center=(0.5, 0.5), radius=0.35) # Slightly larger radius
    
    # The Mesh class constructor takes the raw numpy arrays.
    mesh = Mesh(nodes=nodes, 
                element_connectivity=elems,
                edges_connectivity=None,  # Not needed for structured mesh
                elements_corner_nodes=corners,
                element_type='quad', 
                poly_order=1)

    # 2. Classify elements and edges
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)

    # 3. Verify the results
    element_tags = {elem.tag for elem in mesh.elements_list}
    edge_tags = {edge.tag for edge in mesh.edges_list if edge.right is not None}

    print("\n--- Full Classification Test Results ---")
    print(f"Generated Element Tags: {element_tags}")
    print(f"Generated Interior Edge Tags: {edge_tags}")
    
    # Assert that our setup created all the expected types of regions
    assert 'inside' in element_tags, "Test setup failed: No 'inside' elements were created."
    assert 'outside' in element_tags, "Test setup failed: No 'outside' elements were created."
    assert 'cut' in element_tags, "Test setup failed: No 'cut' elements were created."

    # This is the key assertion: the setup must produce both interface and ghost edges
    assert 'interface' in edge_tags, "Classification failed: No 'interface' edges were tagged."
    assert 'ghost' in edge_tags, "Classification failed: No 'ghost' edges were tagged."

    print("\nTest PASSED: Successfully generated and identified all tag types.")
