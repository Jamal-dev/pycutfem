import numpy as np
from pycutfem.core import Mesh
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.core.topology import Node
from pycutfem.utils.meshgen import structured_quad, structured_triangles
from pycutfem.core.levelset import LevelSetFunction

# Note: The LineLevelSet is not used by these tests but is kept as it was provided.
class LineLevelSet(LevelSetFunction):
    """Defines a level set for a vertical line phi(x,y) = x - c."""
    def __init__(self,m, b: float = 0.0):
        self.m = m
        self.b = b
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.m *x[..., 0] - self.b
    def gradient(self, x: np.ndarray) -> np.ndarray:
        grad = self.m
        if x.ndim > 1:
            return np.tile(grad, (x.shape[0], 1))
        return grad

def test_element_and_edge_classification_simple_corrected():
    """
    Tests classification on a very simple 2-element mesh.
    This test is corrected to assert the specific 'ghost_both' tag.
    """
    # 1. Setup
    nodes_list, elements_np, _, corners_np = structured_triangles(1.0, 1.0, nx_quads=1, ny_quads=1, poly_order=1)
    mesh = Mesh(nodes=nodes_list, element_connectivity=elements_np,
                elements_corner_nodes=corners_np,
                element_type='tri', poly_order=1)
    
    ls = CircleLevelSet(center=(0.5, 0.5), radius=0.3)

    # 2. Classify
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)

    # 3. Assert
    # Both elements will be 'cut'. According to mesh.py logic, the
    # edge between two 'cut' elements is tagged 'ghost_both'.
    interior_edge_tags = {e.tag for e in mesh.edges_list if e.right is not None}
    
    # CORRECTED ASSERTION: Check for the specific tag.
    assert 'ghost_both' in interior_edge_tags, "The edge between two 'cut' elements should be 'ghost_both'."


def test_full_interface_and_ghost_edge_creation_corrected():
    """
    Creates a mesh and a circle level set that reliably partitions it, ensuring
    all element and edge types are generated and correctly identified.
    """
    # 1. Setup
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=8, ny=8, poly_order=1)
    ls = CircleLevelSet(center=(0.5, 0.5), radius=0.35)
    mesh = Mesh(nodes=nodes,
                element_connectivity=elems,
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

    # CORRECTED ASSERTION: Check for the presence of ANY of the specific ghost tags.
    # We check if the set of generated tags has a non-empty intersection with the
    # set of possible ghost tags.
    possible_ghost_tags = {'ghost_pos', 'ghost_neg', 'ghost_both'}
    assert 'interface' in edge_tags, "Classification failed: No 'interface' edges were tagged."
    assert not edge_tags.isdisjoint(possible_ghost_tags), "Classification failed: No ghost edges ('ghost_pos', 'ghost_neg', or 'ghost_both') were tagged."

    print("\nTest PASSED: Successfully generated and identified all tag types.")