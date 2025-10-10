import numpy as np
from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import CircleLevelSet, CompositeLevelSet
from pycutfem.cutters import classify_elements, classify_elements_multi, classify_edges
from pycutfem.utils.meshgen import structured_quad, structured_triangles
from pycutfem.core.topology import Node
def test_multi_level_set_classification():

    nodes, elements, edge_connectivity, corners = structured_triangles(1.0, 1.0, nx_quads=1, ny_quads=1, poly_order=1)
    mesh =Mesh(nodes, elements, edges_connectivity=edge_connectivity, elements_corner_nodes=corners, element_type='tri', poly_order=1)

    # Radii chosen so each circle cuts one of the two triangles in the mesh
    ls1=CircleLevelSet(center=(0.25,0.5), radius=0.6)
    ls2=CircleLevelSet(center=(0.75,0.5), radius=0.6)
    results = mesh.classify_elements_multi( [ls1, ls2])
    # Each results[idx] returns tuple of arrays; verify keys
    assert 0 in results and 1 in results
    # At least one cut element across both sets
    cut_counts = {idx: len(results[idx][2]) for idx in results}
    assert any(count > 0 for count in cut_counts.values()), "Expected at least one cut element across the level sets."
