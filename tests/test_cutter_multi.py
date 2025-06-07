import numpy as np
from pycutfem.core import Mesh, CircleLevelSet, CompositeLevelSet
from pycutfem.cutters import classify_elements, classify_elements_multi, classify_edges
from pycutfem.utils.meshgen import structured_quad, structured_triangles
from pycutfem.core.topology import Node
def test_multi_level_set_classification():
    n1=Node(0, 0, 0)
    n2=Node(1, 1, 0)
    n3=Node(2, 1, 1)
    n4=Node(3, 0, 1)
    nodes=[n1, n2, n3, n4]
    nodes, elements, edge_connectivity, corners = structured_triangles(1.0, 1.0, nx_quads=1, ny_quads=1, poly_order=1)
    mesh =Mesh(nodes, elements, edges_connectivity=edge_connectivity, elements_corner_nodes=corners, element_type='tri', poly_order=1)
    
    ls1=CircleLevelSet(center=(0.25,0.5), radius=0.3)
    ls2=CircleLevelSet(center=(0.75,0.5), radius=0.3)
    results = mesh.classify_elements_multi( [ls1, ls2])
    # Each results[idx] returns tuple of arrays; verify keys
    assert 0 in results and 1 in results
    # At least one cut element across both sets
    total_cut=sum(len(results[i][2]) for i in results)
    assert total_cut>=1
