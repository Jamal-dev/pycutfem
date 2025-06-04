import numpy as np
from pycutfem.core import Mesh
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.cutters import classify_elements, classify_edges

def test_element_and_edge_classification():
    # Square mesh 2 tris
    nodes = np.array([[0,0],[1,0],[1,1],[0,1]])
    elements = np.array([[0,1,2],[0,2,3]])
    mesh = Mesh(nodes, elements, element_type='tri')
    ls = CircleLevelSet(center=(0.5,0.5), radius=0.3)
    inside, outside, cut = classify_elements(mesh, ls)
    # At least one element must be cut in this simple config
    assert len(cut) >= 1
    classify_edges(mesh)
    # Interface edges must exist
    assert 'interface' in mesh.edge_tag
