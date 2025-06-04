import numpy as np
from pycutfem.core import Mesh, CircleLevelSet, CompositeLevelSet
from pycutfem.cutters import classify_elements, classify_elements_multi, classify_edges
def test_multi_level_set_classification():
    nodes=np.array([[0,0],[1,0],[1,1],[0,1]])
    elements=np.array([[0,1,2],[0,2,3]])
    mesh=Mesh(nodes,elements,element_type='tri')
    ls1=CircleLevelSet(center=(0.25,0.5), radius=0.3)
    ls2=CircleLevelSet(center=(0.75,0.5), radius=0.3)
    results=classify_elements_multi(mesh, [ls1, ls2])
    # Each results[idx] returns tuple of arrays; verify keys
    assert 0 in results and 1 in results
    # At least one cut element across both sets
    total_cut=sum(len(results[i][2]) for i in results)
    assert total_cut>=1
