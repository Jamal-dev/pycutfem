import numpy as np
from pycutfem.core import Mesh
def test_neighbors_and_normals():
    nodes=np.array([[0,0],[1,0],[1,1],[0,1]])
    elements=np.array([[0,1,2],[0,2,3]])
    mesh=Mesh(nodes,elements,element_type='tri')
    assert len(mesh.edges)==5
    assert np.allclose([np.linalg.norm(e.normal) for e in mesh.edges], 1.0)
    assert mesh.neighbors()[0]==[1] and mesh.neighbors()[1]==[0]
