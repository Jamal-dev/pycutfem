import numpy as np
from pycutfem.core import Mesh
from pycutfem.fem.transform import x_mapping, det_jacobian
def test_reference_to_global_mapping():
    nodes=np.array([[0,0],[2,0],[0,1]])
    elements=np.array([[0,1,2]])
    mesh=Mesh(nodes,elements,element_type='tri')
    x = x_mapping(mesh,0,(1/3,1/3))
    # Barycentric mapping should give interior point roughly (2/3,1/3)
    assert np.allclose(x, [2/3, 1/3])
    # detJ should be twice area
    area=mesh.areas()[0]
    assert np.isclose(det_jacobian(mesh,0,(0.2,0.2)), 2*area)
