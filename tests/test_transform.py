import numpy as np
from pycutfem.core import Mesh
from pycutfem.fem.transform import x_mapping, det_jacobian
from pycutfem.core.topology import Node
def test_reference_to_global_mapping():
    n1 = Node(0, 0,0)
    n2 = Node(1, 2,0)
    n3 = Node(2, 0,1)
    nodes=np.array([n1,n2,n3])
    element_connectivity=np.array([[0,1,2]])
    edge_connectivity=np.array([[0,1],[1,2],[2,0]])
    element_connectivity_corner_nodes=np.array([[0,1,2]])
    mesh=Mesh(nodes,element_connectivity,edge_connectivity, element_connectivity_corner_nodes,element_type='tri')
    print(mesh.nodes_list)
    x = x_mapping(mesh,0,(1/3,1/3))
    # Barycentric mapping should give interior point roughly (2/3,1/3)
    assert np.allclose(x, [2/3, 1/3])
    # detJ should be twice area
    area=mesh.areas()[0]
    assert np.isclose(det_jacobian(mesh,0,(0.2,0.2)), 2*area)
