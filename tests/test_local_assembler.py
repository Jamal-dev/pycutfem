import numpy as np
from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.assembly.local_assembler import stiffness_matrix

def test_local_stiffness_symmetry():
    nodes, quads, edge_connectivity, elem_connectivity_corner_nodes = structured_quad(1,1, nx=1, ny=1,poly_order=1)
    mesh = Mesh(nodes, quads, edge_connectivity, elem_connectivity_corner_nodes, element_type='quad')
    Ke,_ = stiffness_matrix(mesh, 0)
    assert np.allclose(Ke, Ke.T)
