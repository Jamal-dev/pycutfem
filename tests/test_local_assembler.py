import numpy as np
from pycutfem.utils.meshgen import structured_quad
from pycutfem.core import Mesh
from pycutfem.assembly.local_assembler import stiffness_matrix

def test_local_stiffness_symmetry():
    nodes, quads = structured_quad(1,1, nx=1, ny=1)
    mesh = Mesh(nodes, quads, element_type='quad')
    Ke,_ = stiffness_matrix(mesh, 0)
    assert np.allclose(Ke, Ke.T)
