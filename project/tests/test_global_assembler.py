import numpy as np, scipy.sparse as sp
from pycutfem.utils.meshgen import structured_quad
from pycutfem.core import Mesh
from pycutfem.assembly.local_assembler import stiffness_matrix
from pycutfem.assembly.global_matrix import assemble

def test_global_matrix_shape():
    nodes, quads, edge_connectivity, elem_connectivity_corner_nodes = structured_quad(1,1, nx=4, ny=4, poly_order=1)
    mesh = Mesh(nodes, quads, edge_connectivity, elem_connectivity_corner_nodes, element_type='quad')
    K = assemble(mesh, lambda eid: stiffness_matrix(mesh, eid))
    n = len(nodes)
    assert K.shape == (n, n)
    assert isinstance(K, sp.csr_matrix)
