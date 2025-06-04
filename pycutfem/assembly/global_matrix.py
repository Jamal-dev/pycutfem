"""pycutfem.assembly.global_matrix"""
import numpy as np, scipy.sparse as sp

def assemble(mesh, local_cb):
    n_dofs = len(mesh.nodes)
    rows, cols, data = [], [], []
    for eid, elem in enumerate(mesh.elements):
        Ke, _ = local_cb(eid)
        for a, A in enumerate(elem):
            for b, B in enumerate(elem):
                rows.append(A); cols.append(B); data.append(Ke[a,b])
    K = sp.csr_matrix((data,(rows,cols)), shape=(n_dofs,n_dofs))
    return K
