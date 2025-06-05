"""pycutfem.assembly.dg_global
Block assembly for DG scalar Poisson (SIPG).
"""
import numpy as np, scipy.sparse as sp
from pycutfem.assembly.dg_local import volume_laplace, face_laplace
from pycutfem.assembly.load_vector import dg_element_load


def assemble_dg(mesh, *, poly_order=1, penalty=10.0, quad_order=None,
                dirichlet=lambda x, y: 0.0,
                rhs=lambda x, y: 0.0):
    """
    Assemble CSR matrix and load vector for scalar SIPG Poisson.

    Parameters
    ----------
    dirichlet : callable(x,y)       Dirichlet value on boundary faces
    rhs       : callable(x,y)       Volume load f(x,y)
    """
    n_loc = (poly_order + 1) ** 2 if mesh.element_type == "quad" else \
            (poly_order + 1) * (poly_order + 2) // 2
    n_dofs = n_loc * len(mesh.elements)

    rows, cols, data = [], [], []
    F = np.zeros(n_dofs)

    # ---- volume ------------------------------------------------------
    for eid in range(len(mesh.elements)):
        Ke   = volume_laplace(mesh, eid, poly_order=poly_order,
                              quad_order=quad_order)
        Fe   = dg_element_load(mesh, eid, rhs,
                               poly_order=poly_order,
                               quad_order=quad_order)
        dofs = np.arange(n_loc) + eid * n_loc
        for a, A in enumerate(dofs):
            F[A] += Fe[a]
            for b, B in enumerate(dofs):
                rows.append(A); cols.append(B); data.append(Ke[a, b])

    # ---- faces -------------------------------------------------------
    for edge in mesh.edges:
        K_LL, K_LR, K_RL, K_RR, F_L, F_R = face_laplace(
            mesh, edge.left, edge.right, edge.id,
            poly_order=poly_order, penalty=penalty,
            quad_order=quad_order, dirichlet=dirichlet)

        dofs_L = np.arange(n_loc) + edge.left * n_loc
        F[dofs_L] += F_L
        for a, A in enumerate(dofs_L):
            for b, B in enumerate(dofs_L):
                rows.append(A); cols.append(B); data.append(K_LL[a, b])

        if edge.right is not None:
            dofs_R = np.arange(n_loc) + edge.right * n_loc
            F[dofs_R] += F_R
            for a, A in enumerate(dofs_L):
                for b, B in enumerate(dofs_R):
                    rows.append(A); cols.append(B); data.append(K_LR[a, b])
            for a, A in enumerate(dofs_R):
                for b, B in enumerate(dofs_L):
                    rows.append(A); cols.append(B); data.append(K_RL[a, b])
            for a, A in enumerate(dofs_R):
                for b, B in enumerate(dofs_R):
                    rows.append(A); cols.append(B); data.append(K_RR[a, b])

    K = sp.csr_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs))
    return K, F

