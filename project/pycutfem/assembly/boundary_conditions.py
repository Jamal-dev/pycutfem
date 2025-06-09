
"""pycutfem.assembly.boundary_conditions
Proper Dirichlet elimination with RHS correction.
"""
import numpy as np
import scipy.sparse as sp

def apply_dirichlet(K, F, dbc):
    """Return (K_bc, F_bc).

    Parameters
    ----------
    dbc : dict {dof: value}
        Dirichlet boundary values.
    """
    K = K.tolil(copy=True)
    F = F.copy()

    for dof, val in dbc.items():
        # subtract K[:,dof]*u_D from RHS
        col = K[:, dof].toarray().ravel()
        F -= col * val

        # zero column
        K[:, dof] = 0.0
        # zero row then set diagonal
        K.rows[dof] = [dof]
        K.data[dof] = [1.0]
        F[dof] = val

    return K.tocsr(), F
