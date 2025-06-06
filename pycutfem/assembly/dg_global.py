"""pycutfem.assembly.dg_global
Block assembly for DG scalar Poisson (SIPG).
"""
import numpy as np, scipy.sparse as sp
from pycutfem.assembly.dg_local import volume_laplace, face_laplace
from pycutfem.assembly.load_vector import dg_element_load
from typing import Callable
from pycutfem.fem import transform
from pycutfem.fem.reference import get_reference
from pycutfem.integration import edge, volume
from pycutfem.assembly.dg_local import _find_local_edge, element_char_length



def assemble_dg(mesh, *, poly_order: int = 1, n_comp: int = 1,
                alpha: float = 10.0, symmetry: int = 1,
                quad_order: int | None = None,
                dirichlet: Callable[[float, float], float] | Callable[[float, float], np.ndarray] = lambda x, y: 0.0,
                rhs: Callable[[float, float], float] | Callable[[float, float], np.ndarray] = lambda x, y: 0.0
                ) -> tuple[sp.csr_matrix, np.ndarray]:
    """Assemble global CSR matrix *K* and RHS *F* for scalar/vector SIPG.

    Parameters
    ----------
    n_comp   : number of field components (1 → scalar, 2/3 → vector)
    alpha    : penalty scaling constant in σ=α(p+1)(p+d)/(d h_f)
    symmetry : +1 (SIPG), -1 (NIPG), 0 (IIPG)
    """
    n_loc  = (poly_order + 1) ** 2 if mesh.element_type == "quad" else \
             (poly_order + 1) * (poly_order + 2) // 2
    n_eldof = n_loc * n_comp
    n_dofs  = n_eldof * len(mesh.elements)

    rows, cols, data = [], [], []
    F = np.zeros(n_dofs)

    # ---------------- volume -----------------------------------------
    for eid in range(len(mesh.elements)):
        Ke = volume_laplace(mesh, eid, poly_order=poly_order,
                            quad_order=quad_order, n_comp=n_comp)
        Fe = dg_element_load(mesh, eid, rhs,
                             poly_order=poly_order, quad_order=quad_order)
        # duplicate Fe for each component
        Fe_full = np.tile(Fe, n_comp)
        dofs = np.arange(n_eldof) + eid * n_eldof
        F[dofs] += Fe_full
        rr, cc = np.meshgrid(dofs, dofs, indexing='ij')
        rows.extend(rr.ravel()); cols.extend(cc.ravel()); data.extend(Ke.ravel())

    # ---------------- faces ------------------------------------------
    for edge in mesh.edges:
        Ke_f, Fe_f = face_laplace(mesh, edge.left, edge.right, edge.id,
                                   poly_order=poly_order, alpha=alpha,
                                   symmetry=symmetry, quad_order=quad_order,
                                   n_comp=n_comp, dirichlet=dirichlet)

        if edge.right is None:                     # boundary face
            dofs_L = np.arange(n_eldof) + edge.left * n_eldof
            rr, cc = np.meshgrid(dofs_L, dofs_L, indexing='ij')
            rows.extend(rr.ravel()); cols.extend(cc.ravel()); data.extend(Ke_f.ravel())
            F[dofs_L] += Fe_f
        else:                                      # interior face
            dofs_L = np.arange(n_eldof) + edge.left  * n_eldof
            dofs_R = np.arange(n_eldof) + edge.right * n_eldof
            dofs_face = np.concatenate((dofs_L, dofs_R))
            rr, cc = np.meshgrid(dofs_face, dofs_face, indexing='ij')
            rows.extend(rr.ravel()); cols.extend(cc.ravel()); data.extend(Ke_f.ravel())
            F[dofs_face] += Fe_f

    K = sp.csr_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs))
    return K, F

