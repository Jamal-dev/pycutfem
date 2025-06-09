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
from pycutfem.core import Mesh


def assemble_dg(mesh: 'Mesh', *, n_comp: int = 1,
                alpha: float = 10.0, symmetry: int = 1,
                quad_order: int | None = None,
                dirichlet: Callable = lambda x, y: 0.0,
                rhs: Callable = lambda x, y: 0.0
                ) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Assemble the global stiffness matrix K and right-hand side vector F for a
    Discontinuous Galerkin formulation.
    """
    poly_order = mesh.poly_order
    ref = get_reference(mesh.element_type, poly_order)
    n_loc = len(ref.shape(0, 0))
    n_eldof = n_loc * n_comp
    n_dofs = n_eldof * len(mesh.elements_list)

    rows, cols, data = [], [], []
    F = np.zeros(n_dofs)

    # --- Volume Integrals ---
    for elem in mesh.elements_list:
        Ke_vol = volume_laplace(mesh, elem.id, n_comp=n_comp, quad_order=quad_order)
        
        dofs = np.arange(n_eldof) + elem.id * n_eldof
        rr, cc = np.meshgrid(dofs, dofs, indexing='ij')
        rows.extend(rr.ravel())
        cols.extend(cc.ravel())
        data.extend(Ke_vol.ravel())

        # FIX: Force vector must be computed component-wise for vector problems
        for c in range(n_comp):
            rhs_comp = lambda x, y, c=c: np.atleast_1d(rhs(x, y))[c]
            Fe_comp = dg_element_load(mesh, elem.id, rhs_comp, quad_order=quad_order)
            
            dofs_comp = np.arange(n_loc) + elem.id * n_eldof + c * n_loc
            F[dofs_comp] += Fe_comp
            
    # --- Face (Edge) Integrals ---
    for edge in mesh.edges_list:
        Ke_f, Fe_f = face_laplace(mesh, edge.gid, alpha=alpha,
                                  symmetry=symmetry, quad_order=quad_order,
                                  n_comp=n_comp, dirichlet=dirichlet)

        dofs_L = np.arange(n_eldof) + edge.left * n_eldof
        
        if edge.right is None:  # Boundary face
            rr, cc = np.meshgrid(dofs_L, dofs_L, indexing='ij')
            rows.extend(rr.ravel())
            cols.extend(cc.ravel())
            data.extend(Ke_f.ravel())
            F[dofs_L] += Fe_f
        else:  # Interior face
            dofs_R = np.arange(n_eldof) + edge.right * n_eldof
            dofs_face = np.concatenate((dofs_L, dofs_R))
            rr, cc = np.meshgrid(dofs_face, dofs_face, indexing='ij')
            rows.extend(rr.ravel())
            cols.extend(cc.ravel())
            data.extend(Ke_f.ravel())
            F[dofs_face] += Fe_f

    K = sp.csr_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs))
    return K, F