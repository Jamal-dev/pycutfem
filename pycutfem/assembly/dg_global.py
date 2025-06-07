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



def assemble_dg(mesh: 'Mesh', *, poly_order: int = 1, n_comp: int = 1,
                alpha: float = 10.0, symmetry: int = 1,
                quad_order: int | None = None,
                dirichlet: Callable = lambda x, y: 0.0,
                rhs: Callable = lambda x, y: 0.0
                ) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Assemble the global stiffness matrix K and right-hand side vector F for a
    Discontinuous Galerkin formulation.

    Parameters
    ----------
    mesh : Mesh
        The mesh object containing the geometry and topology.
    poly_order : int
        The polynomial order of the basis functions.
    n_comp : int
        The number of field components (e.g., 1 for scalar, 2 for vector).
    alpha : float
        The penalty parameter for the SIPG/Nitsche formulation.
    symmetry : int
        The symmetry parameter for the DG formulation (+1 for SIPG, -1 for NIPG).
    quad_order : int, optional
        The quadrature order for numerical integration.
    dirichlet : Callable
        A function defining the Dirichlet boundary conditions.
    rhs : Callable
        A function defining the source term for the right-hand side vector.

    Returns
    -------
    tuple[scipy.sparse.csr_matrix, numpy.ndarray]
        The global stiffness matrix K and the global force vector F.
    """
    # Determine the number of local degrees of freedom from a reference element
    ref = get_reference(mesh.element_type, poly_order)
    n_loc = len(ref.shape(0, 0))
    n_eldof = n_loc * n_comp
    n_dofs = n_eldof * len(mesh.elements_list)

    # Use lists to build sparse matrix data efficiently
    rows, cols, data = [], [], []
    F = np.zeros(n_dofs)

    # --- Volume Integrals ---
    for elem in mesh.elements_list:
        Ke_vol = volume_laplace(mesh, elem.id)

        # Assume dg_element_load computes the RHS vector from the source term
        Fe_vol = dg_element_load(mesh, elem.id, rhs,
                                 poly_order=poly_order, quad_order=quad_order)
        
        # Tile the element load vector for each component
        Fe_full = np.tile(Fe_vol, n_comp)
        dofs = np.arange(n_eldof) + elem.id * n_eldof
        F[dofs] += Fe_full
        
        # Add the local stiffness matrix to the global list
        rr, cc = np.meshgrid(dofs, dofs, indexing='ij')
        rows.extend(rr.ravel())
        cols.extend(cc.ravel())
        data.extend(Ke_vol.ravel())

    # --- Face (Edge) Integrals ---
    for edge in mesh.edges_list:
        Ke_f, Fe_f = face_laplace(mesh, edge.gid, alpha=alpha,
                                  symmetry=symmetry, quad_order=quad_order,
                                  n_comp=n_comp, dirichlet=dirichlet)

        # Get the degrees of freedom for the left element
        dofs_L = np.arange(n_eldof) + edge.left * n_eldof
        
        if edge.right is None:  # Boundary face
            # Add the local face matrix to the global list
            rr, cc = np.meshgrid(dofs_L, dofs_L, indexing='ij')
            rows.extend(rr.ravel())
            cols.extend(cc.ravel())
            data.extend(Ke_f.ravel())
            
            # Add the local force vector to the global vector
            F[dofs_L] += Fe_f
        else:  # Interior face
            # Get the degrees of freedom for the right element
            dofs_R = np.arange(n_eldof) + edge.right * n_eldof
            dofs_face = np.concatenate((dofs_L, dofs_R))
            
            # Add the local face matrix (which is 2x2 blocks) to the global list
            rr, cc = np.meshgrid(dofs_face, dofs_face, indexing='ij')
            rows.extend(rr.ravel())
            cols.extend(cc.ravel())
            data.extend(Ke_f.ravel())
            
            # For interior faces, Fe_f is zero in the current formulation,
            # but we add it for completeness.
            F[dofs_face] += Fe_f

    # Create the final sparse matrix
    K = sp.csr_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs))
    return K, F

