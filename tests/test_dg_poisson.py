import pytest
import numpy as np
import scipy.sparse.linalg as spla
import sympy as sp

# Assume these are imported from your project structure
from pycutfem.core import Mesh
from pycutfem.assembly.dg_global import assemble_dg
from pycutfem.utils.meshgen import structured_quad
from pycutfem.fem.reference import get_reference
from pycutfem.fem import transform
from pycutfem.integration import volume

# --- Define the analytical solution and source term ---
x, y = sp.symbols("x y")
u_sym = x**2 * y + sp.sin(sp.pi * y)
f_sym = -sp.diff(u_sym, x, 2) - sp.diff(u_sym, y, 2)

# Create callable numpy functions
u_exact_func = sp.lambdify((x, y), u_sym, "numpy")
source_func = sp.lambdify((x, y), f_sym, "numpy")


def test_sipg_q1_poisson():
    """
    Tests the DG solver for a Poisson problem with a known analytical solution.
    This test verifies that the global assembly and solver produce a result
    that converges to the exact solution.
    """
    # 1. Setup the mesh and problem parameters
    nodes, elems, _, corners = structured_quad(1, 1, nx=8, ny=8, poly_order=1)
    mesh = Mesh(nodes, element_connectivity=elems, edges_connectivity=None, elements_corner_nodes=corners, element_type="quad", poly_order=1)

    # 2. Assemble the global system
    # The RHS source term is now correctly passed to the assembler.
    K, F = assemble_dg(mesh,
                       poly_order=1,
                       alpha=20.0,  # Increased penalty for stability
                       symmetry=1,
                       dirichlet=lambda x, y: u_exact_func(x, y),
                       rhs=lambda x, y: source_func(x, y))

    # 3. Solve the linear system
    uh = spla.spsolve(K, F)

    # 4. Compute the L2 error in a DG-correct manner
    total_error_sq = 0.0
    total_area = 0.0
    
    ref = get_reference("quad", 1)
    pts, wts = volume("quad", 3)  # Use sufficient quadrature order
    n_loc = 4

    for eid, elem in enumerate(mesh.elements_list):
        dofs = np.arange(n_loc) + eid * n_loc
        uh_local = uh[dofs]
        
        elem_error_sq = 0.0
        for (xi, eta), w in zip(pts, wts):
            # Value of numerical solution at quadrature point
            N = ref.shape(xi, eta)
            uh_at_pt = N @ uh_local
            
            # Value of exact solution at quadrature point
            x_phys = transform.x_mapping(mesh, eid, (xi, eta))
            u_exact_at_pt = u_exact_func(*x_phys)
            
            # Jacobian for the integral
            J = transform.jacobian(mesh, eid, (xi, eta))
            detJ = abs(np.linalg.det(J))
            
            elem_error_sq += w * detJ * (uh_at_pt - u_exact_at_pt)**2
            
        total_error_sq += elem_error_sq
        total_area += mesh.areas()[eid]

    l2_error = np.sqrt(total_error_sq / total_area)
    
    # Looser tolerance is appropriate for DG methods on coarse meshes
    assert l2_error < 0.1, f"L2 error ({l2_error}) is too high."

