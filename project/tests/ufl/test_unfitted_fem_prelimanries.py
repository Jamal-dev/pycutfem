import pytest
import numpy as np
from scipy.sparse.linalg import spsolve
from numpy.testing import assert_allclose

# --- Imports from your project structure ---
from pycutfem.core import Mesh
from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.levelset import CircleLevelSet
# --- Imports from the new UFL-like library ---
from ufl.functionspace import FunctionSpace
from ufl.expressions import TrialFunction, TestFunction, Constant, grad, inner, dot, jump, avg, FacetNormal
from ufl.measures import dx, ds
from ufl.forms import assemble_form

def test_unfitted_poisson_symbolic():
    """
    Tests the symbolic solver for an unfitted Poisson interface problem.
    """
    # 1. Setup the mesh and level set
    nodes, elems, _, corners = structured_quad(2, 2, nx=8, ny=8, poly_order=1)
    mesh = Mesh(nodes, element_connectivity=elems, edges_connectivity=None, 
                elements_corner_nodes=corners, element_type="quad", poly_order=1)
    
    levelset = CircleLevelSet(center=(1.0, 1.0), radius=0.7)
    
    # Classify elements. The compiler will use these tags.
    # We need a more detailed classification for unfitted problems.
    # This is a placeholder for where you would call a function that assigns
    # tags like 'NEG', 'POS', and 'IF' (cut) to mesh.elements_list[i].tag
    for elem in mesh.elements_list:
        # Simplified classification for this test
        centroid = np.mean(mesh.nodes[list(elem.corner_nodes)], axis=0)
        if levelset(centroid) < -0.1: elem.tag = 'NEG'
        elif levelset(centroid) > 0.1: elem.tag = 'POS'
        else: elem.tag = 'IF'
        
    # 2. Define Function Spaces and Symbolic Functions
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    
    # 3. Define the Nitsche weak form symbolically
    alpha_neg = Constant(1.0)
    alpha_pos = Constant(20.0)
    f = Constant(1.0)
    n = FacetNormal()
    h = Constant(np.mean([mesh.element_char_length(i) for i in range(len(mesh.elements_list))]))
    gamma = Constant(20.0)

    # Define restricted functions
    u_neg, v_neg = u.restrict('NEG'), v.restrict('NEG')
    u_pos, v_pos = u.restrict('POS'), v.restrict('POS')

    # Volume terms
    a = alpha_neg * inner(grad(u_neg), grad(v_neg)) * dx('NEG') + \
        alpha_pos * inner(grad(u_pos), grad(v_pos)) * dx('POS')

    # Nitsche interface terms (on faces between 'IF' elements)
    # Note: avg() and jump() are now implemented in the compiler
    a -= inner(avg(grad(u)), jump(v, n)) * ds('IF')
    a -= inner(jump(u, n), avg(grad(v))) * ds('IF')
    a += (gamma / h) * inner(jump(u), jump(v)) * ds('IF')
    
    # RHS
    L = f * v_neg * dx('NEG') + f * v_pos * dx('POS')

    equation = (a == L)
    
    # 4. Assemble the system
    K, F = assemble_form(equation, mesh)

    # 5. Assertions
    n_dofs = 4 * len(mesh.elements_list)
    assert K.shape == (n_dofs, n_dofs), "Stiffness matrix has incorrect shape."
    assert F.shape == (n_dofs,), "Force vector has incorrect shape."
    assert K.nnz > 0, "Stiffness matrix should not be empty."
    assert np.linalg.norm(F) > 1e-9, "Force vector should not be zero."
    
    # Simple check for solution
    # Note: Boundary conditions are not applied, so the matrix is singular.
    # We add a small diagonal term to make it invertible for this test.
    K.setdiag(K.diagonal() + 1e-10)
    uh = spsolve(K, F)
    assert np.linalg.norm(uh) > 1e-9, "Solution appears to be trivial."
    
    print("\nUnfitted symbolic Poisson test passed: Assembly and solve successful.")
    print(f"Norm of solution: {np.linalg.norm(uh):.4e}")
