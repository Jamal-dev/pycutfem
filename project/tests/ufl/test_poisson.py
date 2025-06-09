import pytest
import numpy as np
import scipy.sparse as spla
from numpy.testing import assert_allclose

# --- Imports from your project structure ---
from pycutfem.core import Mesh
from pycutfem.utils.meshgen import structured_quad
# Imports from the new UFL-like library
from ufl.expressions import TrialFunction, TestFunction, Constant, grad, inner
from ufl.measures import dx
from ufl.forms import assemble_form

def test_poisson_symbolic_q1():
    """
    Tests the symbolic solver for a Poisson problem with Q1 elements.
    """
    # 1. Setup the mesh
    nodes, elems, _, corners = structured_quad(1, 1, nx=4, ny=4, poly_order=1)
    mesh = Mesh(nodes, element_connectivity=elems, edges_connectivity=None, 
                elements_corner_nodes=corners, element_type="quad", poly_order=1)

    # 2. Define the weak form using the symbolic API
    u = TrialFunction(None) # Placeholder for FunctionSpace
    v = TestFunction(None)  # Placeholder for FunctionSpace
    
    # Define source term f = 1.0
    f = Constant(1.0)
    
    # Weak form: ∫∇v⋅∇u dx = ∫f*v dx
    a = inner(grad(u), grad(v)) * dx()
    L = f * v * dx()

    # The '==' creates an Equation object
    equation = (a == L)
    
    # 3. Assemble the system
    K, F = assemble_form(equation, mesh, quad_order=3)

    # 4. Assertions
    n_elems = len(mesh.elements_list)
    n_dofs = 4 * n_elems 
    assert K.shape == (n_dofs, n_dofs), "Stiffness matrix has incorrect shape."
    assert F.shape == (n_dofs,), "Force vector has incorrect shape."
    assert K.nnz > 0, "Stiffness matrix should not be empty."
    assert np.linalg.norm(F) > 1e-12, "Force vector should not be zero."
    
    # Check if K is symmetric
    assert_allclose(K.toarray(), K.toarray().T, atol=1e-12)
    
    print("\nSymbolic Poisson test passed: Assembly successful.")
    print(f"Matrix shape: {K.shape}, Non-zero entries: {K.nnz}")
    print(f"Force vector norm: {np.linalg.norm(F):.4e}")

