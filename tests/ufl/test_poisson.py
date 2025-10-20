import pytest
import numpy as np
import scipy.sparse as spla
from numpy.testing import assert_allclose

# --- Imports from your project structure ---
from pycutfem.core.mesh import Mesh
from pycutfem.utils.meshgen import structured_quad
# Imports from the new UFL-like library
from pycutfem.ufl.expressions import TrialFunction,dot, TestFunction, Constant, grad, inner
from pycutfem.ufl.measures import dx
from pycutfem.ufl.forms import assemble_form, BoundaryCondition, Equation
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.fem.mixedelement import MixedElement

def test_poisson_symbolic_q1():
    """
    Tests the symbolic solver for a Poisson problem with Q1 elements.
    """
    # 1. Setup the mesh
    nodes, elems, _, corners = structured_quad(1, 1, nx=4, ny=4, poly_order=1)
    mesh = Mesh(nodes, element_connectivity=elems, edges_connectivity=None, 
                elements_corner_nodes=corners, element_type="quad", poly_order=1)

    me = MixedElement(mesh, field_specs={'u': 1})  # Single field for Poisson problem
    
    dof_handler = DofHandler(me, method='cg')
    u = TrialFunction(name='u_trial',field_name='u',dof_handler=dof_handler)
    v = TestFunction (name='v_test', field_name='u',dof_handler=dof_handler)

    # Define source term f = 1.0
    f = Constant(1.0)
    
    # Weak form: ∫∇v⋅∇u dx = ∫f*v dx
    a = inner(grad(u), grad(v)) * dx()
    L = f * v * dx()

    # The '==' creates an Equation object
    equation = (Equation(a,L))

    # 2. Define boundary conditions (Dirichlet)
    bc_tags = {
        'left': lambda x, y: np.isclose(x, 0),
        'right': lambda x, y: np.isclose(x, 1),
        'bottom': lambda x, y: np.isclose(y, 0),
        'top': lambda x, y: np.isclose(y, 1)
    }
    mesh.tag_boundary_edges(bc_tags)
    bcs = [BoundaryCondition('u', 'dirichlet', 'left', lambda x, y: 0.0),
           BoundaryCondition('u', 'dirichlet', 'right', lambda x, y: 0.0),
           BoundaryCondition('u', 'dirichlet', 'bottom', lambda x, y: 0.0),
           BoundaryCondition('u', 'dirichlet', 'top', lambda x, y: 0.0)]

    K, F = assemble_form(equation, dof_handler= dof_handler, bcs=bcs, quad_order=5)
    # 4. Assertions
    n_elems = len(mesh.elements_list)
    dofs_x = np.sqrt(n_elems) + 1  # Q1 has (n+1)x(n+1) nodes for n elements
    n_dofs = dofs_x * dofs_x  # Total number of DOFs for a square mesh 
    assert n_dofs == dof_handler.total_dofs, "Mismatch in total DOFs."
    assert K.shape == (n_dofs, n_dofs), "Stiffness matrix has incorrect shape."
    assert F.shape == (n_dofs,), "Force vector has incorrect shape."
    assert K.nnz > 0, "Stiffness matrix should not be empty."
    assert np.linalg.norm(F) > 1e-12, "Force vector should not be zero."
    
    # Check if K is symmetric
    assert_allclose(K.toarray(), K.toarray().T, atol=1e-12)
    
    print("\nSymbolic Poisson test passed: Assembly successful.")
    print(f"Matrix shape: {K.shape}, Non-zero entries: {K.nnz}")
    print(f"Force vector norm: {np.linalg.norm(F):.4e}")

