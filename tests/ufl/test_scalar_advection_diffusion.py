import pytest
import numpy as np
import scipy.sparse.linalg as spla
import sympy as sp

# --- Core imports from the refactored library ---
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler

# --- UFL-like imports ---
from pycutfem.ufl.expressions import TrialFunction, TestFunction, grad, inner, dot, Constant
from pycutfem.ufl.measures import dx
from pycutfem.ufl.forms import BoundaryCondition, assemble_form
from pycutfem.ufl.analytic import Analytic, x, y
from pycutfem.fem.mixedelement import MixedElement

# --- Utility imports ---
from pycutfem.utils.meshgen import structured_quad



def test_diffusion_solve():
    """
    Tests the symbolic solver for a steady-state advection-diffusion problem.
    -∇⋅(ε∇u)  = f
    
    This test is updated to use the DofHandler-centric framework.
    """
    # 1. Define Analytical Solution and Parameters
    beta_vec = np.array([1.0, 1.0])
    epsilon = 1.0

    u_exact_sym = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    f_sym = -epsilon*sp.diff(u_exact_sym, x, 2) - epsilon*sp.diff(u_exact_sym, y, 2)
            

    # The 'value' for a BC must be a callable function (lambda x, y: ...),
    # so we use sympy.lambdify to create it from the symbolic expression.
    u_exact_func = sp.lambdify((x, y), u_exact_sym, 'numpy')
    
    # The source term `f` can remain an Analytic object for the compiler.
    f = Analytic(f_sym, degree=4, dim=0)

    # 2. Setup Mesh and DofHandler (which replaces FunctionSpace)
    poly_order = 1
    nodes, elems, _, corners = structured_quad(1, 1, nx=16, ny=16, poly_order=poly_order)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, 
                elements_corner_nodes=corners, element_type="quad", poly_order=poly_order)

    # Define the field and create the DofHandler. This is the modern way
    # to define the problem's function space.
    me = MixedElement(mesh, field_specs={'u': poly_order})
    dof_handler = DofHandler(me, method='cg')

    # 3. Define the Weak Form
    # Trial/Test functions are now identified by their field name (string).
    u = TrialFunction('u',dof_handler=dof_handler)
    v = TestFunction( 'u',dof_handler=dof_handler)
    beta = Constant(beta_vec,dim=1)

    # The weak form definition remains syntactically the same.
    a = (epsilon * inner(grad(u), grad(v))  ) * dx(metadata={"q":4})
    L = f * v * dx(metadata={"q":4})
    equation = (a == L)

    # 4. Define Boundary Conditions
    # The BoundaryCondition class now takes the field name as a string.
    bc_tags = {
        'bottom': lambda x,y: np.isclose(y,0),
        'left':   lambda x,y: np.isclose(x,0),
        'right':  lambda x,y: np.isclose(x,1.0),
        'top':     lambda x,y: np.isclose(y,1.0)
    }
    mesh.tag_boundary_edges(bc_tags)
    bcs = [
        BoundaryCondition(
            field='u', 
            method='dirichlet', 
            domain_tag=tag, 
            value=u_exact_func
        ) for tag in bc_tags.keys()
    ]

    # 5. Assemble and Solve
    # The assemble_form function now takes the DofHandler as the main argument
    # for managing the function space and mesh.
    K, F = assemble_form(equation, dof_handler=dof_handler, bcs=bcs, quad_order=5)
    uh_vec = spla.spsolve(K, F)

    # 6. Verify Solution
    # For a CG simulation, the resulting solution vector `uh_vec` directly
    # corresponds to the values at the nodes of the mesh.
    
    # Get the coordinates of all DOFs from the mesh's node list.
    node_coords = dof_handler.get_dof_coords('u')
    
    # Evaluate the exact solution at every node for comparison.
    exact_vals_at_nodes = u_exact_func(node_coords[:, 0], node_coords[:, 1])
    
    # The L2 error can be computed directly between the solution vector
    # and the exact values at the nodes.
    l2_error = np.sqrt(np.mean((uh_vec - exact_vals_at_nodes)**2))
    
    print(f"\nL2 Error for Diffusion problem: {l2_error:.4e}")
    assert l2_error < 0.05, f"Solution error is too high. Expected < 0.05, got {l2_error:.4e}."




def test_advection_diffusion_solve():
    """
    Tests the symbolic solver for a steady-state advection-diffusion problem.
    -∇⋅(ε∇u) + β⋅∇u = f
    
    This test is updated to use the DofHandler-centric framework.
    """
    # 1. Define Analytical Solution and Parameters
    beta_vec = np.array([1.0, 1.0])
    epsilon = 1.0

    u_exact_sym = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    f_sym = -epsilon*sp.diff(u_exact_sym, x, 2) - epsilon*sp.diff(u_exact_sym, y, 2) + \
            beta_vec[0]*sp.diff(u_exact_sym, x) + beta_vec[1]*sp.diff(u_exact_sym, y)

    # The 'value' for a BC must be a callable function (lambda x, y: ...),
    # so we use sympy.lambdify to create it from the symbolic expression.
    u_exact_func = sp.lambdify((x, y), u_exact_sym, 'numpy')
    
    # The source term `f` can remain an Analytic object for the compiler.
    f = Analytic(f_sym, degree=4, dim=0)

    # 2. Setup Mesh and DofHandler (which replaces FunctionSpace)
    poly_order = 1
    nodes, elems, _, corners = structured_quad(1, 1, nx=16, ny=16, poly_order=poly_order)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, 
                elements_corner_nodes=corners, element_type="quad", poly_order=poly_order)

    # Define the field and create the DofHandler. This is the modern way
    # to define the problem's function space.
    me = MixedElement(mesh, field_specs={'u': poly_order})
    dof_handler = DofHandler(me, method='cg')

    # 3. Define the Weak Form
    # Trial/Test functions are now identified by their field name (string).
    u = TrialFunction('u',dof_handler=dof_handler)
    v = TestFunction( 'u',dof_handler=dof_handler)
    beta = Constant(beta_vec,dim=1)

    # The weak form definition remains syntactically the same.
    a = (epsilon * inner(grad(u), grad(v))  + dot(dot(grad(u),beta ) , v)) * dx(metadata={"q":4})
    L = f * v * dx(metadata={"q":4})
    equation = (a == L)

    # 4. Define Boundary Conditions
    # The BoundaryCondition class now takes the field name as a string.
    bc_tags = {
        'bottom': lambda x,y: np.isclose(y,0),
        'left':   lambda x,y: np.isclose(x,0),
        'right':  lambda x,y: np.isclose(x,1.0),
        'top':     lambda x,y: np.isclose(y,1.0)
    }
    mesh.tag_boundary_edges(bc_tags)
    bcs = [
        BoundaryCondition(
            field='u', 
            method='dirichlet', 
            domain_tag=tag, 
            value=u_exact_func
        ) for tag in bc_tags.keys()
    ]

    # 5. Assemble and Solve
    # The assemble_form function now takes the DofHandler as the main argument
    # for managing the function space and mesh.
    K, F = assemble_form(equation, dof_handler=dof_handler, bcs=bcs, quad_order=5)
    uh_vec = spla.spsolve(K, F)

    # 6. Verify Solution
    # For a CG simulation, the resulting solution vector `uh_vec` directly
    # corresponds to the values at the nodes of the mesh.
    
    # Get the coordinates of all DOFs from the mesh's node list.
    node_coords = dof_handler.get_dof_coords('u')
    
    # Evaluate the exact solution at every node for comparison.
    exact_vals_at_nodes = u_exact_func(node_coords[:, 0], node_coords[:, 1])
    
    # The L2 error can be computed directly between the solution vector
    # and the exact values at the nodes.
    l2_error = np.sqrt(np.mean((uh_vec - exact_vals_at_nodes)**2))
    
    print(f"\nL2 Error for Advection-Diffusion problem: {l2_error:.4e}")
    assert l2_error < 0.05, f"Solution error is too high. Expected < 0.05, got {l2_error:.4e}."

