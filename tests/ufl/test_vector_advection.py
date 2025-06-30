import pytest
import numpy as np
import scipy.sparse.linalg as spla
import sympy as sp

# --- Core and UFL Imports ---
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    VectorTrialFunction, VectorTestFunction, VectorFunction,
    grad, inner, dot, Constant
)
from pycutfem.ufl.measures import dx
from pycutfem.ufl.forms import BoundaryCondition, assemble_form
from pycutfem.ufl.analytic import Analytic, x, y
from pycutfem.fem.reference import get_reference
from itertools import product

from pycutfem.fem import transform
from pycutfem.integration import volume
from pycutfem.fem.mixedelement import MixedElement


def test_vector_diffusion_mms():
    """
    Tests the assembly of the vector diffusion term: ∫ ε(∇u : ∇v) dx
    using the method of manufactured solutions for Q2 elements.
    """
    print("\n" + "="*70)
    print("Testing Vector Diffusion Term with Q2 Elements")
    print("="*70)

    # 1. Define Analytical Solution and Parameters using SymPy
    epsilon = 1.0
    
    # Choose a smooth vector solution that is not a simple polynomial
    u_exact_sym_x = sp.sin(sp.pi * x) * sp.cos(sp.pi * y)
    u_exact_sym_y = sp.cos(sp.pi * x) * sp.sin(sp.pi * y)
    
    # Calculate the forcing term f = -εΔu_exact
    f_sym_x = -epsilon * (sp.diff(u_exact_sym_x, x, 2) + sp.diff(u_exact_sym_x, y, 2))
    f_sym_y = -epsilon * (sp.diff(u_exact_sym_y, x, 2) + sp.diff(u_exact_sym_y, y, 2))
    lambdify_f_x = sp.lambdify((x, y), f_sym_x, 'numpy')
    lambdify_f_y = sp.lambdify((x, y), f_sym_y, 'numpy')

    # Lambdify expressions to create callable Python functions
    u_exact_func_x = sp.lambdify((x, y), u_exact_sym_x, 'numpy')
    u_exact_func_y = sp.lambdify((x, y), u_exact_sym_y, 'numpy')
    
    # Pass the forcing term `f` as UFL Analytic objects
    f_x = Analytic(f_sym_x)
    f_y = Analytic(f_sym_y)

    # 2. Setup Q2 Mesh and DofHandler
    poly_order = 2
    nx, ny = 16, 16
    nodes, elems, _, corners = structured_quad(1, 1, nx=nx, ny=ny, poly_order=poly_order)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=poly_order)
    me = MixedElement(mesh,field_specs={'ux': poly_order, 'uy': poly_order})
    dof_handler = DofHandler(me, method='cg')

    # 3. Define the Weak Form
    velocity_space = FunctionSpace("velocity", ['ux', 'uy'])
    u = VectorTrialFunction(velocity_space,dof_handler=dof_handler)
    v = VectorTestFunction( velocity_space,dof_handler=dof_handler)
    f = VectorFunction(name="f", field_names=['ux', 'uy'], dof_handler=dof_handler)
    f.set_values_from_function([lambdify_f_x, lambdify_f_y])
    
    
    # Weak form for -εΔu = f  is  ∫ ε(∇u : ∇v)dx = ∫ f⋅v dx
    a_form = (epsilon * inner(grad(u), grad(v))) * dx(metadata={'quad_order': 4})
    # L_form = (f_x * v[0] + f_y * v[1]) * dx()
    L_form = dot(f,v) * dx()
    equation = (a_form == L_form)

    # 4. Define Boundary Conditions using the exact solution
    bc_tags = {
        'bottom': lambda x,y: np.isclose(y,0),
        'left':   lambda x,y: np.isclose(x,0),
        'right':  lambda x,y: np.isclose(x,1.0),
        'top':     lambda x,y: np.isclose(y,1.0)
    }
    mesh.tag_boundary_edges(bc_tags)
    bcs = [
        BoundaryCondition('ux', 'dirichlet', tag, u_exact_func_x) for tag in bc_tags.keys()]
    bcs = bcs + [
        BoundaryCondition('uy', 'dirichlet', tag, u_exact_func_y) for tag in bc_tags.keys()
    ]

    # 5. Assemble and Solve
    A, b = assemble_form(equation, dof_handler=dof_handler, bcs=bcs, quad_order=5)
    u_vec = spla.spsolve(A, b)

    # 6. Verify Solution against the exact solution
    ux_dofs = dof_handler.get_field_slice('ux')
    uy_dofs = dof_handler.get_field_slice('uy')
    
    # Get node coordinates directly from the mesh for evaluation
    node_coords_ux = dof_handler.get_dof_coords('ux')
    node_coords_uy = dof_handler.get_dof_coords('uy')

    exact = {'ux': u_exact_func_x,
         'uy': u_exact_func_y}
    rel_L2 = dof_handler.l2_error(u_vec,exact,quad_order=5)
    
    # exact_sol_vec = np.zeros_like(u_vec)
    # exact_sol_vec[ux_dofs] = u_exact_func_x(node_coords_ux[:, 0], node_coords_ux[:, 1])
    # exact_sol_vec[uy_dofs] = u_exact_func_y(node_coords_uy[:, 0], node_coords_uy[:, 1])
    
    # l2_error_norm = np.linalg.norm(u_vec - exact_sol_vec)
    # exact_norm = np.linalg.norm(exact_sol_vec)
    # relative_l2_error = l2_error_norm / exact_norm
    
    print(f"\nRelative L2 Error for Q{poly_order} Vector Diffusion: {rel_L2:.4e}")
    assert rel_L2 < 1e-3, "Vector diffusion term assembly failed for Q2 elements."
    print("Vector diffusion test passed successfully!")

def test_q2_dirichlet_dof_collection():
    """
    Verifies that get_dirichlet_data finds ALL nodes (corners and mid-side)
    on a tagged boundary for a Q2 mesh.
    """
    print("\n" + "="*70)
    print("Testing Dirichlet DOF collection for Q2 elements")
    print("="*70)
    
    # 1. Setup a simple 2x2 Q2 mesh
    nx, ny = 2, 2
    poly_order = 2
    nodes, elems, _, corners = structured_quad(1, 1, nx=nx, ny=ny, poly_order=poly_order)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=poly_order)
    me = MixedElement(mesh, field_specs={'u': poly_order})
    dof_handler = DofHandler(me, method='cg')

    # 2. Tag the top boundary
    mesh.tag_boundary_edges({'top': lambda x, y: np.isclose(y, 1.0)})
    
    # Define a simple BC for the top edge
    bcs = [BoundaryCondition('u', 'dirichlet', 'top', lambda x, y: x**2)]

    # 3. Get the Dirichlet data from the DofHandler
    # This is the function we are testing.
    dirichlet_data = dof_handler.get_dirichlet_data(bcs)
    
    # 4. Manually verify the result
    # We will build our own "expected" dictionary by looping through all mesh
    # nodes and checking if they lie on the top boundary.
    expected_data = {}
    top_node_ids = set()
    for node in mesh.nodes_list:
        if np.isclose(node.y, 1.0):
            top_node_ids.add(node.id)
            # Get the global DOF for this node
            dof = dof_handler.dof_map['u'][node.id]
            # Calculate the expected BC value
            expected_value = node.x**2
            expected_data[dof] = expected_value

    print(f"Manually found {len(expected_data)} nodes on the 'top' boundary.")
    print(f"DofHandler found {len(dirichlet_data)} nodes.")

    # For a 2x2 Q2 mesh, there are 2*nx+1 = 5 nodes on the top edge.
    assert len(expected_data) == 2 * nx + 1, "Manual verification logic is wrong."
    assert len(dirichlet_data) == len(expected_data), \
        f"DofHandler missed some nodes! Expected {len(expected_data)}, found {len(dirichlet_data)}."
    
    # Check if the values are correct for the nodes that were found
    for dof, value in expected_data.items():
        assert dof in dirichlet_data, f"DofHandler missed DOF {dof} on the boundary."
        assert np.isclose(dirichlet_data[dof], value), \
            f"Incorrect value for DOF {dof}. Expected {value}, got {dirichlet_data[dof]}."
            
    print("\nTest passed: DofHandler correctly identified all Q2 nodes and values on the boundary.")


def test_q2_node_ordering():
    """
    A diagnostic test to verify that the mesh element's node ordering
    matches the reference element's basis function ordering for Q2 elements.
    """
    print("\n" + "="*70)
    print("Running Q2 Node Ordering Diagnostic Test")
    print("="*70)

    # 1. Create a single Q2 element
    poly_order = 2
    nodes, elems, _, corners = structured_quad(1, 1, nx=1, ny=1, poly_order=poly_order)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=poly_order)
    element = mesh.elements_list[0]
    
    # 2. Get the reference element for Q2 quads
    ref_q2 = get_reference("quad", poly_order)

    # --- NEW: Manually generate the reference node coordinates ---
    # Your `quad_qn` function builds the 2D basis from a tensor product of 1D
    # Lagrange polynomials. The nodes for these are on np.linspace(-1, 1, ...).
    # The loops `for j in eta... for i in xi...` imply a row-major ordering.
    # We must replicate that same ordering here.
    nodes_1d = np.linspace(-1, 1, poly_order + 1)
    # The order 'C' ensures row-major (xi varies fastest), matching the loops.
    ref_coords = np.array(list(product(nodes_1d, nodes_1d)))
    # The output of product is (xi,eta), but the loop is (eta,xi), so we swap them
    ref_coords = ref_coords[:, [1,0]]


    # 3. The Litmus Test
    print("Element's node list from mesh generator:", element.nodes)
    print("Canonical reference node coordinates (xi, eta) generated for test:")
    print(ref_coords)
    print("-" * 70)
    print("Testing interpolation at each local node position...")
    
    all_passed = True
    for local_node_idx in range(len(element.nodes)):
        # Get the canonical coordinate for the i-th basis function
        ref_coord_for_basis_i = tuple(ref_coords[local_node_idx])

        # Create a nodal vector with only this basis function activated
        u_local = np.zeros(len(element.nodes))
        u_local[local_node_idx] = 1.0
        
        # Get all basis function values evaluated at this specific coordinate
        N_at_coord = ref_q2.shape(*ref_coord_for_basis_i)
        
        # Interpolate: N @ u should pick out the 1.0 from u_local
        # This checks if the i-th basis function is 1 at the i-th node's location.
        interpolated_value = N_at_coord @ u_local

        print(f"Test {local_node_idx}: Is basis function N_{local_node_idx} equal to 1 at ref_coord {ref_coord_for_basis_i}? -> {interpolated_value:.4f}")

        if not np.isclose(interpolated_value, 1.0):
            print(f"  └──> !!! FAILED !!! Expected 1.0. The ordering for local node {local_node_idx} is mismatched.")
            all_passed = False

    print("-" * 70)

    assert all_passed, "Node ordering mismatch detected! The mesh node order does not match the reference element's basis function order."
    print("Node ordering test passed successfully!")








def test_vector_advection_diffusion():
    """
    Tests the assembly for a vector advection-diffusion problem using
    the final, robust Function and VectorFunction classes.
    """
    print("\n" + "="*70)
    print("Testing Vector Advection-Diffusion with Manufactured Solution")
    print("="*70)

    # 1. Define Analytical Solution and Parameters using SymPy
    epsilon = 0.1
    beta_sym = sp.Matrix([1.0, 1.0])

    u_exact_sym_x = sp.sin(2 * sp.pi * x) * sp.cos(sp.pi * y)
    u_exact_sym_y = sp.cos(sp.pi * x) * sp.sin(2 * sp.pi * y)
    
    f_sym_x = -epsilon * (sp.diff(u_exact_sym_x, x, 2) + sp.diff(u_exact_sym_x, y, 2)) + \
              (beta_sym[0] * sp.diff(u_exact_sym_x, x) + beta_sym[1] * sp.diff(u_exact_sym_x, y))
    f_sym_y = -epsilon * (sp.diff(u_exact_sym_y, x, 2) + sp.diff(u_exact_sym_y, y, 2)) + \
              (beta_sym[0] * sp.diff(u_exact_sym_y, x) + beta_sym[1] * sp.diff(u_exact_sym_y, y))

    u_exact_func_x = sp.lambdify((x, y), u_exact_sym_x, 'numpy')
    u_exact_func_y = sp.lambdify((x, y), u_exact_sym_y, 'numpy')
    f_x = Analytic(f_sym_x)
    f_y = Analytic(f_sym_y)
    beta_analytic_func = sp.lambdify((x, y), beta_sym, 'numpy')

    # 2. Setup Mesh and DofHandler
    poly_order = 2  # Test with Q2 elements where the error occurred
    nodes, elems, _, corners = structured_quad(1, 1, nx=16, ny=16, poly_order=poly_order)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=poly_order)
    me = MixedElement(mesh, field_specs={'ux': poly_order, 'uy': poly_order})
    dof_handler = DofHandler(me, method='cg')

    # 3. Define the Weak Form
    # --- UPDATED FUNCTION INITIALIZATION ---
    u = VectorTrialFunction(FunctionSpace("velocity", ['ux', 'uy']))
    v = VectorTestFunction(FunctionSpace("velocity", ['ux', 'uy']))
    
    # Create beta and populate it using the new, clean API
    beta = VectorFunction(name="beta", field_names=['ux', 'uy'], dof_handler=dof_handler)
    beta.set_values_from_function(beta_analytic_func)

    # --- Weak form definition remains the same ---
    diffusion = epsilon * inner(grad(u), grad(v))
    advection = dot( dot(grad(u), beta), v) 
    a_form = (diffusion + advection) * dx()
    L_form = (f_x * v[0] + f_y * v[1]) * dx()
    equation = (a_form == L_form)

    # 4. Define Boundary Conditions
    mesh.tag_boundary_edges({'boundary': lambda x, y: True})
    bcs = [
        BoundaryCondition('ux', 'dirichlet', 'boundary', u_exact_func_x),
        BoundaryCondition('uy', 'dirichlet', 'boundary', u_exact_func_y)
    ]

    # 5. Assemble and Solve
    # We are in "Direct Evaluation Mode", no solution_vector needed.
    # The compiler will use the data stored inside the `beta` object.
    A, b = assemble_form(equation, dof_handler=dof_handler, bcs=bcs, quad_order=5)
    u_vec = spla.spsolve(A, b)

    # 6. Verify Solution
    ux_dofs = dof_handler.get_field_slice('ux')
    uy_dofs = dof_handler.get_field_slice('uy')
    
    node_coords = mesh.nodes_x_y_pos
    exact_sol_vec = np.zeros_like(u_vec)
    exact_sol_vec[ux_dofs] = u_exact_func_x(node_coords[:, 0], node_coords[:, 1])
    exact_sol_vec[uy_dofs] = u_exact_func_y(node_coords[:, 0], node_coords[:, 1])
    
    l2_error = np.linalg.norm(u_vec - exact_sol_vec) / np.linalg.norm(exact_sol_vec)
    
    print(f"\nRelative L2 Error for Q{poly_order} Vector Advection-Diffusion: {l2_error:.4e}")
    assert l2_error < 0.01, "Solution error is too high."
    print("Vector advection-diffusion test passed successfully!")