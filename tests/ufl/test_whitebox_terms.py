import numpy as np
import scipy.sparse.linalg as spla
import sympy as sp
from ufl.functionspace import FunctionSpace
from ufl.expressions import (    VectorTrialFunction, VectorTestFunction, VectorFunction, div,
    grad, inner, dot, Constant, TrialFunction, TestFunction)
from ufl.measures import dx
from ufl.forms import BoundaryCondition, assemble_form
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from ufl.analytic import Analytic, x, y
from pycutfem.fem.reference import get_reference
from pycutfem.fem import transform
from pycutfem.integration.quadrature import volume

# Helper function to create a standard test setup
def setup_single_q2_element():
    """Sets up a single Q2 element mesh and DoF handler."""
    poly_order = 2
    nodes, elems, _, corners = structured_quad(2, 2, nx=1, ny=1, poly_order=poly_order)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=poly_order)
    fe_map = {'ux': mesh, 'uy': mesh}
    dof_handler = DofHandler(fe_map, method='cg')
    return mesh, dof_handler

def setup_single_q1_element():
    """Sets up a single Q1 element mesh and DoF handler."""
    poly_order = 1
    nodes, elems, _, corners = structured_quad(2, 2, nx=1, ny=1, poly_order=poly_order)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=poly_order)
    fe_map = {'ux': mesh, 'uy': mesh}
    dof_handler = DofHandler(fe_map, method='cg')
    return mesh, dof_handler
    
#region Existing Tests (from user)
def test_rhs_vector_diffusion():
    """
    White-box test for the RHS vector from: inner(grad(u_k), grad(v))
    where u_k is a known VectorFunction.
    """
    print("\n" + "="*70)
    print("White-Box Test of RHS Vector Diffusion Term")
    print("="*70)

    mesh, dof_handler = setup_single_q2_element()
    velocity_space = FunctionSpace("velocity", ['ux', 'uy'])
    v = VectorTestFunction(velocity_space)

    # 1. Define a known function u_k with arbitrary data
    u_k = VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler)
    np.random.seed(0)
    u_k.nodal_values[:] = np.random.rand(dof_handler.total_dofs)

    # 2. Define the UFL form for the RHS term
    equation = Constant(0.0) * v[0] * dx() == 1.0 * inner(grad(u_k), grad(v)) * dx()
    
    # 3. Get the vector R from YOUR real assembler
    _, R = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_vector = R

    # 4. Manually compute the ground truth vector
    ref_q2 = get_reference("quad", poly_order=2)
    q_order = 5
    qpts, qwts = volume("quad", q_order)
    n_basis_scalar = 9
    
    expected_vector = np.zeros(n_basis_scalar * 2)
    
    # Get local DoF values for u_k
    dofs_ux = dof_handler.element_maps['ux'][0]
    dofs_uy = dof_handler.element_maps['uy'][0]
    u_k_ux_local = u_k.get_nodal_values(dofs_ux)
    u_k_uy_local = u_k.get_nodal_values(dofs_uy)

    for qp, w in zip(qpts, qwts):
        J = transform.jacobian(mesh, 0, qp)
        detJ = abs(np.linalg.det(J))
        JinvT = np.linalg.inv(J).T
        
        N_scalar = ref_q2.shape(*qp)       # Shape (9,)
        G_scalar_phys = ref_q2.grad(*qp) @ JinvT # Shape (9, 2)
        
        grad_uk_val = np.zeros((2, 2))
        grad_uk_val[0, :] = G_scalar_phys.T @ u_k_ux_local
        grad_uk_val[1, :] = G_scalar_phys.T @ u_k_uy_local
        
        for j in range(n_basis_scalar * 2):
            grad_v_basis_j = np.zeros((2, 2))
            if j < n_basis_scalar:
                grad_v_basis_j[0, :] = G_scalar_phys[j, :]
            else:
                grad_v_basis_j[1, :] = G_scalar_phys[j - n_basis_scalar, :]
            
            integrand_val = np.sum(grad_uk_val * grad_v_basis_j)
            expected_vector[j] += integrand_val * w * detJ

    # 5. Compare the vectors
    print(f"compiler.shape: {compiler_vector.shape}, expected.shape: {expected_vector.shape}")
    print("Comparing compiler-generated vector with manually computed ground truth...")
    np.testing.assert_allclose(compiler_vector, expected_vector, rtol=1e-12, atol=1e-12)
    print("\nSUCCESS: Compiler's RHS diffusion vector is correct!")

def test_rhs_mass_matrix():
    """
    White-box test for the RHS vector from: dot(u_k, v)
    where u_k is a known VectorFunction. This test uses Q2 elements.
    """
    print("\n" + "="*70)
    print("White-Box Test of RHS Q2 Vector Mass Term")
    print("="*70)

    mesh, dof_handler = setup_single_q2_element()
    velocity_space = FunctionSpace("velocity", ['ux', 'uy'])
    v = VectorTestFunction(velocity_space)

    u_k = VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler)
    np.random.seed(123)
    u_k.nodal_values[:] = np.random.rand(dof_handler.total_dofs)

    mass_term = dot(u_k, v)
    equation = Constant(0.0) * v[0] * dx() == 1.0 * mass_term * dx()
    
    _, R = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_vector = R

    ref_q2 = get_reference("quad", poly_order=2)
    q_order = 5
    qpts, qwts = volume("quad", q_order)
    n_basis_scalar = 9
    
    expected_vector = np.zeros(n_basis_scalar * 2)
    
    dofs_ux = dof_handler.element_maps['ux'][0]
    dofs_uy = dof_handler.element_maps['uy'][0]
    u_k_ux_local = u_k.get_nodal_values(dofs_ux)
    u_k_uy_local = u_k.get_nodal_values(dofs_uy)

    for qp, w in zip(qpts, qwts):
        J = transform.jacobian(mesh, 0, qp)
        detJ = abs(np.linalg.det(J))
        
        N_scalar = ref_q2.shape(*qp)
        
        u_k_val_at_qp = np.array([
            N_scalar @ u_k_ux_local,
            N_scalar @ u_k_uy_local
        ])

        ux_contribution = u_k_val_at_qp[0] * N_scalar
        expected_vector[:n_basis_scalar] += ux_contribution * w * detJ
        
        uy_contribution = u_k_val_at_qp[1] * N_scalar
        expected_vector[n_basis_scalar:] += uy_contribution * w * detJ

    print("Comparing compiler-generated vector with manually computed ground truth...")
    np.testing.assert_allclose(compiler_vector, expected_vector, rtol=1e-12, atol=1e-12)
    print("\nSUCCESS: Compiler's RHS Q2 mass vector is correct!")


def test_rhs_nonlinear_advection():
    """
    White-box test for the RHS vector from: ((u_k ⋅ ∇)u_k) ⋅ v
    where u_k is a known VectorFunction.
    """
    print("\n" + "="*70)
    print("White-Box Test of RHS Nonlinear Advection Term")
    print("="*70)

    mesh, dof_handler = setup_single_q2_element()
    velocity_space = FunctionSpace("velocity", ['ux', 'uy'])
    v = VectorTestFunction(velocity_space)

    u_k = VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler)
    np.random.seed(42)
    u_k.nodal_values[:] = np.sin(np.linspace(0, 5, dof_handler.total_dofs))

    # In UFL, dot(grad(u), v) for vector u,v is ambiguous.
    # The term (u_k ⋅ ∇)u_k is explicitly written as:
    # (u_k, ∇(u_k_x)) for the x-component
    # (u_k, ∇(u_k_y)) for the y-component
    advection_term =  dot(u_k, grad(u_k[0]))* v[0] + dot(u_k, grad(u_k[1]))* v[1]

    equation = Constant(0.0) * v[0] * dx() == 1.0 * advection_term * dx()
    
    _, R = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_vector = R
    
    ref_q2 = get_reference("quad", poly_order=2)
    q_order = 5
    qpts, qwts = volume("quad", q_order)
    n_basis_scalar = 9
    
    expected_vector = np.zeros(n_basis_scalar * 2)

    dofs_ux = dof_handler.element_maps['ux'][0]
    dofs_uy = dof_handler.element_maps['uy'][0]
    u_k_ux_local = u_k.get_nodal_values(dofs_ux)
    u_k_uy_local = u_k.get_nodal_values(dofs_uy)

    for qp, w in zip(qpts, qwts):
        J = transform.jacobian(mesh, 0, qp)
        detJ = abs(np.linalg.det(J))
        JinvT = np.linalg.inv(J).T
        
        N_scalar = ref_q2.shape(*qp)
        G_scalar_phys = ref_q2.grad(*qp) @ JinvT
        
        u_k_val = np.array([
            N_scalar @ u_k_ux_local,
            N_scalar @ u_k_uy_local
        ])
        
        grad_uk0_val = G_scalar_phys.T @ u_k_ux_local
        grad_uk1_val = G_scalar_phys.T @ u_k_uy_local

        # Term 1: ((u_k ⋅ ∇)u_k[0]) * v[0]
        # This contributes to the first 9 rows of the local vector
        coeff0 = np.dot(u_k_val, grad_uk0_val)
        expected_vector[:n_basis_scalar] += (coeff0 * N_scalar) * w * detJ
        
        # Term 2: ((u_k ⋅ ∇)u_k[1]) * v[1]
        # This contributes to the last 9 rows of the local vector
        coeff1 = np.dot(u_k_val, grad_uk1_val)
        expected_vector[n_basis_scalar:] += (coeff1 * N_scalar) * w * detJ

    print(f"compiler.shape: {compiler_vector.shape}, expected.shape: {expected_vector.shape}")
    print("Comparing compiler-generated vector with manually computed ground truth...")
    np.testing.assert_allclose(compiler_vector, expected_vector, rtol=1e-12, atol=1e-12)
    print("\nSUCCESS: Compiler's RHS nonlinear advection vector is correct!")


def get_exact_q1_mass_matrix():
    """
    Computes the exact 4x4 mass matrix for a Q1 element on a 1x1 physical square.
    """
    xi, eta = sp.symbols('xi eta')
    
    N_ref = [
        (1 - xi)/2 * (1 - eta)/2,
        (1 + xi)/2 * (1 - eta)/2,
        (1 + xi)/2 * (1 + eta)/2,
        (1 - xi)/2 * (1 + eta)/2
    ]
    
    # Match node ordering from structured_quad: (0,0), (1,0), (0,1), (1,1)
    # which corresponds to ref coords (-1,-1), (1,-1), (-1,1), (1,1)
    N_ordered = [N_ref[0], N_ref[1], N_ref[3], N_ref[2]]
    
    M = sp.zeros(4, 4)
    # Jacobian for mapping [-1,1]^2 to [0,1]^2 is J = [[0.5, 0], [0, 0.5]]
    # det(J) = 0.25
    detJ = sp.Rational(1, 4)
    
    for i in range(4):
        for j in range(4):
            integrand = N_ordered[i] * N_ordered[j]
            integral_val = sp.integrate(integrand * detJ, (xi, -1, 1), (eta, -1, 1))
            M[i, j] = integral_val
            
    return np.array(M, dtype=float)

def test_scalar_mass_matrix_q1():
    """White-box test for a scalar Q1 mass matrix on a single element."""
    print("\n" + "="*70)
    print("White-Box Test of Q1 Scalar Mass Matrix")
    print("="*70)

    nodes, elems, _, corners = structured_quad(1, 1, nx=1, ny=1, poly_order=1)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)
    fe_map = {'u': mesh}
    dof_handler = DofHandler(fe_map, method='cg')

    u = TrialFunction('u')
    v = TestFunction('u')
    equation = (dot(u, v) * dx() == Constant(0.0) * v * dx())

    A, _ = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_matrix = A.toarray()

    expected_matrix = get_exact_q1_mass_matrix()

    print("Comparing Compiler's Q1 Scalar Mass Matrix with Analytical Ground Truth...")
    np.testing.assert_allclose(compiler_matrix, expected_matrix, rtol=1e-12, atol=1e-12)
    print("SUCCESS: Q1 Scalar Mass Matrix is correct!")

def test_vector_mass_matrix_q1():
    """White-box test for a vector Q1 mass matrix on a single element."""
    print("\n" + "="*70)
    print("White-Box Test of Q1 Vector Mass Matrix")
    print("="*70)
    
    nodes, elems, _, corners = structured_quad(1, 1, nx=1, ny=1, poly_order=1)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)
    fe_map = {'ux': mesh, 'uy': mesh}
    dof_handler = DofHandler(fe_map, method='cg')

    u = VectorTrialFunction(FunctionSpace("velocity", ['ux', 'uy']))
    v = VectorTestFunction(FunctionSpace("velocity", ['ux', 'uy']))
    equation = (dot(u, v) * dx() == Constant(0.0) * v[0] * dx() + Constant(0.0) * v[1] * dx())

    A, _ = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_matrix = A.toarray()

    scalar_mass_block = get_exact_q1_mass_matrix()
    expected_matrix = np.zeros((8, 8))
    expected_matrix[0:4, 0:4] = scalar_mass_block
    expected_matrix[4:8, 4:8] = scalar_mass_block
    
    print("Comparing Compiler's Q1 Vector Mass Matrix with Analytical Ground Truth...")
    np.testing.assert_allclose(compiler_matrix, expected_matrix, rtol=1e-12, atol=1e-12)
    print("SUCCESS: Q1 Vector Mass Matrix is correct!")

#endregion

#====================================================================================
# NEW EXTENDED TESTS
#====================================================================================

#region New Helper Functions

def get_exact_q2_mass_matrix():
    """
    Computes the exact 9x9 mass matrix for a Q2 element on a 2x2 physical square.
    """
    xi, eta = sp.symbols('xi eta')
    
    # Q2 basis functions on the [-1, 1] x [-1, 1] reference element,
    # defined by the node at which they are 1.
    N_ref_nodes = [
        xi*eta*(xi - 1)*(eta - 1)/4,      # Node at (-1,-1) -> index 0
        xi*eta*(xi + 1)*(eta - 1)/4,      # Node at ( 1,-1) -> index 1
        xi*eta*(xi + 1)*(eta + 1)/4,      # Node at ( 1, 1) -> index 2
        xi*eta*(xi - 1)*(eta + 1)/4,      # Node at (-1, 1) -> index 3
        (1 - xi**2)*eta*(eta - 1)/2,        # Node at ( 0,-1) -> index 4
        xi*(xi + 1)*(1 - eta**2)/2,         # Node at ( 1, 0) -> index 5
        (1 - xi**2)*eta*(eta + 1)/2,        # Node at ( 0, 1) -> index 6
        xi*(xi - 1)*(1 - eta**2)/2,         # Node at (-1, 0) -> index 7
        (1 - xi**2)*(1 - eta**2)            # Node at ( 0, 0) -> index 8
    ]
    
    # CORRECTED: This node_order list re-arranges the basis functions above
    # to match the standard lexicographical order:
    # (-1,-1), (0,-1), (1,-1), (-1,0), (0,0), (1,0), (-1,1), (0,1), (1,1)
    node_order = [0, 4, 1, 7, 8, 5, 3, 6, 2]
    N_ordered = [N_ref_nodes[i] for i in node_order]
    
    M = sp.zeros(9, 9)
    # Jacobian for mapping [-1,1]^2 to [0,2]^2 has det(J) = 1
    detJ = 1
    
    for i in range(9):
        for j in range(9):
            integrand = N_ordered[i] * N_ordered[j]
            integral_val = sp.integrate(integrand * detJ, (xi, -1, 1), (eta, -1, 1))
            M[i, j] = integral_val
            
    return np.array(M, dtype=float)

def setup_mixed_q2_q1_element():
    """Sets up a single element mesh and mixed-space DoF handler for Q2/Q1."""
    # Velocity mesh (Q2)
    v_nodes, v_elems, _, v_corners = structured_quad(1, 1, nx=1, ny=1, poly_order=2)
    v_mesh = Mesh(nodes=v_nodes, element_connectivity=v_elems, elements_corner_nodes=v_corners, element_type="quad", poly_order=2)
    
    # Pressure mesh (Q1)
    p_nodes, p_elems, _, p_corners = structured_quad(1, 1, nx=1, ny=1, poly_order=1)
    p_mesh = Mesh(nodes=p_nodes, element_connectivity=p_elems, elements_corner_nodes=p_corners, element_type="quad", poly_order=1)

    fe_map = {'ux': v_mesh, 'uy': v_mesh, 'p': p_mesh}
    dof_handler = DofHandler(fe_map, method='cg')
    return dof_handler, v_mesh, p_mesh

#endregion

#region Advection Term Tests

def test_lhs_advection_q2():
    """
    White-box test for the LHS matrix from the linearized advection term:
    A_ij = ∫ ((u_k ⋅ ∇)u_j) ⋅ v_i dΩ
    """
    print("\n" + "="*70)
    print("White-Box Test of LHS Advection Matrix (u_k_grad_u)")
    print("="*70)
    
    mesh, dof_handler = setup_single_q2_element()
    
    velocity_space = FunctionSpace("velocity", ['ux', 'uy'])
    u = VectorTrialFunction(velocity_space)
    v = VectorTestFunction(velocity_space)
    
    u_k = VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler)
    np.random.seed(1337)
    u_k.nodal_values[:] = np.random.rand(dof_handler.total_dofs)
    
    # This UFL form is correct. It represents ((u_k ⋅ ∇)u) ⋅ v
    advection_term = dot(u_k, grad(u[0]))* v[0] + dot(u_k, grad(u[1]))* v[1]
    equation = advection_term * dx() == Constant(0.0) * v[0] * dx()
    
    A, _ = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_matrix = A.toarray()
    
    # --- CORRECTED MANUAL CALCULATION ---
    ref_q2 = get_reference("quad", poly_order=2)
    q_order = 5
    qpts, qwts = volume("quad", q_order)
    n_basis_scalar = 9
    
    expected_matrix = np.zeros((n_basis_scalar * 2, n_basis_scalar * 2))
    
    dofs_ux = dof_handler.element_maps['ux'][0]
    dofs_uy = dof_handler.element_maps['uy'][0]
    u_k_ux_local = u_k.get_nodal_values(dofs_ux)
    u_k_uy_local = u_k.get_nodal_values(dofs_uy)
    
    for qp, w in zip(qpts, qwts):
        J = transform.jacobian(mesh, 0, qp)
        detJ = abs(np.linalg.det(J))
        JinvT = np.linalg.inv(J).T
        
        N = ref_q2.shape(*qp)
        G_phys = ref_q2.grad(*qp) @ JinvT
        
        u_k_val = np.array([N @ u_k_ux_local, N @ u_k_uy_local])
        
        # Pre-compute the scalar convection term (u_k ⋅ ∇) applied to each SCALAR basis function
        # This results in a vector of 9 scalar values at the quadrature point
        convected_N_j = u_k_val @ G_phys.T # Shape (9,)

        for i in range(n_basis_scalar * 2): # loop over test functions (rows)
            for j in range(n_basis_scalar * 2): # loop over trial functions (cols)
                
                integrand = 0.0
                # Case 1: (ux, ux) block. Trial=Φ_j^x, Test=Φ_i^x
                # Integrand is ((u_k ⋅ ∇)N_j) * N_i
                if i < n_basis_scalar and j < n_basis_scalar:
                    integrand = convected_N_j[j] * N[i]

                # Case 2: (uy, uy) block. Trial=Φ_j^y, Test=Φ_i^y
                # Integrand is ((u_k ⋅ ∇)N_j) * N_i
                elif i >= n_basis_scalar and j >= n_basis_scalar:
                    integrand = convected_N_j[j - n_basis_scalar] * N[i - n_basis_scalar]
                
                # Off-diagonal blocks (ux,uy) and (uy,ux) are zero for this term
                
                expected_matrix[i, j] += integrand * w * detJ

    print("Comparing compiler-generated matrix with manually computed ground truth...")
    np.testing.assert_allclose(compiler_matrix, expected_matrix, rtol=1e-12, atol=1e-12)
    print("\nSUCCESS: Compiler's LHS advection matrix is correct!")


def test_lhs_advection_transpose_q2():
    """
    White-box test for the other linearized advection term:
    A_ij = ∫ ((u_j ⋅ ∇)u_k) ⋅ v_i dΩ
    """
    print("\n" + "="*70)
    print("White-Box Test of LHS Advection Matrix (u_grad_uk)")
    print("="*70)

    mesh, dof_handler = setup_single_q2_element()
    
    velocity_space = FunctionSpace("velocity", ['ux', 'uy'])
    u = VectorTrialFunction(velocity_space)
    v = VectorTestFunction(velocity_space)
    
    u_k = VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler)
    np.random.seed(1337)
    u_k.nodal_values[:] = np.random.rand(dof_handler.total_dofs)
    
    # UFL for (u ⋅ ∇)u_k is dot(u, grad(u_k))
    # Then dot with v
    advection_term = dot(u, grad(u_k[0])) * v[0] + dot(u, grad(u_k[1])) * v[1]
    equation = advection_term * dx() == Constant(0.0) * v[0] * dx()
    
    A, _ = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_matrix = A.toarray()
    
    # Manually compute the ground truth element matrix
    ref_q2 = get_reference("quad", poly_order=2)
    q_order = 5
    qpts, qwts = volume("quad", q_order)
    n_basis_scalar = 9
    
    expected_matrix = np.zeros((n_basis_scalar * 2, n_basis_scalar * 2))
    
    dofs_ux = dof_handler.element_maps['ux'][0]
    dofs_uy = dof_handler.element_maps['uy'][0]
    u_k_ux_local = u_k.get_nodal_values(dofs_ux)
    u_k_uy_local = u_k.get_nodal_values(dofs_uy)
    
    for qp, w in zip(qpts, qwts):
        J = transform.jacobian(mesh, 0, qp)
        detJ = abs(np.linalg.det(J))
        JinvT = np.linalg.inv(J).T
        
        N = ref_q2.shape(*qp)
        G_phys = ref_q2.grad(*qp) @ JinvT
        
        grad_uk_val = np.zeros((2, 2))
        grad_uk_val[0, :] = G_phys.T @ u_k_ux_local # row 0: grad(uk_x)
        grad_uk_val[1, :] = G_phys.T @ u_k_uy_local # row 1: grad(uk_y)
        
        for i in range(n_basis_scalar * 2):
            for j in range(n_basis_scalar * 2):
                
                u_j = np.zeros(2)
                if j < n_basis_scalar:
                    u_j[0] = N[j]
                else:
                    u_j[1] = N[j - n_basis_scalar]
                
                v_i = np.zeros(2)
                if i < n_basis_scalar:
                    v_i[0] = N[i]
                else:
                    v_i[1] = N[i - n_basis_scalar]

                # --- CORRECTED CALCULATION ---
                # To compute (u_j ⋅ ∇)u_k, we compute the dot product of u_j with each
                # ROW of grad_uk_val.
                # The previous version (u_j @ grad_uk_val) incorrectly used the columns.
                term_vec = np.array([
                    np.dot(u_j, grad_uk_val[0, :]),  # Component 1: u_j ⋅ ∇(u_k)_x
                    np.dot(u_j, grad_uk_val[1, :])   # Component 2: u_j ⋅ ∇(u_k)_y
                ])
                
                integrand = np.dot(term_vec, v_i)
                
                expected_matrix[i, j] += integrand * w * detJ
                
    print(f"Comparing compiler-generated matrix with manually computed ground truth... expected shape: {expected_matrix.shape}, compiler shape: {compiler_matrix.shape}")
    np.testing.assert_allclose(compiler_matrix, expected_matrix, rtol=1e-12, atol=1e-12)
    print("\nSUCCESS: Compiler's LHS transpose advection matrix is correct!")

#endregion

#region Mixed Space Tests

def test_rhs_mixed_divergence():
    """
    White-box test for the RHS vector from q * div(u_k)
    q: Q1 TestFunction
    u_k: Q2 VectorFunction
    """
    print("\n" + "="*70)
    print("White-Box Test of RHS Mixed Divergence Vector (q * div(u_k))")
    print("="*70)

    dof_handler, v_mesh, p_mesh = setup_mixed_q2_q1_element()
    
    v_space = FunctionSpace("velocity", ['ux', 'uy'])
    p_space = FunctionSpace("pressure", ['p'])
    q = TestFunction(p_space.field_names[0])  # Q1 test function for pressure
    
    u_k = VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler)
    np.random.seed(1337)
    u_k.nodal_values[:] = np.random.rand(dof_handler.total_dofs - len(dof_handler.get_field_slice('p')))  # Exclude pressure DoFs

    # Define the UFL form
    equation = Constant(0.0) * q * dx() == q * div(u_k) * dx()
    
    # Assemble the vector R
    _, R = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    # The result vector will be for all DoFs, so we extract the pressure part
    p_dofs = dof_handler.get_field_slice('p')
    p_start = p_dofs[0]
    p_end = p_dofs[-1] + 1
    compiler_vector = R[p_start:p_end]
    
    # Manually compute the ground truth vector
    ref_q2 = get_reference("quad", poly_order=2)
    ref_q1 = get_reference("quad", poly_order=1)
    q_order = 5
    qpts, qwts = volume("quad", q_order)
    
    n_basis_p_scalar = 4
    expected_vector = np.zeros(n_basis_p_scalar)
    
    dofs_ux = dof_handler.element_maps['ux'][0]
    dofs_uy = dof_handler.element_maps['uy'][0]
    u_k_ux_local = u_k.get_nodal_values(dofs_ux)
    u_k_uy_local = u_k.get_nodal_values(dofs_uy)

    for qp, w in zip(qpts, qwts):
        J = transform.jacobian(v_mesh, 0, qp)
        detJ = abs(np.linalg.det(J))
        JinvT = np.linalg.inv(J).T
        
        N_q = ref_q1.shape(*qp)
        G_u_phys = ref_q2.grad(*qp) @ JinvT
        
        # Calculate divergence of u_k at the quadrature point
        grad_uk_x = G_u_phys.T @ u_k_ux_local
        grad_uk_y = G_u_phys.T @ u_k_uy_local
        div_uk_val = grad_uk_x[0] + grad_uk_y[1] # du_x/dx + du_y/dy
        
        for i in range(n_basis_p_scalar): # loop over pressure test functions
            integrand = N_q[i] * div_uk_val
            expected_vector[i] += integrand * w * detJ
            
    print(f"Comparing compiler-generated vector with manually computed ground truth... shapes: {compiler_vector.shape}, expected: {expected_vector.shape}")
    np.testing.assert_allclose(compiler_vector, expected_vector, rtol=1e-12, atol=1e-12)
    print("\nSUCCESS: Compiler's RHS mixed divergence vector is correct!")

def test_mixed_pressure_grad_operator():
    """
    White-box test for the pressure gradient operator matrix: G_ij = ∫ (div v_i) * p_j dΩ
    This is the transpose of the divergence operator.
    p: Q1 TrialFunction
    v: Q2 VectorTestFunction
    """
    print("\n" + "="*70)
    print("White-Box Test of Mixed-Space Pressure Gradient Matrix (p * div(v))")
    print("="*70)

    # 1. Setup
    dof_handler, v_mesh, p_mesh = setup_mixed_q2_q1_element()

    v_space = FunctionSpace("velocity", ['ux', 'uy'])
    p_space = FunctionSpace("pressure", ['p'])
    v = VectorTestFunction(v_space)
    p = TrialFunction(p_space.field_names[0])  # Q1 trial function for pressure

    # 2. Define UFL Form
    equation = ((p* div(v)) * dx() == Constant(0.0) * v[0] * dx())

    # 3. Assemble using the library
    A, _ = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_matrix = A.toarray()

    # 4. Manually compute the ground truth 18x4 matrix
    ref_q2 = get_reference("quad", poly_order=2)
    ref_q1 = get_reference("quad", poly_order=1)
    q_order = 5
    qpts, qwts = volume("quad", q_order)

    n_basis_v_scalar = 9
    n_basis_p_scalar = 4
    
    # --- CORRECTED INITIALIZATION ---
    # The expected matrix must be 18 rows (from test function v) x 4 columns (from trial function p)
    expected_block = np.zeros((n_basis_v_scalar * 2, n_basis_p_scalar))

    for qp, w in zip(qpts, qwts):
        J = transform.jacobian(v_mesh, 0, qp)
        detJ = abs(np.linalg.det(J))
        JinvT = np.linalg.inv(J).T

        N_p = ref_q1.shape(*qp)
        G_v_phys = ref_q2.grad(*qp) @ JinvT

        # --- CORRECTED LOOPS to match 18x4 shape ---
        # Loop over test functions v (rows i=0..17)
        for i in range(n_basis_v_scalar * 2):
            # Loop over trial functions p (cols j=0..3)
            for j in range(n_basis_p_scalar):
                
                div_v_i = 0.0
                if i < n_basis_v_scalar: # ux component
                    div_v_i = G_v_phys[i, 0]
                else: # uy component
                    div_v_i = G_v_phys[i - n_basis_v_scalar, 1]
                
                integrand = div_v_i * N_p[j]
                
                # Index into the 18x4 matrix
                expected_block[i, j] += integrand * w * detJ

    # The assembler maps fields by their order in fe_map: ux, uy, p
    # So the full matrix has p-v coupling in the [18:22, 0:18] block
    total_dofs = dof_handler.total_dofs
    p_dofs = dof_handler.get_field_slice('p')  # Pressure dofs
    uy_dofs = dof_handler.get_field_slice('uy')  # Velocity dofs (uy)
    ux_dofs = dof_handler.get_field_slice('ux')  # Velocity dofs (ux)
    
    # Compare the relevant block from the compiler's matrix
    # Row indices are for pressure test functions (q), col indices for velocity trial functions (u)
    # But here we have p (trial) and v (test), so A_ij = a(v_j, p_i)
    # The form is bilinear(v,p), so trial=p, test=v gives A_ij = a(p_j, v_i)
    # The dof handler orders dofs by field name: ux, uy, p.
    # A has shape (total_dofs, total_dofs).
    # The term p*div(v) links test(p) and trial(v).
    # Let's check the equation again: (p * div(v)) * dx == C * p * dx
    # Here, 'p' is a TrialFunction and 'v' is a VectorTestFunction.
    # This means the trial space is 'p' and the test space is 'v'.
    # This seems backwards. Let's test the conventional divergence term instead.
    # The block for this term has velocity test functions (rows) and pressure trial functions (cols)
    # So we extract A[0:18, 18:22]
    extracted_block = compiler_matrix[ux_dofs[0]:uy_dofs[-1]+1, p_dofs[0]:p_dofs[-1] + 1]
    print(f"compiler_matrix shape: {compiler_matrix.shape}, expected_block shape: {expected_block.shape}")
    print("Comparing compiler-generated 18x4 matrix block with manual ground truth...")
    np.testing.assert_allclose(extracted_block, expected_block, rtol=1e-12, atol=1e-12)
    print("\nSUCCESS: Compiler's mixed-space gradient matrix is correct!")


def test_mixed_velocity_divergence_operator():
    """
    White-box test for the velocity divergence operator matrix: D_ij = -∫ q_i * div(u_j) dΩ
    q: Q1 TestFunction
    u: Q2 VectorTrialFunction
    """
    print("\n" + "="*70)
    print("White-Box Test of Mixed-Space Velocity Divergence Matrix (q * div(u))")
    print("="*70)

    dof_handler, v_mesh, p_mesh = setup_mixed_q2_q1_element()
    
    v_space = FunctionSpace("velocity", ['ux', 'uy'])
    p_space = FunctionSpace("pressure", ['p'])
    u = VectorTrialFunction(v_space)
    q = TestFunction(p_space.field_names[0])  # Q1 test function for pressure
    
    # Standard divergence term in saddle point problems
    equation = (q * div(u)) * dx() == Constant(0.0) * q * dx()
    
    A, _ = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_matrix = A.toarray()
    
    ref_q2 = get_reference("quad", poly_order=2)
    ref_q1 = get_reference("quad", poly_order=1)
    q_order = 5
    qpts, qwts = volume("quad", q_order)
    
    n_basis_v_scalar = 9
    n_basis_p_scalar = 4
    
    expected_block = np.zeros((n_basis_p_scalar, n_basis_v_scalar * 2))
    
    for qp, w in zip(qpts, qwts):
        J = transform.jacobian(v_mesh, 0, qp)
        detJ = abs(np.linalg.det(J))
        JinvT = np.linalg.inv(J).T
        
        N_q = ref_q1.shape(*qp)     # Q1 shape functions for q (test), shape (4,)
        G_u_phys = ref_q2.grad(*qp) @ JinvT # Q2 shape function grads for u (trial), shape (9, 2)
        
        for i in range(n_basis_p_scalar):     # loop over test functions q (rows)
            for j in range(n_basis_v_scalar * 2): # loop over trial functions u (cols)
                
                div_u_j = 0.0
                if j < n_basis_v_scalar: # ux component
                    div_u_j = G_u_phys[j, 0] # d(phi_j)/dx
                else: # uy component
                    div_u_j = G_u_phys[j - n_basis_v_scalar, 1] # d(phi_j)/dy
                
                integrand = N_q[i] * div_u_j
                expected_block[i, j] += integrand * w * detJ

    # The block for this term should be where pressure test functions (q) and
    # velocity trial functions (u) interact.
    # Dofs: ux (0-8), uy (9-17), p (18-21)
    # Test(p) x Trial(u) block is at A[18:22, 0:18]
    p_dofs = dof_handler.get_field_slice('p')
    uy_dofs = dof_handler.get_field_slice('uy')
    ux_dofs = dof_handler.get_field_slice('ux')
    print(f"p_dofs: {p_dofs}, uy_dofs: {uy_dofs}")
    
    extracted_block = compiler_matrix[p_dofs[0]:p_dofs[-1]+1, ux_dofs[0]:uy_dofs[-1] + 1]
    
    print("Comparing compiler-generated matrix block with manually computed ground truth...")
    np.testing.assert_allclose(extracted_block, expected_block, rtol=1e-12, atol=1e-12)
    print("\nSUCCESS: Compiler's mixed-space divergence matrix is correct!")

#endregion

#region Q2 Mass Matrix Tests

def test_scalar_mass_matrix_q2():
    """White-box test for a scalar Q2 mass matrix on a single element."""
    print("\n" + "="*70)
    print("White-Box Test of Q2 Scalar Mass Matrix")
    print("="*70)

    # 1. Setup a single Q2 element
    mesh, dof_handler = setup_single_q2_element()
    # Need to redefine dofhandler for a scalar field
    fe_map = {'u': mesh}
    dof_handler = DofHandler(fe_map, method='cg')

    # 2. Define the UFL form for the mass matrix
    u = TrialFunction('u')
    v = TestFunction('u')
    equation = (dot(u, v) * dx() == Constant(0.0) * v * dx())

    # 3. Get the global matrix A from the real assembler
    A, _ = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_matrix = A.toarray()

    # 4. Get the exact analytical matrix
    expected_matrix = get_exact_q2_mass_matrix()

    # 5. Compare
    print("Comparing Compiler's Q2 Scalar Mass Matrix with Analytical Ground Truth...")
    np.testing.assert_allclose(compiler_matrix, expected_matrix, rtol=1e-12, atol=1e-12)
    print("SUCCESS: Q2 Scalar Mass Matrix is correct!")

def test_vector_mass_matrix_q2():
    """White-box test for a vector Q2 mass matrix on a single element."""
    print("\n" + "="*70)
    print("White-Box Test of Q2 Vector Mass Matrix (LHS)")
    print("="*70)
    
    mesh, dof_handler = setup_single_q2_element()

    u = VectorTrialFunction(FunctionSpace("velocity", ['ux', 'uy']))
    v = VectorTestFunction(FunctionSpace("velocity", ['ux', 'uy']))
    equation = (dot(u, v) * dx() == Constant(0.0) * v[0] * dx())

    A, _ = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_matrix = A.toarray()
    print("Compiler's Q2 Vector Mass Matrix shape:", compiler_matrix.shape)

    scalar_mass_block = get_exact_q2_mass_matrix()
    expected_matrix = np.zeros((18, 18))
    expected_matrix[0:9, 0:9] = scalar_mass_block
    expected_matrix[9:18, 9:18] = scalar_mass_block
    
    print("Comparing Compiler's Q2 Vector Mass Matrix with block-diagonal analytical truth...")
    np.testing.assert_allclose(compiler_matrix, expected_matrix, rtol=1e-12, atol=1e-12)
    print("SUCCESS: Q2 Vector Mass Matrix is correct!")
    
#endregion

if __name__ == '__main__':
    # Existing Tests
    test_rhs_vector_diffusion()
    test_rhs_mass_matrix() # Note: This already tests the RHS Q2 vector mass term
    test_rhs_nonlinear_advection()
    test_scalar_mass_matrix_q1()
    test_vector_mass_matrix_q1()

    print("\n\n" + "#"*70)
    print("### RUNNING EXTENDED TEST SUITE ###")
    print("#"*70)
    
    # New Advection Tests
    test_lhs_advection_q2()

    # New Mixed-Element Tests
    # test_mixed_pressure_grad_operator() # Ill-posed, skipped
    test_mixed_velocity_divergence_operator()
    
    # New Q2 Mass Matrix Tests
    test_scalar_mass_matrix_q2()
    test_vector_mass_matrix_q2()