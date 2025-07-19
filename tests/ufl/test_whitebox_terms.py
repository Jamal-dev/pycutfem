import numpy as np
import scipy.sparse.linalg as spla
import sympy as sp
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    VectorTrialFunction, VectorTestFunction, VectorFunction, div,
    grad, inner, dot, Constant, TrialFunction, TestFunction, Derivative
)
from pycutfem.ufl.measures import dx
from pycutfem.ufl.forms import BoundaryCondition, assemble_form
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.fem.reference import get_reference
from pycutfem.fem import transform
from pycutfem.integration.quadrature import volume
from pycutfem.fem.mixedelement import MixedElement


# Helper function to create a standard test setup
def setup_single_q2_element():
    """Sets up a single Q2 element mesh and a vector-field DoF handler."""
    poly_order = 2
    nodes, elems, _, corners = structured_quad(2, 2, nx=1, ny=1, poly_order=poly_order)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=poly_order)
    me = MixedElement(mesh, field_specs={'ux': poly_order, 'uy': poly_order})
    dof_handler = DofHandler(me, method='cg')
    return mesh, dof_handler

#region Existing Tests (Reviewed and Validated)

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

    u_k = VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler)
    np.random.seed(0)
    u_k.nodal_values[:] = np.random.rand(dof_handler.total_dofs)

    equation = dot(Constant([0.0,0.0],dim=1) , v) * dx() == inner(grad(u_k), grad(v)) * dx(metadata={'q': 4})
    
    _, R = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_vector = R

    ref_q2 = get_reference("quad", poly_order=2)
    q_order = 5
    qpts, qwts = volume("quad", q_order)
    n_basis_scalar = 9
    
    expected_vector = np.zeros(n_basis_scalar * 2)
    
    elemental_dofs = dof_handler.get_elemental_dofs(0)
    u_k_local = u_k.get_nodal_values(elemental_dofs)
    u_k_ux_local = u_k_local[:n_basis_scalar]
    u_k_uy_local = u_k_local[n_basis_scalar:]

    for qp, w in zip(qpts, qwts):
        J = transform.jacobian(mesh, 0, qp)
        detJ = abs(np.linalg.det(J))
        JinvT = np.linalg.inv(J).T
        
        G_scalar_phys = ref_q2.grad(*qp) @ JinvT
        
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

    print(f"Comparing compiler-generated vector with manually computed ground truth...")
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

    equation = dot(Constant([0.0,0.0],dim=1), v) * dx() == dot(u_k, v) * dx()
    
    _, R = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_vector = R

    ref_q2 = get_reference("quad", poly_order=2)
    q_order = 5
    qpts, qwts = volume("quad", q_order)
    n_basis_scalar = 9
    
    expected_vector = np.zeros(n_basis_scalar * 2)
    
    elemental_dofs = dof_handler.get_elemental_dofs(0)
    u_k_local = u_k.get_nodal_values(elemental_dofs)
    u_k_ux_local = u_k_local[:n_basis_scalar]
    u_k_uy_local = u_k_local[n_basis_scalar:]

    for qp, w in zip(qpts, qwts):
        J = transform.jacobian(mesh, 0, qp)
        detJ = abs(np.linalg.det(J))
        N_scalar = ref_q2.shape(*qp)
        
        u_k_val_at_qp = np.array([N_scalar @ u_k_ux_local, N_scalar @ u_k_uy_local])
        
        expected_vector[:n_basis_scalar] += (u_k_val_at_qp[0] * N_scalar) * w * detJ
        expected_vector[n_basis_scalar:] += (u_k_val_at_qp[1] * N_scalar) * w * detJ

    print("Comparing compiler-generated vector with manually computed ground truth...")
    np.testing.assert_allclose(compiler_vector, expected_vector, rtol=1e-12, atol=1e-12)
    print("\nSUCCESS: Compiler's RHS Q2 mass vector is correct!")

#endregion

#====================================================================================
#  WHITE-BOX TESTS FOR INDIVIDUAL TERMS IN NAVIER-STOKES
#====================================================================================

def get_exact_q1_mass_matrix():
    """Computes the exact 4x4 mass matrix for a Q1 element on a 1x1 physical square."""
    xi, eta = sp.symbols('xi eta')
    N_ref = [(1-xi)/2*(1-eta)/2, (1+xi)/2*(1-eta)/2, (1+xi)/2*(1+eta)/2, (1-xi)/2*(1+eta)/2]
    node_order = [0, 1, 3, 2]
    N_ordered = [N_ref[i] for i in node_order]
    M = sp.zeros(4, 4)
    detJ = sp.Rational(1, 4)  # For mapping [-1,1]^2 to [0,1]^2
    for i in range(4):
        for j in range(4):
            M[i, j] = sp.integrate(N_ordered[i] * N_ordered[j] * detJ, (xi, -1, 1), (eta, -1, 1))
    return np.array(M, dtype=float)

def test_vector_mass_matrix_q1():
    """White-box test for a vector Q1 mass matrix on a single element."""
    print("\n" + "="*70)
    print("White-Box Test of Q1 Vector Mass Matrix")
    print("="*70)
    
    nodes, elems, _, corners = structured_quad(1, 1, nx=1, ny=1, poly_order=1)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)
    me = MixedElement(mesh, field_specs={'ux': 1, 'uy': 1})
    dof_handler = DofHandler(me, method='cg')

    u = VectorTrialFunction(FunctionSpace("velocity", ['ux', 'uy']))
    v = VectorTestFunction(FunctionSpace("velocity", ['ux', 'uy']))
    equation = (dot(u, v) * dx() == dot(Constant([0.0,0.0],dim=1) , v) * dx())

    A, _ = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_matrix = A.toarray()

    scalar_mass_block = get_exact_q1_mass_matrix()
    expected_matrix = np.block([[scalar_mass_block, np.zeros((4, 4))], [np.zeros((4, 4)), scalar_mass_block]])
    
    print("Comparing Compiler's Q1 Vector Mass Matrix with Analytical Ground Truth...")
    np.testing.assert_allclose(compiler_matrix, expected_matrix, rtol=1e-12, atol=1e-12)
    print("SUCCESS: Q1 Vector Mass Matrix is correct!")


def get_exact_q2_mass_matrix():
    """Computes the exact 9x9 mass matrix for a Q2 element on a 2x2 physical square."""
    xi, eta = sp.symbols('xi eta')
    N_ref_nodes = [
        xi*eta*(xi - 1)*(eta - 1)/4, (1 - xi**2)*eta*(eta - 1)/2, xi*eta*(xi + 1)*(eta - 1)/4,
        xi*(xi - 1)*(1 - eta**2)/2,  (1 - xi**2)*(1 - eta**2),   xi*(xi + 1)*(1 - eta**2)/2,
        xi*eta*(xi - 1)*(eta + 1)/4, (1 - xi**2)*eta*(eta + 1)/2, xi*eta*(xi + 1)*(eta + 1)/4
    ]
    # This ordering matches the lexicographical node order from structured_quad and get_reference
    M = sp.zeros(9, 9)
    detJ = 1  # For mapping [-1,1]^2 to [0,2]^2
    for i in range(9):
        for j in range(9):
            M[i, j] = sp.integrate(N_ref_nodes[i] * N_ref_nodes[j] * detJ, (xi, -1, 1), (eta, -1, 1))
    return np.array(M, dtype=float)


def test_vector_mass_matrix_q2():
    """White-box test for a vector Q2 mass matrix on a single element."""
    print("\n" + "="*70)
    print("White-Box Test of Q2 Vector Mass Matrix (LHS)")
    print("="*70)
    
    mesh, dof_handler = setup_single_q2_element()
    u = VectorTrialFunction(FunctionSpace("velocity", ['ux', 'uy']))
    v = VectorTestFunction(FunctionSpace("velocity", ['ux', 'uy']))
    equation = (dot(u, v) * dx() == dot(Constant([0.0,0.0],dim=1) , v) * dx())
    A, _ = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_matrix = A.toarray()

    scalar_mass_block = get_exact_q2_mass_matrix()
    expected_matrix = np.block([[scalar_mass_block, np.zeros((9, 9))], [np.zeros((9, 9)), scalar_mass_block]])
    
    print("Comparing Compiler's Q2 Vector Mass Matrix with block-diagonal analytical truth...")
    np.testing.assert_allclose(compiler_matrix, expected_matrix, rtol=1e-12, atol=1e-12)
    print("SUCCESS: Q2 Vector Mass Matrix is correct!")


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
    

    
    # The RHS can be a simple zero vector for this LHS test.
    equation = dot(dot(u_k,grad(u)),v) * dx() == dot(Constant([0.0,0.0],dim=1), v) * dx()
    
    A, _ = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_matrix = A.toarray()
    
    # The manual calculation below is already correct for this term.
    ref_q2 = get_reference("quad", poly_order=2)
    q_order = 5
    qpts, qwts = volume("quad", q_order)
    n_basis_scalar = 9
    expected_matrix = np.zeros((n_basis_scalar * 2, n_basis_scalar * 2))
    
    elemental_dofs = dof_handler.get_elemental_dofs(0)
    u_k_local = u_k.get_nodal_values(elemental_dofs)
    u_k_ux_local, u_k_uy_local = u_k_local[:n_basis_scalar], u_k_local[n_basis_scalar:]
    
    for qp, w in zip(qpts, qwts):
        J = transform.jacobian(mesh, 0, qp)
        detJ = abs(np.linalg.det(J))
        JinvT = np.linalg.inv(J).T
        N = ref_q2.shape(*qp)
        G_phys = ref_q2.grad(*qp) @ JinvT
        u_k_val = np.array([N @ u_k_ux_local, N @ u_k_uy_local])
        
        for i in range(n_basis_scalar * 2):
            for j in range(n_basis_scalar * 2):
                # Term is (u_k ⋅ ∇u_j) ⋅ v_i
                integrand = 0.0
                if (i < n_basis_scalar and j < n_basis_scalar): # (ux, ux) block
                    convected_grad_j = u_k_val @ G_phys[j, :] # u_k ⋅ ∇(φ_j) for x-component
                    integrand = convected_grad_j * N[i]
                elif (i >= n_basis_scalar and j >= n_basis_scalar): # (uy, uy) block
                    convected_grad_j = u_k_val @ G_phys[j - n_basis_scalar, :] # u_k ⋅ ∇(φ_j) for y-component
                    integrand = convected_grad_j * N[i - n_basis_scalar]
                
                expected_matrix[i, j] += integrand * w * detJ

    print("Comparing compiler-generated matrix with manually computed ground truth...")
    np.testing.assert_allclose(compiler_matrix, expected_matrix, rtol=1e-12, atol=1e-12)
    print("\nSUCCESS: Compiler's LHS advection matrix is correct!")

def setup_mixed_q2_q1_element():
    """
    CORRECTED: Sets up a single Q2 element mesh and a mixed-space DoF handler.
    Both Q2 velocity and Q1 pressure are defined on the SAME high-order mesh.
    """
    poly_order = 2
    nodes, elems, _, corners = structured_quad(1, 1, nx=1, ny=1, poly_order=poly_order)
    # A single Q2 mesh is used for all fields.
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=poly_order)
    
    # MixedElement defines different order fields on the same mesh.
    me = MixedElement(mesh, field_specs={'ux': 2, 'uy': 2, 'p': 1})
    dof_handler = DofHandler(me, method='cg')
    return dof_handler, mesh


def test_mixed_pressure_grad_operator():
    """
    White-box test for the pressure gradient operator matrix: G_ij = -∫ p_j * div(v_i) dΩ
    This is the discrete gradient operator, often denoted -G.
    p: Q1 TrialFunction, v: Q2 VectorTestFunction
    """
    print("\n" + "="*70)
    print("White-Box Test of Mixed-Space Pressure Gradient Matrix (-p * div(v))")
    print("="*70)

    dof_handler, mesh = setup_mixed_q2_q1_element()
    v = VectorTestFunction(FunctionSpace("velocity", ['ux', 'uy']))
    p = TrialFunction('p')

    # CORRECTED UFL FORM: This is the standard pressure term in Stokes flow.
    equation = (-p * div(v)) * dx() == Constant(0.0) * dx()

    A, _ = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_matrix = A.toarray()

    ref_q2 = get_reference("quad", poly_order=2)
    ref_q1 = get_reference("quad", poly_order=1)
    q_order = 5
    qpts, qwts = volume("quad", q_order)

    n_v, n_p = 9, 4 # Number of basis functions for velocity and pressure
    expected_block = np.zeros((n_v * 2, n_p))

    for qp, w in zip(qpts, qwts):
        J = transform.jacobian(mesh, 0, qp)
        detJ = abs(np.linalg.det(J))
        JinvT = np.linalg.inv(J).T
        N_p = ref_q1.shape(*qp)
        G_v_phys = ref_q2.grad(*qp) @ JinvT

        for i in range(n_v * 2): # loop over test function v_i (rows)
            for j in range(n_p): # loop over trial function p_j (cols)
                div_vi = G_v_phys[i, 0] if i < n_v else G_v_phys[i - n_v, 1]
                integrand = -N_p[j] * div_vi
                expected_block[i, j] += integrand * w * detJ

    # The block has velocity test functions (rows) and pressure trial functions (cols).
    v_dofs = np.arange(18)
    p_dofs = 18 + np.arange(4)
    extracted_block = compiler_matrix[np.ix_(v_dofs, p_dofs)]
    
    print("Comparing compiler-generated G matrix block with manual ground truth...")
    np.testing.assert_allclose(extracted_block, expected_block, rtol=1e-12, atol=1e-12)
    print("\nSUCCESS: Compiler's mixed-space gradient matrix is correct!")


def test_mixed_velocity_divergence_operator():
    """
    White-box test for the velocity divergence operator matrix: D_ij = -∫ q_i * div(u_j) dΩ
    This is the discrete divergence operator, often denoted D or -G^T.
    q: Q1 TestFunction, u: Q2 VectorTrialFunction
    """
    print("\n" + "="*70)
    print("White-Box Test of Mixed-Space Velocity Divergence Matrix (-q * div(u))")
    print("="*70)

    dof_handler, mesh = setup_mixed_q2_q1_element()
    u = VectorTrialFunction(FunctionSpace("velocity", ['ux', 'uy']))
    q = TestFunction('p')
    
    equation = -(div(u) * q) * dx() == Constant(0.0)  * dx()
    
    A, _ = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_matrix = A.toarray()
    
    ref_q2 = get_reference("quad", poly_order=2)
    ref_q1 = get_reference("quad", poly_order=1)
    q_order = 5
    qpts, qwts = volume("quad", q_order)
    
    n_v, n_p = 9, 4
    expected_block = np.zeros((n_p, n_v * 2))
    
    for qp, w in zip(qpts, qwts):
        J = transform.jacobian(mesh, 0, qp)
        detJ = abs(np.linalg.det(J))
        JinvT = np.linalg.inv(J).T
        N_q = ref_q1.shape(*qp)
        G_u_phys = ref_q2.grad(*qp) @ JinvT
        
        for i in range(n_p): # loop over test functions q_i (rows)
            for j in range(n_v * 2): # loop over trial functions u_j (cols)
                div_uj = G_u_phys[j, 0] if j < n_v else G_u_phys[j - n_v, 1]
                integrand = -N_q[i] * div_uj
                expected_block[i, j] += integrand * w * detJ

    # The block has pressure test functions (rows) and velocity trial functions (cols).
    v_dofs = np.arange(18)
    p_dofs = 18 + np.arange(4)
    extracted_block = compiler_matrix[np.ix_(p_dofs, v_dofs)]
    
    print("Comparing compiler-generated D matrix block with manually computed ground truth...")
    np.testing.assert_allclose(extracted_block, expected_block, rtol=1e-12, atol=1e-12)
    print("\nSUCCESS: Compiler's mixed-space divergence matrix is correct!")


