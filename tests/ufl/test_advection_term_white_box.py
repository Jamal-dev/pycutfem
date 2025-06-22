import numpy as np
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (VectorTrialFunction, VectorTestFunction, VectorFunction,
                                      grad, dot, Constant)
from pycutfem.ufl.measures import dx
from pycutfem.ufl.forms import assemble_form
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.fem.reference import get_reference
from pycutfem.fem import transform
from pycutfem.integration.quadrature import volume

def setup_single_q2_element():
    """Sets up a single Q2 element mesh and DoF handler on [0,1]x[0,1]."""
    nodes, elems, _, corners = structured_quad(1, 1, nx=1, ny=1, poly_order=2)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=2)
    fe_map = {'ux': mesh, 'uy': mesh}
    dof_handler = DofHandler(fe_map, method='cg')
    return mesh, dof_handler


def test_lhs_advection_q2_final_corrected():
    """
    A thorough and mathematically correct white-box test for the term:
    A_ij = ∫ ((u_k ⋅ ∇)u_j) ⋅ v_i dΩ
    This version correctly calculates all four blocks of the matrix,
    including the non-zero off-diagonal cross-terms.
    """
    print("\n" + "="*70)
    print("White-Box Test of Coupled Advection Matrix (uk_grad_u)")
    print("="*70)

    mesh, dof_handler = setup_single_q2_element()
    
    V = FunctionSpace("velocity", ['ux', 'uy'])
    u, v = VectorTrialFunction(V), VectorTestFunction(V)
    
    u_k = VectorFunction("u_k", ['ux', 'uy'], dof_handler)
    np.random.seed(42)
    u_k.nodal_values[:] = np.random.rand(dof_handler.total_dofs)
    
    # UFL form for ((u_k ⋅ ∇)u) ⋅ v
    # advection_term = (dot(u_k, grad(u[0]))* v[0] +
    #                  dot(u_k, grad(u[1])) * v[1] +
    #                  dot(u_k, grad(u[0])) * v[1] +
    #                  dot(u_k, grad(u[1])) * v[0])
    advection_term = dot(dot(u_k, grad(u)), v)
    quad_degree = 5
    form = (advection_term * dx() == Constant(0.0) * v[0] * dx())

    A, _ = assemble_form(form, dof_handler, bcs=[], quad_order=quad_degree)
    compiler_matrix = A.toarray()
    
    # --- Thorough Manual Calculation (Block-by-Block) ---
    ref_q2 = get_reference("quad", poly_order=2)
    qpts, qwts = volume("quad", quad_degree)
    n_s = 9
    
    expected_matrix = np.zeros((n_s * 2, n_s * 2))
    
    dofs_ux = dof_handler.element_maps['ux'][0]
    dofs_uy = dof_handler.element_maps['uy'][0]
    u_k_ux_local = u_k.get_nodal_values(dofs_ux)
    u_k_uy_local = u_k.get_nodal_values(dofs_uy)
    
    for qp, w in zip(qpts, qwts):
        J = transform.jacobian(mesh, 0, qp)
        detJ = abs(np.linalg.det(J))
        JinvT = np.linalg.inv(J).T
        
        N_vals = ref_q2.shape(*qp)
        G_vals_phys = ref_q2.grad(*qp) @ JinvT

        u_k_val_at_qp = np.array([N_vals @ u_k_ux_local, N_vals @ u_k_uy_local])
        
        # --- CORRECTED AND EXPLICIT BLOCK CALCULATION ---
        
        # Pre-compute matrix terms that are repeatedly used.
        # N_outer_G_x[i, j] = N_i * (∂N_j/∂x)
        # N_outer_G_y[i, j] = N_i * (∂N_j/∂y)
        N_outer_G_x = np.outer(N_vals, G_vals_phys[:, 0])
        N_outer_G_y = np.outer(N_vals, G_vals_phys[:, 1])

        # Let Adv_u = (u_k ⋅ ∇)u. The integrand is Adv_u ⋅ v
        # Adv_u_x = u_k_x * (∂u_x/∂x) + u_k_y * (∂u_y/∂x)
        # Adv_u_y = u_k_x * (∂u_x/∂y) + u_k_y * (∂u_y/∂y)
        
        # ux-ux block (Top-Left): Integrand is (u_k_x * ∂u_j_x/∂x) * v_i_x
        # -> u_k_x(qp) * [N_i(qp) * (∂N_j/∂x)(qp)]
        block_ux_ux_contrib = N_outer_G_x * u_k_val_at_qp[0] * w * detJ
        
        # uy-ux block (Bottom-Left): Integrand is (u_k_x * ∂u_j_x/∂y) * v_i_y
        # -> u_k_x(qp) * [N_i(qp) * (∂N_j/∂y)(qp)]
        block_uy_ux_contrib = N_outer_G_y * u_k_val_at_qp[0] * w * detJ

        # ux-uy block (Top-Right): Integrand is (u_k_y * ∂u_j_y/∂x) * v_i_x
        # -> u_k_y(qp) * [N_i(qp) * (∂N_j/∂x)(qp)] (CROSS-TERM)
        block_ux_uy_contrib = N_outer_G_x * u_k_val_at_qp[1] * w * detJ

        # uy-uy block (Bottom-Right): Integrand is (u_k_y * ∂u_j_y/∂y) * v_i_y
        # -> u_k_y(qp) * [N_i(qp) * (∂N_j/∂y)(qp)]
        block_uy_uy_contrib = N_outer_G_y * u_k_val_at_qp[1] * w * detJ
        
        # Add contributions to the final matrix
        expected_matrix[0:n_s, 0:n_s]     += block_ux_ux_contrib
        expected_matrix[n_s:18, 0:n_s]    += block_uy_ux_contrib
        expected_matrix[0:n_s, n_s:18]    += block_ux_uy_contrib
        expected_matrix[n_s:18, n_s:18]   += block_uy_uy_contrib
                
    # --- Comparison ---
    print("Comparing compiler-generated matrix with corrected ground truth...")
    try:
        np.testing.assert_allclose(compiler_matrix, expected_matrix, rtol=1e-9, atol=1e-9)
        print("\n✅ SUCCESS: Compiler's matrix for the (uk_grad_u) term is correct!")
    except AssertionError as e:
        print("\n❌ FAILURE: Matrices do not match.")
        print(e)


def test_rhs_nonlinear_advection_corrected():
    """
    Corrected white-box test for the RHS vector from: ((u_k ⋅ ∇)u_k) ⋅ v
    This version correctly includes the cross-interaction terms.
    """
    print("\n" + "="*70)
    print("White-Box Test of RHS Nonlinear Advection Term (Corrected)")
    print("="*70)

    mesh, dof_handler = setup_single_q2_element()
    velocity_space = FunctionSpace("velocity", ['ux', 'uy'])
    v = VectorTestFunction(velocity_space)

    u_k = VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler)
    np.random.seed(42)
    # Using a simple linear field to make debugging easier if needed
    coords = dof_handler.get_dof_coords('ux')
    u_k.nodal_values[0:9] = 0.2 + 0.5 * coords[:, 0] + 0.3 * coords[:, 1]
    u_k.nodal_values[9:18] = 0.5 - 0.4 * coords[:, 0] - 0.1 * coords[:, 1]

    # The correct, compact UFL form for the fully coupled term
    advection_term = dot(dot(u_k, grad(u_k)), v)

    equation = Constant(0.0) * v[0] * dx() == 1.0 * advection_term * dx()
    
    quad_degree = 5
    _, R = assemble_form(equation, dof_handler=dof_handler, bcs=[], quad_order=quad_degree)
    compiler_vector = R
    
    # --- Manual Calculation with Correct Cross-Terms ---
    ref_q2 = get_reference("quad", poly_order=2)
    qpts, qwts = volume("quad", quad_degree)
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
        
        # Value of u_k at the quadrature point
        u_k_val = np.array([
            N_scalar @ u_k_ux_local,
            N_scalar @ u_k_uy_local
        ])
        
        # Gradient of u_k at the quadrature point
        grad_uk_val = np.zeros((2, 2))
        grad_uk_val[0, :] = G_scalar_phys.T @ u_k_ux_local # row 0: grad(u_k_x)
        grad_uk_val[1, :] = G_scalar_phys.T @ u_k_uy_local # row 1: grad(u_k_y)
        
        # --- Correct calculation of the convected vector components ---
        # x-component: u_k_x * (∂u_k_x/∂x) + u_k_y * (∂u_k_y/∂x)
        convected_vec_x = u_k_val[0] * grad_uk_val[0, 0] + u_k_val[1] * grad_uk_val[1, 0]
        
        # y-component: u_k_x * (∂u_k_x/∂y) + u_k_y * (∂u_k_y/∂y)
        convected_vec_y = u_k_val[0] * grad_uk_val[0, 1] + u_k_val[1] * grad_uk_val[1, 1]
        
        # Add contributions to the final residual vector
        # Contribution to ux equations is ∫ (convected_vec_x) * N_i dΩ
        expected_vector[:n_basis_scalar] += (convected_vec_x * N_scalar) * w * detJ
        
        # Contribution to uy equations is ∫ (convected_vec_y) * N_i dΩ
        expected_vector[n_basis_scalar:] += (convected_vec_y * N_scalar) * w * detJ

    print(f"Comparing compiler-generated vector with manually computed ground truth...")
    try:
        np.testing.assert_allclose(compiler_vector, expected_vector, rtol=1e-9, atol=1e-9)
        print("\n✅ SUCCESS: Compiler's RHS nonlinear advection vector is correct!")
    except AssertionError as e:
        print("\n❌ FAILURE: Vectors do not match.")
        print(e)

if __name__ == '__main__':
    test_lhs_advection_q2_final_corrected()