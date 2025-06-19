import numpy as np
import scipy.sparse.linalg as spla
import sympy as sp
from ufl.functionspace import FunctionSpace
from ufl.expressions import (    VectorTrialFunction, VectorTestFunction, VectorFunction,
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
    # We solve 0 = term, so the RHS vector F will be the assembly of -term.
    # We multiply by -1 to get the vector for the term itself.
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
        
        # Scalar basis shapes and physical gradients
        N_scalar = ref_q2.shape(*qp)       # Shape (9,)
        G_scalar_phys = ref_q2.grad(*qp) @ JinvT # Shape (9, 2)
        
        # Interpolate the gradient of the known function u_k at the quadrature point.
        # This results in a 2x2 matrix.
        grad_uk_val = np.zeros((2, 2))
        grad_uk_val[0, :] = G_scalar_phys.T @ u_k_ux_local # d(u_k.x)/dx, d(u_k.x)/dy
        grad_uk_val[1, :] = G_scalar_phys.T @ u_k_uy_local # d(u_k.y)/dx, d(u_k.y)/dy
        
        # The weak form is ∫ ∇u_k : ∇v dΩ.
        # We need to compute the contribution for each basis function of v.
        for j in range(n_basis_scalar * 2):
            # Construct the gradient of the j-th vector basis function (a 2x2 matrix)
            grad_v_basis_j = np.zeros((2, 2))
            if j < n_basis_scalar: # This is a basis function for the 'ux' component
                grad_v_basis_j[0, :] = G_scalar_phys[j, :]
            else: # This is a basis function for the 'uy' component
                grad_v_basis_j[1, :] = G_scalar_phys[j - n_basis_scalar, :]
            
            # Compute the inner product and accumulate
            integrand_val = np.sum(grad_uk_val * grad_v_basis_j) # A:B = sum(Aij * Bij)
            expected_vector[j] += integrand_val * w * detJ

    # 5. Compare the vectors
    print(f"compiler.shape: {compiler_vector.shape}, expected.shape: {expected_vector.shape}")
    print("Comparing compiler-generated vector with manually computed ground truth...")
    np.testing.assert_allclose(compiler_vector, expected_vector, rtol=1e-12, atol=1e-12)
    print("\nSUCCESS: Compiler's RHS diffusion vector is correct!")

def test_rhs_mass_matrix():
    """
    White-box test for the RHS vector from: dot(u_k, v)
    where u_k is a known VectorFunction.
    """
    print("\n" + "="*70)
    print("White-Box Test of RHS Mass Matrix Term")
    print("="*70)

    mesh, dof_handler = setup_single_q2_element()
    velocity_space = FunctionSpace("velocity", ['ux', 'uy'])
    v = VectorTestFunction(velocity_space)

    # 1. Define a known function u_k with arbitrary, non-zero data
    u_k = VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler)
    np.random.seed(123) # Use a seed for reproducibility
    u_k.nodal_values[:] = np.random.rand(dof_handler.total_dofs)

    # 2. Define the UFL form for the RHS term.
    # We solve 0 = term, so the RHS vector F will be the assembly of -term.
    # We multiply by -1.0 to get the vector for the term itself.
    mass_term = dot(u_k, v)
    equation = Constant(0.0) * v[0] * dx() == 1.0 * mass_term * dx()
    
    # 3. Get the vector R from your real assembler
    _, R = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_vector = R

    # 4. Manually compute the ground truth vector
    ref_q2 = get_reference("quad", poly_order=2)
    q_order = 5 # Use a sufficiently high quadrature order
    qpts, qwts = volume("quad", q_order)
    n_basis_scalar = 9
    
    expected_vector = np.zeros(n_basis_scalar * 2)
    
    # Get the local DoF values for the u_k function on the single element
    dofs_ux = dof_handler.element_maps['ux'][0]
    dofs_uy = dof_handler.element_maps['uy'][0]
    u_k_ux_local = u_k.get_nodal_values(dofs_ux)
    u_k_uy_local = u_k.get_nodal_values(dofs_uy)

    for qp, w in zip(qpts, qwts):
        # Geometric transformations for the current quadrature point
        J = transform.jacobian(mesh, 0, qp)
        detJ = abs(np.linalg.det(J))
        
        # Get the values of the SCALAR basis functions (N_i) at this point
        N_scalar = ref_q2.shape(*qp) # Shape (9,)
        
        # Interpolate the value of the known function u_k at this point
        u_k_val_at_qp = np.array([
            N_scalar @ u_k_ux_local,  # Interpolated ux value
            N_scalar @ u_k_uy_local   # Interpolated uy value
        ]) # Result is a vector of shape (2,)

        # The weak form is ∫ (u_k ⋅ v) dΩ. The local vector F_j = ∫ (u_k ⋅ Φ_j) dΩ.
        # At this quadrature point, the integrand contribution for each basis function is (u_k_val ⋅ Φ_j(qp)).
        # Φ_j for ux is (N_j, 0), so the dot product is u_k_val[0] * N_j
        # Φ_j for uy is (0, N_j), so the dot product is u_k_val[1] * N_j
        
        # Contribution to the 'ux' part of the local vector
        ux_contribution = u_k_val_at_qp[0] * N_scalar
        expected_vector[:n_basis_scalar] += ux_contribution * w * detJ
        
        # Contribution to the 'uy' part of the local vector
        uy_contribution = u_k_val_at_qp[1] * N_scalar
        expected_vector[n_basis_scalar:] += uy_contribution * w * detJ

    # 5. Compare the compiler's result with the manually computed ground truth
    print("Comparing compiler-generated vector with manually computed ground truth...")
    np.testing.assert_allclose(compiler_vector, expected_vector, rtol=1e-12, atol=1e-12)
    
    print("\nSUCCESS: Compiler's RHS mass matrix vector is correct!")


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

    # 1. Define a known function u_k with arbitrary data
    u_k = VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler)
    np.random.seed(42)
    u_k.nodal_values[:] = np.sin(np.linspace(0, 5, dof_handler.total_dofs))

    # 2. Define the UFL form for the RHS term
    advection_term = (
        dot(u_k, grad(u_k[0])) * v[0] +
        dot(u_k, grad(u_k[1])) * v[1]
    )
    equation = Constant(0.0) * v[0] * dx() == 1.0 * advection_term * dx()
    
    # 3. Get the vector R from YOUR real assembler
    _, R = assemble_form(equation, dof_handler=dof_handler, bcs=[])
    compiler_vector = R
    
    # 4. Manually compute the ground truth vector
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
        
        N_scalar = ref_q2.shape(*qp)       # Shape (9,)
        G_scalar_phys = ref_q2.grad(*qp) @ JinvT # Shape (9, 2)
        
        # Interpolate u_k and its component gradients at the quadrature point
        u_k_val = np.array([
            N_scalar @ u_k_ux_local,
            N_scalar @ u_k_uy_local
        ]) # Shape (2,)
        
        grad_uk0_val = G_scalar_phys.T @ u_k_ux_local # Shape (2,)
        grad_uk1_val = G_scalar_phys.T @ u_k_uy_local # Shape (2,)

        # Term 1: (u_k ⋅ ∇u_k[0]) * v[0]
        coeff0 = np.dot(u_k_val, grad_uk0_val)
        expected_vector[:n_basis_scalar] += (coeff0 * N_scalar) * w * detJ
        
        # Term 2: (u_k ⋅ ∇u_k[1]) * v[1]
        coeff1 = np.dot(u_k_val, grad_uk1_val)
        expected_vector[n_basis_scalar:] += (coeff1 * N_scalar) * w * detJ

    # 5. Compare the vectors
    print(f"compiler.shape: {compiler_vector.shape}, expected.shape: {expected_vector.shape}")
    print("Comparing compiler-generated vector with manually computed ground truth...")
    np.testing.assert_allclose(compiler_vector, expected_vector, rtol=1e-12, atol=1e-12)
    print("\nSUCCESS: Compiler's RHS nonlinear advection vector is correct!")



def test_vector_nonlinear_advection_diffusion_theta():
    """
    This version correctly applies homogeneous boundary conditions to the
    Newton update step, which is required for convergence with non-zero
    Dirichlet conditions.
    """
    print("\n" + "="*70)
    print("Test: Nonlinear Time-Dependent Vector Advection-Diffusion (1-Step θ-Scheme)")
    print("="*70)

    # --- 1. Problem Setup (Exact Solution and Forcing Term) ---
    epsilon = 0.1
    t_sym = sp.Symbol('t')
    u_exact_sym_x = t_sym * sp.pi * x * sp.cos(sp.pi * y)
    u_exact_sym_y = t_sym * sp.pi * x * sp.sin(sp.pi * y)

    grad_u_x = sp.Matrix([sp.diff(u_exact_sym_x, x), sp.diff(u_exact_sym_x, y)])
    grad_u_y = sp.Matrix([sp.diff(u_exact_sym_y, x), sp.diff(u_exact_sym_y, y)])
    nonlinear_term_x = u_exact_sym_x * grad_u_x[0] + u_exact_sym_y * grad_u_x[1]
    nonlinear_term_y = u_exact_sym_x * grad_u_y[0] + u_exact_sym_y * grad_u_y[1]

    f_sym_x = sp.diff(u_exact_sym_x, t_sym) - epsilon * (sp.diff(u_exact_sym_x, x, 2) + sp.diff(u_exact_sym_x, y, 2)) + nonlinear_term_x
    f_sym_y = sp.diff(u_exact_sym_y, t_sym) - epsilon * (sp.diff(u_exact_sym_y, x, 2) + sp.diff(u_exact_sym_y, y, 2)) + nonlinear_term_y

    u_exact_func_x = sp.lambdify((x, y, t_sym), u_exact_sym_x, 'numpy')
    u_exact_func_y = sp.lambdify((x, y, t_sym), u_exact_sym_y, 'numpy')
    f_func = sp.lambdify((x, y, t_sym), [f_sym_x, f_sym_y], 'numpy')

    # --- 2. Mesh and Function Space Setup ---
    poly_order = 2
    nodes, elems, _, corners = structured_quad(1, 1, nx=6, ny=6, poly_order=poly_order)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=poly_order)
    fe_map = {'ux': mesh, 'uy': mesh}
    dof_handler = DofHandler(fe_map, method='cg')

    # --- 3. UFL Function Definitions ---
    dt_val = 0.1
    t_n_val = 0.0
    dt = Constant(dt_val)
    eps = Constant(epsilon)
    theta = Constant(1.0) # Crank-Nicolson is a good choice

    u_n = VectorFunction(name="u_n", field_names=['ux', 'uy'], dof_handler=dof_handler)
    u_k = VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler)
    delta_u = VectorTrialFunction(FunctionSpace("velocity", ['ux', 'uy']))
    v = VectorTestFunction(FunctionSpace("velocity", ['ux', 'uy']))
    f_n = VectorFunction(name="f_n", field_names=['ux', 'uy'], dof_handler=dof_handler)
    f_np1 = VectorFunction(name="f_np1", field_names=['ux', 'uy'], dof_handler=dof_handler)

    # --- 4. Set Initial Conditions ---
    u_n.set_values_from_function(lambda x, y: np.array([
        u_exact_func_x(x, y, t_n_val), u_exact_func_y(x, y, t_n_val)
    ]))
    u_k.nodal_values[:] = u_n.nodal_values[:]

    # --- 5. Time-stepping Loop ---
    T_end = 0.5
    num_steps = int(T_end / dt_val)
    # --- Boundary Condition Setup (CORRECTED) ---
    # A. Tag all boundaries of the unit square.
    bc_tags = {
        'boundary': lambda x, y: (np.isclose(x, 0.0) or np.isclose(x, 1.0) or
                                    np.isclose(y, 0.0) or np.isclose(y, 1.0))
    }
    mesh.tag_boundary_edges(bc_tags)
    
    # C. Create the corresponding HOMOGENEOUS BCs for the Newton update (delta_u).
    bcs_homog = [
            BoundaryCondition('ux', 'dirichlet', 'boundary', lambda x, y: 0.0),
            BoundaryCondition('uy', 'dirichlet', 'boundary', lambda x, y: 0.0)
        ]
    for n in range(1,num_steps):
        t_n = n * dt_val
        t_np1 = (n + 1) * dt_val
        print(f"\n--- Solving Time Step {n+1}/{num_steps} | t = {t_np1:.2f}s ---")


        # B. Define the NON-HOMOGENEOUS boundary conditions for the time step t_np1.
        bcs = [
            BoundaryCondition('ux', 'dirichlet', 'boundary', lambda x, y: u_exact_func_x(x, y, t_np1)),
            BoundaryCondition('uy', 'dirichlet', 'boundary', lambda x, y: u_exact_func_y(x, y, t_np1))
        ]

        
        # D. Apply non-homogeneous BCs to the initial guess u_k for this time step.
        # This ensures the first residual calculation is correct.
        dirichlet_data = dof_handler.get_dirichlet_data(bcs)
        for dof, val in dirichlet_data.items():
            u_k.nodal_values[dof] = val

        # --- Newton's Method Loop ---
        max_iter = 10
        tol = 1e-8
        for k in range(max_iter):
            print(f"  Newton iteration {k+1}")

            # Define residual R(u_k) = F(u_k)
            residual = (
                dot(u_k - u_n, v) / dt 
                + theta * eps * inner(grad(u_k), grad(v)) 
                + theta * (dot(u_k, grad(u_k[0])) * v[0] + dot(u_k, grad(u_k[1])) * v[1])
                + (1.0 - theta) * eps * inner(grad(u_n), grad(v)) 
                + (1.0 - theta) * (dot(u_n, grad(u_n[0])) * v[0] + dot(u_n, grad(u_n[1])) * v[1])
                - (1.0 - theta) * dot(f_n, v)
                - theta * dot(f_np1, v)
            ) * dx()

            # Define Jacobian J = F'(u_k)
            jacobian = (
                dot(delta_u, v) / dt 
                + theta * eps * inner(grad(delta_u), grad(v)) 
                + theta * (dot(u_k, grad(delta_u[0])) * v[0] + dot(u_k, grad(delta_u[1])) * v[1]) 
                + theta * (dot(delta_u, grad(u_k[0])) * v[0] + dot(delta_u, grad(u_k[1])) * v[1])
            ) * dx()

            # Populate time-dependent functions
            f_n.set_values_from_function(lambda x, y: f_func(x, y, t_n))
            f_np1.set_values_from_function(lambda x, y: f_func(x, y, t_np1))
            
            # Assemble J*du = -R. We pass J and R to the assembler, and it handles the signs.
            # CRITICAL: Use the homogeneous BCs for the linear system solve.
            A, R_vec = assemble_form(jacobian == -residual, dof_handler=dof_handler, bcs=bcs_homog, quad_order=5)
            
            # The norm is checked on the assembled residual vector.
            norm_res = np.linalg.norm(R_vec)
            print(f"    Residual Norm: {norm_res:.3e}")

            if norm_res < tol:
                print(f"    Newton converged for t={t_np1:.2f} after {k+1} iterations.")
                break

            # Solve J*du = -R for du
            delta_u_vec = spla.spsolve(A, R_vec)
            
            # Update the solution: u_k+1 = u_k + du
            u_k.nodal_values[:] += delta_u_vec
        else:
            raise RuntimeError(f"Newton did not converge after {max_iter} iterations.")

        # --- Error Computation ---
        node_coords = mesh.nodes_x_y_pos
        ux_dofs = dof_handler.get_field_slice('ux')
        uy_dofs = dof_handler.get_field_slice('uy')
        exact_sol = np.zeros_like(u_k.nodal_values)
        exact_sol[ux_dofs] = u_exact_func_x(node_coords[:, 0], node_coords[:, 1], t_np1)
        exact_sol[uy_dofs] = u_exact_func_y(node_coords[:, 0], node_coords[:, 1], t_np1)

        l2_error = np.linalg.norm(u_k.nodal_values - exact_sol) / np.linalg.norm(exact_sol)
        print(f"\n  Relative L2 Error at t={t_np1:.2f}: {l2_error:.4e}")
        assert l2_error < 1e-2, f"Error {l2_error:.2e} is too high."
        
        # Update u_n for the next time step
        u_n.nodal_values[:] = u_k.nodal_values[:]

    print("\nNonlinear time-dependent advection-diffusion test passed successfully!")

