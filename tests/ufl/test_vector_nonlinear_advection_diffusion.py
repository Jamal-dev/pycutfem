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





def test_vector_linear_advection_diffusion_newton():
    """
    Test for the time-dependent LINEAR vector advection-diffusion problem,
    solved using the Newton's method framework.

    This test verifies that the solver structure (Jacobian, residual, BCs)
    is correct for a linear problem, which should converge in one iteration.
    """
    print("\n" + "="*70)
    print("Test: Linear Vector Advection-Diffusion (solved with Newton)")
    print("="*70)

    # --- 1. Problem Setup (Exact Solution and Forcing Term) ---
    epsilon = 0.5
    beta_vec = np.array([0.5, -0.25]) # Constant advection field
    beta = Constant(beta_vec)

    t_sym = sp.Symbol('t')
    u_exact_sym_x = t_sym * sp.pi * x**3 * sp.pi * y
    u_exact_sym_y = t_sym * sp.pi * x * sp.pi * y**3

    # Compute forcing term f = ∂u/∂t - εΔu + (β ⋅ ∇)u
    grad_u_x = sp.Matrix([sp.diff(u_exact_sym_x, x), sp.diff(u_exact_sym_x, y)])
    grad_u_y = sp.Matrix([sp.diff(u_exact_sym_y, x), sp.diff(u_exact_sym_y, y)])
    linear_advection_x = beta_vec[0] * grad_u_x[0] + beta_vec[1] * grad_u_x[1]
    linear_advection_y = beta_vec[0] * grad_u_y[0] + beta_vec[1] * grad_u_y[1]

    f_sym_x = sp.diff(u_exact_sym_x, t_sym) - epsilon * (sp.diff(u_exact_sym_x, x, 2) + sp.diff(u_exact_sym_x, y, 2)) + linear_advection_x
    f_sym_y = sp.diff(u_exact_sym_y, t_sym) - epsilon * (sp.diff(u_exact_sym_y, x, 2) + sp.diff(u_exact_sym_y, y, 2)) + linear_advection_y

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
    theta = Constant(0.5)

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
    
    for n in range(num_steps):
        t_n = n * dt_val
        t_np1 = (n + 1) * dt_val
        print(f"\n--- Solving Time Step {n+1}/{num_steps} | t = {t_np1:.2f}s ---")

        # --- Boundary Condition Setup ---
        bc_tags = {'boundary': lambda x, y: (np.isclose(x, 0.0) or np.isclose(x, 1.0) or np.isclose(y, 0.0) or np.isclose(y, 1.0))}
        mesh.tag_boundary_edges(bc_tags)
        bcs = [
            BoundaryCondition('ux', 'dirichlet', 'boundary', lambda x, y: u_exact_func_x(x, y, t_np1)),
            BoundaryCondition('uy', 'dirichlet', 'boundary', lambda x, y: u_exact_func_y(x, y, t_np1))
        ]
        bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

        dirichlet_data = dof_handler.get_dirichlet_data(bcs)
        for dof, val in dirichlet_data.items():
            u_k.nodal_values[dof] = val

        # --- "Newton's Method" Loop (should converge in 1 iteration) ---
        max_iter = 3 # Allow a few just in case, but expect 1.
        tol = 1e-8
        for k in range(max_iter):
            print(f"  Iteration {k+1}")

            # Define residual R(u_k) = F(u_k) with LINEAR advection
            residual = (
                dot(u_k - u_n, v) / dt 
                + theta * eps * inner(grad(u_k), grad(v)) 
                + theta * (dot(beta, grad(u_k[0])) * v[0] + dot(beta, grad(u_k[1])) * v[1]) # Linear term
                + (1.0 - theta) * eps * inner(grad(u_n), grad(v)) 
                + (1.0 - theta) * (dot(beta, grad(u_n[0])) * v[0] + dot(beta, grad(u_n[1])) * v[1]) # Linear term
                - (1.0 - theta) * dot(f_n, v)
                - theta * dot(f_np1, v)
            ) * dx()

            # Define Jacobian J = F'(u_k) with LINEAR advection
            jacobian = (
                dot(delta_u, v) / dt 
                + theta * eps * inner(grad(delta_u), grad(v)) 
                + theta * (dot(beta, grad(delta_u[0])) * v[0] + dot(beta, grad(delta_u[1])) * v[1]) # Linear term
            ) * dx()

            f_n.set_values_from_function(lambda x, y: f_func(x, y, t_n))
            f_np1.set_values_from_function(lambda x, y: f_func(x, y, t_np1))
            
            A, R_vec = assemble_form(jacobian == -residual, dof_handler=dof_handler, bcs=bcs_homog, quad_order=5)
            
            norm_res = np.linalg.norm(R_vec)
            print(f"    Residual Norm: {norm_res:.3e}")

            if norm_res < tol:
                print(f"    Converged for t={t_np1:.2f} after {k+1} iterations.")
                # For a linear problem, it should converge on the first iteration.
                if k > 0:
                    print("WARNING: Linear problem took more than 1 iteration to converge.")
                break

            delta_u_vec = spla.spsolve(A, R_vec)
            u_k.nodal_values[:] += delta_u_vec
        else:
            raise RuntimeError(f"Solver did not converge after {max_iter} iterations.")

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
        
        u_n.nodal_values[:] = u_k.nodal_values[:]

    print("\nLinear advection-diffusion test (solved via Newton) passed successfully!")


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
    epsilon = 10
    t_sym = sp.Symbol('t')
    u_exact_sym_x = t_sym * sp.pi * x**3 * sp.pi * y
    u_exact_sym_y = t_sym * sp.pi * x * sp.pi * y**3

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
    nodes, elems, _, corners = structured_quad(1, 1, nx=10, ny=10, poly_order=poly_order)
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
    node_coords = mesh.nodes_x_y_pos
    ux_dofs = dof_handler.get_field_slice('ux')
    uy_dofs = dof_handler.get_field_slice('uy')
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
            residual = -(
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
            A, R_vec = assemble_form(jacobian == residual, dof_handler=dof_handler, bcs=bcs_homog, quad_order=6)
            
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
        exact_sol = np.zeros_like(u_k.nodal_values)
        exact_sol[ux_dofs] = u_exact_func_x(node_coords[:, 0], node_coords[:, 1], t_np1)
        exact_sol[uy_dofs] = u_exact_func_y(node_coords[:, 0], node_coords[:, 1], t_np1)

        l2_error = np.linalg.norm(u_k.nodal_values - exact_sol) / np.linalg.norm(exact_sol)
        print(f"\n  Relative L2 Error at t={t_np1:.2f}: {l2_error:.4e}")
        assert l2_error < 1e-2, f"Error {l2_error:.2e} is too high."
        
        # Update u_n for the next time step
        u_n.nodal_values[:] = u_k.nodal_values[:]

    print("\nNonlinear time-dependent advection-diffusion test passed successfully!")

