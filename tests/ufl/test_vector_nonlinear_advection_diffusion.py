import numpy as np
import scipy.sparse.linalg as spla
import sympy as sp
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (    VectorTrialFunction, VectorTestFunction, VectorFunction,
    grad, inner, dot, Constant, TrialFunction, TestFunction)
from pycutfem.ufl.measures import dx
from pycutfem.ufl.forms import BoundaryCondition, assemble_form
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.ufl.analytic import Analytic, x, y
from pycutfem.fem.reference import get_reference
from pycutfem.fem import transform
from pycutfem.integration.quadrature import volume
from pycutfem.fem.mixedelement import MixedElement





# def test_vector_linear_advection_diffusion_newton():
#     """
#     Test for the time-dependent LINEAR vector advection-diffusion problem,
#     solved using the Newton's method framework.

#     This test verifies that the solver structure (Jacobian, residual, BCs)
#     is correct for a linear problem, which should converge in one iteration.
#     """
#     print("\n" + "="*70)
#     print("Test: Linear Vector Advection-Diffusion (solved with Newton)")
#     print("="*70)

#     # --- 1. Problem Setup (Exact Solution and Forcing Term) ---
#     epsilon = 2.0
#     beta_vec = np.array([0.5, -0.25]) # Constant advection field
#     beta = Constant(beta_vec, dim =1)

#     t_sym = sp.Symbol('t')
#     u_exact_sym_x = t_sym * sp.pi * x**3 * sp.pi * y
#     u_exact_sym_y = t_sym * sp.pi * x * sp.pi * y**3

#     # Compute forcing term f = ∂u/∂t - εΔu + (β ⋅ ∇)u
#     grad_u_x = sp.Matrix([sp.diff(u_exact_sym_x, x), sp.diff(u_exact_sym_x, y)])
#     grad_u_y = sp.Matrix([sp.diff(u_exact_sym_y, x), sp.diff(u_exact_sym_y, y)])
#     linear_advection_x = beta_vec[0] * grad_u_x[0] + beta_vec[1] * grad_u_x[1]
#     linear_advection_y = beta_vec[0] * grad_u_y[0] + beta_vec[1] * grad_u_y[1]

#     f_sym_x = sp.diff(u_exact_sym_x, t_sym) - epsilon * (sp.diff(u_exact_sym_x, x, 2) + sp.diff(u_exact_sym_x, y, 2)) + linear_advection_x
#     f_sym_y = sp.diff(u_exact_sym_y, t_sym) - epsilon * (sp.diff(u_exact_sym_y, x, 2) + sp.diff(u_exact_sym_y, y, 2)) + linear_advection_y

#     u_exact_func_x = sp.lambdify((x, y, t_sym), u_exact_sym_x, 'numpy')
#     u_exact_func_y = sp.lambdify((x, y, t_sym), u_exact_sym_y, 'numpy')
#     f_func = sp.lambdify((x, y, t_sym), [f_sym_x, f_sym_y], 'numpy')

#     # --- 2. Mesh and Function Space Setup ---
#     poly_order = 2
#     nx, ny = 6, 6  # Number of elements in x and y directions
#     nodes, elems, _, corners = structured_quad(1, 1, nx=nx, ny=ny, poly_order=poly_order)
#     mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=poly_order)
#     me = MixedElement(mesh, field_specs={'ux': poly_order, 'uy': poly_order})
#     dof_handler = DofHandler(me, method='cg')

#     # --- 3. UFL Function Definitions ---
#     dt_val = 0.1
#     t_n_val = 0.0
#     dt = Constant(dt_val)
#     eps = Constant(epsilon)
#     theta = Constant(0.5)

#     u_n = VectorFunction(name="u_n", field_names=['ux', 'uy'], dof_handler=dof_handler)
#     u_k = VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler)
#     delta_u = VectorTrialFunction(FunctionSpace("velocity", ['ux', 'uy']), dof_handler=dof_handler)
#     v = VectorTestFunction(FunctionSpace("velocity", ['ux', 'uy']), dof_handler=dof_handler)
#     f_n = VectorFunction(name="f_n", field_names=['ux', 'uy'], dof_handler=dof_handler)
#     f_np1 = VectorFunction(name="f_np1", field_names=['ux', 'uy'], dof_handler=dof_handler)

#     # --- 4. Set Initial Conditions ---
#     u_n.set_values_from_function(lambda x, y: np.array([
#         u_exact_func_x(x, y, t_n_val), u_exact_func_y(x, y, t_n_val)
#     ]))
#     u_k.set_values_from_function(lambda x, y: np.array([
#         u_exact_func_x(x, y, t_n_val), u_exact_func_y(x, y, t_n_val)
#     ]))

#     # --- 5. Time-stepping Loop ---
#     T_end = 0.5
#     num_steps = int(T_end / dt_val)
    
#     bc_tags = {'boundary': lambda x, y: (np.isclose(x, 0.0) or np.isclose(x, 1.0) or np.isclose(y, 0.0) or np.isclose(y, 1.0))}
#     mesh.tag_boundary_edges(bc_tags)
#     t_np1 = (0 + 1) * dt_val
#     bcs = [
#             BoundaryCondition('ux', 'dirichlet', 'boundary', lambda x, y: u_exact_func_x(x, y, t_np1)),
#             BoundaryCondition('uy', 'dirichlet', 'boundary', lambda x, y: u_exact_func_y(x, y, t_np1))
#         ]
#     bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]
#     for n in range(num_steps):
#         t_n = n * dt_val
#         t_np1 = (n + 1) * dt_val
#         print(f"\n--- Solving Time Step {n+1}/{num_steps} | t = {t_np1:.2f}s ---")

#         # --- Boundary Condition Setup ---
#         bcs = [
#             BoundaryCondition('ux', 'dirichlet', 'boundary', lambda x, y: u_exact_func_x(x, y, t_np1)),
#             BoundaryCondition('uy', 'dirichlet', 'boundary', lambda x, y: u_exact_func_y(x, y, t_np1))
#         ]


#         # intilailzing the newtown solution
#         u_k.nodal_values[:] = u_n.nodal_values[:]
#         dof_handler.apply_bcs(bcs,u_k)


#         # --- "Newton's Method" Loop (should converge in 1 iteration) ---
#         max_iter = 3 # Allow a few just in case, but expect 1.
#         tol = 1e-8
#         for k in range(max_iter):
#             print(f"  Iteration {k+1}")

#             # Define residual R(u_k) = F(u_k) with LINEAR advection
#             residual = (
#                 dot(u_k - u_n, v) / dt 
#                 + theta * eps * inner(grad(u_k), grad(v)) 
#                 + theta * ( dot(dot(beta, grad(u_k)) , v) ) # Linear term
#                 + (1.0 - theta) * eps * inner(grad(u_n), grad(v)) 
#                 + (1.0 - theta) * ( dot(dot(beta, grad(u_n)) , v) ) # Linear term
#                 - (1.0 - theta) * dot(f_n, v)
#                 - theta * dot(f_np1, v)
#             ) * dx()

#             # Define Jacobian J = F'(u_k) with LINEAR advection
#             jacobian = (
#                 dot(delta_u, v) / dt 
#                 + theta * eps * inner(grad(delta_u), grad(v)) 
#                 + theta * ( dot(dot(beta, grad(delta_u)) , v) ) # Linear term
#             ) * dx()

#             f_n.set_values_from_function(lambda x, y: f_func(x, y, t_n))
#             f_np1.set_values_from_function(lambda x, y: f_func(x, y, t_np1))
            
#             A, R_vec = assemble_form(jacobian == -residual, dof_handler=dof_handler, bcs=bcs_homog, quad_order=5)
            
#             norm_res = np.linalg.norm(R_vec)
#             print(f"    Residual Norm: {norm_res:.3e}")

#             if norm_res < tol:
#                 print(f"    Converged for t={t_np1:.2f} after {k+1} iterations.")
#                 # For a linear problem, it should converge on the first iteration.
#                 if k > 0:
#                     print("WARNING: Linear problem took more than 1 iteration to converge.")
#                 break

#             delta_u_vec = spla.spsolve(A, R_vec)
#             dof_handler.add_to_functions(delta_u_vec, [u_k])
#         else:
#             raise RuntimeError(f"Solver did not converge after {max_iter} iterations.")

#         # --- Error Computation ---
#         exact = {'ux': lambda x,y: u_exact_func_x(x,y,t_np1),
#                  'uy': lambda x,y: u_exact_func_y(x,y,t_np1)}
#         l2_error = dof_handler.l2_error(u_k,exact,quad_order=5)

#         print(f"\n  Relative L2 Error at t={t_np1:.2f}: {l2_error:.4e}")
#         assert l2_error < 1e-2, f"Error {l2_error:.2e} is too high. Expected < 1e-2, got {l2_error:.2e}."
        
#         u_n.nodal_values[:] = u_k.nodal_values[:]

#     print("\nLinear advection-diffusion test (solved via Newton) passed successfully!")


def test_vector_nonlinear_advection_diffusion_theta():
    """
    Tests the solver for the time-dependent vector heat equation using
    the Method of Manufactured Solutions with the NON-CONSERVATIVE advection term.
    
    Equation: ∂u/∂t - εΔu + (∇u)u = f
    """
    print("\n" + "="*70)
    print("Test: NON-CONSERVATIVE Nonlinear Advection-Diffusion")
    print("="*70)

    # --- 1. Problem Setup (Exact Solution and Forcing Term) ---
    epsilon = 3.0
    poly_order = 2
    nx, ny = 6, 6
    t_sym = sp.Symbol('t')
    u_exact_sym_x = t_sym * sp.pi * x**3 * sp.pi * y
    u_exact_sym_y = t_sym * sp.pi * x * sp.pi * y**3

    # --- CORRECTED: SymPy for NON-CONSERVATIVE advection term (∇u)u ---
    # The gradient of u is a tensor: ∇u = [[∂ux/∂x, ∂ux/∂y], [∂uy/∂x, ∂uy/∂y]]
    grad_u_tensor = sp.Matrix([
        [sp.diff(u_exact_sym_x, x), sp.diff(u_exact_sym_x, y)],
        [sp.diff(u_exact_sym_y, x), sp.diff(u_exact_sym_y, y)]
    ])
    # The advection term is the matrix-vector product: (∇u)u
    u_vec_sym = sp.Matrix([u_exact_sym_x, u_exact_sym_y])
    advection_term_sym = grad_u_tensor * u_vec_sym
    
    nonlinear_term_x = advection_term_sym[0]
    nonlinear_term_y = advection_term_sym[1]
    # --- END CORRECTION ---

    f_sym_x = sp.diff(u_exact_sym_x, t_sym) - epsilon * (sp.diff(u_exact_sym_x, x, 2) + sp.diff(u_exact_sym_x, y, 2)) + nonlinear_term_x
    f_sym_y = sp.diff(u_exact_sym_y, t_sym) - epsilon * (sp.diff(u_exact_sym_y, x, 2) + sp.diff(u_exact_sym_y, y, 2)) + nonlinear_term_y

    u_exact_func_x = sp.lambdify((x, y, t_sym), u_exact_sym_x, 'numpy')
    u_exact_func_y = sp.lambdify((x, y, t_sym), u_exact_sym_y, 'numpy')
    f_func = sp.lambdify((x, y, t_sym), [f_sym_x, f_sym_y], 'numpy')

    # --- 2. Mesh and Function Space Setup ---
    nodes, elems, _, corners = structured_quad(1, 1, nx=nx, ny=ny, poly_order=poly_order)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=poly_order)
    me = MixedElement(mesh, field_specs={'ux': poly_order, 'uy': poly_order})
    dof_handler = DofHandler(me, method='cg')

    # --- 3. UFL Function Definitions ---
    dt_val = 0.1
    t_n_val = 0.0
    dt = Constant(dt_val)
    eps = Constant(epsilon)
    theta = Constant(1.0) # Using Backward Euler for simplicity and stability

    u_n = VectorFunction(name="u_n", field_names=['ux', 'uy'], dof_handler=dof_handler)
    u_k = VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler)
    delta_u = VectorTrialFunction(FunctionSpace("velocity", ['ux', 'uy']), dof_handler=dof_handler)
    v = VectorTestFunction(FunctionSpace("velocity", ['ux', 'uy']), dof_handler=dof_handler)
    f_n = VectorFunction(name="f_n", field_names=['ux', 'uy'], dof_handler=dof_handler)
    f_np1 = VectorFunction(name="f_np1", field_names=['ux', 'uy'], dof_handler=dof_handler)

    # --- 4. Set Initial Conditions ---
    u_n.set_values_from_function(lambda x, y: np.array([
        u_exact_func_x(x, y, t_n_val), u_exact_func_y(x, y, t_n_val)
    ]))
    
    # --- 5. Time-stepping Loop ---
    T_end = 0.5
    num_steps = int(T_end / dt_val)
    
    bc_tags = {
        'boundary': lambda x, y: (np.isclose(x, 0.0) or np.isclose(x, 1.0) or
                                    np.isclose(y, 0.0) or np.isclose(y, 1.0))
    }
    mesh.tag_boundary_edges(bc_tags)
    
    bcs_homog = [
            BoundaryCondition('ux', 'dirichlet', 'boundary', lambda x, y: 0.0),
            BoundaryCondition('uy', 'dirichlet', 'boundary', lambda x, y: 0.0)
        ]

    for n in range(num_steps):
        t_n = n * dt_val
        t_np1 = (n + 1) * dt_val
        print(f"\n--- Solving Time Step {n+1}/{num_steps} | t = {t_np1:.2f}s ---")

        bcs = [
            BoundaryCondition('ux', 'dirichlet', 'boundary', lambda x, y: u_exact_func_x(x, y, t_np1)),
            BoundaryCondition('uy', 'dirichlet', 'boundary', lambda x, y: u_exact_func_y(x, y, t_np1))
        ]
        
        u_k.nodal_values[:] = u_n.nodal_values[:]
        dof_handler.apply_bcs(bcs, u_k)

        # --- Newton's Method Loop ---
        max_iter = 10
        tol = 1e-8
        for k_iter in range(max_iter):
            print(f"  Newton iteration {k_iter+1}")

            # --- CORRECTED: Use NON-CONSERVATIVE form (∇u)u in weak form ---
            residual = -(
                dot(u_k - u_n, v) / dt 
                + theta * eps * inner(grad(u_k), grad(v)) 
                + theta * dot(dot(grad(u_k), u_k), v)  # Non-Conservative form
                + (1.0 - theta) * eps * inner(grad(u_n), grad(v)) 
                + (1.0 - theta) * dot(dot(grad(u_n), u_n), v) # Non-Conservative form
                - (1.0 - theta) * dot(f_n, v)
                - theta * dot(f_np1, v)
            ) * dx(metadata={'q': 6})

            jacobian = (
                dot(delta_u, v) / dt 
                + theta * eps * inner(grad(delta_u), grad(v)) 
                + theta * dot(dot(grad(delta_u), u_k), v) # Term 1 of linearized advection
                + theta * dot(dot(grad(u_k), delta_u), v) # Term 2 of linearized advection
            ) * dx(metadata={'q': 6})
            # --- END CORRECTION ---

            f_n.set_values_from_function(lambda x, y: f_func(x, y, t_n))
            f_np1.set_values_from_function(lambda x, y: f_func(x, y, t_np1))
            
            A, R_vec = assemble_form(jacobian == residual, dof_handler=dof_handler, bcs=bcs_homog)
            
            norm_res = np.linalg.norm(R_vec)
            print(f"    Residual Norm: {norm_res:.3e}")

            if norm_res < tol:
                print(f"    Newton converged for t={t_np1:.2f} after {k_iter+1} iterations.")
                break

            delta_u_vec = spla.spsolve(A, R_vec)
            dof_handler.add_to_functions(delta_u_vec, [u_k])
        else:
            raise RuntimeError(f"Newton did not converge after {max_iter} iterations.")

        exact = {'ux': lambda x,y: u_exact_func_x(x,y,t_np1),
                 'uy': lambda x,y: u_exact_func_y(x,y,t_np1)}
        l2_error = dof_handler.l2_error(u_k,exact,quad_order=5)

        print(f"\n  Relative L2 Error at t={t_np1:.2f}: {l2_error:.4e}")
        assert l2_error < 1e-2, f"Error {l2_error:.2e} is too high. Expected < 1e-2, got {l2_error:.2e}."
        
        u_n.nodal_values[:] = u_k.nodal_values[:]

    print("\nNonlinear time-dependent advection-diffusion test passed successfully!")
