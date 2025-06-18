import pytest
import numpy as np
import scipy.sparse.linalg as spla
import sympy as sp

# --- Core and UFL Imports ---
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from ufl.functionspace import FunctionSpace
from ufl.expressions import (
    VectorTrialFunction, VectorTestFunction, VectorFunction,
    grad, inner, dot, Constant
)
from ufl.measures import dx
from ufl.forms import BoundaryCondition, assemble_form
from ufl.analytic import Analytic, x, y

def test_vector_heat_equation_mms():
    """
    Tests the solver for the time-dependent vector heat equation using
    the Method of Manufactured Solutions with Q2 elements.
    
    Equation: ∂u/∂t - εΔu = f
    """
    print("\n" + "="*70)
    print("Testing Time-Dependent Vector Heat Equation with Q2 Elements")
    print("="*70)

    # 1. Define Analytical Solution and Parameters using SymPy
    epsilon = 0.1
    t_sym = sp.Symbol('t')
    
    # Choose a smooth vector solution that depends on both space and time
    u_exact_sym_x = sp.cos(t_sym) * sp.sin(sp.pi * x) * sp.cos(sp.pi * y)
    u_exact_sym_y = sp.cos(t_sym) * sp.cos(sp.pi * x) * sp.sin(sp.pi * y)
    
    # Calculate the forcing term f = ∂u/∂t - εΔu
    f_sym_x = sp.diff(u_exact_sym_x, t_sym) - epsilon * (sp.diff(u_exact_sym_x, x, 2) + sp.diff(u_exact_sym_x, y, 2))
    f_sym_y = sp.diff(u_exact_sym_y, t_sym) - epsilon * (sp.diff(u_exact_sym_y, x, 2) + sp.diff(u_exact_sym_y, y, 2))

    # Lambdify expressions to create callable Python functions
    # Note: The functions now also depend on time `t`
    u_exact_func_x = sp.lambdify((x, y, t_sym), u_exact_sym_x, 'numpy')
    u_exact_func_y = sp.lambdify((x, y, t_sym), u_exact_sym_y, 'numpy')
    f_analytic_func = sp.lambdify((x, y, t_sym), [f_sym_x, f_sym_y], 'numpy')

    # 2. Setup Q2 Mesh and DofHandler
    poly_order = 2  # Q2 elements
    nodes, elems, _, corners = structured_quad(1, 1, nx=4, ny=4, poly_order=poly_order)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=poly_order)
    fe_map = {'ux': mesh, 'uy': mesh}
    dof_handler = DofHandler(fe_map, method='cg')

    # 3. Time-Stepping Loop
    T_end = 0.5
    dt_val = 0.1
    num_steps = int(T_end / dt_val)
    
    # Create UFL Function to hold the solution from the previous step (u_n)
    u_n = VectorFunction(name="u_n", field_names=['ux', 'uy'], dof_handler=dof_handler)
    
    # Initialize u_n with the exact solution at t=0
    u_n_initial_func = lambda x, y: np.array([u_exact_func_x(x, y, 0), u_exact_func_y(x, y, 0)])
    u_n.set_values_from_function(u_n_initial_func)
    
    # UFL Trial/Test functions and constants for the weak form
    u = VectorTrialFunction(FunctionSpace("velocity", ['ux', 'uy']))
    v = VectorTestFunction(FunctionSpace("velocity", ['ux', 'uy']))
    dt = Constant(dt_val)
    eps = Constant(epsilon)
    
    # These will hold the forcing term at times t_n and t_{n+1}
    f_n = VectorFunction(name="f_n", field_names=['ux', 'uy'], dof_handler=dof_handler)
    f_np1 = VectorFunction(name="f_np1", field_names=['ux', 'uy'], dof_handler=dof_handler)

    for n in range(num_steps):
        t_n = n * dt_val
        t_np1 = (n + 1) * dt_val
        print(f"\n--- Solving Time Step {n+1}/{num_steps} | t = {t_np1:.2f}s ---")

        # 4. Define Weak Form using the Theta-method
        theta = 1.0 # Use Backward Euler for stability
        
        # LHS of the system for the unknown u_k (aliased as u)
        a_form = (
            dot(u, v) / dt + 
            theta * eps * inner(grad(u), grad(v))
        ) * dx()
        
        # RHS of the system, depending on the known u_n
        L_form = (
            dot(u_n, v) / dt -
            (1.0 - theta) * eps * inner(grad(u_n), grad(v)) +
            theta * dot(f_np1, v) + 
            (1.0 - theta) * dot(f_n, v)
        ) * dx()
        
        equation = a_form == L_form

        # 5. Populate time-dependent functions and define BCs for t_{n+1}
        f_n.set_values_from_function(lambda x, y: f_analytic_func(x, y, t_n))
        f_np1.set_values_from_function(lambda x, y: f_analytic_func(x, y, t_np1))
        
        bcs = [
            BoundaryCondition('ux', 'dirichlet', 'boundary', lambda x, y: u_exact_func_x(x, y, t_np1)),
            BoundaryCondition('uy', 'dirichlet', 'boundary', lambda x, y: u_exact_func_y(x, y, t_np1))
        ]
        mesh.tag_boundary_edges({'boundary': lambda x, y: True})

        # 6. Assemble and Solve the linear system
        A, b = assemble_form(equation, dof_handler, bcs=bcs, quad_order=5)
        u_k_vec = spla.spsolve(A, b)

        # 7. Verify Solution for the current time step
        node_coords = mesh.nodes_x_y_pos
        exact_sol_vec = np.zeros_like(u_k_vec)
        exact_sol_vec[dof_handler.get_field_slice('ux')] = u_exact_func_x(node_coords[:, 0], node_coords[:, 1], t_np1)
        exact_sol_vec[dof_handler.get_field_slice('uy')] = u_exact_func_y(node_coords[:, 0], node_coords[:, 1], t_np1)
        
        l2_error = np.linalg.norm(u_k_vec - exact_sol_vec) / np.linalg.norm(exact_sol_vec)
        print(f"  > Relative L2 Error at t={t_np1:.2f}: {l2_error:.4e}")
        assert l2_error < 1e-3

        # 8. Update for next step
        u_n.nodal_values[:] = u_k_vec
        
    print("\nVector heat equation test passed successfully!")

if __name__ == "__main__":
    test_vector_heat_equation_mms()
