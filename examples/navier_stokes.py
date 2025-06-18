#!/usr/bin/env python
# coding: utf-8

# # Navier Stokes Equatoin

# In[1]:


import sympy
from ufl.symops import SymbolicOps as so

# 1. Define symbolic constants
rho_sympy, dt_sympy, theta_sympy, mu_sympy = sympy.symbols('rho Delta_t theta mu')

# 2. Define all symbolic fields using the helper class
# Solution at current Newton iteration k
u_k = so.vector_field('u_k')
p_k = so.scalar_field('p_k')

# Solution from previous time step n
u_n = so.vector_field('u_n')
p_n = so.scalar_field('p_n')

# Forcing term
f_n = so.vector_field('f_n')
f_np1 = so.vector_field('f_np1')

# Test functions
v = so.vector_field('v_test')
q = so.scalar_field('q_test')

# Perturbation functions for Jacobian (TrialFunctions)
du = so.vector_field('du_trial')
dp = so.scalar_field('dp_trial')


# In[2]:


# 3. Define the spatial operator F(u, p, f, v, q) from your formula.
# This function defines the weak form of the steady-state equations.
def F_spatial_integrand(u, p, f, v_test, q_test):
    # Convection term: ((u ⋅ ∇)u) ⋅ v
    # This is the correct weak form after integration by parts of the standard
    # This explicit construction is more robust and readable.
    convection_vector = sympy.Matrix([
        so.dot(u, so.grad(u[0])),  # u ⋅ ∇(u_x)
        so.dot(u, so.grad(u[1]))   # u ⋅ ∇(u_y)
    ])
    convection = rho_sympy * so.dot(convection_vector, v_test)
    
    # Diffusion term: μ (∇u : ∇v)
    diffusion = mu_sympy * so.inner(so.grad(u), so.grad(v_test))
    
    # Pressure term: -p (∇ ⋅ v)
    pressure = -p * so.div(v_test)
    
    # Continuity term: q (∇ ⋅ u)
    continuity = q_test * so.div(u)
    
    # Source term: f ⋅ v
    source = so.dot(f, v_test)
    
    return convection + diffusion + pressure + continuity - source

# 4. Build the full time-discretized residual integrand `R`
# This is the "semi-discrete" equation before linearization.
# R(u_k, p_k) = (u_k - u_n)/dt + theta*F(u_k, p_k) + (1-theta)*F(u_n, p_n) = 0
time_term = so.dot((u_k - u_n) / dt_sympy, v)
F_term_k = F_spatial_integrand(u_k, p_k, f_np1, v, q)
F_term_n = F_spatial_integrand(u_n, p_n, f_n, v, q)
R_newton_integrand = time_term + theta_sympy * F_term_k + (1 - theta_sympy) * F_term_n

# 5. Automatically compute the Jacobian using the Gâteaux derivative
solution_variables = [
    (u_k, du),
    (p_k, dp)
]
J_newton_integrand = so.compute_gateaux_derivative(R_newton_integrand, solution_variables)


# In[3]:


R_newton_integrand


# In[4]:


J_newton_integrand


# In[5]:


from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad

# --- UFL-like imports ---
from ufl.functionspace import FunctionSpace
from ufl.expressions import (
    TrialFunction, TestFunction, VectorTrialFunction, VectorTestFunction,
    grad, inner, dot, div, Constant, Function, VectorFunction
)
from ufl.measures import dx
from ufl.forms import BoundaryCondition, assemble_form
import numpy as np
# 1. Setup Meshes and DofHandler for Q2-Q1 elements
L, H = 1.0, 1.0  # Domain size
NX,NY = 6, 6  # Number of elements in x and y directions
nodes_q2, elems_q2, _, corners_q2 = structured_quad(L, H, nx=NX, ny=NY, poly_order=2)
mesh_q2 = Mesh(nodes=nodes_q2, element_connectivity=elems_q2,
                elements_corner_nodes=corners_q2, element_type="quad", poly_order=2)

nodes_q1, elems_q1, _, corners_q1 = structured_quad(L, H, nx=NX, ny=NY, poly_order=1)
mesh_q1 = Mesh(nodes=nodes_q1, element_connectivity=elems_q1,
                elements_corner_nodes=corners_q1, element_type="quad", poly_order=1)

fe_map = {
    'ux': mesh_q2, 'uy': mesh_q2, 'p': mesh_q1
}
dof_handler = DofHandler(fe_map, method='cg')

bc_tags = {
    'bottom_wall': lambda x, y: np.isclose(y, 0),
    'left_wall':   lambda x, y: np.isclose(x, 0),
    'right_wall':  lambda x, y: np.isclose(x, L),
    'top_lid':     lambda x, y: np.isclose(y, H)
}
mesh_q2.tag_boundary_edges(bc_tags)
mesh_q1.nodes_list[0].tag = 'pressure_pin_point'

class DataBC:
    Um = 1.5
    t = 0.0
    H = H
bcs = [
    BoundaryCondition('ux', 'dirichlet', 'left_wall',   lambda x, y: 4 * DataBC.Um * y * (DataBC.H - y) / (DataBC.H ** 2)),
    BoundaryCondition('uy', 'dirichlet', 'left_wall',   lambda x, y: 0.0),
    BoundaryCondition('ux', 'dirichlet', 'right_wall',  lambda x, y: 0.0),
    BoundaryCondition('uy', 'dirichlet', 'right_wall',  lambda x, y: 0.0),
    BoundaryCondition('ux', 'dirichlet', 'bottom_wall', lambda x, y: 0.0),
    BoundaryCondition('uy', 'dirichlet', 'bottom_wall', lambda x, y: 0.0),
    BoundaryCondition('ux', 'dirichlet', 'top_lid',     lambda x, y: 0.0),
    BoundaryCondition('uy', 'dirichlet', 'top_lid',     lambda x, y: 0.0),
    BoundaryCondition('p', 'dirichlet', 'pressure_pin_point', lambda x, y: 0.0)
]

# 2. Define UFL symbols using vector notation
velocity_space = FunctionSpace("velocity", ['ux', 'uy'], dim=1)
pressure_space = FunctionSpace("pressure", ['p'], dim=0)

# du_ufl = VectorTrialFunction(velocity_space)
# v_ufl = VectorTestFunction(velocity_space)
du__x_ufl = TrialFunction(velocity_space.field_names[0])
du__y_ufl = TrialFunction(velocity_space.field_names[1])
v__x_ufl = TestFunction(velocity_space.field_names[0])
v__y_ufl = TestFunction(velocity_space.field_names[1])
dp_ufl = TrialFunction(pressure_space.field_names[0])
q_ufl = TestFunction(pressure_space.field_names[0])
# u_k_ufl = VectorFunction(name="u_k", field_names=['ux', 'uy'],nodal_values=np.zeros((len(mesh_q2.nodes),2)))
u_k__x_ufl = Function(name="u_k_x", field_name='ux', nodal_values=np.zeros(dof_handler.total_dofs))
u_k__y_ufl = Function(name="u_k_y", field_name='uy', nodal_values=np.zeros(dof_handler.total_dofs))
p_k_ufl = Function(name="pk",field_name='p', nodal_values=np.zeros(dof_handler.total_dofs))
# u_n_ufl = VectorFunction(name="u_n", field_names=['ux', 'uy'], nodal_values=np.zeros((len(mesh_q2.nodes),2)))
u_n__x_ufl = Function(name="u_n_x", field_name='ux', nodal_values=np.zeros(dof_handler.total_dofs))
u_n__y_ufl = Function(name="u_n_y", field_name='uy', nodal_values=np.zeros(dof_handler.total_dofs))
p_n_ufl = Function(name="p_n",field_name='p', nodal_values=np.zeros(dof_handler.total_dofs))
# f_n_ufl = VectorFunction(name="f_n", field_names=['ux', 'uy'],nodal_values=np.zeros((len(mesh_q2.nodes),2)))
f_n__x_ufl = Function(name="f_n_x", field_name='ux', nodal_values=np.zeros(dof_handler.total_dofs))
f_n__y_ufl = Function(name="f_n_y", field_name='uy', nodal_values=np.zeros(dof_handler.total_dofs))
# f_np1_ufl = VectorFunction(name="f_np1", field_names=['ux', 'uy'],nodal_values=np.zeros((len(mesh_q2.nodes),2)))
f_np1__x_ufl = Function(name="f_np1_x", field_name='ux', nodal_values=np.zeros(dof_handler.total_dofs))
f_np1__y_ufl = Function(name="f_np1_y", field_name='uy', nodal_values=np.zeros(dof_handler.total_dofs))
rho = Constant(1.0)
dt = Constant(0.1)
theta = Constant(0.5)
mu = Constant(1.0e-2)

# This map is the bridge between SymPy and UFL
symbol_map = {
    # Test Functions (v, q)
    v[0]: v__x_ufl,  # v_test_x(x,y) -> TestFunction('u_k_x')
    v[1]: v__y_ufl,  # v_test_y(x,y) -> TestFunction('u_k_y')
    q:    q_ufl,     # q_test(x,y) -> TestFunction('q_test')
    # Trial Functions (du, dp)
    du[0]: du__x_ufl, # du_trial_x(x,y) -> TrialFunction('u_k_x')
    du[1]: du__y_ufl, # du_trial_y(x,y) -> TrialFunction('u_k_y')
    dp:    dp_ufl,    # dp_trial(x,y) -> TrialFunction('dp_trial')
    # Data Functions (u_k, p_k, u_n, etc.)
    u_k[0]: u_k__x_ufl,  # u_k_x(x,y) -> Function('u_k_x')
    u_k[1]:  u_k__y_ufl,  # u_k_y(x,y) -> Function('u_k_y')
    p_k:    p_k_ufl, 
    u_n[0]: u_n__x_ufl,
    u_n[1]: u_n__y_ufl,
    p_n:    p_n_ufl,
    f_n[0]:   f_n__x_ufl,
    f_n[1]:   f_n__y_ufl,
    f_np1[0]: f_np1__x_ufl,
    f_np1[1]: f_np1__y_ufl,
    rho_sympy:    rho,
    dt_sympy:     dt,
    theta_sympy:  theta,
    mu_sympy:     mu}


# In[6]:


from ufl.compilers import SymPyToUFLVisitor
from ufl.expressions import Prod, Sum, Grad, DivOperation, Div
import functools


# 1. SEPARATE: Assign Jacobian to LHS, and negative Residual to RHS
a_sympy = J_newton_integrand.expand()
L_sympy = (-R_newton_integrand).expand() # Note the negative sign!

# 2. MAP: Instantiate the visitor with the map
visitor = SymPyToUFLVisitor(symbol_map=symbol_map)

# 3. TRANSLATE: Visit the SymPy expression trees to get UFL trees
a_ufl = visitor.visit(a_sympy)
L_ufl = visitor.visit(L_sympy)


# In[7]:


print(a_ufl)


# In[8]:


print(L_ufl)


# In[9]:


equation = (a_ufl * dx() == L_ufl * dx())


# In[10]:


import time
import scipy.sparse.linalg as sp_la
T_end = 1.0
dt_val = dt.value
num_steps = int(T_end / dt_val)
newton_tol = 1e-6
max_newton_iter = 25 # Increased for potentially stiff problems

# 2. Create a second list of Boundary Conditions for the homogeneous case.
#    This is for solving the linear system for the update delta_U.
bcs_homog = [
    BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) 
    for bc in bcs
]

# 3. Initialize solution vectors
# U_k holds the combined solution vector for the current iteration (k)
U_k = np.zeros(dof_handler.total_dofs)
# U_n holds the solution from the previous time step (n)
U_n = np.zeros(dof_handler.total_dofs)

# --- Set the initial condition (at t=0) ---
# Apply the non-homogeneous boundary conditions to the initial state U_n
dirichlet_dofs_map = dof_handler.get_dirichlet_data(bcs) # Get {dof: value} map
dirichlet_dofs = np.array(list(dirichlet_dofs_map.keys()))
dirichlet_values = np.array(list(dirichlet_dofs_map.values()))
U_n[dirichlet_dofs] = dirichlet_values

dofs_ux = list(dof_handler.dof_map['ux'].values())
dofs_uy = list(dof_handler.dof_map['uy'].values())
dofs_p = list(dof_handler.dof_map['p'].values())

# update ufls as well
u_n__x_ufl.nodal_values[dofs_ux] = U_n[dofs_ux]
u_n__y_ufl.nodal_values[dofs_uy] = U_n[dofs_uy]
p_n_ufl.nodal_values[dofs_p] = U_n[dofs_p]

# update u_k_ufl as well
u_k__x_ufl.nodal_values[dofs_ux] = U_n[dofs_ux]
u_k__y_ufl.nodal_values[dofs_uy] = U_n[dofs_uy]
p_k_ufl.nodal_values[dofs_p] = U_n[dofs_p]

# Your UFL Function objects do not need to store the nodal values themselves.
# The FormCompiler will get the data from the global vector U_k via the context.
# This avoids the previous IndexError and is more efficient.

# 4. Start time-stepping
print("--- Starting time integration ---")
start_time = time.time()
for n in range(num_steps):
    current_time = (n + 1) * dt_val
    print(f"\n--- Time step {n+1}/{num_steps} | t = {current_time:.2f}s ---")

    # Initial guess for Newton's is the solution from the previous step
    U_k[:] = U_n

    # Start Newton's iterations
    for k in range(max_newton_iter):
        # Assemble the Jacobian matrix (A) and the residual vector (b).
        # CRUCIAL: We solve the linear system for the *update* (delta_U),
        # which requires HOMOGENEOUS boundary conditions.
        A, b = assemble_form(equation, 
                             dof_handler=dof_handler, 
                             bcs=bcs_homog,  # Use the zero-valued BCs here!
                             quad_order=5) # Pass the current guess

        res_norm = np.linalg.norm(b[b != 0]) # Norm of free DOFs
        print(f"  Newton Iter: {k+1} | Residual norm: {res_norm:.4e}")

        if res_norm < newton_tol:
            print(f"  Newton's method converged in {k+1} iterations.")
            break

        # Solve A * delta_U = b for the correction
        delta_U = sp_la.spsolve(A, b)

        # Apply the correction to the solution
        U_k += delta_U
        # update u_k ufl
        u_k__x_ufl.nodal_values[dofs_ux] = U_k[dofs_ux]
        u_k__y_ufl.nodal_values[dofs_uy] = U_k[dofs_uy]
        p_k_ufl.nodal_values[dofs_p] = U_k[dofs_p]

        
        # Note: There is no need to update the individual u_k__x_ufl.nodal_values
        # inside this loop. The assembler gets everything it needs from the
        # solution_vector=U_k argument.

    else: # This 'else' belongs to the for loop
        print("  Warning: Newton's method did not converge!")
        # You might want to break the time loop if one step fails
        break 

    # Update U_n for the next time step
    U_n[:] = U_k
    # Update the UFL functions for the next step
    u_n__x_ufl.nodal_values[dofs_ux] = U_n[dofs_ux]
    u_n__y_ufl.nodal_values[dofs_uy] = U_n[dofs_uy]
    p_n_ufl.nodal_values[dofs_p] = U_n[dofs_p]
    # Update the forcing term for the next step

end_time = time.time()
print(f"\n--- Simulation finished in {end_time - start_time:.2f} seconds ---")

