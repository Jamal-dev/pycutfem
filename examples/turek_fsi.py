#!/usr/bin/env python
# coding: utf-8

# # Test case 2D-2 (unsteady)

# In[1]:


import numpy as np
import time
import scipy.sparse.linalg as sp_la
import matplotlib.pyplot as plt
import numba
import os

# --- Numba configuration ---
# try:
#     num_cores = os.cpu_count()
#     numba.set_num_threads(num_cores)
#     print(f"Numba is set to use {numba.get_num_threads()} threads.")
# except (ImportError, AttributeError):
#     print("Numba not found or configured. Running in pure Python mode.")

# --- Core pycutfem imports ---
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.utils.domain_manager import get_domain_bitset

# --- UFL-like imports ---
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    TrialFunction, TestFunction, VectorTrialFunction, VectorTestFunction,
    Function, VectorFunction, Constant, grad, inner, dot, div, jump, avg, FacetNormal, CellDiameter
)
from pycutfem.ufl.measures import dx, dS, dGhost, dInterface
from pycutfem.ufl.forms import BoundaryCondition, Equation
from pycutfem.solvers.nonlinear_solver import NewtonSolver, NewtonParameters, TimeStepperParameters
from pycutfem.ufl.compilers import FormCompiler


# In[2]:


# ============================================================================
#    1. BENCHMARK PROBLEM SETUP
# ============================================================================
print("--- Setting up the Turek benchmark (2D-2) for flow around a cylinder ---")

# --- Geometry and Fluid Properties ---
H = 0.41  # Channel height
L = 2.2   # Channel length
D = 0.1   # Cylinder diameter
c_x, c_y = 0.2, 0.2  # Cylinder center
rho_f = 1.0  # Density for fluid
rho_s = 1.0  # Density for solid
mu_f = 1e-3  # Viscosity for fluid
U_mean = 1.0 # Mean inflow velocity
# Lame coefficients for solid
_lambda_s = 0.5e6 # Lame's first parameter for solid
_mu_s = 2.0e6   # Lame's second parameter for solid
Re = rho_f * U_mean * D / mu_f
print(f"Reynolds number (Re): {Re:.2f}")


# In[3]:


from pycutfem.utils.adaptive_mesh_ls_numba import structured_quad_levelset_adaptive
# from pycutfem.utils.adaptive_mesh import structured_quad_levelset_adaptive
# --- Mesh ---
# A finer mesh is needed for this benchmark
NX, NY = 20, 20
# NX, NY = 50, 60
poly_order = 2
level_set = CircleLevelSet(center=(c_x, c_y), radius=D/2.0 ) # needs to correct the radius, also cx modified for debugging

# nodes, elems, _, corners = structured_quad(L, H, nx=NX, ny=NY, poly_order=poly_order)

nodes, elems, edges, corners = structured_quad_levelset_adaptive(
        Lx=L, Ly=H, nx=NX, ny=NY, poly_order=poly_order,
        level_set=CircleLevelSet(center=(c_x, c_y), radius=(D/2.0+0.1*D/2.0) ),
        max_refine_level=1)          # add a single halo, nothing else
mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=poly_order)

# ============================================================================
#    2. BOUNDARY CONDITIONS
# ============================================================================

# --- Tag Boundaries ---
bc_tags = {
    'inlet':  lambda x, y: np.isclose(x, 0),
    'outlet': lambda x, y: np.isclose(x, L),
    'walls':  lambda x, y: np.isclose(y, 0) | np.isclose(y, H),
}


# --- Define Parabolic Inflow Profile ---
def parabolic_inflow(x, y):
    return 4 * U_mean * y * (H - y) / (H**2)

# --- Define Boundary Conditions List ---
bcs = [
    BoundaryCondition('u_pos_x', 'dirichlet', 'inlet', parabolic_inflow),
    BoundaryCondition('u_pos_y', 'dirichlet', 'inlet', lambda x, y: 0.0),
    BoundaryCondition('u_pos_x', 'dirichlet', 'walls', lambda x, y: 0.0),
    BoundaryCondition('u_pos_y', 'dirichlet', 'walls', lambda x, y: 0.0),
    # No-slip on the cylinder is handled by the CutFEM formulation
    # "Do-nothing" at the outlet is the natural BC
]

# Homogeneous BCs for Jacobian assembly
bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]



# In[ ]:


# --- Level Set for the Cylinder Obstacle ---


mesh.classify_elements(level_set)
mesh.classify_edges(level_set)
mesh.build_interface_segments(level_set=level_set)
mesh.tag_boundary_edges(bc_tags)

# --- Define Domains with BitSets ---
fluid_domain = get_domain_bitset(mesh, "element", "outside")
solid_domain = get_domain_bitset(mesh, "element", "inside")
cut_domain = get_domain_bitset(mesh, "element", "cut")
ghost_edges = get_domain_bitset(mesh, "edge", "ghost")
fluid_interface_domain = fluid_domain | cut_domain
solid_interface_domain = solid_domain | cut_domain
solid_ghost_edges = mesh.edge_bitset('ghost_neg')  | mesh.edge_bitset('interface') #| mesh.edge_bitset('ghost_both')
fluid_ghost_edges = mesh.edge_bitset('ghost_pos')  | mesh.edge_bitset('interface') #| mesh.edge_bitset('ghost_both')

# --- Finite Element Space and DofHandler ---
# Taylor-Hood elements (Q2 for velocity, Q1 for pressure)
poly_order = 2
hessian_ghost_enabled = False
mixed_element = MixedElement(mesh, field_specs={'u_pos_x': poly_order, 'u_pos_y': poly_order, 
                                                'p_pos_': poly_order - 1,
                                                'vs_neg_x':poly_order - 1, 'vs_neg_y':poly_order - 1,
                                                'd_neg_x':poly_order - 1,'d_neg_y':poly_order - 1,
                                                })
dof_handler = DofHandler(mixed_element, method='cg')
# dof_handler.info()

print(f"Number of interface edges: {mesh.edge_bitset('interface').cardinality()}")
print(f"Number of ghost edges: {ghost_edges.cardinality()}")
print(f"Number of cut elements: {cut_domain.cardinality()}")
print(f"Number of fluid elements: {fluid_domain.cardinality()}")
print(f"Number of solid elements: {solid_domain.cardinality()}")
print(f"Number of solid ghost edges: {solid_ghost_edges.cardinality()}")
print(f"Number of fluid ghost edges: {fluid_ghost_edges.cardinality()}")


# In[ ]:


dof_handler.tag_dofs_from_element_bitset("inactive", "u_pos_x", "inside", strict=True)
dof_handler.tag_dofs_from_element_bitset("inactive", "u_pos_y", "inside", strict=True)
dof_handler.tag_dofs_from_element_bitset("inactive", "p_pos_", "inside", strict=True)

dof_handler.tag_dofs_from_element_bitset("inactive", "vs_neg_x", "outside", strict=True)
dof_handler.tag_dofs_from_element_bitset("inactive", "vs_neg_y", "outside", strict=True)
dof_handler.tag_dofs_from_element_bitset("inactive", "d_neg_x", "outside", strict=True)
dof_handler.tag_dofs_from_element_bitset("inactive", "d_neg_y", "outside", strict=True)



# In[ ]:


# contraining the displacement and velocity of solid to zero for cyclinder center
# 1. Define the target point.
target_point = np.array([c_x,c_y])

# 2. Get all node IDs that have a pressure DOF associated with them.
pin_dofs = dof_handler.get_field_slice('vs_neg_x')
pin_node_ids = np.array([dof_handler._dof_to_node_map[dof][1] for dof in pin_dofs])

# 3. Get the coordinates of ONLY these pressure-carrying nodes.
pin_node_coords = mesh.nodes_x_y_pos[pin_node_ids]

# 4. Find the node closest to the target point WITHIN this restricted set.
distances = np.linalg.norm(pin_node_coords - target_point, axis=1)
local_index = np.argmin(distances)

# 5. Get the global ID and actual coordinates of that specific pressure node.
closest_p_node_id = pin_node_ids[local_index]
actual_pin_coords = mesh.nodes_x_y_pos[closest_p_node_id]
print(f"Pinning pressure at the node closest to {target_point}, found at {actual_pin_coords}")
for field in ['vs_neg_x', 'vs_neg_y', 'd_neg_x', 'd_neg_y']:
    name = f'pinning_{field}'
    dof_handler.tag_dof_by_locator(
        name, field,
        locator=lambda x, y, x0=actual_pin_coords[0], y0=actual_pin_coords[1]:
            np.isclose(x, x0) and np.isclose(y, y0),
        find_first=True
    )
    bcs.append(
        BoundaryCondition(field, 'dirichlet', name, lambda x, y: 0.0)
    )
    bcs_homog.append(
        BoundaryCondition(field, 'dirichlet', name, lambda x, y: 0.0)
    )


# In[7]:


print(f'Total dirchlet dofs: {len(dof_handler.get_dirichlet_data(bcs))}')


# In[8]:


from pycutfem.io.visualization import plot_mesh_2
fig, ax = plt.subplots(figsize=(15, 30))
plot_mesh_2(mesh, ax=ax, level_set=level_set, show=True, 
              plot_nodes=False, elem_tags=False, edge_colors=True, plot_interface=False,resolution=300)


# In[ ]:


# ============================================================================
#    3. UFL FORMULATION WITH GHOST PENALTY
# ============================================================================
print("\n--- Defining the UFL weak form for Navier-Stokes with ghost penalty ---")

# --- Function Spaces and Functions ---
velocity_fluid_space = FunctionSpace(name="velocity_fluid", field_names=['u_pos_x', 'u_pos_y'],dim=1, side='+')
pressure_fluid_space = FunctionSpace(name="pressure_fluid", field_names=['p_pos_'], dim=0, side='+')
velocity_solid_space = FunctionSpace(name="velocity_solid", field_names=['vs_neg_x', 'vs_neg_y'], dim=1, side='-')
displacement_space = FunctionSpace(name="displacement", field_names=['d_neg_x', 'd_neg_y'], dim=1, side='-')
# Trial and Test functions
du_f = VectorTrialFunction(space=velocity_fluid_space, dof_handler=dof_handler)
dp_f = TrialFunction(name='trial_pressure_fluid', field_name='p_pos_', dof_handler=dof_handler)
du_s = VectorTrialFunction(space=velocity_solid_space, dof_handler=dof_handler)
ddisp_s = VectorTrialFunction(space=displacement_space, dof_handler=dof_handler)
test_vel_f = VectorTestFunction(space=velocity_fluid_space, dof_handler=dof_handler)
test_q_f = TestFunction(name='test_pressure_fluid', field_name='p_pos_', dof_handler=dof_handler)
test_vel_s = VectorTestFunction(space=velocity_solid_space, dof_handler=dof_handler)
test_disp_s = VectorTestFunction(space=displacement_space, dof_handler=dof_handler)

# Solution functions at current (k) and previous (n) time steps
uf_k = VectorFunction(name="u_f_k", field_names=['u_pos_x', 'u_pos_y'], dof_handler=dof_handler)
pf_k = Function(name="p_f_k", field_name='p_pos_', dof_handler=dof_handler)
uf_n = VectorFunction(name="u_f_n", field_names=['u_pos_x', 'u_pos_y'], dof_handler=dof_handler)
pf_n = Function(name="p_f_n", field_name='p_pos_', dof_handler=dof_handler)
us_k = VectorFunction(name="u_s_k", field_names=['vs_neg_x', 'vs_neg_y'], dof_handler=dof_handler)
us_n = VectorFunction(name="u_s_n", field_names=['vs_neg_x', 'vs_neg_y'], dof_handler=dof_handler)
disp_k = VectorFunction(name="disp_k", field_names=['d_neg_x', 'd_neg_y'], dof_handler=dof_handler)
disp_n = VectorFunction(name="disp_n", field_names=['d_neg_x', 'd_neg_y'], dof_handler=dof_handler)

# --- Parameters ---
dt = Constant(0.2)
theta = Constant(1.0) # Crank-Nicolson
mu_f_const = Constant(mu_f)
rho_f_const = Constant(rho_f)
rho_s_const = Constant(rho_s)
mu_s = Constant(_mu_s) # Lame's second parameter for solid
lambda_s = Constant(_lambda_s)  # Lame's first parameter for solid

uf_k.nodal_values.fill(0.0); pf_k.nodal_values.fill(0.0)
uf_n.nodal_values.fill(0.0); pf_n.nodal_values.fill(0.0)
us_k.nodal_values.fill(0.0); us_n.nodal_values.fill(0.0)
disp_k.nodal_values.fill(0.0); disp_n.nodal_values.fill(0.0)
dof_handler.apply_bcs(bcs, uf_n, pf_n, us_n, disp_n)


# In[10]:


uf_n.plot()


# In[ ]:


from pycutfem.ufl.expressions import Derivative, FacetNormal, trace, Jump, Hessian
n = FacetNormal()                    # vector expression (n_x, n_y)

def _dn(expr):
    """Normal derivative  n·∇expr  on an (interior) edge."""
    Dx = Derivative(expr, 1, 0)
    Dy = Derivative(expr, 0, 1)
    _ = Dx + Dy
    return n[0]*Dx + n[1]*Dy

def grad_inner(u, v):
    """⟨∂ₙu, ∂ₙv⟩  (scalar or 2‑D vector)."""
    if getattr(u, "num_components", 1) == 1:      # scalar
        return _dn(u) * _dn(v)

    if u.num_components == v.num_components == 2: # vector
        return _dn(u[0]) * _dn(v[0]) + _dn(u[1]) * _dn(v[1])

    raise ValueError("grad_inner supports only scalars or 2‑D vectors.")

def hessian_inner(u, v):
    if getattr(u, "num_components", 1) == 1:      # scalar
        return _hess_comp(u, v)

    # vector: sum component-wise
    return sum(_hess_comp(u[i], v[i]) for i in range(u.num_components))


def _hess_comp(a, b):
    return (Derivative(a,2,0)*Derivative(b,2,0) +
            2*Derivative(a,1,1)*Derivative(b,1,1) +
            Derivative(a,0,2)*Derivative(b,0,2))
def grad_inner_jump(u, v):
    """⟨∂ₙu, ∂ₙv⟩  (scalar or 2‑D vector)."""
    a = dot(jump(grad(u)), n)
    b = dot(jump(grad(v)), n)
    return inner(a, b)
def hess_inner_jump(u, v):
    a = dot(n, dot(jump(Hessian(u)), n))
    b = dot(n, dot(jump(Hessian(v)), n))
    return inner(a, b)

qvol = 6
dx_fluid  = dx(defined_on=fluid_interface_domain,level_set=level_set,
               metadata={"q":qvol, "side": "+"})               # volume
dx_solid  = dx(defined_on=solid_interface_domain,level_set=level_set,
               metadata={"q":qvol, "side": "-"})               # volume
dΓ        = dInterface(defined_on=mesh.element_bitset('cut'), level_set=level_set, 
                       metadata={"q":qvol+2,'derivs': {(0,0),(0,1),(1,0)}})   # interior surface
if hessian_ghost_enabled:
    dG_fluid       = dGhost(defined_on=fluid_ghost_edges, level_set=level_set,
                            metadata={"q":qvol,'derivs': {(0,0),(0,1),(1,0),(2,0),(0,2),(1,1)}})  # ghost fluid surface
    dG_solid       = dGhost(defined_on=solid_ghost_edges, level_set=level_set,
                            metadata={"q":qvol,'derivs': {(0,0),(0,1),(1,0),(2,0),(0,2),(1,1)}})  # ghost solid surface
else:
    dG_fluid       = dGhost(defined_on=fluid_ghost_edges, level_set=level_set,
                          metadata={"q":qvol,'derivs': {(0,1),(1,0)}})  # ghost fluid surface
    dG_solid       = dGhost(defined_on=solid_ghost_edges, level_set=level_set,
                            metadata={"q":qvol,'derivs': {(0,1),(1,0)}})  # ghost solid surface

cell_h  = CellDiameter() # length‑scale per element
beta_N  = Constant(20.0 * poly_order**2)      # Nitsche penalty (tweak)

def epsilon_f(u):
    return 0.5 * (grad(u) + grad(u).T)

def epsilon_s(u):
    return 0.5 * (grad(u) + grad(u).T)

I2 = Constant(np.eye(2))

def solid_stress(u):
    return 2.0 * mu_s * epsilon_s(u) + lambda_s * trace(epsilon_s(u)) * I2

def traction_fluid(u_vec, p_scal):
    return -2.0 * mu_f_const * dot(epsilon_f(u_vec), n) + p_scal * n

def traction_solid(disp):
    return dot(solid_stress(disp), n)

kappa_pos = Constant(1.0)
kappa_neg = Constant(1.0)

jump_vel_trial = du_f - du_s
jump_vel_test = test_vel_f - test_vel_s
jump_vel_res = uf_k - us_k

avg_flux_trial = (
    kappa_pos * traction_fluid(du_f, dp_f)
    + kappa_neg * traction_solid(ddisp_s)
)

avg_flux_test = (
    kappa_pos * traction_fluid(test_vel_f, test_q_f)
    + kappa_neg * traction_solid(test_disp_s)
)

avg_flux_res = (
    kappa_pos * traction_fluid(uf_k, pf_k)
    + kappa_neg * traction_solid(disp_k)
)

# --- Jacobian contribution on Γsolid --------------------------------
J_int = (
    dot(avg_flux_trial, jump_vel_test)
    + dot(avg_flux_test, jump_vel_trial)
    + (beta_N * mu_f_const / cell_h) * dot(jump_vel_trial, jump_vel_test)
) * dΓ

# --- Residual contribution on Γsolid --------------------------------
R_int = (
    dot(avg_flux_res, jump_vel_test)
    + dot(avg_flux_test, jump_vel_res)
    + (beta_N * mu_f_const / cell_h) * dot(jump_vel_res, jump_vel_test)
) * dΓ

# volume -------------------fluid--------------------------------
a_vol_f = ( rho_f/dt*dot(du_f,test_vel_f)
          + theta*rho_f*dot(dot(grad(uf_k), du_f), test_vel_f)
          + theta*rho_f*dot(dot(grad(du_f), uf_k), test_vel_f)
          + theta*mu_f*inner(grad(du_f), grad(test_vel_f))
          - dp_f*div(test_vel_f) + test_q_f*div(du_f) ) * dx_fluid


r_vol_f = ( rho_f*dot(uf_k-uf_n, test_vel_f)/dt
          + theta*rho_f*dot(dot(grad(uf_k), uf_k), test_vel_f)
          + (1-theta)*rho_f*dot(dot(grad(uf_n), uf_n), test_vel_f)
          + theta*mu_f*inner(grad(uf_k), grad(test_vel_f))
          + (1-theta)*mu_f*inner(grad(uf_n), grad(test_vel_f))
          - pf_k*div(test_vel_f) + test_q_f*div(uf_k) ) * dx_fluid

# volume -------------------solid--------------------------------
a_vol_s = (
    rho_s / dt * dot(du_s, test_vel_s)
    + theta * inner(solid_stress(ddisp_s), epsilon_s(test_vel_s))
) * dx_solid


r_vol_s = (
    rho_s / dt * dot(us_k - us_n, test_vel_s)
    + theta * inner(solid_stress(disp_k), epsilon_s(test_vel_s))
    + (1 - theta) * inner(solid_stress(disp_n), epsilon_s(test_vel_s))
) * dx_solid
# --- solid disp and solid velocity constraint --------------------------------
# \frac{D}{Dt} disp_s = \frac{\partial}{\partialt} disp_s + (v_s^k \cdot \nabla) disp_s
a_svc = (
    dot(ddisp_s, test_disp_s) / dt
    - theta * dot(du_s, test_disp_s)
) * dx_solid
r_svc = (
    dot(disp_k - disp_n, test_disp_s) / dt
    - theta * dot(us_k, test_disp_s)
    - (1 - theta) * dot(us_n, test_disp_s)
) * dx_solid

# ghost stabilisation (add exactly as in your Poisson tests) --------
penalty_val = 1e-3
penalty_grad = 1e-3
penalty_hess = 1e-3
gamma_v = Constant(penalty_val * poly_order**2)
gamma_v_grad= Constant(penalty_grad * poly_order**2)
gamma_p  = Constant(penalty_val * poly_order**1)
gama_p_grad = Constant(penalty_grad * poly_order**1)
def g_v_f(gamma, phi_1, phi_2):
    if hessian_ghost_enabled:
        return (
            gamma * (
                cell_h * grad_inner_jump(phi_1, phi_2)
                + cell_h**3.0/4.0   * hess_inner_jump(phi_1, phi_2)
            )
        )
    else:
        return (
            gamma * (
                cell_h * grad_inner_jump(phi_1, phi_2)
            )
        )
def g_p(gamma, phi_1, phi_2):
    return (
        gamma * (
            cell_h**3.0 * grad_inner_jump(phi_1, phi_2)
        )
    )
def g_v_s(gamma, phi_1, phi_2):
    return (
        gamma * (
            cell_h**3.0 * grad_inner_jump(phi_1, phi_2)
        )
    )
def g_disp_s(gamma, phi_1, phi_2):
    return (
        gamma * (
            cell_h * grad_inner_jump(phi_1, phi_2)
        )
    )


a_stab = (
    (
    Constant(2.0) * mu_f_const * g_v_f(gamma_v, du_f, test_vel_f)
    + g_p(gamma_p, dp_f, test_q_f)
    ) * dG_fluid + 
    (
    rho_s_const * g_v_s(gamma_v, du_s, test_vel_s)
    + Constant(2.0) * mu_s * g_disp_s(gamma_v_grad, ddisp_s, test_disp_s)
    ) * dG_solid
)
r_stab = (
    (
    Constant(2.0) * mu_f_const * g_v_f(gamma_v, uf_k, test_vel_f)
    + g_p(gamma_p, pf_k, test_q_f)
    ) * dG_fluid + 
    (
    rho_s_const * g_v_s(gamma_v, us_k, test_vel_s)
    + Constant(2.0) * mu_s * g_disp_s(gamma_v_grad, disp_k, test_disp_s)
    ) * dG_solid
)
# complete Jacobian and residual -----------------------------------
jacobian_form  = a_vol_f + J_int + a_vol_s + a_svc + a_stab
residual_form  = r_vol_f + R_int + r_vol_s + r_svc + r_stab
# residual_form  = dot(  Constant(np.array([0.0, 0.0]),dim=1), v) * dx
# jacobian_form  = a_vol_f + a_vol_s + a_svc + a_stab
# residual_form  = r_vol_f + r_vol_s + r_svc + r_stab





# In[12]:


# get_ipython().system('rm ~/.cache/pycutfem_jit/*')


# In[13]:


# from pycutfem.ufl.forms import assemble_form
# K,F=assemble_form(jacobian_form==-residual_form, dof_handler=dof_handler, bcs=bcs_homog)
# print(np.linalg.norm(F, ord=np.inf))


# In[14]:


from pycutfem.solvers.nonlinear_solver import NewtonSolver, NewtonParameters, TimeStepperParameters#, AdamNewtonSolver
# from pycutfem.solvers.aainhb_solver import AAINHBSolver           # or get_solver("aainhb")

uf_k.nodal_values.fill(0.0); pf_k.nodal_values.fill(0.0)
uf_n.nodal_values.fill(0.0); pf_n.nodal_values.fill(0.0)
us_k.nodal_values.fill(0.0); us_n.nodal_values.fill(0.0)
disp_k.nodal_values.fill(0.0); disp_n.nodal_values.fill(0.0)
dof_handler.apply_bcs(bcs, uf_n, pf_n, us_n, disp_n)
dof_handler.apply_bcs(bcs, uf_k, pf_k, us_k, disp_k)

# build residual_form, jacobian_form, dof_handler, mixed_element, bcs, bcs_homog …
time_params = TimeStepperParameters(dt=dt.value,max_steps=36 ,stop_on_steady=True, steady_tol=1e-6, theta= theta.value)

solver = NewtonSolver(
    residual_form, jacobian_form,
    dof_handler=dof_handler,
    mixed_element=mixed_element,
    bcs=bcs, bcs_homog=bcs_homog,
    newton_params=NewtonParameters(newton_tol=1e-6, line_search=True),
)
# primary unknowns
functions      = [uf_k, pf_k, us_k, disp_k]
prev_functions = [uf_n, pf_n, us_n, disp_n]
# solver = AdamNewtonSolver(
#     residual_form, jacobian_form,
#     dof_handler=dof_handler,
#     mixed_element=mixed_element,
#     bcs=bcs, bcs_homog=bcs_homog,
#     newton_params=NewtonParameters(newton_tol=1e-6)
# )
# solver = AAINHBSolver(
#     residual_form, jacobian_form,
#     dof_handler=dof_handler,
#     mixed_element=mixed_element,
#     bcs=bcs, bcs_homog=bcs_homog,
#     newton_params=NewtonParameters(newton_tol=1e-6),
# )



solver.solve_time_interval(functions=functions,
                           prev_functions= prev_functions,
                           time_params=time_params,)


# In[ ]:


uf_n.plot(kind="streamline",
         density=4.0,
         linewidth=0.8,
         cmap="plasma",
         title="Turek-Schafer",background = False)
