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
try:
    num_cores = os.cpu_count()
    numba.set_num_threads(num_cores)
    print(f"Numba is set to use {numba.get_num_threads()} threads.")
except (ImportError, AttributeError):
    print("Numba not found or configured. Running in pure Python mode.")

# --- Core pycutfem imports ---
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.utils.domain_manager import get_domain_bitset
from pycutfem.core.geometry import hansbo_cut_ratio

# --- UFL-like imports ---
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    TrialFunction, TestFunction, VectorTrialFunction, VectorTestFunction,
    Function, VectorFunction, Constant, grad, inner, dot, div, jump, avg,
    FacetNormal, CellDiameter, Pos, Neg, ElementWiseConstant, restrict,
    det, inv
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
ENABLE_PLOTTING = False
# --- Geometry and Fluid Properties ---
H = 0.41  # Channel height
L = 2.2   # Channel length
D = 0.1   # Cylinder diameter
c_x, c_y = 0.2, 0.2  # Cylinder center
rho_f = 1.0  # Density for fluid
rho_s = 1000.0  # Density for solid
mu_f = 1e-3  # Viscosity for fluid
U_max = 1.0 # Max inflow velocity
U_mean = 3.0/2.0 * U_max # Mean inflow velocity
# Lame coefficients for solid
_lambda_s = 0.5e6 # Lame's first parameter for solid
_mu_s = 2.0e6   # Lame's second parameter for solid
# beta_penalty = max(mu_f, _mu_s)   # Nitsche penalty parameter
beta_penalty = 90.0 * mu_f   # Nitsche penalty parameter
Re = rho_f * U_max * D / mu_f
print(f"Reynolds number (Re): {Re:.2f}")
with_ghost_enabled = False
hessian_ghost_enabled = False


# In[3]:


from pycutfem.utils.adaptive_mesh_ls_numba import structured_quad_levelset_adaptive
# from pycutfem.utils.adaptive_mesh import structured_quad_levelset_adaptive
# --- Mesh ---
# A finer mesh is needed for this benchmark
NX, NY = 30, 20
# NX, NY = 40, 25
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



# In[4]:


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
has_pos = fluid_domain | cut_domain
has_neg = solid_domain | cut_domain
solid_ghost_edges = mesh.edge_bitset('ghost_neg')  | mesh.edge_bitset('interface') #| mesh.edge_bitset('ghost_both')
fluid_ghost_edges = mesh.edge_bitset('ghost_pos')  | mesh.edge_bitset('interface') #| mesh.edge_bitset('ghost_both')

# --- Finite Element Space and DofHandler ---
# Taylor-Hood elements (Q2 for velocity, Q1 for pressure)
poly_order = 2

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


# In[5]:


dof_handler.tag_dofs_from_element_bitset("inactive", "u_pos_x", "inside", strict=True)
dof_handler.tag_dofs_from_element_bitset("inactive", "u_pos_y", "inside", strict=True)
dof_handler.tag_dofs_from_element_bitset("inactive", "p_pos_", "inside", strict=True)

dof_handler.tag_dofs_from_element_bitset("inactive", "vs_neg_x", "outside", strict=True)
dof_handler.tag_dofs_from_element_bitset("inactive", "vs_neg_y", "outside", strict=True)
dof_handler.tag_dofs_from_element_bitset("inactive", "d_neg_x", "outside", strict=True)
dof_handler.tag_dofs_from_element_bitset("inactive", "d_neg_y", "outside", strict=True)



# In[6]:


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
    tag_name = f'pinning_{field}'
    dof_handler.tag_dof_by_locator(
        tag_name, field,
        locator=lambda x, y, x0=actual_pin_coords[0], y0=actual_pin_coords[1]: 
            np.isclose(x, x0) and np.isclose(y, y0),
        find_first=True
    )
    bcs.append(
        BoundaryCondition(field, 'dirichlet', tag_name, lambda x, y: 0.0)
    )
    bcs_homog.append(
        BoundaryCondition(field, 'dirichlet', tag_name, lambda x, y: 0.0)
    )


# In[7]:


print(f'Total dirchlet dofs: {len(dof_handler.get_dirichlet_data(bcs))}')


# In[8]:


from pycutfem.io.visualization import plot_mesh_2
if ENABLE_PLOTTING:
    fig, ax = plt.subplots(figsize=(15, 30))
    plot_mesh_2(mesh, ax=ax, level_set=level_set, show=True, 
                plot_nodes=False, elem_tags=False, edge_colors=True, plot_interface=False,resolution=300)


# In[9]:


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
dp_f = TrialFunction(name='trial_pressure_fluid', field_name='p_pos_', dof_handler=dof_handler, side='+')
du_s = VectorTrialFunction(space=velocity_solid_space, dof_handler=dof_handler)
ddisp_s = VectorTrialFunction(space=displacement_space, dof_handler=dof_handler)
test_vel_f = VectorTestFunction(space=velocity_fluid_space, dof_handler=dof_handler)
test_q_f = TestFunction(name='test_pressure_fluid', field_name='p_pos_', dof_handler=dof_handler, side='+')
test_vel_s = VectorTestFunction(space=velocity_solid_space, dof_handler=dof_handler)
test_disp_s = VectorTestFunction(space=displacement_space, dof_handler=dof_handler)

# Solution functions at current (k) and previous (n) time steps
uf_k = VectorFunction(name="u_f_k", field_names=['u_pos_x', 'u_pos_y'], dof_handler=dof_handler, side='+')
pf_k = Function(name="p_f_k", field_name='p_pos_', dof_handler=dof_handler, side='+')
uf_n = VectorFunction(name="u_f_n", field_names=['u_pos_x', 'u_pos_y'], dof_handler=dof_handler, side='+')
pf_n = Function(name="p_f_n", field_name='p_pos_', dof_handler=dof_handler, side='+')
us_k = VectorFunction(name="u_s_k", field_names=['vs_neg_x', 'vs_neg_y'], dof_handler=dof_handler, side='-')
us_n = VectorFunction(name="u_s_n", field_names=['vs_neg_x', 'vs_neg_y'], dof_handler=dof_handler, side='-')
disp_k = VectorFunction(name="disp_k", field_names=['d_neg_x', 'd_neg_y'], dof_handler=dof_handler, side='-')
disp_n = VectorFunction(name="disp_n", field_names=['d_neg_x', 'd_neg_y'], dof_handler=dof_handler, side='-')

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

if ENABLE_PLOTTING:
    uf_n.plot()


# In[11]:


from pycutfem.ufl.expressions import (
    Derivative, FacetNormal, trace, Jump, Hessian,
    Pos, Neg, ElementWiseConstant, restrict,
)

n = FacetNormal()                    # vector expression (n_x, n_y)



def grad_inner_jump(u, v):
    a = dot(jump(grad(u)), n)
    b = dot(jump(grad(v)), n)
    return inner(a, b)

def hess_inner_jump(u, v):
    a = dot(n, dot(jump(Hessian(u)), n))
    b = dot(n, dot(jump(Hessian(v)), n))
    return inner(a, b)

qvol = 6
dx_fluid = dx(
    defined_on=fluid_interface_domain,
    level_set=level_set,
    metadata={"q": qvol, "side": "+"},
)
dx_solid = dx(
    defined_on=solid_interface_domain,
    level_set=level_set,
    metadata={"q": qvol, "side": "-"},
)
dΓ = dInterface(
    defined_on=mesh.element_bitset("cut"),
    level_set=level_set,
    metadata={"q": qvol + 2, "derivs": {(0, 0), (0, 1), (1, 0)}},
)
if hessian_ghost_enabled:
    dG_fluid = dGhost(
        defined_on=fluid_ghost_edges,
        level_set=level_set,
        metadata={"q": qvol, "derivs": {(0, 0), (0, 1), (1, 0), (2, 0), (0, 2), (1, 1)}},
    )
    dG_solid = dGhost(
        defined_on=solid_ghost_edges,
        level_set=level_set,
        metadata={"q": qvol, "derivs": {(0, 0), (0, 1), (1, 0), (2, 0), (0, 2), (1, 1)}},
    )
else:
    dG_fluid = dGhost(
        defined_on=fluid_ghost_edges,
        level_set=level_set,
        metadata={"q": qvol, "derivs": {(0, 1), (1, 0)}},
    )
    dG_solid = dGhost(
        defined_on=solid_ghost_edges,
        level_set=level_set,
        metadata={"q": qvol, "derivs": {(0, 1), (1, 0)}},
    )

cell_h = CellDiameter()
beta_N = Constant(beta_penalty * poly_order * (poly_order + 1))

theta_min = 1.0e-3
theta_pos_vals = np.clip(hansbo_cut_ratio(mesh, level_set, side="+"), theta_min, 1.0)
theta_neg_vals = np.clip(hansbo_cut_ratio(mesh, level_set, side="-"), theta_min, 1.0)
kappa_pos = Pos(ElementWiseConstant(theta_pos_vals))
kappa_neg = Neg(ElementWiseConstant(theta_neg_vals))
use_restricted_forms = True
if use_restricted_forms:
    du_f_R =        restrict(du_f, has_pos)
    dp_f_R =        restrict(dp_f, has_pos)
    test_vel_f_R =  restrict(test_vel_f, has_pos)
    test_q_f_R =    restrict(test_q_f, has_pos)
    uf_k_R =        restrict(uf_k, has_pos)
    uf_n_R =        restrict(uf_n, has_pos)
    pf_k_R =        restrict(pf_k, has_pos)
    pf_n_R =        restrict(pf_n, has_pos)

    du_s_R =        restrict(du_s, has_neg)
    ddisp_s_R =     restrict(ddisp_s, has_neg)
    test_vel_s_R =  restrict(test_vel_s, has_neg)
    test_disp_s_R = restrict(test_disp_s, has_neg)
    us_k_R =        restrict(us_k, has_neg)
    us_n_R =        restrict(us_n, has_neg)
    disp_k_R =      restrict(disp_k, has_neg)
    disp_n_R =      restrict(disp_n, has_neg)
else:
    du_f_R =        du_f
    dp_f_R =        dp_f
    test_vel_f_R =  test_vel_f
    test_q_f_R =    test_q_f
    uf_k_R =        uf_k
    uf_n_R =        uf_n
    pf_k_R =        pf_k
    pf_n_R =        pf_n
    du_s_R =        du_s
    ddisp_s_R =     ddisp_s
    test_vel_s_R =  test_vel_s
    test_disp_s_R = test_disp_s
    us_k_R =        us_k
    us_n_R =        us_n
    disp_k_R =      disp_k
    disp_n_R =      disp_n

I2 = Constant(np.eye(2))

def epsilon_f(u):
    return 0.5 * (grad(u) + grad(u).T)

def epsilon_s_linear_L(disp, disp_k):
    return 0.5 * (
        grad(disp) + grad(disp).T
        + dot(grad(disp).T, grad(disp_k))
        + dot(grad(disp_k).T, grad(disp))
    )

def epsilon_s_linear_R(disp_k):
    return 0.5 * (
        grad(disp_k) + grad(disp_k).T
        + dot(grad(disp_k).T, grad(disp_k))
    )

def sigma_s_linear_weak_linear_a(ddisp, disp_k, grad_v_test):
    eps = epsilon_s_linear_L(ddisp, disp_k)
    return 2.0 * mu_s * inner(eps, grad_v_test) + lambda_s * trace(eps) * trace(grad_v_test)

def sigma_s_linear_weak_nonlinear_residual(disp_k, grad_v_test):
    eps = epsilon_s_linear_R(disp_k)
    return 2.0 * mu_s * inner(eps, grad_v_test) + lambda_s * trace(eps) * trace(grad_v_test)

def traction_fluid(u_vec, p_scal):
    return 2.0 * mu_f_const * dot(epsilon_f(u_vec), n) - p_scal * n

# def traction_solid_L(disp_inc, disp_ref):
#     eps = epsilon_s_linear_L(disp_inc, disp_ref)
#     sigma = 2.0 * mu_s * eps + lambda_s * trace(eps) * I2
#     return dot(sigma, n)

# def traction_solid_R(disp_ref):
#     eps = epsilon_s_linear_R(disp_ref)
#     sigma = 2.0 * mu_s * eps + lambda_s * trace(eps) * I2
#     return dot(sigma, n)
# def traction_solid_cross(v_test, ddisp):
#     # ε_cross = 1/2 [ (∇v)^T ∇δd + (∇δd)^T ∇v ]
#     eps_c = 0.5 * ( dot(grad(v_test).T, grad(ddisp))
#                   + dot(grad(ddisp).T, grad(v_test)) )
#     sigma_c = 2.0*mu_s*eps_c + lambda_s*trace(eps_c)*I2
#     return dot(sigma_c, n)  # vector

def F_of(d):            # 2×2
    return I2 + grad(d)     # if grad is with respect to reference; if not, use small-strain approx

def C_of(F):            # 2×2
    return dot(F.T, F)

def E_of(F):            # 2×2
    return 0.5*(C_of(F) - I2)

def S_stvk(E):          # 2×2
    return lambda_s*trace(E)*I2 + 2.0*mu_s*E


# Cauchy stress
def sigma_s_nonlinear(d):
    F = F_of(d)
    E = E_of(F)
    S = S_stvk(E)
    J = det(F)       # or det(F) if available
    return (1.0/J) * dot(dot(F, S), F.T)

# tangent (directional) Cauchy stress
def dsigma_s(d_ref, delta_d):
    Fk = F_of(d_ref); Ek = E_of(Fk); Sk = S_stvk(Ek)
    dF = grad(delta_d)
    dE = 0.5*(dot(dF.T, Fk) + dot(Fk.T, dF))
    dS = lambda_s*trace(dE)*I2 + 2.0*mu_s*dE
    Jk = det(Fk)
    # F^{-1} only needed for δJ; in 2D you can code cof(F)/det(F) explicitly if inv() is missing
    Finv = inv(Fk)      # implement via cofactor/adjoint if your DSL lacks inv()
    dJ = Jk * trace(dot(Finv, dF))
    term = dot(dF, dot(Sk, Fk.T)) + dot(Fk, dot(dS, Fk.T)) + dot(Fk, dot(Sk, dF.T))
    return (1.0/Jk)*term - (dJ/Jk)*sigma_s_nonlinear(d_ref)

def traction_solid_R(d):             # residual traction (nonlinear)
    return dot(sigma_s_nonlinear(d), n)

def traction_solid_L(delta_d, d_ref):  # tangent traction in direction delta_d
    return dot(dsigma_s(d_ref, delta_d), n)

# Variation of Green-Lagrange Strain delta_E = dE/du [w] at state u_ref
# w is the direction (test or trial function)
def delta_E_GreenLagrange(w, u_ref):
    F_ref = F_of(u_ref)        # F(u_ref)
    grad_w = grad(w)           # grad_0(w)
    # delta_E = 0.5 * (grad_w^T * F_ref + F_ref^T * grad_w)
    return Constant(0.5) * (dot(grad_w.T, F_ref) + dot(F_ref.T, grad_w))

def sigma_dot_n_v(u_vec, p_scal, v_test):
    a = dot(grad(u_vec), n)
    b = dot(grad(u_vec).T, n)
    return mu_f * dot(a + b, v_test) - p_scal * dot(v_test, n)


jump_vel_trial = Jump(du_f, du_s)
jump_vel_test = Jump(test_vel_f, test_vel_s)
jump_vel_res = Jump(uf_k, us_k)
kappa_pos = Constant(0.5)
kappa_neg = Constant(0.5)
avg_flux_trial = (
    kappa_pos * traction_fluid(Pos(du_f), Pos(dp_f))
    - kappa_neg * traction_solid_L(Neg(ddisp_s), Neg(disp_k))
)

avg_flux_test = (
    kappa_pos * traction_fluid(Pos(test_vel_f),- Pos(test_q_f))
    - kappa_neg * traction_solid_L(Neg(test_disp_s), Neg(disp_k))
)

avg_flux_res = (
    kappa_pos * traction_fluid(Pos(uf_k), Pos(pf_k))
    - kappa_neg * traction_solid_R(Neg(disp_k))
)




J_int = (
    - dot(avg_flux_trial, jump_vel_test)
    - dot(avg_flux_test,  jump_vel_trial)
    + (beta_N*mu_f/cell_h) * dot(jump_vel_trial, jump_vel_test)
    # + dot(kappa_neg * traction_solid_cross(Neg(test_vel_s), Neg(ddisp_s)),
    #       jump_vel_res)
) * dΓ

R_int = (
    - dot(avg_flux_res, jump_vel_test)
    - dot(avg_flux_test, jump_vel_res)
    + (beta_N * mu_f / cell_h) * dot(jump_vel_res, jump_vel_test)
) * dΓ

a_vol_f = (
    rho_f_const / dt * dot(du_f_R, test_vel_f_R)
    + theta * rho_f_const * dot(dot(grad(uf_k_R), du_f_R), test_vel_f_R)
    + theta * rho_f_const * dot(dot(grad(du_f_R), uf_k_R), test_vel_f_R)
    + theta * mu_f_const * inner(grad(du_f_R), grad(test_vel_f_R))
    - dp_f_R * div(test_vel_f_R)
    + test_q_f_R * div(du_f_R)
) * dx_fluid

r_vol_f = (
    rho_f_const * dot(uf_k_R - uf_n_R, test_vel_f_R) / dt
    + theta * rho_f_const * dot(dot(grad(uf_k_R), uf_k_R), test_vel_f_R)
    + (1 - theta) * rho_f_const * dot(dot(grad(uf_n_R), uf_n_R), test_vel_f_R)
    + theta * mu_f_const * inner(grad(uf_k_R), grad(test_vel_f_R))
    + (1 - theta) * mu_f_const * inner(grad(uf_n_R), grad(test_vel_f_R))
    - pf_k_R * div(test_vel_f_R)
    + test_q_f_R * div(uf_k_R)
) * dx_fluid


# S evaluated at time steps k and n
S_k = S_stvk(E_of(F_of(disp_k_R)))
S_n = S_stvk(E_of(F_of(disp_n_R)))

# delta_E corresponding to the test function w = test_disp_s_R
# evaluated using reference states k and n
delta_E_test_k = delta_E_GreenLagrange(test_disp_s_R, disp_k_R)
delta_E_test_n = delta_E_GreenLagrange(test_disp_s_R, disp_n_R)
# Linearization of (S : delta_E_test) w.r.t displacement trial function delta_u = ddisp_s_R
# d/du [S(u) : delta_E_test(u, w)] [delta_u]
# = (dS/du[delta_u] : delta_E_test) + (S : d(delta_E_test)/du[delta_u])
# = (C : delta_E_trial : delta_E_test) + (S : delta_delta_E_test_trial)
# delta_E corresponding to the trial function delta_u = ddisp_s_R
delta_E_trial_k = delta_E_GreenLagrange(ddisp_s_R, disp_k_R)

# 1. Material Stiffness Term: (C : delta_E_trial) : delta_E_test
# For StVK: C:A = lambda tr(A)I + 2 mu A
C_delta_E_trial = lambda_s * trace(delta_E_trial_k) * I2 + Constant(2.0) * mu_s * delta_E_trial_k
material_stiffness_a = inner(C_delta_E_trial, delta_E_test_k)

# 2. Geometric Stiffness Term: S : delta(delta_E_test) [trial]
# delta(delta_E_test) [trial] = 0.5 * (grad(trial).T * grad(test) + grad(test).T * grad(trial))
geometric_stiffness_a = inner(S_k, Constant(0.5) * (dot(grad(ddisp_s_R).T, grad(test_disp_s_R))
                                                     + dot(grad(test_disp_s_R).T, grad(ddisp_s_R))))

# Corrected a_vol_s
a_vol_s = (
    # This assumes du_s_R is related to ddisp_s_R via time scheme, e.g., du_s = ddisp/dt? Check a_svc.
    rho_s * dot(du_s_R, test_vel_s_R) / dt
    + theta * (material_stiffness_a + geometric_stiffness_a)
) * dx_solid

r_vol_s = (
    rho_s * dot(us_k_R - us_n_R, test_vel_s_R) / dt # Inertia term (uses velocity test fn)
    # Virtual work term: S : delta_E_test (uses displacement test fn)
    + theta * inner(S_k, delta_E_test_k)
    + (1 - theta) * inner(S_n, delta_E_test_n)
) * dx_solid # Assumes integration over Omega_0

a_svc = (
    dot(ddisp_s_R, test_disp_s_R) / dt
    - theta * dot(du_s_R, test_disp_s_R)
) * dx_solid

r_svc = (
    dot(disp_k_R - disp_n_R, test_disp_s_R) / dt
    - theta * dot(us_k_R, test_disp_s_R)
    - (1 - theta) * dot(us_n_R, test_disp_s_R)
) * dx_solid

penalty_val = 1e-1
penalty_grad = 1e-1
penalty_hess = 1e-1
gamma_v = Constant(penalty_val * poly_order**2)
gamma_v_grad = Constant(penalty_grad * poly_order**2)
gamma_p = Constant(penalty_val * poly_order)
gama_p_grad = Constant(penalty_grad * poly_order)

def g_v_f(gamma, phi_1, phi_2):
    if hessian_ghost_enabled:
        return gamma * (
            cell_h * grad_inner_jump(phi_1, phi_2)
            + cell_h**3.0 / 4.0 * hess_inner_jump(phi_1, phi_2)
        )
    return gamma * (cell_h * grad_inner_jump(phi_1, phi_2))

def g_p(gamma, phi_1, phi_2):
    return gamma * (cell_h**3.0 * grad_inner_jump(phi_1, phi_2))

def g_v_s(gamma, phi_1, phi_2):
    return gamma * (cell_h**3.0 * grad_inner_jump(phi_1, phi_2))

def g_disp_s(gamma, phi_1, phi_2):
    return gamma * (cell_h * grad_inner_jump(phi_1, phi_2))

a_stab = (
    (
        Constant(2.0) * mu_f_const * g_v_f(gamma_v, du_f_R, test_vel_f_R)
        + g_p(gamma_p, dp_f_R, test_q_f_R)
    )
    * dG_fluid
    + (
        rho_s_const * g_v_s(gamma_v, du_s_R, test_vel_s_R)
        + Constant(2.0) * mu_s * g_disp_s(gamma_v_grad, ddisp_s_R, test_disp_s_R)
    )
    * dG_solid
)

r_stab = (
    (
        Constant(2.0) * mu_f_const * g_v_f(gamma_v, uf_k_R, test_vel_f_R)
        + g_p(gamma_p, pf_k_R, test_q_f_R)
    )
    * dG_fluid
    + (
        rho_s_const * g_v_s(gamma_v, us_k_R, test_vel_s_R)
        + Constant(2.0) * mu_s * g_disp_s(gamma_v_grad, disp_k_R, test_disp_s_R)
    )
    * dG_solid
)

jacobian_form = a_vol_f + J_int + a_vol_s + a_svc + a_stab
residual_form = r_vol_f + R_int + r_vol_s + r_svc + r_stab
# jacobian_form = a_vol_f   + a_svc + a_stab + a_vol_s
# residual_form = r_vol_f  + r_svc + r_stab + r_vol_s




# In[12]:


# !rm ~/.cache/pycutfem_jit/*


# In[13]:


# from pycutfem.ufl.forms import assemble_form
# K,F=assemble_form(jacobian_form==-residual_form, dof_handler=dof_handler, bcs=bcs_homog)
# print(np.linalg.norm(F, ord=np.inf))


# In[ ]:


from pycutfem.solvers.nonlinear_solver import NewtonSolver, NewtonParameters, TimeStepperParameters#, AdamNewtonSolver
# from pycutfem.solvers.aainhb_solver import AAINHBSolver           # or get_solver("aainhb")

uf_k.nodal_values.fill(0.0); pf_k.nodal_values.fill(0.0)
uf_n.nodal_values.fill(0.0); pf_n.nodal_values.fill(0.0)
us_k.nodal_values.fill(1.0); us_n.nodal_values.fill(0.0)
disp_k.nodal_values.fill(2.0); disp_n.nodal_values.fill(0.0)
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

if ENABLE_PLOTTING:
    uf_n.plot(kind="streamline",
            density=4.0,
            linewidth=0.8,
            cmap="plasma",
            title="Turek-Schafer",background = False)
