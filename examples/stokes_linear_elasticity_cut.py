#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stokes + Linear Elasticity with CutFEM (monolithic, symmetric Nitsche)
- Fluid outside a circular solid inclusion (Turek-style geometry).
- Linear in both fluid (Stokes) and solid (small-strain elasticity).
- Interface coupling: u_f = u_s (symmetric Nitsche) and traction equilibrium.
- Optional ghost-penalty stabilization near the interface.

This file mirrors the deal.II "fsi-cut.cc" structure for apples-to-apples
comparison: mass+viscous Stokes, linear elasticity, kinematic link
(d^{n+1}-d^n)/dt = u_s^{n+1}, and symmetric Nitsche coupling.
"""
import os
from turtle import pen
import numpy as np
import scipy.sparse.linalg as spla
from pathlib import Path

# local JIT cache (avoid permission issues on shared ~/.cache)
JIT_CACHE = Path(__file__).resolve().parent / "_jit_cache"
os.environ.setdefault("PYCUTFEM_CACHE_DIR", str(JIT_CACHE))
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
JIT_CACHE.mkdir(parents=True, exist_ok=True)

# --- Core PyCutFEM imports ---------------------------------------------------
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.geometry import hansbo_cut_ratio

# UFL-like front-end
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    TrialFunction, TestFunction, VectorTrialFunction, VectorTestFunction,
    Function, VectorFunction, Constant, grad, inner, dot, div, Jump,
    FacetNormal, CellDiameter, trace, Grad, Dot, Pos, Neg, jump, restrict,
    ElementWiseConstant
)
from pycutfem.ufl.measures import dx, dInterface, dGhost
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.io.vtk import export_vtk
from pycutfem.fem import transform
from pycutfem.utils.adaptive_mesh_ls_numba import structured_quad_levelset_adaptive



# -----------------------------------------------------------------------------
# 1) Problem setup (geometry, materials, FE, level set)
# -----------------------------------------------------------------------------
H, L = 0.41, 2.2
D = 0.1
cx, cy = 0.2, 0.2

rho_f, mu_f = 1.0, 1.0e-3
rho_s = 1.0
lambda_s, mu_s = 0.5e6, 2.0e6
mu_f_const = Constant(mu_f)
mu_s_const = Constant(mu_s)
lambda_s_const = Constant(lambda_s)

deg_u = 2          # Q2 for velocity
deg_p = 1          # Q1 for pressure
deg_s = 2          # Q2 for solid velocity and displacement

dt = 5.0e-1
num_time_steps = 100
Tfinal = dt * num_time_steps      # keep short for a smoke test; raise to 0.72 if desired
U_mean = 1.0       # mean inflow velocity for the parabolic profile

beta_N = 10.0 * deg_u #* (deg_u + 1)   # Nitsche scaling (same as deal.II file)
ADD_GHOST_PENALTY = False

# mesh (structured quads)
NX, NY = 35, 30
geom_order = 2
nodes, elems, edges, corners = structured_quad_levelset_adaptive(
        Lx=L, Ly=H, nx=NX, ny=NY, poly_order=geom_order,
        level_set=CircleLevelSet(center=(cx, cy), radius=(D/2.0+0.2*D/2.0) ),
        max_refine_level=1)  
mesh = Mesh(nodes=nodes, element_connectivity=elems,
            elements_corner_nodes=corners, element_type="quad",
            poly_order=geom_order)

# level set (solid = inside of the circle)
level_set = CircleLevelSet(center=(cx, cy), radius=0.5*D)
mesh.classify_elements(level_set)
mesh.classify_edges(level_set)
mesh.build_interface_segments(level_set)

# boundary tags
bc_tags = {
    'inlet':  lambda x,y: np.isclose(x, 0.0),
    'outlet': lambda x,y: np.isclose(x, L),
    'walls':  lambda x,y: np.isclose(y, 0.0) | np.isclose(y, H),
}
mesh.tag_boundary_edges(bc_tags)

# convenience bitsets
inside_e  = mesh.element_bitset("inside")   # solid-only elements
outside_e = mesh.element_bitset("outside")  # fluid-only elements
cut_e     = mesh.element_bitset("cut")      # intersected
fluid_dom = outside_e | cut_e
solid_dom = inside_e  | cut_e

# ghost faces close to interface (on either side)
ghost_pos = mesh.edge_bitset('ghost_pos') | mesh.edge_bitset('interface') #| mesh.edge_bitset('ghost_both')
ghost_neg = mesh.edge_bitset('ghost_neg') | mesh.edge_bitset('interface') #| mesh.edge_bitset('ghost_both')

# -----------------------------------------------------------------------------
# 2) FE spaces and unknowns
#    Single monolithic space with "inactive" fields pinned to 0 in the wrong
#    subdomain (mimics FE_Nothing strategy).
# -----------------------------------------------------------------------------
fields = {
    'ux': deg_u, 'uy': deg_u, 'p': deg_p,         # fluid
    'vsx': deg_s, 'vsy': deg_s,                   # solid velocity
    'dx': deg_s, 'dy': deg_s,                     # solid displacement
    'p_mean': ':number:'                          # Lagrange multiplier for pressure mean constraint
}
me  = MixedElement(mesh, field_specs=fields)
dh  = DofHandler(me, method='cg')
dh.info()

# function spaces for grouping
Vf = FunctionSpace("fluid_velocity",      ['ux','uy'])
Pf = FunctionSpace("fluid_pressure",      ['p'])
Vs = FunctionSpace("solid_velocity",      ['vsx','vsy'])
Ds = FunctionSpace("solid_displacement",  ['dx','dy'])

# trial/test
du_f   = VectorTrialFunction(Vf, dof_handler=dh)
dp_f   = TrialFunction('p', dh)
du_s   = VectorTrialFunction(Vs, dof_handler=dh)
ddisp  = VectorTrialFunction(Ds, dof_handler=dh)
p_mean_trial = TrialFunction('p_mean', dh)

v_f    = VectorTestFunction(Vf, dof_handler=dh)
q_f    = TestFunction('p', dh)
v_s    = VectorTestFunction(Vs, dof_handler=dh)
w_s    = VectorTestFunction(Ds, dof_handler=dh)
p_mean_test = TestFunction('p_mean', dh)


# solution vectors at time n and k
uf_n   = VectorFunction("uf_n",  ['ux','uy'], dh)
pf_n   = Function("pf_n",        'p',         dh)
us_n   = VectorFunction("us_n",  ['vsx','vsy'], dh)
d_n    = VectorFunction("d_n",   ['dx','dy'],  dh)

uf_k   = VectorFunction("uf_k",  ['ux','uy'], dh)
pf_k   = Function("pf_k",        'p',         dh)
us_k   = VectorFunction("us_k",  ['vsx','vsy'], dh)
d_k    = VectorFunction("d_k",   ['dx','dy'],  dh)

dp_f_R = restrict(dp_f, fluid_dom)
q_f_R  = restrict(q_f, fluid_dom)
du_f_R = restrict(du_f, fluid_dom)
v_f_R  = restrict(v_f, fluid_dom)
uf_n_R = restrict(uf_n, fluid_dom)
uf_k_R = restrict(uf_k, fluid_dom)

du_s_R   = restrict(du_s, solid_dom)
v_s_R    = restrict(v_s, solid_dom)
ddisp_R  = restrict(ddisp, solid_dom)
w_s_R    = restrict(w_s, solid_dom)
us_n_R   = restrict(us_n, solid_dom)
us_k_R   = restrict(us_k, solid_dom)
d_n_R    = restrict(d_n, solid_dom)
d_k_R    = restrict(d_k, solid_dom)

pf_n_R = restrict(pf_n, fluid_dom)
pf_k_R = restrict(pf_k, fluid_dom)
# zero initial conditions
for F in (uf_n, pf_n, us_n, d_n, uf_k, pf_k, us_k, d_k):
    F.nodal_values.fill(0.0)

# -----------------------------------------------------------------------------
# 3) Measures and helpers
# -----------------------------------------------------------------------------
dx_f = dx(defined_on=fluid_dom, level_set=level_set, metadata={'q': 2*max(deg_u,deg_p)+2, 'side': '+'})
dx_s = dx(defined_on=solid_dom, level_set=level_set, metadata={'q': 2*deg_s+2, 'side': '-'})
dΓ   = dInterface(defined_on=cut_e, level_set=level_set, metadata={'q': 2*max(deg_u,deg_s)+2, 'derivs': {(0,0),(0,1),(1,0)}})
dΓ_force = dInterface(defined_on=cut_e, level_set=level_set, metadata={'q': max(6, 2*deg_u+1)})

dG_f = dGhost(defined_on=ghost_pos, level_set=level_set, metadata={'q': 2*deg_u+2, 'derivs': {(0,0),(0,1),(1,0)}})
dG_s = dGhost(defined_on=ghost_neg, level_set=level_set, metadata={'q': 2*deg_s+2, 'derivs': {(0,0),(0,1),(1,0)}})

n     = FacetNormal()
hcell = CellDiameter()

def epsilon(u):
    return 0.5*(grad(u) + grad(u).T)

def sigma_f(u, p):
    return 2.0 * mu_f_const * epsilon(u) - p * Constant(np.eye(2))

def sigma_s(d):
    return 2.0 * mu_s_const * epsilon(d) + lambda_s_const * trace(epsilon(d)) * Constant(np.eye(2))

def traction_f(u, p):
    return dot(sigma_f(u, p), n)

def traction_s(d):
    return dot(sigma_s(d), n)

e_x = Constant(np.array([1.0, 0.0]), dim=1)
e_y = Constant(np.array([0.0, 1.0]), dim=1)

def traction_dot_dir(u_vec, p_scalar, direction, side="+"):
    if side == "+":
        grad_u = Pos(Grad(u_vec))
        p_side = Pos(p_scalar)
    else:
        grad_u = Neg(Grad(u_vec))
        p_side = Neg(p_scalar)
    a = Dot(grad_u, n)
    b = Dot(grad_u.T, n)
    traction_vec = mu_f_const * (a + b) - p_side * n
    return Dot(traction_vec, direction)

def evaluate_scalar_at_point(dh: DofHandler, mesh: Mesh, field: Function, point):
    x, y = point
    xy = np.asarray(point)
    eid_found = None
    for elem in mesh.elements_list:
        node_ids = elem.nodes
        coords = mesh.nodes_x_y_pos[list(node_ids)]
        if not (coords[:, 0].min() <= x <= coords[:, 0].max() and coords[:, 1].min() <= y <= coords[:, 1].max()):
            continue
        try:
            xi, eta = transform.inverse_mapping(mesh, elem.id, xy)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if -1.00001 <= xi <= 1.00001 and -1.00001 <= eta <= 1.00001:
            eid_found = elem.id
            break
    if eid_found is None:
        raise ValueError(f"Point {point} not located in any element.")
    me = dh.mixed_element
    phi = me.basis(field.field_name, xi, eta)[me.slice(field.field_name)]
    gdofs = dh.element_maps[field.field_name][eid_found]
    vals = field.get_nodal_values(gdofs)
    return float(phi @ vals)

histories = {"time": [], "cd": [], "cl": [], "dp": []}

# -----------------------------------------------------------------------------
# 4) Dirichlet boundary conditions
# -----------------------------------------------------------------------------
def parabolic_inflow(x,y):
    return 4.0*U_mean*y*(H-y)/(H*H)

bcs = [
    # fluid
    BoundaryCondition('ux','dirichlet','inlet', parabolic_inflow),
    BoundaryCondition('uy','dirichlet','inlet', lambda x,y: 0.0),
    BoundaryCondition('ux','dirichlet','walls', lambda x,y: 0.0),
    BoundaryCondition('uy','dirichlet','walls', lambda x,y: 0.0),
]

dh.apply_bcs(bcs, uf_n, pf_n, us_n, d_n)

# activate "FE_Nothing" behavior via zeros in the wrong subdomain
dh.tag_dofs_from_element_bitset("inactive", "ux", "inside", strict=True)
dh.tag_dofs_from_element_bitset("inactive", "uy", "inside", strict=True)
dh.tag_dofs_from_element_bitset("inactive", "p",  "inside", strict=True)

dh.tag_dofs_from_element_bitset("inactive", "vsx","outside", strict=True)
dh.tag_dofs_from_element_bitset("inactive", "vsy","outside", strict=True)
dh.tag_dofs_from_element_bitset("inactive", "dx", "outside", strict=True)
dh.tag_dofs_from_element_bitset("inactive", "dy", "outside", strict=True)


# Pin solid at the circle center to remove rigid modes (vsx,vsy,dx,dy = 0 at (cx,cy))
pin_name = 'pin_solid_center'
pin_fields = ('vsx', 'vsy', 'dx', 'dy')
dh._ensure_dof_coords()
pin_target = np.array([cx, cy])
pin_dofs = dh.dof_tags.setdefault(pin_name, set())
for fld in pin_fields:
    field_ids = np.asarray(dh.get_field_slice(fld), dtype=int)
    if field_ids.size == 0:
        raise RuntimeError(f"No DOFs available for field '{fld}' when pinning the solid center.")
    coords = dh._dof_coords[field_ids]
    local_idx = int(np.argmin(np.sum((coords - pin_target)**2, axis=1)))
    pin_dofs.add(int(field_ids[local_idx]))
zero_bc = lambda x,y: 0.0
for fld in pin_fields:
    bcs.append(BoundaryCondition(fld, 'dirichlet', pin_name, zero_bc))

# -----------------------------------------------------------------------------
# 5) Variational forms (linear)
# -----------------------------------------------------------------------------
theta = Constant(1.0)  # Crank-Nicolson
# Fluid volume (Stokes): 
# implicit pressure
a_fluid = (
    Constant(rho_f/dt) * dot(du_f_R, v_f_R)
  + theta * Constant(2.0*mu_f) * inner(epsilon(du_f_R), epsilon(v_f_R))
  - dp_f_R * div(v_f_R) + q_f_R * div(du_f_R)
) * dx_f
l_fluid = (Constant(rho_f/dt) * dot(uf_n_R, v_f_R) 
           - (1.0 - theta) * Constant(2.0*mu_f) * inner(epsilon(uf_n_R), epsilon(v_f_R))
           ) * dx_f

# Solid momentum (velocity test) + Elasticity (displacement test) + Kinematic link:
a_solid = (
    Constant(rho_s/dt) * dot(du_s_R, w_s_R)                                # momentum
  + theta * Constant(2.0*mu_s) * inner(epsilon(ddisp_R), epsilon(w_s_R))            # elasticity
  + theta * Constant(lambda_s) * trace(epsilon(ddisp_R)) * trace(epsilon(w_s_R))    # elasticity                                                      # kinematic -(u_s, w)
) * dx_s
l_solid = (
    Constant(rho_s/dt) * dot(us_n_R, w_s_R)
    - (1.0 - theta) * Constant(2.0*mu_s) * inner(epsilon(us_n_R), epsilon(w_s_R))
    - (1.0 - theta) * Constant(lambda_s) * trace(epsilon(us_n_R)) * trace(epsilon(w_s_R))
) * dx_s
# solid velocity and displacement coupling
a_svc = (
    Constant(1.0/dt) * dot(ddisp_R, v_s_R)                                  # kinematic (d, w)/dt
  - theta * dot(du_s_R, v_s_R)                                                      # kinematic -(u_s, w)
) * dx_s
l_svc = (
    Constant(1.0/dt) * dot(d_n_R, v_s_R)
    + (1.0 - theta) * dot(us_n_R, v_s_R)
) * dx_s



# Interface Γ: symmetric Nitsche for u_f = u_s; traction equilibrium enforced symmetrically
tau_N = Constant(beta_N)/hcell# * Constant(max(mu_f, mu_s)) 
# Hansbo weights θ^+, θ^- (per element); element-wise constants
theta_pos_vals = hansbo_cut_ratio(mesh, level_set, side='+')
theta_neg_vals = 1.0 - theta_pos_vals
kappa_pos = Pos(ElementWiseConstant(theta_pos_vals))
kappa_neg = Neg(ElementWiseConstant(theta_neg_vals))
jump_u_trial = Pos(du_f) - Neg(du_s)
jump_u_test  = Pos(v_f)  - Neg(v_s)
jump_u_res   = Pos(uf_k) - Neg(us_k)
jump_u_old   = Pos(uf_n) - Neg(us_n)

avg_flux_trial = kappa_pos * traction_f(Pos(du_f), Pos(dp_f))  + kappa_neg * traction_s(Neg(ddisp))
avg_flux_test  = kappa_pos * traction_f(Pos(v_f), -Pos(q_f))    + kappa_neg * traction_s(Neg(w_s))
avg_flux_old  = kappa_pos * traction_f(Pos(uf_n), Pos(pf_n))   + kappa_neg * traction_s(Neg(d_n))
# avg_flux_res   = traction_f(uf_k, pf_k) + traction_s(d_k)



J_int = (
    - theta * dot(avg_flux_trial, jump_u_test)
    - theta * dot(avg_flux_test,  jump_u_trial)
    + theta * tau_N * dot(jump_u_trial, jump_u_test)
) * dΓ

L_int = (
      (1.0 - theta) * dot(avg_flux_old, jump_u_test)
    + (1.0 - theta) * dot(avg_flux_test, jump_u_old)
    - (1.0 - theta) * tau_N * dot(jump_u_old, jump_u_test)
) * dΓ

# Ghost-penalty (mild) – normal-gradient jump of velocities & displacement
a = a_fluid + a_solid + J_int + a_svc
a += (p_mean_trial * q_f_R + p_mean_test * dp_f_R) * dx_f
# Total bilinear and linear forms
l = l_fluid + l_solid + l_svc  + L_int         # no explicit RHS from interface/ghost

def grad_inner_jump(u, v):
    """⟨∂ₙu, ∂ₙv⟩  (scalar or 2‑D vector)."""
    a = dot(jump(grad(u)), n)
    b = dot(jump(grad(v)), n)
    return inner(a, b)
# Optional: if you turn on the ghost penalty, treat it like a stiffness term:
if ADD_GHOST_PENALTY:
    penalty = Constant(1e-3)
    GP_trial =  (
        theta * penalty * hcell * grad_inner_jump(du_f, v_f) * dG_f
      + theta * penalty * hcell * grad_inner_jump(ddisp, w_s) * dG_s
    )
    GP_old =  (
      - (1.0 - theta) * penalty * hcell * grad_inner_jump(uf_n, v_f) * dG_f
      - (1.0 - theta) * penalty * hcell * grad_inner_jump(d_n,  w_s) * dG_s
    )
    a +=  GP_trial
    l +=  GP_old




# -----------------------------------------------------------------------------
# 6) Time loop: assemble, solve, update, write VTK
# -----------------------------------------------------------------------------
out_dir = "output_pycutfem_fsi"
os.makedirs(out_dir, exist_ok=True)

def export(step, sol_vec):
    U_f = VectorFunction("u_f", ['ux','uy'], dh)
    P_f = Function("p_f", 'p', dh)
    U_s = VectorFunction("u_s", ['vsx','vsy'], dh)
    D_s = VectorFunction("d_s", ['dx','dy'], dh)
    dh.add_to_functions(sol_vec, [U_f, P_f, U_s, D_s])
    export_vtk(
        os.path.join(out_dir, f"fsi_stokes_le_{step:04d}.vtu"),
        mesh,
        dh,
        {
            "u_f": U_f,
            "p": P_f,
            "u_s": U_s,
            "d": D_s,
        },
    )
    return U_f, P_f, U_s, D_s

# initial export
sol0 = np.zeros(dh.total_dofs)
U_f, P_f, U_s, D_s = export(0, sol0)

n_steps = int(np.ceil(Tfinal/dt))
for k in range(1, n_steps+1):
    t = k*dt
    print(f"\n=== Time step {k}/{n_steps}  t={t:.3f} ===")

    # assemble linear system
    K, F = assemble_form(Equation(a, l), dof_handler=dh, bcs=bcs, backend='jit')

    K_ff, F_f, free, dir_idx, u_dir, full2red = dh.reduce_linear_system(
        K, F, bcs=bcs, return_dirichlet=True
    )

    field_sets = {f: set(dh.get_field_slice(f)) for f in dh.field_names}
    drop_pairs = [
        ("inactive", "ux"),
        ("inactive", "uy"),
        ("inactive", "p"),
        ("inactive", "vsx"),
        ("inactive", "vsy"),
        ("inactive", "dx"),
        ("inactive", "dy"),
    ]
    drop_full = set()
    for tag, fld in drop_pairs:
        drop_full |= (set(dh.dof_tags.get(tag, set())) & field_sets.get(fld, set()))
    drop_full = np.array(sorted(drop_full), dtype=int)
    drop_red = full2red[drop_full]
    drop_red = drop_red[drop_red >= 0]

    all_red = np.arange(K_ff.shape[0], dtype=int)
    keep_red = np.setdiff1d(all_red, drop_red, assume_unique=False)

    K_rr = K_ff[np.ix_(keep_red, keep_red)].tocsc()
    F_r = F_f[keep_red]
    u_r = spla.spsolve(K_rr, F_r)

    u_free = np.zeros_like(F_f)
    u_free[keep_red] = u_r

    sol = np.zeros(dh.total_dofs)
    sol[free] = u_free
    sol[dir_idx] = u_dir[dir_idx]

    # unpack to field functions and update "k"
    U_f = VectorFunction("u_f", ['ux','uy'], dh)
    P_f = Function("p_f", 'p', dh)
    U_s = VectorFunction("u_s", ['vsx','vsy'], dh)
    D_s = VectorFunction("d_s", ['dx','dy'], dh)
    dh.add_to_functions(sol, [U_f, P_f, U_s, D_s])

    # copy into *_k containers
    uf_k.nodal_values[:] = U_f.nodal_values
    pf_k.nodal_values[:] = P_f.nodal_values
    us_k.nodal_values[:] = U_s.nodal_values
    d_k.nodal_values[:]  = D_s.nodal_values
    dh.apply_bcs(bcs, uf_k, pf_k, us_k, d_k)

    # export
    U_f, P_f, U_s, D_s = export(k, sol)

    # Drag/Lift on the interface
    integrand_drag = traction_dot_dir(U_f, P_f, e_x, side="+")
    integrand_lift = traction_dot_dir(U_f, P_f, e_y, side="+")
    I_drag = integrand_drag * dΓ_force
    I_lift = integrand_lift * dΓ_force

    drag_hook = {I_drag.integrand: {"name": "FD"}}
    lift_hook = {I_lift.integrand: {"name": "FL"}}
    res_Fd = assemble_form(Equation(None, I_drag), dof_handler=dh, bcs=[], assembler_hooks=drag_hook, backend="python")
    res_Fl = assemble_form(Equation(None, I_lift), dof_handler=dh, bcs=[], assembler_hooks=lift_hook, backend="python")
    F_D = float(res_Fd["FD"])
    F_L = float(res_Fl["FL"])
    coeff = 2.0 / (rho_f * (U_mean**2) * D)
    C_D = coeff * F_D
    C_L = coeff * F_L

    pA = evaluate_scalar_at_point(dh, mesh, P_f, (cx - D/2 - 0.01, cy))
    pB = evaluate_scalar_at_point(dh, mesh, P_f, (cx + D/2 + 0.01, cy))
    dp = pA - pB

    histories["time"].append(t)
    histories["cd"].append(C_D)
    histories["cl"].append(C_L)
    histories["dp"].append(dp)
    print(f"    Drag={F_D:.6e}  Lift={F_L:.6e}  C_D={C_D:.6f}  C_L={C_L:.6f}  Δp={dp:.6f}")

    # advance
    uf_n.nodal_values[:] = uf_k.nodal_values
    pf_n.nodal_values[:] = pf_k.nodal_values
    us_n.nodal_values[:] = us_k.nodal_values
    d_n.nodal_values[:]  = d_k.nodal_values
    dh.apply_bcs(bcs, uf_n, pf_n, us_n, d_n)

print("\nDone.")
