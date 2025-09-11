import os
import numpy as np
import scipy.sparse.linalg as spla
import sympy as sp

# --- Core imports ---
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.io.vtk import export_vtk
from pycutfem.ufl.functionspace import FunctionSpace

# --- UFL-like imports ---
from pycutfem.ufl.expressions import (
    TrialFunction, TestFunction, VectorTrialFunction, VectorTestFunction,
    Function, VectorFunction, Constant, grad, inner, dot, div, jump, Pos, Neg,
    FacetNormal, CellDiameter, ElementWiseConstant, Jump, restrict
)
from pycutfem.ufl.measures import dx, dInterface, dGhost
from pycutfem.ufl.forms import BoundaryCondition, assemble_form, Equation

# --- Level Set & Hansbo ratio ---
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.core.geometry import hansbo_cut_ratio
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.analytic import x as x_ana
from pycutfem.ufl.analytic import y as y_ana
from pycutfem.io.visualization import plot_mesh_2
from pycutfem.ufl.helpers import analyze_active_dofs

# import logging
# logger = logging.getLogger(__name__)
# if not logger.hasHandlers():
#     logging.basicConfig(level=logging.INFO)

def exact_solution(mu, R, gammaf):
    # ---------------- 3) Exact fields & manufactured RHS via SymPy ----------------
    sx, sy = sp.symbols('x y', real=True)
    r2 = sx*sx + sy*sy

    aneg = 1.0/mu[0].value
    apos_base = 1.0/mu[1].value
    apos = apos_base + (aneg - apos_base) * sp.exp(r2 - R*R)

    # u_ex = a(r)*(-y, x)*exp(-r^2)  (piecewise: a = aneg in Ω-, apos in Ω+)
    u_neg_x = -aneg*sy*sp.exp(-r2)
    u_neg_y =  aneg*sx*sp.exp(-r2)
    u_pos_x = -apos*sy*sp.exp(-r2)
    u_pos_y =  apos*sx*sp.exp(-r2)

    # p_ex = x^3 (inside), x^3 - γ (outside)
    p_neg_sym = sx**3
    p_pos_sym = sx**3 - gammaf

    # ε(u) = 0.5(∇u + ∇u^T),  σ = -2 μ ε(u) + p I
    def eps_mat(ux, uy):
        du = sp.Matrix([[sp.diff(ux, sx), sp.diff(ux, sy)],
                        [sp.diff(uy, sx), sp.diff(uy, sy)]])
        return 0.5*(du + du.T)

    I2 = sp.eye(2)

    eps_neg = eps_mat(u_neg_x, u_neg_y)
    eps_pos = eps_mat(u_pos_x, u_pos_y)

    sigma_neg = -2*mu[0].value*eps_neg + p_neg_sym*I2
    sigma_pos = -2*mu[1].value*eps_pos + p_pos_sym*I2

    # g = div σ  (row-wise divergence → vector)
    def div_sigma(sig):
        return (sp.diff(sig[0,0], sx) + sp.diff(sig[0,1], sy),
                sp.diff(sig[1,0], sx) + sp.diff(sig[1,1], sy))

    g_neg_x_sym, g_neg_y_sym = div_sigma(sigma_neg)
    g_pos_x_sym, g_pos_y_sym = div_sigma(sigma_pos)

    # --- Lambdify *components* (safe for scalars or arrays) ---
    u_neg_x_fun = sp.lambdify((sx,sy), u_neg_x, 'numpy')
    u_neg_y_fun = sp.lambdify((sx,sy), u_neg_y, 'numpy')
    u_pos_x_fun = sp.lambdify((sx,sy), u_pos_x, 'numpy')
    u_pos_y_fun = sp.lambdify((sx,sy), u_pos_y, 'numpy')

    p_neg_fun   = sp.lambdify((sx,sy), p_neg_sym, 'numpy')
    p_pos_fun   = sp.lambdify((sx,sy), p_pos_sym, 'numpy')

    # If you also need exact gradients for error norms (du/dx, du/dy per component):
    dux_dx_neg = sp.lambdify((sx,sy), sp.diff(u_neg_x, sx), 'numpy')
    dux_dy_neg = sp.lambdify((sx,sy), sp.diff(u_neg_x, sy), 'numpy')
    duy_dx_neg = sp.lambdify((sx,sy), sp.diff(u_neg_y, sx), 'numpy')
    duy_dy_neg = sp.lambdify((sx,sy), sp.diff(u_neg_y, sy), 'numpy')

    dux_dx_pos = sp.lambdify((sx,sy), sp.diff(u_pos_x, sx), 'numpy')
    dux_dy_pos = sp.lambdify((sx,sy), sp.diff(u_pos_x, sy), 'numpy')
    duy_dx_pos = sp.lambdify((sx,sy), sp.diff(u_pos_y, sx), 'numpy')
    duy_dy_pos = sp.lambdify((sx,sy), sp.diff(u_pos_y, sy), 'numpy')

    g_neg_x_fun = sp.lambdify((sx,sy), g_neg_x_sym, 'numpy')
    g_neg_y_fun = sp.lambdify((sx,sy), g_neg_y_sym, 'numpy')
    g_pos_x_fun = sp.lambdify((sx,sy), g_pos_x_sym, 'numpy')
    g_pos_y_fun = sp.lambdify((sx,sy), g_pos_y_sym, 'numpy')

    # --- Vectorized-safe wrappers: stack components on the last axis --------------
    def vec2_callable(fx, fy):
        def f(x, y):
            ax = np.asarray(fx(x, y), dtype=float)
            ay = np.asarray(fy(x, y), dtype=float)
            # broadcast to common shape and stack along the last axis
            return np.stack([ax, ay], axis=-1)
        return f

    def grad2x2_callable(fxx, fxy, fyx, fyy):
        def g(x, y):
            a = np.asarray(fxx(x, y), dtype=float)
            b = np.asarray(fxy(x, y), dtype=float)
            c = np.asarray(fyx(x, y), dtype=float)
            d = np.asarray(fyy(x, y), dtype=float)
            # (..., 2, 2) with last two axes = rows/cols
            return np.stack([np.stack([a, b], axis=-1),
                            np.stack([c, d], axis=-1)], axis=-2)
        return g

    # Exact velocity, pressure, gradients (Ω- and Ω+) — vectorized friendly:
    vel_exact_neg_xy = vec2_callable(u_neg_x_fun, u_neg_y_fun)
    vel_exact_pos_xy = vec2_callable(u_pos_x_fun, u_pos_y_fun)

    # If you need exact ∇u in the error integrator:
    grad_vel_neg_xy  = grad2x2_callable(dux_dx_neg, dux_dy_neg, duy_dx_neg, duy_dy_neg)
    grad_vel_pos_xy  = grad2x2_callable(dux_dx_pos, dux_dy_pos, duy_dx_pos, duy_dy_pos)

    def p_exact_neg_xy(x, y):
        out = np.asarray(p_neg_fun(x, y), dtype=float)
        return out  # scalar or array
    def p_exact_pos_xy(x, y):
        out = np.asarray(p_pos_fun(x, y), dtype=float)
        return out

    # Manufactured body force (vector) — vectorized:
    g_neg_xy = vec2_callable(g_neg_x_fun, g_neg_y_fun)
    g_pos_xy = vec2_callable(g_pos_x_fun, g_pos_y_fun)
    return (vel_exact_neg_xy, vel_exact_pos_xy,
            g_neg_xy, g_pos_xy,
            grad_vel_neg_xy, grad_vel_pos_xy,
            p_exact_neg_xy, p_exact_pos_xy)


def test_stokes_interface_corrected():
    """
    Unfitted Stokes interface (NGSolve-style): Hansbo averaging, Nitsche coupling,
    ghost penalties, SymPy-manufactured RHS, and piecewise L2/H1 errors.
    This version is corrected to use the existing pycutfem API.
    """
    # ---------------- 1) Mesh, order, level set ----------------
    backend = 'python'
    poly_order = 2
    geom_order = 1
    print(f'Using backend: {backend}, polynomial order: {poly_order}')
    maxh = 0.125
    L,H = 2.0, 2.0
    mesh_size = int(L / maxh)

    nodes, elems, _, corners = structured_quad(
        L, H, nx=mesh_size, ny=mesh_size,
        poly_order=geom_order, offset=[-L/2.0, -H/2.0]
    )
    mesh = Mesh(nodes, element_connectivity=elems,
                elements_corner_nodes=corners,
                poly_order=geom_order, element_type='quad')

    mu = [Constant(1.0), Constant(10.0)]        # μ_in, μ_out
    R = 2.0/3.0
    gammaf = 0.5            # surface tension = pressure jump
    level_set = CircleLevelSet(center=(0.0, 0.0), radius=R)

    # Classify & interface
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    # plot_mesh_2(mesh, level_set=level_set)

    # Boundary tags (square boundary)
    boundary_tags = {
        'boundary': lambda x, y: np.isclose(x, -L/2.0) | np.isclose(x, L/2.0) |
                                  np.isclose(y, -H/2.0) | np.isclose(y, H/2.0)
    }
    # mesh.tag_boundary_edges(boundary_tags)

    # BitSets
    inside_e  = mesh.element_bitset("inside")
    outside_e = mesh.element_bitset("outside")
    cut_e     = mesh.element_bitset("cut")
    ghost_pos = mesh.edge_bitset("ghost_pos")
    ghost_neg = mesh.edge_bitset("ghost_neg")
    ghost_both = mesh.edge_bitset("ghost_both")
    ghost_interface = mesh.edge_bitset("interface")
    ghost_pos   = ghost_pos | ghost_both | ghost_interface
    ghost_neg   = ghost_neg | ghost_both | ghost_interface

    has_inside  = inside_e  | cut_e
    has_outside = outside_e | cut_e

    # ---------------- 2) Mixed FE space & dofs ----------------
    me = MixedElement(mesh, field_specs={'u_pos_x': poly_order, 'u_pos_y': poly_order, 'p_pos_': poly_order-1,
                                         'u_neg_x': poly_order, 'u_neg_y': poly_order, 'p_neg_': poly_order-1,
                                         'lm': ':number:'})
    dh = DofHandler(me, method='cg')
    dh.tag_dofs_by_locator_map(boundary_tags, 
                               fields=['u_pos_x','u_pos_y','u_neg_x','u_neg_y']) 

    Vspace_pos    = FunctionSpace("vel_positive", ['u_pos_x', 'u_pos_y'])
    Vspace_neg    = FunctionSpace("vel_negative", ['u_neg_x', 'u_neg_y'])
    vel_trial_pos = VectorTrialFunction(space=Vspace_pos, dof_handler=dh, side='+')
    vel_test_pos  = VectorTestFunction(space=Vspace_pos,  dof_handler=dh, side='+')
    vel_trial_neg = VectorTrialFunction(space=Vspace_neg, dof_handler=dh, side='-')
    vel_test_neg  = VectorTestFunction(space=Vspace_neg,  dof_handler=dh, side='-')
    p_trial_pos   = TrialFunction('p_pos_', dh,name='pressure_pos_trial', side='+')
    q_test_pos    = TestFunction ('p_pos_', dh,name='pressure_pos_test', side='+')
    p_trial_neg   = TrialFunction('p_neg_', dh,name='pressure_neg_trial', side='-')
    q_test_neg    = TestFunction ('p_neg_', dh,name='pressure_neg_test', side='-')
    nL = TrialFunction('lm')   # "n" in the NGSolve snippet
    mL = TestFunction ('lm')   # "m" in the NGSolve snippet

    # Measures & parameters
    qvol   = 2*poly_order + 2
    dx_pos = dx(defined_on=has_outside, level_set=level_set, metadata={'side': '+', 'q': qvol})
    dx_neg = dx(defined_on=has_inside,  level_set=level_set, metadata={'side': '-', 'q': qvol})
    dGamma = dInterface(defined_on=cut_e,  level_set=level_set, metadata={'q': qvol+2, 'derivs': {(1,0),(0,1)}})
    dG_pos = dGhost(defined_on=ghost_pos,    level_set=level_set, metadata={'q': qvol+2, 'derivs': {(1,0),(0,1)}})
    dG_neg = dGhost(defined_on=ghost_neg,    level_set=level_set, metadata={'q': qvol+2, 'derivs': {(1,0),(0,1)}})

    h = CellDiameter()
    n = FacetNormal()

    # Hansbo weights θ^+, θ^- (per element); element-wise constants
    theta_pos_vals = hansbo_cut_ratio(mesh, level_set, side='+')
    theta_neg_vals = 1.0 - theta_pos_vals
    kappa_pos = Pos(ElementWiseConstant(theta_pos_vals))
    kappa_neg = Neg(ElementWiseConstant(theta_neg_vals))

    lambda_nitsche = Constant(0.5 * (mu[0].value + mu[1].value) * 20 * poly_order**2)
    gamma_stab_v   = Constant(0.05)
    gamma_stab_p   = Constant(0.05)

    
    (vel_exact_neg_xy, vel_exact_pos_xy,
            g_neg_xy, g_pos_xy,
            grad_vel_neg_xy, grad_vel_pos_xy,
            p_exact_neg_xy, p_exact_pos_xy)=exact_solution(mu, R, gammaf)
    g_neg = Analytic(g_neg_xy)
    g_pos = Analytic(g_pos_xy)
    # g_neg = VectorFunction(name="g_neg", field_names=['u_neg_x', 'u_neg_y'], dof_handler=dh, side='-')
    # g_pos = VectorFunction(name="g_pos", field_names=['u_pos_x', 'u_pos_y'], dof_handler=dh, side='+')
    # g_neg.set_values_from_function(g_neg_xy)
    # g_pos.set_values_from_function(g_pos_xy)

    # ---------------- 4) Variational forms (NGSolve-style) ----------------
    def epsilon(u):
        return 0.5 * (grad(u) + grad(u).T)

    def traction(mu_val, u_vec, p_scal, normal):
        return -2*mu_val * dot(epsilon(u_vec), normal) + p_scal * normal
    
    # wrap pressure trial/test with domain restrictions
    q_pos_R = restrict(q_test_pos, has_outside)
    p_pos_R = restrict(p_trial_pos, has_outside)

    q_neg_R = restrict(q_test_neg, has_inside)
    p_neg_R = restrict(p_trial_neg, has_inside)

    # volume terms (replace q_test_*, p_trial_* with the restricted ones)
    a  = (2*mu[1]*inner(epsilon(vel_trial_pos), epsilon(vel_test_pos))
        - div(vel_trial_pos)*q_pos_R
        - div(vel_test_pos)*p_pos_R) * dx_pos

    a += (2*mu[0]*inner(epsilon(vel_trial_neg), epsilon(vel_test_neg))
        - div(vel_trial_neg)*q_neg_R
        - div(vel_test_neg)*p_neg_R) * dx_neg

    # mean constraint stays on NEG side:
    a += (nL * Neg(q_neg_R) + mL * Neg(p_neg_R)) * dx_neg
    # a += (nL * Pos(q_test_pos) + mL * Pos(p_trial_pos)) * dx_pos

    traction_pos_n_trial = traction(mu[1], Pos(vel_trial_pos), Pos(p_trial_pos), n)
    traction_neg_n_trial = traction(mu[0], Neg(vel_trial_neg), Neg(p_trial_neg), n)
    avg_flux_trial  = kappa_pos * traction_pos_n_trial + kappa_neg * traction_neg_n_trial

    traction_pos_n_test = traction(mu[1], Pos(vel_test_pos), Pos(q_test_pos), n)
    traction_neg_n_test = traction(mu[0], Neg(vel_test_neg), Neg(q_test_neg), n)
    avg_flux_test  = kappa_pos * traction_pos_n_test + kappa_neg * traction_neg_n_test

    def jump_ng(u_pos, u_neg):
        return Neg(u_neg) - Pos(u_pos)
    jump_vel_trial = jump_ng(vel_trial_pos, vel_trial_neg)
    jump_vel_test  = jump_ng(vel_test_pos,  vel_test_neg)
    jump_p_trial = jump_ng(p_trial_pos, p_trial_neg)
    jump_q_test  = jump_ng(q_test_pos,  q_test_neg)
    # Interface terms
    a += ( 
        #     dot(avg_flux_trial, jump_vel_test)
        #   + dot(avg_flux_test,  jump_vel_trial)
           (lambda_nitsche / 0.125) * dot(jump_vel_trial, jump_vel_test) ) * dGamma



    avg_inv_test = kappa_neg * Pos(vel_test_pos) + kappa_pos * Neg(vel_test_neg)
    f = dot(g_pos, vel_test_pos) * dx_pos \
      + dot(g_neg, vel_test_neg) * dx_neg 
    # f += - gammaf * dot(avg_inv_test, n) * dGamma


   

    
    
    # Ghost penalty terms
    # POS ghost patch
    # jp_pos_tr   = Jump(vel_trial_pos, vel_trial_pos)
    # jp_pos_te   = Jump(vel_test_pos, vel_test_pos)
    # jp_p_pos_tr = Jump(p_trial_pos, p_trial_pos)
    # jp_p_pos_te = Jump(q_test_pos, q_test_pos)
    # a += ((gamma_stab_v / h**2) * dot(jp_pos_tr, jp_pos_te)
    #       -(gamma_stab_p)       *       jp_p_pos_tr * jp_p_pos_te) * dG_pos
    

    # # NEG ghost patch
    # jp_neg_tr   = Jump(vel_trial_neg, vel_trial_neg)
    # jp_neg_te   = Jump(vel_test_neg, vel_test_neg)
    # jp_p_neg_tr = Jump(p_trial_neg, p_trial_neg)
    # jp_p_neg_te = Jump(q_test_neg, q_test_neg)
    # a += ((gamma_stab_v / h**2) * dot(jp_neg_tr, jp_neg_te)
    #       -(gamma_stab_p)        *       jp_p_neg_tr * jp_p_neg_te) * dG_neg




    equation = Equation(a, f)

    # ---------------- 5) Velocity Dirichlet on outer boundary ----------------
    dh.tag_dofs_from_element_bitset("inactive_inside_ux", "u_pos_x", "inside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive_inside_uy", "u_pos_y", "inside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive_outside_ux", "u_neg_x", "outside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive_outside_uy", "u_neg_y", "outside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive_inside_p", "p_pos_", "inside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive_outside_p", "p_neg_", "outside", strict=True)
    bcs = [
        BoundaryCondition('u_pos_x', 'dirichlet', 'boundary', lambda x, y: vel_exact_pos_xy(x, y)[0]),
        BoundaryCondition('u_pos_y', 'dirichlet', 'boundary', lambda x, y: vel_exact_pos_xy(x, y)[1]),
        # BoundaryCondition('p_pos_', 'dirichlet', 'boundary', lambda x, y: p_exact_pos_xy(x, y)),
        # BoundaryCondition('u_neg_x', 'dirichlet', 'boundary', lambda x, y: vel_exact_neg_xy(x, y)[0]),
        # BoundaryCondition('u_neg_y', 'dirichlet', 'boundary', lambda x, y: vel_exact_neg_xy(x, y)[1]),
        # BoundaryCondition('p_neg_', 'dirichlet', 'boundary', lambda x, y: p_exact_neg_xy(x, y)),
        # BoundaryCondition('u_pos_x', 'dirichlet', 'inactive_inside_ux', lambda x, y: 0.0),
        # BoundaryCondition('u_pos_y', 'dirichlet', 'inactive_inside_uy', lambda x, y: 0.0),
        # BoundaryCondition('u_neg_x', 'dirichlet', 'inactive_outside_ux', lambda x, y: 0.0),
        # BoundaryCondition('u_neg_y', 'dirichlet', 'inactive_outside_uy', lambda x, y: 0.0),
        # BoundaryCondition('p_pos_', 'dirichlet', 'inactive_inside_p', lambda x, y: p_exact_pos_xy(x, y)),
        # BoundaryCondition('p_neg_', 'dirichlet', 'inactive_outside_p', lambda x, y: p_exact_neg_xy(x, y)),
    ]


    # ---------------- 6) Assemble & solve ----------------
    K, F = assemble_form(equation, dof_handler=dh, bcs=bcs, quad_order=qvol, backend= backend)

    K_ff, F_f, free, dir_idx, u_dir, full2red = dh.reduce_linear_system(
        K, F, bcs=bcs, return_dirichlet=True
    )

    # --- NEW: Drop only the inactive pressure DOFs (mirror NGSolve Compress) ---
    # Build field-wise DOF sets (so tags don't accidentally drop other fields)
    field_sets = {f: set(dh.get_field_slice(f)) for f in dh.field_names}

    drop_v = set()
    for tag, fld in [
        ("inactive_inside_ux", "u_pos_x"),
        ("inactive_inside_uy", "u_pos_y"),
        ("inactive_outside_ux","u_neg_x"),
        ("inactive_outside_uy","u_neg_y"),
    ]:
        drop_v |= (set(dh.dof_tags.get(tag, set())) & field_sets.get(fld, set()))

    # Reuse your existing pressure-dropping code:
    drop_pos = set(dh.dof_tags.get("inactive_inside_p",  set())) & field_sets.get("p_pos_", set())
    drop_neg = set(dh.dof_tags.get("inactive_outside_p", set())) & field_sets.get("p_neg_", set())

    drop_full = np.array(sorted(drop_v | drop_pos | drop_neg), dtype=int)
    drop_red  = full2red[drop_full]; drop_red = drop_red[drop_red >= 0]

    all_red  = np.arange(K_ff.shape[0], dtype=int)
    keep_red = np.setdiff1d(all_red, drop_red, assume_unique=False)

    # Solve on the kept set
    K_rr = K_ff[np.ix_(keep_red, keep_red)].tocsc()
    F_r  = F_f[keep_red]
    u_r  = spla.spsolve(K_rr, F_r)

    # Re-expand to the full 'free' vector
    u_free = np.zeros_like(F_f)
    u_free[keep_red] = u_r

    # Expand to all DOFs
    sol = np.zeros(dh.total_dofs)
    sol[free]    = u_free
    sol[dir_idx] = u_dir[dir_idx]
    # ---------------- 7) Wrap solution & calculate errors ----------------
    U_pos = VectorFunction(name="velocity_pos", field_names=['u_pos_x','u_pos_y'], dof_handler=dh, side='+')
    P_pos = Function(name="pressure_pos", field_name='p_pos_', dof_handler=dh, side='+')
    U_neg = VectorFunction(name="velocity_neg", field_names=['u_neg_x','u_neg_y'], dof_handler=dh, side='-')
    P_neg = Function(name="pressure_neg", field_name='p_neg_', dof_handler=dh, side='-')
    dh.add_to_functions(sol, [U_pos, P_pos, U_neg, P_neg])



    exact_neg = {'u_neg_x': lambda x,y: vel_exact_neg_xy(x,y)[0], 'u_neg_y': lambda x,y: vel_exact_neg_xy(x,y)[1], 'p_neg_': p_exact_neg_xy}
    exact_pos = {'u_pos_x': lambda x,y: vel_exact_pos_xy(x,y)[0], 'u_pos_y': lambda x,y: vel_exact_pos_xy(x,y)[1], 'p_pos_': p_exact_pos_xy}

    integ_order = 2 * poly_order + 2
    err_vel_L2_neg = dh.l2_error_on_side(U_neg, exact_neg, level_set, 
                                         side='-', 
                                         fields=['u_neg_x', 'u_neg_y'], 
                                         relative=False,
                                         quad_order=integ_order)
    err_vel_L2_pos = dh.l2_error_on_side(U_pos, exact_pos, level_set, 
                                         side='+', 
                                         fields=['u_pos_x', 'u_pos_y'], 
                                         relative=False,
                                         quad_order=integ_order)
    vel_L2 = np.sqrt(err_vel_L2_neg**2 + err_vel_L2_pos**2)

    err_p_L2_neg = dh.l2_error_on_side(P_neg, exact_neg, 
                                       level_set, 
                                       side='-', 
                                       fields=['p_neg_'], 
                                       relative=False, quad_order=integ_order)
    err_p_L2_pos = dh.l2_error_on_side(P_pos, 
                                       exact_pos, 
                                       level_set, 
                                       side='+', 
                                       fields=['p_pos_'], 
                                       relative=False, 
                                       quad_order=integ_order)
    p_L2 = np.sqrt(err_p_L2_neg**2 + err_p_L2_pos**2)


    print(f"L2 Error (velocity): {vel_L2:10.8e}")
    # print(f"H1 Error (velocity): {vel_H1:10.8e}")
    print(f"L2 Error (pressure): {p_L2:10.8e}")
    # vel_H1 = dh.h1_error_vector_piecewise(U_pos, exact_neg, exact_pos, level_set)
    # H1-seminorm of velocity on each side
    vel_H1m_neg = dh.h1_error_vector_on_side(U_neg, grad_vel_neg_xy, 
                                             level_set, side='-', 
                                             fields=['u_neg_x', 'u_neg_y'], 
                                             quad_increase=2)
    vel_H1m_pos = dh.h1_error_vector_on_side(U_pos, grad_vel_pos_xy, 
                                             level_set, side='+', 
                                             fields=['u_pos_x', 'u_pos_y'], 
                                             quad_increase=2)
    vel_H1m = np.sqrt(vel_H1m_neg**2 + vel_H1m_pos**2)   # piecewise |·|_{H1}

    # If you want the *full* H1 norm (seminorm + L2)
    vel_H1 = np.sqrt(vel_L2**2 + vel_H1m**2)

    # print(f"H1-seminorm (velocity): {vel_H1m:10.8e}")
    print(f"H1 norm      (velocity): {vel_H1:10.8e}")

    # ---------------- 8) Export ----------------
    outdir = "stokes_interface_results"
    os.makedirs(outdir, exist_ok=True)
    vtk_path = os.path.join(outdir, "solution.vtu")
    phi_vals = level_set.evaluate_on_nodes(mesh)
    export_vtk(vtk_path, mesh=mesh, dof_handler=dh,
               functions={"velocity_pos": U_pos, "pressure_pos": P_pos, "velocity_neg": U_neg, "pressure_neg": P_neg, "phi": phi_vals})
    print(f"Solution exported to {vtk_path}")

if __name__ == "__main__":
    test_stokes_interface_corrected()
