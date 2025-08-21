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
    FacetNormal, CellDiameter, ElementWiseConstant
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


def test_stokes_interface_corrected():
    """
    Unfitted Stokes interface (NGSolve-style): Hansbo averaging, Nitsche coupling,
    ghost penalties, SymPy-manufactured RHS, and piecewise L2/H1 errors.
    This version is corrected to use the existing pycutfem API.
    """
    # ---------------- 1) Mesh, order, level set ----------------
    poly_order = 2
    maxh = 0.125
    mesh_size = int(2.0 / maxh)

    nodes, elems, _, corners = structured_quad(
        2.0, 2.0, nx=mesh_size, ny=mesh_size,
        poly_order=poly_order, offset=[-1.0, -1.0]
    )
    mesh = Mesh(nodes, element_connectivity=elems,
                elements_corner_nodes=corners,
                poly_order=poly_order, element_type='quad')

    mu = [Constant(1.0), Constant(10.0)]        # μ_in, μ_out
    R = 2.0/3.0
    gammaf = 0.5            # surface tension = pressure jump
    level_set = CircleLevelSet(center=(0.0, 0.0), radius=R)

    # Classify & interface
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    # Boundary tags (square boundary)
    boundary_tags = {
        'boundary': lambda x, y: np.isclose(x, -1.0) | np.isclose(x, 1.0) |
                                  np.isclose(y, -1.0) | np.isclose(y, 1.0)
    }
    mesh.tag_boundary_edges(boundary_tags)

    # BitSets
    inside_e  = mesh.element_bitset("inside")
    outside_e = mesh.element_bitset("outside")
    cut_e     = mesh.element_bitset("cut")
    ghost_e   = mesh.edge_bitset("ghost")

    has_inside  = inside_e  | cut_e
    has_outside = outside_e | cut_e

    # ---------------- 2) Mixed FE space & dofs ----------------
    me = MixedElement(mesh, field_specs={'ux': poly_order, 'uy': poly_order, 'p': poly_order-1})
    dh = DofHandler(me, method='cg')

    Vspace    = FunctionSpace("vel", ['ux', 'uy'])
    vel_trial = VectorTrialFunction(space=Vspace, dof_handler=dh)
    vel_test  = VectorTestFunction(space=Vspace,  dof_handler=dh)
    p_trial   = TrialFunction('p', dh)
    p_test    = TestFunction('p', dh)

    # Measures & parameters
    qvol   = 2*poly_order + 2
    dx_pos = dx(defined_on=has_outside, level_set=level_set, metadata={'side': '+', 'q': qvol})
    dx_neg = dx(defined_on=has_inside,  level_set=level_set, metadata={'side': '-', 'q': qvol})
    dGamma = dInterface(defined_on=cut_e,  level_set=level_set, metadata={'q': qvol})
    dG = dGhost(defined_on=ghost_e,    level_set=level_set, metadata={'q': qvol})

    h = CellDiameter()
    n = FacetNormal()

    # Hansbo weights θ^+, θ^- (per element); element-wise constants
    theta_pos_vals = hansbo_cut_ratio(mesh, level_set, side='+')
    theta_neg_vals = 1.0 - theta_pos_vals
    kappa_pos = ElementWiseConstant(theta_pos_vals)
    kappa_neg = ElementWiseConstant(theta_neg_vals)

    lambda_nitsche = Constant(0.5 * (mu[0].value + mu[1].value) * 20 * poly_order**2)
    gamma_stab_v   = Constant(0.05)
    gamma_stab_p   = Constant(0.05)

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

    # g_neg = Analytic(g_neg_xy)
    # g_pos = Analytic(g_pos_xy)
    g_neg = VectorFunction(name="g_neg", field_names=['ux', 'uy'], dof_handler=dh)
    g_pos = VectorFunction(name="g_pos", field_names=['ux', 'uy'], dof_handler=dh)
    g_neg.set_values_from_function(g_neg_xy)
    g_pos.set_values_from_function(g_pos_xy)

    # ---------------- 4) Variational forms (NGSolve-style) ----------------
    def epsilon(u):
        return 0.5 * (grad(u) + grad(u).T)

    def traction(mu_val, u_vec, p_scal, normal):
        return -2*mu_val * dot(epsilon(u_vec), normal) + p_scal * normal

    # volume terms
    a = (2*mu[1]*inner(epsilon(vel_trial), epsilon(vel_test)) - div(vel_trial)*p_test - div(vel_test)*p_trial) * dx_pos
    a += (2*mu[0]*inner(epsilon(vel_trial), epsilon(vel_test)) - div(vel_trial)*p_test - div(vel_test)*p_trial) * dx_neg

    sig_pos_n_trial = traction(mu[1], Pos(vel_trial), Pos(p_trial), n)
    sig_neg_n_trial = traction(mu[0], Neg(vel_trial), Neg(p_trial), n)
    avg_flux_trial  = kappa_neg * sig_pos_n_trial + kappa_pos * sig_neg_n_trial

    sig_pos_n_test = traction(mu[1], Pos(vel_test), Pos(p_test), n)
    sig_neg_n_test = traction(mu[0], Neg(vel_test), Neg(p_test), n)
    avg_flux_test  = kappa_neg * sig_pos_n_test + kappa_pos * sig_neg_n_test

    # Interface terms
    a += ( dot(avg_flux_trial, jump(vel_test))
          + dot(avg_flux_test,  jump(vel_trial))
          + (lambda_nitsche / h) * dot(jump(vel_trial), jump(vel_test)) ) * dGamma

    # Ghost penalty terms
    a += (gamma_stab_v / h**2) * dot(jump(vel_trial), jump(vel_test)) * dG
    a -= (gamma_stab_p * h**2) * jump(p_trial) * jump(p_test) * dG

    avg_inv_test = kappa_neg * Pos(vel_test) + kappa_pos * Neg(vel_test)
    f = dot(g_pos, vel_test) * dx_pos \
      + dot(g_neg, vel_test) * dx_neg \
      - gammaf * dot(avg_inv_test, n) * dGamma

    equation = Equation(a, f)

    # ---------------- 5) Velocity Dirichlet on outer boundary ----------------
    bcs = [
        BoundaryCondition('ux', 'dirichlet', 'boundary', lambda x, y: vel_exact_pos_xy(x, y)[0]),
        BoundaryCondition('uy', 'dirichlet', 'boundary', lambda x, y: vel_exact_pos_xy(x, y)[1]),
    ]

    # ---------------- 6) Assemble & solve ----------------
    K, F = assemble_form(equation, dof_handler=dh, bcs=bcs, quad_order=qvol)

    p0 = dh.get_field_slice('p')[0]
    K = K.tolil(); F = F.copy()
    K[p0, :] = 0.0; K[:, p0] = 0.0
    K[p0, p0] = 1.0
    F[p0] = 0.0

    try:
        sol = spla.spsolve(K.tocsc(), F)
    except Exception:
        sol = spla.lsqr(K.tocsc(), F)[0]

    # ---------------- 7) Wrap solution & calculate errors ----------------
    U = VectorFunction(name="velocity", field_names=['ux','uy'], dof_handler=dh)
    P = Function(name="pressure", field_name='p', dof_handler=dh)
    dh.add_to_functions(sol, [U, P])

    exact_neg = {'ux': lambda x,y: vel_exact_neg_xy(x,y)[0], 'uy': lambda x,y: vel_exact_neg_xy(x,y)[1], 'p': p_exact_neg_xy}
    exact_pos = {'ux': lambda x,y: vel_exact_pos_xy(x,y)[0], 'uy': lambda x,y: vel_exact_pos_xy(x,y)[1], 'p': p_exact_pos_xy}

    err_vel_L2_neg = dh.l2_error_on_side(U, exact_neg, level_set, side='-', fields=['ux', 'uy'], relative=False)
    err_vel_L2_pos = dh.l2_error_on_side(U, exact_pos, level_set, side='+', fields=['ux', 'uy'], relative=False)
    vel_L2 = np.sqrt(err_vel_L2_neg**2 + err_vel_L2_pos**2)

    err_p_L2_neg = dh.l2_error_on_side(P, exact_neg, level_set, side='-', fields=['p'], relative=False)
    err_p_L2_pos = dh.l2_error_on_side(P, exact_pos, level_set, side='+', fields=['p'], relative=False)
    p_L2 = np.sqrt(err_p_L2_neg**2 + err_p_L2_pos**2)

    vel_H1 = dh.h1_error_vector_piecewise(U, exact_neg, exact_pos, level_set)

    print(f"L2 Error (velocity): {vel_L2:10.8e}")
    print(f"H1 Error (velocity): {vel_H1:10.8e}")
    print(f"L2 Error (pressure): {p_L2:10.8e}")

    # ---------------- 8) Export ----------------
    outdir = "stokes_interface_results"
    os.makedirs(outdir, exist_ok=True)
    vtk_path = os.path.join(outdir, "solution.vtu")
    phi_vals = level_set.evaluate_on_nodes(mesh)
    export_vtk(vtk_path, mesh=mesh, dof_handler=dh,
               functions={"velocity": U, "pressure": P, "phi": phi_vals})
    print(f"Solution exported to {vtk_path}")

if __name__ == "__main__":
    test_stokes_interface_corrected()
