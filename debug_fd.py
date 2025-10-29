import numpy as np
import logging
import os
import sys
from dataclasses import dataclass

logging.getLogger("pycutfem").setLevel(logging.ERROR)

from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.adaptive_mesh_ls_numba import structured_quad_levelset_adaptive
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.utils.domain_manager import get_domain_bitset
from pycutfem.core.geometry import hansbo_cut_ratio

from pycutfem.fem.mixedelement import MixedElement

from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    TrialFunction, TestFunction, VectorTrialFunction, VectorTestFunction,
    Function, VectorFunction, Constant, grad, inner, dot, div, Jump,
    FacetNormal, CellDiameter, Pos, Neg, ElementWiseConstant, restrict,
    det, inv, trace, Hessian, jump, Identity
)
from pycutfem.ufl.measures import dx, dInterface
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.compilers import FormCompiler
H = 0.41
L = 2.2
D = 0.1
c_x, c_y = 0.1973, 0.2031
rho_f = 1.0
rho_s = 1000.0
mu_f = 1e-3
mu_s_val = 2.0e6
lambda_s_val = 0.5e6
beta_penalty = 90.0 * mu_f

NX, NY = 12, 8
poly_order = 2

nodes, elems, edges, corners = structured_quad_levelset_adaptive(
    Lx=L, Ly=H, nx=NX, ny=NY, poly_order=poly_order,
    level_set=CircleLevelSet(center=(c_x, c_y), radius=(D/2.0)*(1 + 0.1)),
    max_refine_level=1
)
mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners,
            element_type="quad", poly_order=poly_order)

level_set = CircleLevelSet(center=(c_x, c_y), radius=D/2.0)

mesh.classify_elements(level_set)
mesh.classify_edges(level_set)
mesh.build_interface_segments(level_set=level_set)

fluid_domain = get_domain_bitset(mesh, "element", "outside")
solid_domain = get_domain_bitset(mesh, "element", "inside")
cut_domain = get_domain_bitset(mesh, "element", "cut")
fluid_interface_domain = fluid_domain | cut_domain
solid_interface_domain = solid_domain | cut_domain
has_pos = fluid_domain | cut_domain
has_neg = solid_domain | cut_domain

# -----------------------------------------------------------------------------
# DOF handler and spaces
# -----------------------------------------------------------------------------
mixed_element = MixedElement(mesh, field_specs={
    'u_pos_x': poly_order,
    'u_pos_y': poly_order,
    'p_pos_': poly_order - 1,
    'vs_neg_x': poly_order - 1,
    'vs_neg_y': poly_order - 1,
    'd_neg_x': poly_order - 1,
    'd_neg_y': poly_order - 1,
})

dof_handler = DofHandler(mixed_element, method='cg')
dof_handler.classify_from_levelset(level_set)

mesh = dof_handler.mixed_element.mesh
n_cut = len(mesh.element_bitset("cut").to_indices())
n_ifc_edges = len(mesh.edge_bitset("interface").to_indices())
n_with_pts = sum(
    1
    for el in mesh.elements_list
    if getattr(el, "tag", None) == "cut" and len(getattr(el, "interface_pts", ())) == 2
)
print(f"[mesh stats] cut elements={n_cut}, interface edges={n_ifc_edges}, cut elems with pts={n_with_pts}")

# Tag inactive DOFs as in the example
dof_handler.tag_dofs_from_element_bitset("inactive", "u_pos_x", "inside", strict=True)
dof_handler.tag_dofs_from_element_bitset("inactive", "u_pos_y", "inside", strict=True)
dof_handler.tag_dofs_from_element_bitset("inactive", "p_pos_", "inside", strict=True)
dof_handler.tag_dofs_from_element_bitset("inactive", "vs_neg_x", "outside", strict=True)
dof_handler.tag_dofs_from_element_bitset("inactive", "vs_neg_y", "outside", strict=True)
dof_handler.tag_dofs_from_element_bitset("inactive", "d_neg_x", "outside", strict=True)
dof_handler.tag_dofs_from_element_bitset("inactive", "d_neg_y", "outside", strict=True)

velocity_fluid_space = FunctionSpace(name="velocity_fluid", field_names=['u_pos_x', 'u_pos_y'], dim=1, side='+')
pressure_fluid_space = FunctionSpace(name="pressure_fluid", field_names=['p_pos_'], dim=0, side='+')
velocity_solid_space = FunctionSpace(name="velocity_solid", field_names=['vs_neg_x', 'vs_neg_y'], dim=1, side='-')
displacement_space = FunctionSpace(name="displacement", field_names=['d_neg_x', 'd_neg_y'], dim=1, side='-')

du_f = VectorTrialFunction(space=velocity_fluid_space, dof_handler=dof_handler)
dp_f = TrialFunction(name='trial_pressure_fluid', field_name='p_pos_', dof_handler=dof_handler, side='+')
du_s = VectorTrialFunction(space=velocity_solid_space, dof_handler=dof_handler)
ddisp_s = VectorTrialFunction(space=displacement_space, dof_handler=dof_handler)

test_vel_f = VectorTestFunction(space=velocity_fluid_space, dof_handler=dof_handler)
test_q_f = TestFunction(name='test_pressure_fluid', field_name='p_pos_', dof_handler=dof_handler, side='+')
test_vel_s = VectorTestFunction(space=velocity_solid_space, dof_handler=dof_handler)
test_disp_s = VectorTestFunction(space=displacement_space, dof_handler=dof_handler)

uf_k = VectorFunction(name="u_f_k", field_names=['u_pos_x', 'u_pos_y'], dof_handler=dof_handler, side='+')
pf_k = Function(name="p_f_k", field_name='p_pos_', dof_handler=dof_handler, side='+')
us_k = VectorFunction(name="u_s_k", field_names=['vs_neg_x', 'vs_neg_y'], dof_handler=dof_handler, side='-')
disp_k = VectorFunction(name="disp_k", field_names=['d_neg_x', 'd_neg_y'], dof_handler=dof_handler, side='-')
uf_n = VectorFunction(name="u_f_n", field_names=['u_pos_x', 'u_pos_y'], dof_handler=dof_handler, side='+')
pf_n = Function(name="p_f_n", field_name='p_pos_', dof_handler=dof_handler, side='+')
us_n = VectorFunction(name="u_s_n", field_names=['vs_neg_x', 'vs_neg_y'], dof_handler=dof_handler, side='-')
disp_n = VectorFunction(name="disp_n", field_names=['d_neg_x', 'd_neg_y'], dof_handler=dof_handler, side='-')

# Randomize states to avoid trivial residuals
rng = np.random.default_rng(123)
for func in [uf_k, us_k, disp_k, uf_n, us_n, disp_n]:
    func.nodal_values[:] = rng.normal(scale=1e-5, size=func.nodal_values.shape)
for func in [pf_k, pf_n]:
    func.nodal_values[:] = rng.normal(scale=1e-5, size=func.nodal_values.shape)

use_restricted_forms = True
if use_restricted_forms:
    du_f_R        = restrict(du_f, has_pos)
    dp_f_R        = restrict(dp_f, has_pos)
    test_vel_f_R  = restrict(test_vel_f, has_pos)
    test_q_f_R    = restrict(test_q_f, has_pos)
    uf_k_R        = restrict(uf_k, has_pos)
    pf_k_R        = restrict(pf_k, has_pos)
    uf_n_R        = restrict(uf_n, has_pos)
    pf_n_R        = restrict(pf_n, has_pos)

    du_s_R        = restrict(du_s, has_neg)
    ddisp_s_R     = restrict(ddisp_s, has_neg)
    test_vel_s_R  = restrict(test_vel_s, has_neg)
    test_disp_s_R = restrict(test_disp_s, has_neg)
    us_k_R        = restrict(us_k, has_neg)
    us_n_R        = restrict(us_n, has_neg)
    disp_k_R      = restrict(disp_k, has_neg)
    disp_n_R      = restrict(disp_n, has_neg)
else:
    du_f_R, dp_f_R, test_vel_f_R, test_q_f_R = du_f, dp_f, test_vel_f, test_q_f
    uf_k_R, pf_k_R, uf_n_R, pf_n_R = uf_k, pf_k, uf_n, pf_n
    du_s_R, ddisp_s_R = du_s, ddisp_s
    test_vel_s_R, test_disp_s_R = test_vel_s, test_disp_s
    us_k_R, us_n_R = us_k, us_n
    disp_k_R, disp_n_R = disp_k, disp_n

# -----------------------------------------------------------------------------
# Measures and constants
# -----------------------------------------------------------------------------
dt = Constant(0.2)
theta = Constant(1.0)
mu_f_const = Constant(mu_f)
rho_f_const = Constant(rho_f)
rho_s_const = Constant(rho_s)
mu_s = Constant(mu_s_val)
lambda_s = Constant(lambda_s_val)

n = FacetNormal()
cell_h = CellDiameter()
beta_N = Constant(beta_penalty * poly_order * (poly_order + 1))

qvol = 5
dx_fluid = dx(defined_on=fluid_interface_domain, level_set=level_set,
              metadata={"q": qvol, "side": "+"})
dx_solid = dx(defined_on=solid_interface_domain, level_set=level_set,
              metadata={"q": qvol, "side": "-"})
dGamma = dInterface(
    defined_on=mesh.element_bitset("cut"),
    level_set=level_set,
    metadata={
        "q": qvol + 2,
        "derivs": {(0, 0), (0, 1), (1, 0)},
        "owner": "+",
        "mortar_pairs": [("u_pos", "vs_neg")],
        "debug_mortar": False,
    },
)

# Hansbo ratios for potential use (not essential for kappa=0.5 but kept for completeness)
theta_min = 1.0e-3
theta_pos_vals = np.clip(hansbo_cut_ratio(mesh, level_set, side="+"), theta_min, 1.0)
theta_neg_vals = np.clip(hansbo_cut_ratio(mesh, level_set, side="-"), theta_min, 1.0)

# -----------------------------------------------------------------------------
# Helper tensor functions
# -----------------------------------------------------------------------------
from pycutfem.ufl.expressions import Identity
I2 = Identity(2)

def epsilon_f(u):
    return Constant(0.5) * (grad(u) + grad(u).T)

def F_of(d):
    return grad(d) + I2

def C_of(F):
    return dot(F.T, F)

def E_of(F):
    return Constant(0.5) * (C_of(F) - I2)

def S_stvk(E):
    return lambda_s * trace(E) * I2 + Constant(2.0) * mu_s * E

def sigma_s_nonlinear(d):
    F = F_of(d)
    E = E_of(F)
    S = S_stvk(E)
    J = det(F)
    return Constant(1.0) / J * dot(dot(F, S), F.T)

def dsigma_s(d_ref, delta_d):
    Fk = F_of(d_ref)
    Ek = E_of(Fk)
    Sk = S_stvk(Ek)
    dF = grad(delta_d)
    dE = Constant(0.5) * (dot(dF.T, Fk) + dot(Fk.T, dF))
    dS = lambda_s * trace(dE) * I2 + Constant(2.0) * mu_s * dE
    Jk = det(Fk)
    Finv = inv(Fk)
    dJ = Jk * trace(dot(Finv, dF))
    term = dot(dF, dot(Sk, Fk.T)) + dot(Fk, dot(dS, Fk.T)) + dot(Fk, dot(Sk, dF.T))
    return Constant(1.0) / Jk * term - (dJ / Jk) * sigma_s_nonlinear(d_ref)

def traction_fluid(u_vec, p_scal):
    return Constant(2.0) * mu_f_const * dot(epsilon_f(u_vec), n) - p_scal * n

def traction_solid_R(d):
    return dot(sigma_s_nonlinear(d), n)

def traction_solid_L(delta_d, d_ref):
    return dot(dsigma_s(d_ref, delta_d), n)

def delta_E_GreenLagrange(w, u_ref):
    F_ref = F_of(u_ref)
    grad_w = grad(w)
    return Constant(0.5) * (dot(grad_w.T, F_ref) + dot(F_ref.T, grad_w))
# **Second variation** used by the geometric tangent
def delta_delta_E(v, du):
    return Constant(0.5) * (dot(grad(du).T, grad(v)) + dot(grad(v).T, grad(du)))

# --- Boundary second variation for StVK (Γ) ---
def d2sigma_s(d_ref, du, w):
    Fk   = F_of(d_ref)
    Jk   = det(Fk)
    Finv = inv(Fk)
    Sk   = S_stvk(E_of(Fk))

    dFk  = grad(du)   # trial direction u
    Aw   = grad(w)    # test direction w

    # First-level variations
    dEk   = Constant(0.5) * (dot(dFk.T, Fk) + dot(Fk.T, dFk))
    dSk   = lambda_s * trace(dEk) * I2 + Constant(2.0) * mu_s * dEk

    dEw   = Constant(0.5) * (dot(Aw.T, Fk) + dot(Fk.T, Aw))
    dSw   = lambda_s * trace(dEw) * I2 + Constant(2.0) * mu_s * dEw

    # Cross (second) variation
    ddEw  = Constant(0.5) * (dot(Aw.T, dFk) + dot(dFk.T, Aw))
    ddSw  = lambda_s * trace(ddEw) * I2 + Constant(2.0) * mu_s * ddEw

    # T[w] and δT[w;u]
    T_w   = dot(Aw, dot(Sk, Fk.T)) + dot(Fk, dot(dSw, Fk.T)) + dot(Fk, dot(Sk, Aw.T))
    dT    = (
        dot(Aw, dot(dSk, Fk.T)) + dot(Aw, dot(Sk, dFk.T))
      + dot(dFk, dot(dSw, Fk.T)) + dot(Fk, dot(ddSw, Fk.T)) + dot(Fk, dot(dSw, dFk.T))
      + dot(dFk, dot(Sk, Aw.T)) + dot(Fk, dot(dSk, Aw.T))
    )

    tr_Finv_dFk      = trace(dot(Finv, dFk))
    tr_Finv_Aw       = trace(dot(Finv, Aw))
    tr_Finv_dFk_FAw  = trace(dot(Finv, dot(dFk, dot(Finv, Aw))))

    sigma_k  = sigma_s_nonlinear(d_ref)
    ds_u     = dsigma_s(d_ref, du)

    # δ(δσ[w])[u]
    return (
        Constant(1.0)/Jk * dT
        - (tr_Finv_dFk / Jk) * T_w
        + tr_Finv_dFk_FAw * sigma_k
        - tr_Finv_Aw * ds_u
    )

def dtraction_solid_ref_L(du, w, d_ref):
    # derivative of traction_solid_L(w,d_ref) w.r.t. d_ref in direction du
    return dot(d2sigma_s(d_ref, du, w), n)


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
    kappa_pos * traction_fluid(Pos(test_vel_f), -Pos(test_q_f))
    - kappa_neg * traction_solid_L(Neg(test_disp_s), Neg(disp_k))
)

avg_flux_res = (
    kappa_pos * traction_fluid(Pos(uf_k), Pos(pf_k))
    - kappa_neg * traction_solid_R(Neg(disp_k))
)

# Decompose interface terms for diagnostics
J_terms = {
    "trial_fluid": (-kappa_pos * dot(traction_fluid(Pos(du_f), Pos(dp_f)), jump_vel_test)) * dGamma,
    "trial_solid": (kappa_neg * dot(traction_solid_L(Neg(ddisp_s), Neg(disp_k)), jump_vel_test)) * dGamma,
    "test_fluid": (-kappa_pos * dot(traction_fluid(Pos(test_vel_f), -Pos(test_q_f)), jump_vel_trial)) * dGamma,
    "test_solid": (kappa_neg * dot(traction_solid_L(Neg(test_disp_s), Neg(disp_k)), jump_vel_trial)) * dGamma,
    "penalty": ((beta_N * mu_f_const / cell_h) * dot(jump_vel_trial, jump_vel_test)) * dGamma,
}
# J_int = sum(J_terms.values())

R_terms = {
    "res_fluid": (-kappa_pos * dot(traction_fluid(Pos(uf_k), Pos(pf_k)), jump_vel_test)) * dGamma,
    "res_solid": (kappa_neg * dot(traction_solid_R(Neg(disp_k)), jump_vel_test)) * dGamma,
    "test_fluid": (-kappa_pos * dot(traction_fluid(Pos(test_vel_f), -Pos(test_q_f)), jump_vel_res)) * dGamma,
    "test_solid": (kappa_neg * dot(traction_solid_L(Neg(test_disp_s), Neg(disp_k)), jump_vel_res)) * dGamma,
    "penalty": ((beta_N * mu_f_const / cell_h) * dot(jump_vel_res, jump_vel_test)) * dGamma,
}
# R_int = sum(R_terms.values())

TEST_STAGE = "interface_only"  # "basic", "elastic", "full", "interface_only"

if TEST_STAGE == "basic":
    J_int = None
    R_int = None
    a_vol_s = inner(grad(ddisp_s_R), grad(test_disp_s_R)) * dx_solid
    r_vol_s = inner(grad(disp_k_R), grad(test_disp_s_R)) * dx_solid
elif TEST_STAGE == "elastic":
    J_int = None
    R_int = None
    S_k = S_stvk(E_of(F_of(disp_k_R)))
    S_n = S_stvk(E_of(F_of(disp_n_R)))
    delta_E_test_k = delta_E_GreenLagrange(test_disp_s_R, disp_k_R)
    delta_E_test_n = delta_E_GreenLagrange(test_disp_s_R, disp_n_R)
    delta_E_trial_k = delta_E_GreenLagrange(ddisp_s_R, disp_k_R)
    C_delta_E_trial = lambda_s * trace(delta_E_trial_k) * I2 + Constant(2.0) * mu_s * delta_E_trial_k
    material_stiffness_a = inner(C_delta_E_trial, delta_E_test_k)
    # sym_grad_test = Constant(0.5) * (grad(test_disp_s_R) + grad(test_disp_s_R).T)
    # delta_delta_E_test = delta_E_GreenLagrange(test_disp_s_R, ddisp_s_R) - sym_grad_test
    geometric_stiffness_a = inner(S_k, delta_delta_E(test_disp_s_R, ddisp_s_R))

    stiffness = material_stiffness_a + geometric_stiffness_a
    a_vol_s = (
        rho_s_const * dot(du_s_R, test_vel_s_R) / dt
        + theta * stiffness
    ) * dx_solid
    r_vol_s = (
        rho_s_const * dot(us_k_R - us_n_R, test_vel_s_R) / dt
        + theta * inner(S_k, delta_E_test_k)
        + (Constant(1.0) - theta) * inner(S_n, delta_E_test_n)
    ) * dx_solid
elif TEST_STAGE == "interface_only":
    a_vol_s = None
    r_vol_s = None
    J_int = (
        - dot(avg_flux_trial, jump_vel_test)
        - dot(avg_flux_test, jump_vel_trial)
        + (beta_N * mu_f_const / cell_h) * dot(jump_vel_trial, jump_vel_test)
    ) * dGamma

    # --- add the missing cross-derivative on the SOLID side (general, side-aware) ---
    J_int += (
         kappa_neg * dot( dtraction_solid_ref_L(Neg(ddisp_s_R), Neg(test_disp_s_R), Neg(disp_k_R)),
                        jump_vel_res )
    ) * dGamma

    R_int = (
        - dot(avg_flux_res, jump_vel_test)
        - dot(avg_flux_test, jump_vel_res)
        + (beta_N * mu_f_const / cell_h) * dot(jump_vel_res, jump_vel_test)
    ) * dGamma
else:
    J_int = (
        - dot(avg_flux_trial, jump_vel_test)
        - dot(avg_flux_test, jump_vel_trial)
        + (beta_N * mu_f_const / cell_h) * dot(jump_vel_trial, jump_vel_test)
    ) * dGamma

    R_int = (
        - dot(avg_flux_res, jump_vel_test)
        - dot(avg_flux_test, jump_vel_res)
        + (beta_N * mu_f_const / cell_h) * dot(jump_vel_res, jump_vel_test)
    ) * dGamma

    S_k = S_stvk(E_of(F_of(disp_k_R)))
    S_n = S_stvk(E_of(F_of(disp_n_R)))

    delta_E_test_k = delta_E_GreenLagrange(test_disp_s_R, disp_k_R)
    delta_E_test_n = delta_E_GreenLagrange(test_disp_s_R, disp_n_R)
    delta_E_trial_k = delta_E_GreenLagrange(ddisp_s_R, disp_k_R)

    C_delta_E_trial = lambda_s * trace(delta_E_trial_k) * I2 + Constant(2.0) * mu_s * delta_E_trial_k
    material_stiffness_a = inner(C_delta_E_trial, delta_E_test_k)

    geometric_stiffness_a = inner(
        S_k,
        Constant(0.5) * (dot(grad(ddisp_s_R).T, grad(test_disp_s_R))
                         + dot(grad(test_disp_s_R).T, grad(ddisp_s_R)))
    )

    a_vol_s = (
        rho_s_const * dot(du_s_R, test_vel_s_R) / dt
        + theta * (material_stiffness_a + geometric_stiffness_a)
    ) * dx_solid

    r_vol_s = (
        rho_s_const * dot(us_k_R - us_n_R, test_vel_s_R) / dt
        + theta * inner(S_k, delta_E_test_k)
        + (Constant(1.0) - theta) * inner(S_n, delta_E_test_n)
    ) * dx_solid

# -----------------------------------------------------------------------------
# Assembly helpers
# -----------------------------------------------------------------------------
@dataclass
class Assembled:
    K: any
    R: any

def assemble_pair(j_form, r_form, backend='python'):
    compiler = FormCompiler(dof_handler, backend=backend)
    K, _ = compiler.assemble(Equation(j_form, None), bcs=[])
    compiler = FormCompiler(dof_handler, backend=backend)
    _, R = compiler.assemble(Equation(None, r_form), bcs=[])
    return Assembled(K, R)
functions_by_field = {
    'u_pos_x': uf_k,
    'u_pos_y': uf_k,
    'vs_neg_x': us_k,
    'vs_neg_y': us_k,
    'd_neg_x': disp_k,
    'd_neg_y': disp_k,
}
def perturb(field, gdof, new_value):
    func = functions_by_field[field]
    old = func.get_nodal_values(np.array([gdof], dtype=int))[0]
    func.set_nodal_values(np.array([gdof], dtype=int), np.array([new_value]))
    return old
def fd_errors(base_pair, assemble_fn, dofs, eps=1e-6):
    rows = []
    debug_snapshot = {}
    for gdof in dofs:
        field, _ = dof_handler._dof_to_node_map[gdof]
        if field not in functions_by_field:
            continue
        old = functions_by_field[field].get_nodal_values(np.array([gdof], dtype=int))[0]
        perturb(field, gdof, old + eps)
        plus = assemble_fn()
        R_plus = plus.R.copy()
        perturb(field, gdof, old - eps)
        minus = assemble_fn()
        R_minus = minus.R.copy()
        perturb(field, gdof, old)
        fd_col = (R_plus - R_minus) / (2 * eps)
        jac_col = base_pair.K[:, gdof].toarray().ravel()
        err_vec = fd_col - jac_col
        err = np.linalg.norm(err_vec, ord=np.inf)
        mag = np.linalg.norm(jac_col, ord=np.inf)
        rel = err / (mag + 1e-14)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(fd_col, jac_col, where=np.abs(jac_col) > 1e-12)
        if gdof not in debug_snapshot:
            debug_snapshot[gdof] = (jac_col.copy(), fd_col.copy(), err_vec.copy())
        rows.append((gdof, field, err, mag, rel, ratio))
    return rows, debug_snapshot



BACKEND = "jit"


fields_to_probe = {
    'u_pos_x': 3,
    'u_pos_y': 3,
    'vs_neg_x': 3,
    'vs_neg_y': 3,
    'd_neg_x': 3,
    'd_neg_y': 3,
}
selected_dofs = []
cut_eids = mesh.element_bitset("cut").to_indices()
probe_eid = int(cut_eids[0]) if len(cut_eids) else 0
for field, count in fields_to_probe.items():
    try:
        local = dof_handler.element_dofs(field, probe_eid)
    except Exception:
        local = []
    selected_dofs.extend(list(local[:count]))
selected_dofs = np.array(sorted(set(selected_dofs)), dtype=int)

J_cross_deriv_s = (
     kappa_neg * dot( dtraction_solid_ref_L(Neg(ddisp_s_R), Neg(test_disp_s_R), Neg(disp_k_R)),
                    jump_vel_res )
) * dGamma
pair_int_components = {} if J_int is None else {name: assemble_pair(form, None, backend=BACKEND) for name, form in J_terms.items()}
pair_res_components = {} if R_int is None else {name: assemble_pair(None, form, backend=BACKEND) for name, form in R_terms.items()}
J_variants = {} if J_int is None else {
    "fluid_only": J_terms["trial_fluid"] + J_terms["test_fluid"],
    "solid_only": J_terms["trial_solid"] + J_terms["test_solid"] + J_cross_deriv_s,
    "penalty_only": J_terms["penalty"],
    "no_penalty": J_int - J_terms["penalty"],
}
pair_int_variants = {} if not J_variants else {name: assemble_pair(form, None, backend=BACKEND) for name, form in J_variants.items()}
R_variants = {} if R_int is None else {
    "fluid_only": R_terms["res_fluid"] + R_terms["test_fluid"],
    "solid_only": R_terms["res_solid"] + R_terms["test_solid"],
    "penalty_only": R_terms["penalty"],
    "no_penalty": R_int - R_terms["penalty"],
}
pair_res_variants = {} if not R_variants else {name: assemble_pair(None, form, backend=BACKEND) for name, form in R_variants.items()}
pair_variants_combined = {}
if J_variants:
    for name, form in J_variants.items():
        R_form = R_variants.get(name) if R_variants else None
        pair_variants_combined[name] = assemble_pair(form, R_form, backend=BACKEND)
variant_fd_results = {}
if pair_variants_combined:
    for name, pair in pair_variants_combined.items():
        J_form = J_variants.get(name)
        R_form = R_variants.get(name) if R_variants else None
        def _builder(Jfrm=J_form, Rfrm=R_form):
            return assemble_pair(Jfrm, Rfrm, backend=BACKEND)
        rows_variant, _ = fd_errors(pair, _builder, selected_dofs, eps=1e-8)
        variant_fd_results[name] = rows_variant
elif J_variants:
    for name, form in J_variants.items():
        def _builder(frm=form):
            return assemble_pair(frm, None, backend=BACKEND)
        base_variant = _builder()
        rows_variant, _ = fd_errors(base_variant, _builder, selected_dofs, eps=1e-8)
        variant_fd_results[name] = rows_variant
pair_int = assemble_pair(J_int, R_int, backend=BACKEND) if J_int is not None else None
pair_solid = assemble_pair(a_vol_s, r_vol_s, backend=BACKEND)
mass_form = (rho_s_const * dot(du_s_R, test_vel_s_R) / dt) * dx_solid
mass_res = (rho_s_const * dot(us_k_R - us_n_R, test_vel_s_R) / dt) * dx_solid
pair_mass = assemble_pair(mass_form, mass_res, backend=BACKEND)

if pair_int is not None:
    print("Interface Jacobian nnz:", pair_int.K.nnz)
    print("Interface residual |R|_inf:", np.linalg.norm(pair_int.R, ord=np.inf))
else:
    print("Interface Jacobian nnz: 0")
    print("Interface residual |R|_inf: 0.0")
print("Solid Jacobian nnz:", pair_solid.K.nnz)
print("Solid residual |R|_inf:", np.linalg.norm(pair_solid.R, ord=np.inf))
print("Mass-only Jacobian nnz:", pair_mass.K.nnz)
if pair_int_components:
    print(
"Interface component summary:")
    for name, pair in pair_int_components.items():
        data = pair.K.data
        norm = np.linalg.norm(data, ord=np.inf) if data.size else 0.0
        print(f"  {name:12s} nnz={pair.K.nnz:6d} |K|_inf={norm:.3e}")
if pair_int_variants:
    print(
"Interface variant summary:")
    for name, pair in pair_int_variants.items():
        data = pair.K.data
        norm = np.linalg.norm(data, ord=np.inf) if data.size else 0.0
        print(f"  {name:12s} nnz={pair.K.nnz:6d} |K|_inf={norm:.3e}")
if pair_res_components:
    print(
"Interface residual components:")
    for name, pair in pair_res_components.items():
        norm = np.linalg.norm(pair.R, ord=np.inf)
        print(f"  {name:12s} |R|_inf={norm:.3e}")
if pair_res_variants:
    print(
"Interface residual variants:")
    for name, pair in pair_res_variants.items():
        norm = np.linalg.norm(pair.R, ord=np.inf)
        print(f"  {name:12s} |R|_inf={norm:.3e}")

# -----------------------------------------------------------------------------
# Finite difference comparison
# -----------------------------------------------------------------------------

if os.environ.get("SKIP_FD_CHECKS") == "1":
    print("Skipping finite-difference checks (SKIP_FD_CHECKS=1).")
    sys.exit(0)









pair_builder_int = (lambda: assemble_pair(J_int, R_int, backend=BACKEND)) if pair_int is not None else None
pair_builder_solid = lambda: assemble_pair(a_vol_s, r_vol_s, backend=BACKEND)

if pair_int is not None:
    print("\nInterface FD check (gdof, field, err_inf, |J_col|_inf, rel):")
    fd_rows_int, dbg_int = fd_errors(pair_int, pair_builder_int, selected_dofs, eps=1e-8)
    sample_dof_int = next((gd for gd, fld, *_ in fd_rows_int if fld.startswith('d_neg')), None)
    if sample_dof_int is not None:
        jac_col_int, fd_col_int, diff_int = dbg_int[sample_dof_int]
        print(f"\nInterface detailed diff for displacement DOF {sample_dof_int}:")
        order = np.abs(diff_int).argsort()[::-1][:10]
        for idx_row in order:
            field_i, _ = dof_handler._dof_to_node_map[idx_row]
            print(f"  row {idx_row:5d} ({field_i:8s}): J={jac_col_int[idx_row]: .6e}, FD={fd_col_int[idx_row]: .6e}, Δ={diff_int[idx_row]: .6e}")
        if pair_int_components:
            for name, pair in pair_int_components.items():
                col = pair.K[:, sample_dof_int].toarray().ravel()
                print(f"    component {name:12s} col |.|_inf={np.linalg.norm(col, ord=np.inf):.3e}")
                if name == "trial_solid":
                    top_rows = np.argsort(np.abs(col))[-5:][::-1]
                    print("      trial_solid top rows:")
                    for ridx in top_rows:
                        field_i, _ = dof_handler._dof_to_node_map[ridx]
                        print(f"        row {ridx:5d} ({field_i:8s}) = {col[ridx]: .6e}")
        if pair_int_variants:
            for name, pair in pair_int_variants.items():
                col = pair.K[:, sample_dof_int].toarray().ravel()
                print(f"    variant   {name:12s} col |.|_inf={np.linalg.norm(col, ord=np.inf):.3e}")
        if pair_int_components:
            sum_col = sum((pair.K[:, sample_dof_int].toarray().ravel() for pair in pair_int_components.values()))
            agg_col = pair_int.K[:, sample_dof_int].toarray().ravel()
            diff_sum = np.linalg.norm(sum_col - agg_col, ord=np.inf)
            print(f"    aggregated column |.|_inf={np.linalg.norm(agg_col, ord=np.inf):.3e}, sum components diff={diff_sum:.3e}")
        if pair_res_components:
            for name, pair in pair_res_components.items():
                val = pair.R[sample_dof_int] if sample_dof_int < pair.R.size else 0.0
                print(f"    residual component {name:12s} entry={val:.3e}")
        if pair_res_variants:
            for name, pair in pair_res_variants.items():
                val = pair.R[sample_dof_int] if sample_dof_int < pair.R.size else 0.0
                print(f"    residual variant   {name:12s} entry={val:.3e}")
        if variant_fd_results:
            for name, rows in variant_fd_results.items():
                match = next((vals for vals in rows if vals[0] == sample_dof_int), None)
                if match:
                    _, fld_name, err_v, mag_v, rel_v, _ = match
                    print(f"    variant FD {name:12s}: err={err_v:.3e} |J|={mag_v:.3e} rel={rel_v:.3e}")
    for gd, fld, err, mag, rel, _ in fd_rows_int:
        print(f"  {gd:5d}  {fld:10s}  err={err:9.3e}  |J|={mag:9.3e}  rel={rel:9.3e}")
    nan_rows_int = np.where(np.isnan(pair_int.R))[0]
    if nan_rows_int.size:
        print(" Interface residual NaNs at rows:", nan_rows_int[:10])

print("\nSolid FD check (gdof, field, err_inf, |J_col|_inf, rel, ratio[min,max]):")
fd_rows_solid, dbg_solid = fd_errors(pair_solid, pair_builder_solid, selected_dofs, eps=1e-8)
for gd, fld, err, mag, rel, ratio in fd_rows_solid:
    finite = ratio[np.isfinite(ratio)]
    if finite.size:
        rmin, rmax = finite.min(), finite.max()
    else:
        rmin = rmax = float('nan')
    print(f"  {gd:5d}  {fld:10s}  err={err:9.3e}  |J|={mag:9.3e}  rel={rel:9.3e}  "
          f"ratio[{rmin:9.3e}, {rmax:9.3e}]")

sample_dof = next((gd for gd, fld, *_ in fd_rows_solid if fld == 'd_neg_x'), None)
if sample_dof is not None:
    jac_col, fd_col, diff = dbg_solid[sample_dof]
    idxs = np.abs(diff).argsort()[::-1][:10]
    print(f"\nDetailed diff for displacement DOF {sample_dof}:")
    for idx in idxs:
        field_i, _ = dof_handler._dof_to_node_map[idx]
        print(f"  row {idx:5d} ({field_i:8s}): J={jac_col[idx]: .6e}, "
              f"FD={fd_col[idx]: .6e}, Δ={diff[idx]: .6e}")
    mass_col = pair_mass.K[:, sample_dof].toarray().ravel()
    if np.any(np.abs(mass_col) > 1e-9):
        rows_nz = np.where(np.abs(mass_col) > 1e-9)[0][:10]
        print("\nMass term nonzeros for same DOF:")
        for idx in rows_nz:
            field_i, _ = dof_handler._dof_to_node_map[idx]
            print(f"  row {idx:5d} ({field_i:8s}): M={mass_col[idx]: .6e}")

print("\nSample DOF field mapping:")
for gd in [813, 819, 923, 925]:
    print(gd, dof_handler._dof_to_node_map[gd])
