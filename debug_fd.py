import numpy as np
import logging
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
dGamma = dInterface(defined_on=mesh.element_bitset("cut"), level_set=level_set,
                    metadata={"q": qvol + 2, "derivs": {(0,0), (0,1), (1,0)}})

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

TEST_STAGE = "elastic"  # "basic", "elastic", "full"

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
    sym_grad_test = Constant(0.5) * (grad(test_disp_s_R) + grad(test_disp_s_R).T)
    delta_delta_E_test = delta_E_GreenLagrange(test_disp_s_R, ddisp_s_R) - sym_grad_test
    geometric_stiffness_a = inner(S_k, delta_delta_E_test)
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
else:
    J_int = None
    R_int = None

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

BACKEND = "python"

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

# -----------------------------------------------------------------------------
# Finite difference comparison
# -----------------------------------------------------------------------------
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

pair_builder_int = (lambda: assemble_pair(J_int, R_int, backend=BACKEND)) if pair_int is not None else None
pair_builder_solid = lambda: assemble_pair(a_vol_s, r_vol_s, backend=BACKEND)

if pair_int is not None:
    print("\nInterface FD check (gdof, field, err_inf, |J_col|_inf, rel):")
    fd_rows_int, dbg_int = fd_errors(pair_int, pair_builder_int, selected_dofs, eps=1e-8)
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
