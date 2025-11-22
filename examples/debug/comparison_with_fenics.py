import numpy as np
import pandas as pd
import os

# FEniCSx imports
from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import basix
import basix.ufl
from basix.ufl import mixed_element
from petsc4py import PETSc # Import PETSc for enums like InsertMode

# pycutfem imports
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    TrialFunction, TestFunction, VectorTrialFunction, VectorTestFunction,
    Function, VectorFunction, Constant, grad, inner,
    dot, div, trace, Hessian, Laplacian, FacetNormal, Identity, det, inv
)
from pycutfem.ufl.measures import dx, dInterface, dS
from pycutfem.ufl.forms import assemble_form
from pycutfem.fem.reference import get_reference
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.forms import Equation

# Imports for mapping and matrix conversion
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
import logging
# logging.basicConfig(
#     level=logging.INFO,  # show debug messages
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )

I2 = Identity(2)
lambda_s = Constant(0.0)
mu_s = Constant(0.0)

# ------------------------------------------------------------------
#  Nonlinear solid helpers (St. Venant–Kirchhoff)
# ------------------------------------------------------------------


def F_of(d):
    """Deformation gradient of the displacement field in reference coordinates."""
    return I2 + grad(d)


def C_of(F):
    """Right Cauchy–Green tensor."""
    return dot(F.T, F)


def E_of(F):
    """Green–Lagrange strain."""
    return Constant(0.5) * (C_of(F) - I2)


def S_stvk(E):
    """Second Piola–Kirchhoff stress for StVK material."""
    return lambda_s * trace(E) * I2 + Constant(2.0) * mu_s * E


def sigma_s_nonlinear(d):
    """Cauchy stress for the current configuration."""
    F = F_of(d)
    E = E_of(F)
    S = S_stvk(E)
    J = det(F)
    return Constant(1.0) / J * dot(dot(F, S), F.T)


def dsigma_s(d_ref, delta_d):
    """Directional derivative of the Cauchy stress w.r.t. displacement."""
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


def traction_solid_L(delta_d, d_ref):
    """Linearized traction in direction delta_d."""
    return dot(dsigma_s(d_ref, delta_d), n)


D2SIGMA_COMPONENT_LABELS = (
    "A: invJ*dT",
    "B: -(tr(F^{-1} dF) over J)*T_w",
    "C: tr(F^{-1} dF F^{-1} Aw)*sigma",
    "D: -tr(F^{-1} Aw)*ds_u",
)
D2SIGMA_TERM_A_SUBLABELS = (
    "A1: invJ*Aw_dot_dSk_FT",
    "A2: invJ*Aw_dot_Sk_dF_T",
    "A3: invJ*dF_dot_dSw_FT",
    "A4: invJ*F_dot_ddSw_FT",
    "A5: invJ*F_dot_dSw_dF_T",
    "A6: invJ*dF_dot_Sk_Aw_T",
    "A7: invJ*F_dot_dSk_Aw_T",
)


def _d2sigma_s_components(d_ref, du_trial, w_test, *, return_parts: bool = False):
    comps = {}
    Fk = F_of(d_ref)
    comps['Fk'] = Fk
    Jk = det(Fk)
    comps['Jk'] = Jk
    Finv = inv(Fk)
    comps['Finv'] = Finv
    Sk = S_stvk(E_of(Fk))
    comps['Sk'] = Sk

    dFk_trial = grad(du_trial)
    comps['dFk_trial'] = dFk_trial
    Aw_test = grad(w_test)
    comps['Aw_test'] = Aw_test

    dEk_trial = Constant(0.5) * (dot(dFk_trial.T, Fk) + dot(Fk.T, dFk_trial))
    comps['dEk_trial'] = dEk_trial
    dSk_trial = lambda_s * trace(dEk_trial) * I2 + Constant(2.0) * mu_s * dEk_trial
    comps['dSk_trial'] = dSk_trial

    dEw_test = Constant(0.5) * (dot(Aw_test.T, Fk) + dot(Fk.T, Aw_test))
    comps['dEw_test'] = dEw_test
    dSw_test = lambda_s * trace(dEw_test) * I2 + Constant(2.0) * mu_s * dEw_test
    comps['dSw_test'] = dSw_test

    ddEw_mixed = Constant(0.5) * (dot(Aw_test.T, dFk_trial) + dot(dFk_trial.T, Aw_test))
    comps['ddEw_mixed'] = ddEw_mixed
    ddSw_mixed = lambda_s * trace(ddEw_mixed) * I2 + Constant(2.0) * mu_s * ddEw_mixed
    comps['ddSw_mixed'] = ddSw_mixed

    # δτ[w]
    T_w = (
        dot(Aw_test, dot(Sk, Fk.T))
        + dot(Fk, dot(dSw_test, Fk.T))
        + dot(Fk, dot(Sk, Aw_test.T))
    )
    comps['T_w_1'] = dot(Aw_test, dot(Sk, Fk.T))
    comps['T_w_2'] = dot(Fk, dot(dSw_test, Fk.T))
    comps['T_w_3'] = dot(Fk, dot(Sk, Aw_test.T))
    comps['T_w'] = T_w

    # δ²τ[w,du] split into parts
    dT_terms = (
        dot(Aw_test, dot(dSk_trial, Fk.T)),
        dot(Aw_test, dot(Sk, dFk_trial.T)),
        dot(dFk_trial, dot(dSw_test, Fk.T)),
        dot(Fk, dot(ddSw_mixed, Fk.T)),
        dot(Fk, dot(dSw_test, dFk_trial.T)),
        dot(dFk_trial, dot(Sk, Aw_test.T)),
        dot(Fk, dot(dSk_trial, Aw_test.T)),
    )

    dT = dT_terms[0]
    comps['dT_1'] = dT
    for k,_term in enumerate(dT_terms[1:], start=2):
        dT = dT + _term
        comps[f'dT_{k}'] = _term
    comps['dT'] = dT
    tr_Finv_dFk = trace(dot(Finv, dFk_trial))                       # β
    tr_Finv_Aw = trace(dot(Finv, Aw_test))                           # α
    tr_Finv_dFk_FAw = trace(dot(Finv, dot(dFk_trial, dot(Finv, Aw_test))))  # γ
    comps['tr_Finv_dFk'] = tr_Finv_dFk
    comps['tr_Finv_Aw'] = tr_Finv_Aw
    comps['tr_Finv_dFk_FAw'] = tr_Finv_dFk_FAw

    sigma_k = sigma_s_nonlinear(d_ref)
    ds_u = dsigma_s(d_ref, du_trial)  # δσ[du]
    comps['sigma_k'] = sigma_k
    comps['ds_u'] = ds_u

    invJ = Constant(1.0) / Jk
    comps['invJ'] = invJ

    # (1/J) δ²τ[w,du]
    term_a_parts = tuple(invJ * part for part in dT_terms)
    term_a = term_a_parts[0]
    for _part in term_a_parts[1:]:
        term_a = term_a + _part

    # δσ[w] = (1/J) δτ[w] - α σ
    ds_w = invJ * T_w - tr_Finv_Aw * sigma_k
    comps['ds_w'] = ds_w

    # -β δσ[w]
    term_b = - tr_Finv_dFk * ds_w
    comps['term_b'] = term_b

    # (γ - αβ) σ
    term_c = (tr_Finv_dFk_FAw - tr_Finv_dFk * tr_Finv_Aw) * sigma_k
    comps['term_c'] = term_c

    # -α δσ[du]
    term_d = - tr_Finv_Aw * ds_u
    comps['term_d'] = term_d

    if return_parts:
        return (term_a, term_b, term_c, term_d), term_a_parts, comps
    return term_a, term_b, term_c, term_d, comps



def d2sigma_s(d_ref, du_trial, w_test):
    term_a, term_b, term_c, term_d, comps = _d2sigma_s_components(d_ref, du_trial, w_test)
    return (term_a + term_b + term_c + term_d),comps


def d2sigma_s_terms(d_ref, du_trial, w_test):
    components, term_a_parts, _ = _d2sigma_s_components(d_ref, du_trial, w_test, return_parts=True)
    terms = dict(zip(D2SIGMA_COMPONENT_LABELS, components))
    for label, part in zip(D2SIGMA_TERM_A_SUBLABELS, term_a_parts):
        terms[label] = part
    return terms


n = FacetNormal()


def dtraction_solid_ref_L(du, w, d_ref):
    terms, _ =d2sigma_s(d_ref, du, w)
    return dot(terms, n)


def _d2sigma_s_fx_components(d_ref, du, w, *, return_parts: bool = False):
    comps = {}
    Fk = F_fx(d_ref)
    comps['Fk'] = Fk
    Jk = ufl.det(Fk)
    comps['Jk'] = Jk
    Finv = ufl.inv(Fk)
    comps['Finv'] = Finv
    Sk = S_fx(d_ref)
    comps['Sk'] = Sk

    dFk = ufl.grad(du)
    comps['dFk'] = dFk
    Aw = ufl.grad(w)
    comps['Aw'] = Aw

    dEk = 0.5 * (ufl.dot(dFk.T, Fk) + ufl.dot(Fk.T, dFk))
    dSk = fenicsx['lambda_s'] * ufl.tr(dEk) * I2_fx + 2.0 * fenicsx['mu_s'] * dEk
    comps['dEk'] = dEk
    comps['dSk'] = dSk

    dEw = 0.5 * (ufl.dot(Aw.T, Fk) + ufl.dot(Fk.T, Aw))
    dSw = fenicsx['lambda_s'] * ufl.tr(dEw) * I2_fx + 2.0 * fenicsx['mu_s'] * dEw
    comps['dEw'] = dEw
    comps['dSw'] = dSw

    ddEw = 0.5 * (ufl.dot(Aw.T, dFk) + ufl.dot(dFk.T, Aw))
    ddSw = fenicsx['lambda_s'] * ufl.tr(ddEw) * I2_fx + 2.0 * fenicsx['mu_s'] * ddEw
    comps['ddEw'] = ddEw
    comps['ddSw'] = ddSw

    # δτ[w]
    T_w = (
        ufl.dot(Aw, ufl.dot(Sk, Fk.T))
        + ufl.dot(Fk, ufl.dot(dSw, Fk.T))
        + ufl.dot(Fk, ufl.dot(Sk, Aw.T))
    )
    comps['T_w_1'] = ufl.dot(Aw, ufl.dot(Sk, Fk.T))
    comps['T_w_2'] = ufl.dot(Fk, ufl.dot(dSw, Fk.T))
    comps['T_w_3'] = ufl.dot(Fk, ufl.dot(Sk, Aw.T))
    comps['T_w'] = T_w

    # δ²τ[w,du] split
    dT_terms = (
        ufl.dot(Aw, ufl.dot(dSk, Fk.T)),
        ufl.dot(Aw, ufl.dot(Sk, dFk.T)),
        ufl.dot(dFk, ufl.dot(dSw, Fk.T)),
        ufl.dot(Fk, ufl.dot(ddSw, Fk.T)),
        ufl.dot(Fk, ufl.dot(dSw, dFk.T)),
        ufl.dot(dFk, ufl.dot(Sk, Aw.T)),
        ufl.dot(Fk, ufl.dot(dSk, Aw.T)),
    )
    dT = dT_terms[0]
    comps['dT_1'] = dT_terms[0]
    for k,_term in enumerate(dT_terms[1:], start=2):
        dT = dT + _term
        comps[f'dT_{k}'] = _term
    comps['dT'] = dT
    tr_Finv_dFk = ufl.tr(ufl.dot(Finv, dFk))                                # β
    tr_Finv_Aw = ufl.tr(ufl.dot(Finv, Aw))                                  # α
    tr_Finv_dFk_FAw = ufl.tr(ufl.dot(Finv, ufl.dot(dFk, ufl.dot(Finv, Aw))))  # γ
    comps['tr_Finv_dFk'] = tr_Finv_dFk
    comps['tr_Finv_Aw'] = tr_Finv_Aw
    comps['tr_Finv_dFk_FAw'] = tr_Finv_dFk_FAw

    sigma_k = sigma_s_nonlinear_fx(d_ref)
    ds_u = dsigma_s_fx(d_ref, du)  # δσ[du]
    comps['sigma_k'] = sigma_k
    comps['ds_u'] = ds_u

    invJ = 1.0 / Jk
    comps['invJ'] = invJ

    # (1/J) δ²τ[w,du]
    term_a_parts = tuple(invJ * part for part in dT_terms)
    term_a = term_a_parts[0]
    for _part in term_a_parts[1:]:
        term_a = term_a + _part

    # δσ[w] = (1/J) δτ[w] - α σ
    ds_w = invJ * T_w - tr_Finv_Aw * sigma_k
    comps['ds_w'] = ds_w

    # -β δσ[w]
    term_b = - tr_Finv_dFk * ds_w
    comps['term_b'] = term_b

    # (γ - αβ) σ
    term_c = (tr_Finv_dFk_FAw - tr_Finv_dFk * tr_Finv_Aw) * sigma_k
    comps['term_c'] = term_c

    # -α δσ[du]
    term_d = - tr_Finv_Aw * ds_u
    comps['term_d'] = term_d

    if return_parts:
        return (term_a, term_b, term_c, term_d), term_a_parts, comps
    return term_a, term_b, term_c, term_d, comps



def d2sigma_s_fx(d_ref, du, w):
    term_a, term_b, term_c, term_d, comps = _d2sigma_s_fx_components(d_ref, du, w)
    return (term_a + term_b + term_c + term_d), comps


def d2sigma_s_fx_terms(d_ref, du, w):
    components, term_a_parts, _ = _d2sigma_s_fx_components(d_ref, du, w, return_parts=True)
    terms = dict(zip(D2SIGMA_COMPONENT_LABELS, components))
    for label, part in zip(D2SIGMA_TERM_A_SUBLABELS, term_a_parts):
        terms[label] = part
    return terms


def dtraction_solid_ref_L_fx(du, w, d_ref, normal_fx):
    terms, _ =d2sigma_s_fx(d_ref, du, w)
    return ufl.dot(terms, normal_fx)

def debug_interpolate(self, f):
    """
    Calculates direct nodal interpolation values for a function space.
    This function ONLY calculates and returns the values; it does not modify the Function object.
    """
    print(f"--- Calculating values for {self.name} ---")
    fs = self.function_space
    try:
        x_dofs = fs.tabulate_dof_coordinates()
    except RuntimeError:
        print(f"Space with element '{type(fs.ufl_element()).__name__}' is a subspace. Collapsing...")
        collapsed_space, _ = fs.collapse()
        x_dofs = collapsed_space.tabulate_dof_coordinates()
    
    if x_dofs.shape[0] == 0:
        return np.array([], dtype=np.float64)

    values = np.asarray(f(x_dofs.T))
    if len(values.shape) == 1:
        values = values.reshape(1, -1)
    
    # Return the flattened, interleaved array of nodal values
    return values.T.flatten()

# MONKEY-PATCH: Add our new method to the dolfinx.fem.Function class
dolfinx.fem.Function.debug_interpolate = debug_interpolate
# Helper functions for coordinates
def get_pycutfem_dof_coords(dof_handler: DofHandler, field: str) -> np.ndarray:
    if field not in dof_handler.field_names:
        raise ValueError(f"Field '{field}' not found in DofHandler")
    return dof_handler.get_dof_coords(field)

def get_all_pycutfem_dof_coords(dof_handler: DofHandler) -> np.ndarray:
    all_coords = np.zeros((dof_handler.total_dofs, 2))
    for field in ['ux', 'uy', 'p']:
        field_dofs = dof_handler.get_field_slice(field)
        field_coords = get_pycutfem_dof_coords(dof_handler, field)
        all_coords[field_dofs] = field_coords
    return all_coords

def get_all_fenicsx_dof_coords(W_fenicsx):
    num_total_dofs = W_fenicsx.dofmap.index_map.size_global
    all_coords = np.zeros((num_total_dofs, 2))
    
    W0, V_map = W_fenicsx.sub(0).collapse()
    W1, P_map_fx = W_fenicsx.sub(1).collapse()
    W00, V0_map = W0.sub(0).collapse()
    W01, V1_map = W0.sub(1).collapse()

    coords_ux = W00.tabulate_dof_coordinates()[:, :2]
    coords_uy = W01.tabulate_dof_coordinates()[:, :2]
    coords_p = W1.tabulate_dof_coordinates()[:, :2]

    dofs_ux = np.array(V_map)[np.array(V0_map)]
    dofs_uy = np.array(V_map)[np.array(V1_map)]
    dofs_p = np.array(P_map_fx)

    all_coords[dofs_ux] = coords_ux
    all_coords[dofs_uy] = coords_uy
    all_coords[dofs_p] = coords_p
    
    return all_coords

def one_to_one_map_coords(coords1, coords2):
    C = np.linalg.norm(coords2[:, np.newaxis, :] - coords1[np.newaxis, :, :], axis=2)
    rows, cols = linear_sum_assignment(C)
    return rows[np.argsort(cols)]

def create_true_dof_map(dof_handler_pc, W_fenicsx):
    print("="*70)
    print("Discovering true DoF map by matching DoF coordinates...")
    print("="*70)
    W0, V_map = W_fenicsx.sub(0).collapse()
    W1, P_map_fx = W_fenicsx.sub(1).collapse()
    W00, V0_map = W0.sub(0).collapse()
    W01, V1_map = W0.sub(1).collapse()
    fx_coords = {
        'ux': W00.tabulate_dof_coordinates()[:, :2], 'uy': W01.tabulate_dof_coordinates()[:, :2],
        'p':  W1.tabulate_dof_coordinates()[:, :2]
    }
    V_map_np = np.array(V_map)
    fx_dofs = {
        'ux': V_map_np[np.array(V0_map)], 'uy': V_map_np[np.array(V1_map)],
        'p':  np.array(P_map_fx)
    }
    pc_coords = {f: get_pycutfem_dof_coords(dof_handler_pc, f) for f in ['ux', 'uy', 'p']}
    pc_dofs = {f: dof_handler_pc.get_field_slice(f) for f in ['ux', 'uy', 'p']}
    P = np.zeros(dof_handler_pc.total_dofs, dtype=int)
    coord_map_q2 = one_to_one_map_coords(pc_coords['ux'], fx_coords['ux'])
    P[pc_dofs['ux']] = fx_dofs['ux'][coord_map_q2]
    P[pc_dofs['uy']] = fx_dofs['uy'][coord_map_q2]
    coord_map_q1 = one_to_one_map_coords(pc_coords['p'], fx_coords['p'])
    P[pc_dofs['p']] = fx_dofs['p'][coord_map_q1]
    print("True DoF map discovered successfully.")
    return P

def setup_problems():
    nx, ny = 1, 1  # Number of elements in x and y directions
    nodes_q2, elems_q2, _, corners_q2 = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=2)
    mesh_q2 = Mesh(nodes=nodes_q2, element_connectivity=elems_q2, elements_corner_nodes=corners_q2, element_type="quad", poly_order=2)
    mesh_q2.tag_boundary_edges({'all': lambda x,y: True})
    mixed_element_pc = MixedElement(mesh_q2, field_specs={'ux': 2, 'uy': 2, 'p': 1})
    dof_handler_pc = DofHandler(mixed_element_pc, method='cg')
    velocity_fs = FunctionSpace("velocity", ['ux', 'uy'], dim=1)
    pressure_fs = FunctionSpace("pressure", ['p'], dim=0)
    pc = {'du': VectorTrialFunction(velocity_fs, dof_handler=dof_handler_pc), 
          'dp': TrialFunction(pressure_fs, dof_handler=dof_handler_pc), 
          'v': VectorTestFunction(velocity_fs, dof_handler=dof_handler_pc), 
          'q': TestFunction(pressure_fs, dof_handler=dof_handler_pc), 
          'u_k': VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler_pc), 
          'p_k': Function(name="p_k", field_name='p', dof_handler=dof_handler_pc), 
          'u_n': VectorFunction(name="u_n", field_names=['ux', 'uy'], dof_handler=dof_handler_pc), 
          'rho': Constant(1.0,dim=0), 
          'dt': Constant(0.1,dim=0), 
          'theta': Constant(0.5,dim=0), 
          'mu': Constant(1.0e-2,dim=0),
          'lambda_s': Constant(0.5e6, dim=0),
          'mu_s': Constant(2.0e6, dim=0),
          'normal': FacetNormal()}

    global lambda_s, mu_s, n
    lambda_s = pc['lambda_s']
    mu_s = pc['mu_s']
    n = pc['normal']
    
    mesh_fx = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, dolfinx.mesh.CellType.quadrilateral)
    gdim = mesh_fx.geometry.dim
    P2_el = basix.ufl.element("Lagrange", 'quadrilateral', 2, shape=(gdim,))
    P1_el = basix.ufl.element("Lagrange", 'quadrilateral', 1)
    W_el = mixed_element([P2_el, P1_el])
    W = dolfinx.fem.functionspace(mesh_fx, W_el)
    fenicsx = {
        'W': W,
        'rho': dolfinx.fem.Constant(mesh_fx, 1.0),
        'dt': dolfinx.fem.Constant(mesh_fx, 0.1),
        'theta': dolfinx.fem.Constant(mesh_fx, 0.5),
        'mu': dolfinx.fem.Constant(mesh_fx, 1.0e-2),
        'lambda_s': dolfinx.fem.Constant(mesh_fx, 0.5e6),
        'mu_s': dolfinx.fem.Constant(mesh_fx, 2.0e6),
    }
    V, _ = W.sub(0).collapse()
    fenicsx['u_n'] = dolfinx.fem.Function(V, name="u_n")
    fenicsx['u_k_p_k'] = dolfinx.fem.Function(W, name="u_k_p_k")
    fenicsx['c'] = dolfinx.fem.Constant(mesh_fx, (0.5, -0.2))
    fenicsx['normal'] = ufl.FacetNormal(mesh_fx) 
    
    return pc, dof_handler_pc, fenicsx

def initialize_functions(pc, fenicsx, dof_handler_pc, P_map):
    print("Initializing and synchronizing function data...")
    np.random.seed(1234)
    def ones(x):
        return np.ones_like(x)
    def u_k_init_func(x):
        # return [np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]) , -np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]) ]
        # return [ x[0] *  x[1]**2 , -x[0] *  x[1] ]
        return [11 + x[0]  * x[1], 33 + x[1]]
        # return [x[0]**2 * x[1], x[1]**2]
        # return [ones(x[0]), ones(x[1])]
    def p_k_init_func(x):
        return np.sin(2 * np.pi * x[0]*x[1])
        # return ones(x[0]) 
    def u_n_init_func(x):
        return [0.5 * val for val in u_k_init_func(x)]
        # return [0.5 * ones(x[0]), 0.5 * ones(x[1])]

    # u_k_p_k_data_pc[3:5] = 8.0
    
    # --- Initialize pycutfem ---
    pc['u_k'].set_values_from_function(lambda x, y: u_k_init_func([x, y]))
    pc['p_k'].set_values_from_function(lambda x, y: p_k_init_func([x, y]))
    pc['u_n'].set_values_from_function(lambda x, y: u_n_init_func([x, y]))
    pc['c'] = Constant([0.5,-0.2],dim=1)
    
    # --- Initialize FEniCSx and Verify ---
    W = fenicsx['W']
    u_k_p_k_fx = fenicsx['u_k_p_k']
    u_n_fx = fenicsx['u_n']
    
    # Get the maps from the subspaces (V, Q) to the parent space (W)
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()

    # Get the component "views"
    u_k_fx = u_k_p_k_fx.sub(0)
    p_k_fx = u_k_p_k_fx.sub(1)

    # --- This is the corrected assignment logic ---
    # 1. Calculate the values for the velocity subspace
    u_k_values = u_k_fx.debug_interpolate(u_k_init_func)
    # 2. Place them into the correct slots of the PARENT vector using the map
    u_k_p_k_fx.x.array[V_to_W] = u_k_values

    # 3. Calculate the values for the pressure subspace
    p_k_values = p_k_fx.debug_interpolate(p_k_init_func)
    # 4. Place them into the correct slots of the PARENT vector using the map
    u_k_p_k_fx.x.array[Q_to_W] = p_k_values
    
    # 5. Synchronize the PARENT vector once after all modifications
    u_k_p_k_fx.x.scatter_forward()
    
    # 6. Initialize the standalone u_n function (this needs its own assignment and scatter)
    u_n_values = u_n_fx.debug_interpolate(u_n_init_func)
    u_n_fx.x.array[:] = u_n_values
    u_n_fx.x.scatter_forward()

    # --- Optional Assertion ---
    pycutfem_uk_values = pc['u_k'].nodal_values
    np.testing.assert_allclose(np.sort(pycutfem_uk_values), np.sort(u_k_values), rtol=1e-8, atol=1e-15)
    print("\n✅ ASSERTION PASSED: pycutfem and FEniCSx calculated the same set of nodal values for u_k.")

def compare_term(term_name, J_pc, R_pc, J_fx, R_fx, P_map, dof_handler_pc, W_fenicsx):
    print("\n" + f"--- Comparing Term: {term_name} ---")

    output_dir = "garbage"
    os.makedirs(output_dir, exist_ok=True)
    safe_term_name = term_name.replace(' ', '_').lower()
    is_successful = True

    if R_pc is not None and R_fx is not None:
        filename = os.path.join(output_dir, f"{safe_term_name}_residual.xlsx")
        R_fx_reordered = R_fx[P_map]
        
        R_pc_flat = R_pc.flatten()
        R_fx_reordered_flat = R_fx_reordered.flatten()

        pc_coords = get_all_pycutfem_dof_coords(dof_handler_pc)
        fx_coords_all = get_all_fenicsx_dof_coords(W_fenicsx)
        fx_coords_reordered = fx_coords_all[P_map]
        
        comparison_data = {
            'pc_dof_index': np.arange(dof_handler_pc.total_dofs),
            'pc_coord_x': pc_coords[:, 0], 'pc_coord_y': pc_coords[:, 1], 'pc_residual': R_pc_flat,
            'fx_coord_x': fx_coords_reordered[:, 0], 'fx_coord_y': fx_coords_reordered[:, 1],
            'fx_reordered_residual': R_fx_reordered_flat, 'abs_difference': np.abs(R_pc_flat - R_fx_reordered_flat)
        }
        pd.DataFrame(comparison_data).to_excel(filename, sheet_name='residual_comparison', index=False)
        print(f"✅ Residual comparison data saved to '{filename}'")

        try:
            np.testing.assert_allclose(R_pc_flat, R_fx_reordered_flat, rtol=1e-8, atol=1e-8)
            print(f"✅ Residual vector for '{term_name}' is numerically equivalent.")
        except AssertionError as e:
            print(f"❌ Residual vector for '{term_name}' is NOT equivalent!")
            print(e)
            is_successful = False

    if J_pc is not None and J_fx is not None:
        filename = os.path.join(output_dir, f"{safe_term_name}_jacobian.xlsx")
        J_pc_dense = J_pc.toarray()
        J_fx_reordered = J_fx[P_map, :][:, P_map]
        with pd.ExcelWriter(filename) as writer:
            pd.DataFrame(J_pc_dense).to_excel(writer, sheet_name='pycutfem', index=False, header=False)
            pd.DataFrame(J_fx_reordered).to_excel(writer, sheet_name='fenics', index=False, header=False)
            pd.DataFrame(np.abs(J_pc_dense - J_fx_reordered)<1e-12).to_excel(writer, sheet_name='difference', index=False, header=False)
        print(f"✅ Jacobian matrices saved to '{filename}'")
        try:
            np.testing.assert_allclose(J_pc_dense, J_fx_reordered, rtol=1e-8, atol=1e-8)
            print(f"✅ Jacobian matrix for '{term_name}' is numerically equivalent.")
        except AssertionError as e:
            print(f"❌ Jacobian matrix for '{term_name}' is NOT equivalent!")
            print(e)
            is_successful = False
    return is_successful

def print_test_summary(success_count, failed_tests):
    """Prints a summary of the test results."""
    total_tests = success_count + len(failed_tests)
    failure_count = len(failed_tests)

    print("\n" + "="*70)
    print(" " * 25 + "TEST SUITE SUMMARY")
    print("="*70)
    print(f"Total tests run: {total_tests}")
    print(f"✅ Successful tests: {success_count}")
    print(f"❌ Failed tests:     {failure_count}")
    
    if failure_count > 0:
        print("\n--- List of Failed Tests ---")
        for test_name in failed_tests:
            print(f"  - {test_name}")
    print("="*70)

# ==============================================================================
#                      MAIN TEST HARNESS
# ==============================================================================
if __name__ == '__main__':
    pc, dof_handler_pc, fenicsx = setup_problems()
    P_map = create_true_dof_map(dof_handler_pc, fenicsx['W'])
    initialize_functions(pc, fenicsx, dof_handler_pc, P_map)

    W_fx = fenicsx['W']
    u_k_fx, p_k_fx = ufl.split(fenicsx['u_k_p_k']) 
    u_n_fx = fenicsx['u_n']
    
    V_subspace = W_fx.sub(0)
    Q_subspace = W_fx.sub(1)
    du, v = ufl.TrialFunction(V_subspace), ufl.TestFunction(V_subspace)
    dp, q = ufl.TrialFunction(Q_subspace), ufl.TestFunction(Q_subspace)

    advection_1_pc = ( dot(dot(grad(pc['du']), pc['u_k']), pc['v'])) * dx()
    c_pc = pc['c']
    c_fx = fenicsx['c']
    # Facet normals
    n_pc = pc['normal']                # PyCutFEM-side facet normal
    n_fx = fenicsx['normal']           # FEniCSx-side facet normal

    jacobian_pc = (
        # Time derivative term
        pc['rho'] * dot(pc['du'], pc['v']) / pc['dt'] +
        
        # Convection terms (linearization of u ⋅ ∇u)
        pc['theta'] * pc['rho'] * dot(dot(grad(pc['u_k']), pc['du']), pc['v']) +
        pc['theta'] * pc['rho'] * dot(dot(grad(pc['du']), pc['u_k']), pc['v']) +

        # Diffusion term
        pc['theta'] * pc['mu'] * inner(grad(pc['du']), grad(pc['v'])) -
        
        # Pressure term (linearization of -p∇⋅v)
        pc['dp'] * div(pc['v']) +
        
        # Continuity term
        pc['q'] * div(pc['du'])
    ) * dx()
    residual_pc = (
        # Time derivative
        pc['rho'] * dot(pc['u_k'] - pc['u_n'], pc['v']) / pc['dt'] +

        # Convection terms (implicit and explicit parts)
        pc['theta'] * pc['rho'] * dot(dot(grad(pc['u_k']), pc['u_k']), pc['v']) +
        (1.0 - pc['theta']) * pc['rho'] * dot(dot(grad(pc['u_n']), pc['u_n']), pc['v']) +

        # Diffusion terms (implicit and explicit parts)
        pc['theta'] * pc['mu'] * inner(grad(pc['u_k']), grad(pc['v'])) +
        (1.0 - pc['theta']) * pc['mu'] * inner(grad(pc['u_n']), grad(pc['v'])) -
        
        # Pressure term
        pc['p_k'] * div(pc['v']) +
        
        # Continuity term
        pc['q'] * div(pc['u_k'])
    ) * dx()


    def create_fenics_ns_jacobian(deg):
        """Creates the UFL form for the Navier-Stokes Jacobian using Trial/Test
        functions from the mixed space W."""
        # Define Trial and Test Functions on the *mixed space* W_fx
        dup_fx, vq_fx = ufl.TrialFunction(W_fx), ufl.TestFunction(W_fx)
        # Split them to get the velocity and pressure components
        du_fx, dp_fx = ufl.split(dup_fx)
        v_fx, q_fx = ufl.split(vq_fx)
        
        # Now, build the form using these correctly-defined components
        return (
            fenicsx['rho'] * ufl.dot(du_fx, v_fx) / fenicsx['dt'] +
            fenicsx['theta'] * fenicsx['rho'] * ufl.dot(ufl.dot(ufl.grad(u_k_fx), du_fx), v_fx) +
            fenicsx['theta'] * fenicsx['rho'] * ufl.dot(ufl.dot(ufl.grad(du_fx), u_k_fx), v_fx) +
            fenicsx['theta'] * fenicsx['mu'] * ufl.inner(ufl.grad(du_fx), ufl.grad(v_fx)) -
            dp_fx * ufl.div(v_fx) +
            q_fx * ufl.div(du_fx)
        ) * ufl.dx(metadata={'quadrature_degree': deg})

    def create_fenics_ns_residual(deg):
        """Creates the UFL form for the Navier-Stokes residual using a
        TestFunction from the mixed space W."""
        # Define a single TestFunction on the parent mixed space
        vq_fx = ufl.TestFunction(W_fx)
        
        # Split it to get the velocity and pressure components
        v_fx, q_fx = ufl.split(vq_fx)
        
        # Now, build the residual form using these correctly-defined test functions
        # and the existing Function objects (u_k_fx, p_k_fx, u_n_fx)
        return (
            # Time derivative
            fenicsx['rho'] * ufl.dot(u_k_fx - u_n_fx, v_fx) / fenicsx['dt'] +

            # Convection terms (implicit and explicit parts)
            fenicsx['theta'] * fenicsx['rho'] * ufl.dot(ufl.dot(ufl.grad(u_k_fx), u_k_fx), v_fx) +
            (1.0 - fenicsx['theta']) * fenicsx['rho'] * ufl.dot(ufl.dot(ufl.grad(u_n_fx), u_n_fx), v_fx) +

            # Diffusion terms (implicit and explicit parts)
            fenicsx['theta'] * fenicsx['mu'] * ufl.inner(ufl.grad(u_k_fx), ufl.grad(v_fx)) +
            (1.0 - fenicsx['theta']) * fenicsx['mu'] * ufl.inner(ufl.grad(u_n_fx), ufl.grad(v_fx)) -
            
            # Pressure term
            p_k_fx * ufl.div(v_fx) +
            
            # Continuity term
            q_fx * ufl.div(u_k_fx)
        ) * ufl.dx(metadata={'quadrature_degree': deg})

    vq_fx = ufl.TestFunction(W_fx)
        
    # Split it to get the velocity and pressure components
    v_fx, q_fx = ufl.split(vq_fx)

    def epsilon_f(u, grad):
        "Symmetric gradient."
        return 0.5 * (grad(u) + grad(u).T)

    def epsilon_s_linear_L(disp, disp_k, grad, dot):
        """Agnostic version of the linearized solid strain tensor (LHS)."""
        # return 0.5 * (grad(disp) + grad(disp).T +  dot(grad(disp).T, grad(disp_k))  + dot(grad(disp_k).T, grad(disp)))
        return 0.5 * ( dot(grad(disp_k).T, grad(disp)))

    def epsilon_s_linear_R(disp_k, grad, dot):
        """Agnostic version of the linearized solid strain tensor (RHS)."""
        return 0.5 * (grad(disp_k) + grad(disp_k).T + dot(grad(disp_k).T, grad(disp_k)))

    def trace_component(disp, disp_k, grad, dot, trace):
        return 0.5 * ( trace(grad(disp)) + trace(grad(disp).T) + trace(dot(grad(disp).T, grad(disp_k))) + trace(dot(grad(disp_k).T, grad(disp))))
    def sigma_s_linear_weak_L(ddisp, disp_k, grad_v_test, inner, trace, grad, dot):
        """Agnostic version of the linearized solid stress tensor (LHS)."""
        strain = epsilon_s_linear_L(ddisp, disp_k, grad, dot)
        return   trace(strain) * trace(grad_v_test) #+ 2.0 * inner(strain, grad_v_test)
        # return 2.0 * inner(strain, grad_v_test) + trace_component(ddisp, disp_k, grad, dot, trace) * trace(grad_v_test)

    def sigma_s_linear_weak_R(disp_k, grad_v_test, inner, trace, grad, dot):
        """Agnostic version of the linearized solid stress tensor (RHS)."""
        strain = epsilon_s_linear_R(disp_k, grad, dot)
        return 2.0 * inner(strain, grad_v_test) + trace(strain) * trace(grad_v_test)

    def Hdotn_fx(u, n):   # (H · n)
        return ufl.dot(ufl.grad(ufl.grad(u)), n)

    def ndotH_fx(u, n):   # (n · H)
        return ufl.dot(n, ufl.grad(ufl.grad(u)))

    def nHn_fx(u, n):     # (nᵀ H n)
        return ufl.dot(n, ufl.dot(ufl.grad(ufl.grad(u)), n))

    I2_pc = Identity(2)
    I2_fx = ufl.Identity(2)

    def F_pc(u):
        return grad(u) + I2_pc

    def F_fx(u):
        return ufl.grad(u) + I2_fx

    def delta_E_pc(w, u_ref):
        F_ref = F_pc(u_ref)
        grad_w = grad(w)
        return 0.5 * (dot(grad_w.T, F_ref) + dot(F_ref.T, grad_w))

    def delta_E_fx(w, u_ref):
        F_ref = F_fx(u_ref)
        grad_w = ufl.grad(w)
        return 0.5 * (ufl.dot(grad_w.T, F_ref) + ufl.dot(F_ref.T, grad_w))

    def E_pc(u):
        F_ref = F_pc(u)
        return 0.5 * (dot(F_ref.T, F_ref) - I2_pc)

    def E_fx(u):
        F_ref = F_fx(u)
        return 0.5 * (ufl.dot(F_ref.T, F_ref) - I2_fx)

    def S_pc(u):
        E_ref = E_pc(u)
        return pc['lambda_s'] * trace(E_ref) * I2_pc + Constant(2.0, dim=0) * pc['mu_s'] * E_ref

    def S_fx(u):
        E_ref = E_fx(u)
        return fenicsx['lambda_s'] * ufl.tr(E_ref) * I2_fx + 2.0 * fenicsx['mu_s'] * E_ref


    def sigma_s_nonlinear_fx(u):
        F = F_fx(u)
        S = S_fx(u)
        J = ufl.det(F)
        return (1.0 / J) * ufl.dot(ufl.dot(F, S), F.T)


    def dsigma_s_fx(d_ref, delta_d):
        Fk = F_fx(d_ref)
        Sk = S_fx(d_ref)
        dF = ufl.grad(delta_d)
        dE = 0.5 * (ufl.dot(dF.T, Fk) + ufl.dot(Fk.T, dF))
        dS = fenicsx['lambda_s'] * ufl.tr(dE) * I2_fx + 2.0 * fenicsx['mu_s'] * dE
        Jk = ufl.det(Fk)
        Finv = ufl.inv(Fk)
        dJ = Jk * ufl.tr(ufl.dot(Finv, dF))
        term = (
            ufl.dot(dF, ufl.dot(Sk, Fk.T))
            + ufl.dot(Fk, ufl.dot(dS, Fk.T))
            + ufl.dot(Fk, ufl.dot(Sk, dF.T))
        )
        return (1.0 / Jk) * term - (dJ / Jk) * sigma_s_nonlinear_fx(d_ref)


    

    F_k_pc = F_pc(pc['u_k'])
    F_n_pc = F_pc(pc['u_n'])
    F_k_fx = F_fx(u_k_fx)
    F_n_fx = F_fx(u_n_fx)

    delta_E_trial_pc = delta_E_pc(pc['du'], pc['u_k'])
    delta_E_test_k_pc = delta_E_pc(pc['v'], pc['u_k'])
    delta_E_test_n_pc = delta_E_pc(pc['v'], pc['u_n'])
    delta_E_trial_fx = delta_E_fx(du, u_k_fx)
    delta_E_test_k_fx = delta_E_fx(v_fx, u_k_fx)
    delta_E_test_n_fx = delta_E_fx(v_fx, u_n_fx)

    C_delta_E_trial_pc = pc['lambda_s'] * trace(delta_E_trial_pc) * I2_pc + Constant(2.0, dim=0) * pc['mu_s'] * delta_E_trial_pc
    C_delta_E_trial_fx = fenicsx['lambda_s'] * ufl.tr(delta_E_trial_fx) * I2_fx + 2.0 * fenicsx['mu_s'] * delta_E_trial_fx

    S_k_pc = S_pc(pc['u_k'])
    S_n_pc = S_pc(pc['u_n'])
    S_k_fx = S_fx(u_k_fx)
    S_n_fx = S_fx(u_n_fx)

    delta_delta_E_test_pc = 0.5 * (dot(grad(pc['du']).T, grad(pc['v'])) + dot(grad(pc['v']).T, grad(pc['du'])))
    delta_delta_E_test_fx = 0.5 * (ufl.dot(ufl.grad(du).T, ufl.grad(v_fx)) + ufl.dot(ufl.grad(v_fx).T, ufl.grad(du)))

    # delta_delta_E_test_pc_2 = 0.5 * (dot(grad(pc['du']), grad(pc['v'])) )
    # delta_delta_E_test_fx_2 = 0.5 * (ufl.dot(ufl.grad(du), ufl.grad(v_fx)) )
    delta_delta_E_test_pc_2 = 0.5 * (dot(grad(pc['v']).T, grad(pc['du'])) )
    delta_delta_E_test_fx_2 = 0.5 * (ufl.dot(ufl.grad(v_fx).T, ufl.grad(du)) )

    d2sigma_pc_expr, pc_geo = d2sigma_s(pc['u_k'], pc['du'], pc['v'])
    d2sigma_fx_expr, fe_geo = d2sigma_s_fx(u_k_fx, du, v_fx)
    d2sigma_pc_terms = d2sigma_s_terms(pc['u_k'], pc['du'], pc['v'])
    d2sigma_fx_terms = d2sigma_s_fx_terms(u_k_fx, du, v_fx)

    # F_trial_pc = F_pc(pc['du'])
    # F_test_pc = F_pc(pc['v'])
    Fk_pc = F_pc(pc['u_k'])
    Fk_fx = F_fx(u_k_fx)
    F_trial_pc = grad(pc['du'])
    F_test_pc = grad(pc['v'])
    F_trial_fx = ufl.grad(du)
    F_test_fx = ufl.grad(v_fx)
    Aw_test_pc = F_test_pc
    Aw_test_fx = F_test_fx
    F_inv_trial_pc = inv(Fk_pc)
    F_inv_trial_fx = ufl.inv(Fk_fx)
    dEk_trial_pc =  (dot(F_trial_pc.T, Fk_pc) + dot(Fk_pc.T, F_trial_pc))
    dEk_trial_fx =  (ufl.dot(F_trial_fx.T, Fk_fx) + ufl.dot(Fk_fx.T, F_trial_fx))
    dSk_trial_pc =  trace(dEk_trial_pc) * I2_pc +  dEk_trial_pc
    dSk_trial_fx =  ufl.tr(dEk_trial_fx) * I2_fx +  dEk_trial_fx
    dEk_test_pc =  (dot(F_test_pc.T, Fk_pc) + dot(Fk_pc.T, F_test_pc))
    dSk_test_pc = trace(dEk_test_pc) * I2_pc +  dEk_test_pc
    dEk_test_fx =  (ufl.dot(F_test_fx.T, Fk_fx) + ufl.dot(Fk_fx.T, F_test_fx))
    dSk_test_fx =  ufl.tr(dEk_test_fx) * I2_fx +  dEk_test_fx

    ddEw_mixed_pc =  (dot(F_test_pc.T, F_trial_pc) + dot(F_trial_pc.T, F_test_pc))
    ddSw_mixed_pc =  trace(ddEw_mixed_pc) * I2_pc +  ddEw_mixed_pc
    ddEw_mixed_fx =  (ufl.dot(F_test_fx.T, F_trial_fx) + ufl.dot(F_trial_fx.T, F_test_fx))
    ddSw_mixed_fx =  ufl.tr(ddEw_mixed_fx) * I2_fx +  ddEw_mixed_fx

    def _solid_cross_component_entry(label: str,deg_fe:int = 6, deg_pc:int = 6):
        pc_expr = d2sigma_pc_terms[label]
        fx_expr = d2sigma_fx_terms[label]
        return {
            'pc': pc['theta'] * inner(pc_expr, I2_pc) * dx(metadata={"q":deg_pc}),
            'f_lambda': lambda deg, expr=fx_expr: fenicsx['theta'] * ufl.inner(expr, I2_fx) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': deg_fe,
        }


    
    terms = {
        "Mixed Basic [Fk]": {
            'pc': inner(pc_geo["Fk"], grad(pc['v'])) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["Fk"], ufl.grad(v_fx)) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 6,
        },

        "Mixed Basic [Fk@Finv]": {
            'pc': inner(pc_geo["Fk"], pc_geo["Finv"]) * pc['dp'] * pc['q'] * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["Fk"], fe_geo["Finv"]) * dp * q_fx \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 8,
        },
        "Mixed Basic residual [Fk@Finv]": {
            'pc': inner(pc_geo["Fk"], pc_geo["Finv"])  * pc['q'] * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["Fk"], fe_geo["Finv"])  * q_fx \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 8,
        },
        "Mixed Basic [Sk]": {
            'pc': inner(pc_geo["Sk"], grad(pc['v'])) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["Sk"], ufl.grad(v_fx)) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 6,
        },
        "Mixed Basic [dFk@Aw]": {
            'pc': inner(pc_geo["dFk_trial"], pc_geo["Aw_test"]) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["dFk"], fe_geo["Aw"]) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Mixed Basic [dEk@dEw]": {
            'pc': inner(pc_geo["dEk_trial"], pc_geo["dEw_test"]) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["dEk"], fe_geo["dEw"]) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Mixed Basic [dSk@dSw]": {
            'pc': inner(pc_geo["dSk_trial"], pc_geo["dSw_test"]) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["dSk"], fe_geo["dSw"]) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Mixed Basic [ddEw@I2]": {
            'pc': inner(pc_geo["ddEw_mixed"], I2_pc) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["ddEw"], I2_fx) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Mixed Basic [ddSw@I2]": {
            'pc': inner(pc_geo["ddSw_mixed"], I2_pc) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["ddSw"], I2_fx) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Mixed Basic residual [I2@T_w_1]": {
            'pc': inner(I2_pc, pc_geo["T_w_1"]) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(I2_fx, fe_geo["T_w_1"]) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 6,
        },
        "Mixed Basic residual [I2@T_w_2]": {
            'pc': inner(I2_pc, pc_geo["T_w_2"]) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(I2_fx, fe_geo["T_w_2"]) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 6,
        },
        "Mixed Basic residual [I2@T_w_3]": {
            'pc': inner(I2_pc, pc_geo["T_w_3"]) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(I2_fx, fe_geo["T_w_3"]) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 6,
        },
        "Mixed Basic residual [I2@T_w]": {
            'pc': inner(I2_pc, pc_geo["T_w"]) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(I2_fx, fe_geo["T_w"]) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 6,
        },
        "Mixed Basic [dT_1@I2]": {
            'pc': inner(pc_geo["dT_1"], I2_pc) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["dT_1"], I2_fx) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Mixed Basic [dT_2@I2]": {
            'pc': inner(pc_geo["dT_2"], I2_pc) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["dT_2"], I2_fx) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Mixed Basic [dT_3@I2]": {
            'pc': inner(pc_geo["dT_3"], I2_pc) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["dT_3"], I2_fx) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Mixed Basic [dT_4@I2]": {
            'pc': inner(pc_geo["dT_4"], I2_pc) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["dT_4"], I2_fx) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Mixed Basic [dT_5@I2]": {
            'pc': inner(pc_geo["dT_5"], I2_pc) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["dT_5"], I2_fx) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Mixed Basic [dT_6@I2]": {
            'pc': inner(pc_geo["dT_6"], I2_pc) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["dT_6"], I2_fx) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Mixed Basic [dT_7@I2]": {
            'pc': inner(pc_geo["dT_7"], I2_pc) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["dT_7"], I2_fx) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Mixed Basic [dT@I2]": {
            'pc': inner(pc_geo["dT"], I2_pc) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["dT"], I2_fx) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Mixed Basic [tr_Finv_dFk@tr_Finv_Aw]": {
            'pc': inner(pc_geo["tr_Finv_dFk"], pc_geo['tr_Finv_Aw']) * dx(metadata={"q": 8}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["tr_Finv_dFk"], fe_geo['tr_Finv_Aw']) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 14,
        },
        "Mixed Basic [tr_Finv_dFk_FAw]": {
            'pc': pc_geo["tr_Finv_dFk_FAw"] * dx(metadata={"q": 8}),
            'f_lambda': lambda deg: fe_geo["tr_Finv_dFk_FAw"] \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 14,
        },
        # ----------------------------------------------------------

        "Mixed Basic [dot(F_test^T, F_trial)]": {
            'pc': inner(dot(F_test_pc.T, F_trial_pc), I2_pc) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(
                ufl.dot(F_test_fx.T, F_trial_fx), I2_fx
            ) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Mixed Right Contraction [dot(dot(F_test^T, F_trial), Fk^T)]": {
            'pc': inner(dot(dot(F_test_pc.T, F_trial_pc), Fk_pc.T), I2_pc) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(
                ufl.dot(ufl.dot(F_test_fx.T, F_trial_fx), Fk_fx.T), I2_fx
            ) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Mixed Left Contraction [dot(Fk, dot(F_test^T, F_trial))]": {
            'pc': inner(dot(Fk_pc, dot(F_test_pc.T, F_trial_pc)), I2_pc) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(
                ufl.dot(Fk_fx, ufl.dot(F_test_fx.T, F_trial_fx)), I2_fx
            ) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Mixed VecOp dot grad [dot(du, grad(v))]": {
            'pc': dot(dot(pc['du'], grad(pc['v'])), pc['c']) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.dot(
                ufl.dot(du, ufl.grad(v_fx)), c_fx
            ) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Mixed VecOp grad dot [dot(grad(v), du)]": {
            'pc': dot(dot(grad(pc['v']), pc['du']), pc['c']) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.dot(
                ufl.dot(ufl.grad(v_fx), du), c_fx
            ) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Diag dT term 1 [Aw · (dSk · F^T)]": {
            'pc': inner(dot(Aw_test_pc, dot(dSk_trial_pc, Fk_pc.T)), I2_pc) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(
                ufl.dot(Aw_test_fx, ufl.dot(dSk_trial_fx, Fk_fx.T)),
                I2_fx
            ) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Solid Cross term 1": {
            'pc':  inner(dEk_trial_pc, dEk_test_pc) * dx(metadata={"q":6}),
            'f_lambda': lambda deg:  ufl.inner(dEk_trial_fx, dEk_test_fx) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Solid Cross term 2": {
            'pc':  inner(dSk_trial_pc, dSk_test_pc) * dx(metadata={"q":6}),
            'f_lambda': lambda deg:  ufl.inner(dSk_trial_fx, dSk_test_fx) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Solid Cross term 3": {
            'pc':  inner(ddEw_mixed_pc, I2_pc) * dx(metadata={"q":6}),
            'f_lambda': lambda deg:  ufl.inner(ddEw_mixed_fx, I2_fx) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Solid Material Tangent": {
            'pc': pc['theta'] * inner(C_delta_E_trial_pc, delta_E_test_k_pc) * dx(metadata={"q":6}),
            'f_lambda': lambda deg: fenicsx['theta'] * ufl.inner(C_delta_E_trial_fx, delta_E_test_k_fx) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Solid Cross Tangent": {
            'pc': pc['theta'] * inner(d2sigma_pc_expr, I2_pc) * dx(metadata={"q":8}),
            'f_lambda': lambda deg: fenicsx['theta'] * ufl.inner(d2sigma_fx_expr, I2_fx) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 14,
        },
        "Solid Cross Tangent [A: invJ*dT]": _solid_cross_component_entry("A: invJ*dT", deg_fe=14, deg_pc=8),
        "Solid Cross Tangent [B: -(tr(F^{-1} dF) over J)*T_w]": _solid_cross_component_entry("B: -(tr(F^{-1} dF) over J)*T_w", deg_fe=14, deg_pc=8),
        "Solid Cross Tangent [C: tr(F^{-1} dF F^{-1} Aw)*sigma]": _solid_cross_component_entry("C: tr(F^{-1} dF F^{-1} Aw)*sigma", deg_fe=14, deg_pc=8),
        "Solid Cross Tangent [D: -tr(F^{-1} Aw)*ds_u]": _solid_cross_component_entry("D: -tr(F^{-1} Aw)*ds_u", deg_fe=14, deg_pc=8),

    "Solid Geometric Tangent": {
        'pc': pc['theta'] * inner(S_k_pc, delta_delta_E_test_pc) * dx(metadata={"q":6}),
        'f_lambda': lambda deg: fenicsx['theta'] * ufl.inner(S_k_fx, delta_delta_E_test_fx) * ufl.dx(metadata={'quadrature_degree': deg}),
        'mat': True,
        'deg': 6,
    },
        "Solid Residual": {
            'pc': (
                pc['theta'] * inner(S_k_pc, delta_E_test_k_pc)
                + (Constant(1.0, dim=0) - pc['theta']) * inner(S_n_pc, delta_E_test_n_pc)
            ) * dx(metadata={"q":6}),
            'f_lambda': lambda deg: (
                fenicsx['theta'] * ufl.inner(S_k_fx, delta_E_test_k_fx)
                + (1.0 - fenicsx['theta']) * ufl.inner(S_n_fx, delta_E_test_n_fx)
            ) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 6,
        },
        "LHS Mass":          {'pc': pc['rho'] * dot(pc['du'], pc['v']) / pc['dt'] * dx(),                                    'f_lambda': lambda deg: fenicsx['rho'] * ufl.dot(du, v) / fenicsx['dt'] * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "LHS Diffusion":     {'pc': pc['theta'] * pc['mu'] * inner(grad(pc['du']), grad(pc['v'])) * dx(metadata={"q":4}),                     'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['mu'] * ufl.inner(ufl.grad(du), ufl.grad(v)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "LHS Advection 1":   {'pc':  ( dot(dot(grad(pc['du']), pc['u_k']), pc['v'])) * dx(metadata={"q":6}),           'f_lambda': lambda deg:  ufl.dot(ufl.dot(ufl.grad(du),u_k_fx), v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 5},
        "LHS Advection 2":   {'pc':  dot(dot(grad(pc['u_k']), pc['du']), pc['v']) * dx(metadata={"q":5}),            'f_lambda': lambda deg: ufl.dot( ufl.dot(ufl.grad(u_k_fx),du), v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 5},
        "LHS Advection 3":   {'pc':  ( dot(dot(pc['u_k'], grad(pc['du']) ), pc['v'])) * dx(metadata={"q":5}),           'f_lambda': lambda deg:  ufl.dot(ufl.dot(u_k_fx,ufl.grad(du)), v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 5},
        "LHS Advection 4":   {'pc': pc['theta'] * pc['rho'] * dot(dot(pc['du'],grad(pc['u_k']) ), pc['v']) * dx(metadata={"q":5}),            'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['rho'] * ufl.dot(ufl.dot(du,ufl.grad(u_k_fx)), v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 5},
        "LHS Pressure":      {'pc': -pc['dp'] * div(pc['v']) * dx(),                                                         'f_lambda': lambda deg: -dp * ufl.div(v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 3},
        "LHS Continuity":    {'pc': pc['q'] * div(pc['du']) * dx(),                                                          'f_lambda': lambda deg: q * ufl.div(du) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 3},
        "RHS Time Derivative": {'pc': (pc['rho'] * dot(pc['u_k'] - pc['u_n'], pc['v']) / pc['dt']) * dx(),                       'f_lambda': lambda deg: fenicsx['rho'] * ufl.dot(u_k_fx - u_n_fx, v) / fenicsx['dt'] * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 5},
        "RHS Advection":     {'pc': pc['theta'] * pc['rho'] * dot(dot(grad(pc['u_k']), pc['u_k']), pc['v']) * dx(),          'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['rho'] * ufl.dot(ufl.dot(ufl.grad(u_k_fx),u_k_fx), v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 5},
        "RHS Advection 2":     {'pc': pc['theta'] * pc['rho'] * dot(dot(pc['u_k'],grad(pc['u_k']) ), pc['v']) * dx(),          'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['rho'] * ufl.dot(ufl.dot(u_k_fx,ufl.grad(u_k_fx)), v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 5},
        "LHS Scalar Advection": {'pc': dot(grad(pc['dp']), pc['u_k']) * pc['q'] * dx(), 'f_lambda': lambda deg: ufl.dot(ufl.grad(dp), u_k_fx) * q * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 3},
        "LHS Scalar Advection 2": {'pc': dot(pc['u_k'], grad(pc['dp'])) * pc['q'] * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.dot(u_k_fx, ufl.grad(dp)) * q * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 3},
        "LHS Vector Advection Constant": {'pc': dot(dot(grad(pc['du']), c_pc), pc['v']) * dx(), 'f_lambda': lambda deg: ufl.dot(ufl.dot(ufl.grad(du), c_fx), v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 5},
        "Navier Stokes LHS": {'pc': jacobian_pc, 'f_lambda':  create_fenics_ns_jacobian, 'mat': True, 'deg': 5},
        "RHS diffusion": {'pc': inner(grad(pc['u_k']), grad(pc['v'])) * dx(metadata={"q":4}),'f_lambda': lambda deg:  ufl.inner(ufl.grad(u_k_fx), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg':4},
        "RHS scalar diffusion": {'pc':  inner(grad(pc['p_k']), grad(pc['q'])) * dx(metadata={"q":4}),'f_lambda': lambda deg: ufl.inner(ufl.grad(p_k_fx), ufl.grad(q_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg':4},
        "RHS diffusion 3": {'pc': (1.0 - pc['theta']) * pc['mu'] * inner(grad(pc['u_n']), grad(pc['v'])) * dx(metadata={'quadrature_degree': 4}),'f_lambda': lambda deg: (1.0 - fenicsx['theta']) * fenicsx['mu'] * ufl.inner(ufl.grad(u_n_fx), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat':False, 'deg':4},
        "Navier Stokes RHS": {'pc': residual_pc, 'f_lambda':  create_fenics_ns_residual, 'mat': False, 'deg': 6},
        "RHS pressure term": {'pc': pc['p_k'] * div(pc['v']) * dx, 'f_lambda': lambda deg: p_k_fx * ufl.div(v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 5},
        "RHS Continuity":    {'pc': pc['q'] * div(pc['u_k']) * dx, 'f_lambda': lambda deg: q_fx * ufl.div(u_k_fx) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 6},# "distributed rhs": {'pc': -(pc['q'] * div(pc['u_k']) * dx - pc['p_k'] * div(pc['v']) * dx), 'f_lambda': lambda deg: -(q_fx * ufl.div(u_k_fx) * ufl.dx(metadata={'quadrature_degree': deg}) - p_k_fx * ufl.div(v_fx) * ufl.dx(metadata={'quadrature_degree': deg})), 'mat': False, 'deg': 6}
        "Dot of gradients LHS ohne transpose": {'pc':  inner(dot(grad(pc['du']), grad(pc['u_k'])),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(du), ufl.grad(u_k_fx)), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Dot of gradients LHS  transpose": {'pc':  inner(dot(grad(pc['du']).T, grad(pc['u_k'])),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(du).T, ufl.grad(u_k_fx)), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Dot of gradients LHS  transpose 2": {'pc':  inner(dot(grad(pc['du']), grad(pc['u_k']).T),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(du), ufl.grad(u_k_fx).T), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Dot of gradients LHS  transpose 3": {'pc':  inner(dot(grad(pc['du']).T, grad(pc['u_k']).T),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(du).T, ufl.grad(u_k_fx).T), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Dot of gradients LHS swap ohne transpose": {'pc':  inner(dot(grad(pc['u_k']),grad(pc['du']) ),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx),ufl.grad(du) ), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Dot of gradients LHS swap  transpose 1": {'pc':  inner(dot(grad(pc['u_k']),grad(pc['du']).T),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx), ufl.grad(du).T), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Dot of gradients LHS swap transpose 2": {'pc':  inner(dot( grad(pc['u_k']).T, grad(pc['du'])),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot( ufl.grad(u_k_fx).T , ufl.grad(du)), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Dot of gradients LHS swap transpose 3": {'pc':  inner(dot( grad(pc['u_k']).T, grad(pc['du']).T),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot( ufl.grad(u_k_fx).T , ufl.grad(du).T), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Dot of gradients RHS ohne transpose": {'pc':  inner(dot(grad(pc['u_k']), grad(pc['u_k'])),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx), ufl.grad(u_k_fx)), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        "Dot of gradients RHS transpose": {'pc':  inner(dot(grad(pc['u_k']).T, grad(pc['u_k'])),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx).T,  ufl.grad(u_k_fx)), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        "Dot of gradients RHS transpose 2": {'pc':  inner(dot(grad(pc['u_k']), grad(pc['u_k']).T),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx),  ufl.grad(u_k_fx).T), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        "Dot of gradients RHS transpose 3": {'pc':  inner(dot(grad(pc['u_k']).T, grad(pc['u_k']).T),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx).T,  ufl.grad(u_k_fx).T), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        "LHS Diffusion transpose 1":     {'pc': pc['theta'] * pc['mu'] * inner(grad(pc['du']), grad(pc['v']).T) * dx(metadata={"q":4}),                     'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['mu'] * ufl.inner(ufl.grad(du), ufl.grad(v).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "LHS Diffusion transpose 2":     {'pc': pc['theta'] * pc['mu'] * inner(grad(pc['du']).T, grad(pc['v'])) * dx(metadata={"q":4}),                     'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['mu'] * ufl.inner(ufl.grad(du).T, ufl.grad(v)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "LHS Diffusion transpose 3":     {'pc': pc['theta'] * pc['mu'] * inner(grad(pc['du']).T, grad(pc['v']).T) * dx(metadata={"q":4}),                     'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['mu'] * ufl.inner(ufl.grad(du).T, ufl.grad(v).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "LHS Diffusion transpose 4":     {'pc': pc['theta'] * pc['mu'] * inner(grad(pc['du']) + grad(pc['du']).T, grad(pc['v']) + grad(pc['v']).T) * dx(metadata={"q":4}),                     'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['mu'] * ufl.inner(ufl.grad(du) + ufl.grad(du).T, ufl.grad(v) + ufl.grad(v).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Dot 2 of gradients LHS ohne transpose": {'pc':  inner(dot(grad(pc['du']), grad(pc['u_k'])),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(du), ufl.grad(u_k_fx)), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Dot 2 of gradients LHS  transpose": {'pc':  inner(dot(grad(pc['du']).T, grad(pc['u_k'])),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(du).T, ufl.grad(u_k_fx)), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Dot 2 of gradients LHS  transpose 2": {'pc':  inner(dot(grad(pc['du']), grad(pc['u_k']).T),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(du), ufl.grad(u_k_fx).T), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Dot 2 of gradients LHS  transpose 3": {'pc':  inner(dot(grad(pc['du']).T, grad(pc['u_k']).T),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(du).T, ufl.grad(u_k_fx).T), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Dot 2 of gradients LHS swap ohne transpose": {'pc':  inner(dot(grad(pc['u_k']),grad(pc['du']) ),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx),ufl.grad(du) ), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Dot 2 of gradients LHS swap  transpose 1": {'pc':  inner(dot(grad(pc['u_k']),grad(pc['du']).T),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx), ufl.grad(du).T), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Dot 2 of gradients LHS swap transpose 2": {'pc':  inner(dot( grad(pc['u_k']).T, grad(pc['du'])),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot( ufl.grad(u_k_fx).T , ufl.grad(du)), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Dot 2 of gradients LHS swap transpose 3": {'pc':  inner(dot( grad(pc['u_k']).T, grad(pc['du']).T),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot( ufl.grad(u_k_fx).T , ufl.grad(du).T), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Dot 2 of gradients RHS ohne transpose": {'pc':  inner(dot(grad(pc['u_k']), grad(pc['u_k'])),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx), ufl.grad(u_k_fx)), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        "Dot 2 of gradients RHS transpose": {'pc':  inner(dot(grad(pc['u_k']).T, grad(pc['u_k'])),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx).T,  ufl.grad(u_k_fx)), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        "Dot 2 of gradients RHS transpose 2": {'pc':  inner(dot(grad(pc['u_k']), grad(pc['u_k']).T),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx),  ufl.grad(u_k_fx).T), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        "Dot 2 of gradients RHS transpose 3": {'pc':  inner(dot(grad(pc['u_k']).T, grad(pc['u_k']).T),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx).T,  ufl.grad(u_k_fx).T), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        
        "Linear Green Stress": {'pc':  sigma_s_linear_weak_L(pc['du'], pc['u_k'], grad(pc['v']),inner, trace, grad, dot) * dx(metadata={"q":6}), 'f_lambda': lambda deg: sigma_s_linear_weak_L(du,u_k_fx, ufl.grad(v_fx), ufl.inner, ufl.tr, ufl.grad, ufl.dot) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 6},
        "Linearized Strain LHS": {'pc': inner(epsilon_s_linear_L(pc['du'], pc['u_k'], grad, dot), grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(epsilon_s_linear_L(du, u_k_fx, ufl.grad, ufl.dot), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},

        "Linearized Strain RHS": {'pc': inner(epsilon_s_linear_R(pc['u_k'], grad, dot), grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(epsilon_s_linear_R(u_k_fx, ufl.grad, ufl.dot), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},

        "Linear Green Stress RHS": {'pc': sigma_s_linear_weak_R(pc['u_k'], grad(pc['v']), inner, trace, grad, dot) * dx(metadata={"q":4}), 'f_lambda': lambda deg: sigma_s_linear_weak_R(u_k_fx, ufl.grad(v_fx), ufl.inner, ufl.tr, ufl.grad, ufl.dot) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},

        "Trace Operator LHS": {'pc': trace(grad(pc['du'])) * trace(grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.tr(ufl.grad(du)) * ufl.tr(ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},

        "Trace Operator RHS": {'pc': trace(grad(pc['u_k'])) * trace(grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.tr(ufl.grad(u_k_fx)) * ufl.tr(ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        "Linear Strain tensor LHS": {'pc': inner(epsilon_f(pc['du'], grad), epsilon_f(pc['v'],grad))  * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(epsilon_f(du, ufl.grad), epsilon_f(v_fx, ufl.grad)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Vector Hessian LHS": {'pc': inner(Hessian(pc['du']), Hessian(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.grad(ufl.grad(du)), ufl.grad(ufl.grad(v_fx))) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Vector Hessian RHS": {'pc': inner(Hessian(pc['u_k']), Hessian(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.grad(ufl.grad(u_k_fx)), ufl.grad(ufl.grad(v_fx))) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        "Vector Laplacian LHS": {'pc': inner(Laplacian(pc['du']), Laplacian(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.div(ufl.grad(du)), ufl.div(ufl.grad(v_fx))) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Vector Laplacian RHS": {'pc': inner(Laplacian(pc['u_k']), Laplacian(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.div(ufl.grad(u_k_fx)), ufl.div(ufl.grad(v_fx))) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        "Scalar Hessian LHS": {'pc': inner(Hessian(pc['dp']), Hessian(pc['q'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.grad(ufl.grad(dp)), ufl.grad(ufl.grad(q_fx))) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Scalar Hessian RHS": {'pc': inner(Hessian(pc['p_k']), Hessian(pc['q'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.grad(ufl.grad(p_k_fx)), ufl.grad(ufl.grad(q_fx))) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        "Scalar Laplacian LHS": {'pc': inner(Laplacian(pc['dp']), Laplacian(pc['q'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.div(ufl.grad(dp)), ufl.div(ufl.grad(q_fx))) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Scalar Laplacian RHS": {'pc': inner(Laplacian(pc['p_k']), Laplacian(pc['q'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.div(ufl.grad(p_k_fx)), ufl.div(ufl.grad(q_fx))) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        
        
        # ===========================
        # Hessian contractions with facet normal (boundary tests, ds)
        # ===========================
        # ---------------- Hessian · c (right contraction) ----------------
        "Vector Hdotc LHS": {
            'pc': inner(dot(Hessian(pc['du']), c_pc), dot(Hessian(pc['v']), c_pc)) * dx(metadata={"q":4}),
            'f_lambda': lambda deg: ufl.inner(Hdotn_fx(du, c_fx), Hdotn_fx(v_fx, c_fx)) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True, 'deg': 4,
        },
        "Vector Hdotc RHS": {
            'pc': inner(dot(Hessian(pc['u_k']), c_pc), dot(Hessian(pc['v']), c_pc)) * dx(metadata={"q":4}),
            'f_lambda': lambda deg: ufl.inner(Hdotn_fx(u_k_fx, c_fx), Hdotn_fx(v_fx, c_fx)) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False, 'deg': 4,
        },

        # ---------------- c · Hessian (left contraction) ----------------
        "Vector cdotH LHS": {
            'pc': inner(dot(c_pc, Hessian(pc['du'])), dot(c_pc, Hessian(pc['v']))) * dx(metadata={"q":6}),
            'f_lambda': lambda deg: ufl.inner(ndotH_fx(du, c_fx), ndotH_fx(v_fx, c_fx)) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True, 'deg': 4,
        },
        "Vector cdotH RHS": {
            'pc': inner(dot(c_pc, Hessian(pc['u_k'])), dot(c_pc, Hessian(pc['v']))) * dx(metadata={"q":4}),
            'f_lambda': lambda deg: ufl.inner(ndotH_fx(u_k_fx, c_fx), ndotH_fx(v_fx, c_fx)) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False, 'deg': 4,
        },

        # ---------------- double contraction cᵀ H c ----------------
        # Vector field: nHn(u) is a 2-vector; inner yields a scalar integrand
        "Vector nHn LHS": {
            'pc': inner( dot(c_pc, dot(Hessian(pc['du']), c_pc)),
                        dot(c_pc, dot(Hessian(pc['v']),  c_pc)) ) * dx(metadata={"q":4}),
            'f_lambda': lambda deg: ufl.inner(nHn_fx(du, c_fx), nHn_fx(v_fx, c_fx)) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True, 'deg': 4,
        },
        "Vector nHn RHS": {
            'pc': inner( dot(c_pc, dot(Hessian(pc['u_k']), c_pc)),
                        dot(c_pc, dot(Hessian(pc['v']),   c_pc)) ) * dx(metadata={"q":4}),
            'f_lambda': lambda deg: ufl.inner(nHn_fx(u_k_fx, c_fx), nHn_fx(v_fx, c_fx)) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False, 'deg': 4,
        },

        # ---------------- scalar versions ----------------
        # For scalar p, H·c and c·H are d-vectors; inner gives a scalar integrand.
        "Scalar Hdotc LHS": {
            'pc': inner(dot(Hessian(pc['dp']), c_pc), dot(Hessian(pc['q']), c_pc)) * dx(metadata={"q":4}),
            'f_lambda': lambda deg: ufl.inner(Hdotn_fx(dp, c_fx), Hdotn_fx(q_fx, c_fx)) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True, 'deg': 4,
        },
        "Scalar Hdotc RHS": {
            'pc': inner(dot(Hessian(pc['p_k']), c_pc), dot(Hessian(pc['q']), c_pc)) * dx(metadata={"q":4}),
            'f_lambda': lambda deg: ufl.inner(Hdotn_fx(p_k_fx, c_fx), Hdotn_fx(q_fx, c_fx)) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False, 'deg': 4,
        },

        "Scalar cdotH LHS": {
            'pc': inner(dot(c_pc, Hessian(pc['dp'])), dot(c_pc, Hessian(pc['q']))) * dx(metadata={"q":4}),
            'f_lambda': lambda deg: ufl.inner(ndotH_fx(dp, c_fx), ndotH_fx(q_fx, c_fx)) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True, 'deg': 4,
        },
        "Scalar cdotH RHS": {
            'pc': inner(dot(c_pc, Hessian(pc['p_k'])), dot(c_pc, Hessian(pc['q']))) * dx(metadata={"q":4}),
            'f_lambda': lambda deg: ufl.inner(ndotH_fx(p_k_fx, c_fx), ndotH_fx(q_fx, c_fx)) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False, 'deg': 4,
        },

        # Scalar nHn: nHn(p) is scalar; inner(scalar, scalar) is fine
        "Scalar nHn LHS": {
            'pc': inner( dot(c_pc, dot(Hessian(pc['dp']), c_pc)),
                        dot(c_pc, dot(Hessian(pc['q']),  c_pc)) ) * dx(metadata={"q":4}),
            'f_lambda': lambda deg: ufl.inner(nHn_fx(dp, c_fx), nHn_fx(q_fx, c_fx)) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True, 'deg': 4,
        },
        "Scalar nHn RHS": {
            'pc': inner( dot(c_pc, dot(Hessian(pc['p_k']), c_pc)),
                        dot(c_pc, dot(Hessian(pc['q']),   c_pc)) ) * dx(metadata={"q":4}),
            'f_lambda': lambda deg: ufl.inner(nHn_fx(p_k_fx, c_fx), nHn_fx(q_fx, c_fx)) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False, 'deg': 4,
        },

        # -------- Vector field: H · n ----------
        "Vector Hdotn LHS (ds)": {
            'pc': inner(dot(Hessian(pc['du']), n_pc), dot(Hessian(pc['v']), n_pc)) * dS(metadata = {"q":4},  tag = "all"),
            'f_lambda': lambda deg: ufl.inner(Hdotn_fx(du, n_fx), Hdotn_fx(v_fx, n_fx)) * ufl.ds(metadata={'quadrature_degree': deg}),
            'mat': True, 'deg': 4,
        },
        "Vector Hdotn RHS (ds)": {
            'pc': inner(dot(Hessian(pc['u_k']), n_pc), dot(Hessian(pc['v']), n_pc)) * dS(metadata = {"q":4},  tag = "all"),
            'f_lambda': lambda deg: ufl.inner(Hdotn_fx(u_k_fx, n_fx), Hdotn_fx(v_fx, n_fx)) * ufl.ds(metadata={'quadrature_degree': deg}),
            'mat': False, 'deg': 4,
        },

        # -------- Vector field: n · H ----------
        "Vector ndotH LHS (ds)": {
            'pc': inner(dot(n_pc, Hessian(pc['du'])), dot(n_pc, Hessian(pc['v']))) * dS(metadata = {"q":4},  tag = "all"),
            'f_lambda': lambda deg: ufl.inner(ndotH_fx(du, n_fx), ndotH_fx(v_fx, n_fx)) * ufl.ds(metadata={'quadrature_degree': deg}),
            'mat': True, 'deg': 4,
        },
        "Vector ndotH RHS (ds)": {
            'pc': inner(dot(n_pc, Hessian(pc['u_k'])), dot(n_pc, Hessian(pc['v']))) * dS(metadata = {"q":4},  tag = "all"),
            'f_lambda': lambda deg: ufl.inner(ndotH_fx(u_k_fx, n_fx), ndotH_fx(v_fx, n_fx)) * ufl.ds(metadata={'quadrature_degree': deg}),
            'mat': False, 'deg': 4,
        },

        # -------- Vector field: nᵀ H n ----------
        "Vector nHn LHS (ds)": {
            'pc': inner(dot(n_pc, dot(Hessian(pc['du']), n_pc)),
                        dot(n_pc, dot(Hessian(pc['v']),  n_pc))) * dS(metadata = {"q":4},  tag = "all"),
            'f_lambda': lambda deg: ufl.inner(nHn_fx(du, n_fx), nHn_fx(v_fx, n_fx)) * ufl.ds(metadata={'quadrature_degree': deg}),
            'mat': True, 'deg': 4,
        },
        "Vector nHn RHS (ds)": {    
            'pc': inner(dot(n_pc, dot(Hessian(pc['u_k']), n_pc)),
                        dot(n_pc, dot(Hessian(pc['v']),   n_pc))) * dS(metadata = {"q":4},  tag = "all"),
            'f_lambda': lambda deg: ufl.inner(nHn_fx(u_k_fx, n_fx), nHn_fx(v_fx, n_fx)) * ufl.ds(metadata={'quadrature_degree': deg}),
            'mat': False, 'deg': 4,
        },

        # -------- Scalar field: H · n ----------
        "Scalar Hdotn LHS (ds)": {
            'pc': inner(dot(Hessian(pc['dp']), n_pc), dot(Hessian(pc['q']), n_pc)) * dS(metadata = {"q":4},  tag = "all"),
            'f_lambda': lambda deg: ufl.inner(Hdotn_fx(dp, n_fx), Hdotn_fx(q_fx, n_fx)) * ufl.ds(metadata={'quadrature_degree': deg}),
            'mat': True, 'deg': 4,
        },
        "Scalar Hdotn RHS (ds)": {
            'pc': inner(dot(Hessian(pc['p_k']), n_pc), dot(Hessian(pc['q']), n_pc)) * dS(metadata = {"q":4},  tag = "all"),
            'f_lambda': lambda deg: ufl.inner(Hdotn_fx(p_k_fx, n_fx), Hdotn_fx(q_fx, n_fx)) * ufl.ds(metadata={'quadrature_degree': deg}),
            'mat': False, 'deg': 4,
        },

        # -------- Scalar field: n · H ----------
        "Scalar ndotH LHS (ds)": {
            'pc': inner(dot(n_pc, Hessian(pc['dp'])), dot(n_pc, Hessian(pc['q']))) * dS(metadata = {"q":4},  tag = "all"),
            'f_lambda': lambda deg: ufl.inner(ndotH_fx(dp, n_fx), ndotH_fx(q_fx, n_fx)) * ufl.ds(metadata={'quadrature_degree': deg}),
            'mat': True, 'deg': 4,
        },
        "Scalar ndotH RHS (ds)": {
            'pc': inner(dot(n_pc, Hessian(pc['p_k'])), dot(n_pc, Hessian(pc['q']))) * dS(metadata = {"q":4},  tag = "all"),
            'f_lambda': lambda deg: ufl.inner(ndotH_fx(p_k_fx, n_fx), ndotH_fx(q_fx, n_fx)) * ufl.ds(metadata={'quadrature_degree': deg}),
            'mat': False, 'deg': 4,
        },

        # -------- Scalar field: nᵀ H n ----------
        "Scalar nHn LHS (ds)": {
            'pc': inner(dot(n_pc, dot(Hessian(pc['dp']), n_pc)),
                        dot(n_pc, dot(Hessian(pc['q']),  n_pc))) * dS(metadata = {"q":4},  tag = "all"),
            'f_lambda': lambda deg: ufl.inner(nHn_fx(dp, n_fx), nHn_fx(q_fx, n_fx)) * ufl.ds(metadata={'quadrature_degree': deg}),
            'mat': True, 'deg': 4,
        },
        "Scalar nHn RHS (ds)": {
            'pc': inner(dot(n_pc, dot(Hessian(pc['p_k']), n_pc)),
                        dot(n_pc, dot(Hessian(pc['q']),   n_pc))) * dS(metadata = {"q":4},  tag = "all"),
            'f_lambda': lambda deg: ufl.inner(nHn_fx(p_k_fx, n_fx), nHn_fx(q_fx, n_fx)) * ufl.ds(metadata={'quadrature_degree': deg}),
            'mat': False, 'deg': 4,
        },
    }
    pc_dummy_side = dot(Constant([0.0,0.0],dim=1), pc['v']) * dx()
    # Keep track of successes and failures
    failed_tests = []
    success_count = 0
    import os
    backend_type = os.environ.get("COMP_FENICS_BACKEND", "jit")
    filter_terms = os.environ.get("COMP_FENICS_TERMS")
    if filter_terms:
        allowed = {name.strip() for name in filter_terms.split(",") if name.strip()}
        terms = {k: v for k, v in terms.items() if k in allowed}
        print(f"Running filtered terms only: {sorted(terms)}")
    for name, forms in terms.items():
        J_pc, R_pc, J_fx, R_fx = None, None, None, None

        form_fx_ufl = forms['f_lambda'](forms['deg'])
        form_fx_compiled = dolfinx.fem.form(form_fx_ufl)
        print(f"Compiling form for '{name}' with degree {forms['deg']}...")

        if forms['mat']:
            J_pc, _ = assemble_form(Equation(forms['pc'], None), dof_handler_pc, quad_degree=forms['deg'], bcs=[], 
                                    backend=backend_type)
            A = dolfinx.fem.petsc.assemble_matrix(form_fx_compiled)
            A.assemble()
            indptr, indices, data = A.getValuesCSR()
            J_fx_sparse = csr_matrix((data, indices, indptr), shape=A.getSize())
            J_fx = J_fx_sparse.toarray()
        else:
            _, R_pc = assemble_form(Equation(None, forms['pc']), dof_handler_pc, bcs=[], 
                                    backend=backend_type)
            vec = dolfinx.fem.petsc.assemble_vector(form_fx_compiled)
            # CORRECTED: ghostUpdate is not needed for serial runs.
            R_fx = vec.array

        is_success = compare_term(name, J_pc, R_pc, J_fx, R_fx, P_map, dof_handler_pc, W_fx)
        if is_success:
            success_count += 1
        else:
            failed_tests.append(name)
    # Print the final summary of all tests
    print_test_summary(success_count, failed_tests)
