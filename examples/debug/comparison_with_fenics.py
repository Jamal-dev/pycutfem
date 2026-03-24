import numpy as np
import pandas as pd
import os
import sys
import scipy.sparse as sp

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
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.expressions import (
    HdivFunction,
    HdivTestFunction,
    HdivTrialFunction,
    TrialFunction, TestFunction, VectorTrialFunction, VectorTestFunction,
    Function, VectorFunction, Constant, grad, inner,
    dot, div, trace, Hessian, Laplacian, FacetNormal, Identity, det, inv,
    pos_part, heaviside, MeshSize,
)
from pycutfem.ufl.measures import dx, dInterface, dS
from pycutfem.ufl.forms import assemble_form
from pycutfem.fem.reference import get_reference
from pycutfem.fem.reference.rt import (
    _edge_normal,
    _edge_param,
    _legendre_all,
    _monomials_quad,
    gauss_legendre,
    _quad_rule,
)
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.forms import Equation
from pycutfem.utils.bitset import BitSet
from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms
from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import (
    _build_forms as build_benchmark7_seboldt_forms,
    _create_problem as create_benchmark7_seboldt_problem,
)

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

def get_all_pycutfem_dof_coords(dof_handler: DofHandler, *, fields: list[str] | None = None) -> np.ndarray:
    all_coords = np.zeros((dof_handler.total_dofs, 2))
    if fields is None:
        fields = list(dof_handler.field_names)
    for field in fields:
        field_dofs = dof_handler.get_field_slice(field)
        field_coords = get_pycutfem_dof_coords(dof_handler, field)
        all_coords[field_dofs] = field_coords
    return all_coords

def get_all_fenicsx_dof_coords(W_fenicsx):
    num_total_dofs = W_fenicsx.dofmap.index_map.size_global
    all_coords = np.zeros((num_total_dofs, 2))
    nsub = int(getattr(W_fenicsx, "num_sub_spaces", 0))

    if nsub == 2:
        W0, V_map = W_fenicsx.sub(0).collapse()
        W1, P_map_fx = W_fenicsx.sub(1).collapse()
        nsub0 = int(getattr(W0, "num_sub_spaces", 0))
        if nsub0 == 2:
            # Mixed space: (vector, scalar)
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
        if nsub0 == 0:
            # Mixed space: (scalar, scalar)
            coords_0 = W0.tabulate_dof_coordinates()[:, :2]
            coords_1 = W1.tabulate_dof_coordinates()[:, :2]
            dofs_0 = np.array(V_map)
            dofs_1 = np.array(P_map_fx)
            all_coords[dofs_0] = coords_0
            all_coords[dofs_1] = coords_1
            return all_coords
        raise NotImplementedError(f"Unsupported (nsub==2) layout with subspace0.num_sub_spaces={nsub0}.")

    if nsub == 3:
        # Mixed space: (vector, vector, scalar) e.g. (v, u, p)
        Wv, Vv_map = W_fenicsx.sub(0).collapse()
        Wu, Vu_map = W_fenicsx.sub(1).collapse()
        Wp, P_map_fx = W_fenicsx.sub(2).collapse()

        Wv0, Vv0_map = Wv.sub(0).collapse()
        Wv1, Vv1_map = Wv.sub(1).collapse()
        Wu0, Vu0_map = Wu.sub(0).collapse()
        Wu1, Vu1_map = Wu.sub(1).collapse()

        coords_vx = Wv0.tabulate_dof_coordinates()[:, :2]
        coords_vy = Wv1.tabulate_dof_coordinates()[:, :2]
        coords_ux = Wu0.tabulate_dof_coordinates()[:, :2]
        coords_uy = Wu1.tabulate_dof_coordinates()[:, :2]
        coords_p = Wp.tabulate_dof_coordinates()[:, :2]

        dofs_vx = np.array(Vv_map)[np.array(Vv0_map)]
        dofs_vy = np.array(Vv_map)[np.array(Vv1_map)]
        dofs_ux = np.array(Vu_map)[np.array(Vu0_map)]
        dofs_uy = np.array(Vu_map)[np.array(Vu1_map)]
        dofs_p = np.array(P_map_fx)

        all_coords[dofs_vx] = coords_vx
        all_coords[dofs_vy] = coords_vy
        all_coords[dofs_ux] = coords_ux
        all_coords[dofs_uy] = coords_uy
        all_coords[dofs_p] = coords_p
        return all_coords

    if nsub == 6:
        # Mixed space layout: (vector, scalar, vector, scalar, scalar, scalar)
        # i.e. (v, p, u, phi, alpha, S).
        Wv, Vv_map = W_fenicsx.sub(0).collapse()
        Wp, P_map_fx = W_fenicsx.sub(1).collapse()
        Wu, Vu_map = W_fenicsx.sub(2).collapse()
        Wphi, Phi_map_fx = W_fenicsx.sub(3).collapse()
        Walpha, Alpha_map_fx = W_fenicsx.sub(4).collapse()
        WS, S_map_fx = W_fenicsx.sub(5).collapse()

        Wv0, Vv0_map = Wv.sub(0).collapse()
        Wv1, Vv1_map = Wv.sub(1).collapse()
        Wu0, Vu0_map = Wu.sub(0).collapse()
        Wu1, Vu1_map = Wu.sub(1).collapse()

        coords_vx = Wv0.tabulate_dof_coordinates()[:, :2]
        coords_vy = Wv1.tabulate_dof_coordinates()[:, :2]
        coords_p = Wp.tabulate_dof_coordinates()[:, :2]
        coords_ux = Wu0.tabulate_dof_coordinates()[:, :2]
        coords_uy = Wu1.tabulate_dof_coordinates()[:, :2]
        coords_phi = Wphi.tabulate_dof_coordinates()[:, :2]
        coords_alpha = Walpha.tabulate_dof_coordinates()[:, :2]
        coords_S = WS.tabulate_dof_coordinates()[:, :2]

        dofs_vx = np.array(Vv_map)[np.array(Vv0_map)]
        dofs_vy = np.array(Vv_map)[np.array(Vv1_map)]
        dofs_p = np.array(P_map_fx)
        dofs_ux = np.array(Vu_map)[np.array(Vu0_map)]
        dofs_uy = np.array(Vu_map)[np.array(Vu1_map)]
        dofs_phi = np.array(Phi_map_fx)
        dofs_alpha = np.array(Alpha_map_fx)
        dofs_S = np.array(S_map_fx)

        all_coords[dofs_vx] = coords_vx
        all_coords[dofs_vy] = coords_vy
        all_coords[dofs_p] = coords_p
        all_coords[dofs_ux] = coords_ux
        all_coords[dofs_uy] = coords_uy
        all_coords[dofs_phi] = coords_phi
        all_coords[dofs_alpha] = coords_alpha
        all_coords[dofs_S] = coords_S
        return all_coords

    if nsub == 8:
        # Mixed space layout: (vector, scalar, vector, scalar, scalar, scalar, scalar, scalar)
        # i.e. (v, p, u, phi, alpha, d, S, X) for full biofilm+damage+detached comparisons.
        Wv, Vv_map = W_fenicsx.sub(0).collapse()
        Wp, P_map_fx = W_fenicsx.sub(1).collapse()
        Wu, Vu_map = W_fenicsx.sub(2).collapse()
        Wphi, Phi_map_fx = W_fenicsx.sub(3).collapse()
        Walpha, Alpha_map_fx = W_fenicsx.sub(4).collapse()
        Wd, D_map_fx = W_fenicsx.sub(5).collapse()
        WS, S_map_fx = W_fenicsx.sub(6).collapse()
        WX, X_map_fx = W_fenicsx.sub(7).collapse()

        Wv0, Vv0_map = Wv.sub(0).collapse()
        Wv1, Vv1_map = Wv.sub(1).collapse()
        Wu0, Vu0_map = Wu.sub(0).collapse()
        Wu1, Vu1_map = Wu.sub(1).collapse()

        coords_vx = Wv0.tabulate_dof_coordinates()[:, :2]
        coords_vy = Wv1.tabulate_dof_coordinates()[:, :2]
        coords_p = Wp.tabulate_dof_coordinates()[:, :2]
        coords_ux = Wu0.tabulate_dof_coordinates()[:, :2]
        coords_uy = Wu1.tabulate_dof_coordinates()[:, :2]
        coords_phi = Wphi.tabulate_dof_coordinates()[:, :2]
        coords_alpha = Walpha.tabulate_dof_coordinates()[:, :2]
        coords_d = Wd.tabulate_dof_coordinates()[:, :2]
        coords_S = WS.tabulate_dof_coordinates()[:, :2]
        coords_X = WX.tabulate_dof_coordinates()[:, :2]

        dofs_vx = np.array(Vv_map)[np.array(Vv0_map)]
        dofs_vy = np.array(Vv_map)[np.array(Vv1_map)]
        dofs_p = np.array(P_map_fx)
        dofs_ux = np.array(Vu_map)[np.array(Vu0_map)]
        dofs_uy = np.array(Vu_map)[np.array(Vu1_map)]
        dofs_phi = np.array(Phi_map_fx)
        dofs_alpha = np.array(Alpha_map_fx)
        dofs_d = np.array(D_map_fx)
        dofs_S = np.array(S_map_fx)
        dofs_X = np.array(X_map_fx)

        all_coords[dofs_vx] = coords_vx
        all_coords[dofs_vy] = coords_vy
        all_coords[dofs_p] = coords_p
        all_coords[dofs_ux] = coords_ux
        all_coords[dofs_uy] = coords_uy
        all_coords[dofs_phi] = coords_phi
        all_coords[dofs_alpha] = coords_alpha
        all_coords[dofs_d] = coords_d
        all_coords[dofs_S] = coords_S
        all_coords[dofs_X] = coords_X
        return all_coords

    if nsub == 9:
        # Mixed space layout: (vector, scalar, vector, vector, scalar, scalar, scalar, scalar, scalar)
        # i.e. (v, p, vS, u, phi, alpha, d, S, X) for one-domain biofilm comparisons.
        Wv, Vv_map = W_fenicsx.sub(0).collapse()
        Wp, P_map_fx = W_fenicsx.sub(1).collapse()
        WvS, VvS_map = W_fenicsx.sub(2).collapse()
        Wu, Vu_map = W_fenicsx.sub(3).collapse()
        Wphi, Phi_map_fx = W_fenicsx.sub(4).collapse()
        Walpha, Alpha_map_fx = W_fenicsx.sub(5).collapse()
        Wd, D_map_fx = W_fenicsx.sub(6).collapse()
        WS, S_map_fx = W_fenicsx.sub(7).collapse()
        WX, X_map_fx = W_fenicsx.sub(8).collapse()

        Wv0, Vv0_map = Wv.sub(0).collapse()
        Wv1, Vv1_map = Wv.sub(1).collapse()
        WvS0, VvS0_map = WvS.sub(0).collapse()
        WvS1, VvS1_map = WvS.sub(1).collapse()
        Wu0, Vu0_map = Wu.sub(0).collapse()
        Wu1, Vu1_map = Wu.sub(1).collapse()

        coords_vx = Wv0.tabulate_dof_coordinates()[:, :2]
        coords_vy = Wv1.tabulate_dof_coordinates()[:, :2]
        coords_p = Wp.tabulate_dof_coordinates()[:, :2]
        coords_vSx = WvS0.tabulate_dof_coordinates()[:, :2]
        coords_vSy = WvS1.tabulate_dof_coordinates()[:, :2]
        coords_ux = Wu0.tabulate_dof_coordinates()[:, :2]
        coords_uy = Wu1.tabulate_dof_coordinates()[:, :2]
        coords_phi = Wphi.tabulate_dof_coordinates()[:, :2]
        coords_alpha = Walpha.tabulate_dof_coordinates()[:, :2]
        coords_d = Wd.tabulate_dof_coordinates()[:, :2]
        coords_S = WS.tabulate_dof_coordinates()[:, :2]
        coords_X = WX.tabulate_dof_coordinates()[:, :2]

        dofs_vx = np.array(Vv_map)[np.array(Vv0_map)]
        dofs_vy = np.array(Vv_map)[np.array(Vv1_map)]
        dofs_p = np.array(P_map_fx)
        dofs_vSx = np.array(VvS_map)[np.array(VvS0_map)]
        dofs_vSy = np.array(VvS_map)[np.array(VvS1_map)]
        dofs_ux = np.array(Vu_map)[np.array(Vu0_map)]
        dofs_uy = np.array(Vu_map)[np.array(Vu1_map)]
        dofs_phi = np.array(Phi_map_fx)
        dofs_alpha = np.array(Alpha_map_fx)
        dofs_d = np.array(D_map_fx)
        dofs_S = np.array(S_map_fx)
        dofs_X = np.array(X_map_fx)

        all_coords[dofs_vx] = coords_vx
        all_coords[dofs_vy] = coords_vy
        all_coords[dofs_p] = coords_p
        all_coords[dofs_vSx] = coords_vSx
        all_coords[dofs_vSy] = coords_vSy
        all_coords[dofs_ux] = coords_ux
        all_coords[dofs_uy] = coords_uy
        all_coords[dofs_phi] = coords_phi
        all_coords[dofs_alpha] = coords_alpha
        all_coords[dofs_d] = coords_d
        all_coords[dofs_S] = coords_S
        all_coords[dofs_X] = coords_X
        return all_coords

    if nsub == 10:
        # Mixed space layout: (vector, scalar, vector, vector, scalar, scalar, scalar, scalar, scalar, scalar)
        # i.e. (v, p, vS, u, phi, alpha, extra, d, S, X), where "extra" is a scalar
        # field (e.g. mu_alpha for Cahn–Hilliard or lambda_alpha for conservative AC).
        Wv, Vv_map = W_fenicsx.sub(0).collapse()
        Wp, P_map_fx = W_fenicsx.sub(1).collapse()
        WvS, VvS_map = W_fenicsx.sub(2).collapse()
        Wu, Vu_map = W_fenicsx.sub(3).collapse()
        Wphi, Phi_map_fx = W_fenicsx.sub(4).collapse()
        Walpha, Alpha_map_fx = W_fenicsx.sub(5).collapse()
        Wextra, Extra_map_fx = W_fenicsx.sub(6).collapse()
        Wd, D_map_fx = W_fenicsx.sub(7).collapse()
        WS, S_map_fx = W_fenicsx.sub(8).collapse()
        WX, X_map_fx = W_fenicsx.sub(9).collapse()

        Wv0, Vv0_map = Wv.sub(0).collapse()
        Wv1, Vv1_map = Wv.sub(1).collapse()
        WvS0, VvS0_map = WvS.sub(0).collapse()
        WvS1, VvS1_map = WvS.sub(1).collapse()
        Wu0, Vu0_map = Wu.sub(0).collapse()
        Wu1, Vu1_map = Wu.sub(1).collapse()

        coords_vx = Wv0.tabulate_dof_coordinates()[:, :2]
        coords_vy = Wv1.tabulate_dof_coordinates()[:, :2]
        coords_p = Wp.tabulate_dof_coordinates()[:, :2]
        coords_vSx = WvS0.tabulate_dof_coordinates()[:, :2]
        coords_vSy = WvS1.tabulate_dof_coordinates()[:, :2]
        coords_ux = Wu0.tabulate_dof_coordinates()[:, :2]
        coords_uy = Wu1.tabulate_dof_coordinates()[:, :2]
        coords_phi = Wphi.tabulate_dof_coordinates()[:, :2]
        coords_alpha = Walpha.tabulate_dof_coordinates()[:, :2]
        coords_extra = Wextra.tabulate_dof_coordinates()[:, :2]
        coords_d = Wd.tabulate_dof_coordinates()[:, :2]
        coords_S = WS.tabulate_dof_coordinates()[:, :2]
        coords_X = WX.tabulate_dof_coordinates()[:, :2]

        dofs_vx = np.array(Vv_map)[np.array(Vv0_map)]
        dofs_vy = np.array(Vv_map)[np.array(Vv1_map)]
        dofs_p = np.array(P_map_fx)
        dofs_vSx = np.array(VvS_map)[np.array(VvS0_map)]
        dofs_vSy = np.array(VvS_map)[np.array(VvS1_map)]
        dofs_ux = np.array(Vu_map)[np.array(Vu0_map)]
        dofs_uy = np.array(Vu_map)[np.array(Vu1_map)]
        dofs_phi = np.array(Phi_map_fx)
        dofs_alpha = np.array(Alpha_map_fx)
        dofs_extra = np.array(Extra_map_fx)
        dofs_d = np.array(D_map_fx)
        dofs_S = np.array(S_map_fx)
        dofs_X = np.array(X_map_fx)

        all_coords[dofs_vx] = coords_vx
        all_coords[dofs_vy] = coords_vy
        all_coords[dofs_p] = coords_p
        all_coords[dofs_vSx] = coords_vSx
        all_coords[dofs_vSy] = coords_vSy
        all_coords[dofs_ux] = coords_ux
        all_coords[dofs_uy] = coords_uy
        all_coords[dofs_phi] = coords_phi
        all_coords[dofs_alpha] = coords_alpha
        all_coords[dofs_extra] = coords_extra
        all_coords[dofs_d] = coords_d
        all_coords[dofs_S] = coords_S
        all_coords[dofs_X] = coords_X
        return all_coords

    # Generic fallback for simple (Lagrange) mixed spaces: collapse each subspace
    # and inject coordinates into the parent vector using the provided map.
    try:
        for i in range(int(nsub)):
            Wi, sub_map = W_fenicsx.sub(i).collapse()
            coords_i = Wi.tabulate_dof_coordinates()[:, :2]
            dofs_i = np.asarray(sub_map, dtype=int)
            if coords_i.shape[0] != dofs_i.shape[0]:
                raise ValueError(
                    f"subspace {i} coordinate count mismatch: coords={coords_i.shape[0]} dofs={dofs_i.shape[0]}"
                )
            all_coords[dofs_i] = coords_i
        return all_coords
    except Exception as exc:
        raise NotImplementedError(f"Unsupported mixed space layout with num_sub_spaces={nsub}.") from exc

def one_to_one_map_coords(coords1, coords2):
    coords1 = np.asarray(coords1, dtype=float)
    coords2 = np.asarray(coords2, dtype=float)
    if coords1.shape != coords2.shape:
        raise ValueError(f"Coordinate arrays must have the same shape, got {coords1.shape} vs {coords2.shape}.")

    n = int(coords1.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=int)

    tol = float(os.environ.get("COMP_FENICS_COORD_TOL", "1e-12"))
    scale = 1.0 / max(tol, 1.0e-16)

    def _quantize(arr):
        return np.rint(np.asarray(arr, dtype=float) * scale).astype(np.int64)

    keys1 = _quantize(coords1)
    keys2 = _quantize(coords2)

    buckets: dict[tuple[int, ...], list[int]] = {}
    for idx, key in enumerate(keys2):
        buckets.setdefault(tuple(int(v) for v in key.tolist()), []).append(int(idx))

    mapping = np.empty(n, dtype=int)
    try:
        for i, key in enumerate(keys1):
            bucket_key = tuple(int(v) for v in key.tolist())
            bucket = buckets.get(bucket_key)
            if not bucket:
                raise KeyError(bucket_key)
            mapping[i] = int(bucket.pop())
        return mapping
    except Exception:
        # Fallback for near-coincident coordinates that escape the quantized map.
        C = np.linalg.norm(coords2[:, np.newaxis, :] - coords1[np.newaxis, :, :], axis=2)
        rows, cols = linear_sum_assignment(C)
        return rows[np.argsort(cols)]


def _fenics_rt_dofs_and_descriptors(V_fenicsx):
    """Return collapsed RT dofs and entity descriptors for quadrilateral H(div) spaces."""
    entity_dofs = V_fenicsx.element.basix_element.entity_dofs
    mesh_fx = V_fenicsx.mesh
    tdim = mesh_fx.topology.dim
    fdim = tdim - 1
    mesh_fx.topology.create_connectivity(tdim, fdim)
    mesh_fx.topology.create_connectivity(fdim, 0)
    mesh_fx.topology.create_connectivity(tdim, 0)
    cell_to_facet = mesh_fx.topology.connectivity(tdim, fdim)
    facet_to_vertex = mesh_fx.topology.connectivity(fdim, 0)
    cell_to_vertex = mesh_fx.topology.connectivity(tdim, 0)

    dof_to_kind: dict[int, str] = {}
    dof_to_point: dict[int, np.ndarray] = {}
    dof_to_mode: dict[int, int] = {}
    geom = np.asarray(mesh_fx.geometry.x, dtype=float)
    for cell, row in enumerate(V_fenicsx.dofmap.list):
        local_facets = cell_to_facet.links(int(cell))
        for local_facet, local_dofs in enumerate(entity_dofs[fdim]):
            facet = int(local_facets[int(local_facet)])
            verts = np.asarray(facet_to_vertex.links(facet), dtype=int)
            midpoint = np.mean(geom[verts, :2], axis=0)
            for mode_idx, local_dof in enumerate(local_dofs):
                gdof = int(row[int(local_dof)])
                dof_to_kind.setdefault(gdof, "edge")
                dof_to_point.setdefault(gdof, np.asarray(midpoint, dtype=float))
                dof_to_mode.setdefault(gdof, int(mode_idx))

        cell_local_dofs = entity_dofs[tdim][0] if len(entity_dofs[tdim]) else []
        if len(cell_local_dofs):
            verts = np.asarray(cell_to_vertex.links(int(cell)), dtype=int)
            centroid = np.mean(geom[verts, :2], axis=0)
            for mode_idx, local_dof in enumerate(cell_local_dofs):
                gdof = int(row[int(local_dof)])
                dof_to_kind.setdefault(gdof, "cell")
                dof_to_point.setdefault(gdof, np.asarray(centroid, dtype=float))
                dof_to_mode.setdefault(gdof, int(mode_idx))

    dofs = np.asarray(sorted(dof_to_point.keys()), dtype=int)
    kinds = [str(dof_to_kind[int(d)]) for d in dofs]
    points = np.vstack([np.asarray(dof_to_point[int(d)], dtype=float) for d in dofs])
    modes = np.asarray([int(dof_to_mode[int(d)]) for d in dofs], dtype=int)
    return dofs, kinds, points, modes


def _pycutfem_rt_dofs_and_descriptors(dof_handler: DofHandler, field: str):
    """Return pycutfem RT dofs and entity descriptors in field-slice order."""
    info = getattr(dof_handler, "_hdiv_field_info", {}).get(str(field), None)
    if info is None:
        raise ValueError(f"Missing H(div) metadata for field {field!r}.")

    mesh = dof_handler.mixed_element.mesh
    corner_conn = getattr(mesh, "corner_connectivity", None)
    if corner_conn is None:
        corner_conn = getattr(mesh, "elements_corner_nodes", None)
    if corner_conn is None:
        raise RuntimeError("Mesh does not expose corner connectivity required for RT descriptor mapping.")

    desc_by_gid: dict[int, tuple[str, np.ndarray, int]] = {}
    edge_base = int(info["edge_base"])
    cell_base = int(info["cell_base"])
    n_edge_dofs = int(info["n_edge_dofs"])
    n_cell_dofs = int(info["n_cell_dofs"])

    for ent, nodes in enumerate(info["edge_entity_nodes"]):
        p0 = np.asarray(mesh.nodes_x_y_pos[int(nodes[0])], dtype=float)
        p1 = np.asarray(mesh.nodes_x_y_pos[int(nodes[1])], dtype=float)
        midpoint = 0.5 * (p0 + p1)
        for mode_idx in range(n_edge_dofs):
            gdof = edge_base + int(ent) * n_edge_dofs + int(mode_idx)
            desc_by_gid[int(gdof)] = ("edge", np.asarray(midpoint, dtype=float), int(mode_idx))

    if n_cell_dofs:
        for eid in range(mesh.n_elements):
            corners = np.asarray(corner_conn[int(eid)], dtype=int)
            centroid = np.mean(np.asarray(mesh.nodes_x_y_pos[corners], dtype=float), axis=0)
            for mode_idx in range(n_cell_dofs):
                gdof = cell_base + int(eid) * n_cell_dofs + int(mode_idx)
                desc_by_gid[int(gdof)] = ("cell", np.asarray(centroid, dtype=float), int(mode_idx))

    dofs = np.asarray(dof_handler.get_field_slice(field), dtype=int)
    kinds = [str(desc_by_gid[int(d)][0]) for d in dofs]
    points = np.vstack([np.asarray(desc_by_gid[int(d)][1], dtype=float) for d in dofs])
    modes = np.asarray([int(desc_by_gid[int(d)][2]) for d in dofs], dtype=int)
    return dofs, kinds, points, modes


def _map_rt_descriptors(pc_kinds, pc_points, pc_modes, fx_kinds, fx_points, fx_modes):
    tol = float(os.environ.get("COMP_FENICS_COORD_TOL", "1e-12"))
    scale = 1.0 / max(tol, 1.0e-16)

    def _key(kind, point, mode):
        qxy = np.rint(np.asarray(point, dtype=float) * scale).astype(np.int64)
        return (str(kind), int(mode), int(qxy[0]), int(qxy[1]))

    buckets: dict[tuple[str, int, int, int], list[int]] = {}
    for idx, (kind, point, mode) in enumerate(zip(fx_kinds, fx_points, fx_modes)):
        buckets.setdefault(_key(kind, point, mode), []).append(int(idx))

    mapping = np.empty(len(pc_kinds), dtype=int)
    for i, (kind, point, mode) in enumerate(zip(pc_kinds, pc_points, pc_modes)):
        key = _key(kind, point, mode)
        bucket = buckets.get(key)
        if not bucket:
            raise KeyError(f"Unable to match RT descriptor {key}.")
        mapping[i] = int(bucket.pop())
    return mapping


def _evaluate_fenicsx_basis_values(V_fenicsx, pts_phys: np.ndarray) -> np.ndarray:
    pts_phys = np.asarray(pts_phys, dtype=float)
    ndofs = int(V_fenicsx.dofmap.index_map.size_global)
    cells = np.array([0], dtype=np.int32)
    values = np.empty((pts_phys.shape[0], ndofs, 2), dtype=float)
    expr_points = np.ascontiguousarray(pts_phys[:, ::-1], dtype=float)
    basis_fn = dolfinx.fem.Function(V_fenicsx)
    for j in range(ndofs):
        basis_fn.x.array[:] = 0.0
        basis_fn.x.array[int(j)] = 1.0
        basis_fn.x.scatter_forward()
        expr = dolfinx.fem.Expression(basis_fn, expr_points)
        raw = np.asarray(expr.eval(V_fenicsx.mesh, cells), dtype=float).reshape(1, pts_phys.shape[0], 2)
        values[:, j, :] = raw[0]
    return values


def _quadrilateral_rt_fenicsx_row_signs_against_pycutfem(V_fenicsx) -> np.ndarray:
    """
    Return the per-row sign needed to map quadrilateral FEniCSx RT coefficients
    into the same local edge-mode convention used by pycutfem.

    Both implementations agree on the bottom and right edge parameter
    directions. On the top and left edges the parameter directions are
    opposite, so odd Legendre edge modes pick up a minus sign.
    """
    fx_dofs, fx_kinds, fx_points, fx_modes = _fenics_rt_dofs_and_descriptors(V_fenicsx)
    signs = np.ones((len(fx_dofs),), dtype=float)
    if len(fx_points) == 0:
        return signs

    tol = float(os.environ.get("COMP_FENICS_COORD_TOL", "1e-12"))
    x_min = float(np.min(fx_points[:, 0]))
    y_max = float(np.max(fx_points[:, 1]))
    for i, (kind, point, mode) in enumerate(zip(fx_kinds, fx_points, fx_modes)):
        if str(kind) != "edge":
            continue
        on_left = abs(float(point[0]) - x_min) <= tol
        on_top = abs(float(point[1]) - y_max) <= tol
        if (on_left or on_top) and (int(mode) % 2 == 1):
            signs[i] = -1.0
    return signs


def _build_rt1_local_transform(me_pc: MixedElement, V_fenicsx, *, field: str = "v") -> np.ndarray:
    """
    Return the single-cell quadrilateral RT_k coefficient transform B such that
        u_fx = B @ u_pc
    maps pycutfem RT_k coefficients to the FEniCSx RT_k coefficients on a unit quad.

    This is a true local basis transform, not just a sign/permutation map.
    """
    ref_pc = me_pc._ref[str(field)]
    nloc = int(ref_pc.n_dofs)
    if ref_pc.element_type != "quad" or int(ref_pc.k) < 1:
        raise NotImplementedError("Whole-domain RT transform helper currently targets quadrilateral RTk with k >= 1 only.")
    k = int(ref_pc.k)

    C_fx_to_pc = np.zeros((nloc, nloc), dtype=float)
    row = 0

    qmom = max(6, 2 * k + 4)
    s, w = gauss_legendre(qmom)
    P = _legendre_all(k, s)
    for edge in range(4):
        xi_ref, eta_ref, w_scale = _edge_param("quad", edge, s)
        pts_phys = np.column_stack((0.5 * (xi_ref + 1.0), 0.5 * (eta_ref + 1.0)))
        vals_fx = _evaluate_fenicsx_basis_values(V_fenicsx, pts_phys)
        nvec = _edge_normal("quad", edge)
        flux = vals_fx[:, :, 0] * float(nvec[0]) + vals_fx[:, :, 1] * float(nvec[1])
        ww = (0.5 * w * w_scale).reshape(-1, 1)
        for mode in range(k + 1):
            C_fx_to_pc[row, :] = np.sum(ww * P[mode][:, None] * flux, axis=0)
            row += 1

    qp_ref, qw_ref = _quad_rule(qmom)
    qp_ref = np.asarray(qp_ref, dtype=float)
    qw_ref = np.asarray(qw_ref, dtype=float)
    pts_phys = 0.5 * (qp_ref + 1.0)
    vals_fx = _evaluate_fenicsx_basis_values(V_fenicsx, pts_phys)
    cell_monos = ((_monomials_quad(k - 1, k), 0), (_monomials_quad(k, k - 1), 1))
    for monos, comp in cell_monos:
        for i, j in monos:
            weight = (0.5 * qw_ref * (qp_ref[:, 0] ** int(i)) * (qp_ref[:, 1] ** int(j))).reshape(-1, 1)
            C_fx_to_pc[row, :] = np.sum(weight * vals_fx[:, :, int(comp)], axis=0)
            row += 1

    if np.linalg.matrix_rank(C_fx_to_pc) != nloc:
        raise RuntimeError("RT local FEniCSx->pycutfem transform is singular.")

    B_pc_to_fx = np.linalg.inv(C_fx_to_pc)
    fx_sign = _quadrilateral_rt_fenicsx_row_signs_against_pycutfem(V_fenicsx)
    if fx_sign.shape[0] != B_pc_to_fx.shape[0]:
        raise RuntimeError("RT transform sign vector size mismatch.")
    return fx_sign[:, None] * B_pc_to_fx


def _recover_sign_congruence(A_pc: np.ndarray, A_fx: np.ndarray, *, tol: float = 1.0e-12) -> np.ndarray:
    """
    Recover a diagonal sign vector s such that A_pc ~= diag(s) A_fx diag(s).

    This is used for H(div) spaces where the global facet orientation may differ
    between pycutfem and FEniCSx.
    """
    A_pc = np.asarray(A_pc, dtype=float)
    A_fx = np.asarray(A_fx, dtype=float)
    if A_pc.shape != A_fx.shape:
        raise ValueError(f"Sign recovery shape mismatch: {A_pc.shape} vs {A_fx.shape}")

    n = int(A_pc.shape[0])
    s = np.ones(n, dtype=float)
    seen = np.zeros(n, dtype=bool)
    graph = (np.abs(A_pc) > tol) | (np.abs(A_fx) > tol)

    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        stack = [int(start)]
        while stack:
            i = int(stack.pop())
            neigh = np.nonzero(graph[i])[0]
            for j in neigh:
                j = int(j)
                if i == j:
                    continue
                aij = float(A_pc[i, j])
                bij = float(A_fx[i, j])
                if abs(aij) <= tol and abs(bij) <= tol:
                    continue
                if abs(aij) <= tol or abs(bij) <= tol:
                    raise RuntimeError(
                        f"Cannot recover RT sign: zero-pattern mismatch at ({i}, {j}) "
                        f"with pycutfem={aij:.3e}, fenics={bij:.3e}."
                    )
                sign_ij = 1.0 if (aij / bij) >= 0.0 else -1.0
                cand = s[i] * sign_ij
                if not seen[j]:
                    s[j] = cand
                    seen[j] = True
                    stack.append(j)
                elif abs(s[j] - cand) > tol:
                    raise RuntimeError(
                        f"Inconsistent RT sign recovery between dofs {i} and {j}: "
                        f"existing {s[j]:+.0f}, candidate {cand:+.0f}."
                    )

    A_fx_signed = s[:, None] * A_fx * s[None, :]
    np.testing.assert_allclose(A_pc, A_fx_signed, rtol=1.0e-9, atol=max(tol, 1.0e-11))
    return s


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _sparse_matrix_allclose(A, B, *, rtol: float = 1.0e-8, atol: float = 1.0e-8):
    A_csr = A.tocsr() if sp.issparse(A) else csr_matrix(np.asarray(A, dtype=float))
    B_csr = B.tocsr() if sp.issparse(B) else csr_matrix(np.asarray(B, dtype=float))
    if A_csr.shape != B_csr.shape:
        return False, float("inf"), ("shape", A_csr.shape, B_csr.shape, float("inf"), 0.0, 0.0)

    diff = (A_csr - B_csr).tocsr()
    diff.eliminate_zeros()
    if diff.nnz == 0:
        return True, 0.0, None

    abs_data = np.abs(diff.data)
    idx = int(np.argmax(abs_data))
    max_abs = float(abs_data[idx])
    rows, cols = diff.nonzero()
    row = int(rows[idx])
    col = int(cols[idx])
    a_val = float(A_csr[row, col])
    b_val = float(B_csr[row, col])
    scale = max(
        float(np.max(np.abs(A_csr.data))) if A_csr.nnz else 0.0,
        float(np.max(np.abs(B_csr.data))) if B_csr.nnz else 0.0,
        1.0,
    )
    tol = float(atol) + float(rtol) * scale
    return max_abs <= tol, max_abs, (row, col, a_val, b_val, max_abs, tol)

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


# ==============================================================================
#  Poroelastic (Eulerian/reference-map) comparison harness (v,u,p) vs FEniCSx
# ==============================================================================


def setup_poro_problems(*, nx: int = 1, ny: int = 1):
    """
    Build a small poro mixed space:
      pycutfem: (v_x,v_y,u_x,u_y) Q2 + p Q1
      dolfinx : (v,u,p) as (P2_vec, P2_vec, P1)
    """
    # --- pycutfem mesh (Q2 geometry so coordinates match dolfinx's Q2 dofs) ---
    nodes_q2, elems_q2, _, corners_q2 = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=2)
    mesh_q2 = Mesh(
        nodes=nodes_q2,
        element_connectivity=elems_q2,
        elements_corner_nodes=corners_q2,
        element_type="quad",
        poly_order=2,
    )
    mesh_q2.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= 1.0e-12,
            "right": lambda x, y: abs(x - 1.0) <= 1.0e-12,
            "bottom": lambda x, y: abs(y - 0.0) <= 1.0e-12,
            "top": lambda x, y: abs(y - 1.0) <= 1.0e-12,
            "all": lambda x, y: True,
        }
    )

    mixed_element_pc = MixedElement(mesh_q2, field_specs={"v_x": 2, "v_y": 2, "u_x": 2, "u_y": 2, "p": 1})
    dof_handler_pc = DofHandler(mixed_element_pc, method="cg")

    Vv_pc = FunctionSpace("v", ["v_x", "v_y"], dim=1)
    Vu_pc = FunctionSpace("u", ["u_x", "u_y"], dim=1)
    Qp_pc = FunctionSpace("p", ["p"], dim=0)

    pc = {
        # trial/test
        "dv": VectorTrialFunction(Vv_pc, dof_handler=dof_handler_pc),
        "du": VectorTrialFunction(Vu_pc, dof_handler=dof_handler_pc),
        "dp": TrialFunction(Qp_pc, dof_handler=dof_handler_pc),
        "w": VectorTestFunction(Vv_pc, dof_handler=dof_handler_pc),
        "eta": VectorTestFunction(Vu_pc, dof_handler=dof_handler_pc),
        "q": TestFunction(Qp_pc, dof_handler=dof_handler_pc),
        # states
        "v_k": VectorFunction(name="v_k", field_names=["v_x", "v_y"], dof_handler=dof_handler_pc),
        "u_k": VectorFunction(name="u_k", field_names=["u_x", "u_y"], dof_handler=dof_handler_pc),
        "p_k": Function(name="p_k", field_name="p", dof_handler=dof_handler_pc),
        "v_n": VectorFunction(name="v_n", field_names=["v_x", "v_y"], dof_handler=dof_handler_pc),
        "u_n": VectorFunction(name="u_n", field_names=["u_x", "u_y"], dof_handler=dof_handler_pc),
        "u_nm1": VectorFunction(name="u_nm1", field_names=["u_x", "u_y"], dof_handler=dof_handler_pc),
        "p_n": Function(name="p_n", field_name="p", dof_handler=dof_handler_pc),
        # parameters
        "rho_f": Constant(0.9, dim=0),
        "mu_f": Constant(1.1, dim=0),
        "rho_s0_tilde": Constant(1.0, dim=0),
        "phi": Constant(0.6, dim=0),
        "dt": Constant(0.1, dim=0),
        "theta": Constant(0.5, dim=0),
        "c_nh": Constant(0.7, dim=0),
        "beta_nh": Constant(0.0, dim=0),
        # permeability K^{-1} in reference domain
        "K_inv_I": Identity(2),
        "K_inv_A": Constant([[2.0, 0.3], [0.1, 1.5]], dim=2),
        "normal": FacetNormal(),
        "mesh": mesh_q2,
    }

    # --- dolfinx mesh / mixed space ---
    mesh_fx = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, dolfinx.mesh.CellType.quadrilateral)
    gdim = mesh_fx.geometry.dim
    P2_vec = basix.ufl.element("Lagrange", "quadrilateral", 2, shape=(gdim,))
    P1 = basix.ufl.element("Lagrange", "quadrilateral", 1)
    W_el = mixed_element([P2_vec, P2_vec, P1])
    W = dolfinx.fem.functionspace(mesh_fx, W_el)

    fenicsx = {
        "W": W,
        "mesh": mesh_fx,
        "rho_f": dolfinx.fem.Constant(mesh_fx, 0.9),
        "mu_f": dolfinx.fem.Constant(mesh_fx, 1.1),
        "rho_s0_tilde": dolfinx.fem.Constant(mesh_fx, 1.0),
        "phi": dolfinx.fem.Constant(mesh_fx, 0.6),
        "dt": dolfinx.fem.Constant(mesh_fx, 0.1),
        "theta": dolfinx.fem.Constant(mesh_fx, 0.5),
        "c_nh": dolfinx.fem.Constant(mesh_fx, 0.7),
        "beta_nh": dolfinx.fem.Constant(mesh_fx, 0.0),
        "normal": ufl.FacetNormal(mesh_fx),
        # states (mixed)
        "w_k": dolfinx.fem.Function(W, name="w_k"),
        "w_n": dolfinx.fem.Function(W, name="w_n"),
        "w_nm1": dolfinx.fem.Function(W, name="w_nm1"),
        # permeability tensors
        "K_inv_I": ufl.Identity(gdim),
        "K_inv_A": ufl.as_matrix(((2.0, 0.3), (0.1, 1.5))),
    }

    return pc, dof_handler_pc, fenicsx


def create_true_dof_map_poro(dof_handler_pc: DofHandler, W_fenicsx):
    print("=" * 70)
    print("Discovering poro DoF map (v,u,p) by matching DoF coordinates...")
    print("=" * 70)

    # fenics: W = (v,u,p)
    Wv, Vv_map = W_fenicsx.sub(0).collapse()
    Wu, Vu_map = W_fenicsx.sub(1).collapse()
    Wp, P_map_fx = W_fenicsx.sub(2).collapse()

    Wv0, Vv0_map = Wv.sub(0).collapse()
    Wv1, Vv1_map = Wv.sub(1).collapse()
    Wu0, Vu0_map = Wu.sub(0).collapse()
    Wu1, Vu1_map = Wu.sub(1).collapse()

    fx_coords = {
        "v_x": Wv0.tabulate_dof_coordinates()[:, :2],
        "v_y": Wv1.tabulate_dof_coordinates()[:, :2],
        "u_x": Wu0.tabulate_dof_coordinates()[:, :2],
        "u_y": Wu1.tabulate_dof_coordinates()[:, :2],
        "p": Wp.tabulate_dof_coordinates()[:, :2],
    }
    fx_dofs = {
        "v_x": np.array(Vv_map)[np.array(Vv0_map)],
        "v_y": np.array(Vv_map)[np.array(Vv1_map)],
        "u_x": np.array(Vu_map)[np.array(Vu0_map)],
        "u_y": np.array(Vu_map)[np.array(Vu1_map)],
        "p": np.array(P_map_fx),
    }

    pc_fields = ["v_x", "v_y", "u_x", "u_y", "p"]
    pc_coords = {f: get_pycutfem_dof_coords(dof_handler_pc, f) for f in pc_fields}
    pc_dofs = {f: dof_handler_pc.get_field_slice(f) for f in pc_fields}

    P = np.zeros(dof_handler_pc.total_dofs, dtype=int)

    # v_x and v_y share coordinates in vector Q2 space
    coord_map_v = one_to_one_map_coords(pc_coords["v_x"], fx_coords["v_x"])
    P[pc_dofs["v_x"]] = fx_dofs["v_x"][coord_map_v]
    P[pc_dofs["v_y"]] = fx_dofs["v_y"][coord_map_v]

    # u_x and u_y share coordinates in vector Q2 space
    coord_map_u = one_to_one_map_coords(pc_coords["u_x"], fx_coords["u_x"])
    P[pc_dofs["u_x"]] = fx_dofs["u_x"][coord_map_u]
    P[pc_dofs["u_y"]] = fx_dofs["u_y"][coord_map_u]

    # p is Q1
    coord_map_p = one_to_one_map_coords(pc_coords["p"], fx_coords["p"])
    P[pc_dofs["p"]] = fx_dofs["p"][coord_map_p]

    print("Poro DoF map discovered successfully.")
    return P


def initialize_poro_functions(pc, fenicsx, dof_handler_pc, P_map):
    print("Initializing poro (v,u,p) functions in pycutfem and FEniCSx...")

    def v_k_init(x):
        return [0.1 + 0.2 * x[0] + 0.05 * x[1], -0.03 + 0.1 * x[0] - 0.07 * x[1]]

    def u_k_init(x):
        # keep gradients small so F = inv(I - grad(u)) is well-defined
        return [0.02 * x[0] * (1.0 - x[0]) + 0.01 * x[1], -0.01 * x[1] * (1.0 - x[1]) + 0.005 * x[0]]

    def p_k_init(x):
        return 0.2 * x[0] - 0.1 * x[1]

    def v_n_init(x):
        vv = v_k_init(x)
        return [0.5 * vv[0], 0.5 * vv[1]]

    def u_n_init(x):
        uu = u_k_init(x)
        return [0.6 * uu[0], 0.6 * uu[1]]

    def u_nm1_init(x):
        uu = u_k_init(x)
        return [0.2 * uu[0], 0.2 * uu[1]]

    def p_n_init(x):
        return 0.5 * p_k_init(x)

    # --- pycutfem ---
    pc["v_k"].set_values_from_function(lambda x, y: v_k_init([x, y]))
    pc["u_k"].set_values_from_function(lambda x, y: u_k_init([x, y]))
    pc["p_k"].set_values_from_function(lambda x, y: p_k_init([x, y]))
    pc["v_n"].set_values_from_function(lambda x, y: v_n_init([x, y]))
    pc["u_n"].set_values_from_function(lambda x, y: u_n_init([x, y]))
    pc["u_nm1"].set_values_from_function(lambda x, y: u_nm1_init([x, y]))
    pc["p_n"].set_values_from_function(lambda x, y: p_n_init([x, y]))

    # --- FEniCSx ---
    W = fenicsx["W"]
    w_k = fenicsx["w_k"]
    w_n = fenicsx["w_n"]
    w_nm1 = fenicsx["w_nm1"]

    Vv, Vv_to_W = W.sub(0).collapse()
    Vu, Vu_to_W = W.sub(1).collapse()
    Q, Q_to_W = W.sub(2).collapse()

    v_k_view = w_k.sub(0)
    u_k_view = w_k.sub(1)
    p_k_view = w_k.sub(2)

    v_n_view = w_n.sub(0)
    u_n_view = w_n.sub(1)
    p_n_view = w_n.sub(2)

    u_nm1_view = w_nm1.sub(1)

    w_k.x.array[Vv_to_W] = v_k_view.debug_interpolate(v_k_init)
    w_k.x.array[Vu_to_W] = u_k_view.debug_interpolate(u_k_init)
    w_k.x.array[Q_to_W] = p_k_view.debug_interpolate(p_k_init)
    w_k.x.scatter_forward()

    w_n.x.array[Vv_to_W] = v_n_view.debug_interpolate(v_n_init)
    w_n.x.array[Vu_to_W] = u_n_view.debug_interpolate(u_n_init)
    w_n.x.array[Q_to_W] = p_n_view.debug_interpolate(p_n_init)
    w_n.x.scatter_forward()

    # only u_{n-1} is used; other subfields are irrelevant
    w_nm1.x.array[Vu_to_W] = u_nm1_view.debug_interpolate(u_nm1_init)
    w_nm1.x.scatter_forward()

    # sanity: check sets of nodal values agree for v_k (unordered)
    np.testing.assert_allclose(np.sort(pc["v_k"].nodal_values), np.sort(w_k.x.array[Vv_to_W]), rtol=1e-8, atol=1e-12)
    print("✅ Poro initialization: nodal value sets match (v_k).")


def _poro_k_inv_pc(u, *, K_inv):
    """k^{-1} = J F^{-T} K^{-1} F^{-1}, with F^{-1} = I - ∇u (Eulerian reference-map)."""
    F = inv(Identity(2) - grad(u))
    J = det(F)
    F_inv = Identity(2) - grad(u)
    return J * dot(F_inv.T, dot(K_inv, F_inv))


def _poro_k_inv_pc_iso_short(u):
    F = inv(Identity(2) - grad(u))
    J = det(F)
    F_inv = Identity(2) - grad(u)
    return J * dot(F_inv.T, F_inv)


def _poro_sigma_nh_pc(u, *, c, beta):
    F = inv(Identity(2) - grad(u))
    J = det(F)
    B = dot(F, F.T)
    a = J ** (-Constant(2.0, dim=0) * beta)
    return (Constant(2.0, dim=0) * c / J) * (B - a * Identity(2))


def _poro_dk_inv_pc(u, du, *, K_inv):
    """Gateaux derivative of k^{-1} = J F^{-T} K^{-1} F^{-1} in direction du."""
    F_inv = Identity(2) - grad(u)
    F = inv(F_inv)
    J = det(F)

    dF = dot(F, dot(grad(du), F))
    dJ = J * trace(dot(F_inv, dF))

    dF_inv = -grad(du)
    dF_inv_T = dF_inv.T

    base = dot(F_inv.T, dot(K_inv, F_inv))
    return dJ * base + J * dot(dF_inv_T, dot(K_inv, F_inv)) + J * dot(F_inv.T, dot(K_inv, dF_inv))


def _poro_dk_inv_pc_iso_short(u, du):
    """Gateaux derivative of k^{-1} = J F^{-T} F^{-1} (avoids Dot(Identity,·))."""
    F_inv = Identity(2) - grad(u)
    F = inv(F_inv)
    J = det(F)

    dF = dot(F, dot(grad(du), F))
    dJ = J * trace(dot(F_inv, dF))

    dF_inv = -grad(du)
    dF_inv_T = dF_inv.T

    base = dot(F_inv.T, F_inv)
    return dJ * base + J * dot(dF_inv_T, F_inv) + J * dot(F_inv.T, dF_inv)


def _poro_dsigma_nh_pc(u, du, *, c, beta):
    """Gateaux derivative of Neo-Hookean Cauchy stress in direction du."""
    F_inv = Identity(2) - grad(u)
    F = inv(F_inv)
    dF = dot(F, dot(grad(du), F))

    J = det(F)
    dJ = J * trace(dot(F_inv, dF))

    I2 = Identity(2)
    a = J ** (-Constant(2.0, dim=0) * beta)
    da = -(Constant(2.0, dim=0) * beta) * a * (dJ / J)

    B = dot(F, F.T)
    dB = dot(dF, F.T) + dot(F, dF.T)

    return Constant(2.0, dim=0) * c * (-(dJ / (J * J)) * (B - a * I2) + (Constant(1.0, dim=0) / J) * (dB - da * I2))


def run_poro_comparison():
    pc, dof_handler_pc, fenicsx = setup_poro_problems()
    W_fx = fenicsx["W"]
    P_map = create_true_dof_map_poro(dof_handler_pc, W_fx)
    initialize_poro_functions(pc, fenicsx, dof_handler_pc, P_map)

    # Split mixed functions and arguments on the FEniCSx side
    dv_fx, du_fx, dp_fx = ufl.split(ufl.TrialFunction(W_fx))
    w_fx, eta_fx, q_fx = ufl.split(ufl.TestFunction(W_fx))

    v_k_fx, u_k_fx, p_k_fx = ufl.split(fenicsx["w_k"])
    v_n_fx, u_n_fx, p_n_fx = ufl.split(fenicsx["w_n"])
    _v_nm1_fx, u_nm1_fx, _p_nm1_fx = ufl.split(fenicsx["w_nm1"])

    # parameters
    rho_f_pc = pc["rho_f"]
    mu_f_pc = pc["mu_f"]
    rho_s_pc = pc["rho_s0_tilde"]
    phi_pc = pc["phi"]
    dt_pc = pc["dt"]
    theta_pc = pc["theta"]
    c_nh_pc = pc["c_nh"]
    beta_nh_pc = pc["beta_nh"]

    rho_f_fx = fenicsx["rho_f"]
    mu_f_fx = fenicsx["mu_f"]
    rho_s_fx = fenicsx["rho_s0_tilde"]
    phi_fx = fenicsx["phi"]
    dt_fx = fenicsx["dt"]
    theta_fx = fenicsx["theta"]
    c_nh_fx = fenicsx["c_nh"]
    beta_nh_fx = fenicsx["beta_nh"]

    # --- Helper kinematics (pc) ---
    v_s_k_pc = (pc["u_k"] - pc["u_n"]) / dt_pc
    v_s_n_pc = (pc["u_n"] - pc["u_nm1"]) / dt_pc

    div_v_s_k_pc = (div(pc["u_k"]) - div(pc["u_n"])) / dt_pc
    div_v_s_n_pc = (div(pc["u_n"]) - div(pc["u_nm1"])) / dt_pc
    grad_v_s_k_pc = (grad(pc["u_k"]) - grad(pc["u_n"])) / dt_pc
    grad_v_s_n_pc = (grad(pc["u_n"]) - grad(pc["u_nm1"])) / dt_pc

    # --- Helper kinematics (fx) ---
    v_s_k_fx = (u_k_fx - u_n_fx) / dt_fx
    v_s_n_fx = (u_n_fx - u_nm1_fx) / dt_fx

    # Permeability variants
    K_inv_I_pc = pc["K_inv_I"]
    K_inv_A_pc = pc["K_inv_A"]
    K_inv_I_fx = fenicsx["K_inv_I"]
    K_inv_A_fx = fenicsx["K_inv_A"]

    # Measures
    qdeg = int(os.environ.get("COMP_FENICS_QDEG", "6"))
    dx_pc = dx(metadata={"q": qdeg})
    dx_fx = ufl.dx(metadata={"quadrature_degree": qdeg})

    # --- Residual subterms (pc) ---
    p_theta_pc = theta_pc * pc["p_k"] + (Constant(1.0, dim=0) - theta_pc) * pc["p_n"]
    div_mix_k_pc = phi_pc * div(pc["v_k"]) + (Constant(1.0, dim=0) - phi_pc) * div_v_s_k_pc
    div_mix_n_pc = phi_pc * div(pc["v_n"]) + (Constant(1.0, dim=0) - phi_pc) * div_v_s_n_pc
    r_mass_pc = pc["q"] * (theta_pc * div_mix_k_pc + (Constant(1.0, dim=0) - theta_pc) * div_mix_n_pc) * dx_pc

    vdot_pc = (pc["v_k"] - pc["v_n"]) / dt_pc
    r_v_inertia_pc = inner(rho_f_pc * vdot_pc, pc["w"]) * dx_pc
    r_v_pres_pc = -inner(p_theta_pc, div(pc["w"])) * dx_pc

    conv_k_pc = -rho_f_pc * dot(grad(pc["v_k"]), v_s_k_pc)
    conv_n_pc = -rho_f_pc * dot(grad(pc["v_n"]), v_s_n_pc)
    r_v_conv_pc = inner(theta_pc * conv_k_pc + (Constant(1.0, dim=0) - theta_pc) * conv_n_pc, pc["w"]) * dx_pc

    # drag: isotropic shortcut
    k_inv_iso_pc = _poro_k_inv_pc_iso_short(pc["u_k"])
    drag_k_iso_pc = mu_f_pc * (phi_pc * phi_pc) * dot(k_inv_iso_pc, (pc["v_k"] - v_s_k_pc))
    drag_n_iso_pc = mu_f_pc * (phi_pc * phi_pc) * dot(_poro_k_inv_pc_iso_short(pc["u_n"]), (pc["v_n"] - v_s_n_pc))
    drag_theta_iso_pc = theta_pc * drag_k_iso_pc + (Constant(1.0, dim=0) - theta_pc) * drag_n_iso_pc
    r_v_drag_iso_pc = inner(drag_theta_iso_pc, pc["w"]) * dx_pc

    # drag: isotropic but force Dot(Identity,·) path in k^{-1}
    k_inv_iso_full_pc = _poro_k_inv_pc(pc["u_k"], K_inv=K_inv_I_pc)
    drag_k_iso_full_pc = mu_f_pc * (phi_pc * phi_pc) * dot(k_inv_iso_full_pc, (pc["v_k"] - v_s_k_pc))
    r_v_drag_iso_full_pc = inner(theta_pc * drag_k_iso_full_pc, pc["w"]) * dx_pc

    # drag: anisotropic
    k_inv_A_pc = _poro_k_inv_pc(pc["u_k"], K_inv=K_inv_A_pc)
    drag_k_A_pc = mu_f_pc * (phi_pc * phi_pc) * dot(k_inv_A_pc, (pc["v_k"] - v_s_k_pc))
    r_v_drag_A_pc = inner(theta_pc * drag_k_A_pc, pc["w"]) * dx_pc

    # skeleton acceleration
    acc_local_pc = (v_s_k_pc - v_s_n_pc) / dt_pc
    adv_k_pc = dot(grad_v_s_k_pc, v_s_k_pc)
    adv_n_pc = dot(grad_v_s_n_pc, v_s_n_pc)
    acc_pc = acc_local_pc + theta_pc * adv_k_pc + (Constant(1.0, dim=0) - theta_pc) * adv_n_pc
    r_u_acc_pc = inner(rho_s_pc * acc_pc, pc["eta"]) * dx_pc

    # skeleton stress
    sig_k_pc = _poro_sigma_nh_pc(pc["u_k"], c=c_nh_pc, beta=beta_nh_pc)
    sig_n_pc = _poro_sigma_nh_pc(pc["u_n"], c=c_nh_pc, beta=beta_nh_pc)
    sig_theta_pc = theta_pc * sig_k_pc + (Constant(1.0, dim=0) - theta_pc) * sig_n_pc
    r_u_sig_pc = inner(sig_theta_pc, grad(pc["eta"])) * dx_pc

    r_u_pres_pc = inner(phi_pc * p_theta_pc, div(pc["eta"])) * dx_pc
    r_u_drag_iso_pc = -inner(drag_theta_iso_pc, pc["eta"]) * dx_pc

    r_total_pc = (
        r_mass_pc
        + r_v_inertia_pc
        + r_v_pres_pc
        + r_v_conv_pc
        + r_v_drag_iso_pc
        + r_u_acc_pc
        + r_u_sig_pc
        + r_u_pres_pc
        + r_u_drag_iso_pc
    )

    # --- Residual subterms (fenics) ---
    I2_fx = ufl.Identity(2)
    p_theta_fx = theta_fx * p_k_fx + (1.0 - theta_fx) * p_n_fx

    div_mix_k_fx = phi_fx * ufl.div(v_k_fx) + (1.0 - phi_fx) * ufl.div(v_s_k_fx)
    div_mix_n_fx = phi_fx * ufl.div(v_n_fx) + (1.0 - phi_fx) * ufl.div(v_s_n_fx)
    r_mass_fx = (q_fx * (theta_fx * div_mix_k_fx + (1.0 - theta_fx) * div_mix_n_fx)) * dx_fx

    vdot_fx = (v_k_fx - v_n_fx) / dt_fx
    r_v_inertia_fx = ufl.inner(rho_f_fx * vdot_fx, w_fx) * dx_fx
    r_v_pres_fx = -ufl.inner(p_theta_fx, ufl.div(w_fx)) * dx_fx

    conv_k_fx = -rho_f_fx * ufl.dot(ufl.grad(v_k_fx), v_s_k_fx)
    conv_n_fx = -rho_f_fx * ufl.dot(ufl.grad(v_n_fx), v_s_n_fx)
    r_v_conv_fx = ufl.inner(theta_fx * conv_k_fx + (1.0 - theta_fx) * conv_n_fx, w_fx) * dx_fx

    Fk_fx = ufl.inv(I2_fx - ufl.grad(u_k_fx))
    Jk_fx = ufl.det(Fk_fx)
    Finv_k_fx = I2_fx - ufl.grad(u_k_fx)
    k_inv_iso_short_fx = Jk_fx * ufl.dot(Finv_k_fx.T, Finv_k_fx)
    k_inv_iso_full_fx = Jk_fx * ufl.dot(Finv_k_fx.T, ufl.dot(K_inv_I_fx, Finv_k_fx))
    k_inv_A_fx = Jk_fx * ufl.dot(Finv_k_fx.T, ufl.dot(K_inv_A_fx, Finv_k_fx))

    drag_k_iso_fx = mu_f_fx * (phi_fx * phi_fx) * ufl.dot(k_inv_iso_short_fx, (v_k_fx - v_s_k_fx))
    Finv_n_fx = I2_fx - ufl.grad(u_n_fx)
    Fn_k_fx = ufl.inv(Finv_n_fx)
    Jn_k_fx = ufl.det(Fn_k_fx)
    k_inv_iso_short_n_fx = Jn_k_fx * ufl.dot(Finv_n_fx.T, Finv_n_fx)
    drag_n_iso_fx = mu_f_fx * (phi_fx * phi_fx) * ufl.dot(k_inv_iso_short_n_fx, (v_n_fx - v_s_n_fx))
    drag_theta_iso_fx = theta_fx * drag_k_iso_fx + (1.0 - theta_fx) * drag_n_iso_fx
    r_v_drag_iso_fx = ufl.inner(drag_theta_iso_fx, w_fx) * dx_fx

    drag_k_iso_full_fx = mu_f_fx * (phi_fx * phi_fx) * ufl.dot(k_inv_iso_full_fx, (v_k_fx - v_s_k_fx))
    r_v_drag_iso_full_fx = ufl.inner(theta_fx * drag_k_iso_full_fx, w_fx) * dx_fx

    drag_k_A_fx = mu_f_fx * (phi_fx * phi_fx) * ufl.dot(k_inv_A_fx, (v_k_fx - v_s_k_fx))
    r_v_drag_A_fx = ufl.inner(theta_fx * drag_k_A_fx, w_fx) * dx_fx

    acc_local_fx = (v_s_k_fx - v_s_n_fx) / dt_fx
    adv_k_fx = ufl.dot(ufl.grad(v_s_k_fx), v_s_k_fx)
    adv_n_fx = ufl.dot(ufl.grad(v_s_n_fx), v_s_n_fx)
    acc_fx = acc_local_fx + theta_fx * adv_k_fx + (1.0 - theta_fx) * adv_n_fx
    r_u_acc_fx = ufl.inner(rho_s_fx * acc_fx, eta_fx) * dx_fx

    Bk_fx = ufl.dot(Fk_fx, Fk_fx.T)
    a_fx = Jk_fx ** (-2.0 * beta_nh_fx)
    sig_k_fx = (2.0 * c_nh_fx / Jk_fx) * (Bk_fx - a_fx * I2_fx)
    Fn_fx = ufl.inv(I2_fx - ufl.grad(u_n_fx))
    Jn_fx = ufl.det(Fn_fx)
    Bn_fx = ufl.dot(Fn_fx, Fn_fx.T)
    a_n_fx = Jn_fx ** (-2.0 * beta_nh_fx)
    sig_n_fx = (2.0 * c_nh_fx / Jn_fx) * (Bn_fx - a_n_fx * I2_fx)
    sig_theta_fx = theta_fx * sig_k_fx + (1.0 - theta_fx) * sig_n_fx
    r_u_sig_fx = ufl.inner(sig_theta_fx, ufl.grad(eta_fx)) * dx_fx

    r_u_pres_fx = ufl.inner(phi_fx * p_theta_fx, ufl.div(eta_fx)) * dx_fx
    r_u_drag_iso_fx = -ufl.inner(drag_theta_iso_fx, eta_fx) * dx_fx

    r_total_fx = (
        r_mass_fx
        + r_v_inertia_fx
        + r_v_pres_fx
        + r_v_conv_fx
        + r_v_drag_iso_fx
        + r_u_acc_fx
        + r_u_sig_fx
        + r_u_pres_fx
        + r_u_drag_iso_fx
    )

    # --- Jacobian subterms (pc) ---
    # mixture divergence Jacobian
    div_dv_s_pc = div(pc["du"]) / dt_pc
    a_mass_pc = pc["q"] * (theta_pc * (phi_pc * div(pc["dv"]) + (Constant(1.0, dim=0) - phi_pc) * div_dv_s_pc)) * dx_pc

    a_v_inertia_pc = inner(rho_f_pc * (pc["dv"] / dt_pc), pc["w"]) * dx_pc
    a_v_pres_pc = -inner(theta_pc * pc["dp"], div(pc["w"])) * dx_pc

    dv_s_pc = pc["du"] / dt_pc
    grad_dv_s_pc = grad(pc["du"]) / dt_pc
    a_v_conv_v_pc = inner(theta_pc * (-rho_f_pc * dot(grad(pc["dv"]), v_s_k_pc)), pc["w"]) * dx_pc
    a_v_conv_u_pc = inner(theta_pc * (-rho_f_pc * dot(grad(pc["v_k"]), dv_s_pc)), pc["w"]) * dx_pc

    # drag Jacobian (isotropic shortcut) split: freeze + dk^{-1}
    dk_inv_iso_pc = _poro_dk_inv_pc_iso_short(pc["u_k"], pc["du"])
    ddrag_iso_freeze_pc = mu_f_pc * (phi_pc * phi_pc) * dot(k_inv_iso_pc, (pc["dv"] - dv_s_pc))
    ddrag_iso_dk_pc = mu_f_pc * (phi_pc * phi_pc) * dot(dk_inv_iso_pc, (pc["v_k"] - v_s_k_pc))
    a_v_drag_iso_freeze_pc = inner(theta_pc * ddrag_iso_freeze_pc, pc["w"]) * dx_pc
    a_v_drag_iso_dk_pc = inner(theta_pc * ddrag_iso_dk_pc, pc["w"]) * dx_pc
    a_v_drag_iso_full_pc = a_v_drag_iso_freeze_pc + a_v_drag_iso_dk_pc

    a_u_drag_iso_freeze_pc = -inner(theta_pc * ddrag_iso_freeze_pc, pc["eta"]) * dx_pc
    a_u_drag_iso_dk_pc = -inner(theta_pc * ddrag_iso_dk_pc, pc["eta"]) * dx_pc
    a_u_drag_iso_full_pc = a_u_drag_iso_freeze_pc + a_u_drag_iso_dk_pc

    # Optional drag Jacobians for debugging: full-I and anisotropic K^{-1}
    dk_inv_iso_full_pc = _poro_dk_inv_pc(pc["u_k"], pc["du"], K_inv=K_inv_I_pc)
    ddrag_iso_full_freeze_pc = mu_f_pc * (phi_pc * phi_pc) * dot(k_inv_iso_full_pc, (pc["dv"] - dv_s_pc))
    ddrag_iso_full_dk_pc = mu_f_pc * (phi_pc * phi_pc) * dot(dk_inv_iso_full_pc, (pc["v_k"] - v_s_k_pc))
    a_v_drag_iso_fullI_freeze_pc = inner(theta_pc * ddrag_iso_full_freeze_pc, pc["w"]) * dx_pc
    a_v_drag_iso_fullI_dk_pc = inner(theta_pc * ddrag_iso_full_dk_pc, pc["w"]) * dx_pc
    a_v_drag_iso_fullI_full_pc = a_v_drag_iso_fullI_freeze_pc + a_v_drag_iso_fullI_dk_pc

    dk_inv_A_pc = _poro_dk_inv_pc(pc["u_k"], pc["du"], K_inv=K_inv_A_pc)
    ddrag_A_freeze_pc = mu_f_pc * (phi_pc * phi_pc) * dot(k_inv_A_pc, (pc["dv"] - dv_s_pc))
    ddrag_A_dk_pc = mu_f_pc * (phi_pc * phi_pc) * dot(dk_inv_A_pc, (pc["v_k"] - v_s_k_pc))
    a_v_drag_A_freeze_pc = inner(theta_pc * ddrag_A_freeze_pc, pc["w"]) * dx_pc
    a_v_drag_A_dk_pc = inner(theta_pc * ddrag_A_dk_pc, pc["w"]) * dx_pc
    a_v_drag_A_full_pc = a_v_drag_A_freeze_pc + a_v_drag_A_dk_pc

    # skeleton acceleration Jacobian: local + advective (Eulerian material acceleration)
    a_u_acc_local_pc = inner(rho_s_pc * (dv_s_pc / dt_pc), pc["eta"]) * dx_pc
    a_u_acc_adv1_pc = inner(rho_s_pc * theta_pc * dot(grad_dv_s_pc, v_s_k_pc), pc["eta"]) * dx_pc
    a_u_acc_adv2_pc = inner(rho_s_pc * theta_pc * dot(grad_v_s_k_pc, dv_s_pc), pc["eta"]) * dx_pc
    a_u_acc_full_pc = a_u_acc_local_pc + a_u_acc_adv1_pc + a_u_acc_adv2_pc

    # skeleton stress Jacobian
    dsig_k_pc = _poro_dsigma_nh_pc(pc["u_k"], pc["du"], c=c_nh_pc, beta=beta_nh_pc)
    a_u_sig_pc = inner(theta_pc * dsig_k_pc, grad(pc["eta"])) * dx_pc

    a_u_pres_pc = inner(phi_pc * theta_pc * pc["dp"], div(pc["eta"])) * dx_pc

    a_total_pc = (
        a_mass_pc
        + a_v_inertia_pc
        + a_v_pres_pc
        + a_v_conv_v_pc
        + a_v_conv_u_pc
        + a_v_drag_iso_full_pc
        + a_u_acc_full_pc
        + a_u_sig_pc
        + a_u_pres_pc
        + a_u_drag_iso_full_pc
    )

    # --- Jacobian subterms (fenics) ---
    dv_s_fx = du_fx / dt_fx
    grad_dv_s_fx = ufl.grad(du_fx) / dt_fx
    grad_v_s_k_fx = (ufl.grad(u_k_fx) - ufl.grad(u_n_fx)) / dt_fx

    a_mass_fx = ufl.derivative(r_mass_fx, fenicsx["w_k"], ufl.TrialFunction(W_fx))
    a_v_inertia_fx = ufl.derivative(r_v_inertia_fx, fenicsx["w_k"], ufl.TrialFunction(W_fx))
    a_v_pres_fx = ufl.derivative(r_v_pres_fx, fenicsx["w_k"], ufl.TrialFunction(W_fx))

    # convection split (equals ufl.derivative(r_v_conv_fx, ...))
    a_v_conv_v_fx = ufl.inner(theta_fx * (-rho_f_fx * ufl.dot(ufl.grad(dv_fx), v_s_k_fx)), w_fx) * dx_fx
    a_v_conv_u_fx = ufl.inner(theta_fx * (-rho_f_fx * ufl.dot(ufl.grad(v_k_fx), dv_s_fx)), w_fx) * dx_fx
    a_v_conv_fx = a_v_conv_v_fx + a_v_conv_u_fx

    # drag (iso-short) split: freeze + dk^{-1}
    a_v_drag_iso_full_fx = ufl.derivative(r_v_drag_iso_fx, fenicsx["w_k"], ufl.TrialFunction(W_fx))
    a_v_drag_iso_freeze_fx = ufl.inner(
        theta_fx * (mu_f_fx * (phi_fx * phi_fx) * ufl.dot(k_inv_iso_short_fx, (dv_fx - dv_s_fx))),
        w_fx,
    ) * dx_fx
    a_v_drag_iso_dk_fx = a_v_drag_iso_full_fx - a_v_drag_iso_freeze_fx

    # drag (full-I) split (diagnostic for Dot(Identity,·) path)
    a_v_drag_iso_fullI_full_fx = ufl.derivative(r_v_drag_iso_full_fx, fenicsx["w_k"], ufl.TrialFunction(W_fx))
    a_v_drag_iso_fullI_freeze_fx = ufl.inner(
        theta_fx * (mu_f_fx * (phi_fx * phi_fx) * ufl.dot(k_inv_iso_full_fx, (dv_fx - dv_s_fx))),
        w_fx,
    ) * dx_fx
    a_v_drag_iso_fullI_dk_fx = a_v_drag_iso_fullI_full_fx - a_v_drag_iso_fullI_freeze_fx

    # drag (anisotropic) split
    a_v_drag_A_full_fx = ufl.derivative(r_v_drag_A_fx, fenicsx["w_k"], ufl.TrialFunction(W_fx))
    a_v_drag_A_freeze_fx = ufl.inner(
        theta_fx * (mu_f_fx * (phi_fx * phi_fx) * ufl.dot(k_inv_A_fx, (dv_fx - dv_s_fx))),
        w_fx,
    ) * dx_fx
    a_v_drag_A_dk_fx = a_v_drag_A_full_fx - a_v_drag_A_freeze_fx

    # skeleton acceleration split
    a_u_acc_full_fx = ufl.derivative(r_u_acc_fx, fenicsx["w_k"], ufl.TrialFunction(W_fx))
    a_u_acc_local_fx = ufl.inner(rho_s_fx * (dv_s_fx / dt_fx), eta_fx) * dx_fx
    a_u_acc_adv1_fx = ufl.inner(rho_s_fx * theta_fx * ufl.dot(grad_dv_s_fx, v_s_k_fx), eta_fx) * dx_fx
    a_u_acc_adv2_fx = ufl.inner(rho_s_fx * theta_fx * ufl.dot(grad_v_s_k_fx, dv_s_fx), eta_fx) * dx_fx

    a_u_sig_fx = ufl.derivative(r_u_sig_fx, fenicsx["w_k"], ufl.TrialFunction(W_fx))
    a_u_pres_fx = ufl.derivative(r_u_pres_fx, fenicsx["w_k"], ufl.TrialFunction(W_fx))

    # skeleton drag reaction (iso-short) split
    a_u_drag_iso_full_fx = ufl.derivative(r_u_drag_iso_fx, fenicsx["w_k"], ufl.TrialFunction(W_fx))
    a_u_drag_iso_freeze_fx = -ufl.inner(
        theta_fx * (mu_f_fx * (phi_fx * phi_fx) * ufl.dot(k_inv_iso_short_fx, (dv_fx - dv_s_fx))),
        eta_fx,
    ) * dx_fx
    a_u_drag_iso_dk_fx = a_u_drag_iso_full_fx - a_u_drag_iso_freeze_fx

    a_total_fx = ufl.derivative(r_total_fx, fenicsx["w_k"], ufl.TrialFunction(W_fx))

    # --- Term table ---
    terms = {
        # Residual pieces
        "Poro mass div_mix (res)": {"pc": r_mass_pc, "fx": r_mass_fx, "mat": False},
        "Poro pore inertia (res)": {"pc": r_v_inertia_pc, "fx": r_v_inertia_fx, "mat": False},
        "Poro pore pressure (res)": {"pc": r_v_pres_pc, "fx": r_v_pres_fx, "mat": False},
        "Poro pore convection (res)": {"pc": r_v_conv_pc, "fx": r_v_conv_fx, "mat": False},
        "Poro pore drag iso (res)": {"pc": r_v_drag_iso_pc, "fx": r_v_drag_iso_fx, "mat": False},
        "Poro pore drag iso fullI (res)": {"pc": r_v_drag_iso_full_pc, "fx": r_v_drag_iso_full_fx, "mat": False},
        "Poro pore drag aniso (res)": {"pc": r_v_drag_A_pc, "fx": r_v_drag_A_fx, "mat": False},
        "Poro skeleton accel (res)": {"pc": r_u_acc_pc, "fx": r_u_acc_fx, "mat": False},
        "Poro skeleton stress (res)": {"pc": r_u_sig_pc, "fx": r_u_sig_fx, "mat": False},
        "Poro skeleton pressure (res)": {"pc": r_u_pres_pc, "fx": r_u_pres_fx, "mat": False},
        "Poro skeleton drag reaction (res)": {"pc": r_u_drag_iso_pc, "fx": r_u_drag_iso_fx, "mat": False},
        "Poro total residual": {"pc": r_total_pc, "fx": r_total_fx, "mat": False},
        # Jacobian pieces (split to isolate implementation issues)
        "Poro mass (jac)": {"pc": a_mass_pc, "fx": a_mass_fx, "mat": True},
        "Poro pore inertia (jac)": {"pc": a_v_inertia_pc, "fx": a_v_inertia_fx, "mat": True},
        "Poro pore pressure (jac)": {"pc": a_v_pres_pc, "fx": a_v_pres_fx, "mat": True},
        "Poro pore convection dv (jac)": {"pc": a_v_conv_v_pc, "fx": a_v_conv_v_fx, "mat": True},
        "Poro pore convection du (jac)": {"pc": a_v_conv_u_pc, "fx": a_v_conv_u_fx, "mat": True},
        "Poro pore convection (jac)": {"pc": a_v_conv_v_pc + a_v_conv_u_pc, "fx": a_v_conv_fx, "mat": True},
        "Poro pore drag iso freeze (jac)": {"pc": a_v_drag_iso_freeze_pc, "fx": a_v_drag_iso_freeze_fx, "mat": True},
        "Poro pore drag iso dk_inv (jac)": {"pc": a_v_drag_iso_dk_pc, "fx": a_v_drag_iso_dk_fx, "mat": True},
        "Poro pore drag iso (jac)": {"pc": a_v_drag_iso_full_pc, "fx": a_v_drag_iso_full_fx, "mat": True},
        "Poro pore drag iso fullI freeze (jac)": {"pc": a_v_drag_iso_fullI_freeze_pc, "fx": a_v_drag_iso_fullI_freeze_fx, "mat": True},
        "Poro pore drag iso fullI dk_inv (jac)": {"pc": a_v_drag_iso_fullI_dk_pc, "fx": a_v_drag_iso_fullI_dk_fx, "mat": True},
        "Poro pore drag iso fullI (jac)": {"pc": a_v_drag_iso_fullI_full_pc, "fx": a_v_drag_iso_fullI_full_fx, "mat": True},
        "Poro pore drag aniso freeze (jac)": {"pc": a_v_drag_A_freeze_pc, "fx": a_v_drag_A_freeze_fx, "mat": True},
        "Poro pore drag aniso dk_inv (jac)": {"pc": a_v_drag_A_dk_pc, "fx": a_v_drag_A_dk_fx, "mat": True},
        "Poro pore drag aniso (jac)": {"pc": a_v_drag_A_full_pc, "fx": a_v_drag_A_full_fx, "mat": True},
        "Poro skeleton accel local (jac)": {"pc": a_u_acc_local_pc, "fx": a_u_acc_local_fx, "mat": True},
        "Poro skeleton accel adv1 (jac)": {"pc": a_u_acc_adv1_pc, "fx": a_u_acc_adv1_fx, "mat": True},
        "Poro skeleton accel adv2 (jac)": {"pc": a_u_acc_adv2_pc, "fx": a_u_acc_adv2_fx, "mat": True},
        "Poro skeleton accel (jac)": {"pc": a_u_acc_full_pc, "fx": a_u_acc_full_fx, "mat": True},
        "Poro skeleton stress (jac)": {"pc": a_u_sig_pc, "fx": a_u_sig_fx, "mat": True},
        "Poro skeleton pressure (jac)": {"pc": a_u_pres_pc, "fx": a_u_pres_fx, "mat": True},
        "Poro skeleton drag iso freeze (jac)": {"pc": a_u_drag_iso_freeze_pc, "fx": a_u_drag_iso_freeze_fx, "mat": True},
        "Poro skeleton drag iso dk_inv (jac)": {"pc": a_u_drag_iso_dk_pc, "fx": a_u_drag_iso_dk_fx, "mat": True},
        "Poro skeleton drag iso (jac)": {"pc": a_u_drag_iso_full_pc, "fx": a_u_drag_iso_full_fx, "mat": True},
        "Poro total jacobian (full)": {"pc": a_total_pc, "fx": a_total_fx, "mat": True},
    }

    # Filter terms: never run everything by default (keeps this debug script fast)
    run_all = os.environ.get("COMP_FENICS_RUN_ALL", "").lower() in {"1", "true", "yes"}
    filter_terms = os.environ.get("COMP_FENICS_TERMS")
    if filter_terms:
        allowed = {name.strip() for name in filter_terms.split(",") if name.strip()}
        terms = {k: v for k, v in terms.items() if k in allowed}
        print(f"Running filtered terms only: {sorted(terms)}")
    elif not run_all:
        default = {
            "Poro mass div_mix (res)",
            "Poro pore convection (res)",
            "Poro pore drag iso (res)",
            "Poro pore drag iso fullI (res)",
            "Poro pore drag aniso (res)",
            "Poro mass (jac)",
            "Poro pore convection (jac)",
            "Poro pore drag iso (jac)",
        }
        terms = {k: v for k, v in terms.items() if k in default}
        print(f"COMP_FENICS_TERMS not set; running safe default subset: {sorted(terms)}")

    backends_spec = os.environ.get("BACKEND", "jit")
    if backends_spec.strip().lower() == "all":
        backends = ["python", "jit", "cpp"]
    else:
        backends = [b.strip() for b in backends_spec.split(",") if b.strip()]
    parity_rtol = float(os.environ.get("COMP_FENICS_PARITY_RTOL", "1e-9"))
    parity_atol = float(os.environ.get("COMP_FENICS_PARITY_ATOL", "1e-9"))

    for backend_type in backends:
        print("\n" + "=" * 70)
        print(f"PORO COMPARISON (backend={backend_type})")
        print("=" * 70)

        failed_tests = []
        success_count = 0

        for name, forms in terms.items():
            J_pc, R_pc, J_fx, R_fx = None, None, None, None

            print(f"\nCompiling/assembling '{name}' [backend={backend_type}, qdeg={qdeg}]")
            try:
                form_fx_compiled = dolfinx.fem.form(forms["fx"])
            except Exception as exc:
                print(f"❌ FEniCSx form compilation failed for '{name}': {exc}")
                failed_tests.append(f"{name} (fenics-compile)")
                continue

            try:
                if forms["mat"]:
                    J_pc, _ = assemble_form(Equation(forms["pc"], None), dof_handler_pc, quad_degree=qdeg, bcs=[], backend=backend_type)
                    A = dolfinx.fem.petsc.assemble_matrix(form_fx_compiled)
                    A.assemble()
                    indptr, indices, data = A.getValuesCSR()
                    J_fx = csr_matrix((data, indices, indptr), shape=A.getSize()).toarray()
                else:
                    _, R_pc = assemble_form(Equation(None, forms["pc"]), dof_handler_pc, bcs=[], backend=backend_type)
                    vec = dolfinx.fem.petsc.assemble_vector(form_fx_compiled)
                    R_fx = vec.array
            except Exception as exc:
                print(f"❌ Assembly failed for '{name}' on backend '{backend_type}': {exc}")
                failed_tests.append(f"{name} (assemble-{backend_type})")
                continue

            is_success = compare_term(
                f"{name} [backend={backend_type}]",
                J_pc,
                R_pc,
                J_fx,
                R_fx,
                P_map,
                dof_handler_pc,
                W_fx,
                rtol=1e-8,
                atol=1e-8,
            )
            if is_success:
                success_count += 1
            else:
                failed_tests.append(name)

        print_test_summary(success_count, failed_tests)


# ==============================================================================
#  Cahn–Hilliard (alpha, mu_alpha) comparison harness vs FEniCSx
# ==============================================================================


def setup_alpha_ch_problems(*, nx: int = 2, ny: int = 2):
    """
    Build a small Cahn–Hilliard mixed space:
      pycutfem: (alpha, mu_alpha) as (Q1, Q1)
      dolfinx : (alpha, mu_alpha) as (P1, P1)
    """
    # --- pycutfem mesh (Q2 geometry so dof coordinates match dolfinx's quad layout) ---
    nodes_q2, elems_q2, _, corners_q2 = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=2)
    mesh_q2 = Mesh(
        nodes=nodes_q2,
        element_connectivity=elems_q2,
        elements_corner_nodes=corners_q2,
        element_type="quad",
        poly_order=2,
    )
    mesh_q2.tag_boundary_edges({"all": lambda x, y: True})

    mixed_element_pc = MixedElement(mesh_q2, field_specs={"alpha": 1, "mu_alpha": 1})
    dof_handler_pc = DofHandler(mixed_element_pc, method="cg")

    dt_val = float(os.environ.get("COMP_FENICS_DT", "0.1"))
    theta_val = float(os.environ.get("COMP_FENICS_THETA", "1.0"))
    M0_val = float(os.environ.get("COMP_FENICS_ALPHA_CH_M", "0.2"))
    gamma_val = float(os.environ.get("COMP_FENICS_ALPHA_CH_GAMMA", "1.0"))
    eps_val = float(os.environ.get("COMP_FENICS_ALPHA_CH_EPS", "0.1"))

    pc = {
        # trial/test
        "dalpha": TrialFunction("alpha", dof_handler=dof_handler_pc),
        "dmu": TrialFunction("mu_alpha", dof_handler=dof_handler_pc),
        "xi": TestFunction("alpha", dof_handler=dof_handler_pc),
        "eta": TestFunction("mu_alpha", dof_handler=dof_handler_pc),
        # states
        "alpha_k": Function(name="alpha_k", field_name="alpha", dof_handler=dof_handler_pc),
        "mu_k": Function(name="mu_k", field_name="mu_alpha", dof_handler=dof_handler_pc),
        "alpha_n": Function(name="alpha_n", field_name="alpha", dof_handler=dof_handler_pc),
        "mu_n": Function(name="mu_n", field_name="mu_alpha", dof_handler=dof_handler_pc),
        # parameters
        "dt": Constant(dt_val, dim=0),
        "theta": Constant(theta_val, dim=0),
        "one_m_theta": Constant(1.0 - theta_val, dim=0),
        "M0": Constant(M0_val, dim=0),
        "gamma": Constant(gamma_val, dim=0),
        "eps": Constant(eps_val, dim=0),
        "mobility": str(os.environ.get("COMP_FENICS_ALPHA_CH_MOBILITY", "constant")),
    }

    # --- dolfinx mesh / mixed space ---
    mesh_fx = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, dolfinx.mesh.CellType.quadrilateral)
    P1 = basix.ufl.element("Lagrange", "quadrilateral", 1)
    W_el = mixed_element([P1, P1])
    W = dolfinx.fem.functionspace(mesh_fx, W_el)

    fenicsx = {
        "W": W,
        "mesh": mesh_fx,
        "dt": dolfinx.fem.Constant(mesh_fx, dt_val),
        "theta": dolfinx.fem.Constant(mesh_fx, theta_val),
        "M0": dolfinx.fem.Constant(mesh_fx, M0_val),
        "gamma": dolfinx.fem.Constant(mesh_fx, gamma_val),
        "eps": dolfinx.fem.Constant(mesh_fx, eps_val),
        "mobility": str(os.environ.get("COMP_FENICS_ALPHA_CH_MOBILITY", "constant")),
        "w_k": dolfinx.fem.Function(W, name="w_k"),
        "w_n": dolfinx.fem.Function(W, name="w_n"),
    }

    return pc, dof_handler_pc, fenicsx


def create_true_dof_map_alpha_ch(dof_handler_pc: DofHandler, W_fenicsx):
    print("=" * 70)
    print("Discovering Cahn–Hilliard DoF map (alpha, mu_alpha) by matching DoF coordinates...")
    print("=" * 70)

    Walpha, Alpha_map_fx = W_fenicsx.sub(0).collapse()
    Wmu, Mu_map_fx = W_fenicsx.sub(1).collapse()

    fx_coords = {
        "alpha": Walpha.tabulate_dof_coordinates()[:, :2],
        "mu_alpha": Wmu.tabulate_dof_coordinates()[:, :2],
    }
    fx_dofs = {"alpha": np.array(Alpha_map_fx), "mu_alpha": np.array(Mu_map_fx)}

    pc_fields = ["alpha", "mu_alpha"]
    pc_coords = {f: get_pycutfem_dof_coords(dof_handler_pc, f) for f in pc_fields}
    pc_dofs = {f: dof_handler_pc.get_field_slice(f) for f in pc_fields}

    P = np.zeros(dof_handler_pc.total_dofs, dtype=int)
    coord_map_a = one_to_one_map_coords(pc_coords["alpha"], fx_coords["alpha"])
    coord_map_mu = one_to_one_map_coords(pc_coords["mu_alpha"], fx_coords["mu_alpha"])

    P[pc_dofs["alpha"]] = fx_dofs["alpha"][coord_map_a]
    P[pc_dofs["mu_alpha"]] = fx_dofs["mu_alpha"][coord_map_mu]

    print("True DoF map discovered successfully.")
    return P


def initialize_alpha_ch_functions(pc, fenicsx, dof_handler_pc, P_map):
    # Smooth, non-trivial initialization (keeps values in a reasonable range).
    def alpha_k_init(x):
        return 0.55 + 0.15 * np.sin(2.0 * np.pi * x[0]) * np.cos(2.0 * np.pi * x[1])

    def alpha_n_init(x):
        return 0.50 + 0.10 * np.sin(2.0 * np.pi * x[0]) * np.cos(2.0 * np.pi * x[1])

    def mu_k_init(x):
        return 0.20 * np.cos(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])

    def mu_n_init(x):
        return 0.10 * np.cos(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])

    pc["alpha_k"].set_values_from_function(lambda x, y: alpha_k_init(np.vstack([np.asarray(x), np.asarray(y)])))
    pc["alpha_n"].set_values_from_function(lambda x, y: alpha_n_init(np.vstack([np.asarray(x), np.asarray(y)])))
    pc["mu_k"].set_values_from_function(lambda x, y: mu_k_init(np.vstack([np.asarray(x), np.asarray(y)])))
    pc["mu_n"].set_values_from_function(lambda x, y: mu_n_init(np.vstack([np.asarray(x), np.asarray(y)])))

    W_fx = fenicsx["W"]
    w_k = fenicsx["w_k"]
    w_n = fenicsx["w_n"]

    Walpha, Alpha_to_W = W_fx.sub(0).collapse()
    Wmu, Mu_to_W = W_fx.sub(1).collapse()

    w_k.x.array[Alpha_to_W] = w_k.sub(0).debug_interpolate(alpha_k_init)
    w_k.x.array[Mu_to_W] = w_k.sub(1).debug_interpolate(mu_k_init)
    w_k.x.scatter_forward()

    w_n.x.array[Alpha_to_W] = w_n.sub(0).debug_interpolate(alpha_n_init)
    w_n.x.array[Mu_to_W] = w_n.sub(1).debug_interpolate(mu_n_init)
    w_n.x.scatter_forward()

    np.testing.assert_allclose(np.sort(pc["alpha_k"].nodal_values), np.sort(w_k.x.array[Alpha_to_W]), rtol=1e-8, atol=1e-12)
    print("✅ Cahn–Hilliard initialization: nodal value sets match (alpha_k).")


def run_alpha_ch_comparison():
    pc, dof_handler_pc, fenicsx = setup_alpha_ch_problems(
        nx=int(os.environ.get("COMP_FENICS_NX", "2")),
        ny=int(os.environ.get("COMP_FENICS_NY", os.environ.get("COMP_FENICS_NX", "2"))),
    )
    W_fx = fenicsx["W"]
    P_map = create_true_dof_map_alpha_ch(dof_handler_pc, W_fx)
    initialize_alpha_ch_functions(pc, fenicsx, dof_handler_pc, P_map)

    # Split mixed functions and arguments on the FEniCSx side
    dw_fx = ufl.TrialFunction(W_fx)
    w_test_fx = ufl.TestFunction(W_fx)
    dalpha_fx, dmu_fx = ufl.split(dw_fx)
    xi_fx, eta_fx = ufl.split(w_test_fx)

    alpha_k_fx, mu_k_fx = ufl.split(fenicsx["w_k"])
    alpha_n_fx, mu_n_fx = ufl.split(fenicsx["w_n"])

    # Parameters
    dt_fx = fenicsx["dt"]
    theta_fx = fenicsx["theta"]
    M0_fx = fenicsx["M0"]
    gamma_fx = fenicsx["gamma"]
    eps_fx = fenicsx["eps"]
    mob_key = str(fenicsx.get("mobility", "constant")).strip().lower()

    # Measures
    qdeg = int(os.environ.get("COMP_FENICS_QDEG", "6"))
    dx_pc = dx(metadata={"q": qdeg})
    dx_fx = ufl.dx(metadata={"quadrature_degree": qdeg})

    # Mobility M(α)
    if mob_key in {"constant", "const"}:
        M_k_fx = M0_fx
        M_n_fx = M0_fx
        M_k_pc = pc["M0"]
        M_n_pc = pc["M0"]
        dM_k_pc = Constant(0.0, dim=0) * pc["dalpha"]
    elif mob_key in {"degenerate", "deg", "alpha(1-alpha)", "alpha*(1-alpha)", "a(1-a)"}:
        M_k_fx = M0_fx * alpha_k_fx * (1.0 - alpha_k_fx)
        M_n_fx = M0_fx * alpha_n_fx * (1.0 - alpha_n_fx)
        M_k_pc = pc["M0"] * pc["alpha_k"] * (Constant(1.0, dim=0) - pc["alpha_k"])
        M_n_pc = pc["M0"] * pc["alpha_n"] * (Constant(1.0, dim=0) - pc["alpha_n"])
        dM_k_pc = pc["M0"] * (Constant(1.0, dim=0) - Constant(2.0, dim=0) * pc["alpha_k"]) * pc["dalpha"]
    else:
        raise ValueError(f"Unknown COMP_FENICS_ALPHA_CH_MOBILITY={fenicsx.get('mobility')!r}. Use 'constant' or 'degenerate'.")

    # Double-well derivatives for W(α)=α^2(1-α)^2.
    Wp_k_fx = 2.0 * alpha_k_fx * (1.0 - alpha_k_fx) * (1.0 - 2.0 * alpha_k_fx)
    Wpp_k_fx = 2.0 - 12.0 * alpha_k_fx + 12.0 * (alpha_k_fx * alpha_k_fx)

    Wp_k_pc = Constant(2.0, dim=0) * pc["alpha_k"] * (Constant(1.0, dim=0) - pc["alpha_k"]) * (
        Constant(1.0, dim=0) - Constant(2.0, dim=0) * pc["alpha_k"]
    )
    Wpp_k_pc = Constant(2.0, dim=0) + (Constant(-12.0, dim=0) * pc["alpha_k"]) + (Constant(12.0, dim=0) * (pc["alpha_k"] * pc["alpha_k"]))

    # ------------------------------------------------------------------
    # Residuals (pc)
    # ------------------------------------------------------------------
    inv_dt_pc = Constant(1.0, dim=0) / pc["dt"]
    r_alpha_pc = pc["xi"] * ((pc["alpha_k"] - pc["alpha_n"]) * inv_dt_pc) * dx_pc
    # Keep GradOpInfo on the left in scalar×grad products for backend compatibility:
    # `scalar_function * grad(function)` is not supported, but `grad(function) * scalar_function` is.
    r_alpha_pc += pc["theta"] * inner(grad(pc["mu_k"]) * M_k_pc, grad(pc["xi"])) * dx_pc
    r_alpha_pc += pc["one_m_theta"] * inner(grad(pc["mu_n"]) * M_n_pc, grad(pc["xi"])) * dx_pc

    r_mu_pc = pc["eta"] * pc["mu_k"] * dx_pc
    r_mu_pc += -(pc["gamma"] * pc["eps"]) * inner(grad(pc["alpha_k"]), grad(pc["eta"])) * dx_pc
    r_mu_pc += -pc["eta"] * ((pc["gamma"] / pc["eps"]) * Wp_k_pc) * dx_pc

    r_total_pc = r_alpha_pc + r_mu_pc

    # Jacobian (pc)
    a_alpha_pc = pc["xi"] * (pc["dalpha"] * inv_dt_pc) * dx_pc
    a_alpha_pc += pc["theta"] * (dM_k_pc * inner(grad(pc["mu_k"]), grad(pc["xi"]))) * dx_pc
    a_alpha_pc += pc["theta"] * inner(M_k_pc * grad(pc["dmu"]), grad(pc["xi"])) * dx_pc

    a_mu_pc = pc["eta"] * pc["dmu"] * dx_pc
    a_mu_pc += -(pc["gamma"] * pc["eps"]) * inner(grad(pc["dalpha"]), grad(pc["eta"])) * dx_pc
    a_mu_pc += -pc["eta"] * ((pc["gamma"] / pc["eps"]) * Wpp_k_pc * pc["dalpha"]) * dx_pc

    a_total_pc = a_alpha_pc + a_mu_pc

    # ------------------------------------------------------------------
    # Residuals (fenics)
    # ------------------------------------------------------------------
    r_alpha_fx = xi_fx * ((alpha_k_fx - alpha_n_fx) / dt_fx) * dx_fx
    r_alpha_fx += theta_fx * M_k_fx * ufl.dot(ufl.grad(mu_k_fx), ufl.grad(xi_fx)) * dx_fx
    r_alpha_fx += (1.0 - theta_fx) * M_n_fx * ufl.dot(ufl.grad(mu_n_fx), ufl.grad(xi_fx)) * dx_fx

    r_mu_fx = eta_fx * mu_k_fx * dx_fx
    r_mu_fx += -(gamma_fx * eps_fx) * ufl.dot(ufl.grad(alpha_k_fx), ufl.grad(eta_fx)) * dx_fx
    r_mu_fx += -(gamma_fx / eps_fx) * eta_fx * Wp_k_fx * dx_fx

    r_total_fx = r_alpha_fx + r_mu_fx
    a_total_fx = ufl.derivative(r_total_fx, fenicsx["w_k"], dw_fx)

    # ------------------------------------------------------------------
    # Terms dictionary
    # ------------------------------------------------------------------
    terms = {
        "Alpha CH alpha (res)": {"pc": r_alpha_pc, "fx": r_alpha_fx, "mat": False},
        "Alpha CH mu (res)": {"pc": r_mu_pc, "fx": r_mu_fx, "mat": False},
        "Alpha CH total residual": {"pc": r_total_pc, "fx": r_total_fx, "mat": False},
        "Alpha CH total jacobian": {"pc": a_total_pc, "fx": a_total_fx, "mat": True},
    }

    filter_terms = os.environ.get("COMP_FENICS_TERMS")
    if filter_terms:
        allowed = {name.strip() for name in filter_terms.split(",") if name.strip()}
        terms = {k: v for k, v in terms.items() if k in allowed}
        print(f"Running filtered terms only: {sorted(terms)}")
    else:
        print(f"Running Cahn–Hilliard suite: {sorted(terms)}")

    # Pre-assemble FEniCSx reference outputs once (backend-independent)
    fenics_ref = {}
    for name, spec in terms.items():
        try:
            form_fx_compiled = dolfinx.fem.form(spec["fx"])
            if spec["mat"]:
                A = dolfinx.fem.petsc.assemble_matrix(form_fx_compiled)
                A.assemble()
                indptr, indices, data = A.getValuesCSR()
                fenics_ref[name] = csr_matrix((data, indices, indptr), shape=A.getSize()).toarray()
            else:
                vec = dolfinx.fem.petsc.assemble_vector(form_fx_compiled)
                fenics_ref[name] = vec.array.copy()
        except Exception as exc:
            print(f"❌ FEniCSx assembly failed for '{name}': {exc}")
            fenics_ref[name] = None

    backends_spec = os.environ.get("BACKEND", "jit")
    if backends_spec.strip().lower() == "all":
        backends = ["python", "jit", "cpp"]
    else:
        backends = [b.strip() for b in backends_spec.split(",") if b.strip()]
    parity_rtol = float(os.environ.get("COMP_FENICS_PARITY_RTOL", "1e-9"))
    parity_atol = float(os.environ.get("COMP_FENICS_PARITY_ATOL", "1e-9"))

    # Optional: backend parity check against a reference backend
    ref_backend = "python" if "python" in backends else (backends[0] if backends else "python")
    ref_pc = {}

    for backend_type in backends:
        print("\n" + "=" * 70)
        print(f"CAHN–HILLIARD COMPARISON (backend={backend_type}, qdeg={qdeg})")
        print("=" * 70)

        failed_tests = []
        success_count = 0

        for name, spec in terms.items():
            if fenics_ref.get(name) is None:
                failed_tests.append(f"{name} (fenics-assemble)")
                continue

            J_pc, R_pc = None, None
            print(f"\nCompiling/assembling '{name}' [backend={backend_type}]")
            try:
                if spec["mat"]:
                    J_pc, _ = assemble_form(Equation(spec["pc"], None), dof_handler_pc, quad_degree=qdeg, bcs=[], backend=backend_type)
                else:
                    _, R_pc = assemble_form(Equation(None, spec["pc"]), dof_handler_pc, bcs=[], backend=backend_type)
            except Exception as exc:
                print(f"❌ pycutfem assembly failed for '{name}' on backend '{backend_type}': {exc}")
                failed_tests.append(f"{name} (assemble-{backend_type})")
                continue

            # Backend parity (pycutfem) against a reference backend
            if backend_type == ref_backend:
                if spec["mat"]:
                    ref_pc[name] = J_pc.toarray()
                else:
                    ref_pc[name] = np.asarray(R_pc, dtype=float).copy()
            else:
                try:
                    if spec["mat"]:
                        np.testing.assert_allclose(J_pc.toarray(), ref_pc[name], rtol=parity_rtol, atol=parity_atol)
                    else:
                        np.testing.assert_allclose(np.asarray(R_pc, dtype=float), ref_pc[name], rtol=parity_rtol, atol=parity_atol)
                    print(f"✅ pycutfem backend parity OK vs '{ref_backend}'.")
                except Exception as exc:
                    print(f"❌ pycutfem backend parity FAILED vs '{ref_backend}': {exc}")
                    failed_tests.append(f"{name} (parity-{backend_type}-vs-{ref_backend})")

            # Compare to FEniCSx reference
            J_fx, R_fx = None, None
            if spec["mat"]:
                J_fx = fenics_ref[name]
            else:
                R_fx = fenics_ref[name]

            is_success = compare_term(
                f"{name} [backend={backend_type}]",
                J_pc,
                R_pc,
                J_fx,
                R_fx,
                P_map,
                dof_handler_pc,
                W_fx,
                rtol=1e-8,
                atol=1e-8,
            )
            if is_success:
                success_count += 1
            else:
                failed_tests.append(name)

        print_test_summary(success_count, failed_tests)


# ==============================================================================
#  One-domain biofilm comparison harness (v,p,vS,u,phi,alpha,S,X,...) vs FEniCSx
# ==============================================================================


def setup_biofilm_problems(*, nx: int = 2, ny: int = 2):
    """
    Build a small one-domain biofilm mixed space with optional fracture/detached blocks:
      pycutfem: CG fluid (v_x,v_y) Q2 or H(div) fluid v RT0, plus
                p Q1 + (vS_x,vS_y) Q2 + (u_x,u_y) Q2 + (phi,alpha,d,S,X) Q1
      dolfinx : (v,p,vS,u,phi,alpha,d,S,X) as
                (P2_vec or RT0, P1, P2_vec, P2_vec, P1, P1, P1, P1, P1)
    """
    # --- pycutfem mesh (Q2 geometry so dof coordinates match dolfinx P2 coords) ---
    nodes_q2, elems_q2, _, corners_q2 = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=2)
    mesh_q2 = Mesh(
        nodes=nodes_q2,
        element_connectivity=elems_q2,
        elements_corner_nodes=corners_q2,
        element_type="quad",
        poly_order=2,
    )
    mesh_q2.tag_boundary_edges({"all": lambda x, y: True})

    # Parameters (keep in sync with the unit tests and FEniCSx constants below).
    dt_val = float(os.environ.get("COMP_FENICS_DT", "0.1"))
    theta_val = float(os.environ.get("COMP_FENICS_THETA", "0.5"))
    fluid_convection = str(os.environ.get("COMP_FENICS_FLUID_CONVECTION", "full"))
    fluid_space = str(os.environ.get("COMP_FENICS_FLUID_SPACE", "cg")).strip().lower()
    fluid_hdiv_order = int(os.environ.get("COMP_FENICS_FLUID_HDIV_ORDER", "0"))
    alpha_advect_with = str(os.environ.get("COMP_FENICS_ALPHA_ADVECT_WITH", "vS"))
    alpha_advection_form = str(os.environ.get("COMP_FENICS_ALPHA_ADVECTION_FORM", "advective"))
    kappa_inv_model = str(os.environ.get("COMP_FENICS_KAPPA_INV_MODEL", "spatial")).strip().lower()
    gamma_div_val = float(os.environ.get("COMP_FENICS_GAMMA_DIV", "0.0"))
    gamma_u_val = float(os.environ.get("COMP_FENICS_GAMMA_U", "0.0"))
    u_extension_mode = str(os.environ.get("COMP_FENICS_U_EXTENSION_MODE", "l2"))
    gamma_u_pin_val = float(os.environ.get("COMP_FENICS_GAMMA_U_PIN", "0.0"))
    v_supg_val = float(os.environ.get("COMP_FENICS_V_SUPG", "0.0"))
    v_supg_mode = str(os.environ.get("COMP_FENICS_V_SUPG_MODE", "streamline"))
    v_supg_c_nu_val = float(os.environ.get("COMP_FENICS_V_SUPG_C_NU", "4.0"))
    hdiv_tangential_dirichlet = _env_truthy("COMP_FENICS_HDIV_TANGENTIAL_DIRICHLET", False)
    hdiv_tangential_gamma = float(os.environ.get("COMP_FENICS_HDIV_TANGENTIAL_GAMMA", "20.0"))
    hdiv_tangential_method = str(os.environ.get("COMP_FENICS_HDIV_TANGENTIAL_METHOD", "penalty")).strip().lower()
    if hdiv_tangential_method not in {"penalty", "nitsche"}:
        raise ValueError(f"Unsupported COMP_FENICS_HDIV_TANGENTIAL_METHOD={hdiv_tangential_method!r}.")
    alpha_ch_M_val = float(os.environ.get("COMP_FENICS_ALPHA_CH_M", "0.0"))
    alpha_ch_gamma_val = float(os.environ.get("COMP_FENICS_ALPHA_CH_GAMMA", "0.0"))
    alpha_ch_eps_val = float(os.environ.get("COMP_FENICS_ALPHA_CH_EPS", "0.1"))
    alpha_ch_mobility = str(os.environ.get("COMP_FENICS_ALPHA_CH_MOBILITY", "constant"))
    alpha_ch_enabled = (alpha_ch_M_val != 0.0) and (alpha_ch_gamma_val != 0.0)
    if fluid_space not in {"cg", "hdiv"}:
        raise ValueError(f"Unsupported COMP_FENICS_FLUID_SPACE={fluid_space!r}.")
    if fluid_space == "hdiv" and int(fluid_hdiv_order) != 0:
        if not (int(fluid_hdiv_order) == 1 and int(nx) == 1 and int(ny) == 1):
            raise NotImplementedError(
                "Biofilm H(div) comparison currently supports RT0 on arbitrary structured meshes, "
                "and RT1 on the single-cell whole-domain harness (COMP_FENICS_NX=1, COMP_FENICS_NY=1)."
            )
    # Keep field insertion order stable (matches DofHandler ordering).
    field_specs_pc = {
        "p": 1,
        "vS_x": 2,
        "vS_y": 2,
        "u_x": 2,
        "u_y": 2,
        "phi": 1,
        "alpha": 1,
        **({"mu_alpha": 1} if alpha_ch_enabled else {}),
        "d": 1,
        "S": 1,
        "X": 1,
    }
    if fluid_space == "cg":
        field_specs_pc = {"v_x": 2, "v_y": 2, **field_specs_pc}
    else:
        field_specs_pc = {"v": ("RT", int(fluid_hdiv_order)), **field_specs_pc}
    mixed_element_pc = MixedElement(mesh_q2, field_specs=field_specs_pc)
    dof_handler_pc = DofHandler(mixed_element_pc, method="cg")

    VvS_pc = FunctionSpace("vS", ["vS_x", "vS_y"], dim=1)
    Vu_pc = FunctionSpace("u", ["u_x", "u_y"], dim=1)

    if fluid_space == "cg":
        Vv_pc = FunctionSpace("v", ["v_x", "v_y"], dim=1)
        dv_pc = VectorTrialFunction(Vv_pc, dof_handler=dof_handler_pc)
        w_pc = VectorTestFunction(Vv_pc, dof_handler=dof_handler_pc)
        v_k_pc = VectorFunction(name="v_k", field_names=["v_x", "v_y"], dof_handler=dof_handler_pc)
        v_n_pc = VectorFunction(name="v_n", field_names=["v_x", "v_y"], dof_handler=dof_handler_pc)
    else:
        dv_pc = HdivTrialFunction("v")
        w_pc = HdivTestFunction("v")
        v_k_pc = HdivFunction(name="v_k", field_name="v", dof_handler=dof_handler_pc)
        v_n_pc = HdivFunction(name="v_n", field_name="v", dof_handler=dof_handler_pc)

    pc = {
        # trial/test
        "dv": dv_pc,
        "dp": TrialFunction("p", dof_handler=dof_handler_pc),
        "dvS": VectorTrialFunction(VvS_pc, dof_handler=dof_handler_pc),
        "du": VectorTrialFunction(Vu_pc, dof_handler=dof_handler_pc),
        "dphi": TrialFunction("phi", dof_handler=dof_handler_pc),
        "dalpha": TrialFunction("alpha", dof_handler=dof_handler_pc),
        "dmu_alpha": TrialFunction("mu_alpha", dof_handler=dof_handler_pc) if alpha_ch_enabled else None,
        "dd": TrialFunction("d", dof_handler=dof_handler_pc),
        "dS": TrialFunction("S", dof_handler=dof_handler_pc),
        "dX": TrialFunction("X", dof_handler=dof_handler_pc),
        "w": w_pc,
        "q": TestFunction("p", dof_handler=dof_handler_pc),
        "eta_vS": VectorTestFunction(VvS_pc, dof_handler=dof_handler_pc),
        "eta": VectorTestFunction(Vu_pc, dof_handler=dof_handler_pc),
        "zeta": TestFunction("phi", dof_handler=dof_handler_pc),
        "xi": TestFunction("alpha", dof_handler=dof_handler_pc),
        "eta_mu": TestFunction("mu_alpha", dof_handler=dof_handler_pc) if alpha_ch_enabled else None,
        "chi": TestFunction("d", dof_handler=dof_handler_pc),
        "r": TestFunction("S", dof_handler=dof_handler_pc),
        "y": TestFunction("X", dof_handler=dof_handler_pc),
        # states at k (t_{n+1})
        "v_k": v_k_pc,
        "p_k": Function(name="p_k", field_name="p", dof_handler=dof_handler_pc),
        "vS_k": VectorFunction(name="vS_k", field_names=["vS_x", "vS_y"], dof_handler=dof_handler_pc),
        "u_k": VectorFunction(name="u_k", field_names=["u_x", "u_y"], dof_handler=dof_handler_pc),
        "phi_k": Function(name="phi_k", field_name="phi", dof_handler=dof_handler_pc),
        "alpha_k": Function(name="alpha_k", field_name="alpha", dof_handler=dof_handler_pc),
        "mu_alpha_k": Function(name="mu_alpha_k", field_name="mu_alpha", dof_handler=dof_handler_pc) if alpha_ch_enabled else None,
        "d_k": Function(name="d_k", field_name="d", dof_handler=dof_handler_pc),
        "S_k": Function(name="S_k", field_name="S", dof_handler=dof_handler_pc),
        "X_k": Function(name="X_k", field_name="X", dof_handler=dof_handler_pc),
        # states at n (t_n)
        "v_n": v_n_pc,
        "p_n": Function(name="p_n", field_name="p", dof_handler=dof_handler_pc),
        "vS_n": VectorFunction(name="vS_n", field_names=["vS_x", "vS_y"], dof_handler=dof_handler_pc),
        "u_n": VectorFunction(name="u_n", field_names=["u_x", "u_y"], dof_handler=dof_handler_pc),
        "phi_n": Function(name="phi_n", field_name="phi", dof_handler=dof_handler_pc),
        "alpha_n": Function(name="alpha_n", field_name="alpha", dof_handler=dof_handler_pc),
        "mu_alpha_n": Function(name="mu_alpha_n", field_name="mu_alpha", dof_handler=dof_handler_pc) if alpha_ch_enabled else None,
        "d_n": Function(name="d_n", field_name="d", dof_handler=dof_handler_pc),
        "S_n": Function(name="S_n", field_name="S", dof_handler=dof_handler_pc),
        "X_n": Function(name="X_n", field_name="X", dof_handler=dof_handler_pc),
        # parameters
        "rho_f": Constant(1.0, dim=0),
        "mu_f": Constant(1.0e-2, dim=0),
        "kappa_inv": Constant(10.0, dim=0),
        "kappa_inv_model": str(kappa_inv_model),
        "mu_s": Constant(1.0, dim=0),
        "lambda_s": Constant(1.0, dim=0),
        "dt": Constant(dt_val, dim=0),
        "theta": float(theta_val),
        "fluid_convection": str(fluid_convection),
        "fluid_space": str(fluid_space),
        "fluid_hdiv_order": int(fluid_hdiv_order),
        "alpha_advect_with": str(alpha_advect_with),
        "alpha_advection_form": str(alpha_advection_form),
        "gamma_div": float(gamma_div_val),
        "gamma_u": float(gamma_u_val),
        "u_extension_mode": str(u_extension_mode),
        "gamma_u_pin": float(gamma_u_pin_val),
        "v_supg": float(v_supg_val),
        "v_supg_mode": str(v_supg_mode),
        "v_supg_c_nu": float(v_supg_c_nu_val),
        "hdiv_tangential_dirichlet": bool(hdiv_tangential_dirichlet),
        "hdiv_tangential_gamma": float(hdiv_tangential_gamma),
        "hdiv_tangential_method": str(hdiv_tangential_method),
        # Optional Cahn–Hilliard regularization for alpha (adds mu_alpha when enabled).
        "alpha_ch_enabled": bool(alpha_ch_enabled),
        "alpha_ch_M": float(alpha_ch_M_val),
        "alpha_ch_gamma": float(alpha_ch_gamma_val),
        "alpha_ch_eps": float(alpha_ch_eps_val),
        "alpha_ch_mobility": str(alpha_ch_mobility),
        # transport/growth
        "D_phi": float(os.environ.get("COMP_FENICS_D_PHI", "0.1")),
        "gamma_phi": float(os.environ.get("COMP_FENICS_GAMMA_PHI", "1.0")),
        "D_alpha": float(os.environ.get("COMP_FENICS_D_ALPHA", "0.1")),
        "D_X": float(os.environ.get("COMP_FENICS_D_X", "0.05")),
        "D_S": float(os.environ.get("COMP_FENICS_D_S", "0.1")),
        "mu_max": float(os.environ.get("COMP_FENICS_MU_MAX", "0.4")),
        "K_S": float(os.environ.get("COMP_FENICS_K_S", "0.3")),
        "k_g": float(os.environ.get("COMP_FENICS_K_G", "0.5")),
        "k_d": float(os.environ.get("COMP_FENICS_K_D", "0.1")),
        "Y": float(os.environ.get("COMP_FENICS_Y", "0.8")),
        "rho_s_star": float(os.environ.get("COMP_FENICS_RHO_S_STAR", "1.0")),
        "k_det": float(os.environ.get("COMP_FENICS_K_DET", "0.2")),
        # kinetic damage/fracture-like cohesion-loss block
        "damage_model": str(os.environ.get("COMP_FENICS_DAMAGE_MODEL", "kinetic")),
        "damage_k": float(os.environ.get("COMP_FENICS_DAMAGE_K", "2.0")),
        "damage_sigma_cr": float(os.environ.get("COMP_FENICS_DAMAGE_SIGMA_CR", "0.2")),
        "damage_m": float(os.environ.get("COMP_FENICS_DAMAGE_M", "2.0")),
        "damage_D": float(os.environ.get("COMP_FENICS_DAMAGE_D", "0.03")),
        "damage_gamma_out": float(os.environ.get("COMP_FENICS_DAMAGE_GAMMA_OUT", "0.2")),
        "damage_eta_pos": float(os.environ.get("COMP_FENICS_DAMAGE_ETA_POS", "1.0e-12")),
        "damage_kappa_stiff": float(os.environ.get("COMP_FENICS_DAMAGE_KAPPA_STIFF", "1.0e-6")),
        "damage_kappa_perm": float(os.environ.get("COMP_FENICS_DAMAGE_KAPPA_PERM", "1.0e-6")),
        "mesh": mesh_q2,
    }

    # --- dolfinx mesh / mixed space ---
    mesh_fx = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, dolfinx.mesh.CellType.quadrilateral)
    gdim = mesh_fx.geometry.dim
    P2_vec = basix.ufl.element("Lagrange", "quadrilateral", 2, shape=(gdim,))
    P1 = basix.ufl.element("Lagrange", "quadrilateral", 1)
    fluid_el_fx = basix.ufl.element("RT", "quadrilateral", int(fluid_hdiv_order) + 1) if fluid_space == "hdiv" else P2_vec
    if alpha_ch_enabled:
        # W = (v, p, vS, u, phi, alpha, mu_alpha, d, S, X)
        W_el = mixed_element([fluid_el_fx, P1, P2_vec, P2_vec, P1, P1, P1, P1, P1, P1])
    else:
        # W = (v, p, vS, u, phi, alpha, d, S, X)
        W_el = mixed_element([fluid_el_fx, P1, P2_vec, P2_vec, P1, P1, P1, P1, P1])
    W = dolfinx.fem.functionspace(mesh_fx, W_el)

    facet_tags = None
    if fluid_space == "hdiv" and bool(hdiv_tangential_dirichlet):
        fdim = mesh_fx.topology.dim - 1
        facet_blocks = [
            dolfinx.mesh.locate_entities_boundary(mesh_fx, fdim, lambda x: np.isclose(x[0], 0.0)),
            dolfinx.mesh.locate_entities_boundary(mesh_fx, fdim, lambda x: np.isclose(x[1], 0.0)),
            dolfinx.mesh.locate_entities_boundary(mesh_fx, fdim, lambda x: np.isclose(x[1], 1.0)),
        ]
        tagged_facets = np.unique(np.concatenate([blk.astype(np.int32, copy=False) for blk in facet_blocks]))
        tagged_values = np.full(tagged_facets.shape, 1, dtype=np.int32)
        facet_tags = dolfinx.mesh.meshtags(mesh_fx, fdim, tagged_facets, tagged_values)

    fenicsx = {
        "W": W,
        "mesh": mesh_fx,
        "hdiv_tangential_facet_tags": facet_tags,
        # constants
        "rho_f": dolfinx.fem.Constant(mesh_fx, float(pc["rho_f"].value)),
        "mu_f": dolfinx.fem.Constant(mesh_fx, float(pc["mu_f"].value)),
        "kappa_inv": dolfinx.fem.Constant(mesh_fx, float(pc["kappa_inv"].value)),
        "kappa_inv_model": str(kappa_inv_model),
        "mu_s": dolfinx.fem.Constant(mesh_fx, float(pc["mu_s"].value)),
        "lambda_s": dolfinx.fem.Constant(mesh_fx, float(pc["lambda_s"].value)),
        "dt": dolfinx.fem.Constant(mesh_fx, dt_val),
        "theta": dolfinx.fem.Constant(mesh_fx, theta_val),
        "fluid_convection": str(fluid_convection),
        "alpha_advect_with": str(alpha_advect_with),
        "alpha_advection_form": str(alpha_advection_form),
        "gamma_div": dolfinx.fem.Constant(mesh_fx, float(gamma_div_val)),
        "gamma_u": dolfinx.fem.Constant(mesh_fx, float(gamma_u_val)),
        "u_extension_mode": str(u_extension_mode),
        "gamma_u_pin": dolfinx.fem.Constant(mesh_fx, float(gamma_u_pin_val)),
        "v_supg": dolfinx.fem.Constant(mesh_fx, float(v_supg_val)),
        "v_supg_mode": str(v_supg_mode),
        "v_supg_c_nu": dolfinx.fem.Constant(mesh_fx, float(v_supg_c_nu_val)),
        "hdiv_tangential_dirichlet": bool(hdiv_tangential_dirichlet),
        "hdiv_tangential_gamma": dolfinx.fem.Constant(mesh_fx, float(hdiv_tangential_gamma)),
        "hdiv_tangential_method": str(hdiv_tangential_method),
        "alpha_ch_enabled": bool(alpha_ch_enabled),
        "alpha_ch_M": dolfinx.fem.Constant(mesh_fx, float(alpha_ch_M_val)),
        "alpha_ch_gamma": dolfinx.fem.Constant(mesh_fx, float(alpha_ch_gamma_val)),
        "alpha_ch_eps": dolfinx.fem.Constant(mesh_fx, float(alpha_ch_eps_val)),
        "alpha_ch_mobility": str(alpha_ch_mobility),
        "D_phi": dolfinx.fem.Constant(mesh_fx, float(pc["D_phi"])),
        "gamma_phi": dolfinx.fem.Constant(mesh_fx, float(pc["gamma_phi"])),
        "D_alpha": dolfinx.fem.Constant(mesh_fx, float(pc["D_alpha"])),
        "D_X": dolfinx.fem.Constant(mesh_fx, float(pc["D_X"])),
        "D_S": dolfinx.fem.Constant(mesh_fx, float(pc["D_S"])),
        "mu_max": dolfinx.fem.Constant(mesh_fx, float(pc["mu_max"])),
        "K_S": dolfinx.fem.Constant(mesh_fx, float(pc["K_S"])),
        "k_g": dolfinx.fem.Constant(mesh_fx, float(pc["k_g"])),
        "k_d": dolfinx.fem.Constant(mesh_fx, float(pc["k_d"])),
        "Y": dolfinx.fem.Constant(mesh_fx, float(pc["Y"])),
        "rho_s_star": dolfinx.fem.Constant(mesh_fx, float(pc["rho_s_star"])),
        "k_det": dolfinx.fem.Constant(mesh_fx, float(pc["k_det"])),
        "damage_k": dolfinx.fem.Constant(mesh_fx, float(pc["damage_k"])),
        "damage_sigma_cr": dolfinx.fem.Constant(mesh_fx, float(pc["damage_sigma_cr"])),
        "damage_m": dolfinx.fem.Constant(mesh_fx, float(pc["damage_m"])),
        "damage_D": dolfinx.fem.Constant(mesh_fx, float(pc["damage_D"])),
        "damage_gamma_out": dolfinx.fem.Constant(mesh_fx, float(pc["damage_gamma_out"])),
        "damage_eta_pos": dolfinx.fem.Constant(mesh_fx, float(pc["damage_eta_pos"])),
        "damage_kappa_stiff": dolfinx.fem.Constant(mesh_fx, float(pc["damage_kappa_stiff"])),
        "damage_kappa_perm": dolfinx.fem.Constant(mesh_fx, float(pc["damage_kappa_perm"])),
        # states (mixed)
        "w_k": dolfinx.fem.Function(W, name="w_k"),
        "w_n": dolfinx.fem.Function(W, name="w_n"),
    }

    return pc, dof_handler_pc, fenicsx


def create_true_dof_map_biofilm(dof_handler_pc: DofHandler, W_fenicsx, *, include_mu_alpha: bool = False):
    print("=" * 70)
    if include_mu_alpha:
        print("Discovering biofilm DoF map (v,p,vS,u,phi,alpha,mu_alpha,d,S,X) by matching DoF coordinates...")
    else:
        print("Discovering biofilm DoF map (v,p,vS,u,phi,alpha,d,S,X) by matching DoF coordinates...")
    print("=" * 70)

    # fenics: W = (v, p, vS, u, phi, alpha, [mu_alpha], d, S, X)
    Wv, Vv_map = W_fenicsx.sub(0).collapse()
    Wp, P_map_fx = W_fenicsx.sub(1).collapse()
    WvS, VvS_map = W_fenicsx.sub(2).collapse()
    Wu, Vu_map = W_fenicsx.sub(3).collapse()
    Wphi, Phi_map_fx = W_fenicsx.sub(4).collapse()
    Walpha, Alpha_map_fx = W_fenicsx.sub(5).collapse()
    if include_mu_alpha:
        Wmu_alpha, Mu_map_fx = W_fenicsx.sub(6).collapse()
        d_idx = 7
    else:
        Wmu_alpha, Mu_map_fx = None, None
        d_idx = 6
    s_idx = d_idx + 1
    x_idx = d_idx + 2
    Wd, D_map_fx = W_fenicsx.sub(d_idx).collapse()
    WS, S_map_fx = W_fenicsx.sub(s_idx).collapse()
    WX, X_map_fx = W_fenicsx.sub(x_idx).collapse()

    Wv0, Vv0_map = Wv.sub(0).collapse()
    Wv1, Vv1_map = Wv.sub(1).collapse()
    WvS0, VvS0_map = WvS.sub(0).collapse()
    WvS1, VvS1_map = WvS.sub(1).collapse()
    Wu0, Vu0_map = Wu.sub(0).collapse()
    Wu1, Vu1_map = Wu.sub(1).collapse()

    fx_coords = {
        "v_x": Wv0.tabulate_dof_coordinates()[:, :2],
        "v_y": Wv1.tabulate_dof_coordinates()[:, :2],
        "p": Wp.tabulate_dof_coordinates()[:, :2],
        "vS_x": WvS0.tabulate_dof_coordinates()[:, :2],
        "vS_y": WvS1.tabulate_dof_coordinates()[:, :2],
        "u_x": Wu0.tabulate_dof_coordinates()[:, :2],
        "u_y": Wu1.tabulate_dof_coordinates()[:, :2],
        "phi": Wphi.tabulate_dof_coordinates()[:, :2],
        "alpha": Walpha.tabulate_dof_coordinates()[:, :2],
        **({"mu_alpha": Wmu_alpha.tabulate_dof_coordinates()[:, :2]} if include_mu_alpha else {}),
        "d": Wd.tabulate_dof_coordinates()[:, :2],
        "S": WS.tabulate_dof_coordinates()[:, :2],
        "X": WX.tabulate_dof_coordinates()[:, :2],
    }
    fx_dofs = {
        "v_x": np.array(Vv_map)[np.array(Vv0_map)],
        "v_y": np.array(Vv_map)[np.array(Vv1_map)],
        "p": np.array(P_map_fx),
        "vS_x": np.array(VvS_map)[np.array(VvS0_map)],
        "vS_y": np.array(VvS_map)[np.array(VvS1_map)],
        "u_x": np.array(Vu_map)[np.array(Vu0_map)],
        "u_y": np.array(Vu_map)[np.array(Vu1_map)],
        "phi": np.array(Phi_map_fx),
        "alpha": np.array(Alpha_map_fx),
        **({"mu_alpha": np.array(Mu_map_fx)} if include_mu_alpha else {}),
        "d": np.array(D_map_fx),
        "S": np.array(S_map_fx),
        "X": np.array(X_map_fx),
    }

    pc_fields = ["v_x", "v_y", "p", "vS_x", "vS_y", "u_x", "u_y", "phi", "alpha"]
    if include_mu_alpha:
        pc_fields.append("mu_alpha")
    pc_fields.extend(["d", "S", "X"])
    pc_coords = {f: get_pycutfem_dof_coords(dof_handler_pc, f) for f in pc_fields}
    pc_dofs = {f: dof_handler_pc.get_field_slice(f) for f in pc_fields}

    P = np.zeros(dof_handler_pc.total_dofs, dtype=int)

    # v_x and v_y share coordinates in vector Q2 space
    coord_map_v = one_to_one_map_coords(pc_coords["v_x"], fx_coords["v_x"])
    P[pc_dofs["v_x"]] = fx_dofs["v_x"][coord_map_v]
    P[pc_dofs["v_y"]] = fx_dofs["v_y"][coord_map_v]

    # vS_x and vS_y share coordinates in vector Q2 space
    coord_map_vS = one_to_one_map_coords(pc_coords["vS_x"], fx_coords["vS_x"])
    P[pc_dofs["vS_x"]] = fx_dofs["vS_x"][coord_map_vS]
    P[pc_dofs["vS_y"]] = fx_dofs["vS_y"][coord_map_vS]

    # u_x and u_y share coordinates in vector Q2 space
    coord_map_u = one_to_one_map_coords(pc_coords["u_x"], fx_coords["u_x"])
    P[pc_dofs["u_x"]] = fx_dofs["u_x"][coord_map_u]
    P[pc_dofs["u_y"]] = fx_dofs["u_y"][coord_map_u]

    # Scalar Q1 fields
    scalar_fields = ("p", "phi", "alpha") + (("mu_alpha",) if include_mu_alpha else ()) + ("d", "S", "X")
    for fld in scalar_fields:
        coord_map = one_to_one_map_coords(pc_coords[fld], fx_coords[fld])
        P[pc_dofs[fld]] = fx_dofs[fld][coord_map]

    print("Biofilm DoF map discovered successfully.")
    return P


def _biofilm_hdiv_state_vector(pc, dof_handler_pc: DofHandler, *, suffix: str, include_mu_alpha: bool = False) -> np.ndarray:
    state = np.zeros(dof_handler_pc.total_dofs, dtype=float)
    state[np.asarray(dof_handler_pc.get_field_slice("v"), dtype=int)] = np.asarray(pc[f"v_{suffix}"].nodal_values, dtype=float)
    state[np.asarray(dof_handler_pc.get_field_slice("p"), dtype=int)] = np.asarray(pc[f"p_{suffix}"].nodal_values, dtype=float)
    for base, fields in (("vS", ("vS_x", "vS_y")), ("u", ("u_x", "u_y"))):
        vec = pc[f"{base}_{suffix}"]
        state[np.asarray(dof_handler_pc.get_field_slice(fields[0]), dtype=int)] = np.asarray(vec.nodal_values_component(0), dtype=float)
        state[np.asarray(dof_handler_pc.get_field_slice(fields[1]), dtype=int)] = np.asarray(vec.nodal_values_component(1), dtype=float)
    for fld in ("phi", "alpha", "d", "S", "X"):
        state[np.asarray(dof_handler_pc.get_field_slice(fld), dtype=int)] = np.asarray(pc[f"{fld}_{suffix}"].nodal_values, dtype=float)
    if include_mu_alpha:
        state[np.asarray(dof_handler_pc.get_field_slice("mu_alpha"), dtype=int)] = np.asarray(pc[f"mu_alpha_{suffix}"].nodal_values, dtype=float)
    return state


def create_true_dof_map_biofilm_hdiv(dof_handler_pc: DofHandler, W_fenicsx, *, include_mu_alpha: bool = False):
    print("=" * 70)
    if include_mu_alpha:
        print("Discovering biofilm H(div) DoF map (RT,p,vS,u,phi,alpha,mu_alpha,d,S,X)...")
    else:
        print("Discovering biofilm H(div) DoF map (RT,p,vS,u,phi,alpha,d,S,X)...")
    print("=" * 70)

    P = np.zeros(dof_handler_pc.total_dofs, dtype=int)
    fx_coords_all = np.zeros((W_fenicsx.dofmap.index_map.size_global, 2), dtype=float)

    Wv, Vv_map = W_fenicsx.sub(0).collapse()
    v_fx_dofs_collapsed, v_fx_kinds, v_fx_points, v_fx_modes = _fenics_rt_dofs_and_descriptors(Wv)
    v_fx_parent = np.asarray(Vv_map, dtype=int)[v_fx_dofs_collapsed]
    v_pc_slice = np.asarray(dof_handler_pc.get_field_slice("v"), dtype=int)
    _, v_pc_kinds, v_pc_points, v_pc_modes = _pycutfem_rt_dofs_and_descriptors(dof_handler_pc, "v")
    v_pc_to_fx = _map_rt_descriptors(v_pc_kinds, v_pc_points, v_pc_modes, v_fx_kinds, v_fx_points, v_fx_modes)
    P[v_pc_slice] = v_fx_parent[v_pc_to_fx]
    fx_coords_all[v_fx_parent] = v_fx_points

    Wp, P_map_fx = W_fenicsx.sub(1).collapse()
    p_fx_coords = Wp.tabulate_dof_coordinates()[:, :2]
    p_fx_parent = np.asarray(P_map_fx, dtype=int)
    p_pc_slice = np.asarray(dof_handler_pc.get_field_slice("p"), dtype=int)
    p_pc_coords = get_pycutfem_dof_coords(dof_handler_pc, "p")
    p_pc_to_fx = one_to_one_map_coords(p_pc_coords, p_fx_coords)
    P[p_pc_slice] = p_fx_parent[p_pc_to_fx]
    fx_coords_all[p_fx_parent] = p_fx_coords

    for parent_idx, pc_prefix in ((2, "vS"), (3, "u")):
        Wvec, Vec_map = W_fenicsx.sub(parent_idx).collapse()
        W0, V0_map = Wvec.sub(0).collapse()
        W1, V1_map = Wvec.sub(1).collapse()
        vec_fx_coords = W0.tabulate_dof_coordinates()[:, :2]
        vec_fx_parent0 = np.asarray(Vec_map, dtype=int)[np.asarray(V0_map, dtype=int)]
        vec_fx_parent1 = np.asarray(Vec_map, dtype=int)[np.asarray(V1_map, dtype=int)]
        pc_field0 = f"{pc_prefix}_x"
        pc_field1 = f"{pc_prefix}_y"
        pc_slice0 = np.asarray(dof_handler_pc.get_field_slice(pc_field0), dtype=int)
        pc_slice1 = np.asarray(dof_handler_pc.get_field_slice(pc_field1), dtype=int)
        pc_coords0 = get_pycutfem_dof_coords(dof_handler_pc, pc_field0)
        pc_to_fx = one_to_one_map_coords(pc_coords0, vec_fx_coords)
        P[pc_slice0] = vec_fx_parent0[pc_to_fx]
        P[pc_slice1] = vec_fx_parent1[pc_to_fx]
        fx_coords_all[vec_fx_parent0] = vec_fx_coords
        fx_coords_all[vec_fx_parent1] = W1.tabulate_dof_coordinates()[:, :2]

    scalar_parents = [(4, "phi"), (5, "alpha")]
    next_idx = 6
    if include_mu_alpha:
        scalar_parents.append((next_idx, "mu_alpha"))
        next_idx += 1
    scalar_parents.extend(((next_idx, "d"), (next_idx + 1, "S"), (next_idx + 2, "X")))
    for parent_idx, fld in scalar_parents:
        Ws, S_map_fx = W_fenicsx.sub(parent_idx).collapse()
        fx_coords = Ws.tabulate_dof_coordinates()[:, :2]
        fx_parent = np.asarray(S_map_fx, dtype=int)
        pc_slice = np.asarray(dof_handler_pc.get_field_slice(fld), dtype=int)
        pc_coords = get_pycutfem_dof_coords(dof_handler_pc, fld)
        pc_to_fx = one_to_one_map_coords(pc_coords, fx_coords)
        P[pc_slice] = fx_parent[pc_to_fx]
        fx_coords_all[fx_parent] = fx_coords

    print("Biofilm H(div) DoF map discovered successfully.")
    return {
        "P": P,
        "fx_coords_all": fx_coords_all,
        "v_pc_slice": v_pc_slice,
        "v_pc_to_fx_collapsed": np.asarray(v_pc_to_fx, dtype=int),
    }


def create_true_dof_transform_biofilm_hdiv_rt1_single_cell(
    dof_handler_pc: DofHandler,
    W_fenicsx,
    *,
    include_mu_alpha: bool = False,
):
    """
    Build a full mixed-space transform for the single-cell RT1 biofilm harness.

    The RT block requires a genuine local basis transform rather than the
    signed-permutation map used for RT0.
    """
    base = create_true_dof_map_biofilm_hdiv(
        dof_handler_pc,
        W_fenicsx,
        include_mu_alpha=include_mu_alpha,
    )
    mixed_element_pc = dof_handler_pc.mixed_element
    if int(getattr(mixed_element_pc._ref["v"], "k", -1)) != 1 or int(mixed_element_pc.mesh.n_elements) != 1:
        raise NotImplementedError("RT1 whole-domain transform helper currently targets a single quadrilateral element.")

    n_pc = int(dof_handler_pc.total_dofs)
    n_fx = int(W_fenicsx.dofmap.index_map.size_global)
    T_basis = np.zeros((n_fx, n_pc), dtype=float)
    T_coeff = np.zeros((n_fx, n_pc), dtype=float)

    P = np.asarray(base["P"], dtype=int)
    v_pc_slice = np.asarray(base["v_pc_slice"], dtype=int)
    mask_nonrt = np.ones((n_pc,), dtype=bool)
    mask_nonrt[v_pc_slice] = False
    nonrt_cols = np.nonzero(mask_nonrt)[0]
    T_basis[P[nonrt_cols], nonrt_cols] = 1.0
    T_coeff[P[nonrt_cols], nonrt_cols] = 1.0

    Vv_fx, Vv_to_W = W_fenicsx.sub(0).collapse()
    v_fx_parent = np.asarray(Vv_to_W, dtype=int)
    B_pc_to_fx = _build_rt1_local_transform(mixed_element_pc, Vv_fx, field="v")
    T_basis[np.ix_(v_fx_parent, v_pc_slice)] = np.asarray(B_pc_to_fx, dtype=float)
    T_coeff[np.ix_(v_fx_parent, v_pc_slice)] = np.asarray(B_pc_to_fx, dtype=float)

    base["T"] = T_basis
    base["T_coeff"] = T_coeff
    return base


def initialize_biofilm_hdiv_functions(pc, fenicsx, dof_handler_pc, map_info, *, include_mu_alpha: bool = False):
    print("Initializing biofilm H(div) state vectors in pycutfem and FEniCSx...")

    def v_k_init(x):
        return 0.1 + 0.2 * x[0] + 0.05 * x[1]

    def v_n_init(x):
        return 0.07 + 0.11 * x[0] - 0.03 * x[1]

    def vS_k_init(x):
        return [0.06 + 0.01 * x[0] - 0.03 * x[1], -0.02 + 0.02 * x[0] + 0.01 * x[1]]

    def vS_n_init(x):
        vv = vS_k_init(x)
        return [0.6 * vv[0], 0.6 * vv[1]]

    def u_k_init(x):
        return [0.02 * x[0] * (1.0 - x[0]) + 0.01 * x[1], -0.01 * x[1] * (1.0 - x[1]) + 0.005 * x[0]]

    def u_n_init(x):
        uu = u_k_init(x)
        return [0.6 * uu[0], 0.6 * uu[1]]

    def p_k_init(x):
        return 0.2 * x[0] - 0.1 * x[1] + 0.3

    def p_n_init(x):
        return 0.5 * p_k_init(x)

    def phi_k_init(x):
        return 0.72 + 0.03 * x[0] - 0.02 * x[1]

    def phi_n_init(x):
        return 0.70 + 0.02 * x[0] - 0.01 * x[1]

    def alpha_k_init(x):
        return 0.55 + 0.03 * x[0] + 0.01 * x[1]

    def alpha_n_init(x):
        return 0.50 + 0.02 * x[0] + 0.01 * x[1]

    def mu_alpha_k_init(x):
        return 0.12 + 0.01 * x[0] - 0.015 * x[1]

    def mu_alpha_n_init(x):
        return 0.10 + 0.008 * x[0] - 0.01 * x[1]

    def d_k_init(x):
        return 0.18 + 0.03 * x[0] - 0.02 * x[1]

    def d_n_init(x):
        return 0.12 + 0.02 * x[0] - 0.01 * x[1]

    def S_k_init(x):
        return 0.22 + 0.05 * x[0] + 0.02 * x[1]

    def S_n_init(x):
        return 0.20 + 0.03 * x[0] + 0.01 * x[1]

    def X_k_init(x):
        return 0.08 + 0.01 * x[0] + 0.015 * x[1]

    def X_n_init(x):
        return 0.05 + 0.01 * x[0] + 0.01 * x[1]

    v_coords = get_pycutfem_dof_coords(dof_handler_pc, "v")
    pc["v_k"].nodal_values[:] = np.asarray([v_k_init(xy) for xy in v_coords], dtype=float)
    pc["v_n"].nodal_values[:] = np.asarray([v_n_init(xy) for xy in v_coords], dtype=float)
    pc["vS_k"].set_values_from_function(lambda x, y: vS_k_init([x, y]))
    pc["vS_n"].set_values_from_function(lambda x, y: vS_n_init([x, y]))
    pc["u_k"].set_values_from_function(lambda x, y: u_k_init([x, y]))
    pc["u_n"].set_values_from_function(lambda x, y: u_n_init([x, y]))
    pc["p_k"].set_values_from_function(lambda x, y: p_k_init([x, y]))
    pc["p_n"].set_values_from_function(lambda x, y: p_n_init([x, y]))
    pc["phi_k"].set_values_from_function(lambda x, y: phi_k_init([x, y]))
    pc["phi_n"].set_values_from_function(lambda x, y: phi_n_init([x, y]))
    pc["alpha_k"].set_values_from_function(lambda x, y: alpha_k_init([x, y]))
    pc["alpha_n"].set_values_from_function(lambda x, y: alpha_n_init([x, y]))
    if include_mu_alpha:
        pc["mu_alpha_k"].set_values_from_function(lambda x, y: mu_alpha_k_init([x, y]))
        pc["mu_alpha_n"].set_values_from_function(lambda x, y: mu_alpha_n_init([x, y]))
    pc["d_k"].set_values_from_function(lambda x, y: d_k_init([x, y]))
    pc["d_n"].set_values_from_function(lambda x, y: d_n_init([x, y]))
    pc["S_k"].set_values_from_function(lambda x, y: S_k_init([x, y]))
    pc["S_n"].set_values_from_function(lambda x, y: S_n_init([x, y]))
    pc["X_k"].set_values_from_function(lambda x, y: X_k_init([x, y]))
    pc["X_n"].set_values_from_function(lambda x, y: X_n_init([x, y]))

    x_pc_k = _biofilm_hdiv_state_vector(pc, dof_handler_pc, suffix="k", include_mu_alpha=include_mu_alpha)
    x_pc_n = _biofilm_hdiv_state_vector(pc, dof_handler_pc, suffix="n", include_mu_alpha=include_mu_alpha)
    fenicsx["w_k"].x.array[:] = 0.0
    fenicsx["w_n"].x.array[:] = 0.0
    transform = map_info.get("T", None)
    transform_coeff = map_info.get("T_coeff", transform)
    if transform_coeff is not None:
        T = np.asarray(transform_coeff, dtype=float)
        fenicsx["w_k"].x.array[:] = np.asarray(T @ x_pc_k, dtype=float)
        fenicsx["w_n"].x.array[:] = np.asarray(T @ x_pc_n, dtype=float)
    else:
        sign_map = np.asarray(map_info["sign_map"], dtype=float)
        P_map = np.asarray(map_info["P"], dtype=int)
        fenicsx["w_k"].x.array[P_map] = sign_map * x_pc_k
        fenicsx["w_n"].x.array[P_map] = sign_map * x_pc_n
    fenicsx["w_k"].x.scatter_forward()
    fenicsx["w_n"].x.scatter_forward()


def initialize_biofilm_functions(pc, fenicsx, dof_handler_pc, P_map, *, include_mu_alpha: bool = False):
    if include_mu_alpha:
        print("Initializing biofilm (v,p,vS,u,phi,alpha,mu_alpha,d,S,X) functions in pycutfem and FEniCSx...")
    else:
        print("Initializing biofilm (v,p,vS,u,phi,alpha,d,S,X) functions in pycutfem and FEniCSx...")

    # Vector fields (Q2)
    def v_k_init(x):
        return [0.1 + 0.2 * x[0] + 0.05 * x[1], -0.03 + 0.1 * x[0] - 0.07 * x[1]]

    def v_n_init(x):
        vv = v_k_init(x)
        return [0.7 * vv[0], 0.7 * vv[1]]

    def vS_k_init(x):
        return [0.06 + 0.01 * x[0] - 0.03 * x[1], -0.02 + 0.02 * x[0] + 0.01 * x[1]]

    def vS_n_init(x):
        vv = vS_k_init(x)
        return [0.6 * vv[0], 0.6 * vv[1]]

    def u_k_init(x):
        # keep gradients small
        return [0.02 * x[0] * (1.0 - x[0]) + 0.01 * x[1], -0.01 * x[1] * (1.0 - x[1]) + 0.005 * x[0]]

    def u_n_init(x):
        uu = u_k_init(x)
        return [0.6 * uu[0], 0.6 * uu[1]]

    # Scalar fields (Q1) – keep in safe ranges
    def p_k_init(x):
        return 0.2 * x[0] - 0.1 * x[1] + 0.3

    def p_n_init(x):
        return 0.5 * p_k_init(x)

    def phi_k_init(x):
        return 0.72 + 0.03 * x[0] - 0.02 * x[1]

    def phi_n_init(x):
        return 0.70 + 0.02 * x[0] - 0.01 * x[1]

    def alpha_k_init(x):
        return 0.55 + 0.03 * x[0] + 0.01 * x[1]

    def alpha_n_init(x):
        return 0.50 + 0.02 * x[0] + 0.01 * x[1]

    def mu_alpha_k_init(x):
        return 0.12 + 0.01 * x[0] - 0.015 * x[1]

    def mu_alpha_n_init(x):
        return 0.10 + 0.008 * x[0] - 0.01 * x[1]

    def d_k_init(x):
        return 0.18 + 0.03 * x[0] - 0.02 * x[1]

    def d_n_init(x):
        return 0.12 + 0.02 * x[0] - 0.01 * x[1]

    def S_k_init(x):
        return 0.22 + 0.05 * x[0] + 0.02 * x[1]

    def S_n_init(x):
        return 0.20 + 0.03 * x[0] + 0.01 * x[1]

    def X_k_init(x):
        return 0.08 + 0.01 * x[0] + 0.015 * x[1]

    def X_n_init(x):
        return 0.05 + 0.01 * x[0] + 0.01 * x[1]

    # --- pycutfem ---
    pc["v_k"].set_values_from_function(lambda x, y: v_k_init([x, y]))
    pc["v_n"].set_values_from_function(lambda x, y: v_n_init([x, y]))
    pc["vS_k"].set_values_from_function(lambda x, y: vS_k_init([x, y]))
    pc["vS_n"].set_values_from_function(lambda x, y: vS_n_init([x, y]))
    pc["u_k"].set_values_from_function(lambda x, y: u_k_init([x, y]))
    pc["u_n"].set_values_from_function(lambda x, y: u_n_init([x, y]))
    pc["p_k"].set_values_from_function(lambda x, y: p_k_init([x, y]))
    pc["p_n"].set_values_from_function(lambda x, y: p_n_init([x, y]))
    pc["phi_k"].set_values_from_function(lambda x, y: phi_k_init([x, y]))
    pc["phi_n"].set_values_from_function(lambda x, y: phi_n_init([x, y]))
    pc["alpha_k"].set_values_from_function(lambda x, y: alpha_k_init([x, y]))
    pc["alpha_n"].set_values_from_function(lambda x, y: alpha_n_init([x, y]))
    if include_mu_alpha:
        pc["mu_alpha_k"].set_values_from_function(lambda x, y: mu_alpha_k_init([x, y]))
        pc["mu_alpha_n"].set_values_from_function(lambda x, y: mu_alpha_n_init([x, y]))
    pc["d_k"].set_values_from_function(lambda x, y: d_k_init([x, y]))
    pc["d_n"].set_values_from_function(lambda x, y: d_n_init([x, y]))
    pc["S_k"].set_values_from_function(lambda x, y: S_k_init([x, y]))
    pc["S_n"].set_values_from_function(lambda x, y: S_n_init([x, y]))
    pc["X_k"].set_values_from_function(lambda x, y: X_k_init([x, y]))
    pc["X_n"].set_values_from_function(lambda x, y: X_n_init([x, y]))

    # --- FEniCSx ---
    W = fenicsx["W"]
    w_k = fenicsx["w_k"]
    w_n = fenicsx["w_n"]

    Wv, Vv_to_W = W.sub(0).collapse()
    Wp, P_to_W = W.sub(1).collapse()
    WvS, VvS_to_W = W.sub(2).collapse()
    Wu, Vu_to_W = W.sub(3).collapse()
    Wphi, Phi_to_W = W.sub(4).collapse()
    Walpha, Alpha_to_W = W.sub(5).collapse()
    if include_mu_alpha:
        Wmu_alpha, Mu_to_W = W.sub(6).collapse()
        d_idx = 7
    else:
        Wmu_alpha, Mu_to_W = None, None
        d_idx = 6
    s_idx = d_idx + 1
    x_idx = d_idx + 2
    Wd, D_to_W = W.sub(d_idx).collapse()
    WS, S_to_W = W.sub(s_idx).collapse()
    WX, X_to_W = W.sub(x_idx).collapse()

    w_k.x.array[Vv_to_W] = w_k.sub(0).debug_interpolate(v_k_init)
    w_k.x.array[P_to_W] = w_k.sub(1).debug_interpolate(p_k_init)
    w_k.x.array[VvS_to_W] = w_k.sub(2).debug_interpolate(vS_k_init)
    w_k.x.array[Vu_to_W] = w_k.sub(3).debug_interpolate(u_k_init)
    w_k.x.array[Phi_to_W] = w_k.sub(4).debug_interpolate(phi_k_init)
    w_k.x.array[Alpha_to_W] = w_k.sub(5).debug_interpolate(alpha_k_init)
    if include_mu_alpha:
        w_k.x.array[Mu_to_W] = w_k.sub(6).debug_interpolate(mu_alpha_k_init)
    w_k.x.array[D_to_W] = w_k.sub(d_idx).debug_interpolate(d_k_init)
    w_k.x.array[S_to_W] = w_k.sub(s_idx).debug_interpolate(S_k_init)
    w_k.x.array[X_to_W] = w_k.sub(x_idx).debug_interpolate(X_k_init)
    w_k.x.scatter_forward()

    w_n.x.array[Vv_to_W] = w_n.sub(0).debug_interpolate(v_n_init)
    w_n.x.array[P_to_W] = w_n.sub(1).debug_interpolate(p_n_init)
    w_n.x.array[VvS_to_W] = w_n.sub(2).debug_interpolate(vS_n_init)
    w_n.x.array[Vu_to_W] = w_n.sub(3).debug_interpolate(u_n_init)
    w_n.x.array[Phi_to_W] = w_n.sub(4).debug_interpolate(phi_n_init)
    w_n.x.array[Alpha_to_W] = w_n.sub(5).debug_interpolate(alpha_n_init)
    if include_mu_alpha:
        w_n.x.array[Mu_to_W] = w_n.sub(6).debug_interpolate(mu_alpha_n_init)
    w_n.x.array[D_to_W] = w_n.sub(d_idx).debug_interpolate(d_n_init)
    w_n.x.array[S_to_W] = w_n.sub(s_idx).debug_interpolate(S_n_init)
    w_n.x.array[X_to_W] = w_n.sub(x_idx).debug_interpolate(X_n_init)
    w_n.x.scatter_forward()

    # Basic sanity: value sets match for v_k (unordered)
    np.testing.assert_allclose(np.sort(pc["v_k"].nodal_values), np.sort(w_k.x.array[Vv_to_W]), rtol=1e-8, atol=1e-12)
    print("✅ Biofilm initialization: nodal value sets match (v_k).")


def run_biofilm_comparison():
    nx = int(os.environ.get("COMP_FENICS_NX", "2"))
    ny = int(os.environ.get("COMP_FENICS_NY", os.environ.get("COMP_FENICS_NX", "2")))
    pc, dof_handler_pc, fenicsx = setup_biofilm_problems(
        nx=nx,
        ny=ny,
    )
    W_fx = fenicsx["W"]
    include_mu_alpha = bool(pc.get("alpha_ch_enabled", False))
    fluid_space = str(pc.get("fluid_space", "cg")).strip().lower()
    fluid_hdiv_order = int(pc.get("fluid_hdiv_order", 0))
    use_sparse_compare = _env_truthy("COMP_FENICS_SPARSE_COMPARE", False)
    sign_map = np.ones(dof_handler_pc.total_dofs, dtype=float)
    fx_coords_all = None
    transform_map = None
    if fluid_space == "hdiv":
        if int(fluid_hdiv_order) == 1 and int(nx) == 1 and int(ny) == 1:
            map_info = create_true_dof_transform_biofilm_hdiv_rt1_single_cell(
                dof_handler_pc,
                W_fx,
                include_mu_alpha=include_mu_alpha,
            )
            transform_map = np.asarray(map_info["T"], dtype=float)
            P_map = None
            fx_coords_all = np.asarray(map_info["fx_coords_all"], dtype=float)
            initialize_biofilm_hdiv_functions(
                pc,
                fenicsx,
                dof_handler_pc,
                map_info,
                include_mu_alpha=include_mu_alpha,
            )
        else:
            map_info = create_true_dof_map_biofilm_hdiv(dof_handler_pc, W_fx, include_mu_alpha=include_mu_alpha)
            P_map = np.asarray(map_info["P"], dtype=int)
            fx_coords_all = np.asarray(map_info["fx_coords_all"], dtype=float)
            v_pc_slice = np.asarray(map_info["v_pc_slice"], dtype=int)
            v_pc_to_fx_collapsed = np.asarray(map_info["v_pc_to_fx_collapsed"], dtype=int)

            qdeg_map = int(os.environ.get("COMP_FENICS_QDEG", "6"))
            M_pc_full, _ = assemble_form(
                Equation(inner(pc["dv"], pc["w"]) * dx(metadata={"q": qdeg_map}), None),
                dof_handler_pc,
                bcs=[],
                backend="python",
            )
            M_pc = M_pc_full.tocsr()[v_pc_slice, :][:, v_pc_slice].toarray()

            Vv_fx, _ = W_fx.sub(0).collapse()
            dvv_fx = ufl.TrialFunction(Vv_fx)
            wv_fx = ufl.TestFunction(Vv_fx)
            M_fx_form = dolfinx.fem.form(ufl.inner(dvv_fx, wv_fx) * ufl.dx(metadata={"quadrature_degree": int(max(1, 2 * qdeg_map - 1))}))
            A_mass_fx = dolfinx.fem.petsc.assemble_matrix(M_fx_form)
            A_mass_fx.assemble()
            indptr, indices, data = A_mass_fx.getValuesCSR()
            M_fx = csr_matrix((data, indices, indptr), shape=A_mass_fx.getSize()).tocsr()
            M_fx_perm = M_fx[v_pc_to_fx_collapsed, :][:, v_pc_to_fx_collapsed].toarray()

            sign_map[v_pc_slice] = _recover_sign_congruence(M_pc, M_fx_perm)

            p_pc_slice = np.asarray(dof_handler_pc.get_field_slice("p"), dtype=int)
            C_pc_full, _ = assemble_form(
                Equation(pc["q"] * div(pc["dv"]) * dx(metadata={"q": qdeg_map}), None),
                dof_handler_pc,
                bcs=[],
                backend="python",
            )
            C_pc = C_pc_full.tocsr()[p_pc_slice, :][:, v_pc_slice].toarray()

            dw_anchor_fx = ufl.TrialFunction(W_fx)
            w_anchor_fx = ufl.TestFunction(W_fx)
            dv_anchor_fx = ufl.split(dw_anchor_fx)[0]
            q_anchor_fx = ufl.split(w_anchor_fx)[1]
            C_fx_form = dolfinx.fem.form(
                q_anchor_fx * ufl.div(dv_anchor_fx) * ufl.dx(metadata={"quadrature_degree": int(max(1, 2 * qdeg_map - 1))})
            )
            A_cpl_fx = dolfinx.fem.petsc.assemble_matrix(C_fx_form)
            A_cpl_fx.assemble()
            indptr, indices, data = A_cpl_fx.getValuesCSR()
            C_fx_full = csr_matrix((data, indices, indptr), shape=A_cpl_fx.getSize()).tocsr()
            C_fx = C_fx_full[np.asarray(P_map[p_pc_slice], dtype=int), :][:, np.asarray(P_map[v_pc_slice], dtype=int)].toarray()
            score = float(np.sum(C_pc * (C_fx * sign_map[v_pc_slice][None, :])))
            if score < 0.0:
                sign_map[v_pc_slice] *= -1.0

            map_info["sign_map"] = sign_map
            initialize_biofilm_hdiv_functions(
                pc,
                fenicsx,
                dof_handler_pc,
                map_info,
                include_mu_alpha=include_mu_alpha,
            )
    else:
        P_map = create_true_dof_map_biofilm(dof_handler_pc, W_fx, include_mu_alpha=include_mu_alpha)
        initialize_biofilm_functions(pc, fenicsx, dof_handler_pc, P_map, include_mu_alpha=include_mu_alpha)

    # Split mixed functions and arguments on the FEniCSx side
    dw_fx = ufl.TrialFunction(W_fx)
    w_test_fx = ufl.TestFunction(W_fx)
    if include_mu_alpha:
        dv_fx, dp_fx, dvS_fx, du_fx, dphi_fx, dalpha_fx, dmu_alpha_fx, dd_fx, dS_fx, dX_fx = ufl.split(dw_fx)
        w_fx, q_fx, eta_vS_fx, eta_u_fx, zeta_fx, xi_fx, eta_mu_fx, chi_fx, r_fx, y_fx = ufl.split(w_test_fx)

        v_k_fx, p_k_fx, vS_k_fx, u_k_fx, phi_k_fx, alpha_k_fx, mu_alpha_k_fx, d_k_fx, S_k_fx, X_k_fx = ufl.split(fenicsx["w_k"])
        v_n_fx, p_n_fx, vS_n_fx, u_n_fx, phi_n_fx, alpha_n_fx, mu_alpha_n_fx, d_n_fx, S_n_fx, X_n_fx = ufl.split(fenicsx["w_n"])
    else:
        dv_fx, dp_fx, dvS_fx, du_fx, dphi_fx, dalpha_fx, dd_fx, dS_fx, dX_fx = ufl.split(dw_fx)
        w_fx, q_fx, eta_vS_fx, eta_u_fx, zeta_fx, xi_fx, chi_fx, r_fx, y_fx = ufl.split(w_test_fx)
        dmu_alpha_fx = None
        eta_mu_fx = None

        v_k_fx, p_k_fx, vS_k_fx, u_k_fx, phi_k_fx, alpha_k_fx, d_k_fx, S_k_fx, X_k_fx = ufl.split(fenicsx["w_k"])
        v_n_fx, p_n_fx, vS_n_fx, u_n_fx, phi_n_fx, alpha_n_fx, d_n_fx, S_n_fx, X_n_fx = ufl.split(fenicsx["w_n"])
        mu_alpha_k_fx = None
        mu_alpha_n_fx = None

    # Parameters
    rho_f_fx = fenicsx["rho_f"]
    mu_f_fx = fenicsx["mu_f"]
    kappa_inv_fx = fenicsx["kappa_inv"]
    kappa_inv_model_key = str(fenicsx.get("kappa_inv_model", "spatial")).strip().lower()
    gdim = int(fenicsx["mesh"].geometry.dim)
    mu_s_fx = fenicsx["mu_s"]
    lambda_s_fx = fenicsx["lambda_s"]
    dt_fx = fenicsx["dt"]
    theta_fx = fenicsx["theta"]
    fluid_convection_key = str(fenicsx.get("fluid_convection", "full")).strip().lower()
    alpha_advect_with_key = str(fenicsx.get("alpha_advect_with", "vS")).strip().lower()
    alpha_advection_form_key = str(fenicsx.get("alpha_advection_form", "advective")).strip().lower()
    gamma_div_fx = fenicsx["gamma_div"]
    gamma_u_fx = fenicsx["gamma_u"]
    u_extension_mode_fx = str(fenicsx.get("u_extension_mode", "l2")).strip().lower()
    gamma_u_pin_fx = fenicsx["gamma_u_pin"]
    v_supg_fx = fenicsx["v_supg"]
    v_supg_mode_key = str(fenicsx.get("v_supg_mode", "streamline")).strip().lower()
    v_supg_c_nu_fx = fenicsx["v_supg_c_nu"]

    D_phi_fx = fenicsx["D_phi"]
    gamma_phi_fx = fenicsx["gamma_phi"]
    D_alpha_fx = fenicsx["D_alpha"]
    D_X_fx = fenicsx["D_X"]
    D_S_fx = fenicsx["D_S"]
    mu_max_fx = fenicsx["mu_max"]
    K_S_fx = fenicsx["K_S"]
    k_g_fx = fenicsx["k_g"]
    k_d_fx = fenicsx["k_d"]
    Y_fx = fenicsx["Y"]
    rho_s_star_fx = fenicsx["rho_s_star"]
    k_det_fx = fenicsx["k_det"]
    damage_k_fx = fenicsx["damage_k"]
    damage_sigma_cr_fx = fenicsx["damage_sigma_cr"]
    damage_m_fx = fenicsx["damage_m"]
    damage_D_fx = fenicsx["damage_D"]
    damage_gamma_out_fx = fenicsx["damage_gamma_out"]
    damage_eta_pos_fx = fenicsx["damage_eta_pos"]
    damage_kappa_stiff_fx = fenicsx["damage_kappa_stiff"]
    damage_kappa_perm_fx = fenicsx["damage_kappa_perm"]
    damage_model_key = str(pc["damage_model"]).strip().lower()
    if damage_model_key not in {"kinetic", "legacy"}:
        raise ValueError(
            f"FEniCSx side of run_biofilm_comparison currently supports only kinetic damage model, got {pc['damage_model']!r}."
        )

    # Measures
    # NOTE: pycutfem's volume quadrature metadata uses `q` as a Gauss–Legendre
    # *order* (number of points per 1D rule). In dolfinx/ufl, `quadrature_degree`
    # is a polynomial degree hint; for tensor-product Gauss rules on quads the
    # smallest degree that yields `q` points is `2*q - 1`.
    qdeg = int(os.environ.get("COMP_FENICS_QDEG", "6"))
    qdeg_fx = int(os.environ.get("COMP_FENICS_QDEG_FX", str(max(1, 2 * qdeg - 1))))
    dx_pc = dx(metadata={"q": qdeg})
    dx_fx = ufl.dx(metadata={"quadrature_degree": qdeg_fx})
    ds_hdiv_tangential_pc = None
    ds_hdiv_tangential_fx = None
    if fluid_space == "hdiv" and bool(pc.get("hdiv_tangential_dirichlet", False)):
        edge_mask = np.zeros(len(pc["mesh"].edges_list), dtype=bool)
        for edge in pc["mesh"].edges_list:
            if edge.right is not None:
                continue
            edge_nodes = getattr(edge, "all_nodes", None) or edge.nodes
            coords = np.asarray(pc["mesh"].nodes_x_y_pos[list(edge_nodes)], dtype=float)
            mid = coords[int(coords.shape[0] // 2)] if coords.shape[0] >= 3 else coords.mean(axis=0)
            x_mid = float(mid[0])
            y_mid = float(mid[1])
            if abs(x_mid - 0.0) <= 1.0e-12 or abs(y_mid - 0.0) <= 1.0e-12 or abs(y_mid - 1.0) <= 1.0e-12:
                edge_mask[int(edge.gid)] = True
        ds_hdiv_tangential_pc = dS(
            defined_on=BitSet(edge_mask),
            metadata={"q": qdeg},
        )
        if fenicsx.get("hdiv_tangential_facet_tags") is not None:
            ds_hdiv_tangential_fx = ufl.Measure(
                "ds",
                domain=fenicsx["mesh"],
                subdomain_data=fenicsx["hdiv_tangential_facet_tags"],
                metadata={"quadrature_degree": qdeg_fx},
            )
    print(f"[biofilm] quadrature: pycutfem q={qdeg}, fenics quadrature_degree={qdeg_fx}")
    Jdet_fx = ufl.JacobianDeterminant(fenicsx["mesh"])
    # Match pycutfem MeshSize() on quads.
    #
    # pycutfem uses a [-1,1]^2 reference cell and lowers MeshSize() to
    #   2 * sqrt(|detJ_pc|),
    # which equals the physical element size h for an affine square.
    #
    # FEniCSx/Basix quadrilaterals use the [0,1]^2 reference cell, so the same
    # physical square has detJ_fx = h^2 instead of h^2/4. The equivalent pointwise
    # mesh size is therefore sqrt(|detJ_fx|), not 2*sqrt(|detJ_fx|).
    h_fx = ufl.sqrt(ufl.sqrt(Jdet_fx * Jdet_fx + 1.0e-16))
    inv_h2_fx = 1.0 / (h_fx * h_fx)

    def eps_fx(v):
        return ufl.sym(ufl.grad(v))

    def grad_component_fx(vec_expr, i: int, j: int):
        return vec_expr[i].dx(j)

    def advected_grad_fx(vec_expr, adv_vec):
        comps = []
        for i in range(gdim):
            comp = 0
            for j in range(gdim):
                comp += grad_component_fx(vec_expr, i, j) * adv_vec[j]
            comps.append(comp)
        return ufl.as_vector(comps)

    def _laplace_components_fx(vec_expr):
        comps = []
        for i in range(gdim):
            comps.append(vec_expr[i].dx(0).dx(0) + vec_expr[i].dx(1).dx(1))
        return tuple(comps)

    def _grad_div_components_hdiv_fx(v_expr):
        if int(gdim) != 2:
            raise NotImplementedError("Biofilm H(div) FEniCSx comparison currently targets 2D only.")
        return (
            v_expr[0].dx(0).dx(0) + v_expr[1].dx(0).dx(1),
            v_expr[0].dx(0).dx(1) + v_expr[1].dx(1).dx(1),
        )

    def strong_div_2mu_eps_hdiv_fx(v_expr, mu_expr):
        grad_mu = ufl.grad(mu_expr)
        lap_v = _laplace_components_fx(v_expr)
        grad_div_v = _grad_div_components_hdiv_fx(v_expr)
        comps = []
        for i in range(gdim):
            comp = mu_expr * (lap_v[i] + grad_div_v[i])
            for j in range(gdim):
                eps_ij = 0.5 * (grad_component_fx(v_expr, i, j) + grad_component_fx(v_expr, j, i))
                comp += 2.0 * eps_ij * grad_mu[j]
            comps.append(comp)
        return ufl.as_vector(comps)

    def tangent_fx(n_expr):
        return ufl.as_vector((n_expr[1], -n_expr[0]))

    def tangential_component_fx(vec_expr, n_expr):
        return ufl.dot(vec_expr, tangent_fx(n_expr))

    def tangential_viscous_traction_fx(v_expr, mu_expr, n_expr):
        return ufl.dot(2.0 * mu_expr * eps_fx(v_expr) * n_expr, tangent_fx(n_expr))

    def one_minus(a):
        return 1.0 - a

    def capacity(alpha, phi):
        return one_minus(alpha) + alpha * phi

    def monod(S):
        return mu_max_fx * (S / (S + K_S_fx))

    def Pi_over_rho_s(S, phi, alpha):
        return (monod(S) - k_d_fx) * one_minus(phi) * alpha

    def G(S, phi):
        return k_g_fx * monod(S) * one_minus(phi)

    # Skeleton velocity (Eulerian)
    div_vS_k_fx = ufl.div(vS_k_fx)
    div_vS_n_fx = ufl.div(vS_n_fx)

    # Coefficients
    C_k_fx = capacity(alpha_k_fx, phi_k_fx)
    C_n_fx = capacity(alpha_n_fx, phi_n_fx)
    B_k_fx = alpha_k_fx * one_minus(phi_k_fx)
    B_n_fx = alpha_n_fx * one_minus(phi_n_fx)

    rho_k_fx = rho_f_fx * C_k_fx
    rho_n_fx = rho_f_fx * C_n_fx
    mu_k_fx = mu_f_fx * C_k_fx  # phi_mu choice
    mu_n_fx = mu_f_fx * C_n_fx

    # Damage degradation factors (Miehe-type) used by momentum/skeleton drag and solid stiffness.
    one_m_d_k_fx = one_minus(d_k_fx)
    one_m_d_n_fx = one_minus(d_n_fx)
    g_stiff_k_fx = (1.0 - damage_kappa_stiff_fx) * (one_m_d_k_fx * one_m_d_k_fx) + damage_kappa_stiff_fx
    g_stiff_n_fx = (1.0 - damage_kappa_stiff_fx) * (one_m_d_n_fx * one_m_d_n_fx) + damage_kappa_stiff_fx
    g_perm_k_fx = (1.0 - damage_kappa_perm_fx) * (one_m_d_k_fx * one_m_d_k_fx) + damage_kappa_perm_fx
    g_perm_n_fx = (1.0 - damage_kappa_perm_fx) * (one_m_d_n_fx * one_m_d_n_fx) + damage_kappa_perm_fx

    beta_coeff_k_fx = alpha_k_fx * mu_f_fx * (phi_k_fx * phi_k_fx) * g_perm_k_fx
    beta_coeff_n_fx = alpha_n_fx * mu_f_fx * (phi_n_fx * phi_n_fx) * g_perm_n_fx
    beta_k_fx = beta_coeff_k_fx * kappa_inv_fx
    beta_n_fx = beta_coeff_n_fx * kappa_inv_fx
    use_refmap_drag_fx = kappa_inv_model_key in {"refmap", "eulerian_refmap", "eulerian", "reference_map", "reference-map"}
    if use_refmap_drag_fx:
        I_fx = ufl.Identity(gdim)
        Finv_k_ref_fx = I_fx - ufl.grad(u_k_fx)
        Finv_n_ref_fx = I_fx - ufl.grad(u_n_fx)
        Fk_ref_fx = ufl.inv(Finv_k_ref_fx)
        Fn_ref_fx = ufl.inv(Finv_n_ref_fx)
        Jk_ref_fx = ufl.det(Fk_ref_fx)
        Jn_ref_fx = ufl.det(Fn_ref_fx)
        Kinv_ref_fx = kappa_inv_fx * I_fx
        k_inv_k_fx = Jk_ref_fx * ufl.dot(Finv_k_ref_fx.T, ufl.dot(Kinv_ref_fx, Finv_k_ref_fx))
        k_inv_n_fx = Jn_ref_fx * ufl.dot(Finv_n_ref_fx.T, ufl.dot(Kinv_ref_fx, Finv_n_ref_fx))
        kdrag_k_fx = ufl.dot(k_inv_k_fx, v_k_fx - vS_k_fx)
        kdrag_n_fx = ufl.dot(k_inv_n_fx, v_n_fx - vS_n_fx)
    elif kappa_inv_model_key in {"spatial", "constant", "const"}:
        kdrag_k_fx = None
        kdrag_n_fx = None
    else:
        raise ValueError(f"Unsupported COMP_FENICS_KAPPA_INV_MODEL={kappa_inv_model_key!r} for biofilm comparison.")

    # Detachment coefficient (lagged by v_n)
    eta_n = 1.0e-12
    tau2_fx = ufl.inner(eps_fx(v_n_fx), eps_fx(v_n_fx))
    D_det_prev_fx = k_det_fx * ufl.sqrt(tau2_fx + eta_n)

    # Conservative expanded-divergence helpers (matching biofilm_one_domain.py)
    gradC_k_fx = ufl.grad(alpha_k_fx) * (phi_k_fx - 1.0) + ufl.grad(phi_k_fx) * alpha_k_fx
    gradC_n_fx = ufl.grad(alpha_n_fx) * (phi_n_fx - 1.0) + ufl.grad(phi_n_fx) * alpha_n_fx
    divCv_k_fx = C_k_fx * ufl.div(v_k_fx) + ufl.dot(gradC_k_fx, v_k_fx)
    divCv_n_fx = C_n_fx * ufl.div(v_n_fx) + ufl.dot(gradC_n_fx, v_n_fx)

    gradB_k_fx = ufl.grad(alpha_k_fx) * one_minus(phi_k_fx) - ufl.grad(phi_k_fx) * alpha_k_fx
    gradB_n_fx = ufl.grad(alpha_n_fx) * one_minus(phi_n_fx) - ufl.grad(phi_n_fx) * alpha_n_fx
    divBvS_k_fx = B_k_fx * div_vS_k_fx + ufl.dot(gradB_k_fx, vS_k_fx)
    divBvS_n_fx = B_n_fx * div_vS_n_fx + ufl.dot(gradB_n_fx, vS_n_fx)

    divF_k_fx = divCv_k_fx + divBvS_k_fx
    divF_n_fx = divCv_n_fx + divBvS_n_fx

    # ------------------------------------------------------------------
    # Residual pieces (fenics)
    # ------------------------------------------------------------------
    vdot_fx = (rho_k_fx * v_k_fx - rho_n_fx * v_n_fx) / dt_fx
    conv_k_fx = ufl.dot(ufl.dot(ufl.grad(v_k_fx), v_k_fx), w_fx)
    conv_n_fx = ufl.dot(ufl.dot(ufl.grad(v_n_fx), v_n_fx), w_fx)
    div_rhov_k_fx = rho_f_fx * divCv_k_fx
    div_rhov_n_fx = rho_f_fx * divCv_n_fx
    div_C_w_fx = C_k_fx * ufl.div(w_fx) + ufl.dot(gradC_k_fx, w_fx)

    r_mom_fx = ufl.inner(vdot_fx, w_fx) * dx_fx
    if fluid_convection_key == "full":
        r_mom_fx += theta_fx * (rho_k_fx * conv_k_fx + div_rhov_k_fx * ufl.dot(v_k_fx, w_fx)) * dx_fx
        r_mom_fx += (1.0 - theta_fx) * (rho_n_fx * conv_n_fx + div_rhov_n_fx * ufl.dot(v_n_fx, w_fx)) * dx_fx
    elif fluid_convection_key in {"lagged", "explicit"}:
        conv_lag_k_fx = ufl.dot(ufl.dot(ufl.grad(v_k_fx), v_n_fx), w_fx)
        r_mom_fx += theta_fx * (rho_n_fx * conv_lag_k_fx + div_rhov_n_fx * ufl.dot(v_k_fx, w_fx)) * dx_fx
        r_mom_fx += (1.0 - theta_fx) * (rho_n_fx * conv_n_fx + div_rhov_n_fx * ufl.dot(v_n_fx, w_fx)) * dx_fx
    elif fluid_convection_key == "imex":
        r_mom_fx += (rho_n_fx * conv_n_fx + div_rhov_n_fx * ufl.dot(v_n_fx, w_fx)) * dx_fx
    elif fluid_convection_key == "off":
        pass
    else:
        raise ValueError(f"Unknown COMP_FENICS_FLUID_CONVECTION={fluid_convection_key!r}.")
    r_mom_fx += 2.0 * theta_fx * mu_k_fx * ufl.inner(eps_fx(v_k_fx), eps_fx(w_fx)) * dx_fx
    r_mom_fx += 2.0 * (1.0 - theta_fx) * mu_n_fx * ufl.inner(eps_fx(v_n_fx), eps_fx(w_fx)) * dx_fx
    r_mom_fx += -p_k_fx * div_C_w_fx * dx_fx
    r_mom_fx += gamma_div_fx * divF_k_fx * div_C_w_fx * dx_fx
    if use_refmap_drag_fx:
        r_mom_fx += beta_coeff_k_fx * ufl.dot(kdrag_k_fx, w_fx) * dx_fx
    else:
        r_mom_fx += beta_k_fx * ufl.dot(v_k_fx, w_fx) * dx_fx
        r_mom_fx += -beta_k_fx * ufl.dot(vS_k_fx, w_fx) * dx_fx
    if ds_hdiv_tangential_fx is not None:
        n_b_fx = ufl.FacetNormal(fenicsx["mesh"])
        vt_k_fx = tangential_component_fx(v_k_fx, n_b_fx)
        vt_n_fx = tangential_component_fx(v_n_fx, n_b_fx)
        wt_fx = tangential_component_fx(w_fx, n_b_fx)
        gap_t_k_fx = vt_k_fx
        gap_t_n_fx = vt_n_fx
        penalty_t_fx = fenicsx["hdiv_tangential_gamma"] / (h_fx + 1.0e-16)
        r_mom_fx += (
            penalty_t_fx * (theta_fx * mu_k_fx * gap_t_k_fx + (1.0 - theta_fx) * mu_n_fx * gap_t_n_fx) * wt_fx
        ) * ds_hdiv_tangential_fx(1)
        if str(fenicsx.get("hdiv_tangential_method", "penalty")) == "nitsche":
            traction_t_k_fx = tangential_viscous_traction_fx(v_k_fx, mu_k_fx, n_b_fx)
            traction_t_n_fx = tangential_viscous_traction_fx(v_n_fx, mu_n_fx, n_b_fx)
            traction_t_w_k_fx = tangential_viscous_traction_fx(w_fx, mu_k_fx, n_b_fx)
            traction_t_w_n_fx = tangential_viscous_traction_fx(w_fx, mu_n_fx, n_b_fx)
            r_mom_fx += (
                -(theta_fx * traction_t_k_fx + (1.0 - theta_fx) * traction_t_n_fx) * wt_fx
                - (theta_fx * traction_t_w_k_fx * gap_t_k_fx + (1.0 - theta_fx) * traction_t_w_n_fx * gap_t_n_fx)
            ) * ds_hdiv_tangential_fx(1)

    if float(v_supg_fx.value) != 0.0 and float(rho_f_fx.value) != 0.0:
        if use_refmap_drag_fx:
            raise NotImplementedError("Biofilm comparison harness does not define SUPG drag-rate scaling for refmap kappa_inv_model.")
        if v_supg_mode_key in {"streamline", "weak", "legacy"}:
            vmag_n_fx = ufl.sqrt(ufl.dot(v_n_fx, v_n_fx) + 1.0e-12)
            rho_safe_n_fx = rho_n_fx + 1.0e-16
            nu_eff_n_fx = mu_n_fx / rho_safe_n_fx
            drag_rate_n_fx = beta_n_fx / rho_safe_n_fx
            tau_v_fx = v_supg_fx / ufl.sqrt(
                (2.0 / dt_fx) ** 2
                + (2.0 * vmag_n_fx / (h_fx + 1.0e-16)) ** 2
                + (v_supg_c_nu_fx * nu_eff_n_fx / (h_fx * h_fx + 1.0e-16)) ** 2
                + drag_rate_n_fx * drag_rate_n_fx
                + 1.0e-16
            )
            adv_v_k_fx = advected_grad_fx(v_k_fx, v_n_fx)
            adv_w_fx = advected_grad_fx(w_fx, v_n_fx)
            r_mom_fx += tau_v_fx * (1.0 - alpha_n_fx) * ufl.inner(adv_v_k_fx, adv_w_fx) * dx_fx
        elif v_supg_mode_key in {"residual", "strong", "strong_residual", "strong-residual"}:
            vmag_k_fx = ufl.sqrt(ufl.dot(v_k_fx, v_k_fx) + 1.0e-12)
            rho_safe_k_fx = rho_k_fx + 1.0e-16
            nu_eff_k_fx = mu_k_fx / rho_safe_k_fx
            drag_rate_k_fx = beta_k_fx / rho_safe_k_fx
            tau_v_fx = v_supg_fx / ufl.sqrt(
                (2.0 / dt_fx) ** 2
                + (2.0 * vmag_k_fx / (h_fx + 1.0e-16)) ** 2
                + (v_supg_c_nu_fx * nu_eff_k_fx / (h_fx * h_fx + 1.0e-16)) ** 2
                + drag_rate_k_fx * drag_rate_k_fx
                + 1.0e-16
            )
            if fluid_space == "hdiv":
                strong_visc_k_fx = strong_div_2mu_eps_hdiv_fx(v_k_fx, mu_k_fx)
            else:
                strong_visc_k_fx = ufl.div(2.0 * mu_k_fx * eps_fx(v_k_fx))
            strong_mom_k_fx = ((rho_k_fx * (v_k_fx - v_n_fx)) / dt_fx) - strong_visc_k_fx + C_k_fx * ufl.grad(p_k_fx)
            if fluid_convection_key == "full":
                strong_mom_k_fx += rho_k_fx * advected_grad_fx(v_k_fx, v_k_fx) + div_rhov_k_fx * v_k_fx
            elif fluid_convection_key in {"lagged", "explicit"}:
                strong_mom_k_fx += rho_n_fx * advected_grad_fx(v_k_fx, v_n_fx) + div_rhov_n_fx * v_k_fx
            elif fluid_convection_key == "imex":
                strong_mom_k_fx += rho_n_fx * advected_grad_fx(v_n_fx, v_n_fx) + div_rhov_n_fx * v_n_fx
            if use_refmap_drag_fx:
                strong_mom_k_fx += beta_coeff_k_fx * kdrag_k_fx
            else:
                strong_mom_k_fx += beta_k_fx * (v_k_fx - vS_k_fx)
            adv_w_fx = advected_grad_fx(w_fx, v_k_fx)
            r_mom_fx += tau_v_fx * (1.0 - alpha_k_fx) * ufl.inner(strong_mom_k_fx, adv_w_fx) * dx_fx
        else:
            raise ValueError(f"Unknown COMP_FENICS_V_SUPG_MODE={v_supg_mode_key!r}.")

    # Mass/volume constraint: match the one-domain implementation exactly.
    #
    # This is an algebraic constraint with pressure as the Lagrange multiplier.
    # In the production one-domain form we enforce it fully implicitly at level k:
    #   div(F_k) = alpha_k s_v(k),
    # not with a theta-average of div(F). Keeping the FEniCS mirror aligned here
    # is essential; otherwise the mass block and any total form containing it will
    # show a false mismatch against pycutfem.
    r_mass_fx = q_fx * divF_k_fx * dx_fx

    # Solid kinematics (Eulerian reference-map constraint):
    #   ∂_t u + vS·∇u = vS
    Fkin_dt_fx = (u_k_fx - u_n_fx) / dt_fx
    Fkin_adv_k_fx = ufl.dot(ufl.grad(u_k_fx), vS_k_fx) - vS_k_fx
    Fkin_adv_n_fx = ufl.dot(ufl.grad(u_n_fx), vS_n_fx) - vS_n_fx
    Fkin_k_fx = Fkin_dt_fx + theta_fx * Fkin_adv_k_fx + (1.0 - theta_fx) * Fkin_adv_n_fx
    r_kin_fx = alpha_k_fx * ufl.inner(Fkin_k_fx, eta_u_fx) * dx_fx
    if float(gamma_u_fx.value) != 0.0:
        if u_extension_mode_fx in {"l2", "mass"}:
            r_kin_fx += gamma_u_fx * inv_h2_fx * (1.0 - alpha_k_fx) * ufl.dot(u_k_fx, eta_u_fx) * dx_fx
        elif u_extension_mode_fx in {"grad", "h1"}:
            r_kin_fx += gamma_u_fx * (1.0 - alpha_k_fx) * ufl.inner(ufl.grad(u_k_fx), ufl.grad(eta_u_fx)) * dx_fx
            if float(gamma_u_pin_fx.value) != 0.0:
                w_pin_u_fx = 1.0 - alpha_k_fx
                r_kin_fx += gamma_u_pin_fx * inv_h2_fx * (w_pin_u_fx * w_pin_u_fx) * ufl.dot(u_k_fx, eta_u_fx) * dx_fx
        else:
            raise ValueError(f"Unknown COMP_FENICS_U_EXTENSION_MODE={u_extension_mode_fx!r}.")

    # Skeleton (linear elasticity, with pressure coupling through B and drag).
    def lin_el(u, eta):
        return 2.0 * mu_s_fx * ufl.inner(eps_fx(u), eps_fx(eta)) + lambda_s_fx * ufl.div(u) * ufl.div(eta)

    r_el_k_fx = lin_el(u_k_fx, eta_vS_fx)
    r_el_n_fx = lin_el(u_n_fx, eta_vS_fx)
    div_B_eta_k_fx = B_k_fx * ufl.div(eta_vS_fx) + ufl.dot(gradB_k_fx, eta_vS_fx)
    div_B_eta_n_fx = B_n_fx * ufl.div(eta_vS_fx) + ufl.dot(gradB_n_fx, eta_vS_fx)
    r_skel_press_k_fx = -p_k_fx * div_B_eta_k_fx
    r_skel_press_n_fx = -p_n_fx * div_B_eta_n_fx
    r_skel_press_k_fx += gamma_div_fx * divF_k_fx * div_B_eta_k_fx
    if use_refmap_drag_fx:
        r_skel_drag_k_fx = -beta_coeff_k_fx * ufl.dot(kdrag_k_fx, eta_vS_fx)
        r_skel_drag_n_fx = -beta_coeff_n_fx * ufl.dot(kdrag_n_fx, eta_vS_fx)
    else:
        r_skel_drag_k_fx = -beta_k_fx * ufl.dot(v_k_fx - vS_k_fx, eta_vS_fx)
        r_skel_drag_n_fx = -beta_n_fx * ufl.dot(v_n_fx - vS_n_fx, eta_vS_fx)
    sk_th_fx = 1.0
    sk_one_m_th_fx = 0.0
    r_skeleton_fx = (
        sk_th_fx * alpha_k_fx * (g_stiff_k_fx * r_el_k_fx)
        + sk_one_m_th_fx * alpha_n_fx * (g_stiff_n_fx * r_el_n_fx)
        + sk_th_fx * r_skel_press_k_fx
        + sk_one_m_th_fx * r_skel_press_n_fx
        + sk_th_fx * r_skel_drag_k_fx
        + sk_one_m_th_fx * r_skel_drag_n_fx
    ) * dx_fx
    if float(gamma_u_fx.value) != 0.0:
        if u_extension_mode_fx in {"l2", "mass"}:
            r_skeleton_fx += gamma_u_fx * inv_h2_fx * (1.0 - alpha_k_fx) * ufl.dot(vS_k_fx, eta_vS_fx) * dx_fx
        elif u_extension_mode_fx in {"grad", "h1"}:
            r_skeleton_fx += gamma_u_fx * (1.0 - alpha_k_fx) * ufl.inner(ufl.grad(vS_k_fx), ufl.grad(eta_vS_fx)) * dx_fx
            if float(gamma_u_pin_fx.value) != 0.0:
                w_pin_vS_fx = 1.0 - alpha_k_fx
                r_skeleton_fx += gamma_u_pin_fx * inv_h2_fx * (w_pin_vS_fx * w_pin_vS_fx) * ufl.dot(vS_k_fx, eta_vS_fx) * dx_fx
        else:
            raise ValueError(f"Unknown COMP_FENICS_U_EXTENSION_MODE={u_extension_mode_fx!r}.")

    # Extension-only residuals/Jacobians, kept separate to diagnose conditioning
    # differences in the one-domain free-fluid extension layer.
    r_u_ext_pc = None
    a_u_ext_pc = None
    r_vS_ext_pc = None
    a_vS_ext_pc = None
    if float(pc["gamma_u"]) != 0.0:
        one_minus_alpha_pc = Constant(1.0) - pc["alpha_k"]
        h_pc = MeshSize()
        inv_h2_pc = Constant(1.0) / (h_pc * h_pc)
        if str(pc["u_extension_mode"]).strip().lower() in {"l2", "mass"}:
            r_u_ext_pc = float(pc["gamma_u"]) * inv_h2_pc * one_minus_alpha_pc * dot(pc["u_k"], pc["eta"]) * dx_pc
            a_u_ext_pc = float(pc["gamma_u"]) * inv_h2_pc * (
                (-Constant(1.0) * pc["dalpha"]) * dot(pc["u_k"], pc["eta"])
                + one_minus_alpha_pc * dot(pc["du"], pc["eta"])
            ) * dx_pc
            r_vS_ext_pc = float(pc["gamma_u"]) * inv_h2_pc * one_minus_alpha_pc * dot(pc["vS_k"], pc["eta_vS"]) * dx_pc
            a_vS_ext_pc = float(pc["gamma_u"]) * inv_h2_pc * (
                (-Constant(1.0) * pc["dalpha"]) * dot(pc["vS_k"], pc["eta_vS"])
                + one_minus_alpha_pc * dot(pc["dvS"], pc["eta_vS"])
            ) * dx_pc
        elif str(pc["u_extension_mode"]).strip().lower() in {"grad", "h1"}:
            r_u_ext_pc = float(pc["gamma_u"]) * one_minus_alpha_pc * inner(grad(pc["u_k"]), grad(pc["eta"])) * dx_pc
            a_u_ext_pc = float(pc["gamma_u"]) * (
                (-Constant(1.0) * pc["dalpha"]) * inner(grad(pc["u_k"]), grad(pc["eta"]))
                + one_minus_alpha_pc * inner(grad(pc["du"]), grad(pc["eta"]))
            ) * dx_pc
            r_vS_ext_pc = float(pc["gamma_u"]) * one_minus_alpha_pc * inner(grad(pc["vS_k"]), grad(pc["eta_vS"])) * dx_pc
            a_vS_ext_pc = float(pc["gamma_u"]) * (
                (-Constant(1.0) * pc["dalpha"]) * inner(grad(pc["vS_k"]), grad(pc["eta_vS"]))
                + one_minus_alpha_pc * inner(grad(pc["dvS"]), grad(pc["eta_vS"]))
            ) * dx_pc
        else:
            raise ValueError(f"Unknown u_extension_mode={pc['u_extension_mode']!r}.")

    r_u_ext_fx = None
    a_u_ext_fx = None
    r_vS_ext_fx = None
    a_vS_ext_fx = None
    if float(gamma_u_fx.value) != 0.0:
        one_minus_alpha_fx = 1.0 - alpha_k_fx
        if u_extension_mode_fx in {"l2", "mass"}:
            r_u_ext_fx = gamma_u_fx * inv_h2_fx * one_minus_alpha_fx * ufl.dot(u_k_fx, eta_u_fx) * dx_fx
            r_vS_ext_fx = gamma_u_fx * inv_h2_fx * one_minus_alpha_fx * ufl.dot(vS_k_fx, eta_vS_fx) * dx_fx
        elif u_extension_mode_fx in {"grad", "h1"}:
            r_u_ext_fx = gamma_u_fx * one_minus_alpha_fx * ufl.inner(ufl.grad(u_k_fx), ufl.grad(eta_u_fx)) * dx_fx
            r_vS_ext_fx = gamma_u_fx * one_minus_alpha_fx * ufl.inner(ufl.grad(vS_k_fx), ufl.grad(eta_vS_fx)) * dx_fx
        else:
            raise ValueError(f"Unknown COMP_FENICS_U_EXTENSION_MODE={u_extension_mode_fx!r}.")
        a_u_ext_fx = ufl.derivative(r_u_ext_fx, fenicsx["w_k"], dw_fx)
        a_vS_ext_fx = ufl.derivative(r_vS_ext_fx, fenicsx["w_k"], dw_fx)

    # Porosity evolution
    Pi_k_fx = Pi_over_rho_s(S_k_fx, phi_k_fx, alpha_k_fx)
    Pi_n_fx = Pi_over_rho_s(S_n_fx, phi_n_fx, alpha_n_fx)
    Fphi_k_fx = ufl.dot(ufl.grad(phi_k_fx), vS_k_fx) - one_minus(phi_k_fx) * div_vS_k_fx + Pi_k_fx
    Fphi_n_fx = ufl.dot(ufl.grad(phi_n_fx), vS_n_fx) - one_minus(phi_n_fx) * div_vS_n_fx + Pi_n_fx

    one_m_alpha_k_fx = one_minus(alpha_k_fx)
    w_phi_fluid4_k_fx = one_m_alpha_k_fx * one_m_alpha_k_fx
    w_phi_fluid4_k_fx = w_phi_fluid4_k_fx * w_phi_fluid4_k_fx
    w_phi_fluid8_k_fx = w_phi_fluid4_k_fx * w_phi_fluid4_k_fx
    w_phi_fluid_k_fx = w_phi_fluid8_k_fx * w_phi_fluid8_k_fx

    r_phi_fx = alpha_k_fx * zeta_fx * ((phi_k_fx - phi_n_fx) / dt_fx) * dx_fx
    r_phi_fx += theta_fx * alpha_k_fx * zeta_fx * Fphi_k_fx * dx_fx
    r_phi_fx += (1.0 - theta_fx) * alpha_n_fx * zeta_fx * Fphi_n_fx * dx_fx
    r_phi_fx += D_phi_fx * ufl.dot(ufl.grad(phi_k_fx), ufl.grad(zeta_fx)) * dx_fx
    r_phi_fx += gamma_phi_fx * w_phi_fluid_k_fx * (phi_k_fx - 1.0) * zeta_fx * dx_fx

    # Indicator evolution
    G_k_fx = G(S_k_fx, phi_k_fx)
    G_n_fx = G(S_n_fx, phi_n_fx)
    delta_k_fx = 4.0 * alpha_k_fx * one_minus(alpha_k_fx)
    delta_n_fx = 4.0 * alpha_n_fx * one_minus(alpha_n_fx)
    surf_coef_prev_fx = D_det_prev_fx
    f_alpha_k_fx = (alpha_k_fx - alpha_n_fx) / dt_fx
    if alpha_advect_with_key not in {"vs", "v^s", "v_s", "s", "skeleton", "solid"}:
        raise ValueError(f"Current FEniCS biofilm harness supports only alpha advection with vS; got {alpha_advect_with_key!r}.")
    if alpha_advection_form_key in {"advective", "nonconservative", "v.grad", "v·grad", "vgrad"}:
        adv_alpha_k_fx = ufl.dot(ufl.grad(alpha_k_fx), vS_k_fx)
        adv_alpha_n_fx = ufl.dot(ufl.grad(alpha_n_fx), vS_n_fx)
    elif alpha_advection_form_key in {"conservative", "div", "divergence", "div(alpha*v)"}:
        adv_alpha_k_fx = ufl.dot(ufl.grad(alpha_k_fx), vS_k_fx) + alpha_k_fx * div_vS_k_fx
        adv_alpha_n_fx = ufl.dot(ufl.grad(alpha_n_fx), vS_n_fx) + alpha_n_fx * div_vS_n_fx
    else:
        raise ValueError(f"Unknown COMP_FENICS_ALPHA_ADVECTION_FORM={alpha_advection_form_key!r}.")
    f_alpha_k_fx += theta_fx * (adv_alpha_k_fx - G_k_fx * alpha_k_fx * one_minus(alpha_k_fx) + surf_coef_prev_fx * delta_k_fx)
    f_alpha_k_fx += (1.0 - theta_fx) * (
        adv_alpha_n_fx - G_n_fx * alpha_n_fx * one_minus(alpha_n_fx) + surf_coef_prev_fx * delta_n_fx
    )
    r_alpha_fx = xi_fx * f_alpha_k_fx * dx_fx
    r_alpha_fx += D_alpha_fx * ufl.dot(ufl.grad(alpha_k_fx), ufl.grad(xi_fx)) * dx_fx

    # Optional Cahn–Hilliard regularization for alpha (mass-conserving phase-field).
    r_mu_alpha_fx = None
    if include_mu_alpha:
        M0_ch_fx = fenicsx["alpha_ch_M"]
        gamma_ch_fx = fenicsx["alpha_ch_gamma"]
        eps_ch_fx = fenicsx["alpha_ch_eps"]
        mob_key = str(fenicsx.get("alpha_ch_mobility", "constant")).strip().lower()

        if mob_key in {"constant", "const"}:
            M_ch_k_fx = M0_ch_fx
            M_ch_n_fx = M0_ch_fx
        elif mob_key in {"degenerate", "deg", "alpha(1-alpha)", "alpha*(1-alpha)", "a(1-a)"}:
            M_ch_k_fx = M0_ch_fx * alpha_k_fx * (1.0 - alpha_k_fx)
            M_ch_n_fx = M0_ch_fx * alpha_n_fx * (1.0 - alpha_n_fx)
        else:
            raise ValueError(f"Unknown COMP_FENICS_ALPHA_CH_MOBILITY={fenicsx.get('alpha_ch_mobility')!r}. Use 'constant' or 'degenerate'.")

        # Double-well derivative for W(α)=α^2(1-α)^2: W'(α)=2α(1-α)(1-2α).
        Wp_ch_k_fx = 2.0 * alpha_k_fx * (1.0 - alpha_k_fx) * (1.0 - 2.0 * alpha_k_fx)

        # α equation: +∫ M(α) ∇μ · ∇w (theta-averaged; no-flux boundary)
        r_alpha_fx += theta_fx * M_ch_k_fx * ufl.dot(ufl.grad(mu_alpha_k_fx), ufl.grad(xi_fx)) * dx_fx
        r_alpha_fx += (1.0 - theta_fx) * M_ch_n_fx * ufl.dot(ufl.grad(mu_alpha_n_fx), ufl.grad(xi_fx)) * dx_fx

        # μ equation: ∫ ψ ( μ - γ(-εΔα + (1/ε)W'(α)) ) dx = 0 (drop boundary term).
        r_mu_alpha_fx = eta_mu_fx * mu_alpha_k_fx * dx_fx
        r_mu_alpha_fx += -(gamma_ch_fx * eps_ch_fx) * ufl.dot(ufl.grad(alpha_k_fx), ufl.grad(eta_mu_fx)) * dx_fx
        r_mu_alpha_fx += -(gamma_ch_fx / eps_ch_fx) * eta_mu_fx * Wp_ch_k_fx * dx_fx

    # Bulk damage (kinetic model)
    I2_fx = ufl.Identity(2)
    sig_un_fx = 2.0 * mu_s_fx * eps_fx(u_n_fx) + lambda_s_fx * ufl.div(u_n_fx) * I2_fx
    tr_sig_un_fx = ufl.tr(sig_un_fx)
    s_dev_un_fx = sig_un_fx - (tr_sig_un_fx / 2.0) * I2_fx
    sigma_vm_fx = ufl.sqrt(1.5 * ufl.inner(s_dev_un_fx, s_dev_un_fx) + 1.0e-16)
    ratio_fx = sigma_vm_fx / damage_sigma_cr_fx - 1.0
    pos_ratio_fx = 0.5 * (ratio_fx + ufl.sqrt(ratio_fx * ratio_fx + damage_eta_pos_fx))
    drive_vm_fx = pos_ratio_fx ** damage_m_fx
    rate_fx = damage_k_fx * drive_vm_fx

    f_dmg_k_fx = alpha_k_fx * ((d_k_fx - d_n_fx) / dt_fx)
    f_dmg_k_fx += theta_fx * alpha_k_fx * ufl.dot(ufl.grad(d_k_fx), vS_k_fx)
    f_dmg_k_fx += (1.0 - theta_fx) * alpha_n_fx * ufl.dot(ufl.grad(d_n_fx), vS_n_fx)
    f_dmg_k_fx += -alpha_k_fx * rate_fx * one_m_d_k_fx

    r_damage_fx = chi_fx * f_dmg_k_fx * dx_fx
    r_damage_fx += damage_D_fx * ufl.dot(ufl.grad(d_k_fx), ufl.grad(chi_fx)) * dx_fx
    r_damage_fx += damage_gamma_out_fx * w_phi_fluid_k_fx * d_k_fx * chi_fx * dx_fx

    # Substrate transport
    CSk_fx = C_k_fx * S_k_fx
    CSn_fx = C_n_fx * S_n_fx
    RS_k_fx = rho_s_star_fx * (1.0 / Y_fx) * Pi_k_fx
    RS_n_fx = rho_s_star_fx * (1.0 / Y_fx) * Pi_n_fx

    div_CSv_k_fx = CSk_fx * ufl.div(v_k_fx) + S_k_fx * ufl.dot(gradC_k_fx, v_k_fx) + C_k_fx * ufl.dot(ufl.grad(S_k_fx), v_k_fx)
    div_CSv_n_fx = CSn_fx * ufl.div(v_n_fx) + S_n_fx * ufl.dot(gradC_n_fx, v_n_fx) + C_n_fx * ufl.dot(ufl.grad(S_n_fx), v_n_fx)

    r_S_fx = r_fx * ((CSk_fx - CSn_fx) / dt_fx) * dx_fx
    r_S_fx += r_fx * (theta_fx * div_CSv_k_fx + (1.0 - theta_fx) * div_CSv_n_fx) * dx_fx
    r_S_fx += D_S_fx * theta_fx * ufl.dot(ufl.grad(S_k_fx), ufl.grad(r_fx)) * dx_fx
    r_S_fx += D_S_fx * (1.0 - theta_fx) * ufl.dot(ufl.grad(S_n_fx), ufl.grad(r_fx)) * dx_fx
    r_S_fx += r_fx * (theta_fx * RS_k_fx + (1.0 - theta_fx) * RS_n_fx) * dx_fx

    # Detached biomass transport
    CXk_fx = C_k_fx * X_k_fx
    CXn_fx = C_n_fx * X_n_fx
    div_CXv_k_fx = CXk_fx * ufl.div(v_k_fx) + X_k_fx * ufl.dot(gradC_k_fx, v_k_fx) + C_k_fx * ufl.dot(ufl.grad(X_k_fx), v_k_fx)
    div_CXv_n_fx = CXn_fx * ufl.div(v_n_fx) + X_n_fx * ufl.dot(gradC_n_fx, v_n_fx) + C_n_fx * ufl.dot(ufl.grad(X_n_fx), v_n_fx)
    R_det_k_fx = rho_s_star_fx * one_minus(phi_k_fx) * D_det_prev_fx * delta_k_fx
    R_det_n_fx = rho_s_star_fx * one_minus(phi_n_fx) * D_det_prev_fx * delta_n_fx
    r_X_fx = y_fx * ((CXk_fx - CXn_fx) / dt_fx) * dx_fx
    r_X_fx += y_fx * (theta_fx * div_CXv_k_fx + (1.0 - theta_fx) * div_CXv_n_fx) * dx_fx
    r_X_fx += D_X_fx * theta_fx * ufl.dot(ufl.grad(X_k_fx), ufl.grad(y_fx)) * dx_fx
    r_X_fx += D_X_fx * (1.0 - theta_fx) * ufl.dot(ufl.grad(X_n_fx), ufl.grad(y_fx)) * dx_fx
    r_X_fx += -y_fx * (theta_fx * R_det_k_fx + (1.0 - theta_fx) * R_det_n_fx) * dx_fx

    r_current_total_fx = r_mom_fx + r_mass_fx + r_kin_fx + r_skeleton_fx + r_phi_fx + r_alpha_fx + r_S_fx
    r_total_fx = r_current_total_fx + r_damage_fx + r_X_fx
    if r_mu_alpha_fx is not None:
        r_current_total_fx += r_mu_alpha_fx
        r_total_fx += r_mu_alpha_fx

    # ------------------------------------------------------------------
    # Jacobian pieces (fenics, automatic differentiation w.r.t. w_k)
    # ------------------------------------------------------------------
    a_mom_fx = ufl.derivative(r_mom_fx, fenicsx["w_k"], dw_fx)
    a_mass_fx = ufl.derivative(r_mass_fx, fenicsx["w_k"], dw_fx)
    a_kin_fx = ufl.derivative(r_kin_fx, fenicsx["w_k"], dw_fx)
    a_skeleton_fx = ufl.derivative(r_skeleton_fx, fenicsx["w_k"], dw_fx)
    a_phi_fx = ufl.derivative(r_phi_fx, fenicsx["w_k"], dw_fx)
    a_alpha_fx = ufl.derivative(r_alpha_fx, fenicsx["w_k"], dw_fx)
    a_mu_alpha_fx = ufl.derivative(r_mu_alpha_fx, fenicsx["w_k"], dw_fx) if r_mu_alpha_fx is not None else None
    a_damage_fx = ufl.derivative(r_damage_fx, fenicsx["w_k"], dw_fx)
    a_S_fx = ufl.derivative(r_S_fx, fenicsx["w_k"], dw_fx)
    a_X_fx = ufl.derivative(r_X_fx, fenicsx["w_k"], dw_fx)
    a_current_total_fx = ufl.derivative(r_current_total_fx, fenicsx["w_k"], dw_fx)
    a_total_fx = ufl.derivative(r_total_fx, fenicsx["w_k"], dw_fx)

    # ------------------------------------------------------------------
    # pycutfem forms (biofilm module)
    # ------------------------------------------------------------------
    forms_pc = build_biofilm_one_domain_forms(
        v_k=pc["v_k"],
        p_k=pc["p_k"],
        vS_k=pc["vS_k"],
        u_k=pc["u_k"],
        phi_k=pc["phi_k"],
        alpha_k=pc["alpha_k"],
        mu_alpha_k=pc["mu_alpha_k"],
        d_k=pc["d_k"],
        S_k=pc["S_k"],
        X_k=pc["X_k"],
        v_n=pc["v_n"],
        p_n=pc["p_n"],
        vS_n=pc["vS_n"],
        u_n=pc["u_n"],
        phi_n=pc["phi_n"],
        alpha_n=pc["alpha_n"],
        mu_alpha_n=pc["mu_alpha_n"],
        d_n=pc["d_n"],
        S_n=pc["S_n"],
        X_n=pc["X_n"],
        dv=pc["dv"],
        dp=pc["dp"],
        dvS=pc["dvS"],
        du=pc["du"],
        dphi=pc["dphi"],
        dalpha=pc["dalpha"],
        dmu_alpha=pc["dmu_alpha"],
        dd=pc["dd"],
        dS=pc["dS"],
        dX=pc["dX"],
        v_test=pc["w"],
        q_test=pc["q"],
        vS_test=pc["eta_vS"],
        u_test=pc["eta"],
        phi_test=pc["zeta"],
        alpha_test=pc["xi"],
        mu_alpha_test=pc["eta_mu"],
        d_test=pc["chi"],
        S_test=pc["r"],
        X_test=pc["y"],
        dx=dx_pc,
        dt=pc["dt"],
        theta=float(pc["theta"]),
        rho_f=pc["rho_f"],
        mu_f=pc["mu_f"],
        kappa_inv=pc["kappa_inv"],
        kappa_inv_model=str(pc["kappa_inv_model"]),
        mu_s=pc["mu_s"],
        lambda_s=pc["lambda_s"],
        fluid_convection=str(pc["fluid_convection"]),
        gamma_div=float(pc["gamma_div"]),
        gamma_u=float(pc["gamma_u"]),
        u_extension_mode=str(pc["u_extension_mode"]),
        gamma_u_pin=float(pc["gamma_u_pin"]),
        v_supg=float(pc["v_supg"]),
        v_supg_mode=str(pc["v_supg_mode"]),
        v_supg_c_nu=float(pc["v_supg_c_nu"]),
        ds_hdiv_tangential=ds_hdiv_tangential_pc,
        hdiv_tangential_gamma=float(pc["hdiv_tangential_gamma"]),
        hdiv_tangential_method=str(pc["hdiv_tangential_method"]),
        alpha_advect_with=str(pc["alpha_advect_with"]),
        alpha_advection_form=str(pc["alpha_advection_form"]),
        D_phi=float(pc["D_phi"]),
        gamma_phi=float(pc["gamma_phi"]),
        D_alpha=float(pc["D_alpha"]),
        alpha_ch_M=float(pc["alpha_ch_M"]),
        alpha_ch_gamma=float(pc["alpha_ch_gamma"]),
        alpha_ch_eps=float(pc["alpha_ch_eps"]),
        alpha_ch_mobility=str(pc["alpha_ch_mobility"]),
        D_X=float(pc["D_X"]),
        D_S=float(pc["D_S"]),
        mu_max=float(pc["mu_max"]),
        K_S=float(pc["K_S"]),
        k_g=float(pc["k_g"]),
        k_d=float(pc["k_d"]),
        Y=float(pc["Y"]),
        rho_s_star=float(pc["rho_s_star"]),
        k_det=float(pc["k_det"]),
        damage_model=str(pc["damage_model"]),
        damage_k=float(pc["damage_k"]),
        damage_sigma_cr=float(pc["damage_sigma_cr"]),
        damage_m=float(pc["damage_m"]),
        damage_D=float(pc["damage_D"]),
        damage_gamma_out=float(pc["damage_gamma_out"]),
        damage_eta_pos=float(pc["damage_eta_pos"]),
        damage_kappa_stiff=float(pc["damage_kappa_stiff"]),
        damage_kappa_perm=float(pc["damage_kappa_perm"]),
    )

    # ------------------------------------------------------------------
    # Terms dictionary (split residual/Jacobian blocks)
    # ------------------------------------------------------------------
    terms = {
        # Residual pieces
        "Biofilm momentum (res)": {"pc": forms_pc.r_momentum, "fx": r_mom_fx, "mat": False},
        "Biofilm mass (res)": {"pc": forms_pc.r_mass, "fx": r_mass_fx, "mat": False},
        "Biofilm kinematics (res)": {"pc": forms_pc.r_kinematics, "fx": r_kin_fx, "mat": False},
        "Biofilm skeleton (res)": {"pc": forms_pc.r_skeleton, "fx": r_skeleton_fx, "mat": False},
        "Biofilm phi (res)": {"pc": forms_pc.r_phi, "fx": r_phi_fx, "mat": False},
        "Biofilm alpha (res)": {"pc": forms_pc.r_alpha, "fx": r_alpha_fx, "mat": False},
        "Biofilm damage (res)": {"pc": forms_pc.r_damage, "fx": r_damage_fx, "mat": False},
        "Biofilm substrate (res)": {"pc": forms_pc.r_substrate, "fx": r_S_fx, "mat": False},
        "Biofilm detached (res)": {"pc": forms_pc.r_detached, "fx": r_X_fx, "mat": False},
        "Biofilm current total residual": {
            "pc": forms_pc.r_momentum
            + forms_pc.r_mass
            + forms_pc.r_kinematics
            + forms_pc.r_skeleton
            + forms_pc.r_phi
            + forms_pc.r_alpha
            + forms_pc.r_substrate,
            "fx": r_current_total_fx,
            "mat": False,
        },
        "Biofilm total residual": {"pc": forms_pc.residual_form, "fx": r_total_fx, "mat": False},
        # Jacobian pieces
        "Biofilm momentum (jac)": {"pc": forms_pc.a_momentum, "fx": a_mom_fx, "mat": True},
        "Biofilm mass (jac)": {"pc": forms_pc.a_mass, "fx": a_mass_fx, "mat": True},
        "Biofilm kinematics (jac)": {"pc": forms_pc.a_kinematics, "fx": a_kin_fx, "mat": True},
        "Biofilm skeleton (jac)": {"pc": forms_pc.a_skeleton, "fx": a_skeleton_fx, "mat": True},
        "Biofilm phi (jac)": {"pc": forms_pc.a_phi, "fx": a_phi_fx, "mat": True},
        "Biofilm alpha (jac)": {"pc": forms_pc.a_alpha, "fx": a_alpha_fx, "mat": True},
        "Biofilm damage (jac)": {"pc": forms_pc.a_damage, "fx": a_damage_fx, "mat": True},
        "Biofilm substrate (jac)": {"pc": forms_pc.a_substrate, "fx": a_S_fx, "mat": True},
        "Biofilm detached (jac)": {"pc": forms_pc.a_detached, "fx": a_X_fx, "mat": True},
        "Biofilm current total jacobian": {
            "pc": forms_pc.a_momentum
            + forms_pc.a_mass
            + forms_pc.a_kinematics
            + forms_pc.a_skeleton
            + forms_pc.a_phi
            + forms_pc.a_alpha
            + forms_pc.a_substrate,
            "fx": a_current_total_fx,
            "mat": True,
        },
        "Biofilm total jacobian": {"pc": forms_pc.jacobian_form, "fx": a_total_fx, "mat": True},
    }

    if r_mu_alpha_fx is not None:
        terms["Biofilm mu_alpha (res)"] = {"pc": forms_pc.r_mu_alpha, "fx": r_mu_alpha_fx, "mat": False}
        terms["Biofilm mu_alpha (jac)"] = {"pc": forms_pc.a_mu_alpha, "fx": a_mu_alpha_fx, "mat": True}
        terms["Biofilm current total residual"]["pc"] = terms["Biofilm current total residual"]["pc"] + forms_pc.r_mu_alpha
        terms["Biofilm current total jacobian"]["pc"] = terms["Biofilm current total jacobian"]["pc"] + forms_pc.a_mu_alpha
    if r_u_ext_pc is not None and r_u_ext_fx is not None:
        terms["Biofilm u extension (res)"] = {"pc": r_u_ext_pc, "fx": r_u_ext_fx, "mat": False}
        terms["Biofilm u extension (jac)"] = {"pc": a_u_ext_pc, "fx": a_u_ext_fx, "mat": True}
    if r_vS_ext_pc is not None and r_vS_ext_fx is not None:
        terms["Biofilm vS extension (res)"] = {"pc": r_vS_ext_pc, "fx": r_vS_ext_fx, "mat": False}
        terms["Biofilm vS extension (jac)"] = {"pc": a_vS_ext_pc, "fx": a_vS_ext_fx, "mat": True}

    # Filter terms (full suite is the default for biofilm).
    filter_terms = os.environ.get("COMP_FENICS_TERMS")
    if filter_terms:
        allowed = {name.strip() for name in filter_terms.split(",") if name.strip()}
        terms = {k: v for k, v in terms.items() if k in allowed}
        print(f"Running filtered terms only: {sorted(terms)}")
    else:
        print(f"Running full biofilm suite: {sorted(terms)}")

    # Pre-assemble FEniCSx reference outputs once (backend-independent)
    fenics_ref = {}
    for name, spec in terms.items():
        try:
            form_fx_compiled = dolfinx.fem.form(spec["fx"])
            if spec["mat"]:
                A = dolfinx.fem.petsc.assemble_matrix(form_fx_compiled)
                A.assemble()
                indptr, indices, data = A.getValuesCSR()
                mat_fx = csr_matrix((data, indices, indptr), shape=A.getSize()).tocsr()
                fenics_ref[name] = mat_fx if use_sparse_compare else mat_fx.toarray()
            else:
                vec = dolfinx.fem.petsc.assemble_vector(form_fx_compiled)
                fenics_ref[name] = vec.array.copy()
        except Exception as exc:
            print(f"❌ FEniCSx assembly failed for '{name}': {exc}")
            fenics_ref[name] = None

    backends_spec = os.environ.get("BACKEND", "jit")
    if backends_spec.strip().lower() == "all":
        backends = ["python", "jit", "cpp"]
    else:
        backends = [b.strip() for b in backends_spec.split(",") if b.strip()]
    parity_rtol = float(os.environ.get("COMP_FENICS_PARITY_RTOL", "1e-9"))
    parity_atol = float(os.environ.get("COMP_FENICS_PARITY_ATOL", "1e-9"))

    # Optional: backend parity check against a reference backend
    ref_backend = "python" if "python" in backends else (backends[0] if backends else "python")
    ref_pc = {}

    for backend_type in backends:
        print("\n" + "=" * 70)
        print(f"BIOFILM COMPARISON (backend={backend_type}, qdeg={qdeg})")
        print("=" * 70)

        failed_tests = []
        success_count = 0

        for name, spec in terms.items():
            if fenics_ref.get(name) is None:
                failed_tests.append(f"{name} (fenics-assemble)")
                continue

            J_pc, R_pc = None, None
            print(f"\nCompiling/assembling '{name}' [backend={backend_type}]")
            try:
                if spec["mat"]:
                    J_pc, _ = assemble_form(Equation(spec["pc"], None), dof_handler_pc, quad_degree=qdeg, bcs=[], backend=backend_type)
                else:
                    _, R_pc = assemble_form(Equation(None, spec["pc"]), dof_handler_pc, bcs=[], backend=backend_type)
            except Exception as exc:
                print(f"❌ pycutfem assembly failed for '{name}' on backend '{backend_type}': {exc}")
                failed_tests.append(f"{name} (assemble-{backend_type})")
                continue

            # Backend parity (pycutfem) against a reference backend
            if backend_type == ref_backend:
                if spec["mat"]:
                    ref_pc[name] = J_pc.tocsr().copy() if use_sparse_compare else J_pc.toarray()
                else:
                    ref_pc[name] = np.asarray(R_pc, dtype=float).copy()
            else:
                try:
                    if spec["mat"]:
                        if use_sparse_compare:
                            ok, max_abs, worst = _sparse_matrix_allclose(
                                J_pc.tocsr(),
                                ref_pc[name],
                                rtol=parity_rtol,
                                atol=parity_atol,
                            )
                            if not ok:
                                raise AssertionError(f"sparse mismatch: max_abs={max_abs:.3e}, worst={worst}")
                        else:
                            np.testing.assert_allclose(J_pc.toarray(), ref_pc[name], rtol=parity_rtol, atol=parity_atol)
                    else:
                        np.testing.assert_allclose(np.asarray(R_pc, dtype=float), ref_pc[name], rtol=parity_rtol, atol=parity_atol)
                    print(f"✅ pycutfem backend parity OK vs '{ref_backend}'.")
                except Exception as exc:
                    print(f"❌ pycutfem backend parity FAILED vs '{ref_backend}': {exc}")
                    failed_tests.append(f"{name} (parity-{backend_type}-vs-{ref_backend})")

            # Compare to FEniCSx reference
            J_fx, R_fx = None, None
            if spec["mat"]:
                J_fx = fenics_ref[name]
            else:
                R_fx = fenics_ref[name]

            is_success = compare_term(
                f"{name} [backend={backend_type}]",
                J_pc,
                R_pc,
                J_fx,
                R_fx,
                P_map,
                dof_handler_pc,
                W_fx,
                rtol=1e-8,
                atol=1e-8,
                sign_map=sign_map,
                fx_coords_all=fx_coords_all,
                transform=transform_map,
            )
            if is_success:
                success_count += 1
            else:
                failed_tests.append(name)

        print_test_summary(success_count, failed_tests)


def _benchmark7_hdiv_state_vector(problem, *, suffix: str) -> np.ndarray:
    dh = problem["dh"]
    state = np.zeros(dh.total_dofs, dtype=float)

    state[np.asarray(dh.get_field_slice("v"), dtype=int)] = np.asarray(problem[f"v_{suffix}"].nodal_values, dtype=float)
    state[np.asarray(dh.get_field_slice("p"), dtype=int)] = np.asarray(problem[f"p_{suffix}"].nodal_values, dtype=float)

    for base, fields in (("vS", ("vS_x", "vS_y")), ("u", ("u_x", "u_y"))):
        vec = problem[f"{base}_{suffix}"]
        state[np.asarray(dh.get_field_slice(fields[0]), dtype=int)] = np.asarray(vec.nodal_values_component(0), dtype=float)
        state[np.asarray(dh.get_field_slice(fields[1]), dtype=int)] = np.asarray(vec.nodal_values_component(1), dtype=float)

    state[np.asarray(dh.get_field_slice("alpha"), dtype=int)] = np.asarray(problem[f"alpha_{suffix}"].nodal_values, dtype=float)
    state[np.asarray(dh.get_field_slice("mu_alpha"), dtype=int)] = np.asarray(problem[f"mu_{suffix}"].nodal_values, dtype=float)
    return state


def setup_benchmark7_hdiv_problems(*, nx: int = 2, ny: int = 3):
    Lx = float(os.environ.get("COMP_FENICS_B7_LX", "1.0"))
    Ly = float(os.environ.get("COMP_FENICS_B7_LY", "1.5"))
    poly_order = int(os.environ.get("COMP_FENICS_B7_POLY_ORDER", "2"))
    pressure_order = int(os.environ.get("COMP_FENICS_B7_PRESSURE_ORDER", str(max(1, poly_order - 1))))
    scalar_order = int(os.environ.get("COMP_FENICS_B7_SCALAR_ORDER", str(max(1, poly_order - 1))))
    fluid_hdiv_order = int(os.environ.get("COMP_FENICS_B7_FLUID_HDIV_ORDER", "0"))
    if fluid_hdiv_order != 0:
        raise NotImplementedError("Seboldt H(div) comparison currently targets RT0 only (fluid_hdiv_order=0).")
    problem_pc = create_benchmark7_seboldt_problem(
        Lx=Lx,
        Ly=Ly,
        nx=int(nx),
        ny=int(ny),
        poly_order=int(poly_order),
        pressure_order=int(pressure_order),
        scalar_order=int(scalar_order),
        fluid_space="hdiv",
        fluid_hdiv_order=int(fluid_hdiv_order),
        enable_phi_evolution=False,
    )
    dof_handler_pc = problem_pc["dh"]

    p_family = "DQ" if int(pressure_order) == 0 else "Lagrange"
    s_family = "DQ" if int(scalar_order) == 0 else "Lagrange"

    mesh_fx = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0], dtype=float), np.array([Lx, Ly], dtype=float)],
        [int(nx), int(ny)],
        cell_type=dolfinx.mesh.CellType.quadrilateral,
    )
    gdim = mesh_fx.geometry.dim
    RT = basix.ufl.element("RT", "quadrilateral", int(fluid_hdiv_order) + 1)
    Pp = basix.ufl.element(p_family, "quadrilateral", int(pressure_order))
    Pvec = basix.ufl.element("Lagrange", "quadrilateral", int(poly_order), shape=(gdim,))
    Ps = basix.ufl.element(s_family, "quadrilateral", int(scalar_order))
    W_el = mixed_element([RT, Pp, Pvec, Pvec, Ps, Ps])
    W = dolfinx.fem.functionspace(mesh_fx, W_el)

    params = {
        "Lx": Lx,
        "Ly": Ly,
        "poly_order": int(poly_order),
        "pressure_order": int(pressure_order),
        "scalar_order": int(scalar_order),
        "fluid_hdiv_order": int(fluid_hdiv_order),
        "qdeg": int(os.environ.get("COMP_FENICS_B7_QDEG", str(max(6, 2 * poly_order + 2)))),
        "dt": float(os.environ.get("COMP_FENICS_DT", "0.1")),
        "theta": float(os.environ.get("COMP_FENICS_THETA", "0.5")),
        "rho_f": float(os.environ.get("COMP_FENICS_B7_RHO_F", "1.0")),
        "mu_f": float(os.environ.get("COMP_FENICS_B7_MU_F", "0.035")),
        "mu_b": float(os.environ.get("COMP_FENICS_B7_MU_B", "0.035")),
        "kappa_inv": float(os.environ.get("COMP_FENICS_B7_KAPPA_INV", "1000.0")),
        "mu_s": float(os.environ.get("COMP_FENICS_B7_MU_S", "1.67785e5")),
        "lambda_s": float(os.environ.get("COMP_FENICS_B7_LAMBDA_S", "8.22148e6")),
        "phi_b": float(os.environ.get("COMP_FENICS_B7_PHI_B", "0.5")),
        "M_alpha": float(os.environ.get("COMP_FENICS_B7_M_ALPHA", "1.0")),
        "gamma_alpha": float(os.environ.get("COMP_FENICS_B7_GAMMA_ALPHA", "1.0")),
        "eps_alpha": float(os.environ.get("COMP_FENICS_B7_EPS_ALPHA", "0.05")),
        "gamma_div": float(os.environ.get("COMP_FENICS_B7_GAMMA_DIV", "0.0")),
    }

    fenicsx = {
        "W": W,
        "mesh": mesh_fx,
        "rho_f": dolfinx.fem.Constant(mesh_fx, params["rho_f"]),
        "mu_f": dolfinx.fem.Constant(mesh_fx, params["mu_f"]),
        "mu_b": dolfinx.fem.Constant(mesh_fx, params["mu_b"]),
        "kappa_inv": dolfinx.fem.Constant(mesh_fx, params["kappa_inv"]),
        "mu_s": dolfinx.fem.Constant(mesh_fx, params["mu_s"]),
        "lambda_s": dolfinx.fem.Constant(mesh_fx, params["lambda_s"]),
        "phi_b": dolfinx.fem.Constant(mesh_fx, params["phi_b"]),
        "dt": dolfinx.fem.Constant(mesh_fx, params["dt"]),
        "theta": dolfinx.fem.Constant(mesh_fx, params["theta"]),
        "M_alpha": dolfinx.fem.Constant(mesh_fx, params["M_alpha"]),
        "gamma_alpha": dolfinx.fem.Constant(mesh_fx, params["gamma_alpha"]),
        "eps_alpha": dolfinx.fem.Constant(mesh_fx, params["eps_alpha"]),
        "gamma_div": dolfinx.fem.Constant(mesh_fx, params["gamma_div"]),
        "w_k": dolfinx.fem.Function(W, name="w_k"),
        "w_n": dolfinx.fem.Function(W, name="w_n"),
    }
    return problem_pc, dof_handler_pc, fenicsx, params


def create_true_dof_map_benchmark7_hdiv(dof_handler_pc: DofHandler, W_fenicsx):
    print("=" * 70)
    print("Discovering Seboldt H(div) DoF map (RT,p,vS,u,alpha,mu_alpha)...")
    print("=" * 70)

    P = np.zeros(dof_handler_pc.total_dofs, dtype=int)
    fx_coords_all = np.zeros((W_fenicsx.dofmap.index_map.size_global, 2), dtype=float)

    Wv, Vv_map = W_fenicsx.sub(0).collapse()
    v_fx_dofs_collapsed, v_fx_kinds, v_fx_points, v_fx_modes = _fenics_rt_dofs_and_descriptors(Wv)
    v_fx_parent = np.asarray(Vv_map, dtype=int)[v_fx_dofs_collapsed]
    v_pc_slice = np.asarray(dof_handler_pc.get_field_slice("v"), dtype=int)
    _, v_pc_kinds, v_pc_points, v_pc_modes = _pycutfem_rt_dofs_and_descriptors(dof_handler_pc, "v")
    v_pc_to_fx = _map_rt_descriptors(v_pc_kinds, v_pc_points, v_pc_modes, v_fx_kinds, v_fx_points, v_fx_modes)
    P[v_pc_slice] = v_fx_parent[v_pc_to_fx]
    fx_coords_all[v_fx_parent] = v_fx_points

    Wp, P_map_fx = W_fenicsx.sub(1).collapse()
    p_fx_coords = Wp.tabulate_dof_coordinates()[:, :2]
    p_fx_parent = np.asarray(P_map_fx, dtype=int)
    p_pc_slice = np.asarray(dof_handler_pc.get_field_slice("p"), dtype=int)
    p_pc_coords = get_pycutfem_dof_coords(dof_handler_pc, "p")
    p_pc_to_fx = one_to_one_map_coords(p_pc_coords, p_fx_coords)
    P[p_pc_slice] = p_fx_parent[p_pc_to_fx]
    fx_coords_all[p_fx_parent] = p_fx_coords

    for parent_idx, pc_prefix in ((2, "vS"), (3, "u")):
        Wvec, Vec_map = W_fenicsx.sub(parent_idx).collapse()
        W0, V0_map = Wvec.sub(0).collapse()
        W1, V1_map = Wvec.sub(1).collapse()
        vec_fx_coords = W0.tabulate_dof_coordinates()[:, :2]
        vec_fx_parent0 = np.asarray(Vec_map, dtype=int)[np.asarray(V0_map, dtype=int)]
        vec_fx_parent1 = np.asarray(Vec_map, dtype=int)[np.asarray(V1_map, dtype=int)]
        pc_field0 = f"{pc_prefix}_x"
        pc_field1 = f"{pc_prefix}_y"
        pc_slice0 = np.asarray(dof_handler_pc.get_field_slice(pc_field0), dtype=int)
        pc_slice1 = np.asarray(dof_handler_pc.get_field_slice(pc_field1), dtype=int)
        pc_coords0 = get_pycutfem_dof_coords(dof_handler_pc, pc_field0)
        pc_to_fx = one_to_one_map_coords(pc_coords0, vec_fx_coords)
        P[pc_slice0] = vec_fx_parent0[pc_to_fx]
        P[pc_slice1] = vec_fx_parent1[pc_to_fx]
        fx_coords_all[vec_fx_parent0] = vec_fx_coords
        fx_coords_all[vec_fx_parent1] = W1.tabulate_dof_coordinates()[:, :2]

    for parent_idx, fld in ((4, "alpha"), (5, "mu_alpha")):
        Ws, S_map_fx = W_fenicsx.sub(parent_idx).collapse()
        fx_coords = Ws.tabulate_dof_coordinates()[:, :2]
        fx_parent = np.asarray(S_map_fx, dtype=int)
        pc_slice = np.asarray(dof_handler_pc.get_field_slice(fld), dtype=int)
        pc_coords = get_pycutfem_dof_coords(dof_handler_pc, fld)
        pc_to_fx = one_to_one_map_coords(pc_coords, fx_coords)
        P[pc_slice] = fx_parent[pc_to_fx]
        fx_coords_all[fx_parent] = fx_coords

    print("Seboldt H(div) DoF map discovered successfully.")
    return {
        "P": P,
        "fx_coords_all": fx_coords_all,
        "v_pc_slice": v_pc_slice,
        "v_pc_to_fx_collapsed": np.asarray(v_pc_to_fx, dtype=int),
    }


def initialize_benchmark7_hdiv_functions(problem, fenicsx, map_info):
    print("Initializing Seboldt H(div) state vectors in pycutfem and FEniCSx...")

    dh = problem["dh"]
    v_coords = get_pycutfem_dof_coords(dh, "v")
    problem["v_k"].nodal_values[:] = 0.18 + 0.04 * v_coords[:, 0] - 0.03 * v_coords[:, 1]
    problem["v_n"].nodal_values[:] = -0.07 + 0.03 * v_coords[:, 0] + 0.02 * v_coords[:, 1]

    problem["p_k"].set_values_from_function(lambda x, y: 0.25 + 0.10 * x - 0.05 * y)
    problem["p_n"].set_values_from_function(lambda x, y: 0.12 + 0.04 * x - 0.03 * y)
    problem["vS_k"].set_values_from_function(lambda x, y: np.array([0.05 + 0.02 * x - 0.01 * y, -0.03 + 0.01 * x + 0.015 * y]))
    problem["vS_n"].set_values_from_function(lambda x, y: np.array([0.03 + 0.01 * x - 0.008 * y, -0.015 + 0.008 * x + 0.010 * y]))
    problem["u_k"].set_values_from_function(lambda x, y: np.array([0.015 + 0.010 * x * (1.0 - x), -0.008 + 0.006 * y * (1.0 - y / 1.5)]))
    problem["u_n"].set_values_from_function(lambda x, y: np.array([0.010 + 0.006 * x * (1.0 - x), -0.005 + 0.004 * y * (1.0 - y / 1.5)]))
    problem["alpha_k"].set_values_from_function(lambda x, y: 0.58 + 0.015 * x + 0.010 * y)
    problem["alpha_n"].set_values_from_function(lambda x, y: 0.54 + 0.010 * x + 0.008 * y)
    problem["mu_k"].set_values_from_function(lambda x, y: 0.10 + 0.020 * x - 0.015 * y)
    problem["mu_n"].set_values_from_function(lambda x, y: 0.07 + 0.015 * x - 0.010 * y)

    x_pc_k = _benchmark7_hdiv_state_vector(problem, suffix="k")
    x_pc_n = _benchmark7_hdiv_state_vector(problem, suffix="n")
    sign_map = np.asarray(map_info["sign_map"], dtype=float)
    P_map = np.asarray(map_info["P"], dtype=int)

    fenicsx["w_k"].x.array[:] = 0.0
    fenicsx["w_n"].x.array[:] = 0.0
    fenicsx["w_k"].x.array[P_map] = sign_map * x_pc_k
    fenicsx["w_n"].x.array[P_map] = sign_map * x_pc_n
    fenicsx["w_k"].x.scatter_forward()
    fenicsx["w_n"].x.scatter_forward()


def run_benchmark7_hdiv_comparison():
    problem_pc, dof_handler_pc, fenicsx, params = setup_benchmark7_hdiv_problems(
        nx=int(os.environ.get("COMP_FENICS_NX", "2")),
        ny=int(os.environ.get("COMP_FENICS_NY", "3")),
    )
    W_fx = fenicsx["W"]
    map_info = create_true_dof_map_benchmark7_hdiv(dof_handler_pc, W_fx)
    P_map = np.asarray(map_info["P"], dtype=int)
    fx_coords_all = np.asarray(map_info["fx_coords_all"], dtype=float)
    v_pc_slice = np.asarray(map_info["v_pc_slice"], dtype=int)
    v_pc_to_fx_collapsed = np.asarray(map_info["v_pc_to_fx_collapsed"], dtype=int)
    use_sparse_compare = _env_truthy("COMP_FENICS_SPARSE_COMPARE", False)

    M_pc_full, _ = assemble_form(
        Equation(inner(problem_pc["dv"], problem_pc["v_test"]) * dx(metadata={"q": int(params["qdeg"])}), None),
        dof_handler_pc,
        bcs=[],
        backend="python",
    )
    M_pc = M_pc_full.tocsr()[v_pc_slice, :][:, v_pc_slice].toarray()

    Vv_fx, _ = W_fx.sub(0).collapse()
    dvv_fx = ufl.TrialFunction(Vv_fx)
    wv_fx = ufl.TestFunction(Vv_fx)
    M_fx_form = dolfinx.fem.form(ufl.inner(dvv_fx, wv_fx) * ufl.dx(metadata={"quadrature_degree": int(params["qdeg"])}))
    A_mass_fx = dolfinx.fem.petsc.assemble_matrix(M_fx_form)
    A_mass_fx.assemble()
    indptr, indices, data = A_mass_fx.getValuesCSR()
    M_fx = csr_matrix((data, indices, indptr), shape=A_mass_fx.getSize()).tocsr()
    M_fx_perm = M_fx[v_pc_to_fx_collapsed, :][:, v_pc_to_fx_collapsed].toarray()

    sign_map = np.ones(dof_handler_pc.total_dofs, dtype=float)
    sign_map[v_pc_slice] = _recover_sign_congruence(M_pc, M_fx_perm)

    # The RT mass matrix fixes signs only up to a global +/- per connected RT block.
    # Anchor that remaining ambiguity with a pressure-divergence coupling.
    p_pc_slice = np.asarray(dof_handler_pc.get_field_slice("p"), dtype=int)
    C_pc_full, _ = assemble_form(
        Equation(problem_pc["q_test"] * div(problem_pc["dv"]) * dx(metadata={"q": int(params["qdeg"])}), None),
        dof_handler_pc,
        bcs=[],
        backend="python",
    )
    C_pc = C_pc_full.tocsr()[p_pc_slice, :][:, v_pc_slice].toarray()

    dw_anchor_fx = ufl.TrialFunction(W_fx)
    w_anchor_fx = ufl.TestFunction(W_fx)
    _, _, _, _, _, _ = ufl.split(dw_anchor_fx)
    _, q_anchor_fx, _, _, _, _ = ufl.split(w_anchor_fx)
    dv_anchor_fx, _, _, _, _, _ = ufl.split(dw_anchor_fx)
    C_fx_form = dolfinx.fem.form(q_anchor_fx * ufl.div(dv_anchor_fx) * ufl.dx(metadata={"quadrature_degree": int(params["qdeg"])}))
    A_cpl_fx = dolfinx.fem.petsc.assemble_matrix(C_fx_form)
    A_cpl_fx.assemble()
    indptr, indices, data = A_cpl_fx.getValuesCSR()
    C_fx_full = csr_matrix((data, indices, indptr), shape=A_cpl_fx.getSize()).tocsr()
    C_fx = C_fx_full[np.asarray(P_map[p_pc_slice], dtype=int), :][:, np.asarray(P_map[v_pc_slice], dtype=int)].toarray()
    score = float(np.sum(C_pc * (C_fx * sign_map[v_pc_slice][None, :])))
    if score < 0.0:
        sign_map[v_pc_slice] *= -1.0

    map_info["sign_map"] = sign_map

    initialize_benchmark7_hdiv_functions(problem_pc, fenicsx, map_info)

    forms_pc = build_benchmark7_seboldt_forms(
        problem_pc,
        qdeg=int(params["qdeg"]),
        dt_c=Constant(float(params["dt"])),
        theta=float(params["theta"]),
        rho_f=float(params["rho_f"]),
        mu_f=float(params["mu_f"]),
        mu_b=float(params["mu_b"]),
        mu_b_model="mu",
        kappa_inv=float(params["kappa_inv"]),
        mu_s=float(params["mu_s"]),
        lambda_s=float(params["lambda_s"]),
        phi_b=float(params["phi_b"]),
        M_alpha=float(params["M_alpha"]),
        gamma_alpha=float(params["gamma_alpha"]),
        eps_alpha=float(params["eps_alpha"]),
        solid_visco_eta=0.0,
        gamma_div=float(params["gamma_div"]),
        gamma_u=0.0,
        u_extension_mode="l2",
        gamma_u_pin=0.0,
        u_cip=0.0,
        u_cip_weight="fluid",
        vS_cip=0.0,
        gamma_vS=None,
        vS_extension_mode=None,
        gamma_vS_pin=None,
        D_phi=0.0,
        phi_diffusion_weight="fluid",
        gamma_phi=0.0,
        phi_supg=0.0,
        phi_cip=0.0,
        alpha_regularization="ch",
        alpha_reg_gamma=1.0,
        alpha_reg_eps_normal=float(params["eps_alpha"]),
        alpha_reg_eps_tangent=0.25 * float(params["eps_alpha"]),
        alpha_reg_eta=1.0e-12,
        alpha_advect_with="vS",
        alpha_advection_form="conservative",
        solid_model="linear",
        kappa_inv_model="spatial",
        enable_phi_evolution=False,
    )

    dw_fx = ufl.TrialFunction(W_fx)
    w_test_fx = ufl.TestFunction(W_fx)
    dv_fx, dp_fx, dvS_fx, du_fx, dalpha_fx, dmu_alpha_fx = ufl.split(dw_fx)
    w_fx, q_fx, eta_vS_fx, eta_u_fx, xi_fx, eta_mu_fx = ufl.split(w_test_fx)
    v_k_fx, p_k_fx, vS_k_fx, u_k_fx, alpha_k_fx, mu_alpha_k_fx = ufl.split(fenicsx["w_k"])
    v_n_fx, p_n_fx, vS_n_fx, u_n_fx, alpha_n_fx, mu_alpha_n_fx = ufl.split(fenicsx["w_n"])

    def eps_fx(v):
        return 0.5 * (ufl.grad(v) + ufl.grad(v).T)

    def div_weighted_fx(weight, grad_weight, vec):
        return weight * ufl.div(vec) + ufl.dot(grad_weight, vec)

    dx_fx = ufl.dx(metadata={"quadrature_degree": int(params["qdeg"])})
    th_fx = fenicsx["theta"]
    one_m_th_fx = dolfinx.fem.Constant(fenicsx["mesh"], 1.0) - th_fx
    inv_dt_fx = dolfinx.fem.Constant(fenicsx["mesh"], 1.0) / fenicsx["dt"]

    C_n_fx = 1.0 - alpha_n_fx * (1.0 - fenicsx["phi_b"])
    B_n_fx = alpha_n_fx * (1.0 - fenicsx["phi_b"])
    gradC_n_fx = -(1.0 - fenicsx["phi_b"]) * ufl.grad(alpha_n_fx)
    gradB_n_fx = (1.0 - fenicsx["phi_b"]) * ufl.grad(alpha_n_fx)
    rho_n_fx = fenicsx["rho_f"] * C_n_fx
    mu_n_fx = (1.0 - alpha_n_fx) * fenicsx["mu_f"] + alpha_n_fx * fenicsx["mu_b"]

    div_C_w_fx = div_weighted_fx(C_n_fx, gradC_n_fx, w_fx)
    div_B_vStest_fx = div_weighted_fx(B_n_fx, gradB_n_fx, eta_vS_fx)
    div_C_vk_fx = div_weighted_fx(C_n_fx, gradC_n_fx, v_k_fx)
    div_B_vSk_fx = div_weighted_fx(B_n_fx, gradB_n_fx, vS_k_fx)

    r_mom_fx = (rho_n_fx * inv_dt_fx) * (ufl.inner(v_k_fx, w_fx) - ufl.inner(v_n_fx, w_fx)) * dx_fx
    r_mom_fx += th_fx * rho_n_fx * ufl.dot(ufl.dot(ufl.grad(v_k_fx), v_n_fx), w_fx) * dx_fx
    r_mom_fx += one_m_th_fx * rho_n_fx * ufl.dot(ufl.dot(ufl.grad(v_n_fx), v_n_fx), w_fx) * dx_fx
    r_mom_fx += th_fx * 2.0 * mu_n_fx * ufl.inner(eps_fx(v_k_fx), eps_fx(w_fx)) * dx_fx
    r_mom_fx += one_m_th_fx * 2.0 * mu_n_fx * ufl.inner(eps_fx(v_n_fx), eps_fx(w_fx)) * dx_fx
    r_mom_fx += -(p_k_fx * div_C_w_fx) * dx_fx

    r_mass_fx = q_fx * (div_C_vk_fx + div_weighted_fx(B_n_fx, gradB_n_fx, vS_k_fx)) * dx_fx

    r_el_k_fx = 2.0 * fenicsx["mu_s"] * ufl.inner(eps_fx(u_k_fx), eps_fx(eta_vS_fx))
    r_el_k_fx += fenicsx["lambda_s"] * ufl.div(u_k_fx) * ufl.div(eta_vS_fx)
    r_el_n_fx = 2.0 * fenicsx["mu_s"] * ufl.inner(eps_fx(u_n_fx), eps_fx(eta_vS_fx))
    r_el_n_fx += fenicsx["lambda_s"] * ufl.div(u_n_fx) * ufl.div(eta_vS_fx)

    r_skel_fx = th_fx * alpha_n_fx * r_el_k_fx * dx_fx
    r_skel_fx += one_m_th_fx * alpha_n_fx * r_el_n_fx * dx_fx
    r_skel_fx += -(p_k_fx * div_B_vStest_fx) * dx_fx

    beta_n_fx = alpha_n_fx * fenicsx["mu_f"] * fenicsx["kappa_inv"]
    diff_k_fx = v_k_fx - vS_k_fx
    diff_n_fx = v_n_fx - vS_n_fx
    r_mom_fx += beta_n_fx * (th_fx * ufl.dot(diff_k_fx, w_fx) + one_m_th_fx * ufl.dot(diff_n_fx, w_fx)) * dx_fx
    r_skel_fx += -beta_n_fx * (th_fx * ufl.dot(diff_k_fx, eta_vS_fx) + one_m_th_fx * ufl.dot(diff_n_fx, eta_vS_fx)) * dx_fx

    if float(params["gamma_div"]) != 0.0:
        mass_res_fx = div_C_vk_fx + div_weighted_fx(B_n_fx, gradB_n_fx, vS_k_fx)
        r_mom_fx += fenicsx["gamma_div"] * mass_res_fx * div_C_w_fx * dx_fx
        r_skel_fx += fenicsx["gamma_div"] * mass_res_fx * div_B_vStest_fx * dx_fx

    r_kin_fx = inv_dt_fx * (ufl.inner(u_k_fx, eta_u_fx) - ufl.inner(u_n_fx, eta_u_fx)) * dx_fx
    r_kin_fx += th_fx * ufl.dot(ufl.dot(ufl.grad(u_k_fx), vS_n_fx), eta_u_fx) * dx_fx
    r_kin_fx += one_m_th_fx * ufl.dot(ufl.dot(ufl.grad(u_n_fx), vS_n_fx), eta_u_fx) * dx_fx
    r_kin_fx += -(th_fx * ufl.dot(vS_k_fx, eta_u_fx) + one_m_th_fx * ufl.dot(vS_n_fx, eta_u_fx)) * dx_fx

    div_vS_n_fx = ufl.div(vS_n_fx)
    r_alpha_fx = xi_fx * ((alpha_k_fx - alpha_n_fx) * inv_dt_fx) * dx_fx
    r_alpha_fx += th_fx * xi_fx * (ufl.dot(ufl.grad(alpha_k_fx), vS_n_fx) + alpha_k_fx * div_vS_n_fx) * dx_fx
    r_alpha_fx += one_m_th_fx * xi_fx * (ufl.dot(ufl.grad(alpha_n_fx), vS_n_fx) + alpha_n_fx * div_vS_n_fx) * dx_fx
    r_alpha_fx += fenicsx["M_alpha"] * ufl.dot(ufl.grad(mu_alpha_k_fx), ufl.grad(xi_fx)) * dx_fx

    Wp_alpha_fx = 2.0 * alpha_k_fx * (1.0 - alpha_k_fx) * (1.0 - 2.0 * alpha_k_fx)
    r_mu_alpha_fx = eta_mu_fx * mu_alpha_k_fx * dx_fx
    r_mu_alpha_fx += -(fenicsx["gamma_alpha"] * fenicsx["eps_alpha"]) * ufl.dot(ufl.grad(alpha_k_fx), ufl.grad(eta_mu_fx)) * dx_fx
    r_mu_alpha_fx += -(fenicsx["gamma_alpha"] / fenicsx["eps_alpha"]) * eta_mu_fx * Wp_alpha_fx * dx_fx

    r_total_fx = r_mom_fx + r_mass_fx + r_skel_fx + r_kin_fx + r_alpha_fx + r_mu_alpha_fx

    a_mom_fx = ufl.derivative(r_mom_fx, fenicsx["w_k"], dw_fx)
    a_mass_fx = ufl.derivative(r_mass_fx, fenicsx["w_k"], dw_fx)
    a_skel_fx = ufl.derivative(r_skel_fx, fenicsx["w_k"], dw_fx)
    a_kin_fx = ufl.derivative(r_kin_fx, fenicsx["w_k"], dw_fx)
    a_alpha_fx = ufl.derivative(r_alpha_fx, fenicsx["w_k"], dw_fx)
    a_mu_alpha_fx = ufl.derivative(r_mu_alpha_fx, fenicsx["w_k"], dw_fx)
    a_total_fx = ufl.derivative(r_total_fx, fenicsx["w_k"], dw_fx)

    terms = {
        "Seboldt Hdiv momentum (res)": {"pc": forms_pc.r_momentum, "fx": r_mom_fx, "mat": False},
        "Seboldt Hdiv mass (res)": {"pc": forms_pc.r_mass, "fx": r_mass_fx, "mat": False},
        "Seboldt Hdiv skeleton (res)": {"pc": forms_pc.r_skeleton, "fx": r_skel_fx, "mat": False},
        "Seboldt Hdiv kinematics (res)": {"pc": forms_pc.r_kinematics, "fx": r_kin_fx, "mat": False},
        "Seboldt Hdiv alpha (res)": {"pc": forms_pc.r_alpha, "fx": r_alpha_fx, "mat": False},
        "Seboldt Hdiv mu_alpha (res)": {"pc": forms_pc.r_mu_alpha, "fx": r_mu_alpha_fx, "mat": False},
        "Seboldt Hdiv total residual": {"pc": forms_pc.residual_form, "fx": r_total_fx, "mat": False},
        "Seboldt Hdiv momentum (jac)": {"pc": forms_pc.a_momentum, "fx": a_mom_fx, "mat": True},
        "Seboldt Hdiv mass (jac)": {"pc": forms_pc.a_mass, "fx": a_mass_fx, "mat": True},
        "Seboldt Hdiv skeleton (jac)": {"pc": forms_pc.a_skeleton, "fx": a_skel_fx, "mat": True},
        "Seboldt Hdiv kinematics (jac)": {"pc": forms_pc.a_kinematics, "fx": a_kin_fx, "mat": True},
        "Seboldt Hdiv alpha (jac)": {"pc": forms_pc.a_alpha, "fx": a_alpha_fx, "mat": True},
        "Seboldt Hdiv mu_alpha (jac)": {"pc": forms_pc.a_mu_alpha, "fx": a_mu_alpha_fx, "mat": True},
        "Seboldt Hdiv total jacobian": {"pc": forms_pc.jacobian_form, "fx": a_total_fx, "mat": True},
    }

    filter_terms = os.environ.get("COMP_FENICS_TERMS")
    if filter_terms:
        allowed = {name.strip() for name in filter_terms.split(",") if name.strip()}
        terms = {k: v for k, v in terms.items() if k in allowed}
        print(f"Running filtered terms only: {sorted(terms)}")
    else:
        print(f"Running full Seboldt H(div) suite: {sorted(terms)}")

    fenics_ref = {}
    for name, spec in terms.items():
        try:
            form_fx_compiled = dolfinx.fem.form(spec["fx"])
            if spec["mat"]:
                A = dolfinx.fem.petsc.assemble_matrix(form_fx_compiled)
                A.assemble()
                indptr, indices, data = A.getValuesCSR()
                mat_fx = csr_matrix((data, indices, indptr), shape=A.getSize()).tocsr()
                fenics_ref[name] = mat_fx if use_sparse_compare else mat_fx.toarray()
            else:
                vec = dolfinx.fem.petsc.assemble_vector(form_fx_compiled)
                fenics_ref[name] = vec.array.copy()
        except Exception as exc:
            print(f"❌ FEniCSx assembly failed for '{name}': {exc}")
            fenics_ref[name] = None

    backends_spec = os.environ.get("BACKEND", "jit")
    if backends_spec.strip().lower() == "all":
        backends = ["python", "jit", "cpp"]
    else:
        backends = [b.strip() for b in backends_spec.split(",") if b.strip()]
    parity_rtol = float(os.environ.get("COMP_FENICS_PARITY_RTOL", "1e-9"))
    parity_atol = float(os.environ.get("COMP_FENICS_PARITY_ATOL", "1e-9"))
    ref_backend = "python" if "python" in backends else (backends[0] if backends else "python")
    ref_pc = {}

    for backend_type in backends:
        print("\n" + "=" * 70)
        print(f"SEBOLDT H(div) COMPARISON (backend={backend_type}, qdeg={int(params['qdeg'])})")
        print("=" * 70)

        failed_tests = []
        success_count = 0

        for name, spec in terms.items():
            if fenics_ref.get(name) is None:
                failed_tests.append(f"{name} (fenics-assemble)")
                continue

            J_pc, R_pc = None, None
            print(f"\nCompiling/assembling '{name}' [backend={backend_type}]")
            try:
                if spec["mat"]:
                    J_pc, _ = assemble_form(Equation(spec["pc"], None), dof_handler_pc, quad_degree=int(params["qdeg"]), bcs=[], backend=backend_type)
                else:
                    _, R_pc = assemble_form(Equation(None, spec["pc"]), dof_handler_pc, bcs=[], backend=backend_type)
            except Exception as exc:
                print(f"❌ pycutfem assembly failed for '{name}' on backend '{backend_type}': {exc}")
                failed_tests.append(f"{name} (assemble-{backend_type})")
                continue

            if backend_type == ref_backend:
                if spec["mat"]:
                    ref_pc[name] = J_pc.tocsr().copy() if use_sparse_compare else J_pc.toarray()
                else:
                    ref_pc[name] = np.asarray(R_pc, dtype=float).copy()
            else:
                try:
                    if spec["mat"]:
                        if use_sparse_compare:
                            ok, max_abs, worst = _sparse_matrix_allclose(
                                J_pc.tocsr(),
                                ref_pc[name],
                                rtol=parity_rtol,
                                atol=parity_atol,
                            )
                            if not ok:
                                raise AssertionError(
                                    f"sparse mismatch: max_abs={max_abs:.3e}, worst={worst}"
                                )
                        else:
                            np.testing.assert_allclose(J_pc.toarray(), ref_pc[name], rtol=parity_rtol, atol=parity_atol)
                    else:
                        np.testing.assert_allclose(np.asarray(R_pc, dtype=float), ref_pc[name], rtol=parity_rtol, atol=parity_atol)
                    print(f"✅ pycutfem backend parity OK vs '{ref_backend}'.")
                except Exception as exc:
                    print(f"❌ pycutfem backend parity FAILED vs '{ref_backend}': {exc}")
                    failed_tests.append(f"{name} (parity-{backend_type}-vs-{ref_backend})")

            J_fx = fenics_ref[name] if spec["mat"] else None
            R_fx = fenics_ref[name] if not spec["mat"] else None
            is_success = compare_term(
                f"{name} [backend={backend_type}]",
                J_pc,
                R_pc,
                J_fx,
                R_fx,
                P_map,
                dof_handler_pc,
                W_fx,
                rtol=1e-8,
                atol=1e-8,
                sign_map=sign_map,
                fx_coords_all=fx_coords_all,
            )
            if is_success:
                success_count += 1
            else:
                failed_tests.append(name)

        print_test_summary(success_count, failed_tests)


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
        'mesh': mesh_fx,
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

def compare_term(
    term_name,
    J_pc,
    R_pc,
    J_fx,
    R_fx,
    P_map,
    dof_handler_pc,
    W_fenicsx,
    rtol=1e-8,
    atol=1e-8,
    *,
    sign_map=None,
    fx_coords_all=None,
    transform=None,
):
    print("\n" + f"--- Comparing Term: {term_name} ---")

    output_dir = os.environ.get("COMP_FENICS_OUTDIR", "comparison_outputs")
    os.makedirs(output_dir, exist_ok=True)
    safe_term_name = term_name.replace(' ', '_').lower()
    is_successful = True
    write_xlsx = _env_truthy("COMP_FENICS_WRITE_XLSX", True)
    transform_arr = None if transform is None else np.asarray(transform, dtype=float)
    if transform_arr is None:
        if P_map is None:
            raise ValueError("compare_term requires either P_map or transform.")
        if sign_map is None:
            sign_map = np.ones(len(P_map), dtype=float)
        sign_map = np.asarray(sign_map, dtype=float).reshape(-1)

    if R_pc is not None and R_fx is not None:
        filename = os.path.join(output_dir, f"{safe_term_name}_residual.xlsx")
        if transform_arr is not None:
            R_fx_reordered = np.asarray(transform_arr.T @ np.asarray(R_fx, dtype=float), dtype=float).reshape(-1)
        else:
            R_fx_reordered = sign_map * np.asarray(R_fx, dtype=float)[P_map]
        
        R_pc_flat = R_pc.flatten()
        R_fx_reordered_flat = R_fx_reordered.flatten()

        comparison_data = None
        if transform_arr is None:
            pc_coords = get_all_pycutfem_dof_coords(dof_handler_pc)
            if fx_coords_all is None:
                fx_coords_all = get_all_fenicsx_dof_coords(W_fenicsx)
            fx_coords_reordered = fx_coords_all[P_map]
            comparison_data = {
                'pc_dof_index': np.arange(dof_handler_pc.total_dofs),
                'pc_coord_x': pc_coords[:, 0], 'pc_coord_y': pc_coords[:, 1], 'pc_residual': R_pc_flat,
                'fx_coord_x': fx_coords_reordered[:, 0], 'fx_coord_y': fx_coords_reordered[:, 1],
                'fx_reordered_residual': R_fx_reordered_flat, 'abs_difference': np.abs(R_pc_flat - R_fx_reordered_flat)
            }
        if write_xlsx:
            if comparison_data is None:
                print("ℹ️ Skipping residual Excel export for transform-based comparison.")
            else:
                pd.DataFrame(comparison_data).to_excel(filename, sheet_name='residual_comparison', index=False)
                print(f"✅ Residual comparison data saved to '{filename}'")
        else:
            print("ℹ️ Skipping residual Excel export (COMP_FENICS_WRITE_XLSX=0).")

        try:
            np.testing.assert_allclose(R_pc_flat, R_fx_reordered_flat, rtol=rtol, atol=atol)
            print(f"✅ Residual vector for '{term_name}' is numerically equivalent.")
        except AssertionError as e:
            print(f"❌ Residual vector for '{term_name}' is NOT equivalent!")
            print(e)
            is_successful = False

    if J_pc is not None and J_fx is not None:
        use_sparse_compare = (
            _env_truthy("COMP_FENICS_SPARSE_COMPARE", False)
            or sp.issparse(J_pc)
            or sp.issparse(J_fx)
        )
        if use_sparse_compare:
            J_pc_sparse = J_pc.tocsr() if sp.issparse(J_pc) else csr_matrix(np.asarray(J_pc, dtype=float))
            if transform_arr is not None:
                T_sparse = csr_matrix(transform_arr)
                if sp.issparse(J_fx):
                    J_fx_reordered = (T_sparse.T @ J_fx.tocsr() @ T_sparse).tocsr()
                else:
                    J_fx_reordered = (T_sparse.T @ csr_matrix(np.asarray(J_fx, dtype=float)) @ T_sparse).tocsr()
            else:
                if sp.issparse(J_fx):
                    J_fx_reordered = J_fx.tocsr()[P_map, :][:, P_map]
                else:
                    J_fx_reordered = csr_matrix(np.asarray(J_fx, dtype=float)[P_map, :][:, P_map])
                D_sign = sp.diags(sign_map)
                J_fx_reordered = (D_sign @ J_fx_reordered @ D_sign).tocsr()
            ok, max_abs, worst = _sparse_matrix_allclose(J_pc_sparse, J_fx_reordered, rtol=rtol, atol=atol)
            print(
                f"ℹ️ Sparse Jacobian compare: nnz_pycutfem={J_pc_sparse.nnz}, "
                f"nnz_fenics={J_fx_reordered.nnz}, max_abs={max_abs:.3e}"
            )
            if ok:
                print(f"✅ Jacobian matrix for '{term_name}' is numerically equivalent.")
            else:
                print(f"❌ Jacobian matrix for '{term_name}' is NOT equivalent!")
                if worst is not None:
                    row, col, a_val, b_val, diff, tol = worst
                    print(
                        f"   Worst entry ({row}, {col}): pycutfem={a_val:.16e}, "
                        f"fenics={b_val:.16e}, diff={diff:.3e}, tol={tol:.3e}"
                    )
                is_successful = False
        else:
            filename = os.path.join(output_dir, f"{safe_term_name}_jacobian.xlsx")
            J_pc_dense = J_pc.toarray()
            if transform_arr is not None:
                J_fx_reordered = np.asarray(transform_arr.T @ np.asarray(J_fx, dtype=float) @ transform_arr, dtype=float)
            else:
                J_fx_reordered = np.asarray(J_fx, dtype=float)[P_map, :][:, P_map]
                J_fx_reordered = sign_map[:, None] * J_fx_reordered * sign_map[None, :]
            if write_xlsx:
                if transform_arr is not None:
                    print("ℹ️ Skipping Jacobian Excel export for transform-based comparison.")
                else:
                    with pd.ExcelWriter(filename) as writer:
                        pd.DataFrame(J_pc_dense).to_excel(writer, sheet_name='pycutfem', index=False, header=False)
                        pd.DataFrame(J_fx_reordered).to_excel(writer, sheet_name='fenics', index=False, header=False)
                        pd.DataFrame(np.abs(J_pc_dense - J_fx_reordered) < 1e-12).to_excel(
                            writer, sheet_name='difference', index=False, header=False
                        )
                    print(f"✅ Jacobian matrices saved to '{filename}'")
            else:
                print("ℹ️ Skipping Jacobian Excel export (COMP_FENICS_WRITE_XLSX=0).")
            try:
                np.testing.assert_allclose(J_pc_dense, J_fx_reordered, rtol=rtol, atol=atol)
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
#  Semi-smooth contact (PositivePart/Heaviside) comparison harness vs FEniCSx
# ==============================================================================


def setup_contact_problems(*, nx: int = 2, ny: int = 2):
    """
    Build a small mixed space (u, p) for a boundary contact-form comparison:
      pycutfem: u = (ux,uy) Q2, p Q1
      dolfinx : u = vector Q2, p Q1  (quadrilateral)

    The forms exercise `pos_part` and `heaviside` on a boundary integral.
    """
    # --- pycutfem mesh (Q2 geometry so dof coordinates match dolfinx's Q2 dofs) ---
    nodes_q2, elems_q2, _, corners_q2 = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=2)
    mesh_q2 = Mesh(
        nodes=nodes_q2,
        element_connectivity=elems_q2,
        elements_corner_nodes=corners_q2,
        element_type="quad",
        poly_order=2,
    )
    mesh_q2.tag_boundary_edges({"all": lambda x, y: True})

    mixed_element_pc = MixedElement(mesh_q2, field_specs={"ux": 2, "uy": 2, "p": 1})
    dof_handler_pc = DofHandler(mixed_element_pc, method="cg")

    # --- dolfinx mesh and mixed space ---
    mesh_fx = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD,
        nx,
        ny,
        cell_type=dolfinx.mesh.CellType.quadrilateral,
    )
    cell = mesh_fx.ufl_cell()
    cell_name = cell.cellname() if hasattr(cell, "cellname") else cell.name
    V_el = basix.ufl.element("Lagrange", cell_name, 2, shape=(2,))
    Q_el = basix.ufl.element("Lagrange", cell_name, 1)
    W_el = mixed_element([V_el, Q_el])
    if hasattr(dolfinx.fem, "functionspace"):
        W_fx = dolfinx.fem.functionspace(mesh_fx, W_el)
    else:
        W_fx = dolfinx.fem.FunctionSpace(mesh_fx, W_el)

    w_k_fx = dolfinx.fem.Function(W_fx)
    n_fx = ufl.FacetNormal(mesh_fx)
    ds_fx = ufl.ds(domain=mesh_fx)

    gamma_val = float(os.environ.get("COMP_FENICS_CONTACT_GAMMA", "2.0"))
    g_val = float(os.environ.get("COMP_FENICS_CONTACT_G", "0.5"))
    gamma_fx = dolfinx.fem.Constant(mesh_fx, PETSc.ScalarType(gamma_val))
    g_fx = dolfinx.fem.Constant(mesh_fx, PETSc.ScalarType(g_val))

    fenicsx = {
        "mesh": mesh_fx,
        "W": W_fx,
        "w_k": w_k_fx,
        "normal": n_fx,
        "ds": ds_fx,
        "gamma": gamma_fx,
        "g": g_fx,
    }
    return dof_handler_pc, fenicsx


def run_contact_comparison():
    """
    Compare a semi-smooth contact residual/Jacobian (PositivePart + Heaviside)
    between pycutfem and dolfinx on a simple (u,p) mixed space.

    Two deterministic cases are tested:
      - inactive: u=(0,0)       => P<0 everywhere on ∂Ω  => residual/jacobian = 0
      - active  : u=(0,-1)      => P>0 only on bottom edge (n=(0,-1))
    """
    nx = int(os.environ.get("COMP_FENICS_NX", "2"))
    ny = int(os.environ.get("COMP_FENICS_NY", os.environ.get("COMP_FENICS_NX", "2")))
    qdeg = int(os.environ.get("COMP_FENICS_QDEG", "6"))

    dof_handler_pc, fenicsx = setup_contact_problems(nx=nx, ny=ny)
    W_fx = fenicsx["W"]
    P_map = create_true_dof_map(dof_handler_pc, W_fx)

    # --- pycutfem trial/test/state ---
    V_pc = FunctionSpace("V", ["ux", "uy"])
    Q_pc = FunctionSpace("Q", ["p"])
    du_pc = VectorTrialFunction(V_pc, dof_handler=dof_handler_pc)
    dp_pc = TrialFunction(name="dp", field_name="p", dof_handler=dof_handler_pc)
    v_pc = VectorTestFunction(V_pc, dof_handler=dof_handler_pc)
    q_pc = TestFunction(name="q", field_name="p", dof_handler=dof_handler_pc)
    _ = (dp_pc, q_pc)  # included to keep the mixed layout consistent; not used in the contact forms

    u_pc = VectorFunction("u", ["ux", "uy"], dof_handler_pc)
    p_pc = Function("p", "p", dof_handler_pc)
    n_pc = FacetNormal()

    gamma_pc = Constant(float(fenicsx["gamma"].value))
    g_pc = Constant(float(fenicsx["g"].value))

    P_pc = dot(u_pc, n_pc) - g_pc
    r_pc = gamma_pc * pos_part(P_pc) * dot(v_pc, n_pc) * dS()
    a_pc = gamma_pc * heaviside(P_pc) * dot(du_pc, n_pc) * dot(v_pc, n_pc) * dS()

    # --- dolfinx forms (explicit semi-smooth Jacobian) ---
    w_k_fx = fenicsx["w_k"]
    u_fx, p_fx = ufl.split(w_k_fx)
    dw_fx = ufl.TrialFunction(W_fx)
    du_fx, dp_fx = ufl.split(dw_fx)
    w_test_fx = ufl.TestFunction(W_fx)
    v_fx, q_fx = ufl.split(w_test_fx)
    _ = (p_fx, dp_fx, q_fx)  # p is present only to mirror the mixed layout

    gamma_fx = fenicsx["gamma"]
    g_fx = fenicsx["g"]
    n_fx = fenicsx["normal"]
    ds_fx = fenicsx["ds"]

    P_fx = ufl.dot(u_fx, n_fx) - g_fx
    r_fx = gamma_fx * ufl.max_value(P_fx, 0.0) * ufl.dot(v_fx, n_fx) * ds_fx
    H_fx = ufl.conditional(ufl.gt(P_fx, 0.0), 1.0, 0.0)
    a_fx = gamma_fx * H_fx * ufl.dot(du_fx, n_fx) * ufl.dot(v_fx, n_fx) * ds_fx

    # Cases: choose u so the active set is deterministic and away from P=0.
    cases = {
        "inactive": (0.0, 0.0),
        "active": (0.0, -1.0),
    }

    backends_spec = os.environ.get("BACKEND", "jit")
    if backends_spec.strip().lower() == "all":
        backends = ["python", "jit", "cpp"]
    else:
        backends = [b.strip() for b in backends_spec.split(",") if b.strip()]

    failed_tests = []
    success_count = 0

    for backend_type in backends:
        print("\n" + "=" * 70)
        print(f"CONTACT COMPARISON (backend={backend_type})")
        print("=" * 70)

        for case_name, (ux_val, uy_val) in cases.items():
            # --- set pycutfem coefficients ---
            u_pc.set_values_from_function(lambda x, y, ux=float(ux_val), uy=float(uy_val): np.array([ux, uy], float))
            p_pc.nodal_values[:] = 0.0

            # --- set dolfinx coefficients via coordinate-based DoF map ---
            U_full_pc = np.zeros(dof_handler_pc.total_dofs, dtype=float)
            U_full_pc[dof_handler_pc.get_field_slice("ux")] = float(ux_val)
            U_full_pc[dof_handler_pc.get_field_slice("uy")] = float(uy_val)
            U_full_pc[dof_handler_pc.get_field_slice("p")] = 0.0

            w_k_fx.x.array[:] = 0.0
            w_k_fx.x.array[np.asarray(P_map, dtype=int)] = U_full_pc
            w_k_fx.x.scatter_forward()

            name = f"Contact ({case_name})"
            print(f"\nCompiling/assembling '{name}' [backend={backend_type}, qdeg={qdeg}]")

            try:
                # Assemble pycutfem
                J_pc, _ = assemble_form(
                    Equation(a_pc, None),
                    dof_handler_pc,
                    quad_degree=qdeg,
                    bcs=[],
                    backend=backend_type,
                )
                _, R_pc = assemble_form(
                    Equation(None, r_pc),
                    dof_handler_pc,
                    quad_degree=qdeg,
                    bcs=[],
                    backend=backend_type,
                )
            except Exception as exc:
                print(f"❌ pycutfem assembly failed for '{name}' on backend '{backend_type}': {exc}")
                failed_tests.append(f"{name} (assemble-{backend_type})")
                continue

            try:
                # Assemble dolfinx
                r_fx_form = dolfinx.fem.form(r_fx)
                a_fx_form = dolfinx.fem.form(a_fx)
                vec = dolfinx.fem.petsc.assemble_vector(r_fx_form)
                R_fx = vec.array

                A = dolfinx.fem.petsc.assemble_matrix(a_fx_form)
                A.assemble()
                indptr, indices, data = A.getValuesCSR()
                J_fx = csr_matrix((data, indices, indptr), shape=A.getSize()).toarray()
            except Exception as exc:
                print(f"❌ dolfinx assembly failed for '{name}': {exc}")
                failed_tests.append(f"{name} (fenics-assemble)")
                continue

            is_success = compare_term(
                f"{name} [backend={backend_type}]",
                J_pc,
                R_pc,
                J_fx,
                R_fx,
                P_map,
                dof_handler_pc,
                W_fx,
                rtol=1e-9,
                atol=1e-9,
            )
            if is_success:
                success_count += 1
            else:
                failed_tests.append(f"{name} (mismatch-{backend_type})")

    print_test_summary(success_count, failed_tests)


def run_component_semantics_comparison():
    pc, dof_handler_pc, fenicsx = setup_problems()
    P_map = create_true_dof_map(dof_handler_pc, fenicsx["W"])
    initialize_functions(pc, fenicsx, dof_handler_pc, P_map)

    qdeg = int(os.environ.get("COMP_FENICS_QDEG", 6))
    backend_types = tuple(
        backend.strip()
        for backend in os.environ.get("COMP_FENICS_BACKENDS", "python,jit,cpp").split(",")
        if backend.strip()
    )
    metadata = {"quadrature_degree": qdeg}

    W_fx = fenicsx["W"]
    w_trial = ufl.TrialFunction(W_fx)
    w_test = ufl.TestFunction(W_fx)
    du_fx, dp_fx = ufl.split(w_trial)
    v_fx, q_fx = ufl.split(w_test)
    u_k_fx, p_k_fx = ufl.split(fenicsx["u_k_p_k"])

    B_arr = np.array([[2.0, -0.5], [1.0, 3.0]], dtype=float)
    B_pc = Constant(B_arr, dim=2)
    B_fx = dolfinx.fem.Constant(fenicsx["mesh"], B_arr)

    form_specs = [
        {
            "name": "ScalarGradDot Bilinear Left",
            "kind": "jac",
            "pc": dot(pc["u_k"], grad(pc["dp"])) * pc["q"] * dx(metadata=metadata),
            "fx": ufl.dot(u_k_fx, ufl.grad(dp_fx)) * q_fx * ufl.dx(metadata=metadata),
        },
        {
            "name": "ScalarGradDot Bilinear Right",
            "kind": "jac",
            "pc": dot(grad(pc["dp"]), pc["u_k"]) * pc["q"] * dx(metadata=metadata),
            "fx": ufl.dot(ufl.grad(dp_fx), u_k_fx) * q_fx * ufl.dx(metadata=metadata),
        },
        {
            "name": "ScalarGradDot Residual Left",
            "kind": "res",
            "pc": dot(pc["u_k"], grad(pc["p_k"])) * pc["q"] * dx(metadata=metadata),
            "fx": ufl.dot(u_k_fx, ufl.grad(p_k_fx)) * q_fx * ufl.dx(metadata=metadata),
        },
        {
            "name": "ScalarGradDot Residual Right",
            "kind": "res",
            "pc": dot(grad(pc["p_k"]), pc["u_k"]) * pc["q"] * dx(metadata=metadata),
            "fx": ufl.dot(ufl.grad(p_k_fx), u_k_fx) * q_fx * ufl.dx(metadata=metadata),
        },
        {
            "name": "ScalarGradMatrix Bilinear Left",
            "kind": "jac",
            "pc": inner(B_pc * grad(pc["dp"]), grad(pc["q"])) * dx(metadata=metadata),
            "fx": ufl.inner(ufl.dot(B_fx, ufl.grad(dp_fx)), ufl.grad(q_fx)) * ufl.dx(metadata=metadata),
        },
        {
            "name": "ScalarGradMatrix Bilinear Right",
            "kind": "jac",
            "pc": inner(grad(pc["dp"]) * B_pc, grad(pc["q"])) * dx(metadata=metadata),
            "fx": ufl.inner(ufl.dot(ufl.grad(dp_fx), B_fx), ufl.grad(q_fx)) * ufl.dx(metadata=metadata),
        },
        {
            "name": "ScalarGradMatrix Residual Left",
            "kind": "res",
            "pc": inner(B_pc * grad(pc["p_k"]), grad(pc["q"])) * dx(metadata=metadata),
            "fx": ufl.inner(ufl.dot(B_fx, ufl.grad(p_k_fx)), ufl.grad(q_fx)) * ufl.dx(metadata=metadata),
        },
        {
            "name": "ScalarGradMatrix Residual Right",
            "kind": "res",
            "pc": inner(grad(pc["p_k"]) * B_pc, grad(pc["q"])) * dx(metadata=metadata),
            "fx": ufl.inner(ufl.dot(ufl.grad(p_k_fx), B_fx), ufl.grad(q_fx)) * ufl.dx(metadata=metadata),
        },
        {
            "name": "VectorGrad Row Bilinear 0",
            "kind": "jac",
            "pc": inner(grad(pc["du"])[0], grad(pc["v"])[0]) * dx(metadata=metadata),
            "fx": ufl.inner(ufl.grad(du_fx[0]), ufl.grad(v_fx[0])) * ufl.dx(metadata=metadata),
        },
        {
            "name": "VectorGrad Row Bilinear 1",
            "kind": "jac",
            "pc": inner(grad(pc["du"])[1], grad(pc["v"])[1]) * dx(metadata=metadata),
            "fx": ufl.inner(ufl.grad(du_fx[1]), ufl.grad(v_fx[1])) * ufl.dx(metadata=metadata),
        },
        {
            "name": "VectorGrad Row Residual 0",
            "kind": "res",
            "pc": inner(grad(pc["u_k"])[0], grad(pc["v"])[0]) * dx(metadata=metadata),
            "fx": ufl.inner(ufl.grad(u_k_fx[0]), ufl.grad(v_fx[0])) * ufl.dx(metadata=metadata),
        },
        {
            "name": "VectorGrad Row Residual 1",
            "kind": "res",
            "pc": inner(grad(pc["u_k"])[1], grad(pc["v"])[1]) * dx(metadata=metadata),
            "fx": ufl.inner(ufl.grad(u_k_fx[1]), ufl.grad(v_fx[1])) * ufl.dx(metadata=metadata),
        },
        {
            "name": "VectorGrad Components Bilinear Left",
            "kind": "jac",
            "pc": sum(
                grad(pc["du"])[i][j] * pc["u_k"][j] * pc["v"][i]
                for i in range(2)
                for j in range(2)
            ) * dx(metadata=metadata),
            "fx": sum(
                ufl.grad(du_fx[i])[j] * u_k_fx[j] * v_fx[i]
                for i in range(2)
                for j in range(2)
            ) * ufl.dx(metadata=metadata),
        },
        {
            "name": "VectorGrad Components Bilinear Right",
            "kind": "jac",
            "pc": sum(
                pc["u_k"][i] * grad(pc["du"])[i][j] * pc["v"][j]
                for i in range(2)
                for j in range(2)
            ) * dx(metadata=metadata),
            "fx": sum(
                u_k_fx[i] * ufl.grad(du_fx[i])[j] * v_fx[j]
                for i in range(2)
                for j in range(2)
            ) * ufl.dx(metadata=metadata),
        },
        {
            "name": "VectorGrad Components Residual Left",
            "kind": "res",
            "pc": sum(
                grad(pc["u_k"])[i][j] * pc["u_k"][j] * pc["v"][i]
                for i in range(2)
                for j in range(2)
            ) * dx(metadata=metadata),
            "fx": sum(
                ufl.grad(u_k_fx[i])[j] * u_k_fx[j] * v_fx[i]
                for i in range(2)
                for j in range(2)
            ) * ufl.dx(metadata=metadata),
        },
        {
            "name": "VectorGrad Components Residual Right",
            "kind": "res",
            "pc": sum(
                pc["u_k"][i] * grad(pc["u_k"])[i][j] * pc["v"][j]
                for i in range(2)
                for j in range(2)
            ) * dx(metadata=metadata),
            "fx": sum(
                u_k_fx[i] * ufl.grad(u_k_fx[i])[j] * v_fx[j]
                for i in range(2)
                for j in range(2)
            ) * ufl.dx(metadata=metadata),
        },
    ]

    success_count = 0
    failed_tests = []
    for backend_type in backend_types:
        print(f"\nRunning component semantics comparison with backend='{backend_type}', qdeg={qdeg}")
        for spec in form_specs:
            name = f"{spec['name']} [backend={backend_type}]"
            print(f"\nCompiling/assembling '{name}'")
            try:
                if spec["kind"] == "jac":
                    J_pc, _ = assemble_form(
                        Equation(spec["pc"], None),
                        dof_handler_pc,
                        quad_degree=qdeg,
                        bcs=[],
                        backend=backend_type,
                    )
                    J_fx_form = dolfinx.fem.form(spec["fx"])
                    A = dolfinx.fem.petsc.assemble_matrix(J_fx_form)
                    A.assemble()
                    indptr, indices, data = A.getValuesCSR()
                    J_fx = csr_matrix((data, indices, indptr), shape=A.getSize()).toarray()
                    R_pc = None
                    R_fx = None
                else:
                    _, R_pc = assemble_form(
                        Equation(None, spec["pc"]),
                        dof_handler_pc,
                        quad_degree=qdeg,
                        bcs=[],
                        backend=backend_type,
                    )
                    R_fx_form = dolfinx.fem.form(spec["fx"])
                    vec = dolfinx.fem.petsc.assemble_vector(R_fx_form)
                    R_fx = vec.array
                    J_pc = None
                    J_fx = None
            except Exception as exc:
                print(f"❌ Assembly failed for '{name}': {exc}")
                failed_tests.append(f"{name} (assemble)")
                continue

            is_success = compare_term(
                name,
                J_pc,
                R_pc,
                J_fx,
                R_fx,
                P_map,
                dof_handler_pc,
                W_fx,
                rtol=1e-12,
                atol=1e-12,
            )
            if is_success:
                success_count += 1
            else:
                failed_tests.append(f"{name} (mismatch)")

    print_test_summary(success_count, failed_tests)


# ==============================================================================
#                      MAIN TEST HARNESS
# ==============================================================================
if __name__ == '__main__':
    problem = os.environ.get("COMP_FENICS_PROBLEM", "fluid").strip().lower()
    if problem.startswith("contact") or problem.startswith("semismooth"):
        run_contact_comparison()
        sys.exit(0)
    if problem.startswith("alpha_ch") or problem.startswith("cahn_hilliard") or problem.startswith("cahn-hilliard"):
        run_alpha_ch_comparison()
        sys.exit(0)
    if problem.startswith("seboldt_hdiv") or problem.startswith("benchmark7_hdiv") or problem.startswith("b7_hdiv"):
        run_benchmark7_hdiv_comparison()
        sys.exit(0)
    if problem.startswith("biofilm"):
        run_biofilm_comparison()
        sys.exit(0)
    if problem.startswith("poro"):
        run_poro_comparison()
        sys.exit(0)
    if (
        problem.startswith("scalar_grad")
        or problem.startswith("scalar-grad")
        or problem.startswith("component_semantics")
        or problem.startswith("component-semantics")
    ):
        run_component_semantics_comparison()
        sys.exit(0)

    pc, dof_handler_pc, fenicsx = setup_problems()
    P_map = create_true_dof_map(dof_handler_pc, fenicsx['W'])
    initialize_functions(pc, fenicsx, dof_handler_pc, P_map)

    W_fx = fenicsx['W']
    u_k_fx, p_k_fx = ufl.split(fenicsx['u_k_p_k']) 
    u_n_fx = fenicsx['u_n']
    
    V_subspace = W_fx.sub(0)
    Q_subspace = W_fx.sub(1)
    w_trial = ufl.TrialFunction(W_fx)
    w_test = ufl.TestFunction(W_fx)
    du, dp = ufl.split(w_trial)
    v_fx, q_fx = ufl.split(w_test)
    v = v_fx
    q = q_fx

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

    # --- Basic ALE geometry for fluid forms (mirrors build_forms) ---
    Finv_pc = inv(Fk_pc)
    Finv_fx = ufl.inv(Fk_fx)
    J_geo_pc = det(Fk_pc)
    J_geo_fx = ufl.det(Fk_fx)
    grad_uk_phys_pc = dot(grad(pc['u_k']), Finv_pc)
    grad_v_phys_pc = dot(grad(pc['v']), Finv_pc)
    grad_uk_phys_fx = ufl.dot(ufl.grad(u_k_fx), Finv_fx)
    grad_v_phys_fx = ufl.dot(ufl.grad(v_fx), Finv_fx)
    grad_v_phys_vel = ufl.dot(ufl.grad(v), Finv_fx)
    eps_uk_pc = 0.5 * (grad_uk_phys_pc + grad_uk_phys_pc.T)
    eps_v_pc = 0.5 * (grad_v_phys_pc + grad_v_phys_pc.T)
    eps_uk_fx = 0.5 * (grad_uk_phys_fx + grad_uk_phys_fx.T)
    eps_v_fx = 0.5 * (grad_v_phys_fx + grad_v_phys_fx.T)
    eps_v_vel = 0.5 * (grad_v_phys_vel + grad_v_phys_vel.T)
    div_uk_pc = trace(grad_uk_phys_pc)
    div_v_pc = trace(grad_v_phys_pc)
    div_uk_fx = ufl.tr(grad_uk_phys_fx)
    div_v_fx = ufl.tr(grad_v_phys_fx)
    div_v_vel = ufl.tr(grad_v_phys_vel)
    dF_pc = grad(pc['du'])
    dF_fx = ufl.grad(du)
    dFinv_pc = -dot(Finv_pc, dot(dF_pc, Finv_pc))
    dFinv_fx = -ufl.dot(Finv_fx, ufl.dot(dF_fx, Finv_fx))
    dJ_pc = J_geo_pc * trace(dot(Finv_pc, dF_pc))
    dJ_fx = J_geo_fx * ufl.tr(ufl.dot(Finv_fx, dF_fx))
    grad_uk_shape_pc = dot(grad(pc['u_k']), dFinv_pc)
    grad_v_shape_pc = dot(grad(pc['v']), dFinv_pc)
    grad_uk_shape_fx = ufl.dot(ufl.grad(u_k_fx), dFinv_fx)
    grad_v_shape_fx = ufl.dot(ufl.grad(v_fx), dFinv_fx)
    eps_uk_shape_pc = 0.5 * (grad_uk_shape_pc + grad_uk_shape_pc.T)
    eps_v_shape_pc = 0.5 * (grad_v_shape_pc + grad_v_shape_pc.T)
    eps_uk_shape_fx = 0.5 * (grad_uk_shape_fx + grad_uk_shape_fx.T)
    eps_v_shape_fx = 0.5 * (grad_v_shape_fx + grad_v_shape_fx.T)
    two_mu_f_pc = Constant(2.0, dim=0) * pc['mu']
    two_mu_f_fx = dolfinx.fem.Constant(fenicsx['mesh'], 2.0) * fenicsx['mu']
    sigma_uk_pc = two_mu_f_pc * eps_uk_pc
    sigma_uk_fx = two_mu_f_fx * eps_uk_fx
    term_b_dJ_pc = trace(dot(sigma_uk_pc, eps_v_pc))
    term_b_dJ_fx = ufl.tr(ufl.dot(sigma_uk_fx, eps_v_fx))
    grad_du_phys_pc = dot(grad(pc['du']), Finv_pc)
    grad_du_phys_fx = ufl.dot(ufl.grad(du), Finv_fx)
    eps_du_pc = 0.5 * (grad_du_phys_pc + grad_du_phys_pc.T)
    eps_du_fx = 0.5 * (grad_du_phys_fx + grad_du_phys_fx.T)
    div_du_pc = trace(grad_du_phys_pc)
    div_du_fx = ufl.tr(grad_du_phys_fx)
    stab_eps_pc = Constant(1e-8, dim=0)
    stab_eps_fx = dolfinx.fem.Constant(fenicsx['mesh'], 1e-8)

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
        "RHS Identiy [I2:grad(v)]": {
            'pc': inner(I2_pc, grad(pc['v'])) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(I2_fx, ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 6,
        },
        "RHS Identiy [I2*pk:grad(v)]": {
            'pc': inner(I2_pc * pc['p_k'], grad(pc['v'])) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(I2_fx * p_k_fx, ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 6,
        },
        "RHS Identiy [pk*I2:grad(v)]": {
            'pc': inner(pc['p_k'] *I2_pc , grad(pc['v'])) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(p_k_fx *I2_fx , ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 6,
        },
        "LHS Identiy [p_trial*I2:grad(v)]": {
            'pc': inner(pc['dp'] *I2_pc , grad(pc['v'])) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(dp *I2_fx , ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "LHS Identiy [I2 * p_trial:grad(v)]": {
            'pc': inner(I2_pc * pc['dp'], grad(pc['v'])) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(I2_fx * dp, ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Mixed Basic [Fk]": {
            'pc': inner(pc_geo["Fk"], grad(pc['v'])) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: ufl.inner(fe_geo["Fk"], ufl.grad(v_fx)) \
                                    * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 6,
        },
        # ALE fluid shape terms (from build_forms)
        "Fluid mass_jac_d": {
            'pc': (pc['rho'] / pc['dt']) * pc['theta'] * dJ_pc * dot(pc['u_k'] - pc['u_n'], pc['v']) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: (fenicsx['rho'] / fenicsx['dt']) * fenicsx['theta'] * dJ_fx
                                    * ufl.dot(u_k_fx - u_n_fx, v_fx) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Fluid mass_res": {
            'pc': (pc['rho'] / pc['dt']) * (pc['theta'] * J_geo_pc + (Constant(1.0, dim=0) - pc['theta']) * J_geo_pc)
                   * inner(pc['u_k'] - pc['u_n'], pc['v']) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: (fenicsx['rho'] / fenicsx['dt']) * (fenicsx['theta'] * J_geo_fx + (dolfinx.fem.Constant(fenicsx['mesh'], 1.0) - fenicsx['theta']) * J_geo_fx)
                   * ufl.inner(u_k_fx - u_n_fx, v) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 6,
            # 'rtol': 1e-6,
            # 'atol': 1e-6,
        },
        "Fluid visc_res": {
            'pc': (pc['theta'] * J_geo_pc * inner(sigma_uk_pc, eps_v_pc) + (Constant(1.0, dim=0) - pc['theta']) * J_geo_pc * inner(two_mu_f_pc * eps_uk_pc, eps_v_pc)) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: (fenicsx['theta'] * J_geo_fx * ufl.inner(sigma_uk_fx, eps_v_vel) + (dolfinx.fem.Constant(fenicsx['mesh'], 1.0) - fenicsx['theta']) * J_geo_fx * ufl.inner(two_mu_f_fx * eps_uk_fx, eps_v_vel)) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 8,
            # 'rtol': 1e-6,
            # 'atol': 1e-6,
        },
        "Fluid adv_res": {
            'pc': pc['rho'] * ( pc['theta'] * J_geo_pc * dot(dot(grad_uk_phys_pc, pc['u_k']), pc['v'])
                               + (Constant(1.0, dim=0) - pc['theta']) * J_geo_pc * dot(dot(grad_uk_phys_pc, pc['u_k']), pc['v']) ) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: fenicsx['rho'] * ( fenicsx['theta'] * J_geo_fx * ufl.dot(ufl.dot(grad_uk_phys_fx, u_k_fx), v)
                               + (dolfinx.fem.Constant(fenicsx['mesh'], 1.0) - fenicsx['theta']) * J_geo_fx * ufl.dot(ufl.dot(grad_uk_phys_fx, u_k_fx), v) ) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 6,
            # 'rtol': 1e-6,
            # 'atol': 1e-6,
        },
        "Fluid pres_res 1": {
            'pc': (-J_geo_pc * (pc['p_k'] * div_v_pc ) ) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: (-J_geo_fx * (p_k_fx * div_v_fx ) ) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 6,
            # 'rtol': 1e-6,
            # 'atol': 1e-6,
        },
        "Fluid pres_res": {
            'pc': (-J_geo_pc * (pc['p_k'] * div_v_pc + pc['q'] * div_uk_pc) + stab_eps_pc * pc['p_k'] * pc['q']) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: (-J_geo_fx * (p_k_fx * div_v_fx + q * div_uk_fx) + stab_eps_fx * p_k_fx * q) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 6,
            # 'rtol': 1e-6,
            # 'atol': 1e-6,
        },
        "Fluid visc_jac_d": {
            'pc': pc['theta'] * (
                dJ_pc * term_b_dJ_pc
                + J_geo_pc * two_mu_f_pc * inner(eps_uk_shape_pc, eps_v_pc)
                + J_geo_pc * inner(sigma_uk_pc, eps_v_shape_pc)
            ) * dx(metadata={"q": 4}),
            'f_lambda': lambda deg: fenicsx['theta'] * (
                dJ_fx * term_b_dJ_fx
                + J_geo_fx * two_mu_f_fx * ufl.inner(eps_uk_shape_fx, eps_v_fx)
                + J_geo_fx * ufl.inner(sigma_uk_fx, eps_v_shape_fx)
            ) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 10,
            # 'rtol': 1e-6,
            # 'atol': 2e-5,
        },
        "Fluid mass_jac_u": {
            'pc': (pc['rho'] / pc['dt']) * (pc['theta'] * J_geo_pc + (Constant(1.0, dim=0) - pc['theta']) * J_geo_pc) * inner(pc['du'], pc['v']) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: (fenicsx['rho'] / fenicsx['dt']) * (fenicsx['theta'] * J_geo_fx + (dolfinx.fem.Constant(fenicsx['mesh'], 1.0) - fenicsx['theta']) * J_geo_fx) * ufl.inner(du, v_fx) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
            # 'rtol': 1e-6,
            # 'atol': 1e-6,
        },
        "Fluid visc_jac_u": {
            'pc': pc['theta'] * J_geo_pc * inner(two_mu_f_pc * eps_du_pc, eps_v_pc) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: fenicsx['theta'] * J_geo_fx * ufl.inner(two_mu_f_fx * eps_du_fx, eps_v_vel) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 10,
            # 'rtol': 1e-5,
            # 'atol': 1e-5,
        },
        "Fluid conv_jac_u 1": {
            'pc': pc['rho'] * pc['theta'] * J_geo_pc * ( dot(dot(grad_du_phys_pc, pc['u_k']), pc['v'])  ) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: fenicsx['rho'] * fenicsx['theta'] * J_geo_fx * ( ufl.dot(ufl.dot(grad_du_phys_fx, u_k_fx), v)  ) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
            # 'rtol': 1e-6,
            # 'atol': 1e-6,
        },
        "Fluid conv_jac_u 2": {
            'pc': pc['rho'] * pc['theta'] * J_geo_pc * (  dot(dot(grad_uk_phys_pc, pc['du']), pc['v']) ) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: fenicsx['rho'] * fenicsx['theta'] * J_geo_fx * (  ufl.dot(ufl.dot(grad_uk_phys_fx, du), v) ) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
            # 'rtol': 1e-6,
            # 'atol': 1e-6,
        },
        "Fluid conv_jac_u": {
            'pc': pc['rho'] * pc['theta'] * J_geo_pc * ( dot(dot(grad_du_phys_pc, pc['u_k']), pc['v']) + dot(dot(grad_uk_phys_pc, pc['du']), pc['v']) ) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: fenicsx['rho'] * fenicsx['theta'] * J_geo_fx * ( ufl.dot(ufl.dot(grad_du_phys_fx, u_k_fx), v) + ufl.dot(ufl.dot(grad_uk_phys_fx, du), v) ) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
            # 'rtol': 1e-6,
            # 'atol': 1e-6,
        },
        # Skew-symmetric convection (energy-conserving): 1/2 * ((u·∇u, v) - (u·∇v, u))
        "Fluid conv_skew_res (u_k)": {
            'pc': pc['rho'] * pc['theta'] * J_geo_pc * Constant(0.5, dim=0) * (
                dot(dot(grad_uk_phys_pc, pc['u_k']), pc['v'])
                - dot(dot(grad_v_phys_pc, pc['u_k']), pc['u_k'])
            ) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: fenicsx['rho'] * fenicsx['theta'] * J_geo_fx * dolfinx.fem.Constant(fenicsx['mesh'], 0.5) * (
                ufl.dot(ufl.dot(grad_uk_phys_fx, u_k_fx), v)
                - ufl.dot(ufl.dot(grad_v_phys_vel, u_k_fx), u_k_fx)
            ) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 6,
        },
        "Fluid conv_skew_jac_u": {
            'pc': pc['rho'] * pc['theta'] * J_geo_pc * Constant(0.5, dim=0) * (
                dot(dot(grad_uk_phys_pc, pc['du']), pc['v'])
                + dot(dot(grad_du_phys_pc, pc['u_k']), pc['v'])
                - dot(dot(grad_v_phys_pc, pc['du']), pc['u_k'])
                - dot(dot(grad_v_phys_pc, pc['u_k']), pc['du'])
            ) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: fenicsx['rho'] * fenicsx['theta'] * J_geo_fx * dolfinx.fem.Constant(fenicsx['mesh'], 0.5) * (
                ufl.dot(ufl.dot(grad_uk_phys_fx, du), v)
                + ufl.dot(ufl.dot(grad_du_phys_fx, u_k_fx), v)
                - ufl.dot(ufl.dot(grad_v_phys_fx, du), u_k_fx)
                - ufl.dot(ufl.dot(grad_v_phys_fx, u_k_fx), du)
            ) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
        },
        "Fluid pres_jac": {
            'pc': (-J_geo_pc * pc['q'] * div_du_pc - J_geo_pc * pc['dp'] * div_v_pc + stab_eps_pc * pc['dp'] * pc['q']) * dx(metadata={"q": 6}),
            'f_lambda': lambda deg: (-J_geo_fx * q * div_du_fx - J_geo_fx * dp * div_v_vel + stab_eps_fx * dp * q) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 6,
            # 'rtol': 1e-6,
            # 'atol': 1e-6,
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
        # looked until here
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
            'pc': pc['theta'] * inner(d2sigma_pc_expr, I2_pc) * dx(metadata={"q":6}),
            'f_lambda': lambda deg: fenicsx['theta'] * ufl.inner(d2sigma_fx_expr, I2_fx) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 14,
        },
        "Solid Cross Tangent [A: invJ*dT]": _solid_cross_component_entry("A: invJ*dT", deg_fe=14, deg_pc=8),
        "Solid Cross Tangent [B: -(tr(F^{-1} dF) over J)*T_w]": _solid_cross_component_entry("B: -(tr(F^{-1} dF) over J)*T_w", deg_fe=14, deg_pc=6),
        "Solid Cross Tangent [C: tr(F^{-1} dF F^{-1} Aw)*sigma]": _solid_cross_component_entry("C: tr(F^{-1} dF F^{-1} Aw)*sigma", deg_fe=14, deg_pc=6),
        "Solid Cross Tangent [D: -tr(F^{-1} Aw)*ds_u]": _solid_cross_component_entry("D: -tr(F^{-1} Aw)*ds_u", deg_fe=14, deg_pc=6),

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
        "LHS Mass 2":          {'pc': pc['rho'] * dot(pc['v'], pc['du']) / pc['dt'] * dx(),                                    'f_lambda': lambda deg: fenicsx['rho'] * ufl.dot(v, du) / fenicsx['dt'] * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
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
        "LHS VecGrad Left (grad(du)*u_k)·v": {
            'pc': dot(dot(grad(pc['du']), pc['u_k']), pc['v']) * dx(metadata={"q":4}),
            'f_lambda': lambda deg: ufl.dot(ufl.dot(ufl.grad(du), u_k_fx), v) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 4,
        },
        "LHS VecGrad Right (u_k*grad(du))·v": {
            'pc': dot(dot(pc['u_k'], grad(pc['du'])), pc['v']) * dx(metadata={"q":4}),
            'f_lambda': lambda deg: ufl.dot(ufl.dot(u_k_fx, ufl.grad(du)), v) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': True,
            'deg': 4,
        },
        "RHS VecGrad Left (grad(u_k)*v)·c": {
            'pc': dot(dot(grad(pc['u_k']), pc['v']), c_pc) * dx(metadata={"q":4}),
            'f_lambda': lambda deg: ufl.dot(ufl.dot(ufl.grad(u_k_fx), v), c_fx) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 4,
        },
        "RHS VecGrad Right (v*grad(u_k))·c": {
            'pc': dot(dot(pc['v'], grad(pc['u_k'])), c_pc) * dx(metadata={"q":4}),
            'f_lambda': lambda deg: ufl.dot(ufl.dot(v, ufl.grad(u_k_fx)), c_fx) * ufl.dx(metadata={'quadrature_degree': deg}),
            'mat': False,
            'deg': 4,
        },
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
    _ = pc_dummy_side

    run_all = os.environ.get("COMP_FENICS_RUN_ALL", "").lower() in {"1", "true", "yes"}
    filter_terms = os.environ.get("COMP_FENICS_TERMS")
    if filter_terms:
        allowed = {name.strip() for name in filter_terms.split(",") if name.strip()}
        terms = {k: v for k, v in terms.items() if k in allowed}
        print(f"Running filtered terms only: {sorted(terms)}")
    elif not run_all:
        if problem.startswith("fracture"):
            terms = {k: v for k, v in terms.items() if k.startswith("Solid ")}
            print(f"COMP_FENICS_TERMS not set; running fracture-focused subset: {sorted(terms)}")
        else:
            default = {
                "LHS Mass",
                "LHS Diffusion",
                "LHS Pressure",
                "RHS Time Derivative",
                "RHS pressure term",
                "RHS Identiy [I2:grad(v)]",
            }
            terms = {k: v for k, v in terms.items() if k in default}
            print(f"COMP_FENICS_TERMS not set; running safe default subset: {sorted(terms)}")

    # Pre-assemble FEniCSx reference outputs once (backend-independent).
    fenics_ref = {}
    for name, forms in terms.items():
        try:
            form_fx_ufl = forms["f_lambda"](forms["deg"])
            form_fx_compiled = dolfinx.fem.form(form_fx_ufl)
            if forms["mat"]:
                A = dolfinx.fem.petsc.assemble_matrix(form_fx_compiled)
                A.assemble()
                indptr, indices, data = A.getValuesCSR()
                fenics_ref[name] = csr_matrix((data, indices, indptr), shape=A.getSize()).toarray()
            else:
                vec = dolfinx.fem.petsc.assemble_vector(form_fx_compiled)
                fenics_ref[name] = vec.array.copy()
        except Exception as exc:
            print(f"❌ FEniCSx assembly failed for '{name}': {exc}")
            fenics_ref[name] = None

    backends_spec = os.environ.get("BACKEND", "jit")
    if backends_spec.strip().lower() == "all":
        backends = ["python", "jit", "cpp"]
    else:
        backends = [b.strip() for b in backends_spec.split(",") if b.strip()]
    parity_rtol = float(os.environ.get("COMP_FENICS_PARITY_RTOL", "1e-9"))
    parity_atol = float(os.environ.get("COMP_FENICS_PARITY_ATOL", "1e-9"))

    ref_backend = "python" if "python" in backends else (backends[0] if backends else "python")
    ref_pc = {}

    for backend_type in backends:
        print("\n" + "=" * 70)
        print(f"COMPARISON (problem={problem}, backend={backend_type})")
        print("=" * 70)

        failed_tests = []
        success_count = 0

        for name, forms in terms.items():
            if fenics_ref.get(name) is None:
                failed_tests.append(f"{name} (fenics-assemble)")
                continue

            J_pc, R_pc = None, None
            print(f"Compiling/assembling '{name}' with degree {forms['deg']} [backend={backend_type}]")
            try:
                if forms["mat"]:
                    J_pc, _ = assemble_form(
                        Equation(forms["pc"], None),
                        dof_handler_pc,
                        quad_degree=forms["deg"],
                        bcs=[],
                        backend=backend_type,
                    )
                else:
                    _, R_pc = assemble_form(
                        Equation(None, forms["pc"]),
                        dof_handler_pc,
                        bcs=[],
                        backend=backend_type,
                    )
            except Exception as exc:
                print(f"❌ pycutfem assembly failed for '{name}' on backend '{backend_type}': {exc}")
                failed_tests.append(f"{name} (assemble-{backend_type})")
                continue

            if backend_type == ref_backend:
                if forms["mat"]:
                    ref_pc[name] = J_pc.toarray()
                else:
                    ref_pc[name] = np.asarray(R_pc, dtype=float).copy()
            else:
                try:
                    if forms["mat"]:
                        np.testing.assert_allclose(J_pc.toarray(), ref_pc[name], rtol=parity_rtol, atol=parity_atol)
                    else:
                        np.testing.assert_allclose(np.asarray(R_pc, dtype=float), ref_pc[name], rtol=parity_rtol, atol=parity_atol)
                    print(f"✅ pycutfem backend parity OK vs '{ref_backend}'.")
                except Exception as exc:
                    print(f"❌ pycutfem backend parity FAILED vs '{ref_backend}': {exc}")
                    failed_tests.append(f"{name} (parity-{backend_type}-vs-{ref_backend})")

            J_fx, R_fx = None, None
            if forms["mat"]:
                J_fx = fenics_ref[name]
            else:
                R_fx = fenics_ref[name]

            rtol = forms.get("rtol", 1e-8)
            atol = forms.get("atol", 1e-8)
            is_success = compare_term(
                f"{name} [backend={backend_type}]",
                J_pc,
                R_pc,
                J_fx,
                R_fx,
                P_map,
                dof_handler_pc,
                W_fx,
                rtol=rtol,
                atol=atol,
            )
            if is_success:
                success_count += 1
            else:
                failed_tests.append(name)

        print_test_summary(success_count, failed_tests)
