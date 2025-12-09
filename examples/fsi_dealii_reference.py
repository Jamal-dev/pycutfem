#!/usr/bin/env python
"""
Deal.II FSI benchmark (FSI-1) reproduced in pycutfem.

- Geometry and mesh: use the reference `fsi.inp` UCD mesh from the deal.II
  code (channel 2.5 x 0.41 with a rigid cylinder and an attached beam).
- Formulation: conforming ALE Navier–Stokes + compressible Neo-Hookean solid,
  monolithic theta-scheme as in `fsi_ale_conforming.py`.
- Parameters: identical to `step-fsi.prm` (rho_f=1000, mu_f=1e-3, rho_s=1000,
  mu_s=0.5e6, nu_s=0.4, alpha_u=1e-8, dt=1.0, theta=1.0, 25 steps).

The goal is to mirror the established deal.II example and verify that the
pycutfem formulation converges on the same data set.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pycutfem.core.topology import Node
from pycutfem.core.mesh import Mesh
from pycutfem.utils.bitset import BitSet
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    Identity,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    dot,
    div,
    grad,
    inner,
    det,
    inv,
    trace,
    cof, # cofactor
    FacetNormal,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dx, dS
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters

# ----------------------------------------------------------------------------- 
# Geometry helpers (Turek–Hron benchmark)
# -----------------------------------------------------------------------------
H = 0.41
L = 2.5
RADIUS = 0.05
CENTER = (0.2, 0.2)
BEAM_LENGTH = 0.35
BEAM_HEIGHT = 0.02
BEAM_X0 = CENTER[0] + RADIUS
BEAM_Y0 = CENTER[1] - 0.5 * BEAM_HEIGHT
BEAM_Y1 = CENTER[1] + 0.5 * BEAM_HEIGHT


def load_ucd_mesh(path: Path, poly_order: int = 1) -> Tuple[Mesh, BitSet, BitSet]:
    """
    Minimal UCD reader for the deal.II `fsi.inp` mesh.
    Returns the Mesh plus BitSets for fluid and solid elements.
    """
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().split()
        if len(header) < 2:
            raise RuntimeError(f"Unexpected UCD header in {path}")
        n_nodes, n_cells = int(header[0]), int(header[1])

        nodes: List[Node] = []
        for _ in range(n_nodes):
            nid, xs, ys, *_ = f.readline().split()
            nodes.append(Node(int(nid), float(xs), float(ys)))

        coords = np.array([[n.x, n.y] for n in nodes], dtype=float)

        elem_conn: List[List[int]] = []
        corner_conn: List[List[int]] = []
        elem_tags: List[str] = []
        boundary_segments: List[Tuple[int, int, int]] = []
        for _ in range(n_cells):
            parts = f.readline().split()
            if len(parts) < 3:
                continue
            _, mat_id, cell_type, *conn = parts
            ctype = cell_type.lower()
            if ctype == "quad":
                conn_int = list(map(int, conn))
                corn = conn_int[:4]
                pts = coords[corn]
                xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
                ymin, ymax = pts[:, 1].min(), pts[:, 1].max()

                def _closest(target: Tuple[float, float]) -> int:
                    dist = np.sum((pts - target) ** 2, axis=1)
                    return int(np.argmin(dist))

                bl = _closest((xmin, ymin))
                br = _closest((xmax, ymin))
                tl = _closest((xmin, ymax))
                tr = _closest((xmax, ymax))
                # Ordering (BL, BR, TL, TR) keeps the bilinear map positive on this mesh
                ordered = [corn[i] for i in (bl, br, tl, tr)]
                elem_conn.append(ordered)
                corner_conn.append(ordered)
                elem_tags.append("solid" if int(mat_id) == 1 else "fluid")
            elif ctype == "line" and len(conn) >= 2:
                n1, n2 = map(int, conn[:2])
                boundary_segments.append((int(mat_id), n1, n2))
            else:
                continue

    # Remaining boundary segments are ignored here; we retag geometrically.
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(elem_conn, dtype=int),
        elements_corner_nodes=np.asarray(corner_conn, dtype=int),
        element_type="quad",
        poly_order=poly_order,
    )

    # Cache element bitsets
    fluid_mask = np.fromiter((tag == "fluid" for tag in elem_tags), bool)
    solid_mask = ~fluid_mask
    for el, tag in zip(mesh.elements_list, elem_tags):
        el.tag = tag
    mesh._elem_bitsets = {
        "fluid": BitSet(fluid_mask),
        "solid": BitSet(solid_mask),
    }
    return mesh, mesh.element_bitset("fluid"), mesh.element_bitset("solid")


def retag_boundaries(mesh: Mesh, tol: float = 1.0e-8) -> None:
    """
    Geometric tagging of boundary edges to mirror the deal.II boundary ids:
    - inlet  (x=0)
    - outlet (x=L)
    - walls  (y=0 or y=H)
    - cylinder (circle at CENTER, radius RADIUS)
    - beam_outer (outer beam box except the circular interface)
    """
    cx, cy = CENTER
    r2 = RADIUS * RADIUS
    beam_x1 = BEAM_X0 + BEAM_LENGTH

    def on_cylinder(x: float, y: float) -> bool:
        return abs((x - cx) ** 2 + (y - cy) ** 2 - r2) < 1e-6

    def on_beam_outer(x: float, y: float) -> bool:
        on_x0 = abs(x - BEAM_X0) < tol and BEAM_Y0 - tol <= y <= BEAM_Y1 + tol
        on_x1 = abs(x - beam_x1) < tol and BEAM_Y0 - tol <= y <= BEAM_Y1 + tol
        on_y0 = abs(y - BEAM_Y0) < tol and BEAM_X0 - tol <= x <= beam_x1 + tol
        on_y1 = abs(y - BEAM_Y1) < tol and BEAM_X0 - tol <= x <= beam_x1 + tol
        return on_x0 or on_x1 or on_y0 or on_y1

    mesh.tag_boundary_edges(
        {
            "inlet": lambda x, y: abs(x - 0.0) < tol,
            "outlet": lambda x, y: abs(x - L) < tol,
            "walls": lambda x, y: abs(y - 0.0) < tol or abs(y - H) < tol,
            "cylinder": on_cylinder,
            "beam_outer": on_beam_outer,
        }
    )


def symgrad(u):
    return 0.5 * (grad(u) + grad(u).T)
def transpose(A):
    return A.T  
def _is_zero(expr) -> bool:
    """
    Cheap zero check for scalars/vectors/matrices represented as Constant or numbers.
    Avoids triggering tensor algebra on obvious zeros when linearizing terms.
    """
    if isinstance(expr, Constant):
        arr = np.asarray(expr.value)
        return np.allclose(arr, 0.0)
    if isinstance(expr, (int, float, np.floating)):
        return abs(expr) < 1.0e-14
    return False
class ALE_Helpers:
    """
    ALE kinematic helpers, mirroring the C++ ALE_Transformations namespace.
    All tensor products use `dot` (last index of first tensor with
    first index of second tensor). Cofactors, det, inv as in pycutfem UFL.
    """
    @staticmethod
    def get_F(grad_d):
        return Identity(2) + grad_d

    @staticmethod
    def get_J(F):
        return det(F)

    @staticmethod
    def get_F_inv(F):
        return inv(F)

    @staticmethod
    def get_cof_F(F):
        return cof(F)  # J * F^{-T}

    @staticmethod
    def get_J_LinU(F, grad_dd):
        r"""
        Linearization of J with respect to displacement d.

        δJ = cof(F) : ∇δd = inner(cof(F), grad(δd)).
        """
        cof_F = cof(F)
        return inner(cof_F, grad_dd)


    @staticmethod
    def get_F_inv_LinU(F_inv, grad_dd):
        r"""
        Linearization of F^{-1} with respect to d.

        δ(F^{-1}) = - F^{-1} (δF) F^{-1},  δF = ∇δd.
        """
        return -dot(F_inv, dot(grad_dd, F_inv))

    @staticmethod
    def get_cof_F_LinU(F, F_inv, grad_dd):
        r"""
        Linearization of cof(F) = J F^{-T} with respect to d.

        δ(cof(F)) = δ(J F^{-T}) = δJ F^{-T} + J δ(F^{-T}),
        δJ  = cof(F) : ∇δd,
        δF^{-T} = (δF^{-1})^T = ( -F^{-1} δF F^{-1} )^T.
        """
        J = det(F)
        cof_F = cof(F)

        J_LinU = inner(cof_F, grad_dd)
        F_inv_LinU = -dot(F_inv, dot(grad_dd, F_inv))
        dF_inv_T = F_inv_LinU.T

        return J_LinU * F_inv.T + J * dF_inv_T
class NSE_ALE:
    """
    Fluid NSE terms in ALE formulation.

    All tensor products use `dot` (matrix multiplication / tensor contraction),
    and `inner` is the Frobenius product (double contraction).
    """
    @staticmethod
    def get_stress_fluid_ALE(mu_f, p, grad_v, F_inv):
        r"""
        Cauchy stress in ALE:

        σ = -p I + μ (∇v F^{-1} + F^{-T} ∇v^T).
        """
        I = Identity(2)
        grad_v_T = grad_v.T
        F_inv_T = F_inv.T

        sigma_visc = mu_f * (dot(grad_v, F_inv) + dot(F_inv_T, grad_v_T))
        return -p * I + sigma_visc
    @staticmethod
    def get_stress_fluid_ALE_direct(mu_f, pI, grad_v, F_inv, grad_v_T, F_inv_T):
        r"""
        Cauchy stress in ALE:

        σ = -p I + μ (∇v F^{-1} + F^{-T} ∇v^T).
        """
        sigma_visc = mu_f * (dot(grad_v, F_inv) + dot(F_inv_T, grad_v_T))
        return -pI + sigma_visc

    @staticmethod
    def get_stress_fluid_except_pressure_ALE(mu_f, grad_v, F_inv):
        grad_v_T = grad_v.T
        F_inv_T = F_inv.T
        return  mu_f * (dot(grad_v, F_inv) + dot(F_inv_T, grad_v_T))

    # ------------------------------------------------------------------ #
    # Linearization of stress                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_stress_fluid_ALE_1st_term_LinAll_short(
        pI,
        F_inv_T,
        J_F_inv_T_LinU_trial,
        pI_LinP,
        J):
        return (-J * dot(pI_LinP , F_inv_T) - dot(pI, J_F_inv_T_LinU_trial)) 
    
    @staticmethod
    def get_stress_fluid_ALE_2nd_term_LinAll_short(
        J_F_inv_T_LinU,
        stress_fluid_ALE,
        grad_v,
        grad_v_LinV,
        F_inv,
        F_inv_LinU,
        J,
        mu_f,
    ):
        r"""
        C++: get_stress_fluid_ALE_2nd_term_LinAll_short

        Returns (without density factor):

        μ [ J (σ_LinV + σ_LinU) F^{-T} + σ J_F^{-T}_LinU ],

        where

        σ_LinV = ∇(δv) F^{-1} + F^{-T} ∇(δv)^T,
        σ_LinU = ∇v δF^{-1} + (δF^{-1})^T ∇v^T.
        """
        F_inv_T = F_inv.T

        sigma_terms = []

        # σ_LinV
        if not _is_zero(grad_v_LinV):
            sigma_terms.append(
                dot(grad_v_LinV, F_inv) + dot(F_inv_T, grad_v_LinV.T)
            )

        # σ_LinU
        if not _is_zero(F_inv_LinU):
            sigma_terms.append(
                dot(grad_v, F_inv_LinU) + dot(F_inv_LinU.T, grad_v.T)
            )

        pieces = []

        if sigma_terms:
            sigma_sum = sigma_terms[0]
            for term in sigma_terms[1:]:
                sigma_sum = sigma_sum + term
            # J (σ_LinV + σ_LinU) F^{-T}
            pieces.append(J * dot(sigma_sum, F_inv_T))

        # σ J_F^{-T}_LinU
        if not _is_zero(J_F_inv_T_LinU):
            pieces.append(dot(stress_fluid_ALE, J_F_inv_T_LinU))

        if not pieces:
            return Constant(0.0)

        total = pieces[0]
        for term in pieces[1:]:
            total = total + term

        return mu_f * total

    @staticmethod
    def get_stress_fluid_ALE_3rd_term_LinAll_short(
        F_inv,
        F_inv_LinU_trial,
        grad_v,
        grad_v_LinV_trial,
        mu_f,
        J,
        J_F_inv_T_LinU_trial,
    ):
        r"""
        C++: get_stress_fluid_ALE_3rd_term_LinAll_short

        Returns (without density factor):

        μ [ J_F^{-T}_LinU ∇v^T F^{-T}
          + J F^{-T} ∇(δv)^T F^{-T}
          + J F^{-T} ∇v^T (δF^{-1})^T ].
        """
        F_inv_T = F_inv.T

        term1 = dot(dot(J_F_inv_T_LinU_trial, grad_v.T), F_inv_T)
        term2 = J * dot(dot(F_inv_T, grad_v_LinV_trial.T), F_inv_T)
        term3 = J * dot(dot(F_inv_T, grad_v.T), F_inv_LinU_trial.T)

        return mu_f * (term1 + term2 + term3)
    # ------------------------------------------------------------------ #
    # Incompressibility                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_Incompressibility_ALE(v, F):
        r"""
        g = J F^{-1} : ∇v = cof(F) : ∇v.

        This compact expression is algebraically equivalent to the explicit
        polynomial used in the C++ code.
        """
        return inner(cof(F), grad(v))
    @staticmethod
    def get_Incompressibility_ALE_LinV_optimized(grad_v, grad_v_trial, F, grad_dd_trial):
        r"""
        2D Optimized Linearization of g = cof(F) : ∇v
        
        This bypasses calculating determinants and inverses for the linearization
        of the geometry term, valid ONLY for 2D.
        """
        cof_F = cof(F)
        term_v = inner(cof_F, grad_v_trial)
        delta_cof_F = cof(grad_dd_trial)
        term_geom = inner(delta_cof_F, grad_v)
        return term_v + term_geom
    @staticmethod
    def get_Incompressibility_ALE_LinAll(v, v_trial, F, F_inv, grad_dd):
        r"""
        Linearization of g = cof(F) : ∇v w.r.t. both v and d.

        Using cof(F) = J F^{-T}:

        δg = cof(F) : ∇(δv) + δ(cof(F)) : ∇v
            = cof(F) : ∇(δv) + [δJ F^{-T} + J δF^{-T}] : ∇v.

        We implement this in terms of F, F^{-1}, and δF = ∇δd.
        """
        cof_F = cof(F)

        # First part: cof(F) : ∇(δv)
        term_v = inner(cof_F, grad(v_trial))

        # δJ = cof(F) : δF
        delta_J = inner(cof_F, grad_dd)

        # δF^{-1} = -F^{-1} δF F^{-1}
        F_inv_LinU = -dot(F_inv, dot(grad_dd, F_inv))
        delta_F_inv_T = F_inv_LinU.T

        J = det(F)
        F_inv_T = F_inv.T

        # δcof(F) = δ(J F^{-T}) = δJ F^{-T} + J δF^{-T}
        delta_cof_F = delta_J * F_inv_T + J * delta_F_inv_T

        term_geom = inner(delta_cof_F, grad(v))

        return term_v + term_geom
    
    # ------------------------------------------------------------------ #
    # Convection                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_Convection_LinAll_short(
        phi_grad_v_trial,
        phi_v_trial,
        J,
        J_LinU,
        F_inv,
        F_inv_LinU,
        v,
        grad_v,
        density,
    ):
        r"""
        C++: get_Convection_LinAll_short

        For c = ρ J ∇v F^{-1} v we have

        δc = ρ [ δJ ∇v F^{-1} v
                + J ∇v δF^{-1} v
                + J ∇(δv) F^{-1} v
                + J ∇v F^{-1} δv ].
        """
        grad_v_Finv = dot(grad_v, F_inv)

        conv_LinU = None
        if not _is_zero(J_LinU):
            if conv_LinU is None:
                conv_LinU = J_LinU * dot(grad_v_Finv, v)
            else:
                conv_LinU += J_LinU * dot(grad_v_Finv, v)
        if not _is_zero(F_inv_LinU):
            if conv_LinU is None:
                conv_LinU = J * dot(dot(grad_v, F_inv_LinU), v)
            else:
                conv_LinU += J * dot(dot(grad_v, F_inv_LinU), v)

        conv_LinV = None
        if not _is_zero(phi_grad_v_trial):
            if conv_LinV is None:
                conv_LinV = J * dot(dot(phi_grad_v_trial, F_inv), v)
            else:
                conv_LinV += J * dot(dot(phi_grad_v_trial, F_inv), v)
        if not _is_zero(phi_v_trial):
            if conv_LinV is None:
                conv_LinV = J * dot(grad_v_Finv, phi_v_trial)
            else:
                conv_LinV += J * dot(grad_v_Finv, phi_v_trial)

        if conv_LinU is None and conv_LinV is not None:
            return density * conv_LinV
        elif conv_LinV is None and conv_LinU is not None:
            return density * conv_LinU
        elif conv_LinU is not None and conv_LinV is not None:
            return density * (conv_LinU + conv_LinV)
        else:
            return Constant(0.0)
            

    @staticmethod
    def get_Convection_u_LinAll_short(phi_grad_v_trial, phi_u_disp_trial, J, J_LinU,
                                      F_inv, F_inv_LinU, u_disp, grad_v,
                                      density):
        """
        Same as above, but with u instead of v.
        """
        grad_v_Finv = dot(grad_v, F_inv)
        grad_v_Finv_LinU = dot(grad_v, F_inv_LinU)
        conv_LinU = (J_LinU * dot(grad_v_Finv, u_disp)
                        + J * dot(grad_v_Finv_LinU, u_disp)
                        + J * dot(grad_v_Finv, phi_u_disp_trial)
            )
        conv_LinV = J * dot(phi_grad_v_trial, dot(F_inv, u_disp))
        return density * (conv_LinU + conv_LinV)

    @staticmethod
    def get_Convection_u_old_LinAll_short(phi_grad_v_trial, J, J_LinU,
                                          F_inv, F_inv_LinU,
                                          u_old_disp, grad_v, density):
        """
        Same structure, using old timestep quantities.
        """
        grad_v_Finv = dot(grad_v, F_inv)
        grad_v_Finv_LinU = dot(grad_v, F_inv_LinU)
        conv_LinU = (J_LinU * dot(grad_v_Finv, u_old_disp)
                     + J * dot(grad_v_Finv_LinU, u_old_disp)
            )
        
        F_inv_u_old = dot(F_inv, u_old_disp)
        conv_LinV = J * dot(phi_grad_v_trial, F_inv_u_old)
        return density * (conv_LinU + conv_LinV)

    # ------------------------------------------------------------------ #
    # Acceleration                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_acceleration_term_LinAll(J, J_old, J_LinU_trial,
                                     v, v_old, v_trial, density):
        r"""
        C++ idea:

        ρ/2 [ (J + J_old)/dt (v - v_old) ].

        Linearized:

        δa = ρ/2 [ J_LinU (v - v_old)/dt + (J + J_old)/dt δv ].
        """
        term_geom = (J_LinU_trial * (v - v_old)) 
        term_vel  = (J + J_old) * v_trial 
        return 0.5 * density * (term_geom + term_vel)





class Structure_Terms:
    """
    STVK material terms.
    """
    @staticmethod
    def get_C(F):
        return dot(F.T, F)
    @staticmethod
    def get_E(F):
        I = Identity(2)
        C = Structure_Terms.get_C(F)
        return 0.5 * (C - I)

    @staticmethod
    def get_S(E, mu_s, lambda_s):
        r"""
        Second Piola–Kirchhoff stress (STVK):

        S = λ tr(E) I + 2 μ E.
        """
        I = Identity(2)
        trE = trace(E)
        return lambda_s * trE * I + 2.0 * mu_s * E
    @staticmethod
    def get_F_LinU(grad_dd):
        r"""
        Linearization of F w.r.t. displacement d.

        F = I + ∇d, δF = ∇δd.
        """
        return grad_dd
    @staticmethod
    def get_E_LinU(F, grad_dd):
        r"""
        Linearization of E w.r.t. displacement d.

        F = I + ∇d, δF = ∇δd.

        E = 1/2 (F^T F - I),
        δE = 1/2 (δF^T F + F^T δF).
        """
        delta_F = grad_dd
        delta_E = 0.5 * (dot(delta_F.T, F) + dot(F.T, delta_F))
        return delta_E
    @staticmethod
    def get_Piola_Kirchhoff_1st_LinAll(F, grad_dd, mu_s, lambda_s):
        r"""
        Linearization of 1st Piola–Kirchhoff stress P w.r.t. displacement d.

        F = I + ∇d, δF = ∇δd.

        E = 1/2 (F^T F - I),
        δE = 1/2 (δF^T F + F^T δF),

        S = λ tr(E) I + 2 μ E,
        δS = λ tr(δE) I + 2 μ δE,

        P = F S,
        δP = δF S + F δS.
        """
        delta_F = grad_dd
        E = Structure_Terms.get_E(F)
        S = Structure_Terms.get_S(E, mu_s, lambda_s)

        delta_E = 0.5 * (dot(delta_F.T, F) + dot(F.T, delta_F))
        tr_delta_E = trace(delta_E)
        delta_S = lambda_s * tr_delta_E * Identity(2) + 2.0 * mu_s * delta_E

        delta_P = dot(delta_F, S) + dot(F, delta_S)
        return delta_P

    @staticmethod
    def get_S_LinU(F, grad_dd, mu_s, lambda_s):
        r"""
        Linearization of S and E w.r.t. displacement d.

        F = I + ∇d, δF = ∇δd.

        E = 1/2 (F^T F - I),
        δE = 1/2 (δF^T F + F^T δF),

        S = λ tr(E) I + 2 μ E,
        δS = λ tr(δE) I + 2 μ δE.
        """
        delta_F = grad_dd
        E = Structure_Terms.get_E(F)

        delta_E = 0.5 * (dot(delta_F.T, F) + dot(F.T, delta_F))
        tr_delta_E = trace(delta_E)

        I = Identity(2)
        delta_S = lambda_s * tr_delta_E * I + 2.0 * mu_s * delta_E

        return delta_S, delta_E


def build_jac(
    *,
    uk, u_prev,       # Fluid Velocity (current, old)
    dk, d_prev,       # Displacement (current, old)
    pk, p_prev,       # Pressure (current, old)
    du, dd, dp,       # Trial functions (δv, δd, δp)
    test_v, test_w, test_q,          # Test functions (ψ_v, ψ_d, ψ_p)
    timestep: Constant,
    theta_const: Constant,
    rho_f: Constant,
    mu_f: Constant,
    rho_s: Constant,
    lambda_s: Constant,
    mu_s: Constant,
    alpha_u: Constant,
    stab_eps: Constant,
    fluid_bs,
    solid_bs,
    outlet_bs,
    quad_order: int,
):
    # --- Integration Measures ---
    dx_f = dx(defined_on=fluid_bs, metadata={"q": quad_order})
    dx_s = dx(defined_on=solid_bs, metadata={"q": quad_order})
    dS_outlet = dS(defined_on=outlet_bs, metadata={"q": quad_order})
    n =  FacetNormal()
    
    # --- Geometric State (Current Newton Iteration) ---
    I = Identity(2)
    F = ALE_Helpers.get_F(grad(dk))
    J = ALE_Helpers.get_J(F)
    Finv = ALE_Helpers.get_F_inv(F)
    F_inv_T = Finv.T
    cof_F = ALE_Helpers.get_cof_F(F)  # J * Finv.T
    pI = pk * Identity(2)
    pI_LinP_trial = dp * Identity(2)

    # --- Geometric State (Previous Timestep) ---
    F_old = ALE_Helpers.get_F(grad(d_prev))
    J_old = ALE_Helpers.get_J(F_old)
    
    # --- Geometric Linearization (Shape Derivatives) ---
    # These represent the variation of geometric terms w.r.t displacement trial function (dd)
    grad_dd = grad(dd)
    J_F_inv_T_LinU = cof(grad_dd)  # δ(J F^{-T}) # trial of displacement
    J_LinU = ALE_Helpers.get_J_LinU(Finv, grad_dd) # trial of displacement
    Finv_LinU = ALE_Helpers.get_F_inv_LinU(Finv, grad_dd) # trial of displacement
    cof_F_LinU = ALE_Helpers.get_cof_F_LinU(F, Finv, grad_dd) # δ(J F^{-T}) # trial of displacement

    # test gradients
    test_grad_v = grad(test_v)
    test_grad_w = grad(test_w)

    # ========================================================================
    # 1. FLUID RESIDUAL (ALE Navier-Stokes)
    # ========================================================================
    
    # --- Velocity & Gradients ---
    grad_uk = grad(uk)
    grad_uk_T = grad_uk.T
    grad_u_prev = grad(u_prev)

    sigma_ALE = NSE_ALE.get_stress_fluid_ALE_direct(
        mu_f, pI, grad_uk, Finv, grad_uk_T, F_inv_T
    )
    # ---  Mass / Acceleration Term ---
    acc_term_jac = NSE_ALE.get_acceleration_term_LinAll(
        J, J_old, J_LinU,
        uk, u_prev, du,
        rho_f
    )  
    # ---  Convection Term ---
    convection_fluid_v = NSE_ALE.get_Convection_LinAll_short(
        grad(du), du, J, J_LinU,
        Finv, Finv_LinU, uk, grad_uk,
        rho_f
    )  
    convection_fluid_d = NSE_ALE.get_Convection_u_LinAll_short(
        grad(du), dd, J, J_LinU,
        Finv, Finv_LinU, dk, grad_uk,
        rho_f
    )
    convection_fluid_u_old = NSE_ALE.get_Convection_u_old_LinAll_short(
        grad(du), J, J_LinU,
        Finv, Finv_LinU, d_prev, grad_uk,
        rho_f
    )
    # ---  Diffusion Term ---
    stress_fluid_term_1 = NSE_ALE.get_stress_fluid_ALE_1st_term_LinAll_short(
        pI, F_inv_T,
        J_F_inv_T_LinU,
        pI_LinP_trial,
        J
    )
    stress_fluid_term_2 = NSE_ALE.get_stress_fluid_ALE_2nd_term_LinAll_short(
        J_F_inv_T_LinU,
        sigma_ALE,
        grad_uk,
        grad(du),
        Finv, Finv_LinU,
        J,
        mu_f,
    )
    jac_mass_du = dot(acc_term_jac, test_v) 
    jac_convection_du = timestep * theta_const * dot(convection_fluid_v, test_v)
    jac_convection_du += dot(convection_fluid_d, test_v)
    jac_convection_du += dot(convection_fluid_u_old, test_v)
    jac_diffusion_du = timestep * inner(stress_fluid_term_1, test_grad_v)
    jac_diffusion_du += timestep * theta_const * inner(stress_fluid_term_2, test_grad_v)
    # -------- Biharmonic equation ----------
    jac_biharmonic_dd = (-alpha_u/(J*J) * J_LinU * inner(grad(dk), grad(test_w))
                         + alpha_u/J * inner(grad(dd), grad(test_w)))
    # ---------Incompressibility  ----------
    incompressility_ALE_LinALl = NSE_ALE.get_Incompressibility_ALE_LinV_optimized(
        grad_uk, grad(du), F, grad_dd
    )
    jac_incompressibility_dp = incompressility_ALE_LinALl * test_q

    volume_terms_fluid = (jac_mass_du
            + jac_convection_du
            + jac_diffusion_du
            + jac_biharmonic_dd
            + jac_incompressibility_dp
            ) * dx_f
    # -----------------------------------------------------------------------
    # ----------------- do-nothing bc ---------------------------------------
    neuman_term = NSE_ALE.get_stress_fluid_ALE_3rd_term_LinAll_short(
        Finv,
        Finv_LinU,
        grad_uk,
        grad(du),
        mu_f,
        J,
        J_F_inv_T_LinU,
    )
    neuman_flux = dot(neuman_term, n)
    out_flow_jac = - timestep * theta_const * dot(neuman_flux, test_v) * dS_outlet

    #-----------------------------------------------------------------------
    #---------------- Solid terms ---------------------------------------
    #-----------------------------------------------------------------------
    solid_stress_LinU = Structure_Terms.get_Piola_Kirchhoff_1st_LinAll(
        F, grad_dd, mu_s, lambda_s
    )
    jac_solid = ( rho_s * dot(du, test_v)
                    + timestep * theta_const *inner(solid_stress_LinU, grad(test_v))
                    + rho_s * dot(dd, test_w)
                    - theta_const * theta_const * dot(du, test_w)
                    + dp * test_q

    ) * dx_s


    # return volume_terms_fluid + out_flow_jac + jac_solid
    return volume_terms_fluid  + jac_solid
            

def build_forms(
    *,
    uk, u_prev,       # Fluid Velocity (current, old)
    dk, d_prev,       # Displacement (current, old)
    pk, p_prev,       # Pressure (current, old)
    du, dd, dp,       # Trial functions (δv, δd, δp)
    v, w, q,          # Test functions (ψ_v, ψ_d, ψ_p)
    dt_const: Constant,
    theta_const: Constant,
    rho_f: Constant,
    mu_f: Constant,
    rho_s: Constant,
    lambda_s: Constant,
    mu_s: Constant,
    alpha_u: Constant,
    stab_eps: Constant,
    fluid_bs,
    solid_bs,
    quad_order: int,
):
    # --- Integration Measures ---
    dx_f = dx(defined_on=fluid_bs, metadata={"q": quad_order})
    dx_s = dx(defined_on=solid_bs, metadata={"q": quad_order})
    
    # --- Geometric State (Current Newton Iteration) ---
    I = Identity(2)
    F = ALE_Helpers.get_F(grad(dk))
    J = ALE_Helpers.get_J(F)
    Finv = ALE_Helpers.get_F_inv(F)
    cof_F = ALE_Helpers.get_cof_F(F)  # J * Finv.T

    # --- Geometric State (Previous Timestep) ---
    F_old = ALE_Helpers.get_F(grad(d_prev))
    J_old = ALE_Helpers.get_J(F_old)
    
    # --- Geometric Linearization (Shape Derivatives) ---
    # These represent the variation of geometric terms w.r.t displacement trial function (dd)
    grad_dd = grad(dd)
    J_LinU = ALE_Helpers.get_J_LinU(J, Finv, grad_dd)
    Finv_LinU = ALE_Helpers.get_F_inv_LinU(Finv, grad_dd)
    cof_F_LinU = ALE_Helpers.get_cof_F_LinU(grad_dd) # δ(J F^{-T})

    # ========================================================================
    # 1. FLUID RESIDUAL (ALE Navier-Stokes)
    # ========================================================================
    
    # --- Velocity & Gradients ---
    grad_uk = grad(uk)
    grad_u_prev = grad(u_prev)
    
    # --- 1a. Mass / Acceleration Term ---
    # C++: rho * (J + J_old)/2 * (u - u_old)/dt
    acc_term = (rho_f / dt_const) * 0.5 * (J + J_old) * (uk - u_prev)
    
    # --- 1b. ALE Convection Term ---
    # Fluid velocity v, Grid velocity w_g ~ (d - d_old)/dt
    # Term: ρ J ((v - w_g) · F^{-T}∇) v  => ρ J (∇v F^{-1}) (v - w_g)
    vel_mesh = (dk - d_prev) / dt_const
    diff_vel = uk - vel_mesh
    # Convection: rho * J * (grad(u) * Finv) * (u - u_mesh)
    # Note: dot(grad(u), Finv) is (∇u F^{-1})
    conv_term = rho_f * J * dot(dot(grad_uk, Finv), diff_vel)
    
    # Previous timestep convection (for Theta scheme)
    # Note: Using current geometry F/J for old velocity is a common approximation, 
    # but step-fsi.cc uses old geometry for old convection if theta != 1. 
    # Let's stick to the structure implied by C++ 'old_timestep_convection_fluid'.
    Finv_old = ALE_Helpers.get_F_inv(F_old)
    # Assuming old mesh velocity was stored or approximated. 
    # step-fsi.cc approximates old convection using old J, old F, old u.
    conv_term_old = rho_f * J_old * dot(dot(grad_u_prev, Finv_old), u_prev) # Simplifying assumption: w_g_old ~ 0 or small

    # --- 1c. Stress Term (Viscous + Pressure) ---
    # σ_visc = μ (∇u F^{-1} + F^{-T} ∇u^T)
    sigma_visc = NSE_ALE.get_stress_fluid_ALE(mu_f, 0.0, grad_uk, Finv) # p=0 here
    sigma_visc_old = NSE_ALE.get_stress_fluid_ALE(mu_f, 0.0, grad_u_prev, Finv_old)
    
    # Piola-Kirchhoff I (excluding pressure for now) = J * σ * F^{-T}
    # We integrate (P : ∇ψ)
    P_visc = J * dot(sigma_visc, Finv.T)
    P_visc_old = J_old * dot(sigma_visc_old, Finv_old.T)
    
    # Pressure part: - p * J * F^{-T}
    P_pres = -pk * cof_F

    # --- 1d. Incompressibility ---
    # J ∇·u = cof(F) : ∇u
    continuity = inner(cof_F, grad_uk)

    # --- Fluid Residual Assembly ---
    # R_mom = (Acc + Convection)·v + (P_visc + P_pres) : ∇v
    # Time stepping: θ * current + (1-θ) * old
    
    res_f_mom = (
        dot(acc_term, v) 
        + theta_const * (dot(conv_term, v) + inner(P_visc, grad(v)))
        + (1.0 - theta_const) * (dot(conv_term_old, v) + inner(P_visc_old, grad(v)))
        + inner(P_pres, grad(v)) # Pressure is implicit
    ) * dx_f
    
    res_f_cont = (continuity * q) * dx_f
    
    # Stabilization (optional, usually 0 in step-fsi but good for iterative solvers)
    # res_f_stab = stab_eps * pk * q * dx_f # Can add if needed

    # ========================================================================
    # 2. SOLID RESIDUAL (St. Venant-Kirchhoff)
    # ========================================================================
    
    # --- Kinematics ---
    # Green-Lagrange Strain
    E = Structure_Terms.get_E(F)
    E_old = Structure_Terms.get_E(F_old)
    
    # 2nd Piola-Kirchhoff Stress
    S = Structure_Terms.get_S(E, mu_s, lambda_s)
    S_old = Structure_Terms.get_S(E_old, mu_s, lambda_s)
    
    # 1st Piola-Kirchhoff P = F * S
    P_s = dot(F, S)
    P_s_old = dot(F_old, S_old)
    
    # --- Solid Residual ---
    # Mass: ρ_s/dt * (u - u_old) · v  (Velocity is state variable in monolithic FSI)
    # Momentum: P : ∇v
    # Kinematic link: ρ_s * ( (d - d_old)/dt - (θ u + (1-θ) u_old) ) · w
    
    res_s_mom = (
        (rho_s / dt_const) * dot(uk - u_prev, v)
        + theta_const * inner(P_s, grad(v))
        + (1.0 - theta_const) * inner(P_s_old, grad(v))
        - rho_s * theta_const * dot(Constant((0.0, 0.0)), v) # Body force if any
    ) * dx_s
    
    vel_theta = theta_const * uk + (1.0 - theta_const) * u_prev
    res_s_kin = rho_s * dot((dk - d_prev)/dt_const - vel_theta, w) * dx_s

    # ========================================================================
    # 3. MESH MOTION RESIDUAL (Nonlinear Harmonic)
    # ========================================================================
    # - div( (alpha_u / J) * grad(d) ) = 0
    # Weak form: (alpha_u / J) * ∇d : ∇w
    # Jacobian factor 1/J stiffens small elements to prevent inversion
    
    # Note: step-fsi.cc uses (alpha_u * J^{-1} * ∇u : ∇φ)
    # Here u is displacement d.
    res_m = (alpha_u / J) * inner(grad(dk), grad(w)) * dx_f


    # ========================================================================
    # 4. JACOBIAN ASSEMBLY (Manual Linearization)
    # ========================================================================
    
    # --- FLUID BLOCKS ---
    
    # 4a. Mass Matrix Linearization
    # δ(Acc) = ρ/2dt * [ δJ(u - u_old) + (J + J_old)δu ]
    # J_LinU is δJ w.r.t d. du is δu.
    jac_mass_du = (rho_f / dt_const) * 0.5 * (J + J_old) * dot(du, v)
    jac_mass_dd = (rho_f / dt_const) * 0.5 * dot(J_LinU * (uk - u_prev), v)
    
    # 4b. Convection Linearization
    # Uses helper: δ(ρ J ∇u F^{-1} (u - w))
    # We split this into parts w.r.t du and dd
    
    # Part 1: Linearization w.r.t Velocity (du)
    # δ_v (Conv) = ρ J [ ∇δu F^{-1} (u-w) + ∇u F^{-1} δu ]
    conv_du = NSE_ALE.get_Convection_LinAll_short(
        J, 0.0, Finv, 0.0, diff_vel, grad_uk, du, grad(du), rho_f
    ) # Passing 0.0 for geometric linearizations here to isolate velocity part
    
    # Part 2: Linearization w.r.t Geometry (dd)
    # Includes δJ, δF^{-1}, and δw = δd/dt
    # Term A: Geometric changes to transport: ρ [δJ ∇u F^{-1} (u-w) + J ∇u δF^{-1} (u-w)]
    conv_dd_geom = NSE_ALE.get_Convection_LinAll_short(
        0.0, J_LinU, Finv, Finv_LinU, diff_vel, grad_uk, Constant((0,0)), Constant(((0,0),(0,0))), rho_f
    )
    # Term B: Change in grid velocity w: - ρ J ∇u F^{-1} (δd/dt)
    grad_u_Finv = dot(grad_uk, Finv)
    conv_dd_grid = -rho_f * J * dot(grad_u_Finv, dd / dt_const)
    
    conv_dd = conv_dd_geom + conv_dd_grid

    # 4c. Stress Linearization
    # P_visc = J σ F^{-T}. 
    # δP = δJ σ F^{-T} + J δσ F^{-T} + J σ δF^{-T}
    
    # Velocity part (du): J * δσ(du) * F^{-T}
    # Helper 'get_stress...2nd_term' handles J * (δσ_v + δσ_u) * F^{-T} + σ * δ(J F^{-T})
    # We call it twice: once for du (where geometric variations are 0) and once for dd.
    
    # W.r.t du:
    jac_stress_du = NSE_ALE.get_stress_fluid_ALE_2nd_term_LinAll_short(
        0.0, sigma_visc, grad_uk, grad(du), Finv, 0.0, J, mu_f
    )
    # W.r.t dd:
    jac_stress_dd = NSE_ALE.get_stress_fluid_ALE_2nd_term_LinAll_short(
        cof_F_LinU, sigma_visc, grad_uk, Constant(((0,0),(0,0))), Finv, Finv_LinU, J, mu_f
    )
    
    # 4d. Pressure Linearization
    # P_pres = - p * cof(F)
    # W.r.t dp: - δp * cof(F)
    jac_pres_dp = -dp * cof_F
    # W.r.t dd: - p * δ(cof(F))
    jac_pres_dd = -pk * cof_F_LinU
    
    # 4e. Continuity Linearization
    # δ(cof(F) : ∇u) = cof(F) : ∇δu + δcof(F) : ∇u
    jac_cont_du = inner(cof_F, grad(du))
    jac_cont_dd = inner(cof_F_LinU, grad_uk)

    # --- SOLID BLOCKS ---
    
    # 4f. Solid Momentum
    # P = F S. δP = δF S + F δS.
    # δF = ∇δd. δS from STVK helper.
    delta_S, delta_E = Structure_Terms.get_S_LinU(F, grad_dd, mu_s, lambda_s)
    delta_P_s = dot(grad_dd, S) + dot(F, delta_S)
    
    jac_s_du = (rho_s / dt_const) * dot(du, v)
    jac_s_dd = theta_const * inner(delta_P_s, grad(v))
    
    # 4g. Solid Kinematic Link
    # ρ ( δd/dt - θ δu ) · w
    jac_kin_dd = rho_s * dot(dd / dt_const, w)
    jac_kin_du = -rho_s * theta_const * dot(du, w)

    # --- MESH BLOCKS ---
    
    # 4h. Mesh Motion Linearization
    # R_m = α J^{-1} ∇d : ∇w
    # δR_m = α [ δ(J^{-1}) ∇d + J^{-1} ∇δd ] : ∇w
    # δ(J^{-1}) = - J^{-2} δJ = - J^{-2} (J tr(F^{-1} ∇δd)) = - J^{-1} (F^{-T} : ∇δd)
    term_m_1 = (alpha_u / J) * inner(grad_dd, grad(w))
    term_m_2 = - (alpha_u / J) * inner(F_inv.T, grad_dd) * inner(grad(dk), grad(w))
    jac_m_dd = term_m_1 + term_m_2


    # ========================================================================
    # 5. FINAL FORMS
    # ========================================================================
    
    residual_form = (
        res_f_mom + res_f_cont # + res_f_stab
        + res_s_mom + res_s_kin
        + res_m
    )
    
    # Combine Jacobian blocks
    # Terms * v
    jac_v = (
        # Fluid Momentum
        jac_mass_du + theta_const*(dot(conv_du, v) + inner(jac_stress_du, grad(v))) # w.r.t u
        + jac_mass_dd + theta_const*(dot(conv_dd, v) + inner(jac_stress_dd, grad(v))) # w.r.t d
        + inner(jac_pres_dp + jac_pres_dd, grad(v)) # w.r.t p and d
    ) * dx_f + (
        # Solid Momentum
        jac_s_du + jac_s_dd 
    ) * dx_s
    
    # Terms * q (Continuity)
    jac_q = (jac_cont_du + jac_cont_dd) * q * dx_f # + stab...
    
    # Terms * w (Displacement/Mesh)
    jac_w = (
        jac_m_dd * dx_f  # Mesh motion
        + (jac_kin_dd + jac_kin_du) * dx_s # Solid kinematics
    )
    
    jacobian_form = jac_v + jac_q + jac_w

    return jacobian_form, residual_form


def compute_drag_lift(dh: DofHandler, mesh, u: VectorFunction, d: VectorFunction, p: Function, rho: float, mu: float):
    n = grad(d) * 0  # placeholder to silence linter
    

    n = FacetNormal()
    I = Identity(2)
    F = I + grad(d)
    Finv = inv(F)
    J = det(F)
    grad_u = dot(grad(u), Finv)
    sigma = -p * I + rho * mu * (grad_u + grad_u.T)
    Piola = J * sigma * Finv.T
    traction = dot(Piola, n)
    ex = Constant((1.0, 0.0))
    ey = Constant((0.0, 1.0))
    drag = dot(traction, ex) * dS(defined_on=mesh.edge_bitset("cylinder"))
    lift = dot(traction, ey) * dS(defined_on=mesh.edge_bitset("cylinder"))
    comp = FormCompiler(dh, backend="python")
    drag_val = comp.assemble(Equation(drag, None))[1].sum()
    lift_val = comp.assemble(Equation(lift, None))[1].sum()
    return drag_val, lift_val


def tip_displacement(dh: DofHandler, d: VectorFunction):
    coords = dh.get_dof_coords("dx")
    tip = np.array([BEAM_X0 + BEAM_LENGTH, CENTER[1]])
    diffs = coords - tip
    idx = int(np.argmin(np.einsum("ij,ij->i", diffs, diffs)))
    return float(d.nodal_values[idx]), float(d.nodal_values[idx + len(coords)])


# ----------------------------------------------------------------------------- 
# Boundary data (matches BoundaryParabola in step-fsi.cc)
# -----------------------------------------------------------------------------
def inlet_parabola(y: float, t: float, u_mean: float) -> float:
    scale = 4.0 / (H * H) * y * (H - y)
    if t < 2.0:
        return 1.5 * u_mean * 0.5 * (1.0 - math.cos(0.5 * math.pi * t)) * scale
    return 1.5 * u_mean * scale


def build_bcs(
    u_mean: float,
    theta: float,
) -> Tuple[List[BoundaryCondition], List[BoundaryCondition]]:
    zero = lambda x, y, t=0.0: 0.0

    def u_in(x, y, t=0.0):
        return inlet_parabola(y, t, u_mean)

    vel_bcs = [
        BoundaryCondition("ux", "dirichlet", "inlet", u_in),
        BoundaryCondition("uy", "dirichlet", "inlet", zero),
        BoundaryCondition("ux", "dirichlet", "walls", zero),
        BoundaryCondition("uy", "dirichlet", "walls", zero),
        BoundaryCondition("ux", "dirichlet", "cylinder", zero),
        BoundaryCondition("uy", "dirichlet", "cylinder", zero),
    ]
    # Keep the ALE mesh fixed on the outer fluid boundary and the rigid cylinder.
    disp_bcs = [
        BoundaryCondition("dx", "dirichlet", "inlet", zero),
        BoundaryCondition("dy", "dirichlet", "inlet", zero),
        BoundaryCondition("dx", "dirichlet", "walls", zero),
        BoundaryCondition("dy", "dirichlet", "walls", zero),
        BoundaryCondition("dx", "dirichlet", "outlet", zero),
        BoundaryCondition("dy", "dirichlet", "outlet", zero),
        BoundaryCondition("dx", "dirichlet", "cylinder", zero),
        BoundaryCondition("dy", "dirichlet", "cylinder", zero),
    ]
    # Anchor one pressure node to avoid the nullspace
    # p_bcs = [BoundaryCondition("p", "dirichlet", "outlet", zero)]
    bcs = vel_bcs + disp_bcs #+ p_bcs
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, zero) for b in bcs]
    return bcs, bcs_homog


# ----------------------------------------------------------------------------- 
# Main driver
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deal.II FSI benchmark in pycutfem (ALE, conforming mesh).")
    ap.add_argument("--mesh", type=Path, default=Path("../fsi/fsi.inp"), help="Path to fsi.inp UCD mesh.")
    ap.add_argument("--poly-order", type=int, default=2, help="Polynomial order for velocity/displacement (Taylor–Hood).")
    ap.add_argument("--dt", type=float, default=1.0, help="Time step size (default from step-fsi.prm).")
    ap.add_argument("--theta", type=float, default=1.0, help="Theta scheme parameter (1=BE, 0.5=CN).")
    ap.add_argument("--n-steps", type=int, default=3, help="Number of time steps (default small for quick verification).")
    ap.add_argument("--backend", choices=("jit", "python"), default="python", help="Form compiler backend.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    mesh_path = args.mesh.resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    mesh, fluid_bs, solid_bs = load_ucd_mesh(mesh_path, poly_order=1)
    retag_boundaries(mesh)
    outlet_bs = mesh.edge_bitset("outlet")

    # Quick orientation check
    coords = mesh.nodes_x_y_pos
    areas = []
    for el in mesh.elements_list:
        pts = coords[list(el.corner_nodes)]
        area = 0.5 * np.cross(pts, np.roll(pts, -1, axis=0)).sum()
        areas.append(abs(area))
    print(f"Element area stats: min={np.min(areas):.3e}, max={np.max(areas):.3e}")

    element = MixedElement(
        mesh,
        field_specs={"ux": args.poly_order, "uy": args.poly_order, "dx": args.poly_order, "dy": args.poly_order, "p": args.poly_order - 1},
    )
    dh = DofHandler(element, method="cg")

    vel_space = FunctionSpace(name="vel", field_names=["ux", "uy"], dim=1)
    disp_space = FunctionSpace(name="disp", field_names=["dx", "dy"], dim=1)

    du = VectorTrialFunction(space=vel_space, dof_handler=dh)
    dd = VectorTrialFunction(space=disp_space, dof_handler=dh)
    dp = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)

    v = VectorTestFunction(space=vel_space, dof_handler=dh)
    w = VectorTestFunction(space=disp_space, dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)

    uk = VectorFunction(name="u", field_names=["ux", "uy"], dof_handler=dh)
    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    dk = VectorFunction(name="d", field_names=["dx", "dy"], dof_handler=dh)
    d_prev = VectorFunction(name="d_prev", field_names=["dx", "dy"], dof_handler=dh)
    pk = Function(name="p", field_name="p", dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)
    for f in (uk, u_prev, dk, d_prev, pk, p_prev):
        f.nodal_values.fill(0.0)

    # Physical parameters from step-fsi.prm
    rho_f = Constant(1.0e3)
    nu_f = Constant(1.0e-3)
    mu_f = rho_f * nu_f
    rho_s = Constant(1.0e3)
    mu_s = Constant(0.5e6)
    nu_s = 0.4 # Poisson ratio
    E_s = 2.0 * float(mu_s.value) * (1.0 + nu_s)
    lambda_s = Constant(E_s * nu_s / ((1.0 + nu_s) * (1.0 - 2.0 * nu_s)))
    alpha_u = Constant(1.0e-8) # mesh control parameter
    stab_eps = Constant(1.0e-8)
    dt_const = Constant(args.dt)
    theta_const = Constant(args.theta)
    quad_order = 2 * args.poly_order + 4

    # Drop pressure DOFs living purely in the solid.
    dropped_p = dh.tag_dofs_from_element_bitset("inactive", "p", solid_bs, strict=True)
    if dropped_p:
        print(f"Dropped {len(dropped_p)} pressure DOFs inside the solid.")

    jac_form = build_jac(
        uk=uk,
        u_prev=u_prev,
        dk=dk,
        d_prev=d_prev,
        pk=pk,
        p_prev=p_prev,
        du=du,
        dd=dd,
        dp=dp,
        test_v=v,
        test_w=w,
        test_q=q,
        timestep=dt_const,
        theta_const=theta_const,
        rho_f=rho_f,
        mu_f=mu_f,
        rho_s=rho_s,
        lambda_s=lambda_s,
        mu_s=mu_s,
        alpha_u=alpha_u,
        stab_eps=stab_eps,
        fluid_bs=fluid_bs,
        solid_bs=solid_bs,
        outlet_bs=outlet_bs,
        quad_order=quad_order,
    )
    return assemble_form(Equation(jac_form, None), dh, backend='python')
    jac_form, res_form = build_forms(
        uk=uk,
        u_prev=u_prev,
        dk=dk,
        d_prev=d_prev,
        pk=pk,
        p_prev=p_prev,
        du=du,
        dd=dd,
        dp=dp,
        v=v,
        w=w,
        q=q,
        dt_const=dt_const,
        theta_const=theta_const,
        rho_f=rho_f,
        mu_f=mu_f,
        rho_s=rho_s,
        lambda_s=lambda_s,
        mu_s=mu_s,
        alpha_u=alpha_u,
        stab_eps=stab_eps,
        fluid_bs=fluid_bs,
        solid_bs=solid_bs,
        quad_order=quad_order,
    )

    bcs, bcs_homog = build_bcs(u_mean=0.2, theta=args.theta)

    dh.apply_bcs(bcs, uk, u_prev, dk, d_prev, pk, p_prev)
    print(f"Mesh elements: fluid={fluid_bs.cardinality()}, solid={solid_bs.cardinality()}")
    print(f"Total DOFs: {dh.total_dofs}")
    bc_dofs = dh.get_dirichlet_data(bcs)
    print(f"Dirichlet constraints: {len(bc_dofs)} DOFs")

    solver = NewtonSolver(
        residual_form=res_form,
        jacobian_form=jac_form,
        dof_handler=dh,
        mixed_element=element,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=1e-8, max_newton_iter=20, line_search=True),
        quad_order=quad_order,
        backend=args.backend,
    )
    time_params = TimeStepperParameters(dt=args.dt, max_steps=args.n_steps, stop_on_steady=False, final_time=args.dt * args.n_steps)
    aux_funcs: Dict[str, object] = {
        "u_ux": uk.components[0],
        "u_uy": uk.components[1],
        "u_prev_ux": u_prev.components[0],
        "u_prev_uy": u_prev.components[1],
        "d_dx": dk.components[0],
        "d_dy": dk.components[1],
        "d_prev_dx": d_prev.components[0],
        "d_prev_dy": d_prev.components[1],
    }

    delta, steps, elapsed = solver.solve_time_interval(
        functions=[uk, dk, pk],
        prev_functions=[u_prev, d_prev, p_prev],
        aux_functions=aux_funcs,
        time_params=time_params,
    )

    try:
        drag, lift = compute_drag_lift(dh, mesh, uk, dk, pk, rho_f.value, mu_f.value)
    except Exception as exc:
        drag = lift = float("nan")
        print(f"Drag/lift computation skipped: {exc}")
    tip_dx, tip_dy = tip_displacement(dh, dk)
    print(
        f"Solved {steps} step(s) in {elapsed:.2f}s, ||ΔU||_inf={np.linalg.norm(delta, np.inf):.3e}, "
        f"drag={drag:.3e}, lift={lift:.3e}, tip=({tip_dx:.3e},{tip_dy:.3e})"
    )


if __name__ == "__main__":
    main()
