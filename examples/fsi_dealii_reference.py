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
from pycutfem.utils.gmsh_loader import mesh_from_gmsh

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


def tag_nodes_from_edges(mesh: Mesh) -> Dict[str, int]:
    """
    Populate Node.tag with a comma-separated list of incident boundary tags.
    Returns a count per tag for quick diagnostics.
    """
    tag_to_nodes: Dict[str, set[int]] = {}
    for edge in mesh.edges_list:
        if edge.right is not None:
            continue
        tag = getattr(edge, "tag", "") or ""
        if not tag:
            continue
        nodes = edge.all_nodes if edge.all_nodes else edge.nodes
        tag_to_nodes.setdefault(tag, set()).update(int(n) for n in nodes)

    for nid in range(len(mesh.nodes_list)):
        mesh.nodes_list[nid].tag = ""
    for tag, nodes in tag_to_nodes.items():
        for nid in nodes:
            node = mesh.nodes_list[int(nid)]
            pieces = set(node.tag.split(",")) if node.tag else set()
            pieces.add(tag)
            node.tag = ",".join(sorted(p for p in pieces if p))

    return {tag: len(nodes) for tag, nodes in tag_to_nodes.items()}


def classify_fluid_solid(mesh: Mesh, tol: float = 1.0e-9) -> Tuple[BitSet, BitSet]:
    """
    Ensure the mesh has fluid/solid element bitsets. If tags already exist, reuse them;
    otherwise classify geometrically (beam box minus the circular hole is 'solid').
    """
    cached = getattr(mesh, "_elem_bitsets", {})
    if "fluid" in cached and "solid" in cached:
        return mesh.element_bitset("fluid"), mesh.element_bitset("solid")

    beam_x0 = CENTER[0] + RADIUS
    beam_x1 = beam_x0 + BEAM_LENGTH
    beam_y0 = CENTER[1] - 0.5 * BEAM_HEIGHT
    beam_y1 = CENTER[1] + 0.5 * BEAM_HEIGHT
    rad_tol = RADIUS + tol

    coords = mesh.nodes_x_y_pos[mesh.corner_connectivity].mean(axis=1).T
    cx, cy = coords[0], coords[1]
    inside_mask = (
        (cx >= beam_x0 - tol)
        & (cx <= beam_x1 + tol)
        & (cy >= beam_y0 - tol)
        & (cy <= beam_y1 + tol)
        & (np.hypot(cx - CENTER[0], cy - CENTER[1]) >= rad_tol)
    )
    tags = np.where(inside_mask, "solid", "fluid")
    for el, tag in zip(mesh.elements_list, tags):
        el.tag = str(tag)
    mesh._elem_bitsets = {
        "fluid": BitSet(tags == "fluid"),
        "solid": BitSet(tags == "solid"),
    }
    return mesh._elem_bitsets["fluid"], mesh._elem_bitsets["solid"]


def ensure_boundary_tags(mesh: Mesh, tol: float = 1.0e-6) -> Dict[str, Tuple[int, int]]:
    """
    Make sure standard boundary tags exist; if outlet (or inlet) is empty,
    fall back to geometric retagging. Returns counts before/after.
    """
    tags = ("inlet", "outlet", "walls", "cylinder", "beam_outer", "beam_root")
    counts_before = {t: mesh.edge_bitset(t).cardinality() for t in tags}
    need_retag = counts_before.get("outlet", 0) == 0 or counts_before.get("inlet", 0) == 0
    if need_retag:
        retag_boundaries(mesh, tol=tol)
    counts_after = {t: mesh.edge_bitset(t).cardinality() for t in tags}

    # If outlet is still empty, force-tag boundary edges near x=L.
    if counts_after.get("outlet", 0) == 0:
        for edge in mesh.edges_list:
            if edge.right is not None:
                continue
            mpx, mpy = mesh.nodes_x_y_pos[list(edge.nodes)].mean(axis=0)
            if abs(mpx - L) < max(tol, 1e-6):
                edge.tag = "outlet"
        mask = np.fromiter((e.tag == "outlet" for e in mesh.edges_list), bool)
        if not hasattr(mesh, "_edge_bitsets"):
            mesh._edge_bitsets = {}
        mesh._edge_bitsets["outlet"] = BitSet(mask)
        counts_after["outlet"] = int(mask.sum())
    return {t: (counts_before.get(t, 0), counts_after.get(t, 0)) for t in tags}


def boundary_nodes_by_tag(mesh: Mesh) -> Dict[str, set[int]]:
    """
    Collect boundary nodes for each edge tag, using all_nodes to capture
    higher-order edge nodes.
    """
    out: Dict[str, set[int]] = {}
    for e in mesh.edges_list:
        if e.right is not None:
            continue
        tag = getattr(e, "tag", "") or ""
        if not tag:
            continue
        nodes = e.all_nodes if e.all_nodes else e.nodes
        out.setdefault(tag, set()).update(int(n) for n in nodes)
    return out


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
    def get_Cauchy_stress(F, mu_s, lambda_s):
        r"""
        Cauchy stress (STVK):

        E = 1/2 (F^T F - I),

        S = λ tr(E) I + 2 μ E,

        σ = 1/J F S F^T.
        """
        J = det(F)
        E = Structure_Terms.get_E(F)
        S = Structure_Terms.get_S(E, mu_s, lambda_s)
        return (1.0 / J) * dot(dot(F, S), F.T)
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
    theta: Constant,
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
    J_LinU = ALE_Helpers.get_J_LinU(F, grad_dd) # trial of displacement
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
    jac_convection_du = timestep * theta * dot(convection_fluid_v, test_v)
    jac_convection_du += -dot(convection_fluid_d, test_v)
    jac_convection_du += dot(convection_fluid_u_old, test_v)
    jac_diffusion_du = timestep * inner(stress_fluid_term_1, test_grad_v)
    jac_diffusion_du += timestep * theta * inner(stress_fluid_term_2, test_grad_v)
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
    out_flow_jac = - timestep * theta * dot(neuman_flux, test_v) * dS_outlet

    #-----------------------------------------------------------------------
    #---------------- Solid terms ---------------------------------------
    #-----------------------------------------------------------------------
    solid_stress_LinU = Structure_Terms.get_Piola_Kirchhoff_1st_LinAll(
        F, grad_dd, mu_s, lambda_s
    )
    jac_solid = ( rho_s * dot(du, test_v)
                    + timestep * theta *inner(solid_stress_LinU, grad(test_v))
                    + rho_s * dot(dd, test_w)
                    - timestep * theta  * dot(du, test_w)
                    + dp * test_q

    ) * dx_s


    # return volume_terms_fluid + out_flow_jac + jac_solid
    return volume_terms_fluid  + jac_solid

def build_residual(
    *,
    uk, u_prev,       # Fluid Velocity (current, old)
    dk, d_prev,       # Displacement (current, old)
    pk, p_prev,       # Pressure (current, old)
    v_test, w_test, q_test,          # Test functions (ψ_v, ψ_d, ψ_p)
    dt: Constant,
    theta: Constant,
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
    I = Identity(2)
    n =  FacetNormal()
    grad_v = grad(uk)
    grad_d = grad(dk)
    grad_v_old = grad(u_prev)
    grad_d_old = grad(d_prev)
    F = ALE_Helpers.get_F(grad_d)
    Finv = ALE_Helpers.get_F_inv(F)
    J = ALE_Helpers.get_J(F)
    F_old = ALE_Helpers.get_F(grad_d_old)
    Finv_old = ALE_Helpers.get_F_inv(F_old)
    J_old = ALE_Helpers.get_J(F_old)
    pI = pk * Identity(2)
    # acceleration term
    acc_term = rho_f * 0.5 * (J + J_old) * dot((uk - u_prev), v_test)
    # convection term
    convection_fluid = rho_f * J * dot(dot(grad_v, Finv), uk)
    convection_fluid_with_u = rho_f * J * dot(dot(grad_v, Finv), dk)
    convection_fluid_with_u_old = rho_f * J * dot(dot(grad_v, Finv), d_prev)
    old_convection_fluid = rho_f * J_old * dot(dot(grad_v_old, Finv_old), u_prev)
    convec_term = (
        dt * theta * dot(convection_fluid, v_test)
        + dt * (1.0-theta) * dot(old_convection_fluid, v_test)
        - dot(convection_fluid_with_u - convection_fluid_with_u_old, v_test)
    ) 
    # incompressibility term
    fluid_pressure = -(J * dot(pI, Finv.T))
    pressure_term = dt * inner(fluid_pressure, grad(v_test))
    # stress terms
    sigma_ALE = NSE_ALE.get_stress_fluid_except_pressure_ALE(mu_f, grad_v, Finv)
    sigma_ALE_old = NSE_ALE.get_stress_fluid_except_pressure_ALE(mu_f, grad_v_old, Finv_old)
    stress_fluid_viscous = J * dot(sigma_ALE, Finv.T)
    stress_fluid_viscous_old = J_old * dot(sigma_ALE_old, Finv_old.T)
    stress_term = dt * theta * inner(stress_fluid_viscous, grad(v_test))
    stress_term += dt * (1.0-theta) * inner(stress_fluid_viscous_old, grad(v_test))
    # biharmonic stabilization
    biharmonic_term = (
        alpha_u / J * inner(grad(dk), grad(w_test))
    )
    # incompressibility
    incompressibility_fluid = NSE_ALE.get_Incompressibility_ALE(uk, F)
    incompressibility_term = incompressibility_fluid * q_test

    residual_fluid = (
        acc_term
        + convec_term
        + pressure_term
        + stress_term
        + biharmonic_term
        + incompressibility_term
    ) * dx_f
    # do-nothing BC at outlet
    sigma_ALE_tilde = mu_f * dot(Finv.T, grad_v.T) 
    sigma_ALE_tilde_old = mu_f * dot(Finv_old.T, grad_v_old.T)
    stress_fluid_transpose = J * dot(sigma_ALE_tilde, Finv.T)
    stress_fluid_transpose_old = J_old * dot(sigma_ALE_tilde_old, Finv_old.T)
    neuman_flux = dot(stress_fluid_transpose, n)
    neuman_flux_old = dot(stress_fluid_transpose_old, n)
    out_flow = (- dt * theta * dot(neuman_flux, v_test) 
                - dt * (1.0-theta) * dot(neuman_flux_old, v_test)
    ) 
    residual_outlet = out_flow * dS_outlet

    #-----------------------------------------------------------------------
    #---------------- Solid terms ---------------------------------------
    #-----------------------------------------------------------------------
    solid_stress = Structure_Terms.get_Cauchy_stress(F, mu_s, lambda_s)
    solid_stress_old = Structure_Terms.get_Cauchy_stress(F_old, mu_s, lambda_s)
    solid_stress_transfomed = J * dot(solid_stress, Finv.T)
    solid_stress_transfomed_old = J_old * dot(solid_stress_old, Finv_old.T)
    residual_solid = (
        rho_s * dot(uk - u_prev, v_test)
        + dt * theta * inner(solid_stress_transfomed, grad(v_test))
        + dt * (1.0-theta) * inner(solid_stress_transfomed_old, grad(v_test))
        + rho_s * dot(dk - d_prev, w_test)
        - rho_s * dt * theta * dot(uk, w_test)
        - rho_s * dt * (1.0-theta) * dot(u_prev, w_test)
        + pk * q_test
    ) * dx_s

    
    # return residual_fluid + residual_outlet + residual_solid
    return residual_fluid  + residual_solid
            



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
    ap.add_argument("--mesh", type=Path, default=Path("examples/meshes/fsi_conforming.msh"), help="Path to mesh (.msh from gmsh or .inp UCD).")
    ap.add_argument("--mesh-format", choices=("auto", "gmsh", "ucd"), default="auto", help="Override mesh loader; auto picks by file extension.")
    ap.add_argument("--poly-order", type=int, default=2, help="Polynomial order for velocity/displacement (Taylor–Hood).")
    ap.add_argument("--dt", type=float, default=1.0, help="Time step size (default from step-fsi.prm).")
    ap.add_argument("--theta", type=float, default=1.0, help="Theta scheme parameter (1=BE, 0.5=CN).")
    ap.add_argument("--n-steps", type=int, default=3, help="Number of time steps (default small for quick verification).")
    ap.add_argument("--backend", choices=("jit", "python"), default="python", help="Form compiler backend.")
    ap.add_argument("--boundary-tol", type=float, default=1.0e-6, help="Tolerance for geometric boundary retagging.")
    ap.add_argument("--assemble-only", action="store_true", help="Assemble residual/Jacobian once and exit (no Newton solve).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    mesh_path = args.mesh.resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    use_gmsh = args.mesh_format == "gmsh" or (args.mesh_format == "auto" and mesh_path.suffix.lower() == ".msh")
    if use_gmsh:
        mesh = mesh_from_gmsh(mesh_path, apply_boundary_tags=True)
        fluid_bs, solid_bs = classify_fluid_solid(mesh)
    else:
        mesh, fluid_bs, solid_bs = load_ucd_mesh(mesh_path, poly_order=1)
        fluid_bs, solid_bs = classify_fluid_solid(mesh)

    counts = ensure_boundary_tags(mesh, tol=args.boundary_tol)
    node_counts = tag_nodes_from_edges(mesh)
    outlet_bs = mesh.edge_bitset("outlet")
    if outlet_bs.cardinality() == 0:
        raise RuntimeError("Outlet boundary is empty after retagging; check mesh geometry.")
    counts_msg = ", ".join(f"{k}:{v[0]}->{v[1]}" for k, v in counts.items())
    nodes_msg = ", ".join(f"{k}:{v}" for k, v in node_counts.items())
    print(f"Boundary edges (before→after): {counts_msg}")
    print(f"Boundary nodes per tag: {nodes_msg}")

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
    
    res_form = build_residual(
        uk=uk,
        u_prev=u_prev,
        dk=dk,
        d_prev=d_prev,
        pk=pk,
        p_prev=p_prev,
        v_test=v,   w_test=w, q_test=q,
        dt=dt_const,
        theta=theta_const,
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
    # return assemble_form(Equation(None, res_form), dh, backend='python')

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
        theta=theta_const,
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
    # return assemble_form(Equation(jac_form, None), dh, backend='python')
    

    bcs, bcs_homog = build_bcs(u_mean=0.2, theta=args.theta)

    dh.apply_bcs(bcs, uk, u_prev, dk, d_prev, pk, p_prev)
    print(f"Mesh elements: fluid={fluid_bs.cardinality()}, solid={solid_bs.cardinality()}")
    print(f"Total DOFs: {dh.total_dofs}")
    bc_dofs = dh.get_dirichlet_data(bcs)
    print(f"Dirichlet constraints: {len(bc_dofs)} DOFs")
    bc_dof_set = set(bc_dofs.keys())
    boundary_nodes = boundary_nodes_by_tag(mesh)
    bc_fields_by_tag: Dict[str, set[str]] = {}
    for bc in bcs:
        if getattr(bc, "method", "") != "dirichlet":
            continue
        bc_fields_by_tag.setdefault(bc.domain_tag, set()).add(bc.field)

    coverage_report = []
    geom_nodes_union: set[int] = set()
    for tag, fields in bc_fields_by_tag.items():
        nodes = boundary_nodes.get(tag, set())
        geom_nodes_union |= nodes
        for field in fields:
            node2dof = dh.dof_map.get(field, {})
            expected = {node2dof[n] for n in nodes if n in node2dof}
            missing = expected - bc_dof_set
            coverage_report.append((field, tag, len(expected), len(missing)))
    print(f"Boundary geometry nodes touched by BCs: {len(geom_nodes_union)}")
    for field, tag, exp, miss in coverage_report:
        msg = f"{field}@{tag}: expected {exp}, missing {miss}"
        if miss:
            msg += f" (first few missing: {sorted(list(missing))[:5]})"
        print("  " + msg)
    # Hard verification: all expected boundary DOFs must be present.
    expected_all: set[int] = set()
    for tag, fields in bc_fields_by_tag.items():
        nodes = boundary_nodes.get(tag, set())
        for field in fields:
            node2dof = dh.dof_map.get(field, {})
            expected_all |= {node2dof[n] for n in nodes if n in node2dof}
    missing_all = expected_all - bc_dof_set
    extra_all = bc_dof_set - expected_all
    if missing_all:
        raise RuntimeError(f"Dirichlet coverage incomplete: {len(missing_all)} boundary DOFs missing (examples: {sorted(list(missing_all))[:10]}).")
    if extra_all:
        print(f"[warn] Dirichlet set has {len(extra_all)} DOFs not on tagged boundaries (examples: {sorted(list(extra_all))[:10]}).")
    else:
        print("All boundary DOFs covered by supplied Dirichlet tags.")

    if args.assemble_only:
        print("Assembling once with backend='{0}' for diagnostics...".format(args.backend))
        eq_res = Equation(None, res_form)
        eq_jac = Equation(jac_form, None)
        assemble_form(eq_res, dof_handler=dh, bcs=bcs, backend=args.backend)
        assemble_form(eq_jac, dof_handler=dh, bcs=bcs, backend=args.backend)
        print("Assembly completed; exiting early (--assemble-only).")
        return

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
