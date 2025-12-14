#!/usr/bin/env python
"""
Compare the full FSI residual/Jacobian between PyCutFEM and FEniCSx on the
deal.II reference mesh (examples/meshes/fsi_conforming.msh).

This reuses the PyCutFEM forms from `examples/fsi_dealii_reference.py` and
assembles the same expressions in FEniCSx on the identical mesh (including
physical tags from Gmsh) to pinpoint discrepancies that may explain the lack
of convergence in fsi_dealii_reference.py.

Usage:
    conda run --no-capture-output -n fenicsx \\
        python examples/debug/comparison_fsi_fullmesh.py [--backend jit] [--quad-order N] [--dump]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from scipy.sparse import csr_matrix

import dolfinx
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
import ufl
import basix.ufl
from basix.ufl import mixed_element

from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.topology import Node
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    VectorFunction,
    TrialFunction,
    TestFunction,
    VectorTrialFunction,
    VectorTestFunction,
    Identity,
    grad,
    inner,
    dot,
    det,
    inv,
    cof,
    FacetNormal,
)
from pycutfem.ufl.measures import dx, dS
from pycutfem.ufl.forms import Equation, assemble_form

from examples.fsi_dealii_reference import (
    classify_fluid_solid,
    ensure_boundary_tags,
    build_residual as build_residual_pc,
    build_jac as build_jac_pc,
    ALE_Helpers,
    NSE_ALE,
    Structure_Terms,
)


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
FIELDS = ["ux", "uy", "dx", "dy", "p"]
OUTPUT_DIR = Path("garbage/fsi_fullmesh")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PHYS_TAGS = {11: "inlet", 12: "walls", 13: "outlet", 14: "cylinder", 15: "beam_outer", 16: "beam_root"}
# PyCutFEM stores Q2 quads in row-major order on the 3x3 tensor grid:
#   [0 1 2
#    3 4 5
#    6 7 8]
# Basix/UFC for a Q2 quadrilateral expects:
#   vertices (bl, br, tl, tr), edge mids (bottom, left, right, top), center.
ROW_MAJOR_TO_BASIX_Q2 = np.array([0, 2, 6, 8, 1, 3, 5, 7, 4], dtype=int)
ROW_MAJOR_Q2_CORNERS = np.array([0, 2, 8, 6], dtype=int)  # (bl, br, tr, tl)


# --------------------------------------------------------------------------- #
# Mesh loading (meshio to avoid gmsh MPI issues)                              #
# --------------------------------------------------------------------------- #
def read_mesh(mesh_path: Path):
    import meshio

    msh = meshio.read(mesh_path)
    coords = msh.points[:, :2]

    quad_cells: list[np.ndarray] = []
    quad_tags: list[np.ndarray] = []
    line_cells: list[np.ndarray] = []
    line_tags: list[np.ndarray] = []
    phys = msh.cell_data.get("gmsh:physical", [])
    for (cells, tags) in zip(msh.cells, phys):
        if cells.type.startswith("quad"):
            quad_cells.append(cells.data)
            quad_tags.append(tags)
        elif cells.type.startswith("line"):
            line_cells.append(cells.data)
            line_tags.append(tags)
    if not quad_cells or not line_cells:
        raise RuntimeError("Mesh missing quad or line cells with physical tags.")

    quad_cells = np.vstack(quad_cells).astype(np.int64, copy=False)
    quad_tags = np.concatenate(quad_tags).astype(np.int32, copy=False)
    line_cells = np.vstack(line_cells).astype(np.int64, copy=False)
    line_tags = np.concatenate(line_tags).astype(np.int32, copy=False)

    coords_round = np.round(coords, 14)
    unique_coords, inverse = np.unique(coords_round, axis=0, return_inverse=True)
    quad_cells = inverse[quad_cells]
    line_cells = inverse[line_cells]

    # gmsh (bl, br, tr, tl, mid-b, mid-r, mid-t, mid-l, center)
    # -> PyCutFEM row-major Q2 (matches load_gmsh_mesh in gmsh_loader)
    perm = np.array([0, 4, 1, 7, 8, 5, 3, 6, 2], dtype=int)
    quad_cells = quad_cells[:, perm]
    for i in range(quad_cells.shape[0]):
        arr = quad_cells[i].reshape(3, 3)
        corners = arr[[0, 0, 2, 2], [0, 2, 2, 0]]
        pts = unique_coords[corners]
        signed_area = 0.5 * float(
            np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
        )
        if signed_area <= 0.0:
            arr = arr[:, ::-1]
        quad_cells[i] = arr.reshape(-1)

    used = np.unique(np.concatenate([quad_cells.ravel(), line_cells.ravel()]))
    remap = -np.ones(unique_coords.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    coords_new = unique_coords[used]
    quad_cells = remap[quad_cells]
    line_cells = remap[line_cells]
    line_mask = np.all(line_cells >= 0, axis=1)
    line_cells = line_cells[line_mask]
    line_tags = line_tags[line_mask]

    return coords_new, quad_cells, quad_tags, line_cells, line_tags


# --------------------------------------------------------------------------- #
# DOF mapping between PyCutFEM and FEniCSx                                    #
# --------------------------------------------------------------------------- #
def _coord_key(xy: np.ndarray, *, decimals: int = 12) -> tuple[float, float]:
    return (round(float(xy[0]), decimals), round(float(xy[1]), decimals))


def _edge_key_from_points(p0: np.ndarray, p1: np.ndarray, *, decimals: int = 12) -> tuple[tuple[float, float], tuple[float, float]]:
    a = _coord_key(p0, decimals=decimals)
    b = _coord_key(p1, decimals=decimals)
    return (a, b) if a <= b else (b, a)


def one_to_one_map_coords(coords_pc: np.ndarray, coords_fx: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    """
    Return an index array `perm` such that `coords_fx[perm[i]]` matches `coords_pc[i]`.

    The previous implementation used an O(n^3) assignment solve; for full meshes
    this is prohibitively expensive. Here we use a robust coordinate hash with a
    KDTree fallback.
    """
    if coords_pc.shape[0] != coords_fx.shape[0]:
        raise ValueError(f"Coordinate arrays differ in size: pc={coords_pc.shape[0]} fx={coords_fx.shape[0]}")
    if coords_pc.size == 0:
        return np.array([], dtype=int)

    # Fast path: exact (rounded) coordinate matching.
    fx_map: dict[tuple[float, float], int] = {}
    for j, xy in enumerate(coords_fx):
        key = _coord_key(xy)
        if key in fx_map:
            raise RuntimeError(f"Duplicate FEniCSx DoF coordinates detected at {key}.")
        fx_map[key] = int(j)

    perm = np.empty(coords_pc.shape[0], dtype=int)
    missing = []
    for i, xy in enumerate(coords_pc):
        key = _coord_key(xy)
        j = fx_map.get(key)
        if j is None:
            missing.append((i, key))
            perm[i] = -1
        else:
            perm[i] = j

    if not missing:
        if np.unique(perm).size != perm.size:
            raise RuntimeError("Non-bijective coordinate map (duplicate matches).")
        return perm

    # Fallback: nearest-neighbour match.
    from scipy.spatial import cKDTree

    tree = cKDTree(coords_fx)
    dists, idxs = tree.query(coords_pc, k=1)
    max_dist = float(np.max(dists)) if dists.size else 0.0
    if max_dist > tol:
        i0 = int(np.argmax(dists))
        raise RuntimeError(
            f"Failed to map DoFs by coordinates (max dist {max_dist:.3e} > tol {tol:.3e}). "
            f"Worst pc[{i0}]={coords_pc[i0]} -> fx[{int(idxs[i0])}]={coords_fx[int(idxs[i0])]}"
        )
    if np.unique(idxs).size != idxs.size:
        raise RuntimeError("Nearest-neighbour mapping is not one-to-one (duplicate matches).")
    return idxs.astype(int)


def get_pc_field_coords(dh: DofHandler, field: str) -> Tuple[np.ndarray, np.ndarray]:
    coords = dh.get_dof_coords(field)
    ids = dh.get_field_slice(field)
    return coords, ids


def get_fx_field_coords(W, field: str):
    block_component = {
        "ux": (0, 0),
        "uy": (0, 1),
        "dx": (1, 0),
        "dy": (1, 1),
        "p": (2, None),
    }[field]
    if block_component[1] is None:
        subspace, parent_map = W.sub(block_component[0]).collapse()
    else:
        subspace, parent_map = W.sub(block_component[0]).sub(block_component[1]).collapse()
    coords = subspace.tabulate_dof_coordinates()[:, :2]
    return coords, np.array(parent_map, dtype=int)


def create_true_dof_map(dh_pc: DofHandler, W_fx) -> np.ndarray:
    P = np.zeros(dh_pc.total_dofs, dtype=int)
    for field in FIELDS:
        pc_coords, pc_ids = get_pc_field_coords(dh_pc, field)
        fx_coords, fx_parent = get_fx_field_coords(W_fx, field)
        perm = one_to_one_map_coords(pc_coords, fx_coords)
        P[pc_ids] = fx_parent[perm]
    return P


def fenics_block_sizes(W_fx):
    blocks = []
    nsubs = W_fx.ufl_element().num_sub_elements
    for i in range(nsubs):
        sub, _ = W_fx.sub(i).collapse()
        blocks.append(sub.dofmap.index_map.size_local * sub.dofmap.index_map_bs)
    return blocks


# --------------------------------------------------------------------------- #
# Initial data (non-zero to exercise all terms)                               #
# --------------------------------------------------------------------------- #
def _vel_init(x, y):
    return np.vstack((1.0 + 0.25 * x + 0.1 * y, -0.3 + 0.15 * x * y))


def _disp_init(x, y):
    return np.vstack((0.05 * x, -0.02 * y))


def _pres_init(x, y):
    return 0.25 + 0.1 * x - 0.05 * y


def initialize_pycutfem(pc: Dict):
    pc["uk"].set_values_from_function(lambda x, y: _vel_init(x, y))
    pc["u_prev"].set_values_from_function(lambda x, y: 0.7 * _vel_init(x, y))
    pc["dk"].set_values_from_function(lambda x, y: _disp_init(x, y))
    pc["d_prev"].set_values_from_function(lambda x, y: 0.5 * _disp_init(x, y))
    pc["pk"].set_values_from_function(lambda x, y: _pres_init(x, y))
    pc["p_prev"].set_values_from_function(lambda x, y: 0.8 * _pres_init(x, y))


def _fx_interpolator(func):
    def wrapper(x):
        vals = func(x[0], x[1])
        arr = np.atleast_2d(vals)
        return arr
    return wrapper


def initialize_fenics(fx: Dict):
    W = fx["W"]
    state_k = fx["state_k"]
    state_prev = fx["state_prev"]

    sub_u, map_u = W.sub(0).collapse()
    sub_d, map_d = W.sub(1).collapse()
    sub_p, map_p = W.sub(2).collapse()

    u_fun = dolfinx.fem.Function(sub_u)
    u_fun.interpolate(_fx_interpolator(_vel_init))
    d_fun = dolfinx.fem.Function(sub_d)
    d_fun.interpolate(_fx_interpolator(_disp_init))
    p_fun = dolfinx.fem.Function(sub_p)
    p_fun.interpolate(_fx_interpolator(_pres_init))

    state_k.x.array[map_u] = u_fun.x.array
    state_k.x.array[map_d] = d_fun.x.array
    state_k.x.array[map_p] = p_fun.x.array
    state_k.x.scatter_forward()

    u_prev_fun = dolfinx.fem.Function(sub_u)
    u_prev_fun.interpolate(_fx_interpolator(lambda x, y: 0.7 * _vel_init(x, y)))
    d_prev_fun = dolfinx.fem.Function(sub_d)
    d_prev_fun.interpolate(_fx_interpolator(lambda x, y: 0.5 * _disp_init(x, y)))
    p_prev_fun = dolfinx.fem.Function(sub_p)
    p_prev_fun.interpolate(_fx_interpolator(lambda x, y: 0.8 * _pres_init(x, y)))

    state_prev.x.array[map_u] = u_prev_fun.x.array
    state_prev.x.array[map_d] = d_prev_fun.x.array
    state_prev.x.array[map_p] = p_prev_fun.x.array
    state_prev.x.scatter_forward()


# --------------------------------------------------------------------------- #
# FEniCSx helper formulas                                                     #
# --------------------------------------------------------------------------- #
def cof2(A):
    return ufl.as_tensor(((A[1, 1], -A[1, 0]), (-A[0, 1], A[0, 0])))


def structure_cauchy(F, mu_s, lambda_s):
    E = 0.5 * (ufl.dot(F.T, F) - ufl.Identity(2))
    S = lambda_s * ufl.tr(E) * ufl.Identity(2) + 2.0 * mu_s * E
    J = ufl.det(F)
    return (1.0 / J) * ufl.dot(ufl.dot(F, S), F.T)


def structure_tangent(F, grad_dd, mu_s, lambda_s):
    delta_F = grad_dd
    E = 0.5 * (ufl.dot(F.T, F) - ufl.Identity(2))
    delta_E = 0.5 * (ufl.dot(delta_F.T, F) + ufl.dot(F.T, delta_F))
    delta_S = lambda_s * ufl.tr(delta_E) * ufl.Identity(2) + 2.0 * mu_s * delta_E
    S = lambda_s * ufl.tr(E) * ufl.Identity(2) + 2.0 * mu_s * E
    return ufl.dot(delta_F, S) + ufl.dot(F, delta_S)


def acceleration_term(J, J_old, J_lin, v, v_old, v_trial, rho):
    return 0.5 * rho * (J_lin * (v - v_old) + (J + J_old) * v_trial)


def convection_lin(phi_grad_v_trial, phi_v_trial, J, J_lin, F_inv, F_inv_lin, v, grad_v, rho):
    grad_v_Finv = ufl.dot(grad_v, F_inv)
    grad_v_Finv_lin = ufl.dot(grad_v, F_inv_lin)
    return rho * (
        J_lin * ufl.dot(grad_v_Finv, v)
        + J * ufl.dot(grad_v_Finv_lin, v)
        + J * ufl.dot(ufl.dot(phi_grad_v_trial, F_inv), v)
        + J * ufl.dot(grad_v_Finv, phi_v_trial)
    )


def convection_u_lin(phi_grad_v_trial, phi_u_trial, J, J_lin, F_inv, F_inv_lin, u_disp, grad_v, rho):
    grad_v_Finv = ufl.dot(grad_v, F_inv)
    grad_v_Finv_lin = ufl.dot(grad_v, F_inv_lin)
    return rho * (
        J_lin * ufl.dot(grad_v_Finv, u_disp)
        + J * ufl.dot(grad_v_Finv_lin, u_disp)
        + J * ufl.dot(grad_v_Finv, phi_u_trial)
        + J * ufl.dot(phi_grad_v_trial, ufl.dot(F_inv, u_disp))
    )


def convection_u_old_lin(phi_grad_v_trial, J, J_lin, F_inv, F_inv_lin, u_old_disp, grad_v, rho):
    grad_v_Finv = ufl.dot(grad_v, F_inv)
    grad_v_Finv_lin = ufl.dot(grad_v, F_inv_lin)
    return rho * (
        J_lin * ufl.dot(grad_v_Finv, u_old_disp)
        + J * ufl.dot(grad_v_Finv_lin, u_old_disp)
        + J * ufl.dot(phi_grad_v_trial, ufl.dot(F_inv, u_old_disp))
    )


def stress_fluid_term1(dp, pI, J, F_inv_T, J_F_inv_T_LinU):
    return -J * ufl.dot(dp * ufl.Identity(2), F_inv_T) - ufl.dot(pI, J_F_inv_T_LinU)


def stress_fluid_term2(J_F_inv_T_LinU, stress_ALE, grad_v, grad_v_LinV, F_inv, F_inv_LinU, J, mu_f):
    sigma_LinV = ufl.dot(grad_v_LinV, F_inv) + ufl.dot(F_inv.T, grad_v_LinV.T)
    sigma_LinU = ufl.dot(grad_v, F_inv_LinU) + ufl.dot(F_inv_LinU.T, grad_v.T)
    return mu_f * (J * ufl.dot(sigma_LinV + sigma_LinU, F_inv.T) + ufl.dot(stress_ALE, J_F_inv_T_LinU))


def stress_fluid_term3(F_inv, F_inv_LinU, grad_v, grad_v_LinV, mu_f, J, J_F_inv_T_LinU):
    F_inv_T = F_inv.T
    return mu_f * (
        ufl.dot(ufl.dot(J_F_inv_T_LinU, grad_v.T), F_inv_T)
        + J * ufl.dot(ufl.dot(F_inv_T, grad_v_LinV.T), F_inv_T)
        + J * ufl.dot(ufl.dot(F_inv_T, grad_v.T), F_inv_LinU.T)
    )


def incompressibility_lin(grad_v, grad_v_trial, F, grad_dd_trial):
    return ufl.inner(cof2(F), grad_v_trial) + ufl.inner(cof2(grad_dd_trial), grad_v)


# --------------------------------------------------------------------------- #
# PyCutFEM term breakdown (mirrors examples/fsi_dealii_reference.py)          #
# --------------------------------------------------------------------------- #
def _measures_pc(pc: Dict):
    dx_f = dx(defined_on=pc["fluid_bs"], metadata={"q": pc["quad_order"]})
    dx_s = dx(defined_on=pc["solid_bs"], metadata={"q": pc["quad_order"]})
    dS_out = dS(defined_on=pc["outlet_bs"], metadata={"q": pc["quad_order"]})
    return dx_f, dx_s, dS_out


def build_pycutfem_terms(pc: Dict):
    dx_f, dx_s, dS_out = _measures_pc(pc)
    I = Identity(2)
    n = FacetNormal()
    grad_v = grad(pc["uk"])
    grad_d = grad(pc["dk"])
    grad_v_old = grad(pc["u_prev"])
    grad_d_old = grad(pc["d_prev"])

    F = ALE_Helpers.get_F(grad_d)
    Finv = ALE_Helpers.get_F_inv(F)
    J = ALE_Helpers.get_J(F)

    F_old = ALE_Helpers.get_F(grad_d_old)
    Finv_old = ALE_Helpers.get_F_inv(F_old)
    J_old = ALE_Helpers.get_J(F_old)
    pI = pc["pk"] * I

    # Residual terms
    acc_term = pc["rho_f"] * 0.5 * (J + J_old) * inner((pc["uk"] - pc["u_prev"]), pc["v"])
    convection_fluid = pc["rho_f"] * J * dot(dot(grad_v, Finv), pc["uk"])
    convection_fluid_with_u = pc["rho_f"] * J * dot(dot(grad_v, Finv), pc["dk"])
    convection_fluid_with_u_old = pc["rho_f"] * J * dot(dot(grad_v, Finv), pc["d_prev"])
    old_convection_fluid = pc["rho_f"] * J_old * dot(dot(grad_v_old, Finv_old), pc["u_prev"])
    fluid_pressure = -(J * dot(pI, Finv.T))
    sigma_ALE = NSE_ALE.get_stress_fluid_except_pressure_ALE(pc["mu_f"], grad_v, Finv)
    sigma_ALE_old = NSE_ALE.get_stress_fluid_except_pressure_ALE(pc["mu_f"], grad_v_old, Finv_old)
    stress_fluid_viscous = J * dot(sigma_ALE, Finv.T)
    stress_fluid_viscous_old = J_old * dot(sigma_ALE_old, Finv_old.T)
    sigma_ALE_tilde = pc["mu_f"] * dot(Finv.T, grad_v.T)
    sigma_ALE_tilde_old = pc["mu_f"] * dot(Finv_old.T, grad_v_old.T)
    stress_fluid_transpose = J * dot(sigma_ALE_tilde, Finv.T)
    stress_fluid_transpose_old = J_old * dot(sigma_ALE_tilde_old, Finv_old.T)
    incompressibility_fluid = NSE_ALE.get_Incompressibility_ALE(pc["uk"], F)
    solid_stress = Structure_Terms.get_Cauchy_stress(F, pc["mu_s"], pc["lambda_s"])
    solid_stress_old = Structure_Terms.get_Cauchy_stress(F_old, pc["mu_s"], pc["lambda_s"])
    solid_stress_transfomed = J * dot(solid_stress, Finv.T)
    solid_stress_transfomed_old = J_old * dot(solid_stress_old, Finv_old.T)

    res_terms = {
        "mass": acc_term * dx_f,
        "conv_theta": pc["dt"] * pc["theta"] * dot(convection_fluid, pc["v"]) * dx_f,
        "conv_old": pc["dt"] * (1.0 - pc["theta"]) * dot(old_convection_fluid, pc["v"]) * dx_f,
        "conv_mesh": -dot(convection_fluid_with_u - convection_fluid_with_u_old, pc["v"]) * dx_f,
        "pressure": pc["dt"] * inner(fluid_pressure, grad(pc["v"])) * dx_f,
        "stress_theta": pc["dt"] * pc["theta"] * inner(stress_fluid_viscous, grad(pc["v"])) * dx_f,
        "stress_old": pc["dt"] * (1.0 - pc["theta"]) * inner(stress_fluid_viscous_old, grad(pc["v"])) * dx_f,
        "biharmonic": (pc["alpha_u"] / J) * inner(grad(pc["dk"]), grad(pc["w"])) * dx_f,
        "incompressibility": incompressibility_fluid * pc["q"] * dx_f,
        "outlet_theta": -pc["dt"] * pc["theta"] * dot(dot(stress_fluid_transpose, n), pc["v"]) * dS_out,
        "outlet_old": -pc["dt"] * (1.0 - pc["theta"]) * dot(dot(stress_fluid_transpose_old, n), pc["v"]) * dS_out,
        "solid_mass": pc["rho_s"] * inner(pc["uk"] - pc["u_prev"], pc["v"]) * dx_s,
        "solid_stress_theta": pc["dt"] * pc["theta"] * inner(solid_stress_transfomed, grad(pc["v"])) * dx_s,
        "solid_stress_old": pc["dt"] * (1.0 - pc["theta"]) * inner(solid_stress_transfomed_old, grad(pc["v"])) * dx_s,
        "solid_disp": pc["rho_s"] * inner(pc["dk"] - pc["d_prev"], pc["w"]) * dx_s,
        "solid_mesh_theta": -pc["rho_s"] * pc["dt"] * pc["theta"] * inner(pc["uk"], pc["w"]) * dx_s,
        "solid_mesh_old": -pc["rho_s"] * pc["dt"] * (1.0 - pc["theta"]) * inner(pc["u_prev"], pc["w"]) * dx_s,
        "solid_pressure": pc["pk"] * pc["q"] * dx_s,
    }

    # Jacobian terms
    grad_dd = grad(pc["dd"])
    grad_du = grad(pc["du"])
    grad_v_trial = grad(pc["du"])
    J_LinU = ALE_Helpers.get_J_LinU(F, grad_dd)
    Finv_LinU = ALE_Helpers.get_F_inv_LinU(Finv, grad_dd)
    J_F_inv_T_LinU = ALE_Helpers.get_cof_F_LinU(F, Finv, grad_dd)
    pI_LinP_trial = pc["dp"] * I

    acc_term_jac = 0.5 * pc["rho_f"] * (J_LinU * (pc["uk"] - pc["u_prev"]) + (J + J_old) * pc["du"])

    grad_v_Finv = dot(grad_v, Finv)
    grad_v_Finv_lin = dot(grad_v, Finv_LinU)
    convection_fluid_v = pc["rho_f"] * (
        J_LinU * dot(grad_v_Finv, pc["uk"])
        + J * dot(grad_v_Finv_lin, pc["uk"])
        + J * dot(dot(grad_du, Finv), pc["uk"])
        + J * dot(grad_v_Finv, pc["du"])
    )
    convection_fluid_d = pc["rho_f"] * (
        J_LinU * dot(grad_v_Finv, pc["dk"])
        + J * dot(grad_v_Finv_lin, pc["dk"])
        + J * dot(grad_v_Finv, pc["dd"])
        + J * dot(dot(grad_du, Finv), pc["dk"])
    )
    convection_fluid_u_old = pc["rho_f"] * (
        J_LinU * dot(grad_v_Finv, pc["d_prev"])
        + J * dot(grad_v_Finv_lin, pc["d_prev"])
        + J * dot(dot(grad_du, Finv), pc["d_prev"])
    )

    stress_ALE_full = pc["mu_f"] * (dot(grad_v, Finv) + dot(Finv.T, grad_v.T))
    stress_fluid_term_1 = -J * dot(pI, Finv.T) - dot(pI_LinP_trial, J_F_inv_T_LinU)
    stress_fluid_term_2 = pc["mu_f"] * (
        J * dot(dot(grad_du, Finv), Finv.T)
        + J * dot(Finv.T, grad_du.T)
        + dot(stress_ALE_full, J_F_inv_T_LinU)
        + J * dot(dot(grad_v, Finv_LinU), Finv.T)
        + J * dot(Finv_LinU.T, grad_v.T)
    )
    neuman_term = pc["mu_f"] * (
        dot(dot(J_F_inv_T_LinU, grad_v.T), Finv.T)
        + J * dot(dot(Finv.T, grad_du.T), Finv.T)
        + J * dot(dot(Finv.T, grad_v.T), Finv_LinU.T)
    )
    incompressility_ALE_LinALl = inner(cof(F), grad_du) + inner(cof(grad_dd), grad_v)
    solid_stress_LinU = Structure_Terms.get_Piola_Kirchhoff_1st_LinAll(F, grad_dd, pc["mu_s"], pc["lambda_s"])

    jac_terms = {
        "jac_mass": dot(acc_term_jac, pc["v"]) * dx_f,
        "jac_conv": pc["dt"] * pc["theta"] * dot(convection_fluid_v, pc["v"]) * dx_f
        + -dot(convection_fluid_d, pc["v"]) * dx_f
        + dot(convection_fluid_u_old, pc["v"]) * dx_f,
        "jac_diffusion": pc["dt"] * inner(stress_fluid_term_1, grad(pc["v"])) * dx_f
        + pc["dt"] * pc["theta"] * inner(stress_fluid_term_2, grad(pc["v"])) * dx_f,
        "jac_biharmonic": (-pc["alpha_u"] / (J * J) * J_LinU * inner(grad(pc["dk"]), grad(pc["w"])) + pc["alpha_u"] / J * inner(grad(pc["dd"]), grad(pc["w"]))) * dx_f,
        "jac_incompressibility": incompressility_ALE_LinALl * pc["q"] * dx_f,
        "jac_outlet": -pc["dt"] * pc["theta"] * dot(dot(neuman_term, n), pc["v"]) * dS_out,
        "jac_solid": (
            pc["rho_s"] * dot(pc["du"], pc["v"])
            + pc["dt"] * pc["theta"] * inner(solid_stress_LinU, grad(pc["v"]))
            + pc["rho_s"] * dot(pc["dd"], pc["w"])
            - pc["dt"] * pc["theta"] * dot(pc["du"], pc["w"])
            + pc["dp"] * pc["q"]
        )
        * dx_s,
    }

    return res_terms, jac_terms


# --------------------------------------------------------------------------- #
# Problem setup                                                               #
# --------------------------------------------------------------------------- #
def setup_pycutfem(mesh_data, poly_order: int, dt_val: float, theta_val: float, quad_order: int, marks: Dict[str, int]):
    coords, quad_cells, quad_tags, line_cells, line_tags = mesh_data

    nodes = [Node(i, float(x), float(y), tag="") for i, (x, y) in enumerate(coords)]
    corners = quad_cells[:, ROW_MAJOR_Q2_CORNERS]
    mesh_pc = Mesh(nodes=nodes, element_connectivity=quad_cells, elements_corner_nodes=corners, element_type="quad", poly_order=poly_order)

    # Element subdomains: use Gmsh physical tags when present (matches FEniCSx).
    try:
        from pycutfem.utils.bitset import BitSet

        fluid_mask = quad_tags.astype(int) == int(marks["fluid"])
        solid_mask = quad_tags.astype(int) == int(marks["solid"])
        if not (np.any(fluid_mask) and np.any(solid_mask)):
            fluid_bs, solid_bs = classify_fluid_solid(mesh_pc)
        else:
            fluid_bs = BitSet(fluid_mask)
            solid_bs = BitSet(solid_mask)
            for el, tag in zip(mesh_pc.elements_list, quad_tags):
                tag_i = int(tag)
                if tag_i == int(marks["fluid"]):
                    el.tag = "fluid"
                elif tag_i == int(marks["solid"]):
                    el.tag = "solid"
                else:
                    el.tag = str(tag_i)
    except Exception:
        fluid_bs, solid_bs = classify_fluid_solid(mesh_pc)

    # Boundary facets: transfer Gmsh physical line tags by node-id match.
    # PyCutFEM may split a quadratic edge into multiple sub-edges between consecutive nodes.
    edge_by_nodes: dict[tuple[int, int], object] = {}
    for e in mesh_pc.edges_list:
        if e.right is not None:
            continue
        edge_by_nodes[tuple(sorted((int(e.nodes[0]), int(e.nodes[1]))))] = e

    missed = 0
    for cell_nodes, tag in zip(line_cells, line_tags):
        tag_name = PHYS_TAGS.get(int(tag), str(int(tag)))
        nn = int(cell_nodes.shape[0])
        if nn < 2:
            continue
        if nn == 2:
            segments = [(int(cell_nodes[0]), int(cell_nodes[1]))]
        else:
            # Gmsh line3 ordering: (start, end, mid). Tag both sub-edges.
            segments = [
                (int(cell_nodes[0]), int(cell_nodes[2])),
                (int(cell_nodes[2]), int(cell_nodes[1])),
            ]
        for a, b in segments:
            key = tuple(sorted((a, b)))
            edge = edge_by_nodes.get(key)
            if edge is None:
                missed += 1
                continue
            if getattr(edge, "tag", "") and edge.tag != tag_name:
                raise RuntimeError(f"Boundary edge {key} mapped to multiple tags: {edge.tag} vs {tag_name}")
            edge.tag = tag_name

    outlet_bs = mesh_pc.edge_bitset("outlet")
    if outlet_bs.cardinality() == 0:
        # Fallback to geometric tagging if line-tag transfer failed.
        ensure_boundary_tags(mesh_pc, force_geometric=True, tol=1e-6)
        outlet_bs = mesh_pc.edge_bitset("outlet")
    if outlet_bs.cardinality() == 0:
        raise RuntimeError("Outlet boundary is empty on PyCutFEM mesh.")

    element = MixedElement(
        mesh_pc,
        field_specs={"ux": poly_order, "uy": poly_order, "dx": poly_order, "dy": poly_order, "p": poly_order - 1},
    )
    dh = DofHandler(element, method="cg")
    dh.tag_dofs_from_element_bitset("inactive", "p", solid_bs, strict=True)

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

    rho_f = Constant(1.0e3)
    nu_f = Constant(1.0e-3)
    mu_f = rho_f * nu_f
    rho_s = Constant(1.0e3)
    mu_s = Constant(0.5e6)
    nu_s = 0.4
    E_s = 2.0 * float(mu_s.value) * (1.0 + nu_s)
    lambda_s = Constant(E_s * nu_s / ((1.0 + nu_s) * (1.0 - 2.0 * nu_s)))
    alpha_u = Constant(1.0e-8)
    stab_eps = Constant(1.0e-8)
    dt_c = Constant(dt_val)
    theta_c = Constant(theta_val)

    pc = {
        "mesh": mesh_pc,
        "fluid_bs": fluid_bs,
        "solid_bs": solid_bs,
        "outlet_bs": outlet_bs,
        "du": du,
        "dd": dd,
        "dp": dp,
        "v": v,
        "w": w,
        "q": q,
        "uk": uk,
        "u_prev": u_prev,
        "dk": dk,
        "d_prev": d_prev,
        "pk": pk,
        "p_prev": p_prev,
        "rho_f": rho_f,
        "mu_f": mu_f,
        "rho_s": rho_s,
        "lambda_s": lambda_s,
        "mu_s": mu_s,
        "alpha_u": alpha_u,
        "stab_eps": stab_eps,
        "dt": dt_c,
        "theta": theta_c,
        "quad_order": quad_order,
        "dof_handler": dh,
    }
    return pc


def setup_fenics(mesh_data, poly_order: int, dt_val: float, theta_val: float, quad_order: int, marks: Dict[str, int]):
    coords, quad_cells, quad_tags, line_cells, line_tags = mesh_data

    if quad_cells.shape[1] != ROW_MAJOR_TO_BASIX_Q2.size:
        raise ValueError(f"Expected 9-node quadrilateral connectivity, got {quad_cells.shape[1]}.")
    geom_el = basix.ufl.element("Lagrange", "quadrilateral", poly_order, shape=(2,))
    quad_cells_fx = np.ascontiguousarray(quad_cells[:, ROW_MAJOR_TO_BASIX_Q2], dtype=np.int64)
    mesh_fx = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, quad_cells_fx, geom_el, coords)

    tdim = mesh_fx.topology.dim
    fdim = tdim - 1

    # ------------------------------------------------------------------ #
    # Cell tags (Gmsh physical regions)                                   #
    # ------------------------------------------------------------------ #
    # IMPORTANT: dolfinx.create_mesh may reorder cells internally, so we
    # cannot assume that the input `quad_tags[i]` corresponds to cell `i`.
    # We therefore match cells by their (unordered) vertex coordinates.
    mesh_fx.topology.create_connectivity(tdim, 0)
    mesh_fx.topology.create_connectivity(0, tdim)

    conn_c2v = mesh_fx.topology.connectivity(tdim, 0)
    c2v = conn_c2v.array
    c2v_offsets = conn_c2v.offsets
    n_cells = mesh_fx.topology.index_map(tdim).size_local
    cell_ids = np.arange(n_cells, dtype=np.int32)

    n_vertices = mesh_fx.topology.index_map(0).size_local
    vertices = np.arange(n_vertices, dtype=np.int32)
    v2g = dolfinx.mesh.entities_to_geometry(mesh_fx, 0, vertices).reshape(-1)
    v_coords = mesh_fx.geometry.x[v2g, :2]

    def _cell_key(pts: np.ndarray, *, decimals: int = 12) -> tuple[tuple[float, float], ...]:
        pts_r = np.round(np.asarray(pts, dtype=float), decimals)
        pts_r = pts_r[np.lexsort((pts_r[:, 1], pts_r[:, 0]))]
        return tuple((float(x), float(y)) for x, y in pts_r)

    # Input (meshio) cells: use vertex coordinates from the Q2 row-major layout.
    corner_idx = np.array([0, 2, 6, 8], dtype=int)  # (bl, br, tl, tr) as a set
    in_map: dict[tuple[tuple[float, float], ...], int] = {}
    for i in range(quad_cells.shape[0]):
        key = _cell_key(coords[quad_cells[i, corner_idx], :2])
        if key in in_map:
            raise RuntimeError("Duplicate cell-vertex signature encountered while mapping cell tags.")
        in_map[key] = int(quad_tags[i])

    values = np.empty(n_cells, dtype=np.int32)
    missed = 0
    for c in range(n_cells):
        vv = c2v[c2v_offsets[c] : c2v_offsets[c + 1]]
        if vv.size != 4:
            raise RuntimeError(f"Expected 4 vertices for quad cell, got {vv.size} (cell {c}).")
        key = _cell_key(v_coords[vv])
        tag = in_map.get(key)
        if tag is None:
            missed += 1
            values[c] = -1
        else:
            values[c] = int(tag)
    if missed:
        raise RuntimeError(f"Failed to map {missed} cells to Gmsh physical tags (cell tag transfer).")

    cell_tags = dolfinx.mesh.meshtags(mesh_fx, tdim, cell_ids, values)

    # Map Gmsh physical line tags to **boundary** facets in dolfinx.
    # IMPORTANT: vertex indices are topological; convert to geometry coordinates via entities_to_geometry.
    mesh_fx.topology.create_connectivity(fdim, 0)
    mesh_fx.topology.create_connectivity(fdim, tdim)

    conn_f2v = mesh_fx.topology.connectivity(fdim, 0)
    f2v = conn_f2v.array
    f2v_offsets = conn_f2v.offsets

    conn_f2c = mesh_fx.topology.connectivity(fdim, tdim)
    f2c_offsets = conn_f2c.offsets
    boundary_facets = np.flatnonzero((f2c_offsets[1:] - f2c_offsets[:-1]) == 1).astype(np.int32)

    # Vertex entity -> geometry coordinate
    n_vertices = mesh_fx.topology.index_map(0).size_local
    verts = np.arange(n_vertices, dtype=np.int32)
    v2g = dolfinx.mesh.entities_to_geometry(mesh_fx, 0, verts).reshape(-1)
    v_coords = mesh_fx.geometry.x[v2g, :2]

    # Some Gmsh boundary curves are subdivided into multiple line3 elements that
    # connect (vertex -> mid-edge node) segments. To robustly tag facets, we map
    # both full (vertex, vertex) and half (vertex, midpoint) segments to the
    # corresponding dolfinx facet.
    facet_midpoints = dolfinx.mesh.compute_midpoints(mesh_fx, fdim, boundary_facets)[:, :2]

    facet_by_key: dict[tuple[tuple[float, float], tuple[float, float]], int] = {}
    for facet, pm in zip(boundary_facets, facet_midpoints):
        vv = f2v[f2v_offsets[facet] : f2v_offsets[facet + 1]]
        if vv.size != 2:
            continue
        p0, p1 = v_coords[int(vv[0])], v_coords[int(vv[1])]
        facet_by_key[_edge_key_from_points(p0, p1)] = int(facet)
        facet_by_key[_edge_key_from_points(p0, pm)] = int(facet)
        facet_by_key[_edge_key_from_points(p1, pm)] = int(facet)

    facet_to_tag: dict[int, int] = {}
    missed = 0
    for cell_nodes, tag in zip(line_cells, line_tags):
        # Gmsh line3 ordering: (start, end, mid). Use endpoints for exact matching to topological vertices.
        if int(cell_nodes.shape[0]) < 2:
            continue
        p0 = coords[int(cell_nodes[0])]
        p1 = coords[int(cell_nodes[1])]
        facet = facet_by_key.get(_edge_key_from_points(p0, p1))
        if facet is None:
            missed += 1
            continue
        tag_int = int(tag)
        if facet in facet_to_tag and facet_to_tag[facet] != tag_int:
            raise RuntimeError(f"Facet {facet} mapped to multiple tags: {facet_to_tag[facet]} vs {tag_int}")
        facet_to_tag[facet] = tag_int

    if not facet_to_tag:
        raise RuntimeError("Failed to map line tags to boundary facets.")
    facet_indices = np.asarray(sorted(facet_to_tag.keys()), dtype=np.int32)
    facet_values = np.asarray([facet_to_tag[int(i)] for i in facet_indices], dtype=np.int32)
    facet_tags = dolfinx.mesh.meshtags(
        mesh_fx,
        fdim,
        np.asarray(facet_indices, dtype=np.int32),
        np.asarray(facet_values, dtype=np.int32),
    )

    dx_sub = ufl.Measure("dx", domain=mesh_fx, subdomain_data=cell_tags, metadata={"quadrature_degree": quad_order})
    ds_sub = ufl.Measure("ds", domain=mesh_fx, subdomain_data=facet_tags, metadata={"quadrature_degree": quad_order})

    V_vec = basix.ufl.element("Lagrange", mesh_fx.ufl_cell().cellname(), poly_order, shape=(mesh_fx.geometry.dim,))
    Vp = basix.ufl.element("Lagrange", mesh_fx.ufl_cell().cellname(), poly_order - 1)
    W_el = mixed_element([V_vec, V_vec, Vp])
    W = dolfinx.fem.functionspace(mesh_fx, W_el)

    trial = ufl.TrialFunction(W)
    test = ufl.TestFunction(W)
    du_fx, dd_fx, dp_fx = ufl.split(trial)
    v_fx, w_fx, q_fx = ufl.split(test)

    state_k = dolfinx.fem.Function(W, name="state_k")
    state_prev = dolfinx.fem.Function(W, name="state_prev")
    u_k_fx, d_k_fx, p_k_fx = ufl.split(state_k)
    u_prev_fx, d_prev_fx, p_prev_fx = ufl.split(state_prev)

    rho_f = dolfinx.fem.Constant(mesh_fx, 1.0e3)
    nu_f = dolfinx.fem.Constant(mesh_fx, 1.0e-3)
    mu_f = dolfinx.fem.Constant(mesh_fx, float(rho_f.value) * float(nu_f.value))
    rho_s = dolfinx.fem.Constant(mesh_fx, 1.0e3)
    mu_s = dolfinx.fem.Constant(mesh_fx, 0.5e6)
    nu_s = 0.4
    E_s = 2.0 * float(mu_s.value) * (1.0 + nu_s)
    lambda_s = dolfinx.fem.Constant(mesh_fx, E_s * nu_s / ((1.0 + nu_s) * (1.0 - 2.0 * nu_s)))
    alpha_u = dolfinx.fem.Constant(mesh_fx, 1.0e-8)
    stab_eps = dolfinx.fem.Constant(mesh_fx, 1.0e-8)
    dt_c = dolfinx.fem.Constant(mesh_fx, dt_val)
    theta_c = dolfinx.fem.Constant(mesh_fx, theta_val)

    fx = {
        "mesh": mesh_fx,
        "cell_tags": cell_tags,
        "facet_tags": facet_tags,
        "dx": dx_sub,
        "ds": ds_sub,
        "marks": marks,
        "W": W,
        "trial": (du_fx, dd_fx, dp_fx),
        "test": (v_fx, w_fx, q_fx),
        "state_k": state_k,
        "state_prev": state_prev,
        "u_k": u_k_fx,
        "d_k": d_k_fx,
        "p_k": p_k_fx,
        "u_prev": u_prev_fx,
        "d_prev": d_prev_fx,
        "p_prev": p_prev_fx,
        "rho_f": rho_f,
        "mu_f": mu_f,
        "rho_s": rho_s,
        "lambda_s": lambda_s,
        "mu_s": mu_s,
        "alpha_u": alpha_u,
        "stab_eps": stab_eps,
        "dt": dt_c,
        "theta": theta_c,
        "quad_order": quad_order,
    }
    return fx


# --------------------------------------------------------------------------- #
# FEniCSx forms                                                               #
# --------------------------------------------------------------------------- #
def build_fenics_forms(fx: Dict):
    marks = fx["marks"]
    dx_f = fx["dx"](marks["fluid"])
    dx_s = fx["dx"](marks["solid"])
    ds_out = fx["ds"](marks["outlet"])

    du_fx, dd_fx, dp_fx = fx["trial"]
    v_fx, w_fx, q_fx = fx["test"]
    uk_fx, u_prev_fx = fx["u_k"], fx["u_prev"]
    dk_fx, d_prev_fx = fx["d_k"], fx["d_prev"]
    pk_fx, p_prev_fx = fx["p_k"], fx["p_prev"]

    rho_f = fx["rho_f"]
    mu_f = fx["mu_f"]
    rho_s = fx["rho_s"]
    mu_s = fx["mu_s"]
    lambda_s = fx["lambda_s"]
    alpha_u = fx["alpha_u"]
    dt = fx["dt"]
    theta = fx["theta"]

    I = ufl.Identity(2)
    n = ufl.FacetNormal(fx["mesh"])

    F = I + ufl.grad(dk_fx)
    F_prev = I + ufl.grad(d_prev_fx)
    J = ufl.det(F)
    J_prev = ufl.det(F_prev)
    Finv = ufl.inv(F)
    Finv_prev = ufl.inv(F_prev)
    pI = pk_fx * I

    grad_uk = ufl.grad(uk_fx)
    grad_u_prev = ufl.grad(u_prev_fx)

    acc = rho_f * 0.5 * (J + J_prev) * ufl.dot(uk_fx - u_prev_fx, v_fx)

    conv_new = dt * theta * ufl.dot(rho_f * J * ufl.dot(ufl.dot(grad_uk, Finv), uk_fx), v_fx)
    conv_old = dt * (1.0 - theta) * ufl.dot(rho_f * J_prev * ufl.dot(ufl.dot(grad_u_prev, Finv_prev), u_prev_fx), v_fx)
    conv_mesh = -ufl.dot(rho_f * J * ufl.dot(ufl.dot(grad_uk, Finv), dk_fx - d_prev_fx), v_fx)

    fluid_pressure = -(J * ufl.dot(pI, Finv.T))
    pressure_term = dt * ufl.inner(fluid_pressure, ufl.grad(v_fx))

    sigma_ALE = mu_f * (ufl.dot(grad_uk, Finv) + ufl.dot(Finv.T, grad_uk.T))
    sigma_ALE_prev = mu_f * (ufl.dot(grad_u_prev, Finv_prev) + ufl.dot(Finv_prev.T, grad_u_prev.T))
    stress_term = dt * theta * ufl.inner(J * ufl.dot(sigma_ALE, Finv.T), ufl.grad(v_fx))
    stress_term += dt * (1.0 - theta) * ufl.inner(J_prev * ufl.dot(sigma_ALE_prev, Finv_prev.T), ufl.grad(v_fx))

    biharmonic_term = (alpha_u / J) * ufl.inner(ufl.grad(dk_fx), ufl.grad(w_fx))
    incompressibility_term = ufl.inner(cof2(F), ufl.grad(uk_fx)) * q_fx

    res_fluid = (acc + conv_new + conv_old + conv_mesh + pressure_term + stress_term + biharmonic_term + incompressibility_term) * dx_f

    sigma_ALE_tilde = mu_f * ufl.dot(Finv.T, grad_uk.T)
    sigma_ALE_tilde_prev = mu_f * ufl.dot(Finv_prev.T, grad_u_prev.T)
    neuman_flux = ufl.dot(J * ufl.dot(sigma_ALE_tilde, Finv.T), n)
    neuman_flux_prev = ufl.dot(J_prev * ufl.dot(sigma_ALE_tilde_prev, Finv_prev.T), n)
    res_outlet = (-dt * theta * ufl.dot(neuman_flux, v_fx) - dt * (1.0 - theta) * ufl.dot(neuman_flux_prev, v_fx)) * ds_out

    solid_stress = structure_cauchy(F, mu_s, lambda_s)
    solid_stress_prev = structure_cauchy(F_prev, mu_s, lambda_s)
    res_solid = (
        rho_s * ufl.dot(uk_fx - u_prev_fx, v_fx)
        + dt * theta * ufl.inner(J * ufl.dot(solid_stress, Finv.T), ufl.grad(v_fx))
        + dt * (1.0 - theta) * ufl.inner(J_prev * ufl.dot(solid_stress_prev, Finv_prev.T), ufl.grad(v_fx))
        + rho_s * ufl.dot(dk_fx - d_prev_fx, w_fx)
        - rho_s * dt * theta * ufl.dot(uk_fx, w_fx)
        - rho_s * dt * (1.0 - theta) * ufl.dot(u_prev_fx, w_fx)
        + pk_fx * q_fx
    ) * dx_s

    grad_dd = ufl.grad(dd_fx)
    J_lin = ufl.inner(cof2(F), grad_dd)
    Finv_lin = -ufl.dot(Finv, ufl.dot(grad_dd, Finv))
    J_F_inv_T_lin = cof2(grad_dd)

    acc_jac = acceleration_term(J, J_prev, J_lin, uk_fx, u_prev_fx, du_fx, rho_f)

    stress_ALE = -pI + mu_f * (ufl.dot(grad_uk, Finv) + ufl.dot(Finv.T, grad_uk.T))
    convection_v = convection_lin(ufl.grad(du_fx), du_fx, J, J_lin, Finv, Finv_lin, uk_fx, grad_uk, rho_f)
    convection_d = convection_u_lin(ufl.grad(du_fx), dd_fx, J, J_lin, Finv, Finv_lin, dk_fx, grad_uk, rho_f)
    convection_old = convection_u_old_lin(ufl.grad(du_fx), J, J_lin, Finv, Finv_lin, d_prev_fx, grad_uk, rho_f)

    stress_term1 = stress_fluid_term1(dp_fx, pI, J, Finv.T, J_F_inv_T_lin)
    stress_term2 = stress_fluid_term2(J_F_inv_T_lin, stress_ALE, grad_uk, ufl.grad(du_fx), Finv, Finv_lin, J, mu_f)

    jac_bih = (-alpha_u / (J * J) * J_lin * ufl.inner(ufl.grad(dk_fx), ufl.grad(w_fx)) + alpha_u / J * ufl.inner(ufl.grad(dd_fx), ufl.grad(w_fx)))
    jac_incompress = incompressibility_lin(grad_uk, ufl.grad(du_fx), F, grad_dd) * q_fx

    jac_fluid = (
        ufl.dot(acc_jac, v_fx)
        + dt * theta * ufl.dot(convection_v, v_fx)
        - ufl.dot(convection_d, v_fx)
        + ufl.dot(convection_old, v_fx)
        + dt * ufl.inner(stress_term1, ufl.grad(v_fx))
        + dt * theta * ufl.inner(stress_term2, ufl.grad(v_fx))
        + jac_bih
        + jac_incompress
    ) * dx_f

    neuman_term = stress_fluid_term3(Finv, Finv_lin, grad_uk, ufl.grad(du_fx), mu_f, J, J_F_inv_T_lin)
    jac_outlet = -dt * theta * ufl.dot(ufl.dot(neuman_term, n), v_fx) * ds_out

    solid_tangent = structure_tangent(F, grad_dd, mu_s, lambda_s)
    jac_solid = (
        rho_s * ufl.dot(du_fx, v_fx)
        + dt * theta * ufl.inner(solid_tangent, ufl.grad(v_fx))
        + rho_s * ufl.dot(dd_fx, w_fx)
        - dt * theta * ufl.dot(du_fx, w_fx)
        + dp_fx * q_fx
    ) * dx_s

    res_form = res_fluid + res_outlet + res_solid
    jac_form = jac_fluid + jac_outlet + jac_solid
    return res_form, jac_form


def build_fenics_terms(fx: Dict):
    marks = fx["marks"]
    dx_f = fx["dx"](marks["fluid"])
    dx_s = fx["dx"](marks["solid"])
    ds_out = fx["ds"](marks["outlet"])

    du_fx, dd_fx, dp_fx = fx["trial"]
    v_fx, w_fx, q_fx = fx["test"]
    uk_fx, u_prev_fx = fx["u_k"], fx["u_prev"]
    dk_fx, d_prev_fx = fx["d_k"], fx["d_prev"]
    pk_fx, p_prev_fx = fx["p_k"], fx["p_prev"]

    rho_f = fx["rho_f"]
    mu_f = fx["mu_f"]
    rho_s = fx["rho_s"]
    mu_s = fx["mu_s"]
    lambda_s = fx["lambda_s"]
    alpha_u = fx["alpha_u"]
    dt = fx["dt"]
    theta = fx["theta"]

    I = ufl.Identity(2)
    n = ufl.FacetNormal(fx["mesh"])

    F = I + ufl.grad(dk_fx)
    F_prev = I + ufl.grad(d_prev_fx)
    J = ufl.det(F)
    J_prev = ufl.det(F_prev)
    Finv = ufl.inv(F)
    Finv_prev = ufl.inv(F_prev)
    pI = pk_fx * I

    grad_uk = ufl.grad(uk_fx)
    grad_u_prev = ufl.grad(u_prev_fx)

    acc = rho_f * 0.5 * (J + J_prev) * ufl.dot(uk_fx - u_prev_fx, v_fx)

    convection_fluid = rho_f * J * ufl.dot(ufl.dot(grad_uk, Finv), uk_fx)
    convection_mesh = rho_f * J * ufl.dot(ufl.dot(grad_uk, Finv), dk_fx)
    convection_mesh_old = rho_f * J * ufl.dot(ufl.dot(grad_uk, Finv), d_prev_fx)
    conv_new = dt * theta * ufl.dot(convection_fluid, v_fx)
    conv_old = dt * (1.0 - theta) * ufl.dot(rho_f * J_prev * ufl.dot(ufl.dot(grad_u_prev, Finv_prev), u_prev_fx), v_fx)
    conv_mesh = -ufl.dot(convection_mesh - convection_mesh_old, v_fx)

    fluid_pressure = -(J * ufl.dot(pI, Finv.T))
    pressure_term = dt * ufl.inner(fluid_pressure, ufl.grad(v_fx))

    sigma_ALE = mu_f * (ufl.dot(grad_uk, Finv) + ufl.dot(Finv.T, grad_uk.T))
    sigma_ALE_prev = mu_f * (ufl.dot(grad_u_prev, Finv_prev) + ufl.dot(Finv_prev.T, grad_u_prev.T))
    stress_theta = dt * theta * ufl.inner(J * ufl.dot(sigma_ALE, Finv.T), ufl.grad(v_fx))
    stress_old = dt * (1.0 - theta) * ufl.inner(J_prev * ufl.dot(sigma_ALE_prev, Finv_prev.T), ufl.grad(v_fx))

    biharmonic_term = (alpha_u / J) * ufl.inner(ufl.grad(dk_fx), ufl.grad(w_fx))
    incompressibility_term = ufl.inner(cof2(F), ufl.grad(uk_fx)) * q_fx

    sigma_ALE_tilde = mu_f * ufl.dot(Finv.T, grad_uk.T)
    sigma_ALE_tilde_prev = mu_f * ufl.dot(Finv_prev.T, grad_u_prev.T)
    stress_out = J * ufl.dot(sigma_ALE_tilde, Finv.T)
    stress_out_prev = J_prev * ufl.dot(sigma_ALE_tilde_prev, Finv_prev.T)

    solid_stress = structure_cauchy(F, mu_s, lambda_s)
    solid_stress_prev = structure_cauchy(F_prev, mu_s, lambda_s)

    res_terms = {
        "mass": acc * dx_f,
        "conv_theta": conv_new * dx_f,
        "conv_old": conv_old * dx_f,
        "conv_mesh": conv_mesh * dx_f,
        "pressure": pressure_term * dx_f,
        "stress_theta": stress_theta * dx_f,
        "stress_old": stress_old * dx_f,
        "biharmonic": biharmonic_term * dx_f,
        "incompressibility": incompressibility_term * dx_f,
        "outlet_theta": -dt * theta * ufl.dot(ufl.dot(stress_out, n), v_fx) * ds_out,
        "outlet_old": -dt * (1.0 - theta) * ufl.dot(ufl.dot(stress_out_prev, n), v_fx) * ds_out,
        "solid_mass": rho_s * ufl.dot(uk_fx - u_prev_fx, v_fx) * dx_s,
        "solid_stress_theta": dt * theta * ufl.inner(J * ufl.dot(solid_stress, Finv.T), ufl.grad(v_fx)) * dx_s,
        "solid_stress_old": dt * (1.0 - theta) * ufl.inner(J_prev * ufl.dot(solid_stress_prev, Finv_prev.T), ufl.grad(v_fx)) * dx_s,
        "solid_disp": rho_s * ufl.dot(dk_fx - d_prev_fx, w_fx) * dx_s,
        "solid_mesh_theta": -rho_s * dt * theta * ufl.dot(uk_fx, w_fx) * dx_s,
        "solid_mesh_old": -rho_s * dt * (1.0 - theta) * ufl.dot(u_prev_fx, w_fx) * dx_s,
        "solid_pressure": pk_fx * q_fx * dx_s,
    }

    grad_dd = ufl.grad(dd_fx)
    J_lin = ufl.inner(cof2(F), grad_dd)
    Finv_lin = -ufl.dot(Finv, ufl.dot(grad_dd, Finv))
    J_F_inv_T_lin = cof2(grad_dd)

    acc_jac = acceleration_term(J, J_prev, J_lin, uk_fx, u_prev_fx, du_fx, rho_f)
    stress_ALE = -pI + mu_f * (ufl.dot(grad_uk, Finv) + ufl.dot(Finv.T, grad_uk.T))
    convection_v = convection_lin(ufl.grad(du_fx), du_fx, J, J_lin, Finv, Finv_lin, uk_fx, grad_uk, rho_f)
    convection_d = convection_u_lin(ufl.grad(du_fx), dd_fx, J, J_lin, Finv, Finv_lin, dk_fx, grad_uk, rho_f)
    convection_old = convection_u_old_lin(ufl.grad(du_fx), J, J_lin, Finv, Finv_lin, d_prev_fx, grad_uk, rho_f)
    stress_term1 = stress_fluid_term1(dp_fx, pI, J, Finv.T, J_F_inv_T_lin)
    stress_term2 = stress_fluid_term2(J_F_inv_T_lin, stress_ALE, grad_uk, ufl.grad(du_fx), Finv, Finv_lin, J, mu_f)
    neuman_term = stress_fluid_term3(Finv, Finv_lin, grad_uk, ufl.grad(du_fx), mu_f, J, J_F_inv_T_lin)
    jac_bih = (-alpha_u / (J * J) * J_lin * ufl.inner(ufl.grad(dk_fx), ufl.grad(w_fx)) + alpha_u / J * ufl.inner(ufl.grad(dd_fx), ufl.grad(w_fx)))
    jac_incompress = incompressibility_lin(grad_uk, ufl.grad(du_fx), F, grad_dd) * q_fx
    solid_tangent = structure_tangent(F, grad_dd, mu_s, lambda_s)

    jac_terms = {
        "jac_mass": ufl.dot(acc_jac, v_fx) * dx_f,
        "jac_conv": (
            dt * theta * ufl.dot(convection_v, v_fx)
            - ufl.dot(convection_d, v_fx)
            + ufl.dot(convection_old, v_fx)
        )
        * dx_f,
        "jac_diffusion": dt * ufl.inner(stress_term1, ufl.grad(v_fx)) * dx_f
        + dt * theta * ufl.inner(stress_term2, ufl.grad(v_fx)) * dx_f,
        "jac_biharmonic": jac_bih * dx_f,
        "jac_incompressibility": jac_incompress * dx_f,
        "jac_outlet": -dt * theta * ufl.dot(ufl.dot(neuman_term, n), v_fx) * ds_out,
        "jac_solid": (
            rho_s * ufl.dot(du_fx, v_fx)
            + dt * theta * ufl.inner(solid_tangent, ufl.grad(v_fx))
            + rho_s * ufl.dot(dd_fx, w_fx)
            - dt * theta * ufl.dot(du_fx, w_fx)
            + dp_fx * q_fx
        )
        * dx_s,
    }
    return res_terms, jac_terms


# --------------------------------------------------------------------------- #
# Assembly + comparison                                                       #
# --------------------------------------------------------------------------- #
def assemble_pycutfem(pc: Dict, res_form, jac_form, backend: str):
    eq_res = Equation(None, res_form)
    eq_jac = Equation(jac_form, None)
    K_pc, _ = assemble_form(eq_jac, dof_handler=pc["dof_handler"], quad_degree=pc["quad_order"], backend=backend)
    _, R_pc = assemble_form(eq_res, dof_handler=pc["dof_handler"], quad_degree=pc["quad_order"], backend=backend)
    return K_pc, R_pc


def assemble_pycutfem_terms(pc: Dict, backend: str):
    res_terms, jac_terms = build_pycutfem_terms(pc)
    res = {}
    jac = {}
    for name, form in res_terms.items():
        _, vec = assemble_form(Equation(None, form), dof_handler=pc["dof_handler"], quad_degree=pc["quad_order"], backend=backend)
        res[name] = vec.flatten()
    for name, form in jac_terms.items():
        mat, _ = assemble_form(Equation(form, None), dof_handler=pc["dof_handler"], quad_degree=pc["quad_order"], backend=backend)
        jac[name] = mat.tocsr()
    return res, jac


def assemble_fenics(res_form, jac_form):
    J_compiled = dolfinx.fem.form(jac_form)
    R_compiled = dolfinx.fem.form(res_form)
    A = dolfinx.fem.petsc.assemble_matrix(J_compiled)
    A.assemble()
    vec = dolfinx.fem.petsc.assemble_vector(R_compiled)
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    indptr, indices, data = A.getValuesCSR()
    return csr_matrix((data, indices, indptr), shape=A.getSize()), vec.array


def assemble_fenics_terms(res_terms, jac_terms):
    res = {}
    jac = {}
    for name, form in res_terms.items():
        vec = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(form))
        vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        res[name] = vec.array
    for name, form in jac_terms.items():
        A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(form))
        A.assemble()
        indptr, indices, data = A.getValuesCSR()
        jac[name] = csr_matrix((data, indices, indptr), shape=A.getSize())
    return res, jac


def compare(pc: Dict, fx: Dict, backend: str, rtol: float, atol: float, dump: bool):
    P_map = create_true_dof_map(pc["dof_handler"], fx["W"])

    res_form_pc = build_residual_pc(
        uk=pc["uk"],
        u_prev=pc["u_prev"],
        dk=pc["dk"],
        d_prev=pc["d_prev"],
        pk=pc["pk"],
        p_prev=pc["p_prev"],
        v_test=pc["v"],
        w_test=pc["w"],
        q_test=pc["q"],
        dt=pc["dt"],
        theta=pc["theta"],
        rho_f=pc["rho_f"],
        mu_f=pc["mu_f"],
        rho_s=pc["rho_s"],
        lambda_s=pc["lambda_s"],
        mu_s=pc["mu_s"],
        alpha_u=pc["alpha_u"],
        stab_eps=pc["stab_eps"],
        fluid_bs=pc["fluid_bs"],
        solid_bs=pc["solid_bs"],
        outlet_bs=pc["outlet_bs"],
        quad_order=pc["quad_order"],
    )
    jac_form_pc = build_jac_pc(
        uk=pc["uk"],
        u_prev=pc["u_prev"],
        dk=pc["dk"],
        d_prev=pc["d_prev"],
        pk=pc["pk"],
        p_prev=pc["p_prev"],
        du=pc["du"],
        dd=pc["dd"],
        dp=pc["dp"],
        test_v=pc["v"],
        test_w=pc["w"],
        test_q=pc["q"],
        timestep=pc["dt"],
        theta=pc["theta"],
        rho_f=pc["rho_f"],
        mu_f=pc["mu_f"],
        rho_s=pc["rho_s"],
        lambda_s=pc["lambda_s"],
        mu_s=pc["mu_s"],
        alpha_u=pc["alpha_u"],
        stab_eps=pc["stab_eps"],
        fluid_bs=pc["fluid_bs"],
        solid_bs=pc["solid_bs"],
        outlet_bs=pc["outlet_bs"],
        quad_order=pc["quad_order"],
    )

    K_pc, R_pc = assemble_pycutfem(pc, res_form_pc, jac_form_pc, backend)
    res_fx, jac_fx = build_fenics_forms(fx)
    K_fx, R_fx = assemble_fenics(res_fx, jac_fx)

    K_pc_csr = K_pc.tocsr()
    K_fx_perm = K_fx[P_map, :][:, P_map].tocsr()
    r_pc = R_pc.flatten()
    r_fx = R_fx[P_map]

    diff_jac = (K_pc_csr - K_fx_perm)
    max_jac_diff = float(np.max(np.abs(diff_jac.data))) if diff_jac.nnz else 0.0
    jac_scale = 0.0
    if K_pc_csr.nnz:
        jac_scale = max(jac_scale, float(np.max(np.abs(K_pc_csr.data))))
    if K_fx_perm.nnz:
        jac_scale = max(jac_scale, float(np.max(np.abs(K_fx_perm.data))))
    jac_tol = atol + rtol * jac_scale
    jac_ok = max_jac_diff <= jac_tol

    res_diff = r_pc - r_fx
    max_res_diff = float(np.max(np.abs(res_diff))) if res_diff.size else 0.0
    res_scale = 0.0
    if r_pc.size:
        res_scale = max(res_scale, float(np.max(np.abs(r_pc))))
    if r_fx.size:
        res_scale = max(res_scale, float(np.max(np.abs(r_fx))))
    res_tol = atol + rtol * res_scale
    res_ok = max_res_diff <= res_tol

    if dump:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        from scipy.sparse import save_npz

        save_npz(OUTPUT_DIR / "jac_pc_sparse.npz", K_pc_csr)
        save_npz(OUTPUT_DIR / "jac_fx_sparse.npz", K_fx_perm)
        np.savez_compressed(OUTPUT_DIR / "residuals.npz", res_pc=r_pc, res_fx=r_fx, P_map=P_map)
        print(f"Saved sparse matrices and residuals to {OUTPUT_DIR}")

    print(f"Shapes -> Jacobian: {K_pc_csr.shape}, Residual: {r_pc.shape}")
    print(f"Max |Jac diff|={max_jac_diff:.3e}, Max |Res diff|={max_res_diff:.3e}")

    if jac_ok:
        print("✅ Jacobian matches within tolerances.")
    else:
        print(f"❌ Jacobian mismatch (tol={jac_tol:.3e}).")
    if res_ok:
        print("✅ Residual matches within tolerances.")
    else:
        print(f"❌ Residual mismatch (tol={res_tol:.3e}).")

    return jac_ok and res_ok


def compare_terms(pc: Dict, fx: Dict, backend: str, rtol: float, atol: float):
    P_map = create_true_dof_map(pc["dof_handler"], fx["W"])
    # Assemble terms using the requested backend; pass `--backend python` if you
    # want to avoid JIT compilation in term-by-term mode.
    res_terms_pc, jac_terms_pc = assemble_pycutfem_terms(pc, backend)
    res_forms_fx, jac_forms_fx = build_fenics_terms(fx)
    res_terms_fx, jac_terms_fx = assemble_fenics_terms(res_forms_fx, jac_forms_fx)

    def _summary(name, pc_arr, fx_arr):
        fx_perm = fx_arr[P_map] if isinstance(pc_arr, np.ndarray) else fx_arr[P_map, :][:, P_map]
        diff = pc_arr - fx_perm
        if isinstance(pc_arr, np.ndarray):
            scale = max(np.max(np.abs(pc_arr)) if pc_arr.size else 0.0, np.max(np.abs(fx_perm)) if fx_perm.size else 0.0)
            tol = atol + rtol * scale
            return float(np.max(np.abs(diff))) if diff.size else 0.0, tol
        else:
            scale = 0.0
            if pc_arr.nnz:
                scale = max(scale, float(np.max(np.abs(pc_arr.data))))
            if fx_perm.nnz:
                scale = max(scale, float(np.max(np.abs(fx_perm.data))))
            tol = atol + rtol * scale
            maxdiff = float(np.max(np.abs((pc_arr - fx_perm).data))) if (pc_arr - fx_perm).nnz else 0.0
            return maxdiff, tol

    print("Residual term comparison (max abs diff):")
    for name in sorted(res_terms_pc):
        mdiff, tol = _summary(name, res_terms_pc[name], res_terms_fx[name])
        print(f"  {name:20s} diff={mdiff:.3e} tol={tol:.3e}")

    print("Jacobian term comparison (max abs diff):")
    for name in sorted(jac_terms_pc):
        mdiff, tol = _summary(name, jac_terms_pc[name], jac_terms_fx[name])
        print(f"  {name:20s} diff={mdiff:.3e} tol={tol:.3e}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Compare PyCutFEM vs FEniCSx FSI forms on the full reference mesh.")
    ap.add_argument("--mesh", type=Path, default=Path("examples/meshes/fsi_conforming.msh"), help="Path to the gmsh mesh.")
    ap.add_argument("--poly-order", type=int, default=2, help="Polynomial order (Taylor–Hood).")
    ap.add_argument("--dt", type=float, default=1.0, help="Time step size.")
    ap.add_argument("--theta", type=float, default=1.0, help="Theta parameter.")
    ap.add_argument("--quad-order", type=int, default=None, help="Override quadrature degree (defaults to 2*poly+4).")
    ap.add_argument("--backend", choices=("jit", "python"), default=os.environ.get("BACKEND", "jit"), help="PyCutFEM form backend.")
    ap.add_argument("--rtol", type=float, default=1e-8, help="Relative tolerance for comparisons.")
    ap.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for comparisons.")
    ap.add_argument("--dump", action="store_true", help="Save sparse matrices/residuals to garbage/fsi_fullmesh.")
    ap.add_argument("--terms", action="store_true", help="Assemble and compare term-by-term contributions.")
    return ap.parse_args()


def main():
    args = parse_args()
    quad_order = args.quad_order if args.quad_order is not None else 2 * args.poly_order + 4
    marks = {"fluid": 1, "solid": 2, "outlet": 13}

    print(f"Loading mesh from {args.mesh} ...")
    mesh_data = read_mesh(args.mesh)
    pc = setup_pycutfem(mesh_data, args.poly_order, args.dt, args.theta, quad_order, marks)
    fx = setup_fenics(mesh_data, args.poly_order, args.dt, args.theta, quad_order, marks)

    initialize_pycutfem(pc)
    initialize_fenics(fx)

    n_fx = fx["W"].dofmap.index_map.size_local * fx["W"].dofmap.index_map_bs
    blocks_fx = fenics_block_sizes(fx["W"])
    print(f"PyCutFEM DOFs: {pc['dof_handler'].total_dofs}, FEniCSx DOFs: {n_fx} (blocks {blocks_fx})")
    ok = compare(pc, fx, backend=args.backend, rtol=args.rtol, atol=args.atol, dump=args.dump)
    if args.terms:
        compare_terms(pc, fx, backend=args.backend, rtol=args.rtol, atol=args.atol)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
