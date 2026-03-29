#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mpi4py import MPI
import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import ufl
from basix.ufl import mixed_element
from scipy.sparse import csr_matrix

from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import Constant
from pycutfem.ufl.forms import Equation, assemble_form
from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import (
    _build_forms,
    _condition_balanced_kinematic_setup,
    _condition_balanced_field_scales,
    _create_problem,
    _full_field_scale_vector,
    _latent_inverse_array,
)


@dataclass
class CaseSummary:
    name: str
    directional_fd_max_rel: float
    directional_fd_per_case: dict[str, float]
    alpha_res_max_abs: float | None
    alpha_res_rel: float | None
    alpha_jac_max_abs: float | None
    alpha_jac_rel: float | None
    phi_res_max_abs: float | None = None
    phi_res_rel: float | None = None
    phi_jac_max_abs: float | None = None
    phi_jac_rel: float | None = None
    skeleton_pressure_res_max_abs: float | None = None
    skeleton_pressure_res_rel: float | None = None
    skeleton_pressure_jac_max_abs: float | None = None
    skeleton_pressure_jac_rel: float | None = None
    full_res_max_abs: float | None = None
    full_res_rel: float | None = None
    full_jac_max_abs: float | None = None
    full_jac_rel: float | None = None
    full_blocks: dict[str, dict[str, float]] | None = None


def _one_to_one_map_coords(coords_pc: np.ndarray, coords_fx: np.ndarray, tol: float = 1.0e-12) -> np.ndarray:
    coords_pc = np.asarray(coords_pc, dtype=float)
    coords_fx = np.asarray(coords_fx, dtype=float)
    if coords_pc.shape != coords_fx.shape:
        raise ValueError(f"Coordinate shape mismatch: {coords_pc.shape} vs {coords_fx.shape}")

    scale = 1.0 / max(float(tol), 1.0e-16)

    def _quantize(arr: np.ndarray) -> np.ndarray:
        return np.rint(arr * scale).astype(np.int64)

    keys_pc = _quantize(coords_pc)
    keys_fx = _quantize(coords_fx)
    buckets: dict[tuple[int, ...], list[int]] = {}
    for i, key in enumerate(keys_fx):
        buckets.setdefault(tuple(int(v) for v in key.tolist()), []).append(int(i))

    mapping = np.empty(coords_pc.shape[0], dtype=int)
    for i, key in enumerate(keys_pc):
        bucket = buckets.get(tuple(int(v) for v in key.tolist()))
        if not bucket:
            raise KeyError(f"No coordinate match for {coords_pc[i].tolist()}")
        mapping[i] = int(bucket.pop())
    return mapping


def _resolved_orders(poly_order: int, pressure_order: int | None, scalar_order: int | None) -> tuple[int, int, int]:
    p = int(poly_order)
    pp = int(max(1, p - 1) if pressure_order is None else pressure_order)
    ps = int(max(1, p - 1) if scalar_order is None else scalar_order)
    return p, pp, ps


def _parse_csv_fields(text: str) -> tuple[str, ...]:
    if text is None:
        return tuple()
    return tuple(part.strip() for part in str(text).split(",") if part.strip())


def _latent_map_expr_ufl(z, *, map_kind: str):
    key = str(map_kind or "sigmoid").strip().lower()
    if key == "sigmoid":
        return 1.0 / (1.0 + ufl.exp(-z))
    if key == "tanh":
        return 0.5 * (1.0 + ufl.tanh(z))
    if key == "algebraic":
        return 0.5 * (1.0 + z / ufl.sqrt(1.0 + z * z))
    raise ValueError(f"Unsupported latent_bounded_map={map_kind!r}.")


def _latent_map_prime_expr_ufl(z, *, map_kind: str):
    key = str(map_kind or "sigmoid").strip().lower()
    if key == "sigmoid":
        sig = _latent_map_expr_ufl(z, map_kind=key)
        return sig * (1.0 - sig)
    if key == "tanh":
        th = ufl.tanh(z)
        return 0.5 * (1.0 - th * th)
    if key == "algebraic":
        return 0.5 / ((1.0 + z * z) ** 1.5)
    raise ValueError(f"Unsupported latent_bounded_map={map_kind!r}.")


def _fx_mesh_size(mesh):
    """
    FEniCS-side surrogate for pycutfem `MeshSize()`.

    In pycutfem's Python/JIT backends on quadrilateral meshes, `MeshSize()`
    evaluates like `sqrt(cell_area)` rather than `CellDiameter`. Using
    `CellDiameter` in the audit overstates every stabilization term that scales
    with `MeshSize()` or `avg(MeshSize())`, most visibly the CIP blocks.
    """
    coords = np.asarray(mesh.geometry.x[:, :2], dtype=float)
    Lx = float(coords[:, 0].max() - coords[:, 0].min())
    Ly = float(coords[:, 1].max() - coords[:, 1].min())
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    n_cells = int(cell_map.size_local + cell_map.num_ghosts)
    h = float(np.sqrt((Lx * Ly) / max(n_cells, 1)))
    return dolfinx.fem.Constant(mesh, h)


def _compare_dense(pc_arr: np.ndarray, fx_arr: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(pc_arr, dtype=float) - np.asarray(fx_arr, dtype=float)
    denom = max(
        1.0,
        float(np.linalg.norm(np.asarray(pc_arr, dtype=float), ord=np.inf)),
        float(np.linalg.norm(np.asarray(fx_arr, dtype=float), ord=np.inf)),
    )
    return float(np.max(np.abs(diff))), float(np.linalg.norm(diff, ord=np.inf) / denom)


def _fx_grad_inner_jump(expr_u, expr_v, n_int_fx, gdim: int):
    shape_u = tuple(getattr(expr_u, "ufl_shape", ()) or ())
    if len(shape_u) != 1:
        return ufl.inner(ufl.jump(ufl.grad(expr_u), n_int_fx), ufl.jump(ufl.grad(expr_v), n_int_fx))
    total = None
    for i in range(int(min(int(shape_u[0]), int(gdim)))):
        term = ufl.jump(ufl.grad(expr_u[i]), n_int_fx) * ufl.jump(ufl.grad(expr_v[i]), n_int_fx)
        total = term if total is None else total + term
    return total if total is not None else ufl.inner(ufl.jump(ufl.grad(expr_u), n_int_fx), ufl.jump(ufl.grad(expr_v), n_int_fx))


def _pycutfem_audit_diffuse_traction_coeffs(*, enabled: bool) -> dict[str, object]:
    if not bool(enabled):
        return {
            "g_t_k": None,
            "g_t_n": None,
            "traction_weight_k": None,
            "traction_weight_n": None,
        }

    def _ax(expr):
        return Analytic(expr, degree=2)

    return {
        "g_t_k": (
            _ax(lambda x, y: 0.35 + 0.08 * np.asarray(x, dtype=float) - 0.05 * np.asarray(y, dtype=float)),
            _ax(lambda x, y: -0.21 + 0.03 * np.asarray(x, dtype=float) + 0.07 * np.asarray(y, dtype=float)),
        ),
        "g_t_n": (
            _ax(lambda x, y: 0.29 + 0.06 * np.asarray(x, dtype=float) - 0.04 * np.asarray(y, dtype=float)),
            _ax(lambda x, y: -0.16 + 0.02 * np.asarray(x, dtype=float) + 0.05 * np.asarray(y, dtype=float)),
        ),
        "traction_weight_k": _ax(
            lambda x, y: 0.40 + 0.09 * np.asarray(x, dtype=float) + 0.04 * np.asarray(y, dtype=float)
        ),
        "traction_weight_n": _ax(
            lambda x, y: 0.32 + 0.02 * np.asarray(x, dtype=float) + 0.06 * np.asarray(y, dtype=float)
        ),
    }


def _fenics_audit_diffuse_traction_exprs(mesh):
    x = ufl.SpatialCoordinate(mesh)
    x0 = x[0]
    x1 = x[1]
    return {
        "g_t_k": ufl.as_vector((0.35 + 0.08 * x0 - 0.05 * x1, -0.21 + 0.03 * x0 + 0.07 * x1)),
        "g_t_n": ufl.as_vector((0.29 + 0.06 * x0 - 0.04 * x1, -0.16 + 0.02 * x0 + 0.05 * x1)),
        "traction_weight_k": 0.40 + 0.09 * x0 + 0.04 * x1,
        "traction_weight_n": 0.32 + 0.02 * x0 + 0.06 * x1,
    }


def _initialize_problem(problem: dict[str, object], *, enable_phi_evolution: bool) -> None:
    problem["v_k"].set_values_from_function(lambda x, y: np.array([0.18 + 0.04 * x - 0.03 * y, -0.06 + 0.02 * x + 0.015 * y]))
    problem["v_n"].set_values_from_function(lambda x, y: np.array([0.11 + 0.03 * x - 0.02 * y, -0.03 + 0.015 * x + 0.010 * y]))
    problem["vS_k"].set_values_from_function(lambda x, y: np.array([0.05 + 0.02 * x - 0.01 * y, -0.03 + 0.01 * x + 0.015 * y]))
    problem["vS_n"].set_values_from_function(lambda x, y: np.array([0.03 + 0.01 * x - 0.008 * y, -0.015 + 0.008 * x + 0.010 * y]))
    problem["u_k"].set_values_from_function(lambda x, y: np.array([0.015 + 0.010 * x * (1.0 - x), -0.008 + 0.006 * y * (1.0 - y / 1.5)]))
    problem["u_n"].set_values_from_function(lambda x, y: np.array([0.010 + 0.006 * x * (1.0 - x), -0.005 + 0.004 * y * (1.0 - y / 1.5)]))
    problem["p_k"].set_values_from_function(lambda x, y: 0.25 + 0.10 * x - 0.05 * y)
    problem["p_n"].set_values_from_function(lambda x, y: 0.12 + 0.04 * x - 0.03 * y)
    if problem.get("p_mean_k") is not None:
        problem["p_mean_k"].nodal_values[:] = 0.0
        problem["p_mean_n"].nodal_values[:] = 0.0
    problem["alpha_k"].set_values_from_function(lambda x, y: 0.58 + 0.015 * x + 0.010 * y)
    problem["alpha_n"].set_values_from_function(lambda x, y: 0.54 + 0.010 * x + 0.008 * y)
    problem["mu_k"].set_values_from_function(lambda x, y: 0.10 + 0.020 * x - 0.015 * y)
    problem["mu_n"].set_values_from_function(lambda x, y: 0.07 + 0.015 * x - 0.010 * y)
    if bool(enable_phi_evolution):
        problem["phi_k"].set_values_from_function(lambda x, y: 0.68 + 0.04 * x - 0.03 * y)
        problem["phi_n"].set_values_from_function(lambda x, y: 0.66 + 0.03 * x - 0.02 * y)
        problem["S_k"].set_values_from_function(lambda x, y: 0.20 + 0.02 * x + 0.01 * y)
        problem["S_n"].set_values_from_function(lambda x, y: 0.17 + 0.015 * x + 0.008 * y)


def _field_to_func(problem: dict[str, object], *, enable_phi_evolution: bool) -> dict[str, object]:
    out = {
        "v_x": problem["v_k"].components[0],
        "v_y": problem["v_k"].components[1],
        "p": problem["p_k"],
        "vS_x": problem["vS_k"].components[0],
        "vS_y": problem["vS_k"].components[1],
        "u_x": problem["u_k"].components[0],
        "u_y": problem["u_k"].components[1],
        "alpha": problem["alpha_k"],
        "mu_alpha": problem["mu_k"],
    }
    if problem.get("p_mean_k") is not None:
        out["p_mean"] = problem["p_mean_k"]
    if bool(enable_phi_evolution):
        out["phi"] = problem["phi_k"]
        out["S"] = problem["S_k"]
    if problem.get("alpha_latent_k") is not None:
        out["alpha_latent"] = problem["alpha_latent_k"]
    if bool(enable_phi_evolution) and problem.get("phi_latent_k") is not None:
        out["phi_latent"] = problem["phi_latent_k"]
    return out


def _build_benchmark_case(
    *,
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    poly_order: int,
    pressure_order: int | None,
    scalar_order: int | None,
    dt: float,
    theta: float,
    enable_phi_evolution: bool,
    latent_bounded_transport: bool,
    latent_bounded_fields: tuple[str, ...],
    latent_bounded_eps: float,
    latent_bounded_map: str,
    latent_bounded_formulation: str,
    alpha_advect_with: str,
    alpha_advection_form: str,
    support_physics: str,
    M_alpha: float,
    gamma_alpha: float,
    eps_alpha: float,
    alpha_regularization: str,
    D_phi: float,
    gamma_phi: float,
    phi_supg: float = 0.0,
    phi_cip: float | None = None,
    alpha_supg: float = 0.0,
    alpha_cip: float = 0.0,
    v_supg: float = 0.0,
    v_supg_mode: str = "streamline",
    v_supg_c_nu: float = 4.0,
    u_supg: float = 0.0,
    v_cip: float = 0.0,
    gamma_div: float = 0.0,
    gamma_u: float = 0.0,
    u_extension_mode: str = "l2",
    gamma_u_pin: float = 0.0,
    u_cip: float = 0.0,
    u_cip_weight: str = "fluid",
    vS_cip: float = 0.0,
    gamma_vS: float | None = None,
    vS_extension_mode: str | None = None,
    gamma_vS_pin: float | None = None,
    fluid_convection: str = "full",
    include_skeleton_acceleration: bool = True,
    rho_s0_tilde: float = 1.1,
    skeleton_inertia_convection: str = "full",
    skeleton_pressure_mode: str = "whole_domain",
    alpha_biot: float | None = None,
    use_diffuse_traction: bool = False,
    pressure_mean_constraint: bool = False,
    solid_volumetric_split: bool = False,
    solid_volumetric_penalty: float = 1.0,
    mechanics_nondim_mode: str = "legacy",
    outdir: Path | None = None,
) -> tuple[dict[str, object], object, dict[str, object], int]:
    poly_order, pressure_order, scalar_order = _resolved_orders(poly_order, pressure_order, scalar_order)
    qdeg = max(6, 2 * int(poly_order) + 2)
    problem = _create_problem(
        Lx=float(Lx),
        Ly=float(Ly),
        nx=int(nx),
        ny=int(ny),
        poly_order=int(poly_order),
        pressure_order=int(pressure_order),
        scalar_order=int(scalar_order),
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=bool(enable_phi_evolution),
        latent_bounded_transport=bool(latent_bounded_transport),
        latent_bounded_fields=tuple(latent_bounded_fields),
        latent_bounded_map=str(latent_bounded_map),
        latent_bounded_formulation=str(latent_bounded_formulation),
        pressure_mean_constraint=bool(pressure_mean_constraint),
        solid_volumetric_split=bool(solid_volumetric_split),
    )
    _initialize_problem(problem, enable_phi_evolution=bool(enable_phi_evolution))
    problem["latent_bounded_fields"] = tuple(str(name).strip() for name in tuple(latent_bounded_fields) if str(name).strip())
    if bool(latent_bounded_transport):
        if "alpha" in tuple(problem.get("latent_bounded_fields", tuple()) or tuple()) and problem.get("alpha_latent_k") is not None:
            problem["alpha_latent_k"].nodal_values[:] = _latent_inverse_array(
                problem["alpha_k"].nodal_values,
                eps=float(latent_bounded_eps),
                map_kind=str(latent_bounded_map),
            )
            problem["alpha_latent_n"].nodal_values[:] = _latent_inverse_array(
                problem["alpha_n"].nodal_values,
                eps=float(latent_bounded_eps),
                map_kind=str(latent_bounded_map),
            )
        if (
            bool(enable_phi_evolution)
            and "phi" in tuple(problem.get("latent_bounded_fields", tuple()) or tuple())
            and problem.get("phi_latent_k") is not None
        ):
            problem["phi_latent_k"].nodal_values[:] = _latent_inverse_array(
                problem["phi_k"].nodal_values,
                eps=float(latent_bounded_eps),
                map_kind=str(latent_bounded_map),
            )
            problem["phi_latent_n"].nodal_values[:] = _latent_inverse_array(
                problem["phi_n"].nodal_values,
                eps=float(latent_bounded_eps),
                map_kind=str(latent_bounded_map),
            )
    phi_cip_val = (
        (0.0 if str(support_physics).strip().lower() == "internal_conversion" else 1.0)
        if phi_cip is None
        else float(phi_cip)
    )
    traction_coeffs = _pycutfem_audit_diffuse_traction_coeffs(enabled=bool(use_diffuse_traction))
    forms = _build_forms(
        problem,
        qdeg=int(qdeg),
        dt_c=Constant(float(dt)),
        theta=float(theta),
        rho_f=1.0,
        mu_f=0.035,
        mu_b=0.035,
        mu_b_model="mu",
        kappa_inv=1.0e5,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        phi_b=0.18,
        M_alpha=float(M_alpha),
        gamma_alpha=float(gamma_alpha),
        eps_alpha=float(eps_alpha),
        solid_visco_eta=0.0,
        gamma_div=float(gamma_div),
        gamma_u=float(gamma_u),
        u_extension_mode=str(u_extension_mode),
        gamma_u_pin=float(gamma_u_pin),
        u_cip=float(u_cip),
        u_cip_weight=str(u_cip_weight),
        vS_cip=float(vS_cip),
        gamma_vS=(None if gamma_vS is None else float(gamma_vS)),
        vS_extension_mode=(None if vS_extension_mode is None else str(vS_extension_mode)),
        gamma_vS_pin=(None if gamma_vS_pin is None else float(gamma_vS_pin)),
        D_phi=float(D_phi),
        phi_diffusion_weight="fluid",
        gamma_phi=float(gamma_phi),
        phi_supg=float(phi_supg),
        phi_cip=float(phi_cip_val),
        alpha_supg=float(alpha_supg),
        alpha_cip=float(alpha_cip),
        v_supg=float(v_supg),
        v_supg_mode=str(v_supg_mode),
        v_supg_c_nu=float(v_supg_c_nu),
        u_supg=float(u_supg),
        v_cip=float(v_cip),
        alpha_regularization=str(alpha_regularization),
        alpha_reg_gamma=1.0,
        alpha_reg_eps_normal=float(eps_alpha),
        alpha_reg_eps_tangent=float(0.25 * eps_alpha),
        alpha_reg_eta=1.0e-12,
        alpha_advect_with=str(alpha_advect_with),
        alpha_advection_form=str(alpha_advection_form),
        support_physics=str(support_physics),
        skeleton_pressure_mode=str(skeleton_pressure_mode),
        alpha_biot=(None if alpha_biot is None else float(alpha_biot)),
        g_t_k=traction_coeffs["g_t_k"],
        g_t_n=traction_coeffs["g_t_n"],
        traction_weight_k=traction_coeffs["traction_weight_k"],
        traction_weight_n=traction_coeffs["traction_weight_n"],
        solid_model="linear",
        kappa_inv_model="refmap",
        enable_phi_evolution=bool(enable_phi_evolution),
        include_skeleton_acceleration=bool(include_skeleton_acceleration),
        rho_s0_tilde=float(rho_s0_tilde),
        skeleton_inertia_convection=str(skeleton_inertia_convection),
        fluid_convection=str(fluid_convection),
        solid_volumetric_split=bool(solid_volumetric_split),
        solid_volumetric_penalty=float(solid_volumetric_penalty),
        mechanics_nondim_mode=str(mechanics_nondim_mode),
    )
    problem["_audit_phi_b"] = 0.18
    problem["_audit_enable_phi_evolution"] = bool(enable_phi_evolution)
    problem["_audit_Lx"] = float(Lx)
    problem["_audit_Ly"] = float(Ly)
    problem["_audit_nx"] = int(nx)
    problem["_audit_ny"] = int(ny)
    problem["_audit_poly_order"] = int(poly_order)
    problem["_audit_pressure_order"] = int(pressure_order)
    problem["_audit_scalar_order"] = int(scalar_order)
    problem["_audit_dt"] = float(dt)
    problem["_audit_theta"] = float(theta)
    problem["_audit_D_phi"] = float(D_phi)
    problem["_audit_phi_diffusion_weight"] = "fluid"
    problem["_audit_gamma_phi"] = float(gamma_phi)
    problem["_audit_phi_supg"] = float(phi_supg)
    problem["_audit_phi_cip"] = float(phi_cip_val)
    problem["_audit_alpha_supg"] = float(alpha_supg)
    problem["_audit_alpha_cip"] = float(alpha_cip)
    problem["_audit_v_supg"] = float(v_supg)
    problem["_audit_v_supg_mode"] = str(v_supg_mode)
    problem["_audit_v_supg_c_nu"] = float(v_supg_c_nu)
    problem["_audit_u_supg"] = float(u_supg)
    problem["_audit_v_cip"] = float(v_cip)
    problem["_audit_use_diffuse_traction"] = bool(use_diffuse_traction)
    problem["_audit_mu_max"] = 0.0
    problem["_audit_K_S"] = 1.0
    problem["_audit_k_d"] = 0.0
    problem["_audit_support_physics"] = str(support_physics)
    problem["_audit_skeleton_pressure_mode"] = str(skeleton_pressure_mode)
    problem["_audit_alpha_biot"] = (None if alpha_biot is None else float(alpha_biot))
    problem["_audit_include_skeleton_acceleration"] = bool(include_skeleton_acceleration)
    problem["_audit_rho_s0_tilde"] = float(rho_s0_tilde)
    problem["_audit_latent_bounded_transport"] = bool(latent_bounded_transport)
    problem["_audit_latent_bounded_fields"] = tuple(problem.get("latent_bounded_fields", tuple()) or tuple())
    problem["_audit_latent_bounded_map"] = str(latent_bounded_map)
    problem["_audit_latent_bounded_formulation"] = str(latent_bounded_formulation)
    problem["_audit_pressure_mean_constraint"] = bool(pressure_mean_constraint)
    problem["_audit_solid_volumetric_split"] = bool(solid_volumetric_split)
    problem["_audit_form_params"] = {
        "rho_f": 1.0,
        "mu_f": 0.035,
        "mu_b": 0.035,
        "mu_b_model": "mu",
        "kappa_inv": 1.0e5,
        "kappa_inv_model": "refmap",
        "mu_s": 1.67785e5,
        "lambda_s": 8.22148e6,
        "phi_b": 0.18,
        "solid_visco_eta": 0.0,
        "gamma_div": float(gamma_div),
        "gamma_u": float(gamma_u),
        "u_extension_mode": str(u_extension_mode),
        "gamma_u_pin": float(gamma_u_pin),
        "u_cip": float(u_cip),
        "u_cip_weight": str(u_cip_weight),
        "vS_cip": float(vS_cip),
        "gamma_vS": (None if gamma_vS is None else float(gamma_vS)),
        "vS_extension_mode": (None if vS_extension_mode is None else str(vS_extension_mode)),
        "gamma_vS_pin": (None if gamma_vS_pin is None else float(gamma_vS_pin)),
        "D_phi": float(D_phi),
        "phi_diffusion_weight": "fluid",
        "gamma_phi": float(gamma_phi),
        "phi_supg": float(phi_supg),
        "phi_cip": float(phi_cip_val),
        "alpha_supg": float(alpha_supg),
        "alpha_cip": float(alpha_cip),
        "v_supg": float(v_supg),
        "v_supg_mode": str(v_supg_mode),
        "v_supg_c_nu": float(v_supg_c_nu),
        "u_supg": float(u_supg),
        "v_cip": float(v_cip),
        "use_diffuse_traction": bool(use_diffuse_traction),
        "alpha_advect_with": str(alpha_advect_with),
        "alpha_advection_form": str(alpha_advection_form),
        "support_physics": str(support_physics),
        "skeleton_pressure_mode": str(skeleton_pressure_mode),
        "alpha_biot": (None if alpha_biot is None else float(alpha_biot)),
        "solid_model": "linear",
        "fluid_convection": str(fluid_convection),
        "include_skeleton_acceleration": bool(include_skeleton_acceleration),
        "rho_s0_tilde": float(rho_s0_tilde),
        "skeleton_inertia_convection": str(skeleton_inertia_convection),
        "alpha_mu_aux_pin": 1.0,
        "alpha_regularization": str(alpha_regularization),
        "M_alpha": float(M_alpha),
        "gamma_alpha": float(gamma_alpha),
        "eps_alpha": float(eps_alpha),
        "latent_bounded_transport": bool(latent_bounded_transport),
        "latent_bounded_fields": tuple(problem.get("latent_bounded_fields", tuple()) or tuple()),
        "latent_bounded_map": str(latent_bounded_map),
        "latent_bounded_formulation": str(latent_bounded_formulation),
        "pressure_mean_constraint": bool(pressure_mean_constraint),
        "solid_volumetric_split": bool(solid_volumetric_split),
        "solid_volumetric_penalty": float(solid_volumetric_penalty),
        "mechanics_nondim_mode": str(mechanics_nondim_mode),
        "D_S": 0.0,
        "mu_max": 0.0,
        "K_S": 1.0,
        "k_g": 0.0,
        "k_d": 0.0,
        "Y": 1.0,
        "rho_s_star": 1.0,
    }
    return problem, forms, _field_to_func(problem, enable_phi_evolution=bool(enable_phi_evolution)), int(qdeg)


def _assemble_residual(forms, dh, qdeg: int, *, backend: str) -> np.ndarray:
    _, r = assemble_form(Equation(None, forms.residual_form), dof_handler=dh, bcs=[], quad_order=int(qdeg), backend=str(backend))
    return np.asarray(r, dtype=float)


def _directional_fd_audit(
    problem,
    forms,
    field_to_func: dict[str, object],
    *,
    qdeg: int,
    enable_phi_evolution: bool,
    backend: str,
    n_random: int = 1,
    seed: int = 1234,
    fd_eps: float = 1.0e-6,
) -> tuple[float, dict[str, float]]:
    dh = problem["dh"]
    K, _ = assemble_form(
        Equation(forms.jacobian_form, forms.residual_form),
        dof_handler=dh,
        bcs=[],
        quad_order=int(qdeg),
        backend=str(backend),
    )
    K = K.tocsr()
    ndofs = int(K.shape[1])
    probe_fields = ["v_x", "v_y", "p", "vS_x", "vS_y", "u_x", "u_y", "alpha", "mu_alpha"]
    if problem.get("p_mean_k") is not None:
        probe_fields.append("p_mean")
    if bool(enable_phi_evolution):
        probe_fields.extend(["phi", "S"])
    if problem.get("alpha_latent_k") is not None:
        probe_fields.append("alpha_latent")
    if bool(enable_phi_evolution) and problem.get("phi_latent_k") is not None:
        probe_fields.append("phi_latent")

    rng = np.random.default_rng(int(seed))
    directions: list[tuple[str, np.ndarray]] = []

    def _field_direction(name: str, fields: list[str]) -> None:
        z = np.zeros(ndofs, dtype=float)
        for fld in fields:
            sl = np.asarray(dh.get_field_slice(fld), dtype=int)
            if sl.size:
                z[sl] = rng.standard_normal(sl.size)
        scale = float(np.linalg.norm(z, ord=np.inf))
        if scale > 0.0:
            directions.append((name, z / scale))

    _field_direction("alpha_only", ["alpha"])
    if problem.get("alpha_latent_k") is not None:
        _field_direction("alpha_latent_only", ["alpha_latent"])
    _field_direction("v_only", ["v_x", "v_y"])
    _field_direction("vS_only", ["vS_x", "vS_y"])
    _field_direction("u_only", ["u_x", "u_y"])
    if bool(enable_phi_evolution):
        _field_direction("phiS_only", ["phi", "S"])
        if problem.get("phi_latent_k") is not None:
            _field_direction("phi_latent_only", ["phi_latent"])
    _field_direction("random_all", probe_fields)
    for i in range(int(n_random) - 1):
        _field_direction(f"random_all_{i + 2}", probe_fields)

    eps = float(fd_eps)
    per_case: dict[str, float] = {}
    max_rel = 0.0
    for name, z in directions:
        def _apply(sign: float) -> list[tuple[object, np.ndarray, np.ndarray]]:
            touched: list[tuple[object, np.ndarray, np.ndarray]] = []
            for fld, func in field_to_func.items():
                sl = np.asarray(dh.get_field_slice(fld), dtype=int)
                if sl.size == 0:
                    continue
                dz = z[sl]
                if np.allclose(dz, 0.0):
                    continue
                old = np.asarray(func.get_nodal_values(sl), dtype=float).copy()
                func.set_nodal_values(sl, old + sign * eps * dz)
                touched.append((func, sl, old))
            return touched

        touched_p = _apply(+1.0)
        Rp = _assemble_residual(forms, dh, int(qdeg), backend=str(backend))
        for func, sl, old in touched_p:
            func.set_nodal_values(sl, old)

        touched_m = _apply(-1.0)
        Rm = _assemble_residual(forms, dh, int(qdeg), backend=str(backend))
        for func, sl, old in touched_m:
            func.set_nodal_values(sl, old)

        fd = (Rp - Rm) / (2.0 * eps)
        lin = K @ z
        denom = max(1.0, float(np.linalg.norm(fd, ord=np.inf)), float(np.linalg.norm(lin, ord=np.inf)))
        rel = float(np.linalg.norm(fd - lin, ord=np.inf)) / denom
        per_case[name] = rel
        max_rel = max(max_rel, rel)
    return max_rel, per_case


def _build_fenics_alpha_system(*, Lx: float, Ly: float, nx: int, ny: int, poly_order: int, scalar_order: int, enable_phi_evolution: bool):
    mesh_fx = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0], dtype=float), np.array([float(Lx), float(Ly)], dtype=float)],
        [int(nx), int(ny)],
        cell_type=dolfinx.mesh.CellType.quadrilateral,
    )
    gdim = mesh_fx.geometry.dim
    V_el = basix.ufl.element("Lagrange", "quadrilateral", int(poly_order), shape=(gdim,))
    A_el = basix.ufl.element("Lagrange", "quadrilateral", int(scalar_order))
    W_el = mixed_element([V_el, V_el, A_el, A_el]) if bool(enable_phi_evolution) else mixed_element([V_el, V_el, A_el])
    if hasattr(dolfinx.fem, "functionspace"):
        W = dolfinx.fem.functionspace(mesh_fx, W_el)
    else:
        W = dolfinx.fem.FunctionSpace(mesh_fx, W_el)
    return mesh_fx, W


def _fenics_parent_dofs_for_component(W, subspace_index: int, component: int) -> tuple[np.ndarray, np.ndarray]:
    Wv, V_map = W.sub(subspace_index).collapse()
    Wc, C_map = Wv.sub(component).collapse()
    coords = Wc.tabulate_dof_coordinates()[:, :2]
    parent = np.asarray(V_map, dtype=int)[np.asarray(C_map, dtype=int)]
    return parent, coords


def _fenics_parent_dofs_for_scalar(W, subspace_index: int) -> tuple[np.ndarray, np.ndarray]:
    Ws, S_map = W.sub(subspace_index).collapse()
    coords = Ws.tabulate_dof_coordinates()[:, :2]
    parent = np.asarray(S_map, dtype=int)
    return parent, coords


def _map_pycutfem_to_fenics_alpha(problem, W, *, enable_phi_evolution: bool) -> dict[str, np.ndarray]:
    dh = problem["dh"]
    mapping: dict[str, np.ndarray] = {}
    for fld, sub_idx, comp in (("v_x", 0, 0), ("v_y", 0, 1), ("vS_x", 1, 0), ("vS_y", 1, 1)):
        parent, coords_fx = _fenics_parent_dofs_for_component(W, sub_idx, comp)
        coords_pc = np.asarray(dh.get_dof_coords(fld), dtype=float)
        mapping[fld] = parent[_one_to_one_map_coords(coords_pc, coords_fx)]
    parent_alpha, coords_alpha = _fenics_parent_dofs_for_scalar(W, 2)
    mapping["alpha"] = parent_alpha[_one_to_one_map_coords(np.asarray(dh.get_dof_coords("alpha"), dtype=float), coords_alpha)]
    if bool(enable_phi_evolution):
        parent_phi, coords_phi = _fenics_parent_dofs_for_scalar(W, 3)
        mapping["phi"] = parent_phi[_one_to_one_map_coords(np.asarray(dh.get_dof_coords("phi"), dtype=float), coords_phi)]
    return mapping


def _load_fenics_state(problem, W, mapping: dict[str, np.ndarray], *, enable_phi_evolution: bool) -> tuple[object, object]:
    w_k = dolfinx.fem.Function(W, name="w_k")
    w_n = dolfinx.fem.Function(W, name="w_n")
    fields_k = [
        ("v_x", problem["v_k"].components[0]),
        ("v_y", problem["v_k"].components[1]),
        ("vS_x", problem["vS_k"].components[0]),
        ("vS_y", problem["vS_k"].components[1]),
        ("alpha", problem["alpha_k"]),
    ]
    fields_n = [
        ("v_x", problem["v_n"].components[0]),
        ("v_y", problem["v_n"].components[1]),
        ("vS_x", problem["vS_n"].components[0]),
        ("vS_y", problem["vS_n"].components[1]),
        ("alpha", problem["alpha_n"]),
    ]
    if bool(enable_phi_evolution):
        fields_k.append(("phi", problem["phi_k"]))
        fields_n.append(("phi", problem["phi_n"]))
    for fld, func in fields_k:
        w_k.x.array[mapping[fld]] = np.asarray(func.nodal_values, dtype=float)
    for fld, func in fields_n:
        w_n.x.array[mapping[fld]] = np.asarray(func.nodal_values, dtype=float)
    w_k.x.scatter_forward()
    w_n.x.scatter_forward()
    return w_k, w_n


def _fenics_alpha_forms(
    *,
    W,
    w_k,
    w_n,
    dt: float,
    theta: float,
    qdeg: int,
    alpha_advect_with: str,
    alpha_advection_form: str,
    phi_b: float,
    enable_phi_evolution: bool,
    alpha_supg: float,
):
    dw = ufl.TrialFunction(W)
    wtest = ufl.TestFunction(W)
    if bool(enable_phi_evolution):
        dv_fx, dvS_fx, dalpha_fx, dphi_fx = ufl.split(dw)
        zeta_v_fx, zeta_vS_fx, xi_fx, phi_test_fx = ufl.split(wtest)
        v_k_fx, vS_k_fx, alpha_k_fx, phi_k_fx = ufl.split(w_k)
        v_n_fx, vS_n_fx, alpha_n_fx, phi_n_fx = ufl.split(w_n)
    else:
        dv_fx, dvS_fx, dalpha_fx = ufl.split(dw)
        zeta_v_fx, zeta_vS_fx, xi_fx = ufl.split(wtest)
        v_k_fx, vS_k_fx, alpha_k_fx = ufl.split(w_k)
        v_n_fx, vS_n_fx, alpha_n_fx = ufl.split(w_n)
        dphi_fx = None
        phi_test_fx = None
        phi_k_fx = None
        phi_n_fx = None

    th_fx = dolfinx.fem.Constant(W.mesh, float(theta))
    omth_fx = dolfinx.fem.Constant(W.mesh, 1.0 - float(theta))
    inv_dt_fx = dolfinx.fem.Constant(W.mesh, 1.0 / float(dt))
    dx_fx = ufl.dx(metadata={"quadrature_degree": int(qdeg)})

    adv_key = str(alpha_advect_with).strip().lower()
    if adv_key == "vs":
        adv_u_k, adv_u_n = vS_k_fx, vS_n_fx
    elif adv_key == "v":
        adv_u_k, adv_u_n = v_k_fx, v_n_fx
    elif adv_key == "mix":
        if bool(enable_phi_evolution):
            C_n_fx = 1.0 - alpha_n_fx * (1.0 - phi_n_fx)
            B_n_fx = alpha_n_fx * (1.0 - phi_n_fx)
        else:
            phi_b_fx = dolfinx.fem.Constant(W.mesh, float(phi_b))
            C_n_fx = 1.0 - alpha_n_fx * (1.0 - phi_b_fx)
            B_n_fx = alpha_n_fx * (1.0 - phi_b_fx)
        adv_u_k = C_n_fx * v_k_fx + B_n_fx * vS_k_fx
        adv_u_n = C_n_fx * v_n_fx + B_n_fx * vS_n_fx
    elif adv_key in {"biofilm", "biofilm_volume", "biofilm-volume", "phase", "phase_volume", "phase-volume"}:
        if bool(enable_phi_evolution):
            adv_u_k = phi_k_fx * v_k_fx + (1.0 - phi_k_fx) * vS_k_fx
            adv_u_n = phi_n_fx * v_n_fx + (1.0 - phi_n_fx) * vS_n_fx
        else:
            phi_b_fx = dolfinx.fem.Constant(W.mesh, float(phi_b))
            adv_u_k = phi_b_fx * v_k_fx + (1.0 - phi_b_fx) * vS_k_fx
            adv_u_n = phi_b_fx * v_n_fx + (1.0 - phi_b_fx) * vS_n_fx
    elif adv_key == "interface":
        adv_u_k = 0.5 * (v_k_fx + vS_k_fx)
        adv_u_n = 0.5 * (v_n_fx + vS_n_fx)
    else:
        raise ValueError(f"Unsupported alpha_advect_with={alpha_advect_with!r}")

    form_key = str(alpha_advection_form).strip().lower()
    if form_key == "advective":
        r_alpha_fx = xi_fx * ((alpha_k_fx - alpha_n_fx) * inv_dt_fx) * dx_fx
        r_alpha_fx += th_fx * xi_fx * ufl.dot(ufl.grad(alpha_k_fx), adv_u_k) * dx_fx
        r_alpha_fx += omth_fx * xi_fx * ufl.dot(ufl.grad(alpha_n_fx), adv_u_n) * dx_fx
        adv_alpha_k_fx = ufl.dot(ufl.grad(alpha_k_fx), adv_u_k)
        adv_alpha_n_fx = ufl.dot(ufl.grad(alpha_n_fx), adv_u_n)
    elif form_key == "conservative_weak":
        r_alpha_fx = xi_fx * ((alpha_k_fx - alpha_n_fx) * inv_dt_fx) * dx_fx
        r_alpha_fx += -th_fx * ufl.dot(alpha_k_fx * adv_u_k, ufl.grad(xi_fx)) * dx_fx
        r_alpha_fx += -omth_fx * ufl.dot(alpha_n_fx * adv_u_n, ufl.grad(xi_fx)) * dx_fx
        adv_alpha_k_fx = ufl.dot(ufl.grad(alpha_k_fx), adv_u_k) + alpha_k_fx * ufl.div(adv_u_k)
        adv_alpha_n_fx = ufl.dot(ufl.grad(alpha_n_fx), adv_u_n) + alpha_n_fx * ufl.div(adv_u_n)
    elif form_key == "conservative":
        a_k = alpha_k_fx
        a_n = alpha_n_fx
        r_alpha_fx = xi_fx * ((a_k - a_n) * inv_dt_fx) * dx_fx
        r_alpha_fx += th_fx * xi_fx * (ufl.dot(ufl.grad(a_k), adv_u_k) + a_k * ufl.div(adv_u_k)) * dx_fx
        r_alpha_fx += omth_fx * xi_fx * (ufl.dot(ufl.grad(a_n), adv_u_n) + a_n * ufl.div(adv_u_n)) * dx_fx
        adv_alpha_k_fx = ufl.dot(ufl.grad(a_k), adv_u_k) + a_k * ufl.div(adv_u_k)
        adv_alpha_n_fx = ufl.dot(ufl.grad(a_n), adv_u_n) + a_n * ufl.div(adv_u_n)
    elif form_key == "interface_band_conservative":
        a_k = 4.0 * alpha_k_fx * (1.0 - alpha_k_fx)
        a_n = 4.0 * alpha_n_fx * (1.0 - alpha_n_fx)
        r_alpha_fx = xi_fx * ((a_k - a_n) * inv_dt_fx) * dx_fx
        r_alpha_fx += th_fx * xi_fx * (ufl.dot(ufl.grad(a_k), adv_u_k) + a_k * ufl.div(adv_u_k)) * dx_fx
        r_alpha_fx += omth_fx * xi_fx * (ufl.dot(ufl.grad(a_n), adv_u_n) + a_n * ufl.div(adv_u_n)) * dx_fx
        adv_alpha_k_fx = ufl.dot(ufl.grad(a_k), adv_u_k) + a_k * ufl.div(adv_u_k)
        adv_alpha_n_fx = ufl.dot(ufl.grad(a_n), adv_u_n) + a_n * ufl.div(adv_u_n)
    else:
        raise ValueError(f"Unsupported alpha_advection_form={alpha_advection_form!r}")
    if float(alpha_supg) != 0.0:
        h_a = _fx_mesh_size(W.mesh)
        vmag = ufl.sqrt(ufl.dot(adv_u_n, adv_u_n) + 1.0e-12)
        denom = (2.0 * inv_dt_fx) * (2.0 * inv_dt_fx) + (2.0 * vmag / (h_a + 1.0e-12)) * (2.0 * vmag / (h_a + 1.0e-12))
        tau_supg = float(alpha_supg) / ufl.sqrt(denom + 1.0e-16)
        w_supg = ufl.dot(ufl.grad(xi_fx), adv_u_n)
        f_alpha_supg_k_fx = (alpha_k_fx - alpha_n_fx) * inv_dt_fx
        f_alpha_supg_k_fx += th_fx * adv_alpha_k_fx + omth_fx * adv_alpha_n_fx
        r_alpha_fx += tau_supg * w_supg * f_alpha_supg_k_fx * dx_fx
    a_alpha_fx = ufl.derivative(r_alpha_fx, w_k, dw)
    return r_alpha_fx, a_alpha_fx


def _fenics_phi_forms(
    *,
    W,
    w_k,
    w_n,
    dt: float,
    theta: float,
    qdeg: int,
    support_physics: str,
    D_phi: float,
    phi_diffusion_weight: str,
    gamma_phi: float,
    mu_max: float,
    K_S: float,
    k_d: float,
    phi_supg: float,
    phi_cip: float,
):
    dw = ufl.TrialFunction(W)
    wtest = ufl.TestFunction(W)
    dv_fx, dvS_fx, dalpha_fx, dphi_fx = ufl.split(dw)
    zeta_v_fx, zeta_vS_fx, xi_fx, phi_test_fx = ufl.split(wtest)
    del dv_fx, zeta_v_fx, xi_fx
    v_k_fx, vS_k_fx, alpha_k_fx, phi_k_fx = ufl.split(w_k)
    v_n_fx, vS_n_fx, alpha_n_fx, phi_n_fx = ufl.split(w_n)
    del v_k_fx, v_n_fx

    th_fx = dolfinx.fem.Constant(W.mesh, float(theta))
    omth_fx = dolfinx.fem.Constant(W.mesh, 1.0 - float(theta))
    inv_dt_fx = dolfinx.fem.Constant(W.mesh, 1.0 / float(dt))
    D_phi_fx = dolfinx.fem.Constant(W.mesh, float(D_phi))
    gamma_phi_fx = dolfinx.fem.Constant(W.mesh, float(gamma_phi))
    dx_fx = ufl.dx(metadata={"quadrature_degree": int(qdeg)})
    dS_fx = ufl.dS(metadata={"quadrature_degree": int(qdeg)})

    if float(mu_max) != 0.0:
        raise NotImplementedError("FEniCS phi audit currently assumes mu_max=0 so S does not enter the phi block.")
    if float(K_S) <= 0.0:
        raise ValueError("K_S must be positive.")
    k_d_fx = dolfinx.fem.Constant(W.mesh, float(k_d))
    Pi_k_fx = (-k_d_fx) * alpha_k_fx * (1.0 - phi_k_fx)
    Pi_n_fx = (-k_d_fx) * alpha_n_fx * (1.0 - phi_n_fx)

    phi_diff_key = str(phi_diffusion_weight).strip().lower()
    if phi_diff_key in {"unity", "none", "constant", "const", "all"}:
        w_phi_diff_k_fx = 1.0
    elif phi_diff_key in {"fluid", "one_minus_alpha", "1-alpha", "one-minus-alpha"}:
        w_phi_diff_k_fx = 1.0 - alpha_k_fx
    elif phi_diff_key in {"biofilm", "alpha"}:
        w_phi_diff_k_fx = alpha_k_fx
    else:
        raise ValueError(f"Unsupported phi_diffusion_weight={phi_diffusion_weight!r}")

    one_m_alpha_k_fx = 1.0 - alpha_k_fx
    w_phi_fluid4_k_fx = one_m_alpha_k_fx * one_m_alpha_k_fx
    w_phi_fluid4_k_fx = w_phi_fluid4_k_fx * w_phi_fluid4_k_fx
    w_phi_fluid8_k_fx = w_phi_fluid4_k_fx * w_phi_fluid4_k_fx
    w_phi_fluid_k_fx = w_phi_fluid8_k_fx * w_phi_fluid8_k_fx

    support_key = str(support_physics).strip().lower()
    if support_key == "internal_conversion":
        B_k_fx = alpha_k_fx * (1.0 - phi_k_fx)
        B_n_fx = alpha_n_fx * (1.0 - phi_n_fx)
        r_phi_fx = phi_test_fx * ((B_k_fx - B_n_fx) * inv_dt_fx) * dx_fx
        r_phi_fx += -th_fx * ufl.dot(B_k_fx * vS_k_fx, ufl.grad(phi_test_fx)) * dx_fx
        r_phi_fx += -omth_fx * ufl.dot(B_n_fx * vS_n_fx, ufl.grad(phi_test_fx)) * dx_fx
        r_phi_fx += -phi_test_fx * (th_fx * Pi_k_fx + omth_fx * Pi_n_fx) * dx_fx
    elif support_key == "legacy_exchange":
        div_vS_k_fx = ufl.div(vS_k_fx)
        div_vS_n_fx = ufl.div(vS_n_fx)
        Fphi_k_fx = ufl.dot(ufl.grad(phi_k_fx), vS_k_fx) - (1.0 - phi_k_fx) * div_vS_k_fx + Pi_k_fx
        Fphi_n_fx = ufl.dot(ufl.grad(phi_n_fx), vS_n_fx) - (1.0 - phi_n_fx) * div_vS_n_fx + Pi_n_fx
        r_phi_fx = alpha_k_fx * phi_test_fx * ((phi_k_fx - phi_n_fx) * inv_dt_fx) * dx_fx
        r_phi_fx += th_fx * alpha_k_fx * phi_test_fx * Fphi_k_fx * dx_fx
        r_phi_fx += omth_fx * alpha_n_fx * phi_test_fx * Fphi_n_fx * dx_fx
    else:
        raise ValueError(f"Unsupported support_physics={support_physics!r}")

    r_phi_fx += D_phi_fx * w_phi_diff_k_fx * ufl.inner(ufl.grad(phi_k_fx), ufl.grad(phi_test_fx)) * dx_fx
    r_phi_fx += gamma_phi_fx * w_phi_fluid_k_fx * (phi_k_fx - 1.0) * phi_test_fx * dx_fx

    if support_key == "internal_conversion":
        if float(phi_supg) != 0.0:
            h_p = _fx_mesh_size(W.mesh)
            vmag = ufl.sqrt(ufl.dot(vS_n_fx, vS_n_fx) + 1.0e-12)
            denom = (2.0 * inv_dt_fx) * (2.0 * inv_dt_fx) + (2.0 * vmag / (h_p + 1.0e-12)) * (2.0 * vmag / (h_p + 1.0e-12))
            tau_supg = float(phi_supg) / ufl.sqrt(denom + 1.0e-16)
            w_supg = alpha_n_fx * ufl.dot(ufl.grad(phi_test_fx), vS_n_fx)
            divBvS_k_fx = ufl.div(B_k_fx * vS_k_fx)
            divBvS_n_fx = ufl.div(B_n_fx * vS_n_fx)
            f_phi_supg_k_fx = (B_k_fx - B_n_fx) * inv_dt_fx
            f_phi_supg_k_fx += th_fx * divBvS_k_fx + omth_fx * divBvS_n_fx
            f_phi_supg_k_fx += -(th_fx * Pi_k_fx + omth_fx * Pi_n_fx)
            f_phi_supg_k_fx += gamma_phi_fx * w_phi_fluid_k_fx * (phi_k_fx - 1.0)
            r_phi_fx += tau_supg * w_supg * f_phi_supg_k_fx * dx_fx

        if float(phi_cip) != 0.0:
            n_int = ufl.FacetNormal(W.mesh)
            h_F = ufl.avg(_fx_mesh_size(W.mesh))
            tau_cip = float(phi_cip) * (h_F * h_F * h_F) * inv_dt_fx
            a_avg = ufl.avg(alpha_n_fx)
            a_jump = ufl.jump(alpha_n_fx)
            w_phi_cip_fx = a_avg * a_avg + (-0.25 * a_jump * a_jump)
            phi_trace_k_fx = ufl.avg(phi_k_fx)
            alpha_trace_k_fx = ufl.avg(alpha_k_fx)
            j_alpha_k_fx = ufl.jump(ufl.grad(alpha_k_fx), n_int)
            j_phi_k_fx = ufl.jump(ufl.grad(phi_k_fx), n_int)
            j_B_k_fx = (1.0 - phi_trace_k_fx) * j_alpha_k_fx - alpha_trace_k_fx * j_phi_k_fx
            r_phi_fx += tau_cip * w_phi_cip_fx * ufl.inner(j_B_k_fx, ufl.jump(ufl.grad(phi_test_fx), n_int)) * dS_fx
    a_phi_fx = ufl.derivative(r_phi_fx, w_k, dw)
    return r_phi_fx, a_phi_fx


def _pycutfem_alpha_compare(
    problem,
    forms,
    *,
    qdeg: int,
    alpha_advect_with: str,
    alpha_advection_form: str,
    phi_b: float,
    backend: str,
    full_compare: dict[str, object] | None = None,
) -> tuple[float, float, float, float]:
    alpha_reg_key = str(problem.get("_audit_form_params", {}).get("alpha_regularization", "none")).strip().lower()
    if alpha_reg_key == "ch":
        # The reduced alpha-only helper does not include mu_alpha as an unknown,
        # so it cannot represent the Cahn-Hilliard coupling in the alpha row.
        # Reuse the full-system FEniCSx comparison for CH cases instead.
        if full_compare is None:
            full_compare = _pycutfem_full_system_compare(problem, forms, qdeg=int(qdeg), backend=str(backend))
        alpha_block = dict(full_compare["full_blocks"]["alpha"])
        return (
            float(alpha_block["res_max_abs"]),
            float(alpha_block["res_rel"]),
            float(alpha_block["jac_max_abs"]),
            float(alpha_block["jac_rel"]),
        )

    enable_phi_evolution = bool(problem["_audit_enable_phi_evolution"])
    mesh_fx, W_fx = _build_fenics_alpha_system(
        Lx=float(problem["_audit_Lx"]),
        Ly=float(problem["_audit_Ly"]),
        nx=int(problem["_audit_nx"]),
        ny=int(problem["_audit_ny"]),
        poly_order=int(problem["_audit_poly_order"]),
        scalar_order=int(problem["_audit_scalar_order"]),
        enable_phi_evolution=bool(enable_phi_evolution),
    )
    mapping = _map_pycutfem_to_fenics_alpha(problem, W_fx, enable_phi_evolution=bool(enable_phi_evolution))
    w_k_fx, w_n_fx = _load_fenics_state(problem, W_fx, mapping, enable_phi_evolution=bool(enable_phi_evolution))
    r_fx, a_fx = _fenics_alpha_forms(
        W=W_fx,
        w_k=w_k_fx,
        w_n=w_n_fx,
        dt=float(problem["_audit_dt"]),
        theta=float(problem["_audit_theta"]),
        qdeg=int(qdeg),
        alpha_advect_with=str(alpha_advect_with),
        alpha_advection_form=str(alpha_advection_form),
        phi_b=float(phi_b),
        enable_phi_evolution=bool(enable_phi_evolution),
        alpha_supg=float(problem.get("_audit_alpha_supg", 0.0)),
    )

    form_r_fx = dolfinx.fem.form(r_fx)
    vec_fx = dolfinx.fem.petsc.assemble_vector(form_r_fx)
    r_fx_arr = vec_fx.array.copy()
    form_a_fx = dolfinx.fem.form(a_fx)
    A_fx = dolfinx.fem.petsc.assemble_matrix(form_a_fx)
    A_fx.assemble()
    indptr, indices, data = A_fx.getValuesCSR()
    J_fx = csr_matrix((data, indices, indptr), shape=A_fx.getSize()).tocsr()

    dh = problem["dh"]
    alpha_pc = np.asarray(dh.get_field_slice("alpha"), dtype=int)
    cols_pc_parts = [
        np.asarray(dh.get_field_slice("v_x"), dtype=int),
        np.asarray(dh.get_field_slice("v_y"), dtype=int),
        np.asarray(dh.get_field_slice("vS_x"), dtype=int),
        np.asarray(dh.get_field_slice("vS_y"), dtype=int),
        np.asarray(dh.get_field_slice("alpha"), dtype=int),
    ]
    if bool(enable_phi_evolution):
        cols_pc_parts.append(np.asarray(dh.get_field_slice("phi"), dtype=int))
    cols_pc = np.concatenate(cols_pc_parts)
    alpha_fx = np.asarray(mapping["alpha"], dtype=int)
    cols_fx_parts = [
        np.asarray(mapping["v_x"], dtype=int),
        np.asarray(mapping["v_y"], dtype=int),
        np.asarray(mapping["vS_x"], dtype=int),
        np.asarray(mapping["vS_y"], dtype=int),
        np.asarray(mapping["alpha"], dtype=int),
    ]
    if bool(enable_phi_evolution):
        cols_fx_parts.append(np.asarray(mapping["phi"], dtype=int))
    cols_fx = np.concatenate(cols_fx_parts)

    _, r_pc = assemble_form(Equation(None, forms.r_alpha), dof_handler=dh, bcs=[], quad_order=int(qdeg), backend=str(backend))
    J_pc, _ = assemble_form(Equation(forms.a_alpha, None), dof_handler=dh, bcs=[], quad_order=int(qdeg), backend=str(backend))
    r_pc = np.asarray(r_pc, dtype=float)[alpha_pc]
    J_pc = J_pc.tocsr()[alpha_pc, :][:, cols_pc].toarray()
    r_fx = np.asarray(r_fx_arr, dtype=float)[alpha_fx]
    J_fx = J_fx[alpha_fx, :][:, cols_fx].toarray()

    res_diff = r_pc - r_fx
    jac_diff = J_pc - J_fx
    res_denom = max(1.0, float(np.linalg.norm(r_pc, ord=np.inf)), float(np.linalg.norm(r_fx, ord=np.inf)))
    jac_denom = max(1.0, float(np.linalg.norm(J_pc, ord=np.inf)), float(np.linalg.norm(J_fx, ord=np.inf)))
    return (
        float(np.max(np.abs(res_diff))),
        float(np.linalg.norm(res_diff, ord=np.inf) / res_denom),
        float(np.max(np.abs(jac_diff))),
        float(np.linalg.norm(jac_diff, ord=np.inf) / jac_denom),
    )


def _pycutfem_phi_compare(problem, forms, *, qdeg: int, backend: str) -> tuple[float | None, float | None, float | None, float | None]:
    enable_phi_evolution = bool(problem["_audit_enable_phi_evolution"])
    if not bool(enable_phi_evolution):
        return None, None, None, None

    support_key = str(problem.get("_audit_support_physics", "legacy_exchange")).strip().lower()
    if support_key != "internal_conversion":
        return None, None, None, None
    mesh_fx, W_fx = _build_fenics_alpha_system(
        Lx=float(problem["_audit_Lx"]),
        Ly=float(problem["_audit_Ly"]),
        nx=int(problem["_audit_nx"]),
        ny=int(problem["_audit_ny"]),
        poly_order=int(problem["_audit_poly_order"]),
        scalar_order=int(problem["_audit_scalar_order"]),
        enable_phi_evolution=True,
    )
    mapping = _map_pycutfem_to_fenics_alpha(problem, W_fx, enable_phi_evolution=True)
    w_k_fx, w_n_fx = _load_fenics_state(problem, W_fx, mapping, enable_phi_evolution=True)
    r_fx, a_fx = _fenics_phi_forms(
        W=W_fx,
        w_k=w_k_fx,
        w_n=w_n_fx,
        dt=float(problem["_audit_dt"]),
        theta=float(problem["_audit_theta"]),
        qdeg=int(qdeg),
        support_physics=str(problem["_audit_support_physics"]),
        D_phi=float(problem["_audit_D_phi"]),
        phi_diffusion_weight=str(problem["_audit_phi_diffusion_weight"]),
        gamma_phi=float(problem["_audit_gamma_phi"]),
        mu_max=float(problem["_audit_mu_max"]),
        K_S=float(problem["_audit_K_S"]),
        k_d=float(problem["_audit_k_d"]),
        phi_supg=float(problem.get("_audit_phi_supg", 0.0)),
        phi_cip=float(problem.get("_audit_phi_cip", 0.0)),
    )

    form_r_fx = dolfinx.fem.form(r_fx)
    vec_fx = dolfinx.fem.petsc.assemble_vector(form_r_fx)
    r_fx_arr = vec_fx.array.copy()
    form_a_fx = dolfinx.fem.form(a_fx)
    A_fx = dolfinx.fem.petsc.assemble_matrix(form_a_fx)
    A_fx.assemble()
    indptr, indices, data = A_fx.getValuesCSR()
    J_fx = csr_matrix((data, indices, indptr), shape=A_fx.getSize()).tocsr()

    dh = problem["dh"]
    phi_pc = np.asarray(dh.get_field_slice("phi"), dtype=int)
    cols_pc = np.concatenate(
        [
            np.asarray(dh.get_field_slice("vS_x"), dtype=int),
            np.asarray(dh.get_field_slice("vS_y"), dtype=int),
            np.asarray(dh.get_field_slice("alpha"), dtype=int),
            np.asarray(dh.get_field_slice("phi"), dtype=int),
        ]
    )
    phi_fx = np.asarray(mapping["phi"], dtype=int)
    cols_fx = np.concatenate(
        [
            np.asarray(mapping["vS_x"], dtype=int),
            np.asarray(mapping["vS_y"], dtype=int),
            np.asarray(mapping["alpha"], dtype=int),
            np.asarray(mapping["phi"], dtype=int),
        ]
    )

    _, r_pc = assemble_form(Equation(None, forms.r_phi), dof_handler=dh, bcs=[], quad_order=int(qdeg), backend=str(backend))
    J_pc, _ = assemble_form(Equation(forms.a_phi, None), dof_handler=dh, bcs=[], quad_order=int(qdeg), backend=str(backend))
    r_pc = np.asarray(r_pc, dtype=float)[phi_pc]
    J_pc = J_pc.tocsr()[phi_pc, :][:, cols_pc].toarray()
    r_fx = np.asarray(r_fx_arr, dtype=float)[phi_fx]
    J_fx = J_fx[phi_fx, :][:, cols_fx].toarray()

    res_diff = r_pc - r_fx
    jac_diff = J_pc - J_fx
    res_denom = max(1.0, float(np.linalg.norm(r_pc, ord=np.inf)), float(np.linalg.norm(r_fx, ord=np.inf)))
    jac_denom = max(1.0, float(np.linalg.norm(J_pc, ord=np.inf)), float(np.linalg.norm(J_fx, ord=np.inf)))
    return (
        float(np.max(np.abs(res_diff))),
        float(np.linalg.norm(res_diff, ord=np.inf) / res_denom),
        float(np.max(np.abs(jac_diff))),
        float(np.linalg.norm(jac_diff, ord=np.inf) / jac_denom),
    )


def _build_fenics_skeleton_pressure_system(
    *,
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    poly_order: int,
    pressure_order: int,
    scalar_order: int,
    enable_phi_evolution: bool,
):
    mesh_fx = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0], dtype=float), np.array([float(Lx), float(Ly)], dtype=float)],
        [int(nx), int(ny)],
        cell_type=dolfinx.mesh.CellType.quadrilateral,
    )
    gdim = mesh_fx.geometry.dim
    p_family = "DQ" if int(pressure_order) == 0 else "Lagrange"
    s_family = "DQ" if int(scalar_order) == 0 else "Lagrange"
    Pp = basix.ufl.element(p_family, "quadrilateral", int(pressure_order))
    V_el = basix.ufl.element("Lagrange", "quadrilateral", int(poly_order), shape=(gdim,))
    A_el = basix.ufl.element(s_family, "quadrilateral", int(scalar_order))
    W_el = mixed_element([Pp, V_el, A_el, A_el]) if bool(enable_phi_evolution) else mixed_element([Pp, V_el, A_el])
    if hasattr(dolfinx.fem, "functionspace"):
        W = dolfinx.fem.functionspace(mesh_fx, W_el)
    else:
        W = dolfinx.fem.FunctionSpace(mesh_fx, W_el)
    return mesh_fx, W


def _map_pycutfem_to_fenics_skeleton_pressure(problem, W, *, enable_phi_evolution: bool) -> dict[str, np.ndarray]:
    dh = problem["dh"]
    mapping: dict[str, np.ndarray] = {}
    parent_p, coords_p = _fenics_parent_dofs_for_scalar(W, 0)
    mapping["p"] = parent_p[_one_to_one_map_coords(np.asarray(dh.get_dof_coords("p"), dtype=float), coords_p)]
    for fld, comp in (("vS_x", 0), ("vS_y", 1)):
        parent, coords_fx = _fenics_parent_dofs_for_component(W, 1, comp)
        mapping[fld] = parent[_one_to_one_map_coords(np.asarray(dh.get_dof_coords(fld), dtype=float), coords_fx)]
    parent_alpha, coords_alpha = _fenics_parent_dofs_for_scalar(W, 2)
    mapping["alpha"] = parent_alpha[_one_to_one_map_coords(np.asarray(dh.get_dof_coords("alpha"), dtype=float), coords_alpha)]
    if bool(enable_phi_evolution):
        parent_phi, coords_phi = _fenics_parent_dofs_for_scalar(W, 3)
        mapping["phi"] = parent_phi[_one_to_one_map_coords(np.asarray(dh.get_dof_coords("phi"), dtype=float), coords_phi)]
    return mapping


def _load_fenics_skeleton_pressure_state(problem, W, mapping: dict[str, np.ndarray], *, enable_phi_evolution: bool):
    w_k = dolfinx.fem.Function(W, name="w_k")
    w_n = dolfinx.fem.Function(W, name="w_n")
    fields_k = [
        ("p", problem["p_k"]),
        ("vS_x", problem["vS_k"].components[0]),
        ("vS_y", problem["vS_k"].components[1]),
        ("alpha", problem["alpha_k"]),
    ]
    fields_n = [
        ("p", problem["p_n"]),
        ("vS_x", problem["vS_n"].components[0]),
        ("vS_y", problem["vS_n"].components[1]),
        ("alpha", problem["alpha_n"]),
    ]
    if bool(enable_phi_evolution):
        fields_k.append(("phi", problem["phi_k"]))
        fields_n.append(("phi", problem["phi_n"]))
    for fld, func in fields_k:
        w_k.x.array[mapping[fld]] = np.asarray(func.nodal_values, dtype=float)
    for fld, func in fields_n:
        w_n.x.array[mapping[fld]] = np.asarray(func.nodal_values, dtype=float)
    w_k.x.scatter_forward()
    w_n.x.scatter_forward()
    return w_k, w_n


def _fenics_skeleton_pressure_forms(
    *,
    W,
    w_k,
    w_n,
    theta: float,
    qdeg: int,
    phi_b: float,
    enable_phi_evolution: bool,
    include_skeleton_acceleration: bool,
    skeleton_pressure_mode: str,
    alpha_biot: float | None,
):
    dw = ufl.TrialFunction(W)
    wtest = ufl.TestFunction(W)
    if bool(enable_phi_evolution):
        dp_fx, dvS_fx, dalpha_fx, dphi_fx = ufl.split(dw)
        q_fx, vS_test_fx, alpha_test_fx, phi_test_fx = ufl.split(wtest)
        p_k_fx, vS_k_fx, alpha_k_fx, phi_k_fx = ufl.split(w_k)
        p_n_fx, vS_n_fx, alpha_n_fx, phi_n_fx = ufl.split(w_n)
    else:
        dp_fx, dvS_fx, dalpha_fx = ufl.split(dw)
        q_fx, vS_test_fx, alpha_test_fx = ufl.split(wtest)
        p_k_fx, vS_k_fx, alpha_k_fx = ufl.split(w_k)
        p_n_fx, vS_n_fx, alpha_n_fx = ufl.split(w_n)
        dphi_fx = None
        phi_k_fx = None
        phi_n_fx = None
    del q_fx, alpha_test_fx, dvS_fx

    th_fx = dolfinx.fem.Constant(W.mesh, float(theta))
    omth_fx = dolfinx.fem.Constant(W.mesh, 1.0 - float(theta))
    one_fx = dolfinx.fem.Constant(W.mesh, 1.0)
    zero_fx = dolfinx.fem.Constant(W.mesh, 0.0)
    dx_fx = ufl.dx(metadata={"quadrature_degree": int(qdeg)})

    if bool(enable_phi_evolution):
        B_k_fx = alpha_k_fx * (1.0 - phi_k_fx)
        B_n_fx = alpha_n_fx * (1.0 - phi_n_fx)
        gradB_k_fx = (1.0 - phi_k_fx) * ufl.grad(alpha_k_fx) - alpha_k_fx * ufl.grad(phi_k_fx)
        gradB_n_fx = (1.0 - phi_n_fx) * ufl.grad(alpha_n_fx) - alpha_n_fx * ufl.grad(phi_n_fx)
    else:
        phi_b_fx = dolfinx.fem.Constant(W.mesh, float(phi_b))
        B_k_fx = alpha_k_fx * (1.0 - phi_b_fx)
        B_n_fx = alpha_n_fx * (1.0 - phi_b_fx)
        gradB_k_fx = (1.0 - phi_b_fx) * ufl.grad(alpha_k_fx)
        gradB_n_fx = (1.0 - phi_b_fx) * ufl.grad(alpha_n_fx)

    div_B_vStest_k_fx = B_k_fx * ufl.div(vS_test_fx) + ufl.dot(gradB_k_fx, vS_test_fx)
    div_B_vStest_n_fx = B_n_fx * ufl.div(vS_test_fx) + ufl.dot(gradB_n_fx, vS_test_fx)
    skel_press_key = str(skeleton_pressure_mode).strip().lower().replace("-", "_")
    if skel_press_key not in {"whole_domain", "seboldt"}:
        raise ValueError(
            f"Unsupported skeleton_pressure_mode={skeleton_pressure_mode!r}. "
            "Use 'whole_domain' or 'seboldt'."
        )

    if skel_press_key == "seboldt":
        alpha_biot_fx = dolfinx.fem.Constant(W.mesh, float(1.0 if alpha_biot is None else alpha_biot))
        r_press_k_fx = -(p_k_fx * alpha_biot_fx * alpha_k_fx * ufl.div(vS_test_fx))
        r_press_n_fx = -(p_n_fx * alpha_biot_fx * alpha_n_fx * ufl.div(vS_test_fx))
        sk_th = th_fx if bool(include_skeleton_acceleration) else one_fx
        sk_omth = omth_fx if bool(include_skeleton_acceleration) else zero_fx
        r_press_fx = (sk_th * r_press_k_fx + sk_omth * r_press_n_fx) * dx_fx
    else:
        r_press_k_fx = -p_k_fx * div_B_vStest_k_fx
        r_press_n_fx = -p_n_fx * div_B_vStest_n_fx
        if alpha_biot is not None:
            alpha_biot_fx = dolfinx.fem.Constant(W.mesh, float(alpha_biot))
            r_press_k_fx += -(p_k_fx * (alpha_biot_fx * alpha_k_fx - B_k_fx) * ufl.div(vS_test_fx))
            r_press_n_fx += -(p_n_fx * (alpha_biot_fx * alpha_n_fx - B_n_fx) * ufl.div(vS_test_fx))
        sk_th = th_fx if bool(include_skeleton_acceleration) else one_fx
        sk_omth = omth_fx if bool(include_skeleton_acceleration) else zero_fx
        r_press_fx = (sk_th * r_press_k_fx + sk_omth * r_press_n_fx) * dx_fx

    a_press_fx = ufl.derivative(r_press_fx, w_k, dw)
    return r_press_fx, a_press_fx


def _pycutfem_skeleton_pressure_compare(problem, forms, *, qdeg: int, backend: str) -> tuple[float | None, float | None, float | None, float | None]:
    r_pc_form = getattr(forms, "r_skeleton_pressure", None)
    a_pc_form = getattr(forms, "a_skeleton_pressure", None)
    if r_pc_form is None or a_pc_form is None:
        return None, None, None, None

    enable_phi_evolution = bool(problem["_audit_enable_phi_evolution"])
    mesh_fx, W_fx = _build_fenics_skeleton_pressure_system(
        Lx=float(problem["_audit_Lx"]),
        Ly=float(problem["_audit_Ly"]),
        nx=int(problem["_audit_nx"]),
        ny=int(problem["_audit_ny"]),
        poly_order=int(problem["_audit_poly_order"]),
        pressure_order=int(problem["_audit_pressure_order"]),
        scalar_order=int(problem["_audit_scalar_order"]),
        enable_phi_evolution=bool(enable_phi_evolution),
    )
    mapping = _map_pycutfem_to_fenics_skeleton_pressure(problem, W_fx, enable_phi_evolution=bool(enable_phi_evolution))
    w_k_fx, w_n_fx = _load_fenics_skeleton_pressure_state(problem, W_fx, mapping, enable_phi_evolution=bool(enable_phi_evolution))
    r_fx, a_fx = _fenics_skeleton_pressure_forms(
        W=W_fx,
        w_k=w_k_fx,
        w_n=w_n_fx,
        theta=float(problem["_audit_theta"]),
        qdeg=int(qdeg),
        phi_b=float(problem["_audit_phi_b"]),
        enable_phi_evolution=bool(enable_phi_evolution),
        include_skeleton_acceleration=bool(problem.get("_audit_include_skeleton_acceleration", True)),
        skeleton_pressure_mode=str(problem.get("_audit_skeleton_pressure_mode", "whole_domain")),
        alpha_biot=problem.get("_audit_alpha_biot", None),
    )

    form_r_fx = dolfinx.fem.form(r_fx)
    vec_fx = dolfinx.fem.petsc.assemble_vector(form_r_fx)
    r_fx_arr = vec_fx.array.copy()
    form_a_fx = dolfinx.fem.form(a_fx)
    A_fx = dolfinx.fem.petsc.assemble_matrix(form_a_fx)
    A_fx.assemble()
    indptr, indices, data = A_fx.getValuesCSR()
    J_fx = csr_matrix((data, indices, indptr), shape=A_fx.getSize()).tocsr()

    dh = problem["dh"]
    rows_pc = np.concatenate(
        [
            np.asarray(dh.get_field_slice("vS_x"), dtype=int),
            np.asarray(dh.get_field_slice("vS_y"), dtype=int),
        ]
    )
    cols_pc_parts = [
        np.asarray(dh.get_field_slice("p"), dtype=int),
        np.asarray(dh.get_field_slice("vS_x"), dtype=int),
        np.asarray(dh.get_field_slice("vS_y"), dtype=int),
        np.asarray(dh.get_field_slice("alpha"), dtype=int),
    ]
    rows_fx = np.concatenate(
        [
            np.asarray(mapping["vS_x"], dtype=int),
            np.asarray(mapping["vS_y"], dtype=int),
        ]
    )
    cols_fx_parts = [
        np.asarray(mapping["p"], dtype=int),
        np.asarray(mapping["vS_x"], dtype=int),
        np.asarray(mapping["vS_y"], dtype=int),
        np.asarray(mapping["alpha"], dtype=int),
    ]
    if bool(enable_phi_evolution):
        cols_pc_parts.append(np.asarray(dh.get_field_slice("phi"), dtype=int))
        cols_fx_parts.append(np.asarray(mapping["phi"], dtype=int))
    cols_pc = np.concatenate(cols_pc_parts)
    cols_fx = np.concatenate(cols_fx_parts)

    _, r_pc = assemble_form(Equation(None, r_pc_form), dof_handler=dh, bcs=[], quad_order=int(qdeg), backend=str(backend))
    J_pc, _ = assemble_form(Equation(a_pc_form, None), dof_handler=dh, bcs=[], quad_order=int(qdeg), backend=str(backend))
    r_pc = np.asarray(r_pc, dtype=float)[rows_pc]
    J_pc = J_pc.tocsr()[rows_pc, :][:, cols_pc].toarray()
    r_fx = np.asarray(r_fx_arr, dtype=float)[rows_fx]
    J_fx = J_fx[rows_fx, :][:, cols_fx].toarray()

    res_diff = r_pc - r_fx
    jac_diff = J_pc - J_fx
    res_denom = max(1.0, float(np.linalg.norm(r_pc, ord=np.inf)), float(np.linalg.norm(r_fx, ord=np.inf)))
    jac_denom = max(1.0, float(np.linalg.norm(J_pc, ord=np.inf)), float(np.linalg.norm(J_fx, ord=np.inf)))
    return (
        float(np.max(np.abs(res_diff))),
        float(np.linalg.norm(res_diff, ord=np.inf) / res_denom),
        float(np.max(np.abs(jac_diff))),
        float(np.linalg.norm(jac_diff, ord=np.inf) / jac_denom),
    )


def _build_fenics_full_system(
    *,
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    poly_order: int,
    pressure_order: int,
    scalar_order: int,
    enable_phi_evolution: bool,
    latent_bounded_fields: tuple[str, ...] = tuple(),
    pressure_mean_constraint: bool = False,
    solid_volumetric_split: bool = False,
):
    mesh_fx = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0], dtype=float), np.array([float(Lx), float(Ly)], dtype=float)],
        [int(nx), int(ny)],
        cell_type=dolfinx.mesh.CellType.quadrilateral,
    )
    gdim = mesh_fx.geometry.dim
    p_family = "DQ" if int(pressure_order) == 0 else "Lagrange"
    s_family = "DQ" if int(scalar_order) == 0 else "Lagrange"
    V_el = basix.ufl.element("Lagrange", "quadrilateral", int(poly_order), shape=(gdim,))
    P_el = basix.ufl.element(p_family, "quadrilateral", int(pressure_order))
    A_el = basix.ufl.element(s_family, "quadrilateral", int(scalar_order))
    elements = [V_el, P_el]
    if bool(solid_volumetric_split):
        elements.append(P_el)
    elements.extend([V_el, V_el, A_el, A_el])
    if bool(enable_phi_evolution):
        elements.extend([A_el, A_el])
    latent_field_set = {str(name).strip() for name in tuple(latent_bounded_fields or tuple()) if str(name).strip()}
    if "alpha" in latent_field_set:
        elements.append(A_el)
    if bool(enable_phi_evolution) and "phi" in latent_field_set:
        elements.append(A_el)
    W_el = mixed_element(elements)
    if hasattr(dolfinx.fem, "functionspace"):
        W = dolfinx.fem.functionspace(mesh_fx, W_el)
    else:
        W = dolfinx.fem.FunctionSpace(mesh_fx, W_el)
    return mesh_fx, W


def _audit_field_layout(
    *,
    enable_phi_evolution: bool,
    latent_bounded_fields: tuple[str, ...] = tuple(),
    pressure_mean_constraint: bool = False,
    solid_volumetric_split: bool = False,
) -> list[tuple[str, int, int | None]]:
    layout: list[tuple[str, int, int | None]] = [
        ("v_x", 0, 0),
        ("v_y", 0, 1),
        ("p", 1, None),
    ]
    next_idx = 2
    if bool(pressure_mean_constraint):
        layout.append(("p_mean", next_idx, None))
        next_idx += 1
    if bool(solid_volumetric_split):
        layout.append(("pi_s", next_idx, None))
        next_idx += 1
    layout.extend(
        [
            ("vS_x", next_idx, 0),
            ("vS_y", next_idx, 1),
            ("u_x", next_idx + 1, 0),
            ("u_y", next_idx + 1, 1),
            ("alpha", next_idx + 2, None),
            ("mu_alpha", next_idx + 3, None),
        ]
    )
    if bool(enable_phi_evolution):
        layout.extend(
            [
                ("phi", next_idx + 4, None),
                ("S", next_idx + 5, None),
            ]
        )
    latent_field_set = {str(name).strip() for name in tuple(latent_bounded_fields or tuple()) if str(name).strip()}
    next_idx = next_idx + (6 if bool(enable_phi_evolution) else 4)
    if "alpha" in latent_field_set:
        layout.append(("alpha_latent", next_idx, None))
        next_idx += 1
    if bool(enable_phi_evolution) and "phi" in latent_field_set:
        layout.append(("phi_latent", next_idx, None))
    return layout


def _map_pycutfem_to_fenics_full(
    problem,
    W,
    *,
    enable_phi_evolution: bool,
    latent_bounded_fields: tuple[str, ...] = tuple(),
    pressure_mean_constraint: bool = False,
    solid_volumetric_split: bool = False,
) -> dict[str, np.ndarray]:
    dh = problem["dh"]
    mapping: dict[str, np.ndarray] = {}
    n_base = int(dolfinx.fem.Function(W).x.array.size)
    for fld, sub_idx, comp in _audit_field_layout(
        enable_phi_evolution=bool(enable_phi_evolution),
        latent_bounded_fields=tuple(latent_bounded_fields or tuple()),
        pressure_mean_constraint=bool(pressure_mean_constraint),
        solid_volumetric_split=bool(solid_volumetric_split),
    ):
        if fld == "p_mean":
            mapping[fld] = np.asarray([n_base], dtype=int)
            continue
        fx_sub_idx = int(sub_idx)
        if bool(pressure_mean_constraint) and fx_sub_idx > 1:
            fx_sub_idx -= 1
        if comp is None:
            parent, coords_fx = _fenics_parent_dofs_for_scalar(W, fx_sub_idx)
        else:
            parent, coords_fx = _fenics_parent_dofs_for_component(W, fx_sub_idx, comp)
        coords_pc = np.asarray(dh.get_dof_coords(fld), dtype=float)
        mapping[fld] = parent[_one_to_one_map_coords(coords_pc, coords_fx)]
    return mapping


def _load_fenics_full_state(
    problem,
    W,
    mapping: dict[str, np.ndarray],
    *,
    enable_phi_evolution: bool,
    latent_bounded_fields: tuple[str, ...] = tuple(),
    pressure_mean_constraint: bool = False,
    solid_volumetric_split: bool = False,
):
    w_k = dolfinx.fem.Function(W, name="w_k")
    w_n = dolfinx.fem.Function(W, name="w_n")
    fields_k = [
        ("v_x", problem["v_k"].components[0]),
        ("v_y", problem["v_k"].components[1]),
        ("p", problem["p_k"]),
    ]
    fields_n = [
        ("v_x", problem["v_n"].components[0]),
        ("v_y", problem["v_n"].components[1]),
        ("p", problem["p_n"]),
    ]
    if bool(solid_volumetric_split):
        fields_k.append(("pi_s", problem["pi_s_k"]))
        fields_n.append(("pi_s", problem["pi_s_n"]))
    fields_k.extend(
        [
            ("vS_x", problem["vS_k"].components[0]),
            ("vS_y", problem["vS_k"].components[1]),
            ("u_x", problem["u_k"].components[0]),
            ("u_y", problem["u_k"].components[1]),
            ("alpha", problem["alpha_k"]),
            ("mu_alpha", problem["mu_k"]),
        ]
    )
    fields_n.extend(
        [
        ("vS_x", problem["vS_n"].components[0]),
        ("vS_y", problem["vS_n"].components[1]),
        ("u_x", problem["u_n"].components[0]),
        ("u_y", problem["u_n"].components[1]),
        ("alpha", problem["alpha_n"]),
        ("mu_alpha", problem["mu_n"]),
        ]
    )
    if bool(enable_phi_evolution):
        fields_k.extend(
            [
                ("phi", problem["phi_k"]),
                ("S", problem["S_k"]),
            ]
        )
        fields_n.extend(
            [
                ("phi", problem["phi_n"]),
                ("S", problem["S_n"]),
            ]
        )
    latent_field_set = {str(name).strip() for name in tuple(latent_bounded_fields or tuple()) if str(name).strip()}
    if "alpha" in latent_field_set:
        fields_k.append(("alpha_latent", problem["alpha_latent_k"]))
        fields_n.append(("alpha_latent", problem["alpha_latent_n"]))
    if bool(enable_phi_evolution) and "phi" in latent_field_set:
        fields_k.append(("phi_latent", problem["phi_latent_k"]))
        fields_n.append(("phi_latent", problem["phi_latent_n"]))
    for fld, func in fields_k:
        w_k.x.array[mapping[fld]] = np.asarray(func.nodal_values, dtype=float)
    for fld, func in fields_n:
        w_n.x.array[mapping[fld]] = np.asarray(func.nodal_values, dtype=float)
    w_k.x.scatter_forward()
    w_n.x.scatter_forward()
    return w_k, w_n


def _fenics_full_forms(
    *,
    W,
    w_k,
    w_n,
    qdeg: int,
    enable_phi_evolution: bool,
    params: dict[str, object],
):
    mesh = W.mesh
    gdim = mesh.geometry.dim
    dw = ufl.TrialFunction(W)
    wtest = ufl.TestFunction(W)
    latent_field_set = {str(name).strip() for name in tuple(params.get("latent_bounded_fields", tuple()) or tuple()) if str(name).strip()}
    use_alpha_latent = bool(params.get("latent_bounded_transport", False)) and "alpha" in latent_field_set
    use_phi_latent = bool(enable_phi_evolution) and bool(params.get("latent_bounded_transport", False)) and "phi" in latent_field_set
    latent_map_kind = str(params.get("latent_bounded_map", "sigmoid")).strip().lower()
    latent_formulation_key = str(params.get("latent_bounded_formulation", "embedded")).strip().lower()
    solid_volumetric_split = bool(params.get("solid_volumetric_split", False))

    dw_parts = list(ufl.split(dw))
    test_parts = list(ufl.split(wtest))
    wk_parts = list(ufl.split(w_k))
    wn_parts = list(ufl.split(w_n))
    idx = 0
    dv_fx = dw_parts[idx]; v_test_fx = test_parts[idx]; v_k_fx = wk_parts[idx]; v_n_fx = wn_parts[idx]; idx += 1
    dp_fx = dw_parts[idx]; q_fx = test_parts[idx]; p_k_fx = wk_parts[idx]; p_n_fx = wn_parts[idx]; idx += 1
    dpi_s_fx = None
    pi_s_test_fx = None
    pi_s_k_fx = None
    pi_s_n_fx = None
    if bool(solid_volumetric_split):
        dpi_s_fx = dw_parts[idx]; pi_s_test_fx = test_parts[idx]; pi_s_k_fx = wk_parts[idx]; pi_s_n_fx = wn_parts[idx]; idx += 1
    dvS_fx = dw_parts[idx]; vS_test_fx = test_parts[idx]; vS_k_fx = wk_parts[idx]; vS_n_fx = wn_parts[idx]; idx += 1
    du_fx = dw_parts[idx]; u_test_fx = test_parts[idx]; u_k_fx = wk_parts[idx]; u_n_fx = wn_parts[idx]; idx += 1
    dalpha_fx = dw_parts[idx]; alpha_test_fx = test_parts[idx]; alpha_k_fx = wk_parts[idx]; alpha_n_fx = wn_parts[idx]; idx += 1
    dmu_alpha_fx = dw_parts[idx]; mu_alpha_test_fx = test_parts[idx]; mu_alpha_k_fx = wk_parts[idx]; mu_alpha_n_fx = wn_parts[idx]; idx += 1
    dphi_fx = None
    dS_fx = None
    phi_test_fx = None
    S_test_fx = None
    phi_k_fx = None
    phi_n_fx = None
    S_k_fx = None
    S_n_fx = None
    if bool(enable_phi_evolution):
        dphi_fx = dw_parts[idx]; phi_test_fx = test_parts[idx]; phi_k_fx = wk_parts[idx]; phi_n_fx = wn_parts[idx]; idx += 1
        dS_fx = dw_parts[idx]; S_test_fx = test_parts[idx]; S_k_fx = wk_parts[idx]; S_n_fx = wn_parts[idx]; idx += 1
    dalpha_latent_fx = None
    alpha_latent_test_fx = None
    alpha_latent_k_fx = None
    alpha_latent_n_fx = None
    if bool(use_alpha_latent):
        dalpha_latent_fx = dw_parts[idx]; alpha_latent_test_fx = test_parts[idx]; alpha_latent_k_fx = wk_parts[idx]; alpha_latent_n_fx = wn_parts[idx]; idx += 1
    dphi_latent_fx = None
    phi_latent_test_fx = None
    phi_latent_k_fx = None
    phi_latent_n_fx = None
    if bool(use_phi_latent):
        dphi_latent_fx = dw_parts[idx]; phi_latent_test_fx = test_parts[idx]; phi_latent_k_fx = wk_parts[idx]; phi_latent_n_fx = wn_parts[idx]; idx += 1

    alpha_transport_test_fx = alpha_latent_test_fx if bool(use_alpha_latent) and alpha_latent_test_fx is not None else alpha_test_fx
    phi_transport_test_fx = phi_latent_test_fx if bool(use_phi_latent) and phi_latent_test_fx is not None else phi_test_fx
    if bool(use_alpha_latent) and latent_formulation_key == "transformed":
        alpha_k_fx = _latent_map_expr_ufl(alpha_latent_k_fx, map_kind=latent_map_kind)
        alpha_n_fx = _latent_map_expr_ufl(alpha_latent_n_fx, map_kind=latent_map_kind)
        dalpha_fx = _latent_map_prime_expr_ufl(alpha_latent_k_fx, map_kind=latent_map_kind) * dalpha_latent_fx
    if bool(use_phi_latent) and latent_formulation_key == "transformed":
        phi_k_fx = _latent_map_expr_ufl(phi_latent_k_fx, map_kind=latent_map_kind)
        phi_n_fx = _latent_map_expr_ufl(phi_latent_n_fx, map_kind=latent_map_kind)
        dphi_fx = _latent_map_prime_expr_ufl(phi_latent_k_fx, map_kind=latent_map_kind) * dphi_latent_fx

    def _const(val: float):
        return dolfinx.fem.Constant(mesh, float(val))

    def _eps(v):
        return ufl.sym(ufl.grad(v))

    def _capacity(alpha, phi):
        return 1.0 - alpha + alpha * phi

    def _mu(alpha, phi):
        mu_f_fx = _const(float(params["mu_f"]))
        mu_b_fx = _const(float(params["mu_b"]))
        mu_b_key = str(params["mu_b_model"]).strip().lower()
        if mu_b_key in {"mu", "const", "constant"}:
            mu_b_eff = mu_f_fx
        elif mu_b_key in {"phi_mu", "phi*mu", "phi"}:
            mu_b_eff = phi * mu_f_fx
        elif mu_b_key in {"alpha_mu", "alpha", "mu_b", "biofilm_mu", "biofilm"}:
            mu_b_eff = mu_b_fx
        elif mu_b_key in {"alpha_phi_mu", "alpha_phi", "alpha*phi_mu", "alpha*phi*mu"}:
            mu_b_eff = phi * mu_b_fx
        else:
            raise ValueError(f"Unsupported mu_b_model={params['mu_b_model']!r}.")
        return (1.0 - alpha) * mu_f_fx + alpha * mu_b_eff

    def _phi_state(phi_expr, phi_b_val: float):
        return phi_expr if bool(enable_phi_evolution) else _const(float(phi_b_val))

    th_fx = _const(float(params["theta"]))
    omth_fx = _const(1.0 - float(params["theta"]))
    inv_dt_fx = _const(1.0 / float(params["dt"]))
    one_fx = _const(1.0)
    zero_fx = _const(0.0)
    rho_f_fx = _const(float(params["rho_f"]))
    mu_f_fx = _const(float(params["mu_f"]))
    kappa_inv_fx = _const(float(params["kappa_inv"]))
    mu_s_fx = _const(float(params["mu_s"]))
    lambda_s_fx = _const(float(params["lambda_s"]))
    phi_b_fx = _const(float(params["phi_b"]))
    gamma_div_fx = _const(float(params["gamma_div"]))
    gamma_u_fx = _const(float(params["gamma_u"]))
    gamma_u_pin_fx = _const(float(params["gamma_u_pin"]))
    gamma_phi_fx = _const(float(params["gamma_phi"]))
    M_alpha_fx = _const(float(params["M_alpha"]))
    gamma_alpha_fx = _const(float(params["gamma_alpha"]))
    eps_alpha_fx = _const(float(params["eps_alpha"]))
    rho_s0_fx = _const(float(params["rho_s0_tilde"]))
    alpha_mu_aux_pin_fx = _const(float(params["alpha_mu_aux_pin"]))
    D_phi_fx = _const(float(params["D_phi"]))
    D_S_fx = _const(float(params["D_S"]))
    mu_max_fx = _const(float(params["mu_max"]))
    K_S_fx = _const(float(params["K_S"]))
    k_d_fx = _const(float(params["k_d"]))
    Y_fx = _const(float(params["Y"]))
    rho_s_star_fx = _const(float(params["rho_s_star"]))
    I_fx = ufl.Identity(mesh.geometry.dim)

    dx_fx = ufl.dx(metadata={"quadrature_degree": int(qdeg)})
    dS_meas_fx = ufl.dS(metadata={"quadrature_degree": int(qdeg)})
    n_int_fx = ufl.FacetNormal(mesh)
    h_cell_fx = _fx_mesh_size(mesh)
    h_face_fx = ufl.avg(h_cell_fx)
    inv_h2_fx = 1.0 / (h_cell_fx * h_cell_fx)
    use_diffuse_traction = bool(params.get("use_diffuse_traction", False))
    if bool(use_diffuse_traction):
        traction_exprs_fx = _fenics_audit_diffuse_traction_exprs(mesh)
        g_t_k_fx = traction_exprs_fx["g_t_k"]
        g_t_n_fx = traction_exprs_fx["g_t_n"]
        traction_weight_k_fx = traction_exprs_fx["traction_weight_k"]
        traction_weight_n_fx = traction_exprs_fx["traction_weight_n"]
    else:
        g_t_k_fx = None
        g_t_n_fx = None
        traction_weight_k_fx = None
        traction_weight_n_fx = None

    phi_state_k = _phi_state(phi_k_fx, float(params["phi_b"]))
    phi_state_n = _phi_state(phi_n_fx, float(params["phi_b"]))

    C_k_fx = _capacity(alpha_k_fx, phi_state_k)
    C_n_fx = _capacity(alpha_n_fx, phi_state_n)
    B_k_fx = alpha_k_fx * (1.0 - phi_state_k)
    B_n_fx = alpha_n_fx * (1.0 - phi_state_n)
    gradC_k_fx = (phi_state_k - 1.0) * ufl.grad(alpha_k_fx)
    gradC_n_fx = (phi_state_n - 1.0) * ufl.grad(alpha_n_fx)
    gradB_k_fx = (1.0 - phi_state_k) * ufl.grad(alpha_k_fx)
    gradB_n_fx = (1.0 - phi_state_n) * ufl.grad(alpha_n_fx)
    if bool(enable_phi_evolution):
        gradC_k_fx += alpha_k_fx * ufl.grad(phi_k_fx)
        gradC_n_fx += alpha_n_fx * ufl.grad(phi_n_fx)
        gradB_k_fx += -alpha_k_fx * ufl.grad(phi_k_fx)
        gradB_n_fx += -alpha_n_fx * ufl.grad(phi_n_fx)

    divCv_k_fx = C_k_fx * ufl.div(v_k_fx) + ufl.dot(gradC_k_fx, v_k_fx)
    divCv_n_fx = C_n_fx * ufl.div(v_n_fx) + ufl.dot(gradC_n_fx, v_n_fx)
    divBvS_k_fx = B_k_fx * ufl.div(vS_k_fx) + ufl.dot(gradB_k_fx, vS_k_fx)
    divBvS_n_fx = B_n_fx * ufl.div(vS_n_fx) + ufl.dot(gradB_n_fx, vS_n_fx)
    divF_k_fx = divCv_k_fx + divBvS_k_fx
    divF_n_fx = divCv_n_fx + divBvS_n_fx

    rho_k_fx = rho_f_fx * C_k_fx
    rho_n_fx = rho_f_fx * C_n_fx
    mu_k_fx = _mu(alpha_k_fx, phi_state_k)
    mu_n_fx = _mu(alpha_n_fx, phi_state_n)

    support_key = str(params["support_physics"]).strip().lower()
    if support_key == "internal_conversion":
        drag_occ_k_fx = alpha_k_fx * (1.0 - phi_state_k)
        drag_occ_n_fx = alpha_n_fx * (1.0 - phi_state_n)
        drag_phi_factor_k_fx = phi_state_k * phi_state_k
        drag_phi_factor_n_fx = phi_state_n * phi_state_n
    else:
        drag_occ_k_fx = alpha_k_fx
        drag_occ_n_fx = alpha_n_fx
        # The reduced fixed-phi audit path is built through
        # `deformation_only.py`, whose legacy drag law omits the extra
        # `phi_b**2` factor used by the full evolving-phi one-domain model.
        # Mirror that reduced-model law here so `enable_phi_evolution=False`
        # compares against the actual pycutfem branch rather than the full
        # one-domain system with phi frozen by hand.
        if bool(enable_phi_evolution):
            drag_phi_factor_k_fx = phi_state_k * phi_state_k
            drag_phi_factor_n_fx = phi_state_n * phi_state_n
        else:
            drag_phi_factor_k_fx = 1.0
            drag_phi_factor_n_fx = 1.0
    beta_coeff_k_fx = drag_occ_k_fx * mu_f_fx * drag_phi_factor_k_fx
    beta_coeff_n_fx = drag_occ_n_fx * mu_f_fx * drag_phi_factor_n_fx
    kappa_key = str(params["kappa_inv_model"]).strip().lower()
    if kappa_key in {"refmap", "eulerian_refmap", "eulerian", "reference_map", "reference-map"}:
        K_inv_ref_fx = kappa_inv_fx * I_fx
        F_inv_k_fx = I_fx - ufl.grad(u_k_fx)
        F_inv_n_fx = I_fx - ufl.grad(u_n_fx)
        F_k_fx = ufl.inv(F_inv_k_fx)
        F_n_fx = ufl.inv(F_inv_n_fx)
        J_k_fx = ufl.det(F_k_fx)
        J_n_fx = ufl.det(F_n_fx)
        k_inv_k_fx = J_k_fx * ufl.dot(F_inv_k_fx.T, ufl.dot(K_inv_ref_fx, F_inv_k_fx))
        k_inv_n_fx = J_n_fx * ufl.dot(F_inv_n_fx.T, ufl.dot(K_inv_ref_fx, F_inv_n_fx))
        drag_mode = "matrix"
    elif kappa_key in {"spatial", "constant", "const"}:
        beta_k_fx = beta_coeff_k_fx * kappa_inv_fx
        beta_n_fx = beta_coeff_n_fx * kappa_inv_fx
        drag_mode = "scalar"
    else:
        raise ValueError(f"Unsupported kappa_inv_model={params['kappa_inv_model']!r}.")

    div_C_vtest_k_fx = C_k_fx * ufl.div(v_test_fx) + ufl.dot(gradC_k_fx, v_test_fx)
    div_B_vStest_k_fx = B_k_fx * ufl.div(vS_test_fx) + ufl.dot(gradB_k_fx, vS_test_fx)
    div_B_vStest_n_fx = B_n_fx * ufl.div(vS_test_fx) + ufl.dot(gradB_n_fx, vS_test_fx)

    r_mom_fx = ufl.inner((rho_k_fx * v_k_fx - rho_n_fx * v_n_fx) * inv_dt_fx, v_test_fx) * dx_fx
    fluid_conv_key = str(params["fluid_convection"]).strip().lower()
    div_rhov_k_fx = rho_f_fx * divCv_k_fx
    div_rhov_n_fx = rho_f_fx * divCv_n_fx
    if fluid_conv_key == "full":
        conv_k_fx = ufl.dot(ufl.dot(ufl.grad(v_k_fx), v_k_fx), v_test_fx)
        conv_n_fx = ufl.dot(ufl.dot(ufl.grad(v_n_fx), v_n_fx), v_test_fx)
        r_mom_fx += (
            th_fx * (rho_k_fx * conv_k_fx + div_rhov_k_fx * ufl.dot(v_k_fx, v_test_fx))
            + omth_fx * (rho_n_fx * conv_n_fx + div_rhov_n_fx * ufl.dot(v_n_fx, v_test_fx))
        ) * dx_fx
    elif fluid_conv_key == "lagged":
        conv_k_fx = ufl.dot(ufl.dot(ufl.grad(v_k_fx), v_n_fx), v_test_fx)
        conv_n_fx = ufl.dot(ufl.dot(ufl.grad(v_n_fx), v_n_fx), v_test_fx)
        r_mom_fx += (
            th_fx * (rho_n_fx * conv_k_fx + div_rhov_n_fx * ufl.dot(v_k_fx, v_test_fx))
            + omth_fx * (rho_n_fx * conv_n_fx + div_rhov_n_fx * ufl.dot(v_n_fx, v_test_fx))
        ) * dx_fx
    elif fluid_conv_key == "imex":
        conv_n_fx = ufl.dot(ufl.dot(ufl.grad(v_n_fx), v_n_fx), v_test_fx)
        r_mom_fx += (rho_n_fx * conv_n_fx + div_rhov_n_fx * ufl.dot(v_n_fx, v_test_fx)) * dx_fx
    elif fluid_conv_key != "off":
        raise ValueError(f"Unsupported fluid_convection={params['fluid_convection']!r}.")
    r_mom_fx += 2.0 * (th_fx * mu_k_fx * ufl.inner(_eps(v_k_fx), _eps(v_test_fx)) + omth_fx * mu_n_fx * ufl.inner(_eps(v_n_fx), _eps(v_test_fx))) * dx_fx
    r_mom_fx += -(p_k_fx * div_C_vtest_k_fx) * dx_fx
    if float(params["gamma_div"]) != 0.0:
        r_mom_fx += gamma_div_fx * divF_k_fx * div_C_vtest_k_fx * dx_fx
    if drag_mode == "scalar":
        r_mom_fx += beta_k_fx * ufl.dot(v_k_fx - vS_k_fx, v_test_fx) * dx_fx
    else:
        r_mom_fx += beta_coeff_k_fx * ufl.dot(ufl.dot(k_inv_k_fx, v_k_fx - vS_k_fx), v_test_fx) * dx_fx
    if bool(use_diffuse_traction):
        r_mom_fx += -(
            th_fx * traction_weight_k_fx * ufl.dot(g_t_k_fx, v_test_fx)
            + omth_fx * traction_weight_n_fx * ufl.dot(g_t_n_fx, v_test_fx)
        ) * dx_fx

    if float(params.get("v_supg", 0.0)) != 0.0:
        v_supg_fx = _const(float(params["v_supg"]))
        v_supg_mode = str(params.get("v_supg_mode", "streamline")).strip().lower()
        h_v_safe_fx = h_cell_fx + 1.0e-16
        h_v2_safe_fx = (h_cell_fx * h_cell_fx) + 1.0e-16
        if v_supg_mode in {"streamline", "weak", "legacy"}:
            vmag_n_fx = ufl.sqrt(ufl.dot(v_n_fx, v_n_fx) + 1.0e-12)
            rho_safe_n_fx = rho_n_fx + 1.0e-16
            nu_eff_n_fx = mu_n_fx / rho_safe_n_fx
            drag_rate_n_fx = (beta_n_fx if drag_mode == "scalar" else beta_coeff_n_fx) / rho_safe_n_fx
            tau_v_fx = v_supg_fx / ufl.sqrt(
                (2.0 * inv_dt_fx) ** 2
                + (2.0 * vmag_n_fx / h_v_safe_fx) ** 2
                + (float(params.get("v_supg_c_nu", 4.0)) * nu_eff_n_fx / h_v2_safe_fx) ** 2
                + drag_rate_n_fx * drag_rate_n_fx
                + 1.0e-16
            )
            w_v_fx = 1.0 - alpha_n_fx
            adv_v_k_fx = ufl.dot(ufl.grad(v_k_fx), v_n_fx)
            adv_w_fx = ufl.dot(ufl.grad(v_test_fx), v_n_fx)
            r_mom_fx += tau_v_fx * w_v_fx * ufl.inner(adv_v_k_fx, adv_w_fx) * dx_fx
        elif v_supg_mode in {"residual", "strong", "strong_residual", "strong-residual"}:
            vmag_k_fx = ufl.sqrt(ufl.dot(v_k_fx, v_k_fx) + 1.0e-12)
            rho_safe_k_fx = rho_k_fx + 1.0e-16
            nu_eff_k_fx = mu_k_fx / rho_safe_k_fx
            drag_rate_k_fx = (beta_k_fx if drag_mode == "scalar" else beta_coeff_k_fx) / rho_safe_k_fx
            tau_v_fx = v_supg_fx / ufl.sqrt(
                (2.0 * inv_dt_fx) ** 2
                + (2.0 * vmag_k_fx / h_v_safe_fx) ** 2
                + (float(params.get("v_supg_c_nu", 4.0)) * nu_eff_k_fx / h_v2_safe_fx) ** 2
                + drag_rate_k_fx * drag_rate_k_fx
                + 1.0e-16
            )
            w_v_fx = 1.0 - alpha_k_fx
            adv_w_fx = ufl.dot(ufl.grad(v_test_fx), v_k_fx)
            strong_mom_k_fx = (rho_k_fx * v_k_fx - rho_n_fx * v_n_fx) * inv_dt_fx
            if fluid_conv_key == "full":
                strong_mom_k_fx += rho_k_fx * ufl.dot(ufl.grad(v_k_fx), v_k_fx) + div_rhov_k_fx * v_k_fx
            elif fluid_conv_key == "lagged":
                strong_mom_k_fx += rho_n_fx * ufl.dot(ufl.grad(v_k_fx), v_n_fx) + div_rhov_n_fx * v_k_fx
            elif fluid_conv_key == "imex":
                strong_mom_k_fx += rho_n_fx * ufl.dot(ufl.grad(v_n_fx), v_n_fx) + div_rhov_n_fx * v_n_fx
            elif fluid_conv_key != "off":
                raise ValueError(f"Unsupported fluid_convection={params['fluid_convection']!r}.")
            strong_mom_k_fx += -ufl.div(2.0 * mu_k_fx * _eps(v_k_fx))
            strong_mom_k_fx += C_k_fx * ufl.grad(p_k_fx)
            if drag_mode == "scalar":
                strong_mom_k_fx += beta_k_fx * (v_k_fx - vS_k_fx)
            else:
                strong_mom_k_fx += beta_coeff_k_fx * ufl.dot(k_inv_k_fx, v_k_fx - vS_k_fx)
            if bool(use_diffuse_traction):
                strong_mom_k_fx += -(
                    th_fx * traction_weight_k_fx * g_t_k_fx
                    + omth_fx * traction_weight_n_fx * g_t_n_fx
                )
            r_mom_fx += tau_v_fx * w_v_fx * ufl.inner(strong_mom_k_fx, adv_w_fx) * dx_fx
        else:
            raise ValueError(
                f"Unsupported v_supg_mode={params.get('v_supg_mode')!r}. "
                "Use 'streamline' or 'residual'."
            )

    if float(params.get("v_cip", 0.0)) != 0.0:
        tau_v_cip_fx = _const(float(params["v_cip"])) * (h_face_fx * h_face_fx * h_face_fx) * inv_dt_fx
        r_mom_fx += tau_v_cip_fx * ufl.avg(1.0 - alpha_n_fx) * _fx_grad_inner_jump(v_k_fx, v_test_fx, n_int_fx, gdim) * dS_meas_fx

    r_mass_fx = q_fx * divF_k_fx * dx_fx

    def _linear_elastic(u_expr, test_expr):
        return 2.0 * mu_s_fx * ufl.inner(_eps(u_expr), _eps(test_expr)) + lambda_s_fx * ufl.div(u_expr) * ufl.div(test_expr)

    def _linear_deviatoric_elastic(u_expr, test_expr):
        eps_u = _eps(u_expr)
        eps_test = _eps(test_expr)
        dev_u = eps_u - (ufl.tr(eps_u) / float(gdim)) * I_fx
        dev_test = eps_test - (ufl.tr(eps_test) / float(gdim)) * I_fx
        return 2.0 * mu_s_fx * ufl.inner(dev_u, dev_test)

    fluid_scale_fx = 1.0
    skeleton_scale_fx = 1.0
    kin_scale_override_fx = None
    pressure_block_lift_scale_fx = 0.0
    mechanics_nondim_key = str(params.get("mechanics_nondim_mode", "legacy")).strip().lower()
    if mechanics_nondim_key in {"stress_balance", "condition_balanced"}:
        fluid_ref_val = max(1.0, abs(float(params["mu_f"])), abs(float(params["mu_b"])))
        solid_ref_val = max(1.0, abs(float(2.0 * float(params["mu_s"]) + float(gdim) * float(params["lambda_s"]))))
        fluid_scale_fx = 1.0 / fluid_ref_val
        skeleton_scale_fx = 1.0 / solid_ref_val
        kin_scale_override_fx = 1.0
    kinematic_setup = _condition_balanced_kinematic_setup(
        mechanics_nondim_mode=mechanics_nondim_key,
        mu_f=float(params["mu_f"]),
        kappa_inv=float(params["kappa_inv"]),
        gamma_u=float(params["gamma_u"]),
        u_extension_mode=str(params["u_extension_mode"]),
        gamma_u_pin=float(params["gamma_u_pin"]),
        gamma_vS=params["gamma_vS"],
        vS_extension_mode=params["vS_extension_mode"],
        gamma_vS_pin=params["gamma_vS_pin"],
    )
    gamma_u_eff = float(kinematic_setup["gamma_u"])
    u_ext_mode_eff = str(kinematic_setup["u_extension_mode"]).strip().lower()
    gamma_u_pin_eff = float(kinematic_setup["gamma_u_pin"])
    gamma_vS_eff = (
        float(kinematic_setup["gamma_vS"]) if kinematic_setup["gamma_vS"] is not None else None
    )
    vS_ext_mode_eff = (
        None
        if kinematic_setup["vS_extension_mode"] is None
        else str(kinematic_setup["vS_extension_mode"]).strip().lower()
    )
    gamma_vS_pin_eff = (
        float(kinematic_setup["gamma_vS_pin"]) if kinematic_setup["gamma_vS_pin"] is not None else None
    )
    if mechanics_nondim_key == "condition_balanced":
        kin_scale_override_fx = float(kinematic_setup["kinematics_scale"])
    gamma_u_fx = _const(gamma_u_eff)
    gamma_u_pin_fx = _const(gamma_u_pin_eff)
    fluid_scale_fx = _const(float(fluid_scale_fx))
    skeleton_scale_fx = _const(float(skeleton_scale_fx))
    total_pressure_ref_fx = _const(max(1.0, abs(float(2.0 * float(params["mu_s"]) + float(gdim) * float(params["lambda_s"])))))
    total_pressure_ref_inv_fx = 1.0 / total_pressure_ref_fx
    lambda_over_total_pressure_ref_fx = float(params["lambda_s"]) * total_pressure_ref_inv_fx

    sk_th_fx = th_fx if bool(params["include_skeleton_acceleration"]) else one_fx
    sk_omth_fx = omth_fx if bool(params["include_skeleton_acceleration"]) else zero_fx
    skel_press_key = str(params["skeleton_pressure_mode"]).strip().lower().replace("-", "_")
    if skel_press_key not in {"whole_domain", "seboldt"}:
        raise ValueError(
            f"Unsupported skeleton_pressure_mode={params['skeleton_pressure_mode']!r}. "
            "Use 'whole_domain' or 'seboldt'."
        )
    press_div_coeff_k_fx = None
    press_div_coeff_n_fx = None
    if skel_press_key == "seboldt":
        alpha_biot_fx = _const(1.0 if params["alpha_biot"] is None else float(params["alpha_biot"]))
        press_div_coeff_k_fx = alpha_biot_fx * alpha_k_fx
        press_div_coeff_n_fx = alpha_biot_fx * alpha_n_fx
        r_skel_press_k_fx = -(p_k_fx * (alpha_biot_fx * alpha_k_fx) * ufl.div(vS_test_fx))
        r_skel_press_n_fx = -(p_n_fx * (alpha_biot_fx * alpha_n_fx) * ufl.div(vS_test_fx))
    else:
        press_div_coeff_k_fx = B_k_fx
        press_div_coeff_n_fx = B_n_fx
        r_skel_press_k_fx = -p_k_fx * div_B_vStest_k_fx
        r_skel_press_n_fx = -p_n_fx * div_B_vStest_n_fx
        if params["alpha_biot"] is not None:
            alpha_biot_fx = _const(float(params["alpha_biot"]))
            press_div_coeff_k_fx = alpha_biot_fx * alpha_k_fx
            press_div_coeff_n_fx = alpha_biot_fx * alpha_n_fx
            r_skel_press_k_fx += -(p_k_fx * (alpha_biot_fx * alpha_k_fx - B_k_fx) * ufl.div(vS_test_fx))
            r_skel_press_n_fx += -(p_n_fx * (alpha_biot_fx * alpha_n_fx - B_n_fx) * ufl.div(vS_test_fx))
    if float(params["gamma_div"]) != 0.0:
        r_skel_press_k_fx += gamma_div_fx * divF_k_fx * div_B_vStest_k_fx
    if drag_mode == "scalar":
        r_skel_drag_k_fx = -beta_k_fx * ufl.dot(v_k_fx - vS_k_fx, vS_test_fx)
        r_skel_drag_n_fx = -beta_n_fx * ufl.dot(v_n_fx - vS_n_fx, vS_test_fx)
    else:
        r_skel_drag_k_fx = -beta_coeff_k_fx * ufl.dot(ufl.dot(k_inv_k_fx, v_k_fx - vS_k_fx), vS_test_fx)
        r_skel_drag_n_fx = -beta_coeff_n_fx * ufl.dot(ufl.dot(k_inv_n_fx, v_n_fx - vS_n_fx), vS_test_fx)
    if bool(solid_volumetric_split):
        r_elastic_k_fx = _linear_deviatoric_elastic(u_k_fx, vS_test_fx) + total_pressure_ref_fx * pi_s_k_fx * ufl.div(vS_test_fx)
        r_elastic_n_fx = _linear_deviatoric_elastic(u_n_fx, vS_test_fx) + total_pressure_ref_fx * pi_s_n_fx * ufl.div(vS_test_fx)
        if skel_press_key == "seboldt":
            r_skel_press_k_fx = 0.0
            r_skel_press_n_fx = 0.0
        else:
            r_skel_press_k_fx = -(p_k_fx * ufl.dot(gradB_k_fx, vS_test_fx))
            r_skel_press_n_fx = -(p_n_fx * ufl.dot(gradB_n_fx, vS_test_fx))
        r_volumetric_fx = (
            alpha_k_fx
            * pi_s_test_fx
            * (
                pi_s_k_fx
                - lambda_over_total_pressure_ref_fx * ufl.div(u_k_fx)
                + total_pressure_ref_inv_fx * press_div_coeff_k_fx * p_k_fx
            )
            + float(params.get("solid_volumetric_penalty", 1.0)) * (1.0 - alpha_k_fx) * pi_s_k_fx * pi_s_test_fx
        ) * dx_fx
        if float(pressure_block_lift_scale_fx) != 0.0:
            r_mass_fx += _const(float(pressure_block_lift_scale_fx)) * (
                alpha_k_fx
                * q_fx
                * (
                    pi_s_k_fx
                    - lambda_over_total_pressure_ref_fx * ufl.div(u_k_fx)
                    + total_pressure_ref_inv_fx * press_div_coeff_k_fx * p_k_fx
                )
                + float(params.get("solid_volumetric_penalty", 1.0)) * (1.0 - alpha_k_fx) * pi_s_k_fx * q_fx
            ) * dx_fx
    else:
        r_elastic_k_fx = _linear_elastic(u_k_fx, vS_test_fx)
        r_elastic_n_fx = _linear_elastic(u_n_fx, vS_test_fx)
        r_volumetric_fx = _const(0.0) * alpha_k_fx * dx_fx
    r_skeleton_fx = (
        sk_th_fx * alpha_k_fx * r_elastic_k_fx
        + sk_omth_fx * alpha_n_fx * r_elastic_n_fx
        + sk_th_fx * r_skel_press_k_fx
        + sk_omth_fx * r_skel_press_n_fx
        + sk_th_fx * r_skel_drag_k_fx
        + sk_omth_fx * r_skel_drag_n_fx
    ) * dx_fx
    if bool(use_diffuse_traction):
        r_skeleton_fx += (
            th_fx * traction_weight_k_fx * ufl.dot(g_t_k_fx, vS_test_fx)
            + omth_fx * traction_weight_n_fx * ufl.dot(g_t_n_fx, vS_test_fx)
        ) * dx_fx
    if gamma_vS_eff is None:
        gamma_vS_eff = gamma_u_eff
    if vS_ext_mode_eff is None:
        vS_ext_mode_eff = u_ext_mode_eff
    if gamma_vS_eff != 0.0:
        gamma_vS_fx = _const(gamma_vS_eff)
        if vS_ext_mode_eff in {"l2", "mass"}:
            r_skeleton_fx += gamma_vS_fx * inv_h2_fx * (1.0 - alpha_k_fx) * ufl.dot(vS_k_fx, vS_test_fx) * dx_fx
        elif vS_ext_mode_eff in {"grad", "h1"}:
            r_skeleton_fx += gamma_vS_fx * (1.0 - alpha_k_fx) * ufl.inner(ufl.grad(vS_k_fx), ufl.grad(vS_test_fx)) * dx_fx
            if gamma_vS_pin_eff not in {None, 0.0}:
                r_skeleton_fx += _const(gamma_vS_pin_eff) * inv_h2_fx * (1.0 - alpha_k_fx) ** 2 * ufl.dot(vS_k_fx, vS_test_fx) * dx_fx
        else:
            raise ValueError(f"Unsupported vS_extension_mode={vS_ext_mode_eff!r}.")
    if float(params["vS_cip"]) != 0.0:
        tau_vS_cip_fx = _const(float(params["vS_cip"])) * (h_face_fx * h_face_fx * h_face_fx) * inv_dt_fx
        r_skeleton_fx += tau_vS_cip_fx * ufl.avg(alpha_n_fx) * _fx_grad_inner_jump(vS_k_fx, vS_test_fx, n_int_fx, gdim) * dS_meas_fx
    if bool(params["include_skeleton_acceleration"]) and float(params["rho_s0_tilde"]) != 0.0:
        rhoS_k_fx = rho_s0_fx * B_k_fx
        rhoS_n_fx = rho_s0_fx * B_n_fx
        r_skeleton_fx += ufl.inner((rhoS_k_fx * vS_k_fx - rhoS_n_fx * vS_n_fx) * inv_dt_fx, vS_test_fx) * dx_fx
        div_rhoS_vS_n_fx = rho_s0_fx * divBvS_n_fx
        inertia_key = str(params["skeleton_inertia_convection"]).strip().lower()
        if inertia_key == "full":
            div_rhoS_vS_k_fx = rho_s0_fx * divBvS_k_fx
            advS_k_fx = ufl.dot(ufl.dot(ufl.grad(vS_k_fx), vS_k_fx), vS_test_fx)
            advS_n_fx = ufl.dot(ufl.dot(ufl.grad(vS_n_fx), vS_n_fx), vS_test_fx)
            r_skeleton_fx += th_fx * (rhoS_k_fx * advS_k_fx + div_rhoS_vS_k_fx * ufl.dot(vS_k_fx, vS_test_fx)) * dx_fx
            r_skeleton_fx += omth_fx * (rhoS_n_fx * advS_n_fx + div_rhoS_vS_n_fx * ufl.dot(vS_n_fx, vS_test_fx)) * dx_fx
        elif inertia_key == "lagged":
            advS_k_fx = ufl.dot(ufl.dot(ufl.grad(vS_k_fx), vS_n_fx), vS_test_fx)
            advS_n_fx = ufl.dot(ufl.dot(ufl.grad(vS_n_fx), vS_n_fx), vS_test_fx)
            r_skeleton_fx += th_fx * (rhoS_n_fx * advS_k_fx + div_rhoS_vS_n_fx * ufl.dot(vS_k_fx, vS_test_fx)) * dx_fx
            r_skeleton_fx += omth_fx * (rhoS_n_fx * advS_n_fx + div_rhoS_vS_n_fx * ufl.dot(vS_n_fx, vS_test_fx)) * dx_fx
        else:
            raise ValueError(f"Unsupported skeleton_inertia_convection={params['skeleton_inertia_convection']!r}.")

    kin_scale_val = (
        float(kin_scale_override_fx)
        if kin_scale_override_fx is not None
        else (float(params["rho_s0_tilde"]) if float(params["rho_s0_tilde"]) != 0.0 else 1.0)
    )
    kin_scale_fx = _const(kin_scale_val)
    Fkin_k_fx = (u_k_fx - u_n_fx) * inv_dt_fx
    Fkin_k_fx += th_fx * (ufl.dot(ufl.grad(u_k_fx), vS_k_fx) - vS_k_fx)
    Fkin_k_fx += omth_fx * (ufl.dot(ufl.grad(u_n_fx), vS_n_fx) - vS_n_fx)
    r_kinematics_fx = kin_scale_fx * alpha_k_fx * ufl.dot(Fkin_k_fx, u_test_fx) * dx_fx
    if float(params.get("u_supg", 0.0)) != 0.0:
        tau_u_fx = _const(float(params["u_supg"])) * (h_cell_fx * h_cell_fx) / (
            h_cell_fx * ufl.sqrt(ufl.dot(vS_n_fx, vS_n_fx) + 1.0e-12)
            + (h_cell_fx * h_cell_fx) * inv_dt_fx
            + 1.0e-16
        )
        adv_u_k_fx = ufl.dot(ufl.grad(u_k_fx), vS_n_fx)
        adv_u_test_fx = ufl.dot(ufl.grad(u_test_fx), vS_n_fx)
        r_kinematics_fx += kin_scale_fx * tau_u_fx * alpha_n_fx * ufl.inner(adv_u_k_fx, adv_u_test_fx) * dx_fx
    if gamma_u_eff != 0.0:
        if u_ext_mode_eff in {"l2", "mass"}:
            r_kinematics_fx += kin_scale_fx * gamma_u_fx * inv_h2_fx * (1.0 - alpha_k_fx) * ufl.dot(u_k_fx, u_test_fx) * dx_fx
        elif u_ext_mode_eff in {"grad", "h1"}:
            r_kinematics_fx += kin_scale_fx * gamma_u_fx * (1.0 - alpha_k_fx) * ufl.inner(ufl.grad(u_k_fx), ufl.grad(u_test_fx)) * dx_fx
            if gamma_u_pin_eff != 0.0:
                r_kinematics_fx += kin_scale_fx * _const(gamma_u_pin_eff) * inv_h2_fx * (1.0 - alpha_k_fx) ** 2 * ufl.dot(u_k_fx, u_test_fx) * dx_fx
        else:
            raise ValueError(f"Unsupported u_extension_mode={u_ext_mode_eff!r}.")
    if float(params["u_cip"]) != 0.0:
        tau_u_cip_fx = _const(float(params["u_cip"])) * (h_face_fx * h_face_fx * h_face_fx) * inv_dt_fx
        u_cip_weight_key = str(params["u_cip_weight"]).strip().lower()
        if u_cip_weight_key in {"fluid", "one_minus_alpha", "1-alpha", "one-minus-alpha"}:
            w_u_cip_fx = ufl.avg(1.0 - alpha_n_fx)
        elif u_cip_weight_key in {"biofilm", "alpha"}:
            w_u_cip_fx = ufl.avg(alpha_n_fx)
        else:
            w_u_cip_fx = 1.0
        r_kinematics_fx += kin_scale_fx * tau_u_cip_fx * w_u_cip_fx * ufl.inner(ufl.jump(ufl.grad(u_k_fx), n_int_fx), ufl.jump(ufl.grad(u_test_fx), n_int_fx)) * dS_meas_fx

    if bool(enable_phi_evolution):
        one_m_alpha_k_fx = 1.0 - alpha_k_fx
        w_phi_fluid_k_fx = one_m_alpha_k_fx ** 16
        Pi_k_fx = (-k_d_fx) * alpha_k_fx * (1.0 - phi_k_fx)
        Pi_n_fx = (-k_d_fx) * alpha_n_fx * (1.0 - phi_n_fx)
        if support_key == "internal_conversion":
            r_phi_fx = phi_transport_test_fx * ((B_k_fx - B_n_fx) * inv_dt_fx) * dx_fx
            r_phi_fx += -th_fx * ufl.dot(B_k_fx * vS_k_fx, ufl.grad(phi_transport_test_fx)) * dx_fx
            r_phi_fx += -omth_fx * ufl.dot(B_n_fx * vS_n_fx, ufl.grad(phi_transport_test_fx)) * dx_fx
            r_phi_fx += -phi_transport_test_fx * (th_fx * Pi_k_fx + omth_fx * Pi_n_fx) * dx_fx
        elif support_key == "legacy_exchange":
            Fphi_k_fx = ufl.dot(ufl.grad(phi_k_fx), vS_k_fx) - (1.0 - phi_k_fx) * ufl.div(vS_k_fx) + Pi_k_fx
            Fphi_n_fx = ufl.dot(ufl.grad(phi_n_fx), vS_n_fx) - (1.0 - phi_n_fx) * ufl.div(vS_n_fx) + Pi_n_fx
            r_phi_fx = alpha_k_fx * phi_transport_test_fx * ((phi_k_fx - phi_n_fx) * inv_dt_fx) * dx_fx
            r_phi_fx += th_fx * alpha_k_fx * phi_transport_test_fx * Fphi_k_fx * dx_fx
            r_phi_fx += omth_fx * alpha_n_fx * phi_transport_test_fx * Fphi_n_fx * dx_fx
        else:
            raise ValueError(f"Unsupported support_physics={params['support_physics']!r}.")
        phi_diff_key = str(params["phi_diffusion_weight"]).strip().lower()
        if phi_diff_key in {"fluid", "one_minus_alpha", "1-alpha", "one-minus-alpha"}:
            w_phi_diff_k_fx = 1.0 - alpha_k_fx
        elif phi_diff_key in {"biofilm", "alpha"}:
            w_phi_diff_k_fx = alpha_k_fx
        else:
            w_phi_diff_k_fx = 1.0
        r_phi_fx += D_phi_fx * w_phi_diff_k_fx * ufl.inner(ufl.grad(phi_k_fx), ufl.grad(phi_transport_test_fx)) * dx_fx
        r_phi_fx += gamma_phi_fx * w_phi_fluid_k_fx * (phi_k_fx - 1.0) * phi_transport_test_fx * dx_fx
        if support_key == "internal_conversion" and float(params["phi_supg"]) != 0.0:
            vmag_fx = ufl.sqrt(ufl.dot(vS_n_fx, vS_n_fx) + 1.0e-12)
            denom_fx = (2.0 * inv_dt_fx) ** 2 + (2.0 * vmag_fx / (h_cell_fx + 1.0e-12)) ** 2
            tau_supg_fx = _const(float(params["phi_supg"])) / ufl.sqrt(denom_fx + 1.0e-16)
            w_supg_fx = alpha_n_fx * ufl.dot(ufl.grad(phi_transport_test_fx), vS_n_fx)
            f_phi_supg_fx = (B_k_fx - B_n_fx) * inv_dt_fx
            f_phi_supg_fx += th_fx * ufl.div(B_k_fx * vS_k_fx) + omth_fx * ufl.div(B_n_fx * vS_n_fx)
            f_phi_supg_fx += -(th_fx * Pi_k_fx + omth_fx * Pi_n_fx)
            f_phi_supg_fx += gamma_phi_fx * w_phi_fluid_k_fx * (phi_k_fx - 1.0)
            r_phi_fx += tau_supg_fx * w_supg_fx * f_phi_supg_fx * dx_fx
    else:
        r_phi_fx = _const(0.0) * dx_fx

    adv_key = str(params["alpha_advect_with"]).strip().lower()
    if adv_key == "vs":
        adv_u_k_fx, adv_u_n_fx = vS_k_fx, vS_n_fx
    elif adv_key == "v":
        adv_u_k_fx, adv_u_n_fx = v_k_fx, v_n_fx
    elif adv_key == "mix":
        adv_u_k_fx = C_n_fx * v_k_fx + B_n_fx * vS_k_fx
        adv_u_n_fx = C_n_fx * v_n_fx + B_n_fx * vS_n_fx
    elif adv_key in {"biofilm", "biofilm_volume", "biofilm-volume", "phase", "phase_volume", "phase-volume"}:
        adv_u_k_fx = phi_state_k * v_k_fx + (1.0 - phi_state_k) * vS_k_fx
        adv_u_n_fx = phi_state_n * v_n_fx + (1.0 - phi_state_n) * vS_n_fx
    elif adv_key == "interface":
        adv_u_k_fx = 0.5 * (v_k_fx + vS_k_fx)
        adv_u_n_fx = 0.5 * (v_n_fx + vS_n_fx)
    else:
        raise ValueError(f"Unsupported alpha_advect_with={params['alpha_advect_with']!r}.")
    form_key = str(params["alpha_advection_form"]).strip().lower()
    r_alpha_fx = alpha_transport_test_fx * ((alpha_k_fx - alpha_n_fx) * inv_dt_fx) * dx_fx
    if form_key == "advective":
        r_alpha_fx += th_fx * alpha_transport_test_fx * ufl.dot(ufl.grad(alpha_k_fx), adv_u_k_fx) * dx_fx
        r_alpha_fx += omth_fx * alpha_transport_test_fx * ufl.dot(ufl.grad(alpha_n_fx), adv_u_n_fx) * dx_fx
        adv_alpha_k_fx = ufl.dot(ufl.grad(alpha_k_fx), adv_u_k_fx)
        adv_alpha_n_fx = ufl.dot(ufl.grad(alpha_n_fx), adv_u_n_fx)
    elif form_key == "conservative_weak":
        r_alpha_fx += -th_fx * ufl.dot(alpha_k_fx * adv_u_k_fx, ufl.grad(alpha_transport_test_fx)) * dx_fx
        r_alpha_fx += -omth_fx * ufl.dot(alpha_n_fx * adv_u_n_fx, ufl.grad(alpha_transport_test_fx)) * dx_fx
        adv_alpha_k_fx = ufl.dot(ufl.grad(alpha_k_fx), adv_u_k_fx) + alpha_k_fx * ufl.div(adv_u_k_fx)
        adv_alpha_n_fx = ufl.dot(ufl.grad(alpha_n_fx), adv_u_n_fx) + alpha_n_fx * ufl.div(adv_u_n_fx)
    elif form_key == "conservative":
        r_alpha_fx += th_fx * alpha_transport_test_fx * (ufl.dot(ufl.grad(alpha_k_fx), adv_u_k_fx) + alpha_k_fx * ufl.div(adv_u_k_fx)) * dx_fx
        r_alpha_fx += omth_fx * alpha_transport_test_fx * (ufl.dot(ufl.grad(alpha_n_fx), adv_u_n_fx) + alpha_n_fx * ufl.div(adv_u_n_fx)) * dx_fx
        adv_alpha_k_fx = ufl.dot(ufl.grad(alpha_k_fx), adv_u_k_fx) + alpha_k_fx * ufl.div(adv_u_k_fx)
        adv_alpha_n_fx = ufl.dot(ufl.grad(alpha_n_fx), adv_u_n_fx) + alpha_n_fx * ufl.div(adv_u_n_fx)
    elif form_key == "interface_band_conservative":
        band_k_fx = 4.0 * alpha_k_fx * (1.0 - alpha_k_fx)
        band_n_fx = 4.0 * alpha_n_fx * (1.0 - alpha_n_fx)
        r_alpha_fx = alpha_transport_test_fx * ((band_k_fx - band_n_fx) * inv_dt_fx) * dx_fx
        r_alpha_fx += th_fx * alpha_transport_test_fx * (ufl.dot(ufl.grad(band_k_fx), adv_u_k_fx) + band_k_fx * ufl.div(adv_u_k_fx)) * dx_fx
        r_alpha_fx += omth_fx * alpha_transport_test_fx * (ufl.dot(ufl.grad(band_n_fx), adv_u_n_fx) + band_n_fx * ufl.div(adv_u_n_fx)) * dx_fx
        adv_alpha_k_fx = ufl.dot(ufl.grad(band_k_fx), adv_u_k_fx) + band_k_fx * ufl.div(adv_u_k_fx)
        adv_alpha_n_fx = ufl.dot(ufl.grad(band_n_fx), adv_u_n_fx) + band_n_fx * ufl.div(adv_u_n_fx)
    else:
        raise ValueError(f"Unsupported alpha_advection_form={params['alpha_advection_form']!r}.")
    if float(params["alpha_supg"]) != 0.0:
        vmag_fx = ufl.sqrt(ufl.dot(adv_u_n_fx, adv_u_n_fx) + 1.0e-12)
        denom_fx = (2.0 * inv_dt_fx) ** 2 + (2.0 * vmag_fx / (h_cell_fx + 1.0e-12)) ** 2
        tau_supg_fx = _const(float(params["alpha_supg"])) / ufl.sqrt(denom_fx + 1.0e-16)
        w_supg_fx = ufl.dot(ufl.grad(alpha_transport_test_fx), adv_u_n_fx)
        f_alpha_supg_fx = (alpha_k_fx - alpha_n_fx) * inv_dt_fx
        f_alpha_supg_fx += th_fx * adv_alpha_k_fx + omth_fx * adv_alpha_n_fx
        r_alpha_fx += tau_supg_fx * w_supg_fx * f_alpha_supg_fx * dx_fx
    alpha_reg_key = str(params.get("alpha_regularization", "none")).strip().lower()
    ch_enabled = alpha_reg_key == "ch" and float(params.get("M_alpha", 0.0)) != 0.0 and float(params.get("gamma_alpha", 0.0)) != 0.0
    if bool(ch_enabled):
        Wp_ch_k_fx = 2.0 * alpha_k_fx * (1.0 - alpha_k_fx) * (1.0 - 2.0 * alpha_k_fx)
        r_alpha_fx += th_fx * ufl.inner(ufl.grad(mu_alpha_k_fx) * M_alpha_fx, ufl.grad(alpha_transport_test_fx)) * dx_fx
        r_alpha_fx += omth_fx * ufl.inner(ufl.grad(mu_alpha_n_fx) * M_alpha_fx, ufl.grad(alpha_transport_test_fx)) * dx_fx
        r_mu_alpha_fx = mu_alpha_test_fx * mu_alpha_k_fx * dx_fx
        r_mu_alpha_fx += -(gamma_alpha_fx * eps_alpha_fx) * ufl.inner(ufl.grad(alpha_k_fx), ufl.grad(mu_alpha_test_fx)) * dx_fx
        r_mu_alpha_fx += -mu_alpha_test_fx * ((gamma_alpha_fx / eps_alpha_fx) * Wp_ch_k_fx) * dx_fx
    else:
        r_mu_alpha_fx = alpha_mu_aux_pin_fx * mu_alpha_test_fx * mu_alpha_k_fx * dx_fx

    if bool(use_alpha_latent) and latent_formulation_key != "transformed":
        alpha_sig_fx = _latent_map_expr_ufl(alpha_latent_k_fx, map_kind=latent_map_kind)
        r_alpha_embed_fx = (alpha_k_fx - alpha_sig_fx) * alpha_test_fx * dx_fx
        r_alpha_fx += r_alpha_embed_fx
    if bool(use_phi_latent) and latent_formulation_key != "transformed":
        phi_sig_fx = _latent_map_expr_ufl(phi_latent_k_fx, map_kind=latent_map_kind)
        r_phi_embed_fx = (phi_k_fx - phi_sig_fx) * phi_test_fx * dx_fx
        r_phi_fx += r_phi_embed_fx

    if bool(enable_phi_evolution):
        monod_k_fx = mu_max_fx * (S_k_fx / (K_S_fx + S_k_fx))
        monod_n_fx = mu_max_fx * (S_n_fx / (K_S_fx + S_n_fx))
        RS_k_fx = rho_s_star_fx * (1.0 / Y_fx) * ((monod_k_fx - k_d_fx) * (1.0 - phi_k_fx) * alpha_k_fx)
        RS_n_fx = rho_s_star_fx * (1.0 / Y_fx) * ((monod_n_fx - k_d_fx) * (1.0 - phi_n_fx) * alpha_n_fx)
        CSk_fx = C_k_fx * S_k_fx
        CSn_fx = C_n_fx * S_n_fx
        div_CSv_k_fx = CSk_fx * ufl.div(v_k_fx) + S_k_fx * ufl.dot(gradC_k_fx, v_k_fx) + C_k_fx * ufl.dot(ufl.grad(S_k_fx), v_k_fx)
        div_CSv_n_fx = CSn_fx * ufl.div(v_n_fx) + S_n_fx * ufl.dot(gradC_n_fx, v_n_fx) + C_n_fx * ufl.dot(ufl.grad(S_n_fx), v_n_fx)
        r_sub_fx = S_test_fx * ((CSk_fx - CSn_fx) * inv_dt_fx) * dx_fx
        r_sub_fx += S_test_fx * (th_fx * div_CSv_k_fx + omth_fx * div_CSv_n_fx) * dx_fx
        r_sub_fx += D_S_fx * (th_fx * ufl.inner(ufl.grad(S_k_fx), ufl.grad(S_test_fx)) + omth_fx * ufl.inner(ufl.grad(S_n_fx), ufl.grad(S_test_fx))) * dx_fx
        r_sub_fx += S_test_fx * (th_fx * RS_k_fx + omth_fx * RS_n_fx) * dx_fx
    else:
        r_sub_fx = _const(0.0) * dx_fx

    r_mom_fx = fluid_scale_fx * r_mom_fx
    r_skeleton_fx = skeleton_scale_fx * r_skeleton_fx

    blocks = {
        "momentum": r_mom_fx,
        "mass": r_mass_fx,
        "skeleton": r_skeleton_fx,
        "kinematics": r_kinematics_fx,
        "alpha": r_alpha_fx,
        "mu_alpha": r_mu_alpha_fx,
    }
    if bool(solid_volumetric_split):
        blocks["pi_s"] = r_volumetric_fx
    if bool(enable_phi_evolution):
        blocks["phi"] = r_phi_fx
        blocks["substrate"] = r_sub_fx
    r_total_fx = None
    for block in blocks.values():
        r_total_fx = block if r_total_fx is None else r_total_fx + block
    a_total_fx = ufl.derivative(r_total_fx, w_k, dw)
    return r_total_fx, a_total_fx, blocks


def _pycutfem_full_system_compare(problem, forms, *, qdeg: int, backend: str):
    if str(problem.get("fluid_space", "cg")).strip().lower() != "cg":
        return None
    enable_phi_evolution = bool(problem["_audit_enable_phi_evolution"])
    requested_latent_fields = tuple(problem.get("_audit_latent_bounded_fields", tuple()) or tuple())
    latent_bounded_fields = []
    for name in requested_latent_fields:
        key = str(name).strip()
        if key == "alpha" and problem.get("alpha_latent_k") is not None:
            latent_bounded_fields.append("alpha")
        elif key == "phi" and problem.get("phi_latent_k") is not None:
            latent_bounded_fields.append("phi")
    latent_bounded_fields = tuple(latent_bounded_fields)
    pressure_mean_constraint = bool(problem.get("_audit_pressure_mean_constraint", False))
    solid_volumetric_split = bool(problem.get("_audit_solid_volumetric_split", False))
    params = dict(problem["_audit_form_params"])
    params["dt"] = float(problem["_audit_dt"])
    params["theta"] = float(problem["_audit_theta"])
    mesh_fx, W_fx = _build_fenics_full_system(
        Lx=float(problem["_audit_Lx"]),
        Ly=float(problem["_audit_Ly"]),
        nx=int(problem["_audit_nx"]),
        ny=int(problem["_audit_ny"]),
        poly_order=int(problem["_audit_poly_order"]),
        pressure_order=int(problem["_audit_pressure_order"]),
        scalar_order=int(problem["_audit_scalar_order"]),
        enable_phi_evolution=bool(enable_phi_evolution),
        latent_bounded_fields=tuple(latent_bounded_fields),
        pressure_mean_constraint=bool(pressure_mean_constraint),
        solid_volumetric_split=bool(solid_volumetric_split),
    )
    mapping = _map_pycutfem_to_fenics_full(
        problem,
        W_fx,
        enable_phi_evolution=bool(enable_phi_evolution),
        latent_bounded_fields=tuple(latent_bounded_fields),
        pressure_mean_constraint=bool(pressure_mean_constraint),
        solid_volumetric_split=bool(solid_volumetric_split),
    )
    w_k_fx, w_n_fx = _load_fenics_full_state(
        problem,
        W_fx,
        mapping,
        enable_phi_evolution=bool(enable_phi_evolution),
        latent_bounded_fields=tuple(latent_bounded_fields),
        pressure_mean_constraint=bool(pressure_mean_constraint),
        solid_volumetric_split=bool(solid_volumetric_split),
    )
    r_total_fx, a_total_fx, _ = _fenics_full_forms(
        W=W_fx,
        w_k=w_k_fx,
        w_n=w_n_fx,
        qdeg=int(qdeg),
        enable_phi_evolution=bool(enable_phi_evolution),
        params=params,
    )
    form_r_fx = dolfinx.fem.form(r_total_fx)
    vec_fx = dolfinx.fem.petsc.assemble_vector(form_r_fx)
    r_fx_arr = np.asarray(vec_fx.array, dtype=float).copy()
    form_a_fx = dolfinx.fem.form(a_total_fx)
    A_fx = dolfinx.fem.petsc.assemble_matrix(form_a_fx)
    A_fx.assemble()
    indptr, indices, data = A_fx.getValuesCSR()
    J_fx = csr_matrix((data, indices, indptr), shape=A_fx.getSize()).tocsr()
    if bool(pressure_mean_constraint):
        q_fx_form = dolfinx.fem.form(ufl.split(ufl.TestFunction(W_fx))[1] * ufl.dx(metadata={"quadrature_degree": int(qdeg)}))
        q_vec = dolfinx.fem.petsc.assemble_vector(q_fx_form)
        c_fx = np.asarray(q_vec.array, dtype=float).copy()
        p_mean_val = float(np.asarray(problem["p_mean_k"].nodal_values, dtype=float).reshape(-1)[0])
        p_mean_res = float(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.split(w_k_fx)[1] * ufl.dx(metadata={"quadrature_degree": int(qdeg)}))))
        r_fx_arr = np.concatenate([r_fx_arr + p_mean_val * c_fx, np.asarray([p_mean_res], dtype=float)])
        n_base = int(J_fx.shape[0])
        J_ext = np.zeros((n_base + 1, n_base + 1), dtype=float)
        J_ext[:n_base, :n_base] = J_fx.toarray()
        J_ext[:n_base, n_base] = c_fx
        J_ext[n_base, :n_base] = c_fx
        J_fx = csr_matrix(J_ext)

    dh = problem["dh"]
    _, r_pc = assemble_form(Equation(None, forms.residual_form), dof_handler=dh, bcs=[], quad_order=int(qdeg), backend=str(backend))
    J_pc, _ = assemble_form(Equation(forms.jacobian_form, None), dof_handler=dh, bcs=[], quad_order=int(qdeg), backend=str(backend))
    r_pc = np.asarray(r_pc, dtype=float)
    J_pc = J_pc.tocsr().toarray()

    order_pc_to_fx = np.empty_like(r_pc, dtype=int)
    for fld, _, _ in _audit_field_layout(
        enable_phi_evolution=bool(enable_phi_evolution),
        latent_bounded_fields=tuple(latent_bounded_fields),
        pressure_mean_constraint=bool(pressure_mean_constraint),
        solid_volumetric_split=bool(solid_volumetric_split),
    ):
        sl = np.asarray(dh.get_field_slice(fld), dtype=int)
        order_pc_to_fx[sl] = np.asarray(mapping[fld], dtype=int)

    r_fx_reordered = r_fx_arr[order_pc_to_fx]
    J_fx_reordered = J_fx[order_pc_to_fx, :][:, order_pc_to_fx].toarray()
    field_scales = _condition_balanced_field_scales(
        mechanics_nondim_mode=str(params.get("mechanics_nondim_mode", "legacy")),
        dt=float(params["dt"]),
        mu_f=float(params["mu_f"]),
        kappa_inv=float(params["kappa_inv"]),
        mu_s=float(params["mu_s"]),
        lambda_s=float(params["lambda_s"]),
        rho_s0_tilde=float(params["rho_s0_tilde"]),
        dim=int(getattr(problem["mesh"], "spatial_dim", getattr(problem["mesh"], "dim", 2))),
    )
    if field_scales:
        scale_full = _full_field_scale_vector(dh, field_scales)
        r_pc = scale_full * r_pc
        r_fx_reordered = scale_full * r_fx_reordered
        J_pc = scale_full[:, None] * J_pc * scale_full[None, :]
        J_fx_reordered = scale_full[:, None] * J_fx_reordered * scale_full[None, :]
    full_res_max_abs, full_res_rel = _compare_dense(r_pc, r_fx_reordered)
    full_jac_max_abs, full_jac_rel = _compare_dense(J_pc, J_fx_reordered)

    block_rows = {
        "momentum": np.concatenate(
            [
                np.asarray(dh.get_field_slice("v_x"), dtype=int),
                np.asarray(dh.get_field_slice("v_y"), dtype=int),
            ]
        ),
        "mass": np.asarray(dh.get_field_slice("p"), dtype=int),
        "skeleton": np.concatenate(
            [
                np.asarray(dh.get_field_slice("vS_x"), dtype=int),
                np.asarray(dh.get_field_slice("vS_y"), dtype=int),
            ]
        ),
        "kinematics": np.concatenate(
            [
                np.asarray(dh.get_field_slice("u_x"), dtype=int),
                np.asarray(dh.get_field_slice("u_y"), dtype=int),
            ]
        ),
        "alpha": np.asarray(dh.get_field_slice("alpha"), dtype=int),
        "mu_alpha": np.asarray(dh.get_field_slice("mu_alpha"), dtype=int),
    }
    if bool(pressure_mean_constraint):
        block_rows["pressure_mean"] = np.asarray(dh.get_field_slice("p_mean"), dtype=int)
    if bool(solid_volumetric_split):
        block_rows["pi_s"] = np.asarray(dh.get_field_slice("pi_s"), dtype=int)
    if bool(enable_phi_evolution):
        block_rows["phi"] = np.asarray(dh.get_field_slice("phi"), dtype=int)
        block_rows["substrate"] = np.asarray(dh.get_field_slice("S"), dtype=int)
    if "alpha" in set(latent_bounded_fields):
        block_rows["alpha_latent"] = np.asarray(dh.get_field_slice("alpha_latent"), dtype=int)
    if bool(enable_phi_evolution) and "phi" in set(latent_bounded_fields):
        block_rows["phi_latent"] = np.asarray(dh.get_field_slice("phi_latent"), dtype=int)

    block_metrics: dict[str, dict[str, float]] = {}
    for name, rows in block_rows.items():
        res_abs, res_rel = _compare_dense(r_pc[rows], r_fx_reordered[rows])
        jac_abs, jac_rel = _compare_dense(J_pc[rows, :], J_fx_reordered[rows, :])
        jac_self_abs, jac_self_rel = _compare_dense(
            J_pc[rows, :][:, rows],
            J_fx_reordered[rows, :][:, rows],
        )
        block_metrics[str(name)] = {
            "res_max_abs": float(res_abs),
            "res_rel": float(res_rel),
            "jac_max_abs": float(jac_abs),
            "jac_rel": float(jac_rel),
            "jac_self_max_abs": float(jac_self_abs),
            "jac_self_rel": float(jac_self_rel),
        }

    return {
        "full_res_max_abs": float(full_res_max_abs),
        "full_res_rel": float(full_res_rel),
        "full_jac_max_abs": float(full_jac_max_abs),
        "full_jac_rel": float(full_jac_rel),
        "full_blocks": block_metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark 7 Jacobian audit against FD and FEniCSx derivative.")
    ap.add_argument("--out", type=Path, default=Path("out/benchmark7_fenics_jacobian_audit"))
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--Ly", type=float, default=1.5)
    ap.add_argument("--nx", type=int, default=2)
    ap.add_argument("--ny", type=int, default=3)
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--poly-order", type=int, default=2)
    ap.add_argument("--pressure-order", type=int, default=1)
    ap.add_argument("--scalar-order", type=int, default=1)
    ap.add_argument("--enable-phi-evolution", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--latent-bounded-transport", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--latent-bounded-fields", type=str, default="alpha,phi")
    ap.add_argument("--latent-bounded-eps", type=float, default=1.0e-8)
    ap.add_argument("--latent-bounded-map", type=str, default="sigmoid", choices=("sigmoid", "tanh", "algebraic"))
    ap.add_argument("--latent-bounded-formulation", type=str, default="embedded", choices=("embedded", "transformed"))
    ap.add_argument("--pressure-mean-constraint", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--solid-volumetric-split", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--solid-volumetric-penalty", type=float, default=1.0)
    ap.add_argument(
        "--mechanics-nondim-mode",
        type=str,
        default="legacy",
        choices=("legacy", "stress_balance", "condition_balanced"),
    )
    ap.add_argument("--pycutfem-backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--n-random-directions", type=int, default=1)
    ap.add_argument("--fd-eps", type=float, default=1.0e-6)
    ap.add_argument("--gamma-div", type=float, default=0.0)
    ap.add_argument("--gamma-u", type=float, default=0.0)
    ap.add_argument("--u-extension", type=str, default="l2", choices=("l2", "mass", "grad", "h1"))
    ap.add_argument("--gamma-u-pin", type=float, default=0.0)
    ap.add_argument("--u-cip", type=float, default=0.0)
    ap.add_argument("--u-cip-weight", type=str, default="fluid", choices=("fluid", "biofilm", "both"))
    ap.add_argument("--u-supg", type=float, default=0.0)
    ap.add_argument("--v-supg", type=float, default=0.0)
    ap.add_argument("--v-supg-mode", type=str, default="streamline", choices=("streamline", "residual"))
    ap.add_argument("--v-supg-c-nu", type=float, default=4.0)
    ap.add_argument("--v-cip", type=float, default=0.0)
    ap.add_argument("--vS-cip", type=float, default=0.0)
    ap.add_argument("--gamma-vS", type=float, default=None)
    ap.add_argument("--vS-ext-mode", type=str, default=None, choices=("l2", "mass", "grad", "h1"))
    ap.add_argument("--gamma-vS-pin", type=float, default=None)
    ap.add_argument("--fluid-convection", type=str, default="full", choices=("full", "lagged", "imex", "off"))
    ap.add_argument("--include-skeleton-acceleration", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rho-s0-tilde", type=float, default=1.1)
    ap.add_argument("--skeleton-inertia-convection", type=str, default="full", choices=("lagged", "full"))
    ap.add_argument("--alpha-supg", type=float, default=0.0)
    ap.add_argument("--alpha-cip", type=float, default=0.0)
    ap.add_argument("--M-alpha", type=float, default=1.0)
    ap.add_argument("--gamma-alpha", type=float, default=1.0)
    ap.add_argument("--eps-alpha", type=float, default=0.05)
    ap.add_argument("--alpha-regularization", type=str, default="ch", choices=("none", "ch", "olsson_nt"))
    ap.add_argument("--D-phi", type=float, default=0.0)
    ap.add_argument("--gamma-phi", type=float, default=5.0)
    ap.add_argument("--use-diffuse-traction", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--alpha-biot",
        type=float,
        default=None,
        help="Optional Biot-Willis coefficient used in the skeleton pressure audit. Use 1.0 for the Seboldt linear-Biot coefficient.",
    )
    ap.add_argument(
        "--skeleton-pressure-mode",
        type=str,
        default="whole_domain",
        choices=("whole_domain", "seboldt"),
        help="Skeleton pressure coupling to audit: diffuse one-domain split or sharp Seboldt Biot term.",
    )
    ap.add_argument(
        "--cases",
        type=str,
        nargs="+",
        default=("vS_band", "v_band", "interface_band"),
        choices=(
            "vS_band",
            "v_band",
            "interface_band",
            "vS_adv",
            "v_adv",
            "interface_adv",
            "biofilm_cons",
            "biofilm_cweak",
            "biofilm_cweak_biot1",
            "biofilm_cweak_seboldt",
            "biofilm_cweak_alpha_supg",
            "biofilm_cweak_phi_supg",
            "christan_cweak_fullstab",
            "mix_cweak",
        ),
    )
    args = ap.parse_args()

    all_cases = [
        ("vS_band", "vS", "interface_band_conservative", "legacy_exchange", 0.0, None, 0.0, 0.0, "whole_domain", None),
        ("v_band", "v", "interface_band_conservative", "legacy_exchange", 0.0, None, 0.0, 0.0, "whole_domain", None),
        ("interface_band", "interface", "interface_band_conservative", "legacy_exchange", 0.0, None, 0.0, 0.0, "whole_domain", None),
        ("vS_adv", "vS", "advective", "legacy_exchange", 0.0, None, 0.0, 0.0, "whole_domain", None),
        ("v_adv", "v", "advective", "legacy_exchange", 0.0, None, 0.0, 0.0, "whole_domain", None),
        ("interface_adv", "interface", "advective", "legacy_exchange", 0.0, None, 0.0, 0.0, "whole_domain", None),
        ("biofilm_cons", "biofilm_volume", "conservative", "internal_conversion", 0.0, 0.0, 0.0, 0.0, "whole_domain", None),
        ("biofilm_cweak", "biofilm_volume", "conservative_weak", "internal_conversion", 0.0, 0.0, 0.0, 0.0, "whole_domain", None),
        ("biofilm_cweak_biot1", "biofilm_volume", "conservative_weak", "internal_conversion", 0.0, 0.0, 0.0, 0.0, "whole_domain", 1.0),
        ("biofilm_cweak_seboldt", "biofilm_volume", "conservative_weak", "internal_conversion", 0.0, 0.0, 0.0, 0.0, "seboldt", 1.0),
        ("biofilm_cweak_alpha_supg", "biofilm_volume", "conservative_weak", "internal_conversion", 0.0, 0.0, None, 0.0, "whole_domain", None),
        ("biofilm_cweak_phi_supg", "biofilm_volume", "conservative_weak", "internal_conversion", 1.0, 0.0, 0.0, 0.0, "whole_domain", None),
        ("christan_cweak_fullstab", "biofilm_volume", "conservative_weak", "internal_conversion", 0.0, 0.0, None, 0.0, "whole_domain", None),
        ("mix_cweak", "mix", "conservative_weak", "legacy_exchange", 0.0, None, 0.0, 0.0, "whole_domain", None),
    ]
    wanted = set(args.cases)
    cases = [item for item in all_cases if item[0] in wanted]
    latent_field_set = []
    for name in _parse_csv_fields(args.latent_bounded_fields):
        if name == "alpha":
            latent_field_set.append("alpha")
        elif name == "phi" and bool(args.enable_phi_evolution):
            latent_field_set.append("phi")
    latent_bounded_fields = tuple(dict.fromkeys(latent_field_set))

    summaries: list[CaseSummary] = []
    args.out.mkdir(parents=True, exist_ok=True)
    for (
        name,
        adv_with,
        adv_form,
        support_physics,
        phi_supg,
        phi_cip,
        case_alpha_supg,
        case_alpha_cip,
        case_skeleton_pressure_mode,
        case_alpha_biot,
    ) in cases:
        problem, forms, field_to_func, qdeg = _build_benchmark_case(
            Lx=float(args.Lx),
            Ly=float(args.Ly),
            nx=int(args.nx),
            ny=int(args.ny),
            poly_order=int(args.poly_order),
            pressure_order=int(args.pressure_order),
            scalar_order=int(args.scalar_order),
            dt=float(args.dt),
            theta=float(args.theta),
            enable_phi_evolution=bool(args.enable_phi_evolution),
            latent_bounded_transport=bool(args.latent_bounded_transport),
            latent_bounded_fields=tuple(latent_bounded_fields),
            latent_bounded_eps=float(args.latent_bounded_eps),
            latent_bounded_map=str(args.latent_bounded_map),
            latent_bounded_formulation=str(args.latent_bounded_formulation),
            pressure_mean_constraint=bool(args.pressure_mean_constraint),
            solid_volumetric_split=bool(args.solid_volumetric_split),
            solid_volumetric_penalty=float(args.solid_volumetric_penalty),
            mechanics_nondim_mode=str(args.mechanics_nondim_mode),
            alpha_advect_with=str(adv_with),
            alpha_advection_form=str(adv_form),
            support_physics=str(support_physics),
            M_alpha=float(args.M_alpha),
            gamma_alpha=float(args.gamma_alpha),
            eps_alpha=float(args.eps_alpha),
            alpha_regularization=str(args.alpha_regularization),
            D_phi=float(args.D_phi),
            gamma_phi=float(args.gamma_phi),
            phi_supg=float(phi_supg),
            phi_cip=phi_cip,
            alpha_supg=float(args.alpha_supg if case_alpha_supg is None else case_alpha_supg),
            alpha_cip=float(args.alpha_cip if case_alpha_cip is None else case_alpha_cip),
            v_supg=float(args.v_supg),
            v_supg_mode=str(args.v_supg_mode),
            v_supg_c_nu=float(args.v_supg_c_nu),
            u_supg=float(args.u_supg),
            v_cip=float(args.v_cip),
            gamma_div=float(args.gamma_div),
            gamma_u=float(args.gamma_u),
            u_extension_mode=str(args.u_extension),
            gamma_u_pin=float(args.gamma_u_pin),
            u_cip=float(args.u_cip),
            u_cip_weight=str(args.u_cip_weight),
            vS_cip=float(args.vS_cip),
            gamma_vS=(None if args.gamma_vS is None else float(args.gamma_vS)),
            vS_extension_mode=(None if args.vS_ext_mode is None else str(args.vS_ext_mode)),
            gamma_vS_pin=(None if args.gamma_vS_pin is None else float(args.gamma_vS_pin)),
            fluid_convection=str(args.fluid_convection),
            include_skeleton_acceleration=bool(args.include_skeleton_acceleration),
            rho_s0_tilde=float(args.rho_s0_tilde),
            skeleton_inertia_convection=str(args.skeleton_inertia_convection),
            skeleton_pressure_mode=str(case_skeleton_pressure_mode if case_skeleton_pressure_mode is not None else args.skeleton_pressure_mode),
            alpha_biot=(case_alpha_biot if case_alpha_biot is not None else args.alpha_biot),
            use_diffuse_traction=bool(args.use_diffuse_traction or name == "christan_cweak_fullstab"),
        )
        directional_fd_max_rel, directional_fd_per_case = _directional_fd_audit(
            problem,
            forms,
            field_to_func,
            qdeg=int(qdeg),
            enable_phi_evolution=bool(args.enable_phi_evolution),
            backend=str(args.pycutfem_backend),
            n_random=int(args.n_random_directions),
            fd_eps=float(args.fd_eps),
        )
        full_compare = _pycutfem_full_system_compare(
            problem,
            forms,
            qdeg=int(qdeg),
            backend=str(args.pycutfem_backend),
        )
        alpha_res_max_abs, alpha_res_rel, alpha_jac_max_abs, alpha_jac_rel = _pycutfem_alpha_compare(
            problem,
            forms,
            qdeg=int(qdeg),
            alpha_advect_with=str(adv_with),
            alpha_advection_form=str(adv_form),
            phi_b=float(problem["_audit_phi_b"]),
            backend=str(args.pycutfem_backend),
            full_compare=full_compare,
        ) if not bool(args.latent_bounded_transport) else (None, None, None, None)
        phi_res_max_abs, phi_res_rel, phi_jac_max_abs, phi_jac_rel = _pycutfem_phi_compare(
            problem,
            forms,
            qdeg=int(qdeg),
            backend=str(args.pycutfem_backend),
        ) if (not bool(args.latent_bounded_transport)) else (None, None, None, None)
        skel_res_max_abs, skel_res_rel, skel_jac_max_abs, skel_jac_rel = _pycutfem_skeleton_pressure_compare(
            problem,
            forms,
            qdeg=int(qdeg),
            backend=str(args.pycutfem_backend),
        )
        summaries.append(
            CaseSummary(
                name=str(name),
                directional_fd_max_rel=float(directional_fd_max_rel),
                directional_fd_per_case={k: float(v) for k, v in directional_fd_per_case.items()},
                alpha_res_max_abs=(None if alpha_res_max_abs is None else float(alpha_res_max_abs)),
                alpha_res_rel=(None if alpha_res_rel is None else float(alpha_res_rel)),
                alpha_jac_max_abs=(None if alpha_jac_max_abs is None else float(alpha_jac_max_abs)),
                alpha_jac_rel=(None if alpha_jac_rel is None else float(alpha_jac_rel)),
                phi_res_max_abs=(None if phi_res_max_abs is None else float(phi_res_max_abs)),
                phi_res_rel=(None if phi_res_rel is None else float(phi_res_rel)),
                phi_jac_max_abs=(None if phi_jac_max_abs is None else float(phi_jac_max_abs)),
                phi_jac_rel=(None if phi_jac_rel is None else float(phi_jac_rel)),
                skeleton_pressure_res_max_abs=(None if skel_res_max_abs is None else float(skel_res_max_abs)),
                skeleton_pressure_res_rel=(None if skel_res_rel is None else float(skel_res_rel)),
                skeleton_pressure_jac_max_abs=(None if skel_jac_max_abs is None else float(skel_jac_max_abs)),
                skeleton_pressure_jac_rel=(None if skel_jac_rel is None else float(skel_jac_rel)),
                full_res_max_abs=(None if full_compare is None else float(full_compare["full_res_max_abs"])),
                full_res_rel=(None if full_compare is None else float(full_compare["full_res_rel"])),
                full_jac_max_abs=(None if full_compare is None else float(full_compare["full_jac_max_abs"])),
                full_jac_rel=(None if full_compare is None else float(full_compare["full_jac_rel"])),
                full_blocks=(None if full_compare is None else {str(k): dict(v) for k, v in full_compare["full_blocks"].items()}),
            )
        )
        (args.out / "comparison_with_fenics.json").write_text(
            json.dumps([s.__dict__ for s in summaries], indent=2),
            encoding="utf-8",
        )

    data = [s.__dict__ for s in summaries]
    (args.out / "comparison_with_fenics.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    for item in data:
        print(json.dumps(item, indent=2))


if __name__ == "__main__":
    main()
