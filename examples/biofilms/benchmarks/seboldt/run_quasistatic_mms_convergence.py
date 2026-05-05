#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from pycutfem.solvers.nonlinear_solver import (
    LinearSolverParameters,
    NewtonParameters,
    NewtonSolver,
    TimeStepperParameters,
)
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import Constant, dot
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dx

from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import (
    _build_forms,
    _create_problem,
)
from examples.biofilms.benchmarks.seboldt.quasistatic_mms_from_precice import (
    DEFAULT_FIT_MODULE_PATH,
    SeboldtQuasiStaticMMSStep,
    build_seboldt_quasistatic_mms_step,
)


def _eoc(prev_h: float, h: float, prev_err: float, err: float) -> float:
    if not (prev_h > 0.0 and h > 0.0 and prev_err > 0.0 and err > 0.0):
        return float("nan")
    return float(math.log(prev_err / err) / math.log(prev_h / h))


def _zero_bc(_x, _y):
    return 0.0


def _as_float_time(fn):
    return lambda x, y, t: float(np.asarray(fn(np.asarray(x), np.asarray(y), float(t))).reshape(()))


def _build_current_prev_functions(problem: dict[str, object]):
    current_functions = [
        problem["v_k"],
        problem["p_k"],
        *([problem["p_pore_k"]] if problem.get("p_pore_k") is not None else []),
        *([problem["vP_k"]] if problem.get("vP_k") is not None else []),
        problem["vS_k"],
        problem["u_k"],
        problem["alpha_k"],
        *([problem["rho_s_k"]] if problem.get("rho_s_k") is not None else []),
        *([problem["mu_mass_k"]] if problem.get("mu_mass_k") is not None else []),
        *([problem["mu_normal_k"]] if problem.get("mu_normal_k") is not None else []),
        *([problem["mu_tangent_k"]] if problem.get("mu_tangent_k") is not None else []),
        *([problem["mu_kin_k"]] if problem.get("mu_kin_k") is not None else []),
        *([problem["lm_vf_k"]] if problem.get("lm_vf_k") is not None else []),
        *([problem["lm_p_k"]] if problem.get("lm_p_k") is not None else []),
        *([problem["lm_vP_k"]] if problem.get("lm_vP_k") is not None else []),
        *([problem["lm_vS_k"]] if problem.get("lm_vS_k") is not None else []),
        *([problem["lm_p_pore_k"]] if problem.get("lm_p_pore_k") is not None else []),
        *([problem["lm_phi_k"]] if problem.get("lm_phi_k") is not None else []),
        *([problem["lm_u_k"]] if problem.get("lm_u_k") is not None else []),
        *([problem["B_k"]] if problem.get("B_k") is not None else []),
        *([problem["mu_k"]] if problem.get("mu_k") is not None else []),
    ]
    previous_functions = [
        problem["v_n"],
        problem["p_n"],
        *([problem["p_pore_n"]] if problem.get("p_pore_n") is not None else []),
        *([problem["vP_n"]] if problem.get("vP_n") is not None else []),
        problem["vS_n"],
        problem["u_n"],
        problem["alpha_n"],
        *([problem["rho_s_n"]] if problem.get("rho_s_n") is not None else []),
        *([problem["mu_mass_n"]] if problem.get("mu_mass_n") is not None else []),
        *([problem["mu_normal_n"]] if problem.get("mu_normal_n") is not None else []),
        *([problem["mu_tangent_n"]] if problem.get("mu_tangent_n") is not None else []),
        *([problem["mu_kin_n"]] if problem.get("mu_kin_n") is not None else []),
        *([problem["lm_vf_n"]] if problem.get("lm_vf_n") is not None else []),
        *([problem["lm_p_n"]] if problem.get("lm_p_n") is not None else []),
        *([problem["lm_vP_n"]] if problem.get("lm_vP_n") is not None else []),
        *([problem["lm_vS_n"]] if problem.get("lm_vS_n") is not None else []),
        *([problem["lm_p_pore_n"]] if problem.get("lm_p_pore_n") is not None else []),
        *([problem["lm_phi_n"]] if problem.get("lm_phi_n") is not None else []),
        *([problem["lm_u_n"]] if problem.get("lm_u_n") is not None else []),
        *([problem["B_n"]] if problem.get("B_n") is not None else []),
        *([problem["mu_n"]] if problem.get("mu_n") is not None else []),
    ]
    if problem.get("p_mean_k") is not None:
        current_functions.insert(2, problem["p_mean_k"])
        previous_functions.insert(2, problem["p_mean_n"])
    if problem.get("alpha_mass_lm_k") is not None:
        current_functions.insert(3, problem["alpha_mass_lm_k"])
        previous_functions.insert(3, problem["alpha_mass_lm_n"])
    if problem.get("pi_s_k") is not None:
        current_functions.insert(4, problem["pi_s_k"])
        previous_functions.insert(4, problem["pi_s_n"])
    if problem.get("lambda_drag_k") is not None:
        drag_insert = current_functions.index(problem["alpha_k"])
        prev_drag_insert = previous_functions.index(problem["alpha_n"])
        current_functions.insert(drag_insert, problem["lambda_drag_k"])
        previous_functions.insert(prev_drag_insert, problem["lambda_drag_n"])
    if problem.get("alpha_latent_k") is not None:
        current_functions.append(problem["alpha_latent_k"])
        previous_functions.append(problem["alpha_latent_n"])
    if problem.get("phi_k") is not None:
        current_functions.append(problem["phi_k"])
        previous_functions.append(problem["phi_n"])
    if problem.get("S_k") is not None:
        current_functions.append(problem["S_k"])
        previous_functions.append(problem["S_n"])
    if problem.get("phi_latent_k") is not None:
        current_functions.append(problem["phi_latent_k"])
        previous_functions.append(problem["phi_latent_n"])
    return current_functions, previous_functions


def _set_exact_states(problem: dict[str, object], mms: SeboldtQuasiStaticMMSStep) -> None:
    dh = problem["dh"]

    def _set_scalar(func, value_fn) -> None:
        field = str(func.field_name)
        coords = np.asarray(dh.get_dof_coords(field), dtype=float)
        vals = np.asarray(value_fn(coords[:, 0], coords[:, 1]), dtype=float).reshape(-1)
        func.nodal_values[:] = vals

    def _set_vector(func, value_fn) -> None:
        values = None
        for comp_index, comp in enumerate(func.components):
            field = str(comp.field_name)
            coords = np.asarray(dh.get_dof_coords(field), dtype=float)
            if values is None or values.shape[0] != coords.shape[0]:
                values = np.asarray(value_fn(coords[:, 0], coords[:, 1]), dtype=float)
            comp_vals = np.asarray(values[:, comp_index], dtype=float).reshape(-1)
            comp.nodal_values[:] = comp_vals

    _set_vector(problem["v_n"], mms.v_n)
    _set_scalar(problem["p_n"], mms.p_n)
    _set_vector(problem["vP_n"], mms.vP_n)
    _set_scalar(problem["p_pore_n"], mms.p_pore_n)
    _set_vector(problem["vS_n"], mms.vS_n)
    _set_vector(problem["u_n"], mms.u_n)
    _set_scalar(problem["alpha_n"], mms.alpha_n)
    _set_scalar(problem["phi_n"], mms.phi_n)

    if problem.get("B_n") is not None:
        _set_scalar(
            problem["B_n"],
            lambda x, y: np.asarray(mms.alpha_n(x, y), dtype=float) * (1.0 - np.asarray(mms.phi_n(x, y), dtype=float)),
        )

    for name in (
        "mu_mass_n",
        "mu_mass_k",
        "mu_normal_n",
        "mu_normal_k",
        "mu_tangent_n",
        "mu_tangent_k",
        "mu_kin_n",
        "mu_kin_k",
        "lm_vf_n",
        "lm_vf_k",
        "lm_p_n",
        "lm_p_k",
        "lm_vP_n",
        "lm_vP_k",
        "lm_vS_n",
        "lm_vS_k",
        "lm_p_pore_n",
        "lm_p_pore_k",
        "lm_phi_n",
        "lm_phi_k",
        "lm_u_n",
        "lm_u_k",
        "p_mean_n",
        "p_mean_k",
        "alpha_mass_lm_n",
        "alpha_mass_lm_k",
        "lambda_drag_n",
        "lambda_drag_k",
        "pi_s_n",
        "pi_s_k",
    ):
        if problem.get(name) is not None:
            problem[name].nodal_values[:] = 0.0

    _set_vector(problem["v_k"], mms.v_k)
    _set_scalar(problem["p_k"], mms.p_k)
    _set_vector(problem["vP_k"], mms.vP_k)
    _set_scalar(problem["p_pore_k"], mms.p_pore_k)
    _set_vector(problem["vS_k"], mms.vS_k)
    _set_vector(problem["u_k"], mms.u_k)
    _set_scalar(problem["alpha_k"], mms.alpha_k)
    _set_scalar(problem["phi_k"], mms.phi_k)

    if problem.get("B_k") is not None:
        _set_scalar(
            problem["B_k"],
            lambda x, y: np.asarray(mms.alpha_k(x, y), dtype=float) * (1.0 - np.asarray(mms.phi_k(x, y), dtype=float)),
        )


def _build_bcs(mms: SeboldtQuasiStaticMMSStep):
    bcs = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("v_x", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.v(x, y, t)[..., 0])),
                BoundaryCondition("v_y", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.v(x, y, t)[..., 1])),
                BoundaryCondition("p", "dirichlet", tag, _as_float_time(mms.p)),
                BoundaryCondition("vP_x", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.vP(x, y, t)[..., 0])),
                BoundaryCondition("vP_y", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.vP(x, y, t)[..., 1])),
                BoundaryCondition("p_pore", "dirichlet", tag, _as_float_time(mms.p_pore)),
                BoundaryCondition("vS_x", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.vS(x, y, t)[..., 0])),
                BoundaryCondition("vS_y", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.vS(x, y, t)[..., 1])),
                BoundaryCondition("u_x", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.u(x, y, t)[..., 0])),
                BoundaryCondition("u_y", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.u(x, y, t)[..., 1])),
                BoundaryCondition("alpha", "dirichlet", tag, _as_float_time(mms.alpha)),
                BoundaryCondition("phi", "dirichlet", tag, _as_float_time(mms.phi)),
            ]
        )
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, _zero_bc) for b in bcs]
    return bcs, bcs_homog


def _assemble_residual_vector(*, dh, form_obj, bcs, qdeg: int, backend: str) -> np.ndarray | None:
    if form_obj is None:
        return None
    _, vec = assemble_form(
        Equation(None, form_obj),
        dof_handler=dh,
        bcs=list(bcs or []),
        quad_order=int(qdeg),
        backend=str(backend),
    )
    arr = np.asarray(vec, dtype=float).ravel()
    dirichlet = dh.get_dirichlet_data(list(bcs or [])) or {}
    if dirichlet:
        bc_rows = np.fromiter(dirichlet.keys(), dtype=int)
        if bc_rows.size:
            arr = arr.copy()
            arr[bc_rows] = 0.0
    return arr


def _field_inf_norms_from_vector(*, dh, vec: np.ndarray) -> dict[str, float]:
    arr = np.asarray(vec, dtype=float).ravel()
    norms: dict[str, float] = {}
    for field_name in list(getattr(dh, "field_names", []) or []):
        field_dofs = np.asarray(dh.get_field_slice(str(field_name)), dtype=int).ravel()
        if field_dofs.size == 0:
            continue
        field_vals = arr[field_dofs]
        norms[str(field_name)] = float(np.linalg.norm(field_vals, ord=np.inf)) if field_vals.size else 0.0
    return norms


def _clone_problem_coefficients(problem: dict[str, object]) -> dict[str, object]:
    cloned = dict(problem)
    for key, value in list(problem.items()):
        if not getattr(value, "is_function", False):
            continue
        if bool(getattr(value, "is_test", False)) or bool(getattr(value, "is_trial", False)):
            continue
        copier = getattr(value, "copy", None)
        if callable(copier):
            cloned[key] = copier()
    return cloned


def _solve_one(
    *,
    nx: int,
    qdeg: int,
    qerr: int,
    newton_tol: float,
    max_it: int,
    backend: str,
    linear_backend: str,
    line_search: bool,
    globalization: str,
    adaptive_interface_target_cells: float,
    adaptive_interface_band_halfwidth_factor: float,
    adaptive_interface_max_ref: int,
    mms: SeboldtQuasiStaticMMSStep,
    rho_f: float,
    mu_f: float,
    kappa_inv: float,
    mu_s: float,
    lambda_s: float,
    phi_b: float,
    gamma_v: float,
    gamma_p: float,
    gamma_vP: float,
    gamma_p_pore: float,
    gamma_u: float,
    domain_lm_vf: bool,
    domain_lm_aug_gamma: float,
    transport_update_mode: str,
    lag_phi_in_main: bool | None,
    direct_interface_transfer: bool,
    disable_pore_momentum: bool,
    disable_solid_momentum: bool,
    combined_porous_momentum: bool,
    source_mode: str,
    diagnose_exact_residual: bool,
    diagnose_only: bool,
):
    ny = max(2, int(round(1.5 * int(nx))))
    source_mode_key = str(source_mode or "weak_exact").strip().lower().replace("-", "_")
    if source_mode_key == "strong_analytic":
        if (
            bool(disable_pore_momentum)
            or str(transport_update_mode).strip().lower() != "monolithic"
            or lag_phi_in_main not in (False, None)
            or not bool(direct_interface_transfer)
        ):
            raise NotImplementedError(
                "strong_analytic is only implemented for the older monolithic "
                "quasi-static direct-transfer note branch. Use source_mode=weak_exact "
                "for the post_accept / multiplier / combined porous branches."
            )
    print(f"[mms] nx={int(nx)} ny={int(ny)}: create problem", flush=True)
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=int(nx),
        ny=int(ny),
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
        one_domain_formulation="final_form",
        final_form_phi_mode="transport",
        final_form_constant_rho_s=True,
        final_form_domain_lm=bool(domain_lm_vf),
        final_form_domain_lm_vf=bool(domain_lm_vf),
        transport_update_mode=str(transport_update_mode),
        final_form_lag_phi_in_main=lag_phi_in_main,
        final_form_quasistatic_porous_media=True,
        final_form_direct_interface_transfer=bool(direct_interface_transfer),
        final_form_disable_pore_momentum=bool(disable_pore_momentum),
        final_form_disable_solid_momentum=bool(disable_solid_momentum),
        final_form_combined_porous_momentum=bool(combined_porous_momentum),
        y_interface=float(mms.y_interface),
        eps_alpha=float(mms.eps_alpha),
        adaptive_interface_target_cells=float(adaptive_interface_target_cells),
        adaptive_interface_band_halfwidth_factor=float(adaptive_interface_band_halfwidth_factor),
        adaptive_interface_max_ref=int(adaptive_interface_max_ref),
    )
    problem["support_physics"] = "stored_support"
    problem["geometry_indicator_beta"] = float(mms.support_indicator_beta)
    problem["geometry_indicator_mode"] = "raw"

    print(f"[mms] nx={int(nx)}: set exact states", flush=True)
    _set_exact_states(problem, mms)
    dt_c = Constant(float(mms.dt))
    print(f"[mms] nx={int(nx)}: build forms", flush=True)
    form_build_kwargs = dict(
        qdeg=int(qdeg),
        dt_c=dt_c,
        theta=float(mms.theta),
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        mu_b=float(mu_f),
        mu_b_model="mu",
        kappa_inv=float(kappa_inv),
        mu_s=float(mu_s),
        lambda_s=float(lambda_s),
        phi_b=float(phi_b),
        M_alpha=1.0,
        gamma_alpha=0.0,
        eps_alpha=float(mms.eps_alpha),
        solid_visco_eta=0.0,
        gamma_div=0.0,
        solid_volumetric_split=False,
        solid_volumetric_penalty=1.0,
        gamma_u=float(gamma_u),
        u_extension_mode="h1",
        gamma_u_pin=0.0,
        interface_band_extension_gamma=0.0,
        u_cip=0.0,
        u_cip_weight="fluid",
        vS_cip=0.0,
        gamma_vS=0.0,
        vS_extension_mode="h1",
        gamma_vS_pin=0.0,
        D_phi=0.0,
        phi_diffusion_weight="alpha",
        gamma_phi=0.0,
        gamma_v=float(gamma_v),
        v_extension_mode="h1",
        gamma_v_pin=0.0,
        gamma_p=float(gamma_p),
        p_extension_mode="h1",
        gamma_p_pin=0.0,
        gamma_vP=float(gamma_vP),
        vP_extension_mode="h1",
        gamma_vP_pin=0.0,
        gamma_p_pore=float(gamma_p_pore),
        p_pore_extension_mode="h1",
        gamma_p_pore_pin=0.0,
        gamma_rho_s=0.0,
        rho_s_extension_mode="h1",
        gamma_rho_s_pin=0.0,
        final_form_constant_rho_s=True,
        final_form_domain_lm=bool(domain_lm_vf),
        final_form_domain_lm_vP_tie_vf=False,
        final_form_domain_lm_p_pore_tie_p=False,
        final_form_quasistatic_porous_media=True,
        final_form_quasistatic_flip_pore_stress_sign=False,
        final_form_direct_interface_transfer=bool(direct_interface_transfer),
        final_form_disable_pore_momentum=bool(disable_pore_momentum),
        final_form_disable_solid_momentum=bool(disable_solid_momentum),
        final_form_combined_porous_momentum=bool(combined_porous_momentum),
        final_form_domain_lm_aug_gamma=float(domain_lm_aug_gamma),
        final_form_domain_lm_free_weight_mode="diffuse",
        final_form_domain_lm_free_alpha_max=None,
        final_form_mass_lm_aug_gamma=0.0,
        final_form_normal_lm_aug_gamma=0.0,
        phi_supg=0.0,
        phi_cip=0.0,
        alpha_supg=0.0,
        alpha_cip=0.0,
        v_supg=0.0,
        v_supg_mode="streamline",
        v_supg_c_nu=4.0,
        u_supg=0.0,
        v_cip=0.0,
        alpha_regularization="none",
        alpha_reg_gamma=0.0,
        alpha_reg_eps_normal=float(mms.eps_alpha),
        alpha_reg_eps_tangent=float(0.25 * mms.eps_alpha),
        alpha_reg_eta=0.0,
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_vS_gate_alpha0=0.0,
        alpha_vS_gate_power=8,
        solid_model="linear",
        kappa_inv_model="const",
        enable_phi_evolution=True,
        include_skeleton_acceleration=False,
        rho_s0_tilde=float(mms.rho_s),
        skeleton_inertia_convection="off",
        storativity_c0=0.0,
        fluid_convection=str(mms.fluid_convection),
        fluid_convection_full_weight=None,
        fluid_convection_lagged_weight=None,
        fluid_convection_imex_weight=None,
        support_physics="stored_support",
        ds_alpha_transport=None,
        ds_B_transport=None,
        skeleton_pressure_mode="whole_domain",
        alpha_biot=1.0,
        g_t_k=None,
        g_t_n=None,
        traction_weight_k=None,
        traction_weight_n=None,
        drag_formulation="direct",
        skeleton_acceleration_weight=None,
        skeleton_inertia_full_weight=None,
        skeleton_inertia_lagged_weight=None,
        reduced_support_state="alpha_B",
        full_ratio_free_state=False,
        split_primary_darcy_flux=False,
        split_pore_flux_model="exact_conservative_p",
        split_pore_momentum_model="band_alpha",
        fluid_mass_model="transported_free_content",
        one_pressure_primary_darcy_flux=False,
        stored_support_content_mode="evolve_B",
        pressure_interface_closure=False,
        pressure_interface_closure_strength=0.0,
        p_pore_fluid_gauge=False,
        p_pore_fluid_gauge_strength=0.0,
        interface_entry_closure=False,
        interface_entry_closure_strength=0.0,
        interface_entry_delta=10.0,
        interface_bjs_closure=False,
        interface_bjs_closure_strength=0.0,
        interface_bjs_gamma=0.0,
        final_form_solid_interface_weight=1.0,
        final_form_mass_interface_weight=1.0,
        final_form_normal_interface_weight=1.0,
        final_form_disable_interface_physics=False,
        final_form_disable_normal_interface=False,
        interface_velocity_continuity_closure=False,
        interface_velocity_method="penalty",
        interface_velocity_normal_strength=0.0,
        interface_velocity_tangential_strength=0.0,
        interface_traction_continuity_closure=False,
        interface_traction_method="penalty",
        interface_traction_normal_strength=0.0,
        interface_traction_tangential_strength=0.0,
    )
    forms = _build_forms(problem, **form_build_kwargs)
    forms_exact = None
    if source_mode_key == "weak_exact":
        problem_exact = _clone_problem_coefficients(problem)
        forms_exact = _build_forms(problem_exact, **form_build_kwargs)
        manufactured_residual = forms.residual_form - forms_exact.residual_form
    elif source_mode_key == "strong_analytic":
        qdx = dx(metadata={"q": int(qdeg)})
        src_fluid = Analytic(lambda x, y: mms.f_fluid_momentum(x, y), degree=max(8, int(qdeg)), dim=1)
        src_mass = Analytic(lambda x, y: mms.s_free_mass(x, y), degree=max(8, int(qdeg)))
        src_pore_mom = Analytic(lambda x, y: mms.f_pore_momentum(x, y), degree=max(8, int(qdeg)), dim=1)
        src_pore_mass = Analytic(lambda x, y: mms.s_porous_mass(x, y), degree=max(8, int(qdeg)))
        src_porous_body = Analytic(lambda x, y: mms.f_porous_body(x, y), degree=max(8, int(qdeg)), dim=1)
        src_kin = Analytic(lambda x, y: mms.f_kinematics(x, y), degree=max(8, int(qdeg)), dim=1)
        src_alpha = Analytic(lambda x, y: mms.f_alpha(x, y), degree=max(8, int(qdeg)))
        src_phi = Analytic(lambda x, y: mms.f_phi(x, y), degree=max(8, int(qdeg)))
        manufactured_residual = (
            forms.residual_form
            - (dot(src_fluid, problem["v_test"]) * qdx)
            - (src_mass * problem["q_test"] * qdx)
            - (dot(src_pore_mom, problem["vP_test"]) * qdx)
            - (src_pore_mass * problem["q_pore_test"] * qdx)
            - (dot(src_porous_body, problem["vS_test"]) * qdx)
            - (dot(src_kin, problem["u_test"]) * qdx)
            - (src_alpha * problem["alpha_test"] * qdx)
            - (src_phi * problem["phi_test"] * qdx)
        )
    else:
        raise ValueError(f"Unsupported source_mode={source_mode!r}. Use 'weak_exact' or 'strong_analytic'.")

    bcs, bcs_homog = _build_bcs(mms)
    lin_backend_eff = str(linear_backend)
    if lin_backend_eff == "pardiso":
        try:
            import pypardiso  # noqa: F401
        except Exception:
            lin_backend_eff = "scipy"

    print(f"[mms] nx={int(nx)}: build solver", flush=True)
    solver = NewtonSolver(
        manufactured_residual,
        forms.jacobian_form,
        dof_handler=problem["dh"],
        mixed_element=problem["me"],
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=float(newton_tol),
            max_newton_iter=int(max_it),
            print_level=2,
            line_search=bool(line_search),
            globalization=str(globalization),
        ),
        lin_params=LinearSolverParameters(backend=str(lin_backend_eff), tol=1.0e-12, maxit=10000),
        quad_order=int(qdeg),
        backend=str(backend),
    )
    current_functions, previous_functions = _build_current_prev_functions(problem)
    exact_residual_inf = float("nan")
    exact_residual_field_norms: dict[str, float] = {}
    exact_block_diagnostics: dict[str, object] = {}
    if bool(diagnose_exact_residual):
        print(f"[mms] nx={int(nx)}: assemble exact-state residual", flush=True)
        t_bc = float(mms.t_n) + float(mms.theta) * float(mms.dt)
        bcs_now = solver._freeze_bcs(bcs, t_bc)
        problem["dh"].apply_bcs(bcs_now, *current_functions)
        solver._current_bcs = bcs_now
        coeffs_now = {f.name: f for f in current_functions}
        coeffs_now.update({f.name: f for f in previous_functions})
        coeffs_now["dt"] = dt_c
        _, R_exact = solver._assemble_system_reduced(coeffs_now, need_matrix=False)
        field_names = np.asarray(solver._refresh_reduced_field_names(), dtype=object)
        exact_residual_inf = float(np.linalg.norm(np.asarray(R_exact, dtype=float), ord=np.inf))
        for name in np.unique(field_names):
            mask = field_names == name
            exact_residual_field_norms[str(name)] = float(np.linalg.norm(np.asarray(R_exact, dtype=float)[mask], ord=np.inf))
        top_fields = sorted(exact_residual_field_norms.items(), key=lambda kv: kv[1], reverse=True)[:8]
        print(
            f"[mms] nx={int(nx)}: exact-state residual |R|_inf={exact_residual_inf:.6e} "
            f"top={top_fields}",
            flush=True,
        )
        diagnostic_backend = "python"
        if forms_exact is None:
            block_forms = {
                "skeleton_balance": forms.r_skeleton - (dot(src_porous_body, problem["vS_test"]) * qdx),
                "pore_mass_balance": forms.r_pore - (src_pore_mass * problem["q_pore_test"] * qdx),
                "pore_momentum_balance": (
                    forms.r_momentum_terms.get("pore_bulk") + forms.r_momentum_terms.get("pore_extension_vP")
                    - (dot(src_pore_mom, problem["vP_test"]) * qdx)
                ),
                "free_mass_balance": forms.r_mass - (src_mass * problem["q_test"] * qdx),
            }
            for label, form_obj in block_forms.items():
                vec = _assemble_residual_vector(
                    dh=problem["dh"],
                    form_obj=form_obj,
                    bcs=bcs_now,
                    qdeg=int(qdeg),
                    backend=diagnostic_backend,
                )
                if vec is None:
                    continue
                norms = _field_inf_norms_from_vector(dh=problem["dh"], vec=vec)
                exact_block_diagnostics[str(label)] = {
                    "inf_norm": float(np.linalg.norm(vec, ord=np.inf)),
                    "top_fields": sorted(norms.items(), key=lambda kv: kv[1], reverse=True)[:6],
                }
            if exact_block_diagnostics:
                ordered_blocks = sorted(
                    ((name, info["inf_norm"]) for name, info in exact_block_diagnostics.items()),
                    key=lambda item: item[1],
                    reverse=True,
                )
                print(
                    f"[mms] nx={int(nx)}: exact-state block residuals top={ordered_blocks[:6]}",
                    flush=True,
                )
    if bool(diagnose_only):
        return {
            "nx": int(nx),
            "ny": int(ny),
            "h": float(1.0 / float(nx)),
            "h_band_target": float(problem.get("_mesh_adaptivity_meta", {}).get("target_h", float("nan")) if isinstance(problem.get("_mesh_adaptivity_meta", {}), dict) else float("nan")),
            "cells_across_2eps_min": float(problem.get("_mesh_adaptivity_meta", {}).get("cells_across_2eps_min", float("nan")) if isinstance(problem.get("_mesh_adaptivity_meta", {}), dict) else float((2.0 * float(mms.eps_alpha)) / max(1.0 / float(nx), 1.0e-14))),
            "err_v": float("nan"),
            "err_p": float("nan"),
            "err_vP": float("nan"),
            "err_p_pore": float("nan"),
            "err_vS": float("nan"),
            "err_u": float("nan"),
            "err_alpha": float("nan"),
            "err_phi": float("nan"),
            "domain_lm_vf": float(1.0 if bool(domain_lm_vf) else 0.0),
            "domain_lm_aug_gamma": float(domain_lm_aug_gamma),
            "exact_residual_inf": exact_residual_inf,
            "exact_residual_top_fields": json.dumps(
                sorted(exact_residual_field_norms.items(), key=lambda kv: kv[1], reverse=True)[:8]
            ),
            "exact_residual_block_diagnostics": json.dumps(exact_block_diagnostics),
            "nonlinear_iterations": -1,
            "nonlinear_norm": float("nan"),
        }
    print(f"[mms] nx={int(nx)}: solve one step", flush=True)
    solver.solve_time_interval(
        functions=current_functions,
        prev_functions=previous_functions,
        aux_functions={"dt": dt_c},
        time_params=TimeStepperParameters(
            dt=float(mms.dt),
            final_time=float(mms.dt),
            max_steps=1,
            theta=float(mms.theta),
        ),
    )

    dh = problem["dh"]
    print(f"[mms] nx={int(nx)}: compute errors", flush=True)
    err_v = dh.l2_error(
        problem["v_k"],
        exact={"v_x": lambda x, y: mms.v_k(x, y)[..., 0], "v_y": lambda x, y: mms.v_k(x, y)[..., 1]},
        fields=["v_x", "v_y"],
        quad_order=int(qerr),
        relative=False,
        backend=str(backend),
    )
    err_p = dh.l2_error(
        problem["p_k"],
        exact={"p": lambda x, y: mms.p_k(x, y)},
        fields=["p"],
        quad_order=int(qerr),
        relative=False,
        backend=str(backend),
    )
    err_vP = dh.l2_error(
        problem["vP_k"],
        exact={"vP_x": lambda x, y: mms.vP_k(x, y)[..., 0], "vP_y": lambda x, y: mms.vP_k(x, y)[..., 1]},
        fields=["vP_x", "vP_y"],
        quad_order=int(qerr),
        relative=False,
        backend=str(backend),
    )
    err_p_pore = dh.l2_error(
        problem["p_pore_k"],
        exact={"p_pore": lambda x, y: mms.p_pore_k(x, y)},
        fields=["p_pore"],
        quad_order=int(qerr),
        relative=False,
        backend=str(backend),
    )
    err_vS = dh.l2_error(
        problem["vS_k"],
        exact={"vS_x": lambda x, y: mms.vS_k(x, y)[..., 0], "vS_y": lambda x, y: mms.vS_k(x, y)[..., 1]},
        fields=["vS_x", "vS_y"],
        quad_order=int(qerr),
        relative=False,
        backend=str(backend),
    )
    err_u = dh.l2_error(
        problem["u_k"],
        exact={"u_x": lambda x, y: mms.u_k(x, y)[..., 0], "u_y": lambda x, y: mms.u_k(x, y)[..., 1]},
        fields=["u_x", "u_y"],
        quad_order=int(qerr),
        relative=False,
        backend=str(backend),
    )
    err_alpha = dh.l2_error(
        problem["alpha_k"],
        exact={"alpha": lambda x, y: mms.alpha_k(x, y)},
        fields=["alpha"],
        quad_order=int(qerr),
        relative=False,
        backend=str(backend),
    )
    err_phi = dh.l2_error(
        problem["phi_k"],
        exact={"phi": lambda x, y: mms.phi_k(x, y)},
        fields=["phi"],
        quad_order=int(qerr),
        relative=False,
        backend=str(backend),
    )

    mesh_meta = dict(problem.get("_mesh_adaptivity_meta", {}) or {})
    h_base = float(1.0 / float(nx))
    h_band_target = float(mesh_meta.get("target_h", float("nan")))
    cells_across_2eps_min = float(mesh_meta.get("cells_across_2eps_min", float("nan")))
    if not np.isfinite(cells_across_2eps_min):
        cells_across_2eps_min = float((2.0 * float(mms.eps_alpha)) / max(h_base, 1.0e-14))

    return {
        "nx": int(nx),
        "ny": int(ny),
        "h": h_base,
        "h_band_target": h_band_target,
        "cells_across_2eps_min": cells_across_2eps_min,
        "err_v": float(err_v),
        "err_p": float(err_p),
        "err_vP": float(err_vP),
        "err_p_pore": float(err_p_pore),
        "err_vS": float(err_vS),
        "err_u": float(err_u),
        "err_alpha": float(err_alpha),
        "err_phi": float(err_phi),
        "domain_lm_vf": float(1.0 if bool(domain_lm_vf) else 0.0),
        "domain_lm_aug_gamma": float(domain_lm_aug_gamma),
        "exact_residual_inf": exact_residual_inf,
        "exact_residual_top_fields": json.dumps(
            sorted(exact_residual_field_norms.items(), key=lambda kv: kv[1], reverse=True)[:8]
        ),
        "nonlinear_iterations": int(getattr(solver, "_last_nonlinear_iterations", -1) or -1),
        "nonlinear_norm": float(getattr(solver, "_last_nonlinear_norm", float("nan")) or float("nan")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the cpp-backend Seboldt quasi-static MMS study on the active final-form note branch.")
    ap.add_argument("--fit-module-path", type=str, default=str(DEFAULT_FIT_MODULE_PATH))
    ap.add_argument("--nx-list", type=str, default="30,60")
    ap.add_argument("--dt", type=float, default=5.0e-4)
    ap.add_argument("--t0", type=float, default=1.0)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--qdeg", type=int, default=6)
    ap.add_argument("--qerr", type=int, default=8)
    ap.add_argument("--backend", type=str, default="cpp")
    ap.add_argument("--linear-backend", type=str, default="pardiso")
    ap.add_argument("--newton-tol", type=float, default=1.0e-10)
    ap.add_argument("--max-it", type=int, default=20)
    ap.add_argument("--globalization", type=str, default="line_search_then_trust")
    ap.add_argument("--no-line-search", action="store_true")
    ap.add_argument("--eps-alpha", type=float, default=0.12)
    ap.add_argument("--interface-beta", type=float, default=40.0)
    ap.add_argument("--support-indicator-beta", type=float, default=4.0)
    ap.add_argument("--field-scale", type=float, default=1.0)
    ap.add_argument("--fluid-convection", type=str, default="full")
    ap.add_argument("--rho-f", type=float, default=1.0)
    ap.add_argument("--rho-s", type=float, default=1.0)
    ap.add_argument("--mu-f", type=float, default=3.5e-2)
    ap.add_argument("--kappa-inv", type=float, default=1.0e3)
    ap.add_argument("--mu-s", type=float, default=1.67785e5)
    ap.add_argument("--lambda-s", type=float, default=8.22148e6)
    ap.add_argument("--phi-b", type=float, default=0.30)
    ap.add_argument("--gamma-v", type=float, default=1.0)
    ap.add_argument("--gamma-p", type=float, default=1.0)
    ap.add_argument("--gamma-vP", type=float, default=1.0)
    ap.add_argument("--gamma-p-pore", type=float, default=1.0)
    ap.add_argument("--gamma-u", type=float, default=1.0)
    ap.add_argument("--domain-lm-vf", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--domain-lm-aug-gamma", type=float, default=10.0)
    ap.add_argument("--transport-update-mode", type=str, default="post_accept")
    ap.add_argument("--lag-phi-in-main", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--direct-interface-transfer", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--disable-pore-momentum", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--disable-solid-momentum", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--combined-porous-momentum", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--source-mode", type=str, default="weak_exact", choices=("weak_exact", "strong_analytic"))
    ap.add_argument("--adaptive-interface-target-cells", type=float, default=0.0)
    ap.add_argument("--adaptive-interface-band-halfwidth-factor", type=float, default=1.0)
    ap.add_argument("--adaptive-interface-max-ref", type=int, default=0)
    ap.add_argument("--no-diagnose-exact-residual", action="store_true")
    ap.add_argument("--diagnose-only", action="store_true")
    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    nx_values = [int(tok.strip()) for tok in str(args.nx_list).split(",") if tok.strip()]
    if not nx_values:
        raise ValueError("nx-list must contain at least one mesh size.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = (
        Path(args.outdir)
        if str(args.outdir).strip()
        else Path("out") / f"benchmark7_seboldt_quasistatic_mms_notes_{timestamp}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    mms = build_seboldt_quasistatic_mms_step(
        fit_module_path=args.fit_module_path,
        dt_val=float(args.dt),
        t0=float(args.t0),
        theta=float(args.theta),
        y_interface=1.0,
        eps_alpha=float(args.eps_alpha),
        interface_beta=float(args.interface_beta),
        support_indicator_beta=float(args.support_indicator_beta),
        field_scale=float(args.field_scale),
        rho_f=float(args.rho_f),
        rho_s=float(args.rho_s),
        mu_f=float(args.mu_f),
        kappa_inv=float(args.kappa_inv),
        mu_s=float(args.mu_s),
        lambda_s=float(args.lambda_s),
        phi_b=float(args.phi_b),
        fluid_convection=str(args.fluid_convection),
        gamma_v=float(args.gamma_v),
        gamma_p=float(args.gamma_p),
        gamma_vP=float(args.gamma_vP),
        gamma_p_pore=float(args.gamma_p_pore),
        gamma_u=float(args.gamma_u),
        disable_pore_momentum=bool(args.disable_pore_momentum),
        disable_solid_momentum=bool(args.disable_solid_momentum),
        combined_porous_momentum=bool(args.combined_porous_momentum),
    )

    rows: list[dict[str, object]] = []
    prev = None
    for nx in nx_values:
        row = _solve_one(
            nx=int(nx),
            qdeg=int(args.qdeg),
            qerr=int(args.qerr),
            newton_tol=float(args.newton_tol),
            max_it=int(args.max_it),
            backend=str(args.backend),
            linear_backend=str(args.linear_backend),
            line_search=not bool(args.no_line_search),
            globalization=str(args.globalization),
            adaptive_interface_target_cells=float(args.adaptive_interface_target_cells),
            adaptive_interface_band_halfwidth_factor=float(args.adaptive_interface_band_halfwidth_factor),
            adaptive_interface_max_ref=int(args.adaptive_interface_max_ref),
            mms=mms,
            rho_f=float(args.rho_f),
            mu_f=float(args.mu_f),
            kappa_inv=float(args.kappa_inv),
            mu_s=float(args.mu_s),
            lambda_s=float(args.lambda_s),
            phi_b=float(args.phi_b),
            gamma_v=float(args.gamma_v),
            gamma_p=float(args.gamma_p),
            gamma_vP=float(args.gamma_vP),
            gamma_p_pore=float(args.gamma_p_pore),
            gamma_u=float(args.gamma_u),
            domain_lm_vf=bool(args.domain_lm_vf),
            domain_lm_aug_gamma=float(args.domain_lm_aug_gamma),
            transport_update_mode=str(args.transport_update_mode),
            lag_phi_in_main=args.lag_phi_in_main,
            direct_interface_transfer=bool(args.direct_interface_transfer),
            disable_pore_momentum=bool(args.disable_pore_momentum),
            disable_solid_momentum=bool(args.disable_solid_momentum),
            combined_porous_momentum=bool(args.combined_porous_momentum),
            source_mode=str(args.source_mode),
            diagnose_exact_residual=not bool(args.no_diagnose_exact_residual),
            diagnose_only=bool(args.diagnose_only),
        )
        if prev is None:
            row["eoc_v"] = float("nan")
            row["eoc_p"] = float("nan")
            row["eoc_vP"] = float("nan")
            row["eoc_p_pore"] = float("nan")
            row["eoc_vS"] = float("nan")
            row["eoc_u"] = float("nan")
            row["eoc_alpha"] = float("nan")
            row["eoc_phi"] = float("nan")
        else:
            row["eoc_v"] = _eoc(prev["h"], row["h"], prev["err_v"], row["err_v"])
            row["eoc_p"] = _eoc(prev["h"], row["h"], prev["err_p"], row["err_p"])
            row["eoc_vP"] = _eoc(prev["h"], row["h"], prev["err_vP"], row["err_vP"])
            row["eoc_p_pore"] = _eoc(prev["h"], row["h"], prev["err_p_pore"], row["err_p_pore"])
            row["eoc_vS"] = _eoc(prev["h"], row["h"], prev["err_vS"], row["err_vS"])
            row["eoc_u"] = _eoc(prev["h"], row["h"], prev["err_u"], row["err_u"])
            row["eoc_alpha"] = _eoc(prev["h"], row["h"], prev["err_alpha"], row["err_alpha"])
            row["eoc_phi"] = _eoc(prev["h"], row["h"], prev["err_phi"], row["err_phi"])
        rows.append(row)
        prev = row
        print(
            f"nx={row['nx']:>4d} h={row['h']:.6f} "
            f"cells_2eps={row['cells_across_2eps_min']:.3f} "
            f"err_v={row['err_v']:.6e} err_p={row['err_p']:.6e} "
            f"err_u={row['err_u']:.6e} err_p_pore={row['err_p_pore']:.6e}"
        )

    csv_path = outdir / "quasistatic_mms_convergence.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "fit_module_path": str(Path(args.fit_module_path).resolve()),
        "outdir": str(outdir.resolve()),
        "timestamp": timestamp,
        "residual_source_model": "discrete_weak_exact" if str(args.source_mode) == "weak_exact" else "continuous_note_branch_with_extensions",
        "dt": float(args.dt),
        "t0": float(args.t0),
        "theta": float(args.theta),
        "qdeg": int(args.qdeg),
        "qerr": int(args.qerr),
        "backend": str(args.backend),
        "linear_backend_requested": str(args.linear_backend),
        "eps_alpha": float(args.eps_alpha),
        "interface_beta": float(args.interface_beta),
        "support_indicator_beta": float(args.support_indicator_beta),
        "field_scale": float(args.field_scale),
        "line_search": not bool(args.no_line_search),
        "globalization": str(args.globalization),
        "fluid_convection": str(args.fluid_convection),
        "rho_f": float(args.rho_f),
        "rho_s": float(args.rho_s),
        "mu_f": float(args.mu_f),
        "kappa_inv": float(args.kappa_inv),
        "mu_s": float(args.mu_s),
        "lambda_s": float(args.lambda_s),
        "phi_b": float(args.phi_b),
        "gamma_v": float(args.gamma_v),
        "gamma_p": float(args.gamma_p),
        "gamma_vP": float(args.gamma_vP),
        "gamma_p_pore": float(args.gamma_p_pore),
        "gamma_u": float(args.gamma_u),
        "transport_update_mode": str(args.transport_update_mode),
        "lag_phi_in_main": args.lag_phi_in_main,
        "direct_interface_transfer": bool(args.direct_interface_transfer),
        "disable_pore_momentum": bool(args.disable_pore_momentum),
        "disable_solid_momentum": bool(args.disable_solid_momentum),
        "combined_porous_momentum": bool(args.combined_porous_momentum),
        "source_mode": str(args.source_mode),
        "adaptive_interface_target_cells": float(args.adaptive_interface_target_cells),
        "adaptive_interface_band_halfwidth_factor": float(args.adaptive_interface_band_halfwidth_factor),
        "adaptive_interface_max_ref": int(args.adaptive_interface_max_ref),
        "rows": rows,
    }
    json_path = outdir / "quasistatic_mms_convergence.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
