import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import (
    _apply_field_dependent_lower_bound,
    _apply_field_dependent_upper_bound,
    _apply_open_top_global_phi_cleanup,
    _benchmark7_requires_constrained_solver,
    _benchmark7_solid_model_key,
    _benchmark7_transport_update_label,
    _build_bcs,
    _build_forms,
    _build_support_aware_phi_box_bounds,
    _build_transport_measures,
    _build_vi_linear_equalities,
    _configure_benchmark7_cpp_fuse_integrals,
    _condition_balanced_solid_cutoff_y,
    _condition_balanced_field_scales,
    _create_problem,
    _compute_interface_probe_diagnostics,
    _compute_profile_best_fit_scale,
    _effective_eps_alpha,
    _effective_logistic_bounded_fields,
    _interface_tangent_from_normal_2d,
    _interface_unit_normal_2d,
    _dot_basis_2d,
    _normal_tensor_traction_scalar_2d,
    _normal_viscous_traction_scalar_2d,
    _latent_bounded_fields,
    _normalize_benchmark7_solver_choice,
    _parse_args,
    _pc_p2_easy_dt_value,
    _pc_fluid_convection_selectors,
    _pc_path_tangent_euler_step,
    _pc_p2_lambda_schedule,
    _pc_skeleton_inertia_selectors,
    _pc_should_prefer_exact_probe,
    _pc_should_keep_lambda_stage,
    _rigid_support_diagnostic_active_fields,
    _solver_scaled_reduced_matrix,
    _predictor_corrector_startup_enabled,
    _should_use_frozen_transport_restart,
    _should_use_staggered_predictor_after_large_step,
    _fixed_fluid_poro_solid_diagnostic_active_fields,
    _all_porous_sideflow_diagnostic_active_fields,
    _solid_only_diagnostic_active_fields,
    _solid_cauchy_stress_from_grad_u,
    _startup_first_step_relaxed_accept_ginf,
    _startup_stage_relaxed_accept_ginf,
    _startup_monolithic_max_it,
    _startup_stage_solver_kind,
    _solver_side_alpha_mass_equality_enabled,
    _tangential_viscous_traction_scalar_2d,
    _use_post_accept_transport_update,
    _named_constant,
)
from tests.test_benchmark7_solver_backend_parity import _initialize_small_benchmark7_state
from pycutfem.jit import _form_rank
from pycutfem.jit.cache import KernelCache
from pycutfem.jit.ir import strip_side_metadata
from pycutfem.jit.visitor import IRGenerator
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Constant, Identity, Integral as ExprIntegral, div, dot, grad, inner
from pycutfem.ufl.forms import Equation, Form, assemble_form
from pycutfem.ufl.measures import dx


def _assemble_block(problem, form, field: str) -> np.ndarray:
    _, residual = assemble_form(
        Equation(None, form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    sl = np.asarray(problem["dh"].get_field_slice(field), dtype=int)
    return np.asarray(residual, dtype=float)[sl]


def _vector_component_expr(func, idx: int):
    if func is None:
        return None
    try:
        return func.components[int(idx)]
    except Exception:
        return func[int(idx)]


def _primary_q_field_names(problem) -> list[str]:
    dh_fields = tuple(getattr(problem["dh"], "field_names", tuple()) or tuple())
    if "q" in dh_fields and bool(problem.get("primary_darcy_flux", False)):
        return ["q"]
    if problem.get("lambda_drag_k") is not None:
        return ["lambda_drag_x", "lambda_drag_y"]
    return []


def _primary_q_residual_norm(problem, form) -> float:
    blocks = [_assemble_block(problem, form, fld) for fld in _primary_q_field_names(problem)]
    if not blocks:
        return 0.0
    return float(np.linalg.norm(np.concatenate([np.asarray(b, dtype=float).ravel() for b in blocks]), ord=np.inf))


def _find_named_function(funcs_in, template):
    for f in list(funcs_in or []):
        if str(getattr(f, "name", "")) == str(getattr(template, "name", "")):
            return f
    return template


def _make_args(*, enable_phi_evolution: bool, top_drainage_transport: bool) -> SimpleNamespace:
    return SimpleNamespace(
        alpha_box_constraints=True,
        enable_phi_evolution=bool(enable_phi_evolution),
        top_drainage_transport=bool(top_drainage_transport),
        internal_conversion_open_top_b_transport=False,
        phi_box_constraints=True,
        backend="python",
    )


def _build_full_problem():
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=2,
        ny=3,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
    )
    for key in ("vS_k", "vS_n", "u_k", "u_n"):
        problem[key].nodal_values[:] = 0.0
    for key in ("p_k", "p_n", "mu_k", "mu_n", "S_k", "S_n"):
        problem[key].nodal_values[:] = 0.0
    for key in ("alpha_k", "alpha_n"):
        problem[key].nodal_values[:] = 0.6
    for key in ("phi_k", "phi_n"):
        problem[key].nodal_values[:] = 0.25
    upward = lambda x, y: np.array([0.0, 1.0])
    problem["v_k"].set_values_from_function(upward)
    problem["v_n"].set_values_from_function(upward)
    return problem


def _build_ratio_free_full_problem(
    *,
    pressure_mean_constraint: bool = False,
    stored_support_content_mode: str = "evolve_B",
    drag_formulation: str = "direct",
    split_primary_darcy_flux: bool = False,
    split_pore_flux_model: str = "direct_darcy",
    fluid_space: str = "cg",
    fluid_hdiv_order: int = 0,
    darcy_flux_space: str = "auto",
):
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=1,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space=str(fluid_space),
        fluid_hdiv_order=int(fluid_hdiv_order),
        darcy_flux_space=str(darcy_flux_space),
        enable_phi_evolution=True,
        pressure_mean_constraint=bool(pressure_mean_constraint),
        full_ratio_free_state=True,
        drag_formulation=str(drag_formulation),
        split_primary_darcy_flux=bool(split_primary_darcy_flux),
        split_pore_flux_model=str(split_pore_flux_model),
        stored_support_content_mode=str(stored_support_content_mode),
    )
    for key in ("u_k", "u_n"):
        problem[key].nodal_values[:] = 0.0
    for key in ("mu_k", "mu_n", "S_k", "S_n"):
        problem[key].nodal_values[:] = 0.0
    for key in ("alpha_k", "alpha_n"):
        problem[key].nodal_values[:] = 1.0
    for key in ("B_k", "B_n"):
        problem[key].nodal_values[:] = 0.82
    return problem


def _build_ratio_free_full_forms(problem, **overrides):
    kwargs = dict(
        qdeg=6,
        dt_c=Constant(0.025),
        theta=1.0,
        rho_f=1.0,
        mu_f=0.035,
        mu_b=0.035,
        mu_b_model="mu",
        kappa_inv=1.0e3,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        phi_b=0.18,
        M_alpha=0.0,
        gamma_alpha=1.0,
        eps_alpha=0.05,
        solid_visco_eta=0.0,
        gamma_div=0.0,
        mechanics_nondim_mode="condition_balanced",
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
        alpha_supg=0.0,
        alpha_cip=0.0,
        alpha_regularization="none",
        alpha_reg_gamma=1.0,
        alpha_reg_eps_normal=0.05,
        alpha_reg_eps_tangent=0.0125,
        alpha_reg_eta=1.0e-12,
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_vS_gate_alpha0=0.0,
        alpha_vS_gate_power=8,
        solid_model="linear",
        kappa_inv_model="refmap",
        enable_phi_evolution=True,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        skeleton_inertia_convection="lagged",
        support_physics="stored_support",
        drag_formulation="direct",
        full_ratio_free_state=True,
        ds_alpha_transport=None,
        ds_B_transport=None,
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
    )
    kwargs.update(overrides)
    return _build_forms(problem, **kwargs)


def _build_single_pressure_stored_support_problem():
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=1,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=False,
        reduced_support_state="alpha_B",
    )
    for key in ("u_k", "u_n", "mu_k", "mu_n"):
        problem[key].nodal_values[:] = 0.0
    for key in ("p_k", "p_n"):
        problem[key].nodal_values[:] = 0.0
    for key in ("alpha_k", "alpha_n"):
        problem[key].nodal_values[:] = 1.0
    for key in ("B_k", "B_n"):
        problem[key].nodal_values[:] = 0.82
    return problem


def _build_full_forms(problem, *, ds_alpha_transport=None, ds_B_transport=None, **overrides):
    kwargs = dict(
        qdeg=6,
        dt_c=Constant(0.1),
        theta=1.0,
        rho_f=1.0,
        mu_f=0.035,
        mu_b=0.035,
        mu_b_model="mu",
        kappa_inv=1.0e3,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        phi_b=0.18,
        M_alpha=0.0,
        gamma_alpha=1.0,
        eps_alpha=0.1,
        solid_visco_eta=0.0,
        gamma_div=0.0,
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
        alpha_supg=0.0,
        alpha_cip=0.0,
        alpha_regularization="none",
        alpha_reg_gamma=1.0,
        alpha_reg_eps_normal=0.1,
        alpha_reg_eps_tangent=0.025,
        alpha_reg_eta=1.0e-12,
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        solid_model="linear",
        kappa_inv_model="refmap",
        enable_phi_evolution=True,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        skeleton_inertia_convection="lagged",
        fluid_convection="off",
        support_physics="internal_conversion",
        ds_alpha_transport=ds_alpha_transport,
        ds_B_transport=ds_B_transport,
    )
    kwargs.update(overrides)
    return _build_forms(problem, **kwargs)


def _integral_hashes(form, dh, *, rank: int) -> tuple[str, ...]:
    irg = IRGenerator()
    cache_sig = (dh.mixed_element.signature(), False, int(rank))
    hashes: list[str] = []
    for integral in form.integrals:
        ir = strip_side_metadata(irg.generate(integral.integrand), on_facet=False)
        hashes.append(KernelCache._hash_ir(ir, cache_sig))
    return tuple(hashes)


def _metadata_signature(metadata: dict | None) -> tuple[tuple[str, object], ...]:
    items: list[tuple[str, object]] = []
    for key, value in sorted((metadata or {}).items(), key=lambda kv: str(kv[0])):
        key_str = str(key)
        if isinstance(value, (int, float, str, bool, type(None))):
            items.append((key_str, value))
        else:
            items.append((key_str, (type(value).__name__, int(id(value)))))
    return tuple(items)


def _fuse_form_like_compile_multi(form, *, dof_handler) -> Form:
    compiler = FormCompiler(dof_handler, backend="python")
    mesh = dof_handler.mixed_element.mesh
    p_geo = int(getattr(mesh, "poly_order", 1))

    groups: dict[tuple[object, ...], list[object]] = {}
    order: list[tuple[object, ...]] = []
    for integral in form.integrals:
        domain_type = integral.measure.domain_type
        level_set = getattr(integral.measure, "level_set", None)
        rank = _form_rank(integral.integrand)
        if level_set is None and rank > 0 and domain_type in {"volume", "interior_facet", "exterior_facet"}:
            qdeg0 = int(compiler._find_q_order(integral)) + 2 * max(0, p_geo - 1)
            key = (
                "fuse",
                str(domain_type),
                int(rank),
                bool(integral.measure.on_facet),
                int(qdeg0),
                int(id(getattr(integral.measure, "defined_on", None))),
                int(id(getattr(integral.measure, "deformation", None))),
                _metadata_signature(integral.measure.metadata),
            )
        else:
            key = ("single", int(id(integral)))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(integral)

    fused_integrals = []
    for key in order:
        group = groups[key]
        if len(group) == 1:
            fused_integrals.append(group[0])
            continue
        integrand_sum = group[0].integrand
        for other in group[1:]:
            integrand_sum = integrand_sum + other.integrand
        fused_integrals.append(ExprIntegral(integrand_sum, group[0].measure))
    return Form(fused_integrals)


def _build_small_benchmark7_residual_problem():
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=1,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.05, phi_b=0.18)
    forms = _build_forms(
        problem,
        qdeg=6,
        dt_c=Constant(0.025),
        theta=1.0,
        rho_f=1.0,
        mu_f=0.035,
        mu_b=0.035,
        mu_b_model="mu",
        kappa_inv=1.0e3,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        phi_b=0.18,
        M_alpha=1.0,
        gamma_alpha=1.0,
        eps_alpha=0.05,
        solid_visco_eta=0.0,
        gamma_div=0.0,
        mechanics_nondim_mode="condition_balanced",
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
        gamma_phi=5.0,
        phi_supg=0.0,
        phi_cip=0.0,
        alpha_supg=0.5,
        alpha_cip=0.0,
        alpha_regularization="ch",
        alpha_reg_gamma=1.0,
        alpha_reg_eps_normal=0.05,
        alpha_reg_eps_tangent=0.0125,
        alpha_reg_eta=1.0e-12,
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        support_physics="internal_conversion",
        solid_model="linear",
        kappa_inv_model="refmap",
        drag_formulation="direct",
        fluid_convection="full",
        skeleton_pressure_mode="whole_domain",
        alpha_biot=None,
        enable_phi_evolution=True,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        skeleton_inertia_convection="lagged",
        ds_hdiv_tangential=None,
        ds_alpha_transport=None,
        ds_B_transport=None,
        hdiv_tangential_gamma=20.0,
        hdiv_tangential_method="penalty",
        solid_volumetric_split=False,
        solid_volumetric_penalty=1.0,
        v_supg=0.0,
        v_supg_mode="streamline",
        v_supg_c_nu=4.0,
        u_supg=0.0,
        v_cip=0.0,
    )
    return problem, forms


def test_benchmark7_top_drainage_keeps_alpha_mass_equality_but_releases_phi_mass_equality() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=2,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
    )
    equalities = _build_vi_linear_equalities(
        args=_make_args(enable_phi_evolution=True, top_drainage_transport=True),
        problem=problem,
        qdeg=4,
        alpha_bc_mode_key="natural",
        find_named_function=_find_named_function,
    )
    assert [eq.name for eq in equalities] == ["alpha_mass"]


def test_benchmark7_closed_top_keeps_alpha_and_phi_mass_equalities() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=2,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
    )
    equalities = _build_vi_linear_equalities(
        args=_make_args(enable_phi_evolution=True, top_drainage_transport=False),
        problem=problem,
        qdeg=4,
        alpha_bc_mode_key="natural",
        find_named_function=_find_named_function,
    )
    assert [eq.name for eq in equalities] == ["alpha_mass", "phi_biofilm_fluid_mass"]


def test_embedded_latent_phi_is_filtered_out_of_requested_fields() -> None:
    args = SimpleNamespace(latent_bounded_fields="alpha,phi", latent_bounded_formulation="embedded")
    assert _latent_bounded_fields(args, enable_phi_evolution=True) == ("alpha",)


def test_transformed_latent_fields_keep_alpha_and_phi() -> None:
    args = SimpleNamespace(latent_bounded_fields="alpha,phi", latent_bounded_formulation="transformed")
    assert _latent_bounded_fields(args, enable_phi_evolution=True) == ("alpha", "phi")


def test_refmap_transformed_latent_fields_keep_only_phi() -> None:
    args = SimpleNamespace(
        latent_bounded_fields="alpha,phi",
        latent_bounded_formulation="transformed",
        alpha_from_refmap=True,
    )
    assert _latent_bounded_fields(args, enable_phi_evolution=True) == ("phi",)


def test_create_problem_omits_phi_latent_when_only_alpha_is_latent() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=2,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
        latent_bounded_transport=True,
        latent_bounded_fields=("alpha",),
    )
    assert "alpha_latent" in problem["dh"].field_names
    assert "phi_latent" not in problem["dh"].field_names
    assert problem["alpha_latent_k"] is not None
    assert problem["phi_latent_k"] is None


def test_benchmark7_internal_conversion_top_drainage_keeps_support_and_B_transport_closed() -> None:
    problem = _build_full_problem()
    forms_closed = _build_full_forms(problem, ds_alpha_transport=None, ds_B_transport=None)
    ds_alpha_transport, ds_B_transport = _build_transport_measures(
        problem=problem,
        qdeg=6,
        enable_phi_evolution=True,
        top_drainage_transport=True,
        support_physics="internal_conversion",
    )
    assert ds_alpha_transport is None
    assert ds_B_transport is None
    forms_open = _build_full_forms(
        problem,
        ds_alpha_transport=ds_alpha_transport,
        ds_B_transport=ds_B_transport,
    )

    alpha_closed = _assemble_block(problem, forms_closed.r_alpha, "alpha")
    alpha_open = _assemble_block(problem, forms_open.r_alpha, "alpha")
    phi_closed = _assemble_block(problem, forms_closed.r_phi, "phi")
    phi_open = _assemble_block(problem, forms_open.r_phi, "phi")

    assert np.linalg.norm(alpha_open - alpha_closed, ord=np.inf) < 1.0e-12
    assert np.linalg.norm(phi_open - phi_closed, ord=np.inf) < 1.0e-12
    assert forms_open.r_detached is None


def test_benchmark7_legacy_top_transport_can_still_open_boundary_flux_terms() -> None:
    problem = _build_full_problem()
    ds_alpha_transport, ds_B_transport = _build_transport_measures(
        problem=problem,
        qdeg=6,
        enable_phi_evolution=True,
        top_drainage_transport=True,
        support_physics="legacy_exchange",
    )
    assert ds_alpha_transport is not None
    assert ds_B_transport is not None


def test_benchmark7_internal_conversion_can_optionally_open_only_top_B_transport() -> None:
    problem = _build_full_problem()
    ds_alpha_transport, ds_B_transport = _build_transport_measures(
        problem=problem,
        qdeg=6,
        enable_phi_evolution=True,
        top_drainage_transport=True,
        support_physics="internal_conversion",
        internal_conversion_open_top_b_transport=True,
    )
    assert ds_alpha_transport is None
    assert ds_B_transport is not None


def test_ratio_free_stored_support_alpha_vS_gate_reduces_advective_alpha_drive() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=2,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
        full_ratio_free_state=True,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.05, phi_b=0.18)
    problem["alpha_n"].nodal_values[:] = problem["alpha_k"].nodal_values[:]
    problem["v_k"].nodal_values[:] = 0.0
    problem["v_n"].nodal_values[:] = 0.0
    problem["p_k"].nodal_values[:] = 0.0
    problem["p_n"].nodal_values[:] = 0.0
    problem["p_pore_k"].nodal_values[:] = 0.0
    problem["p_pore_n"].nodal_values[:] = 0.0
    problem["u_k"].nodal_values[:] = 0.0
    problem["u_n"].nodal_values[:] = 0.0
    problem["mu_k"].nodal_values[:] = 0.0
    problem["mu_n"].nodal_values[:] = 0.0
    problem["S_k"].nodal_values[:] = 0.0
    problem["S_n"].nodal_values[:] = 0.0
    problem["vS_k"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))
    problem["vS_n"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))
    problem["B_k"].nodal_values[:] = 0.82 * np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = 0.82 * np.asarray(problem["alpha_n"].nodal_values, dtype=float)

    forms_plain = _build_ratio_free_full_forms(problem)
    forms_gate = _build_ratio_free_full_forms(
        problem,
        alpha_vS_gate_alpha0=0.5,
        alpha_vS_gate_power=8,
    )

    alpha_plain = _assemble_block(problem, forms_plain.r_alpha, "alpha")
    alpha_gate = _assemble_block(problem, forms_gate.r_alpha, "alpha")

    assert np.linalg.norm(alpha_gate, ord=np.inf) < np.linalg.norm(alpha_plain, ord=np.inf)


def test_ratio_free_stored_support_alpha_vS_gate_reduces_kinematic_drive() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=2,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
        full_ratio_free_state=True,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.05, phi_b=0.18)
    problem["alpha_n"].nodal_values[:] = problem["alpha_k"].nodal_values[:]
    problem["v_k"].nodal_values[:] = 0.0
    problem["v_n"].nodal_values[:] = 0.0
    problem["p_k"].nodal_values[:] = 0.0
    problem["p_n"].nodal_values[:] = 0.0
    problem["p_pore_k"].nodal_values[:] = 0.0
    problem["p_pore_n"].nodal_values[:] = 0.0
    problem["u_k"].nodal_values[:] = 0.0
    problem["u_n"].nodal_values[:] = 0.0
    problem["mu_k"].nodal_values[:] = 0.0
    problem["mu_n"].nodal_values[:] = 0.0
    problem["S_k"].nodal_values[:] = 0.0
    problem["S_n"].nodal_values[:] = 0.0
    problem["vS_k"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))
    problem["vS_n"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))
    problem["B_k"].nodal_values[:] = 0.82 * np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = 0.82 * np.asarray(problem["alpha_n"].nodal_values, dtype=float)

    forms_plain = _build_ratio_free_full_forms(problem)
    forms_gate = _build_ratio_free_full_forms(
        problem,
        alpha_vS_gate_alpha0=0.5,
        alpha_vS_gate_power=8,
    )

    u_plain = _assemble_block(problem, forms_plain.r_kinematics, "u_y")
    u_gate = _assemble_block(problem, forms_gate.r_kinematics, "u_y")

    assert np.linalg.norm(u_gate, ord=np.inf) < np.linalg.norm(u_plain, ord=np.inf)


def test_ratio_free_stored_support_pore_row_uses_skeleton_divergence_when_relative_flux_vanishes() -> None:
    problem = _build_ratio_free_full_problem()
    shared_velocity = lambda x, y: np.array([0.1 * x, 0.2 * y])
    for key in ("v_k", "v_n", "vS_k", "vS_n"):
        problem[key].set_values_from_function(shared_velocity)
    problem["p_k"].nodal_values[:] = 0.0
    problem["p_n"].nodal_values[:] = 0.0
    problem["p_pore_k"].nodal_values[:] = 0.0
    problem["p_pore_n"].nodal_values[:] = 0.0

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
    )

    pore_block = _assemble_block(problem, forms.r_pore, "p_pore")
    expected_form = Constant(0.3) * problem["q_pore_test"] * dx(metadata={"q": 6})
    expected_block = _assemble_block(problem, expected_form, "p_pore")

    assert np.allclose(pore_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_stored_support_pore_row_ignores_p_gradient_when_relative_flux_vanishes() -> None:
    problem = _build_ratio_free_full_problem()
    shared_velocity = lambda x, y: np.array([0.1 * x, 0.2 * y])
    for key in ("v_k", "v_n", "vS_k", "vS_n"):
        problem[key].set_values_from_function(shared_velocity)
    problem["p_k"].nodal_values[:] = 0.0
    problem["p_n"].nodal_values[:] = 0.0
    problem["p_pore_k"].nodal_values[:] = 0.0
    problem["p_pore_n"].nodal_values[:] = 0.0
    # Make the current pore fraction P = alpha - B spatially non-uniform so the
    # pore row sees the exact current-frame pore-content flux.
    problem["alpha_k"].set_values_from_function(lambda x, y: 0.8 + 0.1 * x)
    problem["alpha_n"].set_values_from_function(lambda x, y: 0.8 + 0.1 * x)
    problem["B_k"].set_values_from_function(lambda x, y: 0.55 + 0.05 * x)
    problem["B_n"].set_values_from_function(lambda x, y: 0.55 + 0.05 * x)

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
    )

    pore_block = _assemble_block(problem, forms.r_pore, "p_pore")
    direct_expected = _assemble_block(
        problem,
        problem["q_pore_test"]
        * (Constant(0.3) * problem["alpha_k"])
        * dx(metadata={"q": 6}),
        "p_pore",
    )
    assert np.allclose(pore_block, direct_expected, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_stored_support_pore_storage_row_uses_support_weight() -> None:
    problem = _build_ratio_free_full_problem()
    zero_vec = lambda x, y: np.array([0.0, 0.0])
    for key in ("v_k", "v_n", "vS_k", "vS_n"):
        problem[key].set_values_from_function(zero_vec)
    problem["p_n"].nodal_values[:] = 0.2
    problem["p_k"].nodal_values[:] = 1.2
    problem["p_pore_n"].nodal_values[:] = 0.2
    problem["p_pore_k"].nodal_values[:] = 1.2

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        storativity_c0=1.7,
        include_skeleton_acceleration=False,
    )

    pore_block = _assemble_block(problem, forms.r_pore, "p_pore")
    expected_coeff = 1.7 * (1.2 - 0.2) / 0.1
    expected_form = Constant(expected_coeff) * problem["q_pore_test"] * dx(metadata={"q": 6})
    expected_block = _assemble_block(problem, expected_form, "p_pore")

    assert np.allclose(pore_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_stored_support_mass_row_keeps_free_fluid_incompressibility() -> None:
    problem = _build_ratio_free_full_problem()
    for key in ("alpha_k", "alpha_n", "B_k", "B_n"):
        problem[key].nodal_values[:] = 0.0
    fluid_velocity = lambda x, y: np.array([0.1 * x, 0.2 * y])
    zero_vec = lambda x, y: np.array([0.0, 0.0])
    for key in ("v_k", "v_n"):
        problem[key].set_values_from_function(fluid_velocity)
    for key in ("vS_k", "vS_n"):
        problem[key].set_values_from_function(zero_vec)
    problem["p_k"].nodal_values[:] = 0.0
    problem["p_n"].nodal_values[:] = 0.0
    problem["p_pore_k"].nodal_values[:] = 17.0
    problem["p_pore_n"].nodal_values[:] = 17.0

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_biot=1.0,
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
    )

    mass_block = _assemble_block(problem, forms.r_mass, "p")
    expected_form = Constant(0.3) * problem["q_test"] * dx(metadata={"q": 6})
    expected_block = _assemble_block(problem, expected_form, "p")

    assert np.allclose(mass_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_stored_support_mass_row_includes_free_fluid_time_balance() -> None:
    problem = _build_ratio_free_full_problem()
    zero_vec = lambda x, y: np.array([0.0, 0.0])
    for key in ("v_k", "v_n", "vS_k", "vS_n"):
        problem[key].set_values_from_function(zero_vec)
    problem["alpha_n"].nodal_values[:] = 0.40
    problem["alpha_k"].nodal_values[:] = 0.55
    problem["B_n"].nodal_values[:] = 0.20
    problem["B_k"].nodal_values[:] = 0.20
    problem["p_k"].nodal_values[:] = 0.0
    problem["p_n"].nodal_values[:] = 0.0
    problem["p_pore_k"].nodal_values[:] = 0.0
    problem["p_pore_n"].nodal_values[:] = 0.0

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_biot=1.0,
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
    )

    mass_block = _assemble_block(problem, forms.r_mass, "p")
    expected_coeff = ((1.0 - 0.55) - (1.0 - 0.40)) / 0.1
    expected_form = Constant(expected_coeff) * problem["q_test"] * dx(metadata={"q": 6})
    expected_block = _assemble_block(problem, expected_form, "p")

    assert np.allclose(mass_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_stored_support_skeleton_pressure_uses_support_biot_coefficient() -> None:
    problem = _build_ratio_free_full_problem()
    zero_vec = lambda x, y: np.array([0.0, 0.0])
    for key in ("v_k", "v_n"):
        problem[key].set_values_from_function(zero_vec)
    for key in ("vS_k", "vS_n"):
        problem[key].set_values_from_function(lambda x, y: np.array([0.1 * x, 0.2 * y]))
    problem["alpha_k"].set_values_from_function(lambda x, y: 0.55 + 0.15 * x - 0.05 * y)
    problem["alpha_n"].set_values_from_function(lambda x, y: 0.55 + 0.15 * x - 0.05 * y)
    problem["B_k"].set_values_from_function(lambda x, y: 0.35 + 0.05 * x)
    problem["B_n"].set_values_from_function(lambda x, y: 0.35 + 0.05 * x)
    problem["p_k"].nodal_values[:] = 0.5
    problem["p_n"].nodal_values[:] = 0.5
    problem["p_pore_k"].nodal_values[:] = 2.0
    problem["p_pore_n"].nodal_values[:] = 2.0

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.3,
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
    )

    _, residual = assemble_form(
        Equation(None, forms.r_skeleton_terms["pressure"]),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    residual = np.asarray(residual, dtype=float)
    vS_x = np.asarray(problem["dh"].get_field_slice("vS_x"), dtype=int)
    vS_y = np.asarray(problem["dh"].get_field_slice("vS_y"), dtype=int)
    pressure_block = np.concatenate([residual[vS_x], residual[vS_y]])

    expected_form = -(
        Constant(1.3)
        * problem["p_pore_k"]
        * (problem["alpha_k"] * div(problem["vS_test"]) + dot(grad(problem["alpha_k"]), problem["vS_test"]))
    ) * dx(metadata={"q": 6})
    _, expected_residual = assemble_form(
        Equation(None, expected_form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    expected_residual = np.asarray(expected_residual, dtype=float)
    expected_block = np.concatenate([expected_residual[vS_x], expected_residual[vS_y]])

    assert np.allclose(pressure_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_stored_support_default_pore_row_uses_alpha_support_coefficient() -> None:
    problem = _build_ratio_free_full_problem()
    shared_velocity = lambda x, y: np.array([0.1 * x, 0.2 * y])
    for key in ("v_k", "v_n", "vS_k", "vS_n"):
        problem[key].set_values_from_function(shared_velocity)
    problem["alpha_k"].nodal_values[:] = 0.60
    problem["alpha_n"].nodal_values[:] = 0.60
    problem["B_k"].nodal_values[:] = 0.41
    problem["B_n"].nodal_values[:] = 0.41
    problem["p_k"].nodal_values[:] = 0.0
    problem["p_n"].nodal_values[:] = 0.0
    problem["p_pore_k"].nodal_values[:] = 0.0
    problem["p_pore_n"].nodal_values[:] = 0.0

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_biot=None,
        skeleton_pressure_mode="whole_domain",
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
    )

    pore_block = _assemble_block(problem, forms.r_pore, "p_pore")
    expected_form = (Constant(0.3) * problem["alpha_k"]) * problem["q_pore_test"] * dx(metadata={"q": 6})
    expected_block = _assemble_block(problem, expected_form, "p_pore")

    assert np.allclose(pore_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_stored_support_direct_darcy_pore_row_uses_alpha_weighted_pressure_diffusion() -> None:
    problem = _build_ratio_free_full_problem(split_pore_flux_model="direct_darcy")
    zero_vec = lambda x, y: np.array([0.0, 0.0])
    for key in ("v_k", "v_n", "vS_k", "vS_n"):
        problem[key].set_values_from_function(zero_vec)
    problem["alpha_k"].set_values_from_function(lambda x, y: 0.55 + 0.15 * x)
    problem["alpha_n"].set_values_from_function(lambda x, y: 0.55 + 0.15 * x)
    problem["B_k"].nodal_values[:] = 0.41
    problem["B_n"].nodal_values[:] = 0.41
    problem["p_k"].nodal_values[:] = 0.0
    problem["p_n"].nodal_values[:] = 0.0
    problem["p_pore_k"].set_values_from_function(lambda x, y: 0.2 + 0.3 * x)
    problem["p_pore_n"].set_values_from_function(lambda x, y: 0.2 + 0.3 * x)

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        storativity_c0=0.0,
        split_pore_flux_model="direct_darcy",
        include_skeleton_acceleration=False,
    )

    pore_block = _assemble_block(problem, forms.r_pore, "p_pore")
    flux_x = (Constant(1.0) / Constant(0.035 * 1.0e3)) * problem["alpha_k"] * Constant(0.3)
    expected_form = flux_x * _vector_component_expr(grad(problem["q_pore_test"]), 0) * dx(metadata={"q": 6})
    expected_block = _assemble_block(problem, expected_form, "p_pore")

    assert np.allclose(pore_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_stored_support_exact_conservative_pore_row_uses_timeP_and_divPvS() -> None:
    problem = _build_ratio_free_full_problem(split_pore_flux_model="exact_conservative_p")
    zero_vec = lambda x, y: np.array([0.0, 0.0])
    for key in ("v_k", "v_n"):
        problem[key].set_values_from_function(zero_vec)
    for key in ("vS_k", "vS_n"):
        problem[key].set_values_from_function(lambda x, y: np.array([0.1 * x, 0.2 * y]))
    problem["alpha_n"].set_values_from_function(lambda x, y: 0.76 + 0.04 * x + 0.01 * y)
    problem["alpha_k"].set_values_from_function(lambda x, y: 0.82 + 0.08 * x + 0.02 * y)
    problem["B_n"].set_values_from_function(lambda x, y: 0.48 + 0.03 * x)
    problem["B_k"].set_values_from_function(lambda x, y: 0.50 + 0.05 * x)
    problem["p_k"].nodal_values[:] = 0.0
    problem["p_n"].nodal_values[:] = 0.0
    problem["p_pore_k"].nodal_values[:] = 0.0
    problem["p_pore_n"].nodal_values[:] = 0.0

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        storativity_c0=0.0,
        split_pore_flux_model="exact_conservative_p",
        include_skeleton_acceleration=False,
    )

    pore_block = _assemble_block(problem, forms.r_pore, "p_pore")
    P_k = problem["alpha_k"] - problem["B_k"]
    P_n = problem["alpha_n"] - problem["B_n"]
    expected_form = problem["q_pore_test"] * (
        ((P_k - P_n) / Constant(0.1))
        + P_k * div(problem["vS_k"])
        + dot(grad(P_k), problem["vS_k"])
    ) * dx(metadata={"q": 6})
    expected_block = _assemble_block(problem, expected_form, "p_pore")

    assert np.allclose(pore_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_stored_support_exact_conservative_pore_row_keeps_direct_darcy_diffusion() -> None:
    problem = _build_ratio_free_full_problem(split_pore_flux_model="exact_conservative_p")
    zero_vec = lambda x, y: np.array([0.0, 0.0])
    for key in ("v_k", "v_n", "vS_k", "vS_n"):
        problem[key].set_values_from_function(zero_vec)
    problem["alpha_k"].set_values_from_function(lambda x, y: 0.55 + 0.15 * x)
    problem["alpha_n"].set_values_from_function(lambda x, y: 0.55 + 0.15 * x)
    problem["B_k"].nodal_values[:] = 0.41
    problem["B_n"].nodal_values[:] = 0.41
    problem["p_k"].nodal_values[:] = 0.0
    problem["p_n"].nodal_values[:] = 0.0
    problem["p_pore_k"].set_values_from_function(lambda x, y: 0.2 + 0.3 * x)
    problem["p_pore_n"].set_values_from_function(lambda x, y: 0.2 + 0.3 * x)

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        storativity_c0=0.0,
        split_pore_flux_model="exact_conservative_p",
        include_skeleton_acceleration=False,
    )

    pore_block = _assemble_block(problem, forms.r_pore, "p_pore")
    flux_x = (Constant(1.0) / Constant(0.035 * 1.0e3)) * problem["alpha_k"] * Constant(0.3)
    expected_form = flux_x * _vector_component_expr(grad(problem["q_pore_test"]), 0) * dx(metadata={"q": 6})
    expected_block = _assemble_block(problem, expected_form, "p_pore")

    assert np.allclose(pore_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_stored_support_exact_total_continuity_pore_row_couples_free_fluid_flux() -> None:
    problem = _build_ratio_free_full_problem(split_pore_flux_model="exact_total_continuity")
    zero_vec = lambda x, y: np.array([0.0, 0.0])
    for key in ("u_k", "u_n", "vS_k", "vS_n"):
        problem[key].set_values_from_function(zero_vec)
    for key in ("v_k", "v_n"):
        problem[key].set_values_from_function(lambda x, y: np.array([1.0 + 0.2 * x, 0.0]))
    problem["alpha_k"].set_values_from_function(lambda x, y: 0.55 + 0.15 * x)
    problem["alpha_n"].set_values_from_function(lambda x, y: 0.55 + 0.15 * x)
    problem["B_k"].set_values_from_function(lambda x, y: 0.30 + 0.10 * x)
    problem["B_n"].set_values_from_function(lambda x, y: 0.30 + 0.10 * x)
    problem["p_k"].nodal_values[:] = 0.0
    problem["p_n"].nodal_values[:] = 0.0
    problem["p_pore_k"].nodal_values[:] = 0.0
    problem["p_pore_n"].nodal_values[:] = 0.0

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        storativity_c0=0.0,
        split_pore_flux_model="exact_total_continuity",
        split_pore_momentum_model="band_alpha",
        include_skeleton_acceleration=False,
    )

    pore_block = _assemble_block(problem, forms.r_pore, "p_pore")
    expected_form = problem["q_pore_test"] * (
        div((Constant(1.0) - problem["alpha_k"]) * problem["v_k"])
        + div(problem["alpha_k"] * problem["vS_k"])
    ) * dx(metadata={"q": 6})
    expected_block = _assemble_block(problem, expected_form, "p_pore")

    assert np.allclose(pore_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_stored_support_legacy_reduced_divq_row_keeps_zero_when_only_pressure_gradient_is_present() -> None:
    problem = _build_ratio_free_full_problem(split_pore_flux_model="reduced_divq")
    zero_vec = lambda x, y: np.array([0.0, 0.0])
    for key in ("v_k", "v_n", "vS_k", "vS_n"):
        problem[key].set_values_from_function(zero_vec)
    problem["alpha_k"].set_values_from_function(lambda x, y: 0.55 + 0.15 * x)
    problem["alpha_n"].set_values_from_function(lambda x, y: 0.55 + 0.15 * x)
    problem["B_k"].nodal_values[:] = 0.41
    problem["B_n"].nodal_values[:] = 0.41
    problem["p_k"].nodal_values[:] = 0.0
    problem["p_n"].nodal_values[:] = 0.0
    problem["p_pore_k"].set_values_from_function(lambda x, y: 0.2 + 0.3 * x)
    problem["p_pore_n"].set_values_from_function(lambda x, y: 0.2 + 0.3 * x)

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        storativity_c0=0.0,
        split_pore_flux_model="reduced_divq",
        include_skeleton_acceleration=False,
    )

    pore_block = _assemble_block(problem, forms.r_pore, "p_pore")
    assert np.allclose(pore_block, 0.0, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_stored_support_default_skeleton_pressure_uses_whole_domain_support_coefficient() -> None:
    problem = _build_ratio_free_full_problem()
    zero_vec = lambda x, y: np.array([0.0, 0.0])
    for key in ("v_k", "v_n"):
        problem[key].set_values_from_function(zero_vec)
    for key in ("vS_k", "vS_n"):
        problem[key].set_values_from_function(lambda x, y: np.array([0.1 * x, 0.2 * y]))
    problem["p_k"].nodal_values[:] = 0.5
    problem["p_n"].nodal_values[:] = 0.5
    problem["p_pore_k"].nodal_values[:] = 2.0
    problem["p_pore_n"].nodal_values[:] = 2.0

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_biot=None,
        skeleton_pressure_mode="whole_domain",
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
    )

    _, residual = assemble_form(
        Equation(None, forms.r_skeleton_terms["pressure"]),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    residual = np.asarray(residual, dtype=float)
    vS_x = np.asarray(problem["dh"].get_field_slice("vS_x"), dtype=int)
    vS_y = np.asarray(problem["dh"].get_field_slice("vS_y"), dtype=int)
    pressure_block = np.concatenate([residual[vS_x], residual[vS_y]])

    expected_form = -(problem["B_k"] * problem["p_pore_k"] * div(problem["vS_test"])) * dx(metadata={"q": 6})
    _, expected_residual = assemble_form(
        Equation(None, expected_form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    expected_residual = np.asarray(expected_residual, dtype=float)
    expected_block = np.concatenate([expected_residual[vS_x], expected_residual[vS_y]])

    assert np.allclose(pressure_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_split_pore_band_alpha_momentum_loading_ignores_gradB() -> None:
    problem = _build_ratio_free_full_problem(split_pore_flux_model="exact_conservative_p")
    zero_vec = lambda x, y: np.array([0.0, 0.0])
    for key in ("v_k", "v_n", "vS_k", "vS_n", "u_k", "u_n"):
        problem[key].set_values_from_function(zero_vec)
    problem["alpha_k"].nodal_values[:] = 0.8
    problem["alpha_n"].nodal_values[:] = 0.8
    problem["B_k"].set_values_from_function(lambda x, y: 0.45 + 0.10 * x)
    problem["B_n"].set_values_from_function(lambda x, y: 0.45 + 0.10 * x)
    problem["p_k"].nodal_values[:] = 0.0
    problem["p_n"].nodal_values[:] = 0.0
    problem["p_pore_k"].nodal_values[:] = 2.0
    problem["p_pore_n"].nodal_values[:] = 2.0

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        storativity_c0=0.0,
        split_pore_flux_model="exact_conservative_p",
        split_pore_momentum_model="band_alpha",
        include_skeleton_acceleration=False,
    )

    _, residual = assemble_form(
        Equation(None, forms.r_momentum),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    residual = np.asarray(residual, dtype=float)
    v_x = np.asarray(problem["dh"].get_field_slice("v_x"), dtype=int)
    v_y = np.asarray(problem["dh"].get_field_slice("v_y"), dtype=int)
    fluid_block = np.concatenate([residual[v_x], residual[v_y]])

    assert np.allclose(fluid_block, 0.0, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_split_pore_band_alpha_momentum_loading_uses_only_grad_alpha() -> None:
    problem = _build_ratio_free_full_problem(split_pore_flux_model="exact_conservative_p")
    zero_vec = lambda x, y: np.array([0.0, 0.0])
    for key in ("v_k", "v_n", "vS_k", "vS_n", "u_k", "u_n"):
        problem[key].set_values_from_function(zero_vec)
    problem["alpha_k"].set_values_from_function(lambda x, y: 0.55 + 0.15 * x)
    problem["alpha_n"].set_values_from_function(lambda x, y: 0.55 + 0.15 * x)
    problem["B_k"].set_values_from_function(lambda x, y: 0.30 + 0.20 * x)
    problem["B_n"].set_values_from_function(lambda x, y: 0.30 + 0.20 * x)
    problem["p_k"].nodal_values[:] = 0.0
    problem["p_n"].nodal_values[:] = 0.0
    problem["p_pore_k"].nodal_values[:] = 1.5
    problem["p_pore_n"].nodal_values[:] = 1.5

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        storativity_c0=0.0,
        split_pore_flux_model="exact_conservative_p",
        split_pore_momentum_model="band_alpha",
        include_skeleton_acceleration=False,
    )

    _, residual = assemble_form(
        Equation(None, forms.r_momentum),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    residual = np.asarray(residual, dtype=float)
    v_x = np.asarray(problem["dh"].get_field_slice("v_x"), dtype=int)
    v_y = np.asarray(problem["dh"].get_field_slice("v_y"), dtype=int)
    fluid_block = np.concatenate([residual[v_x], residual[v_y]])

    expected_form = -(problem["p_pore_k"] * dot(grad(problem["alpha_k"]), problem["v_test"])) * dx(metadata={"q": 6})
    _, expected_residual = assemble_form(
        Equation(None, expected_form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    expected_residual = np.asarray(expected_residual, dtype=float)
    expected_block = np.concatenate([expected_residual[v_x], expected_residual[v_y]])

    assert np.allclose(fluid_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_split_pore_band_alpha_skeleton_receives_free_fluid_pressure_band() -> None:
    problem = _build_ratio_free_full_problem(split_pore_flux_model="exact_conservative_p")
    zero_vec = lambda x, y: np.array([0.0, 0.0])
    for key in ("v_k", "v_n", "vS_k", "vS_n", "u_k", "u_n"):
        problem[key].set_values_from_function(zero_vec)
    problem["alpha_k"].set_values_from_function(lambda x, y: 0.55 + 0.15 * x)
    problem["alpha_n"].set_values_from_function(lambda x, y: 0.55 + 0.15 * x)
    problem["B_k"].set_values_from_function(lambda x, y: 0.30 + 0.20 * x)
    problem["B_n"].set_values_from_function(lambda x, y: 0.30 + 0.20 * x)
    problem["p_k"].nodal_values[:] = 2.0
    problem["p_n"].nodal_values[:] = 2.0
    problem["p_pore_k"].nodal_values[:] = 0.0
    problem["p_pore_n"].nodal_values[:] = 0.0

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        storativity_c0=0.0,
        split_pore_flux_model="exact_conservative_p",
        split_pore_momentum_model="band_alpha",
        include_skeleton_acceleration=False,
    )

    _, residual = assemble_form(
        Equation(None, forms.r_skeleton_terms["fluid_interface_traction"]),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    residual = np.asarray(residual, dtype=float)
    vS_x = np.asarray(problem["dh"].get_field_slice("vS_x"), dtype=int)
    vS_y = np.asarray(problem["dh"].get_field_slice("vS_y"), dtype=int)
    skeleton_block = np.concatenate([residual[vS_x], residual[vS_y]])

    expected_form = -(problem["p_k"] * dot(grad(problem["alpha_k"]), problem["vS_test"])) * dx(metadata={"q": 6})
    _, expected_residual = assemble_form(
        Equation(None, expected_form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    expected_residual = np.asarray(expected_residual, dtype=float)
    expected_block = np.concatenate([expected_residual[vS_x], expected_residual[vS_y]])

    assert np.allclose(skeleton_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_stored_support_fluid_momentum_uses_free_fluid_pressure_off_support() -> None:
    problem = _build_ratio_free_full_problem()
    for key in ("alpha_k", "alpha_n", "B_k", "B_n"):
        problem[key].nodal_values[:] = 0.0
    zero_vec = lambda x, y: np.array([0.0, 0.0])
    for key in ("v_k", "v_n", "vS_k", "vS_n"):
        problem[key].set_values_from_function(zero_vec)
    problem["p_k"].nodal_values[:] = 3.0
    problem["p_n"].nodal_values[:] = 3.0
    problem["p_pore_k"].nodal_values[:] = 19.0
    problem["p_pore_n"].nodal_values[:] = 19.0

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_biot=1.0,
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
    )

    _, residual = assemble_form(
        Equation(None, forms.r_momentum_terms["pressure"]),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    residual = np.asarray(residual, dtype=float)
    v_x = np.asarray(problem["dh"].get_field_slice("v_x"), dtype=int)
    v_y = np.asarray(problem["dh"].get_field_slice("v_y"), dtype=int)
    pressure_block = np.concatenate([residual[v_x], residual[v_y]])

    expected_form = -(problem["p_k"] * div(problem["v_test"])) * dx(metadata={"q": 6})
    _, expected_residual = assemble_form(
        Equation(None, expected_form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    expected_residual = np.asarray(expected_residual, dtype=float)
    expected_block = np.concatenate([expected_residual[v_x], expected_residual[v_y]])

    assert np.allclose(pressure_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_stored_support_fluid_transient_uses_free_fluid_fraction() -> None:
    problem = _build_ratio_free_full_problem()
    for key in ("alpha_k", "alpha_n"):
        problem[key].nodal_values[:] = 0.5
    for key in ("B_k", "B_n"):
        problem[key].nodal_values[:] = 0.41
    zero_vec = lambda x, y: np.array([0.0, 0.0])
    for key in ("vS_k", "vS_n"):
        problem[key].set_values_from_function(zero_vec)
    problem["v_n"].set_values_from_function(zero_vec)
    problem["v_k"].set_values_from_function(lambda x, y: np.array([1.0, -2.0]))

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_biot=1.0,
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
    )

    _, residual = assemble_form(
        Equation(None, forms.r_momentum_terms["transient"]),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    residual = np.asarray(residual, dtype=float)
    v_x = np.asarray(problem["dh"].get_field_slice("v_x"), dtype=int)
    v_y = np.asarray(problem["dh"].get_field_slice("v_y"), dtype=int)
    transient_block = np.concatenate([residual[v_x], residual[v_y]])

    expected_form = (
        Constant(10.0)
        * (Constant(1.0) - problem["alpha_k"])
        * inner(problem["v_k"] - problem["v_n"], problem["v_test"])
    ) * dx(metadata={"q": 6})
    _, expected_residual = assemble_form(
        Equation(None, expected_form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    expected_residual = np.asarray(expected_residual, dtype=float)
    expected_block = np.concatenate([expected_residual[v_x], expected_residual[v_y]])

    assert np.allclose(transient_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_stored_support_fluid_viscosity_vanishes_in_porous_bulk() -> None:
    problem = _build_ratio_free_full_problem()
    for key in ("alpha_k", "alpha_n"):
        problem[key].nodal_values[:] = 1.0
    for key in ("B_k", "B_n"):
        problem[key].nodal_values[:] = 0.82
    problem["v_k"].set_values_from_function(lambda x, y: np.array([x, -y]))
    problem["v_n"].set_values_from_function(lambda x, y: np.array([x, -y]))

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_biot=1.0,
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
    )

    _, residual = assemble_form(
        Equation(None, forms.r_momentum_terms["viscous"]),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    residual = np.asarray(residual, dtype=float)
    v_x = np.asarray(problem["dh"].get_field_slice("v_x"), dtype=int)
    v_y = np.asarray(problem["dh"].get_field_slice("v_y"), dtype=int)
    viscous_block = np.concatenate([residual[v_x], residual[v_y]])

    assert np.allclose(viscous_block, 0.0, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_pressure_interface_closure_uses_support_bearing_pore_pressure_extension() -> None:
    problem = _build_ratio_free_full_problem()
    zero_vec = lambda x, y: np.array([0.0, 0.0])
    for key in ("v_k", "v_n", "vS_k", "vS_n"):
        problem[key].set_values_from_function(zero_vec)
    for key in ("alpha_k", "alpha_n"):
        problem[key].nodal_values[:] = 0.5
    for key in ("B_k", "B_n"):
        problem[key].nodal_values[:] = 0.41
    problem["p_k"].nodal_values[:] = 3.0
    problem["p_n"].nodal_values[:] = 3.0
    problem["p_pore_k"].nodal_values[:] = -2.0
    problem["p_pore_n"].nodal_values[:] = -2.0

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_biot=None,
        skeleton_pressure_mode="whole_domain",
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
        pressure_interface_closure=True,
        pressure_interface_closure_strength=2.5,
    )

    pressure_interface_form = problem["_pressure_interface_residual_form"]
    assert pressure_interface_form is not None

    p_block = _assemble_block(problem, pressure_interface_form, "p")
    p_pore_block = _assemble_block(problem, pressure_interface_form, "p_pore")
    expected_form = (
        Constant(2.5)
        * Constant(4.0)
        * problem["alpha_k"]
        * (Constant(1.0) - problem["alpha_k"])
        * (problem["p_k"] - problem["p_pore_k"])
        * (problem["q_test"] - problem["q_pore_test"])
        * dx(metadata={"q": 6})
    )
    expected_p_block = _assemble_block(problem, expected_form, "p")
    expected_p_pore_block = _assemble_block(problem, expected_form, "p_pore")

    assert np.allclose(p_block, expected_p_block, atol=1.0e-12, rtol=1.0e-12)
    assert np.allclose(p_pore_block, expected_p_pore_block, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_p_pore_fluid_gauge_pins_off_support_extension_only() -> None:
    problem = _build_ratio_free_full_problem()
    zero_vec = lambda x, y: np.array([0.0, 0.0])
    for key in ("v_k", "v_n", "vS_k", "vS_n"):
        problem[key].set_values_from_function(zero_vec)
    for key in ("alpha_k", "alpha_n", "B_k", "B_n"):
        problem[key].nodal_values[:] = 0.0
    problem["p_k"].nodal_values[:] = 0.0
    problem["p_n"].nodal_values[:] = 0.0
    problem["p_pore_k"].nodal_values[:] = 3.0
    problem["p_pore_n"].nodal_values[:] = 3.0

    _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_biot=1.0,
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
        pressure_interface_closure=False,
        p_pore_fluid_gauge=True,
        p_pore_fluid_gauge_strength=2.5,
        interface_entry_closure=False,
        interface_bjs_closure=False,
    )

    gauge_form = problem["_p_pore_fluid_gauge_residual_form"]
    assert gauge_form is not None
    pore_block = _assemble_block(problem, gauge_form, "p_pore")
    expected_form = Constant(2.5) * problem["p_pore_k"] * problem["q_pore_test"] * dx(metadata={"q": 6})
    expected_block = _assemble_block(problem, expected_form, "p_pore")
    assert np.allclose(pore_block, expected_block, atol=1.0e-12, rtol=1.0e-12)

    for key in ("alpha_k", "alpha_n"):
        problem[key].nodal_values[:] = 1.0
    _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_biot=1.0,
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
        pressure_interface_closure=False,
        p_pore_fluid_gauge=True,
        p_pore_fluid_gauge_strength=2.5,
        interface_entry_closure=False,
        interface_bjs_closure=False,
    )
    zero_block = _assemble_block(problem, problem["_p_pore_fluid_gauge_residual_form"], "p_pore")
    assert np.allclose(zero_block, 0.0, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_entry_interface_closure_adds_normal_interface_force_from_pore_pressure_and_flux() -> None:
    problem = _build_ratio_free_full_problem()
    alpha_coords = np.asarray(problem["dh"].get_dof_coords("alpha"), dtype=float)
    alpha_profile = 0.5 + 0.1 * (alpha_coords[:, 1] - np.mean(alpha_coords[:, 1]))
    for key in ("alpha_k", "alpha_n"):
        problem[key].nodal_values[:] = alpha_profile
    for key in ("B_k", "B_n"):
        problem[key].nodal_values[:] = 0.4
    for key in ("vS_k", "vS_n"):
        problem[key].set_values_from_function(lambda x, y: np.array([0.0, 0.0]))
    for key in ("v_k", "v_n"):
        problem[key].set_values_from_function(lambda x, y: np.array([0.2 * y, 0.1 * y]))
    problem["p_k"].nodal_values[:] = 1.25
    problem["p_n"].nodal_values[:] = 1.25
    problem["p_pore_k"].nodal_values[:] = 0.75
    problem["p_pore_n"].nodal_values[:] = 0.75

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_biot=1.0,
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
        pressure_interface_closure=False,
        interface_entry_closure=True,
        interface_entry_closure_strength=2.0,
        interface_entry_delta=10.0,
        interface_bjs_closure=False,
    )

    entry_form = problem["_entry_interface_residual_form"]
    assert entry_form is not None
    _, residual = assemble_form(
        Equation(None, entry_form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    residual = np.asarray(residual, dtype=float)
    v_x = np.asarray(problem["dh"].get_field_slice("v_x"), dtype=int)
    v_y = np.asarray(problem["dh"].get_field_slice("v_y"), dtype=int)
    vS_x = np.asarray(problem["dh"].get_field_slice("vS_x"), dtype=int)
    vS_y = np.asarray(problem["dh"].get_field_slice("vS_y"), dtype=int)
    p_pore = np.asarray(problem["dh"].get_field_slice("p_pore"), dtype=int)
    assert np.linalg.norm(np.concatenate([residual[v_x], residual[v_y]]), ord=np.inf) > 0.0
    assert np.linalg.norm(np.concatenate([residual[vS_x], residual[vS_y]]), ord=np.inf) > 0.0
    assert np.allclose(residual[p_pore], 0.0, atol=1.0e-12, rtol=1.0e-12)

    n_if = _interface_unit_normal_2d(problem["alpha_n"], eta=1.0e-4)
    weight = Constant(2.0 / 0.1) * Constant(4.0) * problem["alpha_n"] * (Constant(1.0) - problem["alpha_n"])
    P_k = problem["alpha_k"] - problem["B_k"]
    rel_n = _dot_basis_2d(problem["v_k"], n_if) - _dot_basis_2d(problem["vS_k"], n_if)
    rel_n_test = _dot_basis_2d(problem["v_test"], n_if) - _dot_basis_2d(problem["vS_test"], n_if)
    entry_drive = problem["p_pore_k"] + Constant(10.0) * P_k * rel_n
    expected_form = weight * (entry_drive * rel_n_test) * dx(metadata={"q": 6})
    _, expected_residual = assemble_form(
        Equation(None, expected_form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    expected_residual = np.asarray(expected_residual, dtype=float)
    assert np.allclose(residual, expected_residual, atol=1.0e-10, rtol=5.0e-8)


def test_ratio_free_bjs_interface_closure_adds_tangential_fluid_and_solid_residuals() -> None:
    problem = _build_ratio_free_full_problem()
    alpha_coords = np.asarray(problem["dh"].get_dof_coords("alpha"), dtype=float)
    alpha_profile = 0.5 + 0.1 * (alpha_coords[:, 1] - np.mean(alpha_coords[:, 1]))
    for key in ("alpha_k", "alpha_n"):
        problem[key].nodal_values[:] = alpha_profile
    for key in ("B_k", "B_n"):
        problem[key].nodal_values[:] = 0.4
    for key in ("vS_k", "vS_n"):
        problem[key].set_values_from_function(lambda x, y: np.array([0.0, 0.0]))
    for key in ("v_k", "v_n"):
        problem[key].set_values_from_function(lambda x, y: np.array([0.2 * y, 0.0]))
    problem["p_k"].nodal_values[:] = 0.0
    problem["p_n"].nodal_values[:] = 0.0
    problem["p_pore_k"].nodal_values[:] = 0.0
    problem["p_pore_n"].nodal_values[:] = 0.0

    _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_biot=1.0,
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=True,
        interface_bjs_closure_strength=3.0,
        interface_bjs_gamma=1.0e3,
    )

    bjs_form = problem["_bjs_interface_residual_form"]
    assert bjs_form is not None
    _, residual = assemble_form(
        Equation(None, bjs_form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    residual = np.asarray(residual, dtype=float)
    v_x = np.asarray(problem["dh"].get_field_slice("v_x"), dtype=int)
    vS_x = np.asarray(problem["dh"].get_field_slice("vS_x"), dtype=int)
    assert np.linalg.norm(residual[v_x], ord=np.inf) > 0.0
    assert np.linalg.norm(residual[vS_x], ord=np.inf) > 0.0


def test_ratio_free_velocity_interface_closure_adds_fluid_and_solid_residuals() -> None:
    problem = _build_ratio_free_full_problem()
    alpha_coords = np.asarray(problem["dh"].get_dof_coords("alpha"), dtype=float)
    alpha_profile = 0.5 + 0.1 * (alpha_coords[:, 1] - np.mean(alpha_coords[:, 1]))
    for key in ("alpha_k", "alpha_n"):
        problem[key].nodal_values[:] = alpha_profile
    for key in ("B_k", "B_n"):
        problem[key].nodal_values[:] = 0.4
    for key in ("vS_k", "vS_n"):
        problem[key].set_values_from_function(lambda x, y: np.array([0.0, 0.0]))
    for key in ("v_k", "v_n"):
        problem[key].set_values_from_function(lambda x, y: np.array([0.2 * y, -0.1 * y]))

    _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_biot=1.0,
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=True,
        interface_velocity_normal_strength=2.0,
        interface_velocity_tangential_strength=3.0,
    )

    velocity_form = problem["_velocity_interface_residual_form"]
    assert velocity_form is not None
    _, residual = assemble_form(
        Equation(None, velocity_form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    residual = np.asarray(residual, dtype=float)
    v_x = np.asarray(problem["dh"].get_field_slice("v_x"), dtype=int)
    v_y = np.asarray(problem["dh"].get_field_slice("v_y"), dtype=int)
    vS_x = np.asarray(problem["dh"].get_field_slice("vS_x"), dtype=int)
    vS_y = np.asarray(problem["dh"].get_field_slice("vS_y"), dtype=int)
    assert np.linalg.norm(np.concatenate([residual[v_x], residual[v_y]]), ord=np.inf) > 0.0
    assert np.linalg.norm(np.concatenate([residual[vS_x], residual[vS_y]]), ord=np.inf) > 0.0


def test_ratio_free_traction_interface_closure_adds_fluid_and_solid_residuals() -> None:
    problem = _build_ratio_free_full_problem()
    alpha_coords = np.asarray(problem["dh"].get_dof_coords("alpha"), dtype=float)
    alpha_profile = 0.5 + 0.1 * (alpha_coords[:, 1] - np.mean(alpha_coords[:, 1]))
    for key in ("alpha_k", "alpha_n"):
        problem[key].nodal_values[:] = alpha_profile
    for key in ("B_k", "B_n"):
        problem[key].nodal_values[:] = 0.4
    for key in ("u_k", "u_n"):
        problem[key].set_values_from_function(lambda x, y: np.array([0.0, 0.01 * y]))
    for key in ("vS_k", "vS_n"):
        problem[key].set_values_from_function(lambda x, y: np.array([0.0, 0.0]))
    for key in ("v_k", "v_n"):
        problem[key].set_values_from_function(lambda x, y: np.array([0.15 * y, 0.05 * y]))
    problem["p_k"].nodal_values[:] = 1.25
    problem["p_n"].nodal_values[:] = 1.25
    problem["p_pore_k"].nodal_values[:] = 0.75
    problem["p_pore_n"].nodal_values[:] = 0.75

    _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_biot=1.0,
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=False,
        interface_traction_continuity_closure=True,
        interface_traction_normal_strength=2.0,
        interface_traction_tangential_strength=3.0,
    )

    traction_form = problem["_traction_interface_residual_form"]
    assert traction_form is not None
    _, residual = assemble_form(
        Equation(None, traction_form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    residual = np.asarray(residual, dtype=float)
    v_x = np.asarray(problem["dh"].get_field_slice("v_x"), dtype=int)
    v_y = np.asarray(problem["dh"].get_field_slice("v_y"), dtype=int)
    vS_x = np.asarray(problem["dh"].get_field_slice("vS_x"), dtype=int)
    vS_y = np.asarray(problem["dh"].get_field_slice("vS_y"), dtype=int)
    assert np.linalg.norm(np.concatenate([residual[v_x], residual[v_y]]), ord=np.inf) > 0.0
    assert np.linalg.norm(np.concatenate([residual[vS_x], residual[vS_y]]), ord=np.inf) > 0.0


def test_ratio_free_primary_q_pore_row_uses_div_q_flux() -> None:
    problem = _build_ratio_free_full_problem(
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
    )
    problem["v_k"].set_values_from_function(lambda x, y: np.array([y, 0.0]))
    problem["v_n"].set_values_from_function(lambda x, y: np.array([y, 0.0]))
    for key in ("vS_k", "vS_n", "u_k", "u_n"):
        problem[key].nodal_values[:] = 0.0
    for key in ("p_k", "p_n", "p_pore_k", "p_pore_n"):
        problem[key].nodal_values[:] = 0.0
    problem["lambda_drag_k"].set_values_from_function(lambda x, y: np.array([x, 0.0]))
    problem["lambda_drag_n"].set_values_from_function(lambda x, y: np.array([x, 0.0]))

    forms = _build_ratio_free_full_forms(
        problem,
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        storativity_c0=0.0,
    )

    _, residual = assemble_form(
        Equation(None, forms.r_pore),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    residual = np.asarray(residual, dtype=float)
    _, expected = assemble_form(
        Equation(None, problem["q_pore_test"] * Constant(1.0) * dx(metadata={"q": 6})),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    expected = np.asarray(expected, dtype=float)
    assert np.allclose(residual, expected, rtol=0.0, atol=1.0e-10)


def test_ratio_free_primary_q_exact_interface_pressure_transfer_loads_momentum_and_q_rows() -> None:
    problem = _build_ratio_free_full_problem(
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.1, phi_b=0.18)
    alpha_n = np.asarray(problem["alpha_n"].nodal_values, dtype=float)
    alpha_k = np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_n, 0.0, alpha_n)
    problem["B_k"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_k, 0.0, alpha_k)
    problem["v_k"].set_values_from_function(lambda x, y: np.array([y, 0.0]))
    problem["v_n"].set_values_from_function(lambda x, y: np.array([y, 0.0]))
    for key in ("vS_k", "vS_n", "u_k", "u_n"):
        problem[key].nodal_values[:] = 0.0
    problem["p_k"].nodal_values[:] = 0.0
    problem["p_n"].nodal_values[:] = 0.0
    problem["p_pore_k"].nodal_values[:] = 0.75
    problem["p_pore_n"].nodal_values[:] = 0.75
    problem["lambda_drag_k"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))
    problem["lambda_drag_n"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))

    _build_ratio_free_full_forms(
        problem,
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=False,
        interface_traction_continuity_closure=False,
        storativity_c0=0.0,
    )

    pressure_form = problem["_exact_interface_pressure_residual_form"]
    assert pressure_form is not None
    vx_res = _assemble_block(problem, pressure_form, "v_x")
    vy_res = _assemble_block(problem, pressure_form, "v_y")
    pp_res = _assemble_block(problem, pressure_form, "p_pore")
    lam_x_res = _assemble_block(problem, pressure_form, "lambda_drag_x")
    lam_y_res = _assemble_block(problem, pressure_form, "lambda_drag_y")
    assert np.linalg.norm(np.concatenate([vx_res, vy_res]), ord=np.inf) > 0.0
    assert np.linalg.norm(pp_res, ord=np.inf) == 0.0
    assert np.linalg.norm(np.concatenate([lam_x_res, lam_y_res]), ord=np.inf) > 0.0

    n_if = _interface_unit_normal_2d(problem["alpha_n"], eta=1.0e-4)
    weight_n = Constant(1.0 / 0.05) * Constant(4.0) * problem["alpha_n"] * (Constant(1.0) - problem["alpha_n"])
    weight_k = Constant(1.0 / 0.05) * Constant(4.0) * problem["alpha_k"] * (Constant(1.0) - problem["alpha_k"])
    q_n_test = _dot_basis_2d(problem["lambda_drag_test"], n_if)
    rel_n_test = _dot_basis_2d(problem["v_test"], n_if) - _dot_basis_2d(problem["vS_test"], n_if)
    expected_form = (
        weight_n * (problem["p_pore_k"] * rel_n_test)
        + weight_k * (problem["p_pore_k"] * q_n_test)
    ) * dx(metadata={"q": 6})
    _, expected_residual = assemble_form(
        Equation(None, expected_form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    expected_residual = np.asarray(expected_residual, dtype=float)
    _, residual = assemble_form(
        Equation(None, pressure_form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    residual = np.asarray(residual, dtype=float)
    assert np.allclose(residual, expected_residual, atol=1.0e-10, rtol=5.0e-8)


def test_ratio_free_primary_q_entry_closure_adds_only_darcy_robin_q_row() -> None:
    problem = _build_ratio_free_full_problem(
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.1, phi_b=0.18)
    alpha_n = np.asarray(problem["alpha_n"].nodal_values, dtype=float)
    alpha_k = np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_n, 0.0, alpha_n)
    problem["B_k"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_k, 0.0, alpha_k)
    problem["v_k"].set_values_from_function(lambda x, y: np.array([y, 0.0]))
    problem["v_n"].set_values_from_function(lambda x, y: np.array([y, 0.0]))
    for key in ("vS_k", "vS_n", "u_k", "u_n"):
        problem[key].nodal_values[:] = 0.0
    problem["p_k"].nodal_values[:] = 0.0
    problem["p_n"].nodal_values[:] = 0.0
    problem["p_pore_k"].nodal_values[:] = 0.75
    problem["p_pore_n"].nodal_values[:] = 0.75
    problem["lambda_drag_k"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))
    problem["lambda_drag_n"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))

    _build_ratio_free_full_forms(
        problem,
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        pressure_interface_closure=False,
        interface_entry_closure=True,
        interface_entry_closure_strength=1.0,
        interface_entry_delta=10.0,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=False,
        interface_traction_continuity_closure=False,
        storativity_c0=0.0,
    )

    entry_form = problem["_entry_interface_residual_form"]
    assert entry_form is not None
    vx_res = _assemble_block(problem, entry_form, "v_x")
    vy_res = _assemble_block(problem, entry_form, "v_y")
    vS_x_res = _assemble_block(problem, entry_form, "vS_x")
    vS_y_res = _assemble_block(problem, entry_form, "vS_y")
    pp_res = _assemble_block(problem, entry_form, "p_pore")
    lam_x_res = _assemble_block(problem, entry_form, "lambda_drag_x")
    lam_y_res = _assemble_block(problem, entry_form, "lambda_drag_y")
    assert np.linalg.norm(np.concatenate([vx_res, vy_res]), ord=np.inf) == 0.0
    assert np.linalg.norm(np.concatenate([vS_x_res, vS_y_res]), ord=np.inf) == 0.0
    assert np.linalg.norm(pp_res, ord=np.inf) == 0.0
    assert np.linalg.norm(np.concatenate([lam_x_res, lam_y_res]), ord=np.inf) > 0.0

    n_if = _interface_unit_normal_2d(problem["alpha_n"], eta=1.0e-4)
    weight_k = Constant(1.0 / 0.05) * Constant(4.0) * problem["alpha_k"] * (Constant(1.0) - problem["alpha_k"])
    q_n = _dot_basis_2d(problem["lambda_drag_k"], n_if)
    q_n_test = _dot_basis_2d(problem["lambda_drag_test"], n_if)
    expected_form = weight_k * (Constant(10.0) * q_n * q_n_test) * dx(metadata={"q": 6})
    _, expected_residual = assemble_form(
        Equation(None, expected_form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    expected_residual = np.asarray(expected_residual, dtype=float)
    _, residual = assemble_form(
        Equation(None, entry_form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    residual = np.asarray(residual, dtype=float)
    assert np.allclose(residual, expected_residual, atol=1.0e-10, rtol=5.0e-8)


def test_ratio_free_primary_q_entry_delta_zero_adds_no_extra_residual() -> None:
    problem = _build_ratio_free_full_problem(
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.1, phi_b=0.18)
    alpha_n = np.asarray(problem["alpha_n"].nodal_values, dtype=float)
    alpha_k = np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_n, 0.0, alpha_n)
    problem["B_k"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_k, 0.0, alpha_k)
    for key in ("vS_k", "vS_n", "u_k", "u_n"):
        problem[key].nodal_values[:] = 0.0
    problem["p_pore_k"].nodal_values[:] = 0.75
    problem["p_pore_n"].nodal_values[:] = 0.75
    problem["lambda_drag_k"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))
    problem["lambda_drag_n"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))

    _build_ratio_free_full_forms(
        problem,
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        pressure_interface_closure=False,
        interface_entry_closure=True,
        interface_entry_closure_strength=1.0,
        interface_entry_delta=0.0,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=False,
        interface_traction_continuity_closure=False,
        storativity_c0=0.0,
    )

    entry_form = problem["_entry_interface_residual_form"]
    assert entry_form is not None
    vx_res = _assemble_block(problem, entry_form, "v_x")
    vy_res = _assemble_block(problem, entry_form, "v_y")
    vS_x_res = _assemble_block(problem, entry_form, "vS_x")
    vS_y_res = _assemble_block(problem, entry_form, "vS_y")
    pp_res = _assemble_block(problem, entry_form, "p_pore")
    lam_x_res = _assemble_block(problem, entry_form, "lambda_drag_x")
    lam_y_res = _assemble_block(problem, entry_form, "lambda_drag_y")
    assert np.linalg.norm(np.concatenate([vx_res, vy_res]), ord=np.inf) == 0.0
    assert np.linalg.norm(np.concatenate([vS_x_res, vS_y_res]), ord=np.inf) == 0.0
    assert np.linalg.norm(pp_res, ord=np.inf) == 0.0
    assert np.linalg.norm(np.concatenate([lam_x_res, lam_y_res]), ord=np.inf) == 0.0
    _, residual = assemble_form(
        Equation(None, entry_form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    residual = np.asarray(residual, dtype=float)
    assert np.allclose(residual, 0.0, atol=1.0e-12, rtol=1.0e-12)

def test_ratio_free_primary_q_velocity_continuity_closure_loads_q_rows() -> None:
    problem = _build_ratio_free_full_problem(
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.1, phi_b=0.18)
    alpha_n = np.asarray(problem["alpha_n"].nodal_values, dtype=float)
    alpha_k = np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_n, 0.0, alpha_n)
    problem["B_k"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_k, 0.0, alpha_k)
    problem["v_k"].set_values_from_function(lambda x, y: np.array([y, 0.0]))
    problem["v_n"].set_values_from_function(lambda x, y: np.array([y, 0.0]))
    for key in ("vS_k", "vS_n", "u_k", "u_n"):
        problem[key].nodal_values[:] = 0.0
    problem["lambda_drag_k"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))
    problem["lambda_drag_n"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))

    _build_ratio_free_full_forms(
        problem,
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=True,
        interface_velocity_method="penalty",
        interface_velocity_normal_strength=1.0,
        interface_velocity_tangential_strength=0.0,
        storativity_c0=0.0,
    )

    velocity_form = problem["_velocity_interface_residual_form"]
    assert velocity_form is not None
    vx_res = _assemble_block(problem, velocity_form, "v_x")
    vy_res = _assemble_block(problem, velocity_form, "v_y")
    vS_x_res = _assemble_block(problem, velocity_form, "vS_x")
    vS_y_res = _assemble_block(problem, velocity_form, "vS_y")
    assert np.linalg.norm(np.concatenate([vx_res, vy_res]), ord=np.inf) == 0.0
    assert np.linalg.norm(np.concatenate([vS_x_res, vS_y_res]), ord=np.inf) == 0.0
    assert _primary_q_residual_norm(problem, velocity_form) > 0.0


def test_ratio_free_primary_q_velocity_continuity_nitsche_loads_pore_rows() -> None:
    problem = _build_ratio_free_full_problem(
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.1, phi_b=0.18)
    alpha_n = np.asarray(problem["alpha_n"].nodal_values, dtype=float)
    alpha_k = np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_n, 0.0, alpha_n)
    problem["B_k"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_k, 0.0, alpha_k)
    problem["v_k"].set_values_from_function(lambda x, y: np.array([y, 0.0]))
    problem["v_n"].set_values_from_function(lambda x, y: np.array([y, 0.0]))
    for key in ("vS_k", "vS_n", "u_k", "u_n"):
        problem[key].nodal_values[:] = 0.0
    problem["lambda_drag_k"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))
    problem["lambda_drag_n"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))

    _build_ratio_free_full_forms(
        problem,
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=True,
        interface_velocity_method="nitsche",
        interface_velocity_normal_strength=1.0,
        interface_velocity_tangential_strength=0.0,
        storativity_c0=0.0,
    )

    velocity_form = problem["_velocity_interface_residual_form"]
    assert velocity_form is not None
    pp_res = _assemble_block(problem, velocity_form, "p_pore")
    vx_res = _assemble_block(problem, velocity_form, "v_x")
    vy_res = _assemble_block(problem, velocity_form, "v_y")
    assert np.linalg.norm(pp_res, ord=np.inf) > 0.0
    assert _primary_q_residual_norm(problem, velocity_form) == 0.0
    assert np.linalg.norm(np.concatenate([vx_res, vy_res]), ord=np.inf) == 0.0


def test_ratio_free_primary_q_velocity_continuity_nitsche_supports_hdiv_fluid() -> None:
    problem = _build_ratio_free_full_problem(
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        fluid_space="hdiv",
        fluid_hdiv_order=1,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.1, phi_b=0.18)
    alpha_n = np.asarray(problem["alpha_n"].nodal_values, dtype=float)
    alpha_k = np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_n, 0.0, alpha_n)
    problem["B_k"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_k, 0.0, alpha_k)
    # The diffuse interface is horizontal in this micro-problem, so the RT0
    # H(div) probe must excite the vertical normal component. Use an RT0-
    # representable vertical profile instead of the CG-only horizontal probe.
    problem["v_k"].set_values_from_function(lambda x, y: np.array([0.0, y]))
    problem["v_n"].set_values_from_function(lambda x, y: np.array([0.0, y]))
    for key in ("vS_k", "vS_n", "u_k", "u_n"):
        problem[key].nodal_values[:] = 0.0
    problem["lambda_drag_k"].set_values_from_function(lambda x, y: np.array([0.0, 0.0]))
    problem["lambda_drag_n"].set_values_from_function(lambda x, y: np.array([0.0, 0.0]))

    _build_ratio_free_full_forms(
        problem,
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=True,
        interface_velocity_method="nitsche",
        interface_velocity_normal_strength=1.0,
        interface_velocity_tangential_strength=0.0,
        storativity_c0=0.0,
    )

    velocity_form = problem["_velocity_interface_residual_form"]
    assert velocity_form is not None
    pp_res = _assemble_block(problem, velocity_form, "p_pore")
    v_res = _assemble_block(problem, velocity_form, "v")
    # On the 1x1 RT micro-problem the interpolated H(div) state can collapse
    # the localized pore-row drive. The branch contract we need here is that
    # the form assembles on H(div) and still does not leak into the fluid or
    # Darcy-flux rows.
    assert np.all(np.isfinite(pp_res))
    assert _primary_q_residual_norm(problem, velocity_form) == 0.0
    assert np.linalg.norm(v_res, ord=np.inf) == 0.0


def test_ratio_free_primary_q_velocity_tangential_closure_rejects_hdiv_fluid() -> None:
    problem = _build_ratio_free_full_problem(
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        fluid_space="hdiv",
        fluid_hdiv_order=0,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.1, phi_b=0.18)
    alpha_n = np.asarray(problem["alpha_n"].nodal_values, dtype=float)
    alpha_k = np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_n, 0.0, alpha_n)
    problem["B_k"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_k, 0.0, alpha_k)

    with pytest.raises(NotImplementedError, match="normal trace only"):
        _build_ratio_free_full_forms(
            problem,
            drag_formulation="mixed_lm",
            split_primary_darcy_flux=True,
            pressure_interface_closure=False,
            interface_entry_closure=False,
            interface_bjs_closure=False,
            interface_velocity_continuity_closure=True,
            interface_velocity_method="nitsche",
            interface_velocity_normal_strength=1.0,
            interface_velocity_tangential_strength=1.0,
            storativity_c0=0.0,
        )


def test_interface_probe_diagnostics_support_hdiv_primary_q_branch() -> None:
    problem = _build_ratio_free_full_problem(
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        fluid_space="hdiv",
        fluid_hdiv_order=0,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.1, phi_b=0.18)
    alpha_n = np.asarray(problem["alpha_n"].nodal_values, dtype=float)
    alpha_k = np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_n, 0.0, alpha_n)
    problem["B_k"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_k, 0.0, alpha_k)

    diag = _compute_interface_probe_diagnostics(
        problem=problem,
        Lx=1.0,
        y_interface=1.0,
        y_profile=1.25,
        eps_alpha=0.1,
        mu_f=0.035,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        solid_visco_eta=0.0,
        solid_model="linear",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        interface_entry_delta=10.0,
        interface_bjs_gamma=1000.0,
    )

    summary = dict(diag.get("summary", {}) or {})
    assert summary.get("interface_probe_status") == "ok"
    assert "interface_band_mass_transfer_residual_n_meanabs" in summary
    assert "interface_band_traction_jump_support_n_meanabs" in summary


def test_interface_probe_diagnostics_reports_free_fluid_support_field_leakage_checks() -> None:
    problem = _build_ratio_free_full_problem(
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        fluid_space="cg",
        darcy_flux_space="cg",
    )
    for key in ("alpha_k", "alpha_n"):
        problem[key].nodal_values[:] = 0.1
    for key in ("B_k", "B_n"):
        problem[key].nodal_values[:] = 0.0
    problem["u_k"].set_values_from_function(lambda x, y: np.array([0.3, 0.4]))
    problem["u_n"].set_values_from_function(lambda x, y: np.array([0.3, 0.4]))
    problem["vS_k"].set_values_from_function(lambda x, y: np.array([0.0, 0.2]))
    problem["vS_n"].set_values_from_function(lambda x, y: np.array([0.0, 0.2]))
    problem["lambda_drag_k"].set_values_from_function(lambda x, y: np.array([1.0, 0.0]))
    problem["lambda_drag_n"].set_values_from_function(lambda x, y: np.array([1.0, 0.0]))
    problem["p_pore_k"].nodal_values[:] = 2.0
    problem["p_pore_n"].nodal_values[:] = 2.0

    diag = _compute_interface_probe_diagnostics(
        problem=problem,
        Lx=1.0,
        y_interface=1.0,
        y_profile=1.25,
        eps_alpha=0.1,
        mu_f=0.035,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        solid_visco_eta=0.0,
        solid_model="linear",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        interface_entry_delta=10.0,
        interface_bjs_gamma=1000.0,
    )

    summary = dict(diag.get("summary", {}) or {})
    assert summary.get("free_fluid_alpha_lt_0p25_point_count", 0.0) > 0.0
    assert float(summary["free_fluid_alpha_lt_0p25_u_mag_maxabs"]) == pytest.approx(0.5)
    assert float(summary["free_fluid_alpha_lt_0p25_vS_mag_maxabs"]) == pytest.approx(0.2)
    assert float(summary["free_fluid_alpha_lt_0p25_q_mag_maxabs"]) == pytest.approx(1.0)
    assert float(summary["free_fluid_alpha_lt_0p25_p_pore_maxabs"]) == pytest.approx(2.0)
    assert float(summary["free_fluid_alpha_lt_0p25_p_pore_support_maxabs"]) == pytest.approx(2.0)


def test_compute_profile_best_fit_scale_recovers_simple_amplitude_factor() -> None:
    x = np.linspace(0.0, 1.0, 5)
    y_num = np.array([0.0, 1.0, 2.0, 1.0, 0.0], dtype=float)
    y_ref = 2.5 * y_num

    fit = _compute_profile_best_fit_scale(x_num=x, y_num=y_num, x_ref=x, y_ref=y_ref)

    assert float(fit["scale_nonnegative"]) == pytest.approx(2.5)
    assert float(fit["scale_unconstrained"]) == pytest.approx(2.5)
    assert float(fit["rmse_scaled"]) == pytest.approx(0.0)
    assert float(fit["linf_scaled"]) == pytest.approx(0.0)


def test_ratio_free_primary_q_traction_normal_closure_loads_momentum_rows_only() -> None:
    problem = _build_ratio_free_full_problem(
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.1, phi_b=0.18)
    alpha_n = np.asarray(problem["alpha_n"].nodal_values, dtype=float)
    alpha_k = np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_n, 0.0, alpha_n)
    problem["B_k"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_k, 0.0, alpha_k)
    problem["v_k"].set_values_from_function(lambda x, y: np.array([y, 0.0]))
    problem["v_n"].set_values_from_function(lambda x, y: np.array([y, 0.0]))
    for key in ("vS_k", "vS_n", "u_k", "u_n"):
        problem[key].nodal_values[:] = 0.0
    problem["p_k"].nodal_values[:] = 1.25
    problem["p_n"].nodal_values[:] = 1.25
    problem["p_pore_k"].nodal_values[:] = 0.75
    problem["p_pore_n"].nodal_values[:] = 0.75
    problem["lambda_drag_k"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))
    problem["lambda_drag_n"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))

    _build_ratio_free_full_forms(
        problem,
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=False,
        interface_traction_continuity_closure=True,
        interface_traction_normal_strength=1.0,
        interface_traction_tangential_strength=0.0,
        storativity_c0=0.0,
    )

    traction_form = problem["_traction_interface_residual_form"]
    assert traction_form is not None
    v_x_res = _assemble_block(problem, traction_form, "v_x")
    v_y_res = _assemble_block(problem, traction_form, "v_y")
    vS_x_res = _assemble_block(problem, traction_form, "vS_x")
    vS_y_res = _assemble_block(problem, traction_form, "vS_y")
    assert np.linalg.norm(np.concatenate([v_x_res, v_y_res]), ord=np.inf) > 0.0
    assert np.linalg.norm(np.concatenate([vS_x_res, vS_y_res]), ord=np.inf) > 0.0
    assert _primary_q_residual_norm(problem, traction_form) == 0.0


def test_ratio_free_primary_q_traction_normal_closure_adds_only_mechanical_momentum_transfer() -> None:
    problem = _build_ratio_free_full_problem(
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.1, phi_b=0.18)
    alpha_n = np.asarray(problem["alpha_n"].nodal_values, dtype=float)
    alpha_k = np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_n, 0.0, alpha_n)
    problem["B_k"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_k, 0.0, alpha_k)
    for key in ("v_k", "v_n", "vS_k", "vS_n", "u_k", "u_n"):
        problem[key].nodal_values[:] = 0.0
    problem["p_k"].nodal_values[:] = 1.25
    problem["p_n"].nodal_values[:] = 1.25
    problem["p_pore_k"].nodal_values[:] = 0.75
    problem["p_pore_n"].nodal_values[:] = 0.75
    problem["lambda_drag_k"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))
    problem["lambda_drag_n"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))

    _build_ratio_free_full_forms(
        problem,
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=False,
        interface_traction_continuity_closure=True,
        interface_traction_normal_strength=1.0,
        interface_traction_tangential_strength=0.0,
        interface_traction_method="nitsche",
        storativity_c0=0.0,
    )

    traction_form = problem["_traction_interface_residual_form"]
    assert traction_form is not None
    _, residual = assemble_form(
        Equation(None, traction_form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    residual = np.asarray(residual, dtype=float)

    n_if = _interface_unit_normal_2d(problem["alpha_n"], eta=1.0e-4)
    weight_n = Constant(1.0 / 0.05) * Constant(4.0) * problem["alpha_n"] * (Constant(1.0) - problem["alpha_n"])
    rel_n_test = _dot_basis_2d(problem["v_test"], n_if) - _dot_basis_2d(problem["vS_test"], n_if)
    expected_form = weight_n * (-problem["p_k"]) * rel_n_test * dx(metadata={"q": 6})
    _, expected_residual = assemble_form(
        Equation(None, expected_form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend="python",
    )
    expected_residual = np.asarray(expected_residual, dtype=float)

    assert np.allclose(residual, expected_residual, atol=1.0e-10, rtol=5.0e-8)


def test_ratio_free_primary_q_traction_normal_closure_supports_hdiv_fluid() -> None:
    problem = _build_ratio_free_full_problem(
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        fluid_space="hdiv",
        fluid_hdiv_order=0,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.1, phi_b=0.18)
    alpha_n = np.asarray(problem["alpha_n"].nodal_values, dtype=float)
    alpha_k = np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_n, 0.0, alpha_n)
    problem["B_k"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_k, 0.0, alpha_k)
    problem["v_k"].set_values_from_function(lambda x, y: np.array([y, 0.0]))
    problem["v_n"].set_values_from_function(lambda x, y: np.array([y, 0.0]))
    for key in ("vS_k", "vS_n", "u_k", "u_n"):
        problem[key].nodal_values[:] = 0.0
    problem["p_k"].nodal_values[:] = 1.25
    problem["p_n"].nodal_values[:] = 1.25
    problem["p_pore_k"].nodal_values[:] = 0.75
    problem["p_pore_n"].nodal_values[:] = 0.75
    problem["lambda_drag_k"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))
    problem["lambda_drag_n"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))

    _build_ratio_free_full_forms(
        problem,
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=False,
        interface_traction_continuity_closure=True,
        interface_traction_normal_strength=1.0,
        interface_traction_tangential_strength=0.0,
        interface_traction_method="nitsche",
        storativity_c0=0.0,
    )

    traction_form = problem["_traction_interface_residual_form"]
    assert traction_form is not None
    v_res = _assemble_block(problem, traction_form, "v")
    vS_x_res = _assemble_block(problem, traction_form, "vS_x")
    vS_y_res = _assemble_block(problem, traction_form, "vS_y")
    assert np.linalg.norm(v_res, ord=np.inf) > 0.0
    assert np.linalg.norm(np.concatenate([vS_x_res, vS_y_res]), ord=np.inf) > 0.0
    assert _primary_q_residual_norm(problem, traction_form) == 0.0


def test_ratio_free_primary_q_traction_normal_closure_does_not_depend_on_p_pore_value() -> None:
    problem = _build_ratio_free_full_problem(
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        fluid_space="cg",
        darcy_flux_space="hdiv",
        fluid_hdiv_order=0,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.1, phi_b=0.18)
    alpha_n = np.asarray(problem["alpha_n"].nodal_values, dtype=float)
    alpha_k = np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_n, 0.0, alpha_n)
    problem["B_k"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_k, 0.0, alpha_k)
    problem["v_k"].set_values_from_function(lambda x, y: np.array([y, 0.0]))
    problem["v_n"].set_values_from_function(lambda x, y: np.array([y, 0.0]))
    for key in ("vS_k", "vS_n", "u_k", "u_n"):
        problem[key].nodal_values[:] = 0.0
    problem["p_k"].nodal_values[:] = 1.25
    problem["p_n"].nodal_values[:] = 1.25
    problem["lambda_drag_k"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))
    problem["lambda_drag_n"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))

    def _assemble_q_residual_for_pore_value(value: float) -> np.ndarray:
        problem["p_pore_k"].nodal_values[:] = float(value)
        problem["p_pore_n"].nodal_values[:] = float(value)
        _build_ratio_free_full_forms(
            problem,
            drag_formulation="mixed_lm",
            split_primary_darcy_flux=True,
            pressure_interface_closure=False,
            interface_entry_closure=False,
            interface_bjs_closure=False,
            interface_velocity_continuity_closure=False,
            interface_traction_continuity_closure=True,
            interface_traction_normal_strength=1.0,
            interface_traction_tangential_strength=0.0,
            interface_traction_method="nitsche",
            storativity_c0=0.0,
        )
        traction_form = problem["_traction_interface_residual_form"]
        return _assemble_block(problem, traction_form, "q")

    q_res_low = _assemble_q_residual_for_pore_value(0.25)
    q_res_high = _assemble_q_residual_for_pore_value(4.00)

    assert np.allclose(q_res_low, 0.0, atol=1.0e-12, rtol=1.0e-12)
    assert np.allclose(q_res_high, q_res_low, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_primary_q_traction_normal_hdiv_ignores_free_fluid_velocity_gradient() -> None:
    problem = _build_ratio_free_full_problem(
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        fluid_space="hdiv",
        fluid_hdiv_order=0,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.1, phi_b=0.18)
    alpha_n = np.asarray(problem["alpha_n"].nodal_values, dtype=float)
    alpha_k = np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_n, 0.0, alpha_n)
    problem["B_k"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_k, 0.0, alpha_k)
    for key in ("vS_k", "vS_n", "u_k", "u_n"):
        problem[key].nodal_values[:] = 0.0
    problem["p_k"].nodal_values[:] = 1.25
    problem["p_n"].nodal_values[:] = 1.25
    problem["p_pore_k"].nodal_values[:] = 0.75
    problem["p_pore_n"].nodal_values[:] = 0.75
    problem["lambda_drag_k"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))
    problem["lambda_drag_n"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))

    problem["v_k"].set_values_from_function(lambda x, y: np.array([0.0, 0.0]))
    problem["v_n"].set_values_from_function(lambda x, y: np.array([0.0, 0.0]))
    _build_ratio_free_full_forms(
        problem,
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=False,
        interface_traction_continuity_closure=True,
        interface_traction_normal_strength=1.0,
        interface_traction_tangential_strength=0.0,
        interface_traction_method="nitsche",
        storativity_c0=0.0,
    )
    traction_form_zero = problem["_traction_interface_residual_form"]
    assert traction_form_zero is not None
    q_zero = _assemble_block(problem, traction_form_zero, "q")

    problem["v_k"].set_values_from_function(lambda x, y: np.array([0.0, y]))
    problem["v_n"].set_values_from_function(lambda x, y: np.array([0.0, y]))
    _build_ratio_free_full_forms(
        problem,
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=False,
        interface_traction_continuity_closure=True,
        interface_traction_normal_strength=1.0,
        interface_traction_tangential_strength=0.0,
        interface_traction_method="nitsche",
        storativity_c0=0.0,
    )
    traction_form_sheared = problem["_traction_interface_residual_form"]
    assert traction_form_sheared is not None
    q_sheared = _assemble_block(problem, traction_form_sheared, "q")

    assert np.allclose(q_zero, q_sheared, rtol=0.0, atol=1.0e-10)


def test_ratio_free_primary_q_traction_tangential_closure_rejects_hdiv_fluid() -> None:
    problem = _build_ratio_free_full_problem(
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        fluid_space="hdiv",
        fluid_hdiv_order=0,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.1, phi_b=0.18)
    alpha_n = np.asarray(problem["alpha_n"].nodal_values, dtype=float)
    alpha_k = np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_n, 0.0, alpha_n)
    problem["B_k"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_k, 0.0, alpha_k)

    with pytest.raises(NotImplementedError, match="normal pressure-transfer law only"):
        _build_ratio_free_full_forms(
            problem,
            drag_formulation="mixed_lm",
            split_primary_darcy_flux=True,
            pressure_interface_closure=False,
            interface_entry_closure=False,
            interface_bjs_closure=False,
            interface_velocity_continuity_closure=False,
            interface_traction_continuity_closure=True,
            interface_traction_normal_strength=0.0,
            interface_traction_tangential_strength=1.0,
            interface_traction_method="nitsche",
            storativity_c0=0.0,
        )


def test_ratio_free_primary_q_traction_tangential_closure_loads_fluid_and_solid_rows_only() -> None:
    problem = _build_ratio_free_full_problem(
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.1, phi_b=0.18)
    alpha_n = np.asarray(problem["alpha_n"].nodal_values, dtype=float)
    alpha_k = np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_n, 0.0, alpha_n)
    problem["B_k"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_k, 0.0, alpha_k)
    problem["v_k"].set_values_from_function(lambda x, y: np.array([y, 0.0]))
    problem["v_n"].set_values_from_function(lambda x, y: np.array([y, 0.0]))
    for key in ("vS_k", "vS_n", "u_k", "u_n"):
        problem[key].nodal_values[:] = 0.0
    problem["p_k"].nodal_values[:] = 1.25
    problem["p_n"].nodal_values[:] = 1.25
    problem["p_pore_k"].nodal_values[:] = 0.75
    problem["p_pore_n"].nodal_values[:] = 0.75
    problem["lambda_drag_k"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))
    problem["lambda_drag_n"].set_values_from_function(lambda x, y: np.array([0.0, 1.0]))

    _build_ratio_free_full_forms(
        problem,
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=False,
        interface_traction_continuity_closure=True,
        interface_traction_normal_strength=0.0,
        interface_traction_tangential_strength=1.0,
        storativity_c0=0.0,
    )

    traction_form = problem["_traction_interface_residual_form"]
    assert traction_form is not None
    v_x_res = _assemble_block(problem, traction_form, "v_x")
    v_y_res = _assemble_block(problem, traction_form, "v_y")
    vS_x_res = _assemble_block(problem, traction_form, "vS_x")
    vS_y_res = _assemble_block(problem, traction_form, "vS_y")
    lam_x_res = _assemble_block(problem, traction_form, "lambda_drag_x")
    lam_y_res = _assemble_block(problem, traction_form, "lambda_drag_y")
    assert np.linalg.norm(np.concatenate([v_x_res, v_y_res]), ord=np.inf) > 0.0
    assert np.linalg.norm(np.concatenate([vS_x_res, vS_y_res]), ord=np.inf) > 0.0
    assert np.linalg.norm(np.concatenate([lam_x_res, lam_y_res]), ord=np.inf) == 0.0


def test_ratio_free_interface_pressure_entry_jacobian_fd_consistency() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=1,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
        pressure_mean_constraint=True,
        full_ratio_free_state=True,
        drag_formulation="mixed_lm",
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.1, phi_b=0.18)
    alpha_n = np.asarray(problem["alpha_n"].nodal_values, dtype=float)
    alpha_k = np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_n, 0.0, alpha_n)
    problem["B_k"].nodal_values[:] = np.clip(
        (1.0 - 0.18) * alpha_k + 0.005 * np.cos(np.pi * alpha_k),
        0.0,
        alpha_k,
    )
    problem["p_pore_n"].set_values_from_function(lambda x, y: 0.08 + 0.02 * x - 0.01 * y)
    problem["p_pore_k"].set_values_from_function(lambda x, y: 0.11 + 0.03 * x - 0.015 * y)
    problem["S_n"].nodal_values[:] = 1.0
    problem["S_k"].nodal_values[:] = 1.0
    if problem.get("p_mean_n") is not None:
        problem["p_mean_n"].nodal_values[:] = 0.0
        problem["p_mean_k"].nodal_values[:] = 0.0

    forms = _build_forms(
        problem,
        qdeg=6,
        dt_c=Constant(0.1),
        theta=1.0,
        rho_f=1.0,
        mu_f=0.035,
        mu_b=0.035,
        mu_b_model="mu",
        kappa_inv=1.0e3,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        phi_b=0.18,
        M_alpha=0.0,
        gamma_alpha=1.0,
        eps_alpha=0.1,
        solid_visco_eta=0.0,
        gamma_div=0.0,
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
        alpha_supg=0.0,
        alpha_cip=0.0,
        v_supg=0.0,
        v_supg_mode="streamline",
        v_supg_c_nu=4.0,
        u_supg=0.0,
        v_cip=0.0,
        alpha_regularization="none",
        alpha_reg_gamma=1.0,
        alpha_reg_eps_normal=0.1,
        alpha_reg_eps_tangent=0.025,
        alpha_reg_eta=1.0e-12,
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        solid_model="neo_hookean",
        kappa_inv_model="refmap",
        enable_phi_evolution=True,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        skeleton_inertia_convection="lagged",
        storativity_c0=1.0e-3,
        fluid_convection="full",
        support_physics="stored_support",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        drag_formulation="mixed_lm",
        full_ratio_free_state=True,
        pressure_interface_closure=True,
        pressure_interface_closure_strength=1.0,
        p_pore_fluid_gauge=True,
        p_pore_fluid_gauge_strength=1.0,
        interface_entry_closure=True,
        interface_entry_closure_strength=1.0,
        interface_entry_delta=10.0,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=False,
        interface_traction_continuity_closure=False,
    )

    eq = Equation(forms.jacobian_form, forms.residual_form)
    K, R0 = assemble_form(eq, dof_handler=problem["dh"], bcs=[], quad_order=6, backend="python")
    R0 = np.asarray(R0, dtype=float)

    def _assemble_residual() -> np.ndarray:
        _, residual = assemble_form(
            Equation(None, forms.residual_form),
            dof_handler=problem["dh"],
            bcs=[],
            quad_order=6,
            backend="python",
        )
        return np.asarray(residual, dtype=float)

    field_to_func = {
        "v_x": _vector_component_expr(problem["v_k"], 0),
        "v_y": _vector_component_expr(problem["v_k"], 1),
        "p": problem["p_k"],
        "p_pore": problem["p_pore_k"],
        "p_mean": problem.get("p_mean_k"),
        "vS_x": _vector_component_expr(problem["vS_k"], 0),
        "vS_y": _vector_component_expr(problem["vS_k"], 1),
        "u_x": _vector_component_expr(problem["u_k"], 0),
        "u_y": _vector_component_expr(problem["u_k"], 1),
        "alpha": problem["alpha_k"],
        "B": problem["B_k"],
        "mu_alpha": problem["mu_k"],
        "S": problem["S_k"],
        "lambda_drag_x": _vector_component_expr(problem.get("lambda_drag_k"), 0),
        "lambda_drag_y": _vector_component_expr(problem.get("lambda_drag_k"), 1),
        "q": problem.get("lambda_drag_k"),
    }

    probes = []
    for fld in ("v_x", "p", "p_pore", "p_mean", "alpha", "B", "lambda_drag_x", "q"):
        try:
            sl = problem["dh"].get_field_slice(fld)
        except KeyError:
            continue
        if sl:
            probes.append(int(sl[len(sl) // 2]))

    eps = 1.0e-6
    for j in probes:
        fld, _ = problem["dh"]._dof_to_node_map[j]
        func = field_to_func[fld]
        assert func is not None
        old = float(func.get_nodal_values(np.asarray([j], dtype=int))[0])
        func.set_nodal_values(np.asarray([j], dtype=int), np.asarray([old + eps], dtype=float))
        R_plus = _assemble_residual()
        func.set_nodal_values(np.asarray([j], dtype=int), np.asarray([old - eps], dtype=float))
        R_minus = _assemble_residual()
        func.set_nodal_values(np.asarray([j], dtype=int), np.asarray([old], dtype=float))

        fd = (R_plus - R_minus) / (2.0 * eps)
        col = np.asarray(K.getcol(j).toarray()).reshape(-1)
        denom = max(1.0, float(np.linalg.norm(fd, ord=np.inf)), float(np.linalg.norm(col, ord=np.inf)))
        rel = float(np.linalg.norm(fd - col, ord=np.inf)) / denom
        assert rel < 5.0e-6, f"FD mismatch at dof {j} ({fld}): rel={rel:.3e}"


def test_ratio_free_interface_pressure_entry_velocity_jacobian_fd_consistency() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=1,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
        pressure_mean_constraint=True,
        full_ratio_free_state=True,
        drag_formulation="mixed_lm",
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.1, phi_b=0.18)
    alpha_n = np.asarray(problem["alpha_n"].nodal_values, dtype=float)
    alpha_k = np.asarray(problem["alpha_k"].nodal_values, dtype=float)
    problem["B_n"].nodal_values[:] = np.clip((1.0 - 0.18) * alpha_n, 0.0, alpha_n)
    problem["B_k"].nodal_values[:] = np.clip(
        (1.0 - 0.18) * alpha_k + 0.005 * np.cos(np.pi * alpha_k),
        0.0,
        alpha_k,
    )
    problem["p_pore_n"].set_values_from_function(lambda x, y: 0.08 + 0.02 * x - 0.01 * y)
    problem["p_pore_k"].set_values_from_function(lambda x, y: 0.11 + 0.03 * x - 0.015 * y)
    problem["S_n"].nodal_values[:] = 1.0
    problem["S_k"].nodal_values[:] = 1.0
    if problem.get("p_mean_n") is not None:
        problem["p_mean_n"].nodal_values[:] = 0.0
        problem["p_mean_k"].nodal_values[:] = 0.0

    forms = _build_forms(
        problem,
        qdeg=6,
        dt_c=Constant(0.1),
        theta=1.0,
        rho_f=1.0,
        mu_f=0.035,
        mu_b=0.035,
        mu_b_model="mu",
        kappa_inv=1.0e3,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        phi_b=0.18,
        M_alpha=0.0,
        gamma_alpha=1.0,
        eps_alpha=0.1,
        solid_visco_eta=0.0,
        gamma_div=0.0,
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
        alpha_supg=0.0,
        alpha_cip=0.0,
        v_supg=0.0,
        v_supg_mode="streamline",
        v_supg_c_nu=4.0,
        u_supg=0.0,
        v_cip=0.0,
        alpha_regularization="none",
        alpha_reg_gamma=1.0,
        alpha_reg_eps_normal=0.1,
        alpha_reg_eps_tangent=0.025,
        alpha_reg_eta=1.0e-12,
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        solid_model="neo_hookean",
        kappa_inv_model="refmap",
        enable_phi_evolution=True,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        skeleton_inertia_convection="lagged",
        storativity_c0=1.0e-3,
        fluid_convection="full",
        support_physics="stored_support",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        drag_formulation="mixed_lm",
        full_ratio_free_state=True,
        pressure_interface_closure=True,
        pressure_interface_closure_strength=1.0,
        p_pore_fluid_gauge=True,
        p_pore_fluid_gauge_strength=1.0,
        interface_entry_closure=True,
        interface_entry_closure_strength=1.0,
        interface_entry_delta=10.0,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=True,
        interface_velocity_normal_strength=1.0,
        interface_velocity_tangential_strength=1.0,
        interface_traction_continuity_closure=False,
    )

    eq = Equation(forms.jacobian_form, forms.residual_form)
    K, _ = assemble_form(eq, dof_handler=problem["dh"], bcs=[], quad_order=6, backend="python")

    def _assemble_residual() -> np.ndarray:
        _, residual = assemble_form(
            Equation(None, forms.residual_form),
            dof_handler=problem["dh"],
            bcs=[],
            quad_order=6,
            backend="python",
        )
        return np.asarray(residual, dtype=float)

    field_to_func = {
        "v_x": _vector_component_expr(problem["v_k"], 0),
        "v_y": _vector_component_expr(problem["v_k"], 1),
        "p": problem["p_k"],
        "p_pore": problem["p_pore_k"],
        "p_mean": problem.get("p_mean_k"),
        "vS_x": _vector_component_expr(problem["vS_k"], 0),
        "vS_y": _vector_component_expr(problem["vS_k"], 1),
        "u_x": _vector_component_expr(problem["u_k"], 0),
        "u_y": _vector_component_expr(problem["u_k"], 1),
        "alpha": problem["alpha_k"],
        "B": problem["B_k"],
        "mu_alpha": problem["mu_k"],
        "S": problem["S_k"],
        "lambda_drag_x": _vector_component_expr(problem.get("lambda_drag_k"), 0),
        "lambda_drag_y": _vector_component_expr(problem.get("lambda_drag_k"), 1),
        "q": problem.get("lambda_drag_k"),
    }

    probes = []
    for fld in ("v_x", "vS_x", "p", "p_pore", "p_mean", "alpha", "B", "lambda_drag_x", "q"):
        try:
            sl = problem["dh"].get_field_slice(fld)
        except KeyError:
            continue
        if sl:
            probes.append(int(sl[len(sl) // 2]))

    eps = 1.0e-6
    for j in probes:
        fld, _ = problem["dh"]._dof_to_node_map[j]
        func = field_to_func[fld]
        assert func is not None
        old = float(func.get_nodal_values(np.asarray([j], dtype=int))[0])
        func.set_nodal_values(np.asarray([j], dtype=int), np.asarray([old + eps], dtype=float))
        R_plus = _assemble_residual()
        func.set_nodal_values(np.asarray([j], dtype=int), np.asarray([old - eps], dtype=float))
        R_minus = _assemble_residual()
        func.set_nodal_values(np.asarray([j], dtype=int), np.asarray([old], dtype=float))

        fd = (R_plus - R_minus) / (2.0 * eps)
        col = np.asarray(K.getcol(j).toarray()).reshape(-1)
        denom = max(1.0, float(np.linalg.norm(fd, ord=np.inf)), float(np.linalg.norm(col, ord=np.inf)))
        rel = float(np.linalg.norm(fd - col, ord=np.inf)) / denom
        assert rel < 5.0e-6, f"FD mismatch at dof {j} ({fld}): rel={rel:.3e}"


def test_startup_fluid_stage_uses_pdas_when_transport_bounds_are_active() -> None:
    assert _startup_stage_solver_kind(main_solver_kind="pdas", active_fields=["v_x", "v_y", "p", "alpha", "phi"]) == "pdas"
    assert _startup_stage_solver_kind(main_solver_kind="pdas", active_fields=["vS_x", "vS_y", "u_x", "u_y"]) == "newton"
    assert _startup_stage_solver_kind(main_solver_kind="newton", active_fields=["v_x", "v_y", "p", "alpha"]) == "newton"


def test_startup_transport_stage_uses_ipm_when_requested() -> None:
    assert _startup_stage_solver_kind(main_solver_kind="ipm", active_fields=["v_x", "v_y", "alpha"]) == "ipm"
    assert _startup_stage_solver_kind(main_solver_kind="ipm", active_fields=["vS_x", "vS_y", "u_x", "u_y"]) == "newton"


def test_startup_transport_stage_override_can_force_pdas_for_ipm_run() -> None:
    assert (
        _startup_stage_solver_kind(
            main_solver_kind="ipm",
            active_fields=["alpha", "phi"],
            stage_name="transport",
            transport_solver_kind_override="pdas",
        )
        == "pdas"
    )
    assert (
        _startup_stage_solver_kind(
            main_solver_kind="pdas",
            active_fields=["vS_x", "vS_y", "u_x", "u_y"],
            stage_name="transport",
            transport_solver_kind_override="newton",
        )
        == "newton"
    )


def test_startup_monolithic_budget_defaults_to_boosted_first_step_budget() -> None:
    assert _startup_monolithic_max_it(SimpleNamespace(max_it=12, startup_monolithic_max_it=None)) == 24
    assert _startup_monolithic_max_it(SimpleNamespace(max_it=24, startup_monolithic_max_it=None)) == 48
    assert _startup_monolithic_max_it(SimpleNamespace(max_it=24, startup_monolithic_max_it=30)) == 30


def test_startup_stage_relaxed_accept_ginf_defaults_to_scaled_newton_tol() -> None:
    assert _startup_stage_relaxed_accept_ginf(SimpleNamespace(startup_stage_relaxed_ginf=None, newton_tol=1.0e-8)) == 1.0e-5
    assert _startup_stage_relaxed_accept_ginf(SimpleNamespace(startup_stage_relaxed_ginf=None, newton_tol=1.0e-10)) == 1.0e-6
    assert _startup_stage_relaxed_accept_ginf(SimpleNamespace(startup_stage_relaxed_ginf=2.5e-7, newton_tol=1.0e-8)) == 2.5e-7


def test_startup_first_step_relaxed_accept_ginf_uses_restart_floor() -> None:
    args = SimpleNamespace()

    assert _startup_first_step_relaxed_accept_ginf(args, base_relaxed_accept_ginf=0.0) == 2.0e-2
    assert _startup_first_step_relaxed_accept_ginf(args, base_relaxed_accept_ginf=3.0e-2) == 3.0e-2


def test_startup_first_step_relaxed_accept_ginf_boosts_one_domain_seboldt_entry_branch() -> None:
    args = SimpleNamespace(
        full_ratio_free_state=True,
        support_physics="stored_support",
        skeleton_pressure_mode="seboldt",
        interface_entry_closure=True,
        interface_bjs_closure=False,
    )

    assert _startup_first_step_relaxed_accept_ginf(args, base_relaxed_accept_ginf=0.0) == 4.0e-2
    assert _startup_first_step_relaxed_accept_ginf(args, base_relaxed_accept_ginf=3.0e-2) == 4.0e-2


def test_benchmark7_cli_defaults_use_relaxed_newton_target_and_pc_startup(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["paper1_benchmark7_seboldt.py"])
    args = _parse_args()

    assert float(args.newton_tol) == 1.0e-6
    assert float(args.newton_rtol) == 1.0e-6
    assert str(args.mu_b_model) == "phi_mu"
    assert bool(args.enable_phi_evolution)
    assert bool(args.alpha_mass_constraint)
    assert not bool(args.alpha_from_refmap)
    assert not bool(args.condition_balanced_solid_cut_fix)
    assert float(args.gamma_u) == 1.0
    assert str(args.u_extension) == "h1"
    assert str(args.reduced_support_state) == "alpha_B"
    assert str(args.latent_bounded_fields) == "alpha,phi"
    assert bool(args.predictor_corrector_startup)
    assert int(args.pc_p1_max_it) == 12
    assert int(args.pc_p2_max_it) == 12
    assert int(args.pc_exact_probe_max_it) == 1


def test_condition_balanced_scales_use_lambda_reference_for_seboldt_wall() -> None:
    scales = _condition_balanced_field_scales(
        mechanics_nondim_mode="condition_balanced",
        solid_model="neo_hookean",
        drag_formulation="direct",
        dt=0.1,
        mu_f=0.035,
        kappa_inv=1.0e3,
        mu_s=3.0,
        lambda_s=11.0,
        rho_s0_tilde=1.0,
        dim=2,
    )
    assert scales["vS_x"] == pytest.approx(0.1 * np.sqrt(11.0))
    assert scales["vS_y"] == pytest.approx(0.1 * np.sqrt(11.0))


def test_benchmark7_reduced_problem_uses_alpha_B_support_state_by_default() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=2,
        ny=3,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=False,
    )

    assert str(problem["reduced_support_state"]) == "alpha_b"
    assert "B" in problem["dh"].field_names
    assert problem["B_k"] is not None
    assert problem["B_n"] is not None
    assert problem["dB"] is not None
    assert problem["B_test"] is not None


def test_benchmark7_cli_accepts_tanh_latent_map(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["paper1_benchmark7_seboldt.py", "--latent-bounded-map", "tanh"])
    args = _parse_args()

    assert str(args.latent_bounded_map) == "tanh"


def test_benchmark7_maps_default_neo_hookean_to_seboldt_eulerian_wall() -> None:
    assert _benchmark7_solid_model_key("neo_hookean") == "seboldt_neo_hookean"
    assert _benchmark7_solid_model_key("nh") == "seboldt_neo_hookean"
    assert _benchmark7_solid_model_key("neo_hookean_eulerian") == "neo_hookean_eulerian"


def test_benchmark7_seboldt_wall_uniform_dilation_matches_example2_cauchy_stress() -> None:
    s = 1.1
    grad_u = np.array(
        [
            [1.0 - (1.0 / s), 0.0],
            [0.0, 1.0 - (1.0 / s)],
        ],
        dtype=float,
    )
    sigma = _solid_cauchy_stress_from_grad_u(
        grad_u,
        mu_s=2.0,
        lambda_s=7.0,
        solid_model="neo_hookean",
    )
    expected = (((2.0 / (s * s)) + 7.0) * (s * s - 1.0)) * np.eye(2, dtype=float)
    assert np.allclose(sigma, expected)


def test_benchmark7_seboldt_wall_isochoric_stretch_matches_example2_cauchy_stress() -> None:
    s = 1.2
    grad_u = np.array(
        [
            [1.0 - (1.0 / s), 0.0],
            [0.0, 1.0 - s],
        ],
        dtype=float,
    )
    sigma = _solid_cauchy_stress_from_grad_u(
        grad_u,
        mu_s=3.0,
        lambda_s=11.0,
        solid_model="neo_hookean",
    )
    expected = np.diag([3.0 * (s * s - 1.0), 3.0 * ((1.0 / (s * s)) - 1.0)])
    assert np.allclose(sigma, expected)


def test_ratio_free_stored_support_pore_storage_uses_skeleton_material_derivative() -> None:
    problem = _build_ratio_free_full_problem(split_pore_flux_model="reduced_divq")
    problem["alpha_k"].nodal_values[:] = 0.7
    problem["alpha_n"].nodal_values[:] = 0.7
    problem["B_k"].nodal_values[:] = 0.49
    problem["B_n"].nodal_values[:] = 0.49
    problem["p_pore_k"].set_values_from_function(lambda x, y: 0.2 + 0.4 * x - 0.3 * y)
    problem["p_pore_n"].set_values_from_function(lambda x, y: 0.2 + 0.4 * x - 0.3 * y)
    problem["v_k"].set_values_from_function(lambda x, y: np.asarray([0.5, -0.25], dtype=float))
    problem["v_n"].set_values_from_function(lambda x, y: np.asarray([0.5, -0.25], dtype=float))
    problem["vS_k"].set_values_from_function(lambda x, y: np.asarray([0.125, 0.125], dtype=float))
    problem["vS_n"].set_values_from_function(lambda x, y: np.asarray([0.125, 0.125], dtype=float))

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        storativity_c0=2.0,
        split_pore_flux_model="reduced_divq",
        include_skeleton_acceleration=False,
    )

    pore_block = _assemble_block(problem, forms.r_pore, "p_pore")
    material_rate = float(np.dot(np.asarray([0.125, 0.125], dtype=float), np.asarray([0.4, -0.3], dtype=float)))
    expected_form = Constant(0.7 * 2.0 * material_rate) * problem["q_pore_test"] * dx(metadata={"q": 6})
    expected_block = _assemble_block(problem, expected_form, "p_pore")

    assert np.allclose(pore_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_single_pressure_stored_support_core_assembles_exact_mixture_constraint_row() -> None:
    problem = _build_single_pressure_stored_support_problem()
    problem["v_k"].set_values_from_function(lambda x, y: np.asarray([x, 0.0], dtype=float))
    problem["v_n"].set_values_from_function(lambda x, y: np.asarray([x, 0.0], dtype=float))
    problem["vS_k"].set_values_from_function(lambda x, y: np.asarray([0.0, y], dtype=float))
    problem["vS_n"].set_values_from_function(lambda x, y: np.asarray([0.0, y], dtype=float))

    forms = _build_full_forms(
        problem,
        enable_phi_evolution=False,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        skeleton_pressure_mode="whole_domain",
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
    )

    assert forms.r_pore is None
    mass_block = _assemble_block(problem, forms.r_mass, "p")
    expected_form = problem["q_test"] * (Constant(0.18) * div(problem["v_k"]) + Constant(0.82) * div(problem["vS_k"])) * dx(metadata={"q": 6})
    expected_block = _assemble_block(problem, expected_form, "p")

    assert np.allclose(mass_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_single_pressure_stored_support_core_rejects_biot_storage() -> None:
    problem = _build_single_pressure_stored_support_problem()

    with pytest.raises(ValueError, match="single-pressure stored_support\\(alpha,B,p\\) core is incompressible"):
        _build_full_forms(
            problem,
            enable_phi_evolution=False,
            support_physics="stored_support",
            alpha_advect_with="vS",
            alpha_advection_form="advective",
            skeleton_pressure_mode="whole_domain",
            storativity_c0=1.0e-3,
            include_skeleton_acceleration=False,
        )


def test_ratio_free_stored_support_frozen_phi_b_mode_constrains_B_to_alpha() -> None:
    problem = _build_ratio_free_full_problem(stored_support_content_mode="frozen_phi_b")
    problem["alpha_k"].nodal_values[:] = 0.7
    problem["B_k"].nodal_values[:] = 0.5

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
        stored_support_content_mode="frozen_phi_b",
    )

    B_block = _assemble_block(problem, forms.r_B, "B")
    expected_form = Constant(0.5 - ((1.0 - 0.18) * 0.7)) * problem["B_test"] * dx(metadata={"q": 6})
    expected_block = _assemble_block(problem, expected_form, "B")

    assert np.allclose(B_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_ratio_free_stored_support_freeze_B_mode_disables_B_transport() -> None:
    problem = _build_ratio_free_full_problem(stored_support_content_mode="freeze_B")
    problem["B_n"].nodal_values[:] = 0.8
    problem["B_k"].nodal_values[:] = 0.9
    problem["vS_k"].set_values_from_function(lambda x, y: np.asarray([1.0, 0.0], dtype=float))
    problem["vS_n"].set_values_from_function(lambda x, y: np.asarray([1.0, 0.0], dtype=float))

    forms = _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
        stored_support_content_mode="freeze_B",
    )

    B_block = _assemble_block(problem, forms.r_B, "B")
    expected_form = Constant((0.9 - 0.8) / 0.1) * problem["B_test"] * dx(metadata={"q": 6})
    expected_block = _assemble_block(problem, expected_form, "B")

    assert np.allclose(B_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_benchmark7_cpp_fuse_integrals_defaults_on_for_cpp(monkeypatch) -> None:
    monkeypatch.delenv("PYCUTFEM_CPP_FUSE_INTEGRALS", raising=False)
    monkeypatch.setattr(sys, "argv", ["paper1_benchmark7_seboldt.py"])
    args = _parse_args()

    _configure_benchmark7_cpp_fuse_integrals(
        backend=str(args.backend),
        enabled=getattr(args, "cpp_fuse_integrals", None),
    )

    assert os.environ["PYCUTFEM_CPP_FUSE_INTEGRALS"] == "1"


def test_benchmark7_cli_can_disable_cpp_integral_fusion(monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CPP_FUSE_INTEGRALS", "1")
    monkeypatch.setattr(sys, "argv", ["paper1_benchmark7_seboldt.py", "--no-cpp-fuse-integrals"])
    args = _parse_args()

    _configure_benchmark7_cpp_fuse_integrals(
        backend=str(args.backend),
        enabled=getattr(args, "cpp_fuse_integrals", None),
    )

    assert os.environ["PYCUTFEM_CPP_FUSE_INTEGRALS"] == "0"


def test_benchmark7_same_measure_residual_fusion_preserves_response() -> None:
    problem, forms = _build_small_benchmark7_residual_problem()
    fused_form = _fuse_form_like_compile_multi(forms.residual_form, dof_handler=problem["dh"])

    assert len(forms.residual_form.integrals) == 55
    assert len(fused_form.integrals) == 1

    _, residual_plain = assemble_form(
        Equation(None, forms.residual_form),
        dof_handler=problem["dh"],
        bcs=[],
        backend="python",
    )
    _, residual_fused = assemble_form(
        Equation(None, fused_form),
        dof_handler=problem["dh"],
        bcs=[],
        backend="python",
    )

    np.testing.assert_allclose(
        np.asarray(residual_fused, dtype=float),
        np.asarray(residual_plain, dtype=float),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_direct_phi_predictor_corrector_startup_is_not_latent_gated() -> None:
    args = SimpleNamespace(
        predictor_corrector_startup=True,
        enable_phi_evolution=True,
    )
    problem = {"phi_k": object()}

    assert _predictor_corrector_startup_enabled(args, problem)


def test_effective_eps_alpha_respects_eps_over_h_even_with_pc_startup() -> None:
    args = SimpleNamespace(
        predictor_corrector_startup=True,
        eps_alpha=0.05,
        eps_alpha_over_h=0.4,
        Lx=1.0,
        Ly=1.5,
        nx=16,
        ny=24,
    )

    assert _effective_eps_alpha(args) == pytest.approx(0.025)


def test_effective_eps_alpha_falls_back_to_physical_width_without_ratio() -> None:
    args = SimpleNamespace(
        predictor_corrector_startup=True,
        eps_alpha=0.05,
        eps_alpha_over_h=None,
        Lx=1.0,
        Ly=1.5,
        nx=16,
        ny=24,
    )

    assert _effective_eps_alpha(args) == pytest.approx(0.05)


def test_benchmark7_p2_lambda_schedule_starts_with_anchor_zero() -> None:
    assert _pc_p2_lambda_schedule(4, include_zero=True) == pytest.approx([0.0, 1.0 / 16.0, 4.0 / 16.0, 9.0 / 16.0, 1.0])
    assert _pc_p2_lambda_schedule(1, include_zero=True) == pytest.approx([0.0, 1.0])


def test_benchmark7_p2_easy_dt_uses_microstep_divisor() -> None:
    args = SimpleNamespace(pc_p2_easy_dt_divisor=100.0)
    assert _pc_p2_easy_dt_value(args, 2.5e-2) == pytest.approx(2.5e-4)


def test_benchmark7_p2_mode_named_constants_keep_kernel_hashes_stable() -> None:
    problem = _build_full_problem()
    lam = _named_constant("test_pc_p2_lambda_hash", 0.0)
    one_minus_lam = _named_constant("test_pc_p2_one_hash", 1.0) - lam
    easy_fluid_full = _named_constant("test_pc_p2_easy_fluid_full", 0.0)
    easy_fluid_lagged = _named_constant("test_pc_p2_easy_fluid_lagged", 0.0)
    easy_fluid_imex = _named_constant("test_pc_p2_easy_fluid_imex", 0.0)
    exact_fluid_full = _named_constant("test_pc_p2_exact_fluid_full", 0.0)
    exact_fluid_lagged = _named_constant("test_pc_p2_exact_fluid_lagged", 0.0)
    exact_fluid_imex = _named_constant("test_pc_p2_exact_fluid_imex", 0.0)
    easy_skel_accel = _named_constant("test_pc_p2_easy_skel_accel", 0.0)
    exact_skel_accel = _named_constant("test_pc_p2_exact_skel_accel", 1.0)
    easy_skel_full = _named_constant("test_pc_p2_easy_skel_full", 0.0)
    easy_skel_lagged = _named_constant("test_pc_p2_easy_skel_lagged", 0.0)
    exact_skel_full = _named_constant("test_pc_p2_exact_skel_full", 0.0)
    exact_skel_lagged = _named_constant("test_pc_p2_exact_skel_lagged", 1.0)
    forms = _build_full_forms(
        problem,
        fluid_convection="lagged",
        fluid_convection_full_weight=one_minus_lam * easy_fluid_full + lam * exact_fluid_full,
        fluid_convection_lagged_weight=one_minus_lam * easy_fluid_lagged + lam * exact_fluid_lagged,
        fluid_convection_imex_weight=one_minus_lam * easy_fluid_imex + lam * exact_fluid_imex,
        include_skeleton_acceleration=True,
        skeleton_acceleration_weight=one_minus_lam * easy_skel_accel + lam * exact_skel_accel,
        skeleton_inertia_convection="lagged",
        skeleton_inertia_full_weight=one_minus_lam * easy_skel_full + lam * exact_skel_full,
        skeleton_inertia_lagged_weight=one_minus_lam * easy_skel_lagged + lam * exact_skel_lagged,
    )

    easy_fluid = _pc_fluid_convection_selectors("off")
    exact_fluid = _pc_fluid_convection_selectors("full")
    easy_skel = _pc_skeleton_inertia_selectors("lagged")
    exact_skel = _pc_skeleton_inertia_selectors("full")
    easy_fluid_full.value = easy_fluid["full"]
    easy_fluid_lagged.value = easy_fluid["lagged"]
    easy_fluid_imex.value = easy_fluid["imex"]
    exact_fluid_full.value = exact_fluid["full"]
    exact_fluid_lagged.value = exact_fluid["lagged"]
    exact_fluid_imex.value = exact_fluid["imex"]
    easy_skel_accel.value = 0.0
    exact_skel_accel.value = 1.0
    easy_skel_full.value = easy_skel["full"]
    easy_skel_lagged.value = easy_skel["lagged"]
    exact_skel_full.value = exact_skel["full"]
    exact_skel_lagged.value = exact_skel["lagged"]
    lam.value = 0.0
    jac0 = _integral_hashes(forms.jacobian_form, problem["dh"], rank=2)
    res0 = _integral_hashes(forms.residual_form, problem["dh"], rank=1)

    easy_fluid = _pc_fluid_convection_selectors("lagged")
    exact_fluid = _pc_fluid_convection_selectors("imex")
    easy_skel = _pc_skeleton_inertia_selectors("full")
    exact_skel = _pc_skeleton_inertia_selectors("lagged")
    easy_fluid_full.value = easy_fluid["full"]
    easy_fluid_lagged.value = easy_fluid["lagged"]
    easy_fluid_imex.value = easy_fluid["imex"]
    exact_fluid_full.value = exact_fluid["full"]
    exact_fluid_lagged.value = exact_fluid["lagged"]
    exact_fluid_imex.value = exact_fluid["imex"]
    easy_skel_accel.value = 1.0
    exact_skel_accel.value = 0.0
    easy_skel_full.value = easy_skel["full"]
    easy_skel_lagged.value = easy_skel["lagged"]
    exact_skel_full.value = exact_skel["full"]
    exact_skel_lagged.value = exact_skel["lagged"]
    lam.value = 0.75

    assert _integral_hashes(forms.jacobian_form, problem["dh"], rank=2) == jac0
    assert _integral_hashes(forms.residual_form, problem["dh"], rank=1) == res0


def test_benchmark7_p2_accepts_homotopy_progress_before_exact_progress() -> None:
    before = {
        "raw_inf": 1.0,
        "homotopy_raw_inf": 1.0,
        "alpha_mass_rel": 0.0,
    }
    after = {
        "raw_inf": 1.04,
        "homotopy_raw_inf": 1.0e-8,
        "alpha_mass_rel": 0.0,
    }

    keep, info = _pc_should_keep_lambda_stage(
        lam=0.25,
        before_stats=before,
        after_stats=after,
        alpha_mass_ok=True,
        min_abs_decrease=1.0e-10,
        min_rel_improve=1.0e-2,
        homotopy_tol=1.0e-6,
    )

    assert keep
    assert bool(info["homotopy"]["improved"])
    assert not bool(info["exact"]["improved"])


def test_benchmark7_p2_rejects_homotopy_progress_if_exact_residual_degrades_too_far() -> None:
    before = {
        "raw_inf": 1.0,
        "homotopy_raw_inf": 1.0,
        "alpha_mass_rel": 0.0,
    }
    after = {
        "raw_inf": 1.4,
        "homotopy_raw_inf": 1.0e-8,
        "alpha_mass_rel": 0.0,
    }

    keep, info = _pc_should_keep_lambda_stage(
        lam=0.25,
        before_stats=before,
        after_stats=after,
        exact_reference_stats=before,
        alpha_mass_ok=True,
        min_abs_decrease=1.0e-10,
        min_rel_improve=1.0e-2,
        max_exact_worsen_rel=5.0e-2,
        homotopy_tol=1.0e-6,
    )

    assert not keep
    assert bool(info["homotopy"]["improved"])
    assert not bool(info["exact_within_guard"])


def test_benchmark7_exact_probe_skips_p2_when_exact_branch_already_improves() -> None:
    before = {
        "raw_inf": 1.0,
        "homotopy_raw_inf": 1.0,
        "alpha_mass_rel": 0.0,
    }
    after = {
        "raw_inf": 0.4,
        "homotopy_raw_inf": 1.0,
        "alpha_mass_rel": 0.0,
    }

    keep, info = _pc_should_prefer_exact_probe(
        before_stats=before,
        after_stats=after,
        alpha_mass_ok=True,
        min_abs_decrease=1.0e-10,
        min_rel_improve=1.0e-2,
    )

    assert keep
    assert bool(info["exact"]["improved"])


def test_benchmark7_exact_probe_does_not_skip_p2_without_exact_progress() -> None:
    before = {
        "raw_inf": 1.0,
        "homotopy_raw_inf": 1.0,
        "alpha_mass_rel": 0.0,
    }
    after = {
        "raw_inf": 1.02,
        "homotopy_raw_inf": 1.0e-8,
        "alpha_mass_rel": 0.0,
    }

    keep, info = _pc_should_prefer_exact_probe(
        before_stats=before,
        after_stats=after,
        alpha_mass_ok=True,
        min_abs_decrease=1.0e-10,
        min_rel_improve=1.0e-2,
    )

    assert not keep
    assert not bool(info["exact"]["improved"])


def test_benchmark7_path_tangent_euler_step_solves_linearized_homotopy() -> None:
    jac = np.array([[2.0, 0.0], [0.0, 4.0]], dtype=float)
    dH = np.array([2.0, 4.0], dtype=float)
    x0 = np.array([3.0, 5.0], dtype=float)

    x1, z_dot = _pc_path_tangent_euler_step(
        x_red=x0,
        jacobian_red=jac,
        dH_dlambda_red=dH,
        delta_lambda=0.25,
        solve_linear_system=lambda A, rhs: np.linalg.solve(np.asarray(A, dtype=float), np.asarray(rhs, dtype=float)),
    )

    assert z_dot == pytest.approx([-1.0, -1.0])
    assert x1 == pytest.approx([2.75, 4.75])


def test_large_previous_step_uses_staggered_predictor_on_next_step() -> None:
    assert _should_use_staggered_predictor_after_large_step(
        step_no=2,
        last_step_no=1,
        last_step_delta_inf=8.34e2,
        threshold=100.0,
    )
    assert not _should_use_staggered_predictor_after_large_step(
        step_no=1,
        last_step_no=None,
        last_step_delta_inf=8.34e2,
        threshold=100.0,
    )
    assert not _should_use_staggered_predictor_after_large_step(
        step_no=2,
        last_step_no=1,
        last_step_delta_inf=1.0,
        threshold=100.0,
    )


def test_frozen_transport_restart_arms_only_for_stalled_vi_with_stable_transport_set() -> None:
    assert _should_use_frozen_transport_restart(
        enable_phi_evolution=True,
        step_no=2,
        startup_guess_applied_step_no=None,
        metrics={
            "G_inf": 2.5e-1,
            "active_gap_inf": 2.0e-9,
            "equality_inf": 8.0e-12,
            "delta_active": 0.0,
        },
        max_delta_active=2,
        max_gap=1.0e-6,
        max_eq=1.0e-8,
        min_ginf=1.0e-2,
    )

    assert not _should_use_frozen_transport_restart(
        enable_phi_evolution=True,
        step_no=1,
        startup_guess_applied_step_no=None,
        metrics={
            "G_inf": 2.5e-1,
            "active_gap_inf": 2.0e-9,
            "equality_inf": 8.0e-12,
            "delta_active": 0.0,
        },
        max_delta_active=2,
        max_gap=1.0e-6,
        max_eq=1.0e-8,
        min_ginf=1.0e-2,
    )

    assert not _should_use_frozen_transport_restart(
        enable_phi_evolution=True,
        step_no=2,
        startup_guess_applied_step_no=None,
        metrics={
            "G_inf": 2.5e-1,
            "active_gap_inf": 2.0e-3,
            "equality_inf": 8.0e-12,
            "delta_active": 0.0,
        },
        max_delta_active=2,
        max_gap=1.0e-6,
        max_eq=1.0e-8,
        min_ginf=1.0e-2,
    )

    assert not _should_use_frozen_transport_restart(
        enable_phi_evolution=False,
        step_no=2,
        startup_guess_applied_step_no=None,
        metrics={
            "G_inf": 2.5e-1,
            "active_gap_inf": 2.0e-9,
            "equality_inf": 8.0e-12,
            "delta_active": 0.0,
        },
        max_delta_active=2,
        max_gap=1.0e-6,
        max_eq=1.0e-8,
        min_ginf=1.0e-2,
    )


def test_support_aware_phi_bounds_follow_alpha_coordinates_not_global_phi_order() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=2,
        ny=3,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
    )
    alpha_xy = np.asarray(problem["dh"].get_dof_coords("alpha"), dtype=float)
    phi_xy = np.asarray(problem["dh"].get_dof_coords("phi"), dtype=float)
    problem["alpha_k"].nodal_values[:] = alpha_xy[:, 1]

    lower_full, upper_full, support_mask = _build_support_aware_phi_box_bounds(
        problem,
        alpha_func=problem["alpha_k"],
        alpha_threshold=0.75,
    )

    phi_dofs = np.asarray(problem["dh"].get_field_slice("phi"), dtype=int)
    expected_mask = np.asarray(phi_xy[:, 1] > 0.75, dtype=bool)

    assert np.array_equal(support_mask, expected_mask)
    assert np.all(lower_full[phi_dofs[support_mask]] == 0.0)
    assert np.all(upper_full[phi_dofs[support_mask]] == 1.0)
    assert np.all(np.isneginf(lower_full[phi_dofs[~support_mask]]))
    assert np.all(np.isposinf(upper_full[phi_dofs[~support_mask]]))


def test_open_top_phi_cleanup_clips_global_phi_even_outside_support() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=2,
        ny=3,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
    )
    problem["alpha_k"].nodal_values[:] = 1.0e-3
    problem["phi_k"].nodal_values[:] = 1.02
    problem["phi_k"].nodal_values[0] = -0.05

    _apply_open_top_global_phi_cleanup(
        args=_make_args(enable_phi_evolution=True, top_drainage_transport=True),
        problem=problem,
        funcs=[problem["alpha_k"], problem["phi_k"]],
        find_named_function=_find_named_function,
    )

    phi_vals = np.asarray(problem["phi_k"].nodal_values, dtype=float)
    assert np.all(phi_vals >= -1.0e-12)
    assert np.all(phi_vals <= 1.0 + 1.0e-12)


def test_reduced_alpha_B_dependent_upper_bound_tracks_alpha() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=2,
        ny=3,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=False,
        reduced_support_state="alpha_B",
    )
    alpha_vals = np.linspace(0.15, 0.85, problem["alpha_k"].nodal_values.size)
    problem["alpha_k"].nodal_values[:] = alpha_vals
    lower_full = np.full(int(problem["dh"].total_dofs), -np.inf, dtype=float)
    upper_full = np.full(int(problem["dh"].total_dofs), np.inf, dtype=float)

    _apply_field_dependent_upper_bound(
        upper_full,
        problem=problem,
        source_field_name="alpha",
        target_field_name="B",
        source_values=problem["alpha_k"].nodal_values,
    )

    B_dofs = np.asarray(problem["dh"].get_field_slice("B"), dtype=int).ravel()
    assert np.allclose(upper_full[B_dofs], alpha_vals)
    alpha_dofs = np.asarray(problem["dh"].get_field_slice("alpha"), dtype=int).ravel()
    assert np.all(np.isposinf(upper_full[alpha_dofs]))
    assert np.all(np.isneginf(lower_full[B_dofs]))


def test_reduced_alpha_B_dependent_lower_bound_tracks_B() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=2,
        ny=3,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=False,
        reduced_support_state="alpha_B",
    )
    B_vals = np.linspace(0.15, 0.85, problem["B_k"].nodal_values.size)
    problem["B_k"].nodal_values[:] = B_vals
    lower_full = np.full(int(problem["dh"].total_dofs), -np.inf, dtype=float)
    upper_full = np.full(int(problem["dh"].total_dofs), np.inf, dtype=float)

    _apply_field_dependent_lower_bound(
        lower_full,
        problem=problem,
        source_field_name="B",
        target_field_name="alpha",
        source_values=problem["B_k"].nodal_values,
    )

    alpha_dofs = np.asarray(problem["dh"].get_field_slice("alpha"), dtype=int).ravel()
    assert np.allclose(lower_full[alpha_dofs], B_vals)
    B_dofs = np.asarray(problem["dh"].get_field_slice("B"), dtype=int).ravel()
    assert np.all(np.isneginf(lower_full[B_dofs]))
    assert np.all(np.isposinf(upper_full[alpha_dofs]))


def test_benchmark7_constrained_configuration_requires_pdas_solver() -> None:
    args = SimpleNamespace(
        nonlinear_solver="newton",
        alpha_bc_mode="natural",
        alpha_box_constraints=True,
        enable_phi_evolution=False,
        phi_box_constraints=False,
    )

    assert _benchmark7_requires_constrained_solver(args)

    updated = _normalize_benchmark7_solver_choice(args)

    assert updated.nonlinear_solver == "pdas"


def test_benchmark7_unconstrained_configuration_keeps_newton_solver() -> None:
    args = SimpleNamespace(
        nonlinear_solver="newton",
        alpha_bc_mode="dirichlet",
        alpha_box_constraints=False,
        enable_phi_evolution=False,
        phi_box_constraints=False,
    )

    assert not _benchmark7_requires_constrained_solver(args)

    updated = _normalize_benchmark7_solver_choice(args)

    assert updated.nonlinear_solver == "newton"


def test_legacy_single_pressure_branch_disables_pressure_mean_constraint() -> None:
    args = SimpleNamespace(
        nonlinear_solver="newton",
        pressure_mean_constraint=True,
        full_ratio_free_state=False,
        enable_phi_evolution=False,
        reduced_support_state="alpha_B",
        alpha_bc_mode="dirichlet",
        alpha_box_constraints=False,
        phi_box_constraints=False,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert not bool(updated.pressure_mean_constraint)
    assert updated.nonlinear_solver == "newton"


def test_ratio_free_stored_support_can_remain_ungauged_when_requested() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        pressure_mean_constraint=False,
        pressure_mean_gauge=False,
        allow_ungauged_free_fluid_pressure=True,
        full_ratio_free_state=True,
        enable_phi_evolution=True,
        support_physics="stored_support",
        reduced_support_state="alpha_B",
        alpha_bc_mode="dirichlet",
        alpha_box_constraints=False,
        phi_box_constraints=False,
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_mass_constraint=False,
        gamma_div=0.0,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert not bool(updated.pressure_mean_constraint)
    assert not bool(updated.pressure_mean_gauge)


def test_ratio_free_stored_support_defaults_to_solver_side_pressure_gauge() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        pressure_mean_constraint=False,
        pressure_mean_gauge=False,
        allow_ungauged_free_fluid_pressure=False,
        full_ratio_free_state=True,
        enable_phi_evolution=True,
        support_physics="stored_support",
        reduced_support_state="alpha_B",
        alpha_bc_mode="dirichlet",
        alpha_box_constraints=False,
        phi_box_constraints=False,
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_mass_constraint=False,
        gamma_div=0.0,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert not bool(updated.pressure_mean_constraint)
    assert bool(updated.pressure_mean_gauge)


def test_legacy_single_pressure_branch_disables_diffuse_interface_closures() -> None:
    args = SimpleNamespace(
        nonlinear_solver="newton",
        pressure_mean_constraint=False,
        pressure_interface_closure=True,
        interface_entry_closure=True,
        interface_bjs_closure=True,
        interface_velocity_continuity_closure=True,
        interface_traction_continuity_closure=True,
        full_ratio_free_state=False,
        enable_phi_evolution=False,
        reduced_support_state="alpha_B",
        alpha_bc_mode="dirichlet",
        alpha_box_constraints=False,
        phi_box_constraints=False,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert not bool(updated.pressure_interface_closure)
    assert not bool(updated.interface_entry_closure)
    assert not bool(updated.interface_bjs_closure)
    assert not bool(updated.interface_velocity_continuity_closure)
    assert not bool(updated.interface_traction_continuity_closure)
    assert updated.nonlinear_solver == "newton"


def test_ratio_free_stored_support_keeps_constitutive_interface_laws_but_disables_extra_penalties() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        pressure_mean_constraint=False,
        pressure_interface_closure=True,
        interface_entry_closure=True,
        interface_bjs_closure=True,
        interface_velocity_continuity_closure=True,
        interface_traction_continuity_closure=True,
        full_ratio_free_state=True,
        enable_phi_evolution=True,
        reduced_support_state="alpha_B",
        support_physics="stored_support",
        skeleton_pressure_mode="whole_domain",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_mass_constraint=False,
        phi_box_constraints=False,
        alpha_box_constraints=False,
        gamma_div=0.0,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert not bool(updated.pressure_interface_closure)
    assert bool(updated.interface_entry_closure)
    assert bool(updated.interface_bjs_closure)
    assert not bool(updated.interface_velocity_continuity_closure)
    assert not bool(updated.interface_traction_continuity_closure)


def test_ratio_free_stored_support_seboldt_branch_disables_bjs_closure() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        pressure_mean_constraint=False,
        pressure_interface_closure=True,
        interface_entry_closure=True,
        interface_bjs_closure=True,
        interface_velocity_continuity_closure=False,
        interface_traction_continuity_closure=False,
        full_ratio_free_state=True,
        enable_phi_evolution=True,
        reduced_support_state="alpha_B",
        support_physics="stored_support",
        skeleton_pressure_mode="seboldt",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_mass_constraint=False,
        phi_box_constraints=False,
        alpha_box_constraints=False,
        gamma_div=0.0,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert not bool(updated.pressure_interface_closure)
    assert bool(updated.interface_entry_closure)
    assert not bool(updated.interface_bjs_closure)


def test_benchmark7_alpha_from_refmap_ignores_alpha_box_constraints_for_solver_choice() -> None:
    args = SimpleNamespace(
        nonlinear_solver="newton",
        alpha_from_refmap=True,
        alpha_bc_mode="natural",
        alpha_box_constraints=True,
        enable_phi_evolution=False,
        phi_box_constraints=False,
    )

    assert not _benchmark7_requires_constrained_solver(args)

    updated = _normalize_benchmark7_solver_choice(args)

    assert updated.nonlinear_solver == "newton"


def test_benchmark7_constrained_configuration_keeps_explicit_ipm_solver() -> None:
    args = SimpleNamespace(
        nonlinear_solver="ipm",
        alpha_bc_mode="natural",
        alpha_box_constraints=True,
        enable_phi_evolution=False,
        phi_box_constraints=False,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert updated.nonlinear_solver == "ipm"


def test_alpha_from_refmap_filters_alpha_out_of_latent_transport_and_disables_exact_alpha_mass_constraint() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        alpha_from_refmap=True,
        latent_bounded_transport=True,
        latent_bounded_fields="alpha,phi",
        latent_bounded_formulation="transformed",
        alpha_mass_constraint=True,
        predictor_corrector_startup=False,
        enable_phi_evolution=True,
        startup_bootstrap=False,
        logistic_bounded_transform=False,
        latent_block_preconditioner=False,
        stall_frozen_transport_restart=True,
        newton_globalization="line_search",
        alpha_bc_mode="natural",
        alpha_box_constraints=True,
        phi_box_constraints=True,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert bool(updated.latent_bounded_transport)
    assert str(updated.latent_bounded_fields) == "phi"
    assert not bool(updated.alpha_mass_constraint)
    assert updated.nonlinear_solver == "newton"


def test_alpha_from_refmap_disables_latent_transport_when_no_effective_latent_fields_remain() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        alpha_from_refmap=True,
        latent_bounded_transport=True,
        latent_bounded_fields="alpha",
        latent_bounded_formulation="transformed",
        alpha_mass_constraint=True,
        predictor_corrector_startup=False,
        enable_phi_evolution=True,
        startup_bootstrap=False,
        logistic_bounded_transform=False,
        latent_block_preconditioner=False,
        stall_frozen_transport_restart=True,
        newton_globalization="line_search",
        alpha_bc_mode="natural",
        alpha_box_constraints=True,
        phi_box_constraints=True,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert not bool(updated.latent_bounded_transport)
    assert str(updated.latent_bounded_fields) == ""
    assert not bool(updated.alpha_mass_constraint)


def test_reduced_alpha_B_disables_exact_alpha_mass_constraint() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        alpha_from_refmap=False,
        latent_bounded_transport=False,
        latent_bounded_fields="",
        latent_bounded_formulation="transformed",
        alpha_mass_constraint=True,
        predictor_corrector_startup=False,
        enable_phi_evolution=False,
        reduced_support_state="alpha_B",
        startup_bootstrap=False,
        logistic_bounded_transform=False,
        latent_block_preconditioner=False,
        stall_frozen_transport_restart=False,
        newton_globalization="line_search",
        alpha_bc_mode="natural",
        alpha_box_constraints=True,
        phi_box_constraints=True,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert not bool(updated.alpha_mass_constraint)
    assert updated.nonlinear_solver == "pdas"


def test_reduced_alpha_B_disables_proactive_startup_bootstrap_on_bounded_solver() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        alpha_from_refmap=False,
        latent_bounded_transport=False,
        latent_bounded_fields="",
        latent_bounded_formulation="transformed",
        alpha_mass_constraint=False,
        predictor_corrector_startup=False,
        enable_phi_evolution=False,
        reduced_support_state="alpha_B",
        startup_bootstrap=True,
        logistic_bounded_transform=False,
        latent_block_preconditioner=False,
        stall_frozen_transport_restart=False,
        newton_globalization="line_search",
        alpha_bc_mode="natural",
        alpha_box_constraints=True,
        phi_box_constraints=True,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert not bool(updated.startup_bootstrap)
    assert updated.nonlinear_solver == "pdas"


def test_single_pressure_stored_support_core_forces_direct_alpha_and_zero_storage() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        alpha_from_refmap=True,
        latent_bounded_transport=False,
        latent_bounded_fields="",
        latent_bounded_formulation="transformed",
        alpha_mass_constraint=False,
        predictor_corrector_startup=False,
        enable_phi_evolution=False,
        reduced_support_state="alpha_B",
        support_physics="stored_support",
        startup_bootstrap=False,
        logistic_bounded_transform=False,
        latent_block_preconditioner=False,
        stall_frozen_transport_restart=False,
        newton_globalization="line_search",
        alpha_bc_mode="natural",
        alpha_box_constraints=True,
        phi_box_constraints=False,
        full_ratio_free_state=False,
        alpha_advect_with="v",
        alpha_advection_form="conservative",
        storativity_c0=1.0e-3,
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert not bool(updated.alpha_from_refmap)
    assert str(updated.alpha_advect_with) == "vS"
    assert str(updated.alpha_advection_form) == "advective"
    assert float(updated.storativity_c0) == 0.0
    assert str(updated.skeleton_pressure_mode) == "whole_domain"
    assert updated.alpha_biot is None


def test_reduced_alpha_B_enables_alpha_box_constraints_for_constrained_solver() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        alpha_from_refmap=False,
        latent_bounded_transport=False,
        latent_bounded_fields="",
        latent_bounded_formulation="transformed",
        alpha_mass_constraint=False,
        predictor_corrector_startup=False,
        enable_phi_evolution=False,
        reduced_support_state="alpha_B",
        startup_bootstrap=False,
        logistic_bounded_transform=False,
        latent_block_preconditioner=False,
        stall_frozen_transport_restart=False,
        newton_globalization="line_search",
        alpha_bc_mode="natural",
        alpha_box_constraints=False,
        phi_box_constraints=False,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert bool(updated.alpha_box_constraints)
    assert updated.nonlinear_solver == "pdas"


def test_solver_scaled_reduced_matrix_uses_newton_mode_aware_scaling() -> None:
    class _StubNewtonSolver:
        def __init__(self) -> None:
            self.called = False

        def _apply_reduced_system_scaling(self, A_red, R_red):
            self.called = True
            row = np.array([3.0, 4.0], dtype=float)
            col = np.array([5.0, 6.0], dtype=float)
            A_scaled = A_red.multiply(row[:, None]).multiply(col[None, :]).tocsr()
            return A_scaled, np.asarray(R_red, dtype=float), row, col

        def _reduced_row_scale_vector(self, A_red):
            return np.array([13.0, 17.0], dtype=float)

        def _reduced_col_scale_vector(self, A_red):
            return np.array([19.0, 23.0], dtype=float)

    solver = _StubNewtonSolver()
    A_raw = sp.csr_matrix(np.array([[2.0, 0.0], [0.0, 7.0]], dtype=float))

    A_scaled, row_scale, col_scale = _solver_scaled_reduced_matrix(solver, A_raw)

    assert solver.called
    assert np.allclose(row_scale, np.array([3.0, 4.0], dtype=float))
    assert np.allclose(col_scale, np.array([5.0, 6.0], dtype=float))
    assert np.allclose(A_scaled.toarray(), np.array([[30.0, 0.0], [0.0, 168.0]], dtype=float))


def test_alpha_from_refmap_filters_alpha_out_of_logistic_newton_fields() -> None:
    args = SimpleNamespace(
        logistic_bounded_fields="alpha,phi",
        alpha_from_refmap=True,
        enable_phi_evolution=True,
    )

    assert _effective_logistic_bounded_fields(args) == ("phi",)


def test_normalization_keeps_only_phi_logistic_field_for_refmap_newton() -> None:
    args = SimpleNamespace(
        nonlinear_solver="newton",
        alpha_from_refmap=True,
        latent_bounded_transport=False,
        alpha_mass_constraint=False,
        predictor_corrector_startup=False,
        enable_phi_evolution=True,
        startup_bootstrap=False,
        logistic_bounded_transform=True,
        logistic_bounded_fields="alpha,phi",
        latent_block_preconditioner=False,
        stall_frozen_transport_restart=False,
        newton_globalization="line_search",
        alpha_bc_mode="natural",
        alpha_box_constraints=True,
        phi_box_constraints=True,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert bool(updated.logistic_bounded_transform)
    assert str(updated.logistic_bounded_fields) == "phi"


def test_refmap_phi_only_logistic_keeps_startup_bootstrap() -> None:
    args = SimpleNamespace(
        nonlinear_solver="newton",
        alpha_from_refmap=True,
        latent_bounded_transport=False,
        alpha_mass_constraint=False,
        predictor_corrector_startup=False,
        enable_phi_evolution=True,
        startup_bootstrap=True,
        logistic_bounded_transform=True,
        logistic_bounded_fields="alpha,phi",
        latent_block_preconditioner=False,
        stall_frozen_transport_restart=False,
        newton_globalization="line_search",
        alpha_bc_mode="natural",
        alpha_box_constraints=True,
        phi_box_constraints=True,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert bool(updated.startup_bootstrap)
    assert str(updated.logistic_bounded_fields) == "phi"


def test_refmap_phi_only_logistic_raises_exact_probe_budget_for_pc_startup() -> None:
    args = SimpleNamespace(
        nonlinear_solver="newton",
        alpha_from_refmap=True,
        latent_bounded_transport=False,
        alpha_mass_constraint=False,
        predictor_corrector_startup=True,
        enable_phi_evolution=True,
        startup_bootstrap=True,
        logistic_bounded_transform=True,
        logistic_bounded_fields="alpha,phi",
        latent_block_preconditioner=False,
        stall_frozen_transport_restart=False,
        newton_globalization="line_search",
        alpha_bc_mode="natural",
        alpha_box_constraints=True,
        phi_box_constraints=True,
        pc_exact_probe_max_it=1,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert bool(updated.startup_bootstrap)
    assert str(updated.logistic_bounded_fields) == "phi"
    assert int(updated.pc_exact_probe_max_it) == 2


def test_refmap_phi_only_logistic_pc_startup_uses_line_search_then_trust() -> None:
    args = SimpleNamespace(
        nonlinear_solver="newton",
        alpha_from_refmap=True,
        latent_bounded_transport=False,
        alpha_mass_constraint=False,
        predictor_corrector_startup=True,
        enable_phi_evolution=True,
        startup_bootstrap=True,
        logistic_bounded_transform=True,
        logistic_bounded_fields="alpha,phi",
        latent_block_preconditioner=False,
        stall_frozen_transport_restart=False,
        newton_globalization="line_search",
        alpha_bc_mode="natural",
        alpha_box_constraints=True,
        phi_box_constraints=True,
        pc_exact_probe_max_it=1,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert str(updated.newton_globalization) == "line_search_then_trust"
    assert int(updated.pc_exact_probe_max_it) == 2


def test_exact_probe_strong_progress_gate_can_block_small_exact_drop() -> None:
    keep, decision = _pc_should_prefer_exact_probe(
        before_stats={"raw_inf": 1.62e-3},
        after_stats={"raw_inf": 1.513e-3},
        alpha_mass_ok=True,
        min_abs_decrease=1.0e-10,
        min_rel_improve=0.0,
        strong_min_rel_improve=0.25,
    )

    assert not bool(keep)
    assert bool(decision["exact"]["improved"])
    assert not bool(decision["strong_exact"]["improved"])


def test_exact_probe_strong_progress_gate_keeps_material_exact_drop() -> None:
    keep, decision = _pc_should_prefer_exact_probe(
        before_stats={"raw_inf": 1.62e-3},
        after_stats={"raw_inf": 9.0e-4},
        alpha_mass_ok=True,
        min_abs_decrease=1.0e-10,
        min_rel_improve=0.0,
        strong_min_rel_improve=0.25,
    )

    assert bool(keep)
    assert bool(decision["exact"]["improved"])
    assert bool(decision["strong_exact"]["improved"])


def test_nonrefmap_logistic_still_disables_startup_bootstrap() -> None:
    args = SimpleNamespace(
        nonlinear_solver="newton",
        alpha_from_refmap=False,
        latent_bounded_transport=False,
        alpha_mass_constraint=True,
        predictor_corrector_startup=False,
        enable_phi_evolution=True,
        startup_bootstrap=True,
        logistic_bounded_transform=True,
        logistic_bounded_fields="alpha,phi",
        latent_block_preconditioner=False,
        stall_frozen_transport_restart=False,
        newton_globalization="line_search",
        alpha_bc_mode="natural",
        alpha_box_constraints=True,
        phi_box_constraints=True,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert not bool(updated.startup_bootstrap)
    assert str(updated.logistic_bounded_fields) == "alpha,phi"


def test_condition_balanced_solid_cutoff_is_disabled_without_explicit_cut() -> None:
    assert (
        _condition_balanced_solid_cutoff_y(
            mechanics_nondim_mode="condition_balanced",
            y_interface=1.0,
            solid_dof_y_cut=None,
            condition_balanced_solid_cut_fix=True,
        )
        is None
    )
    assert _condition_balanced_solid_cutoff_y(
        mechanics_nondim_mode="condition_balanced",
        y_interface=1.0,
        solid_dof_y_cut=1.2,
        condition_balanced_solid_cut_fix=True,
    ) == pytest.approx(1.2)


def test_latent_pc_startup_uses_trust_region_globalization() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        latent_bounded_transport=True,
        latent_bounded_fields="alpha,phi",
        latent_bounded_formulation="transformed",
        logistic_bounded_transform=False,
        predictor_corrector_startup=True,
        enable_phi_evolution=True,
        startup_bootstrap=False,
        latent_block_preconditioner=False,
        stall_frozen_transport_restart=True,
        newton_globalization="line_search",
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert updated.nonlinear_solver == "newton"
    assert bool(updated.startup_bootstrap)
    assert not bool(updated.stall_frozen_transport_restart)
    assert str(updated.newton_globalization) == "trust_region"


def test_direct_phi_pc_startup_enables_startup_hook_without_latent_transport() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        latent_bounded_transport=False,
        logistic_bounded_transform=False,
        predictor_corrector_startup=True,
        startup_bootstrap=False,
        latent_block_preconditioner=False,
        stall_frozen_transport_restart=True,
        newton_globalization="line_search",
        alpha_bc_mode="natural",
        alpha_box_constraints=True,
        enable_phi_evolution=True,
        phi_box_constraints=True,
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert updated.nonlinear_solver == "pdas"
    assert bool(updated.startup_bootstrap)
    assert bool(updated.stall_frozen_transport_restart)
    assert str(updated.newton_globalization) == "line_search"


def test_ratio_free_post_accept_transport_auto_mode_reconfigures_main_solve() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        latent_bounded_transport=False,
        logistic_bounded_transform=False,
        predictor_corrector_startup=False,
        startup_bootstrap=True,
        latent_block_preconditioner=False,
        stall_frozen_transport_restart=True,
        newton_globalization="line_search",
        predictor="delta",
        alpha_bc_mode="natural",
        alpha_box_constraints=True,
        enable_phi_evolution=True,
        full_ratio_free_state=True,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        pressure_mean_constraint=False,
        pressure_mean_gauge=False,
        allow_ungauged_free_fluid_pressure=False,
        pressure_interface_closure=False,
        interface_entry_closure=True,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=False,
        interface_traction_continuity_closure=False,
        phi_box_constraints=True,
        gamma_div=0.0,
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        alpha_mass_constraint=False,
        alpha_from_refmap=False,
        reduced_support_state="alpha_B",
        stored_support_content_mode="evolve_B",
        transport_update_mode="auto",
        vi_c=1.0e4,
        max_it=24,
        ls_mode="dealii",
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert bool(_use_post_accept_transport_update(updated))
    assert str(_benchmark7_transport_update_label(updated)) == "post_accept"
    assert not bool(updated.startup_bootstrap)
    assert not bool(updated.stall_frozen_transport_restart)
    assert str(updated.predictor) == "prev"
    assert str(updated.newton_globalization) == "line_search_then_trust"
    assert float(updated.trust_radius_max) == pytest.approx(4.0)
    assert float(getattr(updated, "trust_min_rel_residual_drop", 0.0)) == pytest.approx(0.0)


def test_post_accept_transport_disables_solver_side_alpha_mass_equality() -> None:
    args = SimpleNamespace(
        alpha_from_refmap=False,
        enable_phi_evolution=True,
        full_ratio_free_state=True,
        support_physics="stored_support",
        transport_update_mode="post_accept",
    )

    assert bool(_use_post_accept_transport_update(args))
    assert not bool(_solver_side_alpha_mass_equality_enabled(args))


def test_ratio_free_post_accept_transport_can_be_forced_off() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        latent_bounded_transport=False,
        logistic_bounded_transform=False,
        predictor_corrector_startup=False,
        startup_bootstrap=True,
        latent_block_preconditioner=False,
        stall_frozen_transport_restart=True,
        newton_globalization="line_search",
        predictor="prev",
        alpha_bc_mode="natural",
        alpha_box_constraints=True,
        enable_phi_evolution=True,
        full_ratio_free_state=True,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        pressure_mean_constraint=False,
        pressure_mean_gauge=False,
        allow_ungauged_free_fluid_pressure=False,
        pressure_interface_closure=False,
        interface_entry_closure=True,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=False,
        interface_traction_continuity_closure=False,
        phi_box_constraints=True,
        gamma_div=0.0,
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        alpha_mass_constraint=False,
        alpha_from_refmap=False,
        reduced_support_state="alpha_B",
        stored_support_content_mode="evolve_B",
        transport_update_mode="monolithic",
        vi_c=1.0e4,
        max_it=24,
        ls_mode="dealii",
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert not bool(_use_post_accept_transport_update(updated))
    assert str(_benchmark7_transport_update_label(updated)) == "monolithic"
    assert bool(updated.startup_bootstrap)
    assert bool(updated.stall_frozen_transport_restart)
    assert str(updated.newton_globalization) == "line_search"


def test_post_accept_transport_solver_override_is_honored() -> None:
    assert (
        _startup_stage_solver_kind(
            main_solver_kind="pdas",
            active_fields=["alpha", "B", "mu_alpha"],
            stage_name="post_accept_transport",
            transport_solver_kind_override="newton",
        )
        == "newton"
    )


def test_closed_top_phi_cleanup_keeps_phi_unchanged() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=2,
        ny=3,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
    )
    problem["phi_k"].nodal_values[:] = 1.02
    before = np.asarray(problem["phi_k"].nodal_values, dtype=float).copy()

    _apply_open_top_global_phi_cleanup(
        args=_make_args(enable_phi_evolution=True, top_drainage_transport=False),
        problem=problem,
        funcs=[problem["phi_k"]],
        find_named_function=_find_named_function,
    )

    assert np.allclose(np.asarray(problem["phi_k"].nodal_values, dtype=float), before)


def test_create_problem_adds_alpha_mass_lm_only_for_direct_alpha_constraint() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=2,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=False,
        alpha_mass_constraint=True,
    )
    assert problem["alpha_mass_constraint"]
    assert problem["alpha_mass_lm_k"] is not None
    assert "alpha_mass_lm" in problem["dh"].field_names

    latent_problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=2,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=False,
        latent_bounded_transport=True,
        latent_bounded_fields=("alpha",),
        alpha_mass_constraint=True,
    )
    assert not latent_problem["alpha_mass_constraint"]
    assert latent_problem["alpha_mass_lm_k"] is None
    assert "alpha_mass_lm" not in latent_problem["dh"].field_names


def test_vi_linear_equalities_skip_alpha_mass_when_exact_constraint_is_active() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=2,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
        alpha_mass_constraint=True,
    )
    equalities = _build_vi_linear_equalities(
        args=_make_args(enable_phi_evolution=True, top_drainage_transport=False),
        problem=problem,
        qdeg=4,
        alpha_bc_mode_key="natural",
        find_named_function=_find_named_function,
    )
    assert [eq.name for eq in equalities] == ["phi_biofilm_fluid_mass"]


def test_vi_linear_equalities_skip_alpha_mass_for_refmap_alpha_mode() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=2,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
    )
    args = SimpleNamespace(
        alpha_box_constraints=True,
        alpha_from_refmap=True,
        enable_phi_evolution=True,
        top_drainage_transport=False,
        phi_box_constraints=True,
        backend="python",
    )
    equalities = _build_vi_linear_equalities(
        args=args,
        problem=problem,
        qdeg=4,
        alpha_bc_mode_key="natural",
        find_named_function=_find_named_function,
    )
    assert [eq.name for eq in equalities] == ["phi_biofilm_fluid_mass"]


def test_build_bcs_can_apply_equilibrium_phi_dirichlet_on_sides_and_top() -> None:
    bcs = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=True,
        y_interface=1.0,
        eps_alpha=0.05,
        phi_b=0.18,
        v_in=5.0,
        t_ramp=0.0,
        alpha_bc_mode="natural",
        phi_bc_mode="equilibrium",
        pressure_mean_constraint=False,
    )
    phi_bcs = [(bc.field, bc.domain_tag) for bc in bcs if bc.method == "dirichlet" and bc.field == "phi"]
    assert ("phi", "left") in phi_bcs
    assert ("phi", "right") in phi_bcs
    assert ("phi", "top") in phi_bcs
    assert ("phi", "bottom") not in phi_bcs


def test_build_bcs_lateral_clamped_enforces_zero_side_displacement_components() -> None:
    bcs = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=True,
        y_interface=1.0,
        eps_alpha=0.05,
        phi_b=0.18,
        v_in=5.0,
        t_ramp=0.0,
        alpha_bc_mode="natural",
        solid_bc_mode="lateral_clamped",
        pressure_mean_constraint=False,
    )
    side_pairs = {(bc.field, bc.domain_tag) for bc in bcs if bc.method == "dirichlet"}
    assert ("u_x", "left") in side_pairs
    assert ("u_y", "left") in side_pairs
    assert ("u_x", "right") in side_pairs
    assert ("u_y", "right") in side_pairs


def test_build_bcs_wall_normal_leaves_side_tangential_displacement_free() -> None:
    bcs = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=True,
        y_interface=1.0,
        eps_alpha=0.05,
        phi_b=0.18,
        v_in=5.0,
        t_ramp=0.0,
        alpha_bc_mode="natural",
        solid_bc_mode="wall_normal",
        pressure_mean_constraint=False,
    )
    side_pairs = {(bc.field, bc.domain_tag) for bc in bcs if bc.method == "dirichlet"}
    assert ("u_x", "left") in side_pairs
    assert ("u_y", "left") not in side_pairs
    assert ("u_x", "right") in side_pairs
    assert ("u_y", "right") not in side_pairs


def test_build_bcs_keeps_legacy_top_pressure_dirichlet_when_pressure_mean_constraint_is_requested() -> None:
    bcs = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=False,
        y_interface=1.0,
        eps_alpha=0.05,
        phi_b=0.18,
        v_in=5.0,
        t_ramp=0.0,
        alpha_bc_mode="natural",
        pressure_mean_constraint=True,
        full_ratio_free_state=False,
    )
    top_pairs = {(bc.field, bc.domain_tag) for bc in bcs if bc.method == "dirichlet"}
    assert ("p", "top") in top_pairs
    assert ("p_pore", "top") not in top_pairs


def test_build_bcs_split_pressure_mean_constraint_moves_top_constraint_to_pore_pressure() -> None:
    bcs = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=True,
        y_interface=1.0,
        eps_alpha=0.05,
        phi_b=0.18,
        v_in=5.0,
        t_ramp=0.0,
        alpha_bc_mode="natural",
        pressure_mean_constraint=True,
        full_ratio_free_state=True,
    )
    top_pairs = {(bc.field, bc.domain_tag) for bc in bcs if bc.method == "dirichlet"}
    assert ("p", "top") not in top_pairs
    assert ("p_pore", "top") in top_pairs


def test_build_bcs_primary_q_branch_pins_sidewall_q_normal() -> None:
    bcs = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=True,
        y_interface=1.0,
        eps_alpha=0.05,
        phi_b=0.18,
        v_in=5.0,
        t_ramp=0.0,
        alpha_bc_mode="natural",
        pressure_mean_constraint=False,
        full_ratio_free_state=True,
        split_primary_darcy_flux=True,
    )
    side_pairs = {(bc.field, bc.domain_tag) for bc in bcs if bc.method == "dirichlet"}
    assert ("lambda_drag_x", "left") in side_pairs
    assert ("lambda_drag_x", "right") in side_pairs
    assert ("lambda_drag_y", "left") not in side_pairs
    assert ("lambda_drag_y", "right") not in side_pairs


def test_build_bcs_primary_q_branch_supports_hdiv_flux_with_cg_fluid() -> None:
    bcs = _build_bcs(
        fluid_space="cg",
        darcy_flux_space="hdiv",
        enable_phi_evolution=True,
        y_interface=1.0,
        eps_alpha=0.05,
        phi_b=0.18,
        v_in=5.0,
        t_ramp=0.0,
        alpha_bc_mode="natural",
        pressure_mean_constraint=False,
        full_ratio_free_state=True,
        split_primary_darcy_flux=True,
    )
    side_pairs = {(bc.field, bc.domain_tag) for bc in bcs if bc.method == "dirichlet"}
    assert ("q", "left") in side_pairs
    assert ("q", "right") in side_pairs
    assert ("lambda_drag_x", "left") not in side_pairs


def test_create_problem_supports_cg_fluid_with_hdiv_primary_q() -> None:
    problem = _build_ratio_free_full_problem(
        split_primary_darcy_flux=True,
        fluid_space="cg",
        darcy_flux_space="hdiv",
        fluid_hdiv_order=1,
    )

    assert str(problem["fluid_space"]) == "cg"
    assert str(problem["darcy_flux_space"]) == "hdiv"
    assert bool(problem["primary_darcy_hdiv"])
    assert "q" in tuple(getattr(problem["dh"], "field_names", tuple()) or tuple())
    assert problem.get("lambda_drag_k") is not None
    assert problem.get("q_flux_k") is problem.get("lambda_drag_k")


def test_ratio_free_pressure_mean_constraint_weights_only_free_fluid_pressure_region() -> None:
    problem = _build_ratio_free_full_problem(pressure_mean_constraint=True)
    zero_vec = lambda x, y: np.array([0.0, 0.0])
    for key in ("v_k", "v_n", "vS_k", "vS_n", "u_k", "u_n"):
        problem[key].set_values_from_function(zero_vec)
    for key in ("alpha_k", "alpha_n"):
        problem[key].nodal_values[:] = 0.25
    problem["p_k"].nodal_values[:] = 2.0
    problem["p_n"].nodal_values[:] = 2.0
    problem["p_pore_k"].nodal_values[:] = 0.0
    problem["p_pore_n"].nodal_values[:] = 0.0
    problem["p_mean_k"].nodal_values[:] = 0.0
    problem["p_mean_n"].nodal_values[:] = 0.0

    _build_full_forms(
        problem,
        support_physics="stored_support",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_biot=None,
        skeleton_pressure_mode="whole_domain",
        storativity_c0=0.0,
        include_skeleton_acceleration=False,
    )

    mean_block = _assemble_block(problem, problem["_pressure_mean_residual_form"], "p_mean")
    expected_form = Constant((1.0 - 0.25) * 2.0) * problem["p_mean_test"] * dx(metadata={"q": 6})
    expected_block = _assemble_block(problem, expected_form, "p_mean")

    assert np.allclose(mean_block, expected_block, atol=1.0e-12, rtol=1.0e-12)


def test_build_forms_populates_alpha_mass_constraint_row_when_enabled() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=2,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=False,
        alpha_mass_constraint=True,
    )
    for key in ("vS_k", "vS_n", "u_k", "u_n"):
        problem[key].nodal_values[:] = 0.0
    for key in ("p_k", "p_n", "mu_k", "mu_n"):
        problem[key].nodal_values[:] = 0.0
    problem["alpha_n"].nodal_values[:] = 0.25
    problem["alpha_k"].nodal_values[:] = 0.50
    problem["alpha_mass_lm_k"].nodal_values[:] = 0.0
    problem["alpha_mass_lm_n"].nodal_values[:] = 0.0

    _build_forms(
        problem,
        qdeg=4,
        dt_c=Constant(0.1),
        theta=1.0,
        rho_f=1.0,
        mu_f=0.035,
        mu_b=0.035,
        mu_b_model="mu",
        kappa_inv=1.0e3,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        phi_b=0.18,
        M_alpha=0.0,
        gamma_alpha=1.0,
        eps_alpha=0.1,
        solid_visco_eta=0.0,
        gamma_div=0.0,
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
        alpha_supg=0.0,
        alpha_cip=0.0,
        alpha_regularization="none",
        alpha_reg_gamma=1.0,
        alpha_reg_eps_normal=0.1,
        alpha_reg_eps_tangent=0.025,
        alpha_reg_eta=1.0e-12,
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        solid_model="linear",
        kappa_inv_model="refmap",
        enable_phi_evolution=False,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        skeleton_inertia_convection="lagged",
        fluid_convection="off",
        support_physics="internal_conversion",
    )

    assert problem["_alpha_mass_constraint_residual_form"] is not None
    assert problem["_alpha_mass_constraint_jacobian_form"] is not None
    lm_block = _assemble_block(problem, problem["_alpha_mass_constraint_residual_form"], "alpha_mass_lm")
    assert np.linalg.norm(lm_block, ord=np.inf) > 0.0


def test_rigid_support_diagnostic_active_fields_include_split_pressure_gauge_and_q() -> None:
    problem = _build_ratio_free_full_problem(
        pressure_mean_constraint=True,
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
    )

    fields = _rigid_support_diagnostic_active_fields(problem)

    assert fields == ["v_x", "v_y", "p", "p_pore", "p_mean", "lambda_drag_x", "lambda_drag_y"]


def test_solid_only_diagnostic_active_fields_keep_only_skeleton_block() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=1,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
        pressure_mean_constraint=True,
        full_ratio_free_state=True,
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        solid_volumetric_split=True,
    )

    fields = _solid_only_diagnostic_active_fields(problem)

    assert fields == ["vS_x", "vS_y", "u_x", "u_y", "pi_s"]


def test_fixed_fluid_poro_solid_diagnostic_active_fields_keep_pore_q_and_skeleton() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=1,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
        pressure_mean_constraint=True,
        full_ratio_free_state=True,
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        solid_volumetric_split=True,
    )

    fields = _fixed_fluid_poro_solid_diagnostic_active_fields(problem)

    assert fields == ["p_pore", "lambda_drag_x", "lambda_drag_y", "vS_x", "vS_y", "u_x", "u_y", "B", "pi_s"]


def test_all_porous_sideflow_diagnostic_active_fields_keep_pore_q_and_skeleton() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=1,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
        pressure_mean_constraint=False,
        full_ratio_free_state=True,
        drag_formulation="mixed_lm",
        split_primary_darcy_flux=True,
        solid_volumetric_split=True,
    )

    fields = _all_porous_sideflow_diagnostic_active_fields(problem)

    assert fields == ["p_pore", "lambda_drag_x", "lambda_drag_y", "vS_x", "vS_y", "u_x", "u_y", "B", "pi_s"]


def test_rigid_support_diagnostic_forces_monolithic_no_startup_paths() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        enable_phi_evolution=True,
        full_ratio_free_state=True,
        split_primary_darcy_flux=True,
        fluid_space="cg",
        darcy_flux_space="cg",
        reduced_support_state="alpha_B",
        support_physics="stored_support",
        drag_formulation="mixed_lm",
        pressure_mean_constraint=False,
        pressure_mean_gauge=False,
        allow_ungauged_free_fluid_pressure=False,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=True,
        interface_velocity_tangential_strength=0.0,
        interface_traction_continuity_closure=True,
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_mass_constraint=False,
        phi_box_constraints=False,
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        gamma_div=0.0,
        rigid_support_diagnostic=True,
        startup_bootstrap=True,
        predictor_corrector_startup=True,
        stall_frozen_transport_restart=True,
        transport_update_mode="post_accept",
        latent_bounded_transport=False,
        logistic_bounded_transform=False,
        alpha_box_constraints=True,
        alpha_from_refmap=False,
        alpha_bc_mode="natural",
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert bool(updated.rigid_support_diagnostic)
    assert bool(updated.pressure_mean_constraint)
    assert not bool(updated.pressure_mean_gauge)
    assert not bool(updated.pressure_interface_closure)
    assert not bool(updated.interface_entry_closure)
    assert not bool(updated.interface_bjs_closure)
    assert bool(updated.interface_velocity_continuity_closure)
    assert bool(updated.interface_traction_continuity_closure)
    assert not bool(updated.startup_bootstrap)
    assert not bool(updated.predictor_corrector_startup)
    assert not bool(updated.stall_frozen_transport_restart)
    assert str(updated.transport_update_mode) == "monolithic"


def test_mixed_space_q_primary_branch_prefers_physical_pressure_reference() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        enable_phi_evolution=True,
        full_ratio_free_state=True,
        split_primary_darcy_flux=True,
        fluid_space="cg",
        darcy_flux_space="hdiv",
        reduced_support_state="alpha_B",
        support_physics="stored_support",
        drag_formulation="mixed_lm",
        pressure_mean_constraint=False,
        pressure_mean_gauge=False,
        allow_ungauged_free_fluid_pressure=False,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=True,
        interface_velocity_tangential_strength=0.0,
        interface_traction_continuity_closure=True,
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_mass_constraint=False,
        phi_box_constraints=False,
        skeleton_pressure_mode="whole_domain",
        alpha_biot=None,
        gamma_div=0.0,
        rigid_support_diagnostic=True,
        startup_bootstrap=True,
        predictor_corrector_startup=True,
        stall_frozen_transport_restart=True,
        transport_update_mode="post_accept",
        latent_bounded_transport=False,
        logistic_bounded_transform=False,
        alpha_box_constraints=True,
        alpha_from_refmap=False,
        alpha_bc_mode="natural",
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert bool(updated.rigid_support_diagnostic)
    assert not bool(updated.pressure_mean_constraint)
    assert not bool(updated.pressure_mean_gauge)
    assert bool(updated.allow_ungauged_free_fluid_pressure)
    assert str(updated.skeleton_pressure_mode) == "seboldt"
    assert float(updated.alpha_biot) == pytest.approx(1.0)


def test_split_q_primary_branch_promotes_mixed_q_and_nitsche_interface_defaults() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        enable_phi_evolution=True,
        full_ratio_free_state=True,
        split_primary_darcy_flux=True,
        fluid_space="cg",
        darcy_flux_space="auto",
        reduced_support_state="alpha_B",
        support_physics="stored_support",
        drag_formulation="mixed_lm",
        pressure_mean_constraint=False,
        pressure_mean_gauge=False,
        allow_ungauged_free_fluid_pressure=False,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=True,
        interface_velocity_tangential_strength=0.0,
        interface_velocity_method="penalty",
        interface_traction_continuity_closure=True,
        interface_traction_method="penalty",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_mass_constraint=False,
        phi_box_constraints=False,
        skeleton_pressure_mode="whole_domain",
        alpha_biot=None,
        gamma_div=0.0,
        rigid_support_diagnostic=False,
        startup_bootstrap=False,
        predictor_corrector_startup=False,
        stall_frozen_transport_restart=False,
        transport_update_mode="monolithic",
        latent_bounded_transport=False,
        logistic_bounded_transform=False,
        alpha_box_constraints=True,
        alpha_from_refmap=False,
        alpha_bc_mode="natural",
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert str(updated.darcy_flux_space) == "hdiv"
    assert str(updated.interface_velocity_method) == "nitsche"
    assert str(updated.interface_traction_method) == "nitsche"
    assert not bool(updated.interface_entry_closure)
    assert bool(updated.interface_traction_continuity_closure)
    assert str(updated.skeleton_pressure_mode) == "seboldt"
    assert float(updated.alpha_biot) == pytest.approx(1.0)


def test_split_q_primary_branch_auto_mixed_space_prefers_physical_pressure_reference() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        enable_phi_evolution=True,
        full_ratio_free_state=True,
        split_primary_darcy_flux=True,
        fluid_space="cg",
        darcy_flux_space="auto",
        reduced_support_state="alpha_B",
        support_physics="stored_support",
        drag_formulation="mixed_lm",
        pressure_mean_constraint=False,
        pressure_mean_gauge=False,
        allow_ungauged_free_fluid_pressure=False,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=True,
        interface_velocity_tangential_strength=0.0,
        interface_velocity_method="nitsche",
        interface_traction_continuity_closure=True,
        interface_traction_method="nitsche",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_mass_constraint=False,
        phi_box_constraints=False,
        skeleton_pressure_mode="whole_domain",
        alpha_biot=None,
        gamma_div=0.0,
        rigid_support_diagnostic=True,
        startup_bootstrap=True,
        predictor_corrector_startup=True,
        stall_frozen_transport_restart=True,
        transport_update_mode="post_accept",
        latent_bounded_transport=False,
        logistic_bounded_transform=False,
        alpha_box_constraints=True,
        alpha_from_refmap=False,
        alpha_bc_mode="natural",
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert str(updated.darcy_flux_space) == "hdiv"
    assert not bool(updated.pressure_mean_constraint)
    assert not bool(updated.pressure_mean_gauge)
    assert bool(updated.allow_ungauged_free_fluid_pressure)
    assert str(updated.skeleton_pressure_mode) == "seboldt"
    assert float(updated.alpha_biot) == pytest.approx(1.0)


def test_split_q_primary_branch_forces_exact_biot_loading_by_default() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        enable_phi_evolution=True,
        full_ratio_free_state=True,
        split_primary_darcy_flux=True,
        fluid_space="cg",
        darcy_flux_space="cg",
        reduced_support_state="alpha_B",
        support_physics="stored_support",
        drag_formulation="mixed_lm",
        pressure_mean_constraint=False,
        pressure_mean_gauge=False,
        allow_ungauged_free_fluid_pressure=False,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=False,
        interface_velocity_tangential_strength=0.0,
        interface_traction_continuity_closure=False,
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_mass_constraint=False,
        phi_box_constraints=False,
        skeleton_pressure_mode="whole_domain",
        alpha_biot=None,
        gamma_div=0.0,
        rigid_support_diagnostic=False,
        startup_bootstrap=False,
        predictor_corrector_startup=False,
        stall_frozen_transport_restart=False,
        transport_update_mode="monolithic",
        latent_bounded_transport=False,
        logistic_bounded_transform=False,
        alpha_box_constraints=True,
        alpha_from_refmap=False,
        alpha_bc_mode="natural",
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert str(updated.skeleton_pressure_mode) == "seboldt"
    assert float(updated.alpha_biot) == pytest.approx(1.0)
    assert not bool(updated.interface_velocity_continuity_closure)


def test_split_q_primary_branch_keeps_entry_with_traction_continuity() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        enable_phi_evolution=True,
        full_ratio_free_state=True,
        split_primary_darcy_flux=True,
        fluid_space="cg",
        darcy_flux_space="auto",
        reduced_support_state="alpha_B",
        support_physics="stored_support",
        drag_formulation="mixed_lm",
        pressure_mean_constraint=False,
        pressure_mean_gauge=False,
        allow_ungauged_free_fluid_pressure=False,
        pressure_interface_closure=False,
        interface_entry_closure=True,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=True,
        interface_velocity_tangential_strength=0.0,
        interface_velocity_method="nitsche",
        interface_traction_continuity_closure=True,
        interface_traction_method="nitsche",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_mass_constraint=False,
        phi_box_constraints=False,
        skeleton_pressure_mode="whole_domain",
        alpha_biot=None,
        gamma_div=0.0,
        rigid_support_diagnostic=False,
        startup_bootstrap=False,
        predictor_corrector_startup=False,
        stall_frozen_transport_restart=False,
        transport_update_mode="monolithic",
        latent_bounded_transport=False,
        logistic_bounded_transform=False,
        alpha_box_constraints=True,
        alpha_from_refmap=False,
        alpha_bc_mode="natural",
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert bool(updated.interface_entry_closure)
    assert bool(updated.interface_traction_continuity_closure)


def test_split_q_primary_branch_does_not_force_entry_with_traction_continuity() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        enable_phi_evolution=True,
        full_ratio_free_state=True,
        split_primary_darcy_flux=True,
        fluid_space="cg",
        darcy_flux_space="auto",
        reduced_support_state="alpha_B",
        support_physics="stored_support",
        drag_formulation="mixed_lm",
        pressure_mean_constraint=False,
        pressure_mean_gauge=False,
        allow_ungauged_free_fluid_pressure=False,
        pressure_interface_closure=False,
        interface_entry_closure=False,
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=True,
        interface_velocity_tangential_strength=0.0,
        interface_velocity_method="nitsche",
        interface_traction_continuity_closure=True,
        interface_traction_method="nitsche",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_mass_constraint=False,
        phi_box_constraints=False,
        skeleton_pressure_mode="whole_domain",
        alpha_biot=None,
        gamma_div=0.0,
        rigid_support_diagnostic=False,
        startup_bootstrap=False,
        predictor_corrector_startup=False,
        stall_frozen_transport_restart=False,
        transport_update_mode="monolithic",
        latent_bounded_transport=False,
        logistic_bounded_transform=False,
        alpha_box_constraints=True,
        alpha_from_refmap=False,
        alpha_bc_mode="natural",
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert not bool(updated.interface_entry_closure)
    assert bool(updated.interface_traction_continuity_closure)


def test_split_q_primary_branch_keeps_bjs_with_normal_mass_continuity() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        enable_phi_evolution=True,
        full_ratio_free_state=True,
        split_primary_darcy_flux=True,
        fluid_space="cg",
        darcy_flux_space="auto",
        reduced_support_state="alpha_B",
        support_physics="stored_support",
        drag_formulation="mixed_lm",
        pressure_mean_constraint=False,
        pressure_mean_gauge=False,
        allow_ungauged_free_fluid_pressure=False,
        pressure_interface_closure=False,
        interface_entry_closure=True,
        interface_bjs_closure=True,
        interface_velocity_continuity_closure=True,
        interface_velocity_tangential_strength=0.0,
        interface_velocity_method="nitsche",
        interface_traction_continuity_closure=False,
        interface_traction_method="nitsche",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_mass_constraint=False,
        phi_box_constraints=False,
        skeleton_pressure_mode="whole_domain",
        alpha_biot=None,
        gamma_div=0.0,
        rigid_support_diagnostic=False,
        startup_bootstrap=False,
        predictor_corrector_startup=False,
        stall_frozen_transport_restart=False,
        transport_update_mode="monolithic",
        latent_bounded_transport=False,
        logistic_bounded_transform=False,
        alpha_box_constraints=True,
        alpha_from_refmap=False,
        alpha_bc_mode="natural",
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert bool(updated.interface_velocity_continuity_closure)
    assert float(updated.interface_velocity_tangential_strength) == pytest.approx(0.0)
    assert bool(updated.interface_bjs_closure)


def test_split_q_primary_branch_keeps_bjs_when_exact_biot_loading_is_explicit() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        enable_phi_evolution=True,
        full_ratio_free_state=True,
        split_primary_darcy_flux=True,
        fluid_space="cg",
        darcy_flux_space="hdiv",
        reduced_support_state="alpha_B",
        support_physics="stored_support",
        drag_formulation="mixed_lm",
        pressure_mean_constraint=False,
        pressure_mean_gauge=False,
        allow_ungauged_free_fluid_pressure=False,
        pressure_interface_closure=False,
        interface_entry_closure=True,
        interface_bjs_closure=True,
        interface_velocity_continuity_closure=True,
        interface_velocity_tangential_strength=0.0,
        interface_velocity_method="nitsche",
        interface_traction_continuity_closure=False,
        interface_traction_method="nitsche",
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_mass_constraint=False,
        phi_box_constraints=False,
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        gamma_div=0.0,
        rigid_support_diagnostic=False,
        startup_bootstrap=False,
        predictor_corrector_startup=False,
        stall_frozen_transport_restart=False,
        transport_update_mode="monolithic",
        latent_bounded_transport=False,
        logistic_bounded_transform=False,
        alpha_box_constraints=True,
        alpha_from_refmap=False,
        alpha_bc_mode="natural",
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert str(updated.skeleton_pressure_mode) == "seboldt"
    assert float(updated.alpha_biot) == pytest.approx(1.0)
    assert bool(updated.interface_bjs_closure)


def test_solid_only_diagnostic_forces_exact_interface_q_primary_branch() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        enable_phi_evolution=False,
        full_ratio_free_state=False,
        split_primary_darcy_flux=False,
        reduced_support_state="alpha_B",
        support_physics="legacy_exchange",
        drag_formulation="drag_al",
        pressure_mean_constraint=False,
        pressure_mean_gauge=True,
        allow_ungauged_free_fluid_pressure=False,
        pressure_interface_closure=True,
        interface_entry_closure=True,
        interface_bjs_closure=True,
        interface_velocity_continuity_closure=False,
        interface_velocity_tangential_strength=0.0,
        interface_traction_continuity_closure=False,
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_mass_constraint=False,
        phi_box_constraints=False,
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        gamma_div=0.0,
        rigid_support_diagnostic=True,
        solid_only_diagnostic=True,
        startup_bootstrap=True,
        predictor_corrector_startup=True,
        stall_frozen_transport_restart=True,
        transport_update_mode="post_accept",
        latent_bounded_transport=False,
        logistic_bounded_transform=False,
        alpha_box_constraints=True,
        alpha_from_refmap=False,
        alpha_bc_mode="natural",
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert bool(updated.enable_phi_evolution)
    assert bool(updated.full_ratio_free_state)
    assert bool(updated.split_primary_darcy_flux)
    assert str(updated.support_physics) == "stored_support"
    assert str(updated.drag_formulation) == "mixed_lm"
    assert bool(updated.pressure_mean_constraint)
    assert not bool(updated.pressure_mean_gauge)
    assert not bool(updated.pressure_interface_closure)
    assert not bool(updated.interface_entry_closure)
    assert not bool(updated.interface_bjs_closure)
    assert bool(updated.interface_velocity_continuity_closure)
    assert bool(updated.interface_traction_continuity_closure)
    assert not bool(updated.rigid_support_diagnostic)
    assert not bool(updated.startup_bootstrap)
    assert not bool(updated.predictor_corrector_startup)
    assert not bool(updated.stall_frozen_transport_restart)
    assert str(updated.transport_update_mode) == "monolithic"


def test_fixed_fluid_poro_solid_diagnostic_forces_exact_interface_q_primary_branch() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        enable_phi_evolution=False,
        full_ratio_free_state=False,
        split_primary_darcy_flux=False,
        reduced_support_state="alpha_B",
        support_physics="legacy_exchange",
        drag_formulation="drag_al",
        pressure_mean_constraint=False,
        pressure_mean_gauge=True,
        allow_ungauged_free_fluid_pressure=False,
        pressure_interface_closure=True,
        interface_entry_closure=True,
        interface_bjs_closure=True,
        interface_velocity_continuity_closure=False,
        interface_velocity_tangential_strength=0.0,
        interface_traction_continuity_closure=False,
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        alpha_mass_constraint=False,
        phi_box_constraints=False,
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        gamma_div=0.0,
        rigid_support_diagnostic=False,
        solid_only_diagnostic=False,
        fixed_fluid_poro_solid_diagnostic=True,
        startup_bootstrap=True,
        predictor_corrector_startup=True,
        stall_frozen_transport_restart=True,
        transport_update_mode="post_accept",
        latent_bounded_transport=False,
        logistic_bounded_transform=False,
        alpha_box_constraints=True,
        alpha_from_refmap=False,
        alpha_bc_mode="natural",
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert bool(updated.enable_phi_evolution)
    assert bool(updated.full_ratio_free_state)
    assert bool(updated.split_primary_darcy_flux)
    assert str(updated.support_physics) == "stored_support"
    assert str(updated.drag_formulation) == "mixed_lm"
    assert bool(updated.pressure_mean_constraint)
    assert not bool(updated.pressure_mean_gauge)
    assert not bool(updated.pressure_interface_closure)
    assert not bool(updated.interface_entry_closure)
    assert not bool(updated.interface_bjs_closure)
    assert bool(updated.interface_velocity_continuity_closure)
    assert bool(updated.interface_traction_continuity_closure)
    assert not bool(updated.startup_bootstrap)
    assert not bool(updated.predictor_corrector_startup)
    assert not bool(updated.stall_frozen_transport_restart)
    assert str(updated.transport_update_mode) == "monolithic"


def test_all_porous_sideflow_diagnostic_forces_interface_free_q_primary_branch() -> None:
    args = SimpleNamespace(
        nonlinear_solver="stage",
        enable_phi_evolution=False,
        full_ratio_free_state=False,
        split_primary_darcy_flux=False,
        reduced_support_state="alpha_B",
        support_physics="legacy_exchange",
        drag_formulation="drag_al",
        pressure_mean_constraint=True,
        pressure_mean_gauge=True,
        allow_ungauged_free_fluid_pressure=False,
        pressure_interface_closure=True,
        interface_entry_closure=True,
        interface_bjs_closure=True,
        interface_velocity_continuity_closure=True,
        interface_velocity_tangential_strength=0.0,
        interface_traction_continuity_closure=True,
        alpha_advect_with="v",
        alpha_advection_form="conservative",
        alpha_mass_constraint=False,
        phi_box_constraints=False,
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        gamma_div=0.0,
        rigid_support_diagnostic=False,
        solid_only_diagnostic=False,
        fixed_fluid_poro_solid_diagnostic=False,
        all_porous_sideflow_diagnostic=True,
        startup_bootstrap=True,
        predictor_corrector_startup=True,
        stall_frozen_transport_restart=True,
        transport_update_mode="post_accept",
        latent_bounded_transport=False,
        logistic_bounded_transform=False,
        alpha_box_constraints=True,
        alpha_from_refmap=False,
        alpha_bc_mode="natural",
    )

    updated = _normalize_benchmark7_solver_choice(args)

    assert str(updated.nonlinear_solver) == "pdas"
    assert bool(updated.enable_phi_evolution)
    assert bool(updated.full_ratio_free_state)
    assert bool(updated.split_primary_darcy_flux)
    assert str(updated.support_physics) == "stored_support"
    assert str(updated.drag_formulation) == "mixed_lm"
    assert not bool(updated.pressure_interface_closure)
    assert not bool(updated.interface_entry_closure)
    assert not bool(updated.interface_bjs_closure)
    assert not bool(updated.interface_velocity_continuity_closure)
    assert not bool(updated.interface_traction_continuity_closure)
    assert not bool(updated.pressure_mean_constraint)
    assert not bool(updated.pressure_mean_gauge)
    assert bool(updated.allow_ungauged_free_fluid_pressure)
    assert not bool(updated.startup_bootstrap)
    assert not bool(updated.predictor_corrector_startup)
    assert not bool(updated.stall_frozen_transport_restart)
    assert str(updated.transport_update_mode) == "monolithic"
