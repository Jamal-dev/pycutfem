import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import (
    _apply_open_top_global_phi_cleanup,
    _benchmark7_requires_constrained_solver,
    _build_forms,
    _build_support_aware_phi_box_bounds,
    _build_transport_measures,
    _build_vi_linear_equalities,
    _configure_benchmark7_cpp_fuse_integrals,
    _condition_balanced_solid_cutoff_y,
    _create_problem,
    _effective_eps_alpha,
    _effective_logistic_bounded_fields,
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
    _predictor_corrector_startup_enabled,
    _should_use_frozen_transport_restart,
    _should_use_staggered_predictor_after_large_step,
    _startup_stage_relaxed_accept_ginf,
    _startup_monolithic_max_it,
    _startup_stage_solver_kind,
    _named_constant,
)
from tests.test_benchmark7_solver_backend_parity import _initialize_small_benchmark7_state
from pycutfem.jit import _form_rank
from pycutfem.jit.cache import KernelCache
from pycutfem.jit.ir import strip_side_metadata
from pycutfem.jit.visitor import IRGenerator
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Constant, Integral as ExprIntegral
from pycutfem.ufl.forms import Equation, Form, assemble_form


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


def test_benchmark7_cli_defaults_use_relaxed_newton_target_and_pc_startup(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["paper1_benchmark7_seboldt.py"])
    args = _parse_args()

    assert float(args.newton_tol) == 1.0e-6
    assert float(args.newton_rtol) == 1.0e-6
    assert bool(args.alpha_mass_constraint)
    assert not bool(args.alpha_from_refmap)
    assert not bool(args.condition_balanced_solid_cut_fix)
    assert str(args.latent_bounded_fields) == "alpha,phi"
    assert bool(args.predictor_corrector_startup)
    assert int(args.pc_p1_max_it) == 12
    assert int(args.pc_p2_max_it) == 12
    assert int(args.pc_exact_probe_max_it) == 1


def test_benchmark7_cli_accepts_tanh_latent_map(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["paper1_benchmark7_seboldt.py", "--latent-bounded-map", "tanh"])
    args = _parse_args()

    assert str(args.latent_bounded_map) == "tanh"


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
