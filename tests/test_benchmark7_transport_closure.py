import sys
from types import SimpleNamespace

import numpy as np

from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import (
    _apply_open_top_global_phi_cleanup,
    _benchmark7_requires_constrained_solver,
    _build_forms,
    _build_support_aware_phi_box_bounds,
    _build_transport_measures,
    _build_vi_linear_equalities,
    _create_problem,
    _latent_bounded_fields,
    _normalize_benchmark7_solver_choice,
    _parse_args,
    _should_use_frozen_transport_restart,
    _should_use_staggered_predictor_after_large_step,
    _startup_stage_relaxed_accept_ginf,
    _startup_monolithic_max_it,
    _startup_stage_solver_kind,
)
from pycutfem.ufl.expressions import Constant
from pycutfem.ufl.forms import Equation, assemble_form


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


def _build_full_forms(problem, *, ds_alpha_transport=None, ds_B_transport=None):
    return _build_forms(
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
    assert str(args.latent_bounded_fields) == "alpha,phi"
    assert bool(args.predictor_corrector_startup)
    assert int(args.pc_p1_max_it) == 12
    assert int(args.pc_p2_max_it) == 12


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


def test_latent_pc_startup_uses_trust_region_globalization() -> None:
    args = SimpleNamespace(
        nonlinear_solver="pdas",
        latent_bounded_transport=True,
        logistic_bounded_transform=False,
        predictor_corrector_startup=True,
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
