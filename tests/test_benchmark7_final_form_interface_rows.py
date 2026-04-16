import sys
import numpy as np

from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import (
    _build_forms,
    _create_problem,
    _named_constant,
    _normalize_benchmark7_solver_choice,
    _parse_args,
    _primary_darcy_field_names,
    _tag_inactive_fields_above_alpha,
    _tag_inactive_fields_below_alpha,
)
from examples.utils.biofilm.final_form import _named_c, build_biofilm_one_domain_final_form
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.expressions import Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.utils.meshgen import structured_quad


def _build_final_form_problem(
    *,
    p_pore_shift: float = 0.0,
    phi_shift: float = 0.0,
    phi_prev_shift: float = 0.0,
    solid_velocity_shift: float = 0.0,
    u_shift: float = 0.0,
    pore_shear_shift: float = 0.0,
    bjs_coefficient: float = 0.0,
    solid_interface_weight: float = 1.0,
    mass_interface_weight: float | Constant = 1.0,
    normal_interface_weight: float | Constant = 1.0,
    disable_interface_physics: bool = False,
    final_form_constant_rho_s: bool = False,
    final_form_domain_lm: bool = False,
    final_form_domain_lm_aug_gamma: float = 0.0,
    final_form_mass_lm_aug_gamma: float = 0.0,
    final_form_normal_lm_aug_gamma: float = 0.0,
    solid_volumetric_split: bool = False,
    mu_mass_value: float = 0.0,
    mu_normal_value: float = 0.0,
):
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
        one_domain_formulation="final_form",
        final_form_phi_mode="transport",
        final_form_constant_rho_s=final_form_constant_rho_s,
        final_form_domain_lm=final_form_domain_lm,
        solid_volumetric_split=solid_volumetric_split,
    )

    problem["v_k"].set_values_from_function(lambda x, y: np.array([0.15 + 0.02 * x, -0.03 + 0.04 * y]))
    problem["v_n"].set_values_from_function(lambda x, y: np.array([0.10 + 0.01 * x, -0.02 + 0.03 * y]))
    problem["vP_k"].set_values_from_function(
        lambda x, y: np.array(
            [
                0.05 + (0.03 + float(pore_shear_shift)) * x,
                0.07 - (0.02 - 0.5 * float(pore_shear_shift)) * y,
            ]
        )
    )
    problem["vP_n"].set_values_from_function(
        lambda x, y: np.array(
            [
                0.02 + (0.02 + 0.5 * float(pore_shear_shift)) * x,
                0.05 - (0.01 - 0.25 * float(pore_shear_shift)) * y,
            ]
        )
    )
    problem["vS_k"].set_values_from_function(
        lambda x, y: np.array(
            [
                -0.01 + 0.01 * x + 0.5 * float(solid_velocity_shift) * x,
                0.04 + 0.02 * y + float(solid_velocity_shift) * (0.25 + y),
            ]
        )
    )
    problem["vS_n"].set_values_from_function(
        lambda x, y: np.array(
            [
                -0.02 + 0.01 * x + 0.25 * float(solid_velocity_shift) * x,
                0.03 + 0.01 * y + 0.5 * float(solid_velocity_shift) * (0.25 + y),
            ]
        )
    )
    problem["u_k"].set_values_from_function(
        lambda x, y: np.array([0.01 + 0.02 * x * y + float(u_shift) * x, -0.015 + 0.01 * y + float(u_shift) * y])
    )
    problem["u_n"].set_values_from_function(
        lambda x, y: np.array([0.008 + 0.015 * x * y + 0.5 * float(u_shift) * x, -0.010 + 0.006 * y + 0.5 * float(u_shift) * y])
    )
    problem["p_k"].set_values_from_function(lambda x, y: 0.3 + 0.1 * x - 0.05 * y)
    problem["p_n"].set_values_from_function(lambda x, y: 0.2 + 0.05 * x - 0.03 * y)
    problem["p_pore_k"].set_values_from_function(lambda x, y: 0.25 - 0.04 * x + 0.08 * y + float(p_pore_shift))
    problem["p_pore_n"].set_values_from_function(lambda x, y: 0.18 - 0.02 * x + 0.05 * y + float(p_pore_shift))
    problem["alpha_k"].set_values_from_function(lambda x, y: 0.45 + 0.10 * x + 0.08 * y)
    problem["alpha_n"].set_values_from_function(lambda x, y: 0.40 + 0.08 * x + 0.06 * y)
    problem["phi_k"].set_values_from_function(lambda x, y: 0.32 + 0.04 * x - 0.03 * y + float(phi_shift))
    problem["phi_n"].set_values_from_function(
        lambda x, y: 0.30 + 0.03 * x - 0.02 * y + float(phi_shift) + float(phi_prev_shift)
    )
    if problem.get("rho_s_k") is not None:
        problem["rho_s_k"].set_values_from_function(lambda x, y: 1.10 + 0.05 * x + 0.02 * y)
        problem["rho_s_n"].set_values_from_function(lambda x, y: 1.08 + 0.04 * x + 0.01 * y)
    if problem.get("pi_s_k") is not None:
        problem["pi_s_k"].set_values_from_function(lambda x, y: 0.02 + 0.01 * x - 0.015 * y)
        problem["pi_s_n"].set_values_from_function(lambda x, y: 0.01 + 0.005 * x - 0.010 * y)
    for name in (
        "mu_mass_k",
        "mu_mass_n",
        "mu_normal_k",
        "mu_normal_n",
        "mu_tangent_k",
        "mu_tangent_n",
        "mu_kin_k",
        "mu_kin_n",
    ):
        if problem.get(name) is not None:
            problem[name].nodal_values[:] = 0.0
    for name in (
        "lm_vf_k",
        "lm_vf_n",
        "lm_p_k",
        "lm_p_n",
        "lm_vP_k",
        "lm_vP_n",
        "lm_vS_k",
        "lm_vS_n",
        "lm_p_pore_k",
        "lm_p_pore_n",
        "lm_phi_k",
        "lm_phi_n",
        "lm_u_k",
        "lm_u_n",
    ):
        if problem.get(name) is not None:
            problem[name].nodal_values[:] = 0.0
    if problem.get("mu_mass_k") is not None and float(mu_mass_value) != 0.0:
        try:
            problem["mu_mass_k"].set_values_from_function(lambda x, y: float(mu_mass_value))
        except NotImplementedError:
            problem["mu_mass_k"].nodal_values[:] = float(mu_mass_value)
    if problem.get("mu_normal_k") is not None and float(mu_normal_value) != 0.0:
        try:
            problem["mu_normal_k"].set_values_from_function(lambda x, y: float(mu_normal_value))
        except NotImplementedError:
            problem["mu_normal_k"].nodal_values[:] = float(mu_normal_value)

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
        phi_b=0.30,
        M_alpha=1.0,
        gamma_alpha=1.0,
        eps_alpha=0.05,
        solid_visco_eta=0.0,
        gamma_div=0.0,
        solid_volumetric_split=solid_volumetric_split,
        solid_volumetric_penalty=1.0,
        gamma_u=0.0,
        u_extension_mode="l2",
        gamma_u_pin=0.0,
        u_cip=0.0,
        u_cip_weight="fluid",
        gamma_v=0.0,
        v_extension_mode="l2",
        gamma_v_pin=0.0,
        gamma_p=0.0,
        p_extension_mode="l2",
        gamma_p_pin=0.0,
        gamma_vP=0.0,
        vP_extension_mode="l2",
        gamma_vP_pin=0.0,
        gamma_p_pore=0.0,
        p_pore_extension_mode="l2",
        gamma_p_pore_pin=0.0,
        gamma_rho_s=0.0,
        rho_s_extension_mode="l2",
        gamma_rho_s_pin=0.0,
        final_form_constant_rho_s=final_form_constant_rho_s,
        final_form_domain_lm=final_form_domain_lm,
        final_form_domain_lm_aug_gamma=float(final_form_domain_lm_aug_gamma),
        final_form_mass_lm_aug_gamma=float(final_form_mass_lm_aug_gamma),
        final_form_normal_lm_aug_gamma=float(final_form_normal_lm_aug_gamma),
        vS_cip=0.0,
        gamma_vS=0.0,
        vS_extension_mode="l2",
        gamma_vS_pin=0.0,
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
        alpha_advection_form="conservative",
        solid_model="linear",
        kappa_inv_model="spatial",
        enable_phi_evolution=True,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        skeleton_inertia_convection="lagged",
        fluid_convection="off",
        support_physics="stored_support",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        full_ratio_free_state=False,
        split_primary_darcy_flux=False,
        split_pore_flux_model="exact_conservative_p",
        split_pore_momentum_model="band_alpha",
        pressure_interface_closure=False,
        p_pore_fluid_gauge=False,
        interface_entry_closure=False,
        interface_bjs_closure=bool(abs(float(bjs_coefficient)) > 0.0),
        interface_bjs_closure_strength=1.0,
        interface_bjs_gamma=float(bjs_coefficient),
        final_form_solid_interface_weight=float(solid_interface_weight),
        final_form_mass_interface_weight=mass_interface_weight,
        final_form_normal_interface_weight=normal_interface_weight,
        final_form_disable_interface_physics=bool(disable_interface_physics),
        interface_velocity_continuity_closure=False,
        interface_traction_continuity_closure=False,
    )
    return problem, forms


def _assemble_block(problem, form, field: str) -> np.ndarray:
    _, residual = assemble_form(
        Equation(None, form),
        dof_handler=problem["dh"],
        bcs=[],
        backend="python",
    )
    sl = np.asarray(problem["dh"].get_field_slice(field), dtype=int)
    return np.asarray(residual, dtype=float)[sl]


def test_benchmark7_final_form_deformable_pore_row_is_live() -> None:
    problem, forms = _build_final_form_problem(final_form_constant_rho_s=True)

    pore_block = _assemble_block(problem, forms.r_pore, "p_pore")

    assert np.linalg.norm(pore_block, ord=np.inf) > 1.0e-10


def test_benchmark7_final_form_constant_rho_s_phi_row_drops_previous_phi_split() -> None:
    problem_ref, forms_ref = _build_final_form_problem(
        final_form_constant_rho_s=True,
        phi_prev_shift=0.0,
    )
    problem_shift, forms_shift = _build_final_form_problem(
        final_form_constant_rho_s=True,
        phi_prev_shift=0.05,
    )

    phi_ref = _assemble_block(problem_ref, forms_ref.r_phi, "phi")
    phi_shift = _assemble_block(problem_shift, forms_shift.r_phi, "phi")

    assert np.linalg.norm(phi_shift - phi_ref, ord=np.inf) < 1.0e-12


def test_benchmark7_final_form_constant_rho_s_pore_row_uses_combined_porosity_flux() -> None:
    problem_ref, forms_ref = _build_final_form_problem(
        final_form_constant_rho_s=True,
        solid_velocity_shift=0.0,
    )
    problem_shift, forms_shift = _build_final_form_problem(
        final_form_constant_rho_s=True,
        solid_velocity_shift=0.2,
    )

    pore_ref = _assemble_block(problem_ref, forms_ref.r_pore, "p_pore")
    pore_shift = _assemble_block(problem_shift, forms_shift.r_pore, "p_pore")

    assert np.linalg.norm(pore_shift - pore_ref, ord=np.inf) > 1.0e-10


def test_benchmark7_final_form_constant_rho_s_pore_row_drops_previous_phi_split() -> None:
    problem_ref, forms_ref = _build_final_form_problem(
        final_form_constant_rho_s=True,
        phi_prev_shift=0.0,
    )
    problem_shift, forms_shift = _build_final_form_problem(
        final_form_constant_rho_s=True,
        phi_prev_shift=0.05,
    )

    pore_ref = _assemble_block(problem_ref, forms_ref.r_pore, "p_pore")
    pore_shift = _assemble_block(problem_shift, forms_shift.r_pore, "p_pore")

    assert np.linalg.norm(pore_shift - pore_ref, ord=np.inf) < 1.0e-12


def test_benchmark7_final_form_pore_bulk_row_responds_to_viscous_pore_shear() -> None:
    problem_ref, forms_ref = _build_final_form_problem(
        final_form_constant_rho_s=True,
        pore_shear_shift=0.0,
        p_pore_shift=0.0,
    )
    problem_shift, forms_shift = _build_final_form_problem(
        final_form_constant_rho_s=True,
        pore_shear_shift=0.12,
        p_pore_shift=0.0,
    )

    pore_bulk_x_ref = _assemble_block(problem_ref, forms_ref.r_momentum_terms["pore_bulk"], "vP_x")
    pore_bulk_x_shift = _assemble_block(problem_shift, forms_shift.r_momentum_terms["pore_bulk"], "vP_x")
    pore_bulk_y_ref = _assemble_block(problem_ref, forms_ref.r_momentum_terms["pore_bulk"], "vP_y")
    pore_bulk_y_shift = _assemble_block(problem_shift, forms_shift.r_momentum_terms["pore_bulk"], "vP_y")

    assert np.linalg.norm(pore_bulk_x_shift - pore_bulk_x_ref, ord=np.inf) > 1.0e-10
    assert np.linalg.norm(pore_bulk_y_shift - pore_bulk_y_ref, ord=np.inf) > 1.0e-10


def test_benchmark7_final_form_volumetric_split_builds_live_pi_row() -> None:
    problem, forms = _build_final_form_problem(
        final_form_constant_rho_s=True,
        solid_volumetric_split=True,
    )

    assert forms.r_volumetric is not None
    assert forms.a_volumetric is not None

    pi_block = _assemble_block(problem, forms.r_volumetric, "pi_s")

    assert np.linalg.norm(pi_block, ord=np.inf) > 1.0e-10


def test_benchmark7_final_form_zero_solid_interface_weight_drops_u_from_normal_row() -> None:
    problem_on_ref, forms_on_ref = _build_final_form_problem(
        final_form_constant_rho_s=True,
        solid_interface_weight=1.0,
        u_shift=0.0,
    )
    problem_on_shift, forms_on_shift = _build_final_form_problem(
        final_form_constant_rho_s=True,
        solid_interface_weight=1.0,
        u_shift=0.1,
    )
    problem_off_ref, forms_off_ref = _build_final_form_problem(
        final_form_constant_rho_s=True,
        solid_interface_weight=0.0,
        u_shift=0.0,
    )
    problem_off_shift, forms_off_shift = _build_final_form_problem(
        final_form_constant_rho_s=True,
        solid_interface_weight=0.0,
        u_shift=0.1,
    )

    mu_normal_on_ref = _assemble_block(
        problem_on_ref,
        forms_on_ref.r_momentum_terms["interface_normal_constraint"],
        "mu_normal",
    )
    mu_normal_on_shift = _assemble_block(
        problem_on_shift,
        forms_on_shift.r_momentum_terms["interface_normal_constraint"],
        "mu_normal",
    )
    mu_normal_off_ref = _assemble_block(
        problem_off_ref,
        forms_off_ref.r_momentum_terms["interface_normal_constraint"],
        "mu_normal",
    )
    mu_normal_off_shift = _assemble_block(
        problem_off_shift,
        forms_off_shift.r_momentum_terms["interface_normal_constraint"],
        "mu_normal",
    )

    assert np.linalg.norm(mu_normal_on_shift - mu_normal_on_ref, ord=np.inf) > 1.0e-10
    assert np.linalg.norm(mu_normal_off_shift - mu_normal_off_ref, ord=np.inf) < 1.0e-12


def test_benchmark7_final_form_zero_normal_interface_weight_drops_normal_row_response() -> None:
    problem_on_ref, forms_on_ref = _build_final_form_problem(
        final_form_constant_rho_s=True,
        p_pore_shift=0.0,
        normal_interface_weight=1.0,
        mu_normal_value=0.2,
    )
    problem_on_shift, forms_on_shift = _build_final_form_problem(
        final_form_constant_rho_s=True,
        p_pore_shift=0.1,
        normal_interface_weight=1.0,
        mu_normal_value=0.2,
    )
    problem_off_ref, forms_off_ref = _build_final_form_problem(
        final_form_constant_rho_s=True,
        p_pore_shift=0.0,
        normal_interface_weight=0.0,
        mu_normal_value=0.2,
    )
    problem_off_shift, forms_off_shift = _build_final_form_problem(
        final_form_constant_rho_s=True,
        p_pore_shift=0.1,
        normal_interface_weight=0.0,
        mu_normal_value=0.2,
    )

    mu_normal_on_ref = _assemble_block(
        problem_on_ref,
        forms_on_ref.r_momentum_terms["interface_normal_constraint"],
        "mu_normal",
    )
    mu_normal_on_shift = _assemble_block(
        problem_on_shift,
        forms_on_shift.r_momentum_terms["interface_normal_constraint"],
        "mu_normal",
    )
    mu_normal_off_ref = _assemble_block(
        problem_off_ref,
        forms_off_ref.r_momentum_terms["interface_normal_constraint"],
        "mu_normal",
    )
    mu_normal_off_shift = _assemble_block(
        problem_off_shift,
        forms_off_shift.r_momentum_terms["interface_normal_constraint"],
        "mu_normal",
    )

    assert np.linalg.norm(mu_normal_on_shift - mu_normal_on_ref, ord=np.inf) > 1.0e-10
    assert np.linalg.norm(mu_normal_off_shift - mu_normal_off_ref, ord=np.inf) < 1.0e-12

    bulk_on_ref_problem, bulk_on_ref_forms = _build_final_form_problem(
        final_form_constant_rho_s=True,
        normal_interface_weight=1.0,
        mu_normal_value=0.2,
    )
    bulk_on_shift_problem, bulk_on_shift_forms = _build_final_form_problem(
        final_form_constant_rho_s=True,
        normal_interface_weight=1.0,
        mu_normal_value=0.35,
    )
    bulk_off_ref_problem, bulk_off_ref_forms = _build_final_form_problem(
        final_form_constant_rho_s=True,
        normal_interface_weight=0.0,
        mu_normal_value=0.2,
    )
    bulk_off_shift_problem, bulk_off_shift_forms = _build_final_form_problem(
        final_form_constant_rho_s=True,
        normal_interface_weight=0.0,
        mu_normal_value=0.35,
    )

    p_pore_bulk_on_ref = _assemble_block(
        bulk_on_ref_problem,
        bulk_on_ref_forms.r_momentum_terms["interface_normal_bulk_coupling"],
        "p_pore",
    )
    p_pore_bulk_on_shift = _assemble_block(
        bulk_on_shift_problem,
        bulk_on_shift_forms.r_momentum_terms["interface_normal_bulk_coupling"],
        "p_pore",
    )
    p_pore_bulk_off_ref = _assemble_block(
        bulk_off_ref_problem,
        bulk_off_ref_forms.r_momentum_terms["interface_normal_bulk_coupling"],
        "p_pore",
    )
    p_pore_bulk_off_shift = _assemble_block(
        bulk_off_shift_problem,
        bulk_off_shift_forms.r_momentum_terms["interface_normal_bulk_coupling"],
        "p_pore",
    )

    assert np.linalg.norm(p_pore_bulk_on_ref, ord=np.inf) > 1.0e-10
    assert np.linalg.norm(p_pore_bulk_on_shift, ord=np.inf) > 1.0e-10
    assert np.linalg.norm(p_pore_bulk_off_ref, ord=np.inf) < 1.0e-12
    assert np.linalg.norm(p_pore_bulk_off_shift, ord=np.inf) < 1.0e-12


def test_benchmark7_final_form_zero_mass_interface_weight_drops_mass_row_response() -> None:
    problem_on_ref, forms_on_ref = _build_final_form_problem(
        final_form_constant_rho_s=True,
        phi_shift=0.0,
        mass_interface_weight=1.0,
    )
    problem_on_shift, forms_on_shift = _build_final_form_problem(
        final_form_constant_rho_s=True,
        phi_shift=0.1,
        mass_interface_weight=1.0,
    )
    problem_off_ref, forms_off_ref = _build_final_form_problem(
        final_form_constant_rho_s=True,
        phi_shift=0.0,
        mass_interface_weight=0.0,
    )
    problem_off_shift, forms_off_shift = _build_final_form_problem(
        final_form_constant_rho_s=True,
        phi_shift=0.1,
        mass_interface_weight=0.0,
    )

    mu_mass_on_ref = _assemble_block(
        problem_on_ref,
        forms_on_ref.r_momentum_terms["interface_mass_constraint"],
        "mu_mass",
    )
    mu_mass_on_shift = _assemble_block(
        problem_on_shift,
        forms_on_shift.r_momentum_terms["interface_mass_constraint"],
        "mu_mass",
    )
    mu_mass_off_ref = _assemble_block(
        problem_off_ref,
        forms_off_ref.r_momentum_terms["interface_mass_constraint"],
        "mu_mass",
    )
    mu_mass_off_shift = _assemble_block(
        problem_off_shift,
        forms_off_shift.r_momentum_terms["interface_mass_constraint"],
        "mu_mass",
    )

    assert np.linalg.norm(mu_mass_on_shift - mu_mass_on_ref, ord=np.inf) > 1.0e-10
    assert np.linalg.norm(mu_mass_off_shift - mu_mass_off_ref, ord=np.inf) < 1.0e-12

    bulk_on_ref_problem, bulk_on_ref_forms = _build_final_form_problem(
        final_form_constant_rho_s=True,
        mass_interface_weight=1.0,
        mu_mass_value=0.2,
    )
    bulk_on_shift_problem, bulk_on_shift_forms = _build_final_form_problem(
        final_form_constant_rho_s=True,
        mass_interface_weight=1.0,
        mu_mass_value=0.35,
    )
    bulk_off_ref_problem, bulk_off_ref_forms = _build_final_form_problem(
        final_form_constant_rho_s=True,
        mass_interface_weight=0.0,
        mu_mass_value=0.2,
    )
    bulk_off_shift_problem, bulk_off_shift_forms = _build_final_form_problem(
        final_form_constant_rho_s=True,
        mass_interface_weight=0.0,
        mu_mass_value=0.35,
    )

    vP_bulk_on_ref = _assemble_block(
        bulk_on_ref_problem,
        bulk_on_ref_forms.r_momentum_terms["interface_mass_bulk_coupling"],
        "vP_y",
    )
    vP_bulk_on_shift = _assemble_block(
        bulk_on_shift_problem,
        bulk_on_shift_forms.r_momentum_terms["interface_mass_bulk_coupling"],
        "vP_y",
    )
    vP_bulk_off_ref = _assemble_block(
        bulk_off_ref_problem,
        bulk_off_ref_forms.r_momentum_terms["interface_mass_bulk_coupling"],
        "vP_y",
    )
    vP_bulk_off_shift = _assemble_block(
        bulk_off_shift_problem,
        bulk_off_shift_forms.r_momentum_terms["interface_mass_bulk_coupling"],
        "vP_y",
    )

    assert np.linalg.norm(vP_bulk_on_ref, ord=np.inf) > 1.0e-10
    assert np.linalg.norm(vP_bulk_on_shift, ord=np.inf) > 1.0e-10
    assert np.linalg.norm(vP_bulk_off_ref, ord=np.inf) < 1.0e-12
    assert np.linalg.norm(vP_bulk_off_shift, ord=np.inf) < 1.0e-12


def test_benchmark7_final_form_disable_interface_physics_pins_mu_normal() -> None:
    problem_ref, forms_ref = _build_final_form_problem(
        final_form_constant_rho_s=True,
        disable_interface_physics=True,
        p_pore_shift=0.0,
    )
    problem_shift, forms_shift = _build_final_form_problem(
        final_form_constant_rho_s=True,
        disable_interface_physics=True,
        p_pore_shift=0.2,
        u_shift=0.1,
    )

    mu_normal_ref = _assemble_block(
        problem_ref,
        forms_ref.r_momentum_terms["interface_normal_constraint"],
        "mu_normal",
    )
    mu_normal_shift = _assemble_block(
        problem_shift,
        forms_shift.r_momentum_terms["interface_normal_constraint"],
        "mu_normal",
    )

    assert np.linalg.norm(mu_normal_shift - mu_normal_ref, ord=np.inf) < 1.0e-12


def test_benchmark7_final_form_domain_lm_rows_track_wrong_domain_fields() -> None:
    problem_ref, forms_ref = _build_final_form_problem(
        final_form_constant_rho_s=True,
        final_form_domain_lm=True,
    )
    problem_shift, forms_shift = _build_final_form_problem(
        final_form_constant_rho_s=True,
        final_form_domain_lm=True,
        p_pore_shift=0.5,
        phi_shift=0.12,
    )
    problem_shift["v_k"].set_values_from_function(lambda x, y: np.array([0.32 + 0.03 * x, -0.08 + 0.02 * y]))

    vf_row_ref = _assemble_block(problem_ref, forms_ref.r_domain_lm_terms["support_kill_vf_row"], "lm_vf_x")
    vf_row_shift = _assemble_block(problem_shift, forms_shift.r_domain_lm_terms["support_kill_vf_row"], "lm_vf_x")
    pp_row_ref = _assemble_block(problem_ref, forms_ref.r_domain_lm_terms["free_kill_p_pore_row"], "lm_p_pore")
    pp_row_shift = _assemble_block(problem_shift, forms_shift.r_domain_lm_terms["free_kill_p_pore_row"], "lm_p_pore")
    phi_row_ref = _assemble_block(problem_ref, forms_ref.r_domain_lm_terms["free_kill_phi_row"], "lm_phi")
    phi_row_shift = _assemble_block(problem_shift, forms_shift.r_domain_lm_terms["free_kill_phi_row"], "lm_phi")

    assert np.linalg.norm(vf_row_shift - vf_row_ref, ord=np.inf) > 1.0e-10
    assert np.linalg.norm(pp_row_shift - pp_row_ref, ord=np.inf) > 1.0e-10
    assert np.linalg.norm(phi_row_shift - phi_row_ref, ord=np.inf) > 1.0e-10


def test_benchmark7_final_form_domain_lm_bulk_transpose_hits_primary_rows() -> None:
    problem_ref, forms_ref = _build_final_form_problem(
        final_form_constant_rho_s=True,
        final_form_domain_lm=True,
    )
    problem_shift, forms_shift = _build_final_form_problem(
        final_form_constant_rho_s=True,
        final_form_domain_lm=True,
    )
    problem_shift["lm_vf_k"].nodal_values[:] = 0.17
    problem_shift["lm_p_pore_k"].nodal_values[:] = 0.23
    problem_shift["lm_u_k"].nodal_values[:] = -0.08

    vf_bulk_ref = _assemble_block(problem_ref, forms_ref.r_domain_lm_terms["support_kill_vf_bulk"], "v_x")
    vf_bulk_shift = _assemble_block(problem_shift, forms_shift.r_domain_lm_terms["support_kill_vf_bulk"], "v_x")
    pp_bulk_ref = _assemble_block(problem_ref, forms_ref.r_domain_lm_terms["free_kill_p_pore_bulk"], "p_pore")
    pp_bulk_shift = _assemble_block(problem_shift, forms_shift.r_domain_lm_terms["free_kill_p_pore_bulk"], "p_pore")
    u_bulk_ref = _assemble_block(problem_ref, forms_ref.r_domain_lm_terms["free_kill_u_bulk"], "u_x")
    u_bulk_shift = _assemble_block(problem_shift, forms_shift.r_domain_lm_terms["free_kill_u_bulk"], "u_x")

    assert np.linalg.norm(vf_bulk_shift - vf_bulk_ref, ord=np.inf) > 1.0e-10
    assert np.linalg.norm(pp_bulk_shift - pp_bulk_ref, ord=np.inf) > 1.0e-10
    assert np.linalg.norm(u_bulk_shift - u_bulk_ref, ord=np.inf) > 1.0e-10


def test_benchmark7_final_form_domain_lm_fields_use_dg0() -> None:
    problem, _ = _build_final_form_problem(
        final_form_constant_rho_s=True,
        final_form_domain_lm=True,
    )
    families = problem["me"].get_field_families()
    orders = problem["me"].get_field_orders()

    for name in (
        "lm_vf_x",
        "lm_vf_y",
        "lm_p",
        "lm_vP_x",
        "lm_vP_y",
        "lm_vS_x",
        "lm_vS_y",
        "lm_p_pore",
        "lm_phi",
        "lm_u_x",
        "lm_u_y",
    ):
        assert families[name] == "DG"
        assert orders[name] == 0


def test_benchmark7_final_form_interface_multiplier_fields_use_dg0() -> None:
    problem, _ = _build_final_form_problem(
        final_form_constant_rho_s=True,
    )
    families = problem["me"].get_field_families()
    orders = problem["me"].get_field_orders()

    for name in (
        "mu_mass",
        "mu_normal",
        "mu_tangent",
        "mu_kin_x",
        "mu_kin_y",
    ):
        assert families[name] == "DG"
        assert orders[name] == 0


def test_benchmark7_final_form_domain_lm_off_domain_rows_go_null() -> None:
    problem, forms = _build_final_form_problem(
        final_form_constant_rho_s=True,
        final_form_domain_lm=True,
    )
    for key in ("alpha_k", "alpha_n"):
        problem[key].set_values_from_function(lambda x, y: 0.0)
    problem["lm_vf_k"].nodal_values[:] = 0.0
    problem["v_k"].set_values_from_function(lambda x, y: np.array([0.15 + 0.02 * x, -0.03 + 0.04 * y]))
    row_ref = _assemble_block(problem, forms.r_domain_lm_terms["support_kill_vf_row"], "lm_vf_x")

    problem["v_k"].set_values_from_function(lambda x, y: np.array([0.75 - 0.10 * x, 0.25 + 0.05 * y]))
    row_shift_v = _assemble_block(problem, forms.r_domain_lm_terms["support_kill_vf_row"], "lm_vf_x")

    problem["lm_vf_k"].nodal_values[:] = 0.31
    row_shift_lm = _assemble_block(problem, forms.r_domain_lm_terms["support_kill_vf_row"], "lm_vf_x")

    assert np.linalg.norm(row_ref, ord=np.inf) < 1.0e-12
    np.testing.assert_allclose(row_shift_v, row_ref, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(row_shift_lm, row_ref, rtol=1.0e-12, atol=1.0e-12)


def test_benchmark7_final_form_domain_lm_augmented_term_hits_primal_target_row() -> None:
    problem_zero, forms_zero = _build_final_form_problem(
        final_form_constant_rho_s=True,
        final_form_domain_lm=True,
        final_form_domain_lm_aug_gamma=0.0,
    )
    problem_aug, forms_aug = _build_final_form_problem(
        final_form_constant_rho_s=True,
        final_form_domain_lm=True,
        final_form_domain_lm_aug_gamma=1.0,
    )

    vf_aug_zero = _assemble_block(problem_zero, forms_zero.r_domain_lm_terms["support_kill_vf_aug"], "v_x")
    vf_aug_live = _assemble_block(problem_aug, forms_aug.r_domain_lm_terms["support_kill_vf_aug"], "v_x")

    assert np.linalg.norm(vf_aug_zero, ord=np.inf) <= 1.0e-12
    assert np.linalg.norm(vf_aug_live, ord=np.inf) > 1.0e-10


def test_benchmark7_final_form_normal_lm_augmented_term_hits_primal_rows() -> None:
    problem_zero, forms_zero = _build_final_form_problem(
        final_form_constant_rho_s=True,
        final_form_normal_lm_aug_gamma=0.0,
    )
    problem_aug, forms_aug = _build_final_form_problem(
        final_form_constant_rho_s=True,
        final_form_normal_lm_aug_gamma=1.0,
    )

    vf_aug_zero = _assemble_block(problem_zero, forms_zero.r_momentum_terms["interface_normal_aug_bulk"], "v_x")
    vf_aug_live = _assemble_block(problem_aug, forms_aug.r_momentum_terms["interface_normal_aug_bulk"], "v_x")
    pp_aug_zero = _assemble_block(problem_zero, forms_zero.r_momentum_terms["interface_normal_aug_bulk"], "p_pore")
    pp_aug_live = _assemble_block(problem_aug, forms_aug.r_momentum_terms["interface_normal_aug_bulk"], "p_pore")

    assert np.linalg.norm(vf_aug_zero, ord=np.inf) <= 1.0e-12
    assert np.linalg.norm(pp_aug_zero, ord=np.inf) <= 1.0e-12
    assert np.linalg.norm(vf_aug_live, ord=np.inf) > 1.0e-10
    assert np.linalg.norm(pp_aug_live, ord=np.inf) > 1.0e-10


def test_benchmark7_final_form_mass_lm_augmented_term_hits_primal_rows() -> None:
    problem_zero, forms_zero = _build_final_form_problem(
        final_form_constant_rho_s=True,
        final_form_mass_lm_aug_gamma=0.0,
    )
    problem_aug, forms_aug = _build_final_form_problem(
        final_form_constant_rho_s=True,
        final_form_mass_lm_aug_gamma=1.0,
    )

    vf_aug_zero = _assemble_block(problem_zero, forms_zero.r_momentum_terms["interface_mass_aug_bulk"], "v_x")
    vf_aug_live = _assemble_block(problem_aug, forms_aug.r_momentum_terms["interface_mass_aug_bulk"], "v_x")
    alpha_aug_zero = _assemble_block(problem_zero, forms_zero.r_momentum_terms["interface_mass_aug_bulk"], "alpha")
    alpha_aug_live = _assemble_block(problem_aug, forms_aug.r_momentum_terms["interface_mass_aug_bulk"], "alpha")

    assert np.linalg.norm(vf_aug_zero, ord=np.inf) <= 1.0e-12
    assert np.linalg.norm(alpha_aug_zero, ord=np.inf) <= 1.0e-12
    assert np.linalg.norm(vf_aug_live, ord=np.inf) > 1.0e-10
    assert np.linalg.norm(alpha_aug_live, ord=np.inf) > 1.0e-10


def test_benchmark7_final_form_domain_lm_fields_join_inactive_domain_tags() -> None:
    problem_low, _ = _build_final_form_problem(
        final_form_constant_rho_s=True,
        final_form_domain_lm=True,
    )
    problem_low["alpha_k"].nodal_values[:] = 0.0
    porous_fields: list[str] = []
    if problem_low.get("p_pore_k") is not None:
        porous_fields.append("p_pore")
    porous_fields.extend(_primary_darcy_field_names(problem_low))
    for name in (
        "phi",
        "rho_s",
        "vS_x",
        "vS_y",
        "u_x",
        "u_y",
        "lm_vf_x",
        "lm_vf_y",
        "lm_p",
        "mu_mass",
        "mu_normal",
        "mu_tangent",
        "mu_kin_x",
        "mu_kin_y",
    ):
        if name in getattr(problem_low["dh"], "field_names", ()):
            porous_fields.append(name)
    low_counts = _tag_inactive_fields_below_alpha(problem_low, alpha_threshold=0.05, field_names=porous_fields)

    assert low_counts.get("lm_vf_x", 0) > 0
    assert low_counts.get("lm_vf_y", 0) > 0
    assert low_counts.get("lm_p", 0) > 0
    assert low_counts.get("mu_normal", 0) > 0
    assert low_counts.get("mu_mass", 0) > 0

    problem_high, _ = _build_final_form_problem(
        final_form_constant_rho_s=True,
        final_form_domain_lm=True,
    )
    problem_high["alpha_k"].nodal_values[:] = 1.0
    fluid_fields = ["v_x", "v_y", "p"]
    for name in (
        "lm_vP_x",
        "lm_vP_y",
        "lm_vS_x",
        "lm_vS_y",
        "lm_p_pore",
        "lm_phi",
        "lm_u_x",
        "lm_u_y",
        "mu_mass",
        "mu_normal",
        "mu_tangent",
        "mu_kin_x",
        "mu_kin_y",
    ):
        if name in getattr(problem_high["dh"], "field_names", ()):
            fluid_fields.append(name)
    high_counts = _tag_inactive_fields_above_alpha(problem_high, alpha_threshold=0.95, field_names=fluid_fields)

    assert high_counts.get("lm_vP_x", 0) > 0
    assert high_counts.get("lm_vS_x", 0) > 0
    assert high_counts.get("lm_p_pore", 0) > 0
    assert high_counts.get("lm_phi", 0) > 0
    assert high_counts.get("lm_u_x", 0) > 0
    assert high_counts.get("mu_normal", 0) > 0
    assert high_counts.get("mu_mass", 0) > 0


def test_benchmark7_final_form_keeps_predictor_corrector_startup_enabled() -> None:
    saved_argv = list(sys.argv)
    try:
        sys.argv = [
            "pytest",
            "--one-domain-formulation",
            "final_form",
            "--final-form-constant-rho-s",
        ]
        args = _parse_args()
    finally:
        sys.argv = saved_argv

    args = _normalize_benchmark7_solver_choice(args)

    assert bool(args.predictor_corrector_startup)


def test_benchmark7_final_form_domain_lm_disables_predictor_corrector_startup() -> None:
    saved_argv = list(sys.argv)
    try:
        sys.argv = [
            "pytest",
            "--one-domain-formulation",
            "final_form",
            "--final-form-constant-rho-s",
            "--final-form-domain-lm",
        ]
        args = _parse_args()
    finally:
        sys.argv = saved_argv

    args = _normalize_benchmark7_solver_choice(args)

    assert not bool(args.predictor_corrector_startup)


def test_benchmark7_named_constant_reuses_cached_object() -> None:
    c0 = _named_constant("b7_cache_probe", 3.25)
    c1 = _named_constant("b7_cache_probe", 3.25)

    assert c0 is c1


def test_final_form_named_constant_reuses_cached_object() -> None:
    c0 = _named_c("ff_cache_probe", (1.0, -2.0), dim=1)
    c1 = _named_c("ff_cache_probe", (1.0, -2.0), dim=1)

    assert c0 is c1


def test_final_form_normal_interface_weight_constant_is_reused_by_problem_build() -> None:
    weight_c = Constant(1.0)
    problem, forms = _build_final_form_problem(
        final_form_constant_rho_s=True,
        normal_interface_weight=weight_c,
        mu_normal_value=0.2,
    )
    assert problem["final_form_normal_interface_weight_c"] is weight_c
    mu_normal_on = _assemble_block(
        problem,
        forms.r_momentum_terms["interface_normal_constraint"],
        "mu_normal",
    )
    assert np.linalg.norm(mu_normal_on, ord=np.inf) > 1.0e-10


def _build_rigid_final_form_problem(*, p_pore_shift: float = 0.0):
    nodes, elems, _, corners = structured_quad(
        1.0,
        1.0,
        nx=1,
        ny=1,
        poly_order=1,
        offset=(0.0, 0.0),
    )
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)
    me = MixedElement(
        mesh,
        field_specs={
            "ux": 1,
            "uy": 1,
            "p": 1,
            "p_pore": 1,
            "vpx": 1,
            "vpy": 1,
            "vSx": 1,
            "vSy": 1,
            "dx": 1,
            "dy": 1,
            "alpha": 1,
            "phi": 1,
            "rho_s": 1,
            "mu_mass": 1,
            "mu_normal": 1,
            "mu_tangent": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("velocity", ["ux", "uy"], dim=1)
    VP = FunctionSpace("pore_velocity", ["vpx", "vpy"], dim=1)
    VS = FunctionSpace("solid_velocity", ["vSx", "vSy"], dim=1)
    U = FunctionSpace("solid_disp", ["dx", "dy"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    v_test = VectorTestFunction(space=V, dof_handler=dh)
    dvP = VectorTrialFunction(space=VP, dof_handler=dh)
    vP_test = VectorTestFunction(space=VP, dof_handler=dh)
    dvS = VectorTrialFunction(space=VS, dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)

    dp = TrialFunction("p", dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    dp_pore = TrialFunction("p_pore", dof_handler=dh)
    q_pore_test = TestFunction("p_pore", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    drho_s = TrialFunction("rho_s", dof_handler=dh)
    rho_s_test = TestFunction("rho_s", dof_handler=dh)
    dmu_mass = TrialFunction("mu_mass", dof_handler=dh)
    mu_mass_test = TestFunction("mu_mass", dof_handler=dh)
    dmu_normal = TrialFunction("mu_normal", dof_handler=dh)
    mu_normal_test = TestFunction("mu_normal", dof_handler=dh)
    dmu_tangent = TrialFunction("mu_tangent", dof_handler=dh)
    mu_tangent_test = TestFunction("mu_tangent", dof_handler=dh)

    v_k = VectorFunction("v_k", ["ux", "uy"], dof_handler=dh)
    v_n = VectorFunction("v_n", ["ux", "uy"], dof_handler=dh)
    vP_k = VectorFunction("vP_k", ["vpx", "vpy"], dof_handler=dh)
    vP_n = VectorFunction("vP_n", ["vpx", "vpy"], dof_handler=dh)
    vS_k = VectorFunction("vS_k", ["vSx", "vSy"], dof_handler=dh)
    vS_n = VectorFunction("vS_n", ["vSx", "vSy"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["dx", "dy"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["dx", "dy"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    p_pore_k = Function("p_pore_k", "p_pore", dof_handler=dh)
    p_pore_n = Function("p_pore_n", "p_pore", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    rho_s_k = Function("rho_s_k", "rho_s", dof_handler=dh)
    rho_s_n = Function("rho_s_n", "rho_s", dof_handler=dh)
    mu_mass_k = Function("mu_mass_k", "mu_mass", dof_handler=dh)
    mu_mass_n = Function("mu_mass_n", "mu_mass", dof_handler=dh)
    mu_normal_k = Function("mu_normal_k", "mu_normal", dof_handler=dh)
    mu_normal_n = Function("mu_normal_n", "mu_normal", dof_handler=dh)
    mu_tangent_k = Function("mu_tangent_k", "mu_tangent", dof_handler=dh)
    mu_tangent_n = Function("mu_tangent_n", "mu_tangent", dof_handler=dh)

    v_k.set_values_from_function(lambda x, y: np.array([0.12 + 0.03 * x, -0.04 + 0.05 * y]))
    v_n.set_values_from_function(lambda x, y: np.array([0.10 + 0.02 * x, -0.03 + 0.04 * y]))
    vP_k.set_values_from_function(lambda x, y: np.array([0.07 - 0.01 * x, 0.05 + 0.02 * y]))
    vP_n.set_values_from_function(lambda x, y: np.array([0.04 - 0.01 * x, 0.03 + 0.01 * y]))
    p_k.set_values_from_function(lambda x, y: 0.35 + 0.08 * x - 0.06 * y)
    p_n.set_values_from_function(lambda x, y: 0.30 + 0.05 * x - 0.04 * y)
    p_pore_k.set_values_from_function(lambda x, y: 0.18 - 0.02 * x + 0.09 * y + float(p_pore_shift))
    p_pore_n.set_values_from_function(lambda x, y: 0.14 - 0.01 * x + 0.06 * y + float(p_pore_shift))
    alpha_k.set_values_from_function(lambda x, y: 0.25 + 0.30 * x + 0.20 * y)
    alpha_n.set_values_from_function(lambda x, y: 0.20 + 0.25 * x + 0.15 * y)
    phi_k.nodal_values.fill(0.6)
    phi_n.nodal_values.fill(0.6)
    rho_s_k.nodal_values.fill(1.0)
    rho_s_n.nodal_values.fill(1.0)
    vS_k.nodal_values.fill(0.0)
    vS_n.nodal_values.fill(0.0)
    u_k.nodal_values.fill(0.0)
    u_n.nodal_values.fill(0.0)
    for name in (mu_mass_k, mu_mass_n, mu_normal_k, mu_normal_n, mu_tangent_k, mu_tangent_n):
        name.nodal_values.fill(0.0)

    forms = build_biofilm_one_domain_final_form(
        v_k=v_k,
        p_k=p_k,
        vP_k=vP_k,
        p_pore_k=p_pore_k,
        vS_k=vS_k,
        u_k=u_k,
        alpha_k=alpha_k,
        phi_k=phi_k,
        rho_s_k=rho_s_k,
        v_n=v_n,
        p_n=p_n,
        vP_n=vP_n,
        p_pore_n=p_pore_n,
        vS_n=vS_n,
        u_n=u_n,
        alpha_n=alpha_n,
        phi_n=phi_n,
        rho_s_n=rho_s_n,
        dv=dv,
        dp=dp,
        dvP=dvP,
        dp_pore=dp_pore,
        dvS=dvS,
        du=du,
        dalpha=dalpha,
        dphi=dphi,
        drho_s=drho_s,
        dmu_mass=dmu_mass,
        dmu_normal=dmu_normal,
        dmu_tangent=dmu_tangent,
        v_test=v_test,
        q_test=q_test,
        q_pore_test=q_pore_test,
        vP_test=vP_test,
        vS_test=vS_test,
        u_test=u_test,
        alpha_test=alpha_test,
        phi_test=phi_test,
        rho_s_test=rho_s_test,
        mu_mass_test=mu_mass_test,
        mu_normal_test=mu_normal_test,
        mu_tangent_test=mu_tangent_test,
        mu_mass_k=mu_mass_k,
        mu_normal_k=mu_normal_k,
        mu_tangent_k=mu_tangent_k,
        dx=dx(),
        dt=Constant(1.0e6),
        rho_f=Constant(1.0),
        mu_f=Constant(0.05),
        kappa_inv=Constant(2.5),
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        solid_visco_eta=0.0,
        gamma_phi=0.0,
        gamma_v=0.0,
        v_extension_mode="l2",
        gamma_v_pin=0.0,
        gamma_p=0.0,
        p_extension_mode="l2",
        gamma_p_pin=0.0,
        gamma_vP=0.0,
        vP_extension_mode="l2",
        gamma_vP_pin=0.0,
        gamma_p_pore=0.0,
        p_pore_extension_mode="l2",
        gamma_p_pore_pin=0.0,
        gamma_rho_s=0.0,
        rho_s_extension_mode="l2",
        gamma_rho_s_pin=0.0,
        rho_s_ref=1.0,
        constant_rho_s=True,
        gamma_u=0.0,
        u_extension_mode="l2",
        gamma_u_pin=0.0,
        gamma_vS=0.0,
        vS_extension_mode="l2",
        gamma_vS_pin=0.0,
        fluid_convection="off",
        pore_convection="off",
        skeleton_inertia_convection="off",
        solid_model="linear",
        phi_mode="alpha_closure",
        phi_b=0.6,
        normal_pressure_scale=9.81,
        normal_constraint_carrier="multiplier",
        rigid_darcy_head_mode=True,
        bjs_coefficient=0.0,
        interface_formulation="decomposed",
    )
    return {"dh": dh}, forms


def test_final_form_interface_constraints_live_only_on_interface_multiplier_rows() -> None:
    problem, forms = _build_final_form_problem()

    mass_if = forms.r_momentum_terms["interface_mass_constraint"]
    normal_if = forms.r_momentum_terms["interface_normal_constraint"]
    tangential_if = forms.r_momentum_terms["interface_tangential_constraint"]
    kinematic_if = forms.r_kinematics_terms["interface_kinematic_constraint"]

    assert np.linalg.norm(_assemble_block(problem, mass_if, "mu_mass"), ord=np.inf) > 1.0e-10
    assert np.linalg.norm(_assemble_block(problem, normal_if, "mu_normal"), ord=np.inf) > 1.0e-10
    assert np.linalg.norm(_assemble_block(problem, tangential_if, "mu_tangent"), ord=np.inf) > 1.0e-10
    assert np.linalg.norm(_assemble_block(problem, kinematic_if, "mu_kin_x"), ord=np.inf) > 1.0e-10
    assert np.linalg.norm(_assemble_block(problem, kinematic_if, "mu_kin_y"), ord=np.inf) > 1.0e-10

    for field in ("v_x", "v_y", "phi", "rho_s"):
        assert np.linalg.norm(_assemble_block(problem, mass_if, field), ord=np.inf) < 1.0e-12
    for field in ("v_x", "v_y", "p", "p_pore", "u_x", "u_y"):
        assert np.linalg.norm(_assemble_block(problem, normal_if, field), ord=np.inf) < 1.0e-12
    for field in ("v_x", "v_y", "u_x", "u_y", "vS_x", "vS_y"):
        assert np.linalg.norm(_assemble_block(problem, tangential_if, field), ord=np.inf) < 1.0e-12
    for field in ("u_x", "u_y", "vS_x", "vS_y", "alpha"):
        assert np.linalg.norm(_assemble_block(problem, kinematic_if, field), ord=np.inf) < 1.0e-12


def test_final_form_free_fluid_bulk_rows_are_invariant_to_interface_pressure_shift_when_multipliers_zero() -> None:
    problem_a, forms_a = _build_final_form_problem()
    problem_b, forms_b = _build_final_form_problem(p_pore_shift=0.75, phi_shift=0.12)

    res_ax = _assemble_block(problem_a, forms_a.r_momentum, "v_x")
    res_ay = _assemble_block(problem_a, forms_a.r_momentum, "v_y")
    res_bx = _assemble_block(problem_b, forms_b.r_momentum, "v_x")
    res_by = _assemble_block(problem_b, forms_b.r_momentum, "v_y")

    np.testing.assert_allclose(res_ax, res_bx, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(res_ay, res_by, rtol=1.0e-12, atol=1.0e-12)


def test_final_form_interface_tangential_constraint_depends_on_pore_shear_with_brinkman_stress() -> None:
    problem_ref, forms_ref = _build_final_form_problem()
    problem_shift, forms_shift = _build_final_form_problem(pore_shear_shift=0.12)

    tangential_if_ref = forms_ref.r_momentum_terms["interface_tangential_constraint"]
    tangential_if_shift = forms_shift.r_momentum_terms["interface_tangential_constraint"]

    mu_t_ref = _assemble_block(problem_ref, tangential_if_ref, "mu_tangent")
    mu_t_shift = _assemble_block(problem_shift, tangential_if_shift, "mu_tangent")

    assert np.linalg.norm(mu_t_shift - mu_t_ref, ord=np.inf) > 1.0e-10


def test_final_form_bjs_tangential_constraint_depends_on_pore_slip() -> None:
    problem_ref, forms_ref = _build_final_form_problem(bjs_coefficient=1.0)
    problem_shift, forms_shift = _build_final_form_problem(
        bjs_coefficient=1.0,
        pore_shear_shift=0.12,
    )

    tangential_if_ref = forms_ref.r_momentum_terms["interface_tangential_constraint"]
    tangential_if_shift = forms_shift.r_momentum_terms["interface_tangential_constraint"]

    mu_t_ref = _assemble_block(problem_ref, tangential_if_ref, "mu_tangent")
    mu_t_shift = _assemble_block(problem_shift, tangential_if_shift, "mu_tangent")

    assert np.linalg.norm(mu_t_shift - mu_t_ref, ord=np.inf) > 1.0e-10


def test_final_form_bjs_tangential_constraint_drops_pore_pressure_dependence() -> None:
    problem_ref, forms_ref = _build_final_form_problem(bjs_coefficient=1.0)
    problem_shift, forms_shift = _build_final_form_problem(
        bjs_coefficient=1.0,
        p_pore_shift=0.75,
    )

    tangential_if_ref = forms_ref.r_momentum_terms["interface_tangential_constraint"]
    tangential_if_shift = forms_shift.r_momentum_terms["interface_tangential_constraint"]

    mu_t_ref = _assemble_block(problem_ref, tangential_if_ref, "mu_tangent")
    mu_t_shift = _assemble_block(problem_shift, tangential_if_shift, "mu_tangent")

    np.testing.assert_allclose(mu_t_ref, mu_t_shift, rtol=1.0e-12, atol=1.0e-12)


def test_rigid_final_form_interface_constraints_live_only_on_interface_multiplier_rows() -> None:
    problem, forms = _build_rigid_final_form_problem()

    mass_if = forms.r_momentum_terms["interface_mass_constraint"]
    normal_if = forms.r_momentum_terms["interface_normal_constraint"]
    tangential_if = forms.r_momentum_terms["interface_tangential_constraint"]

    assert np.linalg.norm(_assemble_block(problem, mass_if, "mu_mass"), ord=np.inf) > 1.0e-10
    assert np.linalg.norm(_assemble_block(problem, normal_if, "mu_normal"), ord=np.inf) > 1.0e-10
    assert np.linalg.norm(_assemble_block(problem, tangential_if, "mu_tangent"), ord=np.inf) > 1.0e-10

    for field in ("ux", "uy", "vpx", "vpy", "p", "p_pore"):
        assert np.linalg.norm(_assemble_block(problem, mass_if, field), ord=np.inf) < 1.0e-12
        assert np.linalg.norm(_assemble_block(problem, normal_if, field), ord=np.inf) < 1.0e-12
        assert np.linalg.norm(_assemble_block(problem, tangential_if, field), ord=np.inf) < 1.0e-12


def test_rigid_final_form_free_fluid_rows_do_not_pick_up_pore_pressure_interface_shift_when_multipliers_zero() -> None:
    problem_a, forms_a = _build_rigid_final_form_problem()
    problem_b, forms_b = _build_rigid_final_form_problem(p_pore_shift=0.75)

    res_ax = _assemble_block(problem_a, forms_a.r_momentum, "ux")
    res_ay = _assemble_block(problem_a, forms_a.r_momentum, "uy")
    res_bx = _assemble_block(problem_b, forms_b.r_momentum, "ux")
    res_by = _assemble_block(problem_b, forms_b.r_momentum, "uy")

    np.testing.assert_allclose(res_ax, res_bx, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(res_ay, res_by, rtol=1.0e-12, atol=1.0e-12)
