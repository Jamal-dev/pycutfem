import numpy as np
import pytest

from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import _build_forms, _create_problem
from pycutfem.jit import _form_integrals, _integral_fusion_key, _plan_integral_execution_units
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Constant
from pycutfem.ufl.forms import Equation, assemble_form


def _build_final_form_problem(*, final_form_implementation: str):
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
        final_form_implementation=final_form_implementation,
    )

    problem["v_k"].set_values_from_function(lambda x, y: np.array([0.15 + 0.02 * x, -0.03 + 0.04 * y]))
    problem["v_n"].set_values_from_function(lambda x, y: np.array([0.10 + 0.01 * x, -0.02 + 0.03 * y]))
    problem["vP_k"].set_values_from_function(lambda x, y: np.array([0.05 + 0.03 * x, 0.07 - 0.02 * y]))
    problem["vP_n"].set_values_from_function(lambda x, y: np.array([0.02 + 0.02 * x, 0.05 - 0.01 * y]))
    problem["vS_k"].set_values_from_function(lambda x, y: np.array([-0.01 + 0.01 * x, 0.04 + 0.02 * y]))
    problem["vS_n"].set_values_from_function(lambda x, y: np.array([-0.02 + 0.01 * x, 0.03 + 0.01 * y]))
    problem["u_k"].set_values_from_function(lambda x, y: np.array([0.01 + 0.02 * x * y, -0.015 + 0.01 * y]))
    problem["u_n"].set_values_from_function(lambda x, y: np.array([0.008 + 0.015 * x * y, -0.010 + 0.006 * y]))
    problem["p_k"].set_values_from_function(lambda x, y: 0.3 + 0.1 * x - 0.05 * y)
    problem["p_n"].set_values_from_function(lambda x, y: 0.2 + 0.05 * x - 0.03 * y)
    problem["p_pore_k"].set_values_from_function(lambda x, y: 0.25 - 0.04 * x + 0.08 * y)
    problem["p_pore_n"].set_values_from_function(lambda x, y: 0.18 - 0.02 * x + 0.05 * y)
    problem["alpha_k"].set_values_from_function(lambda x, y: 0.45 + 0.10 * x + 0.08 * y)
    problem["alpha_n"].set_values_from_function(lambda x, y: 0.40 + 0.08 * x + 0.06 * y)
    problem["phi_k"].set_values_from_function(lambda x, y: 0.32 + 0.04 * x - 0.03 * y)
    problem["phi_n"].set_values_from_function(lambda x, y: 0.30 + 0.03 * x - 0.02 * y)
    problem["rho_s_k"].set_values_from_function(lambda x, y: 1.10 + 0.05 * x + 0.02 * y)
    problem["rho_s_n"].set_values_from_function(lambda x, y: 1.08 + 0.04 * x + 0.01 * y)
    for name in ("mu_mass_k", "mu_mass_n", "mu_normal_k", "mu_normal_n", "mu_tangent_k", "mu_tangent_n"):
        problem[name].nodal_values[:] = 0.0

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
        final_form_constant_rho_s=False,
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
        interface_bjs_closure=False,
        interface_velocity_continuity_closure=False,
        interface_traction_continuity_closure=False,
    )
    return problem, forms


def _assemble_vector(problem, form, *, backend: str) -> np.ndarray:
    _, rhs = assemble_form(
        Equation(None, form),
        dof_handler=problem["dh"],
        bcs=[],
        backend=backend,
    )
    return np.asarray(rhs, dtype=float)


def _assemble_matrix(problem, form, *, backend: str) -> np.ndarray:
    mat, _ = assemble_form(
        Equation(form, None),
        dof_handler=problem["dh"],
        bcs=[],
        backend=backend,
    )
    if hasattr(mat, "toarray"):
        return np.asarray(mat.toarray(), dtype=float)
    return np.asarray(mat, dtype=float)


def _fusion_group_sizes(problem, form) -> list[int]:
    compiler = FormCompiler(problem["dh"], backend="python")
    p_geo = int(getattr(problem["dh"].mixed_element.mesh, "poly_order", 1))
    groups: dict[tuple, list[int]] = {}
    for idx, integral in enumerate(_form_integrals(form)):
        key = _integral_fusion_key(integral, compiler=compiler, p_geo=p_geo)
        groups.setdefault(key, []).append(idx)
    return sorted(len(items) for items in groups.values())


def test_benchmark7_tensor_final_form_matches_decomposed_reference() -> None:
    problem_tensor, forms_tensor = _build_final_form_problem(final_form_implementation="tensor")
    problem_decomp, forms_decomp = _build_final_form_problem(final_form_implementation="decomposed")

    tensor_residual = _assemble_vector(problem_tensor, forms_tensor.residual_form, backend="python")
    decomp_residual = _assemble_vector(problem_decomp, forms_decomp.residual_form, backend="python")
    tensor_jacobian = _assemble_matrix(problem_tensor, forms_tensor.jacobian_form, backend="python")
    decomp_jacobian = _assemble_matrix(problem_decomp, forms_decomp.jacobian_form, backend="python")

    np.testing.assert_allclose(tensor_residual, decomp_residual, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(tensor_jacobian, decomp_jacobian, rtol=1.0e-10, atol=1.0e-10)

    term_names = (
        "interface_mass_constraint",
        "interface_normal_constraint",
        "interface_tangential_constraint",
        "interface_mass_bulk_coupling",
        "interface_normal_bulk_coupling",
        "interface_tangential_bulk_coupling",
    )
    for name in term_names:
        tensor_term = _assemble_vector(problem_tensor, forms_tensor.r_momentum_terms[name], backend="python")
        decomp_term = _assemble_vector(problem_decomp, forms_decomp.r_momentum_terms[name], backend="python")
        np.testing.assert_allclose(tensor_term, decomp_term, rtol=1.0e-11, atol=1.0e-11)


@pytest.mark.parametrize("backend", ["jit", "cpp"])
def test_benchmark7_tensor_final_form_backend_parity(backend: str) -> None:
    problem, forms = _build_final_form_problem(final_form_implementation="tensor")

    residual_py = _assemble_vector(problem, forms.residual_form, backend="python")
    residual_backend = _assemble_vector(problem, forms.residual_form, backend=backend)
    jacobian_py = _assemble_matrix(problem, forms.jacobian_form, backend="python")
    jacobian_backend = _assemble_matrix(problem, forms.jacobian_form, backend=backend)

    np.testing.assert_allclose(residual_backend, residual_py, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(jacobian_backend, jacobian_py, rtol=1.0e-10, atol=1.0e-10)

    term_names = (
        "interface_mass_constraint",
        "interface_normal_constraint",
        "interface_tangential_constraint",
        "interface_mass_bulk_coupling",
        "interface_normal_bulk_coupling",
        "interface_tangential_bulk_coupling",
    )
    for name in term_names:
        term_py = _assemble_vector(problem, forms.r_momentum_terms[name], backend="python")
        term_backend = _assemble_vector(problem, forms.r_momentum_terms[name], backend=backend)
        np.testing.assert_allclose(term_backend, term_py, rtol=1.0e-10, atol=1.0e-10)


def test_benchmark7_tensor_final_form_cpp_shared_loop_backend_parity(monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CPP_FAST_COMPILE", "1")
    monkeypatch.setenv("PYCUTFEM_CPP_FUSE_INTEGRALS", "1")

    problem, forms = _build_final_form_problem(final_form_implementation="tensor")

    residual_py = _assemble_vector(problem, forms.residual_form, backend="python")
    residual_cpp = _assemble_vector(problem, forms.residual_form, backend="cpp")
    jacobian_py = _assemble_matrix(problem, forms.jacobian_form, backend="python")
    jacobian_cpp = _assemble_matrix(problem, forms.jacobian_form, backend="cpp")

    np.testing.assert_allclose(residual_cpp, residual_py, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(jacobian_cpp, jacobian_py, rtol=1.0e-10, atol=1.0e-10)


def test_benchmark7_tensor_final_form_volume_fusion_groups_stay_coarse() -> None:
    problem, forms = _build_final_form_problem(final_form_implementation="tensor")

    assert _fusion_group_sizes(problem, forms.residual_form) == [len(_form_integrals(forms.residual_form))]
    assert _fusion_group_sizes(problem, forms.jacobian_form) == [len(_form_integrals(forms.jacobian_form))]


def test_benchmark7_tensor_final_form_jit_planner_splits_oversized_groups(monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_JIT_FUSION_IR_BUDGET", "2000")
    problem, forms = _build_final_form_problem(final_form_implementation="tensor")

    compiler = FormCompiler(problem["dh"], backend="jit")
    p_geo = int(getattr(problem["dh"].mixed_element.mesh, "poly_order", 1))
    planned_residual = _plan_integral_execution_units(
        _form_integrals(forms.residual_form),
        compiler=compiler,
        p_geo=p_geo,
        backend="jit",
    )
    planned_jacobian = _plan_integral_execution_units(
        _form_integrals(forms.jacobian_form),
        compiler=compiler,
        p_geo=p_geo,
        backend="jit",
    )

    assert _fusion_group_sizes(problem, forms.residual_form) == [len(_form_integrals(forms.residual_form))]
    assert _fusion_group_sizes(problem, forms.jacobian_form) == [len(_form_integrals(forms.jacobian_form))]
    assert len(planned_residual) > 1
    assert len(planned_jacobian) > 1
