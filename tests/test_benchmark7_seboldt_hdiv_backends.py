import numpy as np
import pytest

from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import _build_forms, _create_problem
from pycutfem.ufl.expressions import Constant, HdivTrialFunction, TestFunction, div, dot, grad, TrialFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx


def _initialize_benchmark7_state(problem: dict[str, object]) -> None:
    dh = problem["dh"]
    v_coords = np.asarray(dh.get_dof_coords("v"), dtype=float)
    problem["v_k"].nodal_values[:] = 0.18 + 0.04 * v_coords[:, 0] - 0.03 * v_coords[:, 1]
    problem["v_n"].nodal_values[:] = -0.07 + 0.03 * v_coords[:, 0] + 0.02 * v_coords[:, 1]

    problem["p_k"].set_values_from_function(lambda x, y: 0.25 + 0.10 * x - 0.05 * y)
    problem["p_n"].set_values_from_function(lambda x, y: 0.12 + 0.04 * x - 0.03 * y)
    problem["vS_k"].set_values_from_function(lambda x, y: np.array([0.05 + 0.02 * x - 0.01 * y, -0.03 + 0.01 * x + 0.015 * y]))
    problem["vS_n"].set_values_from_function(lambda x, y: np.array([0.03 + 0.01 * x - 0.008 * y, -0.015 + 0.008 * x + 0.010 * y]))
    problem["u_k"].set_values_from_function(lambda x, y: np.array([0.015 + 0.010 * x * (1.0 - x), -0.008 + 0.006 * y * (1.0 - y / 1.5)]))
    problem["u_n"].set_values_from_function(lambda x, y: np.array([0.010 + 0.006 * x * (1.0 - x), -0.005 + 0.004 * y * (1.0 - y / 1.5)]))
    problem["alpha_k"].set_values_from_function(lambda x, y: 0.58 + 0.015 * x + 0.010 * y)
    problem["alpha_n"].set_values_from_function(lambda x, y: 0.54 + 0.010 * x + 0.008 * y)
    problem["mu_k"].set_values_from_function(lambda x, y: 0.10 + 0.020 * x - 0.015 * y)
    problem["mu_n"].set_values_from_function(lambda x, y: 0.07 + 0.015 * x - 0.010 * y)
    if problem.get("phi_k") is not None:
        problem["phi_k"].set_values_from_function(lambda x, y: 0.18 + 0.020 * x - 0.010 * y)
        problem["phi_n"].set_values_from_function(lambda x, y: 0.18 + 0.015 * x - 0.008 * y)
    if problem.get("S_k") is not None:
        problem["S_k"].set_values_from_function(lambda x, y: 1.0)
        problem["S_n"].set_values_from_function(lambda x, y: 1.0)


def _assemble_benchmark7_terms(backend: str):
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=2,
        ny=3,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="hdiv",
        fluid_hdiv_order=0,
        enable_phi_evolution=False,
    )
    _initialize_benchmark7_state(problem)

    forms = _build_forms(
        problem,
        qdeg=6,
        dt_c=Constant(0.1),
        theta=0.5,
        rho_f=1.0,
        mu_f=0.035,
        mu_b=0.035,
        mu_b_model="mu",
        kappa_inv=1000.0,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        phi_b=0.5,
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
        alpha_regularization="ch",
        alpha_reg_gamma=1.0,
        alpha_reg_eps_normal=0.05,
        alpha_reg_eps_tangent=0.0125,
        alpha_reg_eta=1.0e-12,
        alpha_advect_with="vS",
        alpha_advection_form="conservative",
        solid_model="linear",
        kappa_inv_model="spatial",
        enable_phi_evolution=False,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        skeleton_inertia_convection="lagged",
    )

    a_mass, _ = assemble_form(Equation(forms.a_mass, None), dof_handler=problem["dh"], bcs=[], backend=backend)
    jacobian, _ = assemble_form(Equation(forms.jacobian_form, None), dof_handler=problem["dh"], bcs=[], backend=backend)
    _, residual = assemble_form(Equation(None, forms.residual_form), dof_handler=problem["dh"], bcs=[], backend=backend)
    return a_mass.toarray(), jacobian.toarray(), np.asarray(residual, dtype=float)


def _assemble_benchmark7_full_hdiv_terms(backend: str):
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=1,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="hdiv",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
    )
    _initialize_benchmark7_state(problem)

    forms = _build_forms(
        problem,
        qdeg=6,
        dt_c=Constant(0.1),
        theta=1.0,
        rho_f=1.0,
        mu_f=0.035,
        mu_b=0.035,
        mu_b_model="mu",
        kappa_inv=1.0e5,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        phi_b=0.18,
        M_alpha=1.0,
        gamma_alpha=1.0,
        eps_alpha=0.05,
        solid_visco_eta=0.0,
        gamma_div=0.0,
        gamma_u=1.0,
        u_extension_mode="l2",
        gamma_u_pin=0.0,
        u_cip=0.0,
        u_cip_weight="fluid",
        vS_cip=1.0,
        gamma_vS=1.0,
        vS_extension_mode="grad",
        gamma_vS_pin=1.0e-6,
        D_phi=0.0,
        phi_diffusion_weight="fluid",
        gamma_phi=5.0,
        phi_supg=0.0,
        phi_cip=0.0,
        alpha_supg=0.0,
        alpha_cip=0.0,
        alpha_regularization="ch",
        alpha_reg_gamma=1.0,
        alpha_reg_eps_normal=0.05,
        alpha_reg_eps_tangent=0.0125,
        alpha_reg_eta=1.0e-12,
        alpha_advect_with="vS",
        alpha_advection_form="conservative",
        solid_model="linear",
        kappa_inv_model="refmap",
        enable_phi_evolution=True,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        skeleton_inertia_convection="lagged",
    )

    jacobian, _ = assemble_form(Equation(forms.jacobian_form, None), dof_handler=problem["dh"], bcs=[], backend=backend)
    _, residual = assemble_form(Equation(None, forms.residual_form), dof_handler=problem["dh"], bcs=[], backend=backend)
    return jacobian.toarray(), np.asarray(residual, dtype=float)


def _assemble_benchmark7_full_hdiv_pressure_block_terms(backend: str):
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=1,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="hdiv",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
    )
    _initialize_benchmark7_state(problem)

    one = Constant(1.0)
    test_p = TestFunction(field_name="p", dof_handler=problem["dh"], name="test_p")
    dalpha = TrialFunction(field_name="alpha", dof_handler=problem["dh"], name="trial_alpha")
    dphi = TrialFunction(field_name="phi", dof_handler=problem["dh"], name="trial_phi")
    dv = HdivTrialFunction(field_name="v")

    alpha_k = problem["alpha_k"]
    phi_k = problem["phi_k"]
    v_k = problem["v_k"]
    dx_q = dx(metadata={"q": 6})

    coeff_k = alpha_k * (phi_k - one)
    grad_coeff_k = grad(alpha_k) * (phi_k - one) + grad(phi_k) * alpha_k
    dcoeff = (phi_k - one) * dalpha + alpha_k * dphi
    dgrad_coeff = (
        grad(dalpha) * (phi_k - one)
        + grad(alpha_k) * dphi
        + grad(phi_k) * dalpha
        + grad(dphi) * alpha_k
    )

    residual_form = test_p * (coeff_k * div(v_k) + dot(grad_coeff_k, v_k)) * dx_q
    jacobian_form = test_p * (
        dcoeff * div(v_k)
        + coeff_k * div(dv)
        + dot(dgrad_coeff, v_k)
        + dot(grad_coeff_k, dv)
    ) * dx_q

    jacobian, _ = assemble_form(Equation(jacobian_form, None), dof_handler=problem["dh"], bcs=[], backend=backend)
    _, residual = assemble_form(Equation(None, residual_form), dof_handler=problem["dh"], bcs=[], backend=backend)
    return jacobian.toarray(), np.asarray(residual, dtype=float)


def _assemble_benchmark7_hdiv_div_warmup(backend: str) -> np.ndarray:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=2,
        ny=3,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="hdiv",
        fluid_hdiv_order=0,
        enable_phi_evolution=False,
    )
    _initialize_benchmark7_state(problem)

    expr = TestFunction(field_name="p", dof_handler=problem["dh"], name="test_p") * div(HdivTrialFunction(field_name="v")) * dx(metadata={"q": 6})
    warmup, _ = assemble_form(Equation(expr, None), dof_handler=problem["dh"], bcs=[], backend=backend)
    return warmup.toarray()


@pytest.mark.parametrize("backend", ("jit", "cpp"))
def test_benchmark7_seboldt_hdiv_mass_and_total_forms_match_python(backend, monkeypatch, tmp_path):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_{backend}"))
    if backend == "cpp":
        monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")
    else:
        monkeypatch.delenv("PYCUTFEM_JIT_BACKEND", raising=False)

    a_mass_ref, jac_ref, res_ref = _assemble_benchmark7_terms("python")
    a_mass, jac, res = _assemble_benchmark7_terms(backend)

    np.testing.assert_allclose(a_mass, a_mass_ref, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(jac, jac_ref, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(res, res_ref, rtol=1.0e-9, atol=1.0e-9)


@pytest.mark.parametrize("backend", ("cpp",))
def test_benchmark7_seboldt_full_hdiv_refmap_grad_backend_matches_python_on_1x1(backend, monkeypatch, tmp_path):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_full_{backend}"))
    if backend == "cpp":
        monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")
    else:
        monkeypatch.delenv("PYCUTFEM_JIT_BACKEND", raising=False)

    jac_ref, res_ref = _assemble_benchmark7_full_hdiv_terms("python")
    jac, res = _assemble_benchmark7_full_hdiv_terms(backend)

    np.testing.assert_allclose(jac, jac_ref, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(res, res_ref, rtol=1.0e-9, atol=1.0e-9)


@pytest.mark.parametrize("backend", ("jit", "cpp"))
def test_benchmark7_full_hdiv_pressure_block_matches_python_on_1x1(backend, monkeypatch, tmp_path):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_pressure_block_{backend}"))
    if backend == "cpp":
        monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")
        monkeypatch.setenv("PYCUTFEM_CPP_FAST_COMPILE", "1")
        monkeypatch.setenv("PYCUTFEM_CPP_FAST_OPT_LEVEL", "0")
        monkeypatch.setenv("PYCUTFEM_CPP_FAST_MARCH_NATIVE", "0")
    else:
        monkeypatch.delenv("PYCUTFEM_JIT_BACKEND", raising=False)

    jac_ref, res_ref = _assemble_benchmark7_full_hdiv_pressure_block_terms("python")
    jac, res = _assemble_benchmark7_full_hdiv_pressure_block_terms(backend)

    np.testing.assert_allclose(jac, jac_ref, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(res, res_ref, rtol=1.0e-9, atol=1.0e-9)


def test_hdiv_ref_table_cache_respects_active_layout(monkeypatch, tmp_path):
    import pycutfem.jit.kernel_args as kernel_args

    kernel_args._REF_TABLE_CACHE.clear()
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "jit_cache_active_layout"))
    monkeypatch.delenv("PYCUTFEM_JIT_BACKEND", raising=False)

    warm_ref = _assemble_benchmark7_hdiv_div_warmup("python")
    warm_jit = _assemble_benchmark7_hdiv_div_warmup("jit")
    np.testing.assert_allclose(warm_jit, warm_ref, rtol=1.0e-9, atol=1.0e-9)

    a_mass_ref, jac_ref, res_ref = _assemble_benchmark7_terms("python")
    a_mass, jac, res = _assemble_benchmark7_terms("jit")

    np.testing.assert_allclose(a_mass, a_mass_ref, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(jac, jac_ref, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(res, res_ref, rtol=1.0e-9, atol=1.0e-9)
