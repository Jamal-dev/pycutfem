import numpy as np
import pytest

from examples.utils.biofilm import three_constituent_one_domain as tc_model
from examples.utils.biofilm.three_constituent_one_domain import (
    build_three_constituent_pdas_solver,
    build_three_constituent_one_domain_forms,
    configure_three_constituent_pdas_bounds,
    three_constituent_box_bounds,
)
from examples.utils.biofilm.three_constituent_mms import (
    backward_euler_three_constituent_sources,
)
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, PdasNewtonSolver, VIParameters
from pycutfem.ufl.expressions import (
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    dot,
    grad,
)
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.jit_parametrization import build_jit_parametrization
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _have_pybind11() -> bool:
    try:
        import pybind11  # noqa: F401
    except Exception:
        return False
    return True


def _build_problem(
    *,
    return_state: bool = False,
    velocity_shift=(0.0, 0.0),
    R_pair_cholesky=None,
    pair_weight_epsilon: float = 0.0,
    gamma_mobility: str = "FP",
):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(
        mesh,
        field_specs={
            "vf_x": 1,
            "vf_y": 1,
            "pf": 1,
            "vp_x": 1,
            "vp_y": 1,
            "pp": 1,
            "vs_x": 1,
            "vs_y": 1,
            "us_x": 1,
            "us_y": 1,
            "alpha": 1,
            "phi": 1,
            "Gamma": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    VF = FunctionSpace("VF", ["vf_x", "vf_y"], dim=1)
    VP = FunctionSpace("VP", ["vp_x", "vp_y"], dim=1)
    VS = FunctionSpace("VS", ["vs_x", "vs_y"], dim=1)
    US = FunctionSpace("US", ["us_x", "us_y"], dim=1)

    trial = {
        "dv_f": VectorTrialFunction(space=VF, dof_handler=dh),
        "dp_f": TrialFunction("pf", dof_handler=dh),
        "dv_p": VectorTrialFunction(space=VP, dof_handler=dh),
        "dp_p": TrialFunction("pp", dof_handler=dh),
        "dv_s": VectorTrialFunction(space=VS, dof_handler=dh),
        "du_s": VectorTrialFunction(space=US, dof_handler=dh),
        "dalpha": TrialFunction("alpha", dof_handler=dh),
        "dphi": TrialFunction("phi", dof_handler=dh),
        "dGamma": TrialFunction("Gamma", dof_handler=dh),
    }
    test = {
        "w_f": VectorTestFunction(space=VF, dof_handler=dh),
        "q_f": TestFunction("pf", dof_handler=dh),
        "w_p": VectorTestFunction(space=VP, dof_handler=dh),
        "q_p": TestFunction("pp", dof_handler=dh),
        "w_s": VectorTestFunction(space=VS, dof_handler=dh),
        "z_u": VectorTestFunction(space=US, dof_handler=dh),
        "z_alpha": TestFunction("alpha", dof_handler=dh),
        "q_s": TestFunction("phi", dof_handler=dh),
        "z_Gamma": TestFunction("Gamma", dof_handler=dh),
    }

    state = {
        "v_f_k": VectorFunction("v_f_k", ["vf_x", "vf_y"], dof_handler=dh),
        "p_f_k": Function("p_f_k", "pf", dof_handler=dh),
        "v_p_k": VectorFunction("v_p_k", ["vp_x", "vp_y"], dof_handler=dh),
        "p_p_k": Function("p_p_k", "pp", dof_handler=dh),
        "v_s_k": VectorFunction("v_s_k", ["vs_x", "vs_y"], dof_handler=dh),
        "u_s_k": VectorFunction("u_s_k", ["us_x", "us_y"], dof_handler=dh),
        "alpha_k": Function("alpha_k", "alpha", dof_handler=dh),
        "phi_k": Function("phi_k", "phi", dof_handler=dh),
        "Gamma_k": Function("Gamma_k", "Gamma", dof_handler=dh),
        "v_f_n": VectorFunction("v_f_n", ["vf_x", "vf_y"], dof_handler=dh),
        "v_p_n": VectorFunction("v_p_n", ["vp_x", "vp_y"], dof_handler=dh),
        "v_s_n": VectorFunction("v_s_n", ["vs_x", "vs_y"], dof_handler=dh),
        "u_s_n": VectorFunction("u_s_n", ["us_x", "us_y"], dof_handler=dh),
        "alpha_n": Function("alpha_n", "alpha", dof_handler=dh),
        "phi_n": Function("phi_n", "phi", dof_handler=dh),
        "Gamma_n": Function("Gamma_n", "Gamma", dof_handler=dh),
    }

    shift = np.asarray(velocity_shift, dtype=float)
    state["v_f_k"].set_values_from_function(lambda x, y: np.asarray([0.10 + 0.03 * x, -0.04 + 0.02 * y]) + shift)
    state["v_f_n"].set_values_from_function(lambda x, y: np.asarray([0.08 + 0.02 * x, -0.03 + 0.01 * y]) + shift)
    state["v_p_k"].set_values_from_function(lambda x, y: np.asarray([0.05 - 0.01 * x, 0.07 + 0.02 * y]) + shift)
    state["v_p_n"].set_values_from_function(lambda x, y: np.asarray([0.04 - 0.01 * x, 0.05 + 0.01 * y]) + shift)
    state["v_s_k"].set_values_from_function(lambda x, y: np.asarray([-0.01 + 0.02 * x, 0.02 - 0.01 * y]) + shift)
    state["v_s_n"].set_values_from_function(lambda x, y: np.asarray([-0.015 + 0.01 * x, 0.018 - 0.01 * y]) + shift)
    state["u_s_k"].set_values_from_function(lambda x, y: np.asarray([0.02 * x * y + 0.01, -0.01 * y + 0.003 * x]))
    state["u_s_n"].set_values_from_function(lambda x, y: np.asarray([0.015 * x * y + 0.008, -0.008 * y + 0.002 * x]))
    state["p_f_k"].set_values_from_function(lambda x, y: 0.20 + 0.05 * x - 0.03 * y)
    state["p_p_k"].set_values_from_function(lambda x, y: 0.15 - 0.02 * x + 0.04 * y)
    state["alpha_k"].set_values_from_function(lambda x, y: 0.35 + 0.08 * x + 0.06 * y)
    state["alpha_n"].set_values_from_function(lambda x, y: 0.33 + 0.05 * x + 0.04 * y)
    state["phi_k"].set_values_from_function(lambda x, y: 0.55 + 0.03 * x - 0.02 * y)
    state["phi_n"].set_values_from_function(lambda x, y: 0.53 + 0.02 * x - 0.01 * y)
    state["Gamma_k"].set_values_from_function(lambda x, y: 0.01 + 0.003 * x - 0.002 * y)
    state["Gamma_n"].set_values_from_function(lambda x, y: 0.008 + 0.002 * x - 0.001 * y)

    forms = build_three_constituent_one_domain_forms(
        **state,
        **trial,
        **test,
        dx=dx(metadata={"q": 4}),
        dt=0.1,
        rho_f=1.0,
        rho_p=1.0,
        rho_s=1.4,
        mu_f=0.02,
        mu_p=0.005,
        mu_s=2.0,
        lambda_s=3.0,
        R_fp=0.7,
        R_fs=0.4,
        R_ps=1.5,
        R_pair_cholesky=R_pair_cholesky,
        pair_weight_epsilon=float(pair_weight_epsilon),
        theta_fp=0.35,
        ell_Gamma=0.2,
        gamma_mobility=gamma_mobility,
    )
    if return_state:
        return dh, forms, test, state
    return dh, forms, test


def _build_closed_mms_problem():
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    me = MixedElement(
        mesh,
        field_specs={
            "vf_x": 2,
            "vf_y": 2,
            "pf": 2,
            "vp_x": 2,
            "vp_y": 2,
            "pp": 2,
            "vs_x": 2,
            "vs_y": 2,
            "us_x": 2,
            "us_y": 2,
            "alpha": 2,
            "phi": 2,
            "Gamma": 2,
        },
    )
    dh = DofHandler(me, method="cg")

    VF = FunctionSpace("VF", ["vf_x", "vf_y"], dim=1)
    VP = FunctionSpace("VP", ["vp_x", "vp_y"], dim=1)
    VS = FunctionSpace("VS", ["vs_x", "vs_y"], dim=1)
    US = FunctionSpace("US", ["us_x", "us_y"], dim=1)

    trial = {
        "dv_f": VectorTrialFunction(space=VF, dof_handler=dh),
        "dp_f": TrialFunction("pf", dof_handler=dh),
        "dv_p": VectorTrialFunction(space=VP, dof_handler=dh),
        "dp_p": TrialFunction("pp", dof_handler=dh),
        "dv_s": VectorTrialFunction(space=VS, dof_handler=dh),
        "du_s": VectorTrialFunction(space=US, dof_handler=dh),
        "dalpha": TrialFunction("alpha", dof_handler=dh),
        "dphi": TrialFunction("phi", dof_handler=dh),
        "dGamma": TrialFunction("Gamma", dof_handler=dh),
    }
    test = {
        "w_f": VectorTestFunction(space=VF, dof_handler=dh),
        "q_f": TestFunction("pf", dof_handler=dh),
        "w_p": VectorTestFunction(space=VP, dof_handler=dh),
        "q_p": TestFunction("pp", dof_handler=dh),
        "w_s": VectorTestFunction(space=VS, dof_handler=dh),
        "z_u": VectorTestFunction(space=US, dof_handler=dh),
        "z_alpha": TestFunction("alpha", dof_handler=dh),
        "q_s": TestFunction("phi", dof_handler=dh),
        "z_Gamma": TestFunction("Gamma", dof_handler=dh),
    }
    state = {
        "v_f_k": VectorFunction("v_f_k", ["vf_x", "vf_y"], dof_handler=dh),
        "p_f_k": Function("p_f_k", "pf", dof_handler=dh),
        "v_p_k": VectorFunction("v_p_k", ["vp_x", "vp_y"], dof_handler=dh),
        "p_p_k": Function("p_p_k", "pp", dof_handler=dh),
        "v_s_k": VectorFunction("v_s_k", ["vs_x", "vs_y"], dof_handler=dh),
        "u_s_k": VectorFunction("u_s_k", ["us_x", "us_y"], dof_handler=dh),
        "alpha_k": Function("alpha_k", "alpha", dof_handler=dh),
        "phi_k": Function("phi_k", "phi", dof_handler=dh),
        "Gamma_k": Function("Gamma_k", "Gamma", dof_handler=dh),
        "v_f_n": VectorFunction("v_f_n", ["vf_x", "vf_y"], dof_handler=dh),
        "v_p_n": VectorFunction("v_p_n", ["vp_x", "vp_y"], dof_handler=dh),
        "v_s_n": VectorFunction("v_s_n", ["vs_x", "vs_y"], dof_handler=dh),
        "u_s_n": VectorFunction("u_s_n", ["us_x", "us_y"], dof_handler=dh),
        "alpha_n": Function("alpha_n", "alpha", dof_handler=dh),
        "phi_n": Function("phi_n", "phi", dof_handler=dh),
        "Gamma_n": Function("Gamma_n", "Gamma", dof_handler=dh),
    }

    def bubble(x, y):
        return x * (1.0 - x) * y * (1.0 - y)

    state["v_f_k"].set_values_from_function(lambda x, y: np.asarray([0.35 * bubble(x, y), -0.22 * bubble(x, y)]))
    state["v_p_k"].set_values_from_function(lambda x, y: np.asarray([-0.18 * bubble(x, y), 0.29 * bubble(x, y)]))
    state["v_s_k"].set_values_from_function(lambda x, y: np.asarray([0.11 * bubble(x, y), -0.07 * bubble(x, y)]))
    state["v_f_n"].set_values_from_function(lambda x, y: np.asarray([0.21 * bubble(x, y), -0.12 * bubble(x, y)]))
    state["v_p_n"].set_values_from_function(lambda x, y: np.asarray([-0.09 * bubble(x, y), 0.16 * bubble(x, y)]))
    state["v_s_n"].set_values_from_function(lambda x, y: np.asarray([0.08 * bubble(x, y), -0.05 * bubble(x, y)]))

    for name in ("p_f_k", "p_p_k"):
        state[name].nodal_values.fill(0.0)
    for name in ("u_s_k", "u_s_n"):
        state[name].nodal_values.fill(0.0)

    state["alpha_k"].set_values_from_function(lambda x, y: 0.42 + 0.05 * x - 0.03 * y)
    state["alpha_n"].set_values_from_function(lambda x, y: 0.39 + 0.04 * x - 0.02 * y)
    state["phi_k"].set_values_from_function(lambda x, y: 0.58 - 0.04 * x + 0.02 * y)
    state["phi_n"].set_values_from_function(lambda x, y: 0.55 - 0.03 * x + 0.01 * y)
    state["Gamma_k"].set_values_from_function(lambda x, y: 0.015 + 0.002 * x - 0.001 * y)
    state["Gamma_n"].set_values_from_function(lambda x, y: 0.011 + 0.001 * x - 0.0005 * y)

    params = {
        "dt": 0.07,
        "rho_f": 1.0,
        "rho_p": 1.0,
        "rho_s": 1.35,
        "mu_f": 0.0,
        "mu_p": 0.0,
        "mu_s": 0.0,
        "lambda_s": 0.0,
        "R_fp": 0.6,
        "R_fs": 0.35,
        "R_ps": 1.25,
        "theta_fp": 0.4,
        "ell_Gamma": 0.15,
        "include_stress_divergence": False,
    }
    sources = backward_euler_three_constituent_sources(**state, **params)
    build_params = dict(params)
    build_params.pop("include_stress_divergence")
    forms = build_three_constituent_one_domain_forms(
        **state,
        **trial,
        **test,
        dx=dx(metadata={"q": 6}),
        **build_params,
        **sources,
    )
    return dh, forms


def _assemble_vector(dh, form, *, backend: str) -> np.ndarray:
    _, rhs = assemble_form(Equation(None, form), dof_handler=dh, bcs=[], backend=backend)
    return np.asarray(rhs, dtype=float)


def _assemble_matrix(dh, form, *, backend: str) -> np.ndarray:
    mat, _ = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend=backend)
    if hasattr(mat, "toarray"):
        return np.asarray(mat.toarray(), dtype=float)
    return np.asarray(mat, dtype=float)


def test_three_constituent_constants_are_named_for_jit_parametrization():
    problems = [
        _build_problem()[1],
        _build_problem(gamma_mobility="interface_delta")[1],
        _build_problem(
            R_pair_cholesky=((0.9, 0.0, 0.0), (0.25, 0.7, 0.0), (-0.18, 0.22, 1.1)),
            pair_weight_epsilon=1.0e-12,
        )[1],
    ]

    for forms in problems:
        for form in (forms.residual_form, forms.jacobian_form):
            for integral in form.integrals:
                names = set(build_jit_parametrization(integral.integrand).const_by_name)
                unnamed = sorted(name for name in names if name.startswith("jit_const_"))
                assert unnamed == []


def test_three_constituent_alpha_phi_bounds_configure_pdas_solver():
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"alpha": 1, "phi": 1})
    dh = DofHandler(me, method="cg")

    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    z_alpha = TestFunction("alpha", dof_handler=dh)
    z_phi = TestFunction("phi", dof_handler=dh)
    d_alpha = TrialFunction("alpha", dof_handler=dh)
    d_phi = TrialFunction("phi", dof_handler=dh)
    residual = (alpha_k * z_alpha + phi_k * z_phi) * dx(metadata={"q": 2})
    jacobian = (d_alpha * z_alpha + d_phi * z_phi) * dx(metadata={"q": 2})

    solver = PdasNewtonSolver(
        residual,
        jacobian,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-12, max_newton_iter=3, print_level=0),
        vi_params=VIParameters(c=1.0),
        backend="python",
    )
    configure_three_constituent_pdas_bounds(solver)

    alpha_slice = np.asarray(dh.get_field_slice("alpha"), dtype=int)
    phi_slice = np.asarray(dh.get_field_slice("phi"), dtype=int)
    assert three_constituent_box_bounds() == {"alpha": (0.0, 1.0), "phi": (0.0, 1.0)}
    assert three_constituent_box_bounds(pore_pressure_bounds=(0.0, None)) == {
        "alpha": (0.0, 1.0),
        "phi": (0.0, 1.0),
        "pp": (0.0, None),
    }
    np.testing.assert_allclose(solver._box_lower_full[alpha_slice], 0.0)
    np.testing.assert_allclose(solver._box_upper_full[alpha_slice], 1.0)
    np.testing.assert_allclose(solver._box_lower_full[phi_slice], 0.0)
    np.testing.assert_allclose(solver._box_upper_full[phi_slice], 1.0)


def test_three_constituent_optional_pore_pressure_lower_bound():
    dh, forms, _ = _build_problem()

    solver = build_three_constituent_pdas_solver(
        forms,
        dof_handler=dh,
        mixed_element=dh.mixed_element,
        newton_params=NewtonParameters(newton_tol=1.0e-12, max_newton_iter=3, print_level=0),
        vi_params=VIParameters(c=1.0),
        backend="python",
        pore_pressure_bounds=(0.0, None),
    )

    pp_slice = np.asarray(dh.get_field_slice("pp"), dtype=int)
    np.testing.assert_allclose(solver._box_lower_full[pp_slice], 0.0)
    assert np.all(np.isposinf(solver._box_upper_full[pp_slice]))


def test_three_constituent_pdas_solver_factory_applies_bounds():
    dh, forms, _ = _build_problem()

    solver = build_three_constituent_pdas_solver(
        forms,
        dof_handler=dh,
        mixed_element=dh.mixed_element,
        newton_params=NewtonParameters(newton_tol=1.0e-12, max_newton_iter=3, print_level=0),
        vi_params=VIParameters(c=1.0),
        backend="python",
    )

    alpha_slice = np.asarray(dh.get_field_slice("alpha"), dtype=int)
    phi_slice = np.asarray(dh.get_field_slice("phi"), dtype=int)
    np.testing.assert_allclose(solver._box_lower_full[alpha_slice], 0.0)
    np.testing.assert_allclose(solver._box_upper_full[alpha_slice], 1.0)
    np.testing.assert_allclose(solver._box_lower_full[phi_slice], 0.0)
    np.testing.assert_allclose(solver._box_upper_full[phi_slice], 1.0)


def test_three_constituent_fraction_bounds_are_active_set_contract_not_raw_clip_api():
    _, _, _, state = _build_problem(return_state=True)

    state["alpha_k"].nodal_values[:] = np.linspace(-0.2, 1.2, state["alpha_k"].nodal_values.size)
    state["phi_k"].nodal_values[:] = np.linspace(-0.3, 1.3, state["phi_k"].nodal_values.size)

    assert "clip_three_constituent_fraction_fields" not in tc_model.__all__
    assert not hasattr(tc_model, "clip_three_constituent_fraction_fields")
    assert float(np.min(state["alpha_k"].nodal_values)) < 0.0
    assert float(np.max(state["alpha_k"].nodal_values)) > 1.0
    assert float(np.min(state["phi_k"].nodal_values)) < 0.0
    assert float(np.max(state["phi_k"].nodal_values)) > 1.0


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
def test_three_constituent_partition_and_structural_gradient_identities_cpp():
    dh, forms, test = _build_problem()

    F = forms.r_internal_force_terms["F"]
    P = forms.r_internal_force_terms["P"]
    B = forms.r_internal_force_terms["B"]
    g_fp = forms.r_internal_force_terms["g_fp"]
    g_fs = forms.r_internal_force_terms["g_fs"]
    g_ps = forms.r_internal_force_terms["g_ps"]

    partition = (F + P + B - 1.0) * test["q_f"] * dx(metadata={"q": 4})
    grad_F = dot(grad(F) + g_fp + g_fs, test["w_f"]) * dx(metadata={"q": 4})
    grad_P = dot(grad(P) - g_fp - g_ps, test["w_f"]) * dx(metadata={"q": 4})
    grad_B = dot(grad(B) - g_fs + g_ps, test["w_f"]) * dx(metadata={"q": 4})

    for form in (partition, grad_F, grad_P, grad_B):
        vec = _assemble_vector(dh, form, backend="cpp")
        np.testing.assert_allclose(vec, np.zeros_like(vec), rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
def test_three_constituent_dissipation_is_nonnegative_cpp():
    dh, forms, test = _build_problem()

    diss_form = forms.r_internal_force_terms["dissipation_density"] * test["q_f"] * dx(metadata={"q": 4})
    diss_vector = _assemble_vector(dh, diss_form, backend="cpp")

    assert float(np.sum(diss_vector)) >= -1.0e-13


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
def test_three_constituent_pairwise_dissipation_is_galilean_invariant_cpp():
    dh_base, forms_base, test_base = _build_problem()
    dh_shift, forms_shift, test_shift = _build_problem(velocity_shift=(2.5, -1.75))

    base_form = forms_base.r_internal_force_terms["dissipation_density"] * test_base["q_f"] * dx(metadata={"q": 4})
    shift_form = forms_shift.r_internal_force_terms["dissipation_density"] * test_shift["q_f"] * dx(metadata={"q": 4})
    base = _assemble_vector(dh_base, base_form, backend="cpp")
    shifted = _assemble_vector(dh_shift, shift_form, backend="cpp")

    np.testing.assert_allclose(shifted, base, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
def test_three_constituent_full_block_resistance_is_live_and_dissipative_cpp():
    chol = (
        (0.90, 0.0, 0.0),
        (0.25, 0.70, 0.0),
        (-0.18, 0.22, 1.10),
    )
    dh, forms, test = _build_problem(R_pair_cholesky=chol, pair_weight_epsilon=1.0e-12)

    cross_form = forms.r_internal_force_terms["C_fp_fs"] * test["q_f"] * dx(metadata={"q": 4})
    cross = _assemble_vector(dh, cross_form, backend="cpp")
    assert abs(float(np.sum(cross))) > 1.0e-12

    force_sum = (
        forms.r_internal_force_terms["I_f"]
        + forms.r_internal_force_terms["I_p"]
        + forms.r_internal_force_terms["I_s"]
    )
    cancellation = _assemble_vector(dh, dot(force_sum, test["w_f"]) * dx(metadata={"q": 4}), backend="cpp")
    dissipation = _assemble_vector(
        dh,
        forms.r_internal_force_terms["dissipation_density"] * test["q_f"] * dx(metadata={"q": 4}),
        backend="cpp",
    )

    np.testing.assert_allclose(cancellation, np.zeros_like(cancellation), rtol=1.0e-11, atol=1.0e-11)
    assert float(np.sum(dissipation)) >= -1.0e-13


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
def test_three_constituent_residual_and_jacobian_cpp_backend_parity():
    dh, forms, _ = _build_problem()

    residual_py = _assemble_vector(dh, forms.residual_form, backend="python")
    residual_cpp = _assemble_vector(dh, forms.residual_form, backend="cpp")
    jacobian_py = _assemble_matrix(dh, forms.jacobian_form, backend="python")
    jacobian_cpp = _assemble_matrix(dh, forms.jacobian_form, backend="cpp")

    np.testing.assert_allclose(residual_cpp, residual_py, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(jacobian_cpp, jacobian_py, rtol=1.0e-10, atol=1.0e-10)


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
def test_three_constituent_jacobian_matches_directional_fd_cpp_backend():
    dh, forms, _, state = _build_problem(return_state=True)

    jacobian = _assemble_matrix(dh, forms.jacobian_form, backend="cpp")
    residual0 = _assemble_vector(dh, forms.residual_form, backend="cpp")
    assert residual0.size == dh.total_dofs

    functions_by_field = {
        "vf_x": state["v_f_k"],
        "vf_y": state["v_f_k"],
        "pf": state["p_f_k"],
        "vp_x": state["v_p_k"],
        "vp_y": state["v_p_k"],
        "pp": state["p_p_k"],
        "vs_x": state["v_s_k"],
        "vs_y": state["v_s_k"],
        "us_x": state["u_s_k"],
        "us_y": state["u_s_k"],
        "alpha": state["alpha_k"],
        "phi": state["phi_k"],
        "Gamma": state["Gamma_k"],
    }
    probe_fields = ("vf_x", "pf", "vp_y", "pp", "vs_x", "us_y", "alpha", "phi", "Gamma")
    direction = np.zeros(dh.total_dofs, dtype=float)
    field_dofs: dict[str, list[int]] = {}
    for field in probe_fields:
        gdof = int(dh.element_dofs(field, 0)[0])
        direction[gdof] = 1.0
        field_dofs.setdefault(field, []).append(gdof)

    base_vals: dict[str, np.ndarray] = {}
    for field, dofs in field_dofs.items():
        arr = np.asarray(dofs, dtype=int)
        base_vals[field] = functions_by_field[field].get_nodal_values(arr)

    eps = 1.0e-7

    def set_perturb(sign: float) -> None:
        for field, dofs in field_dofs.items():
            arr = np.asarray(dofs, dtype=int)
            functions_by_field[field].set_nodal_values(arr, base_vals[field] + sign * eps)

    set_perturb(+1.0)
    residual_plus = _assemble_vector(dh, forms.residual_form, backend="cpp")
    set_perturb(-1.0)
    residual_minus = _assemble_vector(dh, forms.residual_form, backend="cpp")
    set_perturb(0.0)

    fd = (residual_plus - residual_minus) / (2.0 * eps)
    jac_dir = jacobian.dot(direction)
    err = float(np.linalg.norm(fd - jac_dir, ord=np.inf))
    denom = max(1.0, float(np.linalg.norm(fd, ord=np.inf)), float(np.linalg.norm(jac_dir, ord=np.inf)))

    assert err / denom < 1.0e-6


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
def test_three_constituent_internal_forces_cancel_in_cpp_backend():
    dh, forms, test = _build_problem()

    force_sum = (
        forms.r_internal_force_terms["I_f"]
        + forms.r_internal_force_terms["I_p"]
        + forms.r_internal_force_terms["I_s"]
    )
    cancellation_form = dot(force_sum, test["w_f"]) * dx(metadata={"q": 4})
    cancellation = _assemble_vector(dh, cancellation_form, backend="cpp")

    np.testing.assert_allclose(cancellation, np.zeros_like(cancellation), rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
def test_three_constituent_kinematic_row_is_live_with_eulerian_convection():
    dh, forms, _ = _build_problem()

    kin = _assemble_vector(dh, forms.r_kinematics, backend="cpp")

    assert np.linalg.norm(kin) > 1.0e-12


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
def test_three_constituent_pure_free_fluid_limit_deactivates_pore_and_skeleton_rows_cpp():
    dh, forms, _, state = _build_problem(return_state=True)

    state["alpha_k"].nodal_values.fill(0.0)
    state["alpha_n"].nodal_values.fill(0.0)
    state["Gamma_k"].nodal_values.fill(0.0)

    inactive_rows = (
        forms.r_alpha
        + forms.r_mass_p
        + forms.r_mass_s
        + forms.r_momentum_p
        + forms.r_momentum_s
        + forms.r_kinematics
        + forms.r_gamma
    )
    residual = _assemble_vector(dh, inactive_rows, backend="cpp")

    np.testing.assert_allclose(residual, np.zeros_like(residual), rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
def test_three_constituent_fixed_skeleton_keeps_alpha_stationary_cpp():
    dh, forms, _, state = _build_problem(return_state=True)

    state["alpha_n"].nodal_values[:] = state["alpha_k"].nodal_values
    state["v_s_k"].nodal_values.fill(0.0)

    residual = _assemble_vector(dh, forms.r_alpha, backend="cpp")

    np.testing.assert_allclose(residual, np.zeros_like(residual), rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
def test_three_constituent_constant_pore_pressure_has_no_porosity_artifact_cpp():
    dh, forms, test, state = _build_problem(return_state=True)

    state["alpha_k"].nodal_values.fill(1.0)
    state["alpha_n"].nodal_values.fill(1.0)
    state["v_p_k"].nodal_values.fill(0.0)
    state["p_p_k"].nodal_values.fill(0.73)

    pore_pressure_reversible_force = state["p_p_k"] * forms.r_internal_force_terms["g_ps"]
    artifact = forms.r_internal_force_terms["I_p_rev"] - pore_pressure_reversible_force
    residual = _assemble_vector(dh, dot(artifact, test["w_p"]) * dx(metadata={"q": 4}), backend="cpp")

    np.testing.assert_allclose(residual, np.zeros_like(residual), rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.skipif(not _have_pybind11(), reason="pybind11 not available")
def test_three_constituent_closed_mms_sources_make_residual_zero_cpp():
    dh, forms = _build_closed_mms_problem()

    residual = _assemble_vector(dh, forms.residual_form, backend="cpp")

    np.testing.assert_allclose(residual, np.zeros_like(residual), rtol=1.0e-10, atol=1.0e-10)
