import numpy as np
import pytest

from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import _build_forms, _create_problem
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, NewtonSolver
from pycutfem.ufl import linearize_form
from pycutfem.ufl.autodiff import NonDifferentiableGateauxNodeError
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    HdivTrialFunction,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    div,
    dot,
    grad,
    inner,
    pos_part,
)
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _build_scalar_problem():
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    u_k = Function("u_k", "u", dof_handler=dh)
    du = TrialFunction("u", dof_handler=dh)
    v = TestFunction("u", dof_handler=dh)
    u_k.set_values_from_function(lambda x, y: 0.25 + 0.10 * x - 0.05 * y)
    return me, dh, u_k, du, v, dx(metadata={"q": 4})


def _build_navier_like_problem():
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    me = MixedElement(mesh, field_specs={"ux": 2, "uy": 2, "p": 1})
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["ux", "uy"], dim=1)
    dv = VectorTrialFunction(space=V, dof_handler=dh)
    v_test = VectorTestFunction(space=V, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)

    u_k = VectorFunction("u_k", ["ux", "uy"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["ux", "uy"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)

    u_k.set_values_from_function(lambda x, y: np.array([0.30 + 0.05 * x, -0.10 + 0.04 * y]))
    u_n.set_values_from_function(lambda x, y: np.array([0.22 + 0.03 * x, -0.08 + 0.03 * y]))
    p_k.set_values_from_function(lambda x, y: 0.15 + 0.04 * x - 0.02 * y)

    return dh, u_k, u_n, p_k, dv, dp, v_test, q_test, dx(metadata={"q": 4})


def _eps(v):
    half = Constant(0.5)
    return half * (grad(v) + grad(v).T)


def _initialize_benchmark7_state(problem: dict[str, object]) -> None:
    dh = problem["dh"]
    v_coords = np.asarray(dh.get_dof_coords("v"), dtype=float)
    problem["v_k"].nodal_values[:] = 0.18 + 0.04 * v_coords[:, 0] - 0.03 * v_coords[:, 1]
    problem["v_n"].nodal_values[:] = -0.07 + 0.03 * v_coords[:, 0] + 0.02 * v_coords[:, 1]

    problem["p_k"].set_values_from_function(lambda x, y: 0.25 + 0.10 * x - 0.05 * y)
    problem["p_n"].set_values_from_function(lambda x, y: 0.12 + 0.04 * x - 0.03 * y)
    problem["vS_k"].set_values_from_function(
        lambda x, y: np.array([0.05 + 0.02 * x - 0.01 * y, -0.03 + 0.01 * x + 0.015 * y])
    )
    problem["vS_n"].set_values_from_function(
        lambda x, y: np.array([0.03 + 0.01 * x - 0.008 * y, -0.015 + 0.008 * x + 0.010 * y])
    )
    problem["u_k"].set_values_from_function(
        lambda x, y: np.array([0.015 + 0.010 * x * (1.0 - x), -0.008 + 0.006 * y * (1.0 - y / 1.5)])
    )
    problem["u_n"].set_values_from_function(
        lambda x, y: np.array([0.010 + 0.006 * x * (1.0 - x), -0.005 + 0.004 * y * (1.0 - y / 1.5)])
    )
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


def _assemble_dense_matrix(dh, form):
    matrix, _ = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend="python")
    return matrix.toarray()


def test_gateaux_scalar_cubic_matches_manual():
    _, dh, u_k, du, v, dx_q = _build_scalar_problem()
    one = Constant(1.0)
    three = Constant(3.0)
    residual_form = (u_k + u_k * u_k * u_k - one) * v * dx_q
    manual_jacobian = (one + three * u_k * u_k) * du * v * dx_q
    auto_jacobian = linearize_form(residual_form, u_k, du)

    J_manual = _assemble_dense_matrix(dh, manual_jacobian)
    J_auto = _assemble_dense_matrix(dh, auto_jacobian)
    np.testing.assert_allclose(J_auto, J_manual, rtol=1.0e-11, atol=1.0e-11)


def test_gateaux_positive_part_rejects_nonsmooth_dependency():
    _, _, u_k, du, v, dx_q = _build_scalar_problem()
    residual_form = pos_part(u_k) * v * dx_q
    with pytest.raises(NonDifferentiableGateauxNodeError):
        linearize_form(residual_form, u_k, du)


def test_gateaux_navier_stokes_style_matches_manual():
    dh, u_k, u_n, p_k, dv, dp, v_test, q_test, dx_q = _build_navier_like_problem()
    rho = Constant(1.1)
    mu = Constant(0.03)
    inv_dt = Constant(5.0)

    conv_k = dot(grad(u_k), u_k)
    dconv = dot(grad(dv), u_k) + dot(grad(u_k), dv)

    residual_form = (
        rho * dot((u_k - u_n) * inv_dt, v_test)
        + rho * dot(conv_k, v_test)
        + Constant(2.0) * mu * inner(_eps(u_k), _eps(v_test))
        - p_k * div(v_test)
        + q_test * div(u_k)
    ) * dx_q

    manual_jacobian = (
        rho * dot(dv * inv_dt, v_test)
        + rho * dot(dconv, v_test)
        + Constant(2.0) * mu * inner(_eps(dv), _eps(v_test))
        - dp * div(v_test)
        + q_test * div(dv)
    ) * dx_q

    auto_jacobian = linearize_form(residual_form, [u_k, p_k], [dv, dp])
    J_manual = _assemble_dense_matrix(dh, manual_jacobian)
    J_auto = _assemble_dense_matrix(dh, auto_jacobian)
    np.testing.assert_allclose(J_auto, J_manual, rtol=1.0e-10, atol=1.0e-10)


def test_newton_solver_accepts_auto_linearization():
    me, dh, u_k, du, v, dx_q = _build_scalar_problem()
    residual_form = (u_k + u_k * u_k * u_k - Constant(1.0)) * v * dx_q
    solver = NewtonSolver(
        residual_form,
        None,
        linearization_pairs=[(u_k, du)],
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(max_newton_iter=2, newton_tol=1.0e-8),
        lin_params=LinearSolverParameters(),
        quad_order=4,
        backend="python",
    )
    assert solver._jacobian_form is not None
    assert len(getattr(solver._jacobian_form, "integrals", [])) > 0


def test_gateaux_benchmark7_hdiv_pressure_block_matches_manual():
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
    manual_jacobian = test_p * (
        dcoeff * div(v_k)
        + coeff_k * div(dv)
        + dot(dgrad_coeff, v_k)
        + dot(grad_coeff_k, dv)
    ) * dx_q

    auto_jacobian = linearize_form(
        residual_form,
        [alpha_k, phi_k, v_k],
        [dalpha, dphi, dv],
    )

    J_manual = _assemble_dense_matrix(problem["dh"], manual_jacobian)
    J_auto = _assemble_dense_matrix(problem["dh"], auto_jacobian)
    np.testing.assert_allclose(J_auto, J_manual, rtol=1.0e-10, atol=1.0e-10)


def test_gateaux_benchmark7_full_coupled_form_matches_manual():
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

    names = [
        ("v_k", "dv"),
        ("p_k", "dp"),
        ("vS_k", "dvS"),
        ("u_k", "du"),
        ("phi_k", "dphi"),
        ("alpha_k", "dalpha"),
        ("mu_k", "dmu"),
        ("S_k", "dS"),
    ]
    coefficients = []
    directions = []
    for coeff_name, dir_name in names:
        coeff = problem.get(coeff_name)
        direction = problem.get(dir_name)
        if coeff is None or direction is None:
            continue
        coefficients.append(coeff)
        directions.append(direction)

    auto_jacobian = linearize_form(forms.residual_form, coefficients, directions)
    J_manual = _assemble_dense_matrix(problem["dh"], forms.jacobian_form)
    J_auto = _assemble_dense_matrix(problem["dh"], auto_jacobian)
    np.testing.assert_allclose(J_auto, J_manual, rtol=1.0e-9, atol=1.0e-9)
