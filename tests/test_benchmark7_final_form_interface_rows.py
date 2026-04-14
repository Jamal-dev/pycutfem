import numpy as np

from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import _build_forms, _create_problem
from examples.utils.biofilm.final_form import build_biofilm_one_domain_final_form
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.expressions import Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.utils.meshgen import structured_quad


def _build_final_form_problem(*, p_pore_shift: float = 0.0, phi_shift: float = 0.0):
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
    problem["p_pore_k"].set_values_from_function(lambda x, y: 0.25 - 0.04 * x + 0.08 * y + float(p_pore_shift))
    problem["p_pore_n"].set_values_from_function(lambda x, y: 0.18 - 0.02 * x + 0.05 * y + float(p_pore_shift))
    problem["alpha_k"].set_values_from_function(lambda x, y: 0.45 + 0.10 * x + 0.08 * y)
    problem["alpha_n"].set_values_from_function(lambda x, y: 0.40 + 0.08 * x + 0.06 * y)
    problem["phi_k"].set_values_from_function(lambda x, y: 0.32 + 0.04 * x - 0.03 * y + float(phi_shift))
    problem["phi_n"].set_values_from_function(lambda x, y: 0.30 + 0.03 * x - 0.02 * y + float(phi_shift))
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


def _assemble_block(problem, form, field: str) -> np.ndarray:
    _, residual = assemble_form(
        Equation(None, form),
        dof_handler=problem["dh"],
        bcs=[],
        backend="python",
    )
    sl = np.asarray(problem["dh"].get_field_slice(field), dtype=int)
    return np.asarray(residual, dtype=float)[sl]


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

    assert np.linalg.norm(_assemble_block(problem, mass_if, "mu_mass"), ord=np.inf) > 1.0e-10
    assert np.linalg.norm(_assemble_block(problem, normal_if, "mu_normal"), ord=np.inf) > 1.0e-10
    assert np.linalg.norm(_assemble_block(problem, tangential_if, "mu_tangent"), ord=np.inf) > 1.0e-10

    for field in ("v_x", "v_y", "phi", "rho_s"):
        assert np.linalg.norm(_assemble_block(problem, mass_if, field), ord=np.inf) < 1.0e-12
    for field in ("v_x", "v_y", "p", "p_pore", "u_x", "u_y"):
        assert np.linalg.norm(_assemble_block(problem, normal_if, field), ord=np.inf) < 1.0e-12
    for field in ("v_x", "v_y", "u_x", "u_y", "vS_x", "vS_y"):
        assert np.linalg.norm(_assemble_block(problem, tangential_if, field), ord=np.inf) < 1.0e-12


def test_final_form_free_fluid_bulk_rows_are_invariant_to_interface_pressure_shift_when_multipliers_zero() -> None:
    problem_a, forms_a = _build_final_form_problem()
    problem_b, forms_b = _build_final_form_problem(p_pore_shift=0.75, phi_shift=0.12)

    res_ax = _assemble_block(problem_a, forms_a.r_momentum, "v_x")
    res_ay = _assemble_block(problem_a, forms_a.r_momentum, "v_y")
    res_bx = _assemble_block(problem_b, forms_b.r_momentum, "v_x")
    res_by = _assemble_block(problem_b, forms_b.r_momentum, "v_y")

    np.testing.assert_allclose(res_ax, res_bx, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(res_ay, res_by, rtol=1.0e-12, atol=1.0e-12)


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
