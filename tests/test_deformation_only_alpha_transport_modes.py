import numpy as np

from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import _build_bcs, _latent_inverse_array
from examples.utils.biofilm.deformation_only import build_deformation_only_forms
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, Function, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.expressions import TestFunction as UflTestFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _build_problem(
    *,
    alpha_advect_with: str,
    alpha_advection_form: str,
    support_physics: str = "legacy_exchange",
    phi_b: float = 0.5,
    fluid_velocity=None,
    fluid_velocity_k=None,
    skeleton_velocity=None,
    skeleton_velocity_k=None,
    include_skeleton_acceleration: bool = False,
    rho_s0_tilde: float = 0.0,
    skeleton_inertia_convection: str = "lagged",
    skeleton_pressure_mode: str = "whole_domain",
    alpha_biot: float | None = None,
    alpha_value: float = 1.0,
    alpha_profile=None,
    pressure_k_value: float = 0.0,
    pressure_n_value: float = 0.0,
    return_forms: bool = False,
):
    qdeg = 2
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
            "v_x": 2,
            "v_y": 2,
            "p": 1,
            "vS_x": 2,
            "vS_y": 2,
            "u_x": 2,
            "u_y": 2,
            "alpha": 1,
            "mu_alpha": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    dvS = VectorTrialFunction(space=VS, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    dmu = TrialFunction("mu_alpha", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = UflTestFunction("p", dof_handler=dh)
    alpha_test = UflTestFunction("alpha", dof_handler=dh)
    mu_test = UflTestFunction("mu_alpha", dof_handler=dh)

    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    vS_k = VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    mu_k = Function("mu_k", "mu_alpha", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    vS_n = VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    mu_n = Function("mu_n", "mu_alpha", dof_handler=dh)

    for vf in (v_k, v_n, vS_k, vS_n, u_k, u_n):
        vf.nodal_values[:] = 0.0
    for sf in (mu_k, mu_n):
        sf.nodal_values[:] = 0.0
    p_k.nodal_values[:] = float(pressure_k_value)
    p_n.nodal_values[:] = float(pressure_n_value)
    alpha_k.nodal_values[:] = float(alpha_value)
    alpha_n.nodal_values[:] = float(alpha_value)
    if alpha_profile is not None:
        alpha_k.set_values_from_function(alpha_profile)
        alpha_n.set_values_from_function(alpha_profile)

    if fluid_velocity is not None:
        v_n.set_values_from_function(fluid_velocity)
    if fluid_velocity_k is None:
        fluid_velocity_k = fluid_velocity
    if fluid_velocity_k is not None:
        v_k.set_values_from_function(fluid_velocity_k)
    if skeleton_velocity is None:
        # Divergent skeleton velocity: div(vS) = 0.3.
        skeleton_velocity = lambda x, y: np.array([0.1 * x, 0.2 * y])
    vS_n.set_values_from_function(skeleton_velocity)
    if skeleton_velocity_k is None:
        skeleton_velocity_k = skeleton_velocity
    if skeleton_velocity_k is not None:
        vS_k.set_values_from_function(skeleton_velocity_k)

    forms = build_deformation_only_forms(
        v_k=v_k,
        p_k=p_k,
        vS_k=vS_k,
        u_k=u_k,
        alpha_k=alpha_k,
        mu_alpha_k=mu_k,
        v_n=v_n,
        p_n=p_n,
        vS_n=vS_n,
        u_n=u_n,
        alpha_n=alpha_n,
        mu_alpha_n=mu_n,
        dv=dv,
        dp=dp,
        dvS=dvS,
        du=du,
        dalpha=dalpha,
        dmu_alpha=dmu,
        v_test=v_test,
        q_test=q_test,
        vS_test=vS_test,
        u_test=u_test,
        alpha_test=alpha_test,
        mu_alpha_test=mu_test,
        dx=dx(metadata={"q": int(qdeg)}),
        dt=Constant(0.1),
        theta=1.0,
        rho_f=Constant(1.0),
        mu_f=Constant(1.0e-2),
        mu_b=Constant(1.0e-2),
        kappa_inv=Constant(10.0),
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        phi_b=float(phi_b),
        M_alpha=0.0,
        gamma_alpha=1.0,
        eps_alpha=0.1,
        support_physics=str(support_physics),
        include_skeleton_acceleration=bool(include_skeleton_acceleration),
        rho_s0_tilde=Constant(float(rho_s0_tilde)),
        skeleton_inertia_convection=str(skeleton_inertia_convection),
        alpha_advect_with=str(alpha_advect_with),
        alpha_advection_form=str(alpha_advection_form),
        skeleton_pressure_mode=str(skeleton_pressure_mode),
        alpha_biot=alpha_biot,
    )

    if bool(return_forms):
        return dh, forms

    _, residual = assemble_form(
        Equation(None, forms.r_alpha),
        dof_handler=dh,
        bcs=[],
        quad_order=int(qdeg),
        backend="python",
    )
    return dh, np.asarray(residual, dtype=float)


def test_deformation_only_advective_alpha_keeps_uniform_indicator_under_divergent_skeleton_velocity() -> None:
    dh_adv, residual_adv = _build_problem(alpha_advect_with="vS", alpha_advection_form="advective")
    dh_cons, residual_cons = _build_problem(alpha_advect_with="vS", alpha_advection_form="conservative")

    alpha_slice_adv = np.asarray(dh_adv.get_field_slice("alpha"), dtype=int)
    alpha_slice_cons = np.asarray(dh_cons.get_field_slice("alpha"), dtype=int)

    assert np.linalg.norm(residual_adv[alpha_slice_adv], ord=np.inf) < 1.0e-12
    assert np.linalg.norm(residual_cons[alpha_slice_cons], ord=np.inf) > 1.0e-6


def test_deformation_only_biofilm_volume_conservative_alpha_uses_total_occupied_volume_flux() -> None:
    phi_b = 0.5
    skeleton_velocity = lambda x, y: np.array([0.1 * x, 0.2 * y])
    fluid_velocity = lambda x, y: np.array([-0.1 * x, -0.2 * y])

    dh_biofilm, residual_biofilm = _build_problem(
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative",
        phi_b=phi_b,
        fluid_velocity=fluid_velocity,
        skeleton_velocity=skeleton_velocity,
    )
    dh_vs, residual_vs = _build_problem(
        alpha_advect_with="vS",
        alpha_advection_form="conservative",
        phi_b=phi_b,
        fluid_velocity=fluid_velocity,
        skeleton_velocity=skeleton_velocity,
    )

    alpha_slice_biofilm = np.asarray(dh_biofilm.get_field_slice("alpha"), dtype=int)
    alpha_slice_vs = np.asarray(dh_vs.get_field_slice("alpha"), dtype=int)

    assert np.linalg.norm(residual_biofilm[alpha_slice_biofilm], ord=np.inf) < 1.0e-12
    assert np.linalg.norm(residual_vs[alpha_slice_vs], ord=np.inf) > 1.0e-6


def test_benchmark7_alpha_bc_auto_uses_natural_fluxes_in_deformation_only_mode() -> None:
    bcs_reduced = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=False,
        y_interface=1.0,
        eps_alpha=0.05,
        v_in=5.0,
        t_ramp=0.0,
        alpha_bc_mode="auto",
    )
    bcs_full = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=True,
        y_interface=1.0,
        eps_alpha=0.05,
        v_in=5.0,
        t_ramp=0.0,
        alpha_bc_mode="auto",
    )

    reduced_fields = [bc.field for bc in bcs_reduced]
    full_fields = [bc.field for bc in bcs_full]

    assert "alpha" not in reduced_fields
    assert "mu_alpha" not in reduced_fields
    assert "alpha" not in full_fields
    assert "mu_alpha" not in full_fields


def test_benchmark7_base_only_solid_bc_leaves_side_structure_unconstrained() -> None:
    bcs = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=True,
        y_interface=1.0,
        eps_alpha=0.05,
        v_in=5.0,
        t_ramp=0.0,
        alpha_bc_mode="auto",
        solid_bc_mode="base_only",
    )

    side_structure_fields = {
        (bc.field, bc.domain_tag)
        for bc in bcs
        if bc.domain_tag in {"left", "right"} and bc.field in {"vS_x", "vS_y", "u_x", "u_y"}
    }

    assert side_structure_fields == set()


def test_benchmark7_wall_normal_solid_bc_pins_only_side_normal_structure() -> None:
    bcs = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=True,
        y_interface=1.0,
        eps_alpha=0.05,
        v_in=5.0,
        t_ramp=0.0,
        alpha_bc_mode="auto",
        solid_bc_mode="wall_normal",
    )

    side_structure_fields = {
        (bc.field, bc.domain_tag)
        for bc in bcs
        if bc.domain_tag in {"left", "right"} and bc.field in {"vS_x", "vS_y", "u_x", "u_y"}
    }

    expected = {
        ("vS_x", "left"),
        ("u_x", "left"),
        ("vS_x", "right"),
        ("u_x", "right"),
    }
    assert side_structure_fields == expected


def test_benchmark7_lateral_clamped_solid_bc_pins_side_structure() -> None:
    bcs = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=True,
        y_interface=1.0,
        eps_alpha=0.05,
        v_in=5.0,
        t_ramp=0.0,
        alpha_bc_mode="auto",
        solid_bc_mode="lateral_clamped",
    )

    side_structure_fields = {
        (bc.field, bc.domain_tag)
        for bc in bcs
        if bc.domain_tag in {"left", "right"} and bc.field in {"vS_x", "vS_y", "u_x", "u_y"}
    }

    expected = {
        ("vS_x", "left"),
        ("vS_y", "left"),
        ("u_x", "left"),
        ("u_y", "left"),
        ("vS_x", "right"),
        ("vS_y", "right"),
        ("u_x", "right"),
        ("u_y", "right"),
    }
    assert side_structure_fields == expected


def test_benchmark7_alpha_solid_dirichlet_side_and_bottom_are_mirrored_to_latent_inverse_map() -> None:
    bcs = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=False,
        y_interface=1.0,
        eps_alpha=0.05,
        v_in=5.0,
        t_ramp=0.0,
        alpha_bc_mode="natural",
        alpha_solid_dirichlet_sides=True,
        alpha_solid_dirichlet_bottom=True,
        solid_bc_mode="base_only",
        latent_bounded_fields=("alpha",),
        latent_bounded_eps=1.0e-8,
        latent_bounded_map="sigmoid",
    )

    alpha_bcs = {
        (bc.field, bc.domain_tag): bc
        for bc in bcs
        if bc.method == "dirichlet" and bc.field in {"alpha", "alpha_latent"}
    }

    for tag in ("left", "right", "bottom"):
        assert ("alpha", tag) in alpha_bcs
        assert ("alpha_latent", tag) in alpha_bcs
        alpha_val = float(alpha_bcs[("alpha", tag)].value(0.3, 0.0, 0.0))
        latent_val = float(alpha_bcs[("alpha_latent", tag)].value(0.3, 0.0, 0.0))
        expected = float(_latent_inverse_array(np.asarray([1.0], dtype=float), eps=1.0e-8, map_kind="sigmoid")[0])
        assert alpha_val == 1.0
        assert latent_val == expected


def test_deformation_only_reduced_benchmark_honors_skeleton_inertia() -> None:
    vel_n = lambda x, y: np.array([0.2, -0.1])
    vel_k = lambda x, y: np.array([0.35, -0.05])

    dh_no_inertia, forms_no_inertia = _build_problem(
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        fluid_velocity=vel_n,
        fluid_velocity_k=vel_k,
        skeleton_velocity=vel_n,
        skeleton_velocity_k=vel_k,
        include_skeleton_acceleration=False,
        rho_s0_tilde=1.1,
        return_forms=True,
    )
    _, res_no_inertia = assemble_form(
        Equation(None, forms_no_inertia.r_skeleton),
        dof_handler=dh_no_inertia,
        bcs=[],
        quad_order=2,
        backend="python",
    )

    dh_inertia, forms_inertia = _build_problem(
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        fluid_velocity=vel_n,
        fluid_velocity_k=vel_k,
        skeleton_velocity=vel_n,
        skeleton_velocity_k=vel_k,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        return_forms=True,
    )
    _, res_inertia = assemble_form(
        Equation(None, forms_inertia.r_skeleton),
        dof_handler=dh_inertia,
        bcs=[],
        quad_order=2,
        backend="python",
    )

    vS_slice = np.asarray(list(dh_inertia.get_field_slice("vS_x")) + list(dh_inertia.get_field_slice("vS_y")), dtype=int)

    assert np.linalg.norm(np.asarray(res_no_inertia, dtype=float)[vS_slice], ord=np.inf) < 1.0e-12
    assert np.linalg.norm(np.asarray(res_inertia, dtype=float)[vS_slice], ord=np.inf) > 1.0e-6


def test_deformation_only_internal_conversion_drag_scales_with_reduced_solid_fraction() -> None:
    vel_fluid = lambda x, y: np.array([1.0, 0.0])
    vel_solid = lambda x, y: np.array([0.0, 0.0])
    phi_b = 0.18

    dh_legacy, forms_legacy = _build_problem(
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        support_physics="legacy_exchange",
        phi_b=phi_b,
        fluid_velocity=vel_fluid,
        fluid_velocity_k=vel_fluid,
        skeleton_velocity=vel_solid,
        skeleton_velocity_k=vel_solid,
        return_forms=True,
    )
    _, res_legacy = assemble_form(
        Equation(None, forms_legacy.r_skeleton),
        dof_handler=dh_legacy,
        bcs=[],
        quad_order=2,
        backend="python",
    )

    dh_cons, forms_cons = _build_problem(
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        support_physics="internal_conversion",
        phi_b=phi_b,
        fluid_velocity=vel_fluid,
        fluid_velocity_k=vel_fluid,
        skeleton_velocity=vel_solid,
        skeleton_velocity_k=vel_solid,
        return_forms=True,
    )
    _, res_cons = assemble_form(
        Equation(None, forms_cons.r_skeleton),
        dof_handler=dh_cons,
        bcs=[],
        quad_order=2,
        backend="python",
    )

    vS_slice = np.asarray(list(dh_cons.get_field_slice("vS_x")) + list(dh_cons.get_field_slice("vS_y")), dtype=int)
    legacy_norm = np.linalg.norm(np.asarray(res_legacy, dtype=float)[vS_slice], ord=np.inf)
    cons_norm = np.linalg.norm(np.asarray(res_cons, dtype=float)[vS_slice], ord=np.inf)
    expected_scale = (1.0 - phi_b) * (phi_b**2)

    assert legacy_norm > 1.0e-10
    assert cons_norm > 1.0e-10
    assert np.isclose(cons_norm / legacy_norm, expected_scale, rtol=1.0e-8, atol=1.0e-10)


def test_deformation_only_alpha_biot_uses_seboldt_divergence_coefficient() -> None:
    dh_std, forms_std = _build_problem(
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        support_physics="internal_conversion",
        phi_b=0.18,
        alpha_value=0.5,
        pressure_k_value=1.0,
        return_forms=True,
    )
    dh_biot, forms_biot = _build_problem(
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        support_physics="internal_conversion",
        phi_b=0.18,
        alpha_biot=1.0,
        alpha_value=0.5,
        pressure_k_value=1.0,
        return_forms=True,
    )

    def _assemble_pressure_norm(forms_obj, dh_obj):
        _, residual = assemble_form(
            Equation(None, forms_obj.r_skeleton_pressure),
            dof_handler=dh_obj,
            bcs=[],
            quad_order=2,
            backend="python",
        )
        vS_slice = np.asarray(list(dh_obj.get_field_slice("vS_x")) + list(dh_obj.get_field_slice("vS_y")), dtype=int)
        return float(np.linalg.norm(np.asarray(residual, dtype=float)[vS_slice], ord=np.inf))

    std_norm = _assemble_pressure_norm(forms_std, dh_std)
    biot_norm = _assemble_pressure_norm(forms_biot, dh_biot)
    expected_scale = 1.0 / (1.0 - 0.18)

    assert forms_std.r_skeleton_pressure is not None
    assert forms_biot.r_skeleton_pressure is not None
    assert std_norm > 1.0e-10
    assert biot_norm > 1.0e-10
    assert np.isclose(biot_norm / std_norm, expected_scale, rtol=1.0e-8, atol=1.0e-10)


def test_deformation_only_seboldt_pressure_mode_removes_diffuse_gradB_traction() -> None:
    alpha_profile = lambda x, y: 0.2 + 0.3 * x
    dh_whole, forms_whole = _build_problem(
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        support_physics="internal_conversion",
        phi_b=0.18,
        skeleton_pressure_mode="whole_domain",
        alpha_biot=1.0,
        alpha_profile=alpha_profile,
        pressure_k_value=1.0,
        return_forms=True,
    )
    dh_seb, forms_seb = _build_problem(
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        support_physics="internal_conversion",
        phi_b=0.18,
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        alpha_profile=alpha_profile,
        pressure_k_value=1.0,
        return_forms=True,
    )

    _, res_whole = assemble_form(
        Equation(None, forms_whole.r_skeleton_pressure),
        dof_handler=dh_whole,
        bcs=[],
        quad_order=2,
        backend="python",
    )
    _, res_seb = assemble_form(
        Equation(None, forms_seb.r_skeleton_pressure),
        dof_handler=dh_seb,
        bcs=[],
        quad_order=2,
        backend="python",
    )

    vS_slice = np.asarray(list(dh_whole.get_field_slice("vS_x")) + list(dh_whole.get_field_slice("vS_y")), dtype=int)
    whole_norm = float(np.linalg.norm(np.asarray(res_whole, dtype=float)[vS_slice], ord=np.inf))
    seb_norm = float(np.linalg.norm(np.asarray(res_seb, dtype=float)[vS_slice], ord=np.inf))

    assert forms_whole.r_skeleton_pressure is not None
    assert forms_seb.r_skeleton_pressure is not None
    assert whole_norm > seb_norm
