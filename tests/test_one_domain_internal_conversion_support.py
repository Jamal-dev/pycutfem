import numpy as np
import pytest

from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import _create_problem
from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms
from pycutfem.ufl.expressions import Constant
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import ds, dx


def _build_forms(
    *,
    support_physics: str,
    alpha_advect_with: str,
    alpha_advection_form: str,
    alpha0: float = 0.5,
    phi0: float = 0.25,
    S0: float = 1.0,
    mu_max: float = 2.0,
    K_S: float = 1.0,
    k_g: float = 0.0,
    k_d: float = 0.0,
    k_det: float = 0.0,
    nx: int = 1,
    ny: int = 1,
    phi_supg: float = 0.0,
    phi_cip: float = 0.0,
    gamma_phi: float = 0.0,
    s_v=None,
    ds_v=None,
):
    problem = _create_problem(
        Lx=1.0,
        Ly=1.0,
        nx=int(nx),
        ny=int(ny),
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
    )

    for vf in (problem["v_k"], problem["v_n"], problem["vS_k"], problem["vS_n"], problem["u_k"], problem["u_n"]):
        vf.nodal_values[:] = 0.0
    for sf in (problem["p_k"], problem["p_n"], problem["mu_k"], problem["mu_n"]):
        sf.nodal_values[:] = 0.0
    for sf in (problem["alpha_k"], problem["alpha_n"]):
        sf.nodal_values[:] = float(alpha0)
    for sf in (problem["phi_k"], problem["phi_n"]):
        sf.nodal_values[:] = float(phi0)
    for sf in (problem["S_k"], problem["S_n"]):
        sf.nodal_values[:] = float(S0)

    forms = build_biofilm_one_domain_forms(
        v_k=problem["v_k"],
        p_k=problem["p_k"],
        vS_k=problem["vS_k"],
        u_k=problem["u_k"],
        phi_k=problem["phi_k"],
        alpha_k=problem["alpha_k"],
        mu_alpha_k=problem["mu_k"],
        S_k=problem["S_k"],
        v_n=problem["v_n"],
        p_n=problem["p_n"],
        vS_n=problem["vS_n"],
        u_n=problem["u_n"],
        phi_n=problem["phi_n"],
        alpha_n=problem["alpha_n"],
        mu_alpha_n=problem["mu_n"],
        S_n=problem["S_n"],
        dv=problem["dv"],
        dp=problem["dp"],
        dvS=problem["dvS"],
        du=problem["du"],
        dphi=problem["dphi"],
        dalpha=problem["dalpha"],
        dmu_alpha=problem["dmu"],
        dS=problem["dS"],
        v_test=problem["v_test"],
        q_test=problem["q_test"],
        vS_test=problem["vS_test"],
        u_test=problem["u_test"],
        phi_test=problem["phi_test"],
        alpha_test=problem["alpha_test"],
        mu_alpha_test=problem["mu_test"],
        S_test=problem["S_test"],
        dx=dx(metadata={"q": 6}),
        ds_cip=ds(metadata={"q": 6}),
        dt=Constant(0.1),
        theta=1.0,
        rho_f=Constant(1.0),
        mu_f=Constant(0.035),
        mu_b=Constant(0.035),
        kappa_inv=Constant(1.0e5),
        mu_s=Constant(1.67785e5),
        lambda_s=Constant(8.22148e6),
        solid_model="linear",
        kappa_inv_model="refmap",
        D_phi=0.0,
        phi_diffusion_weight="fluid",
        gamma_phi=float(gamma_phi),
        phi_supg=float(phi_supg),
        phi_cip=float(phi_cip),
        D_alpha=0.0,
        alpha_interface_reg="none",
        alpha_mu_aux_pin=1.0,
        alpha_advect_with=str(alpha_advect_with),
        alpha_advection_form=str(alpha_advection_form),
        support_physics=str(support_physics),
        alpha_ch_M=0.0,
        alpha_ch_gamma=0.0,
        alpha_ch_eps=0.05,
        mu_max=float(mu_max),
        K_S=float(K_S),
        k_g=float(k_g),
        k_d=float(k_d),
        Y=1.0,
        rho_s_star=1.0,
        k_det=float(k_det),
        gamma_u=1.0,
        u_extension_mode="grad",
        gamma_u_pin=1.0e-6,
        vS_cip=0.0,
        gamma_vS=1.0,
        vS_extension_mode="grad",
        gamma_vS_pin=1.0e-6,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        skeleton_inertia_convection="lagged",
        fluid_convection="off",
        s_v=s_v,
        ds_v=ds_v,
    )
    return problem, forms


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


def test_internal_conversion_alpha_requires_biofilm_support_flux() -> None:
    with pytest.raises(ValueError, match="biofilm_volume"):
        _build_forms(
            support_physics="internal_conversion",
            alpha_advect_with="mix",
            alpha_advection_form="conservative_weak",
            mu_max=0.0,
        )


def test_internal_conversion_rejects_phi_diffusion_regularization() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.0,
        nx=1,
        ny=1,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
    )
    for vf in (problem["v_k"], problem["v_n"], problem["vS_k"], problem["vS_n"], problem["u_k"], problem["u_n"]):
        vf.nodal_values[:] = 0.0
    for sf in (problem["p_k"], problem["p_n"], problem["mu_k"], problem["mu_n"], problem["alpha_k"], problem["alpha_n"]):
        sf.nodal_values[:] = 0.5
    for sf in (problem["phi_k"], problem["phi_n"]):
        sf.nodal_values[:] = 0.25
    for sf in (problem["S_k"], problem["S_n"]):
        sf.nodal_values[:] = 0.0

    with pytest.raises(ValueError, match="D_phi"):
        build_biofilm_one_domain_forms(
            v_k=problem["v_k"],
            p_k=problem["p_k"],
            vS_k=problem["vS_k"],
            u_k=problem["u_k"],
            phi_k=problem["phi_k"],
            alpha_k=problem["alpha_k"],
            mu_alpha_k=problem["mu_k"],
            S_k=problem["S_k"],
            v_n=problem["v_n"],
            p_n=problem["p_n"],
            vS_n=problem["vS_n"],
            u_n=problem["u_n"],
            phi_n=problem["phi_n"],
            alpha_n=problem["alpha_n"],
            mu_alpha_n=problem["mu_n"],
            S_n=problem["S_n"],
            dv=problem["dv"],
            dp=problem["dp"],
            dvS=problem["dvS"],
            du=problem["du"],
            dphi=problem["dphi"],
            dalpha=problem["dalpha"],
            dmu_alpha=problem["dmu"],
            dS=problem["dS"],
            v_test=problem["v_test"],
            q_test=problem["q_test"],
            vS_test=problem["vS_test"],
            u_test=problem["u_test"],
            phi_test=problem["phi_test"],
            alpha_test=problem["alpha_test"],
            mu_alpha_test=problem["mu_test"],
            S_test=problem["S_test"],
            dx=dx(metadata={"q": 6}),
            dt=Constant(0.1),
            theta=1.0,
            rho_f=Constant(1.0),
            mu_f=Constant(0.035),
            mu_b=Constant(0.035),
            kappa_inv=Constant(1.0e5),
            mu_s=Constant(1.67785e5),
            lambda_s=Constant(8.22148e6),
            solid_model="linear",
            kappa_inv_model="refmap",
            D_phi=1.0e-6,
            gamma_phi=5.0,
            alpha_advect_with="biofilm_volume",
            alpha_advection_form="conservative_weak",
            support_physics="internal_conversion",
        )


def test_internal_conversion_rejects_phi_cip_stabilization() -> None:
    with pytest.raises(ValueError, match="SUPG stabilization"):
        _build_forms(
            support_physics="internal_conversion",
            alpha_advect_with="biofilm_volume",
            alpha_advection_form="conservative_weak",
            alpha0=0.6,
            phi0=0.25,
            mu_max=0.0,
            k_d=0.0,
            nx=2,
            ny=2,
            phi_cip=1.0,
            gamma_phi=0.0,
        )


def test_internal_conversion_rejects_nonzero_s_v() -> None:
    with pytest.raises(ValueError, match="requires s_v=0"):
        _build_forms(
            support_physics="internal_conversion",
            alpha_advect_with="biofilm_volume",
            alpha_advection_form="conservative_weak",
            alpha0=0.6,
            phi0=0.25,
            mu_max=0.0,
            s_v=Constant(1.0),
        )


def test_internal_conversion_phi_uses_single_growth_localization_in_B_equation() -> None:
    alpha0 = 0.5
    problem_legacy, forms_legacy = _build_forms(
        support_physics="legacy_exchange",
        alpha_advect_with="vS",
        alpha_advection_form="conservative",
        alpha0=alpha0,
        phi0=0.25,
        mu_max=2.0,
        k_g=0.0,
        k_d=0.0,
    )
    problem_internal, forms_internal = _build_forms(
        support_physics="internal_conversion",
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        alpha0=alpha0,
        phi0=0.25,
        mu_max=2.0,
        k_g=0.0,
        k_d=0.0,
    )

    legacy_norm = np.linalg.norm(_assemble_block(problem_legacy, forms_legacy.r_phi, "phi"), ord=np.inf)
    internal_norm = np.linalg.norm(_assemble_block(problem_internal, forms_internal.r_phi, "phi"), ord=np.inf)

    assert legacy_norm > 0.0
    assert internal_norm > legacy_norm
    assert np.isclose(internal_norm / legacy_norm, 1.0 / alpha0, rtol=1.0e-10, atol=1.0e-12)


def test_internal_conversion_B_supg_stabilization_is_consistent_on_conservative_state() -> None:
    problem, forms = _build_forms(
        support_physics="internal_conversion",
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        alpha0=0.6,
        phi0=0.25,
        mu_max=0.0,
        k_d=0.0,
        nx=2,
        ny=2,
        phi_supg=1.0,
        gamma_phi=0.0,
    )
    stream = lambda x, y: np.array(
        [
            np.asarray(x, dtype=float) * (1.0 - np.asarray(x, dtype=float)) * (1.0 - 2.0 * np.asarray(y, dtype=float)),
            -(1.0 - 2.0 * np.asarray(x, dtype=float)) * np.asarray(y, dtype=float) * (1.0 - np.asarray(y, dtype=float)),
        ]
    )
    problem["vS_k"].set_values_from_function(stream)
    problem["vS_n"].set_values_from_function(stream)

    phi_res = _assemble_block(problem, forms.r_phi, "phi")
    assert np.linalg.norm(phi_res, ord=np.inf) < 1.0e-11


def test_internal_conversion_alpha_has_no_growth_source_from_internal_conversion() -> None:
    problem, forms = _build_forms(
        support_physics="internal_conversion",
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        alpha0=0.6,
        phi0=0.25,
        mu_max=2.0,
        k_g=0.0,
        k_d=0.1,
    )
    alpha_res = _assemble_block(problem, forms.r_alpha, "alpha")
    assert np.linalg.norm(alpha_res, ord=np.inf) < 1.0e-12


def test_internal_conversion_drag_vanishes_when_B_goes_to_zero() -> None:
    problem, forms = _build_forms(
        support_physics="internal_conversion",
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        alpha0=0.6,
        phi0=1.0,
        mu_max=0.0,
        k_d=0.0,
    )
    upward = lambda x, y: np.array([0.0, 1.0])
    problem["v_k"].set_values_from_function(upward)
    problem["v_n"].set_values_from_function(upward)

    drag_res = _assemble_block(problem, forms.r_momentum_terms["drag"], "v_x")
    drag_res_y = _assemble_block(problem, forms.r_momentum_terms["drag"], "v_y")
    skel_res_x = _assemble_block(problem, forms.r_skeleton, "vS_x")
    skel_res_y = _assemble_block(problem, forms.r_skeleton, "vS_y")

    assert np.linalg.norm(drag_res, ord=np.inf) < 1.0e-12
    assert np.linalg.norm(drag_res_y, ord=np.inf) < 1.0e-12
    assert np.linalg.norm(skel_res_x, ord=np.inf) < 1.0e-12
    assert np.linalg.norm(skel_res_y, ord=np.inf) < 1.0e-12
