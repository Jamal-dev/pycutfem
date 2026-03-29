import numpy as np
from scipy import sparse

from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import _create_problem
from examples.utils.biofilm.deformation_only import build_deformation_only_forms
from examples.utils.biofilm.one_domain import _c, build_biofilm_one_domain_forms
from pycutfem.ufl.expressions import Constant
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx


def _assemble_vector(*, problem, form, qdeg: int) -> np.ndarray:
    _, residual = assemble_form(
        Equation(None, form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=int(qdeg),
        backend="python",
    )
    return np.asarray(residual, dtype=float)


def _assemble_matrix(*, problem, form, qdeg: int) -> np.ndarray:
    matrix, _ = assemble_form(
        Equation(form, None),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=int(qdeg),
        backend="python",
    )
    if sparse.issparse(matrix):
        return matrix.toarray()
    return matrix.to_scipy().toarray()


def _build_reduced_problem():
    problem = _create_problem(
        Lx=1.0,
        Ly=0.25,
        nx=1,
        ny=1,
        poly_order=1,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=False,
    )

    rng = np.random.default_rng(9)
    for key in ("v_k", "v_n", "vS_k", "vS_n", "u_k", "u_n"):
        problem[key].nodal_values[:] = 1.0e-2 * rng.standard_normal(problem[key].nodal_values.shape)
    for key in ("p_k", "p_n", "mu_k", "mu_n"):
        problem[key].nodal_values[:] = 1.0e-2 * rng.standard_normal(problem[key].nodal_values.shape)

    alpha_k_vals = 0.2 + 0.6 * rng.random(problem["alpha_k"].nodal_values.shape)
    alpha_n_vals = 0.2 + 0.6 * rng.random(problem["alpha_n"].nodal_values.shape)
    problem["alpha_k"].nodal_values[:] = alpha_k_vals
    problem["alpha_n"].nodal_values[:] = alpha_n_vals
    return problem


def _build_reduced_forms(problem, *, qdeg: int):
    return build_deformation_only_forms(
        v_k=problem["v_k"],
        p_k=problem["p_k"],
        vS_k=problem["vS_k"],
        u_k=problem["u_k"],
        alpha_k=problem["alpha_k"],
        mu_alpha_k=problem["mu_k"],
        v_n=problem["v_n"],
        p_n=problem["p_n"],
        vS_n=problem["vS_n"],
        u_n=problem["u_n"],
        alpha_n=problem["alpha_n"],
        mu_alpha_n=problem["mu_n"],
        dv=problem["dv"],
        dp=problem["dp"],
        dvS=problem["dvS"],
        du=problem["du"],
        dalpha=problem["dalpha"],
        dmu_alpha=problem["dmu"],
        v_test=problem["v_test"],
        q_test=problem["q_test"],
        vS_test=problem["vS_test"],
        u_test=problem["u_test"],
        alpha_test=problem["alpha_test"],
        mu_alpha_test=problem["mu_test"],
        dx=dx(metadata={"q": int(qdeg)}),
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
        phi_b=0.18,
        M_alpha=1.0,
        gamma_alpha=1.0,
        eps_alpha=0.1,
        support_physics="internal_conversion",
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        fluid_convection="full",
        include_skeleton_acceleration=True,
        rho_s0_tilde=Constant(1.1),
        skeleton_inertia_convection="full",
        skeleton_pressure_mode="whole_domain",
    )


def _build_frozen_one_domain_forms(problem, *, qdeg: int):
    zero_scalar = _c(0.0)
    zero_vector = Constant([0.0, 0.0], dim=1)
    phi_b = _c(0.18)
    zero_alpha_trial = zero_scalar * problem["dalpha"]
    zero_alpha_test = zero_scalar * problem["alpha_test"]

    return build_biofilm_one_domain_forms(
        v_k=problem["v_k"],
        p_k=problem["p_k"],
        vS_k=problem["vS_k"],
        u_k=problem["u_k"],
        phi_k=phi_b,
        alpha_k=problem["alpha_k"],
        mu_alpha_k=problem["mu_k"],
        S_k=zero_scalar,
        v_n=problem["v_n"],
        p_n=problem["p_n"],
        vS_n=problem["vS_n"],
        u_n=problem["u_n"],
        phi_n=phi_b,
        alpha_n=problem["alpha_n"],
        mu_alpha_n=problem["mu_n"],
        S_n=zero_scalar,
        dv=problem["dv"],
        dp=problem["dp"],
        dvS=problem["dvS"],
        du=problem["du"],
        dphi=zero_alpha_trial,
        dalpha=problem["dalpha"],
        dmu_alpha=problem["dmu"],
        dS=zero_alpha_trial,
        v_test=problem["v_test"],
        q_test=problem["q_test"],
        vS_test=problem["vS_test"],
        u_test=problem["u_test"],
        phi_test=zero_alpha_test,
        alpha_test=problem["alpha_test"],
        mu_alpha_test=problem["mu_test"],
        S_test=zero_alpha_test,
        dx=dx(metadata={"q": int(qdeg)}),
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
        D_alpha=0.0,
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        support_physics="internal_conversion",
        alpha_interface_reg="none",
        alpha_mu_aux_pin=1.0,
        alpha_cahn_M=0.0,
        alpha_cahn_gamma=0.0,
        alpha_cahn_eps=0.1,
        alpha_ch_M=1.0,
        alpha_ch_gamma=1.0,
        alpha_ch_eps=0.1,
        alpha_ch_mobility="constant",
        mu_max=0.0,
        K_S=1.0,
        k_g=0.0,
        k_d=0.0,
        Y=1.0,
        rho_s_star=1.0,
        k_det=0.0,
        mu_b_model="mu",
        dim=2,
        include_skeleton_acceleration=True,
        rho_s0_tilde=Constant(1.1),
        skeleton_inertia_convection="full",
        fluid_convection="full",
        gamma_u=0.0,
        u_extension_mode="l2",
        gamma_u_pin=0.0,
        gamma_vS=None,
        vS_extension_mode=None,
        gamma_vS_pin=None,
        D_S=0.0,
        f_v=zero_vector,
        f_u=zero_vector,
        f_phi=zero_scalar,
        f_alpha=zero_scalar,
        f_S=zero_scalar,
        f_X=zero_scalar,
        D_det_prev=None,
        adhesion_a_prev=None,
        solid_visco_eta=0.0,
        gamma_div=0.0,
        drag_formulation="direct",
        skeleton_pressure_mode="whole_domain",
        dGamma=None,
        g_t_k=zero_vector,
        g_t_n=zero_vector,
        traction_weight_k=zero_scalar,
        traction_weight_n=zero_scalar,
    )


def test_deformation_only_matches_frozen_one_domain_internal_conversion_ch() -> None:
    qdeg = 4
    problem = _build_reduced_problem()
    reduced_forms = _build_reduced_forms(problem, qdeg=qdeg)
    frozen_forms = _build_frozen_one_domain_forms(problem, qdeg=qdeg)

    residual_names = (
        "residual_form",
        "r_skeleton",
        "r_kinematics",
        "r_alpha",
        "r_mu_alpha",
    )
    jacobian_names = (
        "jacobian_form",
        "a_skeleton",
        "a_kinematics",
        "a_alpha",
        "a_mu_alpha",
    )

    residual_diffs = {
        name: float(
            np.max(
                np.abs(
                    _assemble_vector(problem=problem, form=getattr(reduced_forms, name), qdeg=qdeg)
                    - _assemble_vector(problem=problem, form=getattr(frozen_forms, name), qdeg=qdeg)
                )
            )
        )
        for name in residual_names
    }
    jacobian_diffs = {
        name: float(
            np.max(
                np.abs(
                    _assemble_matrix(problem=problem, form=getattr(reduced_forms, name), qdeg=qdeg)
                    - _assemble_matrix(problem=problem, form=getattr(frozen_forms, name), qdeg=qdeg)
                )
            )
        )
        for name in jacobian_names
    }

    assert max(residual_diffs.values()) < 2.0e-11, residual_diffs
    assert max(jacobian_diffs.values()) < 1.0e-11, jacobian_diffs
