"""Reduced deformation-only one-domain model used for Paper 1 verification.

This module implements the baseline system described in the manuscript:

  - fluid momentum,
  - one-domain mass constraint,
  - skeleton momentum,
  - Eulerian reference-map kinematics,
  - conservative Cahn--Hilliard transport for alpha.

Compared to `examples.utils.biofilm.one_domain`, this reduced model removes
porosity transport, substrate, detached biomass, growth, detachment, and damage.
It is the form builder that should be used for the Paper-1 verification program.

An optional Kelvin--Voigt skeleton viscosity `solid_visco_eta` is supported for
dynamic application-style benchmarks. It is off by default, so the exact and
manufactured-solution benchmarks remain on the purely elastic reduced model.
"""

from __future__ import annotations

from dataclasses import dataclass

from pycutfem.ufl.expressions import Constant, div, dot, grad, inner

from .one_domain import _as_constant, _c, _epsilon, _linear_elastic_term


def _W_prime(alpha):
    return _c(2.0) * alpha * (_c(1.0) - alpha) * (_c(1.0) - _c(2.0) * alpha)


def _W_second(alpha):
    return _c(2.0) - _c(12.0) * alpha + _c(12.0) * alpha * alpha


def _C(alpha, *, phi_b):
    return _c(1.0) - alpha * (_c(1.0) - phi_b)


def _B(alpha, *, phi_b):
    return alpha * (_c(1.0) - phi_b)


def _mu_mix(alpha, *, mu_f, mu_b):
    return (_c(1.0) - alpha) * mu_f + alpha * mu_b


def _div_weighted_scalar_times_vector(weight, grad_weight, vec):
    return weight * div(vec) + dot(grad_weight, vec)


def _vector_component(vec_expr, idx: int):
    try:
        return vec_expr[idx]
    except Exception:
        val = getattr(vec_expr, "value", None)
        if val is None:
            raise
        return _c(float(val[int(idx)]))


def _dot_2d_components(vec_expr, vec_test):
    return _vector_component(vec_expr, 0) * vec_test[0] + _vector_component(vec_expr, 1) * vec_test[1]


@dataclass(frozen=True)
class DeformationOnlyForms:
    jacobian_form: object
    residual_form: object
    r_momentum: object
    r_mass: object
    r_skeleton: object
    r_kinematics: object
    r_alpha: object
    r_mu_alpha: object
    a_momentum: object
    a_mass: object
    a_skeleton: object
    a_kinematics: object
    a_alpha: object
    a_mu_alpha: object


def build_deformation_only_forms(
    *,
    # unknowns at t_{n+1}
    v_k,
    p_k,
    vS_k,
    u_k,
    alpha_k,
    mu_alpha_k,
    # unknowns at t_n
    v_n,
    p_n,
    vS_n,
    u_n,
    alpha_n,
    mu_alpha_n=None,
    # Newton increments
    dv,
    dp,
    dvS,
    du,
    dalpha,
    dmu_alpha,
    # tests
    v_test,
    q_test,
    vS_test,
    u_test,
    alpha_test,
    mu_alpha_test,
    # measure and stepping
    dx,
    dt,
    theta: float = 1.0,
    # physical parameters
    rho_f=None,
    mu_f=None,
    mu_b=None,
    kappa_inv=None,
    mu_s=None,
    lambda_s=None,
    solid_visco_eta: float = 0.0,
    gamma_div: float = 0.0,
    phi_b: float = 0.5,
    M_alpha: float = 1.0,
    gamma_alpha: float = 1.0,
    eps_alpha: float = 1.0,
    # optional interface traction benchmark hook
    dGamma=None,
    g_t_k=None,
    g_t_n=None,
    traction_weight_k=None,
    traction_weight_n=None,
    # sources
    f_v=None,
    f_u=None,
    f_alpha=None,
) -> DeformationOnlyForms:
    if rho_f is None or mu_f is None or mu_b is None or kappa_inv is None:
        raise ValueError("rho_f, mu_f, mu_b, and kappa_inv are required.")
    if mu_s is None or lambda_s is None:
        raise ValueError("mu_s and lambda_s are required.")

    th = _c(float(theta))
    one_m_th = _c(1.0) - th
    inv_dt = _c(1.0) / dt

    phi_b_c = _c(float(phi_b))
    M_alpha_c = _c(float(M_alpha))
    gamma_alpha_c = _c(float(gamma_alpha))
    eps_alpha_c = _c(float(eps_alpha))

    zero_scalar = _c(0.0)
    zero_vector = Constant([0.0, 0.0], dim=1)
    f_v = f_v if f_v is not None else zero_vector
    f_u = f_u if f_u is not None else zero_vector
    f_alpha = f_alpha if f_alpha is not None else zero_scalar
    g_t_k = g_t_k if g_t_k is not None else zero_vector
    g_t_n = g_t_n if g_t_n is not None else g_t_k
    traction_weight_k = traction_weight_k if traction_weight_k is not None else zero_scalar
    traction_weight_n = traction_weight_n if traction_weight_n is not None else traction_weight_k

    # Frozen coefficients at time level n (matches the one-step analysis in the manuscript).
    C_n = _C(alpha_n, phi_b=phi_b_c)
    B_n = _B(alpha_n, phi_b=phi_b_c)
    gradC_n = -(_c(1.0) - phi_b_c) * grad(alpha_n)
    gradB_n = (_c(1.0) - phi_b_c) * grad(alpha_n)
    rho_n = rho_f * C_n
    mu_n = _mu_mix(alpha_n, mu_f=mu_f, mu_b=mu_b)
    beta_n = alpha_n * mu_f * kappa_inv

    v_th = th * v_k + one_m_th * v_n
    vS_th = th * vS_k + one_m_th * vS_n
    u_th = th * u_k + one_m_th * u_n
    alpha_th = th * alpha_k + one_m_th * alpha_n

    div_C_vtest = _div_weighted_scalar_times_vector(C_n, gradC_n, v_test)
    div_B_vStest = _div_weighted_scalar_times_vector(B_n, gradB_n, vS_test)
    d_div_C_vtest = zero_scalar
    d_div_B_vStest = zero_scalar

    div_C_vk = _div_weighted_scalar_times_vector(C_n, gradC_n, v_k)
    div_B_vSk = _div_weighted_scalar_times_vector(B_n, gradB_n, vS_k)
    div_C_dv = _div_weighted_scalar_times_vector(C_n, gradC_n, dv)
    div_B_dvS = _div_weighted_scalar_times_vector(B_n, gradB_n, dvS)

    # (i) Fluid momentum.
    r_mom = (rho_n * inv_dt) * (inner(v_k, v_test) - inner(v_n, v_test)) * dx
    r_mom += th * rho_n * dot(dot(grad(v_k), v_n), v_test) * dx
    r_mom += one_m_th * rho_n * dot(dot(grad(v_n), v_n), v_test) * dx
    r_mom += th * _c(2.0) * mu_n * inner(_epsilon(v_k), _epsilon(v_test)) * dx
    r_mom += one_m_th * _c(2.0) * mu_n * inner(_epsilon(v_n), _epsilon(v_test)) * dx
    r_mom += -p_k * div_C_vtest * dx
    r_mom += beta_n * (
        th * dot(v_k, v_test)
        + one_m_th * dot(v_n, v_test)
        - th * dot(vS_k, v_test)
        - one_m_th * dot(vS_n, v_test)
    ) * dx
    r_mom += -_dot_2d_components(f_v, v_test) * dx

    a_mom = (rho_n * inv_dt) * inner(dv, v_test) * dx
    a_mom += th * rho_n * dot(dot(grad(dv), v_n), v_test) * dx
    a_mom += th * _c(2.0) * mu_n * inner(_epsilon(dv), _epsilon(v_test)) * dx
    a_mom += -(dp * div_C_vtest) * dx
    a_mom += th * beta_n * (dot(dv, v_test) - dot(dvS, v_test)) * dx

    # (ii) One-domain mass constraint.
    r_mass = q_test * (div_C_vk + div_B_vSk) * dx
    a_mass = q_test * (div_C_dv + div_B_dvS) * dx

    # (iii) Skeleton momentum.
    r_skel = th * alpha_n * _linear_elastic_term(u_k, vS_test, mu_s=mu_s, lambda_s=lambda_s) * dx
    r_skel += one_m_th * alpha_n * _linear_elastic_term(u_n, vS_test, mu_s=mu_s, lambda_s=lambda_s) * dx
    r_skel += -(p_k * div_B_vStest) * dx
    r_skel += -beta_n * (
        th * dot(v_k, vS_test)
        + one_m_th * dot(v_n, vS_test)
        - th * dot(vS_k, vS_test)
        - one_m_th * dot(vS_n, vS_test)
    ) * dx
    r_skel += -alpha_n * _dot_2d_components(f_u, vS_test) * dx

    a_skel = th * alpha_n * _linear_elastic_term(du, vS_test, mu_s=mu_s, lambda_s=lambda_s) * dx
    a_skel += -(dp * div_B_vStest) * dx
    a_skel += -th * beta_n * (dot(dv, vS_test) - dot(dvS, vS_test)) * dx

    if float(solid_visco_eta) != 0.0:
        eta_s_c = _c(float(solid_visco_eta))
        sig_visc_k = _c(2.0) * eta_s_c * _epsilon(vS_k)
        sig_visc_n = _c(2.0) * eta_s_c * _epsilon(vS_n)
        r_visc_k = inner(sig_visc_k, grad(vS_test))
        r_visc_n = inner(sig_visc_n, grad(vS_test))
        r_skel += (th * alpha_n * r_visc_k + one_m_th * alpha_n * r_visc_n) * dx

        sig_dvisc = _c(2.0) * eta_s_c * _epsilon(dvS)
        a_skel += th * alpha_n * inner(sig_dvisc, grad(vS_test)) * dx

    # Optional equal-and-opposite interface traction transfer, used by the
    # FSI benchmark to inject a known tangential traction on the alpha=1/2 contour.
    if dGamma is not None:
        r_mom += -(th * _dot_2d_components(g_t_k, v_test) + one_m_th * _dot_2d_components(g_t_n, v_test)) * dGamma
        r_skel += (th * _dot_2d_components(g_t_k, vS_test) + one_m_th * _dot_2d_components(g_t_n, vS_test)) * dGamma
    if traction_weight_k is not None or traction_weight_n is not None:
        r_mom += -(
            th * traction_weight_k * _dot_2d_components(g_t_k, v_test)
            + one_m_th * traction_weight_n * _dot_2d_components(g_t_n, v_test)
        ) * dx
        r_skel += (
            th * traction_weight_k * _dot_2d_components(g_t_k, vS_test)
            + one_m_th * traction_weight_n * _dot_2d_components(g_t_n, vS_test)
        ) * dx
    if float(gamma_div) != 0.0:
        gamma_div_c = _as_constant(gamma_div)
        mass_res_k = div_C_vk + div_B_vSk
        d_mass_res_k = div_C_dv + div_B_dvS
        r_mom += gamma_div_c * mass_res_k * div_C_vtest * dx
        r_skel += gamma_div_c * mass_res_k * div_B_vStest * dx
        a_mom += gamma_div_c * (d_mass_res_k * div_C_vtest + mass_res_k * d_div_C_vtest) * dx
        a_skel += gamma_div_c * (d_mass_res_k * div_B_vStest + mass_res_k * d_div_B_vStest) * dx

    # (iv) Eulerian reference-map kinematics.
    r_kin = inv_dt * (inner(u_k, u_test) - inner(u_n, u_test)) * dx
    r_kin += th * dot(dot(grad(u_k), vS_n), u_test) * dx
    r_kin += one_m_th * dot(dot(grad(u_n), vS_n), u_test) * dx
    r_kin += -(th * dot(vS_k, u_test) + one_m_th * dot(vS_n, u_test)) * dx

    a_kin = inv_dt * inner(du, u_test) * dx
    a_kin += th * dot(dot(grad(du), vS_n), u_test) * dx
    a_kin += -th * dot(dvS, u_test) * dx

    # (v) Conservative Cahn--Hilliard transport for alpha.
    div_vS_n = div(vS_n)
    r_alpha = alpha_test * ((alpha_k - alpha_n) * inv_dt) * dx
    r_alpha += th * alpha_test * (dot(grad(alpha_k), vS_n) + alpha_k * div_vS_n) * dx
    r_alpha += one_m_th * alpha_test * (dot(grad(alpha_n), vS_n) + alpha_n * div_vS_n) * dx
    r_alpha += M_alpha_c * inner(grad(mu_alpha_k), grad(alpha_test)) * dx
    r_alpha += -alpha_test * f_alpha * dx

    a_alpha = alpha_test * (dalpha * inv_dt) * dx
    a_alpha += th * alpha_test * (dot(grad(dalpha), vS_n) + dalpha * div_vS_n) * dx
    a_alpha += M_alpha_c * inner(grad(dmu_alpha), grad(alpha_test)) * dx

    # (vi) Chemical potential relation.
    Wp_k = _W_prime(alpha_k)
    Wpp_k = _W_second(alpha_k)
    r_mu_alpha = mu_alpha_test * mu_alpha_k * dx
    r_mu_alpha += -(gamma_alpha_c * eps_alpha_c) * inner(grad(alpha_k), grad(mu_alpha_test)) * dx
    r_mu_alpha += -mu_alpha_test * ((gamma_alpha_c / eps_alpha_c) * Wp_k) * dx

    a_mu_alpha = mu_alpha_test * dmu_alpha * dx
    a_mu_alpha += -(gamma_alpha_c * eps_alpha_c) * inner(grad(dalpha), grad(mu_alpha_test)) * dx
    a_mu_alpha += -mu_alpha_test * ((gamma_alpha_c / eps_alpha_c) * Wpp_k * dalpha) * dx

    residual_form = r_mom + r_mass + r_skel + r_kin + r_alpha + r_mu_alpha
    jacobian_form = a_mom + a_mass + a_skel + a_kin + a_alpha + a_mu_alpha

    return DeformationOnlyForms(
        jacobian_form=jacobian_form,
        residual_form=residual_form,
        r_momentum=r_mom,
        r_mass=r_mass,
        r_skeleton=r_skel,
        r_kinematics=r_kin,
        r_alpha=r_alpha,
        r_mu_alpha=r_mu_alpha,
        a_momentum=a_mom,
        a_mass=a_mass,
        a_skeleton=a_skel,
        a_kinematics=a_kin,
        a_alpha=a_alpha,
        a_mu_alpha=a_mu_alpha,
    )
