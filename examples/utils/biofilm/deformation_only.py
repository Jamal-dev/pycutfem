"""Reduced deformation-only one-domain model used for Paper 1 verification.

This module implements the baseline system described in the manuscript:

  - fluid momentum,
  - one-domain mass constraint,
  - skeleton momentum,
  - Eulerian reference-map kinematics,
  - alpha transport with Cahn--Hilliard regularization.

Compared to `examples.utils.biofilm.one_domain`, this reduced model removes
substrate, detached biomass, growth, detachment, and damage. It supports two
transport closures:

  - the historical frozen-`phi_b` reduction, where `B = alpha (1-phi_b)` is
    derived from `alpha`, and
  - an optional native `alpha-B` transport mode for the conserved-support
    `internal_conversion` branch, where `B` is an actual unknown and
    `phi = (alpha-B)/alpha` is derived from the reduced state when needed.

It is the form builder that should be used for the Paper-1 verification program.

An optional Kelvin--Voigt skeleton viscosity `solid_visco_eta` is supported for
dynamic application-style benchmarks. It is off by default, so the exact and
manufactured-solution benchmarks remain on the purely elastic reduced model.

The reduced model supports multiple alpha transport interpretations. For the
Paper-1 one-domain deformation benchmarks, the physically consistent biofilm-
support law is

  - `alpha_advect_with="biofilm_volume"`
  - `alpha_advection_form="conservative_weak"`

which transports alpha with the occupied-support flux

  J_alpha = alpha [phi_b v + (1-phi_b) vS] = P v + B vS,
  P=alpha phi_b,   B=alpha (1-phi_b),

in weak conservative form. This is the option that preserves the total alpha
measure up to boundary flux. Legacy indicator-style options such as
`advective + vS` remain available for debugging and comparison studies.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pycutfem.ufl.expressions import Constant, FacetNormal, Identity, div, dot, grad, inner

from .one_domain import (
    _as_constant,
    _c,
    _epsilon,
    _linear_deviatoric_elastic_term,
    _linear_elastic_term,
    _one_minus,
    _support_physics_key,
)
from ..shared.nonlinear_solid_refmap import deulerian_k_inv, dsigma_neo_hookean, eulerian_k_inv, sigma_neo_hookean


def _W_prime(alpha):
    return _c(2.0) * alpha * (_c(1.0) - alpha) * (_c(1.0) - _c(2.0) * alpha)


def _W_second(alpha):
    return _c(2.0) - _c(12.0) * alpha + _c(12.0) * alpha * alpha


def _C(alpha, *, phi_b):
    return _c(1.0) - alpha * (_c(1.0) - phi_b)


def _B(alpha, *, phi_b):
    return alpha * (_c(1.0) - phi_b)


def _C_from_B(B):
    return _c(1.0) - B


def _P_from_alpha_B(alpha, B):
    return alpha - B


def _phi_sq_from_alpha_B(alpha, B, *, eps: float = 1.0e-12):
    P = _P_from_alpha_B(alpha, B)
    denom = alpha * alpha + _c(float(eps))
    return (P * P) / denom


def _d_phi_sq_from_alpha_B(alpha, B, dalpha, dB, *, eps: float = 1.0e-12):
    P = _P_from_alpha_B(alpha, B)
    dP = dalpha - dB
    denom = alpha * alpha + _c(float(eps))
    ddenom = _c(2.0) * alpha * dalpha
    return ((_c(2.0) * P * dP) * denom - (P * P) * ddenom) / (denom * denom)


def _mu_mix(alpha, *, mu_f, mu_b):
    return (_c(1.0) - alpha) * mu_f + alpha * mu_b


def _div_weighted_scalar_times_vector(weight, grad_weight, vec):
    return weight * div(vec) + dot(grad_weight, vec)


def _vector_component(vec_expr, idx: int):
    try:
        return vec_expr[idx]
    except Exception:
        try:
            basis = Constant([1.0, 0.0] if int(idx) == 0 else [0.0, 1.0], dim=1)
            return dot(vec_expr, basis)
        except Exception:
            pass
        val = getattr(vec_expr, "value", None)
        if val is None:
            raise
        return _c(float(val[int(idx)]))


def _dot_2d_components(vec_expr, vec_test):
    return _vector_component(vec_expr, 0) * _vector_component(vec_test, 0) + _vector_component(vec_expr, 1) * _vector_component(vec_test, 1)


def _weighted_dot_2d_components(weight, vec_expr, vec_test):
    return (
        weight * _vector_component(vec_expr, 0) * _vector_component(vec_test, 0)
        + weight * _vector_component(vec_expr, 1) * _vector_component(vec_test, 1)
    )


def _matvec_2d_components(mat_expr, vec_expr):
    return (
        mat_expr[0, 0] * _vector_component(vec_expr, 0) + mat_expr[0, 1] * _vector_component(vec_expr, 1),
        mat_expr[1, 0] * _vector_component(vec_expr, 0) + mat_expr[1, 1] * _vector_component(vec_expr, 1),
    )


def _alpha_band(alpha):
    return _c(4.0) * alpha * (_c(1.0) - alpha)


def _grad_alpha_band(alpha):
    return _c(4.0) * (_c(1.0) - _c(2.0) * alpha) * grad(alpha)


def _d_alpha_band(alpha, dalpha):
    return _c(4.0) * (_c(1.0) - _c(2.0) * alpha) * dalpha


def _d_grad_alpha_band(alpha, dalpha):
    return _c(4.0) * ((_c(1.0) - _c(2.0) * alpha) * grad(dalpha) - _c(2.0) * dalpha * grad(alpha))


@dataclass(frozen=True)
class DeformationOnlyForms:
    jacobian_form: object
    residual_form: object
    r_momentum: object
    r_mass: object
    r_skeleton: object
    r_kinematics: object
    r_alpha: object
    r_B: object | None
    r_mu_alpha: object
    a_momentum: object
    a_mass: object
    a_skeleton: object
    a_kinematics: object
    a_alpha: object
    a_B: object | None
    a_mu_alpha: object
    r_drag_lambda: object | None = None
    a_drag_lambda: object | None = None
    r_skeleton_pressure: object | None = None
    a_skeleton_pressure: object | None = None
    r_volumetric: object | None = None
    a_volumetric: object | None = None


def build_deformation_only_forms(
    *,
    # unknowns at t_{n+1}
    v_k,
    p_k,
    vS_k,
    u_k,
    alpha_k,
    B_k=None,
    mu_alpha_k,
    lambda_drag_k=None,
    # unknowns at t_n
    v_n,
    p_n,
    vS_n,
    u_n,
    alpha_n,
    B_n=None,
    mu_alpha_n=None,
    lambda_drag_n=None,
    # Newton increments
    dv,
    dp,
    dvS,
    du,
    dalpha,
    dB=None,
    dmu_alpha,
    dlambda_drag=None,
    # tests
    v_test,
    q_test,
    vS_test,
    u_test,
    alpha_test,
    B_test=None,
    mu_alpha_test,
    lambda_drag_test=None,
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
    solid_model: str = "linear",
    c_nh=None,
    beta_nh=None,
    kappa_inv_model: str = "spatial",
    rho_s0_tilde=None,
    include_skeleton_acceleration: bool = False,
    skeleton_inertia_convection: str = "lagged",
    solid_visco_eta: float = 0.0,
    gamma_div: float = 0.0,
    drag_formulation: str = "direct",
    phi_b: float = 0.5,
    M_alpha: float = 1.0,
    gamma_alpha: float = 1.0,
    eps_alpha: float = 1.0,
    alpha_mu_aux_pin: float = 1.0,
    kinematics_scale=None,
    # The reduced verification model has no separate phi equation. The intended
    # physics is therefore the support-preserving internal-conversion limit. If
    # (B_k, B_n, dB, B_test) are provided, the reduced model uses the native
    # alpha-B transport closure. Otherwise it falls back to the historical
    # frozen-phi_b reduction with B=alpha(1-phi_b).
    support_physics: str = "internal_conversion",
    # Recommended support-preserving alpha transport for the reduced one-domain
    # model is `alpha_advect_with="biofilm_volume"` with
    # `alpha_advection_form="conservative_weak"`. These are the defaults so
    # new reduced-model runs start from the conserved biofilm-support law.
    alpha_advect_with: str = "biofilm_volume",
    alpha_advection_form: str = "conservative_weak",
    fluid_convection: str = "full",
    # optional interface traction benchmark hook
    dGamma=None,
    g_t_k=None,
    g_t_n=None,
    traction_weight_k=None,
    traction_weight_n=None,
    dpi_s=None,
    pi_s_test=None,
    pi_s_k=None,
    pi_s_n=None,
    solid_volumetric_split: bool = False,
    solid_volumetric_penalty: float = 1.0,
    # Optional consistent lift of the mixed volumetric relation into the mass
    # row. Keeping the original pi_s row and adding a scaled copy of it to the
    # q-test equation preserves the solution set while making pi_s participate
    # in the pressure block.
    pressure_block_lift_scale: float = 0.0,
    # Skeleton pressure coupling mode:
    # - "whole_domain" keeps the diffuse one-domain split -(p, div(B eta))
    # - "seboldt" uses the sharp Biot term -(alpha_biot * alpha * p, div(eta))
    skeleton_pressure_mode: str = "whole_domain",
    # Optional Biot-Willis coefficient for the sharp Seboldt pressure mode.
    alpha_biot: float | None = None,
    ds_alpha_transport=None,
    ds_B_transport=None,
    # sources
    f_v=None,
    f_u=None,
    f_alpha=None,
    f_B=None,
) -> DeformationOnlyForms:
    if rho_f is None or mu_f is None or mu_b is None or kappa_inv is None:
        raise ValueError("rho_f, mu_f, mu_b, and kappa_inv are required.")
    if mu_s is None or lambda_s is None:
        raise ValueError("mu_s and lambda_s are required.")
    solid_model_key = str(solid_model).strip().lower().replace("-", "_")
    if solid_model_key not in {"linear", "small_strain", "linear_elastic", "neo_hookean", "nh"}:
        raise ValueError(f"Unsupported deformation-only solid model {solid_model!r}.")
    total_pressure_ref = None
    if bool(solid_volumetric_split):
        if solid_model_key not in {"linear", "small_strain", "linear_elastic"}:
            raise ValueError("solid_volumetric_split is currently implemented only for solid_model='linear'.")
        if pi_s_k is None or pi_s_n is None or dpi_s is None or pi_s_test is None:
            raise ValueError("solid_volumetric_split requires (pi_s_k, pi_s_n, dpi_s, pi_s_test).")
        total_pressure_ref = 2.0 * float(mu_s) + 2.0 * float(lambda_s)
        if not np.isfinite(total_pressure_ref) or total_pressure_ref <= 0.0:
            raise ValueError("solid_volumetric_split requires a positive total-pressure reference scale.")

    th = _as_constant(theta)
    one_m_th = _c(1.0) - th
    inv_dt = _c(1.0) / dt

    phi_b_c = _as_constant(phi_b)
    M_alpha_c = _as_constant(M_alpha)
    gamma_alpha_c = _as_constant(gamma_alpha)
    eps_alpha_c = _as_constant(eps_alpha)
    pressure_block_lift_scale_c = _as_constant(float(pressure_block_lift_scale))
    total_pressure_ref_c = _as_constant(1.0 if total_pressure_ref is None else float(total_pressure_ref))
    total_pressure_ref_inv_c = _as_constant(1.0 if total_pressure_ref is None else (1.0 / float(total_pressure_ref)))
    lambda_s_over_total_pressure_ref_c = _as_constant(1.0 if total_pressure_ref is None else (float(lambda_s) / float(total_pressure_ref)))
    ch_enabled = float(M_alpha) != 0.0 and float(gamma_alpha) != 0.0

    zero_scalar = _c(0.0)
    zero_vector = Constant([0.0, 0.0], dim=1)
    f_v = f_v if f_v is not None else zero_vector
    f_u = f_u if f_u is not None else zero_vector
    f_alpha = f_alpha if f_alpha is not None else zero_scalar
    f_B = f_B if f_B is not None else zero_scalar
    g_t_k = g_t_k if g_t_k is not None else zero_vector
    g_t_n = g_t_n if g_t_n is not None else g_t_k
    traction_weight_k = traction_weight_k if traction_weight_k is not None else zero_scalar
    traction_weight_n = traction_weight_n if traction_weight_n is not None else traction_weight_k
    r_skel_pressure = None
    a_skel_pressure = None
    r_volumetric = None
    a_volumetric = None

    use_B_transport = any(val is not None for val in (B_k, B_n, dB, B_test))
    if use_B_transport and not all(val is not None for val in (B_k, B_n, dB, B_test)):
        raise ValueError("alpha-B reduced transport requires (B_k, B_n, dB, B_test) to be provided together.")

    # Reduced coefficients. The historical mode derives B from frozen phi_b;
    # the alpha-B mode treats B as a true unknown and derives C=1-B, P=alpha-B.
    if use_B_transport:
        C_n = _C_from_B(B_n)
        C_k = _C_from_B(B_k)
        P_n = _P_from_alpha_B(alpha_n, B_n)
        P_k = _P_from_alpha_B(alpha_k, B_k)
        gradC_n = -grad(B_n)
        gradC_k = -grad(B_k)
        gradB_n = grad(B_n)
        gradB_k = grad(B_k)
        gradP_n = grad(alpha_n) - grad(B_n)
        gradP_k = grad(alpha_k) - grad(B_k)
        dC_k = -dB
        grad_dC_k = -grad(dB)
        dB_k = dB
        grad_dB_k = grad(dB)
        dP_k = dalpha - dB
        grad_dP_k = grad(dalpha) - grad(dB)
    else:
        C_n = _C(alpha_n, phi_b=phi_b_c)
        B_n = _B(alpha_n, phi_b=phi_b_c)
        C_k = _C(alpha_k, phi_b=phi_b_c)
        B_k = _B(alpha_k, phi_b=phi_b_c)
        P_n = None
        P_k = None
        gradC_n = -(_c(1.0) - phi_b_c) * grad(alpha_n)
        gradC_k = -(_c(1.0) - phi_b_c) * grad(alpha_k)
        gradB_n = (_c(1.0) - phi_b_c) * grad(alpha_n)
        gradB_k = (_c(1.0) - phi_b_c) * grad(alpha_k)
        gradP_n = None
        gradP_k = None
        dC_k = -(_c(1.0) - phi_b_c) * dalpha
        grad_dC_k = -(_c(1.0) - phi_b_c) * grad(dalpha)
        dB_k = (_c(1.0) - phi_b_c) * dalpha
        grad_dB_k = (_c(1.0) - phi_b_c) * grad(dalpha)
        dP_k = None
        grad_dP_k = None
    rho_n = rho_f * C_n
    rho_k = rho_f * C_k
    mu_n = _mu_mix(alpha_n, mu_f=mu_f, mu_b=mu_b)
    mu_k = _mu_mix(alpha_k, mu_f=mu_f, mu_b=mu_b)
    drho = rho_f * dC_k
    dmu = (mu_b - mu_f) * dalpha
    rho_s0_tilde = rho_s0_tilde if rho_s0_tilde is not None else zero_scalar
    kappa_inv_key = str(kappa_inv_model).strip().lower().replace("-", "_")
    use_refmap_drag = kappa_inv_key in {"refmap", "eulerian_refmap", "eulerian", "reference_map"}
    if kappa_inv_key not in {"spatial", "constant", "const"} and not use_refmap_drag:
        raise ValueError(f"Unsupported deformation-only kappa_inv_model {kappa_inv_model!r}.")
    support_physics_key = _support_physics_key(support_physics)
    if use_B_transport and support_physics_key != "internal_conversion":
        raise ValueError("alpha-B reduced transport is only implemented for support_physics='internal_conversion'.")
    if support_physics_key == "internal_conversion":
        adv_with_norm = str(alpha_advect_with or "biofilm_volume").strip().lower().replace("-", "_")
        if adv_with_norm not in {"biofilm", "biofilm_volume", "phase", "phase_volume"}:
            raise ValueError(
                "deformation_only internal_conversion requires alpha_advect_with='biofilm_volume' "
                "because alpha tracks the conserved occupied support."
            )
    drag_form_key = str(drag_formulation or "direct").strip().lower().replace("-", "_")
    if drag_form_key in {"mixed", "mixed_formulation", "mixed_auxiliary", "mixed_lagrange_multiplier"}:
        drag_form_key = "mixed_lm"
    if drag_form_key not in {"direct", "mixed_lm"}:
        raise ValueError(
            f"Unsupported deformation-only drag_formulation={drag_formulation!r}. "
            "Use 'direct' or 'mixed_lm'."
        )
    if drag_form_key == "mixed_lm" and (
        lambda_drag_k is None or lambda_drag_n is None or dlambda_drag is None or lambda_drag_test is None
    ):
        raise ValueError(
            "drag_formulation='mixed_lm' requires (lambda_drag_k, lambda_drag_n, dlambda_drag, lambda_drag_test)."
        )

    v_th = th * v_k + one_m_th * v_n
    vS_th = th * vS_k + one_m_th * vS_n
    u_th = th * u_k + one_m_th * u_n
    alpha_th = th * alpha_k + one_m_th * alpha_n

    alpha_flux_k = None
    alpha_flux_n = None
    alpha_flux_k_comp = None
    alpha_flux_n_comp = None
    d_alpha_flux_k = None
    d_alpha_flux_k_comp = None
    div_alpha_flux_k = None
    div_alpha_flux_n = None
    d_div_alpha_flux_k = None

    adv_with_key = str(alpha_advect_with or "vS").strip().lower()
    if adv_with_key in {"vs", "skeleton", "solid"}:
        adv_u_k = vS_k
        adv_u_n = vS_n
        div_adv_u_k = div(vS_k)
        div_adv_u_n = div(vS_n)
        dadv_u = dvS
        d_div_adv_u = div(dvS)
    elif adv_with_key in {"v", "fluid"}:
        adv_u_k = v_k
        adv_u_n = v_n
        div_adv_u_k = div(v_k)
        div_adv_u_n = div(v_n)
        dadv_u = dv
        d_div_adv_u = div(dv)
    elif adv_with_key in {"mix", "mixture", "f", "flux", "volume"}:
        # Legacy mixture/volume transport flux for the reduced one-domain
        # model:
        #
        #   F = C v + B vS,  with C=(1-alpha)+alpha phi_b and B=alpha(1-phi_b),
        #
        # i.e. the same flux that appears in the one-domain incompressibility
        # constraint div(F)=0. This is useful for comparison studies, but the
        # support-preserving alpha law uses `biofilm_volume` below instead.
        adv_u_k = C_n * v_k + B_n * vS_k
        adv_u_n = C_n * v_n + B_n * vS_n
        dadv_u = C_n * dv + B_n * dvS
        div_adv_u_k = C_n * div(v_k) + dot(gradC_n, v_k) + B_n * div(vS_k) + dot(gradB_n, vS_k)
        div_adv_u_n = C_n * div(v_n) + dot(gradC_n, v_n) + B_n * div(vS_n) + dot(gradB_n, vS_n)
        d_div_adv_u = C_n * div(dv) + dot(gradC_n, dv) + B_n * div(dvS) + dot(gradB_n, dvS)
    elif adv_with_key in {"biofilm", "biofilm_volume", "biofilm-volume", "phase", "phase_volume", "phase-volume"}:
        if use_B_transport:
            # Conservative occupied-support flux:
            #   q_alpha = P v + B vS,  P = alpha - B.
            adv_u_k = None
            adv_u_n = None
            dadv_u = None
            div_adv_u_k = None
            div_adv_u_n = None
            d_div_adv_u = None
            alpha_flux_k = P_k * v_k + B_k * vS_k
            alpha_flux_n = P_n * v_n + B_n * vS_n
            alpha_flux_k_comp = tuple(
                P_k * _vector_component(v_k, i) + B_k * _vector_component(vS_k, i) for i in range(2)
            )
            alpha_flux_n_comp = tuple(
                P_n * _vector_component(v_n, i) + B_n * _vector_component(vS_n, i) for i in range(2)
            )
            d_alpha_flux_k_comp = tuple(
                dP_k * _vector_component(v_k, i)
                + P_k * _vector_component(dv, i)
                + dB_k * _vector_component(vS_k, i)
                + B_k * _vector_component(dvS, i)
                for i in range(2)
            )
            div_alpha_flux_k = (
                P_k * div(v_k)
                + dot(gradP_k, v_k)
                + B_k * div(vS_k)
                + dot(gradB_k, vS_k)
            )
            div_alpha_flux_n = (
                P_n * div(v_n)
                + dot(gradP_n, v_n)
                + B_n * div(vS_n)
                + dot(gradB_n, vS_n)
            )
            d_div_alpha_flux_k = (
                dP_k * div(v_k)
                + P_k * div(dv)
                + dot(grad_dP_k, v_k)
                + dot(gradP_k, dv)
                + dB_k * div(vS_k)
                + B_k * div(dvS)
                + dot(grad_dB_k, vS_k)
                + dot(gradB_k, dvS)
            )
        else:
            # Biofilm-volume velocity from the two constituent volume balances:
            #   ∂t(α φ_b) + div(α φ_b v) = 0,
            #   ∂t(α (1-φ_b)) + div(α (1-φ_b) vS) = 0.
            # Summing gives
            #   ∂t α + div( α [ φ_b v + (1-φ_b) vS ] ) = 0.
            adv_u_k = phi_b_c * v_k + (_c(1.0) - phi_b_c) * vS_k
            adv_u_n = phi_b_c * v_n + (_c(1.0) - phi_b_c) * vS_n
            dadv_u = phi_b_c * dv + (_c(1.0) - phi_b_c) * dvS
            div_adv_u_k = phi_b_c * div(v_k) + (_c(1.0) - phi_b_c) * div(vS_k)
            div_adv_u_n = phi_b_c * div(v_n) + (_c(1.0) - phi_b_c) * div(vS_n)
            d_div_adv_u = phi_b_c * div(dv) + (_c(1.0) - phi_b_c) * div(dvS)
    elif adv_with_key in {"relative", "slip", "v_minus_vs", "v-vs"}:
        adv_u_k = v_k - vS_k
        adv_u_n = v_n - vS_n
        dadv_u = dv - dvS
        div_adv_u_k = div(v_k) - div(vS_k)
        div_adv_u_n = div(v_n) - div(vS_n)
        d_div_adv_u = div(dv) - div(dvS)
    elif adv_with_key in {"interface", "midpoint", "avg", "average"}:
        adv_u_k = _c(0.5) * (v_k + vS_k)
        adv_u_n = _c(0.5) * (v_n + vS_n)
        dadv_u = _c(0.5) * (dv + dvS)
        div_adv_u_k = _c(0.5) * (div(v_k) + div(vS_k))
        div_adv_u_n = _c(0.5) * (div(v_n) + div(vS_n))
        d_div_adv_u = _c(0.5) * (div(dv) + div(dvS))
    else:
        raise ValueError(
            f"Unsupported deformation-only alpha_advect_with={alpha_advect_with!r}. "
            "Use 'vS', 'v', 'mix', 'biofilm_volume', 'relative', or 'interface'."
        )

    fluid_conv_key = str(fluid_convection or "full").strip().lower()
    if fluid_conv_key not in {"full", "lagged", "imex", "off"}:
        raise ValueError(
            f"Unsupported deformation-only fluid_convection={fluid_convection!r}. "
            "Use 'full', 'lagged', 'imex', or 'off'."
        )

    adv_key = str(alpha_advection_form).strip().lower()
    if adv_key in {"advective", "nonconservative", "v.grad", "v·grad", "vgrad"}:
        adv_key = "advective"
    elif adv_key in {"conservative", "div", "divergence", "div(alpha*v)"}:
        adv_key = "conservative"
    elif adv_key in {
        "conservative_weak",
        "conservative-weak",
        "conservative_ibp",
        "conservative-ibp",
        "ibp",
        "weak",
        "weak_conservative",
        "weak-conservative",
    }:
        adv_key = "conservative_weak"
    elif adv_key in {
        "interface_band_conservative",
        "interface-band-conservative",
        "band_conservative",
        "band-conservative",
        "interface_conservative",
        "interface-conservative",
        "localized_interface",
        "localized-interface",
    }:
        adv_key = "interface_band_conservative"
    else:
        raise ValueError(
            f"Unsupported deformation-only alpha_advection_form={alpha_advection_form!r}. "
            "Use 'advective', 'conservative', 'conservative_weak', or 'interface_band_conservative'."
        )
    if support_physics_key == "internal_conversion" and adv_key not in {"conservative", "conservative_weak"}:
        raise ValueError(
            "deformation_only internal_conversion requires a conservative alpha law. "
            "Use alpha_advection_form='conservative_weak' (recommended) or 'conservative'."
        )

    div_C_vtest = _div_weighted_scalar_times_vector(C_k, gradC_k, v_test)
    div_B_vStest = _div_weighted_scalar_times_vector(B_n, gradB_n, vS_test)
    div_Bk_vStest = _div_weighted_scalar_times_vector(B_k, gradB_k, vS_test)
    d_div_C_vtest = dC_k * div(v_test) + dot(grad_dC_k, v_test)
    d_div_B_vStest = zero_scalar

    div_C_vk = _div_weighted_scalar_times_vector(C_k, gradC_k, v_k)
    div_B_vSk = _div_weighted_scalar_times_vector(B_n, gradB_n, vS_k)
    div_B_vSn = _div_weighted_scalar_times_vector(B_n, gradB_n, vS_n)
    d_div_C_vk = dC_k * div(v_k) + C_k * div(dv) + dot(grad_dC_k, v_k) + dot(gradC_k, dv)
    div_B_dvS = _div_weighted_scalar_times_vector(B_n, gradB_n, dvS)
    div_Bk_vSk = _div_weighted_scalar_times_vector(B_k, gradB_k, vS_k)
    d_div_Bk_vStest = dB_k * div(vS_test) + dot(grad_dB_k, vS_test)
    d_div_Bk_vSk = dB_k * div(vS_k) + B_k * div(dvS) + dot(grad_dB_k, vS_k) + dot(gradB_k, dvS)
    div_rhov_k = rho_f * div_C_vk
    div_rhov_n = rho_f * _div_weighted_scalar_times_vector(C_n, gradC_n, v_n)
    d_div_rhov_k = rho_f * d_div_C_vk

    # (i) Fluid momentum.
    r_mom = inner((rho_k * v_k - rho_n * v_n) * inv_dt, v_test) * dx
    if fluid_conv_key == "full":
        conv_k = dot(dot(grad(v_k), v_k), v_test)
        conv_n = dot(dot(grad(v_n), v_n), v_test)
        r_mom += (
            th * (rho_k * conv_k + div_rhov_k * dot(v_k, v_test))
            + one_m_th * (rho_n * conv_n + div_rhov_n * dot(v_n, v_test))
        ) * dx
    elif fluid_conv_key == "lagged":
        conv_k = dot(dot(grad(v_k), v_n), v_test)
        conv_n = dot(dot(grad(v_n), v_n), v_test)
        r_mom += (
            th * (rho_n * conv_k + div_rhov_n * dot(v_k, v_test))
            + one_m_th * (rho_n * conv_n + div_rhov_n * dot(v_n, v_test))
        ) * dx
    elif fluid_conv_key == "imex":
        conv_n = dot(dot(grad(v_n), v_n), v_test)
        r_mom += (rho_n * conv_n + div_rhov_n * dot(v_n, v_test)) * dx
    r_mom += th * _c(2.0) * mu_k * inner(_epsilon(v_k), _epsilon(v_test)) * dx
    r_mom += one_m_th * _c(2.0) * mu_n * inner(_epsilon(v_n), _epsilon(v_test)) * dx
    r_mom += -p_k * div_C_vtest * dx
    r_mom += -_dot_2d_components(f_v, v_test) * dx

    a_mom = inv_dt * (drho * dot(v_k, v_test) + rho_k * dot(dv, v_test)) * dx
    if fluid_conv_key == "full":
        a_mom += th * (
            drho * conv_k
            + rho_k * dot(dot(grad(dv), v_k), v_test)
            + rho_k * dot(dot(grad(v_k), dv), v_test)
        ) * dx
        a_mom += th * (d_div_rhov_k * dot(v_k, v_test) + div_rhov_k * dot(dv, v_test)) * dx
    elif fluid_conv_key == "lagged":
        a_mom += th * rho_n * dot(dot(grad(dv), v_n), v_test) * dx
        a_mom += th * div_rhov_n * dot(dv, v_test) * dx
    a_mom += th * _c(2.0) * (dmu * inner(_epsilon(v_k), _epsilon(v_test)) + mu_k * inner(_epsilon(dv), _epsilon(v_test))) * dx
    a_mom += -(dp * div_C_vtest + p_k * d_div_C_vtest) * dx

    # (ii) One-domain mass constraint.
    r_mass = q_test * (div_C_vk + div_Bk_vSk) * dx
    a_mass = q_test * (d_div_C_vk + d_div_Bk_vSk) * dx

    # (iii) Skeleton momentum.
    if solid_model_key in {"linear", "small_strain", "linear_elastic"}:
        if bool(solid_volumetric_split):
            r_el_k = _linear_deviatoric_elastic_term(u_k, vS_test, mu_s=mu_s, dim=2) + total_pressure_ref_c * pi_s_k * div(vS_test)
            r_el_n = _linear_deviatoric_elastic_term(u_n, vS_test, mu_s=mu_s, dim=2) + total_pressure_ref_c * pi_s_n * div(vS_test)
            a_el = _linear_deviatoric_elastic_term(du, vS_test, mu_s=mu_s, dim=2) + total_pressure_ref_c * dpi_s * div(vS_test)
        else:
            r_el_k = _linear_elastic_term(u_k, vS_test, mu_s=mu_s, lambda_s=lambda_s)
            r_el_n = _linear_elastic_term(u_n, vS_test, mu_s=mu_s, lambda_s=lambda_s)
            a_el = _linear_elastic_term(du, vS_test, mu_s=mu_s, lambda_s=lambda_s)
    else:
        if c_nh is None:
            c_nh = mu_s / _c(2.0)
        if beta_nh is None:
            beta_nh = lambda_s / (_c(2.0) * mu_s)
        sig_k = sigma_neo_hookean(u_k, c_nh, beta_nh, dim=2)
        sig_n = sigma_neo_hookean(u_n, c_nh, beta_nh, dim=2)
        dsig_k = dsigma_neo_hookean(u_k, du, c_nh, beta_nh, dim=2)
        r_el_k = inner(sig_k, grad(vS_test))
        r_el_n = inner(sig_n, grad(vS_test))
        a_el = inner(dsig_k, grad(vS_test))

    sk_th = th if bool(include_skeleton_acceleration) else _c(1.0)
    sk_one_m_th = one_m_th if bool(include_skeleton_acceleration) else _c(0.0)

    skel_press_key = str(skeleton_pressure_mode).strip().lower().replace("-", "_")
    if skel_press_key not in {"whole_domain", "seboldt"}:
        raise ValueError(
            f"Unsupported deformation-only skeleton_pressure_mode={skeleton_pressure_mode!r}. "
            "Use 'whole_domain' or 'seboldt'."
        )

    press_div_coeff_k = None
    press_div_coeff_n = None
    d_press_div_coeff_k = None
    if skel_press_key == "seboldt":
        alpha_biot_c = _as_constant(1.0 if alpha_biot is None else alpha_biot)
        press_coeff_k = alpha_biot_c * alpha_k
        press_coeff_n = alpha_biot_c * alpha_n
        d_press_coeff_k = alpha_biot_c * dalpha
        press_div_coeff_k = press_coeff_k
        press_div_coeff_n = press_coeff_n
        d_press_div_coeff_k = d_press_coeff_k
        r_skel_press_k = -(p_k * press_coeff_k * div(vS_test))
        r_skel_press_n = -(p_n * press_coeff_n * div(vS_test))
    elif alpha_biot is not None:
        alpha_biot_c = _as_constant(alpha_biot)
        biot_corr_coeff_k = alpha_biot_c * alpha_k - B_k
        biot_corr_coeff_n = alpha_biot_c * alpha_n - B_n
        d_press_coeff_k = alpha_biot_c * dalpha - dB_k
        press_div_coeff_k = alpha_biot_c * alpha_k
        press_div_coeff_n = alpha_biot_c * alpha_n
        d_press_div_coeff_k = alpha_biot_c * dalpha
        r_skel_press_k = -(p_k * div_Bk_vStest)
        r_skel_press_k += -(p_k * biot_corr_coeff_k * div(vS_test))
        r_skel_press_n = -(p_n * div_B_vStest)
        r_skel_press_n += -(p_n * biot_corr_coeff_n * div(vS_test))
    else:
        alpha_biot_c = None
        d_press_coeff_k = None
        press_div_coeff_k = B_k
        press_div_coeff_n = B_n
        d_press_div_coeff_k = dB_k
        r_skel_press_k = -(p_k * div_Bk_vStest)
        r_skel_press_n = -(p_n * div_B_vStest)

    r_skel_pressure = (sk_th * r_skel_press_k + sk_one_m_th * r_skel_press_n) * dx

    a_skel = sk_th * (alpha_k * a_el + dalpha * r_el_k) * dx
    if bool(solid_volumetric_split):
        vol_pen_c = _as_constant(float(solid_volumetric_penalty))
        vol_drive_k = (
            pi_s_k
            - lambda_s_over_total_pressure_ref_c * div(u_k)
            + total_pressure_ref_inv_c * press_div_coeff_k * p_k
        )
        d_vol_drive_k = (
            dpi_s
            - lambda_s_over_total_pressure_ref_c * div(du)
            + total_pressure_ref_inv_c * (d_press_div_coeff_k * p_k + press_div_coeff_k * dp)
        )
        r_volumetric = (
            alpha_k * pi_s_test * vol_drive_k
            + vol_pen_c * _one_minus(alpha_k) * pi_s_k * pi_s_test
        ) * dx
        a_volumetric = (
            alpha_k * pi_s_test * d_vol_drive_k
            + dalpha * pi_s_test * vol_drive_k
            + vol_pen_c * ((-dalpha) * pi_s_k * pi_s_test + _one_minus(alpha_k) * dpi_s * pi_s_test)
        ) * dx
        if float(pressure_block_lift_scale) != 0.0:
            r_mass += pressure_block_lift_scale_c * (
                alpha_k * q_test * vol_drive_k
                + vol_pen_c * _one_minus(alpha_k) * pi_s_k * q_test
            ) * dx
            a_mass += pressure_block_lift_scale_c * (
                alpha_k * q_test * d_vol_drive_k
                + dalpha * q_test * vol_drive_k
                + vol_pen_c * ((-dalpha) * pi_s_k * q_test + _one_minus(alpha_k) * dpi_s * q_test)
            ) * dx
        if skel_press_key == "seboldt":
            # Preserve residual/Jacobian roles even though the split-Seboldt
            # pressure remainder vanishes identically.
            r_skel_press_k = _c(0.0) * dot(vS_k, vS_test)
            r_skel_press_n = _c(0.0) * dot(vS_n, vS_test)
            r_skel_pressure = (sk_th * r_skel_press_k + sk_one_m_th * r_skel_press_n) * dx
            a_skel_pressure = _c(0.0) * dot(dvS, vS_test) * dx
        else:
            r_skel_press_k = -(p_k * dot(gradB_k, vS_test))
            r_skel_press_n = -(p_n * dot(gradB_n, vS_test))
            r_skel_pressure = (sk_th * r_skel_press_k + sk_one_m_th * r_skel_press_n) * dx
            a_skel_pressure = sk_th * (-(dp * dot(gradB_k, vS_test) + p_k * dot(grad_dB_k, vS_test))) * dx
    elif skel_press_key == "seboldt":
        a_skel_pressure = sk_th * (-(dp * press_coeff_k + p_k * d_press_coeff_k) * div(vS_test)) * dx
    elif alpha_biot_c is not None:
        a_skel_pressure = sk_th * (-(dp * div_Bk_vStest + p_k * d_div_Bk_vStest)) * dx
        a_skel_pressure += sk_th * (-(dp * biot_corr_coeff_k + p_k * d_press_coeff_k) * div(vS_test)) * dx
    else:
        a_skel_pressure = sk_th * (-(dp * div_Bk_vStest + p_k * d_div_Bk_vStest)) * dx
    a_skel += a_skel_pressure

    diff_k = v_k - vS_k
    diff_n = v_n - vS_n
    ddiff = dv - dvS
    if support_physics_key == "internal_conversion":
        drag_occ_k = B_k
        drag_occ_n = B_n
        ddrag_occ = dB_k
        if use_B_transport:
            drag_phi_factor_k = _phi_sq_from_alpha_B(alpha_k, B_k)
            drag_phi_factor_n = _phi_sq_from_alpha_B(alpha_n, B_n)
            d_drag_phi_factor_k = _d_phi_sq_from_alpha_B(alpha_k, B_k, dalpha, dB_k)
        else:
            drag_phi_factor_k = phi_b_c * phi_b_c
            drag_phi_factor_n = phi_b_c * phi_b_c
            d_drag_phi_factor_k = zero_scalar
    else:
        drag_occ_k = alpha_k
        drag_occ_n = alpha_n
        ddrag_occ = dalpha
        drag_phi_factor_k = _c(1.0)
        drag_phi_factor_n = _c(1.0)
        d_drag_phi_factor_k = zero_scalar
    drag_weight_k = drag_occ_k * drag_phi_factor_k
    drag_weight_n = drag_occ_n * drag_phi_factor_n
    ddrag_weight = ddrag_occ * drag_phi_factor_k + drag_occ_k * d_drag_phi_factor_k
    r_drag_lambda = None
    a_drag_lambda = None
    if use_refmap_drag:
        K_inv = kappa_inv * Identity(2) if getattr(kappa_inv, "dim", None) == 0 else kappa_inv
        k_inv_k = eulerian_k_inv(u_k, K_inv, dim=2)
        k_inv_n = eulerian_k_inv(u_n, K_inv, dim=2)
        dk_inv_k = deulerian_k_inv(u_k, du, K_inv, dim=2)

        kdrag_k = _matvec_2d_components(k_inv_k, diff_k)
        kdrag_n = _matvec_2d_components(k_inv_n, diff_n)
        dkdrag_k_base = _matvec_2d_components(k_inv_k, ddiff)
        dkdrag_k_geom = _matvec_2d_components(dk_inv_k, diff_k)
        dkdrag_k = (
            dkdrag_k_base[0] + dkdrag_k_geom[0],
            dkdrag_k_base[1] + dkdrag_k_geom[1],
        )

        beta_coeff_k = drag_occ_k * mu_f * drag_phi_factor_k
        beta_coeff_n = drag_occ_n * mu_f * drag_phi_factor_n
        dbeta_coeff = mu_f * ddrag_weight
        if drag_form_key == "mixed_lm":
            det_k = k_inv_k[0, 0] * k_inv_k[1, 1] - k_inv_k[0, 1] * k_inv_k[1, 0]
            inv_raw_k = (
                (k_inv_k[1, 1] * _vector_component(lambda_drag_k, 0) - k_inv_k[0, 1] * _vector_component(lambda_drag_k, 1)) / det_k,
                (-k_inv_k[1, 0] * _vector_component(lambda_drag_k, 0) + k_inv_k[0, 0] * _vector_component(lambda_drag_k, 1)) / det_k,
            )
            lam_inv_k = ((_c(1.0) / mu_f) * inv_raw_k[0], (_c(1.0) / mu_f) * inv_raw_k[1])
            core_inv_dlam_k = (
                (_c(1.0) / mu_f)
                * ((k_inv_k[1, 1] * _vector_component(dlambda_drag, 0) - k_inv_k[0, 1] * _vector_component(dlambda_drag, 1)) / det_k),
                (_c(1.0) / mu_f)
                * ((-k_inv_k[1, 0] * _vector_component(dlambda_drag, 0) + k_inv_k[0, 0] * _vector_component(dlambda_drag, 1)) / det_k),
            )
            dcore_raw_rhs = _matvec_2d_components(dk_inv_k, inv_raw_k)
            dcore_raw_k = (
                -((k_inv_k[1, 1] * dcore_raw_rhs[0] - k_inv_k[0, 1] * dcore_raw_rhs[1]) / det_k),
                -((-k_inv_k[1, 0] * dcore_raw_rhs[0] + k_inv_k[0, 0] * dcore_raw_rhs[1]) / det_k),
            )
            dlam_inv_k = (
                (_c(1.0) / mu_f) * dcore_raw_k[0] + core_inv_dlam_k[0],
                (_c(1.0) / mu_f) * dcore_raw_k[1] + core_inv_dlam_k[1],
            )
            r_mom += _dot_2d_components(lambda_drag_k, v_test) * dx
            a_mom += _dot_2d_components(dlambda_drag, v_test) * dx

            r_skel_drag_k = -_dot_2d_components(lambda_drag_k, vS_test)
            r_skel_drag_n = -_dot_2d_components(lambda_drag_n, vS_test)
            a_skel += sk_th * (-_dot_2d_components(dlambda_drag, vS_test)) * dx

            r_drag_lambda = (
                _dot_2d_components(lam_inv_k, lambda_drag_test)
                - _weighted_dot_2d_components(drag_weight_k, diff_k, lambda_drag_test)
            ) * dx
            a_drag_lambda = (
                _dot_2d_components(dlam_inv_k, lambda_drag_test)
                - _weighted_dot_2d_components(ddrag_weight, diff_k, lambda_drag_test)
                - _weighted_dot_2d_components(drag_weight_k, ddiff, lambda_drag_test)
            ) * dx
        else:
            r_mom += beta_coeff_k * _dot_2d_components(kdrag_k, v_test) * dx
            a_mom += (
                dbeta_coeff * _dot_2d_components(kdrag_k, v_test) + beta_coeff_k * _dot_2d_components(dkdrag_k, v_test)
            ) * dx

            r_skel_drag_k = -beta_coeff_k * _dot_2d_components(kdrag_k, vS_test)
            r_skel_drag_n = -beta_coeff_n * _dot_2d_components(kdrag_n, vS_test)
            a_skel += sk_th * (
                -(dbeta_coeff * _dot_2d_components(kdrag_k, vS_test) + beta_coeff_k * _dot_2d_components(dkdrag_k, vS_test))
            ) * dx
    else:
        beta_k = drag_occ_k * mu_f * drag_phi_factor_k * kappa_inv
        beta_n = drag_occ_n * mu_f * drag_phi_factor_n * kappa_inv
        dbeta = mu_f * ddrag_weight * kappa_inv
        if drag_form_key == "mixed_lm":
            drag_core_k = mu_f * kappa_inv
            drag_core_inv_k = _c(1.0) / drag_core_k
            r_mom += _dot_2d_components(lambda_drag_k, v_test) * dx
            a_mom += _dot_2d_components(dlambda_drag, v_test) * dx

            r_skel_drag_k = -_dot_2d_components(lambda_drag_k, vS_test)
            r_skel_drag_n = -_dot_2d_components(lambda_drag_n, vS_test)
            a_skel += sk_th * (-_dot_2d_components(dlambda_drag, vS_test)) * dx

            r_drag_lambda = dot((drag_core_inv_k * lambda_drag_k) - (drag_weight_k * diff_k), lambda_drag_test) * dx
            a_drag_lambda = dot(
                (drag_core_inv_k * dlambda_drag) - (ddrag_weight * diff_k) - (drag_weight_k * ddiff),
                lambda_drag_test,
            ) * dx
        else:
            r_mom += beta_k * dot(diff_k, v_test) * dx
            a_mom += (dbeta * dot(diff_k, v_test) + beta_k * dot(ddiff, v_test)) * dx

            r_skel_drag_k = -beta_k * dot(diff_k, vS_test)
            r_skel_drag_n = -beta_n * dot(diff_n, vS_test)
            a_skel += sk_th * (-(dbeta * dot(diff_k, vS_test) + beta_k * dot(ddiff, vS_test))) * dx

    r_skel = (
        sk_th * alpha_k * r_el_k
        + sk_one_m_th * alpha_n * r_el_n
        + sk_th * r_skel_press_k
        + sk_one_m_th * r_skel_press_n
        + sk_th * r_skel_drag_k
        + sk_one_m_th * r_skel_drag_n
    ) * dx
    r_skel += -dot(alpha_k * f_u, vS_test) * dx

    if float(solid_visco_eta) != 0.0:
        eta_s_c = _c(float(solid_visco_eta))
        sig_visc_k = _c(2.0) * eta_s_c * _epsilon(vS_k)
        sig_visc_n = _c(2.0) * eta_s_c * _epsilon(vS_n)
        r_visc_k = inner(sig_visc_k, grad(vS_test))
        r_visc_n = inner(sig_visc_n, grad(vS_test))
        r_skel += (th * alpha_n * r_visc_k + one_m_th * alpha_n * r_visc_n) * dx

        sig_dvisc = _c(2.0) * eta_s_c * _epsilon(dvS)
        a_skel += th * alpha_n * inner(sig_dvisc, grad(vS_test)) * dx

    if bool(include_skeleton_acceleration) and float(rho_s0_tilde) != 0.0:
        inertia_conv_key = str(skeleton_inertia_convection).strip().lower() if skeleton_inertia_convection is not None else "lagged"
        if inertia_conv_key in {"conservative", "nonlinear"}:
            inertia_conv_key = "full"
        if inertia_conv_key in {"picard", "semi", "semi_implicit", "linear"}:
            inertia_conv_key = "lagged"
        if inertia_conv_key not in {"full", "lagged"}:
            raise ValueError(
                f"Unknown deformation-only skeleton_inertia_convection={skeleton_inertia_convection!r}. "
                "Use 'lagged' (default) or 'full'."
            )

        rho_s0_c = rho_s0_tilde
        rhoS_n = rho_s0_c * B_n
        rhoS_k = rho_s0_c * B_k
        div_rhoS_vS_n = rho_s0_c * div_B_vSn

        momS_dot = (rhoS_k * vS_k - rhoS_n * vS_n) * inv_dt
        r_skel += inner(momS_dot, vS_test) * dx

        grad_vS_k = grad(vS_k)
        grad_vS_n = grad(vS_n)
        if inertia_conv_key == "full":
            advS_k = dot(grad_vS_k, vS_k)
            advS_n = dot(grad_vS_n, vS_n)
            convS_k = dot(advS_k, vS_test)
            convS_n = dot(advS_n, vS_test)
            div_rhoS_vS_k = rho_s0_c * div_Bk_vSk
            r_skel += th * (rhoS_k * convS_k + div_rhoS_vS_k * dot(vS_k, vS_test)) * dx
            r_skel += one_m_th * (rhoS_n * convS_n + div_rhoS_vS_n * dot(vS_n, vS_test)) * dx
        else:
            advS_k = dot(grad_vS_k, vS_n)
            advS_n = dot(grad_vS_n, vS_n)
            convS_k = dot(advS_k, vS_test)
            convS_n = dot(advS_n, vS_test)
            r_skel += th * (rhoS_n * convS_k + div_rhoS_vS_n * dot(vS_k, vS_test)) * dx
            r_skel += one_m_th * (rhoS_n * convS_n + div_rhoS_vS_n * dot(vS_n, vS_test)) * dx

        d_rhoS_k = rho_s0_c * dB_k
        d_momS_dot_vtest = _c(0.0)
        for i in range(2):
            d_momS_dot_vtest += (
                d_rhoS_k * _vector_component(vS_k, i) + rhoS_k * _vector_component(dvS, i)
            ) * _vector_component(vS_test, i)
        a_skel += inv_dt * d_momS_dot_vtest * dx

        if inertia_conv_key == "full":
            grad_dvS = grad(dvS)
            d_advS_k = dot(grad_dvS, vS_k) + dot(grad_vS_k, dvS)
            d_convS_k = dot(d_advS_k, vS_test)
            div_rhoS_vS_k = rho_s0_c * div_Bk_vSk
            d_div_rhoS_vS_k = rho_s0_c * d_div_Bk_vSk
            a_skel += th * (d_rhoS_k * convS_k + rhoS_k * d_convS_k) * dx
            a_skel += th * (d_div_rhoS_vS_k * dot(vS_k, vS_test) + div_rhoS_vS_k * dot(dvS, vS_test)) * dx
        else:
            grad_dvS = grad(dvS)
            d_advS_k = dot(grad_dvS, vS_n)
            d_convS_k = dot(d_advS_k, vS_test)
            a_skel += th * (rhoS_n * d_convS_k + div_rhoS_vS_n * dot(dvS, vS_test)) * dx

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
        mass_res_k = div_C_vk + div_Bk_vSk
        d_mass_res_k = d_div_C_vk + d_div_Bk_vSk
        r_mom += gamma_div_c * mass_res_k * div_C_vtest * dx
        r_skel += gamma_div_c * mass_res_k * div_Bk_vStest * dx
        a_mom += gamma_div_c * (d_mass_res_k * div_C_vtest + mass_res_k * d_div_C_vtest) * dx
        a_skel += gamma_div_c * (d_mass_res_k * div_Bk_vStest + mass_res_k * d_div_Bk_vStest) * dx

    # (iv) Eulerian reference-map kinematics.
    if kinematics_scale is None:
        kinematics_scale = rho_s0_tilde if (rho_s0_tilde is not None and float(rho_s0_tilde) != 0.0) else 1.0
    kin_scale_c = kinematics_scale if hasattr(kinematics_scale, "dim") else _c(float(kinematics_scale))

    Fkin_dt = (u_k - u_n) * inv_dt
    Fkin_adv_k = dot(grad(u_k), vS_k) - vS_k
    Fkin_adv_n = dot(grad(u_n), vS_n) - vS_n
    Fkin_k = Fkin_dt + th * Fkin_adv_k + one_m_th * Fkin_adv_n
    r_kin = kin_scale_c * alpha_k * dot(Fkin_k, u_test) * dx

    dFkin_dt = du * inv_dt
    dFkin_adv_k = dot(grad(du), vS_k) + dot(grad(u_k), dvS) - dvS
    dFkin_k = dFkin_dt + th * dFkin_adv_k
    a_kin = kin_scale_c * (dalpha * dot(Fkin_k, u_test) + alpha_k * dot(dFkin_k, u_test)) * dx

    # (v) Alpha transport + Cahn--Hilliard regularization.
    band_alpha_k = _alpha_band(alpha_k)
    band_alpha_n = _alpha_band(alpha_n)
    dband_alpha = _d_alpha_band(alpha_k, dalpha)
    grad_band_alpha_k = _grad_alpha_band(alpha_k)
    grad_band_alpha_n = _grad_alpha_band(alpha_n)
    d_grad_band_alpha = _d_grad_alpha_band(alpha_k, dalpha)

    if adv_key == "advective":
        adv_alpha_k = dot(grad(alpha_k), adv_u_k)
        adv_alpha_n = dot(grad(alpha_n), adv_u_n)
        time_alpha_k = alpha_k
        time_alpha_n = alpha_n
    elif adv_key == "conservative":
        if use_B_transport:
            adv_alpha_k = div_alpha_flux_k
            adv_alpha_n = div_alpha_flux_n
        else:
            adv_alpha_k = dot(grad(alpha_k), adv_u_k) + alpha_k * div_adv_u_k
            adv_alpha_n = dot(grad(alpha_n), adv_u_n) + alpha_n * div_adv_u_n
        time_alpha_k = alpha_k
        time_alpha_n = alpha_n
    elif adv_key == "interface_band_conservative":
        adv_alpha_k = dot(grad_band_alpha_k, adv_u_k) + band_alpha_k * div_adv_u_k
        adv_alpha_n = dot(grad_band_alpha_n, adv_u_n) + band_alpha_n * div_adv_u_n
        time_alpha_k = band_alpha_k
        time_alpha_n = band_alpha_n
    else:
        adv_alpha_k = None
        adv_alpha_n = None
        time_alpha_k = alpha_k
        time_alpha_n = alpha_n

    r_alpha = alpha_test * ((time_alpha_k - time_alpha_n) * inv_dt) * dx
    if adv_key == "conservative_weak":
        # Weak conservative alpha transport:
        #
        #   (partial_t alpha, w) - (q_alpha, grad(w)) = 0
        #
        # with q_alpha chosen above. This is the form that preserves the domain
        # integral of alpha when the boundary contribution vanishes, either
        # because natural no-flux conditions are used or because the Dirichlet
        # test space vanishes on the relevant boundary.
        if use_B_transport:
            flux_dot_grad_test_k = _c(0.0)
            flux_dot_grad_test_n = _c(0.0)
            for i in range(2):
                flux_dot_grad_test_k = flux_dot_grad_test_k + alpha_flux_k_comp[i] * grad(alpha_test)[i]
                flux_dot_grad_test_n = flux_dot_grad_test_n + alpha_flux_n_comp[i] * grad(alpha_test)[i]
        else:
            flux_dot_grad_test_k = _c(0.0)
            flux_dot_grad_test_n = _c(0.0)
            for i in range(2):
                flux_dot_grad_test_k = flux_dot_grad_test_k + (alpha_k * _vector_component(adv_u_k, i)) * grad(alpha_test)[i]
                flux_dot_grad_test_n = flux_dot_grad_test_n + (alpha_n * _vector_component(adv_u_n, i)) * grad(alpha_test)[i]
        r_alpha += -th * flux_dot_grad_test_k * dx
        r_alpha += -one_m_th * flux_dot_grad_test_n * dx
        if ds_alpha_transport is not None:
            n_b = FacetNormal()
            flux_alpha_bdry_k = _c(0.0)
            flux_alpha_bdry_n = _c(0.0)
            for i in range(2):
                if use_B_transport:
                    flux_alpha_bdry_k = flux_alpha_bdry_k + alpha_flux_k_comp[i] * _vector_component(n_b, i)
                    flux_alpha_bdry_n = flux_alpha_bdry_n + alpha_flux_n_comp[i] * _vector_component(n_b, i)
                else:
                    flux_alpha_bdry_k = flux_alpha_bdry_k + (alpha_k * _vector_component(adv_u_k, i)) * _vector_component(n_b, i)
                    flux_alpha_bdry_n = flux_alpha_bdry_n + (alpha_n * _vector_component(adv_u_n, i)) * _vector_component(n_b, i)
            r_alpha += th * alpha_test * flux_alpha_bdry_k * ds_alpha_transport
            r_alpha += one_m_th * alpha_test * flux_alpha_bdry_n * ds_alpha_transport
    else:
        r_alpha += th * alpha_test * adv_alpha_k * dx
        r_alpha += one_m_th * alpha_test * adv_alpha_n * dx
    if ch_enabled:
        r_alpha += M_alpha_c * inner(grad(mu_alpha_k), grad(alpha_test)) * dx
    r_alpha += -alpha_test * f_alpha * dx

    if adv_key == "interface_band_conservative":
        a_alpha = alpha_test * (dband_alpha * inv_dt) * dx
    else:
        a_alpha = alpha_test * (dalpha * inv_dt) * dx
    if adv_key == "advective":
        a_alpha += th * alpha_test * (dot(grad(dalpha), adv_u_k) + dot(grad(alpha_k), dadv_u)) * dx
    elif adv_key == "conservative":
        if use_B_transport:
            a_alpha += th * alpha_test * d_div_alpha_flux_k * dx
        else:
            a_alpha += th * alpha_test * (
                dot(grad(dalpha), adv_u_k) + dot(grad(alpha_k), dadv_u) + dalpha * div_adv_u_k + alpha_k * d_div_adv_u
            ) * dx
    elif adv_key == "interface_band_conservative":
        a_alpha += th * alpha_test * (
            dot(d_grad_band_alpha, adv_u_k) + dot(grad_band_alpha_k, dadv_u) + dband_alpha * div_adv_u_k + band_alpha_k * d_div_adv_u
        ) * dx
    else:
        if use_B_transport:
            dflux_dot_grad_test_k = _c(0.0)
            for i in range(2):
                dflux_dot_grad_test_k = dflux_dot_grad_test_k + d_alpha_flux_k_comp[i] * grad(alpha_test)[i]
        else:
            dflux_dot_grad_test_k = _c(0.0)
            for i in range(2):
                dflux_dot_grad_test_k = dflux_dot_grad_test_k + (
                    (dalpha * _vector_component(adv_u_k, i) + alpha_k * _vector_component(dadv_u, i)) * grad(alpha_test)[i]
                )
        a_alpha += -th * dflux_dot_grad_test_k * dx
        if ds_alpha_transport is not None:
            n_b = FacetNormal()
            dflux_alpha_bdry_k = _c(0.0)
            for i in range(2):
                if use_B_transport:
                    dflux_alpha_bdry_k = dflux_alpha_bdry_k + d_alpha_flux_k_comp[i] * _vector_component(n_b, i)
                else:
                    dflux_alpha_bdry_k = dflux_alpha_bdry_k + (
                        (dalpha * _vector_component(adv_u_k, i) + alpha_k * _vector_component(dadv_u, i)) * _vector_component(n_b, i)
                    )
            a_alpha += th * alpha_test * dflux_alpha_bdry_k * ds_alpha_transport
    if ch_enabled:
        a_alpha += M_alpha_c * inner(grad(dmu_alpha), grad(alpha_test)) * dx

    r_B = None
    a_B = None
    if use_B_transport:
        r_B = B_test * ((B_k - B_n) * inv_dt) * dx
        flux_B_grad_test_k = _c(0.0)
        flux_B_grad_test_n = _c(0.0)
        for i in range(2):
            flux_B_grad_test_k = flux_B_grad_test_k + (B_k * _vector_component(vS_k, i)) * grad(B_test)[i]
            flux_B_grad_test_n = flux_B_grad_test_n + (B_n * _vector_component(vS_n, i)) * grad(B_test)[i]
        r_B += -th * flux_B_grad_test_k * dx
        r_B += -one_m_th * flux_B_grad_test_n * dx
        r_B += -B_test * f_B * dx
        if ds_B_transport is not None:
            n_b = FacetNormal()
            flux_B_bdry_k = _c(0.0)
            flux_B_bdry_n = _c(0.0)
            for i in range(2):
                flux_B_bdry_k = flux_B_bdry_k + (B_k * _vector_component(vS_k, i)) * _vector_component(n_b, i)
                flux_B_bdry_n = flux_B_bdry_n + (B_n * _vector_component(vS_n, i)) * _vector_component(n_b, i)
            r_B += th * B_test * flux_B_bdry_k * ds_B_transport
            r_B += one_m_th * B_test * flux_B_bdry_n * ds_B_transport

        a_B = B_test * (dB * inv_dt) * dx
        dflux_B_grad_test_k = _c(0.0)
        for i in range(2):
            dflux_B_grad_test_k = dflux_B_grad_test_k + (
                (dB * _vector_component(vS_k, i) + B_k * _vector_component(dvS, i)) * grad(B_test)[i]
            )
        a_B += -th * dflux_B_grad_test_k * dx
        if ds_B_transport is not None:
            n_b = FacetNormal()
            dflux_B_bdry_k = _c(0.0)
            for i in range(2):
                dflux_B_bdry_k = dflux_B_bdry_k + (
                    (dB * _vector_component(vS_k, i) + B_k * _vector_component(dvS, i)) * _vector_component(n_b, i)
                )
            a_B += th * B_test * dflux_B_bdry_k * ds_B_transport

    # (vi) Chemical potential relation.
    if ch_enabled:
        Wp_k = _W_prime(alpha_k)
        Wpp_k = _W_second(alpha_k)
        r_mu_alpha = mu_alpha_test * mu_alpha_k * dx
        r_mu_alpha += -(gamma_alpha_c * eps_alpha_c) * inner(grad(alpha_k), grad(mu_alpha_test)) * dx
        r_mu_alpha += -mu_alpha_test * ((gamma_alpha_c / eps_alpha_c) * Wp_k) * dx

        a_mu_alpha = mu_alpha_test * dmu_alpha * dx
        a_mu_alpha += -(gamma_alpha_c * eps_alpha_c) * inner(grad(dalpha), grad(mu_alpha_test)) * dx
        a_mu_alpha += -mu_alpha_test * ((gamma_alpha_c / eps_alpha_c) * Wpp_k * dalpha) * dx
    else:
        mu_aux_pin_c = _c(float(alpha_mu_aux_pin))
        r_mu_alpha = mu_aux_pin_c * mu_alpha_test * mu_alpha_k * dx
        a_mu_alpha = mu_aux_pin_c * mu_alpha_test * dmu_alpha * dx

    residual_form = r_mom + r_mass + r_skel + r_kin + r_alpha + r_mu_alpha
    jacobian_form = a_mom + a_mass + a_skel + a_kin + a_alpha + a_mu_alpha
    if r_B is not None:
        residual_form += r_B
    if a_B is not None:
        jacobian_form += a_B
    if r_drag_lambda is not None:
        residual_form += r_drag_lambda
    if a_drag_lambda is not None:
        jacobian_form += a_drag_lambda
    if r_volumetric is not None:
        residual_form += r_volumetric
    if a_volumetric is not None:
        jacobian_form += a_volumetric

    return DeformationOnlyForms(
        jacobian_form=jacobian_form,
        residual_form=residual_form,
        r_momentum=r_mom,
        r_mass=r_mass,
        r_skeleton=r_skel,
        r_kinematics=r_kin,
        r_alpha=r_alpha,
        r_B=r_B,
        r_mu_alpha=r_mu_alpha,
        a_momentum=a_mom,
        a_mass=a_mass,
        a_skeleton=a_skel,
        a_kinematics=a_kin,
        a_alpha=a_alpha,
        a_B=a_B,
        a_mu_alpha=a_mu_alpha,
        r_drag_lambda=r_drag_lambda,
        a_drag_lambda=a_drag_lambda,
        r_skeleton_pressure=r_skel_pressure,
        a_skeleton_pressure=a_skel_pressure,
        r_volumetric=r_volumetric,
        a_volumetric=a_volumetric,
    )
