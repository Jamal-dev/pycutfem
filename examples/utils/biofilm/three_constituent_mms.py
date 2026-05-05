"""Manufactured-source helpers for the canonical three-constituent model."""

from __future__ import annotations

from pycutfem.ufl.expressions import div, dot, grad, outer

from .three_constituent_one_domain import (
    _as_named_expr,
    _lit,
    _neg,
    _pair_weight,
    _sub,
    linear_fluid_stress,
    linear_skeleton_stress,
    one_domain_contents,
    pairwise_internal_forces,
)


def backward_euler_three_constituent_sources(
    *,
    v_f_k,
    p_f_k,
    v_p_k,
    p_p_k,
    v_s_k,
    u_s_k,
    alpha_k,
    phi_k,
    Gamma_k,
    v_f_n,
    v_p_n,
    v_s_n,
    u_s_n,
    alpha_n,
    phi_n,
    Gamma_n=None,
    dt,
    rho_f=1.0,
    rho_p=1.0,
    rho_s=1.0,
    mu_f=1.0,
    mu_p=0.0,
    mu_s=1.0,
    lambda_s=1.0,
    R_fp=0.0,
    R_fs=0.0,
    R_ps=1.0,
    R_pair_cholesky=None,
    pair_weight_epsilon=0.0,
    theta_fp=0.5,
    ell_Gamma=0.0,
    gamma_mobility: str = "FP",
    gamma_delta_epsilon: float = 1.0e-12,
    transfer_velocity: str = "average",
    include_stress_divergence: bool = True,
    dim: int = 2,
):
    """Return source expressions that make an arbitrary k/n state exact.

    These sources match the volume weak-form conventions in
    ``build_three_constituent_one_domain_forms`` with no boundary terms.
    """

    dim = int(dim)
    if Gamma_n is not None:
        _ = Gamma_n
    dt_c = _as_named_expr("tc_mms_dt", dt)
    inv_dt = _lit(1.0) / dt_c
    rho_f_c = _as_named_expr("tc_mms_rho_f", rho_f)
    rho_p_c = _as_named_expr("tc_mms_rho_p", rho_p)
    rho_s_c = _as_named_expr("tc_mms_rho_s", rho_s)
    mu_f_c = _as_named_expr("tc_mms_mu_f", mu_f)
    mu_p_c = _as_named_expr("tc_mms_mu_p", mu_p)
    mu_s_c = _as_named_expr("tc_mms_mu_s", mu_s)
    lambda_s_c = _as_named_expr("tc_mms_lambda_s", lambda_s)
    R_fp_c = _as_named_expr("tc_mms_R_fp", R_fp)
    R_fs_c = _as_named_expr("tc_mms_R_fs", R_fs)
    R_ps_c = _as_named_expr("tc_mms_R_ps", R_ps)
    theta_fp_c = _as_named_expr("tc_mms_theta_fp", theta_fp)
    ell_Gamma_c = _as_named_expr("tc_mms_ell_Gamma", ell_Gamma)
    gamma_delta_epsilon_sq_c = _as_named_expr(
        "tc_mms_gamma_delta_epsilon_sq",
        float(gamma_delta_epsilon) * float(gamma_delta_epsilon),
    )

    F_k, P_k, B_k = one_domain_contents(alpha_k, phi_k)
    F_n, P_n, B_n = one_domain_contents(alpha_n, phi_n)
    r_f_k = F_k * rho_f_c
    r_p_k = P_k * rho_p_c
    r_s_k = B_k * rho_s_c
    r_f_n = F_n * rho_f_c
    r_p_n = P_n * rho_p_c
    r_s_n = B_n * rho_s_c

    sigma_f_k = linear_fluid_stress(v_f_k, p_f_k, mu=mu_f_c, dim=dim)
    sigma_p_k = linear_fluid_stress(v_p_k, p_p_k, mu=mu_p_c, dim=dim)
    sigma_s_k = linear_skeleton_stress(u_s_k, mu=mu_s_c, lambda_=lambda_s_c, dim=dim)
    force_terms = pairwise_internal_forces(
        alpha=alpha_k,
        phi=phi_k,
        v_f=v_f_k,
        v_p=v_p_k,
        v_s=v_s_k,
        sigma_f=sigma_f_k,
        sigma_p=sigma_p_k,
        sigma_s=sigma_s_k,
        R_fp=R_fp_c,
        R_fs=R_fs_c,
        R_ps=R_ps_c,
        R_pair_cholesky=R_pair_cholesky,
        pair_weight_epsilon=pair_weight_epsilon,
        theta_fp=theta_fp_c,
    )

    gamma_key = str(gamma_mobility or "FP").strip().lower().replace("-", "_")
    if gamma_key in {"fp", "overlap", "volume_overlap"}:
        L_Gamma = ell_Gamma_c * F_k * P_k
    elif gamma_key in {"interface_delta", "grad_alpha", "phi_grad_alpha", "delta"}:
        delta_alpha = _pair_weight(
            dot(grad(alpha_k), grad(alpha_k)),
            gamma_delta_epsilon_sq_c,
            epsilon_name="tc_mms_gamma_delta_epsilon_sq",
        )
        L_Gamma = ell_Gamma_c * phi_k * delta_alpha
    elif gamma_key in {"off", "none", "zero"}:
        L_Gamma = _lit(0.0)
    else:
        raise ValueError("Unsupported gamma_mobility. Use 'FP', 'interface_delta', or 'off'.")

    transfer_key = str(transfer_velocity or "average").strip().lower().replace("-", "_")
    if transfer_key in {"average", "avg", "midpoint", "half_sum"}:
        u_Gamma = _lit(0.5) * (v_f_k + v_p_k)
    elif transfer_key in {"free", "vf", "v_f"}:
        u_Gamma = v_f_k
    elif transfer_key in {"pore", "vp", "v_p"}:
        u_Gamma = v_p_k
    else:
        raise ValueError("Unsupported transfer_velocity. Use 'average', 'free', or 'pore'.")

    A_Gamma = _sub(p_f_k, p_p_k) / rho_f_c
    M_gamma_f = _neg(Gamma_k * u_Gamma)
    M_gamma_p = Gamma_k * u_Gamma
    M_gamma_s = _lit(0.0) * v_s_k

    def _mass_source(r_k, r_n, v_k, gamma_term):
        return _sub(r_k, r_n) * inv_dt + div(r_k * v_k) + gamma_term

    def _momentum_source(c_a, r_a_k, r_a_n, v_a_k, v_a_n, sigma_a, I_a, M_gamma_a):
        source = (
            _sub(r_a_k * v_a_k, r_a_n * v_a_n) * inv_dt
            + div(outer(r_a_k * v_a_k, v_a_k))
            + _neg(I_a)
            + _neg(M_gamma_a)
        )
        if bool(include_stress_divergence):
            source = source + _neg(div(c_a * sigma_a))
        return source

    return {
        "S_alpha": _sub(alpha_k, alpha_n) * inv_dt + dot(grad(alpha_k), v_s_k),
        "S_mass_f": _mass_source(r_f_k, r_f_n, v_f_k, Gamma_k),
        "S_mass_p": _mass_source(r_p_k, r_p_n, v_p_k, _neg(Gamma_k)),
        "S_mass_s": _mass_source(r_s_k, r_s_n, v_s_k, _lit(0.0)),
        "S_momentum_f": _momentum_source(F_k, r_f_k, r_f_n, v_f_k, v_f_n, sigma_f_k, force_terms["I_f"], M_gamma_f),
        "S_momentum_p": _momentum_source(P_k, r_p_k, r_p_n, v_p_k, v_p_n, sigma_p_k, force_terms["I_p"], M_gamma_p),
        "S_momentum_s": _momentum_source(B_k, r_s_k, r_s_n, v_s_k, v_s_n, sigma_s_k, force_terms["I_s"], M_gamma_s),
        "S_kinematics": _sub(u_s_k, u_s_n) * inv_dt + dot(grad(u_s_k), v_s_k) + _neg(v_s_k),
        "S_Gamma": _sub(Gamma_k, L_Gamma * A_Gamma),
    }


__all__ = ["backward_euler_three_constituent_sources"]
