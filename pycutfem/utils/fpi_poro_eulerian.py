"""Poroelastic sub-problem (fully Eulerian) forms for FPI.

This module is the minimal, *standalone* porous-medium part needed for MMS/FD
verification (Step 1). It mirrors the "manual Jacobian" style used by
`pycutfem/utils/fsi_fully_eulerian.py`:

  - `residual_poro(...)` returns the residual linear form R(w_k; test)
  - `jacobian_poro(...)` returns the Gateaux derivative J(w_k)[dw, test]

Kinematics
----------
We use the fully Eulerian "reference map" displacement u(x) with

    F = (I - ∇u)^{-1},     J = det(F),

so that all integrals are evaluated over the *spatial* (fixed-grid) porous
domain.
"""

from __future__ import annotations

from pycutfem.ufl.expressions import Constant, div, dot, grad, inner

from pycutfem.utils.nonlinear_solid_eulerian_refmap import (
    deulerian_F as dporo_F,
    deulerian_k_inv as dporo_k_inv,
    dsigma_neo_hookean as dporo_sigma_neo_hookean,
    eulerian_F as poro_F,
    eulerian_k_inv as poro_k_inv,
    sigma_neo_hookean as poro_sigma_neo_hookean,
)


# -----------------------------------------------------------------------------
# Residual / Jacobian
# -----------------------------------------------------------------------------


def residual_poro(
    v_k,
    u_k,
    p_k,
    v_n,
    u_n,
    p_n,
    q_test,
    w_test,
    eta_test,
    *,
    u_nm1=None,
    rho_f,
    mu_f,
    rho_s0_tilde,
    phi,
    K_inv,
    c_nh,
    beta_nh,
    dt,
    theta,
    dx_p,
):
    """Residual for the porous sub-problem on the spatial (Eulerian) domain."""
    # Skeleton velocity via backward Euler
    v_s_k = (u_k - u_n) / dt
    div_v_s_k = (div(u_k) - div(u_n)) / dt
    grad_v_s_k = (grad(u_k) - grad(u_n)) / dt

    if u_nm1 is None:
        # Fallback: first step (or caller doesn't store u_{n-1})
        v_s_n = Constant(0.0) * v_s_k
        div_v_s_n = Constant(0.0) * div_v_s_k
        grad_v_s_n = Constant(0.0) * grad_v_s_k
    else:
        v_s_n = (u_n - u_nm1) / dt
        div_v_s_n = (div(u_n) - div(u_nm1)) / dt
        grad_v_s_n = (grad(u_n) - grad(u_nm1)) / dt

    # Mixture incompressibility: div( φ v + (1-φ) v_s ) = 0
    #
    # IMPORTANT: the current FormCompiler only supports div/grad applied
    # directly to base Trial/Test/Function objects, not to general linear
    # combinations. Expand divergences/gradients explicitly.
    div_mix_k = phi * div(v_k) + (Constant(1.0) - phi) * div_v_s_k
    div_mix_n = phi * div(v_n) + (Constant(1.0) - phi) * div_v_s_n
    div_mix_theta = theta * div_mix_k + (Constant(1.0) - theta) * div_mix_n

    r = q_test * div_mix_theta * dx_p

    # Pore momentum (Darcy-type with inertia and skeleton-advection)
    vdot = (v_k - v_n) / dt
    p_theta = theta * p_k + (Constant(1.0) - theta) * p_n

    r += inner(rho_f * vdot, w_test) * dx_p
    r += -inner(p_theta, div(w_test)) * dx_p

    conv_k = -rho_f * dot(grad(v_k), v_s_k)
    conv_n = -rho_f * dot(grad(v_n), v_s_n)
    r += inner(theta * conv_k + (Constant(1.0) - theta) * conv_n, w_test) * dx_p

    k_inv_k = poro_k_inv(u_k, K_inv)
    k_inv_n = poro_k_inv(u_n, K_inv)
    drag_k = mu_f * (phi * phi) * dot(k_inv_k, (v_k - v_s_k))
    drag_n = mu_f * (phi * phi) * dot(k_inv_n, (v_n - v_s_n))
    drag_theta = theta * drag_k + (Constant(1.0) - theta) * drag_n
    r += inner(drag_theta, w_test) * dx_p

    # Skeleton momentum (Eulerian acceleration + Neo-Hookean stress + pressure + drag reaction)
    acc_local = (v_s_k - v_s_n) / dt
    adv_k = dot(grad_v_s_k, v_s_k)
    adv_n = dot(grad_v_s_n, v_s_n)
    # NOTE: In the paper the skeleton momentum is posed on Ω0^P (Lagrangian), so
    # the inertial term is ∂_t^2 u^P with no explicit advection contribution.
    # When writing the skeleton dynamics on the spatial (Eulerian) domain, the
    # same material acceleration appears as a material derivative in spatial
    # coordinates, i.e. ∂_t v_s + (v_s·∇)v_s, hence the *positive* advective term.
    acc = acc_local + theta * adv_k + (Constant(1.0) - theta) * adv_n

    r += inner(rho_s0_tilde * acc, eta_test) * dx_p

    sig_k = poro_sigma_neo_hookean(u_k, c_nh, beta_nh)
    sig_n = poro_sigma_neo_hookean(u_n, c_nh, beta_nh)
    sig_theta = theta * sig_k + (Constant(1.0) - theta) * sig_n
    r += inner(sig_theta, grad(eta_test)) * dx_p

    r += inner(phi * p_theta, div(eta_test)) * dx_p
    r += -inner(drag_theta, eta_test) * dx_p

    return r


def jacobian_poro(
    v_k,
    u_k,
    p_k,
    u_n,
    dv,
    du,
    dp,
    q_test,
    w_test,
    eta_test,
    *,
    u_nm1=None,
    rho_f,
    mu_f,
    rho_s0_tilde,
    phi,
    K_inv,
    c_nh,
    beta_nh,
    dt,
    theta,
    dx_p,
):
    """Gateaux derivative of `residual_poro` w.r.t. (v_k, u_k, p_k)."""
    v_s_k = (u_k - u_n) / dt
    # v_s_n only affects explicit (1-theta) terms; it has zero derivative wrt (v_k,u_k,p_k).
    v_s_n = Constant(0.0) * v_s_k if u_nm1 is None else (u_n - u_nm1) / dt
    dv_s = du / dt

    # Expand div(dmix) explicitly (compiler does not support div() of composite expressions).
    div_dv_s = div(du) / dt
    div_dmix = theta * (phi * div(dv) + (Constant(1.0) - phi) * div_dv_s)
    a = q_test * div_dmix * dx_p

    a += inner(rho_f * (dv / dt), w_test) * dx_p
    a += -inner(theta * dp, div(w_test)) * dx_p

    dconv_k = -rho_f * (dot(grad(dv), v_s_k) + dot(grad(v_k), dv_s))
    a += inner(theta * dconv_k, w_test) * dx_p

    k_inv_k = poro_k_inv(u_k, K_inv)
    dk_inv_k = dporo_k_inv(u_k, du, K_inv)

    ddrag_k = mu_f * (phi * phi) * (
        dot(dk_inv_k, (v_k - v_s_k)) + dot(k_inv_k, (dv - dv_s))
    )
    a += inner(theta * ddrag_k, w_test) * dx_p

    dacc_local = dv_s / dt
    grad_dv_s = grad(du) / dt
    grad_v_s_k = (grad(u_k) - grad(u_n)) / dt
    dadv_k = dot(grad_dv_s, v_s_k) + dot(grad_v_s_k, dv_s)
    dacc = dacc_local + theta * dadv_k
    a += inner(rho_s0_tilde * dacc, eta_test) * dx_p

    dsig_k = dporo_sigma_neo_hookean(u_k, du, c_nh, beta_nh)
    a += inner(theta * dsig_k, grad(eta_test)) * dx_p

    a += inner(phi * theta * dp, div(eta_test)) * dx_p
    a += -inner(theta * ddrag_k, eta_test) * dx_p

    return a
