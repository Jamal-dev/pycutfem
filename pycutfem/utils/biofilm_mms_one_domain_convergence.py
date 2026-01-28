"""Non-affine manufactured solution for the one-domain biofilm model (convergence studies).

This module provides a smooth (trigonometric) manufactured solution and the
corresponding *discrete* (one-step θ-scheme) forcing terms for:
  (v, p, u, φ, α, S)

Unlike the affine MMS in `biofilm_mms_one_domain.py`, this one is intended for
mesh-refinement convergence studies (non-polynomial exact fields).

Implementation notes
--------------------
* pycutfem's current compiler cannot reliably handle `grad(Analytic(...))`.
  Therefore, all forcing terms are supplied as `Analytic(callable)` without
  taking derivatives of Analytic objects inside UFL forms.
* Forcing terms are constructed to satisfy the *discrete* residual of the
  θ-scheme for a single step t_n -> t_k, using the same difference quotient
  for vS_k = (u_k - u_n)/dt as the model implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class BiofilmOneDomainMMSConvergence:
    t_n: float
    t_k: float
    dt: float
    theta: float

    # Time-dependent exact fields (x,y,t) -> value
    v: callable  # (x,y,t)->(...,2)
    p: callable  # (x,y,t)->(...)
    u: callable  # (x,y,t)->(...,2)
    phi: callable  # (x,y,t)->(...)
    alpha: callable  # (x,y,t)->(...)
    S: callable  # (x,y,t)->(...)

    # Snapshots at t_n and t_k (x,y)->value
    v_n: callable
    p_n: callable
    u_n: callable
    phi_n: callable
    alpha_n: callable
    S_n: callable

    v_k: callable
    p_k: callable
    u_k: callable
    phi_k: callable
    alpha_k: callable
    S_k: callable

    # Forcing terms for the discrete step (x,y)->value
    f_v: callable  # (x,y)->(...,2)
    f_u: callable  # (x,y)->(...,2)
    s_v: callable  # (x,y)->(...)
    f_phi: callable  # (x,y)->(...)
    f_alpha: callable  # (x,y)->(...)
    f_S: callable  # (x,y)->(...)

    # Lagged detachment rate used in α-equation (x,y)->(...)
    D_det_prev: callable


def _sym_grad_vec(v, x, y):
    return sp.Matrix([[sp.diff(v[i], var) for var in (x, y)] for i in range(2)])


def _sym_grad_scalar(f, x, y):
    return sp.Matrix([sp.diff(f, x), sp.diff(f, y)])


def _sym_div_vec(v, x, y):
    return sp.diff(v[0], x) + sp.diff(v[1], y)


def _sym_div_mat(M, x, y):
    # (div M)_i = ∂_x M_{i0} + ∂_y M_{i1}
    return sp.Matrix([sp.diff(M[0, 0], x) + sp.diff(M[0, 1], y), sp.diff(M[1, 0], x) + sp.diff(M[1, 1], y)])


def _sym_laplacian(f, x, y):
    return sp.diff(f, x, 2) + sp.diff(f, y, 2)


def _sym_epsilon(v, x, y):
    G = _sym_grad_vec(v, x, y)
    return sp.Rational(1, 2) * (G + G.T)


def _lambdify_scalar_xy(expr, x, y):
    fn = sp.lambdify((x, y), expr, "numpy")

    def _call(xv, yv, _fn=fn):
        return np.asarray(_fn(xv, yv), dtype=float)

    return _call


def _lambdify_vec_xy(expr_vec, x, y):
    fn = sp.lambdify((x, y), expr_vec, "numpy")

    def _call(xv, yv, _fn=fn):
        arr = np.asarray(_fn(xv, yv), dtype=float)
        # SymPy Matrix(2,1) -> numpy array with leading shape (2,1,...).
        if arr.ndim >= 2 and arr.shape[1] == 1:
            arr = arr[:, 0, ...]
        # Move component axis to the last position: (2,...) -> (...,2).
        return np.moveaxis(arr, 0, -1)

    return _call


def _lambdify_scalar_xyt(expr, x, y, t):
    fn = sp.lambdify((x, y, t), expr, "numpy")

    def _call(xv, yv, tv, _fn=fn):
        return np.asarray(_fn(xv, yv, tv), dtype=float)

    return _call


def _lambdify_vec_xyt(expr_vec, x, y, t):
    fn = sp.lambdify((x, y, t), expr_vec, "numpy")

    def _call(xv, yv, tv, _fn=fn):
        arr = np.asarray(_fn(xv, yv, tv), dtype=float)
        if arr.ndim >= 2 and arr.shape[1] == 1:
            arr = arr[:, 0, ...]
        return np.moveaxis(arr, 0, -1)

    return _call


def build_biofilm_one_domain_mms_trig_step(
    *,
    dt_val: float,
    t0: float = 0.0,
    theta: float = 1.0,
    # physical/transport parameters used in the forcing
    rho_f: float = 1.0,
    mu_f: float = 1.0e-2,
    kappa_inv: float = 10.0,
    mu_s: float = 1.0,
    lambda_s: float = 1.0,
    D_phi: float = 0.1,
    gamma_phi: float = 1.0,
    D_alpha: float = 0.1,
    D_S: float = 0.1,
    mu_max: float = 0.4,
    K_S: float = 0.3,
    k_g: float = 0.5,
    k_d: float = 0.1,
    Y: float = 0.8,
    k_det: float = 0.2,
    eta_n: float = 1.0e-12,
) -> BiofilmOneDomainMMSConvergence:
    """
    Build a trigonometric MMS for a single θ-scheme step t_n=t0 -> t_k=t0+dt.

    The returned forcing terms (f_v, f_u, s_v, f_phi, f_alpha, f_S) are
    consistent with `build_biofilm_one_domain_forms` under the default
    "phi_mu" viscosity choice.
    """
    dt = float(dt_val)
    if not (dt > 0.0):
        raise ValueError("dt_val must be positive.")
    t_n = float(t0)
    t_k = float(t0) + dt
    th = float(theta)
    if not (0.0 <= th <= 1.0):
        raise ValueError("theta must be in [0,1].")
    one_m_th = 1.0 - th

    # ------------------------------------------------------------------
    # Symbolic exact fields (smooth, bounded scalars).
    # ------------------------------------------------------------------
    x, y, t = sp.symbols("x y t", real=True)
    s = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)

    v_expr = sp.Matrix(
        [
            s * (1.0 + 0.5 * sp.sin(t)),
            s * (1.0 + 0.5 * sp.cos(t)),
        ]
    )
    u_expr = sp.Matrix(
        [
            s * (1.0 + 0.25 * sp.cos(t)),
            s * (1.0 + 0.25 * sp.sin(t)),
        ]
    )
    p_expr = s * (1.0 + 0.3 * sp.sin(t))

    phi_expr = 0.7 + 0.1 * s * (1.0 + 0.2 * sp.cos(t))
    alpha_expr = 0.5 + 0.2 * s * (1.0 + 0.1 * sp.sin(t))
    S_expr = 0.4 + 0.1 * s * (1.0 + 0.2 * sp.cos(t))

    # Time-dependent lambdas
    v_t = _lambdify_vec_xyt(v_expr, x, y, t)
    u_t = _lambdify_vec_xyt(u_expr, x, y, t)
    p_t = _lambdify_scalar_xyt(p_expr, x, y, t)
    phi_t = _lambdify_scalar_xyt(phi_expr, x, y, t)
    alpha_t = _lambdify_scalar_xyt(alpha_expr, x, y, t)
    S_t = _lambdify_scalar_xyt(S_expr, x, y, t)

    # Snapshots
    subs_n = {t: t_n}
    subs_k = {t: t_k}
    subs_nm1 = {t: t_n - dt}

    v_n_expr = v_expr.subs(subs_n)
    v_k_expr = v_expr.subs(subs_k)
    v_nm1_expr = v_expr.subs(subs_nm1)

    u_n_expr = u_expr.subs(subs_n)
    u_k_expr = u_expr.subs(subs_k)
    u_nm1_expr = u_expr.subs(subs_nm1)

    p_n_expr = p_expr.subs(subs_n)
    p_k_expr = p_expr.subs(subs_k)

    phi_n_expr = phi_expr.subs(subs_n)
    phi_k_expr = phi_expr.subs(subs_k)

    alpha_n_expr = alpha_expr.subs(subs_n)
    alpha_k_expr = alpha_expr.subs(subs_k)

    S_n_expr = S_expr.subs(subs_n)
    S_k_expr = S_expr.subs(subs_k)

    v_n = _lambdify_vec_xy(v_n_expr, x, y)
    v_k = _lambdify_vec_xy(v_k_expr, x, y)
    p_n = _lambdify_scalar_xy(p_n_expr, x, y)
    p_k = _lambdify_scalar_xy(p_k_expr, x, y)
    u_n = _lambdify_vec_xy(u_n_expr, x, y)
    u_k = _lambdify_vec_xy(u_k_expr, x, y)
    phi_n = _lambdify_scalar_xy(phi_n_expr, x, y)
    phi_k = _lambdify_scalar_xy(phi_k_expr, x, y)
    alpha_n = _lambdify_scalar_xy(alpha_n_expr, x, y)
    alpha_k = _lambdify_scalar_xy(alpha_k_expr, x, y)
    S_n = _lambdify_scalar_xy(S_n_expr, x, y)
    S_k = _lambdify_scalar_xy(S_k_expr, x, y)

    # ------------------------------------------------------------------
    # Derived quantities for the discrete step (vS via BE difference).
    # ------------------------------------------------------------------
    vS_k_expr = (u_k_expr - u_n_expr) / dt
    vS_n_expr = (u_n_expr - u_nm1_expr) / dt

    div_vS_k_expr = _sym_div_vec(vS_k_expr, x, y)
    div_vS_n_expr = _sym_div_vec(vS_n_expr, x, y)

    # ------------------------------------------------------------------
    # Coefficients and nonlinear reactions (phi_mu choice).
    # ------------------------------------------------------------------
    C_k_expr = (1.0 - alpha_k_expr) + alpha_k_expr * phi_k_expr
    C_n_expr = (1.0 - alpha_n_expr) + alpha_n_expr * phi_n_expr

    rho_k_expr = float(rho_f) * C_k_expr
    rho_n_expr = float(rho_f) * C_n_expr
    mu_k_expr = float(mu_f) * C_k_expr
    mu_n_expr = float(mu_f) * C_n_expr

    beta_k_expr = alpha_k_expr * float(mu_f) * (phi_k_expr * phi_k_expr) * float(kappa_inv)
    beta_n_expr = alpha_n_expr * float(mu_f) * (phi_n_expr * phi_n_expr) * float(kappa_inv)

    mon_k_expr = float(mu_max) * (S_k_expr / (S_k_expr + float(K_S)))
    mon_n_expr = float(mu_max) * (S_n_expr / (S_n_expr + float(K_S)))

    Pi_k_expr = (mon_k_expr - float(k_d)) * (1.0 - phi_k_expr) * alpha_k_expr
    Pi_n_expr = (mon_n_expr - float(k_d)) * (1.0 - phi_n_expr) * alpha_n_expr

    G_k_expr = float(k_g) * mon_k_expr * (1.0 - phi_k_expr)
    G_n_expr = float(k_g) * mon_n_expr * (1.0 - phi_n_expr)

    RS_k_expr = (1.0 / float(Y)) * Pi_k_expr
    RS_n_expr = (1.0 / float(Y)) * Pi_n_expr

    # Lagged detachment from previous velocity v_n (consistent with the model's default).
    eps_vn = _sym_epsilon(v_n_expr, x, y)
    tau2 = sum(eps_vn[i, j] * eps_vn[i, j] for i in range(2) for j in range(2))
    D_det_prev_expr = float(k_det) * sp.sqrt(tau2 + float(eta_n))

    # ------------------------------------------------------------------
    # Momentum forcing
    # ------------------------------------------------------------------
    vdot_expr = (v_k_expr - v_n_expr) / dt
    conv_k_expr = _sym_grad_vec(v_k_expr, x, y) * v_k_expr
    conv_n_expr = _sym_grad_vec(v_n_expr, x, y) * v_n_expr

    stress_k = mu_k_expr * (_sym_grad_vec(v_k_expr, x, y) + _sym_grad_vec(v_k_expr, x, y).T)
    stress_n = mu_n_expr * (_sym_grad_vec(v_n_expr, x, y) + _sym_grad_vec(v_n_expr, x, y).T)
    div_stress_mix = _sym_div_mat(th * stress_k + one_m_th * stress_n, x, y)

    f_v_expr = rho_k_expr * vdot_expr
    f_v_expr += th * rho_k_expr * conv_k_expr + one_m_th * rho_n_expr * conv_n_expr
    f_v_expr += -div_stress_mix
    f_v_expr += _sym_grad_scalar(p_k_expr, x, y)
    f_v_expr += beta_k_expr * (v_k_expr - vS_k_expr)

    # ------------------------------------------------------------------
    # Mass source: div(F) = α s_v
    # ------------------------------------------------------------------
    B_k_expr = alpha_k_expr * (1.0 - phi_k_expr)
    B_n_expr = alpha_n_expr * (1.0 - phi_n_expr)

    F_k = C_k_expr * v_k_expr + B_k_expr * vS_k_expr
    F_n = C_n_expr * v_n_expr + B_n_expr * vS_n_expr

    divF_k = _sym_div_vec(F_k, x, y)
    divF_n = _sym_div_vec(F_n, x, y)
    s_v_expr = (th * divF_k + one_m_th * divF_n) / alpha_k_expr

    # ------------------------------------------------------------------
    # Skeleton forcing: -(div(α σ)) - αβ(v-vS) = α f_u  (θ-averaged)
    # ------------------------------------------------------------------
    I = sp.eye(2)

    def _stress_s(u_expr_t, p_expr_t, phi_expr_t):
        eps_u = _sym_epsilon(u_expr_t, x, y)
        div_u = _sym_div_vec(u_expr_t, x, y)
        elastic = 2.0 * float(mu_s) * eps_u + float(lambda_s) * div_u * I
        pore_p = -(1.0 - phi_expr_t) * p_expr_t * I
        return elastic + pore_p

    sigma_k = _stress_s(u_k_expr, p_k_expr, phi_k_expr)
    sigma_n = _stress_s(u_n_expr, p_n_expr, phi_n_expr)

    op_u_k = -_sym_div_mat(alpha_k_expr * sigma_k, x, y) - alpha_k_expr * beta_k_expr * (v_k_expr - vS_k_expr)
    op_u_n = -_sym_div_mat(alpha_n_expr * sigma_n, x, y) - alpha_n_expr * beta_n_expr * (v_n_expr - vS_n_expr)
    f_u_expr = (th * op_u_k + one_m_th * op_u_n) / alpha_k_expr

    # ------------------------------------------------------------------
    # Porosity forcing (implicit diffusion at k)
    # ------------------------------------------------------------------
    Fphi_k = _sym_grad_scalar(phi_k_expr, x, y).dot(vS_k_expr) - (1.0 - phi_k_expr) * div_vS_k_expr + Pi_k_expr
    Fphi_n = _sym_grad_scalar(phi_n_expr, x, y).dot(vS_n_expr) - (1.0 - phi_n_expr) * div_vS_n_expr + Pi_n_expr
    f_phi_expr = alpha_k_expr * (phi_k_expr - phi_n_expr) / dt
    f_phi_expr += th * alpha_k_expr * Fphi_k + one_m_th * alpha_n_expr * Fphi_n
    f_phi_expr += -float(D_phi) * _sym_laplacian(phi_k_expr, x, y)
    f_phi_expr += float(gamma_phi) * (1.0 - alpha_k_expr) * (phi_k_expr - 1.0)

    # ------------------------------------------------------------------
    # Indicator forcing (implicit diffusion at k)
    # ------------------------------------------------------------------
    Fal_k = _sym_grad_scalar(alpha_k_expr, x, y).dot(vS_k_expr) - G_k_expr * alpha_k_expr * (1.0 - alpha_k_expr) + D_det_prev_expr * alpha_k_expr
    Fal_n = _sym_grad_scalar(alpha_n_expr, x, y).dot(vS_n_expr) - G_n_expr * alpha_n_expr * (1.0 - alpha_n_expr) + D_det_prev_expr * alpha_n_expr
    f_alpha_expr = (alpha_k_expr - alpha_n_expr) / dt
    f_alpha_expr += th * Fal_k + one_m_th * Fal_n
    f_alpha_expr += -float(D_alpha) * _sym_laplacian(alpha_k_expr, x, y)

    # ------------------------------------------------------------------
    # Substrate forcing (θ-averaged diffusion)
    # ------------------------------------------------------------------
    CS_k_expr = C_k_expr * S_k_expr
    CS_n_expr = C_n_expr * S_n_expr
    div_adv_k = _sym_div_vec(CS_k_expr * v_k_expr, x, y)
    div_adv_n = _sym_div_vec(CS_n_expr * v_n_expr, x, y)
    f_S_expr = (CS_k_expr - CS_n_expr) / dt
    f_S_expr += th * div_adv_k + one_m_th * div_adv_n
    f_S_expr += -float(D_S) * (th * _sym_laplacian(S_k_expr, x, y) + one_m_th * _sym_laplacian(S_n_expr, x, y))
    f_S_expr += th * RS_k_expr + one_m_th * RS_n_expr

    # Lambdify final forcing
    f_v = _lambdify_vec_xy(f_v_expr, x, y)
    f_u = _lambdify_vec_xy(f_u_expr, x, y)
    s_v = _lambdify_scalar_xy(s_v_expr, x, y)
    f_phi = _lambdify_scalar_xy(f_phi_expr, x, y)
    f_alpha = _lambdify_scalar_xy(f_alpha_expr, x, y)
    f_S = _lambdify_scalar_xy(f_S_expr, x, y)
    D_det_prev = _lambdify_scalar_xy(D_det_prev_expr, x, y)

    return BiofilmOneDomainMMSConvergence(
        t_n=t_n,
        t_k=t_k,
        dt=dt,
        theta=th,
        v=v_t,
        p=p_t,
        u=u_t,
        phi=phi_t,
        alpha=alpha_t,
        S=S_t,
        v_n=v_n,
        p_n=p_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        S_n=S_n,
        v_k=v_k,
        p_k=p_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        S_k=S_k,
        f_v=f_v,
        f_u=f_u,
        s_v=s_v,
        f_phi=f_phi,
        f_alpha=f_alpha,
        f_S=f_S,
        D_det_prev=D_det_prev,
    )

