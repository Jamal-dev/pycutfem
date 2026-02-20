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
  θ-scheme for a single step t_n -> t_k. The solid velocity vS is treated as a
  primary unknown; the kinematic constraint between (u, vS) is enforced in the
  model without a forcing term, so the manufactured (u, vS) must satisfy it
  exactly.
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
    vS: callable  # (x,y,t)->(...,2)
    u: callable  # (x,y,t)->(...,2)
    phi: callable  # (x,y,t)->(...)
    alpha: callable  # (x,y,t)->(...)
    S: callable  # (x,y,t)->(...)
    X: callable  # (x,y,t)->(...)

    # Snapshots at t_n and t_k (x,y)->value
    v_n: callable
    p_n: callable
    vS_n: callable
    u_n: callable
    phi_n: callable
    alpha_n: callable
    S_n: callable
    X_n: callable

    v_k: callable
    p_k: callable
    vS_k: callable
    u_k: callable
    phi_k: callable
    alpha_k: callable
    S_k: callable
    X_k: callable

    # Forcing terms for the discrete step (x,y)->value
    f_v: callable  # (x,y)->(...,2)
    f_u: callable  # (x,y)->(...,2)
    s_v: callable  # (x,y)->(...)
    f_phi: callable  # (x,y)->(...)
    f_alpha: callable  # (x,y)->(...)
    f_S: callable  # (x,y)->(...)
    f_X: callable  # (x,y)->(...)

    # Lagged detachment rate used in α-equation (x,y)->(...)
    D_det_prev: callable

    # Optional: conservative Allen–Cahn global multiplier (constants for the step)
    lambda_alpha_n: float = 0.0
    lambda_alpha_k: float = 0.0

    # Optional: Allen–Cahn settings used to generate f_alpha
    alpha_cahn_M: float = 0.0
    alpha_cahn_gamma: float = 0.0
    alpha_cahn_eps: float = 1.0
    alpha_cahn_conservative: bool = False
    alpha_cahn_mobility: str = "constant"

    # Optional: Cahn–Hilliard chemical potential field μ_α (x,y,t)->(...)
    mu_alpha: callable | None = None
    mu_alpha_n: callable | None = None
    mu_alpha_k: callable | None = None

    # Optional: Cahn–Hilliard settings used to generate f_alpha
    alpha_ch_M: float = 0.0
    alpha_ch_gamma: float = 0.0
    alpha_ch_eps: float = 1.0
    alpha_ch_mobility: str = "constant"


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
    # Optional Cahn–Hilliard / phase-field regularization for α (adds μ_α and a 4th-order operator).
    alpha_ch_M: float = 0.0,
    alpha_ch_gamma: float = 0.0,
    alpha_ch_eps: float = 1.0,
    alpha_ch_mobility: str = "constant",
    # Optional Allen–Cahn / phase-field regularization for α (implicit at k, as in the model).
    alpha_cahn_M: float = 0.0,
    alpha_cahn_gamma: float = 0.0,
    alpha_cahn_eps: float = 1.0,
    alpha_cahn_conservative: bool = False,
    alpha_cahn_mobility: str = "constant",
    alpha_wave: int = 1,
    D_S: float = 0.1,
    D_X: float = 0.1,
    mu_max: float = 0.4,
    K_S: float = 0.3,
    k_g: float = 0.5,
    k_d: float = 0.1,
    Y: float = 0.8,
    rho_s_star: float = 1.0,
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
    if int(alpha_wave) < 1:
        raise ValueError("alpha_wave must be >= 1.")
    s_alpha = sp.sin(int(alpha_wave) * sp.pi * x) * sp.sin(int(alpha_wave) * sp.pi * y)

    v_expr = sp.Matrix(
        [
            s * (1.0 + 0.5 * sp.sin(t)),
            s * (1.0 + 0.5 * sp.cos(t)),
        ]
    )
    # Solid velocity vS is a primary unknown. Choose a simple constant (in space)
    # translation velocity and an affine u that satisfies the Eulerian kinematic
    # constraint exactly:
    #   ∂_t u + vS·∇u = vS.
    #
    # With u(x,y,t) = A*[x,y] + b*(t-t_n) and vS constant, grad(u)=A and the
    # constraint reduces to b + A*vS = vS, i.e. b = (I-A)*vS. This satisfies the
    # discrete θ-scheme kinematics for any θ and dt (since all terms are constant
    # over the step).
    vS_expr = sp.Matrix([0.12, -0.08])
    A = sp.Matrix([[0.05, 0.02], [0.02, 0.04]])
    b = (sp.eye(2) - A) * vS_expr
    u_expr = A * sp.Matrix([x, y]) + b * (t - sp.Float(t_n))
    p_expr = s * (1.0 + 0.3 * sp.sin(t))

    phi_expr = 0.7 + 0.1 * s * (1.0 + 0.2 * sp.cos(t))
    alpha_expr = 0.5 + 0.2 * s_alpha * (1.0 + 0.1 * sp.sin(t))
    S_expr = 0.4 + 0.1 * s * (1.0 + 0.2 * sp.cos(t))
    X_expr = 0.1 + 0.05 * s * (1.0 + 0.2 * sp.sin(t))

    # Time-dependent lambdas
    v_t = _lambdify_vec_xyt(v_expr, x, y, t)
    vS_t = _lambdify_vec_xyt(vS_expr, x, y, t)
    u_t = _lambdify_vec_xyt(u_expr, x, y, t)
    p_t = _lambdify_scalar_xyt(p_expr, x, y, t)
    phi_t = _lambdify_scalar_xyt(phi_expr, x, y, t)
    alpha_t = _lambdify_scalar_xyt(alpha_expr, x, y, t)
    S_t = _lambdify_scalar_xyt(S_expr, x, y, t)
    X_t = _lambdify_scalar_xyt(X_expr, x, y, t)

    # Snapshots
    subs_n = {t: t_n}
    subs_k = {t: t_k}

    v_n_expr = v_expr.subs(subs_n)
    v_k_expr = v_expr.subs(subs_k)

    u_n_expr = u_expr.subs(subs_n)
    u_k_expr = u_expr.subs(subs_k)
    vS_n_expr = vS_expr.subs(subs_n)
    vS_k_expr = vS_expr.subs(subs_k)

    p_n_expr = p_expr.subs(subs_n)
    p_k_expr = p_expr.subs(subs_k)

    phi_n_expr = phi_expr.subs(subs_n)
    phi_k_expr = phi_expr.subs(subs_k)

    alpha_n_expr = alpha_expr.subs(subs_n)
    alpha_k_expr = alpha_expr.subs(subs_k)

    S_n_expr = S_expr.subs(subs_n)
    S_k_expr = S_expr.subs(subs_k)
    X_n_expr = X_expr.subs(subs_n)
    X_k_expr = X_expr.subs(subs_k)

    v_n = _lambdify_vec_xy(v_n_expr, x, y)
    v_k = _lambdify_vec_xy(v_k_expr, x, y)
    p_n = _lambdify_scalar_xy(p_n_expr, x, y)
    p_k = _lambdify_scalar_xy(p_k_expr, x, y)
    vS_n = _lambdify_vec_xy(vS_n_expr, x, y)
    vS_k = _lambdify_vec_xy(vS_k_expr, x, y)
    u_n = _lambdify_vec_xy(u_n_expr, x, y)
    u_k = _lambdify_vec_xy(u_k_expr, x, y)
    phi_n = _lambdify_scalar_xy(phi_n_expr, x, y)
    phi_k = _lambdify_scalar_xy(phi_k_expr, x, y)
    alpha_n = _lambdify_scalar_xy(alpha_n_expr, x, y)
    alpha_k = _lambdify_scalar_xy(alpha_k_expr, x, y)
    S_n = _lambdify_scalar_xy(S_n_expr, x, y)
    S_k = _lambdify_scalar_xy(S_k_expr, x, y)
    X_n = _lambdify_scalar_xy(X_n_expr, x, y)
    X_k = _lambdify_scalar_xy(X_k_expr, x, y)

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

    RS_k_expr = float(rho_s_star) * (1.0 / float(Y)) * Pi_k_expr
    RS_n_expr = float(rho_s_star) * (1.0 / float(Y)) * Pi_n_expr

    # Lagged detachment from previous velocity v_n (consistent with the model's default).
    eps_vn = _sym_epsilon(v_n_expr, x, y)
    tau2 = sum(eps_vn[i, j] * eps_vn[i, j] for i in range(2) for j in range(2))
    D_det_prev_expr = float(k_det) * sp.sqrt(tau2 + float(eta_n))

    # ------------------------------------------------------------------
    # Momentum forcing
    # ------------------------------------------------------------------
    # Conservative-in-time momentum: (rho_k v_k - rho_n v_n)/dt.
    momdot_expr = (rho_k_expr * v_k_expr - rho_n_expr * v_n_expr) / dt
    conv_k_expr = _sym_grad_vec(v_k_expr, x, y) * v_k_expr
    conv_n_expr = _sym_grad_vec(v_n_expr, x, y) * v_n_expr

    # Conservative convection correction: v div(rho v), with rho=rho_f*C.
    divCv_k_expr = _sym_div_vec(C_k_expr * v_k_expr, x, y)
    divCv_n_expr = _sym_div_vec(C_n_expr * v_n_expr, x, y)
    div_rhov_k_expr = float(rho_f) * divCv_k_expr
    div_rhov_n_expr = float(rho_f) * divCv_n_expr

    stress_k = mu_k_expr * (_sym_grad_vec(v_k_expr, x, y) + _sym_grad_vec(v_k_expr, x, y).T)
    stress_n = mu_n_expr * (_sym_grad_vec(v_n_expr, x, y) + _sym_grad_vec(v_n_expr, x, y).T)
    div_stress_mix = _sym_div_mat(th * stress_k + one_m_th * stress_n, x, y)

    f_v_expr = momdot_expr
    f_v_expr += th * (rho_k_expr * conv_k_expr + div_rhov_k_expr * v_k_expr)
    f_v_expr += one_m_th * (rho_n_expr * conv_n_expr + div_rhov_n_expr * v_n_expr)
    f_v_expr += -div_stress_mix
    # Pressure coupling in `build_biofilm_one_domain_forms` is the variationally
    # consistent `-(p, div(C w))`, which corresponds (in strong form) to `C ∇p`.
    f_v_expr += C_k_expr * _sym_grad_scalar(p_k_expr, x, y)
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
    # Skeleton forcing: -(div(α σ)) - β(v-vS) = α f_u  (θ-averaged)
    # ------------------------------------------------------------------
    I = sp.eye(2)

    def _stress_s(u_expr_t):
        eps_u = _sym_epsilon(u_expr_t, x, y)
        div_u = _sym_div_vec(u_expr_t, x, y)
        elastic = 2.0 * float(mu_s) * eps_u + float(lambda_s) * div_u * I
        return elastic

    sigma_k = _stress_s(u_k_expr)
    sigma_n = _stress_s(u_n_expr)

    # In `build_biofilm_one_domain_forms`, the pressure coupling in the skeleton
    # equation uses the adjoint form `-(p, div(B η))`, which corresponds to the
    # strong term `B ∇p` (not ∇(B p)).
    B_k_expr = alpha_k_expr * (1.0 - phi_k_expr)
    B_n_expr = alpha_n_expr * (1.0 - phi_n_expr)

    op_u_k = -_sym_div_mat(alpha_k_expr * sigma_k, x, y)
    op_u_k += B_k_expr * _sym_grad_scalar(p_k_expr, x, y)
    op_u_k += -beta_k_expr * (v_k_expr - vS_k_expr)

    op_u_n = -_sym_div_mat(alpha_n_expr * sigma_n, x, y)
    op_u_n += B_n_expr * _sym_grad_scalar(p_n_expr, x, y)
    op_u_n += -beta_n_expr * (v_n_expr - vS_n_expr)
    f_u_expr = (th * op_u_k + one_m_th * op_u_n) / alpha_k_expr

    # ------------------------------------------------------------------
    # Porosity forcing (implicit diffusion at k)
    # ------------------------------------------------------------------
    Fphi_k = _sym_grad_scalar(phi_k_expr, x, y).dot(vS_k_expr) - (1.0 - phi_k_expr) * div_vS_k_expr + Pi_k_expr
    Fphi_n = _sym_grad_scalar(phi_n_expr, x, y).dot(vS_n_expr) - (1.0 - phi_n_expr) * div_vS_n_expr + Pi_n_expr
    f_phi_expr = alpha_k_expr * (phi_k_expr - phi_n_expr) / dt
    f_phi_expr += th * alpha_k_expr * Fphi_k + one_m_th * alpha_n_expr * Fphi_n
    f_phi_expr += -float(D_phi) * _sym_laplacian(phi_k_expr, x, y)
    # Match `build_biofilm_one_domain_forms`: sharpen the fluid-region constraint.
    f_phi_expr += float(gamma_phi) * (1.0 - alpha_k_expr) ** 16 * (phi_k_expr - 1.0)

    # ------------------------------------------------------------------
    # Indicator forcing (implicit diffusion at k)
    # ------------------------------------------------------------------
    delta_k_expr = 4.0 * alpha_k_expr * (1.0 - alpha_k_expr)
    delta_n_expr = 4.0 * alpha_n_expr * (1.0 - alpha_n_expr)
    # NOTE: In the one-domain implementation, the surface-localized erosion /
    # detachment sink is treated as a negative RHS term; therefore it enters the
    # residual with a + sign.
    Fal_k = (
        _sym_grad_scalar(alpha_k_expr, x, y).dot(vS_k_expr)
        + alpha_k_expr * div_vS_k_expr
        - G_k_expr * alpha_k_expr * (1.0 - alpha_k_expr)
        + D_det_prev_expr * delta_k_expr
    )
    Fal_n = (
        _sym_grad_scalar(alpha_n_expr, x, y).dot(vS_n_expr)
        + alpha_n_expr * div_vS_n_expr
        - G_n_expr * alpha_n_expr * (1.0 - alpha_n_expr)
        + D_det_prev_expr * delta_n_expr
    )
    f_alpha_expr = (alpha_k_expr - alpha_n_expr) / dt
    f_alpha_expr += th * Fal_k + one_m_th * Fal_n
    f_alpha_expr += -float(D_alpha) * _sym_laplacian(alpha_k_expr, x, y)

    # Optional Cahn–Hilliard regularization for α (mass-conserving, adds μ_α).
    ch_enabled = float(alpha_ch_M) != 0.0 and float(alpha_ch_gamma) != 0.0

    # Optional Allen–Cahn / phase-field regularization for α (implicit at k).
    ac_enabled = float(alpha_cahn_M) != 0.0 and float(alpha_cahn_gamma) != 0.0
    if ch_enabled and ac_enabled:
        raise ValueError("Allen–Cahn (alpha_cahn_*) and Cahn–Hilliard (alpha_ch_*) cannot both be enabled.")
    if ch_enabled and bool(alpha_cahn_conservative):
        raise ValueError("alpha_cahn_conservative cannot be used together with Cahn–Hilliard regularization (alpha_ch_*).")

    mu_alpha_expr = None
    mu_alpha_n_expr = None
    mu_alpha_k_expr = None
    if ch_enabled:
        eps_ch = float(alpha_ch_eps)
        if not (eps_ch > 0.0):
            raise ValueError("alpha_ch_eps must be > 0 when alpha_ch_M and alpha_ch_gamma are enabled.")

        # Mobility options mirror the model implementation.
        mob_key = str(alpha_ch_mobility).strip().lower()
        if mob_key in {"constant", "const"}:
            M_ch_k_expr = float(alpha_ch_M)
            M_ch_n_expr = float(alpha_ch_M)
        elif mob_key in {"degenerate", "deg", "alpha(1-alpha)", "alpha*(1-alpha)", "a(1-a)"}:
            M_ch_k_expr = float(alpha_ch_M) * alpha_k_expr * (1.0 - alpha_k_expr)
            M_ch_n_expr = float(alpha_ch_M) * alpha_n_expr * (1.0 - alpha_n_expr)
        else:
            raise ValueError(f"Unknown alpha_ch_mobility {alpha_ch_mobility!r}. Use 'constant' or 'degenerate'.")

        # Chemical potential μ_α = γ(-εΔα + (1/ε)W'(α)), with W'(α)=2α(1-α)(1-2α).
        Wp_k_ch = 2.0 * alpha_k_expr * (1.0 - alpha_k_expr) * (1.0 - 2.0 * alpha_k_expr)
        Wp_n_ch = 2.0 * alpha_n_expr * (1.0 - alpha_n_expr) * (1.0 - 2.0 * alpha_n_expr)
        mu_alpha_k_expr = float(alpha_ch_gamma) * ((-eps_ch) * _sym_laplacian(alpha_k_expr, x, y) + (Wp_k_ch / eps_ch))
        mu_alpha_n_expr = float(alpha_ch_gamma) * ((-eps_ch) * _sym_laplacian(alpha_n_expr, x, y) + (Wp_n_ch / eps_ch))

        # Full time-dependent chemical potential for boundary conditions / diagnostics.
        Wp_t_ch = 2.0 * alpha_expr * (1.0 - alpha_expr) * (1.0 - 2.0 * alpha_expr)
        mu_alpha_expr = float(alpha_ch_gamma) * ((-eps_ch) * _sym_laplacian(alpha_expr, x, y) + (Wp_t_ch / eps_ch))

        # Strong CH operator in the alpha equation: -div(M(α) ∇μ_α) (theta-averaged).
        div_flux_k = _sym_div_vec(M_ch_k_expr * _sym_grad_scalar(mu_alpha_k_expr, x, y), x, y)
        div_flux_n = _sym_div_vec(M_ch_n_expr * _sym_grad_scalar(mu_alpha_n_expr, x, y), x, y)
        f_alpha_expr += -(th * div_flux_k + one_m_th * div_flux_n)

    lambda_alpha_n = 0.0
    lambda_alpha_k = 0.0
    if ac_enabled:
        eps_alpha = float(alpha_cahn_eps)
        if not (eps_alpha > 0.0):
            raise ValueError("alpha_cahn_eps must be > 0 when alpha_cahn_M and alpha_cahn_gamma are enabled.")

        # Mobility options mirror the model implementation.
        mob_key = str(alpha_cahn_mobility).strip().lower()
        if mob_key in {"constant", "const"}:
            M_ac_k_expr = float(alpha_cahn_M)
            M_ac_n_expr = float(alpha_cahn_M)
        elif mob_key in {"degenerate", "deg", "alpha(1-alpha)", "alpha*(1-alpha)", "a(1-a)"}:
            M_ac_k_expr = float(alpha_cahn_M) * alpha_k_expr * (1.0 - alpha_k_expr)
            M_ac_n_expr = float(alpha_cahn_M) * alpha_n_expr * (1.0 - alpha_n_expr)
        else:
            raise ValueError(f"Unknown alpha_cahn_mobility {alpha_cahn_mobility!r}. Use 'constant' or 'degenerate'.")

        M_gamma_k_expr = float(alpha_cahn_gamma) * M_ac_k_expr

        # W'(α) for W(α)=α^2(1-α)^2 is 2α(1-α)(1-2α).
        Wp_k_expr = 2.0 * alpha_k_expr * (1.0 - alpha_k_expr) * (1.0 - 2.0 * alpha_k_expr)

        # Strong Allen–Cahn operator: -div(Mγ ε ∇α) + (Mγ/ε) W'(α).
        # Use the exact product rule for variable mobility to match the implementation.
        div_Mgrad = sp.diff(M_gamma_k_expr * sp.diff(alpha_k_expr, x), x) + sp.diff(M_gamma_k_expr * sp.diff(alpha_k_expr, y), y)
        ac_term = -eps_alpha * div_Mgrad + (M_gamma_k_expr / eps_alpha) * Wp_k_expr

        if bool(alpha_cahn_conservative):
            # For this MMS, α(t,x,y) is of the form 0.5 + A(t) * sin(kπx)sin(kπy).
            # Compute λ_α(t) exactly from the constraint:
            #   λ = (∫ M(α) μ dx)/(∫ M(α) dx),  μ=γ(-εΔα + (1/ε)W'(α)).
            k = int(alpha_wave)
            pi = float(np.pi)

            def _A(tval: float) -> float:
                return 0.2 * (1.0 + 0.1 * float(np.sin(float(tval))))

            def _sin_int_odd(kv: int, *, power: int) -> float:
                if kv % 2 == 0:
                    return 0.0
                if power == 1:
                    return 2.0 / (kv * pi)
                if power == 3:
                    return 4.0 / (3.0 * kv * pi)
                if power == 5:
                    return 16.0 / (15.0 * kv * pi)
                raise ValueError(f"Unsupported power={power} for sine integral.")

            def _lambda_for_A(Aval: float) -> float:
                # Moments over the unit square of s = sin(kπx)sin(kπy).
                I1 = _sin_int_odd(k, power=1)
                I3 = _sin_int_odd(k, power=3)
                I5 = _sin_int_odd(k, power=5)
                m1 = I1 * I1
                m3 = I3 * I3
                m5 = I5 * I5

                # μ = γ( -εΔα + (1/ε)W'(α) ), with α = 0.5 + δ, δ = A s:
                #   Δα = -2(kπ)^2 A s
                #   W'(α) = -δ + 4δ^3 = -A s + 4A^3 s^3
                c1 = Aval * (2.0 * eps_alpha * ((k * pi) ** 2) - (1.0 / eps_alpha))
                c3 = (4.0 * (Aval**3)) / eps_alpha

                if mob_key in {"constant", "const"}:
                    return float(alpha_cahn_gamma) * (c1 * m1 + c3 * m3)

                # Degenerate mobility: M(α)=M0 α(1-α) = M0(0.25 - δ^2), M0 cancels in λ.
                # Use exact mean(s^2)=1/4 for integer k.
                denom = (1.0 - Aval * Aval) / 4.0
                if abs(denom) < 1.0e-14:
                    raise ValueError("Degenerate mobility denominator too small (A too large).")

                num = 0.25 * c1 * m1
                num += (0.25 * c3 - (Aval * Aval) * c1) * m3
                num += -(Aval * Aval) * c3 * m5
                return float(alpha_cahn_gamma) * (num / denom)

            lambda_alpha_n = _lambda_for_A(_A(t_n))
            lambda_alpha_k = _lambda_for_A(_A(t_k))
            ac_term = ac_term - M_ac_k_expr * float(lambda_alpha_k)

        f_alpha_expr += ac_term

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

    # ------------------------------------------------------------------
    # Detached biomass forcing (θ-averaged diffusion)
    # ------------------------------------------------------------------
    CX_k_expr = C_k_expr * X_k_expr
    CX_n_expr = C_n_expr * X_n_expr
    div_advX_k = _sym_div_vec(CX_k_expr * v_k_expr, x, y)
    div_advX_n = _sym_div_vec(CX_n_expr * v_n_expr, x, y)

    R_det_k_expr = float(rho_s_star) * (1.0 - phi_k_expr) * D_det_prev_expr * delta_k_expr
    R_det_n_expr = float(rho_s_star) * (1.0 - phi_n_expr) * D_det_prev_expr * delta_n_expr

    f_X_expr = (CX_k_expr - CX_n_expr) / dt
    f_X_expr += th * div_advX_k + one_m_th * div_advX_n
    f_X_expr += -float(D_X) * (th * _sym_laplacian(X_k_expr, x, y) + one_m_th * _sym_laplacian(X_n_expr, x, y))
    f_X_expr += -(th * R_det_k_expr + one_m_th * R_det_n_expr)

    # Lambdify final forcing
    f_v = _lambdify_vec_xy(f_v_expr, x, y)
    f_u = _lambdify_vec_xy(f_u_expr, x, y)
    s_v = _lambdify_scalar_xy(s_v_expr, x, y)
    f_phi = _lambdify_scalar_xy(f_phi_expr, x, y)
    f_alpha = _lambdify_scalar_xy(f_alpha_expr, x, y)
    f_S = _lambdify_scalar_xy(f_S_expr, x, y)
    f_X = _lambdify_scalar_xy(f_X_expr, x, y)
    D_det_prev = _lambdify_scalar_xy(D_det_prev_expr, x, y)
    mu_alpha = _lambdify_scalar_xyt(mu_alpha_expr, x, y, t) if mu_alpha_expr is not None else None
    mu_alpha_n = _lambdify_scalar_xy(mu_alpha_n_expr, x, y) if mu_alpha_n_expr is not None else None
    mu_alpha_k = _lambdify_scalar_xy(mu_alpha_k_expr, x, y) if mu_alpha_k_expr is not None else None

    return BiofilmOneDomainMMSConvergence(
        t_n=t_n,
        t_k=t_k,
        dt=dt,
        theta=th,
        v=v_t,
        p=p_t,
        vS=vS_t,
        u=u_t,
        phi=phi_t,
        alpha=alpha_t,
        S=S_t,
        X=X_t,
        v_n=v_n,
        p_n=p_n,
        vS_n=vS_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        S_n=S_n,
        X_n=X_n,
        v_k=v_k,
        p_k=p_k,
        vS_k=vS_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        S_k=S_k,
        X_k=X_k,
        f_v=f_v,
        f_u=f_u,
        s_v=s_v,
        f_phi=f_phi,
        f_alpha=f_alpha,
        f_S=f_S,
        f_X=f_X,
        D_det_prev=D_det_prev,
        lambda_alpha_n=float(lambda_alpha_n),
        lambda_alpha_k=float(lambda_alpha_k),
        alpha_cahn_M=float(alpha_cahn_M),
        alpha_cahn_gamma=float(alpha_cahn_gamma),
        alpha_cahn_eps=float(alpha_cahn_eps),
        alpha_cahn_conservative=bool(alpha_cahn_conservative),
        alpha_cahn_mobility=str(alpha_cahn_mobility),
        mu_alpha=mu_alpha,
        mu_alpha_n=mu_alpha_n,
        mu_alpha_k=mu_alpha_k,
        alpha_ch_M=float(alpha_ch_M),
        alpha_ch_gamma=float(alpha_ch_gamma),
        alpha_ch_eps=float(alpha_ch_eps),
        alpha_ch_mobility=str(alpha_ch_mobility),
    )
