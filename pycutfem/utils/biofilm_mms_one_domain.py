"""Manufactured-solution utilities for the one-domain biofilm model.

The goal of this module is to provide a *small* MMS that is:
  - compatible with the current pycutfem UFL/compiler stack
  - exactly representable on the default FE spaces used in tests
  - useful for backend-parity / sign / coupling checks

The provided MMS is intentionally affine (or affine-in-time) so that the
residual can be satisfied to machine precision after masking Dirichlet rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BiofilmOneDomainMMS:
    # exact fields (t_n)
    v_n: callable  # (x,y)->(2,)
    p_n: callable  # (x,y)->scalar
    u_n: callable  # (x,y)->(2,)
    phi_n: callable  # (x,y)->scalar
    alpha_n: callable  # (x,y)->scalar
    S_n: callable  # (x,y)->scalar
    # exact fields (t_{n+1})
    v_k: callable
    p_k: callable
    u_k: callable
    phi_k: callable
    alpha_k: callable
    S_k: callable
    # forcing terms for the BE step
    f_v: callable  # (x,y)->(2,)
    f_u: callable  # (x,y)->(2,)
    s_v: callable  # (x,y)->scalar (mass source)
    f_phi: callable  # (x,y)->scalar
    f_alpha: callable  # (x,y)->scalar
    f_S: callable  # (x,y)->scalar
    # component-wise Dirichlet callables at t_{n+1}
    v_x: callable
    v_y: callable
    u_x: callable
    u_y: callable
    phi: callable
    alpha: callable
    S: callable


def build_biofilm_one_domain_mms_affine(
    *,
    dt_val: float,
    # exact affine shear kinematics
    v_amp: float = 0.3,  # v_k = v_amp * [y, x]
    vS_amp: float = 0.1,  # vS_k = vS_amp * [y, x]
    # pressure p(x,y) = p0 + px*x + py*y
    p_lin: tuple[float, float, float] = (0.2, -0.1, 0.1),
    # phi_n(x,y) = phi0 + phix*x + phiy*y,  phi_k = phi_n + dt*phi_t
    phi_affine: tuple[float, float, float] = (0.7, 0.05, 0.03),
    phi_t: float = 0.02,
    # alpha_n(x,y) = a0 + ax*x + ay*y,  alpha_k = alpha_n + dt*alpha_t
    alpha_affine: tuple[float, float, float] = (0.4, 0.02, -0.01),
    alpha_t: float = 0.01,
    # S_n(x,y) = S0 + Sx*x + Sy*y,  S_k = S_n + dt*S_t
    S_affine: tuple[float, float, float] = (0.2, 0.05, 0.02),
    S_t: float = -0.005,
    # physical parameters used in forcing
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
) -> BiofilmOneDomainMMS:
    """
    Build an affine MMS for a single backward-Euler (θ=1) step.

    Notes
    -----
    - The MMS is meant for residual-zero checks (after masking Dirichlet rows),
      not for convergence studies.
    - Detachment is included in the α equation in the same *lagged* form as
      `build_biofilm_one_domain_forms` uses by default (shear from v_n).
    """
    dt = float(dt_val)
    if dt <= 0.0:
        raise ValueError("dt_val must be positive.")

    px, py, p0 = (float(p_lin[0]), float(p_lin[1]), float(p_lin[2]))
    phi0, phix, phiy = (float(phi_affine[0]), float(phi_affine[1]), float(phi_affine[2]))
    a0, ax, ay = (float(alpha_affine[0]), float(alpha_affine[1]), float(alpha_affine[2]))
    S0, Sx, Sy = (float(S_affine[0]), float(S_affine[1]), float(S_affine[2]))

    grad_p = np.array([px, py], dtype=float)
    grad_phi = np.array([phix, phiy], dtype=float)
    grad_alpha = np.array([ax, ay], dtype=float)
    grad_S = np.array([Sx, Sy], dtype=float)

    def _v_shear(amp: float, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return np.stack((float(amp) * y, float(amp) * x), axis=-1)

    def v_n(x, y):
        return _v_shear(0.0, x, y)

    def v_k(x, y):
        return _v_shear(v_amp, x, y)

    def u_n(x, y):
        return _v_shear(0.0, x, y)

    def u_k(x, y):
        return dt * _v_shear(vS_amp, x, y)

    def p_n(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return 0.0 * x + 0.0 * y

    def p_k(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return px * x + py * y + p0

    def phi_n(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return phi0 + phix * x + phiy * y

    def phi_k(x, y):
        return phi_n(x, y) + dt * float(phi_t)

    def alpha_n(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return a0 + ax * x + ay * y

    def alpha_k(x, y):
        return alpha_n(x, y) + dt * float(alpha_t)

    def S_n(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return S0 + Sx * x + Sy * y

    def S_k(x, y):
        return S_n(x, y) + dt * float(S_t)

    # --- coefficients at t_{n+1} (k) --------------------------------
    def _capacity(alpha, phi):
        return (1.0 - alpha) + alpha * phi

    def _grad_capacity(alpha, phi):
        # C = 1 + alpha*(phi-1)  -> grad C = grad(alpha)*(phi-1) + alpha*grad(phi)
        gx = grad_alpha[0] * (phi - 1.0) + alpha * grad_phi[0]
        gy = grad_alpha[1] * (phi - 1.0) + alpha * grad_phi[1]
        return gx, gy

    def _grad_B(alpha, phi):
        # B = alpha*(1-phi) -> grad B = grad(alpha)*(1-phi) - alpha*grad(phi)
        gx = grad_alpha[0] * (1.0 - phi) - alpha * grad_phi[0]
        gy = grad_alpha[1] * (1.0 - phi) - alpha * grad_phi[1]
        return gx, gy

    def _monod(S):
        return float(mu_max) * (S / (S + float(K_S)))

    def _Pi_over_rho_s(S, phi, alpha):
        return (_monod(S) - float(k_d)) * (1.0 - phi) * alpha

    # --- BE forcing terms -------------------------------------------
    def f_v(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        vk = _v_shear(v_amp, x, y)
        vn = _v_shear(0.0, x, y)
        vS = _v_shear(vS_amp, x, y)

        ak = alpha_k(x, y)
        phik = phi_k(x, y)
        an = alpha_n(x, y)
        phin = phi_n(x, y)

        Ck = _capacity(ak, phik)
        Cn = _capacity(an, phin)
        rho = float(rho_f) * Ck
        rho_n = float(rho_f) * Cn
        mu = float(mu_f) * Ck  # phi_mu choice
        beta = ak * float(mu_f) * (phik * phik) * float(kappa_inv)

        # Conservative-in-time momentum: (rho_k v_k - rho_n v_n)/dt.
        momdot = (rho[..., None] * vk - rho_n[..., None] * vn) / dt
        conv = np.stack((float(v_amp) ** 2 * x, float(v_amp) ** 2 * y), axis=-1)

        Cx, Cy = _grad_capacity(ak, phik)
        mu_x = float(mu_f) * Cx
        mu_y = float(mu_f) * Cy

        # For the affine shear v = a*[y,x]: ε(v) is constant [[0,a],[a,0]].
        # -div(2μ ε(v)) = [-2a * ∂y μ, -2a * ∂x μ].
        visc = np.stack((-2.0 * float(v_amp) * mu_y, -2.0 * float(v_amp) * mu_x), axis=-1)

        drag = beta[..., None] * (vk - vS)
        # Conservative convection correction: v * div(rho v) with rho=rho_f*C.
        divCv = Cx * vk[..., 0] + Cy * vk[..., 1]  # div(v)=0 for this affine shear
        div_rhov = float(rho_f) * divCv
        # Pressure gradient: ∇(C p) = C ∇p + p ∇C
        pk = p_k(x, y)
        grad_Cp = Ck[..., None] * grad_p + pk[..., None] * np.stack((Cx, Cy), axis=-1)
        return momdot + rho[..., None] * conv + vk * div_rhov[..., None] + visc + grad_Cp + drag

    def s_v(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        vk = _v_shear(v_amp, x, y)
        vS = _v_shear(vS_amp, x, y)
        ak = alpha_k(x, y)
        phik = phi_k(x, y)

        Cx, Cy = _grad_capacity(ak, phik)
        Bx, By = _grad_B(ak, phik)

        divF = Cx * vk[..., 0] + Cy * vk[..., 1] + Bx * vS[..., 0] + By * vS[..., 1]
        return divF / ak

    def f_u(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        vk = _v_shear(v_amp, x, y)
        vS = _v_shear(vS_amp, x, y)
        pk = p_k(x, y)
        ak = alpha_k(x, y)
        phik = phi_k(x, y)

        beta = ak * float(mu_f) * (phik * phik) * float(kappa_inv)
        # Strong form corresponding to the implemented weak form:
        #   -div(α σ(u)) + ∇(α (1-φ) p) - β(v-vS) = α f_u
        # so
        #   f_u = -(σ(u) ∇α)/α + ∇((1-φ)p) + ((1-φ)p) ∇α/α - (β/α)(v-vS),
        # using div(σ(u))=0 for the affine displacement used here.
        grad_p_term = (1.0 - phik)[..., None] * grad_p - pk[..., None] * grad_phi  # ∇((1-φ)p)
        g = (1.0 - phik) * pk
        extra_p = (g / ak)[..., None] * grad_alpha

        # σ(u_k) is constant for affine u_k = dt*vS_amp*[y,x].
        s_off = 2.0 * float(mu_s) * dt * float(vS_amp)
        sigma = np.array([[0.0, s_off], [s_off, 0.0]], dtype=float)
        sigma_grad_alpha = sigma @ grad_alpha
        extra_el = (1.0 / ak)[..., None] * sigma_grad_alpha

        drag = (beta / ak)[..., None] * (vk - vS)
        return grad_p_term + extra_p - extra_el - drag

    def f_phi(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        vS = _v_shear(vS_amp, x, y)
        phik = phi_k(x, y)
        phin = phi_n(x, y)
        ak = alpha_k(x, y)
        Sk = S_k(x, y)

        Pi = _Pi_over_rho_s(Sk, phik, ak)
        adv = grad_phi[0] * vS[..., 0] + grad_phi[1] * vS[..., 1]  # div(vS)=0
        dphi_dt = (phik - phin) / dt

        # Match the implementation in `build_biofilm_one_domain_forms`: enforce φ≈1 in
        # the free-fluid region with a sharpened weight (1-α)^4 to avoid long-time
        # bleed into the biofilm when D_phi=0.
        w = (1.0 - ak) ** 16
        return ak * (dphi_dt + adv + Pi) + float(gamma_phi) * w * (phik - 1.0)

    def f_alpha(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        vS = _v_shear(vS_amp, x, y)
        ak = alpha_k(x, y)
        an = alpha_n(x, y)
        phik = phi_k(x, y)
        Sk = S_k(x, y)

        mon = _monod(Sk)
        G = float(k_g) * mon * (1.0 - phik)
        adv = grad_alpha[0] * vS[..., 0] + grad_alpha[1] * vS[..., 1]
        dadt = (ak - an) / dt

        # Lagged detachment uses v_n (here zero) -> tau = sqrt(eta_n) constant.
        D_det_prev = float(k_det) * float(np.sqrt(float(eta_n)))
        delta = 4.0 * ak * (1.0 - ak)
        # Implemented residual uses +D_det_prev*delta on the LHS (i.e. -D_det_prev*delta on the RHS).
        return dadt + adv - G * ak * (1.0 - ak) + D_det_prev * delta

    def f_S(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        vk = _v_shear(v_amp, x, y)

        ak = alpha_k(x, y)
        phik = phi_k(x, y)
        Sk = S_k(x, y)

        an = alpha_n(x, y)
        phin = phi_n(x, y)
        Sn = S_n(x, y)

        Ck = _capacity(ak, phik)
        Cn = _capacity(an, phin)

        # time derivative of capacity-weighted substrate
        dCS_dt = (Ck * Sk - Cn * Sn) / dt

        # div(CS v) = CS div(v) + ∇(CS)·v, and div(v)=0 for this affine shear.
        Cx, Cy = _grad_capacity(ak, phik)
        adv = (Sk * Cx + Ck * grad_S[0]) * vk[..., 0] + (Sk * Cy + Ck * grad_S[1]) * vk[..., 1]

        Pi = _Pi_over_rho_s(Sk, phik, ak)
        RS = (1.0 / float(Y)) * Pi
        return dCS_dt + adv + RS

    # Component-wise Dirichlet data at t_{n+1}
    def v_x(x, y):
        return _v_shear(v_amp, x, y)[..., 0]

    def v_y(x, y):
        return _v_shear(v_amp, x, y)[..., 1]

    def u_x(x, y):
        return u_k(x, y)[..., 0]

    def u_y(x, y):
        return u_k(x, y)[..., 1]

    return BiofilmOneDomainMMS(
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
        v_x=v_x,
        v_y=v_y,
        u_x=u_x,
        u_y=u_y,
        phi=phi_k,
        alpha=alpha_k,
        S=S_k,
    )
