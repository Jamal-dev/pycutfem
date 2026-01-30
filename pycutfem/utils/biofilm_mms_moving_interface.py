"""Moving diffuse-interface (detachment-like) MMS for the one-domain biofilm model.

This MMS matches the "clean moving diffuse interface" sketch in
`examples/biofilms/model/model.tex`:

  - Biofilm indicator: alpha(y,t) = sigma((h(t) - y)/eps),  h(t)=h0 - V_det*t
  - Porosity:          phi = 1 - (1 - phi_b) * alpha  (⇒ phi→1 in fluid, phi→phi_b in biofilm)
  - Fluid velocity:    v = (sin(pi*y) * (1 + a*sin(omega*t)), 0)
  - Pressure:          p = 0
  - Skeleton:          u = 0  (vS=0)
  - Substrate:         S = S0 (constant)

The purpose is to exercise:
  - alpha/phi dependence of rho, mu, beta
  - Brinkman drag (beta v) and the skeleton drag forcing
  - diffusion/penalty terms in alpha/phi
  - detached-mass transport (X) driven by surface detachment
  - a time-dependent moving interface ("detachment-like")

Important notes
---------------
* This MMS is designed for backward Euler (theta=1) usage in the current
  one-domain implementation. For theta<1, several residual blocks are
  theta-averaged while the forcing is not, which would require divisions by
  alpha_k and becomes singular when alpha≈0.
* pycutfem currently cannot reliably assemble `grad(Analytic(...))`, so all
  forcing terms here are provided as pure callables (x,y)->value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _asarray(x):
    return np.asarray(x, dtype=float)


def _sigmoid(z):
    """Numerically stable logistic sigmoid for numpy arrays."""
    z = _asarray(z)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass
class BiofilmMovingInterfaceMMS:
    # Interface parameters
    h0: float = 0.4
    V_det: float = 0.2
    eps: float = 0.05

    # Porosity in biofilm (phi ≈ phi_b where alpha≈1)
    phi_b: float = 0.6

    # Flow parameters
    a: float = 0.2
    omega: float = 2.0 * float(np.pi)

    # Substrate constant
    S0: float = 0.5

    # Model/forcing parameters (match build_biofilm_one_domain_forms defaults)
    rho_f: float = 1.0
    mu_f: float = 1.0e-2
    kappa_inv: float = 10.0

    D_phi: float = 0.1
    gamma_phi: float = 1.0
    D_alpha: float = 0.1
    D_X: float = 0.1

    # Lagged detachment rate: D_det_prev = k_det * sqrt(||eps(v_n)||^2 + eta_n)
    k_det: float = 0.2
    eta_n: float = 1.0e-12

    # Detached biomass scaling (intrinsic solid density)
    rho_s_star: float = 1.0

    # Step state (updated by the time-step driver)
    t_n: float = 0.0
    dt: float = 0.05

    def set_step_time(self, t_n: float, dt: float) -> None:
        self.t_n = float(t_n)
        self.dt = float(dt)

    @property
    def t_k(self) -> float:
        return float(self.t_n + self.dt)

    # ------------------------------------------------------------------
    # Exact fields (x,y,t)
    # ------------------------------------------------------------------
    def h(self, t: float) -> float:
        return float(self.h0 - self.V_det * float(t))

    def alpha(self, x, y, t: float):
        yv = _asarray(y)
        z = (self.h(float(t)) - yv) / float(self.eps)
        return _sigmoid(z)

    def phi(self, x, y, t: float):
        alpha = self.alpha(x, y, t)
        return 1.0 - (1.0 - float(self.phi_b)) * alpha

    def v(self, x, y, t: float):
        yv = _asarray(y)
        amp = 1.0 + float(self.a) * np.sin(float(self.omega) * float(t))
        vx = np.sin(np.pi * yv) * amp
        vy = np.zeros_like(vx)
        return np.stack((vx, vy), axis=-1)

    def p(self, x, y, t: float):
        return np.zeros_like(_asarray(y))

    def u(self, x, y, t: float):
        z = np.zeros_like(_asarray(y))
        return np.stack((z, z), axis=-1)

    def S(self, x, y, t: float):
        return np.full_like(_asarray(y), float(self.S0), dtype=float)

    def X(self, x, y, t: float):
        """Detached biomass concentration (chosen smooth MMS field)."""
        yv = _asarray(y)
        amp = 1.0 + 0.2 * np.sin(float(self.omega) * float(t))
        return 0.1 + 0.05 * np.sin(np.pi * yv) * amp

    # ------------------------------------------------------------------
    # Helpers: alpha derivatives (only y-dependence)
    # ------------------------------------------------------------------
    def _alpha_y_from_alpha(self, alpha):
        eps = float(self.eps)
        return -(alpha * (1.0 - alpha)) / eps

    def _alpha_yy_from_alpha(self, alpha):
        eps2 = float(self.eps) * float(self.eps)
        return (alpha * (1.0 - alpha) * (1.0 - 2.0 * alpha)) / eps2

    def _capacity_from_alpha(self, alpha):
        # With phi = 1 - (1-phi_b) alpha, capacity C=(1-alpha)+alpha*phi = 1 - (1-phi_b)*alpha^2.
        return 1.0 - (1.0 - float(self.phi_b)) * (alpha * alpha)

    def _capacity_y_from_alpha(self, alpha, alpha_y):
        # C_y = -2(1-phi_b) alpha alpha_y
        return -2.0 * (1.0 - float(self.phi_b)) * alpha * alpha_y

    # ------------------------------------------------------------------
    # Discrete-step forcing (x,y) for BE (theta=1) use
    # ------------------------------------------------------------------
    def s_v(self, x, y):
        return np.zeros_like(_asarray(y))

    def D_det_prev(self, x, y):
        """Lagged detachment coefficient at time t_n (used in the alpha-equation)."""
        yv = _asarray(y)
        amp = 1.0 + float(self.a) * np.sin(float(self.omega) * float(self.t_n))
        dvx_dy = np.pi * np.cos(np.pi * yv) * amp
        # For v=(vx(y),0): inner(eps(v),eps(v)) = 0.5*(dvx/dy)^2
        tau2 = 0.5 * (dvx_dy * dvx_dy)
        tau = np.sqrt(tau2 + float(self.eta_n))
        return float(self.k_det) * tau

    def f_v(self, x, y):
        """Momentum RHS f_v at the current step (uses v_k, v_n, and coefficients at t_k)."""
        dt = float(self.dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        yv = _asarray(y)
        t_n = float(self.t_n)
        t_k = float(self.t_k)

        # Exact snapshots
        amp_k = 1.0 + float(self.a) * np.sin(float(self.omega) * t_k)
        amp_n = 1.0 + float(self.a) * np.sin(float(self.omega) * t_n)
        vx_k = np.sin(np.pi * yv) * amp_k
        vx_n = np.sin(np.pi * yv) * amp_n

        # Coefficients at k (phi_mu choice ⇒ mu = mu_f*C)
        alpha_k = self.alpha(x, y, t_k)
        alpha_n = self.alpha(x, y, t_n)
        phi_k = 1.0 - (1.0 - float(self.phi_b)) * alpha_k
        phi_n = 1.0 - (1.0 - float(self.phi_b)) * alpha_n
        C_k = self._capacity_from_alpha(alpha_k)
        C_n = self._capacity_from_alpha(alpha_n)
        rho_k = float(self.rho_f) * C_k
        rho_n = float(self.rho_f) * C_n
        mu_k = float(self.mu_f) * C_k

        # Variable-viscosity divergence term: -∂_y( mu * ∂_y vx )
        dvx_dy_k = np.pi * np.cos(np.pi * yv) * amp_k
        d2vx_dy2_k = -(np.pi**2) * np.sin(np.pi * yv) * amp_k

        alpha_y_k = self._alpha_y_from_alpha(alpha_k)
        C_y_k = self._capacity_y_from_alpha(alpha_k, alpha_y_k)
        mu_y_k = float(self.mu_f) * C_y_k
        visc_x = -(mu_y_k * dvx_dy_k + mu_k * d2vx_dy2_k)

        # Brinkman drag: beta(v-vS) with vS=0
        beta_k = alpha_k * float(self.mu_f) * (phi_k * phi_k) * float(self.kappa_inv)
        drag_x = beta_k * vx_k

        # Conservative-in-time momentum: (rho_k v_k - rho_n v_n)/dt.
        vdot_mom_x = (rho_k * vx_k - rho_n * vx_n) / dt
        fvx = vdot_mom_x + visc_x + drag_x
        fvy = np.zeros_like(fvx)
        return np.stack((fvx, fvy), axis=-1)

    def f_u(self, x, y):
        """Skeleton RHS f_u chosen so that u=0 is an exact solution (vS=0, p=0, sigma(u)=0)."""
        yv = _asarray(y)
        t_k = float(self.t_k)
        amp_k = 1.0 + float(self.a) * np.sin(float(self.omega) * t_k)
        vx_k = np.sin(np.pi * yv) * amp_k

        alpha_k = self.alpha(x, y, t_k)
        phi_k = 1.0 - (1.0 - float(self.phi_b)) * alpha_k
        beta_k = alpha_k * float(self.mu_f) * (phi_k * phi_k) * float(self.kappa_inv)

        # Skeleton drag appears as -β(v-vS) in the weak form, while the forcing is weighted
        # by α. For u=0 (vS=0), we therefore need α f_u = -β v  ⇒  f_u = -(β/α) v.
        fux = -(beta_k / alpha_k) * vx_k
        fuy = np.zeros_like(fux)
        return np.stack((fux, fuy), axis=-1)

    def f_alpha(self, x, y):
        """Indicator RHS f_alpha for the current BE step."""
        dt = float(self.dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        t_n = float(self.t_n)
        t_k = float(self.t_k)
        alpha_k = self.alpha(x, y, t_k)
        alpha_n = self.alpha(x, y, t_n)
        alpha_yy_k = self._alpha_yy_from_alpha(alpha_k)

        delta_k = 4.0 * alpha_k * (1.0 - alpha_k)
        return (alpha_k - alpha_n) / dt - self.D_det_prev(x, y) * delta_k - float(self.D_alpha) * alpha_yy_k

    def f_phi(self, x, y):
        """Porosity RHS f_phi for the current BE step (with vS=0 and Π_b=0)."""
        dt = float(self.dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        t_n = float(self.t_n)
        t_k = float(self.t_k)
        alpha_k = self.alpha(x, y, t_k)
        alpha_n = self.alpha(x, y, t_n)

        phi_k = 1.0 - (1.0 - float(self.phi_b)) * alpha_k
        phi_n = 1.0 - (1.0 - float(self.phi_b)) * alpha_n

        alpha_yy_k = self._alpha_yy_from_alpha(alpha_k)
        phi_yy_k = -(1.0 - float(self.phi_b)) * alpha_yy_k

        penalty = float(self.gamma_phi) * (1.0 - alpha_k) * (phi_k - 1.0)
        return alpha_k * (phi_k - phi_n) / dt - float(self.D_phi) * phi_yy_k + penalty

    def f_S(self, x, y):
        """Substrate RHS f_S for the current BE step (S=S0 constant, Π_b=0)."""
        dt = float(self.dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        t_n = float(self.t_n)
        t_k = float(self.t_k)
        alpha_k = self.alpha(x, y, t_k)
        alpha_n = self.alpha(x, y, t_n)
        C_k = self._capacity_from_alpha(alpha_k)
        C_n = self._capacity_from_alpha(alpha_n)
        return float(self.S0) * (C_k - C_n) / dt

    def f_X(self, x, y):
        """Detached biomass RHS f_X for the current BE step."""
        dt = float(self.dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        yv = _asarray(y)
        t_n = float(self.t_n)
        t_k = float(self.t_k)

        alpha_k = self.alpha(x, y, t_k)
        alpha_n = self.alpha(x, y, t_n)
        phi_k = self.phi(x, y, t_k)
        phi_n = self.phi(x, y, t_n)

        C_k = self._capacity_from_alpha(alpha_k)
        C_n = self._capacity_from_alpha(alpha_n)

        X_k = self.X(x, y, t_k)
        X_n = self.X(x, y, t_n)

        # With X=X(y,t) and v=(v_x(y,t),0), div(CX v)=0 (no x-dependence).
        dCX_dt = (C_k * X_k - C_n * X_n) / dt

        # Diffusion term: -D_X * d^2/dy^2 X_k
        amp_k = 1.0 + 0.2 * np.sin(float(self.omega) * t_k)
        X_yy_k = -0.05 * (np.pi**2) * np.sin(np.pi * yv) * amp_k
        diff = -float(self.D_X) * X_yy_k

        # Detachment-driven source (surface localized via delta(alpha)).
        delta_k = 4.0 * alpha_k * (1.0 - alpha_k)
        R_det_k = float(self.rho_s_star) * (1.0 - phi_k) * self.D_det_prev(x, y) * delta_k

        return dCX_dt + diff - R_det_k
