"""Moving porous-cylinder MMS for the one-domain biofilm model.

This manufactured solution is designed to test *rigid-body-like* motion of a
porous inclusion ("biofilm chunk") moving through a surrounding fluid within a
single-domain diffuse-interface formulation.

Key design choices
------------------
* 2D unit square domain is assumed by the example driver.
* A diffuse "cylinder" (disk in 2D) is represented by a smooth indicator field

    alpha(x,y,t) = sigmoid((R^2 - r^2(x,y,t)) / eps2)

  where r^2 = (x-xc(t))^2 + (y-yc(t))^2 and eps2=2*R*eps.
* The cylinder center (xc(t), yc(t)) moves in time (default: horizontal
  oscillation).
* Skeleton displacement is a *global translation* u(x,y,t)=d(t), so vS is
  spatially uniform. This avoids discontinuities in a CG displacement space
  while still testing "chunk" advection and Brinkman drag.
* Fluid velocity is v = vS + w, where w is a divergence-free tangential vortex
  around the moving center. This activates the Brinkman drag term beta*(v-vS).
* Pressure p=0, substrate S=0 and detached biomass X=0.

Important notes
---------------
* This MMS is intended for backward Euler (theta=1). It provides forcing terms
  for the *discrete* BE step (t_n -> t_k=t_n+dt), in the same spirit as
  `BiofilmMovingInterfaceMMS`.
* The momentum forcing below assumes **constant** viscosity via
  `mu_b_model="mu"` in `build_biofilm_one_domain_forms`. If you switch to
  variable viscosity, you must augment the viscous forcing accordingly.
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
class BiofilmMovingCylinderMMS:
    # Geometry / motion (center of the disk)
    x0: float = 0.5
    y0: float = 0.5
    Ax: float = 0.15
    Ay: float = 0.0
    omega: float = 2.0 * float(np.pi)

    # "Cylinder" (disk) and diffuse thickness
    R: float = 0.18
    eps: float = 0.03  # interface thickness in length units (used via eps2=2*R*eps)

    # Porosity in the biofilm chunk
    phi_b: float = 0.6

    # Flow/vortex parameters
    Omega0: float = 1.0
    sigma: float = 0.25  # vortex localization length

    # Physical/model parameters used by forcing
    rho_f: float = 1.0
    mu_f: float = 1.0e-2
    kappa_inv: float = 10.0
    D_phi: float = 0.0
    gamma_phi: float = 0.0
    D_alpha: float = 0.0
    D_S: float = 0.0
    D_X: float = 0.0

    # Step state (updated by the time-step driver)
    t_n: float = 0.0
    dt: float = 0.02

    def set_step_time(self, t_n: float, dt: float) -> None:
        self.t_n = float(t_n)
        self.dt = float(dt)

    @property
    def t_k(self) -> float:
        return float(self.t_n + self.dt)

    # ------------------------------------------------------------------
    # Motion of the cylinder center
    # ------------------------------------------------------------------
    def xc(self, t: float) -> float:
        return float(self.x0 + self.Ax * np.sin(float(self.omega) * float(t)))

    def yc(self, t: float) -> float:
        return float(self.y0 + self.Ay * np.sin(float(self.omega) * float(t)))

    def dxc_dt(self, t: float) -> float:
        return float(self.Ax * float(self.omega) * np.cos(float(self.omega) * float(t)))

    def dyc_dt(self, t: float) -> float:
        return float(self.Ay * float(self.omega) * np.cos(float(self.omega) * float(t)))

    # ------------------------------------------------------------------
    # Exact fields (x,y,t)
    # ------------------------------------------------------------------
    def alpha(self, x, y, t: float):
        x = _asarray(x)
        y = _asarray(y)
        dx = x - self.xc(t)
        dy = y - self.yc(t)
        r2 = dx * dx + dy * dy

        R2 = float(self.R) * float(self.R)
        eps2 = max(1.0e-12, 2.0 * float(self.R) * float(self.eps))
        z = (R2 - r2) / eps2
        return _sigmoid(z)

    def phi(self, x, y, t: float):
        a = self.alpha(x, y, t)
        return 1.0 - (1.0 - float(self.phi_b)) * a

    def p(self, x, y, t: float):
        return np.zeros_like(_asarray(x))

    def S(self, x, y, t: float):
        return np.zeros_like(_asarray(x))

    def X(self, x, y, t: float):
        return np.zeros_like(_asarray(x))

    def u(self, x, y, t: float):
        # Global translation displacement (rigid-body-like) in CG space.
        dx = self.xc(t) - self.xc(0.0)
        dy = self.yc(t) - self.yc(0.0)
        x = _asarray(x)
        z = np.zeros_like(x)
        return np.stack((z + dx, z + dy), axis=-1)

    def vS(self, t: float):
        return np.array([self.dxc_dt(t), self.dyc_dt(t)], dtype=float)

    def v(self, x, y, t: float):
        """v = vS(t) + w(x,y,t), with w tangential and divergence-free."""
        x = _asarray(x)
        y = _asarray(y)
        dx = x - self.xc(t)
        dy = y - self.yc(t)
        r2 = dx * dx + dy * dy

        sigma2 = max(1.0e-12, float(self.sigma) * float(self.sigma))
        g = float(self.Omega0) * np.exp(-r2 / sigma2)
        wx = -g * dy
        wy = g * dx

        vS = self.vS(t)
        vx = wx + vS[0]
        vy = wy + vS[1]
        return np.stack((vx, vy), axis=-1)

    # ------------------------------------------------------------------
    # Helpers: alpha/phi spatial derivatives (x,y,t)
    # ------------------------------------------------------------------
    def _alpha_derivs(self, x, y, t: float):
        """Return (alpha, ax, ay, lap_alpha) at time t."""
        x = _asarray(x)
        y = _asarray(y)
        dx = x - self.xc(t)
        dy = y - self.yc(t)
        r2 = dx * dx + dy * dy

        R2 = float(self.R) * float(self.R)
        eps2 = max(1.0e-12, 2.0 * float(self.R) * float(self.eps))
        z = (R2 - r2) / eps2
        a = _sigmoid(z)

        a1 = a * (1.0 - a)  # dα/dz
        zx = (-2.0 * dx) / eps2
        zy = (-2.0 * dy) / eps2
        ax = a1 * zx
        ay = a1 * zy

        # Laplacian via α_zz |∇z|^2 + α_z Δz.
        a2 = a1 * (1.0 - 2.0 * a)  # d²α/dz²
        zxx_yy = (-4.0) / eps2
        zx2_zy2 = zx * zx + zy * zy
        lap = a2 * zx2_zy2 + a1 * zxx_yy
        return a, ax, ay, lap

    def _phi_derivs_from_alpha(self, a, ax, ay, lap_a):
        fac = -(1.0 - float(self.phi_b))
        phi = 1.0 + fac * a
        phix = fac * ax
        phiy = fac * ay
        lap_phi = fac * lap_a
        return phi, phix, phiy, lap_phi

    def _capacity_from_alpha(self, a):
        # With phi = 1 - (1-phi_b) alpha, capacity C=(1-alpha)+alpha*phi = 1 - (1-phi_b)*alpha^2.
        return 1.0 - (1.0 - float(self.phi_b)) * (a * a)

    def _capacity_grad_from_alpha(self, a, ax, ay):
        fac = -2.0 * (1.0 - float(self.phi_b))
        Cx = fac * a * ax
        Cy = fac * a * ay
        return Cx, Cy

    # ------------------------------------------------------------------
    # Discrete-step forcing for BE (theta=1)
    # ------------------------------------------------------------------
    def s_v(self, x, y):
        # Designed so that div( C v + B vS ) = 0 for the chosen fields.
        return np.zeros_like(_asarray(x))

    def f_S(self, x, y):
        return np.zeros_like(_asarray(x))

    def f_X(self, x, y):
        return np.zeros_like(_asarray(x))

    def f_alpha(self, x, y):
        dt = float(self.dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        t_n = float(self.t_n)
        t_k = float(self.t_k)

        a_k, ax_k, ay_k, lap_a_k = self._alpha_derivs(x, y, t_k)
        a_n = self.alpha(x, y, t_n)

        vS_k = self.vS(t_k)
        adv = ax_k * vS_k[0] + ay_k * vS_k[1]
        return (a_k - a_n) / dt + adv - float(self.D_alpha) * lap_a_k

    def f_phi(self, x, y):
        dt = float(self.dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        t_n = float(self.t_n)
        t_k = float(self.t_k)

        a_k, ax_k, ay_k, lap_a_k = self._alpha_derivs(x, y, t_k)
        a_n = self.alpha(x, y, t_n)

        phi_k, phix_k, phiy_k, lap_phi_k = self._phi_derivs_from_alpha(a_k, ax_k, ay_k, lap_a_k)
        phi_n = 1.0 - (1.0 - float(self.phi_b)) * a_n

        vS_k = self.vS(t_k)
        adv = phix_k * vS_k[0] + phiy_k * vS_k[1]
        # Match `build_biofilm_one_domain_forms`: sharpen the fluid-region constraint.
        penalty = float(self.gamma_phi) * (1.0 - a_k) ** 16 * (phi_k - 1.0)

        return a_k * (phi_k - phi_n) / dt + a_k * adv - float(self.D_phi) * lap_phi_k + penalty

    def f_u(self, x, y):
        """Skeleton forcing for a rigid-translation u (no elastic stress, p=0, no inertia).

        In the weak form, the skeleton drag contributes -β(v-vS) and forcing contributes -α f_u.
        With β = α μ_f φ^2 κ^{-1}, we avoid dividing by α:

            f_u = - μ_f φ^2 κ^{-1} (v - vS).
        """
        t_k = float(self.t_k)
        v_k = self.v(x, y, t_k)
        vS_k = self.vS(t_k)
        a_k = self.alpha(x, y, t_k)
        phi_k = 1.0 - (1.0 - float(self.phi_b)) * a_k

        rel = v_k - vS_k
        coef = -float(self.mu_f) * (phi_k * phi_k) * float(self.kappa_inv)
        return coef[..., None] * rel

    def f_v(self, x, y):
        """Momentum forcing for BE, assuming mu_b_model='mu' (constant viscosity).

        Matches the theta=1 momentum block in `build_biofilm_one_domain_forms`:

          (rho_k v_k - rho_n v_n)/dt + rho_k (v·∇)v + v div(rho v) - div(2 mu eps(v)) + beta (v-vS) = f_v

        with rho = rho_f * C(alpha,phi), mu = mu_f (constant here),
        beta = alpha * mu_f * phi^2 * kappa_inv.
        """
        dt = float(self.dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        t_n = float(self.t_n)
        t_k = float(self.t_k)

        # Snapshots
        v_k = self.v(x, y, t_k)
        v_n = self.v(x, y, t_n)

        # alpha/phi/C and gradients at k
        a_k, ax_k, ay_k, _lap_a_k = self._alpha_derivs(x, y, t_k)
        phi_k = 1.0 - (1.0 - float(self.phi_b)) * a_k
        C_k = self._capacity_from_alpha(a_k)
        rho_k = float(self.rho_f) * C_k

        a_n = self.alpha(x, y, t_n)
        C_n = self._capacity_from_alpha(a_n)
        rho_n = float(self.rho_f) * C_n

        # Conservative-in-time term
        momdot = (rho_k[..., None] * v_k - rho_n[..., None] * v_n) / dt

        # Vortex derivatives (at k)
        x = _asarray(x)
        y = _asarray(y)
        dx = x - self.xc(t_k)
        dy = y - self.yc(t_k)
        r2 = dx * dx + dy * dy

        sigma2 = max(1.0e-12, float(self.sigma) * float(self.sigma))
        exp1 = np.exp(-r2 / sigma2)
        Omega0 = float(self.Omega0)

        # g = Omega0 * exp(-r2/sigma2)
        g = Omega0 * exp1

        # Gradient of w (divergence-free tangential vortex)
        dwx_dx = 2.0 * Omega0 * dx * dy * exp1 / sigma2
        dwx_dy = (2.0 * Omega0 * dy * dy * exp1 / sigma2) - g
        dwy_dx = (-2.0 * Omega0 * dx * dx * exp1 / sigma2) + g
        dwy_dy = -2.0 * Omega0 * dx * dy * exp1 / sigma2

        # Convective term: (v·∇)v = (grad w) v (since grad vS = 0)
        vx = v_k[..., 0]
        vy = v_k[..., 1]
        conv_x = dwx_dx * vx + dwx_dy * vy
        conv_y = dwy_dx * vx + dwy_dy * vy
        conv = np.stack((conv_x, conv_y), axis=-1)

        # Conservative correction: v * div(rho v), with div(rho v) = rho_f * grad(C)·v (div v = 0)
        Cx_k, Cy_k = self._capacity_grad_from_alpha(a_k, ax_k, ay_k)
        divCv_k = Cx_k * vx + Cy_k * vy
        div_rhov_k = float(self.rho_f) * divCv_k
        corr = div_rhov_k[..., None] * v_k

        # Viscous term for constant mu and div v = 0: -mu Δv (translation has zero laplacian)
        lap_wx = 4.0 * Omega0 * dy * (-r2 + 2.0 * sigma2) * exp1 / (sigma2 * sigma2)
        lap_wy = 4.0 * Omega0 * dx * (r2 - 2.0 * sigma2) * exp1 / (sigma2 * sigma2)
        visc = -float(self.mu_f) * np.stack((lap_wx, lap_wy), axis=-1)

        # Brinkman drag
        beta_k = a_k * float(self.mu_f) * (phi_k * phi_k) * float(self.kappa_inv)
        vS_k = self.vS(t_k)
        drag = beta_k[..., None] * (v_k - vS_k)

        return momdot + rho_k[..., None] * conv + corr + visc + drag
