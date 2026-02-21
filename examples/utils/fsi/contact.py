"""
Contact mechanics helpers for the *examples* (not core library).

Implements the relaxed unilateral (wall) contact term and its semi-smooth
Newton linearization from:
  Frei–Knoke–Steinbach–Wenske–Wick, ACSE 2025
  `examples/fsi_contact/paper/main.tex`, Problems 5–7.

Key pattern (paper Eq. (20) + Sec. 5):
  C(U;ψ)      = γ_C k ∫_Γ ⟨P_γC(U)⟩₊  (ψ·n)  ds
  C'(U)[δU;ψ] = γ_C k ∫_Γ H(P_γC(U)) P_γC'(U)[δU] (ψ·n) ds

We explicitly build the semi-smooth Jacobian using `heaviside(P)` and the
directional derivative `dP`, rather than differentiating `pos_part(P)`.
"""

from __future__ import annotations

from dataclasses import dataclass

from pycutfem.ufl.expressions import (
    Constant,
    Identity,
    dot,
    grad,
    trace,
    pos_part,
    heaviside,
)


def _c(val: float) -> Constant:
    return Constant(float(val))


# ---------------------------------------------------------------------------
#  Material models used in the paper (St. Venant–Kirchhoff in Eulerian form)
# ---------------------------------------------------------------------------


def green_lagrange_strain(u):
    """E(u) = 1/2 (∇u + ∇uᵀ + ∇uᵀ ∇u)."""
    Gu = grad(u)
    return _c(0.5) * (Gu + Gu.T + dot(Gu.T, Gu))


def d_green_lagrange_strain(u, du):
    """Directional derivative of E(u) in direction du (paper Eq. (581)–(587))."""
    Gu = grad(u)
    Gdu = grad(du)
    return _c(0.5) * (Gdu + Gdu.T + dot(Gdu.T, Gu) + dot(Gu.T, Gdu))


def sigma_s_stvk(u, *, mu_s, lambda_s):
    """Cauchy stress σ_s(u) = 2 μ_s E(u) + λ_s tr(E(u)) I."""
    E = green_lagrange_strain(u)
    I = Identity(2)
    return _c(2.0) * mu_s * E + lambda_s * trace(E) * I


def dsigma_s_stvk(u, du, *, mu_s, lambda_s):
    """Directional derivative σ_s'(u)[du]."""
    dE = d_green_lagrange_strain(u, du)
    I = Identity(2)
    return _c(2.0) * mu_s * dE + lambda_s * trace(dE) * I


def sigma_f_newtonian(v, p, *, rho_f, nu_f):
    """Fluid stress σ_f(v,p) = ρ_f ν_f (∇v + ∇vᵀ) - p I (paper Eq. (2))."""
    I = Identity(2)
    Gv = grad(v)
    return rho_f * nu_f * (Gv + Gv.T) - p * I


def dsigma_f_newtonian(dv, dp, *, rho_f, nu_f):
    """Directional derivative σ_f'(v,p)[dv,dp] (linear in (v,p))."""
    I = Identity(2)
    Gdv = grad(dv)
    return rho_f * nu_f * (Gdv + Gdv.T) - dp * I


# ---------------------------------------------------------------------------
#  Relaxed unilateral contact: P_γC(U), residual and semi-smooth Jacobian
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelaxedWallContact:
    """
    Convenience container for building relaxed unilateral contact terms.

    Parameters
    ----------
    gamma_C:
        Contact penalty γ_C (>0).
    gap_eps:
        Relaxed gap g_ε (scalar expression on Γ). For a bottom wall at y=0 this
        is typically `y - epsilon`.
    n_s:
        Solid outward normal on the interface Γ (vector expression).
    n_f:
        Fluid outward normal on Γ (vector expression). For a 2-phase cut with
        n_s = ∇φ/‖∇φ‖ pointing from solid (φ<0) to fluid (φ>0), one typically has
        n_f = -n_s.
    penalty_nitsche:
        Scalar penalty term multiplying (v_f - v_s) in the unified traction jump
        (e.g. ρ_f ν_f γ_N / h).
    """

    gamma_C: object
    gap_eps: object
    n_s: object
    n_f: object
    penalty_nitsche: object

    def P_gammaC(self, *, u, u_prev, v_f, p_f, v_s, sigma_s, sigma_f):
        """
        Paper Eq. (21) (with user-supplied stresses and normals).

        P = (u-u_prev)·n_s - g_ε - γ_C^{-1} n_sᵀ( (σ_s-σ_f) n_f + penalty*(v_f-v_s) ).
        """
        inv_gamma_C = _c(1.0) / self.gamma_C
        jump_n = dot(self.n_s, dot((sigma_s - sigma_f), self.n_f)) + self.penalty_nitsche * dot((v_f - v_s), self.n_s)
        return dot((u - u_prev), self.n_s) - self.gap_eps - inv_gamma_C * jump_n

    def dP_gammaC(
        self,
        *,
        du,
        dv_f,
        dp_f,
        dv_s,
        dsigma_s,
        dsigma_f,
    ):
        """Directional derivative P'(U)[δU] consistent with the simplified Newton in the paper."""
        inv_gamma_C = _c(1.0) / self.gamma_C
        djump_n = dot(self.n_s, dot((dsigma_s - dsigma_f), self.n_f)) + self.penalty_nitsche * dot((dv_f - dv_s), self.n_s)
        return dot(du, self.n_s) - inv_gamma_C * djump_n

    def residual_term(self, *, P, k, test_v_s):
        """Contact residual contribution: γ_C k ⟨P⟩₊ (test_v_s·n_s)."""
        return self.gamma_C * k * pos_part(P) * dot(test_v_s, self.n_s)

    def jacobian_term(self, *, P, dP, k, test_v_s):
        """Semi-smooth Newton contact Jacobian: γ_C k H(P) dP (test_v_s·n_s)."""
        return self.gamma_C * k * heaviside(P) * dP * dot(test_v_s, self.n_s)
