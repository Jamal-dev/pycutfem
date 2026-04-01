"""Shared utilities for fully Eulerian (reference-map) nonlinear solids.

This module collects kinematics + constitutive helpers for *large deformation*
solids written on a fixed Eulerian grid using the **reference-map** displacement
u(x,t), i.e. the inverse motion X(x,t) = x - u(x,t).

Kinematics (Eulerian reference map)
----------------------------------
Let u(x) be the Eulerian displacement-like unknown. Define

  F(u) = (I - ∇u)^{-1},   J(u) = det(F),

where F is the Eulerian deformation gradient (push-forward from reference to
spatial frame expressed using the reference-map variable).

Important: this is *not* the standard Lagrangian definition F = I + ∇_X u.

Permeability / tensor push-forward
----------------------------------
If K^{-1} is the inverse permeability in the *reference* configuration, the
corresponding **spatial** inverse permeability for the Eulerian reference-map
formulation is (see existing FPI implementation):

  k^{-1}(u) = J F^{-T} K^{-1} F^{-1}.

Using the Eulerian identity F^{-1} = I - ∇u, this becomes

  k^{-1}(u) = J (I - ∇u)^T K^{-1} (I - ∇u).

This difference (reference vs spatial tensors) is easy to miss; callers must
be explicit about which frame their permeability tensor lives in.

Constitutive model
------------------
We provide two compressible Neo-Hookean **Cauchy** stresses in spatial form:

  1. Historical ref-map/FPI law:
       σ(u) = (2c/J) (B - a I),   B = F F^T,   a = J^{-2β}

  2. Seboldt-consistent fully Eulerian law:
       σ(u) = (μ/J) (B - I) + λ (J - 1) I

Both are expressed directly in the spatial/Eulerian frame using the reference-
map kinematics F=(I-∇u)^{-1}. The second form is the one-domain Eulerian
translation of the compressible wall model used in Seboldt Example 2.
"""

from __future__ import annotations

from pycutfem.ufl.expressions import Constant, Identity, det, dot, grad, inner, inv, trace

from pycutfem.ufl.linalg import (
    d_spectral_log_2x2_sym,
    d_spectral_positive_part_2x2_sym,
    spectral_log_2x2_sym,
    spectral_positive_part_2x2_sym,
    smooth_pos,
    smooth_pos_derivative,
)


def eulerian_F(u, *, dim: int = 2):
    """Eulerian deformation gradient F = (I - ∇u)^{-1}."""
    I = Identity(int(dim))
    return inv(I - grad(u))


def deulerian_F(u, du, *, dim: int = 2):
    """Gateaux derivative δF for F=(I-∇u)^{-1}: δF = F (∇du) F."""
    F = eulerian_F(u, dim=dim)
    return dot(F, dot(grad(du), F))


def eulerian_k_inv(u, K_inv, *, dim: int = 2):
    """Spatial inverse permeability k^{-1} = J F^{-T} K^{-1} F^{-1}."""
    F = eulerian_F(u, dim=dim)
    J = det(F)
    F_inv = Identity(int(dim)) - grad(u)  # Eulerian identity: F^{-1} = I - ∇u.
    return J * dot(F_inv.T, dot(K_inv, F_inv))


def deulerian_k_inv(u, du, K_inv, *, dim: int = 2):
    """Gateaux derivative of `eulerian_k_inv` w.r.t. u in direction du."""
    F = eulerian_F(u, dim=dim)
    dF = deulerian_F(u, du, dim=dim)

    J = det(F)
    F_inv = Identity(int(dim)) - grad(u)
    # Eulerian identity: δJ = J tr(F^{-1} δF).
    dJ = J * trace(dot(F_inv, dF))

    # Eulerian identity: δ(F^{-1}) = -∇(du).
    dF_inv = -grad(du)
    base = dot(F_inv.T, dot(K_inv, F_inv))
    return dJ * base + J * dot(dF_inv.T, dot(K_inv, F_inv)) + J * dot(F_inv.T, dot(K_inv, dF_inv))


def sigma_neo_hookean(u, c, beta, *, dim: int = 2):
    """Compressible Neo-Hookean Cauchy stress (spatial) for Eulerian ref-map."""
    F = eulerian_F(u, dim=dim)
    J = det(F)
    I = Identity(int(dim))

    a = J ** (-Constant(2.0) * beta)
    B = dot(F, F.T)  # left Cauchy-Green
    return (Constant(2.0) * c / J) * (B - a * I)


def dsigma_neo_hookean(u, du, c, beta, *, dim: int = 2):
    """Gateaux derivative of `sigma_neo_hookean` w.r.t. u in direction du."""
    F = eulerian_F(u, dim=dim)
    dF = deulerian_F(u, du, dim=dim)

    J = det(F)
    F_inv = Identity(int(dim)) - grad(u)
    dJ = J * trace(dot(F_inv, dF))

    I = Identity(int(dim))
    a = J ** (-Constant(2.0) * beta)
    da = -(Constant(2.0) * beta) * a * (dJ / J)

    B = dot(F, F.T)
    dB = dot(dF, F.T) + dot(F, dF.T)

    # σ = 2c/J (B - a I)
    return Constant(2.0) * c * (-(dJ / (J * J)) * (B - a * I) + (Constant(1.0) / J) * (dB - da * I))


def sigma_neo_hookean_seboldt(u, mu_s, lambda_s, *, dim: int = 2):
    """
    Seboldt-consistent compressible Neo-Hookean Cauchy stress in Eulerian form.

    The stress is written directly in spatial variables from the reference-map
    kinematics:

      σ(u) = (μ/J) (B - I) + λ (J - 1) I,

    where F=(I-∇u)^{-1}, J=det(F), and B=F Fᵀ.
    """
    F = eulerian_F(u, dim=dim)
    J = det(F)
    I = Identity(int(dim))
    B = dot(F, F.T)
    return (mu_s / J) * (B - I) + lambda_s * (J - Constant(1.0)) * I


def dsigma_neo_hookean_seboldt(u, du, mu_s, lambda_s, *, dim: int = 2):
    """Gateaux derivative of `sigma_neo_hookean_seboldt` w.r.t. u in direction du."""
    F = eulerian_F(u, dim=dim)
    dF = deulerian_F(u, du, dim=dim)

    J = det(F)
    F_inv = Identity(int(dim)) - grad(u)
    dJ = J * trace(dot(F_inv, dF))

    I = Identity(int(dim))
    B = dot(F, F.T)
    dB = dot(dF, F.T) + dot(F, dF.T)

    return mu_s * (-(dJ / (J * J)) * (B - I) + (Constant(1.0) / J) * dB) + lambda_s * dJ * I


def svk_green_lagrange(u, *, dim: int = 2):
    """Green-Lagrange strain E = 0.5 (C - I) with C = FᵀF for Eulerian ref-map F."""
    F = eulerian_F(u, dim=dim)
    C = dot(F.T, F)
    I = Identity(int(dim))
    return Constant(0.5) * (C - I)


def dsvk_green_lagrange(u, du, *, dim: int = 2):
    """Gateaux derivative of `svk_green_lagrange` w.r.t. u in direction du."""
    F = eulerian_F(u, dim=dim)
    dF = deulerian_F(u, du, dim=dim)
    dC = dot(dF.T, F) + dot(F.T, dF)
    return Constant(0.5) * dC


def sigma_svk(u, mu_s, lambda_s, *, dim: int = 2):
    """
    Saint-Venant–Kirchhoff hyperelasticity (Cauchy stress) for Eulerian ref-map.

    Uses the Green-Lagrange strain E = 0.5(C-I), C=FᵀF, with energy
        ψ(E) = μ ||E||² + (λ/2) (tr E)²,
    2nd Piola S = ∂ψ/∂E = 2μ E + λ tr(E) I, and σ = (1/J) F S Fᵀ.
    """
    F = eulerian_F(u, dim=dim)
    J = det(F)
    E = svk_green_lagrange(u, dim=dim)
    I = Identity(int(dim))
    trE = trace(E)
    S = Constant(2.0) * mu_s * E + lambda_s * trE * I
    return (Constant(1.0) / J) * dot(F, dot(S, F.T))


def dsigma_svk(u, du, mu_s, lambda_s, *, dim: int = 2):
    """Gateaux derivative of `sigma_svk` w.r.t. u in direction du."""
    F = eulerian_F(u, dim=dim)
    dF = deulerian_F(u, du, dim=dim)
    J = det(F)
    F_inv = Identity(int(dim)) - grad(u)
    dJ = J * trace(dot(F_inv, dF))

    E = svk_green_lagrange(u, dim=dim)
    dE = dsvk_green_lagrange(u, du, dim=dim)
    I = Identity(int(dim))
    trE = trace(E)
    dtrE = trace(dE)
    S = Constant(2.0) * mu_s * E + lambda_s * trE * I
    dS = Constant(2.0) * mu_s * dE + lambda_s * dtrE * I

    base = dot(F, dot(S, F.T))
    dbase = dot(dF, dot(S, F.T)) + dot(F, dot(dS, F.T)) + dot(F, dot(S, dF.T))
    return -(dJ / (J * J)) * base + (Constant(1.0) / J) * dbase


def svk_tensile_energy_miehe(
    u,
    mu_s,
    lambda_s,
    *,
    dim: int = 2,
    eta_pos: float = 1.0e-12,
    disc_reg: float = 1.0e-16,
):
    """
    Miehe-style tensile energy density based on Green-Lagrange strain E (2D).

      ψ⁺(u) = μ ||E⁺||² + (λ/2) ⟨tr E⟩₊²,
    where E = 0.5(C-I) and E⁺ uses the positive principal strains.
    """
    if int(dim) != 2:
        raise ValueError("svk_tensile_energy_miehe is currently only implemented for dim=2.")
    E = svk_green_lagrange(u, dim=dim)
    E_plus, _, _, _, _ = spectral_positive_part_2x2_sym(E, eta_pos=float(eta_pos), disc_reg=float(disc_reg))
    tr_pos = smooth_pos(trace(E), eta=float(eta_pos))
    return mu_s * inner(E_plus, E_plus) + Constant(0.5) * lambda_s * (tr_pos * tr_pos)


def sigma_svk_miehe_split(
    u,
    mu_s,
    lambda_s,
    *,
    dim: int = 2,
    eta_pos: float = 1.0e-12,
    disc_reg: float = 1.0e-16,
):
    """Return (sigma_plus, sigma_minus) for SVK elasticity using Miehe split (2D)."""
    if int(dim) != 2:
        raise ValueError("sigma_svk_miehe_split is currently only implemented for dim=2.")
    F = eulerian_F(u, dim=dim)
    J = det(F)
    I = Identity(int(dim))

    E = svk_green_lagrange(u, dim=dim)
    E_plus, _, _, _, _ = spectral_positive_part_2x2_sym(E, eta_pos=float(eta_pos), disc_reg=float(disc_reg))
    trE = trace(E)
    tr_pos = smooth_pos(trE, eta=float(eta_pos))

    S_total = Constant(2.0) * mu_s * E + lambda_s * trE * I
    S_plus = Constant(2.0) * mu_s * E_plus + lambda_s * tr_pos * I
    S_minus = S_total - S_plus

    sigma_plus = (Constant(1.0) / J) * dot(F, dot(S_plus, F.T))
    sigma_minus = (Constant(1.0) / J) * dot(F, dot(S_minus, F.T))
    return sigma_plus, sigma_minus


def dsigma_svk_miehe_split(
    u,
    du,
    mu_s,
    lambda_s,
    *,
    dim: int = 2,
    eta_pos: float = 1.0e-12,
    disc_reg: float = 1.0e-16,
):
    """Return (d_sigma_plus, d_sigma_minus) for SVK Miehe split (2D)."""
    if int(dim) != 2:
        raise ValueError("dsigma_svk_miehe_split is currently only implemented for dim=2.")
    F = eulerian_F(u, dim=dim)
    dF = deulerian_F(u, du, dim=dim)
    J = det(F)
    F_inv = Identity(int(dim)) - grad(u)
    dJ = J * trace(dot(F_inv, dF))

    I = Identity(int(dim))
    E = svk_green_lagrange(u, dim=dim)
    dE = dsvk_green_lagrange(u, du, dim=dim)

    E_plus, _, _, _, _ = spectral_positive_part_2x2_sym(E, eta_pos=float(eta_pos), disc_reg=float(disc_reg))
    dE_plus = d_spectral_positive_part_2x2_sym(E, dE, eta_pos=float(eta_pos), disc_reg=float(disc_reg))

    trE = trace(E)
    dtrE = trace(dE)
    tr_pos = smooth_pos(trE, eta=float(eta_pos))
    dtr_pos = smooth_pos_derivative(trE, eta=float(eta_pos)) * dtrE

    S_total = Constant(2.0) * mu_s * E + lambda_s * trE * I
    dS_total = Constant(2.0) * mu_s * dE + lambda_s * dtrE * I
    S_plus = Constant(2.0) * mu_s * E_plus + lambda_s * tr_pos * I
    dS_plus = Constant(2.0) * mu_s * dE_plus + lambda_s * dtr_pos * I
    S_minus = S_total - S_plus
    dS_minus = dS_total - dS_plus

    base_plus = dot(F, dot(S_plus, F.T))
    dbase_plus = dot(dF, dot(S_plus, F.T)) + dot(F, dot(dS_plus, F.T)) + dot(F, dot(S_plus, dF.T))
    base_minus = dot(F, dot(S_minus, F.T))
    dbase_minus = dot(dF, dot(S_minus, F.T)) + dot(F, dot(dS_minus, F.T)) + dot(F, dot(S_minus, dF.T))

    d_sigma_plus = -(dJ / (J * J)) * base_plus + (Constant(1.0) / J) * dbase_plus
    d_sigma_minus = -(dJ / (J * J)) * base_minus + (Constant(1.0) / J) * dbase_minus
    return d_sigma_plus, d_sigma_minus


def hencky_logV(u, *, dim: int = 2, log_eps: float = 1.0e-16, disc_reg: float = 1.0e-16):
    """
    Hencky (logarithmic) strain tensor in the spatial frame for the Eulerian ref-map.

    Uses the left stretch V via B = F Fᵀ:
        log(V) = 0.5 * log(B).
    """
    F = eulerian_F(u, dim=dim)
    B = dot(F, F.T)
    return Constant(0.5) * spectral_log_2x2_sym(B, log_eps=float(log_eps), disc_reg=float(disc_reg))


def dhencky_logV(u, du, *, dim: int = 2, log_eps: float = 1.0e-16, disc_reg: float = 1.0e-16):
    """Gateaux derivative of `hencky_logV` w.r.t. u in direction du."""
    F = eulerian_F(u, dim=dim)
    dF = deulerian_F(u, du, dim=dim)
    B = dot(F, F.T)
    dB = dot(dF, F.T) + dot(F, dF.T)
    return Constant(0.5) * d_spectral_log_2x2_sym(B, dB, log_eps=float(log_eps), disc_reg=float(disc_reg))


def sigma_hencky(u, mu_s, lambda_s, *, dim: int = 2, log_eps: float = 1.0e-16, disc_reg: float = 1.0e-16):
    """
    Quadratic Hencky hyperelasticity (Cauchy stress) in Eulerian ref-map form.

    Energy density in terms of E = log(V) (spatial Hencky strain):
        ψ(E) = μ ||E||² + (λ/2) (tr E)²,
    with Kirchhoff stress τ = ∂ψ/∂E = 2μ E + λ tr(E) I, and σ = τ / J.
    """
    F = eulerian_F(u, dim=dim)
    J = det(F)
    I = Identity(int(dim))
    E = hencky_logV(u, dim=dim, log_eps=float(log_eps), disc_reg=float(disc_reg))
    trE = trace(E)
    tau = Constant(2.0) * mu_s * E + lambda_s * trE * I
    return tau / J


def dsigma_hencky(
    u,
    du,
    mu_s,
    lambda_s,
    *,
    dim: int = 2,
    log_eps: float = 1.0e-16,
    disc_reg: float = 1.0e-16,
):
    """Gateaux derivative of `sigma_hencky` w.r.t. u in direction du."""
    F = eulerian_F(u, dim=dim)
    dF = deulerian_F(u, du, dim=dim)
    J = det(F)
    F_inv = Identity(int(dim)) - grad(u)
    dJ = J * trace(dot(F_inv, dF))

    I = Identity(int(dim))
    E = hencky_logV(u, dim=dim, log_eps=float(log_eps), disc_reg=float(disc_reg))
    dE = dhencky_logV(u, du, dim=dim, log_eps=float(log_eps), disc_reg=float(disc_reg))
    trE = trace(E)
    dtrE = trace(dE)
    tau = Constant(2.0) * mu_s * E + lambda_s * trE * I
    dtau = Constant(2.0) * mu_s * dE + lambda_s * dtrE * I
    return -(dJ / (J * J)) * tau + (Constant(1.0) / J) * dtau


def hencky_tensile_energy_miehe(
    u,
    mu_s,
    lambda_s,
    *,
    dim: int = 2,
    eta_pos: float = 1.0e-12,
    log_eps: float = 1.0e-16,
    disc_reg: float = 1.0e-16,
):
    """
    Miehe-style tensile energy density based on the Hencky strain E=log(V) (2D).

    ψ⁺(u) = μ ||E⁺||² + (λ/2) ⟨tr E⟩₊²,
    where E⁺ is built from the positive principal Hencky strains.
    """
    if int(dim) != 2:
        raise ValueError("hencky_tensile_energy_miehe is currently only implemented for dim=2.")
    E = hencky_logV(u, dim=dim, log_eps=float(log_eps), disc_reg=float(disc_reg))
    E_plus, _, _, _, _ = spectral_positive_part_2x2_sym(E, eta_pos=float(eta_pos), disc_reg=float(disc_reg))
    tr_pos = smooth_pos(trace(E), eta=float(eta_pos))
    return mu_s * inner(E_plus, E_plus) + Constant(0.5) * lambda_s * (tr_pos * tr_pos)


def sigma_hencky_miehe_split(
    u,
    mu_s,
    lambda_s,
    *,
    dim: int = 2,
    eta_pos: float = 1.0e-12,
    log_eps: float = 1.0e-16,
    disc_reg: float = 1.0e-16,
):
    """Return (sigma_plus, sigma_minus) for Hencky elasticity using Miehe split (2D)."""
    if int(dim) != 2:
        raise ValueError("sigma_hencky_miehe_split is currently only implemented for dim=2.")
    F = eulerian_F(u, dim=dim)
    J = det(F)
    I = Identity(int(dim))

    E = hencky_logV(u, dim=dim, log_eps=float(log_eps), disc_reg=float(disc_reg))
    E_plus, _, _, _, _ = spectral_positive_part_2x2_sym(E, eta_pos=float(eta_pos), disc_reg=float(disc_reg))
    trE = trace(E)
    tr_pos = smooth_pos(trE, eta=float(eta_pos))

    tau_total = Constant(2.0) * mu_s * E + lambda_s * trE * I
    tau_plus = Constant(2.0) * mu_s * E_plus + lambda_s * tr_pos * I
    tau_minus = tau_total - tau_plus
    return tau_plus / J, tau_minus / J


def dsigma_hencky_miehe_split(
    u,
    du,
    mu_s,
    lambda_s,
    *,
    dim: int = 2,
    eta_pos: float = 1.0e-12,
    log_eps: float = 1.0e-16,
    disc_reg: float = 1.0e-16,
):
    """Return (d_sigma_plus, d_sigma_minus) for Hencky Miehe split (2D)."""
    if int(dim) != 2:
        raise ValueError("dsigma_hencky_miehe_split is currently only implemented for dim=2.")
    F = eulerian_F(u, dim=dim)
    dF = deulerian_F(u, du, dim=dim)
    J = det(F)
    F_inv = Identity(int(dim)) - grad(u)
    dJ = J * trace(dot(F_inv, dF))

    I = Identity(int(dim))
    E = hencky_logV(u, dim=dim, log_eps=float(log_eps), disc_reg=float(disc_reg))
    dE = dhencky_logV(u, du, dim=dim, log_eps=float(log_eps), disc_reg=float(disc_reg))

    E_plus, _, _, _, _ = spectral_positive_part_2x2_sym(E, eta_pos=float(eta_pos), disc_reg=float(disc_reg))
    dE_plus = d_spectral_positive_part_2x2_sym(E, dE, eta_pos=float(eta_pos), disc_reg=float(disc_reg))

    trE = trace(E)
    dtrE = trace(dE)
    tr_pos = smooth_pos(trE, eta=float(eta_pos))
    dtr_pos = smooth_pos_derivative(trE, eta=float(eta_pos)) * dtrE

    tau_total = Constant(2.0) * mu_s * E + lambda_s * trE * I
    dtau_total = Constant(2.0) * mu_s * dE + lambda_s * dtrE * I
    tau_plus = Constant(2.0) * mu_s * E_plus + lambda_s * tr_pos * I
    dtau_plus = Constant(2.0) * mu_s * dE_plus + lambda_s * dtr_pos * I

    tau_minus = tau_total - tau_plus
    dtau_minus = dtau_total - dtau_plus

    d_sigma_plus = -(dJ / (J * J)) * tau_plus + (Constant(1.0) / J) * dtau_plus
    d_sigma_minus = -(dJ / (J * J)) * tau_minus + (Constant(1.0) / J) * dtau_minus
    return d_sigma_plus, d_sigma_minus
