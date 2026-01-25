"""Nitsche interface terms for fully Eulerian Fluid–Poroelastic Interaction (FPI).

This module implements the Nitsche-based interface contributions from the paper
(`femdiscritization.tex`, eqs. (24) and (28)) for the coupling between:
  - fluid domain on the positive side (+),
  - poroelastic domain on the negative side (−).

The interface normal convention follows the CutFEM utilities:
  - `n = FacetNormal()` is oriented from (−) to (+),
  - the *fluid outward* normal is `nF = -n`.

All integrals are returned as UFL expressions (linear/bilinear forms).
"""

from __future__ import annotations

from dataclasses import dataclass

from pycutfem.ufl.expressions import (
    Constant,
    FacetNormal,
    Identity,
    div,
    dot,
    grad,
    inner,
    outer,
)
from pycutfem.ufl.expressions import Pos, Neg


def _epsilon(v):
    return Constant(0.5) * (grad(v) + grad(v).T)


def _sigma_fluid(v, p, mu):
    I2 = Identity(2)
    return -p * I2 + Constant(2.0) * mu * _epsilon(v)


def _traction_fluid(v, p, mu, n):
    """Fluid traction t = σ(v,p) · n with σ = -p I + 2μ ε(v)."""
    return Constant(2.0) * mu * dot(_epsilon(v), n) - p * n


def _proj_n(v, n):
    """Normal projection: (v·n) n."""
    return dot(v, n) * n


def _proj_t(v, n):
    """Tangential projection: v - (v·n) n."""
    return v - _proj_n(v, n)


@dataclass(frozen=True)
class FPIInterfaceForms:
    residual: object
    jacobian: object
    residual_normal: object
    jacobian_normal: object
    residual_tangential: object
    jacobian_tangential: object


def build_fpi_interface_forms(
    *,
    # unknowns at the current time level (n+1)
    vF_k,
    pF_k,
    vP_k,
    uP_k,
    # previous-step displacement for u-dot (u_n); set uP_n=uP_k for steady tests
    uP_n,
    # Newton increments
    dvF,
    dpF,
    dvP,
    duP,
    # test functions
    dvF_test,
    dpF_test,
    dvP_test,
    duP_test,
    # measures
    dGamma,
    # physical parameters
    mu_f,
    porosity,
    beta_BJ,
    kappa,
    # Nitsche parameters / scalings
    gamma_n,
    gamma_t,
    phi_gamma_F,
    h_gamma,
    zeta,
    dt,
    # manufactured jump data (use zeros for the physical case)
    g_sigma,
    g_sigma_n,
    g_n,
    g_t,
) -> FPIInterfaceForms:
    """Return residual/Jacobian interface forms (normal + tangential Nitsche)."""
    n = FacetNormal()  # (-) -> (+) (poro -> fluid)
    nF = Constant(-1.0) * n  # fluid outward normal

    I2 = Identity(2)
    Pn = outer(nF, nF)
    Pt = I2 - Pn

    # --- kinematics on the interface ---
    u_dot_k = (uP_k - uP_n) / dt

    # --- fluid stress / traction (evaluated on fluid side) ---
    tractionF = _traction_fluid(Pos(vF_k), Pos(pF_k), mu_f, nF)
    tractionF_n = _proj_n(tractionF, nF)

    dtractionF = _traction_fluid(Pos(dvF), Pos(dpF), mu_f, nF)
    dtractionF_n = _proj_n(dtractionF, nF)

    # --- common kinematic mismatch vectors (projected) ---
    kin = Pos(vF_k) - Neg(u_dot_k) - porosity * (Neg(vP_k) - Neg(u_dot_k)) - g_n
    kin_n = _proj_n(kin, nF)

    d_u_dot = duP / dt
    dkin = Pos(dvF) - Neg(d_u_dot) - porosity * (Neg(dvP) - Neg(d_u_dot))
    dkin_n = _proj_n(dkin, nF)

    # --- Eq. (24): normal direction ---
    test_jump_n = Neg(dvP_test) + Neg(duP_test) - Pos(dvF_test)
    R_n = inner(test_jump_n, tractionF_n) * dGamma
    R_n += -inner(Neg(dvP_test), g_sigma_n * nF) * dGamma
    R_n += -inner(Neg(duP_test), _proj_n(g_sigma, nF)) * dGamma
    R_n += -inner(Pos(dpF_test) * nF + zeta * Constant(2.0) * mu_f * dot(_epsilon(Pos(dvF_test)), nF), kin_n) * dGamma
    R_n += (phi_gamma_F * gamma_n / h_gamma) * inner(Pos(dvF_test) - Neg(dvP_test) - Neg(duP_test), kin_n) * dGamma

    J_n = inner(test_jump_n, dtractionF_n) * dGamma
    # g_sigma_n/g_sigma are data => no derivative
    J_n += -inner(Pos(dpF_test) * nF + zeta * Constant(2.0) * mu_f * dot(_epsilon(Pos(dvF_test)), nF), dkin_n) * dGamma
    J_n += (phi_gamma_F * gamma_n / h_gamma) * inner(Pos(dvF_test) - Neg(dvP_test) - Neg(duP_test), dkin_n) * dGamma

    # --- Eq. (28): tangential Nitsche variant for BJ/BJS condition ---
    tractionF_t = _proj_t(tractionF, nF)
    dtractionF_t = _proj_t(dtractionF, nF)

    c_BJ = Pos(vF_k) - Neg(u_dot_k) - beta_BJ * porosity * (Neg(vP_k) - Neg(u_dot_k)) + kappa * tractionF - g_t
    c_t = _proj_t(c_BJ, nF)

    dc_BJ = Pos(dvF) - Neg(d_u_dot) - beta_BJ * porosity * (Neg(dvP) - Neg(d_u_dot)) + kappa * dtractionF
    dc_t = _proj_t(dc_BJ, nF)

    denom_t = kappa * mu_f + gamma_t * h_gamma
    t1 = zeta * (gamma_t * h_gamma / denom_t)
    t2 = mu_f / denom_t

    R_t = inner(Neg(duP_test) - Pos(dvF_test), tractionF_t) * dGamma
    R_t += -inner(Neg(duP_test), _proj_t(g_sigma, nF)) * dGamma
    R_t += t1 * inner(-Constant(2.0) * mu_f * dot(_epsilon(Pos(dvF_test)), nF), c_t) * dGamma
    R_t += t2 * inner(Pos(dvF_test) - Neg(duP_test), c_t) * dGamma

    J_t = inner(Neg(duP_test) - Pos(dvF_test), dtractionF_t) * dGamma
    J_t += t1 * inner(-Constant(2.0) * mu_f * dot(_epsilon(Pos(dvF_test)), nF), dc_t) * dGamma
    J_t += t2 * inner(Pos(dvF_test) - Neg(duP_test), dc_t) * dGamma

    R = R_n + R_t
    J = J_n + J_t
    return FPIInterfaceForms(
        residual=R,
        jacobian=J,
        residual_normal=R_n,
        jacobian_normal=J_n,
        residual_tangential=R_t,
        jacobian_tangential=J_t,
    )
