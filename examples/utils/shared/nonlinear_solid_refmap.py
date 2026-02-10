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
We provide a compressible Neo-Hookean **Cauchy** stress in spatial form, and its
Gateaux derivative w.r.t. u in direction du:

  σ(u) = (2c/J) (B - a I),   B = F F^T,   a = J^{-2β}.

This matches the implementation historically used in `examples/utils/fpi/poro.py`.
"""

from __future__ import annotations

from pycutfem.ufl.expressions import Constant, Identity, det, dot, grad, inv, trace


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
