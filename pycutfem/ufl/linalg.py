"""
Lightweight linear-algebra helpers for the pycutfem UFL layer.

This module intentionally avoids introducing new expression node types.
All helpers build expression trees from already-supported operations so they
work in all backends (python / jit-numba / jit-cpp).

Currently implemented:
  - Smooth positive/negative part (Macaulay brackets) for scalar expressions.
  - 2×2 symmetric spectral decomposition utilities (eigenvalues and projectors)
    expressed via trace/determinant invariants.

Notes
-----
The 2×2 routines assume the input tensor is (approximately) symmetric.
They are designed for phase-field fracture / damage models (Miehe-type
tension/compression splits) in 2D.
"""

from __future__ import annotations

from pycutfem.ufl.expressions import Constant, Identity, cof, det, inner, trace


def _c(val: float) -> Constant:
    return Constant(float(val))


def sym(A):
    """Symmetric part: 0.5 * (A + Aᵀ)."""
    return _c(0.5) * (A + A.T)


def smooth_abs(x, *, eta: float = 1.0e-12):
    """Smooth absolute value: |x| ≈ sqrt(x² + eta)."""
    return (x * x + _c(float(eta))) ** _c(0.5)


def smooth_pos(x, *, eta: float = 1.0e-12):
    """Smooth positive part: ⟨x⟩₊ ≈ 0.5 * (x + sqrt(x² + eta))."""
    return _c(0.5) * (x + smooth_abs(x, eta=float(eta)))


def smooth_neg(x, *, eta: float = 1.0e-12):
    """Smooth negative part: ⟨x⟩₋ ≈ 0.5 * (x - sqrt(x² + eta)) (≤ 0)."""
    return _c(0.5) * (x - smooth_abs(x, eta=float(eta)))


def smooth_pos_derivative(x, *, eta: float = 1.0e-12):
    """Derivative of `smooth_pos` with respect to x."""
    ax = smooth_abs(x, eta=float(eta))
    # Keep the "function-like" operand on the left so the python backend
    # dispatches to VecOpInfo.__add__ (float + VecOpInfo is not supported).
    return _c(0.5) * ((x / ax) + _c(1.0))


def eigvals_2x2_sym(A, *, disc_reg: float = 1.0e-16):
    """
    Eigenvalues of a 2×2 (approximately) symmetric tensor via invariants.

    For A ∈ R^{2×2}:
        λ₁,₂ = 0.5 * (tr(A) ± sqrt(tr(A)² - 4 det(A))).

    Parameters
    ----------
    disc_reg:
        Small non-negative regularization added under the square root to avoid
        division-by-zero / NaNs in near-repeated eigenvalue states.
    """
    trA = trace(A)
    detA = det(A)
    disc = trA * trA - _c(4.0) * detA
    delta = (disc + _c(float(disc_reg))) ** _c(0.5)
    lam1 = _c(0.5) * (trA + delta)
    lam2 = _c(0.5) * (trA - delta)
    return lam1, lam2, delta


def projectors_2x2_sym(A, *, disc_reg: float = 1.0e-16):
    """
    Spectral projectors for a 2×2 symmetric tensor.

    Returns P1, P2 such that:
      - P1 + P2 = I
      - A = λ1 P1 + λ2 P2

    The implementation uses:
      P1 = (A - λ2 I) / (λ1 - λ2),  P2 = I - P1.
    """
    lam1, lam2, delta = eigvals_2x2_sym(A, disc_reg=float(disc_reg))
    I = Identity(2)
    P1 = (A - lam2 * I) / delta
    P2 = I - P1
    return P1, P2, lam1, lam2, delta


def spectral_positive_part_2x2_sym(
    A,
    *,
    eta_pos: float = 1.0e-12,
    disc_reg: float = 1.0e-16,
):
    """
    Positive/negative spectral split for a 2×2 symmetric tensor.

    A⁺ = Σ ⟨λᵢ⟩₊ Pᵢ,  A⁻ = A - A⁺.
    """
    P1, P2, lam1, lam2, delta = projectors_2x2_sym(A, disc_reg=float(disc_reg))
    lam1p = smooth_pos(lam1, eta=float(eta_pos))
    lam2p = smooth_pos(lam2, eta=float(eta_pos))
    A_plus = lam1p * P1 + lam2p * P2
    A_minus = A - A_plus
    return A_plus, A_minus, (lam1, lam2), (P1, P2), delta


def d_spectral_positive_part_2x2_sym(
    A,
    dA,
    *,
    eta_pos: float = 1.0e-12,
    disc_reg: float = 1.0e-16,
):
    """
    Gateaux derivative of the positive spectral part A⁺ with respect to A.

    Returns D(A⁺)[dA] for 2×2 symmetric A.
    """
    I = Identity(2)

    # --- invariants and eigenvalues ---
    trA = trace(A)
    detA = det(A)
    disc = trA * trA - _c(4.0) * detA
    delta = (disc + _c(float(disc_reg))) ** _c(0.5)
    lam1 = _c(0.5) * (trA + delta)
    lam2 = _c(0.5) * (trA - delta)

    # --- variations ---
    dtrA = trace(dA)
    # δ det(A) = cof(A) : δA
    ddetA = inner(cof(A), dA)
    ddisc = _c(2.0) * trA * dtrA - _c(4.0) * ddetA
    ddelta = ddisc / (_c(2.0) * delta)
    dlam1 = _c(0.5) * (dtrA + ddelta)
    dlam2 = _c(0.5) * (dtrA - ddelta)

    # --- projectors ---
    P1 = (A - lam2 * I) / delta
    P2 = I - P1

    # --- positive parts ---
    lam1p = smooth_pos(lam1, eta=float(eta_pos))
    lam2p = smooth_pos(lam2, eta=float(eta_pos))
    dlam1p = smooth_pos_derivative(lam1, eta=float(eta_pos)) * dlam1
    dlam2p = smooth_pos_derivative(lam2, eta=float(eta_pos)) * dlam2

    # δP1 = (δA - δλ2 I - P1 δdelta) / delta
    dP1 = (dA - dlam2 * I - P1 * ddelta) / delta

    # δA⁺ = δλ1⁺ P1 + δλ2⁺ P2 + (λ1⁺-λ2⁺) δP1
    return dlam1p * P1 + dlam2p * P2 + (lam1p - lam2p) * dP1
