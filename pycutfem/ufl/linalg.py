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

from dataclasses import dataclass

from pycutfem.ufl.expressions import Constant, Identity, cof, det, dot, inner, log as ufl_log, trace


def _c(val: float) -> Constant:
    return Constant(float(val))


def sym(A):
    """Symmetric part: 0.5 * (A + Aᵀ)."""
    return _c(0.5) * (A + A.T)


@dataclass(frozen=True)
class PairSpaceCholeskyResult:
    """Result of applying a 3-pair SPD Cholesky resistance in UFL form."""

    coefficients: tuple
    conjugates: tuple
    dissipation_density: object


def _normalize_pair_space_lower_entries(lower_entries):
    try:
        n_outer = len(lower_entries)
    except TypeError as exc:
        raise ValueError("lower_entries must contain six lower entries or a 3x3 lower-triangular factor.") from exc

    if n_outer == 6:
        return tuple(lower_entries)

    if n_outer == 3:
        try:
            row0, row1, row2 = lower_entries
            if len(row0) == 3 and len(row1) == 3 and len(row2) == 3:
                return row0[0], row1[0], row1[1], row2[0], row2[1], row2[2]
        except TypeError as exc:
            raise ValueError("3-entry lower_entries must be a 3x3 lower-triangular factor.") from exc

    raise ValueError("lower_entries must contain six lower entries or a 3x3 lower-triangular factor.")


def pair_space_cholesky_coefficients(weights, lower_entries):
    """Return the six upper-triangle coefficients of ``diag(w) L L.T diag(w)``.

    The helper is intentionally unrolled and expression-only.  Callers that
    need stable C++ kernels should pass named Constant objects for numerical
    Cholesky entries and weights that are already UFL expressions.
    """

    w0, w1, w2 = weights
    l00, l10, l11, l20, l21, l22 = _normalize_pair_space_lower_entries(lower_entries)

    h00 = w0 * l00
    h10 = w1 * l10
    h11 = w1 * l11
    h20 = w2 * l20
    h21 = w2 * l21
    h22 = w2 * l22

    c00 = h00 * h00
    c01 = h00 * h10
    c02 = h00 * h20
    c11 = h10 * h10 + h11 * h11
    c12 = h10 * h20 + h11 * h21
    c22 = h20 * h20 + h21 * h21 + h22 * h22
    return c00, c01, c02, c11, c12, c22


def apply_pair_space_cholesky(relative_vectors, weights, lower_entries) -> PairSpaceCholeskyResult:
    """Apply a full SPD resistance closure to three pair-relative vectors.

    For pair vectors ``r = (r0, r1, r2)`` this builds ``C = D L L.T D`` and
    returns ``Y = C r`` plus ``r.T C r``.  No dense Python loop is involved in
    the expression tree; generated backends receive only scalar algebra and
    vector dot products.
    """

    r0, r1, r2 = relative_vectors
    c00, c01, c02, c11, c12, c22 = pair_space_cholesky_coefficients(weights, lower_entries)

    y0 = c00 * r0 + c01 * r1 + c02 * r2
    y1 = c01 * r0 + c11 * r1 + c12 * r2
    y2 = c02 * r0 + c12 * r1 + c22 * r2

    dissipation_density = (
        c00 * dot(r0, r0)
        + c01 * dot(r0, r1)
        + c01 * dot(r0, r1)
        + c02 * dot(r0, r2)
        + c02 * dot(r0, r2)
        + c11 * dot(r1, r1)
        + c12 * dot(r1, r2)
        + c12 * dot(r1, r2)
        + c22 * dot(r2, r2)
    )

    return PairSpaceCholeskyResult(
        coefficients=(c00, c01, c02, c11, c12, c22),
        conjugates=(y0, y1, y2),
        dissipation_density=dissipation_density,
    )


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


def spectral_log_2x2_sym(
    A,
    *,
    log_eps: float = 1.0e-16,
    disc_reg: float = 1.0e-16,
):
    """
    Spectral matrix logarithm for a 2×2 (approximately) symmetric tensor.

    For SPD A with eigenpairs (λᵢ, Pᵢ), returns:
        log(A) = Σ log(λᵢ) Pᵢ.

    Parameters
    ----------
    log_eps:
        Small positive shift added to eigenvalues before taking log to avoid
        log(0) when eigenvalues become very small in highly deformed states.
    disc_reg:
        Discriminant regularization used in the spectral decomposition.
    """
    P1, P2, lam1, lam2, _ = projectors_2x2_sym(A, disc_reg=float(disc_reg))
    f1 = ufl_log(lam1 + _c(float(log_eps)))
    f2 = ufl_log(lam2 + _c(float(log_eps)))
    return f1 * P1 + f2 * P2


def d_spectral_log_2x2_sym(
    A,
    dA,
    *,
    log_eps: float = 1.0e-16,
    disc_reg: float = 1.0e-16,
):
    """
    Gateaux derivative of the spectral logarithm log(A) for 2×2 symmetric A.

    Uses the standard spectral-function derivative:

      Df(A)[H] = Σ f'(λᵢ) Pᵢ H Pᵢ
               + Σ_{i≠j} (f(λᵢ)-f(λⱼ))/(λᵢ-λⱼ) Pᵢ H Pⱼ,

    with f(λ)=log(λ+log_eps). The discriminant regularization is kept consistent
    with `projectors_2x2_sym` for backend-robustness near repeated eigenvalues.
    """
    I = Identity(2)

    trA = trace(A)
    detA = det(A)
    disc = trA * trA - _c(4.0) * detA
    delta = (disc + _c(float(disc_reg))) ** _c(0.5)
    lam1 = _c(0.5) * (trA + delta)
    lam2 = _c(0.5) * (trA - delta)

    P1 = (A - lam2 * I) / delta
    P2 = I - P1

    # f(λ) = log(λ + eps),  f'(λ) = 1/(λ + eps)
    eps_c = _c(float(log_eps))
    f1 = ufl_log(lam1 + eps_c)
    f2 = ufl_log(lam2 + eps_c)
    f1p = _c(1.0) / (lam1 + eps_c)
    f2p = _c(1.0) / (lam2 + eps_c)

    H11 = dot(P1, dot(dA, P1))
    H22 = dot(P2, dot(dA, P2))
    H12 = dot(P1, dot(dA, P2))
    H21 = dot(P2, dot(dA, P1))

    gamma = (f1 - f2) / delta
    return f1p * H11 + f2p * H22 + gamma * (H12 + H21)
