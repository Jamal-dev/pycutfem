from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogitShiftResult:
    values: np.ndarray
    shift: float
    mass: float
    target_mass: float
    iterations: int


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid for numpy arrays."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit_shift_to_match_integral(
    values: np.ndarray,
    *,
    weights: np.ndarray,
    target_mass: float,
    eps: float = 1.0e-15,
    rtol: float = 1.0e-12,
    max_iter: int = 80,
    z_clip: float = 50.0,
) -> LogitShiftResult:
    """Shift values in logit-space to match a weighted integral.

    Given a scalar field `a` with values in [0,1], define the logit transform
        z = log(a / (1-a)).
    This function finds a scalar shift λ such that:
        sum_i weights[i] * sigmoid(z[i] + λ) == target_mass
    and returns the corrected field.

    This is useful as a lightweight "volume" constraint for diffuse indicators,
    since a constant shift in logit-space approximately translates the interface
    without polluting far-field 0/1 regions.
    """
    a0 = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if a0.shape != w.shape:
        raise ValueError(f"values and weights must have the same shape, got {a0.shape} vs {w.shape}")

    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    total_mass = float(np.dot(w, np.ones_like(w)))
    if total_mass <= 0.0:
        raise ValueError("sum(weights) must be positive")

    tgt = float(target_mass)
    if not np.isfinite(tgt):
        raise ValueError("target_mass must be finite")
    if tgt < 0.0 or tgt > total_mass:
        raise ValueError(f"target_mass must be in [0,sum(weights)] = [0,{total_mass:.3e}], got {tgt:.3e}")

    if tgt == 0.0:
        out = np.zeros_like(a0)
        return LogitShiftResult(values=out, shift=-np.inf, mass=0.0, target_mass=tgt, iterations=0)
    if tgt == total_mass:
        out = np.ones_like(a0)
        return LogitShiftResult(values=out, shift=np.inf, mass=total_mass, target_mass=tgt, iterations=0)

    a = np.clip(a0, 0.0, 1.0)
    mass0 = float(np.dot(w, a))
    if abs(mass0 - tgt) <= rtol * max(1.0, abs(tgt)):
        return LogitShiftResult(values=a, shift=0.0, mass=mass0, target_mass=tgt, iterations=0)

    eps = float(eps)
    if eps <= 0.0 or eps >= 0.5:
        raise ValueError("eps must be in (0,0.5)")
    a_clip = np.clip(a, eps, 1.0 - eps)
    z = np.log(a_clip / (1.0 - a_clip))

    def _mass(lam: float) -> float:
        zz = np.clip(z + float(lam), -float(z_clip), float(z_clip))
        return float(np.dot(w, _sigmoid(zz)))

    # Bracket the solution
    if mass0 < tgt:
        lo, hi = 0.0, 1.0
        m_hi = _mass(hi)
        while m_hi < tgt:
            hi *= 2.0
            if hi > 200.0:
                break
            m_hi = _mass(hi)
    else:
        hi, lo = 0.0, -1.0
        m_lo = _mass(lo)
        while m_lo > tgt:
            lo *= 2.0
            if lo < -200.0:
                break
            m_lo = _mass(lo)

    m_lo = _mass(lo)
    m_hi = _mass(hi)
    if not (m_lo <= tgt <= m_hi):
        raise RuntimeError(
            f"Failed to bracket target mass: mass(lo)={m_lo:.3e}, mass(hi)={m_hi:.3e}, target={tgt:.3e}"
        )

    it = 0
    for it in range(1, int(max_iter) + 1):
        mid = 0.5 * (lo + hi)
        m_mid = _mass(mid)
        if abs(m_mid - tgt) <= rtol * max(1.0, abs(tgt)):
            lo = hi = mid
            break
        if m_mid < tgt:
            lo = mid
        else:
            hi = mid

    lam = 0.5 * (lo + hi)
    zz = np.clip(z + float(lam), -float(z_clip), float(z_clip))
    out = _sigmoid(zz)
    mass_out = float(np.dot(w, out))
    return LogitShiftResult(values=out, shift=float(lam), mass=mass_out, target_mass=tgt, iterations=it)

