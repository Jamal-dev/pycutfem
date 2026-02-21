from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # SciPy is a project dependency, but keep this example importable without it.
    import scipy.sparse as sp  # type: ignore
    import scipy.sparse.linalg as spla  # type: ignore

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - optional fallback for minimal envs
    sp = None  # type: ignore[assignment]
    spla = None  # type: ignore[assignment]
    _HAS_SCIPY = False


@dataclass(frozen=True)
class PDASOptions:
    """
    Options for PDAS / semismooth Newton on the (discrete) obstacle problem:

        A y - λ = f
        y >= ψ,  λ >= 0,  λᵀ (y-ψ) = 0

    Notes
    -----
    We use the standard semismooth NCP function

        C(y, λ) = λ - max(0, λ - c (y-ψ)) = 0

    so the active set indicator is based on the sign of:

        s := λ - c (y-ψ).
    """

    c: float = 1.0
    max_iter: int = 200
    cycle_detection: bool = True
    rtol_active: float = 0.0
    atol_active: float = 0.0


@dataclass
class PDASResult:
    y: np.ndarray
    lam: np.ndarray
    active: np.ndarray
    converged: bool
    n_iter: int
    history: Dict[str, List]


def _as_1d_float(x: np.ndarray | float, *, n: int, name: str) -> np.ndarray:
    if isinstance(x, (int, float, np.floating)):
        return np.full(n, float(x), dtype=float)
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"{name} must be a 1D array of length {n}; got shape {arr.shape}.")
    return arr


def _as_1d_bool(x: np.ndarray | None, *, n: int, name: str) -> np.ndarray:
    if x is None:
        return np.ones(n, dtype=bool)
    arr = np.asarray(x, dtype=bool)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"{name} must be a 1D boolean array of length {n}; got shape {arr.shape}.")
    return arr


def _active_indicator(*, y: np.ndarray, lam: np.ndarray, psi: np.ndarray, c: float) -> np.ndarray:
    # Active where s = λ - c(y-ψ) > 0
    return (lam - float(c) * (y - psi)) > 0.0


def _is_sparse(A) -> bool:
    return bool(_HAS_SCIPY and sp is not None and sp.isspmatrix(A))


def _solve_restricted(
    A,
    f: np.ndarray,
    psi: np.ndarray,
    *,
    active: np.ndarray,
    constrained: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Given an active set, solve:
      - y_A = psi_A
      - λ_I = 0
      - A y - λ = f (globally)

    Returns (y, λ, lin_residual_norm_on_inactive).
    """
    n = int(f.shape[0])
    idx_active = np.flatnonzero(active & constrained)
    idx_inactive = np.flatnonzero(~(active & constrained))

    y = np.empty(n, dtype=float)
    y[idx_active] = psi[idx_active]

    if idx_inactive.size:
        if _is_sparse(A):
            A_II = A[idx_inactive][:, idx_inactive].tocsr()
            if idx_active.size:
                rhs = f[idx_inactive] - A[idx_inactive][:, idx_active] @ y[idx_active]
            else:
                rhs = f[idx_inactive].copy()
            if spla is None:  # pragma: no cover
                raise RuntimeError("SciPy is required for sparse PDAS solves.")
            y[idx_inactive] = spla.spsolve(A_II, rhs)
        else:
            A_II = np.asarray(A[np.ix_(idx_inactive, idx_inactive)], dtype=float)
            if idx_active.size:
                rhs = f[idx_inactive] - np.asarray(A[np.ix_(idx_inactive, idx_active)], dtype=float) @ y[idx_active]
            else:
                rhs = f[idx_inactive].copy()
            y[idx_inactive] = np.linalg.solve(A_II, rhs)

    Ay = (A @ y) if _is_sparse(A) else (np.asarray(A, dtype=float) @ y)
    lam = np.zeros(n, dtype=float)
    lam[idx_active] = Ay[idx_active] - f[idx_active]
    lam[idx_inactive] = 0.0

    rI = Ay[idx_inactive] - f[idx_inactive]
    lin_res = float(np.linalg.norm(rI)) if idx_inactive.size else 0.0
    return y, lam, lin_res


def solve_obstacle_pdas(
    A,
    f: np.ndarray,
    psi: np.ndarray | float,
    *,
    y0: Optional[np.ndarray] = None,
    lam0: Optional[np.ndarray] = None,
    constrained: Optional[np.ndarray] = None,
    opts: Optional[PDASOptions] = None,
) -> PDASResult:
    """
    Primal-Dual Active Set (PDAS) solver for the discrete obstacle problem.

    Parameters
    ----------
    A:
        SPD matrix (sparse or dense).
    f:
        RHS vector.
    psi:
        Lower bound (obstacle), scalar or vector.
    y0, lam0:
        Optional initial guesses. If omitted, start from the unconstrained solve.
    constrained:
        Boolean mask (length n) indicating which DOFs are constrained by y>=psi.
        Unconstrained DOFs are treated as always inactive (λ=0).
    opts:
        PDAS options (c, max_iter, ...).
    """
    f = np.asarray(f, dtype=float).ravel()
    n = int(f.shape[0])
    psi = _as_1d_float(psi, n=n, name="psi")
    constrained = _as_1d_bool(constrained, n=n, name="constrained")
    opts = opts or PDASOptions()

    if _is_sparse(A):
        A = A.tocsr()
        if A.shape != (n, n):
            raise ValueError(f"A must have shape {(n, n)}; got {A.shape}.")
    else:
        A = np.asarray(A, dtype=float)
        if A.shape != (n, n):
            raise ValueError(f"A must have shape {(n, n)}; got {A.shape}.")

    if y0 is None:
        if _is_sparse(A):
            if spla is None:  # pragma: no cover
                raise RuntimeError("SciPy is required for sparse PDAS solves.")
            y = spla.spsolve(A, f)
        else:
            y = np.linalg.solve(A, f)
    else:
        y = np.asarray(y0, dtype=float).ravel().copy()
        if y.shape != (n,):
            raise ValueError(f"y0 must have shape {(n,)}; got {y.shape}.")

    if lam0 is None:
        lam = np.zeros(n, dtype=float)
    else:
        lam = np.asarray(lam0, dtype=float).ravel().copy()
        if lam.shape != (n,):
            raise ValueError(f"lam0 must have shape {(n,)}; got {lam.shape}.")

    active = _active_indicator(y=y, lam=lam, psi=psi, c=opts.c) & constrained

    hist: Dict[str, List] = dict(
        n_active=[],
        lin_res_inactive=[],
        min_gap=[],
        min_lam=[],
        delta_y_norm=[],
        active_changed=[],
    )

    seen: set[bytes] = set()
    converged = False
    for k in range(int(opts.max_iter)):
        if opts.cycle_detection:
            key = active.tobytes()
            if key in seen:
                break
            seen.add(key)

        y_new, lam_new, lin_res = _solve_restricted(A, f, psi, active=active, constrained=constrained)

        delta = float(np.linalg.norm(y_new - y))
        y = y_new
        lam = lam_new

        gap = y - psi
        min_gap = float(np.min(gap[constrained])) if np.any(constrained) else float(np.min(gap))
        min_lam = float(np.min(lam[constrained])) if np.any(constrained) else float(np.min(lam))

        hist["n_active"].append(int(np.count_nonzero(active)))
        hist["lin_res_inactive"].append(float(lin_res))
        hist["min_gap"].append(min_gap)
        hist["min_lam"].append(min_lam)
        hist["delta_y_norm"].append(delta)

        active_new = _active_indicator(y=y, lam=lam, psi=psi, c=opts.c) & constrained
        changed = bool(not np.array_equal(active_new, active))
        hist["active_changed"].append(changed)

        # Stopping rule: active set no longer changes.
        if not changed:
            converged = True
            break
        active = active_new

    return PDASResult(
        y=y,
        lam=lam,
        active=active,
        converged=converged,
        n_iter=len(hist["n_active"]),
        history=hist,
    )


def solve_obstacle_semismooth_newton(
    A,
    f: np.ndarray,
    psi: np.ndarray | float,
    *,
    y0: Optional[np.ndarray] = None,
    lam0: Optional[np.ndarray] = None,
    constrained: Optional[np.ndarray] = None,
    opts: Optional[PDASOptions] = None,
) -> PDASResult:
    """
    Semismooth Newton solver for the obstacle problem using the max-NCP function.

    For this (linear) obstacle problem, one semismooth Newton step is *exactly*
    one PDAS update (same active/inactive split and the same restricted solve).
    """
    return solve_obstacle_pdas(
        A,
        f,
        psi,
        y0=y0,
        lam0=lam0,
        constrained=constrained,
        opts=opts,
    )
