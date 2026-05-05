from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .pdas import PDASOptions, solve_obstacle_pdas, solve_obstacle_semismooth_newton


def _fd_stiffness_1d_dirichlet(n: int) -> np.ndarray:
    """
    1D finite-difference stiffness for -u'' on (0,1) with u(0)=u(1)=0.

    Unknowns are the n interior nodes.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    h = 1.0 / (n + 1)
    main = 2.0 / (h * h)
    off = -1.0 / (h * h)
    A = np.zeros((n, n), dtype=float)
    A.flat[:: n + 1] = main
    A.flat[1:: n + 1] = off
    A.flat[n:: n + 1] = off
    return A


def _analytic_obstacle_solution_1d(x: np.ndarray, *, f: float, g: float) -> np.ndarray:
    """
    Analytical solution for:
        -u'' = f,   u(0)=u(1)=0,   u(x) >= -g  (constant obstacle),  f < 0.

    Contact region is [a, 1-a] with a = sqrt(2g/(-f)).
    """
    if f >= 0:
        raise ValueError("Analytical solution here assumes f < 0.")
    if g <= 0:
        raise ValueError("g must be > 0.")
    a = float(np.sqrt(2.0 * g / (-f)))
    if a >= 0.5:
        # Full contact; u=-g everywhere in the interior in the limit. Keep it simple.
        return -float(g) * np.ones_like(x, dtype=float)
    x = np.asarray(x, dtype=float)
    u = np.empty_like(x, dtype=float)

    left = x <= a
    mid = (x > a) & (x < 1.0 - a)
    right = x >= 1.0 - a

    # u'' = -f > 0; on [0,a]: u(x) = (-f)/2 x^2 - (-f)*a x = (-f)(x^2/2 - a x)
    # and u(a) = -g, u'(a)=0
    u[left] = (-f) * (0.5 * x[left] * x[left] - a * x[left])
    u[mid] = -float(g)
    xr = 1.0 - x[right]
    u[right] = (-f) * (0.5 * xr * xr - a * xr)
    return u


@dataclass(frozen=True)
class Obstacle1DBenchmarkResult:
    status: str
    method: str
    n: int
    f: float
    g: float
    n_iter: int
    equilibrium_residual: float
    post_stable_delta_y: float
    min_gap: float
    min_lam: float
    max_active_gap: float
    l2_error: float
    linf_error: float
    contact_a_exact: float
    contact_a_est: float
    contact_a_error: float


def solve_obstacle_1d_benchmark(
    *,
    n: int = 200,
    f: float = -1.0,
    g: float = 0.05,
    method: str = "pdas",
    opts: Optional[PDASOptions] = None,
) -> Obstacle1DBenchmarkResult:
    """
    Scalar 1D obstacle benchmark for validating PDAS / semismooth Newton.

    Uses a finite-difference discretization on n interior nodes.
    """
    if f >= 0.0:
        raise ValueError("Pick f < 0 for a non-trivial contact patch with obstacle ψ=-g.")
    if g <= 0.0:
        raise ValueError("Pick g > 0.")

    A = _fd_stiffness_1d_dirichlet(n)
    x = np.linspace(0.0, 1.0, n + 2, dtype=float)
    xi = x[1:-1]

    # Discrete RHS corresponds to (f, v) with FD; use nodal forcing f(xi).
    # This keeps the benchmark simple and isolates PDAS logic.
    fvec = np.full(n, float(f), dtype=float)
    psi = -float(g)

    opts = opts or PDASOptions()
    if method.lower() in {"pdas", "primal-dual-active-set"}:
        out = solve_obstacle_pdas(A, fvec, psi, opts=opts)
        method_name = "pdas"
    elif method.lower() in {"ssn", "semismooth", "semismooth-newton", "semi-smooth"}:
        out = solve_obstacle_semismooth_newton(A, fvec, psi, opts=opts)
        method_name = "semismooth_newton"
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'pdas' or 'ssn'.")

    y = out.y
    lam = out.lam
    active = out.active

    Ay = A @ y
    eq_res = float(np.linalg.norm(Ay - lam - fvec))

    # Marker 2: once the active set stabilizes, the Newton update should be (near) zero.
    # For this linear obstacle problem, solving again with the stabilized set reproduces y.
    idx_a = np.flatnonzero(active)
    idx_i = np.flatnonzero(~active)
    y_polish = np.empty_like(y)
    y_polish[idx_a] = psi
    if idx_i.size:
        rhs = fvec[idx_i] - (A[np.ix_(idx_i, idx_a)] @ y_polish[idx_a] if idx_a.size else 0.0)
        y_polish[idx_i] = np.linalg.solve(A[np.ix_(idx_i, idx_i)], rhs)
    post_delta = float(np.linalg.norm(y_polish - y))

    # Reconstruct full u including boundary nodes (Dirichlet u=0).
    u_full = np.zeros(n + 2, dtype=float)
    u_full[1:-1] = y

    u_exact = _analytic_obstacle_solution_1d(x, f=f, g=g)
    err = u_full - u_exact
    try:
        integ = np.trapezoid(err * err, x)  # numpy>=2.0
    except AttributeError:  # pragma: no cover
        integ = np.trapz(err * err, x)  # older numpy
    l2 = float(np.sqrt(integ))
    linf = float(np.max(np.abs(err)))

    gap = y - psi
    min_gap = float(np.min(gap))
    min_lam = float(np.min(lam))
    max_active_gap = float(np.max(np.abs(gap[active]))) if np.any(active) else 0.0

    a_exact = float(np.sqrt(2.0 * g / (-f)))
    if np.any(active):
        # Estimate contact boundary from first/last active node location.
        idx = np.flatnonzero(active)
        a_est = float(xi[int(idx[0])])
    else:
        a_est = 1.0
    a_err = float(abs(a_est - a_exact))

    status = "converged" if out.converged else "not_converged"
    return Obstacle1DBenchmarkResult(
        status=status,
        method=method_name,
        n=int(n),
        f=float(f),
        g=float(g),
        n_iter=int(out.n_iter),
        equilibrium_residual=eq_res,
        post_stable_delta_y=post_delta,
        min_gap=min_gap,
        min_lam=min_lam,
        max_active_gap=max_active_gap,
        l2_error=l2,
        linf_error=linf,
        contact_a_exact=a_exact,
        contact_a_est=a_est,
        contact_a_error=a_err,
    )


def _format_result(res: Obstacle1DBenchmarkResult) -> str:
    return (
        f"status={res.status}, method={res.method}, n={res.n}, it={res.n_iter}, "
        f"||Ay-λ-f||={res.equilibrium_residual:.3e}, postΔy={res.post_stable_delta_y:.3e}, "
        f"min_gap={res.min_gap:.3e}, min_lam={res.min_lam:.3e}, "
        f"max_active_gap={res.max_active_gap:.3e}, "
        f"L2={res.l2_error:.3e}, Linf={res.linf_error:.3e}, "
        f"a_exact={res.contact_a_exact:.6f}, a_est={res.contact_a_est:.6f}, "
        f"|a-a*|={res.contact_a_error:.3e}"
    )


if __name__ == "__main__":
    res = solve_obstacle_1d_benchmark()
    print(_format_result(res))
