"""pycutfem.integration.quadrature
Unified quadrature provider for 1‑D, triangles, and quads (any Gauss order ≥1).
"""
# pycutfem.integration.quadrature
import numpy as np
from functools import lru_cache
from numpy.polynomial.legendre import leggauss
from pycutfem.fem import transform

from .dunavant_data import DUNAVANT
try:
    import numba as _nb  # type: ignore
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False



# -------------------------------------------------------------------------
# 1‑D Gauss–Legendre
# -------------------------------------------------------------------------
def gauss_legendre(order: int):
    if order < 1:
        raise ValueError(order)
    return leggauss(order)  # (points, weights)

def _gl01(order: int):
    """Gauss–Legendre nodes and weights mapped to [0,1]."""
    xi, w = gauss_legendre(int(order))
    lam = 0.5*(xi + 1.0)
    wl  = 0.5*w
    return lam, wl
# -------------------------------------------------------------------------
# Tensor‑product construction helpers
# -------------------------------------------------------------------------
@lru_cache(maxsize=None)
def quad_rule(order: int):
    xi, wi = gauss_legendre(order)
    pts = np.array([[x, y] for x in xi for y in xi])
    wts = np.array([wx * wy for wx in wi for wy in wi])
    return pts, wts

@lru_cache(maxsize=None)
def tri_rule(order: int,dunavant_deg: int = None):
    """Degree‑exact rule built from square → reference triangle mapping."""
    degree = 2 * order if dunavant_deg is None else dunavant_deg
    if degree in range(1, 21):
        pts = DUNAVANT[degree].points
        # Guard against accidental barycentric triplets
        if pts.ndim == 2 and pts.shape[1] == 3:
            pts = pts[:, 1:]  # (xi, eta) = (L2, L3)
        return pts, DUNAVANT[degree].weights
    
    xi, wi = gauss_legendre(order)
    u = 0.5 * (xi + 1.0)   # [0,1]
    w_u = 0.5 * wi
    pts = []
    wts = []
    for i, ui in enumerate(u):
        for j, vj in enumerate(u):
            r = ui
            s = vj * (1.0 - ui)
            weight = w_u[i] * w_u[j] * (1.0 - ui)
            pts.append([r, s])
            wts.append(weight)
    return np.array(pts), np.array(wts)

# -------------------------------------------------------------------------
# Edge / facet rules (reference domain) pycutfem/integration/quadrature.py
# -------------------------------------------------------------------------
def edge(element_type: str, edge_index: int, order: int = 2):
    xi, wi = gauss_legendre(order)
    if element_type == 'tri':
        t = 0.5 * (xi + 1.0)
        if edge_index == 0:   # (0,0)‑(1,0)
            pts = np.column_stack([t, np.zeros_like(t)])
            wts = 0.5 * wi
        elif edge_index == 1: # (1,0)‑(0,1)
            pts = np.column_stack([1 - t, t])
            wts = 0.5 * np.sqrt(2) * wi
        elif edge_index == 2: # (0,1)‑(0,0)
            pts = np.column_stack([np.zeros_like(t), 1 - t])
            wts = 0.5 * wi
        else:
            raise IndexError(edge_index)
        return pts, wts
    if element_type == 'quad':
        t = xi
        if edge_index == 0:   # bottom
            pts = np.column_stack([t, -np.ones_like(t)])
        elif edge_index == 1: # right
            pts = np.column_stack([np.ones_like(t), t])
        elif edge_index == 2: # top
            pts = np.column_stack([t[::-1], np.ones_like(t)])
        elif edge_index == 3: # left
            pts = np.column_stack([-np.ones_like(t), t[::-1]])
        else:
            raise IndexError(edge_index)
        return pts, wi
    raise KeyError(element_type)

# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------
def volume(element_type: str, order: int = 2):
    if element_type == 'tri':
        return tri_rule(order)
    if element_type == 'quad':
        return quad_rule(order)
    raise KeyError(element_type)


if _HAVE_NUMBA:
    @_nb.njit(cache=True, fastmath=True)
    def _map_line_rule(p0, p1, xi, w_ref):
        mid0 = 0.5*(p0[0] + p1[0]); mid1 = 0.5*(p0[1] + p1[1])
        half0 = 0.5*(p1[0] - p0[0]); half1 = 0.5*(p1[1] - p0[1])
        nQ = xi.shape[0]
        pts = np.empty((nQ, 2))
        for q in range(nQ):
            pts[q, 0] = mid0 + xi[q] * half0
            pts[q, 1] = mid1 + xi[q] * half1
        J = (half0*half0 + half1*half1) ** 0.5
        wts = w_ref * J
        return pts, wts

    @_nb.njit(cache=True, fastmath=True, parallel=True)
    def _map_line_rule_batched(P0, P1, xi, w_ref, out_pts, out_wts):
        nE = P0.shape[0]; nQ = xi.shape[0]
        for e in _nb.prange(nE):
            mid0 = 0.5*(P0[e, 0] + P1[e, 0]); mid1 = 0.5*(P0[e, 1] + P1[e, 1])
            half0 = 0.5*(P1[e, 0] - P0[e, 0]); half1 = 0.5*(P1[e, 1] - P0[e, 1])
            J = (half0*half0 + half1*half1) ** 0.5
            for q in range(nQ):
                out_pts[e, q, 0] = mid0 + xi[q] * half0
                out_pts[e, q, 1] = mid1 + xi[q] * half1
                out_wts[e, q] = w_ref[q] * J


def line_quadrature(p0: np.ndarray, p1: np.ndarray, order: int = 2):
    ξ, w_ref = gauss_legendre(order)
    if _HAVE_NUMBA:
        return _map_line_rule(np.asarray(p0, float), np.asarray(p1, float),
                              np.asarray(ξ, float), np.asarray(w_ref, float))
    # fallback (unchanged)
    mid  = 0.5*(p0 + p1)
    half = 0.5*(p1 - p0)
    pts  = mid[None, :] + np.outer(ξ, half)
    J    = np.linalg.norm(half)
    wts  = w_ref * J
    return pts, wts

def _project_to_levelset(point, level_set, *, mesh=None, eid=None, max_steps=3, tol=1e-12):
    """
    Newton-like orthogonal projection of a point onto {phi=0}.
    If mesh & eid are given and the level set is FE-backed, uses fast element-aware
    evaluation (no global element search).
    """
    x = np.asarray(point, float)
    use_elem = (mesh is not None) and (eid is not None) and hasattr(level_set, "value_on_element")
    for _ in range(int(max_steps)):
        if use_elem:
            # evaluate on the known parent element
            xi, eta = transform.inverse_mapping(mesh, int(eid), x)
            phi = level_set.value_on_element(int(eid), (float(xi), float(eta)))
            g = (level_set.gradient_on_element(int(eid), (float(xi), float(eta)))
                 if hasattr(level_set, "gradient_on_element")
                 else level_set.gradient(x))
        else:
            # generic path (analytic LS etc.)
            phi = level_set(x)          # <-- vector call (fixes TypeError)
            g   = level_set.gradient(x)

        g2 = float(np.dot(g, g)) + 1e-30
        if abs(phi) < tol or g2 <= 1e-30:
            break
        x = x - (phi / g2) * g
    return x

def curved_line_quadrature(level_set, p0, p1, order=2, nseg=1, project_steps=2, tol=1e-12):
    """
    Vectorized line quadrature along a curved interface (batched over segments).
    """
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    nseg = int(max(1, nseg))
    T = np.linspace(0.0, 1.0, nseg + 1)
    P = (1.0 - T)[:, None] * p0[None, :] + T[:, None] * p1[None, :]   # (nseg+1,2)

    # Batched Newton updates (generic LS); falls back to FD gradient if needed
    X = P.copy()
    for _ in range(int(max(1, project_steps))):
        phi_vals = np.empty(X.shape[0]); grads = np.empty_like(X)
        for i, xi in enumerate(X):
            try:     phi_vals[i] = float(level_set(xi))
            except:  phi_vals[i] = float(level_set(float(xi[0]), float(xi[1])))
            try:     grads[i] = level_set.gradient(xi)
            except:  # FD fallback
                h=1e-8
                gx=(float(level_set(xi[0]+h,xi[1]))-float(level_set(xi[0]-h,xi[1])))/(2*h)
                gy=(float(level_set(xi[0],xi[1]+h))-float(level_set(xi[0],xi[1]-h)))/(2*h)
                grads[i]=[gx,gy]
        g2 = np.sum(grads*grads, axis=1) + 1e-30
        mask = np.abs(phi_vals) >= tol
        if not np.any(mask): break
        X[mask] -= (phi_vals[mask]/g2[mask])[:,None]*grads[mask]

    ξ, w_ref = gauss_legendre(int(order))
    mid  = 0.5*(X[:-1] + X[1:])
    half = 0.5*(X[1:] - X[:-1])
    qpts = mid[:, None, :] + ξ.reshape(1, -1, 1) * half[:, None, :]
    qwts = w_ref.reshape(1, -1) * np.linalg.norm(half, axis=1).reshape(-1, 1)
    return qpts.reshape(-1, 2), qwts.reshape(-1)


def curved_wedge_quadrature(level_set, apex, arc_nodes, *,
                            order_t: int = 3, order_tau: int = 3):
    """
    Vectorized quadrature for a curved wedge with apex K and polyline 'arc_nodes'.
    Equivalent to the previous loop implementation but uses NumPy broadcasting.

    Returns:
        qpts(N,2), qwts(N) in physical coordinates (weights include area Jacobian).
    """
    K = np.asarray(apex, dtype=float)
    arc = np.asarray(arc_nodes, dtype=float)
    if arc.shape[0] < 2:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    # Gauss nodes on [0,1] for lambda (along-arc) and mu (radial)
    lam_t, w_lam = _gl01(int(order_t))
    mu_t,  w_mu  = _gl01(int(order_tau))

    # Segment endpoints and tangents along the arc
    A = arc[:-1, :]   # (m,2)
    B = arc[1:,  :]   # (m,2)
    dE = B - A        # (m,2)
    m = A.shape[0]
    if m == 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    # E(m,L,2) = (1-lam)*A + lam*B
    lam = lam_t.reshape(1, -1, 1)                         # (1,L,1)
    E = (1.0 - lam) * A[:, None, :] + lam * B[:, None, :] # (m,L,2)

    # |(K - E) x dE|
    dE_exp = dE[:, None, :]                                # (m,1,2)
    cross_mag = np.abs((K[0] - E[..., 0]) * dE_exp[..., 1] - (K[1] - E[..., 1]) * dE_exp[..., 0])  # (m,L)

    # Quadrature points X(m,L,M,2) and weights W(m,L,M)
    mu = mu_t.reshape(1, 1, -1, 1)
    X  = (1.0 - mu) * E[:, :, None, :] + mu * K.reshape(1, 1, 1, 2)
    W  = (cross_mag[:, :, None] *
          w_lam.reshape(1, -1, 1) *
          w_mu.reshape(1, 1, -1) *
          (1.0 - mu_t).reshape(1, 1, -1))

    return X.reshape(-1, 2), W.reshape(-1)

def curved_quad_ruled_quadrature(K1, K2, arc_nodes, *, order_lambda=3, order_mu=3):
    """
    Vectorized quadrature for a curved quadrilateral with straight base K1-K2 and polyline 'arc_nodes'.
    Uses a ruled surface parameterization sharing lambda along base and arc.

    Returns:
        qpts(N,2), qwts(N) with area-scaled weights.
    """
    K1 = np.asarray(K1, dtype=float)
    K2 = np.asarray(K2, dtype=float)
    arc = np.asarray(arc_nodes, dtype=float)
    if arc.shape[0] < 2:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    nseg = arc.shape[0] - 1
    lam_t, w_lam = _gl01(int(order_lambda))
    mu_t,  w_mu  = _gl01(int(order_mu))

    A = arc[:-1, :]         # (m,2)
    B = arc[1:,  :]         # (m,2)
    dE = B - A              # (m,2)
    m = A.shape[0]
    if m == 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    dS_dlam = (K2 - K1) / float(nseg)  # (2,)

    lam = lam_t.reshape(1, -1, 1)  # (1,L,1)
    E = (1.0 - lam) * A[:, None, :] + lam * B[:, None, :]                  # (m,L,2)
    s = (np.arange(m, dtype=float).reshape(-1, 1, 1) + lam) / float(nseg)  # (m,L,1)
    S = (1.0 - s) * K1.reshape(1, 1, 2) + s * K2.reshape(1, 1, 2)          # (m,L,2)

    mu = mu_t.reshape(1, 1, -1, 1)
    dX_dlam = (1.0 - mu) * dS_dlam.reshape(1, 1, 1, 2) + mu * dE.reshape(m, 1, 1, 2)
    dX_dmu  = E[:, :, None, :] - S[:, :, None, :]                           # (m,L,1,2)
    X       = (1.0 - mu) * S[:, :, None, :] + mu * E[:, :, None, :]

    jac = np.abs(dX_dlam[..., 0] * dX_dmu[..., 1] - dX_dlam[..., 1] * dX_dmu[..., 0])
    W   = jac * w_lam.reshape(1, -1, 1) * w_mu.reshape(1, 1, -1)

    return X.reshape(-1, 2), W.reshape(-1)