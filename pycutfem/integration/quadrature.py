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
        # print(f'points shape: {DUNAVANT[degree].points.shape}, weights shape: {DUNAVANT[degree].weights.shape}')
        return DUNAVANT[degree].points, DUNAVANT[degree].weights
    
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
    Integrate along a *curved* interface inside an element by splitting the chord
    p0–p1 into 'nseg' subsegments, Newton-projecting their endpoints to {phi=0},
    and applying Gauss–Legendre on each projected subsegment.

    Returns (qpts, qwts) in *physical* coordinates, weights already scaled by segment lengths.
    """
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    # parametric breakpoints along the chord
    T = np.linspace(0.0, 1.0, int(max(1, nseg)) + 1)
    # initial endpoints on the chord
    P = np.outer(1 - T, p0) + np.outer(T, p1)

    # project endpoints to the level set curve
    P_proj = np.empty_like(P)
    for i, Pi in enumerate(P):
        x = Pi
        for _ in range(int(max(1, project_steps))):
            phi = level_set(x[0], x[1])
            if abs(phi) < tol:
                break
            g = level_set.gradient(x)
            g2 = float(np.dot(g, g)) + 1e-30
            x = x - (phi / g2) * g
        P_proj[i] = x

    # accumulate quadrature from each subsegment
    qpts_list = []
    qwts_list = []
    for i in range(len(P_proj) - 1):
        a, b = P_proj[i], P_proj[i+1]
        # (optional) skip tiny segments
        if np.linalg.norm(b - a) < 1e-14:
            continue
        qp, qw = line_quadrature(a, b, order=order)  # already physical + length scaled
        qpts_list.append(qp)
        qwts_list.append(qw)

    if not qpts_list:
        return np.empty((0, 2), float), np.empty((0,), float)

    qpts = np.vstack(qpts_list)
    qwts = np.concatenate(qwts_list)
    return qpts, qwts

def curved_wedge_quadrature(level_set, apex, arc_nodes, *,
                            order_t: int = 3, order_tau: int = 3):
    """
    Quadrature for the curved wedge: union over polyline segments of the arc.
    Returns (qpts(N,2), qwts(N)).
    """
    K = np.asarray(apex, float)
    arc_nodes = np.asarray(arc_nodes, float)
    if arc_nodes.shape[0] < 2:
        return np.empty((0,2)), np.empty((0,))

    lam_t, wt = _gl01(order_t)
    mu,  wu   = _gl01(order_tau)

    qx = []
    qw = []
    for a, b in zip(arc_nodes[:-1], arc_nodes[1:]):
        dEdlam = (b - a)                     # derivative wrt λ on [0,1]
        # skip tiny pieces (degenerate)
        if np.linalg.norm(dEdlam) < 1e-15:
            continue
        for j in range(len(lam_t)):
            E = (1.0 - lam_t[j]) * a + lam_t[j] * b
            cross_mag = abs((K[0]-E[0])*dEdlam[1] - (K[1]-E[1])*dEdlam[0])
            for i in range(len(mu)):
                X = (1.0 - mu[i]) * E + mu[i] * K
                w = wt[j] * wu[i] * (1.0 - mu[i]) * cross_mag
                qx.append(X);  qw.append(w)
    return np.array(qx), np.array(qw)

def curved_quad_ruled_quadrature(K1, K2, arc_nodes, *, order_lambda=3, order_mu=3):
    """
    Quadrature for a curved quadrilateral with straight base K1-K2 and curved top 'arc_nodes'.
    Returns (qpts(N,2), qwts(N)) in physical coordinates with area-scaled weights.
    """
    K1 = np.asarray(K1, float); K2 = np.asarray(K2, float)
    arc = np.asarray(arc_nodes, float)
    if arc.shape[0] < 2:
        return np.empty((0,2)), np.empty((0,))

    nseg = arc.shape[0] - 1
    lam_t, w_lam = _gl01(order_lambda)  # Gauss points on [0,1]
    mu_t,  w_mu  = _gl01(order_mu)

    qx, qw = [], []
    dS_dlam = (K2 - K1) / float(nseg)   # derivative of S w.r.t. local lam on each arc subsegment

    for i in range(nseg):
        a, b = arc[i], arc[i+1]
        dE_dlam = (b - a)               # derivative of E on this local segment

        for j, lam in enumerate(lam_t):
            # interpolate along arc and base with a *shared* lambda
            E = (1.0 - lam) * a + lam * b
            s = (i + lam) / float(nseg)                # shared parameter in [0,1] along K1->K2
            S = (1.0 - s) * K1 + s * K2

            for k, mu in enumerate(mu_t):
                X = (1.0 - mu) * S + mu * E
                dX_dlam = (1.0 - mu) * dS_dlam + mu * dE_dlam
                dX_dmu  = E - S
                # 2D "cross product" magnitude
                jac = abs(dX_dlam[0]*dX_dmu[1] - dX_dlam[1]*dX_dmu[0])

                w = w_lam[j] * w_mu[k] * jac
                qx.append(X);  qw.append(w)

    return np.array(qx), np.array(qw)
