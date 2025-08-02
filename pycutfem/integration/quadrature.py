"""pycutfem.integration.quadrature
Unified quadrature provider for 1‑D, triangles, and quads (any Gauss order ≥1).
"""
# pycutfem.integration.quadrature
import numpy as np
from functools import lru_cache
from numpy.polynomial.legendre import leggauss
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
        print(f'points shape: {DUNAVANT[degree].points.shape}, weights shape: {DUNAVANT[degree].weights.shape}')
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
