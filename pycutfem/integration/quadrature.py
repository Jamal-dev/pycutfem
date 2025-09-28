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

# ============================================================================
# Vectorized (batched) Qp isoparametric interface line quadrature
# ============================================================================

@lru_cache(maxsize=None)
def _lagrange_nodes_weights(p: int):
    """Equispaced Lagrange nodes on [-1,1] and their barycentric weights."""
    x = np.linspace(-1.0, 1.0, int(p) + 1, dtype=float)
    # barycentric weights
    w = np.ones_like(x)
    for j in range(x.size):
        for k in range(x.size):
            if k != j:
                w[j] /= (x[j] - x[k])
    return x, w

def _barycentric_basis_mats(order: int, p: int):
    """
    Build basis matrices N(s_i), dN(s_i) at Gauss points s_i on [-1,1].
    Returns:
        s_i: (ng,)
        w_g: (ng,)
        N : (ng, p+1)
        dN: (ng, p+1)
    """
    s_i, w_g = gauss_legendre(int(order))          # strictly inside (-1,1)
    x, w = _lagrange_nodes_weights(int(p))         # nodes & bary weights
    # Broadcasted barycentric eval
    # t_{i,j} = w_j / (s_i - x_j)
    denom = (s_i[:, None] - x[None, :])
    t = w[None, :] / denom                         # (ng, m)
    T1 = t.sum(axis=1, keepdims=True)              # (ng,1)
    N = t / T1                                     # (ng, m)

    # For derivatives, use: N'_j(s) = [ -w_j/(s-x_j)^2 * T1 + (w_j/(s-x_j)) * T2 ] / T1^2
    inv = 1.0 / denom
    inv2 = inv * inv                               # (ng, m)
    T2 = (w[None, :] * inv2).sum(axis=1, keepdims=True)  # (ng,1)
    dN = ((-w[None, :] * inv2) * T1 + (w[None, :] * inv) * T2) / (T1 * T1)
    return s_i, w_g, N, dN

def _eval_phi_batch(level_set, X):
    """Robust φ(X) for X shape (...,2), with vectorized fallback."""
    X = np.asarray(X, float)
    try:
        phi = level_set(X)
        phi = np.asarray(phi, float)
        if phi.shape == X.shape[:-1]:
            return phi
    except Exception:
        pass
    # fallback
    phi = np.empty(X.shape[:-1], dtype=float)
    it = np.nditer(np.zeros(phi.shape), flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        x = X[idx]
        try:     phi[idx] = float(level_set(x))
        except:  phi[idx] = float(level_set(float(x[0]), float(x[1])))
        it.iternext()
    return phi

def _eval_grad_batch(level_set, X):
    """Robust ∇φ(X) for X shape (...,2).
    Tries a truly-vectorized gradient first, but validates it against a few
    per-point evaluations to avoid accidental broadcasting bugs in user code
    (e.g. using np.linalg.norm without axis). Falls back to per-point eval.
    """
    X = np.asarray(X, float)
    # Attempt fast vectorized path, but validate correctness on a few samples
    try:
        g_vec = np.asarray(level_set.gradient(X), float)
        if g_vec.shape == X.shape:
            # Validate against scalar calls at up to 3 points
            XS = X.reshape(-1, 2)
            GS = g_vec.reshape(-1, 2)
            n = XS.shape[0]
            sample_ids = [0, max(0, n // 2), max(0, n - 1)]
            ok = True
            for sid in sample_ids:
                try:
                    gi = np.asarray(level_set.gradient(XS[sid]), float)
                except Exception:
                    gi = np.asarray(level_set.gradient(float(XS[sid, 0]), float(XS[sid, 1])), float)
                if not np.all(np.isfinite(gi)):
                    ok = False; break
                if gi.shape != (2,):
                    ok = False; break
                # relative check to allow tiny numerical differences
                denom = max(1.0, float(np.linalg.norm(gi)))
                if float(np.linalg.norm(gi - GS[sid])) > 1e-8 * denom:
                    ok = False; break
            if ok:
                return g_vec
    except Exception:
        pass
    # Try a vectorized finite-difference fallback using the (possibly vectorized)
    # level_set __call__. This remains fast when __call__ supports arrays.
    try:
        h = 1e-8
        e1 = np.array([h, 0.0], dtype=float)
        e2 = np.array([0.0, h], dtype=float)
        f1p = _eval_phi_batch(level_set, X + e1)
        f1m = _eval_phi_batch(level_set, X - e1)
        f2p = _eval_phi_batch(level_set, X + e2)
        f2m = _eval_phi_batch(level_set, X - e2)
        Gfd = np.empty_like(X, dtype=float)
        Gfd[..., 0] = (f1p - f1m) / (2.0 * h)
        Gfd[..., 1] = (f2p - f2m) / (2.0 * h)
        return Gfd
    except Exception:
        pass

    # Fallback: robust per-point gradient evaluation
    G = np.empty(X.shape, dtype=float)
    it = np.nditer(np.zeros(X.shape[:-1]), flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        x = X[idx]
        try:
            G[idx] = np.asarray(level_set.gradient(x), float)
        except Exception:
            G[idx] = np.asarray(level_set.gradient(float(x[0]), float(x[1])), float)
        it.iternext()
    return G

def _project_to_levelset_batch(X0, level_set, *, max_steps=3, tol=1e-12):
    """
    Batched Newton-like projection onto {φ=0}.
    X0: (...,2)
    Returns X with same shape.
    """
    X = np.asarray(X0, float).copy()
    for _ in range(int(max(1, max_steps))):
        phi = _eval_phi_batch(level_set, X)                  # (...,)
        G   = _eval_grad_batch(level_set, X)                 # (...,2)
        g2  = np.sum(G*G, axis=-1) + 1e-30                   # (...,)
        mask = np.abs(phi) > tol
        if not np.any(mask):
            break
        # update only where needed
        upd = (phi[..., None] / g2[..., None]) * G
        X[mask] -= upd[mask]
    return X

def _project_to_levelset_batch_elementwise(X0, level_set, *, mesh, eids, max_steps=3, tol=1e-12):
    """
    Element-aware batched projection onto {φ=0}.
    X0: (E, m, 2) points grouped per element; eids: (E,) element ids.
    Uses value_on_element/gradient_on_element if available, with exact Jacobians,
    avoiding any global searches.
    """
    from pycutfem.fem import transform as _tr
    X = np.asarray(X0, float).copy()
    eids = np.asarray(eids, int)
    if X.ndim != 3:
        raise ValueError("X0 must be (E,m,2)")
    E, m, _ = X.shape
    for _ in range(int(max(1, max_steps))):
        any_mask = False
        for ei in range(E):
            eid = int(eids[ei])
            # inverse-map all points to (xi,eta)
            xi = np.empty(m, dtype=float); eta = np.empty(m, dtype=float)
            for k in range(m):
                xi_k, eta_k = _tr.inverse_mapping(mesh, eid, X[ei, k, :])
                xi[k] = float(xi_k); eta[k] = float(eta_k)

            # φ and ∇φ on element (vectorized if available)
            if hasattr(level_set, 'values_on_element_many'):
                phi = np.asarray(level_set.values_on_element_many(eid, xi, eta), float)
            else:
                phi = np.empty(m, dtype=float)
                if hasattr(level_set, 'value_on_element'):
                    for k in range(m):
                        phi[k] = float(level_set.value_on_element(eid, (xi[k], eta[k])))
                else:
                    for k in range(m):
                        phi[k] = float(level_set(X[ei, k, :]))

            if hasattr(level_set, 'gradients_on_element_many'):
                G = np.asarray(level_set.gradients_on_element_many(eid, xi, eta), float)
            elif hasattr(level_set, 'gradient_on_element'):
                G = np.empty((m, 2), dtype=float)
                for k in range(m):
                    G[k, :] = np.asarray(level_set.gradient_on_element(eid, (xi[k], eta[k])), float)
            else:
                G = np.empty((m, 2), dtype=float)
                for k in range(m):
                    G[k, :] = np.asarray(level_set.gradient(X[ei, k, :]), float)

            g2 = np.sum(G*G, axis=1) + 1e-30
            mask = np.abs(phi) > tol
            if np.any(mask):
                any_mask = True
                X[ei, mask, :] -= (phi[mask][:, None] / g2[mask][:, None]) * G[mask, :]
        if not any_mask:
            break
    return X

def _qpline_nodes_elementwise_ref(level_set, mesh, eids, P0, P1, *, p: int, project_steps: int = 3, tol: float = 1e-12):
    """
    Build Qp line nodes for many elements using reference-space projection.
    Only used when the level set is FE-backed (has dh/field and values_on_element_many).

    Returns Pnodes(E, p+1, 2) physical points on Γ per element.
    """
    import numpy as _np
    from pycutfem.fem import transform as _tr
    assert hasattr(level_set, 'values_on_element_many'), "ref projection needs FE-backed level-set"
    E = int(_np.asarray(eids).size)
    s_nodes, _ = _lagrange_nodes_weights(int(p))
    t_nodes = 0.5*(s_nodes + 1.0)  # [0,1]

    # Reference endpoints via a one-time inverse map
    R0 = _np.empty((E, 2), dtype=float)
    R1 = _np.empty((E, 2), dtype=float)
    for i, eid in enumerate(_np.asarray(eids, int)):
        R0[i, :] = _tr.inverse_mapping(mesh, int(eid), _np.asarray(P0[i], float))
        R1[i, :] = _tr.inverse_mapping(mesh, int(eid), _np.asarray(P1[i], float))

    # Initial reference nodes along the chord
    R = (1.0 - t_nodes[None, :, None]) * R0[:, None, :] + t_nodes[None, :, None] * R1[:, None, :]

    dh = getattr(level_set, 'dh', None)
    field = getattr(level_set, 'field', None)
    me = dh.mixed_element if dh is not None else None
    assert me is not None and field is not None

    for _ in range(int(max(1, project_steps))):
        any_mask = False
        for i, eid in enumerate(_np.asarray(eids, int)):
            xi = R[i, :, 0].copy(); eta = R[i, :, 1].copy()
            phi = _np.asarray(level_set.values_on_element_many(int(eid), xi, eta), float)
            dN_ref = me._eval_scalar_grad_many(field, xi, eta)  # (m, n_loc, 2)
            gd  = _np.asarray(dh.element_maps[field][int(eid)], int)
            vals = level_set._f.get_nodal_values(gd)           # (n_loc,)
            g_ref = _np.einsum('n,mnd->md', vals, dN_ref, optimize=True)
            g2 = _np.sum(g_ref*g_ref, axis=1) + 1e-30
            mask = _np.abs(phi) > tol
            if _np.any(mask):
                any_mask = True
                R[i, mask, :] -= (phi[mask][:, None] / g2[mask][:, None]) * g_ref[mask, :]
                # keep in reference domain for quads
                if mesh.element_type == 'quad':
                    R[i, :, 0] = _np.clip(R[i, :, 0], -1.0, 1.0)
                    R[i, :, 1] = _np.clip(R[i, :, 1], -1.0, 1.0)
        if not any_mask:
            break

    # Map to physical once
    Pnodes = _np.empty((E, t_nodes.size, 2), dtype=float)
    for i, eid in enumerate(_np.asarray(eids, int)):
        for k in range(t_nodes.size):
            Pnodes[i, k, :] = _tr.x_mapping(mesh, int(eid), (float(R[i, k, 0]), float(R[i, k, 1])))
    return Pnodes

def isoparam_interface_line_quadrature_batch(level_set, P0, P1, *, p: int = 2, order: int = 4,
                                             project_steps: int = 3, tol: float = 1e-12,
                                             mesh=None, eids=None):
    """
    Fully vectorized Qp isoparametric interface quadrature for many segments at once.
    Parameters
    ----------
    level_set : object
        Needs __call__(x) and gradient(x); vectorized versions speed it up further.
    P0, P1 : array_like, shape (E,2)
        The two chord endpoints per cut element.
    p : int
        Geometry order (Qp); uses p+1 Lagrange nodes on [-1,1].
    order : int
        Gauss-Legendre order along the curve.
    Returns
    -------
    qpts : (E, order, 2) float
        Physical quadrature points on Γ.
    qw   : (E, order) float
        Quadrature weights (already scaled by arc-length Jacobian).
    """
    P0 = np.asarray(P0, float); P1 = np.asarray(P1, float)
    E = P0.shape[0]
    if E == 0:
        return np.empty((0, order, 2), float), np.empty((0, order), float)

    # Build Qp nodes along the curve Γ per element
    if (mesh is not None) and (eids is not None) and hasattr(level_set, 'values_on_element_many'):
        Pnodes = _qpline_nodes_elementwise_ref(level_set, mesh, eids, P0, P1,
                                               p=int(p), project_steps=int(project_steps), tol=float(tol))
    else:
        s_nodes, _ = _lagrange_nodes_weights(int(p))
        t_nodes = 0.5*(s_nodes + 1.0)                               # map [-1,1] -> [0,1]
        Pguess = (1.0 - t_nodes[None, :, None]) * P0[:, None, :] + t_nodes[None, :, None] * P1[:, None, :]
        Pnodes = _project_to_levelset_batch(Pguess, level_set, max_steps=int(project_steps), tol=float(tol))
    # Basis at Gauss points
    s_i, w_g, N, dN = _barycentric_basis_mats(int(order), int(p))   # (ng,), (ng,), (ng,m), (ng,m)

    # x(s_i) and x'(s_i) in batch: einsum over nodes
    # Pnodes: (E, m, 2) ; N: (ng, m) -> x: (E, ng, 2)
    x  = np.einsum('im,emk->eik', N,  Pnodes, optimize=True)
    xs = np.einsum('im,emk->eik', dN, Pnodes, optimize=True)

    # Weights scaled by |x'(s)|
    J = np.linalg.norm(xs, axis=2)            # (E, ng)
    qw = w_g[None, :] * J                     # broadcast
    return x, qw

# ============================================================================
# Robust & fast polyline-based batch line quadrature for Γ segments
# ============================================================================
def curved_line_quadrature_batch(level_set, P0, P1, *, order=4, nseg=3, project_steps=2, tol=1e-12,
                                 mesh=None, eids=None):
    """
    Vectorized version of curved_line_quadrature for many segments.
    Parameters
    ----------
    level_set : object with __call__(x) and gradient(x)
    P0, P1 : (E,2) arrays of segment endpoints
    order : Gauss-Legendre order per subsegment
    nseg : number of chord subdivisions before projection
    project_steps : Newton updates toward {phi=0}
    Returns
    -------
    qpts : (E, nseg*order, 2)
    qwts : (E, nseg*order,)
    """
    P0 = np.asarray(P0, float); P1 = np.asarray(P1, float)
    E = P0.shape[0]
    nseg = int(max(1, nseg))
    if E == 0:
        return np.empty((0, 0, 2), float), np.empty((0, 0), float)

    T = np.linspace(0.0, 1.0, nseg + 1, dtype=float)             # (nseg+1,)
    # chord nodes for all segments: (E, nseg+1, 2)
    P = (1.0 - T[None, :, None]) * P0[:, None, :] + T[None, :, None] * P1[:, None, :]

    # Project all nodes to {phi=0}
    if (mesh is not None) and (eids is not None) and hasattr(level_set, 'value_on_element'):
        X = _project_to_levelset_batch_elementwise(P, level_set, mesh=mesh, eids=eids,
                                                   max_steps=int(project_steps), tol=float(tol))
    else:
        X = _project_to_levelset_batch(P, level_set, max_steps=int(project_steps), tol=float(tol))

    # subsegments
    A = X[:, :-1, :]                                         # (E, nseg, 2)
    B = X[:,  1:, :]                                         # (E, nseg, 2)

    # per-subsegment Gauss rule on [-1,1]
    xi, w_ref = gauss_legendre(int(order))                   # (q,), (q,)
    # mid + Xi*half, with norms for weights
    mid  = 0.5*(A + B)                                       # (E,nseg,2)
    half = 0.5*(B - A)                                       # (E,nseg,2)

    # Expand to (E,nseg,order,2) and (E,nseg,order)
    qpts = mid[:, :, None, :] + xi.reshape(1,1,-1,1) * half[:, :, None, :]
    qwts = w_ref.reshape(1,1,-1) * np.linalg.norm(half, axis=2).reshape(E, nseg, 1)

    # Collapse the middle two axes (nseg,order) -> (nseg*order)
    E_, S, Q = qwts.shape
    return qpts.reshape(E_, S*Q, 2), qwts.reshape(E_, S*Q)
