from __future__ import annotations
import numpy as np
from pycutfem.core.sideconvention import SIDE
from pycutfem.fem import transform
from typing import Iterable, Tuple, List
from pycutfem.integration.quadrature import _project_to_levelset 

try:
    import numba as _nb
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

# -------- Level-set evaluation that accepts both styles: φ(x) and φ(x,y) -----


# =============================================================================
# Level‑set evaluation (works for analytic or FE‑backed LS)
# =============================================================================
def phi_eval(level_set, x_phys, *, eid=None, xi_eta=None, mesh=None):
    """Safe φ evaluation at a physical point.
    * If a FE level set is attached, we use the element‑aware fast path.
    * Otherwise we call the analytic function (vector or scalar signature).
    """
    if hasattr(level_set,"value_on_element") and (eid is not None):
        if xi_eta is None:
            if mesh is None:
                raise ValueError("phi_eval needs xi_eta or mesh to inverse‑map.")
            xi_eta = transform.inverse_mapping(mesh, int(eid), np.asarray(x_phys, float))
        return float(level_set.value_on_element(int(eid), (float(xi_eta[0]), float(xi_eta[1]))))
    # generic fallback
    try:
        return float(level_set(np.asarray(x_phys, float)))
    except TypeError:
        return float(level_set(float(x_phys[0]), float(x_phys[1])))


# --------------------- Segment zero-crossing (linear φ) ----------------------
def segment_zero_crossing(p0: np.ndarray, p1: np.ndarray, phi0: float, phi1: float, eps: float = 1e-14) -> np.ndarray:
    """Linearized crossing on segment p0→p1 where the line φ crosses zero."""
    denom = (phi0 - phi1)
    if abs(denom) < eps:
        t = 0.5
    else:
        t = phi0 / (phi0 - phi1)
        t = min(max(t, 0.0), 1.0)
    return np.asarray(p0, float) + float(t) * (np.asarray(p1, float) - np.asarray(p0, float))

# =============================================================================
# Straight P1 clipping (triangle) for robust fallbacks
# =============================================================================
def clip_triangle_to_side(v_coords, v_phi, side='+', eps=0.0):
    """Clip a triangle against φ=0; keep 'side' ('+' keeps φ≥0, '−' keeps φ≤0)."""
    phi = np.asarray(v_phi, float)
    V   = [np.asarray(v, float) for v in v_coords]
    if   side == '+': keep = [i for i in range(3) if SIDE.is_pos(phi[i], tol=eps)]
    elif side == '-': keep = [i for i in range(3) if not SIDE.is_pos(phi[i], tol=eps)]
    else:             raise ValueError("side must be '+' or '-'")
    drop = [i for i in range(3) if i not in keep]
    if len(keep) == 3: return [V]
    if len(keep) == 0: return []
    if len(keep) == 1:
        iK = keep[0]; iD1, iD2 = drop
        I1 = segment_zero_crossing(V[iK], V[iD1], phi[iK], phi[iD1])
        I2 = segment_zero_crossing(V[iK], V[iD2], phi[iK], phi[iD2])
        return [[V[iK], I1, I2]]
    iK1, iK2 = keep; iD = drop[0]
    I1 = segment_zero_crossing(V[iK1], V[iD], phi[iK1], phi[iD])
    I2 = segment_zero_crossing(V[iK2], V[iD], phi[iK2], phi[iD])
    return [[V[iK1], I1, I2, V[iK2]]]


# -------------------- JIT versions of the geometric helpers ------------------
if _HAVE_NUMBA:
    @_nb.njit(cache=True, fastmath=True)
    def _clip_triangle_to_side_numba(V, phi, sgn, eps=0.0,
                                    pos_is_phi_nonnegative=True,
                                    zero_to_pos=True):
        """
        V: (3,2), phi: (3,), sgn=+1 for requested '+' side, -1 for '−'.
        pos_is_phi_nonnegative: if False, '+' means φ≤0 (flip mapping globally).
        zero_to_pos: if True, φ≈0 belongs to '+', otherwise to '−'.
        Returns (poly_pts(4,2), n_pts:int). For triangle n_pts=3; quad → 4; empty → 0.
        """
        # Apply global mapping first: if '+' means φ<=0, flip φ once up front
        sign_posmap = 1.0 if pos_is_phi_nonnegative else -1.0
        sgn_eff     = sgn * sign_posmap

        # Work with φ' = sgn_eff * φ so we always test "inside" against '+' logic below
        phi0 = sgn_eff * phi[0]; phi1 = sgn_eff * phi[1]; phi2 = sgn_eff * phi[2]

        # Decide inclusivity at zero by zero_to_pos:
        #   if zero_to_pos: '+' keeps ≥ -eps, '−' keeps > eps
        #   else          : '+' keeps >  eps, '−' keeps ≥ -eps
        def keep_at(phi_val, side_is_plus):
            if side_is_plus:
                return (phi_val >= -eps) if zero_to_pos else (phi_val > eps)
            else:
                return (phi_val >  eps)  if zero_to_pos else (phi_val >= -eps)

        side_is_plus = (sgn == 1.0)
        keep0 = keep_at(phi0, side_is_plus)
        keep1 = keep_at(phi1, side_is_plus)
        keep2 = keep_at(phi2, side_is_plus)
        n_keep = (1 if keep0 else 0) + (1 if keep1 else 0) + (1 if keep2 else 0)

        out = np.empty((4, 2), dtype=np.float64)
        if n_keep == 3:
            out[0] = V[0]; out[1] = V[1]; out[2] = V[2]
            return out, 3
        if n_keep == 0:
            return out, 0

        # linear interpolation on an edge using raw φ (not φ') for correct intersection
        def _seg_inter(Pa, Pb, phia, phib):
            denom = (phia - phib)
            if abs(denom) < 1e-14:
                t = 0.5
            else:
                t = phia / (phia - phib)
                if t < 0.0: t = 0.0
                elif t > 1.0: t = 1.0
            return np.array([Pa[0] + t*(Pb[0]-Pa[0]), Pa[1] + t*(Pb[1]-Pa[1])])

        if n_keep == 1:
            if keep0:  K=0; D1=1; D2=2
            elif keep1:K=1; D1=0; D2=2
            else:      K=2; D1=0; D2=1
            I1 = _seg_inter(V[K], V[D1], phi[K], phi[D1])
            I2 = _seg_inter(V[K], V[D2], phi[K], phi[D2])
            out[0] = V[K]; out[1] = I1; out[2] = I2
            return out, 3

        # n_keep == 2 → quad
        if not keep0:
            K1=1; K2=2; D=0
        elif not keep1:
            K1=0; K2=2; D=1
        else:
            K1=0; K2=1; D=2
        I1 = _seg_inter(V[K1], V[D], phi[K1], phi[D])
        I2 = _seg_inter(V[K2], V[D], phi[K2], phi[D])
        out[0] = V[K1]; out[1] = I1; out[2] = I2; out[3] = V[K2]
        return out, 4


    @_nb.njit(cache=True, fastmath=True)
    def _fan_triangulate_numba(poly, n_pts):
        """
        poly: (4,2) points; n_pts in {0,3,4}.
        Returns tris: (2,3,2) and n_tris in {0,1,2}. tri[0] valid if n_tris>=1; tri[1] if n_tris==2.
        """
        tris = np.empty((2, 3, 2), dtype=np.float64)
        if n_pts < 3:
            return tris, 0
        if n_pts == 3:
            tris[0, 0] = poly[0]; tris[0, 1] = poly[1]; tris[0, 2] = poly[2]
            return tris, 1
        # n_pts == 4: (v0,v1,v2) and (v0,v2,v3)
        tris[0, 0] = poly[0]; tris[0, 1] = poly[1]; tris[0, 2] = poly[2]
        tris[1, 0] = poly[0]; tris[1, 1] = poly[2]; tris[1, 2] = poly[3]
        return tris, 2

    @_nb.njit(cache=True, fastmath=True)
    def _map_ref_tri_to_phys_numba(A, B, C, qp_ref, qw_ref):
        """
        A,B,C: (2,) vertices; qp_ref:(nQ,2) on Δ(0,0)-(1,0)-(0,1), qw_ref:(nQ,)
        Returns (qp_phys(nQ,2), qw_phys(nQ,))
        """
        J00 = B[0] - A[0]; J01 = C[0] - A[0]
        J10 = B[1] - A[1]; J11 = C[1] - A[1]
        det = J00*J11 - J01*J10
        if det < 0.0: det = -det  # area
        nQ = qp_ref.shape[0]
        out_x = np.empty((nQ, 2), dtype=np.float64)
        out_w = np.empty((nQ,), dtype=np.float64)
        for q in range(nQ):
            r = qp_ref[q, 0]; s = qp_ref[q, 1]
            x0 = A[0] + r*J00 + s*J01
            x1 = A[1] + r*J10 + s*J11
            out_x[q, 0] = x0; out_x[q, 1] = x1
            out_w[q]    = qw_ref[q] * det
        return out_x, out_w


# ---------------------------- Fan triangulation ------------------------------
def fan_triangulate(poly: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Triangulate [v0, v1, ..., vk] into [(v0, vi, vi+1)]."""
    if len(poly) < 3: return []
    if len(poly) == 3: return [tuple(poly)]
    return [(poly[0], poly[i], poly[i+1]) for i in range(1, len(poly) - 1)]


# ------------- Map reference triangle quadrature to physical (A,B,C) --------
def map_ref_tri_to_phys(A, B, C, qp_ref, qw_ref):
    """Affine map Δ_ref→Δ_phys (row‑vector convention); scales weights by |detJ|."""
    A = np.asarray(A, float); B = np.asarray(B, float); C = np.asarray(C, float)
    J = np.array([[B[0] - A[0], C[0] - A[0]],
                  [B[1] - A[1], C[1] - A[1]]], float)
    detJ = abs(np.linalg.det(J))
    x_phys = qp_ref @ J.T + A
    w_phys = qw_ref * detJ
    return x_phys, w_phys


# -------------------------- Corner-tri list for an element -------------------
def corner_tris(mesh, elem):
    """Return list of local corner-triplets that tile the element geometry."""
    cn = list(elem.corner_nodes)
    if mesh.element_type == 'quad':
        # return [(0,1,3), (1,2,3)], cn
        # → triangles (0,1,2) and (0,2,3)
        return [(0,1,2), (0,2,3)], cn
    
    elif mesh.element_type == 'tri':
        return [ (0,1,2) ], cn


# ------------ barycentric Lagrange evaluation on [0,1] for tiny n ------------
_BARY_W_CACHE = {}
def _bary_weights(tnodes: np.ndarray) -> np.ndarray:
    key = tuple(np.asarray(tnodes, float).round(15))
    w = _BARY_W_CACHE.get(key)
    if w is None:
        x = np.asarray(tnodes, float)
        n = x.size
        w = np.ones(n, float)
        for j in range(n):
            for k in range(n):
                if k != j:
                    w[j] /= (x[j] - x[k])
        _BARY_W_CACHE[key] = w
    return w

def _bary_eval(t, tnodes, fvals, w=None) -> float:
    t = float(t)
    x = np.asarray(tnodes, float); y = np.asarray(fvals, float)
    if w is None: w = _bary_weights(x)
    diff = t - x
    k = np.argmin(np.abs(diff))
    if abs(diff[k]) <= 1e-15:
        return float(y[k])
    num = np.sum((w * y) / diff)
    den = np.sum(w / diff)
    return float(num / den)

# ------------------------------- Brent–Dekker --------------------------------
def _brent_root(fun, a, b, fa, fb, tol=SIDE.tol, maxit=64):
    if abs(fa) == 0.0: return a, fa
    if abs(fb) == 0.0: return b, fb
    if fa*fb > 0.0:    return None, None
    c, fc = a, fa; d = e = b - a
    for _ in range(int(maxit)):
        if fb*fc > 0.0:
            c, fc = a, fa; d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b; fa, fb, fc = fb, fc, fb
        tol1 = 2.0*np.finfo(float).eps*abs(b) + 0.5*tol
        xm   = 0.5*(c - b)
        if abs(xm) <= tol1 or fb == 0.0:
            return b, fb
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb/fa
            if a == c: p = 2.0*xm*s; q = 1.0 - s
            else:
                q = fa/fc; r = fb/fc
                p = s*(2.0*xm*q*(q-r) - (b-a)*(r-1.0)); q = (q-1.0)*(r-1.0)*(s-1.0)
            if p > 0: q = -q
            p = abs(p)
            if 2.0*p < min(3.0*xm*q - abs(tol1*q), abs(e*q)):
                e = d; d = p/q
            else:
                d = xm; e = d
        else:
            d = xm; e = d
        a, fa = b, fb
        b = b + (d if abs(d) > tol1 else np.sign(xm)*tol1)
        fb = fun(b)
    return b, fb

# ------------- reference param nodes on an edge (element-type aware) ----------
def _edge_ref_nodes(mesh, local_edge: int, p: int):
    """Return (tnodes in [0,1], xi(t), eta(t)) for the chosen local edge."""
    t = np.linspace(0.0, 1.0, int(p) + 1)
    if mesh.element_type == 'quad':
        if   local_edge == 0:  xi = 2*t - 1;  eta = -np.ones_like(t)
        elif local_edge == 1:  xi =  np.ones_like(t); eta = -1 + 2*t
        elif local_edge == 2:  xi =  1 - 2*t; eta =  np.ones_like(t)
        elif local_edge == 3:  xi = -np.ones_like(t); eta =  1 - 2*t
        else: raise IndexError(local_edge)
        return t, xi, eta
    elif mesh.element_type == 'tri':
        if   local_edge == 0: xi, eta = t, np.zeros_like(t)
        elif local_edge == 1: xi, eta = 1.0 - t, t
        elif local_edge == 2: xi, eta = np.zeros_like(t), 1.0 - t
        else: raise IndexError(local_edge)
        return t, xi, eta
    else:
        raise KeyError(mesh.element_type)

def _phi_on_e_points(level_set, mesh, eid: int, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    vals = np.empty_like(xi, float)
    if hasattr(level_set, "value_on_element"):
        for k in range(xi.size):
            vals[k] = float(level_set.value_on_element(int(eid), (float(xi[k]), float(eta[k]))))
    else:
        for k in range(xi.size):
            x = transform.x_mapping(mesh, int(eid), (float(xi[k]), float(eta[k])))
            vals[k] = float(level_set(np.asarray(x, float)))
    return vals

def _corner_ref_coords(mesh):
    if mesh.element_type == 'quad':
        return np.array([[-1.0,-1.0],[+1.0,-1.0],[+1.0,+1.0],[-1.0,+1.0]], float)
    elif mesh.element_type == 'tri':
        return np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0]], float)
    else:
        raise KeyError(mesh.element_type)

def edge_root_pn(level_set, mesh, eid: int, local_edge: int, *, tol: float = SIDE.tol) -> List[np.ndarray]:
    """Intersection(s) of {φ=0} with *this element's* local edge.
    Returns: [] (no hit), [P], or [P0,P1] if the whole edge lies on {φ=0}.
    """
    # sampling order
    if hasattr(level_set, "dh") and hasattr(level_set, "field"):
        p = max(1, int(level_set.dh.mixed_element._field_orders[level_set.field]))
    else:
        p = 1
    tnodes, xi, eta = _edge_ref_nodes(mesh, int(local_edge), p)
    fvals = _phi_on_e_points(level_set, mesh, int(eid), xi, eta)

    # whole edge on the interface?
    if np.all(np.abs(fvals) <= tol):
        P0 = transform.x_mapping(mesh, int(eid), (float(xi[0]),  float(eta[0])))
        P1 = transform.x_mapping(mesh, int(eid), (float(xi[-1]), float(eta[-1])))
        return [np.asarray(P0, float), np.asarray(P1, float)]

    # Brent–Dekker on each bracket
    w = _bary_weights(tnodes)
    fun = lambda tt: _bary_eval(tt, tnodes, fvals, w)
    crossings = []
    for k in range(len(tnodes) - 1):
        a, b = float(tnodes[k]), float(tnodes[k+1])
        fa, fb = float(fvals[k]), float(fvals[k+1])
        if fa * fb > 0.0:
            continue
        r, _ = _brent_root(fun, a, b, fa, fb, tol=tol)
        if r is None:
            continue
        # map r → (xi,eta) and then to physical
        if mesh.element_type == 'quad':
            if   local_edge == 0: xy = (2*r - 1, -1.0)
            elif local_edge == 1: xy = (1.0, -1 + 2*r)
            elif local_edge == 2: xy = (1 - 2*r, 1.0)
            else:                  xy = (-1.0, 1 - 2*r)
        else:  # tri
            if   local_edge == 0: xy = (r, 0.0)
            elif local_edge == 1: xy = (1 - r, r)
            else:                  xy = (0.0, 1 - r)
        P = transform.x_mapping(mesh, int(eid), (float(xy[0]), float(xy[1])))
        pphys = np.asarray(P, float)
        # de‑dup
        if not any(np.linalg.norm(pphys - Q) < SIDE.tol for Q in crossings):
            crossings.append(pphys)
    return crossings

# ---------------------------------------------------------------------------
# Local→reference corner permutation + segment root on element interior
# ---------------------------------------------------------------------------

def _ref_corners(mesh):
    if mesh.element_type == 'quad':
        return np.array([[-1,-1],[+1,-1],[+1,+1],[-1,+1]], float)
    elif mesh.element_type == 'tri':
        return np.array([[0,0],[1,0],[0,1]], float)
    else:
        raise KeyError(mesh.element_type)

def _loc2ref_corner_perm(mesh, eid, V, tol=1e-8):
    """perm[i_local_parent] → i_reference_corner (robust, unique)."""
    ref = _ref_corners(mesh)
    m = V.shape[0]
    perm = [-1]*m; used = set()
    for i in range(m):
        xi, eta = transform.inverse_mapping(mesh, int(eid), (float(V[i,0]), float(V[i,1])))
        X = np.array([float(xi), float(eta)])
        j = int(np.argmin(np.sum((ref - X)**2, axis=1)))
        if np.linalg.norm(ref[j] - X) > 5*tol and mesh.element_type == 'quad':
            sx = -1 if X[0] < 0 else +1
            sy = -1 if X[1] < 0 else +1
            for r, RR in enumerate(ref):
                if int(np.sign(RR[0])) == sx and int(np.sign(RR[1])) == sy and r not in used:
                    j = r; break
        if j in used:
            d = np.sum((ref - X)**2, axis=1)
            for r in np.argsort(d):
                if r not in used:
                    j = int(r); break
        perm[i] = j; used.add(j)
    return np.array(perm, int)


def segment_root_pn(level_set, mesh, eid: int, A_lc, B_lc, *, tol: float = SIDE.tol, order_hint: int = None):
    """Root(s) of φ((1−t)A+tB) on the *element* segment A→B in reference coords.
    Returns None (no crossing), a point P, or a pair (P0,P1) if whole segment lies on {φ=0}.
    """
    if order_hint is not None:
        p = int(order_hint)
    elif hasattr(level_set, "dh") and hasattr(level_set, "field"):
        p = max(1, int(level_set.dh.mixed_element._field_orders[level_set.field]))
    else:
        p = 1
    tnodes = np.linspace(0.0, 1.0, p + 1)
    xi  = (1.0 - tnodes) * float(A_lc[0]) + tnodes * float(B_lc[0])
    eta = (1.0 - tnodes) * float(A_lc[1]) + tnodes * float(B_lc[1])
    fvals = _phi_on_e_points(level_set, mesh, int(eid), xi, eta)

    if np.all(np.abs(fvals) <= tol):
        P0 = transform.x_mapping(mesh, int(eid), (float(xi[0]),  float(eta[0])))
        P1 = transform.x_mapping(mesh, int(eid), (float(xi[-1]), float(eta[-1])))
        return (np.asarray(P0, float), np.asarray(P1, float))

    w = _bary_weights(tnodes); fun = lambda tt: _bary_eval(tt, tnodes, fvals, w)
    for k in range(p):
        a, b = float(tnodes[k]), float(tnodes[k+1])
        fa, fb = float(fvals[k]), float(fvals[k+1])
        if fa * fb > 0.0:
            continue
        r, _ = _brent_root(fun, a, b, fa, fb, tol=tol)
        if r is None: 
            continue
        xr = (float(A_lc[0])*(1.0-r) + float(B_lc[0])*r,
              float(A_lc[1])*(1.0-r) + float(B_lc[1])*r)
        P = transform.x_mapping(mesh, int(eid), xr)
        return np.asarray(P, float)
    return None


# ---------------------------------------------------------------------------
# Interface arc sampling via element-aware projection
# ---------------------------------------------------------------------------

def interface_arc_nodes(level_set, I0, I1, *, mesh, eid, nseg: int, project_steps: int = 3, tol: float = 1e-12):
    """Return nodes on {φ=0} approximating the interface between I0 and I1}.
    Vectorized projection of all split points when the level‑set is analytic (no FE back-end).
    Falls back to element‑aware per‑point projection otherwise.
    """
    from pycutfem.integration.quadrature import _project_to_levelset
    I0 = np.asarray(I0, float); I1 = np.asarray(I1, float)
    nseg = int(max(1, nseg))
    T = np.linspace(0.0, 1.0, nseg + 1)
    P = (1.0 - T)[:, None] * I0[None, :] + T[:, None] * I1[None, :]

    # FE-backed LS? keep robust element-aware projector
    if hasattr(level_set, "value_on_element"):
        out = []
        for Pi in P:
            x = _project_to_levelset(Pi, level_set, mesh=mesh, eid=eid,
                                     max_steps=project_steps, tol=tol)
            if not out or np.linalg.norm(x - out[-1]) > 1e-14:
                out.append(x)
        return np.array(out, float)

    # Analytic LS → batched Newton updates
    X = P.copy()
    for _ in range(int(max(1, project_steps))):
        phi_vals = np.empty(X.shape[0], dtype=float)
        grads    = np.empty_like(X)
        for i, xi in enumerate(X):
            try:    phi_vals[i] = float(level_set(xi))
            except: phi_vals[i] = float(level_set(float(xi[0]), float(xi[1])))
            try:    grads[i]    = level_set.gradient(xi)
            except:
                h=1e-8
                gx=(float(level_set(xi[0]+h,xi[1]))-float(level_set(xi[0]-h,xi[1])))/(2*h)
                gy=(float(level_set(xi[0],xi[1]+h))-float(level_set(xi[0],xi[1]-h)))/(2*h)
                grads[i]=[gx,gy]
        g2 = np.sum(grads*grads, axis=1) + 1e-30
        mask = np.abs(phi_vals) >= tol
        if not np.any(mask): break
        X[mask] -= (phi_vals[mask]/g2[mask])[:,None]*grads[mask]

    # de-dup consecutive duplicates (very rare)
    out = [X[0]]
    for i in range(1, X.shape[0]):
        if np.linalg.norm(X[i] - out[-1]) > 1e-14:
            out.append(X[i])
    return np.array(out, float)


# ---------------------------------------------------------------------------
# Triangle-only: robust search for the two interface points on ∂K
# ---------------------------------------------------------------------------

def _find_two_edge_intersections_tri(level_set, mesh, eid, V, vphi, side, tol=1e-12, qvol=3):
    """Find the two *geometric* intersections {φ=0}∩∂K on this triangle.
    Returns ('pair',(I1,I2)) in physical coords; for degenerate 'whole edge'
    case, we fall back to full straight triangle quadrature ('final', (qx,qw)).
    """
    inter: List[np.ndarray] = []
    for eidx in (0, 1, 2):
        hits = edge_root_pn(level_set, mesh, int(eid), eidx, tol=tol)
        if not hits: continue
        if len(hits) == 2:
            # Degenerate: whole edge lies on φ=0 → use straight triangle
            from pycutfem.integration.quadrature import tri_rule
            qp_ref, qw_ref = tri_rule(qvol)
            qx, qw = map_ref_tri_to_phys(V[0], V[1], V[2], qp_ref, qw_ref)
            return 'final', (qx, qw)
        P = np.asarray(hits[0], float)
        if all(np.linalg.norm(P - Q) > 1e-13 for Q in inter):
            inter.append(P)
    if len(inter) < 2:
        polys = clip_triangle_to_side(V, vphi, side=side, eps=tol)
        if not polys:
            return 'final', (np.empty((0,2)), np.empty((0,)))
        from pycutfem.integration.quadrature import tri_rule
        qp_ref, qw_ref = tri_rule(qvol)
        qx_all, qw_all = [], []
        for poly in polys:
            for A,B,C in fan_triangulate(poly):
                xq, wq = map_ref_tri_to_phys(A,B,C, qp_ref, qw_ref)
                qx_all.append(xq); qw_all.append(wq)
        return 'final', (np.vstack(qx_all), np.hstack(qw_all))
    # return the farthest pair if >2 (guards nearly‑duplicate hits)
    if len(inter) > 2:
        best = (None, None, -1.0)
        for i in range(len(inter)):
            for j in range(i+1, len(inter)):
                d = np.linalg.norm(inter[i] - inter[j])
                if d > best[2]: best = (i, j, d)
        return 'pair', (inter[best[0]], inter[best[1]])
    return 'pair', (inter[0], inter[1])

def _normal_probe_flip(level_set, mesh, eid, side, I1, I2, keep, drop):
    """
    Probe on the *actual* interface normal near the midpoint of I1–I2.
    The old code sampled at the chord midpoint (off the interface), which can
    wrongly flip sides for distance-type φ.
    """
    mid_chord = 0.5*(np.asarray(I1, float) + np.asarray(I2, float))

    # 1) Project to φ=0 to get a reliable normal
    x0 = _project_to_levelset(mid_chord, level_set, mesh=mesh, eid=int(eid),
                              max_steps=8, tol=1e-14)

    # 2) Gradient at the interface point (element-aware if available)
    if hasattr(level_set, "gradient_on_element"):
        xi, eta = transform.inverse_mapping(mesh, int(eid), x0)
        g = level_set.gradient_on_element(int(eid), (float(xi), float(eta)))
    else:
        g = level_set.gradient(x0)

    n = np.asarray(g, float)
    n /= (np.linalg.norm(n) + 1e-30)

    # 3) Step size relative to element size (robust across meshes)
    h = mesh.element_char_length(int(eid)) if hasattr(mesh, "element_char_length") else 1.0
    step = max(1e-12, 1e-8 * h)
    probe = x0 + (step if side == '+' else -step) * n

    phi_probe = phi_eval(level_set, probe, eid=int(eid), mesh=mesh)
    want_pos  = (side == '+')
    ok = (phi_probe >= 0.0) if want_pos else (phi_probe <= 0.0)
    if ok:
        return False, keep, drop

    # Otherwise swap
    new_keep = [i for i in (0,1,2) if i not in keep]
    new_drop = [i for i in (0,1,2) if i not in new_keep]
    return True, new_keep, new_drop
# ---------------------------------------------------------------------------
# Curved quadrature on a *corner triangle* of a parent element (tri / quad)
# ---------------------------------------------------------------------------

def curved_subcell_quadrature_for_cut_triangle(mesh, eid, tri_local_ids, corner_ids, level_set,
                                               *, side='+', qvol: int = 3, nseg_hint: int = None, tol=1e-12):
    """Return (qpts, qwts) on ONE corner-triangle of a parent element.
    For triangle parents we detect I1/I2 with edge_root_pn and build either a
    curved wedge (one kept vertex) or a ruled curved quad (two kept vertices).
    Quads as parents are supported via the robust local→reference mapping.
    """
    from pycutfem.integration.quadrature import (
        tri_rule, curved_wedge_quadrature, curved_quad_ruled_quadrature
    )
    Vids = [corner_ids[i] for i in tri_local_ids]
    V    = mesh.nodes_x_y_pos[Vids].astype(float)      # (3,2)

    # φ at the three triangle vertices
    vphi = np.array([phi_eval(level_set, V[k], eid=eid, mesh=mesh) for k in range(3)], float)

    # requested side
    keep = [i for i in range(3) if (SIDE.is_pos(vphi[i], tol=tol) if side == '+' else not SIDE.is_pos(vphi[i], tol=tol))]
    drop = [i for i in range(3) if i not in keep]

    if len(keep) == 3:
        qp_ref, qw_ref = tri_rule(qvol)
        qx, qw = map_ref_tri_to_phys(V[0], V[1], V[2], qp_ref, qw_ref)
        return qx, qw
    if len(keep) == 0:
        return np.empty((0,2)), np.empty((0,))

    # element polynomial order for arc segmentation
    p_order = getattr(level_set.dh.mixed_element, "_field_orders", {}).get(getattr(level_set, "field", ""), 1) if hasattr(level_set, "dh") else 1
    nseg = max(3, (p_order + qvol//2 )) if (nseg_hint is None) else max(3, int(nseg_hint))

    # --- Triangle parent: get I1/I2 from element edges ----------------------
    if mesh.element_type == 'tri':
        status, payload = _find_two_edge_intersections_tri(level_set, mesh, eid, V, vphi, side, tol, qvol)
        if status == 'final':
            return payload  # (qx, qw) already

        I1, I2 = payload  # two interface points on this element (physical)

        # orientation guard (use the tentative pair for the sign check)
        flipped, keep, drop = _normal_probe_flip(level_set, mesh, eid, side, I1, I2, keep, drop)

        # Helper to get local edge index for a pair of local corners on a triangle
        def _tri_edge_index(a_loc, b_loc):
            ab = tuple(sorted((int(a_loc), int(b_loc))))
            if ab == (0,1): return 0
            if ab == (1,2): return 1
            if ab == (0,2): return 2
            raise IndexError(ab)

        if len(keep) == 1:
            # Wedge: intersections lie on the two edges adjacent to the kept vertex
            k = keep[0]
            d1, d2 = drop
            e1 = _tri_edge_index(k, d1)
            e2 = _tri_edge_index(k, d2)
            I1 = edge_root_pn(level_set, mesh, int(eid), e1, tol=tol)[0]
            I2 = edge_root_pn(level_set, mesh, int(eid), e2, tol=tol)[0]
            K  = V[k]
            arc = interface_arc_nodes(level_set, I1, I2, mesh=mesh, eid=eid,
                                    nseg=nseg, project_steps=3, tol=tol)
            return curved_wedge_quadrature(level_set, K, arc, order_t=qvol, order_tau=qvol)

        else:  # len(keep) == 2
            # Ruled curved quad: intersections lie on the two edges adjacent to the *dropped* vertex
            d = drop[0]
            k1, k2 = keep[0], keep[1]
            e1 = _tri_edge_index(k1, d)
            e2 = _tri_edge_index(k2, d)
            I1 = edge_root_pn(level_set, mesh, int(eid), e1, tol=tol)[0]
            I2 = edge_root_pn(level_set, mesh, int(eid), e2, tol=tol)[0]
            K1, K2 = V[k1], V[k2]
            arc = interface_arc_nodes(level_set, I1, I2, mesh=mesh, eid=eid,
                                    nseg=nseg, project_steps=3, tol=tol)
            return curved_quad_ruled_quadrature(K1, K2, arc, order_lambda=qvol, order_mu=qvol)


    # --- Quad parent: build I1/I2 on triangle edges in reference coords -----
    V_parent = mesh.nodes_x_y_pos[corner_ids].astype(float)
    loc2ref  = _loc2ref_corner_perm(mesh, eid, V_parent)
    corner_ref = _corner_ref_coords(mesh)

    def _edge_point(a_loc, b_loc):
        ai = int(tri_local_ids[a_loc]); bi = int(tri_local_ids[b_loc])
        A_lc = corner_ref[int(loc2ref[ai])]; B_lc = corner_ref[int(loc2ref[bi])]
        P = segment_root_pn(level_set, mesh, int(eid), A_lc, B_lc, tol=tol)
        if P is not None:
            return P if isinstance(P, np.ndarray) else P[0]
        return segment_zero_crossing(V[a_loc], V[b_loc], vphi[a_loc], vphi[b_loc])

    # 1) Build tentative intersections for the current keep/drop
    if len(keep) == 1:
        I1 = _edge_point(keep[0], drop[0]); I2 = _edge_point(keep[0], drop[1])
    else:  # len(keep) == 2
        I1 = _edge_point(keep[0], drop[0]); I2 = _edge_point(keep[1], drop[0])

    # 2) Probe the normal and possibly flip keep/drop …
    flipped, keep, drop = _normal_probe_flip(level_set, mesh, eid, side, I1, I2, keep, drop)

    # 3) … then RE-BRANCH on the *updated* keep/drop
    if len(keep) == 1:
        I1 = _edge_point(keep[0], drop[0]); I2 = _edge_point(keep[0], drop[1])
        K  = V[keep[0]]
        arc = interface_arc_nodes(level_set, I1, I2, mesh=mesh, eid=eid,
                                nseg=nseg, project_steps=3, tol=tol)
        return curved_wedge_quadrature(level_set, K, arc, order_t=qvol, order_tau=qvol)
    else:  # len(keep) == 2
        I1 = _edge_point(keep[0], drop[0]); I2 = _edge_point(keep[1], drop[0])
        K1, K2 = V[keep[0]], V[keep[1]]
        arc = interface_arc_nodes(level_set, I1, I2, mesh=mesh, eid=eid,
                                nseg=nseg, project_steps=3, tol=tol)
        return curved_quad_ruled_quadrature(K1, K2, arc, order_lambda=qvol, order_mu=qvol)

# ---------------------



# def clip_triangle_to_side_pn(mesh, eid, tri_local_ids, corner_ids, level_set, side='+', eps=0.0):
#     """
#     Edge-order-aware triangle clip:
#     - On element boundary edges → use P^n edge_root_pn(...)
#     - On the interior diagonal     → fall back to linear segment_zero_crossing
#     Returns list of polygons (list[list[2]]) like clip_triangle_to_side.
#     """
#     # triangle vertices (global corner node ids → coords)
#     v_ids = [corner_ids[i] for i in tri_local_ids]
#     V = mesh.nodes_x_y_pos[v_ids].astype(float)          # (3,2)
#     # φ at the three vertices (safe path accepts eid)
#     v_phi = np.array([phi_eval(level_set, V[k], eid=eid, mesh=mesh) for k in range(3)], float)

#     # Decide kept vs dropped as in the P1 version
#     keep = []
#     if side == '+':
#         keep = [i for i in range(3) if SIDE.is_pos(v_phi[i], tol=eps)]
#     elif side == '-':
#         keep = [i for i in range(3) if SIDE.is_neg(v_phi[i], tol=eps)]
#     else:
#         raise ValueError("side must be '+' or '-'")
#     drop = [i for i in range(3) if i not in keep]

#     if len(keep) == 3: return [ [V[0], V[1], V[2]] ]
#     if len(keep) == 0: return []

#     # helper: is this triangle edge also a *parent element boundary*?
#     def _is_elem_boundary_pair(a_local_corner, b_local_corner):
#         # element corners are 0..(n-1) in order; boundary edges are consecutive
#         n = len(corner_ids)             # 3 for tri elements, 4 for quads
#         a = int(tri_local_ids[a_local_corner]); b = int(tri_local_ids[b_local_corner])
#         # mapped to element's corner indices
#         A = a; B = b
#         # on a quad, boundary pairs: (0,1),(1,2),(2,3),(3,0)
#         if mesh.element_type == 'quad':
#             s = {(0,1),(1,2),(2,3),(0,3)}
#             return tuple(sorted((A,B))) in s
#         # on a tri, every edge is boundary
#         return True

#     corner_ref = _corner_ref_coords(mesh)

#     def _edge_point(a_loc, b_loc):
#         # local corner indices in the *parent element*
#         ai = int(tri_local_ids[a_loc]); bi = int(tri_local_ids[b_loc])
#         A_lc = corner_ref[ai]; B_lc = corner_ref[bi]

#         # One Pⁿ routine for *any* element segment (boundary or diagonal)
#         P = segment_root_pn(level_set, mesh, int(eid), A_lc, B_lc, tol=eps)
#         if P is not None:
#             # P may be a single point or a (P0,P1) degenerate segment; take a point
#             return P if isinstance(P, np.ndarray) else P[0]

#         # No sign change on this edge → safe linear fallback (rare with correct brackets)
#         return segment_zero_crossing(V[a_loc], V[b_loc], v_phi[a_loc], v_phi[b_loc])


#     if len(keep) == 1:
#         k = keep[0]; d1, d2 = drop
#         I1 = _edge_point(k, d1); I2 = _edge_point(k, d2)
#         return [[V[k], I1, I2]]
#     else:
#         k1, k2 = keep; d = drop[0]
#         I1 = _edge_point(k1, d); I2 = _edge_point(k2, d)
#         return [[V[k1], I1, I2, V[k2]]]


def clip_triangle_to_side_pn(mesh, eid, tri_local_ids, corner_ids, level_set, side='+', eps=0.0):
    # triangle vertices (global ids → coords)
    v_ids = [corner_ids[i] for i in tri_local_ids]
    V = mesh.nodes_x_y_pos[v_ids].astype(float)

    # φ at the three triangle vertices (element-aware)
    v_phi = np.array([phi_eval(level_set, V[k], eid=eid, mesh=mesh) for k in range(3)], float)

    # Decide kept vs dropped (match P1 variant’s convention re: zeros)
    if side == '+':
        keep = [i for i in range(3) if SIDE.is_pos(v_phi[i], tol=eps)]
    elif side == '-':
        keep = [i for i in range(3) if not SIDE.is_pos(v_phi[i], tol=eps)]
    else:
        raise ValueError("side must be '+' or '-'")
    drop = [i for i in range(3) if i not in keep]

    if len(keep) == 3: return [ [V[0], V[1], V[2]] ]
    if len(keep) == 0: return []

    # --- NEW: robust local-corner → reference-corner mapping (of the parent) ---
    V_parent = mesh.nodes_x_y_pos[corner_ids].astype(float)
    loc2ref  = _loc2ref_corner_perm(mesh, eid, V_parent)   # maps 0..m-1 → 0..m-1 in reference
    corner_ref = _corner_ref_coords(mesh)                  # canonical reference coordinates

    def _edge_point(a_loc, b_loc):
        # map triangle-local corner indices → element-local corner indices
        ai = int(tri_local_ids[a_loc]); bi = int(tri_local_ids[b_loc])
        # then to *reference* coordinates using the robust permutation
        A_lc = corner_ref[int(loc2ref[ai])]
        B_lc = corner_ref[int(loc2ref[bi])]

        # One Pⁿ routine for any element segment (boundary or diagonal)
        P = segment_root_pn(level_set, mesh, int(eid), A_lc, B_lc, tol=eps)
        if P is not None:
            return P if isinstance(P, np.ndarray) else P[0]

        # Safe linear fallback (rare when brackets are correct)
        return segment_zero_crossing(V[a_loc], V[b_loc], v_phi[a_loc], v_phi[b_loc])

    # assemble polygon exactly as before using _edge_point(...)
    if len(keep) == 1:
        k = keep[0]; d1, d2 = drop
        I1 = _edge_point(k, d1); I2 = _edge_point(k, d2)
        return [[V[k], I1, I2]]
    else:
        k1, k2 = keep; d = drop[0]
        I1 = _edge_point(k1, d); I2 = _edge_point(k2, d)
        return [[V[k1], I1, I2, V[k2]]]



