import numpy as np
from pycutfem.core.sideconvention import SIDE
from pycutfem.fem import transform

try:
    import numba as _nb
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

# -------- Level-set evaluation that accepts both styles: φ(x) and φ(x,y) -----


def phi_eval(level_set, x_phys, *, eid=None, xi_eta=None, mesh=None):
    # Fast path: FE level set with element context
    if hasattr(level_set, "value_on_element") and (eid is not None):
        if xi_eta is None:
            if mesh is None:
                raise ValueError("phi_eval needs xi_eta or mesh to inverse-map.")
            xi_eta = transform.inverse_mapping(mesh, int(eid), np.asarray(x_phys, float))
        return level_set.value_on_element(int(eid), (float(xi_eta[0]), float(xi_eta[1])))
    # Generic fallback (analytic or FE): may do owner search
    try:
        return level_set(np.asarray(x_phys))
    except TypeError:
        return level_set(x_phys[0], x_phys[1])

# --------------------- Segment zero-crossing (linear φ) ----------------------
def segment_zero_crossing(p0, p1, phi0, phi1, eps=1e-14):
    """Return intersection on segment p0->p1 where the linearized φ crosses 0."""
    denom = (phi0 - phi1)
    if abs(denom) < eps:
        t = 0.5
    else:
        t = phi0 / (phi0 - phi1)
        t = min(max(t, 0.0), 1.0)
    return p0 + t * (p1 - p0)

# ------------- Clip a triangle against φ=0; keep 'side' ('+' or '-') --------
def clip_triangle_to_side(v_coords, v_phi, side='+', eps=0.0):
    """
    v_coords: 3x2 array of triangle vertices (physical)
    v_phi:    φ at those vertices (length 3)
    side:     '+' keeps {φ>=0}, '-' keeps {φ<=0}
    Returns list of polygons (each as list of 2D points).
    """
    # sgn = +1.0 if side == '+' else -1.0
    # phi = sgn * np.asarray(v_phi, dtype=float)
    phi = np.asarray(v_phi, dtype=float)
    V   = [np.asarray(v, dtype=float) for v in v_coords]

    # Decide membership using the global convention only once
    if   side == '+': keep = [i for i in range(3) if SIDE.is_pos(phi[i], tol=eps)]
    elif side == '-': keep = [i for i in range(3) if not SIDE.is_pos(phi[i], tol=eps)]
    else:             raise ValueError("side must be '+' or '-'")
    # # Map requested side into phi, then a single positivity test
    # keep = [i for i in range(3) if SIDE.is_pos(phi[i], tol=eps)]
    drop = [i for i in range(3) if i not in keep]

    if len(keep) == 3:
        return [V]
    if len(keep) == 0:
        return []
    # helper: intersection with dropped side
    from_idx = keep if len(keep) == 1 else drop
    if len(keep) == 1:
        iK = keep[0]
        iD1, iD2 = drop
        I1 = segment_zero_crossing(V[iK], V[iD1], phi[iK], phi[iD1])
        I2 = segment_zero_crossing(V[iK], V[iD2], phi[iK], phi[iD2])
        return [[V[iK], I1, I2]]
    else:
        iK1, iK2 = keep
        iD       = drop[0]
        I1 = segment_zero_crossing(V[iK1], V[iD], phi[iK1], phi[iD])  # on K1–D
        I2 = segment_zero_crossing(V[iK2], V[iD], phi[iK2], phi[iD])  # on K2–D
        # Boundary order: K1 → I1 → I2 → K2
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
def fan_triangulate(poly):
    """Triangulate convex polygon [v0, v1, ..., vk] into [(v0,vi,vi+1)]."""
    if len(poly) < 3: return []
    if len(poly) == 3: return [tuple(poly)]
    return [ (poly[0], poly[i], poly[i+1]) for i in range(1, len(poly)-1) ]

# ------------- Map reference triangle quadrature to physical (A,B,C) --------
def map_ref_tri_to_phys(A, B, C, qp_ref, qw_ref):
    """
    Transforms quadrature points and weights from a reference triangle to a physical triangle.

    This function performs a standard affine mapping from the reference element
    (with vertices at (0,0), (1,0), and (0,1)) to the physical element defined
    by vertices A, B, and C.

    The affine map is defined as:
        x = F(ξ) = J @ ξ + A
    where ξ is a column vector of reference coordinates [ξ, η]ᵀ, and J is the
    Jacobian of the transformation.

    **Row-Vector Convention**:
    The input quadrature points `qp_ref` are provided as a NumPy array of shape
    (nQ, 2), which is a stack of row vectors. To apply the linear transformation
    `J @ ξ` to a row vector `ξ_row = [ξ, η]`, the correct matrix multiplication
    is `ξ_row @ J.T`. This function implements this transposed multiplication
    to correctly handle the batch of row-vector inputs.

    **Weight Transformation**:
    The quadrature weights are scaled by the absolute value of the Jacobian
    determinant, which represents the ratio of the area of the physical element
    to the reference element. The returned weights `w_phys` are ready for direct
    use in the quadrature sum.

    Args:
        A (array-like): Coordinates of the first vertex of the physical triangle (2-vector).
        B (array-like): Coordinates of the second vertex of the physical triangle (2-vector).
        C (array-like): Coordinates of the third vertex of the physical triangle (2-vector).
        qp_ref (np.ndarray): Quadrature points on the reference triangle, shape (nQ, 2).
        qw_ref (np.ndarray): Quadrature weights on the reference triangle, shape (nQ,).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - x_phys (np.ndarray): The quadrature points in physical coordinates, shape (nQ, 2).
            - w_phys (np.ndarray): The quadrature weights in physical space, shape (nQ,).
    """
    A = np.asarray(A); B = np.asarray(B); C = np.asarray(C)
    J = np.array([[B[0] - A[0], C[0] - A[0]],
                  [B[1] - A[1], C[1] - A[1]]], dtype=float)
    detJ = abs(np.linalg.det(J))
    x_phys = (qp_ref @ J.T) + A
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
    if abs(diff[k]) <= 1e-15:   # exact hit
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
    for _ in range(maxit):
        if fb*fc > 0.0:
            c, fc = a, fa; d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2.0*np.finfo(float).eps*abs(b) + 0.5*tol
        xm = 0.5*(c - b)
        if abs(xm) <= tol1 or fb == 0.0:
            return b, fb
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb/fa
            if a == c:
                p = 2.0*xm*s; q = 1.0 - s
            else:
                q = fa/fc; r = fb/fc
                p = s*(2.0*xm*q*(q-r) - (b-a)*(r-1.0))
                q = (q-1.0)*(r-1.0)*(s-1.0)
            if p > 0: q = -q
            p = abs(p)
            min1 = 3.0*xm*q - abs(tol1*q)
            min2 = abs(e*q)
            if 2.0*p < (min1 if min1 < min2 else min2):
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
def _edge_ref_nodes(mesh, local_edge: int, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (tnodes, xi, eta) where tnodes in [0,1] are Lagrange points k/p,
    and (xi,eta) are corresponding reference coords along the chosen edge.
    """
    tnodes = np.linspace(0.0, 1.0, int(p)+1)
    if mesh.element_type == 'quad':
        # local edges: 0: bottom(η=-1), 1: right(ξ=+1), 2: top(η=+1), 3: left(ξ=-1)
        if   local_edge == 0:  xi = 2*tnodes - 1; eta = -np.ones_like(tnodes)
        elif local_edge == 1:  xi =  np.ones_like(tnodes); eta = -1 + 2*tnodes
        elif local_edge == 2:  xi =  1 - 2*tnodes; eta =  np.ones_like(tnodes)
        elif local_edge == 3:  xi = -np.ones_like(tnodes); eta =  1 - 2*tnodes
        else: raise IndexError(local_edge)
        return tnodes, xi, eta
    elif mesh.element_type == 'tri':
        # 0:(0,0)-(1,0)  1:(1,0)-(0,1)  2:(0,1)-(0,0)
        if   local_edge == 0: xi, eta = tnodes, np.zeros_like(tnodes)
        elif local_edge == 1: xi, eta = 1.0 - tnodes, tnodes
        elif local_edge == 2: xi, eta = np.zeros_like(tnodes), 1.0 - tnodes
        else: raise IndexError(local_edge)
        return tnodes, xi, eta
    else:
        raise KeyError(mesh.element_type)

def _phi_on_e_points(level_set, mesh, eid: int, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """φ at a *set* of reference points on element eid (fast when FE‑backed)."""
    vals = np.empty_like(xi, float)
    if hasattr(level_set, "value_on_element"):
        for k in range(xi.size):
            vals[k] = float(level_set.value_on_element(int(eid), (float(xi[k]), float(eta[k]))))
    else:
        for k in range(xi.size):
            x = transform.x_mapping(mesh, int(eid), (float(xi[k]), float(eta[k])))
            vals[k] = float(level_set(np.asarray(x, float)))
    return vals

def edge_root_pn(level_set, mesh, eid: int, local_edge: int, *, tol: float = SIDE.tol):
    """
    Return intersection point(s) between {φ=0} and *this element's* local edge.

    Result:
      []                → no intersection on this edge
      [P]               → a single crossing
      [P0, P1]          → whole edge lies on {φ=0} (degeneracy)
    """
    # 1) degree from FE level set if available; else fall back to linear sampling
    p = 1
    if hasattr(level_set, "dh") and hasattr(level_set, "field"):
        p = max(1, int(level_set.dh.mixed_element._field_orders[level_set.field]))
    tnodes, xi, eta = _edge_ref_nodes(mesh, int(local_edge), p)
    fvals = _phi_on_e_points(level_set, mesh, int(eid), xi, eta)

    # whole edge on interface?
    if np.all(np.abs(fvals) <= tol):
        P0 = transform.x_mapping(mesh, int(eid), (float(xi[0]),   float(eta[0])))
        P1 = transform.x_mapping(mesh, int(eid), (float(xi[-1]),  float(eta[-1])))
        return [np.asarray(P0, float), np.asarray(P1, float)]

    # scan brackets [t_k, t_{k+1}]
    w = _bary_weights(tnodes)
    fun = lambda tt: _bary_eval(tt, tnodes, fvals, w)
    crossings = []
    for k in range(len(tnodes)-1):
        a, b = float(tnodes[k]), float(tnodes[k+1])
        fa, fb = float(fvals[k]), float(fvals[k+1])
        if fa*fb > 0.0:
            continue
        r, _ = _brent_root(fun, a, b, fa, fb, tol=tol)
        if r is None: 
            continue
        # map r → (xi,eta) and then to physical
        if mesh.element_type == 'quad':
            if   local_edge == 0: xy = (2*r-1, -1.0)
            elif local_edge == 1: xy = (1.0, -1+2*r)
            elif local_edge == 2: xy = (1-2*r, 1.0)
            else:                  xy = (-1.0, 1-2*r)
        else:  # tri
            if   local_edge == 0: xy = (r, 0.0)
            elif local_edge == 1: xy = (1-r, r)
            else:                  xy = (0.0, 1-r)
        P = transform.x_mapping(mesh, int(eid), (float(xy[0]), float(xy[1])))
        crossings.append(np.asarray(P, float))
    # de-dup close roots
    out = []
    for p in crossings:
        if not any(np.linalg.norm(p-q) < SIDE.tol for q in out):
            out.append(p)
    return out

def _corner_ref_coords(mesh):
    if mesh.element_type == 'quad':
        # local corners 0..3
        return np.array([[-1.0,-1.0],[+1.0,-1.0],[+1.0,+1.0],[-1.0,+1.0]], float)
    elif mesh.element_type == 'tri':
        # local corners 0..2
        return np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0]], float)
    else:
        raise KeyError(mesh.element_type)

def segment_root_pn(level_set, mesh, eid: int, A_lc, B_lc, *, tol: float = SIDE.tol, order_hint: int = None):
    """
    Root of φ((1-t)A + tB) on the *element* segment A_lc→B_lc, t∈[0,1],
    using Pⁿ sampling (n = FE order if available, else order_hint, else 1).
    Returns:
      None   → no crossing on this segment
      P      → single intersection as physical coordinates
      (P0,P1)→ entire segment lies on φ=0 (degenerate case)
    """
    # decide sampling order
    if order_hint is not None:
        p = int(order_hint)
    elif hasattr(level_set, "dh") and hasattr(level_set, "field"):
        p = max(1, int(level_set.dh.mixed_element._field_orders[level_set.field]))
    else:
        p = 1

    tnodes = np.linspace(0.0, 1.0, p+1)
    xi  = (1.0 - tnodes) * float(A_lc[0]) + tnodes * float(B_lc[0])
    eta = (1.0 - tnodes) * float(A_lc[1]) + tnodes * float(B_lc[1])
    fvals = _phi_on_e_points(level_set, mesh, int(eid), xi, eta)

    # whole segment on interface?
    if np.all(np.abs(fvals) <= tol):
        P0 = transform.x_mapping(mesh, int(eid), (float(xi[0]),  float(eta[0])))
        P1 = transform.x_mapping(mesh, int(eid), (float(xi[-1]), float(eta[-1])))
        return (np.asarray(P0, float), np.asarray(P1, float))

    # scan brackets [t_k,t_{k+1}] with Brent–Dekker on barycentric interpolant
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

def _append_if_new(poly, P, tol=1e-14):
    P = np.asarray(P, float)
    if not poly:
        poly.append(P); return
    if np.linalg.norm(poly[-1] - P) > tol:
        poly.append(P)

def _ref_corners(mesh):
    if mesh.element_type == 'quad':
        return np.array([[-1,-1],[+1,-1],[+1,+1],[-1,+1]], float)  # 0..3
    elif mesh.element_type == 'tri':
        return np.array([[0,0],[1,0],[0,1]], float)                 # 0..2
    else:
        raise KeyError(mesh.element_type)

def _loc2ref_corner_perm(mesh, eid, V, tol=1e-8):
    """Map element-local corner index i → reference corner index r."""
    ref = _ref_corners(mesh)
    m = V.shape[0]
    perm = [-1]*m
    used = set()
    for i in range(m):
        xi, eta = transform.inverse_mapping(mesh, int(eid), (float(V[i,0]), float(V[i,1])))
        X = np.array([float(xi), float(eta)])
        # match to nearest ref corner
        j = int(np.argmin(np.sum((ref - X)**2, axis=1)))
        if np.linalg.norm(ref[j]-X) > 5*tol:
            # last resort: snap by sign of (xi,eta)
            if mesh.element_type == 'quad':
                sx = -1 if X[0] < 0 else +1
                sy = -1 if X[1] < 0 else +1
                for r, RR in enumerate(ref):
                    if int(np.sign(RR[0])) == sx and int(np.sign(RR[1])) == sy and r not in used:
                        j = r; break
        # ensure permutation (unique)
        if j in used:
            # pick the closest unused
            d = np.sum((ref - X)**2, axis=1)
            for r in np.argsort(d):
                if r not in used:
                    j = int(r); break
        perm[i] = j
        used.add(j)
    return np.array(perm, dtype=int)

def _corner_phi_fast(level_set, mesh, cn):
    """φ at corner nodes; use FE nodal read if possible."""
    if hasattr(level_set, "dh") and hasattr(level_set, "field") and hasattr(level_set, "_f"):
        node_map = level_set.dh.dof_map.get(level_set.field, {})
        g2l      = getattr(level_set._f, "_g2l", {})
        nv       = level_set._f.nodal_values
        out = np.empty(len(cn), float)
        for k, nid in enumerate(cn):
            gd = node_map.get(int(nid))
            if gd is not None and gd in g2l:
                out[k] = float(nv[g2l[gd]])
            else:
                xy = mesh.nodes_x_y_pos[int(nid)]
                out[k] = float(level_set(xy))
        return out
    # fallback: point eval
    return np.array([float(level_set(mesh.nodes_x_y_pos[int(nid)])) for nid in cn], float)

def clip_cell_to_side(mesh, eid: int, level_set, side: str = '+', eps: float = None):
    """
    Polygon(s) of the portion of element `eid` on the requested side of φ.
    Uses Pⁿ edge roots and a robust local→reference edge map (no interior diagonal).
    """
    if eps is None:
        eps = getattr(SIDE, "tol", 1e-12)

    elem = mesh.elements_list[int(eid)]
    cn   = list(elem.corner_nodes)
    V    = mesh.nodes_x_y_pos[cn].astype(float)                 # (m,2)
    vphi = _corner_phi_fast(level_set, mesh, cn)

    def inside(phi):
        return SIDE.is_pos(phi, tol=eps) if side == '+' else (SIDE.is_neg(phi, tol=eps))

    m = V.shape[0]
    # local corner order as given by the mesh:
    local_pairs = [(i, (i+1) % m) for i in range(m)] if m == 4 else [(0,1),(1,2),(2,0)]

    # We iterate edges in the mesh's local order, which matches the parent's edge ids.
    # For quads: i=0..3 is exactly the parent local edge id (0: bottom, 1: right, 2: top, 3: left).
    # For tris:  i=0..2 is the parent local edge id (0:(0,1), 1:(1,2), 2:(2,0)).

   
    poly = []
    for (i, j) in local_pairs:
        Pi, Pj = V[i], V[j]
        fi, fj = vphi[i], vphi[j]
        Ii, Ij = inside(fi), inside(fj)

        # keep corner i if it lies on requested side
        if Ii:
            _append_if_new(poly, Pi, tol=eps)

    
        # Robust approach: just ask for roots on THIS local edge 'i'.
        roots = edge_root_pn(level_set, mesh, int(eid), int(i), tol=eps)
        if roots:
            for P in roots:
                _append_if_new(poly, P, tol=eps)

    # close / validate
    if poly and np.linalg.norm(poly[0] - poly[-1]) <= 1e-14:
        poly = poly[:-1]
    if len(poly) < 3:
        return []

    # make CCW for stable fan triangulation
    A = 0.5 * sum(poly[k][0]*poly[(k+1)%len(poly)][1] - poly[(k+1)%len(poly)][0]*poly[k][1]
                  for k in range(len(poly)))
    if A < 0:
        poly = poly[::-1]

    return [np.asarray(poly, float)]


def interface_arc_nodes(level_set, I0, I1, *, mesh, eid, nseg: int, project_steps: int = 3, tol: float = 1e-12):
    """
    Return nodes P[0..nseg] lying on {phi=0} that approximate the curved interface
    between the edge intersections I0 and I1.  P[0]=I0, P[-1]=I1.
    """
    from pycutfem.integration.quadrature import _project_to_levelset
    I0 = np.asarray(I0, float); I1 = np.asarray(I1, float)
    T = np.linspace(0.0, 1.0, int(max(1, nseg)) + 1)
    P = (1.0 - T)[:, None] * I0[None, :] + T[:, None] * I1[None, :]
    # Newton‑project to phi=0
    for k in range(P.shape[0]):
        P[k] = _project_to_levelset(P[k], level_set, mesh=mesh, eid=eid,
                                    max_steps=project_steps, tol=tol)
    # drop accidental duplicates (grazing)
    out = [P[0]]
    for k in range(1, len(P)):
        if np.linalg.norm(P[k] - out[-1]) > 1e-14:
            out.append(P[k])
    return np.array(out, float)
def curved_subcell_quadrature_for_cut_triangle(mesh, eid, tri_local_ids, corner_ids, level_set,
                                               *, side='+', qvol: int = 3, nseg_hint: int = None, tol=1e-12):
    """
    Given ONE corner-triangle of a parent cell, return (qpts, qwts) for the kept side
    using curved wedge(s). Falls back to empty / full straight triangle otherwise.
    """
    from pycutfem.integration.quadrature import curved_wedge_quadrature, curved_quad_ruled_quadrature
    Vids = [corner_ids[i] for i in tri_local_ids]
    V    = mesh.nodes_x_y_pos[Vids].astype(float)      # (3,2)

    # phi at the three triangle vertices – prefer FE fast path when available
    vphi = np.array([phi_eval(level_set, V[k], eid=eid, mesh=mesh) for k in range(3)], float)

    keep = [i for i in range(3) if (SIDE.is_pos(vphi[i], tol=tol) if side=='+' else not SIDE.is_pos(vphi[i], tol=tol))]
    drop = [i for i in range(3) if i not in keep]

    if len(keep) == 3:
        # Full triangle – use your existing straight mapping
        from pycutfem.integration.quadrature import tri_rule
        qp_ref, qw_ref = tri_rule(qvol)
        qx, qw = map_ref_tri_to_phys(V[0], V[1], V[2], qp_ref, qw_ref)
        return qx, qw

    if len(keep) == 0:
        return np.empty((0,2)), np.empty((0,))

    # --- Compute edge intersections I1,I2 on the two edges touching the dropped vertex
    def _edge_point(a_loc, b_loc):
        """
        Return the *physical* intersection point on the edge (a_loc,b_loc).
        On parent boundary edges we use Pⁿ edge_root_pn (which returns point(s));
        on the interior diagonal we fall back to a linear zero crossing.
        """
        ai = int(tri_local_ids[a_loc]); bi = int(tri_local_ids[b_loc])
        if mesh.element_type == 'quad' and {ai, bi} in ({0,1},{1,2},{2,3},{0,3}):
            # map the pair to the parent local edge id expected by edge_root_pn
            pairs = {(0,1):0, (1,2):1, (2,3):2, (0,3):3, (3,0):3}
            l_edge = pairs[tuple(sorted((ai, bi)))]
            roots = edge_root_pn(level_set, mesh, int(eid), int(l_edge), tol=tol)
            if roots:                       # [P] or [P0,P1]
                return np.asarray(roots[0], float)
        # interior diagonal (or no root found) → linear fallback on the local edge
        A = V[a_loc]; B = V[b_loc]
        return segment_zero_crossing(A, B, vphi[a_loc], vphi[b_loc])

    p_order = getattr(level_set.dh.mixed_element, "_field_orders", {}).get(level_set.field, 1) if hasattr(level_set, "dh") else 1
    nseg    = (nseg_hint if nseg_hint is not None else max(2, int(p_order)+1))

    if len(keep) == 1:
        K      = V[keep[0]]
        I1     = _edge_point(keep[0], drop[0])
        I2     = _edge_point(keep[0], drop[1])
        arc = interface_arc_nodes(level_set, I1, I2, mesh=mesh, eid=eid,
                          nseg=nseg, project_steps=3, tol=tol)
        return curved_wedge_quadrature(level_set, K, arc, order_t=qvol, order_tau=qvol)

    # len(keep) == 2
    K1, K2 = V[keep[0]], V[keep[1]]
    I1     = _edge_point(keep[0], drop[0])  # on K1–D
    I2     = _edge_point(keep[1], drop[0])  # on K2–D
    arc    = interface_arc_nodes(level_set, I1, I2, mesh=mesh, eid=eid,
                                  nseg=nseg, project_steps=3, tol=tol)
    return curved_quad_ruled_quadrature(K1, K2, arc, order_lambda=qvol, order_mu=qvol)