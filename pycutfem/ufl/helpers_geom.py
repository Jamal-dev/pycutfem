import numpy as np
from pycutfem.core.sideconvention import SIDE

try:
    import numba as _nb
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

# -------- Level-set evaluation that accepts both styles: φ(x) and φ(x,y) -----
def phi_eval(level_set, x):
    """Evaluate φ on a 2-vector x robustly."""
    try:
        return level_set(np.asarray(x))
    except TypeError:
        return level_set(x[0], x[1])

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
    sgn = +1.0 if side == '+' else -1.0
    phi = sgn * np.asarray(v_phi, dtype=float)
    V   = [np.asarray(v, dtype=float) for v in v_coords]

    # Decide membership using the single source of truth:
    if side == '+':
        keep = [i for i in range(3) if SIDE.is_pos(phi[i], tol=eps)]
    elif side == "-":  # side == '-'
        keep = [i for i in range(3) if SIDE.is_neg(phi[i], tol=eps)]
    else:
        raise ValueError("side must be '+' or '-'")    
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
    """Given vertices A,B,C (2‑vecs) and a reference rule, return (x_phys, w_phys)."""
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
        return [(0,1,3), (1,2,3)], cn
        # → triangles (0,1,2) and (0,2,3)
        # return [(0,1,2), (0,2,3)], cn
    
    elif mesh.element_type == 'tri':
        return [ (0,1,2) ], cn
