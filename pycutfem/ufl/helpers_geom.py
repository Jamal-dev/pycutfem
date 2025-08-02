import numpy as np
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

    keep = [i for i in range(3) if phi[i] >= -eps]
    drop = [i for i in range(3) if phi[i] <  -eps]

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
        I1 = segment_zero_crossing(V[iK1], V[iD], phi[iK1], phi[iD])
        I2 = segment_zero_crossing(V[iK2], V[iD], phi[iK2], phi[iD])
        return [[V[iK1], V[iK2], I2, I1]]

# -------------------- JIT versions of the geometric helpers ------------------
if _HAVE_NUMBA:
    @_nb.njit(cache=True, fastmath=True)
    def _clip_triangle_to_side_numba(V, phi, sgn, eps=0.0):
        """
        V: (3,2) float64, phi: (3,) float64, sgn=+1 for '+' side, -1 for '-' side.
        Returns (poly_pts(4,2), n_pts:int). For triangle result n_pts=3; quad → 4; empty → 0.
        """
        p = _nb.types.float64
        phi0 = sgn * phi[0]; phi1 = sgn * phi[1]; phi2 = sgn * phi[2]
        keep0 = phi0 >= -eps; keep1 = phi1 >= -eps; keep2 = phi2 >= -eps
        n_keep = (1 if keep0 else 0) + (1 if keep1 else 0) + (1 if keep2 else 0)
        out = np.empty((4, 2), dtype=np.float64)
        if n_keep == 3:
            out[0] = V[0]; out[1] = V[1]; out[2] = V[2]
            return out, 3
        if n_keep == 0:
            return out, 0

        # linear interpolation on an edge
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
            # one vertex inside → triangle
            if keep0:
                K = 0; D1 = 1; D2 = 2; phK = phi0; phD1 = phi1; phD2 = phi2
            elif keep1:
                K = 1; D1 = 0; D2 = 2; phK = phi1; phD1 = phi0; phD2 = phi2
            else:
                K = 2; D1 = 0; D2 = 1; phK = phi2; phD1 = phi0; phD2 = phi1
            I1 = _seg_inter(V[K], V[D1], phK, phD1)
            I2 = _seg_inter(V[K], V[D2], phK, phD2)
            out[0] = V[K]; out[1] = I1; out[2] = I2
            return out, 3
        # n_keep == 2 → quad
        if not keep0:
            K1=1; K2=2; D=0; phK1=phi1; phK2=phi2; phD=phi0
        elif not keep1:
            K1=0; K2=2; D=1; phK1=phi0; phK2=phi2; phD=phi1
        else:
            K1=0; K2=1; D=2; phK1=phi0; phK2=phi1; phD=phi2
        I1 = _seg_inter(V[K1], V[D], phK1, phD)
        I2 = _seg_inter(V[K2], V[D], phK2, phD)
        out[0] = V[K1]; out[1] = V[K2]; out[2] = I2; out[3] = I1
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
    else:  # 'tri'
        return [ (0,1,2) ], cn
