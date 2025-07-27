import numpy as np

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
