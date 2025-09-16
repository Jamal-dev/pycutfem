import numpy as np

# antiderivative: ∫ sqrt(R^2 - x^2) dx
def _F(x, R):
    x = np.asarray(x, float)
    # domain clamp (we only call on |x|<=R, but be safe)
    xr = np.clip(x, -R, R)
    return 0.5*(xr*np.sqrt(np.maximum(R*R - xr*xr, 0.0)) + R*R*np.arcsin(np.clip(xr/R, -1.0, 1.0)))

def _y_up(x, R):
    return np.sqrt(np.maximum(R*R - x*x, 0.0))

def circle_rect_intersection_area_exact(x0, y0, x1, y1, R, center=(0.0, 0.0), tol=1e-14):
    """
    Exact area of the intersection between a disk {(x-cx)^2+(y-cy)^2 <= R^2}
    and the axis-aligned rectangle [x0,x1] × [y0,y1].

    Returns: area_inside  (i.e., "neg" if φ = sqrt(x^2+y^2)-R)
    """
    cx, cy = center

    # translate so circle center -> (0,0)
    x0p, x1p = x0 - cx, x1 - cx
    y0p, y1p = y0 - cy, y1 - cy
    if x0p > x1p: x0p, x1p = x1p, x0p
    if y0p > y1p: y0p, y1p = y1p, y0p

    # quick rejects
    if x1p <= -R or x0p >= R:
        return 0.0
    # clamp x to circle's vertical span
    xa = max(x0p, -R)
    xb = min(x1p,  R)
    if xb - xa <= 0.0:
        return 0.0

    # critical xs where the clamp min/max can change:
    xs = [xa, xb]
    for yb in (y0p, y1p, -y0p, -y1p):
        if abs(yb) < R:
            xs.extend([ -np.sqrt(R*R - yb*yb),  np.sqrt(R*R - yb*yb) ])
    xs = np.array(sorted([t for t in xs if xa - tol <= t <= xb + tol]), float)

    # unique & bounded
    xs2 = [xs[0]]
    for t in xs[1:]:
        if abs(t - xs2[-1]) > 1e-12:
            xs2.append(t)
    xs = np.array(xs2, float)

    area = 0.0
    for a, b in zip(xs[:-1], xs[1:]):
        if b - a <= 0: 
            continue
        xm = 0.5*(a + b)
        yup = _y_up(xm, R)
        ylow = -yup

        lower = max(y0p, ylow)
        upper = min(y1p, yup)

        if upper <= lower:
            continue  # zero strip

        # classify the case at midpoint; thanks to the breakpoints, it won't change over (a,b)
        eps = 0.0
        case_upper_from_circle = abs(upper - yup)  < 1e-12
        case_lower_from_circle = abs(lower - ylow) < 1e-12

        if case_upper_from_circle and case_lower_from_circle:
            # L(x) = 2*sqrt(R^2 - x^2)
            area += 2.0 * (_F(b, R) - _F(a, R))
        elif case_upper_from_circle and not case_lower_from_circle:
            # L(x) = sqrt(R^2 - x^2) - y0p
            area += (_F(b, R) - _F(a, R)) - y0p*(b - a)
        elif (not case_upper_from_circle) and case_lower_from_circle:
            # L(x) = y1p + sqrt(R^2 - x^2)
            area += y1p*(b - a) + (_F(b, R) - _F(a, R))
        else:
            # fully covered in y: L(x) = y1p - y0p
            area += (y1p - y0p) * (b - a)

    # numerical safety
    return float(max(area, 0.0))


def per_element_circle_split_structured_quad(mesh, R, center=(0.0,0.0)):
    """
    For each quad element, compute:
      Aneg = |element ∩ disk|,  Apos = |element| - Aneg
    Returns: list of dicts [{eid, x0,y0,x1,y1, area, Aneg, Apos}, ...]
    """
    out = []
    nodes = np.asarray([[n.x, n.y] for n in mesh.nodes_list], float)
    for e in mesh.elements_list:
        cn = list(e.corner_nodes)   # [bl, br, tr, tl] in your generator
        xy = nodes[cn]
        x0, y0 = xy[:,0].min(), xy[:,1].min()
        x1, y1 = xy[:,0].max(), xy[:,1].max()
        area_e = (x1 - x0) * (y1 - y0)

        Aneg = circle_rect_intersection_area_exact(x0, y0, x1, y1, R, center=center)
        Aneg = min(max(Aneg, 0.0), area_e)  # clamp numerically
        Apos = area_e - Aneg
        out.append(dict(eid=int(e.id), x0=x0, y0=y0, x1=x1, y1=y1,
                        area=area_e, Aneg=Aneg, Apos=Apos))
    return out

# --- add in area_checks.py ----------------------------------------------------
def _G_line_integral(a, b, m, c):
    """∫_{x=a}^{b} (m x + c) dx"""
    return 0.5*m*(b*b - a*a) + c*(b - a)

def circle_rect_halfplane_intersection_area_exact(x0, y0, x1, y1,
                                                  m, c, keep='below',
                                                  R=1.0, center=(0.0, 0.0),
                                                  tol=1e-14):
    """
    Exact area of (disk) ∩ (rectangle [x0,x1]×[y0,y1]) ∩ { y ≤ m x + c }   if keep='below'
                                             or ∩ { y ≥ m x + c }         if keep='above'
    Returns area INSIDE the disk (i.e. “neg” if φ = sqrt((x-cx)^2+(y-cy)^2) - R).
    """
    import math
    cx, cy = center

    # clip to where circle has vertical extent and to the rectangle in x
    xa = max(min(x0, x1), cx - R)
    xb = min(max(x0, x1), cx + R)
    if xb - xa <= 0.0:
        return 0.0

    # collect breakpoints where min/max identities can change
    xs = [xa, xb]

    # circle ∩ horizontals y=y0,y1
    for Y in (y0, y1):
        d = R*R - (Y - cy)*(Y - cy)
        if d >= -tol:
            d = max(d, 0.0)
            rd = math.sqrt(d)
            xs.extend([cx - rd, cx + rd])

    # circle ∩ diagonal y = m x + c  → (x - cx)^2 + (m x + c - cy)^2 = R^2
    A = 1.0 + m*m
    B = 2.0*(m*(c - cy) - cx)
    C = cx*cx + (c - cy)*(c - cy) - R*R
    disc = B*B - 4*A*C
    if disc >= -tol:
        disc = max(disc, 0.0)
        s = math.sqrt(disc)
        xs.extend([(-B - s)/(2*A), (-B + s)/(2*A)])

    # keep only within [xa,xb], sorted & unique
    xs = sorted(t for t in xs if xa - 1e-13 <= t <= xb + 1e-13)
    xs2 = [xs[0]]
    for t in xs[1:]:
        if abs(t - xs2[-1]) > 1e-12:
            xs2.append(t)
    xs = xs2

    # helpers
    def y_up(x):    # circle upper
        return _y_up(x - cx, R) + cy
    def y_low(x):   # circle lower
        return -_y_up(x - cx, R) + cy
    def y_diag(x):
        return m*x + c

    area = 0.0
    for a, b in zip(xs[:-1], xs[1:]):
        if b - a <= 0.0:
            continue
        xm = 0.5*(a + b)

        # triangle’s permitted vertical slab at x
        if keep == 'below':
            yL_tri, yU_tri = y0, y_diag(xm)
        else: # 'above'
            yL_tri, yU_tri = y_diag(xm), y1

        # clip with the circle
        U_is_circ = (y_up(xm) <= yU_tri + 1e-12)
        L_is_circ = (y_low(xm) >= yL_tri - 1e-12)

        if keep == 'below':
            # U in {circ, diag}, L in {low, y0}
            if U_is_circ and L_is_circ:
                # yup - ylow
                area += 2.0 * (_F(b - cx, R) - _F(a - cx, R))
            elif U_is_circ and not L_is_circ:
                # yup - y0
                area += (_F(b - cx, R) - _F(a - cx, R)) - y0*(b - a)
            elif (not U_is_circ) and L_is_circ:
                # ydiag - ylow
                area += _G_line_integral(a, b, m, c) + (_F(b - cx, R) - _F(a - cx, R))
            else:
                # ydiag - y0
                area += _G_line_integral(a, b, m, c) - y0*(b - a)
        else:
            # keep == 'above' → U in {circ, y1}, L in {low, diag}
            if U_is_circ and L_is_circ:
                # yup - ylow
                area += 2.0 * (_F(b - cx, R) - _F(a - cx, R))
            elif U_is_circ and not L_is_circ:
                # yup - ydiag
                area += (_F(b - cx, R) - _F(a - cx, R)) - _G_line_integral(a, b, m, c)
            elif (not U_is_circ) and L_is_circ:
                # y1 - ylow
                area += y1*(b - a) + (_F(b - cx, R) - _F(a - cx, R))
            else:
                # y1 - ydiag
                area += y1*(b - a) - _G_line_integral(a, b, m, c)

    return float(max(area, 0.0))


def per_element_circle_split_structured_triangle(mesh, R, center=(0.0, 0.0)):
    """
    For each *triangular* element (from structured_triangles), compute:
      Aneg = |element ∩ disk|,  Apos = |element| - Aneg
    Returns: list of dicts [{eid, x0,y0,x1,y1, area, Aneg, Apos}, ...]
    """
    out = []
    nodes = np.asarray([[n.x, n.y] for n in mesh.nodes_list], float)

    for e in mesh.elements_list:
        cn = list(e.corner_nodes)   # 3 ids
        xy = nodes[cn]              # (3,2)
        # bounding rectangle and its BL->TR diagonal
        x0, y0 = xy[:,0].min(), xy[:,1].min()
        x1, y1 = xy[:,0].max(), xy[:,1].max()
        if abs(x1 - x0) <= 1e-15 or abs(y1 - y0) <= 1e-15:
            continue
        m = (y1 - y0) / (x1 - x0)
        c = y0 - m * x0

        # which half (below/above the diagonal) – use element centroid
        cx_e, cy_e = float(e.centroid_x), float(e.centroid_y) if hasattr(e, 'centroid_y') else float(np.mean(xy, axis=0)[1])
        keep = 'below' if cy_e <= (m*cx_e + c) else 'above'

        area_tri = 0.5 * (x1 - x0) * (y1 - y0)
        Aneg = circle_rect_halfplane_intersection_area_exact(x0, y0, x1, y1, m, c, keep,
                                                             R=R, center=center)
        Aneg = min(max(Aneg, 0.0), area_tri)
        Apos = area_tri - Aneg
        out.append(dict(eid=int(e.id), x0=x0, y0=y0, x1=x1, y1=y1,
                        area=area_tri, Aneg=Aneg, Apos=Apos))
    return out
