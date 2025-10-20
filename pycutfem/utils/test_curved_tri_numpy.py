
import  math, numpy as np
from dataclasses import dataclass

from pycutfem.utils.meshgen import structured_triangles

def load_structured_tri_mesh(Lx=1.0, Ly=1.0, nx_quads=16, ny_quads=16, poly_order=1):
    nodes, elements, edges, corners = structured_triangles(Lx, Ly, 
                                                                   nx_quads=nx_quads, 
                                                                   ny_quads=ny_quads, 
                                                                   poly_order=poly_order)
    pts = np.array([[n.x, n.y] for n in nodes], dtype=float)
    tris = elements.astype(int)
    return pts, tris

def tri_area(A, B, C):
    return 0.5 * abs((B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0]))

def _bisection_root_on_segment(phi, A, B, fa=None, fb=None, tol=1e-14, maxit=80):
    A = np.asarray(A, float); B = np.asarray(B, float)
    if fa is None: fa = float(phi(A))
    if fb is None: fb = float(phi(B))
    if abs(fa) < tol: return A
    if abs(fb) < tol: return B
    if fa * fb > 0.0:
        return None  # no root
    a, b = 0.0, 1.0
    for _ in range(maxit):
        m = 0.5 * (a + b)
        X = A + m * (B - A)
        f_m = float(phi(X))
        if abs(f_m) < tol or (b - a) < 1e-14:
            return X
        if fa * f_m <= 0.0:
            b, fb = m, f_m
        else:
            a, fa = m, f_m
    return A + m * (B - A)

def _gl01(order: int):
    from numpy.polynomial.legendre import leggauss
    xi, w = leggauss(order)
    lam = 0.5*(xi + 1.0)
    wl  = 0.5*w
    return lam, wl

def _project_to_levelset(point, level_set, max_steps=6, tol=1e-12):
    x = np.asarray(point, float)
    for _ in range(max_steps):
        phi = float(level_set(x))
        if abs(phi) < tol:
            break
        g = np.asarray(level_set.gradient(x), float)
        g2 = float(np.dot(g, g)) + 1e-30
        x = x - (phi / g2) * g
    return x

def interface_arc_nodes(level_set, I0, I1, nseg: int, project_steps: int = 4, tol: float = 1e-12):
    I0 = np.asarray(I0, float); I1 = np.asarray(I1, float)
    T = np.linspace(0.0, 1.0, int(max(1, nseg)) + 1)
    P = (1.0 - T)[:, None] * I0[None, :] + T[:, None] * I1[None, :]
    for k in range(P.shape[0]):
        P[k] = _project_to_levelset(P[k], level_set, max_steps=project_steps, tol=tol)
    out = [P[0]]
    for k in range(1, len(P)):
        if np.linalg.norm(P[k] - out[-1]) > 1e-14:
            out.append(P[k])
    return np.array(out, float)

def curved_wedge_quadrature(apex, arc_nodes, order_t=3, order_mu=3):
    K = np.asarray(apex, float)
    arc = np.asarray(arc_nodes, float)
    if arc.shape[0] < 2:
        return np.empty((0,2)), np.empty((0,))
    lam_t, wt = _gl01(order_t)
    mu,  wu   = _gl01(order_mu)
    qx = []; qw = []
    for a, b in zip(arc[:-1], arc[1:]):
        dEdlam = (b - a)
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

def curved_quad_ruled_quadrature(K1, K2, arc_nodes, order_lambda=3, order_mu=3):
    K1 = np.asarray(K1, float); K2 = np.asarray(K2, float)
    arc = np.asarray(arc_nodes, float)
    if arc.shape[0] < 2:
        return np.empty((0,2)), np.empty((0,))
    nseg = arc.shape[0] - 1
    lam_t, w_lam = _gl01(order_lambda)
    mu_t,  w_mu  = _gl01(order_mu)
    qx = []; qw = []
    dS_dlam = (K2 - K1) / float(nseg)
    for i in range(nseg):
        a, b = arc[i], arc[i+1]
        dE_dlam = (b - a)
        for j, lam in enumerate(lam_t):
            E = (1.0 - lam) * a + lam * b
            s = (i + lam) / float(nseg)
            S = (1.0 - s) * K1 + s * K2
            for k, mu in enumerate(mu_t):
                X = (1.0 - mu) * S + mu * E
                dX_dlam = (1.0 - mu) * dS_dlam + mu * dE_dlam
                dX_dmu  = E - S
                jac = abs(dX_dlam[0]*dX_dmu[1] - dX_dlam[1]*dX_dmu[0])
                w = w_lam[j] * w_mu[k] * jac
                qx.append(X);  qw.append(w)
    return np.array(qx), np.array(qw)

def positive_area_in_triangle(A, B, C, level_set, nseg=8, qvol=3, tol=1e-14):
    fa, fb, fc = float(level_set(A)), float(level_set(B)), float(level_set(C))
    keep = [i for i,f in enumerate((fa,fb,fc)) if f >= -tol]
    if len(keep) == 3:  return 0.5*abs(np.cross(B-A, C-A)), 1, 0  # full
    if len(keep) == 0:  return 0.0, 0, 0
    V = [np.asarray(A,float), np.asarray(B,float), np.asarray(C,float)]
    F = [fa, fb, fc]
    def hit(i, j):
        return _bisection_root_on_segment(lambda X: level_set(X), V[i], V[j], F[i], F[j], tol=tol)
    if len(keep) == 1:
        k = keep[0]
        drop = [i for i in (0,1,2) if i != k]
        I1 = hit(k, drop[0])
        I2 = hit(k, drop[1])
        arc = interface_arc_nodes(level_set, I1, I2, nseg=nseg, project_steps=6)
        qx, qw = curved_wedge_quadrature(V[k], arc, order_t=qvol, order_mu=qvol)
        return float(qw.sum()), 0, 1
    else: # len(keep) == 2
        k1, k2 = keep
        drop = [i for i in (0,1,2) if i not in keep][0]
        I1 = hit(k1, drop)
        I2 = hit(k2, drop)
        arc = interface_arc_nodes(level_set, I1, I2, nseg=nseg, project_steps=6)
        qx, qw = curved_quad_ruled_quadrature(V[k1], V[k2], arc, order_lambda=qvol, order_mu=qvol)
        return float(qw.sum()), 0, 1

def integrate_positive_area_over_mesh(pts, tris, level_set, nseg=8, qvol=3):
    total = 0.0
    n_wedge = n_quad = 0
    for tri in tris:
        i,j,k = map(int, tri[:3])
        A, B, C = pts[i], pts[j], pts[k]
        a, nw, nq = positive_area_in_triangle(A,B,C, level_set, nseg=nseg, qvol=qvol)
        total += a; n_wedge += nw; n_quad += nq
    return total, n_wedge, n_quad

class HalfPlane:
    def __init__(self, n, x0):
        self.n = np.asarray(n, float) / (np.linalg.norm(n)+1e-30)
        self.x0= np.asarray(x0, float)
    def __call__(self, x):
        x = np.asarray(x, float)
        return float(np.dot(self.n, x - self.x0))
    def gradient(self, x):
        return self.n

class CirclePhi:
    def __init__(self, c=(0.5,0.5), R=0.3):
        self.c = np.asarray(c, float)
        self.R = float(R)
    def __call__(self, x):
        x = np.asarray(x, float)
        return float(self.R*self.R - np.dot(x - self.c, x - self.c))
    def gradient(self, x):
        x = np.asarray(x, float)
        return -2.0 * (x - self.c)

def main():
    pts, tris = load_structured_tri_mesh(1.0, 1.0, nx_quads=24, ny_quads=24, poly_order=1)
    print(f"Mesh: {len(pts)} nodes, {len(tris)} triangles")
    # half-plane test
    hp = HalfPlane(n=(1.0, 1.0), x0=(0.4, 0.3))  # line: x+y=0.7
    t = 0.7
    area_exact_hp = (1.0 - 0.5*t*t) if t <= 1.0 else (0.5*(2.0 - t)**2)
    for nseg in (2, 4, 8, 12):
        area_num, nw, nq = integrate_positive_area_over_mesh(pts, tris, hp, nseg=nseg, qvol=4)
        print(f"[half-plane] nseg={nseg:2d}  area_num={area_num:.12f}  area_exact={area_exact_hp:.12f}  err={area_num - area_exact_hp:+.3e}")
    # circle test
    R = 0.33; c = (0.5,0.5)
    circ = CirclePhi(c=c, R=R)
    area_exact_circle = math.pi * R * R
    for nseg in (2, 4, 6, 8, 12, 16):
        area_num, nw, nq = integrate_positive_area_over_mesh(pts, tris, circ, nseg=nseg, qvol=4)
        rel = (area_num - area_exact_circle) / area_exact_circle
        print(f"[circle    ] nseg={nseg:2d}  area_num={area_num:.12f}  exact={area_exact_circle:.12f}  relerr={rel:+.3e}")
if __name__ == "__main__":
    main()
