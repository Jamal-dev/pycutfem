from pycutfem.core.mesh import Mesh
from pycutfem.ufl.helpers_geom import (
    curved_subcell_quadrature_for_cut_triangle, corner_tris
)
from pycutfem.integration.quadrature import volume as vol_rule
from pycutfem.fem import transform
from pycutfem.utils.meshgen import structured_triangles, structured_quad
import numpy as np, math
# ============================================================================
# Additional tests: line_quadrature accuracy and interface length convergence
# ============================================================================
from pycutfem.integration.quadrature import line_quadrature, curved_line_quadrature
try:
    # Use batched isoparametric interface rule for accuracy with higher p
    from pycutfem.integration.quadrature import isoparam_interface_line_quadrature_batch as _iso_batch
    _HAVE_ISO_LINE = True
except Exception:
    from pycutfem.integration.quadrature import curved_line_quadrature_batch as _iso_batch
    _HAVE_ISO_LINE = False

class LevelSetFunction:
    """Abstract base class"""
    def __call__(self, x: np.ndarray) -> float:
        raise NotImplementedError
    def gradient(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def evaluate_on_nodes(self, mesh) -> np.ndarray:
        return np.apply_along_axis(self, 1, mesh.nodes_x_y_pos)

class CircleLevelSet(LevelSetFunction):
    def __init__(self, center=(0.,0.), radius: float=1.0):
        self.center=np.asarray(center,dtype=float)
        self.radius=float(radius)
    def __call__(self, x):
        """Signed distance; works for shape (..., 2) or plain (2,)."""
        x = np.asarray(x, dtype=float)
        rel = x - self.center
        # norm along the last axis keeps the leading shape intact
        return np.linalg.norm(rel, axis=-1) - self.radius
    def gradient(self, x):
        d=np.asarray(x-self.center)
        nrm=np.linalg.norm(d)
        return d/nrm if nrm else np.zeros_like(d)
    def evaluate_on_nodes(self, mesh):
        # Fully vectorized evaluation on all mesh nodes
        return self(mesh.nodes_x_y_pos)


def make_meshes(L=1.5, H=1.5, nx=24, ny=24):
    # domain centered at the origin
    offset = [-L/2, -H/2]
    nT, eT, edgeT, cT = structured_triangles(L, H, nx_quads=nx, ny_quads=ny, poly_order=1, offset=offset)
    mesh_tri  = Mesh(nT, eT, edges_connectivity=edgeT, elements_corner_nodes=cT, element_type="tri",  poly_order=1)

    nQ, eQ, edgeQ, cQ = structured_quad(L, H, nx=nx, ny=ny, poly_order=1, offset=offset)
    mesh_quad = Mesh(nQ, eQ, edges_connectivity=edgeQ, elements_corner_nodes=cQ, element_type="quad", poly_order=1)
    return mesh_tri, mesh_quad


def compute_A_side_like_compiler(mesh: Mesh, level_set, *, side='+', qvol=4, nseg_hint=None, tol=1e-12):
    """
    Replicates the structure in compilers.py:
      1) classify; 2) integrate full elements with volume() rule; 3) only dissect cut elements.
    For φ from CircleLS, side='+' means integrate {φ ≥ 0} (inside the circle).
    """
    # --- 1) classification (same entry point compilers.py uses)
    inside_ids, outside_ids, cut_ids = mesh.classify_elements(level_set, tol=tol)

    # compilers.py integrates "full_eids = outside_ids if side=='+' else inside_ids"
    # (because '+' keeps φ≥0). This matches our analytic CircleLS (φ≥0 inside).
    full_eids = outside_ids if side == '+' else inside_ids

    # --- 2) full elements: standard parent-element quadrature (no curved rule)
    qp_ref, qw_ref = vol_rule(mesh.element_type, qvol)
    A = 0.0
    for eid in full_eids:
        for (xi, eta), w in zip(qp_ref, qw_ref):
            detJ = abs(np.linalg.det(transform.jacobian(mesh, int(eid), (float(xi), float(eta)))))
            A += w * detJ

    # --- 3) cut elements: corner-tri loop with curved subcell quadrature
    for eid in cut_ids:
        elem = mesh.elements_list[int(eid)]
        tri_list, corner_ids = corner_tris(mesh, elem)  # triangles that partition the parent
        for loc_tri in tri_list:
            qx, qw = curved_subcell_quadrature_for_cut_triangle(
                mesh, int(eid), loc_tri, list(corner_ids),
                level_set, side=side, qvol=qvol, nseg_hint=nseg_hint, tol=tol
            )
            # weights are already physical; just sum to get area contribution
            A += float(qw.sum())
    return float(A)

def _integrate_on_segment(f, p0, p1, order):
    """Integrate vectorized f(x) along the straight segment p0->p1 with Gauss order."""
    pts, wts = line_quadrature(np.asarray(p0, float), np.asarray(p1, float), order=int(order))
    vals = f(pts)  # expects shape (N,) or (N,1)
    return float(np.sum(vals.reshape(-1) * wts.reshape(-1)))


def _poly_field(deg):
    """Return f(x,y) = x^deg as a vectorized callable over Nx2 arrays."""
    def f(P):
        x = np.asarray(P)[:, 0]
        return x**deg
    return f
def _circle_perimeter(R):
    return 2.0 * math.pi * R

def test_interface_length_convergence_vectorized(quad_order=8):
    """
    Fully vectorized pipeline with accurate timing and minimal recomputation:
    - classify + build endpoints once on Q1 mesh using Mesh.build_interface_segments
    - reuse those endpoints for Q2/Q3 (per user's guidance)
    - run isoparametric batch line quadrature per p
    """
    import time
    L, H = 2.0, 2.0
    R = 2.0/3.0
    nx = ny = 96
    LS = CircleLevelSet(center=(0.0, 0.0), radius=R)

    print("\n== Vectorized pipeline timing ==")

    # 1) Build Q1 mesh and classify once
    t0_all = time.perf_counter()
    nQ1, eQ1, edgeQ1, cQ1 = structured_quad(L, H, nx=nx, ny=ny, poly_order=1, offset=[-L/2, -H/2])
    mesh_q1 = Mesh(nQ1, eQ1, edges_connectivity=edgeQ1, elements_corner_nodes=cQ1, element_type="quad", poly_order=1)
    t0 = time.perf_counter()
    inside_ids, outside_ids, cut_ids = mesh_q1.classify_elements(LS)
    t1 = time.perf_counter()

    # 2) Build endpoints on Q1 only and reuse for all p
    t2 = time.perf_counter()
    mesh_q1.build_interface_segments(LS, tol=1e-12)
    # Extract P0,P1 for the cut elements
    P0_list, P1_list, eids_list = [], [], []
    for eid in cut_ids:
        pts = getattr(mesh_q1.elements_list[int(eid)], 'interface_pts', [])
        if len(pts) >= 2:
            P0_list.append(np.asarray(pts[0], float))
            P1_list.append(np.asarray(pts[1], float))
            eids_list.append(int(eid))
    P0 = np.asarray(P0_list, float)
    P1 = np.asarray(P1_list, float)
    eids = np.asarray(eids_list, int)
    t3 = time.perf_counter()

    print(f"precompute: classify={(t1-t0):.3f} s, endpoints={(t3-t2):.3f} s, cuts={P0.shape[0]}")

    # 3) Run Qp quadrature on p = 1,2,3 reusing P0,P1,eids
    for p in [1, 2, 3]:
        t_mesh0 = time.perf_counter()
        nQ, eQ, edgeQ, cQ = structured_quad(L, H, nx=nx, ny=ny, poly_order=p, offset=[-L/2, -H/2])
        t_mesh1 = time.perf_counter()
        mesh = Mesh(nQ, eQ, edges_connectivity=edgeQ, elements_corner_nodes=cQ, element_type="quad", poly_order=p)
        t_mesh2 = time.perf_counter()
        t_quad0 = time.perf_counter()
        if _HAVE_ISO_LINE:
            qpts, qw = _iso_batch(LS, P0, P1, p=mesh.poly_order, order=quad_order, project_steps=2, tol=1e-12,
                                   mesh=mesh, eids=eids)
        else:
            qpts, qw = _iso_batch(LS, P0, P1, order=quad_order, nseg=3, project_steps=2, tol=1e-12,
                                   mesh=mesh, eids=eids)
        length = float(np.sum(qw))
        t_quad1 = time.perf_counter()

        exact = _circle_perimeter(R)
        rel = abs(length - exact) / exact
        print(f"Q{p}: cuts={P0.shape[0]:4d} |Γ|={length:.10f} rel.err={rel:.3e}  "
              f"[mesh-gen {(t_mesh1-t_mesh0):.3f} s, mesh-build {(t_mesh2-t_mesh1):.3f} s, line-quad {(t_quad1-t_quad0):.3f} s]")

    print(f"total vectorized section: {(time.perf_counter()-t0_all)} s")


if __name__ == "__main__":
    import time
    _script_t0 = time.perf_counter()
    # geometry & level set
    L, H = 2.0, 2.0
    maxh = 0.125
    R = 2.0/3.0
    nx = int(L / maxh)
    ny = int(H / maxh)
    mesh_tri, mesh_quad = make_meshes(L=L, H=H, nx=nx, ny=ny)
    phi = CircleLevelSet(center=(0.0, 0.0), radius=R)
    Acircle = math.pi * R**2
    A_exact_inside = Acircle
    A_exact_outside = L*H - Acircle
    def rel_err(A, A_exact):
        return abs(A - A_exact) / A_exact

    print("== TRI mesh (compiler-style cut assembly) ==")
    for nseg in (3, 5, 7, 9, 13, 17):
        A_pos = compute_A_side_like_compiler(mesh_tri, phi, side='+', qvol=4, nseg_hint=nseg)
        A_neg = compute_A_side_like_compiler(mesh_tri, phi, side='-', qvol=4, nseg_hint=nseg)
        print(f"[tri ] nseg={nseg:2d}  A+={A_pos:.8f}  rel.err(A+)={rel_err(A_pos,A_exact_outside):.3e}, A-={A_neg:.8f}  rel.err(A-)={rel_err(A_neg,A_exact_inside):.3e}")

    print("\n== QUAD mesh (compiler-style cut assembly) ==")
    for nseg in (3, 5, 7, 9, 13, 17):
        A_pos = compute_A_side_like_compiler(mesh_quad, phi, side='+', qvol=4, nseg_hint=nseg)
        A_neg = compute_A_side_like_compiler(mesh_quad, phi, side='-', qvol=4, nseg_hint=nseg)
        print(f"[quad] nseg={nseg:2d}  A+={A_pos:.8f}  rel.err(A+)={rel_err(A_pos,A_exact_outside):.3e}, A-={A_neg:.8f}  rel.err(A-)={rel_err(A_neg,A_exact_inside):.3e}")

    test_interface_length_convergence_vectorized()

    print(f"\nScript total time: {(time.perf_counter()-_script_t0):.3f} s")


    
