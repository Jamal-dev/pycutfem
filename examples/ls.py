from pycutfem.core.mesh import Mesh
from pycutfem.ufl.helpers_geom import (
    curved_subcell_quadrature_for_cut_triangle, corner_tris
)
from pycutfem.integration.quadrature import volume as vol_rule
from pycutfem.fem import transform
from pycutfem.utils.meshgen import structured_triangles, structured_quad
import numpy as np, math


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


def make_meshes(L=1.5, H=1.5, nx=24, ny=24):
    # domain centered at the origin
    offset = [-L/2, -H/2]
    nT, eT, _, cT = structured_triangles(L, H, nx_quads=nx, ny_quads=ny, poly_order=1, offset=offset)
    mesh_tri  = Mesh(nT, eT, elements_corner_nodes=cT, element_type="tri",  poly_order=1)

    nQ, eQ, _, cQ = structured_quad(L, H, nx=nx, ny=ny, poly_order=1, offset=offset)
    mesh_quad = Mesh(nQ, eQ, elements_corner_nodes=cQ, element_type="quad", poly_order=1)
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


if __name__ == "__main__":
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
