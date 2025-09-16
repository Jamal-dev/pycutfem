from pycutfem.core.mesh import Mesh
from pycutfem.ufl.helpers_geom import (
    curved_subcell_quadrature_for_cut_triangle, corner_tris
)
from pycutfem.integration.quadrature import volume as vol_rule
from pycutfem.fem import transform
from pycutfem.utils.meshgen import structured_triangles, structured_quad
import numpy as np, math


class CircleLS:
    """Analytic φ; we add evaluate_on_nodes so classify_elements can use it."""
    def __init__(self, R):
        self.R = float(R)
    def __call__(self, x):
        x = np.asarray(x, float)
        return self.R**2 - (x[0]**2 + x[1]**2)     # φ>=0 = inside the disk
    def gradient(self, x):
        x = np.asarray(x, float)
        return np.array([-2*x[0], -2*x[1]], float)
    def evaluate_on_nodes(self, mesh):
        XY = mesh.nodes_x_y_pos
        return self.R**2 - (XY[:, 0]**2 + XY[:, 1]**2)


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
    mesh_tri, mesh_quad = make_meshes()
    phi = CircleLS(R=2.0/3.0)
    Atrue = math.pi * (2.0/3.0)**2

    print("== TRI mesh (compiler-style cut assembly) ==")
    for nseg in (3, 5, 7, 9, 13, 17):
        A = compute_A_side_like_compiler(mesh_tri, phi, side='+', qvol=4, nseg_hint=nseg)
        print(f"[tri ] nseg={nseg:2d}  A+={A:.8f}  rel.err={abs(A-Atrue)/Atrue:.3e}")

    print("\n== QUAD mesh (compiler-style cut assembly) ==")
    for nseg in (3, 5, 7, 9, 13, 17):
        A = compute_A_side_like_compiler(mesh_quad, phi, side='+', qvol=4, nseg_hint=nseg)
        print(f"[quad] nseg={nseg:2d}  A+={A:.8f}  rel.err={abs(A-Atrue)/Atrue:.3e}")
