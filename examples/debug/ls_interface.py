from pycutfem.core.mesh import Mesh
from pycutfem.utils.meshgen import structured_triangles, structured_quad
from pycutfem.integration.quadrature import curved_line_quadrature
import numpy as np, math

# --- tiny LS (same spirit as in ls.py) ------------------------------------
class LevelSetFunction:
    def __call__(self, x: np.ndarray) -> float: raise NotImplementedError
    def gradient(self, x: np.ndarray) -> np.ndarray: raise NotImplementedError
    def evaluate_on_nodes(self, mesh) -> np.ndarray:
        return np.apply_along_axis(self, 1, mesh.nodes_x_y_pos)

class CircleLevelSet(LevelSetFunction):
    def __init__(self, center=(0.,0.), radius: float=1.0):
        self.center=np.asarray(center,dtype=float)
        self.radius=float(radius)
    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        rel = x - self.center
        return np.linalg.norm(rel, axis=-1) - self.radius
    def gradient(self, x):
        d=np.asarray(x-self.center)
        nrm=np.linalg.norm(d)
        return d/nrm if nrm else np.zeros_like(d)

def make_meshes(L=1.5, H=1.5, nx=24, ny=24):
    offset = [-L/2, -H/2]
    nT, eT, _, cT = structured_triangles(L, H, nx_quads=nx, ny_quads=ny, poly_order=1, offset=offset)
    mesh_tri  = Mesh(nT, eT, elements_corner_nodes=cT, element_type="tri",  poly_order=1)
    nQ, eQ, _, cQ = structured_quad(L, H, nx=nx, ny=ny, poly_order=1, offset=offset)
    mesh_quad = Mesh(nQ, eQ, elements_corner_nodes=cQ, element_type="quad", poly_order=1)
    return mesh_tri, mesh_quad

def compute_interface_length_like_compiler(mesh, level_set, qline=4, nseg=7, tol=1e-12):
    """
    Classify elements, ensure interface segments exist, then on each CUT element
    integrate along φ=0 using a *curved* line rule and sum physical weights.
    """
    # make sure segments exist (elements get elem.interface_pts)
    mesh.classify_elements(level_set, tol=tol)
    mesh.classify_edges(level_set, tol=tol)
    mesh.build_interface_segments(level_set)  # gives elem.interface_pts

    L = 0.0
    for elem in mesh.elements_list:
        if getattr(elem, "tag", None) != "cut": 
            continue
        pts = getattr(elem, "interface_pts", None)
        if not (isinstance(pts, (list, tuple)) and len(pts) == 2):
            continue
        p0, p1 = pts
        qx, qw = curved_line_quadrature(level_set, p0, p1, order=qline, nseg=nseg, project_steps=3, tol=tol)
        L += float(qw.sum())   # weights are already physical arc lengths
    return float(L)

if __name__ == "__main__":
    # geometry & level set (same as ls.py style)
    L, H = 2.0, 2.0
    maxh = 0.125
    R = 2.0/3.0
    nx = int(L / maxh)
    ny = int(H / maxh)
    mesh_tri, mesh_quad = make_meshes(L=L, H=H, nx=nx, ny=ny)
    phi = CircleLevelSet(center=(0.0, 0.0), radius=R)
    L_exact = 2.0 * math.pi * R

    def rel_err(val, exact): return abs(val - exact) / exact

    print("== TRI mesh (compiler-style interface length) ==")
    for nseg in (3, 5, 7, 9, 13, 17):
        Ltri = compute_interface_length_like_compiler(mesh_tri, phi, qline=4, nseg=nseg)
        print(f"[tri ] nseg={nseg:2d}  L={Ltri:.8f}  rel.err={rel_err(Ltri, L_exact):.3e}")

    print("\n== QUAD mesh (compiler-style interface length) ==")
    for nseg in (3, 5, 7, 9, 13, 17):
        Lq = compute_interface_length_like_compiler(mesh_quad, phi, qline=4, nseg=nseg)
        print(f"[quad] nseg={nseg:2d}  L={Lq:.8f}  rel.err={rel_err(Lq, L_exact):.3e}")
