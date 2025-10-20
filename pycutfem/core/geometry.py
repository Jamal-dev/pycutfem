import numpy as np
from pycutfem.ufl.helpers_geom import (
    phi_eval, clip_triangle_to_side, fan_triangulate, corner_tris
)

def _triangle_area(A, B, C):
    A = np.asarray(A); B = np.asarray(B); C = np.asarray(C)
    return 0.5 * abs(np.linalg.det(np.column_stack((B - A, C - A))))

def hansbo_cut_ratio(mesh, level_set, side: str = "+", tol: float = 1e-12) -> np.ndarray:
    """
    theta_e(side) = |K_e ∩ {phi ▷ 0}| / |K_e|  with ▷ determined by the global side
    convention (SIDE): for `side='+'` use SIDE.is_pos(phi,tol), for `side='-'`
    use SIDE.is_neg(phi,tol). Returns theta ∈ [0,1] for every element e.

    Parameters
    ----------
    mesh : Mesh
    level_set : callable
        φ(x) or φ(x,y)
    side : {'+','-'}
        Which half to keep when clipping.
    tol : float
        Tolerance forwarded to the side test (tie handling near φ≈0).
    """
    nE = len(mesh.elements_list)
    theta = np.zeros(nE, dtype=float)

    V = mesh.nodes_x_y_pos
    for eid in range(nE):
        elem = mesh.elements_list[eid]
        tri_local, cn = corner_tris(mesh, elem)

        # total geometric area
        area_K = 0.0
        for (i0, i1, i2) in tri_local:
            area_K += _triangle_area(V[cn[i0]], V[cn[i1]], V[cn[i2]])
        if area_K <= 0.0:
            theta[eid] = 0.0
            continue

        # kept portion (according to global side convention via clip_triangle_to_side)
        area_kept = 0.0
        for (i0, i1, i2) in tri_local:
            A, B, C = V[cn[i0]], V[cn[i1]], V[cn[i2]]
            v_phi = [phi_eval(level_set, A), phi_eval(level_set, B), phi_eval(level_set, C)]
            polys = clip_triangle_to_side([A, B, C], v_phi, side=side, eps=tol)
            for poly in polys:
                for a, b, c in fan_triangulate(poly):
                    area_kept += _triangle_area(a, b, c)

        theta[eid] = min(max(area_kept / area_K, 0.0), 1.0)

    return theta
