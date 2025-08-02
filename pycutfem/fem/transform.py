"""pycutfem.fem.transform
Reference → physical mapping for linear elements.
"""
import numpy as np
from pycutfem.fem.reference import get_reference
from functools import lru_cache

def _shape_and_grad(ref, xi_eta):
    xi,eta=xi_eta
    N = ref.shape(xi,eta)
    N = np.asarray(N).ravel()          # (n_loc,)
    dN = np.asarray(ref.grad(xi,eta))  # (n_loc,2)
    return N, dN

def x_mapping(mesh, elem_id, xi_eta):
    nodes = mesh.nodes[mesh.elements_connectivity[elem_id]]
    nodes_x_y_pos = mesh.nodes_x_y_pos[nodes]
    poly_order = mesh.poly_order
    ref   = get_reference(mesh.element_type, poly_order)
    N,_ = _shape_and_grad(ref, xi_eta)
    return N @ nodes_x_y_pos                  # (2,)

def jacobian(mesh, elem_id, xi_eta):
    nodes = mesh.nodes[mesh.elements_connectivity[elem_id]]
    nodes_x_y_pos = mesh.nodes_x_y_pos[nodes]
    poly_order = mesh.poly_order
    ref   = get_reference(mesh.element_type, poly_order)
    _, dN = _shape_and_grad(ref, xi_eta)
    return dN.T @ nodes_x_y_pos

def det_jacobian(mesh, elem_id, xi_eta):
    return np.linalg.det(jacobian(mesh, elem_id, xi_eta))

def map_grad_scalar(mesh, elem_id, grad_ref, xi_eta):
    J = jacobian(mesh, elem_id, xi_eta)
    invJ = np.linalg.inv(J)
    return invJ.T @ grad_ref

def inv_jac_T(mesh, elem_id, xi_eta):
    J = jacobian(mesh, elem_id, xi_eta)
    return np.linalg.inv(J).T

def inverse_mapping(mesh, elem_id, x, tol=1e-10, maxiter=50):
    ref = get_reference(mesh.element_type, mesh.poly_order)
    # Use centroid as initial guess: (0,0) for quad, (1/3,1/3) for tri
    xi = np.array([0.0, 0.0]) if mesh.element_type == 'quad' else np.array([1/3, 1/3])
    for iter in range(maxiter):
        X = x_mapping(mesh, elem_id, xi)
        J = jacobian(mesh, elem_id, xi)
        try:
            delta = np.linalg.solve(J, x - X)
        except np.linalg.LinAlgError:
            raise ValueError(f"Jacobian singular at iteration {iter} for elem {elem_id}, x={x}")
        xi += delta
        if np.linalg.norm(delta) < tol:
            break
    else:
        raise ValueError(f"Inverse mapping did not converge after {maxiter} iterations for elem {elem_id}, x={x}, residual={np.linalg.norm(x - X)}")
    return xi

@staticmethod
def jacobian_1d(mesh, elem_id: int, ref_coords: tuple, local_edge_idx: int) -> float:
    """
    Computes the 1D Jacobian for a line integral (ratio of physical edge
    length to reference edge length).
    """
    J = jacobian(mesh, elem_id, ref_coords)

    if mesh.element_type == 'quad':
        # Tangent vectors on the reference quad edges [-1,1]x[-1,1]
        t_ref = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)[local_edge_idx]
    else: # tri
        # Tangent vectors on the reference triangle edges
        t_ref = np.array([[1, 0], [-1, 1], [0, -1]], dtype=float)[local_edge_idx]

    return np.linalg.norm(J @ t_ref)


def map_deriv(alpha: tuple[int,int], J: np.ndarray, J_inv: np.ndarray):
    """Return a callable that maps reference ∂^{α}φ to physical coords.

    Args:
        alpha: multi‑index (α_xi, α_eta)
        J, J_inv: Jacobian and its inverse at the quadrature point.

    Note:  We assume the mapping is *affine on the element* so higher
    derivatives of the mapping vanish.  For bilinear Q1 quads this is exact
    because J is constant in ξ, η.  For isoparametric Q2 and above, ghost‑edge
    penalties usually still use this approximation (see Burman 2010, Sec. 4).
    """
    order = sum(alpha)
    if order == 0:
        return lambda ref_vals: ref_vals  # shape (n_basis,)

    J_inv_T = J_inv.T  # (2×2)

    def _push_forward(ref_tensor):
        # ref_tensor shape = (n_basis, 2, 2, ... [order times] ...)
        phys = ref_tensor
        for _ in range(order):
            phys = np.tensordot(phys, J_inv_T, axes=([-1], [0]))
            # tensordot contracts last axis of phys with first axis of J_inv_T
            # Result keeps last axis = 2 (physical dim)
        return phys  # same rank, but expressed in x‑space

    return _push_forward


# --- Numba kernels for inverse mapping on P1/TRI and Q1/QUAD ---------------
try:
    import numba as _nb  # type: ignore
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

if _HAVE_NUMBA:
    @_nb.njit(cache=True, fastmath=True)
    def _invmap_p1(coords, x):
        # coords: (3,2) in order (0,0),(1,0),(0,1)
        X0x = coords[0, 0]; X0y = coords[0, 1]
        a00 = coords[1, 0] - X0x; a01 = coords[2, 0] - X0x
        a10 = coords[1, 1] - X0y; a11 = coords[2, 1] - X0y
        rx0 = x[0] - X0x; rx1 = x[1] - X0y
        det = a00 * a11 - a01 * a10
        r = ( a11 * rx0 - a01 * rx1) / det
        s = (-a10 * rx0 + a00 * rx1) / det
        out = np.empty(2)
        out[0] = r; out[1] = s
        return out

    @_nb.njit(cache=True, fastmath=True)
    def _q1_shape_grad(xi, eta):
        N = np.empty(4); dN = np.empty((4, 2))
        N[0] = 0.25 * (1 - xi) * (1 - eta)
        N[1] = 0.25 * (1 + xi) * (1 - eta)
        N[2] = 0.25 * (1 + xi) * (1 + eta)
        N[3] = 0.25 * (1 - xi) * (1 + eta)
        dN[0, 0] = -0.25 * (1 - eta); dN[0, 1] = -0.25 * (1 - xi)
        dN[1, 0] =  0.25 * (1 - eta); dN[1, 1] = -0.25 * (1 + xi)
        dN[2, 0] =  0.25 * (1 + eta); dN[2, 1] =  0.25 * (1 + xi)
        dN[3, 0] = -0.25 * (1 + eta); dN[3, 1] =  0.25 * (1 - xi)
        return N, dN

    @_nb.njit(cache=True, fastmath=True)
    def _invmap_q1_try(coords, x, tol=1e-10, maxiter=50, det_eps=1e-14):
        xi = 0.0; eta = 0.0
        ok = True
        for _ in range(maxiter):
            N, dN = _q1_shape_grad(xi, eta)
            X0 = 0.0; X1 = 0.0
            for i in range(4):
                X0 += N[i] * coords[i, 0]
                X1 += N[i] * coords[i, 1]
            rx0 = x[0] - X0; rx1 = x[1] - X1
            a00 = a01 = a10 = a11 = 0.0
            for i in range(4):
                gx = dN[i, 0]; gy = dN[i, 1]
                a00 += gx * coords[i, 0]; a01 += gx * coords[i, 1]
                a10 += gy * coords[i, 0]; a11 += gy * coords[i, 1]
            det = a00 * a11 - a01 * a10
            if abs(det) < det_eps:
                ok = False
                break
            inv00 =  a11 / det; inv01 = -a01 / det
            inv10 = -a10 / det; inv11 =  a00 / det
            dxi  = inv00 * rx0 + inv01 * rx1
            deta = inv10 * rx0 + inv11 * rx1
            xi  += dxi; eta += deta
            if (dxi*dxi + deta*deta) ** 0.5 < tol:
                ok = True
                break
        out = np.empty(3)
        out[0] = xi; out[1] = eta; out[2] = 1.0 if ok else 0.0
        return out

def inverse_mapping(mesh, elem_id, x, tol=1e-10, maxiter=50):
    """
    Fast path:
      tri & p==1  → exact affine inverse
      quad & p==1 → Newton on Q1 with det guard; fallback on failure
    Fallback:
      original generic Newton using reference objects.
    """
    coords = mesh.nodes_x_y_pos[mesh.nodes[mesh.elements_connectivity[elem_id]]].astype(float)
    x = np.asarray(x, dtype=float)

    if _HAVE_NUMBA and mesh.poly_order == 1:
        if mesh.element_type == 'tri' and coords.shape[0] >= 3:
            return _invmap_p1(coords[:3], x)
        if mesh.element_type == 'quad' and coords.shape[0] >= 4:
            r = _invmap_q1_try(coords[:4], x, tol, maxiter)
            if r[2] == 1.0:  # success
                return r[:2]

    # --- original generic Newton (unchanged) -------------------------------
    ref = get_reference(mesh.element_type, mesh.poly_order)
    xi = np.array([0.0, 0.0]) if mesh.element_type == 'quad' else np.array([1/3, 1/3])
    for _ in range(maxiter):
        X = x_mapping(mesh, elem_id, xi)
        J = jacobian(mesh, elem_id, xi)
        delta = np.linalg.solve(J, x - X)
        xi += delta
        if np.linalg.norm(delta) < tol:
            break
    return xi
