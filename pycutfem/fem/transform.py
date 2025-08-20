"""pycutfem.fem.transform
Reference → physical mapping for linear elements.
"""
import numpy as np
from pycutfem.fem.reference import get_reference
from functools import lru_cache
import warnings

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


# --- Numba helpers for fast inverse mapping --------------------------------
try:
    import numba as _nb  # type: ignore
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

if _HAVE_NUMBA:
    from pycutfem.integration.pre_tabulates import _q1_shape_grad, _q2_shape_grad
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
    def _invmap_q1_try(coords, x, tol=1e-10, maxiter=50, det_eps=1e-14):
        # Note: Assumes the node coordinates in `coords` match the ordering in _q1_shape_grad
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

    # --- ADDED FOR Q2 ---
    

    @_nb.njit(cache=True, fastmath=True)
    def _invmap_q2_try(coords, x, tol=1e-10, maxiter=50, det_eps=1e-14):
        xi = 0.0; eta = 0.0
        ok = True
        for _ in range(maxiter):
            N, dN = _q2_shape_grad(xi, eta)
            X0 = 0.0; X1 = 0.0
            for i in range(9):
                X0 += N[i] * coords[i, 0]
                X1 += N[i] * coords[i, 1]
            rx0 = x[0] - X0; rx1 = x[1] - X1
            a00 = a01 = a10 = a11 = 0.0
            for i in range(9):
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
    Fast path dispatcher for inverse mapping.
    """
    coords = mesh.nodes_x_y_pos[mesh.nodes[mesh.elements_connectivity[elem_id]]].astype(float)
    x = np.asarray(x, dtype=float)

    if _HAVE_NUMBA:
        # --- Fast path for P1 / Q1 elements ---
        if mesh.poly_order == 1:
            if mesh.element_type == 'tri' and coords.shape[0] >= 3:
                return _invmap_p1(coords[:3], x)
            # IMPORTANT: The _invmap_q1_try assumes a node order that may not match your
            # project's lexicographical order. Ensure `coords[:4]` is ordered correctly.
            if mesh.element_type == 'quad' and coords.shape[0] >= 4:
                r = _invmap_q1_try(coords[:4], x, tol, maxiter)
                if r[2] == 1.0: return r[:2]
        
        # --- ADDED: Fast path for Q2 elements ---
        if mesh.poly_order == 2:
            if mesh.element_type == 'quad' and coords.shape[0] >= 9:
                # This uses the lexicographically ordered _q2_shape_grad
                r = _invmap_q2_try(coords[:9], x, tol, maxiter)
                if r[2] == 1.0: return r[:2]

    # --- Fallback to generic Newton method ---
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



#----------------------------- Hessian and Laplacian -----------------------------
# === NEW: exact Hessian mapping up to order-2 ================================



def _hess_inverse_from_Hx(J_inv: np.ndarray, Hx0: np.ndarray, Hx1: np.ndarray):
    """
    Given A=J^{-1} and Hxβ=∂²x_β/∂(ξ,η)², produce Hξα = ∂² ξ_α / ∂x².
    Hξα(ν,γ) = - Σ_{μ,λ} ( Σ_{β} A_{αβ} Hxβ_{μλ} ) A_{λν} A_{μγ}.
    """
    A = J_inv
    C0 = A[0,0]*Hx0 + A[0,1]*Hx1
    C1 = A[1,0]*Hx0 + A[1,1]*Hx1
    Hxi0 = -np.einsum("ml,ln,mg->ng", C0, A, A, optimize=True)
    Hxi1 = -np.einsum("ml,ln,mg->ng", C1, A, A, optimize=True)
    return Hxi0, Hxi1


def element_Hxi(mesh, elem_id: int, xi_eta: tuple[float,float]):
    """
    Exact inverse-map Hessians Hξ (two 2×2 matrices) for any geometry order.
    Hxi0 = ∂²ξ/∂x∂x, Hxi1 = ∂²η/∂x∂x.
    """
    xi, eta = xi_eta
    ref = get_reference(mesh.element_type, mesh.poly_order)

    # Standard Jacobian and its inverse
    J = jacobian(mesh, elem_id, (xi, eta))          # (2,2)
    A = np.linalg.inv(J)                            # (2,2)

    # Geometry second derivatives in reference coords:
    # HN[i, a, b] = ∂² N_i / ∂ξ_a ∂ξ_b
    HN = np.asarray(ref.hess(xi, eta))              # (nGeom, 2, 2)

    # Element node coordinates
    conn = mesh.elements_connectivity[elem_id]
    gidx = mesh.nodes[conn]                              # (nGeom,)
    X = mesh.nodes_x_y_pos[gidx].astype(float)           # (nGeom, 2)

    # build Hx0, Hx1
    Hx0 = np.empty((2, 2), float)
    Hx1 = np.empty((2, 2), float)
    d20 = HN[:, 0, 0]; d11 = HN[:, 0, 1]; d02 = HN[:, 1, 1]

    # ∂² x_β / ∂(ξ,η)²  = Σ_i HN[i,:,:] * X[i,β]
    # → two 2×2 matrices Hx0, Hx1
    # einsum "iab,i->ab" : sum over node index i
    X0 = X[:, 0]; X1 = X[:, 1]
    Hx0[0, 0] = np.dot(d20, X0); Hx0[0, 1] = np.dot(d11, X0); Hx0[1, 0] = Hx0[0, 1]; Hx0[1, 1] = np.dot(d02, X0)
    Hx1[0, 0] = np.dot(d20, X1); Hx1[0, 1] = np.dot(d11, X1); Hx1[1, 0] = Hx1[0, 1]; Hx1[1, 1] = np.dot(d02, X1)


    # Convert to inverse-map Hessians via the standard formula
    Hxi0, Hxi1 = _hess_inverse_from_Hx(A, Hx0, Hx1)
    return J, A, Hxi0, Hxi1


def _inverse_hessian_from_forward(A: np.ndarray, Hx0: np.ndarray, Hx1: np.ndarray):
    """
    Given A = J^{-1} and forward-map Hessians Hxβ = ∂² x_β / ∂(ξ,η)² (β=0:x,1:y),
    return the inverse-map Hessians (Hξ, Hη), where
      Hξ[i,j]  = ∂² ξ / ∂x^i ∂x^j,
      Hη[i,j]  = ∂² η / ∂x^i ∂x^j.
    Identity (2D):
      A^I{}_{ij} = - sum_β A^I{}_β * (a_i^T Hxβ a_j),   with a_i = A[:, i].
    """
    a0 = A[:, 0]  # column for x-derivatives
    a1 = A[:, 1]  # column for y-derivatives

    # contractions Tβ[i,j] = a_i^T Hxβ a_j
    T0 = np.array([[a0 @ Hx0 @ a0, a0 @ Hx0 @ a1],
                   [a1 @ Hx0 @ a0, a1 @ Hx0 @ a1]], dtype=float)
    T1 = np.array([[a0 @ Hx1 @ a0, a0 @ Hx1 @ a1],
                   [a1 @ Hx1 @ a0, a1 @ Hx1 @ a1]], dtype=float)

    Hxi  = - (A[0, 0] * T0 + A[0, 1] * T1)
    Heta = - (A[1, 0] * T0 + A[1, 1] * T1)
    return Hxi, Heta


def element_inverse_hessians(mesh, elem_id: int, xi_eta: tuple[float, float]):
    """
    Exact inverse-map Hessians for any isoparametric Q^n quad or P^1 tri.
    Returns: (J, A, Hxi, Heta), each H·· is a 2x2 array of ∂²(ξ or η)/∂x².
    """
    xi, eta = xi_eta
    ref = get_reference(mesh.element_type, mesh.poly_order)  

    J = jacobian(mesh, elem_id, (xi, eta))                   # (2,2)  
    A = np.linalg.inv(J)

    # Geometry Hessians in (ξ,η): HN[a,:,:] = ∂² ϕ_a / ∂(ξ,η)² for geometry basis a
    HN = np.asarray(ref.hess(xi, eta))                       # (nGeom, 2, 2)

    # Element geometry node coordinates X[a,β]
    conn = mesh.elements_connectivity[elem_id]
    gidx = mesh.nodes[conn]
    X = mesh.nodes_x_y_pos[gidx].astype(float)               # (nGeom,2)

    # Assemble forward-map Hessians Hxβ = Σ_a X[a,β] * HN[a,:,:]
    d20 = HN[:, 0, 0]; d11 = HN[:, 0, 1]; d02 = HN[:, 1, 1]
    X0 = X[:, 0]; X1 = X[:, 1]

    Hx0 = np.array([[np.dot(d20, X0), np.dot(d11, X0)],
                    [np.dot(d11, X0), np.dot(d02, X0)]], dtype=float)
    Hx1 = np.array([[np.dot(d20, X1), np.dot(d11, X1)],
                    [np.dot(d11, X1), np.dot(d02, X1)]], dtype=float)

    # Convert to inverse-map Hessians (exact identity)
    Hxi, Heta = _inverse_hessian_from_forward(A, Hx0, Hx1)
    return J, A, Hxi, Heta

# ---------- Forward-map material derivatives up to 3rd (as before) ----------
def element_forward_F2_F3(mesh, elem_id: int, xi_eta: Tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    xi, eta = xi_eta
    ref = get_reference(mesh.element_type, mesh.poly_order, max_deriv_order=3)

    conn = mesh.elements_connectivity[elem_id]
    gidx = mesh.nodes[conn]
    X = mesh.nodes_x_y_pos[gidx].astype(float)   # (nGeom, 2)

    # 2nd derivatives via Hessians
    HN = np.asarray(ref.hess(xi, eta))           # (nGeom, 2, 2)
    d20 = HN[:, 0, 0]; d11 = HN[:, 0, 1]; d02 = HN[:, 1, 1]
    X0, X1 = X[:, 0], X[:, 1]

    Hx0 = np.array([[np.dot(d20, X0), np.dot(d11, X0)],
                    [np.dot(d11, X0), np.dot(d02, X0)]], dtype=float)
    Hx1 = np.array([[np.dot(d20, X1), np.dot(d11, X1)],
                    [np.dot(d11, X1), np.dot(d02, X1)]], dtype=float)

    F2 = np.empty((2, 2, 2), float)
    F2[0] = Hx0
    F2[1] = Hx1

    # 3rd derivatives (fill by multi-index counts)
    d30 = np.asarray(ref.derivative(xi, eta, 3, 0))
    d21 = np.asarray(ref.derivative(xi, eta, 2, 1))
    d12 = np.asarray(ref.derivative(xi, eta, 1, 2))
    d03 = np.asarray(ref.derivative(xi, eta, 0, 3))

    def F3_for_beta(coords_1d: np.ndarray) -> np.ndarray:
        v30 = float(np.dot(d30, coords_1d))
        v21 = float(np.dot(d21, coords_1d))
        v12 = float(np.dot(d12, coords_1d))
        v03 = float(np.dot(d03, coords_1d))
        T = np.empty((2, 2, 2), dtype=float)
        T[0,0,0] = v30
        T[0,0,1] = T[0,1,0] = T[1,0,0] = v21
        T[0,1,1] = T[1,0,1] = T[1,1,0] = v12
        T[1,1,1] = v03
        return T

    F3 = np.empty((2, 2, 2, 2), float)
    F3[0] = F3_for_beta(X0)
    F3[1] = F3_for_beta(X1)

    return F2, F3


# ---------- 4th-order forward-map material derivatives ----------
def element_forward_F4(mesh, elem_id: int, xi_eta: Tuple[float, float]) -> np.ndarray:
    """
    F4[beta, L, M, N, P] = ∂^4 x_beta / ∂X^L ∂X^M ∂X^N ∂X^P   (symmetric in L,M,N,P).
    """
    xi, eta = xi_eta
    ref = get_reference(mesh.element_type, mesh.poly_order, max_deriv_order=4)

    conn = mesh.elements_connectivity[elem_id]
    gidx = mesh.nodes[conn]
    X = mesh.nodes_x_y_pos[gidx].astype(float)   # (nGeom, 2)

    # 4th derivatives of shape functions (per-axis degrees)
    d40 = np.asarray(ref.derivative(xi, eta, 4, 0))
    d31 = np.asarray(ref.derivative(xi, eta, 3, 1))
    d22 = np.asarray(ref.derivative(xi, eta, 2, 2))
    d13 = np.asarray(ref.derivative(xi, eta, 1, 3))
    d04 = np.asarray(ref.derivative(xi, eta, 0, 4))

    def F4_for_beta(coords_1d: np.ndarray) -> np.ndarray:
        v40 = float(np.dot(d40, coords_1d))
        v31 = float(np.dot(d31, coords_1d))
        v22 = float(np.dot(d22, coords_1d))
        v13 = float(np.dot(d13, coords_1d))
        v04 = float(np.dot(d04, coords_1d))
        T = np.empty((2, 2, 2, 2), dtype=float)
        # Fill by counts of 0/1 among (L,M,N,P)
        for L in (0, 1):
            for M in (0, 1):
                for N in (0, 1):
                    for P in (0, 1):
                        cnt1 = L + M + N + P   # number of 'eta' indices
                        cnt0 = 4 - cnt1        # number of 'xi' indices
                        if   cnt0 == 4: val = v40
                        elif cnt0 == 3: val = v31
                        elif cnt0 == 2: val = v22
                        elif cnt0 == 1: val = v13
                        else:           val = v04
                        T[L, M, N, P] = val
        return T

    F4 = np.empty((2, 2, 2, 2, 2), float)
    F4[0] = F4_for_beta(X[:, 0])
    F4[1] = F4_for_beta(X[:, 1])
    return F4


# ---------- Exact inverse-map Hessians (order 2) ----------
def hess_inverse_from_forward(A: np.ndarray, Hx0: np.ndarray, Hx1: np.ndarray):
    """
    A2[I,i,j] = ∂^2 X^I / ∂x^i ∂x^j  from forward-map Hessians Hxβ.
    Identity: A^I_{ij} = -∑_β A^I_β (a_i^T Hxβ a_j),  a_i = A[:,i].
    """
    a0 = A[:, 0]; a1 = A[:, 1]
    T0 = np.array([[a0 @ Hx0 @ a0, a0 @ Hx0 @ a1],
                   [a1 @ Hx0 @ a0, a1 @ Hx0 @ a1]], dtype=float)
    T1 = np.array([[a0 @ Hx1 @ a0, a0 @ Hx1 @ a1],
                   [a1 @ Hx1 @ a0, a1 @ Hx1 @ a1]], dtype=float)
    Hxi  = - (A[0, 0] * T0 + A[0, 1] * T1)
    Heta = - (A[1, 0] * T0 + A[1, 1] * T1)
    A2 = np.stack([Hxi, Heta], axis=0)  # (2,2,2)
    return A2


# ---------- 3rd-order inverse jet (material identity; loops only) ----------
def inverse_A3_material(A: np.ndarray, F2: np.ndarray, F3: np.ndarray) -> np.ndarray:
    A3 = np.zeros((2, 2, 2, 2), dtype=float)
    for I in (0, 1):
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    acc = 0.0
                    # Term 1
                    for q in (0, 1):
                        for R in (0, 1):
                            for S in (0, 1):
                                for p in (0, 1):
                                    for J in (0, 1):
                                        for L in (0, 1):
                                            acc += (A[I, q] * F2[q, R, S] * A[S, k] *
                                                    A[R, p] * F2[p, J, L] * A[L, j] * A[J, i])
                    # Term 2
                    for p in (0, 1):
                        for J in (0, 1):
                            for L in (0, 1):
                                for M in (0, 1):
                                    acc += (- A[I, p] * F3[p, J, L, M] *
                                            A[M, k] * A[L, j] * A[J, i])
                    # Term 3
                    for p in (0, 1):
                        for J in (0, 1):
                            for L in (0, 1):
                                for q in (0, 1):
                                    for R in (0, 1):
                                        for S in (0, 1):
                                            acc += (A[I, p] * F2[p, J, L] * A[L, q] *
                                                    F2[q, R, S] * A[S, k] * A[R, j] * A[J, i])
                    # Term 4
                    for p in (0, 1):
                        for J in (0, 1):
                            for L in (0, 1):
                                for q in (0, 1):
                                    for R in (0, 1):
                                        for S in (0, 1):
                                            acc += (A[I, p] * F2[p, J, L] * A[L, j] *
                                                    A[J, q] * F2[q, R, S] * A[S, k] * A[R, i])
                    A3[I, i, j, k] = acc
    return A3


# ---------- 4th-order inverse jet A4 via d/dx_l of A3 (loops only) ----------
def inverse_A4_material(A: np.ndarray, A2: np.ndarray,
                        F2: np.ndarray, F3: np.ndarray, F4: np.ndarray) -> np.ndarray:
    """
    A4[I,i,j,k,l] = ∂^4 X^I / ∂x^i ∂x^j ∂x^k ∂x^l
    by differentiating the A3 material identity once and using:
      ∂_l A^I_q = A2[I,q,l],
      ∂_l A^S_k = A2[S,k,l], etc.,
      ∂_l F2[q,R,S] = F3[q,R,S,T] A[T,l],
      ∂_l F3[p,J,L,M] = F4[p,J,L,M,N] A[N,l].
    """
    A4 = np.zeros((2, 2, 2, 2, 2), dtype=float)
    for I in (0, 1):
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    for l in (0, 1):
                        acc = 0.0
                        # ----- d/dx_l of Term 1 -----
                        for q in (0, 1):
                            for R in (0, 1):
                                for S in (0, 1):
                                    for p in (0, 1):
                                        for J in (0, 1):
                                            for L in (0, 1):
                                                base = F2[q, R, S] * A[S, k] * A[R, p] * F2[p, J, L] * A[L, j] * A[J, i]
                                                # ∂A[I,q]
                                                acc += A2[I, q, l] * base
                                                # ∂F2[q,R,S]
                                                for T in (0, 1):
                                                    acc += A[I, q] * F3[q, R, S, T] * A[T, l] * A[S, k] * A[R, p] * F2[p, J, L] * A[L, j] * A[J, i]
                                                # ∂A[S,k]
                                                acc += A[I, q] * F2[q, R, S] * A2[S, k, l] * A[R, p] * F2[p, J, L] * A[L, j] * A[J, i]
                                                # ∂A[R,p]
                                                acc += A[I, q] * F2[q, R, S] * A[S, k] * A2[R, p, l] * F2[p, J, L] * A[L, j] * A[J, i]
                                                # ∂F2[p,J,L]
                                                for T in (0, 1):
                                                    acc += A[I, q] * F2[q, R, S] * A[S, k] * A[R, p] * F3[p, J, L, T] * A[T, l] * A[L, j] * A[J, i]
                                                # ∂A[L,j]
                                                acc += A[I, q] * F2[q, R, S] * A[S, k] * A[R, p] * F2[p, J, L] * A2[L, j, l] * A[J, i]
                                                # ∂A[J,i]
                                                acc += A[I, q] * F2[q, R, S] * A[S, k] * A[R, p] * F2[p, J, L] * A[L, j] * A2[J, i, l]
                        # ----- d/dx_l of Term 2 -----
                        for p in (0, 1):
                            for J in (0, 1):
                                for L in (0, 1):
                                    for M in (0, 1):
                                        base = F3[p, J, L, M] * A[M, k] * A[L, j] * A[J, i]
                                        # ∂A[I,p]
                                        acc += - A2[I, p, l] * base
                                        # ∂F3[p,J,L,M]
                                        for N in (0, 1):
                                            acc += - A[I, p] * F4[p, J, L, M, N] * A[N, l] * A[M, k] * A[L, j] * A[J, i]
                                        # ∂A[M,k]
                                        acc += - A[I, p] * F3[p, J, L, M] * A2[M, k, l] * A[L, j] * A[J, i]
                                        # ∂A[L,j]
                                        acc += - A[I, p] * F3[p, J, L, M] * A[M, k] * A2[L, j, l] * A[J, i]
                                        # ∂A[J,i]
                                        acc += - A[I, p] * F3[p, J, L, M] * A[M, k] * A[L, j] * A2[J, i, l]
                        # ----- d/dx_l of Term 3 -----
                        for p in (0, 1):
                            for J in (0, 1):
                                for L in (0, 1):
                                    for q in (0, 1):
                                        for R in (0, 1):
                                            for S in (0, 1):
                                                base_left  = F2[p, J, L] * A[L, q]
                                                base_right = F2[q, R, S] * A[S, k] * A[R, j] * A[J, i]
                                                # ∂A[I,p]
                                                acc += A2[I, p, l] * base_left * base_right
                                                # ∂F2[p,J,L]
                                                for T in (0, 1):
                                                    acc += A[I, p] * F3[p, J, L, T] * A[T, l] * A[L, q] * base_right
                                                # ∂A[L,q]
                                                acc += A[I, p] * F2[p, J, L] * A2[L, q, l] * base_right
                                                # ∂F2[q,R,S]
                                                for T in (0, 1):
                                                    acc += A[I, p] * F2[p, J, L] * A[L, q] * F3[q, R, S, T] * A[T, l] * A[S, k] * A[R, j] * A[J, i]
                                                # ∂A[S,k]
                                                acc += A[I, p] * F2[p, J, L] * A[L, q] * F2[q, R, S] * A2[S, k, l] * A[R, j] * A[J, i]
                                                # ∂A[R,j]
                                                acc += A[I, p] * F2[p, J, L] * A[L, q] * F2[q, R, S] * A[S, k] * A2[R, j, l] * A[J, i]
                                                # ∂A[J,i]
                                                acc += A[I, p] * F2[p, J, L] * A[L, q] * F2[q, R, S] * A[S, k] * A[R, j] * A2[J, i, l]
                        # ----- d/dx_l of Term 4 -----
                        for p in (0, 1):
                            for J in (0, 1):
                                for L in (0, 1):
                                    for q in (0, 1):
                                        for R in (0, 1):
                                            for S in (0, 1):
                                                base_left  = F2[p, J, L] * A[L, j] * A[J, q]
                                                base_right = F2[q, R, S] * A[S, k] * A[R, i]
                                                # ∂A[I,p]
                                                acc += A2[I, p, l] * base_left * base_right
                                                # ∂F2[p,J,L]
                                                for T in (0, 1):
                                                    acc += A[I, p] * F3[p, J, L, T] * A[T, l] * A[L, j] * A[J, q] * base_right
                                                # ∂A[L,j]
                                                acc += A[I, p] * F2[p, J, L] * A2[L, j, l] * A[J, q] * base_right
                                                # ∂A[J,q]
                                                acc += A[I, p] * F2[p, J, L] * A[L, j] * A2[J, q, l] * base_right
                                                # ∂F2[q,R,S]
                                                for T in (0, 1):
                                                    acc += A[I, p] * F2[p, J, L] * A[L, j] * A[J, q] * F3[q, R, S, T] * A[T, l] * A[S, k] * A[R, i]
                                                # ∂A[S,k]
                                                acc += A[I, p] * F2[p, J, L] * A[L, j] * A[J, q] * F2[q, R, S] * A2[S, k, l] * A[R, i]
                                                # ∂A[R,i]
                                                acc += A[I, p] * F2[p, J, L] * A[L, j] * A[J, q] * F2[q, R, S] * A[S, k] * A2[R, i, l]

                        A4[I, i, j, k, l] = acc
    return A4