try:
    import numba as _nb  # type: ignore
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False
import numpy as np

if _HAVE_NUMBA:
    @_nb.njit(cache=True, fastmath=True, parallel=True)
    def _tabulate_p1(xi, eta, N, dN):
        """
        Tabulates shape functions and derivatives for a P1 (3-node linear triangle) element.
        """
        nE, nQ = xi.shape
        for e in _nb.prange(nE):
            for q in range(nQ):
                s = xi[e, q]; t = eta[e, q]
                
                # Shape functions
                N[e, q, 0] = 1.0 - s - t
                N[e, q, 1] = s
                N[e, q, 2] = t
                
                # Derivatives [dN/ds, dN/dt]
                dN[e, q, 0, 0] = -1.0; dN[e, q, 0, 1] = -1.0
                dN[e, q, 1, 0] =  1.0; dN[e, q, 1, 1] =  0.0
                dN[e, q, 2, 0] =  0.0; dN[e, q, 2, 1] =  1.0


@_nb.njit(cache=True, fastmath=True, parallel=True)
def _tabulate_q1(xi, eta, N, dN):
    """
    Tabulates shape functions for a Q1 element using
    LEXICOGRAPHICAL node ordering to match quad_qn.py.
    """
    nE, nQ = xi.shape
    for e in _nb.prange(nE):
        for q in range(nQ):
            s = xi[e, q]; t = eta[e, q]
            
            # Node 0: (-1, -1)
            N[e, q, 0] = 0.25 * (1 - s) * (1 - t)
            dN[e, q, 0, 0] = -0.25 * (1 - t); dN[e, q, 0, 1] = -0.25 * (1 - s)
            
            # Node 1: (1, -1)
            N[e, q, 1] = 0.25 * (1 + s) * (1 - t)
            dN[e, q, 1, 0] =  0.25 * (1 - t); dN[e, q, 1, 1] = -0.25 * (1 + s)

            # Node 2: (-1, 1)
            N[e, q, 2] = 0.25 * (1 - s) * (1 + t)
            dN[e, q, 2, 0] = -0.25 * (1 + t); dN[e, q, 2, 1] =  0.25 * (1 - s)

            # Node 3: (1, 1)
            N[e, q, 3] = 0.25 * (1 + s) * (1 + t)
            dN[e, q, 3, 0] =  0.25 * (1 + t); dN[e, q, 3, 1] =  0.25 * (1 + s)


@_nb.njit(cache=True, fastmath=True, parallel=True)
def _tabulate_q2(xi, eta, N, dN):
    """
    Tabulates shape functions for a Q2 element using
    LEXICOGRAPHICAL node ordering to match quad_qn.py.
    """
    nE, nQ = xi.shape
    for e in _nb.prange(nE):
        for q in range(nQ):
            s = xi[e, q]; t = eta[e, q]
            
            s_m1 = s - 1.0; s_p1 = s + 1.0
            t_m1 = t - 1.0; t_p1 = t + 1.0
            s2 = s * s; t2 = t * t
            s2_m1 = 1.0 - s2; t2_m1 = 1.0 - t2

            # --- Row 0: eta = -1 ---
            # Node 0: (-1, -1)
            N[e, q, 0] = 0.25 * s * t * s_m1 * t_m1
            dN[e, q, 0, 0] = 0.25 * t * t_m1 * (2*s - 1.0); dN[e, q, 0, 1] = 0.25 * s * s_m1 * (2*t - 1.0)
            
            # Node 1: (0, -1)
            N[e, q, 1] = 0.5 * t * t_m1 * s2_m1
            dN[e, q, 1, 0] = -s * t * t_m1; dN[e, q, 1, 1] = 0.5 * s2_m1 * (2*t - 1.0)

            # Node 2: (1, -1)
            N[e, q, 2] = 0.25 * s * t * s_p1 * t_m1
            dN[e, q, 2, 0] = 0.25 * t * t_m1 * (2*s + 1.0); dN[e, q, 2, 1] = 0.25 * s * s_p1 * (2*t - 1.0)
            
            # --- Row 1: eta = 0 ---
            # Node 3: (-1, 0)
            N[e, q, 3] = 0.5 * s * s_m1 * t2_m1
            dN[e, q, 3, 0] = 0.5 * (2*s - 1.0) * t2_m1; dN[e, q, 3, 1] = -t * s * s_m1
            
            # Node 4: (0, 0)
            N[e, q, 4] = s2_m1 * t2_m1
            dN[e, q, 4, 0] = -2.0 * s * t2_m1; dN[e, q, 4, 1] = -2.0 * t * s2_m1

            # Node 5: (1, 0)
            N[e, q, 5] = 0.5 * s * s_p1 * t2_m1
            dN[e, q, 5, 0] = 0.5 * (2*s + 1.0) * t2_m1; dN[e, q, 5, 1] = -t * s * s_p1

            # --- Row 2: eta = 1 ---
            # Node 6: (-1, 1)
            N[e, q, 6] = 0.25 * s * t * s_m1 * t_p1
            dN[e, q, 6, 0] = 0.25 * t * t_p1 * (2*s - 1.0); dN[e, q, 6, 1] = 0.25 * s * s_m1 * (2*t + 1.0)

            # Node 7: (0, 1)
            N[e, q, 7] = 0.5 * t * t_p1 * s2_m1
            dN[e, q, 7, 0] = -s * t * t_p1; dN[e, q, 7, 1] = 0.5 * s2_m1 * (2*t + 1.0)

            # Node 8: (1, 1)
            N[e, q, 8] = 0.25 * s * t * s_p1 * t_p1
            dN[e, q, 8, 0] = 0.25 * t * t_p1 * (2*s + 1.0); dN[e, q, 8, 1] = 0.25 * s * s_p1 * (2*t + 1.0)
        




# ----------------------- JIT searchsorted (positions of items) ----------------
if _HAVE_NUMBA:
    @_nb.njit(cache=True)
    def _searchsorted_positions(sorted_unique: np.ndarray, items: np.ndarray) -> np.ndarray:
        n = sorted_unique.shape[0]
        m = items.shape[0]
        out = np.empty(m, dtype=np.int32)
        for j in range(m):
            x = items[j]
            lo = 0
            hi = n
            while lo < hi:
                mid = (lo + hi) // 2
                if sorted_unique[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            # assume present (union = unique(pos ∪ neg))
            out[j] = lo if (lo < n and sorted_unique[lo] == x) else -1
        return out
else:
    def _searchsorted_positions(sorted_unique: np.ndarray, items: np.ndarray) -> np.ndarray:
        # Python fallback
        idx = np.searchsorted(sorted_unique, items)
        mask = (idx < sorted_unique.size) & (sorted_unique[idx] == items)
        out = np.where(mask, idx, -1).astype(np.int32)
        return out

# ----------------------- Q1 (quad, p=1) derivatives up to order 2 --------------
# nodes in row-major: (xi,eta) = (-1,-1),(+1,-1),(+1,+1),(-1,+1)
if _HAVE_NUMBA:
    @_nb.njit(cache=True)
    def _eval_deriv_q1(xi: float, eta: float, dx: int, dy: int) -> np.ndarray:
        # 1D shapes
        L0 = 0.5*(1.0 - xi); L1 = 0.5*(1.0 + xi)
        M0 = 0.5*(1.0 - eta); M1 = 0.5*(1.0 + eta)
        dL0 = -0.5; dL1 = 0.5
        dM0 = -0.5; dM1 = 0.5
        # second derivs are zero for Q1
        if dx == 0 and dy == 0:
            return np.array([L0*M0, L1*M0, L1*M1, L0*M1])
        if dx == 1 and dy == 0:
            return np.array([dL0*M0, dL1*M0, dL1*M1, dL0*M1])
        if dx == 0 and dy == 1:
            return np.array([L0*dM0, L1*dM0, L1*dM1, L0*dM1])
        # all second-order derivatives are zero for Q1
        return np.zeros(4)

    @_nb.njit(cache=True, parallel=True, fastmath=True)
    def _tabulate_deriv_q1(xi_tab, eta_tab, dx: int, dy: int, out):
        nE, nQ = xi_tab.shape
        for e in _nb.prange(nE):
            for q in range(nQ):
                out[e, q, :] = _eval_deriv_q1(xi_tab[e, q], eta_tab[e, q], dx, dy)

# ----------------------- Q2 (quad, p=2) derivatives up to order 2 --------------
# 1D nodes s∈{-1,0,1}:  L0=0.5*s*(s-1), L1=1-s*s, L2=0.5*s*(s+1)
if _HAVE_NUMBA:
    @_nb.njit(cache=True)
    def _L(s):
        L0 = 0.5*s*(s - 1.0)
        L1 = 1.0 - s*s
        L2 = 0.5*s*(s + 1.0)
        dL0 = s - 0.5
        dL1 = -2.0*s
        dL2 = s + 0.5
        ddL0 = 1.0
        ddL1 = -2.0
        ddL2 = 1.0
        return (L0, L1, L2, dL0, dL1, dL2, ddL0, ddL1, ddL2)

    @_nb.njit(cache=True)
    def _eval_deriv_q2(xi: float, eta: float, dx: int, dy: int) -> np.ndarray:
        L0,L1,L2,dL0,dL1,dL2,ddL0,ddL1,ddL2 = _L(xi)
        M0,M1,M2,dM0,dM1,dM2,ddM0,ddM1,ddM2 = _L(eta)
        # tensor product, row-major (eta bottom->top, xi left->right)
        out = np.empty(9)
        if dx == 0 and dy == 0:
            Li = (L0, L1, L2); Mj = (M0, M1, M2)
        elif dx == 1 and dy == 0:
            Li = (dL0, dL1, dL2); Mj = (M0, M1, M2)
        elif dx == 0 and dy == 1:
            Li = (L0, L1, L2); Mj = (dM0, dM1, dM2)
        elif dx == 2 and dy == 0:
            Li = (ddL0, ddL1, ddL2); Mj = (M0, M1, M2)
        elif dx == 0 and dy == 2:
            Li = (L0, L1, L2); Mj = (ddM0, ddM1, ddM2)
        elif dx == 1 and dy == 1:
            # product rule: (d/dxi Li)*(d/deta Mj)
            Li = (dL0, dL1, dL2); Mj = (dM0, dM1, dM2)
        else:
            return np.zeros(9)
        k = 0
        for j in range(3):
            for i in range(3):
                out[k] = Li[i] * Mj[j]
                k += 1
        return out

    @_nb.njit(cache=True, parallel=True, fastmath=True)
    def _tabulate_deriv_q2(xi_tab, eta_tab, dx: int, dy: int, out):
        nE, nQ = xi_tab.shape
        for e in _nb.prange(nE):
            for q in range(nQ):
                out[e, q, :] = _eval_deriv_q2(xi_tab[e, q], eta_tab[e, q], dx, dy)

# ----------------------- P1 (tri, p=1) derivatives up to order 2 ---------------
if _HAVE_NUMBA:
    @_nb.njit(cache=True)
    def _eval_deriv_p1(xi: float, eta: float, dx: int, dy: int) -> np.ndarray:
        # N1=1-xi-eta, N2=xi, N3=eta
        if dx == 0 and dy == 0:
            return np.array([1.0 - xi - eta, xi, eta])
        if dx == 1 and dy == 0:
            return np.array([-1.0, 1.0, 0.0])
        if dx == 0 and dy == 1:
            return np.array([-1.0, 0.0, 1.0])
        # second derivatives are zero for P1
        return np.zeros(3)

    @_nb.njit(cache=True, parallel=True, fastmath=True)
    def _tabulate_deriv_p1(xi_tab, eta_tab, dx: int, dy: int, out):
        nE, nQ = xi_tab.shape
        for e in _nb.prange(nE):
            for q in range(nQ):
                out[e, q, :] = _eval_deriv_p1(xi_tab[e, q], eta_tab[e, q], dx, dy)

    @_nb.njit(cache=True, fastmath=True)
    def _q1_shape_grad(xi, eta):
        # NOTE: This uses a counter-clockwise order, not lexicographical
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
    def _q2_shape_grad(xi, eta):
        """
        Computes Q2 shape functions and gradients in LEXICOGRAPHICAL order.
        """
        N = np.empty(9); dN = np.empty((9, 2))
        s = xi; t = eta
        s_m1 = s - 1.0; s_p1 = s + 1.0
        t_m1 = t - 1.0; t_p1 = t + 1.0
        s2 = s * s; t2 = t * t
        s2_m1 = 1.0 - s2; t2_m1 = 1.0 - t2
        # Node 0: (-1, -1)
        N[0] = 0.25 * s * t * s_m1 * t_m1
        dN[0, 0] = 0.25 * t * t_m1 * (2*s - 1.0); dN[0, 1] = 0.25 * s * s_m1 * (2*t - 1.0)
        # Node 1: (0, -1)
        N[1] = 0.5 * t * t_m1 * s2_m1
        dN[1, 0] = -s * t * t_m1; dN[1, 1] = 0.5 * s2_m1 * (2*t - 1.0)
        # Node 2: (1, -1)
        N[2] = 0.25 * s * t * s_p1 * t_m1
        dN[2, 0] = 0.25 * t * t_m1 * (2*s + 1.0); dN[2, 1] = 0.25 * s * s_p1 * (2*t - 1.0)
        # Node 3: (-1, 0)
        N[3] = 0.5 * s * s_m1 * t2_m1
        dN[3, 0] = 0.5 * (2*s - 1.0) * t2_m1; dN[3, 1] = -t * s * s_m1
        # Node 4: (0, 0)
        N[4] = s2_m1 * t2_m1
        dN[4, 0] = -2.0 * s * t2_m1; dN[4, 1] = -2.0 * t * s2_m1
        # Node 5: (1, 0)
        N[5] = 0.5 * s * s_p1 * t2_m1
        dN[5, 0] = 0.5 * (2*s + 1.0) * t2_m1; dN[5, 1] = -t * s * s_p1
        # Node 6: (-1, 1)
        N[6] = 0.25 * s * t * s_m1 * t_p1
        dN[6, 0] = 0.25 * t * t_p1 * (2*s - 1.0); dN[6, 1] = 0.25 * s * s_m1 * (2*t + 1.0)
        # Node 7: (0, 1)
        N[7] = 0.5 * t * t_p1 * s2_m1
        dN[7, 0] = -s * t * t_p1; dN[7, 1] = 0.5 * s2_m1 * (2*t + 1.0)
        # Node 8: (1, 1)
        N[8] = 0.25 * s * t * s_p1 * t_p1
        dN[8, 0] = 0.25 * t * t_p1 * (2*s + 1.0); dN[8, 1] = 0.25 * s * s_p1 * (2*t + 1.0)
        return N, dN