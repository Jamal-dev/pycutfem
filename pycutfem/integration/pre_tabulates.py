try:
    import numba as _nb  # type: ignore
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

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