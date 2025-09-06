from functools import lru_cache
import sympy as sp
import numpy as np

@lru_cache(maxsize=None)
def _lagrange_basis_1d(n: int, max_deriv_order: int):
    """Return 1D Lagrange basis + derivatives as NUMPY-callable lambdas."""
    x = sp.symbols('x')
    nodes = np.linspace(-1.0, 1.0, n+1)
    L = []
    dL = {k: [] for k in range(max_deriv_order+1)}
    for i, xi in enumerate(nodes):
        num = 1
        den = 1.0
        for j, xj in enumerate(nodes):
            if i == j:
                continue
            num *= (x - xj)
            den *= (xi - xj)
        Li = sp.simplify(num/den)
        # lambdify shape & all required derivatives (SymPy → numpy functions)
        L.append(sp.lambdify(x, Li, 'numpy'))
        for k in range(max_deriv_order+1):
            dL[k].append(sp.lambdify(x, sp.diff(Li, x, k), 'numpy'))
    return nodes, L, dL

@lru_cache(maxsize=None)
def quad_qn(n: int, max_deriv_order: int = 2):
    """
    Tensor-product Q_n on [-1,1]^2.
    Returns: (shape_fn, deriv_fns) where
      shape_fn(xi,eta) -> ( (n+1)^2, )
      deriv_fns[(ax,ay)](xi,eta) -> ( (n+1)^2, ), ax+ay<=max_deriv_order
    Stacking order is (eta outer, xi inner): index = j*(n+1) + i
    """
    nodes1d, L, dL = _lagrange_basis_1d(n, max_deriv_order)

    def _eval_1d(vals, z):
        # vals is a list of 1D lambdas; output shape (n+1,)
        return np.array([f(z) for f in vals], dtype=float)

    def shape(xi, eta):
        lx = _eval_1d(L, xi)          # (n+1,)
        ly = _eval_1d(L, eta)         # (n+1,)
        # eta outer, xi inner
        return np.outer(ly, lx).reshape(-1)

    derivs = {}
    for ax in range(max_deriv_order+1):
        for ay in range(max_deriv_order+1):
            if ax + ay > max_deriv_order:
                continue
            def make(ax=ax, ay=ay):
                def d(xi, eta):
                    dx = _eval_1d(dL[ax], xi)   # d^ax/dxi^ax L_i(xi)
                    dy = _eval_1d(dL[ay], eta)  # d^ay/deta^ay L_j(eta)
                    return np.outer(dy, dx).reshape(-1)
                return d
            derivs[(ax, ay)] = make()
    return shape, derivs


# Example usage in __init__.py would need adjustment; here’s a minimal test
if __name__ == "__main__":
    for n in (1,2,3,4):
        shape, derivs = quad_qn(n, max_deriv_order=3)
        nodes = np.linspace(-1.0,1.0,n+1)
        lattice = [(xi,eta) for eta in nodes for xi in nodes]  # eta outer, xi inner
        # Kronecker property at nodes
        for k,(xi,eta) in enumerate(lattice):
            N = shape(xi,eta)
            assert np.allclose(N[k], 1.0) and np.allclose(np.delete(N,k), 0.0, atol=1e-12)
        # Partition of unity & zero-sum of first derivatives
        xi, eta = 0.123, -0.456
        N = shape(xi,eta)
        Nx = derivs[(1,0)](xi,eta)
        Ny = derivs[(0,1)](xi,eta)
        assert np.allclose(N.sum(), 1.0, atol=1e-12)
        assert np.allclose(Nx.sum(), 0.0, atol=1e-12)
        assert np.allclose(Ny.sum(), 0.0, atol=1e-12)
    print("quad_qn OK")
