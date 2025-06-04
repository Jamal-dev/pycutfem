"""
pycutfem.fem.reference.quad_qn
Tensor-product Qn basis (any n≥1) on [-1,1]×[-1,1].
"""
from functools import lru_cache
import sympy as sp
import numpy as np

@lru_cache(maxsize=None)
def _lagrange_basis_1d(n: int):
    x = sp.Symbol("x")
    nodes = np.linspace(-1, 1, n + 1)
    basis = []
    for i, xi in enumerate(nodes):
        li = 1
        for j, xj in enumerate(nodes):
            if i != j:
                li *= (x - xj) / (xi - xj)
        basis.append(sp.simplify(li))
    return basis, x, nodes

@lru_cache(maxsize=None)
def quad_qn(n: int):
    """Return (shape, grad) callables for Qn."""
    basis_1d, x1d, _ = _lagrange_basis_1d(n)
    xi,  eta  = sp.symbols("xi eta")

    basis_2d = []
    dbasis_2d = []
    for j, Lj in enumerate(basis_1d):
        for i, Li in enumerate(basis_1d):
            phi = sp.simplify(Li.subs({x1d: xi}) * Lj.subs({x1d: eta}))
            basis_2d.append(phi)
            dphi = sp.Matrix([sp.diff(phi, xi), sp.diff(phi, eta)])
            dbasis_2d.append(dphi)

    shape = sp.lambdify((xi, eta), sp.Matrix(basis_2d), "numpy")
    grad  = sp.lambdify((xi, eta), sp.Matrix.hstack(*dbasis_2d), "numpy")
    return shape, grad
