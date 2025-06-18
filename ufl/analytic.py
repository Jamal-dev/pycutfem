import sympy as sp
import numpy as np
from ufl.expressions import Expression

class Analytic(Expression):
    """
    Wraps a SymPy scalar expression *f(x,y,z)*.  It remains symbolic in the
    UFL AST, but can be **evaluated** by calling .eval(x) after assembly
    (useful for error calculation or in BC weak imposition).
    """
    _x, _y = sp.symbols("x y")
    _coord_syms = (_x, _y)
    _coords = _coord_syms
    _space_dimensions = 2

    def __init__(self, sympy_expr, dim=0): # dim is a tensor dimension, not a space dimension
        self.sympy_expr = sympy_expr
        self.dim = dim
        self._func = sp.lambdify(self._coords[:], sympy_expr, "numpy")

    def grad(self): from ufl.expressions import Grad; return Grad(self)
    # nothing extra is needed for UFL algebra – we inherit +,*,grad,…
    def eval(self, X):
        """X : (..., dim) array → numpy array of the same leading shape"""
        X = np.asarray(X)
        return self._func(*[X[..., i] for i in range(self._space_dimensions)])

# helper to avoid typing Analytic._x all the time
x, y = Analytic._coord_syms