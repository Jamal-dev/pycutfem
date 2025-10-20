# analytic.py
import sympy as sp
import numpy as np
from pycutfem.ufl.expressions import Expression

class Analytic(Expression):
    """
    Wraps a SymPy/callable expression f(x,y). Returns scalar (...,)
    or vector (...,k). tensor_shape is () for scalar, (k,) for vector.
    """
    _x, _y = sp.symbols("x y")
    _coord_syms = (_x, _y)
    _coords = _coord_syms
    _space_dimensions = 2

    def __init__(self, sympy_expr, dim=0, degree:int | None = None):
        self.sympy_expr = sympy_expr
        self.dim = dim
        self._degree = degree
        if callable(sympy_expr):
            self._func = sympy_expr
        else:
            self._func = sp.lambdify(self._coords[:], sympy_expr, "numpy")

        # NEW: infer tensor shape (scalar () vs vector (k,))
        try:
            probe = self._func(np.asarray([0.0]), np.asarray([0.0]))
            arr = np.asarray(probe)
            self.tensor_shape = arr.shape[1:] if arr.ndim >= 1 else ()
        except Exception:
            self.tensor_shape = ()

    def grad(self): from pycutfem.ufl.expressions import Grad; return Grad(self)
    @property
    def degree(self):   # used by the estimator
        return self._degree
    def eval(self, X):
        """X : (..., 2) array → numpy array of same leading shape (scalar) or (...,k) (vector)."""
        X = np.asarray(X)
        return self._func(*[X[..., i] for i in range(self._space_dimensions)])

# helper to avoid typing Analytic._x all the time
x, y = Analytic._coord_syms
