# pycutfem/ufl/quadrature.py
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict

# Symbolic building blocks
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    Expression, Constant, TestFunction, TrialFunction, VectorTestFunction,
    VectorTrialFunction, Function, VectorFunction, Grad, DivOperation, Inner,
    Dot, Sum, Sub, Prod, Pos, Neg, Div, FacetNormal, ElementWiseConstant, Jump
)

if TYPE_CHECKING:
    from pycutfem.core.dofhandler import DofHandler

logger = logging.getLogger(__name__)

class PolynomialDegreeEstimator:
    """
    Estimates the polynomial degree of a UFL expression tree.
    This is used by the compiler to select an appropriate quadrature rule
    for each integral term, which can significantly improve performance.
    """

    def __init__(self, dh: 'DofHandler'):
        if dh.mixed_element is None:
            raise ValueError("PolynomialDegreeEstimator requires a MixedElement-backed DofHandler.")
        self.dh = dh
        self.me = dh.mixed_element
        self._geom_deg = max(0, self.me.mesh.poly_order - 1)
        
        # Memoization cache to avoid re-computing degrees for the same sub-expression
        self._cache: Dict[int, int] = {}

    def estimate_degree(self, expr: Expression) -> int:
        """Public method to estimate the polynomial degree of an expression."""
        self._cache.clear()
        degree = self._get_degree(expr)
        logger.debug(f"Estimated polynomial degree for expression '{expr!r}' is {degree}.")
        return degree

    def _get_degree(self, expr: Expression) -> int:
        """
        Recursively traverses the expression tree to determine its
        maximum polynomial degree.
        """
        # NEW: Use id(expr) as the cache key because Expression objects are not hashable.
        key = id(expr)
        if key in self._cache:
            return self._cache[key]

        # --- Base Cases: Leaf nodes of the expression tree ---
        if isinstance(expr, Constant):
            result = 0
        elif isinstance(expr, (TestFunction, TrialFunction, Function)):
            result = self.me._field_orders[expr.field_name]
        elif isinstance(expr, (VectorTestFunction, VectorTrialFunction, VectorFunction)):
            result = self.me._field_orders[expr.field_names[0]]

        # --- Recursive Cases: Operators ---
        elif isinstance(expr, Grad):
            # ∇u :  (deg(u)-1)  on the reference cell  +  J^{-T}  of degree geom-1
            result = max(0, self._get_degree(expr.operand) - 1) + self._geom_deg
        elif isinstance(expr, DivOperation):
            # NEW: Clamp the degree to be non-negative immediately.
            result = max(0, self._get_degree(expr.operand) - 1) + self._geom_deg
        elif isinstance(expr, (Sum, Sub)):
            result = max(self._get_degree(expr.a), self._get_degree(expr.b))
        elif isinstance(expr, (Prod, Dot, Inner)):
            result = self._get_degree(expr.a) + self._get_degree(expr.b)
        elif isinstance(expr, (Pos, Neg)):
            result = self._get_degree(expr.operand)
        elif isinstance(expr, FacetNormal):
            result = self._geom_deg
        elif isinstance(expr, ElementWiseConstant):
            # ElementWiseConstant is a special case that should not be confused with Constant.
            # It represents a constant value that is element-wise, so its degree is 0.
            result = 0
        elif isinstance(expr, Jump):
            result = self._get_degree(expr.operand)
        
        elif isinstance(expr, Analytic):                 # NEW ✱
            # 1. Honour an explicit hint  (Analytic(..., degree=k))
            if expr.degree is not None:
                result = expr.degree
            else:
                # 2. Fallback: be conservative – twice the mesh order
                result = max(2, self.me.mesh.poly_order * 2)
                logger.warning(
                    "Analytic expression without degree hint – "
                    "assuming polynomial degree %d", result)
        elif isinstance(expr, Div):
             deg_a = self._get_degree(expr.a)
             deg_b = self._get_degree(expr.b)
             if deg_b != 0:
                 logger.warning(f"Division by non-constant expression '{expr.b!r}' may result in a non-polynomial. Assuming degree of numerator.")
             result = deg_a
        else:
            raise NotImplementedError(f"Polynomial degree estimation not implemented for type {type(expr)}")

        # Cache the result before returning
        self._cache[key] = result
        return result
