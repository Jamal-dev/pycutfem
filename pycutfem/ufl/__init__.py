"""
Public UFL-style API for pycutfem.

This package exposes the most commonly used symbolic building blocks and the
high-level assembly entry point.
"""

from pycutfem.ufl.forms import Equation, BoundaryCondition, assemble_form
from pycutfem.ufl.expressions import (
    Constant,
    TrialFunction,
    TestFunction,
    VectorTrialFunction,
    VectorTestFunction,
    Function,
    VectorFunction,
    HdivFunction,
    HdivTrialFunction,
    HdivTestFunction,
    grad,
    div,
    dot,
    inner,
)
from pycutfem.ufl.measures import dx, ds, dInterface, dGhost  # noqa: F401
from pycutfem.ufl.analytic import Analytic, x, y  # noqa: F401
from pycutfem.ufl.spaces import FunctionSpace

__all__ = [
    "Equation",
    "BoundaryCondition",
    "assemble_form",
    "Constant",
    "TrialFunction",
    "TestFunction",
    "VectorTrialFunction",
    "VectorTestFunction",
    "Function",
    "VectorFunction",
    "FunctionSpace",
    "HdivFunction",
    "HdivTrialFunction",
    "HdivTestFunction",
    "Analytic",
    "x",
    "y",
    "grad",
    "div",
    "dot",
    "inner",
    "dx",
    "ds",
    "dInterface",
    "dGhost",
]
