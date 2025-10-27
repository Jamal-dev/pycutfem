from pycutfem.ufl.expressions import Expression, Integral, Sum, Sub, Prod, Constant
from typing import Callable, List, Dict
import numbers

class Form(Expression):
    """Represents the sum of several integrals that make up one side of a weak form."""
    def __init__(self, integrals: list):
        self.integrals = []
        for term in integrals:
            if isinstance(term, Form): self.integrals.extend(term.integrals)
            elif isinstance(term, Integral): self.integrals.append(term)
            else:
                raise TypeError(f"A Form can only be constructed from Integral or other Form objects, not {type(term)}")

    def __add__(self, other):
        if isinstance(other, Integral):
            return Form(self.integrals + [other])
        if isinstance(other, Form):
            return Form(self.integrals + other.integrals)
        raise TypeError(f"Can only add an Integral or Form to a Form, not {type(other)}")

    def __repr__(self):
        if not self.integrals:
            return "Form()"
        pieces = []
        for i, integral in enumerate(self.integrals, start=1):
            pieces.append(f"  [{i}] {integral!r}")
        return "Form(\n" + ",\n".join(pieces) + "\n)"

    __str__ = __repr__

    def __sub__(self, other):
        # Use the __neg__ method to create negated versions of the terms to be subtracted.
        if isinstance(other, (Integral, Form)):
             return self.__add__(other.__neg__())
        raise TypeError("Can only subtract an Integral or Form from a Form.")

    def __neg__(self):
        # Create a new Form where each integral's integrand is negated.
        return Form([integral.__neg__() for integral in self.integrals])

    def __eq__(self, other):
        """Handles cases like `my_form == None`."""
        return Equation(self, other)

    def __req__(self, other):
        """
        Handles reverse-equals for cases like `None == my_form`. [THE FIX]
        
        This method is called by Python when `other == self` is evaluated and
        `other` does not have a specific `__eq__` method for a Form. Here,
        `other` is the left-hand side (e.g., None) and `self` is the Form object.
        """
        return Equation(other, self)

class Equation:
    def _valid_form(self, a):
        # Accept: None, Form, Integral. Treat numeric zero as "empty".
        if a is None:
            return None
        if isinstance(a, Form):
            return a
        if isinstance(a, Integral):
            return Form([a])
        # Constant is an Expression *and* a numbers.Number; handle it before numbers.Real
        if isinstance(a, Constant):
            if float(a) == 0.0:
                return None
            raise TypeError(
                "Bare Constant on an Equation side is ambiguous. "
                "Multiply by a Measure to make an Integral (e.g. Constant(c)*dx) "
                "or wrap it into your form appropriately."
            )
        # Plain numerics: only allow exact zero to mean 'no form'
        if isinstance(a, numbers.Real):
            if float(a) == 0.0:
                return None
            raise TypeError(
                "Numeric nonzero on an Equation side is not a Form. "
                "Wrap it with Constant(...) and a Measure (e.g. Constant(c)*dx)."
            )
        raise TypeError(
            f"Equation sides must be Form, Integral, Constant/0.0, or None, not {type(a)}"
        )
    def __init__(self, a:Form, L:Form):
        # Allow a side to be None, otherwise ensure it's a Form object.
        self.a = self._valid_form(a)
        self.L = self._valid_form(L)

class BoundaryCondition:
    def __init__(self, field: str, method: str, domain_tag: str, value: Callable):
        self.field = field
        m = method.lower()
        if m not in ("dirichlet", "neumann"):
            raise ValueError("BC method must be 'dirichlet' or 'neumann'")
        self.method = m
        self.domain_tag = domain_tag
        self.value = value

def assemble_form(equation: Equation, dof_handler, bcs=[], quad_order=None, 
                  assembler_hooks=None,backend='jit', **kwargs):
    """
    High-level function to assemble a weak form into a matrix and vector.
    """
    if not isinstance(equation, Equation):
        raise Warning(
            "assemble_form expects a pycutfem.ufl.forms.Equation; "
            "did you accidentally write None == form? This will give just 0s"
        )
    from pycutfem.ufl.compilers import FormCompiler
    if kwargs.get('quad_degree') is not None:
        quad_order = kwargs['quad_degree']
    
    # We no longer need to preprocess the form.
    # The compiler will handle the list of integrals directly.
    compiler = FormCompiler(dof_handler, quad_order, assembler_hooks=assembler_hooks, backend=backend)
    
    # This runs the full assembly process. K and F are created, and if hooks
    # are present, compiler.ctx['scalar_results'] is populated.
    K, F = compiler.assemble(equation, bcs)

    # After assembly, check the compiler's context for scalar results.
    # If the user provided hooks and those hooks produced results, return them.
    if assembler_hooks and 'scalar_results' in compiler.ctx:
        return compiler.ctx['scalar_results']
    
    # Otherwise, return the standard system matrix and vector.
    return K, F
