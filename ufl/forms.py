from ufl.expressions import Expression, Integral, Sum, Sub, Prod, Constant
from typing import Callable
# from .compilers import FormCompiler

class Form(Expression):
    """Represents the sum of several integrals that make up one side of a weak form."""
    def __init__(self, integrals: list):
        self.integrals = []
        for term in integrals:
            if isinstance(term, Form): self.integrals.extend(term.integrals)
            else: self.integrals.append(term)

    def __add__(self, other):
        if isinstance(other, Integral):          # a + I
            return Form(self.integrals + [other])
        if isinstance(other, Form):              # a + (b + c + …)
            return Form(self.integrals + other.integrals)
        raise TypeError(f"Can only add an Integral or Form to a Form, not {type(other)}")

    def __sub__(self, other):
        def _negate(integral):
             integral.integrand = Prod(Constant(-1.0), integral.integrand)
             return integral
        if isinstance(other, Integral):
            return Form(self.integrals + [_negate(other)])
        if isinstance(other, Form):
            return Form(self.integrals + [_negate(i) for i in other.integrals])
        raise TypeError("Can only subtract an Integral or Form from a Form.")
    def __neg__(self):
        from ufl.expressions import Constant, Prod
        new = []
        for I in self.integrals:
            new.append(Integral(Prod(Constant(-1.0), I.integrand), I.measure))
        return Form(new)


    def __eq__(self, other): return Equation(self, other)

class Equation:
    def __init__(self, a, L):
        self.a = a if isinstance(a, Form) else Form([a])
        self.L = L if isinstance(L, Form) else Form([L])


class BoundaryCondition:
    """
    Defines a boundary condition for a specific field.

    Attributes:
        field (str): The name of the field this BC applies to (e.g., 'velocity_x').
        method (str): The type of BC, e.g., 'dirichlet'.
        domain_tag (str): The tag on the mesh boundary where this BC is applied.
        value (Callable): A function `lambda x, y: value` that gives the BC value.
    """
    def __init__(self, field: str, method: str, domain_tag: str, value: Callable):
        self.field = field
        m = method.lower()
        if m not in ("dirichlet", "neumann"):
            raise ValueError("BC method must be 'dirichlet' or 'neumann'")
        self.method = m
        self.domain_tag = domain_tag
        self.value = value

def assemble_form(equation: Equation, dof_handler, bcs=[], quad_order=None, 
                  assembler_hooks=None, solution_vector=None):
    """
    High-level function to assemble a weak form into a matrix and vector.
    """
    # Local import to avoid circular dependencies
    from ufl.compilers import FormCompiler
    
    # Create the compiler instance
    compiler = FormCompiler(dof_handler, quad_order, assembler_hooks=assembler_hooks)
    
    # --- NEW: Set the solution_vector context if it's provided ---
    # This context will be available to all visitor methods.
    if solution_vector is not None:
        compiler.ctx['solution_vector'] = solution_vector
        
    # The compiler is now configured and ready to assemble.
    return compiler.assemble(equation, bcs)
