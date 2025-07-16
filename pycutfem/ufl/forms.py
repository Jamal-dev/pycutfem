from pycutfem.ufl.expressions import Expression, Integral, Sum, Sub, Prod, Constant
from typing import Callable, List, Dict

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

    def __sub__(self, other):
        # Use the __neg__ method to create negated versions of the terms to be subtracted.
        if isinstance(other, (Integral, Form)):
             return self.__add__(other.__neg__())
        raise TypeError("Can only subtract an Integral or Form from a Form.")

    def __neg__(self):
        # Create a new Form where each integral's integrand is negated.
        return Form([integral.__neg__() for integral in self.integrals])

    def __eq__(self, other): return Equation(self, other)

class Equation:
    def __init__(self, a, L):
        # Ensure both sides of the equation are Form objects
        self.a = a if isinstance(a, Form) else Form([a])
        self.L = L if isinstance(L, Form) else Form([L])

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
    from pycutfem.ufl.compilers import FormCompiler
    if kwargs.get('quad_degree') is not None:
        quad_order = kwargs['quad_degree']
    
    # We no longer need to preprocess the form.
    # The compiler will handle the list of integrals directly.
    compiler = FormCompiler(dof_handler, quad_order, assembler_hooks=assembler_hooks, backend=backend)
    
    # This runs the full assembly process. K and F are created, and if hooks
    # are present, compiler.ctx['scalar_results'] is populated.
    K, F = compiler.assemble(equation, bcs)

    # --- FIXED RETURN LOGIC ---
    # After assembly, check the compiler's context for scalar results.
    # If the user provided hooks and those hooks produced results, return them.
    if assembler_hooks and 'scalar_results' in compiler.ctx:
        return compiler.ctx['scalar_results']
    
    # Otherwise, return the standard system matrix and vector.
    return K, F
