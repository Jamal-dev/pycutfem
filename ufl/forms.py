from ufl.expressions import Expression, Integral, Sum, Sub, Prod, Constant
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


    def __eq__(self, other): return Equation(self, other)

class Equation:
    def __init__(self, a, L):
        self.a = a if isinstance(a, Form) else Form([a])
        self.L = L if isinstance(L, Form) else Form([L])


class BoundaryCondition:
    """type: 'dirichlet' | 'neumann' · tag: mesh facet label · value: Expression"""
    def __init__(self, V, type_, tag, value):
        self.V = V
        t = type_.lower()
        if t not in ("dirichlet", "neumann"):
            raise ValueError("BC type must be Dirichlet or Neumann")
        self.type, self.tag, self.value = t, tag, value

def assemble_form(equation, function_space, mesh,bcs=[], quad_order=None):
    from ufl.compilers import FormCompiler
    compiler = FormCompiler(mesh,function_space, quad_order)
    return compiler.assemble(equation, bcs)
