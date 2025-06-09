import numpy as np

class Expression:
    """Base class for any object in a symbolic FEM expression."""
    def __add__(self, other): return Sum(self, other)
    def __radd__(self, other): return Sum(other, self)
    def __sub__(self, other): return Sub(self, other)
    def __rsub__(self, other): return Sub(other, self)
    
    def __mul__(self, other):
        """Left multiplication.  If *other* is a Measure (dx, ds, …)
        we delegate to its ``__rmul__`` so that an **Integral** node
        is created instead of a plain product."""
        from ufl.measures import Measure            # ← local to avoid cycles
        if isinstance(other, Measure):
            return other.__rmul__(self)
        if not isinstance(other, Expression):
            other = Constant(other)
        return Prod(self, other)

    def __rmul__(self, other):
        """Right-hand multiplication (“other * self”) with the same
        Measure detection logic."""
        from ufl.measures import Measure
        if isinstance(other, Measure):
            return other.__rmul__(self)
        if not isinstance(other, Expression):
            other = Constant(other)
        return Prod(other, self)

    def __truediv__(self, other):
        if not isinstance(other, Expression): other = Constant(other)
        return Div(self, other)

    def __neg__(self): return Prod(Constant(-1.0), self)
    def restrict(self, domain_tag): return Restriction(self, domain_tag)
    def side(self, s: str):            # ‘+’ or ‘-’
        return Side(self, s)

class Function(Expression):
    def __init__(self, fs, name=""): self.function_space = fs; self.name = name

class TrialFunction(Function): pass
class TestFunction(Function): pass
class Constant(Expression):
    def __init__(self, value): self.value = np.asarray(value)

class FacetNormal(Expression): pass
class Sum(Expression):
    def __init__(self, a, b): self.a, self.b = a, b
class Sub(Expression):
    def __init__(self, a, b): self.a, self.b = a, b
class Prod(Expression):
    def __init__(self, a, b): self.a, self.b = a, b
class Div(Expression):
    def __init__(self, a, b): self.a, self.b = a, b
class Grad(Expression):
    def __init__(self, operand): self.operand = operand
class DivOperation(Expression): # Renamed to avoid conflict with Div expression
    def __init__(self, operand): self.operand = operand
class Inner(Expression):
    def __init__(self, a, b): self.a, self.b = a, b
class Outer(Expression):
    def __init__(self, a, b): self.a, self.b = a, b
class Jump(Expression):
    def __init__(self, v, n=None): self.v, self.n = v, n
class Avg(Expression):
    def __init__(self, v): self.v = v
class Restriction(Expression):
    def __init__(self, f, domain_tag): self.f, self.domain_tag = f, domain_tag
class Side(Expression):
    def __init__(self, f, side): self.f, self.side = f, side
class Integral(Expression):
    def __init__(self, integrand, measure):
        self.integrand, self.measure = integrand, measure
    def __add__(self, other): 
        from .forms import Form
        return Form([self, other])
    def __eq__(self, other):
        from .forms import Form, Equation
        return Equation(Form([self]), other)


# --- Helper functions to create operator instances ---
def grad(v): return Grad(v)
def div(v): return DivOperation(v)
def inner(a, b): return Inner(a, b)
def dot(a, b): return Inner(a, b)
def outer(a, b): return Outer(a, b)
def jump(v, n=None): return Jump(v, n)
def avg(v): return Avg(v)
