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
    def find_first(self, criteria):
        """Recursively search for the first node in the expression tree that meets a criteria."""
        if criteria(self):
            return self

        # Recurse through operands
        if hasattr(self, 'a') and hasattr(self, 'b'):
            found = self.a.find_first(criteria)
            if found: return found
            found = self.b.find_first(criteria)
            if found: return found
        elif hasattr(self, 'operand'):
            return self.operand.find_first(criteria)
        elif hasattr(self, 'v'): # For Avg, Jump
            return self.v.find_first(criteria)
        elif hasattr(self, 'f'): # For Restriction, Side
            return self.f.find_first(criteria)
        
        return None

class Function(Expression):
    def __init__(self, field_name, name=""): 
        self.field_name = field_name; self.name = name

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
    def __init__(self, operand):
        self.operand = operand
    def __getitem__(self, index):
        # This allows syntax like grad(u)[0] for the x-component
        return Derivative(self.operand, index)

class Derivative(Expression):
    """Represents the derivative of a function with respect to a specific coordinate."""
    def __init__(self, f, component_index):
        self.f = f
        self.component_index = component_index

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
def div(v):
    """Computes the divergence of a vector field represented by a tuple of functions."""
    if isinstance(v, (list, tuple)) and len(v) == 2:
        # div(u) -> du_x/dx + du_y/dy
        return Derivative(v[0], 0) + Derivative(v[1], 1)
    raise TypeError("div() is only implemented for 2D vector fields (list/tuple of two functions).")
def inner(a, b): return Inner(a, b)
def dot(a, b): return Inner(a, b)
def outer(a, b): return Outer(a, b)
def jump(v, n=None): return Jump(v, n)
def avg(v): return Avg(v)
