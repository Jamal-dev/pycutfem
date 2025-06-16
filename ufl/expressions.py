import numpy as np

class Expression:
    """Base class for any object in a symbolic FEM expression."""
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __add__(self, other): return Sum(self, other)
    def __radd__(self, other): return Sum(other, self)
    def __sub__(self, other): return Sub(self, other)
    def __rsub__(self, other): return Sub(other, self)

    def __mul__(self, other):
        """Left multiplication. If *other* is a Measure (dx, ds, …), create an Integral."""
        from ufl.measures import Measure
        if isinstance(other, Measure):
            return other.__rmul__(self)
        if not isinstance(other, Expression):
            other = Constant(other)
        return Prod(self, other)

    def __rmul__(self, other):
        """Right-hand multiplication."""
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
    def side(self, s: str): return Side(self, s)

    def find_first(self, criteria):
        """Recursively search for the first node in the expression tree that meets a criteria."""
        if criteria(self): return self
        if hasattr(self, 'a') and hasattr(self, 'b'):
            found = self.a.find_first(criteria)
            if found: return found
            found = self.b.find_first(criteria)
            if found: return found
        elif hasattr(self, 'operand'):
            if self.operand is None: return None
            return self.operand.find_first(criteria)
        elif hasattr(self, 'f'):
            return self.f.find_first(criteria)
        elif hasattr(self, 'u_pos'): # For Jump
            found = self.u_pos.find_first(criteria)
            if found: return found
            found = self.u_neg.find_first(criteria)
            if found: return found
        return None

class Function(Expression):
    def __init__(self, field_name, name="",nodal_values = None):
        self.field_name = field_name; self.name = name
        self.nodal_values = nodal_values
    def __repr__(self):
        return f"{self.__class__.__name__}(field='{self.field_name}')"

# In your ufl/expressions.py file

class VectorFunction(Expression):
    """
    Represents a known, data-carrying vector-valued finite element function.
    """
    def __init__(self, name: str, field_names: list[str], nodal_values=None):
        """
        Args:
            name (str): A name for the vector function space, e.g., "velocity".
            field_names (list[str]): A list of the component field names, e.g., ['ux', 'uy'].
            nodal_values (np.ndarray, optional): The array of nodal data, with a
                shape like (n_total_dofs, n_components).
        """
        self.name = name
        self.field_names = field_names
        self.nodal_values = nodal_values
        
        # Create a list of scalar Function objects for polymorphism.
        # These components are primarily for structural compatibility and do not
        # hold data themselves; the data is in self.nodal_values.
        self.components = [Function(fn) for fn in self.field_names]

    def __repr__(self):
        return f"VectorFunction(name='{self.name}', fields={self.field_names})"

    def __getitem__(self, i):
        """Allows accessing components like my_vec_func[0]."""
        return self.components[i]

    def __iter__(self):
        """Allows iterating over components."""
        return iter(self.components)

class NodalFunction(Expression):
    def __init__(self, field_name: str):
        self.field_name = field_name    # key in mesh.node_data
    def __repr__(self): return f"NodalFunction({self.field_name!r})"

class TrialFunction(Function): pass
class TestFunction(Function): pass

class Constant(Expression):
    def __init__(self, value, dim=0): self.value = value; self.dim = dim
    def __repr__(self):
        return f"Constant({self.value})"

class VectorTrialFunction(Expression):
    def __init__(self, space): # space: FunctionSpace
        self.space = space
        self.components = [TrialFunction(fn) for fn in space.field_names]
    def __repr__(self):
        return f"VectorTrialFunction(space='{self.space.name}')"
    def __getitem__(self, i): return self.components[i]
    def __iter__(self): return iter(self.components)

class VectorTestFunction(Expression):
    def __init__(self, space): # space: FunctionSpace
        self.space = space
        self.components = [TestFunction(fn) for fn in space.field_names]
    def __repr__(self):
        return f"VectorTestFunction(space='{self.space.name}')"
    def __getitem__(self, i): return self.components[i]
    def __iter__(self): return iter(self.components)

class ElementWiseConstant(Expression):
    def __init__(self, values: np.ndarray): self.values = values
    def __repr__(self): return f"ElementWiseConstant(...)"

class FacetNormal(Expression): pass

class Sum(Expression):
    def __init__(self, a, b): self.a, self.b = a, b
    def __repr__(self): return f"({self.a!r} + {self.b!r})"

class Sub(Expression):
    def __init__(self, a, b): self.a, self.b = a, b
    def __repr__(self): return f"({self.a!r} - {self.b!r})"

class Prod(Expression):
    def __init__(self, a, b): self.a, self.b = a, b
    def __repr__(self): return f"({self.a!r} * {self.b!r})"

class Div(Expression):
    def __init__(self, a, b): self.a, self.b = a, b
    def __repr__(self): return f"({self.a!r} / {self.b!r})"

class Grad(Expression):
    def __init__(self, operand): self.operand = operand
    def __repr__(self): return f"Grad({self.operand!r})"
    def __getitem__(self, index): return Derivative(self.operand, index)

class Derivative(Expression):
    def __init__(self, f, component_index):
        self.f, self.component_index = f, component_index
    def __repr__(self): return f"Derivative({self.f!r}, {self.component_index})"

class DivOperation(Expression):
    def __init__(self, operand): self.operand = operand
    def __repr__(self): return f"Div({self.operand!r})"

class Inner(Expression):
    def __init__(self, a, b): self.a, self.b = a, b
    def __repr__(self): return f"Inner({self.a!r}, {self.b!r})"

class Outer(Expression):
    def __init__(self, a, b): self.a, self.b = a, b
    def __repr__(self): return f"Outer({self.a!r}, {self.b!r})"

class Pos(Expression):
    def __init__(self, operand): self.operand = operand
    def __repr__(self): return f"Pos({self.operand!r})"
    
class Neg(Expression):
    def __init__(self, operand): self.operand = operand
    def __repr__(self): return f"Neg({self.operand!r})"

from copy import deepcopy
class Jump(Expression):
    """
    Jump( expr_pos, expr_neg )
    If expr_neg is omitted → Jump(expr) expands to  Pos(expr) – Neg(expr).
    """
    def __init__(self, u_on_pos, u_on_neg=None):
        if u_on_neg is None:
            # Automatic expansion: jump(u)  :=  pos(u) – neg(u)
            u_on_pos, u_on_neg = Pos(u_on_pos), Neg(deepcopy(u_on_pos))
        self.u_pos, self.u_neg = u_on_pos, u_on_neg

    def __repr__(self):
        return f"Jump({self.u_pos!r}, {self.u_neg!r})"

class Avg(Expression):
    def __init__(self, v): self.v = v
    def __repr__(self): return f"Avg({self.v!r})"

class Restriction(Expression):
    def __init__(self, f, domain_tag): self.f, self.domain_tag = f, domain_tag
    def __repr__(self): return f"Restriction({self.f!r}, '{self.domain_tag}')"

class Side(Expression):
    def __init__(self, f, side): self.f, self.side = f, side
    def __repr__(self): return f"Side({self.f!r}, '{self.side}')"

class Integral(Expression):
    def __init__(self, integrand, measure):
        self.integrand, self.measure = integrand, measure
    def __repr__(self): return f"Integral({self.integrand!r}, {self.measure!r})"
    def __add__(self, other):
        from .forms import Form
        return Form([self, other])
    def __eq__(self, other):
        from .forms import Form, Equation
        return Equation(Form([self]), other)

class Dot(Expression):
    def __init__(self, a, b): self.a, self.b = a, b
    def __repr__(self): return f"Dot({self.a!r}, {self.b!r})"



# --- Helper functions to create operator instances ---
def grad(v): return Grad(v)
def div(v):
    """Creates a symbolic representation of the divergence of a field."""
    return DivOperation(v)
def inner(a, b): return Inner(a, b)
def outer(a, b): return Outer(a, b)
def jump(v, n=None): return Jump(v, n)
def avg(v): return Avg(v)
def dot(a, b): return Dot(a, b)
