import numpy as np
from typing import Callable


class Expression:
    """Base class for any object in a symbolic FEM expression."""
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __add__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)
        return Sum(self, other)

    def __radd__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)
        return Sum(other, self)

    def __sub__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)
        return Sub(self, other)

    def __rsub__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)
        return Sub(other, self)

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
    """
    Represents a single-field function. Can be a standalone data-carrier
    or a component that provides a view into a parent VectorFunction.
    """
    def __init__(self, name: str, field_name: str, dof_handler: 'DofHandler' = None,
                 parent_vector: 'VectorFunction' = None, component_index: int = None):
        self.name = name
        self.field_name = field_name
        self._dof_handler = dof_handler
        self._parent_vector = parent_vector
        self._component_index = component_index
        self.dim = 0 # always assuming scalar functions for now

        # A Function ONLY manages data if it is a STANDALONE function.
        if self._parent_vector is None and self._dof_handler is not None:
            # This is a data-carrying standalone function (e.g., 'beta' in a test)
            self._my_global_dofs = self._dof_handler.get_field_slice(self.field_name)
            self._global_dof_to_local_idx = {dof: i for i, dof in enumerate(self._my_global_dofs)}
            self._nodal_values = np.zeros(len(self._my_global_dofs))
        else:
            # This is a symbolic function (Trial/Test) OR a component of a VectorFunction.
            # It does not need to store its own data arrays.
            self._my_global_dofs = []
            self._global_dof_to_local_idx = {}
            self._nodal_values = None

    @property
    def nodal_values(self):
        """
        Smart property: Delegates to the parent vector if this is a component,
        otherwise returns its own data.
        """
        if self._parent_vector is not None:
            # Get all global dofs for this component's field
            dofs = self._parent_vector._get_dofs_for_component(self._component_index)
            return self._parent_vector.get_nodal_values(dofs)
        return self._nodal_values

    @nodal_values.setter
    def nodal_values(self, values):
        if self._parent_vector is not None:
            dofs = self._parent_vector._get_dofs_for_component(self._component_index)
            self._parent_vector.set_nodal_values(dofs, values)
        else:
            self._nodal_values[:] = values

    def set_values_from_function(self, func: Callable[[float, float], float]):
        """Populates nodal values by evaluating a function at each DoF's coordinate."""
        global_dofs = self._dof_handler.get_field_dofs_on_nodes(self.field_name)
        coords = self._dof_handler.get_dof_coords(self.field_name)
        values = np.apply_along_axis(lambda c: func(c[0], c[1]), 1, coords).ravel()
        self.set_nodal_values(global_dofs, values)

    def get_nodal_values(self, global_dofs: np.ndarray) -> np.ndarray:
        """Gets values for a specific list of global DOFs from this function's data."""
        if self._parent_vector is not None:
            return self._parent_vector.get_nodal_values(global_dofs)
        
        local_indices = np.array([self._global_dof_to_local_idx[gd] for gd in global_dofs])
        return self._nodal_values[local_indices]

    def set_nodal_values(self, global_dofs: np.ndarray, values: np.ndarray):
        """Sets values for a specific list of global DOFs in this function's data."""
        if self._parent_vector is not None:
            self._parent_vector.set_nodal_values(global_dofs, values)
            return
            
        local_indices = np.array([self._global_dof_to_local_idx[gd] for gd in global_dofs])
        self._nodal_values[local_indices] = values

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', field='{self.field_name}')"
    def is_vector_component_basis(self):
        """Componenet of a Vector is also scalar"""
        return False

# In your ufl/expressions.py file

class VectorFunction(Expression):
    def __init__(self, name: str, field_names: list[str], dof_handler: 'DofHandler'=None):
        self.name = name
        self.field_names = field_names
        self._dof_handler = dof_handler
        self.dim = 1 # always a vector function, this dim is a tensor dimension not the spatial dimension
        
        self._my_global_dofs = np.concatenate([dof_handler.get_field_slice(fn) for fn in field_names])
        self._global_dof_to_local_idx = {dof: i for i, dof in enumerate(self._my_global_dofs)}
        
        self.nodal_values = np.zeros(len(self._my_global_dofs))
        self.components = [
            Function(f"{name}_{fn}", fn, dof_handler, parent_vector=self, component_index=i)
            for i, fn in enumerate(self.field_names)
        ]

    def _get_dofs_for_component(self, component_index: int) -> np.ndarray:
        field_name = self.field_names[component_index]
        return self._dof_handler.get_field_dofs_on_nodes(field_name)

    def set_values_from_function(self, func: Callable[[float, float], np.ndarray]):
        for i, field_name in enumerate(self.field_names):
            global_dofs = self._dof_handler.get_field_dofs_on_nodes(field_name)
            coords = self._dof_handler.get_dof_coords(field_name)
            values = np.apply_along_axis(lambda c: func(c[0], c[1])[i], 1, coords).ravel()
            self.set_nodal_values(global_dofs, values)

    def get_nodal_values(self, global_dofs: np.ndarray) -> np.ndarray:
        local_indices = np.array([self._global_dof_to_local_idx[gd] for gd in global_dofs])
        return self.nodal_values[local_indices]

    def set_nodal_values(self, global_dofs: np.ndarray, values: np.ndarray):
        local_indices = np.array([self._global_dof_to_local_idx[gd] for gd in global_dofs])
        self.nodal_values[local_indices] = values

    def __repr__(self):
        return f"VectorFunction(name='{self.name}', fields={self.field_names})"

    def __getitem__(self, i):
        """Allows accessing components like my_vec_func[0]."""
        return self.components[i]

    def __iter__(self):
        """Allows iterating over components."""
        return iter(self.components)
    def is_vector_component_basis(self):
        """Returns True if this VectorFunction is a component of a vector function."""
        return True

class NodalFunction(Expression):
    def __init__(self, field_name: str):
        self.field_name = field_name    # key in mesh.node_data
    def __repr__(self): return f"NodalFunction({self.field_name!r})"

class TrialFunction(Function):
    def __init__(self, field_name: str):
        # A TrialFunction is purely symbolic. It has no dof_handler or data.
        # Its name and field_name are the same.
        super().__init__(name=field_name, field_name=field_name)
        self.dim = 0
    def is_vector_component_basis(self):
        """Returns True if this TrialFunction is a component of a vector function."""
        return False

class TestFunction(Function):
    def __init__(self, field_name: str):
        # A TestFunction is purely symbolic.
        super().__init__(name=field_name, field_name=field_name)
        self.dim = 0  # Assuming scalar test functions for now
    def is_vector_component_basis(self):
        """Returns True if this TestFunction is a component of a vector function."""
        return False

class Constant(Expression):
    def __init__(self, value, dim=0): self.value = value; self.dim = dim
    def __repr__(self):
        return f"Constant({self.value})"

class VectorTrialFunction(Expression):
    def __init__(self, space): # space: FunctionSpace
        self.space = space
        self.components = [TrialFunction(fn) for fn in space.field_names]
        self.dim = 1
    def __repr__(self):
        return f"VectorTrialFunction(space='{self.space.name}')"
    def __getitem__(self, i): return self.components[i]
    def __iter__(self): return iter(self.components)
    def is_vector_component_basis(self):
        """Returns True if this VectorTrialFunction is a component of a vector function."""
        return True

class VectorTestFunction(Expression):
    def __init__(self, space): # space: FunctionSpace
        self.space = space
        self.components = [TestFunction(fn) for fn in space.field_names]
        self.dim = 1 
    def __repr__(self):
        return f"VectorTestFunction(space='{self.space.name}')"
    def __getitem__(self, i): return self.components[i]
    def __iter__(self): return iter(self.components)
    def is_vector_component_basis(self):
        """Returns True if this VectorTestFunction is a component of a vector function."""
        return True

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

class Derivative(Expression):
    def __init__(self, f, component_index):
        self.f, self.component_index = f, component_index
    def __repr__(self): return f"Derivative({self.f!r}, {self.component_index})"

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
    def __neg__(self):
        from ufl.expressions import Constant, Prod
        return Integral(Prod(Constant(-1.0), self.integrand), self.measure)
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
