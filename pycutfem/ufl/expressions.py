import numpy as np
from typing import Callable, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numbers
from matplotlib.colors import LinearSegmentedColormap
from pycutfem.plotting.triangulate import triangulate_field


custom_cmap = LinearSegmentedColormap.from_list('blue_red', ['blue', 'red'])

class Expression:
    """Base class for any object in a symbolic FEM expression."""
    is_function = False
    is_trial    = False
    is_test     = False
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
        from pycutfem.ufl.measures import Measure
        if isinstance(other, Measure):
            return other.__rmul__(self)
        if not isinstance(other, Expression):
            other = Constant(other)
        return Prod(self, other)

    def __rmul__(self, other):
        """Right-hand multiplication."""
        from pycutfem.ufl.measures import Measure
        if isinstance(other, Measure):
            return other.__rmul__(self)
        if not isinstance(other, Expression):
            other = Constant(other)
        return Prod(other, self)

    def __truediv__(self, other):
        if not isinstance(other, Expression): other = Constant(other)
        return Div(self, other)

    def __neg__(self): return Prod(Constant(-1.0), self)
    def __hash__(self):
        return id(self)
    def __deepcopy__(self, memo):
        return self
    def __copy__(self):
        return self                  # shallow copy ➜ same object
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
    is_function = True
    is_trial = False
    is_test = False
    def __init__(self, name: str, field_name: str, dof_handler: 'DofHandler' = None,
                 parent_vector: 'VectorFunction' = None, component_index: int = None):
        self.name = name
        self.field_name = field_name
        self._dof_handler = dof_handler
        self._parent_vector = parent_vector
        self._component_index = component_index
        self.dim = 0 # always assuming scalar functions for now

        if self._parent_vector is None and self._dof_handler is not None:
            # --- stand‑alone data carrier -----------------------------------
            self._g_dofs = np.asarray(self._dof_handler.get_field_slice(field_name), dtype=int)
            self._g2l: Dict[int, int] = {gd: i for i, gd in enumerate(self._g_dofs)}
            self._values = np.zeros_like(self._g_dofs, dtype=float)
        else:
            self._g_dofs = np.empty(0, dtype=int); self._values = None; self._g2l = {}

    @property
    def nodal_values(self):
        if self._parent_vector is not None:
            return self._parent_vector.nodal_values_component(self._component_index)
        return self._values

    @nodal_values.setter
    def nodal_values(self, v):
        if self._parent_vector is not None:
            self._parent_vector.set_component_values(self._component_index, v); return
        self._values[:] = v

    def set_values_from_function(self, func: Callable[[float, float], float]):
        """Populates nodal values by evaluating a function at each DoF's coordinate."""
        global_dofs = self._dof_handler.get_field_dofs_on_nodes(self.field_name)
        coords = self._dof_handler.get_dof_coords(self.field_name)
        values = np.apply_along_axis(lambda c: func(c[0], c[1]), 1, coords).ravel()
        self.set_nodal_values(global_dofs, values)

    def get_nodal_values(self, global_dofs: np.ndarray) -> np.ndarray:
        """
        For a given array of global DOF indices (e.g., from an element),
        return a corresponding array of values. The returned array has the
        same size as the input, with zeros for DOFs not owned by this Function.
        """
        if self._parent_vector is not None:
            # Delegate to the parent, which manages the full data vector.
            return self._parent_vector.get_nodal_values(global_dofs)

        # Standalone function: build the padded vector manually.
        out = np.zeros(len(global_dofs), dtype=float)
        for i, gdof in enumerate(global_dofs):
            if gdof in self._g2l:
                local_idx = self._g2l[gdof]
                out[i] = self._values[local_idx]
        return out

    def set_nodal_values(self, global_dofs: np.ndarray, vals: np.ndarray):
        if self._parent_vector is not None:
            self._parent_vector.set_nodal_values(global_dofs, vals); return
        
        for gdof, val in zip(global_dofs, vals):
            if gdof in self._g2l:
                self._values[self._g2l[gdof]] = val
    
    def padded_values(self, global_dofs: np.ndarray) -> np.ndarray:
        """Component-only, zero-padded to element size."""
        if self._parent_vector is None:
            return self.get_nodal_values(global_dofs)          # stand-alone scalar
        # component view into VectorFunction
        parent_vals = self._parent_vector.get_nodal_values(global_dofs)
        fld_dofs = set(self._dof_handler.get_field_slice(self.field_name))
        mask = np.isin(global_dofs, list(fld_dofs))
        return parent_vals * mask

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', field='{self.field_name}')"
    
    def plot(self, **kwargs):
        """ Function.plot(**kwargs) -> None
        Creates a 2D filled contour plot of the scalar function.
        """
        if self._dof_handler is None:
            raise RuntimeError("Cannot plot a function without an associated DofHandler.")
        mesh = self._dof_handler.fe_map.get(self.field_name)
        if mesh is None:
            raise RuntimeError(f"Field '{self.field_name}' not found in DofHandler's fe_map.")
        
       
        z = self.nodal_values

            
        triangulation = triangulate_field(mesh, self._dof_handler, self.field_name)
        fig, ax = plt.subplots()
        title = kwargs.pop('title', f'Scalar Field: {self.name}')
        plot_kwargs = {'cmap': 'viridis', 'levels': 15}
        plot_kwargs.update(kwargs)
        
        contour = ax.tricontourf(triangulation, z, **plot_kwargs)
        fig.colorbar(contour, ax=ax, label='Value')
        ax.set_title(title)
        ax.set_xlabel('X coordinate'); ax.set_ylabel('Y coordinate')
        ax.set_aspect('equal', adjustable='box')
        ax.tricontour(triangulation, z, colors='k', linewidths=0.5, levels=plot_kwargs['levels'])
        plt.show()

# In your ufl/expressions.py file

class VectorFunction(Expression):
    is_function = True
    is_trial = False
    is_test = False
    def __init__(self, name: str, field_names: list[str], dof_handler: 'DofHandler'=None):
        super().__init__()
        self.name = name; self.field_names = field_names; self._dh = dof_handler
        self.dim = 1
        
        # This function holds the data for multiple fields.
        g_dofs_list = [dof_handler.get_field_slice(f) for f in field_names]
        self._g_dofs = np.concatenate(g_dofs_list)
        self._g2l = {gd: i for i, gd in enumerate(self._g_dofs)}
        self.nodal_values = np.zeros(len(self._g_dofs), dtype=float)
        self._dof_handler = dof_handler
        
        self.components = [Function(f"{name}_{f}", f, dof_handler,
                                    parent_vector=self, component_index=i)
                           for i, f in enumerate(field_names)]
        

    
    @property
    def shape(self):
        """Returns the shape of the vector function"""
        return self.nodal_values.shape
    

    def nodal_values_component(self, idx: int):
        s = self._dh.get_field_slice(self.field_names[idx])
        return self.nodal_values[[self._g2l[d] for d in s if d in self._g2l]]

    def set_component_values(self, idx: int, vals):
        s = self._dh.get_field_slice(self.field_names[idx])
        for i, gdof in enumerate(s):
            if gdof in self._g2l:
                self.nodal_values[self._g2l[gdof]] = vals[i]

    def get_nodal_values(self, global_dofs: np.ndarray) -> np.ndarray:
        """
        For a given array of global DOF indices (e.g., from an element),
        return a corresponding array of values. The returned array has the
        same size as the input, with zeros for DOFs not owned by this VectorFunction.
        """
        out = np.zeros(len(global_dofs), dtype=float)
        for i, gdof in enumerate(global_dofs):
            if gdof in self._g2l:
                local_idx = self._g2l[gdof]
                out[i] = self.nodal_values[local_idx]
        return out

    def set_nodal_values(self, global_dofs: np.ndarray, vals: np.ndarray):
        for gdof, val in zip(global_dofs, vals):
            if gdof in self._g2l:
                self.nodal_values[self._g2l[gdof]] = val

    def set_values_from_function(self, func: Callable[[float, float], np.ndarray]):
        for i, field_name in enumerate(self.field_names):
            # Get coords for this specific field
            g_slice = self._dh.get_field_slice(field_name)        # full slice
            coords = self._dh.get_dof_coords(field_name)
            if isinstance(func, list):
                component_func = func[i]
                vals = np.array([component_func(*xy) for xy in coords])
            else:
                vals = np.array([func(*xy)[i] for xy in coords])

            self.set_nodal_values(g_slice, vals)                  # store all


    def __repr__(self):
        return f"VectorFunction(name='{self.name}', fields={self.field_names})"

    def __getitem__(self, i):
        """Allows accessing components like my_vec_func[0]."""
        return self.components[i]

    def __iter__(self):
        """Allows iterating over components."""
        return iter(self.components)
    def plot(self, field: str = None, kind: str = 'contour', **kwargs):
        """
        Visualizes the vector function.

        Requires matplotlib to be installed.

        Args:
            field (str, optional): The name of a specific component to plot
                (e.g., 'ux'). If None, the behavior is determined by 'kind'.
                Defaults to None.
            kind (str, optional): The type of plot. Can be 'contour' to show
                scalar components or 'quiver' to show vector arrows.
                This is ignored if 'field' is specified. Defaults to 'contour'.
            **kwargs: Additional keyword arguments passed to the underlying
                      matplotlib plot function (e.g., title='My Plot').
        """
        if field is not None:
            found = False
            for comp in self.components:
                if comp.field_name == field:
                    print(f"Plotting component: '{field}'")
                    comp.plot(**kwargs)
                    found = True
                    break
            if not found:
                raise ValueError(f"Field '{field}' not found in VectorFunction components: {self.field_names}")
            return

        if kind == 'contour':
            print(f"Plotting all components of '{self.name}' as separate contour plots...")
            for i, comp in enumerate(self.components):
                # Pass a default title for each component if none is provided
                comp_title = kwargs.get('title', f'Component: {comp.name}')
                # Create a copy of kwargs to avoid modifying it in the loop
                comp_kwargs = kwargs.copy()
                comp_kwargs['title'] = comp_title
                comp.plot(**comp_kwargs)
        elif kind == 'quiver':
            self._plot_quiver(**kwargs)
        elif kind == 'streamlines' or kind == 'streamline':
            self._plot_streamlines(**kwargs)
        else:
            raise ValueError(f"Unsupported plot kind '{kind}'. Choose 'contour' or 'quiver'.")

    def _plot_quiver(self, *,          # <-- keep the same public signature
                 stride: int = 1,   # decimate arrows to avoid clutter
                 background: bool | str = "magnitude",
                 **kwargs):
        """
        Quiver plot that **always** matches the field’s DOF layout.

        Parameters
        ----------
        stride : int, optional
            Use every *stride*-th DOF for the arrows. 1 → all nodes (default 1).
        background : bool | str, optional
            ``False`` – no colour wash behind arrows.
            ``True``  – filled contour of |u|.
            ``"ux"``/``"uy"``/``"p"`` … any scalar component name → plot that
            component instead of the magnitude.  (Default ``"magnitude"`` which is
            equivalent to ``True``.)
        **kwargs :
            Passed straight to ``ax.quiver`` (e.g. `color='k'`, `scale=50` …).
        """
        if self._dof_handler is None:
            raise RuntimeError("VectorFunction needs an attached DofHandler.")
        if len(self.field_names) != 2:
            raise NotImplementedError("Quiver is implemented only for 2-D vectors.")

        fld_u, fld_v = self.field_names
        dh   = self._dof_handler
        mesh = dh.fe_map[fld_u]

        # --- coordinates + values (1-to-1 with DOFs) ---------------------
        coords = dh.get_dof_coords(fld_u)                                
        x, y   = coords[:, 0], coords[:, 1]
        u_vals = self.components[0].nodal_values
        v_vals = self.components[1].nodal_values
        if len(u_vals) != len(x):
            raise ValueError("Mismatch between nodal values and coordinate length.")

        # optional arrow decimation
        sl = slice(None, None, stride)
        x_q, y_q, u_q, v_q = x[sl], y[sl], u_vals[sl], v_vals[sl]

        # --- background colour-wash --------------------------------------
        tri_obj = triangulate_field(mesh, dh, fld_u)
        fig, ax = plt.subplots(figsize=(8, 8))

        if background:
            if background is True or background == "magnitude":
                scalar, label = np.hypot(u_vals, v_vals), "|u|"
            else:  # any scalar component name
                comp = next((c for c in self.components if c.field_name == background), None)
                if comp is None:
                    raise ValueError(f"background='{background}' not a component.")
                scalar, label = comp.nodal_values, background
            levels = kwargs.pop("levels", 15)
            cmap   = kwargs.pop("cmap",   "viridis")
            tcf = ax.tricontourf(tri_obj, scalar, levels=levels, cmap=cmap)
            fig.colorbar(tcf, ax=ax, label=label)

        # --- arrows -------------------------------------------------------
        q_defaults = dict(angles="xy", scale_units="xy", scale=None, color="k")
        title = kwargs.pop('title', f"Vector field: {self.name}")
        q_defaults.update(kwargs)
        ax.quiver(x_q, y_q, u_q, v_q, **q_defaults)

        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        dx = np.ptp(x)    # instead of x.ptp()
        dy = np.ptp(y)    # instead of y.ptp()
        ax.set_xlim(x.min() - 0.05 * dx, x.max() + 0.05 * dx)
        ax.set_ylim(y.min() - 0.05 * dy, y.max() + 0.05 * dy)
        plt.show()
    
    # ------------------------------------------------------------------
    # NEW: stream-line plot
    # ------------------------------------------------------------------
    def _plot_streamlines(
            self, *,
            grid: int = 200,           # samples per axis for interpolation grid
            density: float = 1.3,      # passed straight to plt.streamplot
            linewidth: float = 1.0,
            cmap: str = "turbo",
            background: bool | str = "magnitude",
            **kwargs):
        """
        Draw stream-lines of the 2-D vector field.

        Parameters
        ----------
        grid        : int
            Resolution of the regular (x,y) grid to which the nodal values are
            interpolated (default 200×200).
        density     : float
            Stream-line density parameter as in ``plt.streamplot``.
        linewidth   : float
            Line width for the stream-lines.
        cmap        : str
            Colour map for line colouring by |u|.
        background  : see ``_plot_quiver`` – allows a filled contour behind
            the stream-lines; use ``False`` for none.
        **kwargs    :
            Extra keyword args go to ``ax.streamplot`` (e.g. arrowsize=1.5).
        """
        import matplotlib.pyplot as plt
        from matplotlib.tri import LinearTriInterpolator

        if self._dof_handler is None or len(self.field_names) != 2:
            raise RuntimeError("Stream-line plot needs a 2-D VectorFunction attached to a DofHandler.")

        fld_u, fld_v = self.field_names
        dh   = self._dof_handler
        mesh = dh.fe_map[fld_u]

        # 1. nodal coordinates and values (exactly as in quiver)
        coords = dh.get_dof_coords(fld_u)
        x, y   = coords[:, 0], coords[:, 1]
        u_vals = self.components[0].nodal_values
        v_vals = self.components[1].nodal_values

        # 2. interpolate to a regular grid so that streamplot can integrate
        tri      = triangulate_field(mesh, dh, fld_u)            # existing helper
        interp_u = LinearTriInterpolator(tri, u_vals)
        interp_v = LinearTriInterpolator(tri, v_vals)

        xi = np.linspace(x.min(), x.max(), grid)
        yi = np.linspace(y.min(), y.max(), grid)
        X, Y = np.meshgrid(xi, yi)
        U = interp_u(X, Y)
        V = interp_v(X, Y)

        # mask out triangles lying outside the domain (Finite-Element holes)
        mask = np.isnan(U) | np.isnan(V)
        U[mask] = 0.0; V[mask] = 0.0

        # 3. plotting ---------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 8))
        title     = kwargs.pop("title", f"Stream-lines: {self.name}") 
        # optional background colour wash
        if background:
            if background is True or background == "magnitude":
                scalar, label = np.hypot(u_vals, v_vals), "|u|"
            else:
                comp = next((c for c in self.components if c.field_name == background), None)
                if comp is None:
                    raise ValueError(f"background='{background}' not a component.")
                scalar, label = comp.nodal_values, background
            tri_obj = triangulate_field(mesh, dh, fld_u)
            tcf = ax.tricontourf(tri_obj, scalar, levels=20, cmap=cmap, alpha=0.8)
            fig.colorbar(tcf, ax=ax, label=label)

        # stream-lines coloured by speed
        speed = np.hypot(U, V)
        strm = ax.streamplot(
            X, Y, U, V,
            density=density,
            linewidth=linewidth,
            color=speed,
            cmap=cmap,
            **kwargs
        )
        fig.colorbar(strm.lines, ax=ax, label="|u|")

        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        plt.show()

    


class NodalFunction(Expression):
    def __init__(self, field_name: str):
        self.field_name = field_name    # key in mesh.node_data
    def __repr__(self): return f"NodalFunction({self.field_name!r})"

class TrialFunction(Function):
    is_trial = True
    is_test = False
    is_function = False
    def __init__(self, field_name: str,dof_handler: 'DofHandler' = None,
                 parent_vector: 'VectorFunction' = None, component_index: int = None, name: str = None):
        # A TrialFunction is purely symbolic. It has no dof_handler or data.
        # Its name and field_name are the same.
        from pycutfem.ufl.functionspace import FunctionSpace
        if isinstance(field_name, FunctionSpace):
            # If field_name is a FunctionSpace, use its name and field_names
            name = field_name.name if name is None else name
            field_name = field_name.field_names[0]
        else:
            if name is None:
                name = f"trial_{field_name}"
        super().__init__(name=name, field_name=field_name, dof_handler=dof_handler,
                         parent_vector=parent_vector, component_index=component_index)
        self.dim = 0


class TestFunction(Function):
    is_test = True
    is_trial = False
    is_function = False
    def __init__(self, field_name: str,dof_handler: 'DofHandler' = None,
                 parent_vector: 'VectorFunction' = None, component_index: int = None, name: str = None):
        # A TestFunction is purely symbolic.
        from pycutfem.ufl.functionspace import FunctionSpace
        if isinstance(field_name, FunctionSpace):
            # If field_name is a FunctionSpace, use its name and field_names
            name = field_name.name if name is None else name
            field_name = field_name.field_names[0]
        else:
            if name is None:
                name = f"test_{field_name}"
        super().__init__(name=name, field_name=field_name, dof_handler=dof_handler,
                         parent_vector=parent_vector, component_index=component_index)
        self.dim = 0  # Assuming scalar test functions for now
 

class Constant(Expression, numbers.Number):
    def __init__(self, value, dim: int = None):
        # Call the __init__ of the next class in the Method Resolution Order (MRO)
        # This ensures all parent classes are properly initialized
        super().__init__() 
        
        self.value = value
        temp_value = np.asarray(value)
        if dim is None:
            if temp_value.ndim == 0:
                self.dim = 0
                self.value = float(value)  # Convert scalar to float
            else:
                self.dim = temp_value.ndim
                self.value = np.asarray(value, dtype=float)
        else:
            self.dim = dim
            if dim >0: self.value = np.asarray(value, dtype=float)
        self.role = 'none'
    @property
    def shape(self):
        """Returns the shape of the constant value."""
        if isinstance(self.value, np.ndarray):
            return self.value.shape
        elif isinstance(self.value, (list, tuple)):
            return (len(self.value),)
        else:
            return ()
    
    def __repr__(self):
        # We override __repr__ to show the value, which is more informative
        return f"Constant({self.value})"

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)
    def __hash__(self):
        # Scalars:   hash(1.0)        → constant regardless of identity
        # Vectors :  hash(bytes(value)) so different arrays are distinct
        if np.ndim(self.value) == 0:
            return hash(float(self.value))
        return hash(self.value.tobytes())

class VectorTrialFunction(Expression):
    is_trial = True
    is_function = False
    is_test = False
    def __init__(self, space, dof_handler: 'DofHandler'=None): # space: FunctionSpace
        self.space = space
        self.field_names = space.field_names
        self.components = [
            TrialFunction(name=f"{space.name}_{fn}",field_name = fn, dof_handler=dof_handler, 
                          parent_vector=self, 
                          component_index=i)
            for i, fn in enumerate(space.field_names)
        ]
        self.dim = 1
    def __repr__(self):
        return f"VectorTrialFunction(space='{self.space.name}')"
    def __getitem__(self, i): return self.components[i]
    def __iter__(self): return iter(self.components)


class VectorTestFunction(Expression):
    is_test = True
    is_function = False
    is_trial = False
    def __init__(self, space, dof_handler=None): # space: FunctionSpace
        self.space = space
        self.field_names = space.field_names
        self.components = [
            TestFunction(name=f"{space.name}_{fn}", 
                        field_name= fn,
                         dof_handler = dof_handler, 
                         parent_vector=self, component_index=i)
            for i, fn in enumerate(space.field_names)
        ]
        self.dim = 1 
    def __repr__(self):
        return f"VectorTestFunction(space='{self.space.name}')"
    def __getitem__(self, i): return self.components[i]
    def __iter__(self): return iter(self.components)
 

class ElementWiseConstant(Expression):
    """
    Per-element constant value that can be scalar **or** an arbitrary
    tensor.  Internally we store an `(n_elem, …)` NumPy array whose *first*
    axis enumerates the mesh elements and whose remaining axes are the
    tensor shape (empty for scalars).
    """
    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values)
        if self.values.ndim == 0:
            raise ValueError("Provide one value **per element**, not a single scalar.")
        self.tensor_shape = self.values.shape[1:]   # () for scalars

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the tensor shape (empty tuple for scalars)."""
        return self.tensor_shape

    # Handy helper – compiler calls this through the visitor method below
    def value_on_element(self, eid: int):
        return self.values[eid]

    def __repr__(self):
        shape = "scalar" if self.tensor_shape == () else f"tensor{self.tensor_shape}"
        return f"ElementWiseConstant({shape})"

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
    def __init__(self, f, *args):
        # ------------------------------------------------------------
        # Accept both   Derivative(f, dir)          (old form)
        #           and Derivative(f, ox, oy)       (new form)
        # ------------------------------------------------------------
        if len(args) == 1:                # old API
            dir = args[0]
            ox, oy = (1, 0) if dir == 0 else (0, 1)
        elif len(args) == 2:              # new API
            ox, oy = args
        else:
            raise TypeError("Derivative expects 2 or 3 arguments")

        self.f     = f
        self.order = (ox, oy)

        # -------- retro-compat field for helper functions ------------
        if ox + oy == 1:                  # first-order only
            self.component_index = 0 if ox == 1 else 1
        # higher-order: leave attribute undefined (walker never asks)

        # role flags (unchanged)
        self.is_function = getattr(f, "is_function", False)
        self.is_trial    = getattr(f, "is_trial",    False)
        self.is_test     = getattr(f, "is_test",     False)


    def __repr__(self):
        return f"Derivative({self.f!r}, {self.order[0]}, {self.order[1]})"


class Grad(Expression):
    """Gradient of a scalar expression in 2-D (returns a length-2 vector)."""

    def __init__(self, operand):
        self.operand = operand          # scalar Expression

    def __repr__(self):
        return f"Grad({self.operand!r})"

    # ------------------------------------------------------------------
    # component access:
    #
    #   Grad(u)[0]   → ∂u/∂x   = Derivative(u, 1, 0)
    #   Grad(u)[1]   → ∂u/∂y   = Derivative(u, 0, 1)
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        if index == 0:
            return Derivative(self.operand, 1, 0)
        elif index == 1:
            return Derivative(self.operand, 0, 1)
        raise IndexError("Grad supports indices 0 (x) or 1 (y)")



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
    def __init__(self, operand): 
        self.operand = operand
    @property
    def is_function(self): return getattr(self.operand, "is_function", False)
    @property
    def is_trial(self):    return getattr(self.operand, "is_trial",    False)
    @property
    def is_test(self):     return getattr(self.operand, "is_test",     False)

    def __getattr__(self, name):
        return getattr(self.operand, name)
    def __repr__(self): return f"Pos({self.operand!r})"
    
class Neg(Expression):
    def __init__(self, operand): 
        self.operand = operand
    @property
    def is_function(self): return getattr(self.operand, "is_function", False)
    @property
    def is_trial(self):    return getattr(self.operand, "is_trial",    False)
    @property
    def is_test(self):     return getattr(self.operand, "is_test",     False)

    def __getattr__(self, name):
        return getattr(self.operand, name)
    def __repr__(self): return f"Neg({self.operand!r})"

from copy import deepcopy
class Jump(Expression):
    """
    Jump( expr_pos, expr_neg )
    If expr_neg is omitted → Jump(expr) expands to  Pos(expr) – Neg(expr).
    """
    def __init__(self, u_on_pos:Union[TrialFunction,TestFunction,VectorFunction,VectorTestFunction,VectorTrialFunction]
                 , u_on_neg=None):
        if u_on_neg is None:
            # Automatic expansion: jump(u)  :=  pos(u) – neg(u)
            u_on_pos, u_on_neg = Pos(u_on_pos), Neg(deepcopy(u_on_pos))
        self.u_pos, self.u_neg = u_on_pos, u_on_neg
        self.is_function = getattr(u_on_pos, "is_function", False)
        self.is_trial    = getattr(u_on_pos, "is_trial",    False)
        self.is_test     = getattr(u_on_pos, "is_test",     False)
  

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

    def __sub__(self, other): 
        from .forms import Form 
        # Create a negated version of the other term and add it. 
        return Form([self, other.__neg__()]) 

    def __neg__(self): 
        return Integral(self.integrand.__neg__(), self.measure) 
    
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
