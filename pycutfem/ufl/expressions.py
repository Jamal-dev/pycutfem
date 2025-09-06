import numpy as np
from typing import Callable, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.tri import LinearTriInterpolator
import numbers
from matplotlib.colors import LinearSegmentedColormap
from pycutfem.plotting.triangulate import triangulate_field
from matplotlib.animation import FuncAnimation
from pycutfem.utils.bitset import BitSet


custom_cmap = LinearSegmentedColormap.from_list('blue_red', ['blue', 'red'])



class Expression:
    """Base class for any object in a symbolic FEM expression."""
    is_function = False
    is_trial    = False
    is_test     = False
    @property
    def T(self):
        """Shorthand to build a transposed view of a tensor expression."""
        return Transpose(self)
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

    def __pow__(self, other):
        """Creates a symbolic Power expression."""
        if not isinstance(other, Expression):
            other = Constant(other)
        return Power(self, other)

    def __rpow__(self, other):
        """Handles cases like `2.0 ** expr`."""
        if not isinstance(other, Expression):
            other = Constant(other)
        return Power(other, self)

    def find_first(self, criteria):
        """
        Depth-first search that stops at the first sub-expression
        satisfying *criteria*.  Guarded against cycles.
        """
        visited = set()

        def dfs(node):
            nid = id(node)
            if nid in visited:
                return None
            visited.add(nid)

            if criteria(node):
                return node

            for v in node.__dict__.values():
                if isinstance(v, Expression):
                    found = dfs(v)
                    if found:
                        return found
                elif isinstance(v, (list, tuple)):
                    for vv in v:
                        if isinstance(vv, Expression):
                            found = dfs(vv)
                            if found:
                                return found
            return None

        return dfs(self)



class Transpose(Expression):
    """Symbolic transpose (for 2-D tensors)."""
    def __init__(self, A: Expression):
        super().__init__()
        self.A = A
    def __repr__(self):
        return f"Transpose({self.A!r})"
class Power(Expression):
    def __init__(self, a, b): self.a, self.b = a, b
    def __repr__(self): return f"({self.a!r} ** {self.b!r})"
# ----------------------------------------------------------
# Small utility to build a per-DOF mask for a given field
# ----------------------------------------------------------
def _node_keep_mask_for_field(dh, field_name, mask):
    """
    Return a boolean array of length len(dh.get_field_slice(field_name))
    with True where the field's DOF is *kept* for plotting.
    mask can be: BitSet (elements), boolean array per node, or callable (x,y)->bool.
    """
    fld_slice = dh.get_field_slice(field_name)
    n = len(fld_slice)
    keep = np.ones(n, dtype=bool) if mask is None else None

    if keep is not None:
        return keep

    if callable(mask):
        xy = dh.get_dof_coords(field_name)
        keep = np.asarray(mask(xy[:, 0], xy[:, 1]), dtype=bool)
        if keep.shape != (n,):
            raise ValueError("callable mask must return shape (n_dofs_field,)")
        return keep

    if isinstance(mask, np.ndarray):
        keep = np.asarray(mask, dtype=bool)
        if keep.shape != (n,):
            raise ValueError("mask boolean array must match the field's DOF count")
        return keep

    if isinstance(mask, BitSet):
        # Mark all DOFs belonging to any *kept* element.
        keep = np.zeros(n, dtype=bool)
        gdofs = dh.get_field_slice(field_name)
        gd2l = {gd: i for i, gd in enumerate(gdofs)}
        # element_maps[field_name][eid] -> list of global dofs for that element/field
        for eid in mask.to_indices():
            for gd in dh.element_maps[field_name][eid]:
                li = gd2l.get(gd)
                if li is not None:
                    keep[li] = True
        return keep

    raise TypeError("Unsupported mask type for plotting. Use BitSet, bool array, or callable.")

def _apply_tri_mask(tri_obj, keep_nodes_bool):
    """
    Given a Triangulation and a per-node boolean 'keep' mask,
    mask out every triangle that touches a 'False' node.
    """
    tris = tri_obj.triangles
    tri_mask = np.any(~keep_nodes_bool[tris], axis=1)
    tri_obj.set_mask(tri_mask)
    return tri_mask

class Function(Expression):
    """
    Represents a single-field function. Can be a standalone data-carrier
    or a component that provides a view into a parent VectorFunction.
    """
    is_function = True
    is_trial = False
    is_test = False
    def __init__(self, name: str, field_name: str, dof_handler: 'DofHandler' = None,
                 parent_vector: 'VectorFunction' = None, 
                 component_index: int = None, side: str = ""):
        self.name = name
        self.field_name = field_name
        self._dof_handler = dof_handler
        self._parent_vector = parent_vector
        self._component_index = component_index
        self.dim = 0 # always assuming scalar functions for now
         # --- Side metadata (inherit from parent unless explicitly set) ---
        self.side = getattr(parent_vector, "side", "") if parent_vector is not None else ""
        self.parent_name = getattr(parent_vector, "name", "") if parent_vector else ""
        if side:
            self.side = side
        # Per-component side tag for codegen
        if self.side in ("+","-"):
            s = "pos" if self.side == "+" else "neg"
            self.field_sides = [s]
        elif parent_vector is not None and getattr(parent_vector, "field_sides", None) is not None and component_index is not None:
            try:
                self.field_sides = [parent_vector.field_sides[component_index]]
            except Exception:
                self.field_sides = None
        else:
            self.field_sides = None

        if self._parent_vector is None and self._dof_handler is not None:
            # --- stand‑alone data carrier -----------------------------------
            self._g_dofs = np.asarray(self._dof_handler.get_field_slice(field_name), dtype=int)
            self._g2l: Dict[int, int] = {gd: i for i, gd in enumerate(self._g_dofs)}
            self._values = np.zeros_like(self._g_dofs, dtype=float)
        else:
            self._g_dofs = np.empty(0, dtype=int); self._values = None; self._g2l = {}
    
    def copy(self):
        """Creates a new Function with a copy of the nodal values."""
        if self._parent_vector:
            # The parent VectorFunction is responsible for copying its components.
            # This logic will be handled by the parent's copy method.
            # We create a placeholder, the parent will populate it.
            return Function(self.name, self.field_name, self._dof_handler,
                            parent_vector=self._parent_vector,
                            component_index=self._component_index)

        # For a standalone function
        new_f = Function(self.name, self.field_name, self._dof_handler)
        if self.nodal_values is not None:
            new_f.nodal_values[:] = self.nodal_values.copy()
        return new_f
    
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
        """
        Plot a scalar Function. New keyword:
            mask : BitSet | ndarray[bool] | callable(x,y)->bool, optional
                Only plot within this region. Implemented by masking triangles.
        """
        if self._dof_handler is None:
            raise RuntimeError("Cannot plot a function without an associated DofHandler.")
        mask_arg = kwargs.pop('mask', None)

        dh   = self._dof_handler
        mesh = dh.fe_map.get(self.field_name)
        if mesh is None:
            raise RuntimeError(f"Field '{self.field_name}' not found in DofHandler's fe_map.")

        z = self.nodal_values
        tri = triangulate_field(mesh, dh, self.field_name)

        if mask_arg is not None:
            keep = _node_keep_mask_for_field(dh, self.field_name, mask_arg)
            tri_mask = _apply_tri_mask(tri, keep)
            if np.all(tri_mask):
                raise ValueError("Mask excludes all triangles — nothing to plot.")

        fig, ax = plt.subplots()
        title = kwargs.pop('title', f'Scalar Field: {self.name}')
        x_label = kwargs.pop('xlabel', 'X-Axis')
        y_label = kwargs.pop('ylabel', 'Y-Axis')
        plot_kwargs = {'cmap': 'viridis', 'levels': 15}
        plot_kwargs.update(kwargs)

        tcf = ax.tricontourf(tri, z, **plot_kwargs)
        fig.colorbar(tcf, ax=ax, label='Value')
        ax.set_title(title); ax.set_xlabel(x_label); ax.set_ylabel(y_label)
        ax.set_aspect('equal', 'box')
        # ax.tricontour(tri, z, colors='k', linewidths=0.5, levels=plot_kwargs['levels'])
        plt.show()

    def plot_deformed(self,
                    displacement: 'VectorFunction',
                    exaggeration: float = 1.0,
                    **kwargs):
        """
        Filled-contour plot of this scalar function on the *deformed*
        geometry.

        Parameters
        ----------
        displacement : VectorFunction
            FE solution that carries ['ux','uy'] (or any 2-D pair).
        exaggeration : float
            Multiply the physical displacement by this factor.
        **kwargs     :
            Passed straight to `tricontourf` (cmap, levels, …).
        """
        if self._dof_handler is None:
            raise RuntimeError("Function needs an attached DofHandler.")
        mesh = self._dof_handler.fe_map[self.field_name]

        # --- original triangulation & coords --------------------------------
        tri_orig = triangulate_field(mesh, self._dof_handler, self.field_name)
        node_ids = [self._dof_handler._dof_to_node_map[d][1]
                    for d in self._dof_handler.get_field_slice(self.field_name)]
        coords   = mesh.nodes_x_y_pos[node_ids]

        # --- look up displacement at those nodes ----------------------------
        fx, fy = displacement.field_names[:2]          # first two components
        disp_x = np.array([displacement.get_value_at_node(nid, fx) for nid in node_ids])
        disp_y = np.array([displacement.get_value_at_node(nid, fy) for nid in node_ids])
        coords_def = coords + exaggeration*np.column_stack((disp_x, disp_y))

        tri_def = tri.Triangulation(coords_def[:, 0], coords_def[:, 1],
                                    tri_orig.triangles)

        z = self.nodal_values
        fig, ax = plt.subplots()
        title = kwargs.pop('title', f"{self.name} (deformed)")
        plot_kw = {'cmap': 'viridis', 'levels': 15} | kwargs

        cf = ax.tricontourf(tri_def, z, **plot_kw)
        fig.colorbar(cf, ax=ax, label='Value')
        ax.tricontour(tri_def, z, levels=plot_kw['levels'],
                    colors='k', linewidths=0.4)
        ax.set(title=title, xlabel='x', ylabel='y', aspect='equal')
        plt.show()


# In your ufl/expressions.py file

class VectorFunction(Expression):
    is_function = True
    is_trial = False
    is_test = False
    def __init__(self, name: str, field_names: list[str], 
                 dof_handler: 'DofHandler'=None, side: str = ""):
        super().__init__()
        self.name = name; self.field_names = field_names; self._dh = dof_handler
        self.dim = 1
        self.parent_name = ""
        # --- Side metadata for vector-valued functions ---
        self.side = side
        if side in ("+","-",""):
            s = "pos" if side == "+" else "neg"
            self.field_sides = [s] * len(self.field_names)
        
        
        # This function holds the data for multiple fields.
        g_dofs_list = [dof_handler.get_field_slice(f) for f in field_names]
        self._g_dofs = np.concatenate(g_dofs_list)
        self._g2l = {gd: i for i, gd in enumerate(self._g_dofs)}
        self.nodal_values = np.zeros(len(self._g_dofs), dtype=float)
        self._dof_handler = dof_handler
        
        self.components = [Function(f"{name}_{f}", f, dof_handler,
                                    parent_vector=self, component_index=i, side=self.field_sides[i])
                           for i, f in enumerate(field_names)]
        

    def copy(self):
        """Creates a new VectorFunction with a copy of the nodal values."""
        # Create a new VectorFunction with the same metadata
        new_vf = VectorFunction(self.name, self.field_names, self._dof_handler)
        # Deep copy the numerical data
        if self.nodal_values is not None:
            new_vf.nodal_values[:] = self.nodal_values.copy()
        return new_vf
    
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

    def _plot_quiver(self, *,
                    stride: int = 1,
                    background: bool | str = "magnitude",
                    mask=None,
                    **kwargs):
        if self._dof_handler is None or len(self.field_names) != 2:
            raise RuntimeError("Quiver requires a 2-D VectorFunction with a DofHandler.")
        fld_u, fld_v = self.field_names
        dh   = self._dof_handler
        mesh = dh.fe_map[fld_u]

        coords = dh.get_dof_coords(fld_u)
        x, y   = coords[:, 0], coords[:, 1]
        u_vals = self.components[0].nodal_values
        v_vals = self.components[1].nodal_values

        # Node-level keep mask
        keep = _node_keep_mask_for_field(dh, fld_u, mask)
        pick = np.flatnonzero(keep)[::max(1, int(stride))]

        fig, ax = plt.subplots(figsize=(8, 8))

        # Optional background respecting the same tri-mask
        if background:
            tri = triangulate_field(mesh, dh, fld_u)
            _apply_tri_mask(tri, keep)
            if background is True or background == "magnitude":
                scalar, label = np.hypot(u_vals, v_vals), "|u|"
            else:
                comp = next((c for c in self.components if c.field_name == background), None)
                if comp is None:
                    raise ValueError(f"background='{background}' not a component.")
                scalar, label = comp.nodal_values, background
            tcf = ax.tricontourf(tri, scalar, levels=15, cmap=kwargs.get('cmap', 'viridis'))
            fig.colorbar(tcf, ax=ax, label=label)

        q = ax.quiver(x[pick], y[pick], u_vals[pick], v_vals[pick],
                    angles='xy', scale_units='xy',
                    color=kwargs.pop('color', 'k'),
                    **kwargs)
        ax.set(title=kwargs.pop('title', f"{self.name}"), xlabel='x', ylabel='y', aspect='equal')
        plt.show()
    
    # ------------------------------------------------------------------
    # NEW: stream-line plot
    # ------------------------------------------------------------------
    def _plot_streamlines(self, *,
                        grid: int = 200,
                        density: float = 1.3,
                        linewidth: float = 1.0,
                        cmap: str = "turbo",
                        background: bool | str = "magnitude",
                        mask=None,
                        **kwargs):
        from matplotlib.tri import LinearTriInterpolator

        if self._dof_handler is None or len(self.field_names) != 2:
            raise RuntimeError("Stream-lines need a 2-D VectorFunction with a DofHandler.")
        fld_u, fld_v = self.field_names
        dh   = self._dof_handler
        mesh = dh.fe_map[fld_u]

        u_vals = self.components[0].nodal_values
        v_vals = self.components[1].nodal_values

        # Triangulation + mask
        tri = triangulate_field(mesh, dh, fld_u)
        keep = _node_keep_mask_for_field(dh, fld_u, mask)
        tri_mask = _apply_tri_mask(tri, keep)
        if np.all(tri_mask):
            raise ValueError("Mask excludes all triangles — no streamlines to draw.")

        interp_u = LinearTriInterpolator(tri, u_vals)
        interp_v = LinearTriInterpolator(tri, v_vals)

        # Grid covering the (unmasked) triangulation
        x_min, x_max = tri.x.min(), tri.x.max()
        y_min, y_max = tri.y.min(), tri.y.max()
        xi = np.linspace(x_min, x_max, grid)
        yi = np.linspace(y_min, y_max, grid)
        X, Y = np.meshgrid(xi, yi)

        # Mask grid cells that fall outside the (masked) triangulation
        trifinder = tri.get_trifinder()
        grid_outside = (trifinder(X, Y) < 0)

        U = np.asarray(interp_u(X, Y))
        V = np.asarray(interp_v(X, Y))
        U = np.ma.array(U, mask=grid_outside)
        V = np.ma.array(V, mask=grid_outside)

        fig, ax = plt.subplots(figsize=(8, 8))
        title = kwargs.pop("title", f"Stream-lines: {self.name}")

        # Optional background respecting tri-mask
        if background:
            if background is True or background == "magnitude":
                scalar, label = np.hypot(u_vals, v_vals), "|u|"
            else:
                comp = next((c for c in self.components if c.field_name == background), None)
                if comp is None:
                    raise ValueError(f"background='{background}' not a component.")
                scalar, label = comp.nodal_values, background
            tcf = ax.tricontourf(tri, scalar, levels=20, cmap=cmap, alpha=0.85)
            fig.colorbar(tcf, ax=ax, label=label)

        strm = ax.streamplot(X, Y, U, V,
                            density=density,
                            linewidth=linewidth,
                            color=np.hypot(U, V),
                            cmap=cmap,
                            **kwargs)
        fig.colorbar(strm.lines, ax=ax, label="|u|")
        ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal", "box")
        plt.show()

    
    def plot_deformed(self,
                    displacement: 'VectorFunction',
                    exaggeration: float = 1.0,
                    kind: str = 'quiver',
                    background: bool | str = "magnitude",
                    stride: int = 1,
                    **kwargs):
        """
        Visualise the vector field on the *deformed* mesh.

        Parameters
        ----------
        displacement : VectorFunction
            Same-mesh solution holding the geometry change.
        exaggeration : float
            Scale factor for the displayed deformation.
        kind : {'quiver', 'streamlines'}
            Choose arrow or streamline style.
        background, stride, **kwargs
            Passed through to the underlying _plot_quiver/_plot_streamlines.
        """
        if self._dof_handler is None:
            raise RuntimeError("VectorFunction needs an attached DofHandler.")
        dh   = self._dof_handler
        mesh = dh.fe_map[self.field_names[0]]

        # -----------------------------------------------------------------------
        # One consistent list of node-ids (pick them from the *first* component)
        # -----------------------------------------------------------------------
        fld_x, fld_y = self.field_names[:2]           # e.g. 'ux', 'uy'
        g_dofs_x     = dh.get_field_slice(fld_x)      # global DOFs for ux
        node_ids     = [dh._dof_to_node_map[gd][1] for gd in g_dofs_x]

        coords       = mesh.nodes_x_y_pos[node_ids]   # (n, 2)

        # -----------------------------------------------------------------------
        # Displacement field — value per node, fetched via global-DOF indexing
        # -----------------------------------------------------------------------
        gx = np.array([dh.dof_map[fld_x][nid] for nid in node_ids])
        gy = np.array([dh.dof_map[fld_y][nid] for nid in node_ids])

        disp_x = displacement.get_nodal_values(gx)
        disp_y = displacement.get_nodal_values(gy)

        coords_def = coords + exaggeration * np.column_stack((disp_x, disp_y))
        x_d, y_d   = coords_def[:, 0], coords_def[:, 1]

        # -----------------------------------------------------------------------
        # Vector values to draw (this VectorFunction, not necessarily displacement)
        # -----------------------------------------------------------------------
        u_vals = self.get_nodal_values(gx)
        v_vals = self.get_nodal_values(gy)

        # --- branch according to kind ---------------------------------------
        if kind == 'quiver':
            # u_vals = self.components[0].nodal_values
            # v_vals = self.components[1].nodal_values
            sl = slice(None, None, stride)
            fig, ax = plt.subplots(figsize=(8, 8))

            # optional colour wash
            if background:
                tri_def = tri.Triangulation(x_d, y_d,
                            triangulate_field(mesh, dh, self.field_names[0]).triangles)
                if background is True or background == "magnitude":
                    scalar, label = np.hypot(u_vals, v_vals), "|u|"
                else:
                    comp = next((c for c in self.components
                                if c.field_name == background), None)
                    scalar, label = comp.nodal_values, background
                tcf = ax.tricontourf(tri_def, scalar, levels=15,
                                    cmap=kwargs.pop('cmap', 'viridis'))
                fig.colorbar(tcf, ax=ax, label=label)

            # pick up an explicit 'scale' if the caller gave one – otherwise let
            # Matplotlib use its default.  Everything else comes from **kwargs.
            scale_kw = kwargs.pop('scale', None)
            q_kwargs = dict(angles='xy', scale_units='xy',
                            color=kwargs.pop('color', 'k'))
            if scale_kw is not None:
                q_kwargs['scale'] = scale_kw
            q_kwargs.update(kwargs)

            ax.quiver(x_d[sl], y_d[sl], u_vals[sl], v_vals[sl], **q_kwargs)
            ax.set(title=kwargs.pop('title', f"{self.name} (deformed quiver)"),
                xlabel='x', ylabel='y', aspect='equal')
            plt.show()

        elif kind in {'streamlines', 'streamline'}:
            # temporarily swap mesh.nodes_x_y_pos for the deformed coords,
            # call existing _plot_streamlines, then restore
            original = mesh.nodes_x_y_pos.copy()
            mesh.nodes_x_y_pos[:] = coords_def
            try:
                self._plot_streamlines(background=background, **kwargs)
            finally:
                mesh.nodes_x_y_pos[:] = original
        else:
            raise ValueError("kind must be 'quiver' or 'streamlines'")
    

    def animate_deformation(self,
                            displacement: 'VectorFunction',
                            frames: int = 30,
                            exaggeration: float = 1.0,
                            interval: int = 100,
                            **kwargs):
        """
        Create a simple Matplotlib animation that morphs the mesh from the
        undeformed state (t=0) to the fully deformed one (t=1).

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            You can save it with anim.save('beam.gif', writer='imagemagick').
        """
        dh   = self._dof_handler
        coords = dh.get_dof_coords(self.field_names[0])
        disp_x = displacement.components[0].nodal_values
        disp_y = displacement.components[1].nodal_values
        u_vals = self.components[0].nodal_values
        v_vals = self.components[1].nodal_values

        fig, ax = plt.subplots(figsize=(8, 4))
        q = ax.quiver(coords[:, 0], coords[:, 1],
                    u_vals, v_vals, angles='xy',
                    scale_units='xy', scale=None, color='k', **kwargs)
        ax.set_aspect('equal')
        ax.set_title('Deformation animation')

        def update(frame):
            f = frame / (frames - 1)
            x_def = coords[:, 0] + f*exaggeration*disp_x
            y_def = coords[:, 1] + f*exaggeration*disp_y
            q.set_offsets(np.c_[x_def, y_def])
            return q,

        anim = FuncAnimation(fig, update, frames=frames,
                            blit=True, interval=interval)
        plt.show()
        return anim


    


class NodalFunction(Expression):
    def __init__(self, field_name: str):
        self.field_name = field_name    # key in mesh.node_data
    def __repr__(self): return f"NodalFunction({self.field_name!r})"

class TrialFunction(Function):
    is_trial = True
    is_test = False
    is_function = False
    def __init__(self, field_name: str,dof_handler: 'DofHandler' = None,
                 parent_vector: 'VectorFunction' = None, 
                 component_index: int = None, name: str = None, side: str = ""):
        # A TrialFunction is purely symbolic. It has no dof_handler or data.
        # Its name and field_name are the same.
        from pycutfem.ufl.functionspace import FunctionSpace
        if isinstance(field_name, FunctionSpace):
            # If field_name is a FunctionSpace, use its name and field_names
            name = field_name.name if name is None else name
            field_name = field_name.field_names[0]
            self.side = getattr(field_name, "side", "") if side == "" else side
        else:
            self.side = side
            if name is None:
                name = f"trial_{field_name}"
        super().__init__(name=name, field_name=field_name, dof_handler=dof_handler,
                         parent_vector=parent_vector, component_index=component_index)
        self.dim = 0
        self.parent_name = getattr(parent_vector, "name", "") if parent_vector else ""
        if self.side in ("+","-"):
            s = "pos" if self.side == "+" else "neg"
            self.field_sides = [s] 
        else:
            self.field_sides = None


class TestFunction(Function):
    is_test = True
    is_trial = False
    is_function = False
    def __init__(self, field_name: str,dof_handler: 'DofHandler' = None,
                 parent_vector: 'VectorFunction' = None, 
                 component_index: int = None, name: str = None,
                 side: str = ""):
        # A TestFunction is purely symbolic.
        from pycutfem.ufl.functionspace import FunctionSpace
        if isinstance(field_name, FunctionSpace):
            # If field_name is a FunctionSpace, use its name and field_names
            name = field_name.name if name is None else name
            field_name = field_name.field_names[0]
            self.side = getattr(field_name, "side", "") if side == "" else side
        else:
            self.side = side
            if name is None:
                name = f"test_{field_name}"
        super().__init__(name=name, field_name=field_name, dof_handler=dof_handler,
                         parent_vector=parent_vector, component_index=component_index)
        self.dim = 0  # Assuming scalar test functions for now
        self.parent_name = getattr(parent_vector, "name", "") if parent_vector else ""
        if self.side in ("+","-"):
            s = "pos" if self.side == "+" else "neg"
            self.field_sides = [s] 
        else:
            self.field_sides = ""
 

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
    parent_name = ""
    def __init__(self, space, dof_handler: 'DofHandler'=None, side: str = ""): # space: FunctionSpace
        self.space = space
        self.field_names = space.field_names
        self.side = getattr(space, "side", "") if side == "" else side
        self.components = [
            TrialFunction(name=f"{space.name}_{fn}",field_name = fn, dof_handler=dof_handler, 
                          parent_vector=self, 
                          component_index=i,
                          side = self.side)
            for i, fn in enumerate(space.field_names)
        ]
        self.dim = 1
        if self.side in ("+","-"):
            s = "pos" if self.side == "+" else "neg"
            self.field_sides = [s] * len(self.field_names)
        else:
            self.field_sides = [""] * len(self.field_names)
    def __repr__(self):
        return f"VectorTrialFunction(space='{self.space.name}')"
    def __getitem__(self, i): return self.components[i]
    def __iter__(self): return iter(self.components)


class VectorTestFunction(Expression):
    is_test = True
    is_function = False
    is_trial = False
    parent_name = ""
    def __init__(self, space, dof_handler=None, side: str = ""): # space: FunctionSpace
        self.space = space
        self.field_names = space.field_names
        self.side = getattr(space, "side", "") if side == "" else side
        self.components = [
            TestFunction(name=f"{space.name}_{fn}", 
                        field_name= fn,
                         dof_handler = dof_handler,
                         parent_vector=self, component_index=i,
                         side = self.side)
            for i, fn in enumerate(space.field_names)
        ]
        self.dim = 1
        if self.side in ("+","-"):
            s = "pos" if self.side == "+" else "neg"
            self.field_sides = [s] * len(self.field_names)
        else:
            self.field_sides = [""] * len(self.field_names)
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

class NormalComponent(Expression):
    """Scalar component of the interface normal vector."""
    def __init__(self, parent: "FacetNormal", idx: int):
        super().__init__()
        self.parent = parent      # the FacetNormal instance
        self.idx    = idx         # 0 → n_x , 1 → n_y

    def __repr__(self):
        return f"NormalComponent({self.idx})"
class FacetNormal(Expression): 
    def __getitem__(self, idx: int):
        if idx not in (0, 1):
            raise IndexError("FacetNormal has two components (0, 1).")
        return NormalComponent(self, idx)

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

class Hessian(Expression):
    """2×2 Hessian of a scalar expression."""
    def __init__(self, operand): self.operand = operand
    def __repr__(self): return f"Hessian({self.operand!r})"

class Laplacian(Expression):
    """Trace of the Hessian (Δu)."""
    def __init__(self, operand): self.operand = operand
    def __repr__(self): return f"Laplacian({self.operand!r})"

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
            u_on_neg = deepcopy(u_on_pos)
            # remove pos from name
            # setattr(u_on_neg, "side", "-")
            # setattr(u_on_pos, "side", "+")
            u_on_pos, u_on_neg = Pos(u_on_pos), Neg(u_on_neg)
        self.u_pos, self.u_neg = u_on_pos, u_on_neg
        self.is_function = getattr(u_on_pos, "is_function", False)
        self.is_trial    = getattr(u_on_pos, "is_trial",    False)
        self.is_test     = getattr(u_on_pos, "is_test",     False)
  

    def __repr__(self):
        return f"Jump({self.u_pos!r}, {self.u_neg!r})"

class Avg(Expression):
    def __init__(self, v): self.v = v
    def __repr__(self): return f"Avg({self.v!r})"



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


class CellDiameter(Expression):
    """Return element‑wise √area — identical to mesh.element_char_length(eid)."""
    def __init__(self):
        super().__init__()
        self.role = "none"


class Restriction(Expression):
    """
    Restricts an expression to be active only on elements with a specific
    domain tag.
    """
    def __init__(self, operand: Expression, domain: BitSet):
        super().__init__()
        if not isinstance(operand, Expression) or not isinstance(domain, BitSet):
            raise TypeError("Restriction takes an Expression and a domain BitSet.")
        self.operand = operand
        self.domain = domain
        
        # Inherit properties from the operand for type checking
        self.is_function = getattr(operand, "is_function", False)
        self.is_trial    = getattr(operand, "is_trial",    False)
        self.is_test     = getattr(operand, "is_test",     False)

    def __repr__(self):
        return f"Restriction({self.operand!r}, '{self.domain}')"

class Trace(Expression):
    """Symbolic trace of a tensor expression."""
    def __init__(self, A: Expression):
        super().__init__()
        self.A = A
    def __repr__(self):
        return f"Trace({self.A!r})"

def trace(A):
    """Helper function to create a Trace expression."""
    return Trace(A)

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
def restrict(expression, domain_tag):
    return Restriction(expression, domain_tag)
