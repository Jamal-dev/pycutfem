# dofhandler.py

from __future__ import annotations
import numpy as np
from typing import Dict, List, Set, Tuple, Callable, Mapping, Iterable, Union, Any, Sequence
import math

# Assume these are available in the project structure
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node, Edge, Element
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.fem import transform
from pycutfem.integration.quadrature import volume

BcLike = Union[BoundaryCondition, Mapping[str, Any]]

# -----------------------------------------------------------------------------
#  Main class
# -----------------------------------------------------------------------------
class DofHandler:
    """Centralised DOF numbering and boundary‑condition helpers."""

    # .........................................................................
    def __init__(self, fe_space: Union[Dict[str, Mesh], 'MixedElement'], method: str = "cg"):
        if method not in {"cg", "dg"}:
            raise ValueError("method must be 'cg' or 'dg'")
        self.method: str = method
        
        # This will store tags for specific DOFs, e.g., {'pressure_pin': {123}}
        self.dof_tags: Dict[str, Set[int]] = {}
        # This will map a global DOF index back to its (field, node_id) origin
        self._dof_to_node_map: Dict[int, Tuple[str, int]] = {}

        # Detect *which* constructor variant we are using --------------------
        if MixedElement is not None and isinstance(fe_space, MixedElement):
            self.mixed_element: MixedElement = fe_space
            self.field_names: List[str] = list(self.mixed_element.field_names)
            # For compatibility keep a fe_map (field → mesh) even though all
            # fields share the same mesh.
            self.fe_map: Dict[str, Mesh] = {f: self.mixed_element.mesh for f in self.field_names}
            # Place‑holders initialised below
            self.field_offsets: Dict[str, int] = {}
            self.field_num_dofs: Dict[str, int] = {}
            self.element_maps: Dict[str, List[List[int]]] = {f: [] for f in self.field_names}
            self.dof_map: Dict[str, Dict] = {f: {} for f in self.field_names}
            self.total_dofs: int = 0
            if method == "cg":
                self._build_maps_cg_mixed()
                self._dg_mode = False
            else:
                self._build_maps_dg_mixed()
                self._dg_mode = True
        else:
            # ---------------- legacy single‑field path ---------------------
            self.mixed_element = None  # type: ignore
            self.fe_map: Dict[str, Mesh] = fe_space  # type: ignore[assignment]
            self.method = method
            self.field_names: List[str] = list(self.fe_map.keys())
            self.field_offsets: Dict[str, int] = {}
            self.field_num_dofs: Dict[str, int] = {}
            self.element_maps: Dict[str, List[List[int]]] = {f: [] for f in self.field_names}
            self.dof_map: Dict[str, Dict] = {f: {} for f in self.field_names}
            self.total_dofs = 0
            if method == "cg":
                self._dg_mode = False
                self._build_maps_cg()
            else:
                self._dg_mode = True
                self._build_maps_dg()

        # After maps are built, create the reverse map for CG mode
        if not self._dg_mode:
            for fld in self.field_names:
                for nid, dof in self.dof_map[fld].items():
                    self._dof_to_node_map[dof] = (fld, nid)

    # ------------------------------------------------------------------
    # MixedElement-aware Builders
    # ------------------------------------------------------------------
    def _local_node_indices_for_field(self, p_mesh: int, p_f: int, elem_type: str, fld: str) -> List[int]:
        """Indices of geometry nodes that a *p_f* field uses inside a *p_mesh* element."""
        if p_f > p_mesh:
            raise ValueError(f"Field order ({p_f}) exceeds mesh order ({p_mesh}).")
        
        # This check might be too restrictive for certain p-refinements, but good for now.
        if p_mesh % p_f != 0 and p_f != 1:
            raise ValueError("Currently require mesh-order to be multiple of field-order (except P1).")

        step = p_mesh // p_f if p_f != 0 else p_mesh
        if elem_type == "quad":
            return [j * (p_mesh + 1) + i
                    for j in range(0, p_mesh + 1, step)
                    for i in range(0, p_mesh + 1, step)]
        elif elem_type == "tri":
            if p_f == 1 and p_mesh == 2: # P1 field on P2 geometry
                return [0, 1, 2] # Corner nodes
            if p_f == p_mesh: # Orders match
                return list(range(self.mixed_element._n_basis[fld]))
            raise NotImplementedError("Mixed-order triangles supported only for P1 on P2 geometry.")
        else:
            raise KeyError(f"Unsupported element_type '{elem_type}'")

    def _build_maps_cg_mixed(self) -> None:
        """Continuous-Galerkin DOF numbering for MixedElement spaces."""
        mesh = self.mixed_element.mesh
        p_mesh = mesh.poly_order
        node_dof_map: Dict[Tuple[str, int], int] = {}
        offset = 0

        # 1. Allocate a global DOF for every *used* mesh node per field.
        #    This loop iterates through elements, discovers all (field, node) pairs,
        #    assigns a unique DOF to each pair when first encountered, and builds
        #    the element-to-DOF map for each field.
        for elem in mesh.elements_list:
            loc2phys = {loc: nid for loc, nid in enumerate(elem.nodes)}
            
            for fld in self.field_names:
                p_f = self.mixed_element._field_orders[fld]
                # Use the robust class method to get needed local node indices
                needed_loc_idx = self._local_node_indices_for_field(p_mesh, p_f, mesh.element_type, fld)
                
                gids = []
                for loc in needed_loc_idx:
                    phys_nid = loc2phys[loc]
                    key = (fld, phys_nid)
                    if key not in node_dof_map:
                        node_dof_map[key] = offset
                        offset += 1
                    gids.append(node_dof_map[key])
                
                self.element_maps[fld].append(gids)

        # 2. Finalize handler state based on the created map.
        self.total_dofs = offset
        
        # Create a temporary reverse map for convenience in BCs/legacy methods
        for (fld, nid), dof in node_dof_map.items():
            if fld not in self.dof_map:
                self.dof_map[fld] = {}
            self.dof_map[fld][nid] = dof

        # Calculate the number of DOFs and offsets for each field
        for fld in self.field_names:
            # For CG-mixed with interleaved numbering, the "offset" is not a
            # contiguous block start, but we can define it as the first DOF index
            # encountered for that field, which is useful for info purposes.
            field_dof_indices = self.dof_map[fld].values()
            if field_dof_indices:
                self.field_offsets[fld] = min(field_dof_indices)
                self.field_num_dofs[fld] = len(field_dof_indices)
            else:
                self.field_offsets[fld] = 0
                self.field_num_dofs[fld] = 0

    def _build_maps_dg_mixed(self) -> None:
        """Discontinuous‑Galerkin numbering – element‑local uniqueness."""
        mesh = self.mixed_element.mesh
        p_mesh = mesh.poly_order

        offset = 0
        for elem in mesh.elements_list:
            loc2phys = {loc: nid for loc, nid in enumerate(elem.nodes)}
            for fld in self.field_names:
                p_f = self.mixed_element._field_orders[fld]
                loc_idx = self._local_node_indices_for_field(p_mesh, p_f, mesh.element_type, fld)
                n_local = len(loc_idx)
                dofs = list(range(offset, offset + n_local))
                self.element_maps[fld].append(dofs)

                # Build per‑node map for BCs ------------------------------
                nd2d: Dict[int, Dict[int, int]] = self.dof_map.setdefault(fld, {})
                for loc, dof in zip(loc_idx, dofs):
                    phys_nid = loc2phys[loc]
                    nd2d.setdefault(phys_nid, {})[elem.id] = dof

                offset += n_local
        self.total_dofs = offset
        self.field_offsets = {fld: 0 for fld in self.field_names}
        self.field_num_dofs = {fld: self.total_dofs for fld in self.field_names} # This is not quite right, but reflects that all DOFs are "in" the field space
    # ------------------------------------------------------------------
    # Legacy Builders (for backward compatibility)
    # ------------------------------------------------------------------
    def _build_maps_cg(self) -> None:
        offset = 0
        for fld, mesh in self.fe_map.items():
            self.field_offsets[fld] = offset
            self.field_num_dofs[fld] = len(mesh.nodes_list)
            self.dof_map[fld] = {nd.id: offset + i for i, nd in enumerate(mesh.nodes_list)}
            self.element_maps[fld] = [[self.dof_map[fld][nid] for nid in el.nodes]
                                       for el in mesh.elements_list]
            offset += len(mesh.nodes_list)
        self.total_dofs = offset

    def _build_maps_dg(self) -> None:
        offset = 0
        for fld, mesh in self.fe_map.items():
            self.field_offsets[fld] = offset
            per_node: Dict[int, Dict[int, int]] = {nd.id: {} for nd in mesh.nodes_list}
            field_dofs = 0
            for el in mesh.elements_list:
                dofs = list(range(offset, offset + len(el.nodes)))
                self.element_maps[fld].append(dofs)
                for loc, nid in enumerate(el.nodes):
                    per_node[nid][el.id] = dofs[loc]
                offset += len(el.nodes)
                field_dofs += len(el.nodes)
            self.dof_map[fld] = per_node
            self.field_num_dofs[fld] = field_dofs
        self.total_dofs = offset

    # ------------------------------------------------------------------
    #  Public helpers
    # ------------------------------------------------------------------
    def get_elemental_dofs(self, element_id: int) -> np.ndarray:
        """Return *stacked* global DOFs for element *element_id*."""
        if self.mixed_element is None:
            raise RuntimeError("get_elemental_dofs requires a MixedElement‑backed DofHandler.")
        parts: List[int] = []
        for fld in self.field_names:
            parts.extend(self.element_maps[fld][element_id])
        return np.asarray(parts, dtype=int)

    def get_reference_element(self, field: str | None = None):
        """Return the per‑field reference or the MixedElement itself."""
        if self.mixed_element is None:
            raise RuntimeError("This DofHandler was not built from a MixedElement.")
        if field is None:
            return self.mixed_element
        return self.mixed_element._ref[field]

    def get_dof_pairs_for_edge(self, field: str, edge_gid: int) -> Tuple[List[int], List[int]]:
        if not self._dg_mode:
            raise RuntimeError("Edge DOF pairs only relevant for DG spaces.")
        mesh = self.fe_map[field]
        edge = mesh.edges_list[edge_gid]
        if edge.left is None or edge.right is None:
            raise ValueError("Edge is on boundary – no right element.")
        
        raise NotImplementedError("get_dof_pairs_for_edge needs careful implementation for mixed DG.")

    def _require_cg(self, name: str) -> None:
        if self._dg_mode:
            raise NotImplementedError(f"{name} not available for DG spaces – every element owns its DOFs.")

    def get_field_slice(self, field: str) -> List[int]:
        """Get all global DOF indices for a given field, sorted ascending."""
        self._require_cg("get_field_slice")
        if field not in self.field_names:
            raise ValueError(f"Unknown field '{field}'.")
        return sorted(list(self.dof_map[field].values()))

    def get_field_dofs_on_nodes(self, field: str) -> np.ndarray:
        self._require_cg("get_field_dofs_on_nodes")
        if field not in self.field_names:
            raise ValueError(f"Unknown field '{field}'.")
        return np.asarray(sorted(self.dof_map[field].values()), dtype=int)

    def get_dof_coords(self, field: str) -> np.ndarray:
        self._require_cg("get_dof_coords")
        mesh = self.fe_map[field]
        coords = [(mesh.nodes_x_y_pos[nid][0], mesh.nodes_x_y_pos[nid][1])
                  for nid in sorted(self.dof_map[field].keys())]
        return np.asarray(coords, dtype=float)
        
    # ------------------------------------------------------------------
    #  Tagging and Dirichlet handling (CG‑only)
    # ------------------------------------------------------------------
    def tag_dof_by_locator(self, 
                           tag: str, 
                           field: str, 
                           locator: Callable[[float, float], bool], 
                           find_first: bool = True):
        """Tags a specific Degree of Freedom (DOF) with a string identifier."""
        self._require_cg("DOF tagging")
        if field not in self.field_names:
            raise ValueError(f"Field '{field}' not found in DofHandler. Available fields: {self.field_names}")

        mesh = self.fe_map[field]
        field_dof_map = self.dof_map[field]
        
        found = False
        for node in mesh.nodes_list:
            if locator(node.x, node.y):
                if node.id in field_dof_map:
                    dof_index = field_dof_map[node.id]
                    self.dof_tags.setdefault(tag, set()).add(dof_index)
                    found = True
                    if find_first:
                        break
        
        if not found:
            print(f"Warning: DofHandler.tag_dof_by_locator did not find any node for field '{field}' with the given locator.")

    def get_dirichlet_data(self,
                           bcs: Union[BcLike, Iterable[BcLike]],
                           locators: Dict[str, Callable[[float, float], bool]] = None) -> Dict[int, float]:
        """Calculates Dirichlet DOF values from a list of BoundaryCondition objects."""
        self._require_cg("Dirichlet BC evaluation")
        data: Dict[int, float] = {}
        bcs_list = bcs if isinstance(bcs, (list, tuple, set)) else [bcs]
        locators = locators or {}

        for bc in bcs_list:
            if not isinstance(bc, BoundaryCondition):
                continue

            field = getattr(bc, "field", None)
            if field is None: continue
            mesh = self.fe_map.get(field)
            if mesh is None: continue

            domain_tag = getattr(bc, "domain_tag", None) or getattr(bc, "tag", None)
            if domain_tag is None: continue

            if domain_tag in self.dof_tags:
                val_is_callable = callable(bc.value)
                for dof in self.dof_tags[domain_tag]:
                    dof_field, node_id = self._dof_to_node_map[dof]
                    if dof_field == field:
                        x, y = mesh.nodes_x_y_pos[node_id]
                        value = bc.value(x, y) if val_is_callable else bc.value
                        data[dof] = value
                continue

            nodes_on_domain: Set[int] = set()
            if domain_tag in locators:
                locator_func = locators[domain_tag]
                for node in mesh.nodes_list:
                    if locator_func(node.x, node.y):
                        nodes_on_domain.add(node.id)
            else:
                found_on_edges = False
                for edge in mesh.edges_list:
                    if getattr(edge, 'tag', None) == domain_tag:
                        nodes_to_add = getattr(edge, 'all_nodes', edge.nodes)
                        if nodes_to_add:
                            nodes_on_domain.update(nodes_to_add)
                            found_on_edges = True
                if not found_on_edges:
                    for node in mesh.nodes_list:
                        if getattr(node, 'tag', None) == domain_tag:
                            nodes_on_domain.add(node.id)

            val_is_callable = callable(bc.value)
            for nid in nodes_on_domain:
                dof = self.dof_map.get(field, {}).get(nid)
                if dof is not None:
                    x, y = mesh.nodes_x_y_pos[nid]
                    value = bc.value(x, y) if val_is_callable else bc.value
                    data[dof] = value
        return data

    # ------------------------------------------------------------------
    #  NEW: Solver loop integration
    # ------------------------------------------------------------------
    def add_to_functions(self, delta: np.ndarray, functions: List[Any]):
        """
        Distributes and adds a global delta vector to the nodal values of
        one or more Function or VectorFunction objects.

        This is the recommended way to update solution functions after a
        solver step, as it correctly handles the mapping from the global
        solution vector to the individual field components by updating the
        underlying data arrays in-place.

        Args:
            delta: The global update vector, typically from a linear solver.
            functions: A list of Function or VectorFunction objects to update.
        """
        # Import here to avoid circular dependency at the top level
        from pycutfem.ufl.expressions import Function, VectorFunction

        if delta.shape[0] != self.total_dofs:
            raise ValueError(f"Shape of delta vector ({delta.shape[0]}) does not match "
                             f"total DOFs in handler ({self.total_dofs}).")

        for func in functions:
            target_array = None
            g2l_map = None

            if isinstance(func, VectorFunction):
                target_array = func.nodal_values
                g2l_map = func._g2l
            elif isinstance(func, Function) and func._parent_vector is None:
                # This is a standalone function
                target_array = func._values
                g2l_map = func._g2l
            
            if target_array is not None and g2l_map is not None:
                for gdof, lidx in g2l_map.items():
                    if gdof < len(delta):
                        target_array[lidx] += delta[gdof]

    def apply_bcs(self, bcs: Union[BcLike, Iterable[BcLike]], *functions: Any):
        """
        Applies boundary conditions directly to Function or VectorFunction objects.

        This method gets all Dirichlet data and correctly sets the nodal values
        on the provided functions, handling the mapping from global DOFs to
        the functions' local data arrays. This is the recommended way to apply BCs
        after a solver update.

        Args:
            bcs: The boundary condition definitions.
            *functions: A variable number of Function or VectorFunction objects to modify.
        """
        # Import here to avoid circular dependency
        from pycutfem.ufl.expressions import Function, VectorFunction

        dirichlet_data = self.get_dirichlet_data(bcs)

        for func in functions:
            if not isinstance(func, (Function, VectorFunction)):
                continue

            g2l_map = func._g2l
            
            if isinstance(func, VectorFunction):
                target_array = func.nodal_values
            elif isinstance(func, Function) and func._parent_vector is None:
                target_array = func._values
            else: # Skip component functions as their parent will be handled
                continue

            for dof, value in dirichlet_data.items():
                if dof in g2l_map:
                    local_idx = g2l_map[dof]
                    if local_idx < len(target_array):
                        target_array[local_idx] = value
    
    def apply_bcs_to_vector(self, vec: np.ndarray, bcs: Union[BcLike, Iterable[BcLike]]):
        """
        DEPRECATED: Prefer `apply_bcs`.
        Applies BCs to a raw NumPy vector representing the global solution.
        """
        for dof, val in self.get_dirichlet_data(bcs).items():
            if dof < vec.size:
                vec[dof] = val

    def _expand_bc_specs(
        self, bcs: Union[BcLike, Iterable[BcLike]]
    ) -> List[Tuple[str, Any, Any]]:
        if not bcs: return []
        items = bcs if isinstance(bcs, (list, tuple, set)) else [bcs]
        out = []
        for bc in items:
            if isinstance(bc, BoundaryCondition):
                domain = getattr(bc, "domain", None) or getattr(bc, "domain_tag", None) or getattr(bc, "tag", None)
                out.append((bc.field, domain, bc.value))
        return out
    
    def l2_error(self,
                u_vec: np.ndarray,
                exact: Mapping[str, Callable[[float, float], Sequence[float]]],
                quad_order: int | None = None,
                relative: bool = True) -> float:
        """
        Compute the global L2-error of *u_vec* against analytical *exact* functions.

        Parameters
        ----------
        u_vec       : global solution vector.
        exact       : mapping  field-name -> callable (x,y) → value or tuple of
                    component values.  Provide one entry for every field you
                    want in the error norm.
        quad_order  : if ``None`` we use 2*mesh.poly_order (safe).
        relative    : return ‖u_h-u‖ / ‖u‖ if *True*, else absolute error.

        Returns
        -------
        float
        """
        mesh  = self.mixed_element.mesh
        me    = self.mixed_element
        qdeg  = quad_order or 2*mesh.poly_order
        qp, qw = volume(mesh.element_type, qdeg)            # ← existing helper
        err2 = exact2 = 0.0

        # loop over physical elements
        for eid, elem in enumerate(mesh.elements_list):
            gdofs = self.get_elemental_dofs(eid)
            u_loc = u_vec[gdofs]                             # element DoFs view

            for (xi, eta), w in zip(qp, qw):
                # mapping & Jacobian
                J      = transform.jacobian(mesh, eid, (xi, eta))
                detJ   = abs(np.linalg.det(J))
                x, y   = transform.x_mapping(mesh, eid, (xi, eta))

                # assemble FE value per requested field/component --------------
                uh_fields = {}
                for fld in exact.keys():
                    phi = me.basis(fld, xi, eta)             # shape (n_loc,)
                    uh_fields[fld] = float(u_loc @ phi)      # scalar value
                                                            # (vector fields are
                                                            # stored component-wise)

                # accumulate error and exact norms -----------------------------
                for fld, u_ex in exact.items():
                    u_exact_val = u_ex(x, y)
                    # accept scalar or sequence – always treat as 1-D np.array
                    u_exact_val = np.atleast_1d(u_exact_val).astype(float)

                    uh_val = np.atleast_1d(uh_fields[fld])
                    diff2 = np.sum((uh_val - u_exact_val)**2)
                    base2 = np.sum(u_exact_val**2)

                    err2   += w*detJ*diff2
                    exact2 += w*detJ*base2

        return math.sqrt(err2 / exact2) if relative else math.sqrt(err2)

    def info(self) -> None:
        print(f"=== DofHandler ({self.method.upper()}) ===")
        for fld in self.field_names:
            print(f"  {fld:>8}: {self.field_num_dofs[fld]} DOFs @ offset {self.field_offsets[fld]}")
        print("  total :", self.total_dofs)
        
    def __repr__(self) -> str:
        if self.mixed_element:
            return f"<DofHandler Mixed, ndofs={self.total_dofs}, method='{self.method}'>"
        return f"<DofHandler legacy, ndofs={self.total_dofs}, method='{self.method}', fields={self.field_names}>"



# ==============================================================================
#  MAIN BLOCK FOR DEMONSTRATION (Using real Mesh class)
# ==============================================================================
if __name__ == '__main__':
    # This block demonstrates the intended workflow using the actual library components.
    
    from pycutfem.utils.meshgen import structured_quad
    from pycutfem.core.topology import Node

    # 1. Generate mesh data using a library utility
    nodes, elems, _, corners = structured_quad(1, 0.5, nx=2, ny=1, poly_order=1)
    mesh = Mesh(nodes=nodes, 
                element_connectivity=elems,
                elements_corner_nodes=corners, 
                element_type="quad", 
                poly_order=1)
    
    bc_dict = {'left': lambda x,y: x==0, 'bottom': lambda x,y:y==0,
               'top': lambda x,y: y==0.5, 'right':lambda x,y:x==1}
    mesh.tag_boundary_edges(bc_dict)
    
    me = MixedElement(mesh, field_specs={'scalar_field':1})
    dof_handler_cg = DofHandler(me, method='cg')
    
    # ... (existing demos can be kept or removed for brevity) ...
    
    # -------------------------------------------------------------------
    # NEW DEMONSTRATION: MIXED-ELEMENT (STOKES-LIKE) WITH DOF TAGGING
    # -------------------------------------------------------------------
    print("\n" + "="*70)
    print("DEMONSTRATION: MIXED-ELEMENT (STOKES-LIKE) WITH DOF TAGGING")
    print("="*70)
    
    # Create a Q2 geometry mesh
    nodes_stokes, elems_stokes, _, corners_stokes = structured_quad(1, 1, nx=2, ny=2, poly_order=2)
    mesh_stokes = Mesh(nodes=nodes_stokes,
                       element_connectivity=elems_stokes,
                       elements_corner_nodes=corners_stokes,
                       element_type="quad",
                       poly_order=2)

    # Create a Q2-Q2-Q1 mixed element space (e.g., for Stokes flow)
    stokes_element = MixedElement(mesh_stokes, field_specs={'ux': 2, 'uy': 2, 'p': 1})
    stokes_dof = DofHandler(stokes_element, method='cg')

    stokes_dof.info()

    # The key new functionality: tag a single pressure DOF.
    # Let's pin the pressure at the node closest to (0, 0).
    print("\nTagging a single pressure DOF at (0,0) with 'pressure_pin'...")
    stokes_dof.tag_dof_by_locator(
        tag='pressure_pin',
        field='p',
        locator=lambda x, y: np.isclose(x, 0) and np.isclose(y, 0),
        find_first=True
    )
    print("Tagged DOFs stored in handler:", stokes_dof.dof_tags)

    # Define boundary conditions, including one using the new tag.
    walls = {'bottom': lambda x,y: np.isclose(y, 0),
             'left'  : lambda x,y: np.isclose(x, 0),
             'right' : lambda x,y: np.isclose(x, 1),
             'top'   : lambda x,y: np.isclose(y, 1)}
    mesh_stokes.tag_boundary_edges(walls)

    stokes_bcs = [
        *[BoundaryCondition(c, 'dirichlet', w, lambda x,y: 0.0)
          for c in ('ux','uy') for w in ('left','right','bottom')],
        BoundaryCondition('ux', 'dirichlet', 'top', lambda x,y: 1.0),
        BoundaryCondition('uy', 'dirichlet', 'top', lambda x,y: 0.0),
        BoundaryCondition('p',  'dirichlet', 'pressure_pin', lambda x,y: 0.0)
    ]

    print("\nApplying boundary conditions...")
    dirichlet_stokes = stokes_dof.get_dirichlet_data(stokes_bcs)

    print("\nResulting Dirichlet DOFs and values:")
    # Find the pinned pressure DOF in the output
    pinned_dof_set = stokes_dof.dof_tags.get('pressure_pin')
    if pinned_dof_set:
        pinned_dof = list(pinned_dof_set)[0]
        print(f"  ... (many velocity DOFs not shown) ...")
        print(f"  DOF {pinned_dof} (from 'pressure_pin'): {dirichlet_stokes.get(pinned_dof)}")
        print(f"Total Dirichlet DOFs applied: {len(dirichlet_stokes)}")
        
        # Verify the pinned dof is correct
        p_dofs = stokes_dof.get_field_slice('p')
        print(f"\nIs the pinned DOF ({pinned_dof}) in the list of all pressure DOFs? {'Yes' if pinned_dof in p_dofs else 'No'}")
    else:
        print("  'pressure_pin' DOF not found in results.")

