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
<<<<<<< Updated upstream
=======
from collections.abc import Sequence
from hashlib import blake2b
from pycutfem.integration.quadrature import line_quadrature
from functools import lru_cache
<<<<<<< Updated upstream
=======


_edge_geom_cache: dict[tuple, dict] = {}     # ← NEW — module-level cache
>>>>>>> Stashed changes


_edge_geom_cache: dict[tuple, dict] = {}     # ← NEW — module-level cache
>>>>>>> Stashed changes

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
            # loc2phys = {loc: nid for loc, nid in enumerate(elem.nodes)}
            
            for fld in self.field_names:
                p_f = self.mixed_element._field_orders[fld]
                # Use the robust class method to get needed local node indices
                needed_loc_idx = self._local_node_indices_for_field(p_mesh, p_f, mesh.element_type, fld)
                
                gids = []
                for loc in needed_loc_idx:
                    phys_nid = elem.nodes[loc]
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
            if field is None: raise ValueError("BoundaryCondition must have a 'field' attribute.")
            # For MixedElement, all fields share the same mesh, so this is fine.
            mesh = self.fe_map.get(field)
            if mesh is None: continue

            domain_tag = getattr(bc, "domain_tag", None) or getattr(bc, "tag", None)
            if domain_tag is None: continue

            # --- START FIX ---
            # The logic to find nodes must be self-contained within the loop for each BC.
            
            nodes_on_domain: Set[int] = set()

            # Path 1: The domain is a pre-tagged set of DOFs (e.g., 'pressure_pin')
            if domain_tag in self.dof_tags:
                val_is_callable = callable(bc.value)
                for dof in self.dof_tags[domain_tag]:
                    dof_field, node_id = self._dof_to_node_map[dof]
                    if dof_field == field: # Apply only if the field matches
                        x, y = mesh.nodes_x_y_pos[node_id]
                        value = bc.value(x, y) if val_is_callable else bc.value
                        data[dof] = value
                continue # Go to the next BC

            # Path 2: The domain is found by a locator function
            if domain_tag in locators:
                locator_func = locators[domain_tag]
                for node in mesh.nodes_list:
                    if locator_func(node.x, node.y):
                        nodes_on_domain.add(node.id)
            # Path 3: The domain is found by tags on geometric entities (edges/nodes)
            else:
                found_on_edges = False
                for edge in mesh.edges_list:
                    if getattr(edge, 'tag', None) == domain_tag:
                        nodes_to_add = getattr(edge, 'all_nodes', None)
                        if nodes_to_add is None: raise ValueError(f"Edge {edge.id} does not have 'all_nodes' attribute.")
                        nodes_on_domain.update(nodes_to_add)
                        found_on_edges = True
                if not found_on_edges:
                    for node in mesh.nodes_list:
                        if getattr(node, 'tag', None) == domain_tag:
                            nodes_on_domain.add(node.id)

            # Now, apply the value for the CURRENT boundary condition `bc`
            # to the nodes found for its specific domain.
            val_is_callable = callable(bc.value)
            for nid in nodes_on_domain:
                dof = self.dof_map.get(field, {}).get(nid)
                if dof is not None:
                    x, y = mesh.nodes_x_y_pos[nid]
                    value = bc.value(x, y) if val_is_callable else bc.value
                    data[dof] = value
            # --- END FIX ---
            
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

        delta_vec = None
        if isinstance(delta, np.ndarray):
            delta_vec = delta
        elif isinstance(delta, (Function, VectorFunction)):
            # If the user passes a Function object, use its internal data array.
            # This makes the function more robust to common usage errors.
            delta_vec = delta.nodal_values
        else:
            raise TypeError(f"Argument 'delta' must be a NumPy array or a Function object, not {type(delta)}")

        if delta_vec.shape[0] != self.total_dofs:
            raise ValueError(f"Shape of delta vector ({delta_vec.shape[0]}) does not match "
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
                    if gdof < len(delta_vec):
                        target_array[lidx] += delta_vec[gdof]

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
    
    def get_dof_coords(self, field: str) -> np.ndarray:
        """Coordinates of the field’s DOFs in the *same* order as get_field_slice."""
        self._require_cg("get_dof_coords")
        if field not in self.field_names:
            raise ValueError(f"Unknown field '{field}'.")
        gdofs  = self.get_field_slice(field)           # canonical order
        mesh   = self.fe_map[field]
        coords = [ mesh.nodes_x_y_pos[self._dof_to_node_map[d][1]] for d in gdofs ]
        return np.asarray(coords, dtype=float)


    


    def l2_error(self,
                 u_vec: Union[np.ndarray, 'Function', 'VectorFunction'], # Accept multiple types
                 exact: Mapping[str, Callable[[float, float], float]],
                 quad_order: int | None = None,
                 relative: bool = True) -> float:
        """
        Element-wise L2-norm  ‖u_h − u‖, handling NumPy arrays or Function objects.
        """
        
        # Import here to avoid circular dependencies
        from pycutfem.ufl.expressions import Function, VectorFunction
        
        
        mesh  = self.mixed_element.mesh
        me    = self.mixed_element
        qdeg  = quad_order or 2 * mesh.poly_order
        qp, qw = volume(mesh.element_type, qdeg)

        err2 = exact2 = 0.0

        for eid in range(len(mesh.elements_list)):
            gdofs  = self.get_elemental_dofs(eid)
            
            
            # Polymorphic handling of the input solution u_vec
            if isinstance(u_vec, np.ndarray):
                # Case 1: Input is a raw NumPy vector from a solver.
                # The original logic is correct for this case.
                u_loc = u_vec[gdofs]
            elif isinstance(u_vec, (Function, VectorFunction)):
                # Case 2: Input is a Function or VectorFunction object.
                # Use the object's dedicated method to get the nodal values.
                u_loc = u_vec.get_nodal_values(gdofs)
            else:
                raise TypeError(f"Unsupported solution type for L2 error calculation: {type(u_vec)}")
            

            for (xi, eta), w in zip(qp, qw):
                J    = transform.jacobian(mesh, eid, (xi, eta))
                detJ = abs(np.linalg.det(J))
                x, y = transform.x_mapping(mesh, eid, (xi, eta))

                for fld, u_exact_func in exact.items():
                    phi = me.basis(fld, xi, eta)
                    uh  = float(u_loc @ phi)
                    u_exact_val = u_exact_func(x, y)
                    diff2  = (uh - u_exact_val)**2
                    base2  = u_exact_val**2
                    err2  += w * detJ * diff2
                    exact2 += w * detJ * base2

        # Avoid division by zero if the exact solution is zero
        if exact2 < 1e-14:
            return math.sqrt(err2)
            
        return math.sqrt(err2 / exact2) if relative else math.sqrt(err2)
<<<<<<< Updated upstream
=======
    
    # ==================================================================
    # NEW: Precompute Geometric Factors for JIT Kernels
    # ==================================================================
    def get_all_dof_coords(self) -> np.ndarray:
        """
        Builds and returns a coordinate array for every DOF in the system.

        The returned array has shape (total_dofs, 2), where the first index
        corresponds directly to the global DOF index. This is the correct
        coordinate array to pass to JIT kernels.

        Returns:
            np.ndarray: The (total_dofs, 2) array of DOF coordinates.
        """
        if self._dg_mode:
            raise NotImplementedError("get_all_dof_coords is not yet implemented for DG methods.")

        # Initialize an array to hold coordinates for every single DOF.
        all_coords = np.zeros((self.total_dofs, 2))

        # Use the internal map that links a global DOF to its original node.
        for dof_idx, (field, node_idx) in self._dof_to_node_map.items():
            # Get the coordinates of the original node.
            coords = self.fe_map[field].nodes_x_y_pos[node_idx]
            # Place these coordinates at the correct index in our new array.
            all_coords[dof_idx] = coords

        return all_coords
    def precompute_geometric_factors(self, quad_order: int, level_set: Callable = None) -> Dict[str, np.ndarray]:
        """
        Calculates all geometric data needed by JIT kernels ahead of time.

        This is the single, authoritative method for preparing geometric
        quadrature data. It computes physical coordinates, scaled quadrature
        weights, inverse Jacobians, and optional level-set values.

        Args:
            quad_order (int): The polynomial degree of the quadrature rule.
            level_set (Callable, optional): A level-set function to evaluate `phi`
                                            at quadrature points. Defaults to None.

        Returns:
            Dict[str, np.ndarray]: A dictionary of pre-computed arrays.
        """
        if self.mixed_element is None:
            raise RuntimeError("This method requires a MixedElement-backed DofHandler.")

        mesh = self.mixed_element.mesh
        num_elements = len(mesh.elements_list)
        spatial_dim = mesh.spatial_dim

        # 1. Get reference quadrature rule
        qp_ref, qw_ref = volume(mesh.element_type, quad_order)
        num_quad_points = len(qw_ref)

        # 2. Initialize output arrays
        qp_phys = np.zeros((num_elements, num_quad_points, spatial_dim))
        qw_scaled = np.zeros((num_elements, num_quad_points))
        detJ = np.zeros((num_elements, num_quad_points))
        J_inv = np.zeros((num_elements, num_quad_points, spatial_dim, spatial_dim))
        phis = np.zeros((num_elements, num_quad_points)) if level_set else None
        normals = np.zeros((num_elements, num_quad_points, spatial_dim)) # Placeholder

        # 3. Loop over all elements and quadrature points
        for e_idx in range(num_elements):
            for q_idx, (qp, qw) in enumerate(zip(qp_ref, qw_ref)):
                xi_tuple = tuple(qp)

                J_matrix = transform.jacobian(mesh, e_idx, xi_tuple)
                det_J_val = np.linalg.det(J_matrix)
                if det_J_val <= 1e-12:
                    raise ValueError(f"Jacobian determinant is non-positive ({det_J_val}) for element {e_idx}.")

                J_inv[e_idx, q_idx] = np.linalg.inv(J_matrix)
                detJ[e_idx, q_idx] = det_J_val
                
                x_phys, y_phys = transform.x_mapping(mesh, e_idx, xi_tuple)
                qp_phys[e_idx, q_idx, 0] = x_phys
                qp_phys[e_idx, q_idx, 1] = y_phys
                
                qw_scaled[e_idx, q_idx] = qw * det_J_val

                if level_set:
                    phis[e_idx, q_idx] = level_set(x_phys, y_phys)

        # 4. Return results in a dictionary for easy use
        return {
            "qp_phys": qp_phys,
            "qw": qw_scaled,
            "detJ": detJ,
            "J_inv": J_inv,
            "normals": normals,
            "phis": phis,
        }
    
    # ------------------------------------------------------------------
    # edge-geometry cache keyed by (hash(ids), qdeg, id(level_set))
    # ------------------------------------------------------------------
    _edge_geom_cache: dict[tuple[int,int,int], dict] = {}

    def _hash_subset(idx: Sequence[int]) -> int:
        """Fast 64-bit signature for any list / BitSet of indices."""
        return hash(bytes(sorted(idx)))
    
    

    # -------------------------------------------------------------------------
    #  DofHandler.precompute_interface_factors
    # -------------------------------------------------------------------------
    def precompute_interface_factors(
        self,
        cut_element_ids: "BitSet | Sequence[int]",
        qdeg: int,
        level_set,
        reuse: bool = True,
    ) -> dict:
        """
        Pre-computes all geometric data for interface integrals on a given
        set of CUT elements.

        This is the authoritative method for preparing data for dInterface JIT kernels.
        It iterates directly over cut elements, not edges, ensuring all geometric
        data (Jacobians, etc.) is sourced from the correct parent element.

        Parameters
        ----------
        cut_element_ids : BitSet | Sequence[int]
            The element IDs of the 'cut' elements to process.
        qdeg : int
            1-D quadrature order along the interface segment.
        level_set : callable
            The level-set function, required for calculating normals.

        Returns
        -------
        dict
            A dictionary of pre-computed arrays, with the first dimension
            corresponding to the order of `cut_element_ids`. Keys include:
            'eids', 'qp_phys', 'qw', 'normals', 'phis', 'detJ', 'J_inv'.
        """
        from pycutfem.integration.quadrature import line_quadrature
        from pycutfem.fem import transform
<<<<<<< Updated upstream

        mesh = self.mixed_element.mesh
        ids = cut_element_ids.to_indices() if hasattr(cut_element_ids, "to_indices") else list(cut_element_ids)

        # Filter for elements that are actually 'cut' and have a valid segment
        valid_cut_eids = [
            eid for eid in ids
            if mesh.elements_list[eid].tag == 'cut' and len(mesh.elements_list[eid].interface_pts) == 2
        ]
        
        if not valid_cut_eids:
            # Return empty arrays with correct shapes if no valid cut elements
            return {
                'eids': np.array([], dtype=int),
                'qp_phys': np.empty((0, 0, 2)), 'qw': np.empty((0, 0)),
                'normals': np.empty((0, 0, 2)), 'phis': np.empty((0, 0)),
                'detJ': np.empty((0, 0)), 'J_inv': np.empty((0, 0, 2, 2)),
            }

        # --- Use a cache if requested ---
        cache_key = (_hash_subset(valid_cut_eids), qdeg, id(level_set))
        if reuse and cache_key in _edge_geom_cache:
            return _edge_geom_cache[cache_key]

        # --- Allocation ---
        # n_elems = len(valid_cut_eids)
        n_elems = mesh.n_elements
        # We need a representative segment to determine n_q
        p0_rep, p1_rep = mesh.elements_list[valid_cut_eids[0]].interface_pts
        q_xi_rep, q_w_rep = line_quadrature(p0_rep, p1_rep, qdeg)
        n_q = len(q_w_rep)

        qp_phys = np.zeros((n_elems, n_q, 2))
        qw = np.zeros((n_elems, n_q))
        normals = np.zeros((n_elems, n_q, 2))
        phis = np.zeros((n_elems, n_q))
        detJ_arr = np.zeros((n_elems, n_q))
        Jinv_arr = np.zeros((n_elems, n_q, 2, 2))
        # ---------- NEW: basis / grad-basis tables on the interface ----------
        me      = self.mixed_element
        fields  = me.field_names            # ['vx', 'vy', …]
        b_tabs  = {f: np.zeros((n_elems, n_q, me.n_dofs_local))         for f in fields}
        g_tabs  = {f: np.zeros((n_elems, n_q, me.n_dofs_local, 2))      for f in fields}


        # --- Loop over valid cut elements ---
        for k, eid in enumerate(valid_cut_eids):
            elem = mesh.elements_list[eid]
            p0, p1 = elem.interface_pts

            # --- Quadrature rule on the physical interface segment ---
            q_xi, q_w = line_quadrature(p0, p1, qdeg)

            for q, (xq, wq) in enumerate(zip(q_xi, q_w)):
                qp_phys[eid, q] = xq
                qw[eid, q] = wq

                # Normal and phi value from the level set
                g = level_set.gradient(xq)
                # norm_g = np.linalg.norm(g)
                normals[eid, q] = g #/ (norm_g + 1e-30)
                phis[eid, q] = level_set(xq)

                # Jacobian of the parent element at the quadrature point
                xi_ref, eta_ref = transform.inverse_mapping(mesh, eid, xq)
                J = transform.jacobian(mesh, eid, (xi_ref, eta_ref))
                detJ_arr[eid, q] = np.linalg.det(J)
                Jinv_arr[eid, q] = np.linalg.inv(J)
                for fld in fields:
                    b_tabs[fld][eid, q] = me.basis      (fld, xi_ref, eta_ref)
                    g_tabs[fld][eid, q] = me.grad_basis (fld, xi_ref, eta_ref)

        # --- Gather results and cache ---
        out = {
            'eids': np.array(valid_cut_eids, dtype=int),
            # 'eids': np.arange(mesh.n_elements, dtype=int),  # All elements, not just cut
            'qp_phys': qp_phys, 'qw': qw, 'normals': normals, 'phis': phis,
            'detJ': detJ_arr, 'J_inv': Jinv_arr,
        }
        for fld in fields:
            out[f"b_{fld}"] = b_tabs[fld]
            out[f"g_{fld}"] = g_tabs[fld]

=======

        mesh = self.mixed_element.mesh
        ids = cut_element_ids.to_indices() if hasattr(cut_element_ids, "to_indices") else list(cut_element_ids)

        # Filter for elements that are actually 'cut' and have a valid segment
        valid_cut_eids = [
            eid for eid in ids
            if mesh.elements_list[eid].tag == 'cut' and len(mesh.elements_list[eid].interface_pts) == 2
        ]
        
        if not valid_cut_eids:
            # Return empty arrays with correct shapes if no valid cut elements
            return {
                'eids': np.array([], dtype=int),
                'qp_phys': np.empty((0, 0, 2)), 'qw': np.empty((0, 0)),
                'normals': np.empty((0, 0, 2)), 'phis': np.empty((0, 0)),
                'detJ': np.empty((0, 0)), 'J_inv': np.empty((0, 0, 2, 2)),
            }

        # --- Use a cache if requested ---
        cache_key = (_hash_subset(valid_cut_eids), qdeg, id(level_set))
        if reuse and cache_key in _edge_geom_cache:
            return _edge_geom_cache[cache_key]

        # --- Allocation ---
        # n_elems = len(valid_cut_eids)
        n_elems = mesh.n_elements
        # We need a representative segment to determine n_q
        p0_rep, p1_rep = mesh.elements_list[valid_cut_eids[0]].interface_pts
        q_xi_rep, q_w_rep = line_quadrature(p0_rep, p1_rep, qdeg)
        n_q = len(q_w_rep)

        qp_phys = np.zeros((n_elems, n_q, 2))
        qw = np.zeros((n_elems, n_q))
        normals = np.zeros((n_elems, n_q, 2))
        phis = np.zeros((n_elems, n_q))
        detJ_arr = np.zeros((n_elems, n_q))
        Jinv_arr = np.zeros((n_elems, n_q, 2, 2))
        # ---------- NEW: basis / grad-basis tables on the interface ----------
        me      = self.mixed_element
        fields  = me.field_names            # ['vx', 'vy', …]
        b_tabs  = {f: np.zeros((n_elems, n_q, me.n_dofs_local))         for f in fields}
        g_tabs  = {f: np.zeros((n_elems, n_q, me.n_dofs_local, 2))      for f in fields}


        # --- Loop over valid cut elements ---
        for k, eid in enumerate(valid_cut_eids):
            elem = mesh.elements_list[eid]
            p0, p1 = elem.interface_pts

            # --- Quadrature rule on the physical interface segment ---
            q_xi, q_w = line_quadrature(p0, p1, qdeg)

            for q, (xq, wq) in enumerate(zip(q_xi, q_w)):
                qp_phys[eid, q] = xq
                qw[eid, q] = wq

                # Normal and phi value from the level set
                g = level_set.gradient(xq)
                # norm_g = np.linalg.norm(g)
                normals[eid, q] = g #/ (norm_g + 1e-30)
                phis[eid, q] = level_set(xq)

                # Jacobian of the parent element at the quadrature point
                xi_ref, eta_ref = transform.inverse_mapping(mesh, eid, xq)
                J = transform.jacobian(mesh, eid, (xi_ref, eta_ref))
                detJ_arr[eid, q] = np.linalg.det(J)
                Jinv_arr[eid, q] = np.linalg.inv(J)
                for fld in fields:
                    b_tabs[fld][eid, q] = me.basis      (fld, xi_ref, eta_ref)
                    g_tabs[fld][eid, q] = me.grad_basis (fld, xi_ref, eta_ref)

        # --- Gather results and cache ---
        out = {
            'eids': np.array(valid_cut_eids, dtype=int),
            # 'eids': np.arange(mesh.n_elements, dtype=int),  # All elements, not just cut
            'qp_phys': qp_phys, 'qw': qw, 'normals': normals, 'phis': phis,
            'detJ': detJ_arr, 'J_inv': Jinv_arr,
        }
        for fld in fields:
            out[f"b_{fld}"] = b_tabs[fld]
            out[f"g_{fld}"] = g_tabs[fld]

>>>>>>> Stashed changes
        if reuse:
            _edge_geom_cache[cache_key] = out
        
        return out
    
    # ---------------------------------------------------------------------
    #  DofHandler.precompute_edge_factors 
    # ---------------------------------------------------------------------
    def precompute_edge_factors(
            self,
            edge_ids: "BitSet | Sequence[int]",
            qdeg: int,
            level_set=None,
            *,
            with_maps: bool = False,
            reuse: bool = True,
        ) -> dict:
        """
        Pre-compute (and cache) every geometric quantity that a facet-based
        JIT kernel may need.  Works for **regular**, **interface** *and*
        **ghost** edges.

        Returns
        -------
        dict
            Keys common to all facets
                qp_phys     (n_e,n_q,2)
                qw          (n_e,n_q)
                normals     (n_e,n_q,2)
                phis        (n_e,n_q)
            +  per-side data for ghost facets
                detJ_pos / detJ_neg      (n_e,n_q)
                J_inv_pos / J_inv_neg    (n_e,n_q,2,2)
            +  optional mapping helpers (only if with_maps=True)
                global_dofs : list[np.ndarray]
                pos_map     : list[np.ndarray]
                neg_map     : list[np.ndarray]
        """
        from pycutfem.integration.quadrature import line_quadrature
        from pycutfem.fem import transform

        mesh   = self.mixed_element.mesh
        ids    = edge_ids.to_indices() if hasattr(edge_ids, "to_indices") else list(edge_ids)

        # ------------------------------------------------------------------ cache
        cache_key = (_hash_subset(ids), qdeg, id(level_set), with_maps)
        if reuse and cache_key in _edge_geom_cache:
            return _edge_geom_cache[cache_key]

        # ------------------------------------------------------------------ allocation
        n_e          = len(ids)
        q_ref, w_ref = line_quadrature((0., 0.), (1., 0.), qdeg)
        n_q          = len(w_ref)

        qp_phys   = np.zeros((n_e, n_q, 2))
        qw        = np.zeros((n_e, n_q))
        normals   = np.zeros((n_e, n_q, 2))
        phis      = np.zeros((n_e, n_q)) if level_set else np.empty(0)

        detJ_pos  = np.zeros((n_e, n_q))
        detJ_neg  = np.zeros((n_e, n_q))
        Jinv_pos  = np.zeros((n_e, n_q, 2, 2))
        Jinv_neg  = np.zeros((n_e, n_q, 2, 2))
        Jinv_mean  = np.zeros((n_e, n_q, 2, 2))
        detJ_mean  = np.zeros((n_e, n_q))

        # (+/–) bookkeeping (needed only for ghost / interface kernels)
        gd_lists, pmaps, nmaps = [], [], []
        pos_eid_arr = np.empty(n_e, dtype=int)
        neg_eid_arr = np.empty(n_e, dtype=int)

        # ------------------------------------------------------------------ loop over facets
        for k, eid_edge in enumerate(ids):
            edge         = mesh.edge(eid_edge)
            p0, p1       = mesh.nodes_x_y_pos[list(edge.nodes)]
            p0, p1       = map(np.asarray, (p0, p1))            # <<< guarant. ndarray
            tang         = p1 - p0
            L_edge       = np.hypot(*tang)

            # ---------------------------------------------------------- classify facet
            tag      = getattr(edge, "tag", "")
            is_iface = tag == "interface"
            is_ghost = tag == "ghost"

            if level_set is None:
                # pure Neumann / Robin edge – outward normal from geometry
                n_vec = np.array([ tang[1], -tang[0] ]) / L_edge
            else:
                # level-set normal (pointing to φ>0)
                mid   = np.asarray(0.5 * (p0 + p1))
                g     = level_set.gradient(mid)
                n_vec = g / (np.linalg.norm(g) + 1e-30)

            # ---------------------------------------------------------- pick integration segment
            if is_iface:
                # interface edges are integrated **once per cut element**;
                # the caller already deduplicated edges → use stored pts
                cut_elem = mesh.elements_list[edge.left] \
                        if mesh.elements_list[edge.left].tag == "cut" \
                        else mesh.elements_list[edge.right]
                seg_p0, seg_p1 = [np.asarray(pt) for pt in cut_elem.interface_pts]
                pos_eid = neg_eid = cut_elem.id
            else:
                # regular or ghost edge – full edge
                seg_p0, seg_p1 = p0, p1
                # determine ± elements from level-set *sign*
                if level_set is None or edge.right is None:
                    pos_eid = edge.left
                    neg_eid = edge.right if edge.right is not None else edge.left
                else:
                    φL = level_set(np.asarray(mesh.elements_list[edge.left ].centroid()))
                    φR = level_set(np.asarray(mesh.elements_list[edge.right].centroid()))
                    pos_eid, neg_eid = (edge.left, edge.right) if φL >= φR else (edge.right, edge.left)

            pos_eid_arr[k] = pos_eid
            neg_eid_arr[k] = neg_eid

            # ---------------------------------------------------------- quadrature
            qpts_phys, qwts = line_quadrature(seg_p0, seg_p1, qdeg)
            for q, (xq, wq) in enumerate(zip(qpts_phys, qwts)):
                xq = np.asarray(xq)                   # ensure ndarray
                qp_phys[k, q] = xq
                qw[k, q]      = wq
                normals[k, q] = n_vec                 # same for both sides
                if level_set is not None:
                    phis[k, q] = level_set(xq)

                # ---- (+) side ---------------------------------------------------
                xi, eta   = transform.inverse_mapping(mesh, pos_eid, xq)
                J_pos     = transform.jacobian(mesh, pos_eid, (xi, eta))
                detJ_pos[k, q] = np.linalg.det(J_pos)
                Jinv_pos[k, q] = np.linalg.inv(J_pos)

                # ---- (–) side ---------------------------------------------------
                xi, eta   = transform.inverse_mapping(mesh, neg_eid, xq)
                J_neg     = transform.jacobian(mesh, neg_eid, (xi, eta))
                detJ_neg[k, q] = np.linalg.det(J_neg)
                Jinv_neg[k, q] = np.linalg.inv(J_neg)
                # ---- mean Jacobian ----------------------------------------------
                Jinv_mean[k, q] = 0.5 * (Jinv_pos[k, q] + Jinv_neg[k, q])
                detJ_mean[k, q] = 0.5 * (detJ_pos[k, q] + detJ_neg[k, q])

            # ---------------------------------------------------------- DOF-maps (ghost / iface)
            if with_maps and (is_ghost or is_iface):
                pos_dofs = self.get_elemental_dofs(pos_eid)
                neg_dofs = self.get_elemental_dofs(neg_eid)
                g_dofs   = np.unique(np.concatenate((pos_dofs, neg_dofs)))
                pmaps.append(np.searchsorted(g_dofs, pos_dofs))
                nmaps.append(np.searchsorted(g_dofs, neg_dofs))
                gd_lists.append(g_dofs)

        # ------------------------------------------------------------------ gather + cache
        out = dict(
            qp_phys = qp_phys,   qw = qw,
            normals = normals,   phis = phis,
            detJ_pos = detJ_pos, detJ_neg = detJ_neg,
            J_inv_pos = Jinv_pos, J_inv_neg = Jinv_neg,
            pos_eid = pos_eid_arr, neg_eid = neg_eid_arr,
            detJ = detJ_mean, J_inv = Jinv_mean
        )
        if with_maps:
            from pycutfem.ufl.helpers_jit import _stack_ragged
            out.update(dict(global_dofs = _stack_ragged(gd_lists), 
                            pos_map = _stack_ragged(pmaps), 
                            neg_map = _stack_ragged(nmaps)
                            ))

        _edge_geom_cache[cache_key] = out
        return out
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes



    
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

