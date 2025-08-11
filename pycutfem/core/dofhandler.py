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
from collections.abc import Sequence
from hashlib import blake2b
from pycutfem.integration.quadrature import line_quadrature
from pycutfem.ufl.helpers_geom import (
    phi_eval, clip_triangle_to_side, fan_triangulate, map_ref_tri_to_phys, corner_tris
)
from pycutfem.utils.bitset import BitSet



BcLike = Union[BoundaryCondition, Mapping[str, Any]]
# ------------------------------------------------------------------
# edge-geometry cache keyed by (hash(ids), qdeg, id(level_set))
# ------------------------------------------------------------------
_edge_geom_cache: dict[tuple[int,int,int], dict] = {}
# geometric volume-cache keyed by (MixedElement.signature(), qdeg, id(level_set) or 0)
_volume_geom_cache: dict[tuple, dict] = {}
# NEW: cut-volume cache (per subset, qdeg, side, derivs, level-set, mixed element)
_cut_volume_cache: dict[tuple, dict] = {}

def clear_caches(self):
    """Clear all precompute caches on the handler."""
    _volume_geom_cache.clear()
    _cut_volume_cache.clear()
    _edge_geom_cache.clear()


def _hash_subset(ids: Sequence[int]) -> int:
    """Stable 64-bit hash for a list / BitSet of indices."""
    h = blake2b(digest_size=8)
    h.update(np.asarray(sorted(ids), dtype=np.int32).tobytes())
    return int.from_bytes(h.digest(), "little")
def _resolve_elem_selection(elem_sel, mesh) -> set[int]:
    """
    Normalize various 'element selection' inputs into a set[int] of element ids.

    Accepts:
      - BitSet:           returns indices where mask is True
      - str (tag name):   looks up mesh element bitset by tag (e.g., 'inside')
      - np.ndarray[bool]: uses truthy positions
      - Iterable[int]:    coerces to ints
    """
    # 1) Direct BitSet
    if isinstance(elem_sel, BitSet):
        return set(map(int, elem_sel.to_indices()))      # BitSet → indices

    # 2) Named tag on this mesh (e.g., 'inside', 'outside', 'cut', or your custom)
    if isinstance(elem_sel, str):
        bs = mesh.element_bitset(elem_sel)               # O(1) if cached
        return set(map(int, bs.to_indices()))            # BitSet → indices

    # 3) Boolean numpy mask
    if hasattr(elem_sel, "__array__"):
        arr = np.asarray(elem_sel)
        if arr.dtype == bool:
            return set(np.flatnonzero(arr))
        # fall through: treat non-bool arrays as list of ids

    # 4) Generic iterable of ids
    try:
        return set(map(int, elem_sel))
    except TypeError as exc:
        raise TypeError(
            "elem_sel must be BitSet, str tag, boolean mask, or iterable of ints"
        ) from exc


# -----------------------------------------------------------------------------
#  Main class
# -----------------------------------------------------------------------------
class DofHandler:
    """Centralised DOF numbering and boundary‑condition helpers."""

    # .........................................................................
    def __init__(self, fe_space: Union[Dict[str, Mesh], 'MixedElement'], method: str = "cg",DEBUG = False):
        if method not in {"cg", "dg"}:
            raise ValueError("method must be 'cg' or 'dg'")
        self.method: str = method
        self.DEBUG: bool = DEBUG
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
    @property
    def union_dofs(self) -> int:
        """Number of distinct global DOFs on one ghost edge (all fields)."""
        if self.mixed_element is None:
            raise RuntimeError("union_dofs requires a MixedElement‑backed DofHandler.")
        if self.method == "cg":
            return self.mixed_element.union_dofs("cg")
        return self.mixed_element.union_dofs("dg")
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
    def tag_dof_by_locator(
            self,
            tag: str,
            field: str,
            locator: Callable[[float, float], bool],
            find_first: bool = True,
            *,
            atol: float = 1e-8,
            rtol: float = 1e-5,
    ):
        """Tag every global DOF of *field* whose node satisfies *locator*."""
        self._require_cg("DOF tagging")
        if field not in self.field_names:
            raise ValueError(f"Unknown field '{field}', choose from {self.field_names}")

        mesh   = self.fe_map[field]
        g2n    = self._dof_to_node_map          # global‑dof → node‑id
        fld_dofs = self.get_field_slice(field)  # all DOFs that belong to *field*

        # ------------------------------------------------------------------ 1. exact match
        matches = []
        for gdof in fld_dofs:
            nid   = g2n[gdof][1]
            x, y  = mesh.nodes_x_y_pos[nid]
            if locator(float(x), float(y)):
                matches.append(gdof)
                if find_first:
                    break

        # ------------------------------------------------------------------ 2. no exact hit? – pick *nearest*
        if not matches:
            # try to guess target (works for lambda x,y: isclose(x,L) & …)
            try:
                free = {v: c.cell_contents
                        for v, c in zip(locator.__code__.co_freevars,
                                        locator.__closure__ or [])}
                x0 = float(free.get("L", free.get("x0")))
                y0 = float(free.get("H", free.get("y0")))/2
            except Exception:
                raise RuntimeError("No node satisfied the locator and "
                                "the target point could not be inferred.")
            coords = mesh.nodes_x_y_pos[[g2n[d][1] for d in fld_dofs]]
            dists  = np.hypot(coords[:, 0] - x0, coords[:, 1] - y0)
            idx    = np.argmin(dists)
            if dists[idx] > (atol + rtol*max(abs(x0), abs(y0))):
                raise RuntimeError("No node close enough to the requested point.")
            matches.append(fld_dofs[idx])

        # ------------------------------------------------------------------ 3. store
        self.dof_tags.setdefault(tag, set()).update(matches)


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
    def precompute_geometric_factors(
        self,
        quad_order: int,
        level_set: Callable = None,
        reuse: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Precompute physical quadrature data (geometry, weights, optional level-set).
        - Caches geometry per (mesh-id, n_elements, element_type, p, quad_order).
        - Uses a Numba kernel for the per-element/per-qp loops when available.
        - Returns:
            qp_phys (nE,nQ,2), qw (nE,nQ), detJ (nE,nQ), J_inv (nE,nQ,2,2),
            normals (nE,nQ,2), phis (nE,nQ or None), h_arr (nE,), eids (nE,),
            entity_kind="element".
        """
        from pycutfem.fem.reference import get_reference  # Ref API (shape, grad)  # noqa: F401
        # volume quadrature, transform are already imported at file top
        # from pycutfem.integration.quadrature import volume
        # from pycutfem.fem import transform

        # Optional numba path -----------------------------------------------------
        try:
            import numba as _nb  # type: ignore
            _HAVE_NUMBA = True
        except Exception:
            _HAVE_NUMBA = False

        if self.mixed_element is None:
            raise RuntimeError("This method requires a MixedElement-backed DofHandler.")

        mesh = self.mixed_element.mesh
        n_el = getattr(mesh, "n_elements", len(mesh.elements_list))
        dim  = mesh.spatial_dim
        if dim != 2:
            raise NotImplementedError("This implementation currently supports 2D only.")

        # ---------------------- cache key (geometry-only) ------------------------
        geom_key = (id(mesh), n_el, mesh.element_type, mesh.poly_order, int(quad_order))
        global _volume_geom_cache
        try:
            _volume_geom_cache
        except NameError:
            _volume_geom_cache = {}

        if reuse and geom_key in _volume_geom_cache:
            geo = _volume_geom_cache[geom_key]
            # On-demand φ(xq) evaluation (cheap vs. geometry)
            phis = None
            if level_set is not None:
                qp = geo["qp_phys"]
                phis = np.empty((qp.shape[0], qp.shape[1]), dtype=np.float64)
                for e in range(qp.shape[0]):
                    for q in range(qp.shape[1]):
                        phis[e, q] = level_set(qp[e, q])
            return {**geo, "phis": phis}

        print(f"Precomputing geometric factors for {mesh.element_type} elements "
              f"with quad_order={quad_order} and p={mesh.poly_order}...")
        # ---------------------- reference quadrature -----------------------------
        qp_ref, qw_ref = volume(mesh.element_type, quad_order)  # (nQ,2), (nQ,)
        qp_ref = np.asarray(qp_ref, dtype=np.float64)
        qw_ref = np.asarray(qw_ref, dtype=np.float64)
        n_q    = qw_ref.shape[0]

        # ---------------------- reference shape/grad tables ----------------------
        ref = get_reference(mesh.element_type, mesh.poly_order)  # Ref provides shape(), grad()  :contentReference[oaicite:9]{index=9}
        # infer n_loc from a single evaluation (Ref has no n_functions)
        n_loc = int(np.asarray(ref.shape(qp_ref[0, 0], qp_ref[0, 1])).size)
        Ntab  = np.empty((n_q, n_loc), dtype=np.float64)        # (nQ, n_loc)
        dNtab = np.empty((n_q, n_loc, 2), dtype=np.float64)     # (nQ, n_loc, 2)
        for q, (xi, eta) in enumerate(qp_ref):
            Ntab[q, :]     = np.asarray(ref.shape(xi, eta), dtype=np.float64).ravel()
            dNtab[q, :, :] = np.asarray(ref.grad(xi, eta),   dtype=np.float64)  # (n_loc,2)  :contentReference[oaicite:10]{index=10}

        # ---------------------- element → node coords (vectorized) ---------------
        # transform.py uses: nodes = mesh.nodes[mesh.elements_connectivity[eid]]
        #                    nodes_x_y_pos[nodes]  → (n_loc,2)                    :contentReference[oaicite:11]{index=11}
        node_ids   = mesh.nodes[mesh.elements_connectivity]            # (nE, n_loc)
        elem_coord = mesh.nodes_x_y_pos[node_ids].astype(np.float64)   # (nE, n_loc, 2)

        # ---------------------- allocate outputs ---------------------------------
        qp_phys = np.zeros((n_el, n_q, 2), dtype=np.float64)
        qw_sc   = np.zeros((n_el, n_q),    dtype=np.float64)
        detJ    = np.zeros((n_el, n_q),    dtype=np.float64)
        J_inv   = np.zeros((n_el, n_q, 2, 2), dtype=np.float64)

        # ---------------------- fast path: Numba ---------------------------------
        if _HAVE_NUMBA:
            @_nb.njit(parallel=True, fastmath=True, cache=True)
            def _geom_kernel(coords, Ntab, dNtab, qwref):
                nE, nLoc, _ = coords.shape
                nQ = qwref.shape[0]
                qp_phys = np.zeros((nE, nQ, 2))
                qw_sc   = np.zeros((nE, nQ))
                detJ    = np.zeros((nE, nQ))
                J_inv   = np.zeros((nE, nQ, 2, 2))
                for e in _nb.prange(nE):
                    for q in range(nQ):
                        # x = Σ_i N_i * X_i
                        x0 = 0.0; x1 = 0.0
                        for i in range(nLoc):
                            Ni = Ntab[q, i]
                            x0 += Ni * coords[e, i, 0]
                            x1 += Ni * coords[e, i, 1]
                        qp_phys[e, q, 0] = x0
                        qp_phys[e, q, 1] = x1
                        # J = dN^T @ X
                        a00 = 0.0; a01 = 0.0; a10 = 0.0; a11 = 0.0
                        for i in range(nLoc):
                            dNix = dNtab[q, i, 0]
                            dNiy = dNtab[q, i, 1]
                            Xix  = coords[e, i, 0]
                            Xiy  = coords[e, i, 1]
                            a00 += dNix * Xix; a01 += dNix * Xiy
                            a10 += dNiy * Xix; a11 += dNiy * Xiy
                        det = a00 * a11 - a01 * a10
                        detJ[e, q] = det
                        inv_det = 1.0 / det
                        J_inv[e, q, 0, 0] =  a11 * inv_det
                        J_inv[e, q, 0, 1] = -a01 * inv_det
                        J_inv[e, q, 1, 0] = -a10 * inv_det
                        J_inv[e, q, 1, 1] =  a00 * inv_det
                        qw_sc[e, q] = qwref[q] * det
                return qp_phys, qw_sc, detJ, J_inv

            qp_phys, qw_sc, detJ, J_inv = _geom_kernel(elem_coord, Ntab, dNtab, qw_ref)

        else:
            # ------------------ safe Python fallback (original logic) ------------
            normals = np.zeros((n_el, n_q, dim), dtype=np.float64)
            phis    = np.zeros((n_el, n_q), dtype=np.float64) if level_set else None
            h_arr   = np.zeros((n_el,), dtype=np.float64)
            eids    = np.arange(n_el, dtype=int)

            for e in range(n_el):
                h_arr[e] = mesh.element_char_length(e)
                for q_idx, (xi_eta, qw) in enumerate(zip(qp_ref, qw_ref)):
                    xi_eta_t = (float(xi_eta[0]), float(xi_eta[1]))
                    J = transform.jacobian(mesh, e, xi_eta_t)
                    det = float(np.linalg.det(J))
                    if det <= 1e-12:
                        raise ValueError(f"Jacobian determinant is non-positive ({det}) for element {e}.")
                    detJ[e, q_idx]  = det
                    J_inv[e, q_idx] = np.linalg.inv(J)
                    qp_phys[e, q_idx] = transform.x_mapping(mesh, e, xi_eta_t)
                    qw_sc[e, q_idx]   = qw * det
                    if phis is not None:
                        phis[e, q_idx] = level_set(qp_phys[e, q_idx])

            geo = {
                "qp_phys": qp_phys, "qw": qw_sc, "detJ": detJ, "J_inv": J_inv,
                "normals": normals, "phis": phis, "h_arr": h_arr, "eids": eids,
                "entity_kind": "element",
            }
            if reuse:
                _volume_geom_cache[geom_key] = {**geo, "phis": None}  # cache geometry only
            return geo

        # ---------------------- post-process & cache -----------------------------
        bad = np.where(detJ <= 1e-12)
        if bad[0].size:
            e_bad, q_bad = int(bad[0][0]), int(bad[1][0])
            raise ValueError(f"Jacobian determinant is non-positive "
                            f"({detJ[e_bad, q_bad]}) for element {e_bad} at qp {q_bad}.")

        normals = np.zeros((n_el, n_q, dim), dtype=np.float64)
        h_arr   = np.empty((n_el,), dtype=np.float64)
        for e in range(n_el):
            h_arr[e] = mesh.element_char_length(e)
        eids = np.arange(n_el, dtype=int)

        geo = {
            "qp_phys": qp_phys, "qw": qw_sc, "detJ": detJ, "J_inv": J_inv,
            "normals": normals, "h_arr": h_arr, "eids": eids,
            "entity_kind": "element",
        }
        geo["owner_id"] = geo["eids"].astype(np.int32)
        if reuse:
            _volume_geom_cache[geom_key] = geo  # cache geometry (no phis)

        phis = None
        if level_set is not None:
            phis = np.empty((n_el, n_q), dtype=np.float64)
            for e in range(n_el):
                for q in range(n_q):
                    phis[e, q] = level_set(qp_phys[e, q])

        return {**geo, "phis": phis}


    
    
    # -------------------------------------------------------------------------
    #  DofHandler.precompute_interface_factors
    # -------------------------------------------------------------------------
     
    def precompute_interface_factors(
        self, cut_element_ids, qdeg: int, level_set, reuse: bool = True
    ) -> dict:
        from pycutfem.integration.quadrature import gauss_legendre, _map_line_rule_batched as _batched  # type: ignore
        from pycutfem.core.levelset import CircleLevelSet, AffineLevelSet
        from pycutfem.core.levelset import _circle_value, _circle_grad, _affine_value, _affine_unit_grad  # type: ignore
        from pycutfem.fem import transform
        from pycutfem.fem.reference import get_reference
        from pycutfem.integration.pre_tabulates import _tabulate_p1, _tabulate_q2, _tabulate_q1

        # ---- numba availability -------------------------------------------------
        try:
            import numba as _nb  # noqa: F401
            _HAVE_NUMBA = True
        except Exception:
            _HAVE_NUMBA = False

        # ---- J from dN and coords (Numba kernel) --------------------------------
        if _HAVE_NUMBA:
            @_nb.njit(cache=True, fastmath=True, parallel=True)
            def _geom_from_dN(coords, dN_tab):
                # coords: (nE, nLoc, 2), dN_tab: (nE, nQ, nLoc, 2)
                nE, nLoc = coords.shape[0], coords.shape[1]
                nQ = dN_tab.shape[1]
                detJ = np.zeros((nE, nQ))
                Jinv = np.zeros((nE, nQ, 2, 2))
                for e in _nb.prange(nE):
                    for q in range(nQ):
                        a00 = 0.0; a01 = 0.0; a10 = 0.0; a11 = 0.0
                        for i in range(nLoc):
                            gx = dN_tab[e, q, i, 0]
                            gy = dN_tab[e, q, i, 1]
                            x = coords[e, i, 0]; y = coords[e, i, 1]
                            a00 += gx * x; a01 += gx * y
                            a10 += gy * x; a11 += gy * y
                        det = a00 * a11 - a01 * a10
                        detJ[e, q] = det
                        inv_det = 1.0 / det
                        Jinv[e, q, 0, 0] =  a11 * inv_det
                        Jinv[e, q, 0, 1] = -a01 * inv_det
                        Jinv[e, q, 1, 0] = -a10 * inv_det
                        Jinv[e, q, 1, 1] =  a00 * inv_det
                return detJ, Jinv

        mesh = self.mixed_element.mesh
        me   = self.mixed_element

        # --- normalize ids → valid cuts with 2 interface points -----------------
        ids = (cut_element_ids.to_indices()
            if hasattr(cut_element_ids, "to_indices")
            else list(cut_element_ids))
        valid_cut_eids = [
            int(eid) for eid in ids
            if (mesh.elements_list[eid].tag == "cut"
                and len(mesh.elements_list[eid].interface_pts) == 2)
        ]
        if not valid_cut_eids:
            z2 = np.empty((0, 0, 2)); z1 = np.empty((0, 0))
            out = {"eids": np.array([], dtype=int), "qp_phys": z2, "qw": z1,
                "normals": z2, "phis": z1, "detJ": z1, "J_inv": np.empty((0, 0, 2, 2)),
                "h_arr": np.empty((0,)), "entity_kind": "element"}
            for fld in me.field_names:
                out[f"b_{fld}"] = z1.reshape(0, 0)
                out[f"g_{fld}"] = np.empty((0, 0, me.n_dofs_local, 2))
            return out

        # --- cache key and lookup ----------------------------------------------
        cache_key = (_hash_subset(valid_cut_eids), int(qdeg), id(level_set))
        if reuse and cache_key in _edge_geom_cache:
            return _edge_geom_cache[cache_key]

        # --- Prepare segments for batched mapping -------------------------------
        nE = len(valid_cut_eids)
        P0 = np.empty((nE, 2), dtype=float)
        P1 = np.empty((nE, 2), dtype=float)
        h_arr = np.empty((nE,), dtype=float)
        for k, eid in enumerate(valid_cut_eids):
            p0, p1 = mesh.elements_list[eid].interface_pts
            P0[k] = p0; P1[k] = p1
            h_arr[k] = mesh.element_char_length(eid)

        xi_1d, w_ref = gauss_legendre(qdeg)
        xi_1d = np.asarray(xi_1d, float); w_ref = np.asarray(w_ref, float)
        nQ = xi_1d.size

        qp_phys = np.empty((nE, nQ, 2), dtype=float)
        qw      = np.empty((nE, nQ),    dtype=float)

        if _HAVE_NUMBA:
            _batched(P0, P1, xi_1d, w_ref, qp_phys, qw)
        else:
            from pycutfem.integration.quadrature import line_quadrature
            for k in range(nE):
                pts, wts = line_quadrature(P0[k], P1[k], qdeg)
                qp_phys[k, :, :] = pts; qw[k, :] = wts

        # --- φ and normals (JIT for common LS) ---------------------------------
        phis    = np.empty((nE, nQ), dtype=float)
        normals = np.empty((nE, nQ, 2), dtype=float)

        if _HAVE_NUMBA and isinstance(level_set, CircleLevelSet):
            cx, cy = float(level_set.center[0]), float(level_set.center[1])
            r = float(level_set.radius)
            for e in range(nE):
                for q in range(nQ):
                    xq = qp_phys[e, q]
                    phis[e, q] = _circle_value(xq, cx, cy, r)
                    normals[e, q] = _circle_grad(xq, cx, cy)
        elif _HAVE_NUMBA and isinstance(level_set, AffineLevelSet):
            a, b, c = level_set.a, level_set.b, level_set.c
            g_unit = _affine_unit_grad(a, b)
            for e in range(nE):
                for q in range(nQ):
                    phis[e, q] = _affine_value(qp_phys[e, q], a, b, c)
                    normals[e, q] = g_unit
        else:
            for e in range(nE):
                for q in range(nQ):
                    xq = qp_phys[e, q]
                    phis[e, q] = level_set(xq)
                    g = level_set.gradient(xq)
                    ng = np.linalg.norm(g)
                    normals[e, q] = g / (ng + 1e-30)

        # --- (ξ,η) at each interface quadrature point --------------------------
        xi_tab  = np.empty((nE, nQ), dtype=float)
        eta_tab = np.empty((nE, nQ), dtype=float)
        for k, eid in enumerate(valid_cut_eids):
            for q in range(nQ):
                xi_eta = transform.inverse_mapping(mesh, eid, qp_phys[k, q])  # fast path if p==1
                xi_tab[k, q]  = float(xi_eta[0])
                eta_tab[k, q] = float(xi_eta[1])

        # --- Reference basis/grad (P1, Q1, Q2 JIT; else generic) ---------------
        elem_type = mesh.element_type
        p_geom    = mesh.poly_order  # geometry/order on the mesh

        # parent element coordinates for geometry/J
        node_ids_all = mesh.nodes[mesh.elements_connectivity]
        coords_all   = mesh.nodes_x_y_pos[node_ids_all].astype(float)
        coords_sel   = coords_all[valid_cut_eids]  # (nE, nLocGeom, 2)
        nLocGeom     = coords_sel.shape[1]
        # Invariant: local-node ordering is row-major (eta rows bottom→top, xi left→right) on quads,
        # and the standard Pk ordering from tri_pn on triangles. Mesh connectivity is built that way
        # (structured_quad/_structured_qn_numba), and tabulators (_tabulate_q1/_tabulate_q2/_tabulate_p1)
        # must emit in exactly the same order. With that, J = dN^T @ X uses consistent indices.


        # Build dN_tab (nE, nQ, nLocGeom, 2) and also N_tab if we can (for basis fill)
        have_jit_ref = False
        if _HAVE_NUMBA and elem_type == "tri" and p_geom == 1:
            # P1 triangle
            N_tab  = np.empty((nE, nQ, 3), dtype=float)
            dN_tab = np.empty((nE, nQ, 3, 2), dtype=float)
            _tabulate_p1(xi_tab, eta_tab, N_tab, dN_tab)
            have_jit_ref = True
        elif _HAVE_NUMBA and elem_type == "quad" and p_geom == 1:
            # Q1 quad
            N_tab  = np.empty((nE, nQ, 4), dtype=float)
            dN_tab = np.empty((nE, nQ, 4, 2), dtype=float)
            _tabulate_q1(xi_tab, eta_tab, N_tab, dN_tab)
            have_jit_ref = True
        elif _HAVE_NUMBA and elem_type == "quad" and p_geom == 2:
            # Q2 quad
            N_tab  = np.empty((nE, nQ, 9), dtype=float)
            dN_tab = np.empty((nE, nQ, 9, 2), dtype=float)
            _tabulate_q2(xi_tab, eta_tab, N_tab, dN_tab)
            have_jit_ref = True
        else:
            # Generic reference (Python) — use Ref object
            ref = get_reference(elem_type, p_geom)
            dN_tab = np.empty((nE, nQ, nLocGeom, 2), dtype=float)
            # N_tab is only needed for basis fallback; allocate lazily below if needed
            for k in range(nE):
                for q in range(nQ):
                    dN_tab[k, q] = np.asarray(ref.grad(xi_tab[k, q], eta_tab[k, q]), dtype=float)

        # --- J, detJ, J_inv from (coords, dN_tab) -------------------------------
        if _HAVE_NUMBA and have_jit_ref:
            detJ, J_inv = _geom_from_dN(coords_sel, dN_tab)
        else:
            detJ = np.empty((nE, nQ), dtype=float)
            J_inv = np.empty((nE, nQ, 2, 2), dtype=float)
            for e in range(nE):
                Xe = coords_sel[e]
                for q in range(nQ):
                    a00 = a01 = a10 = a11 = 0.0
                    for i in range(nLocGeom):
                        gx, gy = dN_tab[e, q, i, 0], dN_tab[e, q, i, 1]
                        x, y = Xe[i, 0], Xe[i, 1]
                        a00 += gx * x; a01 += gx * y
                        a10 += gy * x; a11 += gy * y
                    det = a00 * a11 - a01 * a10
                    detJ[e, q] = det
                    invd = 1.0 / det
                    J_inv[e, q, 0, 0] =  a11 * invd
                    J_inv[e, q, 0, 1] = -a01 * invd
                    J_inv[e, q, 1, 0] = -a10 * invd
                    J_inv[e, q, 1, 1] =  a00 * invd

        # Optional: when debugging local-node order, verify one point
        if getattr(self, "DEBUG", False):
            e0, q0 = 0, 0
            xi_eta0 = (xi_tab[e0, q0], eta_tab[e0, q0])
            J_py = transform.jacobian(mesh, valid_cut_eids[e0], xi_eta0)
            if abs(np.linalg.det(J_py) - detJ[e0, q0]) > 1e-12:
                raise RuntimeError("Local-node ordering mismatch: detJ(py) != detJ(jit).")
        # Basic sanity: non-degenerate mapping
        bad = np.where(detJ <= 1e-12)
        if bad[0].size:
            e_bad, q_bad = int(bad[0][0]), int(bad[1][0])
            raise ValueError(
                f"Jacobian determinant is non-positive ({detJ[e_bad, q_bad]}) "
                f"for element idx {e_bad} at qp {q_bad}."
            )

        # --- Basis & grad-basis on REFERENCE for each field ---------------------
        fields = me.field_names
        b_tabs = {f: np.zeros((nE, nQ, me.n_dofs_local), dtype=float) for f in fields}
        g_tabs = {f: np.zeros((nE, nQ, me.n_dofs_local, 2), dtype=float) for f in fields}

        # We can reuse N_tab for P1/Q1/Q2; otherwise allocate a temp and use MixedElement
        for fld in fields:
            sl = me.component_dof_slices[fld]    # where this field lives in the union vector
            order_f = me._field_orders[fld]      # field polynomial order

            if _HAVE_NUMBA and elem_type == "tri" and order_f == 1 and p_geom == 1 and N_tab.shape[2] == 3:
                b_tabs[fld][:, :, sl]    = N_tab
                g_tabs[fld][:, :, sl, :] = dN_tab
                continue

            if _HAVE_NUMBA and elem_type == "quad" and order_f == 1 and p_geom in (1, 2) and (N_tab.shape[2] == 4):
                # geometry could be Q1 or Q2; field is Q1 (4 local basis)
                # (we already built Q1 N_tab/dN_tab above when p_geom==1; if p_geom==2 and field is Q1,
                #  we still need Q1 reference tables at (xi,eta) – compute them quickly here)
                if p_geom == 2 and N_tab.shape[2] != 4:
                    # make local Q1 tables at the same (xi,eta)
                    Nq1  = np.empty((nE, nQ, 4), dtype=float)
                    dNq1 = np.empty((nE, nQ, 4, 2), dtype=float)
                    _tabulate_q1(xi_tab, eta_tab, Nq1, dNq1)
                    b_tabs[fld][:, :, sl]    = Nq1
                    g_tabs[fld][:, :, sl, :] = dNq1
                else:
                    b_tabs[fld][:, :, sl]    = N_tab
                    g_tabs[fld][:, :, sl, :] = dN_tab
                continue

            if _HAVE_NUMBA and elem_type == "quad" and order_f == 2:
                # ensure we have Q2 ref tables
                if not (have_jit_ref and N_tab.shape[2] == 9):
                    Nq2  = np.empty((nE, nQ, 9), dtype=float)
                    dNq2 = np.empty((nE, nQ, 9, 2), dtype=float)
                    _tabulate_q2(xi_tab, eta_tab, Nq2, dNq2)
                    b_tabs[fld][:, :, sl]    = Nq2
                    g_tabs[fld][:, :, sl, :] = dNq2
                else:
                    b_tabs[fld][:, :, sl]    = N_tab
                    g_tabs[fld][:, :, sl, :] = dN_tab
                continue

            # ---- Generic fallback using MixedElement (reference values) --------
            # If we reach here, use me._eval_scalar_basis/_eval_scalar_grad per point
            for k, eid in enumerate(valid_cut_eids):
                for q in range(nQ):
                    s, t = xi_tab[k, q], eta_tab[k, q]
                    b_tabs[fld][k, q, sl]    = me._eval_scalar_basis(fld, s, t)
                    g_tabs[fld][k, q, sl, :] = me._eval_scalar_grad (fld, s, t)

        out = {
            "eids": np.asarray(valid_cut_eids, dtype=int),
            "qp_phys": qp_phys,
            "qw": qw,
            "normals": normals,
            "phis": phis,
            "detJ": detJ,
            "J_inv": J_inv,
            "h_arr": h_arr,
            "entity_kind": "element",
        }
        out["owner_id"] = out["eids"].astype(np.int32)
        for fld in fields:
            out[f"b_{fld}"] = b_tabs[fld]
            out[f"g_{fld}"] = g_tabs[fld]

        out["J_inv_pos"] = out["J_inv"]      # shape (nE, nQ, 2, 2)
        out["J_inv_neg"] = out["J_inv"]
        n_union = me.n_dofs_local
        out["pos_map"] = np.tile(np.arange(n_union, dtype=np.int32), (len(valid_cut_eids), 1))
        out["neg_map"] = out["pos_map"].copy()

        if reuse:
            _edge_geom_cache[cache_key] = out
        return out



    
    # --------------------------------------------------------------------
    #  DofHandler.precompute_ghost_factors  (new implementation)
    # --------------------------------------------------------------------
    def precompute_ghost_factors(
        self,
        ghost_edge_ids: "BitSet | Sequence[int]",
        qdeg: int,
        level_set,
        derivs: set[tuple[int, int]],
    ) -> dict:
        from pycutfem.integration.quadrature import gauss_legendre, _map_line_rule_batched as _batched  # type: ignore
        from pycutfem.integration.quadrature import line_quadrature
        from pycutfem.fem import transform
        from pycutfem.fem.reference import get_reference
        from pycutfem.integration.pre_tabulates import (
            _searchsorted_positions,
            _tabulate_deriv_q1, _tabulate_deriv_q2, _tabulate_deriv_p1,
        )

        try:
            import numba as _nb
            _HAVE_NUMBA = True
        except Exception:
            _HAVE_NUMBA = False

        # ---- J from dN and coords (Numba kernel) -------------------------------
        if _HAVE_NUMBA:
            @_nb.njit(cache=True, fastmath=True, parallel=True)
            def _geom_from_dN(coords, dN_tab):
                nE, nLoc = coords.shape[0], coords.shape[1]
                nQ = dN_tab.shape[1]
                detJ = np.zeros((nE, nQ))
                Jinv = np.zeros((nE, nQ, 2, 2))
                for e in _nb.prange(nE):
                    for q in range(nQ):
                        a00 = 0.0; a01 = 0.0; a10 = 0.0; a11 = 0.0
                        for i in range(nLoc):
                            gx = dN_tab[e, q, i, 0]
                            gy = dN_tab[e, q, i, 1]
                            x = coords[e, i, 0]; y = coords[e, i, 1]
                            a00 += gx * x; a01 += gx * y
                            a10 += gy * x; a11 += gy * y
                        det = a00 * a11 - a01 * a10
                        detJ[e, q] = det
                        inv_det = 1.0 / det
                        Jinv[e, q, 0, 0] =  a11 * inv_det
                        Jinv[e, q, 0, 1] = -a01 * inv_det
                        Jinv[e, q, 1, 0] = -a10 * inv_det
                        Jinv[e, q, 1, 1] =  a00 * inv_det
                return detJ, Jinv

        mesh = self.mixed_element.mesh
        me   = self.mixed_element
        fields = me.field_names
        n_union = self.union_dofs
        n_loc   = me.n_dofs_per_elem

        # 0) normalise/collect interior ghost edges ------------------------------
        ids = (ghost_edge_ids.to_indices()
            if hasattr(ghost_edge_ids, "to_indices")
            else list(ghost_edge_ids))
        edges = []
        for gid in ids:
            e = mesh.edge(gid)
            if e.right is None:
                continue
            # keep edges if at least one CUT neighbour or already tagged as ghost
            lt = mesh.elements_list[e.left].tag
            rt = mesh.elements_list[e.right].tag
            et = str(getattr(e, "tag", ""))
            if (("cut" in (lt, rt)) or et.startswith("ghost")):
                edges.append(e)
        if not edges:
            raise ValueError("No valid ghost edges found.")

        # 1) batched line rule on each segment ----------------------------------
        nE = len(edges)
        P0 = np.empty((nE, 2), dtype=float)
        P1 = np.empty((nE, 2), dtype=float)
        for i, e in enumerate(edges):
            p0, p1 = mesh.nodes_x_y_pos[list(e.nodes)]
            P0[i] = p0; P1[i] = p1

        xi1, w_ref = gauss_legendre(qdeg)
        xi1 = np.asarray(xi1, float); w_ref = np.asarray(w_ref, float)
        nQ = xi1.size
        qp_phys = np.empty((nE, nQ, 2), dtype=float)
        qw      = np.empty((nE, nQ),    dtype=float)
        if _HAVE_NUMBA:
            _batched(P0, P1, xi1, w_ref, qp_phys, qw)
        else:
            for i in range(nE):
                pts, wts = line_quadrature(P0[i], P1[i], qdeg)
                qp_phys[i, :, :] = pts; qw[i, :] = wts

        # 2) oriented normals & phi ---------------------------------------------
        normals = np.empty((nE, nQ, 2), dtype=float)
        phi_arr = np.zeros((nE, nQ), dtype=float)
        pos_ids = np.empty(nE, dtype=np.int32)
        neg_ids = np.empty(nE, dtype=np.int32)
        for i, e in enumerate(edges):
            # orient from (–) to (+) using the centroid test (compiler path)
            phiL = level_set(np.asarray(mesh.elements_list[e.left].centroid()))
            pos_eid, neg_eid = (e.left, e.right) if phiL >= 0 else (e.right, e.left)
            pos_ids[i] = int(pos_eid)
            neg_ids[i] = int(neg_eid)
            nvec = e.normal
            if np.dot(nvec, mesh.elements_list[pos_eid].centroid() - qp_phys[i, 0]) < 0:
                nvec = -nvec
            for q in range(nQ):
                normals[i, q] = nvec
                phi_arr[i, q] = level_set(qp_phys[i, q])

        # 3) union GDofs and maps (JIT searchsorted) -----------------------------
        gdofs_map = -np.ones((nE, n_union), dtype=np.int64)
        pos_map   = -np.ones((nE, n_loc),   dtype=np.int32)
        neg_map   = -np.ones((nE, n_loc),   dtype=np.int32)

        for i, e in enumerate(edges):
            phiL = level_set(np.asarray(mesh.elements_list[e.left].centroid()))
            pos_eid, neg_eid = (e.left, e.right) if phiL >= 0 else (e.right, e.left)
            pos_dofs = self.get_elemental_dofs(pos_eid)
            neg_dofs = self.get_elemental_dofs(neg_eid)
            global_dofs = np.unique(np.concatenate((pos_dofs, neg_dofs)))
            if global_dofs.size != n_union:
                raise ValueError(f"union size mismatch on edge {e.gid}: {global_dofs.size} vs {n_union}")
            gdofs_map[i, :n_union] = global_dofs
            pos_map[i] = _searchsorted_positions(global_dofs, pos_dofs)
            neg_map[i] = _searchsorted_positions(global_dofs, neg_dofs)

        # 4) (ξ,η) on both sides; build dN (for geometry order) ------------------
        #    and compute J, detJ, J_inv from (coords, dN)
        xi_pos = np.empty((nE, nQ)); eta_pos = np.empty((nE, nQ))
        xi_neg = np.empty((nE, nQ)); eta_neg = np.empty((nE, nQ))
        pos_ids = np.empty(nE, dtype=np.int32)
        neg_ids = np.empty(nE, dtype=np.int32)

        for i, e in enumerate(edges):
            phiL = level_set(np.asarray(mesh.elements_list[e.left].centroid()))
            pos_eid, neg_eid = (e.left, e.right) if phiL >= 0 else (e.right, e.left)
            pos_ids[i] = pos_eid; neg_ids[i] = neg_eid
            for q in range(nQ):
                s, t = transform.inverse_mapping(mesh, pos_eid, qp_phys[i, q])
                xi_pos[i, q]  = float(s); eta_pos[i, q] = float(t)
                s, t = transform.inverse_mapping(mesh, neg_eid, qp_phys[i, q])
                xi_neg[i, q]  = float(s); eta_neg[i, q] = float(t)

        # coordinates of parent elements
        node_ids_all = mesh.nodes[mesh.elements_connectivity]
        coords_all   = mesh.nodes_x_y_pos[node_ids_all].astype(float)
        coords_pos   = coords_all[pos_ids]
        coords_neg   = coords_all[neg_ids]
        nLocGeom     = coords_pos.shape[1]

        # reference geometry dN tables
        from pycutfem.integration.pre_tabulates import _tabulate_q1 as _tab_q1
        from pycutfem.integration.pre_tabulates import _tabulate_q2 as _tab_q2
        from pycutfem.integration.pre_tabulates import _tabulate_p1 as _tab_p1

        dN_pos = np.empty((nE, nQ, nLocGeom, 2), dtype=float)
        dN_neg = np.empty((nE, nQ, nLocGeom, 2), dtype=float)

        if mesh.element_type == "quad" and mesh.poly_order == 1:
            _tab_q1(xi_pos, eta_pos, np.empty((nE, nQ, 4)), dN_pos)
            _tab_q1(xi_neg, eta_neg, np.empty((nE, nQ, 4)), dN_neg)
        elif mesh.element_type == "quad" and mesh.poly_order == 2:
            _tab_q2(xi_pos, eta_pos, np.empty((nE, nQ, 9)), dN_pos)
            _tab_q2(xi_neg, eta_neg, np.empty((nE, nQ, 9)), dN_neg)
        elif mesh.element_type == "tri"  and mesh.poly_order == 1:
            _tab_p1(xi_pos, eta_pos, np.empty((nE, nQ, 3)), dN_pos)
            _tab_p1(xi_neg, eta_neg, np.empty((nE, nQ, 3)), dN_neg)
        else:
            # generic Python fallback for geometry dN
            ref = get_reference(mesh.element_type, mesh.poly_order)
            for i in range(nE):
                for q in range(nQ):
                    dN_pos[i, q] = np.asarray(ref.grad(xi_pos[i, q], eta_pos[i, q]))
                    dN_neg[i, q] = np.asarray(ref.grad(xi_neg[i, q], eta_neg[i, q]))

        # build J, detJ, J_inv
        if _HAVE_NUMBA:
            detJ_pos, J_inv_pos = _geom_from_dN(coords_pos, dN_pos)
            detJ_neg, J_inv_neg = _geom_from_dN(coords_neg, dN_neg)
        else:
            detJ_pos = np.empty((nE, nQ)); J_inv_pos = np.empty((nE, nQ, 2, 2))
            detJ_neg = np.empty((nE, nQ)); J_inv_neg = np.empty((nE, nQ, 2, 2))
            for side, coords, dN, detJ, J_inv in (
                ("pos", coords_pos, dN_pos, detJ_pos, J_inv_pos),
                ("neg", coords_neg, dN_neg, detJ_neg, J_inv_neg),
            ):
                for e in range(nE):
                    Xe = coords[e]
                    for q in range(nQ):
                        a00 = a01 = a10 = a11 = 0.0
                        for i in range(nLocGeom):
                            gx, gy = dN[e, q, i, 0], dN[e, q, i, 1]
                            x, y = Xe[i, 0], Xe[i, 1]
                            a00 += gx * x; a01 += gx * y
                            a10 += gy * x; a11 += gy * y
                        det = a00 * a11 - a01 * a10
                        detJ[e, q] = det
                        invd = 1.0 / det
                        J_inv[e, q, 0, 0] =  a11 * invd
                        J_inv[e, q, 0, 1] = -a01 * invd
                        J_inv[e, q, 1, 0] = -a10 * invd
                        J_inv[e, q, 1, 1] =  a00 * invd

        # sanity: non-degenerate
        bad = np.where((detJ_pos <= 1e-12) | (detJ_neg <= 1e-12))
        if bad[0].size:
            e_bad, q_bad = int(bad[0][0]), int(bad[1][0])
            raise ValueError(f"Non-positive detJ at edge {edges[e_bad].gid}, qp {q_bad}")

        # 5) tabulate reference derivatives (per field) and push-forward ----------
        derivs = set(derivs) | {(0, 0)}  # ensure φ itself
        basis_tables = {}
        for fld in fields:
            p_f = me._field_orders[fld]
            sl  = me.component_dof_slices[fld]  # slice into union-local
            n_f = sl.stop - sl.start

            # choose tabulator for this field
            kind = (mesh.element_type, p_f)
            def _fill(side, xi_tab, eta_tab, J_inv):
                # sx, sy from J_inv (diagonal assumption consistent with earlier code)
                sx = J_inv[..., 0, 0]; sy = J_inv[..., 1, 1]
                for (dx, dy) in derivs:
                    key = f"d{dx}{dy}_{fld}_{side}"
                    arr = np.zeros((nE, nQ, me.n_dofs_per_elem))
                    # local field block (n_f,)
                    loc = np.empty((nE, nQ, n_f))
                    if _HAVE_NUMBA and kind == ("quad", 1):
                        _tabulate_deriv_q1(xi_tab, eta_tab, dx, dy, loc)
                    elif _HAVE_NUMBA and kind == ("quad", 2):
                        _tabulate_deriv_q2(xi_tab, eta_tab, dx, dy, loc)
                    elif _HAVE_NUMBA and kind == ("tri", 1):
                        _tabulate_deriv_p1(xi_tab, eta_tab, dx, dy, loc)
                    else:
                        # generic fallback with per-point MixedElement call
                        for e in range(nE):
                            for q in range(nQ):
                                loc[e, q, :] = me._eval_scalar_deriv(fld, xi_tab[e, q], eta_tab[e, q], dx, dy)

                    # map to physical (structured scaling used in your ghost path)
                    phys = loc * (sx ** dx)[:, :, None] * (sy ** dy)[:, :, None]
                    arr[:, :, sl] = phys
                    basis_tables[key] = arr

            _fill("pos", xi_pos, eta_pos, J_inv_pos)
            _fill("neg", xi_neg, eta_neg, J_inv_neg)

        # 6) element sizes and pack
        h_arr = np.empty((nE,), dtype=float)
        for i, e in enumerate(edges):
            hL = mesh.element_char_length(e.left)
            hR = mesh.element_char_length(e.right)
            h_arr[i] = max(hL, hR)

        out = {
            "eids":        np.fromiter((e.gid for e in edges), dtype=np.int32),
            "qp_phys":     qp_phys,
            "qw":          qw,
            "normals":     normals,
            "gdofs_map":   gdofs_map,
            "pos_map":     pos_map,
            "neg_map":     neg_map,
            "J_inv_pos":   J_inv_pos,
            "J_inv_neg":   J_inv_neg,
            "detJ_pos":    detJ_pos,
            "detJ_neg":    detJ_neg,
            "detJ":        0.5*(detJ_pos + detJ_neg),
            "J_inv":       0.5*(J_inv_pos + J_inv_neg),
            "phis":        phi_arr,
            "h_arr":       h_arr,
            "entity_kind": "edge",
            "owner_pos_id": pos_ids,
            "owner_neg_id": neg_ids,
            # convenience single-owner for simple indexing paths
            "owner_id":     pos_ids,

        }
        out.update(basis_tables)
        return out

    
    # --------------------------------------------------------------------
    #  DofHandler.precompute_boundary_factors   (∫ ⋯ dS backend)
    # --------------------------------------------------------------------
    def precompute_boundary_factors(
        self,
        edge_ids: "BitSet | Sequence[int]",
        qdeg: int,
        derivs: set[tuple[int, int]],
        reuse: bool = True,
    ) -> dict:
        """
        Pre-compute geometry & basis tables for ∫_Γ ⋯ dS on *boundary* edges.
        Returns per-edge arrays sized to the given subset and ready for JIT.
        """
        from pycutfem.integration.quadrature import gauss_legendre, _map_line_rule_batched as _batched  # type: ignore
        from pycutfem.integration.quadrature import line_quadrature
        from pycutfem.fem import transform
        from pycutfem.fem.reference import get_reference
        from pycutfem.integration.pre_tabulates import (
            _tabulate_q1, _tabulate_q2, _tabulate_p1,
            _tabulate_deriv_q1, _tabulate_deriv_q2, _tabulate_deriv_p1,
        )

        try:
            import numba as _nb
            _HAVE_NUMBA = True
        except Exception:
            _HAVE_NUMBA = False

        # --- J from dN and coords (Numba kernel) ---------------------------------
        if _HAVE_NUMBA:
            @_nb.njit(cache=True, fastmath=True, parallel=True)
            def _geom_from_dN(coords, dN_tab):
                nE, nLoc = coords.shape[0], coords.shape[1]
                nQ = dN_tab.shape[1]
                detJ = np.zeros((nE, nQ))
                Jinv = np.zeros((nE, nQ, 2, 2))
                for e in _nb.prange(nE):
                    for q in range(nQ):
                        a00 = 0.0; a01 = 0.0; a10 = 0.0; a11 = 0.0
                        for i in range(nLoc):
                            gx = dN_tab[e, q, i, 0]
                            gy = dN_tab[e, q, i, 1]
                            x = coords[e, i, 0]; y = coords[e, i, 1]
                            a00 += gx * x; a01 += gx * y
                            a10 += gy * x; a11 += gy * y
                        det = a00 * a11 - a01 * a10
                        detJ[e, q] = det
                        inv_det = 1.0 / det
                        Jinv[e, q, 0, 0] =  a11 * inv_det
                        Jinv[e, q, 0, 1] = -a01 * inv_det
                        Jinv[e, q, 1, 0] = -a10 * inv_det
                        Jinv[e, q, 1, 1] =  a00 * inv_det
                return detJ, Jinv

        mesh = self.mixed_element.mesh
        me   = self.mixed_element
        fields = me.field_names

        # ---- normalise / filter to boundary edges (right is None) ---------------
        if hasattr(edge_ids, "to_indices"):
            edge_ids = edge_ids.to_indices()
        edge_ids = [int(e) for e in edge_ids if mesh.edge(e).right is None]
        if not edge_ids:
            return {"eids": np.empty(0, dtype=int)}  # nothing to do

        # ---- reuse cache if possible --------------------------------------------
        global _edge_geom_cache
        try:
            _edge_geom_cache
        except NameError:
            _edge_geom_cache = {}
        cache_key = (_hash_subset(edge_ids), int(qdeg), tuple(sorted(derivs)))
        if reuse and cache_key in _edge_geom_cache:
            return _edge_geom_cache[cache_key]

        # ---- sizes ---------------------------------------------------------------
        n_edges = len(edge_ids)
        # representative to size arrays
        p0r, p1r = mesh.nodes_x_y_pos[list(mesh.edge(edge_ids[0]).nodes)]
        qpr, qwr = line_quadrature(p0r, p1r, qdeg)
        n_q    = len(qwr)
        n_loc  = me.n_dofs_local

        # ---- work arrays ---------------------------------------------------------
        qp_phys  = np.zeros((n_edges, n_q, 2))
        qw       = np.zeros((n_edges, n_q))
        normals  = np.zeros((n_edges, n_q, 2))
        detJ     = np.zeros((n_edges, n_q))
        J_inv    = np.zeros((n_edges, n_q, 2, 2))
        phis     = None  # boundary dS: no level-set needed
        gdofs_map = np.zeros((n_edges, n_loc), dtype=np.int32)
        h_arr     = np.zeros((n_edges,))
        # derivative tables (union-sized)
        basis_tabs = {f"d{dx}{dy}_{fld}": np.zeros((n_edges, n_q, n_loc))
                    for fld in fields for (dx, dy) in derivs}

        # ---- batched edge mapping ------------------------------------------------
        xi1, w_ref = gauss_legendre(qdeg)
        xi1 = np.asarray(xi1, float); w_ref = np.asarray(w_ref, float)
        P0 = np.empty((n_edges, 2)); P1 = np.empty((n_edges, 2))
        for i, gid in enumerate(edge_ids):
            n0, n1 = mesh.edge(gid).nodes
            P0[i], P1[i] = mesh.nodes_x_y_pos[n0], mesh.nodes_x_y_pos[n1]
        if _HAVE_NUMBA:
            _batched(P0, P1, xi1, w_ref, qp_phys, qw)
        else:
            for i in range(n_edges):
                pts, wts = line_quadrature(P0[i], P1[i], qdeg)
                qp_phys[i], qw[i] = pts, wts

        # ---- normals & dof maps (owner is 'left') -------------------------------
        for i, gid in enumerate(edge_ids):
            e = mesh.edge(gid)
            eid = e.left
            normals[i, :, :] = e.normal  # constant along edge; outward for owner
            gdofs_map[i, :]  = self.get_elemental_dofs(eid)
            h_arr[i]         = mesh.element_char_length(eid)

        # ---- (ξ,η) tables on the owner element ---------------------------------
        xi_tab  = np.empty((n_edges, n_q))
        eta_tab = np.empty((n_edges, n_q))
        eids_arr = np.empty((n_edges,), dtype=np.int32)
        for i, gid in enumerate(edge_ids):
            eid = mesh.edge(gid).left
            eids_arr[i] = eid
            for q in range(n_q):
                s, t = transform.inverse_mapping(mesh, eid, qp_phys[i, q])
                xi_tab[i, q]  = float(s)
                eta_tab[i, q] = float(t)

        # ---- reference dN for geometry order; build J, detJ, J⁻¹ ----------------
        node_ids_all = mesh.nodes[mesh.elements_connectivity]
        coords_all   = mesh.nodes_x_y_pos[node_ids_all].astype(float)
        coords_sel   = coords_all[eids_arr]
        nLocGeom     = coords_sel.shape[1]
        dN_tab = np.empty((n_edges, n_q, nLocGeom, 2))
        if mesh.element_type == "quad" and mesh.poly_order == 1:
            _tabulate_q1(xi_tab, eta_tab, np.empty((n_edges, n_q, 4)), dN_tab)
        elif mesh.element_type == "quad" and mesh.poly_order == 2:
            _tabulate_q2(xi_tab, eta_tab, np.empty((n_edges, n_q, 9)), dN_tab)
        elif mesh.element_type == "tri" and mesh.poly_order == 1:
            _tabulate_p1(xi_tab, eta_tab, np.empty((n_edges, n_q, 3)), dN_tab)
        else:
            ref = get_reference(mesh.element_type, mesh.poly_order)
            for i in range(n_edges):
                for q in range(n_q):
                    dN_tab[i, q] = np.asarray(ref.grad(xi_tab[i, q], eta_tab[i, q]), dtype=float)

        if _HAVE_NUMBA:
            detJ[:], J_inv[:] = _geom_from_dN(coords_sel, dN_tab)
        else:
            for e in range(n_edges):
                Xe = coords_sel[e]
                for q in range(n_q):
                    a00 = a01 = a10 = a11 = 0.0
                    for iL in range(nLocGeom):
                        gx, gy = dN_tab[e, q, iL, 0], dN_tab[e, q, iL, 1]
                        x, y = Xe[iL, 0], Xe[iL, 1]
                        a00 += gx * x; a01 += gx * y
                        a10 += gy * x; a11 += gy * y
                    d = a00 * a11 - a01 * a10
                    detJ[e, q] = d
                    invd = 1.0 / d
                    J_inv[e, q, 0, 0] =  a11 * invd
                    J_inv[e, q, 0, 1] = -a01 * invd
                    J_inv[e, q, 1, 0] = -a10 * invd
                    J_inv[e, q, 1, 1] =  a00 * invd

        # ---- tabulate derivatives per field and push-forward --------------------
        derivs = set(derivs)  # ensure (0,0) is included if needed by the kernel build
        for fld in fields:
            sl  = me.component_dof_slices[fld]
            p_f = me._field_orders[fld]
            # choose tabulator for field
            if mesh.element_type == "quad" and p_f == 1:
                tab = _tabulate_deriv_q1
                n_f = 4
            elif mesh.element_type == "quad" and p_f == 2:
                tab = _tabulate_deriv_q2
                n_f = 9
            elif mesh.element_type == "tri" and p_f == 1:
                tab = _tabulate_deriv_p1
                n_f = 3
            else:
                tab = None
                n_f = sl.stop - sl.start

            for (dx, dy) in derivs:
                key = f"d{dx}{dy}_{fld}"
                loc = np.empty((n_edges, n_q, n_f))
                if tab is not None:
                    tab(xi_tab, eta_tab, int(dx), int(dy), loc)
                else:
                    # generic fallback via MixedElement (reference)
                    for e in range(n_edges):
                        for q in range(n_q):
                            loc[e, q, :] = me._eval_scalar_deriv(fld, xi_tab[e, q], eta_tab[e, q], dx, dy)

                # diagonal push-forward used elsewhere in boundary/ghost paths
                sx = J_inv[:, :, 0, 0]; sy = J_inv[:, :, 1, 1]
                phys = loc * (sx ** dx)[:, :, None] * (sy ** dy)[:, :, None]
                basis_tabs[key][:, :, sl] = phys  # scatter into union block

        out = {
            "eids":      np.asarray(edge_ids, dtype=np.int32),
            "qp_phys":   qp_phys,
            "qw":        qw,
            "normals":   normals,
            "detJ":      detJ,          # neutral for dS in kernels that don’t use it
            "J_inv":     J_inv,
            "phis":      phis,          # not used for exterior_facet
            "gdofs_map": gdofs_map,
            "h_arr":     h_arr,
            "entity_kind": "edge",
        }
        out.update(basis_tabs)
        owner_id = np.asarray([self.mesh.edge(eid).left for eid in edge_ids], dtype=np.int32)
        out["owner_id"] = owner_id

        if reuse:
            _edge_geom_cache[cache_key] = out
        return out

    

    # --- cut volume --------------------------------------

    def precompute_cut_volume_factors(
        self,
        element_bitset,
        qdeg: int,
        derivs: set[tuple[int, int]],
        level_set,
        side: str = "+",
        reuse: bool = True,
    ) -> dict:
        """
        Pre-compute geometry/basis for ∫_{Ω ∩ {φ ▷ 0}} (…) dx on CUT elements.
        Returns padded arrays shaped (n_cut, max_q, ...), aligned with 'eids'.
        Keys: eids, qp_phys, qw, J_inv, detJ, phis, normals(zeros),
            and b_*, g_* plus d{dx}{dy}_* (requested).
        """
        from pycutfem.integration.quadrature import tri_rule as tri_volume_rule
        from pycutfem.fem import transform
        from pycutfem.integration.pre_tabulates import (
            _eval_deriv_q1, _eval_deriv_q2, _eval_deriv_p1,
        )
        from pycutfem.ufl.helpers_geom import (
            phi_eval, corner_tris,
            _clip_triangle_to_side_numba, _fan_triangulate_numba,
            _map_ref_tri_to_phys_numba,
        )
        try:
            import numba as _nb  # noqa: F401
            _HAVE_NUMBA = True
        except Exception:
            _HAVE_NUMBA = False

        mesh = self.mixed_element.mesh
        me   = self.mixed_element
        fields = list(me.field_names)

        # collect candidate element ids (CUT only)
        if hasattr(element_bitset, "to_indices"):
            eids_all = element_bitset.to_indices()
        else:
            eids_all = list(element_bitset)
        eids_all = [eid for eid in eids_all if 0 <= eid < mesh.n_elements and mesh.elements_list[eid].tag == "cut"]
        if not eids_all:
            return {
                "eids": np.array([], dtype=np.int32),
                "qp_phys": np.empty((0, 0, 2)), "qw": np.empty((0, 0)),
                "J_inv": np.empty((0, 0, 2, 2)), "detJ": np.empty((0, 0)),
                "normals": np.empty((0, 0, 2)), "phis": np.empty((0, 0)),
                **{f"b_{f}": np.empty((0, 0, me.n_dofs_local)) for f in fields},
                **{f"g_{f}": np.empty((0, 0, me.n_dofs_local, 2)) for f in fields},
            }
        # ---------- NEW: cache key & early return ----------
        # Use a stable subset hash + quadrature + side + requested derivs + mixed-element signature
        # Level set: prefer a user-provided token if available; otherwise its id().
        try:
            subset_hash = _hash_subset(eids_all)  # same helper you use elsewhere
        except NameError:
            subset_hash = tuple(eids_all)         # fallback if helper not imported here
        ls_token = getattr(level_set, "cache_token", None)
        if ls_token is None:
            ls_token = id(level_set)
        # BUG: Key logic needs to be updated for moving level sets
        key = (
            "cutvol",
            id(mesh),
            self.mixed_element.signature(),
            subset_hash,
            int(qdeg),
            str(side),
            tuple(sorted(derivs)),
            ls_token,
        )
        if reuse and key in _cut_volume_cache:
            return _cut_volume_cache[key]

        # reference triangle rule (0,0)-(1,0)-(0,1)
        qp_ref_tri, qw_ref_tri = tri_volume_rule(qdeg)  # on reference triangle
        qp_ref_tri = np.asarray(qp_ref_tri, dtype=float)
        qw_ref_tri = np.asarray(qw_ref_tri, dtype=float)
        nQ_ref = qp_ref_tri.shape[0]

        # ragged accumulators
        valid_eids  = []
        qp_blocks   = []
        qw_blocks   = []
        Jinv_blocks = []
        phi_blocks  = []
        basis_lists: dict[str, list[np.ndarray]] = {}

        sgn = +1 if side == "+" else -1

        for eid in eids_all:
            elem = mesh.elements_list[eid]
            tri_local, cn = corner_tris(mesh, elem)  # fixed order (row-major / tri_pn)
            xq_elem, wq_elem, Jinv_elem, phi_elem = [], [], [], []
            per_elem_basis: dict[str, list[np.ndarray]] = {}

            for loc_tri in tri_local:
                v_ids = [cn[i] for i in loc_tri]               # 3 corner ids
                V     = mesh.nodes_x_y_pos[v_ids].astype(float)  # (3,2)
                v_phi = np.array([phi_eval(level_set, V[0]),
                                phi_eval(level_set, V[1]),
                                phi_eval(level_set, V[2])], dtype=float)

                # clip triangle to requested side in JIT
                poly, n_pts = _clip_triangle_to_side_numba(V, v_phi, sgn)
                if n_pts < 3:
                    continue

                # fan-triangulate (0,1,2) or (0,1,2)+(0,2,3)
                tris, n_tris = _fan_triangulate_numba(poly, n_pts)
                for t in range(n_tris):
                    A = tris[t, 0]; B = tris[t, 1]; C = tris[t, 2]
                    x_phys, w_phys = _map_ref_tri_to_phys_numba(A, B, C, qp_ref_tri, qw_ref_tri)

                    # loop physical QPs
                    for q in range(nQ_ref):
                        x = x_phys[q]; w = w_phys[q]
                        # (ξ,η) on the parent element and J^{-1}
                        xi, eta = transform.inverse_mapping(mesh, eid, x)
                        J       = transform.jacobian(mesh, eid, (xi, eta))
                        Ji      = np.linalg.inv(J)

                        xq_elem.append(x)
                        wq_elem.append(w)
                        Jinv_elem.append(Ji)
                        phi_elem.append(phi_eval(level_set, x))

                        # reference tables per field at (xi,eta)
                        for fld in fields:
                            kb, kg = f"b_{fld}", f"g_{fld}"
                            per_elem_basis.setdefault(kb, []).append(me.basis(fld, xi, eta))
                            per_elem_basis.setdefault(kg, []).append(me.grad_basis(fld, xi, eta))
                            # extra derivatives up to order 2 (union-sized)
                            for (dx, dy) in derivs:
                                if (dx, dy) == (0, 0):
                                    continue
                                kd = f"d{dx}{dy}_{fld}"
                                per_elem_basis.setdefault(kd, []).append(me.deriv_ref(fld, xi, eta, dx, dy))

            if not wq_elem:
                continue

            valid_eids.append(eid)
            qp_blocks.append(np.asarray(xq_elem, dtype=float))       # (n_q,2)
            qw_blocks.append(np.asarray(wq_elem, dtype=float))       # (n_q,)
            Jinv_blocks.append(np.asarray(Jinv_elem, dtype=float))   # (n_q,2,2)
            phi_blocks.append(np.asarray(phi_elem, dtype=float))     # (n_q,)

            # finalize per-element basis arrays
            for key, rows in per_elem_basis.items():
                r0 = rows[0]
                if r0.ndim == 1:                  # b_* and d.._* → (n_q, n_loc)
                    arr = np.vstack(rows).astype(float)
                elif r0.ndim == 2:                # g_* rows are (n_loc,2) → stack to (n_q, n_loc, 2)
                    arr = np.stack(rows, axis=0).astype(float)
                else:
                    raise ValueError(f"Unexpected rank for {key}: {r0.shape}")
                basis_lists.setdefault(key, []).append(arr)

        if not valid_eids:
            return {
                "eids": np.array([], dtype=np.int32),
                "qp_phys": np.empty((0, 0, 2)), "qw": np.empty((0, 0)),
                "J_inv": np.empty((0, 0, 2, 2)), "detJ": np.empty((0, 0)),
                "normals": np.empty((0, 0, 2)), "phis": np.empty((0, 0)),
            }

        # ---- pad ragged → rectangular (qw=0 on padding) -------------------------
        nE   = len(valid_eids)
        sizes = np.array([blk.shape[0] for blk in qw_blocks], dtype=int)
        Qmax = int(sizes.max())

        def _pad(arrs, shape_tail):
            out = np.zeros((nE, Qmax, *shape_tail), dtype=float)
            for i, a in enumerate(arrs):
                n = a.shape[0]
                if a.ndim == 1:
                    out[i, :n, 0] = a
                else:
                    out[i, :n, ...] = a
            return out

        qp_phys = np.zeros((nE, Qmax, 2), dtype=float)
        qw      = np.zeros((nE, Qmax),    dtype=float)
        J_inv   = np.zeros((nE, Qmax, 2, 2), dtype=float)
        phis    = np.zeros((nE, Qmax),    dtype=float)

        for i in range(nE):
            n = qp_blocks[i].shape[0]
            qp_phys[i, :n, :] = qp_blocks[i]
            qw[i, :n]         = qw_blocks[i]
            J_inv[i, :n, :, :] = Jinv_blocks[i]
            phis[i, :n]       = phi_blocks[i]

        out = {
            "eids":   np.asarray(valid_eids, dtype=np.int32),
            "qp_phys": qp_phys,
            "qw":      qw,
            "J_inv":   J_inv,
            "detJ":    np.ones_like(qw),          # not used (weights already physical)
            "normals": np.zeros_like(qp_phys),    # not used in volume dx
            "phis":    phis,
            "entity_kind": "element",
        }
        out["owner_id"] = out["eids"].astype(np.int32)

        # pad and attach per-field tables
        for fld in fields:
            # b_*
            blk_list = basis_lists.get(f"b_{fld}", [])
            b_pad = np.zeros((nE, Qmax, me.n_dofs_local), dtype=float)
            for i, arr in enumerate(blk_list):
                b_pad[i, :arr.shape[0], :] = arr
            out[f"b_{fld}"] = b_pad

            # g_* (reference gradients)
            blk_list = basis_lists.get(f"g_{fld}", [])
            g_pad = np.zeros((nE, Qmax, me.n_dofs_local, 2), dtype=float)
            for i, arr in enumerate(blk_list):
                g_pad[i, :arr.shape[0], :, :] = arr
            out[f"g_{fld}"] = g_pad

            # d{dx}{dy}_*
            for (dx, dy) in derivs:
                if (dx, dy) == (0, 0):
                    continue
                key = f"d{dx}{dy}_{fld}"
                blk_list = basis_lists.get(key, [])
                d_pad = np.zeros((nE, Qmax, me.n_dofs_local), dtype=float)
                for i, arr in enumerate(blk_list):
                    d_pad[i, :arr.shape[0], :] = arr
                out[key] = d_pad

        if reuse:
            _cut_volume_cache[key] = out
        return out


    def tag_dofs_from_element_bitset(
        self,
        tag: str,
        field: str,
        elem_sel,                 # ← BitSet | str | bool mask | Iterable[int]
        *,
        strict: bool = True,
    ) -> set[int]:
        if field not in self.field_names:
            raise ValueError(f"Unknown field '{field}', choose from {self.field_names}")

        mesh = self.fe_map[field]

        # --- clean resolution of element ids
        inside_eids = _resolve_elem_selection(elem_sel, mesh)

        # --- DG: tag all DOFs from those elements directly
        if getattr(self, "_dg_mode", False):
            elem_maps = self.element_maps[field]
            nE = len(elem_maps)
            dofs = set()
            for eid in inside_eids:
                if 0 <= eid < nE:
                    dofs.update(elem_maps[eid])
            self.dof_tags.setdefault(tag, set()).update(dofs)
            return dofs

        # --- CG: candidate nodes = union of nodes from the selected elements
        candidate_nodes: set[int] = set()
        for eid in inside_eids:
            el = mesh.elements_list[eid]
            candidate_nodes.update(el.nodes)

        if strict:
            # Build node → {adjacent eids} once, then keep only nodes whose
            # adjacent elements are all in 'inside_eids'
            node_to_elems: dict[int, set[int]] = {}
            for el in mesh.elements_list:
                for nid in el.nodes:
                    node_to_elems.setdefault(nid, set()).add(el.id)

            inside_eids_set = set(inside_eids)
            nodes = {
                nid for nid in candidate_nodes
                if node_to_elems.get(nid, set()).issubset(inside_eids_set)
            }
        else:
            nodes = candidate_nodes

        # Map nodes → DOFs for this field
        node_to_dof = self.dof_map[field]
        dofs = {node_to_dof[nid] for nid in nodes if nid in node_to_dof}

        self.dof_tags.setdefault(tag, set()).update(dofs)
        return dofs


    def l2_error_on_side(
        self,
        functions,                      # VectorFunction, Function, or dict[str, Function]
        exact: dict[str, callable],     # e.g. {'ux': u_ex_x, 'uy': u_ex_y, 'p': p_ex}
        level_set,                      # φ(x,y) callable
        side: str = '-',                # '-' → φ<0  (inside),  '+' → φ>0 (outside)
        quad_order: int | None = None,
        fields: list[str] | None = None,
        relative: bool = False,
    ) -> float:
        """
        L2-norm of specified fields over Ω_side := Ω ∩ { φ ▷ 0 }, with ▷='<' if side='-', or '>' if side='+'.

        * Fast pure-Python path:
        - full elements: standard reference quadrature + detJ
        - cut elements : clip corner-triangles and integrate physical sub-tris (weights already physical)
        * 'functions' can be a VectorFunction, a single Function, or a dict {'ux': Fun, 'uy': Fun, ...}.
        * 'exact' supplies exact scalar callables per field to compare against.
        """
        import numpy as np
        from pycutfem.fem import transform
        from pycutfem.integration.quadrature import volume as vol_rule
        from pycutfem.ufl.helpers_geom import (
            corner_tris, clip_triangle_to_side, fan_triangulate, map_ref_tri_to_phys, phi_eval
        )

        mesh = self.mixed_element.mesh
        me   = self.mixed_element
        qdeg = quad_order or (2 * mesh.poly_order + 2)

        # --------------- normalize 'functions' → dict[field]->Function -------------
        from pycutfem.ufl.expressions import Function, VectorFunction
        field_funcs: dict[str, Function] = {}

        if isinstance(functions, dict):
            field_funcs = functions
        elif isinstance(functions, VectorFunction):
            for name, comp in zip(functions.field_names, functions.components):
                field_funcs[name] = comp
        elif isinstance(functions, (list, tuple)) and functions and isinstance(functions[0], VectorFunction):
            vf = functions[0]
            for name, comp in zip(vf.field_names, vf.components):
                field_funcs[name] = comp
            for extra in functions[1:]:
                if isinstance(extra, Function):
                    field_funcs[extra.field_name] = extra
        elif isinstance(functions, Function):
            field_funcs[functions.field_name] = functions
        else:
            raise TypeError("'functions' must be a Function, VectorFunction, or dict[str, Function]")

        # which fields to include in the error
        if fields is None:
            fields = [f for f in me.field_names if f in exact]

        # accumulators
        err2, base2 = 0.0, 0.0

        # ---------------------------- classify elements ----------------------------
        inside_ids, outside_ids, cut_ids = mesh.classify_elements(level_set)  # fills element tags too
        full_eids = outside_ids if side == '+' else inside_ids

        # -------------------------- (A) full elements on side ----------------------
        qp_ref, qw_ref = vol_rule(mesh.element_type, qdeg)
        for eid in full_eids:
            # per‑field local dofs for this element (avoids mixing fields)
            gdofs_f = {fld: self.element_maps[fld][eid] for fld in fields}
            for (xi, eta), w in zip(qp_ref, qw_ref):
                J    = transform.jacobian(mesh, eid, (xi, eta))
                detJ = abs(np.linalg.det(J))
                x, y = transform.x_mapping(mesh, eid, (xi, eta))

                for fld in fields:
                    # MixedElement.basis returns a zero‑padded vector → slice to the field’s block
                    # slice to the field block → local basis of size n_loc(fld) (e.g., 9)
                    phi_f = me.basis(fld, xi, eta)[me.slice(fld)]
                    uh    = float(field_funcs[fld].get_nodal_values(gdofs_f[fld]) @ phi_f)
                    ue    = float(exact[fld](x, y))
                    d2    = (uh - ue) ** 2
                    b2    = ue ** 2
                    err2  += w * detJ * d2
                    base2 += w * detJ * b2

        # -------------------------- (B) cut elements on side -----------------------
        # reference rule on the unit triangle
        qp_tri, qw_tri = vol_rule("tri", qdeg)

        for eid in cut_ids:
            elem = mesh.elements_list[eid]
            tri_local, corner_ids = corner_tris(mesh, elem)

            # per‑field local dofs for this element (avoids mixing fields)
            gd_f = {fld: self.element_maps[fld][eid] for fld in fields}

            for loc_tri in tri_local:
                v_ids   = [corner_ids[i] for i in loc_tri]
                V       = mesh.nodes_x_y_pos[v_ids]                     # (3,2)
                v_phi   = np.array([phi_eval(level_set, xy) for xy in V], dtype=float)

                # clip this geometric triangle to requested side -------------------
                polys = clip_triangle_to_side(V, v_phi, side=side)      # list of polygons
                if not polys:
                    continue

                for poly in polys:
                    for A, B, C in fan_triangulate(poly):
                        x_phys, w_phys = map_ref_tri_to_phys(A, B, C, qp_tri, qw_tri)  # physical weights
                        for (x, y), w in zip(x_phys, w_phys):
                            # map back to (xi,eta) of the *parent* element -----------
                            xi, eta = transform.inverse_mapping(mesh, eid, np.array([x, y]))

                            for fld in fields:
                                # slice to the field block → local basis of size n_loc(fld) (e.g., 9)
                                phi_f = me.basis(fld, xi, eta)[me.slice(fld)]   # local (len n_loc(fld))
                                uh    = float(field_funcs[fld].get_nodal_values(gd_f[fld]) @ phi_f)
                                ue    = float(exact[fld](x, y))
                                d2    = (uh - ue) ** 2
                                b2    = ue ** 2
                                err2  += w * d2         # w already physical (no detJ)
                                base2 += w * b2

        return (err2 / base2) ** 0.5 if (relative and base2 > 1e-28) else err2 ** 0.5


    


    
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

