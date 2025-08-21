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
from pycutfem.fem.reference import get_reference
from pycutfem.integration.quadrature import volume
from collections.abc import Sequence
from hashlib import blake2b
from pycutfem.integration.quadrature import line_quadrature
from pycutfem.ufl.helpers_geom import (
    phi_eval, clip_triangle_to_side, fan_triangulate, map_ref_tri_to_phys, corner_tris
)
from pycutfem.utils.bitset import BitSet
from pycutfem.fem import transform
_JET = transform.InverseJetCache()  # module-scope 



BcLike = Union[BoundaryCondition, Mapping[str, Any]]
# ------------------------------------------------------------------
# edge-geometry cache keyed by (hash(ids), qdeg, id(level_set))
# ------------------------------------------------------------------
_edge_geom_cache: dict[tuple[int,int,int], dict] = {}
_ghost_cache: dict[tuple, dict] = {}

# geometric volume-cache keyed by (MixedElement.signature(), qdeg, id(level_set) or 0)
_volume_geom_cache: dict[tuple, dict] = {}
# NEW: cut-volume cache (per subset, qdeg, side, derivs, level-set, mixed element)
_cut_volume_cache: dict[tuple, dict] = {}

def clear_caches(self):
    """Clear all precompute caches on the handler."""
    _volume_geom_cache.clear()
    _cut_volume_cache.clear()
    _edge_geom_cache.clear()
    _ghost_cache.clear()


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




# ---- utilities ---------------------------------------------------------------
def _scatter_union(loc_arr: np.ndarray, sl: slice, final_width: int) -> np.ndarray:
    """
    Scatter a per-field local array into the union-local layout.

    loc_arr: (nE, nQ, n_f) or (nE, nQ, n_f, k)
    sl:     slice of this field in the union-local vector
    final_width: total union-local width across all fields
    """
    if loc_arr.ndim == 3:
        nE, nQ, _ = loc_arr.shape
        out = np.zeros((nE, nQ, final_width), dtype=loc_arr.dtype)
        out[:, :, sl] = loc_arr
        return out
    elif loc_arr.ndim == 4:
        nE, nQ, _, k = loc_arr.shape
        out = np.zeros((nE, nQ, final_width, k), dtype=loc_arr.dtype)
        out[:, :, sl, :] = loc_arr
        return out
    else:
        raise ValueError(f"_scatter_union expects rank-3/4 arrays, got {loc_arr.shape}")


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

            # Path 1: pre-tagged DOFs (field-scoped by construction)
            if domain_tag in self.dof_tags:
                val_is_callable = callable(bc.value)
                for dof in self.dof_tags[domain_tag]:
                    # Optional: still try to evaluate at the node if available
                    node_id = self._dof_to_node_map.get(dof, (field, None))[1]
                    if node_id is not None and val_is_callable:
                        x, y = mesh.nodes_x_y_pos[node_id]
                        data[dof] = bc.value(x, y)
                    else:
                        data[dof] = float(bc.value)  # constants (your case) are fine
                continue

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
        need_hess: bool = False,
        need_o3: bool = False,
        need_o4: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Precompute physical quadrature data (geometry, weights, optional level-set).
        Caches geometry per (mesh-id, n_elements, element_type, p, quad_order).

        Returns (per element):
        qp_phys (nE,nQ,2), qw (nE,nQ), detJ (nE,nQ), J_inv (nE,nQ,2,2),
        normals (nE,nQ,2), phis (nE,nQ or None), h_arr (nE,), eids (nE,),
        owner_id (nE,), entity_kind="element".
        If need_hess=True, also Hxi0/Hxi1 (nE,nQ,2,2) are returned.
        If need_o3=True   → also Txi0/Txi1 (nE,nQ,2,2,2).
        If need_o4=True   → also Qxi0/Qxi1 (nE,nQ,2,2,2,2).
        """
        import numpy as np
        from typing import Dict, Callable
        from pycutfem.fem import transform
        from pycutfem.integration.quadrature import volume as _volume_rule

        # Optional numba path
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
            _volume_geom_cache  # type: ignore
        except NameError:
            _volume_geom_cache = {}

        # ---------- fast path: use cached geometry; add phis/Hxi on-demand ----------
        if reuse and geom_key in _volume_geom_cache:
            geo = _volume_geom_cache[geom_key]
            qp = geo["qp_phys"]
            nE, nQ, _ = qp.shape

            # On-demand φ(xq) (not cached)
            phis = None
            if level_set is not None:
                phis = np.empty((nE, nQ), dtype=np.float64)
                for e in range(nE):
                    for q in range(nQ):
                        phis[e, q] = level_set(qp[e, q])

            # On-demand inverse-map jets if requested (not cached; use qp_ref to avoid inverse mapping)
            if need_hess or need_o3 or need_o4:
                qp_ref = geo.get("qp_ref", None)
                if qp_ref is None:
                    from pycutfem.integration.quadrature import volume as _volume_rule
                    qp_ref, _ = _volume_rule(mesh.element_type, quad_order)
                    qp_ref = np.asarray(qp_ref, dtype=np.float64)
                kmax = 4 if need_o4 else (3 if need_o3 else 2)
                # allocate only what’s requested
                out = {}
                if need_hess:
                    out["Hxi0"] = np.zeros((nE, nQ, 2, 2), dtype=np.float64)
                    out["Hxi1"] = np.zeros((nE, nQ, 2, 2), dtype=np.float64)
                if need_o3:
                    out["Txi0"] = np.zeros((nE, nQ, 2, 2, 2), dtype=np.float64)
                    out["Txi1"] = np.zeros((nE, nQ, 2, 2, 2), dtype=np.float64)
                if need_o4:
                    out["Qxi0"] = np.zeros((nE, nQ, 2, 2, 2, 2), dtype=np.float64)
                    out["Qxi1"] = np.zeros((nE, nQ, 2, 2, 2, 2), dtype=np.float64)
                # fill from inverse-jet cache
                eids = geo["eids"]
                for e in range(nE):
                    eid = int(eids[e])
                    for q in range(nQ):
                        xi, eta = float(qp_ref[q, 0]), float(qp_ref[q, 1])
                        rec = _JET.get(mesh, eid, xi, eta, upto=kmax)
                        if need_hess:
                            A2 = rec["A2"]  # (2,2,2)
                            out["Hxi0"][e, q] = A2[0]
                            out["Hxi1"][e, q] = A2[1]
                        if need_o3:
                            A3 = rec["A3"]  # (2,2,2,2)
                            out["Txi0"][e, q] = A3[0]
                            out["Txi1"][e, q] = A3[1]
                        if need_o4:
                            A4 = rec["A4"]  # (2,2,2,2,2)
                            out["Qxi0"][e, q] = A4[0]
                            out["Qxi1"][e, q] = A4[1]
                return {**geo, "phis": phis, **out}

        # ---------------------- reference quadrature -----------------------------
        qp_ref, qw_ref = _volume_rule(mesh.element_type, quad_order)  # (nQ,2), (nQ,)
        qp_ref = np.asarray(qp_ref, dtype=np.float64)
        qw_ref = np.asarray(qw_ref, dtype=np.float64)
        n_q    = int(qw_ref.shape[0])

        # ---------------------- reference shape/grad tables ----------------------
        kmax = 4 if need_o4 else (3 if need_o3 else 2)
        ref = get_reference(mesh.element_type, mesh.poly_order, max_deriv_order=kmax)
        # infer n_loc from a single evaluation (Ref has no n_functions accessor)
        n_loc = int(np.asarray(ref.shape(qp_ref[0, 0], qp_ref[0, 1])).size)
        Ntab  = np.empty((n_q, n_loc), dtype=np.float64)        # (nQ, n_loc)
        dNtab = np.empty((n_q, n_loc, 2), dtype=np.float64)     # (nQ, n_loc, 2)
        for q, (xi, eta) in enumerate(qp_ref):
            Ntab[q, :]     = np.asarray(ref.shape(xi, eta), dtype=np.float64).ravel()
            dNtab[q, :, :] = np.asarray(ref.grad (xi, eta), dtype=np.float64)

        # ---------------------- element → node coords (vectorized) ---------------
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
            # ------------------ safe Python fallback ------------------------------
            for e in range(n_el):
                for q_idx, (xi_eta, qw0) in enumerate(zip(qp_ref, qw_ref)):
                    xi_eta_t = (float(xi_eta[0]), float(xi_eta[1]))
                    J = transform.jacobian(mesh, e, xi_eta_t)
                    det = float(np.linalg.det(J))
                    if det <= 1e-12:
                        raise ValueError(f"Jacobian determinant is non-positive ({det}) for element {e}.")
                    detJ[e, q_idx]  = det
                    J_inv[e, q_idx] = np.linalg.inv(J)
                    # physical point and scaled weight
                    X = np.zeros(2, dtype=np.float64)
                    for i in range(n_loc):
                        Ni = Ntab[q_idx, i]
                        X[0] += Ni * elem_coord[e, i, 0]
                        X[1] += Ni * elem_coord[e, i, 1]
                    qp_phys[e, q_idx] = X
                    qw_sc[e, q_idx]   = qw0 * det

        # ---------------------- post-process & cache -----------------------------
        bad = np.where(detJ <= 1e-12)
        if bad[0].size:
            e_bad, q_bad = int(bad[0][0]), int(bad[1][0])
            raise ValueError(
                f"Jacobian determinant is non-positive ({detJ[e_bad, q_bad]}) "
                f"for element {e_bad} at qp {q_bad}."
            )

        normals = np.zeros((n_el, n_q, dim), dtype=np.float64)  # unused for volume dx
        h_arr   = np.empty((n_el,), dtype=np.float64)
        for e in range(n_el):
            h_arr[e] = mesh.element_char_length(e)
        eids = np.arange(n_el, dtype=np.int32)

        geo = {
            "qp_phys": qp_phys,
            "qw":      qw_sc,
            "detJ":    detJ,
            "J_inv":   J_inv,
            "normals": normals,
            "h_arr":   h_arr,
            "eids":    eids,
            "owner_id": eids.copy(),
            "entity_kind": "element",
            # store qp_ref in cache so jets never need inverse-mapping on reuse
            # "qp_ref":  qp_ref,
        }

        # Compute inverse-map jets if requested (not cached with geometry)
        if need_hess or need_o3 or need_o4:
            
            if need_hess:
                Hxi0 = np.zeros((n_el, n_q, 2, 2), dtype=np.float64)
                Hxi1 = np.zeros_like(Hxi0)
            if need_o3:
                Txi0 = np.zeros((n_el, n_q, 2, 2, 2), dtype=np.float64)
                Txi1 = np.zeros_like(Txi0)
            if need_o4:
                Qxi0 = np.zeros((n_el, n_q, 2, 2, 2, 2), dtype=np.float64)
                Qxi1 = np.zeros_like(Qxi0)
            for e in range(n_el):
                eid = int(eids[e])
                for q in range(n_q):
                    xi, eta = float(qp_ref[q, 0]), float(qp_ref[q, 1])
                    rec = _JET.get(mesh, eid, xi, eta, upto=kmax)
                    if need_hess:
                        A2 = rec["A2"]
                        Hxi0[e, q] = A2[0]; Hxi1[e, q] = A2[1]
                    if need_o3:
                        A3 = rec["A3"]
                        Txi0[e, q] = A3[0]; Txi1[e, q] = A3[1]
                    if need_o4:
                        A4 = rec["A4"]
                        Qxi0[e, q] = A4[0]; Qxi1[e, q] = A4[1]
            if need_hess:
                geo["Hxi0"] = Hxi0; geo["Hxi1"] = Hxi1
            if need_o3:
                geo["Txi0"] = Txi0; geo["Txi1"] = Txi1
            if need_o4:
                geo["Qxi0"] = Qxi0; geo["Qxi1"] = Qxi1

        # Cache *geometry only* (no phis)
        if reuse:
            _volume_geom_cache[geom_key] = {k: v for k, v in geo.items()
                                            if k not in ("Hxi0","Hxi1","Txi0","Txi1","Qxi0","Qxi1","phis")}

        # On-demand φ(xq) evaluation (cheap vs. geometry)
        phis = None
        if level_set is not None:
            phis = np.empty((n_el, n_q), dtype=np.float64)
            for e in range(n_el):
                for q in range(n_q):
                    phis[e, q] = level_set(qp_phys[e, q])

        # Return geometry + phis + any requested jets (if present in geo)
        ret = {**geo, "phis": phis}
        return ret



    
    
    # -------------------------------------------------------------------------
    #  DofHandler.precompute_interface_factors
    # -------------------------------------------------------------------------
     
    def precompute_interface_factors(
        self,
        cut_element_ids,
        qdeg: int,
        level_set,
        reuse: bool = True,
        need_hess: bool = False,
        need_o3: bool = False,
        need_o4: bool = False
    ) -> dict:
        """
        Pre-compute geometry & basis tables for ∫_{interface∩element} ⋯ dS on *cut* elements.
        Emits REFERENCE basis/derivative tables; push-forward (grad/Hess) is done in codegen.py.
        """
        import numpy as np
        from pycutfem.integration.quadrature import gauss_legendre, _map_line_rule_batched as _batched  # type: ignore
        from pycutfem.integration.quadrature import line_quadrature
        from pycutfem.core.levelset import CircleLevelSet, AffineLevelSet
        from pycutfem.core.levelset import _circle_value, _circle_grad, _affine_value, _affine_unit_grad  # type: ignore
        from pycutfem.fem import transform
        from pycutfem.integration.pre_tabulates import (
            _tabulate_p1, _tabulate_q2, _tabulate_q1,
            _tabulate_deriv_q1, _tabulate_deriv_q2, _tabulate_deriv_p1,
        )

        try:
            import numba as _nb  # noqa: F401
            _HAVE_NUMBA = True
        except Exception:
            _HAVE_NUMBA = False

        # ---- Geometry from dN and coords (Numba kernel) ----------------------------
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
        fields  = me.field_names
        n_union = me.n_dofs_local   # union-local width per element

        # --- normalize ids → valid cuts with exactly 2 interface points -------------
        ids = (cut_element_ids.to_indices()
            if hasattr(cut_element_ids, "to_indices")
            else list(cut_element_ids))
        valid_cut_eids = [
            int(eid) for eid in ids
            if (mesh.elements_list[eid].tag == "cut"
                and len(mesh.elements_list[eid].interface_pts) == 2)
        ]

        # ---- reuse cache if possible -----------------------------------------------
        global _interface_cache
        try:
            _interface_cache
        except NameError:
            _interface_cache = {}
        cache_key = (_hash_subset(valid_cut_eids), int(qdeg), me.signature(), self.method, bool(need_hess), bool(need_o3), bool(need_o4), id(level_set))
        if reuse and cache_key in _interface_cache:
            return _interface_cache[cache_key]

        if not valid_cut_eids:
            z2 = np.empty((0, 0, 2), dtype=float); z1 = np.empty((0, 0), dtype=float)
            out = {
                "eids": np.empty(0, dtype=np.int32),
                "qp_phys": z2, "qw": z1, "normals": z2, "phis": z1,
                "detJ": z1, "J_inv": np.empty((0, 0, 2, 2), dtype=float),
                "h_arr": np.empty((0,), dtype=float),
                "gdofs_map": np.empty((0, n_union), dtype=np.int64),
                "entity_kind": "element",
                "owner_id": np.empty(0, dtype=np.int32),
                "owner_pos_id": np.empty(0, dtype=np.int32),
                "owner_neg_id": np.empty(0, dtype=np.int32),
                "J_inv_pos": np.empty((0, 0, 2, 2), dtype=float),
                "J_inv_neg": np.empty((0, 0, 2, 2), dtype=float),
                "pos_map": np.empty((0, n_union), dtype=np.int32),
                "neg_map": np.empty((0, n_union), dtype=np.int32),
            }
            for fld in fields:
                out[f"b_{fld}"]   = np.empty((0, 0, n_union), dtype=float)
                out[f"g_{fld}"]   = np.empty((0, 0, n_union, 2), dtype=float)
                out[f"d10_{fld}"] = np.empty((0, 0, n_union), dtype=float)
                out[f"d01_{fld}"] = np.empty((0, 0, n_union), dtype=float)
                if need_hess:
                    out[f"d20_{fld}"] = np.empty((0, 0, n_union), dtype=float)
                    out[f"d11_{fld}"] = np.empty((0, 0, n_union), dtype=float)
                    out[f"d02_{fld}"] = np.empty((0, 0, n_union), dtype=float)
                if need_o3:
                    out[f"d30_{fld}"] = np.empty((0, 0, n_union), dtype=float)
                    out[f"d21_{fld}"] = np.empty((0, 0, n_union), dtype=float)
                    out[f"d12_{fld}"] = np.empty((0, 0, n_union), dtype=float)
                    out[f"d03_{fld}"] = np.empty((0, 0, n_union), dtype=float)
                if need_o4:
                    out[f"d40_{fld}"] = np.empty((0, 0, n_union), dtype=float)
                    out[f"d31_{fld}"] = np.empty((0, 0, n_union), dtype=float)
                    out[f"d22_{fld}"] = np.empty((0, 0, n_union), dtype=float)
                    out[f"d13_{fld}"] = np.empty((0, 0, n_union), dtype=float)
                    out[f"d04_{fld}"] = np.empty((0, 0, n_union), dtype=float)
            
            if need_hess:
                out["Hxi0"] = np.empty((0, 0, 2, 2), dtype=float)
                out["Hxi1"] = np.empty((0, 0, 2, 2), dtype=float)
            if need_o3:
                out["Txi0"] = np.empty((0, 0, 2, 2, 2), dtype=float)
                out["Txi1"] = np.empty((0, 0, 2, 2, 2), dtype=float)
            if need_o4:
                out["Qxi0"] = np.empty((0, 0, 2, 2, 2, 2), dtype=float)
                out["Qxi1"] = np.empty((0, 0, 2, 2, 2, 2), dtype=float)
            if reuse:
                _interface_cache[cache_key] = out
            return out

        # --- Prepare segments for batched mapping -----------------------------------
        nE = len(valid_cut_eids)
        P0 = np.empty((nE, 2), dtype=float)
        P1 = np.empty((nE, 2), dtype=float)
        h_arr = np.empty((nE,), dtype=float)
        gdofs_map = np.empty((nE, n_union), dtype=np.int64)
        for k, eid in enumerate(valid_cut_eids):
            p0, p1 = mesh.elements_list[eid].interface_pts
            P0[k] = p0; P1[k] = p1
            h_arr[k] = float(mesh.element_char_length(eid))
            gdofs_map[k, :] = self.get_elemental_dofs(int(eid))

        xi_1d, w_ref = gauss_legendre(qdeg)
        xi_1d = np.asarray(xi_1d, float); w_ref = np.asarray(w_ref, float)
        nQ = xi_1d.size

        qp_phys = np.empty((nE, nQ, 2), dtype=float)
        qw      = np.empty((nE, nQ),    dtype=float)
        if _HAVE_NUMBA:
            _batched(P0, P1, xi_1d, w_ref, qp_phys, qw)
        else:
            for k in range(nE):
                pts, wts = line_quadrature(P0[k], P1[k], qdeg)
                qp_phys[k, :, :] = pts; qw[k, :] = wts

        # --- φ and normals along the interface -------------------------------------
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
                    ng = float(np.linalg.norm(g))
                    normals[e, q] = g / (ng + 1e-30)

        # --- (ξ,η) at each interface quadrature point -------------------------------
        xi_tab  = np.empty((nE, nQ), dtype=float)
        eta_tab = np.empty((nE, nQ), dtype=float)
        for k, eid in enumerate(valid_cut_eids):
            for q in range(nQ):
                s, t = transform.inverse_mapping(mesh, int(eid), qp_phys[k, q])  # fast path if p==1
                xi_tab[k, q]  = float(s)
                eta_tab[k, q] = float(t)

        # --- Reference geometry (N,dN) and J, detJ, J_inv ---------------------------
        elem_type = mesh.element_type
        p_geom    = mesh.poly_order  # geometry/order on the mesh

        # parent element coordinates for geometry/J
        node_ids_all = mesh.nodes[mesh.elements_connectivity]
        coords_all   = mesh.nodes_x_y_pos[node_ids_all].astype(float)
        coords_sel   = coords_all[valid_cut_eids]  # (nE, nLocGeom, 2)
        nLocGeom     = coords_sel.shape[1]

        have_jit_ref = False
        kmax = 4 if need_o4 else (3 if need_o3 else 2)
        # Build dN_tab (nE, nQ, nLocGeom, 2) and, if available, N_tab (for basis fill)
        if _HAVE_NUMBA and elem_type == "tri" and p_geom == 1:
            N_tab  = np.empty((nE, nQ, 3), dtype=float)
            dN_tab = np.empty((nE, nQ, 3, 2), dtype=float)
            _tabulate_p1(xi_tab, eta_tab, N_tab, dN_tab)
            have_jit_ref = True
        elif _HAVE_NUMBA and elem_type == "quad" and p_geom == 1:
            N_tab  = np.empty((nE, nQ, 4), dtype=float)
            dN_tab = np.empty((nE, nQ, 4, 2), dtype=float)
            _tabulate_q1(xi_tab, eta_tab, N_tab, dN_tab)
            have_jit_ref = True
        elif _HAVE_NUMBA and elem_type == "quad" and p_geom == 2:
            N_tab  = np.empty((nE, nQ, 9), dtype=float)
            dN_tab = np.empty((nE, nQ, 9, 2), dtype=float)
            _tabulate_q2(xi_tab, eta_tab, N_tab, dN_tab)
            have_jit_ref = True
        else:
            ref = get_reference(elem_type, p_geom, max_deriv_order=kmax)
            dN_tab = np.empty((nE, nQ, nLocGeom, 2), dtype=float)
            for k in range(nE):
                for q in range(nQ):
                    dN_tab[k, q] = np.asarray(ref.grad(xi_tab[k, q], eta_tab[k, q]), dtype=float)

        # J, detJ, J_inv
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

        # Sanity: non-degenerate mapping
        bad = np.where(detJ <= 1e-12)
        if bad[0].size:
            e_bad, q_bad = int(bad[0][0]), int(bad[1][0])
            raise ValueError(
                f"Jacobian determinant is non-positive ({detJ[e_bad, q_bad]}) "
                f"for element id {valid_cut_eids[e_bad]} at qp {q_bad}."
            )

        # --- Basis & grad-basis on REFERENCE for each field -------------------------
        b_tabs = {f: np.zeros((nE, nQ, n_union), dtype=float) for f in fields}
        g_tabs = {f: np.zeros((nE, nQ, n_union, 2), dtype=float) for f in fields}

        for fld in fields:
            sl = me.component_dof_slices[fld]    # this field's slice in union-local
            ord_f = me._field_orders[fld]

            # Fast paths
            if _HAVE_NUMBA and elem_type == "tri" and ord_f == 1 and p_geom == 1 and 'N_tab' in locals() and N_tab.shape[2] == 3:
                b_tabs[fld][:, :, sl]    = N_tab
                g_tabs[fld][:, :, sl, :] = dN_tab
                continue

            if _HAVE_NUMBA and elem_type == "quad" and ord_f == 1:
                # If geometry is Q2, build local Q1 tables
                if p_geom == 2 and ('N_tab' not in locals() or N_tab.shape[2] != 4):
                    Nq1  = np.empty((nE, nQ, 4), dtype=float)
                    dNq1 = np.empty((nE, nQ, 4, 2), dtype=float)
                    _tabulate_q1(xi_tab, eta_tab, Nq1, dNq1)
                    b_tabs[fld][:, :, sl]    = Nq1
                    g_tabs[fld][:, :, sl, :] = dNq1
                else:
                    # geometry Q1 → already have Q1 in N_tab/dN_tab
                    b_tabs[fld][:, :, sl]    = N_tab
                    g_tabs[fld][:, :, sl, :] = dN_tab
                continue

        if _HAVE_NUMBA and elem_type == "quad" and any(me._field_orders[f]==2 for f in fields):
            pass  # handled per-field below

        # Fallback (and Q2 fast path per field)
        for fld in fields:
            sl = me.component_dof_slices[fld]
            ord_f = me._field_orders[fld]
            if _HAVE_NUMBA and elem_type == "quad" and ord_f == 2:
                Nq2  = np.empty((nE, nQ, 9), dtype=float)
                dNq2 = np.empty((nE, nQ, 9, 2), dtype=float)
                _tabulate_q2(xi_tab, eta_tab, Nq2, dNq2)
                b_tabs[fld][:, :, sl]    = Nq2
                g_tabs[fld][:, :, sl, :] = dNq2
                continue
            # Generic fallback using MixedElement (reference)
            for k, eid in enumerate(valid_cut_eids):
                for q in range(nQ):
                    s, t = xi_tab[k, q], eta_tab[k, q]
                    b_tabs[fld][k, q, sl]    = me._eval_scalar_basis(fld, s, t)
                    g_tabs[fld][k, q, sl, :] = me._eval_scalar_grad (fld, s, t)

        # --- Reference derivative tables d.._{field} for codegen (no push-forward) --
        # Up to the requested max: 1st (always), 2nd (need_hess), 3rd (need_o3), 4th (need_o4)
        base_derivs = {(1,0), (0,1)}
        if need_hess:
            base_derivs.update({(2,0), (1,1), (0,2)})
        if need_o3:
            base_derivs.update({(3,0), (2,1), (1,2), (0,3)})
        if need_o4:
            base_derivs.update({(4,0), (3,1), (2,2), (1,3), (0,4)})
        # If caller also asked for specific derivs, include them (superset)
        base_derivs |= set(getattr(self, "_last_requested_interface_derivs", set()))  # optional reuse
        # In case function parameter 'derivs' exists in your signature elsewhere:
        try:
            # if this function is called with a 'derivs' kw in your codebase later
            base_derivs |= set(derivs)  # type: ignore  # noqa
        except Exception:
            pass

        d_tabs = {}
        for fld in fields:
            sl = me.component_dof_slices[fld]
            ord_f = me._field_orders[fld]
            n_f = sl.stop - sl.start

            def _tab(dx, dy, out_arr):
                if _HAVE_NUMBA and elem_type == "quad" and ord_f == 1:
                    _tabulate_deriv_q1(xi_tab, eta_tab, int(dx), int(dy), out_arr)
                elif _HAVE_NUMBA and elem_type == "quad" and ord_f == 2:
                    _tabulate_deriv_q2(xi_tab, eta_tab, int(dx), int(dy), out_arr)
                elif _HAVE_NUMBA and elem_type == "tri" and ord_f == 1:
                    _tabulate_deriv_p1(xi_tab, eta_tab, int(dx), int(dy), out_arr)
                else:
                    for e in range(nE):
                        for q in range(nQ):
                            out_arr[e, q, :] = me._eval_scalar_deriv(
                                fld, float(xi_tab[e, q]), float(eta_tab[e, q]), int(dx), int(dy)
                            )

            for (dx, dy) in sorted(base_derivs):
                loc = np.empty((nE, nQ, n_f), dtype=float)
                _tab(dx, dy, loc)
                arr = np.zeros((nE, nQ, n_union), dtype=float)
                arr[:, :, sl] = loc
                d_tabs[f"d{dx}{dy}_{fld}"] = arr

        # --- Inverse-map jets for chain rule (Hxi: 2nd, Txi: 3rd, Qxi: 4th) -------
        if need_hess or need_o3 or need_o4:
            kmax = 4 if need_o4 else (3 if need_o3 else 2)
            if need_hess:
                Hxi0 = np.zeros((nE, nQ, 2, 2), dtype=float)
                Hxi1 = np.zeros_like(Hxi0)
            if need_o3:
                Txi0 = np.zeros((nE, nQ, 2, 2, 2), dtype=float)
                Txi1 = np.zeros_like(Txi0)
            if need_o4:
                Qxi0 = np.zeros((nE, nQ, 2, 2, 2, 2), dtype=float)
                Qxi1 = np.zeros_like(Qxi0)

            # Use module-level inverse-jet cache; (xi_tab, eta_tab) already computed
            for i, eid in enumerate(valid_cut_eids):
                eidi = int(eid)
                for q in range(nQ):
                    xi = float(xi_tab[i, q]); eta = float(eta_tab[i, q])
                    rec = _JET.get(mesh, eidi, xi, eta, upto=kmax)
                    if need_hess:
                        A2 = rec["A2"]            # (2,2,2)
                        Hxi0[i, q] = A2[0]
                        Hxi1[i, q] = A2[1]
                    if need_o3:
                        A3 = rec["A3"]            # (2,2,2,2)
                        Txi0[i, q] = A3[0]
                        Txi1[i, q] = A3[1]
                    if need_o4:
                        A4 = rec["A4"]            # (2,2,2,2,2)
                        Qxi0[i, q] = A4[0]
                        Qxi1[i, q] = A4[1]


        # --- Pack outputs -----------------------------------------------------------
        out = {
            "eids":        np.asarray(valid_cut_eids, dtype=np.int32),
            "qp_phys":     qp_phys,
            "qw":          qw,
            "normals":     normals,
            "phis":        phis,
            "detJ":        detJ,
            "J_inv":       J_inv,
            "h_arr":       h_arr,
            "gdofs_map":   gdofs_map,
            "entity_kind": "element",

            # Side-ready aliases for codegen (same element on both "sides")
            "J_inv_pos":   J_inv,
            "J_inv_neg":   J_inv,
            "pos_map":     np.tile(np.arange(n_union, dtype=np.int32), (nE, 1)),
            "neg_map":     np.tile(np.arange(n_union, dtype=np.int32), (nE, 1)),

            # Owner info (both sides are the same element)
            "owner_id":     np.asarray(valid_cut_eids, dtype=np.int32),
            "owner_pos_id": np.asarray(valid_cut_eids, dtype=np.int32),
            "owner_neg_id": np.asarray(valid_cut_eids, dtype=np.int32),
        }
        for fld in fields:
            out[f"b_{fld}"] = b_tabs[fld]
            out[f"g_{fld}"] = g_tabs[fld]
        out.update(d_tabs)
        if need_hess:
            out["Hxi0"] = Hxi0; out["Hxi1"] = Hxi1
        if need_o3:
            out["Txi0"] = Txi0; out["Txi1"] = Txi1
        if need_o4:
            out["Qxi0"] = Qxi0; out["Qxi1"] = Qxi1

        if reuse:
            _interface_cache[cache_key] = out
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
        reuse: bool = True,
        need_hess: bool = False,
        need_o3: bool = False,
        need_o4: bool = False
    ) -> dict:
        import numpy as np
        from pycutfem.integration.quadrature import gauss_legendre, _map_line_rule_batched as _batched  # type: ignore
        from pycutfem.integration.quadrature import line_quadrature
        from pycutfem.fem import transform
        from pycutfem.integration.pre_tabulates import (
            _searchsorted_positions,
            _tabulate_deriv_q1, _tabulate_deriv_q2, _tabulate_deriv_p1,
            _tabulate_q1 as _tab_q1, _tabulate_q2 as _tab_q2, _tabulate_p1 as _tab_p1,
        )

        try:
            import numba as _nb
            _HAVE_NUMBA = True
        except Exception:
            _HAVE_NUMBA = False

        # ---- Fast geometry from dN and coords (Numba kernel) ----------------------
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
                            x  = coords[e, i, 0]
                            y  = coords[e, i, 1]
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
        fields  = me.field_names
        n_union = self.union_dofs
        n_loc   = me.n_dofs_per_elem

        # 0) normalize ghost edge ids and set up cache key --------------------------
        if hasattr(ghost_edge_ids, "to_indices"):
            ghost_ids = tuple(int(i) for i in ghost_edge_ids.to_indices())
        else:
            ghost_ids = tuple(int(i) for i in ghost_edge_ids)

        # Cache key: use the subset tuple directly (stable & hashable)
        derivs_key = tuple(sorted((int(dx), int(dy)) for (dx, dy) in derivs))
        cache_key  = (ghost_ids, int(qdeg), me.signature(), derivs_key, bool(need_hess), bool(need_o3), bool(need_o4), self.method)
        global _ghost_cache
        try:
            _ghost_cache
        except NameError:
            _ghost_cache = {}
        if reuse and cache_key in _ghost_cache:
            return _ghost_cache[cache_key]

        # Keep only true interior ghosts: either neighbor cut or already tagged
        edges = []
        for gid in ghost_ids:
            e = mesh.edge(gid)
            if e.right is None:
                continue
            lt = mesh.elements_list[e.left].tag
            rt = mesh.elements_list[e.right].tag
            et = str(getattr(e, "tag", ""))
            if (("cut" in (lt, rt)) or et.startswith("ghost")):
                edges.append(e)
        if not edges:
            raise ValueError("No valid ghost edges found.")

        # 1) Batched line quadrature on segments ------------------------------------
        nE = len(edges)
        P0 = np.empty((nE, 2), dtype=float)
        P1 = np.empty((nE, 2), dtype=float)
        for i, e in enumerate(edges):
            p0, p1 = mesh.nodes_x_y_pos[list(e.nodes)]
            P0[i] = p0; P1[i] = p1

        xi1, w_ref = gauss_legendre(qdeg)
        xi1 = np.asarray(xi1, float); w_ref = np.asarray(w_ref, float)
        nQ  = xi1.size

        qp_phys = np.empty((nE, nQ, 2), dtype=float)
        qw      = np.empty((nE, nQ),    dtype=float)
        if _HAVE_NUMBA:
            _batched(P0, P1, xi1, w_ref, qp_phys, qw)
        else:
            for i in range(nE):
                pts, wts = line_quadrature(P0[i], P1[i], qdeg)
                qp_phys[i, :, :] = pts; qw[i, :] = wts

        # 2) Oriented normals & signed distance φ -----------------------------------
        normals = np.empty((nE, nQ, 2), dtype=float)
        phi_arr = np.zeros((nE, nQ), dtype=float)
        pos_ids = np.empty(nE, dtype=np.int32)
        neg_ids = np.empty(nE, dtype=np.int32)
        for i, e in enumerate(edges):
            phiL = float(level_set(np.asarray(mesh.elements_list[e.left ].centroid())))
            phiR = float(level_set(np.asarray(mesh.elements_list[e.right].centroid())))
            if phiL >= phiR:
                pos_eid, neg_eid = e.left, e.right
            else:
                pos_eid, neg_eid = e.right, e.left
            pos_ids[i] = int(pos_eid)
            neg_ids[i] = int(neg_eid)

            # ensure normal points from neg → pos
            nvec = e.normal
            cpos = np.asarray(mesh.elements_list[pos_eid].centroid())
            cneg = np.asarray(mesh.elements_list[neg_eid].centroid())
            if np.dot(nvec, cpos - cneg) < 0.0:
                nvec = -nvec
            for q in range(nQ):
                normals[i, q] = nvec
                phi_arr[i, q] = level_set(qp_phys[i, q])

        # 3) Union GDofs and side maps ----------------------------------------------
        gdofs_map = -np.ones((nE, n_union), dtype=np.int64)
        pos_map   = -np.ones((nE, n_loc),   dtype=np.int32)
        neg_map   = -np.ones((nE, n_loc),   dtype=np.int32)
        for i, e in enumerate(edges):
            pos_eid = int(pos_ids[i])
            neg_eid = int(neg_ids[i])
            pos_dofs = self.get_elemental_dofs(pos_eid)
            neg_dofs = self.get_elemental_dofs(neg_eid)
            global_dofs = np.unique(np.concatenate((pos_dofs, neg_dofs)))
            if global_dofs.size != n_union:
                raise ValueError(f"union size mismatch on edge {e.gid}: {global_dofs.size} vs {n_union}")
            gdofs_map[i, :n_union] = global_dofs
            pos_map[i] = _searchsorted_positions(global_dofs, pos_dofs)
            neg_map[i] = _searchsorted_positions(global_dofs, neg_dofs)

        # 4) Reference coords on both sides; geometry dN; build J, detJ, J_inv -------
        xi_pos = np.empty((nE, nQ)); eta_pos = np.empty((nE, nQ))
        xi_neg = np.empty((nE, nQ)); eta_neg = np.empty((nE, nQ))
        for i, e in enumerate(edges):
            pos_eid = int(pos_ids[i]); neg_eid = int(neg_ids[i])
            for q in range(nQ):
                s, t = transform.inverse_mapping(mesh, pos_eid, qp_phys[i, q])
                xi_pos[i, q]  = float(s); eta_pos[i, q] = float(t)
                s, t = transform.inverse_mapping(mesh, neg_eid, qp_phys[i, q])
                xi_neg[i, q]  = float(s); eta_neg[i, q] = float(t)

        # parent element coords
        node_ids_all = mesh.nodes[mesh.elements_connectivity]
        coords_all   = mesh.nodes_x_y_pos[node_ids_all].astype(float)
        coords_pos   = coords_all[pos_ids]
        coords_neg   = coords_all[neg_ids]
        nLocGeom     = coords_pos.shape[1]

        # reference geometry dN on each side
        dN_pos = np.empty((nE, nQ, nLocGeom, 2), dtype=float)
        dN_neg = np.empty((nE, nQ, nLocGeom, 2), dtype=float)
        kmax = 4 if need_o4 else (3 if need_o3 else 2)
        if mesh.element_type == "quad" and mesh.poly_order == 1:
            _tab_q1(xi_pos, xi_neg*0 + eta_pos, np.empty((nE, nQ, 4)), dN_pos)
            _tab_q1(xi_neg, xi_neg*0 + eta_neg, np.empty((nE, nQ, 4)), dN_neg)
        elif mesh.element_type == "quad" and mesh.poly_order == 2:
            _tab_q2(xi_pos, xi_neg*0 + eta_pos, np.empty((nE, nQ, 9)), dN_pos)
            _tab_q2(xi_neg, xi_neg*0 + eta_neg, np.empty((nE, nQ, 9)), dN_neg)
        elif mesh.element_type == "tri" and mesh.poly_order == 1:
            _tab_p1(xi_pos, xi_neg*0 + eta_pos, np.empty((nE, nQ, 3)), dN_pos)
            _tab_p1(xi_neg, xi_neg*0 + eta_neg, np.empty((nE, nQ, 3)), dN_neg)
        else:
            ref = get_reference(mesh.element_type, mesh.poly_order, max_deriv_order = kmax)
            for i in range(nE):
                for q in range(nQ):
                    dN_pos[i, q] = np.asarray(ref.grad(xi_pos[i, q], eta_pos[i, q]))
                    dN_neg[i, q] = np.asarray(ref.grad(xi_neg[i, q], eta_neg[i, q]))

        # J, detJ, J_inv
        if _HAVE_NUMBA:
            detJ_pos, J_inv_pos = _geom_from_dN(coords_pos, dN_pos)
            detJ_neg, J_inv_neg = _geom_from_dN(coords_neg, dN_neg)
        else:
            detJ_pos = np.empty((nE, nQ)); J_inv_pos = np.empty((nE, nQ, 2, 2))
            detJ_neg = np.empty((nE, nQ)); J_inv_neg = np.empty((nE, nQ, 2, 2))
            for coords, dN, detJ, J_inv in ((coords_pos, dN_pos, detJ_pos, J_inv_pos),
                                            (coords_neg, dN_neg, detJ_neg, J_inv_neg)):
                for e in range(nE):
                    Xe = coords[e]
                    for q in range(nQ):
                        a00 = a01 = a10 = a11 = 0.0
                        for iN in range(nLocGeom):
                            gx, gy = dN[e, q, iN, 0], dN[e, q, iN, 1]
                            x,  y  = Xe[iN, 0], Xe[iN, 1]
                            a00 += gx * x; a01 += gx * y
                            a10 += gy * x; a11 += gy * y
                        det = a00 * a11 - a01 * a10
                        detJ[e, q] = det
                        invd = 1.0 / det
                        J_inv[e, q, 0, 0] =  a11 * invd
                        J_inv[e, q, 0, 1] = -a01 * invd
                        J_inv[e, q, 1, 0] = -a10 * invd
                        J_inv[e, q, 1, 1] =  a00 * invd

        # sanity check
        bad = np.where((detJ_pos <= 1e-12) | (detJ_neg <= 1e-12))
        if bad[0].size:
            e_bad, q_bad = int(bad[0][0]), int(bad[1][0])
            raise ValueError(f"Non-positive detJ at edge {edges[e_bad].gid}, qp {q_bad}")

        # 5) Reference derivative tables per side/field ------------------------------
        derivs = set(derivs)  # no need to inject (0,0) here for Hessian/Lap kernels
        basis_tables = {}
        nE, nQ = xi_pos.shape
        final_w = me.n_dofs_per_elem

        def _tab_generic(fld: str, dx: int, dy: int, xi, eta, out_arr):
            for e in range(nE):
                for q in range(nQ):
                    out_arr[e, q, :] = me._eval_scalar_deriv(
                        fld, float(xi[e, q]), float(eta[e, q]), int(dx), int(dy)
                    )

        for fld in fields:
            p_f = me._field_orders[fld]
            sl  = me.component_dof_slices[fld]
            n_f = sl.stop - sl.start

            def _tab(dx, dy, xi, eta, out_arr):
                if _HAVE_NUMBA and mesh.element_type == "quad" and p_f == 1:
                    _tabulate_deriv_q1(xi, eta, dx, dy, out_arr)
                elif _HAVE_NUMBA and mesh.element_type == "quad" and p_f == 2:
                    _tabulate_deriv_q2(xi, eta, dx, dy, out_arr)
                elif _HAVE_NUMBA and mesh.element_type == "tri"  and p_f == 1:
                    _tabulate_deriv_p1(xi, eta, dx, dy, out_arr)
                else:
                    _tab_generic(fld, dx, dy, xi, eta, out_arr)

            need_grad = any(dx + dy == 1 for dx, dy in derivs) or need_hess
            need_h2   = any(dx + dy == 2 for dx, dy in derivs) or need_hess
            need_o3rq   = any(dx + dy == 3 for dx, dy in derivs) or need_o3
            need_o4rq   = any(dx + dy == 4 for dx, dy in derivs) or need_o4

            # POS side reference tables
            if need_grad:
                arr = np.empty((nE, nQ, n_f)); _tab(1, 0, xi_pos, eta_pos, arr)
                basis_tables[f"r10_{fld}_pos"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(0, 1, xi_pos, eta_pos, arr)
                basis_tables[f"r01_{fld}_pos"] = _scatter_union(arr, sl, final_w)
            if need_h2:
                arr = np.empty((nE, nQ, n_f)); _tab(2, 0, xi_pos, eta_pos, arr)
                basis_tables[f"r20_{fld}_pos"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(1, 1, xi_pos, eta_pos, arr)
                basis_tables[f"r11_{fld}_pos"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(0, 2, xi_pos, eta_pos, arr)
                basis_tables[f"r02_{fld}_pos"] = _scatter_union(arr, sl, final_w)
            if need_o3rq:
                arr = np.empty((nE, nQ, n_f)); _tab(3, 0, xi_pos, eta_pos, arr)
                basis_tables[f"r30_{fld}_pos"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(2, 1, xi_pos, eta_pos, arr)
                basis_tables[f"r21_{fld}_pos"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(1, 2, xi_pos, eta_pos, arr)
                basis_tables[f"r12_{fld}_pos"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(0, 3, xi_pos, eta_pos, arr)
                basis_tables[f"r03_{fld}_pos"] = _scatter_union(arr, sl, final_w)
            if need_o4rq:
                arr = np.empty((nE, nQ, n_f)); _tab(4, 0, xi_pos, eta_pos, arr)
                basis_tables[f"r40_{fld}_pos"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(3, 1, xi_pos, eta_pos, arr)
                basis_tables[f"r31_{fld}_pos"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(2, 2, xi_pos, eta_pos, arr)
                basis_tables[f"r22_{fld}_pos"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(1, 3, xi_pos, eta_pos, arr)
                basis_tables[f"r13_{fld}_pos"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(0, 4, xi_pos, eta_pos, arr)
                basis_tables[f"r04_{fld}_pos"] = _scatter_union(arr, sl, final_w)

            # NEG side reference tables
            if need_grad:
                arr = np.empty((nE, nQ, n_f)); _tab(1, 0, xi_neg, eta_neg, arr)
                basis_tables[f"r10_{fld}_neg"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(0, 1, xi_neg, eta_neg, arr)
                basis_tables[f"r01_{fld}_neg"] = _scatter_union(arr, sl, final_w)
            if need_h2:
                arr = np.empty((nE, nQ, n_f)); _tab(2, 0, xi_neg, eta_neg, arr)
                basis_tables[f"r20_{fld}_neg"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(1, 1, xi_neg, eta_neg, arr)
                basis_tables[f"r11_{fld}_neg"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(0, 2, xi_neg, eta_neg, arr)
                basis_tables[f"r02_{fld}_neg"] = _scatter_union(arr, sl, final_w)
            if need_o3rq:
                arr = np.empty((nE, nQ, n_f)); _tab(3, 0, xi_neg, eta_neg, arr)
                basis_tables[f"r30_{fld}_neg"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(2, 1, xi_neg, eta_neg, arr)
                basis_tables[f"r21_{fld}_neg"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(1, 2, xi_neg, eta_neg, arr)
                basis_tables[f"r12_{fld}_neg"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(0, 3, xi_neg, eta_neg, arr)
                basis_tables[f"r03_{fld}_neg"] = _scatter_union(arr, sl, final_w)
            if need_o4rq:
                arr = np.empty((nE, nQ, n_f)); _tab(4, 0, xi_neg, eta_neg, arr)
                basis_tables[f"r40_{fld}_neg"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(3, 1, xi_neg, eta_neg, arr)
                basis_tables[f"r31_{fld}_neg"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(2, 2, xi_neg, eta_neg, arr)
                basis_tables[f"r22_{fld}_neg"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(1, 3, xi_neg, eta_neg, arr)
                basis_tables[f"r13_{fld}_neg"] = _scatter_union(arr, sl, final_w)
                arr = np.empty((nE, nQ, n_f)); _tab(0, 4, xi_neg, eta_neg, arr)
                basis_tables[f"r04_{fld}_neg"] = _scatter_union(arr, sl, final_w)

        # 6) Inverse-map jets for chain rule (sided; up to order 4) -----------------
        pos_Hxi0 = pos_Hxi1 = neg_Hxi0 = neg_Hxi1 = None
        if need_hess or need_o3 or need_o4:
            if need_hess:
                pos_Hxi0 = np.zeros((nE, nQ, 2, 2), dtype=float)
                pos_Hxi1 = np.zeros_like(pos_Hxi0)
                neg_Hxi0 = np.zeros((nE, nQ, 2, 2), dtype=float)
                neg_Hxi1 = np.zeros_like(neg_Hxi0)
            if need_o3:
                pos_Txi0 = np.zeros((nE, nQ, 2, 2, 2), dtype=float)
                pos_Txi1 = np.zeros_like(pos_Txi0)
                neg_Txi0 = np.zeros((nE, nQ, 2, 2, 2), dtype=float)
                neg_Txi1 = np.zeros_like(neg_Txi0)
            if need_o4:
                pos_Qxi0 = np.zeros((nE, nQ, 2, 2, 2, 2), dtype=float)
                pos_Qxi1 = np.zeros_like(pos_Qxi0)
                neg_Qxi0 = np.zeros((nE, nQ, 2, 2, 2, 2), dtype=float)
                neg_Qxi1 = np.zeros_like(neg_Qxi0)
            for i in range(nE):
                pe = int(pos_ids[i]); ne = int(neg_ids[i])
                for q in range(nQ):
                    # POS side
                    rec = _JET.get(mesh, pe, float(xi_pos[i, q]), float(eta_pos[i, q]),
                                   upto = 4 if need_o4 else (3 if need_o3 else 2))
                    if need_hess:
                        A2 = rec["A2"]; pos_Hxi0[i, q] = A2[0]; pos_Hxi1[i, q] = A2[1]
                    if need_o3:
                        A3 = rec["A3"]; pos_Txi0[i, q] = A3[0]; pos_Txi1[i, q] = A3[1]
                    if need_o4:
                        A4 = rec["A4"]; pos_Qxi0[i, q] = A4[0]; pos_Qxi1[i, q] = A4[1]
                    # NEG side
                    rec = _JET.get(mesh, ne, float(xi_neg[i, q]), float(eta_neg[i, q]),
                                   upto = 4 if need_o4 else (3 if need_o3 else 2))
                    if need_hess:
                        A2 = rec["A2"]; neg_Hxi0[i, q] = A2[0]; neg_Hxi1[i, q] = A2[1]
                    if need_o3:
                        A3 = rec["A3"]; neg_Txi0[i, q] = A3[0]; neg_Txi1[i, q] = A3[1]
                    if need_o4:
                        A4 = rec["A4"]; neg_Qxi0[i, q] = A4[0]; neg_Qxi1[i, q] = A4[1]

        # 7) Edge size h (robust for interior faces) --------------------------------
        h_arr = np.empty((nE,), dtype=float)
        for i, e in enumerate(edges):
            hL = mesh.element_char_length(e.left)
            hR = mesh.element_char_length(e.right)
            if hL is None and hR is not None:
                h_arr[i] = hR
            elif hR is None and hL is not None:
                h_arr[i] = hL
            elif hL is not None and hR is not None:
                h_arr[i] = max(hL, hR)
            else:
                raise ValueError(f"Edge {e.gid} has no valid size (both sides None)")

        # 8) Pack results ------------------------------------------------------------
        out = {
            "eids":        np.asarray([e.gid for e in edges], dtype=np.int32),
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
            "owner_id":     pos_ids,  # convenience alias
        }
        out.update(basis_tables)
        if need_hess:
            out.update({
                "pos_Hxi0": pos_Hxi0, "pos_Hxi1": pos_Hxi1,
                "neg_Hxi0": neg_Hxi0, "neg_Hxi1": neg_Hxi1,
            })
        if need_o3:
            out.update({
                "pos_Txi0": pos_Txi0, "pos_Txi1": pos_Txi1,
                "neg_Txi0": neg_Txi0, "neg_Txi1": neg_Txi1,
            })
        if need_o4:
            out.update({
                "pos_Qxi0": pos_Qxi0, "pos_Qxi1": pos_Qxi1,
                "neg_Qxi0": neg_Qxi0, "neg_Qxi1": neg_Qxi1,
            })

        if reuse:
            _ghost_cache[cache_key] = out
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
        need_hess: bool = False,
        need_o3: bool = False,
        need_o4: bool = False,
    ) -> dict:
        """
        Pre-compute geometry & basis tables for ∫_Γ ⋯ dS on *boundary* edges.
        Returns per-edge arrays sized to the given subset and ready for JIT.
        Emits REFERENCE derivative tables d.._{field}; push-forward is done in codegen.py.
        """
        import numpy as np
        from pycutfem.integration.quadrature import gauss_legendre, _map_line_rule_batched as _batched  # type: ignore
        from pycutfem.integration.quadrature import line_quadrature
        from pycutfem.fem import transform
        from pycutfem.integration.pre_tabulates import (
            _tabulate_q1 as _tab_q1,
            _tabulate_q2 as _tab_q2,
            _tabulate_p1 as _tab_p1,
            _tabulate_deriv_q1,
            _tabulate_deriv_q2,
            _tabulate_deriv_p1,
        )
        

        try:
            import numba as _nb
            _HAVE_NUMBA = True
        except Exception:
            _HAVE_NUMBA = False

        # --- J from dN and coords (Numba kernel) -----------------------------------
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
        n_loc = me.n_dofs_per_elem  # union-local width

        # ---- normalize & filter to boundary edges (right is None) ------------------
        if hasattr(edge_ids, "to_indices"):                  # BitSet
            edge_ids = list(int(i) for i in edge_ids.to_indices())
        else:
            edge_ids = list(int(i) for i in edge_ids)        # already a sequence

        edge_ids = np.asarray(edge_ids, dtype=np.int32)
        n_edges  = int(edge_ids.shape[0])
        if n_edges == 0:
            return {"eids": np.empty(0, dtype=np.int32)}  # nothing to do

        # ---- reuse cache if possible -----------------------------------------------
        global _edge_geom_cache
        try:
            _edge_geom_cache
        except NameError:
            _edge_geom_cache = {}
        derivs_key = tuple(sorted((int(dx), int(dy)) for (dx, dy) in derivs))
        cache_key  = (tuple(edge_ids.tolist()), int(qdeg),
                    me.signature(), derivs_key, self.method, bool(need_hess), bool(need_o3), bool(need_o4))
        if reuse and cache_key in _edge_geom_cache:
            return _edge_geom_cache[cache_key]

        # ---- sizes -----------------------------------------------------------------
        # n_edges = len(edge_ids)
        # representative to size arrays
        p0r, p1r = mesh.nodes_x_y_pos[list(mesh.edge(edge_ids[0]).nodes)]
        qpr, qwr = line_quadrature(p0r, p1r, qdeg)
        n_q = len(qwr)

        # ---- work arrays -----------------------------------------------------------
        qp_phys  = np.zeros((n_edges, n_q, 2), dtype=float)
        qw       = np.zeros((n_edges, n_q),    dtype=float)
        normals  = np.zeros((n_edges, n_q, 2), dtype=float)
        detJ     = np.zeros((n_edges, n_q),    dtype=float)
        J_inv    = np.zeros((n_edges, n_q, 2, 2), dtype=float)
        phis     = None  # boundary integral has no level-set
        gdofs_map = np.zeros((n_edges, n_loc), dtype=np.int64)
        h_arr     = np.zeros((n_edges,), dtype=float)

        # derivative tables (union-sized, *reference*!)
        basis_tabs = {}

        # ---- batched edge mapping --------------------------------------------------
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

        # ---- normals & dof maps (owner is 'left') ---------------------------------
        owner = np.empty((n_edges,), dtype=np.int32)
        for i, gid in enumerate(edge_ids):
            e = mesh.edge(gid)
            eid = e.left
            owner[i] = int(eid)
            normals[i, :, :] = e.normal  # assumed outward for owner; constant along edge
            gdofs_map[i, :]  = self.get_elemental_dofs(eid)
            # robust element-length proxy for face penalties
            h = mesh.element_char_length(eid)
            if h is None:
                # fallback: geometric edge length
                p0, p1 = mesh.nodes_x_y_pos[list(e.nodes)]
                h = float(np.linalg.norm(p1 - p0))
            h_arr[i] = float(h)

        # ---- (ξ,η) tables on the owner element ------------------------------------
        xi_tab  = np.empty((n_edges, n_q), dtype=float)
        eta_tab = np.empty((n_edges, n_q), dtype=float)
        eids_arr = owner.copy()
        for i, eid in enumerate(eids_arr):
            for q in range(n_q):
                s, t = transform.inverse_mapping(mesh, int(eid), qp_phys[i, q])
                xi_tab[i, q]  = float(s)
                eta_tab[i, q] = float(t)

        # ---- reference dN for geometry order; build J, detJ, J⁻¹ ------------------
        node_ids_all = mesh.nodes[mesh.elements_connectivity]
        coords_all   = mesh.nodes_x_y_pos[node_ids_all].astype(float)
        coords_sel   = coords_all[eids_arr]
        nLocGeom     = coords_sel.shape[1]
        dN_tab = np.empty((n_edges, n_q, nLocGeom, 2), dtype=float)
        

        if mesh.element_type == "quad" and mesh.poly_order == 1:
            _tab_q1(xi_tab, eta_tab, np.empty((n_edges, n_q, 4)), dN_tab)
        elif mesh.element_type == "quad" and mesh.poly_order == 2:
            _tab_q2(xi_tab, eta_tab, np.empty((n_edges, n_q, 9)), dN_tab)
        elif mesh.element_type == "tri" and mesh.poly_order == 1:
            _tab_p1(xi_tab, eta_tab, np.empty((n_edges, n_q, 3)), dN_tab)
        else:
            ref = get_reference(mesh.element_type, mesh.poly_order, max_deriv_order = 4 if need_o4 else (3 if need_o3 else 2))
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

        # ---- Inverse-map jets for chain rule (Hxi: 2nd, Txi: 3rd, Qxi: 4th) -------
        if need_hess or need_o3 or need_o4:
            kmax = 4 if need_o4 else (3 if need_o3 else 2)
            if need_hess:
                Hxi0 = np.zeros((n_edges, n_q, 2, 2), dtype=float)
                Hxi1 = np.zeros_like(Hxi0)
            if need_o3:
                Txi0 = np.zeros((n_edges, n_q, 2, 2, 2), dtype=float)
                Txi1 = np.zeros_like(Txi0)
            if need_o4:
                Qxi0 = np.zeros((n_edges, n_q, 2, 2, 2, 2), dtype=float)
                Qxi1 = np.zeros_like(Qxi0)
            for i, eid in enumerate(eids_arr):
                eidi = int(eid)
                for q in range(n_q):
                    rec = _JET.get(mesh, eidi, float(xi_tab[i, q]), float(eta_tab[i, q]), upto=kmax)
                    if need_hess:
                        A2 = rec["A2"]; Hxi0[i, q] = A2[0]; Hxi1[i, q] = A2[1]
                    if need_o3:
                        A3 = rec["A3"]; Txi0[i, q] = A3[0]; Txi1[i, q] = A3[1]
                    if need_o4:
                        A4 = rec["A4"]; Qxi0[i, q] = A4[0]; Qxi1[i, q] = A4[1]

        # ---- tabulate REFERENCE derivatives per field (no push-forward here) -------
        # Ensure closure: chain rule for orders 2/3/4 needs all lower orders too.
        derivs = set(derivs)
        max_req = 0
        if derivs:
            max_req = max(dx + dy for dx, dy in derivs)
        if need_o4: max_req = max(max_req, 4)
        elif need_o3: max_req = max(max_req, 3)
        elif need_hess: max_req = max(max_req, 2)
        base = set()
        if max_req >= 1: base |= {(1,0), (0,1)}
        if max_req >= 2: base |= {(2,0), (1,1), (0,2)}
        if max_req >= 3: base |= {(3,0), (2,1), (1,2), (0,3)}
        if max_req >= 4: base |= {(4,0), (3,1), (2,2), (1,3), (0,4)}
        derivs |= base
        for fld in fields:
            sl  = self.mixed_element.component_dof_slices[fld]
            p_f = self.mixed_element._field_orders[fld]
            n_f = sl.stop - sl.start

            def _tab(dx, dy, out_arr):
                if _HAVE_NUMBA and mesh.element_type == "quad" and p_f == 1:
                    _tabulate_deriv_q1(xi_tab, eta_tab, int(dx), int(dy), out_arr)
                elif _HAVE_NUMBA and mesh.element_type == "quad" and p_f == 2:
                    _tabulate_deriv_q2(xi_tab, eta_tab, int(dx), int(dy), out_arr)
                elif _HAVE_NUMBA and mesh.element_type == "tri" and p_f == 1:
                    _tabulate_deriv_p1(xi_tab, eta_tab, int(dx), int(dy), out_arr)
                else:
                    for e in range(n_edges):
                        for q in range(n_q):
                            out_arr[e, q, :] = self.mixed_element._eval_scalar_deriv(
                                fld, float(xi_tab[e, q]), float(eta_tab[e, q]), int(dx), int(dy)
                            )

            for (dx, dy) in derivs:
                loc = np.empty((n_edges, n_q, n_f), dtype=float)
                _tab(dx, dy, loc)
                basis_tabs[f"d{dx}{dy}_{fld}"] = _scatter_union(loc, sl, n_loc)

        # ---- pack & cache ----------------------------------------------------------
        out = {
            "eids":        np.asarray(edge_ids, dtype=np.int32),
            "qp_phys":     qp_phys,
            "qw":          qw,
            "normals":     normals,
            "detJ":        detJ,      # present for uniformity; kernels may ignore it
            "J_inv":       J_inv,
            "phis":        phis,      # unused for boundary
            "gdofs_map":   gdofs_map,
            "h_arr":       h_arr,
            "entity_kind": "edge",
            # side-uniform owner info
            "owner_id":     owner,
            "owner_pos_id": owner,
            "owner_neg_id": -np.ones_like(owner, dtype=np.int32),
        }
        out.update(basis_tabs)
        if need_hess:
            out["Hxi0"] = Hxi0; out["Hxi1"] = Hxi1
        if need_o3:
            out["Txi0"] = Txi0; out["Txi1"] = Txi1
        if need_o4:
            out["Qxi0"] = Qxi0; out["Qxi1"] = Qxi1

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
        need_hess: bool = False,
        need_o3: bool = False,
        need_o4: bool = False,
    ) -> dict:
        """
        Pre-compute geometry/basis for ∫_{Ω ∩ {φ ▷ 0}} (…) dx on CUT elements.
        Returns padded arrays shaped (n_cut, Qmax, ...), aligned with 'eids'.

        Conventions:
        • Only REFERENCE tables are emitted (b_*, g_*, d.._*). Push-forward is done in codegen.py.
        • We triangulate the clipped polygon(s) inside each cut element and map a reference
            triangle quadrature to each subtriangle; physical weights already include |det J_tri|.
        • Padding: per-element quadrature counts may differ → we pad to Qmax and set padded qw=0.
        """
        import numpy as np
        from pycutfem.integration.quadrature import tri_rule as tri_volume_rule
        from pycutfem.fem import transform
        from pycutfem.integration.pre_tabulates import (
            _eval_deriv_q1, _eval_deriv_q2, _eval_deriv_p1,  # (used only if you enable fast per-pt eval)
        )
        # Triangle helpers (Numba + Python fallbacks)
        from pycutfem.ufl.helpers_geom import (
            phi_eval, corner_tris,
            _clip_triangle_to_side_numba,
            _fan_triangulate_numba,
            _map_ref_tri_to_phys_numba,
            clip_triangle_to_side as _clip_triangle_to_side_py,
            fan_triangulate       as _fan_triangulate_py,
            map_ref_tri_to_phys   as _map_ref_tri_to_phys_py,
        )

        try:
            import numba as _nb  # noqa: F401
            _HAVE_NUMBA = True
        except Exception:
            _HAVE_NUMBA = False

        mesh = self.mixed_element.mesh
        me   = self.mixed_element
        fields = list(me.field_names)
        n_union = me.n_dofs_per_elem

        # -------- which reference derivatives do we need? (ensure closure) --------
        derivs = set(derivs)
        max_req = max([0] + [dx + dy for (dx, dy) in derivs])
        if need_o4:   max_req = max(max_req, 4)
        elif need_o3: max_req = max(max_req, 3)
        elif need_hess: max_req = max(max_req, 2)
        derivs_eff = set(derivs)
        if max_req >= 1: derivs_eff |= {(1, 0), (0, 1)}
        if max_req >= 2: derivs_eff |= {(2, 0), (1, 1), (0, 2)}
        if max_req >= 3: derivs_eff |= {(3, 0), (2, 1), (1, 2), (0, 3)}
        if max_req >= 4: derivs_eff |= {(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)}

        # inverse-jet order needed just for geometry mapping (A, A2, A3, A4)
        kmax_jets = 4 if need_o4 else (3 if need_o3 else (2 if need_hess else 1))

        # ---------------- collect CUT elements -------------------------------------
        if hasattr(element_bitset, "to_indices"):
            eids_all = element_bitset.to_indices()
        else:
            eids_all = list(element_bitset)
        eids_all = [int(eid) for eid in eids_all
                    if 0 <= eid < mesh.n_elements and mesh.elements_list[eid].tag == "cut"]
        if not eids_all:
            # empty skeleton with consistent keys
            out = {
                "eids": np.array([], dtype=np.int32),
                "qp_phys": np.empty((0, 0, 2), dtype=float),
                "qw":      np.empty((0, 0),    dtype=float),
                "J_inv":   np.empty((0, 0, 2, 2), dtype=float),
                "detJ":    np.empty((0, 0), dtype=float),
                "normals": np.empty((0, 0, 2), dtype=float),
                "phis":    np.empty((0, 0), dtype=float),
                "gdofs_map": np.empty((0, n_union), dtype=np.int64),
                "h_arr":     np.empty((0,), dtype=float),
                "entity_kind": "element",
            }
            for f in fields:
                out[f"b_{f}"] = np.empty((0, 0, n_union), dtype=float)
                out[f"g_{f}"] = np.empty((0, 0, n_union, 2), dtype=float)
            return out

        # ---------------- cache key & lookup ---------------------------------------
        global _cut_volume_cache
        try:
            _cut_volume_cache
        except NameError:
            _cut_volume_cache = {}
        try:
            subset_hash = _hash_subset(eids_all)
        except NameError:
            subset_hash = tuple(eids_all)
        ls_token = getattr(level_set, "cache_token", None) or id(level_set)
        derivs_key = tuple(sorted((int(dx), int(dy)) for (dx, dy) in derivs))
        cache_key = (
            "cutvol", id(mesh), me.signature(), subset_hash,
            int(qdeg), str(side), derivs_key, self.method, bool(need_hess), bool(need_o3), bool(need_o4), ls_token
        )
        if reuse and cache_key in _cut_volume_cache:
            return _cut_volume_cache[cache_key]

        # ---------------- reference triangle rule ----------------------------------
        qp_ref_tri, qw_ref_tri = tri_volume_rule(qdeg)  # on Δ: (0,0)-(1,0)-(0,1)
        qp_ref_tri = np.asarray(qp_ref_tri, dtype=float)  # (nQ_ref, 2)
        qw_ref_tri = np.asarray(qw_ref_tri, dtype=float)  # (nQ_ref,)
        nQ_ref = qp_ref_tri.shape[0]

        # ---------------- ragged accumulators --------------------------------------
        valid_eids: list[int] = []
        qp_blocks:   list[np.ndarray] = []        # each: (n_q_elem, 2)
        qw_blocks:   list[np.ndarray] = []        # each: (n_q_elem,)
        Jinv_blocks: list[np.ndarray] = []        # each: (n_q_elem, 2, 2)
        phi_blocks:  list[np.ndarray] = []        # each: (n_q_elem,)
        gdofs_map:   list[np.ndarray] = []        # each: (n_union,)
        h_list:      list[float] = []
        H0_blocks:   list[np.ndarray] = []        # each (n_q_elem, 2, 2) if need_hess
        H1_blocks:   list[np.ndarray] = []
        T0_blocks:   list[np.ndarray] = []        # each (n_q_elem, 2, 2, 2) if need_o3
        T1_blocks:   list[np.ndarray] = []
        Q0_blocks:   list[np.ndarray] = []        # each (n_q_elem, 2, 2, 2, 2) if need_o4
        Q1_blocks:   list[np.ndarray] = []

        # per-field ragged basis/deriv lists
        basis_lists: dict[str, list[np.ndarray]] = {}  # keys: "b_f", "g_f", "dXY_f"

        sgn = +1 if side == "+" else -1

        # ---------------- main loop over cut elements ------------------------------
        for eid in eids_all:
            elem = mesh.elements_list[eid]
            # corner triangles of the parent element (tri/tri or quad split into 2 tris)
            tri_local, cn = corner_tris(mesh, elem)  # helper ensures consistent local-node order
            # prepare per-element holders
            xq_elem: list[np.ndarray] = []
            wq_elem: list[float] = []
            Jinv_elem: list[np.ndarray] = []
            phi_elem: list[float] = []
            if need_hess:
                H0_elem: list[np.ndarray] = []
                H1_elem: list[np.ndarray] = []
            if need_o3:
                T0_elem: list[np.ndarray] = []
                T1_elem: list[np.ndarray] = []
            if need_o4:
                Q0_elem: list[np.ndarray] = []
                Q1_elem: list[np.ndarray] = []

            # global DOFs map and element size
            gdofs_map.append(self.get_elemental_dofs(int(eid)))
            h_list.append(float(mesh.element_char_length(int(eid)) or 0.0))

            # loop over the element’s corner triangles
            for loc_tri in tri_local:
                v_ids = [cn[i] for i in loc_tri]                               # 3 node ids
                V     = mesh.nodes_x_y_pos[v_ids].astype(float)                # (3,2)
                v_phi = np.array([phi_eval(level_set, V[0]),
                                phi_eval(level_set, V[1]),
                                phi_eval(level_set, V[2])], dtype=float)

                # clip triangle to requested side and fan-triangulate
                if _HAVE_NUMBA:
                    poly, n_pts = _clip_triangle_to_side_numba(V, v_phi, sgn)
                    if n_pts < 3:
                        continue
                    if n_pts == 3:
                        tris = [(poly[0], poly[1], poly[2])]
                    else:
                        tris_arr, n_tris = _fan_triangulate_numba(poly, n_pts)
                        tris = [(tris_arr[t,0], tris_arr[t,1], tris_arr[t,2]) for t in range(n_tris)]
                else:
                    polys = _clip_triangle_to_side_py(V, v_phi, side=('+' if sgn == +1 else '-'))
                    if not polys:
                        continue
                    tris = []
                    for poly in polys:
                        tris.extend(_fan_triangulate_py(poly))

                # each subtriangle → map ref rule to physical and append QPs
                for (A, B, C) in tris:
                    if _HAVE_NUMBA:
                        x_phys, w_phys = _map_ref_tri_to_phys_numba(
                            np.asarray(A), np.asarray(B), np.asarray(C), qp_ref_tri, qw_ref_tri
                        )
                    else:
                        x_phys, w_phys = _map_ref_tri_to_phys_py(A, B, C, qp_ref_tri, qw_ref_tri)

                    for q in range(nQ_ref):
                        x = x_phys[q]; w = float(w_phys[q])
                        # parent reference coords and inverse-geometry jets
                        s, t  = transform.inverse_mapping(mesh, int(eid), x)
                        rec   = _JET.get(mesh, int(eid), float(s), float(t), upto=kmax_jets)
                        Ji    = rec["A"]  # (2,2)

                        # store per-qp geometry
                        xq_elem.append(np.asarray(x, dtype=float))
                        wq_elem.append(w)
                        Jinv_elem.append(Ji)
                        phi_elem.append(float(phi_eval(level_set, x)))

                        # inverse-map jets (chain rule) if requested
                        if need_hess:
                            A2 = rec["A2"]  # (2,2,2)
                            H0_elem.append(np.asarray(A2[0], dtype=float))
                            H1_elem.append(np.asarray(A2[1], dtype=float))
                        if need_o3:
                            A3 = rec["A3"]  # (2,2,2,2)
                            T0_elem.append(np.asarray(A3[0], dtype=float))
                            T1_elem.append(np.asarray(A3[1], dtype=float))
                        if need_o4:
                            A4 = rec["A4"]  # (2,2,2,2,2)
                            Q0_elem.append(np.asarray(A4[0], dtype=float))
                            Q1_elem.append(np.asarray(A4[1], dtype=float))

                        # Reference basis/derivatives per field at (s,t)
                        for fld in fields:
                            kb, kg = f"b_{fld}", f"g_{fld}"
                            # b_: (n_f,), g_: (n_f,2)
                            basis_lists.setdefault(kb, []).append(
                                np.asarray(me._eval_scalar_basis(fld, float(s), float(t)), dtype=float)
                            )
                            basis_lists.setdefault(kg, []).append(
                                np.asarray(me._eval_scalar_grad (fld, float(s), float(t)), dtype=float)
                            )
                            # requested derivatives (reference), with closure (derivs_eff)
                            for (dx, dy) in derivs_eff:
                                if (dx, dy) == (0, 0):
                                    continue
                                kd = f"d{dx}{dy}_{fld}"
                                basis_lists.setdefault(kd, []).append(
                                    np.asarray(me._eval_scalar_deriv(fld, float(s), float(t),
                                                                    int(dx), int(dy)), dtype=float)
                                )

            # no quadrature in this element? skip
            if not wq_elem:
                continue

            # commit this element’s blocks
            valid_eids.append(int(eid))
            qp_blocks.append(np.vstack(xq_elem).reshape(-1, 2))
            qw_blocks.append(np.asarray(wq_elem, dtype=float).reshape(-1))
            Jinv_blocks.append(np.stack(Jinv_elem, axis=0))      # (n_q,2,2)
            phi_blocks.append(np.asarray(phi_elem, dtype=float).reshape(-1))
            if need_hess:
                H0_blocks.append(np.stack(H0_elem, axis=0))      # (n_q,2,2)
                H1_blocks.append(np.stack(H1_elem, axis=0))      # (n_q,2,2))
            if need_o3:
                T0_blocks.append(np.stack(T0_elem, axis=0))      # (n_q,2,2,2)
                T1_blocks.append(np.stack(T1_elem, axis=0))      # (n_q,2,2,2)
            if need_o4:
                Q0_blocks.append(np.stack(Q0_elem, axis=0))      # (n_q,2,2,2,2)
                Q1_blocks.append(np.stack(Q1_elem, axis=0))      # (n_q,2,2,2,2)

        # ---------------- nothing valid? -------------------------------------------
        if not valid_eids:
            out = {
                "eids": np.array([], dtype=np.int32),
                "qp_phys": np.empty((0, 0, 2), dtype=float),
                "qw":      np.empty((0, 0),    dtype=float),
                "J_inv":   np.empty((0, 0, 2, 2), dtype=float),
                "detJ":    np.empty((0, 0), dtype=float),
                "normals": np.empty((0, 0, 2), dtype=float),
                "phis":    np.empty((0, 0), dtype=float),
                "gdofs_map": np.empty((0, n_union), dtype=np.int64),
                "h_arr":     np.empty((0,), dtype=float),
                "entity_kind": "element",
            }
            for f in fields:
                out[f"b_{f}"] = np.empty((0, 0, n_union), dtype=float)
                out[f"g_{f}"] = np.empty((0, 0, n_union, 2), dtype=float)
            return out

        # ---------------- pad ragged → rectangular (qw=0 on padding) ---------------
        nE   = len(valid_eids)
        sizes = np.array([blk.shape[0] for blk in qw_blocks], dtype=int)
        Qmax = int(sizes.max())

        qp_phys = np.zeros((nE, Qmax, 2), dtype=float)
        qw      = np.zeros((nE, Qmax),    dtype=float)
        J_inv   = np.zeros((nE, Qmax, 2, 2), dtype=float)
        phis    = np.zeros((nE, Qmax),    dtype=float)
        if need_hess:
            Hxi0 = np.zeros((nE, Qmax, 2, 2), dtype=float)
            Hxi1 = np.zeros_like(Hxi0)
        if need_o3:
            Txi0 = np.zeros((nE, Qmax, 2, 2, 2), dtype=float)
            Txi1 = np.zeros_like(Txi0)
        if need_o4:
            Qxi0 = np.zeros((nE, Qmax, 2, 2, 2, 2), dtype=float)
            Qxi1 = np.zeros_like(Qxi0)

        for i in range(nE):
            n = sizes[i]
            qp_phys[i, :n, :]   = qp_blocks[i]
            qw[i, :n]           = qw_blocks[i]
            J_inv[i, :n, :, :]  = Jinv_blocks[i]
            phis[i, :n]         = phi_blocks[i]
            if need_hess:
                Hxi0[i, :n, :, :] = H0_blocks[i]
                Hxi1[i, :n, :, :] = H1_blocks[i]
            if need_o3:
                Txi0[i, :n, :, :, :] = T0_blocks[i]
                Txi1[i, :n, :, :, :] = T1_blocks[i]
            if need_o4:
                Qxi0[i, :n, :, :, :, :] = Q0_blocks[i]
                Qxi1[i, :n, :, :, :, :] = Q1_blocks[i]

        # ---------------- per-field: pad & scatter into union slices ---------------
        out = {
            "eids":      np.asarray(valid_eids, dtype=np.int32),
            "qp_phys":   qp_phys,
            "qw":        qw,                          # already physical weights
            "J_inv":     J_inv,
            "detJ":      np.ones_like(qw),            # not used for dx (kept for uniformity)
            "normals":   np.zeros_like(qp_phys),      # not used in volume integrals
            "phis":      phis,
            "gdofs_map": np.asarray(gdofs_map, dtype=np.int64),
            "h_arr":     np.asarray(h_list, dtype=float),
            "entity_kind": "element",
            "owner_id":   np.asarray(valid_eids, dtype=np.int32),
        }

        # organize ragged per-field lists into per-element arrays
        # We stored values *per quadrature point* in one big stream; here we rebuild per-element blocks.
        # To do that reliably we re-iterate blocks element-by-element (sizes[]).
        # First, collect per-field per-element stacks:
        #   b_lists[f][i] → (n_i, n_f), g_lists[f][i] → (n_i, n_f, 2), d_lists[key][i] → (n_i, n_f)

        # Build indices to slice the flat lists back into elements
        # (We appended one row per qp, per field, in the main loop in element-order.)
        # For each element we have sizes[i] QPs; for each field we appended exactly sizes[i] rows to each list key.
        offsets = np.cumsum([0] + sizes.tolist())  # len = nE+1

        # Collect keys present in basis_lists
        keys_present = list(basis_lists.keys())
        per_field_counts = {f: (me.component_dof_slices[f].stop - me.component_dof_slices[f].start) for f in fields}

        # Helper to rebuild per-field blocks
        def _rebuild_blocks_base(key_fmt: str, fld: str, rank: int):
            """rank=2 → (n_q, n_f), rank=3 → (n_q, n_f,2)."""
            key = key_fmt.format(fld=fld)
            # Extract the flat stacked array for this key
            if key not in basis_lists:
                return None  # was not requested
            flat = np.asarray(basis_lists[key])
            # flat shape: (sum_i n_i, n_f) or (sum_i n_i, n_f, 2)
            n_f = per_field_counts[fld]
            out_list = []
            for i in range(nE):
                start, end = offsets[i], offsets[i+1]
                if rank == 2:
                    out_list.append(flat[start:end, :].reshape(sizes[i], n_f))
                else:
                    out_list.append(flat[start:end, :, :].reshape(sizes[i], n_f, 2))
            return out_list

        # b_*, g_*:
        for fld in fields:
            sl = me.component_dof_slices[fld]
            b_elems = _rebuild_blocks_base("b_{fld}", fld, rank=2)
            g_elems = _rebuild_blocks_base("g_{fld}", fld, rank=3)

            # pad and scatter into union arrays
            b_pad = np.zeros((nE, Qmax, n_union), dtype=float)
            g_pad = np.zeros((nE, Qmax, n_union, 2), dtype=float)
            if b_elems is not None:
                for i in range(nE):
                    n = sizes[i]
                    b_pad[i, :n, sl] = b_elems[i]
            if g_elems is not None:
                for i in range(nE):
                    n = sizes[i]
                    g_pad[i, :n, sl, :] = g_elems[i]
            out[f"b_{fld}"] = b_pad
            out[f"g_{fld}"] = g_pad

        # d{dx}{dy}_* already collected for derivs_eff; pad & scatter

        for fld in fields:
            sl = me.component_dof_slices[fld]
            n_f = sl.stop - sl.start
            for (dx, dy) in sorted(derivs_eff):
                if (dx, dy) == (0, 0):
                    continue
                key = f"d{dx}{dy}_{fld}"
                d_elems = _rebuild_blocks_base(key, fld, rank=2)
                d_pad = np.zeros((nE, Qmax, n_union), dtype=float)
                if d_elems is not None:
                    for i in range(nE):
                        n = sizes[i]
                        d_pad[i, :n, sl] = d_elems[i]
                out[key] = d_pad

        if need_hess:
            out["Hxi0"] = Hxi0
            out["Hxi1"] = Hxi1
        if need_o3:
            out["Txi0"] = Txi0
            out["Txi1"] = Txi1
        if need_o4:
            out["Qxi0"] = Qxi0
            out["Qxi1"] = Qxi1

        if reuse:
            _cut_volume_cache[cache_key] = out
        return out



    # --- tag DOFs from element selection ----------------------------------------
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

