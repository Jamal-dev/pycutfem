# dofhandler.py

from __future__ import annotations

from matplotlib.pylab import f
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
        if self.DEBUG:
            print("-" * 80)
            print(f"Precomputing interface factors for {len(cut_element_ids)} cut elements with qdeg={qdeg}...")
            print("-" * 80)

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
        n_elems = len(valid_cut_eids)
        # n_elems = mesh.n_elements
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
        h_arr = np.zeros((n_elems,))  # Placeholder for element sizes
        

        # --- Loop over valid cut elements ---
        for k, eid in enumerate(valid_cut_eids):
            # h_arr[eid] = mesh.element_char_length(eid)  # Store element size
            # print(f"eid: {eid}, h_arr[eid]: {h_arr[eid]}")  # Debug output
            elem = mesh.elements_list[eid]
            p0, p1 = elem.interface_pts
            h_arr[k] = mesh.element_char_length(eid)

            # --- Quadrature rule on the physical interface segment ---
            q_xi, q_w = line_quadrature(p0, p1, qdeg)

            for q, (xq, wq) in enumerate(zip(q_xi, q_w)):
                qp_phys[k, q] = xq
                qw[k, q] = wq

                # Normal and phi value from the level set
                g = level_set.gradient(xq)
                norm_g = np.linalg.norm(g)
                normals[k, q] = g / (norm_g + 1e-30)
                phis[k, q] = level_set(xq)

                # Jacobian of the parent element at the quadrature point
                xi_ref, eta_ref = transform.inverse_mapping(mesh, eid, xq)
                J = transform.jacobian(mesh, eid, (xi_ref, eta_ref))
                detJ_arr[k, q] = np.linalg.det(J)
                Jinv_arr[k, q] = np.linalg.inv(J)
                for fld in fields:
                    b_tabs[fld][k, q] = me.basis      (fld, xi_ref, eta_ref)
                    g_tabs[fld][k, q] = me.grad_basis (fld, xi_ref, eta_ref)

        # --- Gather results and cache ---
        out = {
            'eids': np.asarray(valid_cut_eids, dtype=int),
            # 'eids': np.arange(n_elems, dtype=int),  
            'qp_phys': qp_phys, 'qw': qw, 'normals': normals, 'phis': phis,
            'detJ': detJ_arr, 'J_inv': Jinv_arr, 'h_arr': h_arr,
            "entity_kind": "element"
        }
        for fld in fields:
            out[f"b_{fld}"] = b_tabs[fld]
            out[f"g_{fld}"] = g_tabs[fld]

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
        level_set: Callable,
        derivs: set[tuple[int, int]],
    ) -> dict:
        """
        Pre‑compute all geometric and algebraic data needed by dGhost JIT kernels.

        The routine now uses the MixedElement‑supplied union size so the output
        is valid for *both* CG and DG discretisations, regardless of how many
        fields or blocks you mix.
        """
        if self.DEBUG:
            print("-" * 80)
            print(f"Precomputing ghost factors for {len(ghost_edge_ids)} edges with qdeg={qdeg} and derivs={derivs}")
            print("-" * 80)
        derivs = derivs | {(0, 0)}
        mesh        = self.mixed_element.mesh
        me          = self.mixed_element
        fields      = me.field_names
        n_union     = self.union_dofs              # <- CG: 36, DG: 44, …
        n_loc       = me.n_dofs_per_elem

        # ---------------------------------- collect valid interior ghost edges
        edge_ids = (ghost_edge_ids.to_indices() if
                    hasattr(ghost_edge_ids, "to_indices") else list(ghost_edge_ids))
        valid_edges = []
        for eid in edge_ids:
            e = mesh.edge(eid)
            if e.right is None:
                continue  # ghost edges are interior

            # Trust the mesh classification / or require at least one CUT neighbor
            left_tag  = mesh.elements_list[e.left].tag
            right_tag = mesh.elements_list[e.right].tag
            if ('cut' in (left_tag, right_tag)) or getattr(e, "tag", "")[:5] == "ghost":
                valid_edges.append(e)

        if not valid_edges:
            raise ValueError("No valid ghost edges found. "
                             "Check that the mesh has been properly cut and tagged.")
            return {"eids": np.array([], dtype=int)}    # early‑out

        # ---------------------------------- allocate dense workspaces
        n_edges = len(valid_edges)
        p0_rep, p1_rep = mesh.nodes_x_y_pos[list(valid_edges[0].nodes)]
        q_xi_rep, q_w_rep = line_quadrature(p0_rep, p1_rep, qdeg)
        n_q = len(q_w_rep)

        # geometry
        qp_phys   = np.zeros((n_edges, n_q, 2))
        qw        = np.zeros((n_edges, n_q))
        normals   = np.zeros((n_edges, n_q, 2))
        J_inv_pos = np.zeros((n_edges, n_q, 2, 2))
        J_inv_neg = np.zeros((n_edges, n_q, 2, 2))
        detJ_pos  = np.zeros((n_edges, n_q))
        detJ_neg  = np.zeros((n_edges, n_q))
        phi_arr = np.zeros((n_edges, n_q)) if level_set else None

        # DOF data (dense)
        gdofs_map = -np.ones((n_edges, n_union), dtype=np.int64)
        pos_map   = -np.ones((n_edges, n_loc),  dtype=np.int32)
        neg_map   = -np.ones((n_edges, n_loc),  dtype=np.int32)

        # basis / derivative tables
        basis_tables = {}
        for fld in fields:
            for side in ("pos", "neg"):
                for dx, dy in derivs:
                    key = f"d{dx}{dy}_{fld}_{side}"
                    basis_tables[key] = np.zeros((n_edges, n_q, n_loc))  # me._n_basis[fld]))

        h_arr = np.zeros((n_edges,))  # Placeholder for element sizes
        # ---------------------------------- main loop over valid edges
        for i, edge in enumerate(valid_edges):
            left_elem = edge.left
            right_elem = edge.right
            if left_elem is not None:
                h_left = mesh.element_char_length(left_elem)
            else:
                h_left = 0.0
            if right_elem is not None:
                h_right = mesh.element_char_length(right_elem)
            else:
                h_right = 0.0
            h_arr[i] = max(h_left, h_right)  # Store element size
            # 1. (+) and (‑) element ids
            phi_left = level_set(np.asarray(mesh.elements_list[edge.left].centroid()))
            pos_eid, neg_eid = (edge.left, edge.right) if phi_left >= 0 else (edge.right, edge.left)

            # 2. union of global DOFs
            pos_dofs = self.get_elemental_dofs(pos_eid)
            neg_dofs = self.get_elemental_dofs(neg_eid)
            global_dofs = np.unique(np.concatenate((pos_dofs, neg_dofs)))

            assert len(global_dofs) == n_union, (
                f"union size mismatch on edge {edge.gid}: "
                f"{len(global_dofs)}  vs  expected {n_union}"
            )

            gdofs_map[i, :n_union] = global_dofs
            pos_map[i] = np.searchsorted(global_dofs, pos_dofs)
            neg_map[i] = np.searchsorted(global_dofs, neg_dofs)

            # 3. quadrature rule & outward normal
            p0, p1 = mesh.nodes_x_y_pos[list(edge.nodes)]
            qp_e, qw_e = line_quadrature(p0, p1, qdeg)
            normal = edge.normal
            if np.dot(normal, np.asarray(mesh.elements_list[pos_eid].centroid()) - qp_e[0]) < 0:
                normal *= -1.0   # ensure normal points from (‑) to (+)

            qp_phys[i] = qp_e
            qw[i]      = qw_e
            normals[i] = normal

            # 4. push‑forward reference bases (both sides)
            for q, xq in enumerate(qp_e):
                phi_arr[i, q] = level_set(xq) if level_set else 0.0
                for side, eid in (("pos", pos_eid), ("neg", neg_eid)):
                    xi, eta = transform.inverse_mapping(mesh, eid, xq)
                    J       = transform.jacobian(mesh, eid, (xi, eta))
                    J_inv   = np.linalg.inv(J)

                    if side == "pos":
                        J_inv_pos[i, q] = J_inv
                        detJ_pos[i, q]  = np.linalg.det(J)
                    else:
                        J_inv_neg[i, q] = J_inv
                        detJ_neg[i, q]  = np.linalg.det(J)

                    sx, sy = J_inv[0, 0], J_inv[1, 1]   # structured quad assumption
                    for fld in fields:
                        for dx, dy in derivs:
                            ref = me.deriv_ref(fld, xi, eta, dx, dy)
                            phys = ref * (sx ** dx) * (sy ** dy)
                            key = f"d{dx}{dy}_{fld}_{side}"
                            basis_tables[key][i, q, :] = phys

        # ---------------------------------- pack & return
        out = {
            "eids":        np.fromiter((e.gid for e in valid_edges), dtype=np.int32),
            "qp_phys":     qp_phys,
            "qw":          qw,
            "normals":     normals,
            "gdofs_map":   gdofs_map,   # dense (‑1 padded) union map
            "pos_map":     pos_map,
            "neg_map":     neg_map,
            "J_inv_pos":   J_inv_pos,
            "J_inv_neg":   J_inv_neg,
            "detJ_pos":    detJ_pos,
            "detJ_neg":    detJ_neg,
            "detJ":        0.5 * (detJ_pos + detJ_neg),
            "J_inv":       0.5 * (J_inv_pos + J_inv_neg),
            "phis":        phi_arr,
            "h_arr":      h_arr,
            "entity_kind": "edge"
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
        Pre-compute all geometry / basis tables that a JIT kernel needs for
        ∫_Γ f dS on *boundary* edges Γ.  Very similar to
        `precompute_ghost_factors` :contentReference[oaicite:0]{index=0} but with only **one**
        element (the left owner) and no ‘neg/pos’ bookkeeping.
        """
        if self.DEBUG:
            print("-" * 80)
            print(f"Precomputing boundary factors for {len(edge_ids)} edges with qdeg={qdeg} and derivs={derivs}")
            print("-" * 80)
        mesh   = self.mixed_element.mesh
        me     = self.mixed_element
        fields = me.field_names

        # ----------- normalise input ------------------------------------
        if hasattr(edge_ids, "to_indices"):
            edge_ids = edge_ids.to_indices()
        edge_ids = [eid for eid in edge_ids if mesh.edge(eid).right is None]
        if not edge_ids:                      # nothing to do
            raise ValueError("No valid boundary edges found in the provided edge IDs.") 
            return {"eids": np.empty(0, dtype=int)}

        cache_key = (_hash_subset(edge_ids), qdeg, tuple(sorted(derivs)))
        if reuse and cache_key in _edge_geom_cache:
            return _edge_geom_cache[cache_key]

        # ----------- representative rule / array sizes ------------------
        p0, p1 = mesh.nodes_x_y_pos[list(mesh.edge(edge_ids[0]).nodes)]
        qp_rep, qw_rep = line_quadrature(p0, p1, qdeg)
        n_q   = len(qw_rep)
        n_loc = me.n_dofs_local
        n_edges = len(edge_ids)

        # ----------- bulk workspaces ------------------------------------
        qp_phys  =  np.zeros((n_edges, n_q, 2))
        qw       =  np.zeros((n_edges, n_q))
        normals  =  np.zeros((n_edges, n_q, 2))
        detJ_arr =  np.zeros((n_edges, n_q))
        phi_arr =   np.zeros((n_edges, n_q))
        Jinv_arr =  np.zeros((n_edges, n_q, 2, 2))
        gdofs_map = np.zeros((n_edges, n_loc), dtype=np.int32)
        h_arr = np.zeros((n_edges,))  # Placeholder for element sizes

        # basis / derivative tables
        basis_tabs = {f"d{dx}{dy}_{fld}": np.zeros((n_edges, n_q, n_loc))
                      for fld in fields for (dx, dy) in derivs}

        # ----------- main loop over edges --------------------------------
        for row, eid_edge in enumerate(edge_ids):
            edge = mesh.edge(eid_edge)
            left_elem = edge.left
            if left_elem is not None:
                h_arr[row] = mesh.element_char_length(left_elem)  # Store element size
            owner = edge.left          # guaranteed not None
            gdofs_map[row] = self.get_elemental_dofs(owner)

            p0, p1 = mesh.nodes_x_y_pos[list(edge.nodes)]
            qpts, qwts = line_quadrature(p0, p1, qdeg)
            n_vec = edge.normal

            qp_phys[row] = qpts
            qw[row]      = qwts
            normals[row] = n_vec        # constant per edge – replicate

            for q, (xq, wq) in enumerate(zip(qpts, qwts)):
                xi, eta = transform.inverse_mapping(mesh, owner, xq)
                J       = transform.jacobian(mesh, owner, (xi, eta))
                J_inv   = np.linalg.inv(J)

                detJ_arr[row, q] = np.linalg.det(J)
                Jinv_arr[row, q] = J_inv

                sx, sy = J_inv[0, 0], J_inv[1, 1]   # axis-aligned quad
                for fld in fields:
                    for dx, dy in derivs:
                        ref   = me.deriv_ref(fld, xi, eta, dx, dy)
                        basis_tabs[f"d{dx}{dy}_{fld}"][row, q] = \
                            ref * (sx**dx) * (sy**dy)

        out = {"eids": np.asarray(edge_ids, dtype=np.int32),
               "qp_phys": qp_phys, "qw": qw, "normals": normals,
               "detJ": detJ_arr, "J_inv": Jinv_arr,
               "gdofs_map": gdofs_map,
               "phis": phi_arr, "h_arr": h_arr,
               "entity_kind": "edge"
               }
        out.update(basis_tabs)

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
    ) -> dict:
        """
        Pre-compute geometry/basis for ∫_{Ω ∩ {φ ▷ 0}} (…) dx on CUT elements only,
        using element-specific clipped triangle quadrature in *physical* space.

        Returns padded arrays shaped (n_cut, max_q, …), aligned with 'eids' order.
        Provides 'qp_phys', 'qw', 'J_inv', 'detJ' (ones), 'normals' (zeros), 'phis',
        and basis tables 'b_<field>', 'g_<field>' (+ 'd{dx}{dy}_{field}' if requested).
        """
        if self.DEBUG:
            print("-" * 80)
            print(f"Precomputing cut volume factors for {len(element_bitset)} elements with qdeg={qdeg} and side='{side}'")
            print("-" * 80)
        import numpy as np
        from pycutfem.integration.quadrature import volume as tri_volume_rule
        from pycutfem.fem import transform
        # helpers shared across Python & JIT paths
        from pycutfem.ufl.helpers_geom import (
            phi_eval, clip_triangle_to_side, fan_triangulate,
            map_ref_tri_to_phys, corner_tris
        )

        mesh = self.mixed_element.mesh
        me   = self.mixed_element
        fields = list(me.field_names)

        # --- collect candidate (cut) element ids ---------------------------------
        eids_all = element_bitset.to_indices() if hasattr(element_bitset, "to_indices") else list(element_bitset)
        eids_all = [eid for eid in eids_all if 0 <= eid < mesh.n_elements]
        if not eids_all:
            return {
                "eids": np.array([], dtype=np.int32),
                "qp_phys": np.empty((0,0,2)), "qw": np.empty((0,0)),
                "J_inv": np.empty((0,0,2,2)), "detJ": np.empty((0,0)),
                "normals": np.empty((0,0,2)), "phis": np.empty((0,0)),
            }

        # reference triangle rule (0,0)-(1,0)-(0,1)
        qp_ref_tri, qw_ref_tri = tri_volume_rule("tri", qdeg)

        # ragged accumulators
        valid_eids   = []
        qp_chunks    = []
        qw_chunks    = []
        Jinv_chunks  = []
        phi_chunks   = []
        basis_lists: dict[str, list[np.ndarray]] = {}

        for eid in eids_all:
            elem = mesh.elements_list[eid]

            # geometric corner-triangle tiling and corner ids
            tri_local, cn = corner_tris(mesh, elem)

            xq_elem, wq_elem, Jinv_elem, phi_elem = [], [], [], []
            per_elem_basis: dict[str, list[np.ndarray]] = {}

            for loc_tri in tri_local:
                v_ids = [cn[i] for i in loc_tri]                # corner ids
                V     = mesh.nodes_x_y_pos[v_ids]              # (3,2) physical vertices
                v_phi = [phi_eval(level_set, xy) for xy in V]  # φ at vertices

                # polygons on requested side ('+' keeps φ>=0, '-' keeps φ<=0)
                polys = clip_triangle_to_side(V, v_phi, side=side)
                if not polys:
                    continue

                for poly in polys:
                    # fan triangulate polygon and integrate each sub-triangle
                    for A, B, C in fan_triangulate(poly):
                        x_phys, w_phys = map_ref_tri_to_phys(A, B, C, qp_ref_tri, qw_ref_tri)

                        for x, w in zip(x_phys, w_phys):
                            # map quadrature point back to parent (ξ,η)
                            xi, eta = transform.inverse_mapping(mesh, eid, x)
                            J       = transform.jacobian(mesh, eid, (xi, eta))
                            Jinv    = np.linalg.inv(J)

                            xq_elem.append(x)
                            wq_elem.append(w)
                            Jinv_elem.append(Jinv)
                            phi_elem.append(phi_eval(level_set, x))

                            # basis & reference grads
                            for fld in fields:
                                kb = f"b_{fld}"
                                kg = f"g_{fld}"
                                per_elem_basis.setdefault(kb, []).append(me.basis(fld, xi, eta))
                                per_elem_basis.setdefault(kg, []).append(me.grad_basis(fld, xi, eta))
                                for (dx, dy) in derivs:
                                    if (dx, dy) == (0, 0):
                                        continue
                                    kd = f"d{dx}{dy}_{fld}"
                                    per_elem_basis.setdefault(kd, []).append(me.deriv_ref(fld, xi, eta, dx, dy))

            if not wq_elem:
                continue

            valid_eids.append(eid)
            qp_chunks.append(np.asarray(xq_elem, dtype=float))       # (n_q,2)
            qw_chunks.append(np.asarray(wq_elem, dtype=float))       # (n_q,)
            Jinv_chunks.append(np.asarray(Jinv_elem, dtype=float))   # (n_q,2,2)
            phi_chunks.append(np.asarray(phi_elem, dtype=float))     # (n_q,)

            # finalize per-element basis arrays → (n_q, n_loc)
            # finalize per-element basis arrays
            for key, rows in per_elem_basis.items():
                arr0 = rows[0]
                if arr0.ndim == 1:                # b_<field>, d{dx}{dy}_<field> → (n_q, n_loc)
                    arr = np.vstack(rows).astype(float)
                elif arr0.ndim == 2:              # g_<field> rows are (n_loc, 2) → stack to (n_q, n_loc, 2)
                    arr = np.stack(rows, axis=0).astype(float)
                else:
                    raise ValueError(f"Unexpected rank for {key}: {arr0.shape}")
                basis_lists.setdefault(key, []).append(arr)


        if not valid_eids:
            raise ValueError("No valid cut elements found in the provided element IDs.")
            return {
                "eids": np.array([], dtype=np.int32),
                "qp_phys": np.empty((0,0,2)), "qw": np.empty((0,0)),
                "J_inv": np.empty((0,0,2,2)), "detJ": np.empty((0,0)),
                "normals": np.empty((0,0,2)), "phis": np.empty((0,0)),
            }
        else: print(f"Found {len(valid_eids)} valid cut elements for cut volume.")

        # --- pad ragged → dense (n_cut, max_q, …) -------------------------------
        def _pad2(list_of_2d, fill=0.0):
            n = len(list_of_2d)
            m = max(a.shape[0] for a in list_of_2d)
            d = list_of_2d[0].shape[1]
            out = np.full((n, m, d), fill, dtype=float)
            for i, a in enumerate(list_of_2d):
                out[i, :a.shape[0], :a.shape[1]] = a
            return out

        def _pad1(list_of_1d, fill=0.0):
            n = len(list_of_1d)
            m = max(a.shape[0] for a in list_of_1d)
            out = np.full((n, m), fill, dtype=float)
            for i, a in enumerate(list_of_1d):
                out[i, :a.shape[0]] = a
            return out

        def _pad22(list_of_3d, fill=0.0):
            n = len(list_of_3d)
            m = max(a.shape[0] for a in list_of_3d)
            out = np.full((n, m, 2, 2), fill, dtype=float)
            for i, a in enumerate(list_of_3d):
                out[i, :a.shape[0], :, :] = a
            return out
        def _pad3(list_of_3d, fill=0.0):
            n = len(list_of_3d)
            m = max(a.shape[0] for a in list_of_3d)   # max_q
            p = list_of_3d[0].shape[1]                # n_loc
            d = list_of_3d[0].shape[2]                # dim (2 in 2D)
            out = np.full((n, m, p, d), fill, dtype=float)
            for i, a in enumerate(list_of_3d):
                out[i, :a.shape[0], :a.shape[1], :a.shape[2]] = a
            return out


        qp_phys = _pad2(qp_chunks, 0.0)
        qw      = _pad1(qw_chunks,  0.0)
        J_inv   = _pad22(Jinv_chunks, 0.0)
        phis    = _pad1(phi_chunks,  0.0)

        # volume-only arrays required by kernel
        normals = np.zeros((qp_phys.shape[0], qp_phys.shape[1], 2), dtype=float)
        detJ    = np.ones_like(qw, dtype=float)   # weights already physical

        out = {
            "eids":    np.asarray(valid_eids, dtype=np.int32),
            "qp_phys": qp_phys,
            "qw":      qw,
            "J_inv":   J_inv,
            "detJ":    detJ,
            "normals": normals,
            "phis":    phis,
            "h_arr":   np.asarray([mesh.element_char_length(e) for e in valid_eids], dtype=float),
            "entity_kind": "element"
        }
        # stitch basis tables (key → (n_cut, max_q, n_loc))
        for key, seq in basis_lists.items():
            if not seq:
                continue
            arr0 = seq[0]
            if arr0.ndim == 3:           # g_<field>: (n_q, n_loc, 2)
                out[key] = _pad3(seq, 0.0)
            elif arr0.ndim == 2:         # b_<field>, d{dx}{dy}_<field>: (n_q, n_loc)
                out[key] = _pad2(seq, 0.0)
            elif arr0.ndim == 1:         # (unlikely here, but safe)
                out[key] = _pad1(seq, 0.0)
            else:
                raise ValueError(f"Unexpected rank for {key}: {arr0.shape}")


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

