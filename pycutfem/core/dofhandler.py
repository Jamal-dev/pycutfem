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
from pycutfem.core.sideconvention import SIDE


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

def clear_caches() -> None:
    """
    Clear all geometry/interface/ghost precompute caches owned by this module.
    Safe to call between test runs or mesh rebuilds.
    """
    _volume_geom_cache.clear()
    _cut_volume_cache.clear()
    _edge_geom_cache.clear()
    _ghost_cache.clear()



def _hash_subset(ids: Sequence[int]) -> int:
    """Stable 64-bit hash for a list / BitSet of indices."""
    h = blake2b(digest_size=8)
    h.update(np.asarray(sorted(ids), dtype=np.int32).tobytes())
    return int.from_bytes(h.digest(), "little")




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


def _ls_fingerprint(level_set):
    """
    Stable identity for a level set. Avoids CPython id() reuse bugs.
    Known types → tuple of parameters; else falls back to id().
    """
    if level_set is None:
        return ("none",)
    # Prefer an explicit token if the class provides one
    tok = getattr(level_set, "cache_token", None)
    if tok is not None:
        return ("token", tok)

    # Affine and Circle are common; capture parameters
    # (works even if you use a subclass that sets these attrs)
    if all(hasattr(level_set, a) for a in ("a", "b", "c")):
        return ("affine", float(level_set.a), float(level_set.b), float(level_set.c))
    if hasattr(level_set, "center") and hasattr(level_set, "radius"):
        cx, cy = map(float, level_set.center)
        return ("circle", cx, cy, float(level_set.radius))

    # Last resort: object identity
    return ("objid", int(id(level_set)))


# -----------------------------------------------------------------------------
#  Main class
# -----------------------------------------------------------------------------
class DofHandler:
    """Centralised DOF numbering and boundary‑condition helpers."""

    # .........................................................................
    def __init__(self, fe_space: Union[Dict[str, Mesh], 'MixedElement'], method: str = "cg",DEBUG = False):
        """
        Initialize a DOF handler.

        Parameters
        ----------
        fe_space : MixedElement | dict[str, Mesh]
            • Preferred: a `MixedElement` whose fields live on a single `Mesh`.
            • Legacy: a mapping {field_name -> Mesh} (one mesh per field).
        method : {'cg', 'dg'}, default 'cg'
            Discretization connectivity:
            - 'cg' builds continuous Lagrange spaces (global continuity within each field).
            - 'dg' builds fully discontinuous spaces (element-local DOFs).
        DEBUG : bool, default False
            Enable extra assertions and verbose internal checks.

        Side effects / Attributes set
        -----------------------------
        field_names : list[str]
            Names of the fields managed by this handler.
        fe_map : dict[str, Mesh]
            Field → mesh association (for MixedElement all fields map to the same mesh).
        element_maps : dict[str, list[list[int]]]
            For each field and element id, the local→global DOF map in lattice order.
        total_dofs : int
            Size of the global union space across all fields.
        dof_map : dict[str, dict]
            CG: {field: {mesh_node_id -> global_dof}} for DOFs that coincide with nodes.
            DG: {field: {mesh_node_id: {elem_id -> global_dof}}}
        _dof_coords : ndarray of shape (total_dofs, 2)
            Physical coordinates of each global DOF (built for MixedElement CG/DG).
        _field_slices : dict[str, ndarray[int]]
            Sorted global DOF ids that belong to each field (CG mixed path).
        field_offsets, field_num_dofs : dict[str, int]
            Legacy offsets and counts (kept for compatibility).
        dof_tags : dict[str, set[int]]
            User-defined DOF tag sets (for BC selection etc.).
        _dof_to_node_map : dict[int, tuple[str, int|None]]
            Reverse map: global_dof → (field, mesh_node_id or None).

        Notes
        -----
        • For MixedElement+CG, global numbering is geometry-independent and field-wise
            continuous. For MixedElement+DG, numbering is element-local.
        • Legacy single-field constructors are retained for backward compatibility.
        """
        if method not in {"cg", "dg"}:
            raise ValueError("method must be 'cg' or 'dg'")
        self.method: str = method
        self._dg_mode: bool = (method == "dg")
        self.DEBUG: bool = DEBUG
        # This will store tags for specific DOFs, e.g., {'pressure_pin': {123}}
        self.dof_tags: Dict[str, Set[int]] = {}
        # This will map a global DOF index back to its (field, node_id) origin
        self._dof_to_node_map: Dict[int, Tuple[str, int]] = {}

        # Detect *which* constructor variant we are using --------------------
        if MixedElement is not None and isinstance(fe_space, MixedElement):
            self.mixed_element: MixedElement = fe_space
            self.field_names: List[str] = list(self.mixed_element.field_names)
            self.q_orders: Dict[str, int] = self.mixed_element.q_orders
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
        """
        Indices (in geometry-lattice order) of geometry nodes used by a field.

        Parameters
        ----------
        p_mesh : int
            Mesh geometry polynomial order for the element (e.g., 3 for Q3).
        p_f : int
            Field polynomial order (e.g., 1 for Q1).
        elem_type : str
            Element type identifier (e.g., 'quad').
        fld : str
            Field name (unused in selection logic, kept for diagnostics).

        Returns
        -------
        list[int]
            Indices into the geometry’s local lattice (η outer, ξ inner) that
            the field uses as interpolation nodes.

        Raises
        ------
        ValueError
            If `p_f > p_mesh` (field cannot exceed geometry order).
        NotImplementedError
            If `elem_type` is not supported.

        Notes
        -----
        • For quads, Qp uses the tensor-product lattice of (p+1)^2 nodes; this
        returns the downsampled subset for the field order.
        """
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
        """
        Build CG global numbering for a MixedElement on a single Mesh, allowing
        *different polynomial orders per field* (e.g., Q2/Q2/Q1).

        Creates
        -------
        element_maps[field][eid] : list[int]
            Per-element local-to-global DOF ids in *lattice order* (η outer, ξ inner).
        _dof_coords[gdof] : (float, float)
            Geometry-independent physical coordinates of every global DOF.
        _field_slices[field] : list[int]
            Global DOFs that belong to that field.
        dof_map[field][mesh_node_id] : int
            For DOFs that coincide with a mesh node (corners & midside); interior
            DOFs do not appear in this map.
        _dof_to_node_map[gdof] : (field:str, mesh_node_id|None)
            Reverse mapping for tests/BC grouping.

        Notes
        -----
        • Per-field continuity is enforced by unifying coincident (x,y) within a field.
        • Fields do not share unknowns: numbering is a *union space* across fields.
        • DOF→node maps are produced by snapping DOF coords to mesh nodes with a tight
        tolerance; this is what makes Q1-on-Q3 map to the coarse 3×3 grid.
        """
        import numpy as np
        from pycutfem.fem import transform

        me   = self.mixed_element
        mesh = me.mesh
        fields = list(self.field_names)
        numset  = getattr(me, "_number_fields", set())
        num_gid = {}      # field -> global id (allocate once)


        # -------- helpers ---------------------------------------------------------
        def _lattice_quad(p: int):
            # (η outer, ξ inner): j=0..p (eta), i=0..p (xi)
            t = np.linspace(-1.0, 1.0, p+1)
            return [(float(xi), float(eta)) for eta in t for xi in t]

        def _lattice_tri(p: int):
            # Reference triangle in (ξ,η): 0 ≤ ξ, 0 ≤ η, ξ+η ≤ 1
            # Lexicographic order (η outer, ξ inner) matches standard Pp node ordering
            if p < 0:
                return []
            t = np.linspace(0.0, 1.0, p+1)
            pts = []
            for j, eta in enumerate(t):
                # ξ runs 0..1-eta in p+1-j steps
                for i in range(p+1 - j):
                    xi = t[i]
                    pts.append((float(xi), float(eta)))
            return pts
        def _q(x: float, ndp: int = 12) -> float:
            # quantize to make coordinate keys stable across elements
            return float(round(x, ndp))

        # -------- pass 1: per-field CG maps using the transform -------------------
        # element_maps[field][eid] -> list of global dofs (lattice order)
        element_maps: dict[str, list[list[int]]] = {f: [] for f in fields}

        # within-field dictionary: (qx, qy) -> gdof  (keeps CG continuity per field)
        key2gdof: dict[str, dict[tuple[float, float], int]] = {f: {} for f in fields}

        # accumulation of coords and field membership
        dof_coords: list[tuple[float, float]] = []
        field_gsets: dict[str, set[int]] = {f: set() for f in fields}

        next_gid = 0

        n_cells = len(mesh.elements_connectivity)
        for eid in range(n_cells):
            for f in fields:
                if f in numset:
                    if f not in num_gid:
                        num_gid[f] = next_gid; next_gid += 1
                        dof_coords.append((0.0, 0.0))
                    field_gsets[f].add(num_gid[f])
                    element_maps[f].append([int(num_gid[f])]) 
                    continue
                p = int(me._field_orders[f])
                lat = _lattice_tri(p) if mesh.element_type == "tri" else _lattice_quad(p)
                loc_gids: list[int] = []

                for (xi, eta) in lat:
                    X = transform.x_mapping(mesh, int(eid), (xi, eta))  # (x,y)
                    k = (_q(float(X[0])), _q(float(X[1])))

                    gd = key2gdof[f].get(k)
                    if gd is None:
                        gd = next_gid
                        next_gid += 1
                        key2gdof[f][k] = gd
                        dof_coords.append((float(X[0]), float(X[1])))

                    loc_gids.append(gd)
                    field_gsets[f].add(gd)

                element_maps[f].append(loc_gids)

        self.element_maps = element_maps
        self.total_dofs   = int(next_gid)
        self._dof_coords  = np.asarray(dof_coords, dtype=float)

        # Per-field slices (all global DOFs belonging to that field)
        self._field_slices = {f: np.array(sorted(field_gsets[f]), dtype=int) for f in fields}

        # -------- pass 2: map DOF coords to mesh nodes (for tests/BC grouping) ----
        # Only DOFs that *coincide* with a mesh node get a node mapping.
        # Interior/lobatto interior points won't map to nodes → (field, None).
        nodes_xy = np.asarray(mesh.nodes_x_y_pos, dtype=float)
        node_lookup = {(_q(float(x)), _q(float(y))): int(nid)
                    for nid, (x, y) in enumerate(nodes_xy)}

        self.dof_map = {f: {} for f in fields}        # node_id -> gdof (per field)
        self._dof_to_node_map = {}                    # gdof -> (field, node_id|None)

        TOL = 1e-12  # geometric snap tolerance (squared distance check uses TOL^2)

        for f in fields:
            for gd in self._field_slices[f]:
                x, y = self._dof_coords[int(gd)]
                nkey = (_q(x), _q(y))
                nid = node_lookup.get(nkey, None)
                if nid is None:
                    # nearest neighbor fallback for tiny drift
                    dx = nodes_xy[:, 0] - x
                    dy = nodes_xy[:, 1] - y
                    j = int(np.argmin(dx*dx + dy*dy))
                    if (dx[j]*dx[j] + dy[j]*dy[j]) <= (TOL*TOL):
                        nid = j

                if nid is not None:
                    self.dof_map[f][int(nid)] = int(gd)
                    self._dof_to_node_map[int(gd)] = (f, int(nid))
                else:
                    self._dof_to_node_map[int(gd)] = (f, None)

        # -------- optional metadata kept minimal & compatible ---------------------
        # Some code reads these; keep harmless defaults.
        self.field_num_dofs = {f: len(self._field_slices[f]) for f in fields}
        self.field_offsets  = {f: 0 for f in fields}  # not used when get_field_slice is available





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
        """
        Build legacy continuous (CG) numbering for the legacy {field -> Mesh} path.

        For each field:
        - Assign one global DOF per mesh node (continuous across elements).
        - Fill `element_maps[field][eid]` with the element's node DOFs in geometry order.
        - Populate `field_offsets`, `field_num_dofs`, `dof_map[field]`.
        - Update `total_dofs`.

        Notes
        -----
        • This path is only used when the handler is constructed with a dict[str, Mesh].
        • For MixedElement, use `_build_maps_cg_mixed` instead.
        """
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
        """
        Build legacy discontinuous (DG) numbering for the legacy {field -> Mesh} path.

        For each field:
        - Allocate element-local DOFs (no sharing between elements).
        - Fill `element_maps[field][eid]` with a fresh block of DOFs for each element.
        - Populate `dof_map[field]` as {mesh_node_id: {elem_id -> global_dof}} to
        support node-addressed queries if needed.
        - Update `field_offsets`, `field_num_dofs`, `total_dofs`.

        Notes
        -----
        • This path is only used when the handler is constructed with a dict[str, Mesh].
        • For MixedElement DG, use `_build_maps_dg_mixed`.
        """
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
    
    def _ensure_lattice_cache(self):
        """
        Cache for each p: list of local indices on the 4 edges of a quad.
        Ordering: edges (0: bottom η=-1, 1: right ξ=+1, 2: top η=+1, 3: left ξ=-1),
        and local indices along each edge in ascending param (xi or eta).
        """
        import numpy as np
        if hasattr(self, "_edge_lattice_cache"):
            return
        self._edge_lattice_cache = {}  # p -> [ [indices_edge0], ..., [indices_edge3] ]
        for p in set(self.mixed_element._field_orders.values()):
            t = np.arange(p+1)
            # map (i,j) -> k = j*(p+1) + i  (eta outer, xi inner)
            def k(i,j): return int(j*(p+1) + i)
            edges = [
                [k(i,0)    for i in t],        # bottom (η=-1): j=0
                [k(p,j)    for j in t],        # right  (ξ=+1): i=p
                [k(i,p)    for i in t[::-1]],  # top    (η=+1): j=p, reverse to go left->right
                [k(0,j)    for j in t[::-1]],  # left   (ξ=-1): i=0,  reverse to go bottom->top
            ]
            self._edge_lattice_cache[p] = edges
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
        """
        Return matching DOF lists (left, right) on a shared interior edge for DG spaces.

        Parameters
        ----------
        field : str
            Field name.
        edge_gid : int
            Global edge id.

        Returns
        -------
        (left_gdofs, right_gdofs) : tuple[list[int], list[int]]
            DOFs along the edge as seen from the left and right owner elements,
            ordered consistently with the geometry's local edge orientation.

        Raises
        ------
        RuntimeError
            If called in CG mode.
        ValueError
            If the edge is a boundary (missing one owner).

        Notes
        -----
        • This requires a precise local-edge → lattice-index mapping per element.
        The DG mixed-order version is not implemented here and will need a careful
        owner/edge-localization consistent with your basis tabulation.
        """
        if not self._dg_mode:
            raise RuntimeError("Edge DOF pairs only relevant for DG spaces.")
        mesh = self.fe_map[field]
        edge = mesh.edges_list[edge_gid]
        if edge.left is None or edge.right is None:
            raise ValueError("Edge is on boundary – no right element.")
        
        raise NotImplementedError("get_dof_pairs_for_edge needs careful implementation for mixed DG.")

    def _require_cg(self, name: str) -> None:
        """
        Guard for CG-only APIs.

        Parameters
        ----------
        name : str
            Name of the API being guarded (used in the error message).

        Raises
        ------
        NotImplementedError
            If the handler is in DG mode.
        """
        if self._dg_mode:
            raise NotImplementedError(f"{name} not available for DG spaces – every element owns its DOFs.")

    def get_field_slice(self, field: str):
        """
        Return the sorted list of global DOF ids that belong to a field (CG only).

        Parameters
        ----------
        field : str
            Field name.

        Returns
        -------
        list[int]
            Sorted global DOF ids of the field.

        Raises
        ------
        KeyError
            If the field is unknown.
        NotImplementedError
            If called in DG mode.

        Notes
        -----
        • In MixedElement CG, `_field_slices[field]` is authoritative.
        • A legacy fallback uses `dof_map[field].values()` if slices are absent.
        """
        self._require_cg("get_field_slice")
        try:
            return self._field_slices[field].tolist()
        except Exception:
            raise KeyError(f"Unknown field '{field}'")
        
    def get_field_dof_coords(self, field: str) -> np.ndarray:
        """
        Physical coordinates of all DOFs of a field (CG).

        Equivalent to `get_dof_coords(field)`.

        Returns
        -------
        np.ndarray, shape (n_field_dofs, 2)
            Coordinates in the same order as `get_field_slice(field)`.
        """
        import numpy as np
        self._require_cg("get_field_dof_coords")
        idx = np.asarray(self.get_field_slice(field), dtype=int)
        return self._dof_coords[idx]






    def get_field_dofs_on_nodes(self, field: str) -> np.ndarray:
        """
        Return DOFs of a field that coincide with mesh nodes (CG only).

        Parameters
        ----------
        field : str
            Field name.

        Returns
        -------
        np.ndarray
            Sorted array of global DOF ids that have an associated mesh node id.

        Raises
        ------
        ValueError
            If the field is unknown.
        NotImplementedError
            If called in DG mode.

        Notes
        -----
        • Interior high-order DOFs (that do not sit on nodes) are not included.
        """
        self._require_cg("get_field_dofs_on_nodes")
        if field not in self.field_names:
            raise ValueError(f"Unknown field '{field}'.")
        return np.asarray(sorted(self.dof_map[field].values()), dtype=int)


        
    # ------------------------------------------------------------------
    #  Tagging and Dirichlet handling (CG‑only)
    # ------------------------------------------------------------------
    def tag_dofs_by_locator_map(
            self,
            tag_map: dict[str, "Callable[[float, float], bool]"],
            fields: list[str] | None = None,
            tol: float = 0.0,
        ) -> dict[str, set[int]]:
        """
        Tag DOFs by coordinate predicates on *DOF coordinates* (geometry-independent).
        Evaluates locator(x,y) on every DOF of the requested fields.
        Automatically skips NumberSpace fields (global scalars).
        """

        # Ensure DOF coordinates exist (built from lattice via transform.x_mapping)
        self._ensure_dof_coords()

        # Field selection, skipping NumberSpace automatically
        number_fields = set(getattr(self.mixed_element, "_number_fields", set()))
        all_fields    = list(self.field_names)
        if fields is None:
            fields = [f for f in all_fields if f not in number_fields]
        else:
            fields = [f for f in fields if f not in number_fields]

        # Helper for optional tolerance wrapper
        def _wrap(locator):
            if tol <= 0: return locator
            def L(x, y):
                v = locator(x, y)
                return bool(v)  # user can bake tol in their locator
            return L

        if not hasattr(self, "dof_tags"):
            self.dof_tags = {}
        out: dict[str, set[int]] = {}

        for tag, loc in tag_map.items():
            locator = _wrap(loc)
            sel: set[int] = set()
            for f in fields:
                ids = np.asarray(self.get_field_slice(f), dtype=int)
                XY  = self._dof_coords[ids]   # (#ids,2)
                # robust per-DOF evaluation (works with numpy or python bools)
                for gd, (x, y) in zip(ids, XY):
                    try:
                        if bool(locator(float(x), float(y))):
                            sel.add(int(gd))
                    except Exception:
                        pass
            self.dof_tags.setdefault(tag, set()).update(sel)
            out[tag] = sel
        return out


    


    def tag_dof_by_locator(self, tag: str, field: str, locator, find_first: bool = True, **_):
        """Deprecated: use `tag_dofs_by_locator_map({tag: locator}, fields=[field])`.
        Kept for compatibility."""
        sel = self.tag_dofs_by_locator_map({tag: locator}, fields=[field])[tag]
        if find_first and sel:
            # reduce to a single (deterministic) DOF
            keep = min(sel)
            self.dof_tags[tag] = {keep}
            return {keep}
        return sel



    def element_dofs(self, field: str, eid: int) -> list[int]:
        """
        Global DOF ids of `field` on element `eid`.

        Returns
        -------
        list[int]
            DOFs in lattice order (η outer, ξ inner) for the field’s order.
        """
        return [int(g) for g in self.element_maps[field][int(eid)]]

    def element_dof_coords(self, field: str, eid: int):
        """
        Physical coordinates of the element’s DOFs for a field.

        Returns
        -------
        np.ndarray, shape (n_loc, 2)
            Coordinates matching `element_dofs(field, eid)` order.
        """
        import numpy as np
        idx = np.asarray(self.element_maps[field][int(eid)], dtype=int)
        return self._dof_coords[idx]


    def edge_dofs(self, field: str, edge_id: int, owner: str = "left") -> list[int]:
        """
        Global DOF ids of `field` that lie on an edge, in a consistent order.

        Parameters
        ----------
        field : str
            Field name.
        edge_id : int
            Global edge id.
        owner : {'left','right'}, default 'left'
            Which owner element to use for local edge indexing; this fixes a
            consistent parametric direction along the edge.

        Returns
        -------
        list[int]
            `p_field+1` DOFs that lie on the edge, ordered along the owner’s
            local edge from start to end (bottom→right→top→left conventions).

        Notes
        -----
        • The count depends on the *field order*, not the mesh geometry order
        (e.g., Q3/Q1 returns 2 DOFs per element along the edge for the Q1 field).
        • For boundary edges (single owner) the existing owner is used.
        """
        self._ensure_lattice_cache()
        p = self.mixed_element._field_orders[field]
        edge = self.mixed_element.mesh.edge(int(edge_id))

        eid = edge.left if owner == "left" else edge.right
        if eid is None:
            # boundary edge with only one owner -> use whichever exists
            eid = edge.left if edge.left is not None else edge.right
        eid = int(eid)

        # find local edge index 0..3 by matching endpoints to the owner's corner nodes
        el = self.mixed_element.mesh.elements_list[eid]
        cn = list(self.mixed_element.mesh.elements_corner_nodes[eid]) if hasattr(self.mixed_element.mesh, "elements_corner_nodes") else None
        # fallback: detect by geometry: choose the edge whose two end nodes match edge.nodes
        local_edge = None
        endset = {int(edge.nodes[0]), int(edge.nodes[1])}
        for le, (nA, nB) in enumerate([(0,1), (1,2), (2,3), (3,0)]):
            if cn is not None:
                if {int(cn[nA]), int(cn[nB])} == endset:
                    local_edge = le; break
        if local_edge is None:
            # geometric fallback (rare)
            local_edge = 0

        loc_idx = self._edge_lattice_cache[p][local_edge]   # local lattice indices along that edge
        gdofs = self.element_maps[field][eid]
        return [int(gdofs[i]) for i in loc_idx]

    
    def build_high_order_topology_view(self, field: str):
        """
        Construct a non-mutating view that exposes per-edge DOFs for a field.

        Returns
        -------
        dict with keys
            'dof_ids'  : {field: list[list[int]]}
                Per-element local→global DOF ids (alias of `element_maps[field]`).
            'dof_xy'   : {field: np.ndarray}
                (n_field_dofs, 2) coordinates for the field’s DOFs (slice order).
            'edge_dofs': {field: dict[int, list[int]]}
                For each edge id, the global DOFs of `field` that lie on that edge,
                chosen from the edge’s owner element to give a consistent ordering.

        Notes
        -----
        • This is a convenience view for assemblers/diagnostics; it does not change
        Mesh, Elements, or Edges.
        """
        import numpy as np
        me   = self.mixed_element
        mesh = me.mesh
        p    = me._field_orders[field]

        # ensure coords at least once
        self._ensure_dof_coords()

        # local lattice to detect which local DOFs live on each *reference* edge
        t = np.linspace(-1.0, 1.0, p + 1)
        lattice = [(float(xi), float(eta)) for eta in t for xi in t]
        tol = 1e-12
        loc_edge = {
            0: [k for k,(xi,eta) in enumerate(lattice) if abs(eta + 1.0) <= tol],  # bottom
            1: [k for k,(xi,eta) in enumerate(lattice) if abs(xi  - 1.0) <= tol],  # right
            2: [k for k,(xi,eta) in enumerate(lattice) if abs(eta - 1.0) <= tol],  # top
            3: [k for k,(xi,eta) in enumerate(lattice) if abs(xi  + 1.0) <= tol],  # left
        }

        edge_dofs: dict[int, list[int]] = {}
        for eid, cell in enumerate(self.element_maps[field]):
            # local edges 0..3 map to mesh edge gids in consistent order
            for l_edge, edge_gid in enumerate(self.mixed_element.mesh.elements_list[eid].edges):
                # take ownership once (left owner)
                eobj = self.mixed_element.mesh.edge(edge_gid)
                if eobj.left != eid:
                    continue
                for k in loc_edge[l_edge]:
                    edge_dofs.setdefault(edge_gid, []).append(int(cell[k]))

        return {
            "dof_ids": {field: self.element_maps[field]},
            "dof_xy":  {field: self.get_field_dof_coords(field)},
            "edge_dofs": {field: {eid: self.edge_dofs(field, eid) for eid in range(len(self.mixed_element.mesh.edges_list))}},
        }

    # ---------- Utility: extract edge ids from a BitSet/mask/list ----------
    def _edge_ids_from_any(self, edge_sel):
        """
        Normalize an edge selection into a list of edge ids.

        Accepts
        -------
        • BitSet with `.to_indices()` or `.array`
        • str : interpreted as an edge tag (via `mesh.edge_bitset(tag)`)
        • 1D bool mask of length n_edges
        • Iterable of ints (edge ids)

        Returns
        -------
        list[int]
            Edge ids.

        Raises
        ------
        ValueError
            If the input cannot be interpreted.
        """
        import numpy as np
        mesh = self.mixed_element.mesh

        # BitSet with to_indices
        if hasattr(edge_sel, "to_indices"):
            return list(edge_sel.to_indices())

        # BitSet with array
        if hasattr(edge_sel, "array"):
            arr = np.asarray(edge_sel.array, dtype=bool)
            return list(np.nonzero(arr)[0])

        # Tag string -> bitset
        if isinstance(edge_sel, str):
            bs = mesh.edge_bitset(edge_sel)
            if hasattr(bs, "to_indices"):
                return list(bs.to_indices())
            arr = np.asarray(getattr(bs, "array", []), dtype=bool)
            return list(np.nonzero(arr)[0])

        # Numpy/bool mask or explicit ids
        arr = np.asarray(edge_sel)
        if arr.dtype == bool:
            return list(np.nonzero(arr)[0])
        if arr.ndim == 1:
            return [int(x) for x in arr.tolist()]

        raise ValueError("Unsupported edge selection type.")

    def ensure_cg_adjacency(self, field: str) -> None:
        """
        Build and cache DOF↔element adjacency for a CG field.

        Populates
        ---------
        self._cg_adj[field] : dict[int, set[int]]
            Map from global DOF id → set of element ids that contain that DOF.

        Parameters
        ----------
        field : str
            Field name (CG).

        Raises
        ------
        ValueError
            If the field is unknown.

        Returns
        -------
        None
        """
        if field not in self.field_names:
            raise ValueError(f"Unknown field '{field}'.")
        if not hasattr(self, "_cg_adj"):
            self._cg_adj = {}
        if field in self._cg_adj:
            return
        d2e = {}
        for eid, gds in enumerate(self.element_maps[field]):
            for g in gds:
                d2e.setdefault(int(g), set()).add(int(eid))
        self._cg_adj[field] = d2e


    def dof_bitset_from_elements(self, field: str, elem_mask, strict: bool=True) -> set[int]:
        """
        Convert an element bitset/mask/list -> set of global DOFs (for `field`).
        strict=True  : DOFs whose *entire* support lies in elem_mask.
        strict=False : DOFs that touch any element in elem_mask.
        """
        import numpy as np
        if hasattr(elem_mask, "to_numpy"):
            sel = np.nonzero(elem_mask.to_numpy())[0]
        else:
            arr = np.asarray(elem_mask)
            sel = np.nonzero(arr)[0] if arr.dtype==bool else np.asarray(arr, int)
        inside = set(int(e) for e in sel)

        self.ensure_cg_adjacency(field)
        d2e = self._cg_adj[field]

        if strict:
            return {gd for gd, adj in d2e.items() if adj.issubset(inside)}
        else:
            return {gd for gd, adj in d2e.items() if (adj & inside)}

    def tag_dof_bitset(self, tag: str, field: str, elem_mask, strict: bool=True) -> set[int]:
        """Convenience: compute a DOF set from elements and store in self.dof_tags[tag]."""
        sel = self.dof_bitset_from_elements(field, elem_mask, strict=strict)
        self.dof_tags.setdefault(tag, set()).update(sel)
        return sel
    
    def _ensure_dof_coords(self):
        """
        Ensure `self._dof_coords` exists.

        For MixedElement:
            Builds coordinates from `element_maps` by evaluating the reference
            lattice points through `transform.x_mapping` on each element.
        For legacy paths where coordinates are already present, this is a no-op.

        Returns
        -------
        None
        """
        from pycutfem.fem import transform

        if isinstance(getattr(self, "_dof_coords", None), np.ndarray):
            return

        me   = self.mixed_element
        mesh = me.mesh

        # gather all DOF ids
        all_ids = set()
        for f in self.field_names:
            for cell in self.element_maps[f]:
                all_ids.update(int(g) for g in cell)
        n = (max(all_ids) + 1) if all_ids else 0
        coords = np.full((n, 2), np.nan, dtype=float)

        # ref lattice (eta outer, xi inner)
        def _lat(p: int, elem_type: str):
            if elem_type == "tri":
                t = np.linspace(0.0, 1.0, p+1)
                return [(float(t[i]), float(t[j])) for j in range(p+1) for i in range(p+1-j)]
            else:
                t = np.linspace(-1.0, 1.0, p+1)
                return [(float(xi), float(eta)) for eta in t for xi in t]

        for f in self.field_names:
            p = me._field_orders[f]
            lat = _lat(p, mesh.element_type)
            for eid, cell in enumerate(self.element_maps[f]):
                for k, gd in enumerate(cell):
                    gd = int(gd)
                    if not np.isfinite(coords[gd, 0]):
                        xi, eta = lat[k]
                        X = transform.x_mapping(mesh, eid, (xi, eta))
                        coords[gd, 0] = float(X[0])
                        coords[gd, 1] = float(X[1])

        self._dof_coords = coords


    def _ensure_node_maps(self, tol: float = 1e-12):
        """
        Populate:
        • self.dof_map[field]: {mesh_node_id -> global_dof}
        • self._dof_to_node_map[gdof] -> (field, mesh_node_id)

        Uses nearest mesh node to each DOF coordinate (within tol), so
        Q2/Q3 fields map to mid-edge nodes too; Q1 maps to the coarser subset.
        """
        import numpy as np

        if getattr(self, "dof_map", None) and all(isinstance(self.dof_map.get(f, None), dict) and self.dof_map[f]
                                                for f in self.field_names):
            return  # already built

        self._ensure_dof_coords()
        mesh = self.mixed_element.mesh
        node_xy = mesh.nodes_x_y_pos

        # quick lookup by quantized coordinate
        def _key(xy):
            return (round(float(xy[0]), 12), round(float(xy[1]), 12))
        node_lookup = {_key(xy): int(i) for i, xy in enumerate(node_xy)}

        self.dof_map = {f: {} for f in self.field_names}
        self._dof_to_node_map = {}

        for f in self.field_names:
            for gd in self.get_field_slice(f):
                x, y = self._dof_coords[int(gd)]
                k = _key((x, y))
                nid = node_lookup.get(k, None)
                if nid is None:
                    # nearest node within tol
                    d2 = ((node_xy[:, 0] - x) ** 2 + (node_xy[:, 1] - y) ** 2)
                    j = int(np.argmin(d2))
                    if d2[j] > tol ** 2:
                        # skip if no nearby mesh node (should not happen on structured meshes)
                        continue
                    nid = j
                self.dof_map[f][int(nid)] = int(gd)
                self._dof_to_node_map[int(gd)] = (f, int(nid))

    def get_dirichlet_data(
        self,
        bcs,
        locators: Dict[str, Callable[[float, float], bool]] | None = None
    ) -> Dict[int, float]:
        """
        Build {global_dof -> value} for Dirichlet BCs using:
        • edge tags (mesh.edges_list with e.tag == domain_tag) → e.all_nodes
        • dof_map[field] to convert nodes -> DOFs (field-order aware)
        • optional self.dof_tags[tag] (intersected with field) and locator overrides
        """
        import numpy as np

        self._require_cg("Dirichlet BC evaluation")
        locators = locators or {}
        mesh = self.mixed_element.mesh

        # make sure coords/maps exist for locator/grouping tests
        if not isinstance(getattr(self, "_dof_coords", None), np.ndarray):
            # any field triggers coord build in your current code path
            _ = self.get_field_slice(self.field_names[0])
        if not getattr(self, "dof_map", None):
            # after the patch above, dof_map is built in _build_maps_cg_mixed
            pass

        out: Dict[int, float] = {}
        if not hasattr(bcs, "__iter__") or isinstance(bcs, (str, bytes)):
            bcs = [bcs]

        field_sets = {f: set(self.get_field_slice(f)) for f in self.field_names}

        for bc in bcs:
            # normalize inputs
            typ = (getattr(bc, "bc_type", None) or getattr(bc, "method", None) or getattr(bc, "type", None) or "").lower()
            if typ != "dirichlet":
                continue
            field = (getattr(bc, "field_name", None) or getattr(bc, "field", None) or getattr(bc, "name", None))
            tag   = (getattr(bc, "domain_tag", None) or getattr(bc, "domain", None) or getattr(bc, "tag", None))
            if not isinstance(field, str) or not isinstance(tag, str):
                continue

            selected: set[int] = set()

            # (A) edge tag → edge.all_nodes → dof_map[field]
            node2dof = self.dof_map.get(field, {})
            for e in getattr(mesh, "edges_list", []):
                if getattr(e, "tag", None) == tag:
                    for nid in getattr(e, "all_nodes", ()):
                        gd = node2dof.get(int(nid))
                        if gd is not None:
                            selected.add(int(gd))

            # (B) pre-tagged DOFs (already in DOF space) → restrict to this field
            if tag in getattr(self, "dof_tags", {}):
                selected |= (set(self.dof_tags[tag]) & field_sets[field])

            # (C) locator override (evaluate on this field's DOF coords)
            locator = locators.get(tag)
            if locator is not None:
                idx = np.asarray(list(field_sets[field]), dtype=int)
                XY  = self._dof_coords[idx]
                mask = np.array([bool(locator(float(x), float(y))) for (x, y) in XY], dtype=bool)
                selected |= set(int(g) for g, m in zip(idx, mask) if m)

            # (D) assign values
            val = getattr(bc, "value", 0.0)
            if callable(val):
                for gd in selected:
                    x, y = self._dof_coords[int(gd)]
                    out[int(gd)] = float(val(float(x), float(y)))
            else:
                vv = float(val)
                for gd in selected:
                    out[int(gd)] = vv

        return out






    
    # --- Mesh helpers (wrappers) -------------------------------------------------
    def classify_from_levelset(self, level_set, tol: float = 1e-12):
        """
        Convenience wrapper: run element/edge classification and interface segment
        construction on the handler’s mesh using a given level set.
        """
        m = self.mixed_element.mesh
        m.classify_elements(level_set, tol=tol)
        m.classify_edges(level_set, tol=tol)
        m.build_interface_segments(level_set)

    def tag_boundary_edges(self, tag_map: dict[str, "Callable[[float,float], bool]"]):
        """
        Tag boundary edges on the handler’s mesh using {name: locator(x,y)}.
        """
        self.mixed_element.mesh.tag_boundary_edges(tag_map)

    def element_bitset(self, name: str):
        """
        Access an element BitSet by name from the handler’s mesh (e.g., 'inside').
        """
        return self.mixed_element.mesh.element_bitset(name)

    def edge_bitset(self, name: str):
        """
        Access an edge BitSet by name from the handler’s mesh (e.g., 'boundary').
        """
        return self.mixed_element.mesh.edge_bitset(name)





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
        DEPRECATED — Prefer `apply_bcs(K, F, bcs)` or the solver’s reduction API.

        Apply Dirichlet boundary conditions directly to a global vector `vec`.
        For each DOF selected by `bcs`, set `vec[dof] = value`.

        Parameters
        ----------
        vec : np.ndarray
            Global vector to be modified in place.
        bcs : BoundaryCondition | Iterable[BoundaryCondition]
            Dirichlet BC specifications. Selection follows `get_dirichlet_data`.

        Returns
        -------
        None
        """
        for dof, val in self.get_dirichlet_data(bcs).items():
            if dof < vec.size:
                vec[dof] = val

    def _expand_bc_specs(
        self, bcs: Union[BcLike, Iterable[BcLike]]
    ) -> List[Tuple[str, Any, Any]]:
        """
        Normalize a collection of boundary condition specifications.

        Parameters
        ----------
        bcs : BoundaryCondition | Mapping | Iterable
            Either a single BC or an iterable of BC-like objects. The BC must
            provide at least: `field`, and one of `domain`, `domain_tag`, or `tag`,
            and a `value` (constant or callable).

        Returns
        -------
        list[tuple[str, Any, Any]]
            A list of triples `(field, domain_tag, value)` suitable for downstream
            selection logic.

        Notes
        -----
        • This is a lightweight adapter used by Dirichlet collection helpers.
        """
        if not bcs: return []
        items = bcs if isinstance(bcs, (list, tuple, set)) else [bcs]
        out = []
        for bc in items:
            if isinstance(bc, BoundaryCondition):
                domain = getattr(bc, "domain", None) or getattr(bc, "domain_tag", None) or getattr(bc, "tag", None)
                out.append((bc.field, domain, bc.value))
        return out
    
    def get_dof_coords(self, field: str):
        """
        Physical coordinates of all DOFs that belong to a field (CG, mixed).

        Parameters
        ----------
        field : str
            Field name.

        Returns
        -------
        np.ndarray, shape (n_field_dofs, 2)
            Coordinates in the same order as `get_field_slice(field)`.

        Notes
        -----
        • Alias used in tests; identical to `get_field_dof_coords`.
        • Raises KeyError if the field is unknown; NotImplementedError in DG mode.
        """
        import numpy as np
        self._ensure_dof_coords()
        idx = np.asarray(self.get_field_slice(field), int)
        return self._dof_coords[idx]

    def rebuild_dof_map_from_coords(self, tol: float = 1e-12):
        """
        Rebuild the node↔DOF maps from DOF coordinates.

        Fills, for each field:
            • `self.dof_map[field][mesh_node_id] = global_dof`
            for DOFs whose coordinates coincide with a mesh node (within `tol`).
            • `self._dof_to_node_map[gdof] = (field, node_id|None)`.

        Parameters
        ----------
        tol : float, default 1e-12
            Euclidean snapping tolerance in physical space. DOFs whose coordinates
            are within `tol` of a mesh node are mapped to that node; interior
            high-order DOFs remain unmapped (node_id = None).

        Returns
        -------
        None
        """
        import numpy as np
        self._ensure_dof_coords()
        mesh = self.mixed_element.mesh
        nodes_xy = np.asarray(mesh.nodes_x_y_pos, float)

        # quantized lookup for exact hits (robust to roundoff)
        def _q(x): return float(round(x, 12))
        node_lookup = {(_q(float(x)), _q(float(y))): int(i)
                    for i, (x, y) in enumerate(nodes_xy)}

        self.dof_map = {f: {} for f in self.field_names}
        self._dof_to_node_map = {}

        for f in self.field_names:
            ids = self.get_field_slice(f)
            for gd in ids:
                x, y = self._dof_coords[int(gd)]
                nid = node_lookup.get((_q(x), _q(y)))
                if nid is None:
                    dx = nodes_xy[:, 0] - x
                    dy = nodes_xy[:, 1] - y
                    j = int(np.argmin(dx*dx + dy*dy))
                    if (dx[j]*dx[j] + dy[j]*dy[j]) <= tol*tol:
                        nid = j
                if nid is not None:
                    self.dof_map[f][int(nid)] = int(gd)
                    self._dof_to_node_map[int(gd)] = (f, int(nid))
                else:
                    self._dof_to_node_map[int(gd)] = (f, None)



    


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
        max_field_q = max(me.q_orders.values())
        q_geom = mesh.poly_order
        q_good = 2 * max_field_q + 2 * (q_geom - 1)
        qdeg  = quad_order or q_good
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
        """Coordinates for every global DOF (total_dofs, 2)."""
        if getattr(self, "_dg_mode", False):
            raise NotImplementedError("get_all_dof_coords not yet implemented for DG.")
        if not hasattr(self, "_dof_coords"):
            raise RuntimeError("Geometry-independent coords not initialized.")
        return self._dof_coords.copy()
    
    def precompute_geometric_factors(
        self,
        quad_order: int,
        level_set: Callable = None,
        reuse: bool = True,
        need_hess: bool = False,
        need_o3: bool = False,
        need_o4: bool = False,
        deformation: Any = None
    ) -> Dict[str, np.ndarray]:
        """
        Precompute physical quadrature data (geometry, weights, optional level-set).
        Caches geometry per (mesh-id, n_elements, element_type, p, quad_order).

        Returns (per element):
        qp_phys (nE,nQ,2), qw (nE,nQ), detJ (nE,nQ), J_inv (nE,nQ,2,2),
        normals (nE,nQ,2), phis (nE,nQ or None), h_arr (nE,), eids (nE,),
        owner_id (nE,), entity_kind="element".
        If need_hess=True → also Hxi0/Hxi1 (nE,nQ,2,2) are returned.
        If need_o3=True   → also Txi0/Txi1 (nE,nQ,2,2,2).
        If need_o4=True   → also Qxi0/Qxi1 (nE,nQ,2,2,2,2).
        """
        import numpy as np
        from typing import Dict, Callable
        from pycutfem.fem import transform
        from pycutfem.integration.quadrature import volume as _volume_rule
        from pycutfem.fem.reference import get_reference as _get_ref

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

        # ---------- fast path: use cached geometry; add φ/Hxi on-demand ----------
        if reuse and deformation is None and geom_key in _volume_geom_cache:
            geo = _volume_geom_cache[geom_key]
            qp  = geo["qp_phys"]
            nE, nQ, _ = qp.shape

            # On-demand φ(xq) (not cached with geometry)
            phis = None
            if level_set is not None:
                phis = np.empty((nE, nQ), dtype=np.float64)
                eids = geo["eids"]
                for e in range(nE):
                    eid = int(eids[e])
                    for q in range(nQ):
                        phis[e, q] = phi_eval(level_set, qp[e, q], eid=eid, mesh=mesh)

            # On-demand inverse-map jets if requested (not cached; use qp_ref to avoid inverse mapping)
            if need_hess or need_o3 or need_o4:
                qp_ref = geo.get("qp_ref", None)
                if qp_ref is None:
                    qp_ref, _ = _volume_rule(mesh.element_type, quad_order)
                    qp_ref = np.asarray(qp_ref, dtype=np.float64)
                kmax = 4 if need_o4 else (3 if need_o3 else 2)
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

                eids = geo["eids"]
                for e in range(nE):
                    eid = int(eids[e])
                    for q in range(nQ):
                        xi, eta = float(qp_ref[q, 0]), float(qp_ref[q, 1])
                        rec = _JET.get(mesh, eid, xi, eta, upto=kmax)
                        if need_hess:
                            A2 = rec["A2"]
                            out["Hxi0"][e, q] = A2[0]
                            out["Hxi1"][e, q] = A2[1]
                        if need_o3:
                            A3 = rec["A3"]
                            out["Txi0"][e, q] = A3[0]
                            out["Txi1"][e, q] = A3[1]
                        if need_o4:
                            A4 = rec["A4"]
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
        ref = _get_ref(mesh.element_type, mesh.poly_order, max_deriv_order=kmax)
        # infer n_loc from a single evaluation (Ref has no n_functions accessor)
        n_loc = int(np.asarray(ref.shape(qp_ref[0, 0], qp_ref[0, 1])).size)
        Ntab  = np.empty((n_q, n_loc), dtype=np.float64)        # (nQ, n_loc)
        dNtab = np.empty((n_q, n_loc, 2), dtype=np.float64)     # (nQ, n_loc, 2)
        for q, (xi, eta) in enumerate(qp_ref):
            Ntab[q, :]     = np.asarray(ref.shape(xi, eta), dtype=np.float64).ravel()
            dNtab[q, :, :] = np.asarray(ref.grad (xi, eta), dtype=np.float64)

        # ---------------------- element → node coords (vectorized) ---------------
        # elem_coord[e,i,:] = (x_i, y_i) of the i-th local geometry node of element e
        elem_coord = mesh.nodes_x_y_pos[mesh.elements_connectivity].astype(np.float64)   # (nE, n_loc, 2)

        # ---------------------- allocate outputs ---------------------------------
        qp_phys = np.zeros((n_el, n_q, 2), dtype=np.float64)
        qw_sc   = np.zeros((n_el, n_q),    dtype=np.float64)
        detJ    = np.zeros((n_el, n_q),    dtype=np.float64)
        J_inv   = np.zeros((n_el, n_q, 2, 2), dtype=np.float64)

        # Keep geometry Jacobian for deformation step (not returned)
        J_geo   = np.zeros((n_el, n_q, 2, 2), dtype=np.float64)

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
                J_geo   = np.zeros((nE, nQ, 2, 2))
                for e in _nb.prange(nE):
                    for q in range(nQ):
                        # x = Σ_i N_i * X_i
                        x0 = 0.0; x1 = 0.0
                        a00 = 0.0; a01 = 0.0; a10 = 0.0; a11 = 0.0
                        for i in range(nLoc):
                            Ni  = Ntab[q, i]
                            dN0 = dNtab[q, i, 0]
                            dN1 = dNtab[q, i, 1]
                            X0  = coords[e, i, 0]
                            X1  = coords[e, i, 1]
                            x0 += Ni  * X0
                            x1 += Ni  * X1
                            a00 += dN0 * X0; a01 += dN0 * X1
                            a10 += dN1 * X0; a11 += dN1 * X1
                        qp_phys[e, q, 0] = x0
                        qp_phys[e, q, 1] = x1
                        J_geo[e, q, 0, 0] = a00; J_geo[e, q, 0, 1] = a01
                        J_geo[e, q, 1, 0] = a10; J_geo[e, q, 1, 1] = a11
                        det = a00 * a11 - a01 * a10
                        detJ[e, q] = det
                        inv_det = 1.0 / det
                        J_inv[e, q, 0, 0] =  a11 * inv_det
                        J_inv[e, q, 0, 1] = -a01 * inv_det
                        J_inv[e, q, 1, 0] = -a10 * inv_det
                        J_inv[e, q, 1, 1] =  a00 * inv_det
                        qw_sc[e, q] = qwref[q] * det
                return qp_phys, qw_sc, detJ, J_inv, J_geo

            qp_phys, qw_sc, detJ, J_inv, J_geo = _geom_kernel(elem_coord, Ntab, dNtab, qw_ref)

        else:
            # ------------------ safe Python fallback ------------------------------
            for e in range(n_el):
                Xe = elem_coord[e, :, :]            # (n_loc,2)
                for q_idx, (xi_eta, w_ref) in enumerate(zip(qp_ref, qw_ref)):
                    xi, eta = float(xi_eta[0]), float(xi_eta[1])
                    # Physical point
                    x = Xe.T @ Ntab[q_idx, :]       # (2,)
                    qp_phys[e, q_idx, :] = x
                    # Geometry Jacobian via dN @ X  (no inv(J_inv) nonsense)
                    dN = dNtab[q_idx, :, :]         # (n_loc,2)
                    A  = Xe.T @ dN                  # (2,2)  (row-form dN^T @ X)
                    J_geo[e, q_idx, :, :] = A
                    det = float(np.linalg.det(A))
                    if det <= 1e-12:
                        raise ValueError(f"Jacobian determinant is non-positive ({det}) for element {e}.")
                    detJ[e, q_idx]  = det
                    J_inv[e, q_idx] = np.linalg.inv(A)
                    qw_sc[e, q_idx] = w_ref * det

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
            "qp_ref":  qp_ref,
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

        # Cache *geometry only* (no φ, jets) for reuse in undeformed case
        if reuse and deformation is None:
            _volume_geom_cache[geom_key] = {k: v for k, v in geo.items()
                                            if k not in ("Hxi0","Hxi1","Txi0","Txi1","Qxi0","Qxi1","phis")}

        # ---------------------- deformation: update qp/detJ/J_inv/qw --------------
        if deformation is not None:
            # FE gradient of displacement in reference variables:
            # u(ξ) = Σ_i U_i N_i(ξ)  →  ∂ξ u = U^T @ dN(ξ)  (2×n) @ (n×2) => (2×2)
            qp_def   = np.empty_like(geo["qp_phys"])
            det_def  = np.empty_like(geo["detJ"])
            Jinv_def = np.empty_like(geo["J_inv"])

            for e in range(n_el):
                conn = np.asarray(mesh.elements_connectivity[int(e)], dtype=int)
                Uloc = np.asarray(deformation.node_displacements[conn], float)  # (n_loc,2)
                for q in range(n_q):
                    # geometry part already computed in J_geo
                    A_g = J_geo[e, q, :, :]               # J_g(ξq) (2×2)
                    dN  = dNtab[q, :, :]                   # (n_loc,2)
                    A_d = Uloc.T @ dN                      # (2×2)
                    A_t = A_g + A_d                        # total J
                    det = float(np.linalg.det(A_t))
                    det_def[e, q]  = det
                    Jinv_def[e, q] = np.linalg.inv(A_t)
                    # update point: x_def = x_g + u_h(ξq)
                    disp = Ntab[q, :] @ Uloc               # (2,)
                    qp_def[e, q, :] = geo["qp_phys"][e, q, :] + disp

            # replace geometry with deformed
            det_geo = geo["detJ"].copy()
            geo["qp_phys"] = qp_def
            geo["detJ"]    = det_def
            geo["J_inv"]   = Jinv_def
            # rescale weights so that qw = w_ref * det_total
            eps = 1e-300
            geo["qw"]      = geo["qw"] * (det_def / (det_geo + eps))

        # ---------------------- φ evaluation (at final qp) ------------------------
        phis = None
        if level_set is not None:
            phis = np.empty((n_el, n_q), dtype=np.float64)
            qp   = geo["qp_phys"]
            for e in range(n_el):
                eid = int(eids[e])
                for q in range(n_q):
                    phis[e, q] = phi_eval(level_set, qp[e, q], eid=eid, mesh=mesh)
        geo["phis"] = phis

        # Return geometry (+ jets if added) + φ
        return geo




    
    
    # -------------------------------------------------------------------------
    #  DofHandler.precompute_interface_factors
    # -------------------------------------------------------------------------
     
    def precompute_interface_factors(
        self,
        cut_element_ids,
        qdeg: int,
        level_set,
        nseg: int | None = None,
        reuse: bool = True,
        need_hess: bool = False,
        need_o3: bool = False,
        need_o4: bool = False,
        deformation: Any = None
    ) -> dict:
        """
        Pre-compute geometry & basis tables for ∫_{interface∩element} ⋯ dS on *cut* elements.
        Emits REFERENCE basis/derivative tables; push-forward (grad/Hess) is done in codegen.py.
        """
        import numpy as np
        from pycutfem.integration.quadrature import gauss_legendre, _map_line_rule_batched as _batched  # type: ignore
        from pycutfem.integration.quadrature import line_quadrature, curved_line_quadrature
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
                            a00 += gx * x; a01 += gy * x
                            a10 += gx * y; a11 += gy * y
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
        out = {}

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

        # --- Prepare segments; curved Γ quadrature ----------------------------------
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

        # choose a uniform nseg so nQ is constant across elements
        nseg_eff = int(nseg if nseg is not None else max(3, mesh.poly_order + qdeg//2))
        nQ = int(qdeg) * int(nseg_eff)
        qp_phys = np.empty((nE, nQ, 2), dtype=float)
        qw      = np.empty((nE, nQ),    dtype=float)
        for k in range(nE):
            pts, wts = curved_line_quadrature(level_set, P0[k], P1[k],
                                              order=qdeg, nseg=nseg_eff,
                                              project_steps=3, tol=SIDE.tol)
            # All elements produce the same nQ if nseg is uniform
            qp_phys[k, :, :] = pts.reshape(nQ, 2)
            qw[k, :]         = wts.reshape(nQ)

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

        # If deformation is provided: scale weights by ||(I + ∂u/∂x)·τ̂||
        if deformation is not None:
            from pycutfem.fem.reference import get_reference as _get_ref
            ref_geom = _get_ref(mesh.element_type, mesh.poly_order)
            # Build reference coordinates for each qp (element-wise inverse map)
            xi_tab  = np.empty((nE, nQ), dtype=float)
            eta_tab = np.empty((nE, nQ), dtype=float)
            for i in range(nE):
                eid = int(valid_cut_eids[i])
                for q in range(nQ):
                    s, t = transform.inverse_mapping(mesh, eid, qp_phys[i, q])
                    xi_tab[i, q]  = float(s); eta_tab[i, q] = float(t)
            # Apply stretch per qp
            for i in range(nE):
                eid = int(valid_cut_eids[i])
                conn = np.asarray(mesh.elements_connectivity[eid], dtype=int)
                Uloc = np.asarray(deformation.node_displacements[conn], float)  # (n_loc,2)
                for q in range(nQ):
                    xi = float(xi_tab[i, q]); eta = float(eta_tab[i, q])
                    Jg = transform.jacobian(mesh, eid, (xi, eta))
                    dN = np.asarray(ref_geom.grad(xi, eta), float)
                    Gref = Uloc.T @ dN
                    # G_phys = solve(Jg^T, Gref^T)^T
                    Gphy = np.linalg.solve(Jg.T, Gref.T).T
                    F = np.eye(2) + Gphy
                    n = normals[i, q]
                    tau = np.array([n[1], -n[0]], dtype=float)
                    qw[q] *= float(np.linalg.norm(F @ tau))
                    # update physical point to deformed location for ana.eval consistency
                    xg = transform.x_mapping(mesh, eid, (xi, eta))
                    disp = deformation.displacement_ref(eid, (xi, eta))
                    qp_phys[i, q, :] = xg + disp

        # --- (ξ,η) at each interface quadrature point -------------------------------
        xi_tab  = np.empty((nE, nQ), dtype=float)
        eta_tab = np.empty((nE, nQ), dtype=float)
        for k, eid in enumerate(valid_cut_eids):
            for q in range(nQ):
                s, t = transform.inverse_mapping(mesh, int(eid), qp_phys[k, q])  # fast path if p==1
                xi_tab[k, q]  = float(s)
                eta_tab[k, q] = float(t)

        # --- Geometry J, detJ, J_inv (use *geometry* p for mapping only) -----------
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
                        a00 += gx * x; a01 += gy * x
                        a10 += gx * y; a11 += gy * y
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

        # --- Basis & grad (per-field orders; ignore p_geom) ------------------------
        b_tabs = {f: np.zeros((nE, nQ, n_union), dtype=float) for f in fields}
        g_tabs = {f: np.zeros((nE, nQ, n_union, 2), dtype=float) for f in fields}
        xi_flat  = xi_tab.reshape(-1)
        eta_flat = eta_tab.reshape(-1)
        for fld in fields:
            sl = me.component_dof_slices[fld]
            # vectorized reference evals from MixedElement, then reshape back
            B = me._eval_scalar_basis_many(fld, xi_flat, eta_flat).reshape(nE, nQ, -1)
            G = me._eval_scalar_grad_many (fld, xi_flat, eta_flat).reshape(nE, nQ, -1, 2)
            b_tabs[fld][:, :, sl]    = B
            g_tabs[fld][:, :, sl, :] = G

        
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
        
        #--- union map for each field
        for fld in fields:
            # number of local dofs for this field on one element
            nloc_f = len(self.element_maps[fld][valid_cut_eids[0]])
            m = np.empty((nE, nloc_f), dtype=np.int32)
            for i, eid in enumerate(valid_cut_eids):
                union = gdofs_map[i, :n_union]        # union layout for this element
                col_of = {int(d): j for j, d in enumerate(union)}
                local_gdofs_f = self.element_maps[fld][eid]
                m[i, :] = [col_of[int(d)] for d in local_gdofs_f]
            out[f"pos_map_{fld}"] = m
            out[f"neg_map_{fld}"] = m  # same element both "sides" on interface

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
        """
        Pre-compute geometry and basis tables for ∫_Γ_g ⋯ dS on ghost edges.

        Parameters
        ----------
        ghost_edge_ids : BitSet | Sequence[int]
            Set of ghost edge ids to include (BitSet preferred).
        qdeg : int
            Line quadrature degree along each ghost edge.
        level_set : object
            Level-set used to determine +/- owners and orientation where needed.
        derivs : set[tuple[int, int]]
            Set of derivative multi-indices (dx, dy) to tabulate for basis functions.
        reuse : bool, default True
            Reuse cached factors for identical (edge subset, qdeg, field orders, derivs, flags).
        need_hess : bool, default False
            Also pre-tabulate all second derivatives (enables Hessian-based terms).
        need_o3 : bool, default False
            Also pre-tabulate all third derivatives (rare).
        need_o4 : bool, default False
            Also pre-tabulate all fourth derivatives (rare).

        Returns
        -------
        dict
            A dictionary with:
            • geometry:
                'edge_ids'       : (nE,)        int32
                'qp_phys'        : (nE, nQ, 2)  float64   physical quadrature points
                'w'              : (nE, nQ)     float64   physical edge weights
                'n_unit'         : (nE, 2)      float64   unit normals (owner convention)
                'J_pos','J_neg'  : (nE, nQ, 2, 2) float64 Jacobians per side
                'detJ_pos','detJ_neg' : (nE, nQ) float64
                (plus optional inverse/jet entries when higher orders requested)
            • union maps and per-field scatter maps:
                'union_map'      : (nE, n_union) int32    union of side DOFs per edge
                'pos_map_{fld}'  : (nE, n_loc_f) int32    side-local → union indices
                'neg_map_{fld}'  : (nE, n_loc_f) int32
            • basis tables per field and side (keys like 'r00_{fld}_pos', 'r10_{fld}_neg', ...):
                Each of shape (nE, nQ, n_union) after scattering to the union layout.

        Notes
        -----
        • The output is designed to be consumed directly by the ghost-edge assembler
        without recomputing mapping, normals, or basis derivatives.
        • Results are cached by a stable key unless `reuse=False`.
        """
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
                            a00 += gx * x; a01 += gy * x
                            a10 += gx * y; a11 += gy * y
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
        out = {}

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

            if SIDE.is_pos(phiL) and not SIDE.is_pos(phiR):
                pos_eid, neg_eid = e.left, e.right
            elif SIDE.is_pos(phiR) and not SIDE.is_pos(phiL):
                pos_eid, neg_eid = e.right, e.left
            else:
                # tie-breaker (both same side within tol): larger φ defines '+'
                pos_eid, neg_eid = (e.left, e.right) if phiL >= phiR else (e.right, e.left)

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
        
        # 3b --- per-field, side-local -> union maps (ghost)
        for fld in fields:
            nloc_f = len(self.element_maps[fld][pos_ids[0]])
            pm = np.empty((nE, nloc_f), dtype=np.int32)
            nm = np.empty((nE, nloc_f), dtype=np.int32)
            for i in range(nE):
                pos_eid = int(pos_ids[i]); neg_eid = int(neg_ids[i])
                # union for this edge
                union = gdofs_map[i, :n_union]
                col_of = {int(d): j for j, d in enumerate(union)}
                # side-local gdofs for this field
                pos_loc = self.element_maps[fld][pos_eid]
                neg_loc = self.element_maps[fld][neg_eid]
                pm[i, :] = [col_of[int(d)] for d in pos_loc]
                nm[i, :] = [col_of[int(d)] for d in neg_loc]
            out[f"pos_map_{fld}"] = pm
            out[f"neg_map_{fld}"] = nm


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

        out.update({
            "xi_pos":  xi_pos,  "eta_pos":  eta_pos,
            "xi_neg":  xi_neg,  "eta_neg":  eta_neg,
        })

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
                            a00 += gx * x; a01 += gy * x
                            a10 += gx * y; a11 += gy * y
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
            arr = np.empty((nE, nQ, n_f))
            _tab(0, 0, xi_pos, eta_pos, arr)   # same _tab(...) you use for r10 etc.
            basis_tables[f"r00_{fld}_pos"] = _scatter_union(arr, sl, final_w)

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
            # NEG side value (r00)
            arr = np.empty((nE, nQ, n_f))
            _tab(0, 0, xi_neg, eta_neg, arr)
            basis_tables[f"r00_{fld}_neg"] = _scatter_union(arr, sl, final_w)
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
                            a00 += gx * x; a01 += gy * x
                            a10 += gx * y; a11 += gy * y
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
                        a00 += gx * x; a01 += gy * x
                        a10 += gx * y; a11 += gy * y
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
        nseg_hint: int | None = None,
        deformation: Any = None,
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
            curved_subcell_quadrature_for_cut_triangle
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
                "is_interface": False,
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
        ls_token = _ls_fingerprint(level_set)
        derivs_key = tuple(sorted((int(dx), int(dy)) for (dx, dy) in derivs))
        def _def_fingerprint(defm):
            if defm is None:
                return ("nodef",)
            tok = getattr(defm, "cache_token", None)
            if tok is not None:
                return ("token", tok)
            # last resort: object identity (works within one process)
            return ("objid", int(id(defm)))

        def_token = _def_fingerprint(deformation)

        cache_key = (
            "cutvol", id(mesh), me.signature(), subset_hash,
            int(qdeg), str(side), derivs_key, 
            self.method, bool(need_hess), 
            bool(need_o3), bool(need_o4), ls_token,
            def_token
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
        from pycutfem.fem.reference import get_reference as _get_ref
        ref_geom = _get_ref(mesh.element_type, mesh.poly_order)

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
                # NEW: curved subcell quadrature on this corner-triangle
                qx, qw = curved_subcell_quadrature_for_cut_triangle(
                    mesh, int(eid), loc_tri, cn, level_set,
                    side=('+' if sgn == +1 else '-'),
                    qvol=int(qdeg), nseg_hint=nseg_hint, tol=1e-12
                )
                if qx.size == 0:
                    continue
                for x, w in zip(qx, qw):
                    s, t  = transform.inverse_mapping(mesh, int(eid), x)
                    rec   = _JET.get(mesh, int(eid), float(s), float(t), upto=kmax_jets)
                    Ji_geo = rec["A"]                          # (2,2) = J_geom^{-1}
                    Jg     = np.linalg.inv(Ji_geo)

                    # default: geometric mapping
                    Ji_use = Ji_geo
                    w_eff  = float(w)
                    x_use  = np.asarray(x, dtype=float)

                    if deformation is not None:
                        # Build deformation Jacobian in reference → physical
                        conn = np.asarray(mesh.elements_connectivity[int(eid)], dtype=int)
                        Uloc = np.asarray(deformation.node_displacements[conn], float)  # (nloc,2)
                        dN   = np.asarray(ref_geom.grad(float(s), float(t)), float)     # (nloc,2)
                        Jd   = Uloc.T @ dN
                        Jt   = Jg + Jd
                        Ji_use = np.linalg.inv(Jt)
                        det_g = abs(float(np.linalg.det(Jg))) + 1e-300
                        det_t = abs(float(np.linalg.det(Jt)))
                        w_eff = float(w) * (det_t / det_g)
                        # update physical point to deformed y = x + u
                        disp = deformation.displacement_ref(int(eid), (float(s), float(t)))
                        x_use = x_use + disp

                    # store per-qp geometry
                    xq_elem.append(x_use)
                    wq_elem.append(w_eff)
                    Jinv_elem.append(Ji_use)
                    phi_elem.append(float(phi_eval(level_set, x)))
                    # inverse-jet chains if requested (unchanged)
                    if need_hess:
                        A2 = rec["A2"]; H0_elem.append(np.asarray(A2[0], float)); H1_elem.append(np.asarray(A2[1], float))
                    if need_o3:
                        A3 = rec["A3"]; T0_elem.append(np.asarray(A3[0], float)); T1_elem.append(np.asarray(A3[1], float))
                    if need_o4:
                        A4 = rec["A4"]; Q0_elem.append(np.asarray(A4[0], float)); Q1_elem.append(np.asarray(A4[1], float))
                    # per-field reference basis/derivs at (s,t) — still per-field order
                    for fld in fields:
                        basis_lists.setdefault(f"b_{fld}", []).append(np.asarray(me._eval_scalar_basis(fld, float(s), float(t)), float))
                        basis_lists.setdefault(f"g_{fld}", []).append(np.asarray(me._eval_scalar_grad (fld, float(s), float(t)), float))
                        for (dx, dy) in derivs_eff:
                            if (dx, dy) == (0, 0): continue
                            basis_lists.setdefault(f"d{dx}{dy}_{fld}", []).append(
                                np.asarray(me._eval_scalar_deriv(fld, float(s), float(t), int(dx), int(dy)), float)
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
                "is_interface": False,
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
            "is_interface": False,
            "J_inv_pos": J_inv,   # for compatibility with interface kernels
            "J_inv_neg": J_inv,   # ditto
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
        elem_sel,                 # BitSet | str | 1D bool mask | Iterable[int] | int
        *,
        strict: bool = True,
    ) -> set[int]:
        """
        Tag DOFs associated with a set of elements, storing them under `self.dof_tags[tag]`.

        Parameters
        ----------
        tag : str
            Name of the tag set to create/update in `self.dof_tags`.
        field : str
            Field whose DOFs will be tagged.
        elem_sel : BitSet | str | 1D bool mask | Iterable[int] | int
            Element selection:
            • BitSet (preferred): with `.to_indices()` or `.array`
            • str: element bitset name on the mesh (e.g. "inside", "outside", "cut")
            • 1D bool mask of length n_elements
            • Iterable of element ids (list/tuple/set/ndarray)
            • Single element id (int)
        strict : bool, default True
            CG only:
            • strict=True  -> tag DOFs whose *entire* support lies inside `elem_sel`
            • strict=False -> tag DOFs that touch `elem_sel` (any adjacency hit)

        Returns
        -------
        set[int]
            The set of global DOF ids that were tagged.

        Notes
        -----
        • DG: All DOFs from the selected elements are tagged (element-local numbering).
        • CG: Uses a cached DOF↔elements adjacency (built on first use per field).
        • Mixed orders are supported; adjacency is built from `element_maps[field]`.
        """
        from pycutfem.ufl.helpers import normalize_elem_ids

        if field not in self.field_names:
            raise ValueError(f"Unknown field '{field}'.")

        mesh = self.mixed_element.mesh

        elem_ids = normalize_elem_ids(self.mixed_element.mesh, elem_sel) or []
        inside_set = set(int(e) for e in elem_ids)

        # --- DG path: tag every DOF from selected elements ------------------------
        method = getattr(self, "method", "cg").lower()
        if method == "dg" or getattr(self, "_dg_mode", False):
            selected: set[int] = set()
            for eid in elem_ids:
                selected.update(int(g) for g in self.element_maps[field][int(eid)])
            self.dof_tags.setdefault(tag, set()).update(selected)
            return selected

        # --- CG path: build DOF↔elements adjacency (once per field) ---------------
        if not hasattr(self, "_cg_adj"):
            self._cg_adj = {}
        d2e = self._cg_adj.get(field)
        if d2e is None:
            d2e = {}
            for eid, gds in enumerate(self.element_maps[field]):
                for g in gds:
                    d2e.setdefault(int(g), set()).add(int(eid))
            self._cg_adj[field] = d2e

        # Select DOFs by adjacency
        if strict:
            selected = {gd for gd, adj in d2e.items() if adj.issubset(inside_set)}
        else:
            selected = {gd for gd, adj in d2e.items() if (adj & inside_set)}

        self.dof_tags.setdefault(tag, set()).update(selected)
        return selected





    def l2_error_on_side(
        self,
        functions,
        exact,
        level_set,
        side: str = "-",
        quad_order: int | None = None,
        fields: list[str] | None = None,
        relative: bool = False,
    ) -> float:
        """
        L2 error of the given fields over Ω_side := Ω ∩ { φ ▷ 0 }, with ▷='<' if side='-', or '>=' if side='+'.

        Accepts:
        - a single Function,
        - a VectorFunction,
        - a dict[str, Function],
        - or a list/tuple mixing the above.

        Robustness notes:
        • Uses field-local basis evaluation for the dot product (length matches local DOFs).
        • Falls back to a mixed-basis + scatter if shapes ever disagree.
        • Works for both full and cut elements; cut integration uses physical weights.
        """
        from pycutfem.fem import transform
        from pycutfem.integration.quadrature import volume as vol_rule
        from pycutfem.ufl.expressions import Function as UFLFunction, VectorFunction as UFLVectorFunction
        from pycutfem.ufl.helpers_geom import (
            corner_tris, clip_triangle_to_side, fan_triangulate, map_ref_tri_to_phys, phi_eval
        )

        mesh = self.mixed_element.mesh
        me   = self.mixed_element
        max_field_q = max(me.q_orders.values())
        q_geom = mesh.poly_order
        q_good = 2 * max_field_q + 2 * (q_geom - 1)
        qdeg  = quad_order or q_good

        # ---------------------- normalize 'functions' -> dict[field] = Function ----------------------
        field_funcs: dict[str, UFLFunction] = {}

        def _add_vec(vecfun: UFLVectorFunction):
            for name, comp in zip(vecfun.field_names, vecfun.components):
                field_funcs[name] = comp

        if isinstance(functions, dict):
            field_funcs.update(functions)
        elif isinstance(functions, UFLVectorFunction):
            _add_vec(functions)
        elif isinstance(functions, (list, tuple)):
            for obj in functions:
                if isinstance(obj, UFLVectorFunction):
                    _add_vec(obj)
                elif isinstance(obj, UFLFunction):
                    field_funcs[obj.field_name] = obj
                elif isinstance(obj, dict):
                    field_funcs.update(obj)
                else:
                    raise TypeError("Unsupported item in 'functions' list/tuple.")
        elif isinstance(functions, UFLFunction):
            field_funcs[functions.field_name] = functions
        else:
            raise TypeError("'functions' must be a Function, VectorFunction, dict[str, Function], or list/tuple of those.")

        # Which fields to include
        if fields is None:
            # Only fields for which we have both a function and an exact callable
            fields = [f for f in me.field_names if f in field_funcs and f in exact]
        else:
            # Keep only those we can actually evaluate
            fields = [f for f in fields if f in field_funcs and f in exact]
        if not fields:
            return 0.0

        # Accumulators
        err2, base2 = 0.0, 0.0

        # ---------------------------- classify elements ----------------------------
        inside_ids, outside_ids, cut_ids = mesh.classify_elements(level_set)
        full_eids = outside_ids if side == "+" else inside_ids

        # Helper: evaluate uh safely (field-local basis preferred; mixed fallback)
        def _eval_uh(fld: str, eid: int, xi: float, eta: float, gdofs: np.ndarray) -> float:
            vals = np.asarray(field_funcs[fld].get_nodal_values(gdofs), dtype=float)
            # Try field-local basis first
            try:
                phi_f = np.asarray(me._eval_scalar_basis(fld, xi, eta), dtype=float)  # length n_loc(field)
            except AttributeError:
                phi_f = None

            if phi_f is not None and phi_f.shape[0] == vals.shape[0]:
                return float(vals @ phi_f)

            # Fallback: build mixed basis and scatter vals into mixed positions, then dot.
            phi_mixed = np.asarray(me.basis(fld, xi, eta), dtype=float)
            sl = me.slice(fld)  # indices for this field in the mixed ordering
            v_mixed = np.zeros_like(phi_mixed, dtype=float)
            # Guard for shape mismatch (robustness)
            n = min(len(sl), vals.shape[0])
            v_mixed[sl[:n]] = vals[:n]
            return float(v_mixed @ phi_mixed)

        # -------------------------- (A) full elements on side ----------------------
        qp_ref, qw_ref = vol_rule(mesh.element_type, qdeg)
        for eid in full_eids:
            # per-field local DOF ids for this element
            gdofs_f = {fld: self.element_maps[fld][eid] for fld in fields}
            for (xi, eta), w in zip(qp_ref, qw_ref):
                J    = transform.jacobian(mesh, eid, (xi, eta))
                detJ = abs(np.linalg.det(J))
                x, y = transform.x_mapping(mesh, eid, (xi, eta))
                for fld in fields:
                    uh = _eval_uh(fld, eid, xi, eta, gdofs_f[fld])
                    ue = float(exact[fld](x, y))
                    d2 = (uh - ue) ** 2
                    b2 = ue ** 2
                    err2  += w * detJ * d2
                    base2 += w * detJ * b2

        # -------------------------- (B) cut elements on side -----------------------
        # Reference rule on unit triangle (mapped to physical with weights)
        qp_tri, qw_tri = vol_rule("tri", qdeg)

        for eid in cut_ids:
            elem = mesh.elements_list[eid]
            tri_local, corner_ids = corner_tris(mesh, elem)

            # per-field local DOF ids for this element
            gd_f = {fld: self.element_maps[fld][eid] for fld in fields}

            for loc_tri in tri_local:
                v_ids = [corner_ids[i] for i in loc_tri]
                V     = mesh.nodes_x_y_pos[v_ids]                       # (3,2)
                v_phi = np.array([phi_eval(level_set, xy) for xy in V], dtype=float)

                # Clip this triangle to requested side
                polys = clip_triangle_to_side(V, v_phi, side=side)
                if not polys:
                    continue

                for poly in polys:
                    for A, B, C in fan_triangulate(poly):
                        x_phys, w_phys = map_ref_tri_to_phys(A, B, C, qp_tri, qw_tri)  # physical points & weights
                        for (x, y), w in zip(x_phys, w_phys):
                            # Map back to parent element reference coords
                            xi, eta = transform.inverse_mapping(mesh, eid, np.array([x, y]))
                            for fld in fields:
                                uh = _eval_uh(fld, eid, xi, eta, gd_f[fld])
                                ue = float(exact[fld](x, y))
                                d2 = (uh - ue) ** 2
                                b2 = ue ** 2
                                err2  += w * d2           # w already physical here
                                base2 += w * b2

        # Return relative or absolute L2
        if relative:
            return (err2 / base2) ** 0.5 if base2 > 1e-28 else 0.0
        return err2 ** 0.5


    
    def l2_error_piecewise(self, functions, exact_neg, exact_pos, level_set, fields=None, quad_order=None, relative=False):
        """
        Piecewise L² error across a two-phase split defined by a level set.

        Parameters
        ----------
        functions : Function | VectorFunction | Sequence
            Discrete function(s) carrying the current solution (or a subset by field).
        exact_neg, exact_pos : Mapping[str, Callable[[float, float], float]]
            Exact field-by-field evaluators on the '-' and '+' subdomains.
        level_set : object
            Level-set providing the phase split and interface location.
        fields : list[str] | None, default None
            If given, restrict the error computation to these fields.
        quad_order : int | None, default None
            Volume quadrature order used on each side; defaults to a safe value
            tied to the field order if None.
        relative : bool, default False
            If True, returns a relative L² error (normalized by exact L² norm).

        Returns
        -------
        float
            The combined piecewise error: sqrt(‖e‖²_{Ω-} + ‖e‖²_{Ω+}).
        """
        e_m = self.l2_error_on_side(functions, exact_neg, level_set, side='-', fields=fields, quad_order=quad_order, relative=relative)
        e_p = self.l2_error_on_side(functions, exact_pos, level_set, side='+', fields=fields, quad_order=quad_order, relative=relative)
        return (e_m**2 + e_p**2)**0.5

    
    # --------------------------------------------------------------------------
    # H1 seminorm (energy) of a VectorFunction on a *side* (φ<0 or φ>0)
    # --------------------------------------------------------------------------
    def _h1_seminorm_on_side(self,
                            vector_function=None,                 # VectorFunction for seminorm mode
                            level_set=None,                       # φ(x,y)
                            side: str = '-',                      # '-' or '+'
                            quad_order: int | None = None,
                            fields: list[str] | None = None,
                            *,
                            function=None,                        # Scalar Function or VectorFunction (error mode)
                            field: str | None = None,
                            exact_grad=None,                      # callable (x,y)-> [du/dx, du/dy]
                            quad_increase: int = 0):
        """
        Dual-mode helper:
        (A) Seminorm mode (when `function is None` and `exact_grad is None`):
            Returns ||∇(vector_function)||_{L2(Ω_side)}.

        (B) Error mode (when `function` AND `field` AND `exact_grad` are given):
            Returns (err2, ref2) with
                err2 = ||∇u_h - ∇u_exact||_{L2(Ω_side)}^2,
                ref2 = ||∇u_exact||_{L2(Ω_side)}^2.
        """
        from pycutfem.integration.quadrature import volume as vol_rule
        from pycutfem.ufl.helpers_geom import (
            corner_tris, clip_triangle_to_side, fan_triangulate, map_ref_tri_to_phys, phi_eval
        )
        from pycutfem.ufl.expressions import VectorFunction as _VF, Function as Function

        mesh = self.mixed_element.mesh
        me   = self.mixed_element
        max_field_q = max(me.q_orders.values())
        q_geom = mesh.poly_order
        q_good = 2 * (max_field_q - 1) + 2 * (q_geom - 1) + int(quad_increase)
        qdeg  = quad_order or q_good
        qdeg = max(qdeg, 2)  # at least linear for gradients

        # classify once
        inside_ids, outside_ids, cut_ids = mesh.classify_elements(level_set)
        full_eids = outside_ids if side == '+' else inside_ids

        # local gradient (physical) at (xi,eta) for a scalar field `fld`
        def _grad_at(eid, xi, eta, fld, values):
            # reference gradients of *this field* only
            G_ref_full = me.grad_basis(fld, xi, eta)                    # (n_loc_total, 2)
            loc = me.component_dof_slices[fld]                          # slice for this field
            G_ref = G_ref_full[loc, :]                                  # (n_f, 2)
            # physical mapping: ∇_x φ = J^{-T} ∇̂φ  (row-by-row)
            J = transform.jacobian(mesh, eid, (xi, eta))               # (2,2)
            Jinv = np.linalg.inv(J)           # (2,2) = J
            G_phy = G_ref @ Jinv                                       # (n_f, 2)
            return values @ G_phy                                       # (2,)

        # -------- seminorm mode --------
        if function is None and exact_grad is None:
            if vector_function is None:
                raise ValueError("seminorm mode requires 'vector_function'.")
            if fields is None:
                fields = list(vector_function.field_names)

            acc = 0.0
            # (A) full elements
            qp_ref, qw_ref = vol_rule(mesh.element_type, qdeg)
            for eid in full_eids:
                gd_f = {fld: self.element_maps[fld][eid] for fld in fields}
                for (xi, eta), w_ref in zip(qp_ref, qw_ref):
                    J = transform.jacobian(mesh, eid, (xi, eta))
                    w = float(w_ref * abs(np.linalg.det(J)))
                    for fld in fields:
                        vals = vector_function.get_nodal_values(gd_f[fld])
                        gu = _grad_at(eid, xi, eta, fld, vals)
                        acc += w * (gu[0]**2 + gu[1]**2)

            # (B) cut elements
            if len(cut_ids):
                # one-point ref rule on unit triangle; mapped by fan-triangulation
                qp_tri = np.array([[1.0/3.0, 1.0/3.0]])
                qw_tri = np.array([0.5])
                for eid in cut_ids:
                    elem = mesh.elements_list[eid]
                    tri_list, cn = corner_tris(mesh, elem)              # ([(i,j,k)], corner node ids)
                    gd_f = {fld: self.element_maps[fld][eid] for fld in fields}
                    for (i, j, k) in tri_list:
                        tri_xy = mesh.nodes_x_y_pos[[cn[i], cn[j], cn[k]]]   # (3,2)
                        v_phi  = [phi_eval(level_set, tri_xy[r]) for r in range(3)]
                        for poly in clip_triangle_to_side(tri_xy, v_phi, side=side):
                            for A, B, C in fan_triangulate(poly):
                                x_phys, w_phys = map_ref_tri_to_phys(A, B, C, qp_tri, qw_tri)
                                for (x, y), w in zip(x_phys, w_phys):
                                    xi, eta = transform.inverse_mapping(mesh, eid, np.array([x, y]))
                                    for fld in fields:
                                        vals = vector_function.get_nodal_values(gd_f[fld])
                                        gu = _grad_at(eid, xi, eta, fld, vals)
                                        acc += w * (gu[0]**2 + gu[1]**2)
            return float(acc) ** 0.5

        # -------- error mode --------
        if function is None or field is None or exact_grad is None:
            raise ValueError("error mode requires 'function', 'field', and 'exact_grad'.")

        # Both Function and VectorFunction expose .get_nodal_values(gd)
        if not isinstance(function, (_VF, Function)):
            raise TypeError("Unsupported 'function' type for H1 error.")
        get_vals = function.get_nodal_values

        err2 = 0.0
        ref2 = 0.0

        # (A) full elements
        qp_ref, qw_ref = vol_rule(mesh.element_type, qdeg)
        for eid in full_eids:
            gd = self.element_maps[field][eid]
            for (xi, eta), w_ref in zip(qp_ref, qw_ref):
                J = transform.jacobian(mesh, eid, (xi, eta))
                w = float(w_ref * abs(np.linalg.det(J)))
                vals = get_vals(gd)
                gu = _grad_at(eid, xi, eta, field, vals)
                x, y = transform.x_mapping(mesh, eid, (xi, eta))
                ge = np.asarray(exact_grad(x, y), dtype=float)
                de = gu - ge
                err2 += w * float(de @ de)
                ref2 += w * float(ge @ ge)

        # (B) cut elements
        if len(cut_ids):
            qp_tri = np.array([[1.0/3.0, 1.0/3.0]])
            qw_tri = np.array([0.5])
            for eid in cut_ids:
                elem = mesh.elements_list[eid]
                tri_list, cn = corner_tris(mesh, elem)
                gd = self.element_maps[field][eid]
                for (i, j, k) in tri_list:
                    tri_xy = mesh.nodes_x_y_pos[[cn[i], cn[j], cn[k]]]   # (3,2)
                    v_phi  = [phi_eval(level_set, tri_xy[r]) for r in range(3)]
                    for poly in clip_triangle_to_side(tri_xy, v_phi, side=side):
                        for A, B, C in fan_triangulate(poly):
                            x_phys, w_phys = map_ref_tri_to_phys(A, B, C, qp_tri, qw_tri)
                            for (x, y), w in zip(x_phys, w_phys):
                                xi, eta = transform.inverse_mapping(mesh, eid, np.array([x, y]))
                                vals = get_vals(gd)
                                gu = _grad_at(eid, xi, eta, field, vals)
                                ge = np.asarray(exact_grad(x, y), dtype=float)
                                de = gu - ge
                                err2 += w * float(de @ de)
                                ref2 += w * float(ge @ ge)

        return float(err2), float(ref2)




    # --------------------------------------------------------------------------
    # Build w := u_h - I_h(u_exact) *at the nodes* (piecewise by φ sign)
    # --------------------------------------------------------------------------
    def _build_vector_diff_from_exact(self,
                                    u_h,                            # VectorFunction ('ux','uy')
                                    exact_neg: dict[str, callable], # {'ux': f(x,y), 'uy': g(x,y)} for φ<0
                                    exact_pos: dict[str, callable], # same for φ>0
                                    level_set) -> 'VectorFunction':
        """
        Returns a new VectorFunction w with nodal values w = u_h - u_exact_interp,
        where u_exact_interp is the FE nodal interpolation of piecewise exact values:
        node in φ<0 → use exact_neg; node in φ>0 → use exact_pos.
        """
        from pycutfem.ufl.expressions import VectorFunction as _VF, Function as Function

        if not hasattr(u_h, "field_names"):
            raise TypeError("_build_vector_diff_from_exact expects a VectorFunction.")

        w = u_h.copy()     # same dof layout; we'll overwrite nodal_values

        # For each scalar component, evaluate piecewise exact at that field's DOF coords
        for i, fld in enumerate(u_h.field_names):
            g_slice = self.get_field_slice(fld)
            coords  = self.get_dof_coords(fld)         # (n_field_dofs, 2)
            values  = np.empty(coords.shape[0], dtype=float)
            for k, (x, y) in enumerate(coords):
                phi = float(level_set([x, y]))
                values[k] = exact_neg[fld](x, y) if SIDE.is_neg(phi) else exact_pos[fld](x, y)


            # subtract into copy: w = u_h - I_h(u_exact)
            # write component-wise using vector function's global dofs
            w.set_nodal_values(g_slice, u_h.get_nodal_values(g_slice) - values)

        return w


    # --------------------------------------------------------------------------
    # Public: H1 error (piecewise) via FE interpolation of exact data
    # --------------------------------------------------------------------------
    def h1_error_vector_piecewise(self,
                                u_h,                               # VectorFunction ('ux','uy')
                                exact_neg: dict[str, callable],
                                exact_pos: dict[str, callable],
                                level_set,
                                quad_order: int | None = None,
                                fields: list[str] | None = None) -> float:
        """
        Returns the FE H1-seminorm error ||∇(u_h - I_h u_exact)|| over Ω^- ∪ Ω^+,
        computed as ( ||∇w||_{Ω^-}^2 + ||∇w||_{Ω^+}^2 )^{1/2} with w := u_h - I_h u_exact.

        No finite differences. No exact gradients required.
        """
        w = self._build_vector_diff_from_exact(u_h, exact_neg, exact_pos, level_set)
        e_m = self._h1_seminorm_on_side(w, level_set, side='-', quad_order=quad_order, fields=fields)
        e_p = self._h1_seminorm_on_side(w, level_set, side='+', quad_order=quad_order, fields=fields)
        return (e_m**2 + e_p**2)**0.5
    
    def h1_error_scalar_on_side(self, uh, exact_grad, level_set, side, field=None, relative=False, quad_increase=0):
        """
        H1-seminorm error on a side: ||∇u_h - ∇u_exact||_{L2(Ω_side)} for a scalar Function.
        - uh:        Function
        - exact_grad: callable (x, y) -> array_like(..., 2)  giving [du/dx, du/dy]
        - level_set: LevelSet
        - side:      '+' or '-'
        - field:     override field name; defaults to uh.field_name
        - relative:  if True, divides by ||∇u_exact||_{L2(Ω_side)}
        """
        fld = field or uh.field_name
        err2, ref2 = self._h1_seminorm_on_side(
            function=uh, field=fld, exact_grad=exact_grad,
            level_set=level_set, side=side, quad_increase=quad_increase
        )
        return (err2**0.5) if not relative else ((err2 / max(ref2, 1e-30))**0.5)

    def h1_error_vector_on_side(self, Uh, exact_grad_vec, level_set, side, fields=None, relative=False, quad_increase=0):
        """
        H1-seminorm error for a 2D vector field on a side:
        ||∇Uh - ∇U_exact||_{L2(Ω_side)} where ∇Uh and ∇U_exact are 2×2.
        - Uh: VectorFunction
        - exact_grad_vec: callable (x,y)-> array_like(..., 2,2)
                        rows = component, cols = derivative (du_i/dx, du_i/dy)
        - fields: order must match Uh.field_names if provided.
        """
        flds = fields or Uh.field_names
        err2_tot = 0.0
        ref2_tot = 0.0
        # compute per-component gradients and sum
        for ic, fld in enumerate(flds):
            def exact_grad_comp(x, y, ic=ic):
                G = exact_grad_vec(x, y)         # (..., 2, 2)
                # return [d u_ic / dx, d u_ic / dy]
                return G[..., ic, :]             # (..., 2)
            err2, ref2 = self._h1_seminorm_on_side(
                function=Uh, field=fld, exact_grad=exact_grad_comp,
                level_set=level_set, side=side, quad_increase=quad_increase
            )
            err2_tot += err2; ref2_tot += ref2
        return (err2_tot**0.5) if not relative else ((err2_tot / max(ref2_tot, 1e-30))**0.5)
    def reduce_linear_system(self, A_full, R_full, *, bcs=None, return_dirichlet=False):
        """
        Reduce (K, F) to the free-DOF block with proper lifting:
            [K_ff K_fd][u_f] = [F_f]  ->  K_ff u_f = (F_f - K_fd u_D)
            [K_df K_dd][u_D]   [F_D]

        Args
        ----
        A_full : (n x n) sparse matrix (scipy)
        R_full : (n,) dense rhs
        bcs    : Dirichlet BC list you pass to assemble_form (or None)
        return_dirichlet : if True, also returns (dir_idx, u_dirichlet)

        Returns
        -------
        K_ff : sparse
        F_f  : ndarray  (lifted RHS = F_f - K_fd u_D)
        free : (nf,) free dof ids
        full_to_red : (n,) map full index -> reduced index, -1 for Dirichlet
        [dir_idx, u_dir] : (optional) Dirichlet ids and values (length ndof array)
        """
        import numpy as np
        from scipy.sparse import csr_matrix

        if A_full.shape[0] != A_full.shape[1]:
            raise ValueError("A_full must be square.")
        if R_full.shape[0] != A_full.shape[0]:
            raise ValueError("R_full length must match K size.")

        ndof = self.total_dofs
        A = A_full.tocsr()
        F = np.asarray(R_full, dtype=float)

        # 1) Collect Dirichlet data and build u_D
        bc_map = self.get_dirichlet_data(bcs or [])
        dir_idx = np.array(sorted(bc_map.keys()), dtype=int)
        u_dir = np.zeros(ndof, dtype=float)
        if dir_idx.size:
            u_dir[dir_idx] = np.array([bc_map[i] for i in dir_idx], dtype=float)

        # 2) Free set
        all_idx = np.arange(ndof, dtype=int)
        free = np.setdiff1d(all_idx, dir_idx, assume_unique=True)

        # 3) Build reduced matrices and apply lifting on RHS
        K_ff = A[np.ix_(free, free)].copy()
        F_f  = F[free].copy()
        if dir_idx.size:
            K_fd = A[np.ix_(free, dir_idx)]
            F_f -= K_fd @ u_dir[dir_idx]      # <--- lifting

        # 4) Map full -> reduced
        full_to_red = -np.ones(ndof, dtype=int)
        full_to_red[free] = np.arange(free.size, dtype=int)

        if return_dirichlet:
            return K_ff, F_f, free, dir_idx, u_dir, full_to_red
        return K_ff, F_f, free, full_to_red
    
    


    
    def assemble_pressure_mean_vector(self, level_set, quad_order=None,
                                  p_pos_field='p_pos_', p_neg_field='p_neg_', backend= 'python'):
        """
        Build r in R^{ndof} with:
        r_j = ∫_{Ω+} φ_j dx  for p_pos_ dofs,
        r_j = ∫_{Ω-} φ_j dx  for p_neg_ dofs,
        r_j = 0 otherwise.

        Uses the normal assembler (TestFunction on dx-side-restricted domains),
        so it’s robust on cut cells and mixed orders.
        """
        from pycutfem.ufl.expressions import TestFunction, Constant
        from pycutfem.ufl.measures import dx
        from pycutfem.ufl.forms import Equation, assemble_form

        mesh = self.mixed_element.mesh
        # Ensure element tags are current
        mesh.classify_elements(level_set)

        has_inside  = mesh.element_bitset("inside")  | mesh.element_bitset("cut")
        has_outside = mesh.element_bitset("outside") | mesh.element_bitset("cut")
        qdeg = quad_order or (2 * mesh.poly_order + 2)

        # Measures for each side (your compiler already uses these for cut volume)
        dx_pos = dx(defined_on=has_outside, level_set=level_set, metadata={'side': '+', 'q': qdeg})
        dx_neg = dx(defined_on=has_inside,  level_set=level_set, metadata={'side': '-', 'q': qdeg})

        L = None
        if p_pos_field in self.field_names:
            q_pos = TestFunction(p_pos_field, self)
            L = (Constant(1.0) * q_pos) * dx_pos
        if p_neg_field in self.field_names:
            q_neg = TestFunction(p_neg_field, self)
            if L is not None: L += (Constant(1.0) * q_neg) * dx_neg
            else: L = (Constant(1.0) * q_neg) * dx_neg

        # No Dirichlet here – this is a pure geometric vector
        K, r = assemble_form(Equation(a=None, L=L), dof_handler=self, bcs=None, quad_order=qdeg, backend=backend)
        return r






    
    def info(self) -> None:
        """
        Print a short summary of the handler: method (CG/DG), per-field DOF counts,
        and the global total number of DOFs. Robust to partially-filled metadata.
        """
        kind = "Mixed" if hasattr(self, "mixed_element") else "legacy"
        method = getattr(self, "method", "cg")
        total = getattr(self, "total_dofs", None)

        print(f"<DofHandler {kind}, method='{method}'>")
        if total is not None:
            print(f"  total DOFs : {total}")

        print("  fields:")
        for f in getattr(self, "field_names", []):
            # prefer slices
            if hasattr(self, "_field_slices") and f in self._field_slices:
                n = len(self._field_slices[f])
            elif hasattr(self, "field_num_dofs") and f in self.field_num_dofs:
                n = int(self.field_num_dofs[f])
            else:
                # fallback: unique DOFs appearing in element_maps
                s = set()
                for cell in self.element_maps.get(f, []):
                    s.update(int(g) for g in cell)
                n = len(s)

            off = getattr(self, "field_offsets", {}).get(f, None)
            off_s = str(off) if off is not None else "-"
            print(f"    - {f:>12s}: ndofs={n:>6d}, offset={off_s}")

        
    def __repr__(self) -> str:
        """
        Return a concise representation with the handler kind and DOF count.

        Examples
        --------
        '<DofHandler Mixed, ndofs=451, method='cg'>'
        '<DofHandler legacy, ndofs=882, method='dg', fields=['u','v']>'
        """
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
