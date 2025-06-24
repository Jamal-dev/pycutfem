# pycutfem/core/dofhandler.py

from __future__ import annotations
import numpy as np
from typing import Dict, List, Set, Tuple, Callable, Mapping, Iterable, Union, Any

# Assume these are available in the project structure
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node, Edge, Element
from pycutfem.ufl.forms import BoundaryCondition





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
            (self._build_maps_cg if method == "cg" else self._build_maps_dg)()
    # ------------------------------------------------------------------
    # MixedElement-aware Builders
    # ------------------------------------------------------------------
    def _local_node_indices_for_field(self, p_mesh: int, p_f: int, elem_type: str, fld: str) -> List[int]:
        """Indices of geometry nodes that a *p_f* field uses inside a *p_mesh* element."""
        if p_f > p_mesh:
            raise ValueError(f"Field order ({p_f}) exceeds mesh order ({p_mesh}).")
        if p_mesh % p_f != 0 and p_f != 1:
            raise ValueError("Currently require mesh‑order to be multiple of field‑order (except P1).")

        step = p_mesh // p_f if p_f != 0 else p_mesh  # p_f==0 should never happen
        if elem_type == "quad":
            return [j * (p_mesh + 1) + i
                    for j in range(0, p_mesh + 1, step)
                    for i in range(0, p_mesh + 1, step)]
        elif elem_type == "tri":
            if p_f == 1 and p_mesh == 2:
                return [0, 1, 2]
            if p_f == p_mesh:
                return list(range(self.mixed_element._n_basis[fld]))
            raise NotImplementedError("Mixed‑order triangles supported only for P1 on P2 geometry.")
        else:
            raise KeyError(f"Unsupported element_type '{elem_type}'")
    
    def _build_maps_cg_mixed(self) -> None:
        """Continuous‑Galerkin DOF numbering for MixedElement spaces."""
        mesh = self.mixed_element.mesh  # convenience alias
        p_mesh = mesh.poly_order

        # Helper: local‑→physical node mapping for a given *field* order p_f.
        def _local_mesh_indices_for_field(p_f: int) -> List[int]:
            step = p_mesh // p_f
            idx: List[int] = []
            for j in range(0, p_mesh + 1, step):
                for i in range(0, p_mesh + 1, step):
                    idx.append(j * (p_mesh + 1) + i)
            return idx

        # ------------------------------------------------------------------
        # 1) Allocate a global DOF for every *used* mesh node per field
        # ------------------------------------------------------------------
        node_dof_map: Dict[Tuple[str, int], int] = {}
        offset = 0

        field_node_sets: Dict[str, Set[int]] = {f: set() for f in self.field_names}
        for elem in mesh.elements_list:
            # Map local index → global node id
            loc2phys = {loc: nid for loc, nid in enumerate(elem.nodes)}
            for fld in self.field_names:
                p_f = self.mixed_element._field_orders[fld]
                needed_loc_idx = _local_mesh_indices_for_field(p_f)
                for loc in needed_loc_idx:
                    phys_nid = loc2phys[loc]
                    field_node_sets[fld].add(phys_nid)

        # Assign DOF numbers --------------------------------------------------
        for fld in self.field_names:
            self.field_offsets[fld] = offset
            for nid in sorted(field_node_sets[fld]):  # deterministic order
                node_dof_map[(fld, nid)] = offset
                offset += 1
            self.field_num_dofs[fld] = len(field_node_sets[fld])
        self.total_dofs = offset

        # ------------------------------------------------------------------
        # 2) Build element‑wise DOF maps (per field)
        # ------------------------------------------------------------------
        for elem in mesh.elements_list:
            loc2phys = {loc: nid for loc, nid in enumerate(elem.nodes)}
            for fld in self.field_names:
                p_f = self.mixed_element._field_orders[fld]
                loc_idx = _local_mesh_indices_for_field(p_f)
                dofs = [node_dof_map[(fld, loc2phys[l])] for l in loc_idx]
                self.element_maps[fld].append(dofs)
    # ..................................................................
    def _build_maps_dg_mixed(self) -> None:
        """Discontinuous‑Galerkin numbering – element‑local uniqueness."""
        mesh = self.mixed_element.mesh
        p_mesh = mesh.poly_order

        def _local_mesh_indices_for_field(p_f: int) -> List[int]:
            step = p_mesh // p_f
            return [j * (p_mesh + 1) + i
                    for j in range(0, p_mesh + 1, step)
                    for i in range(0, p_mesh + 1, step)]

        offset = 0
        for elem in mesh.elements_list:
            loc2phys = {loc: nid for loc, nid in enumerate(elem.nodes)}
            for fld in self.field_names:
                p_f = self.mixed_element._field_orders[fld]
                loc_idx = _local_mesh_indices_for_field(p_f)
                n_local = len(loc_idx)
                dofs = list(range(offset, offset + n_local))
                self.element_maps[fld].append(dofs)

                # Build per‑node map for BCs ------------------------------
                nd2d: Dict[int, Dict[int, int]] = self.dof_map.setdefault(fld, {})  # type: ignore[assignment]
                for loc, dof in zip(loc_idx, dofs):
                    phys_nid = loc2phys[loc]
                    nd2d.setdefault(phys_nid, {})[elem.id] = dof

                offset += n_local
        self.total_dofs = offset
        self.field_offsets = {fld: 0 for fld in self.field_names}  # not used in DG
        self.field_num_dofs = {fld: self.total_dofs for fld in self.field_names}

    # ------------------------------------------------------------------
    # Legacy Builders (for backward compatibility)
    # ------------------------------------------------------------------
    def _build_maps_cg(self) -> None:  # same as original implementation
        offset = 0
        for fld, mesh in self.fe_map.items():
            self.field_offsets[fld] = offset
            self.field_num_dofs[fld] = len(mesh.nodes_list)
            self.dof_map[fld] = {nd.id: offset + i for i, nd in enumerate(mesh.nodes_list)}
            self.element_maps[fld] = [[self.dof_map[fld][nid] for nid in el.nodes]
                                       for el in mesh.elements_list]
            offset += len(mesh.nodes_list)
        self.total_dofs = offset

    def _build_maps_dg(self) -> None:  # same as original
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
        """Return *stacked* global DOFs for element *element_id*.

        Ordering matches :pyattr:`MixedElement.field_names` then the per‑field
        ordering implicit in its reference element.
        """
        if self.mixed_element is None:
            raise RuntimeError("get_elemental_dofs requires a MixedElement‑backed DofHandler.")
        parts: List[int] = []
        for fld in self.field_names:
            parts.extend(self.element_maps[fld][element_id])
        return np.asarray(parts, dtype=int)

    # ..................................................................
    def get_reference_element(self, field: str | None = None):
        """Return the per‑field reference or the MixedElement itself."""
        if self.mixed_element is None:
            raise RuntimeError("This DofHandler was not built from a MixedElement.")
        if field is None:
            return self.mixed_element
        return self.mixed_element._ref[field]

    # ------------------------------------------------------------------
    #  Dirichlet helpers – UNCHANGED from original implementation
    # ------------------------------------------------------------------
    @staticmethod
    def _nodes_on_segment(mesh: Mesh, n0: int, n1: int, tol_rel: float = 1e-12) -> Tuple[int, ...]:
        x0, y0 = mesh.nodes_x_y_pos[n0]
        x1, y1 = mesh.nodes_x_y_pos[n1]
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        tol = tol_rel * np.sqrt(L2) if L2 else 0.0
        idx: List[int] = []
        for nd in mesh.nodes_list:
            cross = abs((nd.x - x0) * dy - (nd.y - y0) * dx)
            if cross > tol:
                continue
            dot = (nd.x - x0) * dx + (nd.y - y0) * dy
            if -tol <= dot <= L2 + tol:
                idx.append(nd.id)
        if L2:
            idx.sort(key=lambda nid: (mesh.nodes_x_y_pos[nid, 0] - x0) * dx +
                                     (mesh.nodes_x_y_pos[nid, 1] - y0) * dy)
        return tuple(idx)


    # ..................................................................
    def get_dof_pairs_for_edge(self, field: str, edge_gid: int) -> Tuple[List[int], List[int]]:
        if not self._dg_mode:
            raise RuntimeError("Edge DOF pairs only relevant for DG spaces.")
        mesh = self.fe_map[field]
        edge = mesh.edges_list[edge_gid]
        if edge.left is None or edge.right is None:
            raise ValueError("Edge is on boundary – no right element.")
        return (self.element_maps[field][edge.left], self.element_maps[field][edge.right])

    # ..................................................................
    def _require_cg(self, name: str) -> None:
        if self._dg_mode:
            raise NotImplementedError(f"{name} not available for DG spaces – every element owns its DOFs.")

    def get_field_slice(self, field: str) -> List[int]:
        """Global DOF list for *field* (CG only)."""
        self._require_cg("get_field_slice")
        if field not in self.field_names:
            raise ValueError(f"Unknown field '{field}'.")
        return list(self.dof_map[field].values())  # type: ignore[call-arg]

    def get_field_dofs_on_nodes(self, field: str) -> np.ndarray:
        self._require_cg("get_field_dofs_on_nodes")
        if field not in self.field_names:
            raise ValueError(f"Unknown field '{field}'.")
        return np.asarray(sorted(self.dof_map[field].values()), dtype=int)  # type: ignore[call-arg]

    def get_dof_coords(self, field: str) -> np.ndarray:
        self._require_cg("get_dof_coords")
        mesh = self.fe_map[field]
        coords = [(mesh.nodes_x_y_pos[nid][0], mesh.nodes_x_y_pos[nid][1])
                  for nid in sorted(self.dof_map[field].keys())]  # type: ignore[call-arg]
        return np.asarray(coords, dtype=float)

    # ..................................................................
   
    # ------------------------------------------------------------------
    #  Dirichlet handling (CG‑only)
    # ------------------------------------------------------------------
    def get_dirichlet_data(self, bcs: Union[BcLike, Iterable[BcLike]]) -> Dict[int, float]:
        self._require_cg("Dirichlet BC evaluation")
        data: Dict[int, float] = {}
        for field, tag, locator, value_fun in self._expand_bc_specs(bcs):
            mesh = self.fe_map.get(field)
            if mesh is None:
                continue
            nodes = self._collect_nodes_by_tag_or_locator(mesh, tag, locator)
            val_is_callable = callable(value_fun)
            for nid in nodes:
                dof = self.dof_map[field].get(nid)  # type: ignore[attr-defined]
                if dof is None:
                    continue
                x, y = mesh.nodes_x_y_pos[nid]
                data[dof] = value_fun(x, y) if val_is_callable else value_fun  # type: ignore[arg-type]
        return data

    # ..................................................................
    def apply_bcs_to_vector(self, vec: np.ndarray, bcs: Union[BcLike, Iterable[BcLike]]):
        for dof, val in self.get_dirichlet_data(bcs).items():
            if dof < vec.size:
                vec[dof] = val

    # ------------------------------------------------------------------
    #  Function update helper (CG only)
    # ------------------------------------------------------------------
    def add_to_functions(self, delta: np.ndarray, functions: List[Union["Function", "VectorFunction"]]):
        self._require_cg("add_to_functions")
        from pycutfem.ufl.expressions import Function, VectorFunction

        field2func: Dict[str, Union[Function, VectorFunction]] = {}
        for f in functions:
            if isinstance(f, VectorFunction):
                for name in f.field_names:
                    field2func[name] = f
            elif isinstance(f, Function):
                field2func[f.field_name] = f

        updated: Set[int] = set()
        for fld in self.field_names:
            if fld not in field2func:
                continue
            tgt = field2func[fld]
            if id(tgt) in updated:
                continue

            if isinstance(tgt, Function):
                dofs = self.get_field_dofs_on_nodes(fld)
                tgt.nodal_values[:] += delta[dofs]
            else:  # VectorFunction
                all_dofs = np.concatenate([self.get_field_dofs_on_nodes(fn) for fn in tgt.field_names])
                tgt.nodal_values[:] += delta[all_dofs]
            updated.add(id(tgt))

    # ------------------------------------------------------------------
    #  BC helpers (mostly unchanged, minor robustness tweaks)
    # ------------------------------------------------------------------
    def _expand_bc_specs(self, bcs: Union[BcLike, Iterable[BcLike]]) -> List[Tuple[str, Any, Any, Any]]:
        if not bcs:
            return []
        items: Iterable[BcLike] = bcs if isinstance(bcs, (list, tuple, set)) else [bcs]
        out: List[Tuple[str, Any, Any, Any]] = []
        for bc in items:
            if isinstance(bc, BoundaryCondition):
                fields = [bc.field]
                tags = [getattr(bc, "domain_tag", None)]
                locator = getattr(bc, "locator", None)
                value = bc.value
            elif isinstance(bc, Mapping):
                fields = bc.get("fields") or [bc.get("field")]
                tags = bc.get("tags") or [bc.get("tag")]
                locator = bc.get("locator")
                value = bc.get("value")
            else:
                continue
            for fld in fields:
                for tag in tags:
                    out.append((fld, tag, locator, value))
        return out

    # ..................................................................
    def _collect_nodes_by_tag_or_locator(self, mesh: Mesh, tag: str | None,
                                         locator: Callable[[float, float], bool] | None) -> Set[int]:
        nodes: Set[int] = set()
        if tag is not None:
            for edge in mesh.edges_list:
                if getattr(edge, "tag", None) == tag:
                    nodes.update(getattr(edge, "all_nodes", edge.nodes))
        if locator is not None:
            for nd in mesh.nodes_list:
                if locator(nd.x, nd.y):
                    nodes.add(nd.id)
        return nodes

    # ..................................................................
    @staticmethod
    def _nodes_on_segment(mesh: Mesh, n0: int, n1: int, tol_rel: float = 1e-12) -> Tuple[int, ...]:
        x0, y0 = mesh.nodes_x_y_pos[n0]
        x1, y1 = mesh.nodes_x_y_pos[n1]
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        tol = tol_rel * np.sqrt(L2) if L2 else 0.0
        idx: List[int] = []
        for nd in mesh.nodes_list:
            cross = abs((nd.x - x0) * dy - (nd.y - y0) * dx)
            if cross > tol:
                continue
            dot = (nd.x - x0) * dx + (nd.y - y0) * dy
            if -tol <= dot <= L2 + tol:
                idx.append(nd.id)
        if L2:
            idx.sort(key=lambda nid: (mesh.nodes_x_y_pos[nid, 0] - x0) * dx +
                                     (mesh.nodes_x_y_pos[nid, 1] - y0) * dy)
        return tuple(idx)

    # ------------------------------------------------------------------
    def element(self, eid: int) -> Element:
        return self.mesh.elements_list[eid]

    
    # ------------------------------------------------------------------
    #  Debug convenience
    # ------------------------------------------------------------------
    def info(self) -> None:
        print(f"=== DofHandler ({self.method.upper()}) ===")
        for fld in self.field_names:
            print(f"  {fld:>8}: {self.field_num_dofs[fld]} DOFs @ offset {self.field_offsets[fld]}")
        print("  total :", self.total_dofs)
    def __repr__(self) -> str:  # pragma: no cover
        if self.mixed_element:
            return f"<DofHandler Mixed, ndofs={self.total_dofs}, method='{self.method}'>"
        return f"<DofHandler legacy, ndofs={self.total_dofs}, method='{self.method}', fields={self.field_names}>"



# ==============================================================================
#  MAIN BLOCK FOR DEMONSTRATION (Using real Mesh class)
# ==============================================================================
if __name__ == '__main__':
    # This block demonstrates the intended workflow using the actual library components.
    
    # These imports assume the user has pycutfem installed or in their PYTHONPATH
    from pycutfem.utils.meshgen import structured_quad
    from pycutfem.core.topology import Node # Mesh needs this

    # 1. Generate mesh data using a library utility
    print("Generating a 2x1 P1 mesh...")
    nodes, elems, _, corners = structured_quad(1, 0.5, nx=2, ny=1, poly_order=1)

    # 2. Instantiate the real Mesh object
    mesh = Mesh(nodes=nodes, 
                element_connectivity=elems,
                elements_corner_nodes=corners, 
                element_type="quad", 
                poly_order=1)

    # 3. Define and apply boundary tags
    bc_dict = {'left': lambda x,y: x==0,
                'bottom': lambda x,y:y==0,
                'top': lambda x,y: y==0.5, 
                'right':lambda x,y:x==1}
    mesh.tag_boundary_edges(bc_dict)

    # 4. Define the FE space and create the DofHandlers
    fe_map = {'scalar_field': mesh}

    print("\n" + "="*70)
    print("DEMONSTRATION: CONTINUOUS GALERKIN (CG)")
    print("="*70)
    dof_handler_cg = DofHandler(fe_map, method='cg')
    
    print("\nTotal Unique Nodes:", len(nodes))
    print("Total DOFs (CG):", dof_handler_cg.total_dofs)
    
    print("\nElement-to-DOF Maps (CG):")
    for i, elem_map in enumerate(dof_handler_cg.element_maps['scalar_field']):
        print(f"  Element {i}: {elem_map}")
    print("--> Note: DOFs on the shared edge are the same in both lists.")

    print("\n--- Testing get_dirichlet_data (CG) ---")
    dirichlet_def = {
        'left_wall': {
            'fields': ['scalar_field'],
            'tags': ['left'],
            'value': lambda x, y: y * 100.0 # Value is 100 * y-coordinate
        }
    }
    dirichlet_data_cg = dof_handler_cg.get_dirichlet_data(dirichlet_def)
    print("DOF values on the 'left' boundary:")
    for dof, val in sorted(dirichlet_data_cg.items()):
        print(f"  Global DOF {dof}: {val:.1f}")

    print("\n\n" + "="*70)

    # -------------------------------------------------------------------
    # DEMONSTRATION: Q2 (9-node) elements, CG
    # -------------------------------------------------------------------
    print("\n" + "="*70)
    print("DEMONSTRATION: CONTINUOUS GALERKIN (CG) – Q2")
    print("="*70)
    nodes_q2, elems_q2, _, corners_q2 = structured_quad(1, 0.5, nx=2, ny=1, poly_order=2)
    mesh_q2 = Mesh(nodes=nodes_q2,
                element_connectivity=elems_q2,
                elements_corner_nodes=corners_q2,
                element_type="quad",
                poly_order=2)
    mesh_q2.tag_boundary_edges(bc_dict)

    dof_q2 = DofHandler({'scalar_field': mesh_q2}, method='cg')
    print("Total nodes (Q2 mesh):", len(nodes_q2))
    print("Total DOFs (Q2 CG):   ", dof_q2.total_dofs)
    print("\nElement-to-DOF Maps (CG):")
    for i, elem_map in enumerate(dof_q2.element_maps['scalar_field']):
        print(f"  Element {i}: {elem_map}")
    print("--> Note: DOFs on the shared edge are the same in both lists.")

    dirichlet_q2 = dof_q2.get_dirichlet_data(dirichlet_def)
    print(f"Dirichlet DOFs on 'left' boundary (expect 3 nodes * ny=1 = 3):\n  {sorted(dirichlet_q2)}")
    print("DEMONSTRATION: DISCONTINUOUS GALERKIN (DG)")
    print("="*70)
    dof_handler_dg = DofHandler(fe_map, method='dg')

    print("\nNodes per Element:", len(elems[0]))
    print("Total DOFs (DG):", dof_handler_dg.total_dofs, f"({len(elems)} elems * {len(elems[0])} nodes/elem)")
    
    print("\nElement-to-DOF Maps (DG):")
    for i, elem_map in enumerate(dof_handler_dg.element_maps['scalar_field']):
        print(f"  Element {i}: {elem_map}")
    print("--> Note: DOF sets are completely separate for each element.")

    # Find the ID of the interior edge between element 0 and 1
    interior_edge_id = -1
    for edge in mesh.edges_list:
        if edge.left is not None and edge.right is not None:
            interior_edge_id = edge.gid
            break
            
    print("\n--- Testing get_dof_pairs_for_edge (DG) ---")
    left_dofs, right_dofs = dof_handler_dg.get_dof_pairs_for_edge('scalar_field', interior_edge_id)
    print(f"DOF pairs for shared edge {interior_edge_id}:")
    print(f"  DOFs from Left Element (Elem {mesh.edges_list[interior_edge_id].left}): {left_dofs}")
    print(f"  DOFs from Right Element (Elem {mesh.edges_list[interior_edge_id].right}): {right_dofs}")

    print("\n--- Testing get_dirichlet_data (DG) ---")
    dirichlet_data_dg = dof_handler_dg.get_dirichlet_data(dirichlet_def)
    print("DOF values on the 'left' boundary:")
    for dof, val in sorted(dirichlet_data_dg.items()):
        print(f"  Global DOF {dof}: {val:.1f}")

