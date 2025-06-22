# pycutfem/core/dofhandler.py

import numpy as np
from typing import Dict, List, Set, Tuple, Callable, Mapping, Iterable, Union, Any


# We assume the Mesh class and its components are in a sibling file.
from pycutfem.core.mesh import Mesh

from pycutfem.ufl.forms import BoundaryCondition





BcLike = Union[BoundaryCondition, Mapping[str, Any]]

# -----------------------------------------------------------------------------
#  Main class
# -----------------------------------------------------------------------------
class DofHandler:
    """Centralised DOF numbering and boundary‑condition helpers."""

    # .........................................................................
    def __init__(self, fe_map: Dict[str, Mesh], method: str = "cg"):
        if method not in {"cg", "dg"}:
            raise ValueError("method must be 'cg' or 'dg'")
        self.fe_map: Dict[str, Mesh] = fe_map
        self.method: str = method
        self.field_names: List[str] = list(fe_map.keys())
        self.field_offsets: Dict[str, int] = {}
        self.field_num_dofs: Dict[str, int] = {}
        self.element_maps: Dict[str, List[List[int]]] = {f: [] for f in self.field_names}
        self.dof_map: Dict[str, Dict] = {f: {} for f in self.field_names}
        self.total_dofs = 0
        (self._build_maps_cg if method == "cg" else self._build_maps_dg)()

    # ------------------------------------------------------------------
    #  DOF numbering builders
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
    #  Dirichlet helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _nodes_on_segment(mesh: Mesh, n0: int, n1: int, tol_rel: float = 1e-12) -> Tuple[int, ...]:
        """Return *all* node‑ids that lie on the straight segment *(n0,n1)*."""
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

    # .................................................................
    def _collect_nodes_by_tag_or_locator(
        self,
        mesh: Mesh,
        tag: str | None,
        locator: Callable[[float, float], bool] | None,
    ) -> Set[int]:
        """Gather node IDs that belong to *tag* or satisfy *locator*."""
        found: Set[int] = set()

        # 1) Edge tags – fast path via `all_nodes` if present
        if tag is not None:
            for edge in mesh.edges_list:
                if edge.tag != tag:
                    continue
                if getattr(edge, "all_nodes", None):
                    found.update(edge.all_nodes)
                else:
                    # lazy compute + cache for legacy meshes
                    edge.all_nodes = self._nodes_on_segment(mesh, *edge.nodes)  # type: ignore[attr-defined]
                    found.update(edge.all_nodes)

        # 2) Node tags (single‑point constraints)
            for nd in mesh.nodes_list:
                if getattr(nd, "tag", None) == tag:
                    found.add(nd.id)

        # 3) Explicit locator
        if locator is not None:
            for nd in mesh.nodes_list:
                if locator(nd.x, nd.y):
                    found.add(nd.id)
        return found

    # .................................................................
    def _expand_bc_specs(self, bcs: Union[BcLike, Iterable[BcLike]]) -> Iterable[Tuple[str, str | None, Callable[[float, float], bool] | None, Callable[[float, float], float]]]:
        """Normalise whatever the caller passes into a flat iterable of
        *(field, tag, locator, value_function)* tuples.
        """
        if bcs is None:
            return []
        if isinstance(bcs, (list, tuple, set)):
            iterable: Iterable[BcLike] = bcs  # type: ignore[assignment]
        elif isinstance(bcs, Mapping):
            iterable = bcs.values()
        else:
            iterable = [bcs]

        for bc in iterable:
            # --------------------------------------------------  UFL object
            if isinstance(bc, BoundaryCondition):
                if getattr(bc, "method", "dirichlet") != "dirichlet":
                    continue
                yield bc.field, getattr(bc, "domain_tag", None), getattr(bc, "locator", None), bc.value
            # --------------------------------------------------  dict spec
            elif isinstance(bc, Mapping):
                if bc.get("type", "dirichlet") != "dirichlet":
                    continue
                fields = bc.get("fields", [])
                tags = bc.get("tags", [])
                locator = bc.get("locator", None)
                value = bc["value"]
                for field in fields:
                    for tag in tags:
                        yield field, tag, locator, value
            else:
                raise TypeError(f"Unsupported BC spec: {type(bc)}")

    # .................................................................
    def get_dirichlet_data(self, bcs: Union[BcLike, Iterable[BcLike], Mapping[str, Any]]) -> Dict[int, float]:
        """Return map *global_dof → prescribed value* for all Dirichlet specs."""
        data: Dict[int, float] = {}
        for field, tag, locator, value_fun in self._expand_bc_specs(bcs):
            mesh = self.fe_map[field]
            nodes = self._collect_nodes_by_tag_or_locator(mesh, tag, locator)
            if not nodes:
                continue
            if self.method == "cg":
                nd2dof = self.dof_map[field]
                for nid in nodes:
                    dof = nd2dof.get(nid)
                    if dof is None:
                        continue
                    x, y = mesh.nodes_x_y_pos[nid]
                    data[dof] = value_fun(x, y)
            else:
                nd2dof: Dict[int, Dict[int, int]] = self.dof_map[field]  # type: ignore[assignment]
                for nid in nodes:
                    for dof in nd2dof.get(nid, {}).values():
                        x, y = mesh.nodes_x_y_pos[nid]
                        data[dof] = value_fun(x, y)
        return data

    # ------------------------------------------------------------------
    #  DG helper (unchanged)
    # ------------------------------------------------------------------
    def get_dof_pairs_for_edge(self, field: str, edge_gid: int) -> Tuple[List[int], List[int]]:
        if self.method != "dg":
            raise RuntimeError("Edge DOF pairs only relevant for DG spaces")
        mesh = self.fe_map[field]
        edge = mesh.edges_list[edge_gid]
        if edge.left is None or edge.right is None:
            raise ValueError("Edge is on boundary – no right element")
        return (self.element_maps[field][edge.left], self.element_maps[field][edge.right])
    
    def get_field_slice(self, field: str) -> List[int]:
        """Return the global DOF indices for the given field."""
        if field not in self.field_names:
            raise ValueError(f"Field '{field}' not found in DofHandler")
        
        return list(self.dof_map[field].values())
    
    def get_field_dofs_on_nodes(self, field: str) -> np.ndarray:
        """
        Returns a sorted array of all global DOF indices for a given field.
        """
        if field not in self.field_names:
            raise ValueError(f"Field '{field}' not found in DofHandler")
        # The values of the dof_map are the global DOF indices for that field
        return np.array(sorted(self.dof_map[field].values()))

    def get_dof_coords(self, field: str) -> np.ndarray:
        """
        Returns an array of (x,y) coordinates for each DOF in a given field,
        sorted in the same order as get_field_dofs_on_nodes.
        """
        if field not in self.field_names:
            raise ValueError(f"Field '{field}' not found in DofHandler")
            
        mesh = self.fe_map[field]
        dof_map = self.dof_map[field] # Dict of {node_id: global_dof}

        # Create a list of (global_dof, coordinate_tuple) pairs
        dof_coord_pairs = [
            (global_dof, tuple(mesh.nodes_x_y_pos[node_id]))
            for node_id, global_dof in dof_map.items()
        ]
        # Sort the list based on the global_dof to ensure order is consistent
        dof_coord_pairs.sort(key=lambda pair: pair[0])
        
        # Unzip the sorted list into just the coordinates
        coords_list = [pair[1] for pair in dof_coord_pairs]
        
        return np.array(coords_list)
    
    def apply_bcs_to_vector(self, vector: np.ndarray, bcs: Union[BcLike, Iterable[BcLike]]):
        """
        Modifies a vector in-place by applying Dirichlet boundary conditions.
        """
        dirichlet_data = self.get_dirichlet_data(bcs)
        for dof, value in dirichlet_data.items():
            if dof < len(vector):
                vector[dof] = value

    def add_to_functions(self, global_delta_vector: np.ndarray, functions: List[Union["Function", "VectorFunction"]]):
        """
        Distributes a global correction vector to the nodal_values of a list
        of Function/VectorFunction objects.
        """
        from pycutfem.ufl.expressions import Function, VectorFunction
        # Create a mapping from each field name to the Function object that contains it.
        field_to_func_map = {}
        for func in functions:
            if isinstance(func, VectorFunction):
                for field_name in func.field_names:
                    field_to_func_map[field_name] = func
            elif isinstance(func, Function):
                field_to_func_map[func.field_name] = func

        # Keep track of functions that have been updated to avoid redundant additions
        # for components of the same VectorFunction.
        updated_funcs = set()

        for field in self.field_names:
            if field not in field_to_func_map:
                continue
            
            target_func = field_to_func_map[field]
            
            # If the target is a scalar Function, update it directly.
            if isinstance(target_func, Function) and not isinstance(target_func, VectorFunction):
                if id(target_func) not in updated_funcs:
                    field_dofs = self.get_field_dofs_on_nodes(field)
                    corrections = global_delta_vector[field_dofs]
                    # Directly add to the function's internal data array
                    target_func.nodal_values[:] += corrections
                    updated_funcs.add(id(target_func))

            # If the target is a VectorFunction, perform one update for all its components.
            elif isinstance(target_func, VectorFunction):
                if id(target_func) not in updated_funcs:
                    # Get all global dofs for this entire vector function
                    func_global_dofs = np.concatenate(
                        [self.get_field_dofs_on_nodes(fn) for fn in target_func.field_names]
                    )
                    # Get the corresponding correction values from the global vector
                    corrections = global_delta_vector[func_global_dofs]
                    
                    # The internal order of nodal_values in VectorFunction matches the
                    # concatenated order of fields from the DofHandler, so we can add directly.
                    target_func.nodal_values[:] += corrections
                    updated_funcs.add(id(target_func))

    # ------------------------------------------------------------------
    #  Debug convenience
    # ------------------------------------------------------------------
    def info(self) -> None:
        print(f"=== DofHandler ({self.method.upper()}) ===")
        for fld in self.field_names:
            print(f"  {fld:>8}: {self.field_num_dofs[fld]} DOFs @ offset {self.field_offsets[fld]}")
        print("  total :", self.total_dofs)




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

