import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Callable


from pycutfem.core.topology import Edge, Node, Element


class Mesh:
    # Defines how to connect a list of corner nodes to form edges in CCW order.
    _EDGE_TABLE = {
        'tri': ((0, 1), (1, 2), (2, 0)),
        'quad': ((0, 1), (1, 2), (2, 3), (3, 0)),
    }

    def __init__(self, nodes: List[Node], element_connectivity: np.ndarray, 
                 edges_connectivity: np.ndarray, elements_corner_nodes: np.ndarray, *,
                 element_type: str = 'tri', poly_order: int = 1,
                 nx: Optional[int] = None, ny: Optional[int] = None):
        
        self.element_type = element_type
        self.poly_order = poly_order
        self.nx = nx # Store grid dimensions if provided
        self.ny = ny # Store grid dimensions if provided

        # --- Store raw data and initialize object lists ---
        self.nodes_list: List[Node] = nodes
        self.elements_connectivity: np.ndarray = element_connectivity
        self.corner_connectivity: np.ndarray = elements_corner_nodes
        
        # Create a NumPy array of node coordinates for fast numerical access
        self.nodes = np.array([[n.x, n.y] for n in self.nodes_list])

        self.elements_list: List[Element] = []
        self.edges_list: List[Edge] = []
        self._edge_dict: Dict[Tuple[int, int], Edge] = {}
        self._neighbors: List[List[int]] = [[] for _ in range(len(self.elements_connectivity))]
        
        # --- Build the complete object-oriented topology ---
        self._build_topology()

    def _build_topology(self):
        """
        A single, robust pass to build all Element and Edge objects
        and establish their relationships using the pre-computed connectivity tables.
        """
        # 1. Create Element objects
        for eid, elem_gids in enumerate(self.elements_connectivity):
            corner_gids = self.corner_connectivity[eid]
            left_n, right_n, bottom_n, top_n = None, None, None, None
            if self.nx is not None and self.ny is not None:
                el_j, el_i = divmod(eid, self.nx)
                if el_i > 0: left_n = eid - 1
                if el_i < self.nx - 1: right_n = eid + 1
                if el_j > 0: bottom_n = eid - self.nx
                if el_j < self.ny - 1: top_n = eid + self.ny

            self.elements_list.append(Element(
                id=eid, element_type=self.element_type, poly_order=self.poly_order,
                nodes=tuple(elem_gids), corner_nodes=tuple(corner_gids),
                left=left_n, right=right_n, top=top_n, bottom=bottom_n
            ))

        # 2. Identify shared edges to determine neighbors
        edge_incidences: Dict[Tuple[int, int], List[int]] = {}
        for eid, corners in enumerate(self.corner_connectivity):
            for idx_c1, idx_c2 in self._EDGE_TABLE[self.element_type]:
                edge_tuple = tuple(sorted((corners[idx_c1], corners[idx_c2])))
                edge_incidences.setdefault(edge_tuple, []).append(eid)
        
        # 3. Create Edge objects, compute their normals, and link neighbors
        for edge_id, (nodes_pair, shared_elems) in enumerate(edge_incidences.items()):
            shared_elems.sort()
            left_elem_id, right_elem_id = shared_elems[0], shared_elems[1] if len(shared_elems) == 2 else None
            
            # **FIX**: Compute the normal *before* creating the Edge object.
            # The normal is defined as pointing outwards from the 'left' element.
            normal_vector = self._compute_normal(nodes_pair, left_elem_id)
            
            # Pass the computed normal to the constructor.
            edge = Edge(id=edge_id, nodes=nodes_pair, left=left_elem_id, right=right_elem_id, normal=normal_vector)
            
            self.edges_list.append(edge)
            self._edge_dict[nodes_pair] = edge
            
            if right_elem_id is not None:
                self._neighbors[left_elem_id].append(right_elem_id)
                self._neighbors[right_elem_id].append(left_elem_id)

        # 4. Link elements to their edges
        for element in self.elements_list:
            elem_edges = []
            for idx_c1, idx_c2 in self._EDGE_TABLE[self.element_type]:
                edge_key = tuple(sorted((element.corner_nodes[idx_c1], element.corner_nodes[idx_c2])))
                elem_edges.append(self._edge_dict[edge_key].id)
            element.edges = tuple(elem_edges)

    def _compute_normal(self, edge_nodes_gids: Tuple[int, int], left_elem_id: int) -> np.ndarray:
        """
        Computes a normal vector for an edge, oriented outwards from the 'left_elem_id'.
        **FIX**: This method now takes node GIDs and an element ID, not an Edge object.
        """
        # We need the corner nodes of the left element to find the directed edge.
        left_elem_corners = self.corner_connectivity[left_elem_id]
        
        directed_edge_vec = None
        # Find the directed edge (v1, v2) as it appears in the CCW ordering
        # of the left element's corners to ensure consistent normal direction.
        for i in range(len(left_elem_corners)):
            c1_gid = left_elem_corners[i]
            c2_gid = left_elem_corners[(i + 1) % len(left_elem_corners)]
            if {c1_gid, c2_gid} == set(edge_nodes_gids):
                directed_edge_vec = self.nodes[c2_gid] - self.nodes[c1_gid]
                break
        
        if directed_edge_vec is None:
            raise RuntimeError(f"Could not find directed edge for {edge_nodes_gids} in its left element {left_elem_id}")

        # Normal is a 90-degree rotation of the CCW directed edge vector, which points outwards.
        raw_normal = np.array([directed_edge_vec[1], -directed_edge_vec[0]]) 
        norm = np.linalg.norm(raw_normal)
        return raw_normal / norm if norm > 1e-12 else np.array([0.0, 0.0])

    # --- Re-integrated Public Methods ---

    def areas(self) -> np.ndarray:
        """Calculates the geometric area of each element using its corner vertices."""
        element_areas = np.zeros(len(self.elements_list))
        if self.poly_order == 0: return element_areas
        for element in self.elements_list:
            corner_coords = self.nodes[list(element.corner_nodes)]
            if self.element_type == 'tri':
                v0, v1, v2 = corner_coords[0], corner_coords[1], corner_coords[2]
                element_areas[element.id] = 0.5 * np.abs(np.cross(v1 - v0, v2 - v0))
            elif self.element_type == 'quad':
                v0, v1, v2, v3 = corner_coords[0], corner_coords[1], corner_coords[2], corner_coords[3]
                # Using the triangulation (v0,v1,v2) and (v0,v2,v3) to be consistent.
                area1 = 0.5 * np.abs(np.cross(v1 - v0, v2 - v0))
                area2 = 0.5 * np.abs(np.cross(v2 - v0, v3 - v0))
                element_areas[element.id] = area1 + area2
        return element_areas

    def element_char_length(self, elem_id: int) -> float:
        """Computes the characteristic length of an element (sqrt(area))."""
        if elem_id is None: return 0.0
        return np.sqrt(self.areas()[elem_id])
        
    def _find_local_edge(self, elem_id: int, global_edge_nodes: tuple) -> int:
        """Finds the local index (0,1,2,...) of a global edge within an element."""
        elem_corners = self.elements_list[elem_id].corner_nodes
        edge_defs = self._EDGE_TABLE[self.element_type]
        set_to_find = set(global_edge_nodes)
        for local_idx, (idx_c1, idx_c2) in enumerate(edge_defs):
            if {elem_corners[idx_c1], elem_corners[idx_c2]} == set_to_find:
                return local_idx
        raise RuntimeError(f"Edge {global_edge_nodes} not found in element {elem_id}.")

    def edge(self, edge_id: int) -> Edge:
        """Returns the Edge object for the given edge ID."""
        if edge_id < 0 or edge_id >= len(self.edges_list):
            raise IndexError(f"Edge ID {edge_id} is out of bounds.")
        return self.edges_list[edge_id]
        
    def neighbors(self, elem_id: int) -> List[int]:
        """Returns the list of neighboring element IDs for the given element ID."""
        if elem_id < 0 or elem_id >= len(self.elements_list):
            raise IndexError(f"Element ID {elem_id} is out of bounds.")
        return self._neighbors[elem_id]
        
    def edge_dict(self) -> Dict[Tuple[int, int], Edge]:
        """Returns the edge dictionary mapping (node1_gid, node2_gid) to Edge object."""
        return self._edge_dict

    def tag_boundary_edges(self, tag_functions: Dict[str, Callable[[float, float], bool]]):
        """Tags boundary edges based on user-provided functions."""
        for edge in self.edges_list:
            if edge.right is not None: continue
            midpoint = self.nodes[list(edge.nodes)].mean(axis=0)
            for tag_name, func in tag_functions.items():
                if func(midpoint[0], midpoint[1]):
                    edge.tag = tag_name
                    break
    
    def __repr__(self):
        return (f'<Mesh n_nodes={len(self.nodes_list)} n_elems={len(self.elements_list)} '
                f'n_edges={len(self.edges_list)} elem_type={self.element_type} poly_order={self.poly_order}>')