"""pycutfem.core.mesh"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np

@dataclass(slots=True)
class Edge:
    id: int                   
    nodes: Tuple[int, int]      # Global node indices of the edge's endpoints      
    left: int | None            # Element ID on the left side of the edge 
    right: int | None           # Element ID on the right side of the edge
    normal: np.ndarray          # Normal vector of the edge, pointing outward from the left element
    tag: str = ""

class Mesh:
    # _EDGE_TABLE defines edges based on LOCAL indices of CORNERS/VERTICES
    # For triangles (0,1,2): edge (V0,V1), (V1,V2), (V2,V0)
    # For quads (0,1,2,3): edge (V0,V1), (V1,V2), (V2,V3), (V3,V0) (e.g., BL,BR,TR,TL)
    _EDGE_TABLE = {
        'tri': ((0,1),(1,2),(2,0)),
        'quad':((0,1),(1,2),(2,3),(3,0)),
    }
    def __init__(self, nodes: np.ndarray, elements: np.ndarray, element_type='tri', element_order: int = 1, **kwargs):
        self.nodes = np.ascontiguousarray(nodes, dtype=float)
        self.elements = np.ascontiguousarray(elements, dtype=int)
        self.element_type = element_type
        self.element_order = element_order
        if self.nodes.ndim!=2 or self.nodes.shape[1]!=2:
            raise ValueError('nodes must be (N,2)')
        if self.elements.ndim!=2:
            raise ValueError('elements must be (M,k_nodes_per_element)')
        if element_type not in self._EDGE_TABLE:
            raise ValueError(element_type)
        
        expected_nodes_per_elem = -1
        if self.element_type == 'tri':
            if self.element_order >= 0: # Pk elements
                expected_nodes_per_elem = (self.element_order + 1) * (self.element_order + 2) // 2
            else:
                raise ValueError(f"Unsupported element_order {self.element_order} for triangle.")
        elif self.element_type == 'quad':
            if self.element_order >= 0: # Qn elements
                expected_nodes_per_elem = (self.element_order + 1)**2
            else:
                raise ValueError(f"Unsupported element_order {self.element_order} for quadrilateral.")
        else:
            raise ValueError(f"Unknown element_type: {self.element_type}")
        
        if self.elements.shape[0] > 0 and self.elements.shape[1] != expected_nodes_per_elem:
            raise ValueError(
                f"Mismatch for {self.element_type} order {self.element_order}: "
                f"elements array has {self.elements.shape[1]} nodes per element, "
                f"but expected {expected_nodes_per_elem}."
            )

        if self.element_type not in self._EDGE_TABLE: # Should not happen if previous check passed
            raise ValueError(f"Element type '{self.element_type}' not in _EDGE_TABLE.")
            
        # --- Initialize members ---
        self.edges: List[Edge] = []
        self._edge_dict: Dict[Tuple[int,int], int] = {}
        self._neighbors: List[List[int]] = [[] for _ in range(len(self.elements))]
        self.edge_tag = np.empty(0, dtype=object)
        self.elem_tag = np.zeros(len(self.elements), dtype=object)
        self._build_edges()
    # ------------------------
    def _get_element_corner_global_indices(self, element_idx: int) -> List[int]:
        """
        Returns a list of global node indices for the CORNER/PRIMARY vertices
        of the specified element. The order of vertices returned is suitable
        for defining a CCW polygon.
        Assumes structured_quad/tri node ordering.
        """
        elem_all_nodes_gids = self.elements[element_idx]
        order = self.element_order

        if self.element_type == 'tri':
            if order == 0: # P0
                # A P0 element is a point; for geometric operations like area/edges,
                # it's degenerate. Return its single node repeated if needed by callers.
                return [elem_all_nodes_gids[0]] * 3 if len(elem_all_nodes_gids) > 0 else []
            
            # Pk vertices (V0, V1, V2 in CCW order for reference mapping)
            # V0: local index 0
            # V1: local index 'order' (k)
            # V2: local index (k+1)(k+2)/2 - 1 (last node in the Pk sequence)
            idx_v0 = 0
            idx_v1 = order
            idx_v2 = (order + 1) * (order + 2) // 2 - 1
            
            # Basic sanity check for indices against element's node list length
            max_idx = len(elem_all_nodes_gids) -1
            if not (idx_v0 <= max_idx and idx_v1 <= max_idx and idx_v2 <= max_idx):
                 raise IndexError(f"Corner indices out of bounds for P{order} tri element {element_idx} "
                                  f"with {len(elem_all_nodes_gids)} nodes. Requested indices: 0, {order}, {idx_v2}")
            return [
                elem_all_nodes_gids[idx_v0],
                elem_all_nodes_gids[idx_v1],
                elem_all_nodes_gids[idx_v2],
            ]
        elif self.element_type == 'quad':
            if order == 0: # Q0
                return [elem_all_nodes_gids[0]] * 4 if len(elem_all_nodes_gids) > 0 else []

            # Qn vertices (Order: BL, BR, TR, TL for CCW polygon)
            # BL: local index 0
            # BR: local index 'order' (k)
            # TL: local index 'order' * ('order' + 1)
            # TR: local index 'order' * ('order' + 1) + 'order'
            idx_bl = 0
            idx_br = order
            idx_tl = order * (order + 1)
            idx_tr = order * (order + 1) + order
            
            max_idx = len(elem_all_nodes_gids) -1
            if not (idx_bl <= max_idx and idx_br <= max_idx and idx_tl <= max_idx and idx_tr <= max_idx):
                 raise IndexError(f"Corner indices out of bounds for Q{order} quad element {element_idx} "
                                  f"with {len(elem_all_nodes_gids)} nodes. Requested: BL={idx_bl}, BR={idx_br}, TL={idx_tl}, TR={idx_tr}")
            return [
                elem_all_nodes_gids[idx_bl], # V0 (BL)
                elem_all_nodes_gids[idx_br], # V1 (BR)
                elem_all_nodes_gids[idx_tr], # V2 (TR)
                elem_all_nodes_gids[idx_tl], # V3 (TL)
            ]
        else: # Should have been caught in __init__
            raise ValueError(f"Unknown element_type: {self.element_type} in _get_element_corner_global_indices")
    # ------------------------
    def areas(self) -> np.ndarray:
        if len(self.elements) == 0:
            return np.array([])
            
        element_areas = np.zeros(len(self.elements))
        if self.element_order == 0: # Point elements have zero geometric area
            return element_areas

        for eid in range(len(self.elements)):
            corner_gids = self._get_element_corner_global_indices(eid)
            # Ensure we have enough corners to form a polygon
            if (self.element_type == 'tri' and len(corner_gids) < 3) or \
               (self.element_type == 'quad' and len(corner_gids) < 4):
                element_areas[eid] = 0.0 # Or handle as error
                continue

            corners_coords = self.nodes[corner_gids]

            if self.element_type == 'tri':
                v0, v1, v2 = corners_coords[0], corners_coords[1], corners_coords[2]
                element_areas[eid] = 0.5 * np.abs(np.cross(v1 - v0, v2 - v0))
            elif self.element_type == 'quad':
                # Using BL, BR, TR, TL order from _get_element_corner_global_indices
                v0, v1, v2, v3 = corners_coords[0], corners_coords[1], corners_coords[2], corners_coords[3]
                # Area by splitting into two triangles (v0,v1,v2) and (v0,v2,v3)
                area1 = 0.5 * np.abs(np.cross(v1 - v0, v2 - v0)) # Triangle (v0,v1,v2)
                area2 = 0.5 * np.abs(np.cross(v2 - v0, v3 - v0)) # Triangle (v0,v2,v3)
                element_areas[eid] = area1 + area2
        return element_areas
    # ------------------------
    def edge(self,eid:int)->Edge:
        return self.edges[eid]
    def edge_dict(self):
        return self._edge_dict
    def neighbors(self):
        return self._neighbors
    # ------------------------
    def _build_edges(self):
        if self.element_order == 0 or len(self.elements) == 0: # Point elements don't have edges here
            self.edges = []
            self._edge_dict = {}
            self._neighbors = [[] for _ in range(len(self.elements))]
            self.edge_tag = np.array([], dtype=object)
            return

        # _EDGE_TABLE defines connectivity based on *local indices of conceptual corners*
        # e.g., for a triangle, edge 0 is between conceptual corner 0 (V0) and conceptual corner 1 (V1).
        local_corner_edge_definitions = self._EDGE_TABLE[self.element_type]
        
        # `inc` maps a sorted tuple of global node IDs (of an edge) to a list of element IDs sharing that edge.
        inc: Dict[Tuple[int, int], List[int]] = {} 

        for eid in range(len(self.elements)):
            # Get global node IDs of the actual corner/primary vertices for this element
            actual_corner_gids_for_elem = self._get_element_corner_global_indices(eid)
            
            if not actual_corner_gids_for_elem: # Should not happen if order > 0
                continue

            # Iterate through the conceptual edges defined by local corner indices (0,1,2 for tri; 0,1,2,3 for quad)
            for idx_corner1_conceptual, idx_corner2_conceptual in local_corner_edge_definitions:
                # Map these conceptual local corner indices to the global node IDs of the actual corners
                gid1 = actual_corner_gids_for_elem[idx_corner1_conceptual]
                gid2 = actual_corner_gids_for_elem[idx_corner2_conceptual]
                
                edge_tuple_sorted_gids = tuple(sorted((gid1, gid2))) # Canonical representation of the edge
                inc.setdefault(edge_tuple_sorted_gids, []).append(eid)
        
        new_edges_list: List[Edge] = []
        new_edge_dict: Dict[Tuple[int, int], int] = {}
        new_neighbors: List[List[int]] = [[] for _ in range(len(self.elements))]

        for edge_id_counter, (edge_gids_pair, shared_elem_ids) in enumerate(inc.items()):
            left_elem_id = shared_elem_ids[0]
            right_elem_id = shared_elem_ids[1] if len(shared_elem_ids) == 2 else None
            
            if right_elem_id is not None: # Internal edge
                new_neighbors[left_elem_id].append(right_elem_id)
                new_neighbors[right_elem_id].append(left_elem_id)
            
            # The edge_gids_pair is already sorted. _compute_normal needs to be robust or
            # we need to pass the directed edge based on left_elem_id's winding.
            # For now, pass the sorted pair and let _compute_normal handle orientation.
            normal_vector = self._compute_normal(edge_gids_pair, left_elem_id) 
            
            new_edges_list.append(Edge(id=edge_id_counter, nodes=edge_gids_pair, 
                                       left=left_elem_id, right=right_elem_id, normal=normal_vector))
            new_edge_dict[edge_gids_pair] = edge_id_counter
            
        self.edges = new_edges_list
        self._edge_dict = new_edge_dict
        self._neighbors = new_neighbors
        self.edge_tag = np.array([''] * len(self.edges), dtype=object) # Initialize tags

    def _compute_normal(self, edge_nodes_gids_tuple: Tuple[int, int], left_elem_id: int) -> np.ndarray:
        # edge_nodes_gids_tuple is sorted (smaller_gid, larger_gid)
        p0_gid, p1_gid = edge_nodes_gids_tuple
        p0_coords, p1_coords = self.nodes[p0_gid], self.nodes[p1_gid]
        
        edge_vector = p1_coords - p0_coords # Vector from node with smaller GID to node with larger GID
        
        # Initial normal (90-degree rotation: (dx, dy) -> (dy, -dx) or (-dy, dx))
        # (dy, -dx) gives an outward normal if edge is traversed CCW w.r.t element
        raw_normal = np.array([edge_vector[1], -edge_vector[0]]) 
        
        norm_val = np.linalg.norm(raw_normal)
        if norm_val < 1e-12: # Degenerate edge or coincident points
            return np.array([0.0, 0.0]) # Or raise error
        unit_normal = raw_normal / norm_val
        
        # Ensure normal points outwards from the 'left_elem_id'
        # Centroid of the 'left_elem_id' (using all its nodes for accuracy)
        elem_all_nodes_for_centroid = self.elements[left_elem_id]
        centroid_coords = self.nodes[elem_all_nodes_for_centroid].mean(axis=0)
        
        edge_midpoint_coords = 0.5 * (p0_coords + p1_coords)
        
        # Vector from edge midpoint to element centroid
        vec_mid_to_centroid = centroid_coords - edge_midpoint_coords
        
        # If the normal points towards the centroid (dot product > 0), flip it
        if np.dot(vec_mid_to_centroid, unit_normal) > 1e-9: # Add tolerance for dot product check
            unit_normal *= -1.0
            
        return unit_normal
    def __repr__(self):
        return f'<Mesh n_nodes={len(self.nodes)} n_elems={len(self.elements)} n_edges={len(self.edges)}>'
