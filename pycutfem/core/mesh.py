import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Callable


from pycutfem.core.topology import Edge, Node, Element
from pycutfem.utils.bitset import BitSet

class Mesh:
    """
    Manages mesh topology, including nodes, elements, and edges.

    This class builds the full connectivity graph from basic node and element
    definitions. It correctly identifies shared edges, assigns geometrically
    correct "left" and "right" elements, and computes outward-pointing normal
    vectors for each edge. It also includes methods for classifying elements
    and edges against a level-set function.
    """
    # Defines the local-corner indices that form each edge, in CCW order.
    _EDGE_TABLE = {
        'tri':  ((0, 1), (1, 2), (2, 0)),
        'quad': ((0, 1), (1, 2), (2, 3), (3, 0)),
    }

    def __init__(self,
                 nodes: List['Node'],
                 element_connectivity: np.ndarray,
                 edges_connectivity: np.ndarray = None,
                 elements_corner_nodes: np.ndarray = None,
                 *,
                 element_type: str = 'tri',
                 poly_order: int = 1):
        """
        Initializes the mesh and builds its topology.
        """
        self.edges_connectivity: np.ndarray = edges_connectivity
        self.element_type = element_type
        self.poly_order = poly_order
        self.nodes_list: List['Node'] = nodes
        self.nodes_x_y_pos = np.array([[n.x, n.y] for n in self.nodes_list], dtype=float)
        self.nodes = np.array([n.id for n in self.nodes_list])
        self.elements_connectivity: np.ndarray = element_connectivity
        self.corner_connectivity: np.ndarray = elements_corner_nodes
        self.elements_list: List['Element'] = []
        self.edges_list: List['Edge'] = []
        self._edge_dict: Dict[Tuple[int, int], 'Edge'] = {}
        self._neighbors: List[List[int]] = [[] for _ in range(len(self.elements_connectivity))]
        self._build_topology()
<<<<<<< Updated upstream
=======
        self.n_elements = len(self.elements_connectivity)
        self.spatial_dim = 2  # Assuming 2D mesh by default
        self._elem_bitsets: Dict[str, BitSet] = {}
        self._edge_bitsets: Dict[str, BitSet] = {}
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

    def _build_topology(self):
        """
        Builds the full mesh topology: Elements, Edges, and Neighbors.
        """
        edge_defs = self._EDGE_TABLE[self.element_type]

        # Step 1: Create basic Element objects
        for eid, elem_nodes in enumerate(self.elements_connectivity):
            centroid_x = np.mean([self.nodes_x_y_pos[nid,0] for nid in self.corner_connectivity[eid]])
            centroid_y = np.mean([self.nodes_x_y_pos[nid,1] for nid in self.corner_connectivity[eid]])

            self.elements_list.append(Element(
                id=eid,
                element_type=self.element_type,
                poly_order=self.poly_order,
                nodes=tuple(elem_nodes),
                corner_nodes=tuple(self.corner_connectivity[eid]),
                centroid_x=centroid_x,
                centroid_y=centroid_y,
            ))

        # Step 2: Build map from each edge to the elements that share it
        edge_incidences: Dict[Tuple[int, int], List[int]] = {}
        for eid, corners in enumerate(self.corner_connectivity):
            for i in range(len(corners)):
                c1, c2 = int(corners[i]), int(corners[(i + 1) % len(corners)])
                key = tuple(sorted((c1, c2)))
                edge_incidences.setdefault(key, []).append(eid)

        def _locate_all_nodes_in_edge(vA: int, vB: int) -> Tuple[int, int]:
            x0, y0 = self.nodes_x_y_pos[vA]
            x1, y1 = self.nodes_x_y_pos[vB]
            dx, dy = x1 - x0, y1 - y0
            L2 = dx*dx + dy*dy
            tol = 1e-12 * np.sqrt(L2)

            ids = []
            for nd in self.nodes_list:
                cross = abs((nd.x - x0)*dy - (nd.y - y0)*dx)
                if cross > tol:
                    continue
                dot = (nd.x - x0)*dx + (nd.y - y0)*dy
                if -tol <= dot <= L2 + tol:
                    ids.append(nd.id)
            # sort along the edge for reproducibility
            ids.sort(key=lambda nid: (self.nodes_x_y_pos[nid,0]-x0)*dx + (self.nodes_x_y_pos[nid,1]-y0)*dy)
            return tuple(ids)
        # Step 3: Create unique Edge objects
        for edge_gid, ((n_min, n_max), shared_eids) in enumerate(edge_incidences.items()):
            left_eid = shared_eids[0]
            vA, vB = -1, -1
            left_elem_corners = self.corner_connectivity[left_eid]
            for i in range(len(left_elem_corners)):
                if {int(left_elem_corners[i]), int(left_elem_corners[(i + 1) % len(left_elem_corners)])} == {n_min, n_max}:
                    vA, vB = int(left_elem_corners[i]), int(left_elem_corners[(i + 1) % len(left_elem_corners)])
                    break
            right_eid = shared_eids[1] if len(shared_eids) > 1 else None
            normal_vec = self._compute_normal((vA, vB))
            all_edge_nodes = _locate_all_nodes_in_edge(vA, vB)
            edge_obj = Edge(gid=edge_gid, nodes=(vA, vB), left=left_eid, right=right_eid, normal=normal_vec,all_nodes=all_edge_nodes)
            self.edges_list.append(edge_obj)
            self._edge_dict[(n_min, n_max)] = edge_obj
            if right_eid is not None:
                self._neighbors[left_eid].append(right_eid)
                self._neighbors[right_eid].append(left_eid)

        # Step 4: Populate each Element's list of its edge GIDs
        for elem in self.elements_list:
            local_edge_gids = [self._edge_dict[tuple(sorted((elem.corner_nodes[c1], elem.corner_nodes[c2])))].gid for c1, c2 in edge_defs]
            elem.edges = tuple(local_edge_gids)
        
        # Step 5: Populate neighbor info on each element
        for elem in self.elements_list:
            for local_edge_idx, edge_gid in enumerate(elem.edges):
                edge = self.edge(edge_gid)
                elem.neighbors[local_edge_idx] = edge.right if edge.left == elem.id else edge.left

    def _compute_normal(self, directed_edge_nodes: Tuple[int, int]) -> np.ndarray:
        """Computes an outward-pointing unit normal for a directed edge."""
        v_start, v_end = self.nodes_x_y_pos[directed_edge_nodes[0]], self.nodes_x_y_pos[directed_edge_nodes[1]]
        directed_vec = v_end - v_start
        raw_normal = np.array([directed_vec[1], -directed_vec[0]], dtype=float)
        length = np.linalg.norm(raw_normal)
        return raw_normal / length if length > 1e-14 else np.array([0.0, 0.0])

    # --- Classification Methods ---

    def _get_node_coords(self) -> np.ndarray:
        """Helper to get node coordinates as a NumPy array."""
        return self.nodes_x_y_pos
        
    def _phi_on_centroids(self, level_set) -> np.ndarray:
        """Compute φ at each element’s centroid."""
        node_coords = self._get_node_coords()
        # Note: Using corner_connectivity is correct for geometric centroid.
        conn = self.corner_connectivity
        corner_coords = node_coords[conn]
        centroids = corner_coords.mean(axis=1)
        
        # FIX: Use apply_along_axis to robustly call the level set function
        # on each centroid, one by one. This mimics the safe evaluation
        # from LevelSetFunction.evaluate_on_nodes and avoids issues with
        # __call__ methods that don't correctly handle batch inputs (N, 2).
        return np.apply_along_axis(level_set, 1, centroids)

    def classify_elements(self, level_set, tol=1e-12):
        """
        Classify each element as 'inside', 'outside', or 'cut'.
        Sets element.tag accordingly and returns indices for each class.
        """
        # Assumes level_set has a method evaluate_on_nodes(mesh)
        phi_nodes = level_set.evaluate_on_nodes(self)
        elem_phi_nodes = phi_nodes[self.corner_connectivity]
        phi_cent = self._phi_on_centroids(level_set)

        min_phi = np.minimum(elem_phi_nodes.min(axis=1), phi_cent)
        max_phi = np.maximum(elem_phi_nodes.max(axis=1), phi_cent)

        inside_mask = (max_phi < -tol)
        outside_mask = (min_phi > tol)
        
        inside_inds = np.where(inside_mask)[0]
        outside_inds = np.where(outside_mask)[0]
        cut_inds = np.where(~(inside_mask | outside_mask))[0]

        for eid in inside_inds: self.elements_list[eid].tag = 'inside'
        for eid in outside_inds: self.elements_list[eid].tag = 'outside'
        for eid in cut_inds: self.elements_list[eid].tag = 'cut'
        tags_el = np.array([e.tag for e in self.elements_list])
        self._elem_bitsets = {t: BitSet(tags_el == t) for t in np.unique(tags_el)}
        
        return inside_inds, outside_inds, cut_inds

    def classify_elements_multi(self, level_sets, tol=1e-12):
        """Classifies elements against multiple level sets."""
        return {idx: self.classify_elements(ls, tol) for idx, ls in enumerate(level_sets)}

    def classify_edges(self, level_set):
        """
        Classify edges as 'interface' or 'ghost' based on element tags.
        """
        phi_nodes = level_set.evaluate_on_nodes(self)
        for edge in self.edges_list:
            # Reset tag to avoid state from previous classifications
            edge.tag = ''

            # --- Primary classification for INTERIOR edges based on element tags ---
            if edge.right is not None:
                left_tag = self.elements_list[edge.left].tag
                right_tag = self.elements_list[edge.right].tag
                tags = {left_tag, right_tag}

                # An edge between two 'cut' elements is a 'ghost' edge.
                if tags == {'cut'}:
                    edge.tag = 'ghost'
                # An edge between a 'cut' element and a non-cut one is an 'interface'.
                elif 'cut' in tags and len(tags) > 1:
                    edge.tag = 'interface'
                # An edge between an 'inside' and 'outside' element is an 'interface'.
                elif tags == {'inside', 'outside'}:
                    edge.tag = 'interface'
            
            # --- Secondary check for any edge whose nodes cross the level set ---
            # The nodal crossing is the strongest indicator of the interface.
            # CRITICAL FIX: This must NOT override a 'ghost' tag.
            if edge.tag != 'ghost' and phi_nodes[edge.nodes[0]] * phi_nodes[edge.nodes[1]] < 0:
                edge.tag = 'interface'
            # Build and cache BitSets *once* – O(n_edges) total
            tags = np.array([e.tag for e in self.edges_list])
            self._edge_bitsets = {
                t: BitSet(tags == t)               # tiny (n_edges) boolean mask
                for t in np.unique(tags) if t      # skip '' untagged edges
            }


    def build_interface_segments(self, level_set, tol=1e-12, qorder=2):
        """
        Populate element.interface_pts for every 'cut' element.
        Each entry is a list of interface points, typically [p0, p1].
        """
        phi_nodes = level_set.evaluate_on_nodes(self)      # φ at every mesh node

        for elem in self.elements_list:
            if elem.tag != 'cut':
                elem.interface_pts = []
                continue

            pts = []
            for gid in elem.edges:                         # loop over its 4 edges
                e = self.edge(gid)
                n0, n1 = e.nodes
                phi0, phi1 = phi_nodes[n0], phi_nodes[n1]

                # Skip edges that are clearly on one side of the interface.
                if (phi0 * phi1 > 0.0) and (abs(phi0) > tol and abs(phi1) > tol):
                    continue
                
                # --- ROBUST INTERPOLATION ---
                if abs(phi0 - phi1) < tol:
                    if abs(phi0) < tol:
                        # The whole edge is on the interface.
                        pts.append(self.nodes_x_y_pos[n0])
                        pts.append(self.nodes_x_y_pos[n1])
                    continue
                else:
                    # Standard linear interpolation
                    t = phi0 / (phi0 - phi1)
                    # Clamp t to the valid range [0, 1] to prevent extrapolation.
                    t = max(0.0, min(1.0, t))
                    
                    p = self.nodes_x_y_pos[n0] + t * (self.nodes_x_y_pos[n1] -
                                                    self.nodes_x_y_pos[n0])
                    pts.append(p)

            # Remove duplicate points that can occur if the interface
            # passes exactly through a mesh node.
            unique_pts = []
            for p in pts:
                is_duplicate = False
                for up in unique_pts:
                    if np.linalg.norm(p - up) < tol:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_pts.append(p)
            
            # --- FIX: A valid interface segment requires exactly two points ---
            # After calculating all intersections and removing duplicates, only
            # proceed if we have exactly two points. Otherwise, it's a
            # degenerate case (grazing a node, etc.), which we filter out.
            if len(unique_pts) == 2:
                # Sort points for reproducibility and assign them.
                unique_pts.sort(key=lambda P: (P[0], P[1]))
                elem.interface_pts = unique_pts
            else:
                # If we don't have exactly two points, treat this element as
                # not having a valid cut segment.
                elem.interface_pts = []
    def edge_bitset(self, tag: str) -> BitSet:
        """Return cached BitSet of edges with the given tag."""
        return self._edge_bitsets.get(tag, BitSet(np.zeros(len(self.edges_list), bool)))

    def element_bitset(self, tag: str) -> BitSet:
        return self._elem_bitsets.get(tag, BitSet(np.zeros(len(self.elements_list), bool)))

    def get_domain_bitset(self, tag: str, *, entity: str = "edge") -> BitSet:     # noqa: N802
        """
        Return a **BitSet** that marks all mesh entities carrying *tag*.

        Parameters
        ----------
        tag     : str
            Mesh or BC tag (e.g. ``'interface'``, ``'ghost'``, ``'left_wall'`` …).
        entity  : {'edge', 'elem', 'element'}
            Select whether the BitSet refers to edges or elements.

        Notes
        -----
        *   For edges and elements the classification routines fill the
            private caches ``_edge_bitsets`` and ``_elem_bitsets`` exactly once
            (O(*n*)).  From then on every call is an O(1) dictionary lookup.
        *   If a cache does *not* exist yet we compute the mask on the fly – this
            costs at most O(*n*) and does **not** pollute the cache (avoids
            hidden state changes).
        """
        entity = entity.lower()
        if entity in {"edge", "edges"}:
            cache = getattr(self, "_edge_bitsets", None)
            if cache is not None and tag in cache:                     # fast path
                return cache[tag]
            mask = np.fromiter((e.tag == tag for e in self.edges_list), bool)  # O(n)
            return BitSet(mask)

        if entity in {"elem", "element", "elements"}:
            cache = getattr(self, "_elem_bitsets", None)
            if cache is not None and tag in cache:
                return cache[tag]
            mask = np.fromiter((el.tag == tag for el in self.elements_list), bool)
            return BitSet(mask)

        raise ValueError(f"Unsupported entity type '{entity}'.")
    
    # --- Public API ---
    def element_char_length(self, elem_id):
        if elem_id is None:
            return 0.0
        return np.sqrt(self.areas()[elem_id])
    def neighbors(self) -> List[int]:
        return self._neighbors

    def edge(self, edge_id: int) -> 'Edge':
        """Return the Edge object corresponding to a global `edge_id`."""
        if not 0 <= edge_id < len(self.edges_list):
            raise IndexError(f"Edge ID {edge_id} out of range.")
        return self.edges_list[edge_id]

    def tag_boundary_edges(self, tag_functions: Dict[str, Callable[[float, float], bool]]):
        """Applies tags to boundary edges based on their midpoint location."""
        for edge in self.edges_list:
            if edge.right is None:
                midpoint = self.nodes_x_y_pos[list(edge.nodes)].mean(axis=0)
                for tag_name, func in tag_functions.items():
                    if func(midpoint[0], midpoint[1]):
                        edge.tag = tag_name
                        break
    
    def tag_edges(self, tag_functions: Dict[str, Callable[[float, float], bool]], overwrite=True):
        """
        Applies tags to ANY edge (boundary or interior) based on its midpoint location.

        Args:
            tag_functions: Dictionary mapping tag names to boolean functions.
            overwrite (bool): If True, it will overwrite any existing tags on the edges.
        """
        for edge in self.edges_list:
            # Skip if the edge already has a tag and we are not overwriting
            if edge.tag and not overwrite:
                continue

            midpoint = self.nodes_x_y_pos[list(edge.nodes)].mean(axis=0)
            for tag_name, func in tag_functions.items():
                if func(midpoint[0], midpoint[1]):
                    edge.tag = tag_name
                    break # Stop after the first matching tag is found

    # ==================================================================
    # NEW FUNCTION 2: Collect DOFs from tagged edges
    # ==================================================================
    def get_dofs_from_tags(self, dof_map: Dict[str, List[str]]) -> Dict[str, List[int]]:
        """
        Collects unique node IDs (DOFs) from edges based on their tags.

        Args:
            dof_map: A dictionary where keys are DOF set names (e.g., 'dirichlet')
                     and values are lists of edge tags to be included in that set.
                     Example: {'dirichlet': ['left', 'wall'], 'neumann': ['inlet']}

        Returns:
            A dictionary where keys are the DOF set names and values are sorted
            lists of unique node IDs.
        """
        # Use sets to automatically handle duplicate node IDs
        dof_sets = {name: set() for name in dof_map.keys()}

        # Create a reverse map (tag -> dof_name) for efficient lookup
        tag_to_dof_name = {}
        for dof_name, tags in dof_map.items():
            for tag in tags:
                tag_to_dof_name[tag] = dof_name

        # Iterate through all edges and collect the nodes
        for edge in self.edges_list:
            if edge.tag in tag_to_dof_name:
                dof_name = tag_to_dof_name[edge.tag]
                # .update() adds all items from the tuple (n1, n2) to the set
                dof_sets[dof_name].update(edge.nodes)

        # Return a dictionary with sorted lists of unique node IDs
        return {name: sorted(list(nodes)) for name, nodes in dof_sets.items()}


    def areas(self) -> np.ndarray:
        """Calculates the geometric area of each element."""
        element_areas = np.zeros(len(self.elements_list))
        for elem in self.elements_list:
            corner_coords = self.nodes_x_y_pos[list(elem.corner_nodes)]
            if self.element_type == 'tri':
                v0, v1, v2 = corner_coords[0], corner_coords[1], corner_coords[2]
                element_areas[elem.id] = 0.5 * np.abs(np.cross(v1 - v0, v2 - v0))
            elif self.element_type == 'quad':
                x, y = corner_coords[:, 0], corner_coords[:, 1]
                element_areas[elem.id] = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return element_areas

    def __repr__(self):
        return (f"<Mesh n_nodes={len(self.nodes_list)}, "
                f"n_elems={len(self.elements_list)}, "
                f"n_edges={len(self.edges_list)}, "
                f"elem_type='{self.element_type}', "
                f"poly_order={self.poly_order}>")
