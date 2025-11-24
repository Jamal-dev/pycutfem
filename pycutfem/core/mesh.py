import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Callable, Union, Iterable


from pycutfem.core.topology import Edge, Node, Element

from pycutfem.utils.bitset import BitSet
from pycutfem.core.sideconvention import SIDE
from pycutfem.ufl.helpers_geom import edge_root_pn

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
        self.n_elements = len(self.elements_connectivity) # number of elements
        self.spatial_dim = 2  # Assuming 2D mesh by default
        self._elem_bitsets: Dict[str, BitSet] = {}
        self._edge_bitsets: Dict[str, BitSet] = {}
        self.areas_list: Optional[np.ndarray] = self.areas()

    def num_elements(self) -> int:
        """Returns the number of elements in the mesh."""
        return self.n_elements
    @staticmethod
    def _reference_corner_coords(mesh: "Mesh") -> np.ndarray:
        if mesh.element_type == 'tri':
            return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], float)
        if mesh.element_type == 'quad':
            return np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], float)
        raise KeyError(mesh.element_type)

    def _build_topology(self):
        """
        Builds the full mesh topology: Elements, Edges, and Neighbors.
        Respects a provided ``edges_connectivity`` if available; otherwise
        infers edges from the polygonal corner connectivity.
        """
        edge_defs = self._EDGE_TABLE[self.element_type]

        # Step 1: Create Element objects
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

        elem_node_sets = [set(el.nodes) for el in self.elements_list]

        def _locate_all_nodes_in_edge(vA: int, vB: int) -> Tuple[int, ...]:
            x0, y0 = self.nodes_x_y_pos[vA]
            x1, y1 = self.nodes_x_y_pos[vB]
            dx, dy = x1 - x0, y1 - y0
            L2 = dx*dx + dy*dy
            tol = SIDE.tol * np.sqrt(max(L2, 0.0))

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

        def _sort_along_edge(n_start: int, n_end: int, nodes: Iterable[int]) -> List[int]:
            x0, y0 = self.nodes_x_y_pos[n_start]
            x1, y1 = self.nodes_x_y_pos[n_end]
            dx, dy = x1 - x0, y1 - y0
            if abs(dx) + abs(dy) < 1e-30:
                return sorted(nodes)
            return sorted(
                [int(n) for n in nodes],
                key=lambda nid: (self.nodes_x_y_pos[nid,0]-x0)*dx + (self.nodes_x_y_pos[nid,1]-y0)*dy,
            )

        # Step 2: Build edges
        if self.edges_connectivity is not None:
            # Existing explicit edges path – keep behaviour, but populate new metadata.
            provided_edges = [list(map(int, e)) for e in np.asarray(self.edges_connectivity)]
            elem_edge_map: Dict[Tuple[int, int], List[Tuple[int, int, Tuple[int, int]]]] = {}
            for eid, corners in enumerate(self.corner_connectivity):
                for lid, (i0, i1) in enumerate(edge_defs):
                    a = int(corners[i0]); b = int(corners[i1])
                    key = tuple(sorted((a, b)))
                    elem_edge_map.setdefault(key, []).append((eid, lid, (a, b)))

            elem_edges: List[List[List[int]]] = [[[] for _ in edge_defs] for _ in self.elements_list]
            for edge_gid, edge_nodes in enumerate(provided_edges):
                coords = self.nodes_x_y_pos[edge_nodes]
                span_x = float(np.ptp(coords[:, 0]))
                span_y = float(np.ptp(coords[:, 1]))
                if span_x >= span_y:
                    order = np.argsort(coords[:, 0] + 1e-12 * coords[:, 1])
                else:
                    order = np.argsort(coords[:, 1] + 1e-12 * coords[:, 0])
                ordered_nodes = [edge_nodes[i] for i in order]
                n_start, n_end = int(ordered_nodes[0]), int(ordered_nodes[-1])
                # Enrich with any intermediate (hanging) nodes that lie on the geometric edge
                geom_nodes = _locate_all_nodes_in_edge(n_start, n_end)
                geom_set = set(geom_nodes)
                for nd in ordered_nodes:
                    if nd not in geom_set:
                        geom_nodes = tuple(list(geom_nodes) + [nd])
                        geom_set.add(nd)
                key = tuple(sorted((n_start, n_end)))
                incidences = elem_edge_map.get(key, [])
                if not incidences:
                    left_eid = None
                    right_eid = None
                    normal_vec = np.array([0.0, 0.0])
                    left_lid = None
                    right_lid = None
                else:
                    left_eid, left_lid, oriented = incidences[0]
                    right_eid = incidences[1][0] if len(incidences) > 1 else None
                    right_lid = incidences[1][1] if len(incidences) > 1 else None
                    vA, vB = oriented
                    normal_vec = self._compute_normal((vA, vB))
                    if right_eid is not None:
                        self._neighbors[left_eid].append(right_eid)
                        self._neighbors[right_eid].append(left_eid)
                        elem_edges[right_eid][right_lid].append(edge_gid)
                edge_obj = Edge(
                    gid=edge_gid,
                    nodes=(n_start, n_end),
                    left=left_eid,
                    right=right_eid,
                    normal=normal_vec,
                    all_nodes=tuple(geom_nodes),
                    lid=left_lid,
                    right_lid=right_lid,
                    left_nodes=tuple(n for n in geom_nodes if left_eid is not None and n in elem_node_sets[left_eid]),
                    right_nodes=tuple(n for n in geom_nodes if right_eid is not None and n in elem_node_sets[right_eid]),
                )
                self.edges_list.append(edge_obj)
                self._edge_dict[key] = edge_obj
                if left_eid is not None and left_lid is not None:
                    elem_edges[left_eid][left_lid].append(edge_gid)

        else:
            # Infer edges and allow multiple sub-edges per geometric side (hanging nodes).
            # 1) Collect all nodes along every element side (includes hanging nodes from neighbours)
            side_nodes: List[List[List[int]]] = []
            for corners in self.corner_connectivity:
                per_side: List[List[int]] = []
                for i0, i1 in edge_defs:
                    per_side.append(list(_locate_all_nodes_in_edge(int(corners[i0]), int(corners[i1]))))
                side_nodes.append(per_side)

            # 2) Build a segment map keyed by consecutive node pairs along a side.
            segment_map: Dict[Tuple[int, int], List[Dict[str, Union[int, Tuple[int, ...]]]]] = {}
            for eid, per_side in enumerate(side_nodes):
                for lid, seq in enumerate(per_side):
                    for a, b in zip(seq[:-1], seq[1:]):
                        key = (min(int(a), int(b)), max(int(a), int(b)))
                        segment_map.setdefault(key, []).append(
                            {
                                "eid": eid,
                                "lid": lid,
                                "start": int(a),
                                "end": int(b),
                                "nodes_full": tuple(seq),
                            }
                        )

            elem_edges: List[List[List[int]]] = [[[] for _ in edge_defs] for _ in self.elements_list]

            for edge_gid, (key, owners) in enumerate(segment_map.items()):
                left_owner = owners[0]
                right_owner = owners[1] if len(owners) > 1 else None
                n_start, n_end = left_owner["start"], left_owner["end"]
                normal_vec = self._compute_normal((n_start, n_end))

                union_nodes = set()
                for ow in owners:
                    union_nodes.update(ow["nodes_full"])
                all_nodes_sorted = tuple(_sort_along_edge(n_start, n_end, union_nodes))

                left_nodes = tuple(n for n in left_owner["nodes_full"] if n in elem_node_sets[left_owner["eid"]])
                right_nodes: Tuple[int, ...] = tuple()
                right_lid = None
                right_eid = None
                if right_owner is not None:
                    right_eid = int(right_owner["eid"])
                    right_nodes = tuple(n for n in right_owner["nodes_full"] if n in elem_node_sets[right_eid])
                    right_lid = int(right_owner["lid"])
                    self._neighbors[left_owner["eid"]].append(right_eid)
                    self._neighbors[right_eid].append(left_owner["eid"])

                edge_obj = Edge(
                    gid=edge_gid,
                    nodes=(n_start, n_end),
                    left=int(left_owner["eid"]),
                    right=right_eid,
                    normal=normal_vec,
                    all_nodes=all_nodes_sorted,
                    lid=int(left_owner["lid"]),
                    right_lid=right_lid,
                    left_nodes=left_nodes,
                    right_nodes=right_nodes,
                )
                self.edges_list.append(edge_obj)
                self._edge_dict[tuple(sorted((n_start, n_end)))] = edge_obj

                elem_edges[int(left_owner["eid"])][int(left_owner["lid"])].append(edge_gid)
                if right_owner is not None:
                    elem_edges[right_eid][right_lid].append(edge_gid)

        # Step 4: Populate per-element edge lists and neighbor info
        for eid, elem in enumerate(self.elements_list):
            if self.edges_connectivity is not None:
                side_lists = elem_edges[eid]
            else:
                side_lists = elem_edges[eid]
            elem.edges_by_side = tuple(tuple(lst) for lst in side_lists)
            elem.edge_gid_to_local = {gid: lid for lid, lst in enumerate(side_lists) for gid in lst}
            # Backward-compatibility: store the first edge per side
            elem.edges = tuple(lst[0] if lst else -1 for lst in side_lists)
            # Neighbour mapping (first neighbour if multiple)
            elem.neighbors = {}
            for lid, lst in enumerate(side_lists):
                nb = None
                for gid in lst:
                    eobj = self.edges_list[gid]
                    other = eobj.right if eobj.left == eid else eobj.left
                    if other is not None:
                        nb = other
                        break
                elem.neighbors[lid] = nb
            # Deduplicate neighbours list
            if self._neighbors[eid]:
                self._neighbors[eid] = list(dict.fromkeys(self._neighbors[eid]))

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

    def classify_elements(self, level_set, tol=SIDE.tol):
        """
        Classify each element as 'inside', 'outside', or 'cut'.
        Sets element.tag accordingly and returns indices for each class.
        """
        # Assumes level_set has a method evaluate_on_nodes(mesh)
        phi_nodes = level_set.evaluate_on_nodes(self)
        elem_phi_nodes = phi_nodes[self.corner_connectivity]
        # phi_cent = self._phi_on_centroids(level_set)

        # min_phi = np.minimum(elem_phi_nodes.min(axis=1), phi_cent)
        # max_phi = np.maximum(elem_phi_nodes.max(axis=1), phi_cent)

        min_phi = elem_phi_nodes.min(axis=1)
        max_phi = elem_phi_nodes.max(axis=1)
        
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

    def classify_elements_multi(self, level_sets, tol=SIDE.tol):
        """Classifies elements against multiple level sets."""
        return {idx: self.classify_elements(ls, tol) for idx, ls in enumerate(level_sets)}

    def classify_edges(self, level_set, tol=SIDE.tol):
        """
        Classify edges as 'interface' or 'ghost' based on element tags.
        """
        phi_nodes = level_set.evaluate_on_nodes(self)
        for edge in self.edges_list:
            if edge.right is None:
                # Boundary edge: KEEP whatever tag the mesh generator gave
                # (e.g., 'left', 'right', 'top', 'bottom' or 'boundary').
                # If you also want to auto-fill when missing, you can set a
                # default like: edge.tag = edge.tag or 'boundary'
                continue

            # Interior edge -> safe to reset and classify
            edge.tag = ''
            n0, n1 = edge.nodes
            p0, p1 = phi_nodes[n0], phi_nodes[n1]
            

            # --- Classification for INTERIOR edges based on element tags ---
            if edge.right is not None:
                left_tag = self.elements_list[edge.left].tag
                right_tag = self.elements_list[edge.right].tag
                tags = {left_tag, right_tag}

                # Prioritize 'interface' if crossed (level set lies on edge)
                if p0 * p1 < 0:
                    # (i) strict crossing
                    edge.tag = 'interface'
                elif abs(p0) <= tol and abs(p1) <= tol:
                    # (ii) whole edge on interface
                    edge.tag = 'interface'
                elif 'cut' in tags:
                    # This logic might be flawed if both elements are 'cut'.
                    # Assuming one is 'cut' and the other is not.
                    non_cut_tags = [t for t in tags if t != 'cut']
                    if non_cut_tags:
                        non_cut = non_cut_tags[0]
                        if non_cut == 'outside':
                            edge.tag = 'ghost_pos'  # Cut and positive side
                        elif non_cut == 'inside':
                            edge.tag = 'ghost_neg'  # Cut and negative side
                    else: # This happens if both tags are 'cut'
                        edge.tag = 'ghost_both'


        # Build and cache BitSets *once* – O(n_edges) total
        tags = np.array([e.tag for e in self.edges_list])
        #Convert np.unique result to a standard Python list
        unique_tags = np.unique(tags).tolist()
        self._edge_bitsets = {t: BitSet(tags == t) for t in unique_tags if t}

        # New: Union bitset for 'ghost' (all ghost_*)
        ghost_pos_bs = self._edge_bitsets.get('ghost_pos', BitSet(np.zeros(len(tags), bool)))
        ghost_neg_bs = self._edge_bitsets.get('ghost_neg', BitSet(np.zeros(len(tags), bool)))
        ghost_both_bs = self._edge_bitsets.get('ghost_both', BitSet(np.zeros(len(tags), bool)))
        self._edge_bitsets['ghost'] = ghost_pos_bs | ghost_neg_bs | ghost_both_bs

    


    def build_interface_segments(self, level_set, tol=SIDE.tol, quadrature_order=2):
        """
        Populate element.interface_pts for every 'cut' element.
        Each entry is a list of interface points, typically [p0, p1].
        """
        # Optional: only needed if you want a cheap endpoint precheck for p==1
        phi_nodes = level_set.evaluate_on_nodes(self)  # φ at mesh corner nodes

        # detect discrete order p of the FE level set if available (default 1)
        p = 1
        if hasattr(level_set, "dh") and hasattr(level_set, "field"):
            try:
                p = int(level_set.dh.mixed_element._field_orders[level_set.field])
            except Exception:
                p = 1

        for elem in self.elements_list:
            if elem.tag != 'cut':
                elem.interface_pts = []
                continue

            pts = []

            # LOCAL edges in canonical order (0..2 tri, 0..3 quad)
            nloc_edges = 3 if len(elem.corner_nodes) == 3 else 4
            for l_edge in range(nloc_edges):

                # --- DO NOT SKIP on endpoint signs for p>=2 ---
                # Only keep the fast skip for strictly p==1 (linear along edge)
                if p == 1:
                    # get endpoints for a quick sign check
                    e_gid = elem.edges[l_edge]
                    e = self.edge(e_gid)
                    n0, n1 = e.nodes
                    phi0, phi1 = phi_nodes[n0], phi_nodes[n1]
                    if (phi0 * phi1 > 0.0) and (abs(phi0) > tol and abs(phi1) > tol):
                        continue

                # robust P^n root(s) on this *local* edge
                roots = edge_root_pn(level_set, self, int(elem.id), int(l_edge), tol=tol)
                if not roots:
                    continue
                # roots may have 1 (crossing) or 2 points (whole edge on {φ=0})
                for P in roots:
                    pts.append(np.asarray(P, float))

            # de-dup (φ=0 through a vertex)
            unique_pts = []
            for pnt in pts:
                if not any(np.linalg.norm(pnt - up) < tol for up in unique_pts):
                    unique_pts.append(pnt)

            # keep exactly two points: choose the longest chord across candidates
            if len(unique_pts) >= 2:
                P = np.asarray(unique_pts, float)
                d2 = ((P[:, None, :] - P[None, :, :])**2).sum(axis=2)
                i, j = divmod(int(d2.argmax()), d2.shape[1])
                pts2 = [tuple(P[i]), tuple(P[j])]
                pts2.sort(key=lambda Q: (Q[0], Q[1]))
                elem.interface_pts = pts2
            else:
                elem.interface_pts = []


    def edge_bitset(self, tag: str) -> BitSet:
        """Return cached BitSet of edges with the given tag."""
        cache = getattr(self, "_edge_bitsets", None)
        if cache is not None and tag in cache:          # fast path
            return cache[tag]

        # --- recompute on the fly (O(n_edges)) ---------------------------
        mask = np.fromiter((e.tag == tag for e in self.edges_list), bool)
        return BitSet(mask)

    def element_bitset(self, tag: str) -> BitSet:
        cache = getattr(self, "_elem_bitsets", None)
        if cache is not None and tag in cache:
            return cache[tag]
        mask = np.fromiter((el.tag == tag for el in self.elements_list), bool)
        return BitSet(mask)

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
        return np.sqrt(self.areas_list[elem_id])
    def face_char_length(self, left_eid: int | None, right_eid: int | None) -> float:
        hs = []
        for eid in (left_eid, right_eid):
            if eid is None: continue
            a = float(self.areas_list[int(eid)])
            if a > 0.0: hs.append(np.sqrt(a))
        if hs: return float(min(hs))
        pos = self.areas_list[self.areas_list > 0.0]
        return float(np.sqrt(pos.min())) if pos.size else 1.0
    def neighbors(self) -> List[int]:
        return self._neighbors

    def edge(self, edge_id: int) -> 'Edge':
        """Return the Edge object corresponding to a global `edge_id`."""
        if not 0 <= edge_id < len(self.edges_list):
            raise IndexError(f"Edge ID {edge_id} out of range.")
        return self.edges_list[edge_id]

    def tag_boundary_edges(self, tag_functions: Dict[str, Callable[[float, float], bool]]):
        """
        Tag every *boundary* edge ( `edge.right is None` ) and – **new** –
        build / refresh the private `_edge_bitsets` cache so that

            >>> mesh.edge_bitset('right_wall')

        is an **O(1)** dictionary lookup instead of a fresh scan.
        """
        n_edges          = len(self.edges_list)
        tag_masks        = {t: np.zeros(n_edges, bool) for t in tag_functions}
        for e in self.edges_list:
            if e.right is not None:                  # interior → skip
                continue
            mpx, mpy   = self.nodes_x_y_pos[list(e.nodes)].mean(axis=0)
            for tag, locator in tag_functions.items():
                if locator(mpx, mpy):
                    e.tag            = tag
                    tag_masks[tag][e.gid] = True
                    break

        if not hasattr(self, "_edge_bitsets"):
            self._edge_bitsets = {}
        self._edge_bitsets.update({tag: BitSet(mask) for tag, mask in tag_masks.items()})
        # Keep a copy of the locator map so downstream code (e.g. DOF tagging)
        # can use the same geometric tests without re-supplying them manually.
        self._boundary_locators = dict(tag_functions)
    
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
