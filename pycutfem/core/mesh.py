import math
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
                 poly_order: int = 1,
                 deduplicate_nodes: bool = False):
        """
        Initializes the mesh and builds its topology.

        Parameters
        ----------
        deduplicate_nodes
            If True, merge nearly coincident nodes (within a small tolerance)
            before building topology. This can be helpful for meshes with
            accidental duplicate coordinates, but **must** be disabled for
            composite/overlapping meshes where coincident coordinates are
            intentional and represent distinct components.
        """
        self.edges_connectivity: np.ndarray = edges_connectivity
        self.element_type = element_type
        self.poly_order = poly_order
        self.nodes_list: List['Node'] = nodes
        self.nodes_x_y_pos = np.array([[n.x, n.y] for n in self.nodes_list], dtype=float)
        self.nodes = np.array([n.id for n in self.nodes_list])
        self.elements_connectivity: np.ndarray = element_connectivity
        self.corner_connectivity: np.ndarray = elements_corner_nodes
        self._char_length = float(np.ptp(self.nodes_x_y_pos, axis=0).max() or 1.0)
        if bool(deduplicate_nodes):
            self._deduplicate_nodes()
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

    def _deduplicate_nodes(self, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> None:
        """
        Merge nearly coincident nodes (within a small absolute/relative tolerance)
        before rebuilding topology so shared edges line up exactly.
        """
        coords = self.nodes_x_y_pos
        span = float(np.ptp(coords, axis=0).max() or 1.0)
        snap = max(abs_tol, rel_tol * span)
        scale = 1.0 / snap

        key_to_new: Dict[Tuple[int, int], int] = {}
        mapping = np.empty(len(coords), dtype=int)
        new_nodes: List['Node'] = []

        for old_id, (x, y) in enumerate(coords):
            key = (int(round(x * scale)), int(round(y * scale)))
            new_id = key_to_new.get(key)
            if new_id is None:
                new_id = len(new_nodes)
                key_to_new[key] = new_id
                tag = getattr(self.nodes_list[old_id], "tag", None)
                new_nodes.append(Node(new_id, float(x), float(y), tag))
            mapping[old_id] = int(new_id)

        if len(new_nodes) == len(coords):
            return

        # Remap connectivity to the deduplicated node ids
        self.elements_connectivity = mapping[self.elements_connectivity]
        self.corner_connectivity = mapping[self.corner_connectivity]
        if self.edges_connectivity is not None:
            self.edges_connectivity = mapping[self.edges_connectivity]

        self.nodes_list = new_nodes
        self.nodes_x_y_pos = np.array([[n.x, n.y] for n in self.nodes_list], dtype=float)
        self.nodes = np.array([n.id for n in self.nodes_list])
        self._char_length = float(np.ptp(self.nodes_x_y_pos, axis=0).max() or 1.0)

    def _build_topology(self):
        """
        Builds the full mesh topology: Elements, Edges, and Neighbors.
        Respects a provided ``edges_connectivity`` if available; otherwise
        infers edges from the polygonal corner connectivity.
        """
        edge_defs = self._EDGE_TABLE[self.element_type]
        char_len = float(getattr(self, "_char_length", 1.0) or 1.0)

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

        def _edge_nodes_from_element_side(eid: int, lid: int) -> Tuple[int, ...]:
            """
            Return the nodes on local edge `lid` of element `eid` in the canonical
            CCW direction given by `_EDGE_TABLE` and the element's local node layout.
            """
            if int(getattr(self, "poly_order", 1) or 1) <= 1:
                return ()
            try:
                lattice = np.asarray(self.elements_connectivity[int(eid)], dtype=int)
            except Exception:
                return ()
            if self.element_type == "quad":
                n_lat = int(self.poly_order) + 1
                if lattice.ndim != 1 or lattice.size != n_lat * n_lat:
                    return ()
                grid = lattice.reshape(n_lat, n_lat)
                if int(lid) == 0:
                    seq = grid[0, :]  # bottom: bl -> br
                elif int(lid) == 1:
                    seq = grid[:, -1]  # right: br -> tr
                elif int(lid) == 2:
                    seq = grid[-1, ::-1]  # top: tr -> tl
                elif int(lid) == 3:
                    seq = grid[::-1, 0]  # left: tl -> bl
                else:
                    return ()
                return tuple(int(n) for n in np.asarray(seq).ravel())
            return ()

        def _locate_all_nodes_in_edge(
            vA: int,
            vB: int,
            edge_nodes_hint: Optional[Tuple[int, ...]] = None,
            candidates: Optional[Iterable[int]] = None,
        ) -> Tuple[int, ...]:
            """
            Return every node that lies on the geometric edge vA→vB.
            Vectorized to avoid Python loops (hot when rebuilding topology
            after adaptive refinement).
            """
            p0 = self.nodes_x_y_pos[vA]
            p1 = self.nodes_x_y_pos[vB]
            d = p1 - p0
            L2 = float(np.dot(d, d))
            if L2 < 1e-30:
                return (int(vA), int(vB))
            L = math.sqrt(L2)
            # Base tolerance for straight edges.
            span_ref = max(L, char_len)
            dist_tol = max(SIDE.tol * L, 5e-8 * span_ref, 1e-10)
            # If this element provides higher-order edge nodes, inflate the tolerance
            # using the observed sagitta so we still recover the intended edge node set
            # even when nodes are not perfectly collinear (e.g. isoparametric boundaries
            # or small geometric drift from refinement).
            if edge_nodes_hint:
                coords_hint = self.nodes_x_y_pos[list(edge_nodes_hint)]
                cross_hint = np.abs((coords_hint[:, 0] - p0[0]) * d[1] - (coords_hint[:, 1] - p0[1]) * d[0])
                sagitta = float(np.max(cross_hint / L)) if cross_hint.size else 0.0
                dist_tol = max(dist_tol, 1.25 * sagitta + 1e-9 * span_ref)
            proj_tol = max(dist_tol * L, 1e-12 * L2, 1e-8 * span_ref * L)

            if candidates is None:
                rel = self.nodes_x_y_pos - p0  # (n_nodes, 2)
                cross = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
                dot = rel @ d
                # cross/|d| is the perpendicular distance from the line
                mask = (cross <= dist_tol * L) & (dot >= -proj_tol) & (dot <= L2 + proj_tol)
                cand = np.nonzero(mask)[0]
                if cand.size == 0:
                    return (int(vA), int(vB))

                proj = dot[cand]
                order = np.argsort(proj)
                ordered = cand[order]
                return tuple(int(idx) for idx in ordered)

            cand_arr = np.fromiter((int(n) for n in candidates), dtype=int)
            if cand_arr.size == 0:
                return (int(vA), int(vB))

            rel = self.nodes_x_y_pos[cand_arr] - p0  # (n_cand, 2)
            cross = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
            dot = rel @ d
            mask = (cross <= dist_tol * L) & (dot >= -proj_tol) & (dot <= L2 + proj_tol)
            cand_f = cand_arr[mask]
            if cand_f.size == 0:
                return (int(vA), int(vB))

            proj = dot[mask]
            order = np.argsort(proj)
            ordered = cand_f[order]
            return tuple(int(idx) for idx in ordered)

        def _sort_along_edge(n_start: int, n_end: int, nodes: Iterable[int]) -> List[int]:
            p0 = self.nodes_x_y_pos[n_start]
            p1 = self.nodes_x_y_pos[n_end]
            d = p1 - p0
            if float(np.dot(d, d)) < 1e-30:
                return sorted(nodes)
            nodes_arr = np.fromiter((int(n) for n in nodes), dtype=int)
            rel = self.nodes_x_y_pos[nodes_arr] - p0
            proj = rel @ d
            return [int(n) for n in nodes_arr[np.argsort(proj)]]

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

                # Determine the full node set on this edge.
                # Prefer nodes from the owning element(s) to avoid accidentally pulling in
                # nodes from other disconnected components that share coordinates (e.g.
                # composite meshes for non-matching coupling).
                geom_nodes: Tuple[int, ...] = tuple(int(n) for n in ordered_nodes)
                if left_eid is not None:
                    cand: set[int] = set(elem_node_sets[int(left_eid)])
                    if right_eid is not None:
                        cand.update(elem_node_sets[int(right_eid)])
                    geom_nodes = _locate_all_nodes_in_edge(
                        n_start,
                        n_end,
                        tuple(int(n) for n in ordered_nodes),
                        candidates=cand,
                    )

                geom_set = set(int(n) for n in geom_nodes)
                for nd in ordered_nodes:
                    if int(nd) not in geom_set:
                        geom_nodes = tuple(list(geom_nodes) + [int(nd)])
                        geom_set.add(int(nd))
                edge_nodes_from_elem = (
                    _edge_nodes_from_element_side(left_eid, left_lid)
                    if left_eid is not None and left_lid is not None
                    else ()
                )
                if edge_nodes_from_elem:
                    geom_nodes = edge_nodes_from_elem

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
            edge_hint_indices = None
            if self.element_type == "quad":
                n_lat = self.poly_order + 1
                edge_hint_indices = (
                    [i for i in range(n_lat)],  # bottom
                    [k * n_lat + (n_lat - 1) for k in range(n_lat)],  # right
                    [n_lat * (n_lat - 1) + i for i in range(n_lat)],  # top
                    [k * n_lat for k in range(n_lat)],  # left
                )

            side_nodes: List[List[List[int]]] = []
            for eid, corners in enumerate(self.corner_connectivity):
                per_side: List[List[int]] = []
                lattice = self.elements_connectivity[eid]
                for lid, (i0, i1) in enumerate(edge_defs):
                    hint = None
                    if edge_hint_indices is not None:
                        hint = tuple(int(lattice[idx]) for idx in edge_hint_indices[lid])
                    per_side.append(list(_locate_all_nodes_in_edge(int(corners[i0]), int(corners[i1]), hint)))
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
        from pycutfem.fem import transform
        from pycutfem.core.levelset import phi_eval as _phi_eval

        phi_nodes = level_set.evaluate_on_nodes(self)
        # Use *all* element nodes (not just corners) so higher-order meshes/level-sets
        # cannot miss interior sign changes (a common source of spurious "aligned" edges).
        elem_phi_nodes = phi_nodes[self.elements_connectivity]
        try:
            phi_cent = self._phi_on_centroids(level_set)
            phi_samples = np.concatenate([elem_phi_nodes, phi_cent[:, None]], axis=1)
        except Exception:
            phi_samples = elem_phi_nodes

        has_neg = phi_samples < -tol
        has_pos = phi_samples > tol

        any_neg = has_neg.any(axis=1)
        any_pos = has_pos.any(axis=1)

        # True sign change → genuine cut element
        cut_mask = any_neg & any_pos

        # Only negative → inside
        inside_mask = any_neg & ~any_pos

        # Only positive → outside
        outside_mask = any_pos & ~any_neg

        # Narrow φ≈0 band (all samples ~0): treat as cut for safety
        cut_mask |= ~(inside_mask | outside_mask | cut_mask)

        # ------------------------------------------------------------------
        # Sliver / grazing robustness:
        # Even if all element nodes are on the same side, higher-order/analytic
        # level sets can cross an element in a tiny corner region (two roots on
        # a single edge). Detect this by probing edge roots on a narrow band
        # around Γ, but avoid re-tagging fully aligned interface facets.
        # ------------------------------------------------------------------
        try:
            h_elem = np.sqrt(np.asarray(self.areas_list, float))
        except Exception:
            h_elem = np.full(len(self.elements_list), float(getattr(self, "_char_length", 1.0) or 1.0))
        min_abs = np.min(np.abs(phi_samples), axis=1)
        cand = np.where((inside_mask | outside_mask) & (min_abs <= h_elem))[0]
        nloc_edges = 4 if self.element_type == "quad" else 3
        for eid in cand:
            if cut_mask[eid]:
                continue
            for l_edge in range(nloc_edges):
                roots = edge_root_pn(level_set, self, int(eid), int(l_edge), tol=tol)
                if not roots:
                    continue

                # Ignore pure vertex hits: roots at edge endpoints only.
                c0, c1 = self._EDGE_TABLE[self.element_type][l_edge]
                n0 = int(self.elements_list[int(eid)].corner_nodes[c0])
                n1 = int(self.elements_list[int(eid)].corner_nodes[c1])
                P0 = self.nodes_x_y_pos[n0]
                P1 = self.nodes_x_y_pos[n1]
                eps_end = max(10.0 * float(tol), 1.0e-8 * float(h_elem[eid]))
                has_interior = False
                for R in roots:
                    R = np.asarray(R, float)
                    if (np.linalg.norm(R - P0) > eps_end) and (np.linalg.norm(R - P1) > eps_end):
                        has_interior = True
                        break
                if not has_interior:
                    continue

                # Skip fully aligned edges (φ=0 along the whole edge): those are
                # handled by edge tagging and the aligned-interface assembler.
                if self.element_type == "quad":
                    if l_edge == 0:
                        xi_eta = (0.0, -1.0)
                    elif l_edge == 1:
                        xi_eta = (1.0, 0.0)
                    elif l_edge == 2:
                        xi_eta = (0.0, 1.0)
                    else:
                        xi_eta = (-1.0, 0.0)
                else:  # tri
                    if l_edge == 0:
                        xi_eta = (0.5, 0.0)
                    elif l_edge == 1:
                        xi_eta = (0.5, 0.5)
                    else:
                        xi_eta = (0.0, 0.5)

                x_mid = transform.x_mapping(self, int(eid), (float(xi_eta[0]), float(xi_eta[1])))
                phi_mid = float(_phi_eval(level_set, x_mid, eid=int(eid), xi_eta=xi_eta, mesh=self))
                if abs(phi_mid) <= tol:
                    continue

                # Genuine intersection missed by nodal sampling → mark cut
                cut_mask[eid] = True
                inside_mask[eid] = False
                outside_mask[eid] = False
                break

        inside_inds  = np.where(inside_mask)[0]
        outside_inds = np.where(outside_mask)[0]
        cut_inds     = np.where(cut_mask)[0]

        for eid in inside_inds:  self.elements_list[eid].tag = "inside"
        for eid in outside_inds: self.elements_list[eid].tag = "outside"
        for eid in cut_inds:     self.elements_list[eid].tag = "cut"

        tags_el = np.array([e.tag for e in self.elements_list])
        self._elem_bitsets = {t: BitSet(tags_el == t) for t in np.unique(tags_el)}
        tok = getattr(level_set, "cache_token", None)
        self._ls_elements_key = ("token", tok, float(tol)) if tok is not None else ("objid", int(id(level_set)), float(tol))
        return inside_inds, outside_inds, cut_inds

    def classify_elements_multi(self, level_sets, tol=SIDE.tol):
        """Classifies elements against multiple level sets."""
        return {idx: self.classify_elements(ls, tol) for idx, ls in enumerate(level_sets)}

    def classify_edges(self, level_set, tol=SIDE.tol):
        """
        Classify edges as 'interface' or 'ghost' based on element tags.
        """
        try:
            prev_tol = getattr(level_set, "edge_tol", None)
            if prev_tol is None or float(tol) > float(prev_tol):
                setattr(level_set, "edge_tol", float(tol))
        except Exception:
            pass
        phi_nodes = level_set.evaluate_on_nodes(self)
        
        for edge in self.edges_list:
            if edge.right is None:
                continue

            edge.tag = ''
            # Detect fully-aligned interface facets using *all* nodes on the edge.
            # For higher-order meshes, endpoints alone are insufficient: the edge can
            # "hit" φ=0 at vertices while remaining strictly on one side in-between.
            edge_nodes = getattr(edge, "all_nodes", None) or edge.nodes
            edge_nodes = tuple(int(n) for n in edge_nodes)
            phi_edge = phi_nodes[list(edge_nodes)]
            edge_aligned = bool(np.all(np.abs(phi_edge) <= tol))
            left_el = self.elements_list[edge.left]
            right_el = self.elements_list[edge.right]
            left_tag = left_el.tag
            right_tag = right_el.tag
            tags = {left_tag, right_tag}

            # 1. Interface edges: only when they separate inside/outside elements.
            if left_tag in {"inside", "outside"} and right_tag in {"inside", "outside"} and left_tag != right_tag:
                if edge_aligned:
                    edge.tag = "interface"
                continue
            
            # 2. Ghost Edge Classification based on Element Tags
            # We only care about stabilization if a Cut element is involved.
            if 'cut' in tags:
                # Case A: Cut + Cut -> Stabilizes Both
                if left_tag == 'cut' and right_tag == 'cut':
                    edge.tag = 'ghost_both'
                
                # Case B: Cut + Outside -> Stabilizes Fluid (Positive)
                elif 'outside' in tags:
                    edge.tag = 'ghost_pos'
                
                # Case C: Cut + Inside -> Stabilizes Solid (Negative)
                elif 'inside' in tags:
                    edge.tag = 'ghost_neg'
            
            # Note: Edges between 'inside' and 'outside' (without 'cut') 
            # are handled by the 'interface' check above or remain standard boundaries.

        # 3. Build BitSets (same as before)
        tags_arr = np.array([e.tag for e in self.edges_list])
        unique_tags = np.unique(tags_arr).tolist()
        self._edge_bitsets = {t: BitSet(tags_arr == t) for t in unique_tags if t}

        # Union for solver sets
        ghost_pos_bs = self._edge_bitsets.get('ghost_pos', BitSet(np.zeros(len(tags_arr), bool)))
        ghost_neg_bs = self._edge_bitsets.get('ghost_neg', BitSet(np.zeros(len(tags_arr), bool)))
        ghost_both_bs = self._edge_bitsets.get('ghost_both', BitSet(np.zeros(len(tags_arr), bool)))
        interface_bs = self._edge_bitsets.get('interface', BitSet(np.zeros(len(tags_arr), bool)))
        
        self._edge_bitsets['ghost_pos'] = ghost_pos_bs - interface_bs
        self._edge_bitsets['ghost_neg'] = ghost_neg_bs - interface_bs
        self._edge_bitsets['ghost_both'] = ghost_both_bs - interface_bs
        self._edge_bitsets['ghost'] = (ghost_pos_bs | ghost_neg_bs | ghost_both_bs) - interface_bs
        tok = getattr(level_set, "cache_token", None)
        self._ls_edges_key = ("token", tok, float(tol)) if tok is not None else ("objid", int(id(level_set)), float(tol))
    
    # def classify_edges(self, level_set, tol=SIDE.tol):
    #     """
    #     Classify edges as 'interface' or 'ghost' based on element tags.
    #     """
    #     from pycutfem.core.levelset import phi_eval
    #     phi_nodes = level_set.evaluate_on_nodes(self)
    #     for edge in self.edges_list:
    #         if edge.right is None:
    #             # Boundary edge: KEEP whatever tag the mesh generator gave
    #             # (e.g., 'left', 'right', 'top', 'bottom' or 'boundary').
    #             # If you also want to auto-fill when missing, you can set a
    #             # default like: edge.tag = edge.tag or 'boundary'
    #             continue

    #         # Interior edge -> safe to reset and classify
    #         edge.tag = ''
    #         n0, n1 = edge.nodes
    #         # mid_xy = 0.5 * (self.nodes_x_y_pos[n0] + self.nodes_x_y_pos[n1])
    #         p0, p1 = phi_nodes[n0], phi_nodes[n1]
    #         # pm = phi_eval(level_set, mid_xy, mesh=self)
            

    #         # --- Classification for INTERIOR edges based on element tags ---
    #         if edge.right is not None:
    #             left_tag = self.elements_list[edge.left].tag
    #             right_tag = self.elements_list[edge.right].tag
    #             tags = {left_tag, right_tag}

    #             # Prioritize 'interface' if crossed (level set lies on edge)
    #             if p0 * p1 < -tol:
    #                 # (i) strict crossing
    #                 edge.tag = 'ghost_both'
    #             elif abs(p0) <= tol and abs(p1) <= tol:
    #                 # (ii) whole edge on interface
    #                 edge.tag = 'interface'
    #             elif 'cut' in tags:
    #                 # This logic might be flawed if both elements are 'cut'.
    #                 # Assuming one is 'cut' and the other is not.
    #                 non_cut_tags = [t for t in tags if t != 'cut']
    #                 if non_cut_tags:
    #                     non_cut = non_cut_tags[0]
    #                     if non_cut == 'outside':
    #                         edge.tag = 'ghost_pos'  # Cut and positive side
    #                     elif non_cut == 'inside':
    #                         edge.tag = 'ghost_neg'  # Cut and negative side
    #                 else: # This happens if both tags are 'cut'
    #                     print(f"[warning!] p0={p0}, p1={p1}, tags={tags}")
    #                     if (p0 <=tol and p1 >=tol) or (p1 <=tol and p0 >=tol):
    #                         edge.tag = 'ghost_pos'
    #                     elif (p0 <=-tol and p1 <=tol) or (p1 <=-tol and p0 <=tol):
    #                         edge.tag = 'ghost_neg'
    #                     else:
    #                         edge.tag = 'ghost_both'


    #     # Build and cache BitSets *once* – O(n_edges) total
    #     tags = np.array([e.tag for e in self.edges_list])
    #     #Convert np.unique result to a standard Python list
    #     unique_tags = np.unique(tags).tolist()
    #     self._edge_bitsets = {t: BitSet(tags == t) for t in unique_tags if t}

    #     # New: Union bitset for 'ghost' (all ghost_*)
    #     ghost_pos_bs = self._edge_bitsets.get('ghost_pos', BitSet(np.zeros(len(tags), bool)))
    #     ghost_neg_bs = self._edge_bitsets.get('ghost_neg', BitSet(np.zeros(len(tags), bool)))
    #     ghost_both_bs = self._edge_bitsets.get('ghost_both', BitSet(np.zeros(len(tags), bool)))
    #     self._edge_bitsets['ghost'] = ghost_pos_bs | ghost_neg_bs | ghost_both_bs

    


    def build_interface_segments(self, level_set, tol=SIDE.tol, quadrature_order=2):
        """
        Populate element.interface_pts for every 'cut' element.
        Each entry is a list of interface points, typically [p0, p1].
        """
        from pycutfem.core.levelset import phi_eval as _phi_eval
        from pycutfem.fem import transform

        def _dedup(points: list[np.ndarray]) -> list[np.ndarray]:
            out: list[np.ndarray] = []
            for pnt in points:
                if not any(np.linalg.norm(pnt - q) < tol for q in out):
                    out.append(pnt)
            return out

        def _ordered_polyline(points: list[np.ndarray]) -> list[np.ndarray]:
            """Return a simple polyline through all points (small n, brute force)."""
            if len(points) <= 2:
                return points
            pts = [np.asarray(p, float) for p in points]
            P = np.stack(pts, axis=0)
            d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
            i, j = divmod(int(d2.argmax()), d2.shape[1])  # farthest pair → endpoints
            rest = [k for k in range(len(pts)) if k not in (i, j)]
            best_order = None
            best_len = np.inf
            # brute-force the small set of interior permutations (interfaces rarely have many nodes)
            import itertools
            for perm in itertools.permutations(rest):
                order = [i, *perm, j]
                length = sum(
                    float(np.linalg.norm(P[order[k + 1]] - P[order[k]]))
                    for k in range(len(order) - 1)
                )
                if length < best_len:
                    best_len = length
                    best_order = order
            return [pts[k] for k in best_order] if best_order is not None else pts

        def _edge_ref_coords(local_edge: int, t: float, *, etype: str) -> tuple[float, float]:
            """Reference (xi, eta) on a local edge parameterised by t∈[0,1]."""
            if etype == "quad":
                if   local_edge == 0: return (2*t - 1.0, -1.0)
                if   local_edge == 1: return (1.0, -1.0 + 2*t)
                if   local_edge == 2: return (1.0 - 2*t, 1.0)
                return (-1.0, 1.0 - 2*t)
            # tri
            if   local_edge == 0: return (t, 0.0)
            if   local_edge == 1: return (1.0 - t, t)
            return (0.0, 1.0 - t)

        def _unit(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, float).reshape(-1)
            if v.size != 2:
                v = v[:2]
            nrm = float(np.linalg.norm(v))
            if nrm <= 1e-30:
                return np.zeros(2, dtype=float)
            return (v / nrm).astype(float)

        def _in_element(eid: int, P: np.ndarray, eps: float = 1e-10) -> bool:
            try:
                xi, eta = transform.inverse_mapping(self, int(eid), np.asarray(P, float))
            except Exception:
                return False
            if self.element_type == "quad":
                return (abs(float(xi)) <= 1.0 + eps) and (abs(float(eta)) <= 1.0 + eps)
            return (float(xi) >= -eps) and (float(eta) >= -eps) and (float(xi) + float(eta) <= 1.0 + eps)

        def _prune_shared_edge_segments(elem, segs: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
            """Avoid double-counting when Γ lies on a shared element edge segment.

            If a segment lies on an interior mesh edge and *both* adjacent elements
            are tagged 'cut', keep it only for the global-edge 'left' owner.
            """
            if not segs:
                return segs

            def _on_segment_line(P: np.ndarray, A: np.ndarray, B: np.ndarray, *, tol_line: float) -> bool:
                v = B - A
                Lv = float(np.linalg.norm(v))
                if Lv <= 1e-30:
                    return False
                w = P - A
                # distance to line via 2D cross product magnitude / |v|
                d = abs(float(v[0] * w[1] - v[1] * w[0])) / Lv
                if d > tol_line:
                    return False
                t = float(np.dot(w, v)) / (Lv * Lv)
                return (-1e-12 <= t <= 1.0 + 1e-12)

            nloc_edges = 4 if len(elem.corner_nodes) == 4 else 3
            cn = self.nodes_x_y_pos[list(elem.corner_nodes)]
            h_loc = float(np.max(np.linalg.norm(np.roll(cn, -1, axis=0) - cn, axis=1)))
            tol_line = max(1e-10, 1e-10 * h_loc)

            out: list[list[tuple[float, float]]] = []
            for p0, p1 in segs:
                P0 = np.asarray(p0, float)
                P1 = np.asarray(p1, float)
                drop = False

                for l_edge in range(nloc_edges):
                    c0, c1 = self._EDGE_TABLE[self.element_type][l_edge]
                    A = self.nodes_x_y_pos[int(elem.corner_nodes[c0])]
                    B = self.nodes_x_y_pos[int(elem.corner_nodes[c1])]
                    if _on_segment_line(P0, A, B, tol_line=tol_line) and _on_segment_line(P1, A, B, tol_line=tol_line):
                        # Segment lies on this element edge; if it's shared by two cut elements,
                        # keep only on the global edge's 'left' owner.
                        try:
                            gid = int(elem.edges[l_edge])
                        except Exception:
                            gid = None
                        if gid is not None:
                            e = self.edge(gid)
                            if e.right is not None:
                                lt = self.elements_list[int(e.left)].tag
                                rt = self.elements_list[int(e.right)].tag
                                if lt == "cut" and rt == "cut" and int(elem.id) != int(e.left):
                                    drop = True
                        break

                if not drop:
                    out.append([tuple(P0), tuple(P1)])
            return out

        # Optional: only needed if you want a cheap endpoint precheck for p==1
        phi_nodes = level_set.evaluate_on_nodes(self)  # φ at mesh corner nodes

        # detect discrete order p of the FE level set if available (default 1)
        is_fe_ls = hasattr(level_set, "value_on_element") or hasattr(level_set, "values_on_element_many")
        p = 1
        is_fe_backed = False
        if is_fe_ls and hasattr(level_set, "dh") and hasattr(level_set, "field"):
            try:
                p = int(level_set.dh.mixed_element._field_orders[level_set.field])
                is_fe_backed = True
            except Exception:
                p = 1
                is_fe_backed = False
        if not is_fe_backed:
            # Analytic / black-box level sets are not linear along edges, so the
            # endpoint sign shortcut (valid only for FE P1) must be disabled.
            p = max(int(p), 2)

        for elem in self.elements_list:
            if elem.tag != 'cut':
                elem.interface_pts = []
                elem.interface_segments = []
                continue

            pts = []

            # LOCAL edges in canonical order (0..2 tri, 0..3 quad)
            nloc_edges = 3 if len(elem.corner_nodes) == 3 else 4
            for l_edge in range(nloc_edges):

                # --- DO NOT SKIP on endpoint signs for p>=2 ---
                # Only keep the fast skip for strictly p==1 (linear along edge)
                if is_fe_ls and p == 1:
                    # Quick sign check using the *corner* nodes of this side;
                    # elem.edges[l_edge] may be a subdivided segment (hanging nodes).
                    c0, c1 = self._EDGE_TABLE[self.element_type][l_edge]
                    n0 = int(elem.corner_nodes[c0]); n1 = int(elem.corner_nodes[c1])
                    phi0, phi1 = phi_nodes[n0], phi_nodes[n1]
                    if (phi0 * phi1 > 0.0) and (abs(phi0) > tol and abs(phi1) > tol):
                        continue

                # robust P^n root(s) on this *local* edge
                roots = edge_root_pn(level_set, self, int(elem.id), int(l_edge), tol=tol)
                if not roots:
                    # No sign change detected; try a soft snap if φ is very small along the edge
                    e_gid = elem.edges[l_edge]
                    e = self.edge(e_gid)
                    pA, pB = self.nodes_x_y_pos[e.nodes[0]], self.nodes_x_y_pos[e.nodes[1]]
                    h_edge = np.linalg.norm(pB - pA)
                    tol_snap = max(10 * tol, 1e-6 * h_edge)
                    ts = np.linspace(0.0, 1.0, 5)
                    fvals = []
                    for t in ts:
                        xi, eta = _edge_ref_coords(l_edge, float(t), etype=self.element_type)
                        phys = transform.x_mapping(self, int(elem.id), (xi, eta))
                        try:
                            val = _phi_eval(level_set, phys, eid=int(elem.id), xi_eta=(xi, eta), mesh=self)
                        except Exception:
                            val = float(level_set(phys))
                        fvals.append(val)
                    fvals = np.asarray(fvals, float)
                    k = int(np.argmin(np.abs(fvals)))
                    if abs(fvals[k]) <= tol_snap:
                        xi, eta = _edge_ref_coords(l_edge, float(ts[k]), etype=self.element_type)
                        P_soft = transform.x_mapping(self, int(elem.id), (float(xi), float(eta)))
                        pts.append(np.asarray(P_soft, float))
                    continue
                # roots may have 1 (crossing) or 2 points (whole edge on {φ=0})
                for P in roots:
                    pts.append(np.asarray(P, float))

            # de-dup (φ=0 through a vertex)
            unique_pts = _dedup([np.asarray(pnt, float) for pnt in pts])

            # If we still have <2 boundary intersections, try a very conservative
            # interior probe (helps near-tangencies) but do not add extra points
            # when we already have enough to define segments: that can create
            # spurious kinks on piecewise-linear interfaces (boxes).
            if len(unique_pts) < 2:
                try:
                    c = np.asarray(elem.centroid(), float)
                    phi_c = _phi_eval(level_set, c, eid=int(elem.id), mesh=self)
                    if abs(phi_c) <= max(tol, 1e-12):
                        unique_pts.append(c)
                except Exception:
                    pass
                for (i0, i1) in self._EDGE_TABLE[self.element_type]:
                    n0 = int(elem.corner_nodes[i0]); n1 = int(elem.corner_nodes[i1])
                    mid = 0.5 * (self.nodes_x_y_pos[n0] + self.nodes_x_y_pos[n1])
                    try:
                        phi_m = _phi_eval(level_set, mid, eid=int(elem.id), mesh=self)
                    except Exception:
                        try:
                            phi_m = level_set(np.asarray(mid, float))
                        except Exception:
                            phi_m = np.inf
                    if abs(phi_m) <= max(tol, 1e-12):
                        unique_pts.append(np.asarray(mid, float))

                unique_pts = _dedup(unique_pts)

            # Corner/kink fixup: if the interface inside this element is not smooth,
            # two boundary intersections alone may represent an L-shaped polyline
            # (e.g. sharp corners of a box). Detect via large normal change and
            # insert the interior kink point as the intersection of tangents.
            if len(unique_pts) == 2 and hasattr(level_set, "gradient"):
                P0 = np.asarray(unique_pts[0], float)
                P1 = np.asarray(unique_pts[1], float)
                try:
                    n0 = _unit(level_set.gradient(P0))
                    n1 = _unit(level_set.gradient(P1))
                except Exception:
                    n0 = np.zeros(2, float)
                    n1 = np.zeros(2, float)

                if (np.linalg.norm(n0) > 0.0) and (np.linalg.norm(n1) > 0.0):
                    cosang = float(np.clip(np.dot(n0, n1), -1.0, 1.0))
                    # If normals differ a lot, the interface may have a kink.
                    if cosang < 0.75:  # ~ > 41 degrees
                        t0 = np.array([-n0[1], n0[0]], dtype=float)
                        t1 = np.array([-n1[1], n1[0]], dtype=float)
                        A = np.column_stack((t0, -t1))
                        det = float(np.linalg.det(A))
                        if abs(det) > 1.0e-14:
                            try:
                                s, _u = np.linalg.solve(A, (P1 - P0))
                            except Exception:
                                s = None
                            if s is not None:
                                C = P0 + float(s) * t0
                                if _in_element(int(elem.id), C, eps=1e-8):
                                    # Project the intersection onto Γ to robustly handle small
                                    # errors in the tangent intersection (e.g. near-degenerate
                                    # corner situations).
                                    from pycutfem.integration.quadrature import _project_to_levelset

                                    C0 = np.asarray(C, float)
                                    Cg = _project_to_levelset(C0, level_set, mesh=self, eid=int(elem.id), max_steps=6, tol=max(1e-14, float(tol)))
                                    Cg = np.asarray(Cg, float)

                                    # Accept only if projection displacement is tiny relative to h.
                                    cn = self.nodes_x_y_pos[list(elem.corner_nodes)]
                                    edges = np.roll(cn, -1, axis=0) - cn
                                    h_elem = float(np.max(np.linalg.norm(edges, axis=1)))
                                    if (np.linalg.norm(Cg - C0) <= max(1e-10, 0.05 * h_elem)) and _in_element(int(elem.id), Cg, eps=1e-8):
                                        if (np.linalg.norm(Cg - P0) > 1e-12) and (np.linalg.norm(Cg - P1) > 1e-12):
                                            unique_pts.append(Cg)
                                            unique_pts = _dedup(unique_pts)

            npts = len(unique_pts)
            if npts < 2:
                elem.interface_pts = []
                elem.interface_segments = []
                continue

            # 1) The common case: a single connected segment (2 points)
            if npts == 2:
                p0 = tuple(np.asarray(unique_pts[0], float))
                p1 = tuple(np.asarray(unique_pts[1], float))
                segs = _prune_shared_edge_segments(elem, [[p0, p1]])
                elem.interface_segments = segs
                elem.interface_pts = [segs[0][0], segs[0][1]] if segs else []
                continue

            # 2) A kinked polyline (typically 3 points: corner inside the element)
            if npts == 3:
                ordered = _ordered_polyline(unique_pts)
                segments = []
                for k in range(len(ordered) - 1):
                    p0 = tuple(np.asarray(ordered[k], float))
                    p1 = tuple(np.asarray(ordered[k + 1], float))
                    segments.append([p0, p1])
                segments = _prune_shared_edge_segments(elem, segments)
                elem.interface_segments = segments
                elem.interface_pts = [segments[0][0], segments[-1][1]] if segments else []
                continue

            # 3) Potentially multiple disconnected components: pair points into segments.
            # This occurs near sharp corners where two interface branches enter through
            # the same element edge, but the corner itself lies in a neighbour element.
            pts_arr = [np.asarray(pnt, float) for pnt in unique_pts]
            normals = None
            if hasattr(level_set, "gradient"):
                normals = []
                for P in pts_arr:
                    try:
                        normals.append(_unit(level_set.gradient(P)))
                    except Exception:
                        normals.append(np.zeros(2, float))

            def _pair_cost(i: int, j: int) -> float:
                Pi, Pj = pts_arr[i], pts_arr[j]
                d = float(np.linalg.norm(Pj - Pi))
                if normals is None:
                    return d
                ni, nj = normals[i], normals[j]
                if (np.linalg.norm(ni) <= 0.0) or (np.linalg.norm(nj) <= 0.0):
                    return d
                # Prefer pairing points on the same interface branch (similar normals).
                penalty = 1.0 - abs(float(np.dot(ni, nj)))
                return d + 0.25 * penalty

            def _best_pairs(ids: list[int]) -> list[tuple[int, int]] | None:
                best_cost = float("inf")
                best: list[tuple[int, int]] | None = None

                def rec(rem: list[int], acc: list[tuple[int, int]], cost: float) -> None:
                    nonlocal best_cost, best
                    if not rem:
                        if cost < best_cost:
                            best_cost = cost
                            best = list(acc)
                        return
                    i = rem[0]
                    for k in range(1, len(rem)):
                        j = rem[k]
                        c = _pair_cost(i, j)
                        new_cost = cost + c
                        if new_cost >= best_cost:
                            continue
                        acc.append((i, j))
                        nxt = rem[1:k] + rem[k + 1 :]
                        rec(nxt, acc, new_cost)
                        acc.pop()

                rec(list(ids), [], 0.0)
                return best

            # Only apply pairing when we have an even number of points; otherwise fall back.
            if (npts % 2) == 0:
                pairs = _best_pairs(list(range(npts)))
                if pairs:
                    segs = []
                    for i, j in pairs:
                        p0 = tuple(np.asarray(pts_arr[i], float))
                        p1 = tuple(np.asarray(pts_arr[j], float))
                        segs.append([p0, p1])
                    segs = _prune_shared_edge_segments(elem, segs)
                    elem.interface_segments = segs
                    elem.interface_pts = [segs[0][0], segs[0][1]] if segs else []
                    continue

            # Fallback: a single polyline through all points.
            ordered = _ordered_polyline(unique_pts)
            segments = []
            for k in range(len(ordered) - 1):
                p0 = tuple(np.asarray(ordered[k], float))
                p1 = tuple(np.asarray(ordered[k + 1], float))
                segments.append([p0, p1])
            segments = _prune_shared_edge_segments(elem, segments)
            elem.interface_segments = segments
            elem.interface_pts = [segments[0][0], segments[-1][1]] if segments else []

        tok = getattr(level_set, "cache_token", None)
        self._ls_segments_key = ("token", tok, float(tol)) if tok is not None else ("objid", int(id(level_set)), float(tol))

    def edge_bitset(self, tag: str) -> BitSet:
        """Return cached BitSet of edges with the given tag."""
        cache = getattr(self, "_edge_bitsets", None)
        if cache is not None and tag in cache:          # fast path
            return cache[tag]

        # --- recompute on the fly (O(n_edges)) ---------------------------
        mask = np.fromiter((e.tag == tag for e in self.edges_list), bool)
        return BitSet(mask)

    def rebuild_edge_bitsets(self) -> None:
        """
        Rebuild the `_edge_bitsets` cache from current `Edge.tag` values.

        This is useful after manual tag edits (e.g. geometric pruning, UCD tag import,
        mesh refinement) where previously cached BitSets may become stale.
        """
        n_edges = len(self.edges_list)
        if n_edges == 0:
            self._edge_bitsets = {}
            return

        tags_arr = np.asarray([str(getattr(e, "tag", "") or "") for e in self.edges_list], dtype=object)
        unique_tags = np.unique(tags_arr).tolist()
        self._edge_bitsets = {t: BitSet(tags_arr == t) for t in unique_tags if t}

        # Preserve the convenience unions used by CutFEM paths.
        ghost_pos_bs = self._edge_bitsets.get("ghost_pos", BitSet(np.zeros(n_edges, bool)))
        ghost_neg_bs = self._edge_bitsets.get("ghost_neg", BitSet(np.zeros(n_edges, bool)))
        ghost_both_bs = self._edge_bitsets.get("ghost_both", BitSet(np.zeros(n_edges, bool)))
        interface_bs = self._edge_bitsets.get("interface", BitSet(np.zeros(n_edges, bool)))

        self._edge_bitsets["ghost_pos"] = ghost_pos_bs - interface_bs
        self._edge_bitsets["ghost_neg"] = ghost_neg_bs - interface_bs
        self._edge_bitsets["ghost_both"] = ghost_both_bs - interface_bs
        self._edge_bitsets["ghost"] = (ghost_pos_bs | ghost_neg_bs | ghost_both_bs) - interface_bs

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

    def count_tjunction_violations(self, max_ratio: float = 2.0) -> Dict[str, Union[int, List[int], float]]:
        """
        Count element sides that are subdivided more than `max_ratio` times.
        Correctly detects 4:1 mismatches by checking the number of edges per geometric side.
        """
        bad_edges: List[int] = []
        worst_ratio = 0.0

        # (A) edge-based owner mismatch (4:1 etc. across a shared edge)
        for e in self.edges_list:
            if e.left is None or e.right is None:
                continue
            l_cnt, r_cnt, shared_cnt = self._edge_owner_counts(e)
            fine = max(l_cnt, r_cnt)
            coarse = min(l_cnt, r_cnt)
            ratio = float(fine) / float(max(coarse, 1))
            worst_ratio = max(worst_ratio, ratio)
            needs = (fine > max_ratio * max(coarse, 1)) or (shared_cnt < 1)
            if needs:
                bad_edges.append(int(e.gid))

        bad_edges = sorted(list(set(bad_edges)))

        return {
            "count": len(bad_edges),
            "edges": bad_edges,
            "worst_ratio": worst_ratio,
            "count_zero_shared": 0,
        }

    # --- Performance Optimization: Spatial Acceleration ---

    def build_grid_search(self, n_bins: int = 50):
        """Builds a simple spatial hash for fast element lookup."""
        nodes = self.nodes_x_y_pos
        xmin, ymin = nodes.min(axis=0)
        xmax, ymax = nodes.max(axis=0)
        self._grid_mins = np.array([xmin, ymin]) - 1e-4
        self._grid_maxs = np.array([xmax, ymax]) + 1e-4
        self._grid_bins = (n_bins, n_bins)
        self._grid_step = (self._grid_maxs - self._grid_mins) / self._grid_bins
        
        self._grid_buckets = {}
        
        for eid, elem in enumerate(self.elements_list):
            # map element bounding box to buckets
            cn = nodes[list(elem.corner_nodes)]
            ex_min, ey_min = cn.min(axis=0)
            ex_max, ey_max = cn.max(axis=0)
            
            i0 = int((ex_min - self._grid_mins[0]) / self._grid_step[0])
            j0 = int((ey_min - self._grid_mins[1]) / self._grid_step[1])
            i1 = int((ex_max - self._grid_mins[0]) / self._grid_step[0])
            j1 = int((ey_max - self._grid_mins[1]) / self._grid_step[1])
            
            for i in range(max(0, i0), min(n_bins, i1 + 1)):
                for j in range(max(0, j0), min(n_bins, j1 + 1)):
                    self._grid_buckets.setdefault((i, j), []).append(eid)

    def find_owner_element_fast(self, x: np.ndarray, tol: float = 1e-12) -> int:
        """Finds element containing x using grid search (O(1))."""
        if not hasattr(self, '_grid_buckets'):
            self.build_grid_search()
            
        i = int((x[0] - self._grid_mins[0]) / self._grid_step[0])
        j = int((x[1] - self._grid_mins[1]) / self._grid_step[1])
        
        # Check buckets (center and neighbors to be safe against boundary cases)
        candidates = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                key = (i + di, j + dj)
                if key in self._grid_buckets:
                    candidates.extend(self._grid_buckets[key])
        
        # Fallback to global search if bucket empty or point slightly outside
        search_list = candidates if candidates else range(len(self.elements_list))
        
        # Use existing geometry check (needs to be imported or copied, 
        # but typically this is called from levelset which has access to transform)
        # Here we rely on the caller (LevelSet) to do the geometry check, 
        # but since this method is in Mesh, we can just return candidates to iterate.
        # NOTE: To keep it compatible with LevelSet logic, we return the LIST of candidates
        # and let the caller loop.
        return candidates

    def _edge_owner_counts(self, edge: Edge, tol: float = 1.0e-12) -> Tuple[int, int, int]:
        """
        Return (#nodes_on_edge_left, #nodes_on_edge_right, #shared_nodes) using
        geometric tests so detection works even when Edge.left_nodes/right_nodes
        are missing intermediate hanging nodes. Prefer edge.all_nodes when present
        to avoid counting unrelated collinear nodes.
        """
        span = float(getattr(self, "_char_length", 1.0) or 1.0)
        tol_dist = max(tol, 1e-8 * span)
        def _count_from_all_nodes(eid: Optional[int]) -> Tuple[int, set]:
            if eid is None:
                return 0, set()
            elem_nodes = set(self.elements_list[int(eid)].nodes)
            ids = {int(nid) for nid in edge.all_nodes if int(nid) in elem_nodes}
            return len(ids), ids

        # Prefer stored all_nodes when available
        if edge.all_nodes:
            l_cnt, l_ids = _count_from_all_nodes(edge.left)
            r_cnt, r_ids = _count_from_all_nodes(edge.right)
            shared = l_ids.intersection(r_ids)
            if l_cnt or r_cnt:
                return l_cnt, r_cnt, len(shared)

        # Fallback: geometric detection
        p0, p1 = self.nodes_x_y_pos[list(edge.nodes)]
        d = p1 - p0
        L2 = float(np.dot(d, d))
        if L2 < tol_dist * tol_dist:
            return 0, 0, 0

        def _count_nodes(eid: Optional[int]) -> Tuple[int, set]:
            if eid is None:
                return 0, set()
            elem = self.elements_list[int(eid)]
            ids = set()
            for nid in elem.nodes:
                q = self.nodes_x_y_pos[int(nid)]
                cross = abs((q[0] - p0[0]) * d[1] - (q[1] - p0[1]) * d[0])
                if cross > math.sqrt(L2) * tol_dist:
                    continue
                t = np.dot(q - p0, d) / L2
                if -tol_dist <= t <= 1.0 + tol_dist:
                    ids.add(int(nid))
            return len(ids), ids

        l_cnt, l_ids = _count_nodes(edge.left)
        r_cnt, r_ids = _count_nodes(edge.right)
        shared = l_ids.intersection(r_ids)
        return l_cnt, r_cnt, len(shared)
    
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
        n_edges = len(self.edges_list)
        tag_masks = {t: np.zeros(n_edges, bool) for t in tag_functions}

        def _edge_matches(locator, coords: np.ndarray) -> bool:
            if coords.size == 0:
                return False
            if coords.shape[0] >= 3:
                mid = coords[int(coords.shape[0] // 2)]
            else:
                mid = coords.mean(axis=0)
            if locator(float(mid[0]), float(mid[1])):
                return True
            if coords.shape[0] >= 2:
                p0 = coords[0]
                p1 = coords[-1]
                if locator(float(p0[0]), float(p0[1])) and locator(float(p1[0]), float(p1[1])):
                    return True
            for pt in coords:
                if not locator(float(pt[0]), float(pt[1])):
                    return False
            return True

        for e in self.edges_list:
            if e.right is not None:  # interior → skip
                continue

            edge_nodes = getattr(e, "all_nodes", None) or e.nodes
            coords = self.nodes_x_y_pos[list(edge_nodes)]
            for tag, locator in tag_functions.items():
                if _edge_matches(locator, coords):
                    e.tag = tag
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

    def refine_uniform(self, levels: int = 1) -> "Mesh":
        """
        Uniformly refine a quad mesh by regular 1→4 subdivision.

        - Produces a conforming mesh (no hanging nodes) by refining **all** cells
          each level (deal.II-style global refinement).
        - Child elements inherit their parent element tags.
        - Child boundary edges inherit their parent edge tags.
        """
        if levels <= 0:
            return self
        mesh = self
        for _ in range(int(levels)):
            mesh = mesh._refine_uniform_once()
        return mesh

    def _refine_uniform_once(self) -> "Mesh":
        if self.element_type != "quad":
            raise NotImplementedError("Uniform refinement is implemented for quad meshes only.")
        if int(getattr(self, "poly_order", 1)) != 1:
            raise NotImplementedError("Uniform refinement currently supports poly_order=1 geometry only.")
        if self.corner_connectivity is None:
            raise ValueError("Mesh.corner_connectivity is required for refinement.")

        coords_old = np.asarray(self.nodes_x_y_pos, dtype=float)
        n_old_nodes = len(self.nodes_list)

        # Preserve parent edge tags for propagation.
        parent_edge_tags: Dict[Tuple[int, int], str] = {}
        for e in self.edges_list:
            tag = str(getattr(e, "tag", "") or "")
            if not tag:
                continue
            a, b = (int(e.nodes[0]), int(e.nodes[1]))
            parent_edge_tags[(min(a, b), max(a, b))] = tag

        # Start new node list with existing nodes (keep Node.tag if present).
        new_nodes: List["Node"] = []
        for nid, (x, y) in enumerate(coords_old):
            tag = getattr(self.nodes_list[nid], "tag", None)
            new_nodes.append(Node(int(nid), float(x), float(y), tag))

        edge_to_mid: Dict[Tuple[int, int], int] = {}
        mid_info: Dict[int, Tuple[int, int, str]] = {}  # mid_id -> (a,b,parent_tag)

        def _midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            mid = edge_to_mid.get(key)
            if mid is not None:
                return int(mid)
            xa, ya = coords_old[key[0]]
            xb, yb = coords_old[key[1]]
            mx, my = 0.5 * (xa + xb), 0.5 * (ya + yb)
            mid = len(new_nodes)
            new_nodes.append(Node(int(mid), float(mx), float(my)))
            edge_to_mid[key] = int(mid)
            mid_info[int(mid)] = (int(key[0]), int(key[1]), parent_edge_tags.get(key, ""))
            return int(mid)

        new_elem_conn: List[List[int]] = []
        new_corner_conn: List[List[int]] = []
        new_elem_tags: List[str] = []

        for eid, corners in enumerate(np.asarray(self.corner_connectivity, dtype=int)):
            n0, n1, n2, n3 = (int(corners[0]), int(corners[1]), int(corners[2]), int(corners[3]))
            m01 = _midpoint(n0, n1)
            m12 = _midpoint(n1, n2)
            m23 = _midpoint(n2, n3)
            m30 = _midpoint(n3, n0)

            cx = float(0.25 * (coords_old[n0, 0] + coords_old[n1, 0] + coords_old[n2, 0] + coords_old[n3, 0]))
            cy = float(0.25 * (coords_old[n0, 1] + coords_old[n1, 1] + coords_old[n2, 1] + coords_old[n3, 1]))
            c_id = len(new_nodes)
            new_nodes.append(Node(int(c_id), cx, cy))

            parent_tag = str(getattr(self.elements_list[eid], "tag", "") or "")
            children_perim = (
                [n0, m01, c_id, m30],   # bottom-left
                [m01, n1, m12, c_id],   # bottom-right
                [c_id, m12, n2, m23],   # top-right
                [m30, c_id, m23, n3],   # top-left
            )
            for perim in children_perim:
                new_corner_conn.append([int(v) for v in perim])
                # Mesh expects lattice row-major order: [BL, BR, TL, TR]
                new_elem_conn.append([int(perim[0]), int(perim[1]), int(perim[3]), int(perim[2])])
                new_elem_tags.append(parent_tag)

        refined = Mesh(
            nodes=new_nodes,
            element_connectivity=np.asarray(new_elem_conn, dtype=int),
            elements_corner_nodes=np.asarray(new_corner_conn, dtype=int),
            element_type="quad",
            poly_order=1,
        )

        # Propagate element tags (and caches) from parents.
        if new_elem_tags:
            for el, tag in zip(refined.elements_list, new_elem_tags):
                el.tag = tag
            tags_el = np.asarray(new_elem_tags, dtype=object)
            refined._elem_bitsets = {t: BitSet(tags_el == t) for t in np.unique(tags_el).tolist() if t}

        # Propagate edge tags: only edges that split a parent edge inherit its tag.
        for e in refined.edges_list:
            a, b = int(e.nodes[0]), int(e.nodes[1])
            info = mid_info.get(a)
            other = b
            if info is None:
                info = mid_info.get(b)
                other = a
            if info is None:
                continue
            pa, pb, tag = info
            if tag and other in (pa, pb):
                e.tag = tag

        refined.rebuild_edge_bitsets()
        return refined

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
