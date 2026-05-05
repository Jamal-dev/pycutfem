from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node
from pycutfem.fem.reference import get_reference


def _quad_corner_indices(p: int) -> tuple[int, int, int, int]:
    """Return (bl, br, tr, tl) indices for a tensor-product lattice (eta outer)."""
    n = p + 1
    bl = 0
    br = p
    tr = p * n + p
    tl = p * n
    return bl, br, tr, tl


def _parent_parametric_grid(p: int) -> np.ndarray:
    """Full parent reference grid (no split applied)."""
    t = np.linspace(-1.0, 1.0, p + 1)
    xi, eta = np.meshgrid(t, t, indexing="xy")
    return np.column_stack([xi.ravel(), eta.ravel()])


def _apply_sequence_to_grid(base_grid: np.ndarray, sequence: Sequence[str]) -> np.ndarray:
    xi_eta = base_grid.copy()
    for op in sequence:
        if op == "left":
            xi_eta[:, 0] = 0.5 * (xi_eta[:, 0] - 1.0)
        elif op == "right":
            xi_eta[:, 0] = 0.5 * (xi_eta[:, 0] + 1.0)
        elif op == "bottom":
            xi_eta[:, 1] = 0.5 * (xi_eta[:, 1] - 1.0)
        elif op == "top":
            xi_eta[:, 1] = 0.5 * (xi_eta[:, 1] + 1.0)
    return xi_eta


def _grid_key(grid: np.ndarray, ndp: int = 12) -> tuple[tuple[float, float], ...]:
    """Hashable, rounded representation of a parametric grid."""
    return tuple((float(np.round(pt[0], ndp)), float(np.round(pt[1], ndp))) for pt in np.asarray(grid))


@lru_cache(maxsize=None)
def _shape_table_for_grid(element_type: str, poly_order: int, grid_key: tuple[tuple[float, float], ...]) -> np.ndarray:
    """
    Tabulate reference shape functions for all points in `grid_key`.
    Cached so repeated refinement stages reuse the same tables.
    """
    ref = get_reference(element_type, poly_order)
    n_pts = len(grid_key)
    n_loc = len(ref.shape(0.0, 0.0))
    tab = np.empty((n_pts, n_loc), dtype=float)
    for i, (xi, eta) in enumerate(grid_key):
        tab[i, :] = ref.shape(float(xi), float(eta))
    return tab


def _map_grid_to_physical(mesh: Mesh, eid: int, grid: np.ndarray) -> np.ndarray:
    """
    Fast batched parent→physical mapping for all points in `grid` on element `eid`.
    Uses cached reference shape tables to avoid repeated transformations.
    """
    key = _grid_key(grid)
    Ntab = _shape_table_for_grid(mesh.element_type, mesh.poly_order, key)
    coords = mesh.nodes_x_y_pos[mesh.elements_connectivity[eid]]
    return Ntab @ coords


def _on_parent_edge(pt: np.ndarray, corners_xy: np.ndarray, tol: float = 1.0e-12) -> bool:
    """Detect whether point lies on any parent edge (straight edges only)."""
    edge_pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
    for i0, i1 in edge_pairs:
        a = corners_xy[i0]
        b = corners_xy[i1]
        ab = b - a
        L2 = float(np.dot(ab, ab))
        if L2 <= tol:
            continue
        cross = abs((pt[0] - a[0]) * ab[1] - (pt[1] - a[1]) * ab[0])
        if cross > tol * max(1.0, math.sqrt(L2)):
            continue
        s = np.dot(pt - a, ab) / L2
        if -tol <= s <= 1.0 + tol:
            return True
    return False


def _refine_element_with_sequences_fast(
    mesh: Mesh,
    eid: int,
    sequences: Sequence[Sequence[str]],
    nodes: list[Node],
    node_lookup: dict[tuple[float, float], int],
    base_grid: np.ndarray,
    xi_to_idx: dict[tuple[float, float], int],
    bl: int,
    br: int,
    tr: int,
    tl: int,
) -> tuple[list[list[int]], list[list[int]]]:
    conns_out: list[list[int]] = []
    corners_out: list[list[int]] = []

    # Cache parametric grids per sequence for this element to avoid repeated transforms.
    seq_cache: dict[tuple[str, ...], np.ndarray] = {}

    for seq in sequences:
        key = tuple(seq)
        grid = seq_cache.get(key)
        if grid is None:
            grid = _apply_sequence_to_grid(base_grid, seq)
            seq_cache[key] = grid
        phys_pts = _map_grid_to_physical(mesh, eid, grid)
        conn = []
        for (xi, eta), p_phys in zip(grid, phys_pts):
            xi_p = float(np.round(xi, 12))
            eta_p = float(np.round(eta, 12))
            key_pe = (xi_p, eta_p)
            idx = xi_to_idx.get(key_pe)
            if idx is not None:
                conn.append(int(mesh.elements_connectivity[eid][idx]))
                continue
            key = (float(round(p_phys[0], 14)), float(round(p_phys[1], 14)))
            nid = node_lookup.get(key)
            if nid is None:
                nid = len(nodes)
                node_lookup[key] = nid
                nodes.append(Node(nid, float(p_phys[0]), float(p_phys[1])))
            conn.append(int(nid))
        corners = [conn[bl], conn[br], conn[tr], conn[tl]]
        conns_out.append(conn)
        corners_out.append(corners)
    return conns_out, corners_out


def _tensor_sequences(n_split_x: int, n_split_y: int) -> list[list[str]]:
    """
    Generate split sequences for a tensor-product refinement with 2^nx × 2^ny
    children. Vertical splits (x-direction) are applied first, followed by
    horizontal splits (y-direction).
    """
    nx = max(int(n_split_x), 0)
    ny = max(int(n_split_y), 0)
    if nx == 0 and ny == 0:
        raise ValueError("tensor sequences requested with zero splits.")
    seqs: list[list[str]] = []
    for ix in range(1 << nx):
        for iy in range(1 << ny):
            seq: list[str] = []
            for bit in range(nx):
                b = (ix >> (nx - 1 - bit)) & 1
                seq.append("left" if b == 0 else "right")
            for bit in range(ny):
                b = (iy >> (ny - 1 - bit)) & 1
                seq.append("bottom" if b == 0 else "top")
            seqs.append(seq)
    return seqs


def _balance_tensor_levels(mesh: Mesh, rx: np.ndarray, ry: np.ndarray, *, max_ref: int, max_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Enforce a 2:1 balance on the planned (nx, ny) split counts.
    Levels are only increased, so the loop terminates in finite steps.
    """
    rx = np.minimum(np.asarray(rx, dtype=int), max_ref)
    ry = np.minimum(np.asarray(ry, dtype=int), max_ref)

    changed = True
    while changed:
        changed = False
        for eid, elem in enumerate(mesh.elements_list):
            ex = int(rx[eid])
            ey = int(ry[eid])
            for lid, nb in elem.neighbors.items():
                if nb is None:
                    continue
                nb = int(nb)
                nx = int(rx[nb])
                ny = int(ry[nb])
                if lid in (0, 2):  # bottom/top → compare x-splits
                    fine = max(ex, nx)
                    coarse = min(ex, nx)
                    if fine > max_ratio * max(coarse, 1):
                        target = fine - 1
                        if ex < fine and ex < target:
                            rx[eid] = min(target, max_ref)
                            ex = rx[eid]
                            changed = True
                        if nx < fine and nx < target:
                            rx[nb] = min(target, max_ref)
                            changed = True
                else:  # left/right → compare y-splits
                    fine = max(ey, ny)
                    coarse = min(ey, ny)
                    if fine > max_ratio * max(coarse, 1):
                        target = fine - 1
                        if ey < fine and ey < target:
                            ry[eid] = min(target, max_ref)
                            ey = ry[eid]
                            changed = True
                        if ny < fine and ny < target:
                            ry[nb] = min(target, max_ref)
                            changed = True
    return rx, ry


def _refine_tensor(mesh_in: Mesh, rx: np.ndarray, ry: np.ndarray, *, max_ref: int) -> Mesh:
    """
    Refine each quad into a 2^nx × 2^ny tensor grid (nx = rx[e], ny = ry[e]).
    Preserves parent→children bookkeeping and hanging node flags.
    """
    if len(rx) != len(mesh_in.elements_list) or len(ry) != len(mesh_in.elements_list):
        raise ValueError("rx/ry length mismatch with mesh elements.")
    rx = np.minimum(np.asarray(rx, dtype=int), max_ref)
    ry = np.minimum(np.asarray(ry, dtype=int), max_ref)

    nodes = list(mesh_in.nodes_list)
    node_lookup = {(round(nd.x, 14), round(nd.y, 14)): int(nd.id) for nd in nodes}
    hanging_nodes: set[int] = set(getattr(mesh_in, "hanging_nodes", []))
    new_elems: list[list[int]] = []
    new_corners: list[list[int]] = []
    parent_to_children: dict[int, list[int]] = {}
    parent_corner_coords: list[np.ndarray] = []

    p = mesh_in.poly_order
    t = np.linspace(-1.0, 1.0, p + 1)
    xi_to_idx_template = {(float(xi), float(eta)): int(j * (p + 1) + i) for j, eta in enumerate(t) for i, xi in enumerate(t)}
    bl, br, tr, tl = _quad_corner_indices(p)
    base_grid_parent = _parent_parametric_grid(p)

    for eid, elem in enumerate(mesh_in.elements_list):
        parent_corners = mesh_in.nodes_x_y_pos[list(elem.corner_nodes)]
        parent_corner_coords.append(parent_corners)
        nx = int(rx[eid])
        ny = int(ry[eid])
        if nx <= 0 and ny <= 0:
            new_elems.append(list(mesh_in.elements_connectivity[eid]))
            new_corners.append(list(mesh_in.corner_connectivity[eid]))
            parent_to_children[eid] = [len(new_elems) - 1]
            continue

        sequences = _tensor_sequences(nx, ny)
        before_nodes = len(nodes)
        conns, corners = _refine_element_with_sequences_fast(
            mesh_in,
            eid,
            sequences,
            nodes,
            node_lookup,
            base_grid_parent,
            xi_to_idx_template,
            bl,
            br,
            tr,
            tl,
        )

        new_node_ids = range(before_nodes, len(nodes))
        for nid in new_node_ids:
            pt = np.array([nodes[nid].x, nodes[nid].y], float)
            if _on_parent_edge(pt, parent_corners):
                hanging_nodes.add(nid)

        idx_children = []
        for conn_child, corners_child in zip(conns, corners):
            idx_children.append(len(new_elems))
            new_elems.append(conn_child)
            new_corners.append(corners_child)
        parent_to_children[eid] = idx_children

    refined = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(new_elems, dtype=int),
        elements_corner_nodes=np.asarray(new_corners, dtype=int),
        element_type="quad",
        poly_order=mesh_in.poly_order,
    )
    refined.hanging_nodes = sorted(hanging_nodes)
    refined.parent_to_children = parent_to_children
    refined.refinement_levels = {"rx": np.asarray(rx, dtype=int), "ry": np.asarray(ry, dtype=int)}
    refined.parent_corner_coords = np.asarray(parent_corner_coords)
    return refined


class TensorRefiner:
    """
    Adaptive quadtree-like refiner with 2:1 balancing and hanging-node tracking.
    Works with arbitrary level-set shapes; mesh size drives how many splits are applied.
    """

    def __init__(self, *, max_ratio: float = 2.0, max_ref: int = 6):
        self.max_ratio = max_ratio
        self.max_ref = max_ref

    def mark_near_levelset(
        self,
        mesh: Mesh,
        level_set,
        band: float,
        levels: int = 1,
        bbox_hint: Optional[Tuple[float, float, float, float]] = None,
    ) -> set[int]:
        """
        Mark elements whose corner box intersects a signed-distance band.
        bbox_hint can be provided as (xmin, xmax, ymin, ymax) to speed up checks
        for arbitrary geometries when a simple bounding box is known.
        """
        marked: set[int] = set()
        corners_all = mesh.nodes_x_y_pos[mesh.corner_connectivity]
        phi_corners = level_set(corners_all.reshape(-1, 2)).reshape(corners_all.shape[:-1])
        ex_min = corners_all[..., 0].min(axis=1)
        ex_max = corners_all[..., 0].max(axis=1)
        ey_min = corners_all[..., 1].min(axis=1)
        ey_max = corners_all[..., 1].max(axis=1)

        if bbox_hint is None:
            cx = getattr(level_set, "cx", None)
            hx = getattr(level_set, "hx", None)
            cy = getattr(level_set, "cy", None)
            hy = getattr(level_set, "hy", None)
            if cx is not None and hx is not None and cy is not None and hy is not None:
                bbox_hint = (float(cx - hx), float(cx + hx), float(cy - hy), float(cy + hy))
        if bbox_hint is None:
            bbox_hint = (0.0, 0.0, 0.0, 0.0)
        xmin_hint, xmax_hint, ymin_hint, ymax_hint = bbox_hint

        for eid, (phi, xmin, xmax, ymin, ymax) in enumerate(zip(phi_corners, ex_min, ex_max, ey_min, ey_max)):
            hits_phi = np.any(phi <= 0.0) or np.any(np.abs(phi) <= band)
            hits_bbox = (
                xmin <= xmax_hint + band
                and xmax >= xmin_hint - band
                and ymin <= ymax_hint + band
                and ymax >= ymin_hint - band
            )
            if hits_phi or hits_bbox:
                marked.add(int(eid))

        for _ in range(max(0, levels - 1)):
            new = set()
            for eid in marked:
                for nb in mesh._neighbors[eid]:
                    if nb is not None:
                        new.add(int(nb))
            marked |= new
        return marked

    def plan_tensor_levels(
        self,
        mesh: Mesh,
        marked: Iterable[int],
        *,
        target_h: float,
        span_x: Optional[Tuple[float, float]] = None,
        target_x: Optional[float] = None,
        span_x_halo: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Decide (rx, ry) splits for marked parents based on element extents and
        target resolution. If span_x is provided, elements overlapping that
        interval receive extra vertical splits to resolve geometry in x.
        """
        rx = np.zeros(len(mesh.elements_list), dtype=int)
        ry = np.zeros(len(mesh.elements_list), dtype=int)
        target_x = target_x or target_h

        xmin_span, xmax_span = span_x if span_x is not None else (None, None)

        for eid in marked:
            cn = mesh.nodes_x_y_pos[list(mesh.elements_list[eid].corner_nodes)]
            hx = float(cn[:, 0].max() - cn[:, 0].min())
            hy = float(cn[:, 1].max() - cn[:, 1].min())

            need_y = 0
            if hy > target_h * 0.9:
                need_y = int(math.ceil(math.log(max(hy / target_h, 1.0), 2.0)))
            need_y = max(1, need_y)

            need_x = 0
            spans_band = False
            if xmin_span is not None and xmax_span is not None:
                spans_band = cn[:, 0].min() <= xmax_span + span_x_halo and cn[:, 0].max() >= xmin_span - span_x_halo
            if spans_band or hx > 3.0 * target_h:
                need_x = int(math.ceil(math.log(max(hx / target_x, 1.0), 2.0)))
                if spans_band:
                    need_x = max(1, need_x)

            rx[eid] = min(max(rx[eid], need_x), self.max_ref)
            ry[eid] = min(max(ry[eid], need_y), self.max_ref)

        return rx, ry

    def balance_levels(self, mesh: Mesh, rx: np.ndarray, ry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _balance_tensor_levels(mesh, rx, ry, max_ref=self.max_ref, max_ratio=self.max_ratio)

    def refine(self, mesh: Mesh, rx: np.ndarray, ry: np.ndarray) -> Mesh:
        return _refine_tensor(mesh, rx, ry, max_ref=self.max_ref)

    def ensure_column_coverage(
        self,
        mesh: Mesh,
        level_set,
        target_h: float,
        *,
        missing_x: Sequence[float],
        y_interval: Tuple[float, float],
    ) -> Mesh:
        """
        Refine columns whose centroid x is near any missing_x location, within
        [y0, y1]. Useful for repairing rectangular/beam-like holes.
        """
        if not missing_x:
            return mesh
        rx = np.zeros(len(mesh.elements_list), dtype=int)
        ry = np.zeros(len(mesh.elements_list), dtype=int)
        y0, y1 = y_interval
        x_tol = target_h
        for eid, e in enumerate(mesh.elements_list):
            cn = mesh.nodes_x_y_pos[list(e.corner_nodes)]
            ex0, ex1 = cn[:, 0].min(), cn[:, 0].max()
            ey0, ey1 = cn[:, 1].min(), cn[:, 1].max()
            if ey1 < y0 or ey0 > y1:
                continue
            cx = 0.5 * (ex0 + ex1)
            if any(abs(cx - x0) <= x_tol for x0 in missing_x):
                rx[eid] = max(rx[eid], 1)
        if not np.any(rx):
            return mesh
        rx, ry = self.balance_levels(mesh, rx, ry)
        refined = self.refine(mesh, rx, ry)
        return refined
