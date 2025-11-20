"""pycutfem.utils.ogrid_meshgen
Structured O-grid generator for a circular hole inside a rectangle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from pycutfem.core.topology import Node

__all__ = ["circular_hole_ogrid"]


@dataclass
class _BlockSpec:
    """Stores geometric data for an O-grid block."""

    name: str
    outer_points: np.ndarray  # (n_segments + 1, 2)
    angles: np.ndarray  # strictly increasing angles corresponding to outer_points


class _NodeRegistry:
    """Deduplicates nodes while accumulating tags."""

    def __init__(self, tol: float = 1.0e-10):
        self._coords: List[Tuple[float, float]] = []
        self._tags: List[set[str]] = []
        self._lookup: dict[Tuple[float, float], int] = {}
        self._decimals = max(8, int(abs(math.log10(tol))) + 2)

    def add(self, x: float, y: float, tags: Iterable[str] = ()) -> int:
        key = (round(float(x), self._decimals), round(float(y), self._decimals))
        idx = self._lookup.get(key)
        if idx is None:
            idx = len(self._coords)
            self._coords.append((float(key[0]), float(key[1])))
            tag_set = set(tags) if tags else set()
            self._tags.append(tag_set)
            self._lookup[key] = idx
        else:
            if tags:
                self._tags[idx].update(tags)
        return idx

    def build_nodes(self, offset: Optional[Tuple[float, float]]) -> List[Node]:
        off_x = 0.0 if offset is None else float(offset[0])
        off_y = 0.0 if offset is None else float(offset[1])
        nodes: List[Node] = []
        for idx, (x, y) in enumerate(self._coords):
            tag = ",".join(sorted(self._tags[idx])) if self._tags[idx] else ""
            nodes.append(Node(idx, x + off_x, y + off_y, tag=tag))
        return nodes


def circular_hole_ogrid(
    Lx: float,
    Ly: float,
    *,
    circle_center: Tuple[float, float],
    circle_radius: float,
    ring_thickness: float,
    n_radial_layers: int,
    nx_outer: Tuple[int, int, int],
    ny_outer: Tuple[int, int, int],
    poly_order: int = 1,
    offset: Optional[Tuple[float, float]] = None,
    outer_box_half_lengths: Optional[Tuple[float, float]] = None,
    tol: float = 1.0e-10,
) -> Tuple[List[Node], np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a structured quadrilateral mesh for a rectangle with a circular hole.

    The domain is the rectangle [0, Lx] x [0, Ly] with a concentric O-grid around a
    circle located anywhere inside the rectangle.  The outer far-field mesh remains
    axis-aligned, while the O-grid smoothly blends from the circle to an inner
    rectangle.

    Parameters
    ----------
    Lx, Ly:
        Domain dimensions along x and y.
    circle_center:
        (cx, cy) coordinates of the circle's center.
    circle_radius:
        Radius of the inner circular hole (must fit inside the domain).
    ring_thickness:
        Radial distance from the circle to the outer O-grid rectangle when
        ``outer_box_half_lengths`` is not provided.
    n_radial_layers:
        Number of layers in the O-grid between the circle and the rectangular
        interface.
    nx_outer, ny_outer:
        Tuples ``(n_left, n_interface, n_right)`` and
        ``(n_bottom, n_interface, n_top)`` describing how many rectangular cells
        to place in each partition of the far-field mesh.  The middle entry in
        each tuple also controls the number of O-grid blocks along that direction.
    poly_order:
        Polynomial order of each quadrilateral element (Qn).
    offset:
        Optional translation applied to all node coordinates after generation.
    outer_box_half_lengths:
        Optional half-widths ``(hx, hy)`` of the rectangle that bounds the O-grid.
        If omitted, ``hx = hy = circle_radius + ring_thickness``.
    tol:
        Geometric tolerance used for tagging and node deduplication.

    Returns
    -------
    nodes, elements, edges, elements_corner_nodes
        Matching the data structure produced by :func:`pycutfem.utils.meshgen.structured_quad`.
    """

    if poly_order < 1:
        raise ValueError("Polynomial order must be a positive integer.")
    if circle_radius <= 0.0:
        raise ValueError("Circle radius must be positive.")
    if n_radial_layers < 1:
        raise ValueError("n_radial_layers must be at least one.")
    if any(n < 0 for n in (*nx_outer, *ny_outer)):
        raise ValueError("Partition counts must be non-negative.")
    if nx_outer[1] == 0 or ny_outer[1] == 0:
        raise ValueError("Interface partitions nx_outer[1] and ny_outer[1] must be positive.")
    if ring_thickness <= 0.0 and outer_box_half_lengths is None:
        raise ValueError("ring_thickness must be positive when outer_box_half_lengths is not provided.")

    cx, cy = circle_center
    domain_bounds = (0.0, Lx, 0.0, Ly)

    if not (0.0 + tol < cx < Lx - tol and 0.0 + tol < cy < Ly - tol):
        raise ValueError("circle_center must lie strictly inside the rectangle.")
    if (
        cx - circle_radius <= domain_bounds[0] + tol
        or cx + circle_radius >= domain_bounds[1] - tol
        or cy - circle_radius <= domain_bounds[2] + tol
        or cy + circle_radius >= domain_bounds[3] - tol
    ):
        raise ValueError("Circle does not fit inside the rectangle with the requested radius.")

    if outer_box_half_lengths is None:
        hx = hy = circle_radius + ring_thickness
    else:
        hx, hy = outer_box_half_lengths

    if hx <= circle_radius or hy <= circle_radius:
        raise ValueError("outer_box_half_lengths must be larger than the circle radius.")

    x_inner_left = cx - hx
    x_inner_right = cx + hx
    y_inner_bottom = cy - hy
    y_inner_top = cy + hy

    if (
        x_inner_left <= domain_bounds[0] + tol
        or x_inner_right >= domain_bounds[1] - tol
        or y_inner_bottom <= domain_bounds[2] + tol
        or y_inner_top >= domain_bounds[3] - tol
    ):
        raise ValueError("O-grid rectangle does not fit inside the domain. Decrease ring_thickness.")

    xi_coords = np.linspace(0.0, 1.0, poly_order + 1)
    eta_coords = xi_coords.copy()
    registry = _NodeRegistry(tol=tol)
    classifier = _build_tag_classifier(
        domain_bounds,
        (x_inner_left, x_inner_right, y_inner_bottom, y_inner_top),
        circle_center,
        circle_radius,
        tol,
    )

    elements: List[List[int]] = []
    elements_corner_nodes: List[List[int]] = []

    # --- Build far-field rectangular blocks ---
    x_parts = {
        "left": _partition_coords(domain_bounds[0], x_inner_left, nx_outer[0]),
        "mid": _partition_coords(x_inner_left, x_inner_right, nx_outer[1]),
        "right": _partition_coords(x_inner_right, domain_bounds[1], nx_outer[2]),
    }
    y_parts = {
        "bottom": _partition_coords(domain_bounds[2], y_inner_bottom, ny_outer[0]),
        "mid": _partition_coords(y_inner_bottom, y_inner_top, ny_outer[1]),
        "top": _partition_coords(y_inner_top, domain_bounds[3], ny_outer[2]),
    }

    # Exclude the center block (mid x mid) because it is occupied by the O-grid.
    rect_blocks = [
        ("bottom_left", x_parts["left"], y_parts["bottom"]),
        ("bottom_center", x_parts["mid"], y_parts["bottom"]),
        ("bottom_right", x_parts["right"], y_parts["bottom"]),
        ("mid_left", x_parts["left"], y_parts["mid"]),
        ("mid_right", x_parts["right"], y_parts["mid"]),
        ("top_left", x_parts["left"], y_parts["top"]),
        ("top_center", x_parts["mid"], y_parts["top"]),
        ("top_right", x_parts["right"], y_parts["top"]),
    ]

    nodes_per_edge = poly_order + 1

    for block_name, x_coords, y_coords in rect_blocks:
        if len(x_coords) <= 1 or len(y_coords) <= 1:
            continue  # Zero-width block.
        for j in range(len(y_coords) - 1):
            y0, y1 = y_coords[j], y_coords[j + 1]
            for i in range(len(x_coords) - 1):
                x0, x1 = x_coords[i], x_coords[i + 1]
                element_nodes, element_coords = _add_rect_element(
                    x0,
                    x1,
                    y0,
                    y1,
                    xi_coords,
                    eta_coords,
                    registry,
                    classifier,
                )
                elements.append(element_nodes)
                elements_corner_nodes.append(
                    _extract_corner_ids(element_nodes, nodes_per_edge)
                )

    # --- Build O-grid blocks (right, top, left, bottom) ---
    outer_blocks = _build_ogrid_blocks(
        x_inner_left,
        x_inner_right,
        y_inner_bottom,
        y_inner_top,
        nx_outer[1],
        ny_outer[1],
        circle_center,
    )

    for block in outer_blocks:
        segments = len(block.outer_points) - 1
        if segments <= 0:
            continue
        for j in range(n_radial_layers):
            for i in range(segments):
                element_nodes, element_coords = _add_curved_element(
                    block.outer_points[i],
                    block.outer_points[i + 1],
                    block.angles[i],
                    block.angles[i + 1],
                    circle_center,
                    circle_radius,
                    j,
                    n_radial_layers,
                    xi_coords,
                    eta_coords,
                    registry,
                    classifier,
                )
                elements.append(element_nodes)
                elements_corner_nodes.append(
                    _extract_corner_ids(element_nodes, nodes_per_edge)
                )

    elements_array = np.array(elements, dtype=np.int64)
    corner_array = np.array(elements_corner_nodes, dtype=np.int64)
    edges = _build_edges_from_corners(corner_array)
    nodes = registry.build_nodes(offset)
    return nodes, elements_array, edges, corner_array


def _partition_coords(start: float, end: float, cells: int) -> np.ndarray:
    """Uniformly partition [start, end] into `cells` segments."""
    if cells < 0:
        raise ValueError("Partition counts must be non-negative.")
    if end < start:
        raise ValueError("Partition interval has negative length.")
    if cells == 0:
        return np.array([start], dtype=float)
    return np.linspace(start, end, cells + 1, dtype=float)


def _build_tag_classifier(
    domain_bounds: Tuple[float, float, float, float],
    inner_bounds: Tuple[float, float, float, float],
    circle_center: Tuple[float, float],
    circle_radius: float,
    tol: float,
) -> Callable[[float, float], Sequence[str]]:
    x0, x1, y0, y1 = domain_bounds
    xi_left, xi_right, yi_bottom, yi_top = inner_bounds
    cx, cy = circle_center
    circle_tol = max(tol * 10.0, circle_radius * 1.0e-9)

    def classify(x: float, y: float) -> Sequence[str]:
        tags: List[str] = []
        if math.isclose(x, x0, abs_tol=tol):
            tags.append("boundary_left")
        if math.isclose(x, x1, abs_tol=tol):
            tags.append("boundary_right")
        if math.isclose(y, y0, abs_tol=tol):
            tags.append("boundary_bottom")
        if math.isclose(y, y1, abs_tol=tol):
            tags.append("boundary_top")
        r = math.hypot(x - cx, y - cy)
        if abs(r - circle_radius) <= circle_tol:
            tags.append("boundary_circle")
        if (
            math.isclose(x, xi_left, abs_tol=tol)
            or math.isclose(x, xi_right, abs_tol=tol)
            or math.isclose(y, yi_bottom, abs_tol=tol)
            or math.isclose(y, yi_top, abs_tol=tol)
        ):
            tags.append("ogrid_interface")
        return tags

    return classify


def _add_rect_element(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    xi_coords: np.ndarray,
    eta_coords: np.ndarray,
    registry: _NodeRegistry,
    classifier: Callable[[float, float], Sequence[str]],
) -> List[int]:
    nodes_per_edge = len(xi_coords)
    element_nodes: List[int] = []
    element_coords: List[Tuple[float, float]] = []
    for eta in eta_coords:
        y = (1.0 - eta) * y0 + eta * y1
        for xi in xi_coords:
            x = (1.0 - xi) * x0 + xi * x1
            element_nodes.append(registry.add(x, y, classifier(x, y)))
            element_coords.append((x, y))
    return _ensure_ccw_order(element_nodes, element_coords, nodes_per_edge)


def _add_curved_element(
    outer_left: np.ndarray,
    outer_right: np.ndarray,
    theta_left: float,
    theta_right: float,
    circle_center: Tuple[float, float],
    circle_radius: float,
    radial_layer: int,
    n_radial_layers: int,
    xi_coords: np.ndarray,
    eta_coords: np.ndarray,
    registry: _NodeRegistry,
    classifier: Callable[[float, float], Sequence[str]],
) -> List[int]:
    circle_center_vec = np.array(circle_center, dtype=float)
    angle_span = theta_right - theta_left
    outer_vec = outer_right - outer_left
    layer_height = 1.0 / n_radial_layers

    nodes_per_edge = len(xi_coords)
    element_nodes: List[int] = []
    element_coords: List[Tuple[float, float]] = []
    for eta in eta_coords:
        global_eta = (radial_layer + eta) * layer_height
        for xi in xi_coords:
            theta = theta_left + angle_span * xi
            circle_pt = circle_center_vec + circle_radius * np.array(
                [math.cos(theta), math.sin(theta)], dtype=float
            )
            interface_pt = outer_left + outer_vec * xi
            mapped = circle_pt + global_eta * (interface_pt - circle_pt)
            x, y = float(mapped[0]), float(mapped[1])
            element_nodes.append(registry.add(x, y, classifier(x, y)))
            element_coords.append((x, y))
    return _ensure_ccw_order(element_nodes, element_coords, nodes_per_edge)


def _ensure_ccw_order(
    node_ids: Sequence[int], coords: Sequence[Tuple[float, float]], nodes_per_edge: int
) -> Tuple[List[int], List[Tuple[float, float]]]:
    nloc = nodes_per_edge * nodes_per_edge
    if len(node_ids) != nloc or len(coords) != nloc:
        raise ValueError("Local node data does not match nodes_per_edge for O-grid element.")
    ids = np.asarray(node_ids, dtype=np.int64).reshape(nodes_per_edge, nodes_per_edge)
    xy = np.asarray(coords, dtype=np.float64).reshape(nodes_per_edge, nodes_per_edge, 2)

    def signed_area(data: np.ndarray) -> float:
        bl = data[0, 0]
        br = data[0, -1]
        tr = data[-1, -1]
        tl = data[-1, 0]
        pts = (bl, br, tr, tl)
        area = 0.0
        for i in range(4):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % 4]
            area += x1 * y2 - x2 * y1
        return 0.5 * area

    def _classify_corners(arr_xy: np.ndarray) -> Tuple[int, int, int, int]:
        corners = [
            arr_xy[0, 0],
            arr_xy[0, -1],
            arr_xy[-1, -1],
            arr_xy[-1, 0],
        ]
        remaining = {0, 1, 2, 3}

        def _pop_min(key):
            idx = min(remaining, key=key)
            remaining.remove(idx)
            return idx

        def _pop_max(key):
            idx = max(remaining, key=key)
            remaining.remove(idx)
            return idx

        bl = _pop_min(lambda i: (corners[i][1], corners[i][0]))
        br = _pop_min(lambda i: (corners[i][1], -corners[i][0]))
        tr = _pop_max(lambda i: (corners[i][1], corners[i][0]))
        tl = remaining.pop()
        return bl, br, tr, tl

    bl_idx, br_idx, tr_idx, tl_idx = _classify_corners(xy)
    ids_r = np.rot90(ids, k=bl_idx)
    xy_r = np.rot90(xy, k=bl_idx, axes=(0, 1))

    br_pos = (br_idx - bl_idx) % 4
    if br_pos == 1:
        pass  # already aligned
    elif br_pos == 3:
        ids_r = ids_r[:, ::-1]
        xy_r = xy_r[:, ::-1, :]
    else:
        raise RuntimeError("Unable to orient O-grid element consistently.")

    if signed_area(xy_r) <= 0.0:
        ids_r = ids_r[::-1, :]
        xy_r = xy_r[::-1, :, :]
        if signed_area(xy_r) <= 0.0:
            raise RuntimeError("Failed to orient O-grid element with positive Jacobian.")

    ids_flat = ids_r.reshape(-1).tolist()
    coords_flat = xy_r.reshape(-1, 2).tolist()
    return ids_flat, coords_flat


def _extract_corner_ids(
    element_nodes: Sequence[int], nodes_per_edge: int
) -> List[int]:
    """
    Return the node ids corresponding to (bl, br, tr, tl) assuming the layout
    already satisfies the canonical row-major ordering.
    """
    ids = np.asarray(element_nodes, dtype=np.int64).reshape(nodes_per_edge, nodes_per_edge)
    return [
        int(ids[0, 0]),
        int(ids[0, -1]),
        int(ids[-1, -1]),
        int(ids[-1, 0]),
    ]


def _build_edges_from_corners(elements_corner_nodes: np.ndarray) -> np.ndarray:
    edge_pairs = set()
    for corners in elements_corner_nodes:
        for i in range(4):
            n0 = int(corners[i])
            n1 = int(corners[(i + 1) % 4])
            edge_pairs.add(tuple(sorted((n0, n1))))
    if not edge_pairs:
        return np.empty((0, 2), dtype=np.int64)
    edge_array = np.array(sorted(edge_pairs), dtype=np.int64)
    return edge_array


def _build_ogrid_blocks(
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
    n_tangent_horizontal: int,
    n_tangent_vertical: int,
    circle_center: Tuple[float, float],
) -> List[_BlockSpec]:
    """Create block specifications in CCW order: right, top, left, bottom."""
    cx, cy = circle_center
    y_span = np.linspace(y_bottom, y_top, n_tangent_vertical + 1, dtype=float)
    x_span = np.linspace(x_left, x_right, n_tangent_horizontal + 1, dtype=float)

    blocks = [
        ("right", np.column_stack((np.full_like(y_span, x_right), y_span))),
        ("top", np.column_stack((x_span[::-1], np.full_like(x_span, y_top)))),
        ("left", np.column_stack((np.full_like(y_span, x_left), y_span[::-1]))),
        ("bottom", np.column_stack((x_span, np.full_like(x_span, y_bottom)))),
    ]

    specs: List[_BlockSpec] = []
    prev_theta: Optional[float] = None
    for name, points in blocks:
        raw_angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
        angles = _enforce_monotonic_angles(raw_angles, prev_theta)
        prev_theta = angles[-1]
        specs.append(_BlockSpec(name, points, angles))
    return specs


def _enforce_monotonic_angles(angles: np.ndarray, prev_final: Optional[float]) -> np.ndarray:
    """Shift angles so they are strictly increasing and continuous in CCW order."""
    adjusted = np.array(angles, dtype=float)
    if prev_final is not None:
        while adjusted[0] <= prev_final:
            adjusted[0] += 2.0 * math.pi
    for i in range(1, len(adjusted)):
        while adjusted[i] <= adjusted[i - 1]:
            adjusted[i] += 2.0 * math.pi
    return adjusted


def _demo():
    """Quick visualization hook for debugging."""
    from pycutfem.io.visualization import visualize_mesh_node_order

    nodes, elements, edges, _ = circular_hole_ogrid(
        4.0,
        2.0,
        circle_center=(1.5, 1.0),
        circle_radius=0.35,
        ring_thickness=0.35,
        n_radial_layers=3,
        nx_outer=(3, 8, 3),
        ny_outer=(2, 6, 2),
        poly_order=2,
    )
    pts = np.array([[node.x, node.y] for node in nodes])
    visualize_mesh_node_order(
        pts, elements, order=2, element_type="quad", title="Circular-hole O-grid"
    )


if __name__ == "__main__":
    _demo()
