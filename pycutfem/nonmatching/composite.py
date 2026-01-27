from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node

from .interface import NonMatchingInterface


@dataclass(frozen=True, slots=True)
class CompositeMeshMapping:
    """Bookkeeping for a 2-component composite mesh (pos + neg)."""

    mesh: Mesh
    pos_node_offset: int
    neg_node_offset: int
    pos_edge_offset: int
    neg_edge_offset: int
    pos_elem_offset: int
    neg_elem_offset: int
    n_pos_nodes: int
    n_neg_nodes: int
    n_pos_edges: int
    n_neg_edges: int
    n_pos_elems: int
    n_neg_elems: int

    @property
    def pos_elem_ids(self) -> np.ndarray:
        return np.arange(self.pos_elem_offset, self.pos_elem_offset + self.n_pos_elems, dtype=int)

    @property
    def neg_elem_ids(self) -> np.ndarray:
        return np.arange(self.neg_elem_offset, self.neg_elem_offset + self.n_neg_elems, dtype=int)


def _edges_connectivity_from_mesh(mesh: Mesh) -> list[list[int]]:
    edges_conn = getattr(mesh, "edges_connectivity", None)
    if edges_conn is not None:
        return [list(map(int, e)) for e in np.asarray(edges_conn, dtype=object)]
    out: list[list[int]] = []
    for e in getattr(mesh, "edges_list", []):
        nodes = getattr(e, "all_nodes", None) or getattr(e, "nodes", ())
        out.append([int(n) for n in nodes])
    return out


def _shift_connectivity(conn: np.ndarray, offset: int) -> np.ndarray:
    arr = np.asarray(conn, dtype=int)
    return arr + int(offset)


def build_composite_mesh(
    *,
    mesh_pos: Mesh,
    mesh_neg: Mesh,
    order: str = "pos_neg",
) -> CompositeMeshMapping:
    """Create a composite Mesh by concatenating two disjoint meshes.

    Notes
    -----
    - The composite mesh preserves the *separate* node ids of each component
      even if coordinates coincide (required for non-matching coupling).
    - We pass explicit `edges_connectivity` to avoid cross-component "edge node"
      discovery when poly_order > 1 (which can otherwise pull nodes from the
      other component if they lie on the same geometric line).
    """

    order = str(order).strip().lower()
    if order not in {"pos_neg", "neg_pos"}:
        raise ValueError("order must be 'pos_neg' or 'neg_pos'")

    if str(getattr(mesh_pos, "element_type", None)) != str(getattr(mesh_neg, "element_type", None)):
        raise ValueError("Composite mesh requires both meshes to have the same element_type.")
    if int(getattr(mesh_pos, "poly_order", 1)) != int(getattr(mesh_neg, "poly_order", 1)):
        raise ValueError("Composite mesh requires both meshes to have the same poly_order.")

    if order == "pos_neg":
        first, second = mesh_pos, mesh_neg
        first_is_pos = True
    else:
        first, second = mesh_neg, mesh_pos
        first_is_pos = False

    n1_nodes = int(getattr(first, "nodes_x_y_pos").shape[0])
    n2_nodes = int(getattr(second, "nodes_x_y_pos").shape[0])
    n1_elems = int(getattr(first, "elements_connectivity").shape[0])
    n2_elems = int(getattr(second, "elements_connectivity").shape[0])

    # Nodes (keep tags if present)
    nodes: list[Node] = []
    for i, nd in enumerate(getattr(first, "nodes_list")):
        nodes.append(Node(int(i), float(nd.x), float(nd.y), getattr(nd, "tag", None)))
    for j, nd in enumerate(getattr(second, "nodes_list")):
        nodes.append(Node(int(n1_nodes + j), float(nd.x), float(nd.y), getattr(nd, "tag", None)))

    # Element connectivity (all nodes) + corner connectivity
    elems_1 = np.asarray(getattr(first, "elements_connectivity"), dtype=int)
    elems_2 = _shift_connectivity(np.asarray(getattr(second, "elements_connectivity"), dtype=int), n1_nodes)
    elems = np.vstack([elems_1, elems_2])

    corners_1 = np.asarray(getattr(first, "corner_connectivity"), dtype=int)
    corners_2 = _shift_connectivity(np.asarray(getattr(second, "corner_connectivity"), dtype=int), n1_nodes)
    corners = np.vstack([corners_1, corners_2])

    # Explicit edges connectivity
    edges_1 = _edges_connectivity_from_mesh(first)
    edges_2 = [[int(n1_nodes + int(n)) for n in edge] for edge in _edges_connectivity_from_mesh(second)]
    edges_conn: list[list[int]] = edges_1 + edges_2

    composite = Mesh(
        nodes,
        elems,
        np.asarray(edges_conn, dtype=object),
        corners,
        element_type=str(getattr(first, "element_type", "tri")),
        poly_order=int(getattr(first, "poly_order", 1)),
        deduplicate_nodes=False,
    )

    n1_edges = int(len(edges_1))
    n2_edges = int(len(edges_2))

    if first_is_pos:
        return CompositeMeshMapping(
            mesh=composite,
            pos_node_offset=0,
            neg_node_offset=n1_nodes,
            pos_edge_offset=0,
            neg_edge_offset=n1_edges,
            pos_elem_offset=0,
            neg_elem_offset=n1_elems,
            n_pos_nodes=n1_nodes,
            n_neg_nodes=n2_nodes,
            n_pos_edges=n1_edges,
            n_neg_edges=n2_edges,
            n_pos_elems=n1_elems,
            n_neg_elems=n2_elems,
        )

    # order == "neg_pos" (swap bookkeeping)
    return CompositeMeshMapping(
        mesh=composite,
        pos_node_offset=n1_nodes,
        neg_node_offset=0,
        pos_edge_offset=n1_edges,
        neg_edge_offset=0,
        pos_elem_offset=n1_elems,
        neg_elem_offset=0,
        n_pos_nodes=n2_nodes,
        n_neg_nodes=n1_nodes,
        n_pos_edges=n2_edges,
        n_neg_edges=n1_edges,
        n_pos_elems=n2_elems,
        n_neg_elems=n1_elems,
    )


def lift_nonmatching_interface_to_composite(
    *,
    interface: NonMatchingInterface,
    mapping: CompositeMeshMapping,
) -> NonMatchingInterface:
    """Lift a NonMatchingInterface built on (mesh_pos, mesh_neg) to a composite mesh."""

    if interface.mesh_pos is interface.mesh_neg:
        raise ValueError("Interface already refers to a single mesh; use it directly.")

    return NonMatchingInterface(
        mesh_neg=mapping.mesh,
        mesh_pos=mapping.mesh,
        neg_edge_ids=np.asarray(interface.neg_edge_ids, dtype=int) + int(mapping.neg_edge_offset),
        pos_edge_ids=np.asarray(interface.pos_edge_ids, dtype=int) + int(mapping.pos_edge_offset),
        neg_elem_ids=np.asarray(interface.neg_elem_ids, dtype=int) + int(mapping.neg_elem_offset),
        pos_elem_ids=np.asarray(interface.pos_elem_ids, dtype=int) + int(mapping.pos_elem_offset),
        P0=np.asarray(interface.P0, dtype=float),
        P1=np.asarray(interface.P1, dtype=float),
        n=np.asarray(interface.n, dtype=float),
        h_neg=np.asarray(interface.h_neg, dtype=float),
        h_pos=np.asarray(interface.h_pos, dtype=float),
    )
