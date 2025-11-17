"""
Helpers for importing 2D Gmsh meshes into :class:`pycutfem.core.mesh.Mesh`.

The module provides a small wrapper around the Gmsh Python API that reads a
``.msh`` file, converts the nodes and element connectivity into the structures
expected by :class:`Mesh`, and optionally transfers physical-line tags onto the
PyCutFEM boundary edges.
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import gmsh
import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node

__all__ = ["GmshMeshData", "load_gmsh_mesh", "mesh_from_gmsh"]


EDGE_TABLE: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "tri": ((0, 1), (1, 2), (2, 0)),
    "quad": ((0, 1), (1, 2), (2, 3), (3, 0)),
}


@dataclass(slots=True)
class GmshMeshData:
    """Light-weight container with the raw data extracted from a Gmsh file."""

    nodes: List[Node]
    element_connectivity: np.ndarray
    edges_connectivity: np.ndarray
    corner_nodes: np.ndarray
    element_type: str
    poly_order: int
    edge_tags: Dict[Tuple[int, int], Tuple[str, ...]]


@contextmanager
def _gmsh_session():
    """
    Context manager that isolates the temporary model used for importing.

    If the user already has an active gmsh session we create a temporary model
    and restore the previous one afterwards. Otherwise we initialise gmsh and
    finalise it when leaving the context.
    """
    need_finalize = False
    previous_model: Optional[str] = None
    tmp_name = f"__pycutfem_gmsh_loader_{uuid.uuid4().hex}__"
    if not gmsh.isInitialized():
        gmsh.initialize()
        need_finalize = True
    else:
        previous_model = gmsh.model.getCurrent()
    gmsh.model.add(tmp_name)
    gmsh.model.setCurrent(tmp_name)
    try:
        yield
    finally:
        # Remove the temporary model and restore the previous one when needed.
        gmsh.model.remove()
        if previous_model:
            gmsh.model.setCurrent(previous_model)
        if need_finalize:
            gmsh.finalize()


def _physical_name(dim: int, tag: int) -> str:
    name = gmsh.model.getPhysicalName(dim, tag)
    return name if name else f"dim{dim}_tag{tag}"


def _element_family(name: str) -> str:
    lowered = name.lower()
    if "triangle" in lowered:
        return "tri"
    if "quad" in lowered or "quadrilateral" in lowered:
        return "quad"
    raise ValueError(f"Unsupported 2D element family '{name}'.")


def _map_node_ids(nodes: np.ndarray, id_map: Dict[int, int]) -> np.ndarray:
    mapped = np.empty_like(nodes, dtype=np.int64)
    flat_nodes = nodes.ravel(order="C")
    for idx, node in enumerate(flat_nodes):
        mapped.flat[idx] = id_map[int(node)]
    return mapped.reshape(nodes.shape, order="C")


def _select_entity_tags(
    dim: int, physical_names: Optional[Sequence[str]]
) -> List[int]:
    all_entities = [tag for (entity_dim, tag) in gmsh.model.getEntities(dim) if entity_dim == dim]
    if not all_entities:
        raise ValueError(f"No geometric entities of dimension {dim} found in the mesh.")
    if not physical_names:
        return all_entities
    requested = set(physical_names)
    selected: set[int] = set()
    for entity_dim, phys_tag in gmsh.model.getPhysicalGroups(dim):
        if entity_dim != dim:
            continue
        name = _physical_name(entity_dim, phys_tag)
        if name in requested:
            for entity_tag in gmsh.model.getEntitiesForPhysicalGroup(entity_dim, phys_tag):
                selected.add(entity_tag)
    if not selected:
        raise ValueError(
            f"None of the requested physical groups {sorted(requested)} exist in the mesh."
        )
    return [tag for tag in all_entities if tag in selected]


def _build_edge_connectivity(corner_nodes: np.ndarray, element_type: str) -> np.ndarray:
    edges = set()
    for elem_corners in corner_nodes:
        for idx_a, idx_b in EDGE_TABLE[element_type]:
            key = tuple(sorted((int(elem_corners[idx_a]), int(elem_corners[idx_b]))))
            edges.add(key)
    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(sorted(edges), dtype=np.int64)


def _collect_boundary_tags(
    id_map: Dict[int, int],
) -> Tuple[Dict[int, List[str]], Dict[Tuple[int, int], List[str]]]:
    node_tags: Dict[int, List[str]] = defaultdict(list)
    edge_tags: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for entity_dim, phys_tag in gmsh.model.getPhysicalGroups(1):
        if entity_dim != 1:
            continue
        phys_name = _physical_name(entity_dim, phys_tag)
        for curve_tag in gmsh.model.getEntitiesForPhysicalGroup(entity_dim, phys_tag):
            elem_types, _, elem_nodes = gmsh.model.mesh.getElements(entity_dim, curve_tag)
            for elem_type, elem_node_data in zip(elem_types, elem_nodes):
                if len(elem_node_data) == 0:
                    continue
                _, _, _, nodes_per_elem, _, _ = gmsh.model.mesh.getElementProperties(elem_type)
                node_array = np.asarray(elem_node_data, dtype=np.int64).reshape(-1, nodes_per_elem)
                for row in node_array:
                    mapped = _map_node_ids(row.reshape(1, -1), id_map)[0].tolist()
                    for ln in mapped:
                        if phys_name not in node_tags[ln]:
                            node_tags[ln].append(phys_name)
                    key = tuple(sorted((mapped[0], mapped[-1])))
                    if phys_name not in edge_tags[key]:
                        edge_tags[key].append(phys_name)
    return node_tags, edge_tags


def load_gmsh_mesh(
    filepath: Union[str, Path],
    *,
    surface_dim: int = 2,
    surface_physical_names: Optional[Sequence[str]] = None,
    include_boundary_tags: bool = True,
) -> GmshMeshData:
    """
    Read a 2D mesh from a ``.msh`` file and convert it into PyCutFEM data.

    Args:
        filepath: Path to the Gmsh ``.msh`` file.
        surface_dim: Topological dimension that will be interpreted as elements.
        surface_physical_names: Optional list of physical group names that
            define the surface entities to keep. When omitted, all surfaces are
            imported.
        include_boundary_tags: When ``True``, boundary physical groups are
            propagated to Nodes and Edges.

    Returns:
        A :class:`GmshMeshData` object with Node objects, connectivity arrays,
        and metadata required to instantiate :class:`Mesh`.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(path)

    with _gmsh_session():
        gmsh.open(str(path))
        gmsh.model.mesh.removeDuplicateNodes()

        node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
        if not len(node_tags):
            raise ValueError("The Gmsh file does not contain any mesh nodes.")
        node_tags = np.asarray(node_tags, dtype=np.int64)
        coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)[:, :2]
        id_map = {int(tag): idx for idx, tag in enumerate(node_tags.tolist())}
        nodes = [Node(id=i, x=float(pt[0]), y=float(pt[1]), tag="") for i, pt in enumerate(coords)]

        entity_tags = _select_entity_tags(surface_dim, surface_physical_names)
        element_blocks: List[np.ndarray] = []
        element_type: Optional[str] = None
        poly_order: Optional[int] = None
        nodes_per_element: Optional[int] = None

        for entity_tag in entity_tags:
            elem_types, _, elem_nodes = gmsh.model.mesh.getElements(surface_dim, entity_tag)
            for elem_type, elem_node_data in zip(elem_types, elem_nodes):
                if len(elem_node_data) == 0:
                    continue
                name, dim, order, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(elem_type)
                if dim != surface_dim:
                    continue
                family = _element_family(name)
                if element_type is None:
                    element_type = family
                    poly_order = order
                    nodes_per_element = num_nodes
                else:
                    if family != element_type:
                        raise ValueError("Mixed element types are not supported.")
                    if order != poly_order or num_nodes != nodes_per_element:
                        raise ValueError("All elements must share the same polynomial order.")
                elem_array = np.asarray(elem_node_data, dtype=np.int64).reshape(-1, num_nodes)
                mapped = _map_node_ids(elem_array, id_map)
                element_blocks.append(mapped)

        if not element_blocks or element_type is None or poly_order is None or nodes_per_element is None:
            raise ValueError("The file does not contain any 2D elements.")

        element_connectivity = np.vstack(element_blocks).astype(np.int64, copy=False)
        n_corners = 3 if element_type == "tri" else 4
        corner_nodes = element_connectivity[:, :n_corners].copy()
        if element_type == "quad":
            for i in range(corner_nodes.shape[0]):
                conn = corner_nodes[i]
                pts = coords[conn]
                centroid = pts.mean(axis=0)
                angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
                order = np.argsort(angles)
                ordered_conn = conn[order]
                ordered_pts = pts[order]
                lex = np.lexsort((ordered_pts[:, 0], ordered_pts[:, 1]))
                start = int(lex[0])
                perm = np.roll(order, -start)
                reordered = np.roll(ordered_conn, -start)
                # Convert CCW [bl, br, tr, tl] into PyCutFEM ordering [bl, br, tl, tr]
                target_idx = np.array([0, 1, 3, 2], dtype=int)
                corner_nodes[i] = reordered[target_idx]
                row = element_connectivity[i, :n_corners].copy()
                element_connectivity[i, :n_corners] = row[perm][target_idx]
        edges_connectivity = _build_edge_connectivity(corner_nodes, element_type)

        edge_tag_map: Dict[Tuple[int, int], Tuple[str, ...]] = {}
        if include_boundary_tags:
            node_tag_map, boundary_tag_map = _collect_boundary_tags(id_map)
            for nid, tags in node_tag_map.items():
                nodes[nid].tag = ",".join(sorted(tags))
            edge_tag_map = {key: tuple(sorted(tags)) for key, tags in boundary_tag_map.items()}

        return GmshMeshData(
            nodes=nodes,
            element_connectivity=element_connectivity,
            edges_connectivity=edges_connectivity,
            corner_nodes=corner_nodes,
            element_type=element_type,
            poly_order=int(poly_order),
            edge_tags=edge_tag_map,
        )


def mesh_from_gmsh(
    filepath: Union[str, Path],
    *,
    surface_dim: int = 2,
    surface_physical_names: Optional[Sequence[str]] = None,
    apply_boundary_tags: bool = True,
) -> Mesh:
    """
    Convenience wrapper that reads a ``.msh`` file and instantiates ``Mesh``.
    """
    data = load_gmsh_mesh(
        filepath,
        surface_dim=surface_dim,
        surface_physical_names=surface_physical_names,
        include_boundary_tags=apply_boundary_tags,
    )
    mesh = Mesh(
        nodes=data.nodes,
        element_connectivity=data.element_connectivity,
        edges_connectivity=data.edges_connectivity,
        elements_corner_nodes=data.corner_nodes,
        element_type=data.element_type,
        poly_order=data.poly_order,
    )
    if apply_boundary_tags and data.edge_tags:
        for edge in mesh.edges_list:
            key = tuple(sorted(edge.nodes))
            tags = data.edge_tags.get(key)
            if tags:
                edge.tag = ",".join(tags)
    return mesh
