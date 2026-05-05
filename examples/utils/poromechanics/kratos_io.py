"""Kratos mdpa helpers for example-level Poromechanics parity cases."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node
from pycutfem.fem.reference import get_reference


@dataclass(frozen=True)
class KratosSubModelPart:
    name: str
    node_ids: tuple[int, ...]
    element_ids: tuple[int, ...]
    condition_ids: tuple[int, ...]
    table_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class KratosElementBlock:
    name: str
    elements: dict[int, tuple[int, ...]]


@dataclass(frozen=True)
class KratosConditionBlock:
    name: str
    conditions: dict[int, tuple[int, ...]]


@dataclass(frozen=True)
class KratosTable:
    table_id: int
    x: np.ndarray
    y: np.ndarray

    def value(self, t: float) -> float:
        return float(np.interp(float(t), self.x, self.y))


@dataclass(frozen=True)
class KratosMDPA:
    path: Path
    nodes: dict[int, tuple[float, float]]
    element_blocks: tuple[KratosElementBlock, ...]
    condition_blocks: tuple[KratosConditionBlock, ...]
    submodelparts: dict[str, KratosSubModelPart]
    tables: dict[int, KratosTable]

    @property
    def elements(self) -> dict[int, tuple[int, ...]]:
        out: dict[int, tuple[int, ...]] = {}
        for block in self.element_blocks:
            out.update(block.elements)
        return out

    @property
    def conditions(self) -> dict[int, tuple[int, ...]]:
        out: dict[int, tuple[int, ...]] = {}
        for block in self.condition_blocks:
            out.update(block.conditions)
        return out

    def element_block(self, name: str) -> KratosElementBlock:
        for block in self.element_blocks:
            if block.name == name:
                return block
        raise KeyError(name)


def load_kratos_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return json.loads(text)


def parse_kratos_mdpa(path: str | Path) -> KratosMDPA:
    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()

    nodes: dict[int, tuple[float, float]] = {}
    element_blocks: list[KratosElementBlock] = []
    condition_blocks: list[KratosConditionBlock] = []
    submodelparts: dict[str, KratosSubModelPart] = {}
    tables: dict[int, KratosTable] = {}

    section = ""
    block_name = ""
    block_items: dict[int, tuple[int, ...]] = {}
    current_part = ""
    part_nodes: list[int] = []
    part_elements: list[int] = []
    part_conditions: list[int] = []
    part_tables: list[int] = []
    current_table_id: int | None = None
    table_x: list[float] = []
    table_y: list[float] = []

    def flush_part() -> None:
        nonlocal current_part, part_nodes, part_elements, part_conditions, part_tables
        if current_part:
            submodelparts[current_part] = KratosSubModelPart(
                name=current_part,
                node_ids=tuple(part_nodes),
                element_ids=tuple(part_elements),
                condition_ids=tuple(part_conditions),
                table_ids=tuple(part_tables),
            )
        current_part = ""
        part_nodes = []
        part_elements = []
        part_conditions = []
        part_tables = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue

        if line.startswith("Begin Table "):
            tokens = line.split()
            current_table_id = int(tokens[2])
            table_x = []
            table_y = []
            section = "table"
            continue
        if line == "End Table":
            if current_table_id is not None:
                tables[current_table_id] = KratosTable(
                    table_id=current_table_id,
                    x=np.asarray(table_x, dtype=float),
                    y=np.asarray(table_y, dtype=float),
                )
            current_table_id = None
            section = ""
            continue

        if line == "Begin Nodes":
            section = "nodes"
            continue
        if line == "End Nodes":
            section = ""
            continue

        if line.startswith("Begin Elements "):
            block_name = line.split()[2]
            block_items = {}
            section = "elements"
            continue
        if line == "End Elements":
            element_blocks.append(KratosElementBlock(name=block_name, elements=dict(block_items)))
            block_name = ""
            block_items = {}
            section = ""
            continue

        if line.startswith("Begin Conditions "):
            block_name = line.split()[2]
            block_items = {}
            section = "conditions"
            continue
        if line == "End Conditions":
            condition_blocks.append(KratosConditionBlock(name=block_name, conditions=dict(block_items)))
            block_name = ""
            block_items = {}
            section = ""
            continue

        if line.startswith("Begin SubModelPart "):
            flush_part()
            current_part = line.split()[2]
            section = "submodelpart"
            continue
        if line == "End SubModelPart":
            flush_part()
            section = ""
            continue

        if line == "Begin SubModelPartNodes":
            section = "submodelpart_nodes"
            continue
        if line == "End SubModelPartNodes":
            section = "submodelpart"
            continue
        if line == "Begin SubModelPartElements":
            section = "submodelpart_elements"
            continue
        if line == "End SubModelPartElements":
            section = "submodelpart"
            continue
        if line == "Begin SubModelPartConditions":
            section = "submodelpart_conditions"
            continue
        if line == "End SubModelPartConditions":
            section = "submodelpart"
            continue
        if line == "Begin SubModelPartTables":
            section = "submodelpart_tables"
            continue
        if line == "End SubModelPartTables":
            section = "submodelpart"
            continue

        if section == "table":
            parts = line.split()
            if len(parts) >= 2:
                table_x.append(float(parts[0]))
                table_y.append(float(parts[1]))
            continue
        if section == "nodes":
            parts = line.split()
            nodes[int(parts[0])] = (float(parts[1]), float(parts[2]))
            continue
        if section in {"elements", "conditions"}:
            parts = line.split()
            block_items[int(parts[0])] = tuple(int(value) for value in parts[2:])
            continue
        if section == "submodelpart_nodes":
            part_nodes.append(int(line.split()[0]))
            continue
        if section == "submodelpart_elements":
            part_elements.append(int(line.split()[0]))
            continue
        if section == "submodelpart_conditions":
            part_conditions.append(int(line.split()[0]))
            continue
        if section == "submodelpart_tables":
            part_tables.append(int(line.split()[0]))
            continue

    flush_part()
    return KratosMDPA(
        path=source,
        nodes=nodes,
        element_blocks=tuple(element_blocks),
        condition_blocks=tuple(condition_blocks),
        submodelparts=submodelparts,
        tables=tables,
    )


def mdpa_volume_mesh_to_pycutfem(
    mdpa: KratosMDPA,
    *,
    element_block_names: Sequence[str],
    domain_tag: str = "body",
    boundary_node_tags: dict[str, str] | None = None,
    boundary_condition_tags: dict[str, str] | None = None,
) -> Mesh:
    old_node_ids = sorted(int(node_id) for node_id in mdpa.nodes)
    old_to_new = {old_id: i for i, old_id in enumerate(old_node_ids)}
    new_to_old = np.asarray(old_node_ids, dtype=int)
    nodes = [
        Node(i, float(mdpa.nodes[old_id][0]), float(mdpa.nodes[old_id][1]))
        for i, old_id in enumerate(old_node_ids)
    ]

    selected: list[tuple[int, tuple[int, ...]]] = []
    for name in element_block_names:
        block = mdpa.element_block(name)
        selected.extend((int(eid), tuple(conn)) for eid, conn in block.elements.items())
    selected.sort(key=lambda item: item[0])
    if not selected:
        raise ValueError(f"No elements selected from {mdpa.path}.")

    raw_connectivity = [tuple(old_to_new[n] for n in conn) for _, conn in selected]
    element_ids = np.asarray([eid for eid, _ in selected], dtype=int)
    arity = len(raw_connectivity[0])
    if any(len(conn) != arity for conn in raw_connectivity):
        raise ValueError("Mixed volume element arity is not supported in one pycutfem mesh.")

    coords = np.asarray([[node.x, node.y] for node in nodes], dtype=float)
    element_type, poly_order, element_connectivity, corner_connectivity = _canonical_connectivity(
        raw_connectivity,
        coords,
        arity,
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=element_connectivity,
        elements_corner_nodes=corner_connectivity,
        element_type=element_type,
        poly_order=poly_order,
    )
    mesh._mdpa_old_to_new_node = dict(old_to_new)
    mesh._mdpa_new_to_old_node = new_to_old.copy()
    mesh._mdpa_element_ids = element_ids.copy()
    mesh._mdpa_element_id_to_index = {int(eid): i for i, eid in enumerate(element_ids)}

    for elem in mesh.elements_list:
        elem.tag = str(domain_tag)

    _tag_boundary_edges_from_mdpa(
        mesh,
        mdpa,
        boundary_node_tags=boundary_node_tags or {},
        boundary_condition_tags=boundary_condition_tags or {},
    )
    return mesh


def field_values_at_mdpa_nodes(mesh: Mesh, dof_handler, solution: np.ndarray, field: str) -> dict[int, float]:
    old_to_new = getattr(mesh, "_mdpa_old_to_new_node", None)
    new_to_old = getattr(mesh, "_mdpa_new_to_old_node", None)
    if old_to_new is None or new_to_old is None:
        raise ValueError("Mesh does not carry mdpa node maps.")

    solution = np.asarray(solution, dtype=float)
    result: dict[int, float] = {}
    for old_id in np.asarray(new_to_old, dtype=int):
        new_id = int(old_to_new[int(old_id)])
        result[int(old_id)] = _field_value_at_mesh_node(mesh, dof_handler, solution, field, new_id)
    return result


def _canonical_connectivity(raw_connectivity: Sequence[Sequence[int]], coords: np.ndarray, arity: int):
    if arity == 3:
        conn = np.asarray(raw_connectivity, dtype=int).copy()
        pts = coords[conn]
        area2 = (pts[:, 1, 0] - pts[:, 0, 0]) * (pts[:, 2, 1] - pts[:, 0, 1]) - (
            pts[:, 1, 1] - pts[:, 0, 1]
        ) * (pts[:, 2, 0] - pts[:, 0, 0])
        flip = area2 < 0.0
        if np.any(flip):
            conn[flip] = conn[flip][:, [0, 2, 1]]
        return "tri", 1, conn, conn.copy()

    if arity == 4:
        corners = np.asarray([_sort_quad_corners(conn, coords) for conn in raw_connectivity], dtype=int)
        return "quad", 1, corners[:, [0, 1, 3, 2]].copy(), corners

    if arity == 9:
        elem_conn: list[list[int]] = []
        corners_out: list[list[int]] = []
        for conn in raw_connectivity:
            lattice, corners = _reorder_q9_to_row_major(conn, coords)
            elem_conn.append(lattice)
            corners_out.append(corners)
        return "quad", 2, np.asarray(elem_conn, dtype=int), np.asarray(corners_out, dtype=int)

    raise ValueError(f"Unsupported Kratos mdpa volume element arity {arity}.")


def _sort_quad_corners(conn: Sequence[int], coords: np.ndarray) -> list[int]:
    conn_arr = np.asarray(conn, dtype=int)
    pts = coords[conn_arr]
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ccw = conn_arr[np.argsort(angles)]
    pts_ccw = coords[ccw]
    start = int(np.lexsort((pts_ccw[:, 0], pts_ccw[:, 0] + pts_ccw[:, 1]))[0])
    rolled = np.roll(ccw, -start)
    # Return [bl, br, tr, tl].
    return [int(v) for v in rolled]


def _reorder_q9_to_row_major(conn: Sequence[int], coords: np.ndarray) -> tuple[list[int], list[int]]:
    conn_arr = np.asarray(conn, dtype=int)
    corner_ids = _sort_quad_corners(conn_arr[:4], coords)
    bl, br, tr, tl = corner_ids
    corner_pts = coords[np.asarray(corner_ids, dtype=int)]
    expected = {
        "bottom": 0.5 * (corner_pts[0] + corner_pts[1]),
        "right": 0.5 * (corner_pts[1] + corner_pts[2]),
        "top": 0.5 * (corner_pts[2] + corner_pts[3]),
        "left": 0.5 * (corner_pts[3] + corner_pts[0]),
        "center": 0.25 * np.sum(corner_pts, axis=0),
    }
    remaining = [int(n) for n in conn_arr if int(n) not in set(corner_ids)]
    assigned: dict[str, int] = {}
    unused = set(remaining)
    for key, point in expected.items():
        best = min(unused, key=lambda node_id: float(np.linalg.norm(coords[int(node_id)] - point)))
        assigned[key] = int(best)
        unused.remove(best)

    lattice = [
        bl,
        assigned["bottom"],
        br,
        assigned["left"],
        assigned["center"],
        assigned["right"],
        tl,
        assigned["top"],
        tr,
    ]
    return lattice, [bl, br, tr, tl]


def _tag_boundary_edges_from_mdpa(
    mesh: Mesh,
    mdpa: KratosMDPA,
    *,
    boundary_node_tags: dict[str, str],
    boundary_condition_tags: dict[str, str],
) -> None:
    new_to_old = np.asarray(mesh._mdpa_new_to_old_node, dtype=int)
    condition_sets: list[tuple[set[int], str]] = []
    for part_name, tag in boundary_condition_tags.items():
        part = mdpa.submodelparts.get(str(part_name))
        if part is None:
            continue
        for condition_id in part.condition_ids:
            nodes = mdpa.conditions.get(int(condition_id), ())
            if nodes:
                condition_sets.append(({int(n) for n in nodes}, str(tag)))

    node_sets = [
        ({int(node_id) for node_id in mdpa.submodelparts[str(part_name)].node_ids}, str(tag))
        for part_name, tag in boundary_node_tags.items()
        if str(part_name) in mdpa.submodelparts
    ]

    for edge in mesh.edges_list:
        if edge.right is not None:
            continue
        all_nodes = edge.all_nodes if edge.all_nodes else edge.nodes
        old_edge_nodes = {int(new_to_old[int(node_id)]) for node_id in all_nodes}
        tag = None
        for condition_nodes, condition_tag in condition_sets:
            if condition_nodes.issubset(old_edge_nodes):
                tag = condition_tag
                break
        if tag is None:
            for part_nodes, node_tag in node_sets:
                if old_edge_nodes.issubset(part_nodes):
                    tag = node_tag
                    break
        if tag is not None:
            edge.tag = tag
    mesh.rebuild_edge_bitsets()


def _field_value_at_mesh_node(mesh: Mesh, dof_handler, solution: np.ndarray, field: str, mesh_node_id: int) -> float:
    dof_map = getattr(dof_handler, "dof_map", {}).get(field, {})
    if int(mesh_node_id) in dof_map:
        return float(solution[int(dof_map[int(mesh_node_id)])])

    field_order = int(dof_handler.mixed_element._field_orders[field])
    geom_order = int(mesh.poly_order)
    ref_field = get_reference(mesh.element_type, field_order)
    for eid, conn in enumerate(np.asarray(mesh.elements_connectivity, dtype=int)):
        local = np.where(conn == int(mesh_node_id))[0]
        if local.size == 0:
            continue
        xi, eta = _geom_lattice_ref_coords(mesh.element_type, geom_order, int(local[0]))
        N = np.asarray(ref_field.shape(float(xi), float(eta)), dtype=float).reshape(-1)
        gdofs = np.asarray(dof_handler.element_maps[field][eid], dtype=int)
        return float(N @ solution[gdofs])
    raise RuntimeError(f"Could not evaluate field {field!r} at mesh node {mesh_node_id}.")


def _geom_lattice_ref_coords(element_type: str, order: int, local_index: int) -> tuple[float, float]:
    if element_type == "quad":
        n = int(order) + 1
        j, i = divmod(int(local_index), n)
        pts = np.linspace(-1.0, 1.0, n)
        return float(pts[i]), float(pts[j])
    if element_type == "tri":
        if int(order) == 1:
            return [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)][int(local_index)]
    raise NotImplementedError(f"Reference coordinates for {element_type} order {order} are not implemented.")
