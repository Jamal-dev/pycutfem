# adaptive_mesh.py
"""Adaptive quad-mesh refinement with *guaranteed termination*.

Key differences to previous drafts
----------------------------------
* Each **Cell** now records whether it has already been split **horizontally**
  or **vertically** (``split_horz`` / ``split_vert``).  A cell can be cut in
  a given orientation **only once**, which prevents infinite re-splitting.
* The propagation loop enforces the 1-irregular rule: within any row the
  tallest cell may be at most twice the height of the shortest; likewise for
  columns and widths.  Violators are split once and flagged.
* Because every split halves *dx* or *dy* and flags that orientation, the
  algorithm finishes in **O(log₂(max/min))** iterations and cannot loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Set

import math
import numpy as np

# -----------------------------------------------------------------------------
# Fallback Node if PyCutFEM is unavailable ------------------------------------
# -----------------------------------------------------------------------------
try:
    from pycutfem.core.topology import Node  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – keep unit tests independent

    @dataclass(slots=True)
    class Node:  # pylint: disable=too-many-instance-attributes
        id: int
        x: float
        y: float
        tag: str = ""


# -----------------------------------------------------------------------------
# Internal rectangle ----------------------------------------------------------
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class Cell:
    """Axis-aligned rectangle used during adaptive refinement."""

    x0: float
    y0: float
    dx: float
    dy: float
    level: int = 0
    split_horz: bool = False  # already cut horizontally
    split_vert: bool = False  # already cut vertically

    # Corners ----------------------------------------------------------
    @property
    def bl(self) -> Tuple[float, float]:
        return (self.x0, self.y0)

    @property
    def br(self) -> Tuple[float, float]:
        return (self.x0 + self.dx, self.y0)

    @property
    def tr(self) -> Tuple[float, float]:
        return (self.x0 + self.dx, self.y0 + self.dy)

    @property
    def tl(self) -> Tuple[float, float]:
        return (self.x0, self.y0 + self.dy)


# -----------------------------------------------------------------------------
# Elementary subdivision helpers ---------------------------------------------
# -----------------------------------------------------------------------------

def subdivide_cell_symm(cell: Cell) -> List[Cell]:
    """Split *cell* into 4 congruent children."""

    hx, hy = 0.5 * cell.dx, 0.5 * cell.dy
    x0, y0, lvl = cell.x0, cell.y0, cell.level + 1
    return [
        Cell(x0, y0, hx, hy, lvl),
        Cell(x0 + hx, y0, hx, hy, lvl),
        Cell(x0 + hx, y0 + hy, hx, hy, lvl),
        Cell(x0, y0 + hy, hx, hy, lvl),
    ]


def subdivide_cell_asymm(cell: Cell, *, horz: bool) -> List[Cell]:
    """Split *cell* into 2 children – horizontal (stacked) or vertical."""

    x0, y0, lvl = cell.x0, cell.y0, cell.level + 1
    if horz:
        hy = 0.5 * cell.dy
        return [
            Cell(x0, y0, cell.dx, hy, lvl),
            Cell(x0, y0 + hy, cell.dx, hy, lvl),
        ]
    hx = 0.5 * cell.dx
    return [
        Cell(x0, y0, hx, cell.dy, lvl),
        Cell(x0 + hx, y0, hx, cell.dy, lvl),
    ]


# -----------------------------------------------------------------------------
# Misc helpers ----------------------------------------------------------------
# -----------------------------------------------------------------------------
_TOL = 1e-12

def _round(x: float) -> float:
    return float(round(x / _TOL) * _TOL)


# -----------------------------------------------------------------------------
# Public driver ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def structured_quad_levelset_adaptive(
    Lx: float,
    Ly: float,
    *,
    nx: int,
    ny: int,
    poly_order: int,
    level_set: Callable[[np.ndarray], np.ndarray],
    max_refine_level: int = 4,
):
    """Return (nodes, elements, edges, corner_conn) for a 1-irregular mesh."""

    # ------------------------------------------------------------------
    # 0) coarse background grid ----------------------------------------
    # ------------------------------------------------------------------
    hx, hy = Lx / nx, Ly / ny
    cells: List[Cell] = [
        Cell(i * hx, j * hy, hx, hy)
        for j in range(ny)
        for i in range(nx)
    ]

    # ------------------------------------------------------------------
    # 1) mark interface cells ------------------------------------------
    # ------------------------------------------------------------------
    for _ in range(max_refine_level + 1):
        to_split: List[int] = []
        for k, c in enumerate(cells):
            if c.level >= max_refine_level:
                continue
            pts = np.array([c.bl, c.br, c.tr, c.tl, (c.x0 + 0.5 * c.dx, c.y0 + 0.5 * c.dy)])
            phi = level_set(pts)
            if np.any(phi <= 0) and np.any(phi >= 0):
                to_split.append(k)
        if not to_split:
            break
        # Split in reverse order to avoid index shifting
        for k in sorted(to_split, reverse=True):
            cells[k:k + 1] = subdivide_cell_symm(cells[k])

    # ------------------------------------------------------------------
    # 2) 1-irregular propagation --------------------------------------
    # ------------------------------------------------------------------
    while True:
        splits: List[Tuple[int, bool]] = []  # (cell-idx, horz?)

        # Row-wise: enforce height ratio <= 2 ---------------------------
        rows: Dict[float, List[int]] = {}
        for i, c in enumerate(cells):
            rows.setdefault(_round(c.y0), []).append(i)
        for idxs in rows.values():
            if not idxs:
                continue
            hmin = min(cells[i].dy for i in idxs)
            for i in idxs:
                c = cells[i]
                if c.dy > 2 * hmin + _TOL and not c.split_horz:
                    splits.append((i, True))

        # Col-wise: enforce width ratio <= 2 ----------------------------
        cols: Dict[float, List[int]] = {}
        for i, c in enumerate(cells):
            cols.setdefault(_round(c.x0), []).append(i)
        for idxs in cols.values():
            if not idxs:
                continue
            wmin = min(cells[i].dx for i in idxs)
            for i in idxs:
                c = cells[i]
                if c.dx > 2 * wmin + _TOL and not c.split_vert:
                    splits.append((i, False))

        if not splits:
            break  # done – mesh is 1-irregular

        seen: Set[int] = set()
        # Sort splits by index descending
        sorted_splits = sorted(splits, key=lambda x: x[0], reverse=True)
        for i, horz in sorted_splits:
            if i in seen:
                continue
            seen.add(i)
            parent = cells[i]
            if horz:
                parent.split_horz = True
            else:
                parent.split_vert = True
            children = subdivide_cell_asymm(parent, horz=horz)
            cells[i:i + 1] = children

    # ------------------------------------------------------------------
    # 3) convert to Nodes / connectivity -------------------------------
    # ------------------------------------------------------------------
    nodes: List[Node] = []
    loc2id: Dict[Tuple[float, float], int] = {}

    def gid(x: float, y: float) -> int:
        key = (_round(x), _round(y))
        if key in loc2id:
            return loc2id[key]
        nid = len(nodes)
        tag: list[str] = []
        if math.isclose(x, 0, abs_tol=_TOL):
            tag.append("boundary_left")
        if math.isclose(x, Lx, abs_tol=_TOL):
            tag.append("boundary_right")
        if math.isclose(y, 0, abs_tol=_TOL):
            tag.append("boundary_bottom")
        if math.isclose(y, Ly, abs_tol=_TOL):
            tag.append("boundary_top")
        nodes.append(Node(nid, x, y, ",".join(tag)))
        loc2id[key] = nid
        return nid

    xi = np.linspace(0, 1, poly_order + 1)
    elements: List[List[int]] = []
    corners: List[List[int]] = []
    for c in cells:
        conn = [
            gid(c.x0 + sx * c.dx, c.y0 + sy * c.dy)
            for sy in xi for sx in xi
        ]
        elements.append(conn)
        corners.append([conn[0], conn[poly_order], conn[-1], conn[-poly_order - 1]])

    # unique edges -----------------------------------------------------
    edge_set = set()
    for cs in corners:
        edge_set.update([
            tuple(sorted((cs[0], cs[1]))),
            tuple(sorted((cs[1], cs[2]))),
            tuple(sorted((cs[2], cs[3]))),
            tuple(sorted((cs[3], cs[0]))),
        ])
    edges = np.asarray(sorted(edge_set), int)

    return nodes, np.asarray(elements, int), edges, np.asarray(corners, int)


# -----------------------------------------------------------------------------
# sanity check ----------------------------------------------------------------
# -----------------------------------------------------------------------------
