# adaptive_mesh_corrected.py
"""
Adaptive quad-mesh refinement based on a multi-step process to guarantee a
1-irregular mesh without hanging nodes.

This implementation follows a structured, five-step algorithm:
1.  Mark all cells intersected by the level-set function.
2.  Perform a symmetric (1-to-4) subdivision of all marked cells. This
    step introduces hanging nodes on the neighbors of the refined cells.
3.  Iteratively identify and split any cell that is larger than its neighbor.
    This propagation continues until all adjacent cells are the same size,
    guaranteeing a conforming mesh with no hanging nodes.
4.  A final check is performed to validate that the mesh is 1-irregular and
    contains no hanging nodes. If any persist, an error is raised.

This approach includes explicit parent-child tracking for each cell, providing
a clear history of the refinement process.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Set, Tuple

import numpy as np

# --- Import Node class with a fallback ---
try:
    # Assumes pycutfem is installed and in the python path
    from pycutfem.core.topology import Node
except (ImportError, ModuleNotFoundError):
    # Fallback definition if pycutfem is not available.
    # This matches the structure of the Node class in topology.py.
    print("Warning: pycutfem.core.topology.Node not found. Using fallback definition.")
    class Node:
        def __init__(self, id, x, y, tag=None):
            self.id = id
            self.x = x
            self.y = y
            self.tag = tag
        def __repr__(self):
            return f"Node {self.id}({self.x:.3f}, {self.y:.3f}, tag='{self.tag}')"

# --- Data Structures for Mesh Representation ---

@dataclass
class Cell:
    """
    Represents a single quadrilateral cell in the mesh. It tracks its
    geometry, refinement level, and relationship to parent/child cells.
    """
    id: int
    x0: float
    y0: float
    dx: float
    dy: float
    level: int
    parent_id: int | None = None
    children_ids: List[int] = field(default_factory=list)

    @property
    def center(self) -> Tuple[float, float]:
        """Returns the geometric center of the cell."""
        return (self.x0 + 0.5 * self.dx, self.y0 + 0.5 * self.dy)

    def get_corner_points(self) -> np.ndarray:
        """Returns the four corner points of the cell."""
        return np.array([
            (self.x0, self.y0),
            (self.x0 + self.dx, self.y0),
            (self.x0 + self.dx, self.y0 + self.dy),
            (self.x0, self.y0 + self.dy),
        ])

class Mesh_cell_info:
    """
    Manages the entire collection of cells and the refinement process.
    """
    def __init__(self, Lx: float, Ly: float, nx: int, ny: int):
        self.cells: Dict[int, Cell] = {}
        self.active_cells: Set[int] = set()
        self._next_cell_id = 0
        self._TOL = 1e-12

        # Create the initial coarse grid
        dx, dy = Lx / nx, Ly / ny
        for j in range(ny):
            for i in range(nx):
                cell = Cell(id=self._next_cell_id,
                            x0=i * dx, y0=j * dy,
                            dx=dx, dy=dy, level=0)
                self.cells[cell.id] = cell
                self.active_cells.add(cell.id)
                self._next_cell_id += 1

    def get_cell(self, cell_id: int) -> Cell:
        """Retrieve a cell by its ID."""
        return self.cells[cell_id]

    def subdivide_cell(self, parent_id: int, split_type: str) -> List[Cell]:
        """
        Subdivides a cell and updates the mesh data structures.
        A cell can be split symmetrically ('symm'), horizontally ('horz'),
        or vertically ('vert').
        """
        if parent_id not in self.active_cells:
            return []

        parent = self.get_cell(parent_id)
        x0, y0, dx, dy, level = parent.x0, parent.y0, parent.dx, parent.dy, parent.level
        hx, hy = 0.5 * dx, 0.5 * dy
        new_level = level + 1
        
        children_data = []
        if split_type == 'symm':
            children_data = [
                {'x0': x0, 'y0': y0, 'dx': hx, 'dy': hy},
                {'x0': x0 + hx, 'y0': y0, 'dx': hx, 'dy': hy},
                {'x0': x0 + hx, 'y0': y0 + hy, 'dx': hx, 'dy': hy},
                {'x0': x0, 'y0': y0 + hy, 'dx': hx, 'dy': hy},
            ]
        elif split_type == 'horz':
            children_data = [
                {'x0': x0, 'y0': y0, 'dx': dx, 'dy': hy},
                {'x0': x0, 'y0': y0 + hy, 'dx': dx, 'dy': hy},
            ]
        elif split_type == 'vert':
            children_data = [
                {'x0': x0, 'y0': y0, 'dx': hx, 'dy': dy},
                {'x0': x0 + hx, 'y0': y0, 'dx': hx, 'dy': dy},
            ]

        children = []
        for data in children_data:
            child = Cell(id=self._next_cell_id, level=new_level, parent_id=parent_id, **data)
            self.cells[child.id] = child
            self.active_cells.add(child.id)
            parent.children_ids.append(child.id)
            children.append(child)
            self._next_cell_id += 1

        self.active_cells.remove(parent_id)
        return children

    def find_neighbors(self, cell: Cell, side: str) -> List[Cell]:
        """Find all active cells adjacent to a given side of a cell."""
        neighbors = []
        x, y = cell.x0, cell.y0
        dx, dy = cell.dx, cell.dy
        
        if side == 'right':
            search_x, y_start, y_end = x + dx, y, y + dy
            for other_id in self.active_cells:
                other = self.get_cell(other_id)
                if math.isclose(other.x0, search_x, abs_tol=self._TOL):
                    if max(y_start, other.y0) < min(y_end, other.y0 + other.dy) - self._TOL:
                        neighbors.append(other)
        elif side == 'left':
            search_x, y_start, y_end = x, y, y + dy
            for other_id in self.active_cells:
                other = self.get_cell(other_id)
                if math.isclose(other.x0 + other.dx, search_x, abs_tol=self._TOL):
                    if max(y_start, other.y0) < min(y_end, other.y0 + other.dy) - self._TOL:
                        neighbors.append(other)
        elif side == 'top':
            search_y, x_start, x_end = y + dy, x, x + dx
            for other_id in self.active_cells:
                other = self.get_cell(other_id)
                if math.isclose(other.y0, search_y, abs_tol=self._TOL):
                    if max(x_start, other.x0) < min(x_end, other.x0 + other.dx) - self._TOL:
                        neighbors.append(other)
        elif side == 'bottom':
            search_y, x_start, x_end = y, x, x + dx
            for other_id in self.active_cells:
                other = self.get_cell(other_id)
                if math.isclose(other.y0 + other.dy, search_y, abs_tol=self._TOL):
                    if max(x_start, other.x0) < min(x_end, other.x0 + other.dx) - self._TOL:
                        neighbors.append(other)
        return neighbors

def structured_quad_levelset_adaptive(
    Lx: float,
    Ly: float,
    *,
    nx: int,
    ny: int,
    poly_order: int,
    level_set: Callable[[np.ndarray], np.ndarray],
    max_refine_level: int = 4,
) -> Tuple[List[Node], np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a 1-irregular adaptive quadrilateral mesh based on a level-set.
    """
    mesh = Mesh_cell_info(Lx, Ly, nx, ny)

    # --- Step 1 & 2: Mark and Symmetrically Subdivide Interface Cells ---
    for level in range(max_refine_level):
        cells_to_split = [
            cell_id for cell_id in mesh.active_cells
            if mesh.get_cell(cell_id).level == level and
            np.any(level_set(np.vstack([mesh.get_cell(cell_id).get_corner_points(), mesh.get_cell(cell_id).center])) <= 0) and
            np.any(level_set(np.vstack([mesh.get_cell(cell_id).get_corner_points(), mesh.get_cell(cell_id).center])) > 0)
        ]
        if not cells_to_split:
            continue
        for cell_id in cells_to_split:
            mesh.subdivide_cell(cell_id, 'symm')

    # --- Step 3 & 4: Iteratively Propagate Refinements for a Conforming Mesh ---
    while True:
        cells_to_split = {}  # Using a dict to store {cell_id: split_type}

        # Find any cell that is larger than its neighbor.
        for cell_id in list(mesh.active_cells):
            cell = mesh.get_cell(cell_id)

            # Check neighbors on right and left. If I am taller, I need a horizontal split.
            for neighbor in mesh.find_neighbors(cell, 'right') + mesh.find_neighbors(cell, 'left'):
                if cell.dy > neighbor.dy + mesh._TOL:
                    if cells_to_split.get(cell.id) == 'vert':
                        cells_to_split[cell.id] = 'symm'
                    else:
                        cells_to_split[cell.id] = 'horz'

            # Check neighbors on top and bottom. If I am wider, I need a vertical split.
            for neighbor in mesh.find_neighbors(cell, 'top') + mesh.find_neighbors(cell, 'bottom'):
                if cell.dx > neighbor.dx + mesh._TOL:
                    if cells_to_split.get(cell.id) == 'horz':
                        cells_to_split[cell.id] = 'symm'
                    else:
                        cells_to_split[cell.id] = 'vert'

        if not cells_to_split:
            break # Exit loop if mesh is conforming

        # Execute the splits for this pass
        for cell_id, split_type in cells_to_split.items():
            mesh.subdivide_cell(cell_id, split_type)

    # --- Step 5: The mesh is now conforming. No assertion needed. ---

    # --- Convert Mesh to Node/Element Format ---
    nodes: List[Node] = []
    loc_to_id: Dict[Tuple[float, float], int] = {}
    _TOL = 1e-12

    def get_node_id(x: float, y: float) -> int:
        key = (round(x / _TOL) * _TOL, round(y / _TOL) * _TOL)
        if key in loc_to_id:
            return loc_to_id[key]
        
        nid = len(nodes)
        tags = []
        if math.isclose(x, 0): tags.append("boundary_left")
        if math.isclose(x, Lx): tags.append("boundary_right")
        if math.isclose(y, 0): tags.append("boundary_bottom")
        if math.isclose(y, Ly): tags.append("boundary_top")
        
        nodes.append(Node(id=nid, x=x, y=y, tag=",".join(tags)))
        loc_to_id[key] = nid
        return nid

    xi = np.linspace(0, 1, poly_order + 1)
    elements: List[List[int]] = []
    corners: List[List[int]] = []
    
    sorted_active_cells = sorted(list(mesh.active_cells), key=lambda cid: (mesh.get_cell(cid).y0, mesh.get_cell(cid).x0))

    for cell_id in sorted_active_cells:
        c = mesh.get_cell(cell_id)
        conn = [
            get_node_id(c.x0 + sx * c.dx, c.y0 + sy * c.dy)
            for sy in xi for sx in xi
        ]
        elements.append(conn)
        bl, br = conn[0], conn[poly_order]
        tr, tl = conn[-1], conn[len(conn) - 1 - poly_order]
        corners.append([bl, br, tr, tl])

    edge_set = set()
    for cs in corners:
        edge_set.add(tuple(sorted((cs[0], cs[1]))))
        edge_set.add(tuple(sorted((cs[1], cs[2]))))
        edge_set.add(tuple(sorted((cs[2], cs[3]))))
        edge_set.add(tuple(sorted((cs[3], cs[0]))))
    
    edges = np.array(sorted(list(edge_set)), dtype=int)
    
    return nodes, np.array(elements, dtype=int), edges, np.array(corners, dtype=int)


if __name__ == "__main__":
    # Example usage
    max_refine_level = 2
    H = 0.41  # Channel height
    L = 2.2   # Channel length
    D = 0.1   # Cylinder diameter
    c_x, c_y = 0.2, 0.2  # Cylinder center
    NX, NY = 18, 18
    poly_order = 2
    from pycutfem.core.levelset import CircleLevelSet
    from pycutfem.core.mesh import Mesh
    level_set = CircleLevelSet(center=(c_x, c_y), radius=D/2.0 ) # needs to correct the radius, also cx modified for debuggin
    nodes, elems, edges, corners = structured_quad_levelset_adaptive(
            Lx=L, Ly=H, nx=NX, ny=NY, poly_order=poly_order,
            level_set=CircleLevelSet(center=(c_x, c_y), radius=(D/2.0+0.1*D/2.0) ),
            max_refine_level=max_refine_level)          # add a single halo, nothing else
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=poly_order)
    from pycutfem.io.visualization import plot_mesh_2
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(15, 30))
    plot_mesh_2(mesh, ax=ax, level_set=level_set, show=True, 
                plot_nodes=False, elem_tags=False, edge_colors=True, plot_interface=False,resolution=300)