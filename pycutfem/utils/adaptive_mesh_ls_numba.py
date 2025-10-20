# adaptive_mesh_numba_debug.py
"""
Numba-friendly adaptive quad-mesh refinement with a debug switch.

This implementation has been refactored from a class-based to a procedural
approach to be compatible with Numba's JIT compilation. It uses NumPy
structured arrays to store cell data, enabling high-performance computation.

To debug this file with standard Python tools (like pdb), set the `DEBUG`
flag at the top of the file to `True`. This will disable Numba's JIT
compilation, allowing you to step through the logic.
"""
from __future__ import annotations
import numpy as np
import numba

# --- Debugging Switch ---
# Set to True to disable Numba's JIT compilation for easier debugging.
DEBUG = False

if DEBUG:
    print("--- NUMBA JIT DISABLED (DEBUG MODE) ---")
    # Create a dummy decorator that does nothing, just returns the function
    def jit_compile(*args, **kwargs):
        if len(args) == 1 and callable(args[0]): # Called as @jit_compile
            return args[0]
        else: # Called as @jit_compile(nopython=True, ...)
            def wrapper(func):
                return func
            return wrapper
    # In debug mode, prange is just the standard Python range
    prange = range
else:
    # When not debugging, use the real Numba JIT decorator and prange
    from numba.typed import Dict
    from numba.core import types
    jit_compile = numba.jit
    prange = numba.prange

# --- Fallback Node Class ---
try:
    from pycutfem.core.topology import Node
except (ImportError, ModuleNotFoundError):
    print("Warning: pycutfem.core.topology.Node not found. Using fallback definition.")
    class Node:
        def __init__(self, id, x, y, tag=None):
            self.id = id; self.x = x; self.y = y; self.tag = tag
        def __repr__(self):
            return f"Node {self.id}({self.x:.3f}, {self.y:.3f}, tag='{self.tag}')"

# --- Numba-compatible Data Structures and Types ---
cell_dtype = np.dtype([
    ('id', np.int32), ('x0', np.float64), ('y0', np.float64),
    ('dx', np.float64), ('dy', np.float64), ('level', np.int32),
    ('parent_id', np.int32), ('active', np.bool_)
])

# Python implementations for debug mode or fallback
def py_unique_rows(a):
    """
    Finds unique rows in a 2D array using pure Python.
    """
    seen = set()
    unique_list = []
    for i in range(a.shape[0]):
        row_tuple = tuple(a[i])
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_list.append(a[i])
    return np.array(unique_list, dtype=a.dtype)

def py_unique_rows_with_inverse(a):
    """
    Finds unique rows and the inverse map using pure Python.
    """
    uniques_map = {}
    uniques_list = []
    inverse = np.empty(a.shape[0], dtype=np.int64)
    unique_count = 0
    for i in range(a.shape[0]):
        row = tuple(a[i])
        if row in uniques_map:
            inverse[i] = uniques_map[row]
        else:
            new_idx = unique_count
            uniques_map[row] = new_idx
            inverse[i] = new_idx
            uniques_list.append(row)
            unique_count += 1
    uniques = np.array(uniques_list, dtype=a.dtype)
    return uniques, inverse

if not DEBUG:
    unituple_int32_2 = types.UniTuple(types.int32, 2)
    unituple_int64_2 = types.UniTuple(types.int64, 2)
    unituple_float64_2 = types.UniTuple(types.float64, 2)

    @jit_compile(nopython=True, cache=True)
    def unique_rows_int32_impl(a):
        if a.shape[0] == 0:
            return np.empty((0, 2), dtype=np.int32)
        
        seen_dict = Dict.empty(
            key_type=unituple_int32_2,
            value_type=types.boolean
        )
        
        unique_list = []
        for i in range(a.shape[0]):
            row_tuple = (a[i, 0], a[i, 1])
            if row_tuple not in seen_dict:
                seen_dict[row_tuple] = True
                unique_list.append(row_tuple)
                
        res = np.empty((len(unique_list), 2), dtype=np.int32)
        for i in range(len(unique_list)):
            res[i, 0] = unique_list[i][0]
            res[i, 1] = unique_list[i][1]
        return res

    @jit_compile(nopython=True, cache=True)
    def unique_rows_int64_impl(a):
        if a.shape[0] == 0:
            return np.empty((0, 2), dtype=np.int64)
        
        seen_dict = Dict.empty(
            key_type=unituple_int64_2,
            value_type=types.boolean
        )
        
        unique_list = []
        for i in range(a.shape[0]):
            row_tuple = (a[i, 0], a[i, 1])
            if row_tuple not in seen_dict:
                seen_dict[row_tuple] = True
                unique_list.append(row_tuple)
                
        res = np.empty((len(unique_list), 2), dtype=np.int64)
        for i in range(len(unique_list)):
            res[i, 0] = unique_list[i][0]
            res[i, 1] = unique_list[i][1]
        return res

    @jit_compile(nopython=True, cache=True)
    def unique_rows_with_inverse_float64_impl(a):
        if a.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.int64)

        uniques_map = Dict.empty(
            key_type=unituple_float64_2,
            value_type=types.int64
        )
        
        uniques_list = []
        inverse = np.empty(a.shape[0], dtype=np.int64)
        
        unique_count = 0
        for i in range(a.shape[0]):
            row = (a[i, 0], a[i, 1])
            if row in uniques_map:
                inverse[i] = uniques_map[row]
            else:
                new_idx = unique_count
                uniques_map[row] = new_idx
                inverse[i] = new_idx
                uniques_list.append(row)
                unique_count += 1
                
        uniques = np.empty((len(uniques_list), 2), dtype=np.float64)
        for i in range(len(uniques_list)):
            uniques[i, 0] = uniques_list[i][0]
            uniques[i, 1] = uniques_list[i][1]

        return uniques, inverse

# Assign appropriate functions based on DEBUG mode
if DEBUG:
    unique_rows_int32 = py_unique_rows
    unique_rows_int64 = py_unique_rows
    unique_rows_with_inverse_float64 = py_unique_rows_with_inverse
else:
    unique_rows_int32 = unique_rows_int32_impl
    unique_rows_int64 = unique_rows_int64_impl
    unique_rows_with_inverse_float64 = unique_rows_with_inverse_float64_impl

# --- Main Adaptive Meshing Logic ---

def structured_quad_levelset_adaptive(
    Lx: float, Ly: float, *, nx: int, ny: int, poly_order: int,
    level_set: callable, max_refine_level: int = 4, parallel: bool = True,
) -> tuple[list[Node], np.ndarray, np.ndarray, np.ndarray]:
    # --- 1. Initial Coarse Grid ---
    initial_cells = np.empty(nx * ny, dtype=cell_dtype)
    dx, dy = Lx / nx, Ly / ny
    idx = 0
    for j in range(ny):
        for i in range(nx):
            initial_cells[idx]['id'] = idx; initial_cells[idx]['x0'] = i * dx
            initial_cells[idx]['y0'] = j * dy; initial_cells[idx]['dx'] = dx
            initial_cells[idx]['dy'] = dy; initial_cells[idx]['level'] = 0
            initial_cells[idx]['parent_id'] = -1; initial_cells[idx]['active'] = True
            idx += 1
    
    cells = initial_cells
    
    # --- 2. Level-set based Refinement ---
    for level in range(max_refine_level):
        active_indices = np.where((cells['level'] == level) & cells['active'])[0]
        if len(active_indices) == 0: continue
        
        corners = np.array([[[c['x0'], c['y0']], [c['x0'] + c['dx'], c['y0']], [c['x0'] + c['dx'], c['y0'] + c['dy']], [c['x0'], c['y0'] + c['dy']]] for c in cells[active_indices]], dtype=np.float64)
        centers = np.array([[c['x0'] + 0.5 * c['dx'], c['y0'] + 0.5 * c['dy']] for c in cells[active_indices]], dtype=np.float64)

        if len(active_indices) > 0:
            ls_values = level_set(np.vstack((corners.reshape(-1, 2), centers)))
            ls_corners = ls_values[:len(active_indices)*4].reshape(len(active_indices), 4)
            ls_centers = ls_values[len(active_indices)*4:]
            cut_mask = (np.any(ls_corners <= 0, axis=1) | (ls_centers <= 0)) & (np.any(ls_corners > 0, axis=1) | (ls_centers > 0))
            indices_to_split = active_indices[cut_mask]
            if len(indices_to_split) > 0:
                cells = _subdivide_cells(cells, indices_to_split, np.full(len(indices_to_split), 0, dtype=np.int32))

    # --- 3. Enforce 1-Irregular Conforming Mesh ---
    while True:
        split_requests = _find_conforming_splits(cells, parallel and not DEBUG)
        if len(split_requests) == 0: break
        
        unique_indices, inv_indices = np.unique(split_requests[:, 0], return_inverse=True)
        split_types = np.zeros(len(unique_indices), dtype=np.int32)
        
        for i, req in enumerate(split_requests):
            split_types[inv_indices[i]] |= req[1]
        cells = _subdivide_cells(cells, unique_indices, split_types)

    # --- 4. Generate Final FEM Mesh from Active Cells ---
    active_cells = cells[cells['active']]
    nodes_coords, elements, edges, corners = _generate_fem_mesh_from_cells(active_cells, Lx, Ly, poly_order, parallel and not DEBUG)
    
    # --- 5. Convert raw coordinates to Node objects ---
    node_objects = [Node(id=i, x=coord[0], y=coord[1]) for i, coord in enumerate(nodes_coords)]
    return node_objects, elements, edges, corners

@jit_compile(nopython=True, parallel=True, cache=True)
def _subdivide_cells(cells, parent_indices, split_types):
    num_parents = len(parent_indices)
    if num_parents == 0:
        return cells
    
    num_children = np.empty(num_parents, dtype=np.int32)
    for i in prange(num_parents):
        stype = split_types[i]
        if stype == 1 or stype == 2:
            num_children[i] = 2
        else:
            num_children[i] = 4
    
    offsets = np.cumsum(num_children)
    offsets = np.concatenate((np.array([0]), offsets[:-1]))
    
    num_new_cells = offsets[-1] + num_children[-1]
    new_cells_arr = np.empty(num_new_cells, dtype=cell_dtype)
    next_cell_id = len(cells)
    
    for i in prange(num_parents):
        cells[parent_indices[i]]['active'] = False
    
    for i in prange(num_parents):
        parent_idx = parent_indices[i]
        parent = cells[parent_idx]
        x0, y0, dx, dy = parent['x0'], parent['y0'], parent['dx'], parent['dy']
        hx, hy = 0.5 * dx, 0.5 * dy
        new_level = parent['level'] + 1
        stype = split_types[i]
        p_idx_32 = np.int32(parent_idx)
        n_lvl_32 = np.int32(new_level)
        start_idx = offsets[i]
        child_id_start = np.int32(next_cell_id + start_idx)
        
        if stype & 1 and not (stype & 2):  # Horizontal
            new_cells_arr[start_idx]['id'] = child_id_start
            new_cells_arr[start_idx]['x0'] = x0
            new_cells_arr[start_idx]['y0'] = y0
            new_cells_arr[start_idx]['dx'] = dx
            new_cells_arr[start_idx]['dy'] = hy
            new_cells_arr[start_idx]['level'] = n_lvl_32
            new_cells_arr[start_idx]['parent_id'] = p_idx_32
            new_cells_arr[start_idx]['active'] = True
            
            new_cells_arr[start_idx+1]['id'] = child_id_start + 1
            new_cells_arr[start_idx+1]['x0'] = x0
            new_cells_arr[start_idx+1]['y0'] = y0 + hy
            new_cells_arr[start_idx+1]['dx'] = dx
            new_cells_arr[start_idx+1]['dy'] = hy
            new_cells_arr[start_idx+1]['level'] = n_lvl_32
            new_cells_arr[start_idx+1]['parent_id'] = p_idx_32
            new_cells_arr[start_idx+1]['active'] = True
        elif stype & 2 and not (stype & 1):  # Vertical
            new_cells_arr[start_idx]['id'] = child_id_start
            new_cells_arr[start_idx]['x0'] = x0
            new_cells_arr[start_idx]['y0'] = y0
            new_cells_arr[start_idx]['dx'] = hx
            new_cells_arr[start_idx]['dy'] = dy
            new_cells_arr[start_idx]['level'] = n_lvl_32
            new_cells_arr[start_idx]['parent_id'] = p_idx_32
            new_cells_arr[start_idx]['active'] = True
            
            new_cells_arr[start_idx+1]['id'] = child_id_start + 1
            new_cells_arr[start_idx+1]['x0'] = x0 + hx
            new_cells_arr[start_idx+1]['y0'] = y0
            new_cells_arr[start_idx+1]['dx'] = hx
            new_cells_arr[start_idx+1]['dy'] = dy
            new_cells_arr[start_idx+1]['level'] = n_lvl_32
            new_cells_arr[start_idx+1]['parent_id'] = p_idx_32
            new_cells_arr[start_idx+1]['active'] = True
        else:  # Symmetric
            new_cells_arr[start_idx]['id'] = child_id_start
            new_cells_arr[start_idx]['x0'] = x0
            new_cells_arr[start_idx]['y0'] = y0
            new_cells_arr[start_idx]['dx'] = hx
            new_cells_arr[start_idx]['dy'] = hy
            new_cells_arr[start_idx]['level'] = n_lvl_32
            new_cells_arr[start_idx]['parent_id'] = p_idx_32
            new_cells_arr[start_idx]['active'] = True
            
            new_cells_arr[start_idx+1]['id'] = child_id_start + 1
            new_cells_arr[start_idx+1]['x0'] = x0 + hx
            new_cells_arr[start_idx+1]['y0'] = y0
            new_cells_arr[start_idx+1]['dx'] = hx
            new_cells_arr[start_idx+1]['dy'] = hy
            new_cells_arr[start_idx+1]['level'] = n_lvl_32
            new_cells_arr[start_idx+1]['parent_id'] = p_idx_32
            new_cells_arr[start_idx+1]['active'] = True
            
            new_cells_arr[start_idx+2]['id'] = child_id_start + 2
            new_cells_arr[start_idx+2]['x0'] = x0 + hx
            new_cells_arr[start_idx+2]['y0'] = y0 + hy
            new_cells_arr[start_idx+2]['dx'] = hx
            new_cells_arr[start_idx+2]['dy'] = hy
            new_cells_arr[start_idx+2]['level'] = n_lvl_32
            new_cells_arr[start_idx+2]['parent_id'] = p_idx_32
            new_cells_arr[start_idx+2]['active'] = True
            
            new_cells_arr[start_idx+3]['id'] = child_id_start + 3
            new_cells_arr[start_idx+3]['x0'] = x0
            new_cells_arr[start_idx+3]['y0'] = y0 + hy
            new_cells_arr[start_idx+3]['dx'] = hx
            new_cells_arr[start_idx+3]['dy'] = hy
            new_cells_arr[start_idx+3]['level'] = n_lvl_32
            new_cells_arr[start_idx+3]['parent_id'] = p_idx_32
            new_cells_arr[start_idx+3]['active'] = True
    
    return np.concatenate((cells, new_cells_arr))

@jit_compile(nopython=True, parallel=True, cache=True)
def _find_conforming_splits(cells, parallel: bool):
    TOL = 1e-12; active_indices = np.where(cells['active'])[0]
    num_active = len(active_indices); requests = np.full((num_active * 4, 2), -1, dtype=np.int32)
    for i in prange(num_active) if parallel else range(num_active):
        cell_idx = active_indices[i]; c = cells[cell_idx]; request_idx = i * 4
        for other_idx in active_indices:
            if cell_idx == other_idx: continue
            o = cells[other_idx]
            if abs(c['x0'] + c['dx'] - o['x0']) < TOL and max(c['y0'], o['y0']) < min(c['y0'] + c['dy'], o['y0'] + o['dy']) - TOL:
                if c['dy'] > o['dy'] + TOL: requests[request_idx, 0] = cell_idx; requests[request_idx, 1] = 1
            if abs(c['x0'] - (o['x0'] + o['dx'])) < TOL and max(c['y0'], o['y0']) < min(c['y0'] + c['dy'], o['y0'] + o['dy']) - TOL:
                if c['dy'] > o['dy'] + TOL: requests[request_idx+1, 0] = cell_idx; requests[request_idx+1, 1] = 1
            if abs(c['y0'] + c['dy'] - o['y0']) < TOL and max(c['x0'], o['x0']) < min(c['x0'] + c['dx'], o['x0'] + o['dx']) - TOL:
                if c['dx'] > o['dx'] + TOL: requests[request_idx+2, 0] = cell_idx; requests[request_idx+2, 1] = 2
            if abs(c['y0'] - (o['y0'] + o['dy'])) < TOL and max(c['x0'], o['x0']) < min(c['x0'] + c['dx'], o['x0'] + o['dx']) - TOL:
                if c['dx'] > o['dx'] + TOL: requests[request_idx+3, 0] = cell_idx; requests[request_idx+3, 1] = 2
    final_mask = requests[:, 0] != -1
    if np.any(final_mask): return unique_rows_int32(requests[final_mask])
    return np.empty((0, 2), dtype=np.int32)

@jit_compile(nopython=True, parallel=True, cache=True)
def _generate_fem_mesh_from_cells(active_cells, Lx, Ly, poly_order, parallel):
    num_cells = len(active_cells); nodes_per_elem = (poly_order + 1)**2
    all_node_coords = np.empty((num_cells * nodes_per_elem, 2), dtype=np.float64)
    xi = np.linspace(0, 1, poly_order + 1)
    for i in prange(num_cells) if parallel else range(num_cells):
        c = active_cells[i]; offset = i * nodes_per_elem; k = 0
        for sy in xi:
            for sx in xi:
                all_node_coords[offset + k, 0] = c['x0'] + sx * c['dx']
                all_node_coords[offset + k, 1] = c['y0'] + sy * c['dy']
                k += 1
    TOL = 1e-9; rounded_coords = np.round(all_node_coords / TOL) * TOL
    unique_nodes, inverse_map = unique_rows_with_inverse_float64(rounded_coords)
    elements = inverse_map.reshape(num_cells, nodes_per_elem)
    elements_corner_nodes = np.empty((num_cells, 4), dtype=np.int64)
    all_edges = np.empty((num_cells * 4, 2), dtype=np.int64)
    for i in prange(num_cells) if parallel else range(num_cells):
        conn = elements[i]; bl, br = conn[0], conn[poly_order]
        tl, tr = conn[nodes_per_elem - 1 - poly_order], conn[-1]
        corners = np.array([bl, br, tr, tl]); elements_corner_nodes[i] = corners
        edge_offset = i * 4
        all_edges[edge_offset+0] = np.sort(np.array([corners[0], corners[1]]))
        all_edges[edge_offset+1] = np.sort(np.array([corners[1], corners[2]]))
        all_edges[edge_offset+2] = np.sort(np.array([corners[2], corners[3]]))
        all_edges[edge_offset+3] = np.sort(np.array([corners[3], corners[0]]))
    if len(all_edges) > 0: edges = unique_rows_int64(all_edges)
    else: edges = np.empty((0,2), dtype=np.int64)
    return unique_nodes, elements, edges, elements_corner_nodes

if __name__ == '__main__':
    try:
        from pycutfem.core.levelset import CircleLevelSet
        from pycutfem.core.mesh import Mesh
        from pycutfem.io.visualization import plot_mesh_2
        import matplotlib.pyplot as plt
        import time
    except ImportError:
        print("Skipping example: pycutfem, levelset_numba, or matplotlib not found.")
    else:
        max_refine_level = 1; H, L, D = 0.41, 2.2, 0.1
        c_x, c_y = 0.4, 0.2; NX, NY = 70, 60; poly_order = 1
        circle_ls = CircleLevelSet(center=(c_x, c_y), radius=D/2.0 + 0.1*D/2.0)
        print(f"--- Running Numba Adaptive Mesh (DEBUG={DEBUG}) ---")
        start_time = time.time()
        nodes, elems, edges, corners = structured_quad_levelset_adaptive(
            Lx=L, Ly=H, nx=NX, ny=NY, poly_order=poly_order,
            level_set=circle_ls, max_refine_level=max_refine_level, parallel=True)
        duration = time.time() - start_time
        print(f"Generated {len(nodes)} nodes and {len(elems)} elements in {duration:.4f} seconds.")
        mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=poly_order)
        fig, ax = plt.subplots(figsize=(16, 8))
        plot_mesh_2(mesh, ax=ax, level_set=circle_ls, show=False, plot_nodes=False, elem_tags=False, edge_colors=True, plot_interface=True, resolution=300)
        ax.set_title(f"Numba Adaptive Mesh ({len(elems)} elements, DEBUG={DEBUG})")
        plt.show()