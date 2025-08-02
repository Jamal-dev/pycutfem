"""pycutfem.utils.meshgen
Mesh generators for quick tests.
"""
import numpy as np
from scipy.spatial import Delaunay
from pycutfem.io.visualization import visualize_mesh_node_order
from pycutfem.core.topology import Node
from typing import List, Tuple, Dict, Optional
import numba
from numba.core import types
from numba.typed import Dict

__all__ = ["delaunay_rectangle", "structured_quad", "structured_triangles"]


def delaunay_rectangle(length: float, height: float, nx: int = 10, ny: int = 10):
    x = np.linspace(0.0, length, nx)
    y = np.linspace(0.0, height, ny)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    tri = Delaunay(pts)
    elems = tri.simplices.copy()

    # make triangles CCW
    def signed_area(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    for t in elems:
        a, b, c = pts[t]
        if signed_area(a, b, c) < 0:
            t[1], t[2] = t[2], t[1]
    return pts, elems

def _translate_nodes(nodes: List[Node], offset: Tuple[float, float]):
    """Translates all nodes by a given offset vector."""
    tx, ty = offset
    if not nodes:
        return
    # It's more efficient to modify a numpy array of coordinates
    # and then update the node objects.
    coords = np.array([[node.x, node.y] for node in nodes])
    coords[:, 0] += tx
    coords[:, 1] += ty
    for i, node in enumerate(nodes):
        node.x = coords[i, 0]
        node.y = coords[i, 1]

@numba.jit(nopython=True, cache=True)
def _translate_coords(coords: np.ndarray, offset: np.ndarray):
    """Translates all node coordinates by a given offset vector."""
    coords[:, 0] += offset[0]
    coords[:, 1] += offset[1]
    return coords

# Python implementations for fallback
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

unituple_int64_2 = types.UniTuple(types.int64, 2)

@numba.jit(nopython=True, cache=True)
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

unique_rows_int64 = unique_rows_int64_impl

def structured_quad(Lx: float, Ly: float, *, nx: int, ny: int, poly_order: int, 
                    offset: Optional[Tuple[float, float]] = None,numba_path=True, parallel: bool = True):
    """
    Main wrapper for generating structured quadrilateral meshes.
    Returns raw data: node objects, element connectivity, edge connectivity,
    and corner node connectivity for each element.
    """
    if not numba_path:
        return _structured_qn(Lx, Ly, nx, ny, poly_order, offset)
    else:
        # 1. Call the fast Numba core to get raw NumPy arrays
        nodes_coords, elements, edges, elements_corner_nodes = _structured_qn_numba(
            Lx, Ly, nx, ny, poly_order, parallel
        )
        if offset is not None:
            nodes_coords = _translate_coords(nodes_coords, np.array(offset, dtype=np.float64))

        # 2. Convert the raw coordinate array back to a list of Node objects
        # This loop is fast and runs in standard Python.
        node_objects = [
            Node(id=i, x=coord[0], y=coord[1])
            for i, coord in enumerate(nodes_coords)
        ]

        # 3. Return the Node objects and other connectivity data
        return node_objects, elements, edges, elements_corner_nodes

@numba.jit(nopython=True, parallel=True, cache=True)
def _structured_qn_numba(
    Lx: float, Ly: float, nx: int, ny: int, order: int, parallel: bool
):
    """
    Generates raw data for a structured Qn quadrilateral mesh using Numba.
    """
    if order < 1:
        # Numba doesn't support raising ValueError with strings, so error is minimal.
        raise ValueError("Polynomial order must be a positive integer.")

    # --- 1. Generate Node Coordinates ---
    num_global_nodes_x = order * nx + 1
    num_global_nodes_y = order * ny + 1
    num_total_nodes = num_global_nodes_x * num_global_nodes_y
    nodes_coords = np.zeros((num_total_nodes, 2), dtype=np.float64)
    
    x_coords = np.linspace(0, Lx, num_global_nodes_x)
    y_coords = np.linspace(0, Ly, num_global_nodes_y)

    # Numba can parallelize this loop efficiently
    for j_glob in numba.prange(num_global_nodes_y) if parallel else range(num_global_nodes_y):
        for i_glob in range(num_global_nodes_x):
            node_id = j_glob * num_global_nodes_x + i_glob
            nodes_coords[node_id, 0] = x_coords[i_glob]
            nodes_coords[node_id, 1] = y_coords[j_glob]

    # --- 2. Generate Element and Edge Connectivity ---
    num_elements = nx * ny
    nodes_per_edge_1d = order + 1
    elements = np.empty((num_elements, nodes_per_edge_1d**2), dtype=np.int64)
    elements_corner_nodes = np.empty((num_elements, 4), dtype=np.int64)
    # A set() is not supported in nopython mode, so we collect all edges
    # and find the unique ones later using numpy.
    all_edges = np.empty((num_elements * 4, 2), dtype=np.int64)

    # This loop is also safe to parallelize
    for el_idx in numba.prange(num_elements) if parallel else range(num_elements):
        el_j = el_idx // nx
        el_i = el_idx % nx
        
        start_ix, start_iy = order * el_i, order * el_j
        
        # A. Build Full Element Connectivity
        local_node_idx = 0
        for local_ny in range(nodes_per_edge_1d):
            for local_nx in range(nodes_per_edge_1d):
                gid = (start_iy + local_ny) * num_global_nodes_x + (start_ix + local_nx)
                elements[el_idx, local_node_idx] = gid
                local_node_idx += 1
        
        # B. Identify and store corners for this element (CCW order)
        bl_gid = start_iy * num_global_nodes_x + start_ix
        br_gid = start_iy * num_global_nodes_x + (start_ix + order)
        tl_gid = (start_iy + order) * num_global_nodes_x + start_ix
        tr_gid = (start_iy + order) * num_global_nodes_x + (start_ix + order)
        corners = np.array([bl_gid, br_gid, tr_gid, tl_gid])
        elements_corner_nodes[el_idx, :] = corners
        
        # C. Store all geometric edges for later unique filtering
        edge_offset = el_idx * 4
        all_edges[edge_offset, :] = np.sort(np.array([corners[0], corners[1]]))
        all_edges[edge_offset + 1, :] = np.sort(np.array([corners[1], corners[2]]))
        all_edges[edge_offset + 2, :] = np.sort(np.array([corners[2], corners[3]]))
        all_edges[edge_offset + 3, :] = np.sort(np.array([corners[3], corners[0]]))

    # --- 3. Filter for Unique Edges (serial operation) ---
    # This is a standard method to find unique rows in a NumPy array.
    if num_elements > 0:
        edges = unique_rows_int64(all_edges)
    else:
        edges = np.empty((0, 2), dtype=np.int64)

    return nodes_coords, elements, edges, elements_corner_nodes

def _structured_qn(Lx: float, Ly: float, nx: int, ny: int, order: int, 
                   offset: Optional[Tuple[float, float]] = None) -> Tuple[List[Node], np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates raw data for a structured quadrilateral mesh of Qn elements.

    Returns:
        tuple: (nodes, elements, edges, elements_corner_nodes)
            nodes (List[Node]): A list of Node objects with tags.
            elements (np.ndarray): Full element connectivity using Node IDs.
            edges (np.ndarray): Edge connectivity (unique pairs of corner node IDs).
            elements_corner_nodes (np.ndarray): Corner node connectivity for each element.
    """
    if not isinstance(order, int) or order < 1:
        raise ValueError("Polynomial order must be a positive integer.")

    # --- 1. Generate all Node objects with tags ---
    nodes_per_edge_1d = order + 1
    num_global_nodes_x = order * nx + 1
    num_global_nodes_y = order * ny + 1
    x_coords = np.linspace(0, Lx, num_global_nodes_x)
    y_coords = np.linspace(0, Ly, num_global_nodes_y)

    nodes: List[Node] = []
    for j_glob in range(num_global_nodes_y):
        for i_glob in range(num_global_nodes_x):
            x, y = x_coords[i_glob], y_coords[j_glob]
            tags = []
            if np.isclose(x, 0): tags.append("boundary_left")
            if np.isclose(x, Lx): tags.append("boundary_right")
            if np.isclose(y, 0): tags.append("boundary_bottom")
            if np.isclose(y, Ly): tags.append("boundary_top")
            is_corner = (i_glob % order == 0) and (j_glob % order == 0)
            is_edge_node = not is_corner and (i_glob % order == 0 or j_glob % order == 0)
            if is_corner: tags.append("corner")
            elif is_edge_node: tags.append("edge")
            else: tags.append("interior")
            nodes.append(Node(id=len(nodes), x=x, y=y, tag=",".join(tags)))

    # --- 2. Generate Element, Edge, and Corner Connectivity ---
    num_elements = nx * ny
    elements = np.empty((num_elements, nodes_per_edge_1d**2), dtype=int)
    elements_corner_nodes = np.empty((num_elements, 4), dtype=int)
    unique_edges = set()
    
    get_node_id = lambda ix, iy: iy * num_global_nodes_x + ix

    for el_j in range(ny):
        for el_i in range(nx):
            eid = el_j * nx + el_i
            start_ix, start_iy = order * el_i, order * el_j
            
            # A. Build Full Element Connectivity
            local_node_idx = 0
            for local_ny in range(nodes_per_edge_1d):
                for local_nx in range(nodes_per_edge_1d):
                    gid = get_node_id(start_ix + local_nx, start_iy + local_ny)
                    elements[eid, local_node_idx] = gid
                    local_node_idx += 1
            
            # B. Identify Corners for this element
            bl_gid = get_node_id(start_ix, start_iy)
            br_gid = get_node_id(start_ix + order, start_iy)
            tr_gid = get_node_id(start_ix + order, start_iy + order)
            tl_gid = get_node_id(start_ix, start_iy + order)
            
            # C. Store the ordered list of corners for this element
            # This order is canonical (CCW) and used for defining local edges.
            corners = [bl_gid, br_gid, tr_gid, tl_gid]
            elements_corner_nodes[eid] = corners
            
            # D. Add geometric edges to the unique set
            unique_edges.add(tuple(sorted((corners[0], corners[1])))) # Bottom
            unique_edges.add(tuple(sorted((corners[1], corners[2])))) # Right
            unique_edges.add(tuple(sorted((corners[2], corners[3])))) # Top
            unique_edges.add(tuple(sorted((corners[3], corners[0])))) # Left

    edges = np.array(list(unique_edges), dtype=int)
    if offset is not None:
        _translate_nodes(nodes, offset)
    return nodes, elements, edges, elements_corner_nodes


def structured_triangles(Lx: float, Ly: float, *, nx_quads: int, ny_quads: int, poly_order: int, offset: Optional[Tuple[float, float]] = None):
    """
    Main wrapper for generating structured triangular meshes.
    Returns raw data: node objects, element connectivity, edge connectivity,
    and corner node connectivity.
    """
    return _structured_pk(Lx, Ly, nx_quads, ny_quads, poly_order, offset=offset)


def _structured_pk(Lx: float, Ly: float, nx_base_quads: int, ny_base_quads: int, order_k: int, offset: Optional[Tuple[float, float]] = None):
    """
    Generates raw data for a structured triangular mesh of Pk elements.
    """
    if not isinstance(order_k, int) or order_k < 1:
        raise ValueError("Polynomial order must be a positive integer.")

    # --- 1. Generate all Node objects with tags ---
    num_fine_nodes_x = order_k * nx_base_quads + 1
    num_fine_nodes_y = order_k * ny_base_quads + 1
    x_fine = np.linspace(0, Lx, num_fine_nodes_x)
    y_fine = np.linspace(0, Ly, num_fine_nodes_y)

    nodes: List[Node] = []
    for j_fine in range(num_fine_nodes_y):
        for i_fine in range(num_fine_nodes_x):
            x, y = x_fine[i_fine], y_fine[j_fine]
            tags = []
            if np.isclose(x, 0): tags.append("boundary_left")
            if np.isclose(x, Lx): tags.append("boundary_right")
            if np.isclose(y, 0): tags.append("boundary_bottom")
            if np.isclose(y, Ly): tags.append("boundary_top")
            is_on_coarse_x = i_fine % order_k == 0
            is_on_coarse_y = j_fine % order_k == 0
            if is_on_coarse_x and is_on_coarse_y: tags.append("corner")
            elif is_on_coarse_x or is_on_coarse_y: tags.append("edge")
            else: tags.append("interior")
            nodes.append(Node(id=len(nodes), x=x, y=y, tag=",".join(tags)))

    # --- 2. Generate Element, Edge, and Corner Connectivity ---
    num_nodes_per_elem = (order_k + 1) * (order_k + 2) // 2
    num_elems = 2 * nx_base_quads * ny_base_quads
    elements = np.empty((num_elems, num_nodes_per_elem), dtype=int)
    elements_corner_nodes = np.empty((num_elems, 3), dtype=int)
    unique_edges = set()
    elem_idx_counter = 0

    get_node_id = lambda ix, iy: iy * num_fine_nodes_x + ix

    for e_iy in range(ny_base_quads):
        for e_ix in range(nx_base_quads):
            v00_idx = (order_k * e_ix, order_k * e_iy)
            v10_idx = (order_k * (e_ix + 1), order_k * e_iy)
            v01_idx = (order_k * e_ix, order_k * (e_iy + 1))
            v11_idx = (order_k * (e_ix + 1), order_k * (e_iy + 1))
            
            base_tri_vertex_indices = [
                (v00_idx, v10_idx, v11_idx),
                (v00_idx, v11_idx, v01_idx)
            ]

            for V0_idx, V1_idx, V2_idx in base_tri_vertex_indices:
                # A. Build Full Element Connectivity
                local_node_idx = 0
                elem_nodes = np.empty(num_nodes_per_elem, dtype=int)
                for j_level in range(order_k + 1): 
                    for i_level in range(order_k + 1 - j_level):
                        node_ix = V0_idx[0] + i_level * ((V1_idx[0] - V0_idx[0]) // order_k) + j_level * ((V2_idx[0] - V0_idx[0]) // order_k)
                        node_iy = V0_idx[1] + i_level * ((V1_idx[1] - V0_idx[1]) // order_k) + j_level * ((V2_idx[1] - V0_idx[1]) // order_k)
                        elem_nodes[local_node_idx] = get_node_id(node_ix, node_iy)
                        local_node_idx += 1
                elements[elem_idx_counter] = elem_nodes
                
                # B. Identify Corners and store them
                v0_gid = get_node_id(*V0_idx)
                v1_gid = get_node_id(*V1_idx)
                v2_gid = get_node_id(*V2_idx)
                elements_corner_nodes[elem_idx_counter] = [v0_gid, v1_gid, v2_gid]
                
                # C. Add geometric edges to the unique set
                unique_edges.add(tuple(sorted((v0_gid, v1_gid))))
                unique_edges.add(tuple(sorted((v1_gid, v2_gid))))
                unique_edges.add(tuple(sorted((v2_gid, v0_gid))))

                elem_idx_counter += 1
                
    edges = np.array(list(unique_edges), dtype=int)
    if offset is not None:
        _translate_nodes(nodes, offset)
    return nodes, elements, edges, elements_corner_nodes






def visualize_test_structured_quad():
    print("Running Mesh Generation Test Cases...")

    # Test Case 1: Single Q1 element
    print("\n--- Test Case 1: Single Q1 Element (1x1 grid) ---")
    Lx1, Ly1, nx1, ny1, order1 = 1.0, 1.0, 1, 1, 1
    nodes1, elements1, edges1, elements_corner_nodes1 = structured_quad(Lx1, Ly1, nx=nx1, ny=ny1, poly_order=order1)
    nodes1 = np.array([[node.x, node.y] for node in nodes1])  # Convert Node objects to numpy array
    print(f"Points shape: {nodes1.shape}")
    print(f"First 5 points:\n{nodes1[:5]}")
    print(f"Quads shape: {elements1.shape}")
    print(f"Quads connectivity:\n{elements1}")
    print(f"Corner connectivity:\n{elements_corner_nodes1}")
    visualize_mesh_node_order(nodes1, elements1, order1, title="Test Case 1: Single Q1 Element (1x1)")

    # Test Case 2: 2x1 Q1 elements
    print("\n--- Test Case 2: 2x1 Q1 Elements ---")
    Lx2, Ly2, nx2, ny2, order2 = 2.0, 1.0, 2, 1, 1
    pts2, quads2 = structured_quad(Lx2, Ly2, nx=nx2, ny=ny2, poly_order=order2)
    print(f"Points shape: {pts2.shape}")
    # print(f"Points:\n{pts2}") # Can be long
    print(f"Quads shape: {quads2.shape}")
    print(f"Quads connectivity:\n{quads2}")
    visualize_mesh_node_order(pts2, quads2, order2, title="Test Case 2: 2x1 Q1 Elements")

    # Test Case 3: Single Q2 element
    print("\n--- Test Case 3: Single Q2 Element (1x1 grid) ---")
    Lx3, Ly3, nx3, ny3, order3 = 1.0, 1.0, 1, 1, 2
    pts3, quads3 = structured_quad(Lx3, Ly3, nx=nx3, ny=ny3, poly_order=order3)
    print(f"Points shape: {pts3.shape}")
    # print(f"Points:\n{pts3}")
    print(f"Quads shape: {quads3.shape}")
    print(f"Quads connectivity:\n{quads3}")
    visualize_mesh_node_order(pts3, quads3, order3, title="Test Case 3: Single Q2 Element (1x1)")
    
    # Test Case 4: 2x2 Q2 elements
    print("\n--- Test Case 4: 2x2 Q2 Elements ---")
    Lx4, Ly4, nx4, ny4, order4 = 2.0, 2.0, 2, 2, 2
    pts4, quads4 = structured_quad(Lx4, Ly4, nx=nx4, ny=ny4, poly_order=order4)
    print(f"Points shape: {pts4.shape}")
    print(f"Quads shape: {quads4.shape}")
    # print(f"Quads connectivity:\n{quads4}") # Can be long
    visualize_mesh_node_order(pts4, quads4, order4, title="Test Case 4: 2x2 Q2 Elements")

    # Test Case 5: Single Q3 element (demonstrates generality)
    print("\n--- Test Case 5: Single Q3 Element (1x1 grid) ---")
    Lx5, Ly5, nx5, ny5, order5 = 1.0, 1.5, 1, 1, 3 # Using Ly=1.5 for non-square
    pts5, quads5 = structured_quad(Lx5, Ly5, nx=nx5, ny=ny5, poly_order=order5)
    print(f"Points shape: {pts5.shape}")
    print(f"Quads shape: {quads5.shape}")
    print(f"Quads connectivity (first element if many, or all if one):\n{quads5[0] if quads5.shape[0] > 0 else quads5}")
    visualize_mesh_node_order(pts5, quads5, order5, title="Test Case 5: Single Q3 Element (1x1)")

    print("\nAll test cases executed. Check the plots for visual verification of node ordering.")

def visualize_test_structured_tri_and_quad():
    print("Running Mesh Generation Test Cases...")

    # === Quadrilateral Tests ===
    print("\n--- Test Case Q1: Single Q1 Element (1x1 grid) ---")
    nodes, elements, edges, corner_connectivity = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    ptsQ1 = np.array([[node.x,node.y] for node in nodes])
    visualize_mesh_node_order(ptsQ1, elements, 1, 'quad', title="Test Q1: Single Q1 Element")

    print("\n--- Test Case Q2: 2x1 Q2 Elements ---")
    nodes, elements, edges, corner_connectivity = structured_quad(2.0, 1.0, nx=2, ny=1, poly_order=2)
    ptsQ2 = np.array([[node.x,node.y] for node in nodes])
    quadsQ2 = elements
    visualize_mesh_node_order(ptsQ2, quadsQ2, 2, 'quad', title="Test Q2: 2x1 Q2 Elements")
    
    print("\n--- Test Case Q0: 2x2 Q0 Elements ---") # Test order 0 for Quads
    # nodes, elements, edges, corner_connectivity = structured_quad(2.0, 2.0, nx=2, ny=2, poly_order=0)
    # ptsQ0 = np.array([[node.x,node.y] for node in nodes])
    # quadsQ0 = elements
    # visualize_mesh_node_order(ptsQ0, quadsQ0, 0, 'quad', title="Test Q0: 2x2 Q0 (Point) Elements")


    # === Triangular Tests ===
    print("\n--- Test Case T1: 1x1 Base Quads, P1 Elements (2 triangles) ---")
    nodes, elements, edges, elements_corner_nodes = structured_triangles(1.0, 1.0, nx_quads=1, ny_quads=1, poly_order=1)
    ptsT1 = np.array([[node.x, node.y] for node in nodes])
    trisT1 = elements
    visualize_mesh_node_order(ptsT1, trisT1, 1, 'triangle', title="Test T1: P1 Triangles (from 1x1 Quads)")

    print("\n--- Test Case T2: 1x1 Base Quads, P2 Elements (2 triangles) ---")
    nodes, elements, edges, elements_corner_nodes = structured_triangles(1.0, 1.0, nx_quads=1, ny_quads=1, poly_order=2)
    ptsT2 = np.array([[node.x, node.y] for node in nodes])
    trisT2 = elements
    visualize_mesh_node_order(ptsT2, trisT2, 2, 'triangle', title="Test T2: P2 Triangles (from 1x1 Quads)")

    print("\n--- Test Case T3: 2x1 Base Quads, P1 Elements (4 triangles) ---")
    nodes, elements, edges, elements_corner_nodes = structured_triangles(2.0, 1.0, nx_quads=2, ny_quads=1, poly_order=1)
    ptsT3 = np.array([[node.x, node.y] for node in nodes])
    trisT3 = elements
    visualize_mesh_node_order(ptsT3, trisT3, 1, 'triangle', title="Test T3: P1 Triangles (from 2x1 Quads)")

    # print("\n--- Test Case T0: 2x2 Base Quads, P0 Elements (8 triangles) ---") # Test order 0 for Triangles
    # nodes, elements, edges, elements_corner_nodes = structured_triangles(2.0, 2.0, nx_quads=2, ny_quads=2, poly_order=0)
    # ptsT0 = np.array([[node.x, node.y] for node in nodes])
    # trisT0 = elements
    # visualize_mesh_node_order(ptsT0, trisT0, 0, 'triangle', title="Test T0: P0 (Point) Elements on Triangles")

    print("\nAll test cases executed. Check the plots for visual verification.")


# --- Test Cases ---
if __name__ == "__main__":
    visualize_test_structured_tri_and_quad()