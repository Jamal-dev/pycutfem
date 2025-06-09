"""pycutfem.utils.meshgen
Mesh generators for quick tests.
"""
import numpy as np
from scipy.spatial import Delaunay
from pycutfem.io.visualization import visualize_mesh_node_order
from pycutfem.core import Node
from typing import List

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

def structured_quad(Lx: float, Ly: float, *, nx: int, ny: int, poly_order: int):
    """
    Main wrapper for generating structured quadrilateral meshes.
    Returns raw data: node objects, element connectivity, edge connectivity,
    and corner node connectivity for each element.
    """
    return _structured_qn(Lx, Ly, nx, ny, poly_order)

def _structured_qn(Lx: float, Ly: float, nx: int, ny: int, order: int):
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
    return nodes, elements, edges, elements_corner_nodes


def structured_triangles(Lx: float, Ly: float, *, nx_quads: int, ny_quads: int, poly_order: int):
    """
    Main wrapper for generating structured triangular meshes.
    Returns raw data: node objects, element connectivity, edge connectivity,
    and corner node connectivity.
    """
    return _structured_pk(Lx, Ly, nx_quads, ny_quads, poly_order)


def _structured_pk(Lx: float, Ly: float, nx_base_quads: int, ny_base_quads: int, order_k: int):
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
    return nodes, elements, edges, elements_corner_nodes

def visualize_test_structured_quad():
    print("Running Mesh Generation Test Cases...")

    # Test Case 1: Single Q1 element
    print("\n--- Test Case 1: Single Q1 Element (1x1 grid) ---")
    Lx1, Ly1, nx1, ny1, order1 = 1.0, 1.0, 1, 1, 1
    pts1, quads1 = structured_quad(Lx1, Ly1, nx=nx1, ny=ny1, poly_order=order1)
    print(f"Points shape: {pts1.shape}")
    print(f"First 5 points:\n{pts1[:5]}")
    print(f"Quads shape: {quads1.shape}")
    print(f"Quads connectivity:\n{quads1}")
    visualize_mesh_node_order(pts1, quads1, order1, title="Test Case 1: Single Q1 Element (1x1)")

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
    ptsQ1, quadsQ1 = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    visualize_mesh_node_order(ptsQ1, quadsQ1, 1, 'quad', title="Test Q1: Single Q1 Element")

    print("\n--- Test Case Q2: 2x1 Q2 Elements ---")
    ptsQ2, quadsQ2 = structured_quad(2.0, 1.0, nx=2, ny=1, poly_order=2)
    visualize_mesh_node_order(ptsQ2, quadsQ2, 2, 'quad', title="Test Q2: 2x1 Q2 Elements")
    
    print("\n--- Test Case Q0: 2x2 Q0 Elements ---") # Test order 0 for Quads
    ptsQ0, quadsQ0 = structured_quad(2.0, 2.0, nx=2, ny=2, poly_order=0)
    visualize_mesh_node_order(ptsQ0, quadsQ0, 0, 'quad', title="Test Q0: 2x2 Q0 (Point) Elements")


    # === Triangular Tests ===
    print("\n--- Test Case T1: 1x1 Base Quads, P1 Elements (2 triangles) ---")
    ptsT1, trisT1 = structured_triangles(1.0, 1.0, nx_quads=1, ny_quads=1, poly_order=1)
    visualize_mesh_node_order(ptsT1, trisT1, 1, 'triangle', title="Test T1: P1 Triangles (from 1x1 Quads)")

    print("\n--- Test Case T2: 1x1 Base Quads, P2 Elements (2 triangles) ---")
    ptsT2, trisT2 = structured_triangles(1.0, 1.0, nx_quads=1, ny_quads=1, poly_order=2)
    visualize_mesh_node_order(ptsT2, trisT2, 2, 'triangle', title="Test T2: P2 Triangles (from 1x1 Quads)")

    print("\n--- Test Case T3: 2x1 Base Quads, P1 Elements (4 triangles) ---")
    ptsT3, trisT3 = structured_triangles(2.0, 1.0, nx_quads=2, ny_quads=1, poly_order=1)
    visualize_mesh_node_order(ptsT3, trisT3, 1, 'triangle', title="Test T3: P1 Triangles (from 2x1 Quads)")

    print("\n--- Test Case T0: 2x2 Base Quads, P0 Elements (8 triangles) ---") # Test order 0 for Triangles
    ptsT0, trisT0 = structured_triangles(2.0, 2.0, nx_quads=2, ny_quads=2, poly_order=0)
    visualize_mesh_node_order(ptsT0, trisT0, 0, 'triangle', title="Test T0: P0 (Point) Elements on Triangles")

    print("\nAll test cases executed. Check the plots for visual verification.")


# --- Test Cases ---
if __name__ == "__main__":
    visualize_test_structured_tri_and_quad()