"""pycutfem.utils.meshgen
Mesh generators for quick tests.
"""
import numpy as np
from scipy.spatial import Delaunay
from pycutfem.io.visualization import visualize_mesh_node_order

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

def _structured_pk(Lx, Ly, nx_base_quads, ny_base_quads, order_k):
    if not isinstance(order_k, int) or order_k < 0: # Allow order 0 for point elements
        raise ValueError("Order k must be a non-negative integer.")

    num_fine_nodes_x = order_k * nx_base_quads + 1 if order_k > 0 else nx_base_quads + 1
    num_fine_nodes_y = order_k * ny_base_quads + 1 if order_k > 0 else ny_base_quads + 1

    if nx_base_quads == 0 or ny_base_quads == 0: # Handle cases with no base quads
        num_fine_nodes_x = nx_base_quads + 1
        num_fine_nodes_y = ny_base_quads + 1


    x_fine = np.linspace(0, Lx, num_fine_nodes_x)
    y_fine = np.linspace(0, Ly, num_fine_nodes_y)

    if num_fine_nodes_x == 0 or num_fine_nodes_y == 0:
         pts = np.array([[0.0,0.0]]) if Lx==0 and Ly==0 and nx_base_quads==1 and ny_base_quads==1 and order_k==0 else np.empty((0,2))
    else:
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
        pts = np.column_stack([X_fine.ravel(), Y_fine.ravel()])


    if order_k == 0: # P0 element
        num_nodes_per_pk_element = 1
    else:
        num_nodes_per_pk_element = (order_k + 1) * (order_k + 2) // 2
    
    num_pk_elements = 2 * nx_base_quads * ny_base_quads
    if nx_base_quads == 0 or ny_base_quads == 0 : # If no base quads, no Pk elements.
         num_pk_elements = 0
    if order_k == 0 and nx_base_quads > 0 and ny_base_quads > 0: # P0 case, one node per base quad, but we make 2 "elements" for consistency
        # This interpretation of P0 on a quad split is a bit unusual.
        # Typically P0 implies one value per original cell. Let's assume each "triangle" gets a P0 node.
        # The P0 node can be the centroid of the base triangle or just its first vertex.
        # For simplicity, let's assign it to the first vertex of each conceptual base triangle.
        pass


    if num_pk_elements == 0:
        return pts, np.empty((0, num_nodes_per_pk_element), dtype=int)
        
    pk_elements_connectivity = np.empty((num_pk_elements, num_nodes_per_pk_element), dtype=int)
    elem_idx_counter = 0

    def get_global_fine_node_id(ix_fine, iy_fine):
        if not (np.isclose(ix_fine, round(ix_fine)) and np.isclose(iy_fine, round(iy_fine))):
             raise ValueError(f"Pk: Computed fine grid indices ({ix_fine}, {iy_fine}) are not close to integers.")
        ix_fine_int, iy_fine_int = int(round(ix_fine)), int(round(iy_fine))
        if not (0 <= ix_fine_int < num_fine_nodes_x and 0 <= iy_fine_int < num_fine_nodes_y):
            raise ValueError(f"Pk: Fine grid indices ({ix_fine_int}, {iy_fine_int}) out of bounds ({num_fine_nodes_x}, {num_fine_nodes_y}). Original: ({ix_fine}, {iy_fine})")
        return iy_fine_int * num_fine_nodes_x + ix_fine_int

    for e_iy_base in range(ny_base_quads):
        for e_ix_base in range(nx_base_quads):
            v00_fgidx_raw_x = order_k * e_ix_base if order_k > 0 else e_ix_base
            v00_fgidx_raw_y = order_k * e_iy_base if order_k > 0 else e_iy_base
            
            v10_fgidx_raw_x = order_k * (e_ix_base + 1) if order_k > 0 else (e_ix_base + 1)
            v10_fgidx_raw_y = order_k * e_iy_base if order_k > 0 else e_iy_base
            
            v01_fgidx_raw_x = order_k * e_ix_base if order_k > 0 else e_ix_base
            v01_fgidx_raw_y = order_k * (e_iy_base + 1) if order_k > 0 else (e_iy_base + 1)

            v11_fgidx_raw_x = order_k * (e_ix_base + 1) if order_k > 0 else (e_ix_base + 1)
            v11_fgidx_raw_y = order_k * (e_iy_base + 1) if order_k > 0 else (e_iy_base + 1)

            v00_fgidx = (v00_fgidx_raw_x, v00_fgidx_raw_y)
            v10_fgidx = (v10_fgidx_raw_x, v10_fgidx_raw_y)
            v01_fgidx = (v01_fgidx_raw_x, v01_fgidx_raw_y)
            v11_fgidx = (v11_fgidx_raw_x, v11_fgidx_raw_y)
            
            base_triangles_vertices_fine_indices = [
                (v00_fgidx, v10_fgidx, v11_fgidx),
                (v00_fgidx, v11_fgidx, v01_fgidx)
            ]

            for V0, V1, V2 in base_triangles_vertices_fine_indices:
                if order_k == 0: # P0 element, one node (e.g., first vertex of base triangle)
                    pk_elements_connectivity[elem_idx_counter, 0] = get_global_fine_node_id(V0[0], V0[1])
                else:
                    current_pk_node_indices = []
                    for j_level in range(order_k + 1): 
                        for i_level in range(order_k + 1 - j_level):
                            node_fine_ix = V0[0] + i_level * ((V1[0] - V0[0]) // order_k) + j_level * ((V2[0] - V0[0]) // order_k)
                            node_fine_iy = V0[1] + i_level * ((V1[1] - V0[1]) // order_k) + j_level * ((V2[1] - V0[1]) // order_k)
                            current_pk_node_indices.append(get_global_fine_node_id(node_fine_ix, node_fine_iy))
                    if len(current_pk_node_indices) != num_nodes_per_pk_element:
                         raise RuntimeError(f"Internal error: Pk node count mismatch.")
                    pk_elements_connectivity[elem_idx_counter] = current_pk_node_indices
                elem_idx_counter += 1
                
    return pts, pk_elements_connectivity




def _structured_qn(Lx, Ly, nx, ny, order):
    """
    Generates a structured quadrilateral mesh for Qn elements.

    Args:
        Lx (float): Length of the domain in the x-direction.
        Ly (float): Length of the domain in the y-direction.
        nx (int): Number of elements in the x-direction.
        ny (int): Number of elements in the y-direction.
        order (int): Order of the Lagrange polynomial (e.g., 1 for Q1, 2 for Q2).
                     This 'order' corresponds to 'n' in Qn.

    Returns:
        tuple: (pts, quads)
            pts (np.ndarray): Array of node coordinates, shape (num_total_nodes, 2).
            quads (np.ndarray): Array of element connectivities.
                                Shape is (num_elements, (order+1)**2).
                                Nodes are ordered lexicographically within each element,
                                consistent with tensor-product shape functions (like those
                                generated by the user's `quad_qn` function).
    """
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a positive integer.")

    # Number of nodes along one edge of a Qn element (e.g., order=1 -> 2 nodes)
    nodes_per_edge_1d = order + 1
    nodes_per_element = nodes_per_edge_1d * nodes_per_edge_1d

    # Total number of unique node lines in the global grid
    num_global_nodes_x = order * nx + 1
    num_global_nodes_y = order * ny + 1

    # Generate global node coordinates
    x_coords = np.linspace(0, Lx, num_global_nodes_x)
    y_coords = np.linspace(0, Ly, num_global_nodes_y)
    X_glob, Y_glob = np.meshgrid(x_coords, y_coords)
    # Node points are ordered such that Y changes slowest, then X.
    # (x0,y0), (x1,y0), ..., (x_N,y0), (x0,y1), (x1,y1), ...
    pts = np.column_stack([X_glob.ravel(), Y_glob.ravel()])

    num_elements = nx * ny
    # Pre-allocate quads array for efficiency
    quads = np.empty((num_elements, nodes_per_element), dtype=int)
    
    current_element_idx = 0

    # Helper function to get global node ID from global grid indices (ix_glob, iy_glob)
    # ix_glob ranges from 0 to num_global_nodes_x - 1
    # iy_glob ranges from 0 to num_global_nodes_y - 1
    def get_global_node_id(ix_glob, iy_glob):
        return iy_glob * num_global_nodes_x + ix_glob

    # Iterate over each element cell (identified by its bottom-left corner's element indices)
    for el_j in range(ny):  # Element index in y-direction
        for el_i in range(nx):  # Element index in x-direction
            # Determine the starting global grid indices for this element's
            # bottom-left-most node (the node with local indices (0,0) within the element).
            start_ix_global_grid = order * el_i
            start_iy_global_grid = order * el_j
            
            current_node_in_element_idx = 0
            # Iterate over the local nodes within the current element.
            # The order (local_ny first, then local_nx) ensures lexicographical ordering
            # (0,0), (1,0), ..., (order,0), (0,1), ..., (order,1), ...
            # which matches the typical tensor-product shape function ordering from `quad_qn`.
            for local_ny in range(nodes_per_edge_1d): # Corresponds to increasing eta node index
                for local_nx in range(nodes_per_edge_1d): # Corresponds to increasing xi node index
                    # Calculate global grid indices for the current local node
                    global_ix_grid = start_ix_global_grid + local_nx
                    global_iy_grid = start_iy_global_grid + local_ny
                    
                    node_id = get_global_node_id(global_ix_grid, global_iy_grid)
                    quads[current_element_idx, current_node_in_element_idx] = node_id
                    current_node_in_element_idx += 1
            
            current_element_idx += 1
            
    return pts, quads

def structured_quad(Lx, Ly, *, nx=4, ny=4, element_order=1):
    """
    Main function to generate a structured quadrilateral mesh.
    This function is now generalized for any order.
    """
    if not isinstance(element_order, int) or element_order < 0:
        # Delegate detailed error checking to _structured_qn, but catch common cases.
        raise ValueError("Order must be a positive integer (e.g., 1 for Q1, 2 for Q2).")
    
    pts, quads = _structured_qn(Lx, Ly, nx, ny, element_order)
    
    return pts, quads

def structured_triangles(Lx, Ly, *, nx_quads=4, ny_quads=4, order=1):
    """
    Main function to generate a structured triangular mesh of Pk elements.

    Args:
        Lx (float): Length of the domain in x.
        Ly (float): Length of the domain in y.
        nx_quads (int): Number of coarse quadrilaterals in x to be split.
                        Results in 2*nx_quads triangles in x-bands.
        ny_quads (int): Number of coarse quadrilaterals in y to be split.
        order (int): Polynomial order k of the Pk elements (e.g., 1 for P1).

    Returns:
        tuple: (pts, tris) as returned by _structured_pk.
    """
    # 'order' here means Pk order (k)
    pts, tris = _structured_pk(Lx, Ly, nx_quads, ny_quads, order)
    return pts, tris

def visualize_test_structured_quad():
    print("Running Mesh Generation Test Cases...")

    # Test Case 1: Single Q1 element
    print("\n--- Test Case 1: Single Q1 Element (1x1 grid) ---")
    Lx1, Ly1, nx1, ny1, order1 = 1.0, 1.0, 1, 1, 1
    pts1, quads1 = structured_quad(Lx1, Ly1, nx=nx1, ny=ny1, element_order=order1)
    print(f"Points shape: {pts1.shape}")
    print(f"First 5 points:\n{pts1[:5]}")
    print(f"Quads shape: {quads1.shape}")
    print(f"Quads connectivity:\n{quads1}")
    visualize_mesh_node_order(pts1, quads1, order1, title="Test Case 1: Single Q1 Element (1x1)")

    # Test Case 2: 2x1 Q1 elements
    print("\n--- Test Case 2: 2x1 Q1 Elements ---")
    Lx2, Ly2, nx2, ny2, order2 = 2.0, 1.0, 2, 1, 1
    pts2, quads2 = structured_quad(Lx2, Ly2, nx=nx2, ny=ny2, element_order=order2)
    print(f"Points shape: {pts2.shape}")
    # print(f"Points:\n{pts2}") # Can be long
    print(f"Quads shape: {quads2.shape}")
    print(f"Quads connectivity:\n{quads2}")
    visualize_mesh_node_order(pts2, quads2, order2, title="Test Case 2: 2x1 Q1 Elements")

    # Test Case 3: Single Q2 element
    print("\n--- Test Case 3: Single Q2 Element (1x1 grid) ---")
    Lx3, Ly3, nx3, ny3, order3 = 1.0, 1.0, 1, 1, 2
    pts3, quads3 = structured_quad(Lx3, Ly3, nx=nx3, ny=ny3, element_order=order3)
    print(f"Points shape: {pts3.shape}")
    # print(f"Points:\n{pts3}")
    print(f"Quads shape: {quads3.shape}")
    print(f"Quads connectivity:\n{quads3}")
    visualize_mesh_node_order(pts3, quads3, order3, title="Test Case 3: Single Q2 Element (1x1)")
    
    # Test Case 4: 2x2 Q2 elements
    print("\n--- Test Case 4: 2x2 Q2 Elements ---")
    Lx4, Ly4, nx4, ny4, order4 = 2.0, 2.0, 2, 2, 2
    pts4, quads4 = structured_quad(Lx4, Ly4, nx=nx4, ny=ny4, element_order=order4)
    print(f"Points shape: {pts4.shape}")
    print(f"Quads shape: {quads4.shape}")
    # print(f"Quads connectivity:\n{quads4}") # Can be long
    visualize_mesh_node_order(pts4, quads4, order4, title="Test Case 4: 2x2 Q2 Elements")

    # Test Case 5: Single Q3 element (demonstrates generality)
    print("\n--- Test Case 5: Single Q3 Element (1x1 grid) ---")
    Lx5, Ly5, nx5, ny5, order5 = 1.0, 1.5, 1, 1, 3 # Using Ly=1.5 for non-square
    pts5, quads5 = structured_quad(Lx5, Ly5, nx=nx5, ny=ny5, element_order=order5)
    print(f"Points shape: {pts5.shape}")
    print(f"Quads shape: {quads5.shape}")
    print(f"Quads connectivity (first element if many, or all if one):\n{quads5[0] if quads5.shape[0] > 0 else quads5}")
    visualize_mesh_node_order(pts5, quads5, order5, title="Test Case 5: Single Q3 Element (1x1)")

    print("\nAll test cases executed. Check the plots for visual verification of node ordering.")

def visualize_test_structured_tri_and_quad():
    print("Running Mesh Generation Test Cases...")

    # === Quadrilateral Tests ===
    print("\n--- Test Case Q1: Single Q1 Element (1x1 grid) ---")
    ptsQ1, quadsQ1 = structured_quad(1.0, 1.0, nx=1, ny=1, element_order=1)
    visualize_mesh_node_order(ptsQ1, quadsQ1, 1, 'quad', title="Test Q1: Single Q1 Element")

    print("\n--- Test Case Q2: 2x1 Q2 Elements ---")
    ptsQ2, quadsQ2 = structured_quad(2.0, 1.0, nx=2, ny=1, element_order=2)
    visualize_mesh_node_order(ptsQ2, quadsQ2, 2, 'quad', title="Test Q2: 2x1 Q2 Elements")
    
    print("\n--- Test Case Q0: 2x2 Q0 Elements ---") # Test order 0 for Quads
    ptsQ0, quadsQ0 = structured_quad(2.0, 2.0, nx=2, ny=2, element_order=0)
    visualize_mesh_node_order(ptsQ0, quadsQ0, 0, 'quad', title="Test Q0: 2x2 Q0 (Point) Elements")


    # === Triangular Tests ===
    print("\n--- Test Case T1: 1x1 Base Quads, P1 Elements (2 triangles) ---")
    ptsT1, trisT1 = structured_triangles(1.0, 1.0, nx_quads=1, ny_quads=1, order=1)
    visualize_mesh_node_order(ptsT1, trisT1, 1, 'triangle', title="Test T1: P1 Triangles (from 1x1 Quads)")

    print("\n--- Test Case T2: 1x1 Base Quads, P2 Elements (2 triangles) ---")
    ptsT2, trisT2 = structured_triangles(1.0, 1.0, nx_quads=1, ny_quads=1, order=2)
    visualize_mesh_node_order(ptsT2, trisT2, 2, 'triangle', title="Test T2: P2 Triangles (from 1x1 Quads)")

    print("\n--- Test Case T3: 2x1 Base Quads, P1 Elements (4 triangles) ---")
    ptsT3, trisT3 = structured_triangles(2.0, 1.0, nx_quads=2, ny_quads=1, order=1)
    visualize_mesh_node_order(ptsT3, trisT3, 1, 'triangle', title="Test T3: P1 Triangles (from 2x1 Quads)")

    print("\n--- Test Case T0: 2x2 Base Quads, P0 Elements (8 triangles) ---") # Test order 0 for Triangles
    ptsT0, trisT0 = structured_triangles(2.0, 2.0, nx_quads=2, ny_quads=2, order=0)
    visualize_mesh_node_order(ptsT0, trisT0, 0, 'triangle', title="Test T0: P0 (Point) Elements on Triangles")

    print("\nAll test cases executed. Check the plots for visual verification.")


# --- Test Cases ---
if __name__ == "__main__":
    visualize_test_structured_tri_and_quad()