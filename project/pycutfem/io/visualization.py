"""pycutfem.io.visualization"""
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
import matplotlib.pyplot as plt
import matplotlib.patches as patches



# --- Helper Color/Style Definitions (from your original code) ---
_ELEM_FILL = {
    "inside": (0.4, 0.6, 1.0, 0.7),
    "outside": (1.0, 0.8, 0.4, 0.7),
    "cut": (1.0, 0.55, 0.0, 0.7),
    "default": (0.9, 0.9, 0.9, 0.5)
}
_EDGE_COLOR = {"interface": "red", "ghost": "blue", "boundary": "dimgray"}

def _edge_col(tag):
    return _EDGE_COLOR.get(tag, 'black')

def _elem_fill(tag):
    # If tag is not in _ELEM_FILL, use the default color.
    # If tag is None or empty string, no fill will be used.
    if not tag:
        return 'none'
    return _ELEM_FILL.get(tag, _ELEM_FILL["default"])

# --- Refactored and Generalized plot_mesh Function ---
def plot_mesh(mesh, *, solution_on_nodes=None, level_set=None, plot_nodes=True, 
              plot_edges=True, elem_tags=True, edge_colors=True, 
              show=True, ax=None, resolution=200):
    """
    Plots a 2D mesh of triangular or quadrilateral elements of any order.

    Args:
        mesh (Mesh): The mesh object to plot. It is expected to have the
                     `_get_element_corner_global_indices` method.
        solution_on_nodes (np.ndarray, optional): A solution vector of nodal values
                                                  to display as a contour plot.
        level_set (object, optional): A level-set object with a callable `evaluate_on_nodes`
                                      or direct callable interface `level_set(points)`
                                      to plot the zero contour line.
        plot_nodes (bool, optional): If True, plots all nodes as points. Defaults to True.
        plot_edges (bool, optional): If True, plots the geometric edges. Defaults to True.
        elem_tags (bool, optional): If True, fills elements based on `mesh.elem_tag`.
                                    Defaults to True.
        edge_colors (bool, optional): If True, colors edges based on `mesh.edge_tag`.
                                      This maps to the `edge_tags` parameter internally.
        show (bool, optional): If True, calls plt.show() at the end. Defaults to True.
        ax (matplotlib.axes.Axes, optional): An existing axes object to plot on.
        resolution (int, optional): Grid resolution for plotting the level-set contour.
    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # --- Plot Filled Element Polygons (using CORNER nodes) ---
    if elem_tags and hasattr(mesh, "elem_tag"):
        polys_by_color = {}
        for eid, tag in enumerate(mesh.elem_tag):
            face_color = _elem_fill(tag)
            if face_color == 'none':
                # Draw a light outline for elements without a fill tag
                polys_by_color.setdefault('outline', []).append(
                    mesh.nodes_x_y_pos[mesh._get_element_corner_global_indices(eid)]
                )
                continue
            
            corner_gids = mesh._get_element_corner_global_indices(eid)
            corner_coords = mesh.nodes_x_y_pos[corner_gids]
            polys_by_color.setdefault(face_color, []).append(corner_coords)

        for color, polys_list in polys_by_color.items():
            if color == 'outline':
                poly_collection = PolyCollection(polys_list, facecolors='none', 
                                                 edgecolors=(0.1, 0.1, 0.1, 0.2), 
                                                 linewidths=0.5, zorder=1)
            else:
                poly_collection = PolyCollection(polys_list, facecolors=color, 
                                                 edgecolors='none', zorder=1, alpha=0.7)
            ax.add_collection(poly_collection)

    # --- Plot Edges ---
    if plot_edges and hasattr(mesh, 'edges'):
        edge_segments = [mesh.nodes_x_y_pos[list(edge.nodes)] for edge in mesh.edges]
        edge_colors_list = []
        for edge in mesh.edges:
            color = 'black'  # Default for internal, untagged edges
            # Check for boundary edges first
            if edge.right is None:
                color = 'dimgray' # Specific color for boundary edges
            # Tagged edges override other colors
            if edge_colors and hasattr(mesh, 'edge_tag') and mesh.edge_tag[edge.id]:
                color = _EDGE_COLOR.get(mesh.edge_tag[edge.id], color)
            edge_colors_list.append(color)
        
        line_collection = LineCollection(edge_segments, colors=edge_colors_list, 
                                         linewidths=0.9, zorder=2)
        ax.add_collection(line_collection)

    # --- Plot Nodes (all of them, differentiating corners) ---
    if plot_nodes:
        # Identify all unique corner node global indices
        corner_node_gids = set()
        if hasattr(mesh, '_get_element_corner_global_indices'):
            for eid in range(len(mesh.elements)):
                corners = mesh._get_element_corner_global_indices(eid)
                for gid in corners:
                    corner_node_gids.add(gid)
        
        all_node_gids = set(range(len(mesh.nodes_x_y_pos)))
        ho_node_gids = list(all_node_gids - corner_node_gids)
        corner_node_gids_list = list(corner_node_gids)

        # Plot higher-order nodes (e.g., edge midpoints) first, smaller and lighter
        if ho_node_gids:
            ax.plot(mesh.nodes_x_y_pos[ho_node_gids, 0], mesh.nodes_x_y_pos[ho_node_gids, 1], 'o', 
                    color='deepskyblue', markersize=2, zorder=3, label="Higher-Order Nodes", linestyle='None')

        # Plot corner nodes on top, more prominently
        if corner_node_gids_list:
            ax.plot(mesh.nodes_x_y_pos[corner_node_gids_list, 0], mesh.nodes_x_y_pos[corner_node_gids_list, 1], 'o',
                    color='navy', markersize=4, zorder=4, label="Corner Nodes", linestyle='None')
        
        # Add a legend if both types of nodes were plotted
        if ho_node_gids and corner_node_gids_list:
            ax.legend()


    # --- Plot Level-Set Zero Contour ---
    if level_set is not None:
        xmin, ymin = mesh.nodes_x_y_pos.min(axis=0)
        xmax, ymax = mesh.nodes_x_y_pos.max(axis=0)
        padding = (xmax - xmin) * 0.05
        
        gx, gy = np.meshgrid(np.linspace(xmin - padding, xmax + padding, resolution),
                            np.linspace(ymin - padding, ymax + padding, resolution))
        
        points_to_eval = np.column_stack([gx.ravel(), gy.ravel()])
        
        if hasattr(level_set, 'evaluate') and callable(level_set.evaluate):
            phi_vals = level_set.evaluate(points_to_eval)
        elif callable(level_set):
            phi_vals = np.apply_along_axis(level_set, 1, points_to_eval)
        else:
            raise TypeError("level_set must be a callable or have an 'evaluate' method.")

        phi_vals = phi_vals.reshape(gx.shape)
        ax.contour(gx, gy, phi_vals, levels=[0.0], colors='green', linewidths=1.5, zorder=5)

    # --- Plot Solution Contour ---
    if solution_on_nodes is not None:
        if len(solution_on_nodes) != len(mesh.nodes_x_y_pos):
            raise ValueError("Length of solution_on_nodes must match the number of mesh nodes.")
        
        contour = ax.tricontourf(mesh.nodes_x_y_pos[:, 0], mesh.nodes_x_y_pos[:, 1], solution_on_nodes,
                                 levels=14, cmap='viridis', zorder=0, alpha=0.8)
        plt.colorbar(contour, ax=ax, label="Solution Value")

    # --- Finalize Plot ---
    ax.set_aspect('equal', 'box')
    xmin, ymin = mesh.nodes_x_y_pos.min(axis=0)
    xmax, ymax = mesh.nodes_x_y_pos.max(axis=0)
    xpad = (xmax - xmin) * 0.05
    ypad = (ymax - ymin) * 0.05
    if xpad == 0: xpad = 0.1
    if ypad == 0: ypad = 0.1
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_title("Mesh Visualization")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")

    if show:
        plt.show()
        
    return ax

def plot_mesh_2(mesh, *, solution_on_nodes=None, level_set=None, plot_nodes=True, 
              plot_edges=True, elem_tags=True, edge_colors=True, 
              show=True, ax=None, resolution=200):
    """
    Plots a 2D mesh, correctly using the nodes_x_y_pos attribute for coordinates
    and adding a descriptive legend for tags.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    node_coords = mesh.nodes_x_y_pos
    legend_handles = []

    if elem_tags:
        polys_by_color = {}
        unique_elem_tags = set(elem.tag for elem in mesh.elements_list if elem.tag)
        
        for elem in mesh.elements_list:
            face_color = _ELEM_FILL.get(elem.tag, _ELEM_FILL["default"])
            corner_coords = node_coords[list(elem.corner_nodes)]
            polys_by_color.setdefault(face_color, []).append(corner_coords)

        for color, polys_list in polys_by_color.items():
            poly_collection = PolyCollection(polys_list, facecolors=color, 
                                             edgecolors='black', lw=0.5, zorder=1)
            ax.add_collection(poly_collection)
        
        for tag in sorted(list(unique_elem_tags)):
            if tag in _ELEM_FILL:
                legend_handles.append(patches.Patch(color=_ELEM_FILL[tag], label=f'Element: {tag}'))

    if plot_edges:
        edge_segments = [node_coords[list(edge.nodes)] for edge in mesh.edges_list]
        colors = [_EDGE_COLOR.get(edge.tag, 'black') if edge.right is not None else _EDGE_COLOR.get('boundary', 'black') for edge in mesh.edges_list]
        line_collection = LineCollection(edge_segments, colors=colors, linewidths=1.2, zorder=2)
        ax.add_collection(line_collection)
        
        unique_edge_tags = set(edge.tag for edge in mesh.edges_list if edge.tag)
        if any(e.right is None for e in mesh.edges_list):
            unique_edge_tags.add('boundary')

        for tag in sorted(list(unique_edge_tags)):
             if tag in _EDGE_COLOR:
                legend_handles.append(plt.Line2D([0], [0], color=_EDGE_COLOR[tag], lw=2, label=f'Edge: {tag}'))

    if level_set is not None:
        xmin, ymin = node_coords.min(axis=0); xmax, ymax = node_coords.max(axis=0)
        padding = (xmax - xmin) * 0.1
        gx, gy = np.meshgrid(np.linspace(xmin - padding, xmax + padding, resolution),
                             np.linspace(ymin - padding, ymax + padding, resolution))
        
        points_to_eval = np.c_[gx.ravel(), gy.ravel()]
        # FIX: Use apply_along_axis for robustness with non-vectorized level sets
        phi_vals = np.apply_along_axis(level_set.__call__, 1, points_to_eval).reshape(gx.shape)

        contour = ax.contour(gx, gy, phi_vals, levels=[0.0], colors='green', linewidths=2.5, zorder=5)
        # Check if any contours were actually drawn before adding to legend
        if len(contour.allsegs[0]) > 0:
            legend_handles.append(plt.Line2D([0], [0], color='green', lw=2, label='Level Set (φ=0)'))

    ax.set_aspect('equal', 'box')
    ax.set_title("Mesh Visualization with Domain Tags")
    if legend_handles:
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])

    if show:
        plt.show()
    return ax




def visualize_mesh_node_order(pts, element_connectivity, order, element_shape, title="Mesh with Node Order"):
    """
    Visualizes the mesh, showing elements and the order of nodes within each element.
    Works for both quadrilateral ('quad') and triangular ('triangle') elements.

    Args:
        pts (np.ndarray): Array of node coordinates, shape (num_total_nodes, 2).
        element_connectivity (np.ndarray): Array of element connectivities,
                                           shape (num_elements, nodes_per_element).
        order (int): The polynomial order of the elements (e.g., 1 for Q1/P1, 2 for Q2/P2).
        element_shape (str): Type of element, either 'quad' or 'triangle'.
        title (str): Title for the plot.
    """
    fig, ax = plt.subplots(figsize=(9, 9)) # Slightly larger for potentially more text
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)

    if element_shape == 'quad':
        expected_nodes_per_element = (order + 1)**2
        nodes_1d_edge = order + 1 # For quads, order is 'n' in (n+1) nodes per edge
    elif element_shape == 'triangle':
        expected_nodes_per_element = (order + 1) * (order + 2) // 2
    else:
        raise ValueError(f"Unsupported element_shape: {element_shape}. Choose 'quad' or 'triangle'.")

    if element_connectivity.shape[0] > 0:
        if element_connectivity.shape[1] != expected_nodes_per_element:
            raise ValueError(
                f"Mismatch in nodes per element for {element_shape} of order {order}. "
                f"Expected {expected_nodes_per_element}, but connectivity array has "
                f"{element_connectivity.shape[1]} nodes."
            )
    nodes_per_element = expected_nodes_per_element

    # Plot all global nodes
    if pts.shape[0] > 0:
        ax.plot(pts[:, 0], pts[:, 1], 'ko', markersize=4, zorder=2, alpha=0.7)

    for i, single_element_node_indices in enumerate(element_connectivity):
        boundary_node_global_indices = []
        if element_shape == 'quad':
            # Local indices for corners in lexicographical order for Qn:
            idx_bl = 0
            idx_br = order # order here is the highest index along one edge (0 to order)
            idx_tl = order * nodes_1d_edge # order * (order + 1)
            idx_tr = order * nodes_1d_edge + order

            if nodes_per_element == 1 and order == 0: # Special case Q0 (a point)
                 # This code isn't designed for Q0, but to prevent error:
                idx_br, idx_tl, idx_tr = idx_bl, idx_bl, idx_bl
            
            boundary_node_global_indices = np.array([
                single_element_node_indices[idx_bl],
                single_element_node_indices[idx_br],
                single_element_node_indices[idx_tr],
                single_element_node_indices[idx_tl]
            ])
        elif element_shape == 'triangle':
            # Local indices for the 3 primary vertices for Pk (V0, V1, V2)
            # V0 is local index 0
            # V1 is local index `order` (node at end of first edge group)
            # V2 is local index `nodes_per_element - 1` (last node in the sequence)
            idx_v0 = 0
            idx_v1 = order
            idx_v2 = nodes_per_element - 1
            
            if nodes_per_element == 1 and order == 0: # Special case P0 (a point)
                idx_v1, idx_v2 = idx_v0, idx_v0

            boundary_node_global_indices = np.array([
                single_element_node_indices[idx_v0],
                single_element_node_indices[idx_v1],
                single_element_node_indices[idx_v2]
            ])

        # Ensure all boundary indices are within the bounds of the pts array
        if np.all(boundary_node_global_indices < pts.shape[0]):
            polygon_coords = pts[boundary_node_global_indices]
            polygon = patches.Polygon(polygon_coords, closed=True, edgecolor='blue', facecolor='lightblue', alpha=0.5, zorder=1)
            ax.add_patch(polygon)
        else:
            print(f"Warning: Element {i} ({element_shape}) has boundary node indices out of bounds for pts array.")

        # Annotate all nodes within the element with their local index
        for local_idx in range(nodes_per_element):
            node_global_idx = single_element_node_indices[local_idx]
            if node_global_idx < pts.shape[0]:
                node_coord = pts[node_global_idx]
                ax.text(node_coord[0], node_coord[1], str(local_idx),
                        color='red', fontsize=7, ha='center', va='center', zorder=3,
                        bbox=dict(facecolor='white', alpha=0.4, pad=0.01, boxstyle='circle'))
            else:
                print(f"Warning: Node with local_idx {local_idx} in element {i} ({element_shape}) "
                      f"has global_idx {node_global_idx} out of bounds.")

    # Set plot limits with padding
    if pts.shape[0] > 0:
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        
        x_range = x_max - x_min if not np.isclose(x_max, x_min) else 1.0
        y_range = y_max - y_min if not np.isclose(y_max, y_min) else 1.0
        
        padding_x = 0.1 * x_range
        padding_y = 0.1 * y_range
        
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)
    else:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()