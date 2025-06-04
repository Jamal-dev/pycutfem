"""pycutfem.io.visualization"""
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
import matplotlib.pyplot as plt
import matplotlib.patches as patches

_ELEM_FILL = {
    "inside": (0.4,0.6,1.0,0.25),   # light blue
    "cut":    (1.0,0.55,0.0,0.45),  # orange
}
_EDGE_COLOR = {"interface":"red", "ghost":"blue"}

def _edge_col(tag): return _EDGE_COLOR.get(tag, 'black')
def _elem_fill(tag): return _ELEM_FILL.get(tag, 'none')

def plot_mesh(mesh, *, level_set=None, edge_colors=True,
              show=True, ax=None, resolution=200):
    if ax is None:
        fig, ax = plt.subplots()

    # fill elems
    polys_by_color = {}
    if getattr(mesh, "elem_tag", None) is not None:
        for eid, tag in enumerate(mesh.elem_tag):
            face = _elem_fill(tag)
            if face == 'none':
                continue
            polys_by_color.setdefault(face, []).append(mesh.nodes[mesh.elements[eid]])
    for col, polys in polys_by_color.items():
        pc = PolyCollection(polys, facecolors=col, edgecolors='none', zorder=1)
        ax.add_collection(pc)

    # edges
    segs = [mesh.nodes[list(e.nodes)] for e in mesh.edges]
    cols = [_edge_col(mesh.edge_tag[e.id]) if edge_colors else 'black' for e in mesh.edges]
    lc = LineCollection(segs, colors=cols, linewidths=0.7, zorder=2)
    ax.add_collection(lc)

    # level-set contour
    if level_set is not None:
        xmin,ymin = mesh.nodes.min(axis=0)
        xmax,ymax = mesh.nodes.max(axis=0)
        gx,gy = np.meshgrid(np.linspace(xmin,xmax,resolution),
                            np.linspace(ymin,ymax,resolution))
        phi = np.apply_along_axis(level_set,1,np.column_stack([gx.ravel(),gy.ravel()]))
        phi = phi.reshape(gx.shape)
        ax.contour(gx,gy,phi,levels=[0.0],colors='green',linewidths=1.0,zorder=3)

    ax.set_aspect('equal','box')
    ax.set_xlim(mesh.nodes[:,0].min(), mesh.nodes[:,0].max())
    ax.set_ylim(mesh.nodes[:,1].min(), mesh.nodes[:,1].max())
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