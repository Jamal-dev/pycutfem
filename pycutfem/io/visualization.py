"""pycutfem.io.visualization"""
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Union, List, Iterable, Optional, Sequence
from pycutfem.ufl.helpers_geom import (corner_tris, 
                                       clip_triangle_to_side, 
                                       fan_triangulate,
                                       clip_triangle_to_side_pn)
from pycutfem.core.sideconvention import SIDE



# --- Helper Color/Style Definitions (from your original code) ---
_ELEM_FILL = {
    "inside": (0.4, 0.6, 1.0, 0.7),
    "outside": (1.0, 0.8, 0.4, 0.7),
    "cut": (1.0, 0.55, 0.0, 0.7),
    "default": (0.9, 0.9, 0.9, 0.5)
}
_EDGE_COLOR = {
    "boundary": "black",
    "ghost": "blue",
    "ghost_pos": "cyan",
    "ghost_neg": "darkviolet",
    "ghost_both": "red",
    "interface": "green",
    "cut_boundary": "red",
    "default": "black",
}

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
              show=True, ax=None, resolution=200, plot_interface=True,
              edge_filter: Union[str, List[str]] = None):
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
            # Use default color if tag is missing to ensure it's drawn
            tag = elem.tag or "default"
            face_color = _ELEM_FILL.get(tag, _ELEM_FILL["default"])
            corner_coords = node_coords[list(elem.corner_nodes)]
            polys_by_color.setdefault(face_color, []).append(corner_coords)

        for color, polys_list in polys_by_color.items():
            poly_collection = PolyCollection(polys_list, facecolors=color,
                                             edgecolors='black', lw=0.5, zorder=1)
            ax.add_collection(poly_collection)

        for tag in sorted(list(unique_elem_tags)):
            if tag in _ELEM_FILL:
                legend_handles.append(patches.Patch(color=_ELEM_FILL[tag], label=f'Element: {tag}'))

    # --- MODIFIED SECTION ---
    if plot_edges:
        # 1. Determine which edges to draw
        edges_to_draw = mesh.edges_list  # Default to all edges

        if edge_filter:
            tags_to_show = [edge_filter] if isinstance(edge_filter, str) else edge_filter
            combined_mask = np.zeros(len(mesh.edges_list), dtype=bool)
            for tag in tags_to_show:
                try:
                    combined_mask |= mesh.edge_bitset(tag).mask
                except Exception:
                    print(f"Warning: Could not find or use BitSet for edge tag '{tag}'.")
            indices = np.flatnonzero(combined_mask)
            edges_to_draw = [mesh.edges_list[i] for i in indices]

        # 2. Build segments and colors using the (potentially filtered) list
        edge_segments = [node_coords[list(edge.nodes)] for edge in edges_to_draw]
        colors = [_EDGE_COLOR.get(edge.tag, 'black') if edge.right is not None else _EDGE_COLOR.get('boundary', 'black') for edge in edges_to_draw]
        line_collection = LineCollection(edge_segments, colors=colors, linewidths=1.2, zorder=2)
        ax.add_collection(line_collection)

        # 3. Build legend using only the tags from the drawn edges
        unique_edge_tags = set(edge.tag for edge in edges_to_draw if edge.tag)
        if any(e.right is None for e in edges_to_draw):
            unique_edge_tags.add('boundary')

        for tag in sorted(list(unique_edge_tags)):
             if tag in _EDGE_COLOR:
                legend_handles.append(plt.Line2D([0], [0], color=_EDGE_COLOR[tag], lw=2, label=f'Edge: {tag}'))
    # --- END MODIFIED SECTION ---

    if level_set is not None:
        # Fast FE path: use nodal φ and a mesh-based triangulation (no owner searches)
        if hasattr(level_set, "evaluate_on_nodes") and callable(getattr(level_set, "evaluate_on_nodes")):
            import matplotlib.tri as mtri

            # 1) φ at mesh nodes (fast for FE-backed level set)
            phi_nodes = np.asarray(level_set.evaluate_on_nodes(mesh), dtype=float)

            # If any NaNs slipped through, fill them via point-eval (rare)
            if np.isnan(phi_nodes).any():
                miss = np.isnan(phi_nodes)
                phi_nodes[miss] = np.array(
                    [float(level_set(node_coords[i])) for i in np.where(miss)[0]],
                    dtype=float
                )

            # 2) Build a triangulation of the mesh using corner nodes
            #    Quads are split along the (0–2) diagonal to match NGSolve.
            tris_idx = []
            for e in mesh.elements_list:
                cn = list(e.corner_nodes)
                if len(cn) == 3:                       # triangle
                    tris_idx.append(cn)
                elif len(cn) == 4:                     # quad → two tris via (0–2)
                    tris_idx.append([cn[0], cn[1], cn[2]])
                    tris_idx.append([cn[0], cn[2], cn[3]])
                else:
                    # Ignore polygons of other arity in this visualization
                    continue

            tri = mtri.Triangulation(node_coords[:, 0], node_coords[:, 1],
                                     np.asarray(tris_idx, dtype=int))

            # 3) φ=0 contour directly from nodal values (fast & accurate for P1/P2)
            contour = ax.tricontour(tri, phi_nodes, levels=[0.0],
                                    colors='green', linewidths=2.5, zorder=5)
            if contour.allsegs and len(contour.allsegs[0]) > 0:
                legend_handles.append(plt.Line2D([0], [0], color='green', lw=2,
                                                 label='Level Set (φ=0)'))
        else:
            # Fallback (analytic/unknown LS): sample on a grid (slower)
            xmin, ymin = node_coords.min(axis=0); xmax, ymax = node_coords.max(axis=0)
            padding = (xmax - xmin) * 0.1 if xmax > xmin else 0.1
            gx, gy = np.meshgrid(
                np.linspace(xmin - padding, xmax + padding, resolution),
                np.linspace(ymin - padding, ymax + padding, resolution)
            )

            pts = np.c_[gx.ravel(), gy.ravel()]
            # Keep your original per-point evaluation for maximum compatibility
            phi_vals = np.apply_along_axis(level_set.__call__, 1, pts).reshape(gx.shape)

            contour = ax.contour(gx, gy, phi_vals, levels=[0.0],
                                 colors='green', linewidths=2.5, zorder=5)
            if contour.allsegs and len(contour.allsegs[0]) > 0:
                legend_handles.append(plt.Line2D([0], [0], color='green', lw=2,
                                                 label='Level Set (φ=0)'))


    if plot_interface:
        all_pts, segments = [], []
        for elem in mesh.elements_list:
            if hasattr(elem, 'interface_pts') and elem.interface_pts:
                all_pts.extend(elem.interface_pts)
                if len(elem.interface_pts) == 2:
                    segments.append(elem.interface_pts)
        if segments:
            lc = LineCollection(segments, colors='magenta', linewidths=3.5, zorder=6, label='Interface Segment')
            ax.add_collection(lc)
            legend_handles.append(lc)
        if all_pts:
            all_pts_np = np.array(all_pts)
            ax.plot(all_pts_np[:, 0], all_pts_np[:, 1], 'o', color='cyan', markersize=9,
                    markeredgecolor='black', zorder=7)
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor='cyan', markeredgecolor='black',
                                             markersize=9, linestyle='None', label='Interface Point'))

    ax.set_aspect('equal', 'box')
    ax.set_title("Mesh Visualization with Domain Tags")
    if legend_handles:
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])

    if show:
        plt.show()
    return ax




def visualize_mesh_node_order(pts, element_connectivity, order, element_type, title="Mesh with Node Order"):
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

    if element_type == 'quad':
        expected_nodes_per_element = (order + 1)**2
        nodes_1d_edge = order + 1 # For quads, order is 'n' in (n+1) nodes per edge
    elif element_type == 'triangle':
        expected_nodes_per_element = (order + 1) * (order + 2) // 2
    else:
        raise ValueError(f"Unsupported element_shape: {element_type}. Choose 'quad' or 'triangle'.")

    if element_connectivity.shape[0] > 0:
        if element_connectivity.shape[1] != expected_nodes_per_element:
            raise ValueError(
                f"Mismatch in nodes per element for {element_type} of order {order}. "
                f"Expected {expected_nodes_per_element}, but connectivity array has "
                f"{element_connectivity.shape[1]} nodes."
            )
    nodes_per_element = expected_nodes_per_element

    # Plot all global nodes
    if pts.shape[0] > 0:
        ax.plot(pts[:, 0], pts[:, 1], 'ko', markersize=4, zorder=2, alpha=0.7)

    for i, single_element_node_indices in enumerate(element_connectivity):
        boundary_node_global_indices = []
        if element_type == 'quad':
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
        elif element_type == 'triangle':
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
            print(f"Warning: Element {i} ({element_type}) has boundary node indices out of bounds for pts array.")

        # Annotate all nodes within the element with their local index
        for local_idx in range(nodes_per_element):
            node_global_idx = single_element_node_indices[local_idx]
            if node_global_idx < pts.shape[0]:
                node_coord = pts[node_global_idx]
                ax.text(node_coord[0], node_coord[1], str(local_idx),
                        color='red', fontsize=7, ha='center', va='center', zorder=3,
                        bbox=dict(facecolor='white', alpha=0.4, pad=0.01, boxstyle='circle'))
            else:
                print(f"Warning: Node with local_idx {local_idx} in element {i} ({element_type}) "
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


def visualize_corner_labels(mesh, *, element_ids=None, max_elements=40, annotate_node_ids=True,
                             ax=None, title="Element corner labeling", corner_labels=None):
    """
    Plot selected elements and annotate each corner with labels (bl/br/tr/tl).

    Args:
        mesh: ``Mesh`` instance with ``corner_connectivity`` populated.
        element_ids: Optional iterable of element indices to annotate. Defaults to
            the first ``max_elements`` elements.
        max_elements: Limit when ``element_ids`` is None to avoid clutter.
        annotate_node_ids: If True, append the global node id next to the label.
        ax: Optional Matplotlib axes to draw on.
        title: Plot title.
        corner_labels: Optional list of four labels replacing ['bl','br','tr','tl'].
    """
    if not hasattr(mesh, "corner_connectivity"):
        raise AttributeError("Mesh is missing corner_connectivity; cannot annotate corners.")

    if element_ids is None:
        total = mesh.num_elements() if hasattr(mesh, "num_elements") else len(mesh.corner_connectivity)
        count = min(total, max_elements if max_elements is not None else total)
        element_ids = list(range(count))
    else:
        element_ids = list(element_ids)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_aspect("equal")
    ax.set_title(title)
    labels = corner_labels or ["bl", "br", "tr", "tl"]

    for eid in element_ids:
        corners = mesh.corner_connectivity[eid]
        coords = mesh.nodes_x_y_pos[list(corners)]
        polygon = patches.Polygon(coords, closed=True, edgecolor="black", facecolor="none", linewidth=1.0)
        ax.add_patch(polygon)

        centroid = coords.mean(axis=0)
        ax.text(centroid[0], centroid[1], f"e{eid}", color="tab:gray", fontsize=7,
                ha="center", va="center", bbox=dict(facecolor="white", alpha=0.5, pad=0.2))

        for label, gid, (x, y) in zip(labels, corners, coords):
            txt = f"{label}\n{gid}" if annotate_node_ids else label
            ax.text(x, y, txt, color="tab:red", fontsize=8, ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.7, pad=0.15))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle=":", alpha=0.4)
    return ax


def visualize_boundary_dofs(
    mesh,
    *,
    tags: Union[str, Iterable[str]],
    dof_handler=None,
    fields: Optional[Union[str, Sequence[str]]] = None,
    annotate_nodes: bool = False,
    annotate_dofs: bool = False,
    ax=None,
    title: str = "Boundary DOF overview",
    markersize: int = 40,
    show: bool = True,
):
    """
    Plot the mesh nodes (and optionally DOFs) that belong to specific edge tags.

    Parameters
    ----------
    mesh : Mesh
        Mesh instance whose edges carry boundary tags.
    tags : str | Iterable[str]
        Boundary tag(s) to visualize (e.g. ``\"cylinder\"``).
    dof_handler : DofHandler, optional
        When provided, DOF ids for the requested fields can be annotated.
    fields : str | Sequence[str], optional
        Field(s) to use when looking up DOF ids. Defaults to all fields on the
        handler when omitted.
    annotate_nodes : bool
        If True, prepend the node id to the annotation label.
    annotate_dofs : bool
        If True, append ``field:global_dof`` information for each requested field.
    ax : matplotlib Axes, optional
        Existing axes object; created automatically when omitted.
    title : str
        Plot title.
    markersize : int
        Marker size passed to ``ax.scatter``.
    """
    if isinstance(tags, str):
        tag_list = [tags]
    else:
        tag_list = list(tags)
    if not tag_list:
        raise ValueError("At least one boundary tag must be provided.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    nodes_by_tag: dict[str, List[int]] = {}
    for tag in tag_list:
        node_ids: set[int] = set()
        for edge in getattr(mesh, "edges_list", []):
            if getattr(edge, "tag", None) != tag:
                continue
            ids = getattr(edge, "all_nodes", ()) or edge.nodes
            node_ids.update(int(nid) for nid in ids)
        if node_ids:
            nodes_by_tag[tag] = sorted(node_ids)
        else:
            nodes_by_tag[tag] = []

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)
    if not colors:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    if dof_handler is not None:
        if fields is None:
            field_names = list(getattr(dof_handler, "field_names", []))
        elif isinstance(fields, str):
            field_names = [fields]
        else:
            field_names = list(fields)
    else:
        field_names = []

    coords = mesh.nodes_x_y_pos
    for idx, (tag, node_ids) in enumerate(nodes_by_tag.items()):
        if not node_ids:
            continue
        color = colors[idx % len(colors)]
        pts = coords[node_ids]
        ax.scatter(pts[:, 0], pts[:, 1], s=markersize, color=color, label=tag, zorder=3)

        for node_id in node_ids:
            labels = []
            if annotate_nodes:
                labels.append(f"n{node_id}")
            if annotate_dofs and dof_handler is not None and field_names:
                dof_labels = []
                for field in field_names:
                    gd = getattr(dof_handler, "dof_map", {}).get(field, {}).get(node_id)
                    if gd is not None:
                        dof_labels.append(f"{field}:{gd}")
                if dof_labels:
                    labels.append("/".join(dof_labels))
            if labels:
                x, y = coords[node_id]
                ax.text(
                    x,
                    y,
                    "\n".join(labels),
                    fontsize=7,
                    color="black",
                    ha="center",
                    va="center",
                    zorder=4,
                    bbox=dict(facecolor="white", alpha=0.7, pad=0.2),
                )

    ax.set_aspect("equal", "box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="best")
    if show:
        plt.show()
    return ax


# ------------- measure area plotting
def _tri_vertex_phi(level_set, node_coords, node_ids, mesh):
    """
    Fast φ at the 3 triangle vertices. Uses FE nodal values if available,
    falls back to point-eval otherwise.
    """
    # FE fast path: read φ DOFs that coincide with mesh nodes
    if hasattr(level_set, "dh") and hasattr(level_set, "field") and hasattr(level_set, "_f"):
        node_map = level_set.dh.dof_map.get(level_set.field, {})
        g2l      = getattr(level_set._f, "_g2l", {})
        nv       = level_set._f.nodal_values
        v_phi    = np.empty(3, dtype=float)
        for j, nid in enumerate(node_ids):
            gd = node_map.get(int(nid))
            if gd is not None and gd in g2l:
                v_phi[j] = float(nv[g2l[gd]])
            else:
                v_phi[j] = float(level_set(node_coords[j]))
        return v_phi
    # Generic path
    return np.array([float(level_set(node_coords[0])),
                     float(level_set(node_coords[1])),
                     float(level_set(node_coords[2]))], dtype=float)


def add_measure_area_overlay(ax, mesh, level_set, *,
                             side='+',
                             include_full_cells=True,
                             facecolor=(0.90, 0.20, 0.60, 0.35),  # RGBA
                             edgecolor='none',
                             label=None):
    """
    Overlay the exact area integrated by:
      - side='+' → dx_has_pos (full 'outside' cells + positive part of cut cells)
      - side='-' → dx_has_neg (full 'inside' cells  + negative part of cut cells)
    Returns (polycollection, legend_patch). Does NOT call ax.legend(...).
    """
    tol = getattr(SIDE, "tol", 1e-12)
    node = mesh.nodes_x_y_pos

    # classify like your assembler
    inside_ids, outside_ids, cut_ids = mesh.classify_elements(level_set)
    full_eids = outside_ids if side == '+' else inside_ids

    tris_to_fill = []

    # --- full cells on requested side ---
    if include_full_cells:
        for eid in full_eids:
            cn = list(mesh.elements_list[eid].corner_nodes)
            if len(cn) == 3:
                tris_to_fill.append(node[cn])                        # (3,2)
            elif len(cn) == 4:
                # match NG / python path: (0–2) diagonal
                tris_to_fill.append(node[[cn[0], cn[1], cn[2]]])
                tris_to_fill.append(node[[cn[0], cn[2], cn[3]]])

    # --- cut cells: clip each corner-triangle ---
    # fast FE nodal φ at mesh nodes (if available)
    def phi_at_nodes(ids):
        if hasattr(level_set, "dh") and hasattr(level_set, "field") and hasattr(level_set, "_f"):
            node_map = level_set.dh.dof_map.get(level_set.field, {})
            g2l      = getattr(level_set._f, "_g2l", {})
            nv       = level_set._f.nodal_values
            out      = np.empty(3, float)
            for j, nid in enumerate(ids):
                gd = node_map.get(int(nid))
                if gd is not None and gd in g2l:
                    out[j] = float(nv[g2l[gd]])
                else:
                    out[j] = float(level_set(node[int(nid)]))
            return out
        # fallback
        return np.array([float(level_set(node[int(ids[0])])),
                         float(level_set(node[int(ids[1])])),
                         float(level_set(node[int(ids[2])]))], float)

    for eid in cut_ids:
        elem = mesh.elements_list[eid]
        tri_local, corner_ids = corner_tris(mesh, elem)  # uses (0–2) diagonal for quads
        for loc_tri in tri_local:
            v_ids    = [corner_ids[i] for i in loc_tri]
            v_coords = node[v_ids]                       # (3,2)
            v_phi    = phi_at_nodes(v_ids)

            # polys = clip_triangle_to_side(v_coords, 
            #                               v_phi, side=side, eps=tol)
            polys = clip_triangle_to_side_pn(mesh, eid, loc_tri, 
                                                        corner_ids, 
                                                        level_set, 
                                                        side=side, 
                                                        eps=SIDE.tol)
            for poly in polys:                           # <- IMPORTANT: iterate polys
                for A, B, C in fan_triangulate(poly):
                    tris_to_fill.append(np.vstack([A, B, C]))  # (3,2)

    if not tris_to_fill:
        return None, None

    pcoll = PolyCollection(tris_to_fill, facecolors=facecolor, edgecolors=edgecolor, zorder=8)
    ax.add_collection(pcoll)

    legend_name = label or (f"has_{'pos' if side=='+' else 'neg'}")
    legend_patch = patches.Patch(color=facecolor if isinstance(facecolor, str) else facecolor[:3],
                                 label=legend_name)
    return pcoll, legend_patch


def _elem_polys(mesh, eids):
    """Return a list of corner-polygons (in order) for the given element ids."""
    pts = mesh.nodes_x_y_pos
    polys = []
    for eid in eids:
        cn = list(mesh.elements_list[int(eid)].corner_nodes)
        polys.append(pts[cn])  # (m,2) with m=4 for quads, 3 for tris
    return polys

def add_element_outline(ax, mesh, eids, *, edgecolor="red", linewidth=2.5, zorder=9, label=None):
    """
    Outline elements with thick borders. Returns (collection, legend_handle).
    """
    polys = _elem_polys(mesh, eids)
    if not polys:
        return None, None
    coll = PolyCollection(polys, facecolors="none", edgecolors=edgecolor,
                          linewidths=linewidth, zorder=zorder)
    ax.add_collection(coll)
    lh = plt.Line2D([0],[0], color=edgecolor, lw=linewidth, label=label or "highlight")
    return coll, lh

def add_element_fill(ax, mesh, eids, *, facecolor=(1.0, 0.0, 0.0, 0.25), edgecolor="none",
                     zorder=8, label=None):
    """
    Softly fill elements (semi-transparent). Returns (collection, legend_handle).
    """
    polys = _elem_polys(mesh, eids)
    if not polys:
        return None, None
    coll = PolyCollection(polys, facecolors=facecolor, edgecolors=edgecolor,
                          linewidths=0.0, zorder=zorder)
    ax.add_collection(coll)
    # legend color: use RGB of facecolor
    rgb = facecolor if isinstance(facecolor, str) else facecolor[:3]
    lh = plt.Line2D([0],[0], color=rgb, lw=8, label=label or "highlight")
    return coll, lh

def zoom_to_elements(ax, mesh, eids, pad=0.05):
    """
    Zoom the view to tightly fit the given elements (with relative padding).
    """
    pts = mesh.nodes_x_y_pos
    xs, ys = [], []
    for eid in eids:
        cn = list(mesh.elements_list[int(eid)].corner_nodes)
        xy = pts[cn]
        xs.extend(xy[:,0]); ys.extend(xy[:,1])
    if xs:
        xmin, xmax = min(xs), max(xs); xr = xmax - xmin
        ymin, ymax = min(ys), max(ys); yr = ymax - ymin
        ax.set_xlim(xmin - pad*xr, xmax + pad*xr)
        ax.set_ylim(ymin - pad*yr, ymax + pad*yr)
