import numpy as np
# Assuming Mesh is imported from your core module, e.g.:
# from pycutfem.core import Mesh 

# Note: The level_set object is assumed to have a method `evaluate_on_nodes(mesh)`
# that returns a numpy array of level set values, one for each node in mesh.nodes.



def _get_node_coords(mesh):
    """
    Return an (n_nodes × 2) ndarray of node coordinates, regardless of whether
    mesh.nodes was passed in as a list of Node‐objects or as a numeric array.
    """
    raw = mesh.nodes
    # If it's already a numpy array of shape (n,2), just return it
    if isinstance(raw, np.ndarray):
        return raw
    # Otherwise, assume it's a list (or other sequence) of Node instances
    try:
        return np.array([[n.x, n.y] for n in raw], dtype=float)
    except Exception:
        raise RuntimeError("Cannot interpret mesh.nodes as coordinates.")


def _phi_on_centroids(mesh, level_set):
    """
    Compute φ at each element’s centroid. Uses _get_node_coords to handle
    both ndarray and list‐of‐Node cases.
    """
    node_coords = _get_node_coords(mesh)                        # (n_nodes, 2)
    conn = mesh.elements_connectivity                           # (n_elems, n_corners)
    # Gather corner coordinates for each element → shape (n_elems, n_corners, 2)
    corner_coords = node_coords[conn]
    centroids = corner_coords.mean(axis=1)                      # (n_elems, 2)
    return level_set(centroids)                                 # (n_elems,)


def classify_elements(mesh, level_set, tol=1e-12):
    """
    Classify each element as 'inside', 'outside', or 'cut' by combining:
      - φ evaluated on the element’s corner-nodes
      - φ evaluated at the element’s centroid

    An element is:
      • 'inside'  if max(φ(corners), φ(centroid)) < -tol
      • 'outside' if min(φ(corners), φ(centroid)) > +tol
      • 'cut'     otherwise

    Sets mesh.elements_list[eid].tag accordingly, and returns
    (inside_indices, outside_indices, cut_indices).
    """
    # 1) φ on every node (level_set.evaluate_on_nodes handles mesh‐structure)
    phi_nodes = level_set.evaluate_on_nodes(mesh)                       # shape (n_nodes,)

    # 2) Gather φ at each element’s corners
    conn = mesh.elements_connectivity                                   # shape (n_elems, n_corners)
    elem_phi_nodes = phi_nodes[conn]                                    # shape (n_elems, n_corners)

    # 3) φ at centroids
    phi_cent = _phi_on_centroids(mesh, level_set)                       # shape (n_elems,)

    # 4) Compute per-element min/max
    min_corner = elem_phi_nodes.min(axis=1)
    max_corner = elem_phi_nodes.max(axis=1)
    min_phi_per_elem = np.minimum(min_corner, phi_cent)
    max_phi_per_elem = np.maximum(max_corner, phi_cent)

    inside_mask  = (max_phi_per_elem < -tol)
    outside_mask = (min_phi_per_elem >  tol)
    cut_mask     = ~(inside_mask | outside_mask)

    inside_inds  = np.where(inside_mask)[0]
    outside_inds = np.where(outside_mask)[0]
    cut_inds     = np.where(cut_mask)[0]

    # 5) Tag elements
    for eid in inside_inds:
        mesh.elements_list[eid].tag = 'inside'
    for eid in outside_inds:
        mesh.elements_list[eid].tag = 'outside'
    for eid in cut_inds:
        mesh.elements_list[eid].tag = 'cut'

    return inside_inds, outside_inds, cut_inds


def classify_elements_multi(mesh, level_sets, tol=1e-12):
    """
    Classifies elements against multiple level sets.
    This function works without modification as it relies on the updated
    classify_elements function.
    """
    results = {}
    for idx, ls in enumerate(level_sets):
        results[idx] = classify_elements(mesh, ls, tol)
    return results