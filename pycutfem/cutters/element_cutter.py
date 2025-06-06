import numpy as np
# Assuming Mesh is imported from your core module, e.g.:
# from pycutfem.core import Mesh 

# Note: The level_set object is assumed to have a method `evaluate_on_nodes(mesh)`
# that returns a numpy array of level set values, one for each node in mesh.nodes.

def _phi_on_centroids(mesh, level_set_callable):
    """
    Evaluates the level set function at the centroid of each element.
    This function remains compatible with the new Mesh class.
    """
    # mesh.nodes is the (N,2) numpy array of coordinates.
    # mesh.elements_connectivity is the (M, k) numpy array of indices.
    # This correctly computes the centroid for both linear and higher-order elements
    # by averaging the coordinates of *all* nodes in the element.
    centroids = mesh.nodes[mesh.elements_connectivity].mean(axis=1)
    
    # level_set_callable is assumed to be a function that can take an (N,2) array of points.
    # If it takes (x,y) separately, np.apply_along_axis is also fine.

    return np.apply_along_axis(level_set_callable, 1, centroids)


def classify_elements(mesh, level_set, tol=1e-12):
    """
    Classifies elements as 'inside', 'outside', or 'cut' by the level set.
    
    This version is updated to work with the new Mesh class by setting the `.tag`
    attribute on each object in `mesh.elements_list`.
    """
    # Evaluate the level set function on all nodes of the mesh.
    phi_nodes = level_set.evaluate_on_nodes(mesh)
    
    # Gather the level set values for all nodes of each element.
    elem_phi_nodes = phi_nodes[mesh.elements_connectivity]
    
    # Also evaluate at the centroid for more robust classification of curved interfaces.
    phi_cent = _phi_on_centroids(mesh, level_set)

    # Determine the classification for each element based on min/max phi values.
    min_phi_per_elem = np.minimum(elem_phi_nodes.min(axis=1), phi_cent)
    max_phi_per_elem = np.maximum(elem_phi_nodes.max(axis=1), phi_cent)

    inside_mask = max_phi_per_elem < -tol
    outside_mask = min_phi_per_elem > tol
    cut_mask = ~(inside_mask | outside_mask)

    # --- UPDATED SECTION ---
    # Set the .tag attribute on each individual Element object.
    
    inside_indices = np.where(inside_mask)[0]
    outside_indices = np.where(outside_mask)[0]
    cut_indices = np.where(cut_mask)[0]

    for eid in inside_indices:
        mesh.elements_list[eid].tag = 'inside'
    for eid in outside_indices:
        mesh.elements_list[eid].tag = 'outside'
    for eid in cut_indices:
        mesh.elements_list[eid].tag = 'cut'
    
    # Return the indices for convenience, as before.
    return inside_indices, outside_indices, cut_indices


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