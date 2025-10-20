import numpy as np

def hansbo_cut_ratio(mesh, level_set, tol=1e-12, n_mc=1000):
    """
    Approximates the cut ratio |T ∩ Ω-| / |T| for each element T
    using a Monte Carlo integration method.

    Args:
        mesh: The mesh object.
        level_set: A callable level-set object.
        tol: Tolerance for the level set.
        n_mc: Number of Monte Carlo sample points per element.

    Returns:
        A numpy array where the i-th entry is the cut ratio for element i.
    """
    cut_ratios = np.zeros(len(mesh.elements_list))
    for i, elem in enumerate(mesh.elements_list):
        if elem.tag != 'IF': # Only compute for cut elements
            cut_ratios[i] = 1.0 if elem.tag == 'NEG' else 0.0
            continue
            
        # Generate random points within the element's bounding box
        # A more robust method would sample within the element itself.
        coords = mesh.nodes_x_y_pos[list(elem.corner_nodes)]
        min_c = coords.min(axis=0)
        max_c = coords.max(axis=0)
        
        rand_pts = np.random.rand(n_mc, 2) * (max_c - min_c) + min_c
        
        # Evaluate level set at random points
        phi_vals = np.apply_along_axis(level_set, 1, rand_pts)
        
        # Ratio is the fraction of points inside the domain
        cut_ratios[i] = np.sum(phi_vals < tol) / n_mc
        
    return cut_ratios
