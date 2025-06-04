"""pycutfem.cutters.element_cutter (revised)
More robust classification: looks at element barycentre in addition to
vertex values so curved interfaces that do not hit a vertex are still
tagged as *cut*.
"""
import numpy as np

def _phi_on_centroids(mesh, level_set):
    centroids=mesh.nodes[mesh.elements].mean(axis=1)
    return np.apply_along_axis(level_set, 1, centroids)

def classify_elements(mesh, level_set, tol=1e-12):
    phi_nodes=level_set.evaluate_on_nodes(mesh)        # (n_nodes,)
    elem_phi_nodes=phi_nodes[mesh.elements]            # (n_elem,n_loc)
    phi_cent=_phi_on_centroids(mesh, level_set)        # (n_elem,)

    min_phi=np.minimum(elem_phi_nodes.min(axis=1), phi_cent)
    max_phi=np.maximum(elem_phi_nodes.max(axis=1), phi_cent)

    inside=max_phi < -tol
    outside=min_phi > tol
    cut=~(inside|outside)

    mesh.elem_tag[inside]='inside'
    mesh.elem_tag[outside]='outside'
    mesh.elem_tag[cut]='cut'
    return inside.nonzero()[0], outside.nonzero()[0], cut.nonzero()[0]

def classify_elements_multi(mesh, level_sets, tol=1e-12):
    results={}
    for idx,ls in enumerate(level_sets):
        results[idx]=classify_elements(mesh, ls, tol)
    return results
