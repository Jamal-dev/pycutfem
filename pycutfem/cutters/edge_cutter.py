"""pycutfem.cutters.edge_cutter
Robust interface/ghost edge classification.

Criteria
--------
1. If *level_set* is given: an edge is 'interface' when φ(a)⋅φ(b) < 0.
2. Else: use element tags produced by *classify_elements*:
   - 'interface' when incident element tags differ ('inside' vs 'outside')
     OR one side is 'cut'.
   - 'ghost' when both incident elements are 'cut'.
"""
import numpy as np

def classify_edges(mesh, level_set=None):
    tags = np.array([''] * len(mesh.edges), dtype=object)

    if level_set is not None:
        phi_nodes = level_set.evaluate_on_nodes(mesh)
        for e in mesh.edges:
            if phi_nodes[e.nodes[0]] * phi_nodes[e.nodes[1]] < 0:
                tags[e.id] = 'interface'

    # Fallback / additional classification via element tags
    for e in mesh.edges:
        if tags[e.id]:  # already set
            continue
        if e.right is None:
            continue  # boundary edge
        left = mesh.elem_tag[e.left]
        right = mesh.elem_tag[e.right]
        if {'inside', 'outside'} == {left, right} or 'cut' in (left, right):
            tags[e.id] = 'interface'
        elif left == right == 'cut':
            tags[e.id] = 'ghost'

    mesh.edge_tag[:] = tags
    return tags
