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

def classify_edges(mesh, level_set):
    """
    Robust interface/ghost edge classification, given each element is already
    tagged in {'inside', 'outside', 'cut'}.

    1. If φ changes sign between the two endpoints of an edge, tag it 'interface'.
    2. Otherwise, for interior edges (edge.right is not None):
       - If {left_tag, right_tag} == {'inside', 'outside'} → 'interface'
       - Elif left_tag=='cut' and right_tag=='cut'            → 'ghost'
       - Elif exactly one side is 'cut' (and the other is 'inside' or 'outside')
             → 'interface'
       - Else → leave tag as '' (no classification needed).
    """
    # Evaluate φ at all nodes once
    phi_nodes = level_set.evaluate_on_nodes(mesh)

    for edge in mesh.edges_list:
        # 1) Primary: sign‐change in φ across the edge’s endpoints?
        if phi_nodes[edge.nodes[0]] * phi_nodes[edge.nodes[1]] < 0:
            edge.tag = 'interface'
            continue

        # 2) Fallback: only if this is an interior edge
        if edge.right is not None:
            left_tag  = mesh.elements_list[edge.left].tag
            right_tag = mesh.elements_list[edge.right].tag

            # inside/outside adjacent → interface
            if {left_tag, right_tag} == {'inside', 'outside'}:
                edge.tag = 'interface'

            # both cut → ghost
            elif left_tag == 'cut' and right_tag == 'cut':
                edge.tag = 'ghost'

            # exactly one side cut → interface
            elif ('cut' in (left_tag, right_tag)) and (left_tag != right_tag):
                edge.tag = 'interface'

            # otherwise (both inside or both outside), leave edge.tag as ''