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
    FIXED: The logic is now a clean if/elif/elif chain to prevent
    misclassification and ensure all conditions are properly checked.
    """
    phi_nodes = level_set.evaluate_on_nodes(mesh)
    for edge in mesh.edges_list:
        # Check 1: Primary classification based on level set crossing vertices.
        if phi_nodes[edge.nodes[0]] * phi_nodes[edge.nodes[1]] < 0:
            edge.tag = 'interface'
            continue # This edge is definitively an interface, move to the next.

        # Check 2: Fallback for interior edges based on element tags.
        if edge.right is not None:
            left_tag = mesh.elements_list[edge.left].tag
            right_tag = mesh.elements_list[edge.right].tag
            
            # This order is important:
            if {'inside', 'outside'} == {left_tag, right_tag}:
                edge.tag = 'interface'
            elif left_tag == 'cut' and right_tag == 'cut':
                edge.tag = 'ghost'
            elif 'cut' in (left_tag, right_tag) and left_tag != right_tag:
                edge.tag = 'interface'
