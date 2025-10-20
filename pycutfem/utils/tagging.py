import numpy as np

def classify_by_levelset(mesh, level_set, tol=1e-12, 
                         neg_tag='inside', pos_tag='outside', cut_tag='cut',
                         ghost_tag='ghost', interface_tag='interface'):
    """
    Classifies elements and edges of a mesh based on a level-set function.
    """
    # This assumes level_set has an `evaluate_on_nodes` method.
    phi_nodes = level_set.evaluate_on_nodes(mesh)
    
    # 1. Classify Elements
    for elem in mesh.elements_list:
        # Use the corner_nodes for a linear classification
        elem_phi_nodes = phi_nodes[list(elem.corner_nodes)]
        
        if np.all(elem_phi_nodes < -tol):
            elem.tag = neg_tag
        elif np.all(elem_phi_nodes > tol):
            elem.tag = pos_tag
        else:
            elem.tag = cut_tag
            
    # 2. Classify Edges
    for edge in mesh.edges_list:
        edge.tag = '' # Reset tag
        if edge.right is not None: # Interior edge
            tag_L = mesh.elements_list[edge.left].tag
            tag_R = mesh.elements_list[edge.right].tag
            
            if tag_L == cut_tag and tag_R == cut_tag:
                edge.tag = ghost_tag
            elif tag_L != tag_R:
                edge.tag = interface_tag
