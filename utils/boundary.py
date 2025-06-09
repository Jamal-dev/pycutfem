import numpy as np

def get_boundary_dofs(function_space, bcs):
    """
    Identifies all global degrees of freedom associated with a list of
    boundary conditions.
    """
    mesh = function_space.mesh
    n_loc = function_space.num_local_dofs()
    n_comp = function_space.dim
    n_eldof = n_loc * n_comp
    
    boundary_dofs = set()
    
    # Create a mapping from node GID to a list of elements it belongs to
    node_to_elem = [[] for _ in range(len(mesh.nodes_list))]
    for elem in mesh.elements_list:
        for node_gid in elem.nodes:
            node_to_elem[node_gid].append(elem.id)
            
    for bc in bcs:
        for edge in mesh.edges_list:
            if edge.tag == bc.tag:
                # Get all nodes on this boundary edge
                for node_gid in edge.nodes:
                    # Find all elements attached to this node
                    for elem_id in node_to_elem[node_gid]:
                        # Map local to global DOFs for this element
                        dofs = np.arange(n_eldof) + elem_id * n_eldof
                        boundary_dofs.update(dofs)
                        
    return list(boundary_dofs)

