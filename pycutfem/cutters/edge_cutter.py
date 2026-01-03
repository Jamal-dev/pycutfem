"""pycutfem.cutters.edge_cutter
Interface/ghost edge classification for cut meshes.

Criteria
--------
1. Aligned interface edges: adjacent elements are inside/outside and both
   edge endpoints satisfy |φ|<=tol.
2. Ghost edges: any interior edge with at least one cut element.
"""
def classify_edges(mesh, level_set, tol=1e-12):
    """
    Interface/ghost edge classification, given each element is already
    tagged in {'inside', 'outside', 'cut'}.

    1. Interface edges: only when they separate inside/outside elements and
       φ≈0 at both endpoints (aligned interface).
    2. Ghost edges: any interior edge with at least one cut element.
    """
    # Evaluate φ at all nodes once
    phi_nodes = level_set.evaluate_on_nodes(mesh)

    for edge in mesh.edges_list:
        if edge.right is None:
            continue

        edge.tag = ''
        left_tag = mesh.elements_list[edge.left].tag
        right_tag = mesh.elements_list[edge.right].tag
        tags = {left_tag, right_tag}

        # Aligned interface facets: inside/outside neighbors with φ≈0 endpoints.
        if left_tag in {"inside", "outside"} and right_tag in {"inside", "outside"} and left_tag != right_tag:
            n0, n1 = edge.nodes
            p0, p1 = phi_nodes[n0], phi_nodes[n1]
            if abs(p0) <= tol and abs(p1) <= tol:
                edge.tag = 'interface'
            continue

        # Ghost stabilization edges: any edge touching a cut element.
        if 'cut' in tags:
            edge.tag = 'ghost'
