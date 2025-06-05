"""pycutfem.assembly.dg_local
Local element and face kernels for **SIPG** (Symmetric Interior               
Penalty Galerkin) discretisation of  −Δu = f.  Works for arbitrary order
P_k / Q_k on triangles or quads.                                           
"""
import numpy as np
from pycutfem.integration import volume, edge
from pycutfem.fem.reference import get_reference
from pycutfem.fem import transform

# -------------------------------------------------------------------------
# Element volume contribution — identical to CG (just uses local DOF ids)
# -------------------------------------------------------------------------

def volume_laplace(mesh, elem_id, *, poly_order, quad_order=None):
    if quad_order is None:
        quad_order = poly_order + 2
    pts, wts = volume(mesh.element_type, quad_order)
    ref = get_reference(mesh.element_type, poly_order)
    n_loc = len(ref.shape(0, 0))
    Ke = np.zeros((n_loc, n_loc))
    for (xi, eta), w in zip(pts, wts):
        dN = ref.grad(xi, eta)                    # (n_loc, 2)
        J = transform.jacobian(mesh, elem_id, (xi, eta))
        invJT = np.linalg.inv(J).T
        grad = dN @ invJT                        # (n_loc, 2)
        detJ = np.abs(np.linalg.det(J))
        Ke += w * detJ * grad @ grad.T
    return Ke

# -------------------------------------------------------------------------
# Face contribution  (SIPG): left/right element pair
# -------------------------------------------------------------------------

# assembly/dg_local.py
# ----------------------------------------------------------------------
def face_laplace(mesh, eL, eR, edge_id, *, poly_order,
                 penalty, quad_order=None,
                 dirichlet=lambda x, y: 0.0):
    """
    Return blocks  (K_LL, K_LR, K_RL, K_RR,  F_L, F_R)

    K_LR … are None on a boundary face (eR is None);
    F_R is None on a boundary face.
    """
    if quad_order is None:
        quad_order = poly_order + 2

    ref   = get_reference(mesh.element_type, poly_order)
    n_loc = len(ref.shape(0, 0))

    # ---- edge geometry ------------------------------------------------
    edge_obj = mesh.edge(edge_id)
    normal   = edge_obj.normal
    idx_L    = _find_local_edge(mesh, eL, edge_obj.nodes)

    # reference-edge quadrature (pts in element reference coords)
    pts_ref, w_ref = edge(mesh.element_type, idx_L, quad_order)

    # physical edge length → 1-D Jacobian
    p0 = mesh.nodes[edge_obj.nodes[0]]
    p1 = mesh.nodes[edge_obj.nodes[1]]
    jac_edge = np.linalg.norm(p1 - p0) / 2.0   # ref edge assumed length 2

    # ---- helpers ------------------------------------------------------
    def grad_phys(elem_id, xi_eta):
        dN = ref.grad(*xi_eta)
        J  = transform.jacobian(mesh, elem_id, xi_eta)
        return dN @ np.linalg.inv(J).T

    # local matrices/vectors
    K_LL = np.zeros((n_loc, n_loc))
    K_LR = K_RL = K_RR = None
    F_L  = np.zeros(n_loc)
    F_R  = None

    if eR is not None:
        K_LR = np.zeros((n_loc, n_loc))
        K_RL = np.zeros((n_loc, n_loc))
        K_RR = np.zeros((n_loc, n_loc))
        F_R  = np.zeros(n_loc)

    # element size & penalty
    hL = element_char_length(mesh, eL)
    hR = element_char_length(mesh, eR) if eR is not None else hL
    h  = 0.5 * (hL + hR)
    sigma = 10.0 * penalty * (poly_order + 1) * (poly_order + 2) / h

    # ---- quadrature loop ---------------------------------------------
    for (xiL, etaL), w in zip(pts_ref, w_ref):
        w_phys = w * jac_edge                      # scale weight

        xphys  = transform.x_mapping(mesh, eL, (xiL, etaL))
        NL     = ref.shape(xiL, etaL)
        gL     = grad_phys(eL, (xiL, etaL))

        if eR is not None:                         # interior face
            xiR, etaR = _inverse_map(mesh, eR, xphys)
            NR        = ref.shape(xiR, etaR)
            gR        = grad_phys(eR, (xiR, etaR))
            avg_g     = 0.5 * (gL + gR)
            jump_L, jump_R = NL, -NR
        else:                                      # boundary face
            avg_g     = gL
            jump_L    = NL
            uD        = dirichlet(*xphys)

        # volume-free blocks -------------------------------------------
        

        if eR is not None:                         # interior blocks
            K_LL += w_phys * ( (avg_g @ normal)[:,None] * jump_L[None,:] +
                           (avg_g @ normal)[None,:] * jump_L[:,None] +
                           sigma * np.outer(jump_L, jump_L) )
            K_LR += w_phys * ( (avg_g @ normal)[:,None] * jump_R[None,:] +
                               sigma * np.outer(jump_L, jump_R) )
            K_RL += w_phys * ( (avg_g @ normal)[:,None] * jump_L[None,:] +
                               sigma * np.outer(jump_R, jump_L) )
            K_RR += w_phys * ( (avg_g @ normal)[:,None] * jump_R[None,:] +
                               (avg_g @ normal)[None,:] * jump_R[:,None] +
                               sigma * np.outer(jump_R, jump_R) )
        else:                                      # Dirichlet RHS
            uD     = dirichlet(*xphys)
            jump_L = NL                            # u_R ≡ u_D

            # only (∇u_L·n) v_L  +  σ(u_L v_L) stays in the matrix
            K_LL += w_phys * ( (gL @ normal)[:,None] * jump_L[None,:] +
                               sigma * np.outer(jump_L, jump_L) )
            F_L += w_phys * ( -(gL @ normal) * uD - sigma * uD ) * NL

    return K_LL, K_LR, K_RL, K_RR, F_L, F_R


# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------




def element_char_length(mesh, elem_id):
    if elem_id is None:
        return 0.0
    return np.sqrt(mesh.areas()[elem_id])

def _find_local_edge(mesh, elem_id_of_interest: int, global_edge_nodes_to_find: tuple):
    """
    Finds the local index (0, 1, 2, or 3 for quads; 0, 1, 2 for tris) 
    of a given global edge within a specific element.

    Args:
        mesh: The Mesh object.
        elem_id_of_interest: The global ID of the element in which to find the local edge.
        global_edge_nodes_to_find: A tuple (gid1, gid2) of the global node IDs 
                                     forming the edge to find. These GIDs should be
                                     the corner/primary vertices of the edge.
    Returns:
        int: The local index of the edge within the element's canonical edge list.
    Raises:
        RuntimeError: If the edge is not found as a local edge of the element.
    """
    # Get the global node IDs of the CORNER/PRIMARY vertices for the element of interest.
    # The order of these GIDs is canonical (e.g., BL, BR, TR, TL for _get_element_corner_global_indices output for quads)
    actual_elem_corners_gids = mesh._get_element_corner_global_indices(elem_id_of_interest)

    if not actual_elem_corners_gids:
        raise RuntimeError(f"Element {elem_id_of_interest} has no identifiable corner vertices.")
    
    # _EDGE_TABLE defines edges using *local indices relative to the ordered list of corners*.
    # e.g., for quads: ((0,1), (1,2), (2,3), (3,0)) means edge between 0th & 1st corner, 1st & 2nd corner, etc.
    local_edge_definitions_via_corner_indices = mesh._EDGE_TABLE[mesh.element_type]

    set_of_gids_to_find = set(global_edge_nodes_to_find)

    for local_edge_canonical_idx, (idx_corner1, idx_corner2) in enumerate(local_edge_definitions_via_corner_indices):
        # Get the global node IDs that form this conceptual local edge
        # by indexing into the element's actual list of corner GIDs.
        
        # Check if conceptual corner indices are valid for the fetched actual_elem_corners_gids
        if idx_corner1 >= len(actual_elem_corners_gids) or idx_corner2 >= len(actual_elem_corners_gids):
            raise IndexError(
                f"Conceptual corner indices ({idx_corner1}, {idx_corner2}) from _EDGE_TABLE "
                f"are out of bounds for the actual number of corners ({len(actual_elem_corners_gids)}) "
                f"found for element {elem_id_of_interest} (type: {mesh.element_type}, order: {mesh.poly_order})."
            )
            
        gid_of_corner1 = actual_elem_corners_gids[idx_corner1]
        gid_of_corner2 = actual_elem_corners_gids[idx_corner2]
        
        current_local_edge_as_set_of_gids = {gid_of_corner1, gid_of_corner2}
        
        if current_local_edge_as_set_of_gids == set_of_gids_to_find:
            return local_edge_canonical_idx # This is the local edge index (0, 1, 2, ...)
            
    raise RuntimeError(
        f"Edge with global nodes {global_edge_nodes_to_find} was not found as a local edge "
        f"of element {elem_id_of_interest} (type: {mesh.element_type}, order: {mesh.poly_order}). "
        f"Actual element corners (GIDs): {actual_elem_corners_gids}."
    )






# Newton solver to get (u,v) in right element (for demo / tests only)
# ----------------------------------------------------------------------

def _inverse_map(mesh, elem_id, x, tol=1e-10, maxiter=10):
    ref   = get_reference(mesh.element_type, mesh.poly_order)
    xi = np.array([0.0, 0.0])
    for _ in range(maxiter):
        X = transform.x_mapping(mesh, elem_id, xi)
        J = transform.jacobian(mesh, elem_id, xi)
        delta = np.linalg.solve(J, x - X)
        xi += delta
        if np.linalg.norm(delta) < tol:
            break
    return xi
