"""pycutfem.assembly.dg_local
Local element and face kernels for **SIPG** (Symmetric Interior               
Penalty Galerkin) discretisation of  −Δu = f.  Works for arbitrary order
P_k / Q_k on triangles or quads.                                           
"""
import numpy as np
from pycutfem.integration import volume, edge
from pycutfem.fem.reference import get_reference
from pycutfem.fem import transform
from typing import Callable, Tuple

# -----------------------------------------------------------------------------
# Volume kernel (unchanged from CG, but loops over vector components if needed)
# -----------------------------------------------------------------------------

def volume_laplace(mesh, elem_id: int, *, poly_order: int, quad_order: int | None = None,
                   n_comp: int = 1) -> np.ndarray:
    """Return local stiffness block of shape (n_loc*n_comp, n_loc*n_comp)."""
    if quad_order is None:
        quad_order = poly_order + 2

    ref  = get_reference(mesh.element_type, poly_order)
    pts, wts = volume(mesh.element_type, quad_order)
    n_loc = len(ref.shape(0, 0))
    Ke    = np.zeros((n_loc * n_comp, n_loc * n_comp))

    for (xi, eta), w in zip(pts, wts):
        dN   = ref.grad(xi, eta)                 # (n_loc,2)
        J    = transform.jacobian(mesh, elem_id, (xi, eta))
        invJT = np.linalg.inv(J).T
        grad = dN @ invJT                       # (n_loc,2)
        detJ = abs(np.linalg.det(J))
        k_s  = w * detJ * grad @ grad.T         # (n_loc,n_loc)
        for c in range(n_comp):                 # diagonal block per component
            i0 = c * n_loc
            Ke[i0:i0+n_loc, i0:i0+n_loc] += k_s
    return Ke

# -----------------------------------------------------------------------------
# Face kernel: row‑matrix‑rowᵀ formulation (works for vector/scalar)
# -----------------------------------------------------------------------------

def face_laplace(mesh, eL: int, eR: int | None, edge_id: int, *,
                 poly_order: int, alpha: float = 10.0, symmetry: int = 1,
                 quad_order: int | None = None, n_comp: int = 1,
                 dirichlet: Callable[[float, float], float] | Callable[[float, float], np.ndarray] = lambda x, y: 0.0
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Ke_face, Fe_face) where
         *Ke_face* is a dense block (n_face_dofs × n_face_dofs).
         The calling assembler adds it to the global CSR.
    """
    if quad_order is None:
        quad_order = poly_order + 2

    ref  = get_reference(mesh.element_type, poly_order)
    n_loc = len(ref.shape(0, 0))
    n_fld = n_loc * (1 + int(eR is not None))  # Scalar case: n_comp=1

    Ke = np.zeros((n_fld, n_fld))
    Fe = np.zeros(n_fld)

    # --- Geometry ------------------------------------------------------
    edge_obj = mesh.edge(edge_id)
    idx_L    = _find_local_edge(mesh, eL, edge_obj.nodes)
    pts_ref, w_ref = edge(mesh.element_type, idx_L, quad_order)

    def geom(xi, eta):
        J = transform.jacobian(mesh, eL, (xi, eta))
        if mesh.element_type == "quad":
            n_ref = np.array([[0,-1],[1,0],[0,1],[-1,0]][idx_L])
            dS_dxi = 1.0
        else:  # tri edges in CCW order
            n_ref = np.array([[0,-1],[1,1],[-1,0]][idx_L]) / np.sqrt(2)
            dS_dxi = 1.0
        v = np.linalg.det(J) * J.T @ n_ref
        n_phys = v / np.linalg.norm(v)
        jac_1d = np.linalg.norm(v) * dS_dxi
        return n_phys, jac_1d

    hL = element_char_length(mesh, eL)
    hR = element_char_length(mesh, eR) if eR is not None else hL
    h  = 0.5 * (hL + hR)
    sigma = alpha * (poly_order + 1) * (poly_order + 2) / h

    B = np.array([[0, symmetry], [-1, sigma]])

    # --- Quadrature Loop -----------------------------------------------
    for (xi, eta), w in zip(pts_ref, w_ref):
        n, jac1d = geom(xi, eta)
        lam = w * jac1d

        φL  = ref.shape(xi, eta)
        gL  = ref.grad(xi, eta) @ transform.inv_jac_T(mesh, eL, (xi, eta))
        dφnL = gL @ n

        if eR is not None:  # Interior face
            xr = transform.x_mapping(mesh, eL, (xi, eta))
            xiR, etaR = _inverse_map(mesh, eR, xr)
            φR  = ref.shape(xiR, etaR)
            gR  = ref.grad(xiR, etaR) @ transform.inv_jac_T(mesh, eR, (xiR, etaR))
            dφnR = gR @ n
            # Concatenate left and right DOFs for jumps
            row1 = np.concatenate((0.5 * dφnL, -0.5 * dφnR))  # [∂n φ] = ∂n φ_L - ∂n φ_R
            row2 = np.concatenate((φL, -φR))                   # [φ] = φ_L - φ_R
        else:  # Boundary face
            uD_val = dirichlet(*transform.x_mapping(mesh, eL, (xi, eta)))
            uD_val = np.atleast_1d(uD_val) if np.isscalar(uD_val) else uD_val
            row1 = 0.5 * dφnL
            row2 = φL

        R = np.vstack((row1, row2))  # (2, n_fld)
        Ke += lam * R.T @ B @ R      # (n_fld, n_fld)

        # Boundary RHS
        if eR is None:
            vL = symmetry * row1 + sigma * row2  # vL is (n_loc,)
            Fe += lam * (-vL * uD_val[0])        # Scalar case: use first component

    return Ke, Fe


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
