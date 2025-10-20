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
from pycutfem.core.mesh import Mesh

# -----------------------------------------------------------------------------
# Volume kernel (unchanged from CG, but loops over vector components if needed)
# -----------------------------------------------------------------------------

def volume_laplace(mesh: 'Mesh', elem_id: int, *, n_comp: int = 1, quad_order: int | None = None) -> np.ndarray:
    """Return local stiffness block for an element."""
    # FIX: poly_order is now retrieved from the mesh object.
    if quad_order is None:
        quad_order = mesh.poly_order + 2

    ref = get_reference(mesh.element_type, mesh.poly_order)
    pts, wts = volume(mesh.element_type, quad_order)
    n_loc = len(ref.shape(0, 0))
    Ke = np.zeros((n_loc * n_comp, n_loc * n_comp))

    for (xi, eta), w in zip(pts, wts):
        dN = ref.grad(xi, eta)
        J = transform.jacobian(mesh, elem_id, (xi, eta))
        invJT = np.linalg.inv(J).T
        grad = dN @ invJT
        detJ = abs(np.linalg.det(J))
        k_s = w * detJ * grad @ grad.T
        for c in range(n_comp):
            i0 = c * n_loc
            Ke[i0:i0+n_loc, i0:i0+n_loc] += k_s
    return Ke




# -----------------------------------------------------------------------------
# Face kernel: row‑matrix‑rowᵀ formulation (works for vector/scalar)
# -----------------------------------------------------------------------------

def face_laplace(mesh: 'Mesh', edge_id: int, *, alpha: float = 10.0, symmetry: int = 1,
                 quad_order: int | None = None, n_comp: int = 1,
                 dirichlet: Callable = lambda x, y: 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the stiffness matrix and force vector for a face (edge) using a
    Symmetric Interior Penalty Galerkin (SIPG) formulation.
    """
    if quad_order is None:
        quad_order = mesh.poly_order + 2

    ref = get_reference(mesh.element_type, mesh.poly_order)
    n_loc = len(ref.shape(0,0))

    edge_obj = mesh.edge(edge_id)
    eL = edge_obj.left
    eR = edge_obj.right
    is_interior = eR is not None
    
    n_dofs_L = n_loc * n_comp
    n_dofs_R = n_loc * n_comp if is_interior else 0
    n_fld = n_dofs_L + n_dofs_R
    
    Ke = np.zeros((n_fld, n_fld))
    Fe = np.zeros(n_fld)

    try:
        local_edge_idx = mesh.elements_list[eL].edges.index(edge_id)
    except ValueError:
        raise RuntimeError(f"Edge {edge_id} not found in element {eL}'s edge list.")

    pts_ref, w_ref = edge(mesh.element_type, local_edge_idx, quad_order)

    hL = mesh.element_char_length(eL)
    hR = mesh.element_char_length(eR) if is_interior else hL
    h_edge = 0.5 * (hL + hR)
    sigma = alpha * (mesh.poly_order + 1)**2 / h_edge

    for (xi, eta), w in zip(pts_ref, w_ref):
        n = edge_obj.normal
        jac1d = transform.jacobian_1d(mesh, eL, (xi, eta), local_edge_idx)
        lam = w * jac1d

        phi_L = ref.shape(xi, eta)
        grad_L_phys = ref.grad(xi, eta) @ transform.inv_jac_T(mesh, eL, (xi, eta))
        dphidn_L = grad_L_phys @ n

        if is_interior:
            x_phys = transform.x_mapping(mesh, eL, (xi, eta))
            xi_R, eta_R = _inverse_map(mesh, eR, x_phys)
            phi_R = ref.shape(xi_R, eta_R)
            grad_R_phys = ref.grad(xi_R, eta_R) @ transform.inv_jac_T(mesh, eR, (xi_R, eta_R))
            dphidn_R = grad_R_phys @ n

            for c in range(n_comp):
                sL = slice(c * n_loc, (c + 1) * n_loc)
                sR = slice(n_dofs_L + c * n_loc, n_dofs_L + (c + 1) * n_loc)
                
                jump_v = np.zeros(n_fld)
                jump_v[sL] = phi_L
                jump_v[sR] = -phi_R

                avg_grad_v_n = np.zeros(n_fld)
                avg_grad_v_n[sL] = 0.5 * dphidn_L
                avg_grad_v_n[sR] = -0.5 * dphidn_R

                Ke += lam * (sigma * np.outer(jump_v, jump_v) - \
                             np.outer(jump_v, avg_grad_v_n) - \
                             symmetry * np.outer(avg_grad_v_n, jump_v))
        else:  # Boundary face
            x_phys = transform.x_mapping(mesh, eL, (xi, eta))
            uD_vals = np.atleast_1d(dirichlet(*x_phys))

            for c in range(n_comp):
                s = slice(c * n_loc, (c + 1) * n_loc)
                uD_val = uD_vals[c]

                Ke[s,s] += lam * (sigma * np.outer(phi_L, phi_L) - \
                                 np.outer(phi_L, dphidn_L) - \
                                 symmetry * np.outer(dphidn_L, phi_L))
                
                Fe[s] += lam * (sigma * phi_L * uD_val - symmetry * dphidn_L * uD_val)
            
    return Ke, Fe

# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------






# Newton solver to get (u,v) in right element (for demo / tests only)
# ----------------------------------------------------------------------

def _inverse_map(mesh, elem_id, x_phys, tol=1e-10, maxiter=10):
    """Newton solver to find reference coordinates for a physical point."""
    ref = get_reference(mesh.element_type, mesh.poly_order)
    xi_ref = np.array([0.5, 0.5]) # Initial guess in center of ref element
    for _ in range(maxiter):
        x_curr = transform.x_mapping(mesh, elem_id, xi_ref)
        J = transform.jacobian(mesh, elem_id, xi_ref)
        if np.linalg.det(J) == 0:
            # Handle singular Jacobian if necessary
            break
        delta = np.linalg.solve(J, x_phys - x_curr)
        xi_ref += delta
        if np.linalg.norm(delta) < tol:
            break
    return xi_ref
