"""pycutfem.fem.transform
Reference → physical mapping for linear elements.
"""
import numpy as np
from pycutfem.fem.reference import get_reference

def _shape_and_grad(ref, xi_eta):
    xi,eta=xi_eta
    N = ref.shape(xi,eta)
    N = np.asarray(N).ravel()          # (n_loc,)
    dN = np.asarray(ref.grad(xi,eta))  # (n_loc,2)
    return N, dN

def x_mapping(mesh, elem_id, xi_eta):
    nodes = mesh.nodes[mesh.elements_connectivity[elem_id]]
    poly_order = mesh.poly_order
    ref   = get_reference(mesh.element_type, poly_order)
    N,_ = _shape_and_grad(ref, xi_eta)
    return N @ nodes                  # (2,)

def jacobian(mesh, elem_id, xi_eta):
    nodes = mesh.nodes[mesh.elements_connectivity[elem_id]]
    poly_order = mesh.poly_order
    ref   = get_reference(mesh.element_type, poly_order)
    _, dN = _shape_and_grad(ref, xi_eta)
    return dN.T @ nodes

def det_jacobian(mesh, elem_id, xi_eta):
    return np.linalg.det(jacobian(mesh, elem_id, xi_eta))

def map_grad_scalar(mesh, elem_id, grad_ref, xi_eta):
    J = jacobian(mesh, elem_id, xi_eta)
    invJ = np.linalg.inv(J)
    return invJ.T @ grad_ref

def inv_jac_T(mesh, elem_id, xi_eta):
    J = jacobian(mesh, elem_id, xi_eta)
    return np.linalg.inv(J).T

def inverse_mapping(mesh, elem_id, x, tol=1e-10, maxiter=50):
    ref = get_reference(mesh.element_type, mesh.poly_order)
    # Use centroid as initial guess: (0,0) for quad, (1/3,1/3) for tri
    xi = np.array([0.0, 0.0]) if mesh.element_type == 'quad' else np.array([1/3, 1/3])
    for iter in range(maxiter):
        X = x_mapping(mesh, elem_id, xi)
        J = jacobian(mesh, elem_id, xi)
        try:
            delta = np.linalg.solve(J, x - X)
        except np.linalg.LinAlgError:
            raise ValueError(f"Jacobian singular at iteration {iter} for elem {elem_id}, x={x}")
        xi += delta
        if np.linalg.norm(delta) < tol:
            break
    else:
        raise ValueError(f"Inverse mapping did not converge after {maxiter} iterations for elem {elem_id}, x={x}, residual={np.linalg.norm(x - X)}")
    return xi

@staticmethod
def jacobian_1d(mesh, elem_id: int, ref_coords: tuple, local_edge_idx: int) -> float:
    """
    Computes the 1D Jacobian for a line integral (ratio of physical edge
    length to reference edge length).
    """
    J = jacobian(mesh, elem_id, ref_coords)

    if mesh.element_type == 'quad':
        # Tangent vectors on the reference quad edges [-1,1]x[-1,1]
        t_ref = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)[local_edge_idx]
    else: # tri
        # Tangent vectors on the reference triangle edges
        t_ref = np.array([[1, 0], [-1, 1], [0, -1]], dtype=float)[local_edge_idx]

    return np.linalg.norm(J @ t_ref)
