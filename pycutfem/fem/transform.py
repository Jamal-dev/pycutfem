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
    nodes = mesh.nodes[mesh.elements[elem_id]]
    order = getattr(mesh, "element_order", 1)
    ref   = get_reference(mesh.element_type, order)
    N,_ = _shape_and_grad(ref, xi_eta)
    return N @ nodes                  # (2,)

def jacobian(mesh, elem_id, xi_eta):
    nodes = mesh.nodes[mesh.elements[elem_id]]
    order = getattr(mesh, "element_order", 1)
    ref   = get_reference(mesh.element_type, order)
    _, dN = _shape_and_grad(ref, xi_eta)
    return dN.T @ nodes

def det_jacobian(mesh, elem_id, xi_eta):
    return np.linalg.det(jacobian(mesh, elem_id, xi_eta))

def map_grad_scalar(mesh, elem_id, grad_ref, xi_eta):
    J = jacobian(mesh, elem_id, xi_eta)
    invJ = np.linalg.inv(J)
    return invJ.T @ grad_ref
