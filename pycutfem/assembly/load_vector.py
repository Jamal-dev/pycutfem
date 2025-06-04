"""pycutfem.assembly.load_vector"""
import numpy as np
from pycutfem.integration import volume
from pycutfem.fem.reference import get_reference
from pycutfem.fem import transform

def element_load(mesh, elem_id, f, order=None):
    # Determine the polynomial order of the element from the mesh object
    poly_ord = getattr(mesh, "element_order", 1)
    # Get the reference element definition for the correct polynomial order
    ref = get_reference(mesh.element_type, poly_ord)
    n_loc = len(ref.shape(0, 0))
    # n_loc = 3 if mesh.element_type=='tri' else 4
    Fe = np.zeros(n_loc)
    order = poly_ord + 3 if order is None else order
    pts, wts = volume(mesh.element_type, order)
    for xi_eta, w in zip(pts, wts):
        N = np.asarray(ref.shape(*xi_eta)).ravel()
        x = transform.x_mapping(mesh, elem_id, xi_eta)
        J = transform.jacobian(mesh, elem_id, xi_eta)
        detJ = np.linalg.det(J)
        Fe += w*detJ * N * f(*x)
    return Fe
