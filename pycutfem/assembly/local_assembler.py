"""pycutfem.assembly.local_assembler
Basic scalar stiffness for uncuts (Tri P1, Quad Q1).
"""
import numpy as np
from pycutfem.integration import volume
from pycutfem.fem.reference import get_reference
from pycutfem.fem import transform

def stiffness_matrix(mesh, elem_id, *, quad_order=None):
    # Choose quadrature automatically: order+2 Gauss (quad) or Dunavant
    poly_ord = mesh.poly_order  # P1/Q1 default; will be overwritten below
    if quad_order is None:
        quad_order = poly_ord + 2

    pts, wts = volume(mesh.element_type, quad_order)
    ref = get_reference(mesh.element_type, poly_ord)
    n_loc = len(ref.shape(0, 0))

    Ke = np.zeros((n_loc, n_loc))
    for (xi, eta), w in zip(pts, wts):
        dN = ref.grad(xi, eta)               # (n_loc, 2) ∇ξ
        N = ref.shape(xi, eta)               # (n_loc,)
        J  = transform.jacobian(mesh, elem_id, (xi, eta))
        invJ = np.linalg.inv(J)           # J^{-1}
        grad = dN @ invJ                    # (n_loc, 2)
        detJ = np.linalg.det(J)
        Ke  += w * detJ * grad @ grad.T
    Fe = np.zeros(n_loc)  # No load term in this example
    return Ke, Fe
