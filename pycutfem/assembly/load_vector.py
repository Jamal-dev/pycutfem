"""pycutfem.assembly.load_vector"""
import numpy as np
from pycutfem.integration import volume
from pycutfem.fem.reference import get_reference
from pycutfem.fem import transform

__all__ = ["cg_element_load", "dg_element_load"]

def cg_element_load(mesh, elem_id, rhs, *, poly_order, quad_order=None):
    if quad_order is None:
        quad_order = poly_order + 2
    pts, wts = volume(mesh.element_type, quad_order)
    ref = get_reference(mesh.element_type, poly_order)
    Fe = np.zeros(len(ref.shape(0, 0)))
    for (xi, eta), w in zip(pts, wts):
        N = ref.shape(xi, eta)
        detJ = abs(np.linalg.det(transform.jacobian(mesh, elem_id, (xi, eta))))
        x = transform.x_mapping(mesh, elem_id, (xi, eta))
        Fe += w * detJ * N * rhs(*x)
    return Fe

def dg_element_load(mesh, elem_id, rhs, *, poly_order, quad_order=None):
    # identical integral, just a separate helper
    return cg_element_load(mesh, elem_id, rhs,
                           poly_order=poly_order,
                           quad_order=quad_order)
