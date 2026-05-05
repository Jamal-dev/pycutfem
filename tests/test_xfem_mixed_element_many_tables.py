import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.meshgen import structured_triangles
from pycutfem.xfem.mixedelement import XFEMMixedElement


def test_xfem_mixed_element_eval_many_tables_are_widened():
    nodes, elems, edges, corners = structured_triangles(
        Lx=1.0,
        Ly=1.0,
        nx_quads=1,
        ny_quads=1,
        poly_order=1,
        offset=(0.0, 0.0),
    )
    mesh = Mesh(nodes, elems, edges, corners, element_type="tri", poly_order=1)
    me = MixedElement(mesh, {"u": 1})
    xme = XFEMMixedElement(me, enrich_kind_by_field={"u": "abs"})

    xi = np.asarray([0.2, 0.4, 0.1], float)
    eta = np.asarray([0.2, 0.1, 0.3], float)

    n0 = int(me._eval_scalar_basis("u", float(xi[0]), float(eta[0])).size)
    tab_basis = xme._eval_scalar_basis_many("u", xi, eta)
    tab_grad = xme._eval_scalar_grad_many("u", xi, eta)

    assert tab_basis.shape == (xi.size, 2 * n0)
    assert tab_grad.shape == (xi.size, 2 * n0, 2)
    assert np.allclose(tab_basis[:, n0:], 0.0)
    assert np.allclose(tab_grad[:, n0:, :], 0.0)

