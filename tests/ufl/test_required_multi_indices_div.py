import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import VectorTrialFunction, div
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.helpers import required_multi_indices
from pycutfem.utils.meshgen import structured_quad


def test_required_multi_indices_includes_div_first_derivatives():
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"u_x": 1, "u_y": 1})
    dh = DofHandler(me, method="cg")
    V = FunctionSpace(name="u", field_names=["u_x", "u_y"], dim=1)
    u = VectorTrialFunction(space=V, dof_handler=dh)

    req = required_multi_indices(div(u))
    assert (1, 0) in req
    assert (0, 1) in req

