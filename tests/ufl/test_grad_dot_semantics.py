import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, VectorFunction, dot, grad
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


def test_grad_dot_semantics_matches_directional_derivative():
    """
    Regression test for tensor contraction semantics:

      dot(dot(grad(u), e_x), e_y) == ∂_x u_y

    For u = (0, x) we have ∂_x u_y = 1, so the integral should equal the domain area.
    """
    L, H = 2.2, 0.41
    nodes, elems, edges, corners = structured_quad(L, H, nx=4, ny=2, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"ux": 2, "uy": 2})
    dof_handler = DofHandler(me, method="cg")

    u = VectorFunction(name="u", field_names=["ux", "uy"], dof_handler=dof_handler, side="+")
    ux_coords = np.asarray(dof_handler.get_dof_coords("ux"), dtype=float)
    uy_coords = np.asarray(dof_handler.get_dof_coords("uy"), dtype=float)
    u.set_component_values(0, np.zeros(len(ux_coords), dtype=float))
    u.set_component_values(1, uy_coords[:, 0].astype(float))

    e_x = Constant(np.array([1.0, 0.0]), dim=1)
    e_y = Constant(np.array([0.0, 1.0]), dim=1)

    q = 4
    dΩ = dx(metadata={"q": q})
    one = Constant(1.0)

    area_form = one * dΩ
    deriv_form = dot(dot(grad(u), e_x), e_y) * dΩ

    area = float(
        np.asarray(
            assemble_form(
                Equation(None, area_form),
                dof_handler=dof_handler,
                bcs=[],
                assembler_hooks={area_form.integrand: {"name": "area"}},
                backend="python",
            )["area"],
            dtype=float,
        ).reshape(-1)[0]
    )
    deriv_int = float(
        np.asarray(
            assemble_form(
                Equation(None, deriv_form),
                dof_handler=dof_handler,
                bcs=[],
                assembler_hooks={deriv_form.integrand: {"name": "deriv"}},
                backend="python",
            )["deriv"],
            dtype=float,
        ).reshape(-1)[0]
    )

    assert abs(deriv_int - area) <= 1.0e-10

