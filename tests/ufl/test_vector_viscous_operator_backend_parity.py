import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Laplacian, VectorTestFunction, VectorTrialFunction, div, grad, inner
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


@pytest.mark.parametrize("backend", ("jit", "cpp"))
def test_vector_viscous_operator_matches_python(backend: str):
    """
    Regression:
    Backends must agree with the Python assembler on the strong viscous-operator
    ingredients used by the residual-based fluid stabilization:
        Delta u + grad(div(u)).
    """
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=3, ny=3, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )

    me = MixedElement(mesh, field_specs={"ux": 2, "uy": 2})
    dh = DofHandler(me, method="cg")

    vel_space = FunctionSpace("velocity", ["ux", "uy"])
    u = VectorTrialFunction(vel_space, dof_handler=dh)
    v = VectorTestFunction(vel_space, dof_handler=dh)

    qdeg = 6
    strong_u = Laplacian(u) + grad(div(u))
    strong_v = Laplacian(v) + grad(div(v))
    a = inner(strong_u, strong_v) * dx(metadata={"q": qdeg})

    K_py, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend="python")
    K_b, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend=backend)

    A_py = np.asarray(K_py.toarray(), dtype=float)
    A_b = np.asarray(K_b.toarray(), dtype=float)

    diff = float(np.max(np.abs(A_py - A_b)))
    scale = max(float(np.max(np.abs(A_py))), 1.0e-14)
    rel = diff / scale

    assert diff < 1.0e-8
    assert rel < 1.0e-9
