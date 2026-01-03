import numpy as np
import pytest

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.expressions import Function, TestFunction
from pycutfem.ufl.measures import dx
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.compilers import FormCompiler


BACKENDS = ("python", "jit", "cpp")


def _build_dh():
    nodes, elements, edges, corners = structured_quad(
        Lx=1.0,
        Ly=1.0,
        nx=2,
        ny=2,
        poly_order=1,
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elements,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    return DofHandler(me, method="cg")


@pytest.mark.parametrize("backend", BACKENDS)
def test_coeff_cache_refresh_between_assemblies(backend):
    dh = _build_dh()
    u_k = Function(name="u_k", field_name="u", dof_handler=dh)
    v = TestFunction(name="v", field_name="u", dof_handler=dh)
    form = u_k * v * dx
    eq = Equation(None, form)

    compiler = FormCompiler(dh, backend=backend)

    u_k.nodal_values.fill(1.0)
    _, f1 = compiler.assemble(eq, bcs=[])
    u_k.nodal_values.fill(2.0)
    _, f2 = compiler.assemble(eq, bcs=[])

    assert np.max(np.abs(f1)) > 0.0
    assert np.allclose(f2, 2.0 * f1, rtol=1.0e-8, atol=1.0e-10)
