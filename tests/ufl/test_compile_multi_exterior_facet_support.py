import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.jit import compile_multi
from pycutfem.ufl.expressions import Constant, FacetNormal, VectorTestFunction, VectorTrialFunction, dot, grad
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dS
from pycutfem.utils.meshgen import structured_quad


@pytest.mark.parametrize("jit_backend", ["cpp", "numba"])
def test_compile_multi_supports_exterior_facet_integrals(monkeypatch, jit_backend: str) -> None:
    """
    Regression: `compile_multi` must handle boundary (dS) integrals that involve
    test/trial functions.

    In particular, exterior-facet kernels require sided reference value tables
    (r00_*) even when `required_multi_indices` only reports derivative orders.
    """
    monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", str(jit_backend))

    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=3, ny=2, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    mesh.tag_boundary_edges(
        {
            "outlet": lambda x, y: np.isclose(x, 1.0),
        }
    )
    outlet = mesh.edge_bitset("outlet")
    assert outlet.cardinality() > 0

    me = MixedElement(mesh, field_specs={"ux": 1, "uy": 1})
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["ux", "uy"])
    u = VectorTrialFunction(space=V, dof_handler=dh)
    v = VectorTestFunction(space=V, dof_handler=dh)
    n = FacetNormal()
    mu = Constant(1.0, dim=0)
    a = mu * dot(dot(grad(u).T, n), v) * dS(defined_on=outlet, metadata={"q": 4})

    kernels = compile_multi(Equation(a, None), dof_handler=dh, mixed_element=me, backend="jit")
    assert kernels

