import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, FacetNormal, VectorTestFunction, VectorTrialFunction, dot, grad
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dS
from pycutfem.utils.meshgen import structured_quad


@pytest.mark.parametrize("jit_backend", ["cpp", "numba"])
def test_dfg_donothing_outlet_term_backend_parity(monkeypatch, jit_backend: str) -> None:
    """
    DFG benchmark outflow uses the do-nothing condition:

        ν ∂_n u - p n = 0  on Γ_out.

    With the symmetric interior form 2ν ε(u):ε(v), the corresponding weak form
    is obtained by adding the transpose correction on Γ_out:

        + ⟨ ν (∇uᵀ·n), v ⟩_{Γ_out}.

    This test checks python/jit/cpp backend parity for the boundary kernel
    `dot(dot(grad(u).T, n), v)` assembled on a tagged outlet boundary.
    """
    monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", str(jit_backend))

    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=4, ny=3, poly_order=1)
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
            "inlet": lambda x, y: np.isclose(x, 0.0),
            "outlet": lambda x, y: np.isclose(x, 1.0),
            "walls": lambda x, y: np.isclose(y, 0.0) | np.isclose(y, 1.0),
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
    dS_out = dS(defined_on=outlet, metadata={"q": 4})
    a = mu * dot(dot(grad(u).T, n), v) * dS_out

    K_py, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend="python")
    K_jit, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend="jit")
    K_cpp, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend="cpp")

    A_py = K_py.tocsr().toarray()
    A_jit = K_jit.tocsr().toarray()
    A_cpp = K_cpp.tocsr().toarray()

    assert np.isfinite(A_jit).all()
    assert float(np.max(np.abs(A_jit))) > 1.0e-12
    assert np.allclose(A_py, A_jit, rtol=1.0e-10, atol=1.0e-12)
    assert np.allclose(A_cpp, A_jit, rtol=1.0e-10, atol=1.0e-12)

