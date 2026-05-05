import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl import Equation, HdivTestFunction, TestFunction, TrialFunction, assemble_form, dot, dx, grad
from pycutfem.utils.meshgen import structured_quad


def _assemble_grad_trial_hdiv_test(backend: str):
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, {"v": ("RT", 0), "alpha": 1})
    dh = DofHandler(me, method="cg")

    w = HdivTestFunction("v")
    dalpha = TrialFunction("alpha", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)

    form = (0.25 * dot(grad(dalpha), w) + dalpha * alpha_test) * dx(metadata={"q": 6})
    A, _ = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend=backend)
    return A.toarray()


def test_cpp_grad_trial_hdiv_test_orientation_matches_python(monkeypatch, tmp_path):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "jit_cache_cpp"))
    monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")

    A_ref = _assemble_grad_trial_hdiv_test("python")
    A_cpp = _assemble_grad_trial_hdiv_test("cpp")

    np.testing.assert_allclose(A_cpp, A_ref, rtol=1.0e-12, atol=1.0e-12)
