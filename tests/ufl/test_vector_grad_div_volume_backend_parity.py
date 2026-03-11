import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import VectorFunction, VectorTestFunction, VectorTrialFunction, div, grad, inner
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def test_vector_grad_div_volume_backend_parity(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))

    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    me = MixedElement(mesh, field_specs={"ux": 2, "uy": 2})
    dh = DofHandler(me, method="cg")

    space = FunctionSpace(name="u", field_names=["ux", "uy"], dim=1)
    u = VectorTrialFunction(space=space, dof_handler=dh)
    v = VectorTestFunction(space=space, dof_handler=dh)
    u_k = VectorFunction(name="u_k", field_names=["ux", "uy"], dof_handler=dh)
    u_k.set_values_from_function(
        lambda x, y: np.asarray([np.sin(1.2 * x) + 0.25 * y, np.cos(0.9 * y) - 0.15 * x], dtype=float)
    )

    a = inner(grad(div(u)), grad(div(v))) * dx(metadata={"q": 4})
    l = inner(grad(div(u_k)), grad(div(v))) * dx(metadata={"q": 4})

    outputs = {}
    for backend in ("python", "jit", "cpp"):
        K, F = assemble_form(Equation(a, l), dof_handler=dh, bcs=[], backend=backend)
        outputs[backend] = (K.tocsr().toarray(), np.asarray(F, dtype=float))

    A_ref, F_ref = outputs["python"]
    for backend in ("jit", "cpp"):
        A, F = outputs[backend]
        assert np.allclose(A, A_ref, rtol=1.0e-10, atol=1.0e-12), (
            f"grad(div(.)) matrix backend parity failed for {backend}: "
            f"max_abs={float(np.max(np.abs(A - A_ref))):.3e}"
        )
        assert np.allclose(F, F_ref, rtol=1.0e-10, atol=1.0e-12), (
            f"grad(div(.)) rhs backend parity failed for {backend}: "
            f"max_abs={float(np.max(np.abs(F - F_ref))):.3e}"
        )
