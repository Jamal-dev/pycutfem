from __future__ import annotations

import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, TestFunction, grad, inner
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


def _available_backends() -> list[str]:
    out = ["python", "jit"]
    try:
        from pycutfem.jit.cpp_backend import compile_backend_cpp  # noqa: F401
    except Exception:
        return out
    out.append("cpp")
    return out


@pytest.mark.parametrize("backend", _available_backends())
@pytest.mark.parametrize("constant_on_left", [True, False])
def test_grad_scalar_constant_compiles_to_zero_for_all_backends(backend, constant_on_left, tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PYCUTFEM_CACHE_DIR",
        str(tmp_path / f"pycutfem_grad_constant_{backend}_{'lhs' if constant_on_left else 'rhs'}"),
    )

    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    v = TestFunction("u", dof_handler=dh)
    zero_grad = grad(Constant(3.0))
    basis_grad = grad(v)
    integrand = inner(zero_grad, basis_grad) if constant_on_left else inner(basis_grad, zero_grad)

    _, rhs = assemble_form(
        Equation(None, integrand * dx(metadata={"q": 4})),
        dof_handler=dh,
        bcs=[],
        backend=backend,
    )

    np.testing.assert_allclose(np.asarray(rhs, dtype=float).reshape(-1), 0.0, atol=1.0e-12, rtol=0.0)
