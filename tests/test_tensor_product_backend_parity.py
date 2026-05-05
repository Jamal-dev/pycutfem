import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import (
    Function,
    TestFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    grad,
)
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _compiled_backends() -> list[str]:
    out = ["jit"]
    try:
        from pycutfem.jit.cpp_backend import compile_backend_cpp  # noqa: F401
    except Exception:
        return out
    out.append("cpp")
    return out


def _build_problem():
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "q": 1})
    dh = DofHandler(me, method="cg")

    velocity = FunctionSpace("u", ["ux", "uy"], dim=1)
    du = VectorTrialFunction(space=velocity, dof_handler=dh)
    v = VectorTestFunction(space=velocity, dof_handler=dh)
    q = TestFunction("q", dof_handler=dh)
    u_k = VectorFunction("u_k", ["ux", "uy"], dof_handler=dh)
    q_k = Function("q_k", "q", dof_handler=dh)

    u_k.set_values_from_function(lambda x, y: np.array([x + 2.0 * y, -0.5 * x + y], dtype=float))
    q_coords = np.asarray(dh.get_dof_coords("q"), dtype=float)
    q_k.nodal_values[:] = 1.0 + q_coords[:, 0] - 0.25 * q_coords[:, 1]

    return dh, du, v, q, u_k, q_k


@pytest.mark.parametrize("backend", _compiled_backends())
def test_vector_dyad_component_bilinear_matches_python(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_tensor_dyad_{backend}"))

    dh, du, v, _q, u_k, _q_k = _build_problem()
    dΩ = dx(metadata={"q": 6})

    form = ((u_k * du)[0, 1]) * v[0] * dΩ

    K_py, _ = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend="python")
    K_backend, _ = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend=backend)

    np.testing.assert_allclose(K_backend.toarray(), K_py.toarray(), rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("backend", _compiled_backends())
def test_vector_tensor_component_bilinear_matches_python(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_tensor_grad_{backend}"))

    dh, du, _v, q, u_k, _q_k = _build_problem()
    dΩ = dx(metadata={"q": 6})

    form = ((u_k * grad(du))[1, 0, 1]) * q * dΩ

    K_py, _ = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend="python")
    K_backend, _ = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend=backend)

    np.testing.assert_allclose(K_backend.toarray(), K_py.toarray(), rtol=1.0e-12, atol=1.0e-12)
