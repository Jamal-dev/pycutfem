import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction, grad, inner
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


def _compiled_backends() -> list[str]:
    out = ["jit"]
    try:
        from pycutfem.jit.cpp_backend import compile_backend_cpp  # noqa: F401
    except Exception:
        return out
    out.append("cpp")
    return out


def _build_scalar_problem():
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"p": 1, "q": 1})
    dh = DofHandler(me, method="cg")

    p_trial = TrialFunction("p", dof_handler=dh)
    q_test = TestFunction("q", dof_handler=dh)
    p_k = Function(name="p_k", field_name="p", dof_handler=dh)

    coords = np.asarray(dh.get_dof_coords("p"), dtype=float)
    p_k.nodal_values[:] = coords[:, 0] + 2.0 * coords[:, 1]

    B = Constant(np.array([[2.0, -0.5], [1.0, 3.0]], dtype=float), dim=2)
    dΩ = dx(metadata={"q": 6})
    return dh, p_trial, q_test, p_k, B, dΩ


@pytest.mark.parametrize("backend", _compiled_backends())
def test_scalar_grad_matrix_bilinear_matches_python(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_jit_cache_{backend}"))

    dh, p_trial, q_test, _, B, dΩ = _build_scalar_problem()

    left_form = inner(B * grad(p_trial), grad(q_test)) * dΩ
    right_form = inner(grad(p_trial) * B, grad(q_test)) * dΩ

    K_left_py, _ = assemble_form(Equation(left_form, None), dof_handler=dh, bcs=[], backend="python")
    K_right_py, _ = assemble_form(Equation(right_form, None), dof_handler=dh, bcs=[], backend="python")
    K_left_backend, _ = assemble_form(Equation(left_form, None), dof_handler=dh, bcs=[], backend=backend)
    K_right_backend, _ = assemble_form(Equation(right_form, None), dof_handler=dh, bcs=[], backend=backend)

    np.testing.assert_allclose(K_left_backend.toarray(), K_left_py.toarray(), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(K_right_backend.toarray(), K_right_py.toarray(), rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("backend", _compiled_backends())
def test_scalar_grad_matrix_residual_matches_python(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_jit_cache_rhs_{backend}"))

    dh, _, q_test, p_k, B, dΩ = _build_scalar_problem()

    left_form = inner(B * grad(p_k), grad(q_test)) * dΩ
    right_form = inner(grad(p_k) * B, grad(q_test)) * dΩ

    _, rhs_left_py = assemble_form(Equation(None, left_form), dof_handler=dh, bcs=[], backend="python")
    _, rhs_right_py = assemble_form(Equation(None, right_form), dof_handler=dh, bcs=[], backend="python")
    _, rhs_left_backend = assemble_form(Equation(None, left_form), dof_handler=dh, bcs=[], backend=backend)
    _, rhs_right_backend = assemble_form(Equation(None, right_form), dof_handler=dh, bcs=[], backend=backend)

    np.testing.assert_allclose(
        np.asarray(rhs_left_backend, dtype=float).reshape(-1),
        np.asarray(rhs_left_py, dtype=float).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(rhs_right_backend, dtype=float).reshape(-1),
        np.asarray(rhs_right_py, dtype=float).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
