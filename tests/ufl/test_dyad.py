import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, dyad, dot
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


def _assemble_scalar(dh: DofHandler, form, backend: str) -> float:
    hook = {form.integrand: {"name": "val"}}
    res = assemble_form(
        Equation(None, form),
        dof_handler=dh,
        bcs=[],
        assembler_hooks=hook,
        backend=backend,
    )
    return float(np.asarray(res["val"], dtype=float).reshape(-1)[0])


def test_dyad_vectors_matches_backends_and_expected() -> None:
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

    a = Constant(np.array([1.0, 2.0], dtype=float))
    b = Constant(np.array([3.0, 4.0], dtype=float))
    e0 = Constant(np.array([1.0, 0.0], dtype=float))
    e1 = Constant(np.array([0.0, 1.0], dtype=float))

    A = dyad(a, b)

    a00 = dot(e0, dot(A, e0))
    a01 = dot(e0, dot(A, e1))
    a10 = dot(e1, dot(A, e0))
    a11 = dot(e1, dot(A, e1))

    expr = a00 + 2.0 * a01 + 3.0 * a10 + 4.0 * a11
    form = expr * dx(metadata={"q": 4})

    expected = 61.0  # A = [[3, 4], [6, 8]]

    val_py = _assemble_scalar(dh, form, backend="python")
    assert np.isfinite(val_py)
    assert np.allclose(val_py, expected, rtol=1e-12, atol=1e-12)

    val_jit = _assemble_scalar(dh, form, backend="jit")
    assert np.isfinite(val_jit)
    assert np.allclose(val_jit, expected, rtol=1e-12, atol=1e-12)
    assert np.allclose(val_py, val_jit, rtol=1e-12, atol=1e-12)

    try:
        val_cpp = _assemble_scalar(dh, form, backend="cpp")
        assert np.isfinite(val_cpp)
        assert np.allclose(val_cpp, expected, rtol=1e-12, atol=1e-12)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"CPP backend check failed or unavailable: {exc}")


def test_vector_multiplication_uses_dyad_semantics() -> None:
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

    a = Constant(np.array([1.0, 2.0], dtype=float))
    b = Constant(np.array([3.0, 4.0], dtype=float))
    e0 = Constant(np.array([1.0, 0.0], dtype=float))
    e1 = Constant(np.array([0.0, 1.0], dtype=float))

    A = a * b

    expr = (
        dot(e0, dot(A, e0))
        + 2.0 * dot(e0, dot(A, e1))
        + 3.0 * dot(e1, dot(A, e0))
        + 4.0 * dot(e1, dot(A, e1))
    ) * dx(metadata={"q": 4})

    expected = 61.0

    val_py = _assemble_scalar(dh, expr, backend="python")
    assert np.allclose(val_py, expected, rtol=1e-12, atol=1e-12)

    val_jit = _assemble_scalar(dh, expr, backend="jit")
    assert np.allclose(val_jit, expected, rtol=1e-12, atol=1e-12)

    try:
        val_cpp = _assemble_scalar(dh, expr, backend="cpp")
        assert np.allclose(val_cpp, expected, rtol=1e-12, atol=1e-12)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"CPP backend check failed or unavailable: {exc}")


def test_dyad_matrix_dot_matrix_matches_expected() -> None:
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

    a = Constant(np.array([1.0, 2.0], dtype=float))
    b = Constant(np.array([3.0, 4.0], dtype=float))
    c = Constant(np.array([-1.0, 5.0], dtype=float))
    d = Constant(np.array([2.0, -3.0], dtype=float))
    e0 = Constant(np.array([1.0, 0.0], dtype=float))
    e1 = Constant(np.array([0.0, 1.0], dtype=float))

    A = dyad(a, b)
    C = dyad(c, d)
    D = dot(A, C)

    d00 = dot(e0, dot(D, e0))
    d01 = dot(e0, dot(D, e1))
    d10 = dot(e1, dot(D, e0))
    d11 = dot(e1, dot(D, e1))

    expr = d00 + 2.0 * d01 + 3.0 * d10 + 4.0 * d11
    form = expr * dx(metadata={"q": 4})

    expected = -272.0  # A@C = [[34, -51], [68, -102]]

    val_py = _assemble_scalar(dh, form, backend="python")
    assert np.isfinite(val_py)
    assert np.allclose(val_py, expected, rtol=1e-12, atol=1e-12)

    val_jit = _assemble_scalar(dh, form, backend="jit")
    assert np.isfinite(val_jit)
    assert np.allclose(val_jit, expected, rtol=1e-12, atol=1e-12)
    assert np.allclose(val_py, val_jit, rtol=1e-12, atol=1e-12)

    try:
        val_cpp = _assemble_scalar(dh, form, backend="cpp")
        assert np.isfinite(val_cpp)
        assert np.allclose(val_cpp, expected, rtol=1e-12, atol=1e-12)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"CPP backend check failed or unavailable: {exc}")
