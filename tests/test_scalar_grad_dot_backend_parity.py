import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Function, TestFunction, VectorFunction, VectorTrialFunction, dot, grad
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


def _build_scalar_grad_problem():
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"vx": 1, "vy": 1, "a": 1, "q": 1})
    dh = DofHandler(me, method="cg")

    v_k = VectorFunction(name="v_k", field_names=["vx", "vy"], dof_handler=dh)
    a_k = Function(name="a_k", field_name="a", dof_handler=dh)

    vx_coords = np.asarray(dh.get_dof_coords("vx"), dtype=float)
    vy_coords = np.asarray(dh.get_dof_coords("vy"), dtype=float)
    a_coords = np.asarray(dh.get_dof_coords("a"), dtype=float)

    v_k.set_component_values(0, vx_coords[:, 0].astype(float))
    v_k.set_component_values(1, (2.0 * vy_coords[:, 1]).astype(float))
    a_k.nodal_values[:] = a_coords[:, 0] + 3.0 * a_coords[:, 1]

    q_test = TestFunction("q", dof_handler=dh)
    V = FunctionSpace("V", ["vx", "vy"], dim=1)
    v_trial = VectorTrialFunction(space=V, dof_handler=dh)
    return dh, v_k, a_k, q_test, v_trial


def _extract_named_scalar(result, key: str) -> float:
    return float(np.asarray(result[key], dtype=float).reshape(-1)[0])


@pytest.mark.parametrize("backend", _compiled_backends())
def test_scalar_grad_value_contractions_match_python_and_exact_value(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_jit_cache_{backend}"))

    dh, v_k, a_k, _, _ = _build_scalar_grad_problem()
    dx_q = dx(metadata={"q": 6})

    left_form = dot(v_k, grad(a_k)) * dx_q
    right_form = dot(grad(a_k), v_k) * dx_q

    left_hooks = {left_form.integrand: {"name": "left"}}
    right_hooks = {right_form.integrand: {"name": "right"}}

    left_py = assemble_form(
        Equation(None, left_form),
        dof_handler=dh,
        bcs=[],
        assembler_hooks=left_hooks,
        backend="python",
    )
    right_py = assemble_form(
        Equation(None, right_form),
        dof_handler=dh,
        bcs=[],
        assembler_hooks=right_hooks,
        backend="python",
    )
    left_backend = assemble_form(
        Equation(None, left_form),
        dof_handler=dh,
        bcs=[],
        assembler_hooks=left_hooks,
        backend=backend,
    )
    right_backend = assemble_form(
        Equation(None, right_form),
        dof_handler=dh,
        bcs=[],
        assembler_hooks=right_hooks,
        backend=backend,
    )

    val_left_py = _extract_named_scalar(left_py, "left")
    val_right_py = _extract_named_scalar(right_py, "right")
    val_left_backend = _extract_named_scalar(left_backend, "left")
    val_right_backend = _extract_named_scalar(right_backend, "right")

    assert abs(val_left_py - 3.5) < 1.0e-12
    assert abs(val_right_py - 3.5) < 1.0e-12
    assert abs(val_left_backend - val_left_py) < 1.0e-12
    assert abs(val_right_backend - val_right_py) < 1.0e-12
    assert abs(val_left_backend - val_right_backend) < 1.0e-12


@pytest.mark.parametrize("backend", _compiled_backends())
def test_scalar_grad_trial_test_contractions_match_python(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_jit_cache_{backend}"))

    dh, _, a_k, q_test, v_trial = _build_scalar_grad_problem()
    dx_q = dx(metadata={"q": 6})

    left_form = q_test * dot(v_trial, grad(a_k)) * dx_q
    right_form = q_test * dot(grad(a_k), v_trial) * dx_q

    K_left_py, _ = assemble_form(Equation(left_form, None), dof_handler=dh, bcs=[], backend="python")
    K_right_py, _ = assemble_form(Equation(right_form, None), dof_handler=dh, bcs=[], backend="python")
    K_left_backend, _ = assemble_form(Equation(left_form, None), dof_handler=dh, bcs=[], backend=backend)
    K_right_backend, _ = assemble_form(Equation(right_form, None), dof_handler=dh, bcs=[], backend=backend)

    A_left_py = K_left_py.toarray()
    A_right_py = K_right_py.toarray()
    A_left_backend = K_left_backend.toarray()
    A_right_backend = K_right_backend.toarray()

    np.testing.assert_allclose(A_left_py, A_right_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(A_left_backend, A_left_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(A_right_backend, A_right_py, rtol=1.0e-12, atol=1.0e-12)
