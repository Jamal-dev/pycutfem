from pathlib import Path

import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.jit.cache import KernelCache
from pycutfem.ufl.expressions import Function, heaviside, pos_part
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


def _setup_single_q1_scalar():
    nodes, elems, _, corners = structured_quad(
        1,
        1,
        nx=1,
        ny=1,
        poly_order=1,
        parallel=False,
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    mixed_element = MixedElement(mesh, field_specs={"a": 1})
    dof_handler = DofHandler(mixed_element, method="cg")
    return mesh, dof_handler


def _assemble_functional(dof_handler, integral, *, backend: str) -> float:
    hooks = {type(integral.integrand): {"name": "I"}}
    res = assemble_form(
        Equation(None, integral),
        dof_handler=dof_handler,
        bcs=[],
        assembler_hooks=hooks,
        backend=backend,
    )
    return float(np.asarray(res["I"]).reshape(()))


def test_positive_part_and_heaviside_python_cpp():
    _, dof_handler = _setup_single_q1_scalar()
    a = Function("a", "a", dof_handler=dof_handler)

    qdx = dx(metadata={"q": 2})

    # Negative value.
    a.nodal_values.fill(-0.3)
    I_pos_py = _assemble_functional(dof_handler, pos_part(a) * qdx, backend="python")
    I_pos_cpp = _assemble_functional(dof_handler, pos_part(a) * qdx, backend="cpp")
    I_h_py = _assemble_functional(dof_handler, heaviside(a) * qdx, backend="python")
    I_h_cpp = _assemble_functional(dof_handler, heaviside(a) * qdx, backend="cpp")
    assert np.isclose(I_pos_py, 0.0)
    assert np.isclose(I_pos_cpp, 0.0)
    assert np.isclose(I_h_py, 0.0)
    assert np.isclose(I_h_cpp, 0.0)

    # Positive value.
    a.nodal_values.fill(0.4)
    I_pos_py = _assemble_functional(dof_handler, pos_part(a) * qdx, backend="python")
    I_pos_cpp = _assemble_functional(dof_handler, pos_part(a) * qdx, backend="cpp")
    I_h_py = _assemble_functional(dof_handler, heaviside(a) * qdx, backend="python")
    I_h_cpp = _assemble_functional(dof_handler, heaviside(a) * qdx, backend="cpp")
    assert np.isclose(I_pos_py, 0.4, atol=1.0e-14, rtol=0.0)
    assert np.isclose(I_pos_cpp, 0.4, atol=1.0e-14, rtol=0.0)
    assert np.isclose(I_h_py, 1.0, atol=1.0e-14, rtol=0.0)
    assert np.isclose(I_h_cpp, 1.0, atol=1.0e-14, rtol=0.0)

    # Strict convention H(0)=0.
    a.nodal_values.fill(0.0)
    I_h_py = _assemble_functional(dof_handler, heaviside(a) * qdx, backend="python")
    I_h_cpp = _assemble_functional(dof_handler, heaviside(a) * qdx, backend="cpp")
    assert np.isclose(I_h_py, 0.0, atol=1.0e-14, rtol=0.0)
    assert np.isclose(I_h_cpp, 0.0, atol=1.0e-14, rtol=0.0)


@pytest.mark.parametrize("jit_backend", ["numba", "cpp"])
def test_positive_part_and_heaviside_jit_backends(jit_backend, monkeypatch):
    # Redirect kernel caches to a writable temp directory.
    cache_dir = Path(".pytest_cache") / "pycutfem_jit"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(cache_dir))
    KernelCache._CACHE_DIR = cache_dir
    try:
        from pycutfem.jit.cpp_backend.cache import CppKernelCache

        cpp_cache_dir = cache_dir / "cpp"
        cpp_cache_dir.mkdir(parents=True, exist_ok=True)
        CppKernelCache._CACHE_DIR = cpp_cache_dir
    except Exception:
        pass

    if jit_backend == "cpp":
        monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")
    else:
        monkeypatch.delenv("PYCUTFEM_JIT_BACKEND", raising=False)

    _, dof_handler = _setup_single_q1_scalar()
    a = Function("a", "a", dof_handler=dof_handler)
    qdx = dx(metadata={"q": 2})

    a.nodal_values.fill(0.4)
    I_pos = _assemble_functional(dof_handler, pos_part(a) * qdx, backend="jit")
    I_h = _assemble_functional(dof_handler, heaviside(a) * qdx, backend="jit")

    assert np.isclose(I_pos, 0.4, atol=1.0e-14, rtol=0.0)
    assert np.isclose(I_h, 1.0, atol=1.0e-14, rtol=0.0)

