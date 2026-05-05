from pathlib import Path

import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.jit.cache import KernelCache
from pycutfem.ufl.expressions import Constant, VectorFunction, grad, inner
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.linalg import spectral_positive_part_2x2_sym, sym
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


def _setup_single_q1_vector():
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
    mixed_element = MixedElement(mesh, field_specs={"ux": 1, "uy": 1})
    dof_handler = DofHandler(mixed_element, method="cg")
    return mesh, dof_handler


def _assemble_scalar(dof_handler, integral, *, backend: str) -> float:
    hooks = {type(integral.integrand): {"name": "val"}}
    res = assemble_form(
        Equation(None, integral),
        dof_handler=dof_handler,
        bcs=[],
        assembler_hooks=hooks,
        backend=backend,
    )
    val = np.asarray(res["val"])
    if val.ndim == 0:
        return float(val)
    return float(val.ravel()[0])


def _expected_positive_strain(mat: np.ndarray) -> np.ndarray:
    # For symmetric mat: A⁺ = Σ max(λ_i,0) n_i ⊗ n_i.
    w, v = np.linalg.eigh(mat)
    w_pos = np.maximum(w, 0.0)
    return (v * w_pos) @ v.T


def test_spectral_positive_part_matches_numpy_python_cpp():
    _, dof_handler = _setup_single_q1_vector()

    # Constant symmetric strain state with one positive and one negative eigenvalue.
    a, b, s = 0.2, -0.1, 0.03
    disp = VectorFunction("disp", ["ux", "uy"], dof_handler=dof_handler)
    disp.set_values_from_function(lambda x, y: np.array([a * x + s * y, s * x + b * y]))

    E = sym(grad(disp))
    E_plus, _, _, _, _ = spectral_positive_part_2x2_sym(E, eta_pos=1.0e-12, disc_reg=1.0e-16)

    expected = _expected_positive_strain(np.array([[a, s], [s, b]], dtype=float))
    diff = E_plus - Constant(expected)
    integral = inner(diff, diff) * dx(metadata={"q": 3})

    val_py = _assemble_scalar(dof_handler, integral, backend="python")
    val_cpp = _assemble_scalar(dof_handler, integral, backend="cpp")

    assert np.isclose(val_py, 0.0, atol=1.0e-11, rtol=1.0e-11)
    assert np.isclose(val_cpp, 0.0, atol=1.0e-11, rtol=1.0e-11)
    assert np.isclose(val_py, val_cpp, atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.parametrize("jit_backend", ["numba", "cpp"])
def test_spectral_positive_part_jit_backends(jit_backend, monkeypatch):
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

    _, dof_handler = _setup_single_q1_vector()

    a, b, s = 0.2, -0.1, 0.03
    disp = VectorFunction("disp", ["ux", "uy"], dof_handler=dof_handler)
    disp.set_values_from_function(lambda x, y: np.array([a * x + s * y, s * x + b * y]))

    E = sym(grad(disp))
    E_plus, _, _, _, _ = spectral_positive_part_2x2_sym(E, eta_pos=1.0e-12, disc_reg=1.0e-16)
    expected = _expected_positive_strain(np.array([[a, s], [s, b]], dtype=float))

    diff = E_plus - Constant(expected)
    integral = inner(diff, diff) * dx(metadata={"q": 3})
    val_jit = _assemble_scalar(dof_handler, integral, backend="jit")
    assert np.isclose(val_jit, 0.0, atol=1.0e-11, rtol=1.0e-11)
