from pathlib import Path

import pytest

import pycutfem.jit as jit_module
import pycutfem.jit.cpp_backend as cpp_backend_module
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.jit import compile_backend
from pycutfem.jit.cache import KernelCache
from pycutfem.jit.cpp_backend import compile_backend_cpp
from pycutfem.jit.cpp_backend import cache as cpp_cache
from pycutfem.ufl.expressions import Grad, Inner, TestFunction, TrialFunction
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


@pytest.fixture(autouse=True)
def _reset_kernel_caches():
    jit_module._KERNEL_CACHE_SINGLETON = None
    jit_module._KERNEL_CACHE_DIR_TOKEN = None
    cpp_backend_module._CPP_KERNEL_CACHE_SINGLETON = None
    cpp_backend_module._CPP_KERNEL_CACHE_DIR_TOKEN = None
    cpp_cache.CppKernelCache._GLOBAL_IN_MEMORY_CACHE.clear()
    yield
    jit_module._KERNEL_CACHE_SINGLETON = None
    jit_module._KERNEL_CACHE_DIR_TOKEN = None
    cpp_backend_module._CPP_KERNEL_CACHE_SINGLETON = None
    cpp_backend_module._CPP_KERNEL_CACHE_DIR_TOKEN = None
    cpp_cache.CppKernelCache._GLOBAL_IN_MEMORY_CACHE.clear()
    cpp_cache.CppKernelCache._CACHE_DIR = None


def _build_poisson_integral(n_fields: int):
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    specs = {"u": 1}
    for i in range(1, n_fields):
        specs[f"junk{i}"] = 1

    me = MixedElement(mesh, field_specs=specs)
    dh = DofHandler(me, method="cg")
    u = TrialFunction("u", "u", dh)
    v = TestFunction("u", "u", dh)
    return Inner(Grad(u), Grad(v)) * dx(metadata={"q": 2}), dh, me


def test_mixed_element_signature_for_fields_drops_inactive_layout() -> None:
    _, _, me_small = _build_poisson_integral(1)
    _, _, me_large = _build_poisson_integral(5)

    assert me_small.signature_for_fields(("u",)) == me_large.signature_for_fields(("u",))


def test_jit_cache_reuses_kernel_across_mixed_supersets(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "jit_cache"))

    compile_calls = {"count": 0}
    original_compile = KernelCache._compile_and_write

    def counting_compile(self, target, ir_seq, codegen):
        compile_calls["count"] += 1
        return original_compile(self, target, ir_seq, codegen)

    monkeypatch.setattr(KernelCache, "_compile_and_write", counting_compile)

    integral_small, dh_small, me_small = _build_poisson_integral(1)
    integral_large, dh_large, me_large = _build_poisson_integral(5)

    runner_small, _ = compile_backend(integral_small, dh_small, me_small)
    runner_large, _ = compile_backend(integral_large, dh_large, me_large)

    assert tuple(getattr(runner_small, "active_fields", ()) or ()) == ("u",)
    assert tuple(getattr(runner_large, "active_fields", ()) or ()) == ("u",)
    assert compile_calls["count"] == 1


def test_cpp_cache_reuses_kernel_across_mixed_supersets(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cpp_cache"
    monkeypatch.setattr(cpp_cache.CppKernelCache, "_CACHE_DIR", cache_dir)

    compile_calls = {"count": 0}
    abi = cpp_cache.CODEGEN_ABI_CPP

    def fake_compile_extension(module_name, source_file, build_dir, include_dirs=None, **_kwargs):
        del source_file, include_dirs
        compile_calls["count"] += 1
        mod_path = Path(build_dir) / f"{module_name}.py"
        mod_path.write_text(
            f'CODEGEN_ABI = "{abi}"\n'
            "PARAM_ORDER = []\n"
            'ACTIVE_FIELDS = ["u"]\n\n'
            "def kernel(*args):\n"
            "    return None\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(cpp_cache, "compile_extension", fake_compile_extension)

    cache = cpp_backend_module._get_cpp_kernel_cache()
    cache._ext_suffix = ".py"

    integral_small, dh_small, me_small = _build_poisson_integral(1)
    integral_large, dh_large, me_large = _build_poisson_integral(5)

    runner_small, _ = compile_backend_cpp(integral_small, dh_small, me_small)
    runner_large, _ = compile_backend_cpp(integral_large, dh_large, me_large)

    assert tuple(getattr(runner_small, "active_fields", ()) or ()) == ("u",)
    assert tuple(getattr(runner_large, "active_fields", ()) or ()) == ("u",)
    assert compile_calls["count"] == 1
