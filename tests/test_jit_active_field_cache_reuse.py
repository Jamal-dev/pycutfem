from pathlib import Path
import ctypes
import sys

import numpy as np
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
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Constant, Grad, Inner, TestFunction, TrialFunction
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


def test_cpp_kernel_exposes_native_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "cpp_native_metadata"))

    integral, dh, me = _build_poisson_integral(1)
    runner, _ = compile_backend_cpp(integral, dh, me)
    module = sys.modules[getattr(runner.kernel, "__module__")]

    assert getattr(module, "NATIVE_KERNEL_ABI") == "2026-05-15-native-kernel-v1"
    assert getattr(module, "NATIVE_FORM_RANK") == 2
    assert getattr(module, "NATIVE_ON_FACET") is False
    assert getattr(module, "NATIVE_FUNCTIONAL_DIM") == 1
    assert getattr(module, "NATIVE_HAS_RAW_ENTRYPOINT") is True
    assert getattr(module, "NATIVE_KERNEL_METADATA") is not None
    assert runner.native_kernel_abi == "2026-05-15-native-kernel-v1"
    assert runner.native_kernel_metadata is getattr(module, "NATIVE_KERNEL_METADATA")

    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    ptr = ctypes.pythonapi.PyCapsule_GetPointer(
        getattr(module, "NATIVE_KERNEL_METADATA"),
        b"pycutfem.native_kernel_metadata.v1",
    )

    class NativeKernelMetadata(ctypes.Structure):
        _fields_ = [
            ("abi", ctypes.c_char_p),
            ("name", ctypes.c_char_p),
            ("form_rank", ctypes.c_int32),
            ("on_facet", ctypes.c_bool),
            ("parameter_names", ctypes.c_void_p),
            ("parameter_count", ctypes.c_size_t),
            ("active_fields", ctypes.c_void_p),
            ("active_field_count", ctypes.c_size_t),
            ("functional_dim", ctypes.c_int64),
            ("entrypoint", ctypes.c_void_p),
        ]

    meta = NativeKernelMetadata.from_address(ptr)
    assert meta.abi == b"2026-05-15-native-kernel-v1"
    assert meta.form_rank == 2
    assert meta.functional_dim == 1
    assert meta.entrypoint


def test_cpp_kernel_native_entrypoint_matches_pybind_call(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "cpp_native_entrypoint"))

    _integral, dh, _me = _build_poisson_integral(1)
    u = TrialFunction("u", "u", dh)
    v = TestFunction("u", "u", dh)
    integral = (Inner(Grad(u), Grad(v)) + Constant(3.0) * u * v) * dx(metadata={"q": 2})
    element_ids = np.arange(int(dh.mixed_element.mesh.n_elements), dtype=np.int32)
    compiler = FormCompiler(dh, quadrature_order=2, backend="cpp")
    runner, current_funcs, static_args, _gdofs_map = compiler._prepare_volume_jit_kernel(
        integral,
        element_ids=element_ids,
        full_local_layout=True,
    )
    K_py, F_py, J_py = runner(current_funcs, static_args)

    module = sys.modules[getattr(runner.kernel, "__module__")]
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    ptr = ctypes.pythonapi.PyCapsule_GetPointer(
        getattr(module, "NATIVE_KERNEL_METADATA"),
        b"pycutfem.native_kernel_metadata.v1",
    )

    class NativeKernelMetadata(ctypes.Structure):
        _fields_ = [
            ("abi", ctypes.c_char_p),
            ("name", ctypes.c_char_p),
            ("form_rank", ctypes.c_int32),
            ("on_facet", ctypes.c_bool),
            ("parameter_names", ctypes.c_void_p),
            ("parameter_count", ctypes.c_size_t),
            ("active_fields", ctypes.c_void_p),
            ("active_field_count", ctypes.c_size_t),
            ("functional_dim", ctypes.c_int64),
            ("entrypoint", ctypes.c_void_p),
        ]

    class NativeArrayView(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.c_void_p),
            ("dtype", ctypes.c_int32),
            ("ndim", ctypes.c_int32),
            ("shape", ctypes.c_int64 * 6),
            ("strides", ctypes.c_int64 * 6),
        ]

    class KernelStaticArgs(ctypes.Structure):
        _fields_ = [
            ("arrays", ctypes.POINTER(NativeArrayView)),
            ("names", ctypes.POINTER(ctypes.c_char_p)),
            ("count", ctypes.c_size_t),
        ]

    class KernelMutableArgs(ctypes.Structure):
        _fields_ = [
            ("arrays", ctypes.POINTER(NativeArrayView)),
            ("names", ctypes.POINTER(ctypes.c_char_p)),
            ("count", ctypes.c_size_t),
        ]

    class KernelOutputs(ctypes.Structure):
        _fields_ = [
            ("K", ctypes.c_void_p),
            ("F", ctypes.c_void_p),
            ("J", ctypes.c_void_p),
            ("n_entities", ctypes.c_int64),
            ("n_local_dofs", ctypes.c_int64),
            ("functional_dim", ctypes.c_int64),
            ("form_rank", ctypes.c_int32),
        ]

    meta = NativeKernelMetadata.from_address(ptr)
    assert meta.entrypoint

    def _dtype_id(arr: np.ndarray) -> int:
        if arr.dtype == np.float64:
            return 1
        if arr.dtype == np.int32:
            return 2
        if arr.dtype == np.uint8:
            return 3
        raise AssertionError(f"unsupported native test dtype {arr.dtype}")

    arrays = []
    keepalive = []
    for name in runner.param_order:
        arr = np.ascontiguousarray(static_args[name])
        if name == "gdofs_map" or name in {"owner_id", "owner_pos_id", "owner_neg_id", "pos_eids", "neg_eids", "qstate_owner_id"} or name.startswith(("pos_map", "neg_map")):
            arr = np.ascontiguousarray(arr, dtype=np.int32)
        elif name.startswith("domain_flag_"):
            arr = np.ascontiguousarray(arr, dtype=np.uint8)
        else:
            arr = np.ascontiguousarray(arr, dtype=np.float64)
        keepalive.append(arr)
        shape = (ctypes.c_int64 * 6)(*([0] * 6))
        strides = (ctypes.c_int64 * 6)(*([0] * 6))
        for axis, value in enumerate(arr.shape):
            shape[axis] = int(value)
        for axis, value in enumerate(arr.strides):
            strides[axis] = int(value)
        arrays.append(
            NativeArrayView(
                ctypes.c_void_p(arr.ctypes.data),
                _dtype_id(arr),
                int(arr.ndim),
                shape,
                strides,
            )
        )

    native_arrays = (NativeArrayView * len(arrays))(*arrays)
    native_names = (ctypes.c_char_p * len(arrays))(
        *[str(name).encode("utf-8") for name in runner.param_order]
    )
    static = KernelStaticArgs(native_arrays, native_names, len(arrays))
    mutable = KernelMutableArgs(None, None, 0)

    n_elem = int(static_args["qp_phys"].shape[0])
    n_union = int(static_args["gdofs_map"].shape[1])
    K_native = np.zeros((n_elem, n_union, n_union), dtype=np.float64)
    F_native = np.zeros((n_elem, n_union), dtype=np.float64)
    J_native = np.zeros((n_elem, int(meta.functional_dim)), dtype=np.float64)
    outputs = KernelOutputs(
        ctypes.c_void_p(K_native.ctypes.data),
        ctypes.c_void_p(F_native.ctypes.data),
        ctypes.c_void_p(J_native.ctypes.data),
        0,
        0,
        0,
        0,
    )

    entrypoint_type = ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(KernelStaticArgs),
        ctypes.POINTER(KernelMutableArgs),
        ctypes.POINTER(KernelOutputs),
    )
    entrypoint = entrypoint_type(meta.entrypoint)
    entrypoint(ctypes.byref(static), ctypes.byref(mutable), ctypes.byref(outputs))

    np.testing.assert_allclose(K_native, np.asarray(K_py, dtype=float), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(F_native, np.asarray(F_py, dtype=float), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(J_native, np.asarray(J_py, dtype=float).reshape(J_native.shape), rtol=1.0e-12, atol=1.0e-12)
    assert outputs.n_entities == n_elem
    assert outputs.n_local_dofs == n_union
    assert outputs.functional_dim == int(meta.functional_dim)
    assert outputs.form_rank == 2
