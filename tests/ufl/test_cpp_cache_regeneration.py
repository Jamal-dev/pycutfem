import builtins
import sys
from pathlib import Path

import pytest

from pycutfem.jit.cpp_backend import cache as cpp_cache
from pycutfem.jit.cpp_backend.compiler import get_compile_mode_tag


@pytest.fixture(autouse=True)
def _clear_cpp_global_kernel_cache():
    cpp_cache.CppKernelCache._GLOBAL_IN_MEMORY_CACHE.clear()
    yield
    cpp_cache.CppKernelCache._GLOBAL_IN_MEMORY_CACHE.clear()


def test_cpp_cache_regenerates_on_stale_source(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cpp"
    cache_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cpp_cache.CppKernelCache, "_CACHE_DIR", cache_dir)
    cache = cpp_cache.CppKernelCache()
    cache._ext_suffix = ".py"
    abi = cpp_cache.CODEGEN_ABI_CPP

    def fake_compile_extension(module_name, source_file, build_dir, include_dirs=None, **_kwargs):
        mod_path = Path(build_dir) / f"{module_name}{cache._ext_suffix}"
        mod_path.write_text(
            f'CODEGEN_ABI = "{abi}"\n'
            "PARAM_ORDER = []\n"
            "ACTIVE_FIELDS = []\n\n"
            "def kernel(*args):\n"
            "    return None\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(cpp_cache, "compile_extension", fake_compile_extension)

    ir_sequence = []
    compile_tag = get_compile_mode_tag()
    module_name = f"_pycutfem_cpp_kernel_{cpp_cache.KernelCache._hash_ir(ir_sequence, None)}_{compile_tag}"
    source_file = cache_dir / f"{module_name}.cpp"
    source_file.write_text('CODEGEN_ABI") = "old-abi"', encoding="utf-8")

    class DummyCodegen:
        include_dirs = []
        active_fields = []

        def generate_source(self, ir_sequence, kernel_name, module_name=None):
            src = f'CODEGEN_ABI") = "{abi}"'
            return src, {}, []

        @staticmethod
        def kernel_export_name(name):
            return name

    cache.get_kernel(ir_sequence, DummyCodegen(), mesh_sig=None)

    assert abi in source_file.read_text(encoding="utf-8")


def test_cpp_cache_resolves_cache_dir_from_runtime_env(tmp_path, monkeypatch):
    root = tmp_path / "runtime_root"
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(root))
    monkeypatch.setattr(cpp_cache.CppKernelCache, "_CACHE_DIR", None)

    cache = cpp_cache.CppKernelCache()

    assert cache._cache_dir == (root / "cpp").resolve()


def test_cpp_cache_falls_back_to_temp_when_user_cache_is_unwritable(tmp_path, monkeypatch):
    xdg_cache = tmp_path / "xdg_cache"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.chmod(0o555)

    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir(parents=True, exist_ok=True)

    monkeypatch.delenv("PYCUTFEM_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cache))
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setattr(cpp_cache.CppKernelCache, "_CACHE_DIR", None)

    cache = cpp_cache.CppKernelCache()

    expected = (Path(cpp_cache.tempfile.gettempdir()) / "pycutfem_jit" / "cpp").resolve()
    assert cache._cache_dir == expected


def test_cpp_cache_reuses_loaded_module_across_cache_instances(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cpp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cpp_cache.CppKernelCache, "_CACHE_DIR", cache_dir)
    abi = cpp_cache.CODEGEN_ABI_CPP

    ir_sequence = []
    compile_tag = get_compile_mode_tag()
    module_name = f"_pycutfem_cpp_kernel_{cpp_cache.KernelCache._hash_ir(ir_sequence, None)}_{compile_tag}"

    class DummyCodegen:
        include_dirs = []
        active_fields = []

        def generate_source(self, ir_sequence, kernel_name, module_name=None):
            src = f'CODEGEN_ABI") = "{abi}"'
            return src, {}, []

        @staticmethod
        def kernel_export_name(name):
            return name

    def fake_compile_extension(module_name, source_file, build_dir, include_dirs=None, **_kwargs):
        mod_path = Path(build_dir) / f"{module_name}.py"
        mod_path.write_text(
            "import builtins\n"
            "_cnt = int(getattr(builtins, '_pycutfem_cpp_import_count', 0)) + 1\n"
            "builtins._pycutfem_cpp_import_count = _cnt\n"
            "if _cnt > 1:\n"
            "    raise RuntimeError('module reloaded')\n"
            f'CODEGEN_ABI = "{abi}"\n'
            "PARAM_ORDER = []\n"
            "ACTIVE_FIELDS = []\n\n"
            "def kernel(*args):\n"
            "    return None\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(cpp_cache, "compile_extension", fake_compile_extension)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    if hasattr(builtins, "_pycutfem_cpp_import_count"):
        delattr(builtins, "_pycutfem_cpp_import_count")

    cache_a = cpp_cache.CppKernelCache()
    cache_a._ext_suffix = ".py"
    cache_b = cpp_cache.CppKernelCache()
    cache_b._ext_suffix = ".py"

    cache_a.get_kernel(ir_sequence, DummyCodegen(), mesh_sig=None)
    cache_b.get_kernel(ir_sequence, DummyCodegen(), mesh_sig=None)

    assert getattr(builtins, "_pycutfem_cpp_import_count", 0) == 1
    delattr(builtins, "_pycutfem_cpp_import_count")


def test_cpp_cache_rebuilds_when_binary_digest_stamp_is_missing(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cpp"
    cache_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cpp_cache.CppKernelCache, "_CACHE_DIR", cache_dir)
    cache = cpp_cache.CppKernelCache()
    cache._ext_suffix = ".py"
    abi = cpp_cache.CODEGEN_ABI_CPP
    compile_calls = {"count": 0}

    def fake_compile_extension(module_name, source_file, build_dir, include_dirs=None, **_kwargs):
        compile_calls["count"] += 1
        mod_path = Path(build_dir) / f"{module_name}{cache._ext_suffix}"
        mod_path.write_text(
            f'CODEGEN_ABI = "{abi}"\n'
            "PARAM_ORDER = []\n"
            "ACTIVE_FIELDS = []\n\n"
            "def kernel(*args):\n"
            "    return None\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(cpp_cache, "compile_extension", fake_compile_extension)

    ir_sequence = []
    compile_tag = get_compile_mode_tag()
    module_name = f"_pycutfem_cpp_kernel_{cpp_cache.KernelCache._hash_ir(ir_sequence, None)}_{compile_tag}"
    source_file = cache_dir / f"{module_name}.cpp"
    source_file.write_text(f'CODEGEN_ABI") = "{abi}"', encoding="utf-8")
    built_module = cache_dir / f"{module_name}.py"
    built_module.write_text(
        f'CODEGEN_ABI = "{abi}"\n'
        "PARAM_ORDER = []\n"
        "ACTIVE_FIELDS = []\n\n"
        "def kernel(*args):\n"
        "    return None\n",
        encoding="utf-8",
    )

    class DummyCodegen:
        include_dirs = []
        active_fields = []

        def generate_source(self, ir_sequence, kernel_name, module_name=None):
            src = f'CODEGEN_ABI") = "{abi}"'
            return src, {}, []

        @staticmethod
        def kernel_export_name(name):
            return name

    cache.get_kernel(ir_sequence, DummyCodegen(), mesh_sig=None)

    assert compile_calls["count"] == 1
    assert (cache_dir / f"{module_name}.sha256").exists()


def test_cpp_cache_reuses_in_memory_kernel_across_cache_dir_changes(tmp_path, monkeypatch):
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    abi = cpp_cache.CODEGEN_ABI_CPP
    compile_calls = {"count": 0}

    class DummyCodegen:
        include_dirs = []
        active_fields = []

        def generate_source(self, ir_sequence, kernel_name, module_name=None):
            src = f'CODEGEN_ABI") = "{abi}"'
            return src, {}, []

        @staticmethod
        def kernel_export_name(name):
            return name

    def fake_compile_extension(module_name, source_file, build_dir, include_dirs=None, **_kwargs):
        compile_calls["count"] += 1
        mod_path = Path(build_dir) / f"{module_name}.py"
        mod_path.write_text(
            f'CODEGEN_ABI = "{abi}"\n'
            "PARAM_ORDER = []\n"
            "ACTIVE_FIELDS = []\n\n"
            "def kernel(*args):\n"
            "    return None\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(cpp_cache, "compile_extension", fake_compile_extension)
    monkeypatch.setattr(cpp_cache.CppKernelCache, "_CACHE_DIR", None)

    ir_sequence = []

    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(root_a))
    cache_a = cpp_cache.CppKernelCache()
    cache_a._ext_suffix = ".py"
    cache_a.get_kernel(ir_sequence, DummyCodegen(), mesh_sig=None)

    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(root_b))
    cache_b = cpp_cache.CppKernelCache()
    cache_b._ext_suffix = ".py"
    cache_b.get_kernel(ir_sequence, DummyCodegen(), mesh_sig=None)

    assert compile_calls["count"] == 1


def test_cpp_cache_retries_once_after_incomplete_binary_import(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cpp"
    cache_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cpp_cache.CppKernelCache, "_CACHE_DIR", cache_dir)
    cache = cpp_cache.CppKernelCache()
    cache._ext_suffix = ".py"
    abi = cpp_cache.CODEGEN_ABI_CPP
    compile_calls = {"count": 0}

    def fake_compile_extension(module_name, source_file, build_dir, include_dirs=None, **_kwargs):
        compile_calls["count"] += 1
        mod_path = Path(build_dir) / f"{module_name}{cache._ext_suffix}"
        mod_path.write_text(
            f'CODEGEN_ABI = "{abi}"\n'
            "PARAM_ORDER = []\n"
            "ACTIVE_FIELDS = []\n\n"
            "def kernel(*args):\n"
            "    return None\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(cpp_cache, "compile_extension", fake_compile_extension)

    ir_sequence = []

    class DummyCodegen:
        include_dirs = []
        active_fields = []

        def generate_source(self, ir_sequence, kernel_name, module_name=None):
            src = f'CODEGEN_ABI") = "{abi}"'
            return src, {}, []

        @staticmethod
        def kernel_export_name(name):
            return name

    import_attempts = {"count": 0}
    original_import = cpp_cache.CppKernelCache._import_module

    def flaky_import(modname, path, *, force_reload=False):
        import_attempts["count"] += 1
        if import_attempts["count"] == 1:
            raise ImportError(f"{path}: file too short")
        return original_import(modname, path, force_reload=force_reload)

    monkeypatch.setattr(cpp_cache.CppKernelCache, "_import_module", staticmethod(flaky_import))

    cache.get_kernel(ir_sequence, DummyCodegen(), mesh_sig=None)

    assert import_attempts["count"] == 2
    assert compile_calls["count"] == 2


def test_cpp_backend_disables_pybind11_subinterpreter_support():
    from pycutfem.jit.cpp_backend import compiler as cpp_compiler

    expected_flag = "-DPYBIND11_HAS_SUBINTERPRETER_SUPPORT=0"
    assert expected_flag in cpp_compiler._compile_args_for_mode("opt")
    assert expected_flag in cpp_compiler._compile_args_for_mode("fast")
