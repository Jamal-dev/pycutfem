from pathlib import Path

from pycutfem.jit.cpp_backend import cache as cpp_cache


def test_cpp_cache_regenerates_on_stale_source(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cpp"
    cache_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cpp_cache.CppKernelCache, "_CACHE_DIR", cache_dir)
    cache = cpp_cache.CppKernelCache()
    cache._ext_suffix = ".py"
    abi = cpp_cache.CODEGEN_ABI_CPP

    def fake_compile_extension(module_name, source_file, build_dir, include_dirs=None):
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
    module_name = f"_pycutfem_cpp_kernel_{cpp_cache.KernelCache._hash_ir(ir_sequence, None)}"
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
