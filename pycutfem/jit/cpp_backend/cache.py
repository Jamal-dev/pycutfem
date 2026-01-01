"""Cache and compiler driver for C++ kernels."""

from __future__ import annotations

import importlib.util
import os
import sysconfig
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pycutfem.jit.cache import KernelCache
from .compiler import compile_extension

CODEGEN_ABI_CPP = "2025-03-05-cpp-active-order"


class CppKernelCache:
    """
    Compile-once, reuse-many cache for C++/pybind11 kernels.

    The interface mirrors :class:`pycutfem.jit.cache.KernelCache`.
    """

    _CACHE_DIR = (
        Path(
            os.environ.get("PYCUTFEM_CACHE_DIR", Path.home() / ".cache" / "pycutfem_jit")
        )
        .expanduser()
        .resolve()
        / "cpp"
    )
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def __init__(self) -> None:
        self.in_memory_cache: Dict[str, Tuple[Any, List[str]]] = {}
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        self._ext_suffix = suffix if suffix else ".so"

    # ------------------------------------------------------------------
    def get_kernel(self, ir_sequence: list, codegen, mesh_sig=None):
        """
        Build (or load) a kernel for the given IR sequence and return
        (kernel_fn, param_order list, active_fields list).
        """
        ir_hash = KernelCache._hash_ir(ir_sequence, mesh_sig)

        if ir_hash in self.in_memory_cache:
            return self.in_memory_cache[ir_hash]

        module_name = f"_pycutfem_cpp_kernel_{ir_hash}"
        source_file = self._CACHE_DIR / f"{module_name}.cpp"
        built_module = self._CACHE_DIR / f"{module_name}{self._ext_suffix}"

        # Generate source if needed.
        if not source_file.exists():
            src, _, param_order = codegen.generate_source(
                ir_sequence, "kernel", module_name=module_name
            )
            source_file.write_text(src, encoding="utf-8")
        else:
            param_order = None

        # Compile when the shared object is missing.
        if not built_module.exists():
            include_dirs = list(codegen.include_dirs)
            compile_extension(
                module_name,
                source_file,
                self._CACHE_DIR,
                include_dirs=include_dirs,
            )

        module = self._import_module(module_name, built_module)
        if getattr(module, "CODEGEN_ABI", None) != CODEGEN_ABI_CPP:
            # Stale ABI – rebuild and reload once.
            source_file.unlink(missing_ok=True)
            built_module.unlink(missing_ok=True)
            src, _, param_order = codegen.generate_source(
                ir_sequence, "kernel", module_name=module_name
            )
            source_file.write_text(src, encoding="utf-8")
            compile_extension(
                module_name,
                source_file,
                self._CACHE_DIR,
                include_dirs=list(codegen.include_dirs),
            )
            module = self._import_module(module_name, built_module)

        param_order = (
            list(getattr(module, "PARAM_ORDER"))
            if hasattr(module, "PARAM_ORDER")
            else param_order
            or []
        )
        active_fields = (
            list(getattr(module, "ACTIVE_FIELDS"))
            if hasattr(module, "ACTIVE_FIELDS")
            else list(getattr(codegen, "active_fields", []) or [])
        )

        kernel_fn = getattr(module, codegen.kernel_export_name("kernel"))
        self.in_memory_cache[ir_hash] = (kernel_fn, param_order, active_fields)
        return kernel_fn, param_order, active_fields

    # ------------------------------------------------------------------
    @staticmethod
    def _import_module(modname: str, path: Path):
        spec = importlib.util.spec_from_file_location(modname, path)
        if not spec or not spec.loader:  # pragma: no cover - safety net
            raise ImportError(f"Could not load spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[arg-type]
        return mod
