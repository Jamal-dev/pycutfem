"""Cache and compiler driver for C++ kernels."""

from __future__ import annotations

import importlib.util
import os
import sys
import sysconfig
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pycutfem.jit.cache import KernelCache
from .compiler import compile_extension, get_compile_mode_tag

# Bump when generated C++ changes in a way that requires recompilation of cached kernels.
# Bump when generated C++ changes or when helper semantics change in a way that
# requires recompilation of cached kernels (e.g. stride-aware accessors).
CODEGEN_ABI_CPP = "2026-02-14-cpp-v18-hdiv-facet-jinv"


class CppKernelCache:
    """
    Compile-once, reuse-many cache for C++/pybind11 kernels.

    The interface mirrors :class:`pycutfem.jit.cache.KernelCache`.
    """

    # Optional test hook; when None, cache dir is resolved from env per instance.
    _CACHE_DIR: Path | None = None

    def __init__(self) -> None:
        self.in_memory_cache: Dict[str, Tuple[Any, List[str]]] = {}
        self._cache_dir = self._resolve_cache_dir()
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        self._ext_suffix = suffix if suffix else ".so"

    @classmethod
    def _resolve_cache_dir(cls) -> Path:
        """Resolve cache dir at runtime so per-test env overrides are honored."""
        override = cls._CACHE_DIR
        if override is not None:
            cache_dir = Path(override).expanduser().resolve()
        else:
            cache_dir = (
                Path(
                    os.environ.get(
                        "PYCUTFEM_CACHE_DIR", Path.home() / ".cache" / "pycutfem_jit"
                    )
                )
                .expanduser()
                .resolve()
                / "cpp"
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    # ------------------------------------------------------------------
    def get_kernel(self, ir_sequence: list, codegen, mesh_sig=None):
        """
        Build (or load) a kernel for the given IR sequence and return
        (kernel_fn, param_order list, active_fields list).
        """
        ir_hash = KernelCache._hash_ir(ir_sequence, mesh_sig)
        compile_tag = get_compile_mode_tag()
        cache_key = f"{ir_hash}:{compile_tag}"

        if cache_key in self.in_memory_cache:
            return self.in_memory_cache[cache_key]

        module_name = f"_pycutfem_cpp_kernel_{ir_hash}_{compile_tag}"
        source_file = self._cache_dir / f"{module_name}.cpp"
        built_module = self._cache_dir / f"{module_name}{self._ext_suffix}"

        # Generate source if needed (or if stale ABI is detected).
        regenerate = True
        if source_file.exists():
            try:
                text = source_file.read_text(encoding="utf-8")
            except OSError:
                text = ""
            regenerate = f'CODEGEN_ABI") = "{CODEGEN_ABI_CPP}"' not in text

        if regenerate:
            src, _, param_order = codegen.generate_source(
                ir_sequence, "kernel", module_name=module_name
            )
            source_file.write_text(src, encoding="utf-8")
            built_module.unlink(missing_ok=True)
        else:
            param_order = None

        # Compile when the shared object is missing.
        if not built_module.exists():
            include_dirs = list(codegen.include_dirs)
            compile_extension(
                module_name,
                source_file,
                self._cache_dir,
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
                self._cache_dir,
                include_dirs=list(codegen.include_dirs),
            )
            module = self._import_module(module_name, built_module, force_reload=True)

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
        self.in_memory_cache[cache_key] = (kernel_fn, param_order, active_fields)
        return kernel_fn, param_order, active_fields

    # ------------------------------------------------------------------
    @staticmethod
    def _import_module(modname: str, path: Path, *, force_reload: bool = False):
        if force_reload:
            sys.modules.pop(modname, None)
        else:
            cached = sys.modules.get(modname)
            if cached is not None:
                return cached
        spec = importlib.util.spec_from_file_location(modname, path)
        if not spec or not spec.loader:  # pragma: no cover - safety net
            raise ImportError(f"Could not load spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        try:
            spec.loader.exec_module(mod)  # type: ignore[arg-type]
        except Exception:
            # Avoid keeping partially-initialized modules around.
            sys.modules.pop(modname, None)
            raise
        return mod
