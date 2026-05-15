from __future__ import annotations

import importlib.util
import os
import sysconfig
from pathlib import Path
from typing import Any

from pycutfem.jit.cpp_backend.cache import _looks_like_incomplete_binary_import, _module_build_lock
from pycutfem.jit.cpp_backend.compiler import compile_extension


ONLINE_GAUSS_NEWTON_CPP_ABI = "2026-05-15-mor-online-gauss-newton-v5"
_MODULE: Any | None = None


def _cache_dir() -> Path:
    root = Path(os.environ.get("PYCUTFEM_CACHE_DIR", Path.home() / ".cache" / "pycutfem_jit")).expanduser().resolve()
    cache_dir = root / "mor_cpp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _import_module(modname: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(modname, path)
    if not spec or not spec.loader:  # pragma: no cover
        raise ImportError(f"Could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


def module() -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    cache = _cache_dir()
    modname = f"_pycutfem_mor_online_gauss_newton_{ONLINE_GAUSS_NEWTON_CPP_ABI.replace('-', '_')}"
    built = cache / f"{modname}{_ext_suffix()}"
    lock = cache / f"{modname}.lock"
    src = Path(__file__).with_name("online_gauss_newton_module.cpp")
    include_dirs = [Path(__file__).parents[2] / "jit" / "cpp_backend"]

    with _module_build_lock(lock):
        if not built.exists():
            compile_extension(modname, src, cache, include_dirs=include_dirs)
        try:
            _MODULE = _import_module(modname, built)
        except Exception as exc:
            if not _looks_like_incomplete_binary_import(exc):
                raise
            built.unlink(missing_ok=True)
            compile_extension(modname, src, cache, include_dirs=include_dirs)
            _MODULE = _import_module(modname, built)
    return _MODULE
