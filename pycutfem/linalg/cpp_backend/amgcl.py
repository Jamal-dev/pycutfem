from __future__ import annotations

import importlib.util
import os
import sysconfig
from pathlib import Path
from typing import Any

from pycutfem.jit.cpp_backend.cache import _looks_like_incomplete_binary_import, _module_build_lock
from pycutfem.jit.cpp_backend.compiler import compile_extension


AMGCL_CPP_ABI = "2026-04-22-amgcl-v5"
_MODULE: Any | None = None


def _cache_dir() -> Path:
    root = Path(os.environ.get("PYCUTFEM_CACHE_DIR", Path.home() / ".cache" / "pycutfem_jit")).expanduser().resolve()
    cache_dir = root / "cpp_linalg"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _import_module(modname: str, path: Path):
    spec = importlib.util.spec_from_file_location(modname, path)
    if not spec or not spec.loader:  # pragma: no cover
        raise ImportError(f"Could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


def _candidate_amgcl_include_dirs() -> list[Path]:
    raw_env = os.getenv("PYCUTFEM_AMGCL_INCLUDE_DIR", "").strip()
    candidates: list[Path] = []
    if raw_env:
        for chunk in raw_env.split(os.pathsep):
            chunk = chunk.strip()
            if chunk:
                candidates.append(Path(chunk).expanduser())
    candidates.extend(
        [
            Path("/usr/include"),
            Path("/usr/local/include"),
            Path("/home/bhatti/opt/Kratos/external_libraries"),
        ]
    )

    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if (path / "amgcl" / "amg.hpp").exists():
            out.append(path)
    return out


def module() -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    cache = _cache_dir()
    modname = f"_pycutfem_cpp_amgcl_{AMGCL_CPP_ABI.replace('-', '_')}"
    built = cache / f"{modname}{_ext_suffix()}"
    lock = cache / f"{modname}.lock"
    src = Path(__file__).with_name("amgcl_module.cpp")
    include_dirs = _candidate_amgcl_include_dirs()
    if not include_dirs:
        raise RuntimeError(
            "Could not locate AMGCL headers. Set PYCUTFEM_AMGCL_INCLUDE_DIR "
            "or provide a local AMGCL installation."
        )

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
