"""C++ helpers for precompute geometry (pybind11 extension)."""

from __future__ import annotations

import importlib.util
import os
import sysconfig
from pathlib import Path
from typing import Any, Tuple

from .compiler import compile_extension


def _cache_dir() -> Path:
    root = Path(os.environ.get("PYCUTFEM_CACHE_DIR", Path.home() / ".cache" / "pycutfem_jit")).expanduser().resolve()
    d = root / "cpp_precompute"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _import_module(modname: str, path: Path):
    spec = importlib.util.spec_from_file_location(modname, path)
    if not spec or not spec.loader:  # pragma: no cover
        raise ImportError(f"Could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


_MODULE = None


def module() -> Any:
    """Build/load the C++ precompute extension and return the imported module."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    cache = _cache_dir()
    modname = "_pycutfem_cpp_precompute_geom"
    built = cache / f"{modname}{_ext_suffix()}"
    src = Path(__file__).with_suffix(".cpp")
    if not built.exists():
        compile_extension(modname, src, cache)
    _MODULE = _import_module(modname, built)
    return _MODULE


def quad_jacobian_det_inv(coords, xi, eta, poly_order: int):
    """
    Return (J, detJ, J_inv) for a batch of Q1/Q2 quads.

    Parameters are NumPy arrays:
      coords: (nE, nLoc, 2) float64
      xi/eta: (nE, nQ) float64
      poly_order: 1 or 2
    """
    m = module()
    return m.quad_jacobian_det_inv(coords, xi, eta, int(poly_order))

