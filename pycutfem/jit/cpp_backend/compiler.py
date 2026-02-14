"""Utility for compiling generated C++ kernels via pybind11."""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path
from typing import Iterable, Sequence


def _fast_compile_enabled() -> bool:
    fast_env = os.getenv("PYCUTFEM_CPP_FAST_COMPILE", "").strip().lower()
    under_pytest = bool(os.getenv("PYTEST_CURRENT_TEST", ""))
    return (fast_env in {"1", "true", "yes"}) or (under_pytest and fast_env not in {"0", "false", "no"})


def get_compile_mode_tag() -> str:
    """
    Return a stable tag describing the compile mode for cache keying.

    NOTE: The C++ cache key must include this tag to avoid reusing -O0 test
    builds in normal runs (or vice versa).
    """
    return "fast" if _fast_compile_enabled() else "opt"


def _default_compile_args() -> list[str]:
    """Baseline compiler flags for performant builds."""
    # NOTE: C++ kernel compilation dominates many small test cases and developer
    # workflows. Allow opting into a fast (lower-optimization) build without
    # changing runtime code paths.
    #
    # - `PYCUTFEM_CPP_FAST_COMPILE=1` forces `-O0` and disables `-march=native`
    # - When running under pytest, default to fast compile to keep the suite
    #   responsive (can be overridden by explicitly setting PYCUTFEM_CPP_FAST_COMPILE=0).
    fast_compile = _fast_compile_enabled()

    args = ["-O0" if fast_compile else "-O3", "-std=c++17"]
    # Prefer native tuning when available (skip in fast-compile mode to reduce build time).
    if not fast_compile:
        args.append("-march=native")
    # Enable OpenMP if the toolchain supports it.
    args.append("-fopenmp")
    return args


def _default_link_args() -> list[str]:
    args = ["-fopenmp"]
    return args


def compile_extension(
    module_name: str,
    source_path: Path,
    build_dir: Path,
    *,
    include_dirs: Iterable[Path] | None = None,
    extra_compile_args: Sequence[str] | None = None,
    extra_link_args: Sequence[str] | None = None,
) -> Path:
    """
    Compile a single C++ translation unit into a Python extension module.

    Parameters
    ----------
    module_name:
        Name of the extension module to build (without suffix).
    source_path:
        Path to the generated .cpp file.
    build_dir:
        Directory where build artifacts (.so/.pyd) should be emitted.
    include_dirs:
        Additional include directories (e.g., Eigen headers).
    extra_compile_args, extra_link_args:
        Optional overrides/augmentations of the default flag sets.

    Returns
    -------
    Path to the built extension (.so/.pyd).
    """
    from setuptools import Extension, setup
    from setuptools.command.build_ext import build_ext

    try:
        import pybind11
        import numpy
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError(
            "pybind11 and numpy are required to build the C++ backend."
        ) from exc

    build_dir.mkdir(parents=True, exist_ok=True)
    tmp_build = build_dir / "build_temp"
    tmp_build.mkdir(parents=True, exist_ok=True)

    include_dirs = list(include_dirs or [])
    eigen_dir = Path(os.environ.get("EIGEN_INCLUDE_DIR", "/usr/include/eigen3"))
    if eigen_dir.exists():
        include_dirs.append(eigen_dir)
    else:
        conda_prefix = Path(os.environ.get("CONDA_PREFIX", ""))
        candidate = conda_prefix / "include" / "eigen3"
        if candidate.exists():
            include_dirs.append(candidate)
    include_dirs.extend([Path(pybind11.get_include()), Path(numpy.get_include())])

    compile_args = list(_default_compile_args())
    link_args = list(_default_link_args())
    if extra_compile_args:
        compile_args.extend(extra_compile_args)
    if extra_link_args:
        link_args.extend(extra_link_args)

    ext = Extension(
        module_name,
        sources=[str(source_path)],
        include_dirs=[str(p) for p in include_dirs],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )

    class _BuildExt(build_ext):  # type: ignore[misc]
        def build_extension(self, ext):  # pragma: no cover - invoked by setuptools
            super().build_extension(ext)

    # Run an isolated build into the cache directory.
    setup(
        name=module_name,
        ext_modules=[ext],
        cmdclass={"build_ext": _BuildExt},
        script_args=[
            "build_ext",
            "--build-temp",
            str(tmp_build),
            "--build-lib",
            str(build_dir),
            "--quiet",
        ],
    )

    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    candidate = build_dir / f"{module_name}{suffix}"
    if not candidate.exists():
        # Fall back to globbing in case setuptools added platform tags.
        matches = list(build_dir.glob(f"{module_name}*{suffix}"))
        if not matches:
            raise FileNotFoundError(
                f"Built extension for {module_name} not found in {build_dir}"
            )
        candidate = matches[0]
    return candidate
