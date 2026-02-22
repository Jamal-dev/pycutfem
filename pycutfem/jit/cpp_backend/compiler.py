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


def _auto_fast_compile_enabled() -> bool:
    raw = os.getenv("PYCUTFEM_CPP_AUTO_FAST_COMPILE", "1").strip().lower()
    return raw not in {"0", "false", "no"}


def _auto_fast_ir_threshold() -> int:
    raw = os.getenv("PYCUTFEM_CPP_AUTO_FAST_IR_OPS", "").strip()
    if not raw:
        # Heuristic: large IR sequences generate enormous translation units and
        # can trigger very long compile times / GCC ICEs under -O2/-O3.
        return 8000
    try:
        return int(raw)
    except Exception:
        return 8000


def get_compile_mode_tag(*, ir_len: int | None = None) -> str:
    """
    Return a stable tag describing the compile mode for cache keying.

    NOTE: The C++ cache key must include this tag to avoid reusing -O0 test
    builds in normal runs (or vice versa).
    """
    base = "fast" if _fast_compile_enabled() else "opt"
    if base == "opt" and _auto_fast_compile_enabled() and ir_len is not None:
        if int(ir_len) >= _auto_fast_ir_threshold():
            return "fast"
    return base


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if raw is None:
        return bool(default)
    raw = str(raw).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


def _opt_level_flag() -> str:
    raw = os.getenv("PYCUTFEM_CPP_OPT_LEVEL", "").strip().lower()
    if not raw:
        # Default to -O2 for stability / compile-time reasons; -O3 often offers
        # limited wins for Eigen-heavy generated code while increasing ICE risk.
        return "-O2"
    if raw in {"0", "1", "2", "3"}:
        return f"-O{raw}"
    if raw.startswith("-o"):
        return raw
    return "-O2"


def _compile_args_for_mode(mode: str) -> list[str]:
    """Baseline compiler flags for a given compile mode."""
    mode = str(mode or "").strip().lower()
    if mode not in {"fast", "opt"}:
        raise ValueError(f"Unknown compile mode: {mode!r}")

    args = ["-O0" if mode == "fast" else _opt_level_flag(), "-std=c++17"]

    # `-march=native` is great for performance but can dominate compile time and
    # has proven brittle with some conda toolchains (multiple -march flags, ICEs).
    # Keep it opt-in.
    if mode == "opt" and _bool_env("PYCUTFEM_CPP_MARCH_NATIVE", default=False):
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
    compile_mode: str | None = None,
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

    compile_mode_resolved = str(compile_mode or get_compile_mode_tag()).strip().lower()
    if compile_mode_resolved not in {"fast", "opt"}:
        raise ValueError(f"compile_mode must be 'fast' or 'opt' (got {compile_mode_resolved!r})")

    build_dir.mkdir(parents=True, exist_ok=True)
    tmp_build = build_dir / f"build_temp_{compile_mode_resolved}"
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

    compile_args = list(_compile_args_for_mode(compile_mode_resolved))
    link_args = list(_default_link_args())
    if extra_compile_args:
        compile_args.extend(extra_compile_args)
    if extra_link_args:
        link_args.extend(extra_link_args)

    def _make_ext(args: list[str], links: list[str]) -> Extension:
        return Extension(
            module_name,
            sources=[str(source_path)],
            include_dirs=[str(p) for p in include_dirs],
            language="c++",
            extra_compile_args=list(args),
            extra_link_args=list(links),
        )

    class _BuildExt(build_ext):  # type: ignore[misc]
        def build_extension(self, ext):  # pragma: no cover - invoked by setuptools
            super().build_extension(ext)

    def _run_build(ext: Extension, tmp: Path) -> None:
        setup(
            name=module_name,
            ext_modules=[ext],
            cmdclass={"build_ext": _BuildExt},
            script_args=[
                "build_ext",
                "--build-temp",
                str(tmp),
                "--build-lib",
                str(build_dir),
                "--quiet",
            ],
        )

    # Run an isolated build into the cache directory. If the compiler fails
    # (including GCC internal compiler errors), retry once with fast flags.
    try:
        _run_build(_make_ext(compile_args, link_args), tmp_build)
    except BaseException as exc_opt:  # pragma: no cover - exercised only on toolchain failures
        if compile_mode_resolved == "fast":
            raise
        tmp_fast = build_dir / "build_temp_fast_fallback"
        tmp_fast.mkdir(parents=True, exist_ok=True)
        fast_args = _compile_args_for_mode("fast")
        try:
            _run_build(_make_ext(fast_args, link_args), tmp_fast)
        except BaseException as exc_fast:
            raise RuntimeError(
                f"Failed building C++ kernel {module_name!r} in modes 'opt' and 'fast'."
            ) from exc_fast

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
