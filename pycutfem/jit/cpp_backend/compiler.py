"""Utility for compiling generated C++ kernels via pybind11."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sysconfig
from pathlib import Path
from typing import Iterable, Sequence


def _fast_compile_enabled() -> bool:
    fast_env = os.getenv("PYCUTFEM_CPP_FAST_COMPILE", "").strip().lower()
    under_pytest = bool(os.getenv("PYTEST_CURRENT_TEST", ""))
    return (fast_env in {"1", "true", "yes"}) or (under_pytest and fast_env not in {"0", "false", "no"})


def _auto_fast_compile_enabled() -> bool:
    # Performance-first default: do not silently drop to "fast" (-O0/-Og) builds
    # for large kernels. Keep this as an opt-in escape hatch for toolchains
    # that cannot compile huge generated translation units under -O2/-O3.
    raw = os.getenv("PYCUTFEM_CPP_AUTO_FAST_COMPILE", "0").strip().lower()
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
            base = "fast"
    if base == "fast":
        tag = f"fast_{_format_opt_tag(_fast_opt_level_flag())}"
        if _bool_env("PYCUTFEM_CPP_FAST_MARCH_NATIVE", default=False):
            tag += "_native"
        return tag
    tag = f"opt_{_format_opt_tag(_opt_level_flag())}"
    if _bool_env("PYCUTFEM_CPP_MARCH_NATIVE", default=True):
        tag += "_native"
    return tag


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
        # Default to -O3 for runtime performance (vectorization/inlining).
        return "-O3"
    if raw in {"0", "1", "2", "3"}:
        return f"-O{raw}"
    if raw.startswith("-o"):
        return raw
    return "-O3"

def _fast_opt_level_flag() -> str:
    """
    Optimization level to use for "fast" builds.

    Default: -O0 for maximum compile robustness.
    Override via PYCUTFEM_CPP_FAST_OPT_LEVEL (e.g. "Og", "1", "2").
    """
    raw = os.getenv("PYCUTFEM_CPP_FAST_OPT_LEVEL", "").strip()
    if not raw:
        return "-O0"
    low = raw.lower()
    if low in {"g", "og"}:
        return "-Og"
    if low in {"0", "1", "2", "3"}:
        return f"-O{low}"
    if low.startswith("-o"):
        return raw
    return "-O0"


def _format_opt_tag(opt_flag: str) -> str:
    """
    Convert an optimization flag (-O0/-O2/-Og/-Ofast/...) to a stable cache tag.
    """
    raw = str(opt_flag or "").strip()
    if not raw:
        return "O0"
    if raw.startswith("-"):
        raw = raw[1:]
    raw = raw.replace("=", "")
    return "".join(ch for ch in raw if ch.isalnum()) or "O"


def _compile_args_for_mode(mode: str) -> list[str]:
    """Baseline compiler flags for a given compile mode."""
    mode = str(mode or "").strip().lower()
    if mode not in {"fast", "opt"}:
        raise ValueError(f"Unknown compile mode: {mode!r}")

    opt_flag = _fast_opt_level_flag() if mode == "fast" else _opt_level_flag()
    args = [
        opt_flag,
        "-std=c++17",
        # Keep diagnostics compact in normal runs; failures still show the first
        # relevant errors and the generated source path. Suppress warning spam
        # from Eigen headers and generated-but-unused temporaries so review logs
        # only show real compile failures.
        "-w",
        "-fdiagnostics-color=never",
        "-fno-diagnostics-show-caret",
        "-fmax-errors=5",
        # pybind11 enables subinterpreter support on Python >= 3.12 by default,
        # which adds per-module thread-specific storage keys. The C++ backend
        # compiles many kernels as distinct extension modules; with subinterpreter
        # support enabled this can exhaust the available TSS keys and crash/abort
        # during module import. Disable it for robustness in long-running runs.
        "-DPYBIND11_HAS_SUBINTERPRETER_SUPPORT=0",
    ]

    if mode == "fast" and opt_flag == "-O0":
        # Conda toolchains often inject -D_FORTIFY_SOURCE=2 globally; under -O0 that
        # produces noisy preprocessor warnings on every kernel compile. Undefine it for
        # fast review builds so the logs stay signal-rich.
        args.extend(["-U_FORTIFY_SOURCE", "-D_FORTIFY_SOURCE=0"])

    # Performance-first default: enable `-march=native` unless explicitly disabled.
    # (This is the single largest win for Eigen-heavy kernels.)
    if mode == "opt" and _bool_env("PYCUTFEM_CPP_MARCH_NATIVE", default=True):
        args.append("-march=native")
    if mode == "fast" and _bool_env("PYCUTFEM_CPP_FAST_MARCH_NATIVE", default=False):
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
    mode_base = compile_mode_resolved.split("_", 1)[0].split("-", 1)[0]
    if mode_base not in {"fast", "opt"}:
        raise ValueError(
            "compile_mode must start with 'fast' or 'opt' "
            f"(got {compile_mode_resolved!r})"
        )

    build_dir.mkdir(parents=True, exist_ok=True)

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

    compile_args = list(_compile_args_for_mode(mode_base))
    link_args = list(_default_link_args())
    if extra_compile_args:
        compile_args.extend(extra_compile_args)
    if extra_link_args:
        link_args.extend(extra_link_args)
    print(
        f"[pycutfem][cpp] compiling kernel {module_name} ({compile_mode_resolved})",
        flush=True,
    )
    if _bool_env("PYCUTFEM_CPP_VERBOSE", default=False):
        print(
            f"[pycutfem][cpp] build {module_name} ({compile_mode_resolved}) "
            f"cxxflags={' '.join(compile_args)}",
            flush=True,
        )

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
        script_args = [
            "build_ext",
            "--build-temp",
            str(tmp),
            "--build-lib",
            str(build_dir),
            "--quiet",
        ]
        if _bool_env("PYCUTFEM_CPP_VERBOSE", default=False):
            setup(
                name=module_name,
                ext_modules=[ext],
                cmdclass={"build_ext": _BuildExt},
                script_args=script_args,
            )
            return

        out_buf = io.StringIO()
        err_buf = io.StringIO()
        prev_disable = logging.root.manager.disable
        try:
            logging.disable(logging.INFO)
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                setup(
                    name=module_name,
                    ext_modules=[ext],
                    cmdclass={"build_ext": _BuildExt},
                    script_args=script_args,
                )
        except Exception:
            captured = (out_buf.getvalue() + err_buf.getvalue()).strip()
            if captured:
                print(captured, flush=True)
            raise
        finally:
            logging.disable(prev_disable)

    def _without_flag(args: list[str], flag: str) -> list[str]:
        return [a for a in args if a != flag]

    def _retry_opt_levels(primary: str) -> list[str]:
        # Conservative fallback ladder for toolchains that ICE at -O3/-march=native.
        # Keep performance-oriented flags (vectorization) as much as possible.
        primary = str(primary or "").strip()
        if primary in {"-O3", "-Ofast"}:
            return ["-O2", "-O1"]
        if primary == "-O2":
            return ["-O1"]
        return []

    allow_o0_fallback = _bool_env("PYCUTFEM_CPP_ALLOW_O0_FALLBACK", default=False)

    attempts: list[tuple[str, list[str]]] = []
    attempts.append((compile_mode_resolved, compile_args))

    if mode_base == "opt":
        # Try a couple of performance-preserving fallbacks before giving up:
        #   1) drop -march=native (still vectorizes, just with a smaller ISA)
        #   2) step down -O3 -> -O2 -> -O1 (keep -march when possible)
        if "-march=native" in compile_args:
            attempts.append((f"{compile_mode_resolved}_nomarch", _without_flag(compile_args, "-march=native")))

        for opt_flag in _retry_opt_levels(compile_args[0] if compile_args else ""):
            stepped = list(compile_args)
            if stepped:
                stepped[0] = opt_flag
            attempts.append((f"{compile_mode_resolved}_{_format_opt_tag(opt_flag)}", stepped))
            if "-march=native" in stepped:
                attempts.append(
                    (f"{compile_mode_resolved}_{_format_opt_tag(opt_flag)}_nomarch", _without_flag(stepped, "-march=native"))
                )

        if allow_o0_fallback:
            attempts.append(("fast_fallback", _compile_args_for_mode("fast")))

    last_exc: BaseException | None = None
    succeeded = False
    for label, args in attempts:
        tmp_build = build_dir / f"build_temp_{label}"
        tmp_build.mkdir(parents=True, exist_ok=True)
        try:
            _run_build(_make_ext(args, link_args), tmp_build)
            succeeded = True
            if label != compile_mode_resolved and (
                _bool_env("PYCUTFEM_CPP_VERBOSE", default=False)
                or _bool_env("PYCUTFEM_CPP_WARN_FALLBACK", default=False)
            ):
                print(
                    f"[pycutfem][cpp] WARNING: kernel {module_name} build fell back to flags: {' '.join(args)}",
                    flush=True,
                )
            break
        except BaseException as exc:  # pragma: no cover - exercised only on toolchain failures
            last_exc = exc

    if not succeeded:
        hint = ""
        if mode_base == "opt" and not allow_o0_fallback:
            hint = " (set PYCUTFEM_CPP_ALLOW_O0_FALLBACK=1 to allow an unoptimized last-resort build)"
        raise RuntimeError(f"Failed building C++ kernel {module_name!r}{hint}.") from last_exc

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
