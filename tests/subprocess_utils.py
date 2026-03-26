from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_module_func_in_subprocess(module: str, func: str) -> None:
    """
    Run `module.func()` in a fresh Python process.

    This is used to isolate C++ backend (pybind11) kernel imports from the main
    pytest process. Importing many distinct pybind11 extension modules in a
    single process can exhaust the available thread-specific storage keys.
    """
    repo_root = Path(__file__).resolve().parents[1]
    code = f"from {module} import {func} as _f; _f()"

    with tempfile.TemporaryDirectory(prefix="pycutfem_subproc_cache_") as tmpdir:
        cache_root = Path(tmpdir).resolve()
        env = os.environ.copy()
        # Subprocess-based C++ tests run alongside xdist workers. Give each
        # subprocess an isolated cache/ref-table root so it never races on the
        # parent worker's cache directory or reuses stale kernels from another
        # worker.
        env["PYCUTFEM_CACHE_DIR"] = str(cache_root)
        env["PYCUTFEM_REF_TABLE_CACHE_DIR"] = str(cache_root / "ref_tables")
        # Headless tmux/CI sessions often have no DISPLAY. Force a non-GUI
        # backend so tests importing matplotlib helpers do not try to open Tk.
        env.setdefault("MPLBACKEND", "Agg")

        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        raise AssertionError(
            "Subprocess run failed.\n"
            f"  module: {module}\n"
            f"  func:   {func}\n"
            f"  rc:     {proc.returncode}\n"
            f"  cache:  {cache_root}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
        )
