from __future__ import annotations

import os
import subprocess
import sys
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

    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "Subprocess run failed.\n"
            f"  module: {module}\n"
            f"  func:   {func}\n"
            f"  rc:     {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
        )
