"""
Benchmark L2/H1 error norm evaluation on the unfitted Stokes example.

This is a thin wrapper around `examples/stokes_cut.py` that enables the
timing/probing block via `PYCUTFEM_BENCH_ERRORS=1` and runs the script
in-process (so the compiled-kernel caches can be observed in a single run).

Example
-------
`PYCUTFEM_JIT_BACKEND=cpp BACKEND=jit WITH_DEF=1 conda run --no-capture-output -n xfemcustom python examples/debug/bench_stokes_cut_errors.py`
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


def main() -> None:
    os.environ.setdefault("PYCUTFEM_BENCH_ERRORS", "1")
    stokes_path = (Path(__file__).resolve().parent.parent / "stokes_cut.py").resolve()
    runpy.run_path(str(stokes_path), run_name="__main__")


if __name__ == "__main__":
    main()

