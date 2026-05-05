#!/usr/bin/env python

"""
Demonstrate the "error mismatch" root cause: under-integration of the L2 norm.

This runs the quad-mesh NGSXFEM-vs-PyCutFEM comparison multiple times while
only varying the *error-evaluation* quadrature order (XFEM_Q_ERR).

Run (inside xfemcustom env)
---------------------------
PYCUTFEM_JIT_BACKEND=cpp conda run --no-capture-output -n xfemcustom \\
  python examples/debug/xfem_interface_compare_qerr_sweep_quad.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


_RE_L2_NG = re.compile(r"^NGSolve / NGSXFEM L2 error: (.*)$")
_RE_L2_PY = re.compile(r"^PyCutFEM \(cpp backend\) L2 error: (.*)$")


def _run_once(q_err: int) -> tuple[float, float]:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "compare_xfem_interface_ngsolve_vs_pycutfem_quad.py"
    if not script.exists():
        raise FileNotFoundError(str(script))

    env = os.environ.copy()
    env["XFEM_Q_ERR"] = str(int(q_err))
    env.setdefault("PYCUTFEM_JIT_BACKEND", "cpp")

    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(root),
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    l2_ng = None
    l2_py = None
    for line in proc.stdout.splitlines():
        m = _RE_L2_NG.match(line.strip())
        if m:
            l2_ng = float(m.group(1))
            continue
        m = _RE_L2_PY.match(line.strip())
        if m:
            l2_py = float(m.group(1))
            continue
    if l2_ng is None or l2_py is None:
        raise RuntimeError("Failed to parse L2 errors from output:\n" + proc.stdout)
    return float(l2_ng), float(l2_py)


def main() -> None:
    q_err_list = [4, 8, 12, 16]
    print("=== XFEM interface (quad) error quadrature sweep ===")
    print("Set XFEM_Q_ERR to see the effect on the *reported* L2 error.\n")
    print("q_err, L2_ngsolve, L2_pycutfem, abs_diff")
    for q_err in q_err_list:
        l2_ng, l2_py = _run_once(q_err)
        print(f"{q_err:d}, {l2_ng:.16e}, {l2_py:.16e}, {abs(l2_ng - l2_py):.3e}")


if __name__ == "__main__":
    main()
