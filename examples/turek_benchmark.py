"""
Deprecated entry-point (kept for backward compatibility).

The CutFEM cylinder benchmark script and its FeatFlow reference data now live in:
  `examples/turek_cylinder/`.

Run:
  `python examples/turek_cylinder/turek_benchmark.py ...`
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).resolve().parent / "turek_cylinder" / "turek_benchmark.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

