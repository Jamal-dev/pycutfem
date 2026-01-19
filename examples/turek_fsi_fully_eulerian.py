#!/usr/bin/env python
# coding: utf-8
"""
Compatibility wrapper for the fully Eulerian Turek–Hron FSI benchmark.

The implementation lives at:
  examples/turek_fsi_fully_eulerian/turek_fsi_fully_eulerian.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).resolve().parent / "turek_fsi_fully_eulerian" / "turek_fsi_fully_eulerian.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
