from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_svg_trace_utils():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "biofilms" / "benchmarks" / "lie" / "svg_trace_utils.py"
    spec = importlib.util.spec_from_file_location("lie_svg_trace_utils", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {script}")
    mod = importlib.util.module_from_spec(spec)
    # Needed for dataclasses + `from __future__ import annotations` under Python 3.12:
    # the dataclass machinery consults sys.modules[__module__] to resolve type strings.
    sys.modules[str(spec.name)] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def svg_utils():
    return _load_svg_trace_utils()


def test_extract_svg_frame_geometry_ignores_disconnected_mark_paths(svg_utils, tmp_path: Path) -> None:
    # Base: horizontal line at y=10 from x=0..10 (10 px).
    # Boundary: open polyline base_left -> left -> top -> right -> base_right.
    # Mark: a small disconnected segment that must be ignored.
    svg = tmp_path / "frame_0000.svg"
    svg.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
  <path d="M 0,10 L 10,10" style="fill:none;stroke:#000000;stroke-width:1px" />
  <path d="M 0,10 L 0,0 L 10,0 L 10,10" style="fill:none;stroke:#000000;stroke-width:1px" />
  <path d="M 5,5 L 6,5" style="fill:none;stroke:#000000;stroke-width:1px" />
</svg>
""",
        encoding="utf-8",
    )

    geom = svg_utils.extract_svg_frame_geometry(svg, block_w_mm=1.0, m_per_px=None, cubic_samples=1, join_tol_px=0.5)
    assert np.allclose(geom.base_left_px, [0.0, 10.0])
    assert np.allclose(geom.base_right_px, [10.0, 10.0])
    assert np.allclose(geom.boundary_px[0], geom.base_left_px)
    assert np.allclose(geom.boundary_px[-1], geom.base_right_px)

    # Base length 10 px -> 1 mm => 0.1 mm/px, so base_right should map to x=1 mm.
    poly = np.asarray(geom.polygon_mm, dtype=float)
    assert poly.shape[1] == 2
    assert np.allclose(poly[0], poly[-1])
    assert np.isclose(float(np.max(poly[:, 0])), 1.0, atol=1.0e-6)
    assert float(np.max(poly[:, 1])) > 0.5

    # Marks helper should return the disconnected segment endpoints (and exclude base endpoints).
    marks = np.asarray(svg_utils.extract_svg_mark_points_px(svg, cubic_samples=1, join_tol_px=0.5), dtype=float)
    assert marks.shape == (2, 2)
    # Order isn't important.
    assert any(np.allclose(m, [5.0, 5.0]) for m in marks)
    assert any(np.allclose(m, [6.0, 5.0]) for m in marks)
