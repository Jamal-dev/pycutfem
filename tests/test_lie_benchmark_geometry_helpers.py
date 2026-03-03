from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_lie_geom_module():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "biofilms" / "benchmarks" / "lie" / "extract_geometry_from_video_s1_frame0.py"
    spec = importlib.util.spec_from_file_location("lie_extract_geom", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def lie_geom():
    try:
        return _load_lie_geom_module()
    except ModuleNotFoundError as exc:
        pytest.skip(f"Skipping Lie geometry helper tests; missing dependency: {exc}")


def test_top_envelope_bins_do_not_stay_all_nan(lie_geom) -> None:
    pts = np.array(
        [
            [0.00, 0.20],
            [0.10, 0.40],
            [0.20, 0.60],
            [0.25, 0.65],
            [0.80, 1.10],
            [0.90, 1.20],
            [1.00, 1.00],
        ],
        dtype=float,
    )
    x_centers, y_top = lie_geom._top_envelope_xy(pts, x_min=0.0, x_max=1.0, n_bins=100)
    assert x_centers.shape == y_top.shape
    assert np.any(np.isfinite(y_top))
    # Peaks near the right side should be preserved.
    assert np.nanmax(y_top) >= 1.15


def test_column_top_uses_longest_run_not_short_streak(lie_geom) -> None:
    mask = np.zeros((12, 5), dtype=bool)
    # Main body: long contiguous run from y=4..10.
    mask[4:11, :] = True
    # Add a short disconnected streak above in one column.
    mask[1:3, 2] = True
    xs, y_top = lie_geom._column_top_from_longest_run(
        mask,
        base_y_i=10,
        x_left_i=0,
        x_right_i=4,
        run_quantile=0.0,
    )
    assert xs.shape == y_top.shape
    assert np.all(np.isfinite(y_top))
    # The short streak at y=1..2 must be ignored; top should remain on the main body.
    assert np.all(y_top >= 4.0)
