from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_similarity_module():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "biofilms" / "benchmarks" / "lie" / "extract_biofilm_similarity_stripes.py"
    spec = importlib.util.spec_from_file_location("lie_extract_similarity", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def lie_similarity():
    try:
        return _load_similarity_module()
    except ModuleNotFoundError as exc:
        pytest.skip(f"Skipping Lie similarity extractor helper tests; missing dependency: {exc}")


def test_clip_thin_artifacts_disabled_is_noop(lie_similarity) -> None:
    mask = np.zeros((12, 5), dtype=np.uint8)
    # Main body: long contiguous run from y=4..10.
    mask[4:11, :] = 255
    # Add a short disconnected streak above in one column.
    mask[1:3, 2] = 255

    out = lie_similarity._clip_thin_artifacts_above_cap(
        mask,
        enabled=False,
        base_y_i=10,
        x_left_i=0,
        x_right_i=4,
        run_quantile=0.0,
    )
    assert np.array_equal(out, mask)


def test_clip_thin_artifacts_enabled_removes_short_streak(lie_similarity) -> None:
    mask = np.zeros((12, 5), dtype=np.uint8)
    mask[4:11, :] = 255
    mask[1:3, 2] = 255

    out = lie_similarity._clip_thin_artifacts_above_cap(
        mask,
        enabled=True,
        base_y_i=10,
        x_left_i=0,
        x_right_i=4,
        run_quantile=0.0,
    )
    # The short streak at y=1..2 must be removed.
    assert int(np.sum(out[1:3, 2])) == 0
    # The main body must remain.
    assert np.all(out[4:11, 2] == 255)

