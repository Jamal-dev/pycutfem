from __future__ import annotations

import os
from pathlib import Path

import pytest

from examples.NIRB.debug.compare_example2_step_history import compare_step_histories


def test_example2_exact_step10_compare_is_closed() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    local_dir_raw = os.getenv("PYCUTFEM_EX2_STEP10_LOCAL_DIR", "").strip()
    if not local_dir_raw:
        pytest.skip("Set PYCUTFEM_EX2_STEP10_LOCAL_DIR to a fresh local Example 2 step-history run.")

    local_dir = Path(local_dir_raw).expanduser()
    if not local_dir.is_absolute():
        local_dir = repo_root / local_dir
    if not local_dir.exists():
        pytest.skip(f"Configured step-10 local artifact is missing: {local_dir}")

    kratos_dir_raw = os.getenv(
        "PYCUTFEM_EX2_STEP10_KRATOS_DIR",
        "examples/NIRB/artifacts/kratos_step_history_0140_0145/step_history",
    ).strip()
    kratos_dir = Path(kratos_dir_raw).expanduser()
    if not kratos_dir.is_absolute():
        kratos_dir = repo_root / kratos_dir
    if not kratos_dir.exists():
        pytest.skip(f"Kratos reference step history is missing: {kratos_dir}")

    summary = compare_step_histories(local_step_dir=local_dir, kratos_step_dir=kratos_dir)

    assert int(summary.get("last_compared_step", 0) or 0) >= 10
    assert summary.get("first_divergence_step") is None

    step10 = next(row for row in summary["rows"] if int(row["step"]) == 10)
    assert float(step10["worst_rel_l2_value"]) <= 1.0e-8
