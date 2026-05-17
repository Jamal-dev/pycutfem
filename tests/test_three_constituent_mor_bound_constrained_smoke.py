from __future__ import annotations

import os

import pytest

from examples.biofilms.benchmarks.three_constituent.three_constituent_mor import run_validation


def _have_cpp_backend() -> bool:
    try:
        import pybind11  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
@pytest.mark.skipif(
    os.environ.get("PYCUTFEM_RUN_THREE_CONSTITUENT_MOR", "0") != "1",
    reason="set PYCUTFEM_RUN_THREE_CONSTITUENT_MOR=1 to run the nine-field native MOR smoke benchmark",
)
def test_three_constituent_mor_native_bound_constrained_smoke(tmp_path) -> None:
    result = run_validation(
        outdir=tmp_path / "three_constituent_mor",
        nx=1,
        heldout_dt=0.060,
        train_dts=(0.035, 0.055, 0.080),
        qdeg=3,
        max_modes=2,
        warmup=False,
    )

    assert result.passed
    assert result.summary["native_reduced"]["backend"] == "cpp_native_deim_pdas_online"
    assert result.summary["native_reduced"]["converged"]
    assert result.summary["bounds"]["rom"]["max_violation"] <= 1.0e-9
    assert result.summary["speedup"]["factor"] > 0.0
    assert result.summary["offline"]["sampled_element_count"] <= result.summary["offline"]["total_element_count"]
