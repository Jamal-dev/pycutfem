from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("scipy")

from examples.biofilms.benchmarks.wang.paper1_benchmark3_wang2014_staircase import run_benchmark


def test_wang2014_staircase_benchmark_writes_outputs_and_improves_with_resolved_smaller_K(tmp_path: Path) -> None:
    outdir = tmp_path / "wang2014_staircase"
    summary = run_benchmark(
        outdir=outdir,
        grid_pairs=[(1.0e-2, 32), (1.0e-3, 64)],
        plot_ks=[1.0e-2],
        dpi=120,
    )

    rows = summary["rows"]
    assert len(rows) == 2
    coarse = rows[0]
    fine = rows[1]
    assert float(coarse["K"]) > float(fine["K"])
    assert int(coarse["nxy"]) < int(fine["nxy"])
    assert float(fine["l2_fluid"]) < float(coarse["l2_fluid"])
    assert float(fine["l2_porous"]) < float(coarse["l2_porous"])
    assert float(fine["profile_linf"]) < float(coarse["profile_linf"])

    assert (outdir / "benchmark3_wang2014_staircase_summary.csv").exists()
    assert (outdir / "benchmark3_wang2014_staircase_summary.json").exists()
    assert (outdir / "benchmark3_wang2014_staircase_geometry.png").exists()
    assert (outdir / "benchmark3_wang2014_staircase_profiles.png").exists()
    assert (outdir / "benchmark3_wang2014_staircase_fields.png").exists()
    assert (outdir / "benchmark3_wang2014_staircase_error_trends.png").exists()
