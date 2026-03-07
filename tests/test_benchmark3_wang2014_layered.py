from __future__ import annotations

from pathlib import Path

from examples.biofilms.benchmarks.wang.paper1_benchmark3_wang2014_layered import run_benchmark


def test_wang2014_layered_benchmark_writes_outputs_and_improves_with_smaller_K(tmp_path: Path) -> None:
    outdir = tmp_path / "wang2014_layered"
    summary = run_benchmark(
        outdir=outdir,
        k_list=[1.0e-2, 1.0e-4],
        ny=600,
        plot_ks=[1.0e-2, 1.0e-4],
        dpi=120,
    )

    rows = summary["rows"]
    assert len(rows) == 2
    coarse = rows[0]
    fine = rows[1]
    assert float(coarse["K"]) > float(fine["K"])
    assert float(fine["l2_fluid"]) < float(coarse["l2_fluid"])
    assert float(fine["l2_porous"]) < float(coarse["l2_porous"])
    assert float(fine["h1_fluid"]) < float(coarse["h1_fluid"])
    assert float(fine["h1_porous"]) < float(coarse["h1_porous"])

    assert (outdir / "benchmark3_wang2014_layered_summary.csv").exists()
    assert (outdir / "benchmark3_wang2014_layered_summary.json").exists()
    assert (outdir / "benchmark3_wang2014_layered_profiles.png").exists()
    assert (outdir / "benchmark3_wang2014_layered_error_trends.png").exists()
