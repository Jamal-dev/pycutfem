from __future__ import annotations

from examples.biofilms.benchmarks.dadu.paper1_benchmark3_candidate_screen import _candidate_dirs


def test_candidate_dirs_filters_for_full_archive_runs(tmp_path) -> None:
    good = tmp_path / "paper1_benchmark3_one_domain_nx060_t28p6"
    good.mkdir()
    for name in ("summary.json", "snaps_alpha.npz", "y_top_timeseries.csv"):
        (good / name).write_text("x", encoding="utf-8")

    missing = tmp_path / "paper1_benchmark3_one_domain_nx080_t28p6"
    missing.mkdir()
    (missing / "summary.json").write_text("x", encoding="utf-8")
    (missing / "snaps_alpha.npz").write_text("x", encoding="utf-8")

    wrong_name = tmp_path / "unrelated_case_t28p6"
    wrong_name.mkdir()
    for name in ("summary.json", "snaps_alpha.npz", "y_top_timeseries.csv"):
        (wrong_name / name).write_text("x", encoding="utf-8")

    short_run = tmp_path / "paper1_benchmark3_one_domain_nx060_t12p7"
    short_run.mkdir()
    for name in ("summary.json", "snaps_alpha.npz", "y_top_timeseries.csv"):
        (short_run / name).write_text("x", encoding="utf-8")

    got = _candidate_dirs(tmp_path, required_tags=("one_domain", "benchmark3"))

    assert got == [good]
