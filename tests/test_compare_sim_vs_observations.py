from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from examples.biofilms.benchmarks.blauert.compare_sim_vs_observations import (
    _evaluate_dynamic_08pa,
    _evaluate_steady_dian,
)


def _write_timeseries(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "t_s",
                "dx_front_global",
                "phi_mean_alpha_weighted",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_snapshot(path: Path, points_m: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["contour_id", "point_id", "x_m", "y_m"])
        writer.writeheader()
        for i, (xx, yy) in enumerate(np.asarray(points_m, dtype=float)):
            writer.writerow(
                {
                    "contour_id": 0,
                    "point_id": i,
                    "x_m": f"{float(xx):.12e}",
                    "y_m": f"{float(yy):.12e}",
                }
            )


def test_dynamic_08pa_observations_from_timeseries(tmp_path: Path) -> None:
    out_dir = tmp_path / "sim"
    _write_timeseries(
        out_dir / "timeseries.csv",
        [
            {"t_s": 0.0, "dx_front_global": 0.0, "phi_mean_alpha_weighted": 0.47},
            {"t_s": 2.0, "dx_front_global": 148.0e-6, "phi_mean_alpha_weighted": 0.45},
            {"t_s": 10.0, "dx_front_global": 148.0e-6, "phi_mean_alpha_weighted": 0.45},
        ],
    )

    measured = _evaluate_dynamic_08pa(out_dir)

    assert measured["front_compression_2p0_um"] == pytest.approx(148.0)
    assert measured["front_plateau_drift_2p0_10p0_um"] == pytest.approx(0.0)
    assert measured["porosity_drop_2p0_pp"] == pytest.approx(2.0)


def test_steady_dian_contour_match_is_exact(tmp_path: Path) -> None:
    out_dir = tmp_path / "sim"
    snap = out_dir / "snapshots" / "snapshot_step0001_t04.000_alpha05.csv"
    rect_mm = np.asarray(
        [
            [0.2, 0.0],
            [0.2, 0.8],
            [0.8, 0.8],
            [0.8, 0.0],
            [0.2, 0.0],
        ],
        dtype=float,
    )
    _write_snapshot(snap, 1.0e-3 * rect_mm)

    svg_path = tmp_path / "steady.svg"
    svg_path.write_text(
        """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<path id="path3724" d="M 20 100 L 20 20 L 80 20 L 80 100 Z" />
</svg>
""",
        encoding="utf-8",
    )
    domain_path = tmp_path / "domain.txt"
    domain_path.write_text(
        "x y\n0 100\n0 0\n100 0\n100 100\n",
        encoding="utf-8",
    )

    measured = _evaluate_steady_dian(
        out_dir,
        steady_time=4.0,
        svg_path=svg_path,
        domain_path=domain_path,
        L_mm=1.0,
        H_mm=1.0,
    )

    assert measured["steady_profile_rmse_um"] == pytest.approx(0.0)
    assert measured["steady_profile_mae_um"] == pytest.approx(0.0)
    assert measured["steady_profile_max_um"] == pytest.approx(0.0)
    assert measured["steady_front_y150_err_um"] == pytest.approx(0.0)
