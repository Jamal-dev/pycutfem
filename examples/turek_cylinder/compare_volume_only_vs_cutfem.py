#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Tuple


def _read_timeseries(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows


def _index_by_time(rows: list[dict[str, str]]) -> Dict[float, dict[str, str]]:
    out: Dict[float, dict[str, str]] = {}
    for row in rows:
        t = float(row["time"])
        key = round(t, 12)
        out[key] = row
    return out


def _get_cols(series: str) -> Tuple[str, str]:
    if series == "surf":
        return "Cd_surf", "Cl_surf"
    if series == "bm":
        return "Cd_bm", "Cl_bm"
    if series == "nitsche":
        return "Cd_nitsche", "Cl_nitsche"
    raise ValueError(f"Unknown CutFEM series: {series}")


def _rms(vals: list[float]) -> float:
    if not vals:
        return math.nan
    return math.sqrt(sum(v * v for v in vals) / float(len(vals)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CutFEM (examples/turek_cylinder/turek_benchmark.py) against the volume-only benchmark."
    )
    parser.add_argument(
        "--cutfem",
        type=Path,
        default=Path("examples/turek_cylinder/turek_results/functionals.csv"),
        help="Path to CutFEM functionals.csv",
    )
    parser.add_argument(
        "--volume",
        type=Path,
        required=True,
        help="Path to volume-only functionals.csv",
    )
    parser.add_argument(
        "--cutfem-series",
        choices=("surf", "bm", "nitsche"),
        default="surf",
        help="Which CutFEM force series to compare.",
    )
    args = parser.parse_args()

    cut_rows = _read_timeseries(args.cutfem)
    vol_rows = _read_timeseries(args.volume)

    cut_cd_col, cut_cl_col = _get_cols(str(args.cutfem_series))
    for name in ("time", cut_cd_col, cut_cl_col, "dp"):
        if name not in cut_rows[0]:
            raise KeyError(f"Missing column '{name}' in {args.cutfem}")
    for name in ("time", "Cd", "Cl", "dp"):
        if name not in vol_rows[0]:
            raise KeyError(f"Missing column '{name}' in {args.volume}")

    cut_by_t = _index_by_time(cut_rows)
    vol_by_t = _index_by_time(vol_rows)

    times = sorted(set(cut_by_t) & set(vol_by_t))
    if not times:
        raise RuntimeError("No overlapping output times found between the two CSV files.")

    d_cd: list[float] = []
    d_cl: list[float] = []
    d_dp: list[float] = []
    for t_key in times:
        cut = cut_by_t[t_key]
        vol = vol_by_t[t_key]
        d_cd.append(float(cut[cut_cd_col]) - float(vol["Cd"]))
        d_cl.append(float(cut[cut_cl_col]) - float(vol["Cl"]))
        d_dp.append(float(cut["dp"]) - float(vol["dp"]))

    print(f"Common points: {len(times)}")
    print(f"Cd: rms={_rms(d_cd):.6e}  max|.|={max(abs(v) for v in d_cd):.6e}")
    print(f"Cl: rms={_rms(d_cl):.6e}  max|.|={max(abs(v) for v in d_cl):.6e}")
    print(f"dp: rms={_rms(d_dp):.6e}  max|.|={max(abs(v) for v in d_dp):.6e}")


if __name__ == "__main__":
    main()

