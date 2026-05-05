#!/usr/bin/env python3
"""Compare Benchmark 6 simulation output against observation-level Blauert/Dian targets.

This tool intentionally avoids one-to-one frame matching against the OCT video.
Instead it evaluates the quantities explicitly reported in the papers:

  - `dynamic_08pa`: patchy first dynamic Blauert experiment
  - `dynamic_164pa`: fully attached second dynamic Blauert experiment
  - `steady_dian`: steady traced contour used by Dian/Feng calibration
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_STEADY_SVG = _REPO_ROOT / "examples/biofilms/benchmarks/blauert/biofilm_preprocessing/Basic_t=2_INK.svg"
_DEFAULT_DOMAIN_TXT = _REPO_ROOT / "examples/biofilms/benchmarks/blauert/biofilm_preprocessing/domain1.txt"


def _read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    if not rows:
        return {}
    out: dict[str, list[float]] = {str(key): [] for key in rows[0].keys()}
    for row in rows:
        for key, value in row.items():
            try:
                out[str(key)].append(float(value))
            except Exception:
                out[str(key)].append(float("nan"))
    return {key: np.asarray(vals, dtype=float) for key, vals in out.items()}


def _interp_scalar(xs: np.ndarray, ys: np.ndarray, xq: float) -> float:
    xx = np.asarray(xs, dtype=float).ravel()
    yy = np.asarray(ys, dtype=float).ravel()
    if xx.size == 0 or yy.size == 0 or xx.size != yy.size:
        return float("nan")
    good = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[good]
    yy = yy[good]
    if xx.size == 0:
        return float("nan")
    order = np.argsort(xx)
    xx = xx[order]
    yy = yy[order]
    if float(xq) < float(xx[0]) or float(xq) > float(xx[-1]):
        return float("nan")
    return float(np.interp(float(xq), xx, yy))


def _read_snapshot_contours_um(path: Path) -> list[np.ndarray]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    groups: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        cid = int(row["contour_id"])
        xx = 1.0e6 * float(row["x_m"])
        yy = 1.0e6 * float(row["y_m"])
        groups.setdefault(cid, []).append((xx, yy))
    return [np.asarray(groups[k], dtype=float) for k in sorted(groups)]


def _read_snapshot_contours_mm(path: Path) -> list[np.ndarray]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    groups: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        cid = int(row["contour_id"])
        xx = 1.0e3 * float(row["x_m"])
        yy = 1.0e3 * float(row["y_m"])
        groups.setdefault(cid, []).append((xx, yy))
    return [np.asarray(groups[k], dtype=float) for k in sorted(groups)]


def _segment_intersections_y(points: np.ndarray, x_sample_um: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return np.empty((0,), dtype=float)
    out: list[float] = []
    for i in range(pts.shape[0] - 1):
        x0, y0 = float(pts[i, 0]), float(pts[i, 1])
        x1, y1 = float(pts[i + 1, 0]), float(pts[i + 1, 1])
        if not ((x0 <= x_sample_um <= x1) or (x1 <= x_sample_um <= x0)):
            continue
        dx = x1 - x0
        if abs(dx) <= 1.0e-12:
            out.extend([y0, y1])
            continue
        tau = (float(x_sample_um) - x0) / dx
        if -1.0e-12 <= tau <= 1.0 + 1.0e-12:
            out.append(y0 + tau * (y1 - y0))
    return np.asarray(out, dtype=float)


def _segment_intersections_x(points: np.ndarray, y_sample: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return np.empty((0,), dtype=float)
    out: list[float] = []
    for i in range(pts.shape[0] - 1):
        x0, y0 = float(pts[i, 0]), float(pts[i, 1])
        x1, y1 = float(pts[i + 1, 0]), float(pts[i + 1, 1])
        if not ((y0 <= y_sample <= y1) or (y1 <= y_sample <= y0)):
            continue
        dy = y1 - y0
        if abs(dy) <= 1.0e-14:
            out.extend([x0, x1])
            continue
        tau = (float(y_sample) - y0) / dy
        if -1.0e-12 <= tau <= 1.0 + 1.0e-12:
            out.append(x0 + tau * (x1 - x0))
    return np.asarray(out, dtype=float)


def _top_height_um(contours_um: list[np.ndarray], x_sample_um: float) -> float:
    ys_all: list[float] = []
    for pts in contours_um:
        ys = _segment_intersections_y(pts, float(x_sample_um))
        if ys.size:
            ys_all.extend(float(v) for v in ys)
    if not ys_all:
        return 0.0
    return float(max(ys_all))


def _contour_bbox_um(contours_um: list[np.ndarray]) -> tuple[float, float, float, float]:
    pts = np.vstack([np.asarray(pts, dtype=float) for pts in contours_um])
    return (
        float(np.min(pts[:, 0])),
        float(np.max(pts[:, 0])),
        float(np.min(pts[:, 1])),
        float(np.max(pts[:, 1])),
    )


def _mean_thickness_um(
    contours_um: list[np.ndarray],
    *,
    x_min_um: float,
    x_max_um: float,
    n_samples: int = 200,
) -> float:
    xs = np.linspace(float(x_min_um), float(x_max_um), int(n_samples), dtype=float)
    hs = np.asarray([_top_height_um(contours_um, float(xx)) for xx in xs], dtype=float)
    return float(np.mean(hs))


def _front_x_um(contours_um: list[np.ndarray]) -> float:
    x_min, _, _, _ = _contour_bbox_um(contours_um)
    return float(x_min)


def _tip_x_um(contours_um: list[np.ndarray], *, height_frac: float = 0.75) -> float:
    _, _, y_min, y_max = _contour_bbox_um(contours_um)
    y_cut = float(y_min + float(height_frac) * (y_max - y_min))
    xs: list[float] = []
    for pts in contours_um:
        arr = np.asarray(pts, dtype=float)
        mask = arr[:, 1] >= float(y_cut)
        if np.any(mask):
            xs.extend(float(v) for v in arr[mask, 0])
    if not xs:
        pts = np.vstack([np.asarray(pts, dtype=float) for pts in contours_um])
        return float(np.max(pts[:, 0]))
    return float(np.max(np.asarray(xs, dtype=float)))


def _front_angle_deg(contours_um: list[np.ndarray]) -> float:
    x_min, x_max, _, _ = _contour_bbox_um(contours_um)
    x_cut = float(x_min + 0.5 * (x_max - x_min))
    top_candidates: list[tuple[float, float]] = []
    for pts in contours_um:
        arr = np.asarray(pts, dtype=float)
        mask = arr[:, 0] <= float(x_cut)
        if np.any(mask):
            sub = arr[mask]
            idx = int(np.argmax(sub[:, 1]))
            top_candidates.append((float(sub[idx, 0]), float(sub[idx, 1])))
    if not top_candidates:
        return float("nan")
    top_x, top_y = max(top_candidates, key=lambda item: item[1])
    x_min, _, _, _ = _contour_bbox_um(contours_um)
    dy = float(top_y)
    if abs(dy) <= 1.0e-12:
        return float("nan")
    return float(np.degrees(np.arctan2(float(top_x - x_min), dy)))


def _find_snapshot(out_dir: Path, t_s: float) -> Path:
    snap_dir = out_dir / "snapshots"
    candidates = sorted(snap_dir.glob(f"*t{float(t_s):06.3f}_alpha05.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Missing snapshot at t={float(t_s):.3f} s in {snap_dir}. "
            "Re-run the simulation with --snapshot-times including this time."
        )
    return candidates[0]


def _timeseries_series(path: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    data = _read_csv_columns(path)
    if "t_s" not in data or str(key) not in data:
        raise KeyError(f"Missing columns t_s/{key} in {path}")
    return np.asarray(data["t_s"], dtype=float).ravel(), np.asarray(data[str(key)], dtype=float).ravel()


def _timeseries_value(path: Path, key: str, t_s: float) -> float:
    t, y = _timeseries_series(path, key)
    return _interp_scalar(t, y, float(t_s))


def _parse_svg_path_points(path_d: str) -> np.ndarray:
    tokens = re.findall(r"[MmLlHhVvZz]|[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", str(path_d))
    pts: list[tuple[float, float]] = []
    i = 0
    x = 0.0
    y = 0.0
    start: tuple[float, float] | None = None
    cmd = ""
    while i < len(tokens):
        tok = tokens[i]
        if re.fullmatch(r"[MmLlHhVvZz]", tok):
            cmd = tok
            i += 1
            if cmd in {"Z", "z"}:
                if start is not None and (not pts or abs(pts[-1][0] - start[0]) > 1.0e-12 or abs(pts[-1][1] - start[1]) > 1.0e-12):
                    pts.append(start)
                continue
        if not cmd:
            raise ValueError("SVG path parser lost the current command.")
        if cmd in {"M", "m", "L", "l"}:
            if i + 1 >= len(tokens):
                break
            dx = float(tokens[i])
            dy = float(tokens[i + 1])
            i += 2
            if cmd in {"M", "L"}:
                x = dx
                y = dy
            else:
                x += dx
                y += dy
            pts.append((float(x), float(y)))
            if cmd == "M":
                start = (float(x), float(y))
                cmd = "L"
            elif cmd == "m":
                start = (float(x), float(y))
                cmd = "l"
            continue
        if cmd in {"H", "h"}:
            dx = float(tokens[i])
            i += 1
            x = dx if cmd == "H" else x + dx
            pts.append((float(x), float(y)))
            continue
        if cmd in {"V", "v"}:
            dy = float(tokens[i])
            i += 1
            y = dy if cmd == "V" else y + dy
            pts.append((float(x), float(y)))
            continue
        raise ValueError(f"Unsupported SVG path command {cmd!r}.")
    arr = np.asarray(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError("Failed parsing SVG path points.")
    return arr


def _svg_contour_to_mm(
    *,
    svg_path: Path,
    domain_path: Path,
    L_mm: float,
    H_mm: float,
) -> np.ndarray:
    text = svg_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'id="path3724"[^>]*d="([^"]+)"', text)
    if m is None:
        m = re.search(r'd="([^"]+)"', text)
    if m is None:
        raise ValueError(f"Failed locating SVG path in {svg_path}")
    pts_px = _parse_svg_path_points(m.group(1))
    domain = np.loadtxt(str(domain_path), skiprows=1, dtype=float)
    if domain.ndim != 2 or domain.shape[0] < 4 or domain.shape[1] < 2:
        raise ValueError(f"Invalid domain control points in {domain_path}")
    x_left = float(np.mean(domain[:2, 0]))
    x_right = float(np.mean(domain[2:, 0]))
    y_top = float(np.mean(domain[1:3, 1]))
    y_bottom = float(np.mean(domain[[0, 3], 1]))
    x_mm = (pts_px[:, 0] - x_left) / max(1.0e-12, x_right - x_left) * float(L_mm)
    y_mm = (y_bottom - pts_px[:, 1]) / max(1.0e-12, y_bottom - y_top) * float(H_mm)
    out = np.column_stack([x_mm, y_mm]).astype(float)
    if not np.allclose(out[0], out[-1], rtol=0.0, atol=1.0e-12):
        out = np.vstack([out, out[0]])
    return out


def _front_profile(points_list: list[np.ndarray], y_samples_mm: np.ndarray) -> np.ndarray:
    ys = np.asarray(y_samples_mm, dtype=float).ravel()
    out = np.full(ys.shape, float("nan"), dtype=float)
    for j, yy in enumerate(ys):
        xs_all: list[float] = []
        for pts in points_list:
            xs = _segment_intersections_x(pts, float(yy))
            if xs.size:
                xs_all.extend(float(v) for v in xs)
        if xs_all:
            out[j] = float(np.min(np.asarray(xs_all, dtype=float)))
    return out


def _contour_profile_metrics(
    *,
    exp_points: list[np.ndarray],
    sim_points: list[np.ndarray],
    y_min_mm: float = 0.02,
    y_max_mm: float = 0.42,
    n_samples: int = 120,
) -> dict[str, float]:
    ys = np.linspace(float(y_min_mm), float(y_max_mm), int(n_samples), dtype=float)
    x_exp = _front_profile(exp_points, ys)
    x_sim = _front_profile(sim_points, ys)
    mask = np.isfinite(x_exp) & np.isfinite(x_sim)
    if not np.any(mask):
        return {
            "n": 0,
            "mae_um": float("nan"),
            "rmse_um": float("nan"),
            "max_um": float("nan"),
            "front_y150_err_um": float("nan"),
        }
    err_um = 1.0e3 * (x_sim[mask] - x_exp[mask])
    x_exp_150 = _front_profile(exp_points, np.asarray([0.15], dtype=float))[0]
    x_sim_150 = _front_profile(sim_points, np.asarray([0.15], dtype=float))[0]
    return {
        "n": int(np.sum(mask)),
        "mae_um": float(np.mean(np.abs(err_um))),
        "rmse_um": float(np.sqrt(np.mean(err_um**2))),
        "max_um": float(np.max(np.abs(err_um))),
        "front_y150_err_um": 1.0e3 * abs(float(x_sim_150 - x_exp_150))
        if np.isfinite(x_exp_150) and np.isfinite(x_sim_150)
        else float("nan"),
    }


def _scenario_targets(name: str) -> dict[str, float]:
    scenario = str(name).strip().lower()
    if scenario == "dynamic_08pa":
        return {
            "front_compression_2p0_um": 148.0,
            "front_plateau_drift_2p0_10p0_um": 0.0,
            "porosity_drop_2p0_pp": 2.0,
        }
    if scenario == "dynamic_164pa":
        return {
            "thickness_drop_0p4_um": 12.0,
            "plateau_drift_0p6_1p3_um": 0.0,
            "thickness_drop_2p1_um": 27.0,
            "deformation_angle_2p1_deg": 3.0,
            "tip_elongation_2p1_um": 220.0,
            "porosity_drop_2p1_pp": 2.0,
        }
    if scenario == "steady_dian":
        return {
            "steady_profile_rmse_um": 0.0,
            "steady_profile_mae_um": 0.0,
            "steady_profile_max_um": 0.0,
            "steady_front_y150_err_um": 0.0,
        }
    raise ValueError(f"Unsupported scenario {name!r}")


def _evaluate_dynamic_08pa(out_dir: Path) -> dict[str, float]:
    ts_path = out_dir / "timeseries.csv"
    dx2 = 1.0e6 * _timeseries_value(ts_path, "dx_front_global", 2.0)
    dx10 = 1.0e6 * _timeseries_value(ts_path, "dx_front_global", 10.0)
    phi0 = _timeseries_value(ts_path, "phi_mean_alpha_weighted", 0.0)
    phi2 = _timeseries_value(ts_path, "phi_mean_alpha_weighted", 2.0)
    return {
        "front_compression_2p0_um": float(dx2),
        "front_plateau_drift_2p0_10p0_um": abs(float(dx10 - dx2))
        if np.isfinite(dx2) and np.isfinite(dx10)
        else float("nan"),
        "porosity_drop_2p0_pp": 100.0 * float(phi0 - phi2)
        if np.isfinite(phi0) and np.isfinite(phi2)
        else float("nan"),
    }


def _evaluate_dynamic_164pa(out_dir: Path) -> dict[str, float]:
    ts_path = out_dir / "timeseries.csv"
    c0 = _read_snapshot_contours_um(_find_snapshot(out_dir, 0.0))
    c04 = _read_snapshot_contours_um(_find_snapshot(out_dir, 0.4))
    c06 = _read_snapshot_contours_um(_find_snapshot(out_dir, 0.6))
    c13 = _read_snapshot_contours_um(_find_snapshot(out_dir, 1.3))
    c21 = _read_snapshot_contours_um(_find_snapshot(out_dir, 2.1))
    x0_min, x0_max, _, _ = _contour_bbox_um(c0)
    L0 = _mean_thickness_um(c0, x_min_um=x0_min, x_max_um=x0_max)
    L04 = _mean_thickness_um(c04, x_min_um=x0_min, x_max_um=x0_max)
    L06 = _mean_thickness_um(c06, x_min_um=x0_min, x_max_um=x0_max)
    L13 = _mean_thickness_um(c13, x_min_um=x0_min, x_max_um=x0_max)
    L21 = _mean_thickness_um(c21, x_min_um=x0_min, x_max_um=x0_max)
    alpha0 = _front_angle_deg(c0)
    alpha21 = _front_angle_deg(c21)
    tip0 = _tip_x_um(c0)
    tip21 = _tip_x_um(c21)
    phi0 = _timeseries_value(ts_path, "phi_mean_alpha_weighted", 0.0)
    phi21 = _timeseries_value(ts_path, "phi_mean_alpha_weighted", 2.1)
    return {
        "thickness_drop_0p4_um": float(L0 - L04),
        "plateau_drift_0p6_1p3_um": abs(float(L13 - L06)),
        "thickness_drop_2p1_um": float(L0 - L21),
        "deformation_angle_2p1_deg": float(alpha21 - alpha0),
        "tip_elongation_2p1_um": float(tip21 - tip0),
        "porosity_drop_2p1_pp": 100.0 * float(phi0 - phi21)
        if np.isfinite(phi0) and np.isfinite(phi21)
        else float("nan"),
    }


def _evaluate_steady_dian(
    out_dir: Path,
    *,
    steady_time: float,
    svg_path: Path,
    domain_path: Path,
    L_mm: float,
    H_mm: float,
) -> dict[str, float]:
    exp_contour = _svg_contour_to_mm(svg_path=svg_path, domain_path=domain_path, L_mm=L_mm, H_mm=H_mm)
    sim_contours = _read_snapshot_contours_mm(_find_snapshot(out_dir, steady_time))
    metrics = _contour_profile_metrics(exp_points=[exp_contour], sim_points=sim_contours)
    return {
        "steady_profile_rmse_um": float(metrics["rmse_um"]),
        "steady_profile_mae_um": float(metrics["mae_um"]),
        "steady_profile_max_um": float(metrics["max_um"]),
        "steady_front_y150_err_um": float(metrics["front_y150_err_um"]),
    }


def _evaluate_scenario(
    out_dir: Path,
    scenario: str,
    *,
    steady_time: float,
    svg_path: Path,
    domain_path: Path,
    L_mm: float,
    H_mm: float,
) -> dict[str, float]:
    name = str(scenario).strip().lower()
    if name == "dynamic_08pa":
        return _evaluate_dynamic_08pa(out_dir)
    if name == "dynamic_164pa":
        return _evaluate_dynamic_164pa(out_dir)
    if name == "steady_dian":
        return _evaluate_steady_dian(
            out_dir,
            steady_time=float(steady_time),
            svg_path=svg_path,
            domain_path=domain_path,
            L_mm=float(L_mm),
            H_mm=float(H_mm),
        )
    raise ValueError(f"Unsupported scenario {scenario!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=str, required=True, help="Simulation output directory.")
    ap.add_argument(
        "--scenario",
        type=str,
        default="steady_dian",
        choices=("dynamic_08pa", "dynamic_164pa", "steady_dian"),
    )
    ap.add_argument("--steady-time", type=float, default=4.0, help="Snapshot time used for the steady Dian contour comparison.")
    ap.add_argument("--steady-svg", type=str, default=str(_DEFAULT_STEADY_SVG))
    ap.add_argument("--domain-path", type=str, default=str(_DEFAULT_DOMAIN_TXT))
    ap.add_argument("--L-mm", type=float, default=5.5)
    ap.add_argument("--H-mm", type=float, default=1.0)
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    out_dir = (_REPO_ROOT / str(args.out_dir)).resolve()
    svg_path = (_REPO_ROOT / str(args.steady_svg)).resolve()
    domain_path = (_REPO_ROOT / str(args.domain_path)).resolve()
    targets = _scenario_targets(str(args.scenario))
    measured = _evaluate_scenario(
        out_dir,
        str(args.scenario),
        steady_time=float(args.steady_time),
        svg_path=svg_path,
        domain_path=domain_path,
        L_mm=float(args.L_mm),
        H_mm=float(args.H_mm),
    )

    errors: dict[str, float] = {}
    for key, target in targets.items():
        value = float(measured[key])
        errors[key] = abs(float(value) - float(target))

    payload = {
        "scenario": str(args.scenario),
        "targets": {k: float(v) for k, v in targets.items()},
        "measured": {k: float(v) for k, v in measured.items()},
        "abs_error": {k: float(v) for k, v in errors.items()},
        "mean_abs_error": float(np.mean(np.asarray(list(errors.values()), dtype=float))),
        "max_abs_error": float(np.max(np.asarray(list(errors.values()), dtype=float))),
    }

    if str(args.json_out).strip():
        out_path = (_REPO_ROOT / str(args.json_out)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
