#!/usr/bin/env python3
"""Score a Benchmark 6 case on corrected history and contour metrics.

This is the calibration scoreboard used after fixing the LM stability issue and
the global-front observable. It compares:

- corrected history RMSEs against the video-extracted front histories,
- the 2 s dynamic-08 scalar front-compression observation,
- contour-profile mismatch against video-segmented contours at representative
  times, using the same video segmentation logic as the checked-in extractor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from examples.biofilms.benchmarks.blauert.compare_sim_vs_observations import _evaluate_dynamic_08pa
from examples.biofilms.benchmarks.blauert.compare_sim_vs_video import (
    _interp_1d,
    _metrics,
    _parse_y_levels_from_columns,
    _read_csv_columns,
    _select_time_window,
)
from examples.biofilms.benchmarks.blauert.extract_front_displacement_from_video import (
    _contour_roi_from_polygon_mm,
    _contour_polygon_mm,
    _crop_mask_to_mm_roi,
    _detect_scale_bar_px,
    _detect_substrate_row,
    _overlay_rects_for_blauert_video,
    _segment_biofilm,
)
from examples.biofilms.benchmarks.blauert.paper1_benchmark6_blauert_channel import (
    _contour_profile_metrics,
    _read_sim_contour_mm,
)


DEFAULT_EXP_CSV = Path("examples/biofilms/benchmarks/blauert/exp_front_displacement_from_video.csv")
DEFAULT_VIDEO = Path("examples/biofilms/benchmarks/blauert/biofilm_preprocessing/1-s2.0-S0043135418307000-mmc1.mp4")
DEFAULT_CONTOUR_ROI_POLY = Path("examples/biofilms/benchmarks/blauert/exp_frame0_polygon_mm.csv")


def _parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_meta_items(items: list[str]) -> dict[str, str]:
    meta: dict[str, str] = {}
    for raw in items:
        text = str(raw).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"--meta requires key=value, got {text!r}")
        key, value = text.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--meta requires a non-empty key, got {text!r}")
        meta[key] = value.strip()
    return meta


def _flatten_leaderboard_row(payload: dict[str, Any], meta: dict[str, str]) -> dict[str, Any]:
    history = dict(payload.get("history", {}))
    contours = dict(payload.get("contours", {}))
    dyn08 = dict(payload.get("dynamic_08pa", {}))
    row: dict[str, Any] = dict(meta)
    row.update(
        {
            "out_dir": str(payload.get("out_dir", "")),
            "t_final_actual": payload.get("t_final_actual", float("nan")),
            "screen_score": payload.get("screen_score", float("nan")),
            "history_global_rmse_um": history.get("global_rmse_um", float("nan")),
            "history_mean_per_y_rmse_um": history.get("mean_per_y_rmse_um", float("nan")),
            "history_max_per_y_rmse_um": history.get("max_per_y_rmse_um", float("nan")),
            "history_global_bias_um": history.get("global_bias_um", float("nan")),
            "contours_mean_rmse_um": contours.get("mean_rmse_um", float("nan")),
            "contours_max_rmse_um": contours.get("max_rmse_um", float("nan")),
            "contours_max_abs_um": contours.get("max_abs_um", float("nan")),
            "front_compression_2p0_um": dyn08.get("front_compression_2p0_um", float("nan")),
            "front_compression_2p0_err_um": dyn08.get("front_compression_2p0_err_um", float("nan")),
            "front_plateau_drift_2p0_10p0_um": dyn08.get("front_plateau_drift_2p0_10p0_um", float("nan")),
            "porosity_drop_2p0_pp": dyn08.get("porosity_drop_2p0_pp", float("nan")),
        }
    )
    per_y = history.get("per_y", {})
    for label in ("y150um", "y250um", "y350um"):
        block = dict(per_y.get(label, {}))
        row[f"{label}_rmse_um"] = block.get("rmse_um", float("nan"))
        row[f"{label}_bias_um"] = block.get("bias_um", float("nan"))
    return row


def _append_leaderboard_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, str]] = []
    fieldnames = list(row.keys())
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            existing_rows = list(reader)
    for key in row.keys():
        if key not in fieldnames:
            fieldnames.append(key)
    case_id_key = str(row.get("case_id", "")).strip()
    out_dir_key = str(row.get("out_dir", ""))
    replaced = False
    for existing in existing_rows:
        existing_case_id = str(existing.get("case_id", "")).strip()
        same_row = False
        if case_id_key:
            same_row = existing_case_id == case_id_key
        else:
            same_row = str(existing.get("out_dir", "")) == out_dir_key
        if same_row:
            for key, value in row.items():
                existing[key] = "" if value is None else str(value)
            replaced = True
            break
    if not replaced:
        existing_rows.append({key: "" if row.get(key) is None else str(row.get(key)) for key in fieldnames})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for existing in existing_rows:
            writer.writerow({key: existing.get(key, "") for key in fieldnames})


class _VideoContourSampler:
    def __init__(
        self,
        video_path: Path,
        *,
        contour_roi_poly: Path | None,
        contour_roi_enabled: bool,
        contour_roi_pad_right_mm: float,
        contour_roi_pad_top_mm: float,
    ) -> None:
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if not (self.fps > 0.0):
            self.fps = 30.0

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame0 = self.cap.read()
        if not ok or frame0 is None:
            raise RuntimeError("Failed to read frame 0 from Benchmark 6 video.")

        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        _, self.px_size_um = _detect_scale_bar_px(gray0, scale_bar_um=250.0)
        self.overlay_rects = _overlay_rects_for_blauert_video(gray0)
        self.y_base0 = _detect_substrate_row(gray0)
        y_cut0 = max(1, int(self.y_base0) - 6)
        work0 = cv2.GaussianBlur(gray0[:y_cut0, :], (0, 0), sigmaX=2.0, sigmaY=2.0)
        self.fixed_thr0, _ = cv2.threshold(work0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.min_area_px = 5000
        self.contour_roi: tuple[float, float, float, float] | None = None
        if contour_roi_enabled and contour_roi_poly is not None and contour_roi_poly.exists():
            self.contour_roi = _contour_roi_from_polygon_mm(
                contour_roi_poly,
                pad_right_mm=float(contour_roi_pad_right_mm),
                pad_top_mm=float(contour_roi_pad_top_mm),
            )

    def close(self) -> None:
        self.cap.release()

    def contour_mm(self, t_s: float) -> np.ndarray:
        frame_idx = int(round(float(t_s) * float(self.fps)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame at t={float(t_s):.3f} s.")
        mask_u8, y_base = _segment_biofilm(
            frame,
            bottom_trim_px=6,
            blur_sigma=2.0,
            close_radius_px=7,
            close_iters=1,
            fill_holes=True,
            min_area_px=5000,
            y_base_override=int(self.y_base0),
            fixed_threshold=float(self.fixed_thr0),
            overlay_rects=self.overlay_rects,
        )
        if self.contour_roi is not None:
            x_min_mm, x_max_mm, y_min_mm, y_max_mm = self.contour_roi
            mask_u8 = _crop_mask_to_mm_roi(
                mask_u8,
                y_base=int(y_base),
                px_size_um=float(self.px_size_um),
                x_min_mm=float(x_min_mm),
                x_max_mm=float(x_max_mm),
                y_min_mm=float(y_min_mm),
                y_max_mm=float(y_max_mm),
                min_area_px=int(self.min_area_px),
            )
        return _contour_polygon_mm(
            mask_u8,
            y_base=int(y_base),
            px_size_um=float(self.px_size_um),
            simplify_eps_px=0.0,
        )


def _history_metrics(*, exp_csv: Path, sim_csv: Path, t_min: float, t_max: float) -> dict[str, Any]:
    exp = _read_csv_columns(exp_csv)
    sim = _read_csv_columns(sim_csv)
    t_exp = np.asarray(exp["t_s"], dtype=float).ravel()
    t_sim = np.asarray(sim["t_s"], dtype=float).ravel()

    idx = _select_time_window(t_exp, t_min=t_min, t_max=t_max)
    t_ref = t_exp[idx]

    dx_exp_um = np.asarray(exp["dx_front_um"], dtype=float).ravel()[idx]
    dx_sim_um = _interp_1d(t_sim, 1.0e6 * np.asarray(sim["dx_front_global"], dtype=float).ravel(), t_ref)
    global_metrics = _metrics(dx_exp_um, dx_sim_um)

    exp_map = _parse_y_levels_from_columns(list(exp.keys()), kind="exp")
    sim_map = _parse_y_levels_from_columns(list(sim.keys()), kind="sim")
    per_y: dict[str, dict[str, float]] = {}
    vals: list[float] = []
    for y_um, exp_col in sorted(exp_map.items()):
        sim_levels = np.array(sorted(sim_map.keys()), dtype=float)
        if sim_levels.size == 0:
            continue
        y_sim_um = int(sim_levels[int(np.argmin(np.abs(sim_levels - float(y_um))))])
        sim_col = sim_map.get(y_sim_um)
        if sim_col is None:
            continue
        y_true = np.asarray(exp[exp_col], dtype=float).ravel()[idx]
        y_pred = _interp_1d(t_sim, 1.0e6 * np.asarray(sim[sim_col], dtype=float).ravel(), t_ref)
        metrics = _metrics(y_true, y_pred)
        per_y[f"y{int(y_um)}um"] = {
            "rmse_um": float(metrics["rmse"]),
            "mae_um": float(metrics["mae"]),
            "bias_um": float(metrics["bias"]),
        }
        if np.isfinite(float(metrics["rmse"])):
            vals.append(float(metrics["rmse"]))

    return {
        "global_rmse_um": float(global_metrics["rmse"]),
        "global_mae_um": float(global_metrics["mae"]),
        "global_bias_um": float(global_metrics["bias"]),
        "mean_per_y_rmse_um": float(np.mean(vals)) if vals else float("nan"),
        "max_per_y_rmse_um": float(np.max(vals)) if vals else float("nan"),
        "per_y": per_y,
    }


def _contour_metrics(
    *,
    out_dir: Path,
    contour_times: list[float],
    x_shift_mm: float,
    sampler: _VideoContourSampler,
) -> dict[str, Any]:
    rows: dict[str, dict[str, float]] = {}
    rmses: list[float] = []
    maxes: list[float] = []
    for t_now in contour_times:
        snap = sorted((out_dir / "snapshots").glob(f"*t{float(t_now):06.3f}_alpha05.csv"))
        if not snap:
            rows[f"{float(t_now):.3f}s"] = {"rmse_um": float("nan"), "mae_um": float("nan"), "max_um": float("nan")}
            continue
        exp_pts = [sampler.contour_mm(float(t_now))]
        sim_pts = _read_sim_contour_mm(snap[0], x_shift_mm=float(x_shift_mm))
        metrics = _contour_profile_metrics(exp_points=exp_pts, sim_points=sim_pts)
        rows[f"{float(t_now):.3f}s"] = {k: float(v) for k, v in metrics.items() if k != "n"}
        if np.isfinite(float(metrics["rmse_um"])):
            rmses.append(float(metrics["rmse_um"]))
        if np.isfinite(float(metrics["max_um"])):
            maxes.append(float(metrics["max_um"]))
    return {
        "times": rows,
        "mean_rmse_um": float(np.mean(rmses)) if rmses else float("nan"),
        "max_rmse_um": float(np.max(rmses)) if rmses else float("nan"),
        "max_abs_um": float(np.max(maxes)) if maxes else float("nan"),
    }


def _screen_score(*, history: dict[str, Any], contours: dict[str, Any], dyn08: dict[str, Any]) -> float:
    terms: list[float] = []
    weights: list[float] = []

    def _add(value: float, target: float, weight: float) -> None:
        if np.isfinite(float(value)):
            terms.append(float(value) / max(1.0e-12, float(target)))
            weights.append(float(weight))

    _add(float(history["global_rmse_um"]), 20.0, 0.35)
    _add(float(history["mean_per_y_rmse_um"]), 20.0, 0.25)
    _add(float(contours["mean_rmse_um"]), 120.0, 0.25)
    _add(abs(float(history["global_bias_um"])), 15.0, 0.05)
    _add(abs(float(dyn08["front_compression_2p0_um"]) - 148.0), 50.0, 0.10)
    if not terms:
        return float("inf")
    w = np.asarray(weights, dtype=float)
    v = np.asarray(terms, dtype=float)
    return float(np.sum(w * v) / np.sum(w))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--exp-csv", type=str, default=str(DEFAULT_EXP_CSV))
    ap.add_argument("--video", type=str, default=str(DEFAULT_VIDEO))
    ap.add_argument("--t-min", type=float, default=0.5)
    ap.add_argument("--t-max", type=float, default=2.0)
    ap.add_argument("--contour-times", type=str, default="1.0,1.5,2.0")
    ap.add_argument("--alpha0-tx-mm", type=float, default=0.5)
    ap.add_argument("--contour-roi-poly-csv", type=str, default=str(DEFAULT_CONTOUR_ROI_POLY))
    ap.add_argument("--contour-roi-pad-right-mm", type=float, default=0.02)
    ap.add_argument("--contour-roi-pad-top-mm", type=float, default=0.02)
    ap.add_argument("--contour-roi", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--json-out", type=str, default="")
    ap.add_argument("--leaderboard-csv", type=str, default="")
    ap.add_argument("--meta", action="append", default=[], help="Repeated key=value metadata entries written to --leaderboard-csv.")
    args = ap.parse_args()

    out_dir = Path(str(args.out_dir))
    sim_csv = out_dir / "timeseries.csv"
    if not sim_csv.exists():
        raise FileNotFoundError(sim_csv)

    exp_csv = Path(str(args.exp_csv))
    sampler = _VideoContourSampler(
        Path(str(args.video)),
        contour_roi_poly=Path(str(args.contour_roi_poly_csv)) if str(args.contour_roi_poly_csv).strip() else None,
        contour_roi_enabled=bool(args.contour_roi),
        contour_roi_pad_right_mm=float(args.contour_roi_pad_right_mm),
        contour_roi_pad_top_mm=float(args.contour_roi_pad_top_mm),
    )
    try:
        history = _history_metrics(exp_csv=exp_csv, sim_csv=sim_csv, t_min=float(args.t_min), t_max=float(args.t_max))
        dyn08 = _evaluate_dynamic_08pa(out_dir)
        contours = _contour_metrics(
            out_dir=out_dir,
            contour_times=_parse_float_list(str(args.contour_times)),
            x_shift_mm=float(args.alpha0_tx_mm),
            sampler=sampler,
        )
    finally:
        sampler.close()

    sim = _read_csv_columns(sim_csv)
    t_final_actual = float(np.nanmax(np.asarray(sim["t_s"], dtype=float).ravel()))

    payload: dict[str, Any] = {
        "out_dir": str(out_dir),
        "t_final_actual": t_final_actual,
        "history": history,
        "dynamic_08pa": {
            "front_compression_2p0_um": float(dyn08.get("front_compression_2p0_um", float("nan"))),
            "front_compression_2p0_err_um": abs(float(dyn08.get("front_compression_2p0_um", float("nan"))) - 148.0)
            if np.isfinite(float(dyn08.get("front_compression_2p0_um", float("nan"))))
            else float("nan"),
            "front_plateau_drift_2p0_10p0_um": float(dyn08.get("front_plateau_drift_2p0_10p0_um", float("nan"))),
            "porosity_drop_2p0_pp": float(dyn08.get("porosity_drop_2p0_pp", float("nan"))),
        },
        "contours": contours,
    }
    payload["screen_score"] = _screen_score(history=history, contours=contours, dyn08=payload["dynamic_08pa"])

    if str(args.json_out).strip():
        out_json = Path(str(args.json_out))
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if str(args.leaderboard_csv).strip():
        meta = _parse_meta_items(list(args.meta))
        row = _flatten_leaderboard_row(payload, meta)
        _append_leaderboard_csv(Path(str(args.leaderboard_csv)), row)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
