#!/usr/bin/env python3
"""
Extract experimental dx(t) from Video S1 using similarity-anchored, prior-guided contour extraction.

Motivation (Lie benchmark)
-------------------------
The basic mask-based extractor (`extract_deformation_timeseries_from_experimental_video_s1.py`) can
under-segment the low-contrast downstream (right) flank of the biofilm in later frames, leading to an
artificially small displacement signal.

This script instead:
  1) locks the anchor (⊗) using `extract_biofilm_similarity_stripes.py` (ECC alignment Fig.5a -> frame0),
  2) extracts a smoothed polygon for each frame using a *prior* polygon search region, and
  3) computes dx(t) from extreme contour intersections at fixed y-levels.

Outputs
-------
Writes:
  - per-frame polygons (mm) to `--out-polys-dir` (ignored by git if you keep it under `out/`)
  - a timeseries CSV compatible with the Lie benchmark compare/optimizer scripts:
      t_s,dx_line1_m,dx_line2_m,dx_line3_m
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np


def _read_poly_mm(path: Path) -> np.ndarray:
    arr = np.genfromtxt(str(path), delimiter=",", skip_header=1, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2 or arr.shape[0] < 3:
        raise ValueError(f"Polygon CSV has insufficient points/cols: {path} shape={arr.shape}")
    return np.asarray(arr[:, :2], dtype=float)


def _x_intersection_on_y(poly_mm: np.ndarray, *, y_line_mm: float, mode: str) -> float:
    p = np.asarray(poly_mm, dtype=float)
    if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] != 2:
        return float("nan")

    mode = str(mode).strip().lower()
    if mode not in {"rightmost", "leftmost"}:
        raise ValueError("mode must be 'rightmost' or 'leftmost'")

    y0 = float(y_line_mm)
    xs: list[float] = []
    for a, b in zip(p, np.vstack([p[1:], p[:1]])):
        x1, y1 = float(a[0]), float(a[1])
        x2, y2 = float(b[0]), float(b[1])
        if (y0 < min(y1, y2)) or (y0 > max(y1, y2)):
            continue
        dy = y2 - y1
        if abs(dy) <= 1.0e-12:
            if abs(y0 - y1) <= 1.0e-9:
                xs.extend([x1, x2])
            continue
        t = (y0 - y1) / dy
        if 0.0 <= t <= 1.0:
            xs.append(x1 + t * (x2 - x1))
    if not xs:
        return float("nan")
    return float(np.max(xs)) if mode == "rightmost" else float(np.min(xs))


def _video_meta(video: str) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 1.0
    return float(fps), int(n_frames)


def _run_polygon_extract(
    *,
    sim_script: Path,
    video: str,
    frame: int,
    fig5a: str,
    anchor_json: str,
    prior_csv: str,
    out_csv: Path,
    debug_dir: Path,
    roi_pad_px: int,
    prior_dilate: int,
    stripe_h: int,
    stripe_q: float,
    stripe_target: float,
    stripe_gain_max: float,
    morph_k: int,
    close_iters: int,
    open_iters: int,
    cap_artifacts: bool,
    cap_run_quantile: float,
    smooth_ds_mm: float,
    smooth_window_mm: float,
    smooth_polyorder: int,
    n_verts: int,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(sim_script),
        "--video",
        str(video),
        "--frame",
        str(int(frame)),
        "--anchor-json",
        str(anchor_json),
        "--prior-csv",
        str(prior_csv),
        "--roi-pad-px",
        str(int(roi_pad_px)),
        "--prior-dilate",
        str(int(prior_dilate)),
        "--stripe-h",
        str(int(stripe_h)),
        "--stripe-q",
        str(float(stripe_q)),
        "--stripe-target",
        str(float(stripe_target)),
        "--stripe-gain-max",
        str(float(stripe_gain_max)),
        "--morph-k",
        str(int(morph_k)),
        "--close-iters",
        str(int(close_iters)),
        "--open-iters",
        str(int(open_iters)),
        "--cap-run-quantile",
        str(float(cap_run_quantile)),
        "--smooth-ds-mm",
        str(float(smooth_ds_mm)),
        "--smooth-window-mm",
        str(float(smooth_window_mm)),
        "--smooth-polyorder",
        str(int(smooth_polyorder)),
        "--n-verts",
        str(int(n_verts)),
        "--out-csv",
        str(out_csv),
        "--debug-dir",
        str(debug_dir),
        "--no-debug-images",
    ]
    if not bool(cap_artifacts):
        cmd.append("--no-cap-artifacts")

    # Only needed when anchor-json does not exist (frame0 bootstrap).
    if fig5a:
        cmd.extend(["--fig5a", str(fig5a)])

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Polygon extraction failed for frame={frame}.\nCommand: {' '.join(cmd)}\nOutput:\n{proc.stdout}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract experimental dx(t) via similarity+prior contour extraction (Lie benchmark).")
    ap.add_argument(
        "--video",
        type=str,
        default="examples/biofilms/benchmarks/lie/additional_data/bit27491-sup-0001-si_v1.avi",
        help="Video S1 AVI path.",
    )
    ap.add_argument(
        "--fig5a",
        type=str,
        default="examples/biofilms/benchmarks/lie/additional_data/figure_5_a.jpg",
        help="Figure 5a image used to auto-calibrate the anchor if needed.",
    )
    ap.add_argument(
        "--anchor-json",
        type=str,
        default="examples/biofilms/benchmarks/lie/anchor_frame0_similarity.json",
        help="Anchor JSON (created/used by the similarity extractor).",
    )
    ap.add_argument(
        "--prior0-csv",
        type=str,
        default="examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_scalebar_smooth_fig5b_v2.csv",
        help="Initial prior polygon (mm) used for frame 0 extraction (Fig.5b trace).",
    )
    ap.add_argument(
        "--poly0-mm-csv",
        type=str,
        default="examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_similarity_smooth.csv",
        help="If this file exists, use it as the fixed frame-0 polygon for Hb and x_ref (and copy it to --out-polys-dir).",
    )
    ap.add_argument("--frame-step", type=int, default=1, help="Process every N-th frame.")
    ap.add_argument("--max-frames", type=int, default=0, help="If >0, only process first N frames.")
    ap.add_argument("--x-intersection", type=str, default="rightmost", choices=("rightmost", "leftmost"))
    ap.add_argument(
        "--prior-mode",
        type=str,
        default="fixed",
        choices=("fixed", "propagate"),
        help="Use a fixed prior (frame0 polygon) for all frames, or propagate the previous extracted polygon as prior.",
    )
    ap.add_argument("--out-polys-dir", type=str, default="out/_lie_similarity_polys", help="Directory for per-frame polygon CSVs (mm).")
    ap.add_argument(
        "--out-csv",
        type=str,
        default="examples/biofilms/benchmarks/lie/exp_s1_dx_rightmost_similarity.csv",
        help="Output deformation time series CSV (meters).",
    )

    # Similarity extractor parameters (keep defaults aligned with the extractor script).
    ap.add_argument("--roi-pad-px", type=int, default=500)
    ap.add_argument("--prior-dilate", type=int, default=201)
    ap.add_argument("--stripe-h", type=int, default=20)
    ap.add_argument("--stripe-q", type=float, default=0.95)
    ap.add_argument("--stripe-target", type=float, default=180.0)
    ap.add_argument("--stripe-gain-max", type=float, default=8.0)
    ap.add_argument("--morph-k", type=int, default=7)
    ap.add_argument("--close-iters", type=int, default=2)
    ap.add_argument("--open-iters", type=int, default=1)
    ap.add_argument("--cap-artifacts", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cap-run-quantile", type=float, default=0.10)
    ap.add_argument("--smooth-ds-mm", type=float, default=0.004)
    ap.add_argument("--smooth-window-mm", type=float, default=0.030)
    ap.add_argument("--smooth-polyorder", type=int, default=3)
    ap.add_argument("--n-verts", type=int, default=260)

    args = ap.parse_args()

    sim_script = Path(__file__).with_name("extract_biofilm_similarity_stripes.py")
    if not sim_script.exists():
        raise RuntimeError(f"Missing extractor script: {sim_script}")

    fps, n_frames = _video_meta(str(args.video))
    frame_step = max(1, int(args.frame_step))
    frames = list(range(0, int(n_frames), frame_step))
    if int(args.max_frames) > 0:
        frames = frames[: int(args.max_frames)]
    if not frames or frames[0] != 0:
        frames = [0] + frames

    anchor_json = Path(str(args.anchor_json))
    prior0_csv = Path(str(args.prior0_csv))
    out_polys = Path(str(args.out_polys_dir))
    out_polys.mkdir(parents=True, exist_ok=True)

    # Ensure anchor-json exists (bootstrap anchor only; polygon output is irrelevant here).
    if not anchor_json.exists():
        tmp0 = out_polys / "_bootstrap_anchor_frame0.csv"
        _run_polygon_extract(
            sim_script=sim_script,
            video=str(args.video),
            frame=0,
            fig5a=str(args.fig5a),
            anchor_json=str(anchor_json),
            prior_csv=str(prior0_csv),
            out_csv=tmp0,
            debug_dir=out_polys / "_debug",
            roi_pad_px=int(args.roi_pad_px),
            prior_dilate=int(args.prior_dilate),
            stripe_h=int(args.stripe_h),
            stripe_q=float(args.stripe_q),
            stripe_target=float(args.stripe_target),
            stripe_gain_max=float(args.stripe_gain_max),
            morph_k=int(args.morph_k),
            close_iters=int(args.close_iters),
            open_iters=int(args.open_iters),
            cap_artifacts=bool(args.cap_artifacts),
            cap_run_quantile=float(args.cap_run_quantile),
            smooth_ds_mm=float(args.smooth_ds_mm),
            smooth_window_mm=float(args.smooth_window_mm),
            smooth_polyorder=int(args.smooth_polyorder),
            n_verts=int(args.n_verts),
        )

    # Frame 0 polygon: prefer a tracked/canonical file if provided.
    poly0_path = out_polys / f"poly_frame_{0:04d}.csv"
    poly0_fixed = Path(str(args.poly0_mm_csv))
    if poly0_fixed.exists():
        shutil.copyfile(str(poly0_fixed), str(poly0_path))
    elif not poly0_path.exists():
        _run_polygon_extract(
            sim_script=sim_script,
            video=str(args.video),
            frame=0,
            fig5a="",
            anchor_json=str(anchor_json),
            prior_csv=str(prior0_csv),
            out_csv=poly0_path,
            debug_dir=out_polys / "_debug",
            roi_pad_px=int(args.roi_pad_px),
            prior_dilate=int(args.prior_dilate),
            stripe_h=int(args.stripe_h),
            stripe_q=float(args.stripe_q),
            stripe_target=float(args.stripe_target),
            stripe_gain_max=float(args.stripe_gain_max),
            morph_k=int(args.morph_k),
            close_iters=int(args.close_iters),
            open_iters=int(args.open_iters),
            cap_artifacts=bool(args.cap_artifacts),
            cap_run_quantile=float(args.cap_run_quantile),
            smooth_ds_mm=float(args.smooth_ds_mm),
            smooth_window_mm=float(args.smooth_window_mm),
            smooth_polyorder=int(args.smooth_polyorder),
            n_verts=int(args.n_verts),
        )

    poly0 = _read_poly_mm(poly0_path)
    Hb_mm = float(np.nanmax(poly0[:, 1]))
    Hb_mm = max(1.0e-9, Hb_mm)
    y_lines_mm = [0.75 * Hb_mm, 0.50 * Hb_mm, 0.25 * Hb_mm]

    x_ref = np.array([_x_intersection_on_y(poly0, y_line_mm=y, mode=str(args.x_intersection)) for y in y_lines_mm], dtype=float)

    out_csv = Path(str(args.out_csv))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("t_s,dx_line1_m,dx_line2_m,dx_line3_m\n", encoding="utf-8")

    prior_csv = poly0_path
    for fr in frames:
        poly_path = out_polys / f"poly_frame_{int(fr):04d}.csv"
        if int(fr) != 0 and not poly_path.exists():
            prior_for_frame = poly0_path if str(args.prior_mode).strip().lower() == "fixed" else prior_csv
            _run_polygon_extract(
                sim_script=sim_script,
                video=str(args.video),
                frame=int(fr),
                fig5a="",
                anchor_json=str(anchor_json),
                prior_csv=str(prior_for_frame),
                out_csv=poly_path,
                debug_dir=out_polys / "_debug",
                roi_pad_px=int(args.roi_pad_px),
                prior_dilate=int(args.prior_dilate),
                stripe_h=int(args.stripe_h),
                stripe_q=float(args.stripe_q),
                stripe_target=float(args.stripe_target),
                stripe_gain_max=float(args.stripe_gain_max),
                morph_k=int(args.morph_k),
                close_iters=int(args.close_iters),
                open_iters=int(args.open_iters),
                cap_artifacts=bool(args.cap_artifacts),
                cap_run_quantile=float(args.cap_run_quantile),
                smooth_ds_mm=float(args.smooth_ds_mm),
                smooth_window_mm=float(args.smooth_window_mm),
                smooth_polyorder=int(args.smooth_polyorder),
                n_verts=int(args.n_verts),
            )

        poly = _read_poly_mm(poly_path)
        xs_now = np.array([_x_intersection_on_y(poly, y_line_mm=y, mode=str(args.x_intersection)) for y in y_lines_mm], dtype=float)
        dx_mm = xs_now - x_ref
        dx_m = dx_mm * 1.0e-3
        t_s = float(fr) / float(fps)
        with out_csv.open("a", encoding="utf-8") as f:
            f.write(f"{t_s:.12e},{dx_m[0]:.12e},{dx_m[1]:.12e},{dx_m[2]:.12e}\n")

        if str(args.prior_mode).strip().lower() == "propagate":
            prior_csv = poly_path

    print(f"[ok] wrote {out_csv}")
    print(f"[info] fps={fps:g}, frames_total={n_frames}, processed={len(frames)}")
    print(f"[info] Hb_mm={Hb_mm:.6g}, y_lines_mm={y_lines_mm}")
    print(f"[info] x_intersection={str(args.x_intersection)}")
    print(f"[info] polys_dir={out_polys}")


if __name__ == "__main__":
    main()
