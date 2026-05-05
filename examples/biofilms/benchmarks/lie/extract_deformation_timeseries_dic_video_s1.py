#!/usr/bin/env python3
"""
Subset-based DIC extraction of dx(t) from Li et al. (2020) Video S1 (OCT).

This implements a *publishable* DIC workflow:
  - rigid stabilization (per-frame similarity transform) using SVG-traced base endpoints,
  - subset correlation (translation-only DIC) with optional subpixel + multi-scale,
  - dx(t) extraction at three tracking heights (fractions of initial biofilm height).

Outputs a CSV compatible with:
  examples/biofilms/benchmarks/lie/compare_exp_sim_timeseries_dx.py

Notes
-----
This is not a full "finite-strain, affine subset" DIC solver. It is translation
DIC with stabilization, which is typically sufficient for extracting the paper's
1D dx(t) time series (Fig.7).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from examples.biofilms.benchmarks.lie.dic_utils import (
    DicSettings,
    erode_mask_u8,
    polygon_mask_u8,
    similarity_affine_from_2pts,
    track_points_translation_dic,
    warp_affine_src_to_dst,
)
from examples.biofilms.benchmarks.lie.svg_trace_utils import extract_svg_frame_geometry, extract_svg_mark_points_px


def _read_frame(cap: cv2.VideoCapture, frame: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    if not ok or img is None:
        raise RuntimeError(f"Could not read frame={frame}")
    return img


def _preprocess_gray(img_bgr: np.ndarray, *, blur_ksize: int) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = int(blur_ksize)
    if k > 1:
        if k % 2 == 0:
            k += 1
        g = cv2.GaussianBlur(g, (k, k), 0)
    return np.asarray(g, dtype=np.uint8)


def _parse_floats_csv(s: str) -> list[float]:
    out: list[float] = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def _x_intersections_on_y(poly_xy: np.ndarray, *, y: float) -> np.ndarray:
    """
    Intersections of an open polyline with the horizontal line y=const.

    poly_xy is expected to start at base_left and end at base_right.
    """
    pts = np.asarray(poly_xy, dtype=float)
    y0 = float(y)
    xs: list[float] = []
    for a, b in zip(pts[:-1], pts[1:]):
        ya, yb = float(a[1]), float(b[1])
        if (ya - y0) * (yb - y0) > 0.0:
            continue
        if abs(yb - ya) <= 1.0e-12:
            continue
        t = (y0 - ya) / (yb - ya)
        if t < 0.0 or t > 1.0:
            continue
        x = float(a[0] + t * (float(b[0]) - float(a[0])))
        xs.append(x)
    return np.asarray(sorted(set(xs)), dtype=float)


def _choose_tracking_points(
    *,
    geom0_boundary_px: np.ndarray,
    base_left_px: np.ndarray,
    base_right_px: np.ndarray,
    y_fracs: list[float],
    x_quantile: float,
    x_span: float,
    samples_per_line: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (pts_ref_xy, line_id).
    """
    boundary = np.asarray(geom0_boundary_px, dtype=float)
    base_y = 0.5 * float(base_left_px[1] + base_right_px[1])
    y_top = float(np.min(boundary[:, 1]))
    hb = float(base_y - y_top)
    if not np.isfinite(hb) or hb <= 5.0:
        raise RuntimeError(f"Invalid initial biofilm height in pixels (hb={hb}).")

    q0 = float(x_quantile)
    span = float(max(0.0, float(x_span)))
    n = int(max(1, int(samples_per_line)))
    if n == 1:
        qs = np.array([q0], dtype=float)
    else:
        qs = np.linspace(q0 - 0.5 * span, q0 + 0.5 * span, num=n, endpoint=True, dtype=float)
        qs = np.clip(qs, 0.0, 1.0)

    pts_out: list[list[float]] = []
    line_id: list[int] = []
    for li, yf in enumerate(y_fracs):
        y_line = float(base_y - float(yf) * hb)
        xs = _x_intersections_on_y(boundary, y=y_line)
        if xs.size < 2:
            raise RuntimeError(f"Could not find left/right intersections on line {li+1} (y_frac={yf}).")
        xL, xR = float(np.min(xs)), float(np.max(xs))
        if not (xR > xL):
            raise RuntimeError(f"Degenerate intersection span on line {li+1} (xL={xL}, xR={xR}).")
        for q in qs.tolist():
            x = float(xL + float(q) * (xR - xL))
            pts_out.append([x, y_line])
            line_id.append(int(li))

    return np.asarray(pts_out, dtype=float), np.asarray(line_id, dtype=int)


def _line_y_positions_px(
    *,
    boundary_px: np.ndarray,
    base_left_px: np.ndarray,
    base_right_px: np.ndarray,
    y_fracs: list[float],
) -> tuple[float, float, list[float]]:
    boundary = np.asarray(boundary_px, dtype=float)
    base_y = 0.5 * float(base_left_px[1] + base_right_px[1])
    y_top = float(np.min(boundary[:, 1]))
    hb = float(base_y - y_top)
    if not np.isfinite(hb) or hb <= 5.0:
        raise RuntimeError(f"Invalid initial biofilm height in pixels (hb={hb}).")
    y_lines = [float(base_y - float(yf) * hb) for yf in y_fracs]
    return float(base_y), float(hb), y_lines


def main() -> None:
    ap = argparse.ArgumentParser(description="DIC-based dx(t) extraction from Video S1 (Lie benchmark).")
    ap.add_argument(
        "--video",
        type=str,
        default="examples/biofilms/benchmarks/lie/additional_data/bit27491-sup-0001-si_v1.avi",
        help="Experimental video (Video S1).",
    )
    ap.add_argument("--svg-dir", type=str, default="examples/biofilms/benchmarks/lie/svg_fles", help="Directory with frame_XXXX.svg.")
    ap.add_argument("--out-csv", type=str, default="out/_lie_exp_s1_dic_dx/timeseries.csv", help="Output CSV.")
    ap.add_argument("--out-npz", type=str, default="", help="Optional NPZ with per-point displacements.")
    ap.add_argument("--out-video", type=str, default="", help="Optional debug video showing tracked points/vectors.")
    ap.add_argument("--frame-step", type=int, default=1, help="Use every N-th SVG/video frame.")
    ap.add_argument("--max-frames", type=int, default=0, help="If >0, stop after N processed frames.")
    ap.add_argument("--block-w-mm", type=float, default=1.0, help="Support width for mm/px scaling (mm).")
    ap.add_argument("--cubic-samples", type=int, default=20, help="SVG cubic Bezier sampling.")
    ap.add_argument("--join-tol-px", type=float, default=5.0, help="SVG segment join tolerance (px).")

    ap.add_argument("--mode", type=str, default="lines", choices=("lines", "grid", "marks"), help="DIC point selection mode.")
    ap.add_argument("--y-fracs", type=str, default="0.75,0.5,0.25", help="Tracking-line y-fractions (top->bottom).")
    ap.add_argument("--x-quantile", type=float, default=0.5, help="x position as quantile between left/right boundary on each line (0=upstream edge, 0.5=middle).")
    ap.add_argument("--x-span", type=float, default=0.4, help="Span in quantile space around --x-quantile (for multiple samples per line).")
    ap.add_argument("--samples-per-line", type=int, default=5, help="How many DIC points to track per line (averaged).")
    ap.add_argument("--grid-step", type=int, default=10, help="Grid spacing (px) for --mode grid.")
    ap.add_argument("--line-band-px", type=float, default=8.0, help="y-band half-width (px) used to select grid points per tracking line.")
    ap.add_argument("--grid-only-lines", action="store_true", help="In --mode grid, only keep points within --line-band-px of any tracking line.")
    ap.add_argument("--max-grid-points", type=int, default=0, help="If >0, randomly subsample grid points to this count.")
    ap.add_argument("--vis-max-points", type=int, default=400, help="Max points to draw in debug video (subsampled if needed).")

    ap.add_argument("--subset", type=int, default=41, help="Subset (template) size in pixels (odd).")
    ap.add_argument("--search-radius", type=int, default=20, help="Search radius in pixels.")
    ap.add_argument("--pyr-levels", type=int, default=3, help="Gaussian pyramid levels.")
    ap.add_argument("--method", type=str, default="zncc", choices=("zncc", "znssd"), help="Matching cost.")
    ap.add_argument("--no-subpixel", action="store_true", help="Disable subpixel peak refinement.")
    ap.add_argument("--min-zncc", type=float, default=0.2, help="Reject matches with ZNCC peak below this (zncc only).")
    ap.add_argument("--blur", type=int, default=3, help="Gaussian blur kernel for preprocessing (0 disables).")
    args = ap.parse_args()

    video = Path(str(args.video))
    svg_dir = Path(str(args.svg_dir))
    out_csv = Path(str(args.out_csv))
    out_npz = Path(str(args.out_npz)) if str(args.out_npz).strip() else None
    out_video = Path(str(args.out_video)) if str(args.out_video).strip() else None

    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")
    if not svg_dir.exists():
        raise FileNotFoundError(f"SVG directory not found: {svg_dir}")

    svg_files = sorted(svg_dir.glob("frame_*.svg"))
    if not svg_files:
        raise FileNotFoundError(f"No SVG frames found in {svg_dir}")
    # Intersect with the video by indexing from the filename.
    frame_ids: list[int] = []
    svg_by_id: dict[int, Path] = {}
    for p in svg_files:
        stem = p.stem
        if not stem.startswith("frame_"):
            continue
        try:
            k = int(stem.split("_", 1)[1])
        except Exception:
            continue
        frame_ids.append(k)
        svg_by_id[k] = p
    frame_ids = sorted(set(frame_ids))
    if not frame_ids:
        raise RuntimeError(f"Could not parse any frame indices from SVGs in {svg_dir}")

    frame_step = int(max(1, int(args.frame_step)))
    frame_ids = [k for k in frame_ids if (k % frame_step) == 0]
    if int(args.max_frames) > 0:
        frame_ids = frame_ids[: int(args.max_frames)]
    if not frame_ids:
        raise RuntimeError("No frames selected after --frame-step/--max-frames filtering.")

    # Reference geometry from frame0 SVG (required).
    if 0 not in svg_by_id:
        raise FileNotFoundError(f"Missing frame_0000.svg in {svg_dir} (required for DIC reference).")
    geom0 = extract_svg_frame_geometry(
        svg_by_id[0],
        block_w_mm=float(args.block_w_mm),
        cubic_samples=int(args.cubic_samples),
        join_tol_px=float(args.join_tol_px),
    )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 1.0

    # Reference frame (stabilized = itself).
    img0 = _read_frame(cap, 0)
    h0, w0 = img0.shape[:2]
    ref_gray = _preprocess_gray(img0, blur_ksize=int(args.blur))

    # Tracking points in the reference (pixel coordinates).
    mode = str(args.mode).strip().lower()
    if mode == "marks":
        marks_px = extract_svg_mark_points_px(
            svg_by_id[0],
            cubic_samples=int(args.cubic_samples),
            join_tol_px=float(args.join_tol_px),
        )
        marks_px = np.asarray(marks_px, dtype=float).reshape((-1, 2))
        if marks_px.shape[0] < 3:
            raise RuntimeError(f"--mode marks requires ≥3 mark points in frame_0000.svg; found {marks_px.shape[0]}")

        y = marks_px[:, 1]
        i_top = int(np.argmin(y))
        i_bot = int(np.argmax(y))
        y_med = float(np.median(y))
        i_mid = int(np.argmin(np.abs(y - y_med)))
        idx = []
        for i in (i_top, i_mid, i_bot):
            if int(i) not in idx:
                idx.append(int(i))
        if len(idx) < 3:
            rest = [int(i) for i in np.argsort(y).tolist() if int(i) not in idx]
            idx.extend(rest[: (3 - len(idx))])
        idx = idx[:3]
        idx = sorted(idx, key=lambda j: float(marks_px[int(j), 1]))  # top->bottom in image coords
        pts_ref = np.asarray(marks_px[idx, :], dtype=float)
        line_id = np.arange(3, dtype=int)
    else:
        y_fracs = _parse_floats_csv(args.y_fracs)
        if len(y_fracs) != 3:
            raise ValueError(f"--y-fracs must contain exactly 3 values; got {len(y_fracs)}")
        if mode == "lines":
            pts_ref, line_id = _choose_tracking_points(
                geom0_boundary_px=geom0.boundary_px,
                base_left_px=geom0.base_left_px,
                base_right_px=geom0.base_right_px,
                y_fracs=y_fracs,
                x_quantile=float(args.x_quantile),
                x_span=float(args.x_span),
                samples_per_line=int(args.samples_per_line),
            )
        elif mode == "grid":
            _base_y, _hb, y_lines = _line_y_positions_px(
                boundary_px=geom0.boundary_px,
                base_left_px=geom0.base_left_px,
                base_right_px=geom0.base_right_px,
                y_fracs=y_fracs,
            )
            poly = np.vstack([np.asarray(geom0.boundary_px, dtype=float), np.asarray(geom0.base_left_px, dtype=float).reshape(1, 2)])
            mask = polygon_mask_u8(shape_hw=(h0, w0), poly_xy=poly)

            subset = int(args.subset)
            subset = subset if (subset % 2) == 1 else (subset + 1)
            erode_r = int(max(0, int(args.search_radius) + (subset // 2) + 2))
            mask_e = erode_mask_u8(mask, radius_px=erode_r)

            step = int(max(1, int(args.grid_step)))
            pts_list: list[list[float]] = []
            for yy in range(0, h0, step):
                row = mask_e[yy, :]
                for xx in range(0, w0, step):
                    if int(row[xx]) == 0:
                        continue
                    pts_list.append([float(xx), float(yy)])
            pts_ref = np.asarray(pts_list, dtype=float)
            if pts_ref.size == 0:
                raise RuntimeError("Grid selection produced zero points; adjust --grid-step/--subset/--search-radius.")

            y_lines_arr = np.asarray(y_lines, dtype=float).reshape(1, 3)
            dy = np.abs(pts_ref[:, 1:2] - y_lines_arr)
            j = np.argmin(dy, axis=1).astype(int)
            dmin = np.min(dy, axis=1)
            line_id = np.full((pts_ref.shape[0],), -1, dtype=int)
            sel = dmin <= float(args.line_band_px)
            line_id[sel] = j[sel]

            if bool(args.grid_only_lines):
                keep = line_id >= 0
                pts_ref = pts_ref[keep, :]
                line_id = line_id[keep]
                if pts_ref.size == 0:
                    raise RuntimeError("--grid-only-lines removed all points; increase --line-band-px.")

            if int(args.max_grid_points) > 0 and pts_ref.shape[0] > int(args.max_grid_points):
                rng = np.random.default_rng(0)
                idx = rng.choice(pts_ref.shape[0], size=int(args.max_grid_points), replace=False)
                pts_ref = pts_ref[idx, :]
                line_id = line_id[idx]
        else:
            raise ValueError(f"Unknown --mode {args.mode!r}.")

    # Per-frame DIC.
    settings = DicSettings(
        subset_px=int(args.subset),
        search_radius_px=int(args.search_radius),
        pyramid_levels=int(args.pyr_levels),
        method=str(args.method),
        subpixel=not bool(args.no_subpixel),
        min_score=float(args.min_zncc),
    )

    m_per_px = float(geom0.mm_per_px) * 1.0e-3
    if not np.isfinite(m_per_px) or m_per_px <= 0.0:
        raise RuntimeError(f"Invalid m_per_px from SVG: {m_per_px}")

    # Optional debug video (use stabilized BGR frames).
    writer = None
    if out_video is not None:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if out_video.suffix.lower() == ".mp4" else "XVID"))
        writer = cv2.VideoWriter(str(out_video), fourcc, float(fps) / float(frame_step), (int(w0), int(h0)))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {out_video}")

    # Storage.
    t_s: list[float] = []
    dx_lines_m: list[list[float]] = []
    disp_all: list[np.ndarray] = []
    score_all: list[np.ndarray] = []

    disp_prev = np.zeros((pts_ref.shape[0], 2), dtype=float)
    vis_idx = np.arange(pts_ref.shape[0], dtype=int)
    if pts_ref.shape[0] > int(args.vis_max_points) and int(args.vis_max_points) > 0:
        rng = np.random.default_rng(0)
        vis_idx = np.sort(rng.choice(pts_ref.shape[0], size=int(args.vis_max_points), replace=False))

    for k in frame_ids:
        svg_path = svg_by_id.get(int(k))
        if svg_path is None or (not svg_path.exists()):
            logging.warning("Skipping frame %d (missing SVG).", int(k))
            continue
        geomk = extract_svg_frame_geometry(
            svg_path,
            block_w_mm=float(args.block_w_mm),
            m_per_px=float(m_per_px),
            cubic_samples=int(args.cubic_samples),
            join_tol_px=float(args.join_tol_px),
        )

        imgk = _read_frame(cap, int(k))
        if imgk.shape[0] != h0 or imgk.shape[1] != w0:
            raise RuntimeError("Video resolution changes across frames (unexpected).")

        M = similarity_affine_from_2pts(
            src_left=geomk.base_left_px,
            src_right=geomk.base_right_px,
            dst_left=geom0.base_left_px,
            dst_right=geom0.base_right_px,
        )
        imgk_w = warp_affine_src_to_dst(imgk, M=M, dsize=(w0, h0), border_value=0)
        cur_gray = _preprocess_gray(imgk_w, blur_ksize=int(args.blur))

        if int(k) == 0:
            disp = np.zeros_like(disp_prev)
            score = np.full((disp.shape[0],), 1.0, dtype=float)
        else:
            disp, score = track_points_translation_dic(
                ref_gray_u8=ref_gray,
                cur_gray_u8=cur_gray,
                pts_ref_xy=pts_ref,
                disp0_xy=disp_prev,
                settings=settings,
            )
        disp_prev = np.where(np.isfinite(disp), disp, disp_prev)

        # Aggregate per line.
        dx_m = disp[:, 0] * float(m_per_px)
        dx_line = []
        for li in range(3):
            sel = line_id == int(li)
            v = dx_m[sel]
            v = v[np.isfinite(v)]
            dx_line.append(float(np.nanmean(v)) if v.size else float("nan"))

        t = float(k) / float(fps)
        t_s.append(t)
        dx_lines_m.append(dx_line)
        disp_all.append(np.asarray(disp, dtype=float))
        score_all.append(np.asarray(score, dtype=float))

        if writer is not None:
            vis = imgk_w.copy()
            for idx in vis_idx.tolist():
                p_ref = pts_ref[int(idx)]
                dxy = disp[int(idx)]
                sc = score[int(idx)]
                if not (np.all(np.isfinite(p_ref)) and np.all(np.isfinite(dxy))):
                    continue
                p0 = (int(round(float(p_ref[0]))), int(round(float(p_ref[1]))))
                p1 = (int(round(float(p_ref[0] + dxy[0]))), int(round(float(p_ref[1] + dxy[1]))))
                cv2.circle(vis, p0, 2, (255, 255, 0), -1)
                cv2.arrowedLine(vis, p0, p1, (0, 255, 0), 1, tipLength=0.25)
                if np.isfinite(float(sc)):
                    cv2.putText(
                        vis,
                        f"{float(sc):.2f}",
                        (p0[0] + 3, p0[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            cv2.putText(vis, f"t={t:.2f}s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(vis)

    cap.release()
    if writer is not None:
        writer.release()

    if not t_s:
        raise RuntimeError("No frames were processed.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack([np.asarray(t_s, dtype=float), np.asarray(dx_lines_m, dtype=float)])
    header = "t_s,dx_line1_m,dx_line2_m,dx_line3_m"
    np.savetxt(str(out_csv), arr, delimiter=",", header=header, comments="")
    print(f"[ok] wrote {out_csv} ({arr.shape[0]} frames)")

    if out_npz is not None:
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(out_npz),
            t_s=np.asarray(t_s, dtype=float),
            pts_ref_xy=np.asarray(pts_ref, dtype=float),
            line_id=np.asarray(line_id, dtype=int),
            disp_xy=np.stack(disp_all, axis=0),
            score=np.stack(score_all, axis=0),
            m_per_px=float(m_per_px),
            fps=float(fps),
        )
        print(f"[ok] wrote {out_npz}")

    if out_video is not None:
        print(f"[ok] wrote {out_video}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
