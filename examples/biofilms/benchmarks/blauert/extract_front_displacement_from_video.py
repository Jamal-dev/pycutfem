"""
Extract a simple experimental deformation timeseries from the Blauert et al. (2015)
supplementary video (mmc1).

We segment the biofilm (Otsu threshold + morphology), then track an upstream/front
position as a left-quantile of mask pixels. Pixel units are converted to micrometers
using the 250 um scale bar visible in the video frames.

Outputs
-------
* CSV timeseries: `t_s, frame, x_front_px, x_front_um, dx_front_um, ...`
* Initial polygon CSV (optional): simplified outer contour from frame 0 in millimeters.

This script is intended as a robust, reproducible extractor for model-vs-video
validation; it is not an exact replica of the ImageJ workflow described in the paper.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def _parse_float_list(s: str) -> list[float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return [float(p) for p in parts]


def _read_polygon_mm(path: Path | str) -> np.ndarray:
    arr = np.genfromtxt(str(path), delimiter=",", names=True, dtype=float)
    if getattr(arr, "ndim", 0) == 0:
        arr = np.asarray([arr], dtype=arr.dtype)
    pts = np.column_stack([np.asarray(arr["x_mm"], dtype=float), np.asarray(arr["y_mm"], dtype=float)]).astype(float)
    if pts.shape[0] < 3:
        raise ValueError(f"Polygon in {str(path)!r} is too short (N={int(pts.shape[0])}).")
    return pts


def _contour_roi_from_polygon_mm(
    path: Path | str,
    *,
    pad_left_mm: float = 0.0,
    pad_right_mm: float = 0.02,
    pad_bottom_mm: float = 0.0,
    pad_top_mm: float = 0.02,
) -> tuple[float, float, float, float]:
    pts = _read_polygon_mm(path)
    x_min = max(0.0, float(np.min(pts[:, 0])) - float(pad_left_mm))
    x_max = float(np.max(pts[:, 0])) + float(pad_right_mm)
    y_min = max(0.0, float(np.min(pts[:, 1])) - float(pad_bottom_mm))
    y_max = float(np.max(pts[:, 1])) + float(pad_top_mm)
    return float(x_min), float(x_max), float(y_min), float(y_max)


def _crop_mask_to_mm_roi(
    mask_u8: np.ndarray,
    *,
    y_base: int,
    px_size_um: float,
    x_min_mm: float | None = None,
    x_max_mm: float | None = None,
    y_min_mm: float | None = None,
    y_max_mm: float | None = None,
    min_area_px: int | None = None,
) -> np.ndarray:
    out = np.asarray(mask_u8, dtype=np.uint8).copy()
    h, w = out.shape
    if x_min_mm is not None:
        x0 = int(math.ceil((1000.0 * float(x_min_mm)) / float(px_size_um)))
        out[:, : max(0, min(w, x0))] = 0
    if x_max_mm is not None:
        x1 = int(math.ceil((1000.0 * float(x_max_mm)) / float(px_size_um)))
        out[:, max(0, min(w, x1)) :] = 0
    if y_max_mm is not None:
        y_top = int(math.floor(float(y_base) - (1000.0 * float(y_max_mm)) / float(px_size_um)))
        out[: max(0, min(h, y_top)), :] = 0
    if y_min_mm is not None:
        y_bot = int(math.ceil(float(y_base) - (1000.0 * float(y_min_mm)) / float(px_size_um)))
        out[max(0, min(h, y_bot)) :, :] = 0
    if min_area_px is not None and int(min_area_px) > 0:
        out = _largest_component_u8(out, min_area=int(min_area_px))
    return out


def _detect_scale_bar_px(
    gray: np.ndarray,
    *,
    scale_bar_um: float,
    roi_y0_frac: float = 0.65,
    roi_x0_frac: float = 0.55,
    thresh: int = 240,
    max_h_px: int = 35,
) -> tuple[int, float]:
    """
    Detect the horizontal white scale bar and return (bar_width_px, px_size_um).

    Implementation notes
    --------------------
    The Blauert video renders the scale bar and a "250 µm" label in pure white.
    We use a bottom-right ROI with a high threshold to isolate overlays, then
    estimate the bar length via connected components with simple shape filters.
    This is robust for the mmc1 clip and avoids confusing the bright substrate
    band with the scale bar.
    """
    g = np.asarray(gray, dtype=np.uint8)
    h, w = g.shape
    y0 = int(max(0, min(h - 1, math.floor(float(roi_y0_frac) * float(h)))))
    x0 = int(max(0, min(w - 1, math.floor(float(roi_x0_frac) * float(w)))))
    roi = g[y0:h, x0:w]

    _, bw = cv2.threshold(roi, int(thresh), 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    best_w = 0
    # Primary: connected components in the ROI.
    try:
        n, _labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        for i in range(1, n):
            _x, y, wi, hi, area = stats[i]
            if int(area) < 200:
                continue
            # Reject tall blobs (label blocks / glare) and very bottom components
            # that often correspond to the bright substrate band.
            if int(hi) > int(max_h_px):
                continue
            if int(y) >= int(0.9 * float(bw.shape[0])):
                continue
            best_w = max(best_w, int(wi))
    except Exception:
        best_w = 0

    if best_w <= 0:
        raise RuntimeError("Failed to detect scale bar (no suitable component).")

    px_size_um = float(scale_bar_um) / float(best_w)
    return int(best_w), float(px_size_um)


def _detect_substrate_row(gray: np.ndarray, *, mean_thresh: float = 180.0, search_frac: float = 0.25) -> int:
    """
    Detect the bright horizontal substrate line near the bottom of the frame.

    Returns the **top** row index of the bright substrate band (so y=0 corresponds
    to the fluid-facing substrate surface, not the bottom of the bright line).
    """
    g = np.asarray(gray, dtype=np.uint8)
    h = int(g.shape[0])
    y_start = int(max(0, math.floor((1.0 - float(search_frac)) * float(h))))
    row_mean = g.mean(axis=1)
    y_bottom = None
    for y in range(h - 1, y_start - 1, -1):
        if float(row_mean[y]) >= float(mean_thresh):
            y_bottom = int(y)
            break
    if y_bottom is None:
        return int(h - 1)

    y_top = int(y_bottom)
    for y in range(y_bottom - 1, y_start - 1, -1):
        if float(row_mean[y]) >= float(mean_thresh):
            y_top = int(y)
        else:
            break
    return int(y_top)


def _overlay_rects_for_blauert_video(gray: np.ndarray, *, scale_bar_pad_px: int = 12) -> list[tuple[int, int, int, int]]:
    """
    Return rectangular regions that contain video overlays (scale bar + time label).

    These overlays are rendered as pure white pixels and can pollute Otsu
    thresholding / morphology by getting connected to the biofilm mask.
    """
    g = np.asarray(gray, dtype=np.uint8)
    h, w = g.shape

    rects: list[tuple[int, int, int, int]] = []

    # Time label is always in the top-right corner.
    rects.append((int(0.86 * w), 0, w, int(0.22 * h)))

    # Scale-bar region: start from a generous bottom-right ROI. We keep this simple
    # and robust (no fragile text parsing).
    rects.append((int(0.72 * w), int(0.78 * h), w, int(0.96 * h)))

    # Also include a tight rectangle around the detected scale bar line if present.
    try:
        roi_y0 = int(math.floor(0.65 * h))
        roi_x0 = int(math.floor(0.55 * w))
        roi = g[roi_y0:h, roi_x0:w]
        _, bw = cv2.threshold(roi, 240, 255, cv2.THRESH_BINARY)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        n, _labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        best = None
        best_w = 0
        for i in range(1, n):
            x, y, wi, hi, area = stats[i]
            if int(hi) > 35:
                continue
            if int(area) < 200:
                continue
            if int(wi) > int(best_w):
                best_w = int(wi)
                best = (int(x), int(y), int(wi), int(hi))
        if best is not None:
            x, y, wi, hi = best
            x0 = max(0, roi_x0 + x - int(scale_bar_pad_px))
            y0 = max(0, roi_y0 + y - int(scale_bar_pad_px))
            x1 = min(w, roi_x0 + x + wi + int(scale_bar_pad_px))
            # Extend downwards to include the "250 µm" label under the bar.
            y1 = min(h, roi_y0 + y + hi + int(scale_bar_pad_px) + 60)
            rects.append((int(x0), int(y0), int(x1), int(y1)))
    except Exception:
        pass

    # Clip + de-duplicate.
    uniq: set[tuple[int, int, int, int]] = set()
    for x0, y0, x1, y1 in rects:
        x0 = max(0, min(int(w), int(x0)))
        x1 = max(0, min(int(w), int(x1)))
        y0 = max(0, min(int(h), int(y0)))
        y1 = max(0, min(int(h), int(y1)))
        if x1 <= x0 or y1 <= y0:
            continue
        uniq.add((x0, y0, x1, y1))
    return sorted(list(uniq))


def _fill_holes_u8(mask_u8: np.ndarray) -> np.ndarray:
    """Fill holes in a binary mask (0/255) using flood fill from the border."""
    m = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255
    h, w = m.shape

    seed = None
    for (sx, sy) in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if m[int(sy), int(sx)] == 0:
            seed = (int(sx), int(sy))
            break
    if seed is None:
        return m

    flood = m.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, seedPoint=seed, newVal=255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(m, flood_inv)


def _largest_component_u8(mask_u8: np.ndarray, *, min_area: int) -> np.ndarray:
    """Keep only the largest connected component (as 0/255 uint8). Returns empty mask if none."""
    m = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask_u8, dtype=np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA].astype(int)
    idx = 1 + int(np.argmax(areas))
    if int(stats[idx, cv2.CC_STAT_AREA]) < int(min_area):
        return np.zeros_like(mask_u8, dtype=np.uint8)
    return (labels == idx).astype(np.uint8) * 255


def _segment_biofilm(
    frame_bgr: np.ndarray,
    *,
    bottom_trim_px: int,
    blur_sigma: float,
    close_radius_px: int,
    close_iters: int,
    fill_holes: bool,
    min_area_px: int,
    y_base_override: int | None = None,
    fixed_threshold: float | None = None,
    overlay_rects: list[tuple[int, int, int, int]] | None = None,
    overlay_thresh: int = 240,
) -> tuple[np.ndarray, int]:
    """
    Segment the biofilm in a frame.

    Returns (mask_u8, y_base) where mask_u8 is 0/255 and y_base is the substrate-line row.
    """
    img = np.asarray(frame_bgr, dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    overlay_mask = None
    if overlay_rects:
        # Mark overlay pixels in the *original* grayscale (before blur). We will
        # remove them from the binary mask after thresholding so blur does not
        # smear overlay intensity into the biomass mask.
        overlay_mask = np.zeros_like(gray, dtype=bool)
        for x0, y0, x1, y1 in overlay_rects:
            x0 = max(0, min(int(gray.shape[1]), int(x0)))
            x1 = max(0, min(int(gray.shape[1]), int(x1)))
            y0 = max(0, min(int(gray.shape[0]), int(y0)))
            y1 = max(0, min(int(gray.shape[0]), int(y1)))
            if x1 <= x0 or y1 <= y0:
                continue
            overlay_mask[y0:y1, x0:x1] |= (gray[y0:y1, x0:x1] >= int(overlay_thresh))

    y_base = int(y_base_override) if y_base_override is not None else _detect_substrate_row(gray)
    y_cut = max(1, int(y_base) - int(max(0, bottom_trim_px)))
    work = gray[:y_cut, :]

    if float(blur_sigma) > 0.0:
        work = cv2.GaussianBlur(work, (0, 0), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma))

    if fixed_threshold is None:
        _thr, bw = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _thr = float(fixed_threshold)
        _, bw = cv2.threshold(work, _thr, 255, cv2.THRESH_BINARY)

    if overlay_mask is not None:
        bw[np.asarray(overlay_mask[:y_cut, :], dtype=bool)] = 0

    if int(close_radius_px) > 0 and int(close_iters) > 0:
        k = 2 * int(close_radius_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=int(close_iters))

    if bool(fill_holes):
        bw = _fill_holes_u8(bw)
    # Ensure overlays do not re-enter the mask via morphology/hole-filling.
    if overlay_mask is not None:
        bw[np.asarray(overlay_mask[:y_cut, :], dtype=bool)] = 0
    bw = _largest_component_u8(bw, min_area=int(min_area_px))
    return bw, int(y_base)


def _front_x_from_mask(
    mask_u8: np.ndarray,
    *,
    q: float,
    y_band: tuple[int, int] | None = None,
) -> float | None:
    m = np.asarray(mask_u8, dtype=np.uint8)
    if y_band is not None:
        y0, y1 = int(y_band[0]), int(y_band[1])
        y0 = max(0, y0)
        y1 = min(int(m.shape[0]), y1)
        if y1 <= y0:
            return None
        m = m[y0:y1, :]

    _ys, xs = np.nonzero(m > 0)
    if xs.size == 0:
        return None
    return float(np.quantile(xs.astype(float), float(q)))


def _front_x_from_row_runs(
    mask_u8: np.ndarray,
    *,
    y_band: tuple[int, int],
    min_run_px: int,
    q: float,
) -> float | None:
    """
    Estimate x_front in a y-band by looking at *row-wise* leftmost contiguous runs.

    This is more robust than taking a quantile over all pixels when the mask contains
    sparse noise/isolated bright pixels upstream of the biofilm.
    """
    m = (np.asarray(mask_u8, dtype=np.uint8) > 0)
    y0, y1 = int(y_band[0]), int(y_band[1])
    y0 = max(0, y0)
    y1 = min(int(m.shape[0]), y1)
    if y1 <= y0:
        return None

    fronts: list[float] = []
    for yy in range(y0, y1):
        row = m[yy, :]
        xs = np.flatnonzero(row)
        if xs.size == 0:
            continue
        # Find contiguous runs in xs.
        breaks = np.flatnonzero(np.diff(xs) > 1)
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, xs.size - 1]
        for s, e in zip(starts, ends):
            x0_run = int(xs[int(s)])
            x1_run = int(xs[int(e)])
            if (x1_run - x0_run + 1) >= int(min_run_px):
                fronts.append(float(x0_run))
                break

    if not fronts:
        return None
    return float(np.quantile(np.asarray(fronts, dtype=float), float(q)))


def _contour_polygon_mm(
    mask_u8: np.ndarray,
    *,
    y_base: int,
    px_size_um: float,
    simplify_eps_px: float,
) -> np.ndarray:
    """
    Return a closed polygon (N,2) in millimeters (x_mm,y_mm), with y measured upward from substrate.
    """
    m = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No contour found in mask (empty segmentation).")
    cont = max(contours, key=cv2.contourArea)

    if float(simplify_eps_px) > 0.0:
        cont = cv2.approxPolyDP(cont, epsilon=float(simplify_eps_px), closed=True)

    pts = cont.reshape(-1, 2).astype(float)  # (x,y) pixels with y downward
    if pts.shape[0] < 3:
        raise RuntimeError("Contour too short to build a polygon.")
    if not np.allclose(pts[0], pts[-1], rtol=0.0, atol=1.0e-12):
        pts = np.vstack([pts, pts[0]])

    x_mm = (pts[:, 0] * float(px_size_um)) * 1.0e-3
    y_mm = ((float(y_base) - pts[:, 1]) * float(px_size_um)) * 1.0e-3
    return np.column_stack([x_mm, y_mm]).astype(float)


def _polygon_from_matlab_preprocessing(
    path: Path | str,
    *,
    L_um: float = 2000.0,
    shift_um: float = 0.0,
) -> np.ndarray:
    """
    Build the initial biofilm polygon from Dianlei Feng's Matlab preprocessing data.

    This matches `biofilm_preprocessing/data_processing.m`:
      - invert y via `y_max - y`,
      - scale both x and y by `x_max` to a reference image length `L_um`,
      - apply an optional x-shift (the Matlab script uses 500 µm).

    Returns a closed polygon (N,2) in **millimeters** (x_mm,y_mm).
    """
    data = np.loadtxt(str(path), dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected Nx2 coordinates in {str(path)}.")
    vx = np.asarray(data[:, 0], dtype=float).ravel()
    vy = np.asarray(data[:, 1], dtype=float).ravel()
    if vx.size < 3:
        raise ValueError(f"Polygon in {str(path)} is too short (N={int(vx.size)}).")

    x_max = float(np.max(vx))
    y_max = float(np.max(vy))
    if not (x_max > 0.0):
        raise ValueError(f"Invalid x_max={x_max:g} in {str(path)}.")

    vy = float(y_max) - vy
    vx_um = (vx / x_max) * float(L_um) + float(shift_um)
    vy_um = (vy / x_max) * float(L_um)

    x_mm = vx_um * 1.0e-3
    y_mm = vy_um * 1.0e-3
    if not (np.isfinite(x_mm).all() and np.isfinite(y_mm).all()):
        raise ValueError(f"Non-finite polygon coordinates produced from {str(path)}.")

    pts = np.column_stack([x_mm, y_mm]).astype(float)
    if not np.allclose(pts[0], pts[-1], rtol=0.0, atol=1.0e-12):
        pts = np.vstack([pts, pts[0]])
    return pts


def _contour_pts_xy(mask_u8: np.ndarray) -> np.ndarray | None:
    m = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cont = max(contours, key=cv2.contourArea)
    if cont.size == 0:
        return None
    return cont.reshape(-1, 2).astype(float)  # (x,y)


def _front_x_from_contour_pts(
    pts_xy: np.ndarray,
    *,
    y_band: tuple[int, int],
    q: float,
) -> float | None:
    pts = np.asarray(pts_xy, dtype=float)
    y0, y1 = int(y_band[0]), int(y_band[1])
    y0 = max(0, y0)
    y1 = max(y0 + 1, y1)
    sel = (pts[:, 1] >= float(y0)) & (pts[:, 1] < float(y1))
    if not np.any(sel):
        return None
    return float(np.quantile(pts[sel, 0], float(q)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--video",
        type=str,
        default="examples/biofilms/benchmarks/blauert/biofilm_preprocessing/1-s2.0-S0043135418307000-mmc1.mp4",
        help="Path to the supplementary video.",
    )
    ap.add_argument("--scale-bar-um", type=float, default=250.0, help="Scale bar length [um].")
    ap.add_argument("--stride", type=int, default=1, help="Frame stride (1=every frame).")
    ap.add_argument("--t-max", type=float, default=float("nan"), help="Optional cutoff time [s].")
    ap.add_argument("--front-quantile", type=float, default=0.005, help="Quantile used for x_front (robust vs noise).")
    ap.add_argument(
        "--y-levels-um",
        type=str,
        default="",
        help="Comma-separated y-levels [um] for additional x_front(y) tracking.",
    )
    ap.add_argument("--y-band-px", type=int, default=4, help="Half-band [px] around each y-level.")

    # Segmentation knobs
    ap.add_argument(
        "--threshold",
        type=str,
        default="per_frame",
        choices=("frame0", "per_frame"),
        help="Thresholding mode: Otsu per-frame (matches paper) or fixed from frame 0.",
    )
    ap.add_argument("--bottom-trim-px", type=int, default=6, help="Pixels trimmed above substrate line before segmentation.")
    ap.add_argument("--blur-sigma", type=float, default=2.0, help="Gaussian blur sigma for segmentation (0 disables).")
    ap.add_argument(
        "--close-radius-px",
        type=int,
        default=7,
        help=(
            "Morphological closing radius [px]. For --front-method contour this should be large enough to "
            "bridge pores; for pixel-based methods a smaller value may be preferable."
        ),
    )
    ap.add_argument("--close-iters", type=int, default=1, help="Morphological closing iterations (tracking).")
    ap.add_argument("--fill-holes", action=argparse.BooleanOptionalAction, default=False, help="Fill holes in mask (tracking).")
    ap.add_argument("--min-area-px", type=int, default=5000, help="Min area for the kept connected component (tracking).")
    ap.add_argument(
        "--area-min-frac",
        type=float,
        default=0.3,
        help="Reject frames where segmented area < area_min_frac * area0 (guards against video glitches/cuts).",
    )
    ap.add_argument(
        "--area-max-frac",
        type=float,
        default=3.0,
        help="Reject frames where segmented area > area_max_frac * area0 (guards against over-segmentation).",
    )
    ap.add_argument(
        "--front-method",
        type=str,
        default="contour",
        choices=("pixel_quantile", "row_runs", "contour"),
        help="How to extract x_front inside a y-band.",
    )
    ap.add_argument(
        "--front-min-run-px",
        type=int,
        default=12,
        help="Min contiguous run length [px] for --front-method row_runs.",
    )
    ap.add_argument(
        "--front-row-quantile",
        type=float,
        default=0.5,
        help="Quantile over per-row front estimates (row_runs). 0.5=median.",
    )
    ap.add_argument(
        "--front-contour-quantile",
        type=float,
        default=0.0,
        help="Quantile over contour points selected by y-band (contour). 0=min, 0.5=median.",
    )

    # Outputs
    ap.add_argument(
        "--out-csv",
        type=str,
        default="examples/biofilms/benchmarks/blauert/exp_front_displacement_from_video.csv",
        help="Output CSV path.",
    )
    ap.add_argument(
        "--out-polygon",
        type=str,
        default="examples/biofilms/benchmarks/blauert/exp_frame0_polygon_mm.csv",
        help="Output polygon CSV path (frame 0 segmentation).",
    )
    ap.add_argument("--write-polygon", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--polygon-source",
        type=str,
        default="matlab_preprocessing",
        choices=("frame0_mask", "matlab_preprocessing"),
        help="How to produce --out-polygon: contour from frame0 segmentation, or from Matlab preprocessing (recommended).",
    )
    ap.add_argument(
        "--matlab-biofilm-txt",
        type=str,
        default="examples/biofilms/benchmarks/blauert/biofilm_preprocessing/biofilm.txt",
        help="Path to Matlab-preprocessed biofilm contour (used with --polygon-source matlab_preprocessing).",
    )
    ap.add_argument("--matlab-L-um", type=float, default=2000.0, help="Reference image length L [um] in data_processing.m.")
    ap.add_argument("--matlab-shift-um", type=float, default=0.0, help="Optional x-shift applied when mapping Matlab contour [um].")
    ap.add_argument("--poly-simplify-eps-px", type=float, default=2.0, help="Polygon simplification epsilon [px].")

    args = ap.parse_args()

    video_path = Path(str(args.video))
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {str(video_path)}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (fps > 0.0):
            fps = 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame0 = cap.read()
        if not ok or frame0 is None:
            raise RuntimeError("Failed to read frame 0.")

        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        bar_px, px_size_um = _detect_scale_bar_px(gray0, scale_bar_um=float(args.scale_bar_um))
        overlay_rects = _overlay_rects_for_blauert_video(gray0)
        y_base0 = _detect_substrate_row(gray0)
        # Optional: fixed Otsu threshold from frame 0.
        y_cut0 = max(1, int(y_base0) - int(max(0, int(args.bottom_trim_px))))
        work0 = gray0[:y_cut0, :]
        if float(args.blur_sigma) > 0.0:
            work0 = cv2.GaussianBlur(work0, (0, 0), sigmaX=float(args.blur_sigma), sigmaY=float(args.blur_sigma))
        fixed_thr0, _ = cv2.threshold(work0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        y_levels_um = _parse_float_list(str(args.y_levels_um)) if str(args.y_levels_um).strip() else []
        y_band = int(args.y_band_px)
        stride = max(1, int(args.stride))
        t_max = float(args.t_max)

        out_csv = Path(str(args.out_csv))
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        cols = ["t_s", "frame", "x_front_px", "x_front_um", "dx_front_um"]
        for y_um in y_levels_um:
            tag = f"x_front_y{int(round(float(y_um)))}um"
            cols.extend([f"{tag}_px", f"{tag}_um", f"d{tag}_um"])

        rows: list[list[float]] = []
        x0_um: float | None = None
        x0_y_um: dict[float, float] = {}

        # Baseline area from frame 0 segmentation (used to filter out video glitches/cuts).
        thr_mode = str(args.threshold).strip().lower()
        fixed_thr = float(fixed_thr0) if thr_mode == "frame0" else None
        mask0_track, _ = _segment_biofilm(
            frame0,
            bottom_trim_px=int(args.bottom_trim_px),
            blur_sigma=float(args.blur_sigma),
            close_radius_px=int(args.close_radius_px),
            close_iters=int(args.close_iters),
            fill_holes=True if str(args.front_method) == "contour" else bool(args.fill_holes),
            min_area_px=int(args.min_area_px),
            y_base_override=int(y_base0),
            fixed_threshold=fixed_thr,
            overlay_rects=overlay_rects,
        )
        area0 = int(np.count_nonzero(mask0_track > 0))
        area_min = float(args.area_min_frac) * float(area0)
        area_max = float(args.area_max_frac) * float(area0)

        frame_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            if frame_idx % stride != 0:
                ok = cap.grab()
                if not ok:
                    break
                frame_idx += 1
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                break

            t_s = float(frame_idx) / float(fps)
            if np.isfinite(t_max) and t_s > t_max:
                break

            mask_u8, y_base = _segment_biofilm(
                frame,
                bottom_trim_px=int(args.bottom_trim_px),
                blur_sigma=float(args.blur_sigma),
                close_radius_px=int(args.close_radius_px),
                close_iters=int(args.close_iters),
                fill_holes=True if str(args.front_method) == "contour" else bool(args.fill_holes),
                min_area_px=int(args.min_area_px),
                y_base_override=int(y_base0),
                fixed_threshold=fixed_thr,
                overlay_rects=overlay_rects,
            )
            area = int(np.count_nonzero(mask_u8 > 0))
            if area0 > 0 and (float(area) < float(area_min) or float(area) > float(area_max)):
                frame_idx += 1
                continue

            x_front_px = _front_x_from_mask(mask_u8, q=float(args.front_quantile))
            if x_front_px is None:
                frame_idx += 1
                continue

            x_front_um = float(x_front_px) * float(px_size_um)
            if x0_um is None:
                x0_um = float(x_front_um)
            dx_front_um = float(x_front_um) - float(x0_um)

            row: list[float] = [t_s, float(frame_idx), float(x_front_px), float(x_front_um), float(dx_front_um)]

            pts_xy = _contour_pts_xy(mask_u8) if str(args.front_method) == "contour" else None

            for y_um in y_levels_um:
                y_px = float(y_base) - (float(y_um) / float(px_size_um))
                y0 = int(math.floor(y_px - float(y_band)))
                y1 = int(math.ceil(y_px + float(y_band))) + 1
                if str(args.front_method) == "pixel_quantile":
                    x_y_px = _front_x_from_mask(mask_u8, q=float(args.front_quantile), y_band=(y0, y1))
                else:
                    if str(args.front_method) == "row_runs":
                        x_y_px = _front_x_from_row_runs(
                            mask_u8,
                            y_band=(y0, y1),
                            min_run_px=int(args.front_min_run_px),
                            q=float(args.front_row_quantile),
                        )
                    else:
                        if pts_xy is None:
                            x_y_px = None
                        else:
                            x_y_px = _front_x_from_contour_pts(
                                pts_xy,
                                y_band=(y0, y1),
                                q=float(args.front_contour_quantile),
                            )
                if x_y_px is None:
                    row.extend([float("nan"), float("nan"), float("nan")])
                    continue
                x_y_um = float(x_y_px) * float(px_size_um)
                if y_um not in x0_y_um:
                    x0_y_um[y_um] = float(x_y_um)
                row.extend([float(x_y_px), float(x_y_um), float(x_y_um) - float(x0_y_um[y_um])])

            rows.append(row)
            frame_idx += 1

        arr = np.asarray(rows, dtype=float)
        with out_csv.open("w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for r in arr:
                f.write(",".join([f"{x:.8g}" for x in r]) + "\n")

        print(f"[blauert] video={str(video_path)}")
        print(f"[blauert] fps={fps:.3f}  frames={n_frames}  processed={arr.shape[0]}  stride={stride}")
        print(
            "[blauert] scale_bar="
            f"{float(args.scale_bar_um):g} um  bar_px={bar_px}  px_size={px_size_um:.6g} um/px"
        )
        print(f"[blauert] wrote {str(out_csv)}")

        if bool(args.write_polygon):
            poly_src = str(args.polygon_source).strip().lower()
            if poly_src == "matlab_preprocessing":
                poly = _polygon_from_matlab_preprocessing(
                    Path(str(args.matlab_biofilm_txt)),
                    L_um=float(args.matlab_L_um),
                    shift_um=float(args.matlab_shift_um),
                )
            elif poly_src == "frame0_mask":
                mask0_u8, y_base0 = _segment_biofilm(
                    frame0,
                    bottom_trim_px=int(args.bottom_trim_px),
                    blur_sigma=float(args.blur_sigma),
                    close_radius_px=max(7, int(args.close_radius_px)),
                    close_iters=max(1, int(args.close_iters)),
                    fill_holes=True,
                    min_area_px=int(args.min_area_px),
                    y_base_override=int(y_base0),
                    fixed_threshold=fixed_thr,
                    overlay_rects=overlay_rects,
                )
                poly = _contour_polygon_mm(
                    mask0_u8,
                    y_base=int(y_base0),
                    px_size_um=float(px_size_um),
                    simplify_eps_px=float(args.poly_simplify_eps_px),
                )
            else:
                raise ValueError(f"Unknown --polygon-source={args.polygon_source!r}.")
            out_poly = Path(str(args.out_polygon))
            out_poly.parent.mkdir(parents=True, exist_ok=True)
            with out_poly.open("w", encoding="utf-8") as f:
                f.write("x_mm,y_mm\n")
                for x_mm, y_mm in poly:
                    f.write(f"{x_mm:.9f},{y_mm:.9f}\n")
            print(f"[blauert] wrote {str(out_poly)}  (source={poly_src}, N={poly.shape[0]})")

    finally:
        cap.release()


if __name__ == "__main__":
    main()
