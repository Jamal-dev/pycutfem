#!/usr/bin/env python3
"""
Extract a publishable t=0 biofilm geometry from Li et al. (2020) Video S1.

Goals (Lie benchmark)
---------------------
* Use the embedded 100 µm scale bar as the source of truth for pixel->meter scaling.
* Straighten the biofilm base on the rigid support and anchor the origin at the paper's ⊗:
    - (x,y) = (0,0) at the *left* end of the support top edge
    - base is a straight segment from x=0 to x=block_w (default 1 mm)
* Remove small fissures/imperfections (Fig. 5(b) intent) while retaining macroscopic features
  such as the upper-right hump.

Outputs
-------
* CSV polygon in mm: x_mm,y_mm with first point repeated at the end (closed)
* Debug overlays (PNG) in --debug-dir
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _parse_crop(crop: str) -> tuple[int, int, int, int] | None:
    crop = str(crop).strip()
    if not crop:
        return None
    parts = [p.strip() for p in crop.split(",")]
    if len(parts) != 4:
        raise ValueError("--crop must be 'x0,y0,w,h'")
    x0, y0, w, h = (int(float(p)) for p in parts)
    if w <= 0 or h <= 0:
        raise ValueError("--crop requires w>0 and h>0")
    return x0, y0, w, h


def _parse_roi_frac(roi: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in str(roi).strip().split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("--scalebar-roi-frac must be 'x0,y0,x1,y1' (fractions in [0,1])")
    x0, y0, x1, y1 = (float(p) for p in parts)
    x0, y0 = float(np.clip(x0, 0.0, 1.0)), float(np.clip(y0, 0.0, 1.0))
    x1, y1 = float(np.clip(x1, 0.0, 1.0)), float(np.clip(y1, 0.0, 1.0))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("--scalebar-roi-frac requires x1>x0 and y1>y0")
    return float(x0), float(y0), float(x1), float(y1)


def _parse_pair(pair: str) -> tuple[float, float]:
    parts = [p.strip() for p in str(pair).strip().split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Expected 'a,b'")
    return float(parts[0]), float(parts[1])


def _top_envelope_xy(points_xy: np.ndarray, *, x_min: float, x_max: float, n_bins: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a simple top envelope y_top(x) by binning in x and taking max y in each bin.
    Returns (x_centers, y_top) with NaNs for empty bins.
    """
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError("points_xy must have shape (N,2), N>=3")
    x = pts[:, 0].astype(float)
    y = pts[:, 1].astype(float)
    x_min = float(x_min)
    x_max = float(x_max)
    if not (np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min):
        raise ValueError("Invalid x_min/x_max")
    n_bins = int(max(20, int(n_bins)))
    edges = np.linspace(x_min, x_max, num=n_bins + 1, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y_top = np.full(n_bins, -np.inf, dtype=float)
    bin_idx = np.digitize(x, edges) - 1
    ok = (bin_idx >= 0) & (bin_idx < n_bins) & np.isfinite(y)
    if not bool(np.any(ok)):
        return centers, np.full(n_bins, np.nan, dtype=float)
    # Max per bin.
    np.maximum.at(y_top, bin_idx[ok], y[ok])
    y_top[np.isneginf(y_top)] = np.nan
    return centers, y_top


def _right_envelope_x_of_y(points_xy: np.ndarray, *, y_min: float, y_max: float, n_bins: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a right envelope x_right(y) by binning in y and taking max x in each bin.
    Returns (y_centers, x_right) with NaNs for empty bins.
    """
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError("points_xy must have shape (N,2), N>=3")
    x = pts[:, 0].astype(float)
    y = pts[:, 1].astype(float)
    y_min = float(y_min)
    y_max = float(y_max)
    if not (np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min):
        raise ValueError("Invalid y_min/y_max")
    n_bins = int(max(20, int(n_bins)))
    edges = np.linspace(y_min, y_max, num=n_bins + 1, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    x_right = np.full(n_bins, -np.inf, dtype=float)
    bin_idx = np.digitize(y, edges) - 1
    ok = (bin_idx >= 0) & (bin_idx < n_bins) & np.isfinite(x)
    if not bool(np.any(ok)):
        return centers, np.full(n_bins, np.nan, dtype=float)
    np.maximum.at(x_right, bin_idx[ok], x[ok])
    x_right[np.isneginf(x_right)] = np.nan
    return centers, x_right


def _interp_nan_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    x = np.arange(y.size, dtype=float)
    ok = np.isfinite(y)
    if not bool(np.any(ok)):
        return y
    out = y.copy()
    out[~ok] = np.interp(x[~ok], x[ok], y[ok])
    return out


def _detect_scalebar_m_per_px(
    img_bgr: np.ndarray,
    *,
    scalebar_um: float,
    roi_frac: tuple[float, float, float, float],
    thr: int,
    kernel: int,
    open_iters: int,
    close_iters: int,
) -> tuple[float, dict]:
    h, w = img_bgr.shape[:2]
    x0f, y0f, x1f, y1f = (float(v) for v in roi_frac)
    x0 = int(round(x0f * float(w)))
    y0 = int(round(y0f * float(h)))
    x1 = int(round(x1f * float(w)))
    y1 = int(round(y1f * float(h)))
    x0 = int(np.clip(x0, 0, w - 1))
    y0 = int(np.clip(y0, 0, h - 1))
    x1 = int(np.clip(x1, x0 + 1, w))
    y1 = int(np.clip(y1, y0 + 1, h))

    roi = img_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bw = (gray >= int(thr)).astype(np.uint8) * 255
    ksz = max(1, int(kernel))
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (ksz, ksz))
    if int(open_iters) > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker, iterations=int(open_iters))
    if int(close_iters) > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker, iterations=int(close_iters))

    on = (bw > 0).astype(np.uint8)

    def _max_run_1d(arr_1d: np.ndarray) -> int:
        max_run = 0
        i = 0
        n = int(arr_1d.size)
        while i < n:
            if int(arr_1d[i]) == 0:
                i += 1
                continue
            i0 = i
            while i < n and int(arr_1d[i]) != 0:
                i += 1
            max_run = max(max_run, int(i - i0))
        return int(max_run)

    max_h = max(_max_run_1d(on[yy, :]) for yy in range(on.shape[0]))
    max_v = max(_max_run_1d(on[:, xx]) for xx in range(on.shape[1]))
    run_px = int(max(max_h, max_v))
    if run_px < 10:
        raise RuntimeError(f"Scale-bar detection failed (run_px={run_px}). Try adjusting threshold/ROI.")

    m_per_px = (float(scalebar_um) * 1.0e-6) / float(run_px)
    meta = {
        "roi_px": (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
        "thr": int(thr),
        "kernel": int(ksz),
        "open_iters": int(open_iters),
        "close_iters": int(close_iters),
        "max_h_run_px": int(max_h),
        "max_v_run_px": int(max_v),
        "run_px": int(run_px),
        "m_per_px": float(m_per_px),
    }
    return float(m_per_px), meta


def _extract_component_mask(
    img_bgr: np.ndarray,
    *,
    crop: tuple[int, int, int, int] | None,
    clahe: bool,
    blur_ksize: int,
    thr_mode: str,
    thr: float,
    invert: bool,
    kernel: int,
    close_iters: int,
    open_iters: int,
    min_area: int,
    component_index: int,
) -> tuple[np.ndarray, dict]:
    x0 = y0 = 0
    img = img_bgr
    if crop is not None:
        x0, y0, w, h = crop
        img = img_bgr[y0 : y0 + h, x0 : x0 + w].copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if bool(clahe):
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = c.apply(gray)

    if int(blur_ksize) and int(blur_ksize) > 1:
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    if str(thr_mode) == "otsu":
        thr_val, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thr_val = float(thr)
        mask = (gray > float(thr_val)).astype(np.uint8) * 255

    if bool(invert):
        mask = 255 - mask

    ksz = max(1, int(kernel))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    if int(close_iters) > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=int(close_iters))
    if int(open_iters) > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, iterations=int(open_iters))

    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for lab in range(1, int(n)):
        bx, by, bw, bh, area = stats[lab].tolist()
        if int(area) < int(min_area):
            continue
        comps.append((int(area), int(lab), (int(bx), int(by), int(bw), int(bh))))
    comps.sort(reverse=True, key=lambda t: t[0])
    if not comps:
        raise RuntimeError("No components after filtering.")

    idx = int(component_index)
    if not (0 <= idx < len(comps)):
        raise ValueError(f"--component-index {idx} out of range; have {len(comps)} components after filtering.")
    area, chosen_lab, bbox = comps[idx]
    chosen_mask = (labels == int(chosen_lab)).astype(np.uint8) * 255
    meta = {
        "crop": crop,
        "x0": int(x0),
        "y0": int(y0),
        "thr_val": float(thr_val),
        "components": int(len(comps)),
        "chosen_index": int(idx),
        "chosen_area": int(area),
        "chosen_bbox": tuple(int(v) for v in bbox),
        "gray": gray,
        "mask": mask,
    }
    return chosen_mask, meta


def _auto_base_from_mask(
    chosen_mask_u8: np.ndarray,
    *,
    tol_px: float,
    base_method: str,
    base_quantile: float,
    min_col_fg: int,
    force_base_y_px: float | None,
) -> tuple[float, int]:
    """
    Return (base_y_px, min_col_fg_used) in cropped coords.
    """
    mask = chosen_mask_u8.astype(np.uint8) > 0
    h, _w = mask.shape
    tol_px = float(max(1.0, float(tol_px)))

    col_fg = np.sum(mask, axis=0).astype(int)
    if int(min_col_fg) <= 0:
        min_col_fg = max(3, int(round(0.01 * float(h))))
    valid_cols = col_fg >= int(min_col_fg)
    if not bool(np.any(valid_cols)):
        valid_cols = np.any(mask, axis=0)

    has_fg = np.any(mask, axis=0)
    rev_idx = np.argmax(mask[::-1, :], axis=0)
    y_bottom = (h - 1) - rev_idx
    y_bottom = y_bottom.astype(float)
    y_bottom[~has_fg] = np.nan
    y_bottom[~valid_cols] = np.nan
    yv = y_bottom[np.isfinite(y_bottom)].astype(int)
    if yv.size < 10:
        raise RuntimeError("Not enough columns to auto-detect a base.")

    if force_base_y_px is not None and bool(np.isfinite(float(force_base_y_px))):
        base_y = float(force_base_y_px)
    else:
        method = str(base_method).strip().lower()
        if method == "mask_quantile":
            q = float(np.clip(float(base_quantile), 0.0, 1.0))
            y0 = int(np.quantile(yv.astype(float), q))
        else:
            counts = np.bincount(yv, minlength=h)
            win = 2 * int(np.ceil(float(tol_px))) + 1
            if win >= 3:
                counts_s = np.convolve(counts, np.ones(win, dtype=int), mode="same")
                y0 = int(np.argmax(counts_s))
            else:
                y0 = int(np.argmax(counts))
        band = int(max(1, int(np.ceil(float(tol_px)))))
        sel = yv[np.abs(yv - y0) <= band]
        base_y = float(np.median(sel)) if sel.size else float(y0)
    return float(base_y), int(min_col_fg)


def _estimate_base_xc_from_mask_band(
    chosen_mask_u8: np.ndarray,
    *,
    base_y_px: float,
    tol_px: float,
    min_col_fg: int,
    span_height_frac: float,
) -> float:
    mask = chosen_mask_u8.astype(np.uint8) > 0
    h, w = mask.shape
    tol_px = float(max(1.0, float(tol_px)))
    base_y_i = int(np.clip(int(round(float(base_y_px))), 0, h - 1))

    col_fg = np.sum(mask, axis=0).astype(int)
    if int(min_col_fg) <= 0:
        min_col_fg = max(3, int(round(0.01 * float(h))))
    valid_cols = col_fg >= int(min_col_fg)
    if not bool(np.any(valid_cols)):
        valid_cols = np.any(mask, axis=0)

    y_lo = int(np.clip(int(np.floor(float(base_y_i) - float(tol_px))), 0, h - 1))
    y_hi = int(np.clip(int(np.ceil(float(base_y_i) + float(tol_px))), 0, h - 1))
    band_rows = mask[y_lo : y_hi + 1, :]
    col_has = np.any(band_rows, axis=0) & valid_cols

    span_height_frac = float(max(0.0, float(span_height_frac)))
    if span_height_frac > 0.0:
        has_fg_col = np.any(mask, axis=0)
        y_top = np.full(w, np.nan, dtype=float)
        y_top[has_fg_col] = np.argmax(mask[:, has_fg_col], axis=0).astype(float)
        heights = float(base_y_i) - y_top
        heights[~np.isfinite(heights)] = 0.0
        h_max = float(np.max(heights)) if np.any(np.isfinite(heights)) else 0.0
        h_thr = max(10.0, span_height_frac * h_max)
        col_has = col_has & (heights >= h_thr)

    xs = np.nonzero(col_has)[0].astype(float)
    if xs.size < 2:
        raise RuntimeError("Could not estimate base x-center (too few columns in base band).")
    return float(np.median(xs))


def _straighten_base_in_mask(
    chosen_mask_u8: np.ndarray,
    *,
    base_y_px: float,
    x_left_px: int,
    x_right_px: int,
    fill_columns: bool,
    clip_to_span: bool,
) -> tuple[np.ndarray, int, int, int]:
    mask = chosen_mask_u8.astype(np.uint8) > 0
    h, w = mask.shape

    base_y_i = int(np.clip(int(round(float(base_y_px))), 0, h - 1))
    x_left = int(np.clip(int(x_left_px), 0, w - 1))
    x_right = int(np.clip(int(x_right_px), 0, w - 1))
    if x_left > x_right:
        x_left, x_right = x_right, x_left

    mask_clip = mask.copy()
    if base_y_i + 1 < h:
        mask_clip[base_y_i + 1 :, :] = False

    mask_out = mask_clip.copy()
    if bool(fill_columns):
        # "Column fill" is useful for deformation tracking (intersection-based dx),
        # but it can over-extend the geometry laterally if there is speckle in a column.
        for x in range(x_left, x_right + 1):
            col = mask_clip[:, x]
            if not bool(np.any(col)):
                continue
            ys = np.nonzero(col)[0]
            if ys.size == 0:
                continue
            d = np.diff(ys)
            breaks = np.nonzero(d > 1)[0]
            starts = np.concatenate([np.array([0], dtype=int), breaks + 1])
            ends = np.concatenate([breaks + 1, np.array([ys.size], dtype=int)])
            run_lens = (ends - starts).astype(int)
            k_best = int(np.argmax(run_lens))
            y_top = int(ys[int(starts[k_best])])
            mask_out[y_top : base_y_i + 1, x] = True

    # Enforce a perfectly straight base line on the support.
    mask_out[base_y_i, :] = False
    mask_out[base_y_i, x_left : x_right + 1] = True

    if bool(clip_to_span):
        if x_left > 0:
            mask_out[:, :x_left] = False
        if x_right + 1 < w:
            mask_out[:, x_right + 1 :] = False

    return (mask_out.astype(np.uint8) * 255), int(base_y_i), int(x_left), int(x_right)


def _smooth_1d(x: np.ndarray, *, window: int, polyorder: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    window = int(window)
    polyorder = int(polyorder)
    if window <= 1 or x.size < max(5, window):
        return x
    if window % 2 == 0:
        window += 1
    window = min(window, x.size - (1 - x.size % 2))  # keep odd <= size
    if window < polyorder + 2:
        return x
    try:
        from scipy.signal import savgol_filter  # type: ignore

        return np.asarray(savgol_filter(x, window_length=window, polyorder=polyorder, mode="interp"), dtype=float)
    except Exception:
        # Fallback: simple moving average (edge-padded).
        k = np.ones((window,), dtype=float) / float(window)
        pad = window // 2
        x_pad = np.pad(x, (pad, pad), mode="edge")
        return np.convolve(x_pad, k, mode="valid")


def _smooth_polyline_xy(
    xy: np.ndarray,
    *,
    ds_target: float,
    window_pts: int,
    polyorder: int,
) -> np.ndarray:
    """
    Resample a closed polyline by arclength and smooth x(s), y(s) independently.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 4:
        return xy
    if float(np.linalg.norm(xy[0] - xy[-1])) > 1.0e-12:
        xy = np.vstack([xy, xy[:1, :]])

    dxy = np.diff(xy, axis=0)
    ds = np.sqrt(np.sum(dxy**2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0.0:
        return xy

    ds_target = float(max(1.0e-6, float(ds_target)))
    n = int(max(80, int(round(total / ds_target))))
    s_new = np.linspace(0.0, total, num=n, endpoint=False)

    x_new = np.interp(s_new, s, xy[:, 0])
    y_new = np.interp(s_new, s, xy[:, 1])

    x_s = _smooth_1d(x_new, window=int(window_pts), polyorder=int(polyorder))
    y_s = _smooth_1d(y_new, window=int(window_pts), polyorder=int(polyorder))

    out = np.column_stack([x_s, y_s])
    out = np.vstack([out, out[:1, :]])
    return out


def _smooth_open_polyline_xy(
    xy: np.ndarray,
    *,
    ds_target: float,
    window_pts: int,
    polyorder: int,
) -> np.ndarray:
    """
    Resample an open polyline by arclength and smooth x(s), y(s) independently.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 4:
        return xy
    dxy = np.diff(xy, axis=0)
    ds = np.sqrt(np.sum(dxy**2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0.0:
        return xy

    ds_target = float(max(1.0e-6, float(ds_target)))
    n = int(max(80, int(round(total / ds_target))))
    s_new = np.linspace(0.0, total, num=n, endpoint=True)
    x_new = np.interp(s_new, s, xy[:, 0])
    y_new = np.interp(s_new, s, xy[:, 1])

    x_s = _smooth_1d(x_new, window=int(window_pts), polyorder=int(polyorder))
    y_s = _smooth_1d(y_new, window=int(window_pts), polyorder=int(polyorder))
    return np.column_stack([x_s, y_s])


def _rotate_closed_poly_to_anchor(poly: np.ndarray, anchor_xy: tuple[float, float]) -> np.ndarray:
    poly = np.asarray(poly, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 4:
        return poly
    if float(np.linalg.norm(poly[0] - poly[-1])) > 1.0e-12:
        poly = np.vstack([poly, poly[:1, :]])
    pts = poly[:-1, :]
    ax, ay = (float(anchor_xy[0]), float(anchor_xy[1]))
    d2 = (pts[:, 0] - ax) ** 2 + (pts[:, 1] - ay) ** 2
    i0 = int(np.argmin(d2))
    pts = np.vstack([pts[i0:, :], pts[:i0, :]])
    return np.vstack([pts, pts[:1, :]])


def _longest_run_1d(indices: np.ndarray) -> tuple[int, int] | None:
    """
    Given sorted integer indices, return (start_idx,end_idx) into the indices array (end exclusive)
    for the longest contiguous run (gap<=1). None if empty.
    """
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return None
    if indices.size == 1:
        return (0, 1)
    d = np.diff(indices)
    breaks = np.nonzero(d > 1)[0]
    starts = np.concatenate([np.array([0], dtype=int), breaks + 1])
    ends = np.concatenate([breaks + 1, np.array([indices.size], dtype=int)])
    run_lens = (ends - starts).astype(int)
    k_best = int(np.argmax(run_lens))
    return (int(starts[k_best]), int(ends[k_best]))


def _column_top_from_longest_run(
    mask_bool: np.ndarray,
    *,
    base_y_i: int,
    x_left_i: int,
    x_right_i: int,
    run_quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each column x in [x_left_i, x_right_i], find the top y (smallest y) of the
    *longest* contiguous foreground run up to base_y_i.

    This rejects thin diagonal artifacts (short runs) while keeping true top curvature.
    Returns (xs, y_top_px) with NaNs for missing columns.
    """
    mask_bool = (mask_bool.astype(np.uint8) > 0).astype(bool)
    h, w = mask_bool.shape
    base_y_i = int(np.clip(int(base_y_i), 0, h - 1))
    x_left_i = int(np.clip(int(x_left_i), 0, w - 1))
    x_right_i = int(np.clip(int(x_right_i), 0, w - 1))
    if x_right_i < x_left_i:
        x_left_i, x_right_i = x_right_i, x_left_i

    xs = np.arange(int(x_left_i), int(x_right_i) + 1, dtype=int)
    y_top_px = np.full(xs.size, np.nan, dtype=float)
    rq = float(np.clip(float(run_quantile), 0.0, 0.30))
    for k, xx in enumerate(xs):
        col = mask_bool[: base_y_i + 1, int(xx)]
        if not bool(np.any(col)):
            continue
        ys = np.nonzero(col)[0].astype(int)
        run = _longest_run_1d(ys)
        if run is None:
            continue
        i0, i1 = run
        run_len = int(max(1, int(i1 - i0)))
        j = int(i0 + int(round(rq * float(max(0, run_len - 1)))))
        j = int(np.clip(j, i0, i1 - 1))
        y_top_px[k] = float(ys[int(j)])
    return xs.astype(float), y_top_px


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract smoothed t=0 geometry from Video S1 (Lie benchmark).")
    ap.add_argument(
        "--video",
        type=str,
        default="examples/biofilms/benchmarks/lie/additional_data/bit27491-sup-0001-si_v1.avi",
        help="Path to Video S1 AVI file.",
    )
    ap.add_argument("--frame", type=int, default=0, help="Frame index (use 0 for the undeformed geometry).")
    ap.add_argument("--crop", type=str, default="", help="Optional crop: 'x0,y0,w,h' in pixels.")

    ap.add_argument(
        "--clahe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply CLAHE contrast enhancement (recommended for Video S1).",
    )
    ap.add_argument("--blur-ksize", type=int, default=5, help="Gaussian blur ksize (odd). 0 disables.")
    ap.add_argument("--thr-mode", type=str, default="otsu", choices=("otsu", "fixed"))
    ap.add_argument("--thr", type=float, default=80.0, help="Threshold for --thr-mode fixed (0..255).")
    ap.add_argument("--invert", action="store_true", help="Invert the thresholded mask.")
    ap.add_argument("--kernel", type=int, default=5, help="Morphology kernel size.")
    ap.add_argument("--close-iters", type=int, default=2, help="Morphological closing iterations.")
    ap.add_argument("--open-iters", type=int, default=0, help="Morphological opening iterations.")
    ap.add_argument("--min-area", type=int, default=5000, help="Filter components smaller than this.")
    ap.add_argument("--component-index", type=int, default=0, help="Component index after sorting by area.")

    ap.add_argument("--base-y-px", type=float, default=float("nan"), help="Manual base y (full-frame coords).")
    ap.add_argument("--base-method", type=str, default="mask_mode", choices=("mask_mode", "mask_quantile"))
    ap.add_argument("--base-quantile", type=float, default=0.95)
    ap.add_argument("--base-tol-px", type=float, default=6.0)
    ap.add_argument("--base-min-col-fg", type=int, default=0)
    ap.add_argument("--base-span-height-frac", type=float, default=0.2)

    ap.add_argument("--block-w", type=float, default=1.0e-3, help="Support width [m] (used to set straightened base span).")
    ap.add_argument("--scalebar-um", type=float, default=100.0)
    ap.add_argument("--scalebar-roi-frac", type=str, default="0.75,0.75,1.0,1.0")
    ap.add_argument("--scalebar-thr", type=int, default=200)
    ap.add_argument("--scalebar-kernel", type=int, default=3)
    ap.add_argument("--scalebar-open-iters", type=int, default=1)
    ap.add_argument("--scalebar-close-iters", type=int, default=1)

    ap.add_argument("--row-min-fg-frac", type=float, default=0.2, help="Ignore rows with < frac*max_row_fg when finding the top (artifact rejection).")
    ap.add_argument(
        "--top-min-height-frac",
        type=float,
        default=0.05,
        help="Columns with height < frac*Hb are treated as missing and filled by edge-hold/interp.",
    )
    ap.add_argument("--top-smooth-window-mm", type=float, default=0.06, help="SavGol/moving-average smoothing window for y_top(x) [mm].")
    ap.add_argument("--top-smooth-polyorder", type=int, default=3, help="SavGol polyorder for y_top(x).")
    ap.add_argument("--n-verts-top", type=int, default=240, help="Target number of vertices on the top boundary (x=0..block_w).")
    ap.add_argument(
        "--polygon-method",
        type=str,
        default="contour",
        choices=("composed", "contour", "topcurve"),
        help="How to form the publishable polygon. 'contour' is recommended for Fig.5(b)-like shapes with side curvature + upper-right hump.",
    )
    ap.add_argument("--rdp-eps-mm", type=float, default=0.010, help="Contour simplification epsilon [mm] (only for --polygon-method contour).")
    ap.add_argument("--smooth-s-ds-mm", type=float, default=0.004, help="Arclength resampling spacing [mm] (only for --polygon-method contour).")
    ap.add_argument("--smooth-s-window-mm", type=float, default=0.030, help="Smoothing window along arclength [mm] (only for --polygon-method contour).")
    ap.add_argument("--smooth-s-polyorder", type=int, default=3, help="SavGol polyorder for (x(s),y(s)) smoothing (only for --polygon-method contour).")
    ap.add_argument("--n-verts-contour", type=int, default=260, help="Target number of vertices for contour polygons (after smoothing).")
    ap.add_argument(
        "--contour-base-trim-mm",
        type=float,
        default=0.08,
        help="Drop interior contour points with y below this threshold [mm] to suppress base-corner jagged loops.",
    )
    ap.add_argument(
        "--contour-right-profile-correct",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For contour polygons, replace the far-right flank by a row-quantile mask profile to retain right-side curvature.",
    )
    ap.add_argument(
        "--contour-right-profile-quantile",
        type=float,
        default=0.94,
        help="Row quantile used for right-flank correction in contour mode (smaller values bend more inward).",
    )
    ap.add_argument(
        "--contour-right-profile-xstart-mm",
        type=float,
        default=0.72,
        help="Apply right-flank correction only for points with x >= this value [mm].",
    )
    ap.add_argument(
        "--contour-use-convex-hull",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the convex hull of the cleaned mask contour to remove concave notches (Fig.5(b)-style smoothing).",
    )
    ap.add_argument("--n-verts-side", type=int, default=140, help="Target number of vertices on each side boundary (only for --polygon-method composed).")
    ap.add_argument("--side-smooth-window-mm", type=float, default=0.05, help="Smoothing window for xL(y)/xR(y) [mm] (only for --polygon-method composed).")
    ap.add_argument("--side-smooth-polyorder", type=int, default=3, help="SavGol polyorder for side smoothing (only for --polygon-method composed).")
    ap.add_argument("--side-x-quantile-left", type=float, default=0.02, help="Per-row quantile for xL(y) (only for --polygon-method composed).")
    ap.add_argument("--side-x-quantile-right", type=float, default=0.98, help="Per-row quantile for xR(y) (only for --polygon-method composed).")
    ap.add_argument("--top-y-quantile", type=float, default=0.02, help="Per-column quantile for yTop(x) (only for --polygon-method composed/topcurve).")
    ap.add_argument(
        "--mask-clean-kernel",
        type=int,
        default=9,
        help="Extra morphology kernel for post-straighten cleanup (to remove fissures).",
    )
    ap.add_argument(
        "--mask-clean-close",
        type=int,
        default=0,
        help="Extra closing iterations after straightening. Use 0 to preserve sharp details (e.g., upper-right point).",
    )
    ap.add_argument("--mask-clean-open", type=int, default=0, help="Extra opening iterations after straightening.")
    ap.add_argument(
        "--cap-run-quantile",
        type=float,
        default=0.10,
        help="When clipping thin artifacts for contour mode, use this quantile within the longest vertical foreground run (0 keeps the absolute top).",
    )
    ap.add_argument(
        "--base-fill-columns",
        action="store_true",
        help="Fill each support column up to the detected top. Improves dx intersection tracking, but can distort the publishable geometry.",
    )
    ap.add_argument(
        "--straighten-clip-to-span",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Hard-clip straightened mask to [x_left, x_right]. Keep disabled to preserve natural side curvature.",
    )
    ap.add_argument(
        "--fig5b-traced-csv",
        type=str,
        default="",
        help="Optional Fig.5(b) traced boundary CSV in mm (x_mm,y_mm) to compare top-right curvature against.",
    )
    ap.add_argument(
        "--fig5b-right-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When --fig5b-traced-csv is provided, use its right-envelope as a prior to recover curved right flank.",
    )
    ap.add_argument(
        "--fig5b-right-prior-xstart-mm",
        type=float,
        default=0.62,
        help="Apply Fig5b right-envelope prior only for contour points with x >= this [mm].",
    )
    ap.add_argument(
        "--fig5b-right-prior-ymin-frac",
        type=float,
        default=0.15,
        help="Apply Fig5b right-envelope prior only above this normalized height y/H_b.",
    )
    ap.add_argument(
        "--fig5b-compare-xrange-mm",
        type=str,
        default="0.60,0.86",
        help="x-range [mm] used for Fig.5(b) top-envelope comparison, as 'x0,x1'.",
    )

    ap.add_argument(
        "--out-poly-mm-csv",
        type=str,
        default="examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_scalebar_smooth.csv",
    )
    ap.add_argument("--debug-dir", type=str, default="out/_lie_exp_geom_debug_s1_frame0")
    args = ap.parse_args()

    crop = _parse_crop(str(args.crop))
    debug_dir = Path(str(args.debug_dir))
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(str(args.out_poly_mm_csv))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.frame))
    ok, img_full = cap.read()
    cap.release()
    if not ok or img_full is None:
        raise RuntimeError(f"Could not read frame={int(args.frame)} from {args.video}")

    cv2.imwrite(str(debug_dir / "frame.png"), img_full)

    # Scale bar (source of truth).
    roi_frac = _parse_roi_frac(str(args.scalebar_roi_frac))
    m_per_px, scalebar_meta = _detect_scalebar_m_per_px(
        img_full,
        scalebar_um=float(args.scalebar_um),
        roi_frac=roi_frac,
        thr=int(args.scalebar_thr),
        kernel=int(args.scalebar_kernel),
        open_iters=int(args.scalebar_open_iters),
        close_iters=int(args.scalebar_close_iters),
    )

    # Segment + select component.
    mask, meta = _extract_component_mask(
        img_full,
        crop=crop,
        clahe=bool(args.clahe),
        blur_ksize=int(args.blur_ksize),
        thr_mode=str(args.thr_mode),
        thr=float(args.thr),
        invert=bool(args.invert),
        kernel=int(args.kernel),
        close_iters=int(args.close_iters),
        open_iters=int(args.open_iters),
        min_area=int(args.min_area),
        component_index=int(args.component_index),
    )
    cv2.imwrite(str(debug_dir / "chosen_mask.png"), mask)

    x0 = int(meta["x0"])
    y0 = int(meta["y0"])

    # Base y (cropped coords).
    base_y_full = float(args.base_y_px)
    if np.isfinite(base_y_full) and crop is not None:
        base_y_full = float(base_y_full) - float(crop[1])
    base_y_auto, min_col_fg_used = _auto_base_from_mask(
        mask,
        tol_px=float(args.base_tol_px),
        base_method=str(args.base_method),
        base_quantile=float(args.base_quantile),
        min_col_fg=int(args.base_min_col_fg),
        force_base_y_px=base_y_full if np.isfinite(base_y_full) else None,
    )
    if not np.isfinite(base_y_full):
        base_y_full = float(base_y_auto)

    # Base span from expected support width (1 mm) and base x-center estimate.
    expected_w_px = int(max(2, int(round(float(args.block_w) / float(m_per_px)))))
    xc = _estimate_base_xc_from_mask_band(
        mask,
        base_y_px=float(base_y_full),
        tol_px=float(args.base_tol_px),
        min_col_fg=int(min_col_fg_used),
        span_height_frac=float(args.base_span_height_frac),
    )
    x_left = int(round(float(xc) - 0.5 * float(expected_w_px)))
    x_right = int(x_left + expected_w_px)

    mask_s, base_y_i, x_left_i, x_right_i = _straighten_base_in_mask(
        mask,
        base_y_px=base_y_full,
        x_left_px=x_left,
        x_right_px=x_right,
        fill_columns=bool(args.base_fill_columns),
        clip_to_span=bool(args.straighten_clip_to_span),
    )
    cv2.imwrite(str(debug_dir / "chosen_mask_straight.png"), mask_s)

    # Extra cleanup (fill fissures, remove small speckle).
    mask_c = mask_s.copy()
    ksz = max(1, int(args.mask_clean_kernel))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    if int(args.mask_clean_close) > 0:
        mask_c = cv2.morphologyEx(mask_c, cv2.MORPH_CLOSE, ker, iterations=int(args.mask_clean_close))
    if int(args.mask_clean_open) > 0:
        mask_c = cv2.morphologyEx(mask_c, cv2.MORPH_OPEN, ker, iterations=int(args.mask_clean_open))
    cv2.imwrite(str(debug_dir / "chosen_mask_clean.png"), mask_c)

    mask_bool = mask_c.astype(np.uint8) > 0
    h, w = mask_bool.shape
    base_y_i = int(np.clip(int(base_y_i), 0, h - 1))

    # Row profile for robust top detection (ignore thin diagonal artifacts).
    row_fg = np.sum(mask_bool[: base_y_i + 1, :], axis=1).astype(float)
    max_fg = float(np.max(row_fg)) if row_fg.size else 0.0
    frac = float(np.clip(float(args.row_min_fg_frac), 0.0, 1.0))
    good = (row_fg >= (frac * max_fg)) if max_fg > 0 else np.zeros_like(row_fg, dtype=bool)
    if not bool(np.any(good)):
        raise RuntimeError("Could not determine top row for geometry extraction (mask too sparse).")
    y_top_i = int(np.min(np.nonzero(good)[0]))

    # Convert to mm in the ⊗-anchored coordinate system.
    # Base left corner (⊗) is x_left_i at y=base_y_i, in full-frame px = (x_left_i+x0, base_y_i+y0).
    dx_mm_per_px = float(m_per_px) * 1.0e3
    # Tiny correction to ensure the saved base is exactly 1.0 mm long (compensates integer rounding).
    base_len_mm_meas = float((x_right_i - x_left_i) * dx_mm_per_px)
    x_corr = (float(args.block_w) * 1.0e3) / max(1.0e-12, base_len_mm_meas)

    poly_method = str(args.polygon_method).strip().lower()
    Hb_mm: float
    if poly_method == "topcurve":
        # Fig. 5(b)-like polygon using the *top boundary* y_top(x) (cap + straight base).
        xs = np.arange(int(x_left_i), int(x_right_i) + 1, dtype=int)
        y_top_col = np.full(xs.size, np.nan, dtype=float)
        for k, xx in enumerate(xs):
            col = mask_bool[: base_y_i + 1, int(xx)]
            if not bool(np.any(col)):
                continue
            ys = np.nonzero(col)[0]
            qy = float(np.clip(float(args.top_y_quantile), 0.0, 0.49))
            y_top_col[k] = float(np.quantile(ys.astype(float), qy))

        y_top_global = float(y_top_i)
        y_top_col = np.where(np.isfinite(y_top_col), np.maximum(y_top_col, y_top_global), np.nan)

        height_px = float(base_y_i) - y_top_col
        Hb_px = float(np.nanmax(height_px)) if np.any(np.isfinite(height_px)) else 0.0
        min_h = float(max(2.0, float(args.top_min_height_frac) * Hb_px))
        bad = ~np.isfinite(height_px) | (height_px < min_h)
        if np.all(bad):
            raise RuntimeError("Top boundary extraction failed (all columns below min height). Try relaxing --top-min-height-frac.")

        idx = np.arange(xs.size, dtype=float)
        good = ~bad
        y_top_fill = y_top_col.copy()
        y_top_fill[bad] = np.interp(idx[bad], idx[good], y_top_col[good])
        first = int(np.nonzero(good)[0][0])
        last = int(np.nonzero(good)[0][-1])
        y_top_fill[:first] = y_top_fill[first]
        y_top_fill[last + 1 :] = y_top_fill[last]

        x_mm = (xs.astype(float) - float(x_left_i)) * dx_mm_per_px * float(x_corr)
        y_top_mm = (float(base_y_i) - y_top_fill.astype(float)) * dx_mm_per_px
        y_top_mm = np.clip(y_top_mm, 0.0, None)
        x_mm[0] = 0.0
        x_mm[-1] = float(args.block_w) * 1.0e3

        dx_mm = float(np.median(np.diff(x_mm))) if x_mm.size >= 2 else float(dx_mm_per_px)
        w_mm = float(max(0.0, float(args.top_smooth_window_mm)))
        win = int(max(1, int(round(w_mm / max(1.0e-12, dx_mm)))))
        y_top_s = _smooth_1d(y_top_mm, window=win, polyorder=int(args.top_smooth_polyorder))
        y_top_s = np.clip(y_top_s, 0.0, None)

        n_top = max(20, int(args.n_verts_top))
        stride = max(1, int(np.ceil(float(x_mm.size) / float(n_top))))
        sel = np.arange(0, x_mm.size, stride, dtype=int)
        if sel[-1] != x_mm.size - 1:
            sel = np.concatenate([sel, np.array([x_mm.size - 1], dtype=int)])

        x_t = x_mm[sel]
        y_t = y_top_s[sel]
        x_t[0] = 0.0
        x_t[-1] = float(args.block_w) * 1.0e3

        base_pts = np.array([[0.0, 0.0], [float(args.block_w) * 1.0e3, 0.0]], dtype=float)
        top_pts = np.column_stack([x_t, y_t])[::-1, :]
        poly = np.vstack([base_pts, top_pts, base_pts[:1, :]])
        Hb_mm = float(np.max(y_top_mm))
    elif poly_method == "composed":
        # Compose polygon from (i) straight base, (ii) right boundary xR(y), (iii) top curve yT(x), (iv) left boundary xL(y).
        # This retains the upper-right hump and removes fissure-scale noise.
        ys = np.arange(base_y_i, y_top_i - 1, -1, dtype=int)
        xL_px = np.full(ys.size, np.nan, dtype=float)
        xR_px = np.full(ys.size, np.nan, dtype=float)
        qL = float(np.clip(float(args.side_x_quantile_left), 0.0, 0.49))
        qR = float(np.clip(float(args.side_x_quantile_right), 0.51, 1.0))
        for k, yy in enumerate(ys):
            row = mask_bool[int(yy), :]
            xs_row = np.nonzero(row)[0]
            if xs_row.size < 2:
                continue
            xf = xs_row.astype(float)
            xL_px[k] = float(np.quantile(xf, qL))
            xR_px[k] = float(np.quantile(xf, qR))

        ok = np.isfinite(xL_px) & np.isfinite(xR_px)
        if int(np.sum(ok)) < 10:
            raise RuntimeError("Too few valid rows for side-boundary extraction. Try adjusting morphology/threshold.")
        idx = np.arange(ys.size, dtype=float)
        xL_px = np.interp(idx, idx[ok], xL_px[ok])
        xR_px = np.interp(idx, idx[ok], xR_px[ok])

        y_side_mm = (float(base_y_i) - ys.astype(float)) * dx_mm_per_px
        xL_mm = (xL_px - float(x_left_i)) * dx_mm_per_px * float(x_corr)
        xR_mm = (xR_px - float(x_left_i)) * dx_mm_per_px * float(x_corr)
        xL_mm = np.clip(xL_mm, 0.0, float(args.block_w) * 1.0e3)
        xR_mm = np.clip(xR_mm, 0.0, float(args.block_w) * 1.0e3)
        y_side_mm = np.clip(y_side_mm, 0.0, None)

        # Top curve yT(x): longest run per column to reject streaks.
        xs = np.arange(int(x_left_i), int(x_right_i) + 1, dtype=int)
        yT_px = np.full(xs.size, np.nan, dtype=float)
        qy = float(np.clip(float(args.top_y_quantile), 0.0, 0.49))
        for k, xx in enumerate(xs):
            col = mask_bool[: base_y_i + 1, int(xx)]
            if not bool(np.any(col)):
                continue
            ys_col = np.nonzero(col)[0]
            yT_px[k] = float(np.quantile(ys_col.astype(float), qy))

        y_top_global = float(y_top_i)
        yT_px = np.where(np.isfinite(yT_px), np.maximum(yT_px, y_top_global), np.nan)
        height_px = float(base_y_i) - yT_px
        Hb_px = float(np.nanmax(height_px)) if np.any(np.isfinite(height_px)) else 0.0
        min_h = float(max(2.0, float(args.top_min_height_frac) * Hb_px))
        bad = ~np.isfinite(height_px) | (height_px < min_h)
        if np.all(bad):
            raise RuntimeError("Top boundary extraction failed (all columns below min height). Try relaxing --top-min-height-frac.")
        idxx = np.arange(xs.size, dtype=float)
        good = ~bad
        yT_fill = yT_px.copy()
        yT_fill[bad] = np.interp(idxx[bad], idxx[good], yT_px[good])
        first = int(np.nonzero(good)[0][0])
        last = int(np.nonzero(good)[0][-1])
        yT_fill[:first] = yT_fill[first]
        yT_fill[last + 1 :] = yT_fill[last]

        x_top_mm = (xs.astype(float) - float(x_left_i)) * dx_mm_per_px * float(x_corr)
        y_top_mm = (float(base_y_i) - yT_fill.astype(float)) * dx_mm_per_px
        x_top_mm[0] = 0.0
        x_top_mm[-1] = float(args.block_w) * 1.0e3
        y_top_mm = np.clip(y_top_mm, 0.0, None)

        # Smooth side and top curves.
        dy_mm = float(np.median(np.diff(y_side_mm))) if y_side_mm.size >= 2 else float(dx_mm_per_px)
        w_side = float(max(0.0, float(args.side_smooth_window_mm)))
        win_side = int(max(1, int(round(w_side / max(1.0e-12, dy_mm)))))
        xL_s = _smooth_1d(xL_mm, window=win_side, polyorder=int(args.side_smooth_polyorder))
        xR_s = _smooth_1d(xR_mm, window=win_side, polyorder=int(args.side_smooth_polyorder))
        xL_s[0] = 0.0
        xR_s[0] = float(args.block_w) * 1.0e3

        dx_mm = float(np.median(np.diff(x_top_mm))) if x_top_mm.size >= 2 else float(dx_mm_per_px)
        w_top = float(max(0.0, float(args.top_smooth_window_mm)))
        win_top = int(max(1, int(round(w_top / max(1.0e-12, dx_mm)))))
        yT_s = _smooth_1d(y_top_mm, window=win_top, polyorder=int(args.top_smooth_polyorder))
        yT_s = np.clip(yT_s, 0.0, None)

        # Downsample to target vertex counts.
        n_side = max(30, int(args.n_verts_side))
        stride_side = max(1, int(np.ceil(float(y_side_mm.size) / float(n_side))))
        sel_side = np.arange(0, y_side_mm.size, stride_side, dtype=int)
        if sel_side[-1] != y_side_mm.size - 1:
            sel_side = np.concatenate([sel_side, np.array([y_side_mm.size - 1], dtype=int)])

        n_top = max(40, int(args.n_verts_top))
        stride_top = max(1, int(np.ceil(float(x_top_mm.size) / float(n_top))))
        sel_top = np.arange(0, x_top_mm.size, stride_top, dtype=int)
        if sel_top[-1] != x_top_mm.size - 1:
            sel_top = np.concatenate([sel_top, np.array([x_top_mm.size - 1], dtype=int)])

        left_pts = np.column_stack([xL_s[sel_side], y_side_mm[sel_side]])[::-1, :]   # top->base
        right_pts = np.column_stack([xR_s[sel_side], y_side_mm[sel_side]])            # base->top
        top_pts = np.column_stack([x_top_mm[sel_top], yT_s[sel_top]])[::-1, :]        # right->left

        base_pts = np.array([[0.0, 0.0], [float(args.block_w) * 1.0e3, 0.0]], dtype=float)
        # Force continuity at the top corners.
        if right_pts.shape[0] >= 2 and top_pts.shape[0] >= 2:
            top_pts[0, :] = right_pts[-1, :]
        if left_pts.shape[0] >= 2 and top_pts.shape[0] >= 2:
            top_pts[-1, :] = left_pts[0, :]
        poly = np.vstack([base_pts, right_pts[1:, :], top_pts[1:, :], left_pts[1:, :], base_pts[:1, :]])
        poly[:, 0] = np.clip(poly[:, 0], 0.0, float(args.block_w) * 1.0e3)
        poly[:, 1] = np.clip(poly[:, 1], 0.0, None)
        poly = _rotate_closed_poly_to_anchor(poly, (0.0, 0.0))
        poly[0, :] = np.array([0.0, 0.0])
        poly[-1, :] = poly[0, :]
        Hb_mm = float(np.max(y_top_mm))
    else:
        # Contour-based polygon: retain side curvature and the upper-right hump while enforcing:
        # - (0,0) at ⊗
        # - straight base from x=0..block_w as the closure segment
        #
        # Important: Video S1 contains a thin diagonal bright streak above the biofilm.
        # Using a *single* global y_top cutoff tends to flatten the true top curvature.
        # Instead, build a per-column "cap" from the top of the *longest* contiguous
        # foreground run (main body), and only clip contour points above that.

        xs_cap_px, y_cap_px = _column_top_from_longest_run(
            mask_bool,
            base_y_i=base_y_i,
            x_left_i=x_left_i,
            x_right_i=x_right_i,
            run_quantile=float(args.cap_run_quantile),
        )
        y_cap_px = _interp_nan_1d(y_cap_px)
        # Light smoothing in x to remove pixel jaggedness without flattening the hump.
        dx_mm = float(dx_mm_per_px) * float(x_corr)
        w_mm = float(max(0.0, float(args.smooth_s_window_mm)))
        win_cap = int(max(5, int(round(w_mm / max(1.0e-12, dx_mm)))))
        y_cap_px = _smooth_1d(y_cap_px, window=win_cap, polyorder=int(args.smooth_s_polyorder))
        y_cap_px = np.clip(y_cap_px, 0.0, float(base_y_i))

        def _y_cap_at_x(x_px: np.ndarray) -> np.ndarray:
            x_px = np.asarray(x_px, dtype=float)
            return np.interp(x_px, xs_cap_px, y_cap_px, left=float(y_cap_px[0]), right=float(y_cap_px[-1]))

        contours, _hier = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise RuntimeError("No contours found in cleaned mask.")
        cont = max(contours, key=cv2.contourArea)
        if bool(args.contour_use_convex_hull):
            cont = cv2.convexHull(cont)

        cont_xy = cont.reshape((-1, 2)).astype(float)
        if cont_xy.shape[0] < 10:
            raise RuntimeError("Contour too small to build a polygon.")

        # Find contour indices closest to the two base corners (in cropped pixel coords).
        corner_L = np.array([float(x_left_i), float(base_y_i)], dtype=float)
        corner_R = np.array([float(x_right_i), float(base_y_i)], dtype=float)
        dL = np.sum((cont_xy - corner_L[None, :]) ** 2, axis=1)
        dR = np.sum((cont_xy - corner_R[None, :]) ** 2, axis=1)
        iL = int(np.argmin(dL))
        iR = int(np.argmin(dR))

        def _path_forward(pts: np.ndarray, i0: int, i1: int) -> np.ndarray:
            if i0 <= i1:
                return pts[i0 : i1 + 1, :]
            return np.vstack([pts[i0:, :], pts[: i1 + 1, :]])

        cand1 = _path_forward(cont_xy, iL, iR)  # along contour order
        cand2 = _path_forward(cont_xy, iR, iL)[::-1, :]  # opposite direction, reversed to keep L->R

        def _score_path(p: np.ndarray) -> float:
            y_px = p[:, 1].astype(float)
            x_px = p[:, 0].astype(float)
            # Cap by the per-column longest-run top to reject thin diagonal streak artifacts.
            y_px = np.maximum(y_px, _y_cap_at_x(x_px))
            y_up = float(base_y_i) - y_px  # larger => higher
            frac_base = float(np.mean(y_up <= 2.0))
            return float(np.mean(y_up) - 80.0 * frac_base)

        upper_px = cand1 if _score_path(cand1) >= _score_path(cand2) else cand2

        # Enforce exact corners at the ends.
        upper_px = upper_px.copy()
        upper_px[0, :] = corner_L
        upper_px[-1, :] = corner_R

        # Simplify the upper boundary in pixel space (open curve).
        eps_mm = float(max(0.0, float(args.rdp_eps_mm)))
        eps_px = eps_mm / max(1.0e-12, float(dx_mm_per_px) * float(x_corr))
        if eps_px > 0.0 and upper_px.shape[0] >= 10:
            approx = cv2.approxPolyDP(upper_px.reshape((-1, 1, 2)).astype(np.float32), epsilon=float(eps_px), closed=False)
            upper_px = approx.reshape((-1, 2)).astype(float)
            upper_px[0, :] = corner_L
            upper_px[-1, :] = corner_R

        # Convert to mm in the ⊗-anchored coordinate system.
        x_px = upper_px[:, 0].astype(float)
        y_px = np.maximum(upper_px[:, 1].astype(float), _y_cap_at_x(x_px))
        x_mm = (upper_px[:, 0].astype(float) - float(x_left_i)) * dx_mm_per_px * float(x_corr)
        y_mm = (float(base_y_i) - y_px) * dx_mm_per_px
        x_right_mm = float(args.block_w) * 1.0e3
        x_mm = np.clip(x_mm, 0.0, None)
        y_mm = np.clip(y_mm, 0.0, None)

        base_trim_mm = float(max(0.0, float(args.contour_base_trim_mm)))
        if base_trim_mm > 0.0 and x_mm.size >= 6:
            keep = np.ones(x_mm.size, dtype=bool)
            interior = np.arange(1, x_mm.size - 1, dtype=int)
            keep[interior[y_mm[interior] < base_trim_mm]] = False
            if int(np.sum(keep)) >= 3:
                x_mm = x_mm[keep]
                y_mm = y_mm[keep]

        # Optional right-flank correction:
        # The raw contour tends to hug the extreme rightmost pixels, which can appear
        # as an artificial straight wall after base straightening. Use a robust row-wise
        # right profile from the mask interior to recover the expected curvature.
        if bool(args.contour_right_profile_correct):
            q_right = float(np.clip(float(args.contour_right_profile_quantile), 0.50, 0.999))
            x_start_mm = float(np.clip(float(args.contour_right_profile_xstart_mm), 0.0, x_right_mm))
            sel = (x_mm >= x_start_mm) & (y_mm > 0.0)
            if bool(np.any(sel)):
                idx_sel = np.nonzero(sel)[0]
                for ii in idx_sel.tolist():
                    yy_px = int(round(float(base_y_i) - float(y_mm[ii]) / max(1.0e-12, float(dx_mm_per_px))))
                    yy_px = int(np.clip(yy_px, 0, mask_bool.shape[0] - 1))
                    row = mask_bool[int(yy_px), int(x_left_i) : int(x_right_i) + 1]
                    xs_row = np.nonzero(row)[0]
                    if xs_row.size < 5:
                        continue
                    xq_px = float(x_left_i + np.quantile(xs_row.astype(float), q_right))
                    xq_mm = (xq_px - float(x_left_i)) * dx_mm_per_px * float(x_corr)
                    x_mm[ii] = min(float(x_mm[ii]), float(np.clip(xq_mm, 0.0, x_right_mm)))

        # Optional figure-guided prior on the right flank.
        # Uses traced Fig.5(b) coordinates to enforce non-vertical right curvature
        # without altering the anchored base or the left boundary extraction.
        if bool(args.fig5b_right_prior):
            fig5b_csv_prior = str(args.fig5b_traced_csv).strip()
            if fig5b_csv_prior:
                try:
                    tr = np.genfromtxt(fig5b_csv_prior, delimiter=",", skip_header=1, dtype=float)
                    if tr.ndim == 1:
                        tr = tr.reshape(1, -1)
                    if tr.shape[0] >= 5 and tr.shape[1] >= 2:
                        xt = np.asarray(tr[:, 0], dtype=float)
                        yt = np.asarray(tr[:, 1], dtype=float)
                        if np.any(np.isfinite(xt)) and np.any(np.isfinite(yt)):
                            x0t, x1t = float(np.nanmin(xt)), float(np.nanmax(xt))
                            y0t, y1t = float(np.nanmin(yt)), float(np.nanmax(yt))
                            if (x1t > x0t) and (y1t > y0t):
                                xt_mm = (xt - x0t) / (x1t - x0t) * float(x_right_mm)
                                yt_n = (yt - y0t) / (y1t - y0t)
                                yb, xr = _right_envelope_x_of_y(
                                    np.column_stack([xt_mm, yt_n]),
                                    y_min=0.0,
                                    y_max=1.0,
                                    n_bins=500,
                                )
                                xr = _interp_nan_1d(xr)
                                hb_loc = float(np.max(y_mm)) if np.any(np.isfinite(y_mm)) else 0.0
                                hb_loc = max(1.0e-12, hb_loc)
                                x_prior_start = float(
                                    np.clip(float(args.fig5b_right_prior_xstart_mm), 0.0, float(x_right_mm))
                                )
                                y0_frac = float(np.clip(float(args.fig5b_right_prior_ymin_frac), 0.0, 0.95))
                                sel = (x_mm >= x_prior_start) & (y_mm > 0.0)
                                if bool(np.any(sel)):
                                    yn = np.clip(y_mm[sel] / hb_loc, 0.0, 1.0)
                                    xr_i = np.interp(yn, yb, xr, left=float(xr[0]), right=float(xr[-1]))
                                    # Smoothly blend toward the Fig.5(b) right envelope only in the upper biofilm.
                                    w = np.clip((yn - y0_frac) / max(1.0e-12, (1.0 - y0_frac)), 0.0, 1.0)
                                    x_blend = (1.0 - w) * x_mm[sel] + w * np.clip(xr_i, 0.0, float(x_right_mm))
                                    x_mm[sel] = np.minimum(x_mm[sel], x_blend)
                except Exception:
                    pass

        upper_mm = np.column_stack([x_mm, y_mm])
        upper_mm[0, :] = np.array([0.0, 0.0])
        upper_mm[-1, :] = np.array([x_right_mm, 0.0])

        # Smooth along arclength (open curve) to remove jaggedness but keep the upper-right hump.
        ds_target = float(max(1.0e-6, float(args.smooth_s_ds_mm)))
        w_mm = float(max(0.0, float(args.smooth_s_window_mm)))
        win_pts = int(max(5, int(round(w_mm / ds_target))))
        upper_mm = _smooth_open_polyline_xy(upper_mm, ds_target=ds_target, window_pts=win_pts, polyorder=int(args.smooth_s_polyorder))
        upper_mm[:, 0] = np.clip(upper_mm[:, 0], 0.0, None)
        upper_mm[:, 1] = np.clip(upper_mm[:, 1], 0.0, None)
        upper_mm[0, :] = np.array([0.0, 0.0])
        upper_mm[-1, :] = np.array([x_right_mm, 0.0])

        if base_trim_mm > 0.0 and upper_mm.shape[0] >= 6:
            keep = (upper_mm[:, 1] >= base_trim_mm)
            keep[0] = True
            keep[-1] = True
            if int(np.sum(keep)) >= 3:
                upper_mm = upper_mm[keep, :]
                upper_mm[0, :] = np.array([0.0, 0.0])
                upper_mm[-1, :] = np.array([x_right_mm, 0.0])

        # Downsample to a publishable number of vertices (open curve).
        n_target = max(80, int(args.n_verts_contour))
        if upper_mm.shape[0] > n_target:
            stride = int(np.ceil(float(upper_mm.shape[0] - 1) / float(n_target - 1)))
            keep = np.arange(0, upper_mm.shape[0], stride, dtype=int)
            if keep[-1] != upper_mm.shape[0] - 1:
                keep = np.concatenate([keep, np.array([upper_mm.shape[0] - 1], dtype=int)])
            upper_mm = upper_mm[keep, :]
            upper_mm[0, :] = np.array([0.0, 0.0])
            upper_mm[-1, :] = np.array([x_right_mm, 0.0])

        poly = np.vstack([upper_mm, upper_mm[:1, :]])  # close (base is the closure segment)
        Hb_mm = float(np.max(poly[:, 1]))

    with out_csv.open("w", encoding="utf-8") as f:
        f.write("x_mm,y_mm\n")
        for x, y in poly:
            f.write(f"{float(x):.9f},{float(y):.9f}\n")

    # Optional comparison against Fig.5(b) traced coordinates (top envelope only).
    fig5b_csv = str(args.fig5b_traced_csv).strip()
    if fig5b_csv:
        try:
            import pandas as pd  # type: ignore

            tr = pd.read_csv(fig5b_csv)
            tr_xy = tr.iloc[:, :2].to_numpy(dtype=float)
            x0c, x1c = _parse_pair(str(args.fig5b_compare_xrange_mm))
            x0c = float(max(0.0, x0c))
            x1c = float(min(float(args.block_w) * 1.0e3, x1c))

            xg, yv = _top_envelope_xy(poly[:-1, :], x_min=0.0, x_max=float(args.block_w) * 1.0e3, n_bins=500)
            xt, yt = _top_envelope_xy(tr_xy, x_min=0.0, x_max=1.0, n_bins=500)
            yv = _interp_nan_1d(yv)
            yt = _interp_nan_1d(yt)

            # Map traced x in [0,1] -> mm using the same support width.
            xt_mm = xt * (float(args.block_w) * 1.0e3)
            ytr_on_xg = np.interp(xg, xt_mm, yt)
            sel = (xg >= x0c) & (xg <= x1c) & np.isfinite(yv) & np.isfinite(ytr_on_xg)
            if bool(np.any(sel)):
                diff = yv[sel] - ytr_on_xg[sel]
                rmse = float(np.sqrt(np.mean(diff**2)))
                max_abs = float(np.max(np.abs(diff)))
                print(f"[info] Fig5b(top-envelope) compare on x in [{x0c:.3f},{x1c:.3f}] mm: RMSE={rmse:.4f} mm, max|Δ|={max_abs:.4f} mm")
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt  # type: ignore

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(xg, yv, "-k", lw=2, label="video-extracted top")
                    ax.plot(xg, ytr_on_xg, "-r", lw=1.5, label="Fig5b traced top")
                    ax.axvspan(x0c, x1c, color="C0", alpha=0.08)
                    ax.set_xlim(0.0, float(args.block_w) * 1.0e3)
                    ax.set_ylim(0.0, 1.05 * float(max(np.nanmax(yv), np.nanmax(ytr_on_xg))))
                    ax.set_xlabel("x [mm]")
                    ax.set_ylabel("y_top [mm]")
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc="best")
                    fig.tight_layout()
                    fig.savefig(str(debug_dir / "compare_fig5b_top_envelope.png"), dpi=200)
                except Exception:
                    pass
        except Exception as e:
            print(f"[warn] Fig5b comparison failed: {e}")

    # Debug overlay (polygon back to pixels).
    overlay = img_full.copy()
    pts_px = []
    for x_mm_i, y_mm_i in poly:
        x_px = float(x_left_i + x0) + (float(x_mm_i) / (float(x_corr) * dx_mm_per_px))
        y_px = float(base_y_i + y0) - (float(y_mm_i) / dx_mm_per_px)
        pts_px.append([int(round(x_px)), int(round(y_px))])
    pts_px = np.asarray(pts_px, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [pts_px], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.circle(overlay, (int(round(x_left_i + x0)), int(round(base_y_i + y0))), 6, (0, 0, 255), -1)  # ⊗
    cv2.imwrite(str(debug_dir / "overlay_polygon_frame.png"), overlay)

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] debug outputs: {debug_dir}")
    print(f"[info] scale-bar: {float(args.scalebar_um):g} um / {int(scalebar_meta['run_px'])} px -> {float(m_per_px):.6e} m/px")
    print(f"[info] inferred support width: {float(args.block_w)/float(m_per_px):.1f} px")
    print(f"[info] base_y_px(cropped)={base_y_i}, base_x_left_px(cropped)={x_left_i}, base_x_right_px(cropped)={x_right_i}")
    print(f"[info] base_len_mm_measured={base_len_mm_meas:.6f}, x_corr={float(x_corr):.9f}")
    print(f"[info] polygon_method={poly_method}, Hb_mm={Hb_mm:.6f}, verts={int(poly.shape[0])}")


if __name__ == "__main__":
    main()
