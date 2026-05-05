#!/usr/bin/env python3
"""
Extract a deformation time series from the *experimental* OCT video (Video S1).

We track an extreme x-intersection of the extracted biofilm contour at 3 fixed
heights (25%, 50%, 75% of the initial biofilm height) and report the horizontal
displacements dx(t) relative to t=0:

- `--x-intersection leftmost` (default): upstream edge (matches the reference
  point convention used in Li et al., Fig. 7)
- `--x-intersection rightmost`: downstream edge

Important: The biofilm stands on a rigid support (1 mm diameter × 3 mm height).
For publishable comparison with Li et al. (2020), we **scale pixels to meters**
using the 100 µm scale bar embedded in the video (`--scale-mode scalebar`),
with an optional fallback that uses the detected support width (`--scale-mode support`).

Outputs
-------
- `--out-csv`: deformation time series
- optional debug overlays in `--debug-dir`
- optional initial polygon in mm (base at y=0) via `--out-poly0-mm-csv`
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _read_frame(cap: cv2.VideoCapture, frame: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    if not ok or img is None:
        raise RuntimeError(f"Could not read frame={frame}")
    return img


def _preprocess_gray(img_bgr: np.ndarray, *, clahe: bool, blur_ksize: int) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if bool(clahe):
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = c.apply(gray)
    if int(blur_ksize) and int(blur_ksize) > 1:
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray


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
    """
    Parse ROI fractions: "x0,y0,x1,y1" in [0,1].
    """
    parts = [p.strip() for p in str(roi).strip().split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("--scalebar-roi-frac must be 'x0,y0,x1,y1' (fractions in [0,1])")
    x0, y0, x1, y1 = (float(p) for p in parts)
    x0, y0 = float(np.clip(x0, 0.0, 1.0)), float(np.clip(y0, 0.0, 1.0))
    x1, y1 = float(np.clip(x1, 0.0, 1.0)), float(np.clip(y1, 0.0, 1.0))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("--scalebar-roi-frac requires x1>x0 and y1>y0")
    return float(x0), float(y0), float(x1), float(y1)


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
    """
    Detect the 100 µm scale bar (L-shape) and compute meters-per-pixel.

    Returns:
      (m_per_px, meta)
    """
    if img_bgr is None or img_bgr.ndim != 3:
        raise ValueError("img_bgr must be a BGR image.")
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

    def _max_run_1d(arr_1d: np.ndarray) -> tuple[int, int, int]:
        # returns (max_run, start, end_exclusive)
        max_run = 0
        best = (0, 0)
        i = 0
        n = int(arr_1d.size)
        while i < n:
            if int(arr_1d[i]) == 0:
                i += 1
                continue
            i0 = i
            while i < n and int(arr_1d[i]) != 0:
                i += 1
            run = int(i - i0)
            if run > max_run:
                max_run = run
                best = (i0, i)
        return int(max_run), int(best[0]), int(best[1])

    max_h = 0
    max_h_info = (0, 0, 0)  # y, x0, x1
    for yy in range(on.shape[0]):
        run, x0r, x1r = _max_run_1d(on[yy, :])
        if run > max_h:
            max_h = int(run)
            max_h_info = (int(yy), int(x0r), int(x1r))

    max_v = 0
    max_v_info = (0, 0, 0)  # x, y0, y1
    for xx in range(on.shape[1]):
        run, y0r, y1r = _max_run_1d(on[:, xx])
        if run > max_v:
            max_v = int(run)
            max_v_info = (int(xx), int(y0r), int(y1r))

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
        "max_h_info": tuple(int(v) for v in max_h_info),
        "max_v_run_px": int(max_v),
        "max_v_info": tuple(int(v) for v in max_v_info),
        "run_px": int(run_px),
        "m_per_px": float(m_per_px),
    }
    return float(m_per_px), meta


def _largest_contour(mask_u8: np.ndarray) -> np.ndarray:
    contours, _hier = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No contours found.")
    c = max(contours, key=cv2.contourArea)
    c = np.asarray(c, dtype=np.int32)
    if c.ndim != 3 or c.shape[1] != 1 or c.shape[2] != 2:
        raise RuntimeError(f"Unexpected contour array shape: {c.shape}")
    return c[:, 0, :]  # (N,2) as (x,y)


def _auto_base_from_mask(
    chosen_mask_u8: np.ndarray,
    *,
    tol_px: float,
    base_method: str,
    base_quantile: float,
    min_col_fg: int,
    force_base_y_px: float | None = None,
) -> tuple[float, int, int, int]:
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
        if method not in {"mask_mode", "mask_quantile"}:
            raise ValueError(f"Unsupported base_method: {base_method}")

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

    # X-span from intersections with a narrow band around base_y.
    y_lo = int(np.clip(int(np.floor(base_y - float(tol_px))), 0, h - 1))
    y_hi = int(np.clip(int(np.ceil(base_y + float(tol_px))), 0, h - 1))
    band_rows = mask[y_lo : y_hi + 1, :]
    col_has = np.any(band_rows, axis=0)
    xs = np.nonzero(col_has & valid_cols)[0]
    if xs.size < 2:
        xs = np.nonzero(col_has)[0]
    if xs.size < 2:
        raise RuntimeError("Could not determine x-span of the base.")
    return float(base_y), int(np.min(xs)), int(np.max(xs)), int(min_col_fg)


def _estimate_base_xc_from_mask_band(
    chosen_mask_u8: np.ndarray,
    *,
    base_y_px: float,
    tol_px: float,
    min_col_fg: int,
    span_height_frac: float,
) -> float:
    """
    Estimate base x-center from a band around base_y.

    Works in *cropped* coordinates.
    """
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
) -> tuple[np.ndarray, int]:
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
    for x in range(x_left, x_right + 1):
        col = mask_clip[:, x]
        if not bool(np.any(col)):
            continue
        y_top = int(np.argmax(col))
        mask_out[y_top : base_y_i + 1, x] = True

    mask_out[base_y_i, x_left : x_right + 1] = True
    return (mask_out.astype(np.uint8) * 255), int(base_y_i)


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
        gray_blur = cv2.GaussianBlur(gray, (k, k), 0)
    else:
        gray_blur = gray

    if str(thr_mode) == "otsu":
        thr_val, mask = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thr_val = float(thr)
        mask = (gray_blur > float(thr_val)).astype(np.uint8) * 255

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
        "gray_blur": gray_blur,
        "mask": mask,
    }
    return chosen_mask, meta


def _simplify_contour(contour_px: np.ndarray, *, eps: float) -> np.ndarray:
    c = contour_px.astype(np.float32)
    if float(eps) <= 0.0 or c.shape[0] < 5:
        return c
    c_in = c.reshape((-1, 1, 2))
    c_out = cv2.approxPolyDP(c_in, epsilon=float(eps), closed=True)
    return np.asarray(c_out, dtype=np.float32).reshape((-1, 2))


def _x_intersection_from_mask(
    mask_u8: np.ndarray,
    *,
    y_px: float,
    mode: str,
    band_px: int,
    q_left: float,
    q_right: float,
) -> float:
    """
    Return an extreme x position on the filled mask at y=y_px (in pixel coords).

    This is more robust than polygon/contour intersection for noisy OCT frames,
    especially near the base tracking line.
    """
    m = (mask_u8.astype(np.uint8) > 0)
    h, w = m.shape
    yy = int(np.clip(int(round(float(y_px))), 0, h - 1))
    band = int(max(0, int(band_px)))
    y0 = max(0, yy - band)
    y1 = min(h - 1, yy + band)
    rows = m[y0 : y1 + 1, :]
    if not bool(np.any(rows)):
        return float("nan")
    ys, xs = np.nonzero(rows)
    if xs.size < 2:
        return float("nan")

    mode = str(mode).strip().lower()
    if mode == "leftmost":
        q = float(np.clip(float(q_left), 0.0, 0.49))
    elif mode == "rightmost":
        q = float(np.clip(float(q_right), 0.51, 1.0))
    else:
        raise ValueError(f"Unknown x-intersection mode: {mode!r}. Use 'rightmost' or 'leftmost'.")
    return float(np.quantile(xs.astype(float), q))


def _x_intersection_on_y(poly: np.ndarray, *, y_line: float, mode: str) -> float:
    """
    Return an extreme x intersection of a closed polygonal chain with y=y_line.

    Parameters
    ----------
    mode
        "rightmost" (max x) or "leftmost" (min x).
    """
    p = np.asarray(poly, dtype=float)
    if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] < 2:
        return float("nan")

    mode = str(mode).strip().lower()
    if mode not in {"rightmost", "leftmost"}:
        raise ValueError(f"Unknown x-intersection mode: {mode!r}. Use 'rightmost' or 'leftmost'.")

    y0 = float(y_line)
    xs: list[float] = []
    for a, b in zip(p, np.vstack([p[1:], p[:1]])):
        x1, y1 = float(a[0]), float(a[1])
        x2, y2 = float(b[0]), float(b[1])
        if (y0 < min(y1, y2)) or (y0 > max(y1, y2)):
            continue
        dy = y2 - y1
        if abs(dy) <= 1.0e-12:
            # horizontal segment on the query line
            if abs(y0 - y1) <= 1.0e-9:
                xs.extend([x1, x2])
            continue
        t = (y0 - y1) / dy
        if 0.0 <= t <= 1.0:
            xs.append(x1 + t * (x2 - x1))
    if not xs:
        return float("nan")
    return float(np.max(xs)) if mode == "rightmost" else float(np.min(xs))


def _read_anchor_px_from_json(path: str | Path) -> tuple[int, int]:
    meta = json.loads(Path(str(path)).read_text(encoding="utf-8"))
    ax = int(meta["anchor_px"]["x"])
    ay = int(meta["anchor_px"]["y"])
    return int(ax), int(ay)


@dataclass(frozen=True)
class _Scale:
    m_per_px: float
    base_xc_px: float
    base_y_px: float

    def contour_px_to_m(self, contour_full_px: np.ndarray) -> np.ndarray:
        c = np.asarray(contour_full_px, dtype=float)
        x = (c[:, 0] - float(self.base_xc_px)) * float(self.m_per_px)
        y = (float(self.base_y_px) - c[:, 1]) * float(self.m_per_px)
        return np.column_stack([x, y])


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract deformation dx(t) from experimental Video S1 (Lie benchmark).")
    ap.add_argument(
        "--video",
        type=str,
        default="examples/biofilms/benchmarks/lie/additional_data/bit27491-sup-0001-si_v1.avi",
        help="Path to Video S1 AVI file.",
    )
    ap.add_argument("--crop", type=str, default="", help="Optional crop: 'x0,y0,w,h' in pixels.")
    ap.add_argument("--frame-step", type=int, default=1, help="Process every N-th frame.")
    ap.add_argument("--max-frames", type=int, default=0, help="If >0, only process first N frames.")

    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--blur-ksize", type=int, default=5)
    ap.add_argument("--thr-mode", type=str, default="otsu", choices=("otsu", "fixed"))
    ap.add_argument("--thr", type=float, default=80.0)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--close-iters", type=int, default=2)
    ap.add_argument("--open-iters", type=int, default=0)
    ap.add_argument("--min-area", type=int, default=5000)
    ap.add_argument("--component-index", type=int, default=0)

    ap.add_argument("--straighten-base", action="store_true", help="Straighten base and clip below it.")
    ap.add_argument(
        "--base-per-frame",
        action="store_true",
        help="Re-detect the base (y and x-span) for each frame to reduce scan drift.",
    )
    ap.add_argument("--base-y-px", type=float, default=float("nan"), help="Manual base y in pixels (full-frame coords).")
    ap.add_argument("--base-method", type=str, default="mask_mode", choices=("mask_mode", "mask_quantile"))
    ap.add_argument("--base-quantile", type=float, default=0.95)
    ap.add_argument("--base-tol-px", type=float, default=6.0)
    ap.add_argument("--base-min-col-fg", type=int, default=0)
    ap.add_argument(
        "--base-span-height-frac",
        type=float,
        default=0.2,
        help=(
            "Filter for base x-center estimation: keep only base-touching columns whose vertical extent is "
            ">= frac*max_height. Helps remove speckle noise touching the base band."
        ),
    )
    ap.add_argument("--simplify-eps", type=float, default=2.0, help="RDP epsilon in pixels (0 disables).")

    ap.add_argument("--block-w", type=float, default=1.0e-3, help="Support width used for pixel->meter scaling [m].")
    ap.add_argument(
        "--scale-mode",
        type=str,
        default="scalebar",
        choices=("scalebar", "support"),
        help="Pixel->meter scaling method. 'scalebar' uses the 100 µm bar in the video; 'support' uses detected base width=block-w.",
    )
    ap.add_argument("--scalebar-um", type=float, default=100.0, help="Scale-bar length [µm] used for scalebar scaling.")
    ap.add_argument(
        "--scalebar-roi-frac",
        type=str,
        default="0.75,0.75,1.0,1.0",
        help="ROI (fractions) for detecting the scale bar: 'x0,y0,x1,y1' in [0,1].",
    )
    ap.add_argument("--scalebar-thr", type=int, default=200, help="Intensity threshold (0..255) for scale bar detection.")
    ap.add_argument("--scalebar-kernel", type=int, default=3, help="Morphology kernel size for scale bar detection.")
    ap.add_argument("--scalebar-open-iters", type=int, default=1, help="Morphology opening iterations for scale bar detection.")
    ap.add_argument("--scalebar-close-iters", type=int, default=1, help="Morphology closing iterations for scale bar detection.")
    ap.add_argument(
        "--hb-quantile",
        type=float,
        default=0.995,
        help="Quantile used to define initial biofilm height H_b from the t=0 contour (helps ignore thin top artifacts).",
    )
    ap.add_argument(
        "--hb-method",
        type=str,
        default="mask",
        choices=("mask", "contour_quantile"),
        help="How to define initial biofilm height H_b from frame 0.",
    )
    ap.add_argument(
        "--hb-row-frac",
        type=float,
        default=0.2,
        help="For --hb-method mask: use the top-most row whose foreground count is >= frac*max_row_count.",
    )
    ap.add_argument(
        "--hb-override-mm",
        type=float,
        default=float("nan"),
        help="If set, override the extracted H_b [mm] used for defining tracking lines (0.75/0.50/0.25*H_b).",
    )
    ap.add_argument(
        "--hb-from-polygon-mm-csv",
        type=str,
        default="",
        help="If set, read H_b from a smoothed t=0 polygon CSV in mm (max y_mm), overriding --hb-method.",
    )
    ap.add_argument(
        "--x-intersection",
        type=str,
        default="leftmost",
        choices=("rightmost", "leftmost"),
        help="Which contour intersection to track for dx(t) along each y-line.",
    )
    ap.add_argument(
        "--x-method",
        type=str,
        default="mask",
        choices=("mask", "contour"),
        help="How to compute x(y) intersections: from filled mask pixels (robust) or from contour intersections.",
    )
    ap.add_argument("--mask-y-band-px", type=int, default=1, help="Half-band (±px) for mask-based x(y) intersections.")
    ap.add_argument("--mask-x-quantile-left", type=float, default=0.02, help="Quantile for leftmost mask intersection.")
    ap.add_argument("--mask-x-quantile-right", type=float, default=0.98, help="Quantile for rightmost mask intersection.")
    ap.add_argument(
        "--subtract-base-drift",
        action="store_true",
        help="Subtract the measured base (y=0) right-edge drift from all dx curves (removes global translation).",
    )
    ap.add_argument(
        "--dx-method",
        type=str,
        default="intersection",
        choices=("intersection", "opticalflow"),
        help="How to extract dx(t): contour/mask intersections (Eulerian) or LK optical flow tracking (DIC-like).",
    )
    ap.add_argument("--flow-offset-in-px", type=int, default=3, help="Offset points into the biofilm interior for optical flow [px].")
    ap.add_argument("--lk-win", type=int, default=31, help="LK optical flow window size (odd).")
    ap.add_argument("--lk-max-level", type=int, default=3, help="LK optical flow pyramid levels.")
    ap.add_argument("--lk-iters", type=int, default=30, help="LK optical flow max iterations.")
    ap.add_argument("--lk-eps", type=float, default=0.01, help="LK optical flow termination epsilon.")
    ap.add_argument("--out-csv", type=str, default="out/_lie_exp_s1_dx_leftmost/timeseries.csv")
    ap.add_argument("--debug-dir", type=str, default="out/_lie_exp_s1_dx_leftmost")
    ap.add_argument("--debug-every", type=int, default=40, help="Write debug overlay every N processed frames (0 disables).")
    ap.add_argument(
        "--out-poly0-mm-csv",
        type=str,
        default="",
        help="If set, write the t=0 polygon in mm coords (base at y=0, x centered on base) to this CSV.",
    )
    ap.add_argument(
        "--anchor-json",
        type=str,
        default="",
        help="Optional anchor JSON from extract_biofilm_similarity_stripes.py (used for polygon-based optical-flow init).",
    )
    ap.add_argument(
        "--init-poly0-mm-csv",
        type=str,
        default="",
        help="If set and --dx-method opticalflow: initialize tracking points from this ⊗-anchored polygon CSV (mm). Requires --anchor-json.",
    )
    args = ap.parse_args()

    crop = _parse_crop(str(args.crop))
    frame_step = max(1, int(args.frame_step))

    out_csv = Path(str(args.out_csv))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(str(args.debug_dir))
    debug_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 1.0

    frames = list(range(0, n_frames, frame_step))
    if int(args.max_frames) > 0:
        frames = frames[: int(args.max_frames)]
    if not frames or frames[0] != 0:
        frames = [0] + frames

    # Frame 0: base detection, scaling and y-lines
    img0 = _read_frame(cap, 0)
    scale_mode = str(args.scale_mode).strip().lower()
    m_per_px_scalebar: float | None = None
    scalebar_meta: dict | None = None
    if scale_mode == "scalebar":
        try:
            roi_frac = _parse_roi_frac(str(args.scalebar_roi_frac))
            m_per_px_scalebar, scalebar_meta = _detect_scalebar_m_per_px(
                img0,
                scalebar_um=float(args.scalebar_um),
                roi_frac=roi_frac,
                thr=int(args.scalebar_thr),
                kernel=int(args.scalebar_kernel),
                open_iters=int(args.scalebar_open_iters),
                close_iters=int(args.scalebar_close_iters),
            )
        except Exception as e:
            print(f"[warn] scale-bar detection failed: {e}. Falling back to support-width scaling.", flush=True)
            scale_mode = "support"
    mask0, meta0 = _extract_component_mask(
        img0,
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

    base_y = float(args.base_y_px)
    if np.isfinite(base_y) and crop is not None:
        base_y = float(base_y) - float(crop[1])

    base_x_left = base_x_right = None
    if bool(args.straighten_base):
        base_y_for_span = float(base_y) if np.isfinite(base_y) else None
        base_y_auto, x_left_mask, x_right_mask, min_col_fg_used = _auto_base_from_mask(
            mask0,
            tol_px=float(args.base_tol_px),
            base_method=str(args.base_method),
            base_quantile=float(args.base_quantile),
            min_col_fg=int(args.base_min_col_fg),
            force_base_y_px=base_y_for_span,
        )
        if not np.isfinite(base_y):
            base_y = float(base_y_auto)
        if scale_mode == "scalebar" and m_per_px_scalebar is not None and np.isfinite(float(m_per_px_scalebar)):
            expected_w_px = int(max(2, int(round(float(args.block_w) / float(m_per_px_scalebar)))))
            xc0 = _estimate_base_xc_from_mask_band(
                mask0,
                base_y_px=float(base_y),
                tol_px=float(args.base_tol_px),
                min_col_fg=int(min_col_fg_used),
                span_height_frac=float(args.base_span_height_frac),
            )
            x_left = int(round(float(xc0) - 0.5 * float(expected_w_px)))
            x_right = int(x_left + expected_w_px)
            base_x_left, base_x_right = int(x_left), int(x_right)
        else:
            base_x_left, base_x_right = int(x_left_mask), int(x_right_mask)
        mask0, base_y_i = _straighten_base_in_mask(mask0, base_y_px=base_y, x_left_px=base_x_left, x_right_px=base_x_right)
        base_y = float(base_y_i)

    contour0 = _simplify_contour(_largest_contour(mask0), eps=float(args.simplify_eps))
    # Convert from cropped to full-frame px
    x0 = int(meta0["x0"])
    y0 = int(meta0["y0"])
    contour0_full_px = contour0 + np.array([float(x0), float(y0)], dtype=float)

    if base_x_left is None or base_x_right is None:
        # Fallback scaling: use bbox width if base detection is disabled.
        xs0 = contour0_full_px[:, 0]
        base_x_left = int(np.min(xs0)) - int(x0)
        base_x_right = int(np.max(xs0)) - int(x0)

    base_x_left_full = int(base_x_left) + int(x0)
    base_x_right_full = int(base_x_right) + int(x0)
    base_w_px = max(1.0, float(base_x_right_full - base_x_left_full))
    if scale_mode == "scalebar" and m_per_px_scalebar is not None and np.isfinite(float(m_per_px_scalebar)):
        m_per_px = float(m_per_px_scalebar)
    else:
        m_per_px = float(args.block_w) / base_w_px
    base_xc_px = 0.5 * (float(base_x_left_full) + float(base_x_right_full))
    base_y_full_px = float(base_y) + float(y0)
    scale = _Scale(m_per_px=m_per_px, base_xc_px=base_xc_px, base_y_px=base_y_full_px)

    poly0_m = scale.contour_px_to_m(contour0_full_px)
    y_min = float(np.min(poly0_m[:, 1]))
    # Base should be at y=0 in this coordinate system. Small drift can occur due to rounding.
    poly0_m[:, 1] -= float(y_min)
    y_min = 0.0

    # Follow the paper's convention (Fig. 7a): line 1 is the topmost line, line 3 is the lowest.
    hb_method = str(args.hb_method).strip().lower()
    if hb_method == "mask":
        # Estimate H_b from the segmented mask itself. This is robust to thin top artifacts
        # (e.g. a bright diagonal streak weakly connected to the biofilm body).
        base_y_i = int(np.clip(int(round(float(base_y))), 0, mask0.shape[0] - 1))
        row_fg = np.sum(mask0[: base_y_i + 1, :].astype(np.uint8) > 0, axis=1).astype(float)
        max_fg = float(np.max(row_fg)) if row_fg.size else 0.0
        frac = float(np.clip(float(args.hb_row_frac), 0.0, 1.0))
        if max_fg > 0.0:
            good = row_fg >= (frac * max_fg)
            if bool(np.any(good)):
                y_top_i = int(np.min(np.nonzero(good)[0]))
                hb_px = float(base_y_i - y_top_i)
                hb = float(hb_px) * float(m_per_px)
            else:
                hb = float(np.max(poly0_m[:, 1]))
        else:
            hb = float(np.max(poly0_m[:, 1]))
    else:
        q_hb = float(np.clip(float(args.hb_quantile), 0.0, 1.0))
        hb = float(np.quantile(poly0_m[:, 1], q_hb))
    hb = max(1.0e-12, float(hb))
    hb_override_mm = float(args.hb_override_mm)
    hb_poly_csv = str(args.hb_from_polygon_mm_csv).strip()
    if hb_poly_csv:
        try:
            poly_mm = np.genfromtxt(hb_poly_csv, delimiter=",", skip_header=1, dtype=float)
            if poly_mm.ndim == 1:
                poly_mm = poly_mm.reshape(1, -1)
            if poly_mm.shape[1] >= 2 and poly_mm.shape[0] >= 3:
                hb = float(np.nanmax(poly_mm[:, 1])) * 1.0e-3
        except Exception:
            pass
    elif np.isfinite(hb_override_mm):
        hb = float(hb_override_mm) * 1.0e-3
    hb = max(1.0e-12, float(hb))
    y_lines = np.array([0.75 * hb, 0.5 * hb, 0.25 * hb], dtype=float)
    x_mode = str(args.x_intersection).strip().lower()
    x_method = str(args.x_method).strip().lower()
    if x_method == "mask":
        # Compute intersections directly from the segmented mask (frame 0).
        base_y0 = float(base_y)  # cropped coords
        y_lines_px = [float(base_y0) - float(y) / float(m_per_px) for y in y_lines.tolist()]
        base_xc0 = float(base_xc_px) - float(x0)  # cropped coords
        xs_px = np.array(
            [
                _x_intersection_from_mask(
                    mask0,
                    y_px=float(ypx),
                    mode=x_mode,
                    band_px=int(args.mask_y_band_px),
                    q_left=float(args.mask_x_quantile_left),
                    q_right=float(args.mask_x_quantile_right),
                )
                for ypx in y_lines_px
            ],
            dtype=float,
        )
        x_ref = (xs_px + float(x0) - float(base_xc_px)) * float(m_per_px)
        x_base_px = _x_intersection_from_mask(
            mask0,
            y_px=float(base_y0),
            mode=x_mode,
            band_px=int(args.mask_y_band_px),
            q_left=float(args.mask_x_quantile_left),
            q_right=float(args.mask_x_quantile_right),
        )
        x_base_ref = (float(x_base_px) + float(x0) - float(base_xc_px)) * float(m_per_px)
    else:
        x_ref = np.array([_x_intersection_on_y(poly0_m, y_line=float(y), mode=x_mode) for y in y_lines], dtype=float)
        x_base_ref = float(_x_intersection_on_y(poly0_m, y_line=0.0, mode=x_mode))

    if str(args.out_poly0_mm_csv).strip():
        out_poly0 = Path(str(args.out_poly0_mm_csv))
        out_poly0.parent.mkdir(parents=True, exist_ok=True)
        poly0_mm = poly0_m * 1.0e3
        with out_poly0.open("w", encoding="utf-8") as f:
            f.write("x_mm,y_mm\n")
            for x, y in poly0_mm:
                f.write(f"{float(x):.9f},{float(y):.9f}\n")

    # Optional DIC-like tracking via optical flow: initialize tracking points at t=0.
    dx_method = str(args.dx_method).strip().lower()
    pts0_full: np.ndarray | None = None
    pts0_init: np.ndarray | None = None
    pts0_prev: np.ndarray | None = None
    prev_gray: np.ndarray | None = None
    if dx_method == "opticalflow":
        off_in = int(max(0, int(args.flow_offset_in_px)))
        off_sgn = +1 if x_mode == "leftmost" else -1
        init_poly0 = str(args.init_poly0_mm_csv).strip()
        if init_poly0:
            if not str(args.anchor_json).strip():
                raise ValueError("--init-poly0-mm-csv requires --anchor-json.")
            ax_px, ay_px = _read_anchor_px_from_json(str(args.anchor_json))

            poly0_mm = np.genfromtxt(init_poly0, delimiter=",", skip_header=1, dtype=float)
            if poly0_mm.ndim == 1:
                poly0_mm = poly0_mm.reshape(1, -1)
            if poly0_mm.shape[1] < 2 or poly0_mm.shape[0] < 3:
                raise ValueError(f"--init-poly0-mm-csv has insufficient points/cols: {init_poly0}")
            poly0_anchor_m = np.asarray(poly0_mm[:, :2], dtype=float) * 1.0e-3
            hb_m_flow = float(np.nanmax(poly0_anchor_m[:, 1]))
            hb_m_flow = max(1.0e-12, hb_m_flow)
            y_lines_flow = np.array([0.75 * hb_m_flow, 0.5 * hb_m_flow, 0.25 * hb_m_flow], dtype=float)

            xs0_m = np.array([_x_intersection_on_y(poly0_anchor_m, y_line=float(y), mode=x_mode) for y in y_lines_flow.tolist()], dtype=float)
            xs0_px_full = [float(ax_px) + float(xm) / float(m_per_px) for xm in xs0_m.tolist()]
            ys0_px_full = [float(ay_px) - float(ym) / float(m_per_px) for ym in y_lines_flow.tolist()]

            pts = []
            for xpf, ypf in zip(xs0_px_full, ys0_px_full):
                pts.append([float(xpf + off_sgn * off_in), float(ypf)])

            # Reference point on the rigid support near ⊗ (used when --subtract-base-drift).
            base_w_px = float(args.block_w) / max(1.0e-30, float(m_per_px))
            if x_mode == "leftmost":
                x_ref_px_full = float(ax_px + off_in)
            else:
                x_ref_px_full = float(ax_px + base_w_px - off_in)
            y_ref_px_full = float(ay_px)
            pts.append([x_ref_px_full, y_ref_px_full])
            pts0_full = np.asarray(pts, dtype=np.float32).reshape((-1, 1, 2))
        else:
            # Use mask intersections at frame 0 to locate the tracking points (green dots in Fig. 7a),
            # then track those material points using LK optical flow.
            base_y0_c = float(base_y)  # cropped coords
            y_lines_px_c = [float(base_y0_c) - float(y) / float(m_per_px) for y in y_lines.tolist()]
            xs0_px_c = [
                _x_intersection_from_mask(
                    mask0,
                    y_px=float(ypx),
                    mode=x_mode,
                    band_px=int(args.mask_y_band_px),
                    q_left=float(args.mask_x_quantile_left),
                    q_right=float(args.mask_x_quantile_right),
                )
                for ypx in y_lines_px_c
            ]
            ys0_px_full = [(float(base_y) + float(y0)) - float(y) / float(m_per_px) for y in y_lines.tolist()]
            xs0_px_full = [(float(xc) + float(x0)) for xc in xs0_px_c]
            pts = []
            for xpf, ypf in zip(xs0_px_full, ys0_px_full):
                pts.append([float(xpf + off_sgn * off_in), float(ypf)])
            if x_mode == "leftmost":
                x_ref_px_full = float(base_x_left_full + off_in)
            else:
                x_ref_px_full = float(base_x_right_full - off_in)
            y_ref_px_full = float(base_y) + float(y0)
            pts.append([x_ref_px_full, y_ref_px_full])
            pts0_full = np.asarray(pts, dtype=np.float32).reshape((-1, 1, 2))

        pts0_init = pts0_full.copy()
        pts0_prev = pts0_full.copy()
        prev_gray = _preprocess_gray(img0, clahe=bool(args.clahe), blur_ksize=int(args.blur_ksize))

    # Write header
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("t_s,dx_line1_m,dx_line2_m,dx_line3_m\n")

    debug_every = int(args.debug_every)
    for k, fr in enumerate(frames):
        img = img0 if fr == 0 else _read_frame(cap, int(fr))
        contour_full_px = None
        xs_now = np.array([float("nan"), float("nan"), float("nan")], dtype=float)

        if dx_method == "opticalflow":
            if pts0_init is None or pts0_prev is None or prev_gray is None:
                raise RuntimeError("Internal error: optical-flow tracking not initialized.")
            gray = _preprocess_gray(img, clahe=bool(args.clahe), blur_ksize=int(args.blur_ksize))
            if int(fr) == 0:
                pts_now = pts0_prev.copy()
            else:
                win = int(max(5, int(args.lk_win)))
                if win % 2 == 0:
                    win += 1
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(args.lk_iters), float(args.lk_eps))
                lk_params = dict(winSize=(win, win), maxLevel=int(args.lk_max_level), criteria=criteria)
                pts_next, st, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts0_prev, None, **lk_params)
                if pts_next is None or st is None:
                    pts_now = np.full_like(pts0_prev, np.nan, dtype=np.float32)
                else:
                    st = st.astype(np.uint8).ravel()
                    pts_now = pts_next.astype(np.float32)
                    # Any failure -> mark as NaN to make it obvious.
                    for ii, ok in enumerate(st.tolist()):
                        if int(ok) != 1:
                            pts_now[ii, 0, :] = np.nan
                pts0_prev = pts_now.copy()
                prev_gray = gray

            # dx relative to t=0.
            dxs = (pts_now[:3, 0, 0] - pts0_init[:3, 0, 0]).astype(np.float64) * float(m_per_px)
            if bool(args.subtract_base_drift):
                dx_ref = float(pts_now[3, 0, 0] - pts0_init[3, 0, 0]) * float(m_per_px)
                dxs = dxs - dx_ref
            dx = np.asarray(dxs, dtype=float)
            xs_now = (pts_now[:3, 0, 0].astype(np.float64) - float(scale.base_xc_px)) * float(m_per_px)

            t_s = float(fr) / float(fps)
            with out_csv.open("a", encoding="utf-8") as f:
                f.write(f"{t_s:.12e},{dx[0]:.12e},{dx[1]:.12e},{dx[2]:.12e}\n")

            if debug_every > 0 and (k == 0 or (k % debug_every) == 0):
                overlay = img.copy()
                for ii in range(pts_now.shape[0]):
                    xpf, ypf = float(pts_now[ii, 0, 0]), float(pts_now[ii, 0, 1])
                    if not (np.isfinite(xpf) and np.isfinite(ypf)):
                        continue
                    col = (0, 0, 255) if ii < 3 else (255, 0, 255)
                    cv2.circle(overlay, (int(round(xpf)), int(round(ypf))), 5, col, -1)
                cv2.imwrite(str(debug_dir / f"frame_{int(fr):04d}_overlay.png"), overlay)
            continue

        try:
            mask, meta = _extract_component_mask(
                img,
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
            if bool(args.straighten_base):
                if base_x_left is None or base_x_right is None:
                    raise RuntimeError("Base span unknown; run with --straighten-base.")

                # Optionally re-detect base per frame to reduce scan drift.
                base_y_fr = float(base_y)
                x_left_fr = int(base_x_left)
                x_right_fr = int(base_x_right)
                if bool(args.base_per_frame):
                    try:
                        base_y_auto, x_left_auto, x_right_auto, min_col_fg_used = _auto_base_from_mask(
                            mask,
                            tol_px=float(args.base_tol_px),
                            base_method=str(args.base_method),
                            base_quantile=float(args.base_quantile),
                            min_col_fg=int(args.base_min_col_fg),
                            force_base_y_px=None,
                        )
                        base_y_fr = float(base_y_auto)
                        if scale_mode == "scalebar" and np.isfinite(float(m_per_px)):
                            expected_w_px = int(max(2, int(round(float(args.block_w) / float(m_per_px)))))
                            xc_fr = _estimate_base_xc_from_mask_band(
                                mask,
                                base_y_px=float(base_y_fr),
                                tol_px=float(args.base_tol_px),
                                min_col_fg=int(min_col_fg_used),
                                span_height_frac=float(args.base_span_height_frac),
                            )
                            x_left_fr = int(round(float(xc_fr) - 0.5 * float(expected_w_px)))
                            x_right_fr = int(x_left_fr + expected_w_px)
                        else:
                            x_left_fr = int(x_left_auto)
                            x_right_fr = int(x_right_auto)
                    except Exception:
                        # Fall back to the frame-0 base if per-frame detection fails.
                        base_y_fr = float(base_y)
                        x_left_fr = int(base_x_left)
                        x_right_fr = int(base_x_right)

                mask, base_y_i = _straighten_base_in_mask(mask, base_y_px=base_y_fr, x_left_px=x_left_fr, x_right_px=x_right_fr)

            contour = _simplify_contour(_largest_contour(mask), eps=float(args.simplify_eps))
            contour_full_px = contour + np.array([float(meta["x0"]), float(meta["y0"])], dtype=float)

            # Compute x(y) in meters for this frame.
            if bool(args.straighten_base) and bool(args.base_per_frame):
                x0_fr = int(meta["x0"])
                y0_fr = int(meta["y0"])
                base_xc_fr_full = 0.5 * (float(x_left_fr + x0_fr) + float(x_right_fr + x0_fr))
                base_y_fr_full = float(base_y_i) + float(y0_fr)
            else:
                x0_fr = int(meta["x0"])
                y0_fr = int(meta["y0"])
                base_xc_fr_full = float(scale.base_xc_px)
                base_y_fr_full = float(scale.base_y_px)

            if x_method == "mask":
                base_y_fr_c = float(base_y_fr_full) - float(y0_fr)
                y_lines_px = [float(base_y_fr_c) - float(y) / float(m_per_px) for y in y_lines.tolist()]
                xs_px = np.array(
                    [
                        _x_intersection_from_mask(
                            mask,
                            y_px=float(ypx),
                            mode=x_mode,
                            band_px=int(args.mask_y_band_px),
                            q_left=float(args.mask_x_quantile_left),
                            q_right=float(args.mask_x_quantile_right),
                        )
                        for ypx in y_lines_px
                    ],
                    dtype=float,
                )
                xs_now = (xs_px + float(x0_fr) - float(base_xc_fr_full)) * float(m_per_px)
                dx = xs_now - x_ref
                if bool(args.subtract_base_drift):
                    x_base_px = _x_intersection_from_mask(
                        mask,
                        y_px=float(base_y_fr_c),
                        mode=x_mode,
                        band_px=int(args.mask_y_band_px),
                        q_left=float(args.mask_x_quantile_left),
                        q_right=float(args.mask_x_quantile_right),
                    )
                    x_base_now = (float(x_base_px) + float(x0_fr) - float(base_xc_fr_full)) * float(m_per_px)
                    dx = dx - float(x_base_now - x_base_ref)
                poly_m = None
            else:
                # Convert to meters relative to the (possibly frame-specific) base (for contour intersections).
                if bool(args.straighten_base) and bool(args.base_per_frame):
                    scale_fr = _Scale(m_per_px=m_per_px, base_xc_px=base_xc_fr_full, base_y_px=base_y_fr_full)
                    poly_m = scale_fr.contour_px_to_m(contour_full_px)
                else:
                    poly_m = scale.contour_px_to_m(contour_full_px)

                poly_m[:, 1] -= float(np.min(poly_m[:, 1]))
                xs_now = np.array([_x_intersection_on_y(poly_m, y_line=float(y), mode=x_mode) for y in y_lines], dtype=float)
                dx = xs_now - x_ref
                if bool(args.subtract_base_drift):
                    x_base_now = float(_x_intersection_on_y(poly_m, y_line=0.0, mode=x_mode))
                    dx = dx - float(x_base_now - x_base_ref)
        except Exception:
            dx = np.array([float("nan"), float("nan"), float("nan")], dtype=float)
            poly_m = None
            contour_full_px = None
            xs_now = np.array([float("nan"), float("nan"), float("nan")], dtype=float)

        t_s = float(fr) / float(fps)
        with out_csv.open("a", encoding="utf-8") as f:
            f.write(f"{t_s:.12e},{dx[0]:.12e},{dx[1]:.12e},{dx[2]:.12e}\n")

        if debug_every > 0 and (k == 0 or (k % debug_every) == 0):
            overlay = img.copy()
            if contour_full_px is not None:
                # Draw contour in pixel coords for readability.
                pts = contour_full_px.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                # Draw y-lines and right-edge intersection points.
                if bool(args.straighten_base) and bool(args.base_per_frame) and x_method != "mask":
                    base_xc_use = float(scale_fr.base_xc_px)
                    base_y_use = float(scale_fr.base_y_px)
                else:
                    base_xc_use = float(scale.base_xc_px)
                    base_y_use = float(scale.base_y_px)
                for i, y_m in enumerate(y_lines.tolist()):
                    y_px = int(round(base_y_use - float(y_m) / float(m_per_px)))
                    cv2.line(overlay, (0, y_px), (overlay.shape[1] - 1, y_px), (255, 0, 0), 1)
                    x_m = float(xs_now[i]) if i < xs_now.size else float("nan")
                    if np.isfinite(x_m) and np.isfinite(y_m):
                        x_px = int(round(base_xc_use + x_m / float(m_per_px)))
                        cv2.circle(overlay, (x_px, y_px), 4, (0, 0, 255), -1)
            cv2.imwrite(str(debug_dir / f"frame_{int(fr):04d}_overlay.png"), overlay)

    cap.release()

    print(f"[ok] wrote {out_csv}")
    print(f"[info] fps={fps}, frames={n_frames}, frame_step={frame_step}, processed={len(frames)}")
    print(f"[info] scale_mode={scale_mode}")
    if scale_mode == "scalebar" and scalebar_meta is not None:
        run_px = int(scalebar_meta.get("run_px", -1))
        print(f"[info] scalebar: {float(args.scalebar_um):g} um / {run_px} px -> {m_per_px:.6e} m/px")
        if np.isfinite(float(m_per_px)) and float(m_per_px) > 0:
            inferred_support_px = float(args.block_w) / float(m_per_px)
            print(f"[info] inferred support width: {inferred_support_px:.1f} px (block_w={float(args.block_w):.3e} m)")
    print(f"[info] base span px (frame0, full): x_left={base_x_left_full}, x_right={base_x_right_full} -> base_w_px={base_w_px:.1f}")
    print(f"[info] scale: {m_per_px:.6e} m/px")
    print(
        f"[info] hb_method={str(args.hb_method)}, hb_row_frac={float(args.hb_row_frac):g}, "
        f"hb_quantile={float(args.hb_quantile):g}, hb_m={hb:.6e}"
    )
    print(f"[info] y_lines_m={y_lines.tolist()}")


if __name__ == "__main__":
    main()
