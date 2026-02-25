#!/usr/bin/env python3
"""
Extract a deformation time series from the *experimental* OCT video (Video S1).

We track the right-edge x-position of the extracted biofilm contour at 3 fixed
heights (25%, 50%, 75% of the initial biofilm height) and report the horizontal
displacements dx(t) relative to t=0.

Important: The biofilm stands on a rigid support (1 mm diameter × 3 mm height).
For comparison with our simulation setup, we **scale pixels to meters** by
enforcing that the straightened base width equals `--block-w` (default 1 mm).

Outputs
-------
- `--out-csv`: deformation time series
- optional debug overlays in `--debug-dir`
- optional initial polygon in mm (base at y=0) via `--out-poly0-mm-csv`
"""

from __future__ import annotations

import argparse
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


def _rightmost_x_on_y(poly: np.ndarray, *, y_line: float) -> float:
    """
    Return the rightmost x intersection of a closed polygonal chain with y=y_line.
    """
    p = np.asarray(poly, dtype=float)
    if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] < 2:
        return float("nan")

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
    return float(np.max(xs)) if xs else float("nan")


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
    ap.add_argument("--simplify-eps", type=float, default=2.0, help="RDP epsilon in pixels (0 disables).")

    ap.add_argument("--block-w", type=float, default=1.0e-3, help="Support width used for pixel->meter scaling [m].")
    ap.add_argument(
        "--subtract-base-drift",
        action="store_true",
        help="Subtract the measured base (y=0) right-edge drift from all dx curves (removes global translation).",
    )
    ap.add_argument("--out-csv", type=str, default="out/_lie_exp_s1_dx/timeseries.csv")
    ap.add_argument("--debug-dir", type=str, default="out/_lie_exp_s1_dx")
    ap.add_argument("--debug-every", type=int, default=40, help="Write debug overlay every N processed frames (0 disables).")
    ap.add_argument(
        "--out-poly0-mm-csv",
        type=str,
        default="",
        help="If set, write the t=0 polygon in mm coords (base at y=0, x centered on base) to this CSV.",
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
        base_y_auto, x_left, x_right, _min_col_fg = _auto_base_from_mask(
            mask0,
            tol_px=float(args.base_tol_px),
            base_method=str(args.base_method),
            base_quantile=float(args.base_quantile),
            min_col_fg=int(args.base_min_col_fg),
            force_base_y_px=base_y_for_span,
        )
        if not np.isfinite(base_y):
            base_y = float(base_y_auto)
        base_x_left, base_x_right = int(x_left), int(x_right)
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
    m_per_px = float(args.block_w) / base_w_px
    base_xc_px = 0.5 * (float(base_x_left_full) + float(base_x_right_full))
    base_y_full_px = float(base_y) + float(y0)
    scale = _Scale(m_per_px=m_per_px, base_xc_px=base_xc_px, base_y_px=base_y_full_px)

    poly0_m = scale.contour_px_to_m(contour0_full_px)
    y_min = float(np.min(poly0_m[:, 1]))
    y_max = float(np.max(poly0_m[:, 1]))
    # Base should be at y=0 in this coordinate system. Small drift can occur due to rounding.
    poly0_m[:, 1] -= float(y_min)
    y_max = float(y_max - y_min)
    y_min = 0.0

    hb = max(1.0e-12, float(y_max - y_min))
    y_lines = np.array([0.25 * hb, 0.5 * hb, 0.75 * hb], dtype=float)
    x_ref = np.array([_rightmost_x_on_y(poly0_m, y_line=float(y)) for y in y_lines], dtype=float)
    x_base_ref = float(_rightmost_x_on_y(poly0_m, y_line=0.0))

    if str(args.out_poly0_mm_csv).strip():
        out_poly0 = Path(str(args.out_poly0_mm_csv))
        out_poly0.parent.mkdir(parents=True, exist_ok=True)
        poly0_mm = poly0_m * 1.0e3
        with out_poly0.open("w", encoding="utf-8") as f:
            f.write("x_mm,y_mm\n")
            for x, y in poly0_mm:
                f.write(f"{float(x):.9f},{float(y):.9f}\n")

    # Write header
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("t_s,dx_line1_m,dx_line2_m,dx_line3_m\n")

    debug_every = int(args.debug_every)
    for k, fr in enumerate(frames):
        img = img0 if fr == 0 else _read_frame(cap, int(fr))
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
                        base_y_auto, x_left_auto, x_right_auto, _min_col_fg = _auto_base_from_mask(
                            mask,
                            tol_px=float(args.base_tol_px),
                            base_method=str(args.base_method),
                            base_quantile=float(args.base_quantile),
                            min_col_fg=int(args.base_min_col_fg),
                            force_base_y_px=None,
                        )
                        base_y_fr = float(base_y_auto)
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

            # Convert to meters relative to the (possibly frame-specific) base.
            if bool(args.straighten_base) and bool(args.base_per_frame):
                x0_fr = int(meta["x0"])
                y0_fr = int(meta["y0"])
                base_xc_fr = 0.5 * (float(x_left_fr + x0_fr) + float(x_right_fr + x0_fr))
                base_y_fr_full = float(base_y_i) + float(y0_fr)
                scale_fr = _Scale(m_per_px=m_per_px, base_xc_px=base_xc_fr, base_y_px=base_y_fr_full)
                poly_m = scale_fr.contour_px_to_m(contour_full_px)
            else:
                poly_m = scale.contour_px_to_m(contour_full_px)

            # Shift so the base is at y=0.
            poly_m[:, 1] -= float(np.min(poly_m[:, 1]))
            xs_now = np.array([_rightmost_x_on_y(poly_m, y_line=float(y)) for y in y_lines], dtype=float)
            dx = xs_now - x_ref
            if bool(args.subtract_base_drift):
                x_base_now = float(_rightmost_x_on_y(poly_m, y_line=0.0))
                dx_base = x_base_now - x_base_ref
                dx = dx - float(dx_base)
        except Exception:
            dx = np.array([float("nan"), float("nan"), float("nan")], dtype=float)
            poly_m = None

        t_s = float(fr) / float(fps)
        with out_csv.open("a", encoding="utf-8") as f:
            f.write(f"{t_s:.12e},{dx[0]:.12e},{dx[1]:.12e},{dx[2]:.12e}\n")

        if debug_every > 0 and (k == 0 or (k % debug_every) == 0):
            overlay = img.copy()
            if poly_m is not None:
                # Draw contour in pixel coords for readability.
                pts = contour_full_px.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                # Draw y-lines and right-edge intersection points.
                if bool(args.straighten_base) and bool(args.base_per_frame):
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
    print(f"[info] base span px (frame0, full): x_left={base_x_left_full}, x_right={base_x_right_full} -> base_w_px={base_w_px:.1f}")
    print(f"[info] scale: {m_per_px:.6e} m/px (block_w={float(args.block_w):.3e} m)")
    print(f"[info] y_lines_m={y_lines.tolist()}")


if __name__ == "__main__":
    main()
