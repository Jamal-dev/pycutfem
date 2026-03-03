#!/usr/bin/env python3
"""
Attempt to extract a biofilm contour from the *experimental* OCT video (Video S1).

Experimental OCT frames are noisy; this script is designed to be debug-first:
it writes intermediate images (mask, component boxes, selected contour overlay)
so you can tune crop/threshold/morphology if needed.

Outputs
-------
- Debug PNGs in `--debug-dir`
- Polygon CSV (pixel coordinates) in `--out-csv`

Example
-------
python examples/biofilms/benchmarks/lie/extract_polygon_from_experimental_video_s1.py \\
  --frame 0 --clahe --thr-mode fixed --thr 80 \\
  --close-iters 2 --kernel 5 \\
  --debug-dir out/_lie_exp_s1_f0_thr80 \\
  --out-csv out/_lie_exp_s1_f0_thr80/biofilm_contour_px.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _read_frame(video_path: str, frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    cap.release()
    if not ok or img is None:
        raise RuntimeError(f"Could not read frame={frame} from video: {video_path}")
    return img


def _parse_crop(crop: str) -> tuple[int, int, int, int] | None:
    """
    Parse crop string: "x0,y0,w,h" (pixels). Empty string -> None.
    """
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
        raise RuntimeError("No contours found. Try changing threshold/crop/morphology.")
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
    span_height_frac: float = 0.0,
    force_base_y_px: float | None = None,
) -> tuple[float, int, int, int]:
    """
    Detect a horizontal "base" for the biofilm from a binary component mask.

    Returns:
      (base_y_px, x_left_px, x_right_px, min_col_fg_used) in *cropped* coordinates.
    """
    if chosen_mask_u8.ndim != 2:
        raise ValueError("chosen_mask_u8 must be 2D (grayscale) mask.")

    mask = chosen_mask_u8.astype(np.uint8) > 0
    h, w = mask.shape
    tol_px = float(max(1.0, float(tol_px)))

    col_fg = np.sum(mask, axis=0).astype(int)
    if int(min_col_fg) <= 0:
        min_col_fg = max(3, int(round(0.01 * float(h))))
    valid_cols = col_fg >= int(min_col_fg)
    # If the threshold is too strict, fall back to "any foreground".
    if not bool(np.any(valid_cols)):
        valid_cols = np.any(mask, axis=0)

    has_fg = np.any(mask, axis=0)
    # Bottom profile: for each x, the largest y with mask=1.
    rev_idx = np.argmax(mask[::-1, :], axis=0)  # first True from bottom
    y_bottom = (h - 1) - rev_idx
    y_bottom = y_bottom.astype(float)
    y_bottom[~has_fg] = np.nan
    y_bottom[~valid_cols] = np.nan

    yv = y_bottom[np.isfinite(y_bottom)].astype(int)
    if yv.size < 10:
        raise RuntimeError("Not enough columns to auto-detect a base. Try changing crop/threshold/min-area.")

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

    # Determine the *horizontal span* at the base by looking at where the mask
    # intersects a narrow band around base_y (rather than using bottommost pixels,
    # which can be dominated by speckle noise below the true base).
    y_lo = int(np.clip(int(np.floor(base_y - float(tol_px))), 0, h - 1))
    y_hi = int(np.clip(int(np.ceil(base_y + float(tol_px))), 0, h - 1))
    band_rows = mask[y_lo : y_hi + 1, :]
    col_has = np.any(band_rows, axis=0)

    # Optional: exclude spurious base-touching columns that have too little
    # vertical extent (typically speckle noise touching the base band).
    #
    # We keep only columns whose mask extends above the base by at least a
    # fraction of the overall height of the component.
    base_y_i = int(np.clip(int(round(float(base_y))), 0, h - 1))
    span_height_frac = float(max(0.0, float(span_height_frac)))
    if span_height_frac > 0.0:
        has_fg_col = np.any(mask, axis=0)
        y_top = np.full(w, np.nan, dtype=float)
        # argmax gives the first True along axis=0 when mask is boolean.
        y_top[has_fg_col] = np.argmax(mask[:, has_fg_col], axis=0).astype(float)
        heights = float(base_y_i) - y_top
        heights[~np.isfinite(heights)] = 0.0
        h_max = float(np.max(heights)) if np.any(np.isfinite(heights)) else 0.0
        h_thr = max(10.0, span_height_frac * h_max)
        col_has = col_has & (heights >= h_thr)

    xs = np.nonzero(col_has & valid_cols)[0]
    if xs.size < 2:
        xs = np.nonzero(col_has)[0]
    if xs.size < 2:
        raise RuntimeError("Could not determine x-span of the base (too few columns).")

    # Choose the longest contiguous run (robust to isolated noisy columns).
    xs = np.sort(xs.astype(int))
    runs: list[tuple[int, int]] = []
    start = int(xs[0])
    prev = int(xs[0])
    for x in xs[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
            continue
        runs.append((start, prev))
        start = prev = x
    runs.append((start, prev))
    x_left, x_right = max(runs, key=lambda r: int(r[1] - r[0]))
    return float(base_y), x_left, x_right, int(min_col_fg)


def _straighten_base_in_mask(
    chosen_mask_u8: np.ndarray,
    *,
    base_y_px: float,
    x_left_px: int,
    x_right_px: int,
) -> tuple[np.ndarray, int]:
    """
    Enforce a perfectly horizontal base at y=base_y for x in [x_left, x_right].

    Returns:
      (mask_u8_straight, base_y_int)
    """
    mask = (chosen_mask_u8.astype(np.uint8) > 0)
    h, w = mask.shape

    base_y_i = int(np.clip(int(round(float(base_y_px))), 0, h - 1))
    x_left = int(np.clip(int(x_left_px), 0, w - 1))
    x_right = int(np.clip(int(x_right_px), 0, w - 1))
    if x_left > x_right:
        x_left, x_right = x_right, x_left

    # Clip everything below the base.
    mask_clip = mask.copy()
    if base_y_i + 1 < h:
        mask_clip[base_y_i + 1 :, :] = False

    mask_out = mask_clip.copy()

    # Fill down to the base in the base x-range to remove noisy protrusions and ensure attachment is straight.
    for x in range(x_left, x_right + 1):
        col = mask_clip[:, x]
        if not bool(np.any(col)):
            continue
        y_top = int(np.argmax(col))  # first True (top-most) because y increases downward
        mask_out[y_top : base_y_i + 1, x] = True

    # Ensure base line is present and *only* on the support span.
    mask_out[base_y_i, :] = False
    mask_out[base_y_i, x_left : x_right + 1] = True

    return (mask_out.astype(np.uint8) * 255), int(base_y_i)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract a contour from experimental Video S1 (OCT).")
    ap.add_argument(
        "--video",
        type=str,
        default="examples/biofilms/benchmarks/lie/additional_data/bit27491-sup-0001-si_v1.avi",
        help="Path to Video S1 AVI file.",
    )
    ap.add_argument("--frame", type=int, default=0, help="Frame index to process.")
    ap.add_argument("--crop", type=str, default="", help="Optional crop: 'x0,y0,w,h' in pixels.")

    ap.add_argument("--clahe", action="store_true", help="Apply CLAHE contrast enhancement.")
    ap.add_argument("--blur-ksize", type=int, default=5, help="Gaussian blur ksize (odd). 0 disables.")
    ap.add_argument("--thr-mode", type=str, default="otsu", choices=("otsu", "fixed"))
    ap.add_argument("--thr", type=float, default=80.0, help="Threshold for --thr-mode fixed (0..255).")
    ap.add_argument("--invert", action="store_true", help="Invert the thresholded mask.")

    ap.add_argument("--kernel", type=int, default=5, help="Morphology kernel size.")
    ap.add_argument("--close-iters", type=int, default=2, help="Morphological closing iterations.")
    ap.add_argument("--open-iters", type=int, default=0, help="Morphological opening iterations.")

    ap.add_argument("--min-area", type=int, default=5000, help="Filter components smaller than this.")
    ap.add_argument(
        "--component-index",
        type=int,
        default=0,
        help="Component index after sorting by area (0=largest, after filtering).",
    )
    ap.add_argument(
        "--straighten-base",
        action="store_true",
        help="Replace the biofilm base with a straight segment (biofilm stands on top of the support).",
    )
    ap.add_argument(
        "--base-y-px",
        type=float,
        default=float("nan"),
        help="Manual base y-coordinate in pixels (image coords, y down). If unset, auto-detect from the component mask.",
    )
    ap.add_argument(
        "--base-method",
        type=str,
        default="mask_mode",
        choices=("mask_mode", "mask_quantile"),
        help="Auto base detector (used when --base-y-px is unset). Uses the bottom profile of the selected component mask.",
    )
    ap.add_argument("--base-quantile", type=float, default=0.95, help="Quantile used when --base-method mask_quantile (0-1).")
    ap.add_argument("--base-tol-px", type=float, default=3.0, help="Tolerance band around base y for selecting base columns [px].")
    ap.add_argument(
        "--base-min-col-fg",
        type=int,
        default=0,
        help="Min number of foreground pixels in a column to consider it for base detection. 0 uses 1%% of image height.",
    )
    ap.add_argument(
        "--base-span-height-frac",
        type=float,
        default=0.0,
        help=(
            "Optional filter for base x-span detection: keep only base-touching columns "
            "whose vertical extent is >= frac*max_height. Helps remove speckle noise touching the base."
        ),
    )
    ap.add_argument("--simplify-eps", type=float, default=2.0, help="RDP simplification epsilon [px]. 0 disables.")

    ap.add_argument("--debug-dir", type=str, default="out/_lie_exp_s1_extract")
    ap.add_argument("--out-csv", type=str, default="out/_lie_exp_s1_extract/biofilm_contour_px.csv")
    args = ap.parse_args()

    debug_dir = Path(str(args.debug_dir))
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(str(args.out_csv))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    img_full = _read_frame(str(args.video), int(args.frame))
    crop = _parse_crop(str(args.crop))
    x0 = y0 = 0
    img = img_full
    if crop is not None:
        x0, y0, w, h = crop
        img = img_full[y0 : y0 + h, x0 : x0 + w].copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if bool(args.clahe):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    if int(args.blur_ksize) and int(args.blur_ksize) > 1:
        k = int(args.blur_ksize)
        if k % 2 == 0:
            k += 1
        gray_blur = cv2.GaussianBlur(gray, (k, k), 0)
    else:
        gray_blur = gray

    thr_val = None
    if str(args.thr_mode) == "otsu":
        thr_val, mask = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thr_val = float(args.thr)
        mask = (gray_blur > thr_val).astype(np.uint8) * 255

    if bool(args.invert):
        mask = 255 - mask

    ksz = max(1, int(args.kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    if int(args.close_iters) > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(args.close_iters))
    if int(args.open_iters) > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=int(args.open_iters))

    # Connected components for selection.
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for lab in range(1, int(n)):
        x, y, w, h, area = stats[lab].tolist()
        if int(area) < int(args.min_area):
            continue
        comps.append((int(area), int(lab), (int(x), int(y), int(w), int(h))))
    comps.sort(reverse=True, key=lambda t: t[0])
    if not comps:
        raise RuntimeError("No components after filtering. Lower --min-area or adjust threshold/crop.")

    # Debug outputs (cropped).
    cv2.imwrite(str(debug_dir / "frame.png"), img)
    cv2.imwrite(str(debug_dir / "gray.png"), gray)
    cv2.imwrite(str(debug_dir / "gray_blur.png"), gray_blur)
    cv2.imwrite(str(debug_dir / "mask.png"), mask)

    overlay = cv2.cvtColor(gray_blur, cv2.COLOR_GRAY2BGR)
    for i, (_area, _lab, (bx, by, bw, bh)) in enumerate(comps[:10]):
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
        cv2.putText(overlay, str(i), (bx, max(0, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(debug_dir / "components_overlay.png"), overlay)

    idx = int(args.component_index)
    if not (0 <= idx < len(comps)):
        raise ValueError(f"--component-index {idx} out of range; have {len(comps)} components after filtering.")
    area, chosen_lab, (bx, by, bw, bh) = comps[idx]

    chosen_mask = (labels == int(chosen_lab)).astype(np.uint8) * 255
    base_y = float(args.base_y_px)
    cv2.imwrite(str(debug_dir / "chosen_mask.png"), chosen_mask)

    base_x_left = base_x_right = None
    min_col_fg_used = None
    if bool(args.straighten_base):
        # Work in cropped-frame coordinates; convert a manual base from full-frame if needed.
        if np.isfinite(base_y) and crop is not None:
            base_y = float(base_y) - float(y0)

        base_y_for_span = float(base_y) if np.isfinite(base_y) else None
        base_y_auto, x_left, x_right, min_col_fg_used = _auto_base_from_mask(
            chosen_mask,
            tol_px=float(args.base_tol_px),
            base_method=str(args.base_method),
            base_quantile=float(args.base_quantile),
            min_col_fg=int(args.base_min_col_fg),
            span_height_frac=float(args.base_span_height_frac),
            force_base_y_px=base_y_for_span,
        )
        if not np.isfinite(base_y):
            base_y = float(base_y_auto)

        chosen_mask, base_y_i = _straighten_base_in_mask(
            chosen_mask,
            base_y_px=float(base_y),
            x_left_px=int(x_left),
            x_right_px=int(x_right),
        )
        base_y = float(base_y_i)
        base_x_left = int(x_left)
        base_x_right = int(x_right)
        cv2.imwrite(str(debug_dir / "chosen_mask_straight.png"), chosen_mask)

    contour = _largest_contour(chosen_mask)
    contour_f = contour.astype(np.float32)

    if float(args.simplify_eps) > 0.0 and contour_f.shape[0] >= 5:
        c_in = contour_f.reshape((-1, 1, 2))
        c_out = cv2.approxPolyDP(c_in, epsilon=float(args.simplify_eps), closed=True)
        contour_f = np.asarray(c_out, dtype=np.float32).reshape((-1, 2))

    contour_full = contour_f + np.array([float(x0), float(y0)], dtype=np.float32)

    with out_csv.open("w", encoding="utf-8") as f:
        f.write("x_px,y_px\n")
        for x, y in contour_full:
            f.write(f"{float(x):.6f},{float(y):.6f}\n")

    # Overlay selected contour on full frame (not cropped).
    overlay2 = img_full.copy()
    pts = contour_full.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay2, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imwrite(str(debug_dir / "selected_contour_overlay.png"), overlay2)

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] debug outputs: {debug_dir}")
    print(f"[info] thr_mode={args.thr_mode}, thr={thr_val}, invert={bool(args.invert)}, crop={crop}")
    print(f"[info] components={len(comps)} (min_area={args.min_area}); chosen index={idx}, area={area}, bbox={(bx, by, bw, bh)}")
    if bool(args.straighten_base):
        print(f"[info] straightened base: base_y_px={base_y}")
        if base_x_left is not None and base_x_right is not None:
            print(f"[info] base span (cropped px): x_left={base_x_left}, x_right={base_x_right}")
        if min_col_fg_used is not None:
            print(f"[info] base detector: method={args.base_method}, min_col_fg={min_col_fg_used}, tol_px={float(args.base_tol_px):g}")


if __name__ == "__main__":
    main()
