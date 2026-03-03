#!/usr/bin/env python3
"""
Biofilm geometry extraction from Li et al. (2020) OCT Video S1
-------------------------------------------------------------

What this script does (frame 0 workflow):
1) Finds the *paper's* anchor ⊗ on Fig. 5a by detecting the red cross.
2) Uses an image-to-image similarity alignment (ECC affine) between Fig. 5a and Video frame 0
   (in the downscaled space) to map that anchor into the full-resolution video frame.
3) Detects the 100 µm scale bar in the video frame to obtain mm/px scaling.
4) Enhances faint lower regions using *horizontal rectangular stripes* (row bands) with
   depth compensation (gain per stripe) + CLAHE.
5) Extracts the outer biofilm contour inside a *prior search region* (dilated polygon):
   - For frame 0, you can use the traced Fig. 5b CSV as the prior.
   - For later frames, use the previous frame's extracted CSV as the prior.
6) Enforces a straight base line between x=0..block_w_mm at y=0 and outputs a valid polygon.

Outputs
-------
- anchor JSON (pixel coordinates, mm/px, etc.)
- extracted polygon CSV in mm (x_mm,y_mm), CCW, closed
- debug overlays

Dependencies: numpy, opencv-python
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np


def read_video_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def detect_red_cross_centroid(bgr: np.ndarray) -> Tuple[float, float]:
    """
    Detect the red cross (⊗) centroid in Fig. 5a/5b.

    Robustness notes:
    - JPEG compression can introduce isolated reddish pixels far away from the cross.
      We therefore threshold for "strong red", then select the *largest connected component*.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # "Strong red" threshold (stricter than a generic red threshold).
    mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 100, 100), (179, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)

    # Fill small gaps so the cross is one component.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        raise RuntimeError("Could not detect the anchor cross (no red component found).")
    # Choose the largest component (ignore background label 0).
    areas = stats[1:, cv2.CC_STAT_AREA]
    lab = int(np.argmax(areas) + 1)
    cx, cy = centroids[lab]
    if not np.isfinite(cx) or not np.isfinite(cy):
        raise RuntimeError("Anchor cross centroid is not finite.")
    return float(cx), float(cy)


def ecc_affine_align(template_gray: np.ndarray, input_gray: np.ndarray, input_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    ECC finds a warp W such that input warped aligns to template.
    Returns 2x3 affine warp.
    """
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2000, 1e-6)
    # ECC is sensitive; a small gaussian helps.
    _cc, warp = cv2.findTransformECC(template_gray, input_gray, warp, cv2.MOTION_AFFINE, criteria, inputMask=input_mask, gaussFiltSize=5)
    return warp


def invert_affine(warp_2x3: np.ndarray) -> np.ndarray:
    A = warp_2x3[:, :2].astype(np.float64)
    t = warp_2x3[:, 2].astype(np.float64)
    A_inv = np.linalg.inv(A)
    t_inv = -A_inv @ t
    out = np.zeros((2, 3), dtype=np.float64)
    out[:, :2] = A_inv
    out[:, 2] = t_inv
    return out.astype(np.float32)


def apply_affine_to_point(warp_2x3: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    p = np.array([x, y, 1.0], dtype=np.float32)
    q = warp_2x3 @ p
    return float(q[0]), float(q[1])


def detect_scalebar_run_px(
    img_bgr: np.ndarray,
    roi_frac: Tuple[float, float, float, float] = (0.75, 0.75, 1.0, 1.0),
    thr: int = 200,
    kernel: int = 3,
    open_iters: int = 1,
    close_iters: int = 1,
) -> int:
    h, w = img_bgr.shape[:2]
    x0f, y0f, x1f, y1f = roi_frac
    x0 = int(round(x0f * w))
    y0 = int(round(y0f * h))
    x1 = int(round(x1f * w))
    y1 = int(round(y1f * h))
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

    def max_run_1d(arr: np.ndarray) -> int:
        max_run = 0
        i = 0
        n = int(arr.size)
        while i < n:
            if int(arr[i]) == 0:
                i += 1
                continue
            j = i
            while j < n and int(arr[j]) != 0:
                j += 1
            max_run = max(max_run, j - i)
            i = j
        return int(max_run)

    max_h = max(max_run_1d(on[yy, :]) for yy in range(on.shape[0]))
    max_v = max(max_run_1d(on[:, xx]) for xx in range(on.shape[1]))
    return int(max(max_h, max_v))


def _smooth_1d(x: np.ndarray, *, window: int, polyorder: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    window = int(window)
    polyorder = int(polyorder)
    if window <= 1 or x.size < max(5, window):
        return x
    if window % 2 == 0:
        window += 1
    # Keep odd <= size.
    window = min(window, x.size - (1 - x.size % 2))
    if window < polyorder + 2:
        return x
    try:
        from scipy.signal import savgol_filter  # type: ignore

        return np.asarray(savgol_filter(x, window_length=window, polyorder=polyorder, mode="interp"), dtype=float)
    except Exception:
        k = np.ones((window,), dtype=float) / float(window)
        pad = window // 2
        x_pad = np.pad(x, (pad, pad), mode="edge")
        return np.asarray(np.convolve(x_pad, k, mode="valid"), dtype=float)


def _resample_open_polyline_xy(xy: np.ndarray, *, n: int) -> np.ndarray:
    """Resample an open polyline to exactly n points by arclength."""
    xy = np.asarray(xy, dtype=float)
    n = int(n)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 2 or n < 2:
        return xy
    dxy = np.diff(xy, axis=0)
    ds = np.sqrt(np.sum(dxy**2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0.0:
        return xy
    s_new = np.linspace(0.0, total, num=n, endpoint=True)
    x_new = np.interp(s_new, s, xy[:, 0])
    y_new = np.interp(s_new, s, xy[:, 1])
    return np.column_stack([x_new, y_new])


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
    *longest* contiguous foreground run up to base_y_i. Returns (xs, y_top_px).

    This rejects thin diagonal artifacts (short runs) while keeping true top curvature.
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


def _clip_thin_artifacts_above_cap(
    mask_u8: np.ndarray,
    *,
    enabled: bool,
    base_y_i: int,
    x_left_i: int,
    x_right_i: int,
    run_quantile: float,
) -> np.ndarray:
    """
    Clip thin diagonal artifacts above the main body, per column.

    When `enabled=False`, returns `mask_u8` unchanged.
    """
    if not bool(enabled):
        return mask_u8
    xs_cap, y_cap = _column_top_from_longest_run(
        mask_u8 > 0,
        base_y_i=int(base_y_i),
        x_left_i=int(x_left_i),
        x_right_i=int(x_right_i),
        run_quantile=float(run_quantile),
    )
    out = mask_u8.copy()
    for k, xx in enumerate(xs_cap.astype(int).tolist()):
        yt = y_cap[k]
        if not np.isfinite(float(yt)):
            continue
        yti = int(np.clip(int(round(float(yt))), 0, out.shape[0] - 1))
        if yti > 0:
            out[:yti, int(xx)] = 0
    return out


def stripe_gain_equalize(
    gray_u8: np.ndarray,
    *,
    stripe_h: int = 20,
    x_range: Optional[Tuple[int, int]] = None,
    q: float = 0.95,
    target: float = 180.0,
    gain_min: float = 0.5,
    gain_max: float = 8.0,
) -> np.ndarray:
    """
    Horizontal rectangular-stripe intensity normalization.
    For each horizontal band, compute q-quantile intensity and apply gain so that quantile maps to `target`.
    """
    gray = gray_u8.astype(np.float32)
    h, w = gray.shape
    if x_range is None:
        x0, x1 = 0, w
    else:
        x0, x1 = x_range
        x0 = int(np.clip(x0, 0, w - 1))
        x1 = int(np.clip(x1, x0 + 1, w))
    out = gray.copy()
    stripe_h = int(max(4, stripe_h))
    for y0 in range(0, h, stripe_h):
        y1 = min(h, y0 + stripe_h)
        stripe = gray[y0:y1, x0:x1]
        val = float(np.quantile(stripe, float(np.clip(q, 0.5, 0.995))))
        if val < 1e-3:
            g = gain_max
        else:
            g = target / val
        g = float(np.clip(g, gain_min, gain_max))
        out[y0:y1, :] *= g
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)


def otsu_threshold(values_u8: np.ndarray) -> int:
    """Compute Otsu threshold on a 1D uint8 array."""
    v = values_u8.astype(np.uint8).ravel()
    if v.size == 0:
        return 128
    hist = np.bincount(v, minlength=256).astype(np.float64)
    total = float(v.size)
    sum_total = float(np.dot(np.arange(256), hist))
    sumB = 0.0
    wB = 0.0
    max_var = -1.0
    thr = 0
    for t in range(256):
        wB += hist[t]
        if wB <= 0:
            continue
        wF = total - wB
        if wF <= 0:
            break
        sumB += float(t) * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > max_var:
            max_var = var_between
            thr = t
    return int(thr)


def polygon_area_xy(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def build_prior_mask_from_csv(
    csv_path: str,
    *,
    anchor_px: Tuple[int, int],
    mm_per_px: float,
    img_shape_hw: Tuple[int, int],
) -> np.ndarray:
    """
    CSV must have columns x_mm,y_mm. Will be interpreted as mm coordinates relative to anchor (0,0).
    Returns filled polygon mask (uint8 0/255).
    """
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] < 3 or data.shape[1] < 2:
        raise RuntimeError(f"Prior CSV has insufficient points: {csv_path}")
    x_mm = data[:, 0].astype(np.float64)
    y_mm = data[:, 1].astype(np.float64)
    x_px = anchor_px[0] + x_mm / float(mm_per_px)
    y_px = anchor_px[1] - y_mm / float(mm_per_px)
    pts = np.stack([x_px, y_px], axis=1).astype(np.int32)
    mask = np.zeros(img_shape_hw, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def extract_upper_boundary_from_mask(
    mask_u8: np.ndarray,
    *,
    base_left_px: int,
    base_right_px: int,
    base_y_px: int,
    roi_x0: int,
    roi_y0: int,
) -> np.ndarray:
    """
    From a cleaned ROI mask that already has:
      - pixels only above base (<= base_y)
      - a straight base line drawn
      - columns filled to base inside [base_left,base_right]
    Extract the upper boundary path from left base corner to right base corner (pixel coords in full frame).
    """
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No contours found in the final mask.")
    cont = max(contours, key=cv2.contourArea).reshape((-1, 2)).astype(np.int32)
    # Convert to full-frame coords
    cont_full = cont.copy()
    cont_full[:, 0] += int(roi_x0)
    cont_full[:, 1] += int(roi_y0)

    cornerL = np.array([int(base_left_px), int(base_y_px)], dtype=np.int32)
    cornerR = np.array([int(base_right_px), int(base_y_px)], dtype=np.int32)

    dL = np.sum((cont_full - cornerL[None, :]) ** 2, axis=1)
    dR = np.sum((cont_full - cornerR[None, :]) ** 2, axis=1)
    iL = int(np.argmin(dL))
    iR = int(np.argmin(dR))

    def path_forward(pts: np.ndarray, i0: int, i1: int) -> np.ndarray:
        if i0 <= i1:
            return pts[i0 : i1 + 1, :]
        return np.vstack([pts[i0:, :], pts[: i1 + 1, :]])

    path_lr_1 = path_forward(cont_full, iL, iR)
    path_lr_2 = path_forward(cont_full, iR, iL)[::-1, :]

    def score(path: np.ndarray) -> float:
        # Larger score => more above base.
        return float(np.mean(float(base_y_px) - path[:, 1].astype(np.float32)))

    # Choose the path that goes around the biofilm (not along the base).
    upper = path_lr_1 if score(path_lr_1) > score(path_lr_2) else path_lr_2

    # Force exact corners at endpoints.
    upper = upper.astype(np.int32)
    upper[0, :] = cornerL
    upper[-1, :] = cornerR
    return upper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True, help="Path to bit27491-sup-0001-si_v1.avi")
    ap.add_argument("--frame", type=int, default=0, help="Frame index")
    ap.add_argument("--fig5a", type=str, default="", help="Path to Figure 5a image (with red ⊗). Used to auto-calibrate anchor.")
    ap.add_argument("--anchor-json", type=str, default="", help="Read/write anchor JSON. If exists, used to keep anchor fixed.")
    ap.add_argument(
        "--use-anchor-json-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If --anchor-json exists, reuse its mm_per_px to keep scaling fixed across frames.",
    )
    ap.add_argument("--prior-csv", type=str, default="", help="Prior polygon CSV in mm (x_mm,y_mm). For frame0: traced Fig5b CSV.")
    ap.add_argument("--block-w-mm", type=float, default=1.0, help="Support width in mm (base span).")
    ap.add_argument("--scalebar-um", type=float, default=100.0)
    ap.add_argument("--scalebar-roi-frac", type=str, default="0.75,0.75,1.0,1.0")
    ap.add_argument("--scalebar-thr", type=int, default=200)

    ap.add_argument("--stripe-h", type=int, default=20, help="Stripe height in pixels for rectangular-stripe equalization.")
    ap.add_argument("--stripe-q", type=float, default=0.95, help="Quantile used per stripe.")
    ap.add_argument("--stripe-target", type=float, default=180.0, help="Target intensity for stripe quantile.")
    ap.add_argument("--stripe-gain-max", type=float, default=8.0)

    ap.add_argument("--roi-pad-px", type=int, default=120, help="Extra ROI padding around the prior polygon [px].")
    ap.add_argument("--prior-dilate", type=int, default=31, help="Dilation kernel size (odd) for prior search region.")
    ap.add_argument("--morph-k", type=int, default=7, help="Morphology kernel size for close/open.")
    ap.add_argument("--close-iters", type=int, default=2)
    ap.add_argument("--open-iters", type=int, default=1)
    ap.add_argument(
        "--seal-slits-k",
        type=int,
        default=15,
        help="Post-processing closing kernel width [px] to seal thin vertical slits before contour extraction. 0 disables.",
    )
    ap.add_argument("--seal-slits-iters", type=int, default=1, help="Iterations for --seal-slits-k closing.")

    ap.add_argument(
        "--cap-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip thin top artifacts per-column using the longest contiguous foreground run (recommended).",
    )
    ap.add_argument("--cap-run-quantile", type=float, default=0.10, help="Quantile inside the longest run used as the cap (0 keeps absolute top).")

    ap.add_argument("--smooth-ds-mm", type=float, default=0.004, help="Arclength resampling spacing for smoothing [mm].")
    ap.add_argument("--smooth-window-mm", type=float, default=0.030, help="Smoothing window along arclength [mm]. 0 disables smoothing.")
    ap.add_argument("--smooth-polyorder", type=int, default=3, help="SavGol polyorder (if SciPy available).")
    ap.add_argument("--n-verts", type=int, default=260, help="Target number of vertices on the extracted upper boundary (after smoothing). 0 keeps all.")

    ap.add_argument("--out-csv", type=str, default="poly_extracted_mm.csv")
    ap.add_argument("--debug-dir", type=str, default="debug_extract")
    ap.add_argument(
        "--debug-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write debug PNGs to --debug-dir (disable for batch processing).",
    )
    args = ap.parse_args()

    debug_dir = Path(args.debug_dir)
    if bool(args.debug_images):
        debug_dir.mkdir(parents=True, exist_ok=True)

    frame = read_video_frame(args.video, int(args.frame))
    h_full, w_full = frame.shape[:2]
    if bool(args.debug_images):
        cv2.imwrite(str(debug_dir / f"frame_{int(args.frame):04d}.png"), frame)

    # --- scale bar ---
    roi_parts = [float(x.strip()) for x in str(args.scalebar_roi_frac).split(",")]
    if len(roi_parts) != 4:
        raise ValueError("--scalebar-roi-frac must be x0,y0,x1,y1")
    roi_frac = (roi_parts[0], roi_parts[1], roi_parts[2], roi_parts[3])
    run_px = detect_scalebar_run_px(frame, roi_frac=roi_frac, thr=int(args.scalebar_thr))
    mm_per_px = (float(args.scalebar_um) * 1e-3) / float(run_px)  # um->mm then /px

    # --- anchor (fixed reference) ---
    anchor_json_path = Path(args.anchor_json) if args.anchor_json else None
    anchor_px: Tuple[int, int] | None = None

    if anchor_json_path and anchor_json_path.exists():
        meta = json.loads(anchor_json_path.read_text(encoding="utf-8"))
        anchor_px = (int(meta["anchor_px"]["x"]), int(meta["anchor_px"]["y"]))
        if bool(args.use_anchor_json_scale):
            try:
                mmpp = float(meta.get("mm_per_px", float("nan")))
                if np.isfinite(mmpp) and mmpp > 0.0:
                    mm_per_px = float(mmpp)
            except Exception:
                pass
    else:
        if not args.fig5a:
            raise RuntimeError("Need either --anchor-json (existing) or --fig5a to auto-calibrate anchor on frame0.")
        fig5a = cv2.imread(str(args.fig5a))
        if fig5a is None:
            raise RuntimeError(f"Could not read fig5a image: {args.fig5a}")

        # detect red cross in fig5a
        ax, ay = detect_red_cross_centroid(fig5a)

        # Align fig5a to the downscaled video frame using ECC affine (affine accounts for tiny resize anisotropy).
        fig5a_gray = cv2.cvtColor(fig5a, cv2.COLOR_BGR2GRAY)
        frame_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (fig5a_gray.shape[1], fig5a_gray.shape[0]), interpolation=cv2.INTER_AREA)

        # Mask out top-left label region and the red cross itself (so ECC ignores it).
        mask_ecc = np.ones_like(fig5a_gray, dtype=np.uint8) * 255
        mask_ecc[0:80, 0:140] = 0  # labels differ between fig5a and video
        hsv = cv2.cvtColor(fig5a, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        m2 = cv2.inRange(hsv, (160, 80, 80), (179, 255, 255))
        mred = cv2.bitwise_or(m1, m2)
        mred = cv2.dilate(mred, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        mask_ecc[mred > 0] = 0

        warp = ecc_affine_align(fig5a_gray, frame_small, input_mask=mask_ecc)
        warp_inv = invert_affine(warp)

        # Map anchor from fig5a coords -> frame_small coords (inverse warp).
        sx_small, sy_small = apply_affine_to_point(warp_inv, ax, ay)

        # Map from frame_small -> full-res frame coordinates via resize scale.
        sx = float(w_full) / float(fig5a_gray.shape[1])
        sy = float(h_full) / float(fig5a_gray.shape[0])
        anchor_px = (int(round(sx_small * sx)), int(round(sy_small * sy)))

        # Save anchor JSON so it stays fixed for later frames.
        if anchor_json_path:
            meta = {
                "video": str(args.video),
                "frame_calibrated": int(args.frame),
                "anchor_px": {"x": int(anchor_px[0]), "y": int(anchor_px[1])},
                "mm_per_px": float(mm_per_px),
                "scale_bar_um": float(args.scalebar_um),
                "scale_bar_run_px": int(run_px),
                "fig5a_anchor_px": {"x": float(ax), "y": float(ay)},
                "ecc_warp_small": warp.tolist(),
            }
            anchor_json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    assert anchor_px is not None
    base_left_px = int(anchor_px[0])
    base_y_px = int(anchor_px[1])
    base_w_px = int(round(float(args.block_w_mm) / float(mm_per_px)))
    base_right_px = int(base_left_px + base_w_px)

    # Debug: anchor overlay
    ov = frame.copy()
    cv2.circle(ov, (base_left_px, base_y_px), 6, (0, 0, 255), -1)
    cv2.circle(ov, (base_right_px, base_y_px), 4, (0, 255, 0), -1)
    cv2.line(ov, (base_left_px, base_y_px), (base_right_px, base_y_px), (255, 0, 0), 2)
    if bool(args.debug_images):
        cv2.imwrite(str(debug_dir / f"anchor_overlay_{int(args.frame):04d}.png"), ov)

    # --- prior mask (search region) ---
    if not args.prior_csv:
        raise RuntimeError("This implementation requires --prior-csv (frame0 traced or previous frame).")
    prior_full = build_prior_mask_from_csv(args.prior_csv, anchor_px=anchor_px, mm_per_px=mm_per_px, img_shape_hw=(h_full, w_full))

    # Tight ROI around prior polygon (expanded)
    ys, xs = np.nonzero(prior_full)
    if ys.size < 10:
        raise RuntimeError("Prior mask is empty after projection to pixels; check anchor/mm_per_px/prior CSV.")
    pad = int(max(0, int(args.roi_pad_px)))
    x0 = int(np.clip(xs.min() - pad, 0, w_full - 1))
    x1 = int(np.clip(xs.max() + pad, x0 + 1, w_full))
    y0 = int(np.clip(ys.min() - pad, 0, h_full - 1))
    y1 = int(np.clip(ys.max() + pad, y0 + 1, h_full))
    roi = frame[y0:y1, x0:x1]
    prior_roi = prior_full[y0:y1, x0:x1]

    # Dilated search region (so we don't grab background far away)
    k = int(args.prior_dilate)
    if k % 2 == 0:
        k += 1
    ker_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    search_roi = cv2.dilate(prior_roi, ker_d)

    # --- stripe enhancement + CLAHE ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # stripe x-range: use support span in ROI coordinates
    x_body0 = int(base_left_px - x0)
    x_body1 = int(base_right_px - x0)
    x_body0 = int(np.clip(x_body0, 0, gray.shape[1] - 1))
    x_body1 = int(np.clip(x_body1, x_body0 + 1, gray.shape[1]))
    gray_stripe = stripe_gain_equalize(
        gray,
        stripe_h=int(args.stripe_h),
        x_range=(x_body0, x_body1),
        q=float(args.stripe_q),
        target=float(args.stripe_target),
        gain_max=float(args.stripe_gain_max),
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enh = clahe.apply(gray_stripe)
    gray_blur = cv2.GaussianBlur(gray_enh, (5, 5), 0)
    if bool(args.debug_images):
        cv2.imwrite(str(debug_dir / f"gray_enh_{int(args.frame):04d}.png"), gray_enh)

    # --- threshold inside search region only ---
    vals = gray_blur[search_roi > 0].astype(np.uint8)
    thr = otsu_threshold(vals)
    seg = ((gray_blur > thr) & (search_roi > 0)).astype(np.uint8) * 255

    # Morphology cleanup
    mk = int(args.morph_k)
    if mk % 2 == 0:
        mk += 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    if int(args.close_iters) > 0:
        seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, ker, iterations=int(args.close_iters))
    if int(args.open_iters) > 0:
        seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, ker, iterations=int(args.open_iters))

    # Keep component with max overlap with prior
    num, labels, stats, _ = cv2.connectedComponentsWithStats(seg, connectivity=8)
    if num <= 1:
        raise RuntimeError("No foreground after segmentation; try adjusting stripe/threshold params.")
    overlap = np.zeros(num, dtype=np.int64)
    overlap += np.bincount(labels[prior_roi > 0].ravel(), minlength=num)
    overlap[0] = 0
    best = int(np.argmax(overlap))
    if best == 0 or overlap[best] < 10:
        # fallback to largest area
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas) + 1)
    mask_best = (labels == best).astype(np.uint8) * 255

    # Enforce base line and fill columns down to base (inside support span) to remove gaps at the bottom.
    base_y_roi = int(base_y_px - y0)
    base_y_roi = int(np.clip(base_y_roi, 0, mask_best.shape[0] - 1))
    # remove anything below base
    mask_best[base_y_roi + 1 :, :] = 0
    # draw straight base line
    xL = int(np.clip(x_body0, 0, mask_best.shape[1] - 1))
    xR = int(np.clip(x_body1, xL + 1, mask_best.shape[1]))
    mask_best[base_y_roi, xL:xR] = 255

    # fill columns from the top-most fg down to base to guarantee a solid domain
    # IMPORTANT: choose the top from the *longest contiguous run* in each column to
    # avoid "connecting" thin diagonal artifacts (short runs) to the main body.
    mask_bool = mask_best > 0
    rq_fill = float(np.clip(float(args.cap_run_quantile), 0.0, 0.30))
    for xx in range(int(xL), int(xR)):
        ys_col = np.where(mask_bool[: base_y_roi + 1, int(xx)])[0].astype(int)
        if ys_col.size == 0:
            continue
        run = _longest_run_1d(ys_col)
        if run is None:
            continue
        i0, i1 = run
        run_len = int(max(1, int(i1 - i0)))
        j = int(i0 + int(round(rq_fill * float(max(0, run_len - 1)))))
        j = int(np.clip(j, i0, i1 - 1))
        y_top = int(ys_col[int(j)])
        mask_best[y_top : base_y_roi + 1, int(xx)] = 255

    # Optional: clip thin diagonal artifacts above the main body, per column.
    mask_best = _clip_thin_artifacts_above_cap(
        mask_best,
        enabled=bool(args.cap_artifacts),
        base_y_i=int(base_y_roi),
        x_left_i=int(xL),
        x_right_i=int(xR - 1),
        run_quantile=float(args.cap_run_quantile),
    )

    # Seal thin vertical "slits" that are open to the background (fissures / segmentation dropouts),
    # which would otherwise create large zig-zags in the extracted external contour.
    seal_k = int(args.seal_slits_k)
    seal_it = int(args.seal_slits_iters)
    if seal_k and seal_k > 1 and seal_it > 0:
        if seal_k % 2 == 0:
            seal_k += 1
        ker_seal = cv2.getStructuringElement(cv2.MORPH_RECT, (int(seal_k), 3))
        mask_best = cv2.morphologyEx(mask_best, cv2.MORPH_CLOSE, ker_seal, iterations=int(seal_it))
        # Re-enforce the straight base and "above base" constraint.
        mask_best[base_y_roi + 1 :, :] = 0
        mask_best[base_y_roi, xL:xR] = 255

    if bool(args.debug_images):
        cv2.imwrite(str(debug_dir / f"mask_final_{int(args.frame):04d}.png"), mask_best)

    # Extract upper boundary
    upper_px = extract_upper_boundary_from_mask(
        mask_best, base_left_px=base_left_px, base_right_px=base_right_px, base_y_px=base_y_px, roi_x0=x0, roi_y0=y0
    )

    # Convert to mm relative to anchor
    x_mm = (upper_px[:, 0].astype(np.float64) - float(base_left_px)) * float(mm_per_px)
    y_mm = (float(base_y_px) - upper_px[:, 1].astype(np.float64)) * float(mm_per_px)
    # Clamp only the base-upstream side; do not cap x on the downstream side
    # (deformed biofilm can extend beyond the 1mm support span).
    x_mm = np.maximum(x_mm, 0.0)
    y_mm = np.maximum(y_mm, 0.0)

    upper_mm = np.column_stack([x_mm, y_mm])

    # Smooth/resample to remove pixel jaggedness.
    w_mm = float(max(0.0, float(args.smooth_window_mm)))
    if w_mm > 0.0 and upper_mm.shape[0] >= 4:
        ds_target = float(max(1.0e-6, float(args.smooth_ds_mm)))
        win_pts = int(max(5, int(round(w_mm / ds_target))))
        if win_pts % 2 == 0:
            win_pts += 1
        upper_mm = _smooth_open_polyline_xy(
            upper_mm,
            ds_target=ds_target,
            window_pts=int(win_pts),
            polyorder=int(args.smooth_polyorder),
        )

    if int(args.n_verts) and int(args.n_verts) > 1:
        upper_mm = _resample_open_polyline_xy(upper_mm, n=int(args.n_verts))

    # Enforce exact anchored base endpoints.
    if upper_mm.shape[0] >= 2:
        upper_mm[0, :] = np.array([0.0, 0.0], dtype=float)
        upper_mm[-1, :] = np.array([float(args.block_w_mm), 0.0], dtype=float)
    # Clamp tiny smoothing overshoots.
    if upper_mm.size:
        upper_mm[:, 0] = np.maximum(upper_mm[:, 0], 0.0)
        upper_mm[:, 1] = np.maximum(upper_mm[:, 1], 0.0)
        # Ensure the only y=0 contact is within the rigid support span.
        base_tol_mm = 1.0e-4
        on_base = upper_mm[:, 1] <= float(base_tol_mm)
        if bool(np.any(on_base)):
            upper_mm[on_base, 1] = 0.0
            upper_mm[on_base, 0] = np.clip(upper_mm[on_base, 0], 0.0, float(args.block_w_mm))

    poly = np.vstack([upper_mm, upper_mm[0, :]])  # close (base segment)

    # enforce CCW
    if polygon_area_xy(poly) < 0:
        pts = poly[:-1, :][::-1, :]
        poly = np.vstack([pts, pts[0, :]])

    # rotate so the first point is the anchor (0,0) in mm
    pts0 = poly[:-1, :]  # drop duplicate close
    d0 = np.sum((pts0 - np.array([0.0, 0.0])[None, :]) ** 2, axis=1)
    k0 = int(np.argmin(d0))
    pts0 = np.vstack([pts0[k0:, :], pts0[:k0, :]])
    poly = np.vstack([pts0, pts0[0, :]])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("x_mm,y_mm\n")
        for x, y in poly:
            f.write(f"{x:.9f},{y:.9f}\n")

    # Debug overlay: extracted polygon + prior polygon
    overlay = frame.copy()
    # prior outline in yellow
    pr_contours, _ = cv2.findContours(prior_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if pr_contours:
        cv2.drawContours(overlay, [max(pr_contours, key=cv2.contourArea)], -1, (0, 255, 255), 2)
    # extracted in green
    cv2.polylines(overlay, [upper_px.reshape((-1, 1, 2))], False, (0, 255, 0), 2)
    # smoothed (mm->px) in cyan, if enabled
    if w_mm > 0.0 and upper_mm.shape[0] >= 2:
        xs_px = (float(base_left_px) + upper_mm[:, 0] / float(mm_per_px)).round().astype(np.int32)
        ys_px = (float(base_y_px) - upper_mm[:, 1] / float(mm_per_px)).round().astype(np.int32)
        pts_px = np.column_stack([xs_px, ys_px])
        pts_px[:, 0] = np.clip(pts_px[:, 0], 0, w_full - 1)
        pts_px[:, 1] = np.clip(pts_px[:, 1], 0, h_full - 1)
        cv2.polylines(overlay, [pts_px.reshape((-1, 1, 2))], False, (255, 255, 0), 2)
    # base line
    cv2.line(overlay, (base_left_px, base_y_px), (base_right_px, base_y_px), (255, 0, 0), 2)
    # anchor
    cv2.circle(overlay, (base_left_px, base_y_px), 6, (0, 0, 255), -1)
    if bool(args.debug_images):
        cv2.imwrite(str(debug_dir / f"overlay_{int(args.frame):04d}.png"), overlay)

    print("[ok] anchor_px =", anchor_px)
    print("[ok] mm_per_px  =", mm_per_px)
    print("[ok] wrote      =", str(out_csv))
    print("[ok] debug_dir  =", str(debug_dir))


if __name__ == "__main__":
    main()
