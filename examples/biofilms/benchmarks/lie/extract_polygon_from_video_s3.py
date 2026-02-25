"""
Extract an initial biofilm geometry polygon from Li et al. Video S3 (simulated synthetic deformation).

This script reads a frame from `additional_data/bit27491-sup-0003-si_v3.avi`, segments the
red boundary curve, maps pixel coordinates to the plot's mm coordinates using detected
axis tick marks, and writes a polygon CSV (x,y).

The output polygon can be used with the one-domain benchmark driver:
  `lie_synthetic_deformation_one_domain.py --alpha0-kind polygon --alpha0-file ...`
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    from shapely.geometry import LineString  # type: ignore
except Exception:  # pragma: no cover
    LineString = None  # type: ignore[assignment]


def _read_video_frame(video_path: Path, *, frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {str(video_path)}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, img = cap.read()
    finally:
        cap.release()
    if not ok or img is None:
        raise RuntimeError(f"Failed to read frame={int(frame)} from {str(video_path)}")
    return img


def _detect_axes_box(img_bgr: np.ndarray, *, gray_thresh: int = 80, frac: float = 0.5) -> tuple[int, int, int, int]:
    """
    Detect the axis-aligned plot box by finding rows/cols containing a long dark border.

    Returns (x0, x1, y0, y1) in pixel coordinates, inclusive.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dark = gray < int(gray_thresh)
    h, w = dark.shape
    row_counts = dark.sum(axis=1)
    col_counts = dark.sum(axis=0)
    rows = np.where(row_counts > float(frac) * float(w))[0]
    cols = np.where(col_counts > float(frac) * float(h))[0]
    if rows.size == 0 or cols.size == 0:
        # Fallback: no detection.
        return 0, w - 1, 0, h - 1
    y0 = int(rows.min())
    y1 = int(rows.max())
    x0 = int(cols.min())
    x1 = int(cols.max())
    return x0, x1, y0, y1


def _red_dominance_mask(
    img_bgr: np.ndarray,
    *,
    rd_thresh: int = 20,
    r_min: int = 120,
    dilate_iters: int = 0,
    close_iters: int = 2,
    kernel_size: int = 3,
) -> np.ndarray:
    b, g, r = cv2.split(img_bgr)
    rd = r.astype(np.int16) - np.maximum(g, b).astype(np.int16)
    mask = ((rd > int(rd_thresh)) & (r > int(r_min))).astype(np.uint8) * 255
    ksz = int(kernel_size)
    if dilate_iters > 0 and ksz >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        mask = cv2.dilate(mask, kernel, iterations=int(dilate_iters))
    if close_iters > 0 and ksz >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(close_iters))
    return mask


def _largest_component(mask_u8: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        raise RuntimeError("No connected components found in mask.")
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    out = (labels == idx).astype(np.uint8) * 255
    return out


def _peak_centers_1d(
    profile: np.ndarray,
    *,
    smooth_win: int = 11,
    min_percentile: float = 95.0,
    cluster_gap: int = 15,
) -> list[int]:
    p = np.asarray(profile, dtype=float).ravel()
    if p.size < 3:
        return []
    if smooth_win and int(smooth_win) > 1:
        k = int(smooth_win)
        kernel = np.ones(k, dtype=float) / float(k)
        p = np.convolve(p, kernel, mode="same")

    # Local maxima with positive magnitude.
    peaks = np.where((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]) & (p[1:-1] > 0.0))[0] + 1
    if peaks.size == 0:
        return []
    vals = p[peaks]
    thresh = float(np.percentile(vals, float(min_percentile)))
    peaks = peaks[vals >= thresh]
    if peaks.size == 0:
        return []

    peaks = np.sort(peaks)
    clusters: list[list[int]] = []
    for pi in peaks:
        pi = int(pi)
        if not clusters or (pi - clusters[-1][-1]) > int(cluster_gap):
            clusters.append([pi])
        else:
            clusters[-1].append(pi)
    centers = [int(round(float(np.mean(c)))) for c in clusters]
    return sorted(centers)


def _detect_x_ticks(dark_mask: np.ndarray, *, band_h: int = 60, border_trim: int = 5) -> list[int]:
    H, W = dark_mask.shape
    m = dark_mask.copy()
    m[: max(0, H - int(band_h)), :] = 0
    m[max(0, H - int(border_trim)) :, :] = 0  # remove bottom border line
    m[:, : int(border_trim)] = 0
    m[:, max(0, W - int(border_trim)) :] = 0
    col_sum = m.sum(axis=0)
    return _peak_centers_1d(col_sum)


def _detect_y_ticks(dark_mask: np.ndarray, *, band_w: int = 60, border_trim: int = 5) -> list[int]:
    H, W = dark_mask.shape
    m = dark_mask.copy()
    m[:, int(band_w) :] = 0
    m[:, : int(border_trim)] = 0  # remove left border line
    m[: int(border_trim), :] = 0
    m[max(0, H - int(border_trim)) :, :] = 0
    row_sum = m.sum(axis=1)
    return _peak_centers_1d(row_sum)


def _fit_affine(px: list[int], values: list[float]) -> tuple[float, float]:
    x = np.asarray(px, dtype=float).ravel()
    y = np.asarray(values, dtype=float).ravel()
    if x.size != y.size or x.size < 2:
        raise ValueError("Need at least 2 (px,value) pairs to fit an affine map.")
    A = np.column_stack([x, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = sol
    return float(a), float(b)


def _endpoints_on_bottom(mask_u8: np.ndarray, *, band: int = 10) -> tuple[tuple[int, int], tuple[int, int]]:
    ys, xs = np.nonzero(mask_u8 > 0)
    if ys.size == 0:
        raise RuntimeError("Empty mask; cannot locate endpoints.")
    y_max = int(ys.max())
    sel = ys >= (y_max - int(band))
    xs_b = xs[sel]
    ys_b = ys[sel]
    if xs_b.size < 2:
        raise RuntimeError("Failed to identify two endpoints (not enough bottom pixels).")
    iL = int(np.argmin(xs_b))
    iR = int(np.argmax(xs_b))
    left = (int(xs_b[iL]), int(ys_b[iL]))
    right = (int(xs_b[iR]), int(ys_b[iR]))
    return left, right


def _bfs_path(mask_u8: np.ndarray, start_xy: tuple[int, int], end_xy: tuple[int, int]) -> np.ndarray:
    """
    BFS on an 8-neighbor pixel graph restricted to mask pixels.

    Returns an (N,2) array of (x,y) pixel points from start to end.
    """
    ys, xs = np.nonzero(mask_u8 > 0)
    coords = np.column_stack([xs, ys]).astype(np.int32)  # store as (x,y)
    if coords.shape[0] == 0:
        raise RuntimeError("Empty mask; cannot trace a path.")
    coord_to_idx = {(int(x), int(y)): int(i) for i, (x, y) in enumerate(coords)}
    s = coord_to_idx.get((int(start_xy[0]), int(start_xy[1])), None)
    t = coord_to_idx.get((int(end_xy[0]), int(end_xy[1])), None)
    if s is None or t is None:
        raise RuntimeError("Start/end is not on the mask pixels.")

    parent = np.full(coords.shape[0], -1, dtype=np.int32)
    q = deque([int(s)])
    parent[int(s)] = int(s)

    neigh = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dx == 0 and dy == 0)]
    found = False
    while q:
        i = int(q.popleft())
        if i == int(t):
            found = True
            break
        x, y = coords[i]
        for dx, dy in neigh:
            j = coord_to_idx.get((int(x + dx), int(y + dy)), None)
            if j is None:
                continue
            if parent[int(j)] != -1:
                continue
            parent[int(j)] = i
            q.append(int(j))

    if not found:
        raise RuntimeError("Failed to trace a connected path between endpoints.")

    path_idx = []
    cur = int(t)
    while True:
        path_idx.append(cur)
        if cur == int(s):
            break
        cur = int(parent[cur])
        if cur < 0:
            raise RuntimeError("Broken parent chain while reconstructing path.")
    path_idx.reverse()
    return coords[np.asarray(path_idx, dtype=np.int32)].astype(float)


def _simplify_polyline(points_xy: np.ndarray, *, eps: float) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        return pts
    if eps <= 0.0:
        return pts
    if LineString is None:  # pragma: no cover
        return pts
    ls = LineString(pts)
    ls2 = ls.simplify(float(eps), preserve_topology=False)
    out = np.asarray(ls2.coords, dtype=float)
    return out if out.shape[0] >= 2 else pts


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract biofilm polygon from Video S3 (simulated synthetic deformation).")
    ap.add_argument(
        "--video",
        type=str,
        default="examples/biofilms/benchmarks/lie/additional_data/bit27491-sup-0003-si_v3.avi",
        help="Path to Video S3 AVI.",
    )
    ap.add_argument("--frame", type=int, default=0, help="Frame index (0-based).")
    ap.add_argument("--out-csv", type=str, default="examples/biofilms/benchmarks/lie/biofilm_v3_frame0_polygon_mm.csv")
    ap.add_argument("--debug-dir", type=str, default="out/_lie_extract")

    # Axis detection + mapping defaults for the provided Video S3 plot.
    ap.add_argument("--gray-thresh", type=int, default=80, help="Threshold for dark pixel detection in grayscale.")
    ap.add_argument("--axes-frac", type=float, default=0.5, help="Fraction of width/height to detect the plot border.")
    ap.add_argument("--x-tick-start", type=float, default=4.0, help="x tick value at the leftmost detected x tick [mm].")
    ap.add_argument("--x-tick-step", type=float, default=0.5, help="x tick increment [mm].")
    ap.add_argument("--y-tick-top", type=float, default=3.2, help="y tick value at the topmost detected y tick [mm].")
    ap.add_argument("--y-tick-step", type=float, default=-0.2, help="y tick increment (negative downward) [mm].")

    # Red segmentation
    ap.add_argument("--rd-thresh", type=int, default=20, help="Red dominance threshold.")
    ap.add_argument("--r-min", type=int, default=120, help="Minimum red channel value.")
    ap.add_argument("--close-iters", type=int, default=2, help="Morphological close iterations.")
    ap.add_argument("--kernel-size", type=int, default=3, help="Morphological kernel size.")
    ap.add_argument("--dilate-iters", type=int, default=1, help="Morphological dilation iterations (helps connect the full curve).")

    # Path + output
    ap.add_argument("--endpoint-band", type=int, default=10, help="Bottom band (px) for endpoint detection.")
    ap.add_argument("--simplify-eps-px", type=float, default=2.0, help="Polyline simplification tolerance [px].")
    ap.add_argument("--output-units", type=str, default="mm", choices=("mm", "m"))

    args = ap.parse_args()

    video_path = Path(str(args.video))
    out_csv = Path(str(args.out_csv))
    debug_dir = Path(str(args.debug_dir))
    debug_dir.mkdir(parents=True, exist_ok=True)

    img = _read_video_frame(video_path, frame=int(args.frame))
    cv2.imwrite(str(debug_dir / "frame.png"), img)

    x0, x1, y0, y1 = _detect_axes_box(img, gray_thresh=int(args.gray_thresh), frac=float(args.axes_frac))
    crop = img[y0 : y1 + 1, x0 : x1 + 1]
    cv2.imwrite(str(debug_dir / "axes_crop.png"), crop)

    # Coordinate mapping from ticks (in crop coords).
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = (gray < int(args.gray_thresh)).astype(np.uint8)
    x_ticks = _detect_x_ticks(dark)
    y_ticks = _detect_y_ticks(dark)
    if len(x_ticks) < 2 or len(y_ticks) < 2:
        raise RuntimeError(f"Tick detection failed (x_ticks={len(x_ticks)}, y_ticks={len(y_ticks)}).")

    x_ticks = sorted(x_ticks)
    y_ticks = sorted(y_ticks)  # y increases downward in pixel coords (topmost first)

    x_vals = [float(args.x_tick_start) + float(args.x_tick_step) * i for i in range(len(x_ticks))]
    y_vals = [float(args.y_tick_top) + float(args.y_tick_step) * i for i in range(len(y_ticks))]

    ax, bx = _fit_affine(x_ticks, x_vals)
    ay, by = _fit_affine(y_ticks, y_vals)

    # Segment the red curve.
    red = _red_dominance_mask(
        crop,
        rd_thresh=int(args.rd_thresh),
        r_min=int(args.r_min),
        dilate_iters=int(args.dilate_iters),
        close_iters=int(args.close_iters),
        kernel_size=int(args.kernel_size),
    )
    cv2.imwrite(str(debug_dir / "red_mask.png"), red)
    red_l = _largest_component(red)
    cv2.imwrite(str(debug_dir / "red_mask_largest.png"), red_l)

    left_px, right_px = _endpoints_on_bottom(red_l, band=int(args.endpoint_band))
    path_px = _bfs_path(red_l, left_px, right_px)  # (N,2) (x,y) in crop coords
    path_px = _simplify_polyline(path_px, eps=float(args.simplify_eps_px))

    # Map to mm coordinates.
    x_mm = ax * path_px[:, 0] + bx
    y_mm = ay * path_px[:, 1] + by
    path_mm = np.column_stack([x_mm, y_mm]).astype(float)

    # Close on a straight base segment.
    base_y = float(min(path_mm[0, 1], path_mm[-1, 1]))
    path_mm[0, 1] = base_y
    path_mm[-1, 1] = base_y
    poly = np.vstack([path_mm, path_mm[0]])

    if str(args.output_units).lower() == "m":
        poly = 1.0e-3 * poly

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = "x_mm,y_mm" if str(args.output_units).lower() == "mm" else "x_m,y_m"
    np.savetxt(str(out_csv), poly, delimiter=",", header=header, comments="")

    # Debug overlay
    overlay = crop.copy()
    # Draw the path in pixel space (polyline on crop).
    pts = np.round(path_px).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [pts], isClosed=False, color=(0, 255, 0), thickness=2)
    cv2.circle(overlay, left_px, 6, (255, 0, 0), -1)
    cv2.circle(overlay, right_px, 6, (0, 0, 255), -1)
    cv2.imwrite(str(debug_dir / "overlay_path.png"), overlay)

    print(f"[ok] wrote polygon: {str(out_csv)}")
    print(f"[ok] debug outputs: {str(debug_dir)}")
    print(f"[map] x_mm = {ax:.6g} * x_px + {bx:.6g}")
    print(f"[map] y_mm = {ay:.6g} * y_px + {by:.6g}")
    print(f"[ticks] x_ticks_px={x_ticks}, x_vals_mm={x_vals}")
    print(f"[ticks] y_ticks_px={y_ticks}, y_vals_mm={y_vals}")


if __name__ == "__main__":
    main()