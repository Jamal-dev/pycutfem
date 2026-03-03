#!/usr/bin/env python3
"""
Extract experimental deformation dx(t) from manually traced Inkscape SVG outlines (Lie benchmark).

Motivation
----------
The OCT Video S1 frames can be difficult to segment robustly (jagged contours, anchor drift).
This utility uses *manual* per-frame traces (SVG) to build a cleaner, publishable dx(t)
signal for calibrating/validating the one-domain model.

Outputs
-------
Writes a CSV with columns:
  t_s, dx_line1_m, dx_line2_m, dx_line3_m

Also writes a t=0 polygon CSV in mm coordinates (x_mm,y_mm), suitable as --alpha0-file
for `lie_synthetic_deformation_one_domain.py`.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from examples.biofilms.benchmarks.lie.svg_trace_utils import extract_svg_frame_geometry


def _parse_float_list(s: str, *, n: int | None = None) -> list[float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    out = [float(p) for p in parts]
    if n is not None and len(out) != int(n):
        raise ValueError(f"Expected {int(n)} comma-separated values; got {len(out)} in {s!r}")
    return out


def _polyline_arclength(xy: np.ndarray) -> tuple[np.ndarray, float]:
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 2:
        return np.zeros((0,), dtype=float), 0.0
    d = np.diff(xy, axis=0)
    ds = np.sqrt(np.sum(d * d, axis=1))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = float(s[-1]) if s.size else 0.0
    return np.asarray(s, dtype=float), float(total)


def _resample_open_polyline_xy(xy: np.ndarray, *, n: int) -> np.ndarray:
    """Resample an open polyline to exactly n points by arclength."""
    xy = np.asarray(xy, dtype=float)
    n = int(n)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 2 or n < 2:
        return xy
    s, total = _polyline_arclength(xy)
    if not np.isfinite(total) or total <= 0.0:
        return xy
    s_new = np.linspace(0.0, total, num=n, endpoint=True)
    x_new = np.interp(s_new, s, xy[:, 0])
    y_new = np.interp(s_new, s, xy[:, 1])
    return np.column_stack([x_new, y_new])


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
        k = np.ones((window,), dtype=float) / float(window)
        pad = window // 2
        x_pad = np.pad(x, (pad, pad), mode="edge")
        return np.asarray(np.convolve(x_pad, k, mode="valid"), dtype=float)


def _smooth_open_polyline_xy(
    xy: np.ndarray,
    *,
    ds_target: float,
    window_mm: float,
    polyorder: int,
    n_out: int,
) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 4:
        return xy
    ds_target = float(max(1.0e-6, float(ds_target)))
    window_mm = float(window_mm)
    if not np.isfinite(window_mm) or window_mm <= 0.0:
        return _resample_open_polyline_xy(xy, n=int(n_out)) if int(n_out) > 0 else xy

    s, total = _polyline_arclength(xy)
    if not np.isfinite(total) or total <= 0.0:
        return xy
    n_mid = int(max(80, int(round(total / ds_target))))
    s_mid = np.linspace(0.0, total, num=n_mid, endpoint=True)
    x_mid = np.interp(s_mid, s, xy[:, 0])
    y_mid = np.interp(s_mid, s, xy[:, 1])

    window_pts = int(max(5, int(round(float(window_mm) / float(ds_target)))))
    x_s = _smooth_1d(x_mid, window=window_pts, polyorder=int(polyorder))
    y_s = _smooth_1d(y_mid, window=window_pts, polyorder=int(polyorder))
    xy_s = np.column_stack([x_s, y_s])
    return _resample_open_polyline_xy(xy_s, n=int(n_out)) if int(n_out) > 0 else xy_s


def _x_intersection_on_y(poly_xy: np.ndarray, *, y: float, mode: str) -> float:
    """
    Intersect a closed polygon with a horizontal line y=const and return the leftmost/rightmost x.
    Coordinates are in the same units as the polygon (here: mm).
    """
    p = np.asarray(poly_xy, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 4:
        return float("nan")
    if not np.allclose(p[0], p[-1], rtol=0.0, atol=1.0e-12):
        p = np.vstack([p, p[0]])

    y0 = float(y)
    mode = str(mode).strip().lower()
    if mode not in {"leftmost", "rightmost"}:
        raise ValueError(f"Unknown mode {mode!r}")

    xs: list[float] = []
    # Half-open rule to avoid double-counting vertices.
    for a, b in zip(p[:-1], p[1:]):
        x1, y1 = float(a[0]), float(a[1])
        x2, y2 = float(b[0]), float(b[1])
        if abs(y2 - y1) <= 1.0e-14:
            continue
        y_lo = min(y1, y2)
        y_hi = max(y1, y2)
        if not (y_lo < y0 <= y_hi):
            continue
        t = (y0 - y1) / (y2 - y1)
        if 0.0 <= t <= 1.0:
            xs.append(x1 + t * (x2 - x1))
    if not xs:
        return float("nan")
    return float(min(xs)) if mode == "leftmost" else float(max(xs))


def _x_quantile_on_y(poly_xy: np.ndarray, *, y: float, mode: str, q: float) -> float:
    """
    Intersect a closed polygon with y=const and return an interior-point x location.

    Definition matches `lie_synthetic_deformation_one_domain.py --dx-quantile`:
      - find the leftmost (xL) and rightmost (xR) intersections of the biofilm cross-section,
      - for mode=leftmost:  x = xL + q * (xR - xL)
      - for mode=rightmost: x = xR - q * (xR - xL)

    For q=0, this reduces to the boundary intersection used previously.
    For q>0, we use the widest inside segment when multiple intersections exist.
    """
    q = float(np.clip(float(q), 0.0, 1.0))
    mode = str(mode).strip().lower()
    if mode not in {"leftmost", "rightmost"}:
        raise ValueError(f"Unknown mode {mode!r}")

    # Backwards-compatible: boundary point.
    if q <= 0.0:
        return float(_x_intersection_on_y(poly_xy, y=float(y), mode=str(mode)))

    p = np.asarray(poly_xy, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 4:
        return float("nan")
    if not np.allclose(p[0], p[-1], rtol=0.0, atol=1.0e-12):
        p = np.vstack([p, p[0]])

    y0 = float(y)
    xs: list[float] = []
    # Half-open rule to avoid double-counting vertices.
    for a, b in zip(p[:-1], p[1:]):
        x1, y1 = float(a[0]), float(a[1])
        x2, y2 = float(b[0]), float(b[1])
        if abs(y2 - y1) <= 1.0e-14:
            continue
        y_lo = min(y1, y2)
        y_hi = max(y1, y2)
        if not (y_lo < y0 <= y_hi):
            continue
        t = (y0 - y1) / (y2 - y1)
        if 0.0 <= t <= 1.0:
            xs.append(float(x1 + t * (x2 - x1)))
    if len(xs) < 2:
        return float("nan")
    xs_s = np.sort(np.asarray(xs, dtype=float))
    if xs_s.size < 2:
        return float("nan")
    if int(xs_s.size) % 2 == 1:
        # Degenerate scanline hit; drop the last intersection to form pairs.
        xs_s = xs_s[:-1]
    if xs_s.size < 2:
        return float("nan")

    # Inside segments are given by pairs (x0,x1), (x2,x3), ...
    segs = xs_s.reshape((-1, 2))
    widths = segs[:, 1] - segs[:, 0]
    j = int(np.argmax(widths))
    xL = float(segs[j, 0])
    xR = float(segs[j, 1])
    if not (np.isfinite(xL) and np.isfinite(xR)):
        return float("nan")
    if float(xR) < float(xL):
        xL, xR = xR, xL
    if mode == "leftmost":
        return float(xL + q * (xR - xL))
    return float(xR - q * (xR - xL))


def _x_quantile_candidates_on_y(poly_xy: np.ndarray, *, y: float, mode: str, q: float) -> list[float]:
    """
    Return candidate x locations (one per inside segment) for continuity-based tracking.

    Only intended for q>0. For q<=0, returns a single boundary candidate.
    """
    q = float(np.clip(float(q), 0.0, 1.0))
    mode = str(mode).strip().lower()
    if mode not in {"leftmost", "rightmost"}:
        raise ValueError(f"Unknown mode {mode!r}")
    if q <= 0.0:
        return [float(_x_intersection_on_y(poly_xy, y=float(y), mode=str(mode)))]

    p = np.asarray(poly_xy, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 4:
        return []
    if not np.allclose(p[0], p[-1], rtol=0.0, atol=1.0e-12):
        p = np.vstack([p, p[0]])

    y0 = float(y)
    xs: list[float] = []
    for a, b in zip(p[:-1], p[1:]):
        x1, y1 = float(a[0]), float(a[1])
        x2, y2 = float(b[0]), float(b[1])
        if abs(y2 - y1) <= 1.0e-14:
            continue
        y_lo = min(y1, y2)
        y_hi = max(y1, y2)
        if not (y_lo < y0 <= y_hi):
            continue
        t = (y0 - y1) / (y2 - y1)
        if 0.0 <= t <= 1.0:
            xs.append(float(x1 + t * (x2 - x1)))
    if len(xs) < 2:
        return []
    xs_s = np.sort(np.asarray(xs, dtype=float))
    if int(xs_s.size) % 2 == 1:
        xs_s = xs_s[:-1]
    if xs_s.size < 2:
        return []
    segs = xs_s.reshape((-1, 2))
    out: list[float] = []
    for x0, x1 in segs.tolist():
        xL = float(min(x0, x1))
        xR = float(max(x0, x1))
        if not (np.isfinite(xL) and np.isfinite(xR)):
            continue
        if mode == "leftmost":
            out.append(float(xL + q * (xR - xL)))
        else:
            out.append(float(xR - q * (xR - xL)))
    return out


def _auto_select_y_fracs(
    *,
    polys_mm: dict[int, np.ndarray],
    hb_mm: float,
    candidate_fracs: np.ndarray,
    x_mode: str,
) -> list[float]:
    """
    Pick 3 y-fractions by minimizing a simple roughness metric on x(y,t).
    We bias toward "middle + upper" by selecting one fraction in each band.
    """
    # Compute roughness metric for each candidate.
    frs = sorted(polys_mm.keys())
    frs_arr = np.array(frs, dtype=int)

    def rough_for(frac: float) -> float:
        y = float(frac) * float(hb_mm)
        xs = []
        for fr in frs:
            x = _x_intersection_on_y(polys_mm[fr], y=y, mode=str(x_mode))
            xs.append(float(x))
        x = np.asarray(xs, dtype=float)
        ok = np.isfinite(x)
        if int(np.sum(ok)) < 10:
            return float("inf")
        x = x[ok]
        if x.size < 5:
            return float("inf")
        # Robust 2nd-difference roughness.
        d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
        return float(np.median(np.abs(d2)))

    rough = np.array([rough_for(float(f)) for f in candidate_fracs], dtype=float)
    # Bands: upper, middle, lower-middle (avoid near-base).
    bands = [(0.70, 0.90), (0.50, 0.70), (0.35, 0.50)]
    chosen: list[float] = []
    for lo, hi in bands:
        mask = (candidate_fracs >= float(lo)) & (candidate_fracs <= float(hi))
        if not bool(np.any(mask)):
            continue
        idx = np.argmin(np.where(mask, rough, float("inf")))
        chosen.append(float(candidate_fracs[idx]))
    # Fallback: if any band failed, just take the globally best 3, preferring larger frac.
    if len(chosen) != 3:
        order = np.lexsort((-candidate_fracs, rough))
        chosen = [float(candidate_fracs[i]) for i in order[:3].tolist()]
    # Return sorted from top (line1) to bottom (line3).
    chosen = sorted(chosen, reverse=True)
    return [float(v) for v in chosen]


@dataclass(frozen=True)
class _Series:
    fps: float
    frames: list[int]
    poly_mm: dict[int, np.ndarray]  # closed polygons in mm
    boundary_mm: dict[int, np.ndarray]  # open boundary polylines in mm


def _load_svg_series(
    *,
    svg_dir: Path,
    fps: float,
    m_per_px: float,
    cubic_samples: int,
    join_tol_px: float,
) -> _Series:
    svg_re = re.compile(r"frame_(\d+)\.svg$")
    poly_mm: dict[int, np.ndarray] = {}
    boundary_mm: dict[int, np.ndarray] = {}
    frames: list[int] = []
    for p in sorted(svg_dir.glob("frame_*.svg")):
        m = svg_re.search(p.name)
        if not m:
            continue
        fr = int(m.group(1))
        g = extract_svg_frame_geometry(p, m_per_px=float(m_per_px), cubic_samples=int(cubic_samples), join_tol_px=float(join_tol_px))
        poly = np.asarray(g.polygon_mm, dtype=float)
        if poly.shape[0] < 4:
            continue
        poly_mm[fr] = poly
        boundary_mm[fr] = g.px_to_mm(g.boundary_px)
        frames.append(fr)
    frames = sorted(set(frames))
    if not frames:
        raise RuntimeError(f"No SVG frames found in {svg_dir}")
    return _Series(fps=float(fps), frames=frames, poly_mm=poly_mm, boundary_mm=boundary_mm)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract dx(t) from SVG-traced outlines (Lie benchmark).")
    ap.add_argument("--svg-dir", type=str, default="examples/biofilms/benchmarks/lie/svg_fles")
    ap.add_argument(
        "--video",
        type=str,
        default="examples/biofilms/benchmarks/lie/additional_data/bit27491-sup-0001-si_v1.avi",
        help="Used only to read FPS if --fps is not provided.",
    )
    ap.add_argument("--fps", type=float, default=float("nan"), help="Frames per second. Default: read from --video.")

    ap.add_argument(
        "--scale-mode",
        type=str,
        default="m_per_px",
        choices=("m_per_px", "base_frame0"),
        help="How to scale SVG pixels to physical units. 'base_frame0' enforces block width on frame 0 base.",
    )
    ap.add_argument("--block-w-mm", type=float, default=1.0, help="Support width used for base scaling (mm).")
    ap.add_argument(
        "--m-per-px",
        type=float,
        default=2.040816e-6,
        help="Pixel-to-meter scaling. Default: 100 um / 49 px (from the scale-bar workflow).",
    )
    ap.add_argument("--x-intersection", type=str, default="leftmost", choices=("leftmost", "rightmost"))
    ap.add_argument("--dx-quantile", type=float, default=0.0, help="Interior-point quantile (0=boundary).")
    ap.add_argument(
        "--track-mode",
        type=str,
        default="independent",
        choices=("independent", "continuity"),
        help="How to choose the intersection when multiple inside segments exist.",
    )
    ap.add_argument("--max-jump-mm", type=float, default=0.25, help="Max allowed per-frame x jump for continuity tracking (mm).")
    ap.add_argument("--y-fracs-mode", type=str, default="paper", choices=("paper", "robust", "auto"))
    ap.add_argument("--y-fracs", type=str, default="", help="Override y-fractions, e.g. '0.75,0.5,0.25'.")

    # SVG sampling / smoothing.
    ap.add_argument("--cubic-samples", type=int, default=20, help="Samples per cubic Bezier segment.")
    ap.add_argument("--join-tol-px", type=float, default=5.0, help="Endpoint join tolerance (pixels).")

    ap.add_argument("--smooth-ds-mm", type=float, default=0.004, help="Arclength step for smoothing/resampling (mm).")
    ap.add_argument("--smooth-window-mm", type=float, default=0.03, help="Smoothing window length (mm). 0 disables smoothing.")
    ap.add_argument("--smooth-polyorder", type=int, default=2, help="Savitzky-Golay polyorder (if SciPy available).")
    ap.add_argument("--n-verts", type=int, default=260, help="Vertices for saved poly0 boundary (open) and for dx evaluation.")

    ap.add_argument(
        "--out-csv",
        type=str,
        default="examples/biofilms/benchmarks/lie/exp_s1_dx_leftmost_svgtrace.csv",
        help="Output deformation CSV.",
    )
    ap.add_argument(
        "--out-poly0-mm-csv",
        type=str,
        default="examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_svgtrace.csv",
        help="Output t=0 polygon in mm (closed).",
    )
    args = ap.parse_args()

    svg_dir = Path(str(args.svg_dir))
    if not svg_dir.exists():
        raise FileNotFoundError(f"SVG directory not found: {svg_dir}")

    fps = float(args.fps)
    if not np.isfinite(fps) or fps <= 0.0:
        import cv2

        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video to read FPS: {args.video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if not np.isfinite(fps) or fps <= 0.0:
            fps = 1.0

    scale_mode = str(args.scale_mode).strip().lower()
    block_w_mm = float(args.block_w_mm)
    if not np.isfinite(block_w_mm) or block_w_mm <= 0.0:
        raise ValueError("--block-w-mm must be >0")

    if scale_mode == "m_per_px":
        m_per_px = float(args.m_per_px)
        if not np.isfinite(m_per_px) or m_per_px <= 0.0:
            raise ValueError("--m-per-px must be >0")
    else:
        # Scale from the traced base width on frame 0.
        fr0_svg = svg_dir / "frame_0000.svg"
        if not fr0_svg.exists():
            cand = sorted(svg_dir.glob("frame_*.svg"))
            if not cand:
                raise RuntimeError(f"No SVG frames found in {svg_dir}")
            fr0_svg = cand[0]
        g0 = extract_svg_frame_geometry(fr0_svg, block_w_mm=float(block_w_mm), m_per_px=None, cubic_samples=int(args.cubic_samples), join_tol_px=float(args.join_tol_px))
        dx0 = float(g0.base_right_px[0] - g0.base_left_px[0])
        dy0 = float(g0.base_right_px[1] - g0.base_left_px[1])
        base_len_px = float(math.hypot(dx0, dy0))
        if not np.isfinite(base_len_px) or base_len_px <= 1.0e-9:
            raise RuntimeError(f"Degenerate base length in {fr0_svg} (len_px={base_len_px})")
        m_per_px = (float(block_w_mm) / float(base_len_px)) * 1.0e-3

    series = _load_svg_series(
        svg_dir=svg_dir,
        fps=float(fps),
        m_per_px=float(m_per_px),
        cubic_samples=int(args.cubic_samples),
        join_tol_px=float(args.join_tol_px),
    )

    # Reference polygon (frame 0) and height.
    fr0 = 0 if 0 in series.poly_mm else int(series.frames[0])
    poly0 = np.asarray(series.poly_mm[fr0], dtype=float)
    hb_mm = float(np.nanmax(poly0[:, 1]))
    if not np.isfinite(hb_mm) or hb_mm <= 0.0:
        raise RuntimeError(f"Invalid H_b from frame {fr0}: {hb_mm}")

    if str(args.y_fracs).strip():
        y_fracs = _parse_float_list(str(args.y_fracs), n=3)
    else:
        mode = str(args.y_fracs_mode).strip().lower()
        if mode == "paper":
            y_fracs = [0.75, 0.50, 0.25]
        elif mode == "robust":
            y_fracs = [0.80, 0.60, 0.40]
        else:
            cand = np.arange(0.35, 0.91, 0.05, dtype=float)
            y_fracs = _auto_select_y_fracs(polys_mm=series.poly_mm, hb_mm=hb_mm, candidate_fracs=cand, x_mode=str(args.x_intersection))

    y_lines_mm = [float(f) * float(hb_mm) for f in y_fracs]

    # Use smoothed/resampled boundary for dx extraction to remove jaggedness.
    boundary0 = np.asarray(series.boundary_mm[fr0], dtype=float)
    boundary0_s = _smooth_open_polyline_xy(
        boundary0,
        ds_target=float(args.smooth_ds_mm),
        window_mm=float(args.smooth_window_mm),
        polyorder=int(args.smooth_polyorder),
        n_out=int(args.n_verts),
    )
    # Enforce exact base endpoints at y=0.
    boundary0_s[0, :] = boundary0[0, :]
    boundary0_s[-1, :] = boundary0[-1, :]
    poly0_s = np.vstack([boundary0_s, boundary0_s[0]])

    # Save poly0 (smoothed).
    out_poly0 = Path(str(args.out_poly0_mm_csv))
    out_poly0.parent.mkdir(parents=True, exist_ok=True)
    with out_poly0.open("w", encoding="utf-8") as f:
        f.write("x_mm,y_mm\n")
        for x, y in np.asarray(poly0_s, dtype=float):
            f.write(f"{float(x):.9f},{float(y):.9f}\n")

    # Reference x positions.
    dx_q = float(np.clip(float(args.dx_quantile), 0.0, 1.0))
    x_ref_mm = [float(_x_quantile_on_y(poly0_s, y=y, mode=str(args.x_intersection), q=dx_q)) for y in y_lines_mm]

    # Compute dx(t).
    rows: list[tuple[float, float, float, float]] = []
    x_prev_mm = list(x_ref_mm)
    track_mode = str(args.track_mode).strip().lower()
    max_jump_mm = float(abs(float(args.max_jump_mm)))
    for fr in series.frames:
        t_s = float(fr) / float(series.fps)
        b = np.asarray(series.boundary_mm[fr], dtype=float)
        b_s = _smooth_open_polyline_xy(
            b,
            ds_target=float(args.smooth_ds_mm),
            window_mm=float(args.smooth_window_mm),
            polyorder=int(args.smooth_polyorder),
            n_out=int(args.n_verts),
        )
        b_s[0, :] = b[0, :]
        b_s[-1, :] = b[-1, :]
        poly = np.vstack([b_s, b_s[0]])
        if track_mode == "continuity":
            xs_mm = []
            for i, y in enumerate(y_lines_mm):
                cand = _x_quantile_candidates_on_y(poly, y=float(y), mode=str(args.x_intersection), q=dx_q)
                if not cand:
                    xs_mm.append(float("nan"))
                    continue
                prev = float(x_prev_mm[i])
                j = int(np.argmin([abs(float(c) - prev) for c in cand]))
                x = float(cand[j])
                if np.isfinite(prev) and np.isfinite(x) and np.isfinite(max_jump_mm) and max_jump_mm > 0.0:
                    if abs(float(x) - prev) > float(max_jump_mm):
                        x = float("nan")
                xs_mm.append(float(x))
                if np.isfinite(x):
                    x_prev_mm[i] = float(x)
        else:
            xs_mm = [float(_x_quantile_on_y(poly, y=y, mode=str(args.x_intersection), q=dx_q)) for y in y_lines_mm]
        dx_m = [(x - xr) * 1.0e-3 if (np.isfinite(x) and np.isfinite(xr)) else float("nan") for x, xr in zip(xs_mm, x_ref_mm)]
        rows.append((t_s, float(dx_m[0]), float(dx_m[1]), float(dx_m[2])))

    out_csv = Path(str(args.out_csv))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("t_s,dx_line1_m,dx_line2_m,dx_line3_m\n")
        for t_s, d1, d2, d3 in rows:
            f.write(f"{t_s:.12e},{d1:.12e},{d2:.12e},{d3:.12e}\n")

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_poly0}")
    print(f"[info] svg_dir={svg_dir} frames={len(series.frames)} fps={series.fps:g} (t_final~{series.frames[-1]/series.fps:.3g}s)")
    print(f"[info] scale_mode={scale_mode} block_w_mm={block_w_mm:g}")
    print(f"[info] m_per_px={m_per_px:.6e} -> um_per_px={m_per_px*1e6:.3g}")
    print(f"[info] frame0={fr0} H_b={hb_mm:.4f} mm")
    print(f"[info] y_fracs={','.join(f'{v:.3g}' for v in y_fracs)} -> y_lines_mm={','.join(f'{y:.4f}' for y in y_lines_mm)}")
    print(f"[info] dx_quantile={dx_q:.3g} (mode={args.x_intersection}) track_mode={track_mode} max_jump_mm={max_jump_mm:g}")


if __name__ == "__main__":
    main()
