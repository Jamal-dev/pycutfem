#!/usr/bin/env python3
"""
Digitize experimental displacement curves from Li et al. (2020) Fig. 7 (b/c/d) plots.

Motivation
----------
The paper reports dx(t) for three tracking lines (Fig. 7b/c/d) using circle markers
for experimental data. This script extracts those circle-marker coordinates directly
from the PNGs in this folder:
  - li_fig7_b.png  (line 1 / top)
  - li_fig7_c.png  (line 2 / middle)
  - li_fig7_d.png  (line 3 / low)

Outputs
-------
Writes a CSV with columns:
  t_s, dx_line1_m, dx_line2_m, dx_line3_m

Implementation notes
--------------------
We use simple computer-vision heuristics:
  - detect the plot bounding box (axes rectangle) via Hough line detection,
  - detect circle markers via Hough circles,
  - map pixel coordinates -> (t, dx) using the known axes ranges,
  - bin in time and pick the upper-most circle per bin (removes most false positives),
  - resample all three curves onto a common time grid.

This is intended to be *reproducible* rather than perfect. Always inspect the
generated debug images if you change OpenCV parameters.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PlotBox:
    x_left: float
    x_right: float
    y_top: float
    y_bottom: float

    def shrink(self, margin_px: float) -> "PlotBox":
        m = float(max(0.0, float(margin_px)))
        return PlotBox(
            x_left=float(self.x_left + m),
            x_right=float(self.x_right - m),
            y_top=float(self.y_top + m),
            y_bottom=float(self.y_bottom - m),
        )


def _detect_plot_box(gray_u8: np.ndarray, *, min_line_len_px: int, max_line_gap_px: int, min_len_factor: float) -> PlotBox:
    edges = cv2.Canny(gray_u8, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=60,
        minLineLength=int(min_line_len_px),
        maxLineGap=int(max_line_gap_px),
    )
    if lines is None:
        raise RuntimeError("Could not detect plot axes via HoughLinesP.")

    horiz: list[tuple[float, float]] = []  # (len, y_mid)
    vert: list[tuple[float, float]] = []  # (len, x_mid)
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        L = float((dx * dx + dy * dy) ** 0.5)
        if L <= 1.0e-9:
            continue
        ang = float(np.degrees(np.arctan2(dy, dx)))
        # Horizontal: ~0 or ~180 degrees.
        if abs(ang) < 10.0 or abs(abs(ang) - 180.0) < 10.0:
            horiz.append((L, 0.5 * float(y1 + y2)))
        # Vertical: ~90 degrees.
        elif abs(abs(ang) - 90.0) < 10.0:
            vert.append((L, 0.5 * float(x1 + x2)))

    if not horiz or not vert:
        raise RuntimeError("Failed detecting sufficient horizontal/vertical axis lines.")

    max_h = max(L for L, _ in horiz)
    max_v = max(L for L, _ in vert)
    keep_h = [(L, y) for (L, y) in horiz if L >= float(min_len_factor) * float(max_h)]
    keep_v = [(L, x) for (L, x) in vert if L >= float(min_len_factor) * float(max_v)]
    if not keep_h or not keep_v:
        raise RuntimeError("Axis-line filtering removed all candidates; adjust min_len_factor/min_line_len_px.")

    y_top = min(y for _, y in keep_h)
    y_bottom = max(y for _, y in keep_h)
    x_left = min(x for _, x in keep_v)
    x_right = max(x for _, x in keep_v)
    if not (x_right > x_left and y_bottom > y_top):
        raise RuntimeError(f"Invalid plot box: x_left={x_left}, x_right={x_right}, y_top={y_top}, y_bottom={y_bottom}")
    return PlotBox(x_left=float(x_left), x_right=float(x_right), y_top=float(y_top), y_bottom=float(y_bottom))


def _detect_circles(
    gray_u8: np.ndarray,
    *,
    dp: float,
    min_dist: float,
    param1: float,
    param2: float,
    min_radius: int,
    max_radius: int,
) -> np.ndarray:
    g = cv2.GaussianBlur(gray_u8, (3, 3), 0)
    circles = cv2.HoughCircles(
        g,
        cv2.HOUGH_GRADIENT,
        dp=float(dp),
        minDist=float(min_dist),
        param1=float(param1),
        param2=float(param2),
        minRadius=int(min_radius),
        maxRadius=int(max_radius),
    )
    if circles is None:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(circles[0, :, :], dtype=float)  # (x,y,r)


def _px_to_data(
    *,
    x_px: float,
    y_px: float,
    box: PlotBox,
    t_max_s: float,
    dx_max_um: float,
) -> tuple[float, float]:
    t = (float(x_px) - float(box.x_left)) / max(1.0e-30, float(box.x_right - box.x_left)) * float(t_max_s)
    # Image y increases down; data dx increases up.
    dx_um = (float(box.y_bottom) - float(y_px)) / max(1.0e-30, float(box.y_bottom - box.y_top)) * float(dx_max_um)
    return float(t), float(dx_um)


def _bin_candidates(points: np.ndarray, *, bin_s: float) -> list[tuple[float, np.ndarray]]:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        return []
    bin_s = float(max(1.0e-9, float(bin_s)))
    k = np.round(pts[:, 0] / bin_s).astype(int)
    bins: list[tuple[float, np.ndarray]] = []
    for kk in sorted(set(k.tolist())):
        sel = pts[k == kk, :]
        t_bin = float(kk) * float(bin_s)
        bins.append((t_bin, np.asarray(sel[:, 1], dtype=float)))
    return bins


def _pick_curve(points: np.ndarray, *, bin_s: float, mode: str, enforce_monotone: bool) -> np.ndarray:
    """
    Reduce a (N,2) array of (t,dx_um) into one (t_bin,dx_um) point per time bin.

    mode:
      - 'topmost': pick max(dx) per bin
      - 'smooth':  pick a smooth, mostly-monotone path via dynamic programming
    """
    bins = _bin_candidates(points, bin_s=float(bin_s))
    if not bins:
        return np.zeros((0, 2), dtype=float)

    mode = str(mode).strip().lower()
    if mode not in {"topmost", "smooth"}:
        raise ValueError(f"Unknown pick mode {mode!r}. Use 'topmost' or 'smooth'.")

    if mode == "topmost":
        out = np.array([[t, float(np.nanmax(dx))] for t, dx in bins], dtype=float)
        out = out[np.argsort(out[:, 0]), :]
    else:
        # DP: choose one dx per bin to minimize jump cost and penalize decreases strongly.
        smooth_w = 1.0
        down_w = 50.0
        t_bins = [t for t, _ in bins]
        cand = [np.asarray(dx, dtype=float) for _, dx in bins]
        # dp[i, j] cost ending at candidate j of bin i
        dp: list[np.ndarray] = []
        prev: list[np.ndarray] = []
        dp0 = np.zeros((cand[0].size,), dtype=float)
        dp.append(dp0)
        prev.append(np.full((cand[0].size,), -1, dtype=int))
        for i in range(1, len(cand)):
            c_prev = cand[i - 1]
            c_cur = cand[i]
            if c_cur.size == 0 or c_prev.size == 0:
                raise RuntimeError("Empty candidate set encountered.")
            cost = np.empty((c_prev.size, c_cur.size), dtype=float)
            for a, dx_prev in enumerate(c_prev.tolist()):
                d = c_cur - float(dx_prev)
                down = np.maximum(0.0, -d)
                cost[a, :] = smooth_w * (d * d) + down_w * (down * down)
            tot = dp[i - 1][:, None] + cost
            j_prev = np.argmin(tot, axis=0)
            dp_i = tot[j_prev, np.arange(c_cur.size)]
            dp.append(np.asarray(dp_i, dtype=float))
            prev.append(np.asarray(j_prev, dtype=int))
        j = int(np.argmin(dp[-1]))
        dx_sel = []
        for i in range(len(cand) - 1, -1, -1):
            dx_sel.append(float(cand[i][j]))
            j = int(prev[i][j]) if i > 0 else -1
        dx_sel = list(reversed(dx_sel))
        out = np.column_stack([np.asarray(t_bins, dtype=float), np.asarray(dx_sel, dtype=float)])

    if bool(enforce_monotone) and out.shape[0] >= 2:
        out[:, 1] = np.maximum.accumulate(out[:, 1])
    return np.asarray(out, dtype=float)


def _resample_to_grid(points: np.ndarray, *, t_grid: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    t = np.asarray(pts[:, 0], dtype=float)
    y = np.asarray(pts[:, 1], dtype=float)
    tg = np.asarray(t_grid, dtype=float)
    if pts.shape[0] < 2:
        return np.full_like(tg, fill_value=float("nan"), dtype=float)
    return np.interp(tg, t, y, left=float(y[0]), right=float(y[-1]))


def _digitize_panel(
    *,
    img_path: Path,
    t_max_s: float,
    dx_max_um: float,
    circle_box_margin_px: float,
    time_bin_s: float,
    pick_mode: str,
    enforce_monotone: bool,
    hough_line_min_len_px: int,
    hough_line_max_gap_px: int,
    hough_line_min_len_factor: float,
    hough_circles_dp: float,
    hough_circles_min_dist_px: float,
    hough_circles_param1: float,
    hough_circles_param2: float,
    hough_circles_min_radius_px: int,
    hough_circles_max_radius_px: int,
    debug_dir: Path | None,
) -> tuple[PlotBox, np.ndarray, np.ndarray]:
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    box = _detect_plot_box(
        gray,
        min_line_len_px=int(hough_line_min_len_px),
        max_line_gap_px=int(hough_line_max_gap_px),
        min_len_factor=float(hough_line_min_len_factor),
    )
    box_in = box.shrink(float(circle_box_margin_px))

    circles = _detect_circles(
        gray,
        dp=float(hough_circles_dp),
        min_dist=float(hough_circles_min_dist_px),
        param1=float(hough_circles_param1),
        param2=float(hough_circles_param2),
        min_radius=int(hough_circles_min_radius_px),
        max_radius=int(hough_circles_max_radius_px),
    )
    if circles.shape[0] == 0:
        raise RuntimeError(f"No circles detected in {img_path}. Adjust Hough circle parameters.")

    inside = circles[
        (circles[:, 0] > float(box_in.x_left))
        & (circles[:, 0] < float(box_in.x_right))
        & (circles[:, 1] > float(box_in.y_top))
        & (circles[:, 1] < float(box_in.y_bottom))
    ]
    if inside.shape[0] == 0:
        raise RuntimeError(f"No circles found inside detected plot area for {img_path}.")

    pts = np.array([_px_to_data(x_px=x, y_px=y, box=box, t_max_s=float(t_max_s), dx_max_um=float(dx_max_um)) for x, y, _r in inside], dtype=float)
    pts_curve = _pick_curve(pts, bin_s=float(time_bin_s), mode=str(pick_mode), enforce_monotone=bool(enforce_monotone))
    if pts_curve.shape[0] == 0:
        raise RuntimeError(f"Digitization produced no curve points for {img_path}.")
    # Ensure endpoints at t=0 and t=t_max exist (plots show displacement from 0).
    if float(pts_curve[0, 0]) > 1.0e-9:
        pts_curve = np.vstack([np.array([[0.0, 0.0]], dtype=float), pts_curve])
    else:
        pts_curve[0, 0] = 0.0
        pts_curve[0, 1] = max(0.0, float(pts_curve[0, 1]))
    if float(t_max_s) - float(pts_curve[-1, 0]) > 1.0e-9:
        pts_curve = np.vstack([pts_curve, np.array([[float(t_max_s), float(pts_curve[-1, 1])]], dtype=float)])
    pts_curve[:, 0] = np.clip(pts_curve[:, 0], 0.0, float(t_max_s))
    pts_curve[:, 1] = np.clip(pts_curve[:, 1], 0.0, float(dx_max_um))

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        dbg = img_bgr.copy()
        # Draw detected plot box.
        cv2.rectangle(
            dbg,
            (int(round(box.x_left)), int(round(box.y_top))),
            (int(round(box.x_right)), int(round(box.y_bottom))),
            (0, 255, 255),
            1,
        )
        # Draw all inside circles (magenta).
        for x, y, r in inside:
            cv2.circle(dbg, (int(round(x)), int(round(y))), int(round(r)), (255, 0, 255), 1)
        # Draw selected points (green) at their original pixel coords by re-mapping.
        for t, dx_um in pts_curve:
            # invert mapping back to pixels for debugging
            x = box.x_left + (t / float(t_max_s)) * (box.x_right - box.x_left)
            y = box.y_bottom - (dx_um / float(dx_max_um)) * (box.y_bottom - box.y_top)
            cv2.circle(dbg, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
        out_dbg = debug_dir / f"{img_path.stem}_digitize_debug.png"
        cv2.imwrite(str(out_dbg), dbg)

    return box, pts, pts_curve


def main() -> None:
    ap = argparse.ArgumentParser(description="Digitize Li et al. (2020) Fig. 7b/c/d experimental dx(t) circle markers.")
    ap.add_argument("--fig7-b", type=str, default="examples/biofilms/benchmarks/lie/li_fig7_b.png")
    ap.add_argument("--fig7-c", type=str, default="examples/biofilms/benchmarks/lie/li_fig7_c.png")
    ap.add_argument("--fig7-d", type=str, default="examples/biofilms/benchmarks/lie/li_fig7_d.png")
    ap.add_argument("--t-max-s", type=float, default=20.0, help="x-axis range in seconds.")
    ap.add_argument("--dx-max-um", type=float, default=150.0, help="y-axis range in micrometers.")
    ap.add_argument("--time-bin-s", type=float, default=0.5, help="Time bin size for picking one circle per time bin.")
    ap.add_argument(
        "--pick-mode",
        type=str,
        default="smooth",
        choices=("smooth", "topmost"),
        help="How to pick one marker per time bin. 'smooth' is more robust than 'topmost'.",
    )
    ap.add_argument(
        "--no-enforce-monotone",
        action="store_true",
        help="Disable enforcing a nondecreasing dx(t) after digitization (default enforces).",
    )
    ap.add_argument("--circle-box-margin-px", type=float, default=2.0, help="Shrink detected plot box by this margin before circle filtering.")
    ap.add_argument("--out-csv", type=str, default="examples/biofilms/benchmarks/lie/exp_fig7_dx_digitized.csv")
    ap.add_argument("--debug-dir", type=str, default="out/_lie_fig7_digitize_debug", help="Write debug images here (empty disables).")

    # Hough line (plot box) detection parameters.
    ap.add_argument("--hough-line-min-len-px", type=int, default=80)
    ap.add_argument("--hough-line-max-gap-px", type=int, default=5)
    ap.add_argument("--hough-line-min-len-factor", type=float, default=0.5)

    # Hough circle detection parameters.
    ap.add_argument("--hough-circles-dp", type=float, default=1.2)
    ap.add_argument("--hough-circles-min-dist-px", type=float, default=8.0)
    ap.add_argument("--hough-circles-param1", type=float, default=100.0)
    ap.add_argument("--hough-circles-param2", type=float, default=15.0)
    ap.add_argument("--hough-circles-min-radius-px", type=int, default=2)
    ap.add_argument("--hough-circles-max-radius-px", type=int, default=8)
    args = ap.parse_args()

    fig7_b = Path(str(args.fig7_b))
    fig7_c = Path(str(args.fig7_c))
    fig7_d = Path(str(args.fig7_d))
    for p in (fig7_b, fig7_c, fig7_d):
        if not p.exists():
            raise FileNotFoundError(p)

    debug_dir = Path(str(args.debug_dir)) if str(args.debug_dir).strip() else None

    _b = _digitize_panel(
        img_path=fig7_b,
        t_max_s=float(args.t_max_s),
        dx_max_um=float(args.dx_max_um),
        circle_box_margin_px=float(args.circle_box_margin_px),
        time_bin_s=float(args.time_bin_s),
        pick_mode=str(args.pick_mode),
        enforce_monotone=(not bool(args.no_enforce_monotone)),
        hough_line_min_len_px=int(args.hough_line_min_len_px),
        hough_line_max_gap_px=int(args.hough_line_max_gap_px),
        hough_line_min_len_factor=float(args.hough_line_min_len_factor),
        hough_circles_dp=float(args.hough_circles_dp),
        hough_circles_min_dist_px=float(args.hough_circles_min_dist_px),
        hough_circles_param1=float(args.hough_circles_param1),
        hough_circles_param2=float(args.hough_circles_param2),
        hough_circles_min_radius_px=int(args.hough_circles_min_radius_px),
        hough_circles_max_radius_px=int(args.hough_circles_max_radius_px),
        debug_dir=debug_dir,
    )
    _c = _digitize_panel(
        img_path=fig7_c,
        t_max_s=float(args.t_max_s),
        dx_max_um=float(args.dx_max_um),
        circle_box_margin_px=float(args.circle_box_margin_px),
        time_bin_s=float(args.time_bin_s),
        pick_mode=str(args.pick_mode),
        enforce_monotone=(not bool(args.no_enforce_monotone)),
        hough_line_min_len_px=int(args.hough_line_min_len_px),
        hough_line_max_gap_px=int(args.hough_line_max_gap_px),
        hough_line_min_len_factor=float(args.hough_line_min_len_factor),
        hough_circles_dp=float(args.hough_circles_dp),
        hough_circles_min_dist_px=float(args.hough_circles_min_dist_px),
        hough_circles_param1=float(args.hough_circles_param1),
        hough_circles_param2=float(args.hough_circles_param2),
        hough_circles_min_radius_px=int(args.hough_circles_min_radius_px),
        hough_circles_max_radius_px=int(args.hough_circles_max_radius_px),
        debug_dir=debug_dir,
    )
    _d = _digitize_panel(
        img_path=fig7_d,
        t_max_s=float(args.t_max_s),
        dx_max_um=float(args.dx_max_um),
        circle_box_margin_px=float(args.circle_box_margin_px),
        time_bin_s=float(args.time_bin_s),
        pick_mode=str(args.pick_mode),
        enforce_monotone=(not bool(args.no_enforce_monotone)),
        hough_line_min_len_px=int(args.hough_line_min_len_px),
        hough_line_max_gap_px=int(args.hough_line_max_gap_px),
        hough_line_min_len_factor=float(args.hough_line_min_len_factor),
        hough_circles_dp=float(args.hough_circles_dp),
        hough_circles_min_dist_px=float(args.hough_circles_min_dist_px),
        hough_circles_param1=float(args.hough_circles_param1),
        hough_circles_param2=float(args.hough_circles_param2),
        hough_circles_min_radius_px=int(args.hough_circles_min_radius_px),
        hough_circles_max_radius_px=int(args.hough_circles_max_radius_px),
        debug_dir=debug_dir,
    )

    pts_b = _b[2]
    pts_c = _c[2]
    pts_d = _d[2]

    # Common time grid: union of b/c/d, snapped to the time-bin grid.
    bin_s = float(max(1.0e-9, float(args.time_bin_s)))
    t_all = np.concatenate([pts_b[:, 0], pts_c[:, 0], pts_d[:, 0]])
    t_snap = np.round(t_all / bin_s) * bin_s
    t_grid = np.unique(np.sort(t_snap))
    t_grid = t_grid[(t_grid >= -1.0e-9) & (t_grid <= float(args.t_max_s) + 1.0e-9)]
    if t_grid.size < 2:
        raise RuntimeError("Too few digitized time points; adjust --time-bin-s or Hough parameters.")

    dx1_um = _resample_to_grid(pts_b, t_grid=t_grid)
    dx2_um = _resample_to_grid(pts_c, t_grid=t_grid)
    dx3_um = _resample_to_grid(pts_d, t_grid=t_grid)

    out_csv = Path(str(args.out_csv))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("t_s,dx_line1_m,dx_line2_m,dx_line3_m\n")
        for t, d1, d2, d3 in zip(t_grid.tolist(), dx1_um.tolist(), dx2_um.tolist(), dx3_um.tolist()):
            f.write(f"{float(t):.12e},{float(d1)*1e-6:.12e},{float(d2)*1e-6:.12e},{float(d3)*1e-6:.12e}\n")

    print(f"[ok] wrote {out_csv}")
    if debug_dir is not None:
        print(f"[ok] wrote debug images in {debug_dir}")
    print(f"[info] time_bin_s={bin_s:g}, n_grid={int(t_grid.size)}")
    print(f"[info] line1 points={int(pts_b.shape[0])}, line2 points={int(pts_c.shape[0])}, line3 points={int(pts_d.shape[0])}")


if __name__ == "__main__":
    main()
