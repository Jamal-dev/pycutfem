#!/usr/bin/env python3
"""
Create an experimental-vs-simulation overlay video for the Lie benchmark using SVG-traced contours.

This script draws:
  - the experimental biofilm outline traced in Inkscape SVGs (green)
  - the simulated biofilm outline (red), either
      * from the α=0.5 contour in the simulation VTU (`--sim-outline alpha`, default), or
      * by deforming the t=0 polygon with the displacement field `u` (`--sim-outline u_deform_poly`).

Why SVG?
--------
The OCT video frames can be difficult to segment robustly. The SVG traces provide:
  - a stable per-frame anchor/base line,
  - a publishable outline without segmentation jitter.

Assumptions
-----------
* The SVG base line is the rigid-support top (width = 1 mm).
* The polygon CSV (mm) is anchored at base-left with y=0 on the base (as produced by
  `extract_deformation_timeseries_from_svg_series.py --scale-mode base_frame0`).
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import cv2
import meshio
import numpy as np
from scipy.spatial import cKDTree

from examples.biofilms.benchmarks.lie.svg_trace_utils import extract_svg_frame_geometry


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
    return int(x0), int(y0), int(w), int(h)


def _read_frame(cap: cv2.VideoCapture, frame: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    if not ok or img is None:
        raise RuntimeError(f"Could not read frame={frame}")
    return img


def _read_polygon_mm_csv(path: Path) -> np.ndarray:
    arr = np.genfromtxt(str(path), delimiter=",", skip_header=1, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    arr = np.asarray(arr, dtype=float)
    if arr.shape[1] < 2:
        raise ValueError(f"Polygon CSV must have at least 2 columns; got shape={arr.shape}")
    pts = arr[:, :2]
    if pts.shape[0] < 3:
        raise ValueError(f"Polygon must have at least 3 points; got {pts.shape[0]}")
    if not np.allclose(pts[0], pts[-1], rtol=0.0, atol=1.0e-12):
        pts = np.vstack([pts, pts[0]])
    return np.asarray(pts, dtype=float)


def _vtk_step_files(vtk_dir: Path) -> dict[int, Path]:
    step_re = re.compile(r"step=(\d+)\.vtu$")
    out: dict[int, Path] = {}
    for p in sorted(vtk_dir.glob("step=*.vtu")):
        m = step_re.search(p.name)
        if not m:
            continue
        out[int(m.group(1))] = p
    if not out:
        raise FileNotFoundError(f"No VTK files found in {vtk_dir} (expected step=XXXX.vtu).")
    return out


def _interp_displacement_idw(
    *,
    tree: cKDTree,
    u_nodes: np.ndarray,
    xq: np.ndarray,
    k: int = 8,
    power: float = 2.0,
) -> np.ndarray:
    xq = np.asarray(xq, dtype=float)
    if xq.ndim != 2 or xq.shape[1] != 2:
        raise ValueError("xq must have shape (N,2)")
    k = int(max(1, k))

    dist, idx = tree.query(xq, k=k)
    if k == 1:
        dist = dist.reshape((-1, 1))
        idx = idx.reshape((-1, 1))

    hit = dist[:, 0] <= 1.0e-14
    u_out = np.empty((xq.shape[0], 2), dtype=float)
    if np.any(hit):
        u_out[hit, :] = u_nodes[idx[hit, 0], :]

    miss = ~hit
    if np.any(miss):
        d = np.maximum(dist[miss, :], 1.0e-14)
        w = 1.0 / (d**float(power))
        w_sum = np.sum(w, axis=1, keepdims=True)
        w = w / np.maximum(w_sum, 1.0e-30)
        u_out[miss, :] = np.sum(u_nodes[idx[miss, :], :] * w[:, :, None], axis=1)
    return u_out


def _x_intersections_on_y(poly_xy: np.ndarray, *, y: float) -> np.ndarray:
    """
    Intersect a closed polygon with the horizontal line y=const.
    Returns all x intersections (unsorted -> sorted).
    """
    p = np.asarray(poly_xy, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 4:
        return np.zeros((0,), dtype=float)
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
        # Half-open rule to avoid double-counting vertices.
        if not (y_lo < y0 <= y_hi):
            continue
        t = (y0 - y1) / (y2 - y1)
        if 0.0 <= t <= 1.0:
            xs.append(float(x1 + t * (x2 - x1)))
    if not xs:
        return np.zeros((0,), dtype=float)
    return np.sort(np.asarray(xs, dtype=float))


def _x_quantile_candidates_on_y(poly_xy: np.ndarray, *, y: float, mode: str, q: float) -> list[float]:
    """
    Return candidate x locations (one per inside segment) for continuity-based tracking.
    For q<=0, returns a single boundary candidate (leftmost/rightmost).
    """
    q = float(np.clip(float(q), 0.0, 1.0))
    mode = str(mode).strip().lower()
    if mode not in {"leftmost", "rightmost"}:
        raise ValueError(f"Unknown mode {mode!r}")

    xs = _x_intersections_on_y(poly_xy, y=float(y))
    if xs.size < 1:
        return []
    if q <= 0.0:
        return [float(xs[0] if mode == "leftmost" else xs[-1])]
    if int(xs.size) % 2 == 1:
        xs = xs[:-1]
    if xs.size < 2:
        return []
    segs = xs.reshape((-1, 2))
    out: list[float] = []
    for x0, x1 in segs.tolist():
        xL = float(min(x0, x1))
        xR = float(max(x0, x1))
        if mode == "leftmost":
            out.append(float(xL + q * (xR - xL)))
        else:
            out.append(float(xR - q * (xR - xL)))
    return out


def _pick_x_on_y(
    poly_xy: np.ndarray,
    *,
    y: float,
    mode: str,
    q: float,
    track_mode: str,
    x_prev: float | None,
    max_jump: float,
) -> float:
    cand = _x_quantile_candidates_on_y(poly_xy, y=float(y), mode=str(mode), q=float(q))
    if not cand:
        return float("nan")
    track_mode = str(track_mode).strip().lower()
    if track_mode not in {"independent", "continuity"}:
        raise ValueError(f"Unknown track_mode {track_mode!r}")
    if track_mode == "independent" or x_prev is None or (not np.isfinite(float(x_prev))):
        # For independent mode, choose the candidate closest to the chosen side.
        return float(min(cand) if str(mode).strip().lower() == "leftmost" else max(cand))
    prev = float(x_prev)
    j = int(np.argmin([abs(float(x) - prev) for x in cand]))
    x = float(cand[j])
    if np.isfinite(float(max_jump)) and float(max_jump) > 0.0 and abs(x - prev) > float(max_jump):
        return float("nan")
    return float(x)


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay simulation on SVG-traced experimental contours (Lie benchmark).")
    ap.add_argument(
        "--exp-video",
        type=str,
        default="examples/biofilms/benchmarks/lie/additional_data/bit27491-sup-0001-si_v1.avi",
        help="Experimental video used as background (frames must match the SVG traces).",
    )
    ap.add_argument("--svg-dir", type=str, default="examples/biofilms/benchmarks/lie/svg_fles", help="Directory with frame_XXXX.svg.")
    ap.add_argument("--sim-dir", type=str, required=True, help="Simulation output directory containing vtk/step=XXXX.vtu and timeseries.csv.")
    ap.add_argument(
        "--sim-poly0-mm-csv",
        type=str,
        default="examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_svgtrace_q010_base_cont.csv",
        help="t=0 polygon in mm coords (base-left anchored, y=0 at base).",
    )
    ap.add_argument("--out-video", type=str, default="", help="Output video. Default: <sim-dir>/exp_sim_overlay_svgtrace.mp4")
    ap.add_argument("--frame-step", type=int, default=1, help="Use every N-th experimental frame.")
    ap.add_argument("--max-frames", type=int, default=0, help="If >0, only write first N processed frames.")
    ap.add_argument("--out-fps", type=float, default=float("nan"), help="Output FPS. Default: exp_fps/frame_step.")
    ap.add_argument("--crop", type=str, default="", help="Optional crop: 'x0,y0,w,h' in pixels (applied to background and overlays).")
    ap.add_argument(
        "--no-time-interp",
        action="store_true",
        help="Disable linear-in-time interpolation of displacement between VTU steps (use nearest step only).",
    )

    # Tracking overlay annotations (publishable).
    ap.add_argument("--show-tracking", action="store_true", help="Draw tracking lines + points used for dx(t) comparison.")
    ap.add_argument(
        "--lines-csv",
        type=str,
        default="",
        help="Optional sim lines CSV (from driver). Default: <sim-dir>/lines.csv if it exists.",
    )
    # Default to "leftmost" to match the simulation driver's default (`lie_synthetic_deformation_one_domain.py`)
    # and avoid hiding points when `--track-mode continuity` with `--max-jump-mm` gating is enabled.
    ap.add_argument("--dx-intersection", type=str, default="leftmost", choices=("leftmost", "rightmost"))
    ap.add_argument("--dx-quantile", type=float, default=0.0, help="Interior quantile (0=boundary).")
    ap.add_argument("--track-mode", type=str, default="continuity", choices=("independent", "continuity"))
    ap.add_argument("--max-jump-mm", type=float, default=0.25, help="Max per-frame x jump for continuity tracking (mm).")
    ap.add_argument("--y-fracs", type=str, default="", help="If --lines-csv not used, y-fracs for tracking lines, e.g. '0.75,0.5,0.25'.")
    ap.add_argument("--draw-base", action="store_true", help="Draw SVG base endpoints (anchor diagnostic).")

    # SVG sampling / scale.
    ap.add_argument("--block-w-mm", type=float, default=1.0, help="Support width used for pixel->mm scaling (mm).")
    ap.add_argument("--cubic-samples", type=int, default=20, help="Samples per cubic Bezier segment (SVG path sampling).")
    ap.add_argument("--join-tol-px", type=float, default=5.0, help="Endpoint join tolerance (pixels) for chaining SVG path segments.")

    # Simulation geometry mapping (match driver defaults).
    ap.add_argument("--block-w", type=float, default=1.0e-3, help="Support width in simulation geometry [m].")
    ap.add_argument("--block-h", type=float, default=3.0e-3, help="Support height in simulation geometry [m].")
    ap.add_argument("--block-xc", type=float, default=float("nan"), help="Support center x [m]. Default: infer as 0.5*L from VTU.")

    ap.add_argument(
        "--sim-outline",
        type=str,
        default="alpha",
        choices=("alpha", "u_deform_poly"),
        help=(
            "How to render the simulated outline. "
            "'alpha' uses the α=0.5 contour from the VTU (recommended for --transport-mode pde). "
            "'u_deform_poly' deforms the t=0 polygon with u (reference-map style)."
        ),
    )

    # VTU sampling options.
    ap.add_argument("--k-nn", type=int, default=8, help="k in k-NN inverse-distance interpolation of displacement.")
    ap.add_argument("--idw-power", type=float, default=2.0, help="Power p in 1/d^p weights for k-NN interpolation.")
    args = ap.parse_args()

    sim_outline = str(args.sim_outline).strip().lower()
    if sim_outline not in {"alpha", "u_deform_poly"}:
        raise ValueError(f"Unknown --sim-outline {args.sim_outline!r}.")

    exp_video = Path(str(args.exp_video))
    svg_dir = Path(str(args.svg_dir))
    sim_dir = Path(str(args.sim_dir))
    if not exp_video.exists():
        raise FileNotFoundError(f"Experimental video not found: {exp_video}")
    if not svg_dir.exists():
        raise FileNotFoundError(f"SVG directory not found: {svg_dir}")
    if not sim_dir.exists():
        raise FileNotFoundError(f"Simulation directory not found: {sim_dir}")

    vtk_dir = sim_dir / "vtk"
    step_files = _vtk_step_files(vtk_dir)

    ts_path = sim_dir / "timeseries.csv"
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing {ts_path}. Run the simulation with --vtk-every >0.")
    ts_arr = np.genfromtxt(str(ts_path), delimiter=",", skip_header=1, dtype=float)
    if ts_arr.ndim == 1:
        ts_arr = ts_arr.reshape(1, -1)
    if ts_arr.shape[1] < 1:
        raise ValueError(f"Unexpected timeseries shape {ts_arr.shape} in {ts_path}")
    t_sim = np.asarray(ts_arr[:, 0], dtype=float)
    if t_sim.size < 1:
        raise ValueError(f"No time points in {ts_path}")
    if not np.all(np.diff(t_sim) >= 0.0):
        raise ValueError(f"Non-monotone time vector in {ts_path}")

    # Frame index -> SVG path (some traces may be missing; use nearest/previous for transforms).
    svg_re = re.compile(r"frame_(\d+)\.svg$")
    svg_by_fr: dict[int, Path] = {}
    for p in sorted(svg_dir.glob("frame_*.svg")):
        m = svg_re.search(p.name)
        if not m:
            continue
        svg_by_fr[int(m.group(1))] = p
    if not svg_by_fr:
        raise RuntimeError(f"No SVG frames found in {svg_dir}")

    # Open video (background + FPS).
    cap = cv2.VideoCapture(str(exp_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {exp_video}")
    fps_exp = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps_exp) or fps_exp <= 0.0:
        fps_exp = 8.0

    crop = _parse_crop(str(args.crop))
    out_fps = float(args.out_fps) if np.isfinite(float(args.out_fps)) else float(fps_exp) / max(1.0, float(args.frame_step))

    if str(args.out_video).strip():
        out_video = Path(str(args.out_video))
    else:
        out_video = sim_dir / "exp_sim_overlay_svgtrace.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)

    # Determine output frame size.
    img0 = _read_frame(cap, 0)
    if crop is not None:
        x0c, y0c, wc, hc = crop
        frame_w, frame_h = int(wc), int(hc)
    else:
        frame_h, frame_w = img0.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if out_video.suffix.lower() == ".mp4" else "XVID"))
    writer = cv2.VideoWriter(str(out_video), fourcc, float(out_fps), (int(frame_w), int(frame_h)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_video}")

    # SVG scale: enforce constant m_per_px from frame 0 base width.
    fr0 = 0 if 0 in svg_by_fr else min(svg_by_fr.keys())
    g0 = extract_svg_frame_geometry(
        svg_by_fr[fr0],
        block_w_mm=float(args.block_w_mm),
        m_per_px=None,
        cubic_samples=int(args.cubic_samples),
        join_tol_px=float(args.join_tol_px),
    )
    dx0 = float(g0.base_right_px[0] - g0.base_left_px[0])
    dy0 = float(g0.base_right_px[1] - g0.base_left_px[1])
    base_len0_px = float(math.hypot(dx0, dy0))
    if not np.isfinite(base_len0_px) or base_len0_px <= 1.0e-9:
        raise RuntimeError(f"Degenerate base length in {svg_by_fr[fr0]}")
    m_per_px = (float(args.block_w_mm) / float(base_len0_px)) * 1.0e-3

    # Simulation: read poly0 and align it the same way as the driver (--alpha0-align block).
    poly0_mm = _read_polygon_mm_csv(Path(str(args.sim_poly0_mm_csv)))
    poly0_m = poly0_mm * 1.0e-3
    hb_mm = float(np.max(poly0_mm[:, 1]))

    # Infer L from an available VTU (for default block_xc).
    vtu0 = meshio.read(str(step_files[min(step_files.keys())]))
    pts0 = np.asarray(vtu0.points[:, :2], dtype=float)
    x_min, x_max = float(np.min(pts0[:, 0])), float(np.max(pts0[:, 0]))
    L = float(x_max - x_min)
    block_w = float(args.block_w)
    block_h = float(args.block_h)
    block_xc = float(args.block_xc) if np.isfinite(float(args.block_xc)) else (x_min + 0.5 * L)
    block_x0 = float(block_xc - 0.5 * block_w)

    _alpha_half_contour_global = None
    if sim_outline == "alpha":
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri

        # Triangulate QUAD cells into triangles for tricontour.
        tris: list[np.ndarray] = []
        for cb in vtu0.cells:
            ctype = str(getattr(cb, "type", "")).lower()
            conn = np.asarray(getattr(cb, "data", cb), dtype=int)
            if conn.size == 0:
                continue
            if ctype == "quad":
                q = conn.reshape((-1, 4))
                tris.append(q[:, [0, 1, 2]])
                tris.append(q[:, [0, 2, 3]])
            elif ctype == "triangle":
                tris.append(conn.reshape((-1, 3)))
        if not tris:
            raise RuntimeError("No triangulatable cells found in VTU (expected quads/triangles).")
        tri_conn = np.vstack(tris).astype(int, copy=False)
        tri = mtri.Triangulation(pts0[:, 0], pts0[:, 1], triangles=tri_conn)

        fig = plt.figure(figsize=(2, 2), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        def _alpha_half_contour_global(alpha_nodes: np.ndarray, *, level: float = 0.5) -> np.ndarray | None:
            a = np.asarray(alpha_nodes, dtype=float).ravel()
            if a.size != pts0.shape[0]:
                raise ValueError(f"alpha_nodes size mismatch: got {a.size}, expected {pts0.shape[0]}")
            ax.clear()
            ax.set_axis_off()
            cs = ax.tricontour(tri, a, levels=[float(level)])
            polys: list[np.ndarray] = []
            try:
                segs = list(getattr(cs, "allsegs", [[[]]])[0])
            except Exception:
                segs = []
            for seg in segs:
                v = np.asarray(seg, dtype=float)
                if v.ndim == 2 and v.shape[0] >= 3 and v.shape[1] >= 2:
                    polys.append(v[:, :2])
            # Remove artists to avoid accumulation.
            try:
                cs.remove()
            except Exception:
                pass
            if not polys:
                return None
            # Prefer the longest polyline (robust even if the contour touches boundaries).
            lens = [float(np.sum(np.linalg.norm(poly[1:] - poly[:-1], axis=1))) for poly in polys]
            poly = polys[int(np.argmax(lens))]
            if not np.allclose(poly[0], poly[-1], rtol=0.0, atol=1.0e-12):
                poly = np.vstack([poly, poly[0]])
            return np.asarray(poly, dtype=float)

    poly_ymin = float(np.min(poly0_m[:, 1]))
    tol_y = max(1.0e-12, 1.0e-6 * max(1.0, float(np.max(np.abs(poly0_m[:, 1])))))
    base_pts = poly0_m[np.abs(poly0_m[:, 1] - poly_ymin) <= tol_y]
    base_xmin = float(np.min(base_pts[:, 0])) if base_pts.shape[0] >= 2 else float(np.min(poly0_m[:, 0]))
    dx_align = float(block_x0 - base_xmin)
    poly0_m_global = poly0_m + np.array([dx_align, float(block_h - poly_ymin)], dtype=float)

    # Tracking lines for annotation.
    y_lines_mm: list[float] = []
    x_ref_init_mm: list[float] = []
    if bool(args.show_tracking):
        lines_csv = Path(str(args.lines_csv)) if str(args.lines_csv).strip() else (sim_dir / "lines.csv")
        if lines_csv.exists():
            arr = np.genfromtxt(str(lines_csv), delimiter=",", skip_header=1, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] >= 3:
                # y_line_m is column 2.
                y_lines_mm = [float((y_m - float(block_h)) * 1.0e3) for y_m in np.asarray(arr[:, 2], dtype=float).ravel().tolist()]
            if arr.shape[1] >= 4:
                # x_ref_m is column 3 (global). Convert to local mm anchored at base-left.
                x_ref_init_mm = [float((x_m - float(block_x0)) * 1.0e3) for x_m in np.asarray(arr[:, 3], dtype=float).ravel().tolist()]
        if not y_lines_mm:
            if str(args.y_fracs).strip():
                fracs = [float(s.strip()) for s in str(args.y_fracs).split(",") if s.strip()]
            else:
                fracs = [0.75, 0.50, 0.25]
            y_lines_mm = [float(f) * float(hb_mm) for f in fracs]

    # Frame list to process.
    frames = list(range(0, n_frames, max(1, int(args.frame_step))))
    if int(args.max_frames) > 0:
        frames = frames[: int(args.max_frames)]
    if not frames or frames[0] != 0:
        frames = [0] + frames

    # Cache VTU -> (KDTree, u_nodes, alpha_nodes).
    vtu_cache: dict[int, tuple[cKDTree, np.ndarray, np.ndarray]] = {}
    step_ids = np.array(sorted(step_files.keys()), dtype=int)

    def _nearest_step_id(step_no: int) -> int:
        if int(step_no) in step_files:
            return int(step_no)
        j = int(np.argmin(np.abs(step_ids - int(step_no))))
        return int(step_ids[j])

    prev_geom = g0
    # Continuity tracking state (mm) for exp and sim points.
    if len(x_ref_init_mm) == 3 and all(np.isfinite(float(x)) for x in x_ref_init_mm):
        x_prev_exp: list[float | None] = [float(x) for x in x_ref_init_mm]
        x_prev_sim: list[float | None] = [float(x) for x in x_ref_init_mm]
    else:
        x_prev_exp = [None, None, None]
        x_prev_sim = [None, None, None]
    for fr in frames:
        img = img0 if fr == 0 else _read_frame(cap, int(fr))
        if crop is not None:
            x0c, y0c, wc, hc = crop
            img = img[y0c : y0c + hc, x0c : x0c + wc].copy()

        # SVG geometry for this frame (fallback to previous on missing/parse error).
        try:
            if int(fr) in svg_by_fr:
                prev_geom = extract_svg_frame_geometry(
                    svg_by_fr[int(fr)],
                    m_per_px=float(m_per_px),
                    cubic_samples=int(args.cubic_samples),
                    join_tol_px=float(args.join_tol_px),
                )
            geom = prev_geom
        except Exception:
            geom = prev_geom

        # Map time -> nearest simulation step index.
        t_s = float(fr) / float(fps_exp)
        if bool(args.no_time_interp) or t_sim.size == 1:
            idx0 = int(np.argmin(np.abs(t_sim - float(t_s))))
            idx1 = idx0
            w = 0.0
        else:
            if float(t_s) <= float(t_sim[0]):
                idx0, idx1, w = 0, 0, 0.0
            elif float(t_s) >= float(t_sim[-1]):
                idx0, idx1, w = int(t_sim.size - 1), int(t_sim.size - 1), 0.0
            else:
                idx1 = int(np.searchsorted(t_sim, float(t_s), side="right"))
                idx0 = max(0, idx1 - 1)
                t0 = float(t_sim[idx0])
                t1 = float(t_sim[idx1])
                w = (float(t_s) - t0) / max(1.0e-30, (t1 - t0))
                w = float(np.clip(w, 0.0, 1.0))

        step0 = _nearest_step_id(int(idx0))
        step1 = _nearest_step_id(int(idx1))
        if step0 == step1:
            w = 0.0

        def _load_step(step_no: int) -> tuple[cKDTree, np.ndarray, np.ndarray]:
            if step_no not in vtu_cache:
                vtu = meshio.read(str(step_files[int(step_no)]))
                pts = np.asarray(vtu.points[:, :2], dtype=float)
                if "u" not in vtu.point_data:
                    raise KeyError(f"VTU {step_files[int(step_no)]} does not contain point_data['u']")
                u = np.asarray(vtu.point_data["u"][:, :2], dtype=float)
                if "alpha" not in vtu.point_data:
                    raise KeyError(f"VTU {step_files[int(step_no)]} does not contain point_data['alpha']")
                alpha = np.asarray(vtu.point_data["alpha"], dtype=float).ravel()
                vtu_cache[int(step_no)] = (cKDTree(pts), u, alpha)
            return vtu_cache[int(step_no)]

        tree0, u0, a0 = _load_step(step0)
        tree1, u1, a1 = _load_step(step1)

        if sim_outline == "u_deform_poly":
            # Mesh is expected to be constant over time; use tree0 and linearly interpolate u.
            u_nodes = (1.0 - float(w)) * np.asarray(u0, dtype=float) + float(w) * np.asarray(u1, dtype=float)
            u_q = _interp_displacement_idw(
                tree=tree0,
                u_nodes=u_nodes,
                xq=poly0_m_global[:, :2],
                k=int(args.k_nn),
                power=float(args.idw_power),
            )
            poly_sim_global = poly0_m_global[:, :2] + u_q
        else:
            assert _alpha_half_contour_global is not None
            a_nodes = (1.0 - float(w)) * np.asarray(a0, dtype=float) + float(w) * np.asarray(a1, dtype=float)
            poly_sim_global = _alpha_half_contour_global(a_nodes, level=0.5)
            if poly_sim_global is None:
                # Fallback: deform polygon if contouring failed.
                u_nodes = (1.0 - float(w)) * np.asarray(u0, dtype=float) + float(w) * np.asarray(u1, dtype=float)
                u_q = _interp_displacement_idw(
                    tree=tree0,
                    u_nodes=u_nodes,
                    xq=poly0_m_global[:, :2],
                    k=int(args.k_nn),
                    power=float(args.idw_power),
                )
                poly_sim_global = poly0_m_global[:, :2] + u_q

        # Convert to local mm anchored at base-left.
        poly_sim_mm = np.column_stack(
            [
                (np.asarray(poly_sim_global)[:, 0] - float(block_x0)) * 1.0e3,
                (np.asarray(poly_sim_global)[:, 1] - float(block_h)) * 1.0e3,
            ]
        )
        sim_px = geom.mm_to_px(poly_sim_mm)
        exp_mm = geom.px_to_mm(np.asarray(geom.boundary_px, dtype=float))
        exp_poly_mm = np.vstack([exp_mm, exp_mm[0]])

        # Crop-adjust contours (SVG coords are full-frame).
        exp_px = np.asarray(geom.boundary_px, dtype=float)
        exp_closed = np.vstack([exp_px, exp_px[0]])
        if crop is not None:
            exp_closed = exp_closed - np.array([float(x0c), float(y0c)], dtype=float)
            sim_px = sim_px - np.array([float(x0c), float(y0c)], dtype=float)

        exp_i = np.round(exp_closed).astype(np.int32).reshape((-1, 1, 2))
        sim_i = np.round(sim_px).astype(np.int32).reshape((-1, 1, 2))

        overlay = img.copy()
        # Experimental outline (green) only if the SVG exists for this frame.
        if int(fr) in svg_by_fr:
            cv2.polylines(overlay, [exp_i], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(overlay, [sim_i], isClosed=True, color=(0, 0, 255), thickness=2)

        if bool(args.draw_base):
            bl = np.asarray(geom.base_left_px, dtype=float)
            br = np.asarray(geom.base_right_px, dtype=float)
            if crop is not None:
                bl = bl - np.array([float(x0c), float(y0c)], dtype=float)
                br = br - np.array([float(x0c), float(y0c)], dtype=float)
            cv2.circle(overlay, (int(round(bl[0])), int(round(bl[1]))), 4, (255, 255, 0), -1)
            cv2.circle(overlay, (int(round(br[0])), int(round(br[1]))), 4, (255, 255, 0), -1)

        if bool(args.show_tracking) and len(y_lines_mm) == 3:
            # Draw three tracking lines and points on exp/sim.
            colors = [(50, 200, 255), (255, 200, 50), (200, 255, 50)]  # BGR-ish per line
            dot_r = 5
            dot_outline = (0, 0, 0)
            dx_mode = str(args.dx_intersection)
            dx_q = float(np.clip(float(args.dx_quantile), 0.0, 1.0))
            track_mode = str(args.track_mode)
            max_jump = float(abs(float(args.max_jump_mm)))
            # Determine x extents from experimental outline (for drawing the lines).
            x_min_mm = float(np.min(exp_mm[:, 0]))
            x_max_mm = float(np.max(exp_mm[:, 0]))
            x_pad = 0.10 * max(1.0e-6, (x_max_mm - x_min_mm))
            for i, (y_mm, col) in enumerate(zip(y_lines_mm, colors)):
                # Draw the line (parallel to base, i.e. constant y in the mm frame).
                seg_mm = np.array([[x_min_mm - x_pad, float(y_mm)], [x_max_mm + x_pad, float(y_mm)]], dtype=float)
                seg_px = geom.mm_to_px(seg_mm)
                if crop is not None:
                    seg_px = seg_px - np.array([float(x0c), float(y0c)], dtype=float)
                p0 = (int(round(seg_px[0, 0])), int(round(seg_px[0, 1])))
                p1 = (int(round(seg_px[1, 0])), int(round(seg_px[1, 1])))
                cv2.line(overlay, p0, p1, col, 1, cv2.LINE_AA)

                # Experimental point.
                x_e = _pick_x_on_y(exp_poly_mm, y=float(y_mm), mode=dx_mode, q=dx_q, track_mode=track_mode, x_prev=x_prev_exp[i], max_jump=max_jump)
                if np.isfinite(x_e):
                    x_prev_exp[i] = float(x_e)
                    pe_px = geom.mm_to_px(np.array([[float(x_e), float(y_mm)]], dtype=float))[0]
                    if crop is not None:
                        pe_px = pe_px - np.array([float(x0c), float(y0c)], dtype=float)
                    ctr = (int(round(pe_px[0])), int(round(pe_px[1])))
                    cv2.circle(overlay, ctr, dot_r + 2, dot_outline, -1)
                    cv2.circle(overlay, ctr, dot_r, (0, 255, 0), -1)
                    cv2.putText(overlay, f"E{i+1}", (ctr[0] + 6, ctr[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Simulated point (on sim outline).
                sim_poly_mm = np.asarray(poly_sim_mm, dtype=float)
                if not np.allclose(sim_poly_mm[0], sim_poly_mm[-1], rtol=0.0, atol=1.0e-12):
                    sim_poly_mm = np.vstack([sim_poly_mm, sim_poly_mm[0]])
                x_s = _pick_x_on_y(sim_poly_mm, y=float(y_mm), mode=dx_mode, q=dx_q, track_mode=track_mode, x_prev=x_prev_sim[i], max_jump=max_jump)
                if np.isfinite(x_s):
                    x_prev_sim[i] = float(x_s)
                    ps_px = geom.mm_to_px(np.array([[float(x_s), float(y_mm)]], dtype=float))[0]
                    if crop is not None:
                        ps_px = ps_px - np.array([float(x0c), float(y0c)], dtype=float)
                    ctr = (int(round(ps_px[0])), int(round(ps_px[1])))
                    cv2.circle(overlay, ctr, dot_r + 2, dot_outline, -1)
                    cv2.circle(overlay, ctr, dot_r, (0, 0, 255), -1)
                    cv2.putText(overlay, f"S{i+1}", (ctr[0] + 6, ctr[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(
            overlay,
            f"exp=green  sim=red({sim_outline})   t={t_s:5.2f}s  step={step0:04d}" + ("" if w <= 0.0 else f"→{step1:04d}"),
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(overlay)

    writer.release()
    cap.release()
    print(f"[ok] wrote {out_video}")
    print(f"[info] exp fps={fps_exp:.3g}, out fps={out_fps:.3g}, frames={len(frames)} (step={int(args.frame_step)})")
    print(f"[info] svg scale m_per_px={m_per_px:.6e} (block_w_mm={float(args.block_w_mm):g}, base_len0_px={base_len0_px:.3g})")
    print(f"[info] sim block_w={block_w:.3e} m, block_h={block_h:.3e} m, block_xc={block_xc:.3e} m, block_x0={block_x0:.3e} m")


if __name__ == "__main__":
    main()
