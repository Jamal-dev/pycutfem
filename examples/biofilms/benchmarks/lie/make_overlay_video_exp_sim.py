#!/usr/bin/env python3
"""
Create an experimental-vs-simulation overlay video for the Lie benchmark (Li et al. 2020, BIT27491).

This script draws:
  - the experimental biofilm contour extracted from the OCT video (green)
  - the simulated biofilm contour (red), obtained by deforming the *t=0 polygon*
    with the displacement field `u` exported in the simulation VTU files.

Notes
-----
* The experimental pixel->meter scaling is obtained by enforcing the rigid-support width
  to be `--block-w` (default 1 mm), matching the deformation extraction script.
* For best results, run the simulation with `--vtk-every 1` (or 2) so every time step has a VTU.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import meshio
import numpy as np
from scipy.spatial import cKDTree


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
    return c[:, 0, :]


def _simplify_contour(contour_px: np.ndarray, *, eps: float) -> np.ndarray:
    c = contour_px.astype(np.float32)
    if float(eps) <= 0.0 or c.shape[0] < 5:
        return c
    c_in = c.reshape((-1, 1, 2))
    c_out = cv2.approxPolyDP(c_in, epsilon=float(eps), closed=True)
    return np.asarray(c_out, dtype=np.float32).reshape((-1, 2))


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
        _thr_val, mask = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mask = (gray_blur > float(thr)).astype(np.uint8) * 255

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
    _area, chosen_lab, bbox = comps[idx]
    chosen_mask = (labels == int(chosen_lab)).astype(np.uint8) * 255

    meta = {
        "crop": crop,
        "x0": int(x0),
        "y0": int(y0),
        "chosen_bbox": tuple(int(v) for v in bbox),
    }
    return chosen_mask, meta


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
    # Ensure closed (use float tolerance).
    if not np.allclose(pts[0], pts[-1], rtol=0.0, atol=1.0e-12):
        pts = np.vstack([pts, pts[0]])
    return pts


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

    # If any query hits a node exactly, use that value.
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay simulation contour on experimental Lie benchmark video.")
    ap.add_argument(
        "--exp-video",
        type=str,
        default="examples/biofilms/benchmarks/lie/additional_data/bit27491-sup-0001-si_v1.avi",
    )
    ap.add_argument("--sim-dir", type=str, required=True, help="Simulation output directory containing vtk/step=XXXX.vtu.")
    ap.add_argument(
        "--sim-poly0-mm-csv",
        type=str,
        default="examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_ts.csv",
        help="t=0 polygon in mm coords (x centered on base, y=0 at base).",
    )
    ap.add_argument("--out-video", type=str, default="", help="Output video (mp4/avi). Default: <sim-dir>/exp_sim_overlay.mp4")
    ap.add_argument("--frame-step", type=int, default=1, help="Use every N-th experimental frame.")
    ap.add_argument("--max-frames", type=int, default=0, help="If >0, only write first N processed frames.")
    ap.add_argument("--out-fps", type=float, default=float("nan"), help="Output FPS. Default: exp_fps/frame_step.")
    ap.add_argument("--crop", type=str, default="", help="Optional crop: 'x0,y0,w,h' in pixels.")

    # Experimental contour extraction options (same defaults as the extraction script).
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
    ap.add_argument("--simplify-eps", type=float, default=2.0)

    # Base handling + scaling.
    ap.add_argument("--straighten-base", action="store_true", help="Straighten/clamp the experimental base (recommended).")
    ap.add_argument("--base-per-frame", action="store_true", help="Re-detect base (x-span and y) per frame to reduce scan drift.")
    ap.add_argument("--base-y-px", type=float, default=float("nan"), help="Manual base y in pixels (full-frame coords).")
    ap.add_argument("--base-method", type=str, default="mask_mode", choices=("mask_mode", "mask_quantile"))
    ap.add_argument("--base-quantile", type=float, default=0.95)
    ap.add_argument("--base-tol-px", type=float, default=6.0)
    ap.add_argument("--base-min-col-fg", type=int, default=0)
    ap.add_argument("--block-w", type=float, default=1.0e-3, help="Support width used for pixel->meter scaling [m].")

    # Simulation geometry mapping (only used to shift poly0 to global coords when sampling `u`).
    ap.add_argument("--block-h", type=float, default=3.0e-3)
    ap.add_argument("--block-xc", type=float, default=7.5e-3)

    # VTU sampling options.
    ap.add_argument("--k-nn", type=int, default=8, help="k in k-NN inverse-distance interpolation of displacement.")
    ap.add_argument("--idw-power", type=float, default=2.0, help="Power p in 1/d^p weights for k-NN interpolation.")
    args = ap.parse_args()

    exp_video = Path(str(args.exp_video))
    if not exp_video.exists():
        raise FileNotFoundError(f"Experimental video not found: {exp_video}")
    sim_dir = Path(str(args.sim_dir))
    vtk_dir = sim_dir / "vtk"
    step_files = _vtk_step_files(vtk_dir)
    ts_path = sim_dir / "timeseries.csv"
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing {ts_path}. Run the simulation first.")
    ts_arr = np.genfromtxt(str(ts_path), delimiter=",", skip_header=1, dtype=float)
    if ts_arr.ndim == 1:
        ts_arr = ts_arr.reshape(1, -1)
    if ts_arr.shape[1] < 1:
        raise ValueError(f"Unexpected timeseries shape {ts_arr.shape} in {ts_path}")
    t_sim = np.asarray(ts_arr[:, 0], dtype=float)

    poly0_mm = _read_polygon_mm_csv(Path(str(args.sim_poly0_mm_csv)))
    poly0_m_local = poly0_mm * 1.0e-3

    cap = cv2.VideoCapture(str(exp_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {exp_video}")
    fps_exp = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps_exp) or fps_exp <= 0.0:
        fps_exp = 1.0

    crop = _parse_crop(str(args.crop))

    # Frame 0: base detection (for scaling).
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

    base_y_auto, base_x_left, base_x_right, _min_col_fg = _auto_base_from_mask(
        mask0,
        tol_px=float(args.base_tol_px),
        base_method=str(args.base_method),
        base_quantile=float(args.base_quantile),
        min_col_fg=int(args.base_min_col_fg),
        force_base_y_px=float(base_y) if np.isfinite(base_y) else None,
    )
    if not np.isfinite(base_y):
        base_y = float(base_y_auto)

    if bool(args.straighten_base):
        mask0_s, base_y_i = _straighten_base_in_mask(mask0, base_y_px=base_y, x_left_px=base_x_left, x_right_px=base_x_right)
        base_y = float(base_y_i)
        mask0 = mask0_s

    base_w_px = float(base_x_right - base_x_left)
    if base_w_px <= 1.0:
        raise RuntimeError("Base width in pixels is too small; check crop/segmentation.")
    m_per_px = float(args.block_w) / base_w_px

    frames = list(range(0, n_frames, max(1, int(args.frame_step))))
    if int(args.max_frames) > 0:
        frames = frames[: int(args.max_frames)]
    if not frames or frames[0] != 0:
        frames = [0] + frames

    out_fps = float(args.out_fps) if np.isfinite(float(args.out_fps)) else float(fps_exp) / max(1.0, float(args.frame_step))

    if str(args.out_video).strip():
        out_video = Path(str(args.out_video))
    else:
        out_video = sim_dir / "exp_sim_overlay.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)

    # Determine output frame size.
    if crop is not None:
        x0, y0, w, h = crop
        frame_w, frame_h = int(w), int(h)
    else:
        frame_h, frame_w = img0.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if out_video.suffix.lower() == ".mp4" else "XVID"))
    writer = cv2.VideoWriter(str(out_video), fourcc, float(out_fps), (int(frame_w), int(frame_h)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_video}")

    # Cache VTU -> (KDTree, u_nodes).
    vtu_cache: dict[int, tuple[cKDTree, np.ndarray]] = {}

    # If step file missing for a given index, fall back to nearest existing.
    step_ids = np.array(sorted(step_files.keys()), dtype=int)

    def _nearest_step(step_no: int) -> int:
        if step_no in step_files:
            return int(step_no)
        j = int(np.argmin(np.abs(step_ids - int(step_no))))
        return int(step_ids[j])

    block_xc = float(args.block_xc)
    block_h = float(args.block_h)

    for fr in frames:
        img = img0 if fr == 0 else _read_frame(cap, int(fr))
        if crop is not None:
            x0, y0, w, h = crop
            img = img[y0 : y0 + h, x0 : x0 + w].copy()

        # Experimental contour for this frame.
        try:
            mask, meta = _extract_component_mask(
                img,
                crop=None,  # already cropped above
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

            base_y_use = float(base_y)
            x_left_use = int(base_x_left)
            x_right_use = int(base_x_right)
            if bool(args.base_per_frame):
                by, xl, xr, _min_col_fg = _auto_base_from_mask(
                    mask,
                    tol_px=float(args.base_tol_px),
                    base_method=str(args.base_method),
                    base_quantile=float(args.base_quantile),
                    min_col_fg=int(args.base_min_col_fg),
                    force_base_y_px=None,
                )
                base_y_use = float(by)
                x_left_use = int(xl)
                x_right_use = int(xr)

            if bool(args.straighten_base):
                mask, base_y_i = _straighten_base_in_mask(mask, base_y_px=base_y_use, x_left_px=x_left_use, x_right_px=x_right_use)
                base_y_use = float(base_y_i)

            contour_px = _simplify_contour(_largest_contour(mask), eps=float(args.simplify_eps))
            # base center in (cropped) pixel coords
            base_xc_px = 0.5 * (float(x_left_use) + float(x_right_use))
            base_y_px = float(base_y_use)
        except Exception:
            contour_px = None
            base_xc_px = 0.5 * (float(base_x_left) + float(base_x_right))
            base_y_px = float(base_y)

        # Simulation time from frame.
        t_s = float(fr) / float(fps_exp)
        # Map time -> nearest simulation step index (row index == step number).
        step_no = int(np.argmin(np.abs(t_sim - float(t_s))))
        step_no = _nearest_step(step_no)

        if step_no not in vtu_cache:
            vtu = meshio.read(str(step_files[step_no]))
            pts = np.asarray(vtu.points[:, :2], dtype=float)
            if "u" not in vtu.point_data:
                raise KeyError(f"VTU {step_files[step_no]} does not contain point_data['u']")
            u = np.asarray(vtu.point_data["u"][:, :2], dtype=float)
            vtu_cache[step_no] = (cKDTree(pts), u)

        tree, u_nodes = vtu_cache[step_no]

        # Sample u at the poly0 vertices (global coords).
        x0_global = np.column_stack([block_xc + poly0_m_local[:, 0], block_h + poly0_m_local[:, 1]])
        u_q = _interp_displacement_idw(tree=tree, u_nodes=u_nodes, xq=x0_global, k=int(args.k_nn), power=float(args.idw_power))
        x_def_global = x0_global + u_q
        x_def_local = np.column_stack([x_def_global[:, 0] - block_xc, x_def_global[:, 1] - block_h])

        # Map to pixels.
        sim_px = np.column_stack(
            [
                base_xc_px + x_def_local[:, 0] / float(m_per_px),
                base_y_px - x_def_local[:, 1] / float(m_per_px),
            ]
        )
        sim_px_i = np.round(sim_px).astype(np.int32).reshape((-1, 1, 2))

        overlay = img.copy()
        if contour_px is not None:
            exp_i = np.round(contour_px).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [exp_i], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(overlay, [sim_px_i], isClosed=True, color=(0, 0, 255), thickness=2)

        cv2.putText(
            overlay,
            f"t={t_s:5.2f}s  step={step_no:04d}",
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
    print(f"[info] scale m_per_px={m_per_px:.6e} (block_w={float(args.block_w):.3e} m, base_w_px={base_w_px:.1f})")


if __name__ == "__main__":
    main()
