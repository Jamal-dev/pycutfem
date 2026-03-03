#!/usr/bin/env python3
"""
Export frames from Li et al. (2020) OCT videos to image files.

This utility exists to support *manual* contour tracing workflows (e.g. Inkscape),
when fully-automatic extraction is unreliable due to jitter, poor contrast, or
anchor drift.

By default, it exports every frame of Video S1 as PNGs and writes an index CSV
containing the frame number and timestamp.
"""

from __future__ import annotations

import argparse
import csv
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
    return int(x0), int(y0), int(w), int(h)


def _preprocess(
    bgr: np.ndarray,
    *,
    to_gray: bool,
    clahe: bool,
    blur_ksize: int,
) -> np.ndarray:
    img: np.ndarray = bgr
    if bool(to_gray) or bool(clahe):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if bool(clahe):
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = c.apply(img)
    k = int(blur_ksize)
    if k and k > 1:
        if k % 2 == 0:
            k += 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="Export video frames to PNG images (Lie benchmark).")
    ap.add_argument(
        "--video",
        type=str,
        default="examples/biofilms/benchmarks/lie/additional_data/bit27491-sup-0001-si_v1.avi",
        help="Path to the AVI video.",
    )
    ap.add_argument("--out-dir", type=str, default="out/_lie_video_s1_frames", help="Output directory for frames.")
    ap.add_argument("--pattern", type=str, default="frame_{frame:04d}.png", help="Filename pattern (format str).")
    ap.add_argument("--index-csv", type=str, default="frames.csv", help="Index CSV filename (inside --out-dir).")

    ap.add_argument("--start", type=int, default=0, help="First frame index (inclusive).")
    ap.add_argument("--end", type=int, default=-1, help="Last frame index (inclusive). -1 means last frame.")
    ap.add_argument("--frame-step", type=int, default=1, help="Export every Nth frame.")

    ap.add_argument("--crop", type=str, default="", help="Optional crop 'x0,y0,w,h' in pixels.")
    ap.add_argument("--scale", type=float, default=1.0, help="Optional resize scale (e.g. 0.5).")
    ap.add_argument("--gray", action="store_true", help="Export grayscale frames (8-bit).")
    ap.add_argument("--clahe", action="store_true", help="Apply CLAHE before export (implies grayscale).")
    ap.add_argument("--blur-ksize", type=int, default=0, help="Gaussian blur kernel size (odd). 0 disables.")
    args = ap.parse_args()

    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    index_csv = out_dir / str(args.index_csv)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 1.0

    start = int(max(0, int(args.start)))
    end = int(args.end)
    if end < 0:
        end = n_frames - 1
    end = int(min(n_frames - 1, end))
    step = int(max(1, int(args.frame_step)))
    crop = _parse_crop(str(args.crop))
    scale = float(args.scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("--scale must be >0")

    exported = 0
    with index_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "file", "width_px", "height_px", "fps", "video"])

        fr = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if fr < start:
                fr += 1
                continue
            if fr > end:
                break
            if ((fr - start) % step) != 0:
                fr += 1
                continue

            img = frame
            if crop is not None:
                x0, y0, cw, ch = crop
                img = img[y0 : y0 + ch, x0 : x0 + cw].copy()
            img = _preprocess(img, to_gray=bool(args.gray), clahe=bool(args.clahe), blur_ksize=int(args.blur_ksize))
            if abs(scale - 1.0) > 1.0e-12:
                h, ww = img.shape[:2]
                img = cv2.resize(img, (int(round(ww * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

            rel = str(args.pattern).format(frame=int(fr))
            out_path = out_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(out_path), img):
                raise RuntimeError(f"Failed writing: {out_path}")

            h2, w2 = img.shape[:2]
            t_s = float(fr) / float(fps)
            w.writerow([int(fr), f"{t_s:.12e}", str(out_path.relative_to(out_dir)), int(w2), int(h2), float(fps), str(args.video)])
            exported += 1
            fr += 1

    cap.release()
    print(f"[ok] video={args.video}")
    print(f"[ok] fps={fps:g}, n_frames={n_frames}, exported={exported}, range=[{start},{end}], step={step}")
    print(f"[ok] out_dir={out_dir}")
    print(f"[ok] index_csv={index_csv}")


if __name__ == "__main__":
    main()

