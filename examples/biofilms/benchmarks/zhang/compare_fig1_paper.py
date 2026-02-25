"""
Digitize Zhang–Cogan–Wang (2008) Fig. 1 (cavity single-hump growth) and compare
against a one-domain run.

This script produces *quantitative* geometry metrics from the paper's published
snapshots by extracting a proxy interface height profile h(x) from each subplot
and computing the same metrics as the one-domain benchmark driver.

Outputs
-------
- `paper_fig1_metrics.csv`: geometry metrics for panels (a,b,c,d)
- `compare_fig1_metrics.csv`: paper-vs-run table (nearest run time to each panel time)

Notes / limitations
-------------------
The paper defines the biofilm/solvent interface as φ_n = 0, but the published
snapshots show only a set of contour lines (starting from a small positive value).
The digitized "interface" is therefore a proxy: the *topmost visible contour
pixels* per x-column (after removing axes/colorbar).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from metrics import InterfaceProfile, compute_profile_metrics


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _render_pdf_page(*, pdf: Path, page: int, dpi: int, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = outdir / f"zhang2008_page{int(page):02d}_{int(dpi)}dpi"
    _run(["pdftoppm", "-f", str(int(page)), "-l", str(int(page)), "-png", "-r", str(int(dpi)), str(pdf), str(prefix)])

    candidates = sorted(prefix.parent.glob(f"{prefix.name}-*.png"))
    if not candidates:
        raise FileNotFoundError(f"pdftoppm did not produce any PNG output for prefix: {prefix}")
    prefer = [p for p in candidates if p.stem.endswith(f"-{int(page)}")]
    return prefer[0] if prefer else candidates[0]


def _crop_fig1_region(page_img: np.ndarray) -> np.ndarray:
    """
    Heuristic crop to the region containing the 2x2 subplots of Fig. 1.

    The crop is defined in relative units so it stays stable across DPI choices.
    """
    H, W = page_img.shape[:2]
    x0 = int(0.08 * W)
    x1 = int(0.92 * W)
    y0 = int(0.04 * H)
    y1 = int(0.46 * H)
    crop = page_img[y0:y1, x0:x1].copy()

    # Remove top header text and most of the caption by trimming relative margins.
    h = crop.shape[0]
    top = int(0.08 * h)
    bottom = int(0.97 * h)
    return crop[top:bottom, :, :]


def _cluster_1d(vals: list[int], *, eps: int = 5) -> list[list[int]]:
    if not vals:
        return []
    s = sorted(int(v) for v in vals)
    clusters: list[list[int]] = [[s[0]]]
    for v in s[1:]:
        if abs(v - clusters[-1][-1]) <= int(eps):
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return clusters


def _estimate_colorbar_start(panel_rgb: np.ndarray, *, chroma_thr: int = 20, frac_thr: float = 0.25) -> int | None:
    """Return the x-index where the right-side colorbar starts, or None if not detected."""
    chroma = panel_rgb.max(axis=2).astype(np.int16) - panel_rgb.min(axis=2).astype(np.int16)
    m = chroma > int(chroma_thr)
    col_counts = m.sum(axis=0)
    thr = int(frac_thr * panel_rgb.shape[0])
    cols = np.where(col_counts > thr)[0]
    if cols.size == 0:
        return None
    # Rightmost run.
    splits = np.where(np.diff(cols) > 1)[0]
    runs = np.split(cols, splits + 1)
    run = max(runs, key=lambda r: float(np.median(r)))
    return int(run.min())


def _detect_plot_bbox(panel_rgb: np.ndarray) -> tuple[int, int, int, int]:
    """
    Detect the plot-frame bounding box inside a single panel image (excludes colorbar).

    Returns (x_left, y_top, x_right, y_bottom) in panel-local pixel coordinates.
    """
    h, w = panel_rgb.shape[:2]
    x_cb = _estimate_colorbar_start(panel_rgb)
    x_max = int(x_cb - 4) if x_cb is not None and x_cb > 20 else w

    roi = panel_rgb[:, :x_max, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    min_len_v = int(0.35 * h)
    min_len_h = int(0.35 * (x_max))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(80, min(min_len_v, min_len_h)),
        maxLineGap=10,
    )
    if lines is None:
        raise RuntimeError("Failed to detect plot frame (no Hough lines).")

    xs: list[int] = []
    y0s: list[int] = []
    y1s: list[int] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = abs(int(x2) - int(x1))
        dy = abs(int(y2) - int(y1))
        if dy >= min_len_v and dy > 5 * dx:
            xs.append(int(round(0.5 * (x1 + x2))))
            y0s.append(int(min(y1, y2)))
            y1s.append(int(max(y1, y2)))

    if len(xs) < 2:
        raise RuntimeError("Failed to detect plot frame (insufficient vertical axis lines).")

    x_clusters = _cluster_1d(xs, eps=6)
    x_clusters.sort(key=lambda c: float(np.median(c)))
    if len(x_clusters) < 2:
        raise RuntimeError("Failed to detect plot frame (could not cluster vertical lines).")

    x_left = int(round(float(np.median(x_clusters[0]))))
    # Prefer a right boundary supported by >=2 line detections (avoids stray colorbar edges).
    x_right_candidates = [c for c in x_clusters[1:] if len(c) >= 2]
    x_right_cluster = x_right_candidates[-1] if x_right_candidates else x_clusters[-1]
    x_right = int(round(float(np.median(x_right_cluster))))

    # y-extents from the selected vertical-line clusters.
    eps = 8
    y_top = None
    y_bottom = None
    for (x_mid, y0, y1) in zip(xs, y0s, y1s):
        if abs(int(x_mid) - int(x_left)) <= eps or abs(int(x_mid) - int(x_right)) <= eps:
            y_top = y0 if y_top is None else min(int(y_top), int(y0))
            y_bottom = y1 if y_bottom is None else max(int(y_bottom), int(y1))
    if y_top is None or y_bottom is None:
        y_top = int(min(y0s))
        y_bottom = int(max(y1s))

    if x_right <= x_left or y_bottom <= y_top:
        raise RuntimeError("Invalid plot bbox detected.")

    return x_left, y_top, x_right, y_bottom


def _height_profile_from_panel(
    *,
    panel_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    Lx: float,
    Hy: float,
    pad: int = 3,
    border: int = 6,
    chroma_thr: int = 20,
    diff_thr: float = 25.0,
    dark_thr: float = 70.0,
    sample_every: int = 2,
) -> InterfaceProfile:
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0i, y0i = x0 + int(pad), y0 + int(pad)
    x1i, y1i = x1 - int(pad), y1 - int(pad)
    if x1i <= x0i or y1i <= y0i:
        raise ValueError("Plot bbox too small after padding.")

    plot = panel_rgb[y0i:y1i, x0i:x1i, :].astype(np.float32)
    h_pix, w_pix = plot.shape[:2]

    diff = np.linalg.norm(plot - 255.0, axis=2)
    chroma = plot.max(axis=2) - plot.min(axis=2)
    mask_color = (chroma > float(chroma_thr)) & (diff > float(diff_thr))
    # If there are essentially no colored pixels (panel a), fall back to dark pixels.
    if float(mask_color.mean()) < 1.0e-5:
        mask = diff > float(dark_thr)
    else:
        mask = mask_color

    b = int(max(0, border))
    if b > 0 and h_pix > 2 * b and w_pix > 2 * b:
        mask[:b, :] = False
        mask[-b:, :] = False
        mask[:, :b] = False
        mask[:, -b:] = False

    xs: list[float] = []
    hs: list[float] = []
    step = max(1, int(sample_every))
    for j in range(0, w_pix, step):
        col = mask[:, j]
        if not np.any(col):
            xs.append(float("nan"))
            hs.append(float("nan"))
            continue
        y_pix = int(np.min(np.where(col)[0]))

        x_full = float(x0i + j)
        y_full = float(y0i + y_pix)
        x_phys = (x_full - float(x0)) / max(float(x1) - float(x0), 1.0) * float(Lx)
        y_phys = (float(y1) - y_full) / max(float(y1) - float(y0), 1.0) * float(Hy)
        xs.append(float(x_phys))
        hs.append(float(y_phys))

    x_arr = np.asarray(xs, dtype=float)
    h_arr = np.asarray(hs, dtype=float)
    m = np.isfinite(x_arr) & np.isfinite(h_arr)
    if np.any(m):
        x_arr = x_arr[m]
        h_arr = h_arr[m]
    return InterfaceProfile(x=x_arr, h=h_arr)


def _load_run_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(str(path), delimiter=",", names=True, dtype=float, encoding=None)
    if getattr(data, "shape", ()) == ():
        data = np.array([data], dtype=data.dtype)
    return np.asarray(data)


def _nearest_row(data: np.ndarray, *, t: float) -> np.void:
    tt = np.asarray(data["t"], dtype=float)
    i = int(np.argmin(np.abs(tt - float(t))))
    return data[i]


def main() -> None:
    ap = argparse.ArgumentParser(description="Digitize Zhang (2008) Fig. 1 and compare metrics with a one-domain run.")
    ap.add_argument("--pdf", type=str, required=True, help="Path to zhang2008.pdf")
    ap.add_argument("--page", type=int, default=10, help="PDF page containing Fig. 1 (default: 10 for provided zhang2008.pdf)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--outdir", type=str, default="out/zhang2008_fig1_compare")
    ap.add_argument("--run-csv", type=str, required=True, help="Path to one-domain run timeseries CSV.")
    ap.add_argument("--Lx", type=float, default=2.0, help="Paper figure x-extent used for mapping (default: 2).")
    ap.add_argument("--Hy", type=float, default=2.0, help="Paper figure y-extent used for mapping (default: 2).")
    ap.add_argument("--sample-every", type=int, default=2)
    ap.add_argument("--pad", type=int, default=3)
    ap.add_argument("--border", type=int, default=6)
    ap.add_argument("--chroma-thr", type=int, default=20)
    ap.add_argument("--diff-thr", type=float, default=25.0)
    ap.add_argument("--dark-thr", type=float, default=70.0)
    ap.add_argument("--debug", type=int, default=1, help="Write debug crops/masks (1) or not (0).")
    args = ap.parse_args()

    outdir = Path(str(args.outdir)).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    pdf = Path(str(args.pdf)).expanduser().resolve()
    page_png = _render_pdf_page(pdf=pdf, page=int(args.page), dpi=int(args.dpi), outdir=outdir)

    page_rgb = np.asarray(Image.open(page_png).convert("RGB"))
    fig_rgb = _crop_fig1_region(page_rgb)
    Image.fromarray(fig_rgb).save(outdir / "fig1_crop.png")

    H, W = fig_rgb.shape[:2]
    midx = W // 2
    midy = H // 2
    panels = {
        "a": fig_rgb[0:midy, 0:midx, :],
        "b": fig_rgb[0:midy, midx:W, :],
        "c": fig_rgb[midy:H, 0:midx, :],
        "d": fig_rgb[midy:H, midx:W, :],
    }
    times = {"a": 0.0, "b": 100.0, "c": 200.0, "d": 300.0}

    paper_rows = []
    for key, panel in panels.items():
        bbox = _detect_plot_bbox(panel)
        if bool(int(args.debug)):
            dbg = panel.copy()
            x0, y0, x1, y1 = bbox
            cv2.rectangle(dbg, (x0, y0), (x1, y1), (255, 0, 0), 2)
            Image.fromarray(dbg).save(outdir / f"panel_{key}_bbox.png")

        prof = _height_profile_from_panel(
            panel_rgb=panel,
            bbox=bbox,
            Lx=float(args.Lx),
            Hy=float(args.Hy),
            pad=int(args.pad),
            border=int(args.border),
            chroma_thr=int(args.chroma_thr),
            diff_thr=float(args.diff_thr),
            dark_thr=float(args.dark_thr),
            sample_every=int(args.sample_every),
        )
        pm = compute_profile_metrics(profile=prof, Lx=float(args.Lx), far_frac=0.25)
        paper_rows.append({"panel": key, "t": float(times[key]), **asdict(pm)})

    paper_csv = outdir / "paper_fig1_metrics.csv"
    with paper_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paper_rows[0].keys()))
        w.writeheader()
        w.writerows(paper_rows)

    run = _load_run_csv(Path(str(args.run_csv)).expanduser().resolve())
    compare_rows = []
    for r in paper_rows:
        t = float(r["t"])
        rr = _nearest_row(run, t=t)
        compare_rows.append(
            {
                "t": t,
                "paper_H_base": float(r["H_base"]),
                "run_H_base": float(rr["H_base"]),
                "paper_H_max": float(r["H_max"]),
                "run_H_max": float(rr["H_max"]),
                "paper_W_half": float(r["W_half"]),
                "run_W_half": float(rr["W_half"]),
                "paper_mush": float(r["mushroomness"]),
                "run_mush": float(rr["mushroomness"]),
            }
        )

    compare_csv = outdir / "compare_fig1_metrics.csv"
    with compare_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
        w.writeheader()
        w.writerows(compare_rows)

    print(f"[done] wrote {paper_csv}")
    print(f"[done] wrote {compare_csv}")


if __name__ == "__main__":
    main()
