#!/usr/bin/env python3
"""Create a clean Benchmark 6 front-shift figure with channel context.

The earlier contour-only figure was hard to read because the whole-frame video
segmentation is not a clean geometric contour. This version instead combines:

  - the hand-traced preprocessing contours (`biofilm1.txt` and `biofilm.txt`)
    to show a clear in-channel shape comparison,
  - the checked-in video extractor CSV to show the direct scale-bar-based
    `~91.19 um` front shift at `t ~= 2 s`.

That makes the calibration mismatch visible:

  - red arrow: preprocessing/manual-trace shift (`~148.92 um`),
  - blue arrow: video-extracted shift (`~91.19 um`).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
from matplotlib.patches import Rectangle

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_EXTRACTED_CSV = _ROOT / "examples/biofilms/benchmarks/blauert/exp_front_displacement_from_video.csv"
_TRACE_T0 = _ROOT / "examples/biofilms/benchmarks/blauert/biofilm_preprocessing/biofilm1.txt"
_TRACE_T2 = _ROOT / "examples/biofilms/benchmarks/blauert/biofilm_preprocessing/biofilm.txt"
_DEFAULT_OUT = _ROOT / "examples/biofilms/benchmarks/blauert/observable_front_shift_t0_t2.png"


def _read_front_row(csv_path: Path, target_t_s: float) -> dict[str, float]:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    best = min(rows, key=lambda row: abs(float(row["t_s"]) - float(target_t_s)))
    return {key: float(value) for key, value in best.items()}


def _close_poly(pts: np.ndarray) -> np.ndarray:
    poly = np.asarray(pts, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError("Expected an Nx2 polygon with at least 3 points.")
    if not np.allclose(poly[0], poly[-1], rtol=0.0, atol=1.0e-12):
        poly = np.vstack([poly, poly[0]])
    return poly


def _trace_polygon_mm(trace_path: Path, *, L_um: float = 2000.0, shift_um: float = 0.0) -> np.ndarray:
    """Map a hand-traced contour using the same normalization as data_processing.m."""
    data = np.loadtxt(str(trace_path), dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected Nx2 trace data in {trace_path}")
    vx = np.asarray(data[:, 0], dtype=float)
    vy = np.asarray(data[:, 1], dtype=float)
    x_max = float(np.max(vx))
    y_max = float(np.max(vy))
    if not (x_max > 0.0):
        raise ValueError(f"Invalid x_max in {trace_path}")

    vy = float(y_max) - vy
    vx_um = (vx / float(x_max)) * float(L_um) + float(shift_um)
    vy_um = (vy / float(x_max)) * float(L_um)
    return _close_poly(np.column_stack([vx_um * 1.0e-3, vy_um * 1.0e-3]))


def _front_x_mm(poly_mm: np.ndarray) -> float:
    return float(np.min(np.asarray(poly_mm, dtype=float)[:, 0]))


def _build_plot(
    *,
    contour_t0_mm: np.ndarray,
    contour_t2_mm: np.ndarray,
    front_video_t0_mm: float,
    front_video_t2_mm: float,
    front_trace_t0_mm: float,
    front_trace_t2_mm: float,
    t2_label_s: float,
    dx_video_um: float,
    dx_trace_um: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.9), constrained_layout=True)

    channel_x0 = 0.0
    channel_x1 = 2.05
    channel_h = 1.0
    ax.add_patch(
        Rectangle(
            (channel_x0, 0.0),
            channel_x1 - channel_x0,
            channel_h,
            facecolor="#dff1f8",
            edgecolor="#7ea7b8",
            linewidth=1.6,
            zorder=0,
        )
    )
    ax.axhspan(0.0, 0.02, color="#a7adb4", alpha=0.95, zorder=1)
    ax.text(0.04, 0.985, "top wall", color="#4a6b7a", fontsize=9, ha="left", va="top")
    ax.text(0.04, 0.028, "substrate", color="#4b4f54", fontsize=9, ha="left", va="bottom")
    ax.annotate(
        "flow",
        xy=(1.97, 0.90),
        xytext=(1.55, 0.90),
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#4a6b7a"},
        color="#4a6b7a",
        ha="left",
        va="center",
        fontsize=10,
    )

    ax.fill(contour_t0_mm[:, 0], contour_t0_mm[:, 1], color="#111111", alpha=0.13, zorder=2)
    ax.fill(contour_t2_mm[:, 0], contour_t2_mm[:, 1], color="#b5422c", alpha=0.16, zorder=2)
    ax.plot(
        contour_t0_mm[:, 0],
        contour_t0_mm[:, 1],
        color="#111111",
        linewidth=2.2,
        zorder=3,
        label="Hand trace used as t = 0 s reference (`biofilm1.txt`)",
    )
    ax.plot(
        contour_t2_mm[:, 0],
        contour_t2_mm[:, 1],
        color="#b5422c",
        linewidth=2.2,
        zorder=3,
        label="Hand trace used as t ~= 2 s reference (`biofilm.txt`)",
    )

    trace_arrow_y_mm = 0.73
    video_arrow_y_mm = 0.57
    guide_y0_mm = 0.0
    ax.vlines(
        [front_trace_t0_mm, front_trace_t2_mm],
        ymin=guide_y0_mm,
        ymax=trace_arrow_y_mm,
        colors=["#555555", "#b5422c"],
        linestyles="--",
        linewidth=1.2,
        zorder=2,
    )
    ax.annotate(
        "",
        xy=(front_trace_t2_mm, trace_arrow_y_mm),
        xytext=(front_trace_t0_mm, trace_arrow_y_mm),
        arrowprops={"arrowstyle": "<->", "lw": 1.9, "color": "#8e2f22"},
        zorder=4,
    )
    ax.text(
        0.5 * (front_trace_t0_mm + front_trace_t2_mm),
        trace_arrow_y_mm + 0.018,
        f"manual-trace calibration: {dx_trace_um:.2f} um",
        color="#8e2f22",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        zorder=5,
    )

    ax.vlines(
        [front_video_t0_mm, front_video_t2_mm],
        ymin=guide_y0_mm,
        ymax=video_arrow_y_mm,
        colors=["#627784", "#1f4f7a"],
        linestyles=":",
        linewidth=1.4,
        zorder=2,
    )
    ax.annotate(
        "",
        xy=(front_video_t2_mm, video_arrow_y_mm),
        xytext=(front_video_t0_mm, video_arrow_y_mm),
        arrowprops={"arrowstyle": "<->", "lw": 1.9, "color": "#1f4f7a"},
        zorder=4,
    )
    ax.text(
        0.5 * (front_video_t0_mm + front_video_t2_mm),
        video_arrow_y_mm + 0.018,
        f"video scale-bar calibration: {dx_video_um:.2f} um",
        color="#1f4f7a",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        zorder=5,
    )

    ax.text(
        1.07,
        0.43,
        "Two different length calibrations are being compared here:\n"
        "red = hand-traced still images normalized by Matlab preprocessing\n"
        "blue = direct video measurement using the 250 um scale bar",
        fontsize=9.2,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.94},
        zorder=5,
    )

    ax.set_xlim(channel_x0, channel_x1)
    ax.set_ylim(-0.02, channel_h + 0.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm] in the traced 2 mm OCT window")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Benchmark 6: clear channel-context view of the competing t ~= {t2_label_s:.3f} s front shifts")
    ax.grid(True, linestyle=":", linewidth=0.6, color="#cccccc")
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=_EXTRACTED_CSV)
    ap.add_argument("--trace-t0", type=Path, default=_TRACE_T0)
    ap.add_argument("--trace-t2", type=Path, default=_TRACE_T2)
    ap.add_argument("--t2", type=float, default=2.0, help="Target second contour time in seconds.")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = ap.parse_args()

    row0 = _read_front_row(args.csv, 0.0)
    row2 = _read_front_row(args.csv, float(args.t2))
    contour_t0_mm = _trace_polygon_mm(args.trace_t0)
    contour_t2_mm = _trace_polygon_mm(args.trace_t2)

    _build_plot(
        contour_t0_mm=contour_t0_mm,
        contour_t2_mm=contour_t2_mm,
        front_video_t0_mm=float(row0["x_front_um"]) * 1.0e-3,
        front_video_t2_mm=float(row2["x_front_um"]) * 1.0e-3,
        front_trace_t0_mm=_front_x_mm(contour_t0_mm),
        front_trace_t2_mm=_front_x_mm(contour_t2_mm),
        t2_label_s=float(row2["t_s"]),
        dx_video_um=float(row2["dx_front_um"]),
        dx_trace_um=1000.0 * (_front_x_mm(contour_t2_mm) - _front_x_mm(contour_t0_mm)),
        out_path=args.out,
    )
    print(args.out)


if __name__ == "__main__":
    main()
