#!/usr/bin/env python3
"""Extract Seboldt et al. (2021) Figure 6 reference curves from the paper SVG.

The extraction is vector-based: the script converts the PDF page to SVG,
identifies the three line-profile panels, maps the colored curves to the paper
legend entries, and writes a tidy CSV that can be used for quantitative
benchmark scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional in lightweight envs
    matplotlib = None
    plt = None

from examples.biofilms.benchmarks.lie.svg_trace_utils import sample_svg_path_d


AXIS_COLOR = "14.892578%, 14.892578%, 14.892578%"
CURVE_LABEL_BY_COLOR = {
    "0%, 0%, 0%": "monolithic_fixed_linear",
    "100%, 0%, 0%": "partitioned_fixed_linear",
    "7.446289%, 62.304688%, 100%": "partitioned_moving_linear",
    "71.679688%, 27.43988%, 100%": "partitioned_moving_nonlinear",
}
PANEL_ORDER = [
    {"panel_index": 0, "kappa": 1.0e-3, "y_max": 4.0e-3},
    {"panel_index": 1, "kappa": 1.0e-4, "y_max": 4.0e-2},
    {"panel_index": 2, "kappa": 1.0e-5, "y_max": 1.5e-1},
]


@dataclass(frozen=True)
class PathRecord:
    stroke: str
    stroke_width: float
    points: np.ndarray

    @property
    def x_min(self) -> float:
        return float(np.min(self.points[:, 0]))

    @property
    def x_max(self) -> float:
        return float(np.max(self.points[:, 0]))

    @property
    def y_min(self) -> float:
        return float(np.min(self.points[:, 1]))

    @property
    def y_max(self) -> float:
        return float(np.max(self.points[:, 1]))

    @property
    def x_span(self) -> float:
        return float(self.x_max - self.x_min)

    @property
    def y_span(self) -> float:
        return float(self.y_max - self.y_min)

    @property
    def x_mid(self) -> float:
        return 0.5 * float(self.x_min + self.x_max)

    @property
    def y_mid(self) -> float:
        return 0.5 * float(self.y_min + self.y_max)


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pdf",
        type=str,
        default=str(here / "20m1386268.pdf"),
        help="Seboldt paper PDF.",
    )
    ap.add_argument(
        "--svg",
        type=str,
        default="",
        help="Optional pre-rendered SVG for the Figure 6 page.",
    )
    ap.add_argument("--page", type=int, default=22, help="PDF page containing Figure 6.")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(here),
        help="Output directory for extracted references.",
    )
    ap.add_argument(
        "--csv-name",
        type=str,
        default="reference_profiles_fig6.csv",
        help="Output CSV filename.",
    )
    ap.add_argument(
        "--metadata-name",
        type=str,
        default="reference_metadata.json",
        help="Output JSON metadata filename.",
    )
    ap.add_argument(
        "--crop-name",
        type=str,
        default="figure6_crop.png",
        help="Output raster crop filename.",
    )
    ap.add_argument(
        "--plot-name",
        type=str,
        default="reference_fig6_reconstructed.png",
        help="Optional reconstructed plot filename.",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=401,
        help="Uniform x-samples written per curve.",
    )
    return ap.parse_args()


def _render_svg(*, pdf_path: Path, page: int, workdir: Path) -> Path:
    svg_path = workdir / f"seboldt_fig6_page{int(page)}.svg"
    cmd = [
        "pdftocairo",
        "-svg",
        "-f",
        str(int(page)),
        "-l",
        str(int(page)),
        str(pdf_path),
        str(svg_path),
    ]
    subprocess.run(cmd, check=True)
    if svg_path.exists():
        return svg_path
    fallback = workdir / f"seboldt_fig6_page{int(page)}"
    if fallback.exists():
        return fallback
    fallback_numbered = workdir / f"seboldt_fig6_page{int(page)}-{int(page)}.svg"
    if fallback_numbered.exists():
        return fallback_numbered
    raise FileNotFoundError(f"Could not locate SVG output for page {page} in {workdir}")


def _render_page_png(*, pdf_path: Path, page: int, workdir: Path) -> Path:
    png_prefix = workdir / f"seboldt_fig6_page{int(page)}"
    cmd = [
        "pdftoppm",
        "-png",
        "-r",
        "200",
        "-f",
        str(int(page)),
        "-l",
        str(int(page)),
        str(pdf_path),
        str(png_prefix),
    ]
    subprocess.run(cmd, check=True)
    return Path(f"{png_prefix}-{int(page)}.png")


def _parse_matrix(transform: str | None) -> tuple[float, float, float, float, float, float]:
    if not transform:
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    text = str(transform).strip()
    if not text.startswith("matrix(") or not text.endswith(")"):
        raise ValueError(f"Unsupported SVG transform: {transform!r}")
    nums = [float(part.strip()) for part in text[7:-1].split(",")]
    if len(nums) != 6:
        raise ValueError(f"Expected 6 transform coefficients, got {len(nums)}")
    return tuple(nums)  # type: ignore[return-value]


def _apply_transform(points: np.ndarray, transform: tuple[float, float, float, float, float, float]) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Expected points with shape (N,2).")
    a, b, c, d, e, f = transform
    out = np.empty_like(pts)
    out[:, 0] = a * pts[:, 0] + c * pts[:, 1] + e
    out[:, 1] = b * pts[:, 0] + d * pts[:, 1] + f
    return out


def _iter_path_records(svg_path: Path) -> list[PathRecord]:
    root = ET.parse(str(svg_path)).getroot()
    records: list[PathRecord] = []
    for elem in root.iter():
        if not str(elem.tag).endswith("path"):
            continue
        stroke = elem.get("stroke")
        d = elem.get("d")
        if not stroke or not d:
            continue
        stroke_width = float(elem.get("stroke-width", "1") or 1.0)
        transform = _parse_matrix(elem.get("transform"))
        try:
            polylines = sample_svg_path_d(d, cubic_samples=24)
        except Exception:
            continue
        for poly in polylines:
            pts = np.asarray(poly, dtype=float)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            pts = _apply_transform(pts, transform)
            if pts.shape[0] < 2:
                continue
            records.append(PathRecord(stroke=str(stroke), stroke_width=stroke_width, points=pts))
    return records


def _cluster_by_xmid(records: Iterable[PathRecord], *, gap_threshold: float = 35.0) -> list[list[PathRecord]]:
    ordered = sorted(records, key=lambda rec: rec.x_mid)
    groups: list[list[PathRecord]] = []
    for rec in ordered:
        if not groups:
            groups.append([rec])
            continue
        prev = groups[-1][-1]
        if abs(rec.x_mid - prev.x_mid) > float(gap_threshold):
            groups.append([rec])
        else:
            groups[-1].append(rec)
    return groups


def _extract_panel_axes(records: list[PathRecord]) -> list[dict[str, float]]:
    axis_records = [rec for rec in records if rec.stroke == f"rgb({AXIS_COLOR})"]
    horizontal = [rec for rec in axis_records if rec.x_span > 70.0 and rec.y_span < 1.0]
    vertical = [rec for rec in axis_records if rec.y_span > 60.0 and rec.x_span < 1.0]
    h_groups = _cluster_by_xmid(horizontal)
    v_groups = _cluster_by_xmid(vertical)
    if len(h_groups) != 3 or len(v_groups) != 3:
        raise RuntimeError(
            f"Expected three panel axis groups, got horizontal={len(h_groups)} vertical={len(v_groups)}."
        )
    panels: list[dict[str, float]] = []
    for idx in range(3):
        h_axis = max(h_groups[idx], key=lambda rec: rec.x_span)
        v_axis = max(v_groups[idx], key=lambda rec: rec.y_span)
        panels.append(
            {
                "x0": float(h_axis.x_min),
                "x1": float(h_axis.x_max),
                "y_base": float(h_axis.y_mid),
                "y_top": float(v_axis.y_min),
            }
        )
    return panels


def _extract_curve_groups(records: list[PathRecord]) -> list[list[PathRecord]]:
    curve_records = [
        rec
        for rec in records
        if rec.stroke in {f"rgb({k})" for k in CURVE_LABEL_BY_COLOR}
        and rec.x_span > 70.0
        and rec.y_span > 20.0
    ]
    groups = _cluster_by_xmid(curve_records)
    if len(groups) != 3:
        raise RuntimeError(f"Expected three curve groups, got {len(groups)}.")
    if any(len(group) != 4 for group in groups):
        raise RuntimeError(f"Expected four curves per group, got {[len(group) for group in groups]}.")
    return groups


def _rgb_text(stroke: str) -> str:
    if stroke.startswith("rgb(") and stroke.endswith(")"):
        return stroke[4:-1]
    return stroke


def _curve_to_data(
    rec: PathRecord,
    *,
    x0: float,
    x1: float,
    y_base: float,
    y_top: float,
    y_max: float,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(rec.points, dtype=float)
    order = np.argsort(pts[:, 0], kind="mergesort")
    x_px = pts[order, 0]
    y_px = pts[order, 1]
    x_rel = (x_px - float(x0)) / max(float(x1 - x0), 1.0e-12)
    y_rel = (float(y_base) - y_px) / max(float(y_base - y_top), 1.0e-12)
    mask = np.isfinite(x_rel) & np.isfinite(y_rel)
    x_rel = x_rel[mask]
    y_rel = y_rel[mask]
    keep = (x_rel >= -1.0e-6) & (x_rel <= 1.0 + 1.0e-6)
    x_rel = x_rel[keep]
    y_rel = y_rel[keep]
    if x_rel.size < 8:
        raise RuntimeError("Too few curve points survived the panel transform.")
    x_rel = np.clip(x_rel, 0.0, 1.0)
    y_data = np.clip(y_rel, 0.0, 1.0) * float(y_max)
    x_unique, inverse = np.unique(np.round(x_rel, 8), return_inverse=True)
    y_unique = np.zeros_like(x_unique)
    counts = np.zeros_like(x_unique)
    for idx_u, idx_y in zip(inverse, range(y_data.size)):
        y_unique[idx_u] += y_data[idx_y]
        counts[idx_u] += 1.0
    y_unique /= np.maximum(counts, 1.0)
    x_eval = np.linspace(0.0, 1.0, int(samples), dtype=float)
    y_eval = np.interp(x_eval, x_unique, y_unique)
    return x_eval, y_eval


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_crop(
    *,
    pdf_path: Path,
    page: int,
    out_path: Path,
    panel_axes: list[dict[str, float]],
) -> None:
    try:
        from PIL import Image
    except Exception:  # pragma: no cover - optional dependency in light envs
        return
    with tempfile.TemporaryDirectory(prefix="seboldt_fig6_") as tmp:
        tmpdir = Path(tmp)
        png_path = _render_page_png(pdf_path=pdf_path, page=int(page), workdir=tmpdir)
        img = Image.open(png_path)
        scale_x = img.size[0] / 612.0
        scale_y = img.size[1] / 792.0
        x0 = min(panel["x0"] for panel in panel_axes)
        x1 = max(panel["x1"] for panel in panel_axes)
        y0 = min(panel["y_top"] for panel in panel_axes)
        y1 = max(panel["y_base"] for panel in panel_axes)
        pad_x = 12.0
        pad_top = 35.0
        pad_bottom = 22.0
        crop = (
            int(max(0.0, math.floor((x0 - pad_x) * scale_x))),
            int(max(0.0, math.floor((y0 - pad_top) * scale_y))),
            int(min(img.size[0], math.ceil((x1 + pad_x) * scale_x))),
            int(min(img.size[1], math.ceil((y1 + pad_bottom) * scale_y))),
        )
        img.crop(crop).save(out_path)


def _write_plot(
    out_path: Path,
    rows: list[dict[str, object]],
) -> None:
    if plt is None:
        return
    by_panel: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for row in rows:
        panel = int(row["panel_index"])
        label = str(row["curve_label"])
        by_panel.setdefault(panel, {}).setdefault(label, ([], []))
    grouped: dict[tuple[int, str], list[tuple[float, float]]] = {}
    for row in rows:
        key = (int(row["panel_index"]), str(row["curve_label"]))
        grouped.setdefault(key, []).append((float(row["x"]), float(row["eta_y"])))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3), constrained_layout=True)
    style = {
        "monolithic_fixed_linear": {"color": "black", "lw": 2.0},
        "partitioned_fixed_linear": {"color": "red", "lw": 2.0, "ls": "--"},
        "partitioned_moving_linear": {"color": "#149dff", "lw": 2.0},
        "partitioned_moving_nonlinear": {"color": "#b748ff", "lw": 2.0},
    }
    for ax, panel_meta in zip(axes, PANEL_ORDER):
        panel_idx = int(panel_meta["panel_index"])
        for label in (
            "monolithic_fixed_linear",
            "partitioned_fixed_linear",
            "partitioned_moving_linear",
            "partitioned_moving_nonlinear",
        ):
            pts = np.asarray(grouped[(panel_idx, label)], dtype=float)
            ax.plot(pts[:, 0], pts[:, 1], label=label, **style[label])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, float(panel_meta["y_max"]))
        ax.set_title(rf"$\kappa={panel_meta['kappa']:.0e}I$")
        ax.set_xlabel("x")
        if panel_idx == 0:
            ax.set_ylabel(r"$\eta_y$")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if args.svg:
        svg_path = Path(args.svg).resolve()
        if not svg_path.exists():
            raise FileNotFoundError(f"SVG not found: {svg_path}")
    else:
        with tempfile.TemporaryDirectory(prefix="seboldt_svg_") as tmp:
            svg_path = _render_svg(pdf_path=pdf_path, page=int(args.page), workdir=Path(tmp))
            records = _iter_path_records(svg_path)
    if args.svg:
        records = _iter_path_records(svg_path)

    panel_axes = _extract_panel_axes(records)
    curve_groups = _extract_curve_groups(records)
    rows: list[dict[str, object]] = []

    for panel_meta, panel_axes_i, curves in zip(PANEL_ORDER, panel_axes, curve_groups):
        grouped = {CURVE_LABEL_BY_COLOR[_rgb_text(rec.stroke)]: rec for rec in curves}
        missing = sorted(set(CURVE_LABEL_BY_COLOR.values()) - set(grouped.keys()))
        if missing:
            raise RuntimeError(f"Missing curve labels in panel {panel_meta['panel_index']}: {missing}")
        for curve_label in (
            "monolithic_fixed_linear",
            "partitioned_fixed_linear",
            "partitioned_moving_linear",
            "partitioned_moving_nonlinear",
        ):
            rec = grouped[curve_label]
            x_eval, y_eval = _curve_to_data(
                rec,
                x0=float(panel_axes_i["x0"]),
                x1=float(panel_axes_i["x1"]),
                y_base=float(panel_axes_i["y_base"]),
                y_top=float(panel_axes_i["y_top"]),
                y_max=float(panel_meta["y_max"]),
                samples=int(args.samples),
            )
            for x_val, y_val in zip(x_eval, y_eval):
                rows.append(
                    {
                        "panel_index": int(panel_meta["panel_index"]),
                        "kappa": float(panel_meta["kappa"]),
                        "curve_label": curve_label,
                        "x": float(x_val),
                        "eta_y": float(y_val),
                        "source": "Seboldt2021_Fig6",
                        "page": int(args.page),
                    }
                )

    csv_path = outdir / str(args.csv_name)
    _write_csv(csv_path, rows)

    metadata = {
        "source": "Seboldt et al. 2021 Figure 6",
        "pdf": str(pdf_path),
        "page": int(args.page),
        "curve_labels": list(CURVE_LABEL_BY_COLOR.values()),
        "panel_order": PANEL_ORDER,
        "panel_axes_page_coordinates": panel_axes,
        "digitization_mode": "vector_svg_extraction",
        "notes": [
            "x-axis is normalized to [0,1] from the paper axes.",
            "y-axis limits follow the panel scales visible in Figure 6.",
            "The extraction keeps the moving-domain linear curve as the primary benchmark target.",
        ],
    }
    metadata_path = outdir / str(args.metadata_name)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    crop_path = outdir / str(args.crop_name)
    _write_crop(pdf_path=pdf_path, page=int(args.page), out_path=crop_path, panel_axes=panel_axes)

    plot_path = outdir / str(args.plot_name)
    _write_plot(plot_path, rows)

    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {metadata_path}")
    if crop_path.exists():
        print(f"[done] wrote {crop_path}")
    if plot_path.exists():
        print(f"[done] wrote {plot_path}")


if __name__ == "__main__":
    main()
