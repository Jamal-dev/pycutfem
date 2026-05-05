#!/usr/bin/env python3
"""Prepare the Biofilm I contours used in the Christan benchmark.

The raw traced coordinates live in the original Blauert preprocessing folder:

  - ``biofilm1.txt`` with ``domain1.txt``: unloaded Biofilm I geometry
  - ``biofilm.txt`` with ``domain.txt``: loaded Biofilm I geometry
  - ``biofilm2.txt`` with ``domain.txt``: alternate loaded trace

Each trace is scaled with its own OCT crop frame into the shared physical
window reported by Picioreanu et al.:

  - crop width  = 2.0 mm
  - crop height = 0.5 mm

The mapped contours are written as repository-local CSV files in millimetres.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
RAW_ROOT = REPO_ROOT / "examples/biofilms/benchmarks/blauert/biofilm_preprocessing"

DEFAULT_OUT_DIR = HERE
DEFAULT_CROP_WIDTH_MM = 2.0
DEFAULT_CROP_HEIGHT_MM = 0.5
DEFAULT_MODEL_BOX_LENGTH_MM = 3.0
DEFAULT_MODEL_BOX_HEIGHT_MM = 1.0


@dataclass(frozen=True)
class GeometrySpec:
    key: str
    label: str
    raw_path: Path
    domain_path: Path
    out_name: str


GEOMETRY_SPECS = (
    GeometrySpec(
        key="initial",
        label="Biofilm I unloaded contour",
        raw_path=RAW_ROOT / "biofilm1.txt",
        domain_path=RAW_ROOT / "domain1.txt",
        out_name="biofilm_I_initial_mm.csv",
    ),
    GeometrySpec(
        key="final_primary",
        label="Biofilm I loaded contour",
        raw_path=RAW_ROOT / "biofilm.txt",
        domain_path=RAW_ROOT / "domain.txt",
        out_name="biofilm_I_final_mm.csv",
    ),
    GeometrySpec(
        key="final_alternative",
        label="Biofilm I loaded contour (alternate trace)",
        raw_path=RAW_ROOT / "biofilm2.txt",
        domain_path=RAW_ROOT / "domain.txt",
        out_name="biofilm_I_final_alt_mm.csv",
    ),
)


def _read_domain_frame(path: Path) -> tuple[float, float, float, float]:
    pts: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pts.append((float(parts[-2]), float(parts[-1])))
        except Exception:
            continue
    arr = np.asarray(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"Could not parse domain frame from {path}.")
    x_left = float(np.min(arr[:, 0]))
    x_right = float(np.max(arr[:, 0]))
    y_top = float(np.min(arr[:, 1]))
    y_bottom = float(np.max(arr[:, 1]))
    if not (x_right > x_left and y_bottom > y_top):
        raise ValueError(f"Invalid domain frame in {path}.")
    return x_left, x_right, y_top, y_bottom


def _ensure_closed(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Expected Nx2 points.")
    if pts.shape[0] < 3:
        raise ValueError("Polygon needs at least three points.")
    if np.allclose(pts[0], pts[-1], rtol=0.0, atol=1.0e-12):
        return pts
    return np.vstack([pts, pts[0]])


def load_raw_polygon_mm(
    raw_path: Path,
    domain_path: Path,
    *,
    crop_width_mm: float = DEFAULT_CROP_WIDTH_MM,
    crop_height_mm: float = DEFAULT_CROP_HEIGHT_MM,
) -> np.ndarray:
    arr = np.loadtxt(str(raw_path), dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected Nx2 raw points in {raw_path}.")
    x_left, x_right, y_top, y_bottom = _read_domain_frame(domain_path)
    xx = (np.asarray(arr[:, 0], dtype=float) - x_left) / (x_right - x_left) * float(crop_width_mm)
    yy = (y_bottom - np.asarray(arr[:, 1], dtype=float)) / (y_bottom - y_top) * float(crop_height_mm)
    return _ensure_closed(np.column_stack([xx, yy]).astype(float))


def _segment_intersections_x(points: np.ndarray, y_sample_mm: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return np.empty((0,), dtype=float)
    out: list[float] = []
    for i in range(pts.shape[0] - 1):
        x0, y0 = float(pts[i, 0]), float(pts[i, 1])
        x1, y1 = float(pts[i + 1, 0]), float(pts[i + 1, 1])
        if not ((y0 <= y_sample_mm <= y1) or (y1 <= y_sample_mm <= y0)):
            continue
        dy = y1 - y0
        if abs(dy) <= 1.0e-14:
            out.extend([x0, x1])
            continue
        tau = (float(y_sample_mm) - y0) / dy
        if -1.0e-12 <= tau <= 1.0 + 1.0e-12:
            out.append(x0 + tau * (x1 - x0))
    return np.asarray(out, dtype=float)


def front_x_mm(points: np.ndarray, y_sample_mm: float) -> float:
    xs = _segment_intersections_x(points, float(y_sample_mm))
    if xs.size == 0:
        return float("nan")
    return float(np.min(xs))


def _pointwise_nearest_dist_um(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size == 0 or bb.size == 0:
        return np.empty((0,), dtype=float)
    diff = aa[:, None, :] - bb[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return 1.0e3 * np.sqrt(np.min(d2, axis=1))


def symmetric_trace_distance_metrics_um(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d_ab = _pointwise_nearest_dist_um(a, b)
    d_ba = _pointwise_nearest_dist_um(b, a)
    both = np.concatenate([d_ab, d_ba]) if d_ab.size or d_ba.size else np.empty((0,), dtype=float)
    if both.size == 0:
        return {
            "mean_um": float("nan"),
            "rmse_um": float("nan"),
            "max_um": float("nan"),
        }
    return {
        "mean_um": float(np.mean(both)),
        "rmse_um": float(np.sqrt(np.mean(both**2))),
        "max_um": float(np.max(both)),
    }


def _bbox_dict(points: np.ndarray) -> dict[str, float]:
    pts = np.asarray(points, dtype=float)
    return {
        "x_min_mm": float(np.min(pts[:, 0])),
        "x_max_mm": float(np.max(pts[:, 0])),
        "y_min_mm": float(np.min(pts[:, 1])),
        "y_max_mm": float(np.max(pts[:, 1])),
    }


def _write_polygon_csv(path: Path, points_mm: np.ndarray) -> None:
    pts = np.asarray(points_mm, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("x_mm,y_mm\n")
        for xx, yy in pts:
            f.write(f"{float(xx):.9f},{float(yy):.9f}\n")


def _front_displacement_block(
    initial: np.ndarray,
    final_primary: np.ndarray,
    final_alternative: np.ndarray,
    *,
    y_levels_um: list[int],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for y_um in y_levels_um:
        y_mm = 1.0e-3 * float(y_um)
        xi = front_x_mm(initial, y_mm)
        xf = front_x_mm(final_primary, y_mm)
        xa = front_x_mm(final_alternative, y_mm)
        out[str(int(y_um))] = {
            "initial_front_x_mm": float(xi),
            "final_primary_front_x_mm": float(xf),
            "final_primary_dx_um": float(1.0e3 * (xf - xi)) if np.isfinite(xi) and np.isfinite(xf) else float("nan"),
            "final_alternative_front_x_mm": float(xa),
            "final_alternative_dx_um": float(1.0e3 * (xa - xi)) if np.isfinite(xi) and np.isfinite(xa) else float("nan"),
        }
    return out


def build_geometry_bundle(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    crop_width_mm: float = DEFAULT_CROP_WIDTH_MM,
    crop_height_mm: float = DEFAULT_CROP_HEIGHT_MM,
    model_box_length_mm: float = DEFAULT_MODEL_BOX_LENGTH_MM,
    model_box_height_mm: float = DEFAULT_MODEL_BOX_HEIGHT_MM,
) -> dict[str, object]:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    contours: dict[str, np.ndarray] = {}
    contour_files: dict[str, str] = {}
    metadata: dict[str, object] = {
        "crop_window_mm": {
            "width_mm": float(crop_width_mm),
            "height_mm": float(crop_height_mm),
        },
        "model_box_mm": {
            "length_mm": float(model_box_length_mm),
            "height_mm": float(model_box_height_mm),
        },
        "recommended_model_placement_mm": {
            "tx_mm": 0.0,
            "ty_mm": 0.0,
            "note": (
                "Place the 2.0 mm x 0.5 mm OCT crop directly inside the 3.0 mm x 1.0 mm "
                "Christan computational box without extra translation."
            ),
        },
        "sources": {},
        "contours": {},
    }

    for spec in GEOMETRY_SPECS:
        pts = load_raw_polygon_mm(
            spec.raw_path,
            spec.domain_path,
            crop_width_mm=float(crop_width_mm),
            crop_height_mm=float(crop_height_mm),
        )
        out_path = out_dir / spec.out_name
        _write_polygon_csv(out_path, pts)
        contours[spec.key] = pts
        contour_files[spec.key] = str(out_path)
        metadata["sources"][spec.key] = {
            "label": spec.label,
            "raw_path": str(spec.raw_path),
            "domain_path": str(spec.domain_path),
        }
        metadata["contours"][spec.key] = {
            "path": str(out_path),
            "n_points": int(pts.shape[0]),
            **_bbox_dict(pts),
        }

    initial = contours["initial"]
    final_primary = contours["final_primary"]
    final_alternative = contours["final_alternative"]

    metadata["front_displacements_um"] = _front_displacement_block(
        initial,
        final_primary,
        final_alternative,
        y_levels_um=[100, 150, 200, 250, 300, 350, 400],
    )
    metadata["trace_disagreement_um"] = {
        "final_primary_vs_alternative": symmetric_trace_distance_metrics_um(final_primary, final_alternative),
        "initial_vs_final_primary": symmetric_trace_distance_metrics_um(initial, final_primary),
        "initial_vs_final_alternative": symmetric_trace_distance_metrics_um(initial, final_alternative),
    }

    metadata_path = out_dir / "biofilm_I_geometry_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    metadata["contour_files"] = contour_files
    return metadata


def ensure_geometry_artifacts(*, force: bool = False, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, object]:
    out_dir = Path(out_dir).resolve()
    required = [out_dir / spec.out_name for spec in GEOMETRY_SPECS]
    metadata_path = out_dir / "biofilm_I_geometry_metadata.json"
    if force or not metadata_path.exists() or any(not path.exists() for path in required):
        return build_geometry_bundle(out_dir=out_dir)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["metadata_path"] = str(metadata_path)
    payload["contour_files"] = {
        spec.key: str(out_dir / spec.out_name)
        for spec in GEOMETRY_SPECS
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--crop-width-mm", type=float, default=DEFAULT_CROP_WIDTH_MM)
    ap.add_argument("--crop-height-mm", type=float, default=DEFAULT_CROP_HEIGHT_MM)
    ap.add_argument("--model-box-length-mm", type=float, default=DEFAULT_MODEL_BOX_LENGTH_MM)
    ap.add_argument("--model-box-height-mm", type=float, default=DEFAULT_MODEL_BOX_HEIGHT_MM)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if bool(args.force):
        payload = build_geometry_bundle(
            out_dir=Path(str(args.out_dir)),
            crop_width_mm=float(args.crop_width_mm),
            crop_height_mm=float(args.crop_height_mm),
            model_box_length_mm=float(args.model_box_length_mm),
            model_box_height_mm=float(args.model_box_height_mm),
        )
    else:
        payload = ensure_geometry_artifacts(force=False, out_dir=Path(str(args.out_dir)))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
