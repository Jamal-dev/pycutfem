from __future__ import annotations

import argparse
import json
from pathlib import Path

import meshio
import numpy as np


def _load_point_data(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    mesh = meshio.read(path)
    pts = np.asarray(mesh.points[:, :2], dtype=float)
    pdata = {k: np.asarray(v) for k, v in mesh.point_data.items()}
    return pts, pdata


def _match_points(ref: np.ndarray, cur: np.ndarray, *, tol: float = 1.0e-12) -> np.ndarray:
    cur_key = {tuple(np.round(xy / tol).astype(np.int64)): i for i, xy in enumerate(cur)}
    order = np.empty(ref.shape[0], dtype=int)
    for i, xy in enumerate(ref):
        key = tuple(np.round(xy / tol).astype(np.int64))
        if key not in cur_key:
            raise KeyError(f"Could not match point {xy} within tolerance {tol}.")
        order[i] = int(cur_key[key])
    return order


def _field_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    da = np.asarray(a, dtype=float)
    db = np.asarray(b, dtype=float)
    diff = da - db
    return {
        "linf": float(np.linalg.norm(diff.reshape(-1), ord=np.inf)),
        "l2": float(np.linalg.norm(diff.reshape(-1), ord=2)),
        "ref_l2": float(max(1.0, np.linalg.norm(da.reshape(-1), ord=2), np.linalg.norm(db.reshape(-1), ord=2))),
        "rel_l2": float(np.linalg.norm(diff.reshape(-1), ord=2) / max(1.0, np.linalg.norm(da.reshape(-1), ord=2), np.linalg.norm(db.reshape(-1), ord=2))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare reduced step-42 2D reference VTUs from deal.II and pycutfem.")
    ap.add_argument("--dealii", type=Path, required=True)
    ap.add_argument("--pycutfem", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pts_d, data_d = _load_point_data(Path(args.dealii))
    pts_p, data_p = _load_point_data(Path(args.pycutfem))
    order = _match_points(pts_d, pts_p)

    fields = sorted(set(data_d.keys()) & set(data_p.keys()))
    report: dict[str, object] = {
        "dealii": str(args.dealii),
        "pycutfem": str(args.pycutfem),
        "n_points": int(pts_d.shape[0]),
        "fields": {},
    }
    for field in fields:
        report["fields"][field] = _field_metrics(np.asarray(data_d[field]), np.asarray(data_p[field])[order])  # type: ignore[index]

    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
      Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
