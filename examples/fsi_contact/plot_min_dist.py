#!/usr/bin/env python3
"""
Plot minimal distance-to-wall vs time from `bouncing_ball.py` outputs.

Input format
------------
`bouncing_ball.py` writes `min_dist.csv` with columns:
  t,dt,min_dist,p_bc,vmax_v_f

This script plots `min_dist` against `t` (and optionally the relaxed distance ε).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot min distance vs time (FSI contact benchmark).")
    p.add_argument("csv", nargs="+", type=str, help="Path(s) to min_dist.csv (one or more).")
    p.add_argument("--labels", nargs="*", default=None, help="Optional legend labels (same count as inputs).")
    p.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Relaxed distance ε to plot as a horizontal line. Default: try read from metrics.json; else 1e-4.",
    )
    p.add_argument("--no-eps-line", action="store_true", help="Disable ε horizontal line.")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path (.png/.pdf). Default: <first_dir>/min_dist.png",
    )
    p.add_argument("--title", type=str, default=None, help="Optional plot title.")
    p.add_argument("--show", action="store_true", help="Show interactive window (in addition to saving).")
    return p.parse_args()


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(str(path), delimiter=",", names=True, dtype=float, encoding=None)
    # genfromtxt returns a 0-d structured array for a single row; coerce to 1D
    if getattr(data, "shape", ()) == ():
        data = np.array([data], dtype=data.dtype)
    cols = {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}
    if "t" not in cols or "min_dist" not in cols:
        raise ValueError(f"{path}: expected columns 't' and 'min_dist'; got {sorted(cols.keys())}")
    return cols


def _guess_eps(csv_path: Path) -> float:
    # Prefer metrics.json in the same directory (or parent).
    for cand in (csv_path.with_name("metrics.json"), csv_path.parent / "metrics.json"):
        if not cand.exists():
            continue
        try:
            d = json.loads(cand.read_text())
            eps = float(d.get("params", {}).get("eps_relax", 1.0e-4))
            if np.isfinite(eps) and eps > 0.0:
                return eps
        except Exception:
            pass
    return 1.0e-4


def _default_label(csv_path: Path) -> str:
    parent = csv_path.parent.name
    if parent and parent != ".":
        return parent
    return csv_path.name


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args()
    csv_paths = [Path(p).expanduser().resolve() for p in args.csv]
    for p in csv_paths:
        if not p.exists():
            raise SystemExit(f"File not found: {p}")

    labels = list(args.labels) if args.labels else None
    if labels is not None and len(labels) != len(csv_paths):
        raise SystemExit(f"--labels count ({len(labels)}) must match number of csv files ({len(csv_paths)}).")
    if labels is None:
        labels = [_default_label(p) for p in csv_paths]

    eps = None if args.no_eps_line else (float(args.eps) if args.eps is not None else _guess_eps(csv_paths[0]))
    out = Path(args.out).expanduser().resolve() if args.out else (csv_paths[0].parent / "min_dist.png").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib  # type: ignore

        # Default to a headless backend to avoid Qt/Wayland issues on servers.
        if not bool(args.show):
            matplotlib.use("Agg", force=True)

        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for plotting. Install it in your environment (e.g. `pip install matplotlib`)."
        ) from e

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)

    for path, lab in zip(csv_paths, labels, strict=True):
        cols = _load_csv(path)
        ax.plot(cols["t"], cols["min_dist"], label=str(lab), linewidth=1.6)

    if eps is not None:
        ax.axhline(float(eps), color="k", linestyle="--", linewidth=1.0, alpha=0.8, label=f"ε={eps:g}")

    ax.set_xlabel("time t [s]")
    ax.set_ylabel("min distance to bottom [m]")
    ax.grid(True, which="both", alpha=0.25)
    if args.title:
        ax.set_title(str(args.title))
    if len(csv_paths) > 1 or eps is not None:
        ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(str(out))
    print(f"Wrote: {out}")
    if args.show:  # pragma: no cover
        plt.show()


if __name__ == "__main__":
    main()

