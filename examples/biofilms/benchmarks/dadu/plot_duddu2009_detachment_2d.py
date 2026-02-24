"""
Small helper to compare Duddu2009 detachment-model runs from `duddu2009_detachment_2d.py`.

Reads the per-model CSV time series and plots one selected metric vs time.
All paths stay inside `examples/biofilms/benchmarks/dadu/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(str(path), delimiter=",", names=True, dtype=float, encoding=None)
    if getattr(data, "shape", ()) == ():
        data = np.array([data], dtype=data.dtype)
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Duddu2009 2D detachment benchmark time series.")
    ap.add_argument("--outdir", type=str, default="examples/biofilms/benchmarks/dadu/results/duddu2009_detachment_2d")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--models", type=str, default="shear,l2,poly", help="Comma list: shear,l2,poly,none or 'all'.")
    ap.add_argument(
        "--metric",
        type=str,
        default="A_alpha",
        help="Column name to plot (e.g. A_alpha, A_alpha_window, L_max, L_mean).",
    )
    ap.add_argument("--save", type=str, default="", help="Output image path (png). Default: <outdir>/plot_<metric>.png")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    backend = str(args.backend)
    metric = str(args.metric)

    raw = str(args.models or "").strip().lower()
    if raw in {"all", "*"}:
        raw = "shear,l2,poly"
    models = [p.strip() for p in raw.split(",") if p.strip()]

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"matplotlib is required for plotting ({exc}).")

    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)

    any_ok = False
    for m in models:
        path = outdir / f"model={m}" / f"backend={backend}_timeseries.csv"
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        data = _load_csv(path)
        names = tuple(getattr(data.dtype, "names", ()) or ())
        if "t_days" not in names or metric not in names:
            print(f"[skip] {path} missing required columns (have {names})")
            continue
        ax.plot(np.asarray(data["t_days"], dtype=float), np.asarray(data[metric], dtype=float), label=m)
        any_ok = True

    if not any_ok:
        raise SystemExit("No matching CSVs found to plot.")

    ax.set_xlabel("t (days)")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)

    save = str(args.save or "").strip()
    if not save:
        save = str(outdir / f"plot_{metric}.png")
    save_path = Path(save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=200)
    print(f"[ok] wrote {save_path}")


if __name__ == "__main__":
    main()

