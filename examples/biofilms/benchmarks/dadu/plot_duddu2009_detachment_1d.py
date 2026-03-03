"""
Plot and summarize the Duddu2009 1D detachment benchmark results.

Reads the CSV time series produced by `duddu2009_detachment_1d.py` and generates:
  - a thickness-vs-time plot (PDE vs ODE) per model
  - a small printed error summary

All files and outputs are kept inside `examples/biofilms/benchmarks/dadu/`.
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


def _summary(data: np.ndarray) -> dict[str, float]:
    err = np.abs(np.asarray(data["L_pde_mm"], dtype=float) - np.asarray(data["L_ode_mm"], dtype=float))
    return {
        "t_final_days": float(np.asarray(data["t_days"], dtype=float)[-1]),
        "max_abs_err_mm": float(np.max(err)) if err.size else float("nan"),
        "mean_abs_err_mm": float(np.mean(err)) if err.size else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Duddu2009 1D detachment benchmark thickness time series.")
    ap.add_argument(
        "--outdir",
        type=str,
        default="examples/biofilms/benchmarks/dadu/results/duddu2009_detachment_1d",
        help="Output directory containing model=*_backend=*_timeseries.csv files.",
    )
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--models", type=str, default="shear,poly", help="Comma list: shear,poly,l2.")
    ap.add_argument("--save", type=str, default="", help="Output image path (png). Default: <outdir>/plot_1d_thickness.png")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    backend = str(args.backend)
    models = [m.strip() for m in str(args.models or "").split(",") if m.strip()]
    if not models:
        raise SystemExit("Empty --models list.")

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"matplotlib is required for plotting ({exc}).")

    # Load data
    series: dict[str, np.ndarray] = {}
    for m in models:
        path = outdir / f"model={m}_backend={backend}_timeseries.csv"
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        series[m] = _load_csv(path)

    if not series:
        raise SystemExit("No matching CSVs found.")

    # Plot
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    for m, data in series.items():
        t = np.asarray(data["t_days"], dtype=float)
        ax.plot(t, np.asarray(data["L_pde_mm"], dtype=float), label=f"{m}: PDE", lw=2.0)
        ax.plot(t, np.asarray(data["L_ode_mm"], dtype=float), label=f"{m}: ODE", lw=1.5, ls="--")
    ax.set_xlabel("t (days)")
    ax.set_ylabel("thickness l(t) (mm)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False, ncol=2)

    save = str(args.save or "").strip()
    if not save:
        save = str(outdir / "plot_1d_thickness.png")
    save_path = Path(save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=200)
    print(f"[ok] wrote {save_path}")

    # Print summaries
    print("\nError summary (PDE vs ODE):")
    for m, data in series.items():
        s = _summary(data)
        print(
            f"  {m:>5s}: t_final={s['t_final_days']:.4g} d  "
            f"max|Δl|={s['max_abs_err_mm']:.3e} mm  mean|Δl|={s['mean_abs_err_mm']:.3e} mm"
        )

    # Optional cross-model comparison (shear vs poly)
    if "shear" in series and "poly" in series:
        a = series["shear"]
        b = series["poly"]
        n = min(int(a.shape[0]), int(b.shape[0]))
        if n > 0 and np.allclose(a["t_days"][:n], b["t_days"][:n]):
            diff = np.abs(np.asarray(a["L_pde_mm"][:n], dtype=float) - np.asarray(b["L_pde_mm"][:n], dtype=float))
            print(f"\nCross-model thickness diff (shear vs poly, PDE): max={float(np.max(diff)):.3e} mm")


if __name__ == "__main__":
    main()

