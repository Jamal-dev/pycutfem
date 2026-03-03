"""
Regenerate Fig.6-style panels (S and Phi/p) for the one-domain Duddu(2007) benchmark
from `final_fields.npz`.

This avoids rerunning the expensive simulation when runs were done with --skip-plots.

Inputs (in --results-dir)
-------------------------
- summary.json
- final_fields.npz   (written by duddu2007_one_domain_growth_2d_fig6_example2.py)

Outputs (in --results-dir)
--------------------------
- fig6b_S.png
- fig6c_Phi.png

Run
---
conda run --no-capture-output -n fenicsx python \
  examples/biofilms/benchmarks/dadu/plot_one_domain_fig6_panels_from_npz.py \
  --results-dir <OUTDIR>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.biofilms.benchmarks.dadu.plot_one_domain_interface_from_snaps import _alpha_grid_from_map, _build_alpha_grid_map


def _plot_scalar(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    Z: np.ndarray,
    title: str,
    outpng: Path,
    cmap: str,
    alpha_grid: np.ndarray | None = None,
    alpha_half: float = 0.5,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig, ax = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    ax.set_title(str(title))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_xlim(float(xs.min()), float(xs.max()))
    ax.set_ylim(float(ys.min()), float(ys.max()))

    # Use contourf to match the paper-like raster appearance.
    cf = ax.contourf(xs, ys, Z, levels=20, cmap=str(cmap))
    fig.colorbar(cf, ax=ax)

    if alpha_grid is not None:
        ax.contour(xs, ys, alpha_grid, levels=[float(alpha_half)], colors="k", linewidths=1.5)

    fig.savefig(outpng, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, required=True)
    ap.add_argument("--summary", type=str, default="summary.json")
    ap.add_argument("--npz", type=str, default="final_fields.npz")
    ap.add_argument("--alpha-half", type=float, default=0.5)
    ap.add_argument("--no-overlay", action="store_true", help="Disable alpha=0.5 overlay.")
    ap.add_argument("--cmap-S", type=str, default="gray_r", help="Matplotlib colormap for S.")
    ap.add_argument("--cmap-Phi", type=str, default="gray", help="Matplotlib colormap for Phi/p.")
    args = ap.parse_args()

    results_dir = Path(str(args.results_dir))
    summary_path = results_dir / str(args.summary)
    npz_path = results_dir / str(args.npz)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    summary = json.loads(summary_path.read_text())
    L = float(summary["L_mm"])
    H = float(summary["H_mm"])
    nx = int(summary["nx"])
    ny = int(summary["ny"])

    d = np.load(npz_path)
    alpha = np.asarray(d["alpha"], dtype=float).ravel()
    S = np.asarray(d["S"], dtype=float).ravel()
    p = np.asarray(d["p"], dtype=float).ravel()

    xs, ys, ii, jj = _build_alpha_grid_map(L=L, H=H, nx=nx, ny=ny)
    A = _alpha_grid_from_map(xs=xs, ys=ys, ii=ii, jj=jj, alpha_dof=alpha)
    Sg = _alpha_grid_from_map(xs=xs, ys=ys, ii=ii, jj=jj, alpha_dof=S)
    pg = _alpha_grid_from_map(xs=xs, ys=ys, ii=ii, jj=jj, alpha_dof=p)

    overlay = None if bool(args.no_overlay) else A
    alpha_half = float(args.alpha_half)

    _plot_scalar(
        xs=xs,
        ys=ys,
        Z=Sg,
        title="S : substrate concentration (final)",
        outpng=results_dir / "fig6b_S.png",
        cmap=str(args.cmap_S),
        alpha_grid=overlay,
        alpha_half=alpha_half,
    )
    _plot_scalar(
        xs=xs,
        ys=ys,
        Z=pg,
        title="p : potential/pressure surrogate (final)",
        outpng=results_dir / "fig6c_Phi.png",
        cmap=str(args.cmap_Phi),
        alpha_grid=overlay,
        alpha_half=alpha_half,
    )

    print(f"- Wrote {results_dir/'fig6b_S.png'}")
    print(f"- Wrote {results_dir/'fig6c_Phi.png'}")


if __name__ == "__main__":
    main()
