#!/usr/bin/env python3
"""
Compare experimental and simulation deformation time series for the Lie benchmark.

Inputs
------
- Experimental CSV from:
    extract_deformation_timeseries_from_experimental_video_s1.py
- Simulation CSV from:
    lie_synthetic_deformation_one_domain.py

Both files are expected to have columns:
  t_s, dx_line1_m, dx_line2_m, dx_line3_m

Outputs
-------
- A comparison plot (PNG) and basic error metrics (RMSE).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _read_timeseries(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.genfromtxt(str(path), delimiter=",", skip_header=1, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 4:
        raise ValueError(f"Expected >=4 columns in {path}; got shape={arr.shape}")
    t = np.asarray(arr[:, 0], dtype=float)
    y = np.asarray(arr[:, 1:4], dtype=float)
    return t, y


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    window = int(window)
    if window <= 1:
        return x
    if x.size < window:
        return x
    k = np.ones((window,), dtype=float) / float(window)
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(x_pad, k, mode="valid")
    return np.asarray(y, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare exp vs sim dx(t) (Lie benchmark).")
    ap.add_argument("--exp-csv", type=str, default="examples/biofilms/benchmarks/lie/exp_s1_dx_leftmost_scalebar.csv")
    ap.add_argument("--sim-csv", type=str, default="", help="Simulation CSV. If omitted, use --sim-dir/timeseries.csv.")
    ap.add_argument("--sim-dir", type=str, default="", help="Simulation output directory containing timeseries.csv.")
    ap.add_argument("--out", type=str, default="", help="Output PNG path. Default: next to sim CSV.")
    ap.add_argument("--smooth-exp", type=int, default=1, help="Moving-average window for experimental curves.")
    ap.add_argument("--t-max", type=float, default=float("nan"), help="Optional max time [s] for comparison.")
    args = ap.parse_args()

    exp_csv = Path(str(args.exp_csv))
    if str(args.sim_csv).strip():
        sim_csv = Path(str(args.sim_csv))
    else:
        sim_dir = Path(str(args.sim_dir))
        if not str(args.sim_dir).strip():
            raise ValueError("Provide --sim-csv or --sim-dir.")
        sim_csv = sim_dir / "timeseries.csv"

    t_exp, dx_exp = _read_timeseries(exp_csv)
    t_sim, dx_sim = _read_timeseries(sim_csv)

    if np.isfinite(float(args.t_max)):
        tmax = float(args.t_max)
        m_exp = t_exp <= tmax
        m_sim = t_sim <= tmax
        t_exp, dx_exp = t_exp[m_exp], dx_exp[m_exp, :]
        t_sim, dx_sim = t_sim[m_sim], dx_sim[m_sim, :]

    # Smooth exp (optional).
    w = int(args.smooth_exp)
    dx_exp_s = np.column_stack([_moving_average(dx_exp[:, i], w) for i in range(dx_exp.shape[1])])

    # Interpolate sim on exp grid.
    dx_sim_i = np.column_stack([np.interp(t_exp, t_sim, dx_sim[:, i]) for i in range(dx_sim.shape[1])])

    err = dx_sim_i - dx_exp_s
    rmse = np.sqrt(np.nanmean(err * err, axis=0))
    rmse_tot = float(np.sqrt(np.nanmean(err * err)))
    mae = np.nanmean(np.abs(err), axis=0)
    mae_tot = float(np.nanmean(np.abs(err)))

    # Relative error as described in the paper (sum over every second datum).
    sel = np.arange(0, t_exp.size, 2, dtype=int)
    de = np.asarray(dx_exp_s[sel, :], dtype=float)
    dm = np.asarray(dx_sim_i[sel, :], dtype=float)
    num = np.nansum(np.abs(dm - de), axis=0)
    den = np.nansum(np.abs(de), axis=0)
    rel = np.where(den > 0.0, (num / den) * 100.0, np.nan)
    rel_tot = float(np.nansum(np.abs(dm - de)) / max(1.0e-30, np.nansum(np.abs(de))) * 100.0)

    print(f"[exp] {exp_csv}")
    print(f"[sim] {sim_csv}")
    print(f"[rmse] line1={rmse[0]:.6e} m, line2={rmse[1]:.6e} m, line3={rmse[2]:.6e} m, total={rmse_tot:.6e} m")
    print(f"[mae]  line1={mae[0]:.6e} m, line2={mae[1]:.6e} m, line3={mae[2]:.6e} m, total={mae_tot:.6e} m")
    print(f"[rel]  line1={rel[0]:.3g} %, line2={rel[1]:.3g} %, line3={rel[2]:.3g} %, total={rel_tot:.3g} % (every 2nd point)")

    # Plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6.8, 7.2), dpi=200)
    axes = np.asarray(axes).ravel()
    colors = ["tab:blue", "tab:orange", "tab:green"]
    y_max_um = float(max(150.0, np.nanmax(dx_exp_s) * 1.0e6, np.nanmax(dx_sim) * 1.0e6))
    for i, (ax, c) in enumerate(zip(axes, colors)):
        ax.plot(
            t_exp,
            dx_exp_s[:, i] * 1.0e6,
            color=c,
            linestyle="none",
            marker="o",
            markersize=3.5,
            markerfacecolor="none",
            label="exp",
        )
        ax.plot(t_sim, dx_sim[:, i] * 1.0e6, color=c, linewidth=2.0, label="sim")
        ax.set_ylabel("dx [µm]")
        ax.set_ylim(0.0, y_max_um)
        ax.grid(True, alpha=0.25)
        ax.set_title(f"Line {i+1}: RMSE={rmse[i]*1e6:.2g} µm, rel={rel[i]:.2g} %", fontsize=10)
        ax.legend(loc="best", fontsize=8, frameon=True)
    axes[-1].set_xlabel("t [s]")
    fig.suptitle(f"Lie benchmark: exp vs sim (total RMSE={rmse_tot*1e6:.2g} µm)", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    if str(args.out).strip():
        out_png = Path(str(args.out))
    else:
        out_png = sim_csv.with_name("compare_exp_sim_dx_subplots.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png))
    plt.close(fig)
    print(f"[ok] wrote {out_png}")


if __name__ == "__main__":
    main()
