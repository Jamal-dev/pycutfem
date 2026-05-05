#!/usr/bin/env python3
"""Paper 1 Benchmark 3: Wang2014-style layered two-domain vs one-domain comparison.

This benchmark replaces the old Duddu growth comparison for Paper 1.
The goal is to isolate the sharp-interface consistency question in a fixed
geometry where the comparison is not polluted by growth kinetics, topology
change, or merger behavior.

We reproduce the idea of Wang2014 Example 6.1:

  - flat interface separating free fluid and porous medium,
  - one-domain transition-layer model on the full strip,
  - two-domain sharp-interface reference on the split strip,
  - comparison of tangential velocity profiles and regional L2 / semi-H1 errors
    as K -> 0.

The implementation here is deliberately lightweight and reproducible:

  - the one-domain profile is solved from the 1D transition-layer ODE using a
    second-order finite-difference discretization on a uniform y-grid,
  - the two-domain reference uses the closed-form piecewise profile consistent
    with the Wang2014 Example 6.1 setup,
  - all outputs stay outside the Overleaf tree and are copied into the paper
    folder only by the paper-ready suite.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class WangExample61Parameters:
    phi_p: float = 0.4
    mu: float = 1.0
    transition_sharpness: float = 100.0
    y_bottom: float = -1.0
    y_top: float = 1.0


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        out.append(float(text))
    if not out:
        raise ValueError("Expected at least one numeric value.")
    return out


def _format_k_tag(K: float) -> str:
    return f"{float(K):.0e}".replace("+", "")


def _phi_1(y: np.ndarray, *, K: float, params: WangExample61Parameters) -> np.ndarray:
    eps = math.sqrt(float(K))
    mu_eff_ratio = 1.0 / float(params.phi_p)
    arg = float(params.transition_sharpness) * np.asarray(y, dtype=float) / eps
    return 0.5 * (1.0 - mu_eff_ratio) * np.tanh(arg) + 0.5 * (1.0 + mu_eff_ratio)


def _phi_2(y: np.ndarray, *, K: float, params: WangExample61Parameters) -> np.ndarray:
    eps = math.sqrt(float(K))
    arg = float(params.transition_sharpness) * np.asarray(y, dtype=float) / eps
    return 0.5 - 0.5 * np.tanh(arg)


def exact_two_domain_profile(y: np.ndarray, *, K: float, params: WangExample61Parameters) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    beta = math.sqrt(float(K) / float(params.phi_p))
    phi0 = 0.5 - (0.5 - float(K)) / (beta + 1.0)
    out = np.empty_like(y)
    fluid = y >= 0.0
    out[fluid] = -0.5 * (y[fluid] * y[fluid] - 1.0) + (0.5 - float(K)) * (y[fluid] - 1.0) / (beta + 1.0)
    a = math.sqrt(float(params.phi_p) / float(K))
    out[~fluid] = phi0 * np.exp(y[~fluid] * a) - (float(K) / float(params.mu)) * np.tanh(y[~fluid] * a)
    return out


def exact_two_domain_gradient(y: np.ndarray, *, K: float, params: WangExample61Parameters) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    beta = math.sqrt(float(K) / float(params.phi_p))
    phi0 = 0.5 - (0.5 - float(K)) / (beta + 1.0)
    out = np.empty_like(y)
    fluid = y >= 0.0
    out[fluid] = -y[fluid] + (0.5 - float(K)) / (beta + 1.0)
    a = math.sqrt(float(params.phi_p) / float(K))
    out[~fluid] = phi0 * a * np.exp(y[~fluid] * a) - (float(K) / float(params.mu)) * a / np.cosh(y[~fluid] * a) ** 2
    return out


def _solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lower = np.asarray(lower, dtype=float).copy()
    diag = np.asarray(diag, dtype=float).copy()
    upper = np.asarray(upper, dtype=float).copy()
    rhs = np.asarray(rhs, dtype=float).copy()
    n = diag.size
    if lower.size != n - 1 or upper.size != n - 1 or rhs.size != n:
        raise ValueError("Invalid tridiagonal dimensions.")
    for i in range(1, n):
        fac = lower[i - 1] / diag[i - 1]
        diag[i] -= fac * upper[i - 1]
        rhs[i] -= fac * rhs[i - 1]
    out = np.empty(n, dtype=float)
    out[-1] = rhs[-1] / diag[-1]
    for i in range(n - 2, -1, -1):
        out[i] = (rhs[i] - upper[i] * out[i + 1]) / diag[i]
    return out


def solve_one_domain_profile(*, K: float, ny: int, params: WangExample61Parameters) -> tuple[np.ndarray, np.ndarray]:
    if int(ny) < 8:
        raise ValueError("ny must be at least 8.")
    y = np.linspace(float(params.y_bottom), float(params.y_top), int(ny) + 1, dtype=float)
    h = float(y[1] - y[0])
    a = float(params.mu) * _phi_1(y, K=float(K), params=params)
    b = (float(params.mu) / float(K)) * _phi_2(y, K=float(K), params=params)
    n = y.size
    lower = np.zeros(n - 1, dtype=float)
    diag = np.zeros(n, dtype=float)
    upper = np.zeros(n - 1, dtype=float)
    rhs = np.ones(n, dtype=float)

    # Bottom Neumann condition: u'(-1) = 0.
    diag[0] = 1.0
    upper[0] = -1.0
    rhs[0] = 0.0

    # Interior second-order conservative discretization of
    #   -(a(y) u')' + b(y) u = 1.
    for i in range(1, n - 1):
        a_plus = 0.5 * (a[i] + a[i + 1])
        a_minus = 0.5 * (a[i] + a[i - 1])
        lower[i - 1] = -a_minus / (h * h)
        diag[i] = (a_minus + a_plus) / (h * h) + b[i]
        upper[i] = -a_plus / (h * h)
        rhs[i] = 1.0

    # Top Dirichlet condition: u(1) = 0.
    diag[-1] = 1.0
    rhs[-1] = 0.0

    u = _solve_tridiagonal(lower, diag, upper, rhs)
    return y, u


def _rho_wang(err: float, K: float) -> float:
    if err <= 0.0 or K <= 0.0:
        return float("nan")
    return math.log(err) / math.log(K)


def _regional_error_metrics(
    *,
    y: np.ndarray,
    u_num: np.ndarray,
    K: float,
    params: WangExample61Parameters,
) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    u_num = np.asarray(u_num, dtype=float)
    u_ref = exact_two_domain_profile(y, K=float(K), params=params)
    du_num = np.gradient(u_num, y)
    du_ref = exact_two_domain_gradient(y, K=float(K), params=params)
    fluid = y >= 0.0
    porous = y < 0.0

    # Use trapz for compatibility with NumPy versions that do not provide np.trapezoid.
    l2_fluid = math.sqrt(float(np.trapz((u_num[fluid] - u_ref[fluid]) ** 2, x=y[fluid])))
    l2_porous = math.sqrt(float(np.trapz((u_num[porous] - u_ref[porous]) ** 2, x=y[porous])))
    h1_fluid = math.sqrt(float(np.trapz((du_num[fluid] - du_ref[fluid]) ** 2, x=y[fluid])))
    h1_porous = math.sqrt(float(np.trapz((du_num[porous] - du_ref[porous]) ** 2, x=y[porous])))
    i0 = int(np.argmin(np.abs(y)))
    return {
        "K": float(K),
        "epsilon": math.sqrt(float(K)),
        "ny": int(y.size - 1),
        "l2_fluid": l2_fluid,
        "rho_l2_fluid": _rho_wang(l2_fluid, float(K)),
        "l2_porous": l2_porous,
        "rho_l2_porous": _rho_wang(l2_porous, float(K)),
        "h1_fluid": h1_fluid,
        "rho_h1_fluid": _rho_wang(h1_fluid, float(K)),
        "h1_porous": h1_porous,
        "rho_h1_porous": _rho_wang(h1_porous, float(K)),
        "u_interface_one_domain": float(u_num[i0]),
        "u_interface_two_domain": float(u_ref[i0]),
        "u_interface_abs_error": abs(float(u_num[i0] - u_ref[i0])),
        "max_profile_abs_error": float(np.max(np.abs(u_num - u_ref))),
    }


def _write_summary_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "K",
        "epsilon",
        "ny",
        "l2_fluid",
        "rho_l2_fluid",
        "l2_porous",
        "rho_l2_porous",
        "h1_fluid",
        "rho_h1_fluid",
        "h1_porous",
        "rho_h1_porous",
        "u_interface_one_domain",
        "u_interface_two_domain",
        "u_interface_abs_error",
        "max_profile_abs_error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_profiles_csv(path: Path, *, y: np.ndarray, profiles: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["y", *profiles.keys()]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i, yy in enumerate(np.asarray(y, dtype=float)):
            writer.writerow([f"{yy:.16e}", *[f"{float(np.asarray(profiles[key])[i]):.16e}" for key in profiles]])


def _plot_profiles(
    path: Path,
    *,
    plot_ks: list[float],
    solutions: dict[float, tuple[np.ndarray, np.ndarray]],
    params: WangExample61Parameters,
    dpi: int,
) -> None:
    ncols = len(plot_ks)
    fig, axes = plt.subplots(1, ncols, figsize=(6.0 * ncols, 5.0), constrained_layout=True, squeeze=False)
    for ax, K in zip(axes[0], plot_ks):
        y, u_one = solutions[float(K)]
        u_two = exact_two_domain_profile(y, K=float(K), params=params)
        ax.plot(u_one, y, lw=2.2, color="#145A7B", label="one-domain transition layer")
        ax.plot(u_two, y, lw=2.0, color="#BA3F1D", linestyle="--", label="two-domain reference")
        ax.axhline(0.0, color="0.35", lw=1.0, linestyle=":")
        ax.set_title(f"$K={float(K):.0e}$")
        ax.set_xlabel(r"$u_x(y)$")
        ax.set_ylabel(r"$y$")
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best")
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def _plot_error_trends(path: Path, *, rows: list[dict[str, float]], dpi: int) -> None:
    K = np.asarray([row["K"] for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    axes[0].loglog(K, [row["l2_fluid"] for row in rows], "o-", lw=2.0, color="#145A7B", label=r"$L^2(\Omega_f)$")
    axes[0].loglog(K, [row["l2_porous"] for row in rows], "s-", lw=2.0, color="#2F8F5B", label=r"$L^2(\Omega_p)$")
    axes[0].set_xlabel(r"$K$")
    axes[0].set_ylabel("L2 error")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].loglog(K, [row["h1_fluid"] for row in rows], "o-", lw=2.0, color="#BA3F1D", label=r"$H^1(\Omega_f)$")
    axes[1].loglog(K, [row["h1_porous"] for row in rows], "s-", lw=2.0, color="#6A4C93", label=r"$H^1(\Omega_p)$")
    axes[1].set_xlabel(r"$K$")
    axes[1].set_ylabel("semi-H1 error")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(loc="best")

    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def run_benchmark(
    *,
    outdir: Path,
    k_list: list[float],
    ny: int,
    plot_ks: list[float],
    dpi: int,
    params: WangExample61Parameters | None = None,
) -> dict[str, object]:
    params = params or WangExample61Parameters()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    solutions: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    k_list = [float(K) for K in k_list]
    plot_ks = [float(K) for K in plot_ks]

    for K in k_list:
        y, u_one = solve_one_domain_profile(K=K, ny=int(ny), params=params)
        rows.append(_regional_error_metrics(y=y, u_num=u_one, K=K, params=params))
        solutions[K] = (y, u_one)
        if K in plot_ks:
            u_two = exact_two_domain_profile(y, K=K, params=params)
            tag = _format_k_tag(K)
            _write_profiles_csv(
                outdir / f"benchmark3_wang2014_layered_profile_{tag}.csv",
                y=y,
                profiles={
                    "one_domain": u_one,
                    "two_domain": u_two,
                    "abs_error": np.abs(u_one - u_two),
                },
            )

    rows.sort(key=lambda row: float(row["K"]), reverse=True)
    summary_csv = outdir / "benchmark3_wang2014_layered_summary.csv"
    _write_summary_csv(summary_csv, rows)
    _plot_profiles(
        outdir / "benchmark3_wang2014_layered_profiles.png",
        plot_ks=plot_ks,
        solutions=solutions,
        params=params,
        dpi=int(dpi),
    )
    _plot_error_trends(
        outdir / "benchmark3_wang2014_layered_error_trends.png",
        rows=rows,
        dpi=int(dpi),
    )

    summary = {
        "benchmark": "wang2014_example61_layered",
        "paper1_scope": "alpha-independent sharp-interface consistency benchmark",
        "params": asdict(params),
        "k_list": k_list,
        "ny": int(ny),
        "plot_ks": plot_ks,
        "rows": rows,
        "summary_csv": str(summary_csv),
        "profiles_png": str(outdir / "benchmark3_wang2014_layered_profiles.png"),
        "error_trends_png": str(outdir / "benchmark3_wang2014_layered_error_trends.png"),
    }
    (outdir / "benchmark3_wang2014_layered_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Paper 1 Benchmark 3: Wang2014-style layered one-domain vs two-domain comparison (alpha-independent reference layer)."
    )
    ap.add_argument("--outdir", type=str, default="examples/biofilms/benchmarks/wang/results/paper1_benchmark3_wang2014_layered")
    ap.add_argument("--k-list", type=str, default="1e-2,1e-3,1e-4,1e-5")
    ap.add_argument("--ny", type=int, default=4000)
    ap.add_argument("--plot-k", type=str, default="1e-2,1e-4")
    ap.add_argument("--png-dpi", type=int, default=220)
    ap.add_argument("--transition-sharpness", type=float, default=100.0)
    args = ap.parse_args()

    summary = run_benchmark(
        outdir=Path(args.outdir),
        k_list=_parse_float_list(args.k_list),
        ny=int(args.ny),
        plot_ks=_parse_float_list(args.plot_k),
        dpi=int(args.png_dpi),
        params=WangExample61Parameters(transition_sharpness=float(args.transition_sharpness)),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
