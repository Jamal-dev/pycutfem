#!/usr/bin/env python3
"""Paper-facing kappa=1e-4 interface-thickness progression plot.

This figure is intentionally narrower than the generic Benchmark 7 overlays:
it compares the one-domain curves only against the Seboldt moving-linear
reference, and it tracks how the profile/error evolve as the resolved physical
interface thickness decreases.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class CaseData:
    label: str
    outdir: Path
    eps_alpha: float
    eps_over_h: float
    h_char: float
    rmse_over_amp: float
    peak_relerr: float
    x: np.ndarray
    uy: np.ndarray


def _load_reference_curve(reference_csv: Path, *, kappa: float, curve_label: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    with reference_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if float(row["kappa"]) != float(kappa):
                continue
            if row["curve_label"] != curve_label:
                continue
            xs.append(float(row["x"]))
            ys.append(float(row["eta_y"]))
    if not xs:
        raise ValueError(f"No reference data for kappa={kappa} curve={curve_label}.")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


def _load_case(case_dir: Path, *, label: str) -> CaseData:
    summary_path = case_dir / "benchmark7_summary.csv"
    profile_path = case_dir / "kappa_1e-4" / "profile_final.csv"
    with summary_path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    if float(row["solve_completed"]) != 1.0:
        raise ValueError(f"{case_dir} is not a completed case.")
    prof = np.loadtxt(profile_path, delimiter=",", skiprows=1)
    return CaseData(
        label=label,
        outdir=case_dir,
        eps_alpha=float(row["eps_alpha"]),
        eps_over_h=float(row["eps_alpha_over_h"]),
        h_char=float(row["h_char"]),
        rmse_over_amp=float(row["rmse_over_amp_moving_linear"]),
        peak_relerr=float(row["peak_amp_relerr_moving_linear"]),
        x=prof[:, 0],
        uy=prof[:, 1],
    )


def _parse_case(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("Each --case must look like label=path/to/outdir")
    label, path = raw.split("=", 1)
    return label.strip(), Path(path).resolve()


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reference-csv",
        type=Path,
        default=here / "reference_profiles_fig6.csv",
        help="Digitized Seboldt Figure 6 reference CSV.",
    )
    ap.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case spec of the form label=/abs/or/rel/outdir. Repeat for each completed case.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("out/benchmark7_kappa14_interface_progression/benchmark7_kappa14_interface_progression.png"),
    )
    args = ap.parse_args()

    if not args.case:
        args.case = [
            "mid-resolved=out/benchmark7_kappa14_mid_dt5e4_epsh0p6",
            "nominal-resolved=out/benchmark7_kappa14_nominal_dt5e4_epsh0p6",
        ]

    cases = [_load_case(path, label=label) for label, path in (_parse_case(raw) for raw in args.case)]
    cases.sort(key=lambda c: c.eps_alpha, reverse=True)
    x_ref, y_ref = _load_reference_curve(args.reference_csv.resolve(), kappa=1.0e-4, curve_label="partitioned_moving_linear")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), constrained_layout=True)

    ax = axes[0]
    ax.plot(x_ref, y_ref, color="#149dff", lw=2.6, label="Seboldt moving linear")
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(cases)))
    for color, case in zip(colors, cases):
        ax.plot(
            case.x,
            case.uy,
            color=color,
            lw=2.2,
            label=(
                f"{case.label}: eps={case.eps_alpha:.3f}, "
                f"eps/h={case.eps_over_h:.2f}, h={case.h_char:.4f}"
            ),
        )
    ax.set_title(r"$\kappa = 10^{-4} I$: one-domain progression toward sharp-interface target")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$u_y(x, y=1.25)$")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)

    ax = axes[1]
    eps = np.asarray([c.eps_alpha for c in cases], dtype=float)
    rmse = np.asarray([c.rmse_over_amp for c in cases], dtype=float)
    peak = np.asarray([c.peak_relerr for c in cases], dtype=float)
    ax.plot(eps, rmse, marker="o", ms=7, lw=2.0, color="tab:green", label="RMSE / amplitude")
    ax.plot(eps, peak, marker="s", ms=6.5, lw=2.0, color="tab:orange", label="peak amplitude rel. error")
    ax.invert_xaxis()
    ax.set_xlabel(r"physical interface thickness $\varepsilon_\alpha$")
    ax.set_ylabel("normalized error")
    ax.set_title("Error trend as the resolved interface is thinned")
    ax.grid(alpha=0.2)
    for case in cases:
        ax.annotate(case.label, (case.eps_alpha, case.rmse_over_amp), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
    ax.legend(fontsize=8)

    fig.savefig(args.out, dpi=220)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
