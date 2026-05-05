"""
Reproduce Duddu et al. (2007) Fig. 4 + Table I (1D slab growth).

This driver runs:
  - XFEM100-like mesh (discontinuous-derivative / shifted-|phi| enrichment)
  - XFEM200-like mesh
  - FD3000 reference (steady 1D BVP)

and produces:
  - results/.../summary.csv : computed speeds + paper values
  - results/.../fig4_compare.png : S(y) and Phi(y) profile comparison

All outputs stay under examples/biofilms/benchmarks/dadu/results/ as requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


PAPER_TABLE_I = {
    "FD500": 0.0038,
    "FD2000": 0.0091,
    "FD2500": 0.0099,
    "FD3000": 0.0103,
    "XFEM100": 0.0093,
    "XFEM200": 0.0097,
}


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _read_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    return np.asarray(data[:, 0], dtype=float), np.asarray(data[:, 1], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        type=str,
        default="examples/biofilms/benchmarks/dadu/results/duddu2007_fig4_table1",
    )
    ap.add_argument("--backend", choices=("cpp",), default="cpp")
    ap.add_argument("--linear-solver", choices=("petsc",), default="petsc")
    ap.add_argument("--q", type=int, default=6)
    ap.add_argument("--Y-wO", type=float, default=0.215)
    args = ap.parse_args()

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).resolve().parent
    xfem_script = here / "duddu2007_growth_1d_fig4.py"
    fd_script = here / "duddu2007_growth_1d_fd_table1.py"
    od_script = here / "duddu2007_one_domain_slab_speed.py"

    # Mesh settings chosen to match the paper's Table I values closely while
    # keeping the domain exactly H=0.3mm, h_b=0.2mm. "XFEM100/200" in Duddu (2007)
    # correspond to ~100/200 line elements in the biofilm thickness.
    runs = [
        {
            "key": "XFEM100",
            "label": "XFEM100",
            "cmd": [
                sys.executable,
                str(xfem_script),
                "--backend",
                str(args.backend),
                "--linear-solver",
                str(args.linear_solver),
                "--q",
                str(int(args.q)),
                "--nx",
                "6",
                "--ny",
                "151",
                "--enrichment",
                "abs",
                "--Y-wO",
                str(float(args.Y_wO)),
                "--outdir",
                str(outdir / "xfem100"),
            ],
            "summary": outdir / "xfem100" / "summary.json",
            "profiles": {
                "S": outdir / "xfem100" / "profile_S.txt",
                "Phi": outdir / "xfem100" / "profile_Phi.txt",
            },
        },
        {
            "key": "XFEM200",
            "label": "XFEM200",
            "cmd": [
                sys.executable,
                str(xfem_script),
                "--backend",
                str(args.backend),
                "--linear-solver",
                str(args.linear_solver),
                "--q",
                str(int(args.q)),
                "--nx",
                "6",
                "--ny",
                "301",
                "--enrichment",
                "abs",
                "--Y-wO",
                str(float(args.Y_wO)),
                "--outdir",
                str(outdir / "xfem200"),
            ],
            "summary": outdir / "xfem200" / "summary.json",
            "profiles": {
                "S": outdir / "xfem200" / "profile_S.txt",
                "Phi": outdir / "xfem200" / "profile_Phi.txt",
            },
        },
        {
            "key": "FD3000",
            "label": "FD3000",
            "cmd": [
                sys.executable,
                str(fd_script),
                "--N",
                "3000",
                "--Y-wO",
                str(float(args.Y_wO)),
                "--outdir",
                str(outdir / "fd3000"),
            ],
            "summary": outdir / "fd3000" / "summary.json",
            "profiles": {
                "S": outdir / "fd3000" / "profile_S.txt",
                "Phi": outdir / "fd3000" / "profile_Phi.txt",
            },
        },
        {
            "key": "one-domain",
            "label": "one-domain",
            "cmd": [
                sys.executable,
                str(od_script),
                "--backend",
                str(args.backend),
                "--linear-solver",
                str(args.linear_solver),
                "--q",
                str(int(args.q)),
                "--nx",
                "4",
                "--ny",
                "240",
                "--eps-alpha",
                "0.01",
                "--phi-b",
                "0.3",
                "--D-S",
                "138.5",
                "--kappa-inv",
                "8.0",
                "--gamma-vS",
                "0.1",
                "--vS-ext-mode",
                "l2",
                "--gamma-vS-pin",
                "0.0",
                "--gamma-p-out-power",
                "100",
                "--outdir",
                str(outdir / "one_domain"),
            ],
            "summary": outdir / "one_domain" / "summary.json",
            "profiles": {
                "S": outdir / "one_domain" / "profile_S.txt",
                "Phi": outdir / "one_domain" / "profile_Phi.txt",
            },
        },
    ]

    for r in runs:
        _run([str(c) for c in r["cmd"]])

    # Collect speed summaries
    rows: list[dict[str, object]] = []
    for r in runs:
        js = json.loads(Path(r["summary"]).read_text())
        if r["key"].startswith("XFEM"):
            F = float(js["F_duddu_mm_per_day"])
        elif r["key"] == "one-domain":
            F = float(js["F_vS_mm_per_day"])
        else:
            F = float(js["F_mm_per_day"])
        rows.append(
            {
                "method": r["label"],
                "F_mm_per_day": F,
                "paper_F_mm_per_day": PAPER_TABLE_I.get(r["label"], ""),
            }
        )

    with (outdir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Plot profiles
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        styles = {
            "XFEM100": dict(color="k", linestyle="-", linewidth=1.5),
            "XFEM200": dict(color="0.3", linestyle="--", linewidth=1.5),
            "FD3000": dict(color="0.6", linestyle=":", linewidth=1.5),
            "one-domain": dict(color="tab:orange", linestyle="-.", linewidth=1.7),
        }

        for r in runs:
            yS, S = _read_profile(Path(r["profiles"]["S"]))
            yP, P = _read_profile(Path(r["profiles"]["Phi"]))
            st = styles.get(r["label"], {})
            ax[0].plot(yS, S, label=r["label"], **st)
            ax[1].plot(yP, P, label=r["label"], **st)

        ax[0].set_xlabel("y (mm)")
        ax[0].set_ylabel("S (mgO2/mm^3)")
        ax[0].set_title("Substrate S(y)")
        ax[0].grid(True, alpha=0.3)

        ax[1].set_xlabel("y (mm)")
        ax[1].set_ylabel("Phi")
        ax[1].set_title("Velocity potential Φ(y)")
        ax[1].grid(True, alpha=0.3)

        for a in ax:
            a.legend(frameon=False, fontsize=9)

        fig.suptitle("Duddu et al. (2007) | Fig. 4 (1D slab) reproduction", fontsize=12)
        outpng = outdir / "fig4_compare.png"
        fig.savefig(outpng, dpi=200)
        plt.close(fig)
        print(f"- Wrote {outpng}")
    except Exception as e:
        print(f"[warn] plotting skipped: {e}")

    print(f"- Wrote {outdir/'summary.csv'}")


if __name__ == "__main__":
    main()
