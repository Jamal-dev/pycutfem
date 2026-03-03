"""
Reproduce Duddu et al. (2007) Fig. 5 (Example 1): 2D growth of one semi-circular colony.

This driver runs:
  (A) XFEM + level set (sharp interface), and
  (B) one-domain diffuse-interface surrogate (growth-only limit),
then produces:
  - VTK snapshots for both runs (ParaView),
  - color PNG panels (interface + S + Phi/p),
  - y_top(t) comparison CSV + plot at the paper target times.

All outputs stay under examples/biofilms/benchmarks/dadu/results/ as requested.

Run (recommended)
----------------
conda run --no-capture-output -n fenicsx python \
  examples/biofilms/benchmarks/dadu/reproduce_duddu2007_fig5_example1.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


_TARGETS_EX1 = ",".join(
    str(v)
    for v in [
        0.0,
        1.0,
        2.3,
        3.7,
        5.3,
        7.0,
        8.9,
        10.8,
        12.8,
        14.7,
        16.6,
        18.5,
        20.4,
        21.9,
        23.7,
        25.6,
        27.6,
        29.4,
        31.3,
        33.2,
        35.0,
        36.8,
        38.5,
        40.2,
        42.1,
        43.7,
        44.5,
    ]
)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        type=str,
        default="examples/biofilms/benchmarks/dadu/results/duddu2007_fig5_example1",
        help="Output directory for the combined reproduction bundle.",
    )
    ap.add_argument("--backend", choices=("cpp",), default="cpp")
    ap.add_argument("--linear-solver", choices=("petsc", "scipy"), default="petsc")

    # Geometry/mesh choices. Duddu (2007) uses 100x100 XFEM triangles and a 200x200 FD level set grid.
    # For faster local runs, use something like --xfem-mesh 60 --ls-grid 120.
    ap.add_argument("--xfem-mesh", type=int, default=60, help="XFEM background mesh quads in x/y (triangles are 2*nx*ny).")
    ap.add_argument("--ls-grid", type=int, default=120, help="Level-set FD grid cells in x/y.")

    # One-domain mesh (Q2 geometry, Q1 scalars). Default aligns with the paper-ish XFEM mesh above.
    ap.add_argument("--one-domain-n", type=int, default=60, help="One-domain quad divisions in x/y.")
    ap.add_argument("--dt", type=float, default=0.2, help="One-domain growth step (days).")
    ap.add_argument("--q", type=int, default=4, help="Quadrature order for both solvers (if supported).")
    ap.add_argument(
        "--one-domain-include-fluid",
        action="store_true",
        help=(
            "Solve the fluid velocity along with (p,vS) in the one-domain step (2). "
            "This avoids the need for an outflow in B vS when the biofilm is localized."
        ),
    )
    ap.add_argument(
        "--eps-alpha",
        type=float,
        default=0.004,
        help="One-domain diffuse interface half-thickness (mm). For r=0.01, values ~0.003–0.006 are typical.",
    )
    ap.add_argument(
        "--alpha-advect-with",
        type=str,
        default="vS",
        choices=("vS", "v", "mix", "mix_biofilm"),
        help="One-domain alpha advection velocity choice.",
    )
    ap.add_argument(
        "--alpha-mix-gate-alpha0",
        type=float,
        default=0.01,
        help="(mix_biofilm) Gate cutoff alpha0 in g(alpha)=alpha^m/(alpha^m+alpha0^m).",
    )
    ap.add_argument(
        "--s-v-mode",
        type=str,
        default="divU",
        choices=("auto", "divU", "BdivU"),
        help="One-domain volume source choice (matches Duddu(2007) when paired with --alpha-advect-with).",
    )
    ap.add_argument("--D-S", type=float, default=138.5, help="One-domain substrate diffusion coefficient (mm^2/day).")
    ap.add_argument("--gamma-vS", type=float, default=0.1, help="One-domain vS extension penalty.")
    ap.add_argument("--vS-ext-mode", type=str, default="l2", choices=("l2", "grad"), help="One-domain vS extension mode.")
    ap.add_argument("--gamma-vS-pin", type=float, default=0.0, help="One-domain vS pin penalty in the pure fluid.")

    # Problem setup
    ap.add_argument("--t-final", type=float, default=44.5)
    ap.add_argument("--center-x", type=float, default=0.25)
    ap.add_argument("--radius", type=float, default=0.01)
    args = ap.parse_args()

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).resolve().parent
    xfem_script = here / "duddu2007_growth_2d_fig5_example1.py"
    one_dom_script = here / "duddu2007_one_domain_growth_2d_fig6_example2.py"
    plot_iface_script = here / "plot_one_domain_interface_from_snaps.py"
    plot_panels_script = here / "plot_one_domain_fig6_panels_from_npz.py"
    compare_y_top_script = here / "compare_duddu2007_fig6_y_top.py"

    xfem_dir = outdir / "xfem"
    od_dir = outdir / "one_domain"

    # --- XFEM run ----------------------------------------------------------
    _run(
        [
            sys.executable,
            str(xfem_script),
            "--outdir",
            str(xfem_dir),
            "--backend",
            str(args.backend),
            "--linear-solver",
            str(args.linear_solver),
            "--q",
            str(int(args.q)),
            "--t-final",
            str(float(args.t_final)),
            "--center-x",
            str(float(args.center_x)),
            "--radius",
            str(float(args.radius)),
            "--mesh-nx",
            str(int(args.xfem_mesh)),
            "--mesh-ny",
            str(int(args.xfem_mesh)),
            "--grid-nx",
            str(int(args.ls_grid)),
            "--grid-ny",
            str(int(args.ls_grid)),
        ]
    )

    # --- one-domain run ----------------------------------------------------
    od_cmd = [
        sys.executable,
        str(one_dom_script),
        "--outdir",
        str(od_dir),
        "--backend",
        str(args.backend),
        "--linear-solver",
        str(args.linear_solver),
        "--q",
        str(int(args.q)),
        "--t-final",
        str(float(args.t_final)),
        "--dt",
        str(float(args.dt)),
        "--nx",
        str(int(args.one_domain_n)),
        "--ny",
        str(int(args.one_domain_n)),
        "--centers-x",
        str(float(args.center_x)),
        "--radii",
        str(float(args.radius)),
        "--eps-alpha",
        str(float(args.eps_alpha)),
        "--phi-update",
        "mix",
        "--alpha-advect-with",
        str(args.alpha_advect_with),
        "--s-v-mode",
        str(args.s_v_mode),
        "--D-S",
        str(float(args.D_S)),
        "--gamma-vS",
        str(float(args.gamma_vS)),
        "--vS-ext-mode",
        str(args.vS_ext_mode),
        "--gamma-vS-pin",
        str(float(args.gamma_vS_pin)),
        "--substrate-solver",
        "newton",
        "--substrate-advection",
        "off",
        "--newton-tol",
        "1e-8",
        "--max-it",
        "25",
        "--targets",
        _TARGETS_EX1,
        "--vtk-full",
        "--skip-plots",
    ]
    if str(args.alpha_advect_with).strip().lower() == "mix_biofilm":
        od_cmd += ["--alpha-mix-gate-alpha0", str(float(args.alpha_mix_gate_alpha0))]
    if bool(args.one_domain_include_fluid):
        od_cmd.insert(od_cmd.index("--centers-x"), "--include-fluid")
    _run(od_cmd)

    # Color PNGs for one-domain (interface + final fields)
    _run(
        [
            sys.executable,
            str(plot_iface_script),
            "--results-dir",
            str(od_dir),
            "--targets",
            _TARGETS_EX1,
            "--color-by-time",
            "--cmap",
            "viridis",
            "--out",
            str(od_dir / "fig5a_interface.png"),
        ]
    )
    _run(
        [
            sys.executable,
            str(plot_panels_script),
            "--results-dir",
            str(od_dir),
            "--cmap-S",
            "viridis",
            "--cmap-Phi",
            "RdBu_r",
        ]
    )
    # Copy to Fig.5-like filenames for convenience (keep originals too).
    for src, dst in [
        (od_dir / "fig6b_S.png", od_dir / "fig5b_S.png"),
        (od_dir / "fig6c_Phi.png", od_dir / "fig5c_Phi.png"),
    ]:
        if src.exists():
            shutil.copyfile(src, dst)

    # y_top(t) comparison at paper target times
    _run(
        [
            sys.executable,
            str(compare_y_top_script),
            "--a",
            str(xfem_dir),
            "--b",
            str(od_dir),
            "--label-a",
            "XFEM",
            "--label-b",
            "one-domain",
            "--targets",
            _TARGETS_EX1,
            "--outdir",
            str(outdir),
        ]
    )

    # Convenience: copy key panels to the top-level outdir
    for src in [
        xfem_dir / "fig5a_interface.png",
        xfem_dir / "fig5b_S.png",
        xfem_dir / "fig5c_Phi.png",
        od_dir / "fig5a_interface.png",
        od_dir / "fig5b_S.png",
        od_dir / "fig5c_Phi.png",
        outdir / "y_top_compare.png",
    ]:
        if src.exists():
            shutil.copyfile(src, outdir / src.name)

    print("- Wrote", outdir)


if __name__ == "__main__":
    main()
