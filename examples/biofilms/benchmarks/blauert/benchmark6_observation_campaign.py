#!/usr/bin/env python3
"""Run the observation-based Benchmark 6 campaign for the Blauert/Dian case.

This script is intentionally explicit. It collects the concrete runs needed to
freeze Benchmark 6 for the paper:

  - steady Dian-style contour calibration on the traced experimental geometry,
  - steady mesh ladder on the same observation block,
  - dynamic 0.8 Pa Blauert transient calibration and mesh ladder,
  - optional exploratory 1.64 Pa transient run.

The wrapper script still performs its own internal calibration/selection; this
campaign file just fixes the scenarios, local refinement settings, time windows,
and output locations so the whole Benchmark 6 program can be rerun
reproducibly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys


_REPO_ROOT = Path(__file__).resolve().parents[4]
_WRAPPER = _REPO_ROOT / "examples/biofilms/benchmarks/blauert/paper1_benchmark6_blauert_channel.py"


@dataclass(frozen=True)
class CampaignCase:
    name: str
    description: str
    args: list[str]


def _build_cases(
    run_root: Path,
    *,
    backend: str,
    kinematics_scale: float,
    v_supg: float,
    v_supg_mode: str,
    v_supg_c_nu: float,
    u_supg: float,
    u_cip: float,
    u_cip_weight: str,
    v_cip: float,
    vS_cip: float,
    gamma_div: float,
    adaptive_gamma_div: bool,
    gamma_div_max: float,
    alpha_ch_eps: float,
    scale_alpha_ch_eps_with_zeta: bool,
    diffuse_shear_scale_ref: float,
    vtk_every: int,
) -> list[CampaignCase]:
    steady_common = [
        "--observation-scenarios",
        "steady_dian",
        "--nonlinear-solver",
        "pdas",
        "--ls-mode",
        "dealii",
        "--backend",
        str(backend),
        "--gamma-u",
        "5.0",
        "--u-extension",
        "l2",
        "--gamma-u-pin",
        "1e-4",
        "--kinematics-scale",
        str(float(kinematics_scale)),
        "--v-supg",
        str(float(v_supg)),
        "--v-supg-mode",
        str(v_supg_mode),
        "--v-supg-c-nu",
        str(float(v_supg_c_nu)),
        "--u-supg",
        str(float(u_supg)),
        "--u-cip",
        str(float(u_cip)),
        "--u-cip-weight",
        str(u_cip_weight),
        "--v-cip",
        str(float(v_cip)),
        "--vS-cip",
        str(float(vS_cip)),
        "--rho-f",
        "0",
        "--phi-b",
        "0.47",
        "--alpha-ch-eps",
        str(float(alpha_ch_eps)),
        "--diffuse-shear-scale-ref",
        str(float(diffuse_shear_scale_ref)),
        "--gamma-div",
        str(float(gamma_div)),
        "--gamma-div-max",
        str(float(gamma_div_max)),
        "--q",
        "4",
        "--dt",
        "0.005",
        "--t-final",
        "1.0",
        "--steady-time",
        "1.0",
        "--t-ramp",
        "0.2",
        "--max-it",
        "25",
        "--refine-biofilm",
        "--refine-band",
        "2.5e-4",
        "--refine-expand-layers",
        "1",
        "--skip-existing",
        "--vtk-every",
        str(int(vtk_every)),
    ]
    steady_common.append("--adaptive-gamma-div" if bool(adaptive_gamma_div) else "--no-adaptive-gamma-div")
    steady_common.append(
        "--scale-alpha-ch-eps-with-zeta" if bool(scale_alpha_ch_eps_with_zeta) else "--no-scale-alpha-ch-eps-with-zeta"
    )
    dynamic08_common = [
        "--observation-scenarios",
        "dynamic_08pa",
        "--nonlinear-solver",
        "pdas",
        "--ls-mode",
        "dealii",
        "--backend",
        str(backend),
        "--gamma-u",
        "5.0",
        "--u-extension",
        "l2",
        "--gamma-u-pin",
        "1e-4",
        "--kinematics-scale",
        str(float(kinematics_scale)),
        "--v-supg",
        str(float(v_supg)),
        "--v-supg-mode",
        str(v_supg_mode),
        "--v-supg-c-nu",
        str(float(v_supg_c_nu)),
        "--u-supg",
        str(float(u_supg)),
        "--u-cip",
        str(float(u_cip)),
        "--u-cip-weight",
        str(u_cip_weight),
        "--v-cip",
        str(float(v_cip)),
        "--vS-cip",
        str(float(vS_cip)),
        "--rho-f",
        "1000",
        "--u-avg",
        "0.1777777778",
        "--phi-b",
        "0.47",
        "--alpha-ch-eps",
        str(float(alpha_ch_eps)),
        "--diffuse-shear-scale-ref",
        str(float(diffuse_shear_scale_ref)),
        "--gamma-div",
        str(float(gamma_div)),
        "--gamma-div-max",
        str(float(gamma_div_max)),
        "--q",
        "4",
        "--dt",
        "0.025",
        "--t-final",
        "10.0",
        "--t-ramp",
        "0.2",
        "--max-it",
        "25",
        "--refine-biofilm",
        "--refine-band",
        "2.5e-4",
        "--refine-expand-layers",
        "1",
        "--skip-existing",
        "--vtk-every",
        str(int(vtk_every)),
    ]
    dynamic08_common.append("--adaptive-gamma-div" if bool(adaptive_gamma_div) else "--no-adaptive-gamma-div")
    dynamic08_common.append(
        "--scale-alpha-ch-eps-with-zeta" if bool(scale_alpha_ch_eps_with_zeta) else "--no-scale-alpha-ch-eps-with-zeta"
    )
    dynamic164_common = [
        "--observation-scenarios",
        "dynamic_164pa",
        "--nonlinear-solver",
        "pdas",
        "--ls-mode",
        "dealii",
        "--backend",
        str(backend),
        "--gamma-u",
        "5.0",
        "--u-extension",
        "l2",
        "--gamma-u-pin",
        "1e-4",
        "--kinematics-scale",
        str(float(kinematics_scale)),
        "--v-supg",
        str(float(v_supg)),
        "--v-supg-mode",
        str(v_supg_mode),
        "--v-supg-c-nu",
        str(float(v_supg_c_nu)),
        "--u-supg",
        str(float(u_supg)),
        "--u-cip",
        str(float(u_cip)),
        "--u-cip-weight",
        str(u_cip_weight),
        "--v-cip",
        str(float(v_cip)),
        "--vS-cip",
        str(float(vS_cip)),
        "--rho-f",
        "1000",
        "--u-avg",
        "0.3644444444",
        "--phi-b",
        "0.66",
        "--alpha-ch-eps",
        str(float(alpha_ch_eps)),
        "--diffuse-shear-scale-ref",
        str(float(diffuse_shear_scale_ref)),
        "--gamma-div",
        str(float(gamma_div)),
        "--gamma-div-max",
        str(float(gamma_div_max)),
        "--q",
        "4",
        "--dt",
        "0.01",
        "--t-final",
        "2.2",
        "--steady-time",
        "2.1",
        "--t-ramp",
        "0.1",
        "--max-it",
        "25",
        "--refine-biofilm",
        "--refine-band",
        "2.5e-4",
        "--refine-expand-layers",
        "1",
        "--skip-existing",
        "--vtk-every",
        str(int(vtk_every)),
    ]
    dynamic164_common.append("--adaptive-gamma-div" if bool(adaptive_gamma_div) else "--no-adaptive-gamma-div")
    dynamic164_common.append(
        "--scale-alpha-ch-eps-with-zeta" if bool(scale_alpha_ch_eps_with_zeta) else "--no-scale-alpha-ch-eps-with-zeta"
    )
    return [
        CampaignCase(
            name="steady_calibration_refined",
            description="Steady Dian-style calibration with local refinement and hanging nodes.",
            args=
            [
                "--profile",
                "baseline",
                "--calibration-only",
                "--nx-list",
                "16,24",
                "--E-list",
                "120,200,320,500",
                "--diffuse-shear-scale-list",
                "0,30,50",
                "--outdir",
                str(run_root / "steady_calibration_refined"),
            ]
            + steady_common,
        ),
        CampaignCase(
            name="steady_mesh_refined",
            description="Steady mesh ladder for the traced contour benchmark.",
            args=
            [
                "--profile",
                "production",
                "--nx-list",
                "16,24,32",
                "--E-list",
                "120,200,320,500",
                "--diffuse-shear-scale-list",
                "0,30,50",
                "--outdir",
                str(run_root / "steady_mesh_refined"),
            ]
            + steady_common,
        ),
        CampaignCase(
            name="dynamic08_calibration_refined",
            description="Transient 0.8 Pa Blauert calibration on the patchy geometry.",
            args=
            [
                "--profile",
                "baseline",
                "--calibration-only",
                "--nx-list",
                "16",
                "--E-list",
                "120,200,320,500",
                "--diffuse-shear-scale-list",
                "0,30,50",
                "--outdir",
                str(run_root / "dynamic08_calibration_refined"),
            ]
            + dynamic08_common,
        ),
        CampaignCase(
            name="dynamic08_mesh_refined",
            description="Transient 0.8 Pa mesh ladder after calibration.",
            args=
            [
                "--profile",
                "production",
                "--nx-list",
                "16,24",
                "--E-list",
                "120,200,320,500",
                "--diffuse-shear-scale-list",
                "0,30,50",
                "--outdir",
                str(run_root / "dynamic08_mesh_refined"),
            ]
            + dynamic08_common,
        ),
        CampaignCase(
            name="dynamic164_exploratory_refined",
            description="Exploratory transient 1.64 Pa run on the attached-dynamic observation block.",
            args=
            [
                "--profile",
                "baseline",
                "--calibration-only",
                "--nx-list",
                "16",
                "--E-list",
                "200,320,500",
                "--diffuse-shear-scale-list",
                "0,30,50",
                "--outdir",
                str(run_root / "dynamic164_exploratory_refined"),
            ]
            + dynamic164_common,
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-root",
        type=str,
        default="examples/biofilms/results/benchmark6_observation_campaign",
        help="Directory under which the campaign case output folders are created.",
    )
    ap.add_argument(
        "--cases",
        type=str,
        default="steady_calibration_refined,steady_mesh_refined,dynamic08_calibration_refined,dynamic08_mesh_refined,dynamic164_exploratory_refined",
        help="Comma-separated case names to run.",
    )
    ap.add_argument(
        "--stream",
        action="store_true",
        help="Stream child wrapper/simulation stdout to the terminal while still writing per-case log files.",
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later campaign cases if one selected case exits with an error.",
    )
    ap.add_argument(
        "--restart-from",
        type=str,
        default="",
        help="Optional checkpoint (.npz) to pass through to the wrapper for resume/debug runs.",
    )
    ap.add_argument(
        "--restart-dt",
        type=float,
        default=float("nan"),
        help="Optional dt override to pass through with --restart-from.",
    )
    ap.add_argument(
        "--restart-write-every",
        type=int,
        default=1,
        help="Checkpoint write frequency forwarded to the wrapper.",
    )
    ap.add_argument(
        "--gamma-div",
        type=float,
        default=5.0e-2,
        help="Consistent mixed-block grad-div / augmented-Lagrangian strength forwarded to the wrapper.",
    )
    ap.add_argument(
        "--backend",
        type=str,
        default="cpp",
        choices=("python", "jit", "cpp"),
        help="Assembly backend forwarded to the wrapper.",
    )
    ap.add_argument(
        "--v-supg",
        type=float,
        default=0.0,
        help="Fluid-momentum SUPG-like streamline diffusion forwarded to the wrapper.",
    )
    ap.add_argument(
        "--v-supg-mode",
        type=str,
        default="streamline",
        choices=("streamline", "residual"),
        help="Fluid momentum stabilization form forwarded to the wrapper.",
    )
    ap.add_argument(
        "--v-supg-c-nu",
        type=float,
        default=4.0,
        help="Viscous constant c_nu in the Green's-function elemental tau forwarded to the wrapper.",
    )
    ap.add_argument(
        "--kinematics-scale",
        type=float,
        default=float("nan"),
        help="Scaling applied to the kinematic constraint, forwarded to the wrapper.",
    )
    ap.add_argument(
        "--u-supg",
        type=float,
        default=0.0,
        help="SUPG strength for the kinematic u-transport equation, forwarded to the wrapper.",
    )
    ap.add_argument(
        "--u-cip",
        type=float,
        default=0.0,
        help="CIP strength for the kinematic u-transport equation, forwarded to the wrapper.",
    )
    ap.add_argument(
        "--u-cip-weight",
        type=str,
        default="biofilm",
        choices=("fluid", "biofilm", "both"),
        help="Localization used by --u-cip, forwarded to the wrapper.",
    )
    ap.add_argument(
        "--v-cip",
        type=float,
        default=0.0,
        help="Continuous-interior-penalty stabilization strength forwarded to the wrapper.",
    )
    ap.add_argument(
        "--vS-cip",
        type=float,
        default=0.0,
        help="Continuous-interior-penalty stabilization strength for skeleton velocity, forwarded to the wrapper.",
    )
    ap.add_argument(
        "--adaptive-gamma-div",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward the Benchmark 6 adaptive gamma_div controller to the wrapper.",
    )
    ap.add_argument(
        "--gamma-div-max",
        type=float,
        default=1.0e-1,
        help="Upper cap forwarded to the wrapper's adaptive gamma_div controller.",
    )
    ap.add_argument("--alpha-ch-eps", type=float, default=2.0e-5, help="Forwarded baseline alpha CH eps.")
    ap.add_argument(
        "--scale-alpha-ch-eps-with-zeta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward the zeta^2-based alpha CH eps scaling rule to the wrapper.",
    )
    ap.add_argument(
        "--diffuse-shear-scale-ref",
        type=float,
        default=50.0,
        help="Reference zeta used when scaling alpha CH eps with zeta.",
    )
    ap.add_argument(
        "--vtk-every",
        type=int,
        default=-1,
        help=(
            "VTK write cadence forwarded to the wrapper. "
            "Negative keeps the wrapper defaults (calibration off, finest production mesh every 10 steps)."
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_root = (_REPO_ROOT / str(args.run_root)).resolve()
    selected = {part.strip() for part in str(args.cases).split(",") if part.strip()}
    all_cases = _build_cases(
        run_root,
        backend=str(args.backend),
        kinematics_scale=float(args.kinematics_scale),
        v_supg=float(args.v_supg),
        v_supg_mode=str(args.v_supg_mode),
        v_supg_c_nu=float(args.v_supg_c_nu),
        u_supg=float(args.u_supg),
        u_cip=float(args.u_cip),
        u_cip_weight=str(args.u_cip_weight),
        v_cip=float(args.v_cip),
        vS_cip=float(args.vS_cip),
        gamma_div=float(args.gamma_div),
        adaptive_gamma_div=bool(args.adaptive_gamma_div),
        gamma_div_max=float(args.gamma_div_max),
        alpha_ch_eps=float(args.alpha_ch_eps),
        scale_alpha_ch_eps_with_zeta=bool(args.scale_alpha_ch_eps_with_zeta),
        diffuse_shear_scale_ref=float(args.diffuse_shear_scale_ref),
        vtk_every=int(args.vtk_every),
    )
    chosen = [case for case in all_cases if case.name in selected]
    missing = sorted(selected - {case.name for case in all_cases})
    if missing:
        raise ValueError(f"Unknown case names: {', '.join(missing)}")
    if not chosen:
        raise ValueError("No campaign cases selected.")

    for case in chosen:
        cmd = [sys.executable, "-u", str(_WRAPPER)] + list(case.args)
        cmd.extend(["--restart-write-every", str(int(args.restart_write_every))])
        if str(args.restart_from).strip():
            cmd.extend(["--restart-from", str(args.restart_from).strip()])
        if args.restart_dt == args.restart_dt:
            cmd.extend(["--restart-dt", str(float(args.restart_dt))])
        if bool(args.stream):
            cmd.append("--stream-subprocess")
        print(f"\n[{case.name}] {case.description}")
        print("+ " + shlex.join(cmd))
        if bool(args.dry_run):
            continue
        try:
            subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)
        except subprocess.CalledProcessError:
            if not bool(args.continue_on_error):
                raise
            print(f"[warn] campaign case failed and was skipped: {case.name}")


if __name__ == "__main__":
    main()
