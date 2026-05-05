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
    vi_enter_tol: float,
    vi_leave_tol: float,
    vi_persistence: int,
    vi_lambda0: float,
    vi_lambda_max: float,
    vi_lambda_growth: float,
    vi_lambda_decay: float,
    vi_active_soft_threshold: int,
    vi_active_soft_alpha: float,
    vi_active_strong_factor: float,
    vi_filter_max_delta_active: int,
    vi_unconstrained_lm: bool,
    vi_lm_lambda0: float,
    vi_lm_lambda_max: float,
    vi_lm_growth: float,
    vi_lm_decay: float,
    vi_lm_accept_ratio: float,
    vi_lm_good_ratio: float,
    vi_lm_max_tries: int,
    linear_ksp_type: str,
    linear_pc_type: str,
    linear_pc_factor_solver_type: str,
    linear_ksp_rtol: float,
    linear_ksp_max_it: int,
    linear_ksp_trace: bool,
    linear_schur: bool,
    linear_schur_pressure_field: str,
    linear_schur_fact: str,
    linear_schur_pre: str,
    linear_schur_rest_ksp: str,
    linear_schur_rest_pc: str,
    linear_schur_rest_factor_solver_type: str,
    linear_schur_pressure_ksp: str,
    linear_schur_pressure_pc: str,
    linear_schur_pressure_factor_solver_type: str,
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
        "--vi-enter-tol",
        str(float(vi_enter_tol)),
        "--vi-leave-tol",
        str(float(vi_leave_tol)),
        "--vi-persistence",
        str(int(vi_persistence)),
        "--vi-lambda0",
        str(float(vi_lambda0)),
        "--vi-lambda-max",
        str(float(vi_lambda_max)),
        "--vi-lambda-growth",
        str(float(vi_lambda_growth)),
        "--vi-lambda-decay",
        str(float(vi_lambda_decay)),
        "--vi-active-soft-threshold",
        str(int(vi_active_soft_threshold)),
        "--vi-active-soft-alpha",
        str(float(vi_active_soft_alpha)),
        "--vi-active-strong-factor",
        str(float(vi_active_strong_factor)),
        "--vi-filter-max-delta-active",
        str(int(vi_filter_max_delta_active)),
        "--vi-unconstrained-lm" if bool(vi_unconstrained_lm) else "--no-vi-unconstrained-lm",
        "--vi-lm-lambda0",
        str(float(vi_lm_lambda0)),
        "--vi-lm-lambda-max",
        str(float(vi_lm_lambda_max)),
        "--vi-lm-growth",
        str(float(vi_lm_growth)),
        "--vi-lm-decay",
        str(float(vi_lm_decay)),
        "--vi-lm-accept-ratio",
        str(float(vi_lm_accept_ratio)),
        "--vi-lm-good-ratio",
        str(float(vi_lm_good_ratio)),
        "--vi-lm-max-tries",
        str(int(vi_lm_max_tries)),
        "--linear-ksp-type",
        str(linear_ksp_type),
        "--linear-pc-type",
        str(linear_pc_type),
        "--linear-pc-factor-solver-type",
        str(linear_pc_factor_solver_type),
        "--linear-ksp-rtol",
        str(float(linear_ksp_rtol)),
        "--linear-ksp-max-it",
        str(int(linear_ksp_max_it)),
        "--linear-ksp-trace" if bool(linear_ksp_trace) else "--no-linear-ksp-trace",
        "--linear-schur" if bool(linear_schur) else "--no-linear-schur",
        "--linear-schur-pressure-field",
        str(linear_schur_pressure_field),
        "--linear-schur-fact",
        str(linear_schur_fact),
        "--linear-schur-pre",
        str(linear_schur_pre),
        "--linear-schur-rest-ksp",
        str(linear_schur_rest_ksp),
        "--linear-schur-rest-pc",
        str(linear_schur_rest_pc),
        "--linear-schur-rest-factor-solver-type",
        str(linear_schur_rest_factor_solver_type),
        "--linear-schur-pressure-ksp",
        str(linear_schur_pressure_ksp),
        "--linear-schur-pressure-pc",
        str(linear_schur_pressure_pc),
        "--linear-schur-pressure-factor-solver-type",
        str(linear_schur_pressure_factor_solver_type),
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
        "--vi-enter-tol",
        str(float(vi_enter_tol)),
        "--vi-leave-tol",
        str(float(vi_leave_tol)),
        "--vi-persistence",
        str(int(vi_persistence)),
        "--vi-lambda0",
        str(float(vi_lambda0)),
        "--vi-lambda-max",
        str(float(vi_lambda_max)),
        "--vi-lambda-growth",
        str(float(vi_lambda_growth)),
        "--vi-lambda-decay",
        str(float(vi_lambda_decay)),
        "--vi-active-soft-threshold",
        str(int(vi_active_soft_threshold)),
        "--vi-active-soft-alpha",
        str(float(vi_active_soft_alpha)),
        "--vi-active-strong-factor",
        str(float(vi_active_strong_factor)),
        "--vi-filter-max-delta-active",
        str(int(vi_filter_max_delta_active)),
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
        "--vi-enter-tol",
        str(float(vi_enter_tol)),
        "--vi-leave-tol",
        str(float(vi_leave_tol)),
        "--vi-persistence",
        str(int(vi_persistence)),
        "--vi-lambda0",
        str(float(vi_lambda0)),
        "--vi-lambda-max",
        str(float(vi_lambda_max)),
        "--vi-lambda-growth",
        str(float(vi_lambda_growth)),
        "--vi-lambda-decay",
        str(float(vi_lambda_decay)),
        "--vi-active-soft-threshold",
        str(int(vi_active_soft_threshold)),
        "--vi-active-soft-alpha",
        str(float(vi_active_soft_alpha)),
        "--vi-active-strong-factor",
        str(float(vi_active_strong_factor)),
        "--vi-filter-max-delta-active",
        str(int(vi_filter_max_delta_active)),
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
    ap.add_argument("--vi-enter-tol", type=float, default=0.0, help="PDAS hysteretic entry threshold forwarded to the wrapper.")
    ap.add_argument("--vi-leave-tol", type=float, default=0.0, help="PDAS hysteretic release threshold forwarded to the wrapper.")
    ap.add_argument("--vi-persistence", type=int, default=0, help="PDAS active-set persistence forwarded to the wrapper.")
    ap.add_argument("--vi-lambda0", type=float, default=0.0, help="Initial inactive-block PDAS regularization lambda forwarded to the wrapper.")
    ap.add_argument("--vi-lambda-max", type=float, default=1.0e6, help="Maximum inactive-block PDAS regularization lambda forwarded to the wrapper.")
    ap.add_argument("--vi-lambda-growth", type=float, default=5.0, help="Inactive-block PDAS regularization growth factor forwarded to the wrapper.")
    ap.add_argument("--vi-lambda-decay", type=float, default=0.5, help="Inactive-block PDAS regularization decay factor forwarded to the wrapper.")
    ap.add_argument("--vi-active-soft-threshold", type=int, default=0, help="PDAS soft-active damping trigger on DeltaA, forwarded to the wrapper.")
    ap.add_argument("--vi-active-soft-alpha", type=float, default=1.0, help="PDAS soft-active damping factor for marginal actives, forwarded to the wrapper.")
    ap.add_argument("--vi-active-strong-factor", type=float, default=5.0, help="Indicator multiple defining clearly active DOFs, forwarded to the wrapper.")
    ap.add_argument("--vi-filter-max-delta-active", type=int, default=0, help="Reject VI line-search trials with larger predicted DeltaA than this threshold, forwarded to the wrapper.")
    ap.add_argument("--vi-unconstrained-lm", action=argparse.BooleanOptionalAction, default=False, help="Enable zero-active-set LM globalization, forwarded to the wrapper.")
    ap.add_argument("--vi-lm-lambda0", type=float, default=1.0e-4, help="Initial zero-active-set LM damping, forwarded to the wrapper.")
    ap.add_argument("--vi-lm-lambda-max", type=float, default=1.0e6, help="Maximum zero-active-set LM damping, forwarded to the wrapper.")
    ap.add_argument("--vi-lm-growth", type=float, default=5.0, help="Zero-active-set LM damping growth factor, forwarded to the wrapper.")
    ap.add_argument("--vi-lm-decay", type=float, default=0.5, help="Zero-active-set LM damping decay factor, forwarded to the wrapper.")
    ap.add_argument("--vi-lm-accept-ratio", type=float, default=1.0e-3, help="Minimum actual/predicted reduction ratio required to accept an LM step, forwarded to the wrapper.")
    ap.add_argument("--vi-lm-good-ratio", type=float, default=0.75, help="Ratio above which zero-active-set LM damping is relaxed after an accepted step, forwarded to the wrapper.")
    ap.add_argument("--vi-lm-max-tries", type=int, default=6, help="Maximum zero-active-set LM retries per Newton iteration, forwarded to the wrapper.")
    ap.add_argument("--linear-ksp-type", type=str, default="", help="Internal PETSc KSP type forwarded to the wrapper.")
    ap.add_argument("--linear-pc-type", type=str, default="", help="Internal PETSc PC type forwarded to the wrapper.")
    ap.add_argument("--linear-pc-factor-solver-type", type=str, default="", help="Internal PETSc direct factor backend forwarded to the wrapper.")
    ap.add_argument("--linear-ksp-rtol", type=float, default=1.0e-8, help="Internal PETSc KSP relative tolerance forwarded to the wrapper.")
    ap.add_argument("--linear-ksp-max-it", type=int, default=200, help="Internal PETSc KSP iteration limit forwarded to the wrapper.")
    ap.add_argument("--linear-ksp-trace", action=argparse.BooleanOptionalAction, default=False, help="Print PETSc KSP diagnostics in child runs.")
    ap.add_argument("--linear-schur", action=argparse.BooleanOptionalAction, default=False, help="Enable PETSc pressure Schur-complement split in child runs.")
    ap.add_argument("--linear-schur-pressure-field", type=str, default="p", help="Pressure field name used by the Schur split, forwarded to the wrapper.")
    ap.add_argument("--linear-schur-fact", type=str, default="full", choices=("full", "upper", "lower", "diag"), help="PETSc Schur factorization type.")
    ap.add_argument("--linear-schur-pre", type=str, default="selfp", choices=("selfp", "a11", "user"), help="PETSc Schur preconditioner type.")
    ap.add_argument("--linear-schur-rest-ksp", type=str, default="preonly", help="KSP type for the non-pressure Schur block.")
    ap.add_argument("--linear-schur-rest-pc", type=str, default="ilu", help="PC type for the non-pressure Schur block.")
    ap.add_argument("--linear-schur-rest-factor-solver-type", type=str, default="", help="Optional direct factor backend for the non-pressure Schur block.")
    ap.add_argument("--linear-schur-pressure-ksp", type=str, default="preonly", help="KSP type for the pressure Schur block.")
    ap.add_argument("--linear-schur-pressure-pc", type=str, default="jacobi", help="PC type for the pressure Schur block.")
    ap.add_argument("--linear-schur-pressure-factor-solver-type", type=str, default="", help="Optional direct factor backend for the pressure Schur block.")
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
        vi_enter_tol=float(args.vi_enter_tol),
        vi_leave_tol=float(args.vi_leave_tol),
        vi_persistence=int(args.vi_persistence),
        vi_lambda0=float(args.vi_lambda0),
        vi_lambda_max=float(args.vi_lambda_max),
        vi_lambda_growth=float(args.vi_lambda_growth),
        vi_lambda_decay=float(args.vi_lambda_decay),
        vi_active_soft_threshold=int(args.vi_active_soft_threshold),
        vi_active_soft_alpha=float(args.vi_active_soft_alpha),
        vi_active_strong_factor=float(args.vi_active_strong_factor),
        vi_filter_max_delta_active=int(args.vi_filter_max_delta_active),
        vi_unconstrained_lm=bool(args.vi_unconstrained_lm),
        vi_lm_lambda0=float(args.vi_lm_lambda0),
        vi_lm_lambda_max=float(args.vi_lm_lambda_max),
        vi_lm_growth=float(args.vi_lm_growth),
        vi_lm_decay=float(args.vi_lm_decay),
        vi_lm_accept_ratio=float(args.vi_lm_accept_ratio),
        vi_lm_good_ratio=float(args.vi_lm_good_ratio),
        vi_lm_max_tries=int(args.vi_lm_max_tries),
        linear_ksp_type=str(args.linear_ksp_type),
        linear_pc_type=str(args.linear_pc_type),
        linear_pc_factor_solver_type=str(args.linear_pc_factor_solver_type),
        linear_ksp_rtol=float(args.linear_ksp_rtol),
        linear_ksp_max_it=int(args.linear_ksp_max_it),
        linear_ksp_trace=bool(args.linear_ksp_trace),
        linear_schur=bool(args.linear_schur),
        linear_schur_pressure_field=str(args.linear_schur_pressure_field),
        linear_schur_fact=str(args.linear_schur_fact),
        linear_schur_pre=str(args.linear_schur_pre),
        linear_schur_rest_ksp=str(args.linear_schur_rest_ksp),
        linear_schur_rest_pc=str(args.linear_schur_rest_pc),
        linear_schur_rest_factor_solver_type=str(args.linear_schur_rest_factor_solver_type),
        linear_schur_pressure_ksp=str(args.linear_schur_pressure_ksp),
        linear_schur_pressure_pc=str(args.linear_schur_pressure_pc),
        linear_schur_pressure_factor_solver_type=str(args.linear_schur_pressure_factor_solver_type),
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
