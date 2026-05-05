#!/usr/bin/env python3
"""
Calibrate the Lie benchmark one-domain model to experimental deformation dx(t).

This script builds an optimization loop around:
  - experimental deformation extracted from Video S1
    (`extract_deformation_timeseries_from_experimental_video_s1.py`)
  - the simulation driver
    (`lie_synthetic_deformation_one_domain.py`)

We treat calibration as a black-box optimization problem:
  minimize  J(theta) = RMSE( dx_sim(theta; t), dx_exp(t) )

For *publishable* matching to the experiment we use the **PDE transport**
of alpha with **conservative advection** and **Cahn–Hilliard** regularization
(mass conserving). `D_alpha` stays 0.

Notes
-----
* Each objective evaluation runs a full PDE solve (expensive).
* Parameters are optimized in log10-space for robustness.
* Results and per-evaluation logs are cached in `--out-base` so runs can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _read_timeseries(path: Path) -> tuple[np.ndarray, np.ndarray]:
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


def _rmse_cost(
    *,
    t_exp: np.ndarray,
    dx_exp: np.ndarray,
    t_sim: np.ndarray,
    dx_sim: np.ndarray,
    smooth_exp: int,
) -> tuple[float, np.ndarray]:
    dx_exp = np.asarray(dx_exp, dtype=float)
    dx_sim = np.asarray(dx_sim, dtype=float)

    w = int(smooth_exp)
    if w > 1:
        dx_exp = np.column_stack([_moving_average(dx_exp[:, i], w) for i in range(dx_exp.shape[1])])

    # Interpolate sim to exp time grid.
    dx_sim_i = np.column_stack([np.interp(t_exp, t_sim, dx_sim[:, i]) for i in range(dx_sim.shape[1])])
    err = dx_sim_i - dx_exp
    rmse_lines = np.sqrt(np.nanmean(err * err, axis=0))
    rmse_tot = float(np.sqrt(np.nanmean(err * err)))
    return rmse_tot, rmse_lines


def _is_complete_timeseries(path: Path, *, t_final: float) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size <= 0:
                return False
            # Read last ~2KB for the last line.
            f.seek(max(0, size - 2048))
            tail = f.read().decode("utf-8", errors="ignore").strip().splitlines()
        if len(tail) < 2:
            return False
        last = tail[-1].strip().split(",")
        if not last or len(last) < 1:
            return False
        t_last = float(last[0])
        return bool(t_last >= float(t_final) - 1.0e-9)
    except Exception:
        return False


@dataclass(frozen=True)
class Trial:
    log10_Gb: float
    log10_mub: float
    log10_kappa_inv: float
    log10_t_ramp: float

    def as_dict(self) -> dict[str, float]:
        return {
            "log10_Gb": float(self.log10_Gb),
            "log10_mub": float(self.log10_mub),
            "log10_kappa_inv": float(self.log10_kappa_inv),
            "log10_t_ramp": float(self.log10_t_ramp),
        }

    def pretty(self) -> str:
        Gb = 10.0 ** float(self.log10_Gb)
        mub = 10.0 ** float(self.log10_mub)
        kappa_inv = 10.0 ** float(self.log10_kappa_inv)
        t_ramp = 10.0 ** float(self.log10_t_ramp)
        return f"G_b={Gb:.3e} Pa, mu_b={mub:.3e} Pa*s, kappa_inv={kappa_inv:.3e} 1/m^2, t_ramp={t_ramp:.3g} s"


def _trial_id(trial: Trial, *, fixed: dict[str, object] | None = None) -> str:
    payload = json.dumps({"trial": trial.as_dict(), "fixed": fixed or {}}, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run_simulation(
    *,
    trial: Trial,
    sim_script: Path,
    out_dir: Path,
    python_cmd: list[str],
    backend: str,
    restrict_skeleton_dofs: bool,
    restrict_skeleton_alpha_min: float,
    L: float,
    H: float,
    nx: int,
    ny: int,
    nx_left: int,
    nx_mid: int,
    nx_right: int,
    ny_bottom: int,
    ny_top: int,
    dt: float,
    t_final: float,
    qdeg: int,
    newton_tol: float,
    max_it: int,
    snes_accept_factor: float,
    transport_mode: str,
    alpha0_file: Path,
    alpha0_scale: float,
    alpha0_align: str,
    eps: float,
    phi_b: float,
    u_avg: float,
    t_ramp: float,
    mu_b_fluid: float,
    mu_b_model: str,
    solid_model: str,
    gamma_u: float,
    u_extension: str,
    gamma_u_pin: float,
    alpha_advection_form: str,
    alpha_ch_M: float,
    alpha_ch_gamma: float,
    alpha_ch_eps: float,
    alpha_ch_mobility: str,
    alpha_cahn_M: float,
    alpha_cahn_gamma: float,
    alpha_cahn_eps: float,
    alpha_cahn_mobility: str,
    alpha_cahn_conservative: bool,
    dx_intersection: str,
    dx_quantile: float,
    dx_tracking: str,
    dx_fixed_point_iters: int,
    y_fracs: str,
    kappa_inv_default: float,
    gamma_phi: float,
    D_alpha: float,
    alpha_supg: float,
    alpha_cip: float,
    allow_dt_reduction: bool,
    dt_min: float,
    dt_reduction_factor: float,
    stop_on_steady: bool,
    steady_tol: float,
    restrict_skeleton_method: str,
    restrict_skeleton_box_pad: float,
    alpha_clip_below_block: bool,
    alpha_pin_block_top: bool,
    alpha_pin_block_top_value: float,
    alpha_pin_block_top_alpha0_min: float,
    refine_biofilm: bool,
    refine_biofilm_pad: float,
    refine_biofilm_levels: int,
    timeout_s: float,
) -> tuple[int, float]:
    Gb = 10.0 ** float(trial.log10_Gb)
    mub = 10.0 ** float(trial.log10_mub)
    kappa_inv = 10.0 ** float(trial.log10_kappa_inv) if np.isfinite(float(trial.log10_kappa_inv)) else float(kappa_inv_default)
    t_ramp_fit = 10.0 ** float(trial.log10_t_ramp) if np.isfinite(float(trial.log10_t_ramp)) else float(t_ramp)

    _ensure_dir(out_dir)
    log_path = out_dir / "run.log"

    cmd = [
        *python_cmd,
        "-u",
        str(sim_script),
        "--backend",
        str(backend),
        "--restrict-skeleton-dofs" if bool(restrict_skeleton_dofs) else "--no-restrict-skeleton-dofs",
        "--restrict-skeleton-method",
        str(restrict_skeleton_method),
        "--restrict-skeleton-alpha-min",
        str(float(restrict_skeleton_alpha_min)),
        "--restrict-skeleton-box-pad",
        str(float(restrict_skeleton_box_pad)),
        "--L",
        str(float(L)),
        "--H",
        str(float(H)),
        "--nx",
        str(int(nx)),
        "--ny",
        str(int(ny)),
        "--nx-left",
        str(int(nx_left)),
        "--nx-mid",
        str(int(nx_mid)),
        "--nx-right",
        str(int(nx_right)),
        "--ny-bottom",
        str(int(ny_bottom)),
        "--ny-top",
        str(int(ny_top)),
        "--q",
        str(int(qdeg)),
        "--dt",
        str(float(dt)),
        "--t-final",
        str(float(t_final)),
        "--newton-tol",
        str(float(newton_tol)),
        "--max-it",
        str(int(max_it)),
        "--snes-accept-factor",
        str(float(snes_accept_factor)),
        "--vtk-every",
        "0",
        "--out-dir",
        str(out_dir),
        "--transport-mode",
        str(transport_mode),
        "--alpha0-file",
        str(alpha0_file),
        "--alpha0-scale",
        str(float(alpha0_scale)),
        "--alpha0-align",
        str(alpha0_align),
        "--eps",
        str(float(eps)),
        "--phi-b",
        str(float(phi_b)),
        "--u-avg",
        str(float(u_avg)),
        "--t-ramp",
        str(float(t_ramp_fit)),
        "--mu-b-fluid",
        str(float(mu_b_fluid)),
        "--mu-b-model",
        str(mu_b_model),
        "--solid-model",
        str(solid_model),
        "--G-b",
        str(float(Gb)),
        "--mu-b",
        str(float(mub)),
        "--gamma-u",
        str(float(gamma_u)),
        "--u-extension",
        str(u_extension),
        "--gamma-u-pin",
        str(float(gamma_u_pin)),
        "--kappa-inv",
        str(float(kappa_inv)),
        "--gamma-phi",
        str(float(gamma_phi)),
        "--D-alpha",
        str(float(D_alpha)),
        "--alpha-advection-form",
        str(alpha_advection_form),
        "--alpha-ch-M",
        str(float(alpha_ch_M)),
        "--alpha-ch-gamma",
        str(float(alpha_ch_gamma)),
        "--alpha-ch-eps",
        str(float(alpha_ch_eps)),
        "--alpha-ch-mobility",
        str(alpha_ch_mobility),
        "--alpha-cahn-M",
        str(float(alpha_cahn_M)),
        "--alpha-cahn-gamma",
        str(float(alpha_cahn_gamma)),
        "--alpha-cahn-eps",
        str(float(alpha_cahn_eps)),
        "--alpha-cahn-mobility",
        str(alpha_cahn_mobility),
        "--alpha-supg",
        str(float(alpha_supg)),
        "--alpha-cip",
        str(float(alpha_cip)),
        "--dx-intersection",
        str(dx_intersection),
        "--dx-quantile",
        str(float(dx_quantile)),
        "--dx-tracking",
        str(dx_tracking),
        "--dx-fixed-point-iters",
        str(int(dx_fixed_point_iters)),
    ]
    if str(y_fracs).strip():
        cmd += ["--y-fracs", str(y_fracs)]
    if bool(alpha_cahn_conservative):
        cmd += ["--alpha-cahn-conservative"]
    if bool(alpha_clip_below_block):
        cmd += ["--alpha-clip-below-block"]
    if bool(alpha_pin_block_top):
        cmd += ["--alpha-pin-block-top"]
        if np.isfinite(float(alpha_pin_block_top_value)):
            cmd += ["--alpha-pin-block-top-value", str(float(alpha_pin_block_top_value))]
        cmd += ["--alpha-pin-block-top-alpha0-min", str(float(alpha_pin_block_top_alpha0_min))]
    if bool(refine_biofilm):
        cmd += [
            "--refine-biofilm",
            "--refine-biofilm-pad",
            str(float(refine_biofilm_pad)),
            "--refine-biofilm-levels",
            str(int(refine_biofilm_levels)),
        ]
    if bool(allow_dt_reduction):
        cmd += [
            "--allow-dt-reduction",
            "--dt-min",
            str(float(dt_min)),
            "--dt-reduction-factor",
            str(float(dt_reduction_factor)),
        ]
    if bool(stop_on_steady):
        cmd += ["--stop-on-steady", "--steady-tol", str(float(steady_tol))]

    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        f.write("# cmd:\n")
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # allow killing the whole process group on timeout
        )
        try:
            proc.wait(timeout=max(1.0, float(timeout_s)))
        except subprocess.TimeoutExpired:
            f.write(f"\n[timeout] exceeded {float(timeout_s):g} s; killing process group.\n")
            f.flush()
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            try:
                proc.wait(timeout=10.0)
            except Exception:
                pass
    wall = float(time.time() - t0)
    rc = int(proc.returncode) if proc.returncode is not None else 124
    return rc, wall


def main() -> None:
    ap = argparse.ArgumentParser(description="Optimize Lie one-domain parameters to match experimental dx(t).")
    ap.add_argument("--exp-csv", type=str, default="examples/biofilms/benchmarks/lie/exp_s1_dx_leftmost_scalebar.csv")
    ap.add_argument("--sim-script", type=str, default="examples/biofilms/benchmarks/lie/lie_synthetic_deformation_one_domain.py")
    ap.add_argument("--out-base", type=str, default="out/_lie_opt")
    ap.add_argument("--smooth-exp", type=int, default=5)

    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--transport-mode", type=str, default="pde", choices=("refmap", "pde"))
    ap.add_argument("--dx-quantile", type=float, default=0.0, help="Pass-through to sim driver: --dx-quantile.")
    ap.add_argument("--y-fracs", type=str, default="", help="Pass-through to sim driver: --y-fracs.")
    ap.add_argument(
        "--dx-tracking",
        type=str,
        default="lagrangian_u",
        choices=("alpha_half", "lagrangian_u"),
        help="Pass-through to sim driver: --dx-tracking.",
    )
    ap.add_argument("--dx-fixed-point-iters", type=int, default=7, help="Pass-through to sim driver: --dx-fixed-point-iters.")

    # Discretization defaults: full experimental geometry, but still coarse enough
    # to be usable for black-box calibration.
    ap.add_argument("--L", type=float, default=15.0e-3)
    ap.add_argument("--H", type=float, default=10.0e-3)
    ap.add_argument("--nx", type=int, default=80)
    ap.add_argument("--ny", type=int, default=40)
    ap.add_argument("--nx-left", type=int, default=28)
    ap.add_argument("--nx-mid", type=int, default=24)
    ap.add_argument("--nx-right", type=int, default=28)
    ap.add_argument("--ny-bottom", type=int, default=8)
    ap.add_argument("--ny-top", type=int, default=32)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--t-final", type=float, default=20.0)
    # NOTE: q=3 is too low for this mixed Q2/Q1 system and can under-integrate
    # coupling terms enough to produce near-zero deformation. Use q>=4.
    ap.add_argument("--q", type=int, default=6)
    # IMPORTANT: Too-loose tolerances can lead to a "fake steady" solve where the
    # flow converges but the coupled skeleton/alpha fields do not move (dx(t)≈0).
    # Use a tighter atol than the quick-and-dirty default.
    ap.add_argument("--newton-tol", type=float, default=3.0e-6)
    ap.add_argument("--max-it", type=int, default=25)
    ap.add_argument("--snes-accept-factor", type=float, default=3.0)
    ap.add_argument("--allow-dt-reduction", action="store_true")
    ap.add_argument("--dt-min", type=float, default=0.005)
    ap.add_argument("--dt-reduction-factor", type=float, default=0.5)
    ap.add_argument("--stop-on-steady", action="store_true", help="Pass --stop-on-steady to the sim driver (speeds up runs that plateau early).")
    ap.add_argument("--steady-tol", type=float, default=1.0e-12, help="Threshold for --stop-on-steady (sim driver).")
    ap.add_argument("--sim-timeout-s", type=float, default=240.0, help="Hard timeout per simulation evaluation (seconds).")

    ap.add_argument(
        "--alpha0-file",
        type=str,
        default="examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_scalebar_smooth.csv",
    )
    ap.add_argument("--alpha0-scale", type=float, default=1.0e-3)
    ap.add_argument("--alpha0-align", type=str, default="block", choices=("none", "block"))
    ap.add_argument("--eps", type=float, default=2.0e-4)
    ap.add_argument("--phi-b", type=float, default=0.47)

    ap.add_argument("--u-avg", type=float, default=6.0e-4)
    ap.add_argument("--t-ramp", type=float, default=8.0, help="Baseline inflow ramp time (used if not optimizing t_ramp).")
    ap.add_argument("--mu-b-fluid", type=float, default=30494.0, help="Biofilm viscosity used in the mixture viscosity model [Pa*s].")
    ap.add_argument(
        "--mu-b-model",
        type=str,
        default="mu",
        choices=("mu", "phi_mu", "alpha_mu", "alpha_phi_mu"),
        help="Mixture viscosity model. For poroelastic calibration, 'mu' (μ=μ_f everywhere) is often the most robust default.",
    )
    ap.add_argument(
        "--solid-model",
        type=str,
        default="linear",
        choices=("linear", "hencky", "svk", "neo_hookean"),
        help="Skeleton constitutive model passed to the simulation driver.",
    )
    ap.add_argument("--gamma-u", type=float, default=1.0e-6)
    ap.add_argument("--u-extension", type=str, default="grad", choices=("l2", "grad"))
    ap.add_argument("--gamma-u-pin", type=float, default=1.0e-10)
    ap.add_argument("--kappa-inv", type=float, default=1.0e8, help="Default when not optimizing kappa_inv.")
    ap.add_argument("--gamma-phi", type=float, default=5.0)
    ap.add_argument("--D-alpha", type=float, default=0.0)
    ap.add_argument("--alpha-advection-form", type=str, default="conservative", choices=("advective", "conservative"))
    ap.add_argument("--alpha-ch-M", type=float, default=1.0e-12)
    ap.add_argument("--alpha-ch-gamma", type=float, default=2.0e-3)
    ap.add_argument("--alpha-ch-eps", type=float, default=float("nan"), help="Default: use --eps.")
    ap.add_argument("--alpha-ch-mobility", type=str, default="degenerate", choices=("constant", "degenerate"))
    ap.add_argument("--alpha-cahn-M", type=float, default=0.0, help="Allen–Cahn mobility M (0 disables).")
    ap.add_argument("--alpha-cahn-gamma", type=float, default=0.0, help="Allen–Cahn gamma (0 disables).")
    ap.add_argument("--alpha-cahn-eps", type=float, default=float("nan"), help="Default: use --eps.")
    ap.add_argument("--alpha-cahn-mobility", type=str, default="constant", choices=("constant", "degenerate"))
    ap.add_argument("--alpha-cahn-conservative", action="store_true", help="Enable conservative Allen–Cahn (global λ_α).")
    ap.add_argument("--dx-intersection", type=str, default="leftmost", choices=("rightmost", "leftmost"))
    ap.add_argument("--alpha-supg", type=float, default=0.0)
    ap.add_argument("--alpha-cip", type=float, default=0.0)
    ap.add_argument(
        "--restrict-skeleton-dofs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --restrict-skeleton-dofs/--no-restrict-skeleton-dofs to the simulation driver.",
    )
    ap.add_argument(
        "--restrict-skeleton-method",
        type=str,
        default="box",
        choices=("alpha", "box"),
        help="Pass-through to sim driver: --restrict-skeleton-method.",
    )
    ap.add_argument(
        "--restrict-skeleton-box-pad",
        type=float,
        default=2.0e-3,
        help="Pass-through to sim driver: --restrict-skeleton-box-pad (used with --restrict-skeleton-method box).",
    )
    ap.add_argument(
        "--restrict-skeleton-alpha-min",
        type=float,
        default=0.01,
        help="Alpha threshold for keeping skeleton DOFs when restriction is enabled (simulation driver).",
    )
    ap.add_argument("--alpha-clip-below-block", action="store_true", help="Pass-through to sim driver: --alpha-clip-below-block.")
    ap.add_argument("--alpha-pin-block-top", action="store_true", help="Pass-through to sim driver: --alpha-pin-block-top.")
    ap.add_argument(
        "--alpha-pin-block-top-value",
        type=float,
        default=float("nan"),
        help="Pass-through to sim driver: --alpha-pin-block-top-value (e.g. 1.0).",
    )
    ap.add_argument(
        "--alpha-pin-block-top-alpha0-min",
        type=float,
        default=0.9,
        help="Pass-through to sim driver: --alpha-pin-block-top-alpha0-min.",
    )
    ap.add_argument("--refine-biofilm", action="store_true", help="Pass-through to sim driver: --refine-biofilm.")
    ap.add_argument("--refine-biofilm-pad", type=float, default=5.0e-4, help="Pass-through to sim driver: --refine-biofilm-pad.")
    ap.add_argument("--refine-biofilm-levels", type=int, default=1, help="Pass-through to sim driver: --refine-biofilm-levels.")

    # Parameter bounds (log10-space for Gb and mu_b).
    # Narrower defaults keep the optimizer away from numerically pathological regions
    # (e.g. extremely small viscosities that can make SNES stall for a long time).
    ap.add_argument("--log10-Gb-min", type=float, default=-4.0)
    ap.add_argument("--log10-Gb-max", type=float, default=1.0)
    ap.add_argument("--log10-mub-min", type=float, default=-2.0)
    ap.add_argument("--log10-mub-max", type=float, default=4.0)
    ap.add_argument("--log10-kappa-inv-min", type=float, default=6.0)
    ap.add_argument("--log10-kappa-inv-max", type=float, default=14.0)
    ap.add_argument("--log10-t-ramp-min", type=float, default=-1.0, help="Lower bound for log10(t_ramp [s]).")
    ap.add_argument("--log10-t-ramp-max", type=float, default=2.0, help="Upper bound for log10(t_ramp [s]).")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-init", type=int, default=6, help="Random initial samples before local optimization.")
    ap.add_argument("--max-evals", type=int, default=20, help="Total simulation evaluations budget (including init).")
    ap.add_argument("--local-method", type=str, default="Powell", choices=("Powell", "Nelder-Mead"))
    ap.add_argument("--resume", action="store_true", help="Reuse cached runs and append to eval log.")
    ap.add_argument("--dry-run", action="store_true", help="Evaluate only the initial guess and exit.")
    ap.add_argument("--use-conda-run", type=str, default="", help="If set, run sim via `conda run -n <ENV>`.")
    args = ap.parse_args()

    exp_csv = Path(str(args.exp_csv))
    sim_script = Path(str(args.sim_script))
    out_base = Path(str(args.out_base))
    _ensure_dir(out_base)

    if not exp_csv.exists():
        raise FileNotFoundError(f"Experimental CSV not found: {exp_csv}")
    if not sim_script.exists():
        raise FileNotFoundError(f"Simulation script not found: {sim_script}")

    t_exp, dx_exp = _read_timeseries(exp_csv)

    # Regularization mode sanity checks (driver enforces too, but fail early here).
    ac_enabled = bool(float(args.alpha_cahn_M) != 0.0 and float(args.alpha_cahn_gamma) != 0.0)
    ch_enabled = bool(float(args.alpha_ch_M) != 0.0 and float(args.alpha_ch_gamma) != 0.0)
    if ac_enabled and ch_enabled:
        raise ValueError("Allen–Cahn (--alpha-cahn-*) and Cahn–Hilliard (--alpha-ch-*) cannot both be enabled.")
    if bool(args.alpha_cahn_conservative) and (not ac_enabled):
        raise ValueError("--alpha-cahn-conservative requires --alpha-cahn-M and --alpha-cahn-gamma to be nonzero.")
    alpha_ch_eps = float(args.alpha_ch_eps) if np.isfinite(float(args.alpha_ch_eps)) else float(args.eps)
    alpha_cahn_eps = float(args.alpha_cahn_eps) if np.isfinite(float(args.alpha_cahn_eps)) else float(args.eps)
    t_final = float(args.t_final)
    m_exp = t_exp <= t_final + 1.0e-12
    t_exp = t_exp[m_exp]
    dx_exp = dx_exp[m_exp, :]

    # Decide how to invoke python for the simulation.
    if str(args.use_conda_run).strip():
        python_cmd = ["conda", "run", "--no-capture-output", "-n", str(args.use_conda_run), "python"]
    else:
        python_cmd = [sys.executable]

    evals_csv = out_base / "evals.csv"
    if not evals_csv.exists() or not bool(args.resume):
        with evals_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "eval_id",
                    "trial_id",
                    "status",
                    "cost_rmse_m",
                    "rmse_line1_m",
                    "rmse_line2_m",
                    "rmse_line3_m",
                    "log10_Gb",
                    "log10_mub",
                    "log10_kappa_inv",
                    "log10_t_ramp",
                    "wall_s",
                    "sim_dir",
                    "returncode",
                ]
            )

    bounds = [
        (float(args.log10_Gb_min), float(args.log10_Gb_max)),
        (float(args.log10_mub_min), float(args.log10_mub_max)),
        (float(args.log10_kappa_inv_min), float(args.log10_kappa_inv_max)),
        (float(args.log10_t_ramp_min), float(args.log10_t_ramp_max)),
    ]

    def clamp_to_bounds(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).copy()
        for i, (lo, hi) in enumerate(bounds):
            if x[i] < lo:
                x[i] = lo
            if x[i] > hi:
                x[i] = hi
        return x

    cache: dict[str, float] = {}
    best_seen_x: np.ndarray | None = None
    best_seen_f: float = float("inf")
    best_seen_trial: Trial | None = None
    fixed_fingerprint = {
        "backend": str(args.backend),
        "transport_mode": str(args.transport_mode),
        "alpha_transport": {
            "D_alpha": float(args.D_alpha),
            "alpha_advection_form": str(args.alpha_advection_form),
            "alpha_supg": float(args.alpha_supg),
            "alpha_cip": float(args.alpha_cip),
            "alpha_ch": {
                "M": float(args.alpha_ch_M),
                "gamma": float(args.alpha_ch_gamma),
                "eps": float(alpha_ch_eps),
                "mobility": str(args.alpha_ch_mobility),
            },
            "alpha_cahn": {
                "M": float(args.alpha_cahn_M),
                "gamma": float(args.alpha_cahn_gamma),
                "eps": float(alpha_cahn_eps),
                "mobility": str(args.alpha_cahn_mobility),
                "conservative": bool(args.alpha_cahn_conservative),
            },
        },
        "dx_intersection": str(args.dx_intersection),
        "dx_quantile": float(args.dx_quantile),
        "dx_tracking": str(args.dx_tracking),
        "dx_fixed_point_iters": int(args.dx_fixed_point_iters),
        "y_fracs": str(args.y_fracs),
        "restrict_skeleton_dofs": bool(args.restrict_skeleton_dofs),
        "restrict_skeleton_method": str(args.restrict_skeleton_method),
        "restrict_skeleton_alpha_min": float(args.restrict_skeleton_alpha_min),
        "restrict_skeleton_box_pad": float(args.restrict_skeleton_box_pad),
        "alpha_clip_below_block": bool(args.alpha_clip_below_block),
        "alpha_pin_block_top": bool(args.alpha_pin_block_top),
        "alpha_pin_block_top_value": float(args.alpha_pin_block_top_value),
        "alpha_pin_block_top_alpha0_min": float(args.alpha_pin_block_top_alpha0_min),
        "refine_biofilm": bool(args.refine_biofilm),
        "refine_biofilm_pad": float(args.refine_biofilm_pad),
        "refine_biofilm_levels": int(args.refine_biofilm_levels),
        "alpha0_file": str(Path(str(args.alpha0_file))),
        "alpha0_scale": float(args.alpha0_scale),
        "alpha0_align": str(args.alpha0_align),
        "eps": float(args.eps),
        "phi_b": float(args.phi_b),
        "flow": {
            "u_avg": float(args.u_avg),
            "t_ramp_baseline": float(args.t_ramp),
        },
        "material_fixed": {
            "mu_b_fluid": float(args.mu_b_fluid),
            "mu_b_model": str(args.mu_b_model),
            "solid_model": str(args.solid_model),
            "gamma_u": float(args.gamma_u),
            "u_extension": str(args.u_extension),
            "gamma_u_pin": float(args.gamma_u_pin),
            "gamma_phi": float(args.gamma_phi),
            "kappa_inv_default": float(args.kappa_inv),
        },
        "dt": float(args.dt),
        "t_final": float(args.t_final),
        "numerics": {
            "q": int(args.q),
            "newton_tol": float(args.newton_tol),
            "max_it": int(args.max_it),
            "snes_accept_factor": float(args.snes_accept_factor),
            "allow_dt_reduction": bool(args.allow_dt_reduction),
            "dt_min": float(args.dt_min),
            "dt_reduction_factor": float(args.dt_reduction_factor),
            "stop_on_steady": bool(args.stop_on_steady),
            "steady_tol": float(args.steady_tol),
        },
        "mesh": {
            "nx": int(args.nx),
            "ny": int(args.ny),
            "nx_left": int(args.nx_left),
            "nx_mid": int(args.nx_mid),
            "nx_right": int(args.nx_right),
            "ny_bottom": int(args.ny_bottom),
            "ny_top": int(args.ny_top),
        },
    }

    def objective(x: np.ndarray) -> float:
        nonlocal best_seen_x, best_seen_f, best_seen_trial
        x = clamp_to_bounds(np.asarray(x, dtype=float))
        trial = Trial(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
        tid = _trial_id(trial, fixed=fixed_fingerprint)
        if tid in cache:
            cost = float(cache[tid])
            if float(cost) < float(best_seen_f):
                best_seen_f = float(cost)
                best_seen_x = np.asarray(x, dtype=float).copy()
                best_seen_trial = trial
            return float(cost)

        run_dir = out_base / f"run_{tid}"
        ts_path = run_dir / "timeseries.csv"

        rc = 0
        wall = 0.0
        if not _is_complete_timeseries(ts_path, t_final=t_final):
            rc, wall = _run_simulation(
                trial=trial,
                sim_script=sim_script,
                out_dir=run_dir,
                python_cmd=python_cmd,
                backend=str(args.backend),
                restrict_skeleton_dofs=bool(args.restrict_skeleton_dofs),
                restrict_skeleton_alpha_min=float(args.restrict_skeleton_alpha_min),
                L=float(args.L),
                H=float(args.H),
                nx=int(args.nx),
                ny=int(args.ny),
                nx_left=int(args.nx_left),
                nx_mid=int(args.nx_mid),
                nx_right=int(args.nx_right),
                ny_bottom=int(args.ny_bottom),
                ny_top=int(args.ny_top),
                dt=float(args.dt),
                t_final=t_final,
                qdeg=int(args.q),
                newton_tol=float(args.newton_tol),
                max_it=int(args.max_it),
                snes_accept_factor=float(args.snes_accept_factor),
                transport_mode=str(args.transport_mode),
                alpha0_file=Path(str(args.alpha0_file)),
                alpha0_scale=float(args.alpha0_scale),
                alpha0_align=str(args.alpha0_align),
                eps=float(args.eps),
                phi_b=float(args.phi_b),
                u_avg=float(args.u_avg),
                t_ramp=float(args.t_ramp),
                mu_b_fluid=float(args.mu_b_fluid),
                mu_b_model=str(args.mu_b_model),
                solid_model=str(args.solid_model),
                gamma_u=float(args.gamma_u),
                u_extension=str(args.u_extension),
                gamma_u_pin=float(args.gamma_u_pin),
                alpha_advection_form=str(args.alpha_advection_form),
                alpha_ch_M=float(args.alpha_ch_M),
                alpha_ch_gamma=float(args.alpha_ch_gamma),
                alpha_ch_eps=float(alpha_ch_eps),
                alpha_ch_mobility=str(args.alpha_ch_mobility),
                alpha_cahn_M=float(args.alpha_cahn_M),
                alpha_cahn_gamma=float(args.alpha_cahn_gamma),
                alpha_cahn_eps=float(alpha_cahn_eps),
                alpha_cahn_mobility=str(args.alpha_cahn_mobility),
                alpha_cahn_conservative=bool(args.alpha_cahn_conservative),
                dx_intersection=str(args.dx_intersection),
                dx_quantile=float(args.dx_quantile),
                dx_tracking=str(args.dx_tracking),
                dx_fixed_point_iters=int(args.dx_fixed_point_iters),
                y_fracs=str(args.y_fracs),
                kappa_inv_default=float(args.kappa_inv),
                gamma_phi=float(args.gamma_phi),
                D_alpha=float(args.D_alpha),
                alpha_supg=float(args.alpha_supg),
                alpha_cip=float(args.alpha_cip),
                allow_dt_reduction=bool(args.allow_dt_reduction),
                dt_min=float(args.dt_min),
                dt_reduction_factor=float(args.dt_reduction_factor),
                stop_on_steady=bool(args.stop_on_steady),
                steady_tol=float(args.steady_tol),
                restrict_skeleton_method=str(args.restrict_skeleton_method),
                restrict_skeleton_box_pad=float(args.restrict_skeleton_box_pad),
                alpha_clip_below_block=bool(args.alpha_clip_below_block),
                alpha_pin_block_top=bool(args.alpha_pin_block_top),
                alpha_pin_block_top_value=float(args.alpha_pin_block_top_value),
                alpha_pin_block_top_alpha0_min=float(args.alpha_pin_block_top_alpha0_min),
                refine_biofilm=bool(args.refine_biofilm),
                refine_biofilm_pad=float(args.refine_biofilm_pad),
                refine_biofilm_levels=int(args.refine_biofilm_levels),
                timeout_s=float(args.sim_timeout_s),
            )
        else:
            # Cached run.
            wall = 0.0
            rc = 0

        status = "ok" if int(rc) == 0 and ts_path.exists() else "fail"
        if status != "ok":
            cost = 1.0e6
            rmse_lines = np.array([np.nan, np.nan, np.nan], dtype=float)
        else:
            t_sim, dx_sim = _read_timeseries(ts_path)
            m_sim = t_sim <= t_final + 1.0e-12
            t_sim = t_sim[m_sim]
            dx_sim = dx_sim[m_sim, :]
            if t_sim.size < 2:
                cost = 1.0e6
                rmse_lines = np.array([np.nan, np.nan, np.nan], dtype=float)
                status = "short"
            else:
                cost, rmse_lines = _rmse_cost(t_exp=t_exp, dx_exp=dx_exp, t_sim=t_sim, dx_sim=dx_sim, smooth_exp=int(args.smooth_exp))

        # Persist row.
        with evals_csv.open("r", encoding="utf-8") as f_in:
            eval_id = int(max(0, sum(1 for _ in f_in) - 1))
        with evals_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    eval_id,
                    tid,
                    status,
                    float(cost),
                    float(rmse_lines[0]),
                    float(rmse_lines[1]),
                    float(rmse_lines[2]),
                    float(trial.log10_Gb),
                    float(trial.log10_mub),
                    float(trial.log10_kappa_inv),
                    float(trial.log10_t_ramp),
                    float(wall),
                    str(run_dir),
                    int(rc),
                ]
            )

        cache[tid] = float(cost)
        if float(cost) < float(best_seen_f):
            best_seen_f = float(cost)
            best_seen_x = np.asarray(x, dtype=float).copy()
            best_seen_trial = trial
        print(f"[eval] {tid} {trial.pretty()} -> RMSE={float(cost)*1e3:.4g} mm ({status})", flush=True)
        return float(cost)

    rng = np.random.default_rng(int(args.seed))

    # Initial guess (effective poroelastic surrogate; tuned near a known-good fit for Video S1).
    x0 = np.array(
        [
            math.log10(5.0e-3),
            math.log10(3.0e-2),
            math.log10(float(args.kappa_inv)),
            math.log10(float(args.t_ramp)),
        ],
        dtype=float,
    )
    x0 = clamp_to_bounds(x0)

    if bool(args.dry_run):
        _ = objective(x0)
        return

    # Random initial sampling.
    best_x = x0.copy()
    best_f = float(objective(best_x))
    n_init = max(0, int(args.n_init))
    for _ in range(n_init):
        x = np.array(
            [
                rng.uniform(bounds[0][0], bounds[0][1]),
                rng.uniform(bounds[1][0], bounds[1][1]),
                rng.uniform(bounds[2][0], bounds[2][1]),
                rng.uniform(bounds[3][0], bounds[3][1]),
            ],
            dtype=float,
        )
        f = float(objective(x))
        if f < best_f:
            best_f, best_x = f, x

    remaining = max(1, int(args.max_evals) - (n_init + 1))
    print(f"[best-init] RMSE={best_f*1e3:.4g} mm at x={best_x.tolist()}", flush=True)

    # Local optimization (bounded via clamping inside objective).
    try:
        from scipy.optimize import minimize
    except Exception as e:  # pragma: no cover
        minimize = None
        print(f"[warn] SciPy not available ({e}); skipping local optimization and keeping best random sample.", flush=True)

    if minimize is not None and remaining > 0:
        options = {"maxfev": int(remaining), "disp": True}
        method = str(args.local_method)
        res = minimize(objective, best_x, method=method, options=options)
        x_opt = clamp_to_bounds(np.asarray(res.x, dtype=float))
        meta = {"success": bool(res.success), "message": str(res.message), "nfev": int(getattr(res, "nfev", -1))}
    else:
        x_opt = clamp_to_bounds(np.asarray(best_x, dtype=float))
        meta = {"success": True, "message": "no_local_optimization", "nfev": 0}

    # Record the best trial across *all* evaluations (random + local optimizer probes).
    # Optimizers like Powell can exceed the evaluation budget mid-iteration and return
    # the starting point; we still want the best evaluated parameters for reproducibility.
    _ = float(objective(x_opt))
    if best_seen_trial is None or best_seen_x is None or not np.isfinite(float(best_seen_f)):
        best_seen_x = np.asarray(x_opt, dtype=float).copy()
        best_seen_f = float(
            cache.get(
                _trial_id(
                    Trial(float(best_seen_x[0]), float(best_seen_x[1]), float(best_seen_x[2]), float(best_seen_x[3])),
                    fixed=fixed_fingerprint,
                ),
                float("nan"),
            )
        )
        best_seen_trial = Trial(float(best_seen_x[0]), float(best_seen_x[1]), float(best_seen_x[2]), float(best_seen_x[3]))
    f_opt = float(best_seen_f)
    trial_opt = best_seen_trial

    best_path = out_base / "best.json"
    best_payload = {
        "trial": trial_opt.as_dict(),
        "pretty": trial_opt.pretty(),
        "rmse_m": float(f_opt),
        "rmse_mm": float(f_opt) * 1.0e3,
        "result": meta,
        "timestamp": float(time.time()),
    }
    best_path.write_text(json.dumps(best_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[best] {trial_opt.pretty()} -> RMSE={f_opt*1e3:.4g} mm")
    print(f"[ok] wrote {best_path}")


if __name__ == "__main__":
    main()
