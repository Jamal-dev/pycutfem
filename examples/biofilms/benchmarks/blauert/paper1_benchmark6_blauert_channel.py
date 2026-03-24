#!/usr/bin/env python3
"""Paper 1 Benchmark 6: Blauert attached-patch channel deformation benchmark.

This benchmark reconnects the one-domain model to the motivating Blauert/Dian
application while staying inside the Paper 1 scope:

  - one-domain fluid momentum + mass + skeleton momentum + kinematics,
  - active porosity transport `phi`,
  - conservative Cahn--Hilliard transport for `alpha`,
  - no growth, detachment, or damage.

The application loading is calibrated against published Blauert/Dian
observations instead of frame-by-frame OCT matching. The primary target is the
steady traced contour used in Dian/Feng, while the secondary checks are the
first dynamic Blauert observations (front compression plateau and porosity
drop). When the baseline one-domain loading underpredicts the observed
deformation, the script can add a diffuse tangential traction correction
localized on the transported interface. That correction stays benchmark-local
and is documented explicitly if it is used in the final paper benchmark. For
the Dian/Blauert geometry we keep the channel dimensions from the Matlab
preprocessing and interpret Dian's reported `v0=6.84e-2 m/s` as the peak
Poiseuille speed, so the parabolic inflow used here defaults to `u_avg = (2/3)
v0 ≈ 4.56e-2 m/s`.

The workflow has two stages:

  1. coarse calibration over `(E, eta_s, zeta_t)` on a fixed mesh using
     observation-level mismatch metrics,
  2. a production rerun on a mesh ladder with the selected parameters, together
     with steady contour overlays and observable histories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional in light envs
    matplotlib = None
    plt = None

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.biofilms.benchmarks.blauert.compare_sim_vs_video import _interp_1d, _read_csv_columns
from examples.biofilms.benchmarks.blauert.compare_sim_vs_observations import _svg_contour_to_mm
from examples.biofilms.benchmarks.blauert.extract_front_displacement_from_video import (
    _contour_roi_from_polygon_mm,
    _crop_mask_to_mm_roi,
)


BLAUERT_DRIVER = _REPO_ROOT / "examples/biofilms/benchmarks/blauert/blauert_biofilm_deformation_one_domain.py"
OBS_COMPARE_SCRIPT = _REPO_ROOT / "examples/biofilms/benchmarks/blauert/compare_sim_vs_observations.py"
COMPARE_SCRIPT = _REPO_ROOT / "examples/biofilms/benchmarks/blauert/compare_sim_vs_video.py"
EXTRACT_SCRIPT = _REPO_ROOT / "examples/biofilms/benchmarks/blauert/extract_front_displacement_from_video.py"
VIDEO_PATH = _REPO_ROOT / "examples/biofilms/benchmarks/blauert/biofilm_preprocessing/1-s2.0-S0043135418307000-mmc1.mp4"
EXP_CSV = _REPO_ROOT / "examples/biofilms/benchmarks/blauert/exp_front_displacement_from_video.csv"
POLY_CSV = _REPO_ROOT / "examples/biofilms/benchmarks/blauert/exp_frame0_polygon_mm.csv"
MATLAB_BIOFILM_TXT = _REPO_ROOT / "examples/biofilms/benchmarks/blauert/biofilm_preprocessing/biofilm.txt"
STEADY_SVG = _REPO_ROOT / "examples/biofilms/benchmarks/blauert/biofilm_preprocessing/Basic_t=2_INK.svg"
STEADY_DOMAIN = _REPO_ROOT / "examples/biofilms/benchmarks/blauert/biofilm_preprocessing/domain1.txt"
CONTOUR_ROI_PAD_RIGHT_MM = 0.02
CONTOUR_ROI_PAD_TOP_MM = 0.02


@dataclass(frozen=True)
class CalibrationRow:
    E: float
    solid_visco_eta: float
    diffuse_shear_scale: float
    nx: int
    ny: int
    steady_profile_rmse_um: float
    steady_front_y150_err_um: float
    front_compression_2p0_err_um: float
    front_plateau_drift_2p0_10p0_um: float
    porosity_drop_2p0_err_pp: float
    score: float
    case_dir: str


@dataclass(frozen=True)
class FailedCalibrationCase:
    E: float
    solid_visco_eta: float
    diffuse_shear_scale: float
    nx: int
    ny: int
    case_dir: str
    stage: str
    error: str


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


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        out.append(int(text))
    if not out:
        raise ValueError("Expected at least one integer value.")
    return out


def _ny_from_nx(nx: int) -> int:
    return max(12, int(round(0.375 * float(nx))))


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    skip_existing: bool = False,
    sentinel: Path | None = None,
    stream: bool = False,
) -> None:
    if skip_existing and sentinel is not None and sentinel.exists():
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("+ " + shlex.join(cmd) + "\n")
        log.flush()
        if bool(stream):
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            proc.wait()
        else:
            proc = subprocess.run(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n" + "\n".join(tail))


def _polygon_from_matlab_preprocessing(path: Path, *, L_um: float = 2000.0, shift_um: float = 0.0) -> np.ndarray:
    data = np.loadtxt(str(path), dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected Nx2 coordinates in {path}")
    vx = np.asarray(data[:, 0], dtype=float).ravel()
    vy = np.asarray(data[:, 1], dtype=float).ravel()
    if vx.size < 3:
        raise ValueError(f"Polygon in {path} is too short (N={int(vx.size)})")
    x_max = float(np.max(vx))
    y_max = float(np.max(vy))
    if not (x_max > 0.0):
        raise ValueError(f"Invalid x_max={x_max:g} in {path}")
    vy = float(y_max) - vy
    vx_um = (vx / x_max) * float(L_um) + float(shift_um)
    vy_um = (vy / x_max) * float(L_um)
    pts = np.column_stack([1.0e-3 * vx_um, 1.0e-3 * vy_um]).astype(float)
    if not np.allclose(pts[0], pts[-1], rtol=0.0, atol=1.0e-12):
        pts = np.vstack([pts, pts[0]])
    return pts


def _ensure_experimental_inputs(*, t_max: float, y_levels_um: str, force: bool) -> None:
    del t_max
    del y_levels_um
    del force

    if not EXP_CSV.exists():
        raise FileNotFoundError(
            f"Missing checked-in experimental trace {EXP_CSV}. "
            "Benchmark 6 expects the pre-extracted video CSV to be present in the repository."
        )
    if POLY_CSV.exists():
        return
    poly = _polygon_from_matlab_preprocessing(MATLAB_BIOFILM_TXT, L_um=2000.0, shift_um=0.0)
    POLY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with POLY_CSV.open("w", encoding="utf-8") as f:
        f.write("x_mm,y_mm\n")
        for xx, yy in np.asarray(poly, dtype=float):
            f.write(f"{float(xx):.9f},{float(yy):.9f}\n")


def _candidate_case_name(*, E: float, solid_visco_eta: float, diffuse_shear_scale: float, nx: int) -> str:
    scale_tag = f"{float(diffuse_shear_scale):.3g}".replace(".", "p")
    return f"E{float(E):.0f}_eta{float(solid_visco_eta):.0f}_zeta{scale_tag}_nx{int(nx)}"


def _simulation_command(
    *,
    case_dir: Path,
    nx: int,
    ny: int,
    E: float,
    solid_visco_eta: float,
    diffuse_shear_scale: float,
    diffuse_shear_model: str,
    diffuse_shear_time_scheme: str,
    diffuse_shear_ramp_time: float,
    nonlinear_solver: str,
    ls_mode: str,
    vi_c: float,
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
    backend: str,
    u_avg: float,
    kappa_inv: float,
    phi_b: float,
    rho_f: float,
    q: int,
    gamma_u: float,
    u_extension: str,
    gamma_u_pin: float,
    kinematics_scale: float,
    v_supg: float,
    v_supg_mode: str,
    v_supg_c_nu: float,
    u_supg: float,
    u_cip: float,
    u_cip_weight: str,
    v_cip: float,
    vS_cip: float,
    alpha_ch_eps: float,
    scale_alpha_ch_eps_with_zeta: bool,
    diffuse_shear_scale_ref: float,
    refine_biofilm: bool,
    refine_band: float,
    refine_expand_layers: int,
    gamma_div: float,
    adaptive_gamma_div: bool,
    gamma_div_max: float,
    newton_tol: float,
    max_it: int,
    dt_min: float,
    accept_nonconverged_atol_factor: float,
    restart_from: str,
    restart_write_every: int,
    restart_dt: float,
    dt: float,
    t_final: float,
    t_ramp: float,
    snapshot_times: str,
    vtk_every: int,
    global_front_quantile: float,
    dx_quantile: float,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(BLAUERT_DRIVER),
        "--transport-mode",
        "pde",
        "--diffuse-shear-traction",
        "--diffuse-shear-model",
        str(diffuse_shear_model),
        "--diffuse-shear-scale",
        str(float(diffuse_shear_scale)),
        "--diffuse-shear-time-scheme",
        str(diffuse_shear_time_scheme),
        "--diffuse-shear-ramp-time",
        str(float(diffuse_shear_ramp_time)),
        "--nonlinear-solver",
        str(nonlinear_solver),
        "--ls-mode",
        str(ls_mode),
        "--vi-c",
        str(float(vi_c)),
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
        "--backend",
        str(backend),
        "--nx",
        str(int(nx)),
        "--ny",
        str(int(ny)),
        "--dt",
        str(float(dt)),
        "--t-final",
        str(float(t_final)),
        "--theta",
        "1.0",
        "--q",
        str(int(q)),
        "--t-ramp",
        str(float(t_ramp)),
        "--E",
        str(float(E)),
        "--nu",
        "0.4",
        "--solid-visco-eta",
        str(float(solid_visco_eta)),
        "--u-avg",
        str(float(u_avg)),
        "--rho-f",
        str(float(rho_f)),
        "--kappa-inv",
        str(float(kappa_inv)),
        "--phi-b",
        str(float(phi_b)),
        "--gamma-u",
        str(float(gamma_u)),
        "--u-extension",
        str(u_extension),
        "--gamma-u-pin",
        str(float(gamma_u_pin)),
        "--kinematics-scale",
        str(float(kinematics_scale)),
        "--alpha-ch-M",
        "1e-12",
        "--alpha-ch-gamma",
        "2e-3",
        "--alpha-ch-eps",
        str(float(alpha_ch_eps)),
        "--diffuse-shear-scale-ref",
        str(float(diffuse_shear_scale_ref)),
        "--alpha-advection-form",
        "conservative_weak",
        "--alpha-advect-with",
        "biofilm_volume",
        "--support-physics",
        "internal_conversion",
        "--alpha-supg",
        "0.5",
        "--alpha-cip",
        "0.0",
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
        "--global-front-quantile",
        str(float(global_front_quantile)),
        "--dx-quantile",
        str(float(dx_quantile)),
        "--gamma-div",
        str(float(gamma_div)),
        "--gamma-div-max",
        str(float(gamma_div_max)),
        "--newton-tol",
        str(float(newton_tol)),
        "--max-it",
        str(int(max_it)),
        "--allow-dt-reduction",
        "--dt-min",
        str(float(dt_min)),
        "--accept-nonconverged-atol-factor",
        str(float(accept_nonconverged_atol_factor)),
        "--restart-write-every",
        str(int(restart_write_every)),
        "--alpha0-file",
        str(POLY_CSV),
        "--alpha0-scale",
        "1e-3",
        "--alpha0-tx",
        "5e-4",
        "--alpha0-ty",
        "0.0",
        "--snapshot-times",
        str(snapshot_times),
        "--vtk-every",
        str(int(vtk_every)),
        "--out-dir",
        str(case_dir),
    ]
    cmd.append("--adaptive-gamma-div" if bool(adaptive_gamma_div) else "--no-adaptive-gamma-div")
    cmd.append("--scale-alpha-ch-eps-with-zeta" if bool(scale_alpha_ch_eps_with_zeta) else "--no-scale-alpha-ch-eps-with-zeta")
    if bool(refine_biofilm):
        cmd.extend(
            [
                "--refine-biofilm",
                "--refine-band",
                str(float(refine_band)),
                "--refine-expand-layers",
                str(int(refine_expand_layers)),
            ]
        )
    if str(restart_from).strip():
        cmd.extend(["--restart-from", str(restart_from).strip()])
    if np.isfinite(float(restart_dt)):
        cmd.extend(["--restart-dt", str(float(restart_dt))])
    return cmd


def _compare_observation_scenario(
    case_dir: Path,
    *,
    scenario: str,
    steady_time: float,
    skip_existing: bool = False,
    stream_subprocess: bool = False,
) -> dict[str, object]:
    out_json = case_dir / f"compare_{str(scenario)}.json"
    cmd = [
        sys.executable,
        "-u",
        str(OBS_COMPARE_SCRIPT),
        "--out-dir",
        str(case_dir),
        "--scenario",
        str(scenario),
        "--steady-time",
        str(float(steady_time)),
        "--json-out",
        str(out_json),
    ]
    _run(
        cmd,
        cwd=_REPO_ROOT,
        log_path=case_dir / f"compare_{str(scenario)}.log",
        skip_existing=False,
        sentinel=out_json,
        stream=bool(stream_subprocess),
    )
    return json.loads(out_json.read_text(encoding="utf-8"))


def _obs_value(payload: dict[str, object], section: str, key: str) -> float:
    block = payload.get(str(section), {})
    if not isinstance(block, dict):
        return float("nan")
    try:
        return float(block.get(str(key), float("nan")))
    except Exception:
        return float("nan")


def _calibration_score(
    *,
    steady_profile_rmse_um: float,
    steady_front_y150_err_um: float,
    front_compression_2p0_err_um: float,
    front_plateau_drift_2p0_10p0_um: float,
    porosity_drop_2p0_err_pp: float,
) -> float:
    weighted: list[tuple[float, float]] = []
    if np.isfinite(steady_profile_rmse_um):
        weighted.append((0.45, steady_profile_rmse_um / 100.0))
    if np.isfinite(steady_front_y150_err_um):
        weighted.append((0.15, steady_front_y150_err_um / 100.0))
    if np.isfinite(front_compression_2p0_err_um):
        weighted.append((0.20, front_compression_2p0_err_um / 148.0))
    if np.isfinite(front_plateau_drift_2p0_10p0_um):
        weighted.append((0.10, front_plateau_drift_2p0_10p0_um / 148.0))
    if np.isfinite(porosity_drop_2p0_err_pp):
        weighted.append((0.10, porosity_drop_2p0_err_pp / 2.0))
    if not weighted:
        return float("inf")
    w_sum = float(sum(weight for weight, _ in weighted))
    return float(sum(weight * value for weight, value in weighted) / max(1.0e-14, w_sum))


def _required_calibration_metrics(
    *,
    obs_scenarios: list[str],
    steady_profile_rmse_um: float,
    steady_front_y150_err_um: float,
    front_compression_2p0_err_um: float,
    front_plateau_drift_2p0_10p0_um: float,
    porosity_drop_2p0_err_pp: float,
) -> dict[str, float]:
    required: dict[str, float] = {}
    if "steady_dian" in obs_scenarios:
        required["steady_profile_rmse_um"] = float(steady_profile_rmse_um)
        required["steady_front_y150_err_um"] = float(steady_front_y150_err_um)
    if "dynamic_08pa" in obs_scenarios:
        required["front_compression_2p0_err_um"] = float(front_compression_2p0_err_um)
        required["front_plateau_drift_2p0_10p0_um"] = float(front_plateau_drift_2p0_10p0_um)
        required["porosity_drop_2p0_err_pp"] = float(porosity_drop_2p0_err_pp)
    return required


def _validate_calibration_metrics(*, case_dir: Path, obs_scenarios: list[str], **metrics: float) -> None:
    required = _required_calibration_metrics(obs_scenarios=obs_scenarios, **metrics)
    missing = [name for name, value in required.items() if not np.isfinite(float(value))]
    if missing:
        raise RuntimeError(
            "Missing required calibration observables "
            f"({', '.join(missing)}) for requested scenarios {obs_scenarios}. "
            f"Case did not reach the required observation window; see {case_dir / 'run.log'}."
        )


def _compare_metrics(case_dir: Path, *, t_max: float, stream_subprocess: bool = False) -> dict[str, object]:
    out_json = case_dir / "compare_all.json"
    cmd = [
        sys.executable,
        "-u",
        str(COMPARE_SCRIPT),
        "--exp-csv",
        str(EXP_CSV),
        "--out-dir",
        str(case_dir),
        "--t-max",
        str(float(t_max)),
        "--compare",
        "all",
        "--json-out",
        str(out_json),
    ]
    _run(cmd, cwd=_REPO_ROOT, log_path=case_dir / "compare.log", sentinel=out_json, stream=bool(stream_subprocess))
    return json.loads(out_json.read_text(encoding="utf-8"))


def _mean_per_y_rmse(compare_data: dict[str, object]) -> float:
    per_y = compare_data.get("per_y", {})
    vals: list[float] = []
    if isinstance(per_y, dict):
        for payload in per_y.values():
            if isinstance(payload, dict):
                try:
                    vals.append(float(payload.get("rmse", float("nan"))))
                except Exception:
                    continue
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _per_y_rmse(compare_data: dict[str, object], key: str) -> float:
    per_y = compare_data.get("per_y", {})
    if not isinstance(per_y, dict):
        return float("nan")
    payload = per_y.get(str(key), {})
    if not isinstance(payload, dict):
        return float("nan")
    try:
        return float(payload.get("rmse", float("nan")))
    except Exception:
        return float("nan")


def _otsu_threshold_u8(gray: np.ndarray) -> float:
    arr = np.asarray(gray, dtype=np.uint8).ravel()
    if arr.size == 0:
        return 0.0
    hist = np.bincount(arr, minlength=256).astype(float)
    total = float(arr.size)
    if total <= 0.0:
        return 0.0
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256, dtype=float))
    mu_t = float(mu[-1])
    denom = omega * (1.0 - omega)
    numer = (mu_t * omega - mu) ** 2
    sigma_b = np.zeros_like(numer)
    good = denom > 1.0e-15
    sigma_b[good] = numer[good] / denom[good]
    return float(np.argmax(sigma_b))


def _binary_component_slices(mask: np.ndarray) -> list[tuple[tuple[int, int, int, int], int]]:
    from scipy import ndimage

    arr = np.asarray(mask, dtype=bool)
    labels, nlab = ndimage.label(arr)
    if int(nlab) <= 0:
        return []
    out: list[tuple[tuple[int, int, int, int], int]] = []
    objects = ndimage.find_objects(labels)
    for idx, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        ys, xs = slc
        y0 = int(ys.start)
        y1 = int(ys.stop)
        x0 = int(xs.start)
        x1 = int(xs.stop)
        area = int(np.count_nonzero(labels[slc] == idx))
        out.append(((x0, y0, x1, y1), area))
    return out


def _disk_structure(radius: int) -> np.ndarray:
    r = max(0, int(radius))
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy) <= (r * r)


def _ffmpeg_extract_frame_png(*, t_s: float, out_png: Path, force: bool = False, stream_subprocess: bool = False) -> None:
    if out_png.exists() and not force:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{float(t_s):.6f}",
        "-i",
        str(VIDEO_PATH),
        "-frames:v",
        "1",
        str(out_png),
    ]
    _run(cmd, cwd=_REPO_ROOT, log_path=out_png.with_suffix(".log"), sentinel=out_png, stream=bool(stream_subprocess))


def _detect_scale_bar_px(gray: np.ndarray, *, scale_bar_um: float, thresh: int = 240, max_h_px: int = 35) -> tuple[int, float]:
    from scipy import ndimage

    g = np.asarray(gray, dtype=np.uint8)
    h, w = g.shape
    y0 = int(max(0, min(h - 1, math.floor(0.65 * float(h)))))
    x0 = int(max(0, min(w - 1, math.floor(0.55 * float(w)))))
    roi = g[y0:h, x0:w]
    bw = roi >= int(thresh)
    bw = ndimage.binary_closing(bw, structure=np.ones((3, 3), dtype=bool), iterations=2)

    best_w = 0
    for (rx0, ry0, rx1, ry1), area in _binary_component_slices(bw):
        wi = int(rx1 - rx0)
        hi = int(ry1 - ry0)
        if area < 200:
            continue
        if hi > int(max_h_px):
            continue
        if int(ry0) >= int(0.9 * float(bw.shape[0])):
            continue
        best_w = max(best_w, wi)
    if best_w <= 0:
        raise RuntimeError("Failed to detect scale bar in Blauert frame.")
    px_size_um = float(scale_bar_um) / float(best_w)
    return int(best_w), float(px_size_um)


def _detect_substrate_row(gray: np.ndarray, *, mean_thresh: float = 180.0, search_frac: float = 0.25) -> int:
    g = np.asarray(gray, dtype=np.uint8)
    h = int(g.shape[0])
    y_start = int(max(0, math.floor((1.0 - float(search_frac)) * float(h))))
    row_mean = g.mean(axis=1)
    y_bottom: int | None = None
    for y in range(h - 1, y_start - 1, -1):
        if float(row_mean[y]) >= float(mean_thresh):
            y_bottom = int(y)
            break
    if y_bottom is None:
        return int(h - 1)
    y_top = int(y_bottom)
    for y in range(y_bottom - 1, y_start - 1, -1):
        if float(row_mean[y]) >= float(mean_thresh):
            y_top = int(y)
        else:
            break
    return int(y_top)


def _overlay_rects_for_blauert_gray(gray: np.ndarray, *, scale_bar_pad_px: int = 12) -> list[tuple[int, int, int, int]]:
    from scipy import ndimage

    g = np.asarray(gray, dtype=np.uint8)
    h, w = g.shape
    rects: list[tuple[int, int, int, int]] = [
        (int(0.86 * w), 0, w, int(0.22 * h)),
        (int(0.72 * w), int(0.78 * h), w, int(0.96 * h)),
    ]

    roi_y0 = int(math.floor(0.65 * h))
    roi_x0 = int(math.floor(0.55 * w))
    roi = g[roi_y0:h, roi_x0:w]
    bw = roi >= 240
    bw = ndimage.binary_closing(bw, structure=np.ones((3, 3), dtype=bool), iterations=2)
    best: tuple[int, int, int, int] | None = None
    best_w = 0
    for (rx0, ry0, rx1, ry1), area in _binary_component_slices(bw):
        wi = int(rx1 - rx0)
        hi = int(ry1 - ry0)
        if hi > 35 or area < 200:
            continue
        if wi > best_w:
            best_w = wi
            best = (rx0, ry0, rx1, ry1)
    if best is not None:
        rx0, ry0, rx1, ry1 = best
        rects.append(
            (
                max(0, roi_x0 + rx0 - int(scale_bar_pad_px)),
                max(0, roi_y0 + ry0 - int(scale_bar_pad_px)),
                min(w, roi_x0 + rx1 + int(scale_bar_pad_px)),
                min(h, roi_y0 + ry1 + int(scale_bar_pad_px) + 60),
            )
        )

    uniq: set[tuple[int, int, int, int]] = set()
    for x0, y0, x1, y1 in rects:
        x0 = max(0, min(int(w), int(x0)))
        x1 = max(0, min(int(w), int(x1)))
        y0 = max(0, min(int(h), int(y0)))
        y1 = max(0, min(int(h), int(y1)))
        if x1 > x0 and y1 > y0:
            uniq.add((x0, y0, x1, y1))
    return sorted(uniq)


def _segment_biofilm_gray(
    gray: np.ndarray,
    *,
    bottom_trim_px: int = 6,
    blur_sigma: float = 2.0,
    close_radius_px: int = 7,
    close_iters: int = 1,
    fill_holes: bool = True,
    min_area_px: int = 5000,
    y_base_override: int | None = None,
    fixed_threshold: float | None = None,
    overlay_rects: list[tuple[int, int, int, int]] | None = None,
    overlay_thresh: int = 240,
) -> tuple[np.ndarray, int]:
    from scipy import ndimage

    g = np.asarray(gray, dtype=np.uint8)
    overlay_mask = np.zeros_like(g, dtype=bool)
    if overlay_rects:
        for x0, y0, x1, y1 in overlay_rects:
            x0 = max(0, min(int(g.shape[1]), int(x0)))
            x1 = max(0, min(int(g.shape[1]), int(x1)))
            y0 = max(0, min(int(g.shape[0]), int(y0)))
            y1 = max(0, min(int(g.shape[0]), int(y1)))
            if x1 <= x0 or y1 <= y0:
                continue
            overlay_mask[y0:y1, x0:x1] |= g[y0:y1, x0:x1] >= int(overlay_thresh)

    y_base = int(y_base_override) if y_base_override is not None else _detect_substrate_row(g)
    y_cut = max(1, int(y_base) - int(max(0, int(bottom_trim_px))))
    work = np.asarray(g[:y_cut, :], dtype=float)
    if float(blur_sigma) > 0.0:
        work = ndimage.gaussian_filter(work, sigma=float(blur_sigma))

    thresh = _otsu_threshold_u8(np.clip(work, 0.0, 255.0).astype(np.uint8)) if fixed_threshold is None else float(fixed_threshold)
    bw = work >= float(thresh)
    if overlay_rects:
        bw[np.asarray(overlay_mask[:y_cut, :], dtype=bool)] = False

    if int(close_radius_px) > 0 and int(close_iters) > 0:
        bw = ndimage.binary_closing(
            bw,
            structure=_disk_structure(int(close_radius_px)),
            iterations=int(close_iters),
        )
    if bool(fill_holes):
        bw = ndimage.binary_fill_holes(bw)
    if overlay_rects:
        bw[np.asarray(overlay_mask[:y_cut, :], dtype=bool)] = False

    labels, nlab = ndimage.label(np.asarray(bw, dtype=bool))
    if int(nlab) <= 0:
        return np.zeros((y_cut, g.shape[1]), dtype=np.uint8), int(y_base)
    best_idx = 0
    best_area = 0
    for idx in range(1, int(nlab) + 1):
        area = int(np.count_nonzero(labels == idx))
        if area > best_area:
            best_area = area
            best_idx = idx
    if best_area < int(min_area_px) or best_idx <= 0:
        return np.zeros((y_cut, g.shape[1]), dtype=np.uint8), int(y_base)
    out = np.zeros((y_cut, g.shape[1]), dtype=np.uint8)
    out[labels == best_idx] = 255
    return out, int(y_base)


def _polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _mask_contour_polygon_mm(mask_u8: np.ndarray, *, y_base: int, px_size_um: float) -> np.ndarray:
    if plt is None:
        raise RuntimeError("matplotlib is required for Benchmark 6 contour extraction.")

    m = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(float)
    if not np.any(m > 0.0):
        raise RuntimeError("Cannot build contour polygon from an empty mask.")
    yy, xx = np.mgrid[0 : m.shape[0], 0 : m.shape[1]]
    fig, ax = plt.subplots()
    try:
        cs = ax.contour(xx, yy, m, levels=[0.5])
        segs = [np.asarray(seg, dtype=float) for seg in cs.allsegs[0] if np.asarray(seg).shape[0] >= 3]
    finally:
        plt.close(fig)
    if not segs:
        raise RuntimeError("matplotlib contour extraction failed for Benchmark 6 frame.")
    pts = max(segs, key=lambda seg: abs(_polygon_area(seg)))
    if not np.allclose(pts[0], pts[-1], rtol=0.0, atol=1.0e-12):
        pts = np.vstack([pts, pts[0]])
    x_mm = (pts[:, 0] * float(px_size_um)) * 1.0e-3
    y_mm = ((float(y_base) - pts[:, 1]) * float(px_size_um)) * 1.0e-3
    return np.column_stack([x_mm, y_mm]).astype(float)


def _extract_video_contours(times: list[float], *, outdir: Path, force: bool = False) -> dict[float, Path]:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - exercised in the solver env
        raise RuntimeError("Pillow is required for Benchmark 6 contour extraction.") from exc

    outdir.mkdir(parents=True, exist_ok=True)
    frame_dir = outdir / "_frames"
    frame0_png = frame_dir / "frame_t0.000.png"
    _ffmpeg_extract_frame_png(t_s=0.0, out_png=frame0_png, force=force)
    gray0 = np.asarray(Image.open(frame0_png).convert("L"), dtype=np.uint8)
    _, px_size_um = _detect_scale_bar_px(gray0, scale_bar_um=250.0)
    overlay_rects = _overlay_rects_for_blauert_gray(gray0)
    y_base0 = _detect_substrate_row(gray0)
    y_cut0 = max(1, int(y_base0) - 6)
    fixed_thr0 = _otsu_threshold_u8(gray0[:y_cut0, :])
    contour_roi = (
        _contour_roi_from_polygon_mm(
            POLY_CSV,
            pad_right_mm=float(CONTOUR_ROI_PAD_RIGHT_MM),
            pad_top_mm=float(CONTOUR_ROI_PAD_TOP_MM),
        )
        if POLY_CSV.exists()
        else None
    )

    outputs: dict[float, Path] = {}
    for target in times:
        t_now = float(target)
        out_csv = outdir / f"video_contour_t{t_now:04.1f}s.csv"
        outputs[t_now] = out_csv
        if out_csv.exists() and not force:
            continue
        frame_png = frame_dir / f"frame_t{t_now:06.3f}.png"
        _ffmpeg_extract_frame_png(t_s=t_now, out_png=frame_png, force=force)
        gray = np.asarray(Image.open(frame_png).convert("L"), dtype=np.uint8)
        mask_u8, y_base = _segment_biofilm_gray(
            gray,
            bottom_trim_px=6,
            blur_sigma=2.0,
            close_radius_px=7,
            close_iters=1,
            fill_holes=True,
            min_area_px=5000,
            y_base_override=int(y_base0),
            fixed_threshold=float(fixed_thr0),
            overlay_rects=overlay_rects,
        )
        if contour_roi is not None:
            x_min_mm, x_max_mm, y_min_mm, y_max_mm = contour_roi
            mask_u8 = _crop_mask_to_mm_roi(
                mask_u8,
                y_base=int(y_base),
                px_size_um=float(px_size_um),
                x_min_mm=float(x_min_mm),
                x_max_mm=float(x_max_mm),
                y_min_mm=float(y_min_mm),
                y_max_mm=float(y_max_mm),
                min_area_px=5000,
            )
        poly_mm = _mask_contour_polygon_mm(mask_u8, y_base=int(y_base), px_size_um=float(px_size_um))
        with out_csv.open("w", encoding="utf-8") as f:
            f.write("x_mm,y_mm\n")
            for xx, yy in np.asarray(poly_mm, dtype=float):
                f.write(f"{float(xx):.9f},{float(yy):.9f}\n")
    return outputs


def _read_video_contour_mm(path: Path) -> np.ndarray:
    arr = np.genfromtxt(str(path), delimiter=",", skip_header=1, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return np.asarray(arr[:, :2], dtype=float)


def _read_sim_contour_mm(path: Path, *, x_shift_mm: float) -> list[np.ndarray]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    groups: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        cid = int(row["contour_id"])
        xx = 1.0e3 * float(row["x_m"]) - float(x_shift_mm)
        yy = 1.0e3 * float(row["y_m"])
        groups.setdefault(cid, []).append((xx, yy))
    return [np.asarray(groups[k], dtype=float) for k in sorted(groups)]


def _segment_intersections_x(points: np.ndarray, y_sample: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return np.empty((0,), dtype=float)
    out: list[float] = []
    for i in range(pts.shape[0] - 1):
        x0, y0 = float(pts[i, 0]), float(pts[i, 1])
        x1, y1 = float(pts[i + 1, 0]), float(pts[i + 1, 1])
        if not ((y0 <= y_sample <= y1) or (y1 <= y_sample <= y0)):
            continue
        dy = y1 - y0
        if abs(dy) <= 1.0e-14:
            out.extend([x0, x1])
            continue
        tau = (float(y_sample) - y0) / dy
        if -1.0e-12 <= tau <= 1.0 + 1.0e-12:
            out.append(x0 + tau * (x1 - x0))
    return np.asarray(out, dtype=float)


def _front_profile(points_list: list[np.ndarray], y_samples_mm: np.ndarray) -> np.ndarray:
    ys = np.asarray(y_samples_mm, dtype=float).ravel()
    out = np.full(ys.shape, float("nan"), dtype=float)
    for j, yy in enumerate(ys):
        xs_all: list[float] = []
        for pts in points_list:
            xs = _segment_intersections_x(pts, float(yy))
            if xs.size:
                xs_all.extend(float(v) for v in xs)
        if xs_all:
            out[j] = float(np.min(np.asarray(xs_all, dtype=float)))
    return out


def _contour_profile_metrics(
    *,
    exp_points: list[np.ndarray],
    sim_points: list[np.ndarray],
    y_min_mm: float = 0.02,
    y_max_mm: float = 0.42,
    n_samples: int = 120,
) -> dict[str, float]:
    ys = np.linspace(float(y_min_mm), float(y_max_mm), int(n_samples), dtype=float)
    x_exp = _front_profile(exp_points, ys)
    x_sim = _front_profile(sim_points, ys)
    mask = np.isfinite(x_exp) & np.isfinite(x_sim)
    if not np.any(mask):
        return {"n": 0, "mae_um": float("nan"), "rmse_um": float("nan"), "max_um": float("nan")}
    err_um = 1.0e3 * (x_sim[mask] - x_exp[mask])
    return {
        "n": int(np.sum(mask)),
        "mae_um": float(np.mean(np.abs(err_um))),
        "rmse_um": float(np.sqrt(np.mean(err_um**2))),
        "max_um": float(np.max(np.abs(err_um))),
    }


def _history_point_metrics(
    *,
    exp_csv: Path,
    sim_csv: Path,
    times_s: list[float],
    exp_key: str = "dx_front_um",
    sim_key: str = "dx_front_global",
    sim_scale: float = 1.0e6,
) -> dict[float, dict[str, float]]:
    exp = _read_csv_columns(exp_csv)
    sim = _read_csv_columns(sim_csv)
    if "t_s" not in exp or "t_s" not in sim or exp_key not in exp or sim_key not in sim:
        return {}
    t_exp = np.asarray(exp["t_s"], dtype=float).ravel()
    t_sim = np.asarray(sim["t_s"], dtype=float).ravel()
    y_exp = np.asarray(exp[exp_key], dtype=float).ravel()
    y_sim = float(sim_scale) * np.asarray(sim[sim_key], dtype=float).ravel()
    out: dict[float, dict[str, float]] = {}
    for t_now in times_s:
        tt = float(t_now)
        exp_val = float(_interp_1d(t_exp, y_exp, np.asarray([tt], dtype=float))[0])
        sim_val = float(_interp_1d(t_sim, y_sim, np.asarray([tt], dtype=float))[0])
        out[tt] = {
            "video_um": exp_val,
            "sim_um": sim_val,
            "abs_err_um": abs(sim_val - exp_val) if np.isfinite(exp_val) and np.isfinite(sim_val) else float("nan"),
        }
    return out


def _nearest_sim_y_key(sim_csv: Path, target_um: int) -> str | None:
    sim = _read_csv_columns(sim_csv)
    keys: list[tuple[int, str]] = []
    for key in sim:
        if not str(key).startswith("dx_front_y") or not str(key).endswith("um"):
            continue
        body = str(key)[len("dx_front_y") : -len("um")]
        try:
            keys.append((int(body), str(key)))
        except Exception:
            continue
    if not keys:
        return None
    return min(keys, key=lambda item: abs(int(item[0]) - int(target_um)))[1]


def _write_csv(path: Path, rows: list[dict[str, object]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _snapshot_path_for_time(case_dir: Path, t_snap: float) -> Path | None:
    candidates = sorted((case_dir / "snapshots").glob(f"*t{float(t_snap):06.3f}_alpha05.csv"))
    return candidates[0] if candidates else None


def _simulation_artifacts_ready(case_dir: Path, *, required_snapshot_times: list[float]) -> bool:
    if not (case_dir / "timeseries.csv").exists():
        return False
    for t_snap in required_snapshot_times:
        if _snapshot_path_for_time(case_dir, float(t_snap)) is None:
            return False
    return True


def _plot_history(*, sim_csv: Path, out_path: Path, include_dynamic_targets: bool) -> None:
    if plt is None:
        return
    sim = _read_csv_columns(sim_csv)
    if "t_s" not in sim or "dx_front_global" not in sim:
        return
    t_sim = np.asarray(sim["t_s"], dtype=float).ravel()
    dx_sim = 1.0e6 * np.asarray(sim["dx_front_global"], dtype=float).ravel()
    phi_sim = 100.0 * np.asarray(sim.get("phi_mean_alpha_weighted", np.full_like(t_sim, np.nan)), dtype=float).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
    axes[0].plot(t_sim, dx_sim, color="#7a1f36", linewidth=1.8, label="one-domain")
    if include_dynamic_targets:
        axes[0].scatter([2.0], [148.0], color="black", s=28, zorder=5, label="Blauert at 2 s")
    if include_dynamic_targets and np.nanmax(t_sim) >= 2.0:
        axes[0].hlines(148.0, 2.0, float(np.nanmax(t_sim)), colors="black", linestyles="--", linewidth=1.1, label="reported plateau")
    axes[0].set_title("Front compression")
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("dx [um]")
    axes[0].grid(True, linestyle=":", linewidth=0.6)

    axes[1].plot(t_sim, phi_sim, color="#194a6a", linewidth=1.8, label="one-domain")
    if include_dynamic_targets:
        axes[1].scatter([0.0, 2.0], [47.0, 45.0], color="black", s=28, zorder=5, label="Blauert porosity")
    if include_dynamic_targets and np.nanmax(t_sim) >= 2.0:
        axes[1].hlines(45.0, 2.0, float(np.nanmax(t_sim)), colors="black", linestyles="--", linewidth=1.1)
    axes[1].set_title("Mean porosity")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("porosity [%]")
    axes[1].grid(True, linestyle=":", linewidth=0.6)

    axes[0].legend(frameon=False, loc="best")
    axes[1].legend(frameon=False, loc="best")
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def _plot_contours(
    *,
    sim_snapshot_path: Path,
    out_path: Path,
    steady_time: float,
) -> None:
    if plt is None:
        return
    exp_pts = _svg_contour_to_mm(svg_path=STEADY_SVG, domain_path=STEADY_DOMAIN, L_mm=5.5, H_mm=1.0)
    sim_pts = _read_sim_contour_mm(sim_snapshot_path, x_shift_mm=0.0)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.4), constrained_layout=True)
    ax.plot(exp_pts[:, 0], exp_pts[:, 1], color="black", linewidth=1.8, label="experimental contour")
    for j, pts in enumerate(sim_pts):
        ax.plot(pts[:, 0], pts[:, 1], color="#7a1f36", linewidth=1.5, alpha=0.95, label="one-domain contour" if j == 0 else None)
    ax.set_title(f"Steady contour comparison at t={float(steady_time):.1f} s")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(frameon=False, loc="best")
    fig.savefig(out_path, dpi=280)
    plt.close(fig)


def _plot_mesh_sensitivity(rows: list[dict[str, object]], *, out_path: Path) -> None:
    if plt is None or not rows:
        return
    nx = np.asarray([int(row["nx"]) for row in rows], dtype=float)
    steady = np.asarray([float(row.get("steady_profile_rmse_um", float("nan"))) for row in rows], dtype=float)
    front = np.asarray([float(row.get("steady_front_y150_err_um", float("nan"))) for row in rows], dtype=float)
    comp = np.asarray([float(row.get("front_compression_2p0_err_um", float("nan"))) for row in rows], dtype=float)
    phi = np.asarray([float(row.get("porosity_drop_2p0_err_pp", float("nan"))) for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    axes[0].plot(nx, steady, marker="o", color="#194a6a", linewidth=1.6)
    axes[0].set_xlabel(r"$n_x$")
    axes[0].set_ylabel("steady contour RMSE [um]")
    axes[0].grid(True, linestyle=":", linewidth=0.6)
    axes[0].set_title("Steady contour")
    axes[1].plot(nx, front, marker="o", color="#7a1f36", linewidth=1.6, label="front-point error")
    if np.any(np.isfinite(comp)):
        axes[1].plot(nx, comp, marker="s", color="#c47a2c", linewidth=1.6, label="compression error")
    if np.any(np.isfinite(phi)):
        axes[1].plot(nx, phi, marker="^", color="#4f7b39", linewidth=1.6, label="porosity error")
    axes[1].set_xlabel(r"$n_x$")
    axes[1].set_ylabel("observable error")
    axes[1].grid(True, linestyle=":", linewidth=0.6)
    axes[1].set_title("Dynamic observations")
    axes[1].legend(frameon=False, loc="best")
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=str, default="examples/biofilms/results/benchmark6_blauert_channel")
    ap.add_argument("--profile", type=str, default="baseline", choices=("smoke", "baseline", "production"))
    ap.add_argument("--E-list", type=str, default="")
    ap.add_argument("--eta-list", type=str, default="")
    ap.add_argument("--diffuse-shear-scale-list", type=str, default="")
    ap.add_argument(
        "--diffuse-shear-model",
        type=str,
        default="poiseuille",
        choices=("lagged_velocity", "lagged_stress", "poiseuille"),
    )
    ap.add_argument("--diffuse-shear-time-scheme", type=str, default="imex", choices=("constant", "imex"))
    ap.add_argument("--diffuse-shear-ramp-time", type=float, default=float("nan"))
    ap.add_argument("--nonlinear-solver", type=str, default="pdas", choices=("pdas", "newton", "snes"))
    ap.add_argument("--ls-mode", type=str, default="dealii", choices=("armijo", "dealii"))
    ap.add_argument("--vi-c", type=float, default=0.0)
    ap.add_argument("--vi-enter-tol", type=float, default=0.0)
    ap.add_argument("--vi-leave-tol", type=float, default=0.0)
    ap.add_argument("--vi-persistence", type=int, default=0)
    ap.add_argument("--vi-lambda0", type=float, default=0.0)
    ap.add_argument("--vi-lambda-max", type=float, default=1.0e6)
    ap.add_argument("--vi-lambda-growth", type=float, default=5.0)
    ap.add_argument("--vi-lambda-decay", type=float, default=0.5)
    ap.add_argument("--vi-active-soft-threshold", type=int, default=0)
    ap.add_argument("--vi-active-soft-alpha", type=float, default=1.0)
    ap.add_argument("--vi-active-strong-factor", type=float, default=5.0)
    ap.add_argument("--vi-filter-max-delta-active", type=int, default=0)
    ap.add_argument("--vi-unconstrained-lm", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--vi-lm-lambda0", type=float, default=1.0e-4)
    ap.add_argument("--vi-lm-lambda-max", type=float, default=1.0e6)
    ap.add_argument("--vi-lm-growth", type=float, default=5.0)
    ap.add_argument("--vi-lm-decay", type=float, default=0.5)
    ap.add_argument("--vi-lm-accept-ratio", type=float, default=1.0e-3)
    ap.add_argument("--vi-lm-good-ratio", type=float, default=0.75)
    ap.add_argument("--vi-lm-max-tries", type=int, default=6)
    ap.add_argument("--linear-ksp-type", type=str, default="")
    ap.add_argument("--linear-pc-type", type=str, default="")
    ap.add_argument("--linear-pc-factor-solver-type", type=str, default="")
    ap.add_argument("--linear-ksp-rtol", type=float, default=1.0e-8)
    ap.add_argument("--linear-ksp-max-it", type=int, default=200)
    ap.add_argument("--linear-ksp-trace", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--linear-schur", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--linear-schur-pressure-field", type=str, default="p")
    ap.add_argument("--linear-schur-fact", type=str, default="full", choices=("full", "upper", "lower", "diag"))
    ap.add_argument("--linear-schur-pre", type=str, default="selfp", choices=("selfp", "a11", "user"))
    ap.add_argument("--linear-schur-rest-ksp", type=str, default="preonly")
    ap.add_argument("--linear-schur-rest-pc", type=str, default="ilu")
    ap.add_argument("--linear-schur-rest-factor-solver-type", type=str, default="")
    ap.add_argument("--linear-schur-pressure-ksp", type=str, default="preonly")
    ap.add_argument("--linear-schur-pressure-pc", type=str, default="jacobi")
    ap.add_argument("--linear-schur-pressure-factor-solver-type", type=str, default="")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--nx-list", type=str, default="")
    ap.add_argument("--u-avg", type=float, default=4.56e-2)
    ap.add_argument("--rho-f", type=float, default=0.0)
    ap.add_argument("--kappa-inv", type=float, default=9.81e11)
    ap.add_argument("--phi-b", type=float, default=0.47)
    ap.add_argument("--q", type=int, default=4)
    ap.add_argument("--gamma-u", type=float, default=5.0)
    ap.add_argument("--u-extension", type=str, default="l2", choices=("l2", "grad"))
    ap.add_argument("--gamma-u-pin", type=float, default=1.0e-4)
    ap.add_argument(
        "--kinematics-scale",
        type=float,
        default=float("nan"),
        help="Scaling applied to the kinematic constraint when forwarding to the Blauert driver.",
    )
    ap.add_argument(
        "--v-supg",
        type=float,
        default=0.0,
        help="Fluid-momentum SUPG-like streamline diffusion forwarded to the Blauert driver.",
    )
    ap.add_argument(
        "--v-supg-mode",
        type=str,
        default="streamline",
        choices=("streamline", "residual"),
        help="Fluid momentum stabilization form forwarded to the Blauert driver.",
    )
    ap.add_argument(
        "--v-supg-c-nu",
        type=float,
        default=4.0,
        help="Viscous constant c_nu in the Green's-function elemental tau for fluid SUPG.",
    )
    ap.add_argument("--u-supg", type=float, default=0.0, help="SUPG strength for the kinematic u-transport equation.")
    ap.add_argument("--u-cip", type=float, default=0.0, help="CIP strength for the kinematic u-transport equation.")
    ap.add_argument(
        "--u-cip-weight",
        type=str,
        default="biofilm",
        choices=("fluid", "biofilm", "both"),
        help="Localization used by --u-cip in the kinematic equation.",
    )
    ap.add_argument(
        "--v-cip",
        type=float,
        default=0.0,
        help="Continuous-interior-penalty stabilization strength for fluid velocity.",
    )
    ap.add_argument(
        "--vS-cip",
        type=float,
        default=0.0,
        help="Continuous-interior-penalty stabilization strength for skeleton velocity.",
    )
    ap.add_argument("--alpha-ch-eps", type=float, default=2.0e-5)
    ap.add_argument(
        "--scale-alpha-ch-eps-with-zeta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale alpha CH eps like zeta^2 above --diffuse-shear-scale-ref.",
    )
    ap.add_argument(
        "--diffuse-shear-scale-ref",
        type=float,
        default=50.0,
        help="Reference zeta used by --scale-alpha-ch-eps-with-zeta.",
    )
    ap.add_argument("--refine-biofilm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--refine-band", type=float, default=2.5e-4)
    ap.add_argument("--refine-expand-layers", type=int, default=1)
    ap.add_argument(
        "--gamma-div",
        type=float,
        default=5.0e-2,
        help="Consistent mixed-block grad-div / augmented-Lagrangian strength.",
    )
    ap.add_argument(
        "--adaptive-gamma-div",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the Benchmark 6 adaptive gamma_div retry/relaxation strategy.",
    )
    ap.add_argument(
        "--gamma-div-max",
        type=float,
        default=1.0e-1,
        help="Upper cap forwarded to the driver's adaptive gamma_div controller.",
    )
    ap.add_argument(
        "--vtk-every",
        type=int,
        default=-1,
        help=(
            "Override VTK cadence for all runs. "
            "Negative keeps the benchmark defaults (calibration off, finest production mesh every 10 steps)."
        ),
    )
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--dt-min", type=float, default=5.0e-3)
    ap.add_argument("--restart-from", type=str, default="")
    ap.add_argument("--restart-write-every", type=int, default=1)
    ap.add_argument("--restart-dt", type=float, default=float("nan"))
    ap.add_argument("--t-final", type=float, default=4.0)
    ap.add_argument("--t-ramp", type=float, default=2.0)
    ap.add_argument("--compare-t-max", type=float, default=4.0)
    ap.add_argument("--snapshot-times", type=str, default="")
    ap.add_argument("--steady-time", type=float, default=float("nan"))
    ap.add_argument("--observation-scenarios", type=str, default="steady_dian")
    ap.add_argument("--global-front-quantile", type=float, default=0.005)
    ap.add_argument("--dx-quantile", type=float, default=0.05)
    ap.add_argument("--newton-tol", type=float, default=1.0e-6)
    ap.add_argument("--max-it", type=int, default=25)
    ap.add_argument("--accept-nonconverged-atol-factor", type=float, default=3.0)
    ap.add_argument(
        "--continue-on-candidate-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue the calibration sweep when an individual candidate run fails, and record the failed case in JSON.",
    )
    ap.add_argument(
        "--stream-subprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stream child simulation/comparison stdout to the terminal while still writing the per-case log files.",
    )
    ap.add_argument("--force-exp", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--calibration-only", action="store_true")
    args = ap.parse_args()

    outdir = (_REPO_ROOT / str(args.outdir)).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if str(args.profile) == "smoke":
        E_list = _parse_float_list(str(args.E_list) or "200")
        eta_list = _parse_float_list(str(args.eta_list) or "0")
        shear_scale_list = _parse_float_list(str(args.diffuse_shear_scale_list) or "30,50")
        nx_list = _parse_int_list(str(args.nx_list) or "24")
    elif str(args.profile) == "production":
        E_list = _parse_float_list(str(args.E_list) or "120,200,320,500")
        eta_list = _parse_float_list(str(args.eta_list) or "0")
        shear_scale_list = _parse_float_list(str(args.diffuse_shear_scale_list) or "30,50,100")
        nx_list = _parse_int_list(str(args.nx_list) or "24,32,40")
    else:
        E_list = _parse_float_list(str(args.E_list) or "120,200,320")
        eta_list = _parse_float_list(str(args.eta_list) or "0")
        shear_scale_list = _parse_float_list(str(args.diffuse_shear_scale_list) or "30,50,100")
        nx_list = _parse_int_list(str(args.nx_list) or "24,32")

    obs_scenarios = [part.strip() for part in str(args.observation_scenarios).split(",") if part.strip()]
    if not obs_scenarios:
        raise ValueError("Expected at least one --observation-scenarios entry.")
    allowed_obs = {"steady_dian", "dynamic_08pa", "dynamic_164pa"}
    bad_obs = sorted(set(obs_scenarios) - allowed_obs)
    if bad_obs:
        raise ValueError(f"Unsupported observation scenarios: {', '.join(bad_obs)}")

    diffuse_shear_ramp_time = float(args.diffuse_shear_ramp_time)
    if not np.isfinite(diffuse_shear_ramp_time):
        diffuse_shear_ramp_time = float(args.t_ramp)

    steady_time = float(args.steady_time) if np.isfinite(float(args.steady_time)) else float(args.t_final)
    if "dynamic_08pa" in obs_scenarios and float(args.t_final) < 10.0 - 1.0e-12:
        raise ValueError("dynamic_08pa requires --t-final >= 10.0 s to check the reported plateau.")
    if "dynamic_164pa" in obs_scenarios and float(args.t_final) < 2.1 - 1.0e-12:
        raise ValueError("dynamic_164pa requires --t-final >= 2.1 s.")
    if "steady_dian" in obs_scenarios and float(args.t_final) + 1.0e-12 < float(steady_time):
        raise ValueError("--steady-time must not exceed --t-final.")

    required_snapshot_times = {0.0}
    if "steady_dian" in obs_scenarios:
        required_snapshot_times.add(float(steady_time))
    if "dynamic_164pa" in obs_scenarios:
        required_snapshot_times.update({0.4, 0.6, 1.3, 2.1})
    snapshot_times = {float(v) for v in str(args.snapshot_times).split(",") if str(v).strip()}
    snapshot_times.update(required_snapshot_times)
    snapshot_times_sorted = sorted(snapshot_times)
    snapshot_tag = ",".join(f"{float(v):g}" for v in snapshot_times_sorted)
    cal_snapshot_tag = ",".join(f"{float(v):g}" for v in sorted(required_snapshot_times))

    _ensure_experimental_inputs(t_max=float(args.compare_t_max), y_levels_um="150,250,350", force=bool(args.force_exp))

    calibration_root = outdir / "calibration"
    calibration_rows: list[CalibrationRow] = []
    failed_calibration_cases: list[FailedCalibrationCase] = []
    vtk_every_override = int(args.vtk_every)
    nx_cal = int(min(nx_list))
    ny_cal = _ny_from_nx(nx_cal)
    for E in E_list:
        for eta_s in eta_list:
            for shear_scale in shear_scale_list:
                tag = _candidate_case_name(
                    E=float(E),
                    solid_visco_eta=float(eta_s),
                    diffuse_shear_scale=float(shear_scale),
                    nx=nx_cal,
                )
                case_dir = calibration_root / tag
                cmd = _simulation_command(
                    case_dir=case_dir,
                    nx=nx_cal,
                    ny=ny_cal,
                    E=float(E),
                    solid_visco_eta=float(eta_s),
                    diffuse_shear_scale=float(shear_scale),
                    diffuse_shear_model=str(args.diffuse_shear_model),
                    diffuse_shear_time_scheme=str(args.diffuse_shear_time_scheme),
                    diffuse_shear_ramp_time=float(diffuse_shear_ramp_time),
                    nonlinear_solver=str(args.nonlinear_solver),
                    ls_mode=str(args.ls_mode),
                    vi_c=float(args.vi_c),
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
                    backend=str(args.backend),
                    u_avg=float(args.u_avg),
                    kappa_inv=float(args.kappa_inv),
                    phi_b=float(args.phi_b),
                    rho_f=float(args.rho_f),
                    q=int(args.q),
                    gamma_u=float(args.gamma_u),
                    u_extension=str(args.u_extension),
                    gamma_u_pin=float(args.gamma_u_pin),
                    kinematics_scale=float(args.kinematics_scale),
                    v_supg=float(args.v_supg),
                    v_supg_mode=str(args.v_supg_mode),
                    v_supg_c_nu=float(args.v_supg_c_nu),
                    u_supg=float(args.u_supg),
                    u_cip=float(args.u_cip),
                    u_cip_weight=str(args.u_cip_weight),
                    v_cip=float(args.v_cip),
                    vS_cip=float(args.vS_cip),
                    alpha_ch_eps=float(args.alpha_ch_eps),
                    scale_alpha_ch_eps_with_zeta=bool(args.scale_alpha_ch_eps_with_zeta),
                    diffuse_shear_scale_ref=float(args.diffuse_shear_scale_ref),
                    refine_biofilm=bool(args.refine_biofilm),
                    refine_band=float(args.refine_band),
                    refine_expand_layers=int(args.refine_expand_layers),
                    gamma_div=float(args.gamma_div),
                    adaptive_gamma_div=bool(args.adaptive_gamma_div),
                    gamma_div_max=float(args.gamma_div_max),
                    newton_tol=float(args.newton_tol),
                    max_it=int(args.max_it),
                    dt_min=float(args.dt_min),
                    accept_nonconverged_atol_factor=float(args.accept_nonconverged_atol_factor),
                    restart_from=str(args.restart_from),
                    restart_write_every=int(args.restart_write_every),
                    restart_dt=float(args.restart_dt),
                    dt=float(args.dt),
                    t_final=float(args.t_final),
                    t_ramp=float(args.t_ramp),
                    snapshot_times=cal_snapshot_tag,
                    vtk_every=int(vtk_every_override) if vtk_every_override >= 0 else 0,
                    global_front_quantile=float(args.global_front_quantile),
                    dx_quantile=float(args.dx_quantile),
                )
                try:
                    _run(
                        cmd,
                        cwd=_REPO_ROOT,
                        log_path=case_dir / "run.log",
                        skip_existing=bool(args.skip_existing)
                        and _simulation_artifacts_ready(
                            case_dir,
                            required_snapshot_times=sorted(required_snapshot_times),
                        ),
                        sentinel=case_dir / "timeseries.csv",
                        stream=bool(args.stream_subprocess),
                    )
                    obs_payloads = {
                        scenario: _compare_observation_scenario(
                            case_dir,
                            scenario=scenario,
                            steady_time=float(steady_time),
                            skip_existing=bool(args.skip_existing),
                            stream_subprocess=bool(args.stream_subprocess),
                        )
                        for scenario in obs_scenarios
                    }
                    steady_profile_rmse = (
                        _obs_value(obs_payloads["steady_dian"], "abs_error", "steady_profile_rmse_um")
                        if "steady_dian" in obs_payloads
                        else float("nan")
                    )
                    steady_front_y150_err = (
                        _obs_value(obs_payloads["steady_dian"], "abs_error", "steady_front_y150_err_um")
                        if "steady_dian" in obs_payloads
                        else float("nan")
                    )
                    front_compression_2p0_err = (
                        _obs_value(obs_payloads["dynamic_08pa"], "abs_error", "front_compression_2p0_um")
                        if "dynamic_08pa" in obs_payloads
                        else float("nan")
                    )
                    front_plateau_drift_2p0_10p0 = (
                        _obs_value(obs_payloads["dynamic_08pa"], "abs_error", "front_plateau_drift_2p0_10p0_um")
                        if "dynamic_08pa" in obs_payloads
                        else float("nan")
                    )
                    porosity_drop_2p0_err = (
                        _obs_value(obs_payloads["dynamic_08pa"], "abs_error", "porosity_drop_2p0_pp")
                        if "dynamic_08pa" in obs_payloads
                        else float("nan")
                    )
                    score = _calibration_score(
                        steady_profile_rmse_um=float(steady_profile_rmse),
                        steady_front_y150_err_um=float(steady_front_y150_err),
                        front_compression_2p0_err_um=float(front_compression_2p0_err),
                        front_plateau_drift_2p0_10p0_um=float(front_plateau_drift_2p0_10p0),
                        porosity_drop_2p0_err_pp=float(porosity_drop_2p0_err),
                    )
                    _validate_calibration_metrics(
                        case_dir=case_dir,
                        obs_scenarios=obs_scenarios,
                        steady_profile_rmse_um=float(steady_profile_rmse),
                        steady_front_y150_err_um=float(steady_front_y150_err),
                        front_compression_2p0_err_um=float(front_compression_2p0_err),
                        front_plateau_drift_2p0_10p0_um=float(front_plateau_drift_2p0_10p0),
                        porosity_drop_2p0_err_pp=float(porosity_drop_2p0_err),
                    )
                    calibration_rows.append(
                        CalibrationRow(
                            E=float(E),
                            solid_visco_eta=float(eta_s),
                            diffuse_shear_scale=float(shear_scale),
                            nx=int(nx_cal),
                            ny=int(ny_cal),
                            steady_profile_rmse_um=float(steady_profile_rmse),
                            steady_front_y150_err_um=float(steady_front_y150_err),
                            front_compression_2p0_err_um=float(front_compression_2p0_err),
                            front_plateau_drift_2p0_10p0_um=float(front_plateau_drift_2p0_10p0),
                            porosity_drop_2p0_err_pp=float(porosity_drop_2p0_err),
                            score=float(score),
                            case_dir=str(case_dir),
                        )
                    )
                except Exception as exc:
                    failed_calibration_cases.append(
                        FailedCalibrationCase(
                            E=float(E),
                            solid_visco_eta=float(eta_s),
                            diffuse_shear_scale=float(shear_scale),
                            nx=int(nx_cal),
                            ny=int(ny_cal),
                            case_dir=str(case_dir),
                            stage="calibration",
                            error=str(exc),
                        )
                    )
                    print(
                        "[warn] calibration candidate failed: "
                        f"E={float(E):g}, eta_s={float(eta_s):g}, zeta={float(shear_scale):g}, nx={int(nx_cal)}"
                    )
                    if not bool(args.continue_on_candidate_failure):
                        raise

    failed_json = outdir / "benchmark6_blauert_channel_failed_cases.json"
    failed_json.write_text(
        json.dumps([row.__dict__ for row in failed_calibration_cases], indent=2) + "\n",
        encoding="utf-8",
    )
    if not calibration_rows:
        raise RuntimeError(
            "All Benchmark 6 calibration candidates failed. "
            f"See {failed_json} and the per-case run.log files under {calibration_root}."
        )
    calibration_rows_sorted = sorted(
        calibration_rows,
        key=lambda row: (row.score, row.steady_profile_rmse_um, row.front_compression_2p0_err_um),
    )
    best = calibration_rows_sorted[0]

    cal_csv = outdir / "benchmark6_blauert_channel_calibration.csv"
    _write_csv(
        cal_csv,
        [row.__dict__ for row in calibration_rows_sorted],
        fieldnames=[
            "E",
            "solid_visco_eta",
            "diffuse_shear_scale",
            "nx",
            "ny",
            "steady_profile_rmse_um",
            "steady_front_y150_err_um",
            "front_compression_2p0_err_um",
            "front_plateau_drift_2p0_10p0_um",
            "porosity_drop_2p0_err_pp",
            "score",
            "case_dir",
        ],
    )

    calibration_summary = {
        "profile": str(args.profile),
        "u_avg": float(args.u_avg),
        "rho_f": float(args.rho_f),
        "phi_b": float(args.phi_b),
        "alpha_ch_eps": float(args.alpha_ch_eps),
        "scale_alpha_ch_eps_with_zeta": bool(args.scale_alpha_ch_eps_with_zeta),
        "diffuse_shear_scale_ref": float(args.diffuse_shear_scale_ref),
        "refine_biofilm": bool(args.refine_biofilm),
        "refine_band": float(args.refine_band),
        "refine_expand_layers": int(args.refine_expand_layers),
        "gamma_div": float(args.gamma_div),
        "adaptive_gamma_div": bool(args.adaptive_gamma_div),
        "gamma_div_max": float(args.gamma_div_max),
        "vtk_every_override": int(vtk_every_override),
        "dt": float(args.dt),
        "t_final": float(args.t_final),
        "steady_time": float(steady_time),
        "observation_scenarios": obs_scenarios,
        "diffuse_shear_model": str(args.diffuse_shear_model),
        "diffuse_shear_time_scheme": str(args.diffuse_shear_time_scheme),
        "diffuse_shear_ramp_time": float(diffuse_shear_ramp_time),
        "best_calibration": best.__dict__,
        "calibration_rows": [row.__dict__ for row in calibration_rows_sorted],
        "failed_calibration_cases": [row.__dict__ for row in failed_calibration_cases],
        "artifacts": {
            "calibration_csv": str(cal_csv),
            "failed_cases_json": str(failed_json),
        },
    }
    (outdir / "benchmark6_blauert_channel_calibration_summary.json").write_text(
        json.dumps(calibration_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    if bool(args.calibration_only):
        print(json.dumps(calibration_summary, indent=2))
        return

    production_rows: list[dict[str, object]] = []
    production_root = outdir / "production"
    finest_nx = max(nx_list)
    history_case_dir: Path | None = None
    steady_snapshot_path: Path | None = None
    for nx in nx_list:
        ny = _ny_from_nx(int(nx))
        case_tag = f"nx{int(nx):03d}_E{int(round(best.E))}_eta{int(round(best.solid_visco_eta))}"
        reuse_calibration_case = int(nx) == int(best.nx)
        if reuse_calibration_case:
            case_dir = Path(str(best.case_dir))
            print(f"[info] reusing calibration case for production mesh nx={int(nx)}: {case_dir}")
        else:
            case_dir = production_root / case_tag
            cmd = _simulation_command(
                case_dir=case_dir,
                nx=int(nx),
                ny=int(ny),
                E=float(best.E),
                solid_visco_eta=float(best.solid_visco_eta),
                diffuse_shear_scale=float(best.diffuse_shear_scale),
                diffuse_shear_model=str(args.diffuse_shear_model),
                diffuse_shear_time_scheme=str(args.diffuse_shear_time_scheme),
                diffuse_shear_ramp_time=float(diffuse_shear_ramp_time),
                nonlinear_solver=str(args.nonlinear_solver),
                ls_mode=str(args.ls_mode),
                vi_c=float(args.vi_c),
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
                backend=str(args.backend),
                u_avg=float(args.u_avg),
                kappa_inv=float(args.kappa_inv),
                phi_b=float(args.phi_b),
                rho_f=float(args.rho_f),
                q=int(args.q),
                gamma_u=float(args.gamma_u),
                u_extension=str(args.u_extension),
                gamma_u_pin=float(args.gamma_u_pin),
                kinematics_scale=float(args.kinematics_scale),
                v_supg=float(args.v_supg),
                v_supg_mode=str(args.v_supg_mode),
                v_supg_c_nu=float(args.v_supg_c_nu),
                u_supg=float(args.u_supg),
                u_cip=float(args.u_cip),
                u_cip_weight=str(args.u_cip_weight),
                v_cip=float(args.v_cip),
                vS_cip=float(args.vS_cip),
                alpha_ch_eps=float(args.alpha_ch_eps),
                scale_alpha_ch_eps_with_zeta=bool(args.scale_alpha_ch_eps_with_zeta),
                diffuse_shear_scale_ref=float(args.diffuse_shear_scale_ref),
                refine_biofilm=bool(args.refine_biofilm),
                refine_band=float(args.refine_band),
                refine_expand_layers=int(args.refine_expand_layers),
                gamma_div=float(args.gamma_div),
                adaptive_gamma_div=bool(args.adaptive_gamma_div),
                gamma_div_max=float(args.gamma_div_max),
                newton_tol=float(args.newton_tol),
                max_it=int(args.max_it),
                dt_min=float(args.dt_min),
                accept_nonconverged_atol_factor=float(args.accept_nonconverged_atol_factor),
                restart_from=str(args.restart_from),
                restart_write_every=int(args.restart_write_every),
                restart_dt=float(args.restart_dt),
                dt=float(args.dt),
                t_final=float(args.t_final),
                t_ramp=float(args.t_ramp),
                snapshot_times=snapshot_tag,
                vtk_every=(
                    int(vtk_every_override)
                    if vtk_every_override >= 0
                    else (10 if int(nx) == int(finest_nx) else 0)
                ),
                global_front_quantile=float(args.global_front_quantile),
                dx_quantile=float(args.dx_quantile),
            )
            _run(
                cmd,
                cwd=_REPO_ROOT,
                log_path=case_dir / "run.log",
                skip_existing=bool(args.skip_existing)
                and _simulation_artifacts_ready(
                    case_dir,
                    required_snapshot_times=snapshot_times_sorted,
                ),
                sentinel=case_dir / "timeseries.csv",
                stream=bool(args.stream_subprocess),
            )
        obs_payloads = {
            scenario: _compare_observation_scenario(
                case_dir,
                scenario=scenario,
                steady_time=float(steady_time),
                skip_existing=bool(args.skip_existing),
                stream_subprocess=bool(args.stream_subprocess),
            )
            for scenario in obs_scenarios
        }
        steady_payload = obs_payloads.get("steady_dian", {})
        dyn08_payload = obs_payloads.get("dynamic_08pa", {})
        dyn164_payload = obs_payloads.get("dynamic_164pa", {})
        row: dict[str, object] = {
            "nx": int(nx),
            "ny": int(ny),
            "E": float(best.E),
            "solid_visco_eta": float(best.solid_visco_eta),
            "diffuse_shear_scale": float(best.diffuse_shear_scale),
            "diffuse_shear_model": str(args.diffuse_shear_model),
            "diffuse_shear_time_scheme": str(args.diffuse_shear_time_scheme),
            "diffuse_shear_ramp_time": float(diffuse_shear_ramp_time),
            "u_avg": float(args.u_avg),
            "rho_f": float(args.rho_f),
            "phi_b": float(args.phi_b),
            "dt": float(args.dt),
            "t_final": float(args.t_final),
            "steady_time": float(steady_time),
            "steady_profile_rmse_um": _obs_value(steady_payload, "measured", "steady_profile_rmse_um"),
            "steady_profile_mae_um": _obs_value(steady_payload, "measured", "steady_profile_mae_um"),
            "steady_profile_max_um": _obs_value(steady_payload, "measured", "steady_profile_max_um"),
            "steady_front_y150_err_um": _obs_value(steady_payload, "measured", "steady_front_y150_err_um"),
            "front_compression_2p0_um": _obs_value(dyn08_payload, "measured", "front_compression_2p0_um"),
            "front_compression_2p0_err_um": _obs_value(dyn08_payload, "abs_error", "front_compression_2p0_um"),
            "front_plateau_drift_2p0_10p0_um": _obs_value(dyn08_payload, "measured", "front_plateau_drift_2p0_10p0_um"),
            "porosity_drop_2p0_pp": _obs_value(dyn08_payload, "measured", "porosity_drop_2p0_pp"),
            "porosity_drop_2p0_err_pp": _obs_value(dyn08_payload, "abs_error", "porosity_drop_2p0_pp"),
            "thickness_drop_0p4_err_um": _obs_value(dyn164_payload, "abs_error", "thickness_drop_0p4_um"),
            "thickness_drop_2p1_err_um": _obs_value(dyn164_payload, "abs_error", "thickness_drop_2p1_um"),
            "tip_elongation_2p1_err_um": _obs_value(dyn164_payload, "abs_error", "tip_elongation_2p1_um"),
            "deformation_angle_2p1_err_deg": _obs_value(dyn164_payload, "abs_error", "deformation_angle_2p1_deg"),
            "case_dir": str(case_dir),
        }
        row["score"] = _calibration_score(
            steady_profile_rmse_um=float(row["steady_profile_rmse_um"]),
            steady_front_y150_err_um=float(row["steady_front_y150_err_um"]),
            front_compression_2p0_err_um=float(row["front_compression_2p0_err_um"]),
            front_plateau_drift_2p0_10p0_um=float(row["front_plateau_drift_2p0_10p0_um"]),
            porosity_drop_2p0_err_pp=float(row["porosity_drop_2p0_err_pp"]),
        )
        production_rows.append(row)
        if int(nx) == int(finest_nx):
            history_case_dir = case_dir
            steady_snapshot_path = _snapshot_path_for_time(case_dir, float(steady_time))

    summary_csv = outdir / "benchmark6_blauert_channel_summary.csv"
    fieldnames = sorted({key for row in production_rows for key in row})
    _write_csv(summary_csv, production_rows, fieldnames=fieldnames)

    if history_case_dir is not None:
        _plot_history(
            sim_csv=history_case_dir / "timeseries.csv",
            out_path=outdir / "benchmark6_blauert_channel_history.png",
            include_dynamic_targets="dynamic_08pa" in obs_scenarios,
        )
        if steady_snapshot_path is not None:
            _plot_contours(
                sim_snapshot_path=steady_snapshot_path,
                out_path=outdir / "benchmark6_blauert_channel_contours.png",
                steady_time=float(steady_time),
            )
    _plot_mesh_sensitivity(production_rows, out_path=outdir / "benchmark6_blauert_channel_mesh_sensitivity.png")

    summary = {
        "profile": str(args.profile),
        "u_avg": float(args.u_avg),
        "rho_f": float(args.rho_f),
        "phi_b": float(args.phi_b),
        "alpha_ch_eps": float(args.alpha_ch_eps),
        "scale_alpha_ch_eps_with_zeta": bool(args.scale_alpha_ch_eps_with_zeta),
        "diffuse_shear_scale_ref": float(args.diffuse_shear_scale_ref),
        "refine_biofilm": bool(args.refine_biofilm),
        "refine_band": float(args.refine_band),
        "refine_expand_layers": int(args.refine_expand_layers),
        "gamma_div": float(args.gamma_div),
        "adaptive_gamma_div": bool(args.adaptive_gamma_div),
        "gamma_div_max": float(args.gamma_div_max),
        "vtk_every_override": int(vtk_every_override),
        "dt": float(args.dt),
        "t_final": float(args.t_final),
        "steady_time": float(steady_time),
        "observation_scenarios": obs_scenarios,
        "diffuse_shear_model": str(args.diffuse_shear_model),
        "diffuse_shear_time_scheme": str(args.diffuse_shear_time_scheme),
        "diffuse_shear_ramp_time": float(diffuse_shear_ramp_time),
        "best_calibration": best.__dict__,
        "production_rows": production_rows,
        "artifacts": {
            "calibration_csv": str(cal_csv),
            "summary_csv": str(summary_csv),
            "history_png": str(outdir / "benchmark6_blauert_channel_history.png"),
            "contours_png": str(outdir / "benchmark6_blauert_channel_contours.png"),
            "mesh_png": str(outdir / "benchmark6_blauert_channel_mesh_sensitivity.png"),
        },
    }
    (outdir / "benchmark6_blauert_channel_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
