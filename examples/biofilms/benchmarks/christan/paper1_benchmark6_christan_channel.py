#!/usr/bin/env python3
"""Paper 1 Benchmark 6: Christan Biofilm I channel deformation benchmark.

This benchmark replaces the older Blauert/Dian observable workflow with the
more explicit Biofilm I geometry pairing documented by Picioreanu et al.

Benchmark inputs:
  - unloaded and loaded Biofilm I contours traced from the Christan paper
  - Christan computational box: 3 mm x 1 mm
  - flow-cell Reynolds number defined with the hydraulic diameter of the
    2 mm x 1 mm rectangular channel

Benchmark outputs:
  - calibration CSV over Young's modulus on the coarsest mesh
  - production summary CSV on a mesh ladder
  - contour overlay, front-displacement profile, and mesh-sensitivity plots
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
except Exception:  # pragma: no cover - plotting is optional
    matplotlib = None
    plt = None


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.biofilms.benchmarks.christan.compare_sim_vs_christan import (  # noqa: E402
    _read_plain_contour_mm,
    _read_snapshot_contours_mm,
    compare_case,
    find_snapshot,
)
from examples.biofilms.benchmarks.christan.prepare_biofilm_I_geometry import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_GEOMETRY_DIR,
    ensure_geometry_artifacts,
    front_x_mm,
)


DRIVER = REPO_ROOT / "examples/biofilms/benchmarks/blauert/blauert_biofilm_deformation_one_domain.py"


@dataclass(frozen=True)
class CalibrationRow:
    E: float
    eps: float
    alpha_ch_eps: float
    nx: int
    ny: int
    primary_profile_rmse_um: float
    alt_profile_rmse_um: float
    combined_profile_rmse_um: float
    combined_mean_dx_abs_error_um: float
    combined_mean_front_abs_error_um: float
    combined_nearest_mean_um: float
    combined_nearest_max_um: float
    reached_time_s: float
    final_alpha_area_rel_drift: float
    max_abs_alpha_area_rel_drift: float
    score: float
    case_dir: str


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for item in str(raw).split(","):
        text = item.strip()
        if text:
            out.append(float(text))
    if not out:
        raise ValueError("Expected at least one float.")
    return out


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for item in str(raw).split(","):
        text = item.strip()
        if text:
            out.append(int(float(text)))
    if not out:
        raise ValueError("Expected at least one integer.")
    return out


def _parse_y_levels(raw: str) -> list[int]:
    return _parse_int_list(raw)


def _float_tag(value: float) -> str:
    return f"{float(value):.2e}".replace("+", "").replace(".", "p")


def _ny_from_nx(nx: int) -> int:
    return max(6, int(round(float(nx) / 3.0)))


def _read_timeseries_metrics(case_dir: Path) -> dict[str, float]:
    ts_path = Path(case_dir) / "timeseries.csv"
    if not ts_path.exists():
        return {
            "reached_time_s": float("nan"),
            "final_alpha_area_rel_drift": float("nan"),
            "max_abs_alpha_area_rel_drift": float("nan"),
        }
    with ts_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {
            "reached_time_s": float("nan"),
            "final_alpha_area_rel_drift": float("nan"),
            "max_abs_alpha_area_rel_drift": float("nan"),
        }
    t_vals = np.asarray([float(row.get("t_s", float("nan"))) for row in rows], dtype=float)
    drift_vals = np.asarray([float(row.get("alpha_area_rel_drift", float("nan"))) for row in rows], dtype=float)
    finite_t = t_vals[np.isfinite(t_vals)]
    finite_drift = drift_vals[np.isfinite(drift_vals)]
    return {
        "reached_time_s": float(finite_t[-1]) if finite_t.size else float("nan"),
        "final_alpha_area_rel_drift": float(finite_drift[-1]) if finite_drift.size else float("nan"),
        "max_abs_alpha_area_rel_drift": float(np.max(np.abs(finite_drift))) if finite_drift.size else float("nan"),
    }


def _hydraulic_diameter_m(*, width_mm: float, height_mm: float) -> float:
    w = 1.0e-3 * float(width_mm)
    h = 1.0e-3 * float(height_mm)
    return float(2.0 * w * h / (w + h))


def _u_avg_from_re(*, Re: float, rho: float, mu: float, width_mm: float, height_mm: float) -> float:
    dh = _hydraulic_diameter_m(width_mm=float(width_mm), height_mm=float(height_mm))
    return float(float(Re) * float(mu) / (float(rho) * dh))


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
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n" + "\n".join(tail))


def _write_csv(path: Path, rows: list[dict[str, object]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _snapshot_ready(case_dir: Path, *, times_s: list[float]) -> bool:
    if not (case_dir / "timeseries.csv").exists():
        return False
    snap_dir = case_dir / "snapshots"
    for t_s in times_s:
        if not any(snap_dir.glob(f"*t{float(t_s):06.3f}_alpha05.csv")):
            return False
    return True


def _simulation_command(
    *,
    case_dir: Path,
    initial_csv: Path,
    nx: int,
    ny: int,
    E: float,
    nu: float,
    dt: float,
    t_final: float,
    steady_time: float,
    t_ramp: float,
    q: int,
    backend: str,
    paper1_reduced: bool,
    linear_backend: str,
    linear_ksp_type: str,
    linear_pc_type: str,
    linear_pc_factor_solver_type: str,
    linear_ksp_rtol: float,
    linear_ksp_max_it: int,
    rho_f: float,
    u_avg: float,
    kappa_inv: float,
    phi_b: float,
    phi_init_mode: str,
    fluid_convection: str,
    include_skeleton_acceleration: bool,
    rho_s0_tilde: float,
    alpha_biot: float,
    skeleton_inertia_convection: str,
    gamma_u: float,
    gamma_u_pin: float,
    kinematics_scale: float,
    eps: float,
    alpha_ch_eps: float,
    diffuse_shear_traction: bool,
    diffuse_shear_model: str,
    diffuse_shear_scale: float,
    diffuse_shear_eta: float,
    diffuse_shear_topweight: bool,
    re_char_length_m: float,
    track_y_um_csv: str,
    v_supg: float,
    v_supg_mode: str,
    v_supg_c_nu: float,
    u_supg: float,
    u_cip: float,
    v_cip: float,
    vS_cip: float,
    gamma_phi: float,
    phi_supg: float,
    phi_cip: float,
    alpha_supg: float,
    alpha_cip: float,
    gamma_div: float,
    gamma_div_max: float,
    startup_staggered_predictor: bool,
    startup_staggered_max_time: float,
    startup_fluid_newton_tol: float,
    startup_solid_newton_tol: float,
    startup_fluid_max_it: int,
    startup_solid_max_it: int,
    startup_staggered_sweeps: int,
    startup_staggered_slip_tol: float,
    newton_tol: float,
    max_it: int,
    dt_min: float,
    accept_nonconverged_atol_factor: float,
    refine_band: float,
    refine_expand_layers: int,
    vtk_every: int,
) -> list[str]:
    snapshot_times = f"0,{float(steady_time):g}"
    cmd = [
        sys.executable,
        "-u",
        str(DRIVER),
        "--backend",
        str(backend),
        "--paper1-reduced" if bool(paper1_reduced) else "--no-paper1-reduced",
        "--transport-mode",
        "pde",
        "--L",
        "3.0e-3",
        "--H",
        "1.0e-3",
        "--re-char-length",
        str(float(re_char_length_m)),
        "--alpha0-file",
        str(initial_csv),
        "--alpha0-scale",
        "1e-3",
        "--alpha0-tx",
        "0.0",
        "--alpha0-ty",
        "0.0",
        "--nonlinear-solver",
        "pdas",
        "--no-predictor-clip-01",
        "--ls-mode",
        "dealii",
        "--vi-unconstrained-lm",
        "--vi-lm-lambda0",
        "1e-4",
        "--vi-lm-lambda-max",
        "1e6",
        "--vi-lm-growth",
        "5",
        "--vi-lm-decay",
        "0.5",
        "--vi-lm-accept-ratio",
        "1e-3",
        "--vi-lm-good-ratio",
        "0.05",
        "--vi-lm-max-tries",
        "6",
        "--linear-backend",
        str(linear_backend),
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
        "--no-linear-schur",
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
        str(float(nu)),
        "--solid-visco-eta",
        "0",
        "--u-avg",
        str(float(u_avg)),
        "--rho-f",
        str(float(rho_f)),
        "--kappa-inv",
        str(float(kappa_inv)),
        "--fluid-convection",
        str(fluid_convection),
        "--phi-b",
        str(float(phi_b)),
        "--phi-init-mode",
        str(phi_init_mode),
        "--include-skeleton-acceleration"
        if bool(include_skeleton_acceleration)
        else "--no-include-skeleton-acceleration",
        "--rho-s0-tilde",
        str(float(rho_s0_tilde)),
        "--alpha-biot",
        str(float(alpha_biot)),
        "--skeleton-inertia-convection",
        str(skeleton_inertia_convection),
        "--gamma-u",
        str(float(gamma_u)),
        "--u-extension",
        "l2",
        "--gamma-u-pin",
        str(float(gamma_u_pin)),
        "--kinematics-scale",
        str(float(kinematics_scale)),
        "--eps",
        str(float(eps)),
        "--alpha-ch-M",
        "1e-12",
        "--alpha-ch-gamma",
        "2e-3",
        "--alpha-ch-eps",
        str(float(alpha_ch_eps)),
        "--diffuse-shear-model",
        str(diffuse_shear_model),
        "--diffuse-shear-scale",
        str(float(diffuse_shear_scale)),
        "--diffuse-shear-eta",
        str(float(diffuse_shear_eta)),
        "--alpha-advection-form",
        "conservative_weak",
        "--alpha-advect-with",
        "biofilm_volume",
        "--support-physics",
        "internal_conversion",
        "--gamma-phi",
        str(float(gamma_phi)),
        "--phi-supg",
        str(float(phi_supg)),
        "--phi-cip",
        str(float(phi_cip)),
        "--alpha-supg",
        str(float(alpha_supg)),
        "--alpha-cip",
        str(float(alpha_cip)),
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
        "biofilm",
        "--v-cip",
        str(float(v_cip)),
        "--vS-cip",
        str(float(vS_cip)),
        "--global-front-quantile",
        "0.05",
        "--dx-quantile",
        "0.05",
        "--track-y-um",
        str(track_y_um_csv),
        "--gamma-div",
        str(float(gamma_div)),
        "--adaptive-gamma-div",
        "--gamma-div-max",
        str(float(gamma_div_max)),
        "--startup-staggered-predictor"
        if bool(startup_staggered_predictor)
        else "--no-startup-staggered-predictor",
        "--startup-staggered-max-time",
        str(float(startup_staggered_max_time)),
        "--startup-fluid-newton-tol",
        str(float(startup_fluid_newton_tol)),
        "--startup-solid-newton-tol",
        str(float(startup_solid_newton_tol)),
        "--startup-fluid-max-it",
        str(int(startup_fluid_max_it)),
        "--startup-solid-max-it",
        str(int(startup_solid_max_it)),
        "--startup-staggered-sweeps",
        str(int(startup_staggered_sweeps)),
        "--startup-staggered-slip-tol",
        str(float(startup_staggered_slip_tol)),
        "--newton-tol",
        str(float(newton_tol)),
        "--max-it",
        str(int(max_it)),
        "--allow-dt-reduction",
        "--dt-min",
        str(float(dt_min)),
        "--accept-nonconverged-atol-factor",
        str(float(accept_nonconverged_atol_factor)),
        "--refine-biofilm",
        "--refine-band",
        str(float(refine_band)),
        "--refine-expand-layers",
        str(int(refine_expand_layers)),
        "--restart-write-every",
        "1",
        "--vtk-every",
        str(int(vtk_every)),
        "--snapshot-times",
        str(snapshot_times),
        "--out-dir",
        str(case_dir),
    ]
    if bool(diffuse_shear_traction):
        cmd.append("--diffuse-shear-traction")
    else:
        cmd.append("--no-diffuse-shear-traction")
    if bool(diffuse_shear_topweight):
        cmd.append("--diffuse-shear-topweight")
    return cmd


def _front_profile_from_list(points_list: list[np.ndarray], y_samples_mm: np.ndarray) -> np.ndarray:
    ys = np.asarray(y_samples_mm, dtype=float).ravel()
    out = np.full_like(ys, float("nan"), dtype=float)
    for i, y_mm in enumerate(ys):
        xs: list[float] = []
        for points in points_list:
            val = front_x_mm(np.asarray(points, dtype=float), float(y_mm))
            if np.isfinite(val):
                xs.append(float(val))
        if xs:
            out[i] = float(min(xs))
    return out


def _contour_y_overlap(a: list[np.ndarray], b: list[np.ndarray]) -> tuple[float, float]:
    a_pts = np.vstack([np.asarray(points, dtype=float) for points in a])
    b_pts = np.vstack([np.asarray(points, dtype=float) for points in b])
    return max(float(np.min(a_pts[:, 1])), float(np.min(b_pts[:, 1]))), min(
        float(np.max(a_pts[:, 1])), float(np.max(b_pts[:, 1]))
    )


def _plot_contours(
    *,
    geometry_dir: Path,
    case_dir: Path,
    steady_time: float,
    out_path: Path,
) -> None:
    if plt is None:
        return
    geom = ensure_geometry_artifacts(force=False, out_dir=Path(geometry_dir))
    initial = _read_plain_contour_mm(Path(str(geom["contour_files"]["initial"])))
    final_primary = _read_plain_contour_mm(Path(str(geom["contour_files"]["final_primary"])))
    final_alt = _read_plain_contour_mm(Path(str(geom["contour_files"]["final_alternative"])))
    sim_initial = _read_snapshot_contours_mm(find_snapshot(case_dir, 0.0))
    sim_final = _read_snapshot_contours_mm(find_snapshot(case_dir, float(steady_time)))

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(1.0e3 * initial[:, 0], 1.0e3 * initial[:, 1], color="#666666", lw=1.4, label="Initial trace")
    ax.plot(1.0e3 * final_primary[:, 0], 1.0e3 * final_primary[:, 1], color="#c0392b", lw=1.8, label="Final trace")
    ax.plot(1.0e3 * final_alt[:, 0], 1.0e3 * final_alt[:, 1], color="#f39c12", lw=1.2, ls="--", label="Final trace alt")
    for idx, contour in enumerate(sim_initial):
        ax.plot(
            1.0e3 * contour[:, 0],
            1.0e3 * contour[:, 1],
            color="#2980b9",
            lw=1.0,
            alpha=0.7,
            label="Simulated initial" if idx == 0 else None,
        )
    for idx, contour in enumerate(sim_final):
        ax.plot(
            1.0e3 * contour[:, 0],
            1.0e3 * contour[:, 1],
            color="#1f7a1f",
            lw=1.6,
            label=f"Simulated t={float(steady_time):g} s" if idx == 0 else None,
        )
    ax.set_xlabel(r"$x\;[\mu\mathrm{m}]$")
    ax.set_ylabel(r"$z\;[\mu\mathrm{m}]$")
    ax.set_xlim(250.0, 2050.0)
    ax.set_ylim(-10.0, 500.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_front_profile(
    *,
    geometry_dir: Path,
    case_dir: Path,
    steady_time: float,
    out_path: Path,
) -> None:
    if plt is None:
        return
    geom = ensure_geometry_artifacts(force=False, out_dir=Path(geometry_dir))
    initial = _read_plain_contour_mm(Path(str(geom["contour_files"]["initial"])))
    final_primary = _read_plain_contour_mm(Path(str(geom["contour_files"]["final_primary"])))
    final_alt = _read_plain_contour_mm(Path(str(geom["contour_files"]["final_alternative"])))
    sim_initial = _read_snapshot_contours_mm(find_snapshot(case_dir, 0.0))
    sim_final = _read_snapshot_contours_mm(find_snapshot(case_dir, float(steady_time)))

    y_min, y_max = _contour_y_overlap([initial], [final_primary])
    y_min = max(float(y_min), 0.08)
    y_max = min(float(y_max), 0.42)
    ys = np.linspace(y_min, y_max, 180, dtype=float)
    dx_primary = 1.0e3 * (_front_profile_from_list([final_primary], ys) - _front_profile_from_list([initial], ys))
    dx_alt = 1.0e3 * (_front_profile_from_list([final_alt], ys) - _front_profile_from_list([initial], ys))
    dx_sim = 1.0e3 * (_front_profile_from_list(sim_final, ys) - _front_profile_from_list(sim_initial, ys))

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.plot(dx_primary, 1.0e3 * ys, color="#c0392b", lw=1.8, label="Final trace")
    ax.plot(dx_alt, 1.0e3 * ys, color="#f39c12", lw=1.2, ls="--", label="Final trace alt")
    ax.plot(dx_sim, 1.0e3 * ys, color="#1f7a1f", lw=1.8, label=f"Simulation t={float(steady_time):g} s")
    ax.set_xlabel(r"$\Delta x_{\mathrm{front}}\;[\mu\mathrm{m}]$")
    ax.set_ylabel(r"$z\;[\mu\mathrm{m}]$")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_mesh_sensitivity(rows: list[dict[str, object]], *, out_path: Path) -> None:
    if plt is None or not rows:
        return
    rows_sorted = sorted(rows, key=lambda row: int(row["nx"]))
    nx = np.asarray([int(row["nx"]) for row in rows_sorted], dtype=float)
    profile = np.asarray([float(row["combined_profile_rmse_um"]) for row in rows_sorted], dtype=float)
    dx = np.asarray([float(row["combined_mean_dx_abs_error_um"]) for row in rows_sorted], dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(nx, profile, marker="o", color="#1f7a1f", lw=1.8, label="Contour RMSE")
    ax.plot(nx, dx, marker="s", color="#c0392b", lw=1.6, label="Mean |dx error|")
    ax.set_xlabel(r"$n_x$")
    ax.set_ylabel(r"Error [$\mu\mathrm{m}$]")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _calibration_row_from_payload(
    *,
    E: float,
    eps: float,
    alpha_ch_eps: float,
    nx: int,
    ny: int,
    payload: dict[str, object],
    case_dir: Path,
) -> CalibrationRow:
    primary = payload["primary"]
    alternate = payload["alternate"]
    combined = payload["combined"]
    ts_metrics = _read_timeseries_metrics(case_dir)
    return CalibrationRow(
        E=float(E),
        eps=float(eps),
        alpha_ch_eps=float(alpha_ch_eps),
        nx=int(nx),
        ny=int(ny),
        primary_profile_rmse_um=float(primary["profile"]["rmse_um"]),
        alt_profile_rmse_um=float(alternate["profile"]["rmse_um"]),
        combined_profile_rmse_um=float(combined["profile_rmse_um"]),
        combined_mean_dx_abs_error_um=float(combined["mean_dx_abs_error_um"]),
        combined_mean_front_abs_error_um=float(combined["mean_front_abs_error_um"]),
        combined_nearest_mean_um=float(combined["nearest_mean_um"]),
        combined_nearest_max_um=float(combined["nearest_max_um"]),
        reached_time_s=float(ts_metrics["reached_time_s"]),
        final_alpha_area_rel_drift=float(ts_metrics["final_alpha_area_rel_drift"]),
        max_abs_alpha_area_rel_drift=float(ts_metrics["max_abs_alpha_area_rel_drift"]),
        score=float(combined["score"]),
        case_dir=str(case_dir),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=str, default="examples/biofilms/results/benchmark6_christan_channel")
    ap.add_argument("--profile", type=str, default="baseline", choices=("smoke", "baseline", "production"))
    ap.add_argument("--geometry-dir", type=str, default=str(DEFAULT_GEOMETRY_DIR))
    ap.add_argument("--E-list", type=str, default="")
    ap.add_argument("--nx-list", type=str, default="")
    ap.add_argument("--steady-time", type=float, default=1.5)
    ap.add_argument("--t-final", type=float, default=float("nan"))
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--t-ramp", type=float, default=1.0)
    ap.add_argument("--backend", type=str, default="cpp", choices=("cpp", "jit", "python"))
    ap.add_argument("--paper1-reduced", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--linear-backend", type=str, default="scipy")
    ap.add_argument("--linear-ksp-type", type=str, default="preonly")
    ap.add_argument("--linear-pc-type", type=str, default="lu")
    ap.add_argument("--linear-pc-factor-solver-type", type=str, default="mumps")
    ap.add_argument("--linear-ksp-rtol", type=float, default=1.0e-8)
    ap.add_argument("--linear-ksp-max-it", type=int, default=200)
    ap.add_argument("--Re", type=float, default=91.0)
    ap.add_argument("--rho-f", type=float, default=1000.0)
    ap.add_argument("--mu-f", type=float, default=1.0e-3)
    ap.add_argument("--flow-cell-width-mm", type=float, default=2.0)
    ap.add_argument("--flow-cell-height-mm", type=float, default=1.0)
    ap.add_argument("--phi-b", type=float, default=0.6)
    ap.add_argument("--phi-init-mode", type=str, default="linear_alpha", choices=("linear_alpha", "constant_phi_b"))
    ap.add_argument("--kappa-inv", type=float, default=1.0e15)
    ap.add_argument("--nu", type=float, default=0.4)
    ap.add_argument("--q", type=int, default=4)
    ap.add_argument(
        "--include-skeleton-acceleration",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    ap.add_argument(
        "--rho-s0-tilde",
        type=float,
        default=float("nan"),
        help="Reference solid density coefficient. Default: rho_f / (1-phi_b).",
    )
    ap.add_argument("--alpha-biot", type=float, default=1.0)
    ap.add_argument(
        "--skeleton-inertia-convection",
        type=str,
        default="full",
        choices=("lagged", "full"),
    )
    ap.add_argument(
        "--fluid-convection",
        type=str,
        default="full",
        choices=("full", "lagged", "imex", "off"),
    )
    ap.add_argument("--gamma-u", type=float, default=20.0)
    ap.add_argument("--gamma-u-pin", type=float, default=1.0e-4)
    ap.add_argument("--kinematics-scale", type=float, default=1000.0)
    ap.add_argument("--eps", type=float, default=float("nan"))
    ap.add_argument("--eps-list", type=str, default="")
    ap.add_argument("--alpha-ch-eps", type=float, default=2.0e-5)
    ap.add_argument("--alpha-ch-eps-list", type=str, default="")
    ap.add_argument("--diffuse-shear-traction", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--diffuse-shear-model",
        type=str,
        default="lagged_stress",
        choices=("lagged_velocity", "lagged_stress", "poiseuille"),
    )
    ap.add_argument("--diffuse-shear-scale", type=float, default=1.0)
    ap.add_argument("--diffuse-shear-eta", type=float, default=1.0e-12)
    ap.add_argument("--diffuse-shear-topweight", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--v-supg", type=float, default=1.0)
    ap.add_argument("--v-supg-mode", type=str, default="residual", choices=("streamline", "residual"))
    ap.add_argument("--v-supg-c-nu", type=float, default=4.0)
    ap.add_argument("--u-supg", type=float, default=1.0)
    ap.add_argument("--u-cip", type=float, default=1.0)
    ap.add_argument("--v-cip", type=float, default=1.0)
    ap.add_argument("--vS-cip", type=float, default=1.0)
    ap.add_argument("--gamma-phi", type=float, default=5.0)
    ap.add_argument("--phi-supg", type=float, default=0.0)
    ap.add_argument("--phi-cip", type=float, default=0.0)
    ap.add_argument("--alpha-supg", type=float, default=0.5)
    ap.add_argument("--alpha-cip", type=float, default=0.0)
    ap.add_argument("--gamma-div", type=float, default=0.2)
    ap.add_argument("--gamma-div-max", type=float, default=0.5)
    ap.add_argument("--startup-staggered-predictor", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--startup-staggered-max-time",
        type=float,
        default=float("nan"),
        help="Default: use --t-ramp when omitted so the staggered preload covers the inflow ramp.",
    )
    ap.add_argument("--startup-fluid-newton-tol", type=float, default=1.0e-10)
    ap.add_argument("--startup-solid-newton-tol", type=float, default=1.0e-10)
    ap.add_argument("--startup-fluid-max-it", type=int, default=12)
    ap.add_argument("--startup-solid-max-it", type=int, default=12)
    ap.add_argument("--startup-staggered-sweeps", type=int, default=2)
    ap.add_argument("--startup-staggered-slip-tol", type=float, default=0.0)
    ap.add_argument("--newton-tol", type=float, default=1.0e-6)
    ap.add_argument("--max-it", type=int, default=12)
    ap.add_argument("--dt-min", type=float, default=5.0e-3)
    ap.add_argument("--accept-nonconverged-atol-factor", type=float, default=4.0)
    ap.add_argument("--refine-band", type=float, default=2.5e-4)
    ap.add_argument("--refine-expand-layers", type=int, default=1)
    ap.add_argument("--vtk-every", type=int, default=0)
    ap.add_argument("--y-levels-um", type=str, default="150,250,350")
    ap.add_argument("--stream-subprocess", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--continue-on-candidate-failure", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--calibration-only", action="store_true")
    args = ap.parse_args()

    outdir = (REPO_ROOT / str(args.outdir)).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    geometry_dir = (REPO_ROOT / str(args.geometry_dir)).resolve()
    geometry_meta = ensure_geometry_artifacts(force=False, out_dir=geometry_dir)
    initial_csv = Path(str(geometry_meta["contour_files"]["initial"]))

    if str(args.profile) == "smoke":
        E_list = _parse_float_list(str(args.E_list) or "60,75,90")
        nx_list = _parse_int_list(str(args.nx_list) or "16")
    elif str(args.profile) == "production":
        E_list = _parse_float_list(str(args.E_list) or "60,70,75,80,90")
        nx_list = _parse_int_list(str(args.nx_list) or "24,32")
    else:
        E_list = _parse_float_list(str(args.E_list) or "60,70,75,80,90")
        nx_list = _parse_int_list(str(args.nx_list) or "16,24")
    alpha_ch_eps_list = (
        _parse_float_list(str(args.alpha_ch_eps_list))
        if str(args.alpha_ch_eps_list).strip()
        else [float(args.alpha_ch_eps)]
    )
    eps_list = _parse_float_list(str(args.eps_list)) if str(args.eps_list).strip() else []

    y_levels_um = _parse_y_levels(str(args.y_levels_um))
    track_y_um_csv = ",".join(str(int(v)) for v in y_levels_um)
    steady_time = float(args.steady_time)
    t_final = float(args.t_final) if np.isfinite(float(args.t_final)) else float(steady_time)
    if t_final + 1.0e-12 < steady_time:
        raise ValueError("--t-final must be >= --steady-time.")
    startup_staggered_max_time = (
        float(args.startup_staggered_max_time)
        if np.isfinite(float(args.startup_staggered_max_time))
        else float(args.t_ramp)
    )
    rho_s0_tilde = (
        float(args.rho_s0_tilde)
        if np.isfinite(float(args.rho_s0_tilde))
        else float(args.rho_f) / max(1.0e-12, 1.0 - float(args.phi_b))
    )

    u_avg = _u_avg_from_re(
        Re=float(args.Re),
        rho=float(args.rho_f),
        mu=float(args.mu_f),
        width_mm=float(args.flow_cell_width_mm),
        height_mm=float(args.flow_cell_height_mm),
    )
    u_max = 1.5 * float(u_avg)
    dh_m = _hydraulic_diameter_m(
        width_mm=float(args.flow_cell_width_mm),
        height_mm=float(args.flow_cell_height_mm),
    )
    re_char_length_m = float(dh_m)

    calibration_root = outdir / "calibration"
    nx_cal = int(min(nx_list))
    ny_cal = _ny_from_nx(nx_cal)
    calibration_rows: list[CalibrationRow] = []
    calibration_payloads: dict[str, dict[str, object]] = {}
    failed_calibration_cases: list[dict[str, object]] = []
    for E in E_list:
        for alpha_ch_eps in alpha_ch_eps_list:
            eps_candidates = [float(alpha_ch_eps)]
            if eps_list:
                eps_candidates = [float(v) for v in eps_list]
            elif np.isfinite(float(args.eps)):
                eps_candidates = [float(args.eps)]
            for eps in eps_candidates:
                if math.isclose(float(eps), float(alpha_ch_eps), rel_tol=0.0, abs_tol=1.0e-18):
                    tag = f"E{float(E):05.1f}_eps{_float_tag(float(eps))}_nx{int(nx_cal):03d}"
                else:
                    tag = (
                        f"E{float(E):05.1f}_eps{_float_tag(float(eps))}_"
                        f"cheps{_float_tag(float(alpha_ch_eps))}_nx{int(nx_cal):03d}"
                    )
                case_dir = calibration_root / tag
                try:
                    cmd = _simulation_command(
                        case_dir=case_dir,
                        initial_csv=initial_csv,
                        nx=int(nx_cal),
                        ny=int(ny_cal),
                        E=float(E),
                        nu=float(args.nu),
                        dt=float(args.dt),
                        t_final=float(t_final),
                        steady_time=float(steady_time),
                        t_ramp=float(args.t_ramp),
                        q=int(args.q),
                        backend=str(args.backend),
                        paper1_reduced=bool(args.paper1_reduced),
                        linear_backend=str(args.linear_backend),
                        linear_ksp_type=str(args.linear_ksp_type),
                        linear_pc_type=str(args.linear_pc_type),
                        linear_pc_factor_solver_type=str(args.linear_pc_factor_solver_type),
                        linear_ksp_rtol=float(args.linear_ksp_rtol),
                        linear_ksp_max_it=int(args.linear_ksp_max_it),
                        rho_f=float(args.rho_f),
                        u_avg=float(u_avg),
                        kappa_inv=float(args.kappa_inv),
                        phi_b=float(args.phi_b),
                        phi_init_mode=str(args.phi_init_mode),
                        fluid_convection=str(args.fluid_convection),
                        include_skeleton_acceleration=bool(args.include_skeleton_acceleration),
                        rho_s0_tilde=float(rho_s0_tilde),
                        alpha_biot=float(args.alpha_biot),
                        skeleton_inertia_convection=str(args.skeleton_inertia_convection),
                        gamma_u=float(args.gamma_u),
                        gamma_u_pin=float(args.gamma_u_pin),
                        kinematics_scale=float(args.kinematics_scale),
                        eps=float(eps),
                        alpha_ch_eps=float(alpha_ch_eps),
                        diffuse_shear_traction=bool(args.diffuse_shear_traction),
                        diffuse_shear_model=str(args.diffuse_shear_model),
                        diffuse_shear_scale=float(args.diffuse_shear_scale),
                        diffuse_shear_eta=float(args.diffuse_shear_eta),
                        diffuse_shear_topweight=bool(args.diffuse_shear_topweight),
                        re_char_length_m=float(re_char_length_m),
                        track_y_um_csv=str(track_y_um_csv),
                        v_supg=float(args.v_supg),
                        v_supg_mode=str(args.v_supg_mode),
                        v_supg_c_nu=float(args.v_supg_c_nu),
                        u_supg=float(args.u_supg),
                        u_cip=float(args.u_cip),
                        v_cip=float(args.v_cip),
                        vS_cip=float(args.vS_cip),
                        gamma_phi=float(args.gamma_phi),
                        phi_supg=float(args.phi_supg),
                        phi_cip=float(args.phi_cip),
                        alpha_supg=float(args.alpha_supg),
                        alpha_cip=float(args.alpha_cip),
                        gamma_div=float(args.gamma_div),
                        gamma_div_max=float(args.gamma_div_max),
                        startup_staggered_predictor=bool(args.startup_staggered_predictor),
                        startup_staggered_max_time=float(startup_staggered_max_time),
                        startup_fluid_newton_tol=float(args.startup_fluid_newton_tol),
                        startup_solid_newton_tol=float(args.startup_solid_newton_tol),
                        startup_fluid_max_it=int(args.startup_fluid_max_it),
                        startup_solid_max_it=int(args.startup_solid_max_it),
                        startup_staggered_sweeps=int(args.startup_staggered_sweeps),
                        startup_staggered_slip_tol=float(args.startup_staggered_slip_tol),
                        newton_tol=float(args.newton_tol),
                        max_it=int(args.max_it),
                        dt_min=float(args.dt_min),
                        accept_nonconverged_atol_factor=float(args.accept_nonconverged_atol_factor),
                        refine_band=float(args.refine_band),
                        refine_expand_layers=int(args.refine_expand_layers),
                        vtk_every=0,
                    )
                    _run(
                        cmd,
                        cwd=REPO_ROOT,
                        log_path=case_dir / "run.log",
                        skip_existing=bool(args.skip_existing) and _snapshot_ready(case_dir, times_s=[0.0, float(steady_time)]),
                        sentinel=case_dir / "timeseries.csv",
                        stream=bool(args.stream_subprocess),
                    )
                    payload = compare_case(
                        out_dir=case_dir,
                        target_time=float(steady_time),
                        initial_time=0.0,
                        y_levels_um=list(y_levels_um),
                        geometry_dir=geometry_dir,
                    )
                    compare_json = case_dir / "compare_christan.json"
                    compare_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                    row = _calibration_row_from_payload(
                        E=float(E),
                        eps=float(eps),
                        alpha_ch_eps=float(alpha_ch_eps),
                        nx=int(nx_cal),
                        ny=int(ny_cal),
                        payload=payload,
                        case_dir=case_dir,
                    )
                    calibration_rows.append(row)
                    calibration_payloads[str(case_dir)] = payload
                except Exception as exc:
                    failed_calibration_cases.append(
                        {
                            "E": float(E),
                            "eps": float(eps),
                            "alpha_ch_eps": float(alpha_ch_eps),
                            "nx": int(nx_cal),
                            "ny": int(ny_cal),
                            "case_dir": str(case_dir),
                            "error": str(exc),
                        }
                    )
                    if not bool(args.continue_on_candidate_failure):
                        raise

    if not calibration_rows:
        failed_json = outdir / "benchmark6_christan_channel_failed_cases.json"
        failed_json.write_text(json.dumps(failed_calibration_cases, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(
            "All Christan calibration candidates failed. "
            f"See {failed_json} and the per-case run.log files under {calibration_root}."
        )

    calibration_rows_sorted = sorted(
        calibration_rows,
        key=lambda row: (row.score, row.combined_profile_rmse_um, row.combined_mean_dx_abs_error_um),
    )
    best = calibration_rows_sorted[0]

    calibration_csv = outdir / "benchmark6_christan_channel_calibration.csv"
    _write_csv(
        calibration_csv,
        [row.__dict__ for row in calibration_rows_sorted],
        fieldnames=[
            "E",
            "eps",
            "alpha_ch_eps",
            "nx",
            "ny",
            "primary_profile_rmse_um",
            "alt_profile_rmse_um",
            "combined_profile_rmse_um",
            "combined_mean_dx_abs_error_um",
            "combined_mean_front_abs_error_um",
            "combined_nearest_mean_um",
            "combined_nearest_max_um",
            "reached_time_s",
            "final_alpha_area_rel_drift",
            "max_abs_alpha_area_rel_drift",
            "score",
            "case_dir",
        ],
    )

    if bool(args.calibration_only):
        summary = {
            "profile": str(args.profile),
            "paper1_reduced": bool(args.paper1_reduced),
            "eps_list": [float(v) for v in eps_list] if eps_list else [],
            "alpha_ch_eps_list": [float(v) for v in alpha_ch_eps_list],
            "geometry_metadata": geometry_meta,
            "u_avg_m_per_s": float(u_avg),
            "u_max_m_per_s": float(u_max),
            "hydraulic_diameter_m": float(dh_m),
            "re_char_length_m": float(re_char_length_m),
            "diffuse_shear_traction": bool(args.diffuse_shear_traction),
            "diffuse_shear_model": str(args.diffuse_shear_model),
            "diffuse_shear_scale": float(args.diffuse_shear_scale),
            "diffuse_shear_eta": float(args.diffuse_shear_eta),
            "diffuse_shear_topweight": bool(args.diffuse_shear_topweight),
            "fluid_convection": str(args.fluid_convection),
            "phi_init_mode": str(args.phi_init_mode),
            "gamma_phi": float(args.gamma_phi),
            "phi_supg": float(args.phi_supg),
            "phi_cip": float(args.phi_cip),
            "alpha_supg": float(args.alpha_supg),
            "alpha_cip": float(args.alpha_cip),
            "include_skeleton_acceleration": bool(args.include_skeleton_acceleration),
            "rho_s0_tilde": float(rho_s0_tilde),
            "alpha_biot": float(args.alpha_biot),
            "skeleton_inertia_convection": str(args.skeleton_inertia_convection),
            "best_calibration": best.__dict__,
            "calibration_rows": [row.__dict__ for row in calibration_rows_sorted],
            "failed_calibration_cases": failed_calibration_cases,
            "artifacts": {"calibration_csv": str(calibration_csv)},
        }
        (outdir / "benchmark6_christan_channel_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))
        return

    production_rows: list[dict[str, object]] = []
    production_root = outdir / "production"
    finest_nx = int(max(nx_list))
    finest_case_dir: Path | None = None
    for nx in nx_list:
        ny = _ny_from_nx(int(nx))
        if math.isclose(float(best.eps), float(best.alpha_ch_eps), rel_tol=0.0, abs_tol=1.0e-18):
            case_dir = production_root / f"nx{int(nx):03d}_E{float(best.E):05.1f}_eps{_float_tag(float(best.eps))}"
        else:
            case_dir = production_root / (
                f"nx{int(nx):03d}_E{float(best.E):05.1f}_eps{_float_tag(float(best.eps))}_"
                f"cheps{_float_tag(float(best.alpha_ch_eps))}"
            )
        if int(nx) == int(best.nx):
            case_dir = Path(str(best.case_dir))
        else:
            cmd = _simulation_command(
                case_dir=case_dir,
                initial_csv=initial_csv,
                nx=int(nx),
                ny=int(ny),
                E=float(best.E),
                nu=float(args.nu),
                dt=float(args.dt),
                t_final=float(t_final),
                steady_time=float(steady_time),
                t_ramp=float(args.t_ramp),
                q=int(args.q),
                backend=str(args.backend),
                paper1_reduced=bool(args.paper1_reduced),
                linear_backend=str(args.linear_backend),
                linear_ksp_type=str(args.linear_ksp_type),
                linear_pc_type=str(args.linear_pc_type),
                linear_pc_factor_solver_type=str(args.linear_pc_factor_solver_type),
                linear_ksp_rtol=float(args.linear_ksp_rtol),
                linear_ksp_max_it=int(args.linear_ksp_max_it),
                rho_f=float(args.rho_f),
                u_avg=float(u_avg),
                kappa_inv=float(args.kappa_inv),
                phi_b=float(args.phi_b),
                phi_init_mode=str(args.phi_init_mode),
                fluid_convection=str(args.fluid_convection),
                include_skeleton_acceleration=bool(args.include_skeleton_acceleration),
                rho_s0_tilde=float(rho_s0_tilde),
                alpha_biot=float(args.alpha_biot),
                skeleton_inertia_convection=str(args.skeleton_inertia_convection),
                gamma_u=float(args.gamma_u),
                gamma_u_pin=float(args.gamma_u_pin),
                kinematics_scale=float(args.kinematics_scale),
                eps=float(best.eps),
                alpha_ch_eps=float(best.alpha_ch_eps),
                diffuse_shear_traction=bool(args.diffuse_shear_traction),
                diffuse_shear_model=str(args.diffuse_shear_model),
                diffuse_shear_scale=float(args.diffuse_shear_scale),
                diffuse_shear_eta=float(args.diffuse_shear_eta),
                diffuse_shear_topweight=bool(args.diffuse_shear_topweight),
                re_char_length_m=float(re_char_length_m),
                track_y_um_csv=str(track_y_um_csv),
                v_supg=float(args.v_supg),
                v_supg_mode=str(args.v_supg_mode),
                v_supg_c_nu=float(args.v_supg_c_nu),
                u_supg=float(args.u_supg),
                u_cip=float(args.u_cip),
                v_cip=float(args.v_cip),
                vS_cip=float(args.vS_cip),
                gamma_phi=float(args.gamma_phi),
                phi_supg=float(args.phi_supg),
                phi_cip=float(args.phi_cip),
                alpha_supg=float(args.alpha_supg),
                alpha_cip=float(args.alpha_cip),
                gamma_div=float(args.gamma_div),
                gamma_div_max=float(args.gamma_div_max),
                startup_staggered_predictor=bool(args.startup_staggered_predictor),
                startup_staggered_max_time=float(startup_staggered_max_time),
                startup_fluid_newton_tol=float(args.startup_fluid_newton_tol),
                startup_solid_newton_tol=float(args.startup_solid_newton_tol),
                startup_fluid_max_it=int(args.startup_fluid_max_it),
                startup_solid_max_it=int(args.startup_solid_max_it),
                startup_staggered_sweeps=int(args.startup_staggered_sweeps),
                startup_staggered_slip_tol=float(args.startup_staggered_slip_tol),
                newton_tol=float(args.newton_tol),
                max_it=int(args.max_it),
                dt_min=float(args.dt_min),
                accept_nonconverged_atol_factor=float(args.accept_nonconverged_atol_factor),
                refine_band=float(args.refine_band),
                refine_expand_layers=int(args.refine_expand_layers),
                vtk_every=int(args.vtk_every) if int(nx) == int(finest_nx) else 0,
            )
            _run(
                cmd,
                cwd=REPO_ROOT,
                log_path=case_dir / "run.log",
                skip_existing=bool(args.skip_existing) and _snapshot_ready(case_dir, times_s=[0.0, float(steady_time)]),
                sentinel=case_dir / "timeseries.csv",
                stream=bool(args.stream_subprocess),
            )
        payload = compare_case(
            out_dir=case_dir,
            target_time=float(steady_time),
            initial_time=0.0,
            y_levels_um=list(y_levels_um),
            geometry_dir=geometry_dir,
        )
        (case_dir / "compare_christan.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        combined = payload["combined"]
        primary = payload["primary"]
        alternate = payload["alternate"]
        ts_metrics = _read_timeseries_metrics(case_dir)
        row = {
            "nx": int(nx),
            "ny": int(ny),
            "E": float(best.E),
            "eps": float(best.eps),
            "alpha_ch_eps": float(best.alpha_ch_eps),
            "nu": float(args.nu),
            "Re_h": float(args.Re),
            "u_avg_m_per_s": float(u_avg),
            "u_max_m_per_s": float(u_max),
            "hydraulic_diameter_m": float(dh_m),
            "re_char_length_m": float(re_char_length_m),
            "steady_time_s": float(steady_time),
            "reached_time_s": float(ts_metrics["reached_time_s"]),
            "dt_s": float(args.dt),
            "phi_b": float(args.phi_b),
            "diffuse_shear_traction": bool(args.diffuse_shear_traction),
            "diffuse_shear_model": str(args.diffuse_shear_model),
            "diffuse_shear_scale": float(args.diffuse_shear_scale),
            "diffuse_shear_eta": float(args.diffuse_shear_eta),
            "diffuse_shear_topweight": bool(args.diffuse_shear_topweight),
            "phi_init_mode": str(args.phi_init_mode),
            "kappa_inv": float(args.kappa_inv),
            "fluid_convection": str(args.fluid_convection),
            "gamma_phi": float(args.gamma_phi),
            "phi_supg": float(args.phi_supg),
            "phi_cip": float(args.phi_cip),
            "alpha_supg": float(args.alpha_supg),
            "alpha_cip": float(args.alpha_cip),
            "include_skeleton_acceleration": float(1.0 if args.include_skeleton_acceleration else 0.0),
            "rho_s0_tilde": float(rho_s0_tilde),
            "alpha_biot": float(args.alpha_biot),
            "skeleton_inertia_convection": str(args.skeleton_inertia_convection),
            "final_alpha_area_rel_drift": float(ts_metrics["final_alpha_area_rel_drift"]),
            "max_abs_alpha_area_rel_drift": float(ts_metrics["max_abs_alpha_area_rel_drift"]),
            "primary_profile_rmse_um": float(primary["profile"]["rmse_um"]),
            "alt_profile_rmse_um": float(alternate["profile"]["rmse_um"]),
            "combined_profile_rmse_um": float(combined["profile_rmse_um"]),
            "combined_mean_dx_abs_error_um": float(combined["mean_dx_abs_error_um"]),
            "combined_mean_front_abs_error_um": float(combined["mean_front_abs_error_um"]),
            "combined_nearest_mean_um": float(combined["nearest_mean_um"]),
            "combined_nearest_max_um": float(combined["nearest_max_um"]),
            "score": float(combined["score"]),
            "case_dir": str(case_dir),
        }
        for y_um in y_levels_um:
            primary_block = primary["per_y"].get(str(int(y_um)), {})
            row[f"primary_dx_error_y{int(y_um)}um"] = float(primary_block.get("dx_error_um", float("nan")))
            row[f"primary_front_error_y{int(y_um)}um"] = float(primary_block.get("front_error_um", float("nan")))
        production_rows.append(row)
        if int(nx) == int(finest_nx):
            finest_case_dir = case_dir

    summary_csv = outdir / "benchmark6_christan_channel_summary.csv"
    fieldnames = sorted({key for row in production_rows for key in row})
    _write_csv(summary_csv, production_rows, fieldnames=fieldnames)

    if finest_case_dir is not None:
        _plot_contours(
            geometry_dir=geometry_dir,
            case_dir=finest_case_dir,
            steady_time=float(steady_time),
            out_path=outdir / "benchmark6_christan_channel_contours.png",
        )
        _plot_front_profile(
            geometry_dir=geometry_dir,
            case_dir=finest_case_dir,
            steady_time=float(steady_time),
            out_path=outdir / "benchmark6_christan_channel_front_profile.png",
        )
    _plot_mesh_sensitivity(production_rows, out_path=outdir / "benchmark6_christan_channel_mesh_sensitivity.png")

    summary = {
        "profile": str(args.profile),
        "paper1_reduced": bool(args.paper1_reduced),
        "eps_list": [float(v) for v in eps_list] if eps_list else [],
        "alpha_ch_eps_list": [float(v) for v in alpha_ch_eps_list],
        "geometry_metadata": geometry_meta,
        "u_avg_m_per_s": float(u_avg),
        "u_max_m_per_s": float(u_max),
        "hydraulic_diameter_m": float(dh_m),
        "re_char_length_m": float(re_char_length_m),
        "fluid_convection": str(args.fluid_convection),
        "phi_init_mode": str(args.phi_init_mode),
        "gamma_phi": float(args.gamma_phi),
        "phi_supg": float(args.phi_supg),
        "phi_cip": float(args.phi_cip),
        "alpha_supg": float(args.alpha_supg),
        "alpha_cip": float(args.alpha_cip),
        "include_skeleton_acceleration": bool(args.include_skeleton_acceleration),
        "rho_s0_tilde": float(rho_s0_tilde),
        "alpha_biot": float(args.alpha_biot),
        "skeleton_inertia_convection": str(args.skeleton_inertia_convection),
        "best_calibration": best.__dict__,
        "calibration_rows": [row.__dict__ for row in calibration_rows_sorted],
        "failed_calibration_cases": failed_calibration_cases,
        "production_rows": production_rows,
        "artifacts": {
            "calibration_csv": str(calibration_csv),
            "summary_csv": str(summary_csv),
            "contours_png": str(outdir / "benchmark6_christan_channel_contours.png"),
            "front_profile_png": str(outdir / "benchmark6_christan_channel_front_profile.png"),
            "mesh_png": str(outdir / "benchmark6_christan_channel_mesh_sensitivity.png"),
        },
    }
    (outdir / "benchmark6_christan_channel_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
