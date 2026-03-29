#!/usr/bin/env python3
"""Paper-ready runner for the reduced Paper 1 verification program.

This coordinator lives with the deformation-only drivers, while the generated
paper assets are still published into the local Overleaf workspace under
``examples/biofilms/deformation_part_numerical_study/69abdda4cb82b99640520d1c``.

Artifacts written by the suite:
  - raw runs under ``paper_ready/results/<run_tag>/``,
  - benchmark status under ``paper_ready/generated/``,
  - manuscript-ready tables/plots under ``verification/generated/``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PAPER_ROOT = REPO_ROOT / "examples/biofilms/deformation_part_numerical_study/69abdda4cb82b99640520d1c"
RESULTS_ROOT = PAPER_ROOT / "paper_ready" / "results"
GENERATED_ROOT = PAPER_ROOT / "paper_ready" / "generated"
VERIFICATION_ROOT = PAPER_ROOT / "verification"
VERIFICATION_GENERATED = VERIFICATION_ROOT / "generated"

NUMBA_DEBUG_ENV_KEYS = (
    "NUMBA_DEBUG",
    "NUMBA_DUMP_BYTECODE",
    "NUMBA_DUMP_IR",
    "NUMBA_DUMP_SSA",
    "NUMBA_DEBUG_ARRAY_OPT",
)

PROFILE_CONFIGS = {
    "smoke": {
        "mms": {
            "nx_list": "4,8",
            "q": "6",
            "q_error": "8",
            "dt": "0.05",
            "max_it": "10",
        },
        "transport": {
            "nx_list": "32",
            "q": "6",
            "q_metrics": "8",
            "theta": "0.5",
            "cfl": "0.35",
            "alpha_supg": "0.5",
            "alpha_cip": "0.0",
            "max_it": "12",
            "geom_every": "1",
            "geom_grid": "96",
            "final_grid": "256",
            "snapshot_grid": "320",
            "png_dpi": "180",
            "overrides": {
                "transport_shear_return": {
                    "cfl": "0.20",
                    "alpha_supg": "0.0",
                    "alpha_cip": "0.0",
                }
            },
        },
        "benchmark3": {
            "overrides": {
                "benchmark3_wang2014_layered": {
                    "k_list": "1e-2,1e-4",
                    "ny": "1200",
                    "plot_k": "1e-2,1e-4",
                    "png_dpi": "160",
                },
                "benchmark3_wang2014_staircase": {
                    "grid_pairs": "1e-2:32,1e-3:64",
                    "plot_k": "1e-2",
                    "png_dpi": "160",
                },
            }
        },
        "benchmark4": {
            "overrides": {
                "benchmark4_terzaghi": {
                    "ny_list": "8",
                    "nx_cells": "2",
                    "steps_per_ny": "2",
                    "Tv_final": "0.5",
                    "sample_tv": "0.20,0.50",
                    "png_dpi": "160",
                }
            }
        },
        "benchmark5": {
            "overrides": {
                "benchmark5_jonas_shear": {
                    "nx_list": "4,8",
                    "q": "8",
                    "q_error": "10",
                    "backend": "cpp",
                    "error_backend": "cpp",
                    "png_dpi": "180",
                }
            }
        },
        "benchmark6": {"overrides": {"benchmark6_christan_channel": {}, "benchmark6_blauert_channel": {}}},
    },
    "baseline": {
        "mms": {
            "nx_list": "4,8,16",
            "q": "6",
            "q_error": "8",
            "dt": "0.05",
            "max_it": "12",
        },
        "transport": {
            "nx_list": "32,64",
            "q": "6",
            "q_metrics": "8",
            "theta": "0.5",
            "cfl": "0.35",
            "alpha_supg": "0.5",
            "alpha_cip": "0.0",
            "max_it": "16",
            "geom_every": "2",
            "geom_grid": "128",
            "final_grid": "384",
            "snapshot_grid": "512",
            "png_dpi": "220",
            "overrides": {
                "transport_shear_return": {
                    "cfl": "0.20",
                    "alpha_supg": "0.0",
                    "alpha_cip": "0.0",
                }
            },
        },
        "benchmark3": {
            "overrides": {
                "benchmark3_wang2014_layered": {
                    "k_list": "1e-2,1e-3,1e-4,1e-5",
                    "ny": "4000",
                    "plot_k": "1e-2,1e-4",
                    "png_dpi": "220",
                },
                "benchmark3_wang2014_staircase": {
                    "grid_pairs": "1e-2:96,1e-3:192,1e-4:384",
                    "plot_k": "1e-2,1e-4",
                    "png_dpi": "220",
                },
            }
        },
        "benchmark4": {
            "overrides": {
                "benchmark4_terzaghi": {
                    "ny_list": "16,32",
                    "nx_cells": "4",
                    "steps_per_ny": "8",
                    "Tv_final": "1.0",
                    "sample_tv": "0.50,1.00",
                    "png_dpi": "220",
                }
            }
        },
        "benchmark5": {
            "overrides": {
                "benchmark5_jonas_shear": {
                    "nx_list": "8,16",
                    "q": "8",
                    "q_error": "10",
                    "backend": "cpp",
                    "error_backend": "cpp",
                    "png_dpi": "220",
                }
            }
        },
        "benchmark6": {"overrides": {"benchmark6_christan_channel": {}, "benchmark6_blauert_channel": {}}},
    },
    "production": {
        "mms": {
            "nx_list": "8,16,32",
            "q": "6",
            "q_error": "8",
            "dt": "0.05",
            "max_it": "12",
        },
        "transport": {
            "nx_list": "32,64",
            "q": "6",
            "q_metrics": "8",
            "theta": "0.5",
            "cfl": "0.35",
            "alpha_supg": "0.5",
            "alpha_cip": "0.0",
            "max_it": "18",
            "geom_every": "4",
            "geom_grid": "128",
            "final_grid": "512",
            "snapshot_grid": "640",
            "png_dpi": "280",
            "overrides": {
                "transport_shear_return": {
                    "cfl": "0.20",
                    "alpha_supg": "0.0",
                    "alpha_cip": "0.0",
                }
            },
        },
        "benchmark3": {
            "overrides": {
                "benchmark3_wang2014_layered": {
                    "k_list": "1e-2,1e-3,1e-4,1e-5,1e-6",
                    "ny": "8000",
                    "plot_k": "1e-2,1e-4",
                    "png_dpi": "280",
                },
                "benchmark3_wang2014_staircase": {
                    "grid_pairs": "1e-2:128,1e-3:256,1e-4:512",
                    "plot_k": "1e-2,1e-4",
                    "png_dpi": "280",
                },
            }
        },
        "benchmark4": {
            "overrides": {
                "benchmark4_terzaghi": {
                    "ny_list": "16,32,64",
                    "nx_cells": "4",
                    "steps_per_ny": "8",
                    "Tv_final": "1.0",
                    "sample_tv": "0.50,1.00",
                    "png_dpi": "280",
                }
            }
        },
        "benchmark5": {
            "overrides": {
                "benchmark5_jonas_shear": {
                    "nx_list": "8,16,32",
                    "q": "8",
                    "q_error": "10",
                    "backend": "cpp",
                    "error_backend": "cpp",
                    "png_dpi": "280",
                }
            }
        },
        "benchmark6": {"overrides": {"benchmark6_christan_channel": {}, "benchmark6_blauert_channel": {}}},
    },
}


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    family: str
    driver_rel: str
    output_key: str


CASE_SPECS = (
    CaseSpec(
        key="static",
        label="MMS: static smooth coupled state",
        family="mms",
        driver_rel="examples/biofilms/deformation_only_mms_static_convergence.py",
        output_key="static",
    ),
    CaseSpec(
        key="translation",
        label="MMS: rigid translation",
        family="mms",
        driver_rel="examples/biofilms/deformation_only_mms_translation_convergence.py",
        output_key="translation",
    ),
    CaseSpec(
        key="shear",
        label="MMS: affine shear / deformation",
        family="mms",
        driver_rel="examples/biofilms/deformation_only_mms_shear_convergence.py",
        output_key="shear",
    ),
    CaseSpec(
        key="transport_translation",
        label="Transport: rigid translation",
        family="transport",
        driver_rel="examples/biofilms/deformation_only_interface_transport_translation.py",
        output_key="translation",
    ),
    CaseSpec(
        key="transport_rotation",
        label="Transport: rigid-body rotation",
        family="transport",
        driver_rel="examples/biofilms/deformation_only_interface_transport_rotation.py",
        output_key="rotation",
    ),
    CaseSpec(
        key="transport_shear_return",
        label="Transport: shear-and-return",
        family="transport",
        driver_rel="examples/biofilms/deformation_only_interface_transport_shear_return.py",
        output_key="shear_return",
    ),
    CaseSpec(
        key="benchmark3_wang2014_layered",
        label="Two-domain vs one-domain: Wang2014 layered benchmark",
        family="benchmark3",
        driver_rel="examples/biofilms/benchmarks/wang/paper1_benchmark3_wang2014_layered.py",
        output_key="wang2014_layered",
    ),
    CaseSpec(
        key="benchmark3_wang2014_staircase",
        label="Two-domain vs one-domain: Wang2014 stepped benchmark",
        family="benchmark3",
        driver_rel="examples/biofilms/benchmarks/wang/paper1_benchmark3_wang2014_staircase.py",
        output_key="wang2014_staircase",
    ),
    CaseSpec(
        key="benchmark4_terzaghi",
        label="Poroelastic benchmark: Terzaghi consolidation",
        family="benchmark4",
        driver_rel="examples/biofilms/benchmarks/poroelastic/paper1_benchmark4_terzaghi_consolidation.py",
        output_key="terzaghi",
    ),
    CaseSpec(
        key="benchmark5_jonas_shear",
        label="FSI benchmark: Jonas-inspired exact shear",
        family="benchmark5",
        driver_rel="examples/biofilms/benchmarks/FSI/paper1_benchmark5_jonas_shear.py",
        output_key="jonas_shear",
    ),
    CaseSpec(
        key="benchmark6_christan_channel",
        label="Application benchmark: Christan Biofilm I channel deformation",
        family="benchmark6",
        driver_rel="examples/biofilms/benchmarks/christan/paper1_benchmark6_christan_channel.py",
        output_key="christan_channel",
    ),
    CaseSpec(
        key="benchmark6_blauert_channel",
        label="Application benchmark: Blauert channel deformation",
        family="benchmark6",
        driver_rel="examples/biofilms/benchmarks/blauert/paper1_benchmark6_blauert_channel.py",
        output_key="blauert_channel",
    ),
)


def _current_conda_env_name() -> str:
    name = str(os.environ.get("CONDA_DEFAULT_ENV", "")).strip()
    if name:
        return name
    prefix = str(os.environ.get("CONDA_PREFIX", "")).strip()
    if prefix:
        return Path(prefix).name
    return ""


def _conda_python(env_name: str, script_rel: str, *args: str) -> list[str]:
    script = str((REPO_ROOT / script_rel).resolve())
    if _current_conda_env_name() == str(env_name).strip():
        return [sys.executable, "-u", script, *[str(a) for a in args]]
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        str(env_name),
        "python",
        "-u",
        script,
        *[str(a) for a in args],
    ]


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in NUMBA_DEBUG_ENV_KEYS:
        env[key] = "0"
    tool_overrides = {
        "CC": shutil.which("gcc"),
        "CXX": shutil.which("g++"),
        "CPP": shutil.which("cpp"),
    }
    for key, tool in tool_overrides.items():
        if tool:
            env[key] = tool
    return env


def _selected_cases(raw: str) -> list[CaseSpec]:
    key_map = {spec.key: spec for spec in CASE_SPECS}
    if str(raw).strip().lower() in {"all", "*"}:
        return list(CASE_SPECS)
    out: list[CaseSpec] = []
    for item in str(raw).split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in key_map:
            raise ValueError(f"Unknown case {key!r}. Valid keys: {', '.join(key_map)}")
        out.append(key_map[key])
    if not out:
        raise ValueError("No cases selected.")
    return out


def _profile_config(profile: str) -> dict[str, dict[str, str]]:
    key = str(profile).strip().lower()
    if key not in PROFILE_CONFIGS:
        raise ValueError(f"Unknown profile {profile!r}.")
    return PROFILE_CONFIGS[key]


def _build_case_command(spec: CaseSpec, *, profile: str, run_dir: Path, env_name: str) -> list[str]:
    cfg = _profile_config(profile)[spec.family]
    if spec.family == "mms":
        return _conda_python(
            env_name,
            spec.driver_rel,
            "--backend",
            "cpp",
            "--error-backend",
            "cpp",
            "--nx-list",
            cfg["nx_list"],
            "--q",
            cfg["q"],
            "--q-error",
            cfg["q_error"],
            "--dt",
            cfg["dt"],
            "--theta",
            "1.0",
            "--newton-tol",
            "1e-10",
            "--max-it",
            cfg["max_it"],
            "--outdir",
            str(run_dir / spec.key),
            "--convergence",
        )
    if spec.family == "transport":
        merged = dict(cfg)
        overrides = merged.pop("overrides", {})
        merged.update(overrides.get(spec.key, {}))
        return _conda_python(
            env_name,
            spec.driver_rel,
            "--backend",
            "cpp",
            "--nx-list",
            merged["nx_list"],
            "--q",
            merged["q"],
            "--q-metrics",
            merged["q_metrics"],
            "--theta",
            merged["theta"],
            "--cfl",
            merged["cfl"],
            "--alpha-supg",
            merged["alpha_supg"],
            "--alpha-cip",
            merged["alpha_cip"],
            "--newton-tol",
            "1e-10",
            "--max-it",
            merged["max_it"],
            "--geom-every",
            merged["geom_every"],
            "--geom-grid",
            merged["geom_grid"],
            "--final-grid",
            merged["final_grid"],
            "--snapshot-grid",
            merged["snapshot_grid"],
            "--png-dpi",
            merged["png_dpi"],
            "--outdir",
            str(run_dir / spec.key),
            "--convergence",
            "--vtk-snapshots" if profile == "production" else "",
        )
    if spec.family == "benchmark4":
        merged = {}
        merged.update(cfg.get("overrides", {}).get(spec.key, {}))
        return _conda_python(
            env_name,
            spec.driver_rel,
            "--outdir",
            str(run_dir / spec.key),
            "--ny-list",
            merged["ny_list"],
            "--nx-cells",
            merged["nx_cells"],
            "--steps-per-ny",
            merged["steps_per_ny"],
            "--Tv-final",
            merged["Tv_final"],
            "--sample-tv",
            merged["sample_tv"],
            "--backend",
            "cpp",
            "--png-dpi",
            merged["png_dpi"],
            "--quiet",
        )
    if spec.family == "benchmark5":
        merged = {}
        merged.update(cfg.get("overrides", {}).get(spec.key, {}))
        cmd = _conda_python(
            env_name,
            spec.driver_rel,
            "--outdir",
            str(run_dir / spec.key),
            "--nx-list",
            merged["nx_list"],
            "--q",
            merged["q"],
            "--q-error",
            merged["q_error"],
            "--backend",
            merged["backend"],
            "--error-backend",
            merged["error_backend"],
            "--png-dpi",
            merged["png_dpi"],
            "--convergence",
        )
        if profile == "production":
            cmd.append("--vtk")
        return cmd
    if spec.family == "benchmark6":
        return _conda_python(
            env_name,
            spec.driver_rel,
            "--outdir",
            str(run_dir / spec.key),
            "--profile",
            str(profile),
        )
    merged = {}
    merged.update(cfg.get("overrides", {}).get(spec.key, {}))
    if spec.output_key == "wang2014_layered":
        return _conda_python(
            env_name,
            spec.driver_rel,
            "--outdir",
            str(run_dir / spec.key),
            "--k-list",
            merged["k_list"],
            "--ny",
            merged["ny"],
            "--plot-k",
            merged["plot_k"],
            "--png-dpi",
            merged["png_dpi"],
        )
    return _conda_python(
        env_name,
        spec.driver_rel,
        "--outdir",
        str(run_dir / spec.key),
        "--grid-pairs",
        merged["grid_pairs"],
        "--plot-k",
        merged["plot_k"],
        "--png-dpi",
        merged["png_dpi"],
    )


def _normalize_cmd(cmd: list[str]) -> list[str]:
    return [part for part in cmd if part]


def _case_dir(run_dir: Path, spec: CaseSpec) -> Path:
    if spec.family in {"benchmark3", "benchmark4", "benchmark5", "benchmark6"}:
        return run_dir / spec.key
    return run_dir / spec.key / spec.output_key


def _case_csv(case_dir: Path, spec: CaseSpec) -> Path:
    if spec.family == "mms":
        return case_dir / f"deformation_only_mms_{spec.output_key}.csv"
    if spec.family == "benchmark3":
        return case_dir / f"benchmark3_{spec.output_key}_summary.csv"
    if spec.family == "benchmark4":
        return case_dir / "benchmark4_terzaghi_summary.csv"
    if spec.family == "benchmark5":
        return case_dir / "benchmark5_jonas_shear_summary.csv"
    if spec.family == "benchmark6":
        return case_dir / "benchmark6_christan_channel_summary.csv"
    return case_dir / f"deformation_only_interface_transport_{spec.output_key}.csv"


def _case_convergence_png(case_dir: Path, spec: CaseSpec) -> Path:
    if spec.family == "mms":
        return case_dir / f"deformation_only_mms_{spec.output_key}_convergence.png"
    if spec.family == "benchmark3":
        return case_dir / f"benchmark3_{spec.output_key}_error_trends.png"
    if spec.family == "benchmark4":
        return case_dir / "benchmark4_terzaghi_error_trends.png"
    if spec.family == "benchmark5":
        return case_dir / "benchmark5_jonas_shear_convergence.png"
    if spec.family == "benchmark6":
        return case_dir / "benchmark6_christan_channel_mesh_sensitivity.png"
    return case_dir / f"deformation_only_interface_transport_{spec.output_key}_convergence.png"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _fmt_float(val: str | float | None, *, precision: int = 3) -> str:
    if val is None:
        return "-"
    try:
        fval = float(val)
    except Exception:
        return str(val)
    if not (fval == fval):
        return "-"
    return f"{fval:.{precision}e}"


def _fmt_int(val: str | float | None) -> str:
    if val is None:
        return "-"
    try:
        return str(int(float(val)))
    except Exception:
        return str(val)


def _latex_escape(text: str) -> str:
    out = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for src, dst in repl.items():
        out = out.replace(src, dst)
    return out


def _run_case(cmd: list[str], *, case_dir: Path, dry_run: bool) -> tuple[int, Path]:
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / "command.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("+ " + shlex.join(cmd) + "\n")
        if dry_run:
            return 0, log_path
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=_clean_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(proc.returncode), log_path


def _tail(path: Path, *, n: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-n:])


def _summary_metric(spec: CaseSpec, finest: dict[str, str]) -> str:
    if spec.family == "mms":
        if spec.output_key == "static":
            return (
                f"EOC(v_L2)={_fmt_float(finest.get('eoc_v_l2'))}, "
                f"EOC(p_L2)={_fmt_float(finest.get('eoc_p_l2'))}, "
                f"EOC(alpha_L2)={_fmt_float(finest.get('eoc_alpha_l2'))}, "
                f"EOC(B_L2)={_fmt_float(finest.get('eoc_B_l2'))}"
            )
        if spec.output_key == "translation":
            return (
                f"EOC(alpha_L2)={_fmt_float(finest.get('eoc_alpha_l2'))}, "
                f"EOC(B_L2)={_fmt_float(finest.get('eoc_B_l2'))}, "
                f"EOC(mu_L2)={_fmt_float(finest.get('eoc_mu_alpha_l2'))}, "
                f"EOC(alpha_H1)={_fmt_float(finest.get('eoc_alpha_h1'))}"
            )
        return (
            f"EOC(v_L2)={_fmt_float(finest.get('eoc_v_l2'))}, "
            f"EOC(vS_L2)={_fmt_float(finest.get('eoc_vS_l2'))}, "
            f"EOC(alpha_L2)={_fmt_float(finest.get('eoc_alpha_l2'))}, "
            f"EOC(B_L2)={_fmt_float(finest.get('eoc_B_l2'))}"
        )
    if spec.family == "benchmark3":
        if spec.output_key == "wang2014_staircase":
            return (
                f"K={_fmt_float(finest.get('K'), precision=0)}, "
                f"n={_fmt_int(finest.get('nxy'))}, "
                f"L2_f={_fmt_float(finest.get('l2_fluid'))}, "
                f"L2_p={_fmt_float(finest.get('l2_porous'))}, "
                f"profile_inf={_fmt_float(finest.get('profile_linf'))}"
            )
        return (
            f"K={_fmt_float(finest.get('K'), precision=0)}, "
            f"L2_f={_fmt_float(finest.get('l2_fluid'))}, "
            f"L2_p={_fmt_float(finest.get('l2_porous'))}, "
                f"H1_f={_fmt_float(finest.get('h1_fluid'))}, "
                f"H1_p={_fmt_float(finest.get('h1_porous'))}"
            )
    if spec.family == "benchmark4":
        return (
            f"n_y={_fmt_int(finest.get('ny'))}, "
            f"max pL2={_fmt_float(finest.get('max_pbar_l2'))}, "
            f"max pFieldL2={_fmt_float(finest.get('max_pbar_field_l2'))}, "
            f"max pInf={_fmt_float(finest.get('max_pbar_linf'))}, "
            f"max s={_fmt_float(finest.get('max_settlement_bar_error'))}"
        )
    if spec.family == "benchmark5":
        return (
            f"n_x={_fmt_int(finest.get('nx'))}, "
            f"EOC(v_L2)={_fmt_float(finest.get('eoc_v_l2'))}, "
            f"EOC(u_L2)={_fmt_float(finest.get('eoc_u_l2'))}, "
            f"EOC(alpha_L2)={_fmt_float(finest.get('eoc_alpha_l2'))}, "
            f"EOC(B_L2)={_fmt_float(finest.get('eoc_B_l2'))}, "
            f"|e_u(\\Gamma)|={_fmt_float(finest.get('u_interface_error'))}"
        )
    if spec.family == "benchmark6":
        return (
            f"n_x={_fmt_int(finest.get('nx'))}, "
            f"contour RMSE={_fmt_float(finest.get('combined_profile_rmse_um'))} um, "
            f"mean |dx|={_fmt_float(finest.get('combined_mean_dx_abs_error_um'))} um, "
            f"nearest max={_fmt_float(finest.get('combined_nearest_max_um'))} um"
        )
    return (
        f"max|dm|={_fmt_float(finest.get('max_mass_drift'))}, "
        f"max ec={_fmt_float(finest.get('max_geom_centroid_err'))}, "
        f"max dG={_fmt_float(finest.get('max_thickness_drift'))}, "
        f"eshape(T)={_fmt_float(finest.get('final_shape_mismatch'))}"
    )


def _summary_note(spec: CaseSpec, finest: dict[str, str]) -> str:
    if spec.family == "mms":
        if spec.output_key == "static":
            return "Smooth-reference reduced alpha-B MMS; vector fields recover Q2 rates and scalar support fields recover Q1 rates."
        if spec.output_key == "translation":
            nx = _fmt_int(finest.get("nx"))
            return (
                "Reduced alpha-B transport MMS; the reported convergence targets are "
                f"alpha, B, and mu_alpha only, and they reach the asymptotic regime by nx={nx}."
            )
        return "Fully coupled reduced alpha-B MMS; fluid, skeleton, reference-map, and support fields are all active."
    if spec.family == "benchmark3":
        if spec.output_key == "wang2014_staircase":
            return (
                "Wang2014 Example 6.2 style stepped-interface reduced benchmark; the diffuse "
                "transition-layer field is compared against a sharp split reference on "
                "interface-resolving grids for a non-flat geometry."
            )
        return (
            "Wang2014 Example 6.1 style layered benchmark; the one-domain transition profile is "
            "compared against a two-domain sharp-interface reference with regional L2 and semi-H1 errors."
        )
    if spec.family == "benchmark4":
        return (
            "Terzaghi single-drainage consolidation benchmark; normalized pore-pressure profiles "
            "and top-settlement history are compared against the analytic series solution, with "
            "profile errors reported after the initial drained-boundary layer is resolved."
        )
    if spec.family == "benchmark5":
        return (
            "Jonas-inspired exact shear benchmark for the reduced alpha-B mechanics block; "
            "tangential traction is localized on the conserved diffuse interface and the "
            "numerical solution is compared against a closed-form reference state with active "
            "support transport and explicit solid-fraction evolution."
        )
    if spec.family == "benchmark6":
        return (
            "Christan Biofilm I channel benchmark; the reduced one-domain deformation path "
            "with fixed body porosity and conserved CH interface transport is calibrated "
            "against the unloaded/loaded OCT contour pair reported by Picioreanu et al., then "
            "checked on a mesh ladder with contour overlays and front-displacement profiles."
        )
    if spec.output_key == "translation":
        return "Rigid-body transport benchmark; Crank-Nicolson with light SUPG is used to suppress artificial interface decay."
    if spec.output_key == "rotation":
        return "Large rigid-body rotation benchmark; this is the strictest geometry-preservation case for the reduced transport operator."
    return "Large affine shear-and-return benchmark; the final-time shape recovery measures reversibility of the transported indicator."


def _table_spec(spec: CaseSpec) -> dict[str, list[tuple[str, str]]]:
    if spec.family == "mms":
        if spec.output_key == "static":
            return {
                "l2": [
                    ("nx", r"$n_x$"),
                    ("v_l2", r"$\|e_{\bm v}\|_{L^2}$"),
                    ("p_l2", r"$\|e_p\|_{L^2}$"),
                    ("alpha_l2", r"$\|e_\alpha\|_{L^2}$"),
                    ("B_l2", r"$\|e_B\|_{L^2}$"),
                    ("mu_alpha_l2", r"$\|e_{\mu_\alpha}\|_{L^2}$"),
                    ("newton_iters", "Newton"),
                ],
                "h1": [
                    ("nx", r"$n_x$"),
                    ("v_h1", r"$\|e_{\bm v}\|_{H^1}$"),
                    ("alpha_h1", r"$\|e_\alpha\|_{H^1}$"),
                    ("B_h1", r"$\|e_B\|_{H^1}$"),
                    ("mu_alpha_h1", r"$\|e_{\mu_\alpha}\|_{H^1}$"),
                    ("newton_iters", "Newton"),
                ],
            }
        if spec.output_key == "translation":
            return {
                "l2": [
                    ("nx", r"$n_x$"),
                    ("alpha_l2", r"$\|e_\alpha\|_{L^2}$"),
                    ("B_l2", r"$\|e_B\|_{L^2}$"),
                    ("mu_alpha_l2", r"$\|e_{\mu_\alpha}\|_{L^2}$"),
                    ("newton_iters", "Newton"),
                ],
                "h1": [
                    ("nx", r"$n_x$"),
                    ("alpha_h1", r"$\|e_\alpha\|_{H^1}$"),
                    ("B_h1", r"$\|e_B\|_{H^1}$"),
                    ("mu_alpha_h1", r"$\|e_{\mu_\alpha}\|_{H^1}$"),
                    ("newton_iters", "Newton"),
                ],
            }
        return {
            "l2": [
                ("nx", r"$n_x$"),
                ("v_l2", r"$\|e_{\bm v}\|_{L^2}$"),
                ("p_l2", r"$\|e_p\|_{L^2}$"),
                ("vS_l2", r"$\|e_{\bm v^S}\|_{L^2}$"),
                ("u_l2", r"$\|e_{\bm u}\|_{L^2}$"),
                ("alpha_l2", r"$\|e_\alpha\|_{L^2}$"),
                ("B_l2", r"$\|e_B\|_{L^2}$"),
                ("mu_alpha_l2", r"$\|e_{\mu_\alpha}\|_{L^2}$"),
                ("newton_iters", "Newton"),
            ],
            "h1": [
                ("nx", r"$n_x$"),
                ("v_h1", r"$\|e_{\bm v}\|_{H^1}$"),
                ("vS_h1", r"$\|e_{\bm v^S}\|_{H^1}$"),
                ("u_h1", r"$\|e_{\bm u}\|_{H^1}$"),
                ("alpha_h1", r"$\|e_\alpha\|_{H^1}$"),
                ("B_h1", r"$\|e_B\|_{H^1}$"),
                ("mu_alpha_h1", r"$\|e_{\mu_\alpha}\|_{H^1}$"),
                ("newton_iters", "Newton"),
            ],
        }
    if spec.family == "benchmark3":
        if spec.output_key == "wang2014_staircase":
            return {
                "l2": [
                    ("K", r"$K$"),
                    ("nxy", r"$n$"),
                    ("l2_fluid", r"$\|u_{\mathrm{one}}-u_{\mathrm{two}}\|_{L^2(\Omega_f)}$"),
                    ("l2_porous", r"$\|u_{\mathrm{one}}-u_{\mathrm{two}}\|_{L^2(\Omega_p)}$"),
                    ("profile_linf", r"$\|e_{\mathrm{prof}}\|_{L^\infty}$"),
                    ("max_field_abs_error", r"$\|e\|_{L^\infty(\Omega)}$"),
                ],
                "h1": [
                    ("K", r"$K$"),
                    ("nxy", r"$n$"),
                    ("h1_fluid", r"$|\!|u_{\mathrm{one}}-u_{\mathrm{two}}|\!|_{H^1(\Omega_f)}$"),
                    ("h1_porous", r"$|\!|u_{\mathrm{one}}-u_{\mathrm{two}}|\!|_{H^1(\Omega_p)}$"),
                    ("profile_l2", r"$\|e_{\mathrm{prof}}\|_{L^2}$"),
                    ("interface_band_linf", r"$\|e\|_{L^\infty(\Gamma_{\mathrm{band}})}$"),
                ],
            }
        return {
            "l2": [
                ("K", r"$K$"),
                ("l2_fluid", r"$\|u_{\mathrm{one}}-u_{\mathrm{two}}\|_{L^2(\Omega_f)}$"),
                ("rho_l2_fluid", r"$\rho_f^{L^2}$"),
                ("l2_porous", r"$\|u_{\mathrm{one}}-u_{\mathrm{two}}\|_{L^2(\Omega_p)}$"),
                ("rho_l2_porous", r"$\rho_p^{L^2}$"),
            ],
            "h1": [
                ("K", r"$K$"),
                ("h1_fluid", r"$|\!|u_{\mathrm{one}}-u_{\mathrm{two}}|\!|_{H^1(\Omega_f)}$"),
                ("rho_h1_fluid", r"$\rho_f^{H^1}$"),
                ("h1_porous", r"$|\!|u_{\mathrm{one}}-u_{\mathrm{two}}|\!|_{H^1(\Omega_p)}$"),
                ("rho_h1_porous", r"$\rho_p^{H^1}$"),
            ]
        }
    if spec.family == "benchmark4":
        return {
            "summary": [
                ("ny", r"$n_y$"),
                ("num_time_steps", "steps"),
                ("max_pbar_l2", r"$e_{\bar p}^{L^2}$"),
                ("max_pbar_field_l2", r"$e_p^{L^2}/p_0$"),
                ("max_pbar_linf", r"$e_{\bar p}^{L^\infty}$"),
                ("max_mid_pressure_bar_error", r"$e_{\bar p}^{\mathrm{mid}}$"),
                ("max_settlement_bar_error", r"$e_{\bar s}$"),
                ("final_settlement_bar_error", r"$e_{\bar s}(T)$"),
            ]
        }
    if spec.family == "benchmark5":
        return {
            "l2": [
                ("nx", r"$n_x$"),
                ("v_l2", r"$\|e_{\bm v}\|_{L^2}$"),
                ("p_l2", r"$\|e_p\|_{L^2}$"),
                ("u_l2", r"$\|e_{\bm u}\|_{L^2}$"),
                ("alpha_l2", r"$\|e_\alpha\|_{L^2}$"),
                ("B_l2", r"$\|e_B\|_{L^2}$"),
                ("mu_alpha_l2", r"$\|e_{\mu_\alpha}\|_{L^2}$"),
                ("u_interface_error", r"$|e_{u,\Gamma}|$"),
                ("newton_iters", "Newton"),
            ],
            "h1": [
                ("nx", r"$n_x$"),
                ("v_h1", r"$\|e_{\bm v}\|_{H^1}$"),
                ("u_h1", r"$\|e_{\bm u}\|_{H^1}$"),
                ("alpha_h1", r"$\|e_\alpha\|_{H^1}$"),
                ("B_h1", r"$\|e_B\|_{H^1}$"),
                ("mu_alpha_h1", r"$\|e_{\mu_\alpha}\|_{H^1}$"),
                ("newton_iters", "Newton"),
            ],
        }
    if spec.family == "benchmark6":
        return {
            "summary": [
                ("nx", r"$n_x$"),
                ("ny", r"$n_y$"),
                ("combined_profile_rmse_um", r"contour RMSE [$\mu$m]"),
                ("combined_mean_dx_abs_error_um", r"mean $|\Delta x|$ error [$\mu$m]"),
                ("combined_mean_front_abs_error_um", r"mean front error [$\mu$m]"),
                ("combined_nearest_max_um", r"max nearest distance [$\mu$m]"),
            ]
        }
    return {
        "summary": [
            ("nx", r"$n_x$"),
            ("max_mass_drift", r"$\max_n |\delta m^n|$"),
            ("max_geom_centroid_err", r"$\max_n e_c^n$"),
            ("max_thickness_drift", r"$\max_n |\delta_\Gamma^n|$"),
            ("final_area_error", r"$|\delta A_{0.5}(T)|$"),
            ("final_shape_mismatch", r"$e_{\mathrm{shape}}(T)$"),
            ("alpha_l2_final", r"$\|e_\alpha(T)\|_{L^2}$"),
        ]
    }


def _format_table_value(key: str, val: str) -> str:
    if key in {"nx", "ny", "nxy", "newton_iters", "num_time_steps"}:
        return _fmt_int(val)
    return _fmt_float(val)


def _resolution_label(spec: CaseSpec, finest: dict[str, str]) -> str:
    if spec.family == "benchmark3":
        if spec.output_key == "wang2014_staircase":
            return f"n={_fmt_int(finest.get('nxy'))}"
        return f"n_y={_fmt_int(finest.get('ny'))}"
    if spec.family == "benchmark4":
        return f"n_y={_fmt_int(finest.get('ny'))}"
    if spec.family == "benchmark5":
        return f"n_x={_fmt_int(finest.get('nx'))}"
    if spec.family == "benchmark6":
        return f"n_x={_fmt_int(finest.get('nx'))}"
    return _fmt_int(finest.get("nx"))


def _write_case_table(
    path: Path,
    *,
    rows: list[dict[str, str]],
    columns: list[tuple[str, str]],
    resize_to_line: bool = False,
    font_cmd: str = "small",
    tabcolsep_pt: float | None = None,
    trim_outer_padding: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    align = "".join("r" for _ in columns)
    if bool(trim_outer_padding):
        align = f"@{{}}{align}@{{}}"
    lines = [f"{{\\{str(font_cmd).strip() or 'small'}", "\\begin{center}"]
    if tabcolsep_pt is not None:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{float(tabcolsep_pt):.1f}pt}}")
    if bool(resize_to_line):
        lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.extend(
        [
            f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(label for _, label in columns) + r" \\",
        "\\midrule",
        ]
    )
    for row in rows:
        vals = [_format_table_value(key, row.get(key, "")) for key, _ in columns]
        lines.append(" & ".join(vals) + r" \\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}" if bool(resize_to_line) else "",
            "\\end{center}",
            "}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _publish_case_assets(spec: CaseSpec, *, case_dir: Path, rows: list[dict[str, str]], profile: str, run_tag: str) -> None:
    VERIFICATION_GENERATED.mkdir(parents=True, exist_ok=True)
    spec_tables = _table_spec(spec)

    if spec.family == "mms":
        _write_case_table(
            VERIFICATION_GENERATED / f"mms_{spec.output_key}_l2_table.tex",
            rows=rows,
            columns=spec_tables["l2"],
        )
        _write_case_table(
            VERIFICATION_GENERATED / f"mms_{spec.output_key}_h1_table.tex",
            rows=rows,
            columns=spec_tables["h1"],
        )
        shutil.copy2(_case_csv(case_dir, spec), VERIFICATION_GENERATED / f"mms_{spec.output_key}_{profile}.csv")
        png_src = _case_convergence_png(case_dir, spec)
        if png_src.exists():
            shutil.copy2(png_src, VERIFICATION_GENERATED / f"mms_{spec.output_key}_convergence.png")
        summary_name = f"mms_{spec.output_key}_summary.json"
    elif spec.family == "transport":
        _write_case_table(
            VERIFICATION_GENERATED / f"interface_transport_{spec.output_key}_table.tex",
            rows=rows,
            columns=spec_tables["summary"],
        )
        shutil.copy2(_case_csv(case_dir, spec), VERIFICATION_GENERATED / f"interface_transport_{spec.output_key}_{profile}.csv")
        png_src = _case_convergence_png(case_dir, spec)
        if png_src.exists():
            shutil.copy2(png_src, VERIFICATION_GENERATED / f"interface_transport_{spec.output_key}_convergence.png")
        finest_nx = int(float(rows[-1]["nx"]))
        snap_src = case_dir / f"nx{finest_nx:03d}" / f"{spec.output_key}_snapshots.png"
        if snap_src.exists():
            shutil.copy2(snap_src, VERIFICATION_GENERATED / f"interface_transport_{spec.output_key}_snapshots.png")
        summary_name = f"interface_transport_{spec.output_key}_summary.json"
    elif spec.family == "benchmark3":
        _write_case_table(
            VERIFICATION_GENERATED / f"benchmark3_{spec.output_key}_l2_table.tex",
            rows=rows,
            columns=spec_tables["l2"],
        )
        _write_case_table(
            VERIFICATION_GENERATED / f"benchmark3_{spec.output_key}_h1_table.tex",
            rows=rows,
            columns=spec_tables["h1"],
        )
        shutil.copy2(_case_csv(case_dir, spec), VERIFICATION_GENERATED / f"benchmark3_{spec.output_key}_{profile}.csv")
        extra_pngs = [
            (f"benchmark3_{spec.output_key}_error_trends.png", f"benchmark3_{spec.output_key}_error_trends.png"),
        ]
        if spec.output_key == "wang2014_layered":
            extra_pngs.extend(
                [
                    ("benchmark3_wang2014_layered_profiles.png", "benchmark3_wang2014_layered_profiles.png"),
                ]
            )
        else:
            extra_pngs.extend(
                [
                    ("benchmark3_wang2014_staircase_geometry.png", "benchmark3_wang2014_staircase_geometry.png"),
                    ("benchmark3_wang2014_staircase_profiles.png", "benchmark3_wang2014_staircase_profiles.png"),
                    ("benchmark3_wang2014_staircase_fields.png", "benchmark3_wang2014_staircase_fields.png"),
                ]
            )
        for src_name, dst_name in extra_pngs:
            src = case_dir / src_name
            if src.exists():
                shutil.copy2(src, VERIFICATION_GENERATED / dst_name)
        summary_name = f"benchmark3_{spec.output_key}_summary.json"
    elif spec.family == "benchmark5":
        _write_case_table(
            VERIFICATION_GENERATED / "benchmark5_jonas_shear_l2_table.tex",
            rows=rows,
            columns=spec_tables["l2"],
        )
        _write_case_table(
            VERIFICATION_GENERATED / "benchmark5_jonas_shear_h1_table.tex",
            rows=rows,
            columns=spec_tables["h1"],
        )
        shutil.copy2(_case_csv(case_dir, spec), VERIFICATION_GENERATED / f"benchmark5_{spec.output_key}_{profile}.csv")
        extra_pngs = [
            ("benchmark5_jonas_shear_convergence.png", "benchmark5_jonas_shear_convergence.png"),
            ("benchmark5_jonas_shear_profiles.png", "benchmark5_jonas_shear_profiles.png"),
            ("benchmark5_jonas_shear_fields.png", "benchmark5_jonas_shear_fields.png"),
        ]
        for src_name, dst_name in extra_pngs:
            src = case_dir / src_name
            if src.exists():
                shutil.copy2(src, VERIFICATION_GENERATED / dst_name)
        summary_name = "benchmark5_jonas_shear_summary.json"
    elif spec.family == "benchmark6":
        _write_case_table(
            VERIFICATION_GENERATED / "benchmark6_christan_channel_table.tex",
            rows=rows,
            columns=spec_tables["summary"],
        )
        shutil.copy2(_case_csv(case_dir, spec), VERIFICATION_GENERATED / f"benchmark6_{spec.output_key}_{profile}.csv")
        extra_files = [
            ("benchmark6_christan_channel_calibration.csv", "benchmark6_christan_channel_calibration.csv"),
            ("benchmark6_christan_channel_contours.png", "benchmark6_christan_channel_contours.png"),
            ("benchmark6_christan_channel_front_profile.png", "benchmark6_christan_channel_front_profile.png"),
            ("benchmark6_christan_channel_mesh_sensitivity.png", "benchmark6_christan_channel_mesh_sensitivity.png"),
        ]
        for src_name, dst_name in extra_files:
            src = case_dir / src_name
            if src.exists():
                shutil.copy2(src, VERIFICATION_GENERATED / dst_name)
        summary_name = "benchmark6_christan_channel_summary.json"
    else:
        _write_case_table(
            VERIFICATION_GENERATED / "benchmark4_terzaghi_table.tex",
            rows=rows,
            columns=spec_tables["summary"],
            font_cmd="scriptsize",
            tabcolsep_pt=2.0,
            trim_outer_padding=True,
        )
        shutil.copy2(_case_csv(case_dir, spec), VERIFICATION_GENERATED / f"benchmark4_{spec.output_key}_{profile}.csv")
        extra_pngs = [
            ("benchmark4_terzaghi_history.png", "benchmark4_terzaghi_history.png"),
            ("benchmark4_terzaghi_profiles.png", "benchmark4_terzaghi_profiles.png"),
            ("benchmark4_terzaghi_error_trends.png", "benchmark4_terzaghi_error_trends.png"),
        ]
        for src_name, dst_name in extra_pngs:
            src = case_dir / src_name
            if src.exists():
                shutil.copy2(src, VERIFICATION_GENERATED / dst_name)
        summary_name = "benchmark4_terzaghi_summary.json"

    summary = {
        "case": spec.key,
        "profile": profile,
        "run_tag": run_tag,
        "rows": rows,
        "finest": rows[-1] if rows else None,
    }
    (VERIFICATION_GENERATED / summary_name).write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_summary(entries: list[dict[str, str]], *, profile: str, run_tag: str) -> None:
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "profile": str(profile),
        "run_tag": str(run_tag),
        "generated_at": stamp,
        "entries": entries,
    }
    (GENERATED_ROOT / "benchmark_status.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    md_lines = [
        "# Paper 1 paper-ready status",
        "",
        f"- profile: `{profile}`",
        f"- run tag: `{run_tag}`",
        f"- generated at: `{stamp}`",
        "",
        "| Case | Status | Resolution | Summary |",
        "| --- | --- | --- | --- |",
    ]
    for entry in entries:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(entry.get("label", "-")),
                    str(entry.get("status", "-")),
                    str(entry.get("finest_nx", "-")),
                    str(entry.get("summary_metric", "-")),
                ]
            )
            + " |"
        )
    md_lines.append("")
    (GENERATED_ROOT / "benchmark_status.md").write_text("\n".join(md_lines), encoding="utf-8")

    tex_lines = [
        "{\\small",
        "\\begin{center}",
        "\\begin{tabular}{p{0.27\\linewidth}p{0.10\\linewidth}p{0.09\\linewidth}p{0.37\\linewidth}}",
        "\\toprule",
        "Case & Status & Resolution & Summary \\\\",
        "\\midrule",
    ]
    for entry in entries:
        tex_lines.append(
            " & ".join(
                [
                    _latex_escape(entry.get("label", "-")),
                    _latex_escape(entry.get("status", "-")),
                    _latex_escape(entry.get("finest_nx", "-")),
                    _latex_escape(entry.get("summary_metric", "-")),
                ]
            )
            + r" \\"
        )
    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{center}",
            "}",
            "",
        ]
    )
    (GENERATED_ROOT / "benchmark_status_table.tex").write_text("\n".join(tex_lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper-ready reduced deformation-only verification suite.")
    ap.add_argument("--profile", type=str, default="baseline", choices=tuple(PROFILE_CONFIGS))
    ap.add_argument("--cases", type=str, default="all")
    ap.add_argument("--conda-env", type=str, default="fenicsx")
    ap.add_argument("--run-tag", type=str, default="")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    profile = str(args.profile).strip().lower()
    run_tag = str(args.run_tag).strip() or f"{profile}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RESULTS_ROOT / run_tag
    specs = _selected_cases(args.cases)

    if args.dry_run:
        for spec in specs:
            cmd = _normalize_cmd(_build_case_command(spec, profile=profile, run_dir=run_dir, env_name=str(args.conda_env)))
            print(shlex.join(cmd))
        return

    entries: list[dict[str, str]] = []
    for spec in specs:
        case_dir = _case_dir(run_dir, spec)
        csv_path = _case_csv(case_dir, spec)
        log_path = case_dir / "command.log"
        if bool(args.skip_existing) and csv_path.exists():
            rc = 0
            if not log_path.exists():
                log_path.write_text("# reused existing outputs\n", encoding="utf-8")
            status = "completed"
        else:
            cmd = _normalize_cmd(_build_case_command(spec, profile=profile, run_dir=run_dir, env_name=str(args.conda_env)))
            rc, log_path = _run_case(cmd, case_dir=case_dir, dry_run=False)
            status = "completed" if rc == 0 else "failed"

        if rc == 0 and csv_path.exists():
            rows = _read_csv_rows(csv_path)
            finest = rows[-1]
            _publish_case_assets(spec, case_dir=case_dir, rows=rows, profile=profile, run_tag=run_tag)
            entries.append(
                {
                    "case": spec.key,
                    "label": spec.label,
                    "status": status,
                    "profile": profile,
                    "run_tag": run_tag,
                    "finest_nx": _resolution_label(spec, finest),
                    "summary_metric": _summary_metric(spec, finest),
                    "note": _summary_note(spec, finest),
                    "case_dir": str(case_dir.relative_to(PAPER_ROOT)),
                    "log": str(log_path.relative_to(PAPER_ROOT)),
                }
            )
            continue

        entries.append(
            {
                "case": spec.key,
                "label": spec.label,
                "status": "failed",
                "profile": profile,
                "run_tag": run_tag,
                "finest_nx": "-",
                "summary_metric": "-",
                "note": _tail(log_path) if log_path.exists() else "no log available",
                "case_dir": str(case_dir.relative_to(PAPER_ROOT)),
                "log": str(log_path.relative_to(PAPER_ROOT)),
            }
        )

    _write_summary(entries, profile=profile, run_tag=run_tag)
    print(json.dumps({"profile": profile, "run_tag": run_tag, "entries": entries}, indent=2))


if __name__ == "__main__":
    main()
