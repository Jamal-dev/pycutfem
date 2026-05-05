#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SweepConfig:
    config_id: str
    label: str
    formulation: str
    conservation_mode: str
    solver_family: str
    extra_args: tuple[str, ...]


def _parse_csv_floats(text: str) -> list[float]:
    vals: list[float] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        raise ValueError("Expected at least one numeric kappa value.")
    return vals


def _kappa_slug(kappa: float) -> str:
    text = f"{kappa:.0e}".replace("+0", "").replace("+", "")
    return text


def _float_or_nan(value) -> float:
    if value in (None, "", "none", "nan", "NaN"):
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line:
            return line
    return ""


def _load_case_summary(summary_csv: Path) -> dict[str, str]:
    if not summary_csv.exists():
        return {}
    with summary_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return dict(rows[-1]) if rows else {}


def _extract_log_metrics(log_text: str) -> dict[str, object]:
    g_vals = [float(v) for v in re.findall(r"\|G\|_∞\s*=\s*([0-9.eE+-]+)", log_text)]
    rraw_vals = [float(v) for v in re.findall(r"\|R_raw\|_∞\s*=\s*([0-9.eE+-]+)", log_text)]
    dt_reductions = re.findall(r"reducing Δt → ([0-9.eE+-]+)", log_text)
    accepted_steps = len(re.findall(r"accepted step", log_text, flags=re.IGNORECASE))
    terminal_reason = ""
    terminal_patterns = [
        r"\[warn\] solve terminated early: ([^\n]+)",
        r"Newton failed at step \d+ with dt=[^:]+: ([^\n]+)",
        r"Line search failed: ([^\n]+)",
    ]
    for pattern in terminal_patterns:
        matches = re.findall(pattern, log_text)
        if matches:
            terminal_reason = str(matches[-1]).strip()
            break
    health_label = "unknown"
    if rraw_vals:
        first_r = max(abs(rraw_vals[0]), 1.0e-300)
        best_r = min(abs(v) for v in rraw_vals)
        ratio = best_r / first_r
        if best_r <= 1.0e-6:
            health_label = "near_tol"
        elif ratio <= 1.0e-3:
            health_label = "strong_descent"
        elif ratio <= 1.0e-2:
            health_label = "good_descent"
        elif ratio <= 1.0e-1:
            health_label = "descent"
        elif ratio <= 5.0e-1:
            health_label = "weak_descent"
        else:
            health_label = "stagnant"
    elif g_vals:
        first_g = max(abs(g_vals[0]), 1.0e-300)
        best_g = min(abs(v) for v in g_vals)
        ratio = best_g / first_g
        health_label = "descent" if ratio <= 1.0e-1 else "stagnant"
    return {
        "first_G_inf": (g_vals[0] if g_vals else float("nan")),
        "best_G_inf": (min(g_vals) if g_vals else float("nan")),
        "last_G_inf": (g_vals[-1] if g_vals else float("nan")),
        "first_R_raw_inf": (rraw_vals[0] if rraw_vals else float("nan")),
        "best_R_raw_inf": (min(rraw_vals) if rraw_vals else float("nan")),
        "last_R_raw_inf": (rraw_vals[-1] if rraw_vals else float("nan")),
        "dt_reductions_used": float(len(dt_reductions)),
        "accepted_step_lines": float(accepted_steps),
        "terminal_reason": terminal_reason,
        "terminal_line": _last_nonempty_line(log_text),
        "health_label": health_label,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(
    path: Path,
    *,
    rows: list[dict[str, object]],
    configs: list[SweepConfig],
    dt0: float,
    dt_min: float,
    t_final: float,
    max_reductions: int,
) -> None:
    config_map = {cfg.config_id: cfg for cfg in configs}
    lines: list[str] = []
    lines.append("# Benchmark 7 State Sweep")
    lines.append("")
    lines.append(f"- generated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- mesh: 12x18")
    lines.append(f"- initial_dt: {dt0:.6e}")
    lines.append(f"- dt_min: {dt_min:.6e}")
    lines.append(f"- t_final: {t_final:.6e}")
    lines.append(f"- allowed_dt_reductions: {max_reductions}")
    lines.append("")
    lines.append("## Configuration Interpretation")
    lines.append("")
    for cfg in configs:
        lines.append(
            f"- `{cfg.config_id}`: {cfg.label}; formulation={cfg.formulation}; "
            f"alpha_conservation={cfg.conservation_mode}; solver={cfg.solver_family}"
        )
    for kappa in sorted({float(row["kappa"]) for row in rows}):
        lines.append("")
        lines.append(f"## kappa = {kappa:.0e}")
        lines.append("")
        lines.append(
            "| config | converged | health | best `|R_raw|_∞` | best `|G|_∞` | dt cuts | alpha drift | uy error | cond1 scaled | terminal reason |"
        )
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
        k_rows = [row for row in rows if float(row["kappa"]) == float(kappa)]
        for row in k_rows:
            cfg = config_map[str(row["config_id"])]
            terminal_reason = str(row.get("terminal_reason", "") or "").replace("|", "/")
            lines.append(
                f"| `{cfg.config_id}` | {int(float(row.get('solve_completed', 0.0) or 0.0))} | "
                f"{row.get('health_label', '')} | "
                f"{float(row.get('best_R_raw_inf', float('nan'))):.3e} | "
                f"{float(row.get('best_G_inf', float('nan'))):.3e} | "
                f"{int(float(row.get('dt_reductions_used', 0.0) or 0.0))} | "
                f"{float(row.get('alpha_area_rel_drift', float('nan'))):.3e} | "
                f"{float(row.get('rmse_over_amp_moving_linear', float('nan'))):.3e} | "
                f"{float(row.get('initial_cond1_solver_scaled_est', float('nan'))):.3e} | "
                f"{terminal_reason} |"
            )
    path.write_text("\n".join(lines) + "\n")


def _build_configs() -> list[SweepConfig]:
    # This matches the user's requested six-method screening interpretation:
    # 1. reduced alpha+B with PDAS
    # 2. reduced alpha+B with IPM
    # 3. full conservative alpha/phi with transformed latent sigmoid
    # 4. full conservative alpha/phi with transformed latent tanh
    # 5. alpha-from-refmap with bounded phi via PDAS
    # 6. full conservative alpha/phi with predictor-corrector startup
    return [
        SweepConfig(
            config_id="alphaB_pdas",
            label="reduced alpha+B / PDAS",
            formulation="reduced alpha+B",
            conservation_mode="weak conservative alpha + B",
            solver_family="bounded semismooth Newton / PDAS",
            extra_args=(
                "--nonlinear-solver",
                "pdas",
                "--no-enable-phi-evolution",
                "--reduced-support-state",
                "alpha_B",
                "--alpha-box-constraints",
                "--no-alpha-mass-constraint",
                "--no-predictor-corrector-startup",
                "--no-startup-bootstrap",
                "--max-it",
                "32",
            ),
        ),
        SweepConfig(
            config_id="alphaB_ipm",
            label="reduced alpha+B / IPM",
            formulation="reduced alpha+B",
            conservation_mode="weak conservative alpha + B",
            solver_family="bounded interior-point Newton",
            extra_args=(
                "--nonlinear-solver",
                "ipm",
                "--no-enable-phi-evolution",
                "--reduced-support-state",
                "alpha_B",
                "--alpha-box-constraints",
                "--no-alpha-mass-constraint",
                "--no-predictor-corrector-startup",
                "--no-startup-bootstrap",
                "--max-it",
                "32",
                "--vi-ipm-max-barrier-steps",
                "4",
            ),
        ),
        SweepConfig(
            config_id="latent_sigmoid",
            label="full alpha/phi transformed latent sigmoid",
            formulation="full alpha+phi",
            conservation_mode="weak conservative alpha transport",
            solver_family="unconstrained Newton in transformed latent coordinates",
            extra_args=(
                "--nonlinear-solver",
                "newton",
                "--enable-phi-evolution",
                "--no-alpha-box-constraints",
                "--no-phi-box-constraints",
                "--latent-bounded-transport",
                "--latent-bounded-formulation",
                "transformed",
                "--latent-bounded-fields",
                "alpha,phi",
                "--latent-bounded-map",
                "sigmoid",
                "--no-predictor-corrector-startup",
                "--no-startup-bootstrap",
                "--newton-reduced-scaling-mode",
                "ruiz",
                "--newton-globalization",
                "line_search_then_trust",
                "--max-it",
                "24",
            ),
        ),
        SweepConfig(
            config_id="latent_tanh",
            label="full alpha/phi transformed latent tanh",
            formulation="full alpha+phi",
            conservation_mode="weak conservative alpha transport",
            solver_family="unconstrained Newton in transformed latent coordinates",
            extra_args=(
                "--nonlinear-solver",
                "newton",
                "--enable-phi-evolution",
                "--no-alpha-box-constraints",
                "--no-phi-box-constraints",
                "--latent-bounded-transport",
                "--latent-bounded-formulation",
                "transformed",
                "--latent-bounded-fields",
                "alpha,phi",
                "--latent-bounded-map",
                "tanh",
                "--no-predictor-corrector-startup",
                "--no-startup-bootstrap",
                "--newton-reduced-scaling-mode",
                "ruiz",
                "--newton-globalization",
                "line_search_then_trust",
                "--max-it",
                "24",
            ),
        ),
        SweepConfig(
            config_id="refmap_phi_pdas",
            label="alpha-from-refmap with bounded phi / PDAS",
            formulation="alpha-from-refmap + direct phi",
            conservation_mode="alpha rebuilt from reference map",
            solver_family="bounded semismooth Newton / PDAS",
            extra_args=(
                "--nonlinear-solver",
                "pdas",
                "--alpha-from-refmap",
                "--enable-phi-evolution",
                "--no-alpha-box-constraints",
                "--phi-box-constraints",
                "--no-latent-bounded-transport",
                "--no-logistic-bounded-transform",
                "--no-predictor-corrector-startup",
                "--no-startup-bootstrap",
                "--max-it",
                "32",
            ),
        ),
        SweepConfig(
            config_id="direct_phi_pc",
            label="full alpha/phi + predictor-corrector startup",
            formulation="full alpha+phi",
            conservation_mode="weak conservative alpha transport",
            solver_family="unconstrained Newton with predictor-corrector startup",
            extra_args=(
                "--nonlinear-solver",
                "newton",
                "--enable-phi-evolution",
                "--no-alpha-box-constraints",
                "--no-phi-box-constraints",
                "--predictor-corrector-startup",
                "--startup-bootstrap",
                "--startup-bootstrap-max-it",
                "8",
                "--startup-monolithic-max-it",
                "32",
                "--pc-exact-probe-max-it",
                "2",
                "--pc-p1-max-it",
                "12",
                "--pc-p2-max-it",
                "12",
                "--newton-reduced-scaling-mode",
                "ruiz",
                "--newton-globalization",
                "line_search_then_trust",
                "--max-it",
                "24",
            ),
        ),
    ]


def _build_command(
    *,
    driver: Path,
    case_outdir: Path,
    kappa: float,
    dt0: float,
    dt_min: float,
    t_final: float,
    config: SweepConfig,
) -> list[str]:
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "fenicsx",
        "python",
        str(driver),
        "--outdir",
        str(case_outdir),
        "--backend",
        "cpp",
        "--linear-backend",
        "scipy",
        "--nx",
        "12",
        "--ny",
        "18",
        "--dt",
        f"{dt0:.16g}",
        "--dt-max",
        f"{dt0:.16g}",
        "--dt-min",
        f"{dt_min:.16g}",
        "--dt-increase-factor",
        "1.0",
        "--dt-reduction-factor",
        "0.5",
        "--t-final",
        f"{t_final:.16g}",
        "--theta",
        "1.0",
        "--kappa-list",
        f"{kappa:.16g}",
        "--phi-b",
        "0.18",
        "--support-physics",
        "internal_conversion",
        "--alpha-advect-with",
        "biofilm_volume",
        "--alpha-advection-form",
        "conservative_weak",
        "--include-skeleton-acceleration",
        "--rho-s0-tilde",
        "1.1",
        "--skeleton-inertia-convection",
        "lagged",
        "--fluid-convection",
        "full",
        "--kappa-inv-model",
        "refmap",
        "--drag-formulation",
        "mixed_lm",
        "--gamma-u",
        "1.0",
        "--u-extension",
        "h1",
        "--gamma-u-pin",
        "1e-6",
        "--vS-cip",
        "1.0",
        "--gamma-vS",
        "5.0",
        "--vS-ext-mode",
        "h1",
        "--gamma-vS-pin",
        "1e-6",
        "--gamma-phi",
        "5.0",
        "--t-ramp",
        "2e-2",
        "--predictor",
        "prev",
        "--predictor-damping",
        "1.0",
        "--alpha-supg",
        "0.5",
        "--eps-alpha-over-h",
        "0.6",
        "--vi-c",
        "1e4",
        "--vtk-every",
        "0",
        "--report-initial-condition-number",
        "--no-condition-balanced-solid-cut-fix",
    ]
    cmd.extend(config.extra_args)
    return cmd


def _run_case(command: list[str], *, log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as logf:
        logf.write(f"[command] {' '.join(command)}\n")
        logf.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
        return process.wait()


def _collect_case_row(
    *,
    config: SweepConfig,
    kappa: float,
    case_outdir: Path,
    log_path: Path,
    returncode: int,
) -> dict[str, object]:
    summary_row = _load_case_summary(case_outdir / "benchmark7_summary.csv")
    log_text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    log_metrics = _extract_log_metrics(log_text)
    row: dict[str, object] = {
        "config_id": config.config_id,
        "config_label": config.label,
        "formulation": config.formulation,
        "alpha_conservation_mode": config.conservation_mode,
        "solver_family": config.solver_family,
        "kappa": float(kappa),
        "returncode": float(returncode),
        "case_outdir": str(case_outdir),
        "log_path": str(log_path),
    }
    for key, value in summary_row.items():
        row[key] = value
    row.update(log_metrics)
    row["solve_completed"] = float(summary_row.get("solve_completed", 0.0) or 0.0) if summary_row else 0.0
    row["final_raw_residual_inf"] = _float_or_nan(summary_row.get("final_raw_residual_inf")) if summary_row else float("nan")
    row["rmse_over_amp_moving_linear"] = _float_or_nan(summary_row.get("rmse_over_amp_moving_linear")) if summary_row else float("nan")
    row["alpha_area_rel_drift"] = _float_or_nan(summary_row.get("alpha_area_rel_drift")) if summary_row else float("nan")
    row["initial_cond1_solver_scaled_est"] = _float_or_nan(summary_row.get("initial_cond1_solver_scaled_est")) if summary_row else float("nan")
    row["steps_recorded"] = _float_or_nan(summary_row.get("steps_recorded")) if summary_row else float("nan")
    return row


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]
    default_outdir = repo_root / "out" / f"benchmark7_state_sweep_{time.strftime('%Y%m%d')}"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=default_outdir)
    ap.add_argument("--driver", type=Path, default=repo_root / "examples" / "biofilms" / "benchmarks" / "seboldt" / "paper1_benchmark7_seboldt.py")
    ap.add_argument("--kappa-list", type=str, default="1e-3,1e-4,1e-5")
    ap.add_argument("--dt", type=float, default=2.5e-3)
    ap.add_argument("--t-final", type=float, default=None, help="Defaults to one step, i.e. t_final = dt.")
    ap.add_argument("--max-reductions", type=int, default=2)
    ap.add_argument("--configs", type=str, default="", help="Optional comma-separated subset of config ids.")
    ap.add_argument("--cache-root", type=str, default=os.environ.get("PYCUTFEM_CACHE_DIR", ""))
    ap.add_argument("--dry-run", action="store_true", default=False)
    args = ap.parse_args()

    dt0 = float(args.dt)
    t_final = float(args.t_final) if args.t_final is not None else dt0
    dt_min = dt0 * (0.5 ** int(args.max_reductions))
    kappas = _parse_csv_floats(args.kappa_list)
    configs = _build_configs()
    if args.configs.strip():
        wanted = {token.strip() for token in args.configs.split(",") if token.strip()}
        configs = [cfg for cfg in configs if cfg.config_id in wanted]
        if not configs:
            raise SystemExit("No matching configs after --configs filtering.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    aggregate_csv = args.outdir / "aggregate_summary.csv"
    aggregate_md = args.outdir / "aggregate_summary.md"
    meta_json = args.outdir / "run_metadata.json"
    meta_json.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "repo_root": str(repo_root),
                "driver": str(args.driver),
                "outdir": str(args.outdir),
                "cache_root": str(args.cache_root),
                "dt": dt0,
                "dt_min": dt_min,
                "t_final": t_final,
                "kappas": kappas,
                "configs": [cfg.__dict__ for cfg in configs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    env = dict(os.environ)
    rows: list[dict[str, object]] = []
    for cfg in configs:
        for kappa in kappas:
            case_outdir = args.outdir / cfg.config_id / f"kappa_{_kappa_slug(kappa)}"
            log_path = case_outdir / "run.log"
            cmd = _build_command(
                driver=args.driver,
                case_outdir=case_outdir,
                kappa=kappa,
                dt0=dt0,
                dt_min=dt_min,
                t_final=t_final,
                config=cfg,
            )
            print(
                f"[sweep] config={cfg.config_id} kappa={kappa:.0e} "
                f"dt0={dt0:.3e} dt_min={dt_min:.3e} t_final={t_final:.3e} -> {case_outdir}",
                flush=True,
            )
            if args.dry_run:
                print("[dry-run] " + " ".join(cmd), flush=True)
                returncode = 0
            else:
                returncode = _run_case(cmd, log_path=log_path, env=env)
            row = _collect_case_row(
                config=cfg,
                kappa=kappa,
                case_outdir=case_outdir,
                log_path=log_path,
                returncode=returncode,
            )
            rows.append(row)
            _write_csv(aggregate_csv, rows)
            _write_markdown(
                aggregate_md,
                rows=rows,
                configs=configs,
                dt0=dt0,
                dt_min=dt_min,
                t_final=t_final,
                max_reductions=int(args.max_reductions),
            )
            print(
                "[sweep-summary] "
                f"config={cfg.config_id} kappa={kappa:.0e} "
                f"solve_completed={int(float(row.get('solve_completed', 0.0) or 0.0))} "
                f"health={row.get('health_label', 'unknown')} "
                f"best_R_raw={float(row.get('best_R_raw_inf', float('nan'))):.3e} "
                f"best_G={float(row.get('best_G_inf', float('nan'))):.3e} "
                f"dt_cuts={int(float(row.get('dt_reductions_used', 0.0) or 0.0))} "
                f"alpha_drift={float(row.get('alpha_area_rel_drift', float('nan'))):.3e}",
                flush=True,
            )

    print(f"[done] aggregate_csv={aggregate_csv}", flush=True)
    print(f"[done] aggregate_md={aggregate_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
