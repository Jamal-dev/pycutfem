from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from pycutfem.benchmarks.featflow import (
    compare_timeseries,
    ensure_featflow_zips,
    load_draglift,
    load_pressure_points,
)


def _split_list(raw: str) -> list[str]:
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def _float_list(raw: str) -> list[float]:
    out: list[float] = []
    for s in _split_list(raw):
        out.append(float(s))
    return out


def _int_list(raw: str) -> list[int]:
    out: list[int] = []
    for s in _split_list(raw):
        out.append(int(s))
    return out


@dataclass(frozen=True)
class RunConfig:
    case: str
    tend: float
    backend: str
    jit_backend: str
    gamma_gp: float
    gamma_n: float
    agfem: bool
    agfem_theta: float
    p_stab_jump: float
    p_stab_avg: float
    gp_theta_exp: float
    gp_theta_min: float

    def run_id(self) -> str:
        bits = [
            self.case,
            f"t{self.tend:g}",
            f"b{self.backend}",
            f"jit{self.jit_backend}",
            f"gn{self.gamma_n:g}",
            f"ggp{self.gamma_gp:g}",
            f"ag{int(self.agfem)}",
            f"agt{self.agfem_theta:g}",
            f"pj{self.p_stab_jump:g}",
            f"pa{self.p_stab_avg:g}",
            f"gpexp{self.gp_theta_exp:g}",
            f"gpmin{self.gp_theta_min:g}",
        ]
        return "_".join(bits)


def _read_functionals_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    def col(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype=float)
    return {
        "time": col("time"),
        "cd": col("Cd"),
        "cl": col("Cl"),
        "dp": col("dp"),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    script = root / "examples" / "turek_benchmark.py"
    if not script.is_file():
        raise SystemExit(f"Cannot find benchmark script at {script}.")

    p = argparse.ArgumentParser(description="Parameter sweep for examples/turek_benchmark.py (FeatFlow validation).")
    p.add_argument("--cases", type=str, default="2D-3", help="comma-separated cases, e.g. 2D-1,2D-2,2D-3")
    p.add_argument("--tend", type=float, default=0.05, help="final time for each run (keep small for quick sweeps)")
    p.add_argument("--backend", choices=("jit", "python"), default="jit")
    p.add_argument("--jit-backend", choices=("cpp", "numba"), default="cpp", help="PYCUTFEM_JIT_BACKEND value when backend=jit")
    p.add_argument("--gamma-n", type=float, default=40.0)
    p.add_argument("--gamma-gp", type=str, default="1e-3", help="comma-separated values")
    p.add_argument("--p-stab-jump", type=str, default="0.0,0.1", help="env PYCUTFEM_PRESSURE_STAB_JUMP values")
    p.add_argument("--p-stab-avg", type=str, default="0.0,0.1", help="env PYCUTFEM_PRESSURE_STAB_AVG values")
    p.add_argument("--gp-theta-exp", type=str, default="0.0,1.0", help="env PYCUTFEM_GHOST_SCALE_THETA_EXP values")
    p.add_argument("--gp-theta-min", type=float, default=1e-3, help="env PYCUTFEM_GHOST_SCALE_THETA_MIN value")
    p.add_argument("--agfem", type=str, default="0,1", help="enable AgFEM (0/1 list)")
    p.add_argument("--agfem-theta", type=str, default="0.05", help="Hansbo theta threshold list when AgFEM enabled")
    p.add_argument("--featflow-level", type=int, default=6, choices=range(1, 7))
    p.add_argument("--featflow-cache", type=str, default=None)
    p.add_argument("--output-root", type=str, default="turek_sweep_runs")
    p.add_argument("--output-csv", type=str, default="turek_sweep_results.csv")
    p.add_argument("--max-runs", type=int, default=0, help="0 = no limit (otherwise stop after N runs)")
    args = p.parse_args()

    cases = _split_list(args.cases)
    gamma_gp_list = _float_list(args.gamma_gp)
    p_jump_list = _float_list(args.p_stab_jump)
    p_avg_list = _float_list(args.p_stab_avg)
    gp_exp_list = _float_list(args.gp_theta_exp)
    agfem_list = [bool(int(x)) for x in _int_list(args.agfem)]
    agfem_theta_list = _float_list(args.agfem_theta)

    cache_dir = Path(args.featflow_cache).expanduser() if args.featflow_cache else None
    drag_zip, pres_zip = ensure_featflow_zips(cache_dir=cache_dir)
    ref_dl = load_draglift(drag_zip, level=int(args.featflow_level))
    ref_pr = load_pressure_points(pres_zip, level=int(args.featflow_level))

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "status",
        "case",
        "backend",
        "jit_backend",
        "tend",
        "gamma_n",
        "gamma_gp",
        "agfem",
        "agfem_theta",
        "p_stab_jump",
        "p_stab_avg",
        "gp_theta_exp",
        "gp_theta_min",
        "runtime_s",
        "steps",
        "cd_end",
        "cl_end",
        "dp_end",
        "cd_rms",
        "cd_max_abs",
        "cl_rms",
        "cl_max_abs",
        "dp_rms",
        "dp_max_abs",
    ]
    with out_csv.open("w", newline="") as f_csv:
        w = csv.DictWriter(f_csv, fieldnames=fieldnames)
        w.writeheader()

        run_count = 0
        for case in cases:
            for gamma_gp, p_jump, p_avg, gp_exp, agfem_on, ag_theta in product(
                gamma_gp_list,
                p_jump_list,
                p_avg_list,
                gp_exp_list,
                agfem_list,
                agfem_theta_list,
            ):
                if args.max_runs and run_count >= int(args.max_runs):
                    return 0

                cfg = RunConfig(
                    case=str(case),
                    tend=float(args.tend),
                    backend=str(args.backend),
                    jit_backend=str(args.jit_backend),
                    gamma_gp=float(gamma_gp),
                    gamma_n=float(args.gamma_n),
                    agfem=bool(agfem_on),
                    agfem_theta=float(ag_theta),
                    p_stab_jump=float(p_jump),
                    p_stab_avg=float(p_avg),
                    gp_theta_exp=float(gp_exp),
                    gp_theta_min=float(args.gp_theta_min),
                )
                run_id = cfg.run_id()
                out_dir = out_root / run_id
                out_dir.mkdir(parents=True, exist_ok=True)

                env = os.environ.copy()
                if cfg.backend == "jit":
                    env["PYCUTFEM_JIT_BACKEND"] = cfg.jit_backend
                env["PYCUTFEM_PRESSURE_STAB_JUMP"] = str(cfg.p_stab_jump)
                env["PYCUTFEM_PRESSURE_STAB_AVG"] = str(cfg.p_stab_avg)
                env["PYCUTFEM_GHOST_SCALE_THETA_EXP"] = str(cfg.gp_theta_exp)
                env["PYCUTFEM_GHOST_SCALE_THETA_MIN"] = str(cfg.gp_theta_min)
                if cfg.agfem:
                    env["PYCUTFEM_AGFEM"] = "1"
                    env["PYCUTFEM_AGFEM_THETA_MIN"] = str(cfg.agfem_theta)
                else:
                    env.pop("PYCUTFEM_AGFEM", None)
                    env.pop("PYCUTFEM_AGFEM_THETA_MIN", None)

                cmd = [
                    sys.executable,
                    str(script),
                    "--case",
                    cfg.case,
                    "--backend",
                    cfg.backend,
                    "--tend",
                    str(cfg.tend),
                    "--gamma-n",
                    str(cfg.gamma_n),
                    "--gamma-gp",
                    str(cfg.gamma_gp),
                    "--output-dir",
                    str(out_dir),
                    "--vtk-interval",
                    "0",
                    "--print-interval",
                    "999999",
                ]
                if cfg.agfem:
                    cmd.extend(["--agfem", "--agfem-theta", str(cfg.agfem_theta)])

                t0 = time.perf_counter()
                proc = subprocess.run(
                    cmd,
                    cwd=str(root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                runtime = time.perf_counter() - t0

                row: dict[str, object] = {
                    "run_id": run_id,
                    "case": cfg.case,
                    "backend": cfg.backend,
                    "jit_backend": cfg.jit_backend if cfg.backend == "jit" else "",
                    "tend": cfg.tend,
                    "gamma_n": cfg.gamma_n,
                    "gamma_gp": cfg.gamma_gp,
                    "agfem": int(cfg.agfem),
                    "agfem_theta": cfg.agfem_theta,
                    "p_stab_jump": cfg.p_stab_jump,
                    "p_stab_avg": cfg.p_stab_avg,
                    "gp_theta_exp": cfg.gp_theta_exp,
                    "gp_theta_min": cfg.gp_theta_min,
                    "runtime_s": runtime,
                }

                if proc.returncode != 0:
                    row.update(
                        {
                            "status": f"fail({proc.returncode})",
                            "steps": 0,
                            "cd_end": np.nan,
                            "cl_end": np.nan,
                            "dp_end": np.nan,
                            "cd_rms": np.nan,
                            "cd_max_abs": np.nan,
                            "cl_rms": np.nan,
                            "cl_max_abs": np.nan,
                            "dp_rms": np.nan,
                            "dp_max_abs": np.nan,
                        }
                    )
                    w.writerow(row)
                    f_csv.flush()
                    tail = "\n".join((proc.stdout or "").splitlines()[-40:])
                    print(f"[sweep] {run_id} FAILED (see last lines):\n{tail}", file=sys.stderr)
                    run_count += 1
                    continue

                try:
                    fcsv = out_dir / "functionals.csv"
                    sim = _read_functionals_csv(fcsv)
                    t = sim["time"]
                    cd = sim["cd"]
                    cl = sim["cl"]
                    dp = sim["dp"]

                    cd_err = compare_timeseries(ref_time=ref_dl.time, ref_val=ref_dl.cd, sim_time=t, sim_val=cd)
                    cl_err = compare_timeseries(ref_time=ref_dl.time, ref_val=ref_dl.cl, sim_time=t, sim_val=cl)
                    dp_err = compare_timeseries(ref_time=ref_pr.time, ref_val=ref_pr.dp, sim_time=t, sim_val=dp)

                    row.update(
                        {
                            "status": "ok",
                            "steps": int(t.size),
                            "cd_end": float(cd[-1]),
                            "cl_end": float(cl[-1]),
                            "dp_end": float(dp[-1]),
                            "cd_rms": cd_err["rms"],
                            "cd_max_abs": cd_err["max_abs"],
                            "cl_rms": cl_err["rms"],
                            "cl_max_abs": cl_err["max_abs"],
                            "dp_rms": dp_err["rms"],
                            "dp_max_abs": dp_err["max_abs"],
                        }
                    )
                except Exception as exc:
                    row.update(
                        {
                            "status": f"parse_error({type(exc).__name__})",
                            "steps": 0,
                            "cd_end": np.nan,
                            "cl_end": np.nan,
                            "dp_end": np.nan,
                            "cd_rms": np.nan,
                            "cd_max_abs": np.nan,
                            "cl_rms": np.nan,
                            "cl_max_abs": np.nan,
                            "dp_rms": np.nan,
                            "dp_max_abs": np.nan,
                        }
                    )
                w.writerow(row)
                f_csv.flush()
                run_count += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

