#!/usr/bin/env python3
"""
Successive-halving search for robust Turek-cylinder stabilization parameters.

Run in `xfemcustom` env (recommended):

  conda run --no-capture-output -n xfemcustom python examples/turek_cylinder/optimize_params.py \\
    --benchmark 2d-2 --level 4 --dt 0.1 --theta 0.5 --with-deformation \\
    --fe-order 2 --p-order 1 --ghost-measure patch --inflow constant --init zero \\
    --n-target 250 --budgets 50,100,250 --eta 3 --lambda-fail 100000 \\
    --beta0 20,30,40,50,60 --gamma-gp 0.003,0.01,0.03 --gamma-gp-hess 0,1e-6,1e-5,1e-4,1e-3

The script writes:
- a run directory under `examples/turek_cylinder/opt_runs/<timestamp>/`
- `summary.json` with per-candidate scores and metadata
- `examples/turek_cylinder/tuned_params.json` (best candidate for this case)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DFG_REFS: dict[str, dict[str, float]] = {
    # DFG 2D-1 (steady, Re=20)
    "2d-1": {
        "Cd": 5.57953523384,
        "Cl": 0.010618948146,
        "dp": 0.11752016697,
    },
    # DFG 2D-2 (unsteady constant inflow, Re=100) – exemplary results (level 6, dt=1/200)
    "2d-2": {
        "Cd_mean": 3.1884,
        "Cd_amp": 6.310e-02,
        "Cl_mean": -0.0173,
        "Cl_amp": 2.0080,
    },
    # DFG 2D-3 (unsteady ramp inflow, Re=100) – exemplary results (level 6, dt=1/1600)
    "2d-3": {
        "Cd_max": 2.9437637214,
        "Cl_max": 0.47748781595,
        "dp_t8": 0.11154138872,
        "t_Cd_max": 3.9365625,
        "t_Cl_max": 5.6928125,
        "t_dp": 8.0,
    },
}


def _parse_csv_floats(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: dict[str, float] = {}
            for k, v in raw.items():
                if v is None:
                    continue
                try:
                    row[k] = float(v)
                except Exception:
                    # Treat non-numeric cells (rare) as NaN.
                    row[k] = float("nan")
            rows.append(row)
    return rows


def _nanmin(vals: list[float]) -> float:
    return min(v for v in vals if math.isfinite(v))


def _nanmax(vals: list[float]) -> float:
    return max(v for v in vals if math.isfinite(v))


def _nanmean(vals: list[float]) -> float:
    finite = [v for v in vals if math.isfinite(v)]
    return float("nan") if not finite else sum(finite) / len(finite)


def _closest_time_row(rows: list[dict[str, float]], t: float) -> dict[str, float] | None:
    best = None
    best_dt = float("inf")
    for r in rows:
        tt = r.get("time", float("nan"))
        if not math.isfinite(tt):
            continue
        dt = abs(tt - t)
        if dt < best_dt:
            best_dt = dt
            best = r
    return best


def _pick_series(rows: list[dict[str, float]], keys: list[str]) -> tuple[str, list[float]] | tuple[None, list[float]]:
    for k in keys:
        if rows and k in rows[0]:
            return k, [r.get(k, float("nan")) for r in rows]
    return None, []


def _score_against_reference(benchmark: str, rows: list[dict[str, float]]) -> float:
    if not rows:
        return 0.0
    ref = DFG_REFS.get(benchmark, {})
    if not ref:
        return 0.0

    cd_key, cd = _pick_series(rows, ["Cd_surf", "Cd_bm"])
    cl_key, cl = _pick_series(rows, ["Cl_surf", "Cl_bm"])
    dp_key, dp = _pick_series(rows, ["dp"])

    score = 0.0
    if benchmark == "2d-1":
        if cd_key and math.isfinite(ref.get("Cd", float("nan"))):
            cd_last = float(cd[-1])
            score += ((cd_last - ref["Cd"]) / ref["Cd"]) ** 2
        if cl_key and math.isfinite(ref.get("Cl", float("nan"))):
            cl_last = float(cl[-1])
            score += (cl_last - ref["Cl"]) ** 2
        if dp_key and math.isfinite(ref.get("dp", float("nan"))):
            dp_last = float(dp[-1])
            score += ((dp_last - ref["dp"]) / ref["dp"]) ** 2
        return score

    if benchmark == "2d-2":
        if cd_key:
            cd_mean = _nanmean(cd)
            cd_amp = _nanmax(cd) - _nanmin(cd)
            score += ((cd_mean - ref["Cd_mean"]) / ref["Cd_mean"]) ** 2
            score += ((cd_amp - ref["Cd_amp"]) / ref["Cd_amp"]) ** 2
        if cl_key:
            cl_mean = _nanmean(cl)
            cl_amp = _nanmax(cl) - _nanmin(cl)
            score += (cl_mean - ref["Cl_mean"]) ** 2
            score += ((cl_amp - ref["Cl_amp"]) / ref["Cl_amp"]) ** 2
        return score

    if benchmark == "2d-3":
        if cd_key:
            score += ((_nanmax(cd) - ref["Cd_max"]) / ref["Cd_max"]) ** 2
        if cl_key:
            score += ((_nanmax(cl) - ref["Cl_max"]) / ref["Cl_max"]) ** 2
        if dp_key:
            row_t8 = _closest_time_row(rows, float(ref.get("t_dp", 8.0)))
            if row_t8 is not None:
                dp_t8 = float(row_t8.get("dp", float("nan")))
                if math.isfinite(dp_t8):
                    score += ((dp_t8 - ref["dp_t8"]) / ref["dp_t8"]) ** 2
        return score

    return score


def _as_float_list(s: str, *, cast=float) -> list[float]:
    if not s:
        return []
    out: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(cast(part)))
    return out


def _fmt_float(x: float) -> str:
    # Stable, filename-safe formatting.
    if x == 0:
        return "0"
    ax = abs(x)
    if ax < 1e-2 or ax >= 1e3:
        return f"{x:.0e}".replace("+", "")
    return f"{x:g}"


@dataclass(frozen=True)
class Candidate:
    beta0: float
    gamma_gp: float
    gamma_gp_p: float | None
    gamma_gp_hess: float

    def as_args(self) -> list[str]:
        args = [
            "--beta0",
            str(self.beta0),
            "--gamma-gp",
            str(self.gamma_gp),
            "--gamma-gp-hess",
            str(self.gamma_gp_hess),
        ]
        if self.gamma_gp_p is not None:
            args += ["--gamma-gp-p", str(self.gamma_gp_p)]
        return args

    def as_key(self) -> str:
        gp_p = self.gamma_gp_p if self.gamma_gp_p is not None else float("nan")
        return (
            f"beta0={_fmt_float(self.beta0)}__"
            f"gg={_fmt_float(self.gamma_gp)}__"
            f"ggp={_fmt_float(gp_p)}__"
            f"ggh={_fmt_float(self.gamma_gp_hess)}"
        )


@dataclass
class RunResult:
    candidate: Candidate
    budget_steps: int
    exit_code: int
    n_done: int
    score: float
    score_fail: float
    score_ref: float
    functionals_csv: Path
    run_dir: Path


def _run_once(
    *,
    candidate: Candidate,
    benchmark: str,
    level: int,
    dt: float,
    theta: float,
    with_deformation: bool,
    fe_order: int,
    p_order: int,
    ghost_measure: str,
    inflow: str,
    init: str,
    budget_steps: int,
    lambda_fail: float,
    root: Path,
    extra_args: list[str],
) -> RunResult:
    run_dir = root / f"{candidate.as_key()}__steps={budget_steps}"
    run_dir.mkdir(parents=True, exist_ok=True)

    functionals_csv = run_dir / "functionals.csv"
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"

    cmd = [
        sys.executable,
        str((Path(__file__).resolve().parent / "turek_benchmark.py").resolve()),
        "--benchmark",
        benchmark,
        "--backend",
        "cpp",
        "--level",
        str(level),
        "--dt",
        str(dt),
        "--theta",
        str(theta),
        "--max-steps",
        str(budget_steps),
        "--fe-order",
        str(fe_order),
        "--p-order",
        str(p_order),
        "--ghost-measure",
        str(ghost_measure),
        "--inflow",
        str(inflow),
        "--init",
        str(init),
        "--vtk-every",
        "0",
        "--output-dir",
        str(run_dir),
    ]
    if with_deformation:
        cmd.append("--with-deformation")
    cmd += candidate.as_args()
    cmd += list(extra_args)

    env = dict(os.environ)
    env.setdefault("PYCUTFEM_JIT_BACKEND", "cpp")

    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt_run = time.time() - t0
    stdout_path.write_text(proc.stdout + f"\n\n[opt] elapsed_s={dt_run:.3f}\n")
    stderr_path.write_text(proc.stderr)

    rows = _parse_csv_floats(functionals_csv)
    n_done = len(rows)

    score_fail = float(lambda_fail) * max(0, int(budget_steps) - int(n_done))
    score_ref = _score_against_reference(benchmark, rows)
    score = float(score_fail + score_ref)

    return RunResult(
        candidate=candidate,
        budget_steps=int(budget_steps),
        exit_code=int(proc.returncode),
        n_done=int(n_done),
        score=float(score),
        score_fail=float(score_fail),
        score_ref=float(score_ref),
        functionals_csv=functionals_csv,
        run_dir=run_dir,
    )


def _update_tuned_params(best: RunResult, *, tuned_path: Path, meta: dict[str, Any]) -> None:
    entry = {
        "benchmark": meta["benchmark"],
        "level": meta["level"],
        "dt": meta["dt"],
        "theta": meta["theta"],
        "ghost_measure": meta["ghost_measure"],
        "with_deformation": meta["with_deformation"],
        "fe_order": meta["fe_order"],
        "p_order": meta["p_order"],
        "beta0": best.candidate.beta0,
        "gamma_gp": best.candidate.gamma_gp,
        "gamma_gp_hess": best.candidate.gamma_gp_hess,
        "score": best.score,
        "n_done": best.n_done,
        "ts": int(time.time()),
    }
    if best.candidate.gamma_gp_p is not None:
        entry["gamma_gp_p"] = float(best.candidate.gamma_gp_p)

    doc: dict[str, Any] = {"version": 1, "entries": []}
    if tuned_path.is_file():
        try:
            loaded = json.loads(tuned_path.read_text())
            if isinstance(loaded, dict):
                doc.update(loaded)
        except Exception:
            pass
    entries = doc.get("entries")
    if not isinstance(entries, list):
        entries = []

    def _same_case(e: Any) -> bool:
        if not isinstance(e, dict):
            return False
        return (
            str(e.get("benchmark")) == str(entry["benchmark"])
            and e.get("level") == entry["level"]
            and str(e.get("ghost_measure")) == str(entry["ghost_measure"])
            and bool(e.get("with_deformation")) == bool(entry["with_deformation"])
            and int(e.get("fe_order", entry["fe_order"])) == int(entry["fe_order"])
            and int(e.get("p_order", entry["p_order"])) == int(entry["p_order"])
            and abs(float(e.get("dt", entry["dt"])) - float(entry["dt"])) <= 1e-12
        )

    same = [e for e in entries if _same_case(e)]
    if same:
        new_score = float(entry.get("score", float("inf")))
        if not math.isfinite(new_score):
            return
        old_scores = []
        for e in same:
            try:
                s = float(e.get("score", float("inf")))
            except Exception:
                s = float("inf")
            if math.isfinite(s):
                old_scores.append(s)
        old_best = min(old_scores) if old_scores else float("inf")
        # Only update this case if we found a strictly better (lower) score.
        if math.isfinite(old_best) and new_score >= old_best:
            return

    entries = [e for e in entries if not _same_case(e)]
    entries.append(entry)
    doc["entries"] = entries
    tuned_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=("2d-1", "2d-2", "2d-3"), default="2d-2")
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--with-deformation", action="store_true")
    ap.add_argument("--fe-order", type=int, default=2)
    ap.add_argument("--p-order", type=int, default=1)
    ap.add_argument("--ghost-measure", choices=("edge", "patch"), default="patch")
    ap.add_argument("--inflow", choices=("constant", "dfg"), default="constant")
    ap.add_argument("--init", choices=("zero", "stokes"), default="zero")
    ap.add_argument("--n-target", type=int, default=250)
    ap.add_argument("--budgets", type=str, default="50,100,250", help="comma-separated max-steps per round")
    ap.add_argument("--eta", type=int, default=3, help="successive halving reduction factor")
    ap.add_argument("--lambda-fail", type=float, default=1e5, help="penalty weight for early failure")
    ap.add_argument("--beta0", type=str, default="20,30,40,50,60", help="comma-separated beta0 candidates")
    ap.add_argument("--gamma-gp", type=str, default="0.003,0.01,0.03", help="comma-separated gamma-gp candidates")
    ap.add_argument(
        "--gamma-gp-p",
        type=str,
        default="",
        help="comma-separated gamma-gp-p candidates (empty => use gamma-gp for pressure)",
    )
    ap.add_argument(
        "--gamma-gp-hess",
        type=str,
        default="0,1e-6,1e-5,1e-4,1e-3",
        help="comma-separated gamma-gp-hess candidates",
    )
    ap.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="extra args forwarded to turek_benchmark.py (repeatable), e.g. --extra-arg --newton-tol=1e-8",
    )
    args = ap.parse_args()

    budgets = [int(x) for x in _as_float_list(args.budgets, cast=int)]
    budgets = [b for b in budgets if b > 0]
    if not budgets:
        raise SystemExit("--budgets must contain at least one positive integer")
    budgets[-1] = int(args.n_target)

    beta0_vals = _as_float_list(args.beta0)
    gamma_gp_vals = _as_float_list(args.gamma_gp)
    gamma_gp_p_vals = _as_float_list(args.gamma_gp_p) if args.gamma_gp_p.strip() else []
    gamma_gp_hess_vals = _as_float_list(args.gamma_gp_hess)

    candidates: list[Candidate] = []
    if gamma_gp_p_vals:
        prod = itertools.product(beta0_vals, gamma_gp_vals, gamma_gp_p_vals, gamma_gp_hess_vals)
        for beta0, gg, ggp, ggh in prod:
            candidates.append(Candidate(beta0=float(beta0), gamma_gp=float(gg), gamma_gp_p=float(ggp), gamma_gp_hess=float(ggh)))
    else:
        prod = itertools.product(beta0_vals, gamma_gp_vals, gamma_gp_hess_vals)
        for beta0, gg, ggh in prod:
            candidates.append(Candidate(beta0=float(beta0), gamma_gp=float(gg), gamma_gp_p=None, gamma_gp_hess=float(ggh)))

    ts = time.strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parent / "opt_runs" / ts
    root.mkdir(parents=True, exist_ok=True)

    meta = {
        "benchmark": str(args.benchmark),
        "level": int(args.level),
        "dt": float(args.dt),
        "theta": float(args.theta),
        "ghost_measure": str(args.ghost_measure),
        "with_deformation": bool(args.with_deformation),
        "fe_order": int(args.fe_order),
        "p_order": int(args.p_order),
        "eta": int(args.eta),
        "lambda_fail": float(args.lambda_fail),
        "budgets": budgets,
        "n_candidates_initial": len(candidates),
        "extra_args": list(args.extra_arg),
    }

    all_results: list[dict[str, Any]] = []
    alive = list(candidates)
    for r, budget in enumerate(budgets):
        print(f"[opt] round {r+1}/{len(budgets)}: budget={budget} candidates={len(alive)}")
        round_results: list[RunResult] = []
        for cand in alive:
            res = _run_once(
                candidate=cand,
                benchmark=str(args.benchmark),
                level=int(args.level),
                dt=float(args.dt),
                theta=float(args.theta),
                with_deformation=bool(args.with_deformation),
                fe_order=int(args.fe_order),
                p_order=int(args.p_order),
                ghost_measure=str(args.ghost_measure),
                inflow=str(args.inflow),
                init=str(args.init),
                budget_steps=int(budget),
                lambda_fail=float(args.lambda_fail),
                root=root,
                extra_args=list(args.extra_arg),
            )
            round_results.append(res)
            all_results.append(
                {
                    "candidate": cand.as_key(),
                    "beta0": cand.beta0,
                    "gamma_gp": cand.gamma_gp,
                    "gamma_gp_p": cand.gamma_gp_p,
                    "gamma_gp_hess": cand.gamma_gp_hess,
                    "budget_steps": res.budget_steps,
                    "exit_code": res.exit_code,
                    "n_done": res.n_done,
                    "score": res.score,
                    "score_fail": res.score_fail,
                    "score_ref": res.score_ref,
                    "run_dir": str(res.run_dir),
                }
            )
            print(
                f"  {cand.as_key()}  n_done={res.n_done}/{budget}  "
                f"score={res.score:.3e} (fail={res.score_fail:.3e}, ref={res.score_ref:.3e})"
            )

        # Keep top fraction
        round_results.sort(key=lambda rr: rr.score)
        keep = max(1, math.ceil(len(round_results) / max(1, int(args.eta))))
        alive = [rr.candidate for rr in round_results[:keep]]
        print(f"[opt] keep={keep}/{len(round_results)}")

    # Best among the last round
    final_results = [r for r in all_results if int(r.get("budget_steps", 0)) == int(budgets[-1])]
    best_key = min(final_results, key=lambda r: float(r["score"]))["candidate"] if final_results else None
    best_run = None
    if best_key is not None:
        for r in all_results:
            if r["candidate"] == best_key and int(r["budget_steps"]) == int(budgets[-1]):
                best_run = r
                break

    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps({"meta": meta, "results": all_results}, indent=2, sort_keys=True) + "\n")

    if best_run is None:
        print("[opt] no successful runs recorded")
        return 2

    print(f"[opt] best={best_run['candidate']}  score={best_run['score']:.3e}  n_done={best_run['n_done']}")

    tuned_path = Path(__file__).resolve().parent / "tuned_params.json"
    # Reconstruct best RunResult minimal view for writing.
    best_candidate = Candidate(
        beta0=float(best_run["beta0"]),
        gamma_gp=float(best_run["gamma_gp"]),
        gamma_gp_p=(None if best_run["gamma_gp_p"] is None else float(best_run["gamma_gp_p"])),
        gamma_gp_hess=float(best_run["gamma_gp_hess"]),
    )
    best = RunResult(
        candidate=best_candidate,
        budget_steps=int(best_run["budget_steps"]),
        exit_code=int(best_run["exit_code"]),
        n_done=int(best_run["n_done"]),
        score=float(best_run["score"]),
        score_fail=float(best_run["score_fail"]),
        score_ref=float(best_run["score_ref"]),
        functionals_csv=Path(best_run["run_dir"]) / "functionals.csv",
        run_dir=Path(best_run["run_dir"]),
    )
    _update_tuned_params(best, tuned_path=tuned_path, meta=meta)
    print(f"[opt] wrote tuned params -> {tuned_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
