"""
Rebuild `tuned_params.json` from finished `opt_runs/*/summary.json`.

Why
---
If multiple `optimize_params.py` jobs run concurrently, a non-atomic update of
`tuned_params.json` can lose entries (last writer wins). Each finished run writes
its own `opt_runs/<timestamp>/summary.json`, which is the authoritative record.

This script scans those summaries and reconstructs the best entry per "case".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any


def _same_case(a: dict[str, Any], b: dict[str, Any]) -> bool:
    def _f(x, default: float) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    return (
        str(a.get("benchmark")) == str(b.get("benchmark"))
        and int(a.get("level", -1)) == int(b.get("level", -2))
        and str(a.get("ghost_measure")) == str(b.get("ghost_measure"))
        and bool(a.get("with_deformation")) == bool(b.get("with_deformation"))
        and int(a.get("fe_order", 0)) == int(b.get("fe_order", 0))
        and int(a.get("p_order", 0)) == int(b.get("p_order", 0))
        and abs(_f(a.get("dt"), float("nan")) - _f(b.get("dt"), float("nan"))) <= 1e-12
        and abs(_f(a.get("theta"), float("nan")) - _f(b.get("theta"), float("nan"))) <= 1e-12
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{int(time.time())}")
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp.write_text(text)
    os.replace(str(tmp), str(path))


def _entry_from_summary(summary: dict[str, Any], *, summary_path: Path) -> dict[str, Any] | None:
    meta = summary.get("meta")
    results = summary.get("results")
    if not isinstance(meta, dict) or not isinstance(results, list) or not results:
        return None

    # Prefer the last-round budget (the one optimize_params.py uses for the final choice).
    max_budget = -1
    for r in results:
        try:
            max_budget = max(max_budget, int(r.get("budget_steps", -1)))
        except Exception:
            continue

    final = [r for r in results if int(r.get("budget_steps", -1)) == max_budget] if max_budget >= 0 else list(results)
    if not final:
        return None

    def _score(r: dict[str, Any]) -> float:
        try:
            return float(r.get("score", float("inf")))
        except Exception:
            return float("inf")

    best = min(final, key=_score)
    score = _score(best)
    if not math.isfinite(score):
        return None

    entry: dict[str, Any] = {
        "benchmark": str(meta.get("benchmark")),
        "level": int(meta.get("level")),
        "dt": float(meta.get("dt")),
        "theta": float(meta.get("theta", 0.5)),
        "ghost_measure": str(meta.get("ghost_measure")),
        "with_deformation": bool(meta.get("with_deformation")),
        "fe_order": int(meta.get("fe_order", 2)),
        "p_order": int(meta.get("p_order", 1)),
        "beta0": float(best.get("beta0")),
        "gamma_gp": float(best.get("gamma_gp")),
        "gamma_gp_hess": float(best.get("gamma_gp_hess")),
        "score": float(score),
        "n_done": int(best.get("n_done", 0)),
        "ts": int(time.time()),
        "source_summary": str(summary_path),
    }
    if best.get("gamma_gp_p", None) is not None:
        entry["gamma_gp_p"] = float(best.get("gamma_gp_p"))
    return entry


def _best_of_case(entries: list[dict[str, Any]], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    same = [e for e in entries if _same_case(e, candidate)]
    if same:
        old_best = min(float(e.get("score", float("inf"))) for e in same)
        new_score = float(candidate.get("score", float("inf")))
        if not math.isfinite(new_score) or new_score >= old_best:
            return entries
    return [e for e in entries if not _same_case(e, candidate)] + [candidate]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--opt-runs-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "opt_runs"),
        help="Directory containing optimize_params.py run folders with summary.json files",
    )
    ap.add_argument(
        "--tuned-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "tuned_params.json"),
        help="Path to tuned_params.json to write",
    )
    ap.add_argument("--dry-run", action="store_true", default=False)
    args = ap.parse_args()

    opt_runs_dir = Path(args.opt_runs_dir).resolve()
    tuned_path = Path(args.tuned_path).resolve()

    summaries = sorted(opt_runs_dir.glob("*/summary.json"))
    if not summaries:
        print(f"[rebuild] no summaries found under {opt_runs_dir}")
        return 2

    entries: list[dict[str, Any]] = []
    for sp in summaries:
        try:
            summary = json.loads(sp.read_text())
        except Exception:
            continue
        entry = _entry_from_summary(summary, summary_path=sp)
        if entry is None:
            continue
        entries = _best_of_case(entries, entry)

    doc = {"version": 1, "entries": entries}
    if args.dry_run:
        print(json.dumps(doc, indent=2, sort_keys=True))
        return 0

    _atomic_write_json(tuned_path, doc)
    print(f"[rebuild] wrote {len(entries)} entries -> {tuned_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

