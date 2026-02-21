#!/usr/bin/env python3
"""
Compare `bouncing_ball.py` outputs against Table 2 in the paper TeX.

This script is intentionally lightweight: it just loads `metrics.json` and
prints absolute/relative differences for the quantities reported in Table 2
(uniform mesh refinement levels 0–3).

Usage
-----
  python examples/fsi_contact/compare_to_paper_table2.py output_bouncing_ball/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PAPER_TABLE2 = {
    # Table 2, uniform mesh refinement levels 0..3 (DoFs as in the paper)
    2695: {
        "t0": 0.2072,
        "t_star": 0.0623,
        "v_star": -0.1021,
        "f_star": -4.6861,
        "t_cont": 0.3465,
        "t_jump": None,
        "h_jump": None,
        "max_p_bc": 677.3779,
        "max_v_f": 0.1485,
        "max_E_el": 0.0001,
        "max_E_kin_f": 0.0087,
        "max_E_kin_s": 0.0030,
        "n_newton_avg": 1.1952,
    },
    9928: {
        "t0": 0.2024,
        "t_star": 0.0613,
        "v_star": -0.1035,
        "f_star": -4.9170,
        "t_cont": 0.2738,
        "t_jump": 0.3037,
        "h_jump": 0.000203,
        "max_p_bc": 577.9210,
        "max_v_f": 0.1630,
        "max_E_el": 0.0031,
        "max_E_kin_f": 0.0089,
        "max_E_kin_s": 0.0032,
        "n_newton_avg": 1.2280,
    },
    37660: {
        "t0": 0.2038,
        "t_star": 0.0613,
        "v_star": -0.1034,
        "f_star": -4.8917,
        "t_cont": 0.2742,
        "t_jump": 0.3068,
        "h_jump": 0.000267,
        "max_p_bc": 575.0429,
        "max_v_f": 0.1705,
        "max_E_el": 0.0034,
        "max_E_kin_f": 0.0090,
        "max_E_kin_s": 0.0032,
        "n_newton_avg": 1.2496,
    },
    146648: {
        "t0": 0.2045,
        "t_star": 0.0615,
        "v_star": -0.1030,
        "f_star": -4.6639,
        "t_cont": 0.2772,
        "t_jump": 0.3053,
        "h_jump": 0.000158,
        "max_p_bc": 575.5481,
        "max_v_f": 0.1737,
        "max_E_el": 0.0021,
        "max_E_kin_f": 0.0091,
        "max_E_kin_s": 0.0031,
        "n_newton_avg": 1.3469,
    },
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare bouncing-ball metrics to Table 2 (paper).")
    p.add_argument("metrics", type=str, help="Path to metrics.json")
    p.add_argument("--dofs", type=int, default=None, help="Override DoF count used to select a Table 2 row.")
    return p.parse_args()


def _fmt(val) -> str:
    if val is None:
        return "—"
    return f"{float(val):.6g}"


def main() -> None:
    args = _parse_args()
    metrics_path = Path(args.metrics)
    data = json.loads(metrics_path.read_text())
    qoi = dict(data.get("qoi", {}))

    dofs_dict = dict(data.get("dofs", {}))
    dofs_from_file = int(dofs_dict.get("paper_initial", dofs_dict.get("paper", dofs_dict.get("total", -1))))
    dofs_used = int(args.dofs) if args.dofs is not None else dofs_from_file
    if dofs_used not in PAPER_TABLE2:
        known = ", ".join(map(str, sorted(PAPER_TABLE2.keys())))
        raise SystemExit(
            f"DoFs={dofs_used} not found in Table 2 map. Known DoFs: {known}. "
            "Pass --dofs to select a row explicitly."
        )

    ref = PAPER_TABLE2[dofs_used]
    keys = [
        "t0",
        "t_star",
        "v_star",
        "f_star",
        "t_cont",
        "t_jump",
        "h_jump",
        "max_p_bc",
        "max_v_f",
        "max_E_el",
        "max_E_kin_f",
        "max_E_kin_s",
        "n_newton_avg",
    ]

    print(f"metrics: {metrics_path}")
    print(f"DoFs (used): {dofs_used}")
    if args.dofs is None:
        dofs_total = int(dofs_dict.get("total", -1))
        dofs_free = int(dofs_dict.get("free", -1))
        dofs_dir = int(dofs_dict.get("dirichlet", -1))
        dofs_free0 = int(dofs_dict.get("free_initial", -1))
        dofs_dir0 = int(dofs_dict.get("dirichlet_initial", -1))
        dofs_paper0 = int(dofs_dict.get("paper_initial", -1))
        print(
            f"  dofs.total={dofs_total}, dofs.free={dofs_free}, dofs.dirichlet={dofs_dir}, "
            f"dofs.free_initial={dofs_free0}, dofs.dirichlet_initial={dofs_dir0}, "
            f"dofs.paper_initial={dofs_paper0}"
        )
    print("")
    print(f"{'key':12s}  {'paper':>12s}  {'pycutfem':>12s}  {'abs diff':>12s}  {'rel diff':>12s}")
    print("-" * 70)
    for k in keys:
        v_ref = ref.get(k, None)
        v = qoi.get(k, None)
        if v_ref is None or v is None:
            print(f"{k:12s}  {_fmt(v_ref):>12s}  {_fmt(v):>12s}  {'—':>12s}  {'—':>12s}")
            continue
        v_ref_f = float(v_ref)
        v_f = float(v)
        abs_diff = abs(v_f - v_ref_f)
        rel = abs_diff / (abs(v_ref_f) if abs(v_ref_f) > 0.0 else 1.0)
        print(f"{k:12s}  {_fmt(v_ref_f):>12s}  {_fmt(v_f):>12s}  {_fmt(abs_diff):>12s}  {_fmt(rel):>12s}")


if __name__ == "__main__":
    main()
