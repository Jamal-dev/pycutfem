from __future__ import annotations

import argparse
from pathlib import Path

from examples.NIRB.debug.compare_example2_step_history import (
    compare_step_histories,
    write_compare_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that a local Example 2 accepted-step history matches the Kratos reference through step 10."
    )
    parser.add_argument(
        "--local-step-dir",
        type=Path,
        required=True,
        help="Local output dir or local step_history dir.",
    )
    parser.add_argument(
        "--kratos-step-dir",
        type=Path,
        default=Path("examples/NIRB/artifacts/kratos_step_history_0140_0145/step_history"),
        help="Kratos run dir or Kratos step_history dir.",
    )
    parser.add_argument(
        "--required-step",
        type=int,
        default=10,
        help="Require this accepted step to be present in the comparison.",
    )
    parser.add_argument(
        "--max-step-worst-rel-l2",
        type=float,
        default=1.0e-8,
        help="Maximum allowed worst relative L2 mismatch on the required step.",
    )
    parser.add_argument(
        "--divergence-threshold",
        type=float,
        default=5.0e-3,
        help="Threshold passed through to compare_example2_step_history.py.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def _required_step_row(summary: dict, required_step: int) -> dict:
    for row in summary.get("rows", []):
        if int(row.get("step", -1)) == int(required_step):
            return row
    raise RuntimeError(f"Required step {int(required_step)} is missing from the comparison output.")


def main() -> None:
    args = parse_args()
    local_step_dir = Path(args.local_step_dir).resolve()
    output_parent = local_step_dir if local_step_dir.name != "step_history" else local_step_dir.parent
    output_json = (
        Path(args.output_json).resolve()
        if args.output_json is not None
        else output_parent / "comparison_step_history.json"
    )
    output_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv is not None
        else output_parent / "comparison_step_history.csv"
    )

    summary = compare_step_histories(
        local_step_dir=local_step_dir,
        kratos_step_dir=Path(args.kratos_step_dir),
        divergence_threshold=float(args.divergence_threshold),
    )
    write_compare_outputs(summary, output_json=output_json, output_csv=output_csv)

    last_compared_step = int(summary.get("last_compared_step", 0) or 0)
    if last_compared_step < int(args.required_step):
        raise RuntimeError(
            f"Comparison stopped at step {last_compared_step}, below required step {int(args.required_step)}."
        )
    if summary.get("first_divergence_step") is not None:
        raise RuntimeError(
            f"Detected divergence at step {int(summary['first_divergence_step'])}; expected no divergence."
        )

    step_row = _required_step_row(summary, int(args.required_step))
    worst_value = float(step_row.get("worst_rel_l2_value", float("inf")))
    if worst_value > float(args.max_step_worst_rel_l2):
        raise RuntimeError(
            "Step "
            f"{int(args.required_step)} worst_rel_l2_value={worst_value:.16e} "
            f"exceeds limit {float(args.max_step_worst_rel_l2):.16e}."
        )

    print(f"comparison_json: {output_json}")
    print(f"comparison_csv: {output_csv}")
    print(
        "step10_exact_ok: "
        f"step={int(args.required_step)} "
        f"worst_key={step_row.get('worst_rel_l2_key')} "
        f"worst_rel_l2_value={worst_value:.16e}"
    )


if __name__ == "__main__":
    main()
