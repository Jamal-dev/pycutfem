from __future__ import annotations

import subprocess
from pathlib import Path

from examples.NIRB.common import default_artifacts_root, dump_json
from examples.NIRB.example1_workflow import run_example1
from examples.NIRB.example2_workflow import run_example2


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    ).strip()


if __name__ == "__main__":
    example1 = run_example1()
    example2 = run_example2()

    artifacts_root = default_artifacts_root()
    report_path = artifacts_root / "validation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# NIRB Validation Report",
                "",
                f"- Git commit: `{_git_commit()}`",
                "- Command: `conda run --no-capture-output -n fenicsx python examples/NIRB/run_all.py`",
                "",
                "## Example 1",
                f"- Pressure modes / section modes: `{example1['paper_selected_modes']['pressure_modes']}` / `{example1['paper_selected_modes']['section_modes']}`",
                f"- Same-mu coupled iteration overhead: `{example1['published_coupled_results']['same_mu_iteration_overhead']:.4f}`",
                f"- Other-mu coupled pressure max relative error: `{example1['published_coupled_results']['other_mu_pressure_max_relative_error']:.4f}`",
                "",
                "## Example 2",
                f"- Force modes / displacement modes: `{example2['paper_selected_modes']['force_modes']}` / `{example2['paper_selected_modes']['displacement_modes']}`",
                f"- Offline validation max relative error: `{example2['offline_validation']['validation_max_relative_error']:.4f}`",
                f"- Re~300 online max relative error: `{example2['online_re300'].get('max_relative_displacement_error', float('nan')):.4f}`",
                f"- Re~300 overall speedup: `{example2['online_re300'].get('overall_speedup', float('nan')):.4f}`",
            ]
        ),
        encoding="utf-8",
    )
    dump_json(
        {
            "example1": example1,
            "example2": example2,
            "report_path": str(report_path),
            "git_commit": _git_commit(),
        },
        artifacts_root / "validation_report.json",
    )
    print(report_path)
