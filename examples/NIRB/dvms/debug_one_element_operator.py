from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Kratos one-element DVMS dump and the pycutfem local Jacobian "
            "comparison in one command."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("examples/NIRB/artifacts/kratos_fluid_local_operator_debug"),
        help="Temporary Kratos run directory.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=None,
        help="Optional DoubleFlap benchmark root. Defaults to the downloaded reference root.",
    )
    parser.add_argument("--element-id", type=int, default=None, help="Optional Kratos element id to dump.")
    parser.add_argument("--interface-part", type=str, default="NoSlip2D_Interface")
    parser.add_argument("--end-time", type=float, default=0.008)
    parser.add_argument(
        "--stage",
        choices=("predicted", "solved"),
        default="solved",
        help="Dump/compare the local operator before SolveSolutionStep() ('predicted') or after the first solved step ('solved').",
    )
    parser.add_argument("--dt", type=float, default=0.008)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    parser.add_argument("--quadrature-order", type=int, default=3)
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default="python")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("examples/NIRB/artifacts/one_element_operator_debug.json"),
        help="Final comparison report path.",
    )
    return parser.parse_args()


def _run(cmd: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    run_dir = Path(args.run_dir).resolve()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_stem = f"fluid_local_element_{str(args.stage)}"
    kratos_dump = run_dir / f"{output_stem}.npz"

    dump_cmd = [
        sys.executable,
        "examples/NIRB/dvms/dump_kratos_local_operator.py",
        "--run-dir",
        str(run_dir),
        "--end-time",
        str(float(args.end_time)),
        "--interface-part",
        str(args.interface_part),
        "--mode",
        "all",
        "--stage",
        str(args.stage),
        "--output-stem",
        output_stem,
    ]
    if args.benchmark_root is not None:
        dump_cmd.extend(["--benchmark-root", str(Path(args.benchmark_root).resolve())])
    if args.element_id is not None:
        dump_cmd.extend(["--element-id", str(int(args.element_id))])

    compare_cmd = [
        sys.executable,
        "examples/NIRB/dvms/compare_kratos_local_terms.py",
        "--kratos-dump",
        str(kratos_dump),
        "--output",
        str(output_json),
        "--quadrature-order",
        str(int(args.quadrature_order)),
        "--dt",
        str(float(args.dt)),
        "--bossak-alpha",
        str(float(args.bossak_alpha)),
        "--backend",
        str(args.backend),
    ]

    _run(dump_cmd, cwd=repo_root)
    _run(compare_cmd, cwd=repo_root)

    report = json.loads(output_json.read_text(encoding="utf-8"))
    key_summary = {
        "kratos_dump": report.get("kratos_dump"),
        "stage": str(args.stage),
        "element_id_kratos": report.get("element_id_kratos"),
        "element_id_pycutfem": report.get("element_id_pycutfem"),
        "kratos_raw_calculate_local_system_is_zero": report.get("kratos_raw_calculate_local_system_is_zero"),
        "system_jacobian_vs_reconstructed_ref": report.get("system_jacobian_vs_reconstructed_ref"),
        "system_condensed_jacobian_vs_reconstructed_ref": report.get("system_condensed_jacobian_vs_reconstructed_ref"),
        "system_toggle_report_vs_reconstructed_ref": report.get("system_toggle_report_vs_reconstructed_ref"),
    }
    print(json.dumps(key_summary, indent=2))
    print(output_json)


if __name__ == "__main__":
    main()
