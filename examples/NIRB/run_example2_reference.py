from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.NIRB.example2_local_setup import load_example2_local_setup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the local Example 2 geometry/BC/form specification from the DoubleFlap reference files."
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="Path to the downloaded DoubleFlap reference directory. Defaults to .tmp/nirb_benchmarks/DoubleFlap.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/NIRB/artifacts/example2_reference"),
        help="Directory for the generated JSON summary and geometry plot.",
    )
    parser.add_argument(
        "--reynolds",
        type=float,
        default=250.0,
        help="Reynolds number used for the local baseline run config summary.",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip the geometry plot.",
    )
    parser.set_defaults(plot=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup = load_example2_local_setup(reference_root=args.reference_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "example2_local_setup.json"
    summary = setup.to_dict(reynolds=args.reynolds)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_path = None
    if args.plot:
        plot_path = setup.reference.plot_geometry(args.output_dir / "double_flap_reference_geometry.png")

    print(f"Wrote Example 2 reference summary to {summary_path}")
    if plot_path is not None:
        print(f"Wrote Example 2 geometry plot to {plot_path}")
    print(
        "Local baseline run config: "
        f"u_mean={summary['baseline_run_config']['u_mean']:.6f}, "
        f"dt={summary['baseline_run_config']['dt']:.6f}, "
        f"mesh={summary['baseline_run_config']['mesh']}"
    )


if __name__ == "__main__":
    main()
