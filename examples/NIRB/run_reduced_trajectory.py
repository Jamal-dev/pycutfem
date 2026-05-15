from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.NIRB.reduced_trajectory import (
    TrajectoryReducedArtifact,
    TrajectoryReducedOnlineSolver,
    build_trajectory_reduced_artifact,
    validate_reconstruction,
    write_timeseries,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and run a coefficient-only reduced replay of an Example 2 FSI trajectory. "
            "Full fields are reconstructed only for validation/output."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-path", type=Path, default=None)
    parser.add_argument("--rebuild", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--interface-modes", type=int, default=None)
    parser.add_argument("--state-modes", type=int, default=None)
    parser.add_argument("--interface-energy", type=float, default=None)
    parser.add_argument("--state-energy", type=float, default=None)
    parser.add_argument("--center", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-reconstruction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--validation-stride",
        type=int,
        default=1,
        help="Validate every Nth accepted step; 1 validates every reconstructed accepted step.",
    )
    parser.add_argument("--save-final-reconstruction", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = Path(args.artifact_path) if args.artifact_path is not None else output_dir / "trajectory_reduced.npz"

    if bool(args.rebuild) or not artifact_path.exists():
        artifact = build_trajectory_reduced_artifact(
            args.source_dir,
            max_steps=args.max_steps,
            interface_modes=args.interface_modes,
            state_modes=args.state_modes,
            interface_energy=args.interface_energy,
            state_energy=args.state_energy,
            center=bool(args.center),
        )
        artifact.save(artifact_path)
    else:
        artifact = TrajectoryReducedArtifact.load(artifact_path)

    solver = TrajectoryReducedOnlineSolver(artifact)
    result = solver.run(max_steps=args.max_steps)
    write_timeseries(output_dir / "timeseries.csv", result.timeseries)

    validation: dict[str, float | int] = {}
    if bool(args.validate_reconstruction):
        stride = max(int(args.validation_stride), 1)
        steps = np.asarray(artifact.accepted_steps, dtype=int)
        if args.max_steps is not None:
            steps = steps[steps <= int(args.max_steps)]
        validate_steps = steps[::stride]
        if steps.size and int(steps[-1]) not in set(int(v) for v in validate_steps):
            validate_steps = np.concatenate([validate_steps, steps[-1:]])
        validation = validate_reconstruction(artifact, args.source_dir, steps=validate_steps)

    if bool(args.save_final_reconstruction) and int(result.steps_converged) > 0:
        final_step = int(artifact.accepted_steps[min(int(result.steps_converged), artifact.accepted_steps.size) - 1])
        final = artifact.reconstruct_step(final_step)
        np.savez_compressed(output_dir / f"reconstruction_step_{final_step:04d}.npz", **final)

    summary = {
        **result.summary(),
        "artifact_path": str(artifact_path),
        "source_dir": str(args.source_dir),
        "reconstruction_validation": validation,
        "artifact_metadata": dict(artifact.metadata),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"artifact: {artifact_path}")
    print(f"steps_requested: {summary['steps_requested']}")
    print(f"steps_converged: {summary['steps_converged']}")
    print(f"mean_coupling_iters: {summary['mean_coupling_iters']:.3f}")
    print(f"online_time_s: {summary['online_time_s']:.6f}")
    if validation:
        print(f"max_reconstruction_relative_error: {validation['max_relative_error']:.6e}")
    print(f"summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
