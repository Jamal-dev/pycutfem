from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.NIRB.reduced_mesh import fit_reduced_mesh_displacement_map
from examples.NIRB.reduced_trajectory import TrajectoryReducedArtifact
from pycutfem.mor.pod import reconstruct_from_basis


def _accepted_coupling_indices(artifact: TrajectoryReducedArtifact) -> np.ndarray:
    indices: list[int] = []
    coupling_steps = np.asarray(artifact.coupling_steps, dtype=int).reshape(-1)
    converged = np.asarray(artifact.coupling_converged, dtype=bool).reshape(-1)
    for step in np.asarray(artifact.accepted_steps, dtype=int).reshape(-1):
        matches = np.flatnonzero((coupling_steps == int(step)) & converged)
        if matches.size == 0:
            matches = np.flatnonzero(coupling_steps == int(step))
        if matches.size == 0:
            raise RuntimeError(f"accepted step {int(step)} has no coupling snapshots.")
        indices.append(int(matches[-1]))
    return np.asarray(indices, dtype=int)


def _reconstruct_columns(coefficients: np.ndarray, basis: np.ndarray, mean: np.ndarray | None) -> np.ndarray:
    coeffs = np.asarray(coefficients, dtype=float)
    columns = [
        reconstruct_from_basis(coeffs[:, idx], basis, mean).reshape(-1)
        for idx in range(int(coeffs.shape[1]))
    ]
    return np.column_stack(columns)


def build_reduced_mesh_surrogate(
    *,
    trajectory_artifact: Path,
    output: Path,
    interface_modes: int | None,
    mesh_modes: int | None,
    ridge: float,
) -> dict[str, object]:
    artifact = TrajectoryReducedArtifact.load(trajectory_artifact)
    accepted_idx = _accepted_coupling_indices(artifact)

    interface_basis_full = np.asarray(artifact.displacement_basis.basis, dtype=float)
    mesh_basis_full = np.asarray(artifact.mesh_displacement_basis.basis, dtype=float)
    n_interface = int(interface_basis_full.shape[1]) if interface_modes is None else int(interface_modes)
    n_mesh = int(mesh_basis_full.shape[1]) if mesh_modes is None else int(mesh_modes)
    n_interface = max(1, min(n_interface, int(interface_basis_full.shape[1])))
    n_mesh = max(1, min(n_mesh, int(mesh_basis_full.shape[1])))
    interface_basis = interface_basis_full[:, :n_interface]
    mesh_basis = mesh_basis_full[:, :n_mesh]

    accepted_interface_coeffs_full = np.asarray(artifact.displacement_coefficients, dtype=float)[:, accepted_idx]
    accepted_mesh_coeffs_full = np.asarray(artifact.accepted_mesh_displacement_coefficients, dtype=float)
    interface_snapshots = _reconstruct_columns(
        accepted_interface_coeffs_full,
        interface_basis_full,
        artifact.displacement_basis.mean,
    )
    mesh_snapshots = _reconstruct_columns(
        accepted_mesh_coeffs_full,
        mesh_basis_full,
        artifact.mesh_displacement_basis.mean,
    )
    fitted = fit_reduced_mesh_displacement_map(
        interface_basis=interface_basis,
        interface_mean=artifact.displacement_basis.mean,
        interface_snapshots=interface_snapshots,
        mesh_basis=mesh_basis,
        mesh_mean=artifact.mesh_displacement_basis.mean,
        mesh_snapshots=mesh_snapshots,
        ridge=float(ridge),
        fluid_coords_ref=np.asarray(artifact.fluid_coords_ref, dtype=float),
        interface_coords_ref=np.asarray(artifact.interface_coords_ref, dtype=float),
        source_path=str(trajectory_artifact),
        training_steps=np.asarray(artifact.accepted_steps, dtype=int),
    )
    fitted.save(output)
    errors = np.asarray(fitted.training_relative_errors, dtype=float)
    summary = {
        "trajectory_artifact": str(trajectory_artifact),
        "output": str(output),
        "accepted_steps": int(np.asarray(artifact.accepted_steps).size),
        "interface_modes": int(n_interface),
        "mesh_modes": int(n_mesh),
        "ridge": float(ridge),
        "mean_training_relative_error": float(np.mean(errors)) if errors.size else 0.0,
        "max_training_relative_error": float(np.max(errors)) if errors.size else 0.0,
        "p95_training_relative_error": float(np.percentile(errors, 95)) if errors.size else 0.0,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a reduced ALE mesh displacement surrogate from a trajectory artifact.")
    parser.add_argument("--trajectory-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interface-modes", type=int, default=None)
    parser.add_argument("--mesh-modes", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=1.0e-10)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_reduced_mesh_surrogate(
        trajectory_artifact=Path(args.trajectory_artifact),
        output=Path(args.output),
        interface_modes=args.interface_modes,
        mesh_modes=args.mesh_modes,
        ridge=float(args.ridge),
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary is not None:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
