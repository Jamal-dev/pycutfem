from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from examples.NIRB.reduced_trajectory import (
    TrajectoryReducedArtifact,
    TrajectoryReducedOnlineSolver,
    build_trajectory_reduced_artifact,
    validate_reconstruction,
)


def _write_fixture(root: Path, *, steps: int = 3) -> None:
    co = root / "coSimData"
    sh = root / "step_history"
    co.mkdir(parents=True)
    sh.mkdir()
    rows: list[dict[str, object]] = []
    load_guess_cols: list[np.ndarray] = []
    load_return_cols: list[np.ndarray] = []
    disp_cols: list[np.ndarray] = []
    iters: list[int] = []
    for step in range(1, steps + 1):
        iters.append(2)
        for it in range(1, 3):
            rows.append({"step": step, "time_s": 0.1 * step, "coupling_iter": it, "converged": it == 2})
            guess = np.array([step + it, -0.5 * step + it], dtype=float)
            ret = guess * (0.2 if it == 1 else 1.0)
            disp = np.array([0.1 * step, 0.05 * it], dtype=float)
            load_guess_cols.append(guess)
            load_return_cols.append(ret)
            disp_cols.append(disp)
        velocity = np.array([[step, 2.0 * step], [0.5 * step, -step]], dtype=float)
        pressure = np.array([[step], [step + 1.0]], dtype=float)
        mesh_disp = 0.1 * velocity
        mesh_vel = 0.2 * velocity
        structure = np.array([[0.3 * step, -0.1 * step], [0.2 * step, 0.4 * step]], dtype=float)
        np.savez_compressed(
            sh / f"step{step:04d}.npz",
            step=np.asarray(step, dtype=int),
            time_s=np.asarray(0.1 * step, dtype=float),
            fluid_node_ids=np.array([10, 11], dtype=int),
            fluid_coords_ref=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
            fluid_velocity_nodal_values=velocity,
            fluid_pressure_nodal_values=pressure,
            fluid_mesh_displacement_nodal_values=mesh_disp,
            fluid_mesh_velocity_nodal_values=mesh_vel,
            structure_node_ids=np.array([20, 21], dtype=int),
            structure_coords_ref=np.array([[0.0, 1.0], [1.0, 1.0]], dtype=float),
            structure_displacement_nodal_values=structure,
            interface_load_coords_ref=np.array([[0.0, 0.0]], dtype=float),
            interface_load_values=load_return_cols[-1].reshape(1, 2),
            interface_disp_coords_ref=np.array([[0.0, 0.0]], dtype=float),
            interface_disp_values=disp_cols[-1].reshape(1, 2),
            interface_velocity_coords_ref=np.array([[0.0, 0.0]], dtype=float),
            interface_velocity_values=np.zeros((1, 2), dtype=float),
        )
    with (root / "snapshot_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "time_s", "coupling_iter", "converged"])
        writer.writeheader()
        writer.writerows(rows)
    np.save(co / "iters.npy", np.asarray(iters, dtype=int))
    np.save(co / "load_guess_data.npy", np.column_stack(load_guess_cols))
    np.save(co / "load_return_data.npy", np.column_stack(load_return_cols))
    np.save(co / "load_data.npy", np.column_stack(load_return_cols))
    np.save(co / "interface_disp_data.npy", np.column_stack(disp_cols))
    np.save(co / "interface_velocity_data.npy", np.zeros((2, steps * 2), dtype=float))
    np.save(co / "coords_interf.npy", np.array([[0.0, 0.0]], dtype=float))
    np.save(co / "coords_interf_fluid.npy", np.array([[0.0, 0.0]], dtype=float))
    (root / "summary.json").write_text('{"dt": 0.1}', encoding="utf-8")


def test_trajectory_reduced_artifact_runs_and_reconstructs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_fixture(source, steps=3)
    artifact = build_trajectory_reduced_artifact(
        source,
        interface_modes=2,
        state_modes=3,
        center=True,
    )
    path = tmp_path / "artifact.npz"
    artifact.save(path)
    loaded = TrajectoryReducedArtifact.load(path)

    result = TrajectoryReducedOnlineSolver(loaded).run()
    assert result.steps_converged == 3
    assert result.coupling_iters_per_step == (2, 2, 2)

    validation = validate_reconstruction(loaded, source)
    assert validation["validated_steps"] == 3
    assert validation["max_relative_error"] < 1.0e-12

    reconstructed = loaded.reconstruct_step(3)
    np.testing.assert_allclose(
        reconstructed["fluid_velocity_nodal_values"],
        np.array([[3.0, 6.0], [1.5, -3.0]], dtype=float),
    )
