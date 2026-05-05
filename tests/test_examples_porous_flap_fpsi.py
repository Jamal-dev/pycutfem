from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pycutfem.mor.snapshots import SnapshotBatch
from pycutfem.nirb.dataset import load_cosim_snapshot_batch

from examples.poromechanics.porous_flap_fpsi import (
    build_porous_flap_interface_stations,
    nominal_double_flap_geometry,
    run_porous_flap_fpsi,
    write_porous_flap_fpsi_result,
    write_porous_flap_fpsi_vtk,
)


def test_porous_flap_interface_stations_follow_nirb_flap_boundary() -> None:
    geometry = nominal_double_flap_geometry()
    stations = build_porous_flap_interface_stations(geometry, target_spacing=0.025)

    assert stations.size == 56
    assert np.all(stations.weights > 0.0)
    assert np.allclose(np.linalg.norm(stations.normals, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(stations.tangents, axis=1), 1.0)
    assert set(stations.segment_names) == {
        "left_outer",
        "left_inner",
        "left_top",
        "right_inner",
        "right_outer",
        "right_top",
        "base_gap_floor",
    }

    eps = 1.0e-6
    inside = stations.points - eps * stations.normals
    outside = stations.points + eps * stations.normals
    assert all(geometry.contains_solid_point(float(x), float(y), tol=1.0e-10) for x, y in inside)
    assert not any(geometry.contains_solid_point(float(x), float(y), tol=1.0e-10) for x, y in outside)


def test_porous_flap_fpsi_constant_relaxation_converges_and_writes_outputs(tmp_path: Path) -> None:
    result = run_porous_flap_fpsi(
        n_steps=4,
        dt=0.02,
        tolerance=1.0e-10,
        max_iterations=80,
        accelerator="constant",
        accelerator_backend="python",
        relaxation=0.45,
    )

    assert result.converged
    assert result.max_iterations <= 30
    assert np.all(np.isfinite(result.final_state))
    assert result.final_state[:, 2].max() > 0.0
    assert max(step.max_displacement for step in result.steps) > 0.0
    assert result.load_guess_data.shape[0] == 3 * result.station_count
    assert result.porous_state_data.shape == result.load_guess_data.shape
    assert result.disp_data.shape[0] == 2 * result.station_count
    assert result.pressure_data.shape[0] == result.station_count
    assert result.n_snapshots == len(result.snapshot_metadata)
    assert result.n_snapshots >= len(result.steps)

    output = write_porous_flap_fpsi_result(result, tmp_path, write_vtk=True)
    assert (output / "summary.json").exists()
    assert (output / "timeseries.csv").exists()
    assert (output / "iterations.csv").exists()
    assert (output / "snapshot_metadata.csv").exists()
    assert (output / "vtk" / "vtk_manifest.csv").exists()
    assert (output / "vtk" / "porous_fpsi.pvd").exists()
    assert (output / "coSimData" / "load_guess_data.npy").exists()
    assert (output / "coSimData" / "porous_state_data.npy").exists()
    assert (output / "fpsi_snapshot_batch.npz").exists()
    assert (output / "final_state.npy").exists()
    vtk_files = sorted((output / "vtk").glob("*.vtu"))
    assert len(vtk_files) == len(result.steps)
    vtk_text = vtk_files[-1].read_text(encoding="utf-8")
    assert "UnstructuredGrid" in vtk_text
    assert "fluid_velocity" in vtk_text
    assert "pore_pressure" in vtk_text
    assert "normal_flux" in vtk_text

    co_sim_batch = load_cosim_snapshot_batch(
        output,
        force_key="load_guess_data",
        displacement_key="porous_state_data",
    )
    np.testing.assert_allclose(co_sim_batch.interface_forces, result.load_guess_data)
    np.testing.assert_allclose(co_sim_batch.full_displacements, result.porous_state_data)

    saved_batch = SnapshotBatch.load(output / "fpsi_snapshot_batch.npz")
    np.testing.assert_allclose(saved_batch.interface_forces, result.load_guess_data)
    np.testing.assert_allclose(saved_batch.full_displacements, result.porous_state_data)

    all_vtk = write_porous_flap_fpsi_vtk(result, tmp_path / "vtk_all", mode="all")
    assert len(all_vtk) == result.n_snapshots


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_porous_flap_fpsi_iqln_converges_on_all_steps(backend: str) -> None:
    result = run_porous_flap_fpsi(
        n_steps=8,
        dt=0.02,
        tolerance=1.0e-10,
        max_iterations=40,
        accelerator="iqln",
        accelerator_backend=backend,
        relaxation=0.45,
        history=8,
        timestep_horizon=3,
    )

    assert result.converged
    assert len(result.steps) == 8
    assert result.max_iterations <= 8
    assert all(step.residual_norm <= 1.0e-10 for step in result.steps)
    assert any(record.used_history for record in result.iterations if record.method == "iqln")
    assert result.steps[-1].max_pressure > result.steps[0].max_pressure
