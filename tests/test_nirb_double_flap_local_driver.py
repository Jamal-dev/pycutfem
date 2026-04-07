from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pycutfem.utils.gmsh_loader import mesh_from_gmsh

from examples.NIRB.example2_problem import build_conforming_mesh
from examples.NIRB.run_example2_local import (
    _aitken_relaxation_factor,
    _boundary_field_data,
    _bossak_coefficients,
    _build_fluid_problem,
    _build_interface_mass_matrix,
    _build_interface_restriction_matrix,
    _build_solid_problem,
    _guess_callback_from_snapshots,
    _interface_load_from_traction,
    _interface_traction_from_load,
    _iqnils_next_iterate,
    _restore_function_values,
    _snapshot_function_values,
)

from .test_nirb_double_flap_problem import _geometry


def test_local_driver_partitioned_boundary_extraction(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / "double_flap_partitioned.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)

    mesh_f = mesh_from_gmsh(mesh_path, surface_physical_names=["fluid"], apply_boundary_tags=True)
    mesh_s = mesh_from_gmsh(mesh_path, surface_physical_names=["solid"], apply_boundary_tags=True)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1)
    solid = _build_solid_problem(mesh_s, poly_order=1)

    fluid_iface_coords, fluid_iface_ids = _boundary_field_data(fluid["dh"], "ux", geometry.interface_tag)
    solid_iface_coords, solid_iface_ids = _boundary_field_data(solid["dh"], "dx", geometry.interface_tag)
    clamp_coords, clamp_ids = _boundary_field_data(solid["dh"], "dx", geometry.clamp_tag)
    restriction = _build_interface_restriction_matrix(solid["dh"], solid["d_k"], geometry.interface_tag)

    assert fluid_iface_coords.shape[0] > 0
    assert solid_iface_coords.shape[0] > 0
    assert fluid_iface_ids.shape[0] == fluid_iface_coords.shape[0]
    assert solid_iface_ids.shape[0] == solid_iface_coords.shape[0]
    assert clamp_coords.shape[0] > 0
    assert clamp_ids.shape[0] == clamp_coords.shape[0]
    assert restriction.shape[0] == 2 * solid_iface_coords.shape[0]
    assert restriction.shape[1] == solid["d_k"].nodal_values.size
    assert "a_prev" in fluid
    assert fluid["a_prev"].nodal_values.shape == fluid["u_prev"].nodal_values.shape


def test_bossak_coefficients_match_kratos_defaults() -> None:
    coeffs = _bossak_coefficients(alpha=-0.3, dt=0.008)
    assert coeffs["gamma"] == pytest.approx(0.8)
    assert coeffs["beta"] == pytest.approx(0.4225)
    assert coeffs["ma0"] == pytest.approx(156.25)
    assert coeffs["ma2"] == pytest.approx(-0.25)
    assert coeffs["mam"] == pytest.approx(203.125)


def test_aitken_relaxation_is_clipped() -> None:
    omega = _aitken_relaxation_factor(
        omega_prev=0.5,
        residual_prev=np.array([1.0, 0.0]),
        residual_curr=np.array([0.2, 0.0]),
        omega_min=1.0e-3,
        omega_max=1.0,
    )
    assert 1.0e-3 <= omega <= 1.0


def test_iqnils_update_falls_back_to_relaxed_picard_without_history() -> None:
    x_curr = np.array([[1.0, 2.0]])
    g_curr = np.array([[3.0, 6.0]])
    next_x = _iqnils_next_iterate(
        x_curr=x_curr,
        g_curr=g_curr,
        x_history=[x_curr],
        g_history=[g_curr],
        omega=0.5,
        horizon=3,
        regularization=1.0e-10,
    )
    assert np.allclose(next_x, np.array([[2.0, 4.0]]))


def test_iqnils_update_reuses_old_matrices_on_first_iteration() -> None:
    x_curr = np.array([[1.0, 2.0]])
    g_curr = np.array([[3.0, 6.0]])
    next_x = _iqnils_next_iterate(
        x_curr=x_curr,
        g_curr=g_curr,
        x_history=[x_curr],
        g_history=[g_curr],
        dr_old_mats=[np.eye(2)],
        dg_old_mats=[2.0 * np.eye(2)],
        omega=0.5,
        horizon=3,
        regularization=1.0e-10,
    )
    assert np.allclose(next_x, np.array([[-1.0, -2.0]]))


def test_state_snapshot_restore_preserves_prev_time_level(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / "double_flap_restore.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)
    mesh_s = mesh_from_gmsh(mesh_path, surface_physical_names=["solid"], apply_boundary_tags=True)
    solid = _build_solid_problem(mesh_s, poly_order=1)

    solid["d_k"].nodal_values[:] = np.linspace(0.1, 1.0, solid["d_k"].nodal_values.size)
    solid["d_prev"].nodal_values[:] = np.linspace(-0.5, 0.5, solid["d_prev"].nodal_values.size)

    current_snapshot = _snapshot_function_values([solid["d_k"]])
    prev_snapshot = _snapshot_function_values([solid["d_prev"]])

    # Mimic solve_time_interval's current -> previous promotion and verify that
    # the driver helpers recover both the warm-start guess and the frozen
    # previous-time-step state for the next coupling iteration.
    solid["d_prev"].nodal_values[:] = solid["d_k"].nodal_values[:]
    solid["d_k"].nodal_values.fill(0.0)

    _guess_callback_from_snapshots(current_snapshot)(functions=[solid["d_k"]])
    _restore_function_values([solid["d_prev"]], prev_snapshot)

    assert np.allclose(solid["d_k"].nodal_values, current_snapshot[0])
    assert np.allclose(solid["d_prev"].nodal_values, prev_snapshot[0])


def test_interface_load_conversion_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / "double_flap_interface_mass.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)
    mesh_s = mesh_from_gmsh(mesh_path, surface_physical_names=["solid"], apply_boundary_tags=True)
    solid = _build_solid_problem(mesh_s, poly_order=1)
    solid_iface_coords, _ = _boundary_field_data(solid["dh"], "dx", geometry.interface_tag)

    mass = _build_interface_mass_matrix(mesh_s, solid_iface_coords, geometry.interface_tag)
    traction = np.column_stack(
        [
            np.linspace(0.1, 0.3, solid_iface_coords.shape[0]),
            np.linspace(-0.2, 0.2, solid_iface_coords.shape[0]),
        ]
    )
    loads = _interface_load_from_traction(mass, traction)
    traction_recovered = _interface_traction_from_load(mass, loads)

    assert mass.shape == (solid_iface_coords.shape[0], solid_iface_coords.shape[0])
    assert np.all(np.diag(mass) > 0.0)
    assert np.allclose(traction_recovered, traction)
