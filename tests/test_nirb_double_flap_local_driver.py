from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.jit_parametrization import build_jit_parametrization
from pycutfem.operators import LocalAssemblyResult
from pycutfem.utils.gmsh_loader import mesh_from_gmsh

from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.dvms import (
    FluidDVMSAddVelocityLocalOperator,
    FluidDVMSCondensedLocalSystemOperator,
    FluidDVMSLocalVelocityContributionOperator,
    assemble_dvms_add_mass_lhs_p1_tri,
    assemble_dvms_add_mass_stabilization_p1_tri,
    assemble_dvms_add_velocity_system_p1_tri,
    assemble_dvms_calculate_local_system_p1_tri,
    assemble_dvms_calculate_local_velocity_contribution_p1_tri,
    _update_fluid_dvms_predicted_subscale,
    _kratos_dvms_current_element_size_array,
    _kratos_dvms_element_size,
)
from examples.NIRB.fluid_fom_operator import (
    FluidBoundaryTags,
    FluidFOMOperator,
    FluidFOMParameters,
)
from examples.NIRB.fluid_basis import enrich_fluid_pod_trial_basis_with_supremizers, fit_fluid_pod_trial_basis
from examples.NIRB.fluid_gnat import (
    FluidGNATSolver,
    collect_fluid_residual_snapshots,
    fit_fluid_gnat_sample_set,
    fluid_elements_touching_rows,
)
from examples.NIRB.fluid_lspg import (
    FluidLSPGVerifier,
    FluidTrialSpace,
    pack_fluid_state,
    write_fluid_state,
)
from examples.NIRB.fluid_mode_selection import run_fluid_mode_cross_validation
from examples.NIRB.fluid_snapshots import (
    FluidStageSnapshotBatch,
    FluidStageSnapshotWriter,
    restore_fluid_stage,
)
from examples.NIRB.dvms.local_operator import (
    _compress_batch_to_fluid_block,
    _dvms_condensed_hidden_state_correction_batch,
    assemble_fluid_dvms_local_contribution_batch,
)
from examples.NIRB.example2_problem import build_conforming_mesh
from examples.NIRB.run_example2_local import (
    CoordinateLookup,
    _aitken_relaxation_factor,
    _assemble_fluid_local_velocity_contribution_raw,
    _boundary_field_data,
    _bossak_coefficients,
    _bossak_displacement_kinematics_values,
    _advance_coupling_load_guess,
    _fluid_dvms_summary,
    _fluid_zero_local_operator_forms,
    _fluid_interface_reaction_loads,
    _build_fluid_problem,
    _build_interface_mass_matrix,
    _build_interface_restriction_matrix,
    _build_solid_problem,
    _fluid_residual_and_jacobian,
    _guess_callback_from_snapshots,
    _reference_interface_point_loads_from_lookup,
    _interface_load_from_traction,
    _interface_traction_from_load,
    _iqnils_iteration_matrices,
    _iqnils_next_iterate,
    _load_reference_partitioned_meshes,
    _mesh_extension_equation,
    _negate_lookup,
    _resample_lookup_to_coords,
    _relative_change,
    _restore_function_values,
    _solid_residual_and_jacobian,
    _snapshot_function_values,
    _vector_point_data_from_function,
    _update_fluid_dvms_state_from_previous_step,
)
from pycutfem.solvers.nonlinear_solver import NewtonSolver

from .test_nirb_double_flap_problem import _geometry


def _build_small_fluid_fom_operator(tmp_path: Path, *, backend: str = "python"):
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / f"double_flap_fluid_fom_operator_{backend}.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)
    mesh_f = mesh_from_gmsh(mesh_path, surface_physical_names=["fluid"], apply_boundary_tags=True)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=6)
    operator = FluidFOMOperator(
        prob=fluid,
        mesh=mesh_f,
        parameters=FluidFOMParameters(
            rho_f=1.0,
            mu_f=1.0,
            dt=1.0,
            quadrature_order=6,
            bossak_alpha=-0.3,
            dynamic_tau=1.0,
            backend=backend,
        ),
        boundary_tags=FluidBoundaryTags(
            interface_tag=geometry.interface_tag,
            outlet_tag=geometry.outlet_tag,
            walls_tag=geometry.walls_tag,
            cylinder_tag=geometry.cylinder_tag,
        ),
    )
    return geometry, mesh_f, fluid, operator


def _fluid_row_dofs_for_elements(operator: FluidFOMOperator, element_ids: np.ndarray) -> np.ndarray:
    dh = operator.dh
    eids = np.asarray(element_ids, dtype=int).reshape(-1)
    rows = np.concatenate(
        [
            np.asarray(dh.element_maps["ux"], dtype=int)[eids].reshape(-1),
            np.asarray(dh.element_maps["uy"], dtype=int)[eids].reshape(-1),
            np.asarray(dh.element_maps["p"], dtype=int)[eids].reshape(-1),
        ]
    )
    rows = np.unique(rows[rows >= 0]).astype(int, copy=False)
    return np.intersect1d(rows, operator.free_fluid_dofs()).astype(int, copy=False)


def test_fluid_fom_operator_configures_boundary_state_and_free_dofs(tmp_path: Path) -> None:
    geometry, _mesh_f, fluid, operator = _build_small_fluid_fom_operator(tmp_path)

    iface_coords, ux_ids = _boundary_field_data(fluid["dh"], "ux", geometry.interface_tag)
    _, uy_ids = _boundary_field_data(fluid["dh"], "uy", geometry.interface_tag)
    iface_values = np.column_stack(
        [
            np.full(int(iface_coords.shape[0]), 0.125, dtype=float),
            np.full(int(iface_coords.shape[0]), -0.25, dtype=float),
        ]
    )
    operator.configure_boundary_conditions(
        iface_velocity=CoordinateLookup(iface_coords, iface_values, dim=2),
        inlet_lookup=lambda x, y: 0.5 * float(y),
        apply_to_state=True,
    )

    ux = fluid["u_k"].components[0].get_nodal_values(ux_ids)
    uy = fluid["u_k"].components[1].get_nodal_values(uy_ids)
    free = operator.free_fluid_dofs()
    fixed = np.fromiter(
        (int(gdof) for gdof in fluid["dh"].get_dirichlet_data(fluid["_current_bcs_homog"]).keys()),
        dtype=int,
    )

    np.testing.assert_allclose(ux, 0.125)
    np.testing.assert_allclose(uy, -0.25)
    assert free.size > 0
    assert fixed.size > 0
    assert np.intersect1d(free, fixed).size == 0


def test_fluid_fom_operator_assembly_matches_legacy_raw_helper(tmp_path: Path) -> None:
    _geometry_obj, _mesh_f, fluid, operator = _build_small_fluid_fom_operator(tmp_path)
    element_ids = np.asarray([0, 1], dtype=int)

    assembly = operator.assemble(
        need_matrix=True,
        element_ids=element_ids,
        convention="kratos_rhs",
        refresh_predicted=False,
    )
    expected_matrix, expected_rhs = _assemble_fluid_local_velocity_contribution_raw(
        prob=fluid,
        rho_f=1.0,
        mu_f=1.0,
        dt=1.0,
        quad_order=6,
        bossak_alpha=-0.3,
        need_matrix=True,
        contribution_mode="system",
        backend="python",
        element_ids=element_ids,
    )
    newton_assembly = operator.assemble(
        need_matrix=False,
        element_ids=element_ids,
        convention="newton",
        refresh_predicted=False,
    )

    assert assembly.matrix is not None
    assert expected_matrix is not None
    matrix_diff = (assembly.matrix - expected_matrix).tocoo()
    max_matrix_diff = float(np.max(np.abs(matrix_diff.data))) if matrix_diff.nnz else 0.0
    assert max_matrix_diff == pytest.approx(0.0, abs=1.0e-12)
    np.testing.assert_allclose(assembly.residual, expected_rhs, atol=1.0e-12)
    np.testing.assert_allclose(newton_assembly.residual, -np.asarray(expected_rhs), atol=1.0e-12)


def test_fluid_fom_operator_reaction_matches_legacy_helper(tmp_path: Path) -> None:
    geometry, _mesh_f, fluid, operator = _build_small_fluid_fom_operator(tmp_path)
    iface_coords, _ = _boundary_field_data(fluid["dh"], "ux", geometry.interface_tag)
    zero_iface = CoordinateLookup(
        iface_coords,
        np.zeros((int(iface_coords.shape[0]), 2), dtype=float),
        dim=2,
    )
    operator.configure_boundary_conditions(
        iface_velocity=zero_iface,
        inlet_lookup=lambda x, y: 0.0,
        apply_to_state=True,
    )

    reaction = operator.reaction_loads(refresh_state=False)
    expected = _fluid_interface_reaction_loads(
        prob=fluid,
        rho_f=1.0,
        mu_f=1.0,
        dt=1.0,
        quad_order=6,
        bossak_alpha=-0.3,
        dynamic_tau=1.0,
        interface_tag=geometry.interface_tag,
        backend="python",
        contribution_mode="system",
        refresh_state=False,
    )

    np.testing.assert_allclose(reaction.coords, expected.coords)
    np.testing.assert_allclose(reaction.values, expected.values)


def test_fluid_fom_operator_snapshots_and_restores_dvms_history(tmp_path: Path) -> None:
    _geometry_obj, _mesh_f, fluid, operator = _build_small_fluid_fom_operator(tmp_path)
    state = fluid["dvms_state"]
    state.predicted_subscale_velocity[:, 0] = 0.5
    state.old_mass_residual[:] = -0.25
    state.sync_coefficients_from_samples()

    snapshot = operator.snapshot_history()
    state.predicted_subscale_velocity[:, :] = 3.0
    state.old_mass_residual[:] = 4.0
    state.sync_coefficients_from_samples()
    operator.restore_history(snapshot)

    np.testing.assert_allclose(state.predicted_subscale_velocity[:, 0], 0.5)
    np.testing.assert_allclose(state.predicted_subscale_velocity[:, 1], 0.0)
    np.testing.assert_allclose(state.old_mass_residual, -0.25)


def test_fluid_lspg_verifier_projects_exact_operator_rows(tmp_path: Path) -> None:
    geometry, _mesh_f, _fluid, operator = _build_small_fluid_fom_operator(tmp_path)
    iface_coords, _ = _boundary_field_data(operator.dh, "ux", geometry.interface_tag)
    operator.configure_boundary_conditions(
        iface_velocity=CoordinateLookup(
            iface_coords,
            np.zeros((int(iface_coords.shape[0]), 2), dtype=float),
            dim=2,
        ),
        inlet_lookup=lambda x, y: 0.0,
        apply_to_state=True,
    )

    element_ids = np.asarray([0, 1, 2], dtype=int)
    row_dofs = _fluid_row_dofs_for_elements(operator, element_ids)
    assert row_dofs.size >= 3
    n_modes = 3
    free_basis = np.eye(int(row_dofs.size), n_modes, dtype=float)
    if row_dofs.size > n_modes:
        free_basis[n_modes:, :] = 0.01 * np.outer(
            np.arange(1, int(row_dofs.size) - n_modes + 1, dtype=float),
            np.arange(1, n_modes + 1, dtype=float),
        )
    trial_space = FluidTrialSpace.from_free_basis(
        operator=operator,
        free_basis=free_basis,
        free_dofs=row_dofs,
    )
    coefficients = np.asarray([0.02, -0.01, 0.015], dtype=float)
    reconstructed = trial_space.write(operator, coefficients)
    np.testing.assert_allclose(pack_fluid_state(operator), reconstructed)

    verifier = FluidLSPGVerifier(operator=operator, trial_space=trial_space, row_dofs=row_dofs)
    system = verifier.assemble_system(
        coefficients,
        element_ids=element_ids,
        refresh_predicted=False,
    )
    direct = operator.assemble(
        need_matrix=True,
        element_ids=element_ids,
        convention="newton",
        refresh_predicted=False,
    )
    assert direct.matrix is not None
    expected_residual = np.asarray(direct.residual[row_dofs], dtype=float)
    expected_trial_jacobian = np.asarray(direct.matrix[row_dofs, :] @ trial_space.basis, dtype=float)

    np.testing.assert_allclose(system.residual, expected_residual, atol=1.0e-12)
    np.testing.assert_allclose(system.trial_jacobian, expected_trial_jacobian, atol=1.0e-12)
    np.testing.assert_allclose(system.normal_matrix, expected_trial_jacobian.T @ expected_trial_jacobian)
    np.testing.assert_allclose(system.normal_rhs, -(expected_trial_jacobian.T @ expected_residual))
    expected_step, *_ = np.linalg.lstsq(expected_trial_jacobian, -expected_residual, rcond=None)
    np.testing.assert_allclose(system.gauss_newton_step(), expected_step)

    weights = np.linspace(0.5, 1.5, int(row_dofs.size), dtype=float)
    weighted_system = FluidLSPGVerifier(
        operator=operator,
        trial_space=trial_space,
        row_dofs=row_dofs,
        row_weights=weights,
    ).assemble_system(
        coefficients,
        element_ids=element_ids,
        refresh_predicted=False,
    )
    scale = np.sqrt(weights)
    expected_weighted_residual = expected_residual * scale
    expected_weighted_trial = expected_trial_jacobian * scale[:, None]
    expected_weighted_step, *_ = np.linalg.lstsq(
        expected_weighted_trial,
        -expected_weighted_residual,
        rcond=None,
    )
    np.testing.assert_allclose(weighted_system.weighted_residual, expected_weighted_residual)
    np.testing.assert_allclose(weighted_system.weighted_trial_jacobian, expected_weighted_trial)
    np.testing.assert_allclose(weighted_system.gauss_newton_step(), expected_weighted_step)


def test_fluid_stage_snapshots_round_trip_and_build_mixed_pod_basis(tmp_path: Path) -> None:
    geometry, _mesh_f, fluid, operator = _build_small_fluid_fom_operator(tmp_path)
    iface_coords, _ = _boundary_field_data(operator.dh, "ux", geometry.interface_tag)
    operator.configure_boundary_conditions(
        iface_velocity=CoordinateLookup(
            iface_coords,
            np.zeros((int(iface_coords.shape[0]), 2), dtype=float),
            dim=2,
        ),
        inlet_lookup=lambda x, y: 0.0,
        apply_to_state=True,
    )
    base_state = pack_fluid_state(operator)
    velocity_dofs = np.intersect1d(
        np.concatenate(
            [
                np.asarray(operator.dh.get_field_slice("ux"), dtype=int),
                np.asarray(operator.dh.get_field_slice("uy"), dtype=int),
            ]
        ),
        operator.free_fluid_dofs(),
    )
    pressure_dofs = np.intersect1d(
        np.asarray(operator.dh.get_field_slice("p"), dtype=int),
        operator.free_fluid_dofs(),
    )
    assert velocity_dofs.size >= 4
    assert pressure_dofs.size >= 3

    velocity_pattern = np.zeros_like(base_state)
    pressure_pattern = np.zeros_like(base_state)
    velocity_pattern[velocity_dofs[:4]] = np.asarray([0.05, -0.02, 0.03, -0.04], dtype=float)
    pressure_pattern[pressure_dofs[:3]] = np.asarray([0.1, -0.08, 0.04], dtype=float)
    writer = FluidStageSnapshotWriter()
    for stage, scale in enumerate((0.0, 1.0, 2.0)):
        state = base_state + float(scale) * (velocity_pattern + pressure_pattern)
        write_fluid_state(operator, state)
        fluid["a_k"].nodal_values[:] = float(scale)
        fluid["dvms_state"].old_mass_residual[:] = 0.25 * float(stage + 1)
        fluid["dvms_state"].predicted_subscale_velocity[:, 0] = 0.1 * float(stage)
        fluid["dvms_state"].sync_coefficients_from_samples()
        writer.append_from_operator(
            operator,
            reaction_loads=CoordinateLookup(
                iface_coords,
                np.full((int(iface_coords.shape[0]), 2), float(stage), dtype=float),
                dim=2,
            ),
            metadata={"stage": int(stage), "scale": float(scale)},
        )

    batch = writer.to_batch()
    snapshot_path = tmp_path / "fluid_stage_snapshots.npz"
    batch.save(snapshot_path)
    loaded = FluidStageSnapshotBatch.load(snapshot_path)
    assert loaded.n_snapshots == 3
    np.testing.assert_array_equal(loaded.free_dofs, batch.free_dofs)
    np.testing.assert_array_equal(loaded.fixed_dofs, batch.fixed_dofs)
    np.testing.assert_allclose(loaded.state, batch.state)
    np.testing.assert_allclose(loaded.reaction_coords, iface_coords)
    assert loaded.metadata[2]["stage"] == 2
    subset = loaded.subset([0, 2])
    assert subset.n_snapshots == 2
    np.testing.assert_allclose(subset.state[:, 1], loaded.state[:, 2])

    write_fluid_state(operator, np.zeros(int(operator.dh.total_dofs), dtype=float))
    fluid["dvms_state"].old_mass_residual[:] = -10.0
    fluid["dvms_state"].sync_coefficients_from_samples()
    restore_fluid_stage(operator, loaded.record(1))
    np.testing.assert_allclose(pack_fluid_state(operator), loaded.state[:, 1])
    np.testing.assert_allclose(fluid["dvms_state"].old_mass_residual, 0.5)

    basis = fit_fluid_pod_trial_basis(
        operator,
        loaded,
        velocity_modes=1,
        pressure_modes=1,
        center=False,
    )
    assert basis.n_velocity_modes == 1
    assert basis.n_pressure_modes == 1
    np.testing.assert_allclose(basis.basis[loaded.fixed_dofs, :], 0.0)
    outside_blocks = np.setdiff1d(
        np.arange(int(operator.dh.total_dofs), dtype=int),
        np.union1d(basis.velocity_dofs, basis.pressure_dofs),
    )
    np.testing.assert_allclose(basis.basis[outside_blocks, :], 0.0)
    coefficients = basis.project_state(loaded.state[:, 2], offset=base_state)
    reconstructed = basis.reconstruct_state(coefficients, offset=base_state)
    np.testing.assert_allclose(reconstructed, loaded.state[:, 2], atol=1.0e-12)
    trial_space = basis.make_trial_space(operator, offset=base_state)
    np.testing.assert_allclose(trial_space.reconstruct(coefficients), reconstructed)

    restore_fluid_stage(operator, loaded.record(2))
    stage_coefficients = basis.project_state(pack_fluid_state(operator), offset=base_state)
    stage_trial_space = basis.make_trial_space(operator, offset=base_state)
    replay_element_ids = np.asarray([0, 1, 2], dtype=int)
    replay_row_dofs = _fluid_row_dofs_for_elements(operator, replay_element_ids)
    assert replay_row_dofs.size > 0
    replay_system = FluidLSPGVerifier(
        operator=operator,
        trial_space=stage_trial_space,
        row_dofs=replay_row_dofs,
    ).assemble_system(
        stage_coefficients,
        element_ids=replay_element_ids,
        refresh_predicted=False,
    )
    direct = operator.assemble(
        need_matrix=True,
        element_ids=replay_element_ids,
        convention="newton",
        refresh_predicted=False,
    )
    assert direct.matrix is not None
    np.testing.assert_allclose(replay_system.residual, direct.residual[replay_row_dofs])
    np.testing.assert_allclose(
        replay_system.trial_jacobian,
        np.asarray(direct.matrix[replay_row_dofs, :] @ stage_trial_space.basis, dtype=float),
    )

    actual_reaction = operator.reaction_loads(refresh_state=False)
    reaction_writer = FluidStageSnapshotWriter()
    reaction_writer.append_from_operator(
        operator,
        reaction_loads=actual_reaction,
        metadata={"stage": "reaction-parity"},
    )
    reaction_batch = reaction_writer.to_batch()
    restore_fluid_stage(operator, reaction_batch.record(0))
    replayed_reaction = operator.reaction_loads(refresh_state=False)
    np.testing.assert_allclose(replayed_reaction.coords, reaction_batch.reaction_coords)
    np.testing.assert_allclose(
        replayed_reaction.values,
        reaction_batch.reaction_values[:, 0].reshape(-1, 2),
    )

    mode_sweep = run_fluid_mode_cross_validation(
        operator,
        loaded,
        velocity_modes=[0, 1],
        pressure_modes=[0, 1],
        test_indices=[2],
        center=True,
        include_reaction_error=False,
    )
    selected = mode_sweep.best(plateau_rel_tol=0.0)
    assert selected.velocity_modes == 1
    assert selected.pressure_modes == 1
    assert selected.state_error <= 1.0e-12


def test_fluid_pod_supremizer_enrichment_keeps_homogeneous_rows(tmp_path: Path) -> None:
    geometry, _mesh_f, fluid, operator = _build_small_fluid_fom_operator(tmp_path)
    iface_coords, _ = _boundary_field_data(operator.dh, "ux", geometry.interface_tag)
    operator.configure_boundary_conditions(
        iface_velocity=CoordinateLookup(
            iface_coords,
            np.zeros((int(iface_coords.shape[0]), 2), dtype=float),
            dim=2,
        ),
        inlet_lookup=lambda x, y: 0.0,
        apply_to_state=True,
    )
    base_state = pack_fluid_state(operator)
    velocity_rows = operator.free_fluid_dofs(("ux", "uy"))
    pressure_rows = operator.free_fluid_dofs(("p",))
    assert velocity_rows.size >= 4
    assert pressure_rows.size >= 3

    writer = FluidStageSnapshotWriter()
    for stage, scale in enumerate((0.0, 0.5, 1.0, 1.5)):
        state = base_state.copy()
        state[velocity_rows[:4]] += float(scale) * np.asarray([0.03, -0.02, 0.015, -0.01], dtype=float)
        state[pressure_rows[:3]] += float(scale) * np.asarray([0.09, -0.04, 0.025], dtype=float)
        write_fluid_state(operator, state)
        writer.append_from_operator(operator, metadata={"stage": int(stage)})

    snapshots = writer.to_batch()
    basis = fit_fluid_pod_trial_basis(
        operator,
        snapshots,
        velocity_modes=1,
        pressure_modes=1,
        center=False,
    )
    enriched = enrich_fluid_pod_trial_basis_with_supremizers(
        operator,
        basis,
        supremizer_modes=1,
        riesz="h1",
        backend="python",
    )

    assert enriched.n_velocity_pod_modes == 1
    assert enriched.n_velocity_supremizer_modes == 1
    assert enriched.n_velocity_modes == 2
    assert enriched.n_pressure_modes == 1
    assert enriched.n_modes == 3
    np.testing.assert_allclose(enriched.basis[snapshots.fixed_dofs, :], 0.0, atol=1.0e-13)
    np.testing.assert_allclose(enriched.basis[:, -1], basis.basis[:, basis.n_velocity_modes])


def test_fluid_gnat_sample_set_projects_sampled_operator(tmp_path: Path) -> None:
    geometry, _mesh_f, fluid, operator = _build_small_fluid_fom_operator(tmp_path)
    iface_coords, _ = _boundary_field_data(operator.dh, "ux", geometry.interface_tag)
    operator.configure_boundary_conditions(
        iface_velocity=CoordinateLookup(
            iface_coords,
            np.zeros((int(iface_coords.shape[0]), 2), dtype=float),
            dim=2,
        ),
        inlet_lookup=lambda x, y: 0.0,
        apply_to_state=True,
    )
    base_state = pack_fluid_state(operator)
    training_element_ids = np.asarray([0, 1, 2, 3], dtype=int)
    training_rows = _fluid_row_dofs_for_elements(operator, training_element_ids)
    velocity_rows = np.intersect1d(
        training_rows,
        np.concatenate(
            [
                np.asarray(operator.dh.get_field_slice("ux"), dtype=int),
                np.asarray(operator.dh.get_field_slice("uy"), dtype=int),
            ]
        ),
    )
    pressure_rows = np.intersect1d(training_rows, np.asarray(operator.dh.get_field_slice("p"), dtype=int))
    assert velocity_rows.size >= 3
    assert pressure_rows.size >= 2

    writer = FluidStageSnapshotWriter()
    for stage, scale in enumerate((0.0, 0.5, 1.0, 1.5)):
        state = base_state.copy()
        state[velocity_rows[:3]] += float(scale) * np.asarray([0.04, -0.03, 0.02], dtype=float)
        state[pressure_rows[:2]] += float(scale) * np.asarray([0.08, -0.05], dtype=float)
        write_fluid_state(operator, state)
        fluid["a_k"].nodal_values[:] = 0.1 * float(stage)
        fluid["dvms_state"].old_mass_residual[:] = 0.05 * float(stage + 1)
        fluid["dvms_state"].sync_coefficients_from_samples()
        writer.append_from_operator(operator, metadata={"stage": int(stage)})
    snapshots = writer.to_batch()
    trial_basis = fit_fluid_pod_trial_basis(
        operator,
        snapshots,
        velocity_modes=1,
        pressure_modes=1,
        center=False,
    )
    residual_snapshots = collect_fluid_residual_snapshots(
        operator,
        snapshots,
        row_dofs=training_rows,
        element_ids=training_element_ids,
        refresh_predicted=False,
    )
    sample_set = fit_fluid_gnat_sample_set(
        operator,
        residual_snapshots,
        basis_dofs=training_rows,
        residual_modes=2,
        row_oversampling=2.0,
        include_interface_elements=False,
    )
    assert sample_set.n_residual_modes == 2
    assert sample_set.n_sample_rows >= sample_set.n_residual_modes
    assert sample_set.sampled_basis_rank == sample_set.n_residual_modes
    np.testing.assert_array_equal(
        np.sort(sample_set.element_ids),
        np.sort(fluid_elements_touching_rows(operator, sample_set.row_dofs)),
    )

    restore_fluid_stage(operator, snapshots.record(3))
    offset = snapshots.state[:, 0]
    coefficients = trial_basis.project_state(pack_fluid_state(operator), offset=offset)
    trial_space = trial_basis.make_trial_space(operator, offset=offset)
    gnat = FluidGNATSolver(operator=operator, trial_space=trial_space, sample_set=sample_set)
    system = gnat.assemble_system(coefficients, refresh_predicted=False)
    direct = operator.assemble(
        need_matrix=True,
        element_ids=sample_set.element_ids,
        convention="newton",
        refresh_predicted=False,
    )
    assert direct.matrix is not None
    expected_sampled_residual = np.asarray(direct.residual[sample_set.row_dofs], dtype=float)
    expected_sampled_trial = np.asarray(direct.matrix[sample_set.row_dofs, :] @ trial_space.basis, dtype=float)
    expected_coefficients = sample_set.sample_to_residual_coefficients @ expected_sampled_residual
    expected_gnat_trial = sample_set.sample_to_residual_coefficients @ expected_sampled_trial
    np.testing.assert_allclose(system.sampled_residual, expected_sampled_residual)
    np.testing.assert_allclose(system.sampled_trial_jacobian, expected_sampled_trial)
    np.testing.assert_allclose(system.residual_coefficients, expected_coefficients)
    np.testing.assert_allclose(system.gnat_trial_jacobian, expected_gnat_trial)
    np.testing.assert_allclose(system.normal_matrix, expected_gnat_trial.T @ expected_gnat_trial)
    np.testing.assert_allclose(system.normal_rhs, -(expected_gnat_trial.T @ expected_coefficients))

    weighted_sample_set = fit_fluid_gnat_sample_set(
        operator,
        residual_snapshots,
        basis_dofs=training_rows,
        residual_modes=2,
        row_oversampling=2.0,
        include_interface_elements=False,
        sample_weighting="snapshot-gram",
    )
    assert weighted_sample_set.sample_weighting == "snapshot-gram"
    assert weighted_sample_set.sample_weights.shape == (weighted_sample_set.n_sample_rows,)
    assert np.all(weighted_sample_set.sample_weights >= 0.0)
    weighted_gnat = FluidGNATSolver(
        operator=operator,
        trial_space=trial_space,
        sample_set=weighted_sample_set,
        objective="sampled_lspg",
    )
    weighted_system = weighted_gnat.assemble_system(coefficients, refresh_predicted=False)
    weighted_direct = operator.assemble(
        need_matrix=True,
        element_ids=weighted_sample_set.element_ids,
        convention="newton",
        refresh_predicted=False,
    )
    assert weighted_direct.matrix is not None
    scale = np.sqrt(np.maximum(weighted_sample_set.sample_weights, 0.0))
    expected_weighted_residual = scale * np.asarray(weighted_direct.residual[weighted_sample_set.row_dofs], dtype=float)
    expected_weighted_trial = (
        np.asarray(weighted_direct.matrix[weighted_sample_set.row_dofs, :] @ trial_space.basis, dtype=float)
        * scale[:, None]
    )
    np.testing.assert_allclose(weighted_system.residual_coefficients, expected_weighted_residual)
    np.testing.assert_allclose(weighted_system.gnat_trial_jacobian, expected_weighted_trial)
    np.testing.assert_allclose(weighted_system.normal_matrix, expected_weighted_trial.T @ expected_weighted_trial)
    np.testing.assert_allclose(weighted_system.normal_rhs, -(expected_weighted_trial.T @ expected_weighted_residual))

    row_weights = np.linspace(0.25, 2.0, int(weighted_sample_set.n_sample_rows), dtype=float)
    block_weighted_gnat = FluidGNATSolver(
        operator=operator,
        trial_space=trial_space,
        sample_set=weighted_sample_set,
        objective="sampled_lspg",
        row_weights=row_weights,
    )
    block_weighted_system = block_weighted_gnat.assemble_system(coefficients, refresh_predicted=False)
    block_scale = np.sqrt(np.maximum(weighted_sample_set.sample_weights, 0.0) * row_weights)
    expected_block_weighted_residual = block_scale * np.asarray(
        weighted_direct.residual[weighted_sample_set.row_dofs],
        dtype=float,
    )
    expected_block_weighted_trial = (
        np.asarray(weighted_direct.matrix[weighted_sample_set.row_dofs, :] @ trial_space.basis, dtype=float)
        * block_scale[:, None]
    )
    np.testing.assert_allclose(block_weighted_system.residual_coefficients, expected_block_weighted_residual)
    np.testing.assert_allclose(block_weighted_system.gnat_trial_jacobian, expected_block_weighted_trial)
    np.testing.assert_allclose(
        block_weighted_system.normal_matrix,
        expected_block_weighted_trial.T @ expected_block_weighted_trial,
    )
    np.testing.assert_allclose(
        block_weighted_system.normal_rhs,
        -(expected_block_weighted_trial.T @ expected_block_weighted_residual),
    )


def test_local_driver_partitioned_boundary_extraction(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / "double_flap_partitioned.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)

    mesh_f = mesh_from_gmsh(mesh_path, surface_physical_names=["fluid"], apply_boundary_tags=True)
    mesh_s = mesh_from_gmsh(mesh_path, surface_physical_names=["solid"], apply_boundary_tags=True)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=6)
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
    assert "dvms_state" in fluid
    assert fluid["dvms_state"].sample_count >= mesh_f.n_elements
    assert fluid["dvms_state"].sample_count % mesh_f.n_elements == 0
    assert fluid["dvms_state"].quadrature_order == 6
    assert fluid["dvms_state"].n_qp_per_element == fluid["dvms_state"].sample_count // mesh_f.n_elements
    assert fluid["dvms_state"].old_mass_residual.shape == (fluid["dvms_state"].sample_count,)
    assert fluid["dvms_state"].registry["example2_local_dvms_old_subscale_velocity"].shape == (
        mesh_f.n_elements,
        fluid["dvms_state"].n_qp_per_element,
        2,
    )
    assert fluid["dvms_state"].registry["example2_local_dvms_old_mass_residual"].shape == (
        mesh_f.n_elements,
        fluid["dvms_state"].n_qp_per_element,
    )


def test_freeze_bcs_keeps_stage_scaled_two_arg_inlet_callback() -> None:
    scale_value = 0.25

    def base_profile(x: float, y: float) -> float:
        del x
        return 2.0 * float(y)

    def scaled_profile(x: float, y: float) -> float:
        return float(scale_value) * float(base_profile(x, y))

    frozen = NewtonSolver._freeze_bcs(
        [BoundaryCondition("ux", "dirichlet", "inlet", scaled_profile)],
        t_now=0.008,
    )
    value = float(frozen[0].value(0.0, 0.4))
    assert value == pytest.approx(scale_value * base_profile(0.0, 0.4))


def test_fluid_dvms_state_tracks_previous_mass_residual_on_zero_state(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / "double_flap_dvms_state.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)
    mesh_f = mesh_from_gmsh(mesh_path, surface_physical_names=["fluid"], apply_boundary_tags=True)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=6)

    _update_fluid_dvms_state_from_previous_step(
        state=fluid["dvms_state"],
        dh=fluid["dh"],
        mesh=mesh_f,
        u_prev=fluid["u_prev"],
        d_prev=fluid["d_prev"],
        d_geo=fluid["d_mesh"],
    )
    summary = _fluid_dvms_summary(fluid)

    assert summary["enabled"] is True
    assert summary["sample_count"] >= mesh_f.n_elements
    assert summary["sample_count"] % mesh_f.n_elements == 0
    assert summary["quadrature_order"] == 6
    assert summary["old_mass_residual_inf_norm"] == pytest.approx(0.0)
    assert summary["old_subscale_velocity_inf_norm"] == pytest.approx(0.0)


def test_predicted_subscale_zero_for_zero_static_residual(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / "double_flap_dvms_predictor_constant_velocity.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)
    mesh_f = mesh_from_gmsh(mesh_path, surface_physical_names=["fluid"], apply_boundary_tags=True)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=6)

    ux_ids = np.asarray(fluid["dh"].get_field_slice("ux"), dtype=int)
    uy_ids = np.asarray(fluid["dh"].get_field_slice("uy"), dtype=int)
    ux_vals = np.full(ux_ids.shape, 0.25, dtype=float)
    uy_vals = np.full(uy_ids.shape, -0.15, dtype=float)
    fluid["u_k"].components[0].set_nodal_values(ux_ids, ux_vals)
    fluid["u_k"].components[1].set_nodal_values(uy_ids, uy_vals)
    fluid["u_prev"].components[0].set_nodal_values(ux_ids, ux_vals)
    fluid["u_prev"].components[1].set_nodal_values(uy_ids, uy_vals)

    _update_fluid_dvms_predicted_subscale(
        state=fluid["dvms_state"],
        dh=fluid["dh"],
        mesh=mesh_f,
        u_k=fluid["u_k"],
        u_prev=fluid["u_prev"],
        a_prev=fluid["a_prev"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        rho_f=1.0,
        mu_f=1.0,
        dt=1.0,
        bossak_alpha=-0.3,
        dynamic_tau=1.0,
        backend="python",
    )

    # For the Kratos predictor, a spatially constant resolved velocity with
    # zero pressure gradient, zero old subscale, and zero current acceleration
    # yields zero static residual.
    assert np.allclose(fluid["dvms_state"].predicted_subscale_velocity, 0.0, atol=1.0e-12)


def test_relative_change_uses_vector_norm_for_relative_part() -> None:
    new = np.asarray([[3.0, 0.0], [0.0, 4.0]], dtype=float)
    old = np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=float)

    abs_norm, rel_norm = _relative_change(new, old)

    expected_abs = float(np.linalg.norm(new.reshape(-1)) / np.sqrt(float(new.size)))
    expected_rel = 1.0

    assert abs_norm == pytest.approx(expected_abs)
    assert rel_norm == pytest.approx(expected_rel)


def test_converged_coupling_iteration_skips_final_iqn_update() -> None:
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    load_guess = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    returned = np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=float)
    guess_history: list[np.ndarray] = []
    return_history: list[np.ndarray] = []

    next_lookup, update_applied = _advance_coupling_load_guess(
        step_converged=True,
        active_force_update="iqnils",
        solid_iface_coords=coords,
        load_guess_vals=load_guess,
        returned_load_vals=returned,
        load_guess_history=guess_history,
        load_return_history=return_history,
        iqn_old_dr_mats=[],
        iqn_old_dg_mats=[],
        omega_force=0.5,
        force_iteration_horizon=50,
        force_regularization=1.0e-10,
    )

    assert update_applied is False
    assert guess_history == []
    assert return_history == []
    assert np.allclose(next_lookup.values, returned)


def test_nonconverged_coupling_iteration_records_iqn_history() -> None:
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    load_guess = np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    returned = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    guess_history: list[np.ndarray] = []
    return_history: list[np.ndarray] = []

    next_lookup, update_applied = _advance_coupling_load_guess(
        step_converged=False,
        active_force_update="iqnils",
        solid_iface_coords=coords,
        load_guess_vals=load_guess,
        returned_load_vals=returned,
        load_guess_history=guess_history,
        load_return_history=return_history,
        iqn_old_dr_mats=[],
        iqn_old_dg_mats=[],
        omega_force=0.5,
        force_iteration_horizon=50,
        force_regularization=1.0e-10,
    )

    assert update_applied is True
    assert len(guess_history) == 1
    assert len(return_history) == 1
    assert np.allclose(guess_history[0], load_guess)
    assert np.allclose(return_history[0], returned)
    assert np.allclose(next_lookup.values, 0.5 * returned)


def test_vector_point_data_from_function_matches_assigned_nodal_values(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / "double_flap_vector_point_data.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)
    mesh_f = mesh_from_gmsh(mesh_path, surface_physical_names=["fluid"], apply_boundary_tags=True)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=6)

    ux_ids = np.asarray(fluid["dh"].get_field_slice("ux"), dtype=int)
    uy_ids = np.asarray(fluid["dh"].get_field_slice("uy"), dtype=int)
    fluid["u_k"].components[0].set_nodal_values(ux_ids, np.full(ux_ids.shape, 0.125, dtype=float))
    fluid["u_k"].components[1].set_nodal_values(uy_ids, np.full(uy_ids.shape, -0.25, dtype=float))

    point_data = _vector_point_data_from_function(fluid["dh"], fluid["u_k"])

    assert point_data.shape == (len(mesh_f.nodes_list), 2)
    assert np.max(np.abs(point_data[:, 0])) == pytest.approx(0.125)
    assert np.max(np.abs(point_data[:, 1])) == pytest.approx(0.25)


def test_local_add_velocity_system_zero_state_is_finite(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / "double_flap_dvms_local_velocity.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)
    mesh_f = mesh_from_gmsh(mesh_path, surface_physical_names=["fluid"], apply_boundary_tags=True)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=6)

    lhs, rhs, gdofs = assemble_dvms_add_velocity_system_p1_tri(
        mesh=mesh_f,
        dh=fluid["dh"],
        eid=0,
        u_k=fluid["u_k"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        state=fluid["dvms_state"],
        rho_f=1.0,
        mu_f=1.0,
        dt=1.0,
        quadrature_order=6,
    )

    assert lhs.shape == (9, 9)
    assert rhs.shape == (9,)
    assert gdofs.shape == (9,)
    assert np.all(np.isfinite(lhs))
    assert np.all(np.isfinite(rhs))
    assert np.allclose(rhs, 0.0, atol=1.0e-12)


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_fluid_zero_local_operator_forms_preserve_fluid_block_layout(tmp_path: Path, backend: str) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / f"double_flap_dvms_zero_forms_{backend}.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)
    mesh_f = mesh_from_gmsh(mesh_path, surface_physical_names=["fluid"], apply_boundary_tags=True)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=6)

    iface_coords, _ = _boundary_field_data(fluid["dh"], "ux", geometry.interface_tag)
    zero_lookup = CoordinateLookup(
        coords=iface_coords,
        values=np.zeros((int(iface_coords.shape[0]), 2), dtype=float),
        dim=2,
    )
    residual, jacobian, _, _ = _fluid_zero_local_operator_forms(
        prob=fluid,
        iface_velocity=zero_lookup,
        inlet_lookup=lambda x, y, t=0.0: 0.0,
        interface_tag=geometry.interface_tag,
        outlet_tag=geometry.outlet_tag,
        walls_tag=geometry.walls_tag,
        cylinder_tag=geometry.cylinder_tag,
        quad_order=6,
    )
    compiler = FormCompiler(fluid["dh"], quadrature_order=6, backend=backend)
    batch = compiler.assemble_volume_local_contributions(
        Equation(jacobian, residual),
        element_ids=np.asarray([0], dtype=int),
    )

    assert batch.K_elem is not None
    assert batch.F_elem is not None
    assert batch.K_elem.shape[0] == 1
    assert batch.F_elem.shape[0] == 1
    assert batch.K_elem.shape[1] > 0
    assert batch.F_elem.shape[1] > 0


def test_kratos_dvms_element_size_matches_minimum_altitude(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / "double_flap_dvms_h.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)
    mesh_f = mesh_from_gmsh(mesh_path, surface_physical_names=["fluid"], apply_boundary_tags=True)

    conn = np.asarray(mesh_f.elements_connectivity[0], dtype=int)
    coords = np.asarray(mesh_f.nodes_x_y_pos[conn], dtype=float)
    area = float(np.asarray(mesh_f.areas_list, dtype=float).reshape(-1)[0])
    edge_lengths = np.asarray(
        [
            np.linalg.norm(coords[0] - coords[1]),
            np.linalg.norm(coords[1] - coords[2]),
            np.linalg.norm(coords[2] - coords[0]),
        ],
        dtype=float,
    )
    h_ref = (2.0 * abs(area)) / float(np.max(edge_lengths))
    h = _kratos_dvms_element_size(mesh_f, 0)

    assert h == pytest.approx(h_ref)


@pytest.mark.parametrize("backend", ["python", "jit", "cpp"])
def test_local_velocity_contribution_operator_matches_direct_element_assembly(tmp_path: Path, backend: str) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / f"double_flap_dvms_local_operator_{backend}.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)
    mesh_f = mesh_from_gmsh(mesh_path, surface_physical_names=["fluid"], apply_boundary_tags=True)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=6)

    lhs, rhs, gdofs = assemble_dvms_calculate_local_velocity_contribution_p1_tri(
        mesh=mesh_f,
        dh=fluid["dh"],
        eid=0,
        u_k=fluid["u_k"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        state=fluid["dvms_state"],
        rho_f=1.0,
        mu_f=1.0,
        dt=1.0,
        quadrature_order=6,
    )

    solver = NewtonSolver.__new__(NewtonSolver)
    solver.backend = str(backend)
    solver.full_to_red = np.arange(int(fluid["dh"].total_dofs), dtype=int)
    op = FluidDVMSLocalVelocityContributionOperator(
        mesh=mesh_f,
        dh=fluid["dh"],
        u_k=fluid["u_k"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        state=fluid["dvms_state"],
        rho_f=1.0,
        mu_f=1.0,
        dt=1.0,
        element_ids=np.asarray([0], dtype=int),
        quadrature_order=6,
    )

    A_red = np.zeros((int(fluid["dh"].total_dofs), int(fluid["dh"].total_dofs)), dtype=float)
    R_red = np.zeros(int(fluid["dh"].total_dofs), dtype=float)
    A_red, R_red = op.after_assembly(
        solver=solver,
        coeffs={},
        A_red=A_red,
        R_red=R_red,
        need_matrix=True,
    )

    A_exp = np.zeros_like(A_red)
    R_exp = np.zeros_like(R_red)
    A_exp[np.ix_(gdofs, gdofs)] += lhs
    R_exp[gdofs] -= rhs

    assert np.allclose(A_red, A_exp)
    assert np.allclose(R_red, R_exp)


def test_local_velocity_contribution_operator_scatter_into_sparse_matrix(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / "double_flap_dvms_local_operator_sparse.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)
    mesh_f = mesh_from_gmsh(mesh_path, surface_physical_names=["fluid"], apply_boundary_tags=True)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=6)

    lhs, rhs, gdofs = assemble_dvms_calculate_local_velocity_contribution_p1_tri(
        mesh=mesh_f,
        dh=fluid["dh"],
        eid=0,
        u_k=fluid["u_k"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        state=fluid["dvms_state"],
        rho_f=1.0,
        mu_f=1.0,
        dt=1.0,
        quadrature_order=6,
    )

    solver = NewtonSolver.__new__(NewtonSolver)
    solver.backend = "cpp"
    solver.full_to_red = np.arange(int(fluid["dh"].total_dofs), dtype=int)
    op = FluidDVMSLocalVelocityContributionOperator(
        mesh=mesh_f,
        dh=fluid["dh"],
        u_k=fluid["u_k"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        state=fluid["dvms_state"],
        rho_f=1.0,
        mu_f=1.0,
        dt=1.0,
        element_ids=np.asarray([0], dtype=int),
        quadrature_order=6,
    )

    A_red = sp.csr_matrix((int(fluid["dh"].total_dofs), int(fluid["dh"].total_dofs)), dtype=float)
    R_red = np.zeros(int(fluid["dh"].total_dofs), dtype=float)
    A_red, R_red = op.after_assembly(
        solver=solver,
        coeffs={},
        A_red=A_red,
        R_red=R_red,
        need_matrix=True,
    )

    assert sp.isspmatrix_csr(A_red)
    assert A_red.nnz > 0
    np.testing.assert_allclose(A_red[np.ix_(gdofs, gdofs)].toarray(), lhs, atol=1.0e-10)
    np.testing.assert_allclose(R_red[gdofs], -rhs, atol=1.0e-10)


def test_bossak_coefficients_match_kratos_defaults() -> None:
    coeffs = _bossak_coefficients(alpha=-0.3, dt=0.008)
    assert coeffs["gamma"] == pytest.approx(0.8)
    assert coeffs["beta"] == pytest.approx(0.4225)
    assert coeffs["ma0"] == pytest.approx(156.25)
    assert coeffs["ma2"] == pytest.approx(-0.25)
    assert coeffs["mam"] == pytest.approx(203.125)


def test_bossak_mesh_velocity_matches_first_step_kratos_kinematics() -> None:
    disp = np.asarray([[2.814130215548585e-05, 0.0]], dtype=float)
    vel, acc = _bossak_displacement_kinematics_values(
        d_curr=disp,
        d_prev=np.zeros_like(disp),
        v_prev=np.zeros_like(disp),
        a_prev=np.zeros_like(disp),
        dt=0.008,
        alpha=-0.3,
    )
    assert float(np.max(np.abs(vel))) == pytest.approx(0.006660663232067657)
    assert float(np.max(np.abs(acc))) == pytest.approx(1.0407286300105714)


def test_local_system_matches_kratos_bossak_combination(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / "double_flap_dvms_local_system.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)
    mesh_f = mesh_from_gmsh(mesh_path, surface_physical_names=["fluid"], apply_boundary_tags=True)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=6)

    # Build a non-trivial current iterate while keeping the first-step history
    # simple enough that the expected Bossak relation can be written directly.
    eid = 0
    ux_ids = np.asarray(fluid["dh"].element_maps["ux"][eid], dtype=int)
    uy_ids = np.asarray(fluid["dh"].element_maps["uy"][eid], dtype=int)
    p_ids = np.asarray(fluid["dh"].element_maps["p"][eid], dtype=int)
    fluid["u_k"].components[0].set_nodal_values(ux_ids, np.asarray([0.2, -0.1, 0.05], dtype=float))
    fluid["u_k"].components[1].set_nodal_values(uy_ids, np.asarray([-0.05, 0.15, 0.1], dtype=float))
    fluid["p_k"].set_nodal_values(p_ids, np.asarray([1.0, -0.25, 0.5], dtype=float))

    velocity_lhs, velocity_rhs, gdofs_v = assemble_dvms_add_velocity_system_p1_tri(
        mesh=mesh_f,
        dh=fluid["dh"],
        eid=eid,
        u_k=fluid["u_k"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        state=fluid["dvms_state"],
        rho_f=1.0,
        mu_f=1.0,
        dt=1.0,
        quadrature_order=6,
    )
    mass_lhs, gdofs_m = assemble_dvms_add_mass_lhs_p1_tri(
        mesh=mesh_f,
        dh=fluid["dh"],
        eid=eid,
        d_mesh=fluid["d_mesh"],
        rho_f=1.0,
        quadrature_order=6,
    )
    mass_stab, gdofs_ms = assemble_dvms_add_mass_stabilization_p1_tri(
        mesh=mesh_f,
        dh=fluid["dh"],
        eid=eid,
        u_k=fluid["u_k"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        state=fluid["dvms_state"],
        rho_f=1.0,
        mu_f=1.0,
        dt=1.0,
        quadrature_order=6,
    )
    system_lhs, system_rhs, gdofs_s = assemble_dvms_calculate_local_system_p1_tri(
        mesh=mesh_f,
        dh=fluid["dh"],
        eid=eid,
        u_k=fluid["u_k"],
        u_prev=fluid["u_prev"],
        a_prev=fluid["a_prev"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        state=fluid["dvms_state"],
        rho_f=1.0,
        mu_f=1.0,
        dt=1.0,
        bossak_alpha=-0.3,
        quadrature_order=6,
        backend="python",
    )

    assert np.array_equal(gdofs_v, gdofs_m)
    assert np.array_equal(gdofs_v, gdofs_ms)
    assert np.array_equal(gdofs_v, gdofs_s)

    coeffs = _bossak_coefficients(alpha=-0.3, dt=1.0)
    a_relaxed = np.zeros_like(system_rhs)
    u_grouped = np.concatenate(
        [
            np.asarray(fluid["u_k"].components[0].get_nodal_values(ux_ids), dtype=float),
            np.asarray(fluid["u_k"].components[1].get_nodal_values(uy_ids), dtype=float),
            np.zeros(3, dtype=float),
        ]
    )
    a_relaxed[:6] = (1.0 - coeffs["alpha"]) * coeffs["ma0"] * u_grouped[:6]

    mass_total = mass_lhs + mass_stab
    lhs_expected = velocity_lhs + coeffs["mam"] * mass_total
    rhs_expected = velocity_rhs - mass_total @ a_relaxed

    assert np.allclose(system_lhs, lhs_expected)
    assert np.allclose(system_rhs, rhs_expected)


def test_condensed_local_system_operator_improves_state_updated_tangent(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / "double_flap_dvms_condensed_local_system.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)
    mesh_f = mesh_from_gmsh(mesh_path, surface_physical_names=["fluid"], apply_boundary_tags=True)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=6)

    eid = 0
    ux_ids = np.asarray(fluid["dh"].element_maps["ux"][eid], dtype=int)
    uy_ids = np.asarray(fluid["dh"].element_maps["uy"][eid], dtype=int)
    p_ids = np.asarray(fluid["dh"].element_maps["p"][eid], dtype=int)
    fluid["u_k"].components[0].set_nodal_values(ux_ids, np.asarray([0.08, -0.03, 0.05], dtype=float))
    fluid["u_k"].components[1].set_nodal_values(uy_ids, np.asarray([-0.02, 0.04, 0.06], dtype=float))
    fluid["p_k"].set_nodal_values(p_ids, np.asarray([0.7, -0.1, 0.2], dtype=float))
    fluid["u_prev"].components[0].set_nodal_values(ux_ids, np.asarray([0.02, -0.01, 0.01], dtype=float))
    fluid["u_prev"].components[1].set_nodal_values(uy_ids, np.asarray([-0.01, 0.01, 0.02], dtype=float))

    solver = NewtonSolver.__new__(NewtonSolver)
    solver.backend = "cpp"
    solver.full_to_red = np.arange(int(fluid["dh"].total_dofs), dtype=int)
    solver._current_bcs = []

    condensed = FluidDVMSCondensedLocalSystemOperator(
        mesh=mesh_f,
        dh=fluid["dh"],
        u_k=fluid["u_k"],
        u_prev=fluid["u_prev"],
        a_prev=fluid["a_prev"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        state=fluid["dvms_state"],
        rho_f=1.0,
        mu_f=1.0,
        dt=1.0,
        bossak_alpha=-0.3,
        dynamic_tau=1.0,
        element_ids=np.asarray([eid], dtype=int),
        quadrature_order=6,
        apply_dirichlet_lift=False,
    )

    def _eval_condensed() -> tuple[np.ndarray, np.ndarray]:
        A_red = np.zeros((int(fluid["dh"].total_dofs), int(fluid["dh"].total_dofs)), dtype=float)
        R_red = np.zeros(int(fluid["dh"].total_dofs), dtype=float)
        A_red, R_red = condensed.after_assembly(
            solver=solver,
            coeffs={},
            A_red=A_red,
            R_red=R_red,
            need_matrix=True,
        )
        local_gdofs = np.concatenate([ux_ids, uy_ids, p_ids])
        return A_red[np.ix_(local_gdofs, local_gdofs)], R_red[local_gdofs].copy()

    K_condensed, r0 = _eval_condensed()
    K_base, _, _ = assemble_dvms_calculate_local_system_p1_tri(
        mesh=mesh_f,
        dh=fluid["dh"],
        eid=eid,
        u_k=fluid["u_k"],
        u_prev=fluid["u_prev"],
        a_prev=fluid["a_prev"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        state=fluid["dvms_state"],
        rho_f=1.0,
        mu_f=1.0,
        dt=1.0,
        bossak_alpha=-0.3,
        quadrature_order=6,
        backend="python",
    )

    local_gdofs = np.concatenate([ux_ids, uy_ids, p_ids])
    fd = np.zeros_like(K_condensed)
    eps = 1.0e-7

    def _perturb_and_eval(col: int, delta: float) -> np.ndarray:
        gdof = int(local_gdofs[col])
        if col < 3:
            field = fluid["u_k"].components[0].nodal_values
        elif col < 6:
            field = fluid["u_k"].components[1].nodal_values
        else:
            field = fluid["p_k"].nodal_values
        field[gdof] += float(delta)
        try:
            _, rval = _eval_condensed()
        finally:
            field[gdof] -= float(delta)
        return rval

    for col in range(int(local_gdofs.shape[0])):
        r_plus = _perturb_and_eval(col, eps)
        r_minus = _perturb_and_eval(col, -eps)
        fd[:, col] = (r_plus - r_minus) / (2.0 * eps)

    err_condensed = float(np.linalg.norm(K_condensed - fd, ord=np.inf))
    err_base = float(np.linalg.norm(K_base - fd, ord=np.inf))

    assert np.all(np.isfinite(K_condensed))
    assert np.all(np.isfinite(fd))
    assert float(np.linalg.norm(K_condensed - K_base, ord=np.inf)) > 0.0
    assert err_condensed <= err_base + 1.0e-3


def test_condensed_symbolic_local_system_matches_split_reference(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / "double_flap_dvms_condensed_symbolic_match.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)
    mesh_f = mesh_from_gmsh(mesh_path, surface_physical_names=["fluid"], apply_boundary_tags=True)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=6)

    eid = 0
    ux_ids = np.asarray(fluid["dh"].element_maps["ux"][eid], dtype=int)
    uy_ids = np.asarray(fluid["dh"].element_maps["uy"][eid], dtype=int)
    p_ids = np.asarray(fluid["dh"].element_maps["p"][eid], dtype=int)
    fluid["u_k"].components[0].set_nodal_values(ux_ids, np.asarray([0.08, -0.03, 0.05], dtype=float))
    fluid["u_k"].components[1].set_nodal_values(uy_ids, np.asarray([-0.02, 0.04, 0.06], dtype=float))
    fluid["p_k"].set_nodal_values(p_ids, np.asarray([0.7, -0.1, 0.2], dtype=float))
    fluid["u_prev"].components[0].set_nodal_values(ux_ids, np.asarray([0.02, -0.01, 0.01], dtype=float))
    fluid["u_prev"].components[1].set_nodal_values(uy_ids, np.asarray([-0.01, 0.01, 0.02], dtype=float))

    _update_fluid_dvms_predicted_subscale(
        state=fluid["dvms_state"],
        dh=fluid["dh"],
        mesh=mesh_f,
        u_k=fluid["u_k"],
        u_prev=fluid["u_prev"],
        a_prev=fluid["a_prev"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        rho_f=1.0,
        mu_f=1.0,
        dt=1.0,
        bossak_alpha=-0.3,
        dynamic_tau=1.0,
        backend="python",
    )

    batch_base = assemble_fluid_dvms_local_contribution_batch(
        mesh=mesh_f,
        dh=fluid["dh"],
        u_k=fluid["u_k"],
        u_prev=fluid["u_prev"],
        a_prev=fluid["a_prev"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        state=fluid["dvms_state"],
        rho_f=1.0,
        mu_f=1.0,
        dt=1.0,
        bossak_alpha=-0.3,
        element_ids=np.asarray([eid], dtype=int),
        quadrature_order=6,
        contribution_mode="system",
        backend="python",
    )
    K_base, F_base, gdofs_base = _compress_batch_to_fluid_block(
        fluid["dh"],
        LocalAssemblyResult(
            K_elem=np.asarray(batch_base.K_elem, dtype=float),
            F_elem=np.asarray(batch_base.F_elem, dtype=float),
            element_ids=np.asarray(batch_base.element_ids, dtype=int),
            gdofs_map=np.asarray(batch_base.gdofs_map, dtype=int),
        ),
    )
    K_corr, F_corr, gdofs_corr = _dvms_condensed_hidden_state_correction_batch(
        mesh=mesh_f,
        dh=fluid["dh"],
        element_ids=np.asarray([eid], dtype=int),
        u_k=fluid["u_k"],
        u_prev=fluid["u_prev"],
        a_prev=fluid["a_prev"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        state=fluid["dvms_state"],
        rho_f=1.0,
        mu_f=1.0,
        dt=1.0,
        bossak_alpha=-0.3,
        dynamic_tau=1.0,
    )

    batch_condensed = assemble_fluid_dvms_local_contribution_batch(
        mesh=mesh_f,
        dh=fluid["dh"],
        u_k=fluid["u_k"],
        u_prev=fluid["u_prev"],
        a_prev=fluid["a_prev"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        state=fluid["dvms_state"],
        rho_f=1.0,
        mu_f=1.0,
        dt=1.0,
        bossak_alpha=-0.3,
        element_ids=np.asarray([eid], dtype=int),
        quadrature_order=6,
        contribution_mode="system_condensed",
        backend="cpp",
    )
    K_fused, F_fused, gdofs_fused = _compress_batch_to_fluid_block(
        fluid["dh"],
        LocalAssemblyResult(
            K_elem=np.asarray(batch_condensed.K_elem, dtype=float),
            F_elem=np.asarray(batch_condensed.F_elem, dtype=float),
            element_ids=np.asarray(batch_condensed.element_ids, dtype=int),
            gdofs_map=np.asarray(batch_condensed.gdofs_map, dtype=int),
        ),
    )

    np.testing.assert_array_equal(gdofs_base, gdofs_corr)
    np.testing.assert_array_equal(gdofs_base, gdofs_fused)
    np.testing.assert_allclose(K_fused[0], K_base[0] + K_corr[0], atol=2.0e-4, rtol=2.0e-4)
    np.testing.assert_allclose(F_fused[0], F_base[0] + F_corr[0], atol=2.0e-4, rtol=2.0e-4)


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


def test_iqnils_iteration_matrices_match_kratos_newest_first_order() -> None:
    x_history = [
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 0.5]]),
        np.array([[1.5, 1.0]]),
    ]
    g_history = [
        np.array([[1.0, 1.0]]),
        np.array([[2.0, 1.5]]),
        np.array([[2.5, 2.0]]),
    ]

    v_new, w_new = _iqnils_iteration_matrices(
        x_history=x_history,
        g_history=g_history,
        iteration_horizon=50,
    )

    assert v_new is not None
    assert w_new is not None
    # Kratos stores newest entries first, so the first column is based on
    # the last two nonlinear iterates.
    expected_r2 = (g_history[2] - x_history[2]).reshape(-1)
    expected_r1 = (g_history[1] - x_history[1]).reshape(-1)
    expected_g2 = g_history[2].reshape(-1)
    expected_g1 = g_history[1].reshape(-1)
    assert np.allclose(v_new[:, 0], expected_r2 - expected_r1)
    assert np.allclose(w_new[:, 0], expected_g2 - expected_g1)


def test_iqnils_horizon_matches_kratos_snapshot_buffer_length() -> None:
    x_history = [
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 0.0]]),
        np.array([[2.0, 0.0]]),
        np.array([[3.0, 0.0]]),
    ]
    g_history = [
        np.array([[0.5, 0.0]]),
        np.array([[1.5, 0.0]]),
        np.array([[2.5, 0.0]]),
        np.array([[3.5, 0.0]]),
    ]

    v_new, w_new = _iqnils_iteration_matrices(
        x_history=x_history,
        g_history=g_history,
        iteration_horizon=3,
    )

    assert v_new is not None
    assert w_new is not None
    # Kratos keeps only three residual/prediction snapshots when
    # iteration_horizon=3, hence only two V/W columns can exist.
    assert v_new.shape == (2, 2)
    assert w_new.shape == (2, 2)


def test_iqnils_update_matches_kratos_unregularized_least_squares() -> None:
    x_history = [
        np.array([[0.0, 0.0]]),
        np.array([[0.8, 0.1]]),
        np.array([[1.2, 0.4]]),
    ]
    g_history = [
        np.array([[0.4, 0.3]]),
        np.array([[1.0, 0.6]]),
        np.array([[1.3, 0.8]]),
    ]
    x_curr = x_history[-1]
    g_curr = g_history[-1]

    next_x = _iqnils_next_iterate(
        x_curr=x_curr,
        g_curr=g_curr,
        x_history=x_history,
        g_history=g_history,
        dr_old_mats=[],
        dg_old_mats=[],
        omega=0.5,
        horizon=50,
        regularization=0.0,
    )

    r_seq = [(g - x).reshape(-1) for x, g in zip(x_history, g_history)]
    r_recent = list(reversed(r_seq))
    g_recent = [g.reshape(-1) for g in reversed(g_history)]
    V = np.column_stack(r_recent[:-1]) - np.column_stack(r_recent[1:])
    W = np.column_stack(g_recent[:-1]) - np.column_stack(g_recent[1:])
    delta_r = -r_recent[0]
    c = np.linalg.lstsq(V, delta_r, rcond=None)[0]
    expected = x_curr.reshape(-1) + W @ c - delta_r

    assert np.allclose(next_x.reshape(-1), expected)


def test_fluid_interface_reaction_loads_matches_kratos_constrained_rhs_sign(monkeypatch) -> None:
    captured: dict[str, np.ndarray] = {}

    def _fake_assemble_fluid_local_velocity_contribution_raw(
        *,
        prob,
        rho_f,
        mu_f,
        dt,
        quad_order,
        bossak_alpha=-0.3,
        need_matrix=False,
        contribution_mode="system",
        apply_dirichlet_lift=False,
        backend="python",
    ):
        del prob, rho_f, mu_f, dt, quad_order, bossak_alpha, need_matrix, contribution_mode, apply_dirichlet_lift, backend
        return None, np.array([1.0, -2.0, 3.0, -4.0])

    def _fake_boundary_vector_from_global_values(dh, *, vector, tag, global_values):
        del dh, vector, tag
        captured["global_values"] = np.asarray(global_values, dtype=float).copy()
        return np.zeros((2, 2), dtype=float), np.zeros((2, 2), dtype=float)

    class _FakeDH:
        def get_dirichlet_data(self, bcs):
            del bcs
            return {0: 0.0, 3: 0.0}

    monkeypatch.setattr(
        "examples.NIRB.run_example2_local._assemble_fluid_local_velocity_contribution_raw",
        _fake_assemble_fluid_local_velocity_contribution_raw,
    )
    monkeypatch.setattr(
        "examples.NIRB.run_example2_local._fluid_interface_velocity_dofs",
        lambda prob, *, interface_tag: np.array([0, 1, 2, 3], dtype=int),
    )
    monkeypatch.setattr(
        "examples.NIRB.run_example2_local._boundary_vector_from_global_values",
        _fake_boundary_vector_from_global_values,
    )

    prob = {
        "dh": _FakeDH(),
        "u_k": object(),
        "_current_bcs": ["unused"],
    }
    _fluid_interface_reaction_loads(
        prob=prob,
        rho_f=1.0,
        mu_f=1.0,
        dt=1.0,
        quad_order=2,
        interface_tag="interface",
    )

    assert np.allclose(captured["global_values"], np.array([-1.0, 0.0, 0.0, 4.0]))


def test_fluid_interface_reaction_loads_refreshes_predicted_subscale_before_assembly(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def _fake_update_fluid_dvms_predicted_subscale(
        *,
        state,
        dh,
        mesh,
        u_k,
        u_prev,
        a_prev,
        p_k,
        d_mesh,
        d_prev,
        d_prev2=None,
        mesh_v_prev=None,
        mesh_a_prev=None,
        rho_f,
        mu_f,
        dt,
        bossak_alpha,
        dynamic_tau,
        backend,
    ):
        observed["called"] = True
        observed["state"] = state
        observed["dh"] = dh
        observed["mesh"] = mesh
        observed["u_k"] = u_k
        observed["u_prev"] = u_prev
        observed["a_prev"] = a_prev
        observed["p_k"] = p_k
        observed["d_mesh"] = d_mesh
        observed["d_prev"] = d_prev
        observed["rho_f"] = float(rho_f)
        observed["mu_f"] = float(mu_f)
        observed["dt"] = float(dt)
        observed["bossak_alpha"] = float(bossak_alpha)
        observed["dynamic_tau"] = float(dynamic_tau)
        observed["backend"] = str(backend)

    def _fake_assemble_fluid_local_velocity_contribution_raw(
        *,
        prob,
        rho_f,
        mu_f,
        dt,
        quad_order,
        bossak_alpha=-0.3,
        need_matrix=False,
        contribution_mode="system",
        apply_dirichlet_lift=False,
        backend="python",
    ):
        del prob, rho_f, mu_f, dt, quad_order, bossak_alpha, need_matrix, contribution_mode, apply_dirichlet_lift, backend
        assert observed.get("called") is True
        return None, np.array([0.0, 0.0])

    def _fake_boundary_vector_from_global_values(dh, *, vector, tag, global_values):
        del dh, vector, tag, global_values
        return np.zeros((1, 2), dtype=float), np.zeros((1, 2), dtype=float)

    class _FakeMixedElement:
        mesh = object()

    class _FakeDH:
        mixed_element = _FakeMixedElement()

        def get_dirichlet_data(self, bcs):
            del bcs
            return {}

    monkeypatch.setattr(
        "examples.NIRB.run_example2_local._update_fluid_dvms_predicted_subscale",
        _fake_update_fluid_dvms_predicted_subscale,
    )
    monkeypatch.setattr(
        "examples.NIRB.run_example2_local._assemble_fluid_local_velocity_contribution_raw",
        _fake_assemble_fluid_local_velocity_contribution_raw,
    )
    monkeypatch.setattr(
        "examples.NIRB.run_example2_local._fluid_interface_velocity_dofs",
        lambda prob, *, interface_tag: np.array([0, 1], dtype=int),
    )
    monkeypatch.setattr(
        "examples.NIRB.run_example2_local._boundary_vector_from_global_values",
        _fake_boundary_vector_from_global_values,
    )

    fake_state = object()
    fake_u_k = object()
    fake_u_prev = object()
    fake_a_prev = object()
    fake_p_k = object()
    fake_d_mesh = object()
    fake_d_prev = object()
    prob = {
        "dh": _FakeDH(),
        "dvms_state": fake_state,
        "u_k": fake_u_k,
        "u_prev": fake_u_prev,
        "a_prev": fake_a_prev,
        "p_k": fake_p_k,
        "d_mesh": fake_d_mesh,
        "d_prev": fake_d_prev,
        "_current_bcs": [],
    }

    _fluid_interface_reaction_loads(
        prob=prob,
        rho_f=2.0,
        mu_f=3.0,
        dt=4.0,
        quad_order=2,
        bossak_alpha=-0.2,
        dynamic_tau=5.0,
        interface_tag="interface",
        backend="cpp",
    )

    assert observed["called"] is True
    assert observed["state"] is fake_state
    assert observed["u_k"] is fake_u_k
    assert observed["u_prev"] is fake_u_prev
    assert observed["a_prev"] is fake_a_prev
    assert observed["p_k"] is fake_p_k
    assert observed["d_mesh"] is fake_d_mesh
    assert observed["d_prev"] is fake_d_prev
    assert observed["rho_f"] == pytest.approx(2.0)
    assert observed["mu_f"] == pytest.approx(3.0)
    assert observed["dt"] == pytest.approx(4.0)
    assert observed["bossak_alpha"] == pytest.approx(-0.2)
    assert observed["dynamic_tau"] == pytest.approx(5.0)
    assert observed["backend"] == "cpp"


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


def test_reference_interface_point_loads_from_lookup_matches_mass_for_linear_field(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / "double_flap_interface_lookup.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)
    mesh_s = mesh_from_gmsh(mesh_path, surface_physical_names=["solid"], apply_boundary_tags=True)
    solid = _build_solid_problem(mesh_s, poly_order=1)
    solid_iface_coords, _ = _boundary_field_data(solid["dh"], "dx", geometry.interface_tag)

    mass = _build_interface_mass_matrix(mesh_s, solid_iface_coords, geometry.interface_tag)
    traction = np.column_stack(
        [
            0.25 + 0.1 * solid_iface_coords[:, 0],
            -0.4 + 0.2 * solid_iface_coords[:, 1],
        ]
    )
    loads_mass = _interface_load_from_traction(mass, traction)
    loads_lookup = _reference_interface_point_loads_from_lookup(
        mesh=mesh_s,
        iface_coords=solid_iface_coords,
        tag=geometry.interface_tag,
        quad_order=4,
        traction_callback=lambda xy, _N: np.asarray(
            [0.25 + 0.1 * float(xy[0]), -0.4 + 0.2 * float(xy[1])],
            dtype=float,
        ),
    )

    assert np.allclose(loads_lookup, loads_mass, atol=1.0e-12, rtol=1.0e-12)


def test_reference_mdpa_meshes_expose_expected_interface_tags() -> None:
    setup = load_example2_local_setup()
    fluid_mesh, solid_mesh = _load_reference_partitioned_meshes(setup=setup)

    fluid_iface = fluid_mesh.edge_bitset(setup.geometry.interface_tag).to_indices()
    solid_iface = solid_mesh.edge_bitset(setup.geometry.interface_tag).to_indices()

    assert fluid_mesh.element_type == "tri"
    assert solid_mesh.element_type == "quad"
    assert len(fluid_mesh.elements_list) == len(setup.reference.fluid.elements)
    assert len(solid_mesh.elements_list) == len(setup.reference.solid.elements)
    assert len(fluid_iface) > 0
    assert len(solid_iface) > 0
    assert "DISPLACEMENT_BCDisp" in setup.reference.solid.submodelparts


def test_resample_lookup_to_coords_reorders_interface_values() -> None:
    source_coords = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    source_values = np.asarray(
        [
            [10.0, -1.0],
            [20.0, -2.0],
            [30.0, -3.0],
        ],
        dtype=float,
    )
    target_coords = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )

    lookup = CoordinateLookup(source_coords, source_values, dim=2)
    reordered = _resample_lookup_to_coords(lookup, target_coords)

    assert np.allclose(reordered.coords, target_coords)
    assert np.allclose(
        reordered.values,
        np.asarray(
            [
                [20.0, -2.0],
                [10.0, -1.0],
                [30.0, -3.0],
            ],
            dtype=float,
        ),
    )


def test_negate_lookup_flips_interface_load_sign() -> None:
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    values = np.asarray([[2.0, -3.0], [-4.0, 5.0]], dtype=float)
    negated = _negate_lookup(CoordinateLookup(coords, values, dim=2))
    assert np.allclose(negated.coords, coords)
    assert np.allclose(negated.values, -values)


def test_current_dvms_element_size_uses_moved_geometry() -> None:
    setup = load_example2_local_setup()
    mesh_f, _ = _load_reference_partitioned_meshes(setup=setup)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=1)

    eid = 0
    conn = np.asarray(mesh_f.elements_connectivity[eid], dtype=int)
    mx_ids = np.asarray(fluid["dh"].element_maps["mx"][eid], dtype=int)
    my_ids = np.asarray(fluid["dh"].element_maps["my"][eid], dtype=int)
    dx_vals = np.asarray([0.0, 2.5e-4, -1.5e-4], dtype=float)
    dy_vals = np.asarray([0.0, -1.0e-4, 2.0e-4], dtype=float)
    fluid["d_mesh"].components[0].set_nodal_values(mx_ids, dx_vals)
    fluid["d_mesh"].components[1].set_nodal_values(my_ids, dy_vals)

    coords_ref = np.asarray(mesh_f.nodes_x_y_pos[conn], dtype=float)
    coords_cur = coords_ref + np.column_stack([dx_vals, dy_vals])
    area = 0.5 * abs(
        np.linalg.det(
            np.asarray(
                [
                    coords_cur[1] - coords_cur[0],
                    coords_cur[2] - coords_cur[0],
                ],
                dtype=float,
            )
        )
    )
    edges = np.asarray(
        [
            np.linalg.norm(coords_cur[0] - coords_cur[1]),
            np.linalg.norm(coords_cur[1] - coords_cur[2]),
            np.linalg.norm(coords_cur[2] - coords_cur[0]),
        ],
        dtype=float,
    )
    expected = float((2.0 * area) / np.max(edges))
    h_ref = float(_kratos_dvms_element_size(mesh_f, eid))
    h_cur = float(
        _kratos_dvms_current_element_size_array(
            mesh_f,
            fluid["dh"],
            fluid["d_mesh"],
            element_ids=np.asarray([eid], dtype=int),
        )[0]
    )

    assert h_cur == pytest.approx(expected)
    assert h_cur != pytest.approx(h_ref)


def test_local_driver_forms_use_named_jit_constants(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / "double_flap_local_named_constants.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)

    mesh_f = mesh_from_gmsh(mesh_path, surface_physical_names=["fluid"], apply_boundary_tags=True)
    mesh_s = mesh_from_gmsh(mesh_path, surface_physical_names=["solid"], apply_boundary_tags=True)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=6)
    solid = _build_solid_problem(mesh_s, poly_order=1)

    fluid_iface_coords, _ = _boundary_field_data(fluid["dh"], "ux", geometry.interface_tag)
    solid_iface_coords, _ = _boundary_field_data(solid["dh"], "dx", geometry.interface_tag)
    zero_fluid_lookup = CoordinateLookup(fluid_iface_coords, np.zeros((fluid_iface_coords.shape[0], 2), dtype=float), dim=2)
    zero_solid_lookup = CoordinateLookup(solid_iface_coords, np.zeros((solid_iface_coords.shape[0], 2), dtype=float), dim=2)

    fluid_residual, fluid_jacobian, _, _ = _fluid_residual_and_jacobian(
        prob=fluid,
        rho_f=1000.0,
        mu_f=1.0,
        dt=0.008,
        bossak_alpha=-0.3,
        dynamic_tau=1.25,
        pressure_gauge=1.0e-6,
        iface_velocity=zero_fluid_lookup,
        inlet_lookup=lambda x, y, t=0.0: 0.0,
        interface_tag=geometry.interface_tag,
        outlet_tag=geometry.outlet_tag,
        walls_tag=geometry.walls_tag,
        cylinder_tag=geometry.cylinder_tag,
        quad_order=4,
    )
    solid_residual, solid_jacobian, _, _ = _solid_residual_and_jacobian(
        prob=solid,
        traction_lookup=zero_solid_lookup,
        mu_s=5.0e5,
        lambda_s=5.0e5,
        interface_tag=geometry.interface_tag,
        clamp_tag=geometry.clamp_tag,
        quad_order=4,
    )
    mesh_extension_eq, _ = _mesh_extension_equation(
        prob=fluid,
        interface_disp=zero_fluid_lookup,
        interface_tag=geometry.interface_tag,
        fixed_tags=(geometry.inlet_tag, geometry.outlet_tag, geometry.walls_tag, geometry.cylinder_tag),
        quad_order=4,
    )

    names: set[str] = set()
    ewc_names: set[str] = set()
    qstate_names: set[str] = set()
    for expr in (fluid_residual, fluid_jacobian, solid_residual, solid_jacobian, mesh_extension_eq.a, mesh_extension_eq.L):
        param = build_jit_parametrization(expr)
        names.update(param.const_by_name)
        ewc_names.update(param.ewc_by_name)
        qstate_names.update(param.qstate_by_name)

    assert {
        "example2_local_dt",
        "example2_local_bossak_ma0",
        "example2_local_bossak_ma2",
        "example2_local_bossak_mass_coeff",
        "example2_local_bossak_alpha",
        "example2_local_mu_f",
        "example2_local_pressure_gauge",
        "example2_local_rho_f",
        "example2_local_inv_dt",
        "example2_local_tau_c1",
        "example2_local_tau_c2",
        "example2_local_dynamic_tau",
        "example2_local_mu_s",
        "example2_local_lambda_s",
        "example2_local_mesh_mu",
        "example2_local_mesh_lambda",
        "example2_local_zero",
        "example2_local_zero_vec",
        "example2_local_half",
        "example2_local_two",
        "example2_local_four",
        "example2_local_conv_eps",
        "example2_half",
        "example2_one",
        "example2_two",
        "example2_two_thirds",
    }.issubset(names)
    assert {
        "example2_local_dvms_old_subscale_velocity",
        "example2_local_dvms_predicted_subscale_velocity",
        "example2_local_dvms_momentum_projection",
        "example2_local_dvms_mass_projection",
        "example2_local_dvms_old_mass_residual",
    }.issubset(qstate_names)
    assert not {
        "example2_local_dvms_old_subscale_velocity",
        "example2_local_dvms_predicted_subscale_velocity",
        "example2_local_dvms_momentum_projection",
        "example2_local_dvms_mass_projection",
        "example2_local_dvms_old_mass_residual",
    } & ewc_names
