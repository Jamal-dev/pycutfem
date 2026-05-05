from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from examples.NIRB.common import dump_json
from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.run_example2_local import (
    CoordinateLookup,
    FluidDVMSCondensedLocalSystemOperator,
    _EX2L_KRATOS_STRUCT_ONE_STEP_ACCEPT_FACTOR,
    _FluidBossakAccelerationOperator,
    _FluidDVMSSolverOperator,
    _ReducedResidualShiftOperator,
    _boundary_field_data,
    _boundary_point_load_vector,
    _boundary_vector_snapshot,
    _build_fluid_problem,
    _build_kratos_mesh_motion_backend,
    _build_mesh_extension_problem,
    _build_solid_problem,
    _maybe_build_kratos_local_solid_backend,
    _KratosLocalSolidSystemOperator,
    _fluid_boundary_conditions,
    _fluid_interface_reaction_loads,
    _fluid_zero_local_operator_forms,
    _get_or_create_cached_stage_solver,
    _guess_callback_from_snapshots_with_dirichlet,
    _load_checkpoint_payload,
    _load_reference_partitioned_meshes,
    _negate_lookup,
    _restore_fluid_dvms_state,
    _resample_lookup_to_coords,
    _solid_interface_disp_velocity,
    _solid_residual_and_jacobian,
    _solve_kratos_mesh_motion_backend,
    _snapshot_function_values,
    _restore_function_values,
    _transfer_vector_field,
    _update_fluid_dvms_state_from_previous_step,
    _vector_field_matrix,
    _vector_lookup_from_field,
    _warm_fluid_exact_operator_kernels,
)
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, NewtonSolver, TimeStepperParameters


def _coord_key(x: float, y: float, ndigits: int = 12) -> tuple[float, float]:
    return (round(float(x), ndigits), round(float(y), ndigits))


class PointLookup:
    def __init__(self, coords: np.ndarray, values: np.ndarray) -> None:
        self.coords = np.asarray(coords, dtype=float)
        self.values = np.asarray(values, dtype=float)
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError("coords must have shape (n, 2)")
        if self.values.ndim != 2 or self.values.shape[0] != self.coords.shape[0]:
            raise ValueError("values must have shape (n, m)")
        self._exact = {
            _coord_key(x, y): np.asarray(self.values[i], dtype=float).copy()
            for i, (x, y) in enumerate(self.coords)
        }

    def sample(self, target_coords: np.ndarray) -> np.ndarray:
        target = np.asarray(target_coords, dtype=float)
        out = np.empty((target.shape[0], self.values.shape[1]), dtype=float)
        for i, (x, y) in enumerate(target):
            hit = self._exact.get(_coord_key(x, y))
            if hit is not None:
                out[i, :] = hit
                continue
            dist2 = np.sum((self.coords - target[i][None, :]) ** 2, axis=1)
            out[i, :] = self.values[int(np.argmin(dist2)), :]
        return out


def _compare_point_fields(
    *,
    ref_coords: np.ndarray,
    ref_values: np.ndarray,
    local_coords: np.ndarray,
    local_values: np.ndarray,
) -> dict[str, float]:
    ref = np.asarray(ref_values, dtype=float)
    loc = PointLookup(local_coords, local_values).sample(ref_coords)
    diff = loc - ref
    ref_flat = ref.reshape(-1)
    loc_flat = loc.reshape(-1)
    denom = max(float(np.linalg.norm(ref_flat)), 1.0e-15)
    cosine = float(np.dot(loc_flat, ref_flat) / max(float(np.linalg.norm(loc_flat) * np.linalg.norm(ref_flat)), 1.0e-15))
    return {
        "rel_l2": float(np.linalg.norm(diff.reshape(-1)) / denom),
        "abs_rms": float(np.linalg.norm(diff.reshape(-1)) / np.sqrt(max(diff.size, 1))),
        "abs_max": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "cosine": cosine,
        "reference_max_norm": float(np.max(np.linalg.norm(ref, axis=1))) if ref.size else 0.0,
        "local_max_norm": float(np.max(np.linalg.norm(loc, axis=1))) if loc.size else 0.0,
    }


def _load_npz_fields(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {str(k): np.asarray(data[k]) for k in data.files}


def _first_available(data: dict[str, np.ndarray], *names: str) -> np.ndarray:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    raise KeyError(f"None of the requested keys are present: {names}")


def _inlet_profile_factory(setup, t_now: float, reference_velocity: float):
    def inlet_profile(x: float, y: float) -> float:
        del x
        return setup.geometry.inlet_velocity(y, t_now, reference_velocity=reference_velocity)

    return inlet_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one local Example 2 coupling stage with exact Kratos interface input.")
    parser.add_argument("--kratos-post-update", type=Path, required=True, help="Kratos stage NPZ with exact updated fluid.load values.")
    parser.add_argument("--kratos-after-structure", type=Path, required=True, help="Kratos stage NPZ after_sync_output_structure for the replayed iteration.")
    parser.add_argument("--kratos-after-fluid", type=Path, required=True, help="Kratos stage NPZ after_sync_output_fluid for the replayed iteration.")
    parser.add_argument(
        "--kratos-full-stage-state",
        type=Path,
        default=None,
        help="Optional full nodal Kratos stage-state NPZ from dump_kratos_example2_stage_state.py.",
    )
    parser.add_argument(
        "--restart-from-checkpoint",
        type=Path,
        default=None,
        help="Optional local accepted-step checkpoint to seed the replay with the actual previous/current state.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default="cpp")
    parser.add_argument("--linear-backend", choices=("scipy", "petsc", "amgcl"), default="petsc")
    parser.add_argument("--quad-order", type=int, default=1)
    parser.add_argument("--solid-quad-order", type=int, default=2)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    parser.add_argument("--dynamic-tau", type=float, default=1.0)
    parser.add_argument("--newton-tol", type=float, default=1.0e-10)
    parser.add_argument("--max-newton-iter", type=int, default=20)
    parser.add_argument("--reference-velocity", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup = load_example2_local_setup()
    mesh_f, mesh_s = _load_reference_partitioned_meshes(setup=setup)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=int(args.quad_order))
    solid = _build_solid_problem(mesh_s, poly_order=1)
    kratos_local_solid_backend = _maybe_build_kratos_local_solid_backend(
        benchmark_root=Path(setup.reference.root),
        prob=solid,
    )
    mesh_ext = _build_mesh_extension_problem(mesh_f, poly_order=1)
    if args.restart_from_checkpoint is not None:
        restart_payload = _load_checkpoint_payload(Path(args.restart_from_checkpoint).resolve())
        solid["d_k"].nodal_values[:] = np.asarray(restart_payload["solid_d_k"], dtype=float)
        solid["d_prev"].nodal_values[:] = np.asarray(restart_payload["solid_d_prev"], dtype=float)
        fluid["u_k"].nodal_values[:] = np.asarray(restart_payload["fluid_u_k"], dtype=float)
        fluid["p_k"].nodal_values[:] = np.asarray(restart_payload["fluid_p_k"], dtype=float)
        fluid["u_prev"].nodal_values[:] = np.asarray(restart_payload["fluid_u_prev"], dtype=float)
        fluid["p_prev"].nodal_values[:] = np.asarray(restart_payload["fluid_p_prev"], dtype=float)
        fluid["a_prev"].nodal_values[:] = np.asarray(restart_payload["fluid_a_prev"], dtype=float)
        fluid["a_k"].nodal_values[:] = np.asarray(
            restart_payload.get("fluid_a_k", restart_payload["fluid_a_prev"]),
            dtype=float,
        )
        fluid["d_mesh"].nodal_values[:] = np.asarray(restart_payload["fluid_d_mesh"], dtype=float)
        fluid["d_prev"].nodal_values[:] = np.asarray(restart_payload["fluid_d_prev"], dtype=float)
        fluid["d_prev2"].nodal_values[:] = np.asarray(restart_payload["fluid_d_prev2"], dtype=float)
        fluid["w_mesh_prev"].nodal_values[:] = np.asarray(restart_payload["fluid_w_mesh_prev"], dtype=float)
        fluid["a_mesh_prev"].nodal_values[:] = np.asarray(restart_payload["fluid_a_mesh_prev"], dtype=float)
        fluid["w_mesh_k"].nodal_values[:] = fluid["w_mesh_prev"].nodal_values[:]
        fluid["a_mesh_k"].nodal_values[:] = fluid["a_mesh_prev"].nodal_values[:]
        _restore_fluid_dvms_state(
            fluid.get("dvms_state"),
            {
                "old_subscale_velocity": np.asarray(
                    restart_payload.get(
                        "dvms_old_subscale_velocity",
                        np.zeros_like(fluid["dvms_state"].old_subscale_velocity),
                    ),
                    dtype=float,
                ),
                "predicted_subscale_velocity": np.asarray(
                    restart_payload.get(
                        "dvms_predicted_subscale_velocity",
                        np.zeros_like(fluid["dvms_state"].predicted_subscale_velocity),
                    ),
                    dtype=float,
                ),
                "momentum_projection": np.asarray(
                    restart_payload.get(
                        "dvms_momentum_projection",
                        np.zeros_like(fluid["dvms_state"].momentum_projection),
                    ),
                    dtype=float,
                ),
                "mass_projection": np.asarray(
                    restart_payload.get(
                        "dvms_mass_projection",
                        np.zeros_like(fluid["dvms_state"].mass_projection),
                    ),
                    dtype=float,
                ),
                "old_mass_residual": np.asarray(
                    restart_payload.get(
                        "dvms_old_mass_residual",
                        np.zeros_like(fluid["dvms_state"].old_mass_residual),
                    ),
                    dtype=float,
                ),
            },
        )
        mesh_restart_lookup = _vector_lookup_from_field(fluid["dh"], fluid["d_mesh"])
        _transfer_vector_field(target_dh=mesh_ext["dh"], target_vec=mesh_ext["m_k"], source_lookup=mesh_restart_lookup)
        _transfer_vector_field(target_dh=mesh_ext["dh"], target_vec=mesh_ext["m_prev_geom"], source_lookup=mesh_restart_lookup)
    fluid_iface_coords, _ = _boundary_field_data(fluid["dh"], "ux", setup.geometry.interface_tag)
    solid_iface_coords, _ = _boundary_field_data(solid["dh"], "dx", setup.geometry.interface_tag)
    if fluid_iface_coords.size == 0 or solid_iface_coords.size == 0:
        raise RuntimeError("Failed to resolve local interface coordinates.")

    post_update = _load_npz_fields(Path(args.kratos_post_update).resolve())
    after_structure = _load_npz_fields(Path(args.kratos_after_structure).resolve())
    after_fluid = _load_npz_fields(Path(args.kratos_after_fluid).resolve())
    full_stage_state = (
        _load_npz_fields(Path(args.kratos_full_stage_state).resolve())
        if args.kratos_full_stage_state is not None
        else None
    )

    structure_load_source = after_structure
    structure_load_source_label = "after_structure"
    if (
        "structure_load_coords_ref" not in structure_load_source
        or "structure_load_values" not in structure_load_source
    ):
        if "structure_load_coords_ref" in post_update and "structure_load_values" in post_update:
            structure_load_source = post_update
            structure_load_source_label = "post_update"
        else:
            raise KeyError(
                "Could not resolve structure_load_* fields from the supplied Kratos artifacts. "
                "Expected them in --kratos-after-structure or --kratos-post-update."
            )

    # Replay the target structure stage with the exact structure-side load
    # that Kratos applied for this iteration. Some focused structure-stage
    # dumps only persist the structure displacement field; in that case the
    # matching structure-side load comes from the previous iteration's
    # post-update payload, which seeds the next structure solve.
    exact_structure_load_lookup = CoordinateLookup(
        np.asarray(structure_load_source["structure_load_coords_ref"], dtype=float),
        np.asarray(structure_load_source["structure_load_values"], dtype=float),
        dim=2,
    )

    mu_f = float(setup.material.density * setup.material.kinematic_viscosity)
    mu_s = float(setup.material.shear_modulus)
    lambda_s = float(setup.material.lame_lambda)
    dt_value = float(setup.boundaries.time_step)
    reference_velocity = (
        float(args.reference_velocity)
        if args.reference_velocity is not None
        else float(setup.material.max_velocity)
    )

    solid_res, solid_jac, solid_bcs, solid_bcs_homog = _solid_residual_and_jacobian(
        prob=solid,
        traction_lookup=CoordinateLookup(solid_iface_coords, np.zeros((solid_iface_coords.shape[0], 2), dtype=float), dim=2),
        mu_s=mu_s,
        lambda_s=lambda_s,
        interface_tag=setup.geometry.interface_tag,
        clamp_tag=setup.geometry.clamp_tag,
        quad_order=int(args.solid_quad_order),
    )
    structure_newton_tol = float(os.getenv("PYCUTFEM_EX2_LOCAL_STRUCTURE_NEWTON_TOL", "1e-8"))
    structure_newton_rtol = float(os.getenv("PYCUTFEM_EX2_LOCAL_STRUCTURE_NEWTON_RTOL", "0.0"))
    structure_max_newton_iter = int(
        max(
            1,
            float(os.getenv("PYCUTFEM_EX2_LOCAL_STRUCTURE_MAX_NEWTON_ITER", str(args.max_newton_iter))),
        )
    )
    structure_linear_backend = str(
        os.getenv("PYCUTFEM_EX2_LOCAL_STRUCTURE_LINEAR_BACKEND", str(args.linear_backend))
    ).strip().lower()
    solid_solver = NewtonSolver(
        residual_form=solid_res,
        jacobian_form=solid_jac,
        dof_handler=solid["dh"],
        mixed_element=solid["me"],
        bcs=solid_bcs,
        bcs_homog=solid_bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=float(structure_newton_tol),
            newton_rtol=float(structure_newton_rtol),
            max_newton_iter=(
                1
                if bool(str(os.getenv("PYCUTFEM_EX2_STRUCT_ONE_STEP", "0") or "0").strip().lower() in {"1", "true", "yes"})
                else int(structure_max_newton_iter)
            ),
            print_level=0,
            accept_nonconverged_atol_factor=(
                float(_EX2L_KRATOS_STRUCT_ONE_STEP_ACCEPT_FACTOR)
                if bool(str(os.getenv("PYCUTFEM_EX2_STRUCT_ONE_STEP", "0") or "0").strip().lower() in {"1", "true", "yes"})
                else 0.0
            ),
            line_search=False,
            globalization="none",
        ),
        lin_params=LinearSolverParameters(backend=str(structure_linear_backend)),
        quad_order=int(args.solid_quad_order),
        backend=str(args.backend),
    )
    solid_point_load_full = _boundary_point_load_vector(
        solid["dh"],
        vector=solid["d_k"],
        tag=setup.geometry.interface_tag,
        values=np.asarray(exact_structure_load_lookup.values, dtype=float),
    )
    solid_point_load_red = np.asarray(solid_point_load_full[np.asarray(solid_solver.active_dofs, dtype=int)], dtype=float)
    solid_runtime_ops = []
    if kratos_local_solid_backend is not None:
        solid_runtime_ops.append(
            _KratosLocalSolidSystemOperator(
                backend=kratos_local_solid_backend,
                d_k=solid["d_k"],
            )
        )
    solid_runtime_ops.append(_ReducedResidualShiftOperator(solid_point_load_red))
    solid_solver.set_runtime_operators(solid_runtime_ops)
    solid_solver.solve_time_interval(
        functions=[solid["d_k"]],
        prev_functions=[solid["d_prev"]],
        time_params=TimeStepperParameters(dt=1.0, max_steps=1, final_time=1.0, stop_on_steady=False),
    )

    prev_iface_mesh_vel_lookup = _resample_lookup_to_coords(
        _vector_lookup_from_field(fluid["dh"], fluid["w_mesh_prev"]),
        solid_iface_coords,
    )
    prev_iface_mesh_acc_lookup = _resample_lookup_to_coords(
        _vector_lookup_from_field(fluid["dh"], fluid["a_mesh_prev"]),
        solid_iface_coords,
    )
    solid_disp_solid_lookup, _ = _solid_interface_disp_velocity(
        dh=solid["dh"],
        mesh=mesh_s,
        d_curr=solid["d_k"],
        d_prev=solid["d_prev"],
        iface_coords=solid_iface_coords,
        dt=dt_value,
        v_prev_lookup=prev_iface_mesh_vel_lookup,
        a_prev_lookup=prev_iface_mesh_acc_lookup,
        bossak_alpha=float(args.bossak_alpha),
    )
    solid_disp_fluid_lookup = _resample_lookup_to_coords(solid_disp_solid_lookup, fluid_iface_coords)

    kratos_mesh_backend = _build_kratos_mesh_motion_backend(
        fluid_mdpa_path=setup.reference.fluid.path,
        dt=dt_value,
        bossak_alpha=float(args.bossak_alpha),
    )
    try:
        from examples.NIRB.run_example2_local import _advance_kratos_mesh_motion_backend_step

        _advance_kratos_mesh_motion_backend_step(backend=kratos_mesh_backend)
        mesh_lookup, mesh_vel_fluid_lookup, mesh_accel_fluid_lookup = _solve_kratos_mesh_motion_backend(
            backend=kratos_mesh_backend,
            interface_disp=solid_disp_fluid_lookup,
        )
    finally:
        from examples.NIRB.run_example2_local import _finalize_kratos_mesh_motion_backend_step

        _finalize_kratos_mesh_motion_backend_step(backend=kratos_mesh_backend)

    _transfer_vector_field(target_dh=fluid["dh"], target_vec=fluid["d_mesh"], source_lookup=mesh_lookup)
    _transfer_vector_field(target_dh=fluid["dh"], target_vec=fluid["w_mesh_k"], source_lookup=mesh_vel_fluid_lookup)
    _transfer_vector_field(
        target_dh=fluid["dh"],
        target_vec=fluid["a_mesh_k"],
        source_lookup=mesh_accel_fluid_lookup,
    )
    _transfer_vector_field(target_dh=mesh_ext["dh"], target_vec=mesh_ext["m_k"], source_lookup=mesh_lookup)
    _transfer_vector_field(target_dh=mesh_ext["dh"], target_vec=mesh_ext["m_prev_geom"], source_lookup=mesh_lookup)

    _update_fluid_dvms_state_from_previous_step(
        state=fluid["dvms_state"],
        dh=fluid["dh"],
        mesh=mesh_f,
        u_prev=fluid["u_prev"],
        d_prev=fluid["d_prev"],
        d_geo=fluid["d_mesh"],
        backend=str(args.backend),
    )
    _warm_fluid_exact_operator_kernels(
        prob=fluid,
        mesh=mesh_f,
        rho_f=float(setup.material.density),
        mu_f=mu_f,
        dt=dt_value,
        bossak_alpha=float(args.bossak_alpha),
        dynamic_tau=float(args.dynamic_tau),
        quad_order=int(args.quad_order),
        backend=str(args.backend),
        contribution_mode="system",
    )

    inlet_profile = _inlet_profile_factory(setup, dt_value, reference_velocity)
    fluid_res, fluid_jac, fluid_bcs, fluid_bcs_homog = _fluid_zero_local_operator_forms(
        prob=fluid,
        iface_velocity=mesh_vel_fluid_lookup,
        inlet_lookup=inlet_profile,
        interface_tag=setup.geometry.interface_tag,
        outlet_tag=setup.geometry.outlet_tag,
        walls_tag=setup.geometry.walls_tag,
        cylinder_tag=setup.geometry.cylinder_tag,
        quad_order=int(args.quad_order),
    )
    fluid_predictor_operator = _FluidDVMSSolverOperator(
        state=fluid["dvms_state"],
        dh=fluid["dh"],
        mesh=mesh_f,
        u_k=fluid["u_k"],
        u_prev=fluid["u_prev"],
        a_prev=fluid["a_prev"],
        a_curr=fluid["a_k"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        d_prev2=fluid["d_prev2"],
        mesh_v=None,
        mesh_v_prev=fluid["w_mesh_prev"],
        mesh_a_prev=fluid["a_mesh_prev"],
        rho_f=float(setup.material.density),
        mu_f=mu_f,
        dt=dt_value,
        bossak_alpha=float(args.bossak_alpha),
        dynamic_tau=float(args.dynamic_tau),
        refresh_on_initial_assembly=False,
        reset_predicted_to_old_on_step_begin=True,
    )
    fluid_exact_operator = FluidDVMSCondensedLocalSystemOperator(
        mesh=mesh_f,
        dh=fluid["dh"],
        u_k=fluid["u_k"],
        u_prev=fluid["u_prev"],
        a_prev=fluid["a_prev"],
        a_curr=fluid["a_k"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        d_prev2=fluid["d_prev2"],
        mesh_v=None,
        mesh_v_prev=fluid["w_mesh_prev"],
        mesh_a_prev=fluid["a_mesh_prev"],
        state=fluid["dvms_state"],
        rho_f=float(setup.material.density),
        mu_f=mu_f,
        dt=dt_value,
        bossak_alpha=float(args.bossak_alpha),
        quadrature_order=int(args.quad_order),
        dynamic_tau=float(args.dynamic_tau),
        refresh_predicted_subscale=False,
        apply_dirichlet_lift=False,
    )
    fluid_predictor_operator.arm_initial_old_subscale_build()
    fluid_accel_operator = _FluidBossakAccelerationOperator(
        u_k=fluid["u_k"],
        a_k=fluid["a_k"],
        dt=dt_value,
        bossak_alpha=float(args.bossak_alpha),
    )
    exact_fluid_newton_tol = min(float(args.newton_tol), 1.0e-6)

    stage_solver = _get_or_create_cached_stage_solver(
        cache_owner=fluid,
        cache_name="_replay_stage_solver_cache",
        cache_key=("replay", str(args.backend), str(args.linear_backend), int(args.quad_order)),
        residual_form=fluid_res,
        jacobian_form=fluid_jac,
        dof_handler=fluid["dh"],
        mixed_element=fluid["me"],
        bcs=fluid_bcs,
        bcs_homog=fluid_bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=float(exact_fluid_newton_tol),
            max_newton_iter=int(args.max_newton_iter),
            print_level=0,
            line_search=False,
            ls_fail_hard=True,
            globalization="none",
        ),
        lin_params=LinearSolverParameters(backend=str(args.linear_backend)),
        quad_order=int(args.quad_order),
        backend=str(args.backend),
        operators=[fluid_accel_operator, fluid_predictor_operator, fluid_exact_operator],
        active_fields=("ux", "uy", "p"),
    )
    fluid_guess = _snapshot_function_values([fluid["u_k"], fluid["p_k"]])
    fluid_prev_step = _snapshot_function_values([fluid["u_prev"], fluid["p_prev"]])
    fluid_mesh_prev_step = _snapshot_function_values(
        [fluid["d_prev"], fluid["d_prev2"], fluid["w_mesh_prev"], fluid["a_mesh_prev"]]
    )
    stage_solver.solve_time_interval(
        functions=[fluid["u_k"], fluid["p_k"]],
        prev_functions=[fluid["u_prev"], fluid["p_prev"]],
        aux_functions={
            "a_prev": fluid["a_prev"],
            "a_k": fluid["a_k"],
            "d_mesh": fluid["d_mesh"],
            "d_prev": fluid["d_prev"],
            "d_prev2": fluid["d_prev2"],
        },
        time_params=TimeStepperParameters(
            dt=dt_value,
            max_steps=1,
            final_time=dt_value,
            stop_on_steady=False,
            step_initial_guess_callback=_guess_callback_from_snapshots_with_dirichlet(
                snapshots=fluid_guess,
                dh=fluid["dh"],
                bcs=fluid_bcs,
            ),
        ),
    )
    _restore_function_values([fluid["u_prev"], fluid["p_prev"]], fluid_prev_step)
    _restore_function_values(
        [fluid["d_prev"], fluid["d_prev2"], fluid["w_mesh_prev"], fluid["a_mesh_prev"]],
        fluid_mesh_prev_step,
    )

    reaction_point_load_lookup = _fluid_interface_reaction_loads(
        prob=fluid,
        rho_f=float(setup.material.density),
        mu_f=mu_f,
        dt=dt_value,
        quad_order=int(args.quad_order),
        bossak_alpha=float(args.bossak_alpha),
        dynamic_tau=float(args.dynamic_tau),
        interface_tag=setup.geometry.interface_tag,
        backend=str(args.backend),
        contribution_mode="system",
    )
    reaction_point_load_lookup_lift = _fluid_interface_reaction_loads(
        prob=fluid,
        rho_f=float(setup.material.density),
        mu_f=mu_f,
        dt=dt_value,
        quad_order=int(args.quad_order),
        bossak_alpha=float(args.bossak_alpha),
        dynamic_tau=float(args.dynamic_tau),
        interface_tag=setup.geometry.interface_tag,
        backend=str(args.backend),
        contribution_mode="system",
        apply_dirichlet_lift=True,
    )

    local_structure_disp_coords, local_structure_disp_values = _boundary_vector_snapshot(
        solid["dh"], solid["d_k"], setup.geometry.interface_tag
    )
    local_fluid_velocity_coords, local_fluid_velocity_values = _boundary_vector_snapshot(
        fluid["dh"], fluid["u_k"], setup.geometry.interface_tag
    )
    local_fluid_load_coords = np.asarray(reaction_point_load_lookup.coords, dtype=float)
    local_fluid_load_values = np.asarray(reaction_point_load_lookup.values, dtype=float)
    local_fluid_coords, local_fluid_velocity_full = _vector_field_matrix(fluid["dh"], fluid["u_k"])
    fluid["dh"]._ensure_dof_coords()
    local_pressure_coords = np.asarray(fluid["dh"]._dof_coords[np.asarray(fluid["dh"].get_field_slice(fluid["p_k"].field_name), dtype=int)], dtype=float)
    local_pressure_values = np.asarray(
        fluid["p_k"].get_nodal_values(np.asarray(fluid["dh"].get_field_slice(fluid["p_k"].field_name), dtype=int)),
        dtype=float,
    ).reshape(-1, 1)
    local_mesh_coords, local_mesh_displacement_values = _vector_field_matrix(fluid["dh"], fluid["d_mesh"])

    structure_disp_cmp = _compare_point_fields(
        ref_coords=np.asarray(after_structure["structure_disp_coords_ref"], dtype=float),
        ref_values=np.asarray(after_structure["structure_disp_values"], dtype=float),
        local_coords=local_structure_disp_coords,
        local_values=local_structure_disp_values,
    )
    fluid_velocity_cmp = _compare_point_fields(
        ref_coords=np.asarray(after_fluid["fluid_velocity_coords_ref"], dtype=float),
        ref_values=np.asarray(after_fluid["fluid_velocity_values"], dtype=float),
        local_coords=local_fluid_velocity_coords,
        local_values=local_fluid_velocity_values,
    )
    fluid_load_cmp = _compare_point_fields(
        ref_coords=np.asarray(after_fluid["fluid_load_coords_ref"], dtype=float),
        ref_values=np.asarray(after_fluid["fluid_load_values"], dtype=float),
        local_coords=local_fluid_load_coords,
        local_values=local_fluid_load_values,
    )
    fluid_load_cmp_lift = _compare_point_fields(
        ref_coords=np.asarray(after_fluid["fluid_load_coords_ref"], dtype=float),
        ref_values=np.asarray(after_fluid["fluid_load_values"], dtype=float),
        local_coords=np.asarray(reaction_point_load_lookup_lift.coords, dtype=float),
        local_values=np.asarray(reaction_point_load_lookup_lift.values, dtype=float),
    )
    next_guess_cmp = _compare_point_fields(
        ref_coords=np.asarray(structure_load_source["structure_load_coords_ref"], dtype=float),
        ref_values=np.asarray(structure_load_source["structure_load_values"], dtype=float),
        local_coords=np.asarray(exact_structure_load_lookup.coords, dtype=float),
        local_values=np.asarray(exact_structure_load_lookup.values, dtype=float),
    )

    summary: dict[str, Any] = {
        "kratos_post_update": str(Path(args.kratos_post_update).resolve()),
        "kratos_after_structure": str(Path(args.kratos_after_structure).resolve()),
        "kratos_after_fluid": str(Path(args.kratos_after_fluid).resolve()),
        "structure_load_source": str(structure_load_source_label),
        "backend": str(args.backend),
        "linear_backend": str(args.linear_backend),
        "quad_order": int(args.quad_order),
        "solid_quad_order": int(args.solid_quad_order),
        "exact_fluid_newton_tol": float(exact_fluid_newton_tol),
        "structure_disp_cmp": structure_disp_cmp,
        "fluid_velocity_cmp": fluid_velocity_cmp,
        "fluid_load_cmp": fluid_load_cmp,
        "fluid_load_cmp_lift": fluid_load_cmp_lift,
        "input_guess_cmp": next_guess_cmp,
    }
    if isinstance(full_stage_state, dict):
        summary["kratos_full_stage_state"] = str(Path(args.kratos_full_stage_state).resolve())
        full_coords_ref = _first_available(full_stage_state, "node_coords_ref", "coords_ref")
        full_velocity = _first_available(full_stage_state, "velocity", "velocity_nodal_values")
        full_pressure = _first_available(full_stage_state, "pressure", "pressure_nodal_values")
        full_mesh_displacement = _first_available(
            full_stage_state,
            "mesh_displacement",
            "mesh_displacement_nodal_values",
        )
        summary["fluid_velocity_full_cmp"] = _compare_point_fields(
            ref_coords=np.asarray(full_coords_ref, dtype=float),
            ref_values=np.asarray(full_velocity, dtype=float),
            local_coords=local_fluid_coords,
            local_values=local_fluid_velocity_full,
        )
        summary["fluid_pressure_full_cmp"] = _compare_point_fields(
            ref_coords=np.asarray(full_coords_ref, dtype=float),
            ref_values=np.asarray(full_pressure, dtype=float).reshape(-1, 1),
            local_coords=local_pressure_coords,
            local_values=local_pressure_values,
        )
        summary["fluid_mesh_displacement_full_cmp"] = _compare_point_fields(
            ref_coords=np.asarray(full_coords_ref, dtype=float),
            ref_values=np.asarray(full_mesh_displacement, dtype=float),
            local_coords=local_mesh_coords,
            local_values=local_mesh_displacement_values,
        )
    dump_json(summary, Path(args.output))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
