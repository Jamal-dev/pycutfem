from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from examples.NIRB.common import dump_json
from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.fluid_basis import fit_fluid_pod_trial_basis
from examples.NIRB.fluid_fom_operator import FluidBoundaryTags, FluidFOMOperator, FluidFOMParameters
from examples.NIRB.fluid_gnat import FluidGNATSolver, fit_fluid_gnat_sample_set
from examples.NIRB.fluid_lspg import FluidLSPGVerifier, pack_fluid_state, write_fluid_state
from examples.NIRB.fluid_mode_selection import run_fluid_mode_cross_validation
from examples.NIRB.fluid_snapshots import FluidStageSnapshotBatch, FluidStageSnapshotWriter
from examples.NIRB.run_example2_local import (
    _bossak_coefficients,
    _boundary_field_data,
    _build_fluid_problem,
    _load_checkpoint_payload,
    _load_reference_partitioned_meshes,
    _resample_lookup_to_coords,
    _restore_fluid_dvms_state,
    _update_fluid_dvms_state_from_previous_step,
    _vector_lookup_from_field,
)


def _parse_mode_candidates(spec: str) -> list[int]:
    text = str(spec).strip()
    if ":" in text:
        left, right = text.split(":", 1)
        start = int(left)
        stop = int(right)
        if stop < start:
            raise ValueError(f"Invalid mode range {spec!r}.")
        return list(range(start, stop + 1))
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Mode candidate list must not be empty.")
    return values


def _checkpoint_paths(root: Path, *, max_snapshots: int | None) -> list[Path]:
    source = Path(root).resolve()
    if source.is_file():
        paths = [source]
    else:
        paths = sorted(source.glob("checkpoint_step_*.npz"))
    if max_snapshots is not None:
        paths = paths[: max(0, int(max_snapshots))]
    if len(paths) < 2:
        raise ValueError(f"Need at least two checkpoints under {source}.")
    return paths


def _restore_dvms_from_payload(fluid: dict[str, object], payload: dict[str, np.ndarray]) -> None:
    state = fluid.get("dvms_state")
    if state is None:
        return
    _restore_fluid_dvms_state(
        state,
        {
            "old_subscale_velocity": np.asarray(
                payload.get("dvms_old_subscale_velocity", np.zeros_like(state.old_subscale_velocity)),
                dtype=float,
            ),
            "predicted_subscale_velocity": np.asarray(
                payload.get("dvms_predicted_subscale_velocity", np.zeros_like(state.predicted_subscale_velocity)),
                dtype=float,
            ),
            "momentum_projection": np.asarray(
                payload.get("dvms_momentum_projection", np.zeros_like(state.momentum_projection)),
                dtype=float,
            ),
            "mass_projection": np.asarray(
                payload.get("dvms_mass_projection", np.zeros_like(state.mass_projection)),
                dtype=float,
            ),
            "old_mass_residual": np.asarray(
                payload.get("dvms_old_mass_residual", np.zeros_like(state.old_mass_residual)),
                dtype=float,
            ),
        },
    )


def _restore_accepted_checkpoint(fluid: dict[str, object], payload: dict[str, np.ndarray]) -> None:
    fluid["u_k"].nodal_values[:] = np.asarray(payload["fluid_u_k"], dtype=float)
    fluid["p_k"].nodal_values[:] = np.asarray(payload["fluid_p_k"], dtype=float)
    fluid["u_prev"].nodal_values[:] = np.asarray(payload["fluid_u_prev"], dtype=float)
    fluid["p_prev"].nodal_values[:] = np.asarray(payload["fluid_p_prev"], dtype=float)
    fluid["a_prev"].nodal_values[:] = np.asarray(payload["fluid_a_prev"], dtype=float)
    fluid["a_k"].nodal_values[:] = np.asarray(payload.get("fluid_a_k", payload["fluid_a_prev"]), dtype=float)
    fluid["d_mesh"].nodal_values[:] = np.asarray(payload["fluid_d_mesh"], dtype=float)
    fluid["d_prev"].nodal_values[:] = np.asarray(payload["fluid_d_prev"], dtype=float)
    fluid["d_prev2"].nodal_values[:] = np.asarray(payload["fluid_d_prev2"], dtype=float)
    fluid["w_mesh_prev"].nodal_values[:] = np.asarray(payload["fluid_w_mesh_prev"], dtype=float)
    fluid["w_mesh_k"].nodal_values[:] = np.asarray(payload.get("fluid_w_mesh_k", payload["fluid_w_mesh_prev"]), dtype=float)
    fluid["a_mesh_prev"].nodal_values[:] = np.asarray(payload["fluid_a_mesh_prev"], dtype=float)
    fluid["a_mesh_k"].nodal_values[:] = np.asarray(payload.get("fluid_a_mesh_k", payload["fluid_a_mesh_prev"]), dtype=float)
    _restore_dvms_from_payload(fluid, payload)


def _restore_replay_initial_state(
    *,
    fluid: dict[str, object],
    mesh,
    previous_payload: dict[str, np.ndarray],
    target_payload: dict[str, np.ndarray],
    backend: str,
) -> None:
    fluid["u_prev"].nodal_values[:] = np.asarray(previous_payload["fluid_u_k"], dtype=float)
    fluid["p_prev"].nodal_values[:] = np.asarray(previous_payload["fluid_p_k"], dtype=float)
    fluid["a_prev"].nodal_values[:] = np.asarray(previous_payload.get("fluid_a_k", previous_payload["fluid_a_prev"]), dtype=float)
    fluid["u_k"].nodal_values[:] = np.asarray(previous_payload["fluid_u_k"], dtype=float)
    fluid["p_k"].nodal_values[:] = np.asarray(previous_payload["fluid_p_k"], dtype=float)
    fluid["a_k"].nodal_values[:] = np.asarray(previous_payload.get("fluid_a_k", previous_payload["fluid_a_prev"]), dtype=float)
    fluid["d_prev"].nodal_values[:] = np.asarray(previous_payload["fluid_d_mesh"], dtype=float)
    fluid["d_prev2"].nodal_values[:] = np.asarray(previous_payload["fluid_d_prev"], dtype=float)
    fluid["w_mesh_prev"].nodal_values[:] = np.asarray(previous_payload["fluid_w_mesh_prev"], dtype=float)
    fluid["a_mesh_prev"].nodal_values[:] = np.asarray(previous_payload["fluid_a_mesh_prev"], dtype=float)
    fluid["d_mesh"].nodal_values[:] = np.asarray(target_payload["fluid_d_mesh"], dtype=float)
    fluid["w_mesh_k"].nodal_values[:] = np.asarray(target_payload.get("fluid_w_mesh_k", target_payload["fluid_w_mesh_prev"]), dtype=float)
    fluid["a_mesh_k"].nodal_values[:] = np.asarray(target_payload.get("fluid_a_mesh_k", target_payload["fluid_a_mesh_prev"]), dtype=float)
    _restore_dvms_from_payload(fluid, previous_payload)
    _update_fluid_dvms_state_from_previous_step(
        state=fluid["dvms_state"],
        dh=fluid["dh"],
        mesh=mesh,
        u_prev=fluid["u_prev"],
        d_prev=fluid["d_prev"],
        d_geo=fluid["d_mesh"],
        backend=str(backend),
    )


def _configure_checkpoint_bcs(
    *,
    operator: FluidFOMOperator,
    setup,
    fluid_iface_coords: np.ndarray,
    time_s: float,
    reference_velocity: float,
) -> None:
    mesh_velocity = _resample_lookup_to_coords(
        _vector_lookup_from_field(operator.dh, operator.prob["w_mesh_k"]),
        fluid_iface_coords,
    )

    def inlet_profile(x: float, y: float) -> float:
        del x
        return setup.geometry.inlet_velocity(float(y), float(time_s), reference_velocity=float(reference_velocity))

    operator.configure_boundary_conditions(
        iface_velocity=mesh_velocity,
        inlet_lookup=inlet_profile,
        apply_to_state=True,
    )


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float).reshape(-1)
    cand = np.asarray(candidate, dtype=float).reshape(-1)
    return float(np.linalg.norm(cand - ref) / max(float(np.linalg.norm(ref)), 1.0e-15))


def _fluid_block_row_weights(
    operator: FluidFOMOperator,
    *,
    row_dofs: np.ndarray,
    residual: np.ndarray,
    floor: float = 1.0e-12,
) -> np.ndarray:
    rows = np.asarray(row_dofs, dtype=int).reshape(-1)
    res = np.asarray(residual, dtype=float).reshape(-1)
    if int(res.size) != int(rows.size):
        raise ValueError("residual must be restricted to row_dofs for block scaling.")
    weights = np.ones(int(rows.size), dtype=float)
    blocks = (
        tuple(
            np.concatenate(
                [
                    np.asarray(operator.dh.get_field_slice("ux"), dtype=int).reshape(-1),
                    np.asarray(operator.dh.get_field_slice("uy"), dtype=int).reshape(-1),
                ]
            ).tolist()
        ),
        tuple(np.asarray(operator.dh.get_field_slice("p"), dtype=int).reshape(-1).tolist()),
    )
    for block in blocks:
        block_rows = np.intersect1d(rows, np.asarray(block, dtype=int), assume_unique=False)
        if block_rows.size == 0:
            continue
        row_mask = np.isin(rows, block_rows)
        rms = float(np.linalg.norm(res[row_mask]) / np.sqrt(max(int(np.count_nonzero(row_mask)), 1)))
        weights[row_mask] = 1.0 / max(rms, float(floor)) ** 2
    return weights


def _make_operator(*, setup, mesh_f, fluid, backend: str, quadrature_order: int, bossak_alpha: float, dynamic_tau: float) -> FluidFOMOperator:
    mu_f = float(setup.material.density * setup.material.kinematic_viscosity)
    return FluidFOMOperator(
        prob=fluid,
        mesh=mesh_f,
        parameters=FluidFOMParameters(
            rho_f=float(setup.material.density),
            mu_f=mu_f,
            dt=float(setup.boundaries.time_step),
            quadrature_order=int(quadrature_order),
            bossak_alpha=float(bossak_alpha),
            dynamic_tau=float(dynamic_tau),
            backend=str(backend),
            contribution_mode="system",
        ),
        boundary_tags=FluidBoundaryTags(
            interface_tag=setup.geometry.interface_tag,
            outlet_tag=setup.geometry.outlet_tag,
            walls_tag=setup.geometry.walls_tag,
            cylinder_tag=setup.geometry.cylinder_tag,
        ),
    )


def _build_snapshot_batch(
    *,
    operator: FluidFOMOperator,
    setup,
    fluid: dict[str, object],
    checkpoint_paths: list[Path],
    fluid_iface_coords: np.ndarray,
    reference_velocity: float,
) -> FluidStageSnapshotBatch:
    writer = FluidStageSnapshotWriter()
    for path in checkpoint_paths:
        payload = _load_checkpoint_payload(Path(path).resolve())
        _restore_accepted_checkpoint(fluid, payload)
        _configure_checkpoint_bcs(
            operator=operator,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            time_s=float(np.asarray(payload.get("time_s", np.asarray(0.0))).reshape(-1)[0]),
            reference_velocity=float(reference_velocity),
        )
        reaction = operator.reaction_loads(refresh_state=False)
        writer.append_from_operator(
            operator,
            reaction_loads=reaction,
            metadata={
                "checkpoint": str(Path(path).resolve()),
                "checkpoint_name": Path(path).name,
                "step": int(np.asarray(payload.get("step", np.asarray(-1)), dtype=int).reshape(-1)[0]),
                "time_s": float(np.asarray(payload.get("time_s", np.asarray(0.0))).reshape(-1)[0]),
            },
        )
    return writer.to_batch()


def _collect_replay_residual_snapshots(
    *,
    operator: FluidFOMOperator,
    setup,
    fluid: dict[str, object],
    mesh_f,
    checkpoint_paths: list[Path],
    indices: np.ndarray,
    rows: np.ndarray,
    fluid_iface_coords: np.ndarray,
    reference_velocity: float,
    backend: str,
) -> np.ndarray:
    residuals: list[np.ndarray] = []
    row_ids = np.asarray(rows, dtype=int).reshape(-1)
    for idx in np.asarray(indices, dtype=int).reshape(-1):
        target_payload = _load_checkpoint_payload(checkpoint_paths[int(idx)])
        _restore_accepted_checkpoint(fluid, target_payload)
        _configure_checkpoint_bcs(
            operator=operator,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            time_s=float(np.asarray(target_payload.get("time_s", np.asarray(0.0))).reshape(-1)[0]),
            reference_velocity=float(reference_velocity),
        )
        accepted = operator.assemble(need_matrix=False, convention="newton", refresh_predicted=False)
        residuals.append(np.asarray(accepted.residual[row_ids], dtype=float).reshape(-1))
        if int(idx) <= 0:
            continue
        previous_payload = _load_checkpoint_payload(checkpoint_paths[int(idx) - 1])
        _restore_replay_initial_state(
            fluid=fluid,
            mesh=mesh_f,
            previous_payload=previous_payload,
            target_payload=target_payload,
            backend=str(backend),
        )
        _configure_checkpoint_bcs(
            operator=operator,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            time_s=float(np.asarray(target_payload.get("time_s", np.asarray(0.0))).reshape(-1)[0]),
            reference_velocity=float(reference_velocity),
        )
        initial = operator.assemble(need_matrix=False, convention="newton", refresh_predicted=True)
        residuals.append(np.asarray(initial.residual[row_ids], dtype=float).reshape(-1))
    if not residuals:
        raise RuntimeError("No residual snapshots were collected for GNAT training.")
    return np.column_stack(residuals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tiba-style fluid mode validation and GNAT replay on accepted Example 2 FSI checkpoints."
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("examples/NIRB/artifacts/nirb_example2_local_long_20260421_190524/checkpoints"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default="cpp")
    parser.add_argument("--quad-order", type=int, default=1)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    parser.add_argument("--dynamic-tau", type=float, default=1.0)
    parser.add_argument("--reference-velocity", type=float, default=None)
    parser.add_argument("--max-snapshots", type=int, default=12)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--validation-strategy", choices=("last", "random"), default="last")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--velocity-modes", type=str, default="1:8")
    parser.add_argument("--pressure-modes", type=str, default="1:5")
    parser.add_argument("--plateau-rel-tol", type=float, default=0.05)
    parser.add_argument("--residual-modes", type=int, default=None)
    parser.add_argument("--row-oversampling", type=float, default=2.0)
    parser.add_argument("--run-full-lspg", action="store_true")
    parser.add_argument("--full-lspg-max-iterations", type=int, default=8)
    parser.add_argument("--full-lspg-residual-tol", type=float, default=1.0e-8)
    parser.add_argument("--full-lspg-line-search", action="store_true")
    parser.add_argument("--full-lspg-block-scale", action="store_true")
    parser.add_argument("--skip-gnat", action="store_true")
    parser.add_argument("--gnat-max-iterations", type=int, default=8)
    parser.add_argument("--gnat-residual-tol", type=float, default=1.0e-8)
    parser.add_argument("--gnat-line-search", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup = load_example2_local_setup()
    mesh_f, _mesh_s = _load_reference_partitioned_meshes(setup=setup)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=int(args.quad_order))
    operator = _make_operator(
        setup=setup,
        mesh_f=mesh_f,
        fluid=fluid,
        backend=str(args.backend),
        quadrature_order=int(args.quad_order),
        bossak_alpha=float(args.bossak_alpha),
        dynamic_tau=float(args.dynamic_tau),
    )
    fluid_iface_coords, _ = _boundary_field_data(operator.dh, "ux", setup.geometry.interface_tag)
    reference_velocity = (
        float(args.reference_velocity)
        if args.reference_velocity is not None
        else float(setup.material.max_velocity)
    )
    checkpoint_paths = _checkpoint_paths(Path(args.checkpoints), max_snapshots=args.max_snapshots)
    batch = _build_snapshot_batch(
        operator=operator,
        setup=setup,
        fluid=fluid,
        checkpoint_paths=checkpoint_paths,
        fluid_iface_coords=fluid_iface_coords,
        reference_velocity=float(reference_velocity),
    )

    if str(args.validation_strategy) == "last":
        n_test = max(1, int(round(float(args.validation_fraction) * batch.n_snapshots)))
        validation_indices = np.arange(batch.n_snapshots - n_test, batch.n_snapshots, dtype=int)
    else:
        validation_indices = None
    mode_sweep = run_fluid_mode_cross_validation(
        operator,
        batch,
        velocity_modes=_parse_mode_candidates(str(args.velocity_modes)),
        pressure_modes=_parse_mode_candidates(str(args.pressure_modes)),
        test_fraction=float(args.validation_fraction),
        random_state=int(args.random_state),
        test_indices=validation_indices,
        center=True,
        include_reaction_error=True,
        reaction_refresh_state=True,
    )
    selected = mode_sweep.best(plateau_rel_tol=float(args.plateau_rel_tol))
    train_batch = batch.subset(mode_sweep.train_indices)
    trial_basis = fit_fluid_pod_trial_basis(
        operator,
        train_batch,
        velocity_modes=int(selected.velocity_modes),
        pressure_modes=int(selected.pressure_modes),
        center=True,
    )
    residual_snapshots = _collect_replay_residual_snapshots(
        operator=operator,
        setup=setup,
        fluid=fluid,
        mesh_f=mesh_f,
        checkpoint_paths=checkpoint_paths,
        indices=mode_sweep.train_indices,
        rows=train_batch.free_dofs,
        fluid_iface_coords=fluid_iface_coords,
        reference_velocity=float(reference_velocity),
        backend=str(args.backend),
    )
    residual_modes = (
        int(args.residual_modes)
        if args.residual_modes is not None
        else max(int(selected.total_modes), min(3, int(residual_snapshots.shape[1])))
    )
    sample_set = fit_fluid_gnat_sample_set(
        operator,
        residual_snapshots,
        basis_dofs=train_batch.free_dofs,
        residual_modes=int(residual_modes),
        row_oversampling=float(args.row_oversampling),
        include_interface_elements=True,
    )

    free = np.asarray(train_batch.free_dofs, dtype=int)
    replay_entries: list[dict[str, Any]] = []
    full_assembly_times: list[float] = []
    gnat_times: list[float] = []
    for idx in mode_sweep.test_indices:
        target_index = int(idx)
        if target_index <= 0:
            continue
        target_payload = _load_checkpoint_payload(checkpoint_paths[target_index])
        previous_payload = _load_checkpoint_payload(checkpoint_paths[target_index - 1])

        _restore_accepted_checkpoint(fluid, target_payload)
        _configure_checkpoint_bcs(
            operator=operator,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            time_s=float(np.asarray(target_payload.get("time_s", np.asarray(0.0))).reshape(-1)[0]),
            reference_velocity=float(reference_velocity),
        )
        target_state = pack_fluid_state(operator)
        target_reaction = np.asarray(operator.reaction_loads(refresh_state=False).values, dtype=float).reshape(-1)

        _restore_replay_initial_state(
            fluid=fluid,
            mesh=mesh_f,
            previous_payload=previous_payload,
            target_payload=target_payload,
            backend=str(args.backend),
        )
        _configure_checkpoint_bcs(
            operator=operator,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            time_s=float(np.asarray(target_payload.get("time_s", np.asarray(0.0))).reshape(-1)[0]),
            reference_velocity=float(reference_velocity),
        )
        initial_state = pack_fluid_state(operator)
        trial_space = trial_basis.make_trial_space(operator, offset=initial_state)
        initial_coefficients = np.zeros(int(trial_space.n_modes), dtype=float)
        prev_u = np.asarray(fluid["u_prev"].nodal_values, dtype=float).copy()
        prev_a = np.asarray(fluid["a_prev"].nodal_values, dtype=float).copy()
        bossak = _bossak_coefficients(alpha=float(args.bossak_alpha), dt=float(setup.boundaries.time_step))

        def update_acceleration(_coefficients: np.ndarray) -> None:
            fluid["a_k"].nodal_values[:] = (
                float(bossak["ma0"]) * (np.asarray(fluid["u_k"].nodal_values, dtype=float) - prev_u)
                + float(bossak["ma2"]) * prev_a
            )

        projected_target_coefficients = trial_basis.project_state(target_state, offset=initial_state)
        projected_target_state = trial_basis.reconstruct_state(projected_target_coefficients, offset=initial_state)
        write_fluid_state(operator, projected_target_state)
        update_acceleration(projected_target_coefficients)
        projected_target_reaction = np.asarray(
            operator.reaction_loads(refresh_state=True).values,
            dtype=float,
        ).reshape(-1)
        projected_target_residual = operator.assemble(
            need_matrix=False,
            convention="newton",
            refresh_predicted=True,
        )
        best_online_state_error = _relative_l2(target_state[free], projected_target_state[free])
        best_online_reaction_error = _relative_l2(target_reaction, projected_target_reaction)
        best_online_residual_norm = float(np.linalg.norm(projected_target_residual.residual[free]))

        _restore_replay_initial_state(
            fluid=fluid,
            mesh=mesh_f,
            previous_payload=previous_payload,
            target_payload=target_payload,
            backend=str(args.backend),
        )
        _configure_checkpoint_bcs(
            operator=operator,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            time_s=float(np.asarray(target_payload.get("time_s", np.asarray(0.0))).reshape(-1)[0]),
            reference_velocity=float(reference_velocity),
        )
        trial_space = trial_basis.make_trial_space(operator, offset=initial_state)

        t0 = time.perf_counter()
        full_system = FluidLSPGVerifier(operator=operator, trial_space=trial_space).assemble_system(
            initial_coefficients,
            refresh_predicted=True,
        )
        full_assembly_times.append(float(time.perf_counter() - t0))
        lspg_row_weights = None
        if bool(args.full_lspg_block_scale):
            lspg_row_weights = _fluid_block_row_weights(
                operator,
                row_dofs=trial_space.free_dofs,
                residual=full_system.residual,
            )

        full_lspg_summary = None
        if bool(args.run_full_lspg):
            _restore_replay_initial_state(
                fluid=fluid,
                mesh=mesh_f,
                previous_payload=previous_payload,
                target_payload=target_payload,
                backend=str(args.backend),
            )
            _configure_checkpoint_bcs(
                operator=operator,
                setup=setup,
                fluid_iface_coords=fluid_iface_coords,
                time_s=float(np.asarray(target_payload.get("time_s", np.asarray(0.0))).reshape(-1)[0]),
                reference_velocity=float(reference_velocity),
            )
            lspg_trial_space = trial_basis.make_trial_space(operator, offset=initial_state)
            lspg = FluidLSPGVerifier(
                operator=operator,
                trial_space=lspg_trial_space,
                row_weights=lspg_row_weights,
                state_update_hook=update_acceleration,
                nonlinear_update_hook=operator.update_oss_after_nonlinear_update,
            )
            t_lspg0 = time.perf_counter()
            lspg_result = lspg.solve(
                initial_coefficients,
                max_iterations=int(args.full_lspg_max_iterations),
                residual_tol=float(args.full_lspg_residual_tol),
                line_search=bool(args.full_lspg_line_search),
            )
            lspg_elapsed = float(time.perf_counter() - t_lspg0)
            lspg_state = pack_fluid_state(operator)
            lspg_reaction = np.asarray(operator.reaction_loads(refresh_state=False).values, dtype=float).reshape(-1)
            full_lspg_summary = {
                "state_error": _relative_l2(target_state[free], lspg_state[free]),
                "reaction_error": _relative_l2(target_reaction, lspg_reaction),
                "reaction_cosine": float(
                    np.dot(target_reaction, lspg_reaction)
                    / max(float(np.linalg.norm(target_reaction) * np.linalg.norm(lspg_reaction)), 1.0e-15)
                ),
                "iterations": int(lspg_result.iterations),
                "converged": bool(lspg_result.converged),
                "residual_norm": float(lspg_result.residual_norm),
                "elapsed_s": float(lspg_elapsed),
                "line_search": bool(args.full_lspg_line_search),
                "block_scale": bool(args.full_lspg_block_scale),
                "trajectory": [dict(item) for item in lspg_result.trajectory],
            }

        if bool(args.skip_gnat):
            replay_entries.append(
                {
                    "checkpoint": str(checkpoint_paths[target_index]),
                    "step": int(np.asarray(target_payload.get("step", np.asarray(target_index)), dtype=int).reshape(-1)[0]),
                    "initial_state_error": _relative_l2(target_state[free], initial_state[free]),
                    "best_online_projection_state_error": float(best_online_state_error),
                    "best_online_projection_reaction_error": float(best_online_reaction_error),
                    "best_online_projection_residual_norm": float(best_online_residual_norm),
                    "full_lspg": full_lspg_summary,
                    "full_initial_lspg_residual_norm": float(full_system.residual_norm),
                    "full_assembly_time_s": float(full_assembly_times[-1]),
                }
            )
            continue

        _restore_replay_initial_state(
            fluid=fluid,
            mesh=mesh_f,
            previous_payload=previous_payload,
            target_payload=target_payload,
            backend=str(args.backend),
        )
        _configure_checkpoint_bcs(
            operator=operator,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            time_s=float(np.asarray(target_payload.get("time_s", np.asarray(0.0))).reshape(-1)[0]),
            reference_velocity=float(reference_velocity),
        )
        trial_space = trial_basis.make_trial_space(operator, offset=initial_state)
        gnat = FluidGNATSolver(
            operator=operator,
            trial_space=trial_space,
            sample_set=sample_set,
            state_update_hook=update_acceleration,
            nonlinear_update_hook=operator.update_oss_after_nonlinear_update,
        )
        t0 = time.perf_counter()
        result = gnat.solve(
            initial_coefficients,
            max_iterations=int(args.gnat_max_iterations),
            residual_tol=float(args.gnat_residual_tol),
            line_search=bool(args.gnat_line_search),
        )
        gnat_elapsed = float(time.perf_counter() - t0)
        gnat_times.append(gnat_elapsed)
        rom_state = pack_fluid_state(operator)
        rom_reaction = np.asarray(operator.reaction_loads(refresh_state=False).values, dtype=float).reshape(-1)
        replay_entries.append(
            {
                "checkpoint": str(checkpoint_paths[target_index]),
                "step": int(np.asarray(target_payload.get("step", np.asarray(target_index)), dtype=int).reshape(-1)[0]),
                "initial_state_error": _relative_l2(target_state[free], initial_state[free]),
                "best_online_projection_state_error": float(best_online_state_error),
                "best_online_projection_reaction_error": float(best_online_reaction_error),
                "best_online_projection_residual_norm": float(best_online_residual_norm),
                "full_lspg": full_lspg_summary,
                "rom_state_error": _relative_l2(target_state[free], rom_state[free]),
                "rom_reaction_error": _relative_l2(target_reaction, rom_reaction),
                "reaction_cosine": float(
                    np.dot(target_reaction, rom_reaction)
                    / max(float(np.linalg.norm(target_reaction) * np.linalg.norm(rom_reaction)), 1.0e-15)
                ),
                "gnat_iterations": int(result.iterations),
                "gnat_converged": bool(result.converged),
                "gnat_line_search": bool(args.gnat_line_search),
                "gnat_estimated_residual_norm": float(result.residual_norm),
                "full_initial_lspg_residual_norm": float(full_system.residual_norm),
                "full_assembly_time_s": float(full_assembly_times[-1]),
                "gnat_solve_time_s": float(gnat_elapsed),
            }
        )

    median_full = float(np.median(full_assembly_times)) if full_assembly_times else float("nan")
    median_gnat = float(np.median(gnat_times)) if gnat_times else float("nan")
    summary: dict[str, Any] = {
        "checkpoints": [str(path) for path in checkpoint_paths],
        "n_snapshots": int(batch.n_snapshots),
        "train_indices": mode_sweep.train_indices.tolist(),
        "test_indices": mode_sweep.test_indices.tolist(),
        "mode_selection": {
            "method": "Tiba-style held-out validation, reaction-dominated score",
            "selected_velocity_modes": int(selected.velocity_modes),
            "selected_pressure_modes": int(selected.pressure_modes),
            "selected_total_modes": int(selected.total_modes),
            "selected_score": float(selected.score),
            "selected_state_error": float(selected.state_error),
            "selected_reaction_error": None if selected.reaction_error is None else float(selected.reaction_error),
            "entries": [entry.__dict__ for entry in mode_sweep.entries],
        },
        "gnat": {
            "residual_modes": int(sample_set.n_residual_modes),
            "sample_rows": int(sample_set.n_sample_rows),
            "sample_elements": int(sample_set.n_sample_elements),
            "sample_element_fraction": float(sample_set.n_sample_elements / max(int(operator.mesh.n_elements), 1)),
            "sampled_basis_rank": int(sample_set.sampled_basis_rank),
            "sampled_basis_condition": float(sample_set.sampled_basis_condition),
        },
        "coupled_replay": replay_entries,
        "timing": {
            "median_full_lspg_assembly_s": median_full,
            "median_gnat_solve_s": median_gnat,
            "median_operator_speedup": float(median_full / median_gnat) if median_gnat > 0.0 else float("nan"),
        },
    }
    dump_json(summary, Path(args.output))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
