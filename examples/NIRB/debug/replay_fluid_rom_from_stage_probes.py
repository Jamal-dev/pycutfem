from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from examples.NIRB.common import dump_json
from examples.NIRB.debug.replay_fluid_gnat_from_checkpoints import (
    _build_snapshot_batch,
    _checkpoint_paths,
    _make_operator,
)
from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.fluid_basis import fit_fluid_pod_trial_basis, fit_fluid_pod_trial_basis_from_state_matrix
from examples.NIRB.fluid_gnat import FluidGNATSolver, FluidGNATSampleSet, fit_fluid_gnat_sample_set
from examples.NIRB.fluid_lspg import FluidLSPGVerifier, pack_fluid_state, write_fluid_state
from examples.NIRB.fluid_stage_probes import (
    FluidStageProbe,
    build_stage_probe_batch,
    find_fluid_stage_probe_pairs,
    load_fluid_stage_probe,
    restore_fluid_stage_probe,
)
from examples.NIRB.run_example2_local import (
    _bossak_coefficients,
    _assemble_fluid_sampled_lspg_element_contributions_raw,
    _boundary_field_data,
    _build_fluid_problem,
    _fluid_interface_reaction_element_ids,
    _fluid_interface_velocity_dofs,
    _load_reference_partitioned_meshes,
)


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float).reshape(-1)
    cand = np.asarray(candidate, dtype=float).reshape(-1)
    return float(np.linalg.norm(cand - ref) / max(float(np.linalg.norm(ref)), 1.0e-15))


def _cosine(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float).reshape(-1)
    cand = np.asarray(candidate, dtype=float).reshape(-1)
    return float(np.dot(ref, cand) / max(float(np.linalg.norm(ref) * np.linalg.norm(cand)), 1.0e-15))


def _fluid_block_row_weights(
    operator,
    *,
    row_dofs: np.ndarray,
    residual: np.ndarray,
    floor: float = 1.0e-12,
    relative_floor: float = 0.0,
) -> np.ndarray:
    rows = np.asarray(row_dofs, dtype=int).reshape(-1)
    res = np.asarray(residual, dtype=float).reshape(-1)
    if int(res.size) != int(rows.size):
        raise ValueError("residual must be restricted to row_dofs for block scaling.")
    weights = np.ones(int(rows.size), dtype=float)
    velocity_rows = np.concatenate(
        [
            np.asarray(operator.dh.get_field_slice("ux"), dtype=int).reshape(-1),
            np.asarray(operator.dh.get_field_slice("uy"), dtype=int).reshape(-1),
        ]
    )
    pressure_rows = np.asarray(operator.dh.get_field_slice("p"), dtype=int).reshape(-1)
    block_data: list[tuple[np.ndarray, float]] = []
    for block in (velocity_rows, pressure_rows):
        row_mask = np.isin(rows, block)
        if not np.any(row_mask):
            continue
        rms = float(np.linalg.norm(res[row_mask]) / np.sqrt(max(int(np.count_nonzero(row_mask)), 1)))
        block_data.append((row_mask, rms))
    reference_rms = max((rms for _mask, rms in block_data), default=0.0)
    relative_floor_value = max(float(relative_floor), 0.0) * float(reference_rms)
    effective_floor = max(float(floor), float(relative_floor_value))
    for row_mask, rms in block_data:
        weights[row_mask] = 1.0 / max(rms, effective_floor) ** 2
    return weights


def _make_inlet_lookup(
    *,
    setup,
    step: int,
    dt: float,
    reference_velocity: float,
) -> Callable[[float, float], float]:
    time_s = float(step) * float(dt)

    def inlet_lookup(x: float, y: float) -> float:
        del x
        return setup.geometry.inlet_velocity(float(y), time_s, reference_velocity=float(reference_velocity))

    return inlet_lookup


def _make_probe_inlet_lookup(
    *,
    setup,
    probe: FluidStageProbe,
    dt: float,
    reference_velocity: float,
) -> Callable[[float, float], float]:
    return _make_inlet_lookup(
        setup=setup,
        step=int(probe.step),
        dt=float(dt),
        reference_velocity=float(reference_velocity),
    )


def _make_stage_acceleration_hook(
    *,
    fluid: dict[str, object],
    bossak: dict[str, float],
    preserve_seed_on_first_zero: bool,
) -> Callable[[np.ndarray], None]:
    prev_u = np.asarray(fluid["u_prev"].nodal_values, dtype=float).copy()
    prev_a = np.asarray(fluid["a_prev"].nodal_values, dtype=float).copy()
    seed_a = np.asarray(fluid["a_k"].nodal_values, dtype=float).copy()
    first = True

    def update_acceleration(coefficients: np.ndarray) -> None:
        nonlocal first
        coeffs = np.asarray(coefficients, dtype=float).reshape(-1)
        if bool(first) and bool(preserve_seed_on_first_zero) and float(np.linalg.norm(coeffs)) <= 1.0e-14:
            fluid["a_k"].nodal_values[:] = seed_a
            first = False
            return
        fluid["a_k"].nodal_values[:] = (
            float(bossak["ma0"]) * (np.asarray(fluid["u_k"].nodal_values, dtype=float) - prev_u)
            + float(bossak["ma2"]) * prev_a
        )
        first = False

    return update_acceleration


def _force_stage_acceleration(*, fluid: dict[str, object], bossak: dict[str, float]) -> None:
    fluid["a_k"].nodal_values[:] = (
        float(bossak["ma0"])
        * (np.asarray(fluid["u_k"].nodal_values, dtype=float) - np.asarray(fluid["u_prev"].nodal_values, dtype=float))
        + float(bossak["ma2"]) * np.asarray(fluid["a_prev"].nodal_values, dtype=float)
    )


def _configure_and_restore_probe(
    *,
    operator,
    probe: FluidStageProbe,
    setup,
    fluid_iface_coords: np.ndarray,
    dt: float,
    reference_velocity: float,
) -> None:
    restore_fluid_stage_probe(
        operator,
        probe,
        fluid_iface_coords=fluid_iface_coords,
        inlet_lookup=_make_probe_inlet_lookup(
            setup=setup,
            probe=probe,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        ),
        apply_bcs_to_state=True,
    )


def _collect_residual_snapshots_from_coefficients(
    *,
    operator,
    trial_space,
    coefficients: list[np.ndarray],
    row_dofs: np.ndarray,
    state_update_hook: Callable[[np.ndarray], None],
) -> np.ndarray:
    rows = np.asarray(row_dofs, dtype=int).reshape(-1)
    residuals: list[np.ndarray] = []
    verifier = FluidLSPGVerifier(
        operator=operator,
        trial_space=trial_space,
        row_dofs=rows,
        state_update_hook=state_update_hook,
        nonlinear_update_hook=operator.update_oss_after_nonlinear_update,
    )
    for coeffs in coefficients:
        system = verifier.assemble_system(np.asarray(coeffs, dtype=float).reshape(-1), refresh_predicted=True)
        residuals.append(np.asarray(system.residual, dtype=float).reshape(-1))
    if not residuals:
        raise ValueError("Cannot collect residual snapshots from an empty coefficient list.")
    return np.column_stack(residuals)


def _coefficient_training_path(
    initial_coefficients: np.ndarray,
    trajectory: tuple[dict[str, float], ...],
) -> list[np.ndarray]:
    coeffs: list[np.ndarray] = [np.asarray(initial_coefficients, dtype=float).reshape(-1).copy()]
    for item in trajectory:
        if "coefficients" not in item:
            continue
        current = np.asarray(item["coefficients"], dtype=float).reshape(-1)
        if current.size != coeffs[0].size:
            raise ValueError("LSPG trajectory coefficient size does not match the trial space.")
        if float(np.linalg.norm(current - coeffs[-1])) > 1.0e-14:
            coeffs.append(current.copy())
    return coeffs


def _collect_training_residual_snapshots_from_probe_pairs(
    *,
    operator,
    fluid: dict[str, object],
    pairs,
    trial_basis,
    setup,
    fluid_iface_coords: np.ndarray,
    dt: float,
    reference_velocity: float,
    bossak: dict[str, float],
    free_dofs: np.ndarray,
    max_iterations: int,
    residual_tol: float,
    line_search: bool,
    block_scale: bool,
    block_scale_relative_floor: float = 0.0,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    residual_columns: list[np.ndarray] = []
    entries: list[dict[str, Any]] = []
    free = np.asarray(free_dofs, dtype=int).reshape(-1)

    for pair in pairs:
        pre_probe = load_fluid_stage_probe(pair.pre_path)
        post_probe = load_fluid_stage_probe(pair.post_path)

        _configure_and_restore_probe(
            operator=operator,
            probe=post_probe,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        )
        target_state = pack_fluid_state(operator)
        target_reaction_lookup = post_probe.reaction_lookup(kind="point")
        restored_target_reaction = np.asarray(operator.reaction_loads(refresh_state=False).values, dtype=float).reshape(-1)
        target_reaction = (
            restored_target_reaction.copy()
            if target_reaction_lookup is None
            else np.asarray(target_reaction_lookup.values, dtype=float).reshape(-1)
        )

        _configure_and_restore_probe(
            operator=operator,
            probe=pre_probe,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        )
        initial_state = pack_fluid_state(operator)
        trial_space = trial_basis.make_trial_space(operator, offset=initial_state)
        initial_coefficients = np.zeros(int(trial_space.n_modes), dtype=float)
        row_weights = None
        if bool(block_scale):
            initial_hook = _make_stage_acceleration_hook(
                fluid=fluid,
                bossak=bossak,
                preserve_seed_on_first_zero=True,
            )
            initial_verifier = FluidLSPGVerifier(
                operator=operator,
                trial_space=trial_space,
                state_update_hook=initial_hook,
                nonlinear_update_hook=operator.update_oss_after_nonlinear_update,
            )
            initial_system = initial_verifier.assemble_system(initial_coefficients, refresh_predicted=True)
            row_weights = _fluid_block_row_weights(
                operator,
                row_dofs=trial_space.free_dofs,
                residual=initial_system.residual,
                relative_floor=float(block_scale_relative_floor),
            )
            _configure_and_restore_probe(
                operator=operator,
                probe=pre_probe,
                setup=setup,
                fluid_iface_coords=fluid_iface_coords,
                dt=float(dt),
                reference_velocity=float(reference_velocity),
            )
        lspg_hook = _make_stage_acceleration_hook(
            fluid=fluid,
            bossak=bossak,
            preserve_seed_on_first_zero=True,
        )
        lspg = FluidLSPGVerifier(
            operator=operator,
            trial_space=trial_space,
            row_weights=row_weights,
            state_update_hook=lspg_hook,
            nonlinear_update_hook=operator.update_oss_after_nonlinear_update,
        )
        t0 = time.perf_counter()
        lspg_result = lspg.solve(
            initial_coefficients,
            max_iterations=int(max_iterations),
            residual_tol=float(residual_tol),
            line_search=bool(line_search),
        )
        elapsed = float(time.perf_counter() - t0)
        lspg_state = pack_fluid_state(operator)
        lspg_reaction = np.asarray(operator.reaction_loads(refresh_state=False).values, dtype=float).reshape(-1)

        trajectory_coefficients = _coefficient_training_path(initial_coefficients, lspg_result.trajectory)
        _configure_and_restore_probe(
            operator=operator,
            probe=pre_probe,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        )
        residual_hook = _make_stage_acceleration_hook(
            fluid=fluid,
            bossak=bossak,
            preserve_seed_on_first_zero=True,
        )
        residual_matrix = _collect_residual_snapshots_from_coefficients(
            operator=operator,
            trial_space=trial_space,
            coefficients=trajectory_coefficients,
            row_dofs=free,
            state_update_hook=residual_hook,
        )
        residual_columns.append(residual_matrix)
        entries.append(
            {
                "step": int(pair.step),
                "coupling_iter": int(pair.coupling_iter),
                "pre_probe": str(pair.pre_path),
                "post_probe": str(pair.post_path),
                "full_lspg_state_error": _relative_l2(target_state[free], lspg_state[free]),
                "full_lspg_reaction_error": _relative_l2(target_reaction, lspg_reaction),
                "full_lspg_reaction_cosine": _cosine(target_reaction, lspg_reaction),
                "full_lspg_iterations": int(lspg_result.iterations),
                "full_lspg_converged": bool(lspg_result.converged),
                "full_lspg_residual_norm": float(lspg_result.residual_norm),
                "full_lspg_elapsed_s": float(elapsed),
                "full_lspg_block_scale": bool(block_scale),
                "residual_snapshot_columns": int(residual_matrix.shape[1]),
            }
        )

    return residual_columns, entries


def _parse_steps(spec: str | None) -> list[int] | None:
    if spec is None or not str(spec).strip():
        return None
    values: list[int] = []
    for chunk in str(spec).split(","):
        text = chunk.strip()
        if not text:
            continue
        if ":" in text:
            left, right = text.split(":", 1)
            values.extend(range(int(left), int(right) + 1))
        else:
            values.append(int(text))
    return values


def _fit_increment_basis_from_probe_pairs(
    *,
    operator,
    pairs,
    setup,
    fluid_iface_coords: np.ndarray,
    dt: float,
    reference_velocity: float,
    velocity_modes: int,
    pressure_modes: int,
):
    deltas: list[np.ndarray] = []
    free_dofs = None
    sources: list[str] = []
    for pair in pairs:
        pre_probe = load_fluid_stage_probe(pair.pre_path)
        post_probe = load_fluid_stage_probe(pair.post_path)
        _configure_and_restore_probe(
            operator=operator,
            probe=pre_probe,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        )
        initial = pack_fluid_state(operator)
        free_dofs = operator.free_fluid_dofs() if free_dofs is None else free_dofs
        _configure_and_restore_probe(
            operator=operator,
            probe=post_probe,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        )
        target = pack_fluid_state(operator)
        deltas.append(np.asarray(target - initial, dtype=float).reshape(-1))
        sources.append(str(pair.post_path))
    if not deltas:
        raise ValueError("Need at least one stage probe pair to fit an increment basis.")
    return (
        fit_fluid_pod_trial_basis_from_state_matrix(
            operator,
            np.column_stack(deltas),
            free_dofs=np.asarray(free_dofs, dtype=int),
            velocity_modes=int(velocity_modes),
            pressure_modes=int(pressure_modes),
            center=False,
        ),
        sources,
    )


def _weighted_sample_lift(sample_basis: np.ndarray, sample_weights: np.ndarray) -> np.ndarray:
    basis = np.asarray(sample_basis, dtype=float)
    weights = np.asarray(sample_weights, dtype=float).reshape(-1)
    if basis.ndim != 2 or int(weights.size) != int(basis.shape[0]):
        raise ValueError("sample_basis/sample_weights have incompatible shapes.")
    sqrt_weights = np.sqrt(np.maximum(weights, 0.0))
    weighted_basis = basis * sqrt_weights[:, None]
    return np.linalg.pinv(weighted_basis) * sqrt_weights[None, :]


def _sample_set_with_rows_and_elements(
    sample_set: FluidGNATSampleSet,
    *,
    row_dofs: np.ndarray,
    row_weights: np.ndarray,
    element_ids: np.ndarray,
    element_weights: np.ndarray,
    element_weighting: str,
    element_weight_fit_relative_error: float,
) -> FluidGNATSampleSet:
    rows = np.asarray(row_dofs, dtype=int).reshape(-1)
    weights = np.asarray(row_weights, dtype=float).reshape(-1)
    elems = np.asarray(element_ids, dtype=int).reshape(-1)
    elem_weights = np.asarray(element_weights, dtype=float).reshape(-1)
    sample_basis = np.asarray(sample_set.residual_basis[rows, :], dtype=float)
    singular_values = np.linalg.svd(sample_basis, compute_uv=False)
    rank = int(np.linalg.matrix_rank(sample_basis))
    condition = float("inf") if singular_values.size == 0 or singular_values[-1] == 0.0 else float(
        singular_values[0] / singular_values[-1]
    )
    return FluidGNATSampleSet(
        residual_basis=np.asarray(sample_set.residual_basis, dtype=float),
        residual_singular_values=np.asarray(sample_set.residual_singular_values, dtype=float),
        residual_energy_fraction=np.asarray(sample_set.residual_energy_fraction, dtype=float),
        basis_dofs=np.asarray(sample_set.basis_dofs, dtype=int),
        row_dofs=rows,
        element_ids=elems,
        sample_to_residual_coefficients=_weighted_sample_lift(sample_basis, weights),
        sampled_basis_rank=rank,
        sampled_basis_condition=condition,
        sample_weights=weights,
        sample_weighting=str(sample_set.sample_weighting),
        sample_weight_fit_relative_error=float(sample_set.sample_weight_fit_relative_error),
        element_weights=elem_weights,
        element_weighting=str(element_weighting),
        element_weight_fit_relative_error=float(element_weight_fit_relative_error),
    )


def _sampled_rows_touched_by_elements(operator, *, element_ids: np.ndarray, rows: np.ndarray) -> np.ndarray:
    elems = np.asarray(element_ids, dtype=int).reshape(-1)
    row_values = np.asarray(rows, dtype=int).reshape(-1)
    if elems.size == 0 or row_values.size == 0:
        return np.zeros(0, dtype=int)
    touched: list[np.ndarray] = []
    for field_name in ("ux", "uy", "p"):
        touched.append(np.asarray(operator.dh.element_maps[str(field_name)], dtype=int)[elems].reshape(-1))
    element_rows = np.unique(np.concatenate(touched))
    element_rows = element_rows[element_rows >= 0]
    return row_values[np.isin(row_values, element_rows)]


def _fit_bounded_nonnegative_weights(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    max_weight: float,
) -> tuple[np.ndarray, float]:
    A = np.asarray(matrix, dtype=float)
    b = np.asarray(target, dtype=float).reshape(-1)
    if A.ndim != 2 or int(A.shape[0]) != int(b.size):
        raise ValueError("Empirical cubature fit matrix and target have incompatible shapes.")
    if A.shape[1] == 0:
        raise ValueError("Cannot fit element weights with zero candidate elements.")
    column_norms = np.linalg.norm(A, axis=0)
    active = column_norms > 1.0e-14
    if not np.any(active):
        return np.ones(int(A.shape[1]), dtype=float), float("inf")
    A_active = A[:, active] / column_norms[active][None, :]
    upper = max(float(max_weight), 1.0e-12) * column_norms[active]
    try:
        from scipy.optimize import lsq_linear

        result = lsq_linear(
            A_active,
            b,
            bounds=(np.zeros(int(A_active.shape[1]), dtype=float), upper),
            lsmr_tol="auto",
            max_iter=300,
        )
        scaled = np.asarray(result.x, dtype=float).reshape(-1)
    except Exception:
        scaled, *_ = np.linalg.lstsq(A_active, b, rcond=None)
        scaled = np.clip(np.asarray(scaled, dtype=float).reshape(-1), 0.0, upper)
    weights = np.zeros(int(A.shape[1]), dtype=float)
    weights[active] = scaled / column_norms[active]
    weights = np.clip(weights, 0.0, max(float(max_weight), 1.0e-12))
    residual = A @ weights - b
    rel_error = float(np.linalg.norm(residual) / max(float(np.linalg.norm(b)), 1.0e-15))
    return weights, rel_error


def _fit_element_cubature_sample_set(
    *,
    operator,
    fluid: dict[str, object],
    pairs,
    trial_basis,
    sample_set: FluidGNATSampleSet,
    setup,
    fluid_iface_coords: np.ndarray,
    dt: float,
    reference_velocity: float,
    bossak: dict[str, float],
    max_iterations: int,
    residual_tol: float,
    line_search: bool,
    block_scale: bool,
    block_scale_relative_floor: float,
    max_training_states: int,
    max_weight: float,
    prune_tol: float,
    keep_interface_elements: bool,
) -> tuple[FluidGNATSampleSet, dict[str, Any]]:
    rows = np.asarray(sample_set.row_dofs, dtype=int).reshape(-1)
    elements = np.asarray(sample_set.element_ids, dtype=int).reshape(-1)
    if rows.size == 0 or elements.size == 0:
        return sample_set, {"enabled": False, "reason": "empty rows or elements"}

    blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    entries: list[dict[str, Any]] = []
    states_used = 0
    p = operator.parameters

    for pair in pairs:
        if states_used >= int(max_training_states):
            break
        pre_probe = load_fluid_stage_probe(pair.pre_path)
        post_probe = load_fluid_stage_probe(pair.post_path)

        _configure_and_restore_probe(
            operator=operator,
            probe=post_probe,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        )
        target_state = pack_fluid_state(operator)

        _configure_and_restore_probe(
            operator=operator,
            probe=pre_probe,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        )
        initial_state = pack_fluid_state(operator)
        trial_space = trial_basis.make_trial_space(operator, offset=initial_state)
        initial_coefficients = np.zeros(int(trial_space.n_modes), dtype=float)
        coefficients: list[np.ndarray] = [initial_coefficients]

        row_weights = None
        if bool(block_scale):
            initial_hook = _make_stage_acceleration_hook(
                fluid=fluid,
                bossak=bossak,
                preserve_seed_on_first_zero=True,
            )
            initial_verifier = FluidLSPGVerifier(
                operator=operator,
                trial_space=trial_space,
                state_update_hook=initial_hook,
                nonlinear_update_hook=operator.update_oss_after_nonlinear_update,
            )
            initial_system = initial_verifier.assemble_system(initial_coefficients, refresh_predicted=True)
            row_weights = _fluid_block_row_weights(
                operator,
                row_dofs=trial_space.free_dofs,
                residual=initial_system.residual,
                relative_floor=float(block_scale_relative_floor),
            )
            _configure_and_restore_probe(
                operator=operator,
                probe=pre_probe,
                setup=setup,
                fluid_iface_coords=fluid_iface_coords,
                dt=float(dt),
                reference_velocity=float(reference_velocity),
            )
            trial_space = trial_basis.make_trial_space(operator, offset=initial_state)

        lspg_hook = _make_stage_acceleration_hook(
            fluid=fluid,
            bossak=bossak,
            preserve_seed_on_first_zero=True,
        )
        lspg = FluidLSPGVerifier(
            operator=operator,
            trial_space=trial_space,
            row_weights=row_weights,
            state_update_hook=lspg_hook,
            nonlinear_update_hook=operator.update_oss_after_nonlinear_update,
        )
        result = lspg.solve(
            initial_coefficients,
            max_iterations=int(max_iterations),
            residual_tol=float(residual_tol),
            line_search=bool(line_search),
        )
        coefficients.extend(_coefficient_training_path(initial_coefficients, result.trajectory)[1:])
        projected = trial_basis.project_state(target_state, offset=initial_state)
        if not coefficients or float(np.linalg.norm(projected - coefficients[-1])) > 1.0e-14:
            coefficients.append(np.asarray(projected, dtype=float).reshape(-1))

        for coeffs in coefficients:
            if states_used >= int(max_training_states):
                break
            _configure_and_restore_probe(
                operator=operator,
                probe=pre_probe,
                setup=setup,
                fluid_iface_coords=fluid_iface_coords,
                dt=float(dt),
                reference_velocity=float(reference_velocity),
            )
            trial_space = trial_basis.make_trial_space(operator, offset=initial_state)
            hook = _make_stage_acceleration_hook(
                fluid=fluid,
                bossak=bossak,
                preserve_seed_on_first_zero=True,
            )
            trial_space.write(operator, coeffs)
            hook(np.asarray(coeffs, dtype=float).reshape(-1))
            operator.refresh_predicted_subscale()
            full = operator.assemble(need_matrix=False, convention="newton", refresh_predicted=False)
            target = np.asarray(full.residual[rows], dtype=float).reshape(-1)
            _ids, residual_by_element, _trial_by_element = _assemble_fluid_sampled_lspg_element_contributions_raw(
                prob=operator.prob,
                rho_f=float(p.rho_f),
                mu_f=float(p.mu_f),
                dt=float(p.dt),
                quad_order=int(p.quadrature_order),
                bossak_alpha=float(p.bossak_alpha),
                contribution_mode=str(p.contribution_mode),
                backend=str(p.backend),
                element_ids=elements,
                row_dofs=rows,
                basis=np.asarray(trial_basis.basis, dtype=float),
            )
            block_scale_value = 1.0 / max(float(np.linalg.norm(target)), 1.0e-12)
            blocks.append(np.asarray(residual_by_element.T, dtype=float) * block_scale_value)
            targets.append(target * block_scale_value)
            states_used += 1

        entries.append(
            {
                "step": int(pair.step),
                "coupling_iter": int(pair.coupling_iter),
                "lspg_iterations": int(result.iterations),
                "lspg_converged": bool(result.converged),
                "states_after_pair": int(states_used),
            }
        )

    if not blocks:
        return sample_set, {"enabled": False, "reason": "no training states"}

    fit_matrix = np.vstack(blocks)
    fit_target = np.concatenate(targets)
    weights, fit_error = _fit_bounded_nonnegative_weights(
        fit_matrix,
        fit_target,
        max_weight=float(max_weight),
    )
    if not np.any(weights > 0.0):
        weights[:] = 1.0
        fit_error = float("inf")

    keep = weights > max(float(prune_tol), 0.0)
    if bool(keep_interface_elements):
        interface_elements = _fluid_interface_reaction_element_ids(
            operator.prob,
            interface_tag=operator.boundary_tags.interface_tag,
        )
        keep |= np.isin(elements, interface_elements)
    if not np.any(keep):
        keep = np.ones(int(elements.size), dtype=bool)
    kept_elements = elements[keep]
    kept_element_weights = weights[keep]
    kept_rows = _sampled_rows_touched_by_elements(operator, element_ids=kept_elements, rows=rows)
    row_keep = np.isin(rows, kept_rows)
    min_rows = max(int(trial_basis.n_modes), int(sample_set.n_residual_modes), 1)
    if int(np.count_nonzero(row_keep)) < min_rows:
        kept_elements = elements
        kept_element_weights = weights
        row_keep = np.ones(int(rows.size), dtype=bool)
    cubature_set = _sample_set_with_rows_and_elements(
        sample_set,
        row_dofs=rows[row_keep],
        row_weights=np.asarray(sample_set.sample_weights, dtype=float).reshape(-1)[row_keep],
        element_ids=kept_elements,
        element_weights=kept_element_weights,
        element_weighting="residual-lsq-bounded-pruned" if np.any(~keep) else "residual-lsq-bounded",
        element_weight_fit_relative_error=float(fit_error),
    )
    info = {
        "enabled": True,
        "training_states": int(states_used),
        "fit_rows": int(fit_matrix.shape[0]),
        "candidate_elements": int(elements.size),
        "kept_elements": int(cubature_set.n_sample_elements),
        "candidate_rows": int(rows.size),
        "kept_rows": int(cubature_set.n_sample_rows),
        "fit_relative_error": float(fit_error),
        "weight_min": float(np.min(cubature_set.element_weights)) if cubature_set.element_weights.size else float("nan"),
        "weight_max": float(np.max(cubature_set.element_weights)) if cubature_set.element_weights.size else float("nan"),
        "weight_sum": float(np.sum(cubature_set.element_weights)),
        "prune_tol": float(prune_tol),
        "max_weight": float(max_weight),
        "keep_interface_elements": bool(keep_interface_elements),
        "entries": entries,
    }
    return cubature_set, info


def _interface_forced_row_dofs(operator, *, mode: str) -> np.ndarray | None:
    """Rows that anchor sampled residual fitting to the coupled reaction path."""

    key = str(mode).strip().lower().replace("-", "_")
    if key in {"", "none", "off", "false"}:
        return None
    rows: list[np.ndarray] = []
    if key in {"velocity", "all", "velocity_pressure", "pressure_velocity"}:
        rows.append(
            np.asarray(
                _fluid_interface_velocity_dofs(
                    operator.prob,
                    interface_tag=operator.boundary_tags.interface_tag,
                ),
                dtype=int,
            ).reshape(-1)
        )
    if key in {"pressure", "all", "velocity_pressure", "pressure_velocity"}:
        rows.append(
            np.asarray(
                _boundary_field_data(operator.dh, "p", operator.boundary_tags.interface_tag)[1],
                dtype=int,
            ).reshape(-1)
        )
    if not rows:
        raise ValueError(f"Unsupported --force-interface-rows value {mode!r}.")
    values = np.concatenate([row for row in rows if row.size])
    if values.size == 0:
        return None
    free = np.asarray(operator.free_fluid_dofs(), dtype=int).reshape(-1)
    forced = np.intersect1d(values, free).astype(int, copy=False)
    return forced if forced.size else None


def _save_sampled_lspg_hrom_model(
    path: Path,
    *,
    trial_basis,
    sample_set,
    args: argparse.Namespace,
    training_source: list[str],
    training_steps: list[int],
) -> None:
    """Persist the deployable arrays needed by `run_example2_local.py`."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        schema_version=np.asarray(1, dtype=int),
        basis=np.asarray(trial_basis.basis, dtype=float),
        free_dofs=np.asarray(trial_basis.free_dofs, dtype=int),
        velocity_dofs=np.asarray(trial_basis.velocity_dofs, dtype=int),
        pressure_dofs=np.asarray(trial_basis.pressure_dofs, dtype=int),
        velocity_singular_values=np.asarray(trial_basis.velocity_singular_values, dtype=float),
        pressure_singular_values=np.asarray(trial_basis.pressure_singular_values, dtype=float),
        velocity_energy_fraction=np.asarray(trial_basis.velocity_energy_fraction, dtype=float),
        pressure_energy_fraction=np.asarray(trial_basis.pressure_energy_fraction, dtype=float),
        velocity_modes=np.asarray(int(args.velocity_modes), dtype=int),
        pressure_modes=np.asarray(int(args.pressure_modes), dtype=int),
        sample_residual_basis=np.asarray(sample_set.residual_basis, dtype=float),
        sample_residual_singular_values=np.asarray(sample_set.residual_singular_values, dtype=float),
        sample_residual_energy_fraction=np.asarray(sample_set.residual_energy_fraction, dtype=float),
        sample_basis_dofs=np.asarray(sample_set.basis_dofs, dtype=int),
        sample_row_dofs=np.asarray(sample_set.row_dofs, dtype=int),
        sample_element_ids=np.asarray(sample_set.element_ids, dtype=int),
        sample_to_residual_coefficients=np.asarray(sample_set.sample_to_residual_coefficients, dtype=float),
        sample_weights=np.asarray(sample_set.sample_weights, dtype=float),
        sample_weighting=np.asarray(str(sample_set.sample_weighting)),
        force_interface_rows=np.asarray(str(args.force_interface_rows)),
        sample_weight_fit_relative_error=np.asarray(float(sample_set.sample_weight_fit_relative_error), dtype=float),
        sample_element_weights=np.asarray(sample_set.element_weights, dtype=float),
        sample_element_weighting=np.asarray(str(sample_set.element_weighting)),
        sample_element_weight_fit_relative_error=np.asarray(
            float(sample_set.element_weight_fit_relative_error),
            dtype=float,
        ),
        sampled_basis_rank=np.asarray(int(sample_set.sampled_basis_rank), dtype=int),
        sampled_basis_condition=np.asarray(float(sample_set.sampled_basis_condition), dtype=float),
        objective=np.asarray(str(args.gnat_objective)),
        basis_kind=np.asarray(str(args.basis_kind)),
        lspg_block_scale=np.asarray(bool(args.lspg_block_scale), dtype=bool),
        lspg_block_scale_relative_floor=np.asarray(float(args.lspg_block_scale_relative_floor), dtype=float),
        max_iterations=np.asarray(int(args.gnat_max_iterations), dtype=int),
        residual_tol=np.asarray(float(args.gnat_residual_tol), dtype=float),
        line_search=np.asarray(bool(args.gnat_line_search), dtype=bool),
        recommended_switch_iter=np.asarray(4, dtype=int),
        training_sources=np.asarray([str(item) for item in training_source]),
        training_steps=np.asarray(training_steps, dtype=int),
        training_all_iters=np.asarray(bool(args.training_all_iters), dtype=bool),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay fluid LSPG/GNAT from exact pre/post fluid stage probes.")
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--steps", type=str, default="10:12")
    parser.add_argument(
        "--replay-all-iters",
        action="store_true",
        help="Replay every paired coupling iteration for the selected steps instead of only each step's final pair.",
    )
    parser.add_argument(
        "--training-checkpoints",
        type=Path,
        default=Path("examples/NIRB/artifacts/nirb_example2_local_long_20260421_190524/checkpoints"),
    )
    parser.add_argument("--training-probe-dir", type=Path, default=None)
    parser.add_argument(
        "--training-steps",
        type=str,
        default=None,
        help="Comma/range step list used for stage-probe basis fitting and GNAT residual training.",
    )
    parser.add_argument(
        "--training-all-iters",
        action="store_true",
        help="Train from every paired coupling iteration for the selected training steps.",
    )
    parser.add_argument("--basis-kind", choices=("absolute", "increment"), default="absolute")
    parser.add_argument("--max-training-snapshots", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default="cpp")
    parser.add_argument("--quad-order", type=int, default=1)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    parser.add_argument("--dynamic-tau", type=float, default=1.0)
    parser.add_argument("--reference-velocity", type=float, default=None)
    parser.add_argument("--velocity-modes", type=int, default=3)
    parser.add_argument("--pressure-modes", type=int, default=5)
    parser.add_argument("--lspg-max-iterations", type=int, default=10)
    parser.add_argument("--lspg-residual-tol", type=float, default=1.0e-8)
    parser.add_argument("--lspg-line-search", action="store_true")
    parser.add_argument("--lspg-block-scale", action="store_true")
    parser.add_argument(
        "--lspg-block-scale-relative-floor",
        type=float,
        default=0.0,
        help=(
            "Relative RMS floor for mixed velocity/pressure block scaling. "
            "For example, 1e-3 caps the block weight ratio at about 1e6."
        ),
    )
    parser.add_argument("--skip-gnat", action="store_true")
    parser.add_argument("--residual-modes", type=int, default=None)
    parser.add_argument("--row-oversampling", type=float, default=4.0)
    parser.add_argument("--min-sample-rows", type=int, default=64)
    parser.add_argument(
        "--gnat-sample-weighting",
        choices=("none", "basis-gram", "snapshot-gram", "basis-gram-bounded", "snapshot-gram-bounded"),
        default="none",
    )
    parser.add_argument(
        "--gnat-element-weighting",
        choices=("none", "residual-lsq-bounded"),
        default="none",
        help="Fit nonnegative element cubature weights against full residual rows after selecting the sample mesh.",
    )
    parser.add_argument(
        "--gnat-element-max-training-states",
        type=int,
        default=24,
        help="Maximum reduced states used in the element cubature least-squares fit.",
    )
    parser.add_argument(
        "--gnat-element-weight-max",
        type=float,
        default=100.0,
        help="Upper bound for each element cubature weight.",
    )
    parser.add_argument(
        "--gnat-element-prune-tol",
        type=float,
        default=1.0e-8,
        help="Drop sample elements whose fitted cubature weight is at or below this threshold.",
    )
    parser.add_argument(
        "--gnat-element-keep-interface",
        action="store_true",
        help="Always retain interface-reaction elements after empirical-cubature pruning.",
    )
    parser.add_argument(
        "--force-interface-rows",
        choices=("none", "velocity", "pressure", "all"),
        default="none",
        help=(
            "Force interface residual rows into the sampled-LSPG objective. "
            "Use 'all' for v2 reaction-aware training."
        ),
    )
    parser.add_argument("--gnat-objective", choices=("gnat", "sampled_lspg"), default="gnat")
    parser.add_argument("--gnat-max-iterations", type=int, default=10)
    parser.add_argument("--gnat-residual-tol", type=float, default=1.0e-8)
    parser.add_argument("--gnat-line-search", action="store_true")
    parser.add_argument(
        "--save-hrom-model",
        type=Path,
        default=None,
        help="Optional .npz path for the deployable sampled-LSPG HROM model used by run_example2_local.py.",
    )
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
    dt = float(setup.boundaries.time_step)
    bossak = _bossak_coefficients(alpha=float(args.bossak_alpha), dt=float(dt))
    training_steps = _parse_steps(args.training_steps)
    gnat_training_pairs = []

    if str(args.basis_kind) == "increment":
        increment_source_dir = args.training_probe_dir if args.training_probe_dir is not None else args.probe_dir
        train_pairs = find_fluid_stage_probe_pairs(
            increment_source_dir,
            final_only=not bool(args.training_all_iters),
            steps=training_steps,
        )
        gnat_training_pairs = list(train_pairs)
        trial_basis, training_source = _fit_increment_basis_from_probe_pairs(
            operator=operator,
            pairs=train_pairs,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
            velocity_modes=int(args.velocity_modes),
            pressure_modes=int(args.pressure_modes),
        )
        training_kind = (
            "stage-increment-probe"
            if args.training_probe_dir is not None or args.training_steps is not None
            else "stage-increment-probe-oracle"
        )
        train_batch = None
    elif args.training_probe_dir is not None:
        train_pairs = find_fluid_stage_probe_pairs(
            args.training_probe_dir,
            final_only=not bool(args.training_all_iters),
            steps=training_steps,
        )
        gnat_training_pairs = list(train_pairs)
        train_probes = [load_fluid_stage_probe(pair.post_path) for pair in train_pairs]
        train_batch = build_stage_probe_batch(
            operator,
            train_probes,
            fluid_iface_coords=fluid_iface_coords,
            inlet_lookup_factory=lambda probe: _make_probe_inlet_lookup(
                setup=setup,
                probe=probe,
                dt=float(dt),
                reference_velocity=float(reference_velocity),
            ),
        )
        training_source = [str(pair.post_path) for pair in train_pairs]
        training_kind = "stage-probe-post-fluid-solve"
    else:
        checkpoint_paths = _checkpoint_paths(args.training_checkpoints, max_snapshots=int(args.max_training_snapshots))
        train_batch = _build_snapshot_batch(
            operator=operator,
            setup=setup,
            fluid=fluid,
            checkpoint_paths=checkpoint_paths,
            fluid_iface_coords=fluid_iface_coords,
            reference_velocity=float(reference_velocity),
        )
        training_source = [str(path) for path in checkpoint_paths]
        training_kind = "accepted-checkpoint"
    if str(args.basis_kind) != "increment":
        trial_basis = fit_fluid_pod_trial_basis(
            operator,
            train_batch,
            velocity_modes=int(args.velocity_modes),
            pressure_modes=int(args.pressure_modes),
            center=True,
        )
        free = np.asarray(train_batch.free_dofs, dtype=int)
    else:
        free = np.asarray(trial_basis.free_dofs, dtype=int)

    stage_pairs = find_fluid_stage_probe_pairs(
        args.probe_dir,
        final_only=not bool(args.replay_all_iters),
        steps=_parse_steps(str(args.steps)),
    )
    replay_entries: list[dict[str, Any]] = []
    full_assembly_times: list[float] = []
    lspg_times: list[float] = []
    gnat_times: list[float] = []
    residual_training_columns: list[np.ndarray] = []
    residual_training_entries: list[dict[str, Any]] = []
    saved_hrom_model_path: str | None = None
    training_pair_steps = [int(pair.step) for pair in gnat_training_pairs]
    if not bool(args.skip_gnat) and gnat_training_pairs:
        residual_training_columns, residual_training_entries = _collect_training_residual_snapshots_from_probe_pairs(
            operator=operator,
            fluid=fluid,
            pairs=gnat_training_pairs,
            trial_basis=trial_basis,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
            bossak=bossak,
            free_dofs=free,
            max_iterations=int(args.lspg_max_iterations),
            residual_tol=float(args.lspg_residual_tol),
            line_search=bool(args.lspg_line_search),
            block_scale=bool(args.lspg_block_scale),
            block_scale_relative_floor=float(args.lspg_block_scale_relative_floor),
        )

    for pair in stage_pairs:
        pre_probe = load_fluid_stage_probe(pair.pre_path)
        post_probe = load_fluid_stage_probe(pair.post_path)

        _configure_and_restore_probe(
            operator=operator,
            probe=post_probe,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        )
        target_state = pack_fluid_state(operator)
        target_reaction_lookup = post_probe.reaction_lookup(kind="point")
        restored_target_reaction = np.asarray(operator.reaction_loads(refresh_state=False).values, dtype=float).reshape(-1)
        if target_reaction_lookup is None:
            target_reaction = restored_target_reaction.copy()
            restore_reaction_error = 0.0
        else:
            target_reaction = np.asarray(target_reaction_lookup.values, dtype=float).reshape(-1)
            restore_reaction_error = _relative_l2(target_reaction, restored_target_reaction)

        _configure_and_restore_probe(
            operator=operator,
            probe=pre_probe,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        )
        initial_state = pack_fluid_state(operator)
        trial_space = trial_basis.make_trial_space(operator, offset=initial_state)
        initial_coefficients = np.zeros(int(trial_space.n_modes), dtype=float)

        projected_coeffs = trial_basis.project_state(target_state, offset=initial_state)
        projected_state = trial_basis.reconstruct_state(projected_coeffs, offset=initial_state)
        write_fluid_state(operator, projected_state)
        _force_stage_acceleration(fluid=fluid, bossak=bossak)
        operator.update_oss_after_nonlinear_update()
        projected_reaction = np.asarray(operator.reaction_loads(refresh_state=True).values, dtype=float).reshape(-1)
        projected_residual = operator.assemble(need_matrix=False, convention="newton", refresh_predicted=True)
        projection_summary = {
            "state_error": _relative_l2(target_state[free], projected_state[free]),
            "reaction_error": _relative_l2(target_reaction, projected_reaction),
            "reaction_cosine": _cosine(target_reaction, projected_reaction),
            "residual_norm": float(np.linalg.norm(projected_residual.residual[free])),
        }

        _configure_and_restore_probe(
            operator=operator,
            probe=pre_probe,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        )
        initial_hook = _make_stage_acceleration_hook(
            fluid=fluid,
            bossak=bossak,
            preserve_seed_on_first_zero=True,
        )
        initial_verifier = FluidLSPGVerifier(
            operator=operator,
            trial_space=trial_space,
            state_update_hook=initial_hook,
            nonlinear_update_hook=operator.update_oss_after_nonlinear_update,
        )
        t_full0 = time.perf_counter()
        initial_system = initial_verifier.assemble_system(initial_coefficients, refresh_predicted=True)
        full_assembly_elapsed = float(time.perf_counter() - t_full0)
        full_assembly_times.append(full_assembly_elapsed)
        row_weights = None
        if bool(args.lspg_block_scale):
            row_weights = _fluid_block_row_weights(
                operator,
                row_dofs=trial_space.free_dofs,
                residual=initial_system.residual,
                relative_floor=float(args.lspg_block_scale_relative_floor),
            )

        _configure_and_restore_probe(
            operator=operator,
            probe=pre_probe,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        )
        lspg_hook = _make_stage_acceleration_hook(
            fluid=fluid,
            bossak=bossak,
            preserve_seed_on_first_zero=True,
        )
        lspg = FluidLSPGVerifier(
            operator=operator,
            trial_space=trial_space,
            row_weights=row_weights,
            state_update_hook=lspg_hook,
            nonlinear_update_hook=operator.update_oss_after_nonlinear_update,
        )
        t_lspg0 = time.perf_counter()
        lspg_result = lspg.solve(
            initial_coefficients,
            max_iterations=int(args.lspg_max_iterations),
            residual_tol=float(args.lspg_residual_tol),
            line_search=bool(args.lspg_line_search),
        )
        lspg_elapsed = float(time.perf_counter() - t_lspg0)
        lspg_times.append(lspg_elapsed)
        lspg_state = pack_fluid_state(operator)
        lspg_reaction = np.asarray(operator.reaction_loads(refresh_state=False).values, dtype=float).reshape(-1)

        trajectory_coefficients = _coefficient_training_path(initial_coefficients, lspg_result.trajectory)
        if trajectory_coefficients and not residual_training_entries:
            _configure_and_restore_probe(
                operator=operator,
                probe=pre_probe,
                setup=setup,
                fluid_iface_coords=fluid_iface_coords,
                dt=float(dt),
                reference_velocity=float(reference_velocity),
            )
            residual_hook = _make_stage_acceleration_hook(
                fluid=fluid,
                bossak=bossak,
                preserve_seed_on_first_zero=True,
            )
            residual_training_columns.append(
                _collect_residual_snapshots_from_coefficients(
                    operator=operator,
                    trial_space=trial_space,
                    coefficients=trajectory_coefficients,
                    row_dofs=free,
                    state_update_hook=residual_hook,
                )
            )

        gnat_summary = None
        if not bool(args.skip_gnat) and residual_training_columns:
            residual_snapshots = np.column_stack(residual_training_columns)
            residual_modes = (
                int(args.residual_modes)
                if args.residual_modes is not None
                else min(max(int(trial_space.n_modes), 2), int(residual_snapshots.shape[1]))
            )
            sample_set = fit_fluid_gnat_sample_set(
                operator,
                residual_snapshots,
                basis_dofs=free,
                residual_modes=int(residual_modes),
                row_oversampling=float(args.row_oversampling),
                min_sample_rows=int(args.min_sample_rows),
                forced_row_dofs=_interface_forced_row_dofs(operator, mode=str(args.force_interface_rows)),
                include_interface_elements=True,
                sample_weighting=str(args.gnat_sample_weighting),
            )
            element_cubature_summary: dict[str, Any] = {"enabled": False, "method": str(args.gnat_element_weighting)}
            if str(args.gnat_element_weighting).strip().lower() != "none":
                sample_set, element_cubature_summary = _fit_element_cubature_sample_set(
                    operator=operator,
                    fluid=fluid,
                    pairs=gnat_training_pairs,
                    trial_basis=trial_basis,
                    sample_set=sample_set,
                    setup=setup,
                    fluid_iface_coords=fluid_iface_coords,
                    dt=float(dt),
                    reference_velocity=float(reference_velocity),
                    bossak=bossak,
                    max_iterations=int(args.lspg_max_iterations),
                    residual_tol=float(args.lspg_residual_tol),
                    line_search=bool(args.lspg_line_search),
                    block_scale=bool(args.lspg_block_scale),
                    block_scale_relative_floor=float(args.lspg_block_scale_relative_floor),
                    max_training_states=max(1, int(args.gnat_element_max_training_states)),
                    max_weight=float(args.gnat_element_weight_max),
                    prune_tol=float(args.gnat_element_prune_tol),
                    keep_interface_elements=bool(args.gnat_element_keep_interface),
                )
            if args.save_hrom_model is not None and saved_hrom_model_path is None:
                _save_sampled_lspg_hrom_model(
                    Path(args.save_hrom_model),
                    trial_basis=trial_basis,
                    sample_set=sample_set,
                    args=args,
                    training_source=[str(item) for item in training_source],
                    training_steps=training_pair_steps,
                )
                saved_hrom_model_path = str(Path(args.save_hrom_model))
            _configure_and_restore_probe(
                operator=operator,
                probe=pre_probe,
                setup=setup,
                fluid_iface_coords=fluid_iface_coords,
                dt=float(dt),
                reference_velocity=float(reference_velocity),
            )
            gnat_trial_space = trial_basis.make_trial_space(operator, offset=initial_state)
            gnat_hook = _make_stage_acceleration_hook(
                fluid=fluid,
                bossak=bossak,
                preserve_seed_on_first_zero=True,
            )
            gnat_row_weights = None
            if bool(args.lspg_block_scale):
                initial_residual_full = np.zeros(int(operator.dh.total_dofs), dtype=float)
                initial_residual_full[free] = np.asarray(initial_system.residual, dtype=float).reshape(-1)
                gnat_row_weights = _fluid_block_row_weights(
                    operator,
                    row_dofs=sample_set.row_dofs,
                    residual=initial_residual_full[sample_set.row_dofs],
                    relative_floor=float(args.lspg_block_scale_relative_floor),
                )
            gnat = FluidGNATSolver(
                operator=operator,
                trial_space=gnat_trial_space,
                sample_set=sample_set,
                state_update_hook=gnat_hook,
                nonlinear_update_hook=operator.update_oss_after_nonlinear_update,
                objective=str(args.gnat_objective),
                row_weights=gnat_row_weights,
            )
            t_gnat0 = time.perf_counter()
            gnat_result = gnat.solve(
                initial_coefficients,
                max_iterations=int(args.gnat_max_iterations),
                residual_tol=float(args.gnat_residual_tol),
                line_search=bool(args.gnat_line_search),
            )
            gnat_elapsed = float(time.perf_counter() - t_gnat0)
            gnat_times.append(gnat_elapsed)
            gnat_state = pack_fluid_state(operator)
            gnat_reaction = np.asarray(operator.reaction_loads(refresh_state=False).values, dtype=float).reshape(-1)
            gnat_summary = {
                "state_error": _relative_l2(target_state[free], gnat_state[free]),
                "reaction_error": _relative_l2(target_reaction, gnat_reaction),
                "reaction_cosine": _cosine(target_reaction, gnat_reaction),
                "iterations": int(gnat_result.iterations),
                "converged": bool(gnat_result.converged),
                "estimated_residual_norm": float(gnat_result.residual_norm),
                "elapsed_s": float(gnat_elapsed),
                "residual_modes": int(sample_set.n_residual_modes),
                "sample_rows": int(sample_set.n_sample_rows),
                "sample_elements": int(sample_set.n_sample_elements),
                "sample_element_fraction": float(sample_set.n_sample_elements / max(int(operator.mesh.n_elements), 1)),
                "sampled_basis_rank": int(sample_set.sampled_basis_rank),
                "sampled_basis_condition": float(sample_set.sampled_basis_condition),
                "sample_weighting": str(sample_set.sample_weighting),
                "force_interface_rows": str(args.force_interface_rows),
                "sample_weight_fit_relative_error": float(sample_set.sample_weight_fit_relative_error),
                "sample_weight_min": float(np.min(sample_set.sample_weights))
                if sample_set.sample_weights.size
                else float("nan"),
                "sample_weight_max": float(np.max(sample_set.sample_weights))
                if sample_set.sample_weights.size
                else float("nan"),
                "sample_weight_sum": float(np.sum(sample_set.sample_weights)),
                "element_weighting": str(sample_set.element_weighting),
                "element_weight_fit_relative_error": float(sample_set.element_weight_fit_relative_error),
                "element_weight_min": float(np.min(sample_set.element_weights))
                if sample_set.element_weights.size
                else float("nan"),
                "element_weight_max": float(np.max(sample_set.element_weights))
                if sample_set.element_weights.size
                else float("nan"),
                "element_weight_sum": float(np.sum(sample_set.element_weights)),
                "element_cubature": element_cubature_summary,
                "lspg_block_scale": bool(args.lspg_block_scale),
                "sample_row_weight_min": float(np.min(gnat_row_weights))
                if gnat_row_weights is not None and gnat_row_weights.size
                else float("nan"),
                "sample_row_weight_max": float(np.max(gnat_row_weights))
                if gnat_row_weights is not None and gnat_row_weights.size
                else float("nan"),
                "objective": str(args.gnat_objective),
            }

        replay_entries.append(
            {
                "step": int(pair.step),
                "coupling_iter": int(pair.coupling_iter),
                "pre_probe": str(pair.pre_path),
                "post_probe": str(pair.post_path),
                "stage_restore_reaction_error": float(restore_reaction_error),
                "initial_state_error": _relative_l2(target_state[free], initial_state[free]),
                "best_online_projection": projection_summary,
                "full_initial_lspg_residual_norm": float(initial_system.residual_norm),
                "full_assembly_time_s": float(full_assembly_elapsed),
                "full_lspg": {
                    "state_error": _relative_l2(target_state[free], lspg_state[free]),
                    "reaction_error": _relative_l2(target_reaction, lspg_reaction),
                    "reaction_cosine": _cosine(target_reaction, lspg_reaction),
                    "iterations": int(lspg_result.iterations),
                    "converged": bool(lspg_result.converged),
                    "residual_norm": float(lspg_result.residual_norm),
                    "elapsed_s": float(lspg_elapsed),
                    "line_search": bool(args.lspg_line_search),
                    "block_scale": bool(args.lspg_block_scale),
                    "trajectory": [dict(item) for item in lspg_result.trajectory],
                },
                "gnat": gnat_summary,
            }
        )

    median_full = float(np.median(full_assembly_times)) if full_assembly_times else float("nan")
    median_lspg = float(np.median(lspg_times)) if lspg_times else float("nan")
    median_gnat = float(np.median(gnat_times)) if gnat_times else float("nan")
    replay_steps = [int(pair.step) for pair in stage_pairs]
    training_replay_overlap = sorted(set(training_pair_steps).intersection(replay_steps))
    summary: dict[str, Any] = {
        "training": {
            "kind": training_kind,
            "sources": training_source,
            "training_steps": training_pair_steps,
            "training_all_iters": bool(args.training_all_iters),
            "replay_step_overlap": training_replay_overlap,
            "velocity_modes": int(args.velocity_modes),
            "pressure_modes": int(args.pressure_modes),
            "total_modes": int(trial_basis.n_modes),
            "mode_choice": "fixed to the previously selected Tiba-style held-out validation result unless overridden",
            "basis_kind": str(args.basis_kind),
            "gnat_residual_training": (
                "training-stage full-LSPG trajectories"
                if residual_training_entries
                else (
                    "diagnostic oracle: full-LSPG trajectory residuals collected during this replay; "
                    "replace with training-step stage trajectories before production timing"
                )
            ),
            "gnat_residual_training_entries": residual_training_entries,
            "gnat_sample_weighting": str(args.gnat_sample_weighting),
            "gnat_element_weighting": str(args.gnat_element_weighting),
            "gnat_element_max_training_states": int(args.gnat_element_max_training_states),
            "gnat_element_weight_max": float(args.gnat_element_weight_max),
            "gnat_element_prune_tol": float(args.gnat_element_prune_tol),
            "gnat_element_keep_interface": bool(args.gnat_element_keep_interface),
            "force_interface_rows": str(args.force_interface_rows),
            "gnat_objective": str(args.gnat_objective),
            "saved_hrom_model": saved_hrom_model_path,
        },
        "stage_pairs": [
            {
                "step": int(pair.step),
                "coupling_iter": int(pair.coupling_iter),
                "pre": str(pair.pre_path),
                "post": str(pair.post_path),
            }
            for pair in stage_pairs
        ],
        "replay_all_iters": bool(args.replay_all_iters),
        "replay": replay_entries,
        "timing": {
            "median_full_lspg_assembly_s": median_full,
            "median_full_lspg_solve_s": median_lspg,
            "median_gnat_solve_s": median_gnat,
            "median_initial_assembly_to_gnat_speed_ratio": float(median_full / median_gnat)
            if median_gnat > 0.0
            else float("nan"),
        },
    }
    dump_json(summary, Path(args.output))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
