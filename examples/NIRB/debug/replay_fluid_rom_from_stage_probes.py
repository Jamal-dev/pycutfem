from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from examples.NIRB.common import dump_json
from examples.NIRB.debug.replay_fluid_gnat_from_checkpoints import (
    _build_snapshot_batch,
    _checkpoint_paths,
    _configure_checkpoint_bcs,
    _make_operator,
    _restore_accepted_checkpoint,
)
from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.fluid_basis import (
    enrich_fluid_pod_trial_basis_with_interface_state_modes,
    enrich_fluid_pod_trial_basis_with_supremizers,
    fit_fluid_pod_trial_basis,
    fit_fluid_pod_trial_basis_from_state_matrix,
)
from examples.NIRB.fluid_gnat import FluidGNATSolver, FluidGNATSampleSet, fit_fluid_gnat_sample_set
from examples.NIRB.fluid_lspg import FluidLSPGVerifier, pack_fluid_state, write_fluid_state
from examples.NIRB.fluid_stage_probes import (
    FluidStageProbe,
    build_stage_probe_batch,
    find_fluid_stage_probe_pairs,
    load_fluid_stage_probe,
    restore_fluid_stage_probe,
)
from examples.NIRB.fluid_snapshots import FluidStageSnapshotWriter, _FIELD_KEYS
from examples.NIRB.run_example2_local import (
    _bossak_coefficients,
    _assemble_fluid_sampled_galerkin_element_contributions_raw,
    _assemble_fluid_sampled_galerkin_reduced_system_raw,
    _assemble_fluid_sampled_lspg_element_contributions_raw,
    _assemble_fluid_sampled_lspg_rows_raw,
    _boundary_field_data,
    _build_fluid_problem,
    _fluid_interface_constrained_velocity_rows,
    _fluid_interface_reaction_element_ids,
    _fluid_interface_reaction_sample_row_values,
    _fluid_interface_velocity_dofs,
    _fluid_reaction_element_ids_for_velocity_rows,
    _load_checkpoint_payload,
    _load_reference_partitioned_meshes,
    _resample_lookup_to_coords,
)


def _progress(message: str) -> None:
    print(f"[replay-fluid-rom] {message}", file=sys.stderr, flush=True)


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


def _append_unique_coefficient_states(
    base: list[np.ndarray],
    extras: list[np.ndarray],
    *,
    tol: float = 1.0e-14,
) -> list[np.ndarray]:
    """Append projected probe states without duplicating an existing coefficient vector."""

    combined = [np.asarray(item, dtype=float).reshape(-1).copy() for item in base]
    for item in extras:
        candidate = np.asarray(item, dtype=float).reshape(-1)
        if combined and candidate.size != combined[0].size:
            raise ValueError("Projected intermediate-state coefficient size does not match the trial space.")
        if any(float(np.linalg.norm(candidate - existing)) <= float(tol) for existing in combined):
            continue
        combined.append(candidate.copy())
    return combined


def _solve_full_mesh_galerkin_training_path(
    *,
    operator,
    trial_space,
    fluid: dict[str, object],
    bossak: dict[str, float],
    initial_coefficients: np.ndarray,
    max_iterations: int,
    residual_tol: float,
    line_search: bool,
    incompressibility_stabilization_scale: float,
) -> tuple[list[np.ndarray], int, bool, float]:
    """Full-mesh reduced Galerkin Newton path for sampled-Galerkin cubature training."""

    coeffs = np.asarray(initial_coefficients, dtype=float).reshape(-1).copy()
    if int(coeffs.size) != int(trial_space.n_modes):
        raise ValueError("Initial Galerkin coefficients do not match the trial space.")
    p = operator.parameters
    all_elements = np.arange(int(operator.mesh.n_elements), dtype=int)
    path: list[np.ndarray] = [coeffs.copy()]
    update_hook = _make_stage_acceleration_hook(
        fluid=fluid,
        bossak=bossak,
        preserve_seed_on_first_zero=True,
    )

    def assemble(current: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        trial_space.write(operator, np.asarray(current, dtype=float).reshape(-1))
        update_hook(np.asarray(current, dtype=float).reshape(-1))
        operator.refresh_predicted_subscale()
        residual, tangent = _assemble_fluid_sampled_galerkin_reduced_system_raw(
            prob=operator.prob,
            rho_f=float(p.rho_f),
            mu_f=float(p.mu_f),
            dt=float(p.dt),
            quad_order=int(p.quadrature_order),
            bossak_alpha=float(p.bossak_alpha),
            contribution_mode=str(p.contribution_mode),
            backend=str(p.backend),
            element_ids=all_elements,
            basis=np.asarray(trial_space.basis, dtype=float),
            incompressibility_stabilization_scale=float(incompressibility_stabilization_scale),
        )
        residual_vec = np.asarray(residual, dtype=float).reshape(-1)
        tangent_mat = np.asarray(tangent, dtype=float)
        return residual_vec, tangent_mat, float(np.linalg.norm(residual_vec))

    last_norm = float("inf")
    for iteration in range(1, max(1, int(max_iterations)) + 1):
        residual, tangent, last_norm = assemble(coeffs)
        if last_norm <= float(residual_tol):
            return path, int(iteration), True, float(last_norm)
        step, *_ = np.linalg.lstsq(tangent, -residual, rcond=None)
        step = np.asarray(step, dtype=float).reshape(-1)
        accepted = coeffs + step
        if bool(line_search):
            history = operator.snapshot_history()
            best_coeffs = accepted.copy()
            best_norm = float("inf")
            for search_iter in range(6):
                alpha = 0.5**search_iter
                trial = coeffs + float(alpha) * step
                operator.restore_history(history)
                _res_trial, _tan_trial, trial_norm = assemble(trial)
                if trial_norm < best_norm:
                    best_norm = float(trial_norm)
                    best_coeffs = np.asarray(trial, dtype=float).reshape(-1).copy()
                if trial_norm <= (1.0 - 1.0e-4 * float(alpha)) * last_norm:
                    best_coeffs = np.asarray(trial, dtype=float).reshape(-1).copy()
                    break
            operator.restore_history(history)
            accepted = best_coeffs
        coeffs = np.asarray(accepted, dtype=float).reshape(-1)
        if float(np.linalg.norm(coeffs - path[-1])) > 1.0e-14:
            path.append(coeffs.copy())
        trial_space.write(operator, coeffs)
        update_hook(coeffs)
        operator.update_oss_after_nonlinear_update()
        if float(np.linalg.norm(step)) <= 1.0e-12 * max(1.0, float(np.linalg.norm(coeffs))):
            _residual, _tangent, last_norm = assemble(coeffs)
            return path, int(iteration), bool(last_norm <= float(residual_tol)), float(last_norm)

    _residual, _tangent, last_norm = assemble(coeffs)
    return path, max(1, int(max_iterations)), bool(last_norm <= float(residual_tol)), float(last_norm)


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
    pair_weights: dict[tuple[int, int], float] | None = None,
    trajectory_source: str = "lspg",
    include_intermediate_states: bool = False,
    incompressibility_stabilization_scale: float = 1.0,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    residual_columns: list[np.ndarray] = []
    entries: list[dict[str, Any]] = []
    free = np.asarray(free_dofs, dtype=int).reshape(-1)

    pair_list = list(pairs)
    phase_t0 = time.perf_counter()
    _progress(
        "collect residual snapshots: "
        f"pairs={len(pair_list)} trajectory={trajectory_source} modes={trial_basis.n_modes}"
    )
    for pair_index, pair in enumerate(pair_list, start=1):
        item_t0 = time.perf_counter()
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
        trajectory_mode = str(trajectory_source).strip().lower()
        exact_state_coefficients: list[np.ndarray] = []
        exact_state_labels: list[str] = []
        if bool(include_intermediate_states):
            for newton_index, newton_path in enumerate(_exact_newton_iterate_probe_paths(pair), start=1):
                newton_probe = load_fluid_stage_probe(newton_path)
                _configure_and_restore_probe(
                    operator=operator,
                    probe=newton_probe,
                    setup=setup,
                    fluid_iface_coords=fluid_iface_coords,
                    dt=float(dt),
                    reference_velocity=float(reference_velocity),
                )
                newton_state = pack_fluid_state(operator)
                exact_state_coefficients.append(
                    np.asarray(trial_basis.project_state(newton_state, offset=initial_state), dtype=float).reshape(-1)
                )
                exact_state_labels.append(f"newton_iter{int(newton_index):04d}")
            _configure_and_restore_probe(
                operator=operator,
                probe=pre_probe,
                setup=setup,
                fluid_iface_coords=fluid_iface_coords,
                dt=float(dt),
                reference_velocity=float(reference_velocity),
            )
        if trajectory_mode == "projection":
            t0 = time.perf_counter()
            projected_coefficients = trial_basis.project_state(target_state, offset=initial_state)
            trial_space.write(operator, projected_coefficients)
            projected_hook = _make_stage_acceleration_hook(
                fluid=fluid,
                bossak=bossak,
                preserve_seed_on_first_zero=True,
            )
            projected_hook(np.asarray(projected_coefficients, dtype=float).reshape(-1))
            operator.update_oss_after_nonlinear_update()
            elapsed = float(time.perf_counter() - t0)
            lspg_state = pack_fluid_state(operator)
            lspg_reaction = np.asarray(operator.reaction_loads(refresh_state=False).values, dtype=float).reshape(-1)
            trajectory_coefficients = [initial_coefficients, np.asarray(projected_coefficients, dtype=float).reshape(-1)]
            lspg_iterations = 0
            lspg_converged = True
            lspg_residual_norm = float("nan")
        elif trajectory_mode in {"galerkin", "sampled_galerkin", "reduced-galerkin", "reduced_galerkin"}:
            t0 = time.perf_counter()
            trajectory_coefficients, lspg_iterations, lspg_converged, lspg_residual_norm = (
                _solve_full_mesh_galerkin_training_path(
                    operator=operator,
                    trial_space=trial_space,
                    fluid=fluid,
                    bossak=bossak,
                    initial_coefficients=initial_coefficients,
                    max_iterations=int(max_iterations),
                    residual_tol=float(residual_tol),
                    line_search=bool(line_search),
                    incompressibility_stabilization_scale=float(incompressibility_stabilization_scale),
                )
            )
            elapsed = float(time.perf_counter() - t0)
            lspg_state = pack_fluid_state(operator)
            lspg_reaction = np.asarray(operator.reaction_loads(refresh_state=False).values, dtype=float).reshape(-1)
        else:
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
            lspg_iterations = int(lspg_result.iterations)
            lspg_converged = bool(lspg_result.converged)
            lspg_residual_norm = float(lspg_result.residual_norm)
        if exact_state_coefficients:
            trajectory_coefficients = _append_unique_coefficient_states(
                trajectory_coefficients,
                exact_state_coefficients,
            )
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
        weight = max(_pair_weight(pair, pair_weights), 0.0)
        residual_columns.append(math.sqrt(weight) * residual_matrix)
        entries.append(
            {
                "step": int(pair.step),
                "coupling_iter": int(pair.coupling_iter),
                "training_weight": float(weight),
                "pre_probe": str(pair.pre_path),
                "post_probe": str(pair.post_path),
                "full_lspg_state_error": _relative_l2(target_state[free], lspg_state[free]),
                "full_lspg_reaction_error": _relative_l2(target_reaction, lspg_reaction),
                "full_lspg_reaction_cosine": _cosine(target_reaction, lspg_reaction),
                "full_lspg_iterations": int(lspg_iterations),
                "full_lspg_converged": bool(lspg_converged),
                "full_lspg_residual_norm": float(lspg_residual_norm),
                "full_lspg_elapsed_s": float(elapsed),
                "full_lspg_block_scale": bool(block_scale),
                "trajectory_source": str(trajectory_mode),
                "residual_snapshot_columns": int(residual_matrix.shape[1]),
                "exact_intermediate_state_columns": int(len(exact_state_coefficients)),
                "exact_intermediate_state_labels": exact_state_labels,
            }
        )
        if pair_index == 1 or pair_index == len(pair_list) or pair_index % 5 == 0:
            _progress(
                "collect residual snapshots: "
                f"{pair_index}/{len(pair_list)} "
                f"step={int(pair.step)} iter={int(pair.coupling_iter)} "
                f"cols={int(residual_matrix.shape[1])} "
                f"elapsed_pair={time.perf_counter() - item_t0:.2f}s "
                f"elapsed_total={time.perf_counter() - phase_t0:.2f}s"
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


def _checkpoint_root(path: Path) -> Path:
    root = Path(path).resolve()
    if root.is_dir() and (root / "checkpoints").is_dir():
        return root / "checkpoints"
    return root


def _checkpoint_step_from_path(path: Path) -> int | None:
    stem = Path(path).stem
    prefix = "checkpoint_step_"
    if stem.startswith(prefix):
        try:
            return int(stem[len(prefix) :])
        except ValueError:
            return None
    return None


def _checkpoint_step_map(path: Path) -> dict[int, Path]:
    root = _checkpoint_root(Path(path))
    if root.is_file():
        step = _checkpoint_step_from_path(root)
        if step is None:
            with np.load(root, allow_pickle=True) as payload:
                step = int(np.asarray(payload["step"], dtype=int).reshape(-1)[0])
        return {int(step): root}
    files = sorted(root.glob("checkpoint_step_*.npz"))
    if not files:
        raise FileNotFoundError(f"No checkpoint_step_*.npz files found under {root}.")
    mapping: dict[int, Path] = {}
    for file_path in files:
        step = _checkpoint_step_from_path(file_path)
        if step is None:
            with np.load(file_path, allow_pickle=True) as payload:
                step = int(np.asarray(payload["step"], dtype=int).reshape(-1)[0])
        mapping[int(step)] = file_path
    return mapping


def _selected_checkpoint_steps(path: Path, spec: str | None) -> list[int]:
    mapping = _checkpoint_step_map(Path(path))
    requested = _parse_steps(spec)
    if requested is None:
        return sorted(mapping)
    missing = [int(step) for step in requested if int(step) not in mapping]
    if missing:
        preview = ", ".join(str(step) for step in missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise FileNotFoundError(f"Requested checkpoint steps missing under {path}: {preview}{suffix}")
    return [int(step) for step in requested]


def _pack_checkpoint_state(
    *,
    operator,
    fluid: dict[str, object],
    path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    payload = _load_checkpoint_payload(Path(path).resolve())
    _restore_accepted_checkpoint(fluid, payload)
    return pack_fluid_state(operator), payload


def _accepted_checkpoint_state_matrix(
    *,
    operator,
    fluid: dict[str, object],
    checkpoint_dir: Path,
    steps_spec: str | None,
    kind: str,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Collect whole-run accepted checkpoint states for homogeneous enrichment.

    For the online fluid solve, the basis is used as a correction from the
    current lifted state.  The default therefore uses accepted step increments
    ``x_n - x_{n-1}``, not absolute states, so the new modes remain compatible
    with the stage-increment basis and do not encode Dirichlet rows.
    """

    mapping = _checkpoint_step_map(Path(checkpoint_dir))
    selected_steps = _selected_checkpoint_steps(Path(checkpoint_dir), steps_spec)
    mode = str(kind).strip().lower().replace("-", "_")
    if mode not in {"increment", "delta", "absolute", "state"}:
        raise ValueError(f"Unsupported accepted checkpoint enrichment kind {kind!r}.")
    states: dict[int, np.ndarray] = {}
    columns: list[np.ndarray] = []
    sources: list[str] = []
    skipped_missing_previous = 0
    skipped_zero = 0
    for step in selected_steps:
        step_int = int(step)
        current_path = mapping[step_int]
        if step_int not in states:
            states[step_int], _payload = _pack_checkpoint_state(operator=operator, fluid=fluid, path=current_path)
        current = states[step_int]
        if mode in {"absolute", "state"}:
            vector = np.asarray(current, dtype=float).reshape(-1)
            label = "accepted_state"
        else:
            previous_step = step_int - 1
            previous_path = mapping.get(previous_step)
            if previous_path is None:
                skipped_missing_previous += 1
                continue
            if previous_step not in states:
                states[previous_step], _payload = _pack_checkpoint_state(
                    operator=operator,
                    fluid=fluid,
                    path=previous_path,
                )
            vector = np.asarray(current - states[previous_step], dtype=float).reshape(-1)
            label = "accepted_increment"
        if float(np.linalg.norm(vector)) <= 1.0e-15:
            skipped_zero += 1
            continue
        columns.append(vector)
        sources.append(f"{current_path}#{label}")
    if not columns:
        return (
            np.zeros((int(operator.dh.total_dofs), 0), dtype=float),
            [],
            {
                "enabled": False,
                "reason": "no_nonzero_checkpoint_states",
                "requested_steps": int(len(selected_steps)),
                "skipped_missing_previous": int(skipped_missing_previous),
                "skipped_zero": int(skipped_zero),
            },
        )
    matrix = np.column_stack(columns)
    return (
        matrix,
        sources,
        {
            "enabled": True,
            "kind": mode,
            "checkpoint_dir": str(_checkpoint_root(Path(checkpoint_dir))),
            "requested_steps": int(len(selected_steps)),
            "states": int(matrix.shape[1]),
            "skipped_missing_previous": int(skipped_missing_previous),
            "skipped_zero": int(skipped_zero),
        },
    )


def _project_training_state_coefficients_from_checkpoints(
    *,
    operator,
    fluid: dict[str, object],
    checkpoint_dir: Path,
    steps_spec: str | None,
    trial_basis,
    kind: str,
) -> np.ndarray:
    mapping = _checkpoint_step_map(Path(checkpoint_dir))
    selected_steps = _selected_checkpoint_steps(Path(checkpoint_dir), steps_spec)
    mode = str(kind).strip().lower().replace("-", "_")
    columns: list[np.ndarray] = []
    states: dict[int, np.ndarray] = {}
    for step in selected_steps:
        step_int = int(step)
        if step_int not in states:
            states[step_int], _payload = _pack_checkpoint_state(
                operator=operator,
                fluid=fluid,
                path=mapping[step_int],
            )
        current = states[step_int]
        if mode in {"absolute", "state"}:
            offset = np.zeros_like(current)
        else:
            previous_step = step_int - 1
            previous_path = mapping.get(previous_step)
            if previous_path is None:
                continue
            if previous_step not in states:
                states[previous_step], _payload = _pack_checkpoint_state(
                    operator=operator,
                    fluid=fluid,
                    path=previous_path,
                )
            offset = states[previous_step]
        columns.append(np.zeros(int(trial_basis.n_modes), dtype=float))
        columns.append(trial_basis.project_state(current, offset=offset))
    if not columns:
        return np.zeros((int(trial_basis.n_modes), 0), dtype=float)
    return np.column_stack(columns)


def _accepted_checkpoint_reaction_snapshots(
    *,
    operator,
    fluid: dict[str, object],
    setup,
    fluid_iface_coords: np.ndarray,
    checkpoint_dir: Path,
    steps_spec: str | None,
    reference_velocity: float,
    reaction_coords: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    mapping = _checkpoint_step_map(Path(checkpoint_dir))
    selected_steps = _selected_checkpoint_steps(Path(checkpoint_dir), steps_spec)
    coords = None if reaction_coords is None else np.asarray(reaction_coords, dtype=float)
    rows: list[np.ndarray] = []
    entries: list[dict[str, Any]] = []
    phase_t0 = time.perf_counter()
    for index, step in enumerate(selected_steps, start=1):
        path = mapping[int(step)]
        _state, payload = _pack_checkpoint_state(operator=operator, fluid=fluid, path=path)
        _configure_checkpoint_bcs(
            operator=operator,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            time_s=float(np.asarray(payload.get("time_s", np.asarray(0.0))).reshape(-1)[0]),
            reference_velocity=float(reference_velocity),
        )
        lookup = operator.reaction_loads(refresh_state=False)
        if coords is None:
            coords = np.asarray(lookup.coords, dtype=float)
            values = np.asarray(lookup.values, dtype=float).reshape(-1)
        else:
            values = np.asarray(_resample_lookup_to_coords(lookup, coords).values, dtype=float).reshape(-1)
        rows.append(values)
        entries.append(
            {
                "source": "accepted_checkpoint",
                "checkpoint": str(path),
                "step": int(step),
                "reaction_norm": float(np.linalg.norm(values)),
            }
        )
        if index == 1 or index == len(selected_steps) or index % 50 == 0:
            _progress(
                "collect accepted checkpoint reactions: "
                f"{index}/{len(selected_steps)} step={int(step)} "
                f"elapsed={time.perf_counter() - phase_t0:.2f}s"
            )
    if coords is None or not rows:
        raise ValueError("No accepted checkpoint reaction snapshots were collected.")
    return np.asarray(coords, dtype=float), np.vstack(rows), entries


def _training_pair_weight_map(
    pairs,
    *,
    mode: str,
    late_start_step: int | None,
    late_factor: float,
    final_factor: float,
    iteration_factor: float,
) -> dict[tuple[int, int], float]:
    """Return deterministic pair weights for operator-aware HROM training.

    The weights are intentionally simple and reproducible.  They are applied as
    sqrt(weight) column scaling in POD/residual snapshots and LS row scaling in
    cubature fitting, which is equivalent to duplicating important snapshots
    without physically copying probe files.
    """

    mode_value = str(mode).strip().lower()
    if mode_value == "uniform":
        return {(int(pair.step), int(pair.coupling_iter)): 1.0 for pair in pairs}
    if mode_value not in {"coupled", "late-coupled", "late_coupled"}:
        raise ValueError(f"Unsupported training weight mode {mode!r}.")

    max_iter_by_step: dict[int, int] = {}
    for pair in pairs:
        step = int(pair.step)
        max_iter_by_step[step] = max(int(max_iter_by_step.get(step, 0)), int(pair.coupling_iter))

    weights: dict[tuple[int, int], float] = {}
    for pair in pairs:
        step = int(pair.step)
        coupling_iter = int(pair.coupling_iter)
        max_iter = max(int(max_iter_by_step.get(step, coupling_iter)), 1)
        normalized_iter = float(coupling_iter) / float(max_iter)
        weight = 1.0 + max(float(iteration_factor), 0.0) * normalized_iter
        if coupling_iter == max_iter:
            weight *= max(float(final_factor), 0.0)
        if late_start_step is not None and step >= int(late_start_step):
            weight *= max(float(late_factor), 0.0)
        weights[(step, coupling_iter)] = max(float(weight), 0.0)
    return weights


def _pair_weight(pair, weights: dict[tuple[int, int], float] | None) -> float:
    if weights is None:
        return 1.0
    return float(weights.get((int(pair.step), int(pair.coupling_iter)), 1.0))


def _exact_newton_iterate_probe_paths(pair) -> list[Path]:
    """Return exact-fluid Newton iterate probes for one coupling stage, if dumped."""

    probe_dir = Path(pair.pre_path).resolve().parent
    prefix = f"step{int(pair.step):04d}_iter{int(pair.coupling_iter):04d}_newton_iter"
    return sorted(path.resolve() for path in probe_dir.glob(f"{prefix}*.npz"))


def _state_probe_sequence_for_pair(
    pair,
    *,
    include_pre: bool,
    include_intermediate: bool,
    include_post: bool,
) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    if bool(include_pre):
        items.append(("pre_fluid_solve", Path(pair.pre_path).resolve()))
    if bool(include_intermediate):
        for path in _exact_newton_iterate_probe_paths(pair):
            label = path.stem.split(f"step{int(pair.step):04d}_iter{int(pair.coupling_iter):04d}_", 1)[-1]
            items.append((label, path))
    if bool(include_post):
        items.append(("post_fluid_solve", Path(pair.post_path).resolve()))
    return items


def _training_state_probe_paths_from_pairs(
    pairs,
    *,
    include_pre: bool,
    include_intermediate: bool,
    include_post: bool,
) -> list[tuple[object, str, Path]]:
    states: list[tuple[object, str, Path]] = []
    for pair in pairs:
        for label, path in _state_probe_sequence_for_pair(
            pair,
            include_pre=include_pre,
            include_intermediate=include_intermediate,
            include_post=include_post,
        ):
            states.append((pair, label, path))
    return states


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
    pair_weights: dict[tuple[int, int], float] | None = None,
    include_intermediate_states: bool = False,
    supremizer_enrichment: bool = False,
    supremizer_modes: int | None = None,
    supremizer_riesz: str = "h1",
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
        weight = max(_pair_weight(pair, pair_weights), 0.0)
        target_paths = _state_probe_sequence_for_pair(
            pair,
            include_pre=False,
            include_intermediate=bool(include_intermediate_states),
            include_post=True,
        )
        for state_label, target_path in target_paths:
            target_probe = post_probe if Path(target_path).resolve() == Path(pair.post_path).resolve() else load_fluid_stage_probe(target_path)
            _configure_and_restore_probe(
                operator=operator,
                probe=target_probe,
                setup=setup,
                fluid_iface_coords=fluid_iface_coords,
                dt=float(dt),
                reference_velocity=float(reference_velocity),
            )
            target = pack_fluid_state(operator)
            delta = np.asarray(target - initial, dtype=float).reshape(-1)
            if float(np.linalg.norm(delta)) <= 1.0e-15:
                continue
            deltas.append(math.sqrt(weight) * delta)
            sources.append(f"{target_path}#{state_label}")
    if not deltas:
        raise ValueError("Need at least one stage probe pair to fit an increment basis.")
    basis = fit_fluid_pod_trial_basis_from_state_matrix(
        operator,
        np.column_stack(deltas),
        free_dofs=np.asarray(free_dofs, dtype=int),
        velocity_modes=int(velocity_modes),
        pressure_modes=int(pressure_modes),
        center=False,
    )
    if bool(supremizer_enrichment):
        basis = enrich_fluid_pod_trial_basis_with_supremizers(
            operator,
            basis,
            supremizer_modes=supremizer_modes,
            riesz=str(supremizer_riesz),
        )
    return basis, sources


def _reaction_values_for_configured_probe(
    operator,
    probe: FluidStageProbe,
    *,
    reaction_coords: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = probe.reaction_lookup(kind="point")
    if lookup is None:
        lookup = operator.reaction_loads(refresh_state=False)
    if reaction_coords is None:
        coords = np.asarray(lookup.coords, dtype=float)
        values = np.asarray(lookup.values, dtype=float).reshape(-1)
        return coords, values
    values = np.asarray(_resample_lookup_to_coords(lookup, np.asarray(reaction_coords, dtype=float)).values, dtype=float)
    return np.asarray(reaction_coords, dtype=float), values.reshape(-1)


def _interface_load_enrichment_matrix_from_probe_pairs(
    *,
    operator,
    pairs,
    setup,
    fluid_iface_coords: np.ndarray,
    dt: float,
    reference_velocity: float,
    pair_weights: dict[tuple[int, int], float] | None = None,
    include_intermediate_states: bool = False,
    weight_mode: str = "reaction_norm",
    weight_exponent: float = 1.0,
    max_weight: float = 25.0,
    max_states: int = 0,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Collect reaction-weighted state increments for interface basis enrichment."""

    deltas: list[np.ndarray] = []
    metrics: list[float] = []
    pair_weight_values: list[float] = []
    sources: list[str] = []
    reaction_coords: np.ndarray | None = None
    mode = str(weight_mode).strip().lower().replace("-", "_")
    pair_list = list(pairs)
    if int(max_states) > 0:
        # Keep deterministic coverage across the pair list instead of only the
        # earliest coupling states.
        pair_stride = max(1, int(math.ceil(max(len(pair_list), 1) / max(int(max_states), 1))))
    else:
        pair_stride = 1
    for pair_index, pair in enumerate(pair_list):
        if int(max_states) > 0 and pair_index % pair_stride != 0 and len(deltas) >= int(max_states):
            continue
        pre_probe = load_fluid_stage_probe(pair.pre_path)
        _configure_and_restore_probe(
            operator=operator,
            probe=pre_probe,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        )
        initial = pack_fluid_state(operator)
        reaction_coords, pre_reaction = _reaction_values_for_configured_probe(
            operator,
            pre_probe,
            reaction_coords=reaction_coords,
        )
        target_paths = _state_probe_sequence_for_pair(
            pair,
            include_pre=False,
            include_intermediate=bool(include_intermediate_states),
            include_post=True,
        )
        for state_label, target_path in target_paths:
            if int(max_states) > 0 and len(deltas) >= int(max_states):
                break
            target_probe = load_fluid_stage_probe(target_path)
            _configure_and_restore_probe(
                operator=operator,
                probe=target_probe,
                setup=setup,
                fluid_iface_coords=fluid_iface_coords,
                dt=float(dt),
                reference_velocity=float(reference_velocity),
            )
            target = pack_fluid_state(operator)
            delta = np.asarray(target - initial, dtype=float).reshape(-1)
            if float(np.linalg.norm(delta)) <= 1.0e-15:
                continue
            reaction_coords, reaction = _reaction_values_for_configured_probe(
                operator,
                target_probe,
                reaction_coords=reaction_coords,
            )
            if mode in {"uniform", "none"}:
                metric = 1.0
            elif mode in {"reaction_change", "load_change", "delta_reaction"}:
                metric = float(np.linalg.norm(reaction - pre_reaction))
            elif mode in {"reaction_norm", "load_norm", "norm"}:
                metric = float(np.linalg.norm(reaction))
            else:
                raise ValueError(f"Unsupported interface-load enrichment weight mode {weight_mode!r}.")
            deltas.append(delta)
            metrics.append(max(float(metric), 0.0))
            pair_weight_values.append(max(_pair_weight(pair, pair_weights), 0.0))
            sources.append(f"{target_path}#{state_label}")
    if not deltas:
        return (
            np.zeros((int(operator.dh.total_dofs), 0), dtype=float),
            [],
            {
                "enabled": False,
                "reason": "no_nonzero_state_increments",
                "states": 0,
            },
        )
    metric_array = np.asarray(metrics, dtype=float)
    positive = metric_array[metric_array > 0.0]
    metric_scale = float(np.median(positive)) if positive.size else 1.0
    metric_scale = max(metric_scale, 1.0e-30)
    relative_metric = np.maximum(metric_array / metric_scale, 1.0e-12)
    exponent = max(float(weight_exponent), 0.0)
    metric_weights = relative_metric**exponent
    if np.isfinite(float(max_weight)) and float(max_weight) > 0.0:
        metric_weights = np.minimum(metric_weights, float(max_weight))
    weights = np.asarray(pair_weight_values, dtype=float) * metric_weights
    weights = np.maximum(weights, 0.0)
    weighted_columns = [
        np.sqrt(float(weight)) * np.asarray(delta, dtype=float).reshape(-1)
        for delta, weight in zip(deltas, weights)
        if float(weight) > 0.0
    ]
    if not weighted_columns:
        return (
            np.zeros((int(operator.dh.total_dofs), 0), dtype=float),
            [],
            {
                "enabled": False,
                "reason": "all_weights_zero",
                "states": len(deltas),
            },
        )
    summary = {
        "enabled": True,
        "states": int(len(weighted_columns)),
        "weight_mode": str(weight_mode),
        "weight_exponent": float(weight_exponent),
        "max_weight": float(max_weight),
        "metric_scale": float(metric_scale),
        "metric_min": float(np.min(metric_array)),
        "metric_max": float(np.max(metric_array)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_mean": float(np.mean(weights)),
    }
    return np.column_stack(weighted_columns), sources, summary


def _fit_reduced_reaction_operator_from_probe_pairs(
    *,
    operator,
    pairs,
    trial_basis,
    setup,
    fluid_iface_coords: np.ndarray,
    dt: float,
    reference_velocity: float,
    include_intermediate_states: bool,
    ridge: float,
    incremental: bool = True,
    validation_stride: int = 5,
) -> dict[str, Any]:
    """Fit an affine reduced map from fluid coefficients to interface load.

    In incremental mode the map learns ``lambda(q) - lambda(q=0)`` and the
    online driver adds the current interface-load guess back before using the
    prediction.  This is usually more stable for FSI than one global absolute
    bias, because every coupling stage has a different pre-fluid offset.
    """

    coefficient_rows: list[np.ndarray] = []
    reaction_rows: list[np.ndarray] = []
    validation_mask: list[bool] = []
    entries: list[dict[str, Any]] = []
    reaction_coords: np.ndarray | None = None
    stride = max(0, int(validation_stride))
    for pair_index, pair in enumerate(list(pairs), start=1):
        pre_probe = load_fluid_stage_probe(pair.pre_path)
        _configure_and_restore_probe(
            operator=operator,
            probe=pre_probe,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        )
        initial = pack_fluid_state(operator)
        reaction_coords, pre_reaction = _reaction_values_for_configured_probe(
            operator,
            pre_probe,
            reaction_coords=reaction_coords,
        )
        target_paths = _state_probe_sequence_for_pair(
            pair,
            include_pre=False,
            include_intermediate=bool(include_intermediate_states),
            include_post=True,
        )
        for state_label, target_path in target_paths:
            target_probe = load_fluid_stage_probe(target_path)
            _configure_and_restore_probe(
                operator=operator,
                probe=target_probe,
                setup=setup,
                fluid_iface_coords=fluid_iface_coords,
                dt=float(dt),
                reference_velocity=float(reference_velocity),
            )
            target_state = pack_fluid_state(operator)
            reaction_coords, target_reaction = _reaction_values_for_configured_probe(
                operator,
                target_probe,
                reaction_coords=reaction_coords,
            )
            coeffs = trial_basis.project_state(target_state, offset=initial)
            values = target_reaction - pre_reaction if bool(incremental) else target_reaction
            coefficient_rows.append(np.asarray(coeffs, dtype=float).reshape(-1))
            reaction_rows.append(np.asarray(values, dtype=float).reshape(-1))
            holdout = bool(stride > 1 and (len(coefficient_rows) % stride == 0))
            validation_mask.append(holdout)
            entries.append(
                {
                    "step": int(pair.step),
                    "coupling_iter": int(pair.coupling_iter),
                    "state_label": str(state_label),
                    "coefficient_norm": float(np.linalg.norm(coeffs)),
                    "reaction_norm": float(np.linalg.norm(target_reaction)),
                    "reaction_increment_norm": float(np.linalg.norm(target_reaction - pre_reaction)),
                    "validation": bool(holdout),
                }
            )
    if reaction_coords is None or not coefficient_rows:
        raise ValueError("No reaction snapshots were available to fit the reduced reaction operator.")
    x = np.vstack(coefficient_rows)
    y = np.vstack(reaction_rows)
    mask = np.asarray(validation_mask, dtype=bool)
    train_mask = ~mask
    if not np.any(train_mask):
        train_mask = np.ones(int(x.shape[0]), dtype=bool)
        mask = np.zeros(int(x.shape[0]), dtype=bool)
    matrix, bias = _fit_affine_map(x[train_mask, :], y[train_mask, :], ridge=float(ridge))
    fitted_all = bias.reshape(1, -1) + x @ matrix.T
    errors = np.asarray(
        [_relative_l2(y[idx, :], fitted_all[idx, :]) for idx in range(int(y.shape[0]))],
        dtype=float,
    )
    train_errors = errors[train_mask]
    validation_errors = errors[mask]
    return {
        "matrix": np.asarray(matrix, dtype=float),
        "bias": np.asarray(bias, dtype=float),
        "coords": np.asarray(reaction_coords, dtype=float),
        "kind": "incremental_point" if bool(incremental) else "point",
        "ridge": float(ridge),
        "entries": entries,
        "fit_count": int(x.shape[0]),
        "train_count": int(np.count_nonzero(train_mask)),
        "validation_count": int(np.count_nonzero(mask)),
        "fit_mean_relative_error": float(np.mean(errors)),
        "fit_max_relative_error": float(np.max(errors)),
        "train_mean_relative_error": float(np.mean(train_errors)) if train_errors.size else float("nan"),
        "train_max_relative_error": float(np.max(train_errors)) if train_errors.size else float("nan"),
        "validation_mean_relative_error": float(np.mean(validation_errors)) if validation_errors.size else float("nan"),
        "validation_max_relative_error": float(np.max(validation_errors)) if validation_errors.size else float("nan"),
    }


def _reaction_output_rows_for_coords(
    operator,
    reaction_coords: np.ndarray,
) -> np.ndarray:
    """Map flattened point-load output positions to fluid velocity global rows."""

    coords = np.asarray(reaction_coords, dtype=float)
    coords_x, gdofs_x = _boundary_field_data(
        operator.dh,
        operator.prob["u_k"].components[0].field_name,
        operator.boundary_tags.interface_tag,
    )
    coords_y, gdofs_y = _boundary_field_data(
        operator.dh,
        operator.prob["u_k"].components[1].field_name,
        operator.boundary_tags.interface_tag,
    )
    if coords_x.shape != coords_y.shape or (coords_x.size and not np.allclose(coords_x, coords_y)):
        raise RuntimeError("Interface velocity component boundary coordinates do not share one ordering.")
    if coords.shape != coords_x.shape:
        raise RuntimeError("Reaction coordinates do not match the fluid interface velocity coordinates.")
    if coords.size and not np.allclose(coords, coords_x):
        rows_x = np.zeros(int(coords.shape[0]), dtype=int)
        rows_y = np.zeros(int(coords.shape[0]), dtype=int)
        for idx, xy in enumerate(coords):
            distances = np.linalg.norm(coords_x - np.asarray(xy, dtype=float).reshape(1, -1), axis=1)
            match = int(np.argmin(distances))
            if float(distances[match]) > 1.0e-10:
                raise RuntimeError("Could not map reaction coordinate to an interface velocity DOF.")
            rows_x[idx] = int(gdofs_x[match])
            rows_y[idx] = int(gdofs_y[match])
    else:
        rows_x = np.asarray(gdofs_x, dtype=int).reshape(-1)
        rows_y = np.asarray(gdofs_y, dtype=int).reshape(-1)
    out = np.empty(2 * int(coords.shape[0]), dtype=int)
    out[0::2] = rows_x
    out[1::2] = rows_y
    return out


def _select_gappy_reaction_positions(
    basis: np.ndarray,
    *,
    candidate_positions: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    """Deterministic oversampled gappy-POD point selection by modal leverage."""

    phi = np.asarray(basis, dtype=float)
    candidates = np.asarray(candidate_positions, dtype=int).reshape(-1)
    if phi.ndim != 2 or phi.shape[1] == 0 or candidates.size == 0:
        return np.zeros((0,), dtype=int)
    count = max(int(phi.shape[1]), int(sample_count))
    count = min(int(count), int(candidates.size))
    leverage = np.sum(phi[candidates, :] ** 2, axis=1)
    order = np.lexsort((candidates, -leverage))
    selected = candidates[order[:count]]
    return np.asarray(selected, dtype=int)


def _fit_sampled_nonlinear_reaction_operator_from_probe_pairs(
    *,
    operator,
    pairs,
    setup,
    fluid_iface_coords: np.ndarray,
    dt: float,
    reference_velocity: float,
    include_intermediate_states: bool,
    modes: int,
    energy: float,
    oversampling: int,
    validation_stride: int,
    checkpoint_dir: Path | None = None,
    checkpoint_steps: str | None = None,
) -> dict[str, Any]:
    """Fit a gappy-POD nonlinear interface reaction reconstruction ``G(Vq)``."""

    reaction_snapshots: list[np.ndarray] = []
    validation_mask: list[bool] = []
    entries: list[dict[str, Any]] = []
    reaction_coords: np.ndarray | None = None
    stride = max(0, int(validation_stride))
    sample_index = 0
    for pair in list(pairs):
        target_paths = _state_probe_sequence_for_pair(
            pair,
            include_pre=True,
            include_intermediate=bool(include_intermediate_states),
            include_post=True,
        )
        for state_label, target_path in target_paths:
            probe = load_fluid_stage_probe(target_path)
            _configure_and_restore_probe(
                operator=operator,
                probe=probe,
                setup=setup,
                fluid_iface_coords=fluid_iface_coords,
                dt=float(dt),
                reference_velocity=float(reference_velocity),
            )
            reaction_coords, values = _reaction_values_for_configured_probe(
                operator,
                probe,
                reaction_coords=reaction_coords,
            )
            sample_index += 1
            reaction_snapshots.append(np.asarray(values, dtype=float).reshape(-1))
            holdout = bool(stride > 1 and (sample_index % stride == 0))
            validation_mask.append(holdout)
            entries.append(
                {
                    "source": "stage_probe",
                    "step": int(pair.step),
                    "coupling_iter": int(pair.coupling_iter),
                    "state_label": str(state_label),
                    "reaction_norm": float(np.linalg.norm(values)),
                    "validation": bool(holdout),
                }
            )
    if checkpoint_dir is not None:
        checkpoint_coords, checkpoint_rows, checkpoint_entries = _accepted_checkpoint_reaction_snapshots(
            operator=operator,
            fluid=operator.prob,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            checkpoint_dir=Path(checkpoint_dir),
            steps_spec=checkpoint_steps,
            reference_velocity=float(reference_velocity),
            reaction_coords=reaction_coords,
        )
        if reaction_coords is None:
            reaction_coords = np.asarray(checkpoint_coords, dtype=float)
        for row, entry in zip(np.asarray(checkpoint_rows, dtype=float), checkpoint_entries):
            sample_index += 1
            holdout = bool(stride > 1 and (sample_index % stride == 0))
            reaction_snapshots.append(np.asarray(row, dtype=float).reshape(-1))
            validation_mask.append(holdout)
            item = dict(entry)
            item["validation"] = bool(holdout)
            entries.append(item)
    if reaction_coords is None or not reaction_snapshots:
        raise ValueError("No reaction snapshots were available to fit the sampled nonlinear reaction operator.")

    y = np.vstack(reaction_snapshots)
    mask = np.asarray(validation_mask, dtype=bool)
    train_mask = ~mask
    if not np.any(train_mask):
        train_mask = np.ones(int(y.shape[0]), dtype=bool)
        mask = np.zeros(int(y.shape[0]), dtype=bool)
    mean = np.mean(y[train_mask, :], axis=0)
    centered_train = (y[train_mask, :] - mean.reshape(1, -1)).T
    u, singular_values, _vt = np.linalg.svd(centered_train, full_matrices=False)
    if singular_values.size:
        cumulative = np.cumsum(singular_values**2)
        total = max(float(cumulative[-1]), 1.0e-30)
        energy_fraction = cumulative / total
    else:
        energy_fraction = np.zeros((0,), dtype=float)
    requested_modes = int(modes)
    if float(energy) > 0.0 and energy_fraction.size:
        requested_modes = min(
            requested_modes if requested_modes > 0 else int(energy_fraction.size),
            int(np.searchsorted(energy_fraction, min(float(energy), 1.0), side="left") + 1),
        )
    if requested_modes <= 0:
        requested_modes = int(min(24, max(1, singular_values.size)))
    requested_modes = max(1, min(int(requested_modes), int(u.shape[1])))
    basis = np.asarray(u[:, :requested_modes], dtype=float)

    output_rows = _reaction_output_rows_for_coords(operator, np.asarray(reaction_coords, dtype=float))
    constrained_rows = _fluid_interface_constrained_velocity_rows(
        prob=operator.prob,
        interface_tag=operator.boundary_tags.interface_tag,
    )
    candidate_positions = np.flatnonzero(np.isin(output_rows, constrained_rows)).astype(int, copy=False)
    if candidate_positions.size == 0:
        raise ValueError("No constrained interface reaction output positions were available for sampling.")
    sample_count = max(int(requested_modes), int(requested_modes) * max(1, int(oversampling)))
    sample_positions = _select_gappy_reaction_positions(
        basis,
        candidate_positions=candidate_positions,
        sample_count=sample_count,
    )
    sample_rows = output_rows[sample_positions].astype(int, copy=False)
    if np.unique(sample_rows).size != int(sample_rows.size):
        unique_rows: list[int] = []
        unique_positions: list[int] = []
        seen: set[int] = set()
        for row, pos in zip(sample_rows, sample_positions):
            if int(row) in seen:
                continue
            seen.add(int(row))
            unique_rows.append(int(row))
            unique_positions.append(int(pos))
        sample_rows = np.asarray(unique_rows, dtype=int)
        sample_positions = np.asarray(unique_positions, dtype=int)
    sample_matrix = basis[sample_positions, :]
    sample_to_coefficients = np.linalg.pinv(sample_matrix, rcond=1.0e-12)
    sample_elements = _fluid_reaction_element_ids_for_velocity_rows(
        operator.prob,
        row_dofs=sample_rows,
    )

    reconstructed = []
    for row in y:
        sample_delta = np.asarray(row[sample_positions] - mean[sample_positions], dtype=float)
        coeffs = sample_to_coefficients @ sample_delta
        reconstructed.append(mean + basis @ coeffs)
    reconstructed_y = np.vstack(reconstructed)
    errors = np.asarray(
        [_relative_l2(y[idx, :], reconstructed_y[idx, :]) for idx in range(int(y.shape[0]))],
        dtype=float,
    )
    train_errors = errors[train_mask]
    validation_errors = errors[mask]
    condition = float(np.linalg.cond(sample_matrix)) if sample_matrix.size else float("inf")
    return {
        "kind": "gappy_pod_point",
        "coords": np.asarray(reaction_coords, dtype=float),
        "basis": np.asarray(basis, dtype=float),
        "mean": np.asarray(mean, dtype=float).reshape(-1),
        "sample_row_dofs": np.asarray(sample_rows, dtype=int),
        "sample_element_ids": np.asarray(sample_elements, dtype=int),
        "sample_to_coefficients": np.asarray(sample_to_coefficients, dtype=float),
        "sample_output_positions": np.asarray(sample_positions, dtype=int),
        "modes": int(requested_modes),
        "oversampling": int(oversampling),
        "sample_count": int(sample_rows.size),
        "sample_element_count": int(sample_elements.size),
        "sample_basis_condition": condition,
        "singular_values": np.asarray(singular_values, dtype=float),
        "energy_fraction": np.asarray(energy_fraction, dtype=float),
        "entries": entries,
        "fit_count": int(y.shape[0]),
        "train_count": int(np.count_nonzero(train_mask)),
        "validation_count": int(np.count_nonzero(mask)),
        "fit_mean_relative_error": float(np.mean(errors)),
        "fit_max_relative_error": float(np.max(errors)),
        "train_mean_relative_error": float(np.mean(train_errors)) if train_errors.size else float("nan"),
        "train_max_relative_error": float(np.max(train_errors)) if train_errors.size else float("nan"),
        "validation_mean_relative_error": float(np.mean(validation_errors)) if validation_errors.size else float("nan"),
        "validation_max_relative_error": float(np.max(validation_errors)) if validation_errors.size else float("nan"),
    }


def _probe_interface_feature_values(
    probe: FluidStageProbe,
    *,
    coords: np.ndarray,
    velocity_scale: float,
) -> np.ndarray:
    """Return displacement and scaled mesh-velocity values on interface coords."""

    target_coords = np.asarray(coords, dtype=float)
    disp = _resample_lookup_to_coords(
        probe.vector_lookup("d_coords", "d_values"),
        target_coords,
    ).values
    velocity = _resample_lookup_to_coords(
        probe.vector_lookup("d_coords", "w_mesh_values"),
        target_coords,
    ).values
    return np.concatenate(
        [
            np.asarray(disp, dtype=float).reshape(-1),
            float(velocity_scale) * np.asarray(velocity, dtype=float).reshape(-1),
        ]
    )


def _fit_interface_impedance_operator_from_probe_pairs(
    *,
    pairs,
    modes: int,
    ridge: float,
    validation_stride: int,
    velocity_scale: float,
) -> dict[str, Any]:
    """Fit a low-rank interface tangent map from kinematic changes to load changes.

    The fitted map is a secant approximation:

        delta(lambda_Gamma) ~= b + Z * POD(delta d_Gamma, dt delta w_Gamma)

    Consecutive coupling iterations within the same time step provide the
    secant pairs.  This is not a replacement for the nonlinear fluid operator;
    it is a compact interface-impedance predictor used to stabilize and
    accelerate the coupling update.
    """

    grouped: dict[int, list[Any]] = {}
    for pair in pairs:
        grouped.setdefault(int(pair.step), []).append(pair)
    for step, items in list(grouped.items()):
        grouped[step] = sorted(items, key=lambda item: int(item.coupling_iter))

    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    validation_mask: list[bool] = []
    entries: list[dict[str, Any]] = []
    reaction_coords: np.ndarray | None = None
    stride = max(0, int(validation_stride))
    sample_index = 0
    for step in sorted(grouped):
        previous_feature: np.ndarray | None = None
        previous_reaction: np.ndarray | None = None
        previous_iter: int | None = None
        for pair in grouped[step]:
            post_probe = load_fluid_stage_probe(pair.post_path)
            reaction_lookup = post_probe.reaction_lookup(kind="point")
            if reaction_lookup is None:
                continue
            if reaction_coords is None:
                reaction_coords = np.asarray(reaction_lookup.coords, dtype=float)
            reaction_values = np.asarray(
                _resample_lookup_to_coords(reaction_lookup, reaction_coords).values,
                dtype=float,
            ).reshape(-1)
            feature_values = _probe_interface_feature_values(
                post_probe,
                coords=reaction_coords,
                velocity_scale=float(velocity_scale),
            )
            if previous_feature is not None and previous_reaction is not None:
                sample_index += 1
                feature_delta = feature_values - previous_feature
                reaction_delta = reaction_values - previous_reaction
                feature_rows.append(feature_delta)
                target_rows.append(reaction_delta)
                holdout = bool(stride > 1 and (sample_index % stride == 0))
                validation_mask.append(holdout)
                entries.append(
                    {
                        "step": int(step),
                        "coupling_iter": int(pair.coupling_iter),
                        "previous_coupling_iter": int(previous_iter if previous_iter is not None else -1),
                        "feature_delta_norm": float(np.linalg.norm(feature_delta)),
                        "reaction_delta_norm": float(np.linalg.norm(reaction_delta)),
                        "validation": bool(holdout),
                    }
                )
            previous_feature = feature_values
            previous_reaction = reaction_values
            previous_iter = int(pair.coupling_iter)

    if reaction_coords is None or not feature_rows:
        raise ValueError("No consecutive interface-reaction pairs were available to fit impedance.")
    features = np.vstack(feature_rows)
    targets = np.vstack(target_rows)
    feature_mean = np.mean(features, axis=0)
    centered = features - feature_mean.reshape(1, -1)
    requested_modes = max(1, min(int(modes), int(centered.shape[0]), int(centered.shape[1])))
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    feature_basis = np.asarray(vt[:requested_modes, :].T, dtype=float)
    reduced_features = centered @ feature_basis

    mask = np.asarray(validation_mask, dtype=bool)
    train_mask = ~mask
    if not np.any(train_mask):
        train_mask = np.ones(int(reduced_features.shape[0]), dtype=bool)
        mask = np.zeros(int(reduced_features.shape[0]), dtype=bool)
    matrix, bias = _fit_affine_map(reduced_features[train_mask, :], targets[train_mask, :], ridge=float(ridge))
    fitted_all = bias.reshape(1, -1) + reduced_features @ matrix.T
    errors = np.asarray(
        [_relative_l2(targets[idx, :], fitted_all[idx, :]) for idx in range(int(targets.shape[0]))],
        dtype=float,
    )
    train_errors = errors[train_mask]
    validation_errors = errors[mask]
    return {
        "matrix": np.asarray(matrix, dtype=float),
        "bias": np.asarray(bias, dtype=float),
        "coords": np.asarray(reaction_coords, dtype=float),
        "feature_basis": np.asarray(feature_basis, dtype=float),
        "feature_mean": np.asarray(feature_mean, dtype=float),
        "velocity_scale": float(velocity_scale),
        "kind": "secant_point",
        "ridge": float(ridge),
        "modes": int(requested_modes),
        "entries": entries,
        "fit_count": int(reduced_features.shape[0]),
        "train_count": int(np.count_nonzero(train_mask)),
        "validation_count": int(np.count_nonzero(mask)),
        "fit_mean_relative_error": float(np.mean(errors)),
        "fit_max_relative_error": float(np.max(errors)),
        "train_mean_relative_error": float(np.mean(train_errors)) if train_errors.size else float("nan"),
        "train_max_relative_error": float(np.max(train_errors)) if train_errors.size else float("nan"),
        "validation_mean_relative_error": float(np.mean(validation_errors)) if validation_errors.size else float("nan"),
        "validation_max_relative_error": float(np.max(validation_errors)) if validation_errors.size else float("nan"),
    }


def _fit_interface_impedance_operator_from_cosim_data(
    *,
    cosim_dir: Path,
    modes: int,
    ridge: float,
    validation_stride: int,
    velocity_scale: float,
    load_key: str = "fluid_load_return_data",
) -> dict[str, Any]:
    """Fit the interface secant map from a full coupled run's coSimData.

    Stage probes are expensive because they store the whole fluid state.  The
    interface impedance only needs the coupling-loop trace:
    ``d_Gamma``, ``w_Gamma`` and the returned interface load.  The production
    exact run stores that trace for every coupling iteration, so this path can
    train the impedance operator on the whole 1..751 run even when rich fluid
    probes only exist for a smaller diagnostic window.
    """

    root = Path(cosim_dir)
    coords_path = root / "coords_interf_fluid.npy"
    disp_path = root / "interface_disp_data.npy"
    velocity_path = root / "interface_velocity_data.npy"
    load_path = root / f"{str(load_key)}.npy"
    iters_path = root / "iters.npy"
    missing = [path for path in (coords_path, disp_path, velocity_path, load_path, iters_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Interface impedance coSimData is incomplete; missing "
            + ", ".join(str(path) for path in missing)
        )

    coords = np.asarray(np.load(coords_path), dtype=float)
    disp = np.asarray(np.load(disp_path), dtype=float)
    velocity = np.asarray(np.load(velocity_path), dtype=float)
    loads = np.asarray(np.load(load_path), dtype=float)
    iters = np.asarray(np.load(iters_path), dtype=int).reshape(-1)
    if coords.ndim != 2 or int(coords.shape[1]) != 2:
        raise ValueError(f"Expected coords_interf_fluid.npy with shape (n, 2), got {coords.shape}.")
    n_outputs = 2 * int(coords.shape[0])
    for name, array in (("interface_disp_data", disp), ("interface_velocity_data", velocity), (str(load_key), loads)):
        if array.ndim != 2 or int(array.shape[0]) != n_outputs:
            raise ValueError(f"Expected {name}.npy with shape ({n_outputs}, n_stages), got {array.shape}.")
    n_stages = int(loads.shape[1])
    if disp.shape[1] != n_stages or velocity.shape[1] != n_stages:
        raise ValueError("coSimData interface arrays do not have the same number of coupling stages.")
    if int(np.sum(iters)) != n_stages:
        raise ValueError(
            f"iters.npy sums to {int(np.sum(iters))}, but interface arrays contain {n_stages} stages."
        )

    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    validation_mask: list[bool] = []
    entries: list[dict[str, Any]] = []
    stride = max(0, int(validation_stride))
    sample_index = 0
    column = 0
    for step_index, count in enumerate(iters, start=1):
        count_int = int(count)
        if count_int <= 0:
            continue
        for local_iter in range(1, count_int):
            previous_col = column + local_iter - 1
            current_col = column + local_iter
            feature_delta = np.concatenate(
                [
                    disp[:, current_col] - disp[:, previous_col],
                    float(velocity_scale) * (velocity[:, current_col] - velocity[:, previous_col]),
                ]
            )
            load_delta = loads[:, current_col] - loads[:, previous_col]
            if not (np.all(np.isfinite(feature_delta)) and np.all(np.isfinite(load_delta))):
                continue
            sample_index += 1
            holdout = bool(stride > 1 and (sample_index % stride == 0))
            feature_rows.append(np.asarray(feature_delta, dtype=float).reshape(-1))
            target_rows.append(np.asarray(load_delta, dtype=float).reshape(-1))
            validation_mask.append(holdout)
            entries.append(
                {
                    "step": int(step_index),
                    "coupling_iter": int(local_iter + 1),
                    "previous_coupling_iter": int(local_iter),
                    "feature_delta_norm": float(np.linalg.norm(feature_delta)),
                    "reaction_delta_norm": float(np.linalg.norm(load_delta)),
                    "validation": bool(holdout),
                }
            )
        column += count_int

    if not feature_rows:
        raise ValueError("No consecutive coSimData interface stages were available to fit impedance.")
    features = np.vstack(feature_rows)
    targets = np.vstack(target_rows)
    feature_mean = np.mean(features, axis=0)
    centered = features - feature_mean.reshape(1, -1)
    requested_modes = max(1, min(int(modes), int(centered.shape[0]), int(centered.shape[1])))
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    feature_basis = np.asarray(vt[:requested_modes, :].T, dtype=float)
    reduced_features = centered @ feature_basis

    mask = np.asarray(validation_mask, dtype=bool)
    train_mask = ~mask
    if not np.any(train_mask):
        train_mask = np.ones(int(reduced_features.shape[0]), dtype=bool)
        mask = np.zeros(int(reduced_features.shape[0]), dtype=bool)
    matrix, bias = _fit_affine_map(reduced_features[train_mask, :], targets[train_mask, :], ridge=float(ridge))
    fitted_all = bias.reshape(1, -1) + reduced_features @ matrix.T
    errors = np.asarray(
        [_relative_l2(targets[idx, :], fitted_all[idx, :]) for idx in range(int(targets.shape[0]))],
        dtype=float,
    )
    train_errors = errors[train_mask]
    validation_errors = errors[mask]
    return {
        "matrix": np.asarray(matrix, dtype=float),
        "bias": np.asarray(bias, dtype=float),
        "coords": np.asarray(coords, dtype=float),
        "feature_basis": np.asarray(feature_basis, dtype=float),
        "feature_mean": np.asarray(feature_mean, dtype=float),
        "velocity_scale": float(velocity_scale),
        "kind": "cosim_secant_point",
        "source": str(root),
        "load_key": str(load_key),
        "ridge": float(ridge),
        "modes": int(requested_modes),
        "entries": entries,
        "fit_count": int(reduced_features.shape[0]),
        "train_count": int(np.count_nonzero(train_mask)),
        "validation_count": int(np.count_nonzero(mask)),
        "fit_mean_relative_error": float(np.mean(errors)),
        "fit_max_relative_error": float(np.max(errors)),
        "train_mean_relative_error": float(np.mean(train_errors)) if train_errors.size else float("nan"),
        "train_max_relative_error": float(np.max(train_errors)) if train_errors.size else float("nan"),
        "validation_mean_relative_error": float(np.mean(validation_errors)) if validation_errors.size else float("nan"),
        "validation_max_relative_error": float(np.max(validation_errors)) if validation_errors.size else float("nan"),
    }


def _fit_affine_map(coefficients: np.ndarray, values: np.ndarray, *, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(coefficients, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or int(x.shape[0]) != int(y.shape[0]):
        raise ValueError("affine fit expects X=(n_samples,n_modes), Y=(n_samples,n_outputs).")
    design = np.column_stack([np.ones(int(x.shape[0]), dtype=float), x])
    gram = design.T @ design
    reg = np.zeros_like(gram)
    if float(ridge) > 0.0:
        reg[1:, 1:] = float(ridge) * np.eye(int(x.shape[1]), dtype=float)
    rhs = design.T @ y
    try:
        solution = np.linalg.solve(gram + reg, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(gram + reg, rhs, rcond=None)[0]
    bias = np.asarray(solution[0, :], dtype=float).reshape(-1)
    matrix = np.asarray(solution[1:, :].T, dtype=float)
    return matrix, bias


def _training_probe_roots(args: argparse.Namespace, *, fallback_to_probe_dir: bool) -> list[Path]:
    roots: list[Path] = []
    if args.training_probe_dir is not None:
        roots.append(Path(args.training_probe_dir))
    roots.extend(Path(item) for item in list(args.extra_training_probe_dir or []))
    if not roots and bool(fallback_to_probe_dir):
        roots.append(Path(args.probe_dir))
    return roots


def _find_fluid_stage_probe_pairs_from_roots(
    roots: list[Path],
    *,
    final_only: bool,
    steps: set[int] | None,
) -> list[Any]:
    """Collect stage pairs from one or more roots with deterministic de-duplication."""

    unique: dict[tuple[int, int], Any] = {}
    for root in roots:
        for pair in find_fluid_stage_probe_pairs(root, final_only=bool(final_only), steps=steps):
            unique.setdefault((int(pair.step), int(pair.coupling_iter)), pair)
    return [unique[key] for key in sorted(unique)]


def _stratified_probe_pair_subset(pairs: list[Any], max_pairs: int) -> list[Any]:
    pair_list = list(pairs)
    limit = int(max_pairs)
    if limit <= 0 or len(pair_list) <= limit:
        return pair_list
    if limit <= 1:
        return [pair_list[-1]]
    indices = np.unique(np.linspace(0, len(pair_list) - 1, num=limit, dtype=int))
    selected = [pair_list[int(index)] for index in indices]
    if len(selected) < limit:
        selected_keys = {(int(item.step), int(item.coupling_iter)) for item in selected}
        for item in pair_list:
            key = (int(item.step), int(item.coupling_iter))
            if key in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(key)
            if len(selected) >= limit:
                break
    return sorted(selected[:limit], key=lambda item: (int(item.step), int(item.coupling_iter)))


def _save_all_state_training_database(
    path: Path,
    *,
    batch,
    residual_row_dofs: np.ndarray,
    residual_snapshots: np.ndarray,
    jacobian_action_snapshots: np.ndarray | None,
    coefficients: np.ndarray,
    state_offsets: np.ndarray,
    summary: dict[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(1, dtype=int),
        "fluid_stage_snapshot_schema_version": np.asarray(int(batch.schema_version), dtype=int),
        "state": np.asarray(batch.state, dtype=float),
        "state_offsets": np.asarray(state_offsets, dtype=float),
        "state_increments": np.asarray(batch.state - state_offsets, dtype=float),
        "free_dofs": np.asarray(batch.free_dofs, dtype=int),
        "fixed_dofs": np.asarray(batch.fixed_dofs, dtype=int),
        "fixed_values": np.asarray(batch.fixed_values, dtype=float),
        "residual_row_dofs": np.asarray(residual_row_dofs, dtype=int),
        "residual_snapshots": np.asarray(residual_snapshots, dtype=float),
        "coefficients": np.asarray(coefficients, dtype=float),
        "metadata_json": np.asarray(json.dumps(batch.metadata)),
        "summary_json": np.asarray(json.dumps(summary)),
        "dvms_keys_json": np.asarray(json.dumps(sorted(batch.dvms.keys()))),
        "dvms_shapes_json": np.asarray(
            json.dumps({key: list(batch.dvms_shapes[key]) for key in sorted(batch.dvms.keys())})
        ),
    }
    for key in _FIELD_KEYS:
        payload[key] = np.asarray(getattr(batch, key), dtype=float)
    for idx, key in enumerate(sorted(batch.dvms.keys())):
        payload[f"dvms_{idx}"] = np.asarray(batch.dvms[key], dtype=float)
    if batch.reaction_coords is not None:
        payload["reaction_coords"] = np.asarray(batch.reaction_coords, dtype=float)
    if batch.reaction_values is not None:
        payload["reaction_values"] = np.asarray(batch.reaction_values, dtype=float)
    if jacobian_action_snapshots is not None:
        payload["jacobian_action_snapshots"] = np.asarray(jacobian_action_snapshots, dtype=float)
    np.savez_compressed(target, **payload)


def _append_exact_probe_all_state(
    *,
    writer: FluidStageSnapshotWriter,
    residuals: list[np.ndarray],
    jacobian_actions: list[np.ndarray],
    coefficients: list[np.ndarray],
    offsets: list[np.ndarray],
    operator,
    probe: FluidStageProbe,
    pair,
    state_label: str,
    trial_basis,
    initial_state: np.ndarray,
    free_dofs: np.ndarray,
    setup,
    fluid_iface_coords: np.ndarray,
    dt: float,
    reference_velocity: float,
    include_jacobian_actions: bool,
    metadata: dict[str, Any] | None = None,
) -> None:
    _configure_and_restore_probe(
        operator=operator,
        probe=probe,
        setup=setup,
        fluid_iface_coords=fluid_iface_coords,
        dt=float(dt),
        reference_velocity=float(reference_velocity),
    )
    coeffs = trial_basis.project_state(pack_fluid_state(operator), offset=initial_state)
    assembly = operator.assemble(need_matrix=True, convention="newton", refresh_predicted=True)
    if assembly.matrix is None:
        raise RuntimeError("All-state export requires a fluid Jacobian.")
    rows = np.asarray(free_dofs, dtype=int).reshape(-1)
    residual = np.asarray(assembly.residual[rows], dtype=float).reshape(-1)
    trial_jacobian = np.asarray(assembly.matrix[rows, :] @ np.asarray(trial_basis.basis, dtype=float), dtype=float)
    reaction = operator.reaction_loads(refresh_state=False)
    record_metadata = {
        "step": int(pair.step),
        "coupling_iter": int(pair.coupling_iter),
        "state_kind": str(state_label),
        "source": "exact_probe",
        "probe_path": str(probe.path),
        "coefficient_norm": float(np.linalg.norm(coeffs)),
        "residual_norm": float(np.linalg.norm(residual)),
    }
    if metadata:
        record_metadata.update(dict(metadata))
    writer.append_from_operator(
        operator,
        reaction_loads=reaction,
        include_reaction=True,
        metadata=record_metadata,
    )
    residuals.append(residual)
    if bool(include_jacobian_actions):
        jacobian_actions.append(trial_jacobian)
    coefficients.append(np.asarray(coeffs, dtype=float).reshape(-1))
    offsets.append(np.asarray(initial_state, dtype=float).reshape(-1))


def _append_lspg_coefficient_all_state(
    *,
    writer: FluidStageSnapshotWriter,
    residuals: list[np.ndarray],
    jacobian_actions: list[np.ndarray],
    coefficients: list[np.ndarray],
    offsets: list[np.ndarray],
    operator,
    trial_space,
    pair,
    coefficient: np.ndarray,
    trajectory_item: dict[str, Any] | None,
    state_label: str,
    free_dofs: np.ndarray,
    fluid: dict[str, object],
    bossak: dict[str, float],
    include_jacobian_actions: bool,
) -> None:
    hook = _make_stage_acceleration_hook(
        fluid=fluid,
        bossak=bossak,
        preserve_seed_on_first_zero=True,
    )
    verifier = FluidLSPGVerifier(
        operator=operator,
        trial_space=trial_space,
        row_dofs=np.asarray(free_dofs, dtype=int).reshape(-1),
        state_update_hook=hook,
        nonlinear_update_hook=operator.update_oss_after_nonlinear_update,
    )
    coeffs = np.asarray(coefficient, dtype=float).reshape(-1)
    system = verifier.assemble_system(coeffs, refresh_predicted=True)
    reaction = operator.reaction_loads(refresh_state=False)
    record_metadata = {
        "step": int(pair.step),
        "coupling_iter": int(pair.coupling_iter),
        "state_kind": str(state_label),
        "source": "lspg_trajectory",
        "coefficient_norm": float(np.linalg.norm(coeffs)),
        "residual_norm": float(system.residual_norm),
    }
    if trajectory_item:
        for key, value in dict(trajectory_item).items():
            if key == "coefficients":
                continue
            try:
                record_metadata[f"lspg_{key}"] = float(value)
            except (TypeError, ValueError):
                record_metadata[f"lspg_{key}"] = str(value)
    writer.append_from_operator(
        operator,
        reaction_loads=reaction,
        include_reaction=True,
        metadata=record_metadata,
    )
    residuals.append(np.asarray(system.residual, dtype=float).reshape(-1))
    if bool(include_jacobian_actions):
        jacobian_actions.append(np.asarray(system.trial_jacobian, dtype=float))
    coefficients.append(coeffs.copy())
    offsets.append(np.asarray(trial_space.offset, dtype=float).reshape(-1))


def _export_all_state_training_database_from_probe_pairs(
    *,
    path: Path,
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
    block_scale_relative_floor: float,
    trajectory_source: str,
    include_jacobian_actions: bool,
    pair_weights: dict[tuple[int, int], float] | None = None,
) -> dict[str, Any]:
    pair_list = list(pairs)
    if not pair_list:
        raise ValueError("Cannot export all-state training data without probe pairs.")
    writer = FluidStageSnapshotWriter()
    residuals: list[np.ndarray] = []
    jacobian_actions: list[np.ndarray] = []
    coefficients: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    free = np.asarray(free_dofs, dtype=int).reshape(-1)
    exact_newton_states = 0
    lspg_states = 0
    phase_t0 = time.perf_counter()
    _progress(
        "export all-state training database: "
        f"pairs={len(pair_list)} trajectory={trajectory_source} path={Path(path)}"
    )
    for pair_index, pair in enumerate(pair_list, start=1):
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
        initial_state = pack_fluid_state(operator)
        trial_space = trial_basis.make_trial_space(operator, offset=initial_state)
        initial_coefficients = np.zeros(int(trial_space.n_modes), dtype=float)
        weight = max(_pair_weight(pair, pair_weights), 0.0)
        _append_lspg_coefficient_all_state(
            writer=writer,
            residuals=residuals,
            jacobian_actions=jacobian_actions,
            coefficients=coefficients,
            offsets=offsets,
            operator=operator,
            trial_space=trial_space,
            pair=pair,
            coefficient=initial_coefficients,
            trajectory_item={"iteration": 0.0, "residual_norm": float("nan"), "step_norm": 0.0},
            state_label="pre_fluid_solve",
            free_dofs=free,
            fluid=fluid,
            bossak=bossak,
            include_jacobian_actions=bool(include_jacobian_actions),
        )

        exact_newton_paths = _exact_newton_iterate_probe_paths(pair)
        if exact_newton_paths:
            for newton_index, newton_path in enumerate(exact_newton_paths, start=1):
                probe = load_fluid_stage_probe(newton_path)
                _append_exact_probe_all_state(
                    writer=writer,
                    residuals=residuals,
                    jacobian_actions=jacobian_actions,
                    coefficients=coefficients,
                    offsets=offsets,
                    operator=operator,
                    probe=probe,
                    pair=pair,
                    state_label=f"newton_iter{newton_index:04d}",
                    trial_basis=trial_basis,
                    initial_state=initial_state,
                    free_dofs=free,
                    setup=setup,
                    fluid_iface_coords=fluid_iface_coords,
                    dt=float(dt),
                    reference_velocity=float(reference_velocity),
                    include_jacobian_actions=bool(include_jacobian_actions),
                    metadata={"training_weight": float(weight)},
                )
                exact_newton_states += 1
        else:
            row_weights = None
            if bool(block_scale):
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
            result = lspg.solve(
                initial_coefficients,
                max_iterations=int(max_iterations),
                residual_tol=float(residual_tol),
                line_search=bool(line_search),
            )
            for item in result.trajectory:
                if "coefficients" not in item:
                    continue
                coeffs = np.asarray(item["coefficients"], dtype=float).reshape(-1)
                if float(np.linalg.norm(coeffs - initial_coefficients)) <= 1.0e-14:
                    continue
                _configure_and_restore_probe(
                    operator=operator,
                    probe=pre_probe,
                    setup=setup,
                    fluid_iface_coords=fluid_iface_coords,
                    dt=float(dt),
                    reference_velocity=float(reference_velocity),
                )
                iteration = int(float(item.get("iteration", 0.0)))
                _append_lspg_coefficient_all_state(
                    writer=writer,
                    residuals=residuals,
                    jacobian_actions=jacobian_actions,
                    coefficients=coefficients,
                    offsets=offsets,
                    operator=operator,
                    trial_space=trial_space,
                    pair=pair,
                    coefficient=coeffs,
                    trajectory_item=item,
                    state_label=f"lspg_iter{iteration:04d}",
                    free_dofs=free,
                    fluid=fluid,
                    bossak=bossak,
                    include_jacobian_actions=bool(include_jacobian_actions),
                )
                lspg_states += 1

        _append_exact_probe_all_state(
            writer=writer,
            residuals=residuals,
            jacobian_actions=jacobian_actions,
            coefficients=coefficients,
            offsets=offsets,
            operator=operator,
            probe=post_probe,
            pair=pair,
            state_label="post_fluid_solve",
            trial_basis=trial_basis,
            initial_state=initial_state,
            free_dofs=free,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
            include_jacobian_actions=bool(include_jacobian_actions),
            metadata={"training_weight": float(weight)},
        )
        if pair_index == 1 or pair_index == len(pair_list) or pair_index % 5 == 0:
            _progress(
                "export all-state training database: "
                f"{pair_index}/{len(pair_list)} "
                f"step={int(pair.step)} iter={int(pair.coupling_iter)} "
                f"states={len(coefficients)} elapsed={time.perf_counter() - phase_t0:.2f}s"
            )

    batch = writer.to_batch()
    residual_matrix = np.column_stack(residuals)
    coefficient_matrix = np.column_stack(coefficients)
    offset_matrix = np.column_stack(offsets)
    jacobian_tensor = np.stack(jacobian_actions, axis=2) if bool(include_jacobian_actions) else None
    coeff_mean = np.mean(coefficient_matrix, axis=1)
    coeff_scale = np.maximum(np.std(coefficient_matrix, axis=1), 1.0e-12)
    coeff_distances = np.linalg.norm((coefficient_matrix - coeff_mean[:, None]) / coeff_scale[:, None], axis=0)
    summary = {
        "pairs": int(len(pair_list)),
        "states": int(coefficient_matrix.shape[1]),
        "pre_states": int(sum(1 for item in batch.metadata if item.get("state_kind") == "pre_fluid_solve")),
        "post_states": int(sum(1 for item in batch.metadata if item.get("state_kind") == "post_fluid_solve")),
        "exact_newton_states": int(exact_newton_states),
        "lspg_states": int(lspg_states),
        "residual_rows": int(free.size),
        "modes": int(trial_basis.n_modes),
        "velocity_modes": int(trial_basis.n_velocity_modes),
        "velocity_pod_modes": int(trial_basis.n_velocity_pod_modes),
        "velocity_supremizer_modes": int(trial_basis.n_velocity_supremizer_modes),
        "pressure_modes": int(trial_basis.n_pressure_modes),
        "supremizer_riesz": str(trial_basis.supremizer_riesz),
        "include_jacobian_actions": bool(include_jacobian_actions),
        "trajectory_source": str(trajectory_source),
        "coefficient_mean_norm": float(np.linalg.norm(coeff_mean)),
        "coefficient_scaled_distance_max": float(np.max(coeff_distances)) if coeff_distances.size else 0.0,
        "coefficient_scaled_distance_mean": float(np.mean(coeff_distances)) if coeff_distances.size else 0.0,
    }
    _save_all_state_training_database(
        Path(path),
        batch=batch,
        residual_row_dofs=free,
        residual_snapshots=residual_matrix,
        jacobian_action_snapshots=jacobian_tensor,
        coefficients=coefficient_matrix,
        state_offsets=offset_matrix,
        summary=summary,
    )
    _progress(
        "export all-state training database: done "
        f"states={int(coefficient_matrix.shape[1])} path={Path(path)} "
        f"elapsed={time.perf_counter() - phase_t0:.2f}s"
    )
    return summary


def _project_training_state_coefficients_from_probe_pairs(
    *,
    operator,
    pairs,
    trial_basis,
    setup,
    fluid_iface_coords: np.ndarray,
    dt: float,
    reference_velocity: float,
    include_intermediate_states: bool,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for pair in pairs:
        pre_probe = load_fluid_stage_probe(pair.pre_path)
        _configure_and_restore_probe(
            operator=operator,
            probe=pre_probe,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        )
        initial = pack_fluid_state(operator)
        columns.append(np.zeros(int(trial_basis.n_modes), dtype=float))
        for _label, state_path in _state_probe_sequence_for_pair(
            pair,
            include_pre=False,
            include_intermediate=bool(include_intermediate_states),
            include_post=True,
        ):
            probe = load_fluid_stage_probe(state_path)
            _configure_and_restore_probe(
                operator=operator,
                probe=probe,
                setup=setup,
                fluid_iface_coords=fluid_iface_coords,
                dt=float(dt),
                reference_velocity=float(reference_velocity),
            )
            columns.append(trial_basis.project_state(pack_fluid_state(operator), offset=initial))
    if not columns:
        return np.zeros((int(trial_basis.n_modes), 0), dtype=float)
    return np.column_stack(columns)


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


def _fit_greedy_nonnegative_weights(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    max_weight: float,
    relative_tol: float = 1.0e-8,
    max_active: int | None = None,
) -> tuple[np.ndarray, float]:
    """Greedy empirical-cubature fit with nonnegative active-set refits."""

    A = np.asarray(matrix, dtype=float)
    b = np.asarray(target, dtype=float).reshape(-1)
    if A.ndim != 2 or int(A.shape[0]) != int(b.size):
        raise ValueError("Empirical cubature fit matrix and target have incompatible shapes.")
    n_candidates = int(A.shape[1])
    if n_candidates == 0:
        raise ValueError("Cannot fit element weights with zero candidate elements.")
    b_norm = max(float(np.linalg.norm(b)), 1.0e-15)
    col_norms = np.linalg.norm(A, axis=0)
    selectable = col_norms > 1.0e-14
    if not np.any(selectable):
        return np.ones(n_candidates, dtype=float), float("inf")
    active_limit = int(max_active) if max_active is not None else min(n_candidates, max(128, min(1000, 4 * int(A.shape[0]))))
    active_limit = max(1, min(active_limit, n_candidates))
    active: list[int] = []
    active_mask = np.zeros(n_candidates, dtype=bool)
    residual = b.copy()
    best_weights = np.zeros(n_candidates, dtype=float)
    best_error = float(np.linalg.norm(residual) / b_norm)
    stalled = 0

    for _iteration in range(active_limit):
        correlations = A.T @ residual
        scores = np.full(n_candidates, -np.inf, dtype=float)
        valid = selectable & (~active_mask)
        if not np.any(valid):
            break
        scores[valid] = correlations[valid] / col_norms[valid]
        chosen = int(np.argmax(scores))
        if not np.isfinite(scores[chosen]):
            break
        if scores[chosen] <= 0.0 and active:
            abs_scores = np.full(n_candidates, -np.inf, dtype=float)
            abs_scores[valid] = np.abs(correlations[valid]) / col_norms[valid]
            chosen = int(np.argmax(abs_scores))
            if not np.isfinite(abs_scores[chosen]):
                break
        active.append(chosen)
        active_mask[chosen] = True
        A_active = A[:, active]
        try:
            from scipy.optimize import nnls

            weights_active, _resnorm = nnls(A_active, b, maxiter=max(3 * len(active), 1))
        except Exception:
            weights_active, *_ = np.linalg.lstsq(A_active, b, rcond=None)
            weights_active = np.maximum(np.asarray(weights_active, dtype=float).reshape(-1), 0.0)
        weights_active = np.clip(np.asarray(weights_active, dtype=float).reshape(-1), 0.0, max(float(max_weight), 1.0e-12))
        residual = b - A_active @ weights_active
        rel_error = float(np.linalg.norm(residual) / b_norm)
        if rel_error < best_error * (1.0 - 1.0e-10):
            best_error = rel_error
            best_weights[:] = 0.0
            best_weights[np.asarray(active, dtype=int)] = weights_active
            stalled = 0
        else:
            stalled += 1
        if rel_error <= float(relative_tol):
            break
        if stalled >= 32:
            break

    if not np.any(best_weights > 0.0):
        return np.ones(n_candidates, dtype=float), float("inf")
    return best_weights, float(best_error)


def _element_ids_touching_dofs(operator, dofs: np.ndarray) -> np.ndarray:
    dof_values = np.asarray(dofs, dtype=int).reshape(-1)
    if dof_values.size == 0:
        return np.zeros(0, dtype=int)
    touched = np.zeros(int(operator.mesh.n_elements), dtype=bool)
    for field_name in ("ux", "uy", "p"):
        element_map = np.asarray(operator.dh.element_maps[str(field_name)], dtype=int)
        touched |= np.isin(element_map, dof_values).any(axis=1)
    return np.flatnonzero(touched).astype(int, copy=False)


def _boundary_element_ids(operator, *, tag: str, fields: tuple[str, ...]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for field_name in fields:
        try:
            _coords, gdofs = _boundary_field_data(operator.dh, str(field_name), str(tag))
        except Exception:
            continue
        rows.append(np.asarray(gdofs, dtype=int).reshape(-1))
    if not rows:
        return np.zeros(0, dtype=int)
    return _element_ids_touching_dofs(operator, np.unique(np.concatenate(rows)))


def _append_scaled_fit_block(
    *,
    blocks: list[np.ndarray],
    targets: list[np.ndarray],
    element_importance: np.ndarray | None,
    contribution_by_element: np.ndarray,
    target: np.ndarray,
    scale: float,
) -> None:
    contributions = np.asarray(contribution_by_element, dtype=float)
    target_vec = np.asarray(target, dtype=float).reshape(-1)
    if contributions.ndim != 2:
        raise ValueError("Empirical-cubature fit contributions must be element-by-feature matrices.")
    if int(contributions.shape[1]) != int(target_vec.size):
        raise ValueError("Empirical-cubature fit contribution/target shapes are incompatible.")
    target_norm = float(np.linalg.norm(target_vec))
    if target_norm <= 1.0e-14:
        return
    block_scale = float(scale) / max(target_norm, 1.0e-12)
    scaled = contributions * block_scale
    blocks.append(scaled.T)
    targets.append(target_vec * block_scale)
    if element_importance is not None:
        element_importance += np.sum(scaled * scaled, axis=1)


def _append_reduced_operator_fit_blocks(
    *,
    blocks: list[np.ndarray],
    targets: list[np.ndarray],
    element_importance: np.ndarray | None,
    residual_by_element: np.ndarray,
    tangent_by_element: np.ndarray,
    trial_basis,
    pair_weight: float,
    split_velocity_pressure: bool,
) -> None:
    residual = np.asarray(residual_by_element, dtype=float)
    tangent = np.asarray(tangent_by_element, dtype=float)
    scale = math.sqrt(max(float(pair_weight), 0.0))
    if not bool(split_velocity_pressure):
        _append_scaled_fit_block(
            blocks=blocks,
            targets=targets,
            element_importance=element_importance,
            contribution_by_element=residual,
            target=np.sum(residual, axis=0),
            scale=scale,
        )
        _append_scaled_fit_block(
            blocks=blocks,
            targets=targets,
            element_importance=element_importance,
            contribution_by_element=tangent.reshape(int(tangent.shape[0]), -1),
            target=np.sum(tangent, axis=0).reshape(-1),
            scale=scale,
        )
        return

    n_velocity = int(getattr(trial_basis, "n_velocity_modes", 0))
    n_pressure = int(getattr(trial_basis, "n_pressure_modes", 0))
    mode_slices: list[tuple[slice, str]] = []
    if n_velocity > 0:
        mode_slices.append((slice(0, n_velocity), "velocity"))
    if n_pressure > 0:
        mode_slices.append((slice(n_velocity, n_velocity + n_pressure), "pressure"))
    for row_slice, _name in mode_slices:
        residual_part = residual[:, row_slice]
        _append_scaled_fit_block(
            blocks=blocks,
            targets=targets,
            element_importance=element_importance,
            contribution_by_element=residual_part,
            target=np.sum(residual_part, axis=0),
            scale=scale,
        )
    for row_slice, _row_name in mode_slices:
        for col_slice, _col_name in mode_slices:
            tangent_part = tangent[:, row_slice, col_slice]
            _append_scaled_fit_block(
                blocks=blocks,
                targets=targets,
                element_importance=element_importance,
                contribution_by_element=tangent_part.reshape(int(tangent_part.shape[0]), -1),
                target=np.sum(tangent_part, axis=0).reshape(-1),
                scale=scale,
            )


def _select_diverse_columns(matrix: np.ndarray, *, count: int) -> np.ndarray:
    A = np.asarray(matrix, dtype=float)
    n_columns = int(A.shape[1])
    target_count = max(0, min(int(count), n_columns))
    if target_count >= n_columns:
        return np.arange(n_columns, dtype=int)
    if target_count <= 0:
        return np.zeros(0, dtype=int)
    norms = np.linalg.norm(A, axis=0)
    nonzero = np.flatnonzero(norms > 1.0e-14)
    if nonzero.size <= target_count:
        return nonzero.astype(int, copy=False)
    pool_count = min(int(nonzero.size), max(4 * target_count, target_count, 256))
    pool = nonzero[np.argsort(norms[nonzero])[::-1][:pool_count]]
    normalized = A[:, pool] / np.maximum(norms[pool], 1.0e-14)[None, :]
    try:
        from scipy.linalg import qr

        _q, _r, pivots = qr(normalized, pivoting=True, mode="economic")
        selected = pool[np.asarray(pivots[:target_count], dtype=int)]
    except Exception:
        selected = pool[:target_count]
    return np.asarray(selected, dtype=int)


def _fit_bounded_regularized_weights(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    max_weight: float,
    prior: np.ndarray,
    regularization: float,
    weight_sum_target: float | None = None,
    weight_sum_scale: float = 0.0,
    solver: str = "bounded",
) -> tuple[np.ndarray, float]:
    A = np.asarray(matrix, dtype=float)
    b = np.asarray(target, dtype=float).reshape(-1)
    prior_values = np.asarray(prior, dtype=float).reshape(-1)
    if A.ndim != 2 or int(A.shape[0]) != int(b.size):
        raise ValueError("Regularized empirical-cubature matrix/target shapes are incompatible.")
    if int(A.shape[1]) != int(prior_values.size):
        raise ValueError("Regularized empirical-cubature prior size is incompatible.")
    if A.shape[1] == 0:
        return np.zeros(0, dtype=float), float("inf")

    rows = [A]
    rhs = [b]
    target_norm = max(float(np.linalg.norm(b)), 1.0e-15)
    prior_norm = max(float(np.linalg.norm(prior_values)), 1.0)
    if float(regularization) > 0.0:
        reg_scale = math.sqrt(float(regularization)) * target_norm / prior_norm
        rows.append(reg_scale * np.eye(int(A.shape[1]), dtype=float))
        rhs.append(reg_scale * prior_values)
    if weight_sum_target is not None and float(weight_sum_scale) > 0.0:
        sum_target = float(weight_sum_target)
        sum_scale = math.sqrt(float(weight_sum_scale)) * target_norm / max(abs(sum_target), 1.0)
        rows.append(sum_scale * np.ones((1, int(A.shape[1])), dtype=float))
        rhs.append(np.asarray([sum_scale * sum_target], dtype=float))
    A_aug = np.vstack(rows)
    b_aug = np.concatenate(rhs)
    upper = np.full(int(A.shape[1]), max(float(max_weight), 1.0e-12), dtype=float)
    solver_key = str(solver).strip().lower().replace("_", "-")
    if solver_key in {"ridge-clip", "fast-ridge", "normal-clip"}:
        normal = A_aug.T @ A_aug
        rhs_normal = A_aug.T @ b_aug
        ridge = max(1.0e-12 * float(np.trace(normal)) / max(int(normal.shape[0]), 1), 1.0e-14)
        normal = normal + ridge * np.eye(int(normal.shape[0]), dtype=float)
        try:
            weights = np.linalg.solve(normal, rhs_normal)
        except np.linalg.LinAlgError:
            weights, *_ = np.linalg.lstsq(normal, rhs_normal, rcond=None)
        weights = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, upper)
    else:
        try:
            from scipy.optimize import lsq_linear

            result = lsq_linear(
                A_aug,
                b_aug,
                bounds=(np.zeros(int(A.shape[1]), dtype=float), upper),
                lsmr_tol="auto",
                max_iter=500,
            )
            weights = np.asarray(result.x, dtype=float).reshape(-1)
        except Exception:
            weights, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
            weights = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, upper)
    residual = A @ weights - b
    rel_error = float(np.linalg.norm(residual) / target_norm)
    return weights, rel_error


def _refit_selected_block_weights(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    selected: np.ndarray,
    initial_weights: np.ndarray,
    block_groups: list[dict[str, Any]],
    max_weight: float,
    regularization: float,
    weight_sum_scale: float,
    solver: str = "bounded",
) -> np.ndarray:
    A = np.asarray(matrix, dtype=float)
    b = np.asarray(target, dtype=float).reshape(-1)
    selected_indices = np.asarray(selected, dtype=int).reshape(-1)
    prior = np.asarray(initial_weights, dtype=float).reshape(-1)
    if selected_indices.size == 0:
        return np.zeros(0, dtype=float)
    if int(prior.size) != int(selected_indices.size):
        raise ValueError("Selected block refit prior size is incompatible.")
    selected_position = {int(element_local): idx for idx, element_local in enumerate(selected_indices)}
    frozen = np.zeros(int(selected_indices.size), dtype=bool)
    for group in block_groups:
        if not bool(group.get("mandatory", False)):
            continue
        group_local = np.asarray(group["selected_local"], dtype=int).reshape(-1)
        for element_local in group_local:
            pos = selected_position.get(int(element_local))
            if pos is not None:
                frozen[int(pos)] = True
    adjustable = ~frozen
    if not np.any(adjustable):
        return prior.copy()

    frozen_indices = selected_indices[frozen]
    adjustable_indices = selected_indices[adjustable]
    frozen_weights = prior[frozen]
    adjustable_prior = prior[adjustable]
    b_effective = b - A[:, frozen_indices] @ frozen_weights if frozen_indices.size else b.copy()
    A_adjustable = A[:, adjustable_indices]
    rows = [A_adjustable]
    rhs = [b_effective]
    target_norm = max(float(np.linalg.norm(b)), 1.0e-15)
    prior_norm = max(float(np.linalg.norm(adjustable_prior)), 1.0)
    if float(regularization) > 0.0:
        reg_scale = math.sqrt(float(regularization)) * target_norm / prior_norm
        rows.append(reg_scale * np.eye(int(adjustable_indices.size), dtype=float))
        rhs.append(reg_scale * adjustable_prior)
    if float(weight_sum_scale) > 0.0:
        adjustable_position = {int(element_local): idx for idx, element_local in enumerate(adjustable_indices)}
        for group in block_groups:
            if bool(group.get("mandatory", False)):
                continue
            group_local = np.asarray(group["selected_local"], dtype=int).reshape(-1)
            positions = [
                adjustable_position[int(element_local)]
                for element_local in group_local
                if int(element_local) in adjustable_position
            ]
            if not positions:
                continue
            row = np.zeros(int(adjustable_indices.size), dtype=float)
            row[np.asarray(positions, dtype=int)] = 1.0
            sum_target = float(group["weight_sum_target"])
            sum_scale = math.sqrt(float(weight_sum_scale)) * target_norm / max(abs(sum_target), 1.0)
            rows.append(sum_scale * row.reshape(1, -1))
            rhs.append(np.asarray([sum_scale * sum_target], dtype=float))
    A_aug = np.vstack(rows)
    b_aug = np.concatenate(rhs)
    upper = np.full(int(adjustable_indices.size), max(float(max_weight), 1.0e-12), dtype=float)
    solver_key = str(solver).strip().lower().replace("_", "-")
    if solver_key in {"ridge-clip", "fast-ridge", "normal-clip"}:
        normal = A_aug.T @ A_aug
        rhs_normal = A_aug.T @ b_aug
        ridge = max(1.0e-12 * float(np.trace(normal)) / max(int(normal.shape[0]), 1), 1.0e-14)
        normal = normal + ridge * np.eye(int(normal.shape[0]), dtype=float)
        try:
            adjustable_out = np.linalg.solve(normal, rhs_normal)
        except np.linalg.LinAlgError:
            adjustable_out, *_ = np.linalg.lstsq(normal, rhs_normal, rcond=None)
        adjustable_out = np.clip(np.asarray(adjustable_out, dtype=float).reshape(-1), 0.0, upper)
    else:
        try:
            from scipy.optimize import lsq_linear

            result = lsq_linear(
                A_aug,
                b_aug,
                bounds=(np.zeros(int(adjustable_indices.size), dtype=float), upper),
                lsmr_tol="auto",
                max_iter=700,
            )
            adjustable_out = np.asarray(result.x, dtype=float).reshape(-1)
        except Exception:
            adjustable_out, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
            adjustable_out = np.clip(np.asarray(adjustable_out, dtype=float).reshape(-1), 0.0, upper)
    out = prior.copy()
    out[adjustable] = adjustable_out
    return out


def _block_conservative_element_partitions(
    operator,
    *,
    elements: np.ndarray,
    element_importance: np.ndarray,
    target_count: int,
) -> list[dict[str, Any]]:
    element_values = np.asarray(elements, dtype=int).reshape(-1)
    n_elements = int(element_values.size)
    if n_elements == 0:
        return []
    element_to_local = {int(element_id): idx for idx, element_id in enumerate(element_values)}

    interface_global = _fluid_interface_reaction_element_ids(
        operator.prob,
        interface_tag=operator.boundary_tags.interface_tag,
    )
    outlet_global = _boundary_element_ids(
        operator,
        tag=operator.boundary_tags.outlet_tag,
        fields=("ux", "uy", "p"),
    )
    fixed_global = np.unique(
        np.concatenate(
            [
                _boundary_element_ids(operator, tag=operator.boundary_tags.inlet_tag, fields=("ux", "uy")),
                _boundary_element_ids(operator, tag=operator.boundary_tags.walls_tag, fields=("ux", "uy")),
                _boundary_element_ids(operator, tag=operator.boundary_tags.cylinder_tag, fields=("ux", "uy")),
            ]
        )
    )

    labels = np.full(n_elements, "interior", dtype=object)

    def mark(global_ids: np.ndarray, label: str) -> None:
        local_ids = [element_to_local[int(eid)] for eid in np.asarray(global_ids, dtype=int).reshape(-1) if int(eid) in element_to_local]
        if local_ids:
            labels[np.asarray(local_ids, dtype=int)] = str(label)

    mark(fixed_global, "fixed_boundary")
    mark(outlet_global, "outlet_gauge")
    mark(interface_global, "interface")

    importance = np.asarray(element_importance, dtype=float).reshape(-1)
    if importance.size != n_elements:
        importance = np.ones(n_elements, dtype=float)
    high_pool = np.flatnonzero(labels == "interior")
    if high_pool.size:
        high_count = min(
            int(high_pool.size),
            max(32, min(max(int(target_count) // 5, 0), int(math.ceil(0.15 * n_elements)))),
        )
        if high_count > 0:
            high_local = high_pool[np.argsort(importance[high_pool])[::-1][:high_count]]
            labels[high_local] = "high_contribution"

    partitions: list[dict[str, Any]] = []
    for label, mandatory in (
        ("interface", True),
        ("outlet_gauge", True),
        ("fixed_boundary", True),
        ("high_contribution", False),
        ("interior", False),
    ):
        local = np.flatnonzero(labels == label)
        if local.size:
            partitions.append(
                {
                    "name": str(label),
                    "local_indices": local.astype(int, copy=False),
                    "mandatory": bool(mandatory),
                    "importance_sum": float(np.sum(np.maximum(importance[local], 0.0))),
                }
            )
    return partitions


def _fit_block_conservative_weights(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    operator,
    elements: np.ndarray,
    element_importance: np.ndarray,
    target_count: int,
    max_weight: float,
    prune_tol: float,
    regularization: float,
    weight_sum_scale: float,
    min_block_elements: int,
    fit_solver: str,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    A = np.asarray(matrix, dtype=float)
    b = np.asarray(target, dtype=float).reshape(-1)
    element_values = np.asarray(elements, dtype=int).reshape(-1)
    if A.ndim != 2 or int(A.shape[0]) != int(b.size):
        raise ValueError("Block-conservative cubature matrix/target shapes are incompatible.")
    if int(A.shape[1]) != int(element_values.size):
        raise ValueError("Block-conservative cubature element count is incompatible.")

    n_elements = int(element_values.size)
    requested_count = int(target_count)
    if requested_count <= 0:
        requested_count = min(n_elements, 1000)
    requested_count = max(1, min(requested_count, n_elements))
    requested_count_initial = int(requested_count)
    partitions = _block_conservative_element_partitions(
        operator,
        elements=element_values,
        element_importance=np.asarray(element_importance, dtype=float),
        target_count=requested_count,
    )
    weights = np.zeros(n_elements, dtype=float)
    mandatory_total = sum(int(np.asarray(part["local_indices"], dtype=int).size) for part in partitions if bool(part["mandatory"]))
    if mandatory_total >= requested_count:
        requested_count = min(
            n_elements,
            mandatory_total + max(4 * int(min_block_elements), 256),
        )
    optional_parts = [part for part in partitions if not bool(part["mandatory"])]
    optional_budget = max(requested_count - mandatory_total, 0)
    optional_importance = sum(max(float(part.get("importance_sum", 0.0)), 0.0) for part in optional_parts)
    if optional_importance <= 0.0:
        optional_importance = float(sum(int(np.asarray(part["local_indices"], dtype=int).size) for part in optional_parts))
    optional_size = float(sum(int(np.asarray(part["local_indices"], dtype=int).size) for part in optional_parts))

    block_infos: list[dict[str, Any]] = []
    selected_groups: list[dict[str, Any]] = []
    for part in partitions:
        local = np.asarray(part["local_indices"], dtype=int).reshape(-1)
        block_A = A[:, local]
        block_target = block_A @ np.ones(int(local.size), dtype=float)
        if bool(part["mandatory"]):
            selected_local = local
        else:
            if optional_budget <= 0:
                block_count = min(int(local.size), max(1, int(min_block_elements)))
            else:
                importance_share = max(float(part.get("importance_sum", 0.0)), 0.0) / max(optional_importance, 1.0e-15)
                size_share = float(local.size) / max(optional_size, 1.0)
                mixed_share = 0.5 * importance_share + 0.5 * size_share
                block_count = int(round(optional_budget * mixed_share))
                block_count = max(int(min_block_elements), block_count)
                block_count = min(int(local.size), block_count)
            selected_offsets = _select_diverse_columns(block_A, count=block_count)
            selected_local = local[selected_offsets]

        if int(selected_local.size) == int(local.size):
            block_weights = np.ones(int(selected_local.size), dtype=float)
            block_error = 0.0
        else:
            prior_value = max(float(local.size) / max(float(selected_local.size), 1.0), 1.0)
            prior = np.full(int(selected_local.size), min(prior_value, max(float(max_weight), 1.0)), dtype=float)
            block_weights, block_error = _fit_bounded_regularized_weights(
                A[:, selected_local],
                block_target,
                max_weight=float(max_weight),
                prior=prior,
                regularization=float(regularization),
                weight_sum_target=float(local.size),
                weight_sum_scale=float(weight_sum_scale),
                solver=str(fit_solver),
            )
        block_weights = np.where(block_weights > max(float(prune_tol), 0.0), block_weights, 0.0)
        weights[selected_local] = block_weights
        selected_groups.append(
            {
                "name": str(part["name"]),
                "all_local": local.copy(),
                "selected_local": np.asarray(selected_local, dtype=int).reshape(-1).copy(),
                "weight_sum_target": float(local.size),
                "initial_fit_relative_error": float(block_error),
                "mandatory": bool(part["mandatory"]),
            }
        )
        block_infos.append(
            {
                "name": str(part["name"]),
                "candidate_elements": int(local.size),
                "selected_elements": int(np.count_nonzero(block_weights > 0.0)),
                "mandatory": bool(part["mandatory"]),
                "fit_relative_error": float(block_error),
                "weight_min": float(np.min(block_weights)) if block_weights.size else float("nan"),
                "weight_max": float(np.max(block_weights)) if block_weights.size else float("nan"),
                "weight_sum": float(np.sum(block_weights)),
            }
        )

    initial_weights = weights.copy()
    initial_global_fit_error = float(np.linalg.norm(A @ initial_weights - b) / max(float(np.linalg.norm(b)), 1.0e-15))
    refit_global_fit_error = float("nan")
    refit_accepted = False
    selected = np.flatnonzero(weights > max(float(prune_tol), 0.0)).astype(int, copy=False)
    if selected.size:
        refit_weights = _refit_selected_block_weights(
            A,
            b,
            selected=selected,
            initial_weights=weights[selected],
            block_groups=selected_groups,
            max_weight=float(max_weight),
            regularization=float(regularization),
            weight_sum_scale=float(weight_sum_scale),
            solver=str(fit_solver),
        )
        weights[:] = 0.0
        weights[selected] = np.where(refit_weights > max(float(prune_tol), 0.0), refit_weights, 0.0)
        refit_global_fit_error = float(np.linalg.norm(A @ weights - b) / max(float(np.linalg.norm(b)), 1.0e-15))
        if refit_global_fit_error <= initial_global_fit_error:
            refit_accepted = True
        else:
            weights[:] = initial_weights
        for info_item, group in zip(block_infos, selected_groups, strict=False):
            local = np.asarray(group["all_local"], dtype=int).reshape(-1)
            block_target = A[:, local] @ np.ones(int(local.size), dtype=float)
            block_residual = A[:, local] @ weights[local] - block_target
            info_item["post_refit_relative_error"] = float(
                np.linalg.norm(block_residual) / max(float(np.linalg.norm(block_target)), 1.0e-15)
            )
            info_item["post_refit_selected_elements"] = int(np.count_nonzero(weights[local] > max(float(prune_tol), 0.0)))
            active_weights = weights[local][weights[local] > max(float(prune_tol), 0.0)]
            info_item["post_refit_weight_min"] = float(np.min(active_weights)) if active_weights.size else float("nan")
            info_item["post_refit_weight_max"] = float(np.max(active_weights)) if active_weights.size else float("nan")
            info_item["post_refit_weight_sum"] = float(np.sum(weights[local]))

    fit_error = float(np.linalg.norm(A @ weights - b) / max(float(np.linalg.norm(b)), 1.0e-15))
    info = {
        "requested_target_count": int(requested_count_initial),
        "effective_target_count": int(requested_count),
        "mandatory_elements": int(mandatory_total),
        "selected_elements": int(np.count_nonzero(weights > max(float(prune_tol), 0.0))),
        "regularization": float(regularization),
        "weight_sum_scale": float(weight_sum_scale),
        "fit_solver": str(fit_solver),
        "min_block_elements": int(min_block_elements),
        "initial_global_fit_relative_error": float(initial_global_fit_error),
        "refit_global_fit_relative_error": float(refit_global_fit_error),
        "refit_accepted": bool(refit_accepted),
        "blocks": block_infos,
    }
    return weights, fit_error, info


def _compress_cubature_fit_rows(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    A = np.asarray(matrix, dtype=float)
    b = np.asarray(target, dtype=float).reshape(-1)
    if int(max_rows) <= 0 or int(A.shape[0]) <= int(max_rows):
        return A, b, {"enabled": False, "original_rows": int(A.shape[0]), "kept_rows": int(A.shape[0])}
    n_rows = int(A.shape[0])
    keep = max(1, min(int(max_rows), n_rows))
    row_norm = np.sqrt(np.sum(A * A, axis=1) + b * b)
    n_top = min(keep // 2, n_rows)
    top = np.argpartition(row_norm, -n_top)[-n_top:] if n_top > 0 else np.zeros(0, dtype=int)
    remaining_mask = np.ones(n_rows, dtype=bool)
    remaining_mask[top] = False
    remaining = np.flatnonzero(remaining_mask)
    n_even = keep - int(top.size)
    if n_even > 0 and remaining.size:
        if remaining.size <= n_even:
            even = remaining
        else:
            even = remaining[np.linspace(0, int(remaining.size) - 1, int(n_even), dtype=int)]
    else:
        even = np.zeros(0, dtype=int)
    selected = np.unique(np.concatenate([top, even]).astype(int, copy=False))
    if selected.size < keep:
        fill = np.argsort(row_norm)[::-1]
        missing = [int(idx) for idx in fill if int(idx) not in set(int(v) for v in selected)]
        if missing:
            selected = np.concatenate([selected, np.asarray(missing[: keep - int(selected.size)], dtype=int)])
    selected = np.sort(selected[:keep].astype(int, copy=False))
    return (
        A[selected, :],
        b[selected],
        {
            "enabled": True,
            "original_rows": int(n_rows),
            "kept_rows": int(selected.size),
            "top_norm_rows": int(top.size),
            "uniform_rows": int(even.size),
            "max_row_norm": float(np.max(row_norm)) if row_norm.size else float("nan"),
            "min_selected_row_norm": float(np.min(row_norm[selected])) if selected.size else float("nan"),
        },
    )


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
    element_weighting: str,
    element_candidate_mode: str = "sample-mesh",
    element_target_count: int = 0,
    block_regularization: float = 1.0e-8,
    block_sum_scale: float = 1.0e-4,
    block_min_elements: int = 32,
    fit_max_rows: int = 0,
    fit_solver: str = "bounded",
    pair_weights: dict[tuple[int, int], float] | None = None,
    trajectory_source: str = "lspg",
    training_state_selection: str = "first",
    include_intermediate_states: bool = False,
    incompressibility_stabilization_scale: float = 1.0,
) -> tuple[FluidGNATSampleSet, dict[str, Any]]:
    rows = np.asarray(sample_set.row_dofs, dtype=int).reshape(-1)
    weighting_key = str(element_weighting).strip().lower().replace("_", "-")
    reduced_operator_fit = weighting_key in {
        "reduced-operator-lsq-bounded",
        "reduced-operator-greedy",
        "galerkin-lsq-bounded",
        "galerkin-greedy",
        "operator-lsq-bounded",
        "operator-greedy",
        "reduced-operator-topnorm",
        "reduced-operator-block-conservative",
    }
    block_conservative_fit = weighting_key in {
        "reduced-operator-block-conservative",
        "operator-block-conservative",
        "galerkin-block-conservative",
    }
    candidate_key = str(element_candidate_mode).strip().lower().replace("_", "-")
    if block_conservative_fit and candidate_key != "all":
        raise ValueError("reduced-operator-block-conservative requires --gnat-element-candidate-elements all.")
    if reduced_operator_fit and candidate_key == "all":
        elements = np.arange(int(operator.mesh.n_elements), dtype=int)
    else:
        elements = np.asarray(sample_set.element_ids, dtype=int).reshape(-1)
    if rows.size == 0 or elements.size == 0:
        return sample_set, {"enabled": False, "reason": "empty rows or elements"}

    blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    entries: list[dict[str, Any]] = []
    states_used = 0
    p = operator.parameters
    element_importance = (
        np.zeros(int(elements.size), dtype=float)
        if ("topnorm" in weighting_key or "block-conservative" in weighting_key)
        else None
    )
    state_selection_key = str(training_state_selection).strip().lower().replace("_", "-")
    if state_selection_key not in {"first", "stratified"}:
        raise ValueError(f"Unsupported element-cubature training state selection {training_state_selection!r}.")
    pair_list = list(pairs)
    phase_t0 = time.perf_counter()
    _progress(
        "fit element cubature: "
        f"pairs={len(pair_list)} max_states={int(max_training_states)} "
        f"candidate_elements={int(elements.size)} weighting={weighting_key}"
    )

    for pair_index, pair in enumerate(pair_list):
        if state_selection_key == "first" and states_used >= int(max_training_states):
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

        projected = trial_basis.project_state(target_state, offset=initial_state)
        exact_state_coefficients: list[np.ndarray] = []
        if bool(include_intermediate_states):
            for newton_path in _exact_newton_iterate_probe_paths(pair):
                newton_probe = load_fluid_stage_probe(newton_path)
                _configure_and_restore_probe(
                    operator=operator,
                    probe=newton_probe,
                    setup=setup,
                    fluid_iface_coords=fluid_iface_coords,
                    dt=float(dt),
                    reference_velocity=float(reference_velocity),
                )
                newton_state = pack_fluid_state(operator)
                exact_state_coefficients.append(
                    np.asarray(trial_basis.project_state(newton_state, offset=initial_state), dtype=float).reshape(-1)
                )
            _configure_and_restore_probe(
                operator=operator,
                probe=pre_probe,
                setup=setup,
                fluid_iface_coords=fluid_iface_coords,
                dt=float(dt),
                reference_velocity=float(reference_velocity),
            )
        trajectory_mode = str(trajectory_source).strip().lower()
        if trajectory_mode in {"galerkin", "sampled_galerkin", "reduced-galerkin", "reduced_galerkin"}:
            coefficients, lspg_iterations, lspg_converged, _galerkin_residual_norm = (
                _solve_full_mesh_galerkin_training_path(
                    operator=operator,
                    trial_space=trial_space,
                    fluid=fluid,
                    bossak=bossak,
                    initial_coefficients=initial_coefficients,
                    max_iterations=int(max_iterations),
                    residual_tol=float(residual_tol),
                    line_search=bool(line_search),
                    incompressibility_stabilization_scale=float(incompressibility_stabilization_scale),
                )
            )
        elif trajectory_mode != "projection":
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
            lspg_iterations = int(result.iterations)
            lspg_converged = bool(result.converged)
        else:
            lspg_iterations = 0
            lspg_converged = True
        if exact_state_coefficients:
            coefficients = _append_unique_coefficient_states(coefficients, exact_state_coefficients)
        if not coefficients or float(np.linalg.norm(projected - coefficients[-1])) > 1.0e-14:
            coefficients.append(np.asarray(projected, dtype=float).reshape(-1))

        selected_coefficients = list(coefficients)
        if state_selection_key == "stratified":
            budget_remaining = int(max_training_states) - int(states_used)
            pairs_remaining = max(int(len(pair_list) - pair_index), 1)
            if budget_remaining <= 0:
                selected_coefficients = []
            else:
                quota = int(math.ceil(float(budget_remaining) / float(pairs_remaining)))
                quota = min(max(quota, 1), int(len(coefficients)), int(budget_remaining))
                if quota == 1:
                    selected_indices = [int(len(coefficients) - 1)]
                else:
                    selected_indices = []
                    for raw in np.linspace(0, int(len(coefficients) - 1), int(quota)):
                        idx = int(round(float(raw)))
                        if idx not in selected_indices:
                            selected_indices.append(idx)
                    candidate_idx = int(len(coefficients) - 1)
                    while len(selected_indices) < quota and candidate_idx >= 0:
                        if candidate_idx not in selected_indices:
                            selected_indices.append(candidate_idx)
                        candidate_idx -= 1
                    selected_indices = sorted(selected_indices[:quota])
                selected_coefficients = [coefficients[int(idx)] for idx in selected_indices]
        for coeffs in selected_coefficients:
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
            if reduced_operator_fit:
                all_elements = np.arange(int(operator.mesh.n_elements), dtype=int)
                _all_ids, full_residual_by_element, full_tangent_by_element = (
                    _assemble_fluid_sampled_galerkin_element_contributions_raw(
                        prob=operator.prob,
                        rho_f=float(p.rho_f),
                        mu_f=float(p.mu_f),
                        dt=float(p.dt),
                        quad_order=int(p.quadrature_order),
                        bossak_alpha=float(p.bossak_alpha),
                        contribution_mode=str(p.contribution_mode),
                        backend=str(p.backend),
                        element_ids=all_elements,
                        basis=np.asarray(trial_basis.basis, dtype=float),
                        incompressibility_stabilization_scale=float(incompressibility_stabilization_scale),
                    )
                )
                if np.array_equal(elements, all_elements):
                    residual_by_element = full_residual_by_element
                    tangent_by_element = full_tangent_by_element
                else:
                    _ids, residual_by_element, tangent_by_element = (
                        _assemble_fluid_sampled_galerkin_element_contributions_raw(
                            prob=operator.prob,
                            rho_f=float(p.rho_f),
                            mu_f=float(p.mu_f),
                            dt=float(p.dt),
                            quad_order=int(p.quadrature_order),
                            bossak_alpha=float(p.bossak_alpha),
                            contribution_mode=str(p.contribution_mode),
                            backend=str(p.backend),
                            element_ids=elements,
                            basis=np.asarray(trial_basis.basis, dtype=float),
                            incompressibility_stabilization_scale=float(incompressibility_stabilization_scale),
                        )
                    )
                target_residual = np.sum(full_residual_by_element, axis=0).reshape(-1)
                target_tangent = np.sum(full_tangent_by_element, axis=0)
                weight = max(_pair_weight(pair, pair_weights), 0.0)
                block_scale_value = math.sqrt(weight)
                if block_conservative_fit:
                    _append_reduced_operator_fit_blocks(
                        blocks=blocks,
                        targets=targets,
                        element_importance=element_importance,
                        residual_by_element=residual_by_element,
                        tangent_by_element=tangent_by_element,
                        trial_basis=trial_basis,
                        pair_weight=weight,
                        split_velocity_pressure=True,
                    )
                    interface_rows = _fluid_interface_velocity_dofs(
                        operator.prob,
                        interface_tag=operator.boundary_tags.interface_tag,
                    )
                    if interface_rows.size:
                        _iface_ids, iface_residual_by_element, _iface_trial_by_element = (
                            _assemble_fluid_sampled_lspg_element_contributions_raw(
                                prob=operator.prob,
                                rho_f=float(p.rho_f),
                                mu_f=float(p.mu_f),
                                dt=float(p.dt),
                                quad_order=int(p.quadrature_order),
                                bossak_alpha=float(p.bossak_alpha),
                                contribution_mode=str(p.contribution_mode),
                                backend=str(p.backend),
                                element_ids=elements,
                                row_dofs=interface_rows,
                                basis=np.asarray(trial_basis.basis, dtype=float),
                                incompressibility_stabilization_scale=float(incompressibility_stabilization_scale),
                            )
                        )
                        _append_scaled_fit_block(
                            blocks=blocks,
                            targets=targets,
                            element_importance=element_importance,
                            contribution_by_element=iface_residual_by_element,
                            target=np.sum(iface_residual_by_element, axis=0),
                            scale=block_scale_value,
                        )
                    try:
                        _outlet_coords, outlet_p_rows = _boundary_field_data(
                            operator.dh,
                            "p",
                            operator.boundary_tags.outlet_tag,
                        )
                    except Exception:
                        outlet_p_rows = np.zeros(0, dtype=int)
                    if np.asarray(outlet_p_rows, dtype=int).size:
                        _out_ids, outlet_residual_by_element, _out_trial_by_element = (
                            _assemble_fluid_sampled_lspg_element_contributions_raw(
                                prob=operator.prob,
                                rho_f=float(p.rho_f),
                                mu_f=float(p.mu_f),
                                dt=float(p.dt),
                                quad_order=int(p.quadrature_order),
                                bossak_alpha=float(p.bossak_alpha),
                                contribution_mode=str(p.contribution_mode),
                                backend=str(p.backend),
                                element_ids=elements,
                                row_dofs=np.asarray(outlet_p_rows, dtype=int).reshape(-1),
                                basis=np.asarray(trial_basis.basis, dtype=float),
                                incompressibility_stabilization_scale=float(incompressibility_stabilization_scale),
                            )
                        )
                        _append_scaled_fit_block(
                            blocks=blocks,
                            targets=targets,
                            element_importance=element_importance,
                            contribution_by_element=outlet_residual_by_element,
                            target=np.sum(outlet_residual_by_element, axis=0),
                            scale=block_scale_value,
                        )
                else:
                    residual_scale = block_scale_value / max(float(np.linalg.norm(target_residual)), 1.0e-12)
                    tangent_scale = block_scale_value / max(float(np.linalg.norm(target_tangent)), 1.0e-12)
                    if element_importance is not None:
                        element_importance += np.sum(
                            np.asarray(residual_by_element * residual_scale, dtype=float) ** 2,
                            axis=1,
                        )
                        element_importance += np.sum(
                            np.asarray(tangent_by_element * tangent_scale, dtype=float) ** 2,
                            axis=(1, 2),
                        )
                    blocks.append(np.asarray(residual_by_element.T, dtype=float) * residual_scale)
                    targets.append(target_residual * residual_scale)
                    blocks.append(
                        np.asarray(tangent_by_element.reshape(int(tangent_by_element.shape[0]), -1).T, dtype=float)
                        * tangent_scale
                    )
                    targets.append(target_tangent.reshape(-1) * tangent_scale)
                states_used += 1
                continue
            elif abs(float(incompressibility_stabilization_scale) - 1.0) > 1.0e-14:
                all_elements = np.arange(int(operator.mesh.n_elements), dtype=int)
                target, _target_trial = _assemble_fluid_sampled_lspg_rows_raw(
                    prob=operator.prob,
                    rho_f=float(p.rho_f),
                    mu_f=float(p.mu_f),
                    dt=float(p.dt),
                    quad_order=int(p.quadrature_order),
                    bossak_alpha=float(p.bossak_alpha),
                    contribution_mode=str(p.contribution_mode),
                    backend=str(p.backend),
                    element_ids=all_elements,
                    row_dofs=rows,
                    basis=np.asarray(trial_basis.basis, dtype=float),
                    incompressibility_stabilization_scale=float(incompressibility_stabilization_scale),
                )
            else:
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
                incompressibility_stabilization_scale=float(incompressibility_stabilization_scale),
            )
            weight = max(_pair_weight(pair, pair_weights), 0.0)
            block_scale_value = math.sqrt(weight) / max(float(np.linalg.norm(target)), 1.0e-12)
            blocks.append(np.asarray(residual_by_element.T, dtype=float) * block_scale_value)
            targets.append(target * block_scale_value)
            states_used += 1

        entries.append(
            {
                "step": int(pair.step),
                "coupling_iter": int(pair.coupling_iter),
                "training_weight": float(max(_pair_weight(pair, pair_weights), 0.0)),
                "lspg_iterations": int(lspg_iterations),
                "lspg_converged": bool(lspg_converged),
                "trajectory_source": str(trajectory_mode),
                "exact_intermediate_states_available": int(len(exact_state_coefficients)),
                "selected_states_for_pair": int(len(selected_coefficients)),
                "states_after_pair": int(states_used),
            }
        )
        if pair_index == 0 or pair_index + 1 == len(pair_list) or (pair_index + 1) % 5 == 0:
            _progress(
                "fit element cubature: "
                f"{pair_index + 1}/{len(pair_list)} "
                f"step={int(pair.step)} iter={int(pair.coupling_iter)} "
                f"states_used={int(states_used)} "
                f"blocks={len(blocks)} "
                f"elapsed={time.perf_counter() - phase_t0:.2f}s"
            )

    if not blocks:
        return sample_set, {"enabled": False, "reason": "no training states"}

    _progress(
        "fit element cubature: assembling least-squares matrix "
        f"blocks={len(blocks)} targets={len(targets)}"
    )
    fit_matrix_full = np.vstack(blocks)
    fit_target_full = np.concatenate(targets)
    unit_weight_fit_error = float(
        np.linalg.norm(fit_matrix_full @ np.ones(int(fit_matrix_full.shape[1]), dtype=float) - fit_target_full)
        / max(float(np.linalg.norm(fit_target_full)), 1.0e-15)
    )
    fit_matrix, fit_target, fit_row_compression = _compress_cubature_fit_rows(
        fit_matrix_full,
        fit_target_full,
        max_rows=int(fit_max_rows),
    )
    _progress(
        "fit element cubature: solving weights "
        f"rows={int(fit_matrix.shape[0])} cols={int(fit_matrix.shape[1])} "
        f"solver={fit_solver}"
    )
    block_conservative_info: dict[str, Any] | None = None
    if block_conservative_fit:
        weights, fit_error, block_conservative_info = _fit_block_conservative_weights(
            fit_matrix,
            fit_target,
            operator=operator,
            elements=elements,
            element_importance=np.asarray(element_importance, dtype=float)
            if element_importance is not None
            else np.ones(int(elements.size), dtype=float),
            target_count=int(element_target_count),
            max_weight=float(max_weight),
            prune_tol=float(prune_tol),
            regularization=float(block_regularization),
            weight_sum_scale=float(block_sum_scale),
            min_block_elements=int(block_min_elements),
            fit_solver=str(fit_solver),
        )
    elif "topnorm" in weighting_key:
        if element_importance is None:
            raise RuntimeError("Top-norm element selection requires reduced-operator training data.")
        target_count = int(element_target_count)
        if target_count <= 0:
            target_count = min(int(elements.size), 1000)
        target_count = max(1, min(target_count, int(elements.size)))
        selected = np.argsort(np.asarray(element_importance, dtype=float))[::-1][:target_count]
        weights = np.zeros(int(elements.size), dtype=float)
        weights[np.asarray(selected, dtype=int)] = 1.0
        fit_error = float(
            np.linalg.norm(fit_matrix @ weights - fit_target)
            / max(float(np.linalg.norm(fit_target)), 1.0e-15)
        )
    elif "greedy" in weighting_key:
        weights, fit_error = _fit_greedy_nonnegative_weights(
            fit_matrix,
            fit_target,
            max_weight=float(max_weight),
            relative_tol=max(float(prune_tol), 1.0e-10),
        )
    else:
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
    if reduced_operator_fit:
        row_keep = np.ones(int(rows.size), dtype=bool)
    else:
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
        element_weighting=f"{weighting_key}-pruned" if np.any(~keep) else weighting_key,
        element_weight_fit_relative_error=float(fit_error),
    )
    info = {
        "enabled": True,
        "training_states": int(states_used),
        "fit_rows": int(fit_matrix.shape[0]),
        "uncompressed_fit_rows": int(fit_matrix_full.shape[0]),
        "fit_row_compression": fit_row_compression,
        "fit_solver": str(fit_solver),
        "candidate_elements": int(elements.size),
        "kept_elements": int(cubature_set.n_sample_elements),
        "candidate_rows": int(rows.size),
        "kept_rows": int(cubature_set.n_sample_rows),
        "fit_relative_error": float(fit_error),
        "unit_weight_fit_relative_error": float(unit_weight_fit_error),
        "weight_min": float(np.min(cubature_set.element_weights)) if cubature_set.element_weights.size else float("nan"),
        "weight_max": float(np.max(cubature_set.element_weights)) if cubature_set.element_weights.size else float("nan"),
        "weight_sum": float(np.sum(cubature_set.element_weights)),
        "prune_tol": float(prune_tol),
        "max_weight": float(max_weight),
        "keep_interface_elements": bool(keep_interface_elements),
        "incompressibility_stabilization_scale": float(incompressibility_stabilization_scale),
        "candidate_mode": str(candidate_key),
        "target_count": int(element_target_count),
        "block_conservative": block_conservative_info,
        "entries": entries,
    }
    _progress(
        "fit element cubature: done "
        f"kept_elements={int(cubature_set.n_sample_elements)} "
        f"kept_rows={int(cubature_set.n_sample_rows)} "
        f"fit_error={float(fit_error):.3e} "
        f"elapsed={time.perf_counter() - phase_t0:.2f}s"
    )
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
    training_coefficients: np.ndarray | None = None,
    reaction_operator: dict[str, Any] | None = None,
    sampled_reaction_operator: dict[str, Any] | None = None,
    impedance_operator: dict[str, Any] | None = None,
) -> None:
    """Persist the deployable arrays needed by `run_example2_local.py`."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    training_coeffs = (
        np.zeros((int(trial_basis.n_modes), 0), dtype=float)
        if training_coefficients is None
        else np.asarray(training_coefficients, dtype=float)
    )
    if training_coeffs.ndim != 2 or int(training_coeffs.shape[0]) != int(trial_basis.n_modes):
        raise ValueError("training_coefficients must have shape (n_modes, n_training_states).")
    if training_coeffs.shape[1]:
        coeff_mean = np.mean(training_coeffs, axis=1)
        coeff_scale = np.maximum(np.std(training_coeffs, axis=1), 1.0e-12)
        scaled_distances = np.linalg.norm((training_coeffs - coeff_mean[:, None]) / coeff_scale[:, None], axis=0)
        coeff_radius = float(np.max(scaled_distances))
    else:
        coeff_mean = np.zeros(int(trial_basis.n_modes), dtype=float)
        coeff_scale = np.ones(int(trial_basis.n_modes), dtype=float)
        coeff_radius = float("inf")
    payload: dict[str, np.ndarray] = dict(
        schema_version=np.asarray(1, dtype=int),
        basis=np.asarray(trial_basis.basis, dtype=float),
        free_dofs=np.asarray(trial_basis.free_dofs, dtype=int),
        velocity_dofs=np.asarray(trial_basis.velocity_dofs, dtype=int),
        pressure_dofs=np.asarray(trial_basis.pressure_dofs, dtype=int),
        velocity_singular_values=np.asarray(trial_basis.velocity_singular_values, dtype=float),
        pressure_singular_values=np.asarray(trial_basis.pressure_singular_values, dtype=float),
        velocity_energy_fraction=np.asarray(trial_basis.velocity_energy_fraction, dtype=float),
        pressure_energy_fraction=np.asarray(trial_basis.pressure_energy_fraction, dtype=float),
        velocity_modes=np.asarray(int(trial_basis.n_velocity_modes), dtype=int),
        pressure_modes=np.asarray(int(trial_basis.n_pressure_modes), dtype=int),
        requested_velocity_modes=np.asarray(int(args.velocity_modes), dtype=int),
        requested_pressure_modes=np.asarray(int(args.pressure_modes), dtype=int),
        velocity_pod_modes=np.asarray(int(trial_basis.n_velocity_pod_modes), dtype=int),
        velocity_supremizer_modes=np.asarray(int(trial_basis.n_velocity_supremizer_modes), dtype=int),
        supremizer_singular_values=np.asarray(trial_basis.supremizer_singular_values, dtype=float),
        supremizer_riesz=np.asarray(str(trial_basis.supremizer_riesz)),
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
        rom_incompressibility_scale=np.asarray(float(args.rom_incompressibility_scale), dtype=float),
        max_iterations=np.asarray(int(args.gnat_max_iterations), dtype=int),
        residual_tol=np.asarray(float(args.gnat_residual_tol), dtype=float),
        line_search=np.asarray(bool(args.gnat_line_search), dtype=bool),
        recommended_switch_iter=np.asarray(4, dtype=int),
        training_sources=np.asarray([str(item) for item in training_source]),
        training_steps=np.asarray(training_steps, dtype=int),
        training_all_iters=np.asarray(bool(args.training_all_iters), dtype=bool),
        training_weight_mode=np.asarray(str(args.training_weight_mode)),
        training_weight_late_start_step=np.asarray(
            -1 if args.training_weight_late_start_step is None else int(args.training_weight_late_start_step),
            dtype=int,
        ),
        training_weight_late_factor=np.asarray(float(args.training_weight_late_factor), dtype=float),
        training_weight_final_factor=np.asarray(float(args.training_weight_final_factor), dtype=float),
        training_weight_iteration_factor=np.asarray(float(args.training_weight_iteration_factor), dtype=float),
        training_all_states=np.asarray(bool(args.training_all_states), dtype=bool),
        training_coefficient_mean=np.asarray(coeff_mean, dtype=float),
        training_coefficient_scale=np.asarray(coeff_scale, dtype=float),
        training_coefficient_radius=np.asarray(float(coeff_radius), dtype=float),
        training_coefficient_count=np.asarray(int(training_coeffs.shape[1]), dtype=int),
    )
    if reaction_operator is not None:
        payload.update(
            {
                "reaction_matrix": np.asarray(reaction_operator["matrix"], dtype=float),
                "reaction_bias": np.asarray(reaction_operator["bias"], dtype=float).reshape(-1),
                "reaction_coords": np.asarray(reaction_operator["coords"], dtype=float),
                "reaction_kind": np.asarray(str(reaction_operator.get("kind", "point"))),
                "reaction_fit_ridge": np.asarray(float(reaction_operator.get("ridge", 0.0)), dtype=float),
                "reaction_fit_count": np.asarray(int(reaction_operator.get("fit_count", 0)), dtype=int),
                "reaction_fit_train_count": np.asarray(int(reaction_operator.get("train_count", 0)), dtype=int),
                "reaction_fit_validation_count": np.asarray(
                    int(reaction_operator.get("validation_count", 0)),
                    dtype=int,
                ),
                "reaction_fit_mean_relative_error": np.asarray(
                    float(reaction_operator.get("fit_mean_relative_error", float("nan"))),
                    dtype=float,
                ),
                "reaction_fit_max_relative_error": np.asarray(
                    float(reaction_operator.get("fit_max_relative_error", float("nan"))),
                    dtype=float,
                ),
                "reaction_validation_mean_relative_error": np.asarray(
                    float(reaction_operator.get("validation_mean_relative_error", float("nan"))),
                    dtype=float,
                ),
                "reaction_validation_max_relative_error": np.asarray(
                    float(reaction_operator.get("validation_max_relative_error", float("nan"))),
                    dtype=float,
                ),
            }
        )
    if sampled_reaction_operator is not None:
        payload.update(
            {
                "reaction_nonlinear_kind": np.asarray(
                    str(sampled_reaction_operator.get("kind", "gappy_pod_point"))
                ),
                "reaction_coords": np.asarray(sampled_reaction_operator["coords"], dtype=float),
                "reaction_basis": np.asarray(sampled_reaction_operator["basis"], dtype=float),
                "reaction_mean": np.asarray(sampled_reaction_operator["mean"], dtype=float).reshape(-1),
                "reaction_sample_row_dofs": np.asarray(
                    sampled_reaction_operator["sample_row_dofs"],
                    dtype=int,
                ),
                "reaction_sample_element_ids": np.asarray(
                    sampled_reaction_operator["sample_element_ids"],
                    dtype=int,
                ),
                "reaction_sample_to_coefficients": np.asarray(
                    sampled_reaction_operator["sample_to_coefficients"],
                    dtype=float,
                ),
                "reaction_sample_output_positions": np.asarray(
                    sampled_reaction_operator["sample_output_positions"],
                    dtype=int,
                ),
                "reaction_nonlinear_modes": np.asarray(
                    int(sampled_reaction_operator.get("modes", 0)),
                    dtype=int,
                ),
                "reaction_nonlinear_sample_count": np.asarray(
                    int(sampled_reaction_operator.get("sample_count", 0)),
                    dtype=int,
                ),
                "reaction_nonlinear_sample_element_count": np.asarray(
                    int(sampled_reaction_operator.get("sample_element_count", 0)),
                    dtype=int,
                ),
                "reaction_nonlinear_sample_basis_condition": np.asarray(
                    float(sampled_reaction_operator.get("sample_basis_condition", float("nan"))),
                    dtype=float,
                ),
                "reaction_nonlinear_singular_values": np.asarray(
                    sampled_reaction_operator.get("singular_values", np.zeros(0, dtype=float)),
                    dtype=float,
                ),
                "reaction_nonlinear_energy_fraction": np.asarray(
                    sampled_reaction_operator.get("energy_fraction", np.zeros(0, dtype=float)),
                    dtype=float,
                ),
                "reaction_nonlinear_fit_count": np.asarray(
                    int(sampled_reaction_operator.get("fit_count", 0)),
                    dtype=int,
                ),
                "reaction_nonlinear_validation_count": np.asarray(
                    int(sampled_reaction_operator.get("validation_count", 0)),
                    dtype=int,
                ),
                "reaction_nonlinear_fit_mean_relative_error": np.asarray(
                    float(sampled_reaction_operator.get("fit_mean_relative_error", float("nan"))),
                    dtype=float,
                ),
                "reaction_nonlinear_fit_max_relative_error": np.asarray(
                    float(sampled_reaction_operator.get("fit_max_relative_error", float("nan"))),
                    dtype=float,
                ),
                "reaction_nonlinear_validation_mean_relative_error": np.asarray(
                    float(sampled_reaction_operator.get("validation_mean_relative_error", float("nan"))),
                    dtype=float,
                ),
                "reaction_nonlinear_validation_max_relative_error": np.asarray(
                    float(sampled_reaction_operator.get("validation_max_relative_error", float("nan"))),
                    dtype=float,
                ),
            }
        )
    if impedance_operator is not None:
        payload.update(
            {
                "impedance_matrix": np.asarray(impedance_operator["matrix"], dtype=float),
                "impedance_bias": np.asarray(impedance_operator["bias"], dtype=float).reshape(-1),
                "impedance_coords": np.asarray(impedance_operator["coords"], dtype=float),
                "impedance_feature_basis": np.asarray(impedance_operator["feature_basis"], dtype=float),
                "impedance_feature_mean": np.asarray(impedance_operator["feature_mean"], dtype=float).reshape(-1),
                "impedance_velocity_scale": np.asarray(
                    float(impedance_operator.get("velocity_scale", 1.0)),
                    dtype=float,
                ),
                "impedance_kind": np.asarray(str(impedance_operator.get("kind", "secant_point"))),
                "impedance_fit_ridge": np.asarray(float(impedance_operator.get("ridge", 0.0)), dtype=float),
                "impedance_modes": np.asarray(int(impedance_operator.get("modes", 0)), dtype=int),
                "impedance_fit_count": np.asarray(int(impedance_operator.get("fit_count", 0)), dtype=int),
                "impedance_fit_train_count": np.asarray(int(impedance_operator.get("train_count", 0)), dtype=int),
                "impedance_fit_validation_count": np.asarray(
                    int(impedance_operator.get("validation_count", 0)),
                    dtype=int,
                ),
                "impedance_fit_mean_relative_error": np.asarray(
                    float(impedance_operator.get("fit_mean_relative_error", float("nan"))),
                    dtype=float,
                ),
                "impedance_fit_max_relative_error": np.asarray(
                    float(impedance_operator.get("fit_max_relative_error", float("nan"))),
                    dtype=float,
                ),
                "impedance_validation_mean_relative_error": np.asarray(
                    float(impedance_operator.get("validation_mean_relative_error", float("nan"))),
                    dtype=float,
                ),
                "impedance_validation_max_relative_error": np.asarray(
                    float(impedance_operator.get("validation_max_relative_error", float("nan"))),
                    dtype=float,
                ),
            }
        )
    np.savez_compressed(target, **payload)


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
        "--extra-training-probe-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional exact stage-probe roots used with --training-probe-dir. "
            "Pairs are de-duplicated by (step, coupling iteration)."
        ),
    )
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
    parser.add_argument(
        "--training-all-states",
        action="store_true",
        help=(
            "Use every available state probe inside each selected training coupling iteration. "
            "For increment bases this adds exact Newton iterate probes when they exist; for absolute bases "
            "this adds pre, Newton-iterate, and post states instead of post states only."
        ),
    )
    parser.add_argument(
        "--export-all-state-training",
        type=Path,
        default=None,
        help=(
            "Write a self-contained all-state training .npz with pre/post states, available Newton probes, "
            "LSPG fallback iterates, residual snapshots, J V snapshots, interface reactions, ALE fields, and DVMS state."
        ),
    )
    parser.add_argument(
        "--no-export-jacobian-actions",
        action="store_true",
        help="Do not store the dense J V tensor in --export-all-state-training.",
    )
    parser.add_argument(
        "--training-weight-mode",
        choices=("uniform", "coupled", "late-coupled"),
        default="uniform",
        help=(
            "Scale solution/residual/cubature training snapshots. 'late-coupled' boosts final coupling "
            "iterations and steps at/after --training-weight-late-start-step."
        ),
    )
    parser.add_argument(
        "--training-weight-late-start-step",
        type=int,
        default=None,
        help="First step receiving the late-window training boost.",
    )
    parser.add_argument(
        "--training-weight-late-factor",
        type=float,
        default=1.0,
        help="Multiplicative weight for steps at/after --training-weight-late-start-step.",
    )
    parser.add_argument(
        "--training-weight-final-factor",
        type=float,
        default=1.0,
        help="Multiplicative weight for the last paired coupling iteration of each step.",
    )
    parser.add_argument(
        "--training-weight-iteration-factor",
        type=float,
        default=0.0,
        help="Additive normalized coupling-iteration weight used by coupled training modes.",
    )
    parser.add_argument("--basis-kind", choices=("absolute", "increment"), default="absolute")
    parser.add_argument("--max-training-snapshots", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default="cpp")
    parser.add_argument("--quad-order", type=int, default=1)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    parser.add_argument("--dynamic-tau", type=float, default=1.0)
    parser.add_argument(
        "--rom-incompressibility-scale",
        type=float,
        default=1.0,
        help=(
            "Scale ALE-DVMS incompressibility stabilization terms in sampled reduced solves. "
            "Use 1.0 to reproduce the exact FOM operator."
        ),
    )
    parser.add_argument("--reference-velocity", type=float, default=None)
    parser.add_argument("--velocity-modes", type=int, default=3)
    parser.add_argument("--pressure-modes", type=int, default=5)
    parser.add_argument(
        "--supremizer-enrichment",
        action="store_true",
        help="Append homogeneous velocity supremizer modes generated from the pressure POD modes.",
    )
    parser.add_argument(
        "--supremizer-modes",
        type=int,
        default=None,
        help="Number of pressure modes used to generate supremizers; default uses all pressure modes.",
    )
    parser.add_argument(
        "--supremizer-riesz",
        choices=("h1", "mass_lhs"),
        default="h1",
        help="Velocity Riesz operator used for the auxiliary supremizer solves.",
    )
    parser.add_argument(
        "--interface-load-enrichment-velocity-modes",
        type=int,
        default=0,
        help=(
            "Append this many velocity POD modes from reaction-weighted all-state increments before "
            "supremizer enrichment."
        ),
    )
    parser.add_argument(
        "--interface-load-enrichment-pressure-modes",
        type=int,
        default=0,
        help="Append this many pressure POD modes from reaction-weighted all-state increments.",
    )
    parser.add_argument(
        "--interface-load-enrichment-weight",
        choices=("uniform", "reaction_norm", "reaction_change"),
        default="reaction_norm",
        help="Quantity used to weight snapshots for interface-load basis enrichment.",
    )
    parser.add_argument(
        "--interface-load-enrichment-weight-exponent",
        type=float,
        default=1.0,
        help="Exponent applied to the normalized interface-load enrichment weights.",
    )
    parser.add_argument(
        "--interface-load-enrichment-max-weight",
        type=float,
        default=25.0,
        help="Upper bound for normalized interface-load enrichment weights.",
    )
    parser.add_argument(
        "--interface-load-enrichment-max-states",
        type=int,
        default=0,
        help="Maximum state increments used for interface-load enrichment; 0 keeps all selected states.",
    )
    parser.add_argument(
        "--accepted-checkpoint-enrichment-dir",
        type=Path,
        default=None,
        help=(
            "Optional full-run checkpoint directory used to enrich the homogeneous fluid basis. "
            "This is the cheap way to train the state space on the whole accepted trajectory."
        ),
    )
    parser.add_argument(
        "--accepted-checkpoint-enrichment-steps",
        type=str,
        default=None,
        help="Comma/range step list for accepted checkpoint basis enrichment; default keeps every checkpoint.",
    )
    parser.add_argument(
        "--accepted-checkpoint-enrichment-kind",
        choices=("increment", "absolute"),
        default="increment",
        help="Use accepted step increments x_n-x_{n-1} or absolute accepted states for the enrichment POD.",
    )
    parser.add_argument(
        "--accepted-checkpoint-enrichment-velocity-modes",
        type=int,
        default=0,
        help="Number of velocity modes appended from the accepted checkpoint trajectory.",
    )
    parser.add_argument(
        "--accepted-checkpoint-enrichment-pressure-modes",
        type=int,
        default=0,
        help="Number of pressure modes appended from the accepted checkpoint trajectory.",
    )
    parser.add_argument("--lspg-max-iterations", type=int, default=10)
    parser.add_argument("--lspg-residual-tol", type=float, default=1.0e-8)
    parser.add_argument("--lspg-line-search", action="store_true")
    parser.add_argument(
        "--training-residual-trajectory",
        choices=("lspg", "projection", "galerkin"),
        default="lspg",
        help=(
            "Source states for residual-basis and element-cubature training. "
            "'projection' is a fast operator-weighted mode using the exact accepted increment projected into the trial basis; "
            "'galerkin' uses the full-mesh reduced Galerkin Newton path followed by the sampled_galerkin branch."
        ),
    )
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
    parser.add_argument(
        "--operator-training-max-pairs",
        type=int,
        default=0,
        help=(
            "Optional stratified cap for residual/cubature operator training pairs. "
            "The trial basis, reaction decoder, and impedance fit still use all selected training pairs; "
            "0 uses all pairs for the operator too."
        ),
    )
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
        choices=(
            "none",
            "residual-lsq-bounded",
            "reduced-operator-lsq-bounded",
            "reduced-operator-greedy",
            "reduced-operator-topnorm",
            "reduced-operator-block-conservative",
        ),
        default="none",
        help=(
            "Fit nonnegative element cubature weights. 'residual-lsq-bounded' matches sampled residual rows; "
            "'reduced-operator-lsq-bounded' and 'reduced-operator-greedy' match element contributions to "
            "V^T R and V^T J V; 'reduced-operator-block-conservative' fits those operators per physical "
            "element block with bounded weights."
        ),
    )
    parser.add_argument(
        "--gnat-element-max-training-states",
        type=int,
        default=24,
        help="Maximum reduced states used in the element cubature least-squares fit.",
    )
    parser.add_argument(
        "--gnat-element-training-state-selection",
        choices=("first", "stratified"),
        default="first",
        help=(
            "How the capped element-cubature training states are selected. "
            "'first' preserves the historical file-order truncation; 'stratified' spreads the budget across "
            "all selected training probe pairs and favors each pair's final projected state when only one "
            "state can be kept."
        ),
    )
    parser.add_argument(
        "--gnat-element-fit-max-rows",
        type=int,
        default=0,
        help=(
            "Optional cap for rows used by the dense element-cubature fit. "
            "Rows are selected by a deterministic mix of largest row norms and uniform coverage; 0 keeps all rows."
        ),
    )
    parser.add_argument(
        "--gnat-element-fit-solver",
        choices=("bounded", "ridge-clip"),
        default="bounded",
        help=(
            "Least-squares backend for element cubature weights. 'bounded' uses SciPy's bounded optimizer; "
            "'ridge-clip' solves ridge normal equations and clips to nonnegative bounded weights for faster sweeps."
        ),
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
        "--gnat-element-candidate-elements",
        choices=("sample-mesh", "all"),
        default="sample-mesh",
        help=(
            "Candidate elements for reduced-operator cubature. 'sample-mesh' is fast; "
            "'all' is more complete but can make the nonnegative LS fit expensive."
        ),
    )
    parser.add_argument(
        "--gnat-element-target-count",
        type=int,
        default=0,
        help=(
            "Target retained element count for reduced-operator-topnorm or block-conservative cubature; "
            "0 uses 1000 or all candidates."
        ),
    )
    parser.add_argument(
        "--gnat-element-block-regularization",
        type=float,
        default=1.0e-8,
        help="Near-prior Tikhonov regularization used by reduced-operator-block-conservative cubature.",
    )
    parser.add_argument(
        "--gnat-element-block-sum-scale",
        type=float,
        default=1.0e-4,
        help="Relative strength of each block's total-weight conservation row.",
    )
    parser.add_argument(
        "--gnat-element-block-min-elements",
        type=int,
        default=32,
        help="Minimum selected elements for each optional physical block in block-conservative cubature.",
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
    parser.add_argument("--gnat-objective", choices=("gnat", "sampled_lspg", "sampled_galerkin"), default="gnat")
    parser.add_argument("--gnat-max-iterations", type=int, default=10)
    parser.add_argument("--gnat-residual-tol", type=float, default=1.0e-8)
    parser.add_argument("--gnat-line-search", action="store_true")
    parser.add_argument(
        "--save-hrom-model",
        type=Path,
        default=None,
        help="Optional .npz path for the deployable sampled-LSPG HROM model used by run_example2_local.py.",
    )
    parser.add_argument(
        "--fit-reduced-reaction-operator",
        action="store_true",
        help=(
            "Fit and store a reduced interface-reaction operator in the saved HROM model. "
            "The default fit is incremental, so the online predictor adds the current load guess."
        ),
    )
    parser.add_argument(
        "--reaction-operator-kind",
        choices=("incremental", "absolute"),
        default="incremental",
        help="Fit reaction increments from the pre-fluid load or absolute reaction values.",
    )
    parser.add_argument("--reaction-operator-ridge", type=float, default=1.0e-10)
    parser.add_argument(
        "--reaction-operator-validation-stride",
        type=int,
        default=5,
        help="Every Nth reaction snapshot is held out for validation; <=1 disables the holdout.",
    )
    parser.add_argument(
        "--fit-sampled-reaction-operator",
        action="store_true",
        help="Fit and store a sampled nonlinear gappy-POD interface reaction operator G(Vq).",
    )
    parser.add_argument("--sampled-reaction-modes", type=int, default=24)
    parser.add_argument("--sampled-reaction-energy", type=float, default=0.0)
    parser.add_argument("--sampled-reaction-oversampling", type=int, default=2)
    parser.add_argument(
        "--sampled-reaction-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Optional full-run checkpoint directory whose accepted fluid reactions are appended to "
            "the gappy-POD reaction training snapshots."
        ),
    )
    parser.add_argument(
        "--sampled-reaction-checkpoint-steps",
        type=str,
        default=None,
        help="Comma/range step list for --sampled-reaction-checkpoint-dir; default keeps every checkpoint.",
    )
    parser.add_argument(
        "--sampled-reaction-validation-stride",
        type=int,
        default=5,
        help="Every Nth sampled nonlinear reaction snapshot is held out for validation; <=1 disables the holdout.",
    )
    parser.add_argument(
        "--fit-interface-impedance-operator",
        action="store_true",
        help="Fit and store a low-rank secant interface impedance/tangent map from coupling-iteration data.",
    )
    parser.add_argument(
        "--interface-impedance-cosim-dir",
        type=Path,
        default=None,
        help=(
            "Optional coSimData directory used to fit the interface impedance on a full coupled run. "
            "When set, this replaces the stage-probe-only secant fit for the impedance operator."
        ),
    )
    parser.add_argument(
        "--interface-impedance-cosim-load-key",
        choices=("fluid_load_return_data", "load_return_data", "load_data"),
        default="fluid_load_return_data",
        help=(
            "coSimData load array used by --interface-impedance-cosim-dir. "
            "Use fluid_load_return_data for the fluid reaction sign stored in HROM reaction lookups."
        ),
    )
    parser.add_argument("--interface-impedance-modes", type=int, default=24)
    parser.add_argument("--interface-impedance-ridge", type=float, default=1.0e-8)
    parser.add_argument(
        "--interface-impedance-validation-stride",
        type=int,
        default=5,
        help="Every Nth interface secant sample is held out for validation; <=1 disables the holdout.",
    )
    parser.add_argument(
        "--save-hrom-model-exit-after-save",
        action="store_true",
        help="Exit immediately after the deployable HROM .npz is written, skipping replay diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _progress(
        "start "
        f"probe_dir={args.probe_dir} steps={args.steps} "
        f"training_steps={args.training_steps} training_all_iters={bool(args.training_all_iters)}"
    )
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
    training_roots = _training_probe_roots(args, fallback_to_probe_dir=str(args.basis_kind) == "increment")
    gnat_training_pairs = []
    training_pair_weights: dict[tuple[int, int], float] | None = None
    training_weight_summary: dict[str, Any] = {
        "mode": str(args.training_weight_mode),
        "enabled": False,
    }

    if str(args.basis_kind) == "increment":
        increment_source_label = ", ".join(str(root) for root in training_roots)
        train_pairs = _find_fluid_stage_probe_pairs_from_roots(
            training_roots,
            final_only=not bool(args.training_all_iters),
            steps=training_steps,
        )
        _progress(
            "fit increment basis: "
            f"pairs={len(train_pairs)} source={increment_source_label} "
            f"velocity_modes={int(args.velocity_modes)} pressure_modes={int(args.pressure_modes)} "
            f"supremizer={int(bool(args.supremizer_enrichment))}"
        )
        gnat_training_pairs = list(train_pairs)
        training_pair_weights = _training_pair_weight_map(
            gnat_training_pairs,
            mode=str(args.training_weight_mode),
            late_start_step=args.training_weight_late_start_step,
            late_factor=float(args.training_weight_late_factor),
            final_factor=float(args.training_weight_final_factor),
            iteration_factor=float(args.training_weight_iteration_factor),
        )
        trial_basis, training_source = _fit_increment_basis_from_probe_pairs(
            operator=operator,
            pairs=train_pairs,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
            velocity_modes=int(args.velocity_modes),
            pressure_modes=int(args.pressure_modes),
            pair_weights=training_pair_weights,
            include_intermediate_states=bool(args.training_all_states),
            supremizer_enrichment=False,
        )
        training_kind = (
            "stage-increment-probe"
            if args.training_probe_dir is not None or args.training_steps is not None
            else "stage-increment-probe-oracle"
        )
        train_batch = None
    elif training_roots:
        train_pairs = _find_fluid_stage_probe_pairs_from_roots(
            training_roots,
            final_only=not bool(args.training_all_iters),
            steps=training_steps,
        )
        _progress(
            "fit post-state basis: "
            f"pairs={len(train_pairs)} source={', '.join(str(root) for root in training_roots)} "
            f"velocity_modes={int(args.velocity_modes)} pressure_modes={int(args.pressure_modes)}"
        )
        gnat_training_pairs = list(train_pairs)
        training_pair_weights = _training_pair_weight_map(
            gnat_training_pairs,
            mode=str(args.training_weight_mode),
            late_start_step=args.training_weight_late_start_step,
            late_factor=float(args.training_weight_late_factor),
            final_factor=float(args.training_weight_final_factor),
            iteration_factor=float(args.training_weight_iteration_factor),
        )
        if bool(args.training_all_states):
            train_state_paths = _training_state_probe_paths_from_pairs(
                train_pairs,
                include_pre=True,
                include_intermediate=True,
                include_post=True,
            )
            train_probes = [load_fluid_stage_probe(path) for _pair, _label, path in train_state_paths]
            training_source = [f"{path}#{label}" for _pair, label, path in train_state_paths]
        else:
            train_probes = [load_fluid_stage_probe(pair.post_path) for pair in train_pairs]
            training_source = [str(pair.post_path) for pair in train_pairs]
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
        training_kind = "stage-probe-all-states" if bool(args.training_all_states) else "stage-probe-post-fluid-solve"
    else:
        checkpoint_paths = _checkpoint_paths(args.training_checkpoints, max_snapshots=int(args.max_training_snapshots))
        _progress(
            "fit checkpoint basis: "
            f"checkpoints={len(checkpoint_paths)} source={args.training_checkpoints}"
        )
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
    if training_pair_weights:
        values = np.asarray(list(training_pair_weights.values()), dtype=float)
        training_weight_summary = {
            "mode": str(args.training_weight_mode),
            "enabled": bool(str(args.training_weight_mode) != "uniform"),
            "late_start_step": None
            if args.training_weight_late_start_step is None
            else int(args.training_weight_late_start_step),
            "late_factor": float(args.training_weight_late_factor),
            "final_factor": float(args.training_weight_final_factor),
            "iteration_factor": float(args.training_weight_iteration_factor),
            "min": float(np.min(values)) if values.size else float("nan"),
            "max": float(np.max(values)) if values.size else float("nan"),
            "mean": float(np.mean(values)) if values.size else float("nan"),
        }
    if str(args.basis_kind) != "increment":
        _progress("fit POD basis from training batch")
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

    interface_enrichment_summary: dict[str, Any] = {"enabled": False}
    interface_velocity_modes = max(0, int(args.interface_load_enrichment_velocity_modes))
    interface_pressure_modes = max(0, int(args.interface_load_enrichment_pressure_modes))
    if interface_velocity_modes > 0 or interface_pressure_modes > 0:
        if not gnat_training_pairs:
            raise ValueError("Interface-load enrichment requires stage-probe training pairs.")
        _progress(
            "collect interface-load enrichment snapshots: "
            f"velocity_modes={interface_velocity_modes} pressure_modes={interface_pressure_modes} "
            f"weight={str(args.interface_load_enrichment_weight)}"
        )
        interface_matrix, interface_sources, interface_enrichment_summary = (
            _interface_load_enrichment_matrix_from_probe_pairs(
                operator=operator,
                pairs=gnat_training_pairs,
                setup=setup,
                fluid_iface_coords=fluid_iface_coords,
                dt=float(dt),
                reference_velocity=float(reference_velocity),
                pair_weights=training_pair_weights,
                include_intermediate_states=bool(args.training_all_states),
                weight_mode=str(args.interface_load_enrichment_weight),
                weight_exponent=float(args.interface_load_enrichment_weight_exponent),
                max_weight=float(args.interface_load_enrichment_max_weight),
                max_states=int(args.interface_load_enrichment_max_states),
            )
        )
        if int(interface_matrix.shape[1]) > 0:
            before_modes = int(trial_basis.n_modes)
            before_velocity_modes = int(trial_basis.n_velocity_modes)
            before_pressure_modes = int(trial_basis.n_pressure_modes)
            trial_basis = enrich_fluid_pod_trial_basis_with_interface_state_modes(
                operator,
                trial_basis,
                interface_matrix,
                velocity_modes=interface_velocity_modes,
                pressure_modes=interface_pressure_modes,
                center=False,
            )
            interface_enrichment_summary.update(
                {
                    "requested_velocity_modes": int(interface_velocity_modes),
                    "requested_pressure_modes": int(interface_pressure_modes),
                    "accepted_velocity_modes": int(trial_basis.n_velocity_modes)
                    - int(before_velocity_modes),
                    "accepted_pressure_modes": int(trial_basis.n_pressure_modes)
                    - int(before_pressure_modes),
                    "total_modes_before": int(before_modes),
                    "total_modes_after": int(trial_basis.n_modes),
                }
            )
            training_source = list(training_source) + [f"interface_enrichment:{item}" for item in interface_sources]
            _progress(
                "interface-load enrichment applied: "
                f"modes_before={before_modes} modes_after={int(trial_basis.n_modes)} "
                f"states={int(interface_matrix.shape[1])}"
            )
        else:
            _progress("interface-load enrichment skipped: no usable weighted snapshots")

    accepted_checkpoint_enrichment_summary: dict[str, Any] = {"enabled": False}
    accepted_velocity_modes = max(0, int(args.accepted_checkpoint_enrichment_velocity_modes))
    accepted_pressure_modes = max(0, int(args.accepted_checkpoint_enrichment_pressure_modes))
    if args.accepted_checkpoint_enrichment_dir is not None and (
        accepted_velocity_modes > 0 or accepted_pressure_modes > 0
    ):
        _progress(
            "collect accepted checkpoint enrichment snapshots: "
            f"dir={Path(args.accepted_checkpoint_enrichment_dir)} "
            f"steps={args.accepted_checkpoint_enrichment_steps or 'all'} "
            f"kind={str(args.accepted_checkpoint_enrichment_kind)} "
            f"velocity_modes={accepted_velocity_modes} pressure_modes={accepted_pressure_modes}"
        )
        accepted_matrix, accepted_sources, accepted_checkpoint_enrichment_summary = (
            _accepted_checkpoint_state_matrix(
                operator=operator,
                fluid=fluid,
                checkpoint_dir=Path(args.accepted_checkpoint_enrichment_dir),
                steps_spec=args.accepted_checkpoint_enrichment_steps,
                kind=str(args.accepted_checkpoint_enrichment_kind),
            )
        )
        if int(accepted_matrix.shape[1]) > 0:
            before_modes = int(trial_basis.n_modes)
            before_velocity_modes = int(trial_basis.n_velocity_modes)
            before_pressure_modes = int(trial_basis.n_pressure_modes)
            trial_basis = enrich_fluid_pod_trial_basis_with_interface_state_modes(
                operator,
                trial_basis,
                accepted_matrix,
                velocity_modes=accepted_velocity_modes,
                pressure_modes=accepted_pressure_modes,
                center=False,
            )
            accepted_checkpoint_enrichment_summary.update(
                {
                    "requested_velocity_modes": int(accepted_velocity_modes),
                    "requested_pressure_modes": int(accepted_pressure_modes),
                    "accepted_velocity_modes": int(trial_basis.n_velocity_modes)
                    - int(before_velocity_modes),
                    "accepted_pressure_modes": int(trial_basis.n_pressure_modes)
                    - int(before_pressure_modes),
                    "total_modes_before": int(before_modes),
                    "total_modes_after": int(trial_basis.n_modes),
                }
            )
            training_source = list(training_source) + [
                f"accepted_checkpoint_enrichment:{item}" for item in accepted_sources
            ]
            _progress(
                "accepted checkpoint enrichment applied: "
                f"modes_before={before_modes} modes_after={int(trial_basis.n_modes)} "
                f"states={int(accepted_matrix.shape[1])}"
            )
        else:
            _progress(
                "accepted checkpoint enrichment skipped: "
                f"{accepted_checkpoint_enrichment_summary.get('reason', 'no usable states')}"
            )

    if bool(args.supremizer_enrichment):
        _progress(
            "enrich POD basis with supremizers: "
            f"requested={args.supremizer_modes if args.supremizer_modes is not None else int(trial_basis.n_pressure_modes)} "
            f"riesz={str(args.supremizer_riesz)}"
        )
        trial_basis = enrich_fluid_pod_trial_basis_with_supremizers(
            operator,
            trial_basis,
            supremizer_modes=args.supremizer_modes,
            riesz=str(args.supremizer_riesz),
        )
    if bool(args.supremizer_enrichment):
        _progress(
            "basis after supremizer enrichment: "
            f"velocity_pod={int(trial_basis.n_velocity_pod_modes)} "
            f"velocity_supremizer={int(trial_basis.n_velocity_supremizer_modes)} "
            f"pressure={int(trial_basis.n_pressure_modes)} "
            f"total={int(trial_basis.n_modes)}"
        )

    all_state_training_summary: dict[str, Any] | None = None
    if args.export_all_state_training is not None:
        if not gnat_training_pairs:
            raise ValueError("--export-all-state-training requires stage-probe training pairs.")
        all_state_training_summary = _export_all_state_training_database_from_probe_pairs(
            path=Path(args.export_all_state_training),
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
            trajectory_source=str(args.training_residual_trajectory),
            include_jacobian_actions=not bool(args.no_export_jacobian_actions),
            pair_weights=training_pair_weights,
        )

    stage_pairs = find_fluid_stage_probe_pairs(
        args.probe_dir,
        final_only=not bool(args.replay_all_iters),
        steps=_parse_steps(str(args.steps)),
    )
    _progress(f"replay pairs selected: {len(stage_pairs)}")
    operator_training_pairs = _stratified_probe_pair_subset(
        list(gnat_training_pairs),
        int(args.operator_training_max_pairs),
    )
    if gnat_training_pairs and len(operator_training_pairs) != len(gnat_training_pairs):
        _progress(
            "operator training pair subset: "
            f"{len(operator_training_pairs)}/{len(gnat_training_pairs)} "
            f"max_pairs={int(args.operator_training_max_pairs)}"
        )
    replay_entries: list[dict[str, Any]] = []
    full_assembly_times: list[float] = []
    lspg_times: list[float] = []
    gnat_times: list[float] = []
    residual_training_columns: list[np.ndarray] = []
    residual_training_entries: list[dict[str, Any]] = []
    saved_hrom_model_path: str | None = None
    reduced_reaction_operator_summary: dict[str, Any] | None = None
    interface_impedance_operator_summary: dict[str, Any] | None = None
    training_pair_steps = [int(pair.step) for pair in gnat_training_pairs]
    if not bool(args.skip_gnat) and operator_training_pairs:
        _progress(f"GNAT/sample training pairs: {len(operator_training_pairs)}")
        residual_training_columns, residual_training_entries = _collect_training_residual_snapshots_from_probe_pairs(
            operator=operator,
            fluid=fluid,
            pairs=operator_training_pairs,
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
            pair_weights=training_pair_weights,
            trajectory_source=str(args.training_residual_trajectory),
            include_intermediate_states=bool(args.training_all_states),
            incompressibility_stabilization_scale=float(args.rom_incompressibility_scale),
        )

    for pair in stage_pairs:
        _progress(f"replay pair: step={int(pair.step)} iter={int(pair.coupling_iter)}")
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
            _progress(
                "fit GNAT sample set: "
                f"residual_shape={tuple(residual_snapshots.shape)} "
                f"residual_modes={args.residual_modes} min_rows={int(args.min_sample_rows)}"
            )
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
                    pairs=operator_training_pairs,
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
                    element_weighting=str(args.gnat_element_weighting),
                    element_candidate_mode=str(args.gnat_element_candidate_elements),
                    element_target_count=int(args.gnat_element_target_count),
                    block_regularization=float(args.gnat_element_block_regularization),
                    block_sum_scale=float(args.gnat_element_block_sum_scale),
                    block_min_elements=int(args.gnat_element_block_min_elements),
                    fit_max_rows=int(args.gnat_element_fit_max_rows),
                    fit_solver=str(args.gnat_element_fit_solver),
                    pair_weights=training_pair_weights,
                    trajectory_source=str(args.training_residual_trajectory),
                    training_state_selection=str(args.gnat_element_training_state_selection),
                    include_intermediate_states=bool(args.training_all_states),
                    incompressibility_stabilization_scale=float(args.rom_incompressibility_scale),
            )
            if args.save_hrom_model is not None and saved_hrom_model_path is None:
                _progress(f"save HROM model: {Path(args.save_hrom_model)}")
                training_coefficients_for_model = _project_training_state_coefficients_from_probe_pairs(
                    operator=operator,
                    pairs=gnat_training_pairs,
                    trial_basis=trial_basis,
                    setup=setup,
                    fluid_iface_coords=fluid_iface_coords,
                    dt=float(dt),
                    reference_velocity=float(reference_velocity),
                    include_intermediate_states=bool(args.training_all_states),
                )
                if args.accepted_checkpoint_enrichment_dir is not None:
                    checkpoint_training_coefficients = _project_training_state_coefficients_from_checkpoints(
                        operator=operator,
                        fluid=fluid,
                        checkpoint_dir=Path(args.accepted_checkpoint_enrichment_dir),
                        steps_spec=args.accepted_checkpoint_enrichment_steps,
                        trial_basis=trial_basis,
                        kind=str(args.accepted_checkpoint_enrichment_kind),
                    )
                    if int(checkpoint_training_coefficients.shape[1]) > 0:
                        training_coefficients_for_model = np.column_stack(
                            [training_coefficients_for_model, checkpoint_training_coefficients]
                        )
                reaction_operator = None
                sampled_reaction_operator = None
                impedance_operator = None
                if bool(args.fit_reduced_reaction_operator):
                    _progress(
                        "fit reduced reaction operator: "
                        f"kind={str(args.reaction_operator_kind)} "
                        f"ridge={float(args.reaction_operator_ridge):.3e}"
                    )
                    reaction_operator = _fit_reduced_reaction_operator_from_probe_pairs(
                        operator=operator,
                        pairs=gnat_training_pairs,
                        trial_basis=trial_basis,
                        setup=setup,
                        fluid_iface_coords=fluid_iface_coords,
                        dt=float(dt),
                        reference_velocity=float(reference_velocity),
                        include_intermediate_states=bool(args.training_all_states),
                        ridge=float(args.reaction_operator_ridge),
                        incremental=str(args.reaction_operator_kind).strip().lower() == "incremental",
                        validation_stride=int(args.reaction_operator_validation_stride),
                    )
                    reduced_reaction_operator_summary = {
                        key: value
                        for key, value in reaction_operator.items()
                        if key not in {"matrix", "bias", "coords"}
                    }
                    _progress(
                        "fit reduced reaction operator: done "
                        f"fit_mean={float(reaction_operator['fit_mean_relative_error']):.3e} "
                        f"fit_max={float(reaction_operator['fit_max_relative_error']):.3e} "
                        f"validation_max={float(reaction_operator['validation_max_relative_error']):.3e}"
                    )
                if bool(args.fit_sampled_reaction_operator):
                    _progress(
                        "fit sampled nonlinear reaction operator: "
                        f"modes={int(args.sampled_reaction_modes)} "
                        f"oversampling={int(args.sampled_reaction_oversampling)}"
                    )
                    sampled_reaction_operator = _fit_sampled_nonlinear_reaction_operator_from_probe_pairs(
                        operator=operator,
                        pairs=gnat_training_pairs,
                        setup=setup,
                        fluid_iface_coords=fluid_iface_coords,
                        dt=float(dt),
                        reference_velocity=float(reference_velocity),
                        include_intermediate_states=bool(args.training_all_states),
                        modes=int(args.sampled_reaction_modes),
                        energy=float(args.sampled_reaction_energy),
                        oversampling=int(args.sampled_reaction_oversampling),
                        validation_stride=int(args.sampled_reaction_validation_stride),
                        checkpoint_dir=args.sampled_reaction_checkpoint_dir,
                        checkpoint_steps=args.sampled_reaction_checkpoint_steps,
                    )
                    _progress(
                        "fit sampled nonlinear reaction operator: done "
                        f"modes={int(sampled_reaction_operator['modes'])} "
                        f"samples={int(sampled_reaction_operator['sample_count'])} "
                        f"sample_elements={int(sampled_reaction_operator['sample_element_count'])} "
                        f"fit_mean={float(sampled_reaction_operator['fit_mean_relative_error']):.3e} "
                        f"validation_max={float(sampled_reaction_operator['validation_max_relative_error']):.3e}"
                    )
                if bool(args.fit_interface_impedance_operator):
                    _progress(
                        "fit interface impedance operator: "
                        f"modes={int(args.interface_impedance_modes)} "
                        f"ridge={float(args.interface_impedance_ridge):.3e} "
                        f"source={Path(args.interface_impedance_cosim_dir) if args.interface_impedance_cosim_dir is not None else 'stage-probes'}"
                    )
                    if args.interface_impedance_cosim_dir is not None:
                        impedance_operator = _fit_interface_impedance_operator_from_cosim_data(
                            cosim_dir=Path(args.interface_impedance_cosim_dir),
                            modes=int(args.interface_impedance_modes),
                            ridge=float(args.interface_impedance_ridge),
                            validation_stride=int(args.interface_impedance_validation_stride),
                            velocity_scale=float(dt),
                            load_key=str(args.interface_impedance_cosim_load_key),
                        )
                    else:
                        impedance_operator = _fit_interface_impedance_operator_from_probe_pairs(
                            pairs=gnat_training_pairs,
                            modes=int(args.interface_impedance_modes),
                            ridge=float(args.interface_impedance_ridge),
                            validation_stride=int(args.interface_impedance_validation_stride),
                            velocity_scale=float(dt),
                        )
                    interface_impedance_operator_summary = {
                        key: value
                        for key, value in impedance_operator.items()
                        if key not in {"matrix", "bias", "coords", "feature_basis", "feature_mean"}
                    }
                    _progress(
                        "fit interface impedance operator: done "
                        f"fit_mean={float(impedance_operator['fit_mean_relative_error']):.3e} "
                        f"fit_max={float(impedance_operator['fit_max_relative_error']):.3e} "
                        f"validation_max={float(impedance_operator['validation_max_relative_error']):.3e}"
                    )
                _save_sampled_lspg_hrom_model(
                    Path(args.save_hrom_model),
                    trial_basis=trial_basis,
                    sample_set=sample_set,
                    args=args,
                    training_source=[str(item) for item in training_source],
                    training_steps=training_pair_steps,
                    training_coefficients=training_coefficients_for_model,
                    reaction_operator=reaction_operator,
                    sampled_reaction_operator=sampled_reaction_operator,
                    impedance_operator=impedance_operator,
                )
                saved_hrom_model_path = str(Path(args.save_hrom_model))
                if bool(args.save_hrom_model_exit_after_save):
                    _progress("save HROM model complete; exiting before replay diagnostics")
                    summary = {
                        "training": {
                            "kind": training_kind,
                            "source": training_source,
                            "steps": training_pair_steps,
                            "all_iters": bool(args.training_all_iters),
                            "all_states": bool(args.training_all_states),
                            "weights": training_weight_summary,
                        },
                        "interface_load_enrichment": interface_enrichment_summary,
                        "accepted_checkpoint_enrichment": accepted_checkpoint_enrichment_summary,
                        "saved_hrom_model": saved_hrom_model_path,
                        "reduced_reaction_operator": None
                        if reaction_operator is None
                        else {
                            key: value
                            for key, value in reaction_operator.items()
                            if key not in {"matrix", "bias", "coords"}
                        },
                        "interface_impedance_operator": None
                        if impedance_operator is None
                        else {
                            key: value
                            for key, value in impedance_operator.items()
                            if key not in {"matrix", "bias", "coords", "feature_basis", "feature_mean"}
                        },
                        "sampled_nonlinear_reaction_operator": None
                        if sampled_reaction_operator is None
                        else {
                            key: value
                            for key, value in sampled_reaction_operator.items()
                            if key
                            not in {
                                "coords",
                                "basis",
                                "mean",
                                "sample_row_dofs",
                                "sample_element_ids",
                                "sample_to_coefficients",
                                "sample_output_positions",
                            }
                        },
                        "operator_training_pairs": int(len(operator_training_pairs)),
                        "operator_training_pair_cap": int(args.operator_training_max_pairs),
                    }
                    dump_json(summary, Path(args.output))
                    return
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
                incompressibility_stabilization_scale=float(args.rom_incompressibility_scale),
            )
            t_gnat0 = time.perf_counter()
            _progress(f"solve GNAT replay: step={int(pair.step)} iter={int(pair.coupling_iter)}")
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
            "training_all_states": bool(args.training_all_states),
            "replay_step_overlap": training_replay_overlap,
            "requested_velocity_modes": int(args.velocity_modes),
            "requested_pressure_modes": int(args.pressure_modes),
            "velocity_modes": int(trial_basis.n_velocity_modes),
            "velocity_pod_modes": int(trial_basis.n_velocity_pod_modes),
            "velocity_supremizer_modes": int(trial_basis.n_velocity_supremizer_modes),
            "pressure_modes": int(trial_basis.n_pressure_modes),
            "total_modes": int(trial_basis.n_modes),
            "supremizer_enrichment": bool(args.supremizer_enrichment),
            "supremizer_riesz": str(trial_basis.supremizer_riesz),
            "interface_load_enrichment": interface_enrichment_summary,
            "accepted_checkpoint_enrichment": accepted_checkpoint_enrichment_summary,
            "mode_choice": "fixed to the previously selected Tiba-style held-out validation result unless overridden",
            "basis_kind": str(args.basis_kind),
            "training_weights": training_weight_summary,
            "training_residual_trajectory": str(args.training_residual_trajectory),
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
            "gnat_element_training_state_selection": str(args.gnat_element_training_state_selection),
            "gnat_element_fit_max_rows": int(args.gnat_element_fit_max_rows),
            "gnat_element_fit_solver": str(args.gnat_element_fit_solver),
            "gnat_element_weight_max": float(args.gnat_element_weight_max),
            "gnat_element_prune_tol": float(args.gnat_element_prune_tol),
            "gnat_element_keep_interface": bool(args.gnat_element_keep_interface),
            "gnat_element_block_regularization": float(args.gnat_element_block_regularization),
            "gnat_element_block_sum_scale": float(args.gnat_element_block_sum_scale),
            "gnat_element_block_min_elements": int(args.gnat_element_block_min_elements),
            "force_interface_rows": str(args.force_interface_rows),
            "gnat_objective": str(args.gnat_objective),
            "rom_incompressibility_scale": float(args.rom_incompressibility_scale),
            "saved_hrom_model": saved_hrom_model_path,
            "reduced_reaction_operator": reduced_reaction_operator_summary,
            "interface_impedance_operator": interface_impedance_operator_summary,
            "all_state_training_database": None
            if args.export_all_state_training is None
            else str(Path(args.export_all_state_training)),
            "all_state_training_summary": all_state_training_summary,
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
    _progress(f"wrote summary: {Path(args.output)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
