from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from examples.NIRB.common import dump_json
from examples.NIRB.debug.replay_fluid_gnat_from_checkpoints import _make_operator
from examples.NIRB.debug.replay_fluid_rom_from_stage_probes import (
    _configure_and_restore_probe,
    _make_probe_inlet_lookup,
    _make_stage_acceleration_hook,
    _parse_steps,
)
from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.fluid_basis import fit_fluid_pod_trial_basis, fit_fluid_pod_trial_basis_from_state_matrix
from examples.NIRB.fluid_intrusive_rom import FluidIntrusiveReducedOperator, FluidReducedOperatorSplit
from examples.NIRB.fluid_lspg import pack_fluid_state
from examples.NIRB.fluid_stage_probes import (
    FluidStageProbe,
    FluidStageProbePair,
    build_stage_probe_batch,
    find_fluid_stage_probe_pairs,
    load_fluid_stage_probe,
)
from examples.NIRB.run_example2_local import (
    _bossak_coefficients,
    _boundary_field_data,
    _build_fluid_problem,
    _load_reference_partitioned_meshes,
)


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float).reshape(-1)
    cand = np.asarray(candidate, dtype=float).reshape(-1)
    return float(np.linalg.norm(cand - ref) / max(float(np.linalg.norm(ref)), 1.0e-15))


def _parse_modes(spec: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in str(spec).split(",") if item.strip())
    if not modes:
        raise ValueError("At least one contribution mode is required.")
    return modes


def _limit_pairs(pairs: Iterable[FluidStageProbePair], *, max_pairs: int | None) -> list[FluidStageProbePair]:
    items = list(pairs)
    if max_pairs is None:
        return items
    return items[: max(0, int(max_pairs))]


def _reaction_values(operator, probe: FluidStageProbe) -> np.ndarray:
    lookup = probe.reaction_lookup(kind="point")
    if lookup is not None:
        return np.asarray(lookup.values, dtype=float).reshape(-1)
    return np.asarray(operator.reaction_loads(refresh_state=False).values, dtype=float).reshape(-1)


def _fit_probe_basis(
    *,
    operator,
    pairs: list[FluidStageProbePair],
    setup,
    fluid_iface_coords: np.ndarray,
    dt: float,
    reference_velocity: float,
    basis_kind: str,
    velocity_modes: int,
    pressure_modes: int,
):
    kind = str(basis_kind)
    if kind == "absolute-post":
        post_probes = [load_fluid_stage_probe(pair.post_path) for pair in pairs]
        batch = build_stage_probe_batch(
            operator,
            post_probes,
            fluid_iface_coords=fluid_iface_coords,
            inlet_lookup_factory=lambda probe: _make_probe_inlet_lookup(
                setup=setup,
                probe=probe,
                dt=float(dt),
                reference_velocity=float(reference_velocity),
            ),
        )
        basis = fit_fluid_pod_trial_basis(
            operator,
            batch,
            velocity_modes=int(velocity_modes),
            pressure_modes=int(pressure_modes),
            center=False,
        )
        source = {
            "kind": kind,
            "columns": int(batch.state.shape[1]),
        }
        return basis, source

    if kind != "increment":
        raise ValueError(f"Unsupported basis kind {basis_kind!r}.")

    columns: list[np.ndarray] = []
    free_dofs: np.ndarray | None = None
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
        pre_state = pack_fluid_state(operator)
        if free_dofs is None:
            free_dofs = operator.free_fluid_dofs()
        _configure_and_restore_probe(
            operator=operator,
            probe=post_probe,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        )
        post_state = pack_fluid_state(operator)
        columns.append(np.asarray(post_state - pre_state, dtype=float).reshape(-1))

    if not columns:
        raise ValueError("Cannot fit an increment basis from zero probe pairs.")
    basis = fit_fluid_pod_trial_basis_from_state_matrix(
        operator,
        np.column_stack(columns),
        free_dofs=operator.free_fluid_dofs() if free_dofs is None else free_dofs,
        velocity_modes=int(velocity_modes),
        pressure_modes=int(pressure_modes),
        center=False,
    )
    source = {
        "kind": kind,
        "columns": len(columns),
    }
    return basis, source


def _split_summary(split: FluidReducedOperatorSplit) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for name, assembly in split.components.items():
        data[str(name)] = {
            "full_residual_norm": float(assembly.full_residual_norm),
            "reduced_residual_norm": float(assembly.reduced_residual_norm),
            "tangent_norm": float(np.linalg.norm(assembly.tangent)),
            "newton_step_norm": float(np.linalg.norm(assembly.newton_step())),
        }
    mass = split.mass
    if mass is not None:
        data["combined_mass"] = {
            "norm": float(np.linalg.norm(mass)),
            "condition": float(np.linalg.cond(mass)) if mass.size else 0.0,
        }
    return data


def _assemble_probe_state(
    *,
    operator,
    fluid: dict[str, object],
    basis,
    pair: FluidStageProbePair,
    setup,
    fluid_iface_coords: np.ndarray,
    dt: float,
    reference_velocity: float,
    bossak: dict[str, float],
    coefficients: np.ndarray,
    contribution_modes: tuple[str, ...],
) -> tuple[FluidReducedOperatorSplit, np.ndarray, np.ndarray, float]:
    pre_probe = load_fluid_stage_probe(pair.pre_path)
    _configure_and_restore_probe(
        operator=operator,
        probe=pre_probe,
        setup=setup,
        fluid_iface_coords=fluid_iface_coords,
        dt=float(dt),
        reference_velocity=float(reference_velocity),
    )
    offset = pack_fluid_state(operator)
    hook = _make_stage_acceleration_hook(
        fluid=fluid,
        bossak=bossak,
        preserve_seed_on_first_zero=True,
    )
    reduced = FluidIntrusiveReducedOperator(
        operator=operator,
        basis=basis,
        state_update_hook=hook,
        nonlinear_update_hook=operator.update_oss_after_nonlinear_update,
    )
    start = time.perf_counter()
    split = reduced.assemble_split(
        np.asarray(coefficients, dtype=float).reshape(-1),
        contribution_modes=contribution_modes,
        offset=offset,
    )
    elapsed = float(time.perf_counter() - start)
    state = pack_fluid_state(operator)
    reaction = np.asarray(operator.reaction_loads(refresh_state=False).values, dtype=float).reshape(-1)
    return split, state, reaction, elapsed


def _solve_probe_state(
    *,
    operator,
    fluid: dict[str, object],
    basis,
    pair: FluidStageProbePair,
    setup,
    fluid_iface_coords: np.ndarray,
    dt: float,
    reference_velocity: float,
    bossak: dict[str, float],
    initial_coefficients: np.ndarray,
    max_iterations: int,
    residual_tol: float,
    line_search: bool,
) -> tuple[Any, np.ndarray, np.ndarray, float]:
    pre_probe = load_fluid_stage_probe(pair.pre_path)
    _configure_and_restore_probe(
        operator=operator,
        probe=pre_probe,
        setup=setup,
        fluid_iface_coords=fluid_iface_coords,
        dt=float(dt),
        reference_velocity=float(reference_velocity),
    )
    offset = pack_fluid_state(operator)
    hook = _make_stage_acceleration_hook(
        fluid=fluid,
        bossak=bossak,
        preserve_seed_on_first_zero=True,
    )
    reduced = FluidIntrusiveReducedOperator(
        operator=operator,
        basis=basis,
        state_update_hook=hook,
        nonlinear_update_hook=operator.update_oss_after_nonlinear_update,
    )
    start = time.perf_counter()
    result = reduced.solve(
        np.asarray(initial_coefficients, dtype=float).reshape(-1),
        offset=offset,
        max_iterations=int(max_iterations),
        residual_tol=float(residual_tol),
        line_search=bool(line_search),
    )
    elapsed = float(time.perf_counter() - start)
    state = pack_fluid_state(operator)
    reaction = np.asarray(operator.reaction_loads(refresh_state=False).values, dtype=float).reshape(-1)
    return result, state, reaction, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the full-mesh intrusive reduced ALE-DVMS operator on saved fluid stage probes.",
    )
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--training-probe-dir", type=Path, default=None)
    parser.add_argument("--training-steps", type=str, default=None)
    parser.add_argument("--steps", type=str, default=None)
    parser.add_argument("--training-all-iters", action="store_true")
    parser.add_argument("--audit-all-iters", action="store_true")
    parser.add_argument("--max-training-pairs", type=int, default=None)
    parser.add_argument("--max-audit-pairs", type=int, default=None)
    parser.add_argument("--basis-kind", choices=("increment", "absolute-post"), default="increment")
    parser.add_argument("--velocity-modes", type=int, default=8)
    parser.add_argument("--pressure-modes", type=int, default=8)
    parser.add_argument(
        "--contribution-modes",
        type=str,
        default="system,mass_lhs,mass_stabilization,velocity",
        help="Comma-separated FluidFOMOperator contribution modes to project.",
    )
    parser.add_argument("--backend", choices=("python", "cpp"), default="cpp")
    parser.add_argument("--quad-order", type=int, default=3)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    parser.add_argument("--dynamic-tau", type=float, default=1.0)
    parser.add_argument("--reference-velocity", type=float, default=None)
    parser.add_argument("--solve-max-iterations", type=int, default=0)
    parser.add_argument("--solve-residual-tol", type=float, default=1.0e-10)
    parser.add_argument("--solve-line-search", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
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
    contribution_modes = _parse_modes(args.contribution_modes)

    training_root = args.training_probe_dir if args.training_probe_dir is not None else args.probe_dir
    training_pairs = _limit_pairs(
        find_fluid_stage_probe_pairs(
            training_root,
            final_only=not bool(args.training_all_iters),
            steps=_parse_steps(args.training_steps),
        ),
        max_pairs=args.max_training_pairs,
    )
    if not training_pairs:
        raise ValueError(f"No training probe pairs found under {training_root}.")

    basis, basis_source = _fit_probe_basis(
        operator=operator,
        pairs=training_pairs,
        setup=setup,
        fluid_iface_coords=fluid_iface_coords,
        dt=float(dt),
        reference_velocity=float(reference_velocity),
        basis_kind=str(args.basis_kind),
        velocity_modes=int(args.velocity_modes),
        pressure_modes=int(args.pressure_modes),
    )

    audit_pairs = _limit_pairs(
        find_fluid_stage_probe_pairs(
            args.probe_dir,
            final_only=not bool(args.audit_all_iters),
            steps=_parse_steps(args.steps),
        ),
        max_pairs=args.max_audit_pairs,
    )
    if not audit_pairs:
        raise ValueError(f"No audit probe pairs found under {args.probe_dir}.")

    entries: list[dict[str, Any]] = []
    for pair in audit_pairs:
        post_probe = load_fluid_stage_probe(pair.post_path)
        _configure_and_restore_probe(
            operator=operator,
            probe=post_probe,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        )
        exact_post_state = pack_fluid_state(operator)
        exact_post_reaction = _reaction_values(operator, post_probe)

        pre_probe = load_fluid_stage_probe(pair.pre_path)
        _configure_and_restore_probe(
            operator=operator,
            probe=pre_probe,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        )
        pre_state = pack_fluid_state(operator)
        target_coefficients = basis.project_state(exact_post_state, offset=pre_state)

        zero_split, zero_state, zero_reaction, zero_elapsed = _assemble_probe_state(
            operator=operator,
            fluid=fluid,
            basis=basis,
            pair=pair,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
            bossak=bossak,
            coefficients=np.zeros(int(basis.n_modes), dtype=float),
            contribution_modes=contribution_modes,
        )
        target_split, target_state, target_reaction, target_elapsed = _assemble_probe_state(
            operator=operator,
            fluid=fluid,
            basis=basis,
            pair=pair,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
            bossak=bossak,
            coefficients=target_coefficients,
            contribution_modes=contribution_modes,
        )

        zero_report = FluidIntrusiveReducedOperator(operator=operator, basis=basis).stability_report(zero_split.system)
        target_report = FluidIntrusiveReducedOperator(operator=operator, basis=basis).stability_report(
            target_split.system
        )
        entry = {
            "step": int(pair.step),
            "coupling_iter": int(pair.coupling_iter),
            "pre_probe": str(pair.pre_path),
            "post_probe": str(pair.post_path),
            "target_coefficient_norm": float(np.linalg.norm(target_coefficients)),
            "zero_state_error": _relative_l2(exact_post_state, zero_state),
            "zero_reaction_error": _relative_l2(exact_post_reaction, zero_reaction),
            "zero_assembly_elapsed_s": float(zero_elapsed),
            "zero_split": _split_summary(zero_split),
            "zero_stability": asdict(zero_report),
            "target_projection_state_error": _relative_l2(exact_post_state, target_state),
            "target_projection_reaction_error": _relative_l2(exact_post_reaction, target_reaction),
            "target_assembly_elapsed_s": float(target_elapsed),
            "target_split": _split_summary(target_split),
            "target_stability": asdict(target_report),
        }
        if int(args.solve_max_iterations) > 0:
            solve_result, solve_state, solve_reaction, solve_elapsed = _solve_probe_state(
                operator=operator,
                fluid=fluid,
                basis=basis,
                pair=pair,
                setup=setup,
                fluid_iface_coords=fluid_iface_coords,
                dt=float(dt),
                reference_velocity=float(reference_velocity),
                bossak=bossak,
                initial_coefficients=np.zeros(int(basis.n_modes), dtype=float),
                max_iterations=int(args.solve_max_iterations),
                residual_tol=float(args.solve_residual_tol),
                line_search=bool(args.solve_line_search),
            )
            entry["reduced_solve"] = {
                "result": asdict(solve_result),
                "elapsed_s": float(solve_elapsed),
                "state_error": _relative_l2(exact_post_state, solve_state),
                "reaction_error": _relative_l2(exact_post_reaction, solve_reaction),
            }
        entries.append(entry)

    output = {
        "probe_dir": str(Path(args.probe_dir).resolve()),
        "training_probe_dir": str(Path(training_root).resolve()),
        "basis": {
            "kind": str(args.basis_kind),
            "source": basis_source,
            "requested_velocity_modes": int(args.velocity_modes),
            "requested_pressure_modes": int(args.pressure_modes),
            "n_velocity_modes": int(basis.n_velocity_modes),
            "n_pressure_modes": int(basis.n_pressure_modes),
            "n_modes": int(basis.n_modes),
            "velocity_singular_values": np.asarray(basis.velocity_singular_values, dtype=float),
            "pressure_singular_values": np.asarray(basis.pressure_singular_values, dtype=float),
            "velocity_energy_fraction": np.asarray(basis.velocity_energy_fraction, dtype=float),
            "pressure_energy_fraction": np.asarray(basis.pressure_energy_fraction, dtype=float),
        },
        "settings": {
            "backend": str(args.backend),
            "quad_order": int(args.quad_order),
            "bossak_alpha": float(args.bossak_alpha),
            "dynamic_tau": float(args.dynamic_tau),
            "reference_velocity": float(reference_velocity),
            "contribution_modes": contribution_modes,
            "solve_max_iterations": int(args.solve_max_iterations),
            "solve_residual_tol": float(args.solve_residual_tol),
            "solve_line_search": bool(args.solve_line_search),
        },
        "entries": entries,
    }
    dump_json(output, args.output)


if __name__ == "__main__":
    main()
