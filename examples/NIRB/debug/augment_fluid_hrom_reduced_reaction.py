from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from examples.NIRB.common import dump_json
from examples.NIRB.debug.replay_fluid_gnat_from_checkpoints import _make_operator
from examples.NIRB.debug.replay_fluid_rom_from_stage_probes import _configure_and_restore_probe, _parse_steps
from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.fluid_lspg import pack_fluid_state
from examples.NIRB.fluid_stage_probes import find_fluid_stage_probe_pairs, load_fluid_stage_probe
from examples.NIRB.run_example2_local import (
    _boundary_field_data,
    _build_fluid_problem,
    _load_reference_partitioned_meshes,
    _resample_lookup_to_coords,
)


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float).reshape(-1)
    cand = np.asarray(candidate, dtype=float).reshape(-1)
    return float(np.linalg.norm(cand - ref) / max(float(np.linalg.norm(ref)), 1.0e-15))


def _fit_affine_map(coefficients: np.ndarray, values: np.ndarray, *, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(coefficients, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or int(x.shape[0]) != int(y.shape[0]):
        raise ValueError("reaction fit expects X=(n_samples,n_modes), Y=(n_samples,n_outputs).")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add an affine reduced interface-reaction operator to a schema-1 sampled-LSPG fluid HROM model.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--steps", type=str, default=None)
    parser.add_argument("--training-all-iters", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=1.0e-10)
    parser.add_argument("--backend", choices=("python", "cpp"), default="cpp")
    parser.add_argument("--quad-order", type=int, default=3)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    parser.add_argument("--dynamic-tau", type=float, default=1.0)
    parser.add_argument("--reference-velocity", type=float, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.model, allow_pickle=False) as raw:
        model_payload = {key: np.asarray(raw[key]) for key in raw.files}
    basis = np.asarray(model_payload["basis"], dtype=float)
    free_dofs = np.asarray(model_payload["free_dofs"], dtype=int).reshape(-1)
    if basis.ndim != 2:
        raise ValueError("model basis must be a 2-D array.")
    if free_dofs.size and (np.any(free_dofs < 0) or np.any(free_dofs >= int(basis.shape[0]))):
        raise ValueError("model free_dofs are out of range.")

    pairs = find_fluid_stage_probe_pairs(
        args.probe_dir,
        final_only=not bool(args.training_all_iters),
        steps=_parse_steps(args.steps),
    )
    if args.max_pairs is not None:
        pairs = pairs[: max(0, int(args.max_pairs))]
    if not pairs:
        raise ValueError(f"No fluid stage probe pairs found under {args.probe_dir}.")

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

    coefficient_rows: list[np.ndarray] = []
    reaction_rows: list[np.ndarray] = []
    reaction_coords: np.ndarray | None = None
    entry_rows: list[dict[str, Any]] = []

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
        post_state = pack_fluid_state(operator)
        reaction_lookup = post_probe.reaction_lookup(kind="point")
        if reaction_lookup is None:
            reaction_lookup = operator.reaction_loads(refresh_state=False)
        if reaction_coords is None:
            reaction_coords = np.asarray(reaction_lookup.coords, dtype=float)
            reaction_values = np.asarray(reaction_lookup.values, dtype=float)
        else:
            reaction_values = np.asarray(
                _resample_lookup_to_coords(reaction_lookup, reaction_coords).values,
                dtype=float,
            )

        _configure_and_restore_probe(
            operator=operator,
            probe=pre_probe,
            setup=setup,
            fluid_iface_coords=fluid_iface_coords,
            dt=float(dt),
            reference_velocity=float(reference_velocity),
        )
        pre_state = pack_fluid_state(operator)
        rhs = np.asarray(post_state - pre_state, dtype=float).reshape(-1)
        coeffs, *_ = np.linalg.lstsq(basis[free_dofs, :], rhs[free_dofs], rcond=None)
        coefficient_rows.append(np.asarray(coeffs, dtype=float).reshape(-1))
        reaction_rows.append(np.asarray(reaction_values, dtype=float).reshape(-1))
        entry_rows.append(
            {
                "step": int(pair.step),
                "coupling_iter": int(pair.coupling_iter),
                "coefficient_norm": float(np.linalg.norm(coeffs)),
                "reaction_norm": float(np.linalg.norm(reaction_values)),
            }
        )

    if reaction_coords is None:
        raise RuntimeError("No reaction coordinates were collected.")
    x = np.vstack(coefficient_rows)
    y = np.vstack(reaction_rows)
    matrix, bias = _fit_affine_map(x, y, ridge=float(args.ridge))
    fitted = bias.reshape(1, -1) + x @ matrix.T
    per_sample_error = [
        _relative_l2(y[idx, :], fitted[idx, :])
        for idx in range(int(y.shape[0]))
    ]

    output_payload = dict(model_payload)
    output_payload.update(
        {
            "reaction_matrix": np.asarray(matrix, dtype=float),
            "reaction_bias": np.asarray(bias, dtype=float),
            "reaction_coords": np.asarray(reaction_coords, dtype=float),
            "reaction_kind": np.asarray("point"),
            "reaction_fit_steps": np.asarray([int(pair.step) for pair in pairs], dtype=int),
            "reaction_fit_coupling_iters": np.asarray([int(pair.coupling_iter) for pair in pairs], dtype=int),
            "reaction_fit_ridge": np.asarray(float(args.ridge), dtype=float),
            "reaction_fit_max_relative_error": np.asarray(float(max(per_sample_error, default=0.0)), dtype=float),
            "reaction_fit_mean_relative_error": np.asarray(float(np.mean(per_sample_error)), dtype=float),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output_payload)

    summary = {
        "model": str(args.model),
        "output": str(args.output),
        "probe_dir": str(args.probe_dir),
        "pairs": len(pairs),
        "n_modes": int(basis.shape[1]),
        "reaction_size": int(y.shape[1]),
        "ridge": float(args.ridge),
        "max_relative_error": float(max(per_sample_error, default=0.0)),
        "mean_relative_error": float(np.mean(per_sample_error)),
        "entries": entry_rows,
    }
    if args.summary is not None:
        dump_json(summary, args.summary)
    else:
        print(summary)


if __name__ == "__main__":
    main()
