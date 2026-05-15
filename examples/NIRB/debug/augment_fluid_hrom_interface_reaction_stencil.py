from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from examples.NIRB.common import dump_json
from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.run_example2_local import (
    _build_fluid_problem,
    _fluid_interface_reaction_element_ids,
    _load_reference_partitioned_meshes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append all interface-reaction elements to a sampled-LSPG fluid HROM "
            "stencil with zero cubature weight. This lets the online load-only "
            "path compute interface reactions from locally decoded element fields "
            "without changing the sampled reduced residual equation."
        ),
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interface-tag", type=str, default=None)
    parser.add_argument("--quad-order", type=int, default=1)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.model, allow_pickle=False) as raw:
        payload = {key: np.asarray(raw[key]) for key in raw.files}

    sample_element_ids = np.asarray(payload["sample_element_ids"], dtype=int).reshape(-1)
    if "sample_element_weights" in payload:
        sample_element_weights = np.asarray(payload["sample_element_weights"], dtype=float).reshape(-1)
    else:
        sample_element_weights = np.ones(int(sample_element_ids.size), dtype=float)
    if int(sample_element_weights.size) != int(sample_element_ids.size):
        raise ValueError("sample_element_weights must match sample_element_ids before augmentation.")

    setup = load_example2_local_setup()
    mesh_f, _mesh_s = _load_reference_partitioned_meshes(setup=setup)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=int(args.quad_order))
    interface_tag = str(args.interface_tag or setup.geometry.interface_tag)
    reaction_element_ids = np.asarray(
        _fluid_interface_reaction_element_ids(fluid, interface_tag=interface_tag),
        dtype=int,
    ).reshape(-1)
    missing = np.setdiff1d(reaction_element_ids, sample_element_ids, assume_unique=False)
    if missing.size:
        sample_element_ids_aug = np.concatenate([sample_element_ids, missing.astype(int, copy=False)])
        sample_element_weights_aug = np.concatenate(
            [sample_element_weights, np.zeros(int(missing.size), dtype=float)]
        )
    else:
        sample_element_ids_aug = sample_element_ids.copy()
        sample_element_weights_aug = sample_element_weights.copy()

    payload["sample_element_ids"] = np.asarray(sample_element_ids_aug, dtype=int)
    payload["sample_element_weights"] = np.asarray(sample_element_weights_aug, dtype=float)
    payload["interface_reaction_stencil_element_ids"] = np.asarray(reaction_element_ids, dtype=int)
    payload["interface_reaction_stencil_added_element_ids"] = np.asarray(missing, dtype=int)
    payload["interface_reaction_stencil_interface_tag"] = np.asarray(interface_tag)
    payload["interface_reaction_stencil_zero_weight"] = np.asarray(True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)

    summary = {
        "model": str(args.model),
        "output": str(args.output),
        "interface_tag": interface_tag,
        "original_sample_elements": int(sample_element_ids.size),
        "original_active_elements": int(np.count_nonzero(sample_element_weights > 0.0)),
        "interface_reaction_elements": int(reaction_element_ids.size),
        "added_interface_reaction_elements": int(missing.size),
        "augmented_sample_elements": int(sample_element_ids_aug.size),
        "augmented_active_elements": int(np.count_nonzero(sample_element_weights_aug > 0.0)),
        "element_weight_sum_before": float(np.sum(sample_element_weights)),
        "element_weight_sum_after": float(np.sum(sample_element_weights_aug)),
    }
    if args.summary is not None:
        dump_json(summary, args.summary)
    else:
        print(summary)


if __name__ == "__main__":
    main()
