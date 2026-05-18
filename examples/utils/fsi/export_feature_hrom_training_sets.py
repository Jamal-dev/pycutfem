#!/usr/bin/env python3
"""Export exact training step sets from a feature-HROM atlas database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _compact_integer_ranges(values: list[int]) -> str:
    unique = sorted({int(item) for item in values})
    if not unique:
        return ""
    ranges: list[str] = []
    start = unique[0]
    previous = unique[0]
    for value in unique[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}:{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}:{previous}")
    return ",".join(ranges)


def _region_by_index(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    regions = summary.get("selected_atlas", {}).get("regions", [])
    result: dict[int, dict[str, Any]] = {}
    for region in regions:
        result[int(region["index"])] = dict(region)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-database", type=Path, required=True)
    parser.add_argument("--atlas-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=75)
    args = parser.parse_args()

    with np.load(args.atlas_database, allow_pickle=True) as data:
        steps = np.asarray(data["steps"], dtype=int).reshape(-1)
        labels = np.asarray(data["labels"], dtype=int).reshape(-1)
        features = np.asarray(data["features"], dtype=float)
        feature_names = [str(item) for item in np.asarray(data["feature_names"], dtype=object).reshape(-1)]
    if steps.shape[0] != labels.shape[0] or features.shape[0] != steps.shape[0]:
        raise ValueError("Atlas database arrays must have the same row count.")
    try:
        coupling_iter_feature_index = feature_names.index("coupling_iter")
    except ValueError as exc:
        raise ValueError("Atlas database must contain a 'coupling_iter' feature.") from exc

    summary = json.loads(args.atlas_summary.read_text(encoding="utf-8"))
    regions = _region_by_index(summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    for region_index in sorted(np.unique(labels).astype(int).tolist()):
        mask = labels == int(region_index)
        region_steps = sorted({int(value) for value in steps[mask]})
        stage_pairs = sorted(
            {
                (int(step_value), int(round(coupling_iter_value)))
                for step_value, coupling_iter_value in zip(
                    steps[mask],
                    features[mask, coupling_iter_feature_index],
                    strict=True,
                )
            }
        )
        coupling_iters = sorted({int(item[1]) for item in stage_pairs})
        region = regions.get(int(region_index), {})
        support = int(np.count_nonzero(mask))
        region_id = str(region.get("id", f"region_{region_index:03d}"))
        bankable = bool(support >= int(args.min_support))
        step_spec = _compact_integer_ranges(region_steps)
        stage_pair_path = output_dir / f"{region_id}_stage_pairs.csv"
        stage_pair_path.write_text(
            "step,coupling_iter\n"
            + "\n".join(f"{int(step_value)},{int(iter_value)}" for step_value, iter_value in stage_pairs)
            + "\n",
            encoding="utf-8",
        )
        item = {
            "id": region_id,
            "index": int(region_index),
            "bankable": bankable,
            "support_count": support,
            "unique_step_count": len(region_steps),
            "step_start": int(min(region_steps)) if region_steps else None,
            "step_end": int(max(region_steps)) if region_steps else None,
            "steps": region_steps,
            "step_spec": step_spec,
            "stage_pairs_file": str(stage_pair_path),
            "stage_pair_count": len(stage_pairs),
            "coupling_iters": coupling_iters,
            "coupling_iter_spec": _compact_integer_ranges(coupling_iters),
            "max_feature_distance": float(region.get("max_feature_distance", float("nan"))),
        }
        exported.append(item)
        (output_dir / f"{region_id}_steps.txt").write_text(step_spec + "\n", encoding="utf-8")
        (output_dir / f"{region_id}_coupling_iters.txt").write_text(
            item["coupling_iter_spec"] + "\n",
            encoding="utf-8",
        )

    payload = {
        "schema_version": 1,
        "atlas_database": str(args.atlas_database),
        "atlas_summary": str(args.atlas_summary),
        "min_support": int(args.min_support),
        "regions": exported,
    }
    (output_dir / "feature_hrom_training_sets.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_dir / 'feature_hrom_training_sets.json'}")


if __name__ == "__main__":
    main()
