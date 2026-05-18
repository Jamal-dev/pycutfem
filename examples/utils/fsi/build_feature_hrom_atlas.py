#!/usr/bin/env python3
"""Build a feature-based local HROM atlas from coupled FSI stage histories.

The script is intentionally example-facing: it knows about common column names
written by the NIRB/FSI ``timeseries.csv`` files, but all clustering and atlas
logic lives in ``pycutfem.mor.feature_atlas``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from pycutfem.mor import feature_atlas_to_bank_manifest, select_feature_atlas_size


DEFAULT_LOG_FEATURES = (
    "disp_abs",
    "disp_rel",
    "load_abs",
    "load_rel",
)

DEFAULT_LINEAR_FEATURES = (
    "coupling_iter",
    "disp_max",
    "load_guess_max",
    "load_return_max",
    "solid_rom_full_residual_rel",
    "solid_rom_interface_disp_rel",
)


def _parse_float(value: str | None, *, default: float = float("nan")) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        result = float(text)
    except ValueError:
        return default
    return result if math.isfinite(result) else default


def _log10_feature(value: float, *, floor: float) -> float:
    if not math.isfinite(value):
        value = 0.0
    return float(math.log10(max(abs(value), floor)))


def _read_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(dict(row) for row in reader)
    if not rows:
        raise ValueError("no rows found in the provided timeseries files")
    return rows


def _available_columns(rows: list[dict[str, str]], requested: Iterable[str]) -> list[str]:
    available = set().union(*(row.keys() for row in rows))
    return [name for name in requested if name in available]


def build_feature_matrix(
    rows: list[dict[str, str]],
    *,
    log_columns: Iterable[str],
    linear_columns: Iterable[str],
    log_floor: float,
    include_step_feature: bool,
    include_time_feature: bool,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    log_names = _available_columns(rows, log_columns)
    linear_names = _available_columns(rows, linear_columns)
    names: list[str] = []
    if include_step_feature:
        names.append("step")
    if include_time_feature and "time_s" in set().union(*(row.keys() for row in rows)):
        names.append("time_s")
    names.extend(f"log10_{name}" for name in log_names)
    names.extend(linear_names)

    features: list[list[float]] = []
    steps: list[int] = []
    for row in rows:
        step = int(round(_parse_float(row.get("step"), default=0.0)))
        if step < 1:
            continue
        values: list[float] = []
        if include_step_feature:
            values.append(float(step))
        if include_time_feature and "time_s" in row:
            values.append(_parse_float(row.get("time_s"), default=0.0))
        for name in log_names:
            values.append(_log10_feature(_parse_float(row.get(name), default=0.0), floor=log_floor))
        for name in linear_names:
            values.append(_parse_float(row.get(name), default=0.0))
        if all(math.isfinite(value) for value in values):
            features.append(values)
            steps.append(step)
    if not features:
        raise ValueError("no finite feature rows could be built")
    return np.asarray(features, dtype=float), np.asarray(steps, dtype=int), tuple(names)


def _compact_selection_summary(selection) -> dict[str, object]:
    candidates = []
    for atlas, diagnostics in selection.candidates:
        candidates.append(
            {
                "k": int(atlas.n_regions),
                "passed": bool(diagnostics.passed),
                "coverage": float(diagnostics.coverage),
                "bankable_coverage": float(diagnostics.bankable_coverage),
                "silhouette": float(atlas.silhouette),
                "min_support": int(diagnostics.min_support),
                "max_radius": float(diagnostics.max_radius),
                "reasons": list(diagnostics.reasons),
            }
        )
    return {
        "selected_k": int(selection.selected.n_regions),
        "selected_passed": bool(selection.diagnostics.passed),
        "selected_coverage": float(selection.diagnostics.coverage),
        "selected_bankable_coverage": float(selection.diagnostics.bankable_coverage),
        "selected_min_support": int(selection.diagnostics.min_support),
        "selected_max_radius": float(selection.diagnostics.max_radius),
        "selected_reasons": list(selection.diagnostics.reasons),
        "candidates": candidates,
    }


def _write_training_plan(
    path: Path,
    *,
    selection,
    manifest_path: Path,
    feature_names: tuple[str, ...],
    timeseries_paths: list[Path],
) -> None:
    atlas = selection.selected
    lines = [
        "# Feature-Based HROM Atlas Training Plan",
        "",
        "This plan is generated from coupled FSI stage features.  The regions are",
        "feature regimes, not hand-picked time intervals.",
        "",
        "## Inputs",
        "",
        "Timeseries files:",
        "",
    ]
    lines.extend(f"- `{item}`" for item in timeseries_paths)
    lines.extend(
        [
            "",
            "Features:",
            "",
        ]
    )
    lines.extend(f"- `{name}`" for name in feature_names)
    lines.extend(
        [
            "",
            "## Selected Atlas",
            "",
            f"- selected feature regions: `{atlas.n_regions}`",
            f"- training coverage: `{atlas.coverage:.6g}`",
            f"- bankable coverage: `{selection.diagnostics.bankable_coverage:.6g}`",
            f"- silhouette: `{atlas.silhouette:.6g}`",
            f"- max feature radius: `{atlas.max_region_radius:.6g}`",
            f"- manifest template: `{manifest_path}`",
            "",
            "## Region Milestones",
            "",
        ]
    )
    for region in atlas.regions:
        blocked = []
        if region.region_id in selection.diagnostics.overfit_region_ids:
            blocked.append("support below min_support")
        if region.region_id in selection.diagnostics.underfit_region_ids:
            blocked.append("radius above max_radius")
        promotion_status = "bankable" if not blocked else "FOM/enrichment outlier: " + ", ".join(blocked)
        lines.extend(
            [
                f"### `{region.region_id}`",
                "",
                f"- promotion status: `{promotion_status}`",
                f"- support stages: `{region.support_count}`",
                f"- step span: `{region.step_start}` to `{region.step_end}`",
                f"- medoid row index: `{region.medoid_index}`",
                f"- radius: `{region.max_feature_distance:.6g}`",
                f"- mean distance: `{region.mean_distance:.6g}`",
                "",
                "Milestone checklist:",
                "",
                "- [ ] collect all-state probes for this feature region plus boundary halo",
                "- [ ] train local trial basis with homogeneous lifting and supremizer enrichment",
                "- [ ] train residual/JV cubature and sampled reaction operator",
                "- [ ] validate held-out states inside the region",
                "- [ ] validate boundary/halo states against neighboring regions",
                "- [ ] add trained model path to the manifest only if certified and faster",
                "",
            ]
        )
    lines.extend(
        [
            "## Acceptance Rule",
            "",
            "A region should be promoted only when:",
            "",
            "```text",
            "held-out eta_Gamma passes",
            "and contraction does not worsen",
            "and coupling iterations do not increase",
            "and accepted HROM stage cost is below exact stage cost",
            "and region support is not too small",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeseries", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--min-support", type=int, default=50)
    parser.add_argument("--target-coverage", type=float, default=0.98)
    parser.add_argument("--max-radius", type=float, default=3.0)
    parser.add_argument("--radius-quantile", type=float, default=0.98)
    parser.add_argument("--radius-safety-factor", type=float, default=1.10)
    parser.add_argument("--log-floor", type=float, default=1.0e-12)
    parser.add_argument(
        "--no-default-features",
        action="store_true",
        help=(
            "Use only explicitly provided --log-feature/--linear-feature values. "
            "This is useful for deployable online atlases, where post-stage columns "
            "such as load_return_max are not available before model selection."
        ),
    )
    parser.add_argument("--log-feature", action="append", default=None)
    parser.add_argument("--linear-feature", action="append", default=None)
    parser.add_argument("--include-step-feature", action="store_true")
    parser.add_argument("--include-time-feature", action="store_true")
    parser.add_argument(
        "--model-path-template",
        default="fluid_hrom_{region_id}.npz",
        help="Template written to the bank manifest. Available fields: region_id, region_index, support_count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.k_min) < 1 or int(args.k_max) < int(args.k_min):
        raise ValueError("--k-min/--k-max define an invalid range")
    log_features = list(args.log_feature or [])
    linear_features = list(args.linear_feature or [])
    if not bool(args.no_default_features):
        log_features = list(DEFAULT_LOG_FEATURES) + log_features
        linear_features = list(DEFAULT_LINEAR_FEATURES) + linear_features
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.timeseries)
    features, steps, feature_names = build_feature_matrix(
        rows,
        log_columns=log_features,
        linear_columns=linear_features,
        log_floor=float(args.log_floor),
        include_step_feature=bool(args.include_step_feature),
        include_time_feature=bool(args.include_time_feature),
    )
    selection = select_feature_atlas_size(
        features,
        k_values=range(int(args.k_min), int(args.k_max) + 1),
        feature_names=feature_names,
        steps=steps,
        min_support=int(args.min_support),
        target_coverage=float(args.target_coverage),
        max_radius=float(args.max_radius),
        radius_quantile=float(args.radius_quantile),
        radius_safety_factor=float(args.radius_safety_factor),
    )
    manifest = feature_atlas_to_bank_manifest(
        selection.selected,
        model_path_template=str(args.model_path_template),
        description="Feature-based local HROM atlas manifest template.",
        min_support=int(args.min_support),
        max_radius=float(args.max_radius),
        extra_metadata={
            "target_coverage": float(args.target_coverage),
            "min_support": int(args.min_support),
            "max_radius": float(args.max_radius),
        },
    )
    database_path = args.output_dir / "feature_atlas_database.npz"
    summary_path = args.output_dir / "feature_atlas_summary.json"
    manifest_path = args.output_dir / "feature_hrom_bank_manifest_template.json"
    plan_path = args.output_dir / "feature_hrom_training_plan.md"

    np.savez(
        database_path,
        features=features,
        steps=steps,
        labels=selection.selected.labels,
        feature_names=np.asarray(feature_names, dtype=object),
    )
    summary = _compact_selection_summary(selection)
    summary["selected_atlas"] = selection.selected.to_dict()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_training_plan(
        plan_path,
        selection=selection,
        manifest_path=manifest_path,
        feature_names=feature_names,
        timeseries_paths=[Path(path) for path in args.timeseries],
    )

    print(f"features: {features.shape[0]} stages x {features.shape[1]} features")
    print(f"selected regions: {selection.selected.n_regions}")
    print(f"bankable regions: {len(manifest['banks'])}")
    print(f"coverage: {selection.diagnostics.coverage:.6g}")
    print(f"bankable coverage: {selection.diagnostics.bankable_coverage:.6g}")
    print(f"min support: {selection.diagnostics.min_support}")
    print(f"max radius: {selection.diagnostics.max_radius:.6g}")
    print(f"summary: {summary_path}")
    print(f"manifest template: {manifest_path}")
    print(f"training plan: {plan_path}")


if __name__ == "__main__":
    main()
