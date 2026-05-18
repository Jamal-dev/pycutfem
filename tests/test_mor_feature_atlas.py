from __future__ import annotations

import json

import numpy as np

from pycutfem.mor import (
    diagnose_feature_atlas,
    feature_atlas_to_bank_manifest,
    fit_feature_atlas,
    load_local_reduced_model_bank_manifest,
    select_feature_atlas_size,
    select_local_reduced_model_bank,
    subspace_chordal_distance,
    subspace_principal_angles,
)


def _two_regime_features() -> tuple[np.ndarray, np.ndarray]:
    left = np.column_stack(
        [
            np.linspace(-1.2, -0.8, 40),
            np.linspace(-0.1, 0.1, 40),
        ]
    )
    right = np.column_stack(
        [
            np.linspace(0.8, 1.2, 40),
            np.linspace(-0.1, 0.1, 40),
        ]
    )
    features = np.vstack([left, right])
    steps = np.arange(1, features.shape[0] + 1)
    return features, steps


def test_feature_atlas_fits_two_regimes_and_generates_manifest(tmp_path) -> None:
    features, steps = _two_regime_features()

    atlas = fit_feature_atlas(
        features,
        n_regions=2,
        feature_names=("load_coeff", "disp_coeff"),
        steps=steps,
        radius_quantile=1.0,
        radius_safety_factor=1.05,
    )

    assert atlas.n_regions == 2
    assert atlas.coverage == 1.0
    assert sorted(atlas.support_counts.tolist()) == [40, 40]
    diagnostics = diagnose_feature_atlas(atlas, min_support=20, target_coverage=1.0, max_radius=2.0)
    assert diagnostics.passed

    manifest = feature_atlas_to_bank_manifest(
        atlas,
        model_path_template="bank_{region_index:03d}.npz",
        description="unit-test atlas",
    )
    manifest_path = tmp_path / "banks.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    banks = load_local_reduced_model_bank_manifest(manifest_path)
    assert len(banks) == 2
    selected = select_local_reduced_model_bank(banks, step=20, feature=np.array([-1.0, 0.0]))

    assert selected.selected
    assert selected.entry is not None
    assert selected.entry.metadata["atlas_support_count"] == 40
    assert selected.distance <= selected.entry.max_feature_distance


def test_feature_atlas_size_selection_prefers_smallest_passing_k() -> None:
    features, steps = _two_regime_features()

    selection = select_feature_atlas_size(
        features,
        k_values=range(1, 5),
        steps=steps,
        min_support=30,
        target_coverage=1.0,
        max_radius=1.1,
        radius_quantile=1.0,
        radius_safety_factor=1.05,
    )

    assert selection.selected.n_regions == 2
    assert selection.diagnostics.passed


def test_feature_atlas_diagnostics_flags_overfit_and_underfit() -> None:
    features = np.vstack(
        [
            np.column_stack([np.linspace(-1.0, -0.8, 5), np.zeros(5)]),
            np.column_stack([np.linspace(0.8, 1.0, 40), np.zeros(40)]),
        ]
    )

    atlas = fit_feature_atlas(features, n_regions=2, radius_quantile=1.0, radius_safety_factor=2.0)
    diagnostics = diagnose_feature_atlas(atlas, min_support=10, target_coverage=1.0, max_radius=0.1)

    assert not diagnostics.passed
    assert diagnostics.overfit_region_ids
    assert diagnostics.underfit_region_ids


def test_subspace_principal_angles_and_chordal_distance() -> None:
    basis_a = np.eye(3, 2)
    basis_b = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ]
    )

    angles = subspace_principal_angles(basis_a, basis_b, degrees=True)

    np.testing.assert_allclose(angles, np.array([0.0, 90.0]), atol=1.0e-10)
    np.testing.assert_allclose(subspace_chordal_distance(basis_a, basis_b), 1.0)
