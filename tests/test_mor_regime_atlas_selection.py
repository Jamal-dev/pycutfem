import numpy as np

from pycutfem.mor.regime_atlas import (
    KMedoidsPartitioner,
    RegimeDataset,
    RegimeAtlasCandidate,
    RegimeAtlasSelector,
    boundary_halo_score,
    summarize_region_errors,
)


def test_regime_atlas_selector_uses_validation_and_complexity_penalties() -> None:
    features = np.vstack(
        [
            np.linspace(-1.0, -0.8, 8),
            np.linspace(0.8, 1.0, 8),
        ]
    ).reshape(-1, 1)
    coarse = KMedoidsPartitioner(n_regions=1).fit(features)
    local = KMedoidsPartitioner(n_regions=2).fit(features)
    candidates = (
        RegimeAtlasCandidate(coarse, summarize_region_errors(coarse, [0.25], tolerance=0.1)),
        RegimeAtlasCandidate(local, summarize_region_errors(local, [0.03, 0.04], tolerance=0.1), complexity=1.0),
    )

    selection = RegimeAtlasSelector(max_validation_error=0.1, complexity_weight=0.01).select(candidates)

    assert selection.selected is candidates[1]
    assert selection.to_dict()["selected_index"] == 1


def test_boundary_halo_score_reports_positive_margin_for_separated_regions() -> None:
    features = np.vstack(
        [
            np.linspace(-1.0, -0.8, 8),
            np.linspace(0.8, 1.0, 8),
        ]
    ).reshape(-1, 1)
    dataset = RegimeDataset(features=features)
    atlas = KMedoidsPartitioner(n_regions=2).fit(dataset)

    assert boundary_halo_score(atlas, dataset) > 0.0
