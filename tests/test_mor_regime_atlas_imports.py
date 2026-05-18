import importlib.util

import numpy as np

from pycutfem.mor import fit_feature_atlas as top_level_fit_feature_atlas
from pycutfem.mor import select_local_reduced_model_bank as top_level_select_bank
from pycutfem.mor.regime_atlas import (
    LocalReducedModelBankEntry,
    fit_feature_atlas,
    select_local_reduced_model_bank,
)


def test_regime_atlas_exports_current_feature_atlas_api() -> None:
    features = np.vstack(
        [
            np.column_stack([np.linspace(-1.0, -0.8, 8), np.zeros(8)]),
            np.column_stack([np.linspace(0.8, 1.0, 8), np.zeros(8)]),
        ]
    )

    atlas = fit_feature_atlas(features, n_regions=2, radius_quantile=1.0)

    assert top_level_fit_feature_atlas is fit_feature_atlas
    assert atlas.n_regions == 2
    assert sorted(atlas.support_counts.tolist()) == [8, 8]


def test_old_feature_atlas_and_local_bank_public_modules_are_removed() -> None:
    assert importlib.util.find_spec("pycutfem.mor.feature_atlas") is None
    assert importlib.util.find_spec("pycutfem.mor.local_banks") is None


def test_regime_atlas_exports_current_local_bank_api() -> None:
    banks = [
        LocalReducedModelBankEntry(
            model_id="near",
            path="near.npz",
            priority=1,
            feature_center=np.zeros(2),
            feature_scale=np.ones(2),
            max_feature_distance=0.5,
        )
    ]

    selection = select_local_reduced_model_bank(banks, step=1, feature=np.array([0.1, 0.0]))

    assert top_level_select_bank is select_local_reduced_model_bank
    assert selection.selected
    assert selection.model_id == "near"
