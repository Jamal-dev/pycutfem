import numpy as np

from pycutfem.mor.regime_atlas import (
    KMedoidsPartitioner,
    LocalReducedModelBankEntry,
    RegimeOnlineSelector,
)


def test_online_selector_accepts_inside_radius_and_rejects_outside_radius() -> None:
    features = np.vstack([np.linspace(-1.0, -0.8, 8), np.linspace(0.8, 1.0, 8)]).reshape(-1, 1)
    atlas = KMedoidsPartitioner(n_regions=2, radius_quantile=1.0).fit(features)
    selector = RegimeOnlineSelector(atlas=atlas, fallback_policy={"kind": "fom"})

    inside = selector.select(feature=np.array([-0.9]))
    outside = selector.select(feature=np.array([10.0]))

    assert inside.selected
    assert inside.reason == "inside_certified_region"
    assert outside.reason == "outside_certified_region"
    assert outside.metadata["fallback_policy"]["kind"] == "fom"


def test_online_selector_uses_bank_entries_when_manifest_entries_are_present() -> None:
    entry = LocalReducedModelBankEntry(
        model_id="local",
        path="local.npz",
        feature_center=np.zeros(1),
        feature_scale=np.ones(1),
        max_feature_distance=0.5,
    )
    selector = RegimeOnlineSelector(entries=(entry,), fallback_policy={"kind": "global_rom"})

    accepted = selector.select(feature=np.array([0.25]))
    rejected = selector.select(feature=np.array([2.0]))

    assert accepted.selected
    assert accepted.model_id == "local"
    assert accepted.distance == 0.25
    assert rejected.reason == "no_active_feature_radius"
    assert rejected.metadata["fallback_policy"]["kind"] == "global_rom"
