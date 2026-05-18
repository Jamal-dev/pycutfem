import numpy as np

from pycutfem.mor.regime_atlas import (
    LocalReducedModelBankEntry,
    build_regime_bank_manifest,
    load_regime_bank_manifest,
    select_local_reduced_model_bank,
)


def test_bank_selection_reports_feature_distance_not_priority() -> None:
    entry = LocalReducedModelBankEntry(
        model_id="near",
        path="near.npz",
        priority=7,
        feature_center=np.zeros(1),
        feature_scale=np.ones(1),
        max_feature_distance=1.0,
    )

    selection = select_local_reduced_model_bank([entry], step=1, feature=np.array([0.25]))

    assert selection.selected
    assert selection.reason == "feature_priority"
    assert selection.distance == 0.25


def test_regime_bank_manifest_v2_round_trips_certificates_and_fallback(tmp_path) -> None:
    manifest = build_regime_bank_manifest(
        [
            LocalReducedModelBankEntry(
                model_id="region_000",
                path="region_000.npz",
                feature_center=np.array([0.0]),
                feature_scale=np.array([1.0]),
                max_feature_distance=0.5,
            )
        ],
        certificates={"region_000": {"max_error": 0.01}},
        fallback_policy={"kind": "fom"},
        metadata={"case": "toy"},
    )
    path = tmp_path / "banks.json"

    manifest.save(path)
    loaded = load_regime_bank_manifest(path)

    assert loaded.schema_version == 2
    assert loaded.entries[0].model_id == "region_000"
    assert loaded.entries[0].path.name == "region_000.npz"
    assert loaded.certificates["region_000"]["max_error"] == 0.01
    assert loaded.fallback_policy["kind"] == "fom"
    assert loaded.metadata["case"] == "toy"
