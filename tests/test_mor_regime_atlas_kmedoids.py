import numpy as np

from pycutfem.mor.regime_atlas import KMedoidsPartitioner, make_regime_partitioner


def test_kmedoids_partitioner_splits_two_compact_regimes() -> None:
    features = np.vstack(
        [
            np.column_stack([np.linspace(-1.0, -0.8, 12), np.zeros(12)]),
            np.column_stack([np.linspace(0.8, 1.0, 12), np.zeros(12)]),
        ]
    )

    atlas = KMedoidsPartitioner(n_regions=2).fit(features)
    factory_atlas = make_regime_partitioner({"kind": "kmedoids", "n_regions": 2}).fit(features)

    assert atlas.n_regions == 2
    assert sorted(atlas.support_counts.tolist()) == [12, 12]
    assert factory_atlas.n_regions == 2
