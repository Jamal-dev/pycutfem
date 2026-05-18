import numpy as np

from pycutfem.mor.regime_atlas import HierarchicalPartitioner, make_regime_partitioner


def test_hierarchical_partitioner_finds_two_separated_regimes() -> None:
    features = np.vstack(
        [
            np.linspace(0.0, 0.05, 8),
            np.linspace(2.0, 2.05, 8),
        ]
    ).reshape(-1, 1)

    for linkage in ("single", "complete", "average"):
        atlas = HierarchicalPartitioner(n_regions=2, linkage=linkage).fit(features)

        assert atlas.n_regions == 2
        assert sorted(atlas.support_counts.tolist()) == [8, 8]


def test_hierarchical_factory_dispatches_strategy() -> None:
    features = np.array([[0.0], [0.1], [2.0], [2.1]])

    atlas = make_regime_partitioner({"kind": "hierarchical", "n_regions": 2}).fit(features)

    assert atlas.n_regions == 2
