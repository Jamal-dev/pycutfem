import numpy as np

from pycutfem.mor.regime_atlas import MixturePartitioner, make_regime_partitioner


def test_mixture_partitioner_separates_two_gaussian_like_regimes() -> None:
    features = np.vstack(
        [
            np.column_stack([np.linspace(-2.0, -1.8, 10), np.zeros(10)]),
            np.column_stack([np.linspace(1.8, 2.0, 10), np.zeros(10)]),
        ]
    )

    atlas = MixturePartitioner(n_components=2, min_probability=0.5).fit(features)

    assert atlas.n_regions == 2
    assert sorted(atlas.support_counts.tolist()) == [10, 10]
    assert atlas.coverage == 1.0
    assert atlas.metadata["partitioner"] == "mixture"


def test_mixture_factory_dispatches_strategy() -> None:
    features = np.array([[-1.0], [-0.9], [0.9], [1.0]])

    atlas = make_regime_partitioner({"kind": "gmm", "n_components": 2}).fit(features)

    assert atlas.n_regions == 2
