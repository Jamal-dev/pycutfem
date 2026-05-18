import numpy as np

from pycutfem.mor.regime_atlas import EpsilonCoverPartitioner


def test_epsilon_cover_adds_centers_until_radius_is_satisfied() -> None:
    features = np.array([[0.0], [0.05], [10.0], [10.05]])

    atlas = EpsilonCoverPartitioner(epsilon=0.2, max_regions=4).fit(features)

    assert atlas.n_regions == 2
    assert sorted(atlas.support_counts.tolist()) == [2, 2]
    for feature in features:
        assert atlas.region_for_feature(feature) is not None
