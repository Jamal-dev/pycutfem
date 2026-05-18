import numpy as np

from pycutfem.mor.regime_atlas import DensityPartitioner, make_regime_partitioner


def test_density_partitioner_keeps_outliers_unassigned() -> None:
    features = np.concatenate(
        [
            np.linspace(0.0, 0.05, 6),
            np.linspace(1.0, 1.05, 6),
            np.array([4.0]),
        ]
    ).reshape(-1, 1)

    atlas = DensityPartitioner(eps=0.08, min_samples=3).fit(features)

    assert atlas.n_regions == 2
    assert sorted(atlas.support_counts.tolist()) == [6, 6]
    assert atlas.outlier_indices.tolist() == [12]
    assert atlas.coverage == 12 / 13


def test_density_factory_dispatches_strategy() -> None:
    features = np.concatenate([np.linspace(0.0, 0.05, 5), np.linspace(1.0, 1.05, 5)]).reshape(-1, 1)

    atlas = make_regime_partitioner({"kind": "density", "eps": 0.08, "min_samples": 3}).fit(features)

    assert atlas.n_regions == 2


def test_density_partitioner_handles_two_moon_style_nonconvex_regimes() -> None:
    theta = np.linspace(0.0, np.pi, 24)
    upper = np.column_stack([np.cos(theta), np.sin(theta)])
    lower = np.column_stack([1.0 - np.cos(theta), -np.sin(theta) - 0.4])
    features = np.vstack([upper, lower])

    atlas = DensityPartitioner(eps=0.18, min_samples=3).fit(features)

    assert atlas.n_regions == 2
    assert sorted(atlas.support_counts.tolist()) == [24, 24]
    assert atlas.coverage == 1.0
