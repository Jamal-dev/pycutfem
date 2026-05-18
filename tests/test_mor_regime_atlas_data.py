import numpy as np

from pycutfem.mor.regime_atlas import RegimeDataset, make_validation_split


def test_regime_dataset_validates_and_subsets_features() -> None:
    dataset = RegimeDataset(
        features=np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]),
        feature_names=("a", "b"),
        groups=np.array(["g0", "g0", "g1"]),
        steps=np.array([1, 2, 3]),
    )

    subset = dataset.subset([0, 2])

    assert dataset.n_samples == 3
    assert dataset.n_features == 2
    assert subset.n_samples == 2
    assert subset.feature_names == ("a", "b")
    assert subset.groups.tolist() == ["g0", "g1"]


def test_grouped_validation_split_holds_out_whole_groups() -> None:
    dataset = RegimeDataset(
        features=np.arange(12, dtype=float).reshape(6, 2),
        groups=np.array(["a", "a", "b", "b", "c", "c"]),
    )

    split = make_validation_split(dataset, test_fraction=1.0 / 3.0, random_state=0)
    train_groups = set(dataset.groups[split.train_indices].tolist())
    validation_groups = set(dataset.groups[split.validation_indices].tolist())

    assert split.group_held_out
    assert train_groups.isdisjoint(validation_groups)
