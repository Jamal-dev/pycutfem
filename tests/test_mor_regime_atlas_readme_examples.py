import numpy as np

from pycutfem.mor.regime_atlas import (
    EpsilonCoverPartitioner,
    KMedoidsPartitioner,
    RegimeDataset,
    RegimeOnlineSelector,
    ResidualGreedyConfig,
    ResidualGreedySplitter,
    make_validation_split,
)


def test_readme_style_feature_partition_and_online_selection_example_runs() -> None:
    features = np.vstack([np.linspace(-1.0, -0.8, 10), np.linspace(0.8, 1.0, 10)]).reshape(-1, 1)

    atlas = EpsilonCoverPartitioner(epsilon=0.15, max_regions=4).fit(features)
    decision = RegimeOnlineSelector(atlas=atlas).select(feature=np.array([-0.9]))

    assert atlas.n_regions == 2
    assert decision.selected


def test_readme_style_grouped_validation_and_residual_greedy_example_runs() -> None:
    dataset = RegimeDataset(
        features=np.concatenate([np.linspace(-1.0, -0.1, 12), np.linspace(0.1, 1.0, 12)]).reshape(-1, 1),
        groups=np.repeat(["left", "right"], 12),
    )
    split = make_validation_split(dataset, test_fraction=0.5, random_state=1)

    def train_model(indices: np.ndarray, dataset: RegimeDataset) -> object:
        return None

    def evaluate_model(model: object, indices: np.ndarray, dataset: RegimeDataset) -> float:
        values = dataset.features[indices, 0]
        return 0.5 if np.any(values < 0.0) and np.any(values > 0.0) else 0.0

    result = ResidualGreedySplitter(
        config=ResidualGreedyConfig(max_regions=2, min_support=6, validation_tolerance=0.1)
    ).fit(dataset, train_model=train_model, evaluate_model=evaluate_model)

    assert split.group_held_out
    assert result.atlas.n_regions == 2
    assert KMedoidsPartitioner(n_regions=2).fit(dataset).n_regions == 2
