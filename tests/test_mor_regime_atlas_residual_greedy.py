import numpy as np

from pycutfem.mor.regime_atlas import RegimeDataset, ResidualGreedyConfig, ResidualGreedySplitter


def _signed_regime_dataset() -> RegimeDataset:
    return RegimeDataset(
        features=np.concatenate([np.linspace(-1.0, -0.1, 20), np.linspace(0.1, 1.0, 20)]).reshape(-1, 1)
    )


def test_residual_greedy_splits_only_the_failing_mixed_region() -> None:
    dataset = _signed_regime_dataset()

    def train_model(indices: np.ndarray, dataset: RegimeDataset) -> object:
        return {"count": int(indices.size)}

    def evaluate_model(model: object, indices: np.ndarray, dataset: RegimeDataset) -> float:
        values = dataset.features[indices, 0]
        mixed = np.any(values < 0.0) and np.any(values > 0.0)
        return 1.0 if mixed else 0.01

    result = ResidualGreedySplitter(
        config=ResidualGreedyConfig(
            max_regions=3,
            min_support=8,
            validation_tolerance=0.1,
            improvement_margin=0.01,
        )
    ).fit(dataset, train_model=train_model, evaluate_model=evaluate_model)

    assert result.atlas.n_regions == 2
    assert result.validation.passed
    assert result.accepted_splits == 1
    assert result.events[0].reason == "accepted"


def test_residual_greedy_rejects_split_without_validation_improvement() -> None:
    dataset = _signed_regime_dataset()

    def train_model(indices: np.ndarray, dataset: RegimeDataset) -> object:
        return None

    def evaluate_model(model: object, indices: np.ndarray, dataset: RegimeDataset) -> float:
        return 1.0

    result = ResidualGreedySplitter(
        config=ResidualGreedyConfig(
            max_regions=3,
            min_support=8,
            validation_tolerance=0.1,
            improvement_margin=0.01,
        )
    ).fit(dataset, train_model=train_model, evaluate_model=evaluate_model)

    assert result.atlas.n_regions == 1
    assert not result.validation.passed
    assert result.rejected_splits == 1
    assert result.events[0].reason == "insufficient_validation_improvement"


def test_residual_greedy_keeps_one_region_when_global_model_passes() -> None:
    dataset = _signed_regime_dataset()

    def train_model(indices: np.ndarray, dataset: RegimeDataset) -> object:
        return None

    def evaluate_model(model: object, indices: np.ndarray, dataset: RegimeDataset) -> float:
        return 0.0

    result = ResidualGreedySplitter(
        config=ResidualGreedyConfig(max_regions=3, min_support=8, validation_tolerance=0.1)
    ).fit(dataset, train_model=train_model, evaluate_model=evaluate_model)

    assert result.atlas.n_regions == 1
    assert result.validation.passed
    assert result.events == ()
