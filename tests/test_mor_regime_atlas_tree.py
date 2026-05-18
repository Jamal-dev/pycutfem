import numpy as np

from pycutfem.mor.regime_atlas import TreeRouter


def test_tree_router_learns_interpretable_feature_threshold() -> None:
    features = np.column_stack(
        [
            np.concatenate([np.linspace(-1.0, -0.1, 8), np.linspace(0.1, 1.0, 8)]),
            np.zeros(16),
        ]
    )
    labels = np.array([0] * 8 + [1] * 8)

    router = TreeRouter(max_depth=2, min_leaf=3).fit(features, labels)

    assert router.predict(features).tolist() == labels.tolist()
    assert router.predict_one(np.array([-0.5, 0.0])) == 0
    assert router.predict_one(np.array([0.5, 0.0])) == 1
