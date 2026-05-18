"""Simple interpretable tree router for regime labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .data import as_feature_matrix


def _gini(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts.astype(float) / float(labels.size)
    return float(1.0 - np.sum(probs * probs))


@dataclass(frozen=True)
class _TreeNode:
    prediction: int
    feature_index: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


@dataclass
class TreeRouter:
    """Greedy decision-tree router trained from region labels."""

    max_depth: int = 3
    min_leaf: int = 5
    feature_names: Sequence[str] | None = None
    metadata: Mapping[str, Any] | None = None
    root: _TreeNode | None = None

    def fit(self, features: np.ndarray, labels: Sequence[int] | np.ndarray) -> "TreeRouter":
        matrix = as_feature_matrix(features)
        y = np.asarray(labels, dtype=int).reshape(-1)
        if y.size != matrix.shape[0]:
            raise ValueError("labels length must match the number of samples.")
        self.root = self._build(matrix, y, depth=0)
        return self

    def _build(self, matrix: np.ndarray, labels: np.ndarray, *, depth: int) -> _TreeNode:
        values, counts = np.unique(labels, return_counts=True)
        prediction = int(values[int(np.argmax(counts))])
        if depth >= int(self.max_depth) or labels.size < 2 * int(self.min_leaf) or values.size == 1:
            return _TreeNode(prediction=prediction)
        parent_impurity = _gini(labels)
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        for j in range(matrix.shape[1]):
            candidates = np.unique(matrix[:, j])
            if candidates.size <= 1:
                continue
            thresholds = 0.5 * (candidates[:-1] + candidates[1:])
            for threshold in thresholds:
                left = matrix[:, j] <= threshold
                right = ~left
                if np.count_nonzero(left) < int(self.min_leaf) or np.count_nonzero(right) < int(self.min_leaf):
                    continue
                impurity = (
                    np.count_nonzero(left) * _gini(labels[left])
                    + np.count_nonzero(right) * _gini(labels[right])
                ) / float(labels.size)
                gain = parent_impurity - impurity
                if gain > best_gain:
                    best_gain = float(gain)
                    best_feature = int(j)
                    best_threshold = float(threshold)
        if best_feature is None or best_threshold is None:
            return _TreeNode(prediction=prediction)
        left_mask = matrix[:, best_feature] <= best_threshold
        return _TreeNode(
            prediction=prediction,
            feature_index=best_feature,
            threshold=best_threshold,
            left=self._build(matrix[left_mask, :], labels[left_mask], depth=depth + 1),
            right=self._build(matrix[~left_mask, :], labels[~left_mask], depth=depth + 1),
        )

    def predict_one(self, feature: Sequence[float] | np.ndarray) -> int:
        if self.root is None:
            raise RuntimeError("TreeRouter must be fit before prediction.")
        values = np.asarray(feature, dtype=float).reshape(-1)
        node = self.root
        while not node.is_leaf:
            assert node.feature_index is not None and node.threshold is not None
            node = node.left if values[node.feature_index] <= node.threshold else node.right
            assert node is not None
        return int(node.prediction)

    def predict(self, features: np.ndarray) -> np.ndarray:
        matrix = as_feature_matrix(features)
        return np.asarray([self.predict_one(row) for row in matrix], dtype=int)


__all__ = ["TreeRouter"]
