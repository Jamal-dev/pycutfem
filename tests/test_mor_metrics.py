import numpy as np

from pycutfem.mor.metrics import (
    accumulated_iteration_overhead,
    max_online_relative_displacement_error,
    mean_sample_l2_error,
    reduced_regression_error,
    snapshot_l2_error,
    speedup,
)


def test_metric_formulas_match_manual_values():
    reference = np.array([[1.0, 2.0], [0.0, 2.0]])
    prediction = np.array([[1.0, 1.0], [1.0, 1.0]])

    assert np.isclose(mean_sample_l2_error(reference, prediction), 0.5 * (1.0 + np.sqrt(2.0)))
    assert np.isclose(snapshot_l2_error(reference, prediction), np.sqrt(3.0))
    assert np.isclose(reduced_regression_error(reference, prediction), np.sqrt(3.0))

    denominator = np.mean([1.0, np.sqrt(8.0)])
    expected_online_max = max(1.0 / denominator, np.sqrt(2.0) / denominator)
    assert np.isclose(max_online_relative_displacement_error(reference, prediction), expected_online_max)

    assert np.isclose(accumulated_iteration_overhead([10, 12], [11, 13]), 2.0 / 22.0)
    assert np.isclose(speedup(20.0, 0.5), 40.0)
