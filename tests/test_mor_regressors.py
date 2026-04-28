import numpy as np

from pycutfem.mor.regressors import (
    PolynomialFeatureMap,
    PolynomialLassoRegressor,
    PolynomialLeastSquaresRegressor,
    ThinPlateSplineRBF,
)


def test_thin_plate_spline_rbf_interpolates_training_points():
    inputs = np.array([[0.0], [1.0], [2.0], [3.0]])
    outputs = np.array([[0.0], [1.0], [4.0], [9.0]])

    model = ThinPlateSplineRBF().fit(inputs, outputs)

    assert np.allclose(model.predict(inputs), outputs)


def test_polynomial_feature_map_matches_expected_degree_two_order():
    feature_map = PolynomialFeatureMap(rank=2)
    features = feature_map.transform(np.array([[2.0, 3.0]]))

    assert np.allclose(features, np.array([[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]]))


def test_polynomial_lasso_regressor_recovers_sparse_quadratic_mapping():
    rng = np.random.default_rng(123)
    inputs = rng.normal(size=(100, 2))
    outputs = np.column_stack(
        [
            1.0 + 2.0 * inputs[:, 0] - 0.5 * inputs[:, 1] + 3.0 * inputs[:, 0] * inputs[:, 1],
            -1.5 + 0.75 * inputs[:, 1] ** 2,
        ]
    )

    model = PolynomialLassoRegressor(standardize_inputs=False).fit(inputs, outputs)
    predictions = model.predict(inputs)

    assert np.max(np.abs(predictions - outputs)) < 1.0e-8
    assert np.count_nonzero(model.sparsity_pattern()) < model.coefficients_.size


def test_polynomial_least_squares_regressor_recovers_dense_quadratic_mapping():
    rng = np.random.default_rng(321)
    inputs = rng.normal(size=(40, 3))
    outputs = np.column_stack(
        [
            0.2 + inputs[:, 0] * inputs[:, 1] - 0.4 * inputs[:, 2] ** 2,
            -0.1 + 0.3 * inputs[:, 0] + 0.7 * inputs[:, 1] * inputs[:, 2],
        ]
    )

    model = PolynomialLeastSquaresRegressor(standardize_inputs=False).fit(inputs, outputs)

    assert np.max(np.abs(model.predict(inputs) - outputs)) < 1.0e-10
