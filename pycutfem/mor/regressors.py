from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LassoLarsIC
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler


def _as_sample_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2:
        raise ValueError("expected a 1D or 2D sample-major array")
    return matrix


def _as_target_matrix(values: np.ndarray, n_samples: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("expected a 1D or 2D target array")
    if matrix.shape[0] != n_samples:
        raise ValueError("target sample count does not match input sample count")
    return matrix


def _thin_plate_spline_kernel(radii: np.ndarray) -> np.ndarray:
    radii = np.asarray(radii, dtype=float)
    kernel = np.zeros_like(radii)
    mask = radii > 0.0
    kernel[mask] = (radii[mask] ** 2) * np.log(radii[mask])
    return kernel


@dataclass
class ThinPlateSplineRBF:
    smoothing: float = 0.0
    centers_: np.ndarray | None = None
    weights_: np.ndarray | None = None

    def fit(self, inputs: np.ndarray, outputs: np.ndarray) -> "ThinPlateSplineRBF":
        x_train = _as_sample_matrix(inputs)
        y_train = _as_target_matrix(outputs, n_samples=x_train.shape[0])
        kernel = _thin_plate_spline_kernel(cdist(x_train, x_train))
        if self.smoothing:
            kernel = kernel + self.smoothing * np.eye(kernel.shape[0])
        try:
            weights = np.linalg.solve(kernel, y_train)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(kernel, y_train, rcond=None)[0]
        self.centers_ = x_train
        self.weights_ = weights
        return self

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if self.centers_ is None or self.weights_ is None:
            raise RuntimeError("ThinPlateSplineRBF must be fit before predict")
        x_eval = _as_sample_matrix(inputs)
        kernel = _thin_plate_spline_kernel(cdist(x_eval, self.centers_))
        return kernel @ self.weights_


@dataclass(frozen=True)
class PolynomialFeatureMap:
    rank: int
    degree: int = 2
    include_bias: bool = True
    include_linear: bool = True
    quadratic_pairs: tuple[tuple[int, int], ...] = field(init=False)

    def __post_init__(self) -> None:
        if self.degree != 2:
            raise ValueError("PolynomialFeatureMap currently implements degree=2 only")
        object.__setattr__(
            self,
            "quadratic_pairs",
            tuple((i, j) for i in range(self.rank) for j in range(i, self.rank)),
        )

    @property
    def n_output_features(self) -> int:
        count = len(self.quadratic_pairs)
        if self.include_linear:
            count += self.rank
        if self.include_bias:
            count += 1
        return count

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        x = _as_sample_matrix(inputs)
        if x.shape[1] != self.rank:
            raise ValueError("input feature dimension does not match feature map rank")
        columns: list[np.ndarray] = []
        if self.include_bias:
            columns.append(np.ones(x.shape[0], dtype=float))
        if self.include_linear:
            for feature_index in range(self.rank):
                columns.append(x[:, feature_index])
        for i, j in self.quadratic_pairs:
            columns.append(x[:, i] * x[:, j])
        return np.column_stack(columns)


@dataclass
class PolynomialLassoRegressor:
    degree: int = 2
    criterion: str = "bic"
    standardize_inputs: bool = True
    scaler_: SklearnStandardScaler | None = None
    feature_map_: PolynomialFeatureMap | None = None
    coefficients_: np.ndarray | None = None
    selected_alpha_: np.ndarray | None = None

    def fit(self, inputs: np.ndarray, outputs: np.ndarray) -> "PolynomialLassoRegressor":
        x_train = _as_sample_matrix(inputs)
        y_train = _as_target_matrix(outputs, n_samples=x_train.shape[0])

        if self.standardize_inputs:
            self.scaler_ = SklearnStandardScaler(with_mean=True, with_std=True)
            x_scaled = self.scaler_.fit_transform(x_train)
        else:
            self.scaler_ = None
            x_scaled = x_train

        self.feature_map_ = PolynomialFeatureMap(rank=x_scaled.shape[1], degree=self.degree)
        design = self.feature_map_.transform(x_scaled)

        coefficients = []
        alphas = []
        for output_index in range(y_train.shape[1]):
            estimator = LassoLarsIC(criterion=self.criterion, fit_intercept=False)
            estimator.fit(design, y_train[:, output_index])
            coefficients.append(estimator.coef_)
            alphas.append(float(estimator.alpha_))

        self.coefficients_ = np.vstack(coefficients)
        self.selected_alpha_ = np.asarray(alphas, dtype=float)
        return self

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if self.feature_map_ is None or self.coefficients_ is None:
            raise RuntimeError("PolynomialLassoRegressor must be fit before predict")
        x_eval = _as_sample_matrix(inputs)
        if self.scaler_ is not None:
            x_eval = self.scaler_.transform(x_eval)
        design = self.feature_map_.transform(x_eval)
        return design @ self.coefficients_.T

    def sparsity_pattern(self) -> np.ndarray:
        if self.coefficients_ is None:
            raise RuntimeError("PolynomialLassoRegressor must be fit before accessing sparsity")
        return np.abs(self.coefficients_) > 0.0


@dataclass
class PolynomialLeastSquaresRegressor:
    degree: int = 2
    regularization: float = 0.0
    standardize_inputs: bool = True
    rcond: float | None = None
    scaler_: SklearnStandardScaler | None = None
    feature_map_: PolynomialFeatureMap | None = None
    coefficients_: np.ndarray | None = None

    def fit(self, inputs: np.ndarray, outputs: np.ndarray) -> "PolynomialLeastSquaresRegressor":
        x_train = _as_sample_matrix(inputs)
        y_train = _as_target_matrix(outputs, n_samples=x_train.shape[0])

        if self.standardize_inputs:
            self.scaler_ = SklearnStandardScaler(with_mean=True, with_std=True)
            x_scaled = self.scaler_.fit_transform(x_train)
        else:
            self.scaler_ = None
            x_scaled = x_train

        self.feature_map_ = PolynomialFeatureMap(rank=x_scaled.shape[1], degree=self.degree)
        design = self.feature_map_.transform(x_scaled)
        regularization = float(self.regularization)
        if regularization > 0.0:
            normal = design.T @ design
            penalty = regularization * np.eye(normal.shape[0], dtype=float)
            if self.feature_map_.include_bias:
                penalty[0, 0] = 0.0
            self.coefficients_ = np.linalg.solve(normal + penalty, design.T @ y_train).T
        else:
            self.coefficients_ = np.linalg.lstsq(design, y_train, rcond=self.rcond)[0].T
        return self

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if self.feature_map_ is None or self.coefficients_ is None:
            raise RuntimeError("PolynomialLeastSquaresRegressor must be fit before predict")
        x_eval = _as_sample_matrix(inputs)
        if self.scaler_ is not None:
            x_eval = self.scaler_.transform(x_eval)
        design = self.feature_map_.transform(x_eval)
        return design @ self.coefficients_.T


def fit_tps_rbf(inputs: np.ndarray, outputs: np.ndarray, *, smoothing: float = 0.0) -> ThinPlateSplineRBF:
    return ThinPlateSplineRBF(smoothing=smoothing).fit(inputs, outputs)


def fit_poly_lasso(
    inputs: np.ndarray,
    outputs: np.ndarray,
    *,
    degree: int = 2,
    criterion: str = "bic",
    standardize_inputs: bool = True,
) -> PolynomialLassoRegressor:
    return PolynomialLassoRegressor(
        degree=degree,
        criterion=criterion,
        standardize_inputs=standardize_inputs,
    ).fit(inputs, outputs)


def fit_poly_least_squares(
    inputs: np.ndarray,
    outputs: np.ndarray,
    *,
    degree: int = 2,
    regularization: float = 0.0,
    standardize_inputs: bool = True,
) -> PolynomialLeastSquaresRegressor:
    return PolynomialLeastSquaresRegressor(
        degree=degree,
        regularization=regularization,
        standardize_inputs=standardize_inputs,
    ).fit(inputs, outputs)
