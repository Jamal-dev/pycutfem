from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pycutfem.mor.interface import InterfaceRestriction
from pycutfem.mor.io import save_model
from pycutfem.mor.pod import PODBasis, fit_pod, project_to_basis
from pycutfem.mor.quadratic_manifold import QuadraticFeatureMap, QuadraticManifoldDecoder, fit_quadratic_decoder
from pycutfem.mor.regressors import (
    KNearestRegressor,
    PolynomialLassoRegressor,
    PolynomialLeastSquaresRegressor,
    ThinPlateSplineRBF,
)

from .dataset import NIRBDataset, load_dataset


@dataclass
class RegressionConfig:
    kind: str = "tps_rbf"
    smoothing: float = 0.0
    degree: int = 2
    criterion: str = "bic"
    standardize_inputs: bool = True
    regularization: float = 0.0
    n_neighbors: int = 8
    power: float = 2.0

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "RegressionConfig":
        if mapping is None:
            return cls()
        return cls(**mapping)


@dataclass
class OfflineConfig:
    dataset_path: str
    model_path: str
    input_modes: int
    output_modes: int
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    center_input: bool = True
    center_output: bool = True
    use_quadratic_decoder: bool = True
    dataset_input_field: str = "input"
    dataset_output_field: str = "output"
    zero_anchor_weight: int = 0
    output_indices: list[int] | None = None
    output_matrix_path: str | None = None
    context_feature_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "OfflineConfig":
        values = dict(mapping)
        values["regression"] = RegressionConfig.from_mapping(values.get("regression"))
        return cls(**values)


@dataclass
class TrainedNIRBModel:
    input_basis: PODBasis
    output_decoder: QuadraticManifoldDecoder
    regressor: Any
    output_restriction: InterfaceRestriction | None = None
    context_feature_names: tuple[str, ...] = ()
    context_feature_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def encode_input(self, input_values: np.ndarray) -> np.ndarray:
        return self.input_basis.project(input_values)

    def _context_feature_matrix(
        self,
        context: dict[str, Any] | np.ndarray | None,
        *,
        n_samples: int,
    ) -> np.ndarray | None:
        names = tuple(str(name) for name in getattr(self, "context_feature_names", ()) or ())
        if not names:
            return None
        if isinstance(context, np.ndarray):
            matrix = np.asarray(context, dtype=float)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            if matrix.shape == (1, len(names)) and int(n_samples) != 1:
                matrix = np.repeat(matrix, int(n_samples), axis=0)
            if matrix.shape != (int(n_samples), len(names)):
                raise ValueError(
                    "context feature matrix has incompatible shape: "
                    f"{matrix.shape} != {(int(n_samples), len(names))}"
                )
            return matrix

        context_map = {} if context is None else dict(context)
        stats = getattr(self, "context_feature_stats", {}) or {}
        columns: list[np.ndarray] = []
        for name in names:
            stat = dict(stats.get(str(name), {}))
            mean = float(stat.get("mean", 0.0))
            scale = float(stat.get("scale", 1.0)) or 1.0
            raw_value = context_map.get(str(name), mean)
            raw = np.asarray(raw_value, dtype=float)
            if raw.ndim == 0:
                raw = np.full((int(n_samples),), float(raw), dtype=float)
            else:
                raw = raw.reshape(-1).astype(float)
                if raw.size == 1 and int(n_samples) != 1:
                    raw = np.full((int(n_samples),), float(raw[0]), dtype=float)
                if raw.size != int(n_samples):
                    raise ValueError(
                        f"context feature {name!r} has {raw.size} samples, expected {int(n_samples)}"
                    )
            min_value = stat.get("min")
            max_value = stat.get("max")
            if min_value is not None or max_value is not None:
                raw = np.clip(
                    raw,
                    -np.inf if min_value is None else float(min_value),
                    np.inf if max_value is None else float(max_value),
                )
            columns.append((raw - mean) / scale)
        return np.column_stack(columns)

    def _augment_reduced_input(
        self,
        reduced_input: np.ndarray,
        context: dict[str, Any] | np.ndarray | None = None,
    ) -> np.ndarray:
        reduced = np.asarray(reduced_input, dtype=float)
        if reduced.ndim != 2:
            raise ValueError("reduced input coordinates must be a 2D sample-major matrix")
        features = self._context_feature_matrix(context, n_samples=reduced.shape[0])
        if features is None:
            return reduced
        return np.column_stack([reduced, features])

    def predict_reduced(
        self,
        input_values: np.ndarray,
        context: dict[str, Any] | np.ndarray | None = None,
    ) -> np.ndarray:
        reduced_input = self.encode_input(input_values).T
        regressor_inputs = self._augment_reduced_input(reduced_input, context)
        reduced_output = self.regressor.predict(regressor_inputs)
        return np.asarray(reduced_output, dtype=float).T

    def predict_reduced_from_input_coefficients(
        self,
        input_coefficients: np.ndarray,
        context: dict[str, Any] | np.ndarray | None = None,
    ) -> np.ndarray:
        coeffs = np.asarray(input_coefficients, dtype=float)
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(1, -1)
        if coeffs.ndim != 2:
            raise ValueError("input_coefficients must be a 1D vector or sample-major 2D matrix")
        regressor_inputs = self._augment_reduced_input(coeffs, context)
        reduced_output = self.regressor.predict(regressor_inputs)
        return np.asarray(reduced_output, dtype=float).T

    def decode_output(self, reduced_output: np.ndarray) -> np.ndarray:
        return self.output_decoder.decode(reduced_output)

    def predict(
        self,
        input_values: np.ndarray,
        context: dict[str, Any] | np.ndarray | None = None,
    ) -> np.ndarray:
        return self.decode_output(self.predict_reduced(input_values, context=context))

    def predict_restricted(
        self,
        input_values: np.ndarray,
        context: dict[str, Any] | np.ndarray | None = None,
    ) -> np.ndarray:
        reduced = self.predict_reduced(input_values, context=context)
        if self.output_restriction is None:
            return self.output_decoder.decode(reduced)
        return self.output_restriction.restrict_decoder(self.output_decoder).decode(reduced)


def _linear_decoder_from_basis(pod_basis: PODBasis) -> QuadraticManifoldDecoder:
    feature_map = QuadraticFeatureMap(rank=pod_basis.basis.shape[1])
    quadratic_basis = np.zeros((pod_basis.basis.shape[0], feature_map.n_terms), dtype=float)
    return QuadraticManifoldDecoder(
        linear_basis=pod_basis.basis,
        quadratic_basis=quadratic_basis,
        mean=pod_basis.mean,
        feature_map=feature_map,
    )


def _build_regressor(config: RegressionConfig) -> Any:
    if config.kind == "tps_rbf":
        return ThinPlateSplineRBF(smoothing=config.smoothing)
    if config.kind == "poly_lasso":
        return PolynomialLassoRegressor(
            degree=config.degree,
            criterion=config.criterion,
            standardize_inputs=config.standardize_inputs,
        )
    if config.kind in {"poly_ls", "poly_ridge"}:
        return PolynomialLeastSquaresRegressor(
            degree=config.degree,
            regularization=config.regularization,
            standardize_inputs=config.standardize_inputs,
        )
    if config.kind in {"knn", "knearest", "nearest"}:
        return KNearestRegressor(
            n_neighbors=int(config.n_neighbors),
            power=float(config.power),
            regularization=max(float(config.regularization), 1.0e-300),
            standardize_inputs=config.standardize_inputs,
        )
    raise ValueError(f"unsupported regression kind: {config.kind}")


def _dataset_context_features(
    dataset: NIRBDataset,
    names: tuple[str, ...],
) -> tuple[np.ndarray | None, dict[str, dict[str, float]]]:
    if not names:
        return None, {}
    columns: list[np.ndarray] = []
    stats: dict[str, dict[str, float]] = {}
    for raw_name in names:
        name = str(raw_name).strip()
        raw = np.asarray(dataset.context(name), dtype=float).reshape(-1)
        if raw.size != dataset.n_snapshots:
            raise ValueError(
                f"context feature {name!r} has {raw.size} samples, expected {dataset.n_snapshots}"
            )
        mean = float(np.mean(raw))
        scale = float(np.std(raw))
        if not np.isfinite(scale) or scale <= 1.0e-14:
            scale = 1.0
        normalized = (raw - mean) / scale
        columns.append(normalized)
        stats[name] = {
            "mean": mean,
            "scale": scale,
            "min": float(np.min(raw)),
            "max": float(np.max(raw)),
        }
    return np.column_stack(columns), stats


def run_offline_pipeline(config: OfflineConfig | dict[str, Any]) -> TrainedNIRBModel:
    if isinstance(config, dict):
        config = OfflineConfig.from_mapping(config)

    dataset = load_dataset(
        config.dataset_path,
        input_field=config.dataset_input_field,
        output_field=config.dataset_output_field,
        output_indices=None if config.output_indices is None else np.asarray(config.output_indices, dtype=int),
    )
    context_feature_names = tuple(str(name).strip() for name in config.context_feature_names if str(name).strip())
    context_features, context_stats = _dataset_context_features(dataset, context_feature_names)
    input_snapshots = dataset.input_snapshots
    output_snapshots = dataset.output_snapshots
    if int(config.zero_anchor_weight) > 0:
        anchor_count = int(config.zero_anchor_weight)
        input_snapshots = np.column_stack(
            [input_snapshots, np.zeros((input_snapshots.shape[0], anchor_count), dtype=float)]
        )
        output_snapshots = np.column_stack(
            [output_snapshots, np.zeros((output_snapshots.shape[0], anchor_count), dtype=float)]
        )
        if context_features is not None:
            context_features = np.vstack(
                [context_features, np.zeros((anchor_count, context_features.shape[1]), dtype=float)]
            )

    input_basis = fit_pod(
        input_snapshots,
        n_modes=config.input_modes,
        center=config.center_input,
    )

    if config.use_quadratic_decoder:
        output_decoder = fit_quadratic_decoder(
            output_snapshots,
            n_modes=config.output_modes,
            center=config.center_output,
        )
    else:
        output_basis = fit_pod(
            output_snapshots,
            n_modes=config.output_modes,
            center=config.center_output,
        )
        output_decoder = _linear_decoder_from_basis(output_basis)

    reduced_input = input_basis.project(input_snapshots).T
    if context_features is not None:
        if context_features.shape[0] != reduced_input.shape[0]:
            raise ValueError("context feature sample count does not match reduced input sample count")
        reduced_input = np.column_stack([reduced_input, context_features])
    reduced_output = project_to_basis(
        output_snapshots,
        output_decoder.linear_basis,
        output_decoder.mean,
    ).T

    regressor = _build_regressor(config.regression)
    regressor.fit(reduced_input, reduced_output)

    restriction = None
    if config.output_matrix_path is not None:
        restriction = InterfaceRestriction(matrix=np.load(config.output_matrix_path))
    elif dataset.output_indices is not None:
        restriction = InterfaceRestriction.from_indices(
            dataset.output_indices,
            dataset.output_snapshots.shape[0],
        )

    model = TrainedNIRBModel(
        input_basis=input_basis,
        output_decoder=output_decoder,
        regressor=regressor,
        output_restriction=restriction,
        context_feature_names=context_feature_names,
        context_feature_stats=context_stats,
        metadata={
            "input_modes": config.input_modes,
            "output_modes": config.output_modes,
            "context_feature_names": list(context_feature_names),
            "context_feature_stats": context_stats,
            "dataset_path": config.dataset_path,
            "dataset_input_field": config.dataset_input_field,
            "dataset_output_field": config.dataset_output_field,
            "zero_anchor_weight": int(config.zero_anchor_weight),
            "regression": config.regression.__dict__,
            "dataset": dataset.metadata,
            **config.metadata,
        },
    )
    save_model(model, config.model_path)
    return model


__all__ = [
    "OfflineConfig",
    "RegressionConfig",
    "TrainedNIRBModel",
    "run_offline_pipeline",
]
