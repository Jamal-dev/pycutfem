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

from .dataset import OfflineDataset, load_dataset


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
    force_modes: int
    displacement_modes: int
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    center_forces: bool = True
    center_displacements: bool = True
    use_quadratic_decoder: bool = True
    dataset_force_key: str = "load_guess_data"
    dataset_displacement_key: str = "disp_data"
    zero_anchor_weight: int = 0
    interface_indices: list[int] | None = None
    interface_matrix_path: str | None = None
    input_feature_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "OfflineConfig":
        values = dict(mapping)
        values["regression"] = RegressionConfig.from_mapping(values.get("regression"))
        return cls(**values)


@dataclass
class TrainedNIRBModel:
    force_basis: PODBasis
    decoder: QuadraticManifoldDecoder
    regressor: Any
    interface_restriction: InterfaceRestriction | None = None
    input_feature_names: tuple[str, ...] = ()
    input_feature_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def encode_forces(self, forces: np.ndarray) -> np.ndarray:
        return self.force_basis.project(forces)

    def _context_feature_matrix(
        self,
        context: dict[str, Any] | np.ndarray | None,
        *,
        n_samples: int,
    ) -> np.ndarray | None:
        names = tuple(str(name) for name in getattr(self, "input_feature_names", ()) or ())
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
        stats = getattr(self, "input_feature_stats", {}) or {}
        columns: list[np.ndarray] = []
        for name in names:
            stat = dict(stats.get(str(name), {}))
            mean = float(stat.get("mean", 0.0))
            scale = float(stat.get("scale", 1.0)) or 1.0
            raw_default = mean
            raw_value = context_map.get(str(name), raw_default)
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

    def _augment_reduced_forces(
        self,
        reduced_forces: np.ndarray,
        context: dict[str, Any] | np.ndarray | None = None,
    ) -> np.ndarray:
        reduced = np.asarray(reduced_forces, dtype=float)
        if reduced.ndim != 2:
            raise ValueError("reduced force coordinates must be a 2D sample-major matrix")
        features = self._context_feature_matrix(context, n_samples=reduced.shape[0])
        if features is None:
            return reduced
        return np.column_stack([reduced, features])

    def predict_reduced(
        self,
        forces: np.ndarray,
        context: dict[str, Any] | np.ndarray | None = None,
    ) -> np.ndarray:
        reduced_forces = self.encode_forces(forces).T
        regressor_inputs = self._augment_reduced_forces(reduced_forces, context)
        reduced_displacements = self.regressor.predict(regressor_inputs)
        return np.asarray(reduced_displacements, dtype=float).T

    def predict_reduced_from_force_coefficients(
        self,
        force_coefficients: np.ndarray,
        context: dict[str, Any] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict displacement coordinates from already-reduced force coordinates.

        Fully reduced online coupling stores the interface load in a reduced
        space.  When that space is the model's force POD space, re-encoding a
        reconstructed full interface load would be unnecessary work and would
        violate the reduced-online contract.  This method bypasses the force
        encoder and evaluates only the learned reduced map.
        """

        coeffs = np.asarray(force_coefficients, dtype=float)
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(1, -1)
        if coeffs.ndim != 2:
            raise ValueError("force_coefficients must be a 1D vector or sample-major 2D matrix")
        regressor_inputs = self._augment_reduced_forces(coeffs, context)
        reduced_displacements = self.regressor.predict(regressor_inputs)
        return np.asarray(reduced_displacements, dtype=float).T

    def predict_full(
        self,
        forces: np.ndarray,
        context: dict[str, Any] | np.ndarray | None = None,
    ) -> np.ndarray:
        return self.decoder.decode(self.predict_reduced(forces, context=context))

    def predict_interface(
        self,
        forces: np.ndarray,
        context: dict[str, Any] | np.ndarray | None = None,
    ) -> np.ndarray:
        reduced = self.predict_reduced(forces, context=context)
        if self.interface_restriction is None:
            return self.decoder.decode(reduced)
        return self.interface_restriction.restrict_decoder(self.decoder).decode(reduced)


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
    dataset: OfflineDataset,
    names: tuple[str, ...],
) -> tuple[np.ndarray | None, dict[str, dict[str, float]]]:
    if not names:
        return None, {}
    columns: list[np.ndarray] = []
    stats: dict[str, dict[str, float]] = {}
    for raw_name in names:
        name = str(raw_name).strip()
        if name == "time":
            if dataset.times is None:
                raise ValueError("input feature 'time' requires snapshot metadata with time_s")
            raw = np.asarray(dataset.times, dtype=float).reshape(-1)
        elif name in {"coupling_iter", "subiteration"}:
            if dataset.subiterations is None:
                raise ValueError("input feature 'coupling_iter' requires snapshot metadata with coupling_iter")
            raw = np.asarray(dataset.subiterations, dtype=float).reshape(-1)
            name = "coupling_iter"
        else:
            raise ValueError(f"unsupported NIRB input feature {raw_name!r}")
        if raw.size != dataset.n_snapshots:
            raise ValueError(
                f"input feature {name!r} has {raw.size} samples, expected {dataset.n_snapshots}"
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

    dataset: OfflineDataset = load_dataset(
        config.dataset_path,
        interface_indices=None if config.interface_indices is None else np.asarray(config.interface_indices, dtype=int),
        force_key=config.dataset_force_key,
        displacement_key=config.dataset_displacement_key,
    )
    input_feature_names = tuple(str(name).strip() for name in config.input_feature_names if str(name).strip())
    context_features, context_stats = _dataset_context_features(dataset, input_feature_names)
    forces = dataset.forces
    displacements = dataset.displacements
    if int(config.zero_anchor_weight) > 0:
        anchor_count = int(config.zero_anchor_weight)
        forces = np.column_stack(
            [forces, np.zeros((forces.shape[0], anchor_count), dtype=float)]
        )
        displacements = np.column_stack(
            [displacements, np.zeros((displacements.shape[0], anchor_count), dtype=float)]
        )
        if context_features is not None:
            context_features = np.vstack(
                [context_features, np.zeros((anchor_count, context_features.shape[1]), dtype=float)]
            )

    force_basis = fit_pod(
        forces,
        n_modes=config.force_modes,
        center=config.center_forces,
    )

    if config.use_quadratic_decoder:
        decoder = fit_quadratic_decoder(
            displacements,
            n_modes=config.displacement_modes,
            center=config.center_displacements,
        )
    else:
        displacement_basis = fit_pod(
            displacements,
            n_modes=config.displacement_modes,
            center=config.center_displacements,
        )
        decoder = _linear_decoder_from_basis(displacement_basis)

    reduced_forces = force_basis.project(forces).T
    if context_features is not None:
        if context_features.shape[0] != reduced_forces.shape[0]:
            raise ValueError("context feature sample count does not match reduced force sample count")
        reduced_forces = np.column_stack([reduced_forces, context_features])
    reduced_displacements = project_to_basis(
        displacements,
        decoder.linear_basis,
        decoder.mean,
    ).T

    regressor = _build_regressor(config.regression)
    regressor.fit(reduced_forces, reduced_displacements)

    restriction = None
    if config.interface_matrix_path is not None:
        restriction = InterfaceRestriction(matrix=np.load(config.interface_matrix_path))
    elif dataset.interface_indices is not None:
        restriction = InterfaceRestriction.from_indices(
            dataset.interface_indices,
            dataset.displacements.shape[0],
        )

    model = TrainedNIRBModel(
        force_basis=force_basis,
        decoder=decoder,
        regressor=regressor,
        interface_restriction=restriction,
        input_feature_names=input_feature_names,
        input_feature_stats=context_stats,
        metadata={
            "force_modes": config.force_modes,
            "displacement_modes": config.displacement_modes,
            "input_feature_names": list(input_feature_names),
            "input_feature_stats": context_stats,
            "dataset_path": config.dataset_path,
            "dataset_force_key": config.dataset_force_key,
            "dataset_displacement_key": config.dataset_displacement_key,
            "zero_anchor_weight": int(config.zero_anchor_weight),
            "regression": config.regression.__dict__,
            "dataset": dataset.metadata,
            **config.metadata,
        },
    )
    save_model(model, config.model_path)
    return model
