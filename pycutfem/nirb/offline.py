from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pycutfem.mor.interface import InterfaceRestriction
from pycutfem.mor.io import save_model
from pycutfem.mor.pod import PODBasis, fit_pod, project_to_basis
from pycutfem.mor.quadratic_manifold import QuadraticFeatureMap, QuadraticManifoldDecoder, fit_quadratic_decoder
from pycutfem.mor.regressors import PolynomialLassoRegressor, PolynomialLeastSquaresRegressor, ThinPlateSplineRBF

from .dataset import OfflineDataset, load_dataset


@dataclass
class RegressionConfig:
    kind: str = "tps_rbf"
    smoothing: float = 0.0
    degree: int = 2
    criterion: str = "bic"
    standardize_inputs: bool = True
    regularization: float = 0.0

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
    metadata: dict[str, Any] = field(default_factory=dict)

    def encode_forces(self, forces: np.ndarray) -> np.ndarray:
        return self.force_basis.project(forces)

    def predict_reduced(self, forces: np.ndarray) -> np.ndarray:
        reduced_forces = self.encode_forces(forces).T
        reduced_displacements = self.regressor.predict(reduced_forces)
        return np.asarray(reduced_displacements, dtype=float).T

    def predict_reduced_from_force_coefficients(self, force_coefficients: np.ndarray) -> np.ndarray:
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
        reduced_displacements = self.regressor.predict(coeffs)
        return np.asarray(reduced_displacements, dtype=float).T

    def predict_full(self, forces: np.ndarray) -> np.ndarray:
        return self.decoder.decode(self.predict_reduced(forces))

    def predict_interface(self, forces: np.ndarray) -> np.ndarray:
        reduced = self.predict_reduced(forces)
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
    raise ValueError(f"unsupported regression kind: {config.kind}")


def run_offline_pipeline(config: OfflineConfig | dict[str, Any]) -> TrainedNIRBModel:
    if isinstance(config, dict):
        config = OfflineConfig.from_mapping(config)

    dataset: OfflineDataset = load_dataset(
        config.dataset_path,
        interface_indices=None if config.interface_indices is None else np.asarray(config.interface_indices, dtype=int),
        force_key=config.dataset_force_key,
        displacement_key=config.dataset_displacement_key,
    )
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
        metadata={
            "force_modes": config.force_modes,
            "displacement_modes": config.displacement_modes,
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
