from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .metrics import mean_sample_l2_error, reduced_regression_error
from .pod import fit_pod, project_to_basis
from .quadratic_manifold import QuadraticManifoldDecoder, QuadraticFeatureMap, fit_quadratic_decoder


@dataclass(frozen=True)
class ModeSweepEntry:
    input_modes: int
    output_modes: int
    validation_error: float
    regression_error: float


@dataclass
class ModeSweepResult:
    entries: list[ModeSweepEntry]

    def best(self) -> ModeSweepEntry:
        if not self.entries:
            raise RuntimeError("no cross-validation entries available")
        return min(self.entries, key=lambda entry: entry.validation_error)


def _build_linear_decoder(pod_basis) -> QuadraticManifoldDecoder:
    feature_map = QuadraticFeatureMap(rank=pod_basis.basis.shape[1])
    quadratic_basis = np.zeros((pod_basis.basis.shape[0], feature_map.n_terms), dtype=float)
    return QuadraticManifoldDecoder(
        linear_basis=pod_basis.basis,
        quadratic_basis=quadratic_basis,
        mean=pod_basis.mean,
        feature_map=feature_map,
    )


def run_mode_cross_validation(
    input_snapshots: np.ndarray,
    output_snapshots: np.ndarray,
    *,
    input_modes: list[int] | range | None = None,
    output_modes: list[int] | range | None = None,
    regressor_factory: Callable[[], object],
    test_fraction: float = 0.2,
    random_state: int = 0,
    use_quadratic_decoder: bool = True,
) -> ModeSweepResult:
    """Run a held-out mode sweep for an input-output NIRB map."""

    if input_modes is None or output_modes is None:
        raise ValueError("input_modes and output_modes must be provided")

    input_matrix = np.asarray(input_snapshots, dtype=float)
    output_matrix = np.asarray(output_snapshots, dtype=float)
    if input_matrix.shape[1] != output_matrix.shape[1]:
        raise ValueError("input and output snapshot counts must match")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie in (0, 1)")

    n_snapshots = input_matrix.shape[1]
    permutation = np.random.default_rng(random_state).permutation(n_snapshots)
    n_test = max(1, int(round(test_fraction * n_snapshots)))
    test_indices = permutation[:n_test]
    train_indices = permutation[n_test:]

    x_train = input_matrix[:, train_indices]
    x_test = input_matrix[:, test_indices]
    y_train = output_matrix[:, train_indices]
    y_test = output_matrix[:, test_indices]

    entries: list[ModeSweepEntry] = []
    for r_x in input_modes:
        input_basis = fit_pod(x_train, n_modes=int(r_x), center=True)
        x_train_red = input_basis.project(x_train).T
        x_test_red = input_basis.project(x_test).T

        for r_y in output_modes:
            if use_quadratic_decoder:
                decoder = fit_quadratic_decoder(y_train, n_modes=int(r_y), center=True)
                y_train_red = project_to_basis(y_train, decoder.linear_basis, decoder.mean)
            else:
                output_basis = fit_pod(y_train, n_modes=int(r_y), center=True)
                decoder = _build_linear_decoder(output_basis)
                y_train_red = output_basis.project(y_train)

            regressor = regressor_factory()
            regressor.fit(x_train_red, y_train_red.T)
            y_pred_red_test = regressor.predict(x_test_red).T
            y_pred_red_train = regressor.predict(x_train_red).T
            y_pred_test = decoder.decode(y_pred_red_test)

            entries.append(
                ModeSweepEntry(
                    input_modes=int(r_x),
                    output_modes=int(r_y),
                    validation_error=mean_sample_l2_error(y_test, y_pred_test),
                    regression_error=reduced_regression_error(y_train_red, y_pred_red_train),
                )
            )

    return ModeSweepResult(entries=entries)
