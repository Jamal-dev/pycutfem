from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .metrics import mean_sample_l2_error, reduced_regression_error
from .pod import fit_pod, project_to_basis
from .quadratic_manifold import QuadraticManifoldDecoder, QuadraticFeatureMap, fit_quadratic_decoder


@dataclass(frozen=True)
class ModeSweepEntry:
    force_modes: int
    displacement_modes: int
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
    forces: np.ndarray,
    displacements: np.ndarray,
    *,
    force_modes: list[int] | range,
    displacement_modes: list[int] | range,
    regressor_factory: Callable[[], object],
    test_fraction: float = 0.2,
    random_state: int = 0,
    use_quadratic_decoder: bool = True,
) -> ModeSweepResult:
    """Run a simple held-out mode sweep using Eq. (26)/(30)/(31)-style errors."""

    force_matrix = np.asarray(forces, dtype=float)
    displacement_matrix = np.asarray(displacements, dtype=float)
    if force_matrix.shape[1] != displacement_matrix.shape[1]:
        raise ValueError("force and displacement snapshot counts must match")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie in (0, 1)")

    n_snapshots = force_matrix.shape[1]
    permutation = np.random.default_rng(random_state).permutation(n_snapshots)
    n_test = max(1, int(round(test_fraction * n_snapshots)))
    test_indices = permutation[:n_test]
    train_indices = permutation[n_test:]

    f_train = force_matrix[:, train_indices]
    f_test = force_matrix[:, test_indices]
    u_train = displacement_matrix[:, train_indices]
    u_test = displacement_matrix[:, test_indices]

    entries: list[ModeSweepEntry] = []
    for r_f in force_modes:
        force_basis = fit_pod(f_train, n_modes=int(r_f), center=True)
        f_train_red = force_basis.project(f_train).T
        f_test_red = force_basis.project(f_test).T

        for r_u in displacement_modes:
            if use_quadratic_decoder:
                decoder = fit_quadratic_decoder(u_train, n_modes=int(r_u), center=True)
                u_train_red = project_to_basis(u_train, decoder.linear_basis, decoder.mean)
            else:
                disp_basis = fit_pod(u_train, n_modes=int(r_u), center=True)
                decoder = _build_linear_decoder(disp_basis)
                u_train_red = disp_basis.project(u_train)

            regressor = regressor_factory()
            regressor.fit(f_train_red, u_train_red.T)
            u_pred_red_test = regressor.predict(f_test_red).T
            u_pred_red_train = regressor.predict(f_train_red).T
            u_pred_test = decoder.decode(u_pred_red_test)

            entries.append(
                ModeSweepEntry(
                    force_modes=int(r_f),
                    displacement_modes=int(r_u),
                    validation_error=mean_sample_l2_error(u_test, u_pred_test),
                    regression_error=reduced_regression_error(u_train_red, u_pred_red_train),
                )
            )

    return ModeSweepResult(entries=entries)
