import numpy as np

from pycutfem.mor.pod import fit_pod, project_to_basis
from pycutfem.mor.quadratic_manifold import (
    QuadraticFeatureMap,
    QuadraticManifoldDecoder,
    fit_quadratic_decoder,
    fit_quadratic_manifold,
)


def test_quadratic_feature_map_uses_deterministic_upper_triangular_order():
    feature_map = QuadraticFeatureMap(rank=3)

    assert feature_map.pairs == ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def test_fit_quadratic_manifold_recovers_orthogonal_quadratic_correction():
    linear_basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    mean = np.array([[0.5], [-0.25], [0.0], [0.0]])
    reduced = np.array(
        [
            [1.0, -0.5, 0.25, 0.75],
            [0.2, 0.3, -0.4, 0.1],
        ]
    )
    feature_map = QuadraticFeatureMap(rank=2)
    quadratic_basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, -2.0, 0.5],
            [0.25, 0.75, -1.5],
        ]
    )
    snapshots = mean + linear_basis @ reduced + quadratic_basis @ feature_map.transform(reduced)

    fitted_quadratic_basis = fit_quadratic_manifold(
        snapshots,
        reduced,
        linear_basis,
        mean=mean,
    )
    decoder = QuadraticManifoldDecoder(
        linear_basis=linear_basis,
        quadratic_basis=fitted_quadratic_basis,
        mean=mean,
        feature_map=feature_map,
    )

    assert np.allclose(linear_basis.T @ fitted_quadratic_basis, 0.0)
    assert np.allclose(decoder.decode(reduced), snapshots)


def test_fit_quadratic_decoder_improves_training_reconstruction_over_linear_pod():
    reduced = np.array(
        [
            [1.0, -0.5, 0.25, 0.75, -0.2],
            [0.2, 0.3, -0.4, 0.1, 0.5],
        ]
    )
    feature_map = QuadraticFeatureMap(rank=2)
    linear_basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    quadratic_basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, -1.0, 2.0],
            [0.25, 0.75, -0.5],
        ]
    )
    mean = np.array([[0.1], [0.2], [0.0], [0.0]])
    snapshots = mean + linear_basis @ reduced + quadratic_basis @ feature_map.transform(reduced)

    linear_pod = fit_pod(snapshots, n_modes=2, center=True)
    linear_reconstruction = linear_pod.reconstruct(linear_pod.project(snapshots))
    decoder = fit_quadratic_decoder(snapshots, n_modes=2, center=True)
    reduced_from_pod = project_to_basis(snapshots, decoder.linear_basis, decoder.mean)
    quadratic_reconstruction = decoder.decode(reduced_from_pod)

    linear_error = np.linalg.norm(linear_reconstruction - snapshots)
    quadratic_error = np.linalg.norm(quadratic_reconstruction - snapshots)

    assert quadratic_error < linear_error
