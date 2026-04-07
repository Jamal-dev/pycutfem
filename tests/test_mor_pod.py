import numpy as np

from pycutfem.mor.pod import fit_pod


def test_fit_pod_projects_and_reconstructs_rank_one_snapshot_matrix():
    snapshots = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
        ]
    )

    pod = fit_pod(snapshots, n_modes=1, center=False)
    reconstructed = pod.reconstruct(pod.project(snapshots))

    assert pod.basis.shape == (3, 1)
    assert np.allclose(pod.basis.T @ pod.basis, np.eye(1))
    assert np.allclose(reconstructed, snapshots)


def test_fit_pod_energy_threshold_selects_expected_rank():
    snapshots = np.diag([3.0, 2.0, 1.0])

    pod = fit_pod(snapshots, energy=0.9, center=False)

    assert pod.n_modes == 2
    assert np.allclose(pod.singular_values, np.array([3.0, 2.0]))
