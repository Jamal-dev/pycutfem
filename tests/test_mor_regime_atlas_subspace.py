import numpy as np

from pycutfem.mor.regime_atlas import SubspacePartitioner, subspace_distance_matrix


def test_subspace_partitioner_clusters_grassmann_neighbors() -> None:
    basis_a = np.eye(4)[:, :2]
    basis_b = np.eye(4)[:, :2]
    basis_c = np.eye(4)[:, 2:4]
    basis_d = np.eye(4)[:, 2:4]

    partition = SubspacePartitioner(n_regions=2).fit([basis_a, basis_b, basis_c, basis_d])

    assert partition.labels[0] == partition.labels[1]
    assert partition.labels[2] == partition.labels[3]
    assert partition.labels[0] != partition.labels[2]
    assert partition.medoid_indices.size == 2


def test_subspace_distance_matrix_is_symmetric() -> None:
    bases = [np.eye(3)[:, :1], np.eye(3)[:, 1:2], np.eye(3)[:, 2:3]]

    distances = subspace_distance_matrix(bases)

    assert distances.shape == (3, 3)
    assert np.allclose(distances, distances.T)
    assert np.allclose(np.diag(distances), 0.0)
