import numpy as np

from pycutfem.mor.interface import InterfaceRestriction, build_restriction_matrix
from pycutfem.mor.quadratic_manifold import QuadraticFeatureMap, QuadraticManifoldDecoder


def test_build_restriction_matrix_selects_requested_degrees_of_freedom():
    restriction = build_restriction_matrix([2, 0], full_dofs=4)
    values = np.array([[1.0], [2.0], [3.0], [4.0]])

    assert np.allclose(restriction @ values, np.array([[3.0], [1.0]]))


def test_restricted_decoder_matches_post_restriction_of_full_reconstruction():
    feature_map = QuadraticFeatureMap(rank=2)
    decoder = QuadraticManifoldDecoder(
        linear_basis=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        quadratic_basis=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.5, -0.25],
                [0.75, -0.5, 1.25],
            ]
        ),
        mean=np.array([[0.1], [0.2], [0.0], [0.0]]),
        feature_map=feature_map,
    )
    reduced = np.array([[0.5, -0.25], [0.1, 0.3]])
    restriction = InterfaceRestriction.from_indices([0, 3], full_dofs=4)

    restricted_decoder = restriction.restrict_decoder(decoder)

    assert np.allclose(
        restricted_decoder.decode(reduced),
        restriction.restrict(decoder.decode(reduced)),
    )
