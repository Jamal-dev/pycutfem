import math

import numpy as np

from pycutfem.utils.meshgen import structured_quad


def _coords(nodes) -> np.ndarray:
    return np.asarray([[float(n.x), float(n.y)] for n in nodes], dtype=float)


def _sorted_rows(a: np.ndarray) -> np.ndarray:
    idx = np.lexsort((a[:, 1], a[:, 0]))
    return a[idx]


def test_structured_quad_rotation_about_center_matches_expected_corners():
    # Build the unit square [-0.5,0.5]^2, then rotate by 45° about the origin.
    nodes, *_ = structured_quad(
        1.0,
        1.0,
        nx=1,
        ny=1,
        poly_order=1,
        offset=(-0.5, -0.5),
        rotation=math.pi / 4.0,
        rotation_center=(0.0, 0.0),
        numba_path=False,
    )
    got = _sorted_rows(_coords(nodes))
    r = math.sqrt(2.0) / 2.0
    expected = _sorted_rows(np.array([[-r, 0.0], [0.0, -r], [0.0, r], [r, 0.0]], dtype=float))
    assert got.shape == expected.shape
    assert np.allclose(got, expected, atol=1.0e-12, rtol=0.0)


def test_structured_quad_rotation_numba_and_python_paths_agree():
    kwargs = dict(
        Lx=1.0,
        Ly=1.0,
        nx=2,
        ny=2,
        poly_order=1,
        offset=(-0.5, -0.5),
        rotation=0.37,
        rotation_center=(0.0, 0.0),
    )
    nodes_py, elems_py, edges_py, corners_py = structured_quad(**kwargs, numba_path=False)
    nodes_nb, elems_nb, edges_nb, corners_nb = structured_quad(**kwargs, numba_path=True, parallel=False)

    assert np.array_equal(np.asarray(elems_py), np.asarray(elems_nb))
    assert np.array_equal(np.asarray(corners_py), np.asarray(corners_nb))
    assert np.array_equal(_sorted_rows(np.asarray(edges_py, dtype=int)), _sorted_rows(np.asarray(edges_nb, dtype=int)))

    got_py = _sorted_rows(_coords(nodes_py))
    got_nb = _sorted_rows(_coords(nodes_nb))
    assert got_py.shape == got_nb.shape
    assert np.allclose(got_py, got_nb, atol=1.0e-12, rtol=0.0)
