import numpy as np


def _sorted_rows(a: np.ndarray) -> np.ndarray:
    idx = np.lexsort((a[:, 1], a[:, 0]))
    return a[idx]


def test_structured_quad_falls_back_when_numba_unavailable(monkeypatch):
    # meshgen's numba dependency is optional; even if numba is missing/unusable,
    # structured_quad(numba_path=True) should transparently fall back.
    import pycutfem.utils.meshgen as mg

    nodes_py, elems_py, edges_py, corners_py = mg.structured_quad(
        1.0, 1.0, nx=2, ny=2, poly_order=1, numba_path=False
    )

    monkeypatch.setattr(mg, "HAS_NUMBA", False)
    nodes_fb, elems_fb, edges_fb, corners_fb = mg.structured_quad(
        1.0, 1.0, nx=2, ny=2, poly_order=1, numba_path=True
    )

    assert np.array_equal(np.asarray(elems_py), np.asarray(elems_fb))
    assert np.array_equal(np.asarray(corners_py), np.asarray(corners_fb))
    assert np.array_equal(_sorted_rows(np.asarray(edges_py, dtype=int)), _sorted_rows(np.asarray(edges_fb, dtype=int)))

    xy_py = np.asarray([[float(n.x), float(n.y)] for n in nodes_py], dtype=float)
    xy_fb = np.asarray([[float(n.x), float(n.y)] for n in nodes_fb], dtype=float)
    assert xy_py.shape == xy_fb.shape
    assert np.allclose(_sorted_rows(xy_py), _sorted_rows(xy_fb), atol=1.0e-12, rtol=0.0)
