from __future__ import annotations

import numpy as np

from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.utils.meshgen import structured_quad


def test_affine_levelset_cache_token_updates_classification_on_mutation() -> None:
    """
    Regression test: `Mesh.classify_elements` caches results per level-set.

    If an AffineLevelSet is *mutated* (e.g. moving internal boundary), the cache
    must be invalidated; otherwise cut/inside/outside sets become stale.
    """
    nodes, elems, _edges, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    # Horizontal line y = 0.25
    ls = AffineLevelSet(0.0, 1.0, -0.25)
    inside0, outside0, cut0 = mesh.classify_elements(ls)

    # Move the same object to y = 0.75 (mutation).
    ls.c = -0.75
    inside1, outside1, cut1 = mesh.classify_elements(ls)

    # Classification must change after moving the line.
    assert not np.array_equal(np.asarray(inside0), np.asarray(inside1))
    assert not np.array_equal(np.asarray(outside0), np.asarray(outside1))
    assert not np.array_equal(np.asarray(cut0), np.asarray(cut1))

