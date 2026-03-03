import numpy as np

from pycutfem.core.levelset import AffineLevelSet, PiecewiseLinearLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.utils.meshgen import structured_triangles


def _make_mesh(nx: int = 6, ny: int = 6) -> Mesh:
    nodes, elems, edges, corners = structured_triangles(
        Lx=1.0,
        Ly=1.0,
        nx_quads=int(nx),
        ny_quads=int(ny),
        poly_order=1,
        offset=(0.0, 0.0),
    )
    return Mesh(nodes, elems, edges, corners, element_type="tri", poly_order=1)


def test_classify_elements_piecewise_linear_matches_affine():
    # Vertical interface x=0.47 on a P1 triangular mesh (not mesh-aligned).
    # For a linear level-set on a P1 mesh, nodal signs fully determine cut elements.
    ls_aff = AffineLevelSet(a=1.0, b=0.0, c=-0.47)

    mesh_pl = _make_mesh()
    ls_pl = PiecewiseLinearLevelSet.from_level_set(mesh_pl, ls_aff)
    inside_pl, outside_pl, cut_pl = mesh_pl.classify_elements(ls_pl, tol=1.0e-12)

    mesh_aff = _make_mesh()
    inside_aff, outside_aff, cut_aff = mesh_aff.classify_elements(ls_aff, tol=1.0e-12)

    assert cut_pl.size > 0
    assert np.array_equal(np.sort(cut_pl), np.sort(cut_aff))
    assert np.array_equal(np.sort(inside_pl), np.sort(inside_aff))
    assert np.array_equal(np.sort(outside_pl), np.sort(outside_aff))

    # Cached reclassification should be stable.
    inside2, outside2, cut2 = mesh_pl.classify_elements(ls_pl, tol=1.0e-12)
    assert np.array_equal(inside_pl, inside2)
    assert np.array_equal(outside_pl, outside2)
    assert np.array_equal(cut_pl, cut2)
