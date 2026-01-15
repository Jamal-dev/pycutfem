import numpy as np
import pytest

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import LevelSetMeshAdaptation, SuperellipseLevelSet


def _find_noncut_neighbor_of_cut(mesh: Mesh, cut_mask: np.ndarray) -> int | None:
    cut_ids = set(np.where(np.asarray(cut_mask, dtype=bool))[0].tolist())
    neigh = mesh.neighbors()
    for eid in range(len(cut_mask)):
        if bool(cut_mask[eid]):
            continue
        for nb in neigh[int(eid)]:
            if int(nb) in cut_ids:
                return int(eid)
    return None


@pytest.mark.parametrize("poly_order", (2, 3))
def test_lsetmeshadap_hier_oswald_populates_interior_nodes(poly_order: int):
    """
    Regression (quad meshes, p>=2):

    NGSXFEM's ProjectShift performs an Oswald average on *hierarchical* H1 DOFs.
    In a nodal Qp lattice, averaging nodal values directly can leave interior
    lattice nodes (cell centers, etc.) exactly zero on non-cut elements even
    though the corresponding hierarchical field is nonzero there.

    The LevelSetMeshAdaptation implementation emulates the hierarchical Oswald
    and then evaluates the resulting field at the lattice nodes. This test
    ensures interior nodes on a non-cut element adjacent to a cut element are
    populated (nonzero).
    """
    pytest.importorskip("scipy")

    L = 3.0
    nx = ny = 6  # small but ensures a non-empty cut band
    nodes, elems, edges, corners = structured_quad(
        L, L, nx=nx, ny=ny, poly_order=int(poly_order), offset=(-L / 2, -L / 2)
    )
    mesh = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=int(poly_order))

    level_set = SuperellipseLevelSet(center=(0.0, 0.0), radius=1.0)
    lsetadap = LevelSetMeshAdaptation(mesh, order=int(poly_order))
    deformation = lsetadap.calc_deformation(level_set)
    lsetp1 = lsetadap.lset_p1
    assert lsetp1 is not None

    mesh.classify_elements(lsetp1)
    cut_mask = np.asarray(mesh.element_bitset("cut").mask, dtype=bool)
    assert bool(np.any(cut_mask)), "Expected at least one cut element"

    eid = _find_noncut_neighbor_of_cut(mesh, cut_mask)
    assert eid is not None, "Expected a non-cut element adjacent to the cut band"

    conn = np.asarray(mesh.elements_connectivity[int(eid)], dtype=int)
    nlat = int(poly_order) + 1
    interior = [int(conn[j * nlat + i]) for j in range(1, int(poly_order)) for i in range(1, int(poly_order))]
    assert len(interior) == (int(poly_order) - 1) ** 2

    disp = np.asarray(deformation.node_displacements[interior], dtype=float)
    max_norm = float(np.max(np.linalg.norm(disp, axis=1)))
    assert max_norm > 1e-12, f"Expected nonzero interior deformation, got max_norm={max_norm:.3e}"

