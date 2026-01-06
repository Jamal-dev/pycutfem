import numpy as np

from pycutfem.core.geometry import hansbo_cut_ratio
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.utils.fsi_fully_eulerian import refresh_sliver_weights
from pycutfem.utils.meshgen import structured_quad


def _make_mesh():
    nodes, elems, edges, corners = structured_quad(
        2.0,
        1.0,
        nx=2,
        ny=1,
        poly_order=1,
        offset=(-1.0, -0.5),
    )
    return Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )


def test_sliver_weights_no_cut_elements():
    mesh = _make_mesh()
    level_set = AffineLevelSet(a=1.0, b=0.0, c=0.0)  # interface aligned at x=0
    mesh.classify_elements(level_set)

    theta_pos = hansbo_cut_ratio(mesh, level_set, side="+")
    theta_neg = hansbo_cut_ratio(mesh, level_set, side="-")
    w_pos = np.ones_like(theta_pos)
    w_neg = np.ones_like(theta_neg)

    refresh_sliver_weights(
        mesh,
        theta_pos,
        theta_neg,
        w_pos,
        w_neg,
        theta0=0.05,
        p=1.0,
        wmax=1000.0,
        thetamin=1.0e-6,
        smooth=1.0,
    )

    assert np.allclose(w_pos, 1.0)
    assert np.allclose(w_neg, 1.0)


def test_sliver_weights_activate_on_tiny_cut():
    mesh = _make_mesh()
    level_set = AffineLevelSet(a=1.0, b=0.0, c=-1.0e-6)  # slight shift → sliver
    mesh.classify_elements(level_set)

    theta_pos = hansbo_cut_ratio(mesh, level_set, side="+")
    theta_neg = hansbo_cut_ratio(mesh, level_set, side="-")
    w_pos = np.ones_like(theta_pos)
    w_neg = np.ones_like(theta_neg)

    refresh_sliver_weights(
        mesh,
        theta_pos,
        theta_neg,
        w_pos,
        w_neg,
        theta0=0.05,
        p=1.0,
        wmax=1000.0,
        thetamin=1.0e-6,
        smooth=1.0,
    )

    cut_ids = mesh.element_bitset("cut").to_indices()
    assert cut_ids.size > 0
    assert float(theta_neg[cut_ids].min()) < 1.0e-4
    assert float(w_neg[cut_ids].max()) > 1.0

    non_cut = np.setdiff1d(np.arange(len(theta_pos)), cut_ids)
    assert np.allclose(w_pos[non_cut], 1.0)
