import numpy as np


def test_cut_volume_phis_use_qref_under_deformation():
    """
    Regression test:
    When a FE-backed level set is used together with a mesh deformation,
    cut-volume precompute must evaluate φ at the *reference* quadrature point
    (qref), not by inverse-mapping the deformed physical point into the
    undeformed mesh.
    """
    from pycutfem.core.mesh import Mesh
    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.core.levelset import LevelSetDeformation, PiecewiseLinearLevelSet
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.utils.meshgen import structured_quad

    # Single Q1 quad on [-1,1]^2 (keeps inverse-mapping well-behaved).
    nodes, elems, edges, corners = structured_quad(
        2.0, 2.0, nx=1, ny=1, poly_order=1, offset=[-1.0, -1.0]
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    # FE-backed level set: phi(x,y)=x cuts the element.
    lset_p1 = PiecewiseLinearLevelSet.from_level_set(mesh, lambda xy: float(xy[0]))
    mesh.classify_elements(lset_p1, tol=1e-14)
    cut = mesh.element_bitset("cut")
    assert cut.to_indices() == [0]

    # Deformation: x -> 0.9 x (represented exactly by Q1 nodal displacements).
    node_disp = np.zeros_like(mesh.nodes_x_y_pos, dtype=float)
    node_disp[:, 0] = -0.1 * mesh.nodes_x_y_pos[:, 0]
    deformation = LevelSetDeformation(mesh, node_disp)

    dh = DofHandler(MixedElement(mesh, {"u": 1}), method="cg")
    out = dh.precompute_cut_volume_factors(
        cut,
        qdeg=4,
        derivs=set(),
        level_set=lset_p1,
        side="+",
        reuse=False,
        deformation=deformation,
    )

    qref = np.asarray(out["qref"], dtype=float)
    phis = np.asarray(out["phis"], dtype=float)
    qw = np.asarray(out["qw"], dtype=float)
    eids = np.asarray(out["eids"], dtype=int)

    for i, eid in enumerate(eids):
        for q in range(qw.shape[1]):
            if qw[i, q] == 0.0:
                continue
            xi = float(qref[i, q, 0])
            eta = float(qref[i, q, 1])
            expected = float(lset_p1.value_on_element(int(eid), (xi, eta)))
            assert abs(float(phis[i, q]) - expected) < 1e-12

