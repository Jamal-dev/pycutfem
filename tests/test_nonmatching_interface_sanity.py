import numpy as np
import pytest
import scipy.sparse as sp

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.fem import transform
from pycutfem.nonmatching import build_composite_mesh, build_nonmatching_interface, lift_nonmatching_interface_to_composite
from pycutfem.nonmatching.interface import NonMatchingInterface
from pycutfem.nonmatching.nitsche import assemble_poisson_nitsche_interface_matrix
from pycutfem.ufl.expressions import CellDiameter, Constant, FacetNormal, Neg, Pos, TestFunction, TrialFunction, dot, grad, inner
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dNonmatchingInterface
from pycutfem.utils.meshgen import structured_quad


def _make_submesh_quad(*, poly_order: int, nx: int, ny: int, offset_x: float) -> Mesh:
    nodes, elems, edges, corners = structured_quad(0.5, 1.0, nx=nx, ny=ny, poly_order=poly_order, offset=(offset_x, 0.0))
    return Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=poly_order)


def _tag_interface_boundary(mesh: Mesh, *, x_iface: float = 0.5) -> None:
    mesh.tag_boundary_edges(
        {
            "interface": lambda x, y: abs(float(x) - float(x_iface)) < 1e-12,
            "boundary": lambda x, y: True,
        }
    )


def _build_interface_pair(*, degree: int, ny_neg: int, ny_pos: int):
    nx_neg = max(2, int(round(0.5 * ny_neg)))
    nx_pos = max(2, int(round(0.5 * ny_pos)))
    mesh_neg = _make_submesh_quad(poly_order=degree, nx=nx_neg, ny=ny_neg, offset_x=0.0)
    mesh_pos = _make_submesh_quad(poly_order=degree, nx=nx_pos, ny=ny_pos, offset_x=0.5)
    _tag_interface_boundary(mesh_neg)
    _tag_interface_boundary(mesh_pos)
    interface = build_nonmatching_interface(mesh_neg=mesh_neg, mesh_pos=mesh_pos)
    mapping = build_composite_mesh(mesh_pos=mesh_pos, mesh_neg=mesh_neg, order="pos_neg")
    interface_c = lift_nonmatching_interface_to_composite(interface=interface, mapping=mapping)
    return mesh_pos, mesh_neg, interface, mapping, interface_c


def test_nonmatching_interface_pairing_coverage_and_ordering():
    _mesh_pos, _mesh_neg, interface, _mapping, _interface_c = _build_interface_pair(degree=1, ny_neg=12, ny_pos=19)

    seg_len = np.linalg.norm(interface.P1 - interface.P0, axis=1)
    assert np.all(seg_len > 1e-14)

    total_len = float(np.sum(seg_len))
    assert abs(total_len - 1.0) < 1e-8

    t = interface.P1[0] - interface.P0[0]
    tnorm = float(np.linalg.norm(t))
    assert tnorm > 0.0
    t /= tnorm
    s_end = interface.P1[:-1] @ t
    s_start_next = interface.P0[1:] @ t
    assert float(np.max(np.abs(s_start_next - s_end))) < 1e-10


def test_nonmatching_interface_normal_orientation_centroid():
    _mesh_pos, _mesh_neg, interface, _mapping, _interface_c = _build_interface_pair(degree=1, ny_neg=10, ny_pos=14)

    cpos = np.asarray([interface.mesh_pos.elements_list[int(eid)].centroid() for eid in interface.pos_elem_ids], dtype=float)
    cneg = np.asarray([interface.mesh_neg.elements_list[int(eid)].centroid() for eid in interface.neg_elem_ids], dtype=float)
    dotv = np.einsum("ij,ij->i", interface.n, cpos - cneg)
    assert np.all(dotv > 0.0), f"Found misoriented normals (min dot={dotv.min():.3e})"


def test_nonmatching_interface_quadrature_weights_and_inverse_map_roundtrip():
    _mesh_pos, _mesh_neg, _interface, mapping, interface_c = _build_interface_pair(degree=2, ny_neg=6, ny_pos=9)

    dh = DofHandler(MixedElement(mapping.mesh, {"u": 2}), method="cg")
    pc = dh.precompute_nonmatching_interface_factors(interface_c, qdeg=8, derivs={(0, 0), (1, 0), (0, 1)})

    P0 = np.asarray(interface_c.P0, dtype=float)
    P1 = np.asarray(interface_c.P1, dtype=float)
    seg_len = np.linalg.norm(P1 - P0, axis=1)

    qw = np.asarray(pc["qw"], dtype=float)
    assert np.all(qw >= 0.0)
    assert np.allclose(np.sum(qw, axis=1), seg_len, rtol=0.0, atol=1e-12)

    qp = np.asarray(pc["qp_phys"], dtype=float)
    xi_pos = np.asarray(pc["xi_pos"], dtype=float)
    eta_pos = np.asarray(pc["eta_pos"], dtype=float)
    xi_neg = np.asarray(pc["xi_neg"], dtype=float)
    eta_neg = np.asarray(pc["eta_neg"], dtype=float)
    pos_ids = np.asarray(pc["owner_pos_id"], dtype=int)
    neg_ids = np.asarray(pc["owner_neg_id"], dtype=int)

    # Round-trip mapping at a few sample points (first/last quad point on a few segments).
    for seg in (0, len(P0) // 2, len(P0) - 1):
        for q in (0, int(qp.shape[1]) - 1):
            xq = qp[int(seg), int(q)]
            pe = int(pos_ids[int(seg)])
            ne = int(neg_ids[int(seg)])
            xhat_p = np.asarray(transform.x_mapping(mapping.mesh, pe, (float(xi_pos[seg, q]), float(eta_pos[seg, q]))), dtype=float)
            xhat_n = np.asarray(transform.x_mapping(mapping.mesh, ne, (float(xi_neg[seg, q]), float(eta_neg[seg, q]))), dtype=float)
            assert float(np.linalg.norm(xq - xhat_p)) < 1e-10
            assert float(np.linalg.norm(xq - xhat_n)) < 1e-10


def test_nonmatching_interface_precompute_basis_matches_direct():
    _mesh_pos, _mesh_neg, _interface, mapping, interface_c = _build_interface_pair(degree=2, ny_neg=6, ny_pos=7)

    dh = DofHandler(MixedElement(mapping.mesh, {"u": 2}), method="cg")
    pc = dh.precompute_nonmatching_interface_factors(interface_c, qdeg=6, derivs={(0, 0)})
    r00_p = np.asarray(pc["r00_u_pos"], dtype=float)
    r00_n = np.asarray(pc["r00_u_neg"], dtype=float)

    xi_pos = np.asarray(pc["xi_pos"], dtype=float)
    eta_pos = np.asarray(pc["eta_pos"], dtype=float)
    xi_neg = np.asarray(pc["xi_neg"], dtype=float)
    eta_neg = np.asarray(pc["eta_neg"], dtype=float)

    me = dh.mixed_element
    assert me is not None

    seg = int(r00_p.shape[0] // 2)
    q = int(r00_p.shape[1] // 3)
    phi_p = np.asarray(me.basis("u", float(xi_pos[seg, q]), float(eta_pos[seg, q])), dtype=float).ravel()
    phi_n = np.asarray(me.basis("u", float(xi_neg[seg, q]), float(eta_neg[seg, q])), dtype=float).ravel()

    assert np.allclose(r00_p[seg, q, : phi_p.size], phi_p, atol=1e-12, rtol=0.0)
    assert np.allclose(r00_n[seg, q, : phi_n.size], phi_n, atol=1e-12, rtol=0.0)


def test_composite_mesh_dof_collision_is_prevented():
    mesh_pos, mesh_neg, _interface, mapping, _interface_c = _build_interface_pair(degree=1, ny_neg=6, ny_pos=7)

    dh_pos = DofHandler(MixedElement(mesh_pos, {"u": 1}), method="cg")
    dh_neg = DofHandler(MixedElement(mesh_neg, {"u": 1}), method="cg")
    dh = DofHandler(MixedElement(mapping.mesh, {"u": 1}), method="cg")

    assert int(dh.total_dofs) == int(dh_pos.total_dofs) + int(dh_neg.total_dofs)

    dh._ensure_node_maps()
    dh_pos._ensure_node_maps()
    dh_neg._ensure_node_maps()

    # Pick a coordinate that exists on both meshes (interface midpoint node for Q1).
    target = np.array([0.5, 0.5], dtype=float)

    def _find_node(mesh: Mesh) -> int:
        xy = np.asarray(mesh.nodes_x_y_pos, dtype=float)
        d2 = np.sum((xy - target[None, :]) ** 2, axis=1)
        return int(np.argmin(d2))

    nid_pos = _find_node(mesh_pos)
    nid_neg = _find_node(mesh_neg)
    gd_pos = int(dh_pos.dof_map["u"][nid_pos])
    gd_neg = int(dh_neg.dof_map["u"][nid_neg])

    gd_comp_pos = int(dh.dof_map["u"][int(mapping.pos_node_offset + nid_pos)])
    gd_comp_neg = int(dh.dof_map["u"][int(mapping.neg_node_offset + nid_neg)])
    assert gd_comp_pos != gd_comp_neg
    assert gd_comp_pos in dh.get_field_slice("u")
    assert gd_comp_neg in dh.get_field_slice("u")

    # Also ensure the two submesh DOFs are distinct.
    assert gd_pos != gd_neg


def test_nonmatching_interface_matrix_symmetric():
    _mesh_pos, _mesh_neg, _interface, mapping, interface_c = _build_interface_pair(degree=1, ny_neg=8, ny_pos=11)

    dh = DofHandler(MixedElement(mapping.mesh, {"u": 1}), method="cg")
    u = TrialFunction("u", dof_handler=dh)
    v = TestFunction("u", dof_handler=dh)
    n = FacetNormal()
    h = CellDiameter()

    k_pos = 3.0
    k_neg = 7.0
    denom = k_pos + k_neg
    kappa_pos = Constant(k_neg / denom)
    kappa_neg = Constant(k_pos / denom)
    gamma = Constant(20.0 * denom) / h

    flux_u_pos = -Constant(k_pos) * dot(grad(Pos(u)), n)
    flux_u_neg = -Constant(k_neg) * dot(grad(Neg(u)), n)
    flux_v_pos = -Constant(k_pos) * dot(grad(Pos(v)), n)
    flux_v_neg = -Constant(k_neg) * dot(grad(Neg(v)), n)
    avg_flux_u = kappa_pos * flux_u_pos + kappa_neg * flux_u_neg
    avg_flux_v = kappa_pos * flux_v_pos + kappa_neg * flux_v_neg
    jump_u = Neg(u) - Pos(u)
    jump_v = Neg(v) - Pos(v)

    a_if = (avg_flux_u * jump_v + avg_flux_v * jump_u + gamma * jump_u * jump_v) * dNonmatchingInterface(
        metadata={"q": 8, "interface": interface_c}
    )
    K, _ = assemble_form(Equation(a_if, None), dof_handler=dh, bcs=[], backend="python")
    diff = (K - K.T).tocoo()
    max_abs = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    assert max_abs < 1e-11


def test_nonmatching_interface_crosscheck_ufl_vs_explicit_nitsche():
    # Q1 so we can map DOFs via node ids deterministically.
    mesh_pos, mesh_neg, interface, mapping, interface_c = _build_interface_pair(degree=1, ny_neg=6, ny_pos=9)

    dh_pos = DofHandler(MixedElement(mesh_pos, {"u": 1}), method="cg")
    dh_neg = DofHandler(MixedElement(mesh_neg, {"u": 1}), method="cg")
    dh = DofHandler(MixedElement(mapping.mesh, {"u": 1}), method="cg")

    k_neg = 1.0
    k_pos = 10.0
    gamma = 20.0
    q = 8

    K_exp = assemble_poisson_nitsche_interface_matrix(
        interface=interface,
        dh_neg=dh_neg,
        dh_pos=dh_pos,
        field="u",
        k_neg=k_neg,
        k_pos=k_pos,
        gamma=gamma,
        quad_order=q,
        backend="python",
    ).tocsr()

    # Assemble the same interface bilinear form in UFL on the composite mesh.
    u = TrialFunction("u", dof_handler=dh)
    v = TestFunction("u", dof_handler=dh)
    n = FacetNormal()
    h = CellDiameter()
    denom = k_pos + k_neg
    kappa_pos = Constant(k_neg / denom)
    kappa_neg = Constant(k_pos / denom)
    stab = Constant(gamma * denom) / h

    flux_u_pos = -Constant(k_pos) * dot(grad(Pos(u)), n)
    flux_u_neg = -Constant(k_neg) * dot(grad(Neg(u)), n)
    flux_v_pos = -Constant(k_pos) * dot(grad(Pos(v)), n)
    flux_v_neg = -Constant(k_neg) * dot(grad(Neg(v)), n)
    avg_flux_u = kappa_pos * flux_u_pos + kappa_neg * flux_u_neg
    avg_flux_v = kappa_pos * flux_v_pos + kappa_neg * flux_v_neg
    jump_u = Neg(u) - Pos(u)
    jump_v = Neg(v) - Pos(v)
    a_if = (avg_flux_u * jump_v + avg_flux_v * jump_u + stab * jump_u * jump_v) * dNonmatchingInterface(
        metadata={"q": q, "interface": interface_c}
    )
    K_ufl, _ = assemble_form(Equation(a_if, None), dof_handler=dh, bcs=[], backend="python")
    K_ufl = K_ufl.tocsr()

    # Build explicit->composite permutation from node ids (valid for Q1).
    dh_pos._ensure_node_maps()
    dh_neg._ensure_node_maps()
    dh._ensure_node_maps()

    n_pos = int(dh_pos.total_dofs)
    n_neg = int(dh_neg.total_dofs)
    n_tot = n_pos + n_neg

    old_for_new = -np.ones(n_tot, dtype=int)

    for nid, gd_pos in dh_pos.dof_map["u"].items():
        gd_pos = int(gd_pos)
        gd_comp = int(dh.dof_map["u"][int(mapping.pos_node_offset + int(nid))])
        old_for_new[gd_comp] = gd_pos

    for nid, gd_neg in dh_neg.dof_map["u"].items():
        gd_neg = int(gd_neg)
        gd_comp = int(dh.dof_map["u"][int(mapping.neg_node_offset + int(nid))])
        old_for_new[gd_comp] = n_pos + gd_neg

    assert np.all(old_for_new >= 0), "Failed to build a full DOF permutation (expected for Q1)."

    K_exp_reordered = K_exp[old_for_new, :][:, old_for_new].tocsr()
    diff = (K_exp_reordered - K_ufl).tocoo()
    max_abs = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    assert max_abs < 1e-10


def test_nonmatching_interface_empty_selection_errors():
    mesh_neg = _make_submesh_quad(poly_order=1, nx=4, ny=4, offset_x=0.0)
    mesh_pos = _make_submesh_quad(poly_order=1, nx=4, ny=4, offset_x=0.5)
    mesh_neg.tag_boundary_edges({"boundary": lambda x, y: True})
    mesh_pos.tag_boundary_edges({"boundary": lambda x, y: True})
    with pytest.raises(ValueError):
        build_nonmatching_interface(mesh_neg=mesh_neg, mesh_pos=mesh_pos)


def test_nonmatching_interface_precompute_reorients_flipped_normals():
    _mesh_pos, _mesh_neg, interface, mapping, interface_c = _build_interface_pair(degree=1, ny_neg=6, ny_pos=8)

    # Deliberately flip normals in the interface description; precompute must
    # reorient them using owner-element centroids.
    interface_bad = NonMatchingInterface(
        mesh_neg=interface_c.mesh_neg,
        mesh_pos=interface_c.mesh_pos,
        neg_edge_ids=np.asarray(interface_c.neg_edge_ids, dtype=int),
        pos_edge_ids=np.asarray(interface_c.pos_edge_ids, dtype=int),
        neg_elem_ids=np.asarray(interface_c.neg_elem_ids, dtype=int),
        pos_elem_ids=np.asarray(interface_c.pos_elem_ids, dtype=int),
        P0=np.asarray(interface_c.P0, dtype=float),
        P1=np.asarray(interface_c.P1, dtype=float),
        n=-np.asarray(interface_c.n, dtype=float),
        h_neg=np.asarray(interface_c.h_neg, dtype=float),
        h_pos=np.asarray(interface_c.h_pos, dtype=float),
    )

    dh = DofHandler(MixedElement(mapping.mesh, {"u": 1}), method="cg")
    pc = dh.precompute_nonmatching_interface_factors(interface_bad, qdeg=6, derivs={(0, 0)})

    n0 = np.asarray(pc["normals"][:, 0, :], dtype=float)
    cpos = np.asarray([mapping.mesh.elements_list[int(eid)].centroid() for eid in interface_bad.pos_elem_ids], dtype=float)
    cneg = np.asarray([mapping.mesh.elements_list[int(eid)].centroid() for eid in interface_bad.neg_elem_ids], dtype=float)
    dotv = np.einsum("ij,ij->i", n0, cpos - cneg)
    assert np.all(dotv > 0.0)

