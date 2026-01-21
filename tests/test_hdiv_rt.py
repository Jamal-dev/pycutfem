import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem import transform
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.fem.reference import get_hdiv_reference
from pycutfem.integration.quadrature import line_quadrature, volume
from pycutfem.utils.meshgen import structured_quad, structured_triangles


@pytest.mark.parametrize("etype", ["tri", "quad"])
@pytest.mark.parametrize("k", [0, 1, 2])
def test_rt_reference_unisolvency(etype, k):
    rt = get_hdiv_reference(etype, k)
    A = np.asarray(rt.A, dtype=float)
    C = np.asarray(rt.C, dtype=float)
    I = A @ C
    err = float(np.linalg.norm(I - np.eye(rt.n_dofs), ord=np.inf))
    assert err < 1e-12


def _rt_interpolate_vector_field_on_edges(dh: DofHandler, field: str, u_exact, *, qdeg: int = 6) -> np.ndarray:
    """
    Construct global RT coefficients by matching edge flux moment DOFs against `u_exact`.

    This is the canonical RT interpolant for k=0 (and sets edge moments for k>=1).
    """
    from numpy.polynomial.legendre import legval
    from pycutfem.integration.quadrature import gauss_legendre

    if dh.mixed_element is None:
        raise RuntimeError("Expected MixedElement-backed DofHandler.")
    me = dh.mixed_element
    fam = getattr(me, "_field_families", {}).get(field, None)
    if fam != "RT":
        raise ValueError(f"Field '{field}' is not RT (got {fam}).")

    info = getattr(dh, "_hdiv_field_info", {}).get(field, None)
    if not isinstance(info, dict):
        raise RuntimeError(f"Missing H(div) numbering info for field '{field}'.")
    k = int(info["k"])
    n_edge = int(info["n_edge_dofs"])
    if n_edge != k + 1:
        raise RuntimeError("Internal error: RT edge mode count mismatch.")

    U = np.zeros(dh.total_dofs, dtype=float)

    s, w_ref = gauss_legendre(int(qdeg))
    s = np.asarray(s, dtype=float)
    w_ref = np.asarray(w_ref, dtype=float)

    for e in dh.mixed_element.mesh.edges_list:
        edge_gid = int(e.gid)
        n_global = np.asarray(e.normal, dtype=float)
        p0 = dh.mixed_element.mesh.nodes_x_y_pos[int(e.nodes[0])]
        p1 = dh.mixed_element.mesh.nodes_x_y_pos[int(e.nodes[1])]
        mid = 0.5 * (p0 + p1)
        half = 0.5 * (p1 - p0)
        qpts = mid[None, :] + s[:, None] * half[None, :]
        wts = w_ref * float(np.linalg.norm(half))

        uvals = np.asarray([u_exact(xy) for xy in qpts], dtype=float)  # (nq,2)
        flux = uvals @ n_global  # (nq,)

        # edge trace DOFs are moments against Legendre P_m(s) on [-1,1]
        dofs = dh.edge_trace_dofs(field, edge_gid)
        if len(dofs) != n_edge:
            raise RuntimeError("Internal error: edge_trace_dofs length mismatch.")
        for m in range(n_edge):
            coeff = np.zeros(m + 1, dtype=float)
            coeff[-1] = 1.0
            Pm = legval(s, coeff)
            U[int(dofs[m])] = float(np.sum(wts * flux * Pm))

    return U


def test_rt_patch_constant_divergence_affine_quads():
    # Mesh: 2x2 quads, globally affine-distorted (each element remains affine).
    nodes, elem_conn, edges, corner = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1)
    A = np.array([[1.0, 0.35], [0.15, 1.0]], dtype=float)
    for nd in nodes:
        x = np.array([nd.x, nd.y], dtype=float)
        xp = A @ x
        nd.x = float(xp[0])
        nd.y = float(xp[1])
    mesh = Mesh(
        nodes,
        elem_conn,
        edges_connectivity=edges,
        elements_corner_nodes=corner,
        element_type="quad",
        poly_order=1,
    )

    def u_exact(xy):
        x, y = float(xy[0]), float(xy[1])
        return np.array([x, y], dtype=float)

    me = MixedElement(mesh, {"u": ("RT", 0)})
    dh = DofHandler(me, method="cg")
    U = _rt_interpolate_vector_field_on_edges(dh, "u", u_exact, qdeg=6)

    # Evaluate div(u_h) at multiple reference points in every element.
    pts = [(-0.5, -0.3), (0.2, -0.1), (0.1, 0.4), (-0.4, 0.6)]
    for eid in range(mesh.num_elements()):
        loc = np.asarray(dh.element_maps["u"][int(eid)], dtype=int)
        sgn = np.asarray(dh.element_signs["u"][int(eid)], dtype=float)
        u_loc = U[loc] * sgn
        for xi, eta in pts:
            divV = me.tabulate_div("u", float(xi), float(eta), element_id=int(eid))
            div_uh = float(u_loc @ divV)
            assert abs(div_uh - 2.0) < 1e-12


def test_rt_flux_continuity_two_triangles_constant_flow():
    # Mesh: 2 triangles sharing one edge.
    nodes, elem_conn, edges, corner = structured_triangles(1.0, 1.0, nx_quads=1, ny_quads=1, poly_order=1)
    mesh = Mesh(
        nodes,
        elem_conn,
        edges_connectivity=edges,
        elements_corner_nodes=corner,
        element_type="tri",
        poly_order=1,
    )
    interior_edges = [e for e in mesh.edges_list if (e.left is not None and e.right is not None)]
    assert interior_edges
    E = interior_edges[0]
    eL, eR = int(E.left), int(E.right)
    n_edge = np.asarray(E.normal, float)  # shared normal (left-oriented)

    def u_exact(_xy):
        return np.array([1.0, 0.0], dtype=float)

    me = MixedElement(mesh, {"u": ("RT", 0)})
    dh = DofHandler(me, method="cg")
    U = _rt_interpolate_vector_field_on_edges(dh, "u", u_exact, qdeg=4)

    def eval_u_on_elem(eid: int, xi: float, eta: float) -> np.ndarray:
        loc = np.asarray(dh.element_maps["u"][int(eid)], dtype=int)
        sgn = np.asarray(dh.element_signs["u"][int(eid)], dtype=float)
        u_loc = U[loc] * sgn
        V = me.tabulate_value("u", float(xi), float(eta), element_id=int(eid))
        return u_loc @ V

    p0 = mesh.nodes_x_y_pos[int(E.nodes[0])]
    p1 = mesh.nodes_x_y_pos[int(E.nodes[1])]
    qpts, _wts = line_quadrature(p0, p1, order=6)
    for xq in qpts:
        xiL, etaL = transform.inverse_mapping(mesh, int(eL), np.asarray(xq, float))
        xiR, etaR = transform.inverse_mapping(mesh, int(eR), np.asarray(xq, float))
        uL = eval_u_on_elem(eL, float(xiL), float(etaL))
        uR = eval_u_on_elem(eR, float(xiR), float(etaR))
        flux_L = float(uL @ n_edge)
        flux_R = float(uR @ n_edge)
        assert abs(flux_L - flux_R) < 1e-12


@pytest.mark.parametrize("etype", ["tri", "quad"])
@pytest.mark.parametrize("k", [0, 1, 2])
def test_hdiv_normal_continuity_and_gauss_theorem(etype, k):
    # --- 1) build a minimal 2-element patch with one interior edge ---
    if etype == "quad":
        nodes, elem_conn, edges, corner = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=1)
        mesh = Mesh(
            nodes,
            elem_conn,
            edges_connectivity=edges,
            elements_corner_nodes=corner,
            element_type="quad",
            poly_order=1,
        )
    else:
        nodes, elem_conn, edges, corner = structured_triangles(1.0, 1.0, nx_quads=1, ny_quads=1, poly_order=1)
        mesh = Mesh(
            nodes,
            elem_conn,
            edges_connectivity=edges,
            elements_corner_nodes=corner,
            element_type="tri",
            poly_order=1,
        )

    interior_edges = [e for e in mesh.edges_list if (e.left is not None and e.right is not None)]
    assert interior_edges
    E = interior_edges[0]
    e_gid = int(E.gid)
    eL, eR = int(E.left), int(E.right)

    # --- 2) Mixed space: u in RT_k (Hdiv), p in DG_k (L2-like) ---
    me = MixedElement(
        mesh,
        {
            "u": ("RT", k),
            "p": ("DG", k),
        },
    )
    dh = DofHandler(me, method="cg")

    # --- 3) random global coefficients for u only; p left at zero ---
    U = np.zeros(dh.total_dofs, float)
    u_gids = np.asarray(dh.get_field_slice("u"), dtype=int)
    rng = np.random.default_rng(0)
    U[u_gids] = rng.standard_normal(len(u_gids))

    def eval_u_on_elem(eid: int, xi: float, eta: float) -> np.ndarray:
        loc = np.asarray(dh.element_maps["u"][int(eid)], dtype=int)
        sgn = np.asarray(dh.element_signs["u"][int(eid)], dtype=float)
        u_loc = U[loc] * sgn
        V = me.tabulate_value("u", float(xi), float(eta), element_id=int(eid))  # (n_loc,2)
        return u_loc @ V  # (2,)

    def eval_div_on_elem(eid: int, xi: float, eta: float) -> float:
        loc = np.asarray(dh.element_maps["u"][int(eid)], dtype=int)
        sgn = np.asarray(dh.element_signs["u"][int(eid)], dtype=float)
        u_loc = U[loc] * sgn
        divV = me.tabulate_div("u", float(xi), float(eta), element_id=int(eid))  # (n_loc,)
        return float(u_loc @ divV)

    # --- 4) NORMAL CONTINUITY on the shared edge ---
    # Evaluate both traces at the *same physical points* on the shared edge.
    n_edge = np.asarray(E.normal, float)
    p0 = mesh.nodes_x_y_pos[int(E.nodes[0])]
    p1 = mesh.nodes_x_y_pos[int(E.nodes[1])]
    qpts, _wts = line_quadrature(p0, p1, order=8)
    for xq in qpts:
        xiL, etaL = transform.inverse_mapping(mesh, int(eL), np.asarray(xq, float))
        xiR, etaR = transform.inverse_mapping(mesh, int(eR), np.asarray(xq, float))

        uL = eval_u_on_elem(eL, float(xiL), float(etaL))
        uR = eval_u_on_elem(eR, float(xiR), float(etaR))

        nL = n_edge
        nR = -n_edge
        jump = float(uL @ nL + uR @ nR)
        assert abs(jump) < 1e-9, f"normal jump too large: {jump}"

    # --- 5) GAUSS THEOREM on each element: ∫_K div u dx ≈ ∫_{∂K} u·n ds ---
    qdeg_vol = int(max(4, 2 * int(k) + 4))
    qdeg_edge = int(max(6, 2 * int(k) + 4))
    qp, qw = volume(mesh.element_type, order=qdeg_vol)
    qp = np.asarray(qp, dtype=float)
    qw = np.asarray(qw, dtype=float)

    for eid in [eL, eR]:
        div_int = 0.0
        for (xi, eta), w in zip(qp, qw):
            J = np.asarray(transform.jacobian(mesh, int(eid), (float(xi), float(eta))), dtype=float)
            detJ = float(np.linalg.det(J))
            div_int += float(w) * eval_div_on_elem(int(eid), float(xi), float(eta)) * abs(detJ)

        flux_int = 0.0
        elem = mesh.elements_list[int(eid)]
        for edge_gid in elem.edges:
            if int(edge_gid) < 0:
                continue
            edge = mesh.edge(int(edge_gid))
            n_out = np.asarray(edge.normal, float) if int(edge.left) == int(eid) else -np.asarray(edge.normal, float)
            p0 = mesh.nodes_x_y_pos[int(edge.nodes[0])]
            p1 = mesh.nodes_x_y_pos[int(edge.nodes[1])]
            qpts, wts = line_quadrature(p0, p1, order=qdeg_edge)
            for xq, w in zip(qpts, wts):
                xi, eta = transform.inverse_mapping(mesh, int(eid), np.asarray(xq, float))
                u = eval_u_on_elem(int(eid), float(xi), float(eta))
                flux_int += float(w) * float(u @ n_out)

        assert abs(div_int - flux_int) < 1e-8, f"Gauss theorem mismatch on element {eid}"


@pytest.mark.parametrize("k", [0, 1, 2])
def test_hdiv_gauss_theorem_isoparametric_quad(k):
    # Single quadratic (Q2) quad with a non-affine mapping (interior node moved).
    nodes, elem_conn, edges, corner = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=2)
    for nd in nodes:
        if abs(float(nd.x) - 0.5) < 1e-12 and abs(float(nd.y) - 0.5) < 1e-12:
            nd.y = float(nd.y) + 0.15
            break

    mesh = Mesh(
        nodes,
        elem_conn,
        edges_connectivity=edges,
        elements_corner_nodes=corner,
        element_type="quad",
        poly_order=2,
    )

    me = MixedElement(mesh, {"u": ("RT", int(k))})
    dh = DofHandler(me, method="cg")

    rng = np.random.default_rng(0)
    U = rng.standard_normal(int(dh.total_dofs))

    eid = 0
    loc = np.asarray(dh.element_maps["u"][eid], dtype=int)
    sgn = np.asarray(dh.element_signs["u"][eid], dtype=float)
    u_loc = U[loc] * sgn

    qdeg_vol = int(max(12, 2 * int(k) + 10))
    qp, qw = volume("quad", order=qdeg_vol)
    qp = np.asarray(qp, dtype=float)
    qw = np.asarray(qw, dtype=float)

    # Volume integral of div(u_h)
    div_int = 0.0
    min_det = np.inf
    for (xi, eta), w in zip(qp, qw):
        xi = float(xi)
        eta = float(eta)
        J = np.asarray(transform.jacobian(mesh, eid, (xi, eta)), dtype=float)
        detJ = float(np.linalg.det(J))
        min_det = min(min_det, detJ)
        divV = np.asarray(me.tabulate_div("u", xi, eta, element_id=eid), dtype=float).ravel()
        div_int += float(w) * float(u_loc @ divV) * abs(detJ)
    assert min_det > 0.0

    # Boundary flux integral of u_h · n over ∂K (edges remain straight for Q2 when only interior node moves).
    qdeg_edge = int(max(16, 2 * int(k) + 12))
    flux_int = 0.0
    elem = mesh.elements_list[eid]
    for edge_gid in elem.edges:
        edge = mesh.edge(int(edge_gid))
        n_out = np.asarray(edge.normal, float) if int(edge.left) == eid else -np.asarray(edge.normal, float)
        p0, p1 = mesh.nodes_x_y_pos[list(edge.nodes)]
        qpts, wts = line_quadrature(p0, p1, order=qdeg_edge)
        for xq, w in zip(qpts, wts):
            xi, eta = transform.inverse_mapping(mesh, eid, np.asarray(xq, float))
            V = np.asarray(me.tabulate_value("u", float(xi), float(eta), element_id=eid), dtype=float)
            u = np.asarray(u_loc @ V, dtype=float)
            flux_int += float(w) * float(u @ n_out)

    assert abs(div_int - flux_int) < 5e-9
