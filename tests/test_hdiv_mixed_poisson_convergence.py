import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem import transform
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.integration.quadrature import volume
from pycutfem.utils.meshgen import structured_quad


def _assemble_and_solve_mixed_poisson_rt1_dg1(nx: int):
    """
    Solve mixed Poisson on the unit square with the standard RT1-DG1 pair.

    PDE (manufactured):
        u + grad(p) = 0
        div(u) = g
        p = 0 on ∂Ω  (encoded by omitting the boundary term in the mixed weak form)

    Exact:
        p = sin(pi x) sin(pi y)
        u = -grad(p)
        g = div(u) = -Δp
    """
    nodes, elem_conn, edges, corner = structured_quad(1.0, 1.0, nx=nx, ny=nx, poly_order=1)
    mesh = Mesh(
        nodes,
        elem_conn,
        edges_connectivity=edges,
        elements_corner_nodes=corner,
        element_type="quad",
        poly_order=1,
    )

    me = MixedElement(mesh, {"u": ("RT", 1), "p": ("DG", 1)})
    dh = DofHandler(me, method="cg")

    qp, qw = volume("quad", order=4)
    qp = np.asarray(qp, dtype=float)
    qw = np.asarray(qw, dtype=float)

    pi = float(np.pi)

    def p_exact(x: float, y: float) -> float:
        return float(np.sin(pi * x) * np.sin(pi * y))

    def u_exact(x: float, y: float) -> np.ndarray:
        return np.array(
            [
                -pi * np.cos(pi * x) * np.sin(pi * y),
                -pi * np.sin(pi * x) * np.cos(pi * y),
            ],
            dtype=float,
        )

    def g_exact(x: float, y: float) -> float:
        return float(2.0 * pi * pi * np.sin(pi * x) * np.sin(pi * y))

    n = int(dh.total_dofs)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(n, dtype=float)

    for eid in range(mesh.num_elements()):
        u_g = np.asarray(dh.element_maps["u"][int(eid)], dtype=int)
        sgn_u = np.asarray(dh.element_signs["u"][int(eid)], dtype=float)
        p_g = np.asarray(dh.element_maps["p"][int(eid)], dtype=int)

        nu = int(u_g.size)
        np_ = int(p_g.size)

        Auu = np.zeros((nu, nu), dtype=float)
        B = np.zeros((np_, nu), dtype=float)
        Fp = np.zeros(np_, dtype=float)

        for (xi, eta), w in zip(qp, qw):
            xi = float(xi)
            eta = float(eta)
            J = np.asarray(transform.jacobian(mesh, int(eid), (xi, eta)), dtype=float)
            detJ = float(np.linalg.det(J))
            w_phys = float(w) * abs(detJ)

            X = np.asarray(transform.x_mapping(mesh, int(eid), (xi, eta)), dtype=float)
            x = float(X[0])
            y = float(X[1])

            Vu = np.asarray(me.tabulate_value("u", xi, eta, element_id=int(eid)), dtype=float)
            Vu = sgn_u[:, None] * Vu
            divu = np.asarray(me.tabulate_div("u", xi, eta, element_id=int(eid)), dtype=float).ravel()
            divu = sgn_u * divu
            phip = np.asarray(me._ref["p"].shape(xi, eta), dtype=float).ravel()

            Auu += w_phys * (Vu @ Vu.T)
            B += w_phys * (phip[:, None] * divu[None, :])
            Fp += w_phys * phip * g_exact(x, y)

        # Assemble local blocks into the global saddle-point system:
        #   [ A   -B^T ] [u] = [0]
        #   [ B    0  ] [p]   [g]
        for i in range(nu):
            gi = int(u_g[i])
            for j in range(nu):
                rows.append(gi)
                cols.append(int(u_g[j]))
                data.append(float(Auu[i, j]))

        for q in range(np_):
            gq = int(p_g[q])
            rhs[gq] += float(Fp[q])
            for i in range(nu):
                gi = int(u_g[i])
                val = float(B[q, i])
                rows.append(gq)
                cols.append(gi)
                data.append(val)
                rows.append(gi)
                cols.append(gq)
                data.append(-val)

    K = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    sol = spla.spsolve(K, rhs)

    # L2 errors of u and p
    err_u = 0.0
    err_p = 0.0
    for eid in range(mesh.num_elements()):
        u_g = np.asarray(dh.element_maps["u"][int(eid)], dtype=int)
        sgn_u = np.asarray(dh.element_signs["u"][int(eid)], dtype=float)
        p_g = np.asarray(dh.element_maps["p"][int(eid)], dtype=int)
        u_loc = np.asarray(sol[u_g], dtype=float) * sgn_u
        p_loc = np.asarray(sol[p_g], dtype=float)

        for (xi, eta), w in zip(qp, qw):
            xi = float(xi)
            eta = float(eta)
            J = np.asarray(transform.jacobian(mesh, int(eid), (xi, eta)), dtype=float)
            detJ = float(np.linalg.det(J))
            w_phys = float(w) * abs(detJ)

            X = np.asarray(transform.x_mapping(mesh, int(eid), (xi, eta)), dtype=float)
            x = float(X[0])
            y = float(X[1])

            Vu = np.asarray(me.tabulate_value("u", xi, eta, element_id=int(eid)), dtype=float)
            uh = np.asarray(u_loc @ Vu, dtype=float)
            ue = u_exact(x, y)
            err_u += w_phys * float(np.dot(uh - ue, uh - ue))

            phip = np.asarray(me._ref["p"].shape(xi, eta), dtype=float).ravel()
            ph = float(p_loc @ phip)
            pe = p_exact(x, y)
            err_p += w_phys * float((ph - pe) ** 2)

    h = 1.0 / float(nx)
    return h, float(np.sqrt(err_u)), float(np.sqrt(err_p))


def test_mixed_poisson_convergence_rt1_dg1_quads():
    mesh_ns = [2, 4, 8]
    hs: list[float] = []
    eu: list[float] = []
    ep: list[float] = []

    for nx in mesh_ns:
        h, err_u, err_p = _assemble_and_solve_mixed_poisson_rt1_dg1(nx)
        hs.append(h)
        eu.append(err_u)
        ep.append(err_p)

    hs = np.asarray(hs, dtype=float)
    eu = np.asarray(eu, dtype=float)
    ep = np.asarray(ep, dtype=float)

    rate_u = np.diff(np.log(eu)) / np.diff(np.log(hs))
    rate_p = np.diff(np.log(ep)) / np.diff(np.log(hs))

    # Expect ~O(h^2) for RT1 and DG1 in L2 on affine quads.
    assert np.all(np.isfinite(rate_u))
    assert np.all(np.isfinite(rate_p))
    assert np.all(rate_u > 1.7)
    assert np.all(rate_u < 2.3)
    assert np.all(rate_p > 1.7)
    assert np.all(rate_p < 2.3)

