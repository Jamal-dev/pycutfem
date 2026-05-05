from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem import transform
from pycutfem.integration.quadrature import gauss_legendre

from .interface import NonMatchingInterface
from .mortar import MortarCoupling
from .nitsche import _field_order


try:  # pragma: no cover - exercised when numba is available
    import numba as nb

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    nb = None
    _HAVE_NUMBA = False


def _require_numba() -> None:
    if not _HAVE_NUMBA:  # pragma: no cover
        raise RuntimeError("backend='jit' requires numba to be installed.")


def _interface_quadrature(interface: NonMatchingInterface, quad_order: int) -> tuple[np.ndarray, np.ndarray]:
    xi, w_ref = gauss_legendre(int(quad_order))
    xi = np.asarray(xi, dtype=float)
    w_ref = np.asarray(w_ref, dtype=float)

    P0 = np.asarray(interface.P0, dtype=float)
    P1 = np.asarray(interface.P1, dtype=float)
    mid = 0.5 * (P0 + P1)
    half = 0.5 * (P1 - P0)
    seg_J = np.linalg.norm(half, axis=1)
    qpts = mid[:, None, :] + xi[None, :, None] * half[:, None, :]
    qw = w_ref[None, :] * seg_J[:, None]
    return qpts, qw


def assemble_poisson_nitsche_interface_matrix(
    *,
    interface: NonMatchingInterface,
    dh_neg: DofHandler,
    dh_pos: DofHandler,
    field: str = "u",
    k_neg: float = 1.0,
    k_pos: float = 1.0,
    gamma: float = 20.0,
    quad_order: int | None = None,
) -> sp.csr_matrix:
    _require_numba()

    if interface.n_segments() <= 0:
        n_total = int(dh_pos.total_dofs) + int(dh_neg.total_dofs)
        return sp.csr_matrix((n_total, n_total), dtype=float)

    if quad_order is None:
        p_pos = _field_order(dh_pos, field)
        p_neg = _field_order(dh_neg, field)
        quad_order = int(2 * max(p_pos, p_neg) + 2)

    qpts, qw = _interface_quadrature(interface, int(quad_order))

    k_pos = float(k_pos)
    k_neg = float(k_neg)
    denom = k_pos + k_neg
    if denom <= 0.0:
        raise ValueError("k_pos + k_neg must be > 0 for Nitsche weights.")
    kappa_pos = k_neg / denom
    kappa_neg = k_pos / denom

    n_pos = int(dh_pos.total_dofs)
    n_neg = int(dh_neg.total_dofs)
    n_total = n_pos + n_neg

    me_pos = dh_pos.mixed_element
    me_neg = dh_neg.mixed_element
    mesh_pos = interface.mesh_pos
    mesh_neg = interface.mesh_neg

    n_seg = interface.n_segments()
    # Precompute gdofs (assume fixed local size).
    gd_pos0 = np.asarray(dh_pos.get_elemental_dofs(int(interface.pos_elem_ids[0])), dtype=np.int64)
    gd_neg0 = np.asarray(dh_neg.get_elemental_dofs(int(interface.neg_elem_ids[0])), dtype=np.int64)
    nlp = int(gd_pos0.size)
    nln = int(gd_neg0.size)

    gd_pos = np.empty((n_seg, nlp), dtype=np.int64)
    gd_neg = np.empty((n_seg, nln), dtype=np.int64)
    stab = np.empty(n_seg, dtype=np.float64)

    phi_p = np.zeros((n_seg, qpts.shape[1], nlp), dtype=np.float64)
    gn_p = np.zeros_like(phi_p)
    phi_n = np.zeros((n_seg, qpts.shape[1], nln), dtype=np.float64)
    gn_n = np.zeros_like(phi_n)

    for s in range(n_seg):
        pos_eid = int(interface.pos_elem_ids[s])
        neg_eid = int(interface.neg_elem_ids[s])
        gd_p = np.asarray(dh_pos.get_elemental_dofs(pos_eid), dtype=np.int64)
        gd_n = np.asarray(dh_neg.get_elemental_dofs(neg_eid), dtype=np.int64)
        if gd_p.size != nlp or gd_n.size != nln:
            raise ValueError("JIT backend currently requires constant local DOF counts per segment.")
        gd_pos[s, :] = gd_p
        gd_neg[s, :] = gd_n + n_pos

        h = float(min(interface.h_pos[s], interface.h_neg[s]) or 0.0)
        if h <= 0.0:
            h = 1.0
        stab[s] = float(gamma) * (k_pos + k_neg) / h

        n_vec = np.asarray(interface.n[s], dtype=float)
        for q in range(qpts.shape[1]):
            xq = np.asarray(qpts[s, q], dtype=float)
            wq = float(qw[s, q])
            if wq == 0.0:
                continue

            xi_p, eta_p = transform.inverse_mapping(mesh_pos, pos_eid, xq)
            Jp = transform.jacobian(mesh_pos, pos_eid, (float(xi_p), float(eta_p)))
            Jip = np.linalg.inv(Jp)
            phi = np.asarray(me_pos.basis(field, float(xi_p), float(eta_p)), dtype=float).ravel()
            grad = np.asarray(me_pos.grad_basis(field, float(xi_p), float(eta_p)), dtype=float) @ Jip
            phi_p[s, q, :] = phi
            gn_p[s, q, :] = grad @ n_vec

            xi_nq, eta_nq = transform.inverse_mapping(mesh_neg, neg_eid, xq)
            Jn = transform.jacobian(mesh_neg, neg_eid, (float(xi_nq), float(eta_nq)))
            Jin = np.linalg.inv(Jn)
            phi = np.asarray(me_neg.basis(field, float(xi_nq), float(eta_nq)), dtype=float).ravel()
            grad = np.asarray(me_neg.grad_basis(field, float(xi_nq), float(eta_nq)), dtype=float) @ Jin
            phi_n[s, q, :] = phi
            gn_n[s, q, :] = grad @ n_vec

    rows, cols, data = _poisson_nitsche_triplets(
        gd_pos, gd_neg, phi_p, gn_p, phi_n, gn_n, qw, stab, kappa_pos, kappa_neg, k_pos, k_neg
    )
    K = sp.coo_matrix((data, (rows, cols)), shape=(n_total, n_total))
    K.sum_duplicates()
    return K.tocsr()


def assemble_stokes_nitsche_interface_matrix(
    *,
    interface: NonMatchingInterface,
    dh_neg: DofHandler,
    dh_pos: DofHandler,
    mu_neg: float,
    mu_pos: float,
    gamma: float = 20.0,
    quad_order: int | None = None,
    u_fields: tuple[str, str] = ("ux", "uy"),
    p_field: str = "p",
) -> sp.csr_matrix:
    _require_numba()

    if interface.n_segments() <= 0:
        n_total = int(dh_pos.total_dofs) + int(dh_neg.total_dofs)
        return sp.csr_matrix((n_total, n_total), dtype=float)

    mu_pos = float(mu_pos)
    mu_neg = float(mu_neg)
    denom = mu_pos + mu_neg
    if denom <= 0.0:
        raise ValueError("mu_pos + mu_neg must be > 0.")
    kappa_pos = mu_neg / denom
    kappa_neg = mu_pos / denom

    if quad_order is None:
        p_u = max(_field_order(dh_pos, u_fields[0]), _field_order(dh_neg, u_fields[0]))
        quad_order = int(2 * p_u + 2)

    qpts, qw = _interface_quadrature(interface, int(quad_order))

    n_pos = int(dh_pos.total_dofs)
    n_neg = int(dh_neg.total_dofs)
    n_total = n_pos + n_neg

    me_pos = dh_pos.mixed_element
    me_neg = dh_neg.mixed_element
    mesh_pos = interface.mesh_pos
    mesh_neg = interface.mesh_neg

    n_seg = interface.n_segments()
    gd_pos0 = np.asarray(dh_pos.get_elemental_dofs(int(interface.pos_elem_ids[0])), dtype=np.int64)
    gd_neg0 = np.asarray(dh_neg.get_elemental_dofs(int(interface.neg_elem_ids[0])), dtype=np.int64)
    nlp = int(gd_pos0.size)
    nln = int(gd_neg0.size)

    gd_pos = np.empty((n_seg, nlp), dtype=np.int64)
    gd_neg = np.empty((n_seg, nln), dtype=np.int64)
    stab = np.empty(n_seg, dtype=np.float64)

    Uvec_p = np.zeros((n_seg, qpts.shape[1], nlp, 2), dtype=np.float64)
    Uvec_n = np.zeros((n_seg, qpts.shape[1], nln, 2), dtype=np.float64)
    T_p = np.zeros((n_seg, qpts.shape[1], nlp, 2), dtype=np.float64)
    T_n = np.zeros((n_seg, qpts.shape[1], nln, 2), dtype=np.float64)

    for s in range(n_seg):
        pos_eid = int(interface.pos_elem_ids[s])
        neg_eid = int(interface.neg_elem_ids[s])
        gd_p = np.asarray(dh_pos.get_elemental_dofs(pos_eid), dtype=np.int64)
        gd_n = np.asarray(dh_neg.get_elemental_dofs(neg_eid), dtype=np.int64)
        if gd_p.size != nlp or gd_n.size != nln:
            raise ValueError("JIT backend currently requires constant local DOF counts per segment.")
        gd_pos[s, :] = gd_p
        gd_neg[s, :] = gd_n + n_pos

        h = float(min(interface.h_pos[s], interface.h_neg[s]) or 0.0)
        if h <= 0.0:
            h = 1.0
        stab[s] = float(gamma) * (mu_pos + mu_neg) / h

        n_vec = np.asarray(interface.n[s], dtype=float)
        nx, ny = float(n_vec[0]), float(n_vec[1])
        for q in range(qpts.shape[1]):
            xq = np.asarray(qpts[s, q], dtype=float)
            wq = float(qw[s, q])
            if wq == 0.0:
                continue

            # POS
            xi_p, eta_p = transform.inverse_mapping(mesh_pos, pos_eid, xq)
            Jp = transform.jacobian(mesh_pos, pos_eid, (float(xi_p), float(eta_p)))
            Jip = np.linalg.inv(Jp)
            phi_ux_p = np.asarray(me_pos.basis(u_fields[0], float(xi_p), float(eta_p)), dtype=float).ravel()
            phi_uy_p = np.asarray(me_pos.basis(u_fields[1], float(xi_p), float(eta_p)), dtype=float).ravel()
            phi_p_p = np.asarray(me_pos.basis(p_field, float(xi_p), float(eta_p)), dtype=float).ravel()
            g_ux_p = np.asarray(me_pos.grad_basis(u_fields[0], float(xi_p), float(eta_p)), dtype=float) @ Jip
            g_uy_p = np.asarray(me_pos.grad_basis(u_fields[1], float(xi_p), float(eta_p)), dtype=float) @ Jip

            Uvec_p[s, q, :, 0] = phi_ux_p
            Uvec_p[s, q, :, 1] = phi_uy_p

            dux_dx = g_ux_p[:, 0]
            dux_dy = g_ux_p[:, 1]
            duy_dx = g_uy_p[:, 0]
            duy_dy = g_uy_p[:, 1]
            tp_x = np.zeros(nlp, dtype=np.float64)
            tp_y = np.zeros(nlp, dtype=np.float64)
            tp_x += 2.0 * mu_pos * (dux_dx * nx + 0.5 * dux_dy * ny)
            tp_y += 2.0 * mu_pos * (0.5 * dux_dy * nx)
            tp_x += 2.0 * mu_pos * (0.5 * duy_dx * ny)
            tp_y += 2.0 * mu_pos * (0.5 * duy_dx * nx + duy_dy * ny)
            tp_x += -phi_p_p * nx
            tp_y += -phi_p_p * ny
            T_p[s, q, :, 0] = tp_x
            T_p[s, q, :, 1] = tp_y

            # NEG
            xi_nq, eta_nq = transform.inverse_mapping(mesh_neg, neg_eid, xq)
            Jn = transform.jacobian(mesh_neg, neg_eid, (float(xi_nq), float(eta_nq)))
            Jin = np.linalg.inv(Jn)
            phi_ux_n = np.asarray(me_neg.basis(u_fields[0], float(xi_nq), float(eta_nq)), dtype=float).ravel()
            phi_uy_n = np.asarray(me_neg.basis(u_fields[1], float(xi_nq), float(eta_nq)), dtype=float).ravel()
            phi_p_n = np.asarray(me_neg.basis(p_field, float(xi_nq), float(eta_nq)), dtype=float).ravel()
            g_ux_n = np.asarray(me_neg.grad_basis(u_fields[0], float(xi_nq), float(eta_nq)), dtype=float) @ Jin
            g_uy_n = np.asarray(me_neg.grad_basis(u_fields[1], float(xi_nq), float(eta_nq)), dtype=float) @ Jin

            Uvec_n[s, q, :, 0] = phi_ux_n
            Uvec_n[s, q, :, 1] = phi_uy_n

            dux_dx = g_ux_n[:, 0]
            dux_dy = g_ux_n[:, 1]
            duy_dx = g_uy_n[:, 0]
            duy_dy = g_uy_n[:, 1]
            tn_x = np.zeros(nln, dtype=np.float64)
            tn_y = np.zeros(nln, dtype=np.float64)
            tn_x += 2.0 * mu_neg * (dux_dx * nx + 0.5 * dux_dy * ny)
            tn_y += 2.0 * mu_neg * (0.5 * dux_dy * nx)
            tn_x += 2.0 * mu_neg * (0.5 * duy_dx * ny)
            tn_y += 2.0 * mu_neg * (0.5 * duy_dx * nx + duy_dy * ny)
            tn_x += -phi_p_n * nx
            tn_y += -phi_p_n * ny
            T_n[s, q, :, 0] = tn_x
            T_n[s, q, :, 1] = tn_y

    rows, cols, data = _stokes_nitsche_triplets(
        gd_pos, gd_neg, Uvec_p, T_p, Uvec_n, T_n, qw, stab, kappa_pos, kappa_neg
    )
    K = sp.coo_matrix((data, (rows, cols)), shape=(n_total, n_total))
    K.sum_duplicates()
    return K.tocsr()


def assemble_poisson_mortar_coupling(
    *,
    interface: NonMatchingInterface,
    dh_neg: DofHandler,
    dh_pos: DofHandler,
    field: str = "u",
    p_lambda: int | None = None,
    master: str = "auto",
    quad_order: int | None = None,
) -> MortarCoupling:
    _require_numba()

    if interface.n_segments() <= 0:
        n_lambda = 0
        return MortarCoupling(
            B_pos=sp.csr_matrix((n_lambda, int(dh_pos.total_dofs))),
            B_neg=sp.csr_matrix((n_lambda, int(dh_neg.total_dofs))),
            p_lambda=int(p_lambda or 0),
        )

    if p_lambda is None:
        p_lambda = 1
    p_lambda = int(p_lambda)
    if p_lambda != 1:
        raise NotImplementedError("Mortar coupling currently supports p_lambda=1 only.")

    master = str(master).strip().lower()
    if master not in {"auto", "pos", "neg"}:
        raise ValueError("master must be 'auto', 'pos', or 'neg'")

    if quad_order is None:
        p_pos = _field_order(dh_pos, field)
        p_neg = _field_order(dh_neg, field)
        quad_order = int(2 * max(p_pos, p_neg, p_lambda) + 2)

    qpts, qw = _interface_quadrature(interface, int(quad_order))

    n_pos = int(dh_pos.total_dofs)
    n_neg = int(dh_neg.total_dofs)
    n_seg = interface.n_segments()
    mesh_pos = interface.mesh_pos
    mesh_neg = interface.mesh_neg

    def _edge_nodes(mesh, edge_ids: np.ndarray) -> np.ndarray:
        nodes: set[int] = set()
        for gid in np.unique(np.asarray(edge_ids, dtype=int)):
            e = mesh.edge(int(gid))
            nodes.update(int(n) for n in e.nodes)
        node_ids = np.fromiter(nodes, dtype=int)
        d0 = np.asarray(interface.P1[0] - interface.P0[0], dtype=float)
        t = d0 / max(float(np.linalg.norm(d0)), 1.0e-16)
        s = mesh.nodes_x_y_pos[node_ids] @ t
        return node_ids[np.argsort(s)]

    neg_nodes = _edge_nodes(mesh_neg, interface.neg_edge_ids)
    pos_nodes = _edge_nodes(mesh_pos, interface.pos_edge_ids)
    if master == "auto":
        master = "neg" if neg_nodes.size <= pos_nodes.size else "pos"

    if master == "neg":
        mesh_master = mesh_neg
        master_edge_ids = interface.neg_edge_ids
        master_nodes = neg_nodes
    else:
        mesh_master = mesh_pos
        master_edge_ids = interface.pos_edge_ids
        master_nodes = pos_nodes

    node_to_lam = {int(n): i for i, n in enumerate(master_nodes.tolist())}
    n_lambda = int(master_nodes.size)

    me_pos = dh_pos.mixed_element
    me_neg = dh_neg.mixed_element

    # fixed local sizes
    gd_pos0 = np.asarray(dh_pos.get_elemental_dofs(int(interface.pos_elem_ids[0])), dtype=np.int64)
    gd_neg0 = np.asarray(dh_neg.get_elemental_dofs(int(interface.neg_elem_ids[0])), dtype=np.int64)
    nlp = int(gd_pos0.size)
    nln = int(gd_neg0.size)

    gd_pos = np.empty((n_seg, nlp), dtype=np.int64)
    gd_neg = np.empty((n_seg, nln), dtype=np.int64)
    lam_ids = np.empty((n_seg, 2), dtype=np.int64)

    phi_p = np.zeros((n_seg, qpts.shape[1], nlp), dtype=np.float64)
    phi_n = np.zeros((n_seg, qpts.shape[1], nln), dtype=np.float64)
    mu = np.zeros((n_seg, qpts.shape[1], 2), dtype=np.float64)

    for s in range(n_seg):
        pos_eid = int(interface.pos_elem_ids[s])
        neg_eid = int(interface.neg_elem_ids[s])
        gd_p = np.asarray(dh_pos.get_elemental_dofs(pos_eid), dtype=np.int64)
        gd_n = np.asarray(dh_neg.get_elemental_dofs(neg_eid), dtype=np.int64)
        if gd_p.size != nlp or gd_n.size != nln:
            raise ValueError("JIT backend currently requires constant local DOF counts per segment.")
        gd_pos[s, :] = gd_p
        gd_neg[s, :] = gd_n

        e_master = mesh_master.edge(int(master_edge_ids[s]))
        n0 = int(e_master.nodes[0])
        n1 = int(e_master.nodes[1])
        lam_ids[s, 0] = int(node_to_lam[n0])
        lam_ids[s, 1] = int(node_to_lam[n1])

        p0m = np.asarray(mesh_master.nodes_x_y_pos[n0], dtype=float)
        p1m = np.asarray(mesh_master.nodes_x_y_pos[n1], dtype=float)
        d = p1m - p0m
        L2 = float(np.dot(d, d)) or 1.0

        for q in range(qpts.shape[1]):
            xq = np.asarray(qpts[s, q], dtype=float)
            wq = float(qw[s, q])
            if wq == 0.0:
                continue
            r = float(np.dot(xq - p0m, d) / L2)
            r = min(1.0, max(0.0, r))
            xi_edge = 2.0 * r - 1.0
            mu[s, q, 0] = 0.5 * (1.0 - xi_edge)
            mu[s, q, 1] = 0.5 * (1.0 + xi_edge)

            xi_p, eta_p = transform.inverse_mapping(mesh_pos, pos_eid, xq)
            phi_p[s, q, :] = np.asarray(me_pos.basis(field, float(xi_p), float(eta_p)), dtype=float).ravel()

            xi_nq, eta_nq = transform.inverse_mapping(mesh_neg, neg_eid, xq)
            phi_n[s, q, :] = np.asarray(me_neg.basis(field, float(xi_nq), float(eta_nq)), dtype=float).ravel()

    rows_p, cols_p, data_p, rows_n, cols_n, data_n = _mortar_triplets(lam_ids, gd_pos, gd_neg, mu, phi_p, phi_n, qw)

    B_pos = sp.coo_matrix((data_p, (rows_p, cols_p)), shape=(n_lambda, n_pos))
    B_pos.sum_duplicates()
    B_neg = sp.coo_matrix((data_n, (rows_n, cols_n)), shape=(n_lambda, n_neg))
    B_neg.sum_duplicates()
    return MortarCoupling(B_pos=B_pos.tocsr(), B_neg=B_neg.tocsr(), p_lambda=p_lambda)


if _HAVE_NUMBA:  # pragma: no cover

    @nb.njit(cache=True)
    def _poisson_nitsche_triplets(
        gd_pos,
        gd_neg,
        phi_p,
        gn_p,
        phi_n,
        gn_n,
        qw,
        stab,
        kappa_pos,
        kappa_neg,
        k_pos,
        k_neg,
    ):
        n_seg = gd_pos.shape[0]
        nlp = gd_pos.shape[1]
        nln = gd_neg.shape[1]
        ntrip_per_seg = nlp * nlp + nlp * nln + nln * nlp + nln * nln
        rows = np.empty(n_seg * ntrip_per_seg, dtype=np.int64)
        cols = np.empty(n_seg * ntrip_per_seg, dtype=np.int64)
        data = np.empty(n_seg * ntrip_per_seg, dtype=np.float64)

        out = 0
        n_q = qw.shape[1]
        for s in range(n_seg):
            A_pp = np.zeros((nlp, nlp), dtype=np.float64)
            A_pn = np.zeros((nlp, nln), dtype=np.float64)
            A_np = np.zeros((nln, nlp), dtype=np.float64)
            A_nn = np.zeros((nln, nln), dtype=np.float64)
            st = stab[s]
            for q in range(n_q):
                wq = qw[s, q]
                if wq == 0.0:
                    continue
                for i in range(nlp):
                    phi_pi = phi_p[s, q, i]
                    gn_pi = gn_p[s, q, i]
                    for j in range(nlp):
                        A_pp[i, j] += wq * (
                            kappa_pos * k_pos * (phi_pi * gn_p[s, q, j] + gn_pi * phi_p[s, q, j])
                            + st * phi_pi * phi_p[s, q, j]
                        )
                    for j in range(nln):
                        A_pn[i, j] += wq * (
                            kappa_neg * k_neg * phi_pi * gn_n[s, q, j]
                            - kappa_pos * k_pos * gn_pi * phi_n[s, q, j]
                            - st * phi_pi * phi_n[s, q, j]
                        )
                for i in range(nln):
                    phi_ni = phi_n[s, q, i]
                    gn_ni = gn_n[s, q, i]
                    for j in range(nlp):
                        A_np[i, j] += wq * (
                            -kappa_pos * k_pos * phi_ni * gn_p[s, q, j]
                            + kappa_neg * k_neg * gn_ni * phi_p[s, q, j]
                            - st * phi_ni * phi_p[s, q, j]
                        )
                    for j in range(nln):
                        A_nn[i, j] += wq * (
                            -kappa_neg * k_neg * (phi_ni * gn_n[s, q, j] + gn_ni * phi_n[s, q, j])
                            + st * phi_ni * phi_n[s, q, j]
                        )

            for i in range(nlp):
                gi = gd_pos[s, i]
                for j in range(nlp):
                    rows[out] = gi
                    cols[out] = gd_pos[s, j]
                    data[out] = A_pp[i, j]
                    out += 1
                for j in range(nln):
                    rows[out] = gi
                    cols[out] = gd_neg[s, j]
                    data[out] = A_pn[i, j]
                    out += 1
            for i in range(nln):
                gi = gd_neg[s, i]
                for j in range(nlp):
                    rows[out] = gi
                    cols[out] = gd_pos[s, j]
                    data[out] = A_np[i, j]
                    out += 1
                for j in range(nln):
                    rows[out] = gi
                    cols[out] = gd_neg[s, j]
                    data[out] = A_nn[i, j]
                    out += 1

        return rows, cols, data

    @nb.njit(cache=True)
    def _stokes_nitsche_triplets(gd_pos, gd_neg, Uvec_p, T_p, Uvec_n, T_n, qw, stab, kappa_pos, kappa_neg):
        n_seg = gd_pos.shape[0]
        nlp = gd_pos.shape[1]
        nln = gd_neg.shape[1]
        ntrip_per_seg = nlp * nlp + nlp * nln + nln * nlp + nln * nln
        rows = np.empty(n_seg * ntrip_per_seg, dtype=np.int64)
        cols = np.empty(n_seg * ntrip_per_seg, dtype=np.int64)
        data = np.empty(n_seg * ntrip_per_seg, dtype=np.float64)

        out = 0
        n_q = qw.shape[1]
        for s in range(n_seg):
            A_pp = np.zeros((nlp, nlp), dtype=np.float64)
            A_pn = np.zeros((nlp, nln), dtype=np.float64)
            A_np = np.zeros((nln, nlp), dtype=np.float64)
            A_nn = np.zeros((nln, nln), dtype=np.float64)
            st = stab[s]
            for q in range(n_q):
                wq = qw[s, q]
                if wq == 0.0:
                    continue

                # Term 1: {t(u)} · (v_neg - v_pos)
                for i in range(nlp):
                    uxi = Uvec_p[s, q, i, 0]
                    uyi = Uvec_p[s, q, i, 1]
                    for j in range(nlp):
                        txj = T_p[s, q, j, 0]
                        tyj = T_p[s, q, j, 1]
                        A_pp[i, j] += wq * (-kappa_pos) * (uxi * txj + uyi * tyj)
                    for j in range(nln):
                        txj = T_n[s, q, j, 0]
                        tyj = T_n[s, q, j, 1]
                        A_pn[i, j] += wq * (-kappa_neg) * (uxi * txj + uyi * tyj)
                for i in range(nln):
                    uxi = Uvec_n[s, q, i, 0]
                    uyi = Uvec_n[s, q, i, 1]
                    for j in range(nlp):
                        txj = T_p[s, q, j, 0]
                        tyj = T_p[s, q, j, 1]
                        A_np[i, j] += wq * (kappa_pos) * (uxi * txj + uyi * tyj)
                    for j in range(nln):
                        txj = T_n[s, q, j, 0]
                        tyj = T_n[s, q, j, 1]
                        A_nn[i, j] += wq * (kappa_neg) * (uxi * txj + uyi * tyj)

                # Term 2: {t(v)} · (u_neg - u_pos)
                for i in range(nlp):
                    txi = T_p[s, q, i, 0]
                    tyi = T_p[s, q, i, 1]
                    for j in range(nlp):
                        uxj = Uvec_p[s, q, j, 0]
                        uyj = Uvec_p[s, q, j, 1]
                        A_pp[i, j] += wq * (-kappa_pos) * (txi * uxj + tyi * uyj)
                    for j in range(nln):
                        uxj = Uvec_n[s, q, j, 0]
                        uyj = Uvec_n[s, q, j, 1]
                        A_pn[i, j] += wq * (kappa_pos) * (txi * uxj + tyi * uyj)
                for i in range(nln):
                    txi = T_n[s, q, i, 0]
                    tyi = T_n[s, q, i, 1]
                    for j in range(nlp):
                        uxj = Uvec_p[s, q, j, 0]
                        uyj = Uvec_p[s, q, j, 1]
                        A_np[i, j] += wq * (-kappa_neg) * (txi * uxj + tyi * uyj)
                    for j in range(nln):
                        uxj = Uvec_n[s, q, j, 0]
                        uyj = Uvec_n[s, q, j, 1]
                        A_nn[i, j] += wq * (kappa_neg) * (txi * uxj + tyi * uyj)

                # Penalty: stab (u_neg-u_pos)·(v_neg-v_pos)
                for i in range(nlp):
                    uxi = Uvec_p[s, q, i, 0]
                    uyi = Uvec_p[s, q, i, 1]
                    for j in range(nlp):
                        uxj = Uvec_p[s, q, j, 0]
                        uyj = Uvec_p[s, q, j, 1]
                        A_pp[i, j] += wq * st * (uxi * uxj + uyi * uyj)
                    for j in range(nln):
                        uxj = Uvec_n[s, q, j, 0]
                        uyj = Uvec_n[s, q, j, 1]
                        A_pn[i, j] += wq * (-st) * (uxi * uxj + uyi * uyj)
                for i in range(nln):
                    uxi = Uvec_n[s, q, i, 0]
                    uyi = Uvec_n[s, q, i, 1]
                    for j in range(nlp):
                        uxj = Uvec_p[s, q, j, 0]
                        uyj = Uvec_p[s, q, j, 1]
                        A_np[i, j] += wq * (-st) * (uxi * uxj + uyi * uyj)
                    for j in range(nln):
                        uxj = Uvec_n[s, q, j, 0]
                        uyj = Uvec_n[s, q, j, 1]
                        A_nn[i, j] += wq * st * (uxi * uxj + uyi * uyj)

            for i in range(nlp):
                gi = gd_pos[s, i]
                for j in range(nlp):
                    rows[out] = gi
                    cols[out] = gd_pos[s, j]
                    data[out] = A_pp[i, j]
                    out += 1
                for j in range(nln):
                    rows[out] = gi
                    cols[out] = gd_neg[s, j]
                    data[out] = A_pn[i, j]
                    out += 1
            for i in range(nln):
                gi = gd_neg[s, i]
                for j in range(nlp):
                    rows[out] = gi
                    cols[out] = gd_pos[s, j]
                    data[out] = A_np[i, j]
                    out += 1
                for j in range(nln):
                    rows[out] = gi
                    cols[out] = gd_neg[s, j]
                    data[out] = A_nn[i, j]
                    out += 1

        return rows, cols, data

    @nb.njit(cache=True)
    def _mortar_triplets(lam_ids, gd_pos, gd_neg, mu, phi_p, phi_n, qw):
        n_seg = gd_pos.shape[0]
        nlp = gd_pos.shape[1]
        nln = gd_neg.shape[1]
        n_q = qw.shape[1]

        ntrip_p = n_seg * (2 * nlp)
        ntrip_n = n_seg * (2 * nln)
        rows_p = np.empty(ntrip_p, dtype=np.int64)
        cols_p = np.empty(ntrip_p, dtype=np.int64)
        data_p = np.empty(ntrip_p, dtype=np.float64)
        rows_n = np.empty(ntrip_n, dtype=np.int64)
        cols_n = np.empty(ntrip_n, dtype=np.int64)
        data_n = np.empty(ntrip_n, dtype=np.float64)

        out_p = 0
        out_n = 0
        for s in range(n_seg):
            Bp = np.zeros((2, nlp), dtype=np.float64)
            Bn = np.zeros((2, nln), dtype=np.float64)
            for q in range(n_q):
                wq = qw[s, q]
                if wq == 0.0:
                    continue
                mu0 = mu[s, q, 0]
                mu1 = mu[s, q, 1]
                for j in range(nlp):
                    val = phi_p[s, q, j]
                    Bp[0, j] += wq * mu0 * val
                    Bp[1, j] += wq * mu1 * val
                for j in range(nln):
                    val = phi_n[s, q, j]
                    Bn[0, j] += wq * mu0 * val
                    Bn[1, j] += wq * mu1 * val

            lam0 = lam_ids[s, 0]
            lam1 = lam_ids[s, 1]
            for j in range(nlp):
                rows_p[out_p] = lam0
                cols_p[out_p] = gd_pos[s, j]
                data_p[out_p] = Bp[0, j]
                out_p += 1
            for j in range(nlp):
                rows_p[out_p] = lam1
                cols_p[out_p] = gd_pos[s, j]
                data_p[out_p] = Bp[1, j]
                out_p += 1

            for j in range(nln):
                rows_n[out_n] = lam0
                cols_n[out_n] = gd_neg[s, j]
                data_n[out_n] = Bn[0, j]
                out_n += 1
            for j in range(nln):
                rows_n[out_n] = lam1
                cols_n[out_n] = gd_neg[s, j]
                data_n[out_n] = Bn[1, j]
                out_n += 1

        return rows_p, cols_p, data_p, rows_n, cols_n, data_n

