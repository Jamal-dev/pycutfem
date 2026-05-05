from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem import transform
from pycutfem.integration.quadrature import gauss_legendre

from .interface import NonMatchingInterface
from .nitsche import _field_order


@lru_cache(maxsize=None)
def _lagrange_nodes_1d(p: int) -> np.ndarray:
    p = int(p)
    if p < 0:
        raise ValueError("p must be >= 0")
    if p == 0:
        return np.asarray([0.0], dtype=float)
    return np.linspace(-1.0, 1.0, p + 1, dtype=float)


@lru_cache(maxsize=None)
def _barycentric_weights_1d(p: int) -> np.ndarray:
    nodes = _lagrange_nodes_1d(int(p))
    n = nodes.size
    w = np.ones(n, dtype=float)
    for i in range(n):
        prod = 1.0
        xi = float(nodes[i])
        for j in range(n):
            if j == i:
                continue
            prod *= (xi - float(nodes[j]))
        w[i] = 1.0 / prod if prod != 0.0 else 0.0
    return w


def lagrange_basis_1d(p: int, x: np.ndarray, *, tol: float = 1.0e-14) -> np.ndarray:
    """Lagrange basis values at points `x` on [-1,1] (barycentric form)."""
    p = int(p)
    x = np.asarray(x, dtype=float).ravel()
    nodes = _lagrange_nodes_1d(p)
    w = _barycentric_weights_1d(p)
    n = nodes.size
    out = np.empty((x.size, n), dtype=float)
    for k, xx in enumerate(x):
        diff = xx - nodes
        hit = np.nonzero(np.abs(diff) <= tol)[0]
        if hit.size:
            out[k, :] = 0.0
            out[k, int(hit[0])] = 1.0
            continue
        tmp = w / diff
        denom = float(np.sum(tmp))
        out[k, :] = tmp / denom
    return out


@dataclass(frozen=True, slots=True)
class MortarCoupling:
    """Mortar coupling matrices for P1 multipliers (node-based)."""

    B_pos: sp.csr_matrix  # shape (n_lambda, n_pos)
    B_neg: sp.csr_matrix  # shape (n_lambda, n_neg)
    p_lambda: int

    @property
    def n_lambda(self) -> int:
        return int(self.B_pos.shape[0])


def assemble_poisson_mortar_coupling(
    *,
    interface: NonMatchingInterface,
    dh_neg: DofHandler,
    dh_pos: DofHandler,
    field: str = "u",
    p_lambda: int | None = None,
    master: str = "auto",
    quad_order: int | None = None,
    backend: str = "python",
) -> MortarCoupling:
    """Assemble mortar coupling operators for Poisson using a master-side P1 space.

    Notes
    -----
    This implements the classical Lagrange-multiplier mortar coupling with a
    *node-based* P1 multiplier space defined on the chosen master interface
    discretization (coarser side by default). This avoids the rank-deficiency
    issues that arise when taking the multiplier space on the full common
    refinement.
    """

    backend = str(backend).strip().lower()
    if backend != "python":
        if backend == "jit":
            from .jit_backend import assemble_poisson_mortar_coupling as _jit

            return _jit(
                interface=interface,
                dh_neg=dh_neg,
                dh_pos=dh_pos,
                field=field,
                p_lambda=p_lambda,
                master=master,
                quad_order=quad_order,
            )
        if backend in {"cpp", "c++"}:
            from .cpp_backend import assemble_poisson_mortar_coupling as _cpp

            return _cpp(
                interface=interface,
                dh_neg=dh_neg,
                dh_pos=dh_pos,
                field=field,
                p_lambda=p_lambda,
                master=master,
                quad_order=quad_order,
            )
        raise ValueError("backend must be 'python', 'jit', or 'cpp'.")

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

    n_pos = int(dh_pos.total_dofs)
    n_neg = int(dh_neg.total_dofs)
    n_seg = interface.n_segments()
    n_loc_lam = 2  # P1 only
    mesh_pos = interface.mesh_pos
    mesh_neg = interface.mesh_neg

    def _edge_nodes(mesh, edge_ids: np.ndarray) -> np.ndarray:
        nodes: set[int] = set()
        for gid in np.unique(np.asarray(edge_ids, dtype=int)):
            e = mesh.edge(int(gid))
            nodes.update(int(n) for n in e.nodes)
        node_ids = np.fromiter(nodes, dtype=int)
        # stable ordering along interface tangent
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

    rows_pos: list[int] = []
    cols_pos: list[int] = []
    data_pos: list[float] = []

    rows_neg: list[int] = []
    cols_neg: list[int] = []
    data_neg: list[float] = []

    me_pos = dh_pos.mixed_element
    me_neg = dh_neg.mixed_element

    for s in range(n_seg):
        pos_eid = int(interface.pos_elem_ids[s])
        neg_eid = int(interface.neg_elem_ids[s])

        gd_pos = np.asarray(dh_pos.get_elemental_dofs(pos_eid), dtype=int)
        gd_neg = np.asarray(dh_neg.get_elemental_dofs(neg_eid), dtype=int)
        nlp = int(gd_pos.size)
        nln = int(gd_neg.size)

        Bp = np.zeros((n_loc_lam, nlp), dtype=float)
        Bn = np.zeros((n_loc_lam, nln), dtype=float)

        e_master = mesh_master.edge(int(master_edge_ids[s]))
        p0m = np.asarray(mesh_master.nodes_x_y_pos[int(e_master.nodes[0])], dtype=float)
        p1m = np.asarray(mesh_master.nodes_x_y_pos[int(e_master.nodes[1])], dtype=float)
        d = p1m - p0m
        L2 = float(np.dot(d, d)) or 1.0

        for q in range(xi.size):
            xq = np.asarray(qpts[s, q], dtype=float)
            wq = float(qw[s, q])
            if wq == 0.0:
                continue
            r = float(np.dot(xq - p0m, d) / L2)
            r = min(1.0, max(0.0, r))
            xi_edge = 2.0 * r - 1.0
            mu = np.asarray([0.5 * (1.0 - xi_edge), 0.5 * (1.0 + xi_edge)], dtype=float)

            xi_p, eta_p = transform.inverse_mapping(mesh_pos, pos_eid, xq)
            phi_p = np.asarray(me_pos.basis(field, float(xi_p), float(eta_p)), dtype=float).ravel()

            xi_n, eta_n = transform.inverse_mapping(mesh_neg, neg_eid, xq)
            phi_n = np.asarray(me_neg.basis(field, float(xi_n), float(eta_n)), dtype=float).ravel()

            Bp += wq * np.outer(mu, phi_p)
            Bn += wq * np.outer(mu, phi_n)

        lam_ids = np.asarray(
            [node_to_lam[int(e_master.nodes[0])], node_to_lam[int(e_master.nodes[1])]],
            dtype=int,
        )

        rr, cc = np.meshgrid(lam_ids, gd_pos, indexing="ij")
        rows_pos.extend(rr.ravel().tolist())
        cols_pos.extend(cc.ravel().tolist())
        data_pos.extend(Bp.ravel().tolist())

        rr, cc = np.meshgrid(lam_ids, gd_neg, indexing="ij")
        rows_neg.extend(rr.ravel().tolist())
        cols_neg.extend(cc.ravel().tolist())
        data_neg.extend(Bn.ravel().tolist())

    B_pos = sp.coo_matrix(
        (np.asarray(data_pos, float), (np.asarray(rows_pos, int), np.asarray(cols_pos, int))),
        shape=(n_lambda, n_pos),
    )
    B_pos.sum_duplicates()
    B_neg = sp.coo_matrix(
        (np.asarray(data_neg, float), (np.asarray(rows_neg, int), np.asarray(cols_neg, int))),
        shape=(n_lambda, n_neg),
    )
    B_neg.sum_duplicates()
    return MortarCoupling(B_pos=B_pos.tocsr(), B_neg=B_neg.tocsr(), p_lambda=p_lambda)


def assemble_mortar_saddle_matrix(
    *,
    K_pos: sp.spmatrix,
    K_neg: sp.spmatrix,
    coupling: MortarCoupling,
) -> sp.csr_matrix:
    """Build the 3x3 saddle-point matrix for the mortar system."""
    K_pos = K_pos.tocsr()
    K_neg = K_neg.tocsr()
    n_pos = int(K_pos.shape[0])
    n_neg = int(K_neg.shape[0])
    n_lam = int(coupling.n_lambda)

    Zpp = sp.csr_matrix((n_pos, n_neg))
    Zpl = sp.csr_matrix((n_pos, n_lam))
    Znp = sp.csr_matrix((n_neg, n_pos))
    Znl = sp.csr_matrix((n_neg, n_lam))
    Zlp = sp.csr_matrix((n_lam, n_pos))
    Zln = sp.csr_matrix((n_lam, n_neg))
    Zll = sp.csr_matrix((n_lam, n_lam))

    Bp = coupling.B_pos
    Bn = coupling.B_neg

    # [ Kp  0   Bp^T ]
    # [ 0   Kn -Bn^T ]
    # [ Bp -Bn   0  ]
    top = sp.hstack([K_pos, Zpp, Bp.T], format="csr")
    mid = sp.hstack([Znp, K_neg, -Bn.T], format="csr")
    bot = sp.hstack([Bp, -Bn, Zll], format="csr")
    return sp.vstack([top, mid, bot], format="csr")
