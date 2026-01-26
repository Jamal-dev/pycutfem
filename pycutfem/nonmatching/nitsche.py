from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem import transform
from pycutfem.integration.quadrature import gauss_legendre

from .interface import NonMatchingInterface


def _field_order(dh: DofHandler, field: str) -> int:
    me = getattr(dh, "mixed_element", None)
    if me is None:
        raise ValueError("Non-matching coupling requires a MixedElement-backed DofHandler.")
    orders = getattr(me, "get_field_orders", None)
    if callable(orders):
        return int(orders().get(field, 1))
    return int(getattr(me, "_field_orders", {}).get(field, 1))


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
    backend: str = "python",
) -> sp.csr_matrix:
    """Assemble the Nitsche interface coupling matrix for Poisson (scalar).

    The assembled matrix is for the global ordering::

        U = [U_pos, U_neg]

    i.e. positive-side DOFs first, then negative-side DOFs.

    Notes
    -----
    This follows the project's CutFEM convention (see `examples/xfem_interface_poisson.py`):
    - `interface.n` is oriented neg->pos.
    - Jump convention:
        [u] = u_neg - u_pos,   [v] = v_neg - v_pos.
    - Physical normal flux uses the *same* normal direction on both sides:
        q = -k ∇u · n.

    The implemented interface bilinear form is:

        ⟨ {q(u)}, [v] ⟩ + ⟨ {q(v)}, [u] ⟩ + ⟨ stab [u], [v] ⟩.
    """

    backend = str(backend).strip().lower()
    if backend != "python":
        if backend == "jit":
            from .jit_backend import assemble_poisson_nitsche_interface_matrix as _jit

            return _jit(
                interface=interface,
                dh_neg=dh_neg,
                dh_pos=dh_pos,
                field=field,
                k_neg=k_neg,
                k_pos=k_pos,
                gamma=gamma,
                quad_order=quad_order,
            )
        if backend in {"cpp", "c++"}:
            from .cpp_backend import assemble_poisson_nitsche_interface_matrix as _cpp

            return _cpp(
                interface=interface,
                dh_neg=dh_neg,
                dh_pos=dh_pos,
                field=field,
                k_neg=k_neg,
                k_pos=k_pos,
                gamma=gamma,
                quad_order=quad_order,
            )
        raise ValueError("backend must be 'python', 'jit', or 'cpp'.")

    if interface.n_segments() <= 0:
        n_total = int(dh_pos.total_dofs) + int(dh_neg.total_dofs)
        return sp.csr_matrix((n_total, n_total), dtype=float)

    # Quadrature order ---------------------------------------------------------
    if quad_order is None:
        p_pos = _field_order(dh_pos, field)
        p_neg = _field_order(dh_neg, field)
        quad_order = int(2 * max(p_pos, p_neg) + 2)

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

    # Constants ----------------------------------------------------------------
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

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    me_pos = dh_pos.mixed_element
    me_neg = dh_neg.mixed_element
    mesh_pos = interface.mesh_pos
    mesh_neg = interface.mesh_neg

    for s in range(interface.n_segments()):
        pos_eid = int(interface.pos_elem_ids[s])
        neg_eid = int(interface.neg_elem_ids[s])
        n_vec = np.asarray(interface.n[s], dtype=float)
        h = float(min(interface.h_pos[s], interface.h_neg[s]) or 0.0)
        if h <= 0.0:
            h = 1.0
        stab = float(gamma) * (k_pos + k_neg) / h

        gd_pos = np.asarray(dh_pos.get_elemental_dofs(pos_eid), dtype=int)
        gd_neg = np.asarray(dh_neg.get_elemental_dofs(neg_eid), dtype=int) + n_pos
        nlp = int(gd_pos.size)
        nln = int(gd_neg.size)

        A_pp = np.zeros((nlp, nlp), dtype=float)
        A_pn = np.zeros((nlp, nln), dtype=float)
        A_np = np.zeros((nln, nlp), dtype=float)
        A_nn = np.zeros((nln, nln), dtype=float)

        for q in range(qpts.shape[1]):
            xq = np.asarray(qpts[s, q], dtype=float)
            wq = float(qw[s, q])
            if wq == 0.0:
                continue

            xi_p, eta_p = transform.inverse_mapping(mesh_pos, pos_eid, xq)
            Jp = transform.jacobian(mesh_pos, pos_eid, (float(xi_p), float(eta_p)))
            Jip = np.linalg.inv(Jp)
            phi_p = np.asarray(me_pos.basis(field, float(xi_p), float(eta_p)), dtype=float).ravel()
            grad_p = np.asarray(me_pos.grad_basis(field, float(xi_p), float(eta_p)), dtype=float) @ Jip
            gn_p = grad_p @ n_vec

            xi_n, eta_n = transform.inverse_mapping(mesh_neg, neg_eid, xq)
            Jn = transform.jacobian(mesh_neg, neg_eid, (float(xi_n), float(eta_n)))
            Jin = np.linalg.inv(Jn)
            phi_n = np.asarray(me_neg.basis(field, float(xi_n), float(eta_n)), dtype=float).ravel()
            grad_n = np.asarray(me_neg.grad_basis(field, float(xi_n), float(eta_n)), dtype=float) @ Jin
            gn_n = grad_n @ n_vec

            A_pp += wq * (
                kappa_pos * k_pos * (np.outer(phi_p, gn_p) + np.outer(gn_p, phi_p))
                + stab * np.outer(phi_p, phi_p)
            )
            A_pn += wq * (
                kappa_neg * k_neg * np.outer(phi_p, gn_n)
                - kappa_pos * k_pos * np.outer(gn_p, phi_n)
                - stab * np.outer(phi_p, phi_n)
            )
            A_np += wq * (
                -kappa_pos * k_pos * np.outer(phi_n, gn_p)
                + kappa_neg * k_neg * np.outer(gn_n, phi_p)
                - stab * np.outer(phi_n, phi_p)
            )
            A_nn += wq * (
                -kappa_neg * k_neg * (np.outer(phi_n, gn_n) + np.outer(gn_n, phi_n))
                + stab * np.outer(phi_n, phi_n)
            )

        # Scatter to global COO triplets --------------------------------------
        rr, cc = np.meshgrid(gd_pos, gd_pos, indexing="ij")
        rows.extend(rr.ravel().tolist())
        cols.extend(cc.ravel().tolist())
        data.extend(A_pp.ravel().tolist())

        rr, cc = np.meshgrid(gd_pos, gd_neg, indexing="ij")
        rows.extend(rr.ravel().tolist())
        cols.extend(cc.ravel().tolist())
        data.extend(A_pn.ravel().tolist())

        rr, cc = np.meshgrid(gd_neg, gd_pos, indexing="ij")
        rows.extend(rr.ravel().tolist())
        cols.extend(cc.ravel().tolist())
        data.extend(A_np.ravel().tolist())

        rr, cc = np.meshgrid(gd_neg, gd_neg, indexing="ij")
        rows.extend(rr.ravel().tolist())
        cols.extend(cc.ravel().tolist())
        data.extend(A_nn.ravel().tolist())

    K = sp.coo_matrix((np.asarray(data, float), (np.asarray(rows, int), np.asarray(cols, int))), shape=(n_total, n_total))
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
    backend: str = "python",
) -> sp.csr_matrix:
    """Assemble a symmetric Nitsche coupling for Stokes–Stokes across a non-matching interface.

    Unknown ordering is `[U_pos, U_neg]`, where each block contains *all* fields
    of the corresponding DofHandler (typically Taylor–Hood: ux, uy, p).

    Form (neg/pos convention, mirroring the scalar interface routine)
    ---------------------------------------------------------------
    Let `n` be oriented neg->pos (from `interface.n`), and define the stress
    tensor σ(u,p) = 2μ ε(u) - p I with ε(u) = (∇u + ∇u^T)/2. We enforce:
      - velocity continuity: u_neg = u_pos on Γ,
      - traction continuity: σ_neg(u,p) n = σ_pos(u,p) n on Γ,
    using the symmetric interface bilinear form:

        ⟨ {t(u,p)}, [v] ⟩ + ⟨ {t(v,q)}, [u] ⟩ + ⟨ stab [u],[v] ⟩,

    where traction t(u,p) = σ(u,p) n, jump [u] = u_neg - u_pos, and {·} is a
    viscosity-weighted average.
    """

    backend = str(backend).strip().lower()
    if backend != "python":
        if backend == "jit":
            from .jit_backend import assemble_stokes_nitsche_interface_matrix as _jit

            return _jit(
                interface=interface,
                dh_neg=dh_neg,
                dh_pos=dh_pos,
                mu_neg=mu_neg,
                mu_pos=mu_pos,
                gamma=gamma,
                quad_order=quad_order,
                u_fields=u_fields,
                p_field=p_field,
            )
        if backend in {"cpp", "c++"}:
            from .cpp_backend import assemble_stokes_nitsche_interface_matrix as _cpp

            return _cpp(
                interface=interface,
                dh_neg=dh_neg,
                dh_pos=dh_pos,
                mu_neg=mu_neg,
                mu_pos=mu_pos,
                gamma=gamma,
                quad_order=quad_order,
                u_fields=u_fields,
                p_field=p_field,
            )
        raise ValueError("backend must be 'python', 'jit', or 'cpp'.")

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
    n_total = n_pos + n_neg

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    me_pos = dh_pos.mixed_element
    me_neg = dh_neg.mixed_element
    mesh_pos = interface.mesh_pos
    mesh_neg = interface.mesh_neg

    for s in range(interface.n_segments()):
        pos_eid = int(interface.pos_elem_ids[s])
        neg_eid = int(interface.neg_elem_ids[s])
        n_vec = np.asarray(interface.n[s], dtype=float)
        nx, ny = float(n_vec[0]), float(n_vec[1])
        h = float(min(interface.h_pos[s], interface.h_neg[s]) or 0.0)
        if h <= 0.0:
            h = 1.0
        stab = float(gamma) * (mu_pos + mu_neg) / h

        gd_pos = np.asarray(dh_pos.get_elemental_dofs(pos_eid), dtype=int)
        gd_neg = np.asarray(dh_neg.get_elemental_dofs(neg_eid), dtype=int) + n_pos
        nlp = int(gd_pos.size)
        nln = int(gd_neg.size)

        A_pp = np.zeros((nlp, nlp), dtype=float)
        A_pn = np.zeros((nlp, nln), dtype=float)
        A_np = np.zeros((nln, nlp), dtype=float)
        A_nn = np.zeros((nln, nln), dtype=float)

        for q in range(qpts.shape[1]):
            xq = np.asarray(qpts[s, q], dtype=float)
            wq = float(qw[s, q])
            if wq == 0.0:
                continue

            # --- POS side (u,p) basis/grad in physical coords -----------------
            xi_p, eta_p = transform.inverse_mapping(mesh_pos, pos_eid, xq)
            Jp = transform.jacobian(mesh_pos, pos_eid, (float(xi_p), float(eta_p)))
            Jip = np.linalg.inv(Jp)

            phi_ux_p = np.asarray(me_pos.basis(u_fields[0], float(xi_p), float(eta_p)), dtype=float).ravel()
            phi_uy_p = np.asarray(me_pos.basis(u_fields[1], float(xi_p), float(eta_p)), dtype=float).ravel()
            phi_p_p = np.asarray(me_pos.basis(p_field, float(xi_p), float(eta_p)), dtype=float).ravel()

            g_ux_p = np.asarray(me_pos.grad_basis(u_fields[0], float(xi_p), float(eta_p)), dtype=float) @ Jip
            g_uy_p = np.asarray(me_pos.grad_basis(u_fields[1], float(xi_p), float(eta_p)), dtype=float) @ Jip

            Uvec_p = np.stack([phi_ux_p, phi_uy_p], axis=1)  # (nlp,2)

            # Traction basis for POS dofs: T_p[dof] is a 2-vector
            T_p = np.zeros((nlp, 2), dtype=float)
            dux_dx = g_ux_p[:, 0]
            dux_dy = g_ux_p[:, 1]
            duy_dx = g_uy_p[:, 0]
            duy_dy = g_uy_p[:, 1]
            # u = (phi,0)
            T_p[:, 0] += 2.0 * mu_pos * (dux_dx * nx + 0.5 * dux_dy * ny)
            T_p[:, 1] += 2.0 * mu_pos * (0.5 * dux_dy * nx)
            # u = (0,phi)
            T_p[:, 0] += 2.0 * mu_pos * (0.5 * duy_dx * ny)
            T_p[:, 1] += 2.0 * mu_pos * (0.5 * duy_dx * nx + duy_dy * ny)
            # pressure: -p n
            T_p[:, 0] += -phi_p_p * nx
            T_p[:, 1] += -phi_p_p * ny

            # --- NEG side -----------------------------------------------------
            xi_n, eta_n = transform.inverse_mapping(mesh_neg, neg_eid, xq)
            Jn = transform.jacobian(mesh_neg, neg_eid, (float(xi_n), float(eta_n)))
            Jin = np.linalg.inv(Jn)

            phi_ux_n = np.asarray(me_neg.basis(u_fields[0], float(xi_n), float(eta_n)), dtype=float).ravel()
            phi_uy_n = np.asarray(me_neg.basis(u_fields[1], float(xi_n), float(eta_n)), dtype=float).ravel()
            phi_p_n = np.asarray(me_neg.basis(p_field, float(xi_n), float(eta_n)), dtype=float).ravel()

            g_ux_n = np.asarray(me_neg.grad_basis(u_fields[0], float(xi_n), float(eta_n)), dtype=float) @ Jin
            g_uy_n = np.asarray(me_neg.grad_basis(u_fields[1], float(xi_n), float(eta_n)), dtype=float) @ Jin

            Uvec_n = np.stack([phi_ux_n, phi_uy_n], axis=1)  # (nln,2)

            T_n = np.zeros((nln, 2), dtype=float)
            dux_dx = g_ux_n[:, 0]
            dux_dy = g_ux_n[:, 1]
            duy_dx = g_uy_n[:, 0]
            duy_dy = g_uy_n[:, 1]
            T_n[:, 0] += 2.0 * mu_neg * (dux_dx * nx + 0.5 * dux_dy * ny)
            T_n[:, 1] += 2.0 * mu_neg * (0.5 * dux_dy * nx)
            T_n[:, 0] += 2.0 * mu_neg * (0.5 * duy_dx * ny)
            T_n[:, 1] += 2.0 * mu_neg * (0.5 * duy_dx * nx + duy_dy * ny)
            T_n[:, 0] += -phi_p_n * nx
            T_n[:, 1] += -phi_p_n * ny

            # Term 1: {t(u,p)} · (v_neg - v_pos)
            A_pp += wq * (-kappa_pos) * (Uvec_p @ T_p.T)
            A_pn += wq * (-kappa_neg) * (Uvec_p @ T_n.T)
            A_np += wq * (kappa_pos) * (Uvec_n @ T_p.T)
            A_nn += wq * (kappa_neg) * (Uvec_n @ T_n.T)

            # Term 2: {t(v,q)} · (u_neg - u_pos)
            A_pp += wq * (-kappa_pos) * (T_p @ Uvec_p.T)
            A_pn += wq * (kappa_pos) * (T_p @ Uvec_n.T)
            A_np += wq * (-kappa_neg) * (T_n @ Uvec_p.T)
            A_nn += wq * (kappa_neg) * (T_n @ Uvec_n.T)

            # Penalty: stab (u_neg-u_pos)·(v_neg-v_pos)
            A_pp += wq * stab * (Uvec_p @ Uvec_p.T)
            A_pn += wq * (-stab) * (Uvec_p @ Uvec_n.T)
            A_np += wq * (-stab) * (Uvec_n @ Uvec_p.T)
            A_nn += wq * stab * (Uvec_n @ Uvec_n.T)

        # Scatter to global COO triplets
        rr, cc = np.meshgrid(gd_pos, gd_pos, indexing="ij")
        rows.extend(rr.ravel().tolist())
        cols.extend(cc.ravel().tolist())
        data.extend(A_pp.ravel().tolist())

        rr, cc = np.meshgrid(gd_pos, gd_neg, indexing="ij")
        rows.extend(rr.ravel().tolist())
        cols.extend(cc.ravel().tolist())
        data.extend(A_pn.ravel().tolist())

        rr, cc = np.meshgrid(gd_neg, gd_pos, indexing="ij")
        rows.extend(rr.ravel().tolist())
        cols.extend(cc.ravel().tolist())
        data.extend(A_np.ravel().tolist())

        rr, cc = np.meshgrid(gd_neg, gd_neg, indexing="ij")
        rows.extend(rr.ravel().tolist())
        cols.extend(cc.ravel().tolist())
        data.extend(A_nn.ravel().tolist())

    K = sp.coo_matrix((np.asarray(data, float), (np.asarray(rows, int), np.asarray(cols, int))), shape=(n_total, n_total))
    K.sum_duplicates()
    return K.tocsr()
