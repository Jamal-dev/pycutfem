from __future__ import annotations

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem import transform
from pycutfem.integration.quadrature import gauss_legendre

from .interface import NonMatchingInterface
from .nitsche import _field_order


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


def scalar_jump_L2(
    *,
    interface: NonMatchingInterface,
    dh_neg: DofHandler,
    u_neg: np.ndarray,
    dh_pos: DofHandler,
    u_pos: np.ndarray,
    field: str = "u",
    quad_order: int | None = None,
) -> float:
    """Compute ||u_neg - u_pos||_{L2(Gamma)} using the interface quadrature."""
    if interface.n_segments() <= 0:
        return 0.0

    if quad_order is None:
        p_pos = _field_order(dh_pos, field)
        p_neg = _field_order(dh_neg, field)
        quad_order = int(2 * max(p_pos, p_neg) + 2)

    qpts, qw = _interface_quadrature(interface, int(quad_order))
    me_pos = dh_pos.mixed_element
    me_neg = dh_neg.mixed_element
    mesh_pos = interface.mesh_pos
    mesh_neg = interface.mesh_neg

    acc = 0.0
    for s in range(interface.n_segments()):
        pos_eid = int(interface.pos_elem_ids[s])
        neg_eid = int(interface.neg_elem_ids[s])
        gd_pos = np.asarray(dh_pos.get_elemental_dofs(pos_eid), dtype=int)
        gd_neg = np.asarray(dh_neg.get_elemental_dofs(neg_eid), dtype=int)
        uel_pos = np.asarray(u_pos, dtype=float)[gd_pos]
        uel_neg = np.asarray(u_neg, dtype=float)[gd_neg]
        for q in range(qpts.shape[1]):
            wq = float(qw[s, q])
            if wq == 0.0:
                continue
            xq = np.asarray(qpts[s, q], dtype=float)
            xi_p, eta_p = transform.inverse_mapping(mesh_pos, pos_eid, xq)
            xi_n, eta_n = transform.inverse_mapping(mesh_neg, neg_eid, xq)
            phi_p = np.asarray(me_pos.basis(field, float(xi_p), float(eta_p)), dtype=float).ravel()
            phi_n = np.asarray(me_neg.basis(field, float(xi_n), float(eta_n)), dtype=float).ravel()
            jump = float(phi_n @ uel_neg - phi_p @ uel_pos)
            acc += wq * jump * jump
    return float(np.sqrt(acc))


def poisson_flux_mismatch_L2(
    *,
    interface: NonMatchingInterface,
    dh_neg: DofHandler,
    u_neg: np.ndarray,
    dh_pos: DofHandler,
    u_pos: np.ndarray,
    field: str = "u",
    k_neg: float = 1.0,
    k_pos: float = 1.0,
    quad_order: int | None = None,
) -> float:
    """Compute ||k_neg ∇u_neg·n - k_pos ∇u_pos·n||_{L2(Gamma)} (n is neg->pos)."""
    if interface.n_segments() <= 0:
        return 0.0

    if quad_order is None:
        p_pos = _field_order(dh_pos, field)
        p_neg = _field_order(dh_neg, field)
        quad_order = int(2 * max(p_pos, p_neg) + 2)

    qpts, qw = _interface_quadrature(interface, int(quad_order))
    me_pos = dh_pos.mixed_element
    me_neg = dh_neg.mixed_element
    mesh_pos = interface.mesh_pos
    mesh_neg = interface.mesh_neg

    k_pos = float(k_pos)
    k_neg = float(k_neg)
    acc = 0.0
    for s in range(interface.n_segments()):
        pos_eid = int(interface.pos_elem_ids[s])
        neg_eid = int(interface.neg_elem_ids[s])
        n_vec = np.asarray(interface.n[s], dtype=float)
        gd_pos = np.asarray(dh_pos.get_elemental_dofs(pos_eid), dtype=int)
        gd_neg = np.asarray(dh_neg.get_elemental_dofs(neg_eid), dtype=int)
        uel_pos = np.asarray(u_pos, dtype=float)[gd_pos]
        uel_neg = np.asarray(u_neg, dtype=float)[gd_neg]
        for q in range(qpts.shape[1]):
            wq = float(qw[s, q])
            if wq == 0.0:
                continue
            xq = np.asarray(qpts[s, q], dtype=float)
            xi_p, eta_p = transform.inverse_mapping(mesh_pos, pos_eid, xq)
            Jp = transform.jacobian(mesh_pos, pos_eid, (float(xi_p), float(eta_p)))
            Jip = np.linalg.inv(Jp)
            gphi_p = np.asarray(me_pos.grad_basis(field, float(xi_p), float(eta_p)), dtype=float) @ Jip
            grad_pos = np.asarray(uel_pos, dtype=float) @ gphi_p
            gn_pos = float(grad_pos @ n_vec)

            xi_n, eta_n = transform.inverse_mapping(mesh_neg, neg_eid, xq)
            Jn = transform.jacobian(mesh_neg, neg_eid, (float(xi_n), float(eta_n)))
            Jin = np.linalg.inv(Jn)
            gphi_n = np.asarray(me_neg.grad_basis(field, float(xi_n), float(eta_n)), dtype=float) @ Jin
            grad_neg = np.asarray(uel_neg, dtype=float) @ gphi_n
            gn_neg = float(grad_neg @ n_vec)

            mismatch = k_neg * gn_neg - k_pos * gn_pos
            acc += wq * mismatch * mismatch
    return float(np.sqrt(acc))


def stokes_velocity_jump_L2(
    *,
    interface: NonMatchingInterface,
    dh_neg: DofHandler,
    U_neg: np.ndarray,
    dh_pos: DofHandler,
    U_pos: np.ndarray,
    u_fields: tuple[str, str] = ("ux", "uy"),
    quad_order: int | None = None,
) -> float:
    """Compute ||u_neg - u_pos||_{L2(Gamma)} for a 2D velocity field."""
    if interface.n_segments() <= 0:
        return 0.0

    if quad_order is None:
        p_u = max(_field_order(dh_pos, u_fields[0]), _field_order(dh_neg, u_fields[0]))
        quad_order = int(2 * p_u + 2)

    qpts, qw = _interface_quadrature(interface, int(quad_order))
    me_pos = dh_pos.mixed_element
    me_neg = dh_neg.mixed_element
    mesh_pos = interface.mesh_pos
    mesh_neg = interface.mesh_neg

    acc = 0.0
    for s in range(interface.n_segments()):
        pos_eid = int(interface.pos_elem_ids[s])
        neg_eid = int(interface.neg_elem_ids[s])
        gd_pos = np.asarray(dh_pos.get_elemental_dofs(pos_eid), dtype=int)
        gd_neg = np.asarray(dh_neg.get_elemental_dofs(neg_eid), dtype=int)
        Uel_pos = np.asarray(U_pos, dtype=float)[gd_pos]
        Uel_neg = np.asarray(U_neg, dtype=float)[gd_neg]
        for q in range(qpts.shape[1]):
            wq = float(qw[s, q])
            if wq == 0.0:
                continue
            xq = np.asarray(qpts[s, q], dtype=float)
            xi_p, eta_p = transform.inverse_mapping(mesh_pos, pos_eid, xq)
            xi_n, eta_n = transform.inverse_mapping(mesh_neg, neg_eid, xq)
            phi_ux_p = np.asarray(me_pos.basis(u_fields[0], float(xi_p), float(eta_p)), dtype=float).ravel()
            phi_uy_p = np.asarray(me_pos.basis(u_fields[1], float(xi_p), float(eta_p)), dtype=float).ravel()
            phi_ux_n = np.asarray(me_neg.basis(u_fields[0], float(xi_n), float(eta_n)), dtype=float).ravel()
            phi_uy_n = np.asarray(me_neg.basis(u_fields[1], float(xi_n), float(eta_n)), dtype=float).ravel()

            up = np.array([float(phi_ux_p @ Uel_pos), float(phi_uy_p @ Uel_pos)], dtype=float)
            un = np.array([float(phi_ux_n @ Uel_neg), float(phi_uy_n @ Uel_neg)], dtype=float)
            du = un - up
            acc += wq * float(du @ du)
    return float(np.sqrt(acc))


def stokes_traction_mismatch_L2(
    *,
    interface: NonMatchingInterface,
    dh_neg: DofHandler,
    U_neg: np.ndarray,
    dh_pos: DofHandler,
    U_pos: np.ndarray,
    mu_neg: float,
    mu_pos: float,
    u_fields: tuple[str, str] = ("ux", "uy"),
    p_field: str = "p",
    quad_order: int | None = None,
) -> float:
    """Compute ||σ_neg n - σ_pos n||_{L2(Gamma)} (n is neg->pos)."""
    if interface.n_segments() <= 0:
        return 0.0

    if quad_order is None:
        p_u = max(_field_order(dh_pos, u_fields[0]), _field_order(dh_neg, u_fields[0]))
        quad_order = int(2 * p_u + 2)

    qpts, qw = _interface_quadrature(interface, int(quad_order))
    me_pos = dh_pos.mixed_element
    me_neg = dh_neg.mixed_element
    mesh_pos = interface.mesh_pos
    mesh_neg = interface.mesh_neg

    mu_pos = float(mu_pos)
    mu_neg = float(mu_neg)

    acc = 0.0
    for s in range(interface.n_segments()):
        pos_eid = int(interface.pos_elem_ids[s])
        neg_eid = int(interface.neg_elem_ids[s])
        n_vec = np.asarray(interface.n[s], dtype=float)
        nx, ny = float(n_vec[0]), float(n_vec[1])
        gd_pos = np.asarray(dh_pos.get_elemental_dofs(pos_eid), dtype=int)
        gd_neg = np.asarray(dh_neg.get_elemental_dofs(neg_eid), dtype=int)
        Uel_pos = np.asarray(U_pos, dtype=float)[gd_pos]
        Uel_neg = np.asarray(U_neg, dtype=float)[gd_neg]
        for q in range(qpts.shape[1]):
            wq = float(qw[s, q])
            if wq == 0.0:
                continue
            xq = np.asarray(qpts[s, q], dtype=float)

            # --- POS side
            xi_p, eta_p = transform.inverse_mapping(mesh_pos, pos_eid, xq)
            Jp = transform.jacobian(mesh_pos, pos_eid, (float(xi_p), float(eta_p)))
            Jip = np.linalg.inv(Jp)
            g_ux_p = np.asarray(me_pos.grad_basis(u_fields[0], float(xi_p), float(eta_p)), dtype=float) @ Jip
            g_uy_p = np.asarray(me_pos.grad_basis(u_fields[1], float(xi_p), float(eta_p)), dtype=float) @ Jip
            phi_p_p = np.asarray(me_pos.basis(p_field, float(xi_p), float(eta_p)), dtype=float).ravel()

            dux_p = np.asarray(Uel_pos, dtype=float) @ g_ux_p
            duy_p = np.asarray(Uel_pos, dtype=float) @ g_uy_p
            p_p = float(phi_p_p @ Uel_pos)

            uxx, uxy = float(dux_p[0]), float(dux_p[1])
            uyx, uyy = float(duy_p[0]), float(duy_p[1])
            sxx = 2.0 * mu_pos * uxx - p_p
            sxy = mu_pos * (uxy + uyx)
            syy = 2.0 * mu_pos * uyy - p_p
            tpos = np.array([sxx * nx + sxy * ny, sxy * nx + syy * ny], dtype=float)

            # --- NEG side
            xi_n, eta_n = transform.inverse_mapping(mesh_neg, neg_eid, xq)
            Jn = transform.jacobian(mesh_neg, neg_eid, (float(xi_n), float(eta_n)))
            Jin = np.linalg.inv(Jn)
            g_ux_n = np.asarray(me_neg.grad_basis(u_fields[0], float(xi_n), float(eta_n)), dtype=float) @ Jin
            g_uy_n = np.asarray(me_neg.grad_basis(u_fields[1], float(xi_n), float(eta_n)), dtype=float) @ Jin
            phi_p_n = np.asarray(me_neg.basis(p_field, float(xi_n), float(eta_n)), dtype=float).ravel()

            dux_n = np.asarray(Uel_neg, dtype=float) @ g_ux_n
            duy_n = np.asarray(Uel_neg, dtype=float) @ g_uy_n
            p_n = float(phi_p_n @ Uel_neg)

            uxx, uxy = float(dux_n[0]), float(dux_n[1])
            uyx, uyy = float(duy_n[0]), float(duy_n[1])
            sxx = 2.0 * mu_neg * uxx - p_n
            sxy = mu_neg * (uxy + uyx)
            syy = 2.0 * mu_neg * uyy - p_n
            tneg = np.array([sxx * nx + sxy * ny, sxy * nx + syy * ny], dtype=float)

            dt = tneg - tpos
            acc += wq * float(dt @ dt)
    return float(np.sqrt(acc))

