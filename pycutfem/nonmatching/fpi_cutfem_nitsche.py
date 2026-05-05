"""Non-matching FPI Nitsche coupling on a CutFEM-resolved interface.

This module assembles the *interface* residual/Jacobian contributions of
`examples/utils/fpi/interface.py`, but for the two-mesh setting:

  - fluid unknowns live on a CutFEM background mesh (pos side),
  - poroelastic unknowns live on a separate body-fitted mesh (neg side),
  - the interface Γ^FP is integrated using the CutFEM interface segments of the
    fluid mesh (a partition of Γ); poro basis functions are evaluated by inverse
    mapping into the containing poro element for each segment.

Important
---------
This avoids relying on Mesh edge tags like "interface" (which have special
meaning in the CutFEM assembly when the interface aligns with mesh edges).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp

from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem import transform
from pycutfem.integration.quadrature import gauss_legendre


Array = np.ndarray


@dataclass(frozen=True, slots=True)
class CutFEMNonMatchingInterface:
    """Interface described by CutFEM segments on the fluid mesh.

    Conventions:
      - `n` is oriented from poro (neg) to fluid (pos).
      - Fluid outward normal is `nF = -n`.
    """

    pos_elem_ids: Array  # (n_seg,)
    neg_elem_ids: Array  # (n_seg,)
    P0: Array  # (n_seg, 2)
    P1: Array  # (n_seg, 2)
    n: Array  # (n_seg, 2) unit normals (neg->pos)
    h_pos: Array  # (n_seg,)
    h_neg: Array  # (n_seg,)

    def n_segments(self) -> int:
        return int(self.P0.shape[0])


def _find_owner_element(mesh, x: Array, *, tol: float = 1.0e-12) -> int:
    """Return an element id whose reference image contains point `x`."""
    x = np.asarray(x, float).reshape(2,)
    cand = None
    if hasattr(mesh, "find_owner_element_fast"):
        try:
            cand = list(mesh.find_owner_element_fast(x, tol))
        except Exception:
            cand = None
    if not cand:
        cand = list(range(len(mesh.elements_list)))
    for eid in cand:
        try:
            xi, eta = transform.inverse_mapping(mesh, int(eid), x)
        except Exception:
            continue
        if mesh.element_type == "quad":
            if (-1.0 - tol <= float(xi) <= 1.0 + tol) and (-1.0 - tol <= float(eta) <= 1.0 + tol):
                return int(eid)
        else:
            if float(xi) >= -tol and float(eta) >= -tol and float(xi) + float(eta) <= 1.0 + tol:
                return int(eid)
    # centroid fallback
    d = [np.linalg.norm(np.asarray(e.centroid(), float) - x) for e in mesh.elements_list]
    return int(np.argmin(d))


def build_interface_from_cutfem_segments(
    *,
    mesh_f,
    fluid_ls,
    poro_ls,
    mesh_p,
    x0: float,
    tol_inlet: float = 5.0e-6,
    tol_owner: float = 1.0e-10,
) -> tuple[CutFEMNonMatchingInterface, dict[str, Array]]:
    """Build non-matching Γ^FP segments from the already-built fluid interface segments.

    Returns:
      (interface_fp, extra) where `extra` also contains inlet segments:
        extra["inlet_P0"], extra["inlet_P1"] for Γ^{F,N} at x=x0.
    """
    pos_ids: list[int] = []
    neg_ids: list[int] = []
    P0: list[Array] = []
    P1: list[Array] = []
    nn: list[Array] = []
    h_pos: list[float] = []
    h_neg: list[float] = []

    inlet_P0: list[Array] = []
    inlet_P1: list[Array] = []
    inlet_pos_ids: list[int] = []

    eps = 1.0e-8

    for elem in mesh_f.elements_list:
        segs = getattr(elem, "interface_segments", None)
        if not segs:
            continue
        for a, b in segs:
            p0 = np.asarray(a, float)
            p1 = np.asarray(b, float)
            mid = 0.5 * (p0 + p1)
            if abs(float(mid[0]) - float(x0)) <= float(tol_inlet):
                inlet_P0.append(p0)
                inlet_P1.append(p1)
                inlet_pos_ids.append(int(elem.id))
                continue

            # Poro owner element for this segment
            neg_eid = _find_owner_element(mesh_p, mid, tol=tol_owner)

            # Discrete segment normal (paper Remark 8): use the *piecewise-linear*
            # interface geometry rather than the smooth level-set gradient.
            t = p1 - p0
            tnorm = float(np.linalg.norm(t))
            if tnorm <= 1.0e-14:
                continue
            nvec = np.array([t[1], -t[0]], dtype=float) / tnorm
            probe = mid + eps * nvec
            try:
                inside_fluid = float(fluid_ls(probe)) > 0.0
            except Exception:
                inside_fluid = True
            if not inside_fluid:
                nvec = -nvec

            pos_ids.append(int(elem.id))
            neg_ids.append(int(neg_eid))
            P0.append(p0)
            P1.append(p1)
            nn.append(nvec)
            h_pos.append(float(mesh_f.element_char_length(int(elem.id)) or 0.0))
            h_neg.append(float(mesh_p.element_char_length(int(neg_eid)) or 0.0))

    if not P0:
        raise ValueError("No CutFEM interface segments found for Γ^FP (check mesh_f.build_interface_segments).")

    iface = CutFEMNonMatchingInterface(
        pos_elem_ids=np.asarray(pos_ids, dtype=int),
        neg_elem_ids=np.asarray(neg_ids, dtype=int),
        P0=np.asarray(P0, dtype=float),
        P1=np.asarray(P1, dtype=float),
        n=np.asarray(nn, dtype=float),
        h_pos=np.asarray(h_pos, dtype=float),
        h_neg=np.asarray(h_neg, dtype=float),
    )
    extra = dict(
        inlet_pos_elem_ids=np.asarray(inlet_pos_ids, dtype=int),
        inlet_P0=np.asarray(inlet_P0, dtype=float) if inlet_P0 else np.zeros((0, 2), dtype=float),
        inlet_P1=np.asarray(inlet_P1, dtype=float) if inlet_P1 else np.zeros((0, 2), dtype=float),
    )
    return iface, extra


def _eps_dot_n_for_vx(grad_phi: Array, nF: Array) -> Array:
    """Return (ε( [phi,0] )·nF) as a 2-vector."""
    dphix, dphiy = float(grad_phi[0]), float(grad_phi[1])
    nFx, nFy = float(nF[0]), float(nF[1])
    return np.array([dphix * nFx + 0.5 * dphiy * nFy, 0.5 * dphiy * nFx], dtype=float)


def _eps_dot_n_for_vy(grad_phi: Array, nF: Array) -> Array:
    """Return (ε( [0,phi] )·nF) as a 2-vector."""
    dphix, dphiy = float(grad_phi[0]), float(grad_phi[1])
    nFx, nFy = float(nF[0]), float(nF[1])
    return np.array([0.5 * dphix * nFy, 0.5 * dphix * nFx + dphiy * nFy], dtype=float)


def assemble_fpi_interface_nitsche(
    *,
    interface: CutFEMNonMatchingInterface,
    dh_f: DofHandler,
    dh_p: DofHandler,
    # current solution vectors (global per-mesh DOF order)
    Uf: Array,
    Up: Array,
    # previous displacement (poro) for u-dot
    Up_n: Array,
    dt: float,
    # parameters
    rho_f: float,
    mu_f: float,
    porosity: float,
    beta_BJ: float,
    kappa: float,
    gamma_n: float,
    gamma_t: float,
    zeta: float,
    vF_inf: float,
    c_v_gamma: float,
    c_t_gamma: float,
    theta: float = 1.0,
    quad_order: int = 6,
    # manufactured interface data (functions of x,y,nF)
    g_sigma: Callable[[Array, Array], Array] | None = None,
    g_sigma_n: Callable[[Array, Array], float] | None = None,
    g_n: Callable[[Array, Array], Array] | None = None,
    g_t: Callable[[Array, Array], Array] | None = None,
) -> tuple[sp.csr_matrix, Array]:
    """Assemble interface Jacobian and residual for the coupled two-mesh system.

    Global ordering is `[U_f, U_p]` (fluid DOFs first, then poro DOFs).
    """
    if interface.n_segments() <= 0:
        n_total = int(dh_f.total_dofs) + int(dh_p.total_dofs)
        return sp.csr_matrix((n_total, n_total), dtype=float), np.zeros(n_total, dtype=float)

    mesh_f = dh_f.mixed_element.mesh
    mesh_p = dh_p.mixed_element.mesh
    me_f = dh_f.mixed_element
    me_p = dh_p.mixed_element

    # field names (must match caller)
    vfx, vfy, pf = "v_pos_x", "v_pos_y", "p_pos_"
    vpx, vpy, upx, upy = "v_neg_x", "v_neg_y", "u_neg_x", "u_neg_y"

    xi, w_ref = gauss_legendre(int(quad_order))
    xi = np.asarray(xi, dtype=float)
    w_ref = np.asarray(w_ref, dtype=float)

    n_f = int(dh_f.total_dofs)
    n_p = int(dh_p.total_dofs)
    n_total = n_f + n_p

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    R = np.zeros(n_total, dtype=float)

    # defaults: physical case => all zeros
    if g_sigma is None:
        g_sigma = lambda x, nF: np.zeros(2, dtype=float)
    if g_sigma_n is None:
        g_sigma_n = lambda x, nF: 0.0
    if g_n is None:
        g_n = lambda x, nF: np.zeros(2, dtype=float)
    if g_t is None:
        g_t = lambda x, nF: np.zeros(2, dtype=float)

    mu_f = float(mu_f)
    rho_f = float(rho_f)
    porosity = float(porosity)
    beta_BJ = float(beta_BJ)
    kappa = float(kappa)
    gamma_n = float(gamma_n)
    gamma_t = float(gamma_t)
    zeta = float(zeta)
    dt = float(dt)
    th = float(theta)
    vF_inf = float(vF_inf)

    for s in range(interface.n_segments()):
        ef = int(interface.pos_elem_ids[s])
        ep = int(interface.neg_elem_ids[s])

        n_vec = np.asarray(interface.n[s], dtype=float).reshape(2,)
        nF = -n_vec  # fluid outward

        # paper h_gamma: take arithmetic mean (two meshes)
        h_gamma = 0.5 * (float(interface.h_pos[s]) + float(interface.h_neg[s]))
        if h_gamma <= 0.0:
            h_gamma = 1.0

        phi_gamma_F = (
            mu_f
            + h_gamma * float(c_v_gamma) * rho_f * vF_inf
            + (h_gamma * h_gamma) * float(c_t_gamma) * (rho_f / (th * dt))
        )

        denom_t = kappa * mu_f + gamma_t * h_gamma
        t1 = zeta * (gamma_t * h_gamma / denom_t)
        t2 = mu_f / denom_t

        # segment quadrature points
        P0 = np.asarray(interface.P0[s], float)
        P1 = np.asarray(interface.P1[s], float)
        mid = 0.5 * (P0 + P1)
        half = 0.5 * (P1 - P0)
        segJ = float(np.linalg.norm(half))
        if segJ <= 0.0:
            continue

        # local dofs per field
        gd_vfx = np.asarray(dh_f.element_maps[vfx][ef], dtype=int)
        gd_vfy = np.asarray(dh_f.element_maps[vfy][ef], dtype=int)
        gd_pf = np.asarray(dh_f.element_maps[pf][ef], dtype=int)
        gd_vpx = np.asarray(dh_p.element_maps[vpx][ep], dtype=int) + n_f
        gd_vpy = np.asarray(dh_p.element_maps[vpy][ep], dtype=int) + n_f
        gd_upx = np.asarray(dh_p.element_maps[upx][ep], dtype=int) + n_f
        gd_upy = np.asarray(dh_p.element_maps[upy][ep], dtype=int) + n_f

        # coefficient vectors on this element
        vf_x = np.asarray(Uf, float)[gd_vfx]
        vf_y = np.asarray(Uf, float)[gd_vfy]
        pf_c = np.asarray(Uf, float)[gd_pf]
        vp_x = np.asarray(Up, float)[gd_vpx - n_f]
        vp_y = np.asarray(Up, float)[gd_vpy - n_f]
        up_x = np.asarray(Up, float)[gd_upx - n_f]
        up_y = np.asarray(Up, float)[gd_upy - n_f]
        upn_x = np.asarray(Up_n, float)[gd_upx - n_f]
        upn_y = np.asarray(Up_n, float)[gd_upy - n_f]

        for q, wq_ref in enumerate(w_ref):
            xq = mid + float(xi[q]) * half
            wq = float(wq_ref) * segJ
            if wq == 0.0:
                continue

            # reference coordinates on each element
            xi_f, eta_f = transform.inverse_mapping(mesh_f, ef, xq)
            xi_p, eta_p = transform.inverse_mapping(mesh_p, ep, xq)

            Jf = transform.jacobian(mesh_f, ef, (float(xi_f), float(eta_f)))
            Jp = transform.jacobian(mesh_p, ep, (float(xi_p), float(eta_p)))
            Jif = np.linalg.inv(Jf)
            Jip = np.linalg.inv(Jp)

            # basis and grads (MixedElement.basis/grad_basis are over the full element DOF vector)
            sl_vfx = me_f.component_dof_slices[vfx]
            sl_vfy = me_f.component_dof_slices[vfy]
            sl_pf = me_f.component_dof_slices[pf]
            sl_vpx = me_p.component_dof_slices[vpx]
            sl_vpy = me_p.component_dof_slices[vpy]
            sl_upx = me_p.component_dof_slices[upx]
            sl_upy = me_p.component_dof_slices[upy]

            phi_vfx = np.asarray(me_f.basis(vfx, float(xi_f), float(eta_f)), float).ravel()[sl_vfx]
            phi_vfy = np.asarray(me_f.basis(vfy, float(xi_f), float(eta_f)), float).ravel()[sl_vfy]
            phi_pf = np.asarray(me_f.basis(pf, float(xi_f), float(eta_f)), float).ravel()[sl_pf]
            g_vfx = (np.asarray(me_f.grad_basis(vfx, float(xi_f), float(eta_f)), float) @ Jif)[sl_vfx]
            g_vfy = (np.asarray(me_f.grad_basis(vfy, float(xi_f), float(eta_f)), float) @ Jif)[sl_vfy]

            phi_vpx = np.asarray(me_p.basis(vpx, float(xi_p), float(eta_p)), float).ravel()[sl_vpx]
            phi_vpy = np.asarray(me_p.basis(vpy, float(xi_p), float(eta_p)), float).ravel()[sl_vpy]
            phi_upx = np.asarray(me_p.basis(upx, float(xi_p), float(eta_p)), float).ravel()[sl_upx]
            phi_upy = np.asarray(me_p.basis(upy, float(xi_p), float(eta_p)), float).ravel()[sl_upy]

            # solution values
            vF = np.array([float(vf_x @ phi_vfx), float(vf_y @ phi_vfy)], dtype=float)
            pF = float(pf_c @ phi_pf)
            vP = np.array([float(vp_x @ phi_vpx), float(vp_y @ phi_vpy)], dtype=float)
            uP = np.array([float(up_x @ phi_upx), float(up_y @ phi_upy)], dtype=float)
            uPn = np.array([float(upn_x @ phi_upx), float(upn_y @ phi_upy)], dtype=float)
            u_dot = (uP - uPn) / dt

            # grad vF and symmetric gradient eps(vF)
            dvx = vf_x @ g_vfx  # (2,)
            dvy = vf_y @ g_vfy  # (2,)
            grad_v = np.array([[float(dvx[0]), float(dvx[1])], [float(dvy[0]), float(dvy[1])]], dtype=float)
            eps_v = 0.5 * (grad_v + grad_v.T)

            tractionF = (2.0 * mu_f) * (eps_v @ nF) - pF * nF  # σ nF
            trn = float(tractionF @ nF)
            traction_n = trn * nF
            traction_t = tractionF - traction_n

            # manufactured data at this point (use discrete nF)
            gsig = np.asarray(g_sigma(xq, nF), float).reshape(2,)
            gsig_n = float(g_sigma_n(xq, nF))
            gn = np.asarray(g_n(xq, nF), float).reshape(2,)
            gt = np.asarray(g_t(xq, nF), float).reshape(2,)

            kin = vF - u_dot - porosity * (vP - u_dot) - gn
            kin_n = float(kin @ nF) * nF

            cBJ = vF - u_dot - beta_BJ * porosity * (vP - u_dot) + kappa * tractionF - gt
            c_t = cBJ - float(cBJ @ nF) * nF

            # -------------------------
            # Residual contributions
            # -------------------------
            # fluid velocity test dofs (v_pos_x)
            for i in range(phi_vfx.size):
                gi = int(gd_vfx[i])
                wv = float(phi_vfx[i])
                # R_n: test_jump_n includes -dvF_test
                R[gi] += wq * (-wv * float(traction_n[0]))
                # R_n: adjoint/consistent term with eps(test)
                epsn = _eps_dot_n_for_vx(g_vfx[i], nF)
                R[gi] += wq * (-float((zeta * 2.0 * mu_f) * (epsn @ kin_n)))
                # R_n: penalty
                R[gi] += wq * ((phi_gamma_F * gamma_n / h_gamma) * float(wv * (kin_n[0])))
                # R_t: -dvF_test · traction_t
                R[gi] += wq * (-wv * float(traction_t[0]))
                # R_t: t1 term
                R[gi] += wq * (t1 * float(((-2.0 * mu_f) * epsn) @ c_t))
                # R_t: t2 term
                R[gi] += wq * (t2 * float(wv * c_t[0]))

            # fluid velocity test dofs (v_pos_y)
            for i in range(phi_vfy.size):
                gi = int(gd_vfy[i])
                wv = float(phi_vfy[i])
                R[gi] += wq * (-wv * float(traction_n[1]))
                epsn = _eps_dot_n_for_vy(g_vfy[i], nF)
                R[gi] += wq * (-float((zeta * 2.0 * mu_f) * (epsn @ kin_n)))
                R[gi] += wq * ((phi_gamma_F * gamma_n / h_gamma) * float(wv * (kin_n[1])))
                R[gi] += wq * (-wv * float(traction_t[1]))
                R[gi] += wq * (t1 * float(((-2.0 * mu_f) * epsn) @ c_t))
                R[gi] += wq * (t2 * float(wv * c_t[1]))

            # fluid pressure test dofs
            for i in range(phi_pf.size):
                gi = int(gd_pf[i])
                wp = float(phi_pf[i])
                # - inner(dpF_test*nF, kin_n) = -wp * (nF·kin_n) = -wp*(kin·nF)
                R[gi] += wq * (-wp * float(kin @ nF))

            # poro velocity tests: dvP_test in normal only
            for i in range(phi_vpx.size):
                gi = int(gd_vpx[i])
                wv = float(phi_vpx[i])
                R[gi] += wq * (wv * float(traction_n[0]))  # test_jump_n
                R[gi] += wq * (-wv * gsig_n * float(nF[0]))
                R[gi] += wq * (-(phi_gamma_F * gamma_n / h_gamma) * wv * float(kin_n[0]))

            for i in range(phi_vpy.size):
                gi = int(gd_vpy[i])
                wv = float(phi_vpy[i])
                R[gi] += wq * (wv * float(traction_n[1]))
                R[gi] += wq * (-wv * gsig_n * float(nF[1]))
                R[gi] += wq * (-(phi_gamma_F * gamma_n / h_gamma) * wv * float(kin_n[1]))

            # poro displacement tests
            proj_gsig_n = float(gsig @ nF) * nF
            proj_gsig_t = gsig - proj_gsig_n
            for i in range(phi_upx.size):
                gi = int(gd_upx[i])
                wu = float(phi_upx[i])
                R[gi] += wq * (wu * float(traction_n[0]))  # test_jump_n
                R[gi] += wq * (-wu * float(proj_gsig_n[0]))
                R[gi] += wq * (-(phi_gamma_F * gamma_n / h_gamma) * wu * float(kin_n[0]))
                R[gi] += wq * (wu * float(traction_t[0]))  # tangential: +duP_test · traction_t
                R[gi] += wq * (-wu * float(proj_gsig_t[0]))
                R[gi] += wq * (t2 * float((-wu) * c_t[0]))

            for i in range(phi_upy.size):
                gi = int(gd_upy[i])
                wu = float(phi_upy[i])
                R[gi] += wq * (wu * float(traction_n[1]))
                R[gi] += wq * (-wu * float(proj_gsig_n[1]))
                R[gi] += wq * (-(phi_gamma_F * gamma_n / h_gamma) * wu * float(kin_n[1]))
                R[gi] += wq * (wu * float(traction_t[1]))
                R[gi] += wq * (-wu * float(proj_gsig_t[1]))
                R[gi] += wq * (t2 * float((-wu) * c_t[1]))

            # -------------------------
            # Jacobian contributions
            # (assembled in a matrix-free friendly COO fashion)
            # -------------------------
            # Precompute dtraction basis vectors for fluid trial dofs.
            # dvF_x trials
            dtr_vfx = []
            dtrn_vfx = []
            dtrt_vfx = []
            dkin_vfx = []
            dkin_n_vfx = []
            dc_t_vfx = []
            for j in range(phi_vfx.size):
                epsn = _eps_dot_n_for_vx(g_vfx[j], nF)
                tvec = (2.0 * mu_f) * epsn  # contribution to tractionF from this basis (no pressure part)
                trn_j = float(tvec @ nF) * nF
                dtr_vfx.append(tvec)
                dtrn_vfx.append(trn_j)
                dtrt_vfx.append(tvec - trn_j)
                dv = np.array([float(phi_vfx[j]), 0.0], dtype=float)
                dkin = dv
                dkin_n = float(dkin @ nF) * nF
                dkin_vfx.append(dkin)
                dkin_n_vfx.append(dkin_n)
                dc = dv + kappa * tvec
                dc_t_vfx.append(dc - float(dc @ nF) * nF)

            # dvF_y trials
            dtr_vfy = []
            dtrn_vfy = []
            dtrt_vfy = []
            dkin_vfy = []
            dkin_n_vfy = []
            dc_t_vfy = []
            for j in range(phi_vfy.size):
                epsn = _eps_dot_n_for_vy(g_vfy[j], nF)
                tvec = (2.0 * mu_f) * epsn
                trn_j = float(tvec @ nF) * nF
                dtr_vfy.append(tvec)
                dtrn_vfy.append(trn_j)
                dtrt_vfy.append(tvec - trn_j)
                dv = np.array([0.0, float(phi_vfy[j])], dtype=float)
                dkin = dv
                dkin_n = float(dkin @ nF) * nF
                dkin_vfy.append(dkin)
                dkin_n_vfy.append(dkin_n)
                dc = dv + kappa * tvec
                dc_t_vfy.append(dc - float(dc @ nF) * nF)

            # dpF trials
            dtr_pf = []
            dtrn_pf = []
            dtrt_pf = []
            dc_t_pf = []
            for j in range(phi_pf.size):
                tvec = -float(phi_pf[j]) * nF
                trn_j = float(tvec @ nF) * nF
                dtr_pf.append(tvec)
                dtrn_pf.append(trn_j)
                dtrt_pf.append(tvec - trn_j)
                dc = kappa * tvec
                dc_t_pf.append(dc - float(dc @ nF) * nF)

            # dvP trials (poro)
            dkin_vpx = []
            dkin_n_vpx = []
            dc_t_vpx = []
            for j in range(phi_vpx.size):
                dv = np.array([-porosity * float(phi_vpx[j]), 0.0], dtype=float)
                dkin_n = float(dv @ nF) * nF
                dkin_vpx.append(dv)
                dkin_n_vpx.append(dkin_n)
                dc = np.array([-beta_BJ * porosity * float(phi_vpx[j]), 0.0], dtype=float)
                dc_t_vpx.append(dc - float(dc @ nF) * nF)

            dkin_vpy = []
            dkin_n_vpy = []
            dc_t_vpy = []
            for j in range(phi_vpy.size):
                dv = np.array([0.0, -porosity * float(phi_vpy[j])], dtype=float)
                dkin_n = float(dv @ nF) * nF
                dkin_vpy.append(dv)
                dkin_n_vpy.append(dkin_n)
                dc = np.array([0.0, -beta_BJ * porosity * float(phi_vpy[j])], dtype=float)
                dc_t_vpy.append(dc - float(dc @ nF) * nF)

            # duP trials via u_dot
            dkin_upx = []
            dkin_n_upx = []
            dc_t_upx = []
            for j in range(phi_upx.size):
                d_u_dot = np.array([float(phi_upx[j]) / dt, 0.0], dtype=float)
                dv = -(1.0 - porosity) * d_u_dot
                dkin_n = float(dv @ nF) * nF
                dkin_upx.append(dv)
                dkin_n_upx.append(dkin_n)
                dc = -(1.0 - beta_BJ * porosity) * d_u_dot
                dc_t_upx.append((dc - float(dc @ nF) * nF))

            dkin_upy = []
            dkin_n_upy = []
            dc_t_upy = []
            for j in range(phi_upy.size):
                d_u_dot = np.array([0.0, float(phi_upy[j]) / dt], dtype=float)
                dv = -(1.0 - porosity) * d_u_dot
                dkin_n = float(dv @ nF) * nF
                dkin_upy.append(dv)
                dkin_n_upy.append(dkin_n)
                dc = -(1.0 - beta_BJ * porosity) * d_u_dot
                dc_t_upy.append((dc - float(dc @ nF) * nF))

            # Helper: add COO block from row ids, col ids, with value = wq * a*b (scalar)
            def _add_outer(row_ids, col_ids, avals, bvals, scale):
                for ii, ri in enumerate(row_ids):
                    ai = float(avals[ii])
                    if ai == 0.0:
                        continue
                    for jj, cj in enumerate(col_ids):
                        bj = float(bvals[jj])
                        if bj == 0.0:
                            continue
                        rows.append(int(ri))
                        cols.append(int(cj))
                        data.append(wq * float(scale) * ai * bj)

            # --- J_n: inner(test_jump_n, dtraction_n) ---
            # fluid tests (vfx) with dvF trials (vfx,vfy,pf)
            for i in range(phi_vfx.size):
                ri = int(gd_vfx[i])
                wi = float(phi_vfx[i])
                for j in range(phi_vfx.size):
                    rows.append(ri); cols.append(int(gd_vfx[j])); data.append(wq * (-wi) * float(dtrn_vfx[j][0]))
                for j in range(phi_vfy.size):
                    rows.append(ri); cols.append(int(gd_vfy[j])); data.append(wq * (-wi) * float(dtrn_vfy[j][0]))
                for j in range(phi_pf.size):
                    rows.append(ri); cols.append(int(gd_pf[j])); data.append(wq * (-wi) * float(dtrn_pf[j][0]))

            for i in range(phi_vfy.size):
                ri = int(gd_vfy[i])
                wi = float(phi_vfy[i])
                for j in range(phi_vfx.size):
                    rows.append(ri); cols.append(int(gd_vfx[j])); data.append(wq * (-wi) * float(dtrn_vfx[j][1]))
                for j in range(phi_vfy.size):
                    rows.append(ri); cols.append(int(gd_vfy[j])); data.append(wq * (-wi) * float(dtrn_vfy[j][1]))
                for j in range(phi_pf.size):
                    rows.append(ri); cols.append(int(gd_pf[j])); data.append(wq * (-wi) * float(dtrn_pf[j][1]))

            # poro tests (vP) and (uP) in test_jump_n couple similarly with + sign
            for i in range(phi_vpx.size):
                ri = int(gd_vpx[i]); wi = float(phi_vpx[i])
                for j in range(phi_vfx.size):
                    rows.append(ri); cols.append(int(gd_vfx[j])); data.append(wq * (wi) * float(dtrn_vfx[j][0]))
                for j in range(phi_vfy.size):
                    rows.append(ri); cols.append(int(gd_vfy[j])); data.append(wq * (wi) * float(dtrn_vfy[j][0]))
                for j in range(phi_pf.size):
                    rows.append(ri); cols.append(int(gd_pf[j])); data.append(wq * (wi) * float(dtrn_pf[j][0]))

            for i in range(phi_vpy.size):
                ri = int(gd_vpy[i]); wi = float(phi_vpy[i])
                for j in range(phi_vfx.size):
                    rows.append(ri); cols.append(int(gd_vfx[j])); data.append(wq * (wi) * float(dtrn_vfx[j][1]))
                for j in range(phi_vfy.size):
                    rows.append(ri); cols.append(int(gd_vfy[j])); data.append(wq * (wi) * float(dtrn_vfy[j][1]))
                for j in range(phi_pf.size):
                    rows.append(ri); cols.append(int(gd_pf[j])); data.append(wq * (wi) * float(dtrn_pf[j][1]))

            for i in range(phi_upx.size):
                ri = int(gd_upx[i]); wi = float(phi_upx[i])
                for j in range(phi_vfx.size):
                    rows.append(ri); cols.append(int(gd_vfx[j])); data.append(wq * (wi) * float(dtrn_vfx[j][0]))
                for j in range(phi_vfy.size):
                    rows.append(ri); cols.append(int(gd_vfy[j])); data.append(wq * (wi) * float(dtrn_vfy[j][0]))
                for j in range(phi_pf.size):
                    rows.append(ri); cols.append(int(gd_pf[j])); data.append(wq * (wi) * float(dtrn_pf[j][0]))

            for i in range(phi_upy.size):
                ri = int(gd_upy[i]); wi = float(phi_upy[i])
                for j in range(phi_vfx.size):
                    rows.append(ri); cols.append(int(gd_vfx[j])); data.append(wq * (wi) * float(dtrn_vfx[j][1]))
                for j in range(phi_vfy.size):
                    rows.append(ri); cols.append(int(gd_vfy[j])); data.append(wq * (wi) * float(dtrn_vfy[j][1]))
                for j in range(phi_pf.size):
                    rows.append(ri); cols.append(int(gd_pf[j])); data.append(wq * (wi) * float(dtrn_pf[j][1]))

            # dpF_test term: - inner(dpF_test*nF, dkin_n)  => -wp*(dkin·nF)
            for i in range(phi_pf.size):
                ri = int(gd_pf[i]); wpv = float(phi_pf[i])
                # dvF_x
                for j in range(phi_vfx.size):
                    cj = int(gd_vfx[j]); val = -wpv * float(dkin_vfx[j] @ nF)
                    if val:
                        rows.append(ri); cols.append(cj); data.append(wq * val)
                for j in range(phi_vfy.size):
                    cj = int(gd_vfy[j]); val = -wpv * float(dkin_vfy[j] @ nF)
                    if val:
                        rows.append(ri); cols.append(cj); data.append(wq * val)
                for j in range(phi_vpx.size):
                    cj = int(gd_vpx[j]); val = -wpv * float(dkin_vpx[j] @ nF)
                    if val:
                        rows.append(ri); cols.append(cj); data.append(wq * val)
                for j in range(phi_vpy.size):
                    cj = int(gd_vpy[j]); val = -wpv * float(dkin_vpy[j] @ nF)
                    if val:
                        rows.append(ri); cols.append(cj); data.append(wq * val)
                for j in range(phi_upx.size):
                    cj = int(gd_upx[j]); val = -wpv * float(dkin_upx[j] @ nF)
                    if val:
                        rows.append(ri); cols.append(cj); data.append(wq * val)
                for j in range(phi_upy.size):
                    cj = int(gd_upy[j]); val = -wpv * float(dkin_upy[j] @ nF)
                    if val:
                        rows.append(ri); cols.append(cj); data.append(wq * val)

            # -------------------------
            # Remaining Jacobian terms:
            # - normal adjoint/consistency term: -zeta*2mu <eps(test)·nF, dkin_n>
            # - normal penalty: (phi_gamma_F*gamma_n/h_gamma) <test_jump_n, dkin_n>
            # - tangential traction: +/- <test, dtraction_t>
            # - tangential t1/t2 terms: <..., dc_t>
            # -------------------------

            def _add_dkin_n_row(*, ri: int, wtest: float, comp: int, scale: float):
                # scale * wtest * dkin_n[comp]
                for j in range(phi_vfx.size):
                    rows.append(ri); cols.append(int(gd_vfx[j])); data.append(wq * scale * wtest * float(dkin_n_vfx[j][comp]))
                for j in range(phi_vfy.size):
                    rows.append(ri); cols.append(int(gd_vfy[j])); data.append(wq * scale * wtest * float(dkin_n_vfy[j][comp]))
                for j in range(phi_vpx.size):
                    rows.append(ri); cols.append(int(gd_vpx[j])); data.append(wq * scale * wtest * float(dkin_n_vpx[j][comp]))
                for j in range(phi_vpy.size):
                    rows.append(ri); cols.append(int(gd_vpy[j])); data.append(wq * scale * wtest * float(dkin_n_vpy[j][comp]))
                for j in range(phi_upx.size):
                    rows.append(ri); cols.append(int(gd_upx[j])); data.append(wq * scale * wtest * float(dkin_n_upx[j][comp]))
                for j in range(phi_upy.size):
                    rows.append(ri); cols.append(int(gd_upy[j])); data.append(wq * scale * wtest * float(dkin_n_upy[j][comp]))

            def _add_adjoint_row(*, ri: int, epsn_test: Array, scale: float):
                # scale * dot(epsn_test, dkin_n)
                ex, ey = float(epsn_test[0]), float(epsn_test[1])
                for j in range(phi_vfx.size):
                    dkn = dkin_n_vfx[j]
                    rows.append(ri); cols.append(int(gd_vfx[j])); data.append(wq * scale * (ex * float(dkn[0]) + ey * float(dkn[1])))
                for j in range(phi_vfy.size):
                    dkn = dkin_n_vfy[j]
                    rows.append(ri); cols.append(int(gd_vfy[j])); data.append(wq * scale * (ex * float(dkn[0]) + ey * float(dkn[1])))
                for j in range(phi_vpx.size):
                    dkn = dkin_n_vpx[j]
                    rows.append(ri); cols.append(int(gd_vpx[j])); data.append(wq * scale * (ex * float(dkn[0]) + ey * float(dkn[1])))
                for j in range(phi_vpy.size):
                    dkn = dkin_n_vpy[j]
                    rows.append(ri); cols.append(int(gd_vpy[j])); data.append(wq * scale * (ex * float(dkn[0]) + ey * float(dkn[1])))
                for j in range(phi_upx.size):
                    dkn = dkin_n_upx[j]
                    rows.append(ri); cols.append(int(gd_upx[j])); data.append(wq * scale * (ex * float(dkn[0]) + ey * float(dkn[1])))
                for j in range(phi_upy.size):
                    dkn = dkin_n_upy[j]
                    rows.append(ri); cols.append(int(gd_upy[j])); data.append(wq * scale * (ex * float(dkn[0]) + ey * float(dkn[1])))

            def _add_dtrt_row(*, ri: int, wtest: float, comp: int, scale: float):
                # scale * wtest * dtraction_t[comp]
                for j in range(phi_vfx.size):
                    rows.append(ri); cols.append(int(gd_vfx[j])); data.append(wq * scale * wtest * float(dtrt_vfx[j][comp]))
                for j in range(phi_vfy.size):
                    rows.append(ri); cols.append(int(gd_vfy[j])); data.append(wq * scale * wtest * float(dtrt_vfy[j][comp]))
                for j in range(phi_pf.size):
                    rows.append(ri); cols.append(int(gd_pf[j])); data.append(wq * scale * wtest * float(dtrt_pf[j][comp]))

            def _add_dc_t_row(*, ri: int, wtest: float, comp: int, scale: float):
                # scale * wtest * dc_t[comp] (dc_t already tangentially projected)
                for j in range(phi_vfx.size):
                    rows.append(ri); cols.append(int(gd_vfx[j])); data.append(wq * scale * wtest * float(dc_t_vfx[j][comp]))
                for j in range(phi_vfy.size):
                    rows.append(ri); cols.append(int(gd_vfy[j])); data.append(wq * scale * wtest * float(dc_t_vfy[j][comp]))
                for j in range(phi_pf.size):
                    rows.append(ri); cols.append(int(gd_pf[j])); data.append(wq * scale * wtest * float(dc_t_pf[j][comp]))
                for j in range(phi_vpx.size):
                    rows.append(ri); cols.append(int(gd_vpx[j])); data.append(wq * scale * wtest * float(dc_t_vpx[j][comp]))
                for j in range(phi_vpy.size):
                    rows.append(ri); cols.append(int(gd_vpy[j])); data.append(wq * scale * wtest * float(dc_t_vpy[j][comp]))
                for j in range(phi_upx.size):
                    rows.append(ri); cols.append(int(gd_upx[j])); data.append(wq * scale * wtest * float(dc_t_upx[j][comp]))
                for j in range(phi_upy.size):
                    rows.append(ri); cols.append(int(gd_upy[j])); data.append(wq * scale * wtest * float(dc_t_upy[j][comp]))

            def _add_t1_row(*, ri: int, epsn_test: Array, scale: float):
                # scale * dot((-2mu*epsn_test), dc_t)
                vx = float((-2.0 * mu_f) * epsn_test[0])
                vy = float((-2.0 * mu_f) * epsn_test[1])
                for j in range(phi_vfx.size):
                    dc = dc_t_vfx[j]
                    rows.append(ri); cols.append(int(gd_vfx[j])); data.append(wq * scale * (vx * float(dc[0]) + vy * float(dc[1])))
                for j in range(phi_vfy.size):
                    dc = dc_t_vfy[j]
                    rows.append(ri); cols.append(int(gd_vfy[j])); data.append(wq * scale * (vx * float(dc[0]) + vy * float(dc[1])))
                for j in range(phi_pf.size):
                    dc = dc_t_pf[j]
                    rows.append(ri); cols.append(int(gd_pf[j])); data.append(wq * scale * (vx * float(dc[0]) + vy * float(dc[1])))
                for j in range(phi_vpx.size):
                    dc = dc_t_vpx[j]
                    rows.append(ri); cols.append(int(gd_vpx[j])); data.append(wq * scale * (vx * float(dc[0]) + vy * float(dc[1])))
                for j in range(phi_vpy.size):
                    dc = dc_t_vpy[j]
                    rows.append(ri); cols.append(int(gd_vpy[j])); data.append(wq * scale * (vx * float(dc[0]) + vy * float(dc[1])))
                for j in range(phi_upx.size):
                    dc = dc_t_upx[j]
                    rows.append(ri); cols.append(int(gd_upx[j])); data.append(wq * scale * (vx * float(dc[0]) + vy * float(dc[1])))
                for j in range(phi_upy.size):
                    dc = dc_t_upy[j]
                    rows.append(ri); cols.append(int(gd_upy[j])); data.append(wq * scale * (vx * float(dc[0]) + vy * float(dc[1])))

            pen_scale = float(phi_gamma_F * gamma_n / h_gamma)
            adj_scale = float((-zeta) * (2.0 * mu_f))

            # fluid velocity test rows: adjoint + penalty + tangential traction + t1 + t2
            for i in range(phi_vfx.size):
                ri = int(gd_vfx[i])
                wtest = float(phi_vfx[i])
                epsn_test = _eps_dot_n_for_vx(g_vfx[i], nF)
                _add_adjoint_row(ri=ri, epsn_test=epsn_test, scale=adj_scale)
                _add_dkin_n_row(ri=ri, wtest=wtest, comp=0, scale=pen_scale)
                _add_dtrt_row(ri=ri, wtest=wtest, comp=0, scale=-1.0)
                _add_t1_row(ri=ri, epsn_test=epsn_test, scale=float(t1))
                _add_dc_t_row(ri=ri, wtest=wtest, comp=0, scale=float(t2))

            for i in range(phi_vfy.size):
                ri = int(gd_vfy[i])
                wtest = float(phi_vfy[i])
                epsn_test = _eps_dot_n_for_vy(g_vfy[i], nF)
                _add_adjoint_row(ri=ri, epsn_test=epsn_test, scale=adj_scale)
                _add_dkin_n_row(ri=ri, wtest=wtest, comp=1, scale=pen_scale)
                _add_dtrt_row(ri=ri, wtest=wtest, comp=1, scale=-1.0)
                _add_t1_row(ri=ri, epsn_test=epsn_test, scale=float(t1))
                _add_dc_t_row(ri=ri, wtest=wtest, comp=1, scale=float(t2))

            # poro velocity test rows: normal penalty only (traction_n already covered above)
            for i in range(phi_vpx.size):
                ri = int(gd_vpx[i]); wtest = float(phi_vpx[i])
                _add_dkin_n_row(ri=ri, wtest=wtest, comp=0, scale=-pen_scale)
            for i in range(phi_vpy.size):
                ri = int(gd_vpy[i]); wtest = float(phi_vpy[i])
                _add_dkin_n_row(ri=ri, wtest=wtest, comp=1, scale=-pen_scale)

            # poro displacement test rows: normal penalty + tangential traction + t2
            for i in range(phi_upx.size):
                ri = int(gd_upx[i]); wtest = float(phi_upx[i])
                _add_dkin_n_row(ri=ri, wtest=wtest, comp=0, scale=-pen_scale)
                _add_dtrt_row(ri=ri, wtest=wtest, comp=0, scale=+1.0)
                _add_dc_t_row(ri=ri, wtest=-wtest, comp=0, scale=float(t2))  # -u_test in (v_test-u_test)
            for i in range(phi_upy.size):
                ri = int(gd_upy[i]); wtest = float(phi_upy[i])
                _add_dkin_n_row(ri=ri, wtest=wtest, comp=1, scale=-pen_scale)
                _add_dtrt_row(ri=ri, wtest=wtest, comp=1, scale=+1.0)
                _add_dc_t_row(ri=ri, wtest=-wtest, comp=1, scale=float(t2))

    K = sp.coo_matrix((np.asarray(data, float), (np.asarray(rows, int), np.asarray(cols, int))), shape=(n_total, n_total))
    K.sum_duplicates()
    return K.tocsr(), R


def assemble_inlet_traction_rhs(
    *,
    inlet_P0: Array,
    inlet_P1: Array,
    inlet_pos_elem_ids: Array,
    dh_f: DofHandler,
    sigmaF_A: Callable[[Array], Array],
    quad_order: int = 6,
    x0: float = -0.45,
) -> Array:
    """Assemble the Neumann traction RHS on the inlet cut x=x0.

    This matches `examples/FPI/fpi_mms_example41.py:998` but integrates over the
    CutFEM inlet segments explicitly (no dInterface measure).
    """
    nF = np.array([-1.0, 0.0], dtype=float)  # outward from Ω^F at x=x0
    if inlet_P0.shape[0] == 0:
        return np.zeros(int(dh_f.total_dofs), dtype=float)

    mesh_f = dh_f.mixed_element.mesh
    me_f = dh_f.mixed_element
    vfx, vfy = "v_pos_x", "v_pos_y"
    sl_vfx = me_f.component_dof_slices[vfx]
    sl_vfy = me_f.component_dof_slices[vfy]

    xi, w_ref = gauss_legendre(int(quad_order))
    xi = np.asarray(xi, dtype=float)
    w_ref = np.asarray(w_ref, dtype=float)

    R = np.zeros(int(dh_f.total_dofs), dtype=float)
    for s in range(int(inlet_P0.shape[0])):
        ef = int(inlet_pos_elem_ids[s])
        P0 = np.asarray(inlet_P0[s], float)
        P1 = np.asarray(inlet_P1[s], float)
        mid = 0.5 * (P0 + P1)
        half = 0.5 * (P1 - P0)
        segJ = float(np.linalg.norm(half))
        if segJ <= 0.0:
            continue

        gd_vfx = np.asarray(dh_f.element_maps[vfx][ef], dtype=int)
        gd_vfy = np.asarray(dh_f.element_maps[vfy][ef], dtype=int)

        for q, wq_ref in enumerate(w_ref):
            xq = mid + float(xi[q]) * half
            wq = float(wq_ref) * segJ
            if wq == 0.0:
                continue
            xi_f, eta_f = transform.inverse_mapping(mesh_f, ef, xq)
            phi_vfx = np.asarray(me_f.basis(vfx, float(xi_f), float(eta_f)), float).ravel()[sl_vfx]
            phi_vfy = np.asarray(me_f.basis(vfy, float(xi_f), float(eta_f)), float).ravel()[sl_vfy]

            sig = np.asarray(sigmaF_A(np.asarray(xq, float).reshape(1, 2)), float).reshape(2, 2)
            traction = sig @ nF

            for i in range(phi_vfx.size):
                R[int(gd_vfx[i])] += -wq * float(phi_vfx[i]) * float(traction[0])
            for i in range(phi_vfy.size):
                R[int(gd_vfy[i])] += -wq * float(phi_vfy[i]) * float(traction[1])

    return R
