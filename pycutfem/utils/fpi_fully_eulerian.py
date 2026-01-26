"""Fully Eulerian Fluid–Poroelastic Interaction (FPI) weak forms.

This module composes the already-verified building blocks:
  - fluid volume Navier–Stokes (Eulerian) on Ω⁺,
  - poroelastic volume sub-problem on Ω⁻ (see `fpi_poro_eulerian.py`),
  - Nitsche interface coupling terms on Γ (see `fpi_interface_eulerian.py`),
and adds lightweight CutFEM facet stabilizations (optional) for equal-order
pressure/velocity discretizations.

The goal is a *clean, debuggable* entry-point for the full FPI MMS (Example 4.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    FacetNormal,
    avg,
    div,
    dot,
    grad,
    inner,
    jump,
)
from pycutfem.ufl.measures import dCutSkeleton

from pycutfem.utils.fpi_interface_eulerian import FPIInterfaceForms, build_fpi_interface_forms
from pycutfem.utils.fpi_poro_eulerian import jacobian_poro, residual_poro


def _epsilon(v):
    return Constant(0.5) * (grad(v) + grad(v).T)


def _grad_inner_jump(u, v, n):
    """Penalty on the jump of the *normal derivative* across an interior facet."""
    return inner(dot(jump(grad(u)), n), dot(jump(grad(v)), n))


def _const_scalar(val: float) -> Constant:
    return Constant(float(val), dim=0)


def _effective_k_inv_scalar(K_inv) -> float:
    """Conservative scalar proxy for `K_inv` used in stabilization scalings."""
    try:
        mat = np.asarray(getattr(K_inv, "value", None), dtype=float)
    except Exception:
        return 1.0
    if mat.shape != (2, 2):
        return 1.0
    try:
        smax = float(np.max(np.linalg.svd(mat, compute_uv=False)))
    except Exception:
        smax = float(np.linalg.norm(mat))
    return smax if smax > 0.0 else 1.0


@dataclass(frozen=True)
class FPIEulerianForms:
    jacobian_form: object
    residual_form: object
    a_fluid: object
    r_fluid: object
    a_poro: object
    r_poro: object
    interface: FPIInterfaceForms | None
    a_stab: object
    r_stab: object


def build_fpi_eulerian_forms(
    *,
    # unknowns at t_{n+1}
    vF_k,
    pF_k,
    vP_k,
    uP_k,
    pP_k,
    # unknowns at t_n
    vF_n,
    pF_n,
    vP_n,
    uP_n,
    pP_n,
    # optional unknowns at t_{n-1} (for poro-skeleton acceleration)
    uP_nm1=None,
    # Newton increments
    dvF,
    dpF,
    dvP,
    duP,
    dpP,
    # test functions
    vF_test,
    qF_test,
    vP_test,
    uP_test,
    qP_test,
    # CutFEM measures
    dx_f,
    dx_p,
    dGamma,
    dG_f,
    dG_p,
    level_set,
    # optional: separate level sets for facet stabilizations
    level_set_f=None,
    level_set_p=None,
    # time integration
    dt,
    theta: float = 1.0,
    # physical parameters
    rho_f,
    mu_f,
    rho_s0_tilde,
    porosity,
    K_inv,
    c_nh,
    beta_nh,
    # interface parameters
    beta_BJ,
    kappa,
    gamma_n,
    gamma_t,
    zeta: float = 1.0,
    # stabilization parameters (set to 0 to disable individual terms)
    gamma_F_p: float = 0.0,
    gamma_F_gp: float = 0.0,
    gamma_P_p: float = 0.0,
    gamma_P_gp: float = 0.0,
    # manufactured interface data (use zeros for the physical case)
    g_sigma=None,
    g_sigma_n=None,
    g_n=None,
    g_t=None,
    # optional override for the penalty scaling φ^F_Γ (paper eq. (24))
    use_paper_phi_gamma: bool = False,
    vF_inf=None,
    c_v_gamma: float = 1.0 / 6.0,
    c_t_gamma: float = 1.0 / 12.0,
    # optional override for the interface mesh-size parameter h_Γ (paper eq. (24))
    h_gamma=None,
    # feature toggles
    use_interface_terms: bool = True,
    use_stabilization: bool = True,
    # optional: paper-consistent CIP/ghost penalty (eqs. (21)-(23))
    use_paper_stabilization: bool = False,
    poly_order: int = 1,
    gamma_u: float = 0.05,
    gamma_p: float = 0.05,
    gamma_div_factor: float = 1.0e-3,
    gamma_gp_nu: float = 0.1,
    gamma_gp_t: float = 0.001,
    gp_second_weight: float = 0.05,
) -> FPIEulerianForms:
    """
    Build the coupled FPI residual/Jacobian forms on a fixed (Eulerian) mesh.

    Notes
    -----
    - Fluid is on the positive side Ω⁺, porous domain on the negative side Ω⁻.
    - `dt` may be a float or a `Constant`; `theta` is the one-step-θ factor.
    - Stabilization is intentionally minimal (pressure CIP + pressure ghost penalty)
      to keep the MMS driver fast and the terms easy to debug.
    """
    cell_h = CellDiameter()
    h_gamma_expr = h_gamma if h_gamma is not None else cell_h
    th = Constant(float(theta))
    zeta_c = Constant(float(zeta))

    # ------------------------------------------------------------------
    # Fluid volume (Navier–Stokes on Ω⁺)
    # ------------------------------------------------------------------
    vdot = (vF_k - vF_n) / dt

    r_fluid = inner(rho_f * vdot, vF_test) * dx_f
    r_fluid += th * rho_f * dot(dot(grad(vF_k), vF_k), vF_test) * dx_f
    r_fluid += (Constant(1.0) - th) * rho_f * dot(dot(grad(vF_n), vF_n), vF_test) * dx_f
    r_fluid += Constant(2.0) * th * mu_f * inner(_epsilon(vF_k), _epsilon(vF_test)) * dx_f
    r_fluid += Constant(2.0) * (Constant(1.0) - th) * mu_f * inner(_epsilon(vF_n), _epsilon(vF_test)) * dx_f
    r_fluid += -pF_k * div(vF_test) * dx_f
    r_fluid += qF_test * div(vF_k) * dx_f

    a_fluid = rho_f / dt * dot(dvF, vF_test) * dx_f
    a_fluid += th * rho_f * dot(dot(grad(vF_k), dvF), vF_test) * dx_f
    a_fluid += th * rho_f * dot(dot(grad(dvF), vF_k), vF_test) * dx_f
    a_fluid += Constant(2.0) * th * mu_f * inner(_epsilon(dvF), _epsilon(vF_test)) * dx_f
    a_fluid += -dpF * div(vF_test) * dx_f
    a_fluid += qF_test * div(dvF) * dx_f

    # ------------------------------------------------------------------
    # Porous volume (Ω⁻) from the validated module
    # ------------------------------------------------------------------
    r_poro = residual_poro(
        vP_k,
        uP_k,
        pP_k,
        vP_n,
        uP_n,
        pP_n,
        qP_test,
        vP_test,
        uP_test,
        u_nm1=uP_nm1,
        rho_f=rho_f,
        mu_f=mu_f,
        rho_s0_tilde=rho_s0_tilde,
        phi=porosity,
        K_inv=K_inv,
        c_nh=c_nh,
        beta_nh=beta_nh,
        dt=dt,
        theta=th,
        dx_p=dx_p,
    )
    a_poro = jacobian_poro(
        vP_k,
        uP_k,
        pP_k,
        uP_n,
        dvP,
        duP,
        dpP,
        qP_test,
        vP_test,
        uP_test,
        u_nm1=uP_nm1,
        rho_f=rho_f,
        mu_f=mu_f,
        rho_s0_tilde=rho_s0_tilde,
        phi=porosity,
        K_inv=K_inv,
        c_nh=c_nh,
        beta_nh=beta_nh,
        dt=dt,
        theta=th,
        dx_p=dx_p,
    )

    # ------------------------------------------------------------------
    # Interface coupling (Γ): Nitsche normal + BJ tangential terms
    # ------------------------------------------------------------------
    interface: FPIInterfaceForms | None = None
    a_ifc = Constant(0.0) * dpF * qF_test * dGamma
    r_ifc = Constant(0.0) * qF_test * dGamma
    if use_interface_terms:
        if g_sigma is None:
            g_sigma = Constant((0.0, 0.0), dim=1)
        if g_sigma_n is None:
            g_sigma_n = Constant(0.0, dim=0)
        if g_n is None:
            g_n = Constant((0.0, 0.0), dim=1)
        if g_t is None:
            g_t = Constant((0.0, 0.0), dim=1)

        # Penalty scaling φ^F_Γ (paper eq. (24)).
        if use_paper_phi_gamma:
            if vF_inf is None:
                vF_inf_expr = Constant(0.0)
            elif isinstance(vF_inf, (int, float)):
                vF_inf_expr = Constant(float(vF_inf))
            else:
                vF_inf_expr = vF_inf
            phi_gamma_F = (
                mu_f
                + h_gamma_expr * Constant(float(c_v_gamma)) * rho_f * vF_inf_expr
                + (h_gamma_expr**2) * Constant(float(c_t_gamma)) * (rho_f / (th * dt))
            )
        else:
            # Lightweight default (kept for backwards compatibility with earlier MMS tests).
            phi_gamma_F = mu_f + (h_gamma_expr**2) * (rho_f / dt)

        interface = build_fpi_interface_forms(
            vF_k=vF_k,
            pF_k=pF_k,
            vP_k=vP_k,
            uP_k=uP_k,
            uP_n=uP_n,
            dvF=dvF,
            dpF=dpF,
            dvP=dvP,
            duP=duP,
            dvF_test=vF_test,
            dpF_test=qF_test,
            dvP_test=vP_test,
            duP_test=uP_test,
            dGamma=dGamma,
            mu_f=mu_f,
            porosity=porosity,
            beta_BJ=beta_BJ,
            kappa=kappa,
            gamma_n=gamma_n,
            gamma_t=gamma_t,
            phi_gamma_F=phi_gamma_F,
            h_gamma=h_gamma_expr,
            zeta=zeta_c,
            dt=dt,
            g_sigma=g_sigma,
            g_sigma_n=g_sigma_n,
            g_n=g_n,
            g_t=g_t,
        )
        a_ifc = interface.jacobian
        r_ifc = interface.residual

    # ------------------------------------------------------------------
    # Facet stabilizations (optional)
    # ------------------------------------------------------------------
    a_stab = Constant(0.0) * dpF * qF_test * dx_f
    r_stab = Constant(0.0) * qF_test * dx_f
    if use_stabilization:
        n = FacetNormal()

        # Use the same quadrature as the volume measures unless overridden.
        qdeg = int((dx_f.metadata or {}).get("q", 6))
        ls_f = level_set_f if level_set_f is not None else level_set
        ls_p = level_set_p if level_set_p is not None else level_set
        derivs = {(1, 0), (0, 1)}
        dS_f = dCutSkeleton(level_set=ls_f, metadata={"q": qdeg, "side": "+", "derivs": derivs})
        dS_p = dCutSkeleton(level_set=ls_p, metadata={"q": qdeg, "side": "-", "derivs": derivs})
        dG_f_stab = dG_f(metadata={"derivs": derivs})
        dG_p_stab = dG_p(metadata={"derivs": derivs})

        if use_paper_stabilization:
            # CIP parameters (paper eqs. (21)-(22)) and ghost penalty (paper eq. (23)).
            h_F = avg(cell_h)

            # For MMS studies we approximate ‖v‖_{∞,F} by a global supplied bound.
            if vF_inf is None:
                v_inf_expr = _const_scalar(0.0)
            elif isinstance(vF_inf, (int, float)):
                v_inf_expr = _const_scalar(float(vF_inf))
            else:
                v_inf_expr = vF_inf

            c_v = Constant(float(c_v_gamma))
            c_t = Constant(float(c_t_gamma))
            Phi_F = mu_f + h_F * c_v * rho_f * v_inf_expr + (h_F**2) * c_t * (rho_f / (th * dt))

            gamma_div = float(gamma_div_factor) * float(gamma_p)
            tau_F_u = Constant(float(gamma_u)) * ((rho_f * v_inf_expr) ** 2) * (h_F**3) / Phi_F
            tau_F_p = Constant(float(gamma_p)) * (h_F**3) / Phi_F
            tau_F_div = Constant(float(gamma_div)) * h_F * Phi_F

            # Poro scaling uses a reactive term based on φ° and K° (paper eq. (22)).
            K_inv_eff = _effective_k_inv_scalar(K_inv)
            Phi_P = (h_F**2) * (
                Constant(1.0) * mu_f * porosity * Constant(float(K_inv_eff))
                + c_t * (rho_f / (th * dt))
            )
            tau_P_p = Constant(float(gamma_p)) * (h_F**3) / Phi_P
            tau_P_div = Constant(float(gamma_div)) * h_F * Phi_P

            # CIP on interior faces.
            a_stab += tau_F_u * _grad_inner_jump(dvF, vF_test, n) * dS_f
            r_stab += tau_F_u * _grad_inner_jump(vF_k, vF_test, n) * dS_f

            a_stab += tau_F_p * _grad_inner_jump(dpF, qF_test, n) * dS_f
            r_stab += tau_F_p * _grad_inner_jump(pF_k, qF_test, n) * dS_f

            a_stab += tau_F_div * jump(div(dvF)) * jump(div(vF_test)) * dS_f
            r_stab += tau_F_div * jump(div(vF_k)) * jump(div(vF_test)) * dS_f

            a_stab += tau_P_p * _grad_inner_jump(dpP, qP_test, n) * dS_p
            r_stab += tau_P_p * _grad_inner_jump(pP_k, qP_test, n) * dS_p

            a_stab += tau_P_div * jump(div(dvP)) * jump(div(vP_test)) * dS_p
            r_stab += tau_P_div * jump(div(vP_k)) * jump(div(vP_test)) * dS_p

            # Ghost penalty on near-interface faces (for Q1: only j=1 is non-zero).
            tau_GP_p1 = tau_F_p
            tau_GP_div1 = tau_F_div
            tau_GP_u1 = tau_F_u + Constant(float(gamma_gp_nu)) * h_F * mu_f + Constant(float(gamma_gp_t)) * (h_F**3) * (rho_f / (th * dt))

            a_stab += tau_GP_p1 * _grad_inner_jump(dpF, qF_test, n) * dG_f_stab
            r_stab += tau_GP_p1 * _grad_inner_jump(pF_k, qF_test, n) * dG_f_stab

            a_stab += tau_GP_div1 * jump(div(dvF)) * jump(div(vF_test)) * dG_f_stab
            r_stab += tau_GP_div1 * jump(div(vF_k)) * jump(div(vF_test)) * dG_f_stab

            a_stab += tau_GP_u1 * _grad_inner_jump(dvF, vF_test, n) * dG_f_stab
            r_stab += tau_GP_u1 * _grad_inner_jump(vF_k, vF_test, n) * dG_f_stab

            # Poro-side ghost penalties: our fully Eulerian/CutFEM setup also cuts Ω⁻,
            # unlike the reference implementation in the paper. Penalize jumps on
            # the poro ghost facets to control cut-cell ill-conditioning, which is
            # especially important for the BJ variant (β_BJ=1) where v^P enters the
            # tangential kinematic constraint.
            tau_GP_P_p1 = tau_P_p
            tau_GP_P_div1 = tau_P_div
            mu_react_P = mu_f * porosity * Constant(float(K_inv_eff))
            tau_GP_P_v1 = Constant(float(gamma_gp_nu)) * h_F * mu_react_P + Constant(float(gamma_gp_t)) * (h_F**3) * (
                rho_f / (th * dt)
            )

            a_stab += tau_GP_P_p1 * _grad_inner_jump(dpP, qP_test, n) * dG_p_stab
            r_stab += tau_GP_P_p1 * _grad_inner_jump(pP_k, qP_test, n) * dG_p_stab

            a_stab += tau_GP_P_div1 * jump(div(dvP)) * jump(div(vP_test)) * dG_p_stab
            r_stab += tau_GP_P_div1 * jump(div(vP_k)) * jump(div(vP_test)) * dG_p_stab

            a_stab += tau_GP_P_v1 * _grad_inner_jump(dvP, vP_test, n) * dG_p_stab
            r_stab += tau_GP_P_v1 * _grad_inner_jump(vP_k, vP_test, n) * dG_p_stab

            # Skeleton displacement ghost penalty (elastic scaling).
            # Use the same γ^{GP}_ν coefficient as for the fluid velocity GP to
            # avoid over-penalizing the poro skeleton (the paper uses a fitted
            # mesh on Ω^P so this term is not present there).
            tau_GP_uP1 = Constant(float(gamma_gp_nu)) * Constant(2.0) * c_nh * h_F
            a_stab += tau_GP_uP1 * _grad_inner_jump(duP, uP_test, n) * dG_p_stab
            r_stab += tau_GP_uP1 * _grad_inner_jump(uP_k, uP_test, n) * dG_p_stab

            # Optional j=2 terms for bi-quadratic discretizations.
            if int(poly_order) >= 2 and float(gp_second_weight) != 0.0:
                w2 = Constant(float(gp_second_weight)) * (h_F**2)
                tau_GP_p2 = w2 * tau_GP_p1
                tau_GP_u2 = w2 * (tau_GP_u1 + tau_GP_div1)

                def _dn(u):
                    return dot(grad(u), n)

                def _dn2(u):
                    return dot(grad(_dn(u)), n)

                a_stab += tau_GP_p2 * inner(jump(_dn2(dpF)), jump(_dn2(qF_test))) * dG_f_stab
                r_stab += tau_GP_p2 * inner(jump(_dn2(pF_k)), jump(_dn2(qF_test))) * dG_f_stab

                a_stab += tau_GP_u2 * inner(jump(_dn2(dvF)), jump(_dn2(vF_test))) * dG_f_stab
                r_stab += tau_GP_u2 * inner(jump(_dn2(vF_k)), jump(_dn2(vF_test))) * dG_f_stab

        else:
            # Legacy lightweight stabilization knobs (kept for earlier MMS/debug scripts).
            if float(gamma_F_p) != 0.0:
                a_stab += Constant(float(gamma_F_p)) * (cell_h**3) * _grad_inner_jump(dpF, qF_test, n) * dS_f
                r_stab += Constant(float(gamma_F_p)) * (cell_h**3) * _grad_inner_jump(pF_k, qF_test, n) * dS_f

            if float(gamma_F_gp) != 0.0:
                a_stab += Constant(float(gamma_F_gp)) * (cell_h**3) * _grad_inner_jump(dpF, qF_test, n) * dG_f_stab
                r_stab += Constant(float(gamma_F_gp)) * (cell_h**3) * _grad_inner_jump(pF_k, qF_test, n) * dG_f_stab

            if float(gamma_P_p) != 0.0:
                a_stab += Constant(float(gamma_P_p)) * (cell_h**3) * _grad_inner_jump(dpP, qP_test, n) * dS_p
                r_stab += Constant(float(gamma_P_p)) * (cell_h**3) * _grad_inner_jump(pP_k, qP_test, n) * dS_p

            if float(gamma_P_gp) != 0.0:
                a_stab += Constant(float(gamma_P_gp)) * (cell_h**3) * _grad_inner_jump(dpP, qP_test, n) * dG_p_stab
                r_stab += Constant(float(gamma_P_gp)) * (cell_h**3) * _grad_inner_jump(pP_k, qP_test, n) * dG_p_stab

    jacobian_form = a_fluid + a_poro + a_ifc + a_stab
    residual_form = r_fluid + r_poro + r_ifc + r_stab

    return FPIEulerianForms(
        jacobian_form=jacobian_form,
        residual_form=residual_form,
        a_fluid=a_fluid,
        r_fluid=r_fluid,
        a_poro=a_poro,
        r_poro=r_poro,
        interface=interface,
        a_stab=a_stab,
        r_stab=r_stab,
    )
