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

from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    FacetNormal,
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
    # feature toggles
    use_interface_terms: bool = True,
    use_stabilization: bool = True,
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

        # Penalty scaling (paper's φ^F_Γ): keep a simple robust variant here.
        # The full ||v||_{∞,Γ} scaling is implemented at the driver level by
        # allowing callers to pass an ElementWiseConstant if desired.
        phi_gamma_F = mu_f + (cell_h**2) * (rho_f / dt)

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
            h_gamma=cell_h,
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
        dS_f = dCutSkeleton(level_set=level_set, metadata={"q": qdeg, "side": "+"})
        dS_p = dCutSkeleton(level_set=level_set, metadata={"q": qdeg, "side": "-"})

        if float(gamma_F_p) != 0.0:
            a_stab += Constant(float(gamma_F_p)) * (cell_h**3) * _grad_inner_jump(dpF, qF_test, n) * dS_f
            r_stab += Constant(float(gamma_F_p)) * (cell_h**3) * _grad_inner_jump(pF_k, qF_test, n) * dS_f

        if float(gamma_F_gp) != 0.0:
            a_stab += Constant(float(gamma_F_gp)) * (cell_h**3) * _grad_inner_jump(dpF, qF_test, n) * dG_f
            r_stab += Constant(float(gamma_F_gp)) * (cell_h**3) * _grad_inner_jump(pF_k, qF_test, n) * dG_f

        if float(gamma_P_p) != 0.0:
            a_stab += Constant(float(gamma_P_p)) * (cell_h**3) * _grad_inner_jump(dpP, qP_test, n) * dS_p
            r_stab += Constant(float(gamma_P_p)) * (cell_h**3) * _grad_inner_jump(pP_k, qP_test, n) * dS_p

        if float(gamma_P_gp) != 0.0:
            a_stab += Constant(float(gamma_P_gp)) * (cell_h**3) * _grad_inner_jump(dpP, qP_test, n) * dG_p
            r_stab += Constant(float(gamma_P_gp)) * (cell_h**3) * _grad_inner_jump(pP_k, qP_test, n) * dG_p

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
