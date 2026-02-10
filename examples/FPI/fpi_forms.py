"""Fully Eulerian CutFEM formulation for Fluid–Poroelastic Interaction (FPI).

This module mirrors the API/style used in the existing fully Eulerian FSI utilities.
It provides:
  - Residual and Jacobian contributions for the semi-/fully-discrete weak form
    (paper eqs. (19)–(24), (27)–(28) combined in (30)).
  - A convenience ``build_fpi_eulerian_forms`` that assembles the complete
    residual and Jacobian UFL forms.

Notes
-----
* The implementation is written in the same "manual Jacobian" style as the FSI
  reference: we explicitly code the Gateaux derivatives w.r.t. the increment
  variables.
* Stabilization parameters (CIP/ghost) and interface scaling (phi_F_Gamma) are
  treated as *lagged* (computed from previous time step by default) to avoid
  extra nonlinearities in the Jacobian, consistent with the FSI utilities.
* The poroelastic part is implemented as in the TeX weak form you provided.
  In particular, the reference-domain terms are integrated on the porous side
  ("neg" domain) and use F=I+grad(u^P), J=det(F), etc.
  If you later decide to switch to an Eulerian solid description (as done for
  the solid in the FSI file), you can replace the kinematics/stress helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from pycutfem.ufl.expressions import (
    Constant,
    Identity,
    FacetNormal,
    CellDiameter,
    grad,
    div,
    dot,
    inner,
    outer,
    inv,
    det,
    trace,
    jump,
    sqrt,
    restrict,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    cof,
    split,  # Added
)
# Assuming IncrementFunction exists in pycutfem.ufl.spaces or similar
# If not available, usually defined as Function(V)
try:
    from pycutfem.ufl.expressions import IncrementFunction
except ImportError:
    from pycutfem.ufl.functions import Function as IncrementFunction

from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.measures import dx, dInterface, dFacetPatch

geom_dim = 2  # Default geometry dimension (2D)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def epsilon(v):
    """Symmetric gradient."""
    return 0.5 * (grad(v) + grad(v).T)


def sigma_fluid(v, p, mu):
    """Newtonian Cauchy stress for the fluid."""
    Id = Identity(geom_dim)
    return -p * Id + 2.0 * mu * epsilon(v)


def proj_n(n):
    """Normal projector (dyad n ⊗ n)."""
    return outer(n, n)


def proj_t(n):
    """Tangential projector I - n ⊗ n."""
    Id = Identity(geom_dim)
    return Id - outer(n, n)


# --- Poroelastic kinematics / constitutive (compressible Neo-Hookean) --------
class KinematicsHelpers:
    """
    Kinematic helpers for continuum mechanics (ALE or Lagrangian).
    Handles deformation gradients, Jacobians, and their linearizations (Gateaux derivatives).
     
    Variables:
    d  : Displacement field
    dd : Variation/Test function of displacement (delta d)
    F  : Deformation gradient (I + grad(d))
    J  : Determinant of F
    """
     
    # --- Basic Kinematics ---
    # --- Deformation Gradient Definitions ---
    
    @staticmethod
    def get_F_Lagrangian(grad_u):
        """
        Deformation Gradient F for Lagrangian/ALE formulations.
        Map: X -> x(X) = X + u(X)
        F = I + grad_X(u)
        """
        return Identity(geom_dim) + grad_u

    @staticmethod
    def get_F_Eulerian(grad_u):
        """
        Deformation Gradient F for Fully Eulerian (Fixed Grid) formulations.
        Map: x -> X(x) = x - u(x)
        F^-1 = I - grad_x(u)  =>  F = inv(I - grad_x(u))
        """
        Id = Identity(geom_dim)
        return inv(Id - grad_u)
    
    # --- Gradient Operators ---

    @staticmethod
    def grad_spatial(u):
        """
        Spatial gradient (w.r.t current configuration x).
        For Eulerian mesh, this is just grad(u).
        """
        return grad(u)

    @staticmethod
    def grad_material(u, F):
        """
        Material gradient (w.r.t reference configuration X), pushed to spatial frame.
        grad_X(u) = grad_x(u) * F
        """
        return dot(grad(u), F)

    @staticmethod
    def div_spatial(u):
        """Spatial divergence (trace of spatial gradient)."""
        return div(u)

    @staticmethod
    def div_material(u, F):
        """Material divergence (trace of material gradient)."""
        return trace(KinematicsHelpers.grad_material(u, F))
    
    def get_F_Eulerian_Lin(F, grad_delta_u):
        """
        Linearization of Eulerian F = inv(I - grad(u)).
        delta_F = F * grad(delta_u) * F
        """
        return dot(F, dot(grad_delta_u, F))

    @staticmethod
    def get_J_Lin(J, F_inv, grad_delta_u):
        """
        Linearization of J = det(F).
        delta_J = J * tr( F^-1 * delta_F ) 
                = J * tr( F^-1 * F * grad(delta_u) * F )
                = J * tr( grad(delta_u) * F )
        """
        # Note: In pure Eulerian velocity formulations, dJ/dt = J div(v). 
        # Here we use the displacement linearization.
        return J * inner(F_inv.T, KinematicsHelpers.get_F_Eulerian_Lin(inv(F_inv), grad_delta_u))

    @staticmethod
    def get_J(F):
        return det(F)

    @staticmethod
    def get_F_inv(F):
        return inv(F)

    @staticmethod
    def get_C_inv(F_inv):
        return dot(F_inv, F_inv.T)

    @staticmethod
    def get_cof_F(F):
        return cof(F)  # J * F^{-T}

    # --- Linearizations (Gateaux Derivatives) ---
    # These functions compute the directional derivative along 'grad_dd' (∇δd)

    @staticmethod
    def get_F_LinU(dd):
        r"""
        Linearization of F.
        δF = ∇δd
        """
        grad_dd = grad(dd)
        return grad_dd

    @staticmethod
    def get_J_LinU(F, grad_dd):
        r"""
        Linearization of J.
        δJ = cof(F) : ∇δd
        """
        return inner(cof(F), grad_dd)

    @staticmethod
    def get_F_inv_LinU(F_inv, grad_dd):
        r"""
        Linearization of F^{-1}.
        δ(F^{-1}) = - F^{-1} (∇δd) F^{-1}
        """
        return -dot(F_inv, dot(grad_dd, F_inv))

    @staticmethod
    def get_F_inv_T_LinU(F_inv, grad_dd):
        r"""
        Linearization of F^{-1}^T.
        δ(F^{-T}) = - F^{-T} (∇δd).T F^{-1}^T
        """
        return -dot(F_inv.T, dot(grad_dd.T, F_inv.T))

    @staticmethod
    def get_C_inv_LinU(F_inv, dF_inv):
        r"""
        Linearization of C^{-1} = F^{-1} F^{-T}.
        δ(C^{-1}) = δ(F^{-1}) F^{-T} + F^{-1} δ(F^{-T})
         
        Note: dF_inv should be computed via get_F_inv_LinU first.
        """
        dF_inv_T = dF_inv.T
        return dot(dF_inv, F_inv.T) + dot(F_inv, dF_inv_T)

    @staticmethod
    def get_a_LinU(a, beta, J, dJ):
        r"""
        Linearization of scaling factor a = J^{-2β}.
        δa = -2β * a * (δJ / J)
        """
        return -2 * beta * a * (dJ / J)

    # --- Stress Linearizations ---

    @staticmethod
    def get_dS_el(c, a, da, C_inv, dC_inv):
        r"""
        Linearization of 2nd Piola-Kirchhoff stress part S_el.
        S_el = -2 * c * a * C^{-1}
         
        δS_el = -2c [ (δa) C^{-1} + a (δC^{-1}) ]
        """
        return -2 * c * (da * C_inv + a * dC_inv)

    @staticmethod
    def get_dP_el(F, grad_dd, S_el, dS_el):
        r"""
        Linearization of 1st Piola-Kirchhoff stress P_el = F * S_el.
         
        δP_el = (δF) S_el + F (δS_el)
        Note: grad_dd acts as δF here.
        """
        return dot(grad_dd, S_el) + dot(F, dS_el)


class PoroNeoHookeanParams:
    """Material parameters for compressible Neo-Hookean model."""

    def __init__(self, E: float, nu: float, geom_dim: int = geom_dim):
        self.E = E
        # poisson's ratio
        self.nu = nu
        self.geom_dim = geom_dim
        self.c = self.calc_c()
        self.beta = self.calc_beta()

    def calc_beta(self) -> float:
        """Calculate beta parameter from E, nu."""
        return self.nu / (1.0 - self.nu * (self.geom_dim - 1))

    def calc_c(self) -> float:
        """Calculate c parameter from E, nu."""
        return self.E / (4.0 * (1.0 + self.nu))


class PoroElasticity:
    """
    Constitutive models for Poroelasticity (Neo-Hookean) and Permeability evolution.
    Uses KinematicsHelpers for basic kinematic variables and their linearizations.
    """
    geom_dim: int = geom_dim  # Default geometry dimension (2D)
     
    @staticmethod
    def get_S(u, c, beta):
        """
        Second Piola–Kirchhoff stress (Compressible Neo-Hookean).
        S = 2c [ I - J^{-2β} C^{-1} ],  with C^{-1} = F^{-1} F^{-T}.
        For 2d and 3d the stress would remain same but only the beta parameter changes.
        """
        grad_u = grad(u)
         
        # Kinematics
        F = KinematicsHelpers.get_F(grad_u)
        J = KinematicsHelpers.get_J(F)
        F_inv = KinematicsHelpers.get_F_inv(F)
        C_inv = KinematicsHelpers.get_C_inv(F_inv)
        Id = Identity(PoroElasticity.geom_dim)
         
        # Material Parameter
        a = J ** (-2.0 * beta)
         
        return 2.0 * c * (Id - a * C_inv)

    @staticmethod
    def get_P(u, c, beta):
        """
        First Piola–Kirchhoff stress P = F · S
        """
        grad_u = grad(u)
        F = KinematicsHelpers.get_F(grad_u)
        S = PoroElasticity.get_S(u, c, beta)
        return dot(F, S)

    @staticmethod
    def get_dP(u, du, c, beta):
        """
        Linearization of P (Gateaux derivative) for the Newton Jacobian.
        δP = (δF) S + F (δS)
        """
        grad_u = grad(u)
        dF = grad(du)  # This is δF
         
        # 1. Kinematics
        F = KinematicsHelpers.get_F(grad_u)
        J = KinematicsHelpers.get_J(F)
        F_inv = KinematicsHelpers.get_F_inv(F)
        C_inv = KinematicsHelpers.get_C_inv(F_inv)
        Id = Identity(PoroElasticity.geom_dim)
         
        # 2. Linearizations of Kinematics
        dJ = KinematicsHelpers.get_J_LinU(F, dF)
        dF_inv = KinematicsHelpers.get_F_inv_LinU(F_inv, dF)
        dC_inv = KinematicsHelpers.get_C_inv_LinU(F_inv, dF_inv)
         
        # 3. Material Parameters & Linearization
        a = J ** (-2.0 * beta)
        da = KinematicsHelpers.get_a_LinU(a, beta, J, dJ)
         
        # 4. Stress S and Linearization dS
        # S = 2c (I - a C^{-1})
        S = 2.0 * c * (Id - a * C_inv)
         
        # δS = -2c ( δa C^{-1} + a δC^{-1} )
        dS = -2.0 * c * (da * C_inv + a * dC_inv)
         
        # 5. Final Calculation: δP = δF S + F δS
        # Note: grad_du is δF
        return dot(dF, S) + dot(F, dS)

    @staticmethod
    def get_k(u, K):
        """
        Permeability push-forward: k = J^{-1} F K F^T
        """
        F = KinematicsHelpers.get_F(u)
        J = KinematicsHelpers.get_J(F)
         
        return (1.0 / J) * dot(F, dot(K, F.T))
    
    @staticmethod
    def get_k_inv(u, K_inv):
        """
        Permeability push-forward: k^{-1} = J F^{-T} K^{-1} F^{-1}
        """
        grad_u = grad(u)
        F = KinematicsHelpers.get_F(grad_u)
        J = KinematicsHelpers.get_J(F)
        F_inv = KinematicsHelpers.get_F_inv(F)
         
        return J * dot(F_inv.T, dot(K_inv, F_inv))

    @staticmethod
    def get_dk_inv(u, du, K_inv):
        """
        Linearization of k^{-1}.
        δ(k^{-1}) = δJ (F^{-T} K^{-1} F^{-1}) 
                  + J (δF^{-T}) K^{-1} F^{-1} 
                  + J F^{-T} K^{-1} (δF^{-1})
        """
        grad_u = grad(u)
        dF = grad(du)
         
        # Kinematics
        F = KinematicsHelpers.get_F(grad_u)
        J = KinematicsHelpers.get_J(F)
        F_inv = KinematicsHelpers.get_F_inv(F)
         
        # Linearizations
        dJ = KinematicsHelpers.get_J_LinU(F, dF)
        dF_inv = KinematicsHelpers.get_F_inv_LinU(F_inv, dF)
        dF_inv_T = dF_inv.T
         
        # Term 1: Variation of J
        term0 = dJ * dot(F_inv.T, dot(K_inv, F_inv))
         
        # Term 2: Variation of F^{-T}
        term1 = J * dot(dF_inv_T, dot(K_inv, F_inv))
         
        # Term 3: Variation of F^{-1}
        term2 = J * dot(F_inv.T, dot(K_inv, dF_inv))
         
        return term0 + term1 + term2
    
    @staticmethod
    def get_rho_tilda(rho_s0, phi):
        """
        Effective solid density in porous medium: ρ̃^P = (1 - φ) ρ_s0
        """
        return (1.0 - phi) * rho_s0


# -----------------------------------------------------------------------------
# Wrappers to map the requested 'poro_X' API to the Classes above
# -----------------------------------------------------------------------------
def poro_F(u):
    return KinematicsHelpers.get_F(grad(u))


def poro_k_inv(u, K_inv):
    return PoroElasticity.get_k_inv(u, K_inv)


def dporo_k_inv(u, du, K_inv):
    return PoroElasticity.get_dk_inv(u, du, K_inv)


def poro_P_neo_hookean(u, c, beta):
    return PoroElasticity.get_P(u, c, beta)


def dporo_P_neo_hookean(u, du, c, beta):
    return PoroElasticity.get_dP(u, du, c, beta)


# -----------------------------------------------------------------------------
# Residual/Jacobian building blocks
# -----------------------------------------------------------------------------
class StabilizationParams:
    gamma_u: float = 0.05
    gamma_p: float = 0.05
    gamma_div: float = 1e-3 * gamma_p
    c_v: float = 1.0/6.0
    c_k: float = 1.0
    c_t: float = 1.0/12.0


class BuildForms:
    def __init__(self, theta: Union[float, Constant],
                 rho_f: Union[float, Constant],
                 mu_f: Union[float, Constant],
                 phi: Union[float, Constant],
                 dt: Union[float, Constant]):
        self.theta = theta
        self.rho_f = rho_f
        self.mu_f = mu_f
        self.phi = phi
        self.dt = dt
        self.one_minus_theta = Constant(1.0) - theta
        self.lhs_eps_constant = Constant(2.0) * theta * mu_f
        self.rhs_eps_constant = Constant(2.0) * self.one_minus_theta * mu_f
        self.rho_f_over_dt = rho_f / dt
        self.theta_mult_rho_f = theta * rho_f
     
    # --- Fluid domain -----------------------------------------------------------
    def residual_fluid(self, v_k, p_k, v_n, p_n, 
                       dv_test, dp_test, 
                       *, dx_f):
        return (
            self.rho_f_over_dt * inner(v_k - v_n, dv_test)
            + self.theta_mult_rho_f * dot(dot(grad(v_k), v_k), dv_test)
            + self.one_minus_theta * dot(dot(grad(v_n), v_n), dv_test)
            + self.lhs_eps_constant * inner(epsilon(v_k), epsilon(dv_test))
            + self.rhs_eps_constant * inner(epsilon(v_n), epsilon(dv_test))
            - p_k * div(dv_test)
            + dp_test * div(v_k)
        ) * dx_f

    def jacobian_fluid(self, v_k, p_k, dv, dp, 
                       dv_test, dp_test, 
                       *, dx_f):
        return (
            self.rho_f_over_dt * inner(dv, dv_test)
            + self.theta_mult_rho_f * (dot(dot(grad(dv), v_k), dv_test) + dot(dot(grad(v_k), dv), dv_test))
            + self.lhs_eps_constant * inner(epsilon(dv), epsilon(dv_test))
            - dp * div(dv_test)
            + dp_test * div(dv)
        ) * dx_f
    # -- Porous Domain ---------------------------------------------------------
    def residual_poro(self, vp_k, up_k, pp_k,
                      vp_n, up_n, pp_n,
                      dvp_test, dup_test, dpp_test,
                      phi,
                      *,
                      rho_s0_tilde,
                      K_inv,
                      c_nh,
                      beta_nh,
                      dx_p):
        # Time derivatives
        vdot = (vp_k - vp_n) / self.dt
        udot = (up_k - up_n) / self.dt
        # spatial permability
        k_inv_k = poro_k_inv(up_k, K_inv)
        # First Piola-Kirchhoff stress
        Pk = poro_P_neo_hookean(up_k, c=c_nh, beta=beta_nh)
        Jk = det(poro_F(up_k))
        FinvT = inv(poro_F(up_k)).T
        # assuming phi to be constant
        # dphi = 0
        r = (
            # --- Mass balance in porous domain (test q_test = δp^P) -----------------
            inner(phi * div(udot), dpp_test)
            -inner(phi * (v_k - udot), grad(dpp_test))
            # boundary contribution is not added yet
            # --- Darcy momentum (test w_test = δv^P) ---------------------------------
            +inner(self.rho_f * vdot, dvp_test)
            - pk * div(dvp_test)
            -inner(self.rho_f * dot(grad(vp_k), udot), dvp_test)
            +inner(self.mu_f * phi * dot(k_inv_k, vp_k), dvp_test)
            -inner(self.mu_f * phi * dot(k_inv_k, udot), dvp_test)
            # --- Solid momentum (test U_test = δu^P) ---------------------------------
        )
        r = inner(dpp_test, self.phi * div(udot)) * dx_p
        r += -inner(grad(dpp_test), self.phi * (vp_k - udot)) * dx_p

        r += inner(dvp_test, rho_f * vdot) * dx_p
        r += -inner(div(dvp_test), pp_k) * dx_p

        r += -inner(dvp_test, rho_f * dot(grad(vp_k), udot)) * dx_p

        k_inv_k = poro_k_inv(up_k, K_inv)
        r += inner(dvp_test, self.mu_f * self.phi * dot(k_inv_k, vp_k)) * dx_p
        r += -inner(dvp_test, self.mu_f * self.phi * dot(k_inv_k, udot)) * dx_p

        uddot = (up_k - up_n) / (self.dt * self.dt)
        r += inner(dup_test, rho_s0_tilde * uddot) * dx_p

        Pk = poro_P_neo_hookean(up_k, c=c_nh, beta=beta_nh)
        r += inner(grad(dup_test), Pk) * dx_p

        Jk = det(poro_F(up_k))
        r += inner(dup_test, self.mu_f * Jk * (self.phi * self.phi) * dot(k_inv_k, udot)) * dx_p
        r += -inner(dup_test, self.mu_f * Jk * (self.phi * self.phi) * dot(k_inv_k, vp_k)) * dx_p

        return r


def residual_stab_cip_fluid(
    v,
    p,
    dv_test,
    dp_test,
    *,
    rho,
    mu,
    dt,
    theta,
    gamma_u,
    gamma_p,
    gamma_div,
    c_v,
    c_mu,
    c_t,
    dS_f,
    lagged_v_mag: Optional[object] = None,
):
    """Fluid CIP stabilization: paper eq. (21).

    By default we use a lagged velocity magnitude for the scaling.
    """
    h = CellDiameter()
    if lagged_v_mag is None:
        # Use current v in the scaling if no lagged magnitude is provided.
        vmag = sqrt(inner(v, v) + Constant(1.0e-12))
    else:
        vmag = lagged_v_mag

    # Phi_F_F = h^2( c_v rho |v|/h + 4 c_mu mu/h^2 + c_t rho/(theta dt) )
    Phi = h * c_v * rho * vmag + 4.0 * c_mu * mu + (h * h) * c_t * rho / (theta * dt)

    tau_u = gamma_u * Phi
    tau_p = gamma_p * (h * h) / Phi
    tau_div = gamma_div * Phi

    r = tau_u * inner(jump(grad(dv_test)), jump(grad(v))) * dS_f
    r += tau_p * inner(jump(grad(dp_test)), jump(grad(p))) * dS_f
    r += tau_div * inner(jump(div(dv_test)), jump(div(v))) * dS_f
    return r


def jacobian_stab_cip_fluid(
    dv,
    dp,
    dv_test,
    dp_test,
    *,
    rho,
    mu,
    dt,
    theta,
    gamma_u,
    gamma_p,
    gamma_div,
    c_v,
    c_mu,
    c_t,
    dS_f,
    lagged_v_mag: object,
):
    """Jacobian of fluid CIP (treating tau's as lagged/constant)."""
    h = CellDiameter()
    vmag = lagged_v_mag
    Phi = h * c_v * rho * vmag + 4.0 * c_mu * mu + (h * h) * c_t * rho / (theta * dt)
    tau_u = gamma_u * Phi
    tau_p = gamma_p * (h * h) / Phi
    tau_div = gamma_div * Phi

    a = tau_u * inner(jump(grad(dv_test)), jump(grad(dv))) * dS_f
    a += tau_p * inner(jump(grad(dp_test)), jump(grad(dp))) * dS_f
    a += tau_div * inner(jump(div(dv_test)), jump(div(dv))) * dS_f
    return a


def residual_ghost_penalty_fluid(
    v,
    p,
    dv_test,
    dp_test,
    *,
    mu,
    tau_p,
    tau_div,
    gamma_u_gp,
    gamma_p_gp,
    gamma_div_gp,
    dG_f,
):
    """Fluid ghost penalty: simplified j=1 version of paper eq. (23).

    We penalize jumps of normal derivatives across ghost facets.
    """
    h = CellDiameter()
    n = FacetNormal()

    # tau_GP,u = gamma_u_gp * mu * h
    tau_u_gp = gamma_u_gp * mu * h
    tau_p_gp = gamma_p_gp * tau_p
    tau_div_gp = gamma_div_gp * tau_div

    dnv = dot(grad(v), n)
    dntv = dot(grad(dv_test), n)
    dnp = dot(grad(p), n)
    dntp = dot(grad(dp_test), n)
    dndivv = dot(grad(div(v)), n)
    dntdivv = dot(grad(div(dv_test)), n)

    r = tau_u_gp * inner(jump(dntv), jump(dnv)) * dG_f
    r += tau_p_gp * inner(jump(dntp), jump(dnp)) * dG_f
    r += tau_div_gp * inner(jump(dntdivv), jump(dndivv)) * dG_f
    return r


def jacobian_ghost_penalty_fluid(
    dv,
    dp,
    dv_test,
    dp_test,
    *,
    mu,
    tau_p,
    tau_div,
    gamma_u_gp,
    gamma_p_gp,
    gamma_div_gp,
    dG_f,
):
    """Jacobian of the simplified ghost penalty (tau's treated constant)."""
    h = CellDiameter()
    n = FacetNormal()

    tau_u_gp = gamma_u_gp * mu * h
    tau_p_gp = gamma_p_gp * tau_p
    tau_div_gp = gamma_div_gp * tau_div

    dnv = dot(grad(dv), n)
    dntv = dot(grad(dv_test), n)
    dnp = dot(grad(dp), n)
    dntp = dot(grad(dp_test), n)
    dndivv = dot(grad(div(dv)), n)
    dntdivv = dot(grad(div(dv_test)), n)

    a = tau_u_gp * inner(jump(dntv), jump(dnv)) * dG_f
    a += tau_p_gp * inner(jump(dntp), jump(dnp)) * dG_f
    a += tau_div_gp * inner(jump(dntdivv), jump(dndivv)) * dG_f
    return a


# --- Poroelastic domain ------------------------------------------------------


def residual_poro(
    v_k,
    u_k,
    p_k,
    v_n,
    u_n,
    p_n,
    q_test,
    w_test,
    eta_test,
    *,
    rho_f,
    mu_f,
    rho_s0_tilde,
    phi,
    K_inv,
    c_nh,
    beta_nh,
    dt,
    theta,
    dx_p,
    dGamma_p=None,
):
    """Poroelastic residual: paper eq. (20) (without external loads).

    Unknowns are (v^P, u^P, p^P) where u^P is the skeleton displacement.
    """
    # Time derivatives
    vdot = (v_k - v_n) / dt
    udot = (u_k - u_n) / dt

    # --- Mass balance in porous domain (test q_test = δp^P) -----------------
    r = inner(q_test, phi * div(udot)) * dx_p
    r += -inner(grad(q_test), phi * (v_k - udot)) * dx_p

    if dGamma_p is not None:
        nP = FacetNormal()
        r += inner(q_test, phi * dot(v_k - udot, nP)) * dGamma_p

    # --- Darcy momentum (test w_test = δv^P) ---------------------------------
    r += inner(w_test, rho_f * vdot) * dx_p
    r += -inner(div(w_test), p_k) * dx_p

    # convection due to solid motion: (udot · ∇)v
    r += -inner(w_test, rho_f * dot(grad(v_k), udot)) * dx_p

    k_inv_k = poro_k_inv(u_k, K_inv)
    r += inner(w_test, mu_f * phi * dot(k_inv_k, v_k)) * dx_p
    r += -inner(w_test, mu_f * phi * dot(k_inv_k, udot)) * dx_p

    # --- Skeleton momentum (test eta_test = δu^P) ----------------------------
    # Second time derivative (backward difference). For a true 2nd order scheme
    # you may want to pass u_{n-1} via aux function and use (u_k - 2u_n + u_{n-1})/dt^2.
    uddot = (u_k - u_n) / (dt * dt)
    r += inner(eta_test, rho_s0_tilde * uddot) * dx_p

    Pk = poro_P_neo_hookean(u_k, c=c_nh, beta=beta_nh)
    r += inner(grad(eta_test), Pk) * dx_p

    Jk = det(poro_F(u_k))
    # reference drag uses J * k^{-1} (k^{-1} already contains J)
    r += inner(eta_test, mu_f * Jk * (phi * phi) * dot(k_inv_k, udot)) * dx_p
    r += -inner(eta_test, mu_f * Jk * (phi * phi) * dot(k_inv_k, v_k)) * dx_p

    FinvT = inv(poro_F(u_k)).T
    r += -inner(eta_test, Jk * phi * dot(FinvT, grad(p_k))) * dx_p

    return r


def jacobian_poro(
    v_k,
    u_k,
    p_k,
    u_n,
    dv,
    du,
    dp,
    q_test,
    w_test,
    eta_test,
    *,
    rho_f,
    mu_f,
    rho_s0_tilde,
    phi,
    K_inv,
    c_nh,
    beta_nh,
    dt,
    theta,
    dx_p,
    dGamma_p=None,
):
    """Jacobian of residual_poro (Gateaux derivative)."""
    dvdt = dv / dt
    dudt = du / dt

    a = inner(q_test, phi * div(dudt)) * dx_p
    a += -inner(grad(q_test), phi * (dv - dudt)) * dx_p
    if dGamma_p is not None:
        nP = FacetNormal()
        a += inner(q_test, phi * dot(dv - dudt, nP)) * dGamma_p

    # Darcy momentum
    a += inner(w_test, rho_f * dvdt) * dx_p
    a += -inner(div(w_test), dp) * dx_p

    udot = (u_k - u_n) / dt
    # convection: - rho (udot · ∇)v
    a += -inner(w_test, rho_f * dot(grad(dv), udot)) * dx_p
    a += -inner(w_test, rho_f * dot(grad(v_k), dudt)) * dx_p

    k_inv_k = poro_k_inv(u_k, K_inv)
    dk_inv = dporo_k_inv(u_k, du, K_inv)

    a += inner(w_test, mu_f * phi * (dot(k_inv_k, dv) + dot(dk_inv, v_k))) * dx_p
    a += -inner(w_test, mu_f * phi * (dot(k_inv_k, dudt) + dot(dk_inv, udot))) * dx_p

    # Skeleton momentum
    duddot = du / (dt * dt)
    a += inner(eta_test, rho_s0_tilde * duddot) * dx_p

    dP = dporo_P_neo_hookean(u_k, du, c=c_nh, beta=beta_nh)
    a += inner(grad(eta_test), dP) * dx_p

    Fk = poro_F(u_k)
    Jk = det(Fk)
    dJ = Jk * trace(dot(inv(Fk), grad(du)))

    # k^{-1} and its variation
    # k_inv_k already includes J. The reference drag uses J * k_inv.
    d_k_inv = dk_inv

    # term: mu * J * phi^2 * k_inv · udot
    a += inner(
        eta_test,
        mu_f
        * (phi * phi)
        * (
            dJ * dot(k_inv_k, udot)
            + Jk * (dot(d_k_inv, udot) + dot(k_inv_k, dudt))
        ),
    ) * dx_p

    # term: - mu * J * phi^2 * k_inv · v
    a += -inner(
        eta_test,
        mu_f
        * (phi * phi)
        * (
            dJ * dot(k_inv_k, v_k)
            + Jk * (dot(d_k_inv, v_k) + dot(k_inv_k, dv))
        ),
    ) * dx_p

    Finv = inv(Fk)
    FinvT = Finv.T
    dFinvT = (-dot(FinvT, dot(grad(du).T, FinvT)))

    # term: - eta · (J phi FinvT grad(p))
    a += -inner(
        eta_test,
        phi
        * (
            dJ * dot(FinvT, grad(p_k))
            + Jk * (dot(dFinvT, grad(p_k)) + dot(FinvT, grad(dp)))
        ),
    ) * dx_p

    return a


def residual_stab_cip_poro(
    v,
    p,
    w_test,
    q_test,
    *,
    rho_f,
    mu_f,
    phi,
    K_inv_scalar,
    dt,
    theta,
    gamma_p,
    gamma_div,
    c_k,
    c_t,
    dS_p,
):
    """Porous CIP stabilization: paper eq. (22)."""
    h = CellDiameter()
    # A scalar permeability magnitude (e.g. K) is expected for the scaling.
    Phi = (h * h) * (c_k * mu_f * phi / K_inv_scalar + c_t * rho_f / (theta * dt))
    tau_p = gamma_p * Phi
    tau_div = gamma_div * Phi

    r = tau_p * inner(jump(grad(q_test)), jump(grad(p))) * dS_p
    r += tau_div * inner(jump(div(w_test)), jump(div(v))) * dS_p
    return r


def jacobian_stab_cip_poro(
    dv,
    dp,
    w_test,
    q_test,
    *,
    rho_f,
    mu_f,
    phi,
    K_inv_scalar,
    dt,
    theta,
    gamma_p,
    gamma_div,
    c_k,
    c_t,
    dS_p,
):
    """Jacobian of porous CIP (tau's treated constant)."""
    h = CellDiameter()
    Phi = (h * h) * (c_k * mu_f * phi / K_inv_scalar + c_t * rho_f / (theta * dt))
    tau_p = gamma_p * Phi
    tau_div = gamma_div * Phi

    a = tau_p * inner(jump(grad(q_test)), jump(grad(dp))) * dS_p
    a += tau_div * inner(jump(div(w_test)), jump(div(dv))) * dS_p
    return a


# --- Interface terms ---------------------------------------------------------


def residual_interface_normal(
    vF,
    pF,
    vP,
    uP,
    dvF_test,
    dpF_test,
    dvP_test,
    duP_test,
    *,
    rho_f,
    mu_f,
    phi,
    dt,
    theta,
    gamma_n,
    zeta,
    c_v_gamma,
    c_t_gamma,
    g_sigma=None,
    g_sigma_n=None,
    g_n=None,
    dGamma=None,
    lagged_vF_mag=None,
):
    """Normal-direction Nitsche coupling, paper eq. (24)."""
    if dGamma is None:
        return Constant(0.0)
    n = FacetNormal()  # NEG->POS (poro -> fluid)
    nF = -n
    Pn = proj_n(n)

    # stress traction (fluid)
    tF = dot(sigma_fluid(vF, pF, mu_f), nF)
    tF_n = dot(Pn, tF)

    # kinematic mismatch (normal)
    udot = uP  # here uP is displacement; we need its time derivative
    # default: backward Euler time derivative
    # For interface terms we typically use the implicit velocity approximation.
    # If uP is already a velocity field, pass that instead.
    # In this template we assume uP is displacement and approximate udot.
    # (Caller may override by passing uP as already time-derivative.)
    # -> We interpret uP as displacement and set udot = (uP-uP_prev)/dt outside.
    # Here we assume uP provided is already udot.

    if g_sigma is None:
        g_sigma = Constant((0.0, 0.0))
    if g_sigma_n is None:
        g_sigma_n = Constant(0.0)
    if g_n is None:
        g_n = Constant(0.0)

    kin = vF - udot - phi * (vP - udot) - g_n * n  # vector; only normal used
    kin_n = dot(Pn, kin)

    # Interface scaling phi_F_Gamma (lagged)
    h = CellDiameter()
    if lagged_vF_mag is None:
        vmag = sqrt(inner(vF, vF) + Constant(1.0e-12))
    else:
        vmag = lagged_vF_mag
    phi_FG = mu_f + h * c_v_gamma * rho_f * vmag + (h * h) * c_t_gamma * rho_f / (theta * dt)

    r = inner(dvP_test + duP_test - dvF_test, tF_n) * dGamma
    r += -inner(dvP_test, g_sigma_n * dot(Pn, n)) * dGamma
    r += -inner(duP_test, dot(Pn, g_sigma)) * dGamma

    adj = dpF_test * nF + zeta * 2.0 * mu_f * dot(epsilon(dvF_test), nF)
    r += -inner(adj, kin_n) * dGamma

    r += phi_FG * gamma_n / h * inner(dvF_test - dvP_test - duP_test, kin_n) * dGamma
    return r


def jacobian_interface_normal(
    vF,
    pF,
    vP,
    udot,
    dvF,
    dpF,
    dvP,
    dudot,
    dvF_test,
    dpF_test,
    dvP_test,
    duP_test,
    *,
    rho_f,
    mu_f,
    phi,
    dt,
    theta,
    gamma_n,
    zeta,
    c_v_gamma,
    c_t_gamma,
    dGamma=None,
    lagged_vF_mag=None,
):
    if dGamma is None:
        return Constant(0.0)
    n = FacetNormal()
    nF = -n
    Pn = proj_n(n)

    # d(σ·nF)
    dsig = sigma_fluid(dvF, dpF, mu_f)
    dtF = dot(dsig, nF)
    dtF_n = dot(Pn, dtF)

    a = inner(dvP_test + duP_test - dvF_test, dtF_n) * dGamma

    dkin = dvF - dudot - phi * (dvP - dudot)
    dkin_n = dot(Pn, dkin)

    adj = dpF_test * nF + zeta * 2.0 * mu_f * dot(epsilon(dvF_test), nF)
    a += -inner(adj, dkin_n) * dGamma

    h = CellDiameter()
    vmag = lagged_vF_mag if lagged_vF_mag is not None else sqrt(inner(vF, vF) + Constant(1.0e-12))
    phi_FG = mu_f + h * c_v_gamma * rho_f * vmag + (h * h) * c_t_gamma * rho_f / (theta * dt)
    a += phi_FG * gamma_n / h * inner(dvF_test - dvP_test - duP_test, dkin_n) * dGamma
    return a


def residual_interface_tangential_nitsche(
    vF,
    pF,
    vP,
    udot,
    dvF_test,
    duP_test,
    *,
    mu_f,
    phi,
    beta_BJ,
    kappa,
    gamma_t,
    zeta,
    g_sigma=None,
    g_t=None,
    dGamma=None,
):
    """Tangential Nitsche coupling, paper eq. (28)."""
    if dGamma is None:
        return Constant(0.0)
    n = FacetNormal()
    nF = -n
    Pt = proj_t(n)
    h = CellDiameter()

    if g_sigma is None:
        g_sigma = Constant((0.0, 0.0))
    if g_t is None:
        g_t = Constant((0.0, 0.0))

    tF = dot(sigma_fluid(vF, pF, mu_f), nF)
    tF_t = dot(Pt, tF)

    c_BJ = vF - udot - beta_BJ * phi * (vP - udot) + kappa * tF - g_t
    c_BJ_t = dot(Pt, c_BJ)

    denom = kappa * mu_f + gamma_t * h

    r = inner(duP_test - dvF_test, tF_t) * dGamma
    r += -inner(duP_test, dot(Pt, g_sigma)) * dGamma

    r += zeta * (gamma_t * h) / denom * inner(
        -2.0 * mu_f * dot(epsilon(dvF_test), nF),
        c_BJ_t,
    ) * dGamma

    r += mu_f / denom * inner(dvF_test - duP_test, c_BJ_t) * dGamma
    return r


def jacobian_interface_tangential_nitsche(
    vF,
    pF,
    vP,
    udot,
    dvF,
    dpF,
    dvP,
    dudot,
    dvF_test,
    duP_test,
    *,
    mu_f,
    phi,
    beta_BJ,
    kappa,
    gamma_t,
    zeta,
    dGamma=None,
):
    if dGamma is None:
        return Constant(0.0)
    n = FacetNormal()
    nF = -n
    Pt = proj_t(n)
    h = CellDiameter()

    # traction derivative
    dtF = dot(sigma_fluid(dvF, dpF, mu_f), nF)
    dtF_t = dot(Pt, dtF)

    tF = dot(sigma_fluid(vF, pF, mu_f), nF)
    denom = kappa * mu_f + gamma_t * h

    a = inner(duP_test - dvF_test, dtF_t) * dGamma

    # c_BJ derivative
    dc_BJ = dvF - dudot - beta_BJ * phi * (dvP - dudot) + kappa * dtF
    dc_BJ_t = dot(Pt, dc_BJ)

    a += zeta * (gamma_t * h) / denom * inner(
        -2.0 * mu_f * dot(epsilon(dvF_test), nF),
        dc_BJ_t,
    ) * dGamma

    a += mu_f / denom * inner(dvF_test - duP_test, dc_BJ_t) * dGamma
    return a


# -----------------------------------------------------------------------------
# Build the full coupled forms
# -----------------------------------------------------------------------------


@dataclass
class FPIEulerianForms:
    residual_form: object
    jacobian_form: object


def build_fpi_eulerian_forms(
    *,
    mesh,
    dof_handler,
    level_set,
    domains: dict,
    mixed_space,
    w_k,
    w_n,
    dt: float,
    theta: float,
    # --- physical parameters
    rho_f: float,
    mu_f: float,
    rho_s0_tilde: float,
    phi: float = 0.5,
    K_inv: Optional[object] = None,
    # --- Neo-Hookean params
    c_nh: float = 1.0,
    beta_nh: float = 0.0,
    # --- stabilization params
    gamma_F_u: float = 1.0,
    gamma_F_p: float = 1.0,
    gamma_F_div: float = 1.0,
    c_v: float = 1.0,
    c_mu: float = 1.0,
    c_t: float = 1.0,
    gamma_GP_u: float = 1.0,
    gamma_GP_p: float = 1.0,
    gamma_GP_div: float = 1.0,
    gamma_P_p: float = 1.0,
    gamma_P_div: float = 1.0,
    c_k: float = 1.0,
    # --- interface params
    gamma_n: float = 10.0,
    gamma_t: float = 10.0,
    zeta: float = 1.0,
    beta_BJ: float = 1.0,
    kappa: float = 1.0,
    c_v_gamma: float = 1.0,
    c_t_gamma: float = 1.0,
    use_tangential_nitsche: bool = True,
    # --- optional manufactured/interface data
    g_sigma=None,
    g_sigma_n=None,
    g_n=None,
    g_t=None,
    quadrature_degree: int = 6,
):
    """Build the coupled residual and Jacobian forms.

    The mixed space is assumed to be ordered as
        (v^F, p^F, v^P, u^P, p^P)
    consistent with the provided TeX and prior discussion.
    """

    # Default permeability inverse (material) if not provided.
    if K_inv is None:
        K_inv = Identity(2)

    # Split unknowns and previous time step
    vF_k, pF_k, vP_k, uP_k, pP_k = split(w_k)
    vF_n, pF_n, vP_n, uP_n, pP_n = split(w_n)

    # Test + increments
    test_w = TestFunction(mixed_space)
    dvF_test, dpF_test, dvP_test, duP_test, dpP_test = split(test_w)

    dw = IncrementFunction(mixed_space)
    dvF, dpF, dvP, duP, dpP = split(dw)

    # Domain restrictions
    dom_F = domains["has_pos"]
    dom_P = domains["has_neg"]

    vF_kR = restrict(vF_k, dom_F)
    pF_kR = restrict(pF_k, dom_F)
    vF_nR = restrict(vF_n, dom_F)
    pF_nR = restrict(pF_n, dom_F)

    dvF_testR = restrict(dvF_test, dom_F)
    dpF_testR = restrict(dpF_test, dom_F)

    dvF_R = restrict(dvF, dom_F)
    dpF_R = restrict(dpF, dom_F)

    vP_kR = restrict(vP_k, dom_P)
    uP_kR = restrict(uP_k, dom_P)
    pP_kR = restrict(pP_k, dom_P)
    vP_nR = restrict(vP_n, dom_P)
    uP_nR = restrict(uP_n, dom_P)
    pP_nR = restrict(pP_n, dom_P)

    dvP_testR = restrict(dvP_test, dom_P)
    duP_testR = restrict(duP_test, dom_P)
    dpP_testR = restrict(dpP_test, dom_P)

    dvP_R = restrict(dvP, dom_P)
    duP_R = restrict(duP, dom_P)
    dpP_R = restrict(dpP, dom_P)

    # Measures
    dx_f = dx(
        defined_on=domains.get("active_cells", domains["has_pos"]),
        level_set=level_set,
        metadata={"quadrature_degree": quadrature_degree, "side": +1},
    )
    dx_p = dx(
        defined_on=domains.get("active_cells", domains["has_neg"]),
        level_set=level_set,
        metadata={"quadrature_degree": quadrature_degree, "side": -1},
    )
    dGamma = dInterface(
        defined_on=domains.get("cut_edges", domains.get("cut_facets")),
        level_set=level_set,
        metadata={"quadrature_degree": quadrature_degree},
    )

    dS_f = dFacetPatch(
        defined_on=domains.get("ghost_facets_pos", domains.get("facet_patches_pos")),
        level_set=level_set,
        metadata={"quadrature_degree": quadrature_degree},
    )
    dS_p = dFacetPatch(
        defined_on=domains.get("ghost_facets_neg", domains.get("facet_patches_neg")),
        level_set=level_set,
        metadata={"quadrature_degree": quadrature_degree},
    )
    dG_f = dFacetPatch(
        defined_on=domains.get("ghost_facets_pos", domains.get("facet_patches_pos")),
        level_set=level_set,
        metadata={"quadrature_degree": quadrature_degree},
    )

    # --- Fluid residual/Jacobian
    rhoF = Constant(rho_f)
    muF = Constant(mu_f)
    dt_c = Constant(dt)
    theta_c = Constant(theta)

    r = residual_fluid(
        vF_kR,
        pF_kR,
        vF_nR,
        pF_nR,
        dvF_testR,
        dpF_testR,
        rho=rhoF,
        mu=muF,
        dt=dt_c,
        theta=theta_c,
        dx_f=dx_f,
    )
    a = jacobian_fluid(
        vF_kR,
        pF_kR,
        dvF_R,
        dpF_R,
        dvF_testR,
        dpF_testR,
        rho=rhoF,
        mu=muF,
        dt=dt_c,
        theta=theta_c,
        dx_f=dx_f,
    )

    # CIP + ghost penalty (fluid)
    vFmag_lag = sqrt(inner(vF_nR, vF_nR) + Constant(1.0e-12))

    r_cip_f = residual_stab_cip_fluid(
        vF_kR,
        pF_kR,
        dvF_testR,
        dpF_testR,
        rho=rhoF,
        mu=muF,
        dt=dt_c,
        theta=theta_c,
        gamma_u=Constant(gamma_F_u),
        gamma_p=Constant(gamma_F_p),
        gamma_div=Constant(gamma_F_div),
        c_v=Constant(c_v),
        c_mu=Constant(c_mu),
        c_t=Constant(c_t),
        dS_f=dS_f,
        lagged_v_mag=vFmag_lag,
    )
    a_cip_f = jacobian_stab_cip_fluid(
        dvF_R,
        dpF_R,
        dvF_testR,
        dpF_testR,
        rho=rhoF,
        mu=muF,
        dt=dt_c,
        theta=theta_c,
        gamma_u=Constant(gamma_F_u),
        gamma_p=Constant(gamma_F_p),
        gamma_div=Constant(gamma_F_div),
        c_v=Constant(c_v),
        c_mu=Constant(c_mu),
        c_t=Constant(c_t),
        dS_f=dS_f,
        lagged_v_mag=vFmag_lag,
    )

    # We need tau_p and tau_div from the CIP scaling for ghost penalty.
    h = CellDiameter()
    PhiF = h * Constant(c_v) * rhoF * vFmag_lag + 4.0 * Constant(c_mu) * muF + (h * h) * Constant(c_t) * rhoF / (theta_c * dt_c)
    tau_p_f = Constant(gamma_F_p) * (h * h) / PhiF
    tau_div_f = Constant(gamma_F_div) * PhiF

    r_gp_f = residual_ghost_penalty_fluid(
        vF_kR,
        pF_kR,
        dvF_testR,
        dpF_testR,
        mu=muF,
        tau_p=tau_p_f,
        tau_div=tau_div_f,
        gamma_u_gp=Constant(gamma_GP_u),
        gamma_p_gp=Constant(gamma_GP_p),
        gamma_div_gp=Constant(gamma_GP_div),
        dG_f=dG_f,
    )
    a_gp_f = jacobian_ghost_penalty_fluid(
        dvF_R,
        dpF_R,
        dvF_testR,
        dpF_testR,
        mu=muF,
        tau_p=tau_p_f,
        tau_div=tau_div_f,
        gamma_u_gp=Constant(gamma_GP_u),
        gamma_p_gp=Constant(gamma_GP_p),
        gamma_div_gp=Constant(gamma_GP_div),
        dG_f=dG_f,
    )

    r += r_cip_f + r_gp_f
    a += a_cip_f + a_gp_f

    # --- Porous residual/Jacobian
    rP = residual_poro(
        vP_kR,
        uP_kR,
        pP_kR,
        vP_nR,
        uP_nR,
        pP_nR,
        dpP_testR,
        dvP_testR,
        duP_testR,
        rho_f=rhoF,
        mu_f=muF,
        rho_s0_tilde=Constant(rho_s0_tilde),
        phi=Constant(phi),
        K_inv=K_inv,
        c_nh=Constant(c_nh),
        beta_nh=Constant(beta_nh),
        dt=dt_c,
        theta=theta_c,
        dx_p=dx_p,
        dGamma_p=None,
    )
    aP = jacobian_poro(
        vP_kR,
        uP_kR,
        pP_kR,
        uP_nR,
        dvP_R,
        duP_R,
        dpP_R,
        dpP_testR,
        dvP_testR,
        duP_testR,
        rho_f=rhoF,
        mu_f=muF,
        rho_s0_tilde=Constant(rho_s0_tilde),
        phi=Constant(phi),
        K_inv=K_inv,
        c_nh=Constant(c_nh),
        beta_nh=Constant(beta_nh),
        dt=dt_c,
        theta=theta_c,
        dx_p=dx_p,
        dGamma_p=None,
    )

    # Porous CIP stabilization
    rP += residual_stab_cip_poro(
        vP_kR,
        pP_kR,
        dvP_testR,
        dpP_testR,
        rho_f=rhoF,
        mu_f=muF,
        phi=Constant(phi),
        K_inv_scalar=Constant(1.0),
        dt=dt_c,
        theta=theta_c,
        gamma_p=Constant(gamma_P_p),
        gamma_div=Constant(gamma_P_div),
        c_k=Constant(c_k),
        c_t=Constant(c_t),
        dS_p=dS_p,
    )
    aP += jacobian_stab_cip_poro(
        dvP_R,
        dpP_R,
        dvP_testR,
        dpP_testR,
        rho_f=rhoF,
        mu_f=muF,
        phi=Constant(phi),
        K_inv_scalar=Constant(1.0),
        dt=dt_c,
        theta=theta_c,
        gamma_p=Constant(gamma_P_p),
        gamma_div=Constant(gamma_P_div),
        c_k=Constant(c_k),
        c_t=Constant(c_t),
        dS_p=dS_p,
    )

    r += rP
    a += aP

    # --- Interface contributions
    # IMPORTANT: In the weak forms (24),(28) the solid velocity is ∂_t u^P.
    # In this 5-field formulation u^P is the displacement, so we approximate
    # udot via backward Euler:
    uPdot_kR = (uP_kR - uP_nR) / dt_c
    uPdot_nR = (uP_nR - uP_nR) / dt_c  # zero by default (no u_{n-1} available)
    dudot_R = duP_R / dt_c

    # Normal
    rN = residual_interface_normal(
        vF_kR,
        pF_kR,
        vP_kR,
        uPdot_kR,
        dvF_testR,
        dpF_testR,
        dvP_testR,
        duP_testR,
        rho_f=rhoF,
        mu_f=muF,
        phi=Constant(phi),
        dt=dt_c,
        theta=theta_c,
        gamma_n=Constant(gamma_n),
        zeta=Constant(zeta),
        c_v_gamma=Constant(c_v_gamma),
        c_t_gamma=Constant(c_t_gamma),
        g_sigma=g_sigma,
        g_sigma_n=g_sigma_n,
        g_n=g_n,
        dGamma=dGamma,
        lagged_vF_mag=vFmag_lag,
    )
    aN = jacobian_interface_normal(
        vF_kR,
        pF_kR,
        vP_kR,
        uPdot_kR,
        dvF_R,
        dpF_R,
        dvP_R,
        dudot_R,
        dvF_testR,
        dpF_testR,
        dvP_testR,
        duP_testR,
        rho_f=rhoF,
        mu_f=muF,
        phi=Constant(phi),
        dt=dt_c,
        theta=theta_c,
        gamma_n=Constant(gamma_n),
        zeta=Constant(zeta),
        c_v_gamma=Constant(c_v_gamma),
        c_t_gamma=Constant(c_t_gamma),
        dGamma=dGamma,
        lagged_vF_mag=vFmag_lag,
    )

    r += rN
    a += aN

    # Tangential (default: Nitsche eq. (28))
    if use_tangential_nitsche:
        rT = residual_interface_tangential_nitsche(
            vF_kR,
            pF_kR,
            vP_kR,
            uPdot_kR,
            dvF_testR,
            duP_testR,
            mu_f=muF,
            phi=Constant(phi),
            beta_BJ=Constant(beta_BJ),
            kappa=Constant(kappa),
            gamma_t=Constant(gamma_t),
            zeta=Constant(zeta),
            g_sigma=g_sigma,
            g_t=g_t,
            dGamma=dGamma,
        )
        aT = jacobian_interface_tangential_nitsche(
            vF_kR,
            pF_kR,
            vP_kR,
            uPdot_kR,
            dvF_R,
            dpF_R,
            dvP_R,
            dudot_R,
            dvF_testR,
            duP_testR,
            mu_f=muF,
            phi=Constant(phi),
            beta_BJ=Constant(beta_BJ),
            kappa=Constant(kappa),
            gamma_t=Constant(gamma_t),
            zeta=Constant(zeta),
            dGamma=dGamma,
        )
        r += rT
        a += aT

    return FPIEulerianForms(residual_form=r, jacobian_form=a)
