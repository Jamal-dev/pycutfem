"""Eulerian nonlinear poromechanics mixture helpers.

This module is deliberately separate from the Kratos-parity U-Pl helpers.  It
implements the reduced nonlinear saturated-mixture model described in the
private notes:

* Eulerian reference-map kinematics ``F = (I - grad(u_s))^{-1}``
* porosity from incompressible solid grains, ``phi = 1 - (1 - phi0) / J``
* isothermal compressible pore-fluid density
* deformation-dependent spatial permeability
* quasi-static mixture momentum coupled to the reduced pressure equation
* one-step-theta time discretization with a manual Newton tangent

All scalar/tensor constants created by this module are named so compiled
kernels can reuse the same structure when material values change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    det,
    div,
    dot,
    exp,
    grad,
    inner,
    inv,
    trace,
)


SkeletonLaw2D = Literal["neo_hookean"]


def _named_constant(value, name: str, *, dim: int | None = None, preserve: bool = True):
    """Create a JIT-stable named constant.

    The local code intentionally avoids anonymous numeric literals in UFL forms.
    This keeps C++/JIT kernel structure independent of runtime material values.
    """

    if isinstance(value, Constant):
        c = value
    else:
        c = Constant(value, dim=dim) if dim is not None else Constant(value)
    c._jit_name = str(name)
    if preserve:
        c._preserve_runtime_structure = True
    return c


@dataclass(frozen=True)
class ExponentialFluidEOS:
    """Isothermal barotropic EOS with positive density.

    ``rho(p) = rho_ref exp(c_f (p - p_ref))``.
    """

    density_ref: float
    compressibility: float
    pressure_ref: float = 0.0

    def rho(self, p, *, prefix: str = "nlp") -> object:
        rho_ref = _named_constant(self.density_ref, f"{prefix}_rho_f_ref")
        cf = _named_constant(self.compressibility, f"{prefix}_c_f")
        p_ref = _named_constant(self.pressure_ref, f"{prefix}_p_ref")
        return rho_ref * exp(cf * (p - p_ref))

    def compressibility_expr(self, *, prefix: str = "nlp") -> object:
        return _named_constant(self.compressibility, f"{prefix}_c_f")

    def drho_dp(self, p, *, prefix: str = "nlp") -> object:
        return self.rho(p, prefix=prefix) * self.compressibility_expr(prefix=prefix)

    def rho_value(self, p: float) -> float:
        return float(self.density_ref) * float(np.exp(float(self.compressibility) * (float(p) - float(self.pressure_ref))))

    def drho_dp_value(self, p: float) -> float:
        return self.rho_value(float(p)) * float(self.compressibility)


@dataclass(frozen=True)
class NonlinearPoromechanicsMaterial2D:
    """Material data for the Eulerian nonlinear mixture model."""

    phi0: float
    density_solid: float
    fluid_eos: ExponentialFluidEOS
    dynamic_viscosity_fluid: float
    permeability_ref: tuple[tuple[float, float], tuple[float, float]]
    skeleton_mu: float
    skeleton_lambda: float
    skeleton_law: SkeletonLaw2D = "neo_hookean"
    body_acceleration: tuple[float, float] = (0.0, 0.0)

    @property
    def permeability_ref_array(self) -> np.ndarray:
        arr = np.asarray(self.permeability_ref, dtype=float)
        if arr.shape != (2, 2):
            raise ValueError("permeability_ref must be a 2x2 tensor.")
        if not np.allclose(arr, arr.T):
            raise ValueError("permeability_ref must be symmetric.")
        eig = np.linalg.eigvalsh(arr)
        if np.any(eig <= 0.0):
            raise ValueError("permeability_ref must be positive definite.")
        return arr

    def validate(self) -> None:
        if not (0.0 < float(self.phi0) < 1.0):
            raise ValueError("phi0 must satisfy 0 < phi0 < 1.")
        if float(self.density_solid) <= 0.0:
            raise ValueError("density_solid must be positive.")
        if float(self.fluid_eos.density_ref) <= 0.0:
            raise ValueError("fluid reference density must be positive.")
        if float(self.fluid_eos.compressibility) < 0.0:
            raise ValueError("fluid compressibility must be non-negative.")
        if float(self.dynamic_viscosity_fluid) <= 0.0:
            raise ValueError("dynamic_viscosity_fluid must be positive.")
        _ = self.permeability_ref_array
        if float(self.skeleton_mu) <= 0.0:
            raise ValueError("skeleton_mu must be positive.")
        if float(self.skeleton_lambda) < 0.0:
            raise ValueError("skeleton_lambda must be non-negative.")
        if self.skeleton_law != "neo_hookean":
            raise ValueError(f"Unsupported skeleton_law {self.skeleton_law!r}.")


@dataclass(frozen=True)
class NonlinearPoromechanicsThetaSystem2D:
    """Reduced nonlinear residual and Newton tangent forms."""

    residual_form: object
    jacobian_form: object
    momentum_residual: object
    pressure_residual: object
    momentum_jacobian: object
    pressure_jacobian: object


@dataclass(frozen=True)
class NonlinearPoromechanicsFullDynamicThetaSystem2D:
    """Full two-velocity nonlinear mixture residual and auto-linearized tangent."""

    residual_form: object
    jacobian_form: object
    fluid_mass_residual: object
    solid_mass_residual: object
    fluid_momentum_residual: object
    solid_momentum_residual: object
    kinematic_residual: object


def eulerian_deformation_gradient_2d(u):
    """Return ``F = (I - grad(u))^{-1}`` for Eulerian displacement ``u``."""

    identity = _named_constant(np.eye(2, dtype=float), "nlp_identity_2d", dim=2, preserve=False)
    return inv(identity - grad(u))


def eulerian_deformation_gradient_variation_2d(u, du):
    """Return ``delta F = F grad(du) F``."""

    F = eulerian_deformation_gradient_2d(u)
    return dot(F, dot(grad(du), F))


def eulerian_jacobian_2d(u):
    """Return ``J = det(F)``."""

    return det(eulerian_deformation_gradient_2d(u))


def eulerian_jacobian_variation_2d(u, du):
    """Return ``delta J`` for Eulerian reference-map kinematics."""

    F = eulerian_deformation_gradient_2d(u)
    dF = eulerian_deformation_gradient_variation_2d(u, du)
    F_inv = _named_constant(np.eye(2, dtype=float), "nlp_identity_2d", dim=2, preserve=False) - grad(u)
    return det(F) * trace(dot(F_inv, dF))


def porosity_from_jacobian_2d(J, material: NonlinearPoromechanicsMaterial2D, *, prefix: str = "nlp"):
    """Return ``phi = 1 - (1 - phi0) / J``."""

    one = _named_constant(1.0, f"{prefix}_one", preserve=False)
    phi0 = _named_constant(material.phi0, f"{prefix}_phi0")
    return one - (one - phi0) / J


def porosity_from_jacobian_value(J: float, phi0: float) -> float:
    """Numeric companion for verification tests."""

    Jf = float(J)
    if Jf <= 0.0:
        raise ValueError("J must be positive.")
    return 1.0 - (1.0 - float(phi0)) / Jf


def spatial_permeability_from_reference_value(F, k0) -> np.ndarray:
    """Numeric push-forward ``k = J^{-1} F k0 F^T``."""

    F_arr = np.asarray(F, dtype=float)
    k0_arr = np.asarray(k0, dtype=float)
    J = float(np.linalg.det(F_arr))
    if J <= 0.0:
        raise ValueError("det(F) must be positive.")
    return (F_arr @ k0_arr @ F_arr.T) / J


def spatial_inverse_permeability_from_reference_value(F, k0) -> np.ndarray:
    """Numeric push-forward ``k^{-1} = J F^{-T} k0^{-1} F^{-1}``."""

    F_arr = np.asarray(F, dtype=float)
    k0_arr = np.asarray(k0, dtype=float)
    J = float(np.linalg.det(F_arr))
    if J <= 0.0:
        raise ValueError("det(F) must be positive.")
    F_inv = np.linalg.inv(F_arr)
    return J * (F_inv.T @ np.linalg.inv(k0_arr) @ F_inv)


def porosity_from_displacement_2d(u, material: NonlinearPoromechanicsMaterial2D, *, prefix: str = "nlp"):
    """Porosity induced by incompressible solid-grain mass conservation."""

    return porosity_from_jacobian_2d(eulerian_jacobian_2d(u), material, prefix=prefix)


def porosity_variation_2d(u, du, material: NonlinearPoromechanicsMaterial2D, *, prefix: str = "nlp"):
    """Return ``delta phi = (1 - phi) tr(F^{-1} delta F)``."""

    one = _named_constant(1.0, f"{prefix}_one", preserve=False)
    F_inv = _named_constant(np.eye(2, dtype=float), "nlp_identity_2d", dim=2, preserve=False) - grad(u)
    phi = porosity_from_displacement_2d(u, material, prefix=prefix)
    dF = eulerian_deformation_gradient_variation_2d(u, du)
    return (one - phi) * trace(dot(F_inv, dF))


def spatial_mobility_from_reference_2d(u, material: NonlinearPoromechanicsMaterial2D, *, prefix: str = "nlp"):
    """Return spatial hydraulic mobility ``k / mu_f``.

    The input permeability is interpreted as reference-frame permeability
    ``k0``. The spatial push-forward is ``k = J^{-1} F k0 F^T``.
    """

    F = eulerian_deformation_gradient_2d(u)
    J = det(F)
    one = _named_constant(1.0, f"{prefix}_one", preserve=False)
    inv_mu = one / _named_constant(material.dynamic_viscosity_fluid, f"{prefix}_mu_f")
    k0 = _named_constant(material.permeability_ref_array, f"{prefix}_k0", dim=2)
    return (inv_mu / J) * dot(F, dot(k0, F.T))


def spatial_mobility_variation_from_reference_2d(
    u,
    du,
    material: NonlinearPoromechanicsMaterial2D,
    *,
    prefix: str = "nlp",
):
    """Gateaux derivative of ``spatial_mobility_from_reference_2d``."""

    F = eulerian_deformation_gradient_2d(u)
    dF = eulerian_deformation_gradient_variation_2d(u, du)
    J = det(F)
    dJ = eulerian_jacobian_variation_2d(u, du)
    one = _named_constant(1.0, f"{prefix}_one", preserve=False)
    inv_mu = one / _named_constant(material.dynamic_viscosity_fluid, f"{prefix}_mu_f")
    k0 = _named_constant(material.permeability_ref_array, f"{prefix}_k0", dim=2)
    pushed = dot(F, dot(k0, F.T))
    dpushed = dot(dF, dot(k0, F.T)) + dot(F, dot(k0, dF.T))
    return inv_mu * ((dpushed / J) - (dJ / (J * J)) * pushed)


def spatial_inverse_permeability_from_reference_2d(
    u,
    material: NonlinearPoromechanicsMaterial2D,
    *,
    prefix: str = "nlp",
):
    """Return spatial inverse permeability ``k^{-1}``.

    If ``k0`` is reference permeability, then
    ``k^{-1} = J F^{-T} k0^{-1} F^{-1}``.
    """

    F = eulerian_deformation_gradient_2d(u)
    J = det(F)
    F_inv = _named_constant(np.eye(2, dtype=float), "nlp_identity_2d", dim=2, preserve=False) - grad(u)
    k0_inv = _named_constant(np.linalg.inv(material.permeability_ref_array), f"{prefix}_k0_inv", dim=2)
    return J * dot(F_inv.T, dot(k0_inv, F_inv))


def spatial_inverse_permeability_variation_from_reference_2d(
    u,
    du,
    material: NonlinearPoromechanicsMaterial2D,
    *,
    prefix: str = "nlp",
):
    """Gateaux derivative of ``spatial_inverse_permeability_from_reference_2d``."""

    F = eulerian_deformation_gradient_2d(u)
    dF = eulerian_deformation_gradient_variation_2d(u, du)
    J = det(F)
    F_inv = _named_constant(np.eye(2, dtype=float), "nlp_identity_2d", dim=2, preserve=False) - grad(u)
    dJ = J * trace(dot(F_inv, dF))
    dF_inv = _named_constant(-1.0, f"{prefix}_minus_one", preserve=False) * grad(du)
    k0_inv = _named_constant(np.linalg.inv(material.permeability_ref_array), f"{prefix}_k0_inv", dim=2)
    base = dot(F_inv.T, dot(k0_inv, F_inv))
    dbase = dot(dF_inv.T, dot(k0_inv, F_inv)) + dot(F_inv.T, dot(k0_inv, dF_inv))
    return dJ * base + J * dbase


def fluid_density_2d(p, material: NonlinearPoromechanicsMaterial2D, *, prefix: str = "nlp"):
    return material.fluid_eos.rho(p, prefix=prefix)


def fluid_density_variation_2d(p, dp, material: NonlinearPoromechanicsMaterial2D, *, prefix: str = "nlp"):
    return material.fluid_eos.drho_dp(p, prefix=prefix) * dp


def fluid_compressibility_2d(material: NonlinearPoromechanicsMaterial2D, *, prefix: str = "nlp"):
    return material.fluid_eos.compressibility_expr(prefix=prefix)


def darcy_flux_2d(u, p, material: NonlinearPoromechanicsMaterial2D, *, prefix: str = "nlp"):
    """Return ``q = -(k/mu_f) (grad(p) - rho_f b)``."""

    minus_one = _named_constant(-1.0, f"{prefix}_minus_one", preserve=False)
    b = _named_constant(np.asarray(material.body_acceleration, dtype=float), f"{prefix}_body_acceleration", dim=1)
    rho_f = fluid_density_2d(p, material, prefix=prefix)
    g = grad(p) - rho_f * b
    return minus_one * dot(spatial_mobility_from_reference_2d(u, material, prefix=prefix), g)


def darcy_flux_variation_2d(u, du, p, dp, material: NonlinearPoromechanicsMaterial2D, *, prefix: str = "nlp"):
    """Gateaux derivative of the Darcy flux with respect to ``(u,p)``."""

    minus_one = _named_constant(-1.0, f"{prefix}_minus_one", preserve=False)
    b = _named_constant(np.asarray(material.body_acceleration, dtype=float), f"{prefix}_body_acceleration", dim=1)
    Km = spatial_mobility_from_reference_2d(u, material, prefix=prefix)
    dKm = spatial_mobility_variation_from_reference_2d(u, du, material, prefix=prefix)
    rho_f = fluid_density_2d(p, material, prefix=prefix)
    drho_f = fluid_density_variation_2d(p, dp, material, prefix=prefix)
    g = grad(p) - rho_f * b
    dg = grad(dp) - drho_f * b
    return minus_one * (dot(dKm, g) + dot(Km, dg))


def skeleton_stress_neo_hookean_2d(u, material: NonlinearPoromechanicsMaterial2D, *, prefix: str = "nlp"):
    """Compressible finite-deformation effective skeleton Cauchy stress.

    ``sigma = mu/J (B - I) + lambda (J - 1) I``.
    """

    F = eulerian_deformation_gradient_2d(u)
    J = det(F)
    identity = _named_constant(np.eye(2, dtype=float), "nlp_identity_2d", dim=2, preserve=False)
    one = _named_constant(1.0, f"{prefix}_one", preserve=False)
    mu_s = _named_constant(material.skeleton_mu, f"{prefix}_skeleton_mu")
    lambda_s = _named_constant(material.skeleton_lambda, f"{prefix}_skeleton_lambda")
    B = dot(F, F.T)
    return (mu_s / J) * (B - identity) + lambda_s * (J - one) * identity


def skeleton_stress_variation_neo_hookean_2d(
    u,
    du,
    material: NonlinearPoromechanicsMaterial2D,
    *,
    prefix: str = "nlp",
):
    """Gateaux derivative of ``skeleton_stress_neo_hookean_2d``."""

    F = eulerian_deformation_gradient_2d(u)
    dF = eulerian_deformation_gradient_variation_2d(u, du)
    J = det(F)
    dJ = eulerian_jacobian_variation_2d(u, du)
    identity = _named_constant(np.eye(2, dtype=float), "nlp_identity_2d", dim=2, preserve=False)
    mu_s = _named_constant(material.skeleton_mu, f"{prefix}_skeleton_mu")
    lambda_s = _named_constant(material.skeleton_lambda, f"{prefix}_skeleton_lambda")
    B = dot(F, F.T)
    dB = dot(dF, F.T) + dot(F, dF.T)
    pref = mu_s / J
    dpref = _named_constant(-1.0, f"{prefix}_minus_one", preserve=False) * mu_s * dJ / (J * J)
    return dpref * (B - identity) + pref * dB + lambda_s * dJ * identity


def skeleton_stress_2d(u, material: NonlinearPoromechanicsMaterial2D, *, prefix: str = "nlp"):
    if material.skeleton_law == "neo_hookean":
        return skeleton_stress_neo_hookean_2d(u, material, prefix=prefix)
    raise ValueError(f"Unsupported skeleton_law {material.skeleton_law!r}.")


def skeleton_stress_variation_2d(u, du, material: NonlinearPoromechanicsMaterial2D, *, prefix: str = "nlp"):
    if material.skeleton_law == "neo_hookean":
        return skeleton_stress_variation_neo_hookean_2d(u, du, material, prefix=prefix)
    raise ValueError(f"Unsupported skeleton_law {material.skeleton_law!r}.")


def mixture_density_2d(phi, rho_f, material: NonlinearPoromechanicsMaterial2D, *, prefix: str = "nlp"):
    one = _named_constant(1.0, f"{prefix}_one", preserve=False)
    rho_s = _named_constant(material.density_solid, f"{prefix}_rho_s")
    return phi * rho_f + (one - phi) * rho_s


def mixture_density_variation_2d(
    phi,
    dphi,
    rho_f,
    drho_f,
    material: NonlinearPoromechanicsMaterial2D,
    *,
    prefix: str = "nlp",
):
    rho_s = _named_constant(material.density_solid, f"{prefix}_rho_s")
    return dphi * (rho_f - rho_s) + phi * drho_f


def _solid_velocity_theta_terms(u_k, u_n, u_nm1, du, dt, theta, *, prefix: str = "nlp"):
    one = _named_constant(1.0, f"{prefix}_one", preserve=False)
    v_s_k = (u_k - u_n) / dt
    if u_nm1 is None:
        zero = _named_constant(0.0, f"{prefix}_zero", preserve=False)
        v_s_n = zero * v_s_k
    else:
        v_s_n = (u_n - u_nm1) / dt
    v_s_theta = theta * v_s_k + (one - theta) * v_s_n
    dv_s = du / dt
    return v_s_k, v_s_n, v_s_theta, dv_s


def build_nonlinear_poromechanics_reduced_theta_system_2d(
    *,
    u_trial: VectorTrialFunction,
    p_trial: TrialFunction,
    u_test: VectorTestFunction,
    p_test: TestFunction,
    u_current: VectorFunction,
    p_current: Function,
    u_prev: VectorFunction,
    p_prev: Function,
    material: NonlinearPoromechanicsMaterial2D,
    dt,
    theta,
    dx_measure,
    u_prevprev: VectorFunction | None = None,
    pressure_source=None,
    body_force=None,
    traction=None,
    traction_measure=None,
) -> NonlinearPoromechanicsThetaSystem2D:
    """Build residual and manual Newton tangent for the reduced model.

    ``u_current`` and ``p_current`` are the Newton iterate at ``n+1``.
    ``u_trial`` and ``p_trial`` are Newton increments.  The returned forms are
    suitable for ``NewtonSolver(..., backend="cpp")``.
    """

    material.validate()

    dt_c = _named_constant(dt, "nlp_dt")
    theta_c = _named_constant(theta, "nlp_theta")
    one = _named_constant(1.0, "nlp_one", preserve=False)
    cf = fluid_compressibility_2d(material)
    identity = _named_constant(np.eye(2, dtype=float), "nlp_identity_2d", dim=2, preserve=False)
    b = _named_constant(np.asarray(material.body_acceleration, dtype=float), "nlp_body_acceleration", dim=1)

    phi_k = porosity_from_displacement_2d(u_current, material)
    phi_n = porosity_from_displacement_2d(u_prev, material)
    phi_theta = theta_c * phi_k + (one - theta_c) * phi_n
    dphi_k = porosity_variation_2d(u_current, u_trial, material)

    p_theta = theta_c * p_current + (one - theta_c) * p_prev
    rho_f_k = fluid_density_2d(p_current, material)
    rho_f_n = fluid_density_2d(p_prev, material)
    rho_f_theta = fluid_density_2d(p_theta, material)
    drho_f_k = fluid_density_variation_2d(p_current, p_trial, material)
    drho_f_theta = fluid_density_variation_2d(p_theta, theta_c * p_trial, material)

    rho_mix_theta = mixture_density_2d(phi_theta, rho_f_theta, material)
    drho_mix_theta = mixture_density_variation_2d(
        phi_theta,
        theta_c * dphi_k,
        rho_f_theta,
        drho_f_theta,
        material,
    )

    sigma_k = skeleton_stress_2d(u_current, material)
    sigma_n = skeleton_stress_2d(u_prev, material)
    sigma_theta = theta_c * sigma_k + (one - theta_c) * sigma_n
    dsigma_k = skeleton_stress_variation_2d(u_current, u_trial, material)

    total_stress_theta = (one - phi_theta) * sigma_theta - phi_theta * p_theta * identity
    dphi_theta = theta_c * dphi_k
    dsigma_theta = theta_c * dsigma_k
    dp_theta = theta_c * p_trial
    dtotal_stress = (
        (_named_constant(-1.0, "nlp_minus_one", preserve=False) * dphi_theta) * sigma_theta
        + (one - phi_theta) * dsigma_theta
        - (dphi_theta * p_theta + phi_theta * dp_theta) * identity
    )

    momentum_residual = inner(total_stress_theta, grad(u_test)) * dx_measure
    if body_force is None:
        momentum_residual += _named_constant(-1.0, "nlp_minus_one", preserve=False) * dot(
            rho_mix_theta * b,
            u_test,
        ) * dx_measure
    else:
        momentum_residual += _named_constant(-1.0, "nlp_minus_one", preserve=False) * dot(
            body_force,
            u_test,
        ) * dx_measure
    if traction is not None and traction_measure is not None:
        momentum_residual += _named_constant(-1.0, "nlp_minus_one", preserve=False) * dot(
            traction,
            u_test,
        ) * traction_measure

    momentum_jacobian = inner(dtotal_stress, grad(u_test)) * dx_measure
    if body_force is None:
        momentum_jacobian += _named_constant(-1.0, "nlp_minus_one", preserve=False) * dot(
            drho_mix_theta * b,
            u_test,
        ) * dx_measure

    v_s_k, v_s_n, v_s_theta, dv_s = _solid_velocity_theta_terms(
        u_current,
        u_prev,
        u_prevprev,
        u_trial,
        dt_c,
        theta_c,
    )

    p_dot = (p_current - p_prev) / dt_c
    dp_dot = p_trial / dt_c
    adv_p_k = dot(v_s_k, grad(p_current))
    adv_p_n = dot(v_s_n, grad(p_prev))
    d_adv_p_k = dot(dv_s, grad(p_current)) + dot(v_s_k, grad(p_trial))
    material_pressure_derivative = p_dot + theta_c * adv_p_k + (one - theta_c) * adv_p_n
    d_material_pressure_derivative = dp_dot + theta_c * d_adv_p_k

    div_v_s_k = (div(u_current) - div(u_prev)) / dt_c
    if u_prevprev is None:
        div_v_s_n = _named_constant(0.0, "nlp_zero", preserve=False) * div_v_s_k
    else:
        div_v_s_n = (div(u_prev) - div(u_prevprev)) / dt_c
    div_v_s_theta = theta_c * div_v_s_k + (one - theta_c) * div_v_s_n
    d_div_v_s_theta = theta_c * div(u_trial) / dt_c

    q_k = darcy_flux_2d(u_current, p_current, material)
    q_n = darcy_flux_2d(u_prev, p_prev, material)
    q_theta = theta_c * q_k + (one - theta_c) * q_n
    dq_k = darcy_flux_variation_2d(u_current, u_trial, p_current, p_trial, material)
    dq_theta = theta_c * dq_k

    grad_p_theta = theta_c * grad(p_current) + (one - theta_c) * grad(p_prev)
    d_grad_p_theta = theta_c * grad(p_trial)

    source = _named_constant(0.0, "nlp_zero", preserve=False) if pressure_source is None else pressure_source

    pressure_residual = p_test * (
        phi_theta * cf * material_pressure_derivative
        + cf * dot(q_theta, grad_p_theta)
        + div_v_s_theta
        - source
    ) * dx_measure
    pressure_residual += _named_constant(-1.0, "nlp_minus_one", preserve=False) * inner(
        q_theta,
        grad(p_test),
    ) * dx_measure

    pressure_jacobian = p_test * (
        (theta_c * dphi_k) * cf * material_pressure_derivative
        + phi_theta * cf * d_material_pressure_derivative
        + cf * dot(dq_theta, grad_p_theta)
        + cf * dot(q_theta, d_grad_p_theta)
        + d_div_v_s_theta
    ) * dx_measure
    pressure_jacobian += _named_constant(-1.0, "nlp_minus_one", preserve=False) * inner(
        dq_theta,
        grad(p_test),
    ) * dx_measure

    residual = momentum_residual + pressure_residual
    jacobian = momentum_jacobian + pressure_jacobian

    return NonlinearPoromechanicsThetaSystem2D(
        residual_form=residual,
        jacobian_form=jacobian,
        momentum_residual=momentum_residual,
        pressure_residual=pressure_residual,
        momentum_jacobian=momentum_jacobian,
        pressure_jacobian=pressure_jacobian,
    )


def build_nonlinear_poromechanics_full_dynamic_theta_system_2d(
    *,
    u_trial: VectorTrialFunction,
    vs_trial: VectorTrialFunction,
    vf_trial: VectorTrialFunction,
    p_trial: TrialFunction,
    phi_trial: TrialFunction,
    u_test: VectorTestFunction,
    vs_test: VectorTestFunction,
    vf_test: VectorTestFunction,
    p_test: TestFunction,
    phi_test: TestFunction,
    u_current: VectorFunction,
    vs_current: VectorFunction,
    vf_current: VectorFunction,
    p_current: Function,
    phi_current: Function,
    u_prev: VectorFunction,
    vs_prev: VectorFunction,
    vf_prev: VectorFunction,
    p_prev: Function,
    phi_prev: Function,
    material: NonlinearPoromechanicsMaterial2D,
    dt,
    theta,
    dx_measure,
    fluid_mass_source=None,
    solid_mass_source=None,
    fluid_momentum_source=None,
    solid_momentum_source=None,
    kinematic_source=None,
    solid_traction=None,
    solid_traction_measure=None,
) -> NonlinearPoromechanicsFullDynamicThetaSystem2D:
    """Build the full dynamic two-velocity model.

    This is the stage-5 form entry point.  It keeps porosity independent and
    solves fluid mass, solid mass, fluid momentum, solid momentum, and the
    Eulerian kinematic constraint.  Because the full tangent is large, the
    Jacobian is generated through pycutfem's native Gateaux differentiation.
    """

    material.validate()

    dt_c = _named_constant(dt, "nlp_dt")
    theta_c = _named_constant(theta, "nlp_theta")
    one = _named_constant(1.0, "nlp_one", preserve=False)
    minus_one = _named_constant(-1.0, "nlp_minus_one", preserve=False)
    cf = fluid_compressibility_2d(material)
    rho_s = _named_constant(material.density_solid, "nlp_rho_s")
    mu_f = _named_constant(material.dynamic_viscosity_fluid, "nlp_mu_f")
    b = _named_constant(np.asarray(material.body_acceleration, dtype=float), "nlp_body_acceleration", dim=1)

    p_theta = theta_c * p_current + (one - theta_c) * p_prev
    phi_theta = theta_c * phi_current + (one - theta_c) * phi_prev
    vf_theta = theta_c * vf_current + (one - theta_c) * vf_prev
    vs_theta = theta_c * vs_current + (one - theta_c) * vs_prev
    grad_p_theta = theta_c * grad(p_current) + (one - theta_c) * grad(p_prev)
    grad_phi_theta = theta_c * grad(phi_current) + (one - theta_c) * grad(phi_prev)
    div_vf_theta = theta_c * div(vf_current) + (one - theta_c) * div(vf_prev)
    div_vs_theta = theta_c * div(vs_current) + (one - theta_c) * div(vs_prev)
    grad_vf_theta = theta_c * grad(vf_current) + (one - theta_c) * grad(vf_prev)
    grad_vs_theta = theta_c * grad(vs_current) + (one - theta_c) * grad(vs_prev)

    rho_f_current = fluid_density_2d(p_current, material)
    rho_f_theta = fluid_density_2d(p_theta, material)

    p_dot = (p_current - p_prev) / dt_c
    phi_dot = (phi_current - phi_prev) / dt_c
    vf_dot = (vf_current - vf_prev) / dt_c
    vs_dot = (vs_current - vs_prev) / dt_c
    u_dot = (u_current - u_prev) / dt_c

    dp_theta = theta_c * p_trial
    dphi_theta = theta_c * phi_trial
    dvf_theta = theta_c * vf_trial
    dvs_theta = theta_c * vs_trial
    grad_dp_theta = theta_c * grad(p_trial)
    grad_dphi_theta = theta_c * grad(phi_trial)
    div_dvf_theta = theta_c * div(vf_trial)
    div_dvs_theta = theta_c * div(vs_trial)
    grad_dvf_theta = theta_c * grad(vf_trial)
    grad_dvs_theta = theta_c * grad(vs_trial)

    d_f_p_theta = p_dot + dot(grad_p_theta, vf_theta)
    delta_d_f_p_theta = p_trial / dt_c + dot(grad_p_theta, dvf_theta) + dot(grad_dp_theta, vf_theta)

    c_f_phi_theta = phi_dot + dot(grad_phi_theta, vf_theta) + phi_theta * div_vf_theta
    delta_c_f_phi_theta = (
        phi_trial / dt_c
        + dot(grad_phi_theta, dvf_theta)
        + dot(grad_dphi_theta, vf_theta)
        + dphi_theta * div_vf_theta
        + phi_theta * div_dvf_theta
    )
    fluid_mass_rhs = _named_constant(0.0, "nlp_zero", preserve=False) if fluid_mass_source is None else fluid_mass_source
    fluid_mass = p_test * (phi_theta * cf * d_f_p_theta + c_f_phi_theta - fluid_mass_rhs) * dx_measure
    fluid_mass_jac = p_test * (
        dphi_theta * cf * d_f_p_theta
        + phi_theta * cf * delta_d_f_p_theta
        + delta_c_f_phi_theta
    ) * dx_measure

    solid_mass_flux_div = (
        minus_one * dot(grad_phi_theta, vs_theta)
        + (one - phi_theta) * div_vs_theta
    )
    solid_mass_rhs = _named_constant(0.0, "nlp_zero", preserve=False) if solid_mass_source is None else solid_mass_source
    solid_mass = phi_test * (minus_one * phi_dot + solid_mass_flux_div - solid_mass_rhs) * dx_measure
    solid_mass_jac = phi_test * (
        minus_one * (phi_trial / dt_c)
        - dot(grad_phi_theta, dvs_theta)
        - dot(grad_dphi_theta, vs_theta)
        - dphi_theta * div_vs_theta
        + (one - phi_theta) * div_dvs_theta
    ) * dx_measure

    q_theta = phi_theta * (vf_theta - vs_theta)
    dq_theta = dphi_theta * (vf_theta - vs_theta) + phi_theta * (dvf_theta - dvs_theta)
    k_inv_theta = theta_c * spatial_inverse_permeability_from_reference_2d(u_current, material) + (
        one - theta_c
    ) * spatial_inverse_permeability_from_reference_2d(u_prev, material)
    dk_inv_theta = theta_c * spatial_inverse_permeability_variation_from_reference_2d(
        u_current,
        u_trial,
        material,
    )
    drag_theta = phi_theta * mu_f * dot(k_inv_theta, q_theta)
    ddrag_theta = (
        dphi_theta * mu_f * dot(k_inv_theta, q_theta)
        + phi_theta * mu_f * (dot(dk_inv_theta, q_theta) + dot(k_inv_theta, dq_theta))
    )

    d_f_vf_theta = vf_dot + dot(grad_vf_theta, vf_theta)
    delta_d_f_vf_theta = (
        vf_trial / dt_c
        + dot(grad_dvf_theta, vf_theta)
        + dot(grad_vf_theta, dvf_theta)
    )
    fluid_momentum_rhs = _named_constant(
        np.zeros(2, dtype=float),
        "nlp_zero_vector",
        dim=1,
        preserve=False,
    ) if fluid_momentum_source is None else fluid_momentum_source
    fluid_momentum = inner(
        phi_theta * rho_f_theta * d_f_vf_theta
        + phi_theta * grad(p_theta)
        + drag_theta
        - phi_theta * rho_f_theta * b
        - fluid_momentum_rhs,
        vf_test,
    ) * dx_measure
    drho_f_theta = fluid_density_variation_2d(p_theta, dp_theta, material)
    fluid_momentum_jac = inner(
        (dphi_theta * rho_f_theta + phi_theta * drho_f_theta) * d_f_vf_theta
        + phi_theta * rho_f_theta * delta_d_f_vf_theta
        + dphi_theta * grad_p_theta
        + phi_theta * grad_dp_theta
        + ddrag_theta
        - (dphi_theta * rho_f_theta + phi_theta * drho_f_theta) * b,
        vf_test,
    ) * dx_measure

    d_s_vs_theta = vs_dot + dot(grad_vs_theta, vs_theta)
    delta_d_s_vs_theta = (
        vs_trial / dt_c
        + dot(grad_dvs_theta, vs_theta)
        + dot(grad_vs_theta, dvs_theta)
    )
    sigma_theta = theta_c * skeleton_stress_2d(u_current, material) + (one - theta_c) * skeleton_stress_2d(
        u_prev,
        material,
    )
    dsigma_theta = theta_c * skeleton_stress_variation_2d(u_current, u_trial, material)
    solid_momentum_rhs = _named_constant(
        np.zeros(2, dtype=float),
        "nlp_zero_vector",
        dim=1,
        preserve=False,
    ) if solid_momentum_source is None else solid_momentum_source
    solid_momentum = inner((one - phi_theta) * rho_s * d_s_vs_theta, vs_test) * dx_measure
    solid_momentum += inner((one - phi_theta) * sigma_theta, grad(vs_test)) * dx_measure
    solid_momentum += dot(p_theta * grad_phi_theta, vs_test) * dx_measure
    solid_momentum += minus_one * dot(drag_theta, vs_test) * dx_measure
    solid_momentum += minus_one * dot((one - phi_theta) * rho_s * b, vs_test) * dx_measure
    solid_momentum += minus_one * dot(solid_momentum_rhs, vs_test) * dx_measure
    if solid_traction is not None and solid_traction_measure is not None:
        solid_momentum += minus_one * dot(solid_traction, vs_test) * solid_traction_measure
    solid_momentum_jac = inner(
        minus_one * dphi_theta * rho_s * d_s_vs_theta
        + (one - phi_theta) * rho_s * delta_d_s_vs_theta,
        vs_test,
    ) * dx_measure
    solid_momentum_jac += inner(
        minus_one * dphi_theta * sigma_theta + (one - phi_theta) * dsigma_theta,
        grad(vs_test),
    ) * dx_measure
    solid_momentum_jac += dot(dp_theta * grad_phi_theta + p_theta * grad_dphi_theta, vs_test) * dx_measure
    solid_momentum_jac += minus_one * dot(ddrag_theta, vs_test) * dx_measure
    solid_momentum_jac += dot(dphi_theta * rho_s * b, vs_test) * dx_measure

    d_s_u_theta = u_dot + dot(grad(u_current), vs_theta)
    kinematic_rhs = _named_constant(
        np.zeros(2, dtype=float),
        "nlp_zero_vector",
        dim=1,
        preserve=False,
    ) if kinematic_source is None else kinematic_source
    kinematic = inner(d_s_u_theta - vs_theta - kinematic_rhs, u_test) * dx_measure
    kinematic_jac = inner(
        u_trial / dt_c + dot(grad(u_trial), vs_theta) + dot(grad(u_current), dvs_theta) - dvs_theta,
        u_test,
    ) * dx_measure

    residual = fluid_mass + solid_mass + fluid_momentum + solid_momentum + kinematic
    jacobian = fluid_mass_jac + solid_mass_jac + fluid_momentum_jac + solid_momentum_jac + kinematic_jac

    return NonlinearPoromechanicsFullDynamicThetaSystem2D(
        residual_form=residual,
        jacobian_form=jacobian,
        fluid_mass_residual=fluid_mass,
        solid_mass_residual=solid_mass,
        fluid_momentum_residual=fluid_momentum,
        solid_momentum_residual=solid_momentum,
        kinematic_residual=kinematic,
    )

__all__ = [
    "ExponentialFluidEOS",
    "NonlinearPoromechanicsMaterial2D",
    "NonlinearPoromechanicsFullDynamicThetaSystem2D",
    "NonlinearPoromechanicsThetaSystem2D",
    "build_nonlinear_poromechanics_reduced_theta_system_2d",
    "build_nonlinear_poromechanics_full_dynamic_theta_system_2d",
    "darcy_flux_2d",
    "darcy_flux_variation_2d",
    "eulerian_deformation_gradient_2d",
    "eulerian_deformation_gradient_variation_2d",
    "eulerian_jacobian_2d",
    "eulerian_jacobian_variation_2d",
    "fluid_compressibility_2d",
    "fluid_density_2d",
    "fluid_density_variation_2d",
    "mixture_density_2d",
    "mixture_density_variation_2d",
    "porosity_from_displacement_2d",
    "porosity_from_jacobian_2d",
    "porosity_from_jacobian_value",
    "porosity_variation_2d",
    "skeleton_stress_2d",
    "skeleton_stress_variation_2d",
    "spatial_mobility_from_reference_2d",
    "spatial_mobility_variation_from_reference_2d",
    "spatial_inverse_permeability_from_reference_2d",
    "spatial_inverse_permeability_variation_from_reference_2d",
    "spatial_inverse_permeability_from_reference_value",
    "spatial_permeability_from_reference_value",
]
