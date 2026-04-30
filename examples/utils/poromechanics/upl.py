"""2D U-Pl small-strain poromechanics forms.

The sign convention follows the existing pycutfem consolidation benchmark and
Kratos' elemental RHS layout: the displacement equation is assembled with
``-sigma(u):epsilon(v) + alpha p div(v)``. This preserves the current local
regression while making the U-Pl block reusable by NIRB and other examples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycutfem.ufl.expressions import (
    Constant,
    ElementWiseConstant,
    div,
    dot,
    grad,
    inner,
    trace,
)

from .materials import UPlMaterial2D


def _named_constant(value, name: str | None = None, *, dim: int | None = None, preserve: bool = True):
    """Create or annotate a JIT-stable constant used by poromechanics forms."""

    if isinstance(value, (Constant, ElementWiseConstant)):
        c = value
    elif hasattr(value, "__dict__") and not isinstance(value, np.ndarray):
        return value
    else:
        c = Constant(value, dim=dim) if dim is not None else Constant(value)
    if name:
        c._jit_name = name
    if preserve:
        c._preserve_runtime_structure = True
    return c


def epsilon_2d(u):
    """Small-strain tensor epsilon(u)."""

    half = _named_constant(0.5, "upl_strain_half", preserve=False)
    return half * (grad(u) + grad(u).T)


def effective_stress_linear_2d(u, material: UPlMaterial2D):
    """Linear elastic effective stress for the porous skeleton."""

    mu = _named_constant(material.mu, "mu_s")
    lam = _named_constant(material.lambda_, "lambda_s")
    eps_u = epsilon_2d(u)
    two = _named_constant(2.0, "upl_stress_two", preserve=False)
    identity = _named_constant(np.eye(2, dtype=float), "upl_identity_2d", dim=2, preserve=False)
    return two * mu * eps_u + lam * trace(eps_u) * identity


def hydraulic_conductivity_form_2d(p, q, material: UPlMaterial2D):
    """Return ``(K / mu_f grad p) . grad q`` for scalar pressure unknowns."""

    K = material.darcy_conductivity_matrix
    if abs(K[0, 1]) == 0.0 and abs(K[0, 0] - K[1, 1]) == 0.0:
        k_iso = _named_constant(float(K[0, 0]), "permeability_over_viscosity")
        return inner(k_iso * grad(p), grad(q))

    K_c = _named_constant(
        np.asarray(K, dtype=float),
        "permeability_over_viscosity",
        dim=2,
        preserve=False,
    )
    return dot(dot(K_c, grad(p)), grad(q))


@dataclass(frozen=True)
class UPlThetaSystem2D:
    """Linear one-step-theta U-Pl system forms."""

    lhs_form: object
    rhs_form: object
    stiffness_lhs: object
    coupling_lhs: object
    storage_lhs: object
    permeability_lhs: object
    stiffness_rhs: object
    coupling_rhs: object
    storage_rhs: object
    permeability_rhs: object


@dataclass(frozen=True)
class UPlKratosQuasistaticSystem2D:
    """Absolute linear system equivalent to Kratos quasi-static U-Pl at convergence."""

    lhs_form: object
    rhs_form: object
    stiffness_lhs: object
    coupling_lhs: object
    rate_coupling_lhs: object
    storage_lhs: object
    permeability_lhs: object
    rate_coupling_rhs: object
    storage_rhs: object


@dataclass(frozen=True)
class UPlKratosFICQuasistaticSystem2D:
    """Kratos quasi-static U-Pl system with 2D triangular FIC stabilization."""

    lhs_form: object
    rhs_form: object
    base_system: UPlKratosQuasistaticSystem2D
    pressure_gradient_lhs: object
    pressure_gradient_rhs: object


@dataclass(frozen=True)
class UPlKratosDamageQuasistaticSystem2D:
    """Kratos quasi-static U-Pl system with a frozen quadrature damage coefficient."""

    lhs_form: object
    rhs_form: object
    stiffness_lhs: object
    coupling_lhs: object
    rate_coupling_lhs: object
    storage_lhs: object
    permeability_lhs: object
    rate_coupling_rhs: object
    storage_rhs: object


def build_upl_theta_system_2d(
    *,
    u_trial,
    p_trial,
    u_test,
    p_test,
    u_prev,
    p_prev,
    material: UPlMaterial2D,
    dt,
    theta,
    dx_measure,
    body_acceleration=None,
) -> UPlThetaSystem2D:
    """Build the 2D linear small-strain U-Pl theta system.

    Parameters are symbolic pycutfem expressions. ``u_trial`` and ``p_trial``
    are the unknowns at the new time level; ``u_prev`` and ``p_prev`` are
    coefficient functions from the previous accepted time step.

    The returned forms are ready for ``Equation(lhs_form, rhs_form)`` or direct
    assembly with ``FormCompiler``. Boundary/interface work is intentionally
    separate so FSI/FPSI coupling can own its sign convention explicitly.
    """

    alpha = _named_constant(material.biot_coefficient, "biot_coef")
    invM = _named_constant(material.biot_modulus_inverse, "inv_biot_modulus")
    dt_c = _named_constant(dt, "dt")
    theta_c = _named_constant(theta, "theta")
    one = _named_constant(1.0, "upl_one", preserve=False)
    minus_one = _named_constant(-1.0, "upl_minus_one", preserve=False)

    sigma_trial = effective_stress_linear_2d(u_trial, material)
    sigma_prev = effective_stress_linear_2d(u_prev, material)
    H_trial = hydraulic_conductivity_form_2d(p_trial, p_test, material)
    H_prev = hydraulic_conductivity_form_2d(p_prev, p_test, material)

    stiffness_lhs = minus_one * inner(sigma_trial, epsilon_2d(u_test)) * dx_measure
    coupling_lhs = alpha * p_trial * div(u_test) * dx_measure + alpha * div(u_trial) * p_test * dx_measure
    storage_lhs = invM * p_trial * p_test * dx_measure
    permeability_lhs = theta_c * dt_c * H_trial * dx_measure

    stiffness_rhs = minus_one * inner(sigma_prev, epsilon_2d(u_test)) * dx_measure
    coupling_rhs = alpha * p_prev * div(u_test) * dx_measure + alpha * div(u_prev) * p_test * dx_measure
    storage_rhs = invM * p_prev * p_test * dx_measure
    permeability_rhs = (theta_c - one) * dt_c * H_prev * dx_measure

    rhs_form = stiffness_rhs + coupling_rhs + storage_rhs + permeability_rhs
    if body_acceleration is not None:
        rho = _named_constant(material.mixture_density, "mixture_density")
        rhs_form += inner(rho * body_acceleration, u_test) * dx_measure

    return UPlThetaSystem2D(
        lhs_form=stiffness_lhs + coupling_lhs + storage_lhs + permeability_lhs,
        rhs_form=rhs_form,
        stiffness_lhs=stiffness_lhs,
        coupling_lhs=coupling_lhs,
        storage_lhs=storage_lhs,
        permeability_lhs=permeability_lhs,
        stiffness_rhs=stiffness_rhs,
        coupling_rhs=coupling_rhs,
        storage_rhs=storage_rhs,
        permeability_rhs=permeability_rhs,
    )


def build_kratos_quasistatic_upl_system_2d(
    *,
    u_trial,
    p_trial,
    u_test,
    p_test,
    u_prev,
    p_prev,
    material: UPlMaterial2D,
    dt,
    theta_u,
    theta_p,
    dx_measure,
    velocity_prev=None,
    p_rate_prev=None,
    body_acceleration=None,
) -> UPlKratosQuasistaticSystem2D:
    """Build the 2D system matching Kratos' quasi-static U-Pl convergence equations.

    Kratos assembles a Newton residual with:

    - ``VELOCITY_COEFFICIENT = 1 / (theta_u dt)``
    - ``DT_LIQUID_PRESSURE_COEFFICIENT = 1 / (theta_p dt)``

    For ``theta < 1``, Kratos' GN11 derivative update also carries the previous
    velocity and pressure-rate values. Pass ``velocity_prev`` and
    ``p_rate_prev`` for exact transient parity; omitting them keeps the
    backward-compatible ``theta=1`` form.

    At convergence for the linear small-strain case, this is equivalent to the
    absolute system returned here. External face loads should be added to
    ``rhs_form`` with the same sign as Kratos' ``FACE_LOAD`` condition,
    i.e. ``+ int_Gamma traction . u_test``.
    """

    alpha = _named_constant(material.biot_coefficient, "biot_coef")
    invM = _named_constant(material.biot_modulus_inverse, "inv_biot_modulus")
    dt_c = _named_constant(dt, "dt")
    theta_u_c = _named_constant(theta_u, "theta_u")
    theta_p_c = _named_constant(theta_p, "theta_p")
    one = _named_constant(1.0, "upl_one", preserve=False)
    minus_one = _named_constant(-1.0, "upl_minus_one", preserve=False)

    velocity_coefficient = one / (theta_u_c * dt_c)
    dt_pressure_coefficient = one / (theta_p_c * dt_c)
    prev_velocity_factor = (one - theta_u_c) / theta_u_c
    prev_p_rate_factor = (one - theta_p_c) / theta_p_c

    stiffness_lhs = inner(effective_stress_linear_2d(u_trial, material), epsilon_2d(u_test)) * dx_measure
    coupling_lhs = minus_one * alpha * p_trial * div(u_test) * dx_measure
    rate_coupling_lhs = velocity_coefficient * alpha * div(u_trial) * p_test * dx_measure
    storage_lhs = dt_pressure_coefficient * invM * p_trial * p_test * dx_measure
    permeability_lhs = hydraulic_conductivity_form_2d(p_trial, p_test, material) * dx_measure

    rate_coupling_rhs = velocity_coefficient * alpha * div(u_prev) * p_test * dx_measure
    storage_rhs = dt_pressure_coefficient * invM * p_prev * p_test * dx_measure
    if velocity_prev is not None:
        rate_coupling_rhs += prev_velocity_factor * alpha * div(velocity_prev) * p_test * dx_measure
    if p_rate_prev is not None:
        storage_rhs += prev_p_rate_factor * invM * p_rate_prev * p_test * dx_measure
    rhs_form = rate_coupling_rhs + storage_rhs

    if body_acceleration is not None:
        body = _named_constant(body_acceleration, "body_acceleration", preserve=False)
        rho = _named_constant(material.mixture_density, "mixture_density")
        rho_l = _named_constant(material.liquid_density, "liquid_density")
        mu_l = _named_constant(material.dynamic_viscosity_liquid, "dynamic_viscosity_liquid")
        K_intrinsic = _named_constant(
            material.permeability_matrix,
            "intrinsic_permeability",
            dim=2,
            preserve=False,
        )
        rhs_form += inner(rho * body, u_test) * dx_measure
        rhs_form += dot(
            dot(K_intrinsic, grad(p_test)),
            (rho_l / mu_l) * body,
        ) * dx_measure

    return UPlKratosQuasistaticSystem2D(
        lhs_form=stiffness_lhs + coupling_lhs + rate_coupling_lhs + storage_lhs + permeability_lhs,
        rhs_form=rhs_form,
        stiffness_lhs=stiffness_lhs,
        coupling_lhs=coupling_lhs,
        rate_coupling_lhs=rate_coupling_lhs,
        storage_lhs=storage_lhs,
        permeability_lhs=permeability_lhs,
        rate_coupling_rhs=rate_coupling_rhs,
        storage_rhs=storage_rhs,
    )


def plane_stress_internal_work_2d(u, v, material: UPlMaterial2D, *, damage=None):
    """Return plane-stress ``sigma(u):epsilon(v)`` with optional scalar damage.

    ``damage`` is treated as a frozen coefficient. This is the correct
    Kratos-parity tangent for the Modified Mises nonlocal damage law, where the
    Poromechanics implementation intentionally uses the secant constitutive
    tensor in the nonlocal branch.
    """

    E = _named_constant(material.young_modulus, "plane_stress_young_modulus")
    nu = _named_constant(material.poisson_ratio, "plane_stress_poisson_ratio")
    thickness = _named_constant(material.thickness, "plane_stress_thickness")
    one = _named_constant(1.0, "plane_stress_one", preserve=False)
    half = _named_constant(0.5, "plane_stress_half", preserve=False)
    c = E / (one - nu * nu)
    shear = half * c * (one - nu)

    eps_xx = grad(u)[0, 0]
    eps_yy = grad(u)[1, 1]
    gamma_xy = grad(u)[0, 1] + grad(u)[1, 0]
    test_xx = grad(v)[0, 0]
    test_yy = grad(v)[1, 1]
    test_gamma_xy = grad(v)[0, 1] + grad(v)[1, 0]

    sigma_xx = c * (eps_xx + nu * eps_yy)
    sigma_yy = c * (nu * eps_xx + eps_yy)
    sigma_xy = shear * gamma_xy
    work = sigma_xx * test_xx + sigma_yy * test_yy + sigma_xy * test_gamma_xy
    if damage is None:
        return thickness * work
    return thickness * (one - damage) * work


def plane_strain_internal_work_2d(u, v, material: UPlMaterial2D, *, damage=None):
    """Return 2D plane-strain ``sigma(u):epsilon(v)`` with optional damage.

    Kratos' fluid-pumping fracture case uses
    ``ModifiedMisesNonlocalDamagePlaneStrain2DLaw`` for the bulk.  The damage
    law is updated outside UFL and the tangent consumes a frozen scalar damage
    coefficient, so this helper keeps the elastic operator generic and simply
    scales the plane-strain effective stress by ``(1-d)``.
    """

    thickness = _named_constant(material.thickness, "plane_strain_thickness")
    one = _named_constant(1.0, "plane_strain_one", preserve=False)
    work = inner(effective_stress_linear_2d(u, material), epsilon_2d(v))
    if damage is None:
        return thickness * work
    return thickness * (one - damage) * work


def build_kratos_quasistatic_damage_upl_system_2d(
    *,
    u_trial,
    p_trial,
    u_test,
    p_test,
    u_prev,
    p_prev,
    material: UPlMaterial2D,
    damage,
    dt,
    theta_u,
    theta_p,
    dx_measure,
    velocity_prev=None,
    p_rate_prev=None,
    body_acceleration=None,
) -> UPlKratosDamageQuasistaticSystem2D:
    """Build a 2D plane-stress U-Pl system with frozen nonlocal damage.

    This is intended for nonlocal-damage continuation solves. The constitutive
    state update is performed outside UFL and exposed as a quadrature-state
    coefficient through ``damage``.
    """

    alpha = _named_constant(material.biot_coefficient, "biot_coef")
    invM = _named_constant(material.biot_modulus_inverse, "inv_biot_modulus")
    dt_c = _named_constant(dt, "dt")
    theta_u_c = _named_constant(theta_u, "theta_u")
    theta_p_c = _named_constant(theta_p, "theta_p")
    one = _named_constant(1.0, "upl_one", preserve=False)
    minus_one = _named_constant(-1.0, "upl_minus_one", preserve=False)

    velocity_coefficient = one / (theta_u_c * dt_c)
    dt_pressure_coefficient = one / (theta_p_c * dt_c)
    prev_velocity_factor = (one - theta_u_c) / theta_u_c
    prev_p_rate_factor = (one - theta_p_c) / theta_p_c

    stiffness_lhs = plane_stress_internal_work_2d(u_trial, u_test, material, damage=damage) * dx_measure
    coupling_lhs = minus_one * alpha * p_trial * div(u_test) * dx_measure
    rate_coupling_lhs = velocity_coefficient * alpha * div(u_trial) * p_test * dx_measure
    storage_lhs = dt_pressure_coefficient * invM * p_trial * p_test * dx_measure
    permeability_lhs = hydraulic_conductivity_form_2d(p_trial, p_test, material) * dx_measure

    rate_coupling_rhs = velocity_coefficient * alpha * div(u_prev) * p_test * dx_measure
    storage_rhs = dt_pressure_coefficient * invM * p_prev * p_test * dx_measure
    if velocity_prev is not None:
        rate_coupling_rhs += prev_velocity_factor * alpha * div(velocity_prev) * p_test * dx_measure
    if p_rate_prev is not None:
        storage_rhs += prev_p_rate_factor * invM * p_rate_prev * p_test * dx_measure
    rhs_form = rate_coupling_rhs + storage_rhs

    if body_acceleration is not None:
        body = _named_constant(body_acceleration, "body_acceleration", preserve=False)
        rho = _named_constant(material.mixture_density, "mixture_density")
        rhs_form += inner(rho * body, u_test) * dx_measure

    return UPlKratosDamageQuasistaticSystem2D(
        lhs_form=stiffness_lhs + coupling_lhs + rate_coupling_lhs + storage_lhs + permeability_lhs,
        rhs_form=rhs_form,
        stiffness_lhs=stiffness_lhs,
        coupling_lhs=coupling_lhs,
        rate_coupling_lhs=rate_coupling_lhs,
        storage_lhs=storage_lhs,
        permeability_lhs=permeability_lhs,
        rate_coupling_rhs=rate_coupling_rhs,
        storage_rhs=storage_rhs,
    )


def build_kratos_fic_triangle_damage_upl_system_2d(
    *,
    u_trial,
    p_trial,
    u_test,
    p_test,
    u_prev,
    p_prev,
    material: UPlMaterial2D,
    damage,
    dt,
    theta_u,
    theta_p,
    dx_measure,
    element_length_squared,
    velocity_prev=None,
    p_rate_prev=None,
    body_acceleration=None,
) -> UPlKratosFICQuasistaticSystem2D:
    """Build Kratos' triangular FIC U-Pl bulk form with frozen bulk damage.

    This is the damage-aware counterpart of
    :func:`build_kratos_fic_triangle_upl_system_2d`.  It preserves the FIC
    pressure-gradient stabilization and replaces only the elastic skeleton
    stiffness by the plane-strain damaged secant stiffness used by the
    fluid-pumping fracture body law.
    """

    alpha = _named_constant(material.biot_coefficient, "biot_coef")
    invM = _named_constant(material.biot_modulus_inverse, "inv_biot_modulus")
    dt_c = _named_constant(dt, "dt")
    theta_u_c = _named_constant(theta_u, "theta_u")
    theta_p_c = _named_constant(theta_p, "theta_p")
    one = _named_constant(1.0, "upl_one", preserve=False)
    two = _named_constant(2.0, "fic_two", preserve=False)
    three = _named_constant(3.0, "fic_three", preserve=False)
    eight = _named_constant(8.0, "fic_eight", preserve=False)
    minus_one = _named_constant(-1.0, "upl_minus_one", preserve=False)

    velocity_coefficient = one / (theta_u_c * dt_c)
    dt_pressure_coefficient = one / (theta_p_c * dt_c)
    prev_velocity_factor = (one - theta_u_c) / theta_u_c
    prev_p_rate_factor = (one - theta_p_c) / theta_p_c

    stiffness_lhs = plane_strain_internal_work_2d(u_trial, u_test, material, damage=damage) * dx_measure
    coupling_lhs = minus_one * alpha * p_trial * div(u_test) * dx_measure
    rate_coupling_lhs = velocity_coefficient * alpha * div(u_trial) * p_test * dx_measure
    storage_lhs = dt_pressure_coefficient * invM * p_trial * p_test * dx_measure
    permeability_lhs = hydraulic_conductivity_form_2d(p_trial, p_test, material) * dx_measure

    rate_coupling_rhs = velocity_coefficient * alpha * div(u_prev) * p_test * dx_measure
    storage_rhs = dt_pressure_coefficient * invM * p_prev * p_test * dx_measure
    if velocity_prev is not None:
        rate_coupling_rhs += prev_velocity_factor * alpha * div(velocity_prev) * p_test * dx_measure
    if p_rate_prev is not None:
        storage_rhs += prev_p_rate_factor * invM * p_rate_prev * p_test * dx_measure

    rhs_form = rate_coupling_rhs + storage_rhs
    if body_acceleration is not None:
        body = _named_constant(body_acceleration, "body_acceleration", preserve=False)
        rho = _named_constant(material.mixture_density, "mixture_density")
        rho_l = _named_constant(material.liquid_density, "liquid_density")
        mu_l = _named_constant(material.dynamic_viscosity_liquid, "dynamic_viscosity_liquid")
        K_intrinsic = _named_constant(
            material.permeability_matrix,
            "intrinsic_permeability",
            dim=2,
            preserve=False,
        )
        rhs_form += inner(rho * body, u_test) * dx_measure
        rhs_form += dot(
            dot(K_intrinsic, grad(p_test)),
            (rho_l / mu_l) * body,
        ) * dx_measure

    base = UPlKratosQuasistaticSystem2D(
        lhs_form=stiffness_lhs + coupling_lhs + rate_coupling_lhs + storage_lhs + permeability_lhs,
        rhs_form=rhs_form,
        stiffness_lhs=stiffness_lhs,
        coupling_lhs=coupling_lhs,
        rate_coupling_lhs=rate_coupling_lhs,
        storage_lhs=storage_lhs,
        permeability_lhs=permeability_lhs,
        rate_coupling_rhs=rate_coupling_rhs,
        storage_rhs=storage_rhs,
    )

    mu = _named_constant(material.mu, "fic_shear_modulus")
    h2 = _as_elementwise_or_constant(element_length_squared, name="fic_element_length_squared")
    tau_pressure = (h2 * alpha / (eight * mu)) * (
        alpha - (two * mu * invM) / (three * alpha)
    )
    pressure_gradient_lhs = (
        dt_pressure_coefficient * tau_pressure * inner(grad(p_trial), grad(p_test)) * dx_measure
    )
    pressure_gradient_rhs = (
        dt_pressure_coefficient * tau_pressure * inner(grad(p_prev), grad(p_test)) * dx_measure
    )
    if p_rate_prev is not None:
        pressure_gradient_rhs += prev_p_rate_factor * tau_pressure * inner(grad(p_rate_prev), grad(p_test)) * dx_measure

    return UPlKratosFICQuasistaticSystem2D(
        lhs_form=base.lhs_form + pressure_gradient_lhs,
        rhs_form=base.rhs_form + pressure_gradient_rhs,
        base_system=base,
        pressure_gradient_lhs=pressure_gradient_lhs,
        pressure_gradient_rhs=pressure_gradient_rhs,
    )


def kratos_fic_triangle_element_length_squared(mesh) -> np.ndarray:
    """Return Kratos' 2D triangular FIC ``ElementLength^2 = 4A/pi`` per cell."""

    if getattr(mesh, "element_type", None) != "tri":
        raise ValueError("Kratos 2D FIC triangular element length is only defined for triangular meshes.")
    areas = np.asarray(getattr(mesh, "areas_list", None), dtype=float)
    if areas.ndim != 1 or areas.size != int(getattr(mesh, "n_elements", areas.size)):
        areas = np.asarray(mesh.areas(), dtype=float)
    if np.any(areas <= 0.0):
        raise ValueError("FIC element lengths require strictly positive triangle areas.")
    return 4.0 * areas / np.pi


def _as_elementwise_or_constant(value, *, name: str | None = None):
    if isinstance(value, (Constant, ElementWiseConstant)):
        return _named_constant(value, name)
    if hasattr(value, "__dict__"):
        return value
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return _named_constant(float(arr), name)
    ewc = ElementWiseConstant(arr)
    if name:
        ewc._jit_name = name
    return ewc


def build_kratos_fic_triangle_upl_system_2d(
    *,
    u_trial,
    p_trial,
    u_test,
    p_test,
    u_prev,
    p_prev,
    material: UPlMaterial2D,
    dt,
    theta_u,
    theta_p,
    dx_measure,
    element_length_squared,
    velocity_prev=None,
    p_rate_prev=None,
    body_acceleration=None,
) -> UPlKratosFICQuasistaticSystem2D:
    """Build Kratos' 2D3 U-Pl FIC bulk form on top of the quasi-static system.

    For ``UPlSmallStrainFICElement<2,3>`` Kratos disables the strain-gradient
    term; with a linear elastic constant material the remaining FIC contribution
    is the pressure-gradient stabilization:

    ``tau_p grad(p) . grad(q)``, where
    ``tau_p = h^2 alpha/(8 mu) * (alpha - 2 mu invM/(3 alpha))``.
    """

    base = build_kratos_quasistatic_upl_system_2d(
        u_trial=u_trial,
        p_trial=p_trial,
        u_test=u_test,
        p_test=p_test,
        u_prev=u_prev,
        p_prev=p_prev,
        material=material,
        dt=dt,
        theta_u=theta_u,
        theta_p=theta_p,
        dx_measure=dx_measure,
        velocity_prev=velocity_prev,
        p_rate_prev=p_rate_prev,
        body_acceleration=body_acceleration,
    )

    alpha = _named_constant(material.biot_coefficient, "fic_biot_coef")
    invM = _named_constant(material.biot_modulus_inverse, "fic_inv_biot_modulus")
    mu = _named_constant(material.mu, "fic_shear_modulus")
    h2 = _as_elementwise_or_constant(element_length_squared, name="fic_element_length_squared")
    dt_c = _named_constant(dt, "fic_dt")
    theta_p_c = _named_constant(theta_p, "fic_theta_p")
    one = _named_constant(1.0, "fic_one", preserve=False)
    two = _named_constant(2.0, "fic_two", preserve=False)
    three = _named_constant(3.0, "fic_three", preserve=False)
    eight = _named_constant(8.0, "fic_eight", preserve=False)
    dt_pressure_coefficient = one / (theta_p_c * dt_c)
    prev_p_rate_factor = (one - theta_p_c) / theta_p_c

    tau_pressure = (h2 * alpha / (eight * mu)) * (
        alpha - (two * mu * invM) / (three * alpha)
    )

    pressure_gradient_lhs = (
        dt_pressure_coefficient * tau_pressure * inner(grad(p_trial), grad(p_test)) * dx_measure
    )
    pressure_gradient_rhs = (
        dt_pressure_coefficient * tau_pressure * inner(grad(p_prev), grad(p_test)) * dx_measure
    )
    if p_rate_prev is not None:
        pressure_gradient_rhs += prev_p_rate_factor * tau_pressure * inner(grad(p_rate_prev), grad(p_test)) * dx_measure

    return UPlKratosFICQuasistaticSystem2D(
        lhs_form=base.lhs_form + pressure_gradient_lhs,
        rhs_form=base.rhs_form + pressure_gradient_rhs,
        base_system=base,
        pressure_gradient_lhs=pressure_gradient_lhs,
        pressure_gradient_rhs=pressure_gradient_rhs,
    )


def normal_liquid_flux_rhs_2d(p_test, normal_flux, boundary_measure, *, scale=None):
    """Kratos standard/interface normal-liquid-flux pressure RHS.

    Kratos stores the nodal variable with the convention "positive value =
    inlet"; after the sign flip inside ``UPlNormalLiquidFluxCondition`` and
    ``UPlNormalLiquidFluxInterfaceCondition`` the assembled pressure RHS is
    ``+ int_Gamma normal_flux q``.
    """

    flux = _named_constant(normal_flux, "normal_liquid_flux")
    if scale is None:
        scale_expr = _named_constant(1.0, "normal_liquid_flux_scale", preserve=False)
    else:
        scale_expr = _named_constant(scale, "normal_liquid_flux_scale")
    return scale_expr * flux * p_test * boundary_measure


def displacement_neumann_rhs(u_test, traction, boundary_measure, *, scale=None):
    """Boundary work helper matching the consolidation pressure-load sign."""

    if scale is None:
        scale = _named_constant(1.0, "displacement_neumann_scale", preserve=False)
    else:
        scale = _named_constant(scale, "displacement_neumann_scale")
    minus_one = _named_constant(-1.0, "displacement_neumann_minus_one", preserve=False)
    return minus_one * scale * dot(traction, u_test) * boundary_measure
