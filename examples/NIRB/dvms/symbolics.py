from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycutfem.ufl.expressions import (
    Constant,
    Expression,
    Function,
    Identity,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    cof,
    det,
    dot,
    grad,
    inner,
    inv,
)


def _as_expression(value) -> Expression:
    if isinstance(value, Expression):
        return value
    return Constant(value)


def _matvec2_components(A, b) -> tuple[Expression, Expression]:
    return (
        A[0, 0] * b[0] + A[0, 1] * b[1],
        A[1, 0] * b[0] + A[1, 1] * b[1],
    )


@dataclass(frozen=True)
class FluidDVMSKinematics:
    F: Expression
    Finv: Expression
    J: Expression
    cof_F: Expression
    grad_u_phys: Expression
    div_u_phys: Expression
    w_mesh: Expression
    resolved_conv_velocity: Expression


@dataclass(frozen=True)
class FluidDVMSPredictorSymbolics:
    kinematics: FluidDVMSKinematics
    grad_p_phys: Expression
    a_curr: Expression
    a_relaxed: Expression
    static_residual: Expression


@dataclass(frozen=True)
class FluidDVMSPredictorIterationSymbolics:
    predictor: FluidDVMSPredictorSymbolics
    conv_velocity: Expression
    conv_speed: Expression
    inv_tau: Expression
    linearization_matrix: Expression
    fixed_point_residual: Expression


@dataclass(frozen=True)
class FluidDVMSLocalForms:
    add_velocity_residual: object
    add_velocity_jacobian: object
    mass_lhs: object
    mass_stabilization: object
    system_residual: object
    system_jacobian: object
    system_condensed_residual: object
    system_condensed_jacobian: object


@dataclass(frozen=True)
class FluidDVMSKratosSplitForms:
    lhs_terms: dict[str, object]
    rhs_terms: dict[str, object]
    mass_terms: dict[str, object]


def build_fluid_cauchy_stress(
    *,
    p,
    grad_u_phys,
    div_u_phys,
    mu,
) -> Expression:
    identity = Identity(2)
    mu_expr = _as_expression(mu)
    viscous = mu_expr * (grad_u_phys + grad_u_phys.T - Constant(2.0 / 3.0) * div_u_phys * identity)
    return -p * identity + viscous


def build_fluid_viscous_cauchy_stress(
    *,
    grad_u_phys,
    div_u_phys,
    mu,
) -> Expression:
    identity = Identity(2)
    mu_expr = _as_expression(mu)
    return mu_expr * (grad_u_phys + grad_u_phys.T - Constant(2.0 / 3.0) * div_u_phys * identity)


def build_fluid_dvms_kinematics(
    *,
    u: VectorFunction,
    d: VectorFunction,
    d_prev: VectorFunction,
    d_prev2: VectorFunction | None = None,
    mesh_v_prev: VectorFunction | None = None,
    mesh_a_prev: VectorFunction | None = None,
    bossak_alpha=None,
    dt,
) -> FluidDVMSKinematics:
    dt_expr = _as_expression(dt)
    F = Identity(2) + grad(d)
    J = det(F)
    cof_F = cof(F)
    Finv = cof_F.T / J
    grad_u_phys = dot(grad(u), Finv)
    div_u_phys = inner(cof_F, grad(u)) / J
    if mesh_v_prev is not None and mesh_a_prev is not None and bossak_alpha is not None:
        alpha_expr = _as_expression(bossak_alpha)
        gamma_expr = Constant(0.5) - alpha_expr
        beta_expr = Constant(0.25) * (Constant(1.0) - alpha_expr) * (Constant(1.0) - alpha_expr)
        a_mesh = (
            d
            - d_prev
            - dt_expr * mesh_v_prev
            - dt_expr * dt_expr * (Constant(0.5) - beta_expr) * mesh_a_prev
        ) / (beta_expr * dt_expr * dt_expr)
        w_mesh = mesh_v_prev + dt_expr * ((Constant(1.0) - gamma_expr) * mesh_a_prev + gamma_expr * a_mesh)
    elif d_prev2 is None:
        w_mesh = (d - d_prev) / dt_expr
    else:
        w_mesh = (Constant(1.5) * d - Constant(2.0) * d_prev + Constant(0.5) * d_prev2) / dt_expr
    resolved_conv_velocity = u - w_mesh
    return FluidDVMSKinematics(
        F=F,
        Finv=Finv,
        J=J,
        cof_F=cof_F,
        grad_u_phys=grad_u_phys,
        div_u_phys=div_u_phys,
        w_mesh=w_mesh,
        resolved_conv_velocity=resolved_conv_velocity,
    )


def build_fluid_dvms_old_mass_residual(
    *,
    u_prev: VectorFunction,
    d_prev: VectorFunction,
) -> Expression:
    F_old = Identity(2) + grad(d_prev)
    return -(inner(cof(F_old), grad(u_prev)) / det(F_old))


def build_fluid_dvms_predictor_symbolics(
    *,
    u_k: VectorFunction,
    u_prev: VectorFunction,
    a_prev: VectorFunction,
    p_k: Function,
    d_mesh: VectorFunction,
    d_prev: VectorFunction,
    d_prev2: VectorFunction | None = None,
    mesh_v_prev: VectorFunction | None = None,
    mesh_a_prev: VectorFunction | None = None,
    dt,
    bossak_ma0,
    bossak_ma2,
    bossak_alpha,
    rho,
    old_subscale,
    momentum_projection,
) -> FluidDVMSPredictorSymbolics:
    kin = build_fluid_dvms_kinematics(
        u=u_k,
        d=d_mesh,
        d_prev=d_prev,
        d_prev2=d_prev2,
        mesh_v_prev=mesh_v_prev,
        mesh_a_prev=mesh_a_prev,
        bossak_alpha=bossak_alpha,
        dt=dt,
    )
    bossak_ma0_expr = _as_expression(bossak_ma0)
    bossak_ma2_expr = _as_expression(bossak_ma2)
    bossak_alpha_expr = _as_expression(bossak_alpha)
    rho_expr = _as_expression(rho)
    dt_expr = _as_expression(dt)

    a_curr = bossak_ma0_expr * (u_k - u_prev) + bossak_ma2_expr * a_prev
    a_relaxed = (Constant(1.0) - bossak_alpha_expr) * a_curr + bossak_alpha_expr * a_prev
    grad_p_phys = dot(kin.Finv.T, grad(p_k))
    # The predictor update uses the Bossak-relaxed acceleration field in the
    # static momentum residual together with the old-subscale time-history term.
    static_residual = -(
        rho_expr * a_relaxed
        + rho_expr * dot(kin.grad_u_phys, kin.resolved_conv_velocity)
        + grad_p_phys
        + momentum_projection
    ) + (rho_expr / dt_expr) * old_subscale

    return FluidDVMSPredictorSymbolics(
        kinematics=kin,
        grad_p_phys=grad_p_phys,
        a_curr=a_curr,
        a_relaxed=a_relaxed,
        static_residual=static_residual,
    )


def build_fluid_dvms_predictor_iteration_symbolics(
    *,
    u_k: VectorFunction,
    u_prev: VectorFunction,
    a_prev: VectorFunction,
    p_k: Function,
    d_mesh: VectorFunction,
    d_prev: VectorFunction,
    d_prev2: VectorFunction | None = None,
    mesh_v_prev: VectorFunction | None = None,
    mesh_a_prev: VectorFunction | None = None,
    dt,
    bossak_ma0,
    bossak_ma2,
    bossak_alpha,
    rho,
    mu,
    h,
    predicted_subscale,
    old_subscale,
    momentum_projection,
) -> FluidDVMSPredictorIterationSymbolics:
    predictor = build_fluid_dvms_predictor_symbolics(
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        d_prev2=d_prev2,
        mesh_v_prev=mesh_v_prev,
        mesh_a_prev=mesh_a_prev,
        dt=dt,
        bossak_ma0=bossak_ma0,
        bossak_ma2=bossak_ma2,
        bossak_alpha=bossak_alpha,
        rho=rho,
        old_subscale=old_subscale,
        momentum_projection=momentum_projection,
    )
    rho_expr = _as_expression(rho)
    mu_expr = _as_expression(mu)
    dt_expr = _as_expression(dt)
    h_expr = _as_expression(h)
    conv_velocity = predictor.kinematics.resolved_conv_velocity + predicted_subscale
    conv_speed = (dot(conv_velocity, conv_velocity)) ** Constant(0.5)
    inv_tau = (
        Constant(8.0) * mu_expr / (h_expr * h_expr)
        + rho_expr * (Constant(1.0) / dt_expr + Constant(2.0) * conv_speed / h_expr)
    )
    linearization_matrix = rho_expr * predictor.kinematics.grad_u_phys + inv_tau * Identity(2)
    fixed_point_residual = predictor.static_residual - dot(linearization_matrix, predicted_subscale)
    return FluidDVMSPredictorIterationSymbolics(
        predictor=predictor,
        conv_velocity=conv_velocity,
        conv_speed=conv_speed,
        inv_tau=inv_tau,
        linearization_matrix=linearization_matrix,
        fixed_point_residual=fixed_point_residual,
    )


def build_fluid_dvms_local_forms(
    *,
    du: VectorTrialFunction,
    dp: TrialFunction,
    v: VectorTestFunction,
    q: TestFunction,
    u_k: VectorFunction,
    u_prev: VectorFunction,
    a_prev: VectorFunction,
    p_k: Function,
    d_mesh: VectorFunction,
    d_prev: VectorFunction,
    d_prev2: VectorFunction | None = None,
    mesh_v_prev: VectorFunction | None = None,
    mesh_a_prev: VectorFunction | None = None,
    dx_measure,
    rho,
    mu,
    dt,
    h,
    bossak_ma0,
    bossak_ma2,
    bossak_alpha,
    predicted_subscale,
    old_subscale,
    momentum_projection,
    mass_projection,
    old_mass_residual,
    body_force=None,
    use_oss: bool = False,
) -> FluidDVMSLocalForms:
    rho_expr = _as_expression(rho)
    mu_expr = _as_expression(mu)
    dt_expr = _as_expression(dt)
    h_expr = _as_expression(h)
    bossak_ma0_expr = _as_expression(bossak_ma0)
    bossak_ma2_expr = _as_expression(bossak_ma2)
    bossak_alpha_expr = _as_expression(bossak_alpha)
    body_expr = Constant(np.zeros((2,), dtype=float), dim=1) if body_force is None else _as_expression(body_force)

    predictor = build_fluid_dvms_predictor_symbolics(
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        d_prev2=d_prev2,
        mesh_v_prev=mesh_v_prev,
        mesh_a_prev=mesh_a_prev,
        dt=dt_expr,
        bossak_ma0=bossak_ma0_expr,
        bossak_ma2=bossak_ma2_expr,
        bossak_alpha=bossak_alpha_expr,
        rho=rho_expr,
        old_subscale=old_subscale,
        momentum_projection=momentum_projection,
    )
    kin = predictor.kinematics
    conv_velocity = kin.resolved_conv_velocity + predicted_subscale
    conv_speed = (dot(conv_velocity, conv_velocity)) ** Constant(0.5)
    inv_dt = Constant(1.0) / dt_expr
    tau_one = Constant(1.0) / (
        Constant(8.0) * mu_expr / (h_expr * h_expr)
        + rho_expr * (inv_dt + Constant(2.0) * conv_speed / h_expr)
    )
    tau_two = mu_expr + rho_expr * conv_speed * h_expr / Constant(4.0)
    tau_p = rho_expr * h_expr * h_expr * inv_dt / Constant(8.0)

    grad_du_phys = dot(grad(du), kin.Finv)
    div_du_phys = inner(kin.cof_F, grad(du)) / kin.J
    div_v_phys = inner(kin.cof_F, grad(v)) / kin.J
    grad_q_phys = dot(kin.Finv.T, grad(q))
    grad_dp_phys = dot(kin.Finv.T, grad(dp))
    sigma = build_fluid_cauchy_stress(
        p=p_k,
        grad_u_phys=kin.grad_u_phys,
        div_u_phys=kin.div_u_phys,
        mu=mu_expr,
    )
    sigma_du = build_fluid_cauchy_stress(
        p=dp,
        grad_u_phys=grad_du_phys,
        div_u_phys=div_du_phys,
        mu=mu_expr,
    )
    viscous_sigma = build_fluid_viscous_cauchy_stress(
        grad_u_phys=kin.grad_u_phys,
        div_u_phys=kin.div_u_phys,
        mu=mu_expr,
    )
    viscous_sigma_du = build_fluid_viscous_cauchy_stress(
        grad_u_phys=grad_du_phys,
        div_u_phys=div_du_phys,
        mu=mu_expr,
    )

    old_uss_term = rho_expr * inv_dt * old_subscale
    add_velocity_source = body_expr - momentum_projection + old_uss_term

    tau_test_conv = rho_expr * dot(dot(grad(v), kin.Finv), conv_velocity)
    tau_test_pres = grad_q_phys
    tau_test_velocity = tau_test_conv - rho_expr * inv_dt * v + tau_test_pres
    tau_test_mass = tau_test_conv - inv_dt * v + tau_test_pres
    tau_test_source_v = tau_test_conv - rho_expr * inv_dt * v

    tau_res_static_conv = rho_expr * dot(kin.grad_u_phys, conv_velocity)
    tau_res_static_pres = predictor.grad_p_phys
    tau_dres_static_conv_1 = rho_expr * dot(grad_du_phys, conv_velocity)
    tau_dres_static_pres = grad_dp_phys

    add_velocity_residual = (
        kin.J * dot(body_expr + old_uss_term, v) * dx_measure
        + kin.J * tau_one * dot(tau_test_source_v, add_velocity_source) * dx_measure
        + kin.J * tau_one * dot(grad_q_phys, add_velocity_source) * dx_measure
        - kin.J * ((tau_two + tau_p) * mass_projection * div_v_phys) * dx_measure
        - kin.J * (tau_p * old_mass_residual * div_v_phys) * dx_measure
        - kin.J * rho_expr * dot(dot(kin.grad_u_phys, conv_velocity), v) * dx_measure
        - kin.J * tau_one * dot(tau_test_source_v, tau_res_static_conv) * dx_measure
        - kin.J * tau_one * dot(tau_test_conv, tau_res_static_pres) * dx_measure
        + kin.J * tau_one * dot(rho_expr * inv_dt * v, tau_res_static_pres) * dx_measure
        + inner(kin.cof_F, grad(v)) * p_k * dx_measure
        - kin.J * tau_one * dot(grad_q_phys, tau_res_static_conv) * dx_measure
        - inner(kin.cof_F, grad(u_k)) * q * dx_measure
        - kin.J * tau_one * dot(grad_q_phys, tau_res_static_pres) * dx_measure
        - kin.J * ((tau_two + tau_p) * kin.div_u_phys * div_v_phys) * dx_measure
        - inner(kin.J * dot(viscous_sigma, kin.Finv.T), grad(v)) * dx_measure
    )
    add_velocity_jacobian = (
        kin.J * rho_expr * dot(dot(grad_du_phys, conv_velocity), v) * dx_measure
        + kin.J * tau_one * dot(tau_test_source_v, tau_dres_static_conv_1) * dx_measure
        + kin.J * tau_one * dot(tau_test_conv, tau_dres_static_pres) * dx_measure
        - kin.J * tau_one * dot(rho_expr * inv_dt * v, tau_dres_static_pres) * dx_measure
        - inner(kin.cof_F, grad(v)) * dp * dx_measure
        + kin.J * tau_one * dot(grad_q_phys, tau_dres_static_conv_1) * dx_measure
        + inner(kin.cof_F, grad(du)) * q * dx_measure
        + kin.J * tau_one * dot(grad_q_phys, grad_dp_phys) * dx_measure
        + kin.J * ((tau_two + tau_p) * div_du_phys * div_v_phys) * dx_measure
        + inner(kin.J * dot(viscous_sigma_du, kin.Finv.T), grad(v)) * dx_measure
    )

    bossak_mam = (Constant(1.0) - bossak_alpha_expr) * bossak_ma0_expr
    mass_lhs = kin.J * rho_expr * dot(du, v) * dx_measure
    if bool(use_oss):
        mass_stabilization = Constant(0.0) * dot(du, v) * dx_measure
        mass_stabilization_action = Constant(0.0) * dot(body_expr, v) * dx_measure
        system_mass_stabilization = Constant(0.0) * dot(du, v) * dx_measure
    else:
        mass_stabilization = kin.J * tau_one * dot(tau_test_mass, rho_expr * du) * dx_measure
        mass_stabilization_action = kin.J * tau_one * dot(tau_test_mass, rho_expr * predictor.a_relaxed) * dx_measure
        system_mass_stabilization = kin.J * bossak_mam * tau_one * dot(tau_test_mass, rho_expr * du) * dx_measure

    system_mass_lhs = kin.J * bossak_mam * rho_expr * dot(du, v) * dx_measure
    # Kratos' Bossak scheme combines the exact local DVMS blocks as:
    #   LHS = D + mam * M
    #   RHS = velocity_rhs - M * a_relaxed
    # where D is CalculateLocalVelocityContribution and M is the full mass
    # matrix returned by CalculateMassMatrix (including AddMassStabilization
    # when OSS is disabled). The local symbolic "system" form must therefore
    # subtract the mass action from the RHS rather than add it.
    system_residual = (
        add_velocity_residual
        - kin.J * rho_expr * dot(predictor.a_relaxed, v) * dx_measure
        - mass_stabilization_action
    )
    system_jacobian = add_velocity_jacobian + system_mass_lhs + system_mass_stabilization

    # Hidden-state condensation uses the locally eliminated predicted subscale
    # equation G(U, z)=0. The exact local Schur complement is evaluated with the
    # current K_zz = dG/dz and the fixed-point residual r_z = G(U, z).
    #
    # The correction below matches the previously verified split local algebra:
    #   r_hat = r_u + R_z K_zz^{-1} r_z
    #   K_hat = K_uu + R_z K_zz^{-1} G_U
    #
    # This keeps the current state-updated tangent semantics, but fuses the
    # correction into a single symbolic FE assembly path so cpp/jit only
    # traverse the mesh once per residual/Jacobian evaluation.
    Kzz = rho_expr * kin.grad_u_phys + (Constant(1.0) / tau_one) * Identity(2)
    Kzz_hidden0, Kzz_hidden1 = _matvec2_components(Kzz, predicted_subscale)
    hidden_residual_0 = predictor.static_residual[0] - Kzz_hidden0
    hidden_residual_1 = predictor.static_residual[1] - Kzz_hidden1

    grad_du_conv_0, grad_du_conv_1 = _matvec2_components(grad_du_phys, conv_velocity)
    grad_u_du_0, grad_u_du_1 = _matvec2_components(kin.grad_u_phys, du)
    hidden_trial_0 = -(
        rho_expr * bossak_mam * du[0]
        + rho_expr * grad_du_conv_0
        + rho_expr * grad_u_du_0
        + grad_dp_phys[0]
    )
    hidden_trial_1 = -(
        rho_expr * bossak_mam * du[1]
        + rho_expr * grad_du_conv_1
        + rho_expr * grad_u_du_1
        + grad_dp_phys[1]
    )

    def _rz_action(hidden0, hidden1):
        grad_v_phys = dot(grad(v), kin.Finv)
        tau_test_hidden_0 = rho_expr * (grad_v_phys[0, 0] * hidden0 + grad_v_phys[0, 1] * hidden1)
        tau_test_hidden_1 = rho_expr * (grad_v_phys[1, 0] * hidden0 + grad_v_phys[1, 1] * hidden1)
        tau_res_hidden_0 = rho_expr * (kin.grad_u_phys[0, 0] * hidden0 + kin.grad_u_phys[0, 1] * hidden1)
        tau_res_hidden_1 = rho_expr * (kin.grad_u_phys[1, 0] * hidden0 + kin.grad_u_phys[1, 1] * hidden1)
        tau_test_hidden_dot_source = tau_test_hidden_0 * add_velocity_source[0] + tau_test_hidden_1 * add_velocity_source[1]
        adv_hidden_dot_v = tau_res_hidden_0 * v[0] + tau_res_hidden_1 * v[1]
        tau_test_hidden_dot_adv = tau_test_hidden_0 * tau_res_static_conv[0] + tau_test_hidden_1 * tau_res_static_conv[1]
        tau_test_source_dot_hidden = tau_test_source_v[0] * tau_res_hidden_0 + tau_test_source_v[1] * tau_res_hidden_1
        tau_test_hidden_dot_pressure = tau_test_hidden_0 * tau_res_static_pres[0] + tau_test_hidden_1 * tau_res_static_pres[1]
        tau_test_hidden_dot_mass = tau_test_hidden_0 * (rho_expr * predictor.a_relaxed[0]) + tau_test_hidden_1 * (rho_expr * predictor.a_relaxed[1])
        grad_q_dot_hidden = grad_q_phys[0] * tau_res_hidden_0 + grad_q_phys[1] * tau_res_hidden_1
        return (
            kin.J * tau_one * tau_test_hidden_dot_source * dx_measure
            - kin.J * rho_expr * adv_hidden_dot_v * dx_measure
            - kin.J * tau_one * tau_test_hidden_dot_adv * dx_measure
            - kin.J * tau_one * tau_test_source_dot_hidden * dx_measure
            - kin.J * tau_one * tau_test_hidden_dot_pressure * dx_measure
            - kin.J * tau_one * tau_test_hidden_dot_mass * dx_measure
            - kin.J * tau_one * grad_q_dot_hidden * dx_measure
        )

    invKzz = inv(Kzz)
    hidden_state_residual_0 = invKzz[0, 0] * hidden_residual_0 + invKzz[0, 1] * hidden_residual_1
    hidden_state_residual_1 = invKzz[1, 0] * hidden_residual_0 + invKzz[1, 1] * hidden_residual_1
    hidden_state_trial_0 = invKzz[0, 0] * hidden_trial_0 + invKzz[0, 1] * hidden_trial_1
    hidden_state_trial_1 = invKzz[1, 0] * hidden_trial_0 + invKzz[1, 1] * hidden_trial_1
    system_condensed_residual = system_residual + _rz_action(hidden_state_residual_0, hidden_state_residual_1)
    system_condensed_jacobian = system_jacobian + _rz_action(hidden_state_trial_0, hidden_state_trial_1)

    return FluidDVMSLocalForms(
        add_velocity_residual=add_velocity_residual,
        add_velocity_jacobian=add_velocity_jacobian,
        mass_lhs=mass_lhs,
        mass_stabilization=mass_stabilization,
        system_residual=system_residual,
        system_jacobian=system_jacobian,
        system_condensed_residual=system_condensed_residual,
        system_condensed_jacobian=system_condensed_jacobian,
    )


def build_fluid_dvms_kratos_split_forms(
    *,
    du: VectorTrialFunction,
    dp: TrialFunction,
    v: VectorTestFunction,
    q: TestFunction,
    u_k: VectorFunction,
    u_prev: VectorFunction,
    a_prev: VectorFunction,
    p_k: Function,
    d_mesh: VectorFunction,
    d_prev: VectorFunction,
    d_prev2: VectorFunction | None = None,
    mesh_v_prev: VectorFunction | None = None,
    mesh_a_prev: VectorFunction | None = None,
    dx_measure,
    rho,
    mu,
    dt,
    h,
    bossak_ma0,
    bossak_ma2,
    bossak_alpha,
    predicted_subscale,
    old_subscale,
    momentum_projection,
    mass_projection,
    old_mass_residual,
    body_force=None,
    use_oss: bool = False,
) -> FluidDVMSKratosSplitForms:
    rho_expr = _as_expression(rho)
    mu_expr = _as_expression(mu)
    dt_expr = _as_expression(dt)
    h_expr = _as_expression(h)
    bossak_ma0_expr = _as_expression(bossak_ma0)
    bossak_ma2_expr = _as_expression(bossak_ma2)
    bossak_alpha_expr = _as_expression(bossak_alpha)
    body_expr = Constant(np.zeros((2,), dtype=float), dim=1) if body_force is None else _as_expression(body_force)
    predictor = build_fluid_dvms_predictor_symbolics(
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        d_prev2=d_prev2,
        dt=dt_expr,
        bossak_ma0=bossak_ma0_expr,
        bossak_ma2=bossak_ma2_expr,
        bossak_alpha=bossak_alpha_expr,
        rho=rho_expr,
        old_subscale=old_subscale,
        momentum_projection=momentum_projection,
    )
    kin = predictor.kinematics
    conv_velocity = kin.resolved_conv_velocity + predicted_subscale
    conv_speed = (dot(conv_velocity, conv_velocity)) ** Constant(0.5)
    inv_dt = Constant(1.0) / dt_expr
    tau_one = Constant(1.0) / (
        Constant(8.0) * mu_expr / (h_expr * h_expr)
        + rho_expr * (inv_dt + Constant(2.0) * conv_speed / h_expr)
    )
    tau_two = mu_expr + rho_expr * conv_speed * h_expr / Constant(4.0)
    tau_p = rho_expr * h_expr * h_expr * inv_dt / Constant(8.0)
    grad_du_phys = dot(grad(du), kin.Finv)
    div_du_phys = inner(kin.cof_F, grad(du)) / kin.J
    div_v_phys = inner(kin.cof_F, grad(v)) / kin.J
    grad_q_phys = dot(kin.Finv.T, grad(q))
    grad_dp_phys = dot(kin.Finv.T, grad(dp))
    grad_p_phys = predictor.grad_p_phys
    tau_res_static_conv = rho_expr * dot(kin.grad_u_phys, conv_velocity)
    tau_dres_static_conv_1 = rho_expr * dot(grad_du_phys, conv_velocity)
    old_uss_term = rho_expr * inv_dt * old_subscale
    add_velocity_source = body_expr - momentum_projection + old_uss_term
    tau_test_conv = rho_expr * dot(dot(grad(v), kin.Finv), conv_velocity)
    tau_test_source_v = tau_test_conv - rho_expr * inv_dt * v
    tau_test_mass = tau_test_conv - inv_dt * v + grad_q_phys
    viscous_sigma = build_fluid_viscous_cauchy_stress(
        grad_u_phys=kin.grad_u_phys,
        div_u_phys=kin.div_u_phys,
        mu=mu_expr,
    )
    viscous_sigma_du = build_fluid_viscous_cauchy_stress(
        grad_u_phys=grad_du_phys,
        div_u_phys=div_du_phys,
        mu=mu_expr,
    )

    lhs_terms = {
        "advective": kin.J * rho_expr * dot(dot(grad_du_phys, conv_velocity), v) * dx_measure,
        "tau_advective": kin.J * tau_one * dot(tau_test_source_v, tau_dres_static_conv_1) * dx_measure,
        "v_grad_p_tau": kin.J * tau_one * dot(tau_test_conv, grad_dp_phys) * dx_measure,
        "v_grad_p_dt": -kin.J * tau_one * dot(rho_expr * inv_dt * v, grad_dp_phys) * dx_measure,
        "pressure_galerkin": -inner(kin.cof_F, grad(v)) * dp * dx_measure,
        "q_div_tau": kin.J * tau_one * dot(grad_q_phys, tau_dres_static_conv_1) * dx_measure,
        "continuity": inner(kin.cof_F, grad(du)) * q * dx_measure,
        "q_p_stabilization": kin.J * tau_one * dot(grad_q_phys, grad_dp_phys) * dx_measure,
        "div_stabilization": kin.J * ((tau_two + tau_p) * div_du_phys * div_v_phys) * dx_measure,
        "viscous": inner(kin.J * dot(viscous_sigma_du, kin.Finv.T), grad(v)) * dx_measure,
    }
    rhs_terms = {
        "body_old_subscale": kin.J * dot(body_expr + old_uss_term, v) * dx_measure,
        "tau_velocity_source": kin.J * tau_one * dot(tau_test_source_v, add_velocity_source) * dx_measure,
        "tau_q_source": kin.J * tau_one * dot(grad_q_phys, add_velocity_source) * dx_measure,
        "mass_projection": -kin.J * ((tau_two + tau_p) * mass_projection * div_v_phys) * dx_measure,
        "old_mass_residual": -kin.J * (tau_p * old_mass_residual * div_v_phys) * dx_measure,
        "minus_advective_current": -kin.J * rho_expr * dot(dot(kin.grad_u_phys, conv_velocity), v) * dx_measure,
        "minus_tau_advective_current": -kin.J * tau_one * dot(tau_test_source_v, tau_res_static_conv) * dx_measure,
        "minus_v_grad_p_tau_current": -kin.J * tau_one * dot(tau_test_conv, grad_p_phys) * dx_measure,
        "minus_v_grad_p_dt_current": kin.J * tau_one * dot(rho_expr * inv_dt * v, grad_p_phys) * dx_measure,
        "minus_pressure_galerkin_current": inner(kin.cof_F, grad(v)) * p_k * dx_measure,
        "minus_q_div_tau_current": -kin.J * tau_one * dot(grad_q_phys, tau_res_static_conv) * dx_measure,
        "minus_continuity_current": -inner(kin.cof_F, grad(u_k)) * q * dx_measure,
        "minus_q_p_stabilization_current": -kin.J * tau_one * dot(grad_q_phys, grad_p_phys) * dx_measure,
        "minus_div_stabilization_current": -kin.J * ((tau_two + tau_p) * kin.div_u_phys * div_v_phys) * dx_measure,
        "viscous": -inner(kin.J * dot(viscous_sigma, kin.Finv.T), grad(v)) * dx_measure,
    }
    forms = build_fluid_dvms_local_forms(
        du=du,
        dp=dp,
        v=v,
        q=q,
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        d_prev2=d_prev2,
        mesh_v_prev=mesh_v_prev,
        mesh_a_prev=mesh_a_prev,
        dx_measure=dx_measure,
        rho=rho,
        mu=mu,
        dt=dt,
        h=h,
        bossak_ma0=bossak_ma0,
        bossak_ma2=bossak_ma2,
        bossak_alpha=bossak_alpha,
        predicted_subscale=predicted_subscale,
        old_subscale=old_subscale,
        momentum_projection=momentum_projection,
        mass_projection=mass_projection,
        old_mass_residual=old_mass_residual,
        body_force=body_force,
        use_oss=use_oss,
    )
    mass_terms = {
        "mass_lhs": forms.mass_lhs,
        "mass_stabilization": forms.mass_stabilization,
        "velocity_residual": forms.add_velocity_residual,
        "velocity_jacobian": forms.add_velocity_jacobian,
        "system_residual": forms.system_residual,
        "system_jacobian": forms.system_jacobian,
        "system_condensed_residual": forms.system_condensed_residual,
        "system_condensed_jacobian": forms.system_condensed_jacobian,
    }
    return FluidDVMSKratosSplitForms(lhs_terms=lhs_terms, rhs_terms=rhs_terms, mass_terms=mass_terms)


__all__ = [
    "FluidDVMSKratosSplitForms",
    "FluidDVMSLocalForms",
    "FluidDVMSKinematics",
    "FluidDVMSPredictorIterationSymbolics",
    "FluidDVMSPredictorSymbolics",
    "build_fluid_cauchy_stress",
    "build_fluid_dvms_kinematics",
    "build_fluid_dvms_local_forms",
    "build_fluid_dvms_kratos_split_forms",
    "build_fluid_dvms_old_mass_residual",
    "build_fluid_dvms_predictor_iteration_symbolics",
    "build_fluid_dvms_predictor_symbolics",
    "build_fluid_viscous_cauchy_stress",
]
