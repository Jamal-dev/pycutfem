"""One-domain diffuse-interface biofilm model (Navier–Stokes–Brinkman–Biot + transport).

This module implements the weak residual and a manually coded consistent Jacobian
for the model described in `examples/biofilms/model/model.tex`.

Design goals
------------
* Keep the implementation debuggable by exposing per-block residuals.
* Use a one-step-θ scheme (θ=1: backward Euler, θ=0.5: Crank–Nicolson).
* Be compatible with all pycutfem backends ("python", "jit", "cpp").

Important compiler limitation
-----------------------------
The current `FormCompiler` only reliably supports `grad(...)` / `div(...)` when
applied directly to base Trial/Test/Function objects. Avoid `div(a*v)` and
`div(vS_k)` where `vS_k` is a linear combination. This module expands such
divergences explicitly (product rule / difference-of-divergences).
"""

from __future__ import annotations

from dataclasses import dataclass

from pycutfem.ufl.expressions import (
    Constant,
    Identity,
    div,
    dot,
    grad,
    inner,
    outer,
    trace,
)

from pycutfem.utils.nonlinear_solid_eulerian_refmap import (
    deulerian_k_inv,
    dsigma_neo_hookean,
    eulerian_k_inv,
    sigma_neo_hookean,
)


def _c(val: float) -> Constant:
    return Constant(float(val))


def _sqrt(expr):
    return expr ** _c(0.5)


def _one_minus(expr):
    # IMPORTANT: keep the "function-like" operand on the left so VecOpInfo handles
    # the arithmetic (float - VecOpInfo is not supported by the compiler).
    return (-expr) + _c(1.0)


def _epsilon(v):
    return _c(0.5) * (grad(v) + grad(v).T)


def _linear_elastic_term(u, eta, *, mu_s, lambda_s):
    # For symmetric stress, σ(u):∇η = 2μ ε(u):ε(η) + λ div(u) div(η).
    return _c(2.0) * mu_s * inner(_epsilon(u), _epsilon(eta)) + lambda_s * div(u) * div(eta)


def _chi_b(alpha):
    return alpha


def _chi_f(alpha):
    return _one_minus(alpha)


def _capacity(alpha, phi):
    # C = (1-α) + α φ
    return _chi_f(alpha) + _chi_b(alpha) * phi


def _rho(alpha, phi, *, rho_f):
    # ρ = (1-α) ρ_f + α ρ_f φ = ρ_f ((1-α) + α φ)
    return rho_f * _capacity(alpha, phi)


def _mu(alpha, phi, *, mu_f, mu_b_model: str = "phi_mu"):
    """
    Effective viscosity μ(α,φ).

    Choices:
      - "mu":      μ_B = μ_f (constant)               → μ = μ_f (no α/φ dependence)
      - "phi_mu":  μ_B = φ μ_f (Brinkman scaling)     → μ = μ_f ((1-α) + α φ)
    """
    mu_b_model = str(mu_b_model).strip().lower()
    if mu_b_model in {"mu", "const", "constant"}:
        mu_b = mu_f
    elif mu_b_model in {"phi_mu", "phi*mu", "phi"}:
        mu_b = phi * mu_f
    else:
        raise ValueError(f"Unknown mu_b_model {mu_b_model!r}.")
    return _chi_f(alpha) * mu_f + _chi_b(alpha) * mu_b


def _beta(alpha, phi, *, mu_f, kappa_inv):
    # β = α μ_f φ^2 κ^{-1} (here κ^{-1} can be scalar or tensor; we treat it as scalar for now)
    return _chi_b(alpha) * mu_f * (phi * phi) * kappa_inv


def _monod(S, *, mu_max, K_S):
    # Keep Function-like operand on the left: (S + K_S) not (K_S + S).
    return mu_max * (S / (S + K_S))


def _G(S, phi, *, k_g, mu_max, K_S):
    return k_g * _monod(S, mu_max=mu_max, K_S=K_S) * _one_minus(phi)


def _Pi_over_rho_s(S, phi, alpha, *, mu_max, K_S, k_d):
    # Π_b / ρ_s* (model.tex Monod choice) = (μ_max S/(K+S) - k_d) (1-φ) α
    return (_monod(S, mu_max=mu_max, K_S=K_S) - k_d) * _one_minus(phi) * alpha


def _R_S_consumption(S, phi, alpha, *, mu_max, K_S, k_d, Y):
    # R_S = (1/Y) Π_b/ρ_s* (default) (positive sink in the strong form).
    #
    # Unit note:
    # - If Π_b is a *mass* source [kg/(m^3 s)] and ρ_s* is an intrinsic solid density [kg/m^3],
    #   then Π_b/ρ_s* has units [1/s]. This matches a substrate variable S that is normalized
    #   by ρ_s* (dimensionless). For a *mass concentration* substrate, use R_S = Π_b/Y instead.
    return (_c(1.0) / Y) * _Pi_over_rho_s(S, phi, alpha, mu_max=mu_max, K_S=K_S, k_d=k_d)


def _detachment_from_shear_prev(
    *,
    v_prev,
    alpha_prev,
    mu_f,
    k_det,
    eta_n: float = 1.0e-12,
    dim: int = 2,
):
    """
    Build a *lagged* detachment rate D(τ) from the previous-step velocity.

    We deliberately use a simple shear proxy that is robust in the current UFL/assembler
    stack (no division by |∇α|). D is treated as a known coefficient in the Newton step,
    so the Jacobian remains consistent.
    """
    # τ ≈ ||ε(v_prev)|| (Frobenius)
    tau2 = inner(_epsilon(v_prev), _epsilon(v_prev))
    tau = _sqrt(tau2 + _c(float(eta_n)))
    return k_det * tau


@dataclass(frozen=True)
class BiofilmOneDomainForms:
    jacobian_form: object
    residual_form: object
    r_momentum: object
    r_mass: object
    r_skeleton: object
    r_phi: object
    r_alpha: object
    r_substrate: object
    # Optional per-block Jacobian contributions (useful for debugging/verification)
    a_momentum: object | None = None
    a_mass: object | None = None
    a_skeleton: object | None = None
    a_phi: object | None = None
    a_alpha: object | None = None
    a_substrate: object | None = None


def build_biofilm_one_domain_forms(
    *,
    # unknowns at t_{n+1}
    v_k,
    p_k,
    u_k,
    phi_k,
    alpha_k,
    S_k,
    # unknowns at t_n
    v_n,
    p_n,
    u_n,
    phi_n,
    alpha_n,
    S_n,
    # optional u_{n-1} (for θ terms with vS_n)
    u_nm1=None,
    # Newton increments
    dv,
    dp,
    du,
    dphi,
    dalpha,
    dS,
    # test functions
    v_test,
    q_test,
    u_test,
    phi_test,
    alpha_test,
    S_test,
    # measure
    dx,
    # time integration
    dt,
    theta: float = 1.0,
    # physical parameters
    rho_f=None,
    mu_f=None,
    kappa_inv=None,
    mu_s=None,
    lambda_s=None,
    # optional solid inertia (Eulerian skeleton acceleration)
    rho_s0_tilde=None,
    include_skeleton_acceleration: bool = False,
    # solid/permeability modelling toggles
    solid_model: str = "linear",
    c_nh=None,
    beta_nh=None,
    kappa_inv_model: str = "spatial",
    # transport parameters
    D_phi: float = 0.0,
    gamma_phi: float = 0.0,
    gamma_u: float = 0.0,
    D_alpha: float = 0.0,
    D_S: float = 0.0,
    # growth / detachment parameters
    mu_max: float = 0.0,
    K_S: float = 1.0,
    k_g: float = 0.0,
    k_d: float = 0.0,
    Y: float = 1.0,
    k_det: float = 0.0,
    # modelling toggles
    mu_b_model: str = "phi_mu",
    dim: int = 2,
    # sources (all optional; default to 0)
    f_v=None,
    f_u=None,
    s_v=None,
    f_phi=None,
    f_alpha=None,
    f_S=None,
    # detachment rate override (if given, used instead of shear-based lagged rate)
    D_det_prev=None,
) -> BiofilmOneDomainForms:
    """
    Build residual + consistent Jacobian for the one-domain biofilm system.

    Notes
    -----
    - `dt` may be a float or a `Constant`.
    - All source terms are *added* on the RHS of the strong form; i.e. we build
      residuals of the form "LHS - RHS" so the default is homogeneous (0).
    """
    if rho_f is None or mu_f is None or kappa_inv is None:
        raise ValueError("Missing required physical parameters: rho_f, mu_f, kappa_inv.")

    solid_model_key = str(solid_model).strip().lower()
    if solid_model_key in {"linear", "small_strain", "linear_elastic"}:
        if mu_s is None or lambda_s is None:
            raise ValueError("Solid model 'linear' requires mu_s and lambda_s.")
    elif solid_model_key in {"neo_hookean", "neo_hookean_eulerian", "nh"}:
        # Allow explicit neo-Hookean parameters; otherwise derive from (mu_s, lambda_s).
        if (c_nh is None or beta_nh is None) and (mu_s is None or lambda_s is None):
            raise ValueError("Solid model 'neo_hookean' requires either (c_nh,beta_nh) or (mu_s,lambda_s).")
    else:
        raise ValueError(f"Unknown solid_model {solid_model!r}.")

    th = _c(float(theta))
    one_m_th = _c(1.0) - th
    inv_dt = _c(1.0) / dt

    # Optional sources default to 0 (as *expressions*, not test-weighted terms).
    zero_scalar = _c(0.0)
    zero_vector = Constant([0.0] * int(dim), dim=1)
    f_v = f_v if f_v is not None else zero_vector
    f_u = f_u if f_u is not None else zero_vector
    s_v = s_v if s_v is not None else zero_scalar
    f_phi = f_phi if f_phi is not None else zero_scalar
    f_alpha = f_alpha if f_alpha is not None else zero_scalar
    f_S = f_S if f_S is not None else zero_scalar

    # ------------------------------------------------------------------
    # Kinematics: skeleton velocity via backward Euler
    # ------------------------------------------------------------------
    vS_k = (u_k - u_n) / dt
    div_vS_k = (div(u_k) - div(u_n)) / dt
    if u_nm1 is None:
        vS_n = _c(0.0) * vS_k
        div_vS_n = _c(0.0) * div_vS_k
    else:
        vS_n = (u_n - u_nm1) / dt
        div_vS_n = (div(u_n) - div(u_nm1)) / dt

    dvS = du / dt
    div_dvS = div(du) / dt

    # ------------------------------------------------------------------
    # Coefficients (at n/k) and their variations (at k only)
    # ------------------------------------------------------------------
    rho_k = _rho(alpha_k, phi_k, rho_f=rho_f)
    rho_n = _rho(alpha_n, phi_n, rho_f=rho_f)

    mu_k = _mu(alpha_k, phi_k, mu_f=mu_f, mu_b_model=mu_b_model)
    mu_n = _mu(alpha_n, phi_n, mu_f=mu_f, mu_b_model=mu_b_model)

    # Coefficient variations w.r.t (α,φ) at k:
    dC = (phi_k - _c(1.0)) * dalpha + alpha_k * dphi  # δ((1-α)+αφ)
    drho = rho_f * dC

    if str(mu_b_model).strip().lower() in {"mu", "const", "constant"}:
        dmu = _c(0.0) * dphi
    else:
        dmu = mu_f * dC

    # Optional skeleton inertia coefficient (0 by default).
    rho_s0_tilde = rho_s0_tilde if rho_s0_tilde is not None else zero_scalar

    # ------------------------------------------------------------------
    # Permeability / drag handling
    # ------------------------------------------------------------------
    # Drag coefficient (without κ^{-1}): α μ_f φ^2.
    beta_coeff_k = alpha_k * mu_f * (phi_k * phi_k)
    beta_coeff_n = alpha_n * mu_f * (phi_n * phi_n)
    dbeta_coeff = mu_f * ((phi_k * phi_k) * dalpha + (_c(2.0) * alpha_k * phi_k) * dphi)

    kappa_inv_key = str(kappa_inv_model).strip().lower()

    # Fast/robust path for *scalar spatial* κ^{-1}: keep the legacy scalar formulation.
    if kappa_inv_key in {"spatial", "constant", "const"} and getattr(kappa_inv, "dim", None) == 0:
        drag_mode = "scalar"
        beta_k = beta_coeff_k * kappa_inv
        beta_n = beta_coeff_n * kappa_inv
        dbeta = dbeta_coeff * kappa_inv
    else:
        drag_mode = "matrix"

        def _as_invperm_matrix(k_inv, *, dim: int):
            # Accept scalar Constant/Expression as isotropic k_inv * I.
            if getattr(k_inv, "dim", None) == 0:
                return k_inv * Identity(int(dim))
            return k_inv

        K_inv = _as_invperm_matrix(kappa_inv, dim=int(dim))
        if kappa_inv_key in {"refmap", "eulerian_refmap", "eulerian", "reference_map", "reference-map"}:
            k_inv_k = eulerian_k_inv(u_k, K_inv, dim=int(dim))
            k_inv_n = eulerian_k_inv(u_n, K_inv, dim=int(dim))
            dk_inv_k = deulerian_k_inv(u_k, du, K_inv, dim=int(dim))
        elif kappa_inv_key in {"spatial", "constant", "const"}:
            k_inv_k = K_inv
            k_inv_n = K_inv
            dk_inv_k = None
        else:
            raise ValueError(f"Unknown kappa_inv_model {kappa_inv_model!r}.")

        diff_k = v_k - vS_k
        diff_n = v_n - vS_n
        ddiff = dv - dvS

        kdrag_k = dot(k_inv_k, diff_k)
        kdrag_n = dot(k_inv_n, diff_n)

        dkdrag_k = dot(k_inv_k, ddiff)
        if dk_inv_k is not None:
            dkdrag_k += dot(dk_inv_k, diff_k)

    # Detachment coefficient: lagged by default (depends on previous v/α)
    if D_det_prev is None:
        D_det_prev = _detachment_from_shear_prev(v_prev=v_n, alpha_prev=alpha_n, mu_f=mu_f, k_det=_c(float(k_det)), dim=dim)

    # ------------------------------------------------------------------
    # (i) Momentum: generalized Navier–Stokes–Brinkman
    # ------------------------------------------------------------------
    vdot = (v_k - v_n) * inv_dt

    conv_k = dot(dot(grad(v_k), v_k), v_test)
    conv_n = dot(dot(grad(v_n), v_n), v_test)

    r_mom = inner(rho_k * vdot, v_test) * dx
    r_mom += th * rho_k * conv_k * dx + one_m_th * rho_n * conv_n * dx
    r_mom += _c(2.0) * th * mu_k * inner(_epsilon(v_k), _epsilon(v_test)) * dx
    r_mom += _c(2.0) * one_m_th * mu_n * inner(_epsilon(v_n), _epsilon(v_test)) * dx
    r_mom += -p_k * div(v_test) * dx
    if drag_mode == "scalar":
        r_mom += beta_k * dot(v_k, v_test) * dx
        r_mom += -beta_k * dot(vS_k, v_test) * dx
    else:
        r_mom += beta_coeff_k * dot(kdrag_k, v_test) * dx
    r_mom += -dot(f_v, v_test) * dx

    a_mom = rho_k * inv_dt * dot(dv, v_test) * dx
    a_mom += drho * dot(vdot, v_test) * dx

    a_mom += th * (drho * conv_k + rho_k * dot(dot(grad(dv), v_k), v_test) + rho_k * dot(dot(grad(v_k), dv), v_test)) * dx
    a_mom += _c(2.0) * th * (dmu * inner(_epsilon(v_k), _epsilon(v_test)) + mu_k * inner(_epsilon(dv), _epsilon(v_test))) * dx

    a_mom += -dp * div(v_test) * dx
    if drag_mode == "scalar":
        a_mom += dbeta * (dot(v_k, v_test) - dot(vS_k, v_test)) * dx
        a_mom += beta_k * dot(dv, v_test) * dx
        a_mom += -beta_k * dot(dvS, v_test) * dx
    else:
        a_mom += dbeta_coeff * dot(kdrag_k, v_test) * dx
        a_mom += beta_coeff_k * dot(dkdrag_k, v_test) * dx

    # ------------------------------------------------------------------
    # (ii) Mass / volume constraint (expanded divergence)
    # ------------------------------------------------------------------
    # F_m = C v + B vS,  C=(1-α)+αφ,  B=α(1-φ)
    C_k = _capacity(alpha_k, phi_k)
    C_n = _capacity(alpha_n, phi_n)
    B_k = alpha_k * _one_minus(phi_k)
    B_n = alpha_n * _one_minus(phi_n)

    # div(C v) = C div(v) + grad(C)·v, with grad(C) = (φ-1) grad(α) + α grad(φ)
    gradC_k = grad(alpha_k) * (phi_k - _c(1.0)) + grad(phi_k) * alpha_k
    gradC_n = grad(alpha_n) * (phi_n - _c(1.0)) + grad(phi_n) * alpha_n

    divCv_k = C_k * div(v_k) + dot(gradC_k, v_k)
    divCv_n = C_n * div(v_n) + dot(gradC_n, v_n)

    # div(B vS) = B div(vS) + grad(B)·vS, with grad(B) = (1-φ) grad(α) - α grad(φ)
    gradB_k = grad(alpha_k) * _one_minus(phi_k) - grad(phi_k) * alpha_k
    gradB_n = grad(alpha_n) * _one_minus(phi_n) - grad(phi_n) * alpha_n

    divBvS_k = B_k * div_vS_k + dot(gradB_k, vS_k)
    divBvS_n = B_n * div_vS_n + dot(gradB_n, vS_n)

    divF_k = divCv_k + divBvS_k
    divF_n = divCv_n + divBvS_n

    r_mass = q_test * (th * divF_k + one_m_th * divF_n - alpha_k * s_v) * dx

    # Jacobian of divF_k (k-part only)
    # δC = (φ-1) δα + α δφ
    dC_k = (phi_k - _c(1.0)) * dalpha + alpha_k * dphi
    dB_k = _one_minus(phi_k) * dalpha - alpha_k * dphi

    # NOTE: Keep GradOpInfo on the left in scalar×grad products (trial×grad(function)
    # is not implemented, but grad(function)×trial is).
    dgradC_k = grad(dalpha) * (phi_k - _c(1.0)) + grad(alpha_k) * dphi + grad(phi_k) * dalpha + grad(dphi) * alpha_k
    dgradB_k = grad(dalpha) * _one_minus(phi_k) - grad(alpha_k) * dphi - grad(phi_k) * dalpha - grad(dphi) * alpha_k

    d_divCv_k = dC_k * div(v_k) + C_k * div(dv) + dot(dgradC_k, v_k) + dot(gradC_k, dv)
    d_divBvS_k = dB_k * div_vS_k + B_k * div_dvS + dot(dgradB_k, vS_k) + dot(gradB_k, dvS)

    a_mass = q_test * th * (d_divCv_k + d_divBvS_k - dalpha * s_v) * dx

    # ------------------------------------------------------------------
    # (iii) Skeleton momentum (optional inertia + linear/neo-Hookean stress)
    # ------------------------------------------------------------------
    if solid_model_key in {"linear", "small_strain", "linear_elastic"}:
        r_el_k = _linear_elastic_term(u_k, u_test, mu_s=mu_s, lambda_s=lambda_s)
        r_el_n = _linear_elastic_term(u_n, u_test, mu_s=mu_s, lambda_s=lambda_s)
        a_el = _linear_elastic_term(du, u_test, mu_s=mu_s, lambda_s=lambda_s)
    else:
        # Eulerian reference-map Neo-Hookean stress (Cauchy), compatible with FPI poro Eulerian module.
        if c_nh is None:
            c_nh = mu_s / _c(2.0)
        if beta_nh is None:
            beta_nh = lambda_s / (_c(2.0) * mu_s)

        sig_k = sigma_neo_hookean(u_k, c_nh, beta_nh, dim=int(dim))
        sig_n = sigma_neo_hookean(u_n, c_nh, beta_nh, dim=int(dim))
        r_el_k = inner(sig_k, grad(u_test))
        r_el_n = inner(sig_n, grad(u_test))
        dsig_k = dsigma_neo_hookean(u_k, du, c_nh, beta_nh, dim=int(dim))
        a_el = inner(dsig_k, grad(u_test))

    r_skel_k = r_el_k
    r_skel_n = r_el_n

    # pressure coupling: -(1-φ) p div(η)
    r_skel_k += -(_one_minus(phi_k) * p_k) * div(u_test)
    r_skel_n += -(_one_minus(phi_n) * p_n) * div(u_test)

    # drag reaction: -β (v - vS)
    # Since beta already contains α, if we use alpha again then it would square and it won't 
    # be equal to the drag from the momentum of the fluid.
    if drag_mode == "scalar":
        r_skel_drag_k = -beta_k * dot(v_k - vS_k, u_test)
        r_skel_drag_n = -beta_n * dot(v_n - vS_n, u_test)
    else:
        r_skel_drag_k = -beta_coeff_k * dot(kdrag_k, u_test)
        r_skel_drag_n = -beta_coeff_n * dot(kdrag_n, u_test)

    r_skeleton = (th * alpha_k * r_skel_k + one_m_th * alpha_n * r_skel_n
                  + th * r_skel_drag_k + one_m_th * r_skel_drag_n) * dx
    r_skeleton += -dot(alpha_k * f_u, u_test) * dx

    # Optional extension penalty to keep u well-posed in the free-fluid region.
    # This is a pragmatic stabilization for the one-domain formulation: as α→0,
    # the skeleton equation vanishes and the u-DOFs can become nearly unconstrained,
    # leading to ill-conditioned Jacobians and Newton stagnation.
    gamma_u_c = _c(float(gamma_u))
    if float(gamma_u) != 0.0:
        r_skeleton += gamma_u_c * _one_minus(alpha_k) * dot(u_k, u_test) * dx

    # Jacobian contributions (k-part only)
    if drag_mode == "scalar":
        drag_term_k = -beta_k * dot(v_k, u_test) + beta_k * dot(vS_k, u_test)
        d_drag_term_k = -dbeta * (dot(v_k, u_test) - dot(vS_k, u_test))
        d_drag_term_k += -beta_k * (dot(dv, u_test) - dot(dvS, u_test))
    else:
        drag_term_k = -beta_coeff_k * dot(kdrag_k, u_test)
        d_drag_term_k = -(dbeta_coeff * dot(kdrag_k, u_test) + beta_coeff_k * dot(dkdrag_k, u_test))

    a_skel = th * (
        alpha_k * a_el
        + dalpha * r_el_k
        - dalpha * (_one_minus(phi_k) * p_k) * div(u_test)
        + alpha_k * (dphi * p_k) * div(u_test)
        - alpha_k * (_one_minus(phi_k) * dp) * div(u_test)
    ) * dx
    # Drag term is *not* multiplied by alpha again: beta already contains alpha (one-domain blend).
    a_skel += th * d_drag_term_k * dx
    a_skel += -dot(dalpha * f_u, u_test) * dx

    if float(gamma_u) != 0.0:
        a_skel += gamma_u_c * ((-_c(1.0) * dalpha) * dot(u_k, u_test) + _one_minus(alpha_k) * dot(du, u_test)) * dx

    # Optional Eulerian skeleton inertia (a^S = ∂_t vS + (vS·∇)vS).
    if bool(include_skeleton_acceleration) and float(rho_s0_tilde) != 0.0:
        grad_vS_k = (grad(u_k) - grad(u_n)) / dt
        if u_nm1 is None:
            grad_vS_n = _c(0.0) * grad_vS_k
        else:
            grad_vS_n = (grad(u_n) - grad(u_nm1)) / dt

        acc_local = (vS_k - vS_n) / dt
        adv_k = dot(grad_vS_k, vS_k)
        adv_n = dot(grad_vS_n, vS_n)
        acc = acc_local + th * adv_k + one_m_th * adv_n

        m_s = rho_s0_tilde * _one_minus(phi_k)
        r_skeleton += alpha_k * inner(m_s * acc, u_test) * dx

        # δm_s = -rho_s0_tilde δφ,  δacc = δvS/dt + θ δ((∇vS)vS)
        dm_s = rho_s0_tilde * (-dphi)
        dvS_over_dt = dvS / dt
        grad_dvS = grad(du) / dt
        dadv_k = dot(grad_dvS, vS_k) + dot(grad_vS_k, dvS)
        dacc = dvS_over_dt + th * dadv_k

        a_skel += inner((dalpha * m_s + alpha_k * dm_s) * acc + alpha_k * m_s * dacc, u_test) * dx

    # ------------------------------------------------------------------
    # (iv) Porosity evolution (Eulerian, advected by vS)
    # ------------------------------------------------------------------
    D_phi_c = _c(float(D_phi))
    gamma_phi_c = _c(float(gamma_phi))

    Pi_k = _Pi_over_rho_s(S_k, phi_k, alpha_k, mu_max=_c(float(mu_max)), K_S=_c(float(K_S)), k_d=_c(float(k_d)))
    Pi_n = _Pi_over_rho_s(S_n, phi_n, alpha_n, mu_max=_c(float(mu_max)), K_S=_c(float(K_S)), k_d=_c(float(k_d)))

    # F_phi = vS·∇φ - (1-φ) div vS + Π/ρ_s*
    # NOTE: the dot-visitor supports Grad(Function)·Vec(Function) but not Vec(Function)·Grad(Function).
    Fphi_k = dot(grad(phi_k), vS_k) - _one_minus(phi_k) * div_vS_k + Pi_k
    Fphi_n = dot(grad(phi_n), vS_n) - _one_minus(phi_n) * div_vS_n + Pi_n

    r_phi = alpha_k * phi_test * ((phi_k - phi_n) * inv_dt) * dx
    r_phi += th * alpha_k * phi_test * Fphi_k * dx
    r_phi += one_m_th * alpha_n * phi_test * Fphi_n * dx
    r_phi += D_phi_c * inner(grad(phi_k), grad(phi_test)) * dx
    r_phi += gamma_phi_c * _one_minus(alpha_k) * (phi_k - _c(1.0)) * phi_test * dx
    r_phi += -phi_test * f_phi * dx

    # Jacobian (k-part only)
    # δΠ
    mu_max_c = _c(float(mu_max))
    K_S_c = _c(float(K_S))
    mon_k = _monod(S_k, mu_max=mu_max_c, K_S=K_S_c)
    dmon_dS = mu_max_c * (K_S_c / ((S_k + K_S_c) * (S_k + K_S_c)))
    # Π = (mon - k_d) (1-φ) α
    dPi = (dmon_dS * dS) * _one_minus(phi_k) * alpha_k
    dPi += (mon_k - _c(float(k_d))) * (-(dphi) * alpha_k + _one_minus(phi_k) * dalpha)

    a_phi = alpha_k * phi_test * (dphi * inv_dt) * dx
    a_phi += dalpha * phi_test * ((phi_k - phi_n) * inv_dt) * dx
    a_phi += th * dalpha * phi_test * Fphi_k * dx
    a_phi += th * alpha_k * phi_test * (
        dot(grad(phi_k), dvS) + dot(grad(dphi), vS_k) + dphi * div_vS_k - _one_minus(phi_k) * div_dvS + dPi
    ) * dx
    a_phi += D_phi_c * inner(grad(dphi), grad(phi_test)) * dx
    a_phi += gamma_phi_c * ((-_c(1.0) * dalpha) * (phi_k - _c(1.0)) + _one_minus(alpha_k) * dphi) * phi_test * dx

    # ------------------------------------------------------------------
    # (v) Indicator evolution (advection–diffusion–reaction)
    # ------------------------------------------------------------------
    D_alpha_c = _c(float(D_alpha))
    G_k = _G(S_k, phi_k, k_g=_c(float(k_g)), mu_max=_c(float(mu_max)), K_S=_c(float(K_S)))
    G_n = _G(S_n, phi_n, k_g=_c(float(k_g)), mu_max=_c(float(mu_max)), K_S=_c(float(K_S)))

    f_alpha_k = (alpha_k - alpha_n) * inv_dt
    f_alpha_k += th * (dot(grad(alpha_k), vS_k) - G_k * alpha_k * _one_minus(alpha_k) + D_det_prev * alpha_k)
    f_alpha_k += one_m_th * (dot(grad(alpha_n), vS_n) - G_n * alpha_n * _one_minus(alpha_n) + D_det_prev * alpha_n)

    r_alpha = alpha_test * f_alpha_k * dx
    r_alpha += D_alpha_c * inner(grad(alpha_k), grad(alpha_test)) * dx
    r_alpha += -alpha_test * f_alpha * dx

    # Jacobian (k-part only)
    # δG
    dG = (_c(float(k_g)) * _one_minus(phi_k) * (dmon_dS * dS) + (-_c(float(k_g)) * mon_k) * dphi)
    # δ[ G α(1-α) ] = (δG) α(1-α) + G (1-2α) δα
    dalpha_logistic = _one_minus(_c(2.0) * alpha_k) * dalpha

    a_alpha = alpha_test * (dalpha * inv_dt) * dx
    a_alpha += alpha_test * th * (dot(grad(alpha_k), dvS) + dot(grad(dalpha), vS_k)) * dx
    a_alpha += alpha_test * th * (-(dG * alpha_k * _one_minus(alpha_k) + G_k * dalpha_logistic) + D_det_prev * dalpha) * dx
    a_alpha += D_alpha_c * inner(grad(dalpha), grad(alpha_test)) * dx

    # ------------------------------------------------------------------
    # (vi) Substrate transport
    # ------------------------------------------------------------------
    D_S_c = _c(float(D_S))
    CSk = _capacity(alpha_k, phi_k) * S_k
    CSn = _capacity(alpha_n, phi_n) * S_n

    # Strong: ∂t(CS) + div(CS v) - div(D grad S) + R_S = f_S.
    #
    # IMPORTANT: avoid `dot(CS*v, grad(test))` because the current dot/left_dot
    # implementation cannot handle scalar-test gradients with vector trials.
    # We instead expand div(CS v) = CS div(v) + ∇(CS)·v and use
    # ∇(CS) = S ∇C + C ∇S.
    RS_k = _R_S_consumption(S_k, phi_k, alpha_k, mu_max=_c(float(mu_max)), K_S=_c(float(K_S)), k_d=_c(float(k_d)), Y=_c(float(Y)))
    RS_n = _R_S_consumption(S_n, phi_n, alpha_n, mu_max=_c(float(mu_max)), K_S=_c(float(K_S)), k_d=_c(float(k_d)), Y=_c(float(Y)))

    div_CSv_k = CSk * div(v_k) + S_k * dot(gradC_k, v_k) + C_k * dot(grad(S_k), v_k)
    div_CSv_n = CSn * div(v_n) + S_n * dot(gradC_n, v_n) + C_n * dot(grad(S_n), v_n)

    r_sub = S_test * ((CSk - CSn) * inv_dt) * dx
    r_sub += S_test * (th * div_CSv_k + one_m_th * div_CSv_n) * dx
    r_sub += D_S_c * th * inner(grad(S_k), grad(S_test)) * dx + D_S_c * one_m_th * inner(grad(S_n), grad(S_test)) * dx
    r_sub += S_test * (th * RS_k + one_m_th * RS_n) * dx
    r_sub += -S_test * f_S * dx

    # Jacobian (k-part only)
    dCSk = dC * S_k + _capacity(alpha_k, phi_k) * dS
    dRS = (_c(1.0) / _c(float(Y))) * dPi  # RS = (1/Y) (Π_b/ρ_s*)  (see unit note in _R_S_consumption)

    d_div_CSv_k = dCSk * div(v_k) + CSk * div(dv)
    d_div_CSv_k += dS * dot(gradC_k, v_k) + S_k * dot(dgradC_k, v_k) + S_k * dot(gradC_k, dv)
    d_div_CSv_k += dC_k * dot(grad(S_k), v_k) + C_k * dot(grad(dS), v_k) + C_k * dot(grad(S_k), dv)

    a_sub = S_test * (dCSk * inv_dt) * dx
    a_sub += S_test * th * d_div_CSv_k * dx
    a_sub += D_S_c * th * inner(grad(dS), grad(S_test)) * dx
    a_sub += S_test * th * dRS * dx

    # ------------------------------------------------------------------
    residual_form = r_mom + r_mass + r_skeleton + r_phi + r_alpha + r_sub
    jacobian_form = a_mom + a_mass + a_skel + a_phi + a_alpha + a_sub

    return BiofilmOneDomainForms(
        jacobian_form=jacobian_form,
        residual_form=residual_form,
        r_momentum=r_mom,
        r_mass=r_mass,
        r_skeleton=r_skeleton,
        r_phi=r_phi,
        r_alpha=r_alpha,
        r_substrate=r_sub,
        a_momentum=a_mom,
        a_mass=a_mass,
        a_skeleton=a_skel,
        a_phi=a_phi,
        a_alpha=a_alpha,
        a_substrate=a_sub,
    )
