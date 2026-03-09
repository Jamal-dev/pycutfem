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
    FacetNormal,
    Identity,
    Laplacian,
    MeshSize,
    avg,
    div,
    dot,
    grad,
    inner,
    jump,
    outer,
    trace,
)

from pycutfem.ufl.linalg import (
    d_spectral_positive_part_2x2_sym,
    spectral_positive_part_2x2_sym,
    smooth_pos as _smooth_pos_u,
    smooth_pos_derivative as _smooth_pos_u_prime,
)

from ..shared.nonlinear_solid_refmap import (
    deulerian_k_inv,
    dsigma_svk,
    dsigma_svk_miehe_split,
    dsigma_hencky,
    dsigma_hencky_miehe_split,
    dsigma_neo_hookean,
    eulerian_k_inv,
    svk_tensile_energy_miehe,
    hencky_tensile_energy_miehe,
    sigma_svk,
    sigma_svk_miehe_split,
    sigma_hencky,
    sigma_hencky_miehe_split,
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


def _grad_inner_jump(u, v, n):
    """Penalty on the jump of the normal derivative across an interior facet."""
    # Use the UFL-style "jump with normal" form to avoid ambiguous normals in
    # `dot(jump(...), n)` and keep backend parity for scalar/vector fields:
    #   jump(grad(u), n) = grad(u+)·n+ + grad(u-)·n-
    # which equals (grad(u+) - grad(u-))·n+.
    ju = jump(grad(u), n)
    jv = jump(grad(v), n)
    return inner(ju, jv)


def _linear_elastic_term(u, eta, *, mu_s, lambda_s):
    # For symmetric stress, σ(u):∇η = 2μ ε(u):ε(η) + λ div(u) div(η).
    return _c(2.0) * mu_s * inner(_epsilon(u), _epsilon(eta)) + lambda_s * div(u) * div(eta)


def _vector_component(vec_expr, idx: int):
    try:
        return vec_expr[idx]
    except Exception:
        val = getattr(vec_expr, "value", None)
        if val is None:
            raise
        return _c(float(val[int(idx)]))


def _dot_2d_components(vec_expr, vec_test):
    return _vector_component(vec_expr, 0) * vec_test[0] + _vector_component(vec_expr, 1) * vec_test[1]


def _smooth_pos(x, *, eta: float = 1.0e-12):
    """Smooth positive part ⟨x⟩_+ ≈ 0.5 (x + sqrt(x^2 + eta))."""
    return _c(0.5) * (x + _sqrt(x * x + _c(float(eta))))


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


def _mu(alpha, phi, *, mu_f, mu_b=None, mu_b_model: str = "phi_mu"):
    """
    Effective viscosity μ(α,φ).

    Choices:
      - "mu":      μ_B = μ_f (constant)               → μ = μ_f (no α/φ dependence)
      - "phi_mu":  μ_B = φ μ_f (Brinkman scaling)     → μ = μ_f ((1-α) + α φ)
      - "alpha_mu": μ_B = μ_b (constant)              → μ = (1-α) μ_f + α μ_b
      - "alpha_phi_mu": μ_B = φ μ_b                   → μ = (1-α) μ_f + α φ μ_b
    """
    mu_b_model = str(mu_b_model).strip().lower()
    if mu_b_model in {"mu", "const", "constant"}:
        mu_b_eff = mu_f
    elif mu_b_model in {"phi_mu", "phi*mu", "phi"}:
        mu_b_eff = phi * mu_f
    elif mu_b_model in {"alpha_mu", "alpha", "mu_b", "biofilm_mu", "biofilm"}:
        if mu_b is None:
            raise ValueError("mu_b_model='alpha_mu' requires mu_b to be provided.")
        mu_b_eff = mu_b
    elif mu_b_model in {"alpha_phi_mu", "alpha_phi", "alpha*phi_mu", "alpha*phi*mu"}:
        if mu_b is None:
            raise ValueError("mu_b_model='alpha_phi_mu' requires mu_b to be provided.")
        mu_b_eff = phi * mu_b
    else:
        raise ValueError(f"Unknown mu_b_model {mu_b_model!r}.")
    return _chi_f(alpha) * mu_f + _chi_b(alpha) * mu_b_eff


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
    # R_S = (1/Y) Π_b/ρ_s*  (sink in the strong form) where Π_b/ρ_s* has units [1/s].
    #
    # Unit convention used by the default implementation:
    # - `_Pi_over_rho_s` returns Π_b/ρ_s* with units [1/s] if (mu_max,k_d) are rates [1/s].
    # - If the substrate variable S is normalized by ρ_s* (dimensionless), then R_S above
    #   is consistent.
    #
    # If instead S is a *mass concentration* [kg/m^3], then the physical sink is
    #   R_S = Π_b / Y = (ρ_s* / Y) (Π_b/ρ_s*).
    #
    # This scaling is handled in `build_biofilm_one_domain_forms` via the `rho_s_star`
    # parameter.
    return (_c(1.0) / Y) * _Pi_over_rho_s(S, phi, alpha, mu_max=mu_max, K_S=K_S, k_d=k_d)


def _detachment_from_shear_prev(
    *,
    v_prev,
    alpha_prev,
    mu_prev,
    k_det,
    eta_n: float = 1.0e-12,
    dim: int = 2,
):
    """
    Build a *lagged* detachment rate D_det from previous-step fields.

    This model uses a *rate coefficient* D_det that is treated as known in the Newton step.
    Detachment is then localized to the diffuse interface via a smooth delta δ(α) (handled
    in the α and X equations).

    Implementation note: while an interface-traction model can be written in terms of
    n = ∇α/||∇α|| and tangential projections, the current pycutfem UFL/compiler stack
    does not robustly support the required tensor products for gradients in all backends.
    We therefore use a simple, robust shear proxy:
        τ ≈ ||ε(v_prev)||_F
        D_det = k_det * sqrt( τ^2 + η )
    """
    tau2 = inner(_epsilon(v_prev), _epsilon(v_prev))
    tau = _sqrt(tau2 + _c(float(eta_n)))
    return k_det * tau


@dataclass(frozen=True)
class BiofilmOneDomainForms:
    jacobian_form: object
    residual_form: object
    r_momentum: object
    r_mass: object
    # Solid kinematics (Eulerian reference-map constraint linking u and vS)
    r_kinematics: object
    r_skeleton: object
    r_phi: object
    r_alpha: object
    # Optional: Cahn–Hilliard chemical potential equation for α (μ_α)
    r_mu_alpha: object | None
    r_damage: object | None
    r_substrate: object
    # Optional per-block Jacobian contributions (useful for debugging/verification)
    a_momentum: object | None = None
    a_mass: object | None = None
    a_kinematics: object | None = None
    a_skeleton: object | None = None
    a_phi: object | None = None
    a_alpha: object | None = None
    a_mu_alpha: object | None = None
    a_damage: object | None = None
    a_substrate: object | None = None
    r_detached: object | None = None
    a_detached: object | None = None
    # Optional: conservative Allen–Cahn (global Lagrange multiplier λ_α)
    r_alpha_lambda: object | None = None
    a_alpha_lambda: object | None = None


def build_biofilm_one_domain_forms(
    *,
    # unknowns at t_{n+1}
    v_k,
    p_k,
    # solid velocity (primary unknown)
    vS_k,
    # Eulerian reference-map variable / skeleton displacement-like field
    u_k,
    phi_k,
    alpha_k,
    mu_alpha_k=None,
    S_k,
    # optional: bulk damage / cohesion loss (0=intact, 1=failed)
    d_k=None,
    # optional: detached-biomass concentration at t_{n+1}
    X_k=None,
    # unknowns at t_n
    v_n,
    p_n,
    vS_n,
    u_n,
    phi_n,
    alpha_n,
    mu_alpha_n=None,
    S_n,
    # optional: bulk damage at t_n
    d_n=None,
    # optional: detached-biomass concentration at t_n
    X_n=None,
    # Newton increments
    dv,
    dp,
    dvS,
    du,
    dphi,
    dalpha,
    dmu_alpha=None,
    dS,
    dd=None,
    dX=None,
    # test functions
    v_test,
    q_test,
    vS_test,
    u_test,
    phi_test,
    alpha_test,
    mu_alpha_test=None,
    S_test,
    d_test=None,
    X_test=None,
    # measure
    dx,
    # time integration
    dt,
    theta: float = 1.0,
    # physical parameters
    rho_f=None,
    mu_f=None,
    mu_b=None,
    kappa_inv=None,
    mu_s=None,
    lambda_s=None,
    # Optional Kelvin–Voigt viscoelasticity for the skeleton (small-strain only):
    # σ_visc = 2 η_s ε(v^S), with v^S treated as a primary unknown.
    solid_visco_eta: float = 0.0,
    # optional solid inertia (Eulerian skeleton acceleration)
    rho_s0_tilde=None,
    include_skeleton_acceleration: bool = False,
    # How to treat the convective part of the Eulerian skeleton inertia
    # div(ρ_S v^S ⊗ v^S). "lagged" (default) uses a Picard-like linearization
    # div(ρ_S^n v^{S,n} ⊗ v^{S,k}) for robustness; "full" keeps the nonlinear
    # term.
    skeleton_inertia_convection: str = "lagged",
    # How to treat the convective part of the fluid momentum.
    #
    # The conservative convection in this one-domain formulation includes both
    #   ρ (v·∇)v  and  v div(ρ v),
    # where ρ = ρ_f C(α,φ). This can be a major source of nonlinearity in the
    # monolithic Newton solve.
    #
    # - "full" (default): fully nonlinear convection at the k-level
    # - "lagged": Picard/IMEX-like linearization using n-level coefficients/advectors
    # - "off": omit convection entirely (Stokes/Brinkman limit)
    fluid_convection: str = "full",
    # solid/permeability modelling toggles
    solid_model: str = "linear",
    c_nh=None,
    beta_nh=None,
    kappa_inv_model: str = "spatial",
    kappa_inv_phi_ref: float = 0.3,
    kappa_inv_kc_eps: float = 1.0e-12,
    # transport parameters
    D_phi: float = 0.0,
    gamma_phi: float = 0.0,
    # transport stabilizations (consistent, for advection-dominated cases)
    phi_supg: float = 0.0,
    phi_cip: float = 0.0,
    gamma_u: float = 0.0,
    u_extension_mode: str = "l2",
    gamma_u_pin: float = 0.0,
    # Optional SUPG-like streamline diffusion stabilizations:
    # - v_supg: fluid momentum convection stabilization
    # - u_supg: kinematic (u) transport stabilization
    v_supg: float = 0.0,
    u_supg: float = 0.0,
    # Optional CIP (interior penalty) stabilizations on the mesh skeleton:
    # - v_cip:  fluid velocity regularization (helps at moderate/high Re on coarse meshes)
    # - vS_cip: skeleton velocity regularization (helps near the diffuse interface)
    v_cip: float = 0.0,
    vS_cip: float = 0.0,
    # Optional augmented-Lagrangian / grad-div style stabilization:
    # adds γ_div * (div(C v + B vS), div(C w) + div(B η)) to the momentum/skeleton
    # equations. This is consistent (vanishes when the constraint holds) and can
    # improve conditioning of the transient convection-dominated solve.
    gamma_div: float = 0.0,
    D_alpha: float = 0.0,
    # Which velocity advects the diffuse indicator α:
    # - "vS"  (default): skeleton velocity v^S
    # - "v":            fluid velocity v
    # - "mix":          mixture/volume velocity F = C v + B v^S, with C=(1-α)+αφ and B=α(1-φ)
    # - "mix_biofilm":  like "mix", but gate the fluid part (Cv) by a smooth α-cutoff to
    #                   avoid advecting α through the pure-fluid (α≈0) region
    alpha_advect_with: str = "vS",
    # Parameters used by alpha_advect_with="mix_biofilm":
    # gate C by  g(α) = α^m / (α^m + α0^m). For α≫α0, g≈1; for α≪α0, g≈0.
    alpha_mix_gate_alpha0: float = 0.1,
    alpha_mix_gate_power: int = 4,
    # How to advect α by the chosen velocity field:
    # - "advective" (default): u·∇α   (indicator/level-set style; preserves α along characteristics)
    # - "conservative":        div(α u) = u·∇α + α div(u)
    alpha_advection_form: str = "advective",
    # Optional interface traction benchmark hook.
    dGamma=None,
    g_t_k=None,
    g_t_n=None,
    traction_weight_k=None,
    traction_weight_n=None,
    # Allen–Cahn / phase-field interface regularization for α
    alpha_cahn_M: float = 0.0,
    alpha_cahn_gamma: float = 0.0,
    alpha_cahn_eps: float = 1.0,
    # Conservative Allen–Cahn: mass-conserving via a global Lagrange multiplier λ_α
    alpha_cahn_conservative: bool = False,
    # Conservative Allen–Cahn implementation:
    # - "unknown" (default): include a global Lagrange multiplier λ_α as an unknown
    #   with the constraint equation ∫ M(α)(μ_α-λ_α)=0.
    # - "eliminate": treat λ_α as a dependent coefficient (computed externally,
    #   e.g. by projecting λ_α = (∫ M μ)/(∫ M)) to avoid ill-conditioning when
    #   using degenerate mobility.
    alpha_cahn_conservative_mode: str = "unknown",
    alpha_cahn_mobility: str = "constant",
    # Optional mobility floor for degenerate Allen–Cahn: M(α)=M0(α(1-α)+m_floor)
    # This can prevent complete "deactivation" of the phase-field regularization in
    # bulk regions when α drifts away from {0,1} due to numerical diffusion.
    alpha_cahn_mobility_floor: float = 0.0,
    lambda_alpha_k=None,
    lambda_alpha_n=None,
    dlambda_alpha=None,
    lambda_alpha_test=None,
    # Optional scaling for the conservative Allen–Cahn constraint equation
    # (improves conditioning when using degenerate mobility so ∫M is small).
    alpha_cahn_lambda_scale=None,
    # Cahn–Hilliard regularization for α (mass-conserving phase-field)
    alpha_ch_M: float = 0.0,
    alpha_ch_gamma: float = 0.0,
    alpha_ch_eps: float = 1.0,
    alpha_ch_mobility: str = "constant",
    # Crack propagation: additional surface speed term V_crack via δ(α)
    alpha_crack_k: float = 0.0,
    alpha_crack_Dc: float = 0.0,
    alpha_crack_m: float = 1.0,
    alpha_crack_gamma_kappa: float = 0.0,
    alpha_crack_eta_kappa: float = 1.0e-12,
    alpha_crack_eta_pos: float = 1.0e-12,
    alpha_crack_eta_mech: float = 1.0e-12,
    alpha_crack_driver: str = "shear",
    V_crack_prev=None,
    # transport stabilizations (consistent, for advection-dominated cases)
    alpha_supg: float = 0.0,
    alpha_cip: float = 0.0,
    u_cip: float = 0.0,
    u_cip_weight: str = "fluid",
    ds_cip=None,
    # Damage (bulk cohesion loss) model parameters
    damage_k: float = 0.0,
    damage_sigma_cr: float = 0.0,
    damage_m: float = 1.0,
    damage_D: float = 0.0,
    damage_gamma_out: float = 0.0,
    damage_eta_pos: float = 1.0e-12,
    damage_kappa_stiff: float = 1.0e-8,
    damage_kappa_perm: float = 1.0e-8,
    damage_model: str = "kinetic",
    damage_eta: float = 0.0,
    damage_Gc: float = 0.0,
    damage_l: float = 0.0,
    damage_psi0: float = 0.0,
    damage_pf_driver: str = "von_mises",
    # Optional (lagged) phase-field damage history field H^{prev} (prevents healing when the drive relaxes).
    damage_H_prev=None,
    damage_stiff_split: str = "full",
    D_S: float = 0.0,
    # Substrate reaction time discretization:
    # - "theta" (default): use the global θ-scheme (may be CN if θ=0.5)
    # - "implicit"/"imex": treat the reaction term fully implicitly (L-stable for stiff decay)
    # - "explicit": treat the reaction term explicitly (not recommended for stiff kinetics)
    substrate_reaction_scheme: str = "theta",
    # Substrate diffusion time discretization (same choices as reaction scheme).
    substrate_diffusion_scheme: str = "theta",
    D_X: float = 0.0,
    # growth / detachment parameters
    mu_max: float = 0.0,
    K_S: float = 1.0,
    k_g: float = 0.0,
    k_d: float = 0.0,
    Y: float = 1.0,
    rho_s_star: float = 1.0,
    k_det: float = 0.0,
    # modelling toggles
    mu_b_model: str = "phi_mu",
    dim: int = 2,
    # sources (all optional; default to 0)
    f_v=None,
    f_u=None,
    s_v=None,
    ds_v=None,
    f_phi=None,
    f_alpha=None,
    f_S=None,
    f_X=None,
    # detachment rate override (if given, used instead of shear-based lagged rate)
    D_det_prev=None,
    # optional wall adhesion traction (applied to skeleton on a boundary segment)
    ds_adh=None,
    adhesion_k_n: float = 0.0,
    adhesion_k_t: float = 0.0,
    adhesion_gamma_n: float = 0.0,
    adhesion_gamma_t: float = 0.0,
    adhesion_a_prev=None,
    # Solid velocity extension stabilization outside biofilm (α≈0):
    # Keep vS well-posed in a one-domain CG setting by adding a weak extension
    # penalty in the fluid region. By default, we mirror the u-extension settings.
    gamma_vS: float | None = None,
    vS_extension_mode: str | None = None,
    gamma_vS_pin: float | None = None,
    gamma_vS_pin_power: int = 2,
    # Scaling for the kinematic constraint (improves monolithic conditioning).
    kinematics_scale: float | None = None,
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
    elif solid_model_key in {"hencky", "hencky_log", "hencky_log_strain"}:
        if mu_s is None or lambda_s is None:
            raise ValueError("Solid model 'hencky' requires mu_s and lambda_s.")
    elif solid_model_key in {"stvk", "svk", "saint_venant_kirchhoff", "saint-venant-kirchhoff"}:
        if mu_s is None or lambda_s is None:
            raise ValueError("Solid model 'stvk' requires mu_s and lambda_s.")
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
    # NOTE: `ds_v` is the Gateaux derivative of the volumetric source `s_v`.
    # When no derivative is provided (or a Picard/frozen linearization is used),
    # we must still keep the Jacobian term `-alpha_k * ds_v` in the *trial* role.
    # Using a bare scalar `0` here can leave `alpha_k*0` tagged as a function-like
    # operand in the symbolic pipeline, which then breaks assembly when combined
    # with other trial-family contributions. Use an explicit zero-trial instead.
    if ds_v is None:
        ds_v = zero_scalar * dS
    else:
        # Some drivers pass a frozen/Picard linearization as `Constant(0.0)`.
        # Treat this as the intended "zero derivative" and keep it in the trial
        # role to avoid trial/function shape-mismatch errors during assembly.
        try:
            if float(ds_v) == 0.0:
                ds_v = zero_scalar * dS
        except Exception:
            pass
    f_phi = f_phi if f_phi is not None else zero_scalar
    f_alpha = f_alpha if f_alpha is not None else zero_scalar
    f_S = f_S if f_S is not None else zero_scalar
    f_X = f_X if f_X is not None else zero_scalar
    g_t_k = g_t_k if g_t_k is not None else zero_vector
    g_t_n = g_t_n if g_t_n is not None else g_t_k
    traction_weight_k = traction_weight_k if traction_weight_k is not None else zero_scalar
    traction_weight_n = traction_weight_n if traction_weight_n is not None else traction_weight_k

    # ------------------------------------------------------------------
    # Skeleton velocity is now a primary unknown (vS).
    # ------------------------------------------------------------------
    div_vS_k = div(vS_k)
    div_vS_n = div(vS_n)
    div_dvS = div(dvS)

    # ------------------------------------------------------------------
    # Optional bulk damage field d: cohesion loss / fracture-like weakening
    # ------------------------------------------------------------------
    use_damage = d_k is not None
    if use_damage:
        if d_n is None or dd is None or d_test is None:
            raise ValueError("d_k provided but one of (d_n, dd, d_test) is missing.")

        kappa_stiff = _c(float(damage_kappa_stiff))
        kappa_perm = _c(float(damage_kappa_perm))
        one_m_kappa_stiff = _one_minus(kappa_stiff)
        one_m_kappa_perm = _one_minus(kappa_perm)

        one_m_d_k = _one_minus(d_k)
        one_m_d_n = _one_minus(d_n)

        # Miehe-type regularized degradation:
        #   g(d) = (1 - κ) (1 - d)^2 + κ,
        # so g(0)=1 and g(1)=κ.
        g_stiff_k = one_m_kappa_stiff * (one_m_d_k * one_m_d_k) + kappa_stiff
        g_stiff_n = one_m_kappa_stiff * (one_m_d_n * one_m_d_n) + kappa_stiff
        dg_stiff_k = (-_c(2.0) * one_m_kappa_stiff * one_m_d_k) * dd

        g_perm_k = one_m_kappa_perm * (one_m_d_k * one_m_d_k) + kappa_perm
        g_perm_n = one_m_kappa_perm * (one_m_d_n * one_m_d_n) + kappa_perm
        dg_perm_k = (-_c(2.0) * one_m_kappa_perm * one_m_d_k) * dd
    else:
        g_stiff_k = _c(1.0)
        g_stiff_n = _c(1.0)
        dg_stiff_k = _c(0.0)
        g_perm_k = _c(1.0)
        g_perm_n = _c(1.0)
        dg_perm_k = _c(0.0)

    # ------------------------------------------------------------------
    # Coefficients (at n/k) and their variations (at k only)
    # ------------------------------------------------------------------
    rho_k = _rho(alpha_k, phi_k, rho_f=rho_f)
    rho_n = _rho(alpha_n, phi_n, rho_f=rho_f)

    mu_b_key = str(mu_b_model).strip().lower()
    mu_b_expr = mu_b
    if mu_b_expr is not None and not hasattr(mu_b_expr, "dim"):
        mu_b_expr = _c(float(mu_b_expr))

    mu_k = _mu(alpha_k, phi_k, mu_f=mu_f, mu_b=mu_b_expr, mu_b_model=mu_b_model)
    mu_n = _mu(alpha_n, phi_n, mu_f=mu_f, mu_b=mu_b_expr, mu_b_model=mu_b_model)

    # Coefficient variations w.r.t (α,φ) at k:
    dC = (phi_k - _c(1.0)) * dalpha + alpha_k * dphi  # δ((1-α)+αφ)
    drho = rho_f * dC

    if mu_b_key in {"mu", "const", "constant"}:
        dmu = _c(0.0) * dphi
    elif mu_b_key in {"phi_mu", "phi*mu", "phi"}:
        dmu = mu_f * dC
    elif mu_b_key in {"alpha_mu", "alpha", "mu_b", "biofilm_mu", "biofilm"}:
        if mu_b_expr is None:
            raise ValueError("mu_b_model='alpha_mu' requires mu_b to be provided.")
        dmu = (mu_b_expr - mu_f) * dalpha
    elif mu_b_key in {"alpha_phi_mu", "alpha_phi", "alpha*phi_mu", "alpha*phi*mu"}:
        if mu_b_expr is None:
            raise ValueError("mu_b_model='alpha_phi_mu' requires mu_b to be provided.")
        dmu = (phi_k * mu_b_expr - mu_f) * dalpha + (alpha_k * mu_b_expr) * dphi
    else:
        raise ValueError(f"Unknown mu_b_model {mu_b_model!r}.")

    # Optional skeleton inertia coefficient (0 by default).
    rho_s0_tilde = rho_s0_tilde if rho_s0_tilde is not None else zero_scalar

    # ------------------------------------------------------------------
    # Permeability / drag handling
    # ------------------------------------------------------------------
    # Drag coefficient (without κ^{-1}): α μ_f φ^2, optionally degraded by damage.
    # The damage-dependent factor g_perm(d) drives β→0 in failed/cracked regions so
    # the mixture becomes hydraulically "open" (fluid-like) while α can remain a
    # material/occupancy indicator.
    beta_coeff_k = alpha_k * mu_f * (phi_k * phi_k) * g_perm_k
    beta_coeff_n = alpha_n * mu_f * (phi_n * phi_n) * g_perm_n
    dbeta_coeff = mu_f * ((phi_k * phi_k) * dalpha + (_c(2.0) * alpha_k * phi_k) * dphi) * g_perm_k
    if use_damage:
        dbeta_coeff += alpha_k * mu_f * (phi_k * phi_k) * dg_perm_k

    kappa_inv_key = str(kappa_inv_model).strip().lower()

    kc_keys = {
        "kozeny",
        "kozeny_carman",
        "kozeny-carman",
        "kozeny_carman_phi",
        "kc",
    }

    def _kozeny_carman_scale(phi, *, phi_ref, eps_kc):
        # Kozeny–Carman scaling (dimensionless) for inverse permeability:
        #   k^{-1}(phi) ∝ (1-phi)^2 / phi^3
        # We normalize by phi_ref so that k^{-1}(phi_ref) = kappa_inv.
        one_m = _one_minus(phi)
        num = one_m * one_m
        den = phi * phi * phi + eps_kc
        g = num / den

        one_m0 = _one_minus(phi_ref)
        g0 = (one_m0 * one_m0) / (phi_ref * phi_ref * phi_ref + eps_kc)
        return g / g0

    def _d_kozeny_carman_scale(phi, dphi, *, phi_ref, eps_kc):
        # Linearization of the normalized scale: δ[ g(phi)/g(phi_ref) ].
        # g(phi) = (1-phi)^2 / (phi^3+eps)
        one_m = _one_minus(phi)
        num = one_m * one_m
        den = phi * phi * phi + eps_kc

        one_m0 = _one_minus(phi_ref)
        g0 = (one_m0 * one_m0) / (phi_ref * phi_ref * phi_ref + eps_kc)

        # dg/dphi = (num' den - num den') / den^2,  num'=-2(1-phi), den'=3 phi^2
        dg_dphi = ((-_c(2.0) * one_m) * den - num * (_c(3.0) * phi * phi)) / (den * den)
        return (dg_dphi / g0) * dphi

    # Fast/robust path for *scalar spatial* κ^{-1}: keep the legacy scalar formulation.
    if kappa_inv_key in {"spatial", "constant", "const"} and getattr(kappa_inv, "dim", None) == 0:
        drag_mode = "scalar"
        beta_k = beta_coeff_k * kappa_inv
        beta_n = beta_coeff_n * kappa_inv
        dbeta = dbeta_coeff * kappa_inv
    elif kappa_inv_key in kc_keys and getattr(kappa_inv, "dim", None) == 0:
        # Kozeny–Carman: permeability depends on porosity (phi).
        drag_mode = "scalar"

        phi_ref_val = float(kappa_inv_phi_ref)
        if not (0.0 < phi_ref_val < 1.0):
            raise ValueError(f"kappa_inv_phi_ref must be in (0,1); got {phi_ref_val}.")
        phi_ref_c = _c(phi_ref_val)
        eps_kc = _c(float(kappa_inv_kc_eps))

        scale_k = _kozeny_carman_scale(phi_k, phi_ref=phi_ref_c, eps_kc=eps_kc)
        scale_n = _kozeny_carman_scale(phi_n, phi_ref=phi_ref_c, eps_kc=eps_kc)
        dscale_k = _d_kozeny_carman_scale(phi_k, dphi, phi_ref=phi_ref_c, eps_kc=eps_kc)

        k_inv_k = kappa_inv * scale_k
        k_inv_n = kappa_inv * scale_n
        dk_inv_k = kappa_inv * dscale_k

        beta_k = beta_coeff_k * k_inv_k
        beta_n = beta_coeff_n * k_inv_n
        dbeta = dbeta_coeff * k_inv_k + beta_coeff_k * dk_inv_k
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
        elif kappa_inv_key in kc_keys:
            # Kozeny–Carman scaling applied as an isotropic factor to the (possibly anisotropic) K_inv.
            phi_ref_val = float(kappa_inv_phi_ref)
            if not (0.0 < phi_ref_val < 1.0):
                raise ValueError(f"kappa_inv_phi_ref must be in (0,1); got {phi_ref_val}.")
            phi_ref_c = _c(phi_ref_val)
            eps_kc = _c(float(kappa_inv_kc_eps))

            scale_k = _kozeny_carman_scale(phi_k, phi_ref=phi_ref_c, eps_kc=eps_kc)
            scale_n = _kozeny_carman_scale(phi_n, phi_ref=phi_ref_c, eps_kc=eps_kc)
            dscale_k = _d_kozeny_carman_scale(phi_k, dphi, phi_ref=phi_ref_c, eps_kc=eps_kc)

            k_inv_k = scale_k * K_inv
            k_inv_n = scale_n * K_inv
            dk_inv_k = dscale_k * K_inv
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
        D_det_prev = _detachment_from_shear_prev(
            v_prev=v_n, alpha_prev=alpha_n, mu_prev=mu_n, k_det=_c(float(k_det)), eta_n=1.0e-12, dim=dim
        )

    # ------------------------------------------------------------------
    # Shared helper quantities for conservative forms (expanded divergence)
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

    # Jacobian helpers for divCv_k and divBvS_k (k-part only)
    dC_k = (phi_k - _c(1.0)) * dalpha + alpha_k * dphi
    dB_k = _one_minus(phi_k) * dalpha - alpha_k * dphi

    # NOTE: Keep GradOpInfo on the left in scalar×grad products (trial×grad(function)
    # is not implemented, but grad(function)×trial is).
    dgradC_k = grad(dalpha) * (phi_k - _c(1.0)) + grad(alpha_k) * dphi + grad(phi_k) * dalpha + grad(dphi) * alpha_k
    dgradB_k = grad(dalpha) * _one_minus(phi_k) - grad(alpha_k) * dphi - grad(phi_k) * dalpha - grad(dphi) * alpha_k

    d_divCv_k = dC_k * div(v_k) + C_k * div(dv) + dot(dgradC_k, v_k) + dot(gradC_k, dv)
    d_divBvS_k = dB_k * div_vS_k + B_k * div_dvS + dot(dgradB_k, vS_k) + dot(gradB_k, dvS)

    # Divergence of the test fluxes and their coefficient variations.
    div_C_vtest_k = C_k * div(v_test) + dot(gradC_k, v_test)
    div_B_vStest_k = B_k * div(vS_test) + dot(gradB_k, vS_test)

    # Expand dgrad·test component-wise to avoid backend-dependent contraction
    # paths for Grad(trial-scalar) · VectorTest.
    d_div_C_vtest_k = dC_k * div(v_test)
    d_div_B_vStest_k = dB_k * div(vS_test)
    if int(dim) == 2:
        dgradC_k_x = (phi_k - _c(1.0)) * grad(dalpha)[0] + dphi * grad(alpha_k)[0] + dalpha * grad(phi_k)[0] + alpha_k * grad(dphi)[0]
        dgradC_k_y = (phi_k - _c(1.0)) * grad(dalpha)[1] + dphi * grad(alpha_k)[1] + dalpha * grad(phi_k)[1] + alpha_k * grad(dphi)[1]
        d_div_C_vtest_k += dgradC_k_x * v_test[0] + dgradC_k_y * v_test[1]

        dgradB_k_x = _one_minus(phi_k) * grad(dalpha)[0] - dphi * grad(alpha_k)[0] - dalpha * grad(phi_k)[0] - alpha_k * grad(dphi)[0]
        dgradB_k_y = _one_minus(phi_k) * grad(dalpha)[1] - dphi * grad(alpha_k)[1] - dalpha * grad(phi_k)[1] - alpha_k * grad(dphi)[1]
        d_div_B_vStest_k += dgradB_k_x * vS_test[0] + dgradB_k_y * vS_test[1]
    else:
        d_div_C_vtest_k += dot(dgradC_k, v_test)
        d_div_B_vStest_k += dot(dgradB_k, vS_test)

    # ------------------------------------------------------------------
    # (i) Momentum: conservative Navier–Stokes–Brinkman
    # ------------------------------------------------------------------
    momdot = (rho_k * v_k - rho_n * v_n) * inv_dt
    fluid_conv_key = str(fluid_convection).strip().lower()
    if fluid_conv_key in {"explicit"}:
        fluid_conv_key = "imex"
    if fluid_conv_key not in {"full", "lagged", "imex", "off"}:
        raise ValueError(
            f"Unknown fluid_convection={fluid_convection!r}. Use 'full' (default), 'lagged', 'imex', or 'off'."
        )

    # Conservative convection: div(ρ v⊗v) = ρ (v·∇)v + v div(ρ v), with ρ=ρ_f C.
    div_rhov_k = rho_f * divCv_k
    div_rhov_n = rho_f * divCv_n

    r_mom = inner(momdot, v_test) * dx
    if fluid_conv_key == "full":
        conv_k = dot(dot(grad(v_k), v_k), v_test)
        conv_n = dot(dot(grad(v_n), v_n), v_test)
        r_mom += th * (rho_k * conv_k + div_rhov_k * dot(v_k, v_test)) * dx
        r_mom += one_m_th * (rho_n * conv_n + div_rhov_n * dot(v_n, v_test)) * dx
    elif fluid_conv_key == "lagged":
        # Picard/IMEX-like linearization:
        #   div(ρ^n v^n ⊗ v^k) = ρ^n (v^n·∇)v^k + v^k div(ρ^n v^n)
        conv_k = dot(dot(grad(v_k), v_n), v_test)
        conv_n = dot(dot(grad(v_n), v_n), v_test)
        r_mom += th * (rho_n * conv_k + div_rhov_n * dot(v_k, v_test)) * dx
        r_mom += one_m_th * (rho_n * conv_n + div_rhov_n * dot(v_n, v_test)) * dx
    elif fluid_conv_key == "imex":
        # IMEX-style explicit convection: treat the convective term fully at the n-level
        # (no v^k dependence). This removes the non-symmetric convection block from the
        # Jacobian and can significantly improve robustness for long transient runs.
        conv_k = None
        conv_n = dot(dot(grad(v_n), v_n), v_test)
        r_mom += (rho_n * conv_n + div_rhov_n * dot(v_n, v_test)) * dx
    else:
        conv_k = None
    r_mom += _c(2.0) * th * mu_k * inner(_epsilon(v_k), _epsilon(v_test)) * dx
    r_mom += _c(2.0) * one_m_th * mu_n * inner(_epsilon(v_n), _epsilon(v_test)) * dx
    # Pressure term for the mixture constraint div(C v + B vS)=... :
    # variationally consistent fluid coupling is C grad(p), i.e.
    #   -(p, div(C w)) = -p (C div(w) + grad(C)·w)
    # which is the exact adjoint of the C v part of the constraint.
    r_mom += -p_k * div_C_vtest_k * dx
    if float(gamma_div) != 0.0:
        gamma_div_c = _c(float(gamma_div))
        # Consistent augmented-Lagrangian / grad-div stabilization for the mixture
        # volume constraint div(F)=0 with F=C v + B vS.
        r_mom += gamma_div_c * divF_k * div_C_vtest_k * dx
    if drag_mode == "scalar":
        r_mom += beta_k * dot(v_k, v_test) * dx
        r_mom += -beta_k * dot(vS_k, v_test) * dx
    else:
        r_mom += beta_coeff_k * dot(kdrag_k, v_test) * dx
    if dGamma is not None:
        r_mom += -(th * _dot_2d_components(g_t_k, v_test) + one_m_th * _dot_2d_components(g_t_n, v_test)) * dGamma
    if traction_weight_k is not None or traction_weight_n is not None:
        r_mom += -(
            th * traction_weight_k * _dot_2d_components(g_t_k, v_test)
            + one_m_th * traction_weight_n * _dot_2d_components(g_t_n, v_test)
        ) * dx
    r_mom += -dot(f_v, v_test) * dx

    a_mom = inv_dt * (drho * dot(v_k, v_test) + rho_k * dot(dv, v_test)) * dx

    if fluid_conv_key == "full":
        a_mom += th * (
            drho * conv_k
            + rho_k * dot(dot(grad(dv), v_k), v_test)
            + rho_k * dot(dot(grad(v_k), dv), v_test)
        ) * dx
        # Jacobian of the conservative correction v div(ρ v).
        # Keep trial-family contributions separated to avoid mixed-role metadata
        # leakage in the assembler (v-trial vs alpha/phi-trial pieces).
        d_divCv_k_ap = dC_k * div(v_k) + dot(dgradC_k, v_k)
        d_divCv_k_v = C_k * div(dv) + dot(gradC_k, dv)
        a_mom += th * (rho_f * d_divCv_k_ap * dot(v_k, v_test)) * dx
        a_mom += th * (rho_f * d_divCv_k_v * dot(v_k, v_test)) * dx
        a_mom += th * (div_rhov_k * dot(dv, v_test)) * dx
    elif fluid_conv_key == "lagged":
        a_mom += th * (rho_n * dot(dot(grad(dv), v_n), v_test)) * dx
        a_mom += th * (div_rhov_n * dot(dv, v_test)) * dx
    elif fluid_conv_key == "imex":
        # Explicit convection contributes no Jacobian terms.
        pass
    a_mom += _c(2.0) * th * (dmu * inner(_epsilon(v_k), _epsilon(v_test)) + mu_k * inner(_epsilon(dv), _epsilon(v_test))) * dx

    # δ[-p div(C w)] = -(δp) div(C w) - p δ(div(C w)),
    # with δ(div(C w)) = δC div(w) + δgrad(C)·w for fixed test w.
    a_mom += -(dp * div_C_vtest_k + p_k * d_div_C_vtest_k) * dx
    if float(gamma_div) != 0.0:
        gamma_div_c = _c(float(gamma_div))
        d_divF_k = d_divCv_k + d_divBvS_k
        a_mom += gamma_div_c * (d_divF_k * div_C_vtest_k + divF_k * d_div_C_vtest_k) * dx
    if drag_mode == "scalar":
        a_mom += dbeta * (dot(v_k, v_test) - dot(vS_k, v_test)) * dx
        a_mom += beta_k * dot(dv, v_test) * dx
        a_mom += -beta_k * dot(dvS, v_test) * dx
    else:
        a_mom += dbeta_coeff * dot(kdrag_k, v_test) * dx
        a_mom += beta_coeff_k * dot(dkdrag_k, v_test) * dx

    # Optional SUPG-like streamline diffusion for the fluid convection term.
    #
    # This is a conservative, easy-to-linearize variant:
    #   τ ( (v^n·∇)v^k , (v^n·∇)w ) in the fluid region.
    #
    # It is primarily intended as a robustness knob for long transient runs.
    if float(v_supg) != 0.0:
        rho_f_val = None
        try:
            rho_f_val = float(rho_f)
        except Exception:
            rho_f_val = None

        # If rho_f==0 then inertia/convection vanishes and this stabilization is irrelevant.
        # Avoid dividing by rho_f in that case.
        if rho_f_val is not None and abs(rho_f_val) < 1.0e-16:
            pass
        else:
            h_v = MeshSize()
            vmag2 = v_n[0] * v_n[0] + v_n[1] * v_n[1]
            vmag = _sqrt(vmag2 + _c(1.0e-12))
            # Standard SUPG scaling: τ ~ h^2 / (ν + h|v| + h^2/dt).
            nu_f = mu_f / rho_f
            denom = _c(6.0) * nu_f + h_v * vmag + (h_v * h_v) * inv_dt
            tau_v = _c(float(v_supg)) * (h_v * h_v) / (denom + _c(1.0e-16))
            w_v = _one_minus(alpha_n)  # lagged "fluid-only" localization
            adv_v_k = dot(grad(v_k), v_n)
            adv_w = dot(grad(v_test), v_n)
            r_mom += tau_v * w_v * inner(adv_v_k, adv_w) * dx
            a_mom += tau_v * w_v * inner(dot(grad(dv), v_n), adv_w) * dx

    # Optional CIP (continuous interior penalty) stabilization for the fluid velocity.
    # This is a consistent high-frequency regularization that can improve robustness on coarse meshes.
    if float(v_cip) != 0.0 and ds_cip is not None:
        n_int = FacetNormal()
        h_F = avg(MeshSize())
        tau_cip = _c(float(v_cip)) * (h_F * h_F * h_F) * inv_dt
        w_v = avg(_one_minus(alpha_n))
        r_mom += tau_cip * w_v * _grad_inner_jump(v_k, v_test, n_int) * ds_cip
        a_mom += tau_cip * w_v * _grad_inner_jump(dv, v_test, n_int) * ds_cip

    # ------------------------------------------------------------------
    # (ii) Mass / volume constraint (expanded divergence)
    # ------------------------------------------------------------------
    # IMPORTANT (time discretization):
    #
    # This is an algebraic (DAE) constraint whose Lagrange multiplier is `p`.
    # Using a θ-average of *only* the divergence while keeping the source fully
    # implicit leads to a mixed time level for the constraint. For θ<1 (e.g.
    # Crank–Nicolson), this can introduce large explicit forcing from the
    # previous step and destabilize the coupled solve (in particular when
    # coupled to stiff substrate kinetics).
    #
    # We therefore enforce the constraint fully implicitly at the k-level:
    #     div(F_k) = α_k s_v(k),
    # consistent with taking the pressure coupling terms at the k-level.
    r_mass = q_test * (divF_k - alpha_k * s_v) * dx

    # Jacobian of divF_k (k-part only)
    # δC = (φ-1) δα + α δφ
    # δ(α s_v) = (δα) s_v + α (δs_v).
    a_mass = q_test * (d_divCv_k + d_divBvS_k - dalpha * s_v - alpha_k * ds_v) * dx

    # ------------------------------------------------------------------
    # (iii) Skeleton momentum (optional inertia + linear/neo-Hookean stress)
    # ------------------------------------------------------------------
    damage_stiff_split_key = str(damage_stiff_split).strip().lower()
    use_miehe_stiff_split = bool(
        use_damage
        and damage_stiff_split_key in {"miehe", "tensile", "tension_compression", "tension-compression", "tc"}
    )

    if use_miehe_stiff_split and solid_model_key not in {
        "linear",
        "small_strain",
        "linear_elastic",
        "hencky",
        "hencky_log",
        "hencky_log_strain",
        "stvk",
        "svk",
        "saint_venant_kirchhoff",
        "saint-venant-kirchhoff",
    }:
        raise ValueError(
            "damage_stiff_split='miehe' is currently only implemented for solid_model in "
            "{'linear','stvk','hencky'} (2D)."
        )
    if use_miehe_stiff_split and int(dim) != 2:
        raise ValueError("damage_stiff_split='miehe' is currently only implemented for dim=2.")

    # Elastic residual/Jacobian contributions.
    #
    # - Default: full-stress degradation via scalar g_stiff(d) multiplier.
    # - Optional: Miehe (tension/compression) split for linear elasticity:
    #     σ = g(d) σ⁺(u) + σ⁻(u),  with σ⁺ built from the positive principal strains.
    #
    # - For finite-strain Hencky hyperelasticity, we implement the classic
    #   Miehe-style tension/compression split at the *energy* level using the
    #   spatial Hencky strain E = log(V). This yields:
    #     ψ⁺ = μ||E⁺||² + (λ/2)⟨tr E⟩₊²,
    #   and the corresponding Cauchy-stress split σ = σ⁺ + σ⁻ is obtained by
    #   differentiating the split energy (Kirchhoff stress conjugate to log strain).
    if solid_model_key in {"hencky", "hencky_log", "hencky_log_strain"} and use_miehe_stiff_split:
        eta_pos = float(damage_eta_pos)
        sig_plus_k, sig_minus_k = sigma_hencky_miehe_split(
            u_k, mu_s, lambda_s, dim=int(dim), eta_pos=eta_pos
        )
        sig_plus_n, sig_minus_n = sigma_hencky_miehe_split(
            u_n, mu_s, lambda_s, dim=int(dim), eta_pos=eta_pos
        )
        dsig_plus_k, dsig_minus_k = dsigma_hencky_miehe_split(
            u_k, du, mu_s, lambda_s, dim=int(dim), eta_pos=eta_pos
        )

        r_el_plus_k = inner(sig_plus_k, grad(vS_test))
        r_el_minus_k = inner(sig_minus_k, grad(vS_test))
        r_el_plus_n = inner(sig_plus_n, grad(vS_test))
        r_el_minus_n = inner(sig_minus_n, grad(vS_test))

        a_el_plus = inner(dsig_plus_k, grad(vS_test))
        a_el_minus = inner(dsig_minus_k, grad(vS_test))
    elif solid_model_key in {"stvk", "svk", "saint_venant_kirchhoff", "saint-venant-kirchhoff"} and use_miehe_stiff_split:
        eta_pos = float(damage_eta_pos)
        disc_reg = 1.0e-16
        sig_plus_k, sig_minus_k = sigma_svk_miehe_split(
            u_k, mu_s, lambda_s, dim=int(dim), eta_pos=eta_pos, disc_reg=disc_reg
        )
        sig_plus_n, sig_minus_n = sigma_svk_miehe_split(
            u_n, mu_s, lambda_s, dim=int(dim), eta_pos=eta_pos, disc_reg=disc_reg
        )
        dsig_plus_k, dsig_minus_k = dsigma_svk_miehe_split(
            u_k, du, mu_s, lambda_s, dim=int(dim), eta_pos=eta_pos, disc_reg=disc_reg
        )

        r_el_plus_k = inner(sig_plus_k, grad(vS_test))
        r_el_minus_k = inner(sig_minus_k, grad(vS_test))
        r_el_plus_n = inner(sig_plus_n, grad(vS_test))
        r_el_minus_n = inner(sig_minus_n, grad(vS_test))

        a_el_plus = inner(dsig_plus_k, grad(vS_test))
        a_el_minus = inner(dsig_minus_k, grad(vS_test))
    elif solid_model_key in {"linear", "small_strain", "linear_elastic"} and use_miehe_stiff_split:
        eta_pos = float(damage_eta_pos)
        disc_reg = 1.0e-16
        I = Identity(int(dim))

        # --- k-level split ---
        E_k = _epsilon(u_k)
        E_plus_k, E_minus_k, _, _, _ = spectral_positive_part_2x2_sym(E_k, eta_pos=eta_pos, disc_reg=disc_reg)
        trE_k = div(u_k)
        trE_pos_k = _smooth_pos_u(trE_k, eta=eta_pos)

        sig_plus_k = lambda_s * trE_pos_k * I + _c(2.0) * mu_s * E_plus_k
        sig_minus_k = lambda_s * (trE_k - trE_pos_k) * I + _c(2.0) * mu_s * E_minus_k

        r_el_plus_k = inner(sig_plus_k, grad(vS_test))
        r_el_minus_k = inner(sig_minus_k, grad(vS_test))

        # --- n-level split (lagged, no Jacobian contribution) ---
        E_n = _epsilon(u_n)
        E_plus_n, E_minus_n, _, _, _ = spectral_positive_part_2x2_sym(E_n, eta_pos=eta_pos, disc_reg=disc_reg)
        trE_n = div(u_n)
        trE_pos_n = _smooth_pos_u(trE_n, eta=eta_pos)

        sig_plus_n = lambda_s * trE_pos_n * I + _c(2.0) * mu_s * E_plus_n
        sig_minus_n = lambda_s * (trE_n - trE_pos_n) * I + _c(2.0) * mu_s * E_minus_n

        r_el_plus_n = inner(sig_plus_n, grad(vS_test))
        r_el_minus_n = inner(sig_minus_n, grad(vS_test))

        # --- consistent Jacobian: Gateaux derivatives ---
        dE = _epsilon(du)
        dE_plus = d_spectral_positive_part_2x2_sym(E_k, dE, eta_pos=eta_pos, disc_reg=disc_reg)
        dtrE = div(du)
        dtrE_pos = _smooth_pos_u_prime(trE_k, eta=eta_pos) * dtrE

        dsig_plus_k = lambda_s * dtrE_pos * I + _c(2.0) * mu_s * dE_plus
        dsig_minus_k = lambda_s * (dtrE - dtrE_pos) * I + _c(2.0) * mu_s * (dE - dE_plus)

        a_el_plus = inner(dsig_plus_k, grad(vS_test))
        a_el_minus = inner(dsig_minus_k, grad(vS_test))
    else:
        # Full-stress (legacy) elastic residual/Jacobian.
        if solid_model_key in {"linear", "small_strain", "linear_elastic"}:
            r_el_k = _linear_elastic_term(u_k, vS_test, mu_s=mu_s, lambda_s=lambda_s)
            r_el_n = _linear_elastic_term(u_n, vS_test, mu_s=mu_s, lambda_s=lambda_s)
            a_el = _linear_elastic_term(du, vS_test, mu_s=mu_s, lambda_s=lambda_s)
        elif solid_model_key in {"hencky", "hencky_log", "hencky_log_strain"}:
            sig_k = sigma_hencky(u_k, mu_s, lambda_s, dim=int(dim))
            sig_n = sigma_hencky(u_n, mu_s, lambda_s, dim=int(dim))
            r_el_k = inner(sig_k, grad(vS_test))
            r_el_n = inner(sig_n, grad(vS_test))
            dsig_k = dsigma_hencky(u_k, du, mu_s, lambda_s, dim=int(dim))
            a_el = inner(dsig_k, grad(vS_test))
        elif solid_model_key in {"stvk", "svk", "saint_venant_kirchhoff", "saint-venant-kirchhoff"}:
            sig_k = sigma_svk(u_k, mu_s, lambda_s, dim=int(dim))
            sig_n = sigma_svk(u_n, mu_s, lambda_s, dim=int(dim))
            r_el_k = inner(sig_k, grad(vS_test))
            r_el_n = inner(sig_n, grad(vS_test))
            dsig_k = dsigma_svk(u_k, du, mu_s, lambda_s, dim=int(dim))
            a_el = inner(dsig_k, grad(vS_test))
        else:
            # Eulerian reference-map Neo-Hookean stress (Cauchy), compatible with FPI poro Eulerian module.
            if c_nh is None:
                c_nh = mu_s / _c(2.0)
            if beta_nh is None:
                beta_nh = lambda_s / (_c(2.0) * mu_s)

            sig_k = sigma_neo_hookean(u_k, c_nh, beta_nh, dim=int(dim))
            sig_n = sigma_neo_hookean(u_n, c_nh, beta_nh, dim=int(dim))
            r_el_k = inner(sig_k, grad(vS_test))
            r_el_n = inner(sig_n, grad(vS_test))
            dsig_k = dsigma_neo_hookean(u_k, du, c_nh, beta_nh, dim=int(dim))
            a_el = inner(dsig_k, grad(vS_test))

        r_el_plus_k = r_el_k
        r_el_minus_k = _c(0.0)
        r_el_plus_n = r_el_n
        r_el_minus_n = _c(0.0)
        a_el_plus = a_el
        a_el_minus = _c(0.0)

    # Pressure coupling from the B vS part of the constraint:
    #   -(p, div(B η)) = -p (B div(η) + grad(B)·η)
    # Use dot(gradB, eta_test) ordering for backend compatibility.
    div_B_vStest_k = B_k * div(vS_test) + dot(gradB_k, vS_test)
    div_B_vStest_n = B_n * div(vS_test) + dot(gradB_n, vS_test)
    r_skel_press_k = -p_k * div_B_vStest_k
    r_skel_press_n = -p_n * div_B_vStest_n
    if float(gamma_div) != 0.0:
        gamma_div_c = _c(float(gamma_div))
        # Consistent augmented-Lagrangian stabilization for the mixture constraint
        # div(F)=0 with F=C v + B vS. The vS variation contributes div(B η).
        r_skel_press_k += gamma_div_c * divF_k * div_B_vStest_k

    # drag reaction: -β (v - vS)
    # Since beta already contains α, if we use alpha again then it would square and it won't 
    # be equal to the drag from the momentum of the fluid.
    if drag_mode == "scalar":
        r_skel_drag_k = -beta_k * dot(v_k - vS_k, vS_test)
        r_skel_drag_n = -beta_n * dot(v_n - vS_n, vS_test)
    else:
        r_skel_drag_k = -beta_coeff_k * dot(kdrag_k, vS_test)
        r_skel_drag_n = -beta_coeff_n * dot(kdrag_n, vS_test)

    # Time discretization for the (quasi-static) skeleton momentum balance.
    #
    # When `include_skeleton_acceleration=False`, the skeleton equation is an
    # algebraic equilibrium (no time derivative). A θ-average would introduce
    # explicit forcing from the previous step unless *all* terms (including the
    # pressure coupling) are treated consistently. For robustness, we therefore
    # evaluate the quasi-static balance fully at the k-level.
    sk_th = th if bool(include_skeleton_acceleration) else _c(1.0)
    sk_one_m_th = one_m_th if bool(include_skeleton_acceleration) else _c(0.0)

    r_skeleton = (
        sk_th * alpha_k * (g_stiff_k * r_el_plus_k + r_el_minus_k)
        + sk_one_m_th * alpha_n * (g_stiff_n * r_el_plus_n + r_el_minus_n)
        + sk_th * r_skel_press_k
        + sk_one_m_th * r_skel_press_n
        + sk_th * r_skel_drag_k
        + sk_one_m_th * r_skel_drag_n
    ) * dx
    # External body force is weighted by biofilm presence α, but not degraded by g_stiff(d).
    r_skeleton += -dot(alpha_k * f_u, vS_test) * dx
    if dGamma is not None:
        r_skeleton += (th * _dot_2d_components(g_t_k, vS_test) + one_m_th * _dot_2d_components(g_t_n, vS_test)) * dGamma
    if traction_weight_k is not None or traction_weight_n is not None:
        r_skeleton += (
            th * traction_weight_k * _dot_2d_components(g_t_k, vS_test)
            + one_m_th * traction_weight_n * _dot_2d_components(g_t_n, vS_test)
        ) * dx

    # Extension / stabilization coefficients.
    # - `gamma_u` controls u-extension in the kinematic constraint below.
    # - vS also needs an extension penalty in the free-fluid region (α≈0) to keep
    #   the one-domain CG formulation well-posed (otherwise vS DOFs in pure fluid
    #   elements can become weakly constrained / singular).
    gamma_u_c = _c(float(gamma_u))
    gamma_vS_eff = float(gamma_u) if gamma_vS is None else float(gamma_vS)
    vS_ext_mode = str(u_extension_mode if vS_extension_mode is None else vS_extension_mode).strip().lower()
    gamma_vS_c = _c(float(gamma_vS_eff))
    # Mirror u-extension pinning by default: both u-extension (grad-mode) and the
    # vS-extension (grad-mode) have a global-translation nullspace in a one-domain
    # CG setting. A tiny L2 pin in the fluid region breaks the nullspace and
    # materially improves Newton robustness without affecting the biofilm region
    # (the pin weight scales like (1-α)^2).
    if gamma_vS_pin is None and vS_ext_mode in {"grad", "h1"} and float(gamma_u_pin) != 0.0:
        gamma_vS_pin_eff = float(gamma_u_pin)
    else:
        gamma_vS_pin_eff = 0.0 if gamma_vS_pin is None else float(gamma_vS_pin)
    gamma_vS_pin_c = _c(float(gamma_vS_pin_eff))
    vS_pin_pow = int(gamma_vS_pin_power)
    if vS_pin_pow < 1:
        raise ValueError(f"gamma_vS_pin_power must be >= 1; got {gamma_vS_pin_power}.")
    a_skel_visco_alpha = None
    a_skel_visco_vS = None
    if float(solid_visco_eta) != 0.0:
        eta_s_c = _c(float(solid_visco_eta))
        # grad_vS_k = (grad(u_k) - grad(u_n))/dt
        # eps_vS_k = 0.5*(grad_vS_k + grad_vS_k.T)
        sig_visc_k = _c(2.0) * eta_s_c * _epsilon(vS_k)
        sig_visc_n = _c(2.0) * eta_s_c * _epsilon(vS_n)
        r_visc_k = inner(sig_visc_k, grad(vS_test))
        r_visc_n = inner(sig_visc_n, grad(vS_test))
        # Treat viscosity as part of the skeleton response: localize by α and apply the same stiffness
        # degradation g_stiff(d) used for elasticity.
        r_skeleton += (th * alpha_k * g_stiff_k * r_visc_k + one_m_th * alpha_n * g_stiff_n * r_visc_n) * dx

        # Consistent k-part Jacobian: δ[α g σ_visc(vS)] = δ(α g) r_visc + α g a_visc.
        #
        # IMPORTANT: keep trial-family contributions separated. Mixing dalpha (and/or dd)
        # with dvS in a single integrand sum can trigger VecOpInfo shape-mismatch errors
        # in the current compiler/assembler pipeline.
        sig_dvisc = _c(2.0) * eta_s_c * _epsilon(dvS)
        a_visc = inner(sig_dvisc, grad(vS_test))
        if use_damage:
            w_ag = dalpha * g_stiff_k + alpha_k * dg_stiff_k
        else:
            # Avoid mixing trial and function roles in the symbolic pipeline:
            # when damage is disabled, dg_stiff_k is exactly 0 but still carries
            # "function" metadata, and (trial + 0*function) can break assembly.
            w_ag = dalpha * g_stiff_k
        a_skel_visco_alpha = sk_th * (w_ag * r_visc_k) * dx
        a_skel_visco_vS = sk_th * (alpha_k * g_stiff_k * a_visc) * dx

    # Optional extension penalty for vS in the free-fluid region.
    if float(gamma_vS_eff) != 0.0:
        if vS_ext_mode in {"l2", "mass"}:
            h_u = MeshSize()
            inv_h2 = _c(1.0) / (h_u * h_u)
            r_skeleton += gamma_vS_c * inv_h2 * _one_minus(alpha_k) * dot(vS_k, vS_test) * dx
        elif vS_ext_mode in {"grad", "h1"}:
            # Gradient penalty does not fight rigid translations (∇vS≈0).
            r_skeleton += gamma_vS_c * _one_minus(alpha_k) * inner(grad(vS_k), grad(vS_test)) * dx
            if float(gamma_vS_pin_eff) != 0.0:
                h_u = MeshSize()
                inv_h2 = _c(1.0) / (h_u * h_u)
                w_pin = _one_minus(alpha_k)
                w_pin_pow = w_pin
                for _ in range(vS_pin_pow - 1):
                    w_pin_pow = w_pin_pow * w_pin
                r_skeleton += gamma_vS_pin_c * inv_h2 * w_pin_pow * dot(vS_k, vS_test) * dx
        else:
            raise ValueError(f"Unknown vS_extension_mode {vS_extension_mode!r}.")

    # Jacobian contributions (k-part only)
    if drag_mode == "scalar":
        drag_term_k = -beta_k * dot(v_k, vS_test) + beta_k * dot(vS_k, vS_test)
        d_drag_term_k = -dbeta * (dot(v_k, vS_test) - dot(vS_k, vS_test))
        d_drag_term_k += -beta_k * (dot(dv, vS_test) - dot(dvS, vS_test))
    else:
        drag_term_k = -beta_coeff_k * dot(kdrag_k, vS_test)
        d_drag_term_k = -(dbeta_coeff * dot(kdrag_k, vS_test) + beta_coeff_k * dot(dkdrag_k, vS_test))

    # Jacobian of the elastic part (k-part only).
    #
    # For the default full-stress model, this reduces to:
    #   δ[α g(d) r_el(u)] = α g a_el + δ(α g) r_el.
    #
    # For Miehe split (linear elasticity), g(d) multiplies only the tensile part:
    #   δ[α (g r⁺ + r⁻)] = α (δg r⁺ + g δr⁺ + δr⁻) + δα (g r⁺ + r⁻).
    elastic_jac_k = g_stiff_k * a_el_plus + a_el_minus
    if use_damage:
        # Only include δg(d)·r⁺ when the damage field is part of the unknown vector.
        # Otherwise this term would appear as a test-only contribution in the Jacobian
        # (0·test) and break matrix assembly in the python backend.
        elastic_jac_k += dg_stiff_k * r_el_plus_k

    a_skel = sk_th * (alpha_k * elastic_jac_k + dalpha * (g_stiff_k * r_el_plus_k + r_el_minus_k)) * dx
    if a_skel_visco_alpha is not None:
        a_skel += a_skel_visco_alpha
    if a_skel_visco_vS is not None:
        a_skel += a_skel_visco_vS
    # Jacobian of the pressure coupling -p div(B η).
    a_skel += sk_th * (-(dp * div_B_vStest_k + p_k * d_div_B_vStest_k)) * dx
    if float(gamma_div) != 0.0:
        gamma_div_c = _c(float(gamma_div))
        d_divF_k = d_divCv_k + d_divBvS_k
        a_skel += sk_th * gamma_div_c * (d_divF_k * div_B_vStest_k + divF_k * d_div_B_vStest_k) * dx
    # Drag term is *not* multiplied by alpha again: beta already contains alpha (one-domain blend).
    a_skel += sk_th * d_drag_term_k * dx
    a_skel += -dot(dalpha * f_u, vS_test) * dx
    # Jacobian of the vS extension penalty (k-part only).
    if float(gamma_vS_eff) != 0.0:
        if vS_ext_mode in {"l2", "mass"}:
            h_u = MeshSize()
            inv_h2 = _c(1.0) / (h_u * h_u)
            a_skel += gamma_vS_c * inv_h2 * (
                (-_c(1.0) * dalpha) * dot(vS_k, vS_test) + _one_minus(alpha_k) * dot(dvS, vS_test)
            ) * dx
        elif vS_ext_mode in {"grad", "h1"}:
            a_skel += gamma_vS_c * (
                (-_c(1.0) * dalpha) * inner(grad(vS_k), grad(vS_test))
                + _one_minus(alpha_k) * inner(grad(dvS), grad(vS_test))
            ) * dx

            if float(gamma_vS_pin_eff) != 0.0:
                h_u = MeshSize()
                inv_h2 = _c(1.0) / (h_u * h_u)
                w_pin = _one_minus(alpha_k)
                w_pin_pow = w_pin
                for _ in range(vS_pin_pow - 1):
                    w_pin_pow = w_pin_pow * w_pin

                if vS_pin_pow == 1:
                    w_pin_pow_m1 = _c(1.0)
                else:
                    w_pin_pow_m1 = w_pin
                    for _ in range(vS_pin_pow - 2):
                        w_pin_pow_m1 = w_pin_pow_m1 * w_pin
                dw_pin_pow = (-_c(float(vS_pin_pow)) * w_pin_pow_m1) * dalpha
                a_skel += gamma_vS_pin_c * inv_h2 * (
                    dw_pin_pow * dot(vS_k, vS_test) + w_pin_pow * dot(dvS, vS_test)
                ) * dx
        else:
            raise ValueError(f"Unknown vS_extension_mode {vS_extension_mode!r}.")

    # Optional Eulerian skeleton inertia.
    #
    # The continuous term is the conservative form
    #   ∂t(ρ_S v^S) + div(ρ_S v^S ⊗ v^S),
    # with ρ_S = rho_s0_tilde * α(1-φ) = rho_s0_tilde * B.
    #
    # For robustness in monolithic solves, the default is a Picard-like lagging
    # of the convective part:
    #   div(ρ_S v^S ⊗ v^S) ≈ div(ρ_S^n v^{S,n} ⊗ v^{S,k}).
    if bool(include_skeleton_acceleration) and float(rho_s0_tilde) != 0.0:
        inertia_conv_key = str(skeleton_inertia_convection).strip().lower() if skeleton_inertia_convection is not None else "lagged"
        if inertia_conv_key in {"conservative", "nonlinear"}:
            inertia_conv_key = "full"
        if inertia_conv_key in {"picard", "semi", "semi_implicit", "linear"}:
            inertia_conv_key = "lagged"

        rho_s0_c = rho_s0_tilde

        rhoS_k = rho_s0_c * B_k
        rhoS_n = rho_s0_c * B_n

        # Conservative-in-time momentum term.
        momS_dot = (rhoS_k * vS_k - rhoS_n * vS_n) * inv_dt

        r_skeleton += inner(momS_dot, vS_test) * dx

        # Conservative convection (two modes):
        # - full:    div(ρ_S^k v^{S,k} ⊗ v^{S,k}) (nonlinear)
        # - lagged:  div(ρ_S^n v^{S,n} ⊗ v^{S,k}) (Picard linearization)
        if inertia_conv_key not in {"full", "lagged"}:
            raise ValueError(
                f"Unknown skeleton_inertia_convection={skeleton_inertia_convection!r}. "
                "Use 'lagged' (default) or 'full'."
            )

        grad_vS_k = grad(vS_k)
        grad_vS_n = grad(vS_n)

        div_rhoS_vS_n = rho_s0_c * divBvS_n

        if inertia_conv_key == "full":
            advS_k = dot(grad_vS_k, vS_k)
            advS_n = dot(grad_vS_n, vS_n)
            convS_k = dot(advS_k, vS_test)
            convS_n = dot(advS_n, vS_test)

            div_rhoS_vS_k = rho_s0_c * divBvS_k

            r_skeleton += th * (rhoS_k * convS_k + div_rhoS_vS_k * dot(vS_k, vS_test)) * dx
            r_skeleton += one_m_th * (rhoS_n * convS_n + div_rhoS_vS_n * dot(vS_n, vS_test)) * dx
        else:
            # Lagged/Picard form: div(ρ^n v^n ⊗ v^k) = ρ^n (v^n·∇)v^k + v^k div(ρ^n v^n).
            advS_k = dot(grad_vS_k, vS_n)
            convS_k = dot(advS_k, vS_test)

            advS_n = dot(grad_vS_n, vS_n)
            convS_n = dot(advS_n, vS_test)

            r_skeleton += th * (rhoS_n * convS_k + div_rhoS_vS_n * dot(vS_k, vS_test)) * dx
            r_skeleton += one_m_th * (rhoS_n * convS_n + div_rhoS_vS_n * dot(vS_n, vS_test)) * dx

        # Jacobian (k-part only): always include δ[ (ρ_S v^S)/dt ].
        d_rhoS_k = rho_s0_c * dB_k
        d_momS_dot = (d_rhoS_k * vS_k + rhoS_k * dvS) * inv_dt
        a_skel += inner(d_momS_dot, vS_test) * dx

        if inertia_conv_key == "full":
            # δ[ ρ_S (v^S·∇)v^S + v^S div(ρ_S v^S) ] at k-level.
            grad_dvS = grad(dvS)
            d_advS_k = dot(grad_dvS, vS_k) + dot(grad_vS_k, dvS)
            d_convS_k = dot(d_advS_k, vS_test)
            a_skel += th * (d_rhoS_k * convS_k + rhoS_k * d_convS_k) * dx

            div_rhoS_vS_k = rho_s0_c * divBvS_k
            d_div_rhoS_vS_k = rho_s0_c * d_divBvS_k
            a_skel += th * (d_div_rhoS_vS_k * dot(vS_k, vS_test) + div_rhoS_vS_k * dot(dvS, vS_test)) * dx
        elif inertia_conv_key == "lagged":
            # δ[ ρ_S^n (v^{S,n}·∇)v^{S} + v^{S} div(ρ_S^n v^{S,n}) ] at k-level.
            grad_dvS = grad(dvS)
            d_advS_k = dot(grad_dvS, vS_n)
            d_convS_k = dot(d_advS_k, vS_test)
            a_skel += th * (rhoS_n * d_convS_k) * dx

            div_rhoS_vS_n = rho_s0_c * divBvS_n
            a_skel += th * (div_rhoS_vS_n * dot(dvS, vS_test)) * dx

    # NOTE: u-CIP stabilization is applied to the *kinematic constraint* for u
    # (see below), not to the vS momentum balance.
    # ------------------------------------------------------------------
    # (iii-b) Optional wall adhesion traction (spring + dashpot)
    # ------------------------------------------------------------------
    if ds_adh is not None and (
        float(adhesion_k_n) != 0.0
        or float(adhesion_k_t) != 0.0
        or float(adhesion_gamma_n) != 0.0
        or float(adhesion_gamma_t) != 0.0
    ):
        # Adhesion integrity a is treated as lagged/known in the Newton step.
        a_prev = adhesion_a_prev if adhesion_a_prev is not None else _c(1.0)

        k_n_c = _c(float(adhesion_k_n))
        k_t_c = _c(float(adhesion_k_t))
        g_n_c = _c(float(adhesion_gamma_n))
        g_t_c = _c(float(adhesion_gamma_t))

        n_b = FacetNormal()

        def _proj_n(vec):
            return dot(vec, n_b) * n_b

        def _proj_t(vec):
            return vec - _proj_n(vec)

        # k-level traction uses (u_k, vS_k); n-level uses (u_n, vS_n) for θ-scheme consistency.
        u_nvec_k = _proj_n(u_k)
        u_tvec_k = u_k - u_nvec_k
        vS_nvec_k = _proj_n(vS_k)
        vS_tvec_k = vS_k - vS_nvec_k

        u_nvec_n = _proj_n(u_n)
        u_tvec_n = u_n - u_nvec_n
        vS_nvec_n = _proj_n(vS_n)
        vS_tvec_n = vS_n - vS_nvec_n

        t_adh_k = k_n_c * u_nvec_k + k_t_c * u_tvec_k + g_n_c * vS_nvec_k + g_t_c * vS_tvec_k
        t_adh_n = k_n_c * u_nvec_n + k_t_c * u_tvec_n + g_n_c * vS_nvec_n + g_t_c * vS_tvec_n

        r_skeleton += (
            sk_th * alpha_k * a_prev * dot(t_adh_k, vS_test)
            + sk_one_m_th * alpha_n * a_prev * dot(t_adh_n, vS_test)
        ) * ds_adh

        # Jacobian (k-part only): δ[ α a_prev t_adh(u,vS) ].
        du_nvec = _proj_n(du)
        du_tvec = du - du_nvec
        dvS_nvec = _proj_n(dvS)
        dvS_tvec = dvS - dvS_nvec
        dt_adh = k_n_c * du_nvec + k_t_c * du_tvec + g_n_c * dvS_nvec + g_t_c * dvS_tvec

        a_skel += sk_th * (dalpha * a_prev * dot(t_adh_k, vS_test) + alpha_k * a_prev * dot(dt_adh, vS_test)) * ds_adh

    # Optional CIP (continuous interior penalty) stabilization for vS.
    # Regularizes vS in the diffuse interface / near-zero-α region without changing the continuous limit.
    if float(vS_cip) != 0.0 and ds_cip is not None:
        n_int = FacetNormal()
        h_F = avg(MeshSize())
        tau_cip = _c(float(vS_cip)) * (h_F * h_F * h_F) * inv_dt
        w_s = avg(alpha_n)
        r_skeleton += tau_cip * w_s * _grad_inner_jump(vS_k, vS_test, n_int) * ds_cip
        a_skel += tau_cip * w_s * _grad_inner_jump(dvS, vS_test, n_int) * ds_cip

    # ------------------------------------------------------------------
    # (iii-c) Solid kinematics (Eulerian reference-map constraint)
    # ------------------------------------------------------------------
    # For an Eulerian reference map X(x,t) (material coordinate of the point at x),
    #   ∂_t X + vS·∇X = 0.
    # With u = x - X, this becomes:
    #   ∂_t u + vS·∇u = vS.
    #
    # We enforce this as a separate (first-order) PDE for u, localized to the
    # biofilm region via α. Outside the biofilm (α≈0), u is defined by the
    # extension penalty below.
    #
    # Scaling: multiplying the *entire* u-equation by a positive scalar does not
    # change the solution set, but it can improve conditioning of the monolithic
    # Newton solve and the line-search norm weighting (important when the u
    # residual is orders of magnitude smaller than the vS residual).
    if kinematics_scale is None:
        kinematics_scale = rho_s0_tilde if (rho_s0_tilde is not None and float(rho_s0_tilde) != 0.0) else 1.0
    kin_scale_c = kinematics_scale if hasattr(kinematics_scale, "dim") else _c(float(kinematics_scale))

    Fkin_dt = (u_k - u_n) * inv_dt
    Fkin_adv_k = dot(grad(u_k), vS_k) - vS_k
    Fkin_adv_n = dot(grad(u_n), vS_n) - vS_n
    Fkin_k = Fkin_dt + th * Fkin_adv_k + one_m_th * Fkin_adv_n

    r_kinematics = kin_scale_c * alpha_k * dot(Fkin_k, u_test) * dx

    # Jacobian (k-part only)
    dFkin_dt = du * inv_dt
    dFkin_adv_k = dot(grad(du), vS_k) + dot(grad(u_k), dvS) - dvS
    dFkin_k = dFkin_dt + th * dFkin_adv_k
    a_kinematics = kin_scale_c * (dalpha * dot(Fkin_k, u_test) + alpha_k * dot(dFkin_k, u_test)) * dx

    # Optional SUPG-like streamline diffusion for the u-transport (kinematic constraint).
    #
    # This adds an artificial diffusion along vS in the solid region:
    #   τ ( (vS^n·∇)u^k , (vS^n·∇)ξ )_{Ω}  localized by α^n.
    if float(u_supg) != 0.0:
        h_u = MeshSize()
        vmag2 = vS_n[0] * vS_n[0] + vS_n[1] * vS_n[1]
        vmag = _sqrt(vmag2 + _c(1.0e-12))
        denom = h_u * vmag + (h_u * h_u) * inv_dt
        tau_u = _c(float(u_supg)) * (h_u * h_u) / (denom + _c(1.0e-16))
        w_u = alpha_n  # lagged "solid-only" localization
        adv_u_k = dot(grad(u_k), vS_n)
        adv_xi = dot(grad(u_test), vS_n)
        r_kinematics += kin_scale_c * tau_u * w_u * inner(adv_u_k, adv_xi) * dx
        a_kinematics += kin_scale_c * tau_u * w_u * inner(dot(grad(du), vS_n), adv_xi) * dx

    # Optional extension penalty to keep u well-posed in the free-fluid region (α≈0).
    if float(gamma_u) != 0.0:
        u_ext_mode = str(u_extension_mode).strip().lower()
        if u_ext_mode in {"l2", "mass"}:
            h_u = MeshSize()
            inv_h2 = _c(1.0) / (h_u * h_u)
            r_kinematics += kin_scale_c * gamma_u_c * inv_h2 * _one_minus(alpha_k) * dot(u_k, u_test) * dx
            a_kinematics += kin_scale_c * gamma_u_c * inv_h2 * (
                (-_c(1.0) * dalpha) * dot(u_k, u_test) + _one_minus(alpha_k) * dot(du, u_test)
            ) * dx
        elif u_ext_mode in {"grad", "h1"}:
            r_kinematics += kin_scale_c * gamma_u_c * _one_minus(alpha_k) * inner(grad(u_k), grad(u_test)) * dx
            a_kinematics += kin_scale_c * gamma_u_c * (
                (-_c(1.0) * dalpha) * inner(grad(u_k), grad(u_test)) + _one_minus(alpha_k) * inner(grad(du), grad(u_test))
            ) * dx

            if float(gamma_u_pin) != 0.0:
                h_u = MeshSize()
                inv_h2 = _c(1.0) / (h_u * h_u)
                pin_c = _c(float(gamma_u_pin))
                w_pin = _one_minus(alpha_k)
                w_pin2 = w_pin * w_pin
                dw_pin2 = (-_c(2.0) * w_pin) * dalpha
                r_kinematics += kin_scale_c * pin_c * inv_h2 * w_pin2 * dot(u_k, u_test) * dx
                a_kinematics += kin_scale_c * pin_c * inv_h2 * (dw_pin2 * dot(u_k, u_test) + w_pin2 * dot(du, u_test)) * dx
        else:
            raise ValueError(f"Unknown u_extension_mode {u_extension_mode!r}.")

    # Optional facet stabilization for u (CIP/ghost-penalty on interior facets).
    if float(u_cip) != 0.0 and ds_cip is not None:
        n_int = FacetNormal()
        h_F = avg(MeshSize())
        # NOTE: keep the CIP weight backend-identical. A velocity-dependent term
        # (|vS|/h) would require a robust sqrt on interior facets; use inv_dt.
        scale = inv_dt
        tau_u_cip = _c(float(u_cip)) * (h_F * h_F * h_F) * scale
        w_key = str(u_cip_weight or "fluid").strip().lower()
        if w_key in {"fluid", "one_minus_alpha", "1-alpha", "one-minus-alpha"}:
            w_u_cip = avg(_one_minus(alpha_n))
        elif w_key in {"biofilm", "alpha"}:
            w_u_cip = avg(alpha_n)
        elif w_key in {"both", "all", "unity", "1"}:
            w_u_cip = _c(1.0)
        else:
            raise ValueError(
                f"Unknown u_cip_weight={u_cip_weight!r}. Use 'fluid' (default), 'biofilm', or 'both'."
            )
        r_kinematics += kin_scale_c * tau_u_cip * w_u_cip * _grad_inner_jump(u_k, u_test, n_int) * ds_cip
        a_kinematics += kin_scale_c * tau_u_cip * w_u_cip * _grad_inner_jump(du, u_test, n_int) * ds_cip

    # ------------------------------------------------------------------
    # (iv) Porosity evolution (Eulerian, advected by vS)
    # ------------------------------------------------------------------
    D_phi_c = _c(float(D_phi))
    gamma_phi_c = _c(float(gamma_phi))
    # Regularization weight for enforcing φ≈1 in the free-fluid region.
    #
    # IMPORTANT: we intentionally *sharpen* the weight using (1-α)^16 so the
    # constraint does not bleed into the biofilm over long time horizons when
    # D_phi=0 (pure transport). Using (1-α) directly is overly aggressive in the
    # diffuse interface (α≈0.5) for CG spaces.
    one_m_alpha_k = _one_minus(alpha_k)
    # (1-α)^16 = ((1-α)^4)^4
    w_phi_fluid4_k = one_m_alpha_k * one_m_alpha_k
    w_phi_fluid4_k = w_phi_fluid4_k * w_phi_fluid4_k  # (1-α)^4
    w_phi_fluid8_k = w_phi_fluid4_k * w_phi_fluid4_k  # (1-α)^8
    w_phi_fluid_k = w_phi_fluid8_k * w_phi_fluid8_k  # (1-α)^16

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
    r_phi += gamma_phi_c * w_phi_fluid_k * (phi_k - _c(1.0)) * phi_test * dx
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
    # δ[(1-α)^16] = 16 (1-α)^15 δ(1-α) = -16 (1-α)^15 δα
    one_m_alpha3_k = one_m_alpha_k * one_m_alpha_k * one_m_alpha_k  # (1-α)^3
    dw_phi_fluid_k = (-_c(16.0) * (w_phi_fluid8_k * w_phi_fluid4_k * one_m_alpha3_k)) * dalpha
    a_phi += gamma_phi_c * (dw_phi_fluid_k * (phi_k - _c(1.0)) + w_phi_fluid_k * dphi) * phi_test * dx

    # Optional consistent stabilization for advection-dominated φ (useful with D_phi=0).
    # - SUPG: τ (vS·∇w) R_phi
    # - CIP:  γ h^3 (1/dt + |vS|/h) <[∂_n φ],[∂_n w]>_F on interior facets
    #
    # We again use lagged vS_n in τ and in the test-direction to keep the Jacobian
    # coupling limited to the (already-present) vS_k dependence inside R_phi.
    if float(phi_supg) != 0.0:
        h_p = MeshSize()
        vmag2 = vS_n[0] * vS_n[0] + vS_n[1] * vS_n[1]
        vmag = _sqrt(vmag2 + _c(1.0e-12))
        denom = (_c(2.0) * inv_dt) * (_c(2.0) * inv_dt) + (_c(2.0) * vmag / (h_p + _c(1.0e-12))) * (
            _c(2.0) * vmag / (h_p + _c(1.0e-12))
        )
        tau_supg = _c(float(phi_supg)) / _sqrt(denom + _c(1.0e-16))
        # Avoid `dot(grad(test), v)` due to current dot/left_dot limitations in the
        # C++ backend for some mixed expression types; expand componentwise.
        g_test = grad(phi_test)
        w_supg = g_test[0] * vS_n[0] + g_test[1] * vS_n[1]

        # "Strong" residual (excluding diffusion; diffusion is a stabilizing term already).
        f_phi_supg_k = alpha_k * ((phi_k - phi_n) * inv_dt)
        f_phi_supg_k += th * alpha_k * Fphi_k + one_m_th * alpha_n * Fphi_n
        f_phi_supg_k += gamma_phi_c * w_phi_fluid_k * (phi_k - _c(1.0))
        f_phi_supg_k += -f_phi

        r_phi += tau_supg * w_supg * f_phi_supg_k * dx

        dFphi_k = dot(grad(phi_k), dvS) + dot(grad(dphi), vS_k) + dphi * div_vS_k - _one_minus(phi_k) * div_dvS + dPi
        df_phi_supg_k = dalpha * ((phi_k - phi_n) * inv_dt) + alpha_k * (dphi * inv_dt)
        df_phi_supg_k += th * (dalpha * Fphi_k + alpha_k * dFphi_k)
        df_phi_supg_k += gamma_phi_c * (dw_phi_fluid_k * (phi_k - _c(1.0)) + w_phi_fluid_k * dphi)

        a_phi += tau_supg * w_supg * df_phi_supg_k * dx

    if float(phi_cip) != 0.0 and ds_cip is not None:
        n_int = FacetNormal()
        h_F = avg(MeshSize())
        scale = inv_dt
        tau_cip = _c(float(phi_cip)) * (h_F * h_F * h_F) * scale
        # Localize facet stabilization to the biofilm region to avoid smearing the
        # imposed fluid value (φ=1) into the biofilm over long time horizons.
        #
        # Use α^- α^+ (computed robustly from avg/jump) as a weight:
        #   α^- α^+ = avg(α)^2 - (jump(α)/2)^2
        a_avg = avg(alpha_n)
        a_jump = jump(alpha_n)
        w_phi_cip = a_avg * a_avg + (-_c(0.25) * a_jump * a_jump)
        r_phi += tau_cip * w_phi_cip * _grad_inner_jump(phi_k, phi_test, n_int) * ds_cip
        a_phi += tau_cip * w_phi_cip * _grad_inner_jump(dphi, phi_test, n_int) * ds_cip

    # ------------------------------------------------------------------
    # (v) Indicator evolution (advection–diffusion–reaction)
    # ------------------------------------------------------------------
    D_alpha_c = _c(float(D_alpha))
    k_g_c = k_g if hasattr(k_g, "dim") else _c(float(k_g))
    mu_max_c = mu_max if hasattr(mu_max, "dim") else _c(float(mu_max))
    K_S_c = K_S if hasattr(K_S, "dim") else _c(float(K_S))
    G_k = _G(S_k, phi_k, k_g=k_g_c, mu_max=mu_max_c, K_S=K_S_c)
    G_n = _G(S_n, phi_n, k_g=k_g_c, mu_max=mu_max_c, K_S=K_S_c)

    # Phase-field (Allen–Cahn) regularization parameters for α.
    # Only active when alpha_cahn_M*alpha_cahn_gamma != 0.
    eps_alpha_val = float(alpha_cahn_eps)
    if eps_alpha_val <= 0.0 and (float(alpha_cahn_M) != 0.0 or float(alpha_crack_k) != 0.0):
        raise ValueError(f"alpha_cahn_eps must be > 0 when phase-field/crack terms are enabled; got {eps_alpha_val}.")
    eps_alpha_c = _c(max(eps_alpha_val, 1.0e-12))

    M_alpha_c = _c(float(alpha_cahn_M))
    gamma_alpha_c = _c(float(alpha_cahn_gamma))
    M_gamma_alpha = M_alpha_c * gamma_alpha_c

    f_alpha_k = (alpha_k - alpha_n) * inv_dt
    # Surface-localized detachment sink: D_det_prev * δ(α), where δ is a smooth interface delta.
    # We use δ(α) = 4 α (1-α), which is supported robustly by the current UFL/compiler stack.
    delta_k = _c(4.0) * alpha_k * _one_minus(alpha_k)
    delta_n = _c(4.0) * alpha_n * _one_minus(alpha_n)

    # Optional crack-propagation surface speed V_crack^prev (lagged): adds an additional
    # surface-localized sink proportional to δ(α).
    crack_coef_prev = _c(0.0)
    if float(alpha_crack_k) != 0.0:
        if V_crack_prev is not None:
            Vc_prev = V_crack_prev if hasattr(V_crack_prev, "dim") else _c(float(V_crack_prev))
        else:
            driver_key = str(alpha_crack_driver).strip().lower()
            eta_mech = _c(float(alpha_crack_eta_mech))

            # IMPORTANT: avoid vector inner products of function-function in LHS assembly
            # (the current visitor only supports VecOpInfo.inner for test/trial pairs).
            # Use driver measures built from GradOpInfo/HessOpInfo or scalar expressions.
            if driver_key in {"shear", "fluid_shear", "tau"}:
                # Fluid shear stress proxy (works robustly in all backends):
                #   τ ≈ 2 μ ||ε(v)||_F
                tau2 = inner(_epsilon(v_n), _epsilon(v_n))
                tau = _sqrt(tau2 + eta_mech)
                D_mech_n = _c(2.0) * mu_n * tau
            elif driver_key in {"solid_strain", "strain"}:
                # Skeleton strain proxy (dimensionless):
                #   ||ε(u)||_F
                eps2 = inner(_epsilon(u_n), _epsilon(u_n))
                D_mech_n = _sqrt(eps2 + eta_mech)
            elif driver_key in {"solid_von_mises", "von_mises", "von-mises", "vm", "solid_vm"}:
                # Solid von Mises equivalent stress (Pa), based on the *elastic* Cauchy stress.
                # Use the same constitutive law as the skeleton equation.
                if solid_model_key in {"linear", "small_strain", "linear_elastic"}:
                    sig = _c(2.0) * mu_s * _epsilon(u_n) + lambda_s * div(u_n) * Identity(int(dim))
                elif solid_model_key in {"hencky", "hencky_log", "hencky_log_strain"}:
                    sig = sigma_hencky(u_n, mu_s, lambda_s, dim=int(dim))
                elif solid_model_key in {"stvk", "svk", "saint_venant_kirchhoff", "saint-venant-kirchhoff"}:
                    sig = sigma_svk(u_n, mu_s, lambda_s, dim=int(dim))
                else:
                    if c_nh is None:
                        c_nh = mu_s / _c(2.0)
                    if beta_nh is None:
                        beta_nh = lambda_s / (_c(2.0) * mu_s)
                    sig = sigma_neo_hookean(u_n, c_nh, beta_nh, dim=int(dim))

                tr_sig = trace(sig)
                I = Identity(int(dim))
                s_dev = sig - (tr_sig / _c(float(dim))) * I
                # σ_vm = sqrt(3/2 s:s). For 2D we still use the 3D-style J2 scaling as a robust proxy.
                vm2 = (_c(1.5) * inner(s_dev, s_dev)) + eta_mech
                D_mech_n = _sqrt(vm2)
            elif driver_key in {"drag", "brinkman_drag", "brinkman"}:
                # Approximate drag-driven tearing with the shear proxy due to current compiler limits.
                tau2 = inner(_epsilon(v_n), _epsilon(v_n))
                tau = _sqrt(tau2 + eta_mech)
                D_mech_n = _c(2.0) * mu_n * tau
            else:
                raise ValueError(
                    f"Unknown alpha_crack_driver {alpha_crack_driver!r}. Use 'shear', 'solid_strain', 'solid_von_mises', or 'drag' (alias)."
                )

            # Ensure the driver is treated as a function-like scalar (VecOpInfo on the left)
            # so subsequent arithmetic does not hit unsupported float - VecOpInfo branches
            # in LHS assembly.
            D_mech_n = (_c(0.0) * alpha_n) + D_mech_n

            # Curvature proxy from α^n using only primitive grad/Laplacian operators
            # (keeps compatibility with all backends, including the C++ kernel).
            g_n = grad(alpha_n)
            g2 = inner(g_n, g_n)
            denom = g2 + _c(float(alpha_crack_eta_kappa))
            denom_sqrt = _sqrt(denom)
            # The exact mean curvature κ = div(∇α/|∇α|) is not fully supported in the
            # current symbolic pipeline. We use a robust approximation based on the
            # Laplacian (trace of the Hessian):
            #   κ̃ = (|∇α|^2 Δα) / (|∇α|^2 + η)^{3/2}  ≈ Δα/|∇α|.
            lap_n = Laplacian(alpha_n)
            kappa_n = (g2 * lap_n) / (denom * denom_sqrt)

            D_c = _c(float(alpha_crack_Dc))
            gamma_kappa = _c(float(alpha_crack_gamma_kappa))
            drive = D_mech_n - gamma_kappa * kappa_n - D_c

            # Smooth positive part: <x>_+ ≈ 0.5 (x + sqrt(x^2 + η)).
            eta_pos = _c(float(alpha_crack_eta_pos))
            pos = _c(0.5) * (drive + _sqrt(drive * drive + eta_pos))

            m_pow = float(alpha_crack_m)
            if m_pow < 1.0:
                raise ValueError(f"alpha_crack_m must be >= 1; got {m_pow}.")
            Vc_prev = _c(float(alpha_crack_k)) * (pos ** _c(m_pow))

        # Convert speed [length/time] to a rate [1/time] via (4 ε)^{-1}.
        crack_coef_prev = Vc_prev / (_c(4.0) * eps_alpha_c)

    # Surface-localized erosion/detachment sink is -D_det_prev δ(α) on the RHS,
    # so it enters the residual with a + sign (same convention as the X source).
    surf_coef_prev = D_det_prev + crack_coef_prev

    # Select the advecting velocity for α.
    #
    # NOTE: For the "mix" option we intentionally use *lagged* (C_n,B_n) weights so the
    # α advection remains linear for frozen (v,vS) updates in operator-splitting schemes.
    adv_with_key = str(alpha_advect_with or "vS").strip().lower()
    if adv_with_key in {"vs", "v^s", "v_s", "s", "skeleton", "solid"}:
        adv_u_k = vS_k
        adv_u_n = vS_n
        adv_u_k_comp = [vS_k[i] for i in range(int(dim))]
        adv_u_n_comp = [vS_n[i] for i in range(int(dim))]
        dadv_u = dvS
        div_adv_u_k = div_vS_k
        div_adv_u_n = div_vS_n
        d_div_adv_u = div_dvS
    elif adv_with_key in {"v", "fluid"}:
        adv_u_k = v_k
        adv_u_n = v_n
        adv_u_k_comp = [v_k[i] for i in range(int(dim))]
        adv_u_n_comp = [v_n[i] for i in range(int(dim))]
        dadv_u = dv
        div_adv_u_k = div(v_k)
        div_adv_u_n = div(v_n)
        d_div_adv_u = div(dv)
    elif adv_with_key in {"mix", "mixture", "f", "flux", "volume"}:
        # Mixture/volume velocity: F = C v + B vS,  with C=(1-α)+αφ and B=α(1-φ).
        adv_u_k = C_n * v_k + B_n * vS_k
        adv_u_n = C_n * v_n + B_n * vS_n
        adv_u_k_comp = [C_n * v_k[i] + B_n * vS_k[i] for i in range(int(dim))]
        adv_u_n_comp = [C_n * v_n[i] + B_n * vS_n[i] for i in range(int(dim))]
        dadv_u = C_n * dv + B_n * dvS

        # Conservative form needs div(F). Expand div(C_n v + B_n vS) explicitly.
        div_adv_u_k = C_n * div(v_k) + dot(gradC_n, v_k) + B_n * div_vS_k + dot(gradB_n, vS_k)
        div_adv_u_n = divCv_n + divBvS_n
        d_div_adv_u = C_n * div(dv) + dot(gradC_n, dv) + B_n * div_dvS + dot(gradB_n, dvS)
    elif adv_with_key in {"mix_biofilm", "mix-biofilm", "mix_bio", "mixbio", "mix_cutoff", "mixcutoff"}:
        # Mixture/volume velocity but *gated* so we do not advect α through the pure fluid
        # where α≈0 (which can create a spurious thin "chimney" when v has strong outflow).
        #
        # We gate only the C-part by a smooth cutoff g(α^n), leaving the biofilm-side
        # velocity unchanged for α above the cutoff:
        #   F_adv = (g C) v + B vS,   g(α) = α^m / (α^m + α0^m).
        alpha0 = float(alpha_mix_gate_alpha0)
        if not (0.0 < alpha0 < 1.0):
            raise ValueError(f"alpha_mix_gate_alpha0 must be in (0,1); got {alpha0}.")
        m_pow = int(alpha_mix_gate_power)
        if m_pow < 1:
            raise ValueError(f"alpha_mix_gate_power must be >= 1; got {m_pow}.")

        # Build α^m with repeated multiplication (keeps backend compatibility).
        a_pow = alpha_n
        for _ in range(m_pow - 1):
            a_pow = a_pow * alpha_n
        a0_pow = _c(alpha0 ** float(m_pow))
        gate_n = a_pow / (a_pow + a0_pow + _c(1.0e-12))

        Cg_n = gate_n * C_n
        gradCg_n = grad(Cg_n)

        adv_u_k = Cg_n * v_k + B_n * vS_k
        adv_u_n = Cg_n * v_n + B_n * vS_n
        adv_u_k_comp = [Cg_n * v_k[i] + B_n * vS_k[i] for i in range(int(dim))]
        adv_u_n_comp = [Cg_n * v_n[i] + B_n * vS_n[i] for i in range(int(dim))]
        dadv_u = Cg_n * dv + B_n * dvS

        div_adv_u_k = Cg_n * div(v_k) + dot(gradCg_n, v_k) + B_n * div_vS_k + dot(gradB_n, vS_k)
        div_adv_u_n = Cg_n * div(v_n) + dot(gradCg_n, v_n) + divBvS_n
        d_div_adv_u = Cg_n * div(dv) + dot(gradCg_n, dv) + B_n * div_dvS + dot(gradB_n, dvS)
    else:
        raise ValueError(
            f"Unknown alpha_advect_with={alpha_advect_with!r}. Use 'vS' (default), 'v', 'mix', or 'mix_biofilm'."
        )

    adv_key = str(alpha_advection_form).strip().lower()
    if adv_key in {"advective", "nonconservative", "v.grad", "v·grad", "vgrad"}:
        adv_key = "advective"
    elif adv_key in {"conservative", "div", "divergence", "div(alpha*v)"}:
        adv_key = "conservative"
    else:
        raise ValueError(
            f"Unknown alpha_advection_form={alpha_advection_form!r}. Use 'advective' or 'conservative'."
        )

    if adv_key == "advective":
        # Indicator advection: ∂t α + u·∇α = ...
        adv_alpha_k = dot(grad(alpha_k), adv_u_k)
        adv_alpha_n = dot(grad(alpha_n), adv_u_n)
    else:
        # Conservative advection: ∂t α + div(α u) = ...
        #
        # We implement div(α u) as u·∇α + α div(u) to avoid relying on backend
        # support for `div(alpha*u)` in all code paths.
        adv_alpha_k = dot(grad(alpha_k), adv_u_k) + alpha_k * div_adv_u_k
        adv_alpha_n = dot(grad(alpha_n), adv_u_n) + alpha_n * div_adv_u_n
    f_alpha_k += th * (adv_alpha_k - G_k * alpha_k * _one_minus(alpha_k) + surf_coef_prev * delta_k)
    f_alpha_k += one_m_th * (adv_alpha_n - G_n * alpha_n * _one_minus(alpha_n) + surf_coef_prev * delta_n)

    r_alpha = alpha_test * f_alpha_k * dx
    r_alpha += D_alpha_c * inner(grad(alpha_k), grad(alpha_test)) * dx

    # Cahn–Hilliard regularization: α_t + ... = div( M(α) ∇μ_α ), μ_α = γ(-εΔα + (1/ε)W'(α)).
    ch_enabled = float(alpha_ch_M) != 0.0 and float(alpha_ch_gamma) != 0.0
    if bool(alpha_cahn_conservative) and ch_enabled:
        raise ValueError("alpha_cahn_conservative cannot be used together with Cahn–Hilliard regularization (alpha_ch_*).")

    # Allen–Cahn regularization: -(Mγ ε Δα) + (Mγ/ε) W'(α) in the residual.
    ac_enabled = float(alpha_cahn_M) != 0.0 and float(alpha_cahn_gamma) != 0.0
    if ac_enabled and ch_enabled:
        raise ValueError("Allen–Cahn (alpha_cahn_*) and Cahn–Hilliard (alpha_ch_*) cannot both be enabled simultaneously.")
    if bool(alpha_cahn_conservative):
        cons_mode = str(alpha_cahn_conservative_mode).strip().lower()
        if cons_mode in {"solve", "unknown", "lagrange", "constraint"}:
            cons_mode = "unknown"
        elif cons_mode in {"eliminate", "elim", "project", "projected", "explicit"}:
            cons_mode = "eliminate"
        else:
            raise ValueError(
                f"Unknown alpha_cahn_conservative_mode {alpha_cahn_conservative_mode!r}. Use 'unknown' or 'eliminate'."
            )
        if not ac_enabled:
            raise ValueError("alpha_cahn_conservative=True requires alpha_cahn_M and alpha_cahn_gamma to be nonzero.")
    else:
        cons_mode = "unknown"
    if ac_enabled:
        # W'(α) for W(α)=α^2(1-α)^2 is 2α(1-α)(1-2α).
        Wp_k = _c(2.0) * alpha_k * _one_minus(alpha_k) * _one_minus(_c(2.0) * alpha_k)
        mob_key = str(alpha_cahn_mobility).strip().lower()
        if mob_key in {"constant", "const"}:
            M_ac_k = M_alpha_c
            dM_ac = _c(0.0) * dalpha
            mob_prime_k = _c(0.0) * alpha_k
        elif mob_key in {"degenerate", "deg", "alpha(1-alpha)", "alpha*(1-alpha)", "a(1-a)"}:
            # Degenerate mobility: M(α)=M0 α(1-α)
            floor = float(alpha_cahn_mobility_floor)
            if floor < 0.0:
                raise ValueError(f"alpha_cahn_mobility_floor must be >= 0; got {floor}.")
            mob_k = alpha_k * _one_minus(alpha_k) + _c(floor)
            mob_prime_k = _one_minus(_c(2.0) * alpha_k)  # d/dα [α(1-α)] = 1-2α
            M_ac_k = M_alpha_c * mob_k
            dM_ac = M_alpha_c * mob_prime_k * dalpha
        else:
            raise ValueError(f"Unknown alpha_cahn_mobility {alpha_cahn_mobility!r}. Use 'constant' or 'degenerate'.")

        M_gamma_k = gamma_alpha_c * M_ac_k
        dM_gamma = gamma_alpha_c * dM_ac

        # Strong term: +M(α) μ_α, μ_α=γ(-εΔα + (1/ε)W'(α)).
        # Integrate by parts the Laplacian term; if M depends on α, an extra
        # interface-localized term ε ∇M·∇α appears.
        r_alpha += (M_gamma_k * eps_alpha_c) * inner(grad(alpha_k), grad(alpha_test)) * dx
        if mob_key not in {"constant", "const"}:
            r_alpha += alpha_test * ((eps_alpha_c * gamma_alpha_c * M_alpha_c) * mob_prime_k * inner(grad(alpha_k), grad(alpha_k))) * dx
        r_alpha += alpha_test * ((M_gamma_k / eps_alpha_c) * Wp_k) * dx

        if bool(alpha_cahn_conservative):
            if lambda_alpha_k is None:
                raise ValueError("alpha_cahn_conservative=True requires lambda_alpha_k to be provided.")
            r_alpha += -alpha_test * (M_ac_k * lambda_alpha_k) * dx

    r_mu_alpha = None
    a_mu_alpha = None
    if ch_enabled:
        if mu_alpha_k is None or mu_alpha_n is None or dmu_alpha is None or mu_alpha_test is None:
            raise ValueError(
                "Cahn–Hilliard regularization requires (mu_alpha_k, mu_alpha_n, dmu_alpha, mu_alpha_test) to be provided."
            )
        eps_ch_val = float(alpha_ch_eps)
        if eps_ch_val <= 0.0:
            raise ValueError(f"alpha_ch_eps must be > 0 when Cahn–Hilliard is enabled; got {eps_ch_val}.")
        eps_ch_c = _c(max(eps_ch_val, 1.0e-12))

        M_ch_c = _c(float(alpha_ch_M))
        gamma_ch_c = _c(float(alpha_ch_gamma))

        # Double-well derivative W'(α) for W(α)=α^2(1-α)^2 is 2α(1-α)(1-2α).
        Wp_ch_k = _c(2.0) * alpha_k * _one_minus(alpha_k) * _one_minus(_c(2.0) * alpha_k)
        # W''(α) = 2 - 12α + 12α^2
        Wpp_ch_k = (-_c(12.0) * alpha_k) + (_c(12.0) * (alpha_k * alpha_k)) + _c(2.0)

        mob_key = str(alpha_ch_mobility).strip().lower()
        if mob_key in {"constant", "const"}:
            M_ch_k = M_ch_c
            M_ch_n = M_ch_c
            dM_ch = _c(0.0) * dalpha
        elif mob_key in {"degenerate", "deg", "alpha(1-alpha)", "alpha*(1-alpha)", "a(1-a)"}:
            # Degenerate mobility: M(α)=M0 α(1-α)
            mob_k = alpha_k * _one_minus(alpha_k)
            mob_n = alpha_n * _one_minus(alpha_n)
            mob_prime_k = _one_minus(_c(2.0) * alpha_k)  # d/dα [α(1-α)] = 1-2α
            M_ch_k = M_ch_c * mob_k
            M_ch_n = M_ch_c * mob_n
            dM_ch = M_ch_c * mob_prime_k * dalpha
        else:
            raise ValueError(f"Unknown alpha_ch_mobility {alpha_ch_mobility!r}. Use 'constant' or 'degenerate'.")

        # α equation: +∫ M(α) ∇μ · ∇w  (no-flux boundary)
        #
        # IMPORTANT: keep GradOpInfo on the left in scalar×grad products for backend
        # compatibility (function×grad(function) is not implemented, but grad(function)×function is).
        r_alpha += th * inner(grad(mu_alpha_k) * M_ch_k, grad(alpha_test)) * dx
        r_alpha += one_m_th * inner(grad(mu_alpha_n) * M_ch_n, grad(alpha_test)) * dx

        # μ equation: ∫ ψ ( μ - γ(-εΔα + (1/ε)W'(α)) ) dx = 0 (drop boundary term).
        r_mu_alpha = mu_alpha_test * mu_alpha_k * dx
        r_mu_alpha += -(gamma_ch_c * eps_ch_c) * inner(grad(alpha_k), grad(mu_alpha_test)) * dx
        r_mu_alpha += -mu_alpha_test * ((gamma_ch_c / eps_ch_c) * Wp_ch_k) * dx
    r_alpha += -alpha_test * f_alpha * dx

    # Jacobian (k-part only)
    # δG
    dG = (_c(float(k_g)) * _one_minus(phi_k) * (dmon_dS * dS) + (-_c(float(k_g)) * mon_k) * dphi)
    # δ[ G α(1-α) ] = (δG) α(1-α) + G (1-2α) δα
    dalpha_logistic = _one_minus(_c(2.0) * alpha_k) * dalpha

    a_alpha = alpha_test * (dalpha * inv_dt) * dx
    if adv_key == "advective":
        a_alpha += alpha_test * th * (dot(grad(alpha_k), dadv_u) + dot(grad(dalpha), adv_u_k)) * dx
    else:
        a_alpha += alpha_test * th * (
            dot(grad(alpha_k), dadv_u)
            + dot(grad(dalpha), adv_u_k)
            + dalpha * div_adv_u_k
            + alpha_k * d_div_adv_u
        ) * dx
    # δ[ (D_det_prev + crack_coef_prev) * δ(α) ] = (D_det_prev + crack_coef_prev) * δ'(α) δα
    # (surf coefficients are lagged).
    d_delta_k = _c(4.0) * (_one_minus(_c(2.0) * alpha_k)) * dalpha
    d_surf = surf_coef_prev * d_delta_k
    a_alpha += alpha_test * th * (-(dG * alpha_k * _one_minus(alpha_k) + G_k * dalpha_logistic) + d_surf) * dx
    a_alpha += D_alpha_c * inner(grad(dalpha), grad(alpha_test)) * dx
    if ac_enabled:
        # W''(α) = 2 - 12α + 12α^2
        # NOTE: keep the function-like term on the left; float - VecOpInfo is not supported.
        Wpp_k = (-_c(12.0) * alpha_k) + (_c(12.0) * (alpha_k * alpha_k)) + _c(2.0)
        if mob_key in {"constant", "const"}:
            a_alpha += (M_gamma_alpha * eps_alpha_c) * inner(grad(dalpha), grad(alpha_test)) * dx
            a_alpha += alpha_test * ((M_gamma_alpha / eps_alpha_c) * Wpp_k * dalpha) * dx
            if bool(alpha_cahn_conservative) and cons_mode == "unknown":
                if dlambda_alpha is None:
                    raise ValueError(
                        "alpha_cahn_conservative_mode='unknown' requires dlambda_alpha (lambda_alpha as unknown)."
                    )
                a_alpha += -alpha_test * (M_alpha_c * dlambda_alpha) * dx
        else:
            # d[ ε Mγ ∇α·∇w ] = ε (dMγ ∇α·∇w + Mγ ∇δα·∇w)
            a_alpha += (eps_alpha_c * dM_gamma) * inner(grad(alpha_k), grad(alpha_test)) * dx
            a_alpha += (M_gamma_k * eps_alpha_c) * inner(grad(dalpha), grad(alpha_test)) * dx

            # d[ ε w (∇Mγ·∇α) ] with M= M0 α(1-α): ∇Mγ = γ M0 (1-2α) ∇α
            g2 = inner(grad(alpha_k), grad(alpha_k))
            a_alpha += alpha_test * ((eps_alpha_c * gamma_alpha_c * M_alpha_c) * ((-_c(2.0) * dalpha) * g2)) * dx
            a_alpha += alpha_test * ((eps_alpha_c * gamma_alpha_c * M_alpha_c) * (mob_prime_k * (_c(2.0) * inner(grad(alpha_k), grad(dalpha))))) * dx

            # d[ w (Mγ/ε) W'(α) ] = w/ε (dMγ W'(α) + Mγ W''(α) δα)
            a_alpha += alpha_test * (((dM_gamma / eps_alpha_c) * Wp_k) + ((M_gamma_k / eps_alpha_c) * Wpp_k * dalpha)) * dx
            if bool(alpha_cahn_conservative):
                if lambda_alpha_k is None:
                    raise ValueError("alpha_cahn_conservative=True requires lambda_alpha_k to be provided.")
                a_alpha += -alpha_test * (dM_ac * lambda_alpha_k) * dx
                if cons_mode == "unknown":
                    if dlambda_alpha is None:
                        raise ValueError(
                            "alpha_cahn_conservative_mode='unknown' requires dlambda_alpha (lambda_alpha as unknown)."
                        )
                    a_alpha += -alpha_test * (M_ac_k * dlambda_alpha) * dx

    if ch_enabled:
        # k-part Jacobian of ∫ M(α) ∇μ · ∇w
        a_alpha += th * (dM_ch * inner(grad(mu_alpha_k), grad(alpha_test))) * dx
        a_alpha += th * inner(M_ch_k * grad(dmu_alpha), grad(alpha_test)) * dx

        # Jacobian of μ equation.
        a_mu_alpha = mu_alpha_test * dmu_alpha * dx
        a_mu_alpha += -(gamma_ch_c * eps_ch_c) * inner(grad(dalpha), grad(mu_alpha_test)) * dx
        a_mu_alpha += -mu_alpha_test * ((gamma_ch_c / eps_ch_c) * Wpp_ch_k * dalpha) * dx

    # Optional consistent stabilization for advection-dominated α (useful with D_alpha=0).
    # - SUPG: τ (u·∇w) R_alpha
    # - CIP:  γ h^3 (1/dt + |u|/h) <[∂_n α],[∂_n w]>_F on interior facets
    #
    # Notes:
    # - We use a lagged advector u_n in τ and in the test-direction to keep the Jacobian
    #   consistent (no extra u-coupling from the stabilization weights).
    # - CIP only affects regions where ∇α is non-zero (i.e. the diffuse interface),
    #   and remains consistent because [∂_n α]=0 for smooth α.
    if float(alpha_supg) != 0.0:
        h_a = MeshSize()
        vmag2 = _c(0.0)
        for j in range(int(dim)):
            vmag2 += adv_u_n_comp[j] * adv_u_n_comp[j]
        vmag = _sqrt(vmag2 + _c(1.0e-12))
        denom = (_c(2.0) * inv_dt) * (_c(2.0) * inv_dt) + (_c(2.0) * vmag / (h_a + _c(1.0e-12))) * (
            _c(2.0) * vmag / (h_a + _c(1.0e-12))
        )
        tau_supg = _c(float(alpha_supg)) / _sqrt(denom + _c(1.0e-16))
        # Avoid `dot(grad(test), v)` due to current dot/left_dot limitations in the
        # C++ backend for some mixed expression types; expand componentwise.
        g_test = grad(alpha_test)
        w_supg = _c(0.0)
        for j in range(int(dim)):
            w_supg += g_test[j] * adv_u_n_comp[j]
        r_alpha += tau_supg * w_supg * f_alpha_k * dx

        df_alpha_k = dalpha * inv_dt
        if adv_key == "advective":
            df_alpha_k += th * (
                dot(grad(alpha_k), dadv_u)
                + dot(grad(dalpha), adv_u_k)
                - (dG * alpha_k * _one_minus(alpha_k) + G_k * dalpha_logistic)
                + d_surf
            )
        else:
            df_alpha_k += th * (
                dot(grad(alpha_k), dadv_u)
                + dot(grad(dalpha), adv_u_k)
                + dalpha * div_adv_u_k
                + alpha_k * d_div_adv_u
                - (dG * alpha_k * _one_minus(alpha_k) + G_k * dalpha_logistic)
                + d_surf
            )
        a_alpha += tau_supg * w_supg * df_alpha_k * dx

    if float(alpha_cip) != 0.0 and ds_cip is not None:
        n_int = FacetNormal()
        h_F = avg(MeshSize())
        scale = inv_dt
        tau_cip = _c(float(alpha_cip)) * (h_F * h_F * h_F) * scale
        r_alpha += tau_cip * _grad_inner_jump(alpha_k, alpha_test, n_int) * ds_cip
        a_alpha += tau_cip * _grad_inner_jump(dalpha, alpha_test, n_int) * ds_cip

    # ------------------------------------------------------------------
    # (v-a) Conservative Allen–Cahn constraint (optional): determine λ_α(t)
    # ------------------------------------------------------------------
    r_alpha_lambda = None
    a_alpha_lambda = None
    if bool(alpha_cahn_conservative) and cons_mode == "unknown":
        if not ac_enabled:
            raise ValueError("alpha_cahn_conservative=True requires alpha_cahn_M and alpha_cahn_gamma to be nonzero.")
        if lambda_alpha_k is None or dlambda_alpha is None or lambda_alpha_test is None:
            raise ValueError(
                "alpha_cahn_conservative=True requires (lambda_alpha_k, dlambda_alpha, lambda_alpha_test) to be provided."
            )

        # Constraint: ∫ M(α) (μ_α - λ_α) dx = 0, with μ_α = γ(-ε Δα + (1/ε)W'(α)).
        #
        # IMPORTANT: avoid assembling Laplacian(alpha) directly in the constraint.
        # For the domain integral of M(-εΔα), use integration by parts and drop the
        # boundary term (consistent with the natural no-flux condition for α used
        # in the biofilm model):
        #   ∫ M(-εΔα) dx = ε ∫ ∇M·∇α dx.
        #
        # This uses only first derivatives and is numerically more robust for high-order runs.
        lam_scale = alpha_cahn_lambda_scale if alpha_cahn_lambda_scale is not None else _c(1.0)
        r_alpha_lambda = lam_scale * lambda_alpha_test * (
            ((gamma_alpha_c / eps_alpha_c) * (M_ac_k * Wp_k)) - (M_ac_k * lambda_alpha_k)
        ) * dx

        # Domain correction: γ ε ∇M·∇α (only nonzero for variable mobility).
        if mob_key not in {"constant", "const"}:
            g2 = inner(grad(alpha_k), grad(alpha_k))
            r_alpha_lambda += lam_scale * lambda_alpha_test * (
                (eps_alpha_c * gamma_alpha_c * M_alpha_c) * mob_prime_k * g2
            ) * dx

        # Jacobian (k-part only)
        # d[ γ(M/ε)W'(α) - M λ ]
        a_alpha_lambda = lam_scale * lambda_alpha_test * (
            (gamma_alpha_c / eps_alpha_c) * ((dM_ac * Wp_k) + (M_ac_k * Wpp_k * dalpha)) + (-(dM_ac * lambda_alpha_k) - (M_ac_k * dlambda_alpha))
        ) * dx

        if mob_key not in {"constant", "const"}:
            g2 = inner(grad(alpha_k), grad(alpha_k))
            a_alpha_lambda += lam_scale * lambda_alpha_test * (
                (eps_alpha_c * gamma_alpha_c * M_alpha_c) * ((-_c(2.0) * dalpha) * g2 + mob_prime_k * (_c(2.0) * inner(grad(alpha_k), grad(dalpha))))
            ) * dx

    # ------------------------------------------------------------------
    # (v-b) Bulk damage evolution (optional): cohesion loss driven by solid stress
    # ------------------------------------------------------------------
    r_damage = None
    a_damage = None
    if use_damage:
        damage_model_key = str(damage_model).strip().lower()
        D_d_c = _c(float(damage_D))
        gamma_out_c = _c(float(damage_gamma_out))
        # Lagged von Mises driver from previous skeleton state u_n.
        # Used by both damage models below; lagging keeps Newton tangents robust.
        sigma_vm = _c(0.0)
        drive_vm = _c(0.0)
        if solid_model_key in {"linear", "small_strain", "linear_elastic"}:
            eps_un = _epsilon(u_n)
            sig_un = _c(2.0) * mu_s * eps_un + lambda_s * div(u_n) * Identity(int(dim))
        elif solid_model_key in {"hencky", "hencky_log", "hencky_log_strain"}:
            sig_un = sigma_hencky(u_n, mu_s, lambda_s, dim=int(dim))
        elif solid_model_key in {"stvk", "svk", "saint_venant_kirchhoff", "saint-venant-kirchhoff"}:
            sig_un = sigma_svk(u_n, mu_s, lambda_s, dim=int(dim))
        else:
            if c_nh is None:
                c_nh = mu_s / _c(2.0)
            if beta_nh is None:
                beta_nh = lambda_s / (_c(2.0) * mu_s)
            sig_un = sigma_neo_hookean(u_n, c_nh, beta_nh, dim=int(dim))

        tr_sig = trace(sig_un)
        s_dev = sig_un - (tr_sig / _c(float(dim))) * Identity(int(dim))
        vm2 = _c(1.5) * inner(s_dev, s_dev)
        sigma_vm = _sqrt(vm2 + _c(1.0e-16))

        if float(damage_sigma_cr) > 0.0:
            sigma_cr_c = _c(float(damage_sigma_cr))
            ratio = sigma_vm / sigma_cr_c - _c(1.0)
            pos_ratio = _smooth_pos(ratio, eta=float(damage_eta_pos))
            drive_vm = pos_ratio ** _c(float(damage_m))
        else:
            drive_vm = sigma_vm

        if damage_model_key in {"kinetic", "legacy"}:
            # Legacy advection-reaction-diffusion model:
            #   α (∂t d + vS·∇d) - α rate (1-d) - div(D_d ∇d) + γ_out (1-α)^16 d = 0.
            rate = _c(0.0)
            if float(damage_k) != 0.0:
                rate = _c(float(damage_k)) * drive_vm

            f_dmg_k = alpha_k * ((d_k - d_n) * inv_dt)
            f_dmg_k += th * alpha_k * dot(grad(d_k), vS_k) + one_m_th * alpha_n * dot(grad(d_n), vS_n)
            f_dmg_k += -alpha_k * rate * one_m_d_k

            r_damage = d_test * f_dmg_k * dx
            r_damage += D_d_c * inner(grad(d_k), grad(d_test)) * dx
            if float(damage_gamma_out) != 0.0:
                r_damage += gamma_out_c * w_phi_fluid_k * d_k * d_test * dx

            # Jacobian (k-part only)
            df_dmg_k = dalpha * ((d_k - d_n) * inv_dt) + alpha_k * (dd * inv_dt)
            df_dmg_k += th * (dalpha * dot(grad(d_k), vS_k) + alpha_k * (dot(grad(dd), vS_k) + dot(grad(d_k), dvS)))
            df_dmg_k += -dalpha * rate * one_m_d_k + alpha_k * rate * dd

            a_damage = d_test * df_dmg_k * dx
            a_damage += D_d_c * inner(grad(dd), grad(d_test)) * dx
            if float(damage_gamma_out) != 0.0:
                # δ[ w(α) d ] = w δd + (δw) d, with δw = dw_phi_fluid_k.
                a_damage += gamma_out_c * (w_phi_fluid_k * dd + dw_phi_fluid_k * d_k) * d_test * dx
        elif damage_model_key in {"phase_field", "phase-field", "at2", "energy"}:
            # Energy-derived phase-field damage (AT2-like, lagged drive):
            #   α η_d D_t^S d - div(α G_c l ∇d) + α (G_c/l) d = 2 α (1-d) H_prev,
            # with a *lagged* driving field H_prev (updated between steps for robustness).
            Gc_val = float(damage_Gc)
            ell_val = float(damage_l)
            if Gc_val <= 0.0 or ell_val <= 0.0:
                raise ValueError("damage_model='phase_field' requires damage_Gc>0 and damage_l>0.")
            eta_d_c = _c(float(damage_eta))
            Gc_c = _c(Gc_val)
            ell_c = _c(ell_val)
            Gc_over_l = Gc_c / ell_c
            Gc_l = Gc_c * ell_c

            if damage_H_prev is not None:
                # Optional irreversibility: supply a history field H^{prev}(x)
                # (typically updated as max over time) so damage cannot heal when
                # the instantaneous drive relaxes.
                H_prev = damage_H_prev
            else:
                damage_pf_driver_key = str(damage_pf_driver).strip().lower()
                if damage_pf_driver_key in {"von_mises", "vm", "von-mises"}:
                    # Legacy: scale a von-Mises-based proxy into an energy density.
                    psi0_val = float(damage_psi0)
                    if psi0_val <= 0.0:
                        psi0_val = Gc_val / max(ell_val, 1.0e-12)
                    H_prev = _c(psi0_val) * drive_vm
                elif damage_pf_driver_key in {"miehe", "miehe_energy", "energy", "psi_plus", "psi+"}:
                    eta_pos = float(damage_eta_pos)
                    if solid_model_key in {"linear", "small_strain", "linear_elastic"}:
                        # Miehe-type tensile energy density ψ⁺(u) for linear elasticity:
                        #   ψ⁺ = μ ||ε⁺||² + (λ/2) ⟨tr ε⟩₊².
                        disc_reg = 1.0e-16
                        eps_un = _epsilon(u_n)
                        eps_plus_un, _, _, _, _ = spectral_positive_part_2x2_sym(
                            eps_un, eta_pos=eta_pos, disc_reg=disc_reg
                        )
                        tr_eps = div(u_n)
                        tr_pos = _smooth_pos_u(tr_eps, eta=eta_pos)
                        psi_plus = mu_s * inner(eps_plus_un, eps_plus_un) + _c(0.5) * lambda_s * (tr_pos * tr_pos)
                    elif solid_model_key in {"hencky", "hencky_log", "hencky_log_strain"}:
                        # Finite-strain Hencky tensile energy density ψ⁺(u) in 2D:
                        #   ψ⁺ = μ ||E⁺||² + (λ/2) ⟨tr E⟩₊²,  E=log(V).
                        psi_plus = hencky_tensile_energy_miehe(
                            u_n, mu_s, lambda_s, dim=int(dim), eta_pos=eta_pos
                        )
                    elif solid_model_key in {"stvk", "svk", "saint_venant_kirchhoff", "saint-venant-kirchhoff"}:
                        # Finite-strain SVK tensile energy density ψ⁺(u) in 2D:
                        #   ψ⁺ = μ ||E⁺||² + (λ/2) ⟨tr E⟩₊²,  E=0.5(C-I).
                        psi_plus = svk_tensile_energy_miehe(
                            u_n, mu_s, lambda_s, dim=int(dim), eta_pos=eta_pos
                        )
                    else:
                        raise ValueError(
                            "damage_pf_driver='miehe_energy' requires solid_model in {'linear','stvk','hencky'} (2D). "
                            "Use damage_pf_driver='von_mises' for neo-Hookean."
                        )
                    scale = float(damage_psi0)
                    if scale > 0.0:
                        psi_plus = _c(scale) * psi_plus
                    H_prev = psi_plus
                else:
                    raise ValueError(f"Unknown damage_pf_driver {damage_pf_driver!r}.")

            DtS_d_k = (d_k - d_n) * inv_dt
            DtS_d_k += th * dot(grad(d_k), vS_k) + one_m_th * dot(grad(d_n), vS_n)

            # Consistent with g(d) = (1-κ)(1-d)^2 + κ:
            #   ∂/∂d [ g(d) ψ⁺ ] = -2 (1-κ) (1-d) ψ⁺.
            f_pf_k = eta_d_c * DtS_d_k + Gc_over_l * d_k - _c(2.0) * one_m_kappa_stiff * one_m_d_k * H_prev

            r_damage = alpha_k * d_test * f_pf_k * dx
            r_damage += alpha_k * Gc_l * inner(grad(d_k), grad(d_test)) * dx
            if float(damage_gamma_out) != 0.0:
                r_damage += gamma_out_c * w_phi_fluid_k * d_k * d_test * dx

            d_DtS_d_k = dd * inv_dt + th * (dot(grad(dd), vS_k) + dot(grad(d_k), dvS))
            df_pf_k = eta_d_c * d_DtS_d_k + Gc_over_l * dd + _c(2.0) * one_m_kappa_stiff * H_prev * dd

            a_damage = (dalpha * f_pf_k + alpha_k * df_pf_k) * d_test * dx
            a_damage += (dalpha * Gc_l * inner(grad(d_k), grad(d_test)) + alpha_k * Gc_l * inner(grad(dd), grad(d_test))) * dx
            if float(damage_gamma_out) != 0.0:
                a_damage += gamma_out_c * (w_phi_fluid_k * dd + dw_phi_fluid_k * d_k) * d_test * dx
        else:
            raise ValueError(f"Unknown damage_model {damage_model!r}.")

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
    # Substrate sink R_S:
    # - With S normalized by ρ_s*: R_S = (1/Y) (Π_b/ρ_s*).
    # - With S as mass concentration: multiply by ρ_s* (handled via rho_s_star).
    rho_s_star_c = _c(float(rho_s_star))
    Y_c = _c(float(Y))
    RS_k = rho_s_star_c * _R_S_consumption(
        S_k, phi_k, alpha_k, mu_max=_c(float(mu_max)), K_S=_c(float(K_S)), k_d=_c(float(k_d)), Y=Y_c
    )
    RS_n = rho_s_star_c * _R_S_consumption(
        S_n, phi_n, alpha_n, mu_max=_c(float(mu_max)), K_S=_c(float(K_S)), k_d=_c(float(k_d)), Y=Y_c
    )

    # IMEX split for stiff substrate reaction:
    # The Monod sink can be very stiff once scaled by ρ_s^*/Y, so Crank–Nicolson
    # (θ=0.5) may produce oscillatory updates unless Δt is extremely small.
    # Allow treating the reaction term fully implicitly while keeping the rest
    # of the transport equation at the global θ-scheme.
    rs_key = str(substrate_reaction_scheme).strip().lower()
    if rs_key in {"theta", "cn", "trap", "trapezoid", "trapezoidal"}:
        th_RS = th
        one_m_th_RS = one_m_th
    elif rs_key in {"implicit", "imex", "be", "backward_euler", "backward-euler", "lstable", "l-stable"}:
        th_RS = _c(1.0)
        one_m_th_RS = _c(0.0)
    elif rs_key in {"explicit", "fe", "forward_euler", "forward-euler"}:
        th_RS = _c(0.0)
        one_m_th_RS = _c(1.0)
    else:
        raise ValueError(
            f"Unknown substrate_reaction_scheme={substrate_reaction_scheme!r}. "
            "Use 'theta' (default), 'implicit'/'imex', or 'explicit'."
        )

    diff_key = str(substrate_diffusion_scheme).strip().lower()
    if diff_key in {"theta", "cn", "trap", "trapezoid", "trapezoidal"}:
        th_diff = th
        one_m_th_diff = one_m_th
    elif diff_key in {"implicit", "imex", "be", "backward_euler", "backward-euler", "lstable", "l-stable"}:
        th_diff = _c(1.0)
        one_m_th_diff = _c(0.0)
    elif diff_key in {"explicit", "fe", "forward_euler", "forward-euler"}:
        th_diff = _c(0.0)
        one_m_th_diff = _c(1.0)
    else:
        raise ValueError(
            f"Unknown substrate_diffusion_scheme={substrate_diffusion_scheme!r}. "
            "Use 'theta' (default), 'implicit'/'imex', or 'explicit'."
        )

    div_CSv_k = CSk * div(v_k) + S_k * dot(gradC_k, v_k) + C_k * dot(grad(S_k), v_k)
    div_CSv_n = CSn * div(v_n) + S_n * dot(gradC_n, v_n) + C_n * dot(grad(S_n), v_n)

    r_sub = S_test * ((CSk - CSn) * inv_dt) * dx
    r_sub += S_test * (th * div_CSv_k + one_m_th * div_CSv_n) * dx
    r_sub += D_S_c * th_diff * inner(grad(S_k), grad(S_test)) * dx + D_S_c * one_m_th_diff * inner(grad(S_n), grad(S_test)) * dx
    r_sub += S_test * (th_RS * RS_k + one_m_th_RS * RS_n) * dx
    r_sub += -S_test * f_S * dx

    # Jacobian (k-part only)
    dCSk = dC * S_k + _capacity(alpha_k, phi_k) * dS
    dRS = rho_s_star_c * ((_c(1.0) / Y_c) * dPi)  # RS = (rho_s_star/Y) (Π_b/ρ_s*)

    d_div_CSv_k = dCSk * div(v_k) + CSk * div(dv)
    d_div_CSv_k += dS * dot(gradC_k, v_k) + S_k * dot(dgradC_k, v_k) + S_k * dot(gradC_k, dv)
    d_div_CSv_k += dC_k * dot(grad(S_k), v_k) + C_k * dot(grad(dS), v_k) + C_k * dot(grad(S_k), dv)

    a_sub = S_test * (dCSk * inv_dt) * dx
    a_sub += S_test * th * d_div_CSv_k * dx
    a_sub += D_S_c * th_diff * inner(grad(dS), grad(S_test)) * dx
    a_sub += S_test * th_RS * dRS * dx

    # ------------------------------------------------------------------
    # (vii) Detached biomass transport (optional)
    # ------------------------------------------------------------------
    r_detached = None
    a_detached = None
    if X_k is not None:
        if X_n is None or dX is None or X_test is None:
            raise ValueError("X_k provided but one of (X_n, dX, X_test) is missing.")

        D_X_c = _c(float(D_X))

        CXk = C_k * X_k
        CXn = C_n * X_n

        div_CXv_k = CXk * div(v_k) + X_k * dot(gradC_k, v_k) + C_k * dot(grad(X_k), v_k)
        div_CXv_n = CXn * div(v_n) + X_n * dot(gradC_n, v_n) + C_n * dot(grad(X_n), v_n)

        # Source from detachment: R_det = ρ_s* (1-φ) D_det_prev δ(α).
        R_det_k = rho_s_star_c * _one_minus(phi_k) * D_det_prev * delta_k
        R_det_n = rho_s_star_c * _one_minus(phi_n) * D_det_prev * delta_n

        r_detached = X_test * ((CXk - CXn) * inv_dt) * dx
        r_detached += X_test * (th * div_CXv_k + one_m_th * div_CXv_n) * dx
        r_detached += D_X_c * th * inner(grad(X_k), grad(X_test)) * dx + D_X_c * one_m_th * inner(grad(X_n), grad(X_test)) * dx
        r_detached += -X_test * (th * R_det_k + one_m_th * R_det_n) * dx
        r_detached += -X_test * f_X * dx

        # Jacobian (k-part only)
        dCXk = dC * X_k + C_k * dX

        d_div_CXv_k = dCXk * div(v_k) + CXk * div(dv)
        d_div_CXv_k += dX * dot(gradC_k, v_k) + X_k * dot(dgradC_k, v_k) + X_k * dot(gradC_k, dv)
        d_div_CXv_k += dC * dot(grad(X_k), v_k) + C_k * dot(grad(dX), v_k) + C_k * dot(grad(X_k), dv)

        # δR_det_k (D_det_prev is lagged): ρ_s* D_det_prev [ -(δφ) δ(α) + (1-φ) δδ(α) ].
        d_delta_k = _c(4.0) * (_one_minus(_c(2.0) * alpha_k)) * dalpha
        dR_det_k = rho_s_star_c * D_det_prev * ((-dphi) * delta_k + _one_minus(phi_k) * d_delta_k)

        a_detached = X_test * (dCXk * inv_dt) * dx
        a_detached += X_test * th * d_div_CXv_k * dx
        a_detached += D_X_c * th * inner(grad(dX), grad(X_test)) * dx
        a_detached += -X_test * th * dR_det_k * dx

    # ------------------------------------------------------------------
    residual_form = r_mom + r_mass + r_kinematics + r_skeleton + r_phi + r_alpha + r_sub
    jacobian_form = a_mom + a_mass + a_kinematics + a_skel + a_phi + a_alpha + a_sub
    if r_mu_alpha is not None:
        residual_form += r_mu_alpha
    if a_mu_alpha is not None:
        jacobian_form += a_mu_alpha
    if r_alpha_lambda is not None:
        residual_form += r_alpha_lambda
    if a_alpha_lambda is not None:
        jacobian_form += a_alpha_lambda
    if r_damage is not None:
        residual_form += r_damage
    if a_damage is not None:
        jacobian_form += a_damage
    if r_detached is not None:
        residual_form += r_detached
    if a_detached is not None:
        jacobian_form += a_detached

    return BiofilmOneDomainForms(
        jacobian_form=jacobian_form,
        residual_form=residual_form,
        r_momentum=r_mom,
        r_mass=r_mass,
        r_kinematics=r_kinematics,
        r_skeleton=r_skeleton,
        r_phi=r_phi,
        r_alpha=r_alpha,
        r_mu_alpha=r_mu_alpha,
        r_damage=r_damage,
        r_substrate=r_sub,
        a_momentum=a_mom,
        a_mass=a_mass,
        a_kinematics=a_kinematics,
        a_skeleton=a_skel,
        a_phi=a_phi,
        a_alpha=a_alpha,
        a_mu_alpha=a_mu_alpha,
        a_damage=a_damage,
        a_substrate=a_sub,
        r_detached=r_detached,
        a_detached=a_detached,
        r_alpha_lambda=r_alpha_lambda,
        a_alpha_lambda=a_alpha_lambda,
    )
