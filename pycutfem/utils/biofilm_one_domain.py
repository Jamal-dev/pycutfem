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


def _grad_inner_jump(u, v, n):
    """Penalty on the jump of the normal derivative across an interior facet."""
    return inner(dot(jump(grad(u)), n), dot(jump(grad(v)), n))


def _linear_elastic_term(u, eta, *, mu_s, lambda_s):
    # For symmetric stress, σ(u):∇η = 2μ ε(u):ε(η) + λ div(u) div(η).
    return _c(2.0) * mu_s * inner(_epsilon(u), _epsilon(eta)) + lambda_s * div(u) * div(eta)

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
    r_skeleton: object
    r_phi: object
    r_alpha: object
    r_damage: object | None
    r_substrate: object
    # Optional per-block Jacobian contributions (useful for debugging/verification)
    a_momentum: object | None = None
    a_mass: object | None = None
    a_skeleton: object | None = None
    a_phi: object | None = None
    a_alpha: object | None = None
    a_damage: object | None = None
    a_substrate: object | None = None
    r_detached: object | None = None
    a_detached: object | None = None


def build_biofilm_one_domain_forms(
    *,
    # unknowns at t_{n+1}
    v_k,
    p_k,
    u_k,
    phi_k,
    alpha_k,
    S_k,
    # optional: bulk damage / cohesion loss (0=intact, 1=failed)
    d_k=None,
    # optional: detached-biomass concentration at t_{n+1}
    X_k=None,
    # unknowns at t_n
    v_n,
    p_n,
    u_n,
    phi_n,
    alpha_n,
    S_n,
    # optional: bulk damage at t_n
    d_n=None,
    # optional: detached-biomass concentration at t_n
    X_n=None,
    # optional u_{n-1} (for θ terms with vS_n)
    u_nm1=None,
    # Newton increments
    dv,
    dp,
    du,
    dphi,
    dalpha,
    dS,
    dd=None,
    dX=None,
    # test functions
    v_test,
    q_test,
    u_test,
    phi_test,
    alpha_test,
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
    D_alpha: float = 0.0,
    # Allen–Cahn / phase-field interface regularization for α
    alpha_cahn_M: float = 0.0,
    alpha_cahn_gamma: float = 0.0,
    alpha_cahn_eps: float = 1.0,
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
    damage_stiff_split: str = "full",
    D_S: float = 0.0,
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
    f_X = f_X if f_X is not None else zero_scalar

    # ------------------------------------------------------------------
    # Kinematics: skeleton velocity via backward Euler
    # ------------------------------------------------------------------
    vS_k = (u_k - u_n) / dt
    div_vS_k = (div(u_k) - div(u_n)) / dt
    if u_nm1 is None:
        # Use explicit constants (not 0*vS_k) so the Python backend does not
        # treat lagged quantities as depending on current trial/functions.
        vS_n = Constant([0.0] * int(dim), dim=1)
        div_vS_n = _c(0.0)
    else:
        vS_n = (u_n - u_nm1) / dt
        div_vS_n = (div(u_n) - div(u_nm1)) / dt

    dvS = du / dt
    div_dvS = div(du) / dt

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

    # ------------------------------------------------------------------
    # (i) Momentum: conservative Navier–Stokes–Brinkman
    # ------------------------------------------------------------------
    momdot = (rho_k * v_k - rho_n * v_n) * inv_dt
    conv_k = dot(dot(grad(v_k), v_k), v_test)
    conv_n = dot(dot(grad(v_n), v_n), v_test)

    # Conservative convection: div(ρ v⊗v) = ρ (v·∇)v + v div(ρ v), with ρ=ρ_f C.
    div_rhov_k = rho_f * divCv_k
    div_rhov_n = rho_f * divCv_n

    r_mom = inner(momdot, v_test) * dx
    r_mom += th * (rho_k * conv_k + div_rhov_k * dot(v_k, v_test)) * dx
    r_mom += one_m_th * (rho_n * conv_n + div_rhov_n * dot(v_n, v_test)) * dx
    r_mom += _c(2.0) * th * mu_k * inner(_epsilon(v_k), _epsilon(v_test)) * dx
    r_mom += _c(2.0) * one_m_th * mu_n * inner(_epsilon(v_n), _epsilon(v_test)) * dx
    # Pressure term for the mixture constraint div(C v + B vS)=... :
    # variationally consistent fluid coupling is C grad(p), i.e.
    #   -(p, div(C w)) = -p (C div(w) + grad(C)·w)
    # which is the exact adjoint of the C v part of the constraint.
    # Use dot(gradC, w_test) ordering for backend compatibility.
    div_C_vtest_k = C_k * div(v_test) + dot(gradC_k, v_test)
    r_mom += -p_k * div_C_vtest_k * dx
    if drag_mode == "scalar":
        r_mom += beta_k * dot(v_k, v_test) * dx
        r_mom += -beta_k * dot(vS_k, v_test) * dx
    else:
        r_mom += beta_coeff_k * dot(kdrag_k, v_test) * dx
    r_mom += -dot(f_v, v_test) * dx

    a_mom = inv_dt * (drho * dot(v_k, v_test) + rho_k * dot(dv, v_test)) * dx

    a_mom += th * (drho * conv_k + rho_k * dot(dot(grad(dv), v_k), v_test) + rho_k * dot(dot(grad(v_k), dv), v_test)) * dx
    # Jacobian of the conservative correction v div(ρ v).
    # Keep trial-family contributions separated to avoid mixed-role metadata
    # leakage in the assembler (v-trial vs alpha/phi-trial pieces).
    d_divCv_k_ap = dC_k * div(v_k) + dot(dgradC_k, v_k)
    d_divCv_k_v = C_k * div(dv) + dot(gradC_k, dv)
    a_mom += th * (rho_f * d_divCv_k_ap * dot(v_k, v_test)) * dx
    a_mom += th * (rho_f * d_divCv_k_v * dot(v_k, v_test)) * dx
    a_mom += th * (div_rhov_k * dot(dv, v_test)) * dx
    a_mom += _c(2.0) * th * (dmu * inner(_epsilon(v_k), _epsilon(v_test)) + mu_k * inner(_epsilon(dv), _epsilon(v_test))) * dx

    # δ[-p div(C w)] = -(δp) div(C w) - p δ(div(C w)),
    # with δ(div(C w)) = δC div(w) + δgrad(C)·w for fixed test w.
    # Expand dgrad·w component-wise to avoid backend-dependent contraction
    # paths for Grad(trial-scalar) · VectorTest.
    d_div_C_vtest_k = dC_k * div(v_test)
    if int(dim) == 2:
        dgradC_k_x = (phi_k - _c(1.0)) * grad(dalpha)[0] + dphi * grad(alpha_k)[0] + dalpha * grad(phi_k)[0] + alpha_k * grad(dphi)[0]
        dgradC_k_y = (phi_k - _c(1.0)) * grad(dalpha)[1] + dphi * grad(alpha_k)[1] + dalpha * grad(phi_k)[1] + alpha_k * grad(dphi)[1]
        d_div_C_vtest_k += dgradC_k_x * v_test[0] + dgradC_k_y * v_test[1]
    else:
        d_div_C_vtest_k += dot(dgradC_k, v_test)
    a_mom += -(dp * div_C_vtest_k + p_k * d_div_C_vtest_k) * dx
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
    r_mass = q_test * (th * divF_k + one_m_th * divF_n - alpha_k * s_v) * dx

    # Jacobian of divF_k (k-part only)
    # δC = (φ-1) δα + α δφ
    a_mass = q_test * th * (d_divCv_k + d_divBvS_k - dalpha * s_v) * dx

    # ------------------------------------------------------------------
    # (iii) Skeleton momentum (optional inertia + linear/neo-Hookean stress)
    # ------------------------------------------------------------------
    damage_stiff_split_key = str(damage_stiff_split).strip().lower()
    use_miehe_stiff_split = bool(
        use_damage
        and damage_stiff_split_key in {"miehe", "tensile", "tension_compression", "tension-compression", "tc"}
    )

    if use_miehe_stiff_split and solid_model_key not in {"linear", "small_strain", "linear_elastic"}:
        raise ValueError(
            "damage_stiff_split='miehe' is currently only implemented for solid_model='linear' "
            "(small-strain). Use damage_stiff_split='full' for neo-Hookean."
        )
    if use_miehe_stiff_split and int(dim) != 2:
        raise ValueError("damage_stiff_split='miehe' is currently only implemented for dim=2.")

    # Elastic residual/Jacobian contributions.
    #
    # - Default: full-stress degradation via scalar g_stiff(d) multiplier.
    # - Optional: Miehe (tension/compression) split for linear elasticity:
    #     σ = g(d) σ⁺(u) + σ⁻(u),  with σ⁺ built from the positive principal strains.
    if solid_model_key in {"linear", "small_strain", "linear_elastic"} and use_miehe_stiff_split:
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

        r_el_plus_k = inner(sig_plus_k, grad(u_test))
        r_el_minus_k = inner(sig_minus_k, grad(u_test))

        # --- n-level split (lagged, no Jacobian contribution) ---
        E_n = _epsilon(u_n)
        E_plus_n, E_minus_n, _, _, _ = spectral_positive_part_2x2_sym(E_n, eta_pos=eta_pos, disc_reg=disc_reg)
        trE_n = div(u_n)
        trE_pos_n = _smooth_pos_u(trE_n, eta=eta_pos)

        sig_plus_n = lambda_s * trE_pos_n * I + _c(2.0) * mu_s * E_plus_n
        sig_minus_n = lambda_s * (trE_n - trE_pos_n) * I + _c(2.0) * mu_s * E_minus_n

        r_el_plus_n = inner(sig_plus_n, grad(u_test))
        r_el_minus_n = inner(sig_minus_n, grad(u_test))

        # --- consistent Jacobian: Gateaux derivatives ---
        dE = _epsilon(du)
        dE_plus = d_spectral_positive_part_2x2_sym(E_k, dE, eta_pos=eta_pos, disc_reg=disc_reg)
        dtrE = div(du)
        dtrE_pos = _smooth_pos_u_prime(trE_k, eta=eta_pos) * dtrE

        dsig_plus_k = lambda_s * dtrE_pos * I + _c(2.0) * mu_s * dE_plus
        dsig_minus_k = lambda_s * (dtrE - dtrE_pos) * I + _c(2.0) * mu_s * (dE - dE_plus)

        a_el_plus = inner(dsig_plus_k, grad(u_test))
        a_el_minus = inner(dsig_minus_k, grad(u_test))
    else:
        # Full-stress (legacy) elastic residual/Jacobian.
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

        r_el_plus_k = r_el_k
        r_el_minus_k = _c(0.0)
        r_el_plus_n = r_el_n
        r_el_minus_n = _c(0.0)
        a_el_plus = a_el
        a_el_minus = _c(0.0)

    # Pressure coupling from the B vS part of the constraint:
    #   -(p, div(B η)) = -p (B div(η) + grad(B)·η)
    # Use dot(gradB, eta_test) ordering for backend compatibility.
    div_B_utest_k = B_k * div(u_test) + dot(gradB_k, u_test)
    div_B_utest_n = B_n * div(u_test) + dot(gradB_n, u_test)
    r_skel_press_k = -p_k * div_B_utest_k
    r_skel_press_n = -p_n * div_B_utest_n

    # drag reaction: -β (v - vS)
    # Since beta already contains α, if we use alpha again then it would square and it won't 
    # be equal to the drag from the momentum of the fluid.
    if drag_mode == "scalar":
        r_skel_drag_k = -beta_k * dot(v_k - vS_k, u_test)
        r_skel_drag_n = -beta_n * dot(v_n - vS_n, u_test)
    else:
        r_skel_drag_k = -beta_coeff_k * dot(kdrag_k, u_test)
        r_skel_drag_n = -beta_coeff_n * dot(kdrag_n, u_test)

    r_skeleton = (
        th * alpha_k * (g_stiff_k * r_el_plus_k + r_el_minus_k)
        + one_m_th * alpha_n * (g_stiff_n * r_el_plus_n + r_el_minus_n)
        + th * r_skel_press_k
        + one_m_th * r_skel_press_n
        + th * r_skel_drag_k
        + one_m_th * r_skel_drag_n
    ) * dx
    # External body force is weighted by biofilm presence α, but not degraded by g_stiff(d).
    r_skeleton += -dot(alpha_k * f_u, u_test) * dx

    # Optional extension penalty to keep u well-posed in the free-fluid region.
    # This is a pragmatic stabilization for the one-domain formulation: as α→0,
    # the skeleton equation vanishes and the u-DOFs can become nearly unconstrained,
    # leading to ill-conditioned Jacobians and Newton stagnation.
    gamma_u_c = _c(float(gamma_u))
    if float(gamma_u) != 0.0:
        u_ext_mode = str(u_extension_mode).strip().lower()
        if u_ext_mode in {"l2", "mass"}:
            # Scale the L2 penalty by 1/h^2 so its strength does not vanish under refinement.
            h_u = MeshSize()
            inv_h2 = _c(1.0) / (h_u * h_u)
            r_skeleton += gamma_u_c * inv_h2 * _one_minus(alpha_k) * dot(u_k, u_test) * dx
        elif u_ext_mode in {"grad", "h1"}:
            # Gradient penalty does not fight rigid-body translations (∇u≈0).
            r_skeleton += gamma_u_c * _one_minus(alpha_k) * inner(grad(u_k), grad(u_test)) * dx

            # Add an optional tiny L2 pin to remove the global-translation nullspace.
            if float(gamma_u_pin) != 0.0:
                h_u = MeshSize()
                inv_h2 = _c(1.0) / (h_u * h_u)
                # Use a stronger weight (1-α)^2 for the pin so it is negligible in the
                # diffuse interface region (α≈0.5) but still removes the far-field
                # translation nullspace when α≈0.
                w_pin = _one_minus(alpha_k)
                w_pin2 = w_pin * w_pin
                r_skeleton += _c(float(gamma_u_pin)) * inv_h2 * w_pin2 * dot(u_k, u_test) * dx
        else:
            raise ValueError(f"Unknown u_extension_mode {u_extension_mode!r}.")

    # Jacobian contributions (k-part only)
    if drag_mode == "scalar":
        drag_term_k = -beta_k * dot(v_k, u_test) + beta_k * dot(vS_k, u_test)
        d_drag_term_k = -dbeta * (dot(v_k, u_test) - dot(vS_k, u_test))
        d_drag_term_k += -beta_k * (dot(dv, u_test) - dot(dvS, u_test))
    else:
        drag_term_k = -beta_coeff_k * dot(kdrag_k, u_test)
        d_drag_term_k = -(dbeta_coeff * dot(kdrag_k, u_test) + beta_coeff_k * dot(dkdrag_k, u_test))

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

    a_skel = th * (alpha_k * elastic_jac_k + dalpha * (g_stiff_k * r_el_plus_k + r_el_minus_k)) * dx
    # Jacobian of the pressure coupling -p div(B η).
    # Expand dgrad·eta component-wise to avoid backend-dependent contraction
    # paths for Grad(trial-scalar) · VectorTest.
    d_div_B_utest_k = dB_k * div(u_test)
    if int(dim) == 2:
        dgradB_k_x = _one_minus(phi_k) * grad(dalpha)[0] - dphi * grad(alpha_k)[0] - dalpha * grad(phi_k)[0] - alpha_k * grad(dphi)[0]
        dgradB_k_y = _one_minus(phi_k) * grad(dalpha)[1] - dphi * grad(alpha_k)[1] - dalpha * grad(phi_k)[1] - alpha_k * grad(dphi)[1]
        d_div_B_utest_k += dgradB_k_x * u_test[0] + dgradB_k_y * u_test[1]
    else:
        d_div_B_utest_k += dot(dgradB_k, u_test)
    a_skel += th * (-(dp * div_B_utest_k + p_k * d_div_B_utest_k)) * dx
    # Drag term is *not* multiplied by alpha again: beta already contains alpha (one-domain blend).
    a_skel += th * d_drag_term_k * dx
    a_skel += -dot(dalpha * f_u, u_test) * dx

    if float(gamma_u) != 0.0:
        u_ext_mode = str(u_extension_mode).strip().lower()
        if u_ext_mode in {"l2", "mass"}:
            h_u = MeshSize()
            inv_h2 = _c(1.0) / (h_u * h_u)
            a_skel += gamma_u_c * inv_h2 * (
                (-_c(1.0) * dalpha) * dot(u_k, u_test) + _one_minus(alpha_k) * dot(du, u_test)
            ) * dx
        elif u_ext_mode in {"grad", "h1"}:
            a_skel += gamma_u_c * ((-_c(1.0) * dalpha) * inner(grad(u_k), grad(u_test)) + _one_minus(alpha_k) * inner(grad(du), grad(u_test))) * dx

            if float(gamma_u_pin) != 0.0:
                h_u = MeshSize()
                inv_h2 = _c(1.0) / (h_u * h_u)
                pin_c = _c(float(gamma_u_pin))
                w_pin = _one_minus(alpha_k)
                w_pin2 = w_pin * w_pin
                dw_pin2 = (-_c(2.0) * w_pin) * dalpha
                a_skel += pin_c * inv_h2 * (
                    dw_pin2 * dot(u_k, u_test) + w_pin2 * dot(du, u_test)
                ) * dx
        else:
            raise ValueError(f"Unknown u_extension_mode {u_extension_mode!r}.")

    # Optional Eulerian skeleton inertia (conservative form).
    #
    # When inertia is enabled and the solid mass coefficient depends on (α,φ),
    # we use the conservative form to avoid missing "variable density" terms:
    #   ∂t(ρ_s vS) + div(ρ_s vS ⊗ vS) = ρ_s (vS·∇)vS + vS div(ρ_s vS) + ∂t(ρ_s vS),
    # with ρ_s = rho_s0_tilde * α(1-φ) = rho_s0_tilde * B.
    if bool(include_skeleton_acceleration) and float(rho_s0_tilde) != 0.0:
        rho_s0_c = rho_s0_tilde

        rhoS_k = rho_s0_c * B_k
        rhoS_n = rho_s0_c * B_n

        # Conservative-in-time momentum term
        momS_dot = (rhoS_k * vS_k - rhoS_n * vS_n) * inv_dt

        # Conservative convection: div(ρ_s vS⊗vS) = ρ_s (vS·∇)vS + vS div(ρ_s vS).
        #
        # IMPORTANT: avoid grad(vS_k) directly because the current UFL visitor only
        # supports grad() of primitive Functions/Trials/Tests (not arbitrary expressions like (u_k-u_n)/dt).
        grad_vS_k = (grad(u_k) - grad(u_n)) / dt
        if u_nm1 is None:
            grad_vS_n = Constant([[0.0] * int(dim) for _ in range(int(dim))], dim=2)
        else:
            grad_vS_n = (grad(u_n) - grad(u_nm1)) / dt

        advS_k = dot(grad_vS_k, vS_k)
        advS_n = dot(grad_vS_n, vS_n)
        convS_k = dot(advS_k, u_test)
        convS_n = dot(advS_n, u_test)

        div_rhoS_vS_k = rho_s0_c * divBvS_k
        div_rhoS_vS_n = rho_s0_c * divBvS_n

        r_skeleton += inner(momS_dot, u_test) * dx
        r_skeleton += th * (rhoS_k * convS_k + div_rhoS_vS_k * dot(vS_k, u_test)) * dx
        r_skeleton += one_m_th * (rhoS_n * convS_n + div_rhoS_vS_n * dot(vS_n, u_test)) * dx

        # Jacobian (k-part only): δ[ (ρ_s vS)/dt ] + θ δ[ ρ_s (vS·∇)vS + vS div(ρ_s vS) ].
        d_rhoS_k = rho_s0_c * dB_k
        d_momS_dot = (d_rhoS_k * vS_k + rhoS_k * dvS) * inv_dt
        a_skel += inner(d_momS_dot, u_test) * dx

        grad_dvS = grad(du) / dt
        d_advS_k = dot(grad_dvS, vS_k) + dot(grad_vS_k, dvS)
        d_convS_k = dot(d_advS_k, u_test)
        a_skel += th * (d_rhoS_k * convS_k + rhoS_k * d_convS_k) * dx

        d_div_rhoS_vS_k = rho_s0_c * d_divBvS_k
        a_skel += th * (d_div_rhoS_vS_k * dot(vS_k, u_test) + div_rhoS_vS_k * dot(dvS, u_test)) * dx

    # Optional facet stabilization for u (CIP/ghost-penalty on interior facets).
    # This improves robustness when u becomes weakly constrained (e.g. α→0 in
    # parts of the domain) and helps avoid Newton stagnation in multi-step runs.
    # The term is consistent because [∂_n u]=0 for smooth u.
    if float(u_cip) != 0.0 and ds_cip is not None:
        n_int = FacetNormal()
        h_F = avg(MeshSize())
        vmag = _sqrt(inner(vS_n, vS_n) + _c(1.0e-12))
        scale = inv_dt + vmag / (h_F + _c(1.0e-12))
        tau_u_cip = _c(float(u_cip)) * (h_F * h_F * h_F) * scale
        # Weight by (1-α^n) to primarily stabilize the *extension* region (where α≈0)
        # without polluting the physical biofilm chunk (α≈1). Using α^n keeps the
        # Jacobian free of extra α-coupling from the stabilization weight.
        w_u_cip = avg(_one_minus(alpha_n))
        r_skeleton += tau_u_cip * w_u_cip * _grad_inner_jump(u_k, u_test, n_int) * ds_cip
        a_skel += tau_u_cip * w_u_cip * _grad_inner_jump(du, u_test, n_int) * ds_cip

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
            th * alpha_k * a_prev * dot(t_adh_k, u_test)
            + one_m_th * alpha_n * a_prev * dot(t_adh_n, u_test)
        ) * ds_adh

        # Jacobian (k-part only): δ[ α a_prev t_adh(u,vS) ].
        du_nvec = _proj_n(du)
        du_tvec = du - du_nvec
        dvS_nvec = _proj_n(dvS)
        dvS_tvec = dvS - dvS_nvec
        dt_adh = k_n_c * du_nvec + k_t_c * du_tvec + g_n_c * dvS_nvec + g_t_c * dvS_tvec

        a_skel += th * (dalpha * a_prev * dot(t_adh_k, u_test) + alpha_k * a_prev * dot(dt_adh, u_test)) * ds_adh

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
        vmag = _sqrt(inner(vS_n, vS_n) + _c(1.0e-12))
        denom = (_c(2.0) * inv_dt) * (_c(2.0) * inv_dt) + (_c(2.0) * vmag / (h_p + _c(1.0e-12))) * (
            _c(2.0) * vmag / (h_p + _c(1.0e-12))
        )
        tau_supg = _c(float(phi_supg)) / _sqrt(denom + _c(1.0e-16))
        w_supg = dot(grad(phi_test), vS_n)

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
        vmag = _sqrt(inner(vS_n, vS_n) + _c(1.0e-12))
        scale = inv_dt + vmag / (h_F + _c(1.0e-12))
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
    G_k = _G(S_k, phi_k, k_g=_c(float(k_g)), mu_max=_c(float(mu_max)), K_S=_c(float(K_S)))
    G_n = _G(S_n, phi_n, k_g=_c(float(k_g)), mu_max=_c(float(mu_max)), K_S=_c(float(K_S)))

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
    f_alpha_k += th * (dot(grad(alpha_k), vS_k) - G_k * alpha_k * _one_minus(alpha_k) + surf_coef_prev * delta_k)
    f_alpha_k += one_m_th * (dot(grad(alpha_n), vS_n) - G_n * alpha_n * _one_minus(alpha_n) + surf_coef_prev * delta_n)

    r_alpha = alpha_test * f_alpha_k * dx
    r_alpha += D_alpha_c * inner(grad(alpha_k), grad(alpha_test)) * dx
    # Allen–Cahn regularization: -(Mγ ε Δα) + (Mγ/ε) W'(α) in the residual.
    if float(alpha_cahn_M) != 0.0 and float(alpha_cahn_gamma) != 0.0:
        # W'(α) for W(α)=α^2(1-α)^2 is 2α(1-α)(1-2α).
        Wp_k = _c(2.0) * alpha_k * _one_minus(alpha_k) * _one_minus(_c(2.0) * alpha_k)
        r_alpha += (M_gamma_alpha * eps_alpha_c) * inner(grad(alpha_k), grad(alpha_test)) * dx
        r_alpha += alpha_test * ((M_gamma_alpha / eps_alpha_c) * Wp_k) * dx
    r_alpha += -alpha_test * f_alpha * dx

    # Jacobian (k-part only)
    # δG
    dG = (_c(float(k_g)) * _one_minus(phi_k) * (dmon_dS * dS) + (-_c(float(k_g)) * mon_k) * dphi)
    # δ[ G α(1-α) ] = (δG) α(1-α) + G (1-2α) δα
    dalpha_logistic = _one_minus(_c(2.0) * alpha_k) * dalpha

    a_alpha = alpha_test * (dalpha * inv_dt) * dx
    a_alpha += alpha_test * th * (dot(grad(alpha_k), dvS) + dot(grad(dalpha), vS_k)) * dx
    # δ[ (D_det_prev + crack_coef_prev) * δ(α) ] = (D_det_prev + crack_coef_prev) * δ'(α) δα
    # (surf coefficients are lagged).
    d_delta_k = _c(4.0) * (_one_minus(_c(2.0) * alpha_k)) * dalpha
    d_surf = surf_coef_prev * d_delta_k
    a_alpha += alpha_test * th * (-(dG * alpha_k * _one_minus(alpha_k) + G_k * dalpha_logistic) + d_surf) * dx
    a_alpha += D_alpha_c * inner(grad(dalpha), grad(alpha_test)) * dx
    if float(alpha_cahn_M) != 0.0 and float(alpha_cahn_gamma) != 0.0:
        # W''(α) = 2 - 12α + 12α^2
        # NOTE: keep the function-like term on the left; float - VecOpInfo is not supported.
        Wpp_k = (-_c(12.0) * alpha_k) + (_c(12.0) * (alpha_k * alpha_k)) + _c(2.0)
        a_alpha += (M_gamma_alpha * eps_alpha_c) * inner(grad(dalpha), grad(alpha_test)) * dx
        a_alpha += alpha_test * ((M_gamma_alpha / eps_alpha_c) * Wpp_k * dalpha) * dx

    # Optional consistent stabilization for advection-dominated α (useful with D_alpha=0).
    # - SUPG: τ (vS·∇w) R_alpha
    # - CIP:  γ h^3 (1/dt + |vS|/h) <[∂_n α],[∂_n w]>_F on interior facets
    #
    # Notes:
    # - We use lagged vS_n in τ and in the test-direction to keep the Jacobian
    #   consistent (no extra u-coupling from the stabilization weights).
    # - CIP only affects regions where ∇α is non-zero (i.e. the diffuse interface),
    #   and remains consistent because [∂_n α]=0 for smooth α.
    if float(alpha_supg) != 0.0:
        h_a = MeshSize()
        vmag = _sqrt(inner(vS_n, vS_n) + _c(1.0e-12))
        denom = (_c(2.0) * inv_dt) * (_c(2.0) * inv_dt) + (_c(2.0) * vmag / (h_a + _c(1.0e-12))) * (
            _c(2.0) * vmag / (h_a + _c(1.0e-12))
        )
        tau_supg = _c(float(alpha_supg)) / _sqrt(denom + _c(1.0e-16))
        w_supg = dot(grad(alpha_test), vS_n)
        r_alpha += tau_supg * w_supg * f_alpha_k * dx

        df_alpha_k = dalpha * inv_dt
        df_alpha_k += th * (
            dot(grad(alpha_k), dvS)
            + dot(grad(dalpha), vS_k)
            - (dG * alpha_k * _one_minus(alpha_k) + G_k * dalpha_logistic)
            + d_surf
        )
        a_alpha += tau_supg * w_supg * df_alpha_k * dx

    if float(alpha_cip) != 0.0 and ds_cip is not None:
        n_int = FacetNormal()
        h_F = avg(MeshSize())
        vmag = _sqrt(inner(vS_n, vS_n) + _c(1.0e-12))
        scale = inv_dt + vmag / (h_F + _c(1.0e-12))
        tau_cip = _c(float(alpha_cip)) * (h_F * h_F * h_F) * scale
        r_alpha += tau_cip * _grad_inner_jump(alpha_k, alpha_test, n_int) * ds_cip
        a_alpha += tau_cip * _grad_inner_jump(dalpha, alpha_test, n_int) * ds_cip

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
            df_dmg_k += th * (dalpha * dot(grad(d_k), vS_k) + alpha_k * (dot(grad(dd), vS_k) + dot(grad(d_k), du) * inv_dt))
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

            damage_pf_driver_key = str(damage_pf_driver).strip().lower()
            if damage_pf_driver_key in {"von_mises", "vm", "von-mises"}:
                # Legacy: scale a von-Mises-based proxy into an energy density.
                psi0_val = float(damage_psi0)
                if psi0_val <= 0.0:
                    psi0_val = Gc_val / max(ell_val, 1.0e-12)
                H_prev = _c(psi0_val) * drive_vm
            elif damage_pf_driver_key in {"miehe", "miehe_energy", "energy", "psi_plus", "psi+"}:
                # Miehe-type tensile energy density ψ⁺(u) for linear elasticity:
                #   ψ⁺ = μ ||ε⁺||² + (λ/2) ⟨tr ε⟩₊².
                if solid_model_key not in {"linear", "small_strain", "linear_elastic"}:
                    raise ValueError(
                        "damage_pf_driver='miehe_energy' currently requires solid_model='linear'. "
                        "Use damage_pf_driver='von_mises' for neo-Hookean."
                    )
                eta_pos = float(damage_eta_pos)
                disc_reg = 1.0e-16
                eps_un = _epsilon(u_n)
                eps_plus_un, _, _, _, _ = spectral_positive_part_2x2_sym(eps_un, eta_pos=eta_pos, disc_reg=disc_reg)
                tr_eps = div(u_n)
                tr_pos = _smooth_pos_u(tr_eps, eta=eta_pos)
                psi_plus = mu_s * inner(eps_plus_un, eps_plus_un) + _c(0.5) * lambda_s * (tr_pos * tr_pos)
                scale = float(damage_psi0)
                if scale > 0.0:
                    psi_plus = _c(scale) * psi_plus
                H_prev = psi_plus
            else:
                raise ValueError(f"Unknown damage_pf_driver {damage_pf_driver!r}.")

            DtS_d_k = (d_k - d_n) * inv_dt
            DtS_d_k += th * dot(grad(d_k), vS_k) + one_m_th * dot(grad(d_n), vS_n)

            f_pf_k = eta_d_c * DtS_d_k + Gc_over_l * d_k - _c(2.0) * one_m_d_k * H_prev

            r_damage = alpha_k * d_test * f_pf_k * dx
            r_damage += alpha_k * Gc_l * inner(grad(d_k), grad(d_test)) * dx
            if float(damage_gamma_out) != 0.0:
                r_damage += gamma_out_c * w_phi_fluid_k * d_k * d_test * dx

            d_DtS_d_k = dd * inv_dt + th * (dot(grad(dd), vS_k) + dot(grad(d_k), du) * inv_dt)
            df_pf_k = eta_d_c * d_DtS_d_k + Gc_over_l * dd + _c(2.0) * H_prev * dd

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

    div_CSv_k = CSk * div(v_k) + S_k * dot(gradC_k, v_k) + C_k * dot(grad(S_k), v_k)
    div_CSv_n = CSn * div(v_n) + S_n * dot(gradC_n, v_n) + C_n * dot(grad(S_n), v_n)

    r_sub = S_test * ((CSk - CSn) * inv_dt) * dx
    r_sub += S_test * (th * div_CSv_k + one_m_th * div_CSv_n) * dx
    r_sub += D_S_c * th * inner(grad(S_k), grad(S_test)) * dx + D_S_c * one_m_th * inner(grad(S_n), grad(S_test)) * dx
    r_sub += S_test * (th * RS_k + one_m_th * RS_n) * dx
    r_sub += -S_test * f_S * dx

    # Jacobian (k-part only)
    dCSk = dC * S_k + _capacity(alpha_k, phi_k) * dS
    dRS = rho_s_star_c * ((_c(1.0) / Y_c) * dPi)  # RS = (rho_s_star/Y) (Π_b/ρ_s*)

    d_div_CSv_k = dCSk * div(v_k) + CSk * div(dv)
    d_div_CSv_k += dS * dot(gradC_k, v_k) + S_k * dot(dgradC_k, v_k) + S_k * dot(gradC_k, dv)
    d_div_CSv_k += dC_k * dot(grad(S_k), v_k) + C_k * dot(grad(dS), v_k) + C_k * dot(grad(S_k), dv)

    a_sub = S_test * (dCSk * inv_dt) * dx
    a_sub += S_test * th * d_div_CSv_k * dx
    a_sub += D_S_c * th * inner(grad(dS), grad(S_test)) * dx
    a_sub += S_test * th * dRS * dx

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
    residual_form = r_mom + r_mass + r_skeleton + r_phi + r_alpha + r_sub
    jacobian_form = a_mom + a_mass + a_skel + a_phi + a_alpha + a_sub
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
        r_skeleton=r_skeleton,
        r_phi=r_phi,
        r_alpha=r_alpha,
        r_damage=r_damage,
        r_substrate=r_sub,
        a_momentum=a_mom,
        a_mass=a_mass,
        a_skeleton=a_skel,
        a_phi=a_phi,
        a_alpha=a_alpha,
        a_damage=a_damage,
        a_substrate=a_sub,
        r_detached=r_detached,
        a_detached=a_detached,
    )
