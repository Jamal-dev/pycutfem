"""Final-form one-domain FPSI builder with explicit bulk/interface separation.

This module implements the formulation stated in
examples/biofilms/benchmarks/seboldt/final_form.md.

Primary physical variables:
  - free-fluid velocity/pressure:     v_f, p
  - pore velocity/pressure:           v_p, p_p
  - solid velocity / ref-map field:   v_s, u
  - diffuse geometry/support:         alpha
  - porosity and intrinsic solid rho: phi, rho_s

The interface transfer is kept explicit:
  - mass transfer:     grad(alpha)·(rho_f v_f - rho_s phi v_p - rho_s (1-phi) v_s)
  - traction transfer: grad(alpha)·(sigma_f - phi sigma_p - (1-phi) sigma_s)
  - kinematic transfer: |grad(alpha)| * (u_t + (grad(u)) v_s - v_s)
"""

from __future__ import annotations

import math
from hashlib import blake2b

import numpy as np

from pycutfem.ufl.expressions import (
    Constant,
    FacetNormal,
    Identity,
    MeshSize,
    avg,
    div,
    dot,
    grad,
    inner,
    jump,
    tanh,
)

from ..shared.nonlinear_solid_refmap import (
    dsigma_neo_hookean_seboldt,
    dsigma_svk,
    sigma_neo_hookean_seboldt,
    sigma_svk,
)
from .one_domain import BiofilmOneDomainForms

_LITERAL_CONSTANT_CACHE: dict[float, Constant] = {}
_NAMED_CONSTANT_CACHE: dict[tuple[str, int, tuple[int, ...], tuple[float, ...]], Constant] = {}


def _c(val: float) -> Constant:
    sval = float(val)
    const = _LITERAL_CONSTANT_CACHE.get(sval)
    if const is None:
        token = blake2b(format(sval, ".16g").encode("ascii"), digest_size=8).hexdigest()
        const = _named_c(f"ff_lit_{token}", sval)
        _LITERAL_CONSTANT_CACHE[sval] = const
    return const


def _named_c(name: str, value, *, dim: int | None = None) -> Constant:
    if isinstance(value, Constant):
        raw_value = value.value
        source_dim = int(getattr(value, "dim", np.asarray(raw_value).ndim))
        target_dim = source_dim if dim is None else int(dim)
    else:
        raw_value = value
        target_dim = int(np.asarray(raw_value).ndim) if dim is None else int(dim)
    raw_array = np.asarray(raw_value, dtype=float)
    flat_key = (float(raw_array),) if raw_array.ndim == 0 else tuple(float(v) for v in raw_array.ravel())
    shape_key = tuple(int(v) for v in raw_array.shape)
    cache_key = (str(name), int(target_dim), shape_key, flat_key)
    const = _NAMED_CONSTANT_CACHE.get(cache_key)
    if const is None:
        if isinstance(value, Constant) and source_dim == target_dim:
            const = value
        else:
            const = Constant(raw_value, dim=target_dim)
        _NAMED_CONSTANT_CACHE[cache_key] = const
    try:
        const._jit_name = str(name)
    except Exception:
        pass
    try:
        const._name = str(name)
    except Exception:
        pass
    return const


def _one_minus(expr):
    return (-expr) + _c(1.0)


def _support_indicator_mode_key(raw) -> str:
    key = str(raw or "raw").strip().lower().replace("-", "_")
    if key not in {"raw", "tanh", "parabolic_envelope"}:
        raise ValueError(
            f"Unsupported support_indicator_mode={raw!r}. Use 'raw', 'tanh', or 'parabolic_envelope'."
        )
    return key


def _parabolic_interface_envelope_expr(alpha_expr):
    delta = alpha_expr - _c(0.5)
    return _c(4.0) * delta * delta


def _parabolic_interface_envelope_prime_expr(alpha_expr):
    return _c(8.0) * (alpha_expr - _c(0.5))


def _smooth_support_indicator_expr(alpha_expr, *, beta: float):
    beta_val = float(beta)
    if beta_val <= 0.0:
        return alpha_expr
    z = _c(beta_val) * (alpha_expr - _c(0.5))
    return _c(0.5) * (_c(1.0) + tanh(z))


def _smooth_support_indicator_prime_expr(alpha_expr, *, beta: float):
    beta_val = float(beta)
    if beta_val <= 0.0:
        return _c(1.0)
    z = _c(beta_val) * (alpha_expr - _c(0.5))
    th = tanh(z)
    return _c(0.5 * beta_val) * (_c(1.0) - th * th)


def _support_indicator_expr(alpha_expr, *, beta: float, mode: str = "raw"):
    mode_key = _support_indicator_mode_key(mode)
    if mode_key == "parabolic_envelope":
        envelope = _parabolic_interface_envelope_expr(alpha_expr)
        return alpha_expr * envelope
    return _smooth_support_indicator_expr(alpha_expr, beta=beta)


def _support_indicator_prime_expr(alpha_expr, *, beta: float, mode: str = "raw"):
    mode_key = _support_indicator_mode_key(mode)
    if mode_key == "parabolic_envelope":
        envelope = _parabolic_interface_envelope_expr(alpha_expr)
        denvelope = _parabolic_interface_envelope_prime_expr(alpha_expr)
        return envelope + alpha_expr * denvelope
    return _smooth_support_indicator_prime_expr(alpha_expr, beta=beta)


def _free_indicator_expr(alpha_expr, *, beta: float, mode: str = "raw"):
    mode_key = _support_indicator_mode_key(mode)
    if mode_key == "parabolic_envelope":
        envelope = _parabolic_interface_envelope_expr(alpha_expr)
        return _one_minus(alpha_expr) * envelope
    return _one_minus(_support_indicator_expr(alpha_expr, beta=beta, mode=mode_key))


def _free_indicator_prime_expr(alpha_expr, *, beta: float, mode: str = "raw"):
    mode_key = _support_indicator_mode_key(mode)
    if mode_key == "parabolic_envelope":
        envelope = _parabolic_interface_envelope_expr(alpha_expr)
        denvelope = _parabolic_interface_envelope_prime_expr(alpha_expr)
        return -envelope + _one_minus(alpha_expr) * denvelope
    return -_support_indicator_prime_expr(alpha_expr, beta=beta, mode=mode_key)


def _vector_component(vec_expr, idx: int):
    try:
        return vec_expr[idx]
    except Exception:
        basis = _named_c(
            "ff_basis_x" if int(idx) == 0 else "ff_basis_y",
            [1.0, 0.0] if int(idx) == 0 else [0.0, 1.0],
            dim=1,
        )
        return dot(vec_expr, basis)


def _grad_inner_jump(u, v, n):
    """Penalty on the jump of the normal derivative across an interior facet."""
    ncomp = 0
    if hasattr(u, "num_components"):
        try:
            ncomp = int(getattr(u, "num_components", 0))
        except Exception:
            ncomp = 0
    if ncomp <= 0:
        shape_u = getattr(u, "shape", None)
        if isinstance(shape_u, tuple) and len(shape_u) == 1:
            try:
                ncomp = int(shape_u[0])
            except Exception:
                ncomp = 0
    if ncomp <= 1:
        gu = grad(u)
        gv = grad(v)
        ju = jump(gu[0]) * n[0] + jump(gu[1]) * n[1]
        jv = jump(gv[0]) * n[0] + jump(gv[1]) * n[1]
        return ju * jv

    acc = _c(0.0)
    for i in range(ncomp):
        gui = grad(_vector_component(u, i))
        gvi = grad(_vector_component(v, i))
        ju = jump(gui[0]) * n[0] + jump(gui[1]) * n[1]
        jv = jump(gvi[0]) * n[0] + jump(gvi[1]) * n[1]
        acc += ju * jv
    return acc


def _form_add(lhs, rhs):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return lhs + rhs


def _eps(v):
    return _c(0.5) * (grad(v) + grad(v).T)


def _tensor_trace(A, *, dim: int):
    acc = _c(0.0)
    for i in range(int(dim)):
        acc += A[i, i]
    return acc


def _deviatoric_tensor(A, *, dim: int):
    return A - (_tensor_trace(A, dim=int(dim)) / _c(float(dim))) * Identity(int(dim))

def _deviatoric_shear_stress(u, *, dim: int, mu:float):
    deviatoric_strain = _deviatoric_tensor(_eps(u), dim=int(dim))
    return _c(2.0) * mu * deviatoric_strain

def _shear_stress(u, *, mu: float):
    return _c(2.0) * mu * _eps(u)

def _linear_stress_fluid_model(u, p, *, mu: float, dim: int):
    return _shear_stress(u, mu=mu) - p * Identity(int(dim)) 

def _linear_stress_solid_model(u, *, mu: float, lambda_: float, dim: int):
    return _c(2.0) * mu * _eps(u) + lambda_ * div(u) * Identity(int(dim))

def _solid_model_key(model_key: str) -> str:
    return str(model_key or "linear").strip().lower().replace("-", "_")


def _solid_stress_and_tangent(*, solid_model: str, u_k, du, mu_s, lambda_s, dim: int):
    key = _solid_model_key(solid_model)
    if key in {"linear", "small_strain", "linear_elastic"}:
        sigma_k = _linear_stress_solid_model(u_k, mu=mu_s, lambda_=lambda_s, dim=int(dim))
        dsigma = _linear_stress_solid_model(du, mu=mu_s, lambda_=lambda_s, dim=int(dim))
        return sigma_k, dsigma
    if key in {"neo_hookean_seboldt", "seboldt_neo_hookean"}:
        return (
            sigma_neo_hookean_seboldt(u_k, mu_s, lambda_s, dim=int(dim)),
            dsigma_neo_hookean_seboldt(u_k, du, mu_s, lambda_s, dim=int(dim)),
        )
    if key in {"stvk", "svk", "saint_venant_kirchhoff", "saint_venant_kirchhoff"}:
        return (
            sigma_svk(u_k, mu_s, lambda_s, dim=int(dim)),
            dsigma_svk(u_k, du, mu_s, lambda_s, dim=int(dim)),
        )
    raise NotImplementedError(f"final_form does not support solid_model={solid_model!r}.")


def _final_form_interface_formulation_key(raw) -> str:
    key = str(raw or "tensor").strip().lower().replace("-", "_")
    if key not in {"tensor", "decomposed"}:
        raise ValueError(
            f"Unsupported interface_formulation={raw!r}. Use 'tensor' or 'decomposed'."
        )
    return key


def build_biofilm_one_domain_final_form(
    *,
    v_k,
    p_k,
    vP_k,
    p_pore_k,
    vS_k,
    u_k,
    alpha_k,
    phi_k,
    rho_s_k,
    v_n,
    p_n,
    vP_n,
    p_pore_n,
    vS_n,
    u_n,
    alpha_n,
    phi_n,
    rho_s_n,
    dv,
    dp,
    dvP,
    dp_pore,
    dvS,
    du,
    dalpha,
    dphi,
    drho_s,
    dmu_mass,
    dmu_normal,
    dmu_tangent,
    dmu_kin=None,
    dlm_vf=None,
    dlm_p=None,
    dlm_vP=None,
    dlm_vS=None,
    dlm_p_pore=None,
    dlm_phi=None,
    dlm_u=None,
    v_test,
    q_test,
    q_pore_test,
    vP_test,
    vS_test,
    u_test,
    alpha_test,
    phi_test,
    rho_s_test,
    mu_mass_test,
    mu_normal_test,
    mu_tangent_test,
    mu_kin_test=None,
    lm_vf_test=None,
    lm_p_test=None,
    lm_vP_test=None,
    lm_vS_test=None,
    lm_p_pore_test=None,
    lm_phi_test=None,
    lm_u_test=None,
    mu_mass_k,
    mu_normal_k,
    mu_tangent_k,
    mu_kin_k=None,
    lm_vf_k=None,
    lm_p_k=None,
    lm_vP_k=None,
    lm_vS_k=None,
    lm_p_pore_k=None,
    lm_phi_k=None,
    lm_u_k=None,
    dx,
    dt,
    rho_f,
    mu_f,
    kappa_inv,
    mu_s,
    lambda_s,
    solid_visco_eta: float = 0.0,
    rho_s_fluid_gauge_strength: float = 1.0,
    mu_b=None,
    mu_b_model: str = "phi_mu",
    gamma_phi: float = 0.0,
    gamma_v: float = 0.0,
    v_extension_mode: str = "h1",
    gamma_v_pin: float = 0.0,
    gamma_p: float = 0.0,
    p_extension_mode: str = "h1",
    gamma_p_pin: float = 0.0,
    gamma_vP: float = 0.0,
    vP_extension_mode: str = "h1",
    gamma_vP_pin: float = 0.0,
    gamma_p_pore: float = 0.0,
    p_pore_extension_mode: str = "h1",
    gamma_p_pore_pin: float = 0.0,
    gamma_rho_s: float = 0.0,
    rho_s_extension_mode: str = "h1",
    gamma_rho_s_pin: float = 0.0,
    rho_s_ref: float = 1.0,
    constant_rho_s: bool = False,
    gamma_u: float = 0.0,
    u_extension_mode: str = "h1",
    gamma_u_pin: float = 0.0,
    domain_lm_aug_gamma: float = 10.0,
    domain_lm_free_weight_mode: str = "diffuse",
    domain_lm_free_alpha_max: float | None = None,
    mass_lm_aug_gamma: float = 0.0,
    normal_lm_aug_gamma: float = 0.0,
    interface_band_extension_gamma: float = 0.0,
    u_cip: float = 0.0,
    u_cip_weight: str = "fluid",
    gamma_vS: float | None = None,
    vS_extension_mode: str | None = None,
    gamma_vS_pin: float | None = None,
    vS_cip: float = 0.0,
    ds_cip=None,
    fluid_convection: str = "full",
    pore_convection: str = "full",
    skeleton_inertia_convection: str = "full",
    solid_model: str = "linear",
    solid_volumetric_split: bool = False,
    solid_volumetric_penalty: float = 1.0,
    phi_mode: str = "transport",
    phi_b: float = 0.18,
    alpha_biot: float = 1.0,
    normal_pressure_scale: float = 1.0,
    normal_constraint_carrier: str = "multiplier",
    rigid_darcy_head_mode: bool = False,
    bjs_coefficient: float = 0.0,
    solid_interface_traction_weight: float = 1.0,
    mass_interface_weight: float = 1.0,
    normal_interface_weight: float = 1.0,
    disable_interface_physics: bool = False,
    disable_normal_interface: bool = False,
    domain_lm: bool = False,
    domain_lm_free_tie_vP_to_vf: bool = False,
    domain_lm_free_tie_p_pore_to_p: bool = False,
    quasi_static_porous_media: bool = False,
    quasi_static_flip_pore_stress_sign: bool = False,
    quasi_static_disable_pore_momentum: bool = False,
    quasi_static_disable_solid_momentum: bool = False,
    quasi_static_use_combined_porous_momentum: bool = False,
    direct_interface_transfer: bool = False,
    support_indicator_beta: float = 0.0,
    support_indicator_mode: str = "raw",
    alpha_advect_with: str | None = None,
    alpha_advection_form: str = "conservative_weak",
    support_physics: str = "internal_conversion",
    alpha_vS_gate_alpha0: float = 0.0,
    alpha_vS_gate_power: int = 8,
    ds_alpha_transport=None,
    interface_formulation: str = "tensor",
    dpi_s=None,
    pi_s_test=None,
    pi_s_k=None,
    pi_s_n=None,
) -> BiofilmOneDomainForms:
    dim = 2
    dt_c = _named_c("ff_dt", dt)
    inv_dt = _c(1.0) / dt_c
    rho_f_c = _named_c("ff_rho_f", rho_f)
    mu_f_c = _named_c("ff_mu_f", mu_f)
    mu_b_c = None if mu_b is None else _named_c("ff_mu_b", mu_b)
    kappa_inv_c = _named_c("ff_kappa_inv", kappa_inv)
    mu_s_c = _named_c("ff_mu_s", mu_s)
    lambda_s_c = _named_c("ff_lambda_s", lambda_s)
    eta_s_c = _named_c("ff_solid_visco_eta", float(solid_visco_eta))
    rho_s_gauge_c = _named_c("ff_rho_s_gauge_strength", float(rho_s_fluid_gauge_strength))
    gamma_phi_c = _named_c("ff_gamma_phi", float(gamma_phi))
    gamma_v_c = _named_c("ff_gamma_v", float(gamma_v))
    gamma_v_pin_c = _named_c("ff_gamma_v_pin", float(gamma_v_pin))
    gamma_p_c = _named_c("ff_gamma_p", float(gamma_p))
    gamma_p_pin_c = _named_c("ff_gamma_p_pin", float(gamma_p_pin))
    gamma_vP_c = _named_c("ff_gamma_vP", float(gamma_vP))
    gamma_vP_pin_c = _named_c("ff_gamma_vP_pin", float(gamma_vP_pin))
    gamma_p_pore_c = _named_c("ff_gamma_p_pore", float(gamma_p_pore))
    gamma_p_pore_pin_c = _named_c("ff_gamma_p_pore_pin", float(gamma_p_pore_pin))
    gamma_rho_s_c = _named_c("ff_gamma_rho_s", float(gamma_rho_s))
    gamma_rho_s_pin_c = _named_c("ff_gamma_rho_s_pin", float(gamma_rho_s_pin))
    rho_s_ref_c = _named_c("ff_rho_s_ref", float(rho_s_ref))
    gamma_u_c = _named_c("ff_gamma_u", float(gamma_u))
    gamma_u_pin_c = _named_c("ff_gamma_u_pin", float(gamma_u_pin))
    domain_lm_aug_gamma_c = _named_c("ff_domain_lm_aug_gamma", float(domain_lm_aug_gamma))
    mass_lm_aug_gamma_c = _named_c("ff_mass_lm_aug_gamma", float(mass_lm_aug_gamma))
    normal_lm_aug_gamma_c = _named_c("ff_normal_lm_aug_gamma", float(normal_lm_aug_gamma))
    gamma_band_ext_c = _named_c("ff_gamma_band_ext", float(interface_band_extension_gamma))
    u_cip_c = _named_c("ff_u_cip", float(u_cip))
    phi_b_c = _named_c("ff_phi_b", float(phi_b))
    alpha_biot_c = _named_c("ff_alpha_biot", float(alpha_biot))
    normal_pressure_scale_c = _named_c("ff_normal_pressure_scale", float(normal_pressure_scale))
    bjs_coeff_c = _named_c("ff_bjs_coefficient", float(bjs_coefficient))
    solid_interface_traction_weight_c = _named_c(
        "ff_solid_interface_traction_weight",
        float(solid_interface_traction_weight),
    )
    def _interface_weight_expr_and_value(name: str, raw_weight):
        if isinstance(raw_weight, Constant):
            weight_raw = raw_weight.value
            weight_arr = np.asarray(weight_raw, dtype=float).ravel()
            if weight_arr.size != 1:
                raise ValueError(f"{name} must be scalar.")
            weight_val = float(weight_arr[0])
            if (not math.isfinite(weight_val)) or not (0.0 <= weight_val <= 1.0):
                raise ValueError(f"{name} must be finite and lie in [0, 1].")
            return raw_weight, weight_val
        try:
            weight_arr = np.asarray(raw_weight, dtype=float).ravel()
        except Exception:
            return raw_weight, None
        if weight_arr.size != 1:
            raise ValueError(f"{name} must be scalar.")
        weight_val = float(weight_arr[0])
        if (not math.isfinite(weight_val)) or not (0.0 <= weight_val <= 1.0):
            raise ValueError(f"{name} must be finite and lie in [0, 1].")
        return _named_c(f"ff_{name}", float(weight_val)), weight_val

    mass_interface_weight_c, mass_interface_weight_val = _interface_weight_expr_and_value(
        "mass_interface_weight",
        mass_interface_weight,
    )
    normal_interface_weight_c, normal_interface_weight_val = _interface_weight_expr_and_value(
        "normal_interface_weight",
        normal_interface_weight,
    )
    disable_interface_physics = bool(disable_interface_physics)
    one_m_mass_interface_weight_c = _c(1.0) - mass_interface_weight_c
    sqrt_kappa_inv_c = kappa_inv_c ** _c(0.5)
    use_bjs_tangential_law = abs(float(bjs_coefficient)) > 0.0
    zero = _c(0.0)

    def _safe_zero_scalar_linear_form(anchor_expr, test_expr):
        if anchor_expr is None or test_expr is None:
            raise ValueError("safe zero scalar linear form requires both anchor_expr and test_expr.")
        # Keep a nontrivial coefficient/test product so the python backend does
        # not simplify disabled scalar rows to a pure 0*dx functional.
        return _c(0.0) * anchor_expr * test_expr * dx

    def _safe_zero_vector_linear_form(anchor_vec, test_vec):
        if anchor_vec is None or test_vec is None:
            raise ValueError("safe zero vector linear form requires both anchor_vec and test_vec.")
        return _c(0.0) * dot(anchor_vec, test_vec) * dx
    phi_mode_key = str(phi_mode or "transport").strip().lower().replace("-", "_")
    if phi_mode_key not in {"transport", "alpha_closure"}:
        raise ValueError(
            f"Unsupported phi_mode={phi_mode!r}. Use 'transport' or 'alpha_closure'."
        )
    normal_carrier_key = str(normal_constraint_carrier or "multiplier").strip().lower().replace("-", "_")
    if normal_carrier_key not in {"multiplier", "p_pore"}:
        raise ValueError(
            "Unsupported normal_constraint_carrier="
            f"{normal_constraint_carrier!r}. Use 'multiplier' or 'p_pore'."
        )
    rigid_darcy_head_mode = bool(rigid_darcy_head_mode)
    constant_rho_s = bool(constant_rho_s)
    interface_formulation_key = _final_form_interface_formulation_key(interface_formulation)
    solid_model_key = _solid_model_key(solid_model)
    solid_volumetric_split = bool(solid_volumetric_split)
    support_indicator_beta_val = float(support_indicator_beta)
    support_indicator_mode_key = _support_indicator_mode_key(support_indicator_mode)
    disable_normal_interface = bool(disable_normal_interface)
    use_domain_lm = bool(domain_lm)
    quasi_static_porous_media = bool(quasi_static_porous_media)
    quasi_static_flip_pore_stress_sign = bool(quasi_static_flip_pore_stress_sign)
    quasi_static_disable_pore_momentum = bool(quasi_static_disable_pore_momentum)
    quasi_static_disable_solid_momentum = bool(quasi_static_disable_solid_momentum)
    quasi_static_use_combined_porous_momentum = bool(quasi_static_use_combined_porous_momentum)
    direct_interface_transfer = bool(direct_interface_transfer)
    if direct_interface_transfer and (not quasi_static_porous_media):
        raise ValueError(
            "direct_interface_transfer is currently implemented only on quasi_static_porous_media=True."
        )
    if quasi_static_porous_media:
        if not constant_rho_s:
            raise ValueError(
                "quasi_static_porous_media currently requires constant_rho_s=True so the porous block "
                "uses only div(phi v_p + (1-phi) v_s)=0 without a separate solid-mass row."
            )
        # The simplified direct-transfer debug hook replaces the multiplier
        # rows with explicit grad(alpha)-driven transfer terms. When that hook
        # is disabled, the quasi-static branch must keep the interface rows
        # active; otherwise the porous side can collapse to the dead branch
        # with nonzero diagnostic jumps and zero support response.
        if direct_interface_transfer:
            disable_interface_physics = True
    if disable_normal_interface:
        normal_interface_weight_c = _named_c("ff_normal_interface_weight", 0.0)
        normal_interface_weight_val = 0.0
    else:
        normal_interface_weight_c = normal_interface_weight_c
    one_m_normal_interface_weight_c = _c(1.0) - normal_interface_weight_c
    alpha_support_k = _smooth_support_indicator_expr(alpha_k, beta=support_indicator_beta_val)
    alpha_support_n = _smooth_support_indicator_expr(alpha_n, beta=support_indicator_beta_val)
    dalpha_support = _smooth_support_indicator_prime_expr(alpha_k, beta=support_indicator_beta_val) * dalpha
    F_support_k = _one_minus(alpha_support_k)
    dF_support = -dalpha_support
    # Bulk domain-LM constraints should use true diffuse domain-membership
    # weights, not the parabolic bulk envelope used by the simplified debug
    # bulk rows. Otherwise "free-side" LM terms stay active well into the
    # porous side when support_indicator_mode='parabolic_envelope'.
    alpha_domain_lm_k = alpha_support_k
    dalpha_domain_lm = dalpha_support
    free_domain_lm_weight_mode_key = str(domain_lm_free_weight_mode or "diffuse").strip().lower().replace("-", "_")
    if free_domain_lm_weight_mode_key == "diffuse":
        F_domain_lm_k = F_support_k
        dF_domain_lm = dF_support
    elif free_domain_lm_weight_mode_key in {"sharp", "sharp16", "free_interior", "interior"}:
        F_domain_lm_2 = F_support_k * F_support_k
        F_domain_lm_4 = F_domain_lm_2 * F_domain_lm_2
        F_domain_lm_8 = F_domain_lm_4 * F_domain_lm_4
        F_domain_lm_k = F_domain_lm_8 * F_domain_lm_8
        dF_domain_lm = _c(16.0) * (
            F_domain_lm_8 * F_domain_lm_4 * (F_support_k * F_support_k * F_support_k)
        ) * dF_support
    elif free_domain_lm_weight_mode_key in {"step_cutoff", "interior_cutoff", "step_interior"}:
        alpha_free_max_val = 0.05 if domain_lm_free_alpha_max is None else float(domain_lm_free_alpha_max)
        alpha_free_max_c = _named_c("ff_domain_lm_free_alpha_max", alpha_free_max_val)
        cutoff_sharpness_c = _named_c("ff_domain_lm_free_cutoff_sharpness", 100.0)
        # Keep the LM support fixed over a Newton step by using the previous
        # accepted alpha-state, then suppress it rapidly outside the deep
        # free-fluid interior.
        F_domain_lm_k = _c(0.5) * (_c(1.0) - tanh(cutoff_sharpness_c * (alpha_support_n - alpha_free_max_c)))
        dF_domain_lm = zero
    else:
        raise ValueError(
            "Unknown domain_lm_free_weight_mode="
            f"{domain_lm_free_weight_mode!r}. Use 'diffuse', 'sharp16', or 'step_cutoff'."
        )
    interface_band_weight_k = _c(4.0) * alpha_support_k * F_support_k
    dinterface_band_weight = _c(4.0) * (
        dalpha_support * F_support_k + alpha_support_k * dF_support
    )
    alpha_bulk_k = _support_indicator_expr(alpha_k, beta=support_indicator_beta_val, mode=support_indicator_mode_key)
    alpha_bulk_n = _support_indicator_expr(alpha_n, beta=support_indicator_beta_val, mode=support_indicator_mode_key)
    dalpha_bulk = _support_indicator_prime_expr(alpha_k, beta=support_indicator_beta_val, mode=support_indicator_mode_key) * dalpha
    div_alpha_bulk_vStest_k = alpha_bulk_k * div(vS_test) + dot(grad(alpha_bulk_k), vS_test)
    ddiv_alpha_bulk_vStest = dalpha_bulk * div(vS_test) + dot(grad(dalpha_bulk), vS_test)
    F_bulk_k = _free_indicator_expr(alpha_k, beta=support_indicator_beta_val, mode=support_indicator_mode_key)
    dF_bulk = _free_indicator_prime_expr(alpha_k, beta=support_indicator_beta_val, mode=support_indicator_mode_key) * dalpha
    selected_groups: list[str] = []
    if use_domain_lm:
        domain_lm_groups = {
            "vf": {
                "dlm": dlm_vf,
                "test": lm_vf_test,
                "state": lm_vf_k,
            },
            "p": {
                "dlm": dlm_p,
                "test": lm_p_test,
                "state": lm_p_k,
            },
            "vP": {
                "dlm": dlm_vP,
                "test": lm_vP_test,
                "state": lm_vP_k,
            },
            "vS": {
                "dlm": dlm_vS,
                "test": lm_vS_test,
                "state": lm_vS_k,
            },
            "p_pore": {
                "dlm": dlm_p_pore,
                "test": lm_p_pore_test,
                "state": lm_p_pore_k,
            },
            "phi": {
                "dlm": dlm_phi,
                "test": lm_phi_test,
                "state": lm_phi_k,
            },
            "u": {
                "dlm": dlm_u,
                "test": lm_u_test,
                "state": lm_u_k,
            },
        }
        selected_groups = [
            name for name, group in domain_lm_groups.items() if any(value is not None for value in group.values())
        ]
        if not selected_groups:
            raise ValueError("domain_lm=True requires at least one bulk domain LM target to be present.")
        missing: list[str] = []
        for name in selected_groups:
            group = domain_lm_groups[name]
            for key, value in group.items():
                if value is None:
                    missing.append(f"{name}:{key}")
        if missing:
            raise ValueError(
                "domain_lm=True requires each selected domain LM target to provide the LM field, test, and increment. "
                f"Missing: {', '.join(missing)}."
            )
    if rigid_darcy_head_mode and use_domain_lm:
        unsupported_rigid_targets = sorted(name for name in selected_groups if name not in {"vf"})
        if unsupported_rigid_targets:
            raise ValueError(
                "rigid_darcy_head_mode=True currently supports bulk domain LM only for the "
                f"free-velocity target. Unsupported targets: {', '.join(unsupported_rigid_targets)}."
            )
    one_m_phi_k = _one_minus(phi_k)
    one_m_phi_n = _one_minus(phi_n)
    rho_s_phys_k = rho_s_ref_c if constant_rho_s else rho_s_k
    rho_s_phys_n = rho_s_ref_c if constant_rho_s else rho_s_n
    drho_s_phys = zero if constant_rho_s else drho_s
    sigma_f_k = _c(2.0) * mu_f_c * _eps(v_k) - p_k * Identity(dim)
    dsigma_f = _c(2.0) * mu_f_c * _eps(dv) - dp * Identity(dim)
    mu_b_key = str(mu_b_model or "phi_mu").strip().lower()

    def _pore_effective_viscosity_terms(phi_expr, dphi_expr):
        # For the constant case it is mu_f and dmu=0
        if mu_b_key in {"mu", "const", "constant"}:
            return mu_f_c, zero
        # For this case we are assuming mu_b = phi * mu_f, dmu_b =  mu_f * dphi
        if mu_b_key in {"phi_mu", "phi*mu", "phi"}:
            return phi_expr * mu_f_c, dphi_expr * mu_f_c
        if mu_b_key in {"alpha_mu", "alpha", "mu_b", "biofilm_mu", "biofilm"}:
            if mu_b_c is None:
                raise ValueError("mu_b_model='alpha_mu' requires mu_b to be provided.")
            return mu_b_c, zero
        if mu_b_key in {"alpha_phi_mu", "alpha_phi", "alpha*phi_mu", "alpha*phi*mu"}:
            if mu_b_c is None:
                raise ValueError("mu_b_model='alpha_phi_mu' requires mu_b to be provided.")
            return phi_expr * mu_b_c, dphi_expr * mu_b_c
        raise ValueError(f"Unknown mu_b_model={mu_b_model!r}.")

    mu_pore_eff_k, dmu_pore_eff = _pore_effective_viscosity_terms(phi_k, dphi)
    sigma_p_k = _c(2.0) * mu_pore_eff_k * _eps(vP_k) - p_pore_k * Identity(dim)
    dsigma_p = (
        _c(2.0) * dmu_pore_eff * _eps(vP_k)
        + _c(2.0) * mu_pore_eff_k * _eps(dvP)
        - dp_pore * Identity(dim)
    )
    sigma_s_full_k, dsigma_s_full = _solid_stress_and_tangent(
        solid_model=str(solid_model),
        u_k=u_k,
        du=du,
        mu_s=mu_s_c,
        lambda_s=lambda_s_c,
        dim=int(dim),
    )
    total_pressure_ref_c = None
    total_pressure_ref_inv_c = None
    drained_bulk_over_total_pressure_ref_c = None
    vol_pen_c = None
    vol_drive_k = None
    dvol_drive = None
    if solid_volumetric_split:
        if solid_model_key not in {"linear", "small_strain", "linear_elastic", "neo_hookean_seboldt", "seboldt_neo_hookean"}:
            raise ValueError(
                "solid_volumetric_split is currently implemented for the linear and Seboldt Neo-Hookean solid models."
            )
        if pi_s_k is None or pi_s_n is None or dpi_s is None or pi_s_test is None:
            raise ValueError("solid_volumetric_split requires (pi_s_k, pi_s_n, dpi_s, pi_s_test).")
        if solid_model_key in {"neo_hookean_seboldt", "seboldt_neo_hookean"}:
            total_pressure_ref = float(lambda_s)
            drained_bulk_modulus = float(lambda_s)
        else:
            total_pressure_ref = float(2.0 * float(mu_s) + 2.0 * float(lambda_s))
            drained_bulk_modulus = float(lambda_s) + float(mu_s)
        if (not math.isfinite(total_pressure_ref)) or total_pressure_ref <= 0.0:
            raise ValueError("solid_volumetric_split requires a positive total-pressure reference scale.")
        total_pressure_ref_c = _named_c("ff_total_pressure_ref", total_pressure_ref)
        total_pressure_ref_inv_c = _named_c("ff_total_pressure_ref_inv", 1.0 / total_pressure_ref)
        drained_bulk_over_total_pressure_ref_c = _named_c(
            "ff_drained_bulk_over_total_pressure_ref",
            drained_bulk_modulus / total_pressure_ref,
        )
        vol_pen_c = _named_c("ff_solid_volumetric_penalty", float(solid_volumetric_penalty))
        if solid_model_key in {"linear", "small_strain", "linear_elastic"}:
            sigma_s_dev_k = _c(2.0) * mu_s_c * _deviatoric_tensor(_eps(u_k), dim=int(dim))
            dsigma_s_dev = _c(2.0) * mu_s_c * _deviatoric_tensor(_eps(du), dim=int(dim))
            vol_drive_k = pi_s_k - drained_bulk_over_total_pressure_ref_c * div(u_k)
            dvol_drive = dpi_s - drained_bulk_over_total_pressure_ref_c * div(du)
        else:
            mean_dr_k = _tensor_trace(sigma_s_full_k, dim=int(dim)) / _c(float(dim))
            dmean_dr = _tensor_trace(dsigma_s_full, dim=int(dim)) / _c(float(dim))
            sigma_s_dev_k = sigma_s_full_k - mean_dr_k * Identity(dim)
            dsigma_s_dev = dsigma_s_full - dmean_dr * Identity(dim)
            vol_drive_k = pi_s_k - total_pressure_ref_inv_c * mean_dr_k
            dvol_drive = dpi_s - total_pressure_ref_inv_c * dmean_dr
        sigma_s_k = sigma_s_dev_k + total_pressure_ref_c * pi_s_k * Identity(dim)
        dsigma_s = dsigma_s_dev + total_pressure_ref_c * dpi_s * Identity(dim)
    else:
        sigma_s_k = sigma_s_full_k
        dsigma_s = dsigma_s_full

    fluid_conv_key = str(fluid_convection or "full").strip().lower().replace("-", "_")
    pore_conv_key = str(pore_convection or "full").strip().lower().replace("-", "_")
    solid_conv_key = str(skeleton_inertia_convection or "full").strip().lower().replace("-", "_")
    v_ext_mode = str(v_extension_mode or "h1").strip().lower()
    if v_ext_mode not in {"l2", "mass", "grad", "h1"}:
        raise ValueError(f"Unknown v_extension_mode={v_extension_mode!r}.")
    p_ext_mode = str(p_extension_mode or "h1").strip().lower()
    if p_ext_mode not in {"l2", "mass", "grad", "h1"}:
        raise ValueError(f"Unknown p_extension_mode={p_extension_mode!r}.")
    vP_ext_mode = str(vP_extension_mode or "h1").strip().lower()
    if vP_ext_mode not in {"l2", "mass", "grad", "h1"}:
        raise ValueError(f"Unknown vP_extension_mode={vP_extension_mode!r}.")
    p_pore_ext_mode = str(p_pore_extension_mode or "h1").strip().lower()
    if p_pore_ext_mode not in {"l2", "mass", "grad", "h1"}:
        raise ValueError(f"Unknown p_pore_extension_mode={p_pore_extension_mode!r}.")
    rho_s_ext_mode = str(rho_s_extension_mode or "h1").strip().lower()
    if rho_s_ext_mode not in {"l2", "mass", "grad", "h1"}:
        raise ValueError(f"Unknown rho_s_extension_mode={rho_s_extension_mode!r}.")
    u_ext_mode = str(u_extension_mode or "h1").strip().lower()
    if u_ext_mode not in {"l2", "mass", "grad", "h1"}:
        raise ValueError(f"Unknown u_extension_mode={u_extension_mode!r}.")
    gamma_vS_input = float(gamma_u) if gamma_vS is None else float(gamma_vS)
    gamma_vS_c = _named_c("ff_gamma_vS", gamma_vS_input)
    vS_ext_mode = str(u_extension_mode if vS_extension_mode is None else vS_extension_mode).strip().lower()
    if vS_ext_mode not in {"l2", "mass", "grad", "h1"}:
        raise ValueError(f"Unknown vS_extension_mode={vS_extension_mode!r}.")
    if gamma_vS_pin is None and vS_ext_mode in {"grad", "h1"} and float(gamma_u_pin) != 0.0:
        gamma_vS_pin_input = float(gamma_u_pin)
    else:
        gamma_vS_pin_input = 0.0 if gamma_vS_pin is None else float(gamma_vS_pin)
    gamma_vS_pin_c = _named_c("ff_gamma_vS_pin", gamma_vS_pin_input)
    vS_cip_c = _named_c("ff_vS_cip", float(vS_cip))
    grad_alpha_k = grad(alpha_k)
    dgrad_alpha = grad(dalpha)
    grad_alpha_sq = dot(grad_alpha_k, grad_alpha_k)
    grad_alpha_eps = _named_c("ff_grad_alpha_eps", 1.0e-12)
    grad_alpha_mag = (grad_alpha_sq + grad_alpha_eps * grad_alpha_eps) ** _c(0.5)
    inv_grad_alpha_mag = _c(1.0) / grad_alpha_mag
    rot90 = _named_c("ff_rot90", ((0.0, -1.0), (1.0, 0.0)), dim=2)
    dgrad_alpha_mag = dot(grad_alpha_k, dgrad_alpha) * inv_grad_alpha_mag
    normal_vec = grad_alpha_k * inv_grad_alpha_mag
    dnormal_vec = (dgrad_alpha - normal_vec * dgrad_alpha_mag) * inv_grad_alpha_mag
    tangent_vec = dot(rot90, normal_vec)
    dtangent_vec = dot(rot90, dnormal_vec)
    n0 = normal_vec[0]
    n1 = normal_vec[1]
    dn0 = dnormal_vec[0]
    dn1 = dnormal_vec[1]
    t0 = tangent_vec[0]
    t1 = tangent_vec[1]
    dt0 = dtangent_vec[0]
    dt1 = dtangent_vec[1]

    def _stress_times_dir_components(stress, dir0, dir1):
        return (
            stress[0, 0] * dir0 + stress[0, 1] * dir1,
            stress[1, 0] * dir0 + stress[1, 1] * dir1,
        )

    def _normal_trace_from_components(comp_x, comp_y, normal0, normal1):
        return normal0 * comp_x + normal1 * comp_y

    def _tangential_trace_from_components(comp_x, comp_y, tang0, tang1):
        return tang0 * comp_x + tang1 * comp_y

    zero_v = _named_c("ff_zero_v", (0.0, 0.0))
    zero_u = _named_c("ff_zero_u", (0.0, 0.0))
    qs_pore_stress_sign_c = _named_c(
        "ff_qs_pore_stress_sign",
        (1.0 if quasi_static_flip_pore_stress_sign else -1.0),
    )

    # Bulk free-fluid momentum: chi_f(alpha) rho_f D^f(v_f) = chi_f(alpha) div(sigma_f),
    # with chi_f = 1 - chi(alpha) and the explicit transfer laws kept on raw grad(alpha).
    r_mom_f = F_bulk_k * rho_f_c * dot((v_k - v_n) * inv_dt, v_test) * dx
    a_mom_f = (
        dF_bulk * rho_f_c * dot((v_k - v_n) * inv_dt, v_test)
        + F_bulk_k * rho_f_c * dot(dv * inv_dt, v_test)
    ) * dx
    if fluid_conv_key != "off":
        conv_f_k = dot(grad(v_k), v_k)
        dconv_f = dot(grad(dv), v_k) + dot(grad(v_k), dv)
        r_mom_f += F_bulk_k * rho_f_c * dot(conv_f_k, v_test) * dx
        a_mom_f += (
            dF_bulk * rho_f_c * dot(conv_f_k, v_test)
            + F_bulk_k * rho_f_c * dot(dconv_f, v_test)
        ) * dx
    r_mom_f += F_bulk_k * _c(2.0) * mu_f_c * inner(_eps(v_k), _eps(v_test)) * dx
    a_mom_f += (
        dF_bulk * _c(2.0) * mu_f_c * inner(_eps(v_k), _eps(v_test))
        + F_bulk_k * _c(2.0) * mu_f_c * inner(_eps(dv), _eps(v_test))
    ) * dx
    r_mom_f += -(F_bulk_k * p_k) * div(v_test) * dx
    a_mom_f += (-dF_bulk * p_k - F_bulk_k * dp) * div(v_test) * dx

    r_mom_f_bulk = r_mom_f
    a_mom_f_bulk = a_mom_f

    rel_p_k = vP_k - vS_k
    drel_p = dvP - dvS
    drag_coeff_k = alpha_bulk_k * phi_k * phi_k * kappa_inv_c
    ddrag_coeff = (dalpha_bulk * phi_k * phi_k + alpha_bulk_k * _c(2.0) * phi_k * dphi) * kappa_inv_c

    if quasi_static_porous_media:
        sigma_p_qs_k = qs_pore_stress_sign_c * p_pore_k * Identity(dim)
        dsigma_p_qs = qs_pore_stress_sign_c * dp_pore * Identity(dim)
        r_mom_p = zero * dot(vP_k, vP_test) * dx
        a_mom_p = zero * dot(vP_k, vP_test) * dx
        if not quasi_static_disable_pore_momentum:
            # Quasi-static pore-fluid momentum with sigma_p = s p_pore I,
            # reduced to the pressure-plus-drag form requested for this debug hook:
            #   -s grad(p_pore) + phi/K (v_p-v_s) = 0
            # and test function w_p = vP_test.
            r_mom_p = alpha_bulk_k * (
                (qs_pore_stress_sign_c * p_pore_k * div(vP_test))
                + (phi_k * kappa_inv_c) * dot(rel_p_k, vP_test)
            ) * dx
            a_mom_p = (
                dalpha_bulk
                * (
                    (qs_pore_stress_sign_c * p_pore_k * div(vP_test))
                    + (phi_k * kappa_inv_c) * dot(rel_p_k, vP_test)
                )
                + alpha_bulk_k
                * (
                    (qs_pore_stress_sign_c * dp_pore * div(vP_test))
                    + (dphi * kappa_inv_c) * dot(rel_p_k, vP_test)
                    + (phi_k * kappa_inv_c) * dot(drel_p, vP_test)
                )
            ) * dx
    else:
        # Bulk pore momentum:
        #   alpha*phi*rho_f*D^p(v_p)
        #   = alpha*div(phi*sigma_p) + alpha*p_p*grad(phi) - alpha*phi^2/K*(v_p-v_s)
        # with
        #   sigma_p = 2*mu_p(phi)*eps(v_p) - p_p*I.
        #
        # Expanding the product term gives
        #   div(phi*sigma_p) + p_p*grad(phi)
        #   = phi*div(sigma_p) + sigma_p*grad(phi) + p_p*grad(phi)
        #   = phi*div(sigma_p) + 2*mu_p(phi)*eps(v_p)*grad(phi),
        # so the extra viscous grad(phi) contribution is already contained in the
        # assembled product-form weak row below and must not be dropped when the
        # strong form is written out.
        pore_coeff_k = alpha_bulk_k * phi_k
        dpore_coeff = dalpha_bulk * phi_k + alpha_bulk_k * dphi
        r_mom_p = pore_coeff_k * rho_f_c * dot((vP_k - vP_n) * inv_dt, vP_test) * dx
        a_mom_p = (
            dpore_coeff * rho_f_c * dot((vP_k - vP_n) * inv_dt, vP_test)
            + pore_coeff_k * rho_f_c * dot(dvP * inv_dt, vP_test)
        ) * dx
        if pore_conv_key != "off":
            conv_p_k = dot(grad(vP_k), vP_k)
            dconv_p = dot(grad(dvP), vP_k) + dot(grad(vP_k), dvP)
            r_mom_p += pore_coeff_k * rho_f_c * dot(conv_p_k, vP_test) * dx
            a_mom_p += (
                dpore_coeff * rho_f_c * dot(conv_p_k, vP_test)
                + pore_coeff_k * rho_f_c * dot(dconv_p, vP_test)
            ) * dx
        r_mom_p += alpha_bulk_k * phi_k * inner(sigma_p_k, grad(vP_test)) * dx
        a_mom_p += (
            (dalpha_bulk * phi_k + alpha_bulk_k * dphi) * inner(sigma_p_k, grad(vP_test))
            + alpha_bulk_k * phi_k * inner(dsigma_p, grad(vP_test))
        ) * dx
        r_mom_p += -(alpha_bulk_k * p_pore_k) * dot(grad(phi_k), vP_test) * dx
        a_mom_p += -(
            (dalpha_bulk * p_pore_k + alpha_bulk_k * dp_pore) * dot(grad(phi_k), vP_test)
            + alpha_bulk_k * p_pore_k * dot(grad(dphi), vP_test)
        ) * dx
        # Keep the undivided whole-domain form so the pore row deactivates smoothly
        # when alpha->0 or phi->0. Dividing out phi would hide that weighting and is
        # not robust at vanishing pore fraction.
        r_mom_p += drag_coeff_k * dot(rel_p_k, vP_test) * dx
        a_mom_p += (ddrag_coeff * dot(rel_p_k, vP_test) + drag_coeff_k * dot(drel_p, vP_test)) * dx

    r_mom_p_bulk = r_mom_p
    a_mom_p_bulk = a_mom_p

    if rigid_darcy_head_mode:
        if normal_carrier_key != "multiplier":
            raise ValueError(
                "rigid_darcy_head_mode requires normal_constraint_carrier='multiplier' "
                "so the interface laws remain on their dedicated residual rows."
            )

        def _rigid_traction_directional(*, dv_var=None, dp_var=None, dp_pore_var=None):
            dv_loc = zero_v if dv_var is None else dv_var
            dp_loc = zero if dp_var is None else dp_var
            dp_pore_loc = zero if dp_pore_var is None else dp_pore_var

            dsigma_f_loc = _c(2.0) * mu_f_c * _eps(dv_loc) - dp_loc * Identity(dim)
            if interface_formulation_key == "decomposed":
                dfluid_n_x_loc = dsigma_f_loc[0, 0] * n0 + dsigma_f_loc[0, 1] * n1
                dfluid_n_y_loc = dsigma_f_loc[1, 0] * n0 + dsigma_f_loc[1, 1] * n1
                dfluid_normal_trac_loc = n0 * dfluid_n_x_loc + n1 * dfluid_n_y_loc
                dfluid_tang_trac_loc = t0 * dfluid_n_x_loc + t1 * dfluid_n_y_loc
            else:
                dfluid_n_x_loc, dfluid_n_y_loc = _stress_times_dir_components(dsigma_f_loc, n0, n1)
                dfluid_normal_trac_loc = _normal_trace_from_components(
                    dfluid_n_x_loc, dfluid_n_y_loc, n0, n1
                )
                dfluid_tang_trac_loc = _tangential_trace_from_components(
                    dfluid_n_x_loc, dfluid_n_y_loc, t0, t1
                )

            dnormal_jump_loc = dfluid_normal_trac_loc + normal_pressure_scale_c * phi_k * dp_pore_loc
            dtangential_jump_loc = dfluid_tang_trac_loc
            if float(bjs_coefficient) != 0.0 and dv_var is not None:
                if interface_formulation_key == "decomposed":
                    fluid_tangential_velocity_loc = t0 * dv_loc[0] + t1 * dv_loc[1]
                else:
                    fluid_tangential_velocity_loc = dot(tangent_vec, dv_loc)
                dtangential_jump_loc = dtangential_jump_loc + bjs_coeff_c * fluid_tangential_velocity_loc
            return grad_alpha_mag * dnormal_jump_loc, grad_alpha_mag * dtangential_jump_loc

        if interface_formulation_key == "decomposed":
            fluid_n_x = sigma_f_k[0, 0] * n0 + sigma_f_k[0, 1] * n1
            fluid_n_y = sigma_f_k[1, 0] * n0 + sigma_f_k[1, 1] * n1
            fluid_normal_trac = n0 * fluid_n_x + n1 * fluid_n_y
            fluid_tang_trac = t0 * fluid_n_x + t1 * fluid_n_y
            fluid_tangential_velocity_k = t0 * v_k[0] + t1 * v_k[1]
        else:
            fluid_n_x, fluid_n_y = _stress_times_dir_components(sigma_f_k, n0, n1)
            fluid_normal_trac = _normal_trace_from_components(fluid_n_x, fluid_n_y, n0, n1)
            fluid_tang_trac = _tangential_trace_from_components(fluid_n_x, fluid_n_y, t0, t1)
            fluid_tangential_velocity_k = dot(tangent_vec, v_k)

        # Fluid mass block
        r_mass = q_test * (F_bulk_k * div(v_k)) * dx
        a_mass = q_test * (dF_bulk * div(v_k) + F_bulk_k * div(dv)) * dx

        # Porous mass block
        pore_bulk_flux_k = alpha_bulk_k * (phi_k * div(vP_k) + dot(grad(phi_k), vP_k))
        dpore_bulk_flux = (
            dalpha_bulk * (phi_k * div(vP_k) + dot(grad(phi_k), vP_k))
            + alpha_bulk_k
            * (
                dphi * div(vP_k)
                + phi_k * div(dvP)
                + dot(grad(dphi), vP_k)
                + dot(grad(phi_k), dvP)
            )
        )
        r_pore = q_pore_test * pore_bulk_flux_k * dx
        a_pore = q_pore_test * dpore_bulk_flux * dx

        mass_jump_rigid_k = rho_f_c * v_k - rho_s_phys_k * phi_k * vP_k
        dmass_jump_rigid = (
            rho_f_c * dv
            - ((rho_s_phys_k * dphi) * vP_k + rho_s_phys_k * phi_k * dvP)
        )
        mass_constraint_rigid_k = dot(grad_alpha_k, mass_jump_rigid_k)
        dmass_constraint_rigid = dot(grad_alpha_k, dmass_jump_rigid)

        normal_jump_rigid_k = fluid_normal_trac + normal_pressure_scale_c * phi_k * p_pore_k
        dnormal_jump_rigid = _rigid_traction_directional(
            dv_var=dv, dp_var=dp, dp_pore_var=dp_pore
        )[0]
        tangential_jump_rigid_k = fluid_tang_trac
        dtangential_jump_rigid = _rigid_traction_directional(dv_var=dv)[1]
        if float(bjs_coefficient) != 0.0:
            tangential_jump_rigid_k = tangential_jump_rigid_k + bjs_coeff_c * fluid_tangential_velocity_k

        normal_constraint_rigid_k = grad_alpha_mag * normal_jump_rigid_k
        tangential_constraint_rigid_k = grad_alpha_mag * tangential_jump_rigid_k
        dnormal_constraint_rigid = dnormal_jump_rigid
        dtangential_constraint_rigid = dtangential_jump_rigid

        r_interface_mass = mu_mass_test * mass_constraint_rigid_k * dx
        a_interface_mass = mu_mass_test * dmass_constraint_rigid * dx
        r_interface_normal = mu_normal_test * normal_constraint_rigid_k * dx
        a_interface_normal = mu_normal_test * dnormal_constraint_rigid * dx
        r_interface_tangential = mu_tangent_test * tangential_constraint_rigid_k * dx
        a_interface_tangential = mu_tangent_test * dtangential_constraint_rigid * dx

        mass_test_bracket_v = rho_f_c * dot(grad_alpha_k, v_test)
        mass_test_bracket_vP = -(rho_s_phys_k * phi_k) * dot(grad_alpha_k, vP_test)
        normal_test_bracket_v, tangential_test_bracket_v = _rigid_traction_directional(dv_var=v_test)
        normal_test_bracket_p, _ = _rigid_traction_directional(dp_var=q_test)
        normal_test_bracket_p_pore, _ = _rigid_traction_directional(dp_pore_var=q_pore_test)

        r_mom_f = r_mom_f + mu_mass_k * mass_test_bracket_v * dx
        a_mom_f = a_mom_f + dmu_mass * mass_test_bracket_v * dx
        r_mom_f = r_mom_f + mu_normal_k * normal_test_bracket_v * dx
        a_mom_f = a_mom_f + dmu_normal * normal_test_bracket_v * dx
        r_mom_f = r_mom_f + mu_tangent_k * tangential_test_bracket_v * dx
        a_mom_f = a_mom_f + dmu_tangent * tangential_test_bracket_v * dx

        r_mom_p = r_mom_p + mu_mass_k * mass_test_bracket_vP * dx
        a_mom_p = a_mom_p + dmu_mass * mass_test_bracket_vP * dx
        r_mass = r_mass + mu_normal_k * normal_test_bracket_p * dx
        a_mass = a_mass + dmu_normal * normal_test_bracket_p * dx
        r_pore = r_pore + mu_normal_k * normal_test_bracket_p_pore * dx
        a_pore = a_pore + dmu_normal * normal_test_bracket_p_pore * dx

        h_ext_rigid = MeshSize()
        inv_h2_rigid = _c(1.0) / (h_ext_rigid * h_ext_rigid)

        if float(gamma_v) != 0.0:
            if v_ext_mode in {"l2", "mass"}:
                r_mom_f = r_mom_f + gamma_v_c * inv_h2_rigid * alpha_support_k * dot(v_k, v_test) * dx
                a_mom_f = a_mom_f + gamma_v_c * inv_h2_rigid * (
                    dalpha_support * dot(v_k, v_test) + alpha_support_k * dot(dv, v_test)
                ) * dx
            else:
                r_mom_f = r_mom_f + gamma_v_c * alpha_support_k * inner(grad(v_k), grad(v_test)) * dx
                a_mom_f = a_mom_f + gamma_v_c * (
                    dalpha_support * inner(grad(v_k), grad(v_test)) + alpha_support_k * inner(grad(dv), grad(v_test))
                ) * dx
                if float(gamma_v_pin) != 0.0:
                    w_pin2 = alpha_support_k * alpha_support_k
                    dw_pin2 = _c(2.0) * alpha_support_k * dalpha_support
                    r_mom_f = r_mom_f + gamma_v_pin_c * inv_h2_rigid * w_pin2 * dot(v_k, v_test) * dx
                    a_mom_f = a_mom_f + gamma_v_pin_c * inv_h2_rigid * (
                        dw_pin2 * dot(v_k, v_test) + w_pin2 * dot(dv, v_test)
                    ) * dx

        if float(gamma_p) != 0.0:
            if p_ext_mode in {"l2", "mass"}:
                r_mass = r_mass + gamma_p_c * inv_h2_rigid * alpha_support_k * p_k * q_test * dx
                a_mass = a_mass + gamma_p_c * inv_h2_rigid * (
                    dalpha_support * p_k + alpha_support_k * dp
                ) * q_test * dx
            else:
                r_mass = r_mass + gamma_p_c * alpha_support_k * dot(grad(p_k), grad(q_test)) * dx
                a_mass = a_mass + gamma_p_c * (
                    dalpha_support * dot(grad(p_k), grad(q_test)) + alpha_support_k * dot(grad(dp), grad(q_test))
                ) * dx
                if float(gamma_p_pin) != 0.0:
                    w_pin2 = alpha_support_k * alpha_support_k
                    dw_pin2 = _c(2.0) * alpha_support_k * dalpha_support
                    r_mass = r_mass + gamma_p_pin_c * inv_h2_rigid * w_pin2 * p_k * q_test * dx
                    a_mass = a_mass + gamma_p_pin_c * inv_h2_rigid * (
                        dw_pin2 * p_k + w_pin2 * dp
                    ) * q_test * dx

        if float(gamma_vP) != 0.0:
            if vP_ext_mode in {"l2", "mass"}:
                r_mom_p = r_mom_p + gamma_vP_c * inv_h2_rigid * F_support_k * dot(vP_k, vP_test) * dx
                a_mom_p = a_mom_p + gamma_vP_c * inv_h2_rigid * (
                    dF_support * dot(vP_k, vP_test) + F_support_k * dot(dvP, vP_test)
                ) * dx
            else:
                r_mom_p = r_mom_p + gamma_vP_c * F_support_k * inner(grad(vP_k), grad(vP_test)) * dx
                a_mom_p = a_mom_p + gamma_vP_c * (
                    dF_support * inner(grad(vP_k), grad(vP_test)) + F_support_k * inner(grad(dvP), grad(vP_test))
                ) * dx
                if float(gamma_vP_pin) != 0.0:
                    w_pin2 = F_support_k * F_support_k
                    dw_pin2 = _c(2.0) * F_support_k * dF_support
                    r_mom_p = r_mom_p + gamma_vP_pin_c * inv_h2_rigid * w_pin2 * dot(vP_k, vP_test) * dx
                    a_mom_p = a_mom_p + gamma_vP_pin_c * inv_h2_rigid * (
                        dw_pin2 * dot(vP_k, vP_test) + w_pin2 * dot(dvP, vP_test)
                    ) * dx

        if float(gamma_p_pore) != 0.0:
            if p_pore_ext_mode in {"l2", "mass"}:
                r_pore = r_pore + gamma_p_pore_c * inv_h2_rigid * F_support_k * p_pore_k * q_pore_test * dx
                a_pore = a_pore + gamma_p_pore_c * inv_h2_rigid * (
                    dF_support * p_pore_k + F_support_k * dp_pore
                ) * q_pore_test * dx
            else:
                r_pore = r_pore + gamma_p_pore_c * F_support_k * dot(grad(p_pore_k), grad(q_pore_test)) * dx
                a_pore = a_pore + gamma_p_pore_c * (
                    dF_support * dot(grad(p_pore_k), grad(q_pore_test))
                    + F_support_k * dot(grad(dp_pore), grad(q_pore_test))
                ) * dx
                if float(gamma_p_pore_pin) != 0.0:
                    w_pin2 = F_support_k * F_support_k
                    dw_pin2 = _c(2.0) * F_support_k * dF_support
                    r_pore = r_pore + gamma_p_pore_pin_c * inv_h2_rigid * w_pin2 * p_pore_k * q_pore_test * dx
                    a_pore = a_pore + gamma_p_pore_pin_c * inv_h2_rigid * (
                        dw_pin2 * p_pore_k + w_pin2 * dp_pore
                    ) * q_pore_test * dx

        zero_scalar_test = q_test if q_test is not None else q_pore_test
        zero_phi_test = phi_test if phi_test is not None else zero_scalar_test
        zero_rho_s_test = rho_s_test if rho_s_test is not None else zero_scalar_test
        zero_vec_form = zero * dot(vS_k, vS_test) * dx
        zero_u_form = zero * dot(u_k, u_test) * dx
        zero_phi_form = _safe_zero_scalar_linear_form(phi_k, zero_phi_test)
        zero_rho_s_form = _safe_zero_scalar_linear_form(rho_s_phys_k, zero_rho_s_test)
        zero_alpha_form = _safe_zero_scalar_linear_form(alpha_k, alpha_test)
        zero_q_anchor = p_k if q_test is not None else p_pore_k
        zero_q_form = _safe_zero_scalar_linear_form(zero_q_anchor, zero_scalar_test)

        r_momentum = r_mom_f + r_mom_p
        a_momentum = a_mom_f + a_mom_p
        residual_form = (
            r_momentum
            + r_mass
            + r_pore
            + r_interface_mass
            + r_interface_normal
            + r_interface_tangential
        )
        jacobian_form = (
            a_momentum
            + a_mass
            + a_pore
            + a_interface_mass
            + a_interface_normal
            + a_interface_tangential
        )

        return BiofilmOneDomainForms(
            jacobian_form=jacobian_form,
            residual_form=residual_form,
            r_momentum=r_momentum,
            r_mass=r_mass,
            r_kinematics=zero_u_form,
            r_skeleton=zero_vec_form,
            r_phi=zero_phi_form,
            r_phi_transport=zero_phi_form,
            r_alpha=zero_alpha_form,
            r_alpha_transport=zero_alpha_form,
            r_mu_alpha=zero_alpha_form,
            r_damage=None,
            r_substrate=zero_alpha_form,
            r_B=None,
            r_pore=r_pore,
            r_total_mass=r_mass + r_pore + r_interface_mass,
            r_momentum_terms={
                "free_bulk": r_mom_f_bulk,
                "pore_bulk": r_mom_p_bulk,
                "solid_bulk": zero_vec_form,
                "free_extension_v": r_mom_f - r_mom_f_bulk,
                "pore_extension_vP": r_mom_p - r_mom_p_bulk,
                "interface_mass_constraint": r_interface_mass,
                "interface_normal_constraint": r_interface_normal,
                "interface_tangential_constraint": r_interface_tangential,
                "interface_mass_bulk_coupling": mu_mass_k * (mass_test_bracket_v + mass_test_bracket_vP) * dx,
                "interface_normal_bulk_coupling": mu_normal_k * (
                    normal_test_bracket_v + normal_test_bracket_p + normal_test_bracket_p_pore
                ) * dx,
                "interface_normal_aug_bulk": zero_q_form,
                "interface_tangential_bulk_coupling": mu_tangent_k * tangential_test_bracket_v * dx,
            },
            r_kinematics_terms=None,
            r_skeleton_terms={
                "solid_bulk": zero_vec_form,
                "pore_pressure_grad_phi": zero_vec_form,
                "drag": zero_vec_form,
            },
            a_momentum=a_momentum,
            a_momentum_terms={
                "interface_mass_constraint": a_interface_mass,
                "interface_normal_constraint": a_interface_normal,
                "interface_tangential_constraint": a_interface_tangential,
                "interface_mass_bulk_coupling": dmu_mass * (mass_test_bracket_v + mass_test_bracket_vP) * dx,
                "interface_normal_bulk_coupling": dmu_normal * (
                    normal_test_bracket_v + normal_test_bracket_p + normal_test_bracket_p_pore
                ) * dx,
                "interface_normal_aug_bulk": zero_q_form,
                "interface_tangential_bulk_coupling": dmu_tangent * tangential_test_bracket_v * dx,
            },
            a_mass=a_mass,
            a_pore=a_pore,
            a_total_mass=a_mass + a_pore + a_interface_mass,
            a_kinematics=zero_u_form,
            a_skeleton=zero_vec_form,
            a_skeleton_terms={
                "solid_bulk": zero_vec_form,
                "pore_pressure_grad_phi": zero_vec_form,
                "drag": zero_vec_form,
            },
            a_phi=zero_rho_s_form,
            a_phi_transport=zero_rho_s_form,
            a_B=None,
            a_alpha=zero_alpha_form,
            a_alpha_transport=zero_alpha_form,
            a_mu_alpha=None,
            a_damage=None,
            a_substrate=None,
            r_detached=None,
            a_detached=None,
            r_alpha_lambda=None,
            a_alpha_lambda=None,
            r_drag_lambda=None,
            a_drag_lambda=None,
            r_skeleton_pressure=None,
            a_skeleton_pressure=None,
            r_volumetric=None,
            a_volumetric=None,
        )

    # Bulk solid momentum uses the support indicator chi(alpha) for occupancy,
    # while the explicit transfer terms below remain on raw grad(alpha).
    r_interface_direct_traction_free = zero * dot(v_k, v_test) * dx
    a_interface_direct_traction_free = zero * dot(v_k, v_test) * dx
    r_interface_direct_tangential_drag_free = zero * dot(v_k, v_test) * dx
    a_interface_direct_tangential_drag_free = zero * dot(v_k, v_test) * dx
    r_interface_direct_tangential_drag = zero * dot(vS_k, vS_test) * dx
    a_interface_direct_tangential_drag = zero * dot(vS_k, vS_test) * dx
    if quasi_static_porous_media:
        sigma_p_qs_k = -(p_pore_k * Identity(dim))
        dsigma_p_qs = -(dp_pore * Identity(dim))
        r_skel = zero * dot(vS_k, vS_test) * dx
        a_skel = zero * dot(vS_k, vS_test) * dx
        if not quasi_static_disable_solid_momentum:
            # Quasi-static porous-solid momentum:
            #   div((1-phi) sigma_s) - alpha_biot p_pore div(alpha test)
            #   + phi^2/K (v_p-v_s) = 0
            # tested with w_u = vS_test.
            r_skel = (
                -(alpha_bulk_k * one_m_phi_k) * inner(sigma_s_k, _eps(vS_test))
                - alpha_biot_c * p_pore_k * div_alpha_bulk_vStest_k
                - drag_coeff_k * dot(rel_p_k, vS_test)
            ) * dx
            a_skel = -(
                (dalpha_bulk * one_m_phi_k - alpha_bulk_k * dphi) * inner(sigma_s_k, _eps(vS_test))
                + alpha_bulk_k * one_m_phi_k * inner(dsigma_s, _eps(vS_test))
            ) * dx + (
                - alpha_biot_c * (dp_pore * div_alpha_bulk_vStest_k + p_pore_k * ddiv_alpha_bulk_vStest)
                - (ddrag_coeff * dot(rel_p_k, vS_test) + drag_coeff_k * dot(drel_p, vS_test))
            ) * dx
        r_skel_bulk = r_skel
        a_skel_bulk = a_skel
        r_combined_porous_bulk = zero * dot(vS_k, vS_test) * dx
        a_combined_porous_bulk = zero * dot(vS_k, vS_test) * dx
        if quasi_static_use_combined_porous_momentum:
            porous_sigma_qs_k = phi_k * sigma_p_qs_k + one_m_phi_k * sigma_s_k
            dporous_sigma_qs = dphi * sigma_p_qs_k + phi_k * dsigma_p_qs - dphi * sigma_s_k + one_m_phi_k * dsigma_s
            # Debug hook requested by the user: a single combined porous-body
            # momentum row on the solid test space.
            r_combined_porous_bulk = alpha_bulk_k * inner(porous_sigma_qs_k, _eps(vS_test)) * dx
            a_combined_porous_bulk = (
                dalpha_bulk * inner(porous_sigma_qs_k, _eps(vS_test))
                + alpha_bulk_k * inner(dporous_sigma_qs, _eps(vS_test))
            ) * dx
            r_skel = r_skel + r_combined_porous_bulk
            a_skel = a_skel + a_combined_porous_bulk
        r_interface_direct_traction = zero * dot(vS_k, vS_test) * dx
        a_interface_direct_traction = zero * dot(vS_k, vS_test) * dx
        if direct_interface_transfer:
            # Keep the direct porous-side traction jump consistent with the
            # combined porous-body row: sigma_f - phi*sigma_p - (1-phi)*sigma_s.
            porous_interface_sigma_k = (
                phi_k * sigma_p_qs_k
                + solid_interface_traction_weight_c * one_m_phi_k * sigma_s_k
            )
            dporous_interface_sigma = (
                dphi * sigma_p_qs_k
                + phi_k * dsigma_p_qs
                + solid_interface_traction_weight_c
                * (-dphi * sigma_s_k + one_m_phi_k * dsigma_s)
            )
            porous_traction_transfer_k = dot(porous_interface_sigma_k, grad_alpha_k)
            dporous_traction_transfer = (
                dot(dporous_interface_sigma, grad_alpha_k)
                + dot(porous_interface_sigma_k, grad(dalpha))
            )
            fluid_traction_transfer_k = dot(sigma_f_k, grad_alpha_k)
            dfluid_traction_transfer = (
                dot(dsigma_f, grad_alpha_k)
                + dot(sigma_f_k, grad(dalpha))
            )
            # On the quasi-static debug hook the multiplier rows are disabled and
            # the full diffuse traction jump is injected on the porous-side test
            # space. Splitting the fluid and porous pieces onto separate self
            # rows leaves the porous residual identically zero at the trivial
            # porous state, which uncouples the mixed problem.
            traction_jump_transfer_k = fluid_traction_transfer_k - porous_traction_transfer_k
            dtraction_jump_transfer = dfluid_traction_transfer - dporous_traction_transfer
            r_interface_direct_traction = dot(traction_jump_transfer_k, vS_test) * dx
            a_interface_direct_traction = dot(dtraction_jump_transfer, vS_test) * dx
            r_skel = r_skel + r_interface_direct_traction
            a_skel = a_skel + a_interface_direct_traction
            if quasi_static_use_combined_porous_momentum and use_bjs_tangential_law:
                porous_mix_velocity_k = phi_k * vP_k + one_m_phi_k * vS_k
                dporous_mix_velocity = dphi * rel_p_k + phi_k * dvP + one_m_phi_k * dvS
                bjs_beta_interface_c = bjs_coeff_c * mu_f_c * sqrt_kappa_inv_c
                tangential_slip_k = dot(tangent_vec, v_k - porous_mix_velocity_k)
                dtangential_slip = (
                    dot(dtangent_vec, v_k - porous_mix_velocity_k)
                    + dot(tangent_vec, dv - dporous_mix_velocity)
                )
                tangential_drag_vec_k = (
                    grad_alpha_mag * bjs_beta_interface_c * tangential_slip_k * tangent_vec
                )
                dtangential_drag_vec = bjs_beta_interface_c * (
                    dgrad_alpha_mag * tangential_slip_k * tangent_vec
                    + grad_alpha_mag * dtangential_slip * tangent_vec
                    + grad_alpha_mag * tangential_slip_k * dtangent_vec
                )
                r_interface_direct_tangential_drag_free = -dot(
                    tangential_drag_vec_k, v_test
                ) * dx
                a_interface_direct_tangential_drag_free = -dot(
                    dtangential_drag_vec, v_test
                ) * dx
                r_mom_f = r_mom_f + r_interface_direct_tangential_drag_free
                a_mom_f = a_mom_f + a_interface_direct_tangential_drag_free
                r_interface_direct_tangential_drag = dot(
                    tangential_drag_vec_k, vS_test
                ) * dx
                a_interface_direct_tangential_drag = dot(
                    dtangential_drag_vec, vS_test
                ) * dx
                r_skel = r_skel + r_interface_direct_tangential_drag
                a_skel = a_skel + a_interface_direct_tangential_drag
    else:
        solid_coeff_k = alpha_bulk_k * rho_s_phys_k * one_m_phi_k
        dsolid_coeff = (
            dalpha_bulk * rho_s_phys_k * one_m_phi_k
            + alpha_bulk_k * drho_s_phys * one_m_phi_k
            - alpha_bulk_k * rho_s_phys_k * dphi
        )
        r_skel = solid_coeff_k * dot((vS_k - vS_n) * inv_dt, vS_test) * dx
        a_skel = (
            dsolid_coeff * dot((vS_k - vS_n) * inv_dt, vS_test)
            + solid_coeff_k * dot(dvS * inv_dt, vS_test)
        ) * dx
        if solid_conv_key != "off":
            conv_s_k = dot(grad(vS_k), vS_k)
            dconv_s = dot(grad(dvS), vS_k) + dot(grad(vS_k), dvS)
            r_skel += solid_coeff_k * dot(conv_s_k, vS_test) * dx
            a_skel += dsolid_coeff * dot(conv_s_k, vS_test) * dx + solid_coeff_k * dot(dconv_s, vS_test) * dx
        r_skel += -(alpha_bulk_k * one_m_phi_k) * inner(sigma_s_k, grad(vS_test)) * dx
        a_skel += -(
            (dalpha_bulk * one_m_phi_k - alpha_bulk_k * dphi) * inner(sigma_s_k, grad(vS_test))
            + alpha_bulk_k * one_m_phi_k * inner(dsigma_s, grad(vS_test))
        ) * dx
        if float(solid_visco_eta) != 0.0:
            r_skel += alpha_bulk_k * one_m_phi_k * _c(2.0) * eta_s_c * inner(_eps(vS_k), _eps(vS_test)) * dx
            a_skel += (
                (dalpha_bulk * one_m_phi_k - alpha_bulk_k * dphi) * _c(2.0) * eta_s_c * inner(_eps(vS_k), _eps(vS_test))
                + alpha_bulk_k * one_m_phi_k * _c(2.0) * eta_s_c * inner(_eps(dvS), _eps(vS_test))
            ) * dx
        r_skel += -(alpha_biot_c * p_pore_k * div_alpha_bulk_vStest_k) * dx
        a_skel += (
            -(alpha_biot_c * (dp_pore * div_alpha_bulk_vStest_k + p_pore_k * ddiv_alpha_bulk_vStest))
        ) * dx
        r_skel += -drag_coeff_k * dot(rel_p_k, vS_test) * dx
        a_skel += -(ddrag_coeff * dot(rel_p_k, vS_test) + drag_coeff_k * dot(drel_p, vS_test)) * dx
        r_combined_porous_bulk = zero * dot(vS_k, vS_test) * dx
        a_combined_porous_bulk = zero * dot(vS_k, vS_test) * dx
        r_interface_direct_traction = zero * dot(vS_k, vS_test) * dx
        a_interface_direct_traction = zero * dot(vS_k, vS_test) * dx
        r_skel_bulk = r_skel
        a_skel_bulk = a_skel

    
    # Interface mass transfer
    mass_jump_k = rho_f_c * v_k - rho_s_phys_k * phi_k * vP_k - rho_s_phys_k * one_m_phi_k * vS_k
    if constant_rho_s:
        dmass_jump = (
            rho_f_c * dv
            - (( rho_s_phys_k * dphi) * vP_k + rho_s_phys_k * phi_k * dvP)
            - (( - rho_s_phys_k * dphi) * vS_k + rho_s_phys_k * one_m_phi_k * dvS)
        )
    else:
        dmass_jump = (
            rho_f_c * dv
            - ((drho_s_phys * phi_k + rho_s_phys_k * dphi) * vP_k + rho_s_phys_k * phi_k * dvP)
            - ((drho_s_phys * one_m_phi_k - rho_s_phys_k * dphi) * vS_k + rho_s_phys_k * one_m_phi_k * dvS)
        )
    mass_constraint_k = dot(grad(alpha_k), mass_jump_k)
    dmass_constraint = dot(grad(dalpha), mass_jump_k) + dot(grad(alpha_k), dmass_jump)
    direct_interface_mass_free_k = _safe_zero_scalar_linear_form(p_k, q_test)
    ddirect_interface_mass_free = _safe_zero_scalar_linear_form(p_k, q_test)
    direct_interface_mass_porous_k = _safe_zero_scalar_linear_form(p_pore_k, q_pore_test)
    ddirect_interface_mass_porous = _safe_zero_scalar_linear_form(p_pore_k, q_pore_test)
    # Free-fluid mass row: free-region incompressibility only.
    r_mass_bulk = q_test * (F_bulk_k * div(v_k)) * dx
    a_mass_bulk = q_test * (
        dF_bulk * div(v_k)
        + F_bulk_k * div(dv)
    ) * dx
    r_mass = r_mass_bulk
    a_mass = a_mass_bulk
    r_pore = None
    a_pore = None

    # Bulk porous continuity laws.
    #
    # Variable-density branch:
    #   alpha * (d_t phi + div(phi v_p)) = 0.
    #
    # Constant-density branch:
    #   alpha * div(phi v_p + (1-phi) v_s) = 0.
    #
    # The explicit interface mass-transfer law remains on its dedicated
    # multiplier row and is not folded into this bulk equation.
    phi_rate_k = (phi_k - phi_n) * inv_dt
    dphi_rate = dphi * inv_dt
    pore_transport_k = phi_rate_k + phi_k * div(vP_k) + dot(grad(phi_k), vP_k)
    dpore_transport = (
        dphi_rate
        + dphi * div(vP_k)
        + phi_k * div(dvP)
        + dot(grad(dphi), vP_k)
        + dot(grad(phi_k), dvP)
    )
    # mass balance of the porous phase
    if constant_rho_s:
        rho_rate_k = zero
        drho_rate = zero
    else:
        rho_rate_k = (rho_s_phys_k - rho_s_phys_n) * inv_dt + dot(vS_k, grad(rho_s_k))
        drho_rate = drho_s_phys * inv_dt + dot(dvS, grad(rho_s_k)) + dot(vS_k, grad(drho_s))
    transfer_vel_k = vP_k - vS_k
    dtransfer_vel = dvP - dvS
    div_transfer_k = phi_k * div(transfer_vel_k) + dot(grad(phi_k), transfer_vel_k)
    ddiv_transfer = (
        dphi * div(transfer_vel_k)
        + phi_k * div(dtransfer_vel)
        + dot(grad(dphi), transfer_vel_k)
        + dot(grad(phi_k), dtransfer_vel)
    )
    if quasi_static_porous_media:
        porous_mass_flux_k = phi_k * vP_k + one_m_phi_k * vS_k
        dporous_mass_flux = dphi * rel_p_k + phi_k * dvP + one_m_phi_k * dvS
        r_pore = -(alpha_bulk_k * dot(grad(q_pore_test), porous_mass_flux_k)) * dx
        a_pore = -(
            dalpha_bulk * dot(grad(q_pore_test), porous_mass_flux_k)
            + alpha_bulk_k * dot(grad(q_pore_test), dporous_mass_flux)
        ) * dx
        if direct_interface_transfer:
            # The quasi-static porous continuity equation always carries the
            # porous-side diffuse mass-transfer law, regardless of how phi is
            # evolved. alpha_closure replaces only the phi row, not the
            # p_pore/q_pore continuity law.
            direct_interface_mass_jump_k = rho_f_c * v_k - rho_s_phys_k * porous_mass_flux_k
            ddirect_interface_mass_jump = (
                rho_f_c * dv
                - (drho_s_phys * porous_mass_flux_k + rho_s_phys_k * dporous_mass_flux)
            )
            direct_interface_mass_porous_k = -(
                q_pore_test * dot(grad_alpha_k, direct_interface_mass_jump_k)
            ) * dx
            ddirect_interface_mass_porous = -(
                q_pore_test * dot(grad(dalpha), direct_interface_mass_jump_k)
                + q_pore_test * dot(grad_alpha_k, ddirect_interface_mass_jump)
            ) * dx
            r_pore = r_pore + direct_interface_mass_porous_k
            a_pore = a_pore + ddirect_interface_mass_porous
    elif constant_rho_s:
        combined_porous_continuity_k = div(vS_k) + div_transfer_k
        dcombined_porous_continuity = div(dvS) + ddiv_transfer
        r_pore = q_pore_test * (alpha_bulk_k * combined_porous_continuity_k) * dx
        a_pore = q_pore_test * (
            dalpha_bulk * combined_porous_continuity_k
            + alpha_bulk_k * dcombined_porous_continuity
        ) * dx
    else:
        r_pore = q_pore_test * (alpha_bulk_k * pore_transport_k) * dx
        a_pore = q_pore_test * (
            dalpha_bulk * pore_transport_k
            + alpha_bulk_k * dpore_transport
        ) * dx
    if phi_mode_key == "alpha_closure":
        phi_target_k = _c(1.0) - (_c(1.0) - phi_b_c) * alpha_bulk_k
        dphi_target = -(_c(1.0) - phi_b_c) * dalpha_bulk
        r_phi_bulk = phi_test * (phi_k - phi_target_k) * dx
        a_phi_bulk = phi_test * (dphi - dphi_target) * dx
        r_solid_mass = None
        a_solid_mass = None
    elif constant_rho_s:
        if quasi_static_porous_media:
            if phi_test is None:
                raise ValueError(
                    "quasi_static_porous_media on the constant-density final_form branch requires "
                    "the phi field so the pore-mass equation d_t(phi) + div(phi v_p) = 0 can be assembled."
                )
            # Quasi-static affects the porous momentum laws, not the transient
            # pore-mass transport. Keep the conservative phi equation live:
            #   d_t(phi) + div(phi v_p) = 0.
            #
            # The free-fluid clamp (1-alpha)^16 (phi-1) remains added below
            # through gamma_phi, so this row becomes
            #   alpha * (d_t(phi) + div(phi v_p)) + gamma_phi * (1-alpha)^16 * (phi-1) = 0.
            r_phi_bulk = phi_test * (alpha_bulk_k * pore_transport_k) * dx
            a_phi_bulk = phi_test * (
                dalpha_bulk * pore_transport_k
                + alpha_bulk_k * dpore_transport
            ) * dx
        else:
            r_phi_bulk = _safe_zero_scalar_linear_form(phi_k, phi_test)
            a_phi_bulk = _safe_zero_scalar_linear_form(phi_k, phi_test)
        r_solid_mass = None
        a_solid_mass = None
    else:
        combined_mass_bulk_k = (
            one_m_phi_k * rho_rate_k
            + rho_s_phys_k * div(vS_k)
            + rho_s_phys_k * div_transfer_k
        )
        dcombined_mass_bulk = (
            -dphi * rho_rate_k
            + one_m_phi_k * drho_rate
            + drho_s_phys * (div(vS_k) + div_transfer_k)
            + rho_s_phys_k * (div(dvS) + ddiv_transfer)
        )
        r_phi_bulk = _safe_zero_scalar_linear_form(phi_k, phi_test)
        a_phi_bulk = _safe_zero_scalar_linear_form(phi_k, phi_test)
        r_solid_mass = rho_s_test * (alpha_bulk_k * combined_mass_bulk_k) * dx
        a_solid_mass = rho_s_test * (
            dalpha_bulk * combined_mass_bulk_k + alpha_bulk_k * dcombined_mass_bulk
        ) * dx
    r_phi = _form_add(r_phi_bulk, r_solid_mass)
    a_phi = _form_add(a_phi_bulk, a_solid_mass)
    r_volumetric = None
    a_volumetric = None
    if solid_volumetric_split:
        r_volumetric = (
            alpha_bulk_k * pi_s_test * vol_drive_k
            + vol_pen_c * F_support_k * pi_s_k * pi_s_test
        ) * dx
        a_volumetric = (
            alpha_bulk_k * pi_s_test * dvol_drive
            + dalpha_bulk * pi_s_test * vol_drive_k
            + vol_pen_c * (dF_support * pi_s_k * pi_s_test + F_support_k * dpi_s * pi_s_test)
        ) * dx

    stored_support_vS_gate_n = _c(1.0)
    if float(alpha_vS_gate_alpha0) > 0.0:
        gate_pow = int(alpha_vS_gate_power)
        if gate_pow < 1:
            raise ValueError(f"alpha_vS_gate_power must be >= 1; got {alpha_vS_gate_power}.")
        gate_num = alpha_n
        for _ in range(gate_pow - 1):
            gate_num = gate_num * alpha_n
        gate_denom = gate_num + _c(float(alpha_vS_gate_alpha0) ** float(gate_pow)) + _c(1.0e-12)
        stored_support_vS_gate_n = gate_num / gate_denom
    grad_stored_support_vS_gate_n = grad(stored_support_vS_gate_n)
    vS_kin_k = stored_support_vS_gate_n * vS_k
    vS_kin_n = stored_support_vS_gate_n * vS_n
    dvS_kin = stored_support_vS_gate_n * dvS

    kin_jump_k = (u_k - u_n) * inv_dt + dot(grad(u_k), vS_kin_k) - vS_kin_k
    dkin_jump = du * inv_dt + dot(grad(du), vS_kin_k) + dot(grad(u_k), dvS_kin) - dvS_kin
    r_kin = alpha_bulk_k * dot(kin_jump_k, u_test) * dx
    a_kin = (
        dalpha_bulk * dot(kin_jump_k, u_test)
        + alpha_bulk_k
        * dot(dkin_jump, u_test)
    ) * dx

    support_physics_key = str(support_physics or "internal_conversion").strip().lower().replace("-", "_")
    adv_with_default = "biofilm_volume" if support_physics_key == "internal_conversion" else "vS"
    adv_with_key = str(alpha_advect_with or adv_with_default).strip().lower().replace("-", "_")
    adv_key = str(alpha_advection_form or "conservative_weak").strip().lower().replace("-", "_")
    if adv_key in {"conservative-weak", "weak", "weak_conservative"}:
        adv_key = "conservative_weak"
    elif adv_key in {"conservative", "div", "divergence"}:
        adv_key = "conservative"
    elif adv_key in {"advective", "nonconservative", "vgrad", "v_grad"}:
        adv_key = "advective"
    elif adv_key in {"interface_band_conservative", "interface-band-conservative", "band_conservative"}:
        adv_key = "interface_band_conservative"
    if support_physics_key == "internal_conversion" and adv_key not in {"conservative", "conservative_weak"}:
        raise ValueError(
            "final_form support_physics='internal_conversion' requires a conservative alpha law. "
            "Use alpha_advection_form='conservative_weak' (recommended) or 'conservative'."
        )
    if adv_with_key in {"vs", "v_s", "s", "skeleton", "solid"}:
        adv_u_k = vS_kin_k
        dadv_u = dvS_kin
        div_adv_u_k = stored_support_vS_gate_n * div(vS_k) + dot(grad_stored_support_vS_gate_n, vS_k)
        d_div_adv_u = stored_support_vS_gate_n * div(dvS) + dot(grad_stored_support_vS_gate_n, dvS)
    elif adv_with_key in {"v", "fluid"}:
        adv_u_k = v_k
        dadv_u = dv
        div_adv_u_k = div(v_k)
        d_div_adv_u = div(dv)
    elif adv_with_key in {"biofilm", "biofilm_volume", "phase", "phase_volume"}:
        grad_phi_k = grad(phi_k)
        adv_u_k = phi_k * vP_k + one_m_phi_k * vS_k
        dadv_u = phi_k * dvP + one_m_phi_k * dvS + dphi * (vP_k - vS_k)
        div_adv_u_k = (
            phi_k * div(vP_k)
            + dot(grad_phi_k, vP_k)
            + one_m_phi_k * div(vS_k)
            - dot(grad_phi_k, vS_k)
        )
        d_div_adv_u = (
            phi_k * div(dvP)
            + dot(grad_phi_k, dvP)
            + one_m_phi_k * div(dvS)
            - dot(grad_phi_k, dvS)
            + dphi * (div(vP_k) - div(vS_k))
            + dot(grad(dphi), vP_k - vS_k)
        )
    elif adv_with_key in {"relative", "rel", "slip", "v_minus_vs", "v_vs"}:
        adv_u_k = vP_k - vS_k
        dadv_u = dvP - dvS
        div_adv_u_k = div(vP_k) - div(vS_k)
        d_div_adv_u = div(dvP) - div(dvS)
    elif adv_with_key in {"interface", "avg", "average", "midpoint"}:
        adv_u_k = _c(0.5) * (vP_k + vS_k)
        dadv_u = _c(0.5) * (dvP + dvS)
        div_adv_u_k = _c(0.5) * (div(vP_k) + div(vS_k))
        d_div_adv_u = _c(0.5) * (div(dvP) + div(dvS))
    else:
        raise ValueError(
            f"Unsupported final_form alpha_advect_with={alpha_advect_with!r}. "
            "Use 'vS', 'v', 'biofilm_volume', 'relative', or 'interface'."
        )

    time_alpha_k = (alpha_k - alpha_n) * inv_dt
    if adv_key == "conservative_weak":
        flux_alpha_k = alpha_k * adv_u_k
        dflux_alpha = dalpha * adv_u_k + alpha_k * dadv_u
        r_alpha = alpha_test * time_alpha_k * dx - dot(flux_alpha_k, grad(alpha_test)) * dx
        a_alpha = alpha_test * (dalpha * inv_dt) * dx - dot(dflux_alpha, grad(alpha_test)) * dx
        if ds_alpha_transport is not None:
            n_b = FacetNormal()
            flux_bdry_k = dot(flux_alpha_k, n_b)
            dflux_bdry = dot(dflux_alpha, n_b)
            r_alpha = r_alpha + alpha_test * flux_bdry_k * ds_alpha_transport
            a_alpha = a_alpha + alpha_test * dflux_bdry * ds_alpha_transport
    elif adv_key == "conservative":
        adv_alpha_k = dot(grad(alpha_k), adv_u_k) + alpha_k * div_adv_u_k
        d_adv_alpha = (
            dot(grad(dalpha), adv_u_k)
            + dot(grad(alpha_k), dadv_u)
            + dalpha * div_adv_u_k
            + alpha_k * d_div_adv_u
        )
        r_alpha = alpha_test * (time_alpha_k + adv_alpha_k) * dx
        a_alpha = alpha_test * (dalpha * inv_dt + d_adv_alpha) * dx
    elif adv_key == "advective":
        adv_alpha_k = dot(grad(alpha_k), adv_u_k)
        d_adv_alpha = dot(grad(dalpha), adv_u_k) + dot(grad(alpha_k), dadv_u)
        r_alpha = alpha_test * (time_alpha_k + adv_alpha_k) * dx
        a_alpha = alpha_test * (dalpha * inv_dt + d_adv_alpha) * dx
    else:
        raise ValueError(
            f"Unsupported final_form alpha_advection_form={alpha_advection_form!r}. "
            "Use 'advective', 'conservative', or 'conservative_weak'."
        )

    one_m_alpha4_k = F_support_k * F_support_k
    one_m_alpha4_k = one_m_alpha4_k * one_m_alpha4_k
    one_m_alpha8_k = one_m_alpha4_k * one_m_alpha4_k
    rho_s_gauge_weight_k = one_m_alpha8_k * one_m_alpha8_k
    drho_s_gauge_weight = _c(16.0) * (
        one_m_alpha8_k * one_m_alpha4_k * (F_support_k * F_support_k * F_support_k)
    ) * dF_support
    if not constant_rho_s:
        r_rho_s_gauge = rho_s_gauge_c * rho_s_gauge_weight_k * (rho_s_k - rho_s_n) * rho_s_test * dx
        a_rho_s_gauge = rho_s_gauge_c * (
            drho_s_gauge_weight * (rho_s_k - rho_s_n) + rho_s_gauge_weight_k * drho_s
        ) * rho_s_test * dx
        r_phi = _form_add(r_phi, r_rho_s_gauge)
        a_phi = _form_add(a_phi, a_rho_s_gauge)

    h_ext = MeshSize()
    inv_h2_ext = _c(1.0) / (h_ext * h_ext)

    if float(gamma_phi) != 0.0:
        r_phi = _form_add(r_phi, gamma_phi_c * rho_s_gauge_weight_k * (phi_k - _c(1.0)) * phi_test * dx)
        a_phi = _form_add(a_phi, gamma_phi_c * (
            drho_s_gauge_weight * (phi_k - _c(1.0)) + rho_s_gauge_weight_k * dphi
        ) * phi_test * dx)

    if float(interface_band_extension_gamma) != 0.0:
        if phi_test is not None:
            r_phi = _form_add(
                r_phi,
                gamma_band_ext_c * interface_band_weight_k * dot(grad(phi_k), grad(phi_test)) * dx,
            )
            a_phi = _form_add(
                a_phi,
                gamma_band_ext_c * (
                    dinterface_band_weight * dot(grad(phi_k), grad(phi_test))
                    + interface_band_weight_k * dot(grad(dphi), grad(phi_test))
                ) * dx,
            )
        r_kin = r_kin + gamma_band_ext_c * interface_band_weight_k * inner(grad(u_k), grad(u_test)) * dx
        a_kin = a_kin + gamma_band_ext_c * (
            dinterface_band_weight * inner(grad(u_k), grad(u_test))
            + interface_band_weight_k * inner(grad(du), grad(u_test))
        ) * dx
        r_skel = r_skel + gamma_band_ext_c * interface_band_weight_k * inner(grad(vS_k), grad(vS_test)) * dx
        a_skel = a_skel + gamma_band_ext_c * (
            dinterface_band_weight * inner(grad(vS_k), grad(vS_test))
            + interface_band_weight_k * inner(grad(dvS), grad(vS_test))
        ) * dx

    if (not constant_rho_s) and float(gamma_rho_s) != 0.0:
        if rho_s_ext_mode in {"l2", "mass"}:
            r_phi = _form_add(r_phi, gamma_rho_s_c * inv_h2_ext * F_support_k * (rho_s_k - rho_s_ref_c) * rho_s_test * dx)
            a_phi = _form_add(a_phi, gamma_rho_s_c * inv_h2_ext * (
                dF_support * (rho_s_k - rho_s_ref_c) + F_support_k * drho_s
            ) * rho_s_test * dx)
        else:
            r_phi = _form_add(r_phi, gamma_rho_s_c * F_support_k * dot(grad(rho_s_k), grad(rho_s_test)) * dx)
            a_phi = _form_add(a_phi, gamma_rho_s_c * (
                dF_support * dot(grad(rho_s_k), grad(rho_s_test))
                + F_support_k * dot(grad(drho_s), grad(rho_s_test))
            ) * dx)
            if float(gamma_rho_s_pin) != 0.0:
                w_pin2 = F_support_k * F_support_k
                dw_pin2 = _c(2.0) * F_support_k * dF_support
                r_phi = _form_add(r_phi, gamma_rho_s_pin_c * inv_h2_ext * w_pin2 * (rho_s_k - rho_s_ref_c) * rho_s_test * dx)
                a_phi = _form_add(a_phi, gamma_rho_s_pin_c * inv_h2_ext * (
                    dw_pin2 * (rho_s_k - rho_s_ref_c) + w_pin2 * drho_s
                ) * rho_s_test * dx)

    if float(gamma_v) != 0.0:
        if v_ext_mode in {"l2", "mass"}:
            r_mom_f = r_mom_f + gamma_v_c * inv_h2_ext * alpha_support_k * dot(v_k, v_test) * dx
            a_mom_f = a_mom_f + gamma_v_c * inv_h2_ext * (
                dalpha_support * dot(v_k, v_test) + alpha_support_k * dot(dv, v_test)
            ) * dx
        else:
            r_mom_f = r_mom_f + gamma_v_c * alpha_support_k * inner(grad(v_k), grad(v_test)) * dx
            a_mom_f = a_mom_f + gamma_v_c * (
                dalpha_support * inner(grad(v_k), grad(v_test)) + alpha_support_k * inner(grad(dv), grad(v_test))
            ) * dx
            if float(gamma_v_pin) != 0.0:
                w_pin2 = alpha_support_k * alpha_support_k
                dw_pin2 = _c(2.0) * alpha_support_k * dalpha_support
                r_mom_f = r_mom_f + gamma_v_pin_c * inv_h2_ext * w_pin2 * dot(v_k, v_test) * dx
                a_mom_f = a_mom_f + gamma_v_pin_c * inv_h2_ext * (
                    dw_pin2 * dot(v_k, v_test) + w_pin2 * dot(dv, v_test)
                ) * dx

    if float(gamma_p) != 0.0:
        if p_ext_mode in {"l2", "mass"}:
            r_mass = r_mass + gamma_p_c * inv_h2_ext * alpha_support_k * p_k * q_test * dx
            a_mass = a_mass + gamma_p_c * inv_h2_ext * (
                dalpha_support * p_k + alpha_support_k * dp
            ) * q_test * dx
        else:
            r_mass = r_mass + gamma_p_c * alpha_support_k * dot(grad(p_k), grad(q_test)) * dx
            a_mass = a_mass + gamma_p_c * (
                dalpha_support * dot(grad(p_k), grad(q_test)) + alpha_support_k * dot(grad(dp), grad(q_test))
            ) * dx
            if float(gamma_p_pin) != 0.0:
                w_pin2 = alpha_support_k * alpha_support_k
                dw_pin2 = _c(2.0) * alpha_support_k * dalpha_support
                r_mass = r_mass + gamma_p_pin_c * inv_h2_ext * w_pin2 * p_k * q_test * dx
                a_mass = a_mass + gamma_p_pin_c * inv_h2_ext * (
                    dw_pin2 * p_k + w_pin2 * dp
                ) * q_test * dx

    if float(gamma_vP) != 0.0:
        if vP_ext_mode in {"l2", "mass"}:
            r_mom_p = r_mom_p + gamma_vP_c * inv_h2_ext * F_support_k * dot(vP_k, vP_test) * dx
            a_mom_p = a_mom_p + gamma_vP_c * inv_h2_ext * (
                dF_support * dot(vP_k, vP_test) + F_support_k * dot(dvP, vP_test)
            ) * dx
        else:
            r_mom_p = r_mom_p + gamma_vP_c * F_support_k * inner(grad(vP_k), grad(vP_test)) * dx
            a_mom_p = a_mom_p + gamma_vP_c * (
                dF_support * inner(grad(vP_k), grad(vP_test)) + F_support_k * inner(grad(dvP), grad(vP_test))
            ) * dx
            if float(gamma_vP_pin) != 0.0:
                w_pin2 = F_support_k * F_support_k
                dw_pin2 = _c(2.0) * F_support_k * dF_support
                r_mom_p = r_mom_p + gamma_vP_pin_c * inv_h2_ext * w_pin2 * dot(vP_k, vP_test) * dx
                a_mom_p = a_mom_p + gamma_vP_pin_c * inv_h2_ext * (
                    dw_pin2 * dot(vP_k, vP_test) + w_pin2 * dot(dvP, vP_test)
                ) * dx

    if float(gamma_u) != 0.0:
        if u_ext_mode in {"l2", "mass"}:
            r_kin = r_kin + gamma_u_c * inv_h2_ext * F_support_k * dot(u_k, u_test) * dx
            a_kin = a_kin + gamma_u_c * inv_h2_ext * (
                dF_support * dot(u_k, u_test) + F_support_k * dot(du, u_test)
            ) * dx
        else:
            r_kin = r_kin + gamma_u_c * F_support_k * inner(grad(u_k), grad(u_test)) * dx
            a_kin = a_kin + gamma_u_c * (
                dF_support * inner(grad(u_k), grad(u_test)) + F_support_k * inner(grad(du), grad(u_test))
            ) * dx
            if float(gamma_u_pin) != 0.0:
                w_pin2 = F_support_k * F_support_k
                dw_pin2 = _c(2.0) * F_support_k * dF_support
                r_kin = r_kin + gamma_u_pin_c * inv_h2_ext * w_pin2 * dot(u_k, u_test) * dx
                a_kin = a_kin + gamma_u_pin_c * inv_h2_ext * (
                    dw_pin2 * dot(u_k, u_test) + w_pin2 * dot(du, u_test)
                ) * dx

    if float(u_cip) != 0.0 and ds_cip is not None:
        n_int = FacetNormal()
        h_F = avg(MeshSize())
        tau_u_cip = u_cip_c * (h_F * h_F * h_F) * inv_dt
        w_key = str(u_cip_weight or "fluid").strip().lower()
        if w_key in {"fluid", "one_minus_alpha", "1-alpha", "one-minus-alpha"}:
            w_u_cip = avg(F_support_k)
        elif w_key in {"biofilm", "alpha", "porous", "support"}:
            w_u_cip = avg(alpha_support_n)
        elif w_key in {"both", "all", "unity", "1"}:
            w_u_cip = _c(1.0)
        else:
            raise ValueError(
                f"Unknown u_cip_weight={u_cip_weight!r}. Use 'fluid' (default), 'biofilm', or 'both'."
            )
        r_kin = r_kin + tau_u_cip * w_u_cip * _grad_inner_jump(u_k, u_test, n_int) * ds_cip
        a_kin = a_kin + tau_u_cip * w_u_cip * _grad_inner_jump(du, u_test, n_int) * ds_cip

    if float(gamma_vS_input) != 0.0:
        if vS_ext_mode in {"l2", "mass"}:
            r_skel = r_skel + gamma_vS_c * inv_h2_ext * F_support_k * dot(vS_k, vS_test) * dx
            a_skel = a_skel + gamma_vS_c * inv_h2_ext * (
                dF_support * dot(vS_k, vS_test) + F_support_k * dot(dvS, vS_test)
            ) * dx
        else:
            r_skel = r_skel + gamma_vS_c * F_support_k * inner(grad(vS_k), grad(vS_test)) * dx
            a_skel = a_skel + gamma_vS_c * (
                dF_support * inner(grad(vS_k), grad(vS_test)) + F_support_k * inner(grad(dvS), grad(vS_test))
            ) * dx
            if float(gamma_vS_pin_input) != 0.0:
                w_pin2 = F_support_k * F_support_k
                dw_pin2 = _c(2.0) * F_support_k * dF_support
                r_skel = r_skel + gamma_vS_pin_c * inv_h2_ext * w_pin2 * dot(vS_k, vS_test) * dx
                a_skel = a_skel + gamma_vS_pin_c * inv_h2_ext * (
                    dw_pin2 * dot(vS_k, vS_test) + w_pin2 * dot(dvS, vS_test)
                ) * dx

    if float(vS_cip) != 0.0 and ds_cip is not None:
        n_int = FacetNormal()
        h_F = avg(MeshSize())
        tau_vS_cip = vS_cip_c * (h_F * h_F * h_F) * inv_dt
        w_s_cip = avg(alpha_support_n)
        r_skel = r_skel + tau_vS_cip * w_s_cip * _grad_inner_jump(vS_k, vS_test, n_int) * ds_cip
        a_skel = a_skel + tau_vS_cip * w_s_cip * _grad_inner_jump(dvS, vS_test, n_int) * ds_cip

    if mu_kin_k is not None and mu_kin_test is not None and dmu_kin is not None:
        kin_constraint_k = grad_alpha_mag * kin_jump_k
        dkin_constraint = dgrad_alpha_mag * kin_jump_k + grad_alpha_mag * dkin_jump
        kinematic_test_bracket_u = grad_alpha_mag * (u_test * inv_dt + dot(grad(u_test), vS_k))
        kinematic_test_bracket_vS = grad_alpha_mag * (dot(grad(u_k), vS_test) - vS_test)
        kinematic_test_bracket_alpha = (
            dot(grad(alpha_k), grad(alpha_test)) * inv_grad_alpha_mag
        ) * kin_jump_k
        r_interface_kin = dot(mu_kin_test, kin_constraint_k) * dx
        a_interface_kin = dot(mu_kin_test, dkin_constraint) * dx
        r_interface_kin_bulk_u = dot(mu_kin_k, kinematic_test_bracket_u) * dx
        a_interface_kin_bulk_u = dot(dmu_kin, kinematic_test_bracket_u) * dx
        r_interface_kin_bulk_vS = dot(mu_kin_k, kinematic_test_bracket_vS) * dx
        a_interface_kin_bulk_vS = dot(dmu_kin, kinematic_test_bracket_vS) * dx
        r_interface_kin_bulk_alpha = dot(mu_kin_k, kinematic_test_bracket_alpha) * dx
        a_interface_kin_bulk_alpha = dot(dmu_kin, kinematic_test_bracket_alpha) * dx
    else:
        zero_mu_kin_form = zero * dot(u_k, u_test) * dx
        r_interface_kin = zero_mu_kin_form
        a_interface_kin = zero_mu_kin_form
        r_interface_kin_bulk_u = zero_mu_kin_form
        a_interface_kin_bulk_u = zero_mu_kin_form
        r_interface_kin_bulk_vS = zero_mu_kin_form
        a_interface_kin_bulk_vS = zero_mu_kin_form
        r_interface_kin_bulk_alpha = _safe_zero_scalar_linear_form(alpha_k, alpha_test)
        a_interface_kin_bulk_alpha = _safe_zero_scalar_linear_form(alpha_k, alpha_test)

    if interface_formulation_key == "decomposed":
        def _traction_directional(
            *,
            dv_var=None,
            dp_var=None,
            dvP_var=None,
            dp_pore_var=None,
            du_var=None,
            dphi_var=None,
            dalpha_var=None,
        ):
            dv_loc = zero_v if dv_var is None else dv_var
            dp_loc = zero if dp_var is None else dp_var
            dvP_loc = zero_v if dvP_var is None else dvP_var
            dp_pore_loc = zero if dp_pore_var is None else dp_pore_var
            du_loc = zero_u if du_var is None else du_var
            dphi_loc = zero if dphi_var is None else dphi_var
            dalpha_loc = zero if dalpha_var is None else dalpha_var

            dsigma_f_loc = _c(2.0) * mu_f_c * _eps(dv_loc) - dp_loc * Identity(dim)
            mu_pore_eff_loc, dmu_pore_eff_loc = _pore_effective_viscosity_terms(phi_k, dphi_loc)
            dsigma_p_loc = (
                _c(2.0) * dmu_pore_eff_loc * _eps(vP_k)
                + _c(2.0) * mu_pore_eff_loc * _eps(dvP_loc)
                - dp_pore_loc * Identity(dim)
            )
            if solid_volumetric_split:
                if solid_model_key in {"linear", "small_strain", "linear_elastic"}:
                    dsigma_s_dev_loc = _c(2.0) * mu_s_c * _deviatoric_tensor(_eps(du_loc), dim=int(dim))
                else:
                    _, dsigma_s_full_loc = _solid_stress_and_tangent(
                        solid_model=str(solid_model),
                        u_k=u_k,
                        du=du_loc,
                        mu_s=mu_s_c,
                        lambda_s=lambda_s_c,
                        dim=int(dim),
                    )
                    dmean_dr_loc = _tensor_trace(dsigma_s_full_loc, dim=int(dim)) / _c(float(dim))
                    dsigma_s_dev_loc = dsigma_s_full_loc - dmean_dr_loc * Identity(dim)
                dsigma_s_loc = dsigma_s_dev_loc
            else:
                _, dsigma_s_loc = _solid_stress_and_tangent(
                    solid_model=str(solid_model),
                    u_k=u_k,
                    du=du_loc,
                    mu_s=mu_s_c,
                    lambda_s=lambda_s_c,
                    dim=int(dim),
                )
            dgrad_alpha_loc = grad(dalpha_loc)
            dgrad_alpha_mag_loc = dot(grad_alpha_k, dgrad_alpha_loc) * inv_grad_alpha_mag
            dn0_loc = (dgrad_alpha_loc[0] - n0 * dgrad_alpha_mag_loc) * inv_grad_alpha_mag
            dn1_loc = (dgrad_alpha_loc[1] - n1 * dgrad_alpha_mag_loc) * inv_grad_alpha_mag
            dt0_loc = -dn1_loc
            dt1_loc = dn0_loc

            dfluid_n_x_loc = (
                dsigma_f_loc[0, 0] * n0
                + sigma_f_k[0, 0] * dn0_loc
                + dsigma_f_loc[0, 1] * n1
                + sigma_f_k[0, 1] * dn1_loc
            )
            dfluid_n_y_loc = (
                dsigma_f_loc[1, 0] * n0
                + sigma_f_k[1, 0] * dn0_loc
                + dsigma_f_loc[1, 1] * n1
                + sigma_f_k[1, 1] * dn1_loc
            )
            dpore_n_x_loc = (
                dsigma_p_loc[0, 0] * n0
                + sigma_p_k[0, 0] * dn0_loc
                + dsigma_p_loc[0, 1] * n1
                + sigma_p_k[0, 1] * dn1_loc
            )
            dpore_n_y_loc = (
                dsigma_p_loc[1, 0] * n0
                + sigma_p_k[1, 0] * dn0_loc
                + dsigma_p_loc[1, 1] * n1
                + sigma_p_k[1, 1] * dn1_loc
            )
            dsolid_n_x_loc = (
                dsigma_s_loc[0, 0] * n0
                + sigma_s_k[0, 0] * dn0_loc
                + dsigma_s_loc[0, 1] * n1
                + sigma_s_k[0, 1] * dn1_loc
            )
            dsolid_n_y_loc = (
                dsigma_s_loc[1, 0] * n0
                + sigma_s_k[1, 0] * dn0_loc
                + dsigma_s_loc[1, 1] * n1
                + sigma_s_k[1, 1] * dn1_loc
            )
            dfluid_normal_trac_loc = (
                dn0_loc * fluid_n_x
                + n0 * dfluid_n_x_loc
                + dn1_loc * fluid_n_y
                + n1 * dfluid_n_y_loc
            )
            dpore_normal_trac_loc = (
                dn0_loc * pore_n_x
                + n0 * dpore_n_x_loc
                + dn1_loc * pore_n_y
                + n1 * dpore_n_y_loc
            )
            dsolid_normal_trac_loc = (
                dn0_loc * solid_n_x
                + n0 * dsolid_n_x_loc
                + dn1_loc * solid_n_y
                + n1 * dsolid_n_y_loc
            )
            dfluid_tang_trac_loc = (
                dt0_loc * fluid_n_x
                + t0 * dfluid_n_x_loc
                + dt1_loc * fluid_n_y
                + t1 * dfluid_n_y_loc
            )
            dpore_tang_trac_loc = (
                dt0_loc * pore_n_x
                + t0 * dpore_n_x_loc
                + dt1_loc * pore_n_y
                + t1 * dpore_n_y_loc
            )
            dsolid_tang_trac_loc = (
                dt0_loc * solid_n_x
                + t0 * dsolid_n_x_loc
                + dt1_loc * solid_n_y
                + t1 * dsolid_n_y_loc
            )
            dnormal_jump_loc = (
                dfluid_normal_trac_loc
                - (dphi_loc * pore_normal_trac + phi_k * dpore_normal_trac_loc)
                + solid_interface_traction_weight_c * dphi_loc * solid_normal_trac
                - solid_interface_traction_weight_c * one_m_phi_k * dsolid_normal_trac_loc
            )
            dtangential_jump_loc = (
                dfluid_tang_trac_loc
                - (dphi_loc * pore_tang_trac + phi_k * dpore_tang_trac_loc)
                + solid_interface_traction_weight_c * dphi_loc * solid_tang_trac
                - solid_interface_traction_weight_c * one_m_phi_k * dsolid_tang_trac_loc
            )
            return (
                dgrad_alpha_mag_loc * normal_traction_jump + grad_alpha_mag * dnormal_jump_loc,
                dgrad_alpha_mag_loc * tangential_traction_jump + grad_alpha_mag * dtangential_jump_loc,
            )

        fluid_n_x = sigma_f_k[0, 0] * n0 + sigma_f_k[0, 1] * n1
        fluid_n_y = sigma_f_k[1, 0] * n0 + sigma_f_k[1, 1] * n1
        dfluid_n_x = dsigma_f[0, 0] * n0 + sigma_f_k[0, 0] * dn0 + dsigma_f[0, 1] * n1 + sigma_f_k[0, 1] * dn1
        dfluid_n_y = dsigma_f[1, 0] * n0 + sigma_f_k[1, 0] * dn0 + dsigma_f[1, 1] * n1 + sigma_f_k[1, 1] * dn1
        fluid_normal_trac = n0 * fluid_n_x + n1 * fluid_n_y
        dfluid_normal_trac = dn0 * fluid_n_x + n0 * dfluid_n_x + dn1 * fluid_n_y + n1 * dfluid_n_y
        pore_n_x = sigma_p_k[0, 0] * n0 + sigma_p_k[0, 1] * n1
        pore_n_y = sigma_p_k[1, 0] * n0 + sigma_p_k[1, 1] * n1
        dpore_n_x = dsigma_p[0, 0] * n0 + sigma_p_k[0, 0] * dn0 + dsigma_p[0, 1] * n1 + sigma_p_k[0, 1] * dn1
        dpore_n_y = dsigma_p[1, 0] * n0 + sigma_p_k[1, 0] * dn0 + dsigma_p[1, 1] * n1 + sigma_p_k[1, 1] * dn1
        pore_normal_trac = n0 * pore_n_x + n1 * pore_n_y
        dpore_normal_trac = dn0 * pore_n_x + n0 * dpore_n_x + dn1 * pore_n_y + n1 * dpore_n_y
        solid_n_x = sigma_s_k[0, 0] * n0 + sigma_s_k[0, 1] * n1
        solid_n_y = sigma_s_k[1, 0] * n0 + sigma_s_k[1, 1] * n1
        dsolid_n_x = dsigma_s[0, 0] * n0 + sigma_s_k[0, 0] * dn0 + dsigma_s[0, 1] * n1 + sigma_s_k[0, 1] * dn1
        dsolid_n_y = dsigma_s[1, 0] * n0 + sigma_s_k[1, 0] * dn0 + dsigma_s[1, 1] * n1 + sigma_s_k[1, 1] * dn1
        solid_normal_trac = n0 * solid_n_x + n1 * solid_n_y
        dsolid_normal_trac = dn0 * solid_n_x + n0 * dsolid_n_x + dn1 * solid_n_y + n1 * dsolid_n_y
        fluid_tang_trac = t0 * fluid_n_x + t1 * fluid_n_y
        dfluid_tang_trac = dt0 * fluid_n_x + t0 * dfluid_n_x + dt1 * fluid_n_y + t1 * dfluid_n_y
        pore_tang_trac = t0 * pore_n_x + t1 * pore_n_y
        dpore_tang_trac = dt0 * pore_n_x + t0 * dpore_n_x + dt1 * pore_n_y + t1 * dpore_n_y
        solid_tang_trac = t0 * solid_n_x + t1 * solid_n_y
        dsolid_tang_trac = dt0 * solid_n_x + t0 * dsolid_n_x + dt1 * solid_n_y + t1 * dsolid_n_y
        normal_traction_jump = (
            fluid_normal_trac
            - phi_k * pore_normal_trac
            - solid_interface_traction_weight_c * one_m_phi_k * solid_normal_trac
        )
        dnormal_traction_jump = (
            dfluid_normal_trac
            - (dphi * pore_normal_trac + phi_k * dpore_normal_trac)
            + solid_interface_traction_weight_c * dphi * solid_normal_trac
            - solid_interface_traction_weight_c * one_m_phi_k * dsolid_normal_trac
        )
        tangential_traction_jump = (
            fluid_tang_trac
            - phi_k * pore_tang_trac
            - solid_interface_traction_weight_c * one_m_phi_k * solid_tang_trac
        )
        dtangential_traction_jump = (
            dfluid_tang_trac
            - (dphi * pore_tang_trac + phi_k * dpore_tang_trac)
            + solid_interface_traction_weight_c * dphi * solid_tang_trac
            - solid_interface_traction_weight_c * one_m_phi_k * dsolid_tang_trac
        )
        fluid_tangential_velocity_k = dot(tangent_vec, v_k)
        dfluid_tangential_velocity = dot(dtangent_vec, v_k) + dot(tangent_vec, dv)
        pore_tangential_velocity_k = dot(tangent_vec, vP_k)
        dpore_tangential_velocity = dot(dtangent_vec, vP_k) + dot(tangent_vec, dvP)
        solid_tangential_velocity_k = dot(tangent_vec, vS_k)
        dsolid_tangential_velocity = dot(dtangent_vec, vS_k) + dot(tangent_vec, dvS)
        bjs_mixture_tangential_velocity_k = (
            phi_k * pore_tangential_velocity_k + one_m_phi_k * solid_tangential_velocity_k
        )
        dbjs_mixture_tangential_velocity = (
            dphi * pore_tangential_velocity_k
            + phi_k * dpore_tangential_velocity
            - dphi * solid_tangential_velocity_k
            + one_m_phi_k * dsolid_tangential_velocity
        )
        bjs_beta_k = bjs_coeff_c * mu_f_c * sqrt_kappa_inv_c
        dbjs_beta = zero
        bjs_slip_t_k = fluid_tangential_velocity_k - bjs_mixture_tangential_velocity_k
        dbjs_slip_t = dfluid_tangential_velocity - dbjs_mixture_tangential_velocity
        bjs_tangential_jump = fluid_tang_trac - bjs_beta_k * bjs_slip_t_k
        dbjs_tangential_jump = (
            dfluid_tang_trac
            - (dbjs_beta * bjs_slip_t_k + bjs_beta_k * dbjs_slip_t)
        )
        fluid_tangential_velocity_test = t0 * v_test[0] + t1 * v_test[1]

        def _active_tangential_interface_directional(
            *,
            dv_var=None,
            dvP_var=None,
            dvS_var=None,
            du_var=None,
            dphi_var=None,
            dalpha_var=None,
        ):
            if not use_bjs_tangential_law:
                return _traction_directional(
                    dv_var=dv_var,
                    dvP_var=dvP_var,
                    du_var=du_var,
                    dphi_var=dphi_var,
                    dalpha_var=dalpha_var,
                )[1]
            dv_loc = zero_v if dv_var is None else dv_var
            dvP_loc = zero_v if dvP_var is None else dvP_var
            dvS_loc = zero_v if dvS_var is None else dvS_var
            dphi_loc = zero if dphi_var is None else dphi_var
            dalpha_loc = zero if dalpha_var is None else dalpha_var
            dgrad_alpha_loc = grad(dalpha_loc)
            dgrad_alpha_mag_loc = dot(grad_alpha_k, dgrad_alpha_loc) * inv_grad_alpha_mag
            dn0_loc = (dgrad_alpha_loc[0] - n0 * dgrad_alpha_mag_loc) * inv_grad_alpha_mag
            dn1_loc = (dgrad_alpha_loc[1] - n1 * dgrad_alpha_mag_loc) * inv_grad_alpha_mag
            dt0_loc = -dn1_loc
            dt1_loc = dn0_loc
            dsigma_f_loc = _c(2.0) * mu_f_c * _eps(dv_loc)
            dfluid_n_x_loc = (
                dsigma_f_loc[0, 0] * n0
                + sigma_f_k[0, 0] * dn0_loc
                + dsigma_f_loc[0, 1] * n1
                + sigma_f_k[0, 1] * dn1_loc
            )
            dfluid_n_y_loc = (
                dsigma_f_loc[1, 0] * n0
                + sigma_f_k[1, 0] * dn0_loc
                + dsigma_f_loc[1, 1] * n1
                + sigma_f_k[1, 1] * dn1_loc
            )
            dfluid_tang_trac_loc = (
                dt0_loc * fluid_n_x
                + t0 * dfluid_n_x_loc
                + dt1_loc * fluid_n_y
                + t1 * dfluid_n_y_loc
            )
            dfluid_tangential_velocity_loc = (
                dt0_loc * v_k[0] + t0 * dv_loc[0] + dt1_loc * v_k[1] + t1 * dv_loc[1]
            )
            dpore_tangential_velocity_loc = (
                dt0_loc * vP_k[0] + t0 * dvP_loc[0] + dt1_loc * vP_k[1] + t1 * dvP_loc[1]
            )
            dsolid_tangential_velocity_loc = (
                dt0_loc * vS_k[0] + t0 * dvS_loc[0] + dt1_loc * vS_k[1] + t1 * dvS_loc[1]
            )
            dbjs_mixture_tangential_velocity_loc = (
                dphi_loc * pore_tangential_velocity_k + phi_k * dpore_tangential_velocity_loc
                - dphi_loc * solid_tangential_velocity_k + one_m_phi_k * dsolid_tangential_velocity_loc
            )
            dbjs_beta_loc = zero
            dbjs_slip_t_loc = dfluid_tangential_velocity_loc - dbjs_mixture_tangential_velocity_loc
            dbjs_tangential_jump_loc = (
                dfluid_tang_trac_loc
                - (dbjs_beta_loc * bjs_slip_t_k + bjs_beta_k * dbjs_slip_t_loc)
            )
            return (
                dgrad_alpha_mag_loc * bjs_tangential_jump
                + grad_alpha_mag * dbjs_tangential_jump_loc
            )
    else:
        def _traction_directional(
            *,
            dv_var=None,
            dp_var=None,
            dvP_var=None,
            dp_pore_var=None,
            du_var=None,
            dphi_var=None,
            dalpha_var=None,
        ):
            dv_loc = zero_v if dv_var is None else dv_var
            dp_loc = zero if dp_var is None else dp_var
            dvP_loc = zero_v if dvP_var is None else dvP_var
            dp_pore_loc = zero if dp_pore_var is None else dp_pore_var
            du_loc = zero_u if du_var is None else du_var
            dphi_loc = zero if dphi_var is None else dphi_var
            dalpha_loc = zero if dalpha_var is None else dalpha_var

            dsigma_f_loc = _c(2.0) * mu_f_c * _eps(dv_loc) - dp_loc * Identity(dim)
            mu_pore_eff_loc, dmu_pore_eff_loc = _pore_effective_viscosity_terms(phi_k, dphi_loc)
            dsigma_p_loc = (
                _c(2.0) * dmu_pore_eff_loc * _eps(vP_k)
                + _c(2.0) * mu_pore_eff_loc * _eps(dvP_loc)
                - dp_pore_loc * Identity(dim)
            )
            if solid_volumetric_split:
                if solid_model_key in {"linear", "small_strain", "linear_elastic"}:
                    dsigma_s_dev_loc = _c(2.0) * mu_s_c * _deviatoric_tensor(_eps(du_loc), dim=int(dim))
                else:
                    _, dsigma_s_full_loc = _solid_stress_and_tangent(
                        solid_model=str(solid_model),
                        u_k=u_k,
                        du=du_loc,
                        mu_s=mu_s_c,
                        lambda_s=lambda_s_c,
                        dim=int(dim),
                    )
                    dmean_dr_loc = _tensor_trace(dsigma_s_full_loc, dim=int(dim)) / _c(float(dim))
                    dsigma_s_dev_loc = dsigma_s_full_loc - dmean_dr_loc * Identity(dim)
                dsigma_s_loc = dsigma_s_dev_loc
            else:
                _, dsigma_s_loc = _solid_stress_and_tangent(
                    solid_model=str(solid_model),
                    u_k=u_k,
                    du=du_loc,
                    mu_s=mu_s_c,
                    lambda_s=lambda_s_c,
                    dim=int(dim),
                )
            dgrad_alpha_loc = grad(dalpha_loc)
            dgrad_alpha_mag_loc = dot(grad_alpha_k, dgrad_alpha_loc) * inv_grad_alpha_mag
            dnormal_loc = (dgrad_alpha_loc - normal_vec * dgrad_alpha_mag_loc) * inv_grad_alpha_mag
            dn0_loc = dnormal_loc[0]
            dn1_loc = dnormal_loc[1]
            dt0_loc = -dn1_loc
            dt1_loc = dn0_loc

            dsigma_f_n_x_loc, dsigma_f_n_y_loc = _stress_times_dir_components(dsigma_f_loc, n0, n1)
            sigma_f_dn_x_loc, sigma_f_dn_y_loc = _stress_times_dir_components(sigma_f_k, dn0_loc, dn1_loc)
            dfluid_n_x_loc = dsigma_f_n_x_loc + sigma_f_dn_x_loc
            dfluid_n_y_loc = dsigma_f_n_y_loc + sigma_f_dn_y_loc

            dsigma_p_n_x_loc, dsigma_p_n_y_loc = _stress_times_dir_components(dsigma_p_loc, n0, n1)
            sigma_p_dn_x_loc, sigma_p_dn_y_loc = _stress_times_dir_components(sigma_p_k, dn0_loc, dn1_loc)
            dpore_n_x_loc = dsigma_p_n_x_loc + sigma_p_dn_x_loc
            dpore_n_y_loc = dsigma_p_n_y_loc + sigma_p_dn_y_loc

            dsigma_s_n_x_loc, dsigma_s_n_y_loc = _stress_times_dir_components(dsigma_s_loc, n0, n1)
            sigma_s_dn_x_loc, sigma_s_dn_y_loc = _stress_times_dir_components(sigma_s_k, dn0_loc, dn1_loc)
            dsolid_n_x_loc = dsigma_s_n_x_loc + sigma_s_dn_x_loc
            dsolid_n_y_loc = dsigma_s_n_y_loc + sigma_s_dn_y_loc

            dfluid_normal_trac_loc = (
                dn0_loc * fluid_n_x + n0 * dfluid_n_x_loc + dn1_loc * fluid_n_y + n1 * dfluid_n_y_loc
            )
            dpore_normal_trac_loc = (
                dn0_loc * pore_n_x + n0 * dpore_n_x_loc + dn1_loc * pore_n_y + n1 * dpore_n_y_loc
            )
            dsolid_normal_trac_loc = (
                dn0_loc * solid_n_x + n0 * dsolid_n_x_loc + dn1_loc * solid_n_y + n1 * dsolid_n_y_loc
            )
            dfluid_tang_trac_loc = (
                dt0_loc * fluid_n_x + t0 * dfluid_n_x_loc + dt1_loc * fluid_n_y + t1 * dfluid_n_y_loc
            )
            dpore_tang_trac_loc = (
                dt0_loc * pore_n_x + t0 * dpore_n_x_loc + dt1_loc * pore_n_y + t1 * dpore_n_y_loc
            )
            dsolid_tang_trac_loc = (
                dt0_loc * solid_n_x + t0 * dsolid_n_x_loc + dt1_loc * solid_n_y + t1 * dsolid_n_y_loc
            )
            dnormal_jump_loc = (
                dfluid_normal_trac_loc
                - (dphi_loc * pore_normal_trac + phi_k * dpore_normal_trac_loc)
                + solid_interface_traction_weight_c * dphi_loc * solid_normal_trac
                - solid_interface_traction_weight_c * one_m_phi_k * dsolid_normal_trac_loc
            )
            dtangential_jump_loc = (
                dfluid_tang_trac_loc
                - (dphi_loc * pore_tang_trac + phi_k * dpore_tang_trac_loc)
                + solid_interface_traction_weight_c * dphi_loc * solid_tang_trac
                - solid_interface_traction_weight_c * one_m_phi_k * dsolid_tang_trac_loc
            )
            return (
                dgrad_alpha_mag_loc * normal_traction_jump + grad_alpha_mag * dnormal_jump_loc,
                dgrad_alpha_mag_loc * tangential_traction_jump + grad_alpha_mag * dtangential_jump_loc,
            )

        fluid_n_x, fluid_n_y = _stress_times_dir_components(sigma_f_k, n0, n1)
        dfluid_sigma_n_x, dfluid_sigma_n_y = _stress_times_dir_components(dsigma_f, n0, n1)
        fluid_sigma_dn_x, fluid_sigma_dn_y = _stress_times_dir_components(sigma_f_k, dn0, dn1)
        dfluid_n_x = dfluid_sigma_n_x + fluid_sigma_dn_x
        dfluid_n_y = dfluid_sigma_n_y + fluid_sigma_dn_y
        fluid_normal_trac = _normal_trace_from_components(fluid_n_x, fluid_n_y, n0, n1)
        dfluid_normal_trac = dn0 * fluid_n_x + n0 * dfluid_n_x + dn1 * fluid_n_y + n1 * dfluid_n_y

        pore_n_x, pore_n_y = _stress_times_dir_components(sigma_p_k, n0, n1)
        dpore_sigma_n_x, dpore_sigma_n_y = _stress_times_dir_components(dsigma_p, n0, n1)
        pore_sigma_dn_x, pore_sigma_dn_y = _stress_times_dir_components(sigma_p_k, dn0, dn1)
        dpore_n_x = dpore_sigma_n_x + pore_sigma_dn_x
        dpore_n_y = dpore_sigma_n_y + pore_sigma_dn_y
        pore_normal_trac = _normal_trace_from_components(pore_n_x, pore_n_y, n0, n1)
        dpore_normal_trac = dn0 * pore_n_x + n0 * dpore_n_x + dn1 * pore_n_y + n1 * dpore_n_y

        solid_n_x, solid_n_y = _stress_times_dir_components(sigma_s_k, n0, n1)
        dsolid_sigma_n_x, dsolid_sigma_n_y = _stress_times_dir_components(dsigma_s, n0, n1)
        solid_sigma_dn_x, solid_sigma_dn_y = _stress_times_dir_components(sigma_s_k, dn0, dn1)
        dsolid_n_x = dsolid_sigma_n_x + solid_sigma_dn_x
        dsolid_n_y = dsolid_sigma_n_y + solid_sigma_dn_y
        solid_normal_trac = _normal_trace_from_components(solid_n_x, solid_n_y, n0, n1)
        dsolid_normal_trac = dn0 * solid_n_x + n0 * dsolid_n_x + dn1 * solid_n_y + n1 * dsolid_n_y

        fluid_tang_trac = _tangential_trace_from_components(fluid_n_x, fluid_n_y, t0, t1)
        dfluid_tang_trac = dt0 * fluid_n_x + t0 * dfluid_n_x + dt1 * fluid_n_y + t1 * dfluid_n_y
        pore_tang_trac = _tangential_trace_from_components(pore_n_x, pore_n_y, t0, t1)
        dpore_tang_trac = dt0 * pore_n_x + t0 * dpore_n_x + dt1 * pore_n_y + t1 * dpore_n_y
        solid_tang_trac = _tangential_trace_from_components(solid_n_x, solid_n_y, t0, t1)
        dsolid_tang_trac = dt0 * solid_n_x + t0 * dsolid_n_x + dt1 * solid_n_y + t1 * dsolid_n_y
        normal_traction_jump = (
            fluid_normal_trac
            - phi_k * pore_normal_trac
            - solid_interface_traction_weight_c * one_m_phi_k * solid_normal_trac
        )
        dnormal_traction_jump = (
            dfluid_normal_trac
            - (dphi * pore_normal_trac + phi_k * dpore_normal_trac)
            + solid_interface_traction_weight_c * dphi * solid_normal_trac
            - solid_interface_traction_weight_c * one_m_phi_k * dsolid_normal_trac
        )
        tangential_traction_jump = (
            fluid_tang_trac
            - phi_k * pore_tang_trac
            - solid_interface_traction_weight_c * one_m_phi_k * solid_tang_trac
        )
        dtangential_traction_jump = (
            dfluid_tang_trac
            - (dphi * pore_tang_trac + phi_k * dpore_tang_trac)
            + solid_interface_traction_weight_c * dphi * solid_tang_trac
            - solid_interface_traction_weight_c * one_m_phi_k * dsolid_tang_trac
        )
        fluid_tangential_velocity_k = dot(tangent_vec, v_k)
        dfluid_tangential_velocity = dot(dtangent_vec, v_k) + dot(tangent_vec, dv)
        pore_tangential_velocity_k = dot(tangent_vec, vP_k)
        dpore_tangential_velocity = dot(dtangent_vec, vP_k) + dot(tangent_vec, dvP)
        solid_tangential_velocity_k = dot(tangent_vec, vS_k)
        dsolid_tangential_velocity = dot(dtangent_vec, vS_k) + dot(tangent_vec, dvS)
        bjs_mixture_tangential_velocity_k = (
            phi_k * pore_tangential_velocity_k + one_m_phi_k * solid_tangential_velocity_k
        )
        dbjs_mixture_tangential_velocity = (
            dphi * pore_tangential_velocity_k
            + phi_k * dpore_tangential_velocity
            - dphi * solid_tangential_velocity_k
            + one_m_phi_k * dsolid_tangential_velocity
        )
        bjs_beta_k = bjs_coeff_c * mu_f_c * sqrt_kappa_inv_c
        dbjs_beta = zero
        bjs_slip_t_k = fluid_tangential_velocity_k - bjs_mixture_tangential_velocity_k
        dbjs_slip_t = dfluid_tangential_velocity - dbjs_mixture_tangential_velocity
        bjs_tangential_jump = fluid_tang_trac - bjs_beta_k * bjs_slip_t_k
        dbjs_tangential_jump = (
            dfluid_tang_trac
            - (dbjs_beta * bjs_slip_t_k + bjs_beta_k * dbjs_slip_t)
        )
        fluid_tangential_velocity_test = dot(tangent_vec, v_test)

        def _active_tangential_interface_directional(
            *,
            dv_var=None,
            dvP_var=None,
            dvS_var=None,
            du_var=None,
            dphi_var=None,
            dalpha_var=None,
        ):
            if not use_bjs_tangential_law:
                return _traction_directional(
                    dv_var=dv_var,
                    dvP_var=dvP_var,
                    du_var=du_var,
                    dphi_var=dphi_var,
                    dalpha_var=dalpha_var,
                )[1]
            dv_loc = zero_v if dv_var is None else dv_var
            dvP_loc = zero_v if dvP_var is None else dvP_var
            dvS_loc = zero_v if dvS_var is None else dvS_var
            dphi_loc = zero if dphi_var is None else dphi_var
            dalpha_loc = zero if dalpha_var is None else dalpha_var
            dgrad_alpha_loc = grad(dalpha_loc)
            dgrad_alpha_mag_loc = dot(grad_alpha_k, dgrad_alpha_loc) * inv_grad_alpha_mag
            dnormal_loc = (dgrad_alpha_loc - normal_vec * dgrad_alpha_mag_loc) * inv_grad_alpha_mag
            dn0_loc = dnormal_loc[0]
            dn1_loc = dnormal_loc[1]
            dt0_loc = -dn1_loc
            dt1_loc = dn0_loc
            dsigma_f_loc = _c(2.0) * mu_f_c * _eps(dv_loc)
            dsigma_f_n_x_loc, dsigma_f_n_y_loc = _stress_times_dir_components(dsigma_f_loc, n0, n1)
            sigma_f_dn_x_loc, sigma_f_dn_y_loc = _stress_times_dir_components(sigma_f_k, dn0_loc, dn1_loc)
            dfluid_n_x_loc = dsigma_f_n_x_loc + sigma_f_dn_x_loc
            dfluid_n_y_loc = dsigma_f_n_y_loc + sigma_f_dn_y_loc
            dfluid_tang_trac_loc = (
                dt0_loc * fluid_n_x + t0 * dfluid_n_x_loc + dt1_loc * fluid_n_y + t1 * dfluid_n_y_loc
            )
            dfluid_tangential_velocity_loc = (
                dt0_loc * v_k[0] + t0 * dv_loc[0] + dt1_loc * v_k[1] + t1 * dv_loc[1]
            )
            dpore_tangential_velocity_loc = (
                dt0_loc * vP_k[0] + t0 * dvP_loc[0] + dt1_loc * vP_k[1] + t1 * dvP_loc[1]
            )
            dsolid_tangential_velocity_loc = (
                dt0_loc * vS_k[0] + t0 * dvS_loc[0] + dt1_loc * vS_k[1] + t1 * dvS_loc[1]
            )
            dbjs_mixture_tangential_velocity_loc = (
                dphi_loc * pore_tangential_velocity_k + phi_k * dpore_tangential_velocity_loc
                - dphi_loc * solid_tangential_velocity_k + one_m_phi_k * dsolid_tangential_velocity_loc
            )
            dbjs_beta_loc = zero
            dbjs_slip_t_loc = dfluid_tangential_velocity_loc - dbjs_mixture_tangential_velocity_loc
            dbjs_tangential_jump_loc = (
                dfluid_tang_trac_loc
                - (dbjs_beta_loc * bjs_slip_t_k + bjs_beta_k * dbjs_slip_t_loc)
            )
            return (
                dgrad_alpha_mag_loc * bjs_tangential_jump
                + grad_alpha_mag * dbjs_tangential_jump_loc
            )

    if rigid_darcy_head_mode:
        mass_test_bracket_v = rho_f_c * dot(grad(alpha_k), v_test)
        mass_test_bracket_vP = -(rho_s_phys_k * phi_k) * dot(grad(alpha_k), vP_test)
        mass_test_bracket_alpha = dot(grad(alpha_test), mass_jump_k)
        interface_mass_test_bracket = (
            mass_test_bracket_v
            + mass_test_bracket_vP
            + mass_test_bracket_alpha
        )

        r_interface_mass = mu_mass_test * (
            one_m_mass_interface_weight_c * mu_mass_k
            + mass_interface_weight_c * mass_constraint_k
        ) * dx
        a_interface_mass = mu_mass_test * (
            one_m_mass_interface_weight_c * dmu_mass
            + mass_interface_weight_c * dmass_constraint
        ) * dx
        r_interface_mass_bulk = (
            mass_interface_weight_c * mu_mass_k * interface_mass_test_bracket
        ) * dx
        a_interface_mass_bulk = (
            mass_interface_weight_c * dmu_mass * interface_mass_test_bracket
        ) * dx

        normal_test_bracket_v, tangential_test_bracket_v = _traction_directional(dv_var=v_test)
        interface_normal_test_bracket, interface_tangential_test_bracket = _traction_directional(
            dv_var=v_test,
            dp_var=q_test,
            dp_pore_var=q_pore_test,
        )
        normal_test_bracket_phi = zero
        normal_test_bracket_u = zero
        normal_test_bracket_alpha = zero
        tangential_test_bracket_u = zero
        tangential_test_bracket_alpha = zero

        rigid_normal_jump = fluid_normal_trac + normal_pressure_scale_c * phi_k * p_pore_k
        rigid_dnormal_jump = dfluid_normal_trac + normal_pressure_scale_c * (
            dphi * p_pore_k + phi_k * dp_pore
        )
        normal_constraint_k = grad_alpha_mag * rigid_normal_jump
        dnormal_constraint = dgrad_alpha_mag * rigid_normal_jump + grad_alpha_mag * rigid_dnormal_jump

        rigid_tangential_jump = fluid_tang_trac + bjs_coeff_c * fluid_tangential_velocity_k
        rigid_dtangential_jump = dfluid_tang_trac + bjs_coeff_c * dfluid_tangential_velocity
        tangential_constraint_k = grad_alpha_mag * rigid_tangential_jump
        dtangential_constraint = (
            dgrad_alpha_mag * rigid_tangential_jump
            + grad_alpha_mag * rigid_dtangential_jump
        )

        r_interface_normal = mu_normal_test * (
            one_m_normal_interface_weight_c * mu_normal_k
            + normal_interface_weight_c * normal_constraint_k
        ) * dx
        a_interface_normal = mu_normal_test * (
            one_m_normal_interface_weight_c * dmu_normal
            + normal_interface_weight_c * dnormal_constraint
        ) * dx
        r_interface_tangential = mu_tangent_test * tangential_constraint_k * dx
        a_interface_tangential = mu_tangent_test * dtangential_constraint * dx
        r_interface_normal_bulk = (
            normal_interface_weight_c * mu_normal_k * interface_normal_test_bracket
        ) * dx
        a_interface_normal_bulk = (
            normal_interface_weight_c * dmu_normal * interface_normal_test_bracket
        ) * dx
        r_interface_tangential_bulk = mu_tangent_k * interface_tangential_test_bracket * dx
        a_interface_tangential_bulk = dmu_tangent * interface_tangential_test_bracket * dx
    else:
        mass_test_bracket_v = rho_f_c * dot(grad(alpha_k), v_test)
        mass_test_bracket_vP = -(rho_s_phys_k * phi_k) * dot(grad(alpha_k), vP_test)
        mass_test_bracket_vS = -(rho_s_phys_k * one_m_phi_k) * dot(grad(alpha_k), vS_test)
        mass_test_bracket_phi = zero
        if phi_test is not None:
            mass_test_bracket_phi = -rho_s_phys_k * dot(grad(alpha_k), rel_p_k) * phi_test
        mass_test_bracket_rho_s = zero
        if (rho_s_test is not None) and (not constant_rho_s):
            solid_mass_flux_k = phi_k * vP_k + one_m_phi_k * vS_k
            mass_test_bracket_rho_s = -dot(grad(alpha_k), solid_mass_flux_k) * rho_s_test
        mass_test_bracket_alpha = dot(grad(alpha_test), mass_jump_k)

        interface_mass_test_bracket = (
            mass_test_bracket_v
            + mass_test_bracket_vP
            + mass_test_bracket_vS
            + mass_test_bracket_phi
            + mass_test_bracket_rho_s
            + mass_test_bracket_alpha
        )

        r_interface_mass = mu_mass_test * (
            one_m_mass_interface_weight_c * mu_mass_k
            + mass_interface_weight_c * mass_constraint_k
        ) * dx
        a_interface_mass = mu_mass_test * (
            one_m_mass_interface_weight_c * dmu_mass
            + mass_interface_weight_c * dmass_constraint
        ) * dx
        r_interface_mass_bulk = (
            mass_interface_weight_c * mu_mass_k * interface_mass_test_bracket
        ) * dx
        a_interface_mass_bulk = (
            mass_interface_weight_c * dmu_mass * interface_mass_test_bracket
        ) * dx

        normal_test_bracket_v = _traction_directional(dv_var=v_test)[0]
        tangential_test_bracket_v = _active_tangential_interface_directional(dv_var=v_test)
        normal_test_bracket_vP = _traction_directional(dvP_var=vP_test)[0]
        tangential_test_bracket_vP = _active_tangential_interface_directional(dvP_var=vP_test)
        normal_test_bracket_vS = -solid_interface_traction_weight_c * one_m_phi_k * grad_alpha_mag * dot(
            normal_vec, vS_test
        )
        tangential_test_bracket_vS = -solid_interface_traction_weight_c * one_m_phi_k * grad_alpha_mag * dot(
            tangent_vec, vS_test
        )
        if use_bjs_tangential_law:
            tangential_test_bracket_vS = _active_tangential_interface_directional(dvS_var=vS_test)
        normal_test_bracket_p, _ = _traction_directional(dp_var=q_test)
        normal_test_bracket_p_pore, _ = _traction_directional(dp_pore_var=q_pore_test)
        normal_test_bracket_pi_s = zero
        if solid_volumetric_split and pi_s_test is not None and total_pressure_ref_c is not None:
            normal_test_bracket_pi_s = (
                -solid_interface_traction_weight_c
                * one_m_phi_k
                * total_pressure_ref_c
                * grad_alpha_mag
                * pi_s_test
            )
        normal_test_bracket_phi = zero if phi_test is None else _traction_directional(dphi_var=phi_test)[0]
        normal_test_bracket_u = _traction_directional(du_var=u_test)[0]
        tangential_test_bracket_u = _active_tangential_interface_directional(du_var=u_test)
        normal_test_bracket_alpha = _traction_directional(dalpha_var=alpha_test)[0]
        tangential_test_bracket_alpha = _active_tangential_interface_directional(dalpha_var=alpha_test)
        interface_normal_test_bracket = _traction_directional(
            dv_var=v_test,
            dp_var=q_test,
            dvP_var=vP_test,
            dp_pore_var=q_pore_test,
            dphi_var=zero if phi_test is None else phi_test,
            du_var=u_test,
            dalpha_var=alpha_test,
        )[0]
        interface_tangential_test_bracket = _active_tangential_interface_directional(
            dv_var=v_test,
            dvP_var=vP_test,
            dvS_var=vS_test,
            dphi_var=zero if phi_test is None else phi_test,
            du_var=u_test,
            dalpha_var=alpha_test,
        )

        normal_constraint_k = grad_alpha_mag * normal_traction_jump
        tangential_constraint_k = (
            grad_alpha_mag * tangential_traction_jump
            if not use_bjs_tangential_law
            else grad_alpha_mag * bjs_tangential_jump
        )
        dnormal_constraint = dgrad_alpha_mag * normal_traction_jump + grad_alpha_mag * dnormal_traction_jump
        dtangential_constraint = (
            dgrad_alpha_mag * tangential_traction_jump + grad_alpha_mag * dtangential_traction_jump
            if not use_bjs_tangential_law
            else dgrad_alpha_mag * bjs_tangential_jump + grad_alpha_mag * dbjs_tangential_jump
        )

        if normal_carrier_key == "multiplier":
            # Keep the actual multiplier row live on the deformable branch.
            # The transpose couplings into (v, vP, u, p, p_pore, phi, alpha)
            # are added below block-by-block, but the mu_normal test row must
            # still enforce the normal jump itself. Otherwise the explicit
            # normal law survives only through the AL bulk penalty.
            r_interface_normal = mu_normal_test * (
                one_m_normal_interface_weight_c * mu_normal_k
                + normal_interface_weight_c * normal_constraint_k
            ) * dx
            a_interface_normal = mu_normal_test * (
                one_m_normal_interface_weight_c * dmu_normal
                + normal_interface_weight_c * dnormal_constraint
            ) * dx
        else:
            # When the normal jump is carried on p_pore, mu_normal is not part
            # of the physical interface law anymore. Keep that unused field
            # pinned to zero so the mixed system does not carry a null row.
            r_interface_normal = mu_normal_test * mu_normal_k * dx
            a_interface_normal = mu_normal_test * dmu_normal * dx
        r_interface_tangential = mu_tangent_test * tangential_constraint_k * dx
        a_interface_tangential = mu_tangent_test * dtangential_constraint * dx
        r_interface_normal_bulk = _safe_zero_scalar_linear_form(p_pore_k, q_pore_test)
        a_interface_normal_bulk = _safe_zero_scalar_linear_form(p_pore_k, q_pore_test)
        r_interface_tangential_bulk = mu_tangent_k * interface_tangential_test_bracket * dx
        a_interface_tangential_bulk = dmu_tangent * interface_tangential_test_bracket * dx
    if float(mass_lm_aug_gamma) != 0.0:
        r_interface_mass_aug_bulk = (
            mass_lm_aug_gamma_c
            * mass_interface_weight_c
            * mass_constraint_k
            * interface_mass_test_bracket
            * dx
        )
        a_interface_mass_aug_bulk = (
            mass_lm_aug_gamma_c
            * mass_interface_weight_c
            * dmass_constraint
            * interface_mass_test_bracket
            * dx
        )
    else:
        r_interface_mass_aug_bulk = _safe_zero_scalar_linear_form(p_k, q_test)
        a_interface_mass_aug_bulk = _safe_zero_scalar_linear_form(p_k, q_test)
    normal_aug_weight_c = normal_interface_weight_c
    if float(normal_lm_aug_gamma) != 0.0:
        # Diffuse-interface AL term on the normal traction jump. The extra
        # inv_grad_alpha factor avoids promoting the surface measure to a
        # squared-delta penalty; with lagged alpha in the mechanics solve this
        # behaves like gamma * grad_alpha_mag * jump * d(jump).
        r_interface_normal_aug_bulk = (
            normal_lm_aug_gamma_c
            * normal_aug_weight_c
            * normal_constraint_k
            * inv_grad_alpha_mag
            * interface_normal_test_bracket
            * dx
        )
        a_interface_normal_aug_bulk = (
            normal_lm_aug_gamma_c
            * normal_aug_weight_c
            * dnormal_constraint
            * inv_grad_alpha_mag
            * interface_normal_test_bracket
            * dx
        )
    else:
        r_interface_normal_aug_bulk = _safe_zero_scalar_linear_form(p_pore_k, q_pore_test)
        a_interface_normal_aug_bulk = _safe_zero_scalar_linear_form(p_pore_k, q_pore_test)

    if disable_interface_physics:
        zero_q_form = _safe_zero_scalar_linear_form(p_k, q_test)
        zero_u_form = _safe_zero_vector_linear_form(u_k, u_test)
        zero_vS_form = _safe_zero_vector_linear_form(vS_k, vS_test)
        zero_alpha_form = _safe_zero_scalar_linear_form(alpha_k, alpha_test)

        r_interface_mass = mu_mass_test * mu_mass_k * dx
        a_interface_mass = mu_mass_test * dmu_mass * dx
        r_interface_normal = mu_normal_test * mu_normal_k * dx
        a_interface_normal = mu_normal_test * dmu_normal * dx
        r_interface_tangential = mu_tangent_test * mu_tangent_k * dx
        a_interface_tangential = mu_tangent_test * dmu_tangent * dx
        r_interface_mass_bulk = zero_q_form
        a_interface_mass_bulk = zero_q_form
        r_interface_mass_aug_bulk = zero_q_form
        a_interface_mass_aug_bulk = zero_q_form
        r_interface_normal_bulk = zero_q_form
        a_interface_normal_bulk = zero_q_form
        r_interface_tangential_bulk = zero_q_form
        a_interface_tangential_bulk = zero_q_form
        if mu_kin_k is not None and mu_kin_test is not None and dmu_kin is not None:
            r_interface_kin = dot(mu_kin_test, mu_kin_k) * dx
            a_interface_kin = dot(mu_kin_test, dmu_kin) * dx
        else:
            r_interface_kin = zero_u_form
            a_interface_kin = zero_u_form
        r_interface_kin_bulk_u = zero_u_form
        a_interface_kin_bulk_u = zero_u_form
        r_interface_kin_bulk_vS = zero_vS_form
        a_interface_kin_bulk_vS = zero_vS_form
        r_interface_kin_bulk_alpha = zero_alpha_form
        a_interface_kin_bulk_alpha = zero_alpha_form
    else:
        r_mom_f = r_mom_f + mass_interface_weight_c * mu_mass_k * (rho_f_c * dot(grad(alpha_k), v_test)) * dx
        a_mom_f = a_mom_f + mass_interface_weight_c * dmu_mass * (rho_f_c * dot(grad(alpha_k), v_test)) * dx
        if float(mass_lm_aug_gamma) != 0.0:
            r_mom_f = r_mom_f + mass_lm_aug_gamma_c * mass_interface_weight_c * mass_constraint_k * mass_test_bracket_v * dx
            a_mom_f = a_mom_f + mass_lm_aug_gamma_c * mass_interface_weight_c * dmass_constraint * mass_test_bracket_v * dx
        if normal_carrier_key == "multiplier":
            if not rigid_darcy_head_mode:
                r_interface_normal = mu_normal_test * (
                    one_m_normal_interface_weight_c * mu_normal_k
                    + normal_interface_weight_c * normal_constraint_k
                ) * dx
                a_interface_normal = mu_normal_test * (
                    one_m_normal_interface_weight_c * dmu_normal
                    + normal_interface_weight_c * dnormal_constraint
                ) * dx
                r_interface_normal_bulk = (
                    normal_interface_weight_c * mu_normal_k * interface_normal_test_bracket
                ) * dx
                a_interface_normal_bulk = (
                    normal_interface_weight_c * dmu_normal * interface_normal_test_bracket
                ) * dx
            r_mom_f = r_mom_f + normal_interface_weight_c * mu_normal_k * normal_test_bracket_v * dx
            a_mom_f = a_mom_f + normal_interface_weight_c * dmu_normal * normal_test_bracket_v * dx
        else:
            if normal_interface_weight_val != 1.0:
                raise ValueError(
                    "normal_interface_weight != 1 currently requires normal_constraint_carrier='multiplier'."
                )
            if not disable_normal_interface:
                r_pore = q_pore_test * normal_constraint_k * dx
                a_pore = q_pore_test * dnormal_constraint * dx
                r_interface_normal_bulk = p_pore_k * interface_normal_test_bracket * dx
                a_interface_normal_bulk = dp_pore * interface_normal_test_bracket * dx
                r_mom_f = r_mom_f + p_pore_k * normal_test_bracket_v * dx
                a_mom_f = a_mom_f + dp_pore * normal_test_bracket_v * dx
        if float(normal_lm_aug_gamma) != 0.0:
            r_mom_f = r_mom_f + (
                normal_lm_aug_gamma_c
                * normal_aug_weight_c
                * normal_constraint_k
                * inv_grad_alpha_mag
                * normal_test_bracket_v
                * dx
            )
            a_mom_f = a_mom_f + (
                normal_lm_aug_gamma_c
                * normal_aug_weight_c
                * dnormal_constraint
                * inv_grad_alpha_mag
                * normal_test_bracket_v
                * dx
            )
        r_mom_f = r_mom_f + mu_tangent_k * tangential_test_bracket_v * dx
        a_mom_f = a_mom_f + dmu_tangent * tangential_test_bracket_v * dx

        r_mom_p = r_mom_p + mass_interface_weight_c * mu_mass_k * (-(rho_s_phys_k * phi_k) * dot(grad(alpha_k), vP_test)) * dx
        a_mom_p = a_mom_p + mass_interface_weight_c * dmu_mass * (-(rho_s_phys_k * phi_k) * dot(grad(alpha_k), vP_test)) * dx
        if float(mass_lm_aug_gamma) != 0.0:
            r_mom_p = r_mom_p + mass_lm_aug_gamma_c * mass_interface_weight_c * mass_constraint_k * mass_test_bracket_vP * dx
            a_mom_p = a_mom_p + mass_lm_aug_gamma_c * mass_interface_weight_c * dmass_constraint * mass_test_bracket_vP * dx
        if not rigid_darcy_head_mode:
            if normal_carrier_key == "multiplier":
                r_mom_p = r_mom_p + normal_interface_weight_c * mu_normal_k * normal_test_bracket_vP * dx
                a_mom_p = a_mom_p + normal_interface_weight_c * dmu_normal * normal_test_bracket_vP * dx
                if float(normal_lm_aug_gamma) != 0.0:
                    r_mom_p = r_mom_p + (
                        normal_lm_aug_gamma_c
                        * normal_aug_weight_c
                        * normal_constraint_k
                        * inv_grad_alpha_mag
                        * normal_test_bracket_vP
                        * dx
                    )
                    a_mom_p = a_mom_p + (
                        normal_lm_aug_gamma_c
                        * normal_aug_weight_c
                        * dnormal_constraint
                        * inv_grad_alpha_mag
                        * normal_test_bracket_vP
                        * dx
                    )
            elif not disable_normal_interface:
                r_mom_p = r_mom_p + p_pore_k * normal_test_bracket_vP * dx
                a_mom_p = a_mom_p + dp_pore * normal_test_bracket_vP * dx
            r_mom_p = r_mom_p + mu_tangent_k * tangential_test_bracket_vP * dx
            a_mom_p = a_mom_p + dmu_tangent * tangential_test_bracket_vP * dx

        if not rigid_darcy_head_mode:
            r_skel = r_skel + mass_interface_weight_c * mu_mass_k * (-(rho_s_phys_k * one_m_phi_k) * dot(grad(alpha_k), vS_test)) * dx
            a_skel = a_skel + mass_interface_weight_c * dmu_mass * (-(rho_s_phys_k * one_m_phi_k) * dot(grad(alpha_k), vS_test)) * dx
            if float(mass_lm_aug_gamma) != 0.0:
                r_skel = r_skel + mass_lm_aug_gamma_c * mass_interface_weight_c * mass_constraint_k * mass_test_bracket_vS * dx
                a_skel = a_skel + mass_lm_aug_gamma_c * mass_interface_weight_c * dmass_constraint * mass_test_bracket_vS * dx
            if normal_carrier_key == "multiplier":
                r_skel = r_skel + normal_interface_weight_c * mu_normal_k * normal_test_bracket_vS * dx
                a_skel = a_skel + normal_interface_weight_c * dmu_normal * normal_test_bracket_vS * dx
                if float(normal_lm_aug_gamma) != 0.0:
                    r_skel = r_skel + (
                        normal_lm_aug_gamma_c
                        * normal_aug_weight_c
                        * normal_constraint_k
                        * inv_grad_alpha_mag
                        * normal_test_bracket_vS
                        * dx
                    )
                    a_skel = a_skel + (
                        normal_lm_aug_gamma_c
                        * normal_aug_weight_c
                        * dnormal_constraint
                        * inv_grad_alpha_mag
                        * normal_test_bracket_vS
                        * dx
                    )
            if normal_carrier_key == "multiplier" and pi_s_test is not None:
                r_skel = r_skel + normal_interface_weight_c * mu_normal_k * normal_test_bracket_pi_s * dx
                a_skel = a_skel + normal_interface_weight_c * dmu_normal * normal_test_bracket_pi_s * dx
                if float(normal_lm_aug_gamma) != 0.0:
                    r_skel = r_skel + (
                        normal_lm_aug_gamma_c
                        * normal_aug_weight_c
                        * normal_constraint_k
                        * inv_grad_alpha_mag
                        * normal_test_bracket_pi_s
                        * dx
                    )
                    a_skel = a_skel + (
                        normal_lm_aug_gamma_c
                        * normal_aug_weight_c
                        * dnormal_constraint
                        * inv_grad_alpha_mag
                        * normal_test_bracket_pi_s
                        * dx
                    )
            elif not disable_normal_interface:
                r_skel = r_skel + p_pore_k * normal_test_bracket_vS * dx
                a_skel = a_skel + dp_pore * normal_test_bracket_vS * dx
            r_skel = r_skel + mu_tangent_k * tangential_test_bracket_vS * dx
            a_skel = a_skel + dmu_tangent * tangential_test_bracket_vS * dx

        if rigid_darcy_head_mode:
            pore_bulk_flux_k = alpha_bulk_k * (phi_k * div(vP_k) + dot(grad(phi_k), vP_k))
            dpore_bulk_flux = (
                dalpha_bulk * (phi_k * div(vP_k) + dot(grad(phi_k), vP_k))
                + alpha_bulk_k * (dphi * div(vP_k) + phi_k * div(dvP) + dot(grad(dphi), vP_k) + dot(grad(phi_k), dvP))
            )
            r_pore = q_pore_test * (pore_bulk_flux_k - rho_f_c * dot(grad(alpha_k), v_k)) * dx
            a_pore = q_pore_test * (dpore_bulk_flux - rho_f_c * (dot(grad(dalpha), v_k) + dot(grad(alpha_k), dv))) * dx
            if normal_carrier_key == "multiplier":
                r_mass = r_mass + normal_interface_weight_c * mu_normal_k * normal_test_bracket_p * dx
                a_mass = a_mass + normal_interface_weight_c * dmu_normal * normal_test_bracket_p * dx
                if float(normal_lm_aug_gamma) != 0.0:
                    r_mass = r_mass + (
                        normal_lm_aug_gamma_c
                        * normal_aug_weight_c
                        * normal_constraint_k
                        * inv_grad_alpha_mag
                        * normal_test_bracket_p
                        * dx
                    )
                    a_mass = a_mass + (
                        normal_lm_aug_gamma_c
                        * normal_aug_weight_c
                        * dnormal_constraint
                        * inv_grad_alpha_mag
                        * normal_test_bracket_p
                        * dx
                    )
                r_pore = _form_add(
                    r_pore,
                    normal_interface_weight_c * mu_normal_k * normal_test_bracket_p_pore * dx,
                )
                a_pore = _form_add(
                    a_pore,
                    normal_interface_weight_c * dmu_normal * normal_test_bracket_p_pore * dx,
                )
        elif normal_carrier_key == "multiplier":
            r_kin = r_kin + normal_interface_weight_c * mu_normal_k * normal_test_bracket_u * dx + mu_tangent_k * tangential_test_bracket_u * dx
            a_kin = a_kin + normal_interface_weight_c * dmu_normal * normal_test_bracket_u * dx + dmu_tangent * tangential_test_bracket_u * dx
            if float(normal_lm_aug_gamma) != 0.0:
                r_kin = r_kin + (
                    normal_lm_aug_gamma_c
                    * normal_aug_weight_c
                    * normal_constraint_k
                    * inv_grad_alpha_mag
                    * normal_test_bracket_u
                    * dx
                )
                a_kin = a_kin + (
                    normal_lm_aug_gamma_c
                    * normal_aug_weight_c
                    * dnormal_constraint
                    * inv_grad_alpha_mag
                    * normal_test_bracket_u
                    * dx
                )
            r_mass = r_mass + normal_interface_weight_c * mu_normal_k * normal_test_bracket_p * dx
            a_mass = a_mass + normal_interface_weight_c * dmu_normal * normal_test_bracket_p * dx
            if float(normal_lm_aug_gamma) != 0.0:
                r_mass = r_mass + (
                    normal_lm_aug_gamma_c
                    * normal_aug_weight_c
                    * normal_constraint_k
                    * inv_grad_alpha_mag
                    * normal_test_bracket_p
                    * dx
                )
                a_mass = a_mass + (
                    normal_lm_aug_gamma_c
                    * normal_aug_weight_c
                    * dnormal_constraint
                    * inv_grad_alpha_mag
                    * normal_test_bracket_p
                    * dx
                )
            r_pore = _form_add(
                r_pore,
                normal_interface_weight_c * mu_normal_k * normal_test_bracket_p_pore * dx,
            )
            a_pore = _form_add(
                a_pore,
                normal_interface_weight_c * dmu_normal * normal_test_bracket_p_pore * dx,
            )
            if float(normal_lm_aug_gamma) != 0.0:
                r_pore = _form_add(
                    r_pore,
                    normal_lm_aug_gamma_c
                    * normal_aug_weight_c
                    * normal_constraint_k
                    * inv_grad_alpha_mag
                    * normal_test_bracket_p_pore
                    * dx,
                )
                a_pore = _form_add(
                    a_pore,
                    normal_lm_aug_gamma_c
                    * normal_aug_weight_c
                    * dnormal_constraint
                    * inv_grad_alpha_mag
                    * normal_test_bracket_p_pore
                    * dx,
                )
        else:
            if not disable_normal_interface:
                r_kin = r_kin + p_pore_k * normal_test_bracket_u * dx + mu_tangent_k * tangential_test_bracket_u * dx
                a_kin = a_kin + dp_pore * normal_test_bracket_u * dx + dmu_tangent * tangential_test_bracket_u * dx
                r_mass = r_mass + p_pore_k * normal_test_bracket_p * dx
                a_mass = a_mass + dp_pore * normal_test_bracket_p * dx
            else:
                r_kin = r_kin + mu_tangent_k * tangential_test_bracket_u * dx
                a_kin = a_kin + dmu_tangent * tangential_test_bracket_u * dx

    if float(gamma_p_pore) != 0.0:
        if p_pore_ext_mode in {"l2", "mass"}:
            r_pore = _form_add(r_pore, gamma_p_pore_c * inv_h2_ext * F_bulk_k * p_pore_k * q_pore_test * dx)
            a_pore = _form_add(a_pore, gamma_p_pore_c * inv_h2_ext * (
                dF_bulk * p_pore_k + F_bulk_k * dp_pore
            ) * q_pore_test * dx)
        else:
            r_pore = _form_add(r_pore, gamma_p_pore_c * F_bulk_k * dot(grad(p_pore_k), grad(q_pore_test)) * dx)
            a_pore = _form_add(a_pore, gamma_p_pore_c * (
                dF_bulk * dot(grad(p_pore_k), grad(q_pore_test))
                + F_bulk_k * dot(grad(dp_pore), grad(q_pore_test))
            ) * dx)
            if float(gamma_p_pore_pin) != 0.0:
                w_pin2 = F_bulk_k * F_bulk_k
                dw_pin2 = _c(2.0) * F_bulk_k * dF_bulk
                r_pore = _form_add(r_pore, gamma_p_pore_pin_c * inv_h2_ext * w_pin2 * p_pore_k * q_pore_test * dx)
                a_pore = _form_add(a_pore, gamma_p_pore_pin_c * inv_h2_ext * (
                    dw_pin2 * p_pore_k + w_pin2 * dp_pore
                ) * q_pore_test * dx)

    # Preserve the uncoupled alpha/phi transport rows for the post_accept
    # transport stage before interface-law bulk couplings are appended below.
    r_phi_transport = r_phi
    a_phi_transport = a_phi
    r_alpha_transport = r_alpha
    a_alpha_transport = a_alpha

    if not disable_interface_physics:
        if phi_test is not None and not rigid_darcy_head_mode:
            r_phi = _form_add(r_phi, mass_interface_weight_c * mu_mass_k * (-rho_s_phys_k * dot(grad(alpha_k), rel_p_k) * phi_test) * dx)
            a_phi = _form_add(a_phi, mass_interface_weight_c * dmu_mass * (-rho_s_phys_k * dot(grad(alpha_k), rel_p_k) * phi_test) * dx)
            if float(mass_lm_aug_gamma) != 0.0:
                r_phi = _form_add(
                    r_phi,
                    mass_lm_aug_gamma_c * mass_interface_weight_c * mass_constraint_k * mass_test_bracket_phi * dx,
                )
                a_phi = _form_add(
                    a_phi,
                    mass_lm_aug_gamma_c * mass_interface_weight_c * dmass_constraint * mass_test_bracket_phi * dx,
                )
            if normal_carrier_key == "multiplier":
                r_phi = _form_add(r_phi, normal_interface_weight_c * mu_normal_k * normal_test_bracket_phi * dx)
                a_phi = _form_add(a_phi, normal_interface_weight_c * dmu_normal * normal_test_bracket_phi * dx)
                if float(normal_lm_aug_gamma) != 0.0:
                    r_phi = _form_add(
                        r_phi,
                        normal_lm_aug_gamma_c
                        * normal_aug_weight_c
                        * normal_constraint_k
                        * inv_grad_alpha_mag
                        * normal_test_bracket_phi
                        * dx,
                    )
                    a_phi = _form_add(
                        a_phi,
                        normal_lm_aug_gamma_c
                        * normal_aug_weight_c
                        * dnormal_constraint
                        * inv_grad_alpha_mag
                        * normal_test_bracket_phi
                        * dx,
                    )
            elif not disable_normal_interface:
                r_phi = _form_add(r_phi, p_pore_k * normal_test_bracket_phi * dx)
                a_phi = _form_add(a_phi, dp_pore * normal_test_bracket_phi * dx)
        if (rho_s_test is not None) and (not constant_rho_s) and (not rigid_darcy_head_mode):
            solid_mass_flux_k = phi_k * vP_k + one_m_phi_k * vS_k
            r_phi = _form_add(r_phi, mass_interface_weight_c * mu_mass_k * (-dot(grad(alpha_k), solid_mass_flux_k) * rho_s_test) * dx)
            a_phi = _form_add(a_phi, mass_interface_weight_c * dmu_mass * (-dot(grad(alpha_k), solid_mass_flux_k) * rho_s_test) * dx)
            if float(mass_lm_aug_gamma) != 0.0:
                r_phi = _form_add(
                    r_phi,
                    mass_lm_aug_gamma_c * mass_interface_weight_c * mass_constraint_k * mass_test_bracket_rho_s * dx,
                )
                a_phi = _form_add(
                    a_phi,
                    mass_lm_aug_gamma_c * mass_interface_weight_c * dmass_constraint * mass_test_bracket_rho_s * dx,
                )
        if not rigid_darcy_head_mode:
            r_alpha = r_alpha + mass_interface_weight_c * mu_mass_k * dot(grad(alpha_test), mass_jump_k) * dx
            a_alpha = a_alpha + mass_interface_weight_c * dmu_mass * dot(grad(alpha_test), mass_jump_k) * dx
            if float(mass_lm_aug_gamma) != 0.0:
                r_alpha = r_alpha + mass_lm_aug_gamma_c * mass_interface_weight_c * mass_constraint_k * mass_test_bracket_alpha * dx
                a_alpha = a_alpha + mass_lm_aug_gamma_c * mass_interface_weight_c * dmass_constraint * mass_test_bracket_alpha * dx
        if rigid_darcy_head_mode:
            r_alpha = r_alpha + mass_interface_weight_c * mu_mass_k * dot(grad(alpha_test), mass_jump_k) * dx
            a_alpha = a_alpha + mass_interface_weight_c * dmu_mass * dot(grad(alpha_test), mass_jump_k) * dx
            if float(mass_lm_aug_gamma) != 0.0:
                r_alpha = r_alpha + mass_lm_aug_gamma_c * mass_interface_weight_c * mass_constraint_k * mass_test_bracket_alpha * dx
                a_alpha = a_alpha + mass_lm_aug_gamma_c * mass_interface_weight_c * dmass_constraint * mass_test_bracket_alpha * dx
        elif normal_carrier_key == "multiplier":
            r_alpha = r_alpha + normal_interface_weight_c * mu_normal_k * normal_test_bracket_alpha * dx + mu_tangent_k * tangential_test_bracket_alpha * dx
            a_alpha = a_alpha + normal_interface_weight_c * dmu_normal * normal_test_bracket_alpha * dx + dmu_tangent * tangential_test_bracket_alpha * dx
            if float(normal_lm_aug_gamma) != 0.0:
                r_alpha = r_alpha + (
                    normal_lm_aug_gamma_c
                    * normal_aug_weight_c
                    * normal_constraint_k
                    * inv_grad_alpha_mag
                    * normal_test_bracket_alpha
                    * dx
                )
                a_alpha = a_alpha + (
                    normal_lm_aug_gamma_c
                    * normal_aug_weight_c
                    * dnormal_constraint
                    * inv_grad_alpha_mag
                    * normal_test_bracket_alpha
                    * dx
                )
        else:
            if not disable_normal_interface:
                r_alpha = r_alpha + p_pore_k * normal_test_bracket_alpha * dx + mu_tangent_k * tangential_test_bracket_alpha * dx
                a_alpha = a_alpha + dp_pore * normal_test_bracket_alpha * dx + dmu_tangent * tangential_test_bracket_alpha * dx
            else:
                r_alpha = r_alpha + mu_tangent_k * tangential_test_bracket_alpha * dx
                a_alpha = a_alpha + dmu_tangent * tangential_test_bracket_alpha * dx

        r_kin = r_kin + r_interface_kin_bulk_u
        a_kin = a_kin + a_interface_kin_bulk_u
        r_skel = r_skel + r_interface_kin_bulk_vS
        a_skel = a_skel + a_interface_kin_bulk_vS
        r_alpha = r_alpha + r_interface_kin_bulk_alpha
        a_alpha = a_alpha + a_interface_kin_bulk_alpha

    r_domain_lm = None
    a_domain_lm = None
    r_domain_lm_terms: dict[str, object] | None = None
    a_domain_lm_terms: dict[str, object] | None = None
    if use_domain_lm:
        r_domain_lm = _safe_zero_scalar_linear_form(p_k, q_test)
        a_domain_lm = _safe_zero_scalar_linear_form(p_k, q_test)
        r_domain_lm_terms = {}
        a_domain_lm_terms = {}

        def _domain_lm_scalar(
            *,
            target_k,
            dtarget,
            target_test,
            lm_k,
            dlm,
            lm_test,
            active_weight,
            dactive_weight,
            target_shift=None,
            dtarget_shift=None,
            shift_test=None,
        ):
            shift_k = zero if target_shift is None else target_shift
            dshift = zero if dtarget_shift is None else dtarget_shift
            target_residual = target_k - shift_k
            dtarget_residual = dtarget - dshift
            r_row = lm_test * (active_weight * target_residual) * dx
            a_row = lm_test * (
                dactive_weight * target_residual
                + active_weight * dtarget_residual
            ) * dx
            r_bulk = active_weight * lm_k * target_test * dx
            a_bulk = (
                dactive_weight * lm_k * target_test
                + active_weight * dlm * target_test
            ) * dx
            if shift_test is None:
                r_shift = _safe_zero_scalar_linear_form(target_k, target_test)
                a_shift = _safe_zero_scalar_linear_form(target_k, target_test)
            else:
                r_shift = -(active_weight * lm_k * shift_test) * dx
                a_shift = -(
                    dactive_weight * lm_k * shift_test
                    + active_weight * dlm * shift_test
                ) * dx
            active_constraint = active_weight * target_residual
            dactive_constraint = dactive_weight * target_residual + active_weight * dtarget_residual
            r_aug = domain_lm_aug_gamma_c * active_constraint * active_weight * target_test * dx
            a_aug = domain_lm_aug_gamma_c * (
                dactive_constraint * active_weight * target_test
                + active_constraint * dactive_weight * target_test
            ) * dx
            if shift_test is None:
                r_aug_shift = _safe_zero_scalar_linear_form(target_k, target_test)
                a_aug_shift = _safe_zero_scalar_linear_form(target_k, target_test)
            else:
                r_aug_shift = -(domain_lm_aug_gamma_c * active_constraint * active_weight * shift_test) * dx
                a_aug_shift = -(
                    domain_lm_aug_gamma_c
                    * (
                        dactive_constraint * active_weight * shift_test
                        + active_constraint * dactive_weight * shift_test
                    )
                ) * dx
            return r_row, a_row, r_bulk, a_bulk, r_shift, a_shift, r_aug, a_aug, r_aug_shift, a_aug_shift

        def _domain_lm_vector(
            *,
            target_k,
            dtarget,
            target_test,
            lm_k,
            dlm,
            lm_test,
            active_weight,
            dactive_weight,
            target_shift=None,
            dtarget_shift=None,
            shift_test=None,
        ):
            shift_k = zero_v if target_shift is None else target_shift
            dshift = zero_v if dtarget_shift is None else dtarget_shift
            target_residual = target_k - shift_k
            dtarget_residual = dtarget - dshift
            r_row = dot(lm_test, active_weight * target_residual) * dx
            a_row = dot(
                lm_test,
                dactive_weight * target_residual
                + active_weight * dtarget_residual
            ) * dx
            r_bulk = active_weight * dot(lm_k, target_test) * dx
            a_bulk = (
                dactive_weight * dot(lm_k, target_test)
                + active_weight * dot(dlm, target_test)
            ) * dx
            if shift_test is None:
                r_shift = zero * dot(target_k, target_test) * dx
                a_shift = zero * dot(target_k, target_test) * dx
            else:
                r_shift = -(active_weight * dot(lm_k, shift_test)) * dx
                a_shift = -(
                    dactive_weight * dot(lm_k, shift_test)
                    + active_weight * dot(dlm, shift_test)
                ) * dx
            active_constraint = active_weight * target_residual
            dactive_constraint = dactive_weight * target_residual + active_weight * dtarget_residual
            r_aug = domain_lm_aug_gamma_c * active_weight * dot(active_constraint, target_test) * dx
            a_aug = domain_lm_aug_gamma_c * (
                active_weight * dot(dactive_constraint, target_test)
                + dactive_weight * dot(active_constraint, target_test)
            ) * dx
            if shift_test is None:
                r_aug_shift = zero * dot(target_k, target_test) * dx
                a_aug_shift = zero * dot(target_k, target_test) * dx
            else:
                r_aug_shift = -(domain_lm_aug_gamma_c * active_weight * dot(active_constraint, shift_test)) * dx
                a_aug_shift = -(
                    domain_lm_aug_gamma_c
                    * (
                        active_weight * dot(dactive_constraint, shift_test)
                        + dactive_weight * dot(active_constraint, shift_test)
                    )
                ) * dx
            return r_row, a_row, r_bulk, a_bulk, r_shift, a_shift, r_aug, a_aug, r_aug_shift, a_aug_shift

        def _record_domain_lm(name: str, r_row, a_row, r_bulk, a_bulk, r_aug, a_aug) -> None:
            nonlocal r_domain_lm, a_domain_lm
            r_domain_lm = _form_add(r_domain_lm, r_row)
            a_domain_lm = _form_add(a_domain_lm, a_row)
            assert r_domain_lm_terms is not None
            assert a_domain_lm_terms is not None
            r_domain_lm_terms[f"{name}_row"] = r_row
            r_domain_lm_terms[f"{name}_bulk"] = r_bulk
            r_domain_lm_terms[f"{name}_aug"] = r_aug
            a_domain_lm_terms[f"{name}_row"] = a_row
            a_domain_lm_terms[f"{name}_bulk"] = a_bulk
            a_domain_lm_terms[f"{name}_aug"] = a_aug

        if lm_vf_k is not None and dlm_vf is not None and lm_vf_test is not None:
            r_row, a_row, r_bulk, a_bulk, _r_shift, _a_shift, r_aug, a_aug, _r_aug_shift, _a_aug_shift = _domain_lm_vector(
                target_k=v_k,
                dtarget=dv,
                target_test=v_test,
                lm_k=lm_vf_k,
                dlm=dlm_vf,
                lm_test=lm_vf_test,
                active_weight=alpha_domain_lm_k,
                dactive_weight=dalpha_domain_lm,
            )
            _record_domain_lm("support_kill_vf", r_row, a_row, r_bulk, a_bulk, r_aug, a_aug)
            r_mom_f = r_mom_f + r_bulk + r_aug
            a_mom_f = a_mom_f + a_bulk + a_aug

        if lm_p_k is not None and dlm_p is not None and lm_p_test is not None:
            r_row, a_row, r_bulk, a_bulk, _r_shift, _a_shift, r_aug, a_aug, _r_aug_shift, _a_aug_shift = _domain_lm_scalar(
                target_k=p_k,
                dtarget=dp,
                target_test=q_test,
                lm_k=lm_p_k,
                dlm=dlm_p,
                lm_test=lm_p_test,
                active_weight=alpha_domain_lm_k,
                dactive_weight=dalpha_domain_lm,
            )
            _record_domain_lm("support_kill_p", r_row, a_row, r_bulk, a_bulk, r_aug, a_aug)
            r_mass = r_mass + r_bulk + r_aug
            a_mass = a_mass + a_bulk + a_aug

        if lm_vP_k is not None and dlm_vP is not None and lm_vP_test is not None:
            vP_shift_k = v_k if bool(domain_lm_free_tie_vP_to_vf) else None
            dvP_shift = dv if bool(domain_lm_free_tie_vP_to_vf) else None
            vP_shift_test = v_test if bool(domain_lm_free_tie_vP_to_vf) else None
            (
                r_row,
                a_row,
                r_bulk,
                a_bulk,
                r_shift,
                a_shift,
                r_aug,
                a_aug,
                r_aug_shift,
                a_aug_shift,
            ) = _domain_lm_vector(
                target_k=vP_k,
                dtarget=dvP,
                target_test=vP_test,
                lm_k=lm_vP_k,
                dlm=dlm_vP,
                lm_test=lm_vP_test,
                active_weight=F_domain_lm_k,
                dactive_weight=dF_domain_lm,
                target_shift=vP_shift_k,
                dtarget_shift=dvP_shift,
                shift_test=vP_shift_test,
            )
            _record_domain_lm("free_kill_vP", r_row, a_row, r_bulk, a_bulk, r_aug, a_aug)
            r_mom_p = r_mom_p + r_bulk + r_aug
            a_mom_p = a_mom_p + a_bulk + a_aug
            r_mom_f = r_mom_f + r_shift + r_aug_shift
            a_mom_f = a_mom_f + a_shift + a_aug_shift

        if lm_vS_k is not None and dlm_vS is not None and lm_vS_test is not None:
            r_row, a_row, r_bulk, a_bulk, _r_shift, _a_shift, r_aug, a_aug, _r_aug_shift, _a_aug_shift = _domain_lm_vector(
                target_k=vS_k,
                dtarget=dvS,
                target_test=vS_test,
                lm_k=lm_vS_k,
                dlm=dlm_vS,
                lm_test=lm_vS_test,
                active_weight=F_domain_lm_k,
                dactive_weight=dF_domain_lm,
            )
            _record_domain_lm("free_kill_vS", r_row, a_row, r_bulk, a_bulk, r_aug, a_aug)
            r_skel = r_skel + r_bulk + r_aug
            a_skel = a_skel + a_bulk + a_aug

        if lm_p_pore_k is not None and dlm_p_pore is not None and lm_p_pore_test is not None:
            p_pore_shift_k = p_k if bool(domain_lm_free_tie_p_pore_to_p) else None
            dp_pore_shift = dp if bool(domain_lm_free_tie_p_pore_to_p) else None
            p_pore_shift_test = q_test if bool(domain_lm_free_tie_p_pore_to_p) else None
            (
                r_row,
                a_row,
                r_bulk,
                a_bulk,
                r_shift,
                a_shift,
                r_aug,
                a_aug,
                r_aug_shift,
                a_aug_shift,
            ) = _domain_lm_scalar(
                target_k=p_pore_k,
                dtarget=dp_pore,
                target_test=q_pore_test,
                lm_k=lm_p_pore_k,
                dlm=dlm_p_pore,
                lm_test=lm_p_pore_test,
                active_weight=F_domain_lm_k,
                dactive_weight=dF_domain_lm,
                target_shift=p_pore_shift_k,
                dtarget_shift=dp_pore_shift,
                shift_test=p_pore_shift_test,
            )
            _record_domain_lm("free_kill_p_pore", r_row, a_row, r_bulk, a_bulk, r_aug, a_aug)
            r_pore = _form_add(r_pore, _form_add(r_bulk, r_aug))
            a_pore = _form_add(a_pore, _form_add(a_bulk, a_aug))
            r_mass = _form_add(r_mass, _form_add(r_shift, r_aug_shift))
            a_mass = _form_add(a_mass, _form_add(a_shift, a_aug_shift))

        if lm_phi_k is not None and dlm_phi is not None and lm_phi_test is not None and phi_test is not None:
            r_row, a_row, r_bulk, a_bulk, _r_shift, _a_shift, r_aug, a_aug, _r_aug_shift, _a_aug_shift = _domain_lm_scalar(
                target_k=phi_k,
                dtarget=dphi,
                target_test=phi_test,
                lm_k=lm_phi_k,
                dlm=dlm_phi,
                lm_test=lm_phi_test,
                active_weight=F_domain_lm_k,
                dactive_weight=dF_domain_lm,
                target_shift=_c(1.0),
            )
            _record_domain_lm("free_kill_phi", r_row, a_row, r_bulk, a_bulk, r_aug, a_aug)
            r_phi = _form_add(r_phi, _form_add(r_bulk, r_aug))
            a_phi = _form_add(a_phi, _form_add(a_bulk, a_aug))

        if lm_u_k is not None and dlm_u is not None and lm_u_test is not None:
            r_row, a_row, r_bulk, a_bulk, _r_shift, _a_shift, r_aug, a_aug, _r_aug_shift, _a_aug_shift = _domain_lm_vector(
                target_k=u_k,
                dtarget=du,
                target_test=u_test,
                lm_k=lm_u_k,
                dlm=dlm_u,
                lm_test=lm_u_test,
                active_weight=F_domain_lm_k,
                dactive_weight=dF_domain_lm,
            )
            _record_domain_lm("free_kill_u", r_row, a_row, r_bulk, a_bulk, r_aug, a_aug)
            r_kin = r_kin + r_bulk + r_aug
            a_kin = a_kin + a_bulk + a_aug

    r_momentum = r_mom_f + r_mom_p
    a_momentum = a_mom_f + a_mom_p

    residual_form = (
        r_momentum
        + r_mass
        + r_pore
        + r_skel
        + r_kin
        + r_phi
        + r_alpha
        + r_interface_kin
        + r_interface_mass
        + r_interface_normal
        + r_interface_tangential
    )
    if r_domain_lm is not None:
        residual_form = residual_form + r_domain_lm
    if r_volumetric is not None:
        residual_form = residual_form + r_volumetric
    jacobian_form = (
        a_momentum
        + a_mass
        + a_pore
        + a_skel
        + a_kin
        + a_phi
        + a_alpha
        + a_interface_kin
        + a_interface_mass
        + a_interface_normal
        + a_interface_tangential
    )
    if a_domain_lm is not None:
        jacobian_form = jacobian_form + a_domain_lm
    if a_volumetric is not None:
        jacobian_form = jacobian_form + a_volumetric

    return BiofilmOneDomainForms(
        jacobian_form=jacobian_form,
        residual_form=residual_form,
        r_momentum=r_momentum,
        r_mass=r_mass,
        r_kinematics=r_kin,
        r_skeleton=r_skel,
        r_phi=r_phi,
        r_phi_transport=r_phi_transport,
        r_alpha=r_alpha,
        r_alpha_transport=r_alpha_transport,
        r_mu_alpha=_safe_zero_scalar_linear_form(alpha_k, alpha_test),
        r_damage=None,
        r_substrate=_safe_zero_scalar_linear_form(alpha_k, alpha_test),
        r_B=None,
        r_pore=r_pore,
        r_total_mass=r_mass + r_pore + r_phi + r_interface_mass,
        r_momentum_terms={
            "free_bulk": r_mom_f_bulk,
            "pore_bulk": r_mom_p_bulk,
            "solid_bulk": r_skel_bulk,
            "direct_interface_traction_free": r_interface_direct_traction_free,
            "direct_interface_tangential_drag_free": r_interface_direct_tangential_drag_free,
            "free_extension_v": r_mom_f - r_mom_f_bulk,
            "pore_extension_vP": r_mom_p - r_mom_p_bulk,
            "interface_mass_constraint": r_interface_mass,
            "interface_normal_constraint": r_interface_normal,
            "interface_tangential_constraint": r_interface_tangential,
            "interface_mass_bulk_coupling": r_interface_mass_bulk,
            "interface_mass_aug_bulk": r_interface_mass_aug_bulk,
            "interface_normal_bulk_coupling": r_interface_normal_bulk,
            "interface_normal_aug_bulk": r_interface_normal_aug_bulk,
            "interface_tangential_bulk_coupling": r_interface_tangential_bulk,
        },
        r_mass_terms={
            "free_bulk": r_mass_bulk,
            "pore_bulk": r_pore,
            "direct_interface_mass_free": direct_interface_mass_free_k,
            "direct_interface_mass_porous": direct_interface_mass_porous_k,
        },
        r_kinematics_terms={
            "bulk": alpha_bulk_k * dot(kin_jump_k, u_test) * dx,
            "interface_kinematic_constraint": r_interface_kin,
            "interface_kinematic_bulk_coupling_u": r_interface_kin_bulk_u,
            "interface_kinematic_bulk_coupling_vS": r_interface_kin_bulk_vS,
            "interface_kinematic_bulk_coupling_alpha": r_interface_kin_bulk_alpha,
        },
        r_skeleton_terms={
            "solid_bulk": r_skel_bulk,
            "combined_porous_bulk": r_combined_porous_bulk,
            "direct_interface_traction": r_interface_direct_traction,
            "direct_interface_traction_porous": r_interface_direct_traction,
            "direct_interface_tangential_drag": r_interface_direct_tangential_drag,
            "pore_pressure_grad_phi": (
                (-(alpha_biot_c * p_pore_k * div_alpha_bulk_vStest_k) * dx)
                if quasi_static_porous_media
                else (-(alpha_biot_c * p_pore_k * div_alpha_bulk_vStest_k) * dx)
            ),
            "drag": -drag_coeff_k * dot(rel_p_k, vS_test) * dx,
        },
        a_momentum=a_momentum,
        a_momentum_terms={
            "direct_interface_traction_free": a_interface_direct_traction_free,
            "direct_interface_tangential_drag_free": a_interface_direct_tangential_drag_free,
            "interface_mass_constraint": a_interface_mass,
            "interface_normal_constraint": a_interface_normal,
            "interface_tangential_constraint": a_interface_tangential,
            "interface_mass_bulk_coupling": a_interface_mass_bulk,
            "interface_mass_aug_bulk": a_interface_mass_aug_bulk,
            "interface_normal_bulk_coupling": a_interface_normal_bulk,
            "interface_normal_aug_bulk": a_interface_normal_aug_bulk,
            "interface_tangential_bulk_coupling": a_interface_tangential_bulk,
        },
        a_mass=a_mass,
        a_mass_terms={
            "free_bulk": a_mass_bulk,
            "direct_interface_mass_free": ddirect_interface_mass_free,
            "direct_interface_mass_porous": ddirect_interface_mass_porous,
        },
        a_pore=a_pore,
        a_total_mass=a_mass + a_pore + a_phi + a_interface_mass,
        a_kinematics=a_kin,
        a_skeleton=a_skel,
        a_skeleton_terms={
            "solid_bulk": a_skel_bulk,
            "combined_porous_bulk": a_combined_porous_bulk,
                "direct_interface_traction": a_interface_direct_traction,
                "direct_interface_traction_porous": a_interface_direct_traction,
                "direct_interface_tangential_drag": a_interface_direct_tangential_drag,
                "pore_pressure_grad_phi": (
                    (
                        -(alpha_biot_c * (dp_pore * div_alpha_bulk_vStest_k + p_pore_k * ddiv_alpha_bulk_vStest))
                    )
                    if quasi_static_porous_media
                    else (
                    -(alpha_biot_c * (dp_pore * div_alpha_bulk_vStest_k + p_pore_k * ddiv_alpha_bulk_vStest))
                )
            ) * dx,
            "drag": -(ddrag_coeff * dot(rel_p_k, vS_test) + drag_coeff_k * dot(drel_p, vS_test)) * dx,
        },
        a_phi=a_phi,
        a_phi_transport=a_phi_transport,
        a_B=None,
        a_alpha=a_alpha,
        a_alpha_transport=a_alpha_transport,
        a_mu_alpha=None,
        a_damage=None,
        a_substrate=None,
        r_detached=None,
        a_detached=None,
        r_alpha_lambda=None,
        a_alpha_lambda=None,
        r_drag_lambda=None,
        a_drag_lambda=None,
        r_skeleton_pressure=None,
        a_skeleton_pressure=None,
        r_volumetric=r_volumetric,
        a_volumetric=a_volumetric,
        r_domain_lm=r_domain_lm,
        a_domain_lm=a_domain_lm,
        r_domain_lm_terms=r_domain_lm_terms,
        a_domain_lm_terms=a_domain_lm_terms,
    )


def build_biofilm_one_domain_final_form_decomposed(**kwargs) -> BiofilmOneDomainForms:
    return build_biofilm_one_domain_final_form(interface_formulation="decomposed", **kwargs)
