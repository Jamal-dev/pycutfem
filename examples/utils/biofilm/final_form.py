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


def _solid_model_key(model_key: str) -> str:
    return str(model_key or "linear").strip().lower().replace("-", "_")


def _solid_stress_and_tangent(*, solid_model: str, u_k, du, mu_s, lambda_s, dim: int):
    key = _solid_model_key(solid_model)
    if key in {"linear", "small_strain", "linear_elastic"}:
        eps_u = _eps(u_k)
        eps_du = _eps(du)
        sigma_k = _c(2.0) * mu_s * eps_u + lambda_s * div(u_k) * Identity(int(dim))
        dsigma = _c(2.0) * mu_s * eps_du + lambda_s * div(du) * Identity(int(dim))
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
    support_indicator_beta: float = 0.0,
    support_indicator_mode: str = "raw",
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
    normal_pressure_scale_c = _named_c("ff_normal_pressure_scale", float(normal_pressure_scale))
    bjs_coeff_c = _named_c("ff_bjs_coefficient", float(bjs_coefficient))
    solid_interface_traction_weight_c = _named_c(
        "ff_solid_interface_traction_weight",
        float(solid_interface_traction_weight),
    )
    if isinstance(mass_interface_weight, Constant):
        mass_interface_weight_raw = mass_interface_weight.value
    else:
        mass_interface_weight_raw = mass_interface_weight
    mass_interface_weight_arr = np.asarray(mass_interface_weight_raw, dtype=float).ravel()
    if mass_interface_weight_arr.size != 1:
        raise ValueError("mass_interface_weight must be scalar.")
    mass_interface_weight_val = float(mass_interface_weight_arr[0])
    if (not math.isfinite(mass_interface_weight_val)) or not (0.0 <= mass_interface_weight_val <= 1.0):
        raise ValueError("mass_interface_weight must be finite and lie in [0, 1].")
    if isinstance(mass_interface_weight, Constant):
        mass_interface_weight_c = mass_interface_weight
    else:
        mass_interface_weight_c = _named_c("ff_mass_interface_weight", float(mass_interface_weight_val))
    if isinstance(normal_interface_weight, Constant):
        normal_interface_weight_raw = normal_interface_weight.value
    else:
        normal_interface_weight_raw = normal_interface_weight
    normal_interface_weight_arr = np.asarray(normal_interface_weight_raw, dtype=float).ravel()
    if normal_interface_weight_arr.size != 1:
        raise ValueError("normal_interface_weight must be scalar.")
    normal_interface_weight_val = float(normal_interface_weight_arr[0])
    if (not math.isfinite(normal_interface_weight_val)) or not (0.0 <= normal_interface_weight_val <= 1.0):
        raise ValueError("normal_interface_weight must be finite and lie in [0, 1].")
    disable_interface_physics = bool(disable_interface_physics)
    one_m_mass_interface_weight_c = _c(1.0) - mass_interface_weight_c
    sqrt_kappa_inv_c = kappa_inv_c ** _c(0.5)
    use_bjs_tangential_law = abs(float(bjs_coefficient)) > 0.0
    zero = _c(0.0)
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
    if disable_normal_interface:
        normal_interface_weight_c = _named_c("ff_normal_interface_weight", 0.0)
    else:
        normal_interface_weight_c = _named_c(
            "ff_normal_interface_weight",
            normal_interface_weight,
        )
    one_m_normal_interface_weight_c = _c(1.0) - normal_interface_weight_c
    alpha_support_k = _smooth_support_indicator_expr(alpha_k, beta=support_indicator_beta_val)
    alpha_support_n = _smooth_support_indicator_expr(alpha_n, beta=support_indicator_beta_val)
    dalpha_support = _smooth_support_indicator_prime_expr(alpha_k, beta=support_indicator_beta_val) * dalpha
    F_support_k = _one_minus(alpha_support_k)
    dF_support = -dalpha_support
    interface_band_weight_k = _c(4.0) * alpha_support_k * F_support_k
    dinterface_band_weight = _c(4.0) * (
        dalpha_support * F_support_k + alpha_support_k * dF_support
    )
    alpha_bulk_k = _support_indicator_expr(alpha_k, beta=support_indicator_beta_val, mode=support_indicator_mode_key)
    alpha_bulk_n = _support_indicator_expr(alpha_n, beta=support_indicator_beta_val, mode=support_indicator_mode_key)
    dalpha_bulk = _support_indicator_prime_expr(alpha_k, beta=support_indicator_beta_val, mode=support_indicator_mode_key) * dalpha
    F_bulk_k = _free_indicator_expr(alpha_k, beta=support_indicator_beta_val, mode=support_indicator_mode_key)
    dF_bulk = _free_indicator_prime_expr(alpha_k, beta=support_indicator_beta_val, mode=support_indicator_mode_key) * dalpha
    if use_domain_lm:
        required_domain_lm = (
            ("dlm_vf", dlm_vf),
            ("dlm_p", dlm_p),
            ("dlm_vP", dlm_vP),
            ("dlm_vS", dlm_vS),
            ("dlm_p_pore", dlm_p_pore),
            ("dlm_phi", dlm_phi),
            ("dlm_u", dlm_u),
            ("lm_vf_test", lm_vf_test),
            ("lm_p_test", lm_p_test),
            ("lm_vP_test", lm_vP_test),
            ("lm_vS_test", lm_vS_test),
            ("lm_p_pore_test", lm_p_pore_test),
            ("lm_phi_test", lm_phi_test),
            ("lm_u_test", lm_u_test),
            ("lm_vf_k", lm_vf_k),
            ("lm_p_k", lm_p_k),
            ("lm_vP_k", lm_vP_k),
            ("lm_vS_k", lm_vS_k),
            ("lm_p_pore_k", lm_p_pore_k),
            ("lm_phi_k", lm_phi_k),
            ("lm_u_k", lm_u_k),
        )
        missing = [name for name, value in required_domain_lm if value is None]
        if missing:
            raise ValueError(
                "domain_lm=True requires the domain LM fields/tests/increments to be present. "
                f"Missing: {', '.join(missing)}."
            )
    if rigid_darcy_head_mode and use_domain_lm:
        raise ValueError("domain_lm is not implemented for rigid_darcy_head_mode=True.")
    one_m_phi_k = _one_minus(phi_k)
    one_m_phi_n = _one_minus(phi_n)
    rho_s_phys_k = rho_s_ref_c if constant_rho_s else rho_s_k
    rho_s_phys_n = rho_s_ref_c if constant_rho_s else rho_s_n
    drho_s_phys = zero if constant_rho_s else drho_s
    sigma_f_k = _c(2.0) * mu_f_c * _eps(v_k) - p_k * Identity(dim)
    dsigma_f = _c(2.0) * mu_f_c * _eps(dv) - dp * Identity(dim)
    mu_b_key = str(mu_b_model or "phi_mu").strip().lower()

    def _pore_effective_viscosity_terms(phi_expr, dphi_expr):
        if mu_b_key in {"mu", "const", "constant"}:
            return mu_f_c, zero
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
    zero_v = _named_c("ff_zero_v", (0.0, 0.0))
    zero_u = _named_c("ff_zero_u", (0.0, 0.0))

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
    drag_coeff_k = alpha_bulk_k * phi_k * phi_k * kappa_inv_c
    ddrag_coeff = (dalpha_bulk * phi_k * phi_k + alpha_bulk_k * _c(2.0) * phi_k * dphi) * kappa_inv_c
    rel_p_k = vP_k - vS_k
    drel_p = dvP - dvS
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
                dfluid_traction_loc = dot(dsigma_f_loc, normal_vec)
                dfluid_normal_trac_loc = dot(normal_vec, dfluid_traction_loc)
                dfluid_tang_trac_loc = dot(tangent_vec, dfluid_traction_loc)

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
            fluid_traction = dot(sigma_f_k, normal_vec)
            fluid_normal_trac = dot(normal_vec, fluid_traction)
            fluid_tang_trac = dot(tangent_vec, fluid_traction)
            fluid_tangential_velocity_k = dot(tangent_vec, v_k)

        r_mass = q_test * (F_bulk_k * div(v_k)) * dx
        a_mass = q_test * (dF_bulk * div(v_k) + F_bulk_k * div(dv)) * dx

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
        zero_phi_form = zero * zero_phi_test * dx
        zero_rho_s_form = zero * zero_rho_s_test * dx
        zero_alpha_form = zero * alpha_test * dx
        zero_q_form = zero * zero_scalar_test * dx

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
            r_alpha=zero_alpha_form,
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
            a_B=None,
            a_alpha=zero_alpha_form,
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
    r_skel += (alpha_bulk_k * p_pore_k) * dot(grad(phi_k), vS_test) * dx
    a_skel += (
        (dalpha_bulk * p_pore_k + alpha_bulk_k * dp_pore) * dot(grad(phi_k), vS_test)
        + alpha_bulk_k * p_pore_k * dot(grad(dphi), vS_test)
    ) * dx
    r_skel += -drag_coeff_k * dot(rel_p_k, vS_test) * dx
    a_skel += -(ddrag_coeff * dot(rel_p_k, vS_test) + drag_coeff_k * dot(drel_p, vS_test)) * dx

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
    if phi_mode_key != "alpha_closure":
        if constant_rho_s:
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
        r_phi_bulk = zero * phi_test * dx
        a_phi_bulk = zero * phi_test * dx
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
        r_phi_bulk = zero * phi_test * dx
        a_phi_bulk = zero * phi_test * dx
        r_solid_mass = rho_s_test * (alpha_bulk_k * combined_mass_bulk_k) * dx
        a_solid_mass = rho_s_test * (
            dalpha_bulk * combined_mass_bulk_k + alpha_bulk_k * dcombined_mass_bulk
        ) * dx
    mass_constraint_k = dot(grad(alpha_k), mass_jump_k)
    dmass_constraint = dot(grad(dalpha), mass_jump_k) + dot(grad(alpha_k), dmass_jump)
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

    kin_jump_k = (u_k - u_n) * inv_dt + dot(grad(u_k), vS_k) - vS_k
    dkin_jump = du * inv_dt + dot(grad(du), vS_k) + dot(grad(u_k), dvS) - dvS
    r_kin = alpha_bulk_k * dot(kin_jump_k, u_test) * dx
    a_kin = (
        dalpha_bulk * dot(kin_jump_k, u_test)
        + alpha_bulk_k
        * dot(dkin_jump, u_test)
    ) * dx

    r_alpha = alpha_test * ((alpha_k - alpha_n) * inv_dt + dot(vS_k, grad(alpha_k))) * dx
    a_alpha = alpha_test * (dalpha * inv_dt + dot(dvS, grad(alpha_k)) + dot(vS_k, grad(dalpha))) * dx

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
        r_interface_kin_bulk_alpha = zero * alpha_test * dx
        a_interface_kin_bulk_alpha = zero * alpha_test * dx

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
        fluid_tangential_velocity_k = t0 * v_k[0] + t1 * v_k[1]
        dfluid_tangential_velocity = (
            dt0 * v_k[0] + t0 * dv[0] + dt1 * v_k[1] + t1 * dv[1]
        )
        pore_tangential_velocity_k = t0 * vP_k[0] + t1 * vP_k[1]
        dpore_tangential_velocity = (
            dt0 * vP_k[0] + t0 * dvP[0] + dt1 * vP_k[1] + t1 * dvP[1]
        )
        bjs_beta_k = bjs_coeff_c * mu_f_c * alpha_k * sqrt_kappa_inv_c
        dbjs_beta = bjs_coeff_c * mu_f_c * dalpha * sqrt_kappa_inv_c
        bjs_slip_t_k = fluid_tangential_velocity_k - phi_k * pore_tangential_velocity_k
        dbjs_slip_t = dfluid_tangential_velocity - (
            dphi * pore_tangential_velocity_k + phi_k * dpore_tangential_velocity
        )
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
            dbjs_beta_loc = bjs_coeff_c * mu_f_c * dalpha_loc * sqrt_kappa_inv_c
            dbjs_slip_t_loc = dfluid_tangential_velocity_loc - (
                dphi_loc * pore_tangential_velocity_k + phi_k * dpore_tangential_velocity_loc
            )
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
            dtangent_loc = dot(rot90, dnormal_loc)

            dfluid_traction_loc = dot(dsigma_f_loc, normal_vec) + dot(sigma_f_k, dnormal_loc)
            dpore_traction_loc = dot(dsigma_p_loc, normal_vec) + dot(sigma_p_k, dnormal_loc)
            dsolid_traction_loc = dot(dsigma_s_loc, normal_vec) + dot(sigma_s_k, dnormal_loc)
            dfluid_normal_trac_loc = dot(dnormal_loc, fluid_traction) + dot(normal_vec, dfluid_traction_loc)
            dpore_normal_trac_loc = dot(dnormal_loc, pore_traction) + dot(normal_vec, dpore_traction_loc)
            dsolid_normal_trac_loc = dot(dnormal_loc, solid_traction) + dot(normal_vec, dsolid_traction_loc)
            dfluid_tang_trac_loc = dot(dtangent_loc, fluid_traction) + dot(tangent_vec, dfluid_traction_loc)
            dpore_tang_trac_loc = dot(dtangent_loc, pore_traction) + dot(tangent_vec, dpore_traction_loc)
            dsolid_tang_trac_loc = dot(dtangent_loc, solid_traction) + dot(tangent_vec, dsolid_traction_loc)
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

        fluid_traction = dot(sigma_f_k, normal_vec)
        dfluid_traction = dot(dsigma_f, normal_vec) + dot(sigma_f_k, dnormal_vec)
        fluid_normal_trac = dot(normal_vec, fluid_traction)
        dfluid_normal_trac = dot(dnormal_vec, fluid_traction) + dot(normal_vec, dfluid_traction)
        pore_traction = dot(sigma_p_k, normal_vec)
        dpore_traction = dot(dsigma_p, normal_vec) + dot(sigma_p_k, dnormal_vec)
        pore_normal_trac = dot(normal_vec, pore_traction)
        dpore_normal_trac = dot(dnormal_vec, pore_traction) + dot(normal_vec, dpore_traction)
        solid_traction = dot(sigma_s_k, normal_vec)
        dsolid_traction = dot(dsigma_s, normal_vec) + dot(sigma_s_k, dnormal_vec)
        solid_normal_trac = dot(normal_vec, solid_traction)
        dsolid_normal_trac = dot(dnormal_vec, solid_traction) + dot(normal_vec, dsolid_traction)
        fluid_tang_trac = dot(tangent_vec, fluid_traction)
        dfluid_tang_trac = dot(dtangent_vec, fluid_traction) + dot(tangent_vec, dfluid_traction)
        pore_tang_trac = dot(tangent_vec, pore_traction)
        dpore_tang_trac = dot(dtangent_vec, pore_traction) + dot(tangent_vec, dpore_traction)
        solid_tang_trac = dot(tangent_vec, solid_traction)
        dsolid_tang_trac = dot(dtangent_vec, solid_traction) + dot(tangent_vec, dsolid_traction)
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
        bjs_beta_k = bjs_coeff_c * mu_f_c * alpha_k * sqrt_kappa_inv_c
        dbjs_beta = bjs_coeff_c * mu_f_c * dalpha * sqrt_kappa_inv_c
        bjs_slip_t_k = fluid_tangential_velocity_k - phi_k * pore_tangential_velocity_k
        dbjs_slip_t = dfluid_tangential_velocity - (
            dphi * pore_tangential_velocity_k + phi_k * dpore_tangential_velocity
        )
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
            dphi_loc = zero if dphi_var is None else dphi_var
            dalpha_loc = zero if dalpha_var is None else dalpha_var
            dgrad_alpha_loc = grad(dalpha_loc)
            dgrad_alpha_mag_loc = dot(grad_alpha_k, dgrad_alpha_loc) * inv_grad_alpha_mag
            dnormal_loc = (dgrad_alpha_loc - normal_vec * dgrad_alpha_mag_loc) * inv_grad_alpha_mag
            dtangent_loc = dot(rot90, dnormal_loc)
            dsigma_f_loc = _c(2.0) * mu_f_c * _eps(dv_loc)
            dfluid_traction_loc = dot(dsigma_f_loc, normal_vec) + dot(sigma_f_k, dnormal_loc)
            dfluid_tang_trac_loc = dot(dtangent_loc, fluid_traction) + dot(tangent_vec, dfluid_traction_loc)
            dfluid_tangential_velocity_loc = dot(dtangent_loc, v_k) + dot(tangent_vec, dv_loc)
            dpore_tangential_velocity_loc = dot(dtangent_loc, vP_k) + dot(tangent_vec, dvP_loc)
            dbjs_beta_loc = bjs_coeff_c * mu_f_c * dalpha_loc * sqrt_kappa_inv_c
            dbjs_slip_t_loc = dfluid_tangential_velocity_loc - (
                dphi_loc * pore_tangential_velocity_k + phi_k * dpore_tangential_velocity_loc
            )
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
        normal_test_bracket_p, _ = _traction_directional(dp_var=q_test)
        normal_test_bracket_p_pore, _ = _traction_directional(dp_pore_var=q_pore_test)
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

        r_interface_normal = zero * mu_normal_test * dx
        a_interface_normal = zero * mu_normal_test * dx
        r_interface_tangential = mu_tangent_test * tangential_constraint_k * dx
        a_interface_tangential = mu_tangent_test * dtangential_constraint * dx
        r_interface_normal_bulk = zero * q_pore_test * dx
        a_interface_normal_bulk = zero * q_pore_test * dx
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
        r_interface_mass_aug_bulk = zero * q_test * dx
        a_interface_mass_aug_bulk = zero * q_test * dx
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
        r_interface_normal_aug_bulk = zero * q_pore_test * dx
        a_interface_normal_aug_bulk = zero * q_pore_test * dx

    if disable_interface_physics:
        zero_q_form = zero * q_test * dx
        zero_u_form = zero * dot(u_k, u_test) * dx
        zero_vS_form = zero * dot(vS_k, vS_test) * dx
        zero_alpha_form = zero * alpha_test * dx

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
        r_domain_lm = zero * q_test * dx
        a_domain_lm = zero * q_test * dx
        r_domain_lm_terms = {}
        a_domain_lm_terms = {}

        def _domain_lm_scalar(*, target_k, dtarget, target_test, lm_k, dlm, lm_test, active_weight, dactive_weight, target_shift=zero):
            target_residual = target_k - target_shift
            r_row = lm_test * (active_weight * target_residual) * dx
            a_row = lm_test * (
                dactive_weight * target_residual
                + active_weight * dtarget
            ) * dx
            r_bulk = active_weight * lm_k * target_test * dx
            a_bulk = (
                dactive_weight * lm_k * target_test
                + active_weight * dlm * target_test
            ) * dx
            active_constraint = active_weight * target_residual
            dactive_constraint = dactive_weight * target_residual + active_weight * dtarget
            r_aug = domain_lm_aug_gamma_c * active_constraint * active_weight * target_test * dx
            a_aug = domain_lm_aug_gamma_c * (
                dactive_constraint * active_weight * target_test
                + active_constraint * dactive_weight * target_test
            ) * dx
            return r_row, a_row, r_bulk, a_bulk, r_aug, a_aug

        def _domain_lm_vector(*, target_k, dtarget, target_test, lm_k, dlm, lm_test, active_weight, dactive_weight, target_shift=None):
            target_residual = target_k if target_shift is None else (target_k - target_shift)
            r_row = dot(lm_test, active_weight * target_residual) * dx
            a_row = dot(
                lm_test,
                dactive_weight * target_residual
                + active_weight * dtarget
            ) * dx
            r_bulk = active_weight * dot(lm_k, target_test) * dx
            a_bulk = (
                dactive_weight * dot(lm_k, target_test)
                + active_weight * dot(dlm, target_test)
            ) * dx
            active_constraint = active_weight * target_residual
            dactive_constraint = dactive_weight * target_residual + active_weight * dtarget
            r_aug = domain_lm_aug_gamma_c * active_weight * dot(active_constraint, target_test) * dx
            a_aug = domain_lm_aug_gamma_c * (
                active_weight * dot(dactive_constraint, target_test)
                + dactive_weight * dot(active_constraint, target_test)
            ) * dx
            return r_row, a_row, r_bulk, a_bulk, r_aug, a_aug

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

        r_row, a_row, r_bulk, a_bulk, r_aug, a_aug = _domain_lm_vector(
            target_k=v_k,
            dtarget=dv,
            target_test=v_test,
            lm_k=lm_vf_k,
            dlm=dlm_vf,
            lm_test=lm_vf_test,
            active_weight=alpha_bulk_k,
            dactive_weight=dalpha_bulk,
        )
        _record_domain_lm("support_kill_vf", r_row, a_row, r_bulk, a_bulk, r_aug, a_aug)
        r_mom_f = r_mom_f + r_bulk + r_aug
        a_mom_f = a_mom_f + a_bulk + a_aug

        r_row, a_row, r_bulk, a_bulk, r_aug, a_aug = _domain_lm_scalar(
            target_k=p_k,
            dtarget=dp,
            target_test=q_test,
            lm_k=lm_p_k,
            dlm=dlm_p,
            lm_test=lm_p_test,
            active_weight=alpha_bulk_k,
            dactive_weight=dalpha_bulk,
        )
        _record_domain_lm("support_kill_p", r_row, a_row, r_bulk, a_bulk, r_aug, a_aug)
        r_mass = r_mass + r_bulk + r_aug
        a_mass = a_mass + a_bulk + a_aug

        r_row, a_row, r_bulk, a_bulk, r_aug, a_aug = _domain_lm_vector(
            target_k=vP_k,
            dtarget=dvP,
            target_test=vP_test,
            lm_k=lm_vP_k,
            dlm=dlm_vP,
            lm_test=lm_vP_test,
            active_weight=F_bulk_k,
            dactive_weight=dF_bulk,
        )
        _record_domain_lm("free_kill_vP", r_row, a_row, r_bulk, a_bulk, r_aug, a_aug)
        r_mom_p = r_mom_p + r_bulk + r_aug
        a_mom_p = a_mom_p + a_bulk + a_aug

        r_row, a_row, r_bulk, a_bulk, r_aug, a_aug = _domain_lm_vector(
            target_k=vS_k,
            dtarget=dvS,
            target_test=vS_test,
            lm_k=lm_vS_k,
            dlm=dlm_vS,
            lm_test=lm_vS_test,
            active_weight=F_bulk_k,
            dactive_weight=dF_bulk,
        )
        _record_domain_lm("free_kill_vS", r_row, a_row, r_bulk, a_bulk, r_aug, a_aug)
        r_skel = r_skel + r_bulk + r_aug
        a_skel = a_skel + a_bulk + a_aug

        r_row, a_row, r_bulk, a_bulk, r_aug, a_aug = _domain_lm_scalar(
            target_k=p_pore_k,
            dtarget=dp_pore,
            target_test=q_pore_test,
            lm_k=lm_p_pore_k,
            dlm=dlm_p_pore,
            lm_test=lm_p_pore_test,
            active_weight=F_bulk_k,
            dactive_weight=dF_bulk,
        )
        _record_domain_lm("free_kill_p_pore", r_row, a_row, r_bulk, a_bulk, r_aug, a_aug)
        r_pore = _form_add(r_pore, _form_add(r_bulk, r_aug))
        a_pore = _form_add(a_pore, _form_add(a_bulk, a_aug))

        r_row, a_row, r_bulk, a_bulk, r_aug, a_aug = _domain_lm_scalar(
            target_k=phi_k,
            dtarget=dphi,
            target_test=phi_test,
            lm_k=lm_phi_k,
            dlm=dlm_phi,
            lm_test=lm_phi_test,
            active_weight=F_bulk_k,
            dactive_weight=dF_bulk,
            target_shift=_c(1.0),
        )
        _record_domain_lm("free_kill_phi", r_row, a_row, r_bulk, a_bulk, r_aug, a_aug)
        r_phi = _form_add(r_phi, _form_add(r_bulk, r_aug))
        a_phi = _form_add(a_phi, _form_add(a_bulk, a_aug))

        r_row, a_row, r_bulk, a_bulk, r_aug, a_aug = _domain_lm_vector(
            target_k=u_k,
            dtarget=du,
            target_test=u_test,
            lm_k=lm_u_k,
            dlm=dlm_u,
            lm_test=lm_u_test,
            active_weight=F_bulk_k,
            dactive_weight=dF_bulk,
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
        r_alpha=r_alpha,
        r_mu_alpha=zero * alpha_test * dx,
        r_damage=None,
        r_substrate=zero * alpha_test * dx,
        r_B=None,
        r_pore=r_pore,
        r_total_mass=r_mass + r_pore + r_phi + r_interface_mass,
        r_momentum_terms={
            "free_bulk": r_mom_f_bulk,
            "pore_bulk": r_mom_p_bulk,
            "solid_bulk": r_skel_bulk,
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
        r_kinematics_terms={
            "bulk": alpha_bulk_k * dot(kin_jump_k, u_test) * dx,
            "interface_kinematic_constraint": r_interface_kin,
            "interface_kinematic_bulk_coupling_u": r_interface_kin_bulk_u,
            "interface_kinematic_bulk_coupling_vS": r_interface_kin_bulk_vS,
            "interface_kinematic_bulk_coupling_alpha": r_interface_kin_bulk_alpha,
        },
        r_skeleton_terms={
            "solid_bulk": r_skel_bulk,
            "pore_pressure_grad_phi": (alpha_bulk_k * p_pore_k) * dot(grad(phi_k), vS_test) * dx,
            "drag": -drag_coeff_k * dot(rel_p_k, vS_test) * dx,
        },
        a_momentum=a_momentum,
        a_momentum_terms={
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
        a_pore=a_pore,
        a_total_mass=a_mass + a_pore + a_phi + a_interface_mass,
        a_kinematics=a_kin,
        a_skeleton=a_skel,
        a_skeleton_terms={
            "solid_bulk": a_skel_bulk,
            "pore_pressure_grad_phi": (
                (dalpha_bulk * p_pore_k + alpha_bulk_k * dp_pore) * dot(grad(phi_k), vS_test)
                + alpha_bulk_k * p_pore_k * dot(grad(dphi), vS_test)
            ) * dx,
            "drag": -(ddrag_coeff * dot(rel_p_k, vS_test) + drag_coeff_k * dot(drel_p, vS_test)) * dx,
        },
        a_phi=a_phi,
        a_B=None,
        a_alpha=a_alpha,
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
