"""Canonical three-constituent one-domain FPSI form builder.

This module implements the first conservative Eulerian version of the model in
``private/notes/examples/one-domain-model/three_constituent_one_domain_implementation_plan.md``.

Primary fields are global:

* free liquid: ``v_f, p_f``
* pore liquid: ``v_p, p_p``
* solid skeleton: ``v_s, u_s``
* contents: ``alpha, phi``
* free/pore relabeling rate: ``Gamma``

The displacement-like skeleton field is always active.  Its Eulerian kinematic
row is

    d_t u_s + (grad u_s) v_s - v_s = 0.

The implementation supports both the diagonal calibration law and the full
Cholesky-parameterized pair-resistance law.  The algebra is written with named,
runtime-preserved constants so C++ kernels are not specialized on every
numerical parameter value.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b

import numpy as np

from pycutfem.ufl.autodiff import linearize_form
from pycutfem.ufl.expressions import Constant, div, dot, grad, inner
from pycutfem.ufl.linalg import apply_pair_space_cholesky

_LITERAL_CONSTANT_CACHE: dict[float, Constant] = {}
_NAMED_CONSTANT_CACHE: dict[tuple[str, int, tuple[int, ...], tuple[float, ...]], Constant] = {}


def _named_c(name: str, value, *, dim: int | None = None) -> Constant:
    """Return a named runtime constant with stable JIT structure."""

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
    setattr(const, "_jit_name", str(name))
    setattr(const, "_name", str(name))
    setattr(const, "_preserve_runtime_structure", True)
    return const


def _lit(value: float) -> Constant:
    """Named literal used for fixed algebraic coefficients."""

    val = float(value)
    const = _LITERAL_CONSTANT_CACHE.get(val)
    if const is None:
        token = blake2b(format(val, ".16g").encode("ascii"), digest_size=8).hexdigest()
        const = _named_c(f"tc_lit_{token}", val)
        _LITERAL_CONSTANT_CACHE[val] = const
    return const


def _as_named_expr(name: str, value, *, dim: int | None = None):
    if value is None:
        return None
    if isinstance(value, (int, float, np.number, Constant, tuple, list, np.ndarray)):
        return _named_c(name, value, dim=dim)
    return value


def _one_minus(expr):
    return _lit(1.0) + _neg(expr)


def _neg(expr):
    return _lit(-1.0) * expr


def _sub(a, b):
    return a + _neg(b)


def _eps(v):
    return _lit(0.5) * (grad(v) + grad(v).T)


def _identity(dim: int):
    return _named_c(f"tc_identity_{int(dim)}", np.eye(int(dim)), dim=2)


def _sqrt(expr):
    return expr ** _lit(0.5)


def one_domain_contents(alpha, phi):
    """Return ``(F, P, B)`` contents from ``alpha`` and ``phi``."""

    F = _one_minus(alpha)
    P = alpha * phi
    B = alpha * _one_minus(phi)
    return F, P, B


def three_constituent_box_bounds(
    *,
    alpha_bounds: tuple[float | None, float | None] = (0.0, 1.0),
    phi_bounds: tuple[float | None, float | None] = (0.0, 1.0),
    pore_pressure_bounds: tuple[float | None, float | None] = (None, None),
) -> dict[str, tuple[float | None, float | None]]:
    """Return canonical physical box bounds for PDAS/IPM solvers.

    ``alpha`` is the porous-body support/geometry field and ``phi`` is the
    pore-liquid fraction inside that support.  Both are physical fractions, so
    the canonical admissible interval is ``[0, 1]`` unless a caller narrows it.
    Pore pressure is left unconstrained by default because pressure sign can be
    gauge-dependent; open-drainage benchmarks may opt into ``p_p >= 0``.
    """

    bounds = {
        "alpha": (alpha_bounds[0], alpha_bounds[1]),
        "phi": (phi_bounds[0], phi_bounds[1]),
    }
    if pore_pressure_bounds != (None, None):
        bounds["pp"] = (pore_pressure_bounds[0], pore_pressure_bounds[1])
    return bounds


def configure_three_constituent_pdas_bounds(
    solver,
    *,
    alpha_bounds: tuple[float | None, float | None] = (0.0, 1.0),
    phi_bounds: tuple[float | None, float | None] = (0.0, 1.0),
    pore_pressure_bounds: tuple[float | None, float | None] = (None, None),
):
    """Attach canonical physical box bounds to a PDAS/IPM Newton solver."""

    if not hasattr(solver, "set_box_bounds"):
        raise TypeError("solver must provide set_box_bounds(...), e.g. PdasNewtonSolver or InteriorPointNewtonSolver")
    solver.set_box_bounds(
        by_field=three_constituent_box_bounds(
            alpha_bounds=alpha_bounds,
            phi_bounds=phi_bounds,
            pore_pressure_bounds=pore_pressure_bounds,
        )
    )
    return solver


def build_three_constituent_pdas_solver(
    forms: "ThreeConstituentOneDomainForms",
    *,
    dof_handler,
    mixed_element,
    bcs=None,
    bcs_homog=None,
    newton_params=None,
    vi_params=None,
    backend: str = "cpp",
    quad_order: int = 6,
    alpha_bounds: tuple[float | None, float | None] = (0.0, 1.0),
    phi_bounds: tuple[float | None, float | None] = (0.0, 1.0),
    pore_pressure_bounds: tuple[float | None, float | None] = (None, None),
    **solver_kwargs,
):
    """Build a PDAS Newton solver for the canonical bounded model."""

    from pycutfem.solvers.nonlinear_solver import NewtonParameters, PdasNewtonSolver, VIParameters

    solver = PdasNewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=dof_handler,
        mixed_element=mixed_element,
        bcs=[] if bcs is None else bcs,
        bcs_homog=[] if bcs_homog is None else bcs_homog,
        newton_params=NewtonParameters() if newton_params is None else newton_params,
        vi_params=VIParameters() if vi_params is None else vi_params,
        backend=str(backend),
        quad_order=int(quad_order),
        **solver_kwargs,
    )
    return configure_three_constituent_pdas_bounds(
        solver,
        alpha_bounds=alpha_bounds,
        phi_bounds=phi_bounds,
        pore_pressure_bounds=pore_pressure_bounds,
    )


def structural_gradients(alpha, phi):
    """Return pairwise structural gradients ``(g_fp, g_fs, g_ps)``."""

    return (
        phi * grad(alpha),
        _one_minus(phi) * grad(alpha),
        alpha * grad(phi),
    )


def linear_fluid_stress(v, p, *, mu, dim: int):
    return _lit(2.0) * mu * _eps(v) + _neg(p * _identity(int(dim)))


def linear_skeleton_stress(u, *, mu, lambda_, dim: int):
    return _lit(2.0) * mu * _eps(u) + lambda_ * div(u) * _identity(int(dim))


def _pair_weight(raw_weight_sq, eps, *, epsilon_name: str = "tc_pair_weight_epsilon"):
    eps_expr = _as_named_expr(str(epsilon_name), 0.0 if eps is None else eps)
    try:
        eps_value = float(np.asarray(getattr(eps_expr, "value", eps_expr), dtype=float))
    except Exception:
        eps_value = float("nan")
    if np.isfinite(eps_value) and abs(float(eps_value)) <= 0.0:
        return _sqrt(raw_weight_sq)
    return _sqrt(raw_weight_sq + eps_expr) + _neg(_sqrt(eps_expr))


def _lower_cholesky_entries(R_pair_cholesky):
    arr = np.asarray(R_pair_cholesky, dtype=float)
    if arr.shape == (6,):
        l00, l10, l11, l20, l21, l22 = [float(v) for v in arr.reshape(6)]
    elif arr.shape == (3, 3):
        if np.any(np.abs(np.triu(arr, k=1)) > 1.0e-14):
            raise ValueError("R_pair_cholesky must be lower triangular.")
        l00 = float(arr[0, 0])
        l10 = float(arr[1, 0])
        l11 = float(arr[1, 1])
        l20 = float(arr[2, 0])
        l21 = float(arr[2, 1])
        l22 = float(arr[2, 2])
    else:
        raise ValueError("R_pair_cholesky must be a 3x3 lower-triangular matrix or 6 lower entries.")
    vals = (l00, l10, l11, l20, l21, l22)
    if not all(np.isfinite(v) for v in vals):
        raise ValueError("R_pair_cholesky entries must be finite.")
    return tuple(_as_named_expr(f"tc_R_pair_cholesky_{name}", val) for name, val in zip(("00", "10", "11", "20", "21", "22"), vals))


def full_block_pairwise_internal_forces(
    *,
    alpha,
    phi,
    v_f,
    v_p,
    v_s,
    sigma_f,
    sigma_p,
    sigma_s,
    R_pair_cholesky,
    theta_fp,
    pair_weight_epsilon=0.0,
):
    """Tri-phasic forces for a full SPD pair-resistance closure.

    The pair-space resistance is parameterized by a lower Cholesky factor
    ``L`` with ``R = L L.T``.  Weighted pair coefficients use
    ``diag(m_fp, m_fs, m_ps) R diag(m_fp, m_fs, m_ps)`` so dissipation remains
    nonnegative by construction.
    """

    F, P, B = one_domain_contents(alpha, phi)
    g_fp, g_fs, g_ps = structural_gradients(alpha, phi)

    sigma_fp = theta_fp * sigma_f + _one_minus(theta_fp) * sigma_p
    sigma_fs = sigma_f
    sigma_ps = sigma_p

    I_f_rev = dot(sigma_fp, g_fp) + dot(sigma_fs, g_fs)
    I_p_rev = _neg(dot(sigma_fp, g_fp)) + _neg(dot(sigma_ps, g_ps))
    I_s_rev = _neg(dot(sigma_fs, g_fs)) + dot(sigma_ps, g_ps)

    rel_fp = v_f + _neg(v_p)
    rel_fs = v_f + _neg(v_s)
    rel_ps = v_p + _neg(v_s)

    m_fp = _pair_weight(F * P, pair_weight_epsilon)
    m_fs = _pair_weight(F * B, pair_weight_epsilon)
    m_ps = P

    spd_result = apply_pair_space_cholesky(
        (rel_fp, rel_fs, rel_ps),
        (m_fp, m_fs, m_ps),
        _lower_cholesky_entries(R_pair_cholesky),
    )
    C_fp_fp, C_fp_fs, C_fp_ps, C_fs_fs, C_fs_ps, C_ps_ps = spd_result.coefficients
    Y_fp, Y_fs, Y_ps = spd_result.conjugates

    I_f_diss = _neg(Y_fp) + _neg(Y_fs)
    I_p_diss = Y_fp + _neg(Y_ps)
    I_s_diss = Y_fs + Y_ps

    return {
        "F": F,
        "P": P,
        "B": B,
        "g_fp": g_fp,
        "g_fs": g_fs,
        "g_ps": g_ps,
        "sigma_fp": sigma_fp,
        "sigma_fs": sigma_fs,
        "sigma_ps": sigma_ps,
        "I_f_rev": I_f_rev,
        "I_p_rev": I_p_rev,
        "I_s_rev": I_s_rev,
        "I_f_diss": I_f_diss,
        "I_p_diss": I_p_diss,
        "I_s_diss": I_s_diss,
        "I_f": I_f_rev + I_f_diss,
        "I_p": I_p_rev + I_p_diss,
        "I_s": I_s_rev + I_s_diss,
        "dissipation_density": spd_result.dissipation_density,
        "pair_resistance_model": "full_cholesky",
        "m_fp": m_fp,
        "m_fs": m_fs,
        "m_ps": m_ps,
        "C_fp_fp": C_fp_fp,
        "C_fp_fs": C_fp_fs,
        "C_fp_ps": C_fp_ps,
        "C_fs_fs": C_fs_fs,
        "C_fs_ps": C_fs_ps,
        "C_ps_ps": C_ps_ps,
    }


def diagonal_pairwise_internal_forces(
    *,
    alpha,
    phi,
    v_f,
    v_p,
    v_s,
    sigma_f,
    sigma_p,
    sigma_s,
    R_fp,
    R_fs,
    R_ps,
    theta_fp,
):
    """Tri-phasic internal forces for the diagonal resistance closure.

    The implementation uses the squared pair weights directly:

    * ``m_fp^2 = F P``
    * ``m_fs^2 = F B``
    * ``m_ps^2 = P^2``

    This is algebraically equivalent to the diagonal potential and avoids
    differentiating square roots at vanishing contents.
    """

    F, P, B = one_domain_contents(alpha, phi)
    g_fp, g_fs, g_ps = structural_gradients(alpha, phi)

    sigma_fp = theta_fp * sigma_f + _one_minus(theta_fp) * sigma_p
    sigma_fs = sigma_f
    sigma_ps = sigma_p

    I_f_rev = dot(sigma_fp, g_fp) + dot(sigma_fs, g_fs)
    I_p_rev = _neg(dot(sigma_fp, g_fp)) + _neg(dot(sigma_ps, g_ps))
    I_s_rev = _neg(dot(sigma_fs, g_fs)) + dot(sigma_ps, g_ps)

    rel_fp = v_f + _neg(v_p)
    rel_fs = v_f + _neg(v_s)
    rel_ps = v_p + _neg(v_s)

    chi_fp = F * P * R_fp
    chi_fs = F * B * R_fs
    chi_ps = P * P * R_ps

    I_f_diss = _neg(chi_fp * rel_fp) + _neg(chi_fs * rel_fs)
    I_p_diss = (chi_fp * rel_fp) + _neg(chi_ps * rel_ps)
    I_s_diss = (chi_fs * rel_fs) + (chi_ps * rel_ps)

    return {
        "F": F,
        "P": P,
        "B": B,
        "g_fp": g_fp,
        "g_fs": g_fs,
        "g_ps": g_ps,
        "sigma_fp": sigma_fp,
        "sigma_fs": sigma_fs,
        "sigma_ps": sigma_ps,
        "I_f_rev": I_f_rev,
        "I_p_rev": I_p_rev,
        "I_s_rev": I_s_rev,
        "I_f_diss": I_f_diss,
        "I_p_diss": I_p_diss,
        "I_s_diss": I_s_diss,
        "I_f": I_f_rev + I_f_diss,
        "I_p": I_p_rev + I_p_diss,
        "I_s": I_s_rev + I_s_diss,
        "pair_resistance_model": "diagonal",
        "C_fp_fp": chi_fp,
        "C_fp_fs": _lit(0.0),
        "C_fp_ps": _lit(0.0),
        "C_fs_fs": chi_fs,
        "C_fs_ps": _lit(0.0),
        "C_ps_ps": chi_ps,
        "dissipation_density": (
            chi_fp * dot(rel_fp, rel_fp)
            + chi_fs * dot(rel_fs, rel_fs)
            + chi_ps * dot(rel_ps, rel_ps)
        ),
    }


def pairwise_internal_forces(
    *,
    alpha,
    phi,
    v_f,
    v_p,
    v_s,
    sigma_f,
    sigma_p,
    sigma_s,
    R_fp,
    R_fs,
    R_ps,
    theta_fp,
    R_pair_cholesky=None,
    pair_weight_epsilon=0.0,
):
    """Return internal forces for diagonal or full Cholesky pair resistance."""

    if R_pair_cholesky is None:
        return diagonal_pairwise_internal_forces(
            alpha=alpha,
            phi=phi,
            v_f=v_f,
            v_p=v_p,
            v_s=v_s,
            sigma_f=sigma_f,
            sigma_p=sigma_p,
            sigma_s=sigma_s,
            R_fp=R_fp,
            R_fs=R_fs,
            R_ps=R_ps,
            theta_fp=theta_fp,
        )
    return full_block_pairwise_internal_forces(
        alpha=alpha,
        phi=phi,
        v_f=v_f,
        v_p=v_p,
        v_s=v_s,
        sigma_f=sigma_f,
        sigma_p=sigma_p,
        sigma_s=sigma_s,
        R_pair_cholesky=R_pair_cholesky,
        theta_fp=theta_fp,
        pair_weight_epsilon=pair_weight_epsilon,
    )


@dataclass(frozen=True)
class ThreeConstituentOneDomainForms:
    jacobian_form: object
    residual_form: object
    r_alpha: object
    r_mass_f: object
    r_mass_p: object
    r_mass_s: object
    r_momentum_f: object
    r_momentum_p: object
    r_momentum_s: object
    r_kinematics: object
    r_gamma: object
    r_inactive_extension: object
    r_total_mass: object
    r_total_momentum: object
    r_mass_terms: dict[str, object]
    r_momentum_terms: dict[str, object]
    r_internal_force_terms: dict[str, object]
    r_kinematics_terms: dict[str, object]
    a_terms: dict[str, object]


def build_three_constituent_one_domain_forms(
    *,
    v_f_k,
    p_f_k,
    v_p_k,
    p_p_k,
    v_s_k,
    u_s_k,
    alpha_k,
    phi_k,
    Gamma_k,
    v_f_n,
    v_p_n,
    v_s_n,
    u_s_n,
    alpha_n,
    phi_n,
    Gamma_n=None,
    dv_f,
    dp_f,
    dv_p,
    dp_p,
    dv_s,
    du_s,
    dalpha,
    dphi,
    dGamma,
    w_f,
    q_f,
    w_p,
    q_p,
    w_s,
    q_s,
    z_u,
    z_alpha,
    z_Gamma,
    dx,
    dt,
    rho_f=1.0,
    rho_p=1.0,
    rho_s=1.0,
    mu_f=1.0,
    mu_p=0.0,
    mu_s=1.0,
    lambda_s=1.0,
    R_fp=0.0,
    R_fs=0.0,
    R_ps=1.0,
    R_pair_cholesky=None,
    pair_weight_epsilon=0.0,
    theta_fp=0.5,
    ell_Gamma=0.0,
    gamma_mobility: str = "FP",
    gamma_delta_epsilon: float = 1.0e-12,
    transfer_velocity: str = "average",
    lag_alpha_in_constitutive_laws: bool = False,
    inactive_velocity_extension_factor: float = 0.0,
    inactive_pressure_extension_factor: float = 0.0,
    inactive_phi_extension_factor: float = 0.0,
    inactive_displacement_extension_factor: float = 0.0,
    phi_extension_value: float = 0.18,
    b_f=None,
    b_p=None,
    b_s=None,
    S_alpha=None,
    S_mass_f=None,
    S_mass_p=None,
    S_mass_s=None,
    S_momentum_f=None,
    S_momentum_p=None,
    S_momentum_s=None,
    S_kinematics=None,
    S_Gamma=None,
    dim: int = 2,
) -> ThreeConstituentOneDomainForms:
    """Build residual and consistent Jacobian for the canonical model.

    Boundary traction terms are intentionally not included in this base volume
    builder.  The constituent mass rows are assembled in strong divergence form
    so open inflow/outflow boundaries can be imposed through the velocity and
    pressure boundary conditions without a missing integrated-by-parts flux term.

    When ``lag_alpha_in_constitutive_laws`` is true, the coupled contents,
    stresses, transfer mobility, and internal-force coefficients use
    ``alpha_n`` while the alpha transport row still solves the bounded
    ``alpha_k`` update.  This keeps only the geometry/support field lagged;
    ``phi_k`` remains current.
    """

    dim = int(dim)
    dt_c = _as_named_expr("tc_dt", dt)
    inv_dt = _lit(1.0) / dt_c
    rho_f_c = _as_named_expr("tc_rho_f", rho_f)
    rho_p_c = _as_named_expr("tc_rho_p", rho_p)
    rho_s_c = _as_named_expr("tc_rho_s", rho_s)
    mu_f_c = _as_named_expr("tc_mu_f", mu_f)
    mu_p_c = _as_named_expr("tc_mu_p", mu_p)
    mu_s_c = _as_named_expr("tc_mu_s", mu_s)
    lambda_s_c = _as_named_expr("tc_lambda_s", lambda_s)
    R_fp_c = _as_named_expr("tc_R_fp", R_fp)
    R_fs_c = _as_named_expr("tc_R_fs", R_fs)
    R_ps_c = _as_named_expr("tc_R_ps", R_ps)
    theta_fp_c = _as_named_expr("tc_theta_fp", theta_fp)
    ell_Gamma_c = _as_named_expr("tc_ell_Gamma", ell_Gamma)
    gamma_delta_epsilon_sq_c = _as_named_expr(
        "tc_gamma_delta_epsilon_sq",
        float(gamma_delta_epsilon) * float(gamma_delta_epsilon),
    )
    inactive_velocity_extension_factor_c = _as_named_expr(
        "tc_inactive_velocity_extension_factor",
        inactive_velocity_extension_factor,
    )
    inactive_pressure_extension_factor_c = _as_named_expr(
        "tc_inactive_pressure_extension_factor",
        inactive_pressure_extension_factor,
    )
    inactive_phi_extension_factor_c = _as_named_expr(
        "tc_inactive_phi_extension_factor",
        inactive_phi_extension_factor,
    )
    inactive_displacement_extension_factor_c = _as_named_expr(
        "tc_inactive_displacement_extension_factor",
        inactive_displacement_extension_factor,
    )
    phi_extension_value_c = _as_named_expr("tc_phi_extension_value", phi_extension_value)

    zero = _lit(0.0)
    zero_vec_value = tuple(0.0 for _ in range(dim))
    b_f_expr = _as_named_expr("tc_b_f", zero_vec_value if b_f is None else b_f, dim=1)
    b_p_expr = _as_named_expr("tc_b_p", zero_vec_value if b_p is None else b_p, dim=1)
    b_s_expr = _as_named_expr("tc_b_s", zero_vec_value if b_s is None else b_s, dim=1)
    S_alpha_expr = _as_named_expr("tc_S_alpha", 0.0 if S_alpha is None else S_alpha)
    S_mass_f_expr = _as_named_expr("tc_S_mass_f", 0.0 if S_mass_f is None else S_mass_f)
    S_mass_p_expr = _as_named_expr("tc_S_mass_p", 0.0 if S_mass_p is None else S_mass_p)
    S_mass_s_expr = _as_named_expr("tc_S_mass_s", 0.0 if S_mass_s is None else S_mass_s)
    S_mom_f_expr = _as_named_expr("tc_S_momentum_f", zero_vec_value if S_momentum_f is None else S_momentum_f, dim=1)
    S_mom_p_expr = _as_named_expr("tc_S_momentum_p", zero_vec_value if S_momentum_p is None else S_momentum_p, dim=1)
    S_mom_s_expr = _as_named_expr("tc_S_momentum_s", zero_vec_value if S_momentum_s is None else S_momentum_s, dim=1)
    S_kin_expr = _as_named_expr("tc_S_kinematics", zero_vec_value if S_kinematics is None else S_kinematics, dim=1)
    S_Gamma_expr = _as_named_expr("tc_S_Gamma", 0.0 if S_Gamma is None else S_Gamma)

    alpha_model_k = alpha_n if bool(lag_alpha_in_constitutive_laws) else alpha_k

    F_k, P_k, B_k = one_domain_contents(alpha_model_k, phi_k)
    F_n, P_n, B_n = one_domain_contents(alpha_n, phi_n)

    r_f_k = F_k * rho_f_c
    r_p_k = P_k * rho_p_c
    r_s_k = B_k * rho_s_c
    r_f_n = F_n * rho_f_c
    r_p_n = P_n * rho_p_c
    r_s_n = B_n * rho_s_c

    sigma_f_k = linear_fluid_stress(v_f_k, p_f_k, mu=mu_f_c, dim=dim)
    sigma_p_k = linear_fluid_stress(v_p_k, p_p_k, mu=mu_p_c, dim=dim)
    sigma_s_k = linear_skeleton_stress(u_s_k, mu=mu_s_c, lambda_=lambda_s_c, dim=dim)

    force_terms = pairwise_internal_forces(
        alpha=alpha_model_k,
        phi=phi_k,
        v_f=v_f_k,
        v_p=v_p_k,
        v_s=v_s_k,
        sigma_f=sigma_f_k,
        sigma_p=sigma_p_k,
        sigma_s=sigma_s_k,
        R_fp=R_fp_c,
        R_fs=R_fs_c,
        R_ps=R_ps_c,
        R_pair_cholesky=R_pair_cholesky,
        pair_weight_epsilon=pair_weight_epsilon,
        theta_fp=theta_fp_c,
    )

    gamma_mobility_key = str(gamma_mobility or "FP").strip().lower().replace("-", "_")
    if gamma_mobility_key in {"fp", "overlap", "volume_overlap"}:
        L_Gamma = ell_Gamma_c * F_k * P_k
    elif gamma_mobility_key in {"interface_delta", "grad_alpha", "phi_grad_alpha", "delta"}:
        delta_alpha = _pair_weight(
            dot(grad(alpha_model_k), grad(alpha_model_k)),
            gamma_delta_epsilon_sq_c,
            epsilon_name="tc_gamma_delta_epsilon_sq",
        )
        L_Gamma = ell_Gamma_c * phi_k * delta_alpha
    elif gamma_mobility_key in {"off", "none", "zero"}:
        L_Gamma = zero
    else:
        raise ValueError("Unsupported gamma_mobility. Use 'FP', 'interface_delta', or 'off'.")

    A_Gamma = _sub(p_f_k, p_p_k) / rho_f_c
    transfer_velocity_key = str(transfer_velocity or "average").strip().lower().replace("-", "_")
    if transfer_velocity_key in {"average", "avg", "midpoint", "half_sum"}:
        u_Gamma = _lit(0.5) * (v_f_k + v_p_k)
    elif transfer_velocity_key in {"free", "vf", "v_f"}:
        u_Gamma = v_f_k
    elif transfer_velocity_key in {"pore", "vp", "v_p"}:
        u_Gamma = v_p_k
    else:
        raise ValueError("Unsupported transfer_velocity. Use 'average', 'free', or 'pore'.")

    # Alpha is a skeleton-carried geometry/support field.
    r_alpha = z_alpha * (_sub(alpha_k, alpha_n) * inv_dt + dot(grad(alpha_k), v_s_k) + _neg(S_alpha_expr)) * dx

    grad_r_f_k = rho_f_c * _neg(force_terms["g_fp"] + force_terms["g_fs"])
    grad_r_p_k = rho_p_c * (force_terms["g_fp"] + force_terms["g_ps"])
    grad_r_s_k = rho_s_c * (force_terms["g_fs"] + _neg(force_terms["g_ps"]))

    def _mass_divergence(r_a, grad_r_a, v_a):
        return r_a * div(v_a) + dot(grad_r_a, v_a)

    r_mass_f = q_f * (
        _sub(r_f_k, r_f_n) * inv_dt
        + _mass_divergence(r_f_k, grad_r_f_k, v_f_k)
        + Gamma_k
        + _neg(S_mass_f_expr)
    ) * dx
    r_mass_p = q_p * (
        _sub(r_p_k, r_p_n) * inv_dt
        + _mass_divergence(r_p_k, grad_r_p_k, v_p_k)
        + _neg(Gamma_k)
        + _neg(S_mass_p_expr)
    ) * dx
    r_mass_s = q_s * (
        _sub(r_s_k, r_s_n) * inv_dt
        + _mass_divergence(r_s_k, grad_r_s_k, v_s_k)
        + _neg(S_mass_s_expr)
    ) * dx

    def _momentum_row(w, c_a, r_a_k, r_a_n, v_a_k, v_a_n, sigma_a, body_a, I_a, M_gamma_a, S_mom_a):
        transient = dot(_sub(r_a_k * v_a_k, r_a_n * v_a_n) * inv_dt, w) * dx
        convection = _neg(r_a_k * dot(dot(grad(w), v_a_k), v_a_k)) * dx
        stress = c_a * inner(sigma_a, grad(w)) * dx
        sources = _neg(dot(r_a_k * body_a + I_a + M_gamma_a + S_mom_a, w)) * dx
        return transient + convection + stress + sources

    M_gamma_f = _neg(Gamma_k * u_Gamma)
    M_gamma_p = Gamma_k * u_Gamma
    M_gamma_s = zero * v_s_k

    r_momentum_f = _momentum_row(
        w_f,
        F_k,
        r_f_k,
        r_f_n,
        v_f_k,
        v_f_n,
        sigma_f_k,
        b_f_expr,
        force_terms["I_f"],
        M_gamma_f,
        S_mom_f_expr,
    )
    r_momentum_p = _momentum_row(
        w_p,
        P_k,
        r_p_k,
        r_p_n,
        v_p_k,
        v_p_n,
        sigma_p_k,
        b_p_expr,
        force_terms["I_p"],
        M_gamma_p,
        S_mom_p_expr,
    )
    r_momentum_s = _momentum_row(
        w_s,
        B_k,
        r_s_k,
        r_s_n,
        v_s_k,
        v_s_n,
        sigma_s_k,
        b_s_expr,
        force_terms["I_s"],
        M_gamma_s,
        S_mom_s_expr,
    )

    kin_residual = _sub(u_s_k, u_s_n) * inv_dt + dot(grad(u_s_k), v_s_k) + _neg(v_s_k) + _neg(S_kin_expr)
    r_kinematics = B_k * dot(kin_residual, z_u) * dx

    if Gamma_n is not None:
        # Gamma_n is accepted to keep time-state call sites symmetric.  The
        # current backward-Euler closure is algebraic in the k-state.
        _ = Gamma_n
    r_gamma = z_Gamma * (_sub(Gamma_k, L_Gamma * A_Gamma) + _neg(S_Gamma_expr)) * dx

    # Inactive-phase extension: v_s, v_p, p_p, u_s, and phi are not physically
    # defined where alpha -> 0.  These optional terms vanish rapidly in the
    # porous body and provide a controlled extension on the free-fluid side so
    # null modes cannot pollute alpha/phi transport.
    F_ext = F_k * F_k * F_k * F_k
    k_v_ext = inactive_velocity_extension_factor_c * R_ps_c
    k_p_ext = inactive_pressure_extension_factor_c * ell_Gamma_c / rho_f_c
    k_phi_ext = inactive_phi_extension_factor_c * rho_s_c * inv_dt
    k_u_ext = inactive_displacement_extension_factor_c * rho_s_c * inv_dt
    r_inactive_extension = F_ext * (
        k_v_ext * dot(_sub(v_s_k, v_f_k), w_s)
        + k_v_ext * dot(_sub(v_p_k, v_f_k), w_p)
        + k_p_ext * _sub(p_p_k, p_f_k) * q_p
        + k_phi_ext * _sub(phi_k, phi_extension_value_c) * q_s
        + k_u_ext * dot(_sub(u_s_k, u_s_n), z_u)
    ) * dx

    r_total_mass = r_mass_f + r_mass_p + r_mass_s
    r_total_momentum = r_momentum_f + r_momentum_p + r_momentum_s
    residual_form = (
        r_alpha
        + r_total_mass
        + r_total_momentum
        + r_kinematics
        + r_gamma
        + r_inactive_extension
    )

    coefficients = [v_f_k, p_f_k, v_p_k, p_p_k, v_s_k, u_s_k, alpha_k, phi_k, Gamma_k]
    directions = [dv_f, dp_f, dv_p, dp_p, dv_s, du_s, dalpha, dphi, dGamma]
    jacobian_form = linearize_form(residual_form, coefficients, directions)

    return ThreeConstituentOneDomainForms(
        jacobian_form=jacobian_form,
        residual_form=residual_form,
        r_alpha=r_alpha,
        r_mass_f=r_mass_f,
        r_mass_p=r_mass_p,
        r_mass_s=r_mass_s,
        r_momentum_f=r_momentum_f,
        r_momentum_p=r_momentum_p,
        r_momentum_s=r_momentum_s,
        r_kinematics=r_kinematics,
        r_gamma=r_gamma,
        r_inactive_extension=r_inactive_extension,
        r_total_mass=r_total_mass,
        r_total_momentum=r_total_momentum,
        r_mass_terms={
            "free": r_mass_f,
            "pore": r_mass_p,
            "solid": r_mass_s,
            "gamma_closure": r_gamma,
            "gamma_mobility": L_Gamma,
            "gamma_affinity": A_Gamma,
        },
        r_momentum_terms={
            "free": r_momentum_f,
            "pore": r_momentum_p,
            "solid": r_momentum_s,
        },
        r_internal_force_terms=force_terms,
        r_kinematics_terms={
            "eulerian_displacement_velocity": r_kinematics,
            "inactive_phase_extension": r_inactive_extension,
        },
        a_terms={
            "total": jacobian_form,
        },
    )


__all__ = [
    "ThreeConstituentOneDomainForms",
    "_named_c",
    "build_three_constituent_pdas_solver",
    "build_three_constituent_one_domain_forms",
    "configure_three_constituent_pdas_bounds",
    "diagonal_pairwise_internal_forces",
    "full_block_pairwise_internal_forces",
    "linear_fluid_stress",
    "linear_skeleton_stress",
    "one_domain_contents",
    "pairwise_internal_forces",
    "structural_gradients",
    "three_constituent_box_bounds",
]
