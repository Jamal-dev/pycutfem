"""Wall adhesion + shear-based degradation utilities for the one-domain biofilm model.

These helpers are intended for *lagged* (explicit) updates of an adhesion
integrity scalar ``a`` used in the skeleton wall traction term:

  t_adh = α a (k_n u_n n + k_t u_τ + γ_n vS_n n + γ_t vS_τ)

The wall shear proxy used to degrade ``a`` is based on the tangential traction
at the wall:

  t = 2 μ(α,φ) ε(v) n
  t_τ = t - (t·n) n
  τ_w = |t_τ|

For robustness and compiler simplicity, we assemble ``τ_w^2`` and take the
square-root in Python.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycutfem.ufl.expressions import Constant, FacetNormal, Identity, TestFunction, div, dot, grad, inner, trace
from pycutfem.ufl.forms import Equation, assemble_form


def _c(val: float) -> Constant:
    return Constant(float(val))


def _one_minus(expr):
    return (-expr) + _c(1.0)


def _epsilon(v):
    return _c(0.5) * (grad(v) + grad(v).T)


def mu_mixture(alpha, phi, *, mu_f, mu_b_model: str = "phi_mu"):
    mu_b_model = str(mu_b_model).strip().lower()
    if mu_b_model in {"mu", "const", "constant"}:
        mu_b = mu_f
    elif mu_b_model in {"phi_mu", "phi*mu", "phi"}:
        mu_b = phi * mu_f
    else:
        raise ValueError(f"Unknown mu_b_model {mu_b_model!r}.")
    return _one_minus(alpha) * mu_f + alpha * mu_b


@dataclass(frozen=True)
class WallShearRMS:
    tau_rms: float
    alpha_area: float
    alpha_tau2: float


@dataclass(frozen=True)
class AdhesionFieldUpdate:
    """Diagnostics returned by `update_adhesion_integrity_field_on_boundary`."""

    a_min: float
    a_max: float
    tau_min: float
    tau_max: float
    n_active: int


def assemble_scalar(dof_handler, functional, *, backend: str = "cpp", quad_order: int | None = None) -> float:
    """Assemble a pure functional (no test/trial) and return it as a float."""
    hooks = {functional.integrand: {"name": "val"}}
    res = assemble_form(Equation(None, functional), dof_handler=dof_handler, bcs=[], quad_order=quad_order, backend=backend, assembler_hooks=hooks)
    return float(res["val"])


def wall_shear_rms_on_boundary(
    *,
    dof_handler,
    v,
    alpha,
    phi,
    ds_wall,
    mu_f,
    mu_b_model: str = "phi_mu",
    backend: str = "cpp",
    quad_order: int | None = None,
    area_eps: float = 1.0e-14,
) -> WallShearRMS:
    """
    Compute an α-weighted RMS wall shear:

      τ_rms = sqrt( ∫ α τ_w^2 ds / (∫ α ds + area_eps) )
    """
    mu = mu_mixture(alpha, phi, mu_f=mu_f, mu_b_model=mu_b_model)
    n = FacetNormal()

    # Tangential traction vector: t_τ = t - (t·n)n,  with t = 2 μ ε(v) n.
    t = _c(2.0) * mu * dot(_epsilon(v), n)
    t_t = t - dot(t, n) * n
    tau2 = inner(t_t, t_t)

    I_alpha = (alpha) * ds_wall
    I_alpha_tau2 = (alpha * tau2) * ds_wall

    alpha_area = assemble_scalar(dof_handler, I_alpha, backend=backend, quad_order=quad_order)
    alpha_tau2 = assemble_scalar(dof_handler, I_alpha_tau2, backend=backend, quad_order=quad_order)

    tau_rms = float(np.sqrt(max(0.0, alpha_tau2) / (float(alpha_area) + float(area_eps))))
    return WallShearRMS(tau_rms=tau_rms, alpha_area=float(alpha_area), alpha_tau2=float(alpha_tau2))


def wall_shear_mass_lumped_on_boundary(
    *,
    dof_handler,
    field: str,
    v,
    alpha,
    phi,
    ds_wall,
    mu_f,
    mu_b_model: str = "phi_mu",
    backend: str = "cpp",
    quad_order: int | None = None,
    area_eps: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a nodal (DOF-wise) wall shear magnitude on `ds_wall` by a mass-lumped projection.

    We assemble (with `ψ` a scalar test function on the chosen `field`):

      w_i   = ∫ ψ_i α ds,
      rhs_i = ∫ ψ_i α τ_w^2 ds,
      τ_i   = sqrt(rhs_i / (w_i + area_eps)).

    Returns
    -------
    tau : ndarray
        Array of τ_i values in the DOF ordering of `dof_handler.get_field_slice(field)`.
    weight : ndarray
        Array of w_i values in the same ordering (useful for masking active DOFs).
    """
    mu = mu_mixture(alpha, phi, mu_f=mu_f, mu_b_model=mu_b_model)
    n = FacetNormal()
    t = _c(2.0) * mu * dot(_epsilon(v), n)
    t_t = t - dot(t, n) * n
    tau2 = inner(t_t, t_t)

    psi = TestFunction(field, dof_handler=dof_handler)
    w_form = (psi * alpha) * ds_wall
    rhs_form = (psi * alpha * tau2) * ds_wall

    _, w_vec = assemble_form(Equation(None, w_form), dof_handler=dof_handler, bcs=[], quad_order=quad_order, backend=backend)
    _, rhs_vec = assemble_form(Equation(None, rhs_form), dof_handler=dof_handler, bcs=[], quad_order=quad_order, backend=backend)

    sl = np.asarray(dof_handler.get_field_slice(field), dtype=int)
    w = np.asarray(w_vec[sl], dtype=float)
    rhs = np.asarray(rhs_vec[sl], dtype=float)

    tau = np.zeros_like(w)
    mask = w > float(area_eps)
    if np.any(mask):
        tau2_lumped = np.maximum(0.0, rhs[mask] / (w[mask] + float(area_eps)))
        tau[mask] = np.sqrt(tau2_lumped)
    return tau, w


def solid_von_mises_mass_lumped_on_boundary_linear(
    *,
    dof_handler,
    field: str,
    u,
    alpha,
    ds_wall,
    mu_s,
    lambda_s,
    backend: str = "cpp",
    quad_order: int | None = None,
    area_eps: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Mass-lumped von Mises stress on a boundary for a *linear* solid.

    Computes nodal values by assembling:
      w_i   = ∫ ψ_i α ds,
      rhs_i = ∫ ψ_i α σ_vm^2 ds,
      σ_i   = sqrt(rhs_i / (w_i + area_eps)).

    The stress is based on the elastic Cauchy stress:
      σ(u) = 2 μ_s ε(u) + λ_s (div u) I,
      σ_vm = sqrt( 3/2 dev(σ):dev(σ) ).
    """
    mu_s = mu_s if hasattr(mu_s, "dim") else _c(float(mu_s))
    lambda_s = lambda_s if hasattr(lambda_s, "dim") else _c(float(lambda_s))

    eps_u = _c(0.5) * (grad(u) + grad(u).T)
    sig = _c(2.0) * mu_s * eps_u + lambda_s * div(u) * Identity(2)
    tr_sig = trace(sig)
    s_dev = sig - (tr_sig / _c(2.0)) * Identity(2)
    vm2 = _c(1.5) * inner(s_dev, s_dev)

    psi = TestFunction(field, dof_handler=dof_handler)
    w_form = (psi * alpha) * ds_wall
    rhs_form = (psi * alpha * vm2) * ds_wall

    _, w_vec = assemble_form(Equation(None, w_form), dof_handler=dof_handler, bcs=[], quad_order=quad_order, backend=backend)
    _, rhs_vec = assemble_form(Equation(None, rhs_form), dof_handler=dof_handler, bcs=[], quad_order=quad_order, backend=backend)

    sl = np.asarray(dof_handler.get_field_slice(field), dtype=int)
    w = np.asarray(w_vec[sl], dtype=float)
    rhs = np.asarray(rhs_vec[sl], dtype=float)

    sigma = np.zeros_like(w)
    mask = w > float(area_eps)
    if np.any(mask):
        vm2_lumped = np.maximum(0.0, rhs[mask] / (w[mask] + float(area_eps)))
        sigma[mask] = np.sqrt(vm2_lumped)
    return sigma, w


def solid_von_mises_mass_lumped_in_domain(
    *,
    dof_handler,
    field: str,
    u,
    alpha,
    dx_domain,
    mu_s,
    lambda_s,
    solid_model: str = "linear",
    backend: str = "cpp",
    quad_order: int | None = None,
    area_eps: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Mass-lumped von Mises stress in the domain (α-weighted projection).

    Computes nodal values by assembling:
      w_i   = ∫ ψ_i α dx,
      rhs_i = ∫ ψ_i α σ_vm^2 dx,
      σ_i   = sqrt(rhs_i / (w_i + area_eps)).

    The projection is performed in the scalar FE space given by `field`
    (typically a Q2 scalar, e.g. ``field="u_x"`` for visualization on a Q2 mesh).
    """
    mu_s = mu_s if hasattr(mu_s, "dim") else _c(float(mu_s))
    lambda_s = lambda_s if hasattr(lambda_s, "dim") else _c(float(lambda_s))

    solid_model_key = str(solid_model).strip().lower()
    if solid_model_key in {"neo-hookean", "neo_hookean", "nh"}:
        from pycutfem.utils.nonlinear_solid_eulerian_refmap import sigma_neo_hookean

        c_nh = mu_s / _c(2.0)
        beta_nh = lambda_s / (_c(2.0) * mu_s)
        sig = sigma_neo_hookean(u, c_nh, beta_nh, dim=2)
    else:
        eps_u = _c(0.5) * (grad(u) + grad(u).T)
        sig = _c(2.0) * mu_s * eps_u + lambda_s * div(u) * Identity(2)

    tr_sig = trace(sig)
    s_dev = sig - (tr_sig / _c(2.0)) * Identity(2)
    vm2 = _c(1.5) * inner(s_dev, s_dev)

    psi = TestFunction(field, dof_handler=dof_handler)
    w_form = (psi * alpha) * dx_domain
    rhs_form = (psi * alpha * vm2) * dx_domain

    _, w_vec = assemble_form(Equation(None, w_form), dof_handler=dof_handler, bcs=[], quad_order=quad_order, backend=backend)
    _, rhs_vec = assemble_form(Equation(None, rhs_form), dof_handler=dof_handler, bcs=[], quad_order=quad_order, backend=backend)

    sl = np.asarray(dof_handler.get_field_slice(field), dtype=int)
    w = np.asarray(w_vec[sl], dtype=float)
    rhs = np.asarray(rhs_vec[sl], dtype=float)

    sigma = np.zeros_like(w)
    mask = w > float(area_eps)
    if np.any(mask):
        vm2_lumped = np.maximum(0.0, rhs[mask] / (w[mask] + float(area_eps)))
        sigma[mask] = np.sqrt(vm2_lumped)
    return sigma, w


def update_adhesion_integrity_field_on_boundary_von_mises(
    *,
    dof_handler,
    a_field,
    dt: float,
    u,
    alpha,
    ds_wall,
    mu_s,
    lambda_s,
    k_break: float = 0.0,
    sigma_cr: float,
    m: float = 1.0,
    a_snap: float = 0.0,
    backend: str = "cpp",
    quad_order: int | None = None,
    area_eps: float = 1.0e-14,
) -> AdhesionFieldUpdate:
    """Wall adhesion degradation driven by solid von Mises stress (lagged).

    If ``k_break <= 0`` this uses a *binary* paper-style criterion:

      if σ_vm(s) > σ_cr  ⇒  a(s) := 0   (irreversible).

    If ``k_break > 0`` it uses a smoother irreversible rate law:

      a^{n+1}(s) = a^n(s) exp(-dt k_break ⟨σ_vm/σ_cr - 1⟩^m),

    where ⟨·⟩ is the positive part (only degrades when the threshold is exceeded).
    """
    sigma_cr = float(sigma_cr)
    if sigma_cr <= 0.0:
        raise ValueError("sigma_cr must be positive.")

    sigma, w = solid_von_mises_mass_lumped_on_boundary_linear(
        dof_handler=dof_handler,
        field=str(getattr(a_field, "field_name", "a")),
        u=u,
        alpha=alpha,
        ds_wall=ds_wall,
        mu_s=mu_s,
        lambda_s=lambda_s,
        backend=backend,
        quad_order=quad_order,
        area_eps=area_eps,
    )

    a_vals = np.asarray(a_field.nodal_values, dtype=float)
    if a_vals.shape != sigma.shape:
        raise ValueError(
            f"a_field.nodal_values has shape {a_vals.shape}, but sigma array has shape {sigma.shape} "
            f"(field={getattr(a_field, 'field_name', None)!r})."
        )

    active = w > float(area_eps)
    dt = float(dt)
    k_break = float(k_break)
    m = float(m)
    a_snap = float(a_snap)

    if k_break <= 0.0 or dt <= 0.0:
        # Binary failure.
        failed = active & (sigma > sigma_cr)
        if np.any(failed):
            a_vals[failed] = 0.0
    else:
        # Smooth irreversible degradation.
        ratio = np.maximum(0.0, (sigma / sigma_cr) - 1.0)
        rate = k_break * (ratio**m)
        if np.any(active):
            a_new = a_vals.copy()
            a_new[active] = np.clip(a_vals[active] * np.exp(-dt * rate[active]), 0.0, 1.0)
            a_vals = a_new
    if a_snap > 0.0 and np.any(active):
        snapped = active & (a_vals < a_snap)
        if np.any(snapped):
            a_vals[snapped] = 0.0
    a_vals[:] = np.clip(a_vals, 0.0, 1.0)
    a_field.nodal_values[:] = a_vals

    sigma_active = sigma[active] if np.any(active) else np.asarray([], dtype=float)
    return AdhesionFieldUpdate(
        a_min=float(np.min(a_vals)) if a_vals.size else 0.0,
        a_max=float(np.max(a_vals)) if a_vals.size else 0.0,
        tau_min=float(np.min(sigma_active)) if sigma_active.size else 0.0,
        tau_max=float(np.max(sigma_active)) if sigma_active.size else 0.0,
        n_active=int(np.count_nonzero(active)),
    )


def update_adhesion_integrity_field_on_boundary(
    *,
    dof_handler,
    a_field,
    dt: float,
    v,
    alpha,
    phi,
    ds_wall,
    mu_f,
    k_break: float,
    tau_c: float,
    m: float = 1.0,
    a_snap: float = 0.0,
    mu_b_model: str = "phi_mu",
    backend: str = "cpp",
    quad_order: int | None = None,
    area_eps: float = 1.0e-14,
) -> AdhesionFieldUpdate:
    """
    Update a spatially varying adhesion integrity field `a` on the wall (explicit/lagged).

      a^{n+1}(s) = clip_{[0,1]}( a^n(s) exp(-dt k_break (τ_w(s)/τ_c)^m ) )

    where τ_w(s) is computed nodally on `ds_wall` by a mass-lumped projection.
    The update is only applied to DOFs with non-negligible α-weighted boundary weight.
    """
    dt = float(dt)
    if dt <= 0.0:
        vals = np.asarray(a_field.nodal_values, dtype=float)
        return AdhesionFieldUpdate(
            a_min=float(np.min(vals)) if vals.size else 0.0,
            a_max=float(np.max(vals)) if vals.size else 0.0,
            tau_min=0.0,
            tau_max=0.0,
            n_active=0,
        )

    k_break = float(k_break)
    if k_break <= 0.0:
        vals = np.asarray(a_field.nodal_values, dtype=float)
        return AdhesionFieldUpdate(
            a_min=float(np.min(vals)) if vals.size else 0.0,
            a_max=float(np.max(vals)) if vals.size else 0.0,
            tau_min=0.0,
            tau_max=0.0,
            n_active=0,
        )

    tau_c = float(tau_c)
    if tau_c <= 0.0:
        raise ValueError("tau_c must be positive.")

    m = float(m)
    a_snap = float(a_snap)
    tau, w = wall_shear_mass_lumped_on_boundary(
        dof_handler=dof_handler,
        field=str(getattr(a_field, "field_name", "a")),
        v=v,
        alpha=alpha,
        phi=phi,
        ds_wall=ds_wall,
        mu_f=mu_f,
        mu_b_model=mu_b_model,
        backend=backend,
        quad_order=quad_order,
        area_eps=area_eps,
    )

    a_vals = np.asarray(a_field.nodal_values, dtype=float)
    if a_vals.shape != tau.shape:
        raise ValueError(
            f"a_field.nodal_values has shape {a_vals.shape}, but wall shear array has shape {tau.shape} "
            f"(field={getattr(a_field, 'field_name', None)!r})."
        )

    active = w > float(area_eps)
    n_active = int(np.count_nonzero(active))
    if n_active:
        rate = k_break * (np.maximum(0.0, tau[active]) / tau_c) ** m
        a_new = a_vals.copy()
        a_new[active] = np.clip(a_vals[active] * np.exp(-dt * rate), 0.0, 1.0)
        if a_snap > 0.0:
            snap_mask = a_new[active] < a_snap
            if np.any(snap_mask):
                a_new[active] = np.where(snap_mask, 0.0, a_new[active])
        a_field.nodal_values[:] = a_new
        tau_active = tau[active]
        a_active = a_new[active]
        return AdhesionFieldUpdate(
            a_min=float(np.min(a_active)),
            a_max=float(np.max(a_active)),
            tau_min=float(np.min(tau_active)),
            tau_max=float(np.max(tau_active)),
            n_active=n_active,
        )

    # Nothing active on the wall (no biofilm contact).
    return AdhesionFieldUpdate(
        a_min=float(np.min(a_vals)) if a_vals.size else 0.0,
        a_max=float(np.max(a_vals)) if a_vals.size else 0.0,
        tau_min=0.0,
        tau_max=0.0,
        n_active=0,
    )


def update_adhesion_integrity(
    *,
    a_n: float,
    dt: float,
    tau_rms: float,
    k_break: float,
    tau_c: float,
    m: float = 1.0,
) -> float:
    """
    Explicit degradation update:

      a^{n+1} = a^n exp(-dt k_break (tau_rms/tau_c)^m)
    """
    a_n = float(a_n)
    if a_n <= 0.0:
        return 0.0
    dt = float(dt)
    if dt <= 0.0:
        return float(np.clip(a_n, 0.0, 1.0))
    k_break = float(k_break)
    if k_break <= 0.0:
        return float(np.clip(a_n, 0.0, 1.0))
    tau_c = float(tau_c)
    if tau_c <= 0.0:
        raise ValueError("tau_c must be positive.")
    tau_rms = max(0.0, float(tau_rms))
    m = float(m)
    rate = k_break * (tau_rms / tau_c) ** m
    return float(np.clip(a_n * float(np.exp(-dt * rate)), 0.0, 1.0))
