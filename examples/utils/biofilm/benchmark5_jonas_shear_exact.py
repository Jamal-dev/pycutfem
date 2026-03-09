"""Jonas-inspired exact shear benchmark for the reduced deformation-only model.

The benchmark keeps the conservative Cahn--Hilliard indicator active while
adding a known tangential interface traction on a flat interface. The geometry
is a unit square split by the horizontal interface y = y_interface:

  - a smooth tangential fluid shear acts across the diffuse interface,
  - a smooth elastic displacement is localized in the solid-rich region,
  - alpha is a stationary tanh profile centered on the interface,
  - the exact tangential traction is localized by the conserved alpha profile.

The exact fields are steady, smooth, and tailored to the reduced
Paper-1 mechanics block. Volume forcing is derived symbolically from the
reduced equations; the traction is supplied separately through a diffuse
interface weight so the
benchmark directly verifies the tangential stress-transfer path used by
Benchmark 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .mms_deformation_only import (
    _double_well_prime,
    _lambdify_mat_xy,
    _lambdify_scalar_xy,
    _lambdify_vec_xy,
    _sym_div_mat,
    _sym_div_vec,
    _sym_epsilon,
    _sym_grad_scalar,
    _sym_grad_vec,
    _sym_is_zero,
    _sym_laplacian,
)


@dataclass(frozen=True)
class JonasShearBenchmark:
    case_id: str
    title: str
    geometry: str
    interface_y: float
    dt: float
    theta: float
    params: dict[str, float]
    v: callable
    p: callable
    vS: callable
    u: callable
    alpha: callable
    mu_alpha: callable
    grad_v: callable
    grad_p: callable
    grad_vS: callable
    grad_u: callable
    grad_alpha: callable
    grad_mu_alpha: callable
    v_n: callable
    p_n: callable
    vS_n: callable
    u_n: callable
    alpha_n: callable
    mu_alpha_n: callable
    v_k: callable
    p_k: callable
    vS_k: callable
    u_k: callable
    alpha_k: callable
    mu_alpha_k: callable
    f_v: callable
    f_u: callable
    f_alpha: callable
    traction_weight: callable
    g_t: callable
    traction_fluid: callable
    traction_solid: callable


def build_jonas_shear_benchmark(
    *,
    dt_val: float = 0.05,
    theta: float = 1.0,
    y_interface: float = 0.5,
    wall_speed: float = 1.0,
    rho_f: float = 1.0,
    mu_f: float = 0.1,
    mu_b: float = 0.1,
    kappa_inv: float = 20.0,
    mu_s: float = 1.0,
    lambda_s: float = 0.0,
    phi_b: float = 0.5,
    M_alpha: float = 1.0,
    gamma_alpha: float = 1.0,
    eps_alpha: float = 0.06,
) -> JonasShearBenchmark:
    dt = float(dt_val)
    if not (dt > 0.0):
        raise ValueError("dt_val must be positive.")
    th = float(theta)
    if not (0.0 <= th <= 1.0):
        raise ValueError("theta must lie in [0,1].")
    yi = float(y_interface)
    if not (0.0 < yi < 1.0):
        raise ValueError("y_interface must lie strictly inside (0,1).")
    if float(eps_alpha) <= 0.0:
        raise ValueError("eps_alpha must be positive.")

    x, y = sp.symbols("x y", real=True)

    alpha_expr = sp.Rational(1, 2) * (1 + sp.tanh((y - sp.Float(yi)) / (sp.sqrt(2) * sp.Float(eps_alpha))))
    mu_expr = sp.Integer(0)

    alpha_fluid_expr = sp.simplify(1.0 - alpha_expr)
    v_x_expr = sp.simplify(sp.Float(wall_speed) * (1 - y / sp.Float(yi)) * alpha_fluid_expr)
    v_expr = sp.Matrix([v_x_expr, sp.Integer(0)])
    p_expr = sp.Integer(0)
    vS_expr = sp.Matrix([sp.Integer(0), sp.Integer(0)])

    u_interface = sp.Float(mu_f * wall_speed * (1.0 - yi) / (mu_s * yi))
    u_amplitude = sp.Float(2.0) * u_interface
    u_x_expr = sp.simplify(u_amplitude * ((1 - y) / sp.Float(1.0 - yi)) * alpha_expr)
    u_expr = sp.Matrix([u_x_expr, sp.Integer(0)])

    C_expr = sp.simplify(1.0 - alpha_expr * (1.0 - float(phi_b)))
    B_expr = sp.simplify(alpha_expr * (1.0 - float(phi_b)))
    rho_expr = sp.simplify(float(rho_f) * C_expr)
    mu_mix_expr = sp.simplify((1.0 - alpha_expr) * float(mu_f) + alpha_expr * float(mu_b))
    beta_expr = sp.simplify(alpha_expr * float(mu_f) * float(kappa_inv))

    grad_p_expr = _sym_grad_scalar(p_expr, x, y)
    eps_v_expr = _sym_epsilon(v_expr, x, y)
    sigma_u_expr = 2.0 * float(mu_s) * _sym_epsilon(u_expr, x, y) + float(lambda_s) * _sym_div_vec(u_expr, x, y) * sp.eye(2)

    mom_dt = sp.Matrix([sp.Integer(0), sp.Integer(0)])
    mom_conv = rho_expr * (_sym_grad_vec(v_expr, x, y) * v_expr)
    mom_visc = -_sym_div_mat(2.0 * mu_mix_expr * eps_v_expr, x, y)
    mom_press = C_expr * grad_p_expr
    mom_drag = beta_expr * (v_expr - vS_expr)
    traction_weight_expr = sp.simplify(sp.diff(alpha_expr, y))
    traction_vec_expr = sp.Matrix([-float(mu_f) * float(wall_speed) / float(yi), 0.0])
    f_v_expr = sp.simplify(mom_dt + mom_conv + mom_visc + mom_press + mom_drag - traction_weight_expr * traction_vec_expr)

    skel_expr = sp.simplify(-_sym_div_mat(alpha_expr * sigma_u_expr, x, y) + B_expr * grad_p_expr - beta_expr * (v_expr - vS_expr))
    f_u_expr = sp.simplify((skel_expr + traction_weight_expr * traction_vec_expr) / alpha_expr)

    alpha_adv = sp.Integer(0)
    f_alpha_expr = sp.simplify(alpha_adv - float(M_alpha) * _sym_laplacian(mu_expr, x, y))
    mu_residual_expr = sp.simplify(mu_expr - float(gamma_alpha) * ((-float(eps_alpha)) * _sym_laplacian(alpha_expr, x, y) + (_double_well_prime(alpha_expr) / float(eps_alpha))))

    mass_expr = sp.simplify(_sym_div_vec(C_expr * v_expr + B_expr * vS_expr, x, y))
    kin_expr = sp.simplify(_sym_grad_vec(u_expr, x, y) * vS_expr - vS_expr)
    if not _sym_is_zero(mass_expr):
        raise ValueError("Jonas shear benchmark does not satisfy the one-domain mass constraint exactly.")
    if any(not _sym_is_zero(comp) for comp in kin_expr):
        raise ValueError("Jonas shear benchmark does not satisfy the Eulerian kinematic constraint exactly.")
    if not _sym_is_zero(mu_residual_expr):
        raise ValueError("Jonas shear benchmark alpha profile is not an exact CH equilibrium.")
    if not _sym_is_zero(f_alpha_expr):
        raise ValueError("Jonas shear benchmark unexpectedly needs alpha forcing.")

    traction_fluid_expr = sp.Matrix([float(traction_vec_expr[0]), 0.0])
    traction_solid_expr = sp.Matrix([float(traction_vec_expr[0]), 0.0])

    params = {
        "rho_f": float(rho_f),
        "mu_f": float(mu_f),
        "mu_b": float(mu_b),
        "kappa_inv": float(kappa_inv),
        "mu_s": float(mu_s),
        "lambda_s": float(lambda_s),
        "phi_b": float(phi_b),
        "M_alpha": float(M_alpha),
        "gamma_alpha": float(gamma_alpha),
        "eps_alpha": float(eps_alpha),
        "wall_speed": float(wall_speed),
    }

    grad_v_expr = _sym_grad_vec(v_expr, x, y)
    grad_p_expr = _sym_grad_scalar(p_expr, x, y)
    grad_vS_expr = _sym_grad_vec(vS_expr, x, y)
    grad_u_expr = _sym_grad_vec(u_expr, x, y)
    grad_alpha_expr = _sym_grad_scalar(alpha_expr, x, y)
    grad_mu_expr = _sym_grad_scalar(mu_expr, x, y)

    return JonasShearBenchmark(
        case_id="jonas_shear",
        title="Jonas-inspired planar shear with exact tangential traction",
        geometry="unit square with horizontal interface y=0.5",
        interface_y=float(yi),
        dt=float(dt),
        theta=float(th),
        params=params,
        v=lambda xx, yy, _fn=_lambdify_vec_xy(v_expr, x, y): _fn(xx, yy),
        p=lambda xx, yy, _fn=_lambdify_scalar_xy(p_expr, x, y): _fn(xx, yy),
        vS=lambda xx, yy, _fn=_lambdify_vec_xy(vS_expr, x, y): _fn(xx, yy),
        u=lambda xx, yy, _fn=_lambdify_vec_xy(u_expr, x, y): _fn(xx, yy),
        alpha=lambda xx, yy, _fn=_lambdify_scalar_xy(alpha_expr, x, y): _fn(xx, yy),
        mu_alpha=lambda xx, yy, _fn=_lambdify_scalar_xy(mu_expr, x, y): _fn(xx, yy),
        grad_v=lambda xx, yy, _fn=_lambdify_mat_xy(grad_v_expr, x, y): _fn(xx, yy),
        grad_p=lambda xx, yy, _fn=_lambdify_vec_xy(grad_p_expr, x, y): _fn(xx, yy),
        grad_vS=lambda xx, yy, _fn=_lambdify_mat_xy(grad_vS_expr, x, y): _fn(xx, yy),
        grad_u=lambda xx, yy, _fn=_lambdify_mat_xy(grad_u_expr, x, y): _fn(xx, yy),
        grad_alpha=lambda xx, yy, _fn=_lambdify_vec_xy(grad_alpha_expr, x, y): _fn(xx, yy),
        grad_mu_alpha=lambda xx, yy, _fn=_lambdify_vec_xy(grad_mu_expr, x, y): _fn(xx, yy),
        v_n=lambda xx, yy, _fn=_lambdify_vec_xy(v_expr, x, y): _fn(xx, yy),
        p_n=lambda xx, yy, _fn=_lambdify_scalar_xy(p_expr, x, y): _fn(xx, yy),
        vS_n=lambda xx, yy, _fn=_lambdify_vec_xy(vS_expr, x, y): _fn(xx, yy),
        u_n=lambda xx, yy, _fn=_lambdify_vec_xy(u_expr, x, y): _fn(xx, yy),
        alpha_n=lambda xx, yy, _fn=_lambdify_scalar_xy(alpha_expr, x, y): _fn(xx, yy),
        mu_alpha_n=lambda xx, yy, _fn=_lambdify_scalar_xy(mu_expr, x, y): _fn(xx, yy),
        v_k=lambda xx, yy, _fn=_lambdify_vec_xy(v_expr, x, y): _fn(xx, yy),
        p_k=lambda xx, yy, _fn=_lambdify_scalar_xy(p_expr, x, y): _fn(xx, yy),
        vS_k=lambda xx, yy, _fn=_lambdify_vec_xy(vS_expr, x, y): _fn(xx, yy),
        u_k=lambda xx, yy, _fn=_lambdify_vec_xy(u_expr, x, y): _fn(xx, yy),
        alpha_k=lambda xx, yy, _fn=_lambdify_scalar_xy(alpha_expr, x, y): _fn(xx, yy),
        mu_alpha_k=lambda xx, yy, _fn=_lambdify_scalar_xy(mu_expr, x, y): _fn(xx, yy),
        f_v=lambda xx, yy, _fn=_lambdify_vec_xy(f_v_expr, x, y): _fn(xx, yy),
        f_u=lambda xx, yy, _fn=_lambdify_vec_xy(f_u_expr, x, y): _fn(xx, yy),
        f_alpha=lambda xx, yy, _fn=_lambdify_scalar_xy(f_alpha_expr, x, y): _fn(xx, yy),
        traction_weight=lambda xx, yy, _fn=_lambdify_scalar_xy(traction_weight_expr, x, y): _fn(xx, yy),
        g_t=lambda xx, yy, _fn=_lambdify_vec_xy(traction_vec_expr, x, y): _fn(xx, yy),
        traction_fluid=lambda xx, yy, _fn=_lambdify_vec_xy(traction_fluid_expr, x, y): _fn(xx, yy),
        traction_solid=lambda xx, yy, _fn=_lambdify_vec_xy(traction_solid_expr, x, y): _fn(xx, yy),
    )
