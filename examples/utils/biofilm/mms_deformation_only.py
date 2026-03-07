"""Manufactured-solution utilities for the reduced deformation-only model.

The reduced Paper-1 model keeps only:
  - fluid momentum,
  - one-domain mass constraint,
  - skeleton momentum,
  - Eulerian kinematics,
  - conservative Cahn--Hilliard transport for alpha.

This module builds manufactured fields together with forcing terms for a single
one-step-theta update of `examples.utils.biofilm.deformation_only`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp


def _sym_grad_vec(v, x, y):
    return sp.Matrix([[sp.diff(v[i], var) for var in (x, y)] for i in range(2)])


def _sym_grad_scalar(f, x, y):
    return sp.Matrix([sp.diff(f, x), sp.diff(f, y)])


def _sym_div_vec(v, x, y):
    return sp.diff(v[0], x) + sp.diff(v[1], y)


def _sym_div_mat(M, x, y):
    return sp.Matrix([sp.diff(M[0, 0], x) + sp.diff(M[0, 1], y), sp.diff(M[1, 0], x) + sp.diff(M[1, 1], y)])


def _sym_laplacian(f, x, y):
    return sp.diff(f, x, 2) + sp.diff(f, y, 2)


def _sym_epsilon(v, x, y):
    G = _sym_grad_vec(v, x, y)
    return sp.Rational(1, 2) * (G + G.T)


def _sym_is_zero(expr, *, tol: float = 1.0e-10) -> bool:
    simp = sp.simplify(sp.expand(expr))
    if simp == 0:
        return True
    try:
        if bool(simp.equals(0)):
            return True
    except Exception:
        pass
    free = sorted(simp.free_symbols, key=lambda s: s.name)
    if not free:
        try:
            return abs(complex(sp.N(simp, 30))) <= float(tol)
        except Exception:
            return False
    sample_sets = (
        (0.17, 0.29, 0.11),
        (0.31, 0.43, 0.19),
        (0.57, 0.61, 0.07),
    )
    for sample in sample_sets:
        subs = {sym: float(sample[min(i, len(sample) - 1)]) for i, sym in enumerate(free)}
        try:
            val = complex(sp.N(simp.subs(subs), 30))
        except Exception:
            return False
        if abs(val) > float(tol):
            return False
    return True


def _lambdify_scalar_xy(expr, x, y):
    fn = sp.lambdify((x, y), expr, "numpy")

    def _call(xv, yv, _fn=fn):
        return np.asarray(_fn(xv, yv), dtype=float)

    return _call


def _lambdify_vec_xy(expr_vec, x, y):
    comp_fns = [sp.lambdify((x, y), expr_vec[i], "numpy") for i in range(int(expr_vec.rows))]

    def _call(xv, yv, _fns=comp_fns):
        xv = np.asarray(xv, dtype=float)
        yv = np.asarray(yv, dtype=float)
        comps = [np.asarray(fn(xv, yv), dtype=float) for fn in _fns]
        shape = np.broadcast(xv, yv, *comps).shape
        comps = [np.broadcast_to(comp, shape) for comp in comps]
        return np.stack(comps, axis=-1)

    return _call


def _lambdify_mat_xy(expr_mat, x, y):
    nrows = int(expr_mat.rows)
    ncols = int(expr_mat.cols)
    comp_fns = [[sp.lambdify((x, y), expr_mat[i, j], "numpy") for j in range(ncols)] for i in range(nrows)]

    def _call(xv, yv, _fns=comp_fns):
        xv = np.asarray(xv, dtype=float)
        yv = np.asarray(yv, dtype=float)
        vals = [[np.asarray(fn(xv, yv), dtype=float) for fn in row] for row in _fns]
        flat = [val for row in vals for val in row]
        shape = np.broadcast(xv, yv, *flat).shape
        out = np.empty(shape + (nrows, ncols), dtype=float)
        for i in range(nrows):
            for j in range(ncols):
                out[..., i, j] = np.broadcast_to(vals[i][j], shape)
        return out

    return _call


def _lambdify_scalar_xyt(expr, x, y, t):
    fn = sp.lambdify((x, y, t), expr, "numpy")

    def _call(xv, yv, tv, _fn=fn):
        return np.asarray(_fn(xv, yv, tv), dtype=float)

    return _call


def _lambdify_vec_xyt(expr_vec, x, y, t):
    comp_fns = [sp.lambdify((x, y, t), expr_vec[i], "numpy") for i in range(int(expr_vec.rows))]

    def _call(xv, yv, tv, _fns=comp_fns):
        xv = np.asarray(xv, dtype=float)
        yv = np.asarray(yv, dtype=float)
        tv = np.asarray(tv, dtype=float)
        comps = [np.asarray(fn(xv, yv, tv), dtype=float) for fn in _fns]
        shape = np.broadcast(xv, yv, tv, *comps).shape
        comps = [np.broadcast_to(comp, shape) for comp in comps]
        return np.stack(comps, axis=-1)

    return _call


def _lambdify_mat_xyt(expr_mat, x, y, t):
    nrows = int(expr_mat.rows)
    ncols = int(expr_mat.cols)
    comp_fns = [[sp.lambdify((x, y, t), expr_mat[i, j], "numpy") for j in range(ncols)] for i in range(nrows)]

    def _call(xv, yv, tv, _fns=comp_fns):
        xv = np.asarray(xv, dtype=float)
        yv = np.asarray(yv, dtype=float)
        tv = np.asarray(tv, dtype=float)
        vals = [[np.asarray(fn(xv, yv, tv), dtype=float) for fn in row] for row in _fns]
        flat = [val for row in vals for val in row]
        shape = np.broadcast(xv, yv, tv, *flat).shape
        out = np.empty(shape + (nrows, ncols), dtype=float)
        for i in range(nrows):
            for j in range(ncols):
                out[..., i, j] = np.broadcast_to(vals[i][j], shape)
        return out

    return _call


@dataclass(frozen=True)
class DeformationOnlyMMS:
    case_id: str
    title: str
    geometry: str
    t_n: float
    t_k: float
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


def _double_well_prime(alpha):
    return 2.0 * alpha * (1.0 - alpha) * (1.0 - 2.0 * alpha)


def _build_case(
    *,
    case_id: str,
    title: str,
    geometry: str,
    dt_val: float,
    theta: float,
    rho_f: float,
    mu_f: float,
    mu_b: float,
    kappa_inv: float,
    mu_s: float,
    lambda_s: float,
    phi_b: float,
    M_alpha: float,
    gamma_alpha: float,
    eps_alpha: float,
    v_expr,
    p_expr,
    vS_expr,
    u_expr,
    alpha_expr,
) -> DeformationOnlyMMS:
    dt = float(dt_val)
    if not (dt > 0.0):
        raise ValueError("dt_val must be positive.")
    th = float(theta)
    if not (0.0 <= th <= 1.0):
        raise ValueError("theta must be in [0,1].")

    x, y, t = sp.symbols("x y t", real=True)
    t_n = 0.0
    t_k = float(dt_val)

    v_expr = sp.Matrix(v_expr)
    p_expr = sp.sympify(p_expr)
    vS_expr = sp.Matrix(vS_expr)
    u_expr = sp.Matrix(u_expr)
    alpha_expr = sp.sympify(alpha_expr)

    mu_expr = float(gamma_alpha) * ((-float(eps_alpha)) * _sym_laplacian(alpha_expr, x, y) + (_double_well_prime(alpha_expr) / float(eps_alpha)))

    subs_n = {t: t_n}
    subs_k = {t: t_k}

    v_n_expr = sp.simplify(v_expr.subs(subs_n))
    p_n_expr = sp.simplify(p_expr.subs(subs_n))
    vS_n_expr = sp.simplify(vS_expr.subs(subs_n))
    u_n_expr = sp.simplify(u_expr.subs(subs_n))
    alpha_n_expr = sp.simplify(alpha_expr.subs(subs_n))
    mu_n_expr = sp.simplify(mu_expr.subs(subs_n))

    v_k_expr = sp.simplify(v_expr.subs(subs_k))
    p_k_expr = sp.simplify(p_expr.subs(subs_k))
    vS_k_expr = sp.simplify(vS_expr.subs(subs_k))
    u_k_expr = sp.simplify(u_expr.subs(subs_k))
    alpha_k_expr = sp.simplify(alpha_expr.subs(subs_k))
    mu_k_expr = sp.simplify(mu_expr.subs(subs_k))

    one_m_th = 1.0 - th
    v_th_expr = sp.simplify(th * v_k_expr + one_m_th * v_n_expr)
    vS_th_expr = sp.simplify(th * vS_k_expr + one_m_th * vS_n_expr)
    u_th_expr = sp.simplify(th * u_k_expr + one_m_th * u_n_expr)
    alpha_th_expr = sp.simplify(th * alpha_k_expr + one_m_th * alpha_n_expr)

    C_n_expr = sp.simplify(1.0 - alpha_n_expr * (1.0 - float(phi_b)))
    B_n_expr = sp.simplify(alpha_n_expr * (1.0 - float(phi_b)))
    rho_n_expr = sp.simplify(float(rho_f) * C_n_expr)
    mu_mix_n_expr = sp.simplify((1.0 - alpha_n_expr) * float(mu_f) + alpha_n_expr * float(mu_b))
    beta_n_expr = sp.simplify(alpha_n_expr * float(mu_f) * float(kappa_inv))

    grad_p_k = _sym_grad_scalar(p_k_expr, x, y)
    grad_alpha_n = _sym_grad_scalar(alpha_n_expr, x, y)
    eps_v_th = _sym_epsilon(v_th_expr, x, y)
    sigma_u_th = 2.0 * float(mu_s) * _sym_epsilon(u_th_expr, x, y) + float(lambda_s) * _sym_div_vec(u_th_expr, x, y) * sp.eye(2)

    mom_dt = rho_n_expr * ((v_k_expr - v_n_expr) / dt)
    mom_conv = rho_n_expr * (_sym_grad_vec(v_th_expr, x, y) * v_n_expr)
    mom_visc = -_sym_div_mat(2.0 * mu_mix_n_expr * eps_v_th, x, y)
    mom_press = C_n_expr * grad_p_k
    mom_drag = beta_n_expr * (v_th_expr - vS_th_expr)
    f_v_expr = sp.simplify(mom_dt + mom_conv + mom_visc + mom_press + mom_drag)

    mass_expr = sp.simplify(_sym_div_vec(C_n_expr * v_k_expr + B_n_expr * vS_k_expr, x, y))

    kin_expr = sp.simplify(((u_k_expr - u_n_expr) / dt) + (_sym_grad_vec(u_th_expr, x, y) * vS_n_expr) - vS_th_expr)

    skel_expr = sp.simplify(-_sym_div_mat(alpha_n_expr * sigma_u_th, x, y) + B_n_expr * grad_p_k - beta_n_expr * (v_th_expr - vS_th_expr))
    f_u_expr = sp.simplify(skel_expr / alpha_n_expr)

    alpha_adv = (_sym_grad_scalar(alpha_th_expr, x, y).dot(vS_n_expr)) + alpha_th_expr * _sym_div_vec(vS_n_expr, x, y)
    f_alpha_expr = sp.simplify(((alpha_k_expr - alpha_n_expr) / dt) + alpha_adv - float(M_alpha) * _sym_laplacian(mu_k_expr, x, y))

    mu_residual_expr = sp.simplify(mu_k_expr - float(gamma_alpha) * ((-float(eps_alpha)) * _sym_laplacian(alpha_k_expr, x, y) + (_double_well_prime(alpha_k_expr) / float(eps_alpha))))
    mass_components = list(mass_expr) if isinstance(mass_expr, sp.MatrixBase) else [mass_expr]
    if any(not _sym_is_zero(comp) for comp in mass_components):
        raise ValueError(f"Case {case_id!r} does not satisfy the one-domain mass constraint exactly.")
    if any(not _sym_is_zero(comp) for comp in kin_expr):
        raise ValueError(f"Case {case_id!r} does not satisfy the Eulerian kinematic constraint exactly.")
    if not _sym_is_zero(mu_residual_expr):
        raise ValueError(f"Case {case_id!r} does not satisfy the chemical-potential relation exactly.")

    grad_v_expr = _sym_grad_vec(v_expr, x, y)
    grad_p_expr = _sym_grad_scalar(p_expr, x, y)
    grad_vS_expr = _sym_grad_vec(vS_expr, x, y)
    grad_u_expr = _sym_grad_vec(u_expr, x, y)
    grad_alpha_expr = _sym_grad_scalar(alpha_expr, x, y)
    grad_mu_expr = _sym_grad_scalar(mu_expr, x, y)

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
    }

    return DeformationOnlyMMS(
        case_id=str(case_id),
        title=str(title),
        geometry=str(geometry),
        t_n=float(t_n),
        t_k=float(t_k),
        dt=float(dt),
        theta=float(th),
        params=params,
        v=_lambdify_vec_xyt(v_expr, x, y, t),
        p=_lambdify_scalar_xyt(p_expr, x, y, t),
        vS=_lambdify_vec_xyt(vS_expr, x, y, t),
        u=_lambdify_vec_xyt(u_expr, x, y, t),
        alpha=_lambdify_scalar_xyt(alpha_expr, x, y, t),
        mu_alpha=_lambdify_scalar_xyt(mu_expr, x, y, t),
        grad_v=_lambdify_mat_xyt(grad_v_expr, x, y, t),
        grad_p=_lambdify_vec_xyt(grad_p_expr, x, y, t),
        grad_vS=_lambdify_mat_xyt(grad_vS_expr, x, y, t),
        grad_u=_lambdify_mat_xyt(grad_u_expr, x, y, t),
        grad_alpha=_lambdify_vec_xyt(grad_alpha_expr, x, y, t),
        grad_mu_alpha=_lambdify_vec_xyt(grad_mu_expr, x, y, t),
        v_n=_lambdify_vec_xy(v_n_expr, x, y),
        p_n=_lambdify_scalar_xy(p_n_expr, x, y),
        vS_n=_lambdify_vec_xy(vS_n_expr, x, y),
        u_n=_lambdify_vec_xy(u_n_expr, x, y),
        alpha_n=_lambdify_scalar_xy(alpha_n_expr, x, y),
        mu_alpha_n=_lambdify_scalar_xy(mu_n_expr, x, y),
        v_k=_lambdify_vec_xy(v_k_expr, x, y),
        p_k=_lambdify_scalar_xy(p_k_expr, x, y),
        vS_k=_lambdify_vec_xy(vS_k_expr, x, y),
        u_k=_lambdify_vec_xy(u_k_expr, x, y),
        alpha_k=_lambdify_scalar_xy(alpha_k_expr, x, y),
        mu_alpha_k=_lambdify_scalar_xy(mu_k_expr, x, y),
        f_v=_lambdify_vec_xy(f_v_expr, x, y),
        f_u=_lambdify_vec_xy(f_u_expr, x, y),
        f_alpha=_lambdify_scalar_xy(f_alpha_expr, x, y),
    )


def build_deformation_only_mms_static(*, dt_val: float, theta: float = 1.0) -> DeformationOnlyMMS:
    x, y, t = sp.symbols("x y t", real=True)
    alpha_expr = 0.72 + 0.10 * sp.sin(sp.pi * y) * (1.0 + 0.05 * sp.sin(t))
    v_expr = sp.Matrix([0.30 * sp.sin(sp.pi * y) * (1.0 + 0.10 * sp.cos(t)), 0.0])
    vS_expr = sp.Matrix([0.0, 0.0])
    u_expr = sp.Matrix([0.0, 0.0])
    p_expr = 0.15 * sp.sin(sp.pi * x) * (1.0 + 0.10 * sp.sin(t))
    return _build_case(
        case_id="static",
        title="Static smooth coupled MMS",
        geometry="Unit square with smooth coefficient variations and stationary interface profile.",
        dt_val=dt_val,
        theta=theta,
        rho_f=1.0,
        mu_f=1.0e-2,
        mu_b=3.0e-2,
        kappa_inv=8.0,
        mu_s=0.5,
        lambda_s=0.5,
        phi_b=0.45,
        M_alpha=0.05,
        gamma_alpha=0.2,
        eps_alpha=0.10,
        v_expr=v_expr,
        p_expr=p_expr,
        vS_expr=vS_expr,
        u_expr=u_expr,
        alpha_expr=alpha_expr,
    )


def build_deformation_only_mms_translation(*, dt_val: float, theta: float = 1.0) -> DeformationOnlyMMS:
    x, y, t = sp.symbols("x y t", real=True)
    speed = 0.15
    eps = 0.06
    x_c = 0.42 + speed * t
    alpha_expr = sp.Rational(1, 2) * (1.0 - sp.tanh((x - x_c) / eps))
    v_expr = sp.Matrix([speed, 0.0])
    vS_expr = sp.Matrix([speed, 0.0])
    u_expr = sp.Matrix([speed * t, 0.0])
    p_expr = 0.0
    return _build_case(
        case_id="translation",
        title="Moving diffuse-interface translation MMS",
        geometry="Unit square with a translating diffuse vertical interface.",
        dt_val=dt_val,
        theta=theta,
        rho_f=1.0,
        mu_f=1.0e-2,
        mu_b=1.0e-2,
        kappa_inv=5.0,
        mu_s=0.5,
        lambda_s=0.5,
        phi_b=0.55,
        M_alpha=0.02,
        gamma_alpha=0.10,
        eps_alpha=0.08,
        v_expr=v_expr,
        p_expr=p_expr,
        vS_expr=vS_expr,
        u_expr=u_expr,
        alpha_expr=alpha_expr,
    )


def build_deformation_only_mms_shear(*, dt_val: float, theta: float = 1.0) -> DeformationOnlyMMS:
    x, y, t = sp.symbols("x y t", real=True)
    eps = 0.08
    layer = sp.Rational(1, 2) * (1.0 - sp.tanh((y - 0.42) / eps))
    alpha_expr = 0.20 + 0.65 * layer
    v_expr = sp.Matrix([0.12 * y, 0.0])
    vS_expr = sp.Matrix([0.20 * y, 0.0])
    u_expr = sp.Matrix([0.20 * y * t, 0.0])
    p_expr = 0.10 * sp.sin(sp.pi * x)
    return _build_case(
        case_id="shear",
        title="Affine shear/deformation MMS",
        geometry="Unit square interpreted as an attached porous layer under horizontal shear.",
        dt_val=dt_val,
        theta=theta,
        rho_f=1.0,
        mu_f=1.0e-2,
        mu_b=4.0e-2,
        kappa_inv=10.0,
        mu_s=0.8,
        lambda_s=0.6,
        phi_b=0.50,
        M_alpha=0.03,
        gamma_alpha=0.15,
        eps_alpha=0.08,
        v_expr=v_expr,
        p_expr=p_expr,
        vS_expr=vS_expr,
        u_expr=u_expr,
        alpha_expr=alpha_expr,
    )
