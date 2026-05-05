"""Manufactured solution utilities for FPI Example 4.1 (paper).

This module provides a *fast* SymPy-derived MMS that is consistent with the
Eulerian FPI operators implemented in:
  - `examples/utils/fpi/fully_eulerian.py`
  - `examples/utils/fpi/poro.py`
  - `examples/utils/fpi/interface.py`

The geometry in the paper is complex; the MMS itself is purely analytic and can
be evaluated on arbitrary meshes/level sets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import sympy as sp

from pycutfem.core.levelset import RotatedBoxLevelSet


def _kinv_matrix(case: str, *, K: float) -> np.ndarray:
    case = str(case).strip().lower()
    if case in {"iso", "identity", "i"}:
        return (1.0 / float(K)) * np.eye(2, dtype=float)
    if case in {"aniso", "anisotropic", "a"}:
        base = np.array([[2.0, 0.3], [0.1, 1.5]], dtype=float)
        return (1.0 / float(K)) * base
    raise ValueError(f"Unknown K_inv case {case!r}")


@dataclass(frozen=True)
class Example41MMS:
    # exact fields at t_n and t_{n+1} (single BE step)
    vF_n: callable
    pF_n: callable
    vP_n: callable
    uP_n: callable
    uP_nm1: callable | None
    pP_n: callable
    vF_k: callable
    pF_k: callable
    vP_k: callable
    uP_k: callable
    pP_k: callable
    # body forces for BE step
    fF: callable
    fD: callable
    fS: callable
    # porous mass-balance source (q_P equation)
    f_mass: callable
    # interface data (paper eqs (40)-(43) adapted to our normal convention)
    g_sigma: callable
    g_sigma_n: callable
    g_n: callable
    g_t: callable
    # scalar component callables for Dirichlet BCs at t_{n+1}
    vF_x: callable
    vF_y: callable
    pF_s: callable
    vP_x: callable
    vP_y: callable
    uP_x: callable
    uP_y: callable
    pP_s: callable
    # exact gradients at t_{n+1} (for H1-seminorm errors)
    grad_vF_k: callable  # (x,y)->(2,2)
    grad_uP_k: callable  # (x,y)->(2,2)
    # exact Cauchy stresses at t_{n+1} (for traction-based manufactured data)
    sigmaF_k: callable  # (x,y)->(2,2)
    sigmaP_k: callable  # (x,y)->(2,2)


@lru_cache(maxsize=None)
def build_example41_mms(
    *,
    dt_val: float,
    kinv_case: str,
    t_prev: float = 0.0,
    beta_BJ: float = 1.0,
    interface: str = "vertical",
    interface_params: tuple[float, ...] = (),
) -> Example41MMS:
    """
    Build a BE-step MMS consistent with the implemented Eulerian FPI forms.

    We use the analytic fields from `num_example.tex`
    (eqs. (31)-(35)). Body forces and interface jump data are derived
    symbolically from the same operators used in the code.
    """
    # Parameters from `num_example.tex` (Example 4.1)
    B = 1.0
    C = 0.01
    A_F = 0.10
    A_P = 0.21
    # Paper (num_example.tex): choose A_S^P so kinematic constraints vanish at t=0.
    A_PS = -0.01

    rho_f = 1.0
    mu_f = 1.0
    phi = 0.5
    K = 0.10
    alpha_BJ = 1.0

    E = 1000.0
    nu = 0.30

    mu_s = E / (2.0 * (1.0 + nu))
    c_nh = mu_s / 2.0
    # Paper (eq. 36): plane strain / 3D compressible NH parameter.
    beta_nh = nu / (1.0 - 2.0 * nu)

    kappa = math.sqrt(K) / (alpha_BJ * mu_f * math.sqrt(phi))

    # Symbolic setup
    x, y = sp.symbols("x y", real=True)
    pi = sp.pi
    s = sp.Float(B) * pi

    dt = sp.Float(float(dt_val))
    t0 = sp.Float(float(t_prev))
    t1 = t0 + dt
    t_nm1 = t0 - dt

    lam_u = -2.0 * (sp.Float(C) ** 2) * (pi**2) * sp.Float(mu_f) / sp.Float(rho_f)
    gu0 = sp.exp(lam_u * t0)
    gu1 = sp.exp(lam_u * t1)
    gp0 = sp.exp(2 * lam_u * t0)  # gp(t) in the paper
    gp1 = sp.exp(2 * lam_u * t1)

    # exact fields (as in the paper)
    vx_F_1 = -sp.Float(A_F) * sp.cos(s * x) * sp.sin(s * y) * gu1
    vy_F_1 = sp.Float(A_F) * sp.sin(s * x) * sp.cos(s * y) * gu1
    vF1 = sp.Matrix([vx_F_1, vy_F_1])
    vF0 = sp.Matrix(
        [
            -sp.Float(A_F) * sp.cos(s * x) * sp.sin(s * y) * gu0,
            sp.Float(A_F) * sp.sin(s * x) * sp.cos(s * y) * gu0,
        ]
    )

    vx_P_1 = -sp.Float(A_P) * sp.cos(s * x) * sp.sin(s * y) * gu1
    vy_P_1 = sp.Float(A_P) * sp.sin(s * x) * sp.cos(s * y) * gu1
    vP1 = sp.Matrix([vx_P_1, vy_P_1])
    vP0 = sp.Matrix(
        [
            -sp.Float(A_P) * sp.cos(s * x) * sp.sin(s * y) * gu0,
            sp.Float(A_P) * sp.sin(s * x) * sp.cos(s * y) * gu0,
        ]
    )

    p0 = -sp.Rational(1, 4) * (sp.cos(2 * s * x) + sp.cos(2 * s * y)) * sp.Float(rho_f) * gp0
    p1 = -sp.Rational(1, 4) * (sp.cos(2 * s * x) + sp.cos(2 * s * y)) * sp.Float(rho_f) * gp1

    denom = lam_u  # == -2 C^2 pi^2 mu/rho
    fac0 = (sp.Float(1.0) - gu0) / denom
    fac1 = (sp.Float(1.0) - gu1) / denom
    fac_nm1 = (sp.Float(1.0) - sp.exp(lam_u * t_nm1)) / denom
    has_nm1 = bool(float(t_prev) >= float(dt_val))
    ux1 = sp.Float(A_PS) * sp.cos(s * x) * sp.sin(s * y) * fac1
    uy1 = sp.Float(A_PS) * sp.sin(s * x) * sp.cos(s * y) * fac1
    u1 = sp.Matrix([ux1, uy1])
    u0 = sp.Matrix(
        [
            sp.Float(A_PS) * sp.cos(s * x) * sp.sin(s * y) * fac0,
            sp.Float(A_PS) * sp.sin(s * x) * sp.cos(s * y) * fac0,
        ]
    )
    u_nm1 = (
        sp.Matrix(
            [
                sp.Float(A_PS) * sp.cos(s * x) * sp.sin(s * y) * fac_nm1,
                sp.Float(A_PS) * sp.sin(s * x) * sp.cos(s * y) * fac_nm1,
            ]
        )
        if has_nm1
        else None
    )

    # Discrete BE time derivatives
    vdotF = (vF1 - vF0) / dt
    vdotP = (vP1 - vP0) / dt
    v_s = (u1 - u0) / dt
    v_s_n = (u0 - u_nm1) / dt if u_nm1 is not None else sp.Matrix([0.0, 0.0])

    def _grad_vec(a: sp.Matrix) -> sp.Matrix:
        return sp.Matrix([[sp.diff(a[0], x), sp.diff(a[0], y)], [sp.diff(a[1], x), sp.diff(a[1], y)]])

    def _div_tensor(T: sp.Matrix) -> sp.Matrix:
        return sp.Matrix([sp.diff(T[0, 0], x) + sp.diff(T[0, 1], y), sp.diff(T[1, 0], x) + sp.diff(T[1, 1], y)])

    def _epsilon(v: sp.Matrix) -> sp.Matrix:
        G = _grad_vec(v)
        return sp.Rational(1, 2) * (G + G.T)

    # --- Fluid forcing: rho vdot + rho (v·∇)v - div(sigma) ---
    I2 = sp.eye(2)
    sigmaF = -p1 * I2 + 2.0 * sp.Float(mu_f) * _epsilon(vF1)
    convF = _grad_vec(vF1) * vF1
    fF = sp.Float(rho_f) * vdotF + sp.Float(rho_f) * convF - _div_tensor(sigmaF)

    # --- Poro kinematics (Eulerian reference map): F = (I - ∇u)^{-1} ---
    # Explicit 2x2 inverse (grad u is symmetric for this MMS) keeps expressions compact.
    grad_u = _grad_vec(u1)
    a_u = grad_u[0, 0]
    b_u = grad_u[0, 1]
    Finv = sp.Matrix([[1.0 - a_u, -b_u], [-b_u, 1.0 - a_u]])
    det_finv = (1.0 - a_u) ** 2 - b_u**2
    F = (1.0 / det_finv) * sp.Matrix([[1.0 - a_u, b_u], [b_u, 1.0 - a_u]])
    J = 1.0 / det_finv

    K_inv = sp.Matrix(_kinv_matrix(kinv_case, K=K))
    k_inv = J * (Finv.T * K_inv * Finv)

    grad_p = sp.Matrix([sp.diff(p1, x), sp.diff(p1, y)])

    # --- Darcy forcing ---
    convP = -sp.Float(rho_f) * (_grad_vec(vP1) * v_s)
    dragP = sp.Float(mu_f) * (sp.Float(phi) ** 2) * (k_inv * (vP1 - v_s))
    fD = sp.Float(rho_f) * vdotP + convP + grad_p + dragP

    # --- Porous mass-balance forcing ---
    # Our Eulerian poro formulation enforces:
    #   div( φ v + (1-φ) v_s ) = f_mass
    # For the BE θ-scheme with θ=1 used in Example 4.1, this is evaluated at t_{n+1}.
    div_vP1 = sp.diff(vP1[0], x) + sp.diff(vP1[1], y)
    div_u1 = sp.diff(u1[0], x) + sp.diff(u1[1], y)
    div_u0 = sp.diff(u0[0], x) + sp.diff(u0[1], y)
    div_vs = (div_u1 - div_u0) / dt
    f_mass = sp.Float(phi) * div_vP1 + (sp.Float(1.0) - sp.Float(phi)) * div_vs

    # --- Skeleton forcing ---
    acc_local = (v_s - v_s_n) / dt
    acc = acc_local + (_grad_vec(v_s) * v_s)
    aJ = J ** (-2.0 * sp.Float(beta_nh))
    Bmat = F * F.T
    sigmaP = (2.0 * sp.Float(c_nh) / J) * (Bmat - aJ * I2)
    fS = acc - _div_tensor(sigmaP) - sp.Float(phi) * grad_p - dragP

    # --- Interface jump data (Γ): build from analytic fields ---
    g_n = vF1 - v_s - sp.Float(phi) * (vP1 - v_s)

    # ---- lambdify to NumPy ----
    def _scalar(expr):
        return sp.lambdify((x, y), expr, "numpy")

    def _vec(expr_vec: sp.Matrix):
        f0 = sp.lambdify((x, y), expr_vec[0], "numpy")
        f1 = sp.lambdify((x, y), expr_vec[1], "numpy")

        def cb(xv, yv):
            xv = np.asarray(xv, dtype=float)
            g0 = np.asarray(f0(xv, yv), dtype=float)
            g1 = np.asarray(f1(xv, yv), dtype=float)
            if g0.shape != xv.shape:
                g0 = np.broadcast_to(g0, xv.shape)
            if g1.shape != xv.shape:
                g1 = np.broadcast_to(g1, xv.shape)
            return np.stack((g0, g1), axis=-1)

        return cb

    def _mat(expr_mat: sp.Matrix):
        f00 = sp.lambdify((x, y), expr_mat[0, 0], "numpy")
        f01 = sp.lambdify((x, y), expr_mat[0, 1], "numpy")
        f10 = sp.lambdify((x, y), expr_mat[1, 0], "numpy")
        f11 = sp.lambdify((x, y), expr_mat[1, 1], "numpy")

        def cb(xv, yv):
            xv = np.asarray(xv, dtype=float)
            g00 = np.asarray(f00(xv, yv), dtype=float)
            g01 = np.asarray(f01(xv, yv), dtype=float)
            g10 = np.asarray(f10(xv, yv), dtype=float)
            g11 = np.asarray(f11(xv, yv), dtype=float)
            if g00.shape != xv.shape:
                g00 = np.broadcast_to(g00, xv.shape)
            if g01.shape != xv.shape:
                g01 = np.broadcast_to(g01, xv.shape)
            if g10.shape != xv.shape:
                g10 = np.broadcast_to(g10, xv.shape)
            if g11.shape != xv.shape:
                g11 = np.broadcast_to(g11, xv.shape)
            row0 = np.stack((g00, g01), axis=-1)
            row1 = np.stack((g10, g11), axis=-1)
            return np.stack((row0, row1), axis=-2)

        return cb

    vF_k_fun = _vec(vF1)
    vP_k_fun = _vec(vP1)
    uP_k_fun = _vec(u1)

    vF_n_fun = _vec(vF0)
    vP_n_fun = _vec(vP0)
    uP_n_fun = _vec(u0)
    uP_nm1_fun = _vec(u_nm1) if u_nm1 is not None else None

    p_k_fun = _scalar(p1)
    p_n_fun = _scalar(p0)

    fF_fun = _vec(fF)
    fD_fun = _vec(fD)
    fS_fun = _vec(fS)
    f_mass_fun = _scalar(f_mass)

    gn_fun = _vec(g_n)
    sigmaF_fun = _mat(sigmaF)
    sigmaP_fun = _mat(sigmaP)
    v_s_fun = _vec(v_s)

    interface_key = str(interface).strip().lower()
    if interface_key in {"vertical", "line", "affine"}:

        def nF_fun(xv, yv):
            xv = np.asarray(xv, dtype=float)
            nx = -np.ones_like(xv, dtype=float)
            ny = np.zeros_like(xv, dtype=float)
            return np.stack((nx, ny), axis=-1)

    elif interface_key in {"rotated_box", "rotbox", "box"}:
        if len(interface_params) != 5:
            raise ValueError("rotated_box interface requires interface_params=(cx, cy, hx, hy, angle).")
        cx, cy, hx, hy, angle = (float(v) for v in interface_params)
        ls = RotatedBoxLevelSet(center=(cx, cy), hx=hx, hy=hy, angle=angle)

        def nF_fun(xv, yv):
            xv = np.asarray(xv, dtype=float)
            yv = np.asarray(yv, dtype=float)
            xy = np.stack((xv, yv), axis=-1)
            n = ls.gradient(xy)  # (-) -> (+) (poro -> fluid)
            return -np.asarray(n, dtype=float)  # fluid outward

    else:
        raise ValueError(f"Unknown interface {interface!r}; expected 'vertical' or 'rotated_box'.")

    def _mv(mat, vec):
        return np.einsum("...ij,...j->...i", mat, vec)

    def gsig_fun(xv, yv):
        nF = nF_fun(xv, yv)
        sigF = sigmaF_fun(xv, yv)
        sigP = sigmaP_fun(xv, yv)
        return _mv(sigF, nF) - _mv(sigP, nF)

    def gsn_fun(xv, yv):
        nF = nF_fun(xv, yv)
        sigF = sigmaF_fun(xv, yv)
        tF = _mv(sigF, nF)
        return np.einsum("...i,...i->...", tF, nF) + p_k_fun(xv, yv)

    def gt_fun(xv, yv):
        nF = nF_fun(xv, yv)
        sigF = sigmaF_fun(xv, yv)
        tF = _mv(sigF, nF)
        vFv = vF_k_fun(xv, yv)
        vPv = vP_k_fun(xv, yv)
        vsv = v_s_fun(xv, yv)
        return vFv - vsv - float(beta_BJ) * float(phi) * (vPv - vsv) + float(kappa) * tF

    grad_vF_k_fun = _mat(_grad_vec(vF1))
    grad_uP_k_fun = _mat(_grad_vec(u1))

    return Example41MMS(
        vF_n=vF_n_fun,
        pF_n=p_n_fun,
        vP_n=vP_n_fun,
        uP_n=uP_n_fun,
        uP_nm1=uP_nm1_fun,
        pP_n=p_n_fun,
        vF_k=vF_k_fun,
        pF_k=p_k_fun,
        vP_k=vP_k_fun,
        uP_k=uP_k_fun,
        pP_k=p_k_fun,
        fF=fF_fun,
        fD=fD_fun,
        fS=fS_fun,
        f_mass=f_mass_fun,
        g_sigma=gsig_fun,
        g_sigma_n=gsn_fun,
        g_n=gn_fun,
        g_t=gt_fun,
        vF_x=lambda xx, yy: float(vF_k_fun(xx, yy)[0]),
        vF_y=lambda xx, yy: float(vF_k_fun(xx, yy)[1]),
        pF_s=lambda xx, yy: float(p_k_fun(xx, yy)),
        vP_x=lambda xx, yy: float(vP_k_fun(xx, yy)[0]),
        vP_y=lambda xx, yy: float(vP_k_fun(xx, yy)[1]),
        uP_x=lambda xx, yy: float(uP_k_fun(xx, yy)[0]),
        uP_y=lambda xx, yy: float(uP_k_fun(xx, yy)[1]),
        pP_s=lambda xx, yy: float(p_k_fun(xx, yy)),
        grad_vF_k=grad_vF_k_fun,
        grad_uP_k=grad_uP_k_fun,
        sigmaF_k=sigmaF_fun,
        sigmaP_k=sigmaP_fun,
    )
