"""Manufactured solution utilities for FPI Example 4.1 (paper).

This module provides a *fast* SymPy-derived MMS that is consistent with the
Eulerian FPI operators implemented in:
  - `pycutfem/utils/fpi_fully_eulerian.py`
  - `pycutfem/utils/fpi_poro_eulerian.py`
  - `pycutfem/utils/fpi_interface_eulerian.py`

The geometry in the paper is complex; the MMS itself is purely analytic and can
be evaluated on arbitrary meshes/level sets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import sympy as sp


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


@lru_cache(maxsize=None)
def build_example41_mms(*, dt_val: float, kinv_case: str) -> Example41MMS:
    """
    Build a BE-step MMS consistent with the implemented Eulerian FPI forms.

    We fix t_n=0, t_{n+1}=dt and use the analytic fields from `num_example.tex`
    (eqs. (31)-(35)). Body forces and interface jump data are derived
    symbolically from the same operators used in the code.
    """
    # Parameters from `num_example.tex` (Example 4.1)
    B = 1.0
    C = 0.01
    A_F = 0.10
    A_P = 0.21
    A_PS = 0.025

    rho_f = 1.0
    mu_f = 1.0
    phi = 0.5
    K = 0.10
    beta_BJ = 1.0
    alpha_BJ = 1.0

    E = 1000.0
    nu = 0.30

    mu_s = E / (2.0 * (1.0 + nu))
    c_nh = mu_s / 2.0
    beta_nh = nu / (1.0 - nu)  # true-2D

    kappa = math.sqrt(K) / (alpha_BJ * mu_f * math.sqrt(phi))

    # Symbolic setup
    x, y = sp.symbols("x y", real=True)
    pi = sp.pi
    s = sp.Float(B) * pi

    dt = sp.Float(float(dt_val))
    t0 = sp.Float(0.0)
    t1 = dt

    lam_u = -2.0 * (sp.Float(C) ** 2) * (pi**2) * sp.Float(mu_f) / sp.Float(rho_f)
    gu0 = sp.exp(lam_u * t0)
    gu1 = sp.exp(lam_u * t1)
    gp1 = sp.exp(2 * lam_u * t1)  # gp(t) in the paper

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

    p1 = -sp.Rational(1, 4) * (sp.cos(2 * s * x) + sp.cos(2 * s * y)) * sp.Float(rho_f) * gp1

    denom = lam_u  # == -2 C^2 pi^2 mu/rho
    fac1 = (sp.Float(1.0) - gu1) / denom
    ux1 = sp.Float(A_PS) * sp.cos(s * x) * sp.sin(s * y) * fac1
    uy1 = sp.Float(A_PS) * sp.sin(s * x) * sp.cos(s * y) * fac1
    u1 = sp.Matrix([ux1, uy1])
    u0 = sp.Matrix([0.0, 0.0])

    # Discrete BE time derivatives
    vdotF = (vF1 - vF0) / dt
    vdotP = (vP1 - vP0) / dt
    v_s = (u1 - u0) / dt

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

    # --- Skeleton forcing ---
    acc = (v_s / dt) + (_grad_vec(v_s) * v_s)  # first BE step
    aJ = J ** (-2.0 * sp.Float(beta_nh))
    Bmat = F * F.T
    sigmaP = (2.0 * sp.Float(c_nh) / J) * (Bmat - aJ * I2)
    fS = acc - _div_tensor(sigmaP) - sp.Float(phi) * grad_p - dragP

    # --- Interface jump data (Γ): build from analytic fields ---
    nF = sp.Matrix([-1.0, 0.0])  # fluid outward for vertical interface
    tractionF = sigmaF * nF
    tractionP = sigmaP * nF
    g_sigma = tractionF - tractionP
    g_sigma_n = (nF.T * sigmaF * nF)[0] + p1  # eq (41): n·σ_F·n + p_P
    g_n = vF1 - v_s - sp.Float(phi) * (vP1 - v_s)
    g_t = vF1 - v_s - sp.Float(beta_BJ) * sp.Float(phi) * (vP1 - v_s) + sp.Float(kappa) * tractionF

    # ---- lambdify to NumPy ----
    def _scalar(expr):
        return sp.lambdify((x, y), expr, "numpy")

    def _vec(expr_vec: sp.Matrix):
        f0 = sp.lambdify((x, y), expr_vec[0], "numpy")
        f1 = sp.lambdify((x, y), expr_vec[1], "numpy")

        def cb(xv, yv):
            return np.stack((f0(xv, yv), f1(xv, yv)), axis=-1)

        return cb

    vF_k_fun = _vec(vF1)
    vP_k_fun = _vec(vP1)
    uP_k_fun = _vec(u1)

    vF_n_fun = _vec(vF0)
    vP_n_fun = _vec(vP0)
    uP_n_fun = lambda xv, yv: np.stack((0.0 * np.asarray(xv), 0.0 * np.asarray(xv)), axis=-1)

    p_k_fun = _scalar(p1)
    p_n_fun = _scalar(-sp.Rational(1, 4) * (sp.cos(2 * s * x) + sp.cos(2 * s * y)) * sp.Float(rho_f))

    fF_fun = _vec(fF)
    fD_fun = _vec(fD)
    fS_fun = _vec(fS)

    gsig_fun = _vec(g_sigma)
    gn_fun = _vec(g_n)
    gt_fun = _vec(g_t)
    gsn_fun = _scalar(g_sigma_n)

    return Example41MMS(
        vF_n=vF_n_fun,
        pF_n=p_n_fun,
        vP_n=vP_n_fun,
        uP_n=uP_n_fun,
        pP_n=p_n_fun,
        vF_k=vF_k_fun,
        pF_k=p_k_fun,
        vP_k=vP_k_fun,
        uP_k=uP_k_fun,
        pP_k=p_k_fun,
        fF=fF_fun,
        fD=fD_fun,
        fS=fS_fun,
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
    )

