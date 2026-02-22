#!/usr/bin/env python
"""
Warner & Gujer (1986) multispecies biofilm benchmark (1D in ζ).

Implements the 5 case studies from:
  O. Wanner and W. Gujer, "A multispecies biofilm model", Biotechnol. Bioeng.
  28(3):314–328, 1986.

The implementation follows the FORTRAN UPDATE listing in
`examples/biofilms/benchmarks/warner1986.tex` (Appendix B) and reproduces the
reference Table VI profiles for:
  - Case 1 at t = 6.0 d
  - Case 4 at t = 6.0 d (0.006 d after sloughing)

Backends:
  - python: numpy implementation of RHS
  - jit:    numba-compiled RHS (if numba is available)
  - cpp:    pybind11-compiled RHS via pycutfem's C++ compiler helper

Typical usage:
  python -u examples/biofilms/benchmarks/warner1986_benchmark.py --case 1 --backend cpp --outdir examples/biofilms/results/warner1986
  python -u examples/biofilms/benchmarks/warner1986_benchmark.py --case all --backend cpp --compare-backends
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from scipy.integrate import solve_ivp


Backend = Literal["python", "jit", "cpp"]


@dataclass(frozen=True)
class Warner1986Params:
    # Discretization (matches UPDATE: NPOINT=15).
    npoint: int = 15

    # Biomass density (g COD / m^3 biofilm).
    rho: float = 5000.0

    # Substrate diffusivities D_i (m^2 / d).
    D1: float = 83.0e-6
    D2: float = 149.0e-6
    D3: float = 175.0e-6

    # Kinetic parameters.
    mu_hat_1: float = 4.8
    mu_hat_2: float = 0.95
    Ks1: float = 5.0
    Ks2: float = 1.0
    Ko1: float = 0.1
    Ko2: float = 0.1
    Y1: float = 0.4
    Y2: float = 0.22
    b1: float = 0.2
    b2: float = 0.05
    k1: float = 0.1
    k2: float = 0.1
    alpha1: float = 1.0
    alpha2: float = 4.57

    # Initial conditions.
    L0: float = 5.0e-6
    f1_0: float = 0.65
    f2_0: float = 0.35
    S1_0: float = 3.0
    S2_0: float = 13.0
    S3_0: float = 8.0

    # Case 3 shear coefficient λ (m^{-1} d^{-1}): σ = -λ L^2.
    shear_lambda: float = 750.0

    # Case 4 sloughing (σ in m/d over [t0,t1]).
    slough_sigma: float = -0.05
    slough_t0: float = 5.984
    slough_t1: float = 5.994

    # Case 5: external mass transfer + completely mixed reactor.
    # D_i / D_Li = 0.8, laminar sublayer thickness LL.
    D_ratio: float = 0.8
    AL: float = 1.0
    LL: float = 2.0e-5
    Q: float = 0.5
    VR: float = 0.01


def _zeta_grid(params: Warner1986Params) -> tuple[np.ndarray, float]:
    zeta = np.linspace(0.0, 1.0, int(params.npoint), dtype=float)
    dz = float(zeta[1] - zeta[0])
    return zeta, dz


def _unpack_state(y: np.ndarray, *, n: int, with_bulk: bool) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    State layout:
      without_bulk: [L, f1(0..n-1), f2(0..n-1), S1, S2, S3]
      with_bulk:    [L, f1, f2, S1, S2, S3, SL1, SL2, SL3]
    """
    off = 0
    L = float(y[off])
    off += 1
    f1 = np.asarray(y[off : off + n], dtype=float)
    off += n
    f2 = np.asarray(y[off : off + n], dtype=float)
    off += n
    S1 = np.asarray(y[off : off + n], dtype=float)
    off += n
    S2 = np.asarray(y[off : off + n], dtype=float)
    off += n
    S3 = np.asarray(y[off : off + n], dtype=float)
    off += n
    SL = None
    if with_bulk:
        SL = np.asarray(y[off : off + 3], dtype=float)
        off += 3
    if off != y.size:
        raise ValueError(f"Bad state layout: consumed {off} entries but y has {y.size}.")
    return L, f1, f2, S1, S2, S3, SL


def _pack_state(
    *,
    L: float,
    f1: np.ndarray,
    f2: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
    S3: np.ndarray,
    SL: np.ndarray | None,
) -> np.ndarray:
    parts = [np.asarray([float(L)]), np.asarray(f1), np.asarray(f2), np.asarray(S1), np.asarray(S2), np.asarray(S3)]
    if SL is not None:
        parts.append(np.asarray(SL))
    return np.concatenate(parts).astype(float, copy=False)


def _sigma(case_id: int, t: float, L: float, params: Warner1986Params) -> float:
    if case_id in (1, 2, 5):
        return 0.0
    if case_id == 3:
        return -float(params.shear_lambda) * float(L) * float(L)
    if case_id == 4:
        if float(params.slough_t0) < float(t) < float(params.slough_t1):
            return float(params.slough_sigma)
        return 0.0
    raise ValueError(f"Unknown case_id={case_id}")


def _surface_concentrations_case_1_2_3_4(case_id: int, t: float, params: Warner1986Params) -> np.ndarray:
    # Cases 1/3/4: constant surface substrates.
    if case_id in (1, 3, 4):
        return np.asarray([params.S1_0, params.S2_0, params.S3_0], dtype=float)
    # Case 2: step change of organics after 6 days.
    if case_id == 2:
        s1 = params.S1_0 if float(t) < 6.0 else 0.0
        return np.asarray([s1, params.S2_0, params.S3_0], dtype=float)
    raise ValueError(f"Unsupported case_id={case_id} in surface_concentrations.")


def _central_first_derivative(arr: np.ndarray, dz: float) -> np.ndarray:
    out = np.zeros_like(arr, dtype=float)
    if arr.size < 3:
        out[0] = 0.0
        if arr.size == 2:
            out[0] = (arr[1] - arr[0]) / dz
            out[1] = out[0]
        return out
    out[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dz)
    # 2nd-order one-sided stencils at the ends.
    out[0] = (-3.0 * arr[0] + 4.0 * arr[1] - arr[2]) / (2.0 * dz)
    out[-1] = (3.0 * arr[-1] - 4.0 * arr[-2] + arr[-3]) / (2.0 * dz)
    return out


def _central_second_derivative_neumann_left(arr: np.ndarray, dz: float, *, bc_right: Literal["dirichlet", "robin"], right_value: float, right_flux: float | None = None) -> np.ndarray:
    """
    Second derivative with:
      - left boundary: Neumann (∂/∂ζ = 0) implemented via mirror ghost
      - right boundary: either Dirichlet (arr[-1]=right_value) or Robin where
        right_flux is ∂arr/∂ζ at ζ=1 (in ζ-coordinates)
    """
    n = int(arr.size)
    if n < 3:
        raise ValueError("Need at least 3 points for second derivative.")

    a = np.asarray(arr, dtype=float).copy()
    if bc_right == "dirichlet":
        a[-1] = float(right_value)
        ghost_right = 2.0 * a[-1] - a[-2]  # linear extrapolation (unused for interior)
    else:
        if right_flux is None:
            raise ValueError("Robin requires right_flux.")
        # Forward ghost so that (ghost - a[-1]) / dz = right_flux.
        ghost_right = a[-1] + dz * float(right_flux)

    out = np.zeros_like(a, dtype=float)

    # left ghost for Neumann: a[-1] = a[1] (mirror)
    ghost_left = a[1]

    # interior
    out[1:-1] = (a[2:] - 2.0 * a[1:-1] + a[:-2]) / (dz * dz)
    # left boundary
    out[0] = (a[1] - 2.0 * a[0] + ghost_left) / (dz * dz)
    # right boundary
    out[-1] = (ghost_right - 2.0 * a[-1] + a[-2]) / (dz * dz)
    return out


def _rhs_python(case_id: int, params: Warner1986Params) -> Callable[[float, np.ndarray], np.ndarray]:
    zeta, dz = _zeta_grid(params)
    n = int(params.npoint)
    D = np.asarray([params.D1, params.D2, params.D3], dtype=float)

    mu_hat = np.asarray([params.mu_hat_1, params.mu_hat_2], dtype=float)
    Ks = np.asarray([params.Ks1, params.Ks2], dtype=float)
    Ko = np.asarray([params.Ko1, params.Ko2], dtype=float)
    Y = np.asarray([params.Y1, params.Y2], dtype=float)
    b = np.asarray([params.b1, params.b2], dtype=float)
    k_inact = np.asarray([params.k1, params.k2], dtype=float)
    alpha = np.asarray([params.alpha1, params.alpha2], dtype=float)

    with_bulk = bool(case_id == 5)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        L, f1, f2, S1, S2, S3, SL = _unpack_state(y, n=n, with_bulk=with_bulk)
        f3 = 1.0 - f1 - f2

        sig = _sigma(case_id, t, L, params)

        # Surface substrate concentrations.
        if case_id != 5:
            SL_surf = _surface_concentrations_case_1_2_3_4(case_id, t, params)
        else:
            assert SL is not None
            SL_surf = SL

        # Enforce case 1-4 Dirichlet at ζ=1 (surface).
        if case_id != 5:
            S1 = np.asarray(S1, dtype=float).copy()
            S2 = np.asarray(S2, dtype=float).copy()
            S3 = np.asarray(S3, dtype=float).copy()
            S1[-1] = SL_surf[0]
            S2[-1] = SL_surf[1]
            S3[-1] = SL_surf[2]

        # --- Kinetics (Appendix B, section C4) ---
        O2_term_1 = S3 / (Ko[0] + S3)
        O2_term_2 = S3 / (Ko[1] + S3)

        mu0_1 = (mu_hat[0] * S1 / (Ks[0] + S1) - b[0]) * O2_term_1 - k_inact[0]
        mu0_2 = (mu_hat[1] * S2 / (Ks[1] + S2) - b[1]) * O2_term_2 - k_inact[1]

        r1 = -mu_hat[0] * params.rho * f1 * O2_term_1 * S1 / (Ks[0] + S1) / Y[0]
        r2 = -mu_hat[1] * params.rho * f2 * O2_term_2 * S2 / (Ks[1] + S2) / Y[1]
        r3 = (alpha[0] - Y[0]) * r1 - b[0] * params.rho * f1 * O2_term_1 + (alpha[1] - Y[1]) * r2 - b[1] * params.rho * f2 * O2_term_2

        # Mean observed specific growth rate μ̄0 (inactivation cancels via inert).
        mu_bar = (mu0_1 + k_inact[0]) * f1 + (mu0_2 + k_inact[1]) * f2

        # --- Velocities (Appendix B, section C5) ---
        # Cumulative integral IMOB via trapezoid on uniform ζ grid.
        imob = np.zeros_like(mu_bar, dtype=float)
        imob[1:] = np.cumsum(0.5 * (mu_bar[1:] + mu_bar[:-1]) * dz)
        u = L * imob
        uL = L * imob[-1] + sig

        # --- Spatial derivatives (method of lines) ---
        f1_z = _central_first_derivative(f1, dz)
        f2_z = _central_first_derivative(f2, dz)

        # Substrate derivatives with Neumann at ζ=0.
        if case_id != 5:
            # Dirichlet at ζ=1
            S1_z = _central_first_derivative(S1, dz)
            S2_z = _central_first_derivative(S2, dz)
            S3_z = _central_first_derivative(S3, dz)
            S1_zz = _central_second_derivative_neumann_left(S1, dz, bc_right="dirichlet", right_value=SL_surf[0])
            S2_zz = _central_second_derivative_neumann_left(S2, dz, bc_right="dirichlet", right_value=SL_surf[1])
            S3_zz = _central_second_derivative_neumann_left(S3, dz, bc_right="dirichlet", right_value=SL_surf[2])
        else:
            # Robin at ζ=1: ∂S/∂ζ = (L / (D_ratio*LL)) (S_L - S_surface)
            coeff = L / (params.D_ratio * params.LL)
            S1_flux = coeff * (SL_surf[0] - S1[-1])
            S2_flux = coeff * (SL_surf[1] - S2[-1])
            S3_flux = coeff * (SL_surf[2] - S3[-1])

            S1_z = _central_first_derivative(S1, dz)
            S2_z = _central_first_derivative(S2, dz)
            S3_z = _central_first_derivative(S3, dz)
            # Override surface derivative by the Robin condition.
            S1_z[-1] = S1_flux
            S2_z[-1] = S2_flux
            S3_z[-1] = S3_flux

            S1_zz = _central_second_derivative_neumann_left(S1, dz, bc_right="robin", right_value=0.0, right_flux=S1_flux)
            S2_zz = _central_second_derivative_neumann_left(S2, dz, bc_right="robin", right_value=0.0, right_flux=S2_flux)
            S3_zz = _central_second_derivative_neumann_left(S3, dz, bc_right="robin", right_value=0.0, right_flux=S3_flux)

        # --- Time derivatives ---
        L_t = uL

        f1_t = (mu0_1 - mu_bar) * f1 - (u - zeta * uL) * f1_z / L
        f2_t = (mu0_2 - mu_bar) * f2 - (u - zeta * uL) * f2_z / L

        S1_t = r1 + D[0] * S1_zz / (L * L) + uL * zeta * S1_z / L
        S2_t = r2 + D[1] * S2_zz / (L * L) + uL * zeta * S2_z / L
        S3_t = r3 + D[2] * S3_zz / (L * L) + uL * zeta * S3_z / L

        # Enforce Dirichlet nodes as algebraic constraints (cases 1-4).
        SL_t = None
        if case_id != 5:
            S1_t[-1] = 0.0
            S2_t[-1] = 0.0
            S3_t[-1] = 0.0
        else:
            # Bulk liquid ODEs (Appendix B, Case 5 modifications).
            assert SL is not None
            VL = params.VR - params.AL * (L + params.LL)
            if VL <= 0.0:
                raise RuntimeError(f"Case 5 reactor volume became non-positive: VL={VL}")
            SI = np.asarray([params.S1_0, params.S2_0, params.S3_0], dtype=float)
            # D_Li = D_i / D_ratio.
            DL_over_LL = D / (params.D_ratio * params.LL)
            # Eq (38): dSL/dt = (Q(SI-SL) - AL*(D_L/LL - uL)*(SL - S_surface))/VL
            SL_t = np.zeros((3,), dtype=float)
            SL_t[2] = 0.0  # oxygen held constant
            SL_t[0] = (params.Q * (SI[0] - SL[0]) - params.AL * (DL_over_LL[0] - uL) * (SL[0] - S1[-1])) / VL
            SL_t[1] = (params.Q * (SI[1] - SL[1]) - params.AL * (DL_over_LL[1] - uL) * (SL[1] - S2[-1])) / VL

        return _pack_state(L=L_t, f1=f1_t, f2=f2_t, S1=S1_t, S2=S2_t, S3=S3_t, SL=SL_t)

    return rhs


def _initial_state(case_id: int, params: Warner1986Params) -> np.ndarray:
    n = int(params.npoint)
    f1 = np.full((n,), float(params.f1_0), dtype=float)
    f2 = np.full((n,), float(params.f2_0), dtype=float)
    S1 = np.full((n,), float(params.S1_0), dtype=float)
    S2 = np.full((n,), float(params.S2_0), dtype=float)
    S3 = np.full((n,), float(params.S3_0), dtype=float)

    SL = None
    if case_id == 5:
        SL = np.asarray([params.S1_0, params.S2_0, params.S3_0], dtype=float)
    return _pack_state(L=float(params.L0), f1=f1, f2=f2, S1=S1, S2=S2, S3=S3, SL=SL)


def _solve_case(
    *,
    case_id: int,
    params: Warner1986Params,
    rhs: Callable[[float, np.ndarray], np.ndarray],
    t_final: float,
    t_eval: np.ndarray,
    rtol: float,
    atol: float,
) -> object:
    y0 = _initial_state(case_id, params)
    # Case 2 has a discontinuous Dirichlet BC at t=6: integrate in two segments.
    if case_id == 2 and (t_eval.min() < 6.0 < t_final):
        from types import SimpleNamespace

        t_eval_1 = t_eval[t_eval <= 6.0]
        t_eval_2 = t_eval[t_eval >= 6.0]

        sol1 = solve_ivp(rhs, (float(t_eval_1[0]), 6.0), y0, method="BDF", t_eval=t_eval_1, rtol=rtol, atol=atol)
        if not sol1.success:
            raise RuntimeError(f"Case 2 segment 1 failed: {sol1.message}")

        y6 = np.asarray(sol1.y[:, -1], dtype=float)
        # Apply boundary step: organics at surface -> 0.0.
        n = int(params.npoint)
        # Layout: [L, f1(n), f2(n), S1(n), ...]
        off_s1 = 1 + 2 * n
        y6[off_s1 + (n - 1)] = 0.0

        sol2 = solve_ivp(rhs, (6.0, float(t_final)), y6, method="BDF", t_eval=t_eval_2, rtol=rtol, atol=atol)
        if not sol2.success:
            raise RuntimeError(f"Case 2 segment 2 failed: {sol2.message}")

        # Stitch (avoid duplicate t=6 if present twice).
        t = np.concatenate([sol1.t, sol2.t[1:]]) if sol2.t.size and sol1.t.size and sol1.t[-1] == sol2.t[0] else np.concatenate([sol1.t, sol2.t])
        y = np.concatenate([sol1.y, sol2.y[:, 1:]], axis=1) if sol2.y.shape[1] and sol1.y.shape[1] and sol1.t[-1] == sol2.t[0] else np.concatenate([sol1.y, sol2.y], axis=1)
        return SimpleNamespace(t=t, y=y, success=True, message="stitched")

    sol = solve_ivp(rhs, (float(t_eval[0]), float(t_final)), y0, method="BDF", t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"Case {case_id} failed: {sol.message}")
    return sol


def _table_vi_reference() -> dict[str, dict[str, np.ndarray]]:
    # Numeric reference from Table VI in warner1986.tex.
    ref = {}
    ref["case1_t6"] = {
        "z_um": np.asarray([0.0, 96.9, 193.7, 290.6, 387.5, 484.3, 581.2, 678.1]),
        "f1": np.asarray([0.482, 0.473, 0.505, 0.625, 0.751, 0.841, 0.898, 0.931]),
        "f2": np.asarray([0.220, 0.246, 0.242, 0.149, 0.067, 0.027, 0.011, 0.005]),
        "f3": np.asarray([0.298, 0.282, 0.253, 0.226, 0.182, 0.132, 0.091, 0.064]),
        "rho_u": np.asarray([0.000, 0.001, 0.015, 0.045, 0.055, 0.089, 0.261, 0.752]),
        "S1": np.asarray([0.052, 0.052, 0.055, 0.078, 0.164, 0.417, 1.132, 3.000]),
        "J1": np.asarray([0.000, -0.000, -0.007, -0.038, -0.121, -0.351, -0.969, -2.412]),
        "S2": np.asarray([11.192, 11.194, 11.219, 11.382, 11.702, 12.106, 12.546, 13.000]),
        "J2": np.asarray([0.000, -0.005, -0.113, -0.389, -0.574, -0.657, -0.696, -0.705]),
        "S3": np.asarray([-0.001, 0.005, 0.103, 0.730, 1.990, 3.645, 5.607, 8.000]),
        "J3": np.asarray([0.000, -0.021, -0.509, -1.771, -2.692, -3.263, -3.864, -4.883]),
    }
    ref["case4_t6"] = {
        "z_um": np.asarray([0.0, 25.4, 50.9, 76.3, 101.8, 127.2, 152.6, 178.1]),
        "f1": np.asarray([0.483, 0.482, 0.479, 0.476, 0.474, 0.475, 0.481, 0.495]),
        "f2": np.asarray([0.220, 0.222, 0.230, 0.239, 0.248, 0.255, 0.256, 0.249]),
        "f3": np.asarray([0.297, 0.295, 0.291, 0.286, 0.278, 0.270, 0.263, 0.255]),
        "rho_u": np.asarray([0.000, 0.081, 0.163, 0.250, 0.342, 0.442, 0.550, 0.670]),
        "S1": np.asarray([1.576, 1.603, 1.684, 1.820, 2.014, 2.272, 2.598, 3.000]),
        "J1": np.asarray([0.000, -0.174, -0.352, -0.538, -0.735, -0.948, -1.183, -1.446]),
        "S2": np.asarray([12.511, 12.521, 12.549, 12.597, 12.666, 12.755, 12.867, 13.000]),
        "J2": np.asarray([0.000, -0.110, -0.223, -0.340, -0.462, -0.589, -0.717, -0.845]),
        "S3": np.asarray([5.737, 5.780, 5.911, 6.133, 6.448, 6.862, 7.379, 8.000]),
        "J3": np.asarray([-0.000, -0.598, -1.209, -1.844, -2.506, -3.196, -3.911, -4.637]),
    }
    return ref


def _profiles_at_time(sol, *, t: float, case_id: int, params: Warner1986Params) -> dict[str, np.ndarray]:
    # Find exact index in t_eval.
    idx = int(np.argmin(np.abs(sol.t - float(t))))
    if abs(float(sol.t[idx]) - float(t)) > 1.0e-12:
        raise ValueError(f"Requested t={t} not found in t_eval (closest {sol.t[idx]}).")

    y = sol.y[:, idx]
    zeta, dz = _zeta_grid(params)
    n = int(params.npoint)

    L, f1, f2, S1, S2, S3, SL = _unpack_state(y, n=n, with_bulk=(case_id == 5))
    if case_id != 5:
        SL_surf = _surface_concentrations_case_1_2_3_4(case_id, t, params)
        S1 = np.asarray(S1, dtype=float).copy()
        S2 = np.asarray(S2, dtype=float).copy()
        S3 = np.asarray(S3, dtype=float).copy()
        S1[-1] = SL_surf[0]
        S2[-1] = SL_surf[1]
        S3[-1] = SL_surf[2]

    f3 = 1.0 - f1 - f2

    # Recompute mu_bar/u for rho*u output.
    Ko = np.asarray([params.Ko1, params.Ko2], dtype=float)
    Ks = np.asarray([params.Ks1, params.Ks2], dtype=float)
    mu_hat = np.asarray([params.mu_hat_1, params.mu_hat_2], dtype=float)
    b = np.asarray([params.b1, params.b2], dtype=float)
    k_inact = np.asarray([params.k1, params.k2], dtype=float)

    O2_term_1 = S3 / (Ko[0] + S3)
    O2_term_2 = S3 / (Ko[1] + S3)
    mu0_1 = (mu_hat[0] * S1 / (Ks[0] + S1) - b[0]) * O2_term_1 - k_inact[0]
    mu0_2 = (mu_hat[1] * S2 / (Ks[1] + S2) - b[1]) * O2_term_2 - k_inact[1]
    mu_bar = (mu0_1 + k_inact[0]) * f1 + (mu0_2 + k_inact[1]) * f2

    imob = np.zeros_like(mu_bar, dtype=float)
    imob[1:] = np.cumsum(0.5 * (mu_bar[1:] + mu_bar[:-1]) * dz)
    u = L * imob

    D = np.asarray([params.D1, params.D2, params.D3], dtype=float)
    # Fluxes J = -D/L * dS/dζ (as printed in UPDATE).
    S1_z = _central_first_derivative(S1, dz)
    S2_z = _central_first_derivative(S2, dz)
    S3_z = _central_first_derivative(S3, dz)
    if case_id == 5:
        coeff = L / (params.D_ratio * params.LL)
        assert SL is not None
        S1_z[-1] = coeff * (SL[0] - S1[-1])
        S2_z[-1] = coeff * (SL[1] - S2[-1])
        S3_z[-1] = coeff * (SL[2] - S3[-1])
    J1 = -D[0] * S1_z / L
    J2 = -D[1] * S2_z / L
    J3 = -D[2] * S3_z / L

    return {
        "z_um": 1.0e6 * zeta * L,
        "L_um": np.asarray([1.0e6 * L]),
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "rho_u": params.rho * u,
        "S1": S1,
        "S2": S2,
        "S3": S3,
        "J1": J1,
        "J2": J2,
        "J3": J3,
    }


def _time_series(sol, *, case_id: int, params: Warner1986Params) -> dict[str, np.ndarray]:
    """
    Derived time series used in Warner (1986) figures:
      - L(t)
      - substrate removal fluxes j_Li(t) at film-water interface (eq. 39)
      - (case 5) bulk liquid concentrations S_Li(t)
    """
    n = int(params.npoint)
    zeta, dz = _zeta_grid(params)
    D = np.asarray([params.D1, params.D2, params.D3], dtype=float)
    mu_hat = np.asarray([params.mu_hat_1, params.mu_hat_2], dtype=float)
    Ks = np.asarray([params.Ks1, params.Ks2], dtype=float)
    Ko = np.asarray([params.Ko1, params.Ko2], dtype=float)
    b = np.asarray([params.b1, params.b2], dtype=float)
    k_inact = np.asarray([params.k1, params.k2], dtype=float)

    t = np.asarray(sol.t, dtype=float)
    m = int(t.size)

    L = np.asarray(sol.y[0, :], dtype=float)
    j_diff = np.zeros((m, 3), dtype=float)
    j_L = np.zeros((m, 3), dtype=float)
    S_surf = np.zeros((m, 3), dtype=float)
    f_surf = np.zeros((m, 3), dtype=float)
    SL_bulk = np.full((m, 3), np.nan, dtype=float)

    for k in range(m):
        tk = float(t[k])
        yk = np.asarray(sol.y[:, k], dtype=float)
        Lk, f1, f2, S1, S2, S3, SL = _unpack_state(yk, n=n, with_bulk=(case_id == 5))

        # Surface concentrations and enforcement of Dirichlet BC (cases 1-4).
        if case_id != 5:
            SL_surf = _surface_concentrations_case_1_2_3_4(case_id, tk, params)
            S1 = np.asarray(S1, dtype=float).copy()
            S2 = np.asarray(S2, dtype=float).copy()
            S3 = np.asarray(S3, dtype=float).copy()
            S1[-1] = SL_surf[0]
            S2[-1] = SL_surf[1]
            S3[-1] = SL_surf[2]
        else:
            assert SL is not None
            SL_surf = SL
            SL_bulk[k, :] = SL

        # Mean observed specific growth rate + u_L for eq (39).
        O2_term_1 = S3 / (Ko[0] + S3)
        O2_term_2 = S3 / (Ko[1] + S3)
        mu0_1 = (mu_hat[0] * S1 / (Ks[0] + S1) - b[0]) * O2_term_1 - k_inact[0]
        mu0_2 = (mu_hat[1] * S2 / (Ks[1] + S2) - b[1]) * O2_term_2 - k_inact[1]
        mu_bar = (mu0_1 + k_inact[0]) * f1 + (mu0_2 + k_inact[1]) * f2
        imob = np.zeros_like(mu_bar, dtype=float)
        imob[1:] = np.cumsum(0.5 * (mu_bar[1:] + mu_bar[:-1]) * dz)
        uL = Lk * imob[-1] + _sigma(case_id, tk, Lk, params)

        # Diffusive flux at surface (inside biofilm).
        S1_z = _central_first_derivative(S1, dz)
        S2_z = _central_first_derivative(S2, dz)
        S3_z = _central_first_derivative(S3, dz)
        if case_id == 5:
            coeff = Lk / (params.D_ratio * params.LL)
            S1_z[-1] = coeff * (SL_surf[0] - S1[-1])
            S2_z[-1] = coeff * (SL_surf[1] - S2[-1])
            S3_z[-1] = coeff * (SL_surf[2] - S3[-1])

        j_diff[k, 0] = -D[0] * S1_z[-1] / Lk
        j_diff[k, 1] = -D[1] * S2_z[-1] / Lk
        j_diff[k, 2] = -D[2] * S3_z[-1] / Lk

        # Eq (39): j_Li = j_i(t,1) - u_L S_i(t,1)
        j_L[k, :] = j_diff[k, :] - uL * np.asarray([S1[-1], S2[-1], S3[-1]], dtype=float)

        S_surf[k, :] = np.asarray([S1[-1], S2[-1], S3[-1]], dtype=float)
        f3 = 1.0 - f1 - f2
        f_surf[k, :] = np.asarray([f1[-1], f2[-1], f3[-1]], dtype=float)

    return {
        "t_days": t,
        "L_um": 1.0e6 * L,
        "j_diff_1": j_diff[:, 0],
        "j_diff_2": j_diff[:, 1],
        "j_diff_3": j_diff[:, 2],
        "jL_1": j_L[:, 0],
        "jL_2": j_L[:, 1],
        "jL_3": j_L[:, 2],
        "S_surf_1": S_surf[:, 0],
        "S_surf_2": S_surf[:, 1],
        "S_surf_3": S_surf[:, 2],
        "f_surf_1": f_surf[:, 0],
        "f_surf_2": f_surf[:, 1],
        "f_surf_3": f_surf[:, 2],
        "SL_bulk_1": SL_bulk[:, 0],
        "SL_bulk_2": SL_bulk[:, 1],
        "SL_bulk_3": SL_bulk[:, 2],
    }


def _compare_to_table_vi(*, prof: dict[str, np.ndarray], ref: dict[str, np.ndarray], label: str) -> dict[str, float]:
    # Table VI prints every other node (N=15 -> indices 0,2,...,14).
    idx = np.asarray([0, 2, 4, 6, 8, 10, 12, 14], dtype=int)

    out: dict[str, float] = {}
    out["L_um_err_abs"] = float(abs(float(prof["z_um"][idx[-1]]) - float(ref["z_um"][-1])))

    for key in ("f1", "f2", "f3", "rho_u", "S1", "S2", "S3", "J1", "J2", "J3"):
        val = np.asarray(prof[key], dtype=float)[idx]
        target = np.asarray(ref[key], dtype=float)
        out[f"{key}_err_inf"] = float(np.max(np.abs(val - target)))
        out[f"{key}_err_l2"] = float(np.sqrt(np.mean((val - target) ** 2)))
    return out


def _import_compiled_module(module_name: str, path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


def _rhs_cpp(case_id: int, params: Warner1986Params) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Build/load a pybind11 extension implementing the RHS for the given case.
    """
    from pycutfem.jit.cpp_backend.compiler import compile_extension, get_compile_mode_tag

    # Keep the translation unit small and case-specialized to reduce compile time.
    # Cache key includes (case_id, params) and an explicit ABI tag so changes to the
    # generated C++ source trigger recompilation.
    cpp_abi = "2026-02-22-warner1986-rhs-v2"
    mode_tag = get_compile_mode_tag()
    src_tag = f"{cpp_abi}_case{case_id}_{mode_tag}"
    src_hash = hashlib.sha256((src_tag + repr(params)).encode("utf-8")).hexdigest()
    module_name = f"_pycutfem_warner1986_rhs_{src_hash}"

    import os

    cache_root = Path(
        os.environ.get("PYCUTFEM_CACHE_DIR", str(Path.home() / ".cache" / "pycutfem_jit"))
    ).expanduser().resolve()
    cache_dir = cache_root / "warner1986_cpp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    so_path = cache_dir / f"{module_name}{suffix}"
    cpp_path = cache_dir / f"{module_name}.cpp"

    if not so_path.exists():
        # Generate C++ source.
        zeta, dz = _zeta_grid(params)
        n = int(params.npoint)

        # fmt: off
        src = f"""
// Generated by warner1986_benchmark.py
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <array>

namespace py = pybind11;

static constexpr int N = {n};
static constexpr double dz = {dz:.17g};

static inline double sigma(double t, double L) {{
    if ({case_id} == 1 || {case_id} == 2 || {case_id} == 5) return 0.0;
    if ({case_id} == 3) return -{params.shear_lambda:.17g} * L * L;
    if ({case_id} == 4) {{
        if ({params.slough_t0:.17g} < t && t < {params.slough_t1:.17g}) return {params.slough_sigma:.17g};
        return 0.0;
    }}
    return 0.0;
}}

static inline void central_first(const double* a, double* out) {{
    out[0] = (-3.0*a[0] + 4.0*a[1] - a[2]) / (2.0*dz);
    for (int i = 1; i < N-1; ++i) out[i] = (a[i+1] - a[i-1]) / (2.0*dz);
    out[N-1] = (3.0*a[N-1] - 4.0*a[N-2] + a[N-3]) / (2.0*dz);
}}

static inline void second_neumann_left_dirichlet_right(const double* a_in, double* out, double right_value) {{
    double a[N];
    for (int i = 0; i < N; ++i) a[i] = a_in[i];
    a[N-1] = right_value;
    const double ghost_left = a[1];
    out[0] = (a[1] - 2.0*a[0] + ghost_left) / (dz*dz);
    for (int i = 1; i < N-1; ++i) out[i] = (a[i+1] - 2.0*a[i] + a[i-1]) / (dz*dz);
    // right boundary (linear extrapolation ghost)
    const double ghost_right = 2.0*a[N-1] - a[N-2];
    out[N-1] = (ghost_right - 2.0*a[N-1] + a[N-2]) / (dz*dz);
}}

static inline void second_neumann_left_robin_right(const double* a_in, double* out, double right_flux) {{
    const double* a = a_in;
    const double ghost_left = a[1];
    out[0] = (a[1] - 2.0*a[0] + ghost_left) / (dz*dz);
    for (int i = 1; i < N-1; ++i) out[i] = (a[i+1] - 2.0*a[i] + a[i-1]) / (dz*dz);
    const double ghost_right = a[N-1] + dz * right_flux;
    out[N-1] = (ghost_right - 2.0*a[N-1] + a[N-2]) / (dz*dz);
}}

py::array_t<double> rhs(double t, py::array_t<double, py::array::c_style | py::array::forcecast> y) {{
    const int with_bulk = ({case_id} == 5) ? 1 : 0;
    const int n_expected = 1 + 5*N + (with_bulk ? 3 : 0);
    auto ybuf = y.request();
    if (ybuf.ndim != 1) throw std::runtime_error("Expected y to be 1D");
    if ((int)ybuf.shape[0] != n_expected) throw std::runtime_error("Bad y size");
    const double* yptr = static_cast<const double*>(ybuf.ptr);
    py::array_t<double> dy(n_expected);
    auto dv = dy.mutable_unchecked<1>();

    int off = 0;
    const double L = yptr[off++];
    const double* f1 = yptr + off; off += N;
    const double* f2 = yptr + off; off += N;
    const double* S1_in = yptr + off; off += N;
    const double* S2_in = yptr + off; off += N;
    const double* S3_in = yptr + off; off += N;
    const double* SL = with_bulk ? (yptr + off) : nullptr;

    // Copy substrates to allow clamping surface BC for cases 1-4.
    double S1[N], S2[N], S3[N];
    for (int i = 0; i < N; ++i) {{ S1[i] = S1_in[i]; S2[i] = S2_in[i]; S3[i] = S3_in[i]; }}

    std::array<double,3> SL_surf = {{ {params.S1_0:.17g}, {params.S2_0:.17g}, {params.S3_0:.17g} }};
    if ({case_id} == 2) {{
        SL_surf[0] = (t < 6.0) ? {params.S1_0:.17g} : 0.0;
    }}
    if (with_bulk) {{
        SL_surf[0] = SL[0]; SL_surf[1] = SL[1]; SL_surf[2] = SL[2];
    }} else {{
        S1[N-1] = SL_surf[0];
        S2[N-1] = SL_surf[1];
        S3[N-1] = SL_surf[2];
    }}

    const double sig = sigma(t, L);

    // Kinetics
    const double rho = {params.rho:.17g};
    const double mu_hat_1 = {params.mu_hat_1:.17g};
    const double mu_hat_2 = {params.mu_hat_2:.17g};
    const double Ks1 = {params.Ks1:.17g};
    const double Ks2 = {params.Ks2:.17g};
    const double Ko1 = {params.Ko1:.17g};
    const double Ko2 = {params.Ko2:.17g};
    const double Y1 = {params.Y1:.17g};
    const double Y2 = {params.Y2:.17g};
    const double b1 = {params.b1:.17g};
    const double b2 = {params.b2:.17g};
    const double k1 = {params.k1:.17g};
    const double k2 = {params.k2:.17g};
    const double alpha1 = {params.alpha1:.17g};
    const double alpha2 = {params.alpha2:.17g};

    double mu0_1[N], mu0_2[N], mu_bar[N];
    double r1[N], r2[N], r3[N];
    for (int i = 0; i < N; ++i) {{
        const double O2_1 = S3[i] / (Ko1 + S3[i]);
        const double O2_2 = S3[i] / (Ko2 + S3[i]);
        mu0_1[i] = (mu_hat_1 * S1[i] / (Ks1 + S1[i]) - b1) * O2_1 - k1;
        mu0_2[i] = (mu_hat_2 * S2[i] / (Ks2 + S2[i]) - b2) * O2_2 - k2;
        r1[i] = -mu_hat_1 * rho * f1[i] * O2_1 * S1[i] / (Ks1 + S1[i]) / Y1;
        r2[i] = -mu_hat_2 * rho * f2[i] * O2_2 * S2[i] / (Ks2 + S2[i]) / Y2;
        r3[i] = (alpha1 - Y1) * r1[i] - b1 * rho * f1[i] * O2_1 + (alpha2 - Y2) * r2[i] - b2 * rho * f2[i] * O2_2;
        mu_bar[i] = (mu0_1[i] + k1) * f1[i] + (mu0_2[i] + k2) * f2[i];
    }}

    // Velocities via cumulative trapezoid.
    double imob[N]; imob[0] = 0.0;
    for (int i = 1; i < N; ++i) {{
        imob[i] = imob[i-1] + 0.5 * (mu_bar[i] + mu_bar[i-1]) * dz;
    }}
    double u[N];
    for (int i = 0; i < N; ++i) u[i] = L * imob[i];
    const double uL = L * imob[N-1] + sig;

    // Derivatives.
    double f1_z[N], f2_z[N];
    central_first(f1, f1_z);
    central_first(f2, f2_z);

    double S1_z[N], S2_z[N], S3_z[N];
    central_first(S1, S1_z);
    central_first(S2, S2_z);
    central_first(S3, S3_z);

    double S1_zz[N], S2_zz[N], S3_zz[N];
    if (!with_bulk) {{
        second_neumann_left_dirichlet_right(S1, S1_zz, SL_surf[0]);
        second_neumann_left_dirichlet_right(S2, S2_zz, SL_surf[1]);
        second_neumann_left_dirichlet_right(S3, S3_zz, SL_surf[2]);
    }} else {{
        const double coeff = L / ({params.D_ratio:.17g} * {params.LL:.17g});
        const double flux1 = coeff * (SL_surf[0] - S1[N-1]);
        const double flux2 = coeff * (SL_surf[1] - S2[N-1]);
        const double flux3 = coeff * (SL_surf[2] - S3[N-1]);
        S1_z[N-1] = flux1; S2_z[N-1] = flux2; S3_z[N-1] = flux3;
        second_neumann_left_robin_right(S1, S1_zz, flux1);
        second_neumann_left_robin_right(S2, S2_zz, flux2);
        second_neumann_left_robin_right(S3, S3_zz, flux3);
    }}

    // Pack dy.
    off = 0;
    dv(off++) = uL; // dL/dt

    for (int i = 0; i < N; ++i) {{
        const double z = (double)i * dz; // ζ
        dv(off + i) = (mu0_1[i] - mu_bar[i]) * f1[i] - (u[i] - z * uL) * f1_z[i] / L;
    }}
    off += N;
    for (int i = 0; i < N; ++i) {{
        const double z = (double)i * dz;
        dv(off + i) = (mu0_2[i] - mu_bar[i]) * f2[i] - (u[i] - z * uL) * f2_z[i] / L;
    }}
    off += N;

    const double D1 = {params.D1:.17g}, D2 = {params.D2:.17g}, D3 = {params.D3:.17g};
    for (int i = 0; i < N; ++i) {{
        const double z = (double)i * dz;
        dv(off + i) = r1[i] + D1 * S1_zz[i] / (L*L) + uL * z * S1_z[i] / L;
    }}
    if (!with_bulk) dv(off + (N-1)) = 0.0;
    off += N;
    for (int i = 0; i < N; ++i) {{
        const double z = (double)i * dz;
        dv(off + i) = r2[i] + D2 * S2_zz[i] / (L*L) + uL * z * S2_z[i] / L;
    }}
    if (!with_bulk) dv(off + (N-1)) = 0.0;
    off += N;
    for (int i = 0; i < N; ++i) {{
        const double z = (double)i * dz;
        dv(off + i) = r3[i] + D3 * S3_zz[i] / (L*L) + uL * z * S3_z[i] / L;
    }}
    if (!with_bulk) dv(off + (N-1)) = 0.0;
    off += N;

    if (with_bulk) {{
        const double AL = {params.AL:.17g};
        const double LL = {params.LL:.17g};
        const double Q = {params.Q:.17g};
        const double VR = {params.VR:.17g};
        const double VL = VR - AL * (L + LL);
        if (VL <= 0.0) throw std::runtime_error("VL <= 0");
        const double SI1 = {params.S1_0:.17g};
        const double SI2 = {params.S2_0:.17g};
        const double DL1_over_LL = D1 / ({params.D_ratio:.17g} * LL);
        const double DL2_over_LL = D2 / ({params.D_ratio:.17g} * LL);
        // oxygen held constant
        dv(off + 2) = 0.0;
        dv(off + 0) = (Q*(SI1 - SL_surf[0]) - AL*(DL1_over_LL - uL)*(SL_surf[0] - S1[N-1])) / VL;
        dv(off + 1) = (Q*(SI2 - SL_surf[1]) - AL*(DL2_over_LL - uL)*(SL_surf[1] - S2[N-1])) / VL;
    }}

    return dy;
}}

PYBIND11_MODULE({module_name}, m) {{
    m.def("rhs", &rhs, "Compute RHS for Warner1986 benchmark");
}}
"""
        # fmt: on

        cpp_path.write_text(src, encoding="utf-8")
        compile_extension(module_name, cpp_path, cache_dir, compile_mode=mode_tag)

    mod = _import_compiled_module(module_name, so_path)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.asarray(mod.rhs(float(t), np.asarray(y, dtype=float)), dtype=float)

    return rhs


def _rhs_jit(case_id: int, params: Warner1986Params) -> Callable[[float, np.ndarray], np.ndarray]:
    try:
        import numba  # noqa: F401
        from numba import njit
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("backend='jit' requested but numba is not available") from exc

    zeta, dz = _zeta_grid(params)
    n = int(params.npoint)
    D = np.asarray([params.D1, params.D2, params.D3], dtype=np.float64)

    mu_hat = np.asarray([params.mu_hat_1, params.mu_hat_2], dtype=np.float64)
    Ks = np.asarray([params.Ks1, params.Ks2], dtype=np.float64)
    Ko = np.asarray([params.Ko1, params.Ko2], dtype=np.float64)
    Y = np.asarray([params.Y1, params.Y2], dtype=np.float64)
    b = np.asarray([params.b1, params.b2], dtype=np.float64)
    k_inact = np.asarray([params.k1, params.k2], dtype=np.float64)
    alpha = np.asarray([params.alpha1, params.alpha2], dtype=np.float64)

    rho = float(params.rho)
    shear_lambda = float(params.shear_lambda)
    slough_t0 = float(params.slough_t0)
    slough_t1 = float(params.slough_t1)
    slough_sigma = float(params.slough_sigma)
    S1_0 = float(params.S1_0)
    S2_0 = float(params.S2_0)
    S3_0 = float(params.S3_0)
    D_ratio = float(params.D_ratio)
    LL = float(params.LL)
    AL = float(params.AL)
    Q = float(params.Q)
    VR = float(params.VR)

    with_bulk = bool(case_id == 5)
    y_size = 1 + 5 * n + (3 if with_bulk else 0)

    @njit(cache=True)
    def rhs_numba(t: float, y: np.ndarray) -> np.ndarray:
        dy = np.zeros((y_size,), dtype=np.float64)

        off = 0
        L = y[off]
        off += 1
        f1 = y[off : off + n]
        off += n
        f2 = y[off : off + n]
        off += n
        S1 = y[off : off + n].copy()
        off += n
        S2 = y[off : off + n].copy()
        off += n
        S3 = y[off : off + n].copy()
        off += n
        SL = None
        if with_bulk:
            SL = y[off : off + 3]

        # Sigma.
        sig = 0.0
        if case_id == 3:
            sig = -shear_lambda * L * L
        elif case_id == 4:
            if slough_t0 < t < slough_t1:
                sig = slough_sigma

        # Surface concentrations.
        SL0 = np.array([S1_0, S2_0, S3_0], dtype=np.float64)
        if case_id == 2:
            SL0[0] = S1_0 if t < 6.0 else 0.0
        if with_bulk:
            SL_surf = SL
        else:
            SL_surf = SL0
            S1[-1] = SL_surf[0]
            S2[-1] = SL_surf[1]
            S3[-1] = SL_surf[2]

        # Kinetics.
        mu0_1 = np.zeros((n,), dtype=np.float64)
        mu0_2 = np.zeros((n,), dtype=np.float64)
        r1 = np.zeros((n,), dtype=np.float64)
        r2 = np.zeros((n,), dtype=np.float64)
        r3 = np.zeros((n,), dtype=np.float64)
        mu_bar = np.zeros((n,), dtype=np.float64)
        for i in range(n):
            O2_1 = S3[i] / (Ko[0] + S3[i])
            O2_2 = S3[i] / (Ko[1] + S3[i])
            mu0_1[i] = (mu_hat[0] * S1[i] / (Ks[0] + S1[i]) - b[0]) * O2_1 - k_inact[0]
            mu0_2[i] = (mu_hat[1] * S2[i] / (Ks[1] + S2[i]) - b[1]) * O2_2 - k_inact[1]
            r1[i] = -mu_hat[0] * rho * f1[i] * O2_1 * S1[i] / (Ks[0] + S1[i]) / Y[0]
            r2[i] = -mu_hat[1] * rho * f2[i] * O2_2 * S2[i] / (Ks[1] + S2[i]) / Y[1]
            r3[i] = (alpha[0] - Y[0]) * r1[i] - b[0] * rho * f1[i] * O2_1 + (alpha[1] - Y[1]) * r2[i] - b[1] * rho * f2[i] * O2_2
            mu_bar[i] = (mu0_1[i] + k_inact[0]) * f1[i] + (mu0_2[i] + k_inact[1]) * f2[i]

        # Cumulative integral.
        imob = np.zeros((n,), dtype=np.float64)
        for i in range(1, n):
            imob[i] = imob[i - 1] + 0.5 * (mu_bar[i] + mu_bar[i - 1]) * dz
        u = L * imob
        uL = L * imob[-1] + sig

        # Derivatives (first).
        def central_first(a):
            out = np.zeros((n,), dtype=np.float64)
            out[0] = (-3.0 * a[0] + 4.0 * a[1] - a[2]) / (2.0 * dz)
            for j in range(1, n - 1):
                out[j] = (a[j + 1] - a[j - 1]) / (2.0 * dz)
            out[-1] = (3.0 * a[-1] - 4.0 * a[-2] + a[-3]) / (2.0 * dz)
            return out

        f1_z = central_first(f1)
        f2_z = central_first(f2)
        S1_z = central_first(S1)
        S2_z = central_first(S2)
        S3_z = central_first(S3)

        # Second derivatives with Neumann at left.
        def second_neumann_left_dirichlet_right(a, right_value):
            aa = a.copy()
            aa[-1] = right_value
            out = np.zeros((n,), dtype=np.float64)
            ghost_left = aa[1]
            out[0] = (aa[1] - 2.0 * aa[0] + ghost_left) / (dz * dz)
            for j in range(1, n - 1):
                out[j] = (aa[j + 1] - 2.0 * aa[j] + aa[j - 1]) / (dz * dz)
            ghost_right = 2.0 * aa[-1] - aa[-2]
            out[-1] = (ghost_right - 2.0 * aa[-1] + aa[-2]) / (dz * dz)
            return out

        def second_neumann_left_robin_right(a, right_flux):
            out = np.zeros((n,), dtype=np.float64)
            ghost_left = a[1]
            out[0] = (a[1] - 2.0 * a[0] + ghost_left) / (dz * dz)
            for j in range(1, n - 1):
                out[j] = (a[j + 1] - 2.0 * a[j] + a[j - 1]) / (dz * dz)
            ghost_right = a[-1] + dz * right_flux
            out[-1] = (ghost_right - 2.0 * a[-1] + a[-2]) / (dz * dz)
            return out

        if not with_bulk:
            S1_zz = second_neumann_left_dirichlet_right(S1, SL_surf[0])
            S2_zz = second_neumann_left_dirichlet_right(S2, SL_surf[1])
            S3_zz = second_neumann_left_dirichlet_right(S3, SL_surf[2])
        else:
            coeff = L / (D_ratio * LL)
            flux1 = coeff * (SL_surf[0] - S1[-1])
            flux2 = coeff * (SL_surf[1] - S2[-1])
            flux3 = coeff * (SL_surf[2] - S3[-1])
            S1_z[-1] = flux1
            S2_z[-1] = flux2
            S3_z[-1] = flux3
            S1_zz = second_neumann_left_robin_right(S1, flux1)
            S2_zz = second_neumann_left_robin_right(S2, flux2)
            S3_zz = second_neumann_left_robin_right(S3, flux3)

        # Pack dy.
        off2 = 0
        dy[off2] = uL
        off2 += 1

        for i in range(n):
            dy[off2 + i] = (mu0_1[i] - mu_bar[i]) * f1[i] - (u[i] - zeta[i] * uL) * f1_z[i] / L
        off2 += n
        for i in range(n):
            dy[off2 + i] = (mu0_2[i] - mu_bar[i]) * f2[i] - (u[i] - zeta[i] * uL) * f2_z[i] / L
        off2 += n

        for i in range(n):
            dy[off2 + i] = r1[i] + D[0] * S1_zz[i] / (L * L) + uL * zeta[i] * S1_z[i] / L
        if not with_bulk:
            dy[off2 + (n - 1)] = 0.0
        off2 += n
        for i in range(n):
            dy[off2 + i] = r2[i] + D[1] * S2_zz[i] / (L * L) + uL * zeta[i] * S2_z[i] / L
        if not with_bulk:
            dy[off2 + (n - 1)] = 0.0
        off2 += n
        for i in range(n):
            dy[off2 + i] = r3[i] + D[2] * S3_zz[i] / (L * L) + uL * zeta[i] * S3_z[i] / L
        if not with_bulk:
            dy[off2 + (n - 1)] = 0.0
        off2 += n

        if with_bulk:
            SLt = np.zeros((3,), dtype=np.float64)
            SLt[2] = 0.0
            VL = VR - AL * (L + LL)
            DL_over_LL = D / (D_ratio * LL)
            SLt[0] = (Q * (S1_0 - SL_surf[0]) - AL * (DL_over_LL[0] - uL) * (SL_surf[0] - S1[-1])) / VL
            SLt[1] = (Q * (S2_0 - SL_surf[1]) - AL * (DL_over_LL[1] - uL) * (SL_surf[1] - S2[-1])) / VL
            dy[off2 : off2 + 3] = SLt

        return dy

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.asarray(rhs_numba(float(t), np.asarray(y, dtype=np.float64)), dtype=float)

    return rhs


def _build_rhs(case_id: int, params: Warner1986Params, backend: Backend) -> Callable[[float, np.ndarray], np.ndarray]:
    if backend == "python":
        return _rhs_python(case_id, params)
    if backend == "jit":
        return _rhs_jit(case_id, params)
    if backend == "cpp":
        return _rhs_cpp(case_id, params)
    raise ValueError(f"Unknown backend={backend}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Warner & Gujer (1986) 1D multispecies biofilm benchmark (5 cases).")
    ap.add_argument("--case", type=str, default="all", choices=("all", "1", "2", "3", "4", "5"))
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--compare-backends", action="store_true", help="Assemble/run python+jit+cpp and compare final states.")
    ap.add_argument("--t-final", type=float, default=10.0)
    ap.add_argument("--dt-out", type=float, default=0.5, help="Output interval (days). Matches DTOUT=0.5 in UPDATE.")
    ap.add_argument("--rtol", type=float, default=1.0e-7)
    ap.add_argument("--atol", type=float, default=1.0e-9)
    ap.add_argument("--outdir", type=str, default="examples/biofilms/results/warner1986")
    ap.add_argument("--paper-figdir", type=str, default="", help="Optional directory to write LaTeX-ready PDF figures.")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    params = Warner1986Params()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paper_figdir = Path(args.paper_figdir).expanduser().resolve() if args.paper_figdir else None
    if paper_figdir is not None:
        paper_figdir.mkdir(parents=True, exist_ok=True)

    cases = [1, 2, 3, 4, 5] if args.case == "all" else [int(args.case)]
    t_eval = np.arange(0.0, float(args.t_final) + 1.0e-12, float(args.dt_out), dtype=float)
    if 6.0 not in t_eval:
        t_eval = np.unique(np.sort(np.concatenate([t_eval, np.asarray([6.0])])))

    refs = _table_vi_reference()
    ts_by_case: dict[int, dict[str, np.ndarray]] = {}

    def _run_one(case_id: int, backend: Backend):
        rhs = _build_rhs(case_id, params, backend=backend)
        sol = _solve_case(
            case_id=case_id,
            params=params,
            rhs=rhs,
            t_final=float(args.t_final),
            t_eval=t_eval,
            rtol=float(args.rtol),
            atol=float(args.atol),
        )
        return sol

    for case_id in cases:
        print(f"\n=== Warner1986 case {case_id} | backend={args.backend} ===")
        sol = _run_one(case_id, backend=str(args.backend))

        ts = _time_series(sol, case_id=case_id, params=params)
        ts_by_case[case_id] = ts
        ts_keys = [
            "t_days",
            "L_um",
            "jL_1",
            "jL_2",
            "jL_3",
            "j_diff_1",
            "j_diff_2",
            "j_diff_3",
            "S_surf_1",
            "S_surf_2",
            "S_surf_3",
            "f_surf_1",
            "f_surf_2",
            "f_surf_3",
            "SL_bulk_1",
            "SL_bulk_2",
            "SL_bulk_3",
        ]
        np.savetxt(
            outdir / f"case{case_id}_backend={args.backend}_timeseries.csv",
            np.column_stack([ts[k] for k in ts_keys]),
            delimiter=",",
            header=",".join(ts_keys),
            comments="",
        )

        if case_id in (1, 4):
            prof = _profiles_at_time(sol, t=6.0, case_id=case_id, params=params)
            key = "case1_t6" if case_id == 1 else "case4_t6"
            err = _compare_to_table_vi(prof=prof, ref=refs[key], label=key)
            print(f"Table VI check ({key}):")
            for k, v in err.items():
                print(f"  {k}: {v:.3e}")
            # Persist errors for paper.
            (outdir / f"case{case_id}_backend={args.backend}_tableVI_errors.txt").write_text(
                "\n".join([f"{k} {v:.16e}" for k, v in err.items()]) + "\n",
                encoding="utf-8",
            )
            prof_keys = ["z_um", "f1", "f2", "f3", "rho_u", "S1", "S2", "S3", "J1", "J2", "J3"]
            np.savetxt(
                outdir / f"case{case_id}_backend={args.backend}_profiles_t=6d.csv",
                np.column_stack([prof[k] for k in prof_keys]),
                delimiter=",",
                header=",".join(prof_keys),
                comments="",
            )

        if args.compare_backends:
            sols = {args.backend: sol}
            for other in ("python", "jit", "cpp"):
                if other == args.backend:
                    continue
                try:
                    sols[other] = _run_one(case_id, backend=other)  # type: ignore[arg-type]
                except Exception as exc:
                    print(f"[compare-backends] backend={other} failed: {exc}")
            if len(sols) >= 2:
                keys = list(sols.keys())
                y_ref = np.asarray(sols[keys[0]].y[:, -1], dtype=float)
                for k in keys[1:]:
                    y = np.asarray(sols[k].y[:, -1], dtype=float)
                    dy = float(np.max(np.abs(y - y_ref)))
                    print(f"[compare-backends] max|y_final({k})-y_final({keys[0]})| = {dy:.3e}")

        if not args.no_plots:
            import matplotlib.pyplot as plt

            # Thickness.
            plt.figure()
            plt.plot(ts["t_days"], ts["L_um"], "-o", ms=3)
            plt.xlabel("t [d]")
            plt.ylabel("L [µm]")
            plt.title(f"Warner1986 case {case_id}: thickness (backend={args.backend})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(outdir / f"case{case_id}_backend={args.backend}_L_um.png", dpi=200)
            if paper_figdir is not None:
                plt.savefig(paper_figdir / f"warner1986_case{case_id}_L_um_backend={args.backend}.pdf")
            plt.close()

            # Substrate removal from bulk liquid: j_Li (eq. 39).
            plt.figure()
            plt.plot(ts["t_days"], ts["jL_1"], label="COD (j_L1)")
            plt.plot(ts["t_days"], ts["jL_2"], label="NH4 (j_L2)")
            plt.plot(ts["t_days"], ts["jL_3"], label="O2 (j_L3)")
            plt.xlabel("t [d]")
            plt.ylabel("j_L [g m$^{-2}$ d$^{-1}$]")
            plt.title(f"Warner1986 case {case_id}: substrate removal (backend={args.backend})")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / f"case{case_id}_backend={args.backend}_jL.png", dpi=200)
            if paper_figdir is not None:
                plt.savefig(paper_figdir / f"warner1986_case{case_id}_jL_backend={args.backend}.pdf")
            plt.close()

            if case_id == 5:
                plt.figure()
                plt.plot(ts["t_days"], ts["SL_bulk_1"], label="S_L1 (COD)")
                plt.plot(ts["t_days"], ts["SL_bulk_2"], label="S_L2 (NH4)")
                plt.xlabel("t [d]")
                plt.ylabel("S_L [g m$^{-3}$]")
                plt.title(f"Warner1986 case 5: bulk liquid concentrations (backend={args.backend})")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(outdir / f"case{case_id}_backend={args.backend}_bulk_SL.png", dpi=200)
                if paper_figdir is not None:
                    plt.savefig(paper_figdir / f"warner1986_case5_bulk_SL_backend={args.backend}.pdf")
                plt.close()

            if case_id == 1:
                prof = _profiles_at_time(sol, t=6.0, case_id=case_id, params=params)
                z_um = prof["z_um"]
                plt.figure()
                plt.plot(z_um, prof["S1"], label="S1 (COD)")
                plt.plot(z_um, prof["S2"], label="S2 (NH4)")
                plt.plot(z_um, prof["S3"], label="S3 (O2)")
                plt.gca().invert_xaxis()
                plt.xlabel("z [µm] (0=base, L=surface)")
                plt.ylabel("S [g m$^{-3}$]")
                plt.title(f"Warner1986 case 1: substrate profiles at t=6 d (backend={args.backend})")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(outdir / f"case{case_id}_backend={args.backend}_profiles_t=6d_substrates.png", dpi=200)
                if paper_figdir is not None:
                    plt.savefig(paper_figdir / f"warner1986_case1_profiles_t6d_substrates_backend={args.backend}.pdf")
                plt.close()

    if not args.no_plots and len(ts_by_case) > 1:
        import matplotlib.pyplot as plt

        cases_sorted = sorted(ts_by_case.keys())
        plt.figure()
        for cid in cases_sorted:
            ts = ts_by_case[cid]
            plt.plot(ts["t_days"], ts["L_um"], "-o", ms=2, label=f"Case {cid}")
        plt.xlabel("t [d]")
        plt.ylabel("L [µm]")
        plt.title(f"Warner1986: thickness comparison (backend={args.backend})")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(outdir / f"all_cases_backend={args.backend}_L_um.png", dpi=200)
        if paper_figdir is not None:
            plt.savefig(paper_figdir / f"warner1986_all_cases_L_um_backend={args.backend}.pdf")
        plt.close()

        fig, ax = plt.subplots(3, 1, figsize=(6.0, 7.0), sharex=True)
        for cid in cases_sorted:
            ts = ts_by_case[cid]
            ax[0].plot(ts["t_days"], ts["jL_1"], label=f"Case {cid}")
            ax[1].plot(ts["t_days"], ts["jL_2"], label=f"Case {cid}")
            ax[2].plot(ts["t_days"], ts["jL_3"], label=f"Case {cid}")
        ax[0].set_ylabel("j_L1 [g m$^{-2}$ d$^{-1}$]")
        ax[1].set_ylabel("j_L2 [g m$^{-2}$ d$^{-1}$]")
        ax[2].set_ylabel("j_L3 [g m$^{-2}$ d$^{-1}$]")
        ax[2].set_xlabel("t [d]")
        ax[0].grid(True, alpha=0.3)
        ax[1].grid(True, alpha=0.3)
        ax[2].grid(True, alpha=0.3)
        ax[0].legend(ncol=2, fontsize=8)
        fig.suptitle(f"Warner1986: substrate removal comparison (backend={args.backend})")
        fig.tight_layout()
        fig.savefig(outdir / f"all_cases_backend={args.backend}_jL.png", dpi=200)
        if paper_figdir is not None:
            fig.savefig(paper_figdir / f"warner1986_all_cases_jL_backend={args.backend}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
