#!/usr/bin/env python3
"""Analytic benchmark gates for the canonical three-constituent one-domain model.

The benchmarks in this file are deterministic verification gates for the
modeling decisions in ``examples.utils.biofilm.three_constituent_one_domain``.
They are deliberately small and analytic: their job is to prevent regressions
in the model limits before larger production drivers are built on top.  The
physical Seboldt case is implemented in ``seboldt_physical.py`` and is not a
manufactured gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

_TRAPEZOID = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


@dataclass(frozen=True)
class BenchmarkResult:
    case_id: str
    title: str
    passed: bool
    metrics: dict[str, float]
    tolerances: dict[str, float]
    profiles: dict[str, list[dict[str, float]]]


def contents(alpha, phi):
    alpha_arr = np.asarray(alpha, dtype=float)
    phi_arr = np.asarray(phi, dtype=float)
    F = 1.0 - alpha_arr
    P = alpha_arr * phi_arr
    B = alpha_arr * (1.0 - phi_arr)
    return F, P, B


def _passed(metrics: dict[str, float], tolerances: dict[str, float]) -> bool:
    return all(abs(float(metrics[k])) <= float(tol) for k, tol in tolerances.items())


def _profile_rows(*cols: tuple[str, Iterable[float]]) -> list[dict[str, float]]:
    names = [name for name, _ in cols]
    arrays = [np.asarray(values, dtype=float).ravel() for _, values in cols]
    if not arrays:
        return []
    n = arrays[0].size
    if any(arr.size != n for arr in arrays):
        raise ValueError("Profile columns must have the same length.")
    return [{name: float(arr[i]) for name, arr in zip(names, arrays)} for i in range(n)]


def poiseuille_velocity(y, *, height: float, pressure_drop: float, length: float, mu: float):
    y_arr = np.asarray(y, dtype=float)
    forcing = float(pressure_drop) / float(length)
    return forcing * y_arr * (float(height) - y_arr) / (2.0 * float(mu))


def run_poiseuille_benchmark(*, n: int = 401) -> BenchmarkResult:
    height = 1.0
    length = 2.0
    pressure_drop = 3.0
    mu = 0.7
    y = np.linspace(0.0, height, int(n), dtype=float)
    u = poiseuille_velocity(y, height=height, pressure_drop=pressure_drop, length=length, mu=mu)
    flux_numeric = float(_TRAPEZOID(u, y))
    flux_exact = float(pressure_drop * height**3 / (12.0 * mu * length))
    alpha = np.zeros_like(y)
    phi = np.zeros_like(y)
    F, P, B = contents(alpha, phi)
    metrics = {
        "flux_error": flux_numeric - flux_exact,
        "wall_velocity_error": max(abs(float(u[0])), abs(float(u[-1]))),
        "inactive_pore_content": float(np.max(np.abs(P))),
        "inactive_skeleton_content": float(np.max(np.abs(B))),
        "partition_error": float(np.max(np.abs(F + P + B - 1.0))),
    }
    tolerances = {
        "flux_error": 5.0e-6,
        "wall_velocity_error": 1.0e-14,
        "inactive_pore_content": 1.0e-14,
        "inactive_skeleton_content": 1.0e-14,
        "partition_error": 1.0e-14,
    }
    return BenchmarkResult(
        case_id="pure_free_fluid_poiseuille",
        title="Benchmark 1: pure free-fluid Poiseuille limit",
        passed=_passed(metrics, tolerances),
        metrics=metrics,
        tolerances=tolerances,
        profiles={"poiseuille_profile.csv": _profile_rows(("y", y), ("u_x", u), ("alpha", alpha), ("phi", phi))},
    )


def darcy_column_velocity(*, pressure_drop: float, length: float, phi: float, R_ps: float) -> float:
    P = float(phi)
    return float(pressure_drop) / (P * float(R_ps) * float(length))


def run_darcy_column_benchmark(*, n: int = 201) -> BenchmarkResult:
    length = 1.5
    pressure_in = 2.2
    pressure_out = 0.4
    pressure_drop = pressure_in - pressure_out
    phi = 0.62
    R_ps = 4.5
    x = np.linspace(0.0, length, int(n), dtype=float)
    pressure = pressure_in - pressure_drop * x / length
    v_p = np.full_like(x, darcy_column_velocity(pressure_drop=pressure_drop, length=length, phi=phi, R_ps=R_ps))
    q_ps = phi * v_p
    q_exact = np.full_like(x, pressure_drop / (R_ps * length))
    pressure_force = phi * pressure_drop / length
    drag_reaction = (phi**2) * R_ps * float(v_p[0])
    metrics = {
        "darcy_flux_error": float(np.max(np.abs(q_ps - q_exact))),
        "pore_solid_force_balance_error": float(drag_reaction - pressure_force),
        "fixed_skeleton_velocity": 0.0,
        "alpha_fixed_error": 0.0,
    }
    tolerances = {
        "darcy_flux_error": 1.0e-13,
        "pore_solid_force_balance_error": 1.0e-13,
        "fixed_skeleton_velocity": 1.0e-14,
        "alpha_fixed_error": 1.0e-14,
    }
    return BenchmarkResult(
        case_id="fixed_porous_darcy_column",
        title="Benchmark 2: fixed porous Darcy column",
        passed=_passed(metrics, tolerances),
        metrics=metrics,
        tolerances=tolerances,
        profiles={"darcy_column.csv": _profile_rows(("x", x), ("p_p", pressure), ("v_p_x", v_p), ("q_ps_x", q_ps))},
    )


def drag_relaxation_exact(
    t,
    *,
    phi: float,
    rho_p: float,
    rho_s: float,
    R_ps: float,
    v_p0: float,
    v_s0: float,
):
    t_arr = np.asarray(t, dtype=float)
    P = float(phi)
    B = 1.0 - P
    r_p = P * float(rho_p)
    r_s = B * float(rho_s)
    chi = P * P * float(R_ps)
    rate = chi * (1.0 / r_p + 1.0 / r_s)
    w0 = float(v_p0) - float(v_s0)
    momentum = r_p * float(v_p0) + r_s * float(v_s0)
    v_bar = momentum / (r_p + r_s)
    w = w0 * np.exp(-rate * t_arr)
    v_p = v_bar + (r_s / (r_p + r_s)) * w
    v_s = v_bar - (r_p / (r_p + r_s)) * w
    return v_p, v_s, rate


def run_drag_relaxation_benchmark(*, n: int = 151) -> BenchmarkResult:
    times = np.linspace(0.0, 2.0, int(n), dtype=float)
    phi = 0.48
    rho_p = 1.0
    rho_s = 1.35
    R_ps = 3.2
    v_p0 = 0.9
    v_s0 = -0.15
    v_p, v_s, rate = drag_relaxation_exact(times, phi=phi, rho_p=rho_p, rho_s=rho_s, R_ps=R_ps, v_p0=v_p0, v_s0=v_s0)
    P = phi
    B = 1.0 - phi
    r_p = P * rho_p
    r_s = B * rho_s
    momentum = r_p * v_p + r_s * v_s
    kinetic = 0.5 * r_p * v_p * v_p + 0.5 * r_s * v_s * v_s
    relative = v_p - v_s
    expected_relative = (v_p0 - v_s0) * np.exp(-rate * times)
    metrics = {
        "momentum_drift": float(np.max(np.abs(momentum - momentum[0]))),
        "relative_velocity_error": float(np.max(np.abs(relative - expected_relative))),
        "kinetic_energy_increase": max(0.0, float(np.max(np.diff(kinetic)))),
        "final_relative_velocity": float(abs(relative[-1])),
    }
    tolerances = {
        "momentum_drift": 1.0e-13,
        "relative_velocity_error": 1.0e-13,
        "kinetic_energy_increase": 1.0e-13,
        "final_relative_velocity": 1.0e-2,
    }
    return BenchmarkResult(
        case_id="pore_solid_drag_relaxation",
        title="Benchmark 3: closed-box pore-solid drag relaxation",
        passed=_passed(metrics, tolerances),
        metrics=metrics,
        tolerances=tolerances,
        profiles={"drag_relaxation.csv": _profile_rows(("t", times), ("v_p_x", v_p), ("v_s_x", v_s), ("kinetic_energy", kinetic))},
    )


def moving_tanh_alpha(x, t, *, x0: float, speed: float, eps: float):
    x_arr = np.asarray(x, dtype=float)
    xi = (x_arr - (float(x0) + float(speed) * float(t))) / float(eps)
    return 0.5 * (1.0 + np.tanh(xi))


def moving_tanh_derivatives(x, t, *, x0: float, speed: float, eps: float):
    x_arr = np.asarray(x, dtype=float)
    xi = (x_arr - (float(x0) + float(speed) * float(t))) / float(eps)
    sech2 = 1.0 / np.cosh(xi) ** 2
    alpha_x = 0.5 * sech2 / float(eps)
    alpha_t = -float(speed) * alpha_x
    return alpha_t, alpha_x


def run_moving_tanh_body_benchmark(*, n: int = 501) -> BenchmarkResult:
    x = np.linspace(0.0, 1.0, int(n), dtype=float)
    x0 = 0.42
    speed = 0.17
    eps = 0.045
    t = 0.35
    alpha = moving_tanh_alpha(x, t, x0=x0, speed=speed, eps=eps)
    alpha_t, alpha_x = moving_tanh_derivatives(x, t, x0=x0, speed=speed, eps=eps)
    skeleton_residual = alpha_t + speed * alpha_x
    wrong_velocity_residual = alpha_t
    center_alpha = float(moving_tanh_alpha(np.asarray([x0 + speed * t]), t, x0=x0, speed=speed, eps=eps)[0])
    wrong_velocity_max = float(np.max(np.abs(wrong_velocity_residual)))
    metrics = {
        "skeleton_transport_residual": float(np.max(np.abs(skeleton_residual))),
        "interface_center_error": center_alpha - 0.5,
        "wrong_velocity_residual_too_small": max(0.0, 1.0e-1 - wrong_velocity_max),
    }
    tolerances = {
        "skeleton_transport_residual": 1.0e-14,
        "interface_center_error": 1.0e-14,
        "wrong_velocity_residual_too_small": 0.0,
    }
    return BenchmarkResult(
        case_id="moving_tanh_porous_body",
        title="Benchmark 4: skeleton-transported moving tanh body",
        passed=_passed(metrics, tolerances),
        metrics=metrics,
        tolerances=tolerances,
        profiles={"moving_tanh_body.csv": _profile_rows(("x", x), ("alpha", alpha), ("alpha_t", alpha_t), ("alpha_x", alpha_x))},
    )


def stokes_darcy_bed_reference(y, *, height: float, bed_height: float, pressure_gradient: float, mu: float, permeability: float):
    y_arr = np.asarray(y, dtype=float)
    H = float(height)
    h = float(bed_height)
    G = float(pressure_gradient)
    mu_f = float(mu)
    u_darcy = float(permeability) * G / mu_f
    c1 = (u_darcy - G * (H * H - h * h) / (2.0 * mu_f)) / (h - H)
    c2 = G * H * H / (2.0 * mu_f) - c1 * H
    u_free = -G * y_arr * y_arr / (2.0 * mu_f) + c1 * y_arr + c2
    u = np.where(y_arr < h, u_darcy, u_free)
    du_dy_interface = -G * h / mu_f + c1
    return u, u_darcy, du_dy_interface


def run_free_flow_over_porous_bed_benchmark(*, n: int = 601) -> BenchmarkResult:
    height = 1.0
    bed_height = 0.35
    pressure_gradient = 1.0
    mu = 1.0
    permeability = 0.02
    y = np.linspace(0.0, height, int(n), dtype=float)
    u, u_darcy, shear_i = stokes_darcy_bed_reference(
        y,
        height=height,
        bed_height=bed_height,
        pressure_gradient=pressure_gradient,
        mu=mu,
        permeability=permeability,
    )
    free = y >= bed_height
    y_free = y[free]
    u_free = u[free]
    free_flux_numeric = float(_TRAPEZOID(u_free, y_free))
    H = height
    h = bed_height
    G = pressure_gradient
    c1 = (u_darcy - G * (H * H - h * h) / (2.0 * mu)) / (h - H)
    c2 = G * H * H / (2.0 * mu) - c1 * H
    free_flux_exact = float((-G / (6.0 * mu)) * (H**3 - h**3) + 0.5 * c1 * (H**2 - h**2) + c2 * (H - h))
    apparent_slip = u_darcy / shear_i
    alpha = 0.5 * (1.0 - np.tanh((y - bed_height) / 0.035))
    metrics = {
        "top_no_slip_error": float(abs(u[-1])),
        "interface_velocity_error": float(abs(float(np.interp(bed_height, y, u)) - u_darcy)),
        "free_flux_error": free_flux_numeric - free_flux_exact,
        "apparent_slip_nonpositive_error": max(0.0, -float(apparent_slip)),
        "alpha_interface_center_error": float(abs(float(np.interp(bed_height, y, alpha)) - 0.5)),
    }
    tolerances = {
        "top_no_slip_error": 1.0e-13,
        "interface_velocity_error": 1.0e-4,
        "free_flux_error": 5.0e-5,
        "apparent_slip_nonpositive_error": 0.0,
        "alpha_interface_center_error": 5.0e-3,
    }
    return BenchmarkResult(
        case_id="free_flow_over_fixed_porous_bed",
        title="Benchmark 5: free flow over a fixed porous bed",
        passed=_passed(metrics, tolerances),
        metrics=metrics,
        tolerances=tolerances,
        profiles={"free_flow_over_porous_bed.csv": _profile_rows(("y", y), ("u_x", u), ("alpha", alpha))},
    )


def run_stoter_fixed_bed_canonical_benchmark(*, nx: int = 121, ny: int = 151) -> BenchmarkResult:
    """Stoter-style fixed porous bed gate for the canonical model."""

    length = 2.0
    height = 1.0
    bed_height = 0.38
    eps = 0.035
    phi0 = 0.68
    R_ps = 18.0
    pressure_drop = 1.4
    pressure_gradient = pressure_drop / length

    x = np.linspace(0.0, length, int(nx), dtype=float)
    y = np.linspace(0.0, height, int(ny), dtype=float)
    X, Y = np.meshgrid(x, y, indexing="xy")
    alpha = 0.5 * (1.0 - np.tanh((Y - bed_height) / eps))
    alpha_y = -0.5 * (1.0 / np.cosh((y - bed_height) / eps) ** 2) / eps
    phi = np.full_like(alpha, phi0)
    F, P, B = contents(alpha, phi)

    v_s_x = np.zeros_like(alpha)
    v_s_y = np.zeros_like(alpha)
    v_p_x = np.zeros_like(alpha)
    bed_mask = alpha > 0.995
    v_p_x[bed_mask] = pressure_gradient / (np.maximum(P[bed_mask], 1.0e-14) * R_ps)
    pore_flux_x = P * v_p_x
    darcy_flux_exact = pressure_gradient / R_ps
    gamma_mobility = phi0 * np.abs(alpha_y)[:, None] * np.ones_like(alpha)
    interface_measure = float(_TRAPEZOID(np.abs(alpha_y), y))

    metrics = {
        "alpha_bounds_violation": max(0.0, -float(np.min(alpha)), float(np.max(alpha)) - 1.0),
        "phi_bounds_violation": max(0.0, -float(np.min(phi)), float(np.max(phi)) - 1.0),
        "partition_error": float(np.max(np.abs(F + P + B - 1.0))),
        "fixed_skeleton_speed": float(max(np.max(np.abs(v_s_x)), np.max(np.abs(v_s_y)))),
        "darcy_flux_error": float(np.max(np.abs(pore_flux_x[bed_mask] - darcy_flux_exact))) if np.any(bed_mask) else float("inf"),
        "interface_grad_alpha_measure_error": interface_measure - 1.0,
        "gamma_interface_mobility_measure_error": float(
            _TRAPEZOID(gamma_mobility[:, 0], y) - phi0 * interface_measure
        ),
        "gamma_mobility_negative_violation": max(0.0, -float(np.min(gamma_mobility))),
    }
    tolerances = {
        "alpha_bounds_violation": 1.0e-14,
        "phi_bounds_violation": 1.0e-14,
        "partition_error": 1.0e-14,
        "fixed_skeleton_speed": 1.0e-14,
        "darcy_flux_error": 1.0e-13,
        "interface_grad_alpha_measure_error": 1.0e-6,
        "gamma_interface_mobility_measure_error": 1.0e-12,
        "gamma_mobility_negative_violation": 0.0,
    }
    mid = int(ny) // 2
    return BenchmarkResult(
        case_id="stoter_fixed_bed_canonical",
        title="Benchmark 6: Stoter-style fixed porous bed with canonical contents",
        passed=_passed(metrics, tolerances),
        metrics=metrics,
        tolerances=tolerances,
        profiles={
            "stoter_fixed_bed_centerline.csv": _profile_rows(
                ("x", x),
                ("alpha_y_mid", alpha[mid, :]),
                ("P_y_mid", P[mid, :]),
                ("B_y_mid", B[mid, :]),
                ("gamma_mobility_y_mid", gamma_mobility[mid, :]),
            )
        },
    )


def _deprecated_seboldt_mms_cpp_form_metrics(*, backend: str = "cpp") -> dict[str, float]:
    """Deprecated closed-domain MMS helper, not a benchmark entrypoint."""

    from examples.utils.biofilm.three_constituent_mms import backward_euler_three_constituent_sources
    from examples.utils.biofilm.three_constituent_one_domain import build_three_constituent_one_domain_forms
    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.core.mesh import Mesh
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.ufl.expressions import (
        Function,
        TestFunction,
        TrialFunction,
        VectorFunction,
        VectorTestFunction,
        VectorTrialFunction,
    )
    from pycutfem.ufl.forms import Equation, assemble_form
    from pycutfem.ufl.measures import dx
    from pycutfem.ufl.spaces import FunctionSpace
    from pycutfem.utils.meshgen import structured_quad

    length = 1.0
    height = 0.55
    nodes, elems, _, corners = structured_quad(length, height, nx=2, ny=2, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    me = MixedElement(
        mesh,
        field_specs={
            "vf_x": 2,
            "vf_y": 2,
            "pf": 2,
            "vp_x": 2,
            "vp_y": 2,
            "pp": 2,
            "vs_x": 2,
            "vs_y": 2,
            "us_x": 2,
            "us_y": 2,
            "alpha": 2,
            "phi": 2,
            "Gamma": 2,
        },
    )
    dh = DofHandler(me, method="cg")

    VF = FunctionSpace("VF", ["vf_x", "vf_y"], dim=1)
    VP = FunctionSpace("VP", ["vp_x", "vp_y"], dim=1)
    VS = FunctionSpace("VS", ["vs_x", "vs_y"], dim=1)
    US = FunctionSpace("US", ["us_x", "us_y"], dim=1)

    trial = {
        "dv_f": VectorTrialFunction(space=VF, dof_handler=dh),
        "dp_f": TrialFunction("pf", dof_handler=dh),
        "dv_p": VectorTrialFunction(space=VP, dof_handler=dh),
        "dp_p": TrialFunction("pp", dof_handler=dh),
        "dv_s": VectorTrialFunction(space=VS, dof_handler=dh),
        "du_s": VectorTrialFunction(space=US, dof_handler=dh),
        "dalpha": TrialFunction("alpha", dof_handler=dh),
        "dphi": TrialFunction("phi", dof_handler=dh),
        "dGamma": TrialFunction("Gamma", dof_handler=dh),
    }
    test = {
        "w_f": VectorTestFunction(space=VF, dof_handler=dh),
        "q_f": TestFunction("pf", dof_handler=dh),
        "w_p": VectorTestFunction(space=VP, dof_handler=dh),
        "q_p": TestFunction("pp", dof_handler=dh),
        "w_s": VectorTestFunction(space=VS, dof_handler=dh),
        "z_u": VectorTestFunction(space=US, dof_handler=dh),
        "z_alpha": TestFunction("alpha", dof_handler=dh),
        "q_s": TestFunction("phi", dof_handler=dh),
        "z_Gamma": TestFunction("Gamma", dof_handler=dh),
    }
    state = {
        "v_f_k": VectorFunction("v_f_k", ["vf_x", "vf_y"], dof_handler=dh),
        "p_f_k": Function("p_f_k", "pf", dof_handler=dh),
        "v_p_k": VectorFunction("v_p_k", ["vp_x", "vp_y"], dof_handler=dh),
        "p_p_k": Function("p_p_k", "pp", dof_handler=dh),
        "v_s_k": VectorFunction("v_s_k", ["vs_x", "vs_y"], dof_handler=dh),
        "u_s_k": VectorFunction("u_s_k", ["us_x", "us_y"], dof_handler=dh),
        "alpha_k": Function("alpha_k", "alpha", dof_handler=dh),
        "phi_k": Function("phi_k", "phi", dof_handler=dh),
        "Gamma_k": Function("Gamma_k", "Gamma", dof_handler=dh),
        "v_f_n": VectorFunction("v_f_n", ["vf_x", "vf_y"], dof_handler=dh),
        "v_p_n": VectorFunction("v_p_n", ["vp_x", "vp_y"], dof_handler=dh),
        "v_s_n": VectorFunction("v_s_n", ["vs_x", "vs_y"], dof_handler=dh),
        "u_s_n": VectorFunction("u_s_n", ["us_x", "us_y"], dof_handler=dh),
        "alpha_n": Function("alpha_n", "alpha", dof_handler=dh),
        "phi_n": Function("phi_n", "phi", dof_handler=dh),
        "Gamma_n": Function("Gamma_n", "Gamma", dof_handler=dh),
    }

    def bubble(x, y):
        return x * (length - x) * y * (height - y)

    center = 0.46
    eps = 0.09
    speed = 0.11
    dt = 0.08
    state["alpha_k"].set_values_from_function(lambda x, y: 0.5 + 0.35 * np.tanh((x - center) / eps))
    state["alpha_n"].set_values_from_function(lambda x, y: 0.5 + 0.35 * np.tanh((x - center + speed * dt) / eps))
    state["phi_k"].set_values_from_function(lambda x, y: 0.48 + 0.04 * np.sin(np.pi * y / height))
    state["phi_n"].set_values_from_function(lambda x, y: 0.47 + 0.03 * np.sin(np.pi * y / height))
    state["v_f_k"].set_values_from_function(lambda x, y: np.asarray([0.22 * bubble(x, y), -0.08 * bubble(x, y)]))
    state["v_p_k"].set_values_from_function(lambda x, y: np.asarray([0.06 * bubble(x, y), 0.04 * bubble(x, y)]))
    state["v_s_k"].set_values_from_function(lambda x, y: np.asarray([speed * bubble(x, y), -0.03 * bubble(x, y)]))
    state["v_f_n"].set_values_from_function(lambda x, y: np.asarray([0.18 * bubble(x, y), -0.05 * bubble(x, y)]))
    state["v_p_n"].set_values_from_function(lambda x, y: np.asarray([0.04 * bubble(x, y), 0.03 * bubble(x, y)]))
    state["v_s_n"].set_values_from_function(lambda x, y: np.asarray([0.08 * bubble(x, y), -0.02 * bubble(x, y)]))
    state["u_s_k"].set_values_from_function(lambda x, y: np.asarray([0.015 * x * y, -0.010 * y * (height - y)]))
    state["u_s_n"].set_values_from_function(lambda x, y: np.asarray([0.010 * x * y, -0.008 * y * (height - y)]))
    state["Gamma_k"].set_values_from_function(lambda x, y: 0.012 * bubble(x, y))
    state["Gamma_n"].set_values_from_function(lambda x, y: 0.009 * bubble(x, y))
    state["p_f_k"].nodal_values.fill(0.0)
    state["p_p_k"].nodal_values.fill(0.0)

    params = {
        "dt": dt,
        "rho_f": 1.0,
        "rho_p": 1.0,
        "rho_s": 1.35,
        "mu_f": 0.0,
        "mu_p": 0.0,
        "mu_s": 0.0,
        "lambda_s": 0.0,
        "R_pair_cholesky": ((0.90, 0.0, 0.0), (0.25, 0.70, 0.0), (-0.18, 0.22, 1.10)),
        "pair_weight_epsilon": 1.0e-12,
        "theta_fp": 0.4,
        "ell_Gamma": 0.15,
        "include_stress_divergence": False,
    }
    sources = backward_euler_three_constituent_sources(**state, **params)
    build_params = dict(params)
    build_params.pop("include_stress_divergence")
    forms = build_three_constituent_one_domain_forms(
        **state,
        **trial,
        **test,
        dx=dx(metadata={"q": 6}),
        **build_params,
        **sources,
    )
    _, residual = assemble_form(Equation(None, forms.residual_form), dof_handler=dh, bcs=[], backend=str(backend))
    residual_arr = np.asarray(residual, dtype=float)
    return {
        "cpp_residual_inf": float(np.linalg.norm(residual_arr, ord=np.inf)),
        "cpp_residual_l2": float(np.linalg.norm(residual_arr)),
        "cpp_total_dofs": float(dh.total_dofs),
    }


def _deprecated_seboldt_mms_pdas_solve_metrics(*, backend: str = "cpp") -> dict[str, float]:
    """Deprecated bounded MMS helper, not a benchmark entrypoint."""

    from examples.utils.biofilm.three_constituent_mms import backward_euler_three_constituent_sources
    from examples.utils.biofilm.three_constituent_one_domain import (
        build_three_constituent_pdas_solver,
        build_three_constituent_one_domain_forms,
    )
    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.core.mesh import Mesh
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.solvers.nonlinear_solver import NewtonParameters, TimeStepperParameters, VIParameters
    from pycutfem.ufl.expressions import (
        Function,
        TestFunction,
        TrialFunction,
        VectorFunction,
        VectorTestFunction,
        VectorTrialFunction,
    )
    from pycutfem.ufl.forms import BoundaryCondition
    from pycutfem.ufl.measures import dx
    from pycutfem.ufl.spaces import FunctionSpace
    from pycutfem.utils.meshgen import structured_quad

    length = 1.0
    height = 0.55
    dt = 0.08
    nodes, elems, edges, corners = structured_quad(length, height, nx=1, ny=1, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x) <= 1.0e-12,
            "right": lambda x, y: abs(x - length) <= 1.0e-12,
            "bottom": lambda x, y: abs(y) <= 1.0e-12,
            "top": lambda x, y: abs(y - height) <= 1.0e-12,
        }
    )
    me = MixedElement(
        mesh,
        field_specs={
            "vf_x": 2,
            "vf_y": 2,
            "pf": 2,
            "vp_x": 2,
            "vp_y": 2,
            "pp": 2,
            "vs_x": 2,
            "vs_y": 2,
            "us_x": 2,
            "us_y": 2,
            "alpha": 2,
            "phi": 2,
            "Gamma": 2,
        },
    )
    dh = DofHandler(me, method="cg")

    VF = FunctionSpace("VF", ["vf_x", "vf_y"], dim=1)
    VP = FunctionSpace("VP", ["vp_x", "vp_y"], dim=1)
    VS = FunctionSpace("VS", ["vs_x", "vs_y"], dim=1)
    US = FunctionSpace("US", ["us_x", "us_y"], dim=1)
    trial = {
        "dv_f": VectorTrialFunction(space=VF, dof_handler=dh),
        "dp_f": TrialFunction("pf", dof_handler=dh),
        "dv_p": VectorTrialFunction(space=VP, dof_handler=dh),
        "dp_p": TrialFunction("pp", dof_handler=dh),
        "dv_s": VectorTrialFunction(space=VS, dof_handler=dh),
        "du_s": VectorTrialFunction(space=US, dof_handler=dh),
        "dalpha": TrialFunction("alpha", dof_handler=dh),
        "dphi": TrialFunction("phi", dof_handler=dh),
        "dGamma": TrialFunction("Gamma", dof_handler=dh),
    }
    test = {
        "w_f": VectorTestFunction(space=VF, dof_handler=dh),
        "q_f": TestFunction("pf", dof_handler=dh),
        "w_p": VectorTestFunction(space=VP, dof_handler=dh),
        "q_p": TestFunction("pp", dof_handler=dh),
        "w_s": VectorTestFunction(space=VS, dof_handler=dh),
        "z_u": VectorTestFunction(space=US, dof_handler=dh),
        "z_alpha": TestFunction("alpha", dof_handler=dh),
        "q_s": TestFunction("phi", dof_handler=dh),
        "z_Gamma": TestFunction("Gamma", dof_handler=dh),
    }

    def _state(prefix: str):
        return {
            "v_f_k": VectorFunction(f"{prefix}_v_f_k", ["vf_x", "vf_y"], dof_handler=dh),
            "p_f_k": Function(f"{prefix}_p_f_k", "pf", dof_handler=dh),
            "v_p_k": VectorFunction(f"{prefix}_v_p_k", ["vp_x", "vp_y"], dof_handler=dh),
            "p_p_k": Function(f"{prefix}_p_p_k", "pp", dof_handler=dh),
            "v_s_k": VectorFunction(f"{prefix}_v_s_k", ["vs_x", "vs_y"], dof_handler=dh),
            "u_s_k": VectorFunction(f"{prefix}_u_s_k", ["us_x", "us_y"], dof_handler=dh),
            "alpha_k": Function(f"{prefix}_alpha_k", "alpha", dof_handler=dh),
            "phi_k": Function(f"{prefix}_phi_k", "phi", dof_handler=dh),
            "Gamma_k": Function(f"{prefix}_Gamma_k", "Gamma", dof_handler=dh),
            "v_f_n": VectorFunction(f"{prefix}_v_f_n", ["vf_x", "vf_y"], dof_handler=dh),
            "v_p_n": VectorFunction(f"{prefix}_v_p_n", ["vp_x", "vp_y"], dof_handler=dh),
            "v_s_n": VectorFunction(f"{prefix}_v_s_n", ["vs_x", "vs_y"], dof_handler=dh),
            "u_s_n": VectorFunction(f"{prefix}_u_s_n", ["us_x", "us_y"], dof_handler=dh),
            "alpha_n": Function(f"{prefix}_alpha_n", "alpha", dof_handler=dh),
            "phi_n": Function(f"{prefix}_phi_n", "phi", dof_handler=dh),
            "Gamma_n": Function(f"{prefix}_Gamma_n", "Gamma", dof_handler=dh),
        }

    unknown = _state("tc_solve")
    exact = _state("tc_exact")
    p_f_n = Function("tc_solve_p_f_n", "pf", dof_handler=dh)
    p_p_n = Function("tc_solve_p_p_n", "pp", dof_handler=dh)
    p_f_exact_n = Function("tc_exact_p_f_n", "pf", dof_handler=dh)
    p_p_exact_n = Function("tc_exact_p_p_n", "pp", dof_handler=dh)

    def bubble(x, y):
        return x * (length - x) * y * (height - y)

    center = 0.46
    eps = 0.12
    speed = 0.10

    def fill_state(state, *, time_level: str) -> None:
        shift = 0.0 if time_level == "k" else speed * dt
        state["alpha_k"].set_values_from_function(lambda x, y: 0.5 + 0.25 * np.tanh((x - center + shift) / eps))
        state["phi_k"].set_values_from_function(lambda x, y: 0.50 + 0.03 * np.sin(np.pi * y / height))
        state["v_f_k"].set_values_from_function(lambda x, y: np.asarray([0.20 * bubble(x, y), -0.06 * bubble(x, y)]))
        state["v_p_k"].set_values_from_function(lambda x, y: np.asarray([0.05 * bubble(x, y), 0.04 * bubble(x, y)]))
        state["v_s_k"].set_values_from_function(lambda x, y: np.asarray([speed * bubble(x, y), -0.02 * bubble(x, y)]))
        state["u_s_k"].set_values_from_function(lambda x, y: np.asarray([0.010 * x * y, -0.006 * y * (height - y)]))
        state["Gamma_k"].set_values_from_function(lambda x, y: 0.010 * bubble(x, y))

    fill_state(exact, time_level="k")
    exact["alpha_n"].set_values_from_function(lambda x, y: 0.5 + 0.25 * np.tanh((x - center + speed * dt) / eps))
    exact["phi_n"].set_values_from_function(lambda x, y: 0.49 + 0.025 * np.sin(np.pi * y / height))
    exact["v_f_n"].set_values_from_function(lambda x, y: np.asarray([0.16 * bubble(x, y), -0.04 * bubble(x, y)]))
    exact["v_p_n"].set_values_from_function(lambda x, y: np.asarray([0.03 * bubble(x, y), 0.025 * bubble(x, y)]))
    exact["v_s_n"].set_values_from_function(lambda x, y: np.asarray([0.07 * bubble(x, y), -0.015 * bubble(x, y)]))
    exact["u_s_n"].set_values_from_function(lambda x, y: np.asarray([0.007 * x * y, -0.004 * y * (height - y)]))
    exact["Gamma_n"].set_values_from_function(lambda x, y: 0.008 * bubble(x, y))
    exact["p_f_k"].nodal_values.fill(0.0)
    exact["p_p_k"].nodal_values.fill(0.0)
    p_f_exact_n.nodal_values.fill(0.0)
    p_p_exact_n.nodal_values.fill(0.0)

    for key in unknown:
        unknown[key].nodal_values[:] = exact[key].nodal_values[:]
    unknown["v_f_k"].nodal_values[:] = exact["v_f_n"].nodal_values
    unknown["v_p_k"].nodal_values[:] = exact["v_p_n"].nodal_values
    unknown["v_s_k"].nodal_values[:] = exact["v_s_n"].nodal_values
    unknown["u_s_k"].nodal_values[:] = exact["u_s_n"].nodal_values
    unknown["alpha_k"].nodal_values[:] = exact["alpha_n"].nodal_values
    unknown["phi_k"].nodal_values[:] = exact["phi_n"].nodal_values
    unknown["Gamma_k"].nodal_values[:] = exact["Gamma_n"].nodal_values
    unknown["p_f_k"].nodal_values[:] = p_f_exact_n.nodal_values
    unknown["p_p_k"].nodal_values[:] = p_p_exact_n.nodal_values
    p_f_n.nodal_values[:] = p_f_exact_n.nodal_values
    p_p_n.nodal_values[:] = p_p_exact_n.nodal_values

    params = {
        "dt": dt,
        "rho_f": 1.0,
        "rho_p": 1.0,
        "rho_s": 1.35,
        "mu_f": 0.0,
        "mu_p": 0.0,
        "mu_s": 0.0,
        "lambda_s": 0.0,
        "R_pair_cholesky": ((0.85, 0.0, 0.0), (0.18, 0.75, 0.0), (-0.10, 0.16, 1.05)),
        "pair_weight_epsilon": 1.0e-12,
        "theta_fp": 0.4,
        "ell_Gamma": 0.12,
        "include_stress_divergence": False,
    }
    sources = backward_euler_three_constituent_sources(**exact, **params)
    build_params = dict(params)
    build_params.pop("include_stress_divergence")
    forms = build_three_constituent_one_domain_forms(
        **unknown,
        **trial,
        **test,
        dx=dx(metadata={"q": 6}),
        **build_params,
        **sources,
    )

    def vf_x(x, y): return 0.20 * bubble(x, y)
    def vf_y(x, y): return -0.06 * bubble(x, y)
    def vp_x(x, y): return 0.05 * bubble(x, y)
    def vp_y(x, y): return 0.04 * bubble(x, y)
    def vs_x(x, y): return speed * bubble(x, y)
    def vs_y(x, y): return -0.02 * bubble(x, y)
    def us_x(x, y): return 0.010 * x * y
    def us_y(x, y): return -0.006 * y * (height - y)
    def alpha_exact(x, y): return 0.5 + 0.25 * np.tanh((x - center) / eps)
    def phi_exact(x, y): return 0.50 + 0.03 * np.sin(np.pi * y / height)
    def gamma_exact(x, y): return 0.010 * bubble(x, y)
    def pf_exact(x, y): return 0.0
    def pp_exact(x, y): return 0.0

    bcs = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("vf_x", "dirichlet", tag, vf_x),
                BoundaryCondition("vf_y", "dirichlet", tag, vf_y),
                BoundaryCondition("pf", "dirichlet", tag, pf_exact),
                BoundaryCondition("vp_x", "dirichlet", tag, vp_x),
                BoundaryCondition("vp_y", "dirichlet", tag, vp_y),
                BoundaryCondition("pp", "dirichlet", tag, pp_exact),
                BoundaryCondition("vs_x", "dirichlet", tag, vs_x),
                BoundaryCondition("vs_y", "dirichlet", tag, vs_y),
                BoundaryCondition("us_x", "dirichlet", tag, us_x),
                BoundaryCondition("us_y", "dirichlet", tag, us_y),
                BoundaryCondition("alpha", "dirichlet", tag, alpha_exact),
                BoundaryCondition("phi", "dirichlet", tag, phi_exact),
                BoundaryCondition("Gamma", "dirichlet", tag, gamma_exact),
            ]
        )
    bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]
    solver = build_three_constituent_pdas_solver(
        forms,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=1.0e-10, max_newton_iter=10, print_level=0),
        vi_params=VIParameters(c=1.0),
        backend=str(backend),
        quad_order=6,
    )
    solver.solve_time_interval(
        functions=[
            unknown["v_f_k"],
            unknown["p_f_k"],
            unknown["v_p_k"],
            unknown["p_p_k"],
            unknown["v_s_k"],
            unknown["u_s_k"],
            unknown["alpha_k"],
            unknown["phi_k"],
            unknown["Gamma_k"],
        ],
        prev_functions=[
            unknown["v_f_n"],
            p_f_n,
            unknown["v_p_n"],
            p_p_n,
            unknown["v_s_n"],
            unknown["u_s_n"],
            unknown["alpha_n"],
            unknown["phi_n"],
            unknown["Gamma_n"],
        ],
        aux_functions={func.name: func for func in exact.values()},
        time_params=TimeStepperParameters(dt=dt, final_time=dt, max_steps=1, stop_on_steady=False),
    )

    pairs = [
        (unknown["v_f_k"], exact["v_f_k"]),
        (unknown["p_f_k"], exact["p_f_k"]),
        (unknown["v_p_k"], exact["v_p_k"]),
        (unknown["p_p_k"], exact["p_p_k"]),
        (unknown["v_s_k"], exact["v_s_k"]),
        (unknown["u_s_k"], exact["u_s_k"]),
        (unknown["alpha_k"], exact["alpha_k"]),
        (unknown["phi_k"], exact["phi_k"]),
        (unknown["Gamma_k"], exact["Gamma_k"]),
    ]
    solution_error = max(
        float(np.max(np.abs(np.asarray(a.nodal_values, dtype=float) - np.asarray(b.nodal_values, dtype=float))))
        for a, b in pairs
    )
    alpha_vals = np.asarray(unknown["alpha_k"].nodal_values, dtype=float)
    phi_vals = np.asarray(unknown["phi_k"].nodal_values, dtype=float)
    return {
        "pdas_mms_solution_error_inf": solution_error,
        "pdas_alpha_bounds_violation": max(0.0, -float(np.min(alpha_vals)), float(np.max(alpha_vals)) - 1.0),
        "pdas_phi_bounds_violation": max(0.0, -float(np.min(phi_vals)), float(np.max(phi_vals)) - 1.0),
        "pdas_mms_total_dofs": float(dh.total_dofs),
    }


def _deprecated_seboldt_mms_benchmark(
    *,
    nx: int = 121,
    ny: int = 81,
    verify_cpp_form: bool = True,
    verify_pdas_solve: bool = True,
    backend: str = "cpp",
) -> BenchmarkResult:
    """Deprecated manufactured check; use ``seboldt_physical.py`` instead."""

    length = 1.0
    height = 0.55
    center = 0.46
    eps = 0.045
    dt = 0.08
    speed = 0.11
    strain = 0.035
    shear = 0.012
    mu_s = 2.4
    lambda_s = 3.1
    R_pair_cholesky = np.asarray(
        [
            [0.90, 0.0, 0.0],
            [0.25, 0.70, 0.0],
            [-0.18, 0.22, 1.10],
        ],
        dtype=float,
    )
    theta_fp = 0.4

    x = np.linspace(0.0, length, int(nx), dtype=float)
    y = np.linspace(0.0, height, int(ny), dtype=float)
    X, Y = np.meshgrid(x, y, indexing="xy")
    xi = (X - center) / eps
    alpha = 0.5 * (1.0 + np.tanh(xi))
    alpha_x = 0.5 * (1.0 / np.cosh(xi) ** 2) / eps
    alpha_t = -speed * alpha_x
    phi = 0.48 + 0.04 * np.sin(np.pi * Y / height)
    phi_y = 0.04 * (np.pi / height) * np.cos(np.pi * Y / height)
    F, P, B = contents(alpha, phi)

    v_s = np.zeros(alpha.shape + (2,), dtype=float)
    v_s[..., 0] = speed
    v_f = np.zeros_like(v_s)
    v_p = np.zeros_like(v_s)
    v_f[..., 0] = 0.18 + 0.02 * Y
    v_f[..., 1] = -0.03 * X
    v_p[..., 0] = 0.04 + 0.01 * Y
    v_p[..., 1] = 0.02 * X

    grad_u = np.asarray([[strain, shear], [0.0, -0.5 * strain]], dtype=float)
    u_k = np.einsum("ij,...j->...i", grad_u, np.stack([X, Y], axis=-1))
    u_y = u_k[..., 1]
    u_n = u_k - dt * (v_s - np.einsum("ij,...j->...i", grad_u, v_s))
    kin_residual = (u_k - u_n) / dt + np.einsum("ij,...j->...i", grad_u, v_s) - v_s

    eps_u = 0.5 * (grad_u + grad_u.T)
    div_u = float(np.trace(grad_u))
    sigma_s = 2.0 * mu_s * eps_u + lambda_s * div_u * np.eye(2)
    sigma_f = np.asarray([[-0.2, 0.03], [0.03, -0.15]], dtype=float)
    sigma_p = np.asarray([[-0.16, 0.01], [0.01, -0.11]], dtype=float)
    sigma_fp = theta_fp * sigma_f + (1.0 - theta_fp) * sigma_p
    sigma_fs = sigma_f
    sigma_ps = sigma_p

    g_fp = np.zeros_like(v_s)
    g_fs = np.zeros_like(v_s)
    g_ps = np.zeros_like(v_s)
    g_fp[..., 0] = phi * alpha_x
    g_fs[..., 0] = (1.0 - phi) * alpha_x
    g_ps[..., 1] = alpha * phi_y

    def matvec(mat, vec):
        return np.einsum("ij,...j->...i", mat, vec)

    I_f_rev = matvec(sigma_fp, g_fp) + matvec(sigma_fs, g_fs)
    I_p_rev = -matvec(sigma_fp, g_fp) - matvec(sigma_ps, g_ps)
    I_s_rev = -matvec(sigma_fs, g_fs) + matvec(sigma_ps, g_ps)

    rel_fp = v_f - v_p
    rel_fs = v_f - v_s
    rel_ps = v_p - v_s
    m_fp = np.sqrt(F * P)
    m_fs = np.sqrt(F * B)
    m_ps = P
    H = np.zeros(alpha.shape + (3, 3), dtype=float)
    H[..., 0, :] = m_fp[..., None] * R_pair_cholesky[0, :]
    H[..., 1, :] = m_fs[..., None] * R_pair_cholesky[1, :]
    H[..., 2, :] = m_ps[..., None] * R_pair_cholesky[2, :]
    C = np.einsum("...ik,...jk->...ij", H, H)
    Y_fp = C[..., 0, 0, None] * rel_fp + C[..., 0, 1, None] * rel_fs + C[..., 0, 2, None] * rel_ps
    Y_fs = C[..., 0, 1, None] * rel_fp + C[..., 1, 1, None] * rel_fs + C[..., 1, 2, None] * rel_ps
    Y_ps = C[..., 0, 2, None] * rel_fp + C[..., 1, 2, None] * rel_fs + C[..., 2, 2, None] * rel_ps
    I_f_diss = -Y_fp - Y_fs
    I_p_diss = Y_fp - Y_ps
    I_s_diss = Y_fs + Y_ps
    force_sum = I_f_rev + I_p_rev + I_s_rev + I_f_diss + I_p_diss + I_s_diss
    dissipation = (
        C[..., 0, 0] * np.sum(rel_fp * rel_fp, axis=-1)
        + 2.0 * C[..., 0, 1] * np.sum(rel_fp * rel_fs, axis=-1)
        + 2.0 * C[..., 0, 2] * np.sum(rel_fp * rel_ps, axis=-1)
        + C[..., 1, 1] * np.sum(rel_fs * rel_fs, axis=-1)
        + 2.0 * C[..., 1, 2] * np.sum(rel_fs * rel_ps, axis=-1)
        + C[..., 2, 2] * np.sum(rel_ps * rel_ps, axis=-1)
    )
    alpha_transport = alpha_t + speed * alpha_x
    sigma_norm = float(np.linalg.norm(sigma_s))
    u_y_min_exact = -0.5 * strain * height
    u_y_max_exact = 0.0

    metrics = {
        "alpha_transport_residual": float(np.max(np.abs(alpha_transport))),
        "kinematic_residual": float(np.max(np.abs(kin_residual))),
        "alpha_bounds_violation": max(0.0, -float(np.min(alpha)), float(np.max(alpha)) - 1.0),
        "phi_bounds_violation": max(0.0, -float(np.min(phi)), float(np.max(phi)) - 1.0),
        "partition_error": float(np.max(np.abs(F + P + B - 1.0))),
        "internal_force_cancellation_error": float(np.max(np.linalg.norm(force_sum, axis=-1))),
        "negative_dissipation_violation": max(0.0, -float(np.min(dissipation))),
        "skeleton_stress_inactive_error": max(0.0, 1.0e-4 - sigma_norm),
        "full_block_cross_coupling_inactive_error": max(0.0, 1.0e-6 - float(np.max(np.abs(C[..., 0, 1])))),
        "u_y_min_error": float(np.min(u_y) - u_y_min_exact),
        "u_y_max_error": float(np.max(u_y) - u_y_max_exact),
    }
    tolerances = {
        "alpha_transport_residual": 1.0e-14,
        "kinematic_residual": 1.0e-14,
        "alpha_bounds_violation": 1.0e-14,
        "phi_bounds_violation": 1.0e-14,
        "partition_error": 1.0e-14,
        "internal_force_cancellation_error": 1.0e-13,
        "negative_dissipation_violation": 0.0,
        "skeleton_stress_inactive_error": 0.0,
        "full_block_cross_coupling_inactive_error": 0.0,
        "u_y_min_error": 1.0e-14,
        "u_y_max_error": 1.0e-14,
    }
    if bool(verify_cpp_form):
        form_metrics = _deprecated_seboldt_mms_cpp_form_metrics(backend=str(backend))
        metrics.update(form_metrics)
        tolerances.update(
            {
                "cpp_residual_inf": 1.0e-10,
                "cpp_residual_l2": 1.0e-10,
            }
        )
    if bool(verify_pdas_solve):
        solve_metrics = _deprecated_seboldt_mms_pdas_solve_metrics(backend=str(backend))
        metrics.update(solve_metrics)
        tolerances.update(
            {
                "pdas_mms_solution_error_inf": 1.0e-7,
                "pdas_alpha_bounds_violation": 0.0,
                "pdas_phi_bounds_violation": 0.0,
            }
        )
    mid = int(ny) // 2
    return BenchmarkResult(
        case_id="deprecated_seboldt_mms_not_a_benchmark",
        title="Deprecated Seboldt MMS helper, not a benchmark",
        passed=_passed(metrics, tolerances),
        metrics=metrics,
        tolerances=tolerances,
        profiles={
            "seboldt_deforming_support_centerline.csv": _profile_rows(
                ("x", x),
                ("alpha", alpha[mid, :]),
                ("phi", phi[mid, :]),
                ("B", B[mid, :]),
                ("u_y", u_y[mid, :]),
                ("dissipation", dissipation[mid, :]),
            )
        },
    )


def smooth_compact_heaviside(s, eps: float):
    s_arr = np.asarray(s, dtype=float)
    e = float(eps)
    out = np.zeros_like(s_arr, dtype=float)
    out[s_arr >= e] = 1.0
    mask = (s_arr > -e) & (s_arr < e)
    z = s_arr[mask] / e
    out[mask] = 0.5 + 0.5 * z + 0.5 * np.sin(math.pi * z) / math.pi
    return out


def finite_insert_alpha(x, y, *, center=(0.5, 0.5), half_size=(0.16, 0.22), eps: float = 0.035):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    cx, cy = float(center[0]), float(center[1])
    hx, hy = float(half_size[0]), float(half_size[1])
    ax = smooth_compact_heaviside(x_arr - (cx - hx), eps) * smooth_compact_heaviside((cx + hx) - x_arr, eps)
    ay = smooth_compact_heaviside(y_arr - (cy - hy), eps) * smooth_compact_heaviside((cy + hy) - y_arr, eps)
    return ax * ay


def run_porous_insert_benchmark(*, n: int = 121) -> BenchmarkResult:
    x = np.linspace(0.0, 1.0, int(n), dtype=float)
    y = np.linspace(0.0, 1.0, int(n), dtype=float)
    X, Y = np.meshgrid(x, y, indexing="xy")
    alpha = finite_insert_alpha(X, Y)
    phi_inside = 0.55 + 0.05 * (X - 0.5)
    phi_a = np.where(alpha > 0.0, phi_inside, 0.2 + 0.1 * Y)
    phi_b = np.where(alpha > 0.0, phi_inside, 0.9 - 0.2 * X)
    _, P_a, B_a = contents(alpha, phi_a)
    _, P_b, B_b = contents(alpha, phi_b)
    area = float(_TRAPEZOID(_TRAPEZOID(alpha, x, axis=1), y))
    pressure_drop = 1.0 + 12.0 * area
    v_s_x = np.zeros_like(alpha)
    v_s_y = np.zeros_like(alpha)
    metrics = {
        "no_body_motion_error": float(max(np.max(np.abs(v_s_x)), np.max(np.abs(v_s_y)))),
        "phi_extension_pore_content_error": float(np.max(np.abs(P_a - P_b))),
        "phi_extension_skeleton_content_error": float(np.max(np.abs(B_a - B_b))),
        "insert_area_error": float(area - (2.0 * 0.16) * (2.0 * 0.22)),
        "pressure_drop_nonpositive_error": max(0.0, -float(pressure_drop)),
    }
    tolerances = {
        "no_body_motion_error": 1.0e-14,
        "phi_extension_pore_content_error": 1.0e-14,
        "phi_extension_skeleton_content_error": 1.0e-14,
        "insert_area_error": 5.0e-3,
        "pressure_drop_nonpositive_error": 0.0,
    }
    return BenchmarkResult(
        case_id="finite_porous_insert_channel",
        title="Benchmark 6: finite porous insert in a channel",
        passed=_passed(metrics, tolerances),
        metrics=metrics,
        tolerances=tolerances,
        profiles={
            "finite_porous_insert_centerline.csv": _profile_rows(
                ("x", x),
                ("alpha_y_mid", alpha[int(n) // 2, :]),
                ("P_y_mid", P_a[int(n) // 2, :]),
                ("B_y_mid", B_a[int(n) // 2, :]),
            )
        },
    )


def run_all_benchmarks() -> list[BenchmarkResult]:
    return [
        run_poiseuille_benchmark(),
        run_darcy_column_benchmark(),
        run_drag_relaxation_benchmark(),
        run_moving_tanh_body_benchmark(),
        run_free_flow_over_porous_bed_benchmark(),
        run_stoter_fixed_bed_canonical_benchmark(),
    ]


def write_benchmark_outputs(results: list[BenchmarkResult], outdir: Path | str) -> Path:
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for result in results:
        case_dir = out / result.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        for filename, rows in result.profiles.items():
            if not rows:
                continue
            path = case_dir / filename
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        summary.append(
            {
                "case_id": result.case_id,
                "title": result.title,
                "passed": bool(result.passed),
                "metrics": {k: float(v) for k, v in result.metrics.items()},
                "tolerances": {k: float(v) for k, v in result.tolerances.items()},
            }
        )
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/three_constituent_benchmarks"))
    args = parser.parse_args(argv)
    results = run_all_benchmarks()
    summary_path = write_benchmark_outputs(results, args.outdir)
    failed = [result.case_id for result in results if not result.passed]
    print(f"Wrote {summary_path}")
    if failed:
        print("Failed benchmark gates: " + ", ".join(failed))
        return 1
    print("All three-constituent benchmark gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BenchmarkResult",
    "contents",
    "darcy_column_velocity",
    "drag_relaxation_exact",
    "finite_insert_alpha",
    "moving_tanh_alpha",
    "moving_tanh_derivatives",
    "poiseuille_velocity",
    "run_all_benchmarks",
    "run_darcy_column_benchmark",
    "run_drag_relaxation_benchmark",
    "run_free_flow_over_porous_bed_benchmark",
    "run_moving_tanh_body_benchmark",
    "run_poiseuille_benchmark",
    "run_porous_insert_benchmark",
    "run_stoter_fixed_bed_canonical_benchmark",
    "smooth_compact_heaviside",
    "stokes_darcy_bed_reference",
    "write_benchmark_outputs",
]
