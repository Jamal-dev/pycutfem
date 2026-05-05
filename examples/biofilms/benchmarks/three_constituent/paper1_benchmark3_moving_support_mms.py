#!/usr/bin/env python3
"""Test case 3: moving-support MMS for the full three-constituent model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp

from examples.biofilms.benchmarks.three_constituent.paper1_benchmark1_mms import (
    _add_eocs,
    _add_reference_order_lines,
    _as_analytic,
    _bc_scalar,
    _div_tensor,
    _div_vec,
    _grad_scalar,
    _grad_vec,
    _lambdify_scalar,
    _lambdify_vector,
    _matvec,
    _outer,
    _sym_grad,
    _tag_unit_square_boundaries,
)
from examples.biofilms.benchmarks.three_constituent.seboldt_physical import (
    _make_homogeneous_bcs,
    _make_spaces,
    _make_state,
    _make_trial_test,
)
from examples.utils.biofilm.three_constituent_one_domain import (
    build_three_constituent_one_domain_forms,
    build_three_constituent_pdas_solver,
)
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, TimeStepperParameters, VIParameters
from pycutfem.ufl.expressions import Function
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


@dataclass(frozen=True)
class Benchmark3MovingSupportMMSResult:
    passed: bool
    outdir: Path
    rows: list[dict[str, float]]
    summary: dict[str, object]


def _json_finite(value):
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    if isinstance(value, tuple):
        return [_json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _build_moving_support_problem(
    *,
    t0: float,
    t1: float,
    gamma_delta_epsilon: float,
    radius: float,
    eps_alpha: float,
    amplitude_x: float,
    phi_mean: float,
    phi_amplitude: float,
    alpha_floor: float,
):
    x, y = sp.symbols("x y")
    pi = sp.pi
    t0_s = sp.Float(float(t0))
    t1_s = sp.Float(float(t1))
    dt_s = sp.Float(float(t1) - float(t0))
    eps_g = sp.Float(float(gamma_delta_epsilon))
    R = sp.Float(float(radius))
    eps = sp.Float(float(eps_alpha))
    A = sp.Float(float(amplitude_x))
    omega = sp.Float(2.0) * pi
    c0x = sp.Float(0.50)
    c0y = sp.Float(0.50)

    def fields(tval: float):
        t = sp.Float(float(tval))
        cx = c0x + A * sp.sin(omega * t)
        cy = c0y
        cx_dot = A * omega * sp.cos(omega * t)
        xh = x - cx
        yh = y - cy
        xi = xh
        sx = sp.sin(pi * x)
        sy = sp.sin(pi * y)
        cxm = sp.cos(pi * xi)
        sxm = sp.sin(pi * xi)
        s2xm = sp.sin(sp.Float(2.0) * pi * xi)
        c2xm = sp.cos(sp.Float(2.0) * pi * xi)
        eta = (R * R - xh * xh - yh * yh) / (sp.Float(2.0) * R * eps)
        sigma_alpha = sp.Rational(1, 2) * (sp.Float(1.0) + sp.tanh(eta))
        alpha = sp.Float(float(alpha_floor)) + (sp.Float(1.0) - sp.Float(float(alpha_floor))) * sigma_alpha
        phi = sp.Float(float(phi_mean)) + sp.Float(float(phi_amplitude)) * s2xm * sy * (
            sp.Float(1.0) + sp.Float(0.10) * sp.sin(omega * t)
        )
        v_s = sp.Matrix(
            [
                cx_dot + sp.Float(0.0060) * sxm * sy * (sp.Float(1.0) + sp.Float(0.15) * t),
                sp.Float(0.0040) * cxm * sy * (sp.Float(1.0) - sp.Float(0.10) * t),
            ]
        )
        u_s = sp.Matrix(
            [
                cx - c0x + sp.Float(0.0060) * sxm * sy * (sp.Float(1.0) + sp.Float(0.20) * t),
                sp.Float(0.0045) * cxm * sy * (sp.Float(1.0) + sp.Float(0.10) * t),
            ]
        )
        swirl = sp.Matrix([-yh * alpha, xh * alpha])
        v_f = v_s + sp.Float(0.025) * swirl + sp.Matrix(
            [
                sp.Float(0.014) * sx * sy * (sp.Float(1.0) + sp.Float(0.05) * t),
                -sp.Float(0.011) * sp.cos(pi * x) * sy * (sp.Float(1.0) - sp.Float(0.04) * t),
            ]
        )
        v_p = v_s + sp.Float(0.012) * swirl + sp.Matrix(
            [
                -sp.Float(0.006) * sxm * sy * (sp.Float(1.0) + sp.Float(0.07) * t),
                sp.Float(0.005) * cxm * sy * (sp.Float(1.0) - sp.Float(0.03) * t),
            ]
        )
        p_f = sp.Float(0.11) + sp.Float(0.018) * sp.cos(pi * x) * sy * (sp.Float(1.0) + sp.Float(0.06) * t)
        p_p = sp.Float(0.07) + sp.Float(0.014) * sx * sp.cos(pi * y) * (sp.Float(1.0) - sp.Float(0.05) * t)
        Gamma = sp.Float(0.004) + sp.Float(0.0015) * s2xm * sy * (sp.Float(1.0) + sp.Float(0.08) * t)
        return {
            "v_f": v_f,
            "p_f": p_f,
            "v_p": v_p,
            "p_p": p_p,
            "v_s": v_s,
            "u_s": u_s,
            "alpha": alpha,
            "phi": phi,
            "Gamma": Gamma,
        }

    k = fields(float(t1))
    n = fields(float(t0))

    rho_f = sp.Float(1.0)
    rho_p = sp.Float(1.0)
    rho_s = sp.Float(1.35)
    mu_f = sp.Float(0.020)
    mu_p = sp.Float(0.006)
    mu_s = sp.Float(0.70)
    lambda_s = sp.Float(1.00)
    R_fp = sp.Float(0.50)
    R_fs = sp.Float(0.30)
    R_ps = sp.Float(0.80)
    theta = sp.Float(0.45)
    ell = sp.Float(0.16)
    I2 = sp.eye(2)

    F_k = 1 - k["alpha"]
    P_k = k["alpha"] * k["phi"]
    B_k = k["alpha"] * (1 - k["phi"])
    F_n = 1 - n["alpha"]
    P_n = n["alpha"] * n["phi"]
    B_n = n["alpha"] * (1 - n["phi"])
    r_fk = F_k * rho_f
    r_pk = P_k * rho_p
    r_sk = B_k * rho_s
    r_fn = F_n * rho_f
    r_pn = P_n * rho_p
    r_sn = B_n * rho_s

    sigma_f = 2 * mu_f * _sym_grad(k["v_f"], x, y) - k["p_f"] * I2
    sigma_p = 2 * mu_p * _sym_grad(k["v_p"], x, y) - k["p_p"] * I2
    sigma_s = 2 * mu_s * _sym_grad(k["u_s"], x, y) + lambda_s * _div_vec(k["u_s"], x, y) * I2

    grad_alpha = _grad_scalar(k["alpha"], x, y)
    grad_phi = _grad_scalar(k["phi"], x, y)
    g_fp = k["phi"] * grad_alpha
    g_fs = (1 - k["phi"]) * grad_alpha
    g_ps = k["alpha"] * grad_phi
    sigma_fp = theta * sigma_f + (1 - theta) * sigma_p
    sigma_fs = sigma_f
    sigma_ps = sigma_p
    I_f_rev = _matvec(sigma_fp, g_fp) + _matvec(sigma_fs, g_fs)
    I_p_rev = -_matvec(sigma_fp, g_fp) - _matvec(sigma_ps, g_ps)
    I_s_rev = -_matvec(sigma_fs, g_fs) + _matvec(sigma_ps, g_ps)

    rel_fp = k["v_f"] - k["v_p"]
    rel_fs = k["v_f"] - k["v_s"]
    rel_ps = k["v_p"] - k["v_s"]
    chi_fp = F_k * P_k * R_fp
    chi_fs = F_k * B_k * R_fs
    chi_ps = P_k * P_k * R_ps
    I_f = I_f_rev - chi_fp * rel_fp - chi_fs * rel_fs
    I_p = I_p_rev + chi_fp * rel_fp - chi_ps * rel_ps
    I_s = I_s_rev + chi_fs * rel_fs + chi_ps * rel_ps

    u_gamma = k["v_f"]
    M_gamma_f = -k["Gamma"] * u_gamma
    M_gamma_p = k["Gamma"] * u_gamma
    M_gamma_s = sp.Matrix([0, 0])

    def mass_source(rk, rn, vk, gamma_term):
        return (rk - rn) / dt_s + _div_vec(sp.Matrix([rk * vk[0], rk * vk[1]]), x, y) + gamma_term

    def momentum_source(c, rk, rn, vk, vn, sigma, I_force, M_gamma):
        rvk = sp.Matrix([rk * vk[0], rk * vk[1]])
        rvn = sp.Matrix([rn * vn[0], rn * vn[1]])
        flux = _outer(rvk, vk)
        return (rvk - rvn) / dt_s + _div_tensor(flux, x, y) - _div_tensor(c * sigma, x, y) - I_force - M_gamma

    grad_alpha_norm = sp.sqrt(grad_alpha.dot(grad_alpha) + eps_g * eps_g) - eps_g
    L_gamma = ell * k["phi"] * grad_alpha_norm
    A_gamma = (k["p_f"] - k["p_p"]) / rho_f

    sources = {
        "S_alpha": (k["alpha"] - n["alpha"]) / dt_s + grad_alpha.dot(k["v_s"]),
        "S_mass_f": mass_source(r_fk, r_fn, k["v_f"], k["Gamma"]),
        "S_mass_p": mass_source(r_pk, r_pn, k["v_p"], -k["Gamma"]),
        "S_mass_s": mass_source(r_sk, r_sn, k["v_s"], 0),
        "S_momentum_f": momentum_source(F_k, r_fk, r_fn, k["v_f"], n["v_f"], sigma_f, I_f, M_gamma_f),
        "S_momentum_p": momentum_source(P_k, r_pk, r_pn, k["v_p"], n["v_p"], sigma_p, I_p, M_gamma_p),
        "S_momentum_s": momentum_source(B_k, r_sk, r_sn, k["v_s"], n["v_s"], sigma_s, I_s, M_gamma_s),
        "S_kinematics": (k["u_s"] - n["u_s"]) / dt_s + _grad_vec(k["u_s"], x, y) * k["v_s"] - k["v_s"],
        "S_Gamma": k["Gamma"] - L_gamma * A_gamma,
    }

    params = {
        "dt": float(float(t1) - float(t0)),
        "rho_f": 1.0,
        "rho_p": 1.0,
        "rho_s": 1.35,
        "mu_f": 0.020,
        "mu_p": 0.006,
        "mu_s": 0.70,
        "lambda_s": 1.00,
        "R_fp": 0.50,
        "R_fs": 0.30,
        "R_ps": 0.80,
        "R_pair_cholesky": ((math.sqrt(0.50), 0.0, 0.0), (0.0, math.sqrt(0.30), 0.0), (0.0, 0.0, math.sqrt(0.80))),
        "pair_weight_epsilon": 0.0,
        "theta_fp": 0.45,
        "ell_Gamma": 0.16,
        "gamma_mobility": "interface_delta",
        "gamma_delta_epsilon": float(gamma_delta_epsilon),
        "transfer_velocity": "free",
    }
    exact = {"k": k, "n": n, "sources": sources, "params": params}
    exact["callables"] = {
        "p_f": _lambdify_scalar(k["p_f"], x, y),
        "p_p": _lambdify_scalar(k["p_p"], x, y),
        "alpha": _lambdify_scalar(k["alpha"], x, y),
        "phi": _lambdify_scalar(k["phi"], x, y),
        "Gamma": _lambdify_scalar(k["Gamma"], x, y),
        "v_f": _lambdify_vector(k["v_f"], x, y),
        "v_p": _lambdify_vector(k["v_p"], x, y),
        "v_s": _lambdify_vector(k["v_s"], x, y),
        "u_s": _lambdify_vector(k["u_s"], x, y),
        "n": {
            "p_f": _lambdify_scalar(n["p_f"], x, y),
            "p_p": _lambdify_scalar(n["p_p"], x, y),
            "alpha": _lambdify_scalar(n["alpha"], x, y),
            "phi": _lambdify_scalar(n["phi"], x, y),
            "Gamma": _lambdify_scalar(n["Gamma"], x, y),
            "v_f": _lambdify_vector(n["v_f"], x, y),
            "v_p": _lambdify_vector(n["v_p"], x, y),
            "v_s": _lambdify_vector(n["v_s"], x, y),
            "u_s": _lambdify_vector(n["u_s"], x, y),
        },
    }
    exact["source_callables"] = {
        key: (_lambdify_vector(value, x, y) if isinstance(value, sp.MatrixBase) else _lambdify_scalar(value, x, y))
        for key, value in sources.items()
    }
    return exact


def _build_discrete_problem(nx: int):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=int(nx), ny=int(nx), poly_order=2)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=2)
    _tag_unit_square_boundaries(mesh)
    me = MixedElement(
        mesh,
        field_specs={
            "vf_x": 2,
            "vf_y": 2,
            "pf": 1,
            "vp_x": 2,
            "vp_y": 2,
            "pp": 1,
            "vs_x": 2,
            "vs_y": 2,
            "us_x": 2,
            "us_y": 2,
            "alpha": 1,
            "phi": 1,
            "Gamma": 1,
        },
    )
    dh = DofHandler(me, method="cg")
    spaces = _make_spaces(dh)
    trial, test = _make_trial_test(dh, spaces)
    state = _make_state(dh)
    return mesh, me, dh, trial, test, state


def _set_state_from_exact(dh: DofHandler, state: dict[str, object], exact) -> tuple[Function, Function]:
    c = exact["callables"]
    cn = c["n"]
    state["v_f_k"].set_values_from_function(c["v_f"])
    state["v_p_k"].set_values_from_function(c["v_p"])
    state["v_s_k"].set_values_from_function(c["v_s"])
    state["u_s_k"].set_values_from_function(c["u_s"])
    state["p_f_k"].set_values_from_function(c["p_f"])
    state["p_p_k"].set_values_from_function(c["p_p"])
    state["alpha_k"].set_values_from_function(c["alpha"])
    state["phi_k"].set_values_from_function(c["phi"])
    state["Gamma_k"].set_values_from_function(c["Gamma"])

    state["v_f_n"].set_values_from_function(cn["v_f"])
    state["v_p_n"].set_values_from_function(cn["v_p"])
    state["v_s_n"].set_values_from_function(cn["v_s"])
    state["u_s_n"].set_values_from_function(cn["u_s"])
    state["alpha_n"].set_values_from_function(cn["alpha"])
    state["phi_n"].set_values_from_function(cn["phi"])
    state["Gamma_n"].set_values_from_function(cn["Gamma"])
    p_f_n = Function("p_f_n", "pf", dof_handler=dh)
    p_p_n = Function("p_p_n", "pp", dof_handler=dh)
    p_f_n.set_values_from_function(cn["p_f"])
    p_p_n.set_values_from_function(cn["p_p"])
    return p_f_n, p_p_n


def _build_solver(
    *,
    me: MixedElement,
    dh: DofHandler,
    trial: dict[str, object],
    test: dict[str, object],
    state: dict[str, object],
    exact,
    qdeg: int,
    backend: str,
    linear_backend: str,
    newton_tol: float,
    max_newton_iter: int,
):
    c = exact["callables"]
    sources = {key: _as_analytic(fn) for key, fn in exact["source_callables"].items()}
    forms = build_three_constituent_one_domain_forms(
        **state,
        **trial,
        **test,
        dx=dx(metadata={"q": int(qdeg)}),
        **exact["params"],
        **sources,
    )
    bcs: list[BoundaryCondition] = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("vf_x", "dirichlet", tag, _bc_scalar(lambda x, y, f=c["v_f"]: f(x, y)[..., 0])),
                BoundaryCondition("vf_y", "dirichlet", tag, _bc_scalar(lambda x, y, f=c["v_f"]: f(x, y)[..., 1])),
                BoundaryCondition("pf", "dirichlet", tag, _bc_scalar(c["p_f"])),
                BoundaryCondition("vp_x", "dirichlet", tag, _bc_scalar(lambda x, y, f=c["v_p"]: f(x, y)[..., 0])),
                BoundaryCondition("vp_y", "dirichlet", tag, _bc_scalar(lambda x, y, f=c["v_p"]: f(x, y)[..., 1])),
                BoundaryCondition("pp", "dirichlet", tag, _bc_scalar(c["p_p"])),
                BoundaryCondition("vs_x", "dirichlet", tag, _bc_scalar(lambda x, y, f=c["v_s"]: f(x, y)[..., 0])),
                BoundaryCondition("vs_y", "dirichlet", tag, _bc_scalar(lambda x, y, f=c["v_s"]: f(x, y)[..., 1])),
                BoundaryCondition("us_x", "dirichlet", tag, _bc_scalar(lambda x, y, f=c["u_s"]: f(x, y)[..., 0])),
                BoundaryCondition("us_y", "dirichlet", tag, _bc_scalar(lambda x, y, f=c["u_s"]: f(x, y)[..., 1])),
                BoundaryCondition("alpha", "dirichlet", tag, _bc_scalar(c["alpha"])),
                BoundaryCondition("phi", "dirichlet", tag, _bc_scalar(c["phi"])),
                BoundaryCondition("Gamma", "dirichlet", tag, _bc_scalar(c["Gamma"])),
            ]
        )
    return build_three_constituent_pdas_solver(
        forms,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=_make_homogeneous_bcs(bcs),
        newton_params=NewtonParameters(
            newton_tol=float(newton_tol),
            newton_rtol=0.0,
            max_newton_iter=int(max_newton_iter),
            print_level=0,
            line_search=True,
            ls_max_iter=12,
        ),
        vi_params=VIParameters(c=1.0, project_initial_guess=True, active_set_persistence=1),
        lin_params=LinearSolverParameters(backend=str(linear_backend), tol=1.0e-11, maxit=10000),
        backend=str(backend),
        quad_order=int(qdeg),
        alpha_bounds=(0.0, 1.0),
        phi_bounds=(0.0, 1.0),
    )


def _solve_mesh(
    *,
    nx: int,
    t0: float,
    t1: float,
    qdeg: int,
    qerr: int,
    backend: str,
    linear_backend: str,
    newton_tol: float,
    max_newton_iter: int,
    exact,
    keep_state: bool = False,
):
    mesh, me, dh, trial, test, state = _build_discrete_problem(int(nx))
    p_f_n, p_p_n = _set_state_from_exact(dh, state, exact)
    solver = _build_solver(
        me=me,
        dh=dh,
        trial=trial,
        test=test,
        state=state,
        exact=exact,
        qdeg=int(qdeg),
        backend=str(backend),
        linear_backend=str(linear_backend),
        newton_tol=float(newton_tol),
        max_newton_iter=int(max_newton_iter),
    )
    t_start = time.perf_counter()
    _, n_steps, elapsed = solver.solve_time_interval(
        functions=[
            state["v_f_k"],
            state["p_f_k"],
            state["v_p_k"],
            state["p_p_k"],
            state["v_s_k"],
            state["u_s_k"],
            state["alpha_k"],
            state["phi_k"],
            state["Gamma_k"],
        ],
        prev_functions=[
            state["v_f_n"],
            p_f_n,
            state["v_p_n"],
            p_p_n,
            state["v_s_n"],
            state["u_s_n"],
            state["alpha_n"],
            state["phi_n"],
            state["Gamma_n"],
        ],
        time_params=TimeStepperParameters(dt=float(t1) - float(t0), final_time=float(t1) - float(t0), max_steps=1, stop_on_steady=False),
    )
    elapsed = float(elapsed if elapsed is not None else time.perf_counter() - t_start)
    c = exact["callables"]
    row = {
        "nx": float(nx),
        "h": 1.0 / float(nx),
        "t0": float(t0),
        "t1": float(t1),
        "n_steps": float(n_steps),
        "elapsed_s": elapsed,
        "total_dofs": float(dh.total_dofs),
        "active_dofs": float(np.asarray(getattr(solver, "active_dofs", []), dtype=int).size),
        "err_v_f": float(
            dh.l2_error(
                state["v_f_k"],
                exact={"vf_x": lambda x, y: c["v_f"](x, y)[..., 0], "vf_y": lambda x, y: c["v_f"](x, y)[..., 1]},
                fields=["vf_x", "vf_y"],
                quad_order=int(qerr),
                relative=False,
            )
        ),
        "err_p_f": float(dh.l2_error(state["p_f_k"], exact={"pf": c["p_f"]}, fields=["pf"], quad_order=int(qerr), relative=False)),
        "err_v_p": float(
            dh.l2_error(
                state["v_p_k"],
                exact={"vp_x": lambda x, y: c["v_p"](x, y)[..., 0], "vp_y": lambda x, y: c["v_p"](x, y)[..., 1]},
                fields=["vp_x", "vp_y"],
                quad_order=int(qerr),
                relative=False,
            )
        ),
        "err_p_p": float(dh.l2_error(state["p_p_k"], exact={"pp": c["p_p"]}, fields=["pp"], quad_order=int(qerr), relative=False)),
        "err_v_s": float(
            dh.l2_error(
                state["v_s_k"],
                exact={"vs_x": lambda x, y: c["v_s"](x, y)[..., 0], "vs_y": lambda x, y: c["v_s"](x, y)[..., 1]},
                fields=["vs_x", "vs_y"],
                quad_order=int(qerr),
                relative=False,
            )
        ),
        "err_u_s": float(
            dh.l2_error(
                state["u_s_k"],
                exact={"us_x": lambda x, y: c["u_s"](x, y)[..., 0], "us_y": lambda x, y: c["u_s"](x, y)[..., 1]},
                fields=["us_x", "us_y"],
                quad_order=int(qerr),
                relative=False,
            )
        ),
        "err_alpha": float(dh.l2_error(state["alpha_k"], exact={"alpha": c["alpha"]}, fields=["alpha"], quad_order=int(qerr), relative=False)),
        "err_phi": float(dh.l2_error(state["phi_k"], exact={"phi": c["phi"]}, fields=["phi"], quad_order=int(qerr), relative=False)),
        "err_Gamma": float(dh.l2_error(state["Gamma_k"], exact={"Gamma": c["Gamma"]}, fields=["Gamma"], quad_order=int(qerr), relative=False)),
        "alpha_min": float(np.min(np.asarray(state["alpha_k"].nodal_values, dtype=float))),
        "alpha_max": float(np.max(np.asarray(state["alpha_k"].nodal_values, dtype=float))),
        "phi_min": float(np.min(np.asarray(state["phi_k"].nodal_values, dtype=float))),
        "phi_max": float(np.max(np.asarray(state["phi_k"].nodal_values, dtype=float))),
    }
    if keep_state:
        return row, {"mesh": mesh, "dh": dh, "state": state, "exact": exact}
    return row


def _plot_convergence(rows: list[dict[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        ("err_v_f", r"$\mathbf{v}_f$"),
        ("err_p_f", r"$p_f$"),
        ("err_v_p", r"$\mathbf{v}_p$"),
        ("err_p_p", r"$p_p$"),
        ("err_v_s", r"$\mathbf{v}_s$"),
        ("err_u_s", r"$\mathbf{u}_s$"),
        ("err_alpha", r"$\alpha$"),
        ("err_phi", r"$\phi$"),
        ("err_Gamma", r"$\Gamma$"),
    ]
    h = np.asarray([row["h"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    for key, label in fields:
        yv = np.asarray([row[key] for row in rows], dtype=float)
        ax.loglog(h, yv, marker="o", linewidth=1.4, label=label)
    all_errors = np.asarray([[row[key] for key, _ in fields] for row in rows], dtype=float)
    _add_reference_order_lines(ax, h, all_errors.ravel())
    ax.invert_xaxis()
    ax.set_xticks(h)
    ax.set_xticklabels([f"{v:.3g}" for v in h])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="both", alpha=0.28)
    ax.set_xlabel(r"mesh size $h$")
    ax.set_ylabel(r"$L^2$ error")
    ax.set_title("Test case 3: full moving-support MMS")
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_alpha_panel(snapshots: list[dict[str, object]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pycutfem.plotting.triangulate import triangulate_field

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(snapshots), figsize=(4.2 * len(snapshots), 7.4), constrained_layout=True)
    if len(snapshots) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    grid = np.linspace(0.0, 1.0, 260, dtype=float)
    gx, gy = np.meshgrid(grid, grid)
    for col, snap in enumerate(snapshots):
        mesh = snap["mesh"]
        dh = snap["dh"]
        state = snap["state"]
        exact = snap["exact"]
        t1 = float(snap["t1"])
        tri = triangulate_field(mesh, dh, "alpha")
        alpha_num = np.clip(np.asarray(state["alpha_k"].nodal_values, dtype=float).ravel(), 0.0, 1.0)
        alpha_exact = np.asarray(exact["callables"]["alpha"](gx, gy), dtype=float)
        ax = axes[0, col]
        pc = ax.tripcolor(tri, alpha_num, shading="gouraud", vmin=0.0, vmax=1.0, cmap="viridis", rasterized=True)
        ax.contour(gx, gy, alpha_exact, levels=[0.5], colors="white", linewidths=1.2)
        ax.tricontour(tri, alpha_num, levels=[0.5], colors="black", linewidths=0.9)
        ax.set_title(f"numerical, t={t1:.3f}")
        ax.set_aspect("equal", "box")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax = axes[1, col]
        ax.pcolormesh(gx, gy, alpha_exact, shading="auto", vmin=0.0, vmax=1.0, cmap="viridis", rasterized=True)
        ax.contour(gx, gy, alpha_exact, levels=[0.5], colors="black", linewidths=1.0)
        ax.set_title(f"exact, t={t1:.3f}")
        ax.set_aspect("equal", "box")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(pc, ax=axes.ravel().tolist(), shrink=0.88, label=r"$\alpha$")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_latex_table(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    final = rows[-1]
    lines = [
        "% Generated by paper1_benchmark3_moving_support_mms.py",
        r"{\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}rrrrrrrr@{}}",
        r"\toprule",
        r"$n_x$ & $\|\mathbf v_p-\mathbf v_{p,h}\|$ & EOC$_{\mathbf v_p}$ & $\|p_f-p_{f,h}\|$ & EOC$_{p_f}$ & $\|p_p-p_{p,h}\|$ & EOC$_{p_p}$ & $\|\alpha-\alpha_h\|$ \\",
        r"\midrule",
    ]
    for row in rows:
        eoc_vp = "--" if not np.isfinite(row["eoc_v_p"]) else f"{row['eoc_v_p']:.2f}"
        eoc_pf = "--" if not np.isfinite(row["eoc_p_f"]) else f"{row['eoc_p_f']:.2f}"
        eoc_pp = "--" if not np.isfinite(row["eoc_p_p"]) else f"{row['eoc_p_p']:.2f}"
        lines.append(
            f"{int(row['nx'])} & {row['err_v_p']:.3e} & {eoc_vp} & {row['err_p_f']:.3e} & {eoc_pf} & {row['err_p_p']:.3e} & {eoc_pp} & {row['err_alpha']:.3e} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            f"% Finest-grid velocity EOC range: {min(final[k] for k in final if k.startswith('eoc_v_')):.2f}--{max(final[k] for k in final if k.startswith('eoc_v_')):.2f}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark3_moving_support_mms(
    *,
    outdir: Path,
    nx_list: tuple[int, ...] = (8, 12, 16, 24),
    t0: float = 0.0,
    t1: float = 0.02,
    qdeg: int = 8,
    qerr: int = 10,
    backend: str = "cpp",
    linear_backend: str = "scipy",
    newton_tol: float = 1.0e-8,
    max_newton_iter: int = 18,
    gamma_delta_epsilon: float = 1.0e-10,
    radius: float = 0.18,
    eps_alpha: float = 0.25,
    amplitude_x: float = 0.05,
    phi_mean: float = 0.55,
    phi_amplitude: float = 0.04,
    alpha_floor: float = 0.30,
    make_figures: bool = True,
) -> Benchmark3MovingSupportMMSResult:
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    exact = _build_moving_support_problem(
        t0=float(t0),
        t1=float(t1),
        gamma_delta_epsilon=float(gamma_delta_epsilon),
        radius=float(radius),
        eps_alpha=float(eps_alpha),
        amplitude_x=float(amplitude_x),
        phi_mean=float(phi_mean),
        phi_amplitude=float(phi_amplitude),
        alpha_floor=float(alpha_floor),
    )
    rows = [
        _solve_mesh(
            nx=int(nx),
            t0=float(t0),
            t1=float(t1),
            qdeg=int(qdeg),
            qerr=int(qerr),
            backend=str(backend),
            linear_backend=str(linear_backend),
            newton_tol=float(newton_tol),
            max_newton_iter=int(max_newton_iter),
            exact=exact,
        )
        for nx in nx_list
    ]
    _add_eocs(rows)
    bounded = all(
        row["alpha_min"] >= -1.0e-10
        and row["alpha_max"] <= 1.0 + 1.0e-10
        and row["phi_min"] >= -1.0e-10
        and row["phi_max"] <= 1.0 + 1.0e-10
        for row in rows
    )
    improved = all(rows[-1][key] <= 1.05 * rows[0][key] for key in rows[0] if key.startswith("err_"))
    passed = bool(bounded and improved)
    final_eocs = {key: float(rows[-1][key]) for key in rows[-1] if key.startswith("eoc_")}
    summary = {
        "case_id": "benchmark3_moving_support_full_mms",
        "passed": passed,
        "nx_list": [int(v) for v in nx_list],
        "t0": float(t0),
        "t1": float(t1),
        "backend": str(backend),
        "linear_backend": str(linear_backend),
        "radius": float(radius),
        "eps_alpha": float(eps_alpha),
        "amplitude_x": float(amplitude_x),
        "phi_range": [float(phi_mean) - abs(float(phi_amplitude)), float(phi_mean) + abs(float(phi_amplitude))],
        "alpha_floor": float(alpha_floor),
        "final_eocs": final_eocs,
        "rows": rows,
    }
    csv_path = outdir / "benchmark3_moving_support_mms_convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = _json_finite(summary)
    (outdir / "summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_latex_table(rows, outdir / "benchmark3_moving_support_mms_table.tex")
    if make_figures:
        _plot_convergence(rows, outdir / "benchmark3_moving_support_mms_convergence.png")
        t1_f = float(t1)
        snap_times = [max(0.5 * t1_f, min(t1_f, 0.01)), t1_f]
        snapshots = []
        for ts in snap_times:
            snap_exact = _build_moving_support_problem(
                t0=float(t0),
                t1=float(ts),
                gamma_delta_epsilon=float(gamma_delta_epsilon),
                radius=float(radius),
                eps_alpha=float(eps_alpha),
                amplitude_x=float(amplitude_x),
                phi_mean=float(phi_mean),
                phi_amplitude=float(phi_amplitude),
                alpha_floor=float(alpha_floor),
            )
            _, snap = _solve_mesh(
                nx=int(nx_list[-1]),
                t0=float(t0),
                t1=float(ts),
                qdeg=int(qdeg),
                qerr=int(qerr),
                backend=str(backend),
                linear_backend=str(linear_backend),
                newton_tol=float(newton_tol),
                max_newton_iter=int(max_newton_iter),
                exact=snap_exact,
                keep_state=True,
            )
            snap["t1"] = float(ts)
            snapshots.append(snap)
        _plot_alpha_panel(snapshots, outdir / "benchmark3_moving_support_mms_alpha_panel.png")
    return Benchmark3MovingSupportMMSResult(passed=passed, outdir=outdir, rows=rows, summary=summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("out/three_constituent_benchmark3_moving_support_mms"))
    parser.add_argument("--nx-list", type=str, default="8,12,16,24")
    parser.add_argument("--t1", type=float, default=0.02)
    parser.add_argument("--qdeg", type=int, default=8)
    parser.add_argument("--qerr", type=int, default=10)
    parser.add_argument("--backend", type=str, default="cpp")
    parser.add_argument("--linear-backend", type=str, default="scipy")
    parser.add_argument("--newton-tol", type=float, default=1.0e-8)
    parser.add_argument("--max-newton-iter", type=int, default=18)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    nx_list = tuple(int(v.strip()) for v in str(args.nx_list).split(",") if v.strip())
    result = run_benchmark3_moving_support_mms(
        outdir=args.outdir,
        nx_list=nx_list,
        t1=float(args.t1),
        qdeg=int(args.qdeg),
        qerr=int(args.qerr),
        backend=str(args.backend),
        linear_backend=str(args.linear_backend),
        newton_tol=float(args.newton_tol),
        max_newton_iter=int(args.max_newton_iter),
        make_figures=not bool(args.no_figures),
    )
    print(json.dumps(result.summary, allow_nan=False, indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Benchmark3MovingSupportMMSResult", "run_benchmark3_moving_support_mms"]
