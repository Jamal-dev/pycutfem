#!/usr/bin/env python3
"""Benchmark 1: MMS convergence for the physical three-constituent model."""

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
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import Function
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


@dataclass(frozen=True)
class Benchmark1MMSResult:
    passed: bool
    outdir: Path
    rows: list[dict[str, float]]
    summary: dict[str, object]


MMS_ERROR_FIELDS = [
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


def _tag_unit_square_boundaries(mesh: Mesh, *, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - 1.0) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - 1.0) <= tol,
        }
    )


def _lambdify_scalar(expr, x, y):
    fn = sp.lambdify((x, y), expr, "numpy")
    return lambda xv, yv: np.asarray(fn(xv, yv), dtype=float)


def _lambdify_vector(exprs, x, y):
    fns = [sp.lambdify((x, y), expr, "numpy") for expr in exprs]

    def _eval(xv, yv):
        vals = [np.asarray(fn(xv, yv), dtype=float) for fn in fns]
        return np.stack(vals, axis=-1)

    return _eval


def _div_vec(vec, x, y):
    return sp.diff(vec[0], x) + sp.diff(vec[1], y)


def _grad_scalar(expr, x, y):
    return sp.Matrix([sp.diff(expr, x), sp.diff(expr, y)])


def _grad_vec(vec, x, y):
    return sp.Matrix([[sp.diff(vec[i], var) for var in (x, y)] for i in range(2)])


def _sym_grad(vec, x, y):
    G = _grad_vec(vec, x, y)
    return sp.Rational(1, 2) * (G + G.T)


def _div_tensor(T, x, y):
    return sp.Matrix([sp.diff(T[i, 0], x) + sp.diff(T[i, 1], y) for i in range(2)])


def _outer(a, b):
    return sp.Matrix([[a[i] * b[j] for j in range(2)] for i in range(2)])


def _matvec(A, v):
    return sp.Matrix([A[i, 0] * v[0] + A[i, 1] * v[1] for i in range(2)])


def _build_manufactured_problem(*, dt: float, gamma_delta_epsilon: float):
    x, y = sp.symbols("x y")
    pi = sp.pi
    dt_s = sp.Float(float(dt))
    eps_g = sp.Float(float(gamma_delta_epsilon))

    def fields(tval: float):
        t = sp.Float(float(tval))
        sx = sp.sin(pi * x)
        sy = sp.sin(pi * y)
        cx = sp.cos(pi * x)
        cy = sp.cos(pi * y)
        alpha = sp.Float(0.35) + sp.Float(0.08) * x + sp.Float(0.07) * y + sp.Float(0.010) * sx * sy * (1 + sp.Float(0.20) * t)
        phi = sp.Float(0.56) + sp.Float(0.030) * cx * sy * (1 + sp.Float(0.10) * t)
        vf = sp.Matrix(
            [
                sp.Float(0.030) * sx * sy + sp.Float(0.010) * x * (1 - y) * (1 + t),
                -sp.Float(0.025) * cx * sy + sp.Float(0.006) * y * (1 - x) * (1 + sp.Float(0.5) * t),
            ]
        )
        vp = sp.Matrix(
            [
                -sp.Float(0.018) * sx * cy + sp.Float(0.005) * x * y * (1 + sp.Float(0.30) * t),
                sp.Float(0.020) * sx * sy + sp.Float(0.004) * (1 - x) * y * (1 + sp.Float(0.20) * t),
            ]
        )
        vs = sp.Matrix(
            [
                sp.Float(0.012) * sx * sy + sp.Float(0.003) * x * (1 - x) * (1 + sp.Float(0.40) * t),
                -sp.Float(0.010) * sx * cy + sp.Float(0.002) * y * (1 - y) * (1 + sp.Float(0.25) * t),
            ]
        )
        us = sp.Matrix(
            [
                sp.Float(0.010) * sx * sy * (1 + sp.Float(0.50) * t) + sp.Float(0.003) * x * y,
                -sp.Float(0.008) * cx * sy * (1 + sp.Float(0.30) * t) + sp.Float(0.002) * y * (1 - x),
            ]
        )
        pf = sp.Float(0.12) + sp.Float(0.025) * cx * sy * (1 + sp.Float(0.15) * t)
        pp = sp.Float(0.08) + sp.Float(0.020) * sx * cy * (1 + sp.Float(0.10) * t)
        Gamma = sp.Float(0.010) + sp.Float(0.003) * sx * sy * (1 + sp.Float(0.05) * t)
        return {
            "v_f": vf,
            "p_f": pf,
            "v_p": vp,
            "p_p": pp,
            "v_s": vs,
            "u_s": us,
            "alpha": alpha,
            "phi": phi,
            "Gamma": Gamma,
        }

    k = fields(float(dt))
    n = fields(0.0)

    rho_f = sp.Float(1.0)
    rho_p = sp.Float(1.0)
    rho_s = sp.Float(1.35)
    mu_f = sp.Float(0.020)
    mu_p = sp.Float(0.006)
    mu_s = sp.Float(0.80)
    lambda_s = sp.Float(1.10)
    R_fp = sp.Float(0.55)
    R_fs = sp.Float(0.35)
    R_ps = sp.Float(0.90)
    theta = sp.Float(0.40)
    ell = sp.Float(0.18)
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
    exact = {
        "k": k,
        "n": n,
        "sources": sources,
        "params": {
            "dt": float(dt),
            "rho_f": 1.0,
            "rho_p": 1.0,
            "rho_s": 1.35,
            "mu_f": 0.020,
            "mu_p": 0.006,
            "mu_s": 0.80,
            "lambda_s": 1.10,
            "R_fp": 0.55,
            "R_fs": 0.35,
            "R_ps": 0.90,
            "R_pair_cholesky": ((math.sqrt(0.55), 0.0, 0.0), (0.0, math.sqrt(0.35), 0.0), (0.0, 0.0, math.sqrt(0.90))),
            "pair_weight_epsilon": 0.0,
            "theta_fp": 0.40,
            "ell_Gamma": 0.18,
            "gamma_mobility": "interface_delta",
            "gamma_delta_epsilon": float(gamma_delta_epsilon),
            "transfer_velocity": "free",
        },
    }

    scalar_exact = {
        "p_f": _lambdify_scalar(k["p_f"], x, y),
        "p_p": _lambdify_scalar(k["p_p"], x, y),
        "alpha": _lambdify_scalar(k["alpha"], x, y),
        "phi": _lambdify_scalar(k["phi"], x, y),
        "Gamma": _lambdify_scalar(k["Gamma"], x, y),
    }
    vector_exact = {
        "v_f": _lambdify_vector(k["v_f"], x, y),
        "v_p": _lambdify_vector(k["v_p"], x, y),
        "v_s": _lambdify_vector(k["v_s"], x, y),
        "u_s": _lambdify_vector(k["u_s"], x, y),
    }
    exact["callables"] = {
        **scalar_exact,
        **vector_exact,
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


def _as_analytic(fn, *, degree: int = 8) -> Analytic:
    return Analytic(lambda x, y: fn(x, y), degree=int(degree))


def _bc_scalar(fn):
    return lambda x, y, t=0.0: float(np.asarray(fn(x, y), dtype=float))


def _solve_mesh(
    *,
    nx: int,
    dt: float,
    qdeg: int,
    qerr: int,
    backend: str,
    linear_backend: str,
    newton_tol: float,
    max_newton_iter: int,
    exact,
) -> dict[str, float]:
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
    solver = build_three_constituent_pdas_solver(
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
    t0 = time.perf_counter()
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
        time_params=TimeStepperParameters(dt=float(dt), final_time=float(dt), max_steps=1, stop_on_steady=False),
    )
    elapsed = float(elapsed if elapsed is not None else time.perf_counter() - t0)

    row = {
        "nx": float(nx),
        "h": 1.0 / float(nx),
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
    return row


def _add_eocs(rows: list[dict[str, float]]) -> None:
    error_keys = [key for key in rows[0] if key.startswith("err_")]
    for row in rows:
        for key in error_keys:
            row[f"eoc_{key[4:]}"] = float("nan")
    for prev, row in zip(rows[:-1], rows[1:]):
        for key in error_keys:
            if prev[key] > 0.0 and row[key] > 0.0:
                row[f"eoc_{key[4:]}"] = math.log(prev[key] / row[key]) / math.log(prev["h"] / row["h"])


def _add_reference_order_lines(ax, h: np.ndarray, errors: np.ndarray) -> None:
    finite = np.asarray(errors, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0 or h.size < 2:
        return
    xline = np.asarray([float(np.max(h)), float(np.min(h))], dtype=float)
    h_anchor = float(np.min(h))
    y_min = float(np.min(finite))
    styles = [
        (1, y_min * 8.0, (0, (3, 2)), r"$O(h)$"),
        (2, y_min * 2.2, (0, (1, 1)), r"$O(h^2)$"),
    ]
    for order, y_anchor, linestyle, label in styles:
        yline = y_anchor * (xline / h_anchor) ** order
        ax.loglog(xline, yline, color="0.25", linewidth=1.2, linestyle=linestyle, label=label)


def _plot_convergence(
    rows: list[dict[str, float]],
    path: Path,
    *,
    title: str,
    fields: list[tuple[str, str]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = MMS_ERROR_FIELDS if fields is None else fields
    h = np.asarray([row["h"] for row in rows], dtype=float)
    all_errors = np.asarray([[row[key] for key, _ in fields] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    for key, label in fields:
        yv = np.asarray([row[key] for row in rows], dtype=float)
        ax.loglog(h, yv, marker="o", linewidth=1.4, label=label)
    _add_reference_order_lines(ax, h, all_errors.ravel())
    ax.invert_xaxis()
    ax.set_xticks(h)
    ax.set_xticklabels([f"{v:.3g}" for v in h])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="both", alpha=0.28)
    ax.set_xlabel(r"mesh size $h$")
    ax.set_ylabel(r"$L^2$ error")
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_latex_table(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [r"Field"]
    for i, row in enumerate(rows):
        nx = int(row["nx"])
        if i > 0:
            header.append(rf"EOC$_{{{int(rows[i - 1]['nx'])}\to {nx}}}$")
        header.append(rf"$e_{{{nx}}}$")
    align = "@{}l" + "r" * (len(header) - 1) + "@{}"
    lines = [
        "% Generated by paper1_benchmark1_mms.py",
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for key, label in MMS_ERROR_FIELDS:
        parts = [label]
        eoc_key = f"eoc_{key[4:]}"
        for i, row in enumerate(rows):
            if i > 0:
                parts.append("--" if not np.isfinite(row[eoc_key]) else f"{row[eoc_key]:.2f}")
            parts.append(f"{row[key]:.3e}")
        lines.append(" & ".join(parts) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark1_mms(
    *,
    outdir: Path,
    nx_list: tuple[int, ...] = (4, 8, 16, 24),
    dt: float = 0.02,
    qdeg: int = 8,
    qerr: int = 10,
    backend: str = "cpp",
    linear_backend: str = "scipy",
    newton_tol: float = 1.0e-8,
    max_newton_iter: int = 16,
    gamma_delta_epsilon: float = 1.0e-10,
    make_figures: bool = True,
) -> Benchmark1MMSResult:
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    exact = _build_manufactured_problem(dt=float(dt), gamma_delta_epsilon=float(gamma_delta_epsilon))
    rows = [
        _solve_mesh(
            nx=int(nx),
            dt=float(dt),
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
    field_keys = [key for key in rows[-1] if key.startswith("eoc_")]
    final_eocs = {key: float(rows[-1][key]) for key in field_keys}
    bounded = all(row["alpha_min"] >= -1.0e-10 and row["alpha_max"] <= 1.0 + 1.0e-10 and row["phi_min"] >= -1.0e-10 and row["phi_max"] <= 1.0 + 1.0e-10 for row in rows)
    improved = all(rows[-1][key] <= 1.05 * rows[0][key] for key in rows[0] if key.startswith("err_"))
    passed = bool(bounded and improved)
    summary = _json_finite(
        {
        "case_id": "benchmark1_three_constituent_mms",
        "passed": passed,
        "nx_list": [int(v) for v in nx_list],
        "dt": float(dt),
        "backend": str(backend),
        "linear_backend": str(linear_backend),
        "gamma_mobility": "interface_delta",
        "resistance_model": "full_cholesky_diagonal_equivalent",
        "final_eocs": final_eocs,
        "rows": rows,
        }
    )
    csv_path = outdir / "benchmark1_mms_convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (outdir / "summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_latex_table(rows, outdir / "benchmark1_mms_table.tex")
    if make_figures:
        _plot_convergence(rows, outdir / "benchmark1_mms_convergence.png", title="Test case 1: nine-field MMS")
    return Benchmark1MMSResult(passed=passed, outdir=outdir, rows=rows, summary=summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("out/three_constituent_benchmark1_mms"))
    parser.add_argument("--nx-list", type=str, default="4,8,16,24")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--qdeg", type=int, default=8)
    parser.add_argument("--qerr", type=int, default=10)
    parser.add_argument("--backend", type=str, default="cpp")
    parser.add_argument("--linear-backend", type=str, default="scipy")
    parser.add_argument("--newton-tol", type=float, default=1.0e-8)
    parser.add_argument("--max-newton-iter", type=int, default=16)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    nx_list = tuple(int(v.strip()) for v in str(args.nx_list).split(",") if v.strip())
    result = run_benchmark1_mms(
        outdir=args.outdir,
        nx_list=nx_list,
        dt=float(args.dt),
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


__all__ = ["Benchmark1MMSResult", "run_benchmark1_mms"]
