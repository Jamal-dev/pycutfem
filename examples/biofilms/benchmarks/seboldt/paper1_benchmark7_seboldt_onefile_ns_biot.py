#!/usr/bin/env python3
"""Standalone benchmark-local one-domain Navier-Stokes/Biot model for Seboldt Example 2.

This file intentionally keeps the reduced one-domain NS-Biot experiment local to
one standalone benchmark driver. It does not modify the generic one-domain
builders under `examples/utils/biofilm`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting
    plt = None

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, NewtonSolver
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction, div, grad
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


@dataclass(frozen=True)
class ProfileMetrics:
    rmse: float
    linf: float
    amplitude: float
    rmse_over_amplitude: float
    linf_over_amplitude: float
    peak_amplitude: float
    peak_amplitude_ref: float
    peak_amplitude_relative_error: float
    peak_x: float
    peak_x_ref: float
    peak_x_error: float


@dataclass(frozen=True)
class CaseResult:
    kappa: float
    outdir: Path
    summary_row: dict[str, object]
    profile_x: np.ndarray
    profile_uy: np.ndarray
    fixed_metrics: ProfileMetrics | None
    moving_metrics: ProfileMetrics | None


def _parse_float_list(raw: str) -> list[float]:
    vals = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not vals:
        raise ValueError("Expected at least one floating-point value.")
    return vals


def _tag_rectangle_boundaries(mesh: Mesh, *, Lx: float, Ly: float, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - float(Lx)) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - float(Ly)) <= tol,
        }
    )


def _bottom_inlet(x: np.ndarray, y: np.ndarray, t: float, *, v_in: float) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    return 4.0 * float(v_in) * xx * (1.0 - xx)


def _cosine_ramp_value(t_now: float, ramp_time: float) -> float:
    tr = float(ramp_time)
    if (not np.isfinite(tr)) or tr <= 0.0:
        return 1.0
    tt = max(0.0, float(t_now))
    if tt >= tr:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * tt / max(1.0e-12, tr)))


def _alpha_equilibrium(y: np.ndarray, *, y_interface: float, eps_alpha: float) -> np.ndarray:
    yy = np.asarray(y, dtype=float)
    eps = max(float(eps_alpha), 1.0e-12)
    alpha = 0.5 * (1.0 + np.tanh((yy - float(y_interface)) / (math.sqrt(2.0) * eps)))
    return np.clip(alpha, 0.0, 1.0)


def _alpha_gradient_y(y: np.ndarray, *, y_interface: float, eps_alpha: float) -> np.ndarray:
    yy = np.asarray(y, dtype=float)
    eps = max(float(eps_alpha), 1.0e-12)
    arg = (yy - float(y_interface)) / (math.sqrt(2.0) * eps)
    sech2 = 1.0 / np.cosh(arg) ** 2
    return 0.5 * sech2 / (math.sqrt(2.0) * eps)


def _alpha_band_weight(y: np.ndarray, *, y_interface: float, eps_alpha: float) -> np.ndarray:
    alpha = _alpha_equilibrium(y, y_interface=float(y_interface), eps_alpha=float(eps_alpha))
    eps = max(float(eps_alpha), 1.0e-12)
    return 4.0 * alpha * (1.0 - alpha) / eps


def _alpha_exprs(*, y_interface: float, eps_alpha: float) -> tuple[Analytic, Analytic, Analytic, Analytic, Analytic]:
    alpha_fun = lambda x, y: _alpha_equilibrium(y, y_interface=float(y_interface), eps_alpha=float(eps_alpha))
    free_fun = lambda x, y: 1.0 - _alpha_equilibrium(y, y_interface=float(y_interface), eps_alpha=float(eps_alpha))
    grad_fun = lambda x, y: np.stack(
        [
            np.zeros_like(np.asarray(x, dtype=float)),
            _alpha_gradient_y(y, y_interface=float(y_interface), eps_alpha=float(eps_alpha)),
        ],
        axis=-1,
    )
    abs_grad_fun = lambda x, y: np.abs(
        _alpha_gradient_y(y, y_interface=float(y_interface), eps_alpha=float(eps_alpha))
    )
    band_fun = lambda x, y: _alpha_band_weight(y, y_interface=float(y_interface), eps_alpha=float(eps_alpha))
    return (
        Analytic(alpha_fun, degree=8),
        Analytic(free_fun, degree=8),
        Analytic(grad_fun, dim=1, degree=8),
        Analytic(abs_grad_fun, degree=8),
        Analytic(band_fun, degree=8),
    )


def _vector_component(vec_expr, idx: int):
    return vec_expr[int(idx)]


def _grad_component(vec_expr, i: int, j: int):
    return grad(_vector_component(vec_expr, i))[int(j)]


def _eps_component(vec_expr, i: int, j: int):
    if int(i) == int(j):
        return _grad_component(vec_expr, i, j)
    return Constant(0.5) * (_grad_component(vec_expr, i, j) + _grad_component(vec_expr, j, i))


def _eps_inner(a_expr, b_expr):
    acc = Constant(0.0)
    for i in range(2):
        for j in range(2):
            acc += _eps_component(a_expr, i, j) * _eps_component(b_expr, i, j)
    return acc


def _grad_inner_components(a_expr, b_expr):
    acc = Constant(0.0)
    for i in range(2):
        for j in range(2):
            acc += _grad_component(a_expr, i, j) * _grad_component(b_expr, i, j)
    return acc


def _dot_components(a_expr, b_expr):
    acc = Constant(0.0)
    for i in range(2):
        acc += _vector_component(a_expr, i) * _vector_component(b_expr, i)
    return acc


def _convective_residual(adv_expr, vel_expr, test_expr):
    acc = Constant(0.0)
    for i in range(2):
        conv_i = Constant(0.0)
        for j in range(2):
            conv_i += _vector_component(adv_expr, j) * _grad_component(vel_expr, i, j)
        acc += conv_i * _vector_component(test_expr, i)
    return acc


def _convective_jacobian_full(dvel_expr, vel_expr, test_expr):
    acc = Constant(0.0)
    for i in range(2):
        dconv_i = Constant(0.0)
        for j in range(2):
            dconv_i += _vector_component(dvel_expr, j) * _grad_component(vel_expr, i, j)
            dconv_i += _vector_component(vel_expr, j) * _grad_component(dvel_expr, i, j)
        acc += dconv_i * _vector_component(test_expr, i)
    return acc


def _convective_jacobian_lagged(dvel_expr, adv_expr, test_expr):
    acc = Constant(0.0)
    for i in range(2):
        dconv_i = Constant(0.0)
        for j in range(2):
            dconv_i += _vector_component(adv_expr, j) * _grad_component(dvel_expr, i, j)
        acc += dconv_i * _vector_component(test_expr, i)
    return acc


def _fluid_sigma_components(vel_expr, pres_expr, *, mu_f):
    s_xx = Constant(2.0) * mu_f * _eps_component(vel_expr, 0, 0) - pres_expr
    s_xy = Constant(2.0) * mu_f * _eps_component(vel_expr, 0, 1)
    s_yx = Constant(2.0) * mu_f * _eps_component(vel_expr, 1, 0)
    s_yy = Constant(2.0) * mu_f * _eps_component(vel_expr, 1, 1) - pres_expr
    return ((s_xx, s_xy), (s_yx, s_yy))


def _poro_sigma_components(disp_expr, pore_pres_expr, *, mu_s, lambda_s, alpha_biot):
    div_u = div(disp_expr)
    s_xx = Constant(2.0) * mu_s * _eps_component(disp_expr, 0, 0) + lambda_s * div_u - alpha_biot * pore_pres_expr
    s_xy = Constant(2.0) * mu_s * _eps_component(disp_expr, 0, 1)
    s_yx = Constant(2.0) * mu_s * _eps_component(disp_expr, 1, 0)
    s_yy = Constant(2.0) * mu_s * _eps_component(disp_expr, 1, 1) + lambda_s * div_u - alpha_biot * pore_pres_expr
    return ((s_xx, s_xy), (s_yx, s_yy))


def _traction_band(sig_components, test_expr, grad_alpha_expr):
    acc = Constant(0.0)
    for i in range(2):
        for j in range(2):
            acc += sig_components[i][j] * _vector_component(test_expr, i) * _vector_component(grad_alpha_expr, j)
    return acc


def _copy_state(dst_funcs, src_funcs) -> None:
    for dst, src in zip(list(dst_funcs), list(src_funcs)):
        dst.nodal_values[:] = np.asarray(src.nodal_values, dtype=float)


def _load_reference_curve(*, reference_csv: Path, kappa: float, curve_label: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not reference_csv.exists():
        return None
    rows: list[tuple[float, float]] = []
    with reference_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("curve_label", "")).strip() != str(curve_label):
                continue
            try:
                kappa_i = float(row.get("kappa", "nan"))
            except Exception:
                continue
            if (not math.isfinite(kappa_i)) or abs(kappa_i - float(kappa)) > 1.0e-12 * max(1.0, abs(float(kappa))):
                continue
            rows.append((float(row["x"]), float(row["eta_y"])))
    if not rows:
        return None
    arr = np.asarray(sorted(rows, key=lambda pair: pair[0]), dtype=float)
    return arr[:, 0], arr[:, 1]


def _compute_profile_metrics(*, x_num: np.ndarray, y_num: np.ndarray, x_ref: np.ndarray, y_ref: np.ndarray) -> ProfileMetrics:
    x_common = np.asarray(x_num, dtype=float)
    y_num_i = np.asarray(y_num, dtype=float)
    y_ref_i = np.interp(x_common, np.asarray(x_ref, dtype=float), np.asarray(y_ref, dtype=float))
    diff = y_num_i - y_ref_i
    rmse = float(np.sqrt(np.mean(diff * diff)))
    linf = float(np.max(np.abs(diff)))
    amplitude = float(max(np.max(y_ref_i) - np.min(y_ref_i), 1.0e-14))
    peak_idx = int(np.argmax(y_num_i))
    peak_ref_idx = int(np.argmax(y_ref_i))
    peak_amp = float(np.max(y_num_i))
    peak_amp_ref = float(np.max(y_ref_i))
    peak_rel = abs(peak_amp - peak_amp_ref) / max(abs(peak_amp_ref), 1.0e-14)
    peak_x = float(x_common[peak_idx])
    peak_x_ref = float(x_common[peak_ref_idx])
    return ProfileMetrics(
        rmse=rmse,
        linf=linf,
        amplitude=amplitude,
        rmse_over_amplitude=rmse / amplitude,
        linf_over_amplitude=linf / amplitude,
        peak_amplitude=peak_amp,
        peak_amplitude_ref=peak_amp_ref,
        peak_amplitude_relative_error=peak_rel,
        peak_x=peak_x,
        peak_x_ref=peak_x_ref,
        peak_x_error=abs(peak_x - peak_x_ref),
    )


def _eval_scalar_at_point(dh: DofHandler, mesh: Mesh, field_name: str, f_scalar, point: tuple[float, float]) -> float:
    from pycutfem.fem import transform

    xy = np.asarray(point, dtype=float)
    for elem in mesh.elements_list:
        verts = mesh.nodes_x_y_pos[list(elem.nodes)]
        if not (
            verts[:, 0].min() - 1.0e-12 <= xy[0] <= verts[:, 0].max() + 1.0e-12
            and verts[:, 1].min() - 1.0e-12 <= xy[1] <= verts[:, 1].max() + 1.0e-12
        ):
            continue
        try:
            xi, eta = transform.inverse_mapping(mesh, elem.id, xy)
        except Exception:
            continue
        if not (-1.0001 <= xi <= 1.0001 and -1.0001 <= eta <= 1.0001):
            continue
        me = dh.mixed_element
        phi = me.basis(field_name, float(xi), float(eta))[me.slice(field_name)]
        gdofs = np.asarray(dh.element_maps[field_name][elem.id], dtype=int)
        vals = np.asarray(f_scalar.get_nodal_values(gdofs), dtype=float)
        return float(np.asarray(phi, dtype=float) @ vals)
    raise RuntimeError(f"Failed to locate point {point} for field {field_name}.")


def _eval_vector_at_point(
    dh: DofHandler,
    mesh: Mesh,
    field_names: tuple[str, str],
    f_vec: VectorFunction,
    point: tuple[float, float],
) -> np.ndarray:
    return np.asarray(
        [
            _eval_scalar_at_point(dh, mesh, field_names[0], f_vec[0], point),
            _eval_scalar_at_point(dh, mesh, field_names[1], f_vec[1], point),
        ],
        dtype=float,
    )


def _sample_profile(
    *,
    dh: DofHandler,
    mesh: Mesh,
    disp: VectorFunction,
    Lx: float,
    y_profile: float,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, float(Lx), int(n_samples), dtype=float)
    u_y = np.asarray(
        [
            _eval_vector_at_point(dh, mesh, ("u_x", "u_y"), disp, (float(xx), float(y_profile)))[1]
            for xx in x
        ],
        dtype=float,
    )
    return x, u_y


def _write_profile_csv(path: Path, *, x: np.ndarray, u_y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "u_y"])
        for x_i, u_i in zip(np.asarray(x, dtype=float), np.asarray(u_y, dtype=float)):
            writer.writerow([f"{float(x_i):.16e}", f"{float(u_i):.16e}"])


def _write_case_plot(
    path: Path,
    *,
    kappa: float,
    x_num: np.ndarray,
    y_num: np.ndarray,
    fixed_ref: tuple[np.ndarray, np.ndarray] | None,
    moving_ref: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
    ax.plot(x_num, y_num, color="tab:green", lw=2.2, label="one-file NS-Biot")
    if fixed_ref is not None:
        ax.plot(fixed_ref[0], fixed_ref[1], color="#149dff", lw=2.0, label="Seboldt fixed linear")
    if moving_ref is not None:
        ax.plot(moving_ref[0], moving_ref[1], color="#ff7a00", lw=1.8, ls="--", label="Seboldt moving linear")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$u_y(x, y=1.25)$")
    ax.set_title(rf"$\kappa={float(kappa):.0e}I$")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _solve_case(
    *,
    kappa: float,
    outdir: Path,
    nx: int,
    ny: int,
    dt: float,
    t_final: float,
    v_in: float,
    t_ramp: float,
    Lx: float,
    Ly: float,
    y_interface: float,
    y_profile: float,
    profile_samples: int,
    rho_f: float,
    rho_s: float,
    mu_f: float,
    mu_s: float,
    lambda_s: float,
    alpha_biot: float,
    storativity_c0: float,
    eps_alpha: float,
    quad_order: int,
    backend: str,
    linear_backend: str,
    newton_tol: float,
    max_newton_iter: int,
    fluid_convection: str,
    gamma_v: float,
    gamma_v_pin: float,
    gamma_p: float,
    gamma_p_pin: float,
    gamma_q: float,
    gamma_q_pin: float,
    gamma_vS: float,
    gamma_vS_pin: float,
    gamma_u: float,
    gamma_u_pin: float,
    gamma_p_pore: float,
    gamma_p_pore_pin: float,
    p_pore_top_value: float | None,
    pressure_interface_strength: float,
    interface_mass_weight: float,
    interface_mass_penalty: float,
    interface_traction_weight: float,
    interface_entry_delta: float,
    interface_bjs_gamma: float,
    export: bool,
    reference_csv: Path,
) -> CaseResult:
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, elems, _, corners = structured_quad(
        float(Lx),
        float(Ly),
        nx=int(nx),
        ny=int(ny),
        poly_order=2,
        offset=(0.0, 0.0),
    )
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=2)
    _tag_rectangle_boundaries(mesh, Lx=float(Lx), Ly=float(Ly))

    me = MixedElement(
        mesh,
        field_specs={
            "v_x": 2,
            "v_y": 2,
            "p_f": 1,
            "u_x": 2,
            "u_y": 2,
            "p_pore": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    hx = float(Lx) / max(int(nx), 1)
    hy = float(Ly) / max(int(ny), 1)
    h = max(hx, hy)
    interface_cells = (2.0 * math.sqrt(2.0) * float(eps_alpha)) / max(hy, 1.0e-30)
    if interface_cells < 4.0:
        raise ValueError(
            "Diffuse interface is under-resolved: "
            f"2*sqrt(2)*eps_alpha/hy={interface_cells:.3f} < 4. "
            "Increase ny or eps_alpha_over_h."
        )
    tol_pin = 0.51 * h
    dh.tag_dof_by_locator(
        "pressure_pin",
        "p_f",
        lambda x, y: abs(x - 0.0) <= tol_pin and abs(y - 0.0) <= tol_pin,
    )

    V = FunctionSpace("free_velocity", ["v_x", "v_y"], dim=1)
    U = FunctionSpace("solid_disp", ["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    w = VectorTestFunction(space=V, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    z = VectorTestFunction(space=U, dof_handler=dh)
    dp = TrialFunction("p_f", dof_handler=dh)
    q = TestFunction("p_f", dof_handler=dh)
    dpp = TrialFunction("p_pore", dof_handler=dh)
    psi = TestFunction("p_pore", dof_handler=dh)

    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    xi_n = VectorFunction("xi_n", ["u_x", "u_y"], dof_handler=dh)
    xi_k = VectorFunction("xi_k", ["u_x", "u_y"], dof_handler=dh)
    p_k = Function("p_k", "p_f", dof_handler=dh)
    p_n = Function("p_n", "p_f", dof_handler=dh)
    pp_k = Function("pp_k", "p_pore", dof_handler=dh)
    pp_n = Function("pp_n", "p_pore", dof_handler=dh)

    for fn in (v_k, v_n, u_k, u_n, xi_n, xi_k, p_k, p_n, pp_k, pp_n):
        fn.nodal_values.fill(0.0)

    alpha_expr, free_expr, grad_alpha_expr, abs_grad_alpha_expr, interface_band_expr = _alpha_exprs(
        y_interface=float(y_interface),
        eps_alpha=float(eps_alpha),
    )
    dx_form = dx(metadata={"q": int(quad_order)})

    rho_f_c = Constant(float(rho_f))
    rho_s_c = Constant(float(rho_s))
    mu_f_c = Constant(float(mu_f))
    mu_s_c = Constant(float(mu_s))
    lambda_s_c = Constant(float(lambda_s))
    alpha_biot_c = Constant(float(alpha_biot))
    storativity_c0_c = Constant(float(storativity_c0))
    kappa_c = Constant(float(kappa))
    dt_c = Constant(float(dt))
    h_inv_sq_c = Constant(1.0 / max(h * h, 1.0e-30))
    gamma_v_c = Constant(float(gamma_v))
    gamma_v_pin_c = Constant(float(gamma_v_pin))
    gamma_p_c = Constant(float(gamma_p))
    gamma_p_pin_c = Constant(float(gamma_p_pin))
    gamma_vS_c = Constant(float(gamma_vS))
    gamma_vS_pin_c = Constant(float(gamma_vS_pin))
    gamma_u_c = Constant(float(gamma_u))
    gamma_u_pin_c = Constant(float(gamma_u_pin))
    gamma_pp_c = Constant(float(gamma_p_pore))
    gamma_pp_pin_c = Constant(float(gamma_p_pore_pin))
    pressure_interface_strength_c = Constant(float(pressure_interface_strength))
    interface_mass_weight_c = Constant(float(interface_mass_weight))
    interface_mass_penalty_c = Constant(float(interface_mass_penalty))
    interface_traction_weight_c = Constant(float(interface_traction_weight))
    interface_entry_delta_c = Constant(float(interface_entry_delta))
    interface_bjs_gamma_c = Constant(float(interface_bjs_gamma))

    fluid_conv_key = str(fluid_convection).strip().lower().replace("-", "_")
    if fluid_conv_key not in {"full", "lagged", "off"}:
        raise ValueError("fluid_convection must be 'full', 'lagged', or 'off'.")

    fluid_mass = (rho_f_c / dt_c) * (
        (_vector_component(v_k, 0) - _vector_component(v_n, 0)) * _vector_component(w, 0)
        + (_vector_component(v_k, 1) - _vector_component(v_n, 1)) * _vector_component(w, 1)
    )
    fluid_mass_j = (rho_f_c / dt_c) * (
        _vector_component(dv, 0) * _vector_component(w, 0)
        + _vector_component(dv, 1) * _vector_component(w, 1)
    )
    fluid_conv = Constant(0.0)
    fluid_conv_j = Constant(0.0)
    if fluid_conv_key == "full":
        fluid_conv = rho_f_c * _convective_residual(v_k, v_k, w)
        fluid_conv_j = rho_f_c * _convective_jacobian_full(dv, v_k, w)
    elif fluid_conv_key == "lagged":
        fluid_conv = rho_f_c * _convective_residual(v_n, v_k, w)
        fluid_conv_j = rho_f_c * _convective_jacobian_lagged(dv, v_n, w)

    r_fluid_bulk = free_expr * (
        fluid_mass
        + fluid_conv
        + Constant(2.0) * mu_f_c * _eps_inner(v_k, w)
        - p_k * div(w)
        + q * div(v_k)
    ) * dx_form
    j_fluid_bulk = free_expr * (
        fluid_mass_j
        + fluid_conv_j
        + Constant(2.0) * mu_f_c * _eps_inner(dv, w)
        - dp * div(w)
        + q * div(dv)
    ) * dx_form

    solid_mass = (rho_s_c / (dt_c * dt_c)) * (
        (_vector_component(u_k, 0) - _vector_component(u_n, 0) - dt_c * _vector_component(xi_n, 0))
        * _vector_component(z, 0)
        + (_vector_component(u_k, 1) - _vector_component(u_n, 1) - dt_c * _vector_component(xi_n, 1))
        * _vector_component(z, 1)
    )
    solid_mass_j = (rho_s_c / (dt_c * dt_c)) * (
        _vector_component(du, 0) * _vector_component(z, 0)
        + _vector_component(du, 1) * _vector_component(z, 1)
    )
    r_solid_bulk = alpha_expr * (
        solid_mass
        + Constant(2.0) * mu_s_c * _eps_inner(u_k, z)
        + lambda_s_c * div(u_k) * div(z)
        - alpha_biot_c * pp_k * div(z)
    ) * dx_form
    j_solid_bulk = alpha_expr * (
        solid_mass_j
        + Constant(2.0) * mu_s_c * _eps_inner(du, z)
        + lambda_s_c * div(du) * div(z)
        - alpha_biot_c * dpp * div(z)
    ) * dx_form
    r_poro_bulk = alpha_expr * (
        alpha_biot_c * div(u_k - u_n) * psi / dt_c
        + storativity_c0_c * (pp_k - pp_n) * psi / dt_c
        + kappa_c * (grad(pp_k)[0] * grad(psi)[0] + grad(pp_k)[1] * grad(psi)[1])
    ) * dx_form
    j_poro_bulk = alpha_expr * (
        alpha_biot_c * div(du) * psi / dt_c
        + storativity_c0_c * dpp * psi / dt_c
        + kappa_c * (grad(dpp)[0] * grad(psi)[0] + grad(dpp)[1] * grad(psi)[1])
    ) * dx_form

    z_x_if = _vector_component(z, 0)
    z_y_if = _vector_component(z, 1)
    xi_x_k = (_vector_component(u_k, 0) - _vector_component(u_n, 0)) / dt_c
    xi_y_k = (_vector_component(u_k, 1) - _vector_component(u_n, 1)) / dt_c
    d_xi_x = _vector_component(du, 0) / dt_c
    d_xi_y = _vector_component(du, 1) / dt_c
    rel_n_k = _vector_component(v_k, 1) - xi_y_k
    rel_t_k = _vector_component(v_k, 0) - xi_x_k
    d_rel_n = _vector_component(dv, 1) - d_xi_y
    d_rel_t = _vector_component(dv, 0) - d_xi_x
    q_n_k = -(kappa_c * grad(pp_k)[1])
    d_q_n = -(kappa_c * grad(dpp)[1])
    mass_jump_k = rel_n_k - q_n_k
    d_mass_jump = d_rel_n - d_q_n

    interface_weight_expr = abs_grad_alpha_expr
    mass_strength_c = interface_mass_weight_c * interface_mass_penalty_c
    fluid_sigma_k = _fluid_sigma_components(v_k, p_k, mu_f=mu_f_c)
    fluid_sigma_d = _fluid_sigma_components(dv, dp, mu_f=mu_f_c)
    fluid_trac_n_k = fluid_sigma_k[1][1]
    d_fluid_trac_n = fluid_sigma_d[1][1]

    # Direct diffuse interface transfer on the porous-side rows:
    # - normal mass continuity with eliminated Darcy seepage,
    # - fluid normal traction loading on the poroelastic solid,
    # - tangential continuity / BJS-style drag as an action-reaction pair.
    r_if_mass = -(mass_strength_c * interface_weight_expr * mass_jump_k * psi) * dx_form
    j_if_mass = -(mass_strength_c * interface_weight_expr * d_mass_jump * psi) * dx_form

    r_if_normal = (interface_traction_weight_c * interface_weight_expr * fluid_trac_n_k * z_y_if) * dx_form
    j_if_normal = (interface_traction_weight_c * interface_weight_expr * d_fluid_trac_n * z_y_if) * dx_form

    r_if_entry = -(interface_weight_expr * interface_bjs_gamma_c * rel_t_k * _vector_component(w, 0)) * dx_form
    j_if_entry = -(interface_weight_expr * interface_bjs_gamma_c * d_rel_t * _vector_component(w, 0)) * dx_form
    r_if_bjs = interface_weight_expr * interface_bjs_gamma_c * rel_t_k * _vector_component(z, 0) * dx_form
    j_if_bjs = interface_weight_expr * interface_bjs_gamma_c * d_rel_t * _vector_component(z, 0) * dx_form

    r_ext = (
        gamma_v_c * alpha_expr * _grad_inner_components(v_k, w)
        + gamma_v_pin_c * h_inv_sq_c * alpha_expr * alpha_expr * _dot_components(v_k, w)
        + gamma_p_c * alpha_expr * (grad(p_k)[0] * grad(q)[0] + grad(p_k)[1] * grad(q)[1])
        + gamma_p_pin_c * h_inv_sq_c * alpha_expr * alpha_expr * p_k * q
        + gamma_u_c * free_expr * _grad_inner_components(u_k, z)
        + gamma_u_pin_c * h_inv_sq_c * free_expr * free_expr * _dot_components(u_k, z)
        + gamma_pp_c * free_expr * (grad(pp_k)[0] * grad(psi)[0] + grad(pp_k)[1] * grad(psi)[1])
        + gamma_pp_pin_c * h_inv_sq_c * free_expr * free_expr * pp_k * psi
    ) * dx_form
    j_ext = (
        gamma_v_c * alpha_expr * _grad_inner_components(dv, w)
        + gamma_v_pin_c * h_inv_sq_c * alpha_expr * alpha_expr * _dot_components(dv, w)
        + gamma_p_c * alpha_expr * (grad(dp)[0] * grad(q)[0] + grad(dp)[1] * grad(q)[1])
        + gamma_p_pin_c * h_inv_sq_c * alpha_expr * alpha_expr * dp * q
        + gamma_u_c * free_expr * _grad_inner_components(du, z)
        + gamma_u_pin_c * h_inv_sq_c * free_expr * free_expr * _dot_components(du, z)
        + gamma_pp_c * free_expr * (grad(dpp)[0] * grad(psi)[0] + grad(dpp)[1] * grad(psi)[1])
        + gamma_pp_pin_c * h_inv_sq_c * free_expr * free_expr * dpp * psi
    ) * dx_form

    residual_form = (
        r_fluid_bulk
        + r_solid_bulk
        + r_poro_bulk
        + r_if_mass
        + r_if_normal
        + r_if_entry
        + r_if_bjs
        + r_ext
    )
    jacobian_form = (
        j_fluid_bulk
        + j_solid_bulk
        + j_poro_bulk
        + j_if_mass
        + j_if_normal
        + j_if_entry
        + j_if_bjs
        + j_ext
    )

    def _zero(x, y, t=0.0):
        return 0.0

    def _inflow_y(x, y, t=0.0):
        scale = _cosine_ramp_value(float(t), float(t_ramp))
        return float(scale * _bottom_inlet(np.asarray(x), np.asarray(y), float(t), v_in=float(v_in)).reshape(()))

    bcs = [
        BoundaryCondition("v_x", "dirichlet", "left", _zero),
        BoundaryCondition("v_y", "dirichlet", "left", _zero),
        BoundaryCondition("v_x", "dirichlet", "right", _zero),
        BoundaryCondition("v_y", "dirichlet", "right", _zero),
        BoundaryCondition("v_x", "dirichlet", "bottom", _zero),
        BoundaryCondition("v_y", "dirichlet", "bottom", _inflow_y),
        BoundaryCondition("u_x", "dirichlet", "left", _zero),
        BoundaryCondition("u_y", "dirichlet", "left", _zero),
        BoundaryCondition("u_x", "dirichlet", "right", _zero),
        BoundaryCondition("u_y", "dirichlet", "right", _zero),
        BoundaryCondition("p_f", "dirichlet", "pressure_pin", _zero),
    ]
    bcs_homog = [
        BoundaryCondition("v_x", "dirichlet", "left", _zero),
        BoundaryCondition("v_y", "dirichlet", "left", _zero),
        BoundaryCondition("v_x", "dirichlet", "right", _zero),
        BoundaryCondition("v_y", "dirichlet", "right", _zero),
        BoundaryCondition("v_x", "dirichlet", "bottom", _zero),
        BoundaryCondition("v_y", "dirichlet", "bottom", _zero),
        BoundaryCondition("u_x", "dirichlet", "left", _zero),
        BoundaryCondition("u_y", "dirichlet", "left", _zero),
        BoundaryCondition("u_x", "dirichlet", "right", _zero),
        BoundaryCondition("u_y", "dirichlet", "right", _zero),
        BoundaryCondition("p_f", "dirichlet", "pressure_pin", _zero),
    ]
    if p_pore_top_value is not None:
        p_pore_top_scalar = float(p_pore_top_value)

        def _p_pore_top(x, y, t=0.0, value=p_pore_top_scalar):
            return value

        bcs.append(BoundaryCondition("p_pore", "dirichlet", "top", _p_pore_top))
        bcs_homog.append(BoundaryCondition("p_pore", "dirichlet", "top", _p_pore_top))

    solver = NewtonSolver(
        residual_form=residual_form,
        jacobian_form=jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=float(newton_tol),
            max_newton_iter=int(max_newton_iter),
            line_search=True,
            ls_mode="dealii",
            ls_fail_hard=False,
        ),
        lin_params=LinearSolverParameters(backend=str(linear_backend)),
        quad_order=int(quad_order),
        backend=str(backend),
    )

    current_funcs = [v_k, p_k, u_k, pp_k]
    previous_funcs = [v_n, p_n, u_n, pp_n]
    aux_funcs: dict[str, object] = {"xi_n": xi_n}

    t_now = 0.0
    step_no = 0
    timeseries_rows: list[dict[str, float]] = []
    total_newton_its = 0
    solve_started = time.perf_counter()

    bcs_0 = NewtonSolver._freeze_bcs(bcs, 0.0)
    dh.apply_bcs(bcs_0, *current_funcs)
    dh.apply_bcs(bcs_0, *previous_funcs)

    solve_error: str | None = None
    while t_now < float(t_final) - 1.0e-14:
        step_no += 1
        dt_now = min(float(dt), float(t_final) - t_now)
        dt_c.value = float(dt_now)
        t_next = t_now + dt_now
        _copy_state(current_funcs, previous_funcs)
        bcs_now = NewtonSolver._freeze_bcs(bcs, float(t_next))
        dh.apply_bcs(bcs_now, *current_funcs)
        try:
            _delta, converged, iters = solver._newton_loop(current_funcs, previous_funcs, aux_funcs, bcs_now)
        except Exception as exc:
            solve_error = f"Newton solve raised: {exc}"
            break
        xi_k.nodal_values[:] = (np.asarray(u_k.nodal_values, dtype=float) - np.asarray(u_n.nodal_values, dtype=float)) / float(dt_now)
        total_newton_its += int(iters)
        uy_max_now = float(np.max(np.asarray(u_k[1].nodal_values, dtype=float)))
        uy_min_now = float(np.min(np.asarray(u_k[1].nodal_values, dtype=float)))
        xi_max_now = float(np.max(np.abs(np.asarray(xi_k.nodal_values, dtype=float))))
        xi_y_max_now = float(np.max(np.abs(np.asarray(xi_k[1].nodal_values, dtype=float))))
        v_free_y_max_now = float(np.max(np.abs(np.asarray(v_k[1].nodal_values, dtype=float))))
        pp_max_now = float(np.max(np.asarray(pp_k.nodal_values, dtype=float)))
        p_free_max_now = float(np.max(np.asarray(p_k.nodal_values, dtype=float)))
        p_free_min_now = float(np.min(np.asarray(p_k.nodal_values, dtype=float)))
        timeseries_rows.append(
            {
                "step": float(step_no),
                "time": float(t_next),
                "dt": float(dt_now),
                "converged": float(1.0 if bool(converged) else 0.0),
                "newton_iterations": float(iters),
                "xi_max": float(xi_max_now),
                "xi_y_max": float(xi_y_max_now),
                "v_free_y_max": float(v_free_y_max_now),
                "u_y_min": float(uy_min_now),
                "u_y_max": float(uy_max_now),
                "p_pore_max": float(pp_max_now),
                "p_free_max": float(p_free_max_now),
                "p_free_min": float(p_free_min_now),
            }
        )
        if not bool(converged):
            solve_error = f"Newton did not converge at step {step_no} (t={t_next:.6g})."
            break
        _copy_state(previous_funcs, current_funcs)
        xi_n.nodal_values[:] = np.asarray(xi_k.nodal_values, dtype=float)
        t_now = float(t_next)

    solve_seconds = time.perf_counter() - solve_started

    profile_x, profile_uy = _sample_profile(
        dh=dh,
        mesh=mesh,
        disp=u_k,
        Lx=float(Lx),
        y_profile=float(y_profile),
        n_samples=int(profile_samples),
    )
    _write_profile_csv(outdir / "profile_final.csv", x=profile_x, u_y=profile_uy)

    if timeseries_rows:
        with (outdir / "timeseries.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(timeseries_rows[0].keys()))
            writer.writeheader()
            writer.writerows(timeseries_rows)

    if export:
        export_vtk(
            str(outdir / "final_state.vtu"),
            mesh=mesh,
            dof_handler=dh,
            functions={
                "v_free": v_k,
                "p_free": p_k,
                "xi_solid": xi_k,
                "u_solid": u_k,
                "p_pore": pp_k,
            },
        )

    fixed_ref = _load_reference_curve(
        reference_csv=Path(reference_csv),
        kappa=float(kappa),
        curve_label="monolithic_fixed_linear",
    )
    fixed_metrics = None if fixed_ref is None else _compute_profile_metrics(
        x_num=profile_x,
        y_num=profile_uy,
        x_ref=fixed_ref[0],
        y_ref=fixed_ref[1],
    )
    moving_ref = _load_reference_curve(
        reference_csv=Path(reference_csv),
        kappa=float(kappa),
        curve_label="partitioned_moving_linear",
    )
    moving_metrics = None if moving_ref is None else _compute_profile_metrics(
        x_num=profile_x,
        y_num=profile_uy,
        x_ref=moving_ref[0],
        y_ref=moving_ref[1],
    )
    _write_case_plot(
        outdir / "profile_compare.png",
        kappa=float(kappa),
        x_num=profile_x,
        y_num=profile_uy,
        fixed_ref=fixed_ref,
        moving_ref=moving_ref,
    )

    summary_row: dict[str, object] = {
        "kappa": float(kappa),
        "backend": str(backend),
        "fluid_convection": str(fluid_conv_key),
        "newton_tol": float(newton_tol),
        "max_newton_iter": float(max_newton_iter),
        "nx": int(nx),
        "ny": int(ny),
        "mesh_poly_order": 2,
        "dt": float(dt),
        "t_final": float(t_final),
        "steps_completed": float(len(timeseries_rows)),
        "solve_completed": float(1.0 if solve_error is None else 0.0),
        "solve_seconds": float(solve_seconds),
        "total_newton_iterations": float(total_newton_its),
        "rho_f": float(rho_f),
        "rho_s": float(rho_s),
        "mu_f": float(mu_f),
        "mu_s": float(mu_s),
        "lambda_s": float(lambda_s),
        "alpha_biot": float(alpha_biot),
        "storativity_c0": float(storativity_c0),
        "v_in": float(v_in),
        "eps_alpha": float(eps_alpha),
        "eps_alpha_over_h": float(eps_alpha / max(h, 1.0e-30)),
        "eps_alpha_over_hy": float(eps_alpha / max(hy, 1.0e-30)),
        "interface_cells": float(interface_cells),
        "gamma_v": float(gamma_v),
        "gamma_v_pin": float(gamma_v_pin),
        "gamma_p": float(gamma_p),
        "gamma_p_pin": float(gamma_p_pin),
        "gamma_q": float(gamma_q),
        "gamma_q_pin": float(gamma_q_pin),
        "gamma_vS": float(gamma_vS),
        "gamma_vS_pin": float(gamma_vS_pin),
        "gamma_u": float(gamma_u),
        "gamma_u_pin": float(gamma_u_pin),
        "gamma_p_pore": float(gamma_p_pore),
        "gamma_p_pore_pin": float(gamma_p_pore_pin),
        "p_pore_top_value": (None if p_pore_top_value is None else float(p_pore_top_value)),
        "pressure_interface_strength": float(pressure_interface_strength),
        "interface_mass_weight": float(interface_mass_weight),
        "interface_mass_penalty": float(interface_mass_penalty),
        "interface_traction_weight": float(interface_traction_weight),
        "interface_entry_delta": float(interface_entry_delta),
        "interface_bjs_gamma": float(interface_bjs_gamma),
        "u_y_max": float(np.max(profile_uy)),
        "u_y_min": float(np.min(profile_uy)),
        "u_y_peak_x": float(profile_x[int(np.argmax(profile_uy))]),
        "xi_y_max": float(np.max(np.abs(np.asarray(xi_k[1].nodal_values, dtype=float)))),
        "p_pore_max": float(np.max(np.asarray(pp_k.nodal_values, dtype=float))),
        "p_free_max": float(np.max(np.asarray(p_k.nodal_values, dtype=float))),
        "p_free_min": float(np.min(np.asarray(p_k.nodal_values, dtype=float))),
        "profile_csv": str(outdir / "profile_final.csv"),
        "vtk_final": (str(outdir / "final_state.vtu") if bool(export) else ""),
        "solve_error": "" if solve_error is None else str(solve_error),
    }
    if fixed_metrics is not None:
        summary_row.update(
            {
                "rmse_to_fixed_linear": float(fixed_metrics.rmse),
                "linf_to_fixed_linear": float(fixed_metrics.linf),
                "rmse_over_amp_fixed_linear": float(fixed_metrics.rmse_over_amplitude),
                "linf_over_amp_fixed_linear": float(fixed_metrics.linf_over_amplitude),
                "peak_amp_relerr_fixed_linear": float(fixed_metrics.peak_amplitude_relative_error),
                "peak_x_error_fixed_linear": float(fixed_metrics.peak_x_error),
            }
        )
    if moving_metrics is not None:
        summary_row.update(
            {
                "rmse_to_moving_linear": float(moving_metrics.rmse),
                "linf_to_moving_linear": float(moving_metrics.linf),
                "rmse_over_amp_moving_linear": float(moving_metrics.rmse_over_amplitude),
                "linf_over_amp_moving_linear": float(moving_metrics.linf_over_amplitude),
                "peak_amp_relerr_moving_linear": float(moving_metrics.peak_amplitude_relative_error),
                "peak_x_error_moving_linear": float(moving_metrics.peak_x_error),
            }
        )

    summary_payload = {
        "case": summary_row,
        "fixed_linear_metrics": None if fixed_metrics is None else fixed_metrics.__dict__,
        "moving_linear_metrics": None if moving_metrics is None else moving_metrics.__dict__,
        "reference_csv": str(reference_csv) if Path(reference_csv).exists() else "",
        "profile_csv": str(outdir / "profile_final.csv"),
        "timeseries_csv": str(outdir / "timeseries.csv"),
        "vtk_final": (str(outdir / "final_state.vtu") if bool(export) else ""),
    }
    (outdir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return CaseResult(
        kappa=float(kappa),
        outdir=outdir,
        summary_row=summary_row,
        profile_x=profile_x,
        profile_uy=profile_uy,
        fixed_metrics=fixed_metrics,
        moving_metrics=moving_metrics,
    )


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=str, default=str(Path("out") / "benchmark7_onefile_ns_biot"))
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--linear-backend", type=str, default="pardiso", choices=("scipy", "pardiso", "pypardiso"))
    ap.add_argument("--newton-tol", type=float, default=1.0e-4)
    ap.add_argument("--max-newton-iter", type=int, default=8)
    ap.add_argument("--kappa-list", type=str, default="1e-3")
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=30)
    ap.add_argument("--dt", type=float, default=1.0e-3)
    ap.add_argument("--t-final", type=float, default=3.0)
    ap.add_argument("--quad-order", type=int, default=6)
    ap.add_argument("--v-in", type=float, default=5.0)
    ap.add_argument("--t-ramp", type=float, default=0.0)
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--Ly", type=float, default=1.5)
    ap.add_argument("--y-interface", type=float, default=1.0)
    ap.add_argument("--y-profile", type=float, default=1.25)
    ap.add_argument("--profile-samples", type=int, default=201)
    ap.add_argument("--rho-f", type=float, default=1.0)
    ap.add_argument("--rho-s", type=float, default=1.1)
    ap.add_argument("--mu-f", type=float, default=0.035)
    ap.add_argument("--mu-s", type=float, default=1.67785e5)
    ap.add_argument("--lambda-s", type=float, default=8.22148e6)
    ap.add_argument("--alpha-biot", type=float, default=1.0)
    ap.add_argument("--storativity-c0", type=float, default=1.0e-3)
    ap.add_argument("--eps-alpha", type=float, default=None)
    ap.add_argument("--eps-alpha-over-h", type=float, default=1.5)
    ap.add_argument("--fluid-convection", type=str, default="off", choices=("full", "lagged", "off"))
    ap.add_argument("--gamma-v", type=float, default=1.0)
    ap.add_argument("--gamma-v-pin", type=float, default=1.0)
    ap.add_argument("--gamma-p", type=float, default=1.0)
    ap.add_argument("--gamma-p-pin", type=float, default=1.0)
    ap.add_argument("--gamma-q", type=float, default=50.0)
    ap.add_argument("--gamma-q-pin", type=float, default=5000.0)
    ap.add_argument("--gamma-vS", type=float, default=50.0)
    ap.add_argument("--gamma-vS-pin", type=float, default=5000.0)
    ap.add_argument("--gamma-u", type=float, default=1.0)
    ap.add_argument("--gamma-u-pin", type=float, default=1.0)
    ap.add_argument("--gamma-p-pore", type=float, default=1.0)
    ap.add_argument("--gamma-p-pore-pin", type=float, default=1.0)
    ap.add_argument("--p-pore-top-value", type=float, default=0.0)
    ap.add_argument("--pressure-interface-strength", type=float, default=0.0)
    ap.add_argument("--interface-mass-weight", type=float, default=1.0)
    ap.add_argument("--interface-mass-penalty", type=float, default=1.0)
    ap.add_argument("--interface-traction-weight", type=float, default=1750.0)
    ap.add_argument("--interface-entry-delta", type=float, default=10.0)
    ap.add_argument("--interface-bjs-gamma", type=float, default=0.0)
    ap.add_argument("--reference-csv", type=str, default=str(here / "reference_profiles_fig6.csv"))
    ap.add_argument("--export", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir).resolve()
    kappas = _parse_float_list(args.kappa_list)
    h_char = float(args.Ly) / max(int(args.ny), 1)
    eps_alpha = (
        float(args.eps_alpha)
        if args.eps_alpha is not None and np.isfinite(float(args.eps_alpha))
        else float(args.eps_alpha_over_h) * float(h_char)
    )

    results: list[CaseResult] = []
    for kappa in kappas:
        case_id = f"kappa_{kappa:.0e}".replace("+0", "").replace("-0", "-")
        case_outdir = outdir / case_id
        print(
            f"[run] one-file NS-Biot Seboldt: kappa={float(kappa):.6e} "
            f"backend={str(args.backend)} nx={int(args.nx)} ny={int(args.ny)} "
            f"dt={float(args.dt):.3e} T={float(args.t_final):.3e} -> {case_outdir}",
            flush=True,
        )
        result = _solve_case(
            kappa=float(kappa),
            outdir=case_outdir,
            nx=int(args.nx),
            ny=int(args.ny),
            dt=float(args.dt),
            t_final=float(args.t_final),
            v_in=float(args.v_in),
            t_ramp=float(args.t_ramp),
            Lx=float(args.Lx),
            Ly=float(args.Ly),
            y_interface=float(args.y_interface),
            y_profile=float(args.y_profile),
            profile_samples=int(args.profile_samples),
            rho_f=float(args.rho_f),
            rho_s=float(args.rho_s),
            mu_f=float(args.mu_f),
            mu_s=float(args.mu_s),
            lambda_s=float(args.lambda_s),
            alpha_biot=float(args.alpha_biot),
            storativity_c0=float(args.storativity_c0),
            eps_alpha=float(eps_alpha),
            quad_order=int(args.quad_order),
            backend=str(args.backend),
            linear_backend=str(args.linear_backend),
            newton_tol=float(args.newton_tol),
            max_newton_iter=int(args.max_newton_iter),
            fluid_convection=str(args.fluid_convection),
            gamma_v=float(args.gamma_v),
            gamma_v_pin=float(args.gamma_v_pin),
            gamma_p=float(args.gamma_p),
            gamma_p_pin=float(args.gamma_p_pin),
            gamma_q=float(args.gamma_q),
            gamma_q_pin=float(args.gamma_q_pin),
            gamma_vS=float(args.gamma_vS),
            gamma_vS_pin=float(args.gamma_vS_pin),
            gamma_u=float(args.gamma_u),
            gamma_u_pin=float(args.gamma_u_pin),
            gamma_p_pore=float(args.gamma_p_pore),
            gamma_p_pore_pin=float(args.gamma_p_pore_pin),
            p_pore_top_value=(None if args.p_pore_top_value is None else float(args.p_pore_top_value)),
            pressure_interface_strength=float(args.pressure_interface_strength),
            interface_mass_weight=float(args.interface_mass_weight),
            interface_mass_penalty=float(args.interface_mass_penalty),
            interface_traction_weight=float(args.interface_traction_weight),
            interface_entry_delta=float(args.interface_entry_delta),
            interface_bjs_gamma=float(args.interface_bjs_gamma),
            export=bool(args.export),
            reference_csv=Path(args.reference_csv).resolve(),
        )
        results.append(result)

    outdir.mkdir(parents=True, exist_ok=True)
    summary_csv = outdir / "benchmark7_onefile_ns_biot_summary.csv"
    if results:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].summary_row.keys()))
            writer.writeheader()
            for result in results:
                writer.writerow(result.summary_row)
    combined = {
        "cases": [result.summary_row for result in results],
        "reference_csv": str(Path(args.reference_csv).resolve()),
    }
    (outdir / "benchmark7_onefile_ns_biot_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"[done] wrote {summary_csv}", flush=True)
    print(f"[done] wrote {outdir / 'benchmark7_onefile_ns_biot_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
