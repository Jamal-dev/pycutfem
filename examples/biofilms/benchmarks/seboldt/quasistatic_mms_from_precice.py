#!/usr/bin/env python3
"""Quasi-static Seboldt MMS built from the exported preCICE polynomial fit.

This module lifts the detached preCICE fit into the one-domain
quasi-static final-form branch documented in
`current_monolithic_formulation.md`.

The manufactured fields are reusable across the branch variants that share the
same exact state family, including both:

- the older note-branch hook
  `disable_pore_momentum=False, disable_solid_momentum=True,
  combined_porous_momentum=True`
- the current post-accept branch
  `disable_pore_momentum=True, disable_solid_momentum=True,
  combined_porous_momentum=True`

The exact fields remain the simulation-derived blended fields.

The exact fields remain the simulation-derived blended fields. The porous
porosity is reconstructed locally on the active time slab by fitting the
quasi-static transport law

  d_t(phi) + div(phi vS) - kappa_inv^{-1} Δp_p = 0

with a reference-map initial porous state at t_n, and then extended smoothly to
the whole one-domain box so the manufactured loads match the actual residual
rows on this branch, including the extension equations.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path

import numpy as np


DEFAULT_FIT_MODULE_PATH = (
    Path(__file__).resolve().with_name("mms_solution_precice_partial_t1787_deg4.py")
)

_LOCAL_PHI_RECON_CACHE: dict[tuple[object, ...], dict[str, object]] = {}


@dataclass(frozen=True)
class SeboldtQuasiStaticMMSStep:
    fit_module_path: str
    t_origin: float
    t_n: float
    t_k: float
    dt: float
    theta: float
    y_interface: float
    eps_alpha: float
    interface_beta: float
    support_indicator_beta: float
    field_scale: float
    rho_f: float
    rho_s: float
    mu_f: float
    kappa_inv: float
    mu_s: float
    lambda_s: float
    phi_b: float
    fluid_convection: str

    v: callable
    p: callable
    vP: callable
    p_pore: callable
    vS: callable
    u: callable
    alpha: callable
    phi: callable

    v_n: callable
    p_n: callable
    vP_n: callable
    p_pore_n: callable
    vS_n: callable
    u_n: callable
    alpha_n: callable
    phi_n: callable

    v_k: callable
    p_k: callable
    vP_k: callable
    p_pore_k: callable
    vS_k: callable
    u_k: callable
    alpha_k: callable
    phi_k: callable

    f_fluid_momentum: callable
    s_free_mass: callable
    f_pore_momentum: callable
    s_porous_mass: callable
    f_porous_body: callable
    f_kinematics: callable
    f_alpha: callable
    f_phi: callable


def _load_fit_module(path: str | Path):
    fit_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("seboldt_precice_fit_module", fit_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load fit module from {fit_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _alpha_from_eps(y, *, y_interface: float, eps_alpha: float):
    yy = np.asarray(y, dtype=float)
    eps = max(float(eps_alpha), 1.0e-12)
    c = 1.0 / (math.sqrt(2.0) * eps)
    z = c * (yy - float(y_interface))
    th = np.tanh(z)
    sech2 = 1.0 - th * th
    alpha = 0.5 * (1.0 + th)
    alpha_y = 0.5 * c * sech2
    alpha_yy = -(c * c) * th * sech2
    alpha_yyy = -(c * c * c) * sech2 * (1.0 - 3.0 * th * th)
    return alpha, alpha_y, alpha_yy, alpha_yyy


def _alpha_from_beta(y, *, y_interface: float, beta: float):
    yy = np.asarray(y, dtype=float)
    z = float(beta) * (yy - float(y_interface))
    th = np.tanh(z)
    sech2 = 1.0 - th * th
    alpha = 0.5 * (1.0 + th)
    alpha_y = 0.5 * float(beta) * sech2
    alpha_yy = -(float(beta) ** 2) * th * sech2
    alpha_yyy = -(float(beta) ** 3) * sech2 * (1.0 - 3.0 * th * th)
    return alpha, alpha_y, alpha_yy, alpha_yyy


def _support_indicator_raw(alpha, alpha_y, alpha_yy, *, beta: float):
    aa = np.asarray(alpha, dtype=float)
    z = float(beta) * (aa - 0.5)
    th = np.tanh(z)
    sech2 = 1.0 - th * th
    weight = 0.5 * (1.0 + th)
    dweight_dalpha = 0.5 * float(beta) * sech2
    d2weight_dalpha2 = -(float(beta) ** 2) * th * sech2
    weight_y = dweight_dalpha * np.asarray(alpha_y, dtype=float)
    weight_yy = d2weight_dalpha2 * np.asarray(alpha_y, dtype=float) ** 2 + dweight_dalpha * np.asarray(alpha_yy, dtype=float)
    return weight, weight_y, weight_yy


def _field_pack(source_fn, field: str, x, y, t_abs: float):
    return {
        "val": source_fn(field, x, y, t_abs),
        "x": source_fn(field, x, y, t_abs, dx=1),
        "y": source_fn(field, x, y, t_abs, dy=1),
        "t": source_fn(field, x, y, t_abs, dt_ord=1),
        "xx": source_fn(field, x, y, t_abs, dx=2),
        "xy": source_fn(field, x, y, t_abs, dx=1, dy=1),
        "yy": source_fn(field, x, y, t_abs, dy=2),
        "xxx": source_fn(field, x, y, t_abs, dx=3),
        "xxy": source_fn(field, x, y, t_abs, dx=2, dy=1),
        "xyy": source_fn(field, x, y, t_abs, dx=1, dy=2),
        "yyy": source_fn(field, x, y, t_abs, dy=3),
        "tx": source_fn(field, x, y, t_abs, dx=1, dt_ord=1),
        "ty": source_fn(field, x, y, t_abs, dy=1, dt_ord=1),
        "txx": source_fn(field, x, y, t_abs, dx=2, dt_ord=1),
        "txy": source_fn(field, x, y, t_abs, dx=1, dy=1, dt_ord=1),
        "tyy": source_fn(field, x, y, t_abs, dy=2, dt_ord=1),
    }


def _scalar_pack(eval_fn, x, y, t_abs: float):
    return {
        "val": eval_fn(x, y, t_abs),
        "x": eval_fn(x, y, t_abs, dx=1),
        "y": eval_fn(x, y, t_abs, dy=1),
        "t": eval_fn(x, y, t_abs, dt_ord=1),
        "xx": eval_fn(x, y, t_abs, dx=2),
        "xy": eval_fn(x, y, t_abs, dx=1, dy=1),
        "yy": eval_fn(x, y, t_abs, dy=2),
        "xxx": eval_fn(x, y, t_abs, dx=3),
        "xxy": eval_fn(x, y, t_abs, dx=2, dy=1),
        "xyy": eval_fn(x, y, t_abs, dx=1, dy=2),
        "yyy": eval_fn(x, y, t_abs, dy=3),
        "tx": eval_fn(x, y, t_abs, dx=1, dt_ord=1),
        "ty": eval_fn(x, y, t_abs, dy=1, dt_ord=1),
        "txx": eval_fn(x, y, t_abs, dx=2, dt_ord=1),
        "txy": eval_fn(x, y, t_abs, dx=1, dy=1, dt_ord=1),
        "tyy": eval_fn(x, y, t_abs, dy=2, dt_ord=1),
    }


def _weighted_scalar(pack: dict[str, np.ndarray], *, weight, weight_y, weight_yy, weight_yyy):
    return {
        "val": weight * pack["val"],
        "x": weight * pack["x"],
        "y": weight_y * pack["val"] + weight * pack["y"],
        "t": weight * pack["t"],
        "xx": weight * pack["xx"],
        "xy": weight_y * pack["x"] + weight * pack["xy"],
        "yy": weight_yy * pack["val"] + 2.0 * weight_y * pack["y"] + weight * pack["yy"],
        "xxx": weight * pack["xxx"],
        "xxy": weight_y * pack["xx"] + weight * pack["xxy"],
        "xyy": weight_yy * pack["x"] + 2.0 * weight_y * pack["xy"] + weight * pack["xyy"],
        "yyy": (
            weight_yyy * pack["val"]
            + 3.0 * weight_yy * pack["y"]
            + 3.0 * weight_y * pack["yy"]
            + weight * pack["yyy"]
        ),
        "tx": weight * pack["tx"],
        "ty": weight_y * pack["t"] + weight * pack["ty"],
        "txx": weight * pack["txx"],
        "txy": weight_y * pack["tx"] + weight * pack["txy"],
        "tyy": weight_yy * pack["t"] + 2.0 * weight_y * pack["ty"] + weight * pack["tyy"],
    }


def _build_scale(vmin: float, vmax: float) -> dict[str, float]:
    center = 0.5 * (float(vmin) + float(vmax))
    halfspan = max(0.5 * (float(vmax) - float(vmin)), 1.0e-14)
    return {
        "min": float(vmin),
        "max": float(vmax),
        "center": center,
        "halfspan": halfspan,
    }


def _normalize(values, *, center: float, halfspan: float):
    return (np.asarray(values, dtype=float) - float(center)) / float(halfspan)


def _derivative_factor(power: int, order: int) -> float:
    if int(order) > int(power):
        return 0.0
    out = 1.0
    for shift in range(int(order)):
        out *= float(int(power) - shift)
    return out


def _tensor_terms(*, deg_x: int, deg_y: int, deg_t: int) -> list[tuple[int, int, int]]:
    return [
        (ix, iy, it)
        for ix in range(int(deg_x) + 1)
        for iy in range(int(deg_y) + 1)
        for it in range(int(deg_t) + 1)
    ]


def _tensor_design(
    *,
    terms: list[tuple[int, int, int]],
    x_hat: np.ndarray,
    y_hat: np.ndarray,
    t_hat: np.ndarray,
    scales: dict[str, dict[str, float]],
    dx: int = 0,
    dy: int = 0,
    dt: int = 0,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    x_scale = float(scales["x"]["halfspan"]) ** int(dx)
    y_scale = float(scales["y"]["halfspan"]) ** int(dy)
    t_scale = float(scales["t"]["halfspan"]) ** int(dt)
    normalization = x_scale * y_scale * t_scale
    for ix, iy, it in terms:
        fx = _derivative_factor(ix, int(dx))
        fy = _derivative_factor(iy, int(dy))
        ft = _derivative_factor(it, int(dt))
        if fx == 0.0 or fy == 0.0 or ft == 0.0:
            rows.append(np.zeros_like(x_hat, dtype=float))
            continue
        rows.append(
            (
                fx
                * fy
                * ft
                * np.power(x_hat, ix - int(dx))
                * np.power(y_hat, iy - int(dy))
                * np.power(t_hat, it - int(dt))
            )
            / normalization
        )
    return np.column_stack(rows)


def _evaluate_tensor(coefficients, x, y, t, *, scales, dx: int = 0, dy: int = 0, dt: int = 0):
    arr_x = _normalize(x, center=scales["x"]["center"], halfspan=scales["x"]["halfspan"])
    arr_y = _normalize(y, center=scales["y"]["center"], halfspan=scales["y"]["halfspan"])
    arr_t = _normalize(t, center=scales["t"]["center"], halfspan=scales["t"]["halfspan"])
    coeffs = np.asarray(coefficients, dtype=float)
    out = np.zeros(np.broadcast(arr_x, arr_y, arr_t).shape, dtype=float)
    x_scale = float(scales["x"]["halfspan"]) ** int(dx)
    y_scale = float(scales["y"]["halfspan"]) ** int(dy)
    t_scale = float(scales["t"]["halfspan"]) ** int(dt)
    normalization = x_scale * y_scale * t_scale
    for ix in range(coeffs.shape[0]):
        fx = _derivative_factor(ix, int(dx))
        if fx == 0.0:
            continue
        for iy in range(coeffs.shape[1]):
            fy = _derivative_factor(iy, int(dy))
            if fy == 0.0:
                continue
            for it in range(coeffs.shape[2]):
                ft = _derivative_factor(it, int(dt))
                if ft == 0.0:
                    continue
                out = out + (
                    coeffs[ix, iy, it]
                    * fx
                    * fy
                    * ft
                    * np.power(arr_x, ix - int(dx))
                    * np.power(arr_y, iy - int(dy))
                    * np.power(arr_t, it - int(dt))
                )
    return out / normalization


def _fit_local_phi_transport(
    *,
    fit_module_path: str | Path,
    fit_module,
    solid_eval,
    t_n: float,
    t_k: float,
    phi_b: float,
    kappa_inv: float,
    field_scale: float,
    deg_x: int = 4,
    deg_y: int = 4,
    deg_t: int = 4,
    num_x: int = 18,
    num_y: int = 12,
    num_t: int = 7,
    ic_weight: float = 10.0,
) -> dict[str, object]:
    cache_key = (
        str(Path(fit_module_path).resolve()),
        round(float(field_scale), 12),
        round(float(t_n), 12),
        round(float(t_k), 12),
        round(float(phi_b), 12),
        round(float(kappa_inv), 12),
        int(deg_x),
        int(deg_y),
        int(deg_t),
        int(num_x),
        int(num_y),
        int(num_t),
        round(float(ic_weight), 12),
    )
    cached = _LOCAL_PHI_RECON_CACHE.get(cache_key)
    if cached is not None:
        return cached

    solid_report = fit_module.REPORT["domains"]["solid"]
    t_min = float(solid_report["time"]["min"])
    t_max = float(solid_report["time"]["max"])
    if float(t_n) < t_min - 1.0e-12 or float(t_k) > t_max + 1.0e-12:
        raise ValueError(
            f"Requested local phi reconstruction window [{float(t_n):.6e}, {float(t_k):.6e}] "
            f"lies outside the available fit time range [{t_min:.6e}, {t_max:.6e}]."
        )
    x_bounds = solid_report["reference_bounds"]["x"]
    y_bounds = solid_report["reference_bounds"]["y"]
    scales = {
        "x": _build_scale(float(x_bounds[0]), float(x_bounds[1])),
        "y": _build_scale(float(y_bounds[0]), float(y_bounds[1])),
        "t": _build_scale(float(t_n), float(t_k)),
    }
    terms = _tensor_terms(deg_x=int(deg_x), deg_y=int(deg_y), deg_t=int(deg_t))

    x_line = np.linspace(float(x_bounds[0]), float(x_bounds[1]), int(num_x))
    y_line = np.linspace(float(y_bounds[0]), float(y_bounds[1]), int(num_y))
    t_line = np.linspace(float(t_n), float(t_k), int(num_t))
    Xc, Yc, Tc = np.meshgrid(x_line, y_line, t_line, indexing="ij")
    x_col = np.asarray(Xc, dtype=float).ravel()
    y_col = np.asarray(Yc, dtype=float).ravel()
    t_col = np.asarray(Tc, dtype=float).ravel()
    x_hat = _normalize(x_col, center=scales["x"]["center"], halfspan=scales["x"]["halfspan"])
    y_hat = _normalize(y_col, center=scales["y"]["center"], halfspan=scales["y"]["halfspan"])
    t_hat = _normalize(t_col, center=scales["t"]["center"], halfspan=scales["t"]["halfspan"])
    A = _tensor_design(terms=terms, x_hat=x_hat, y_hat=y_hat, t_hat=t_hat, scales=scales)
    A_t = _tensor_design(terms=terms, x_hat=x_hat, y_hat=y_hat, t_hat=t_hat, scales=scales, dt=1)
    A_x = _tensor_design(terms=terms, x_hat=x_hat, y_hat=y_hat, t_hat=t_hat, scales=scales, dx=1)
    A_y = _tensor_design(terms=terms, x_hat=x_hat, y_hat=y_hat, t_hat=t_hat, scales=scales, dy=1)

    vS_x = np.asarray(solid_eval("ux", x_col, y_col, t_col, dt_ord=1), dtype=float)
    vS_y = np.asarray(solid_eval("uy", x_col, y_col, t_col, dt_ord=1), dtype=float)
    div_vS = np.asarray(
        solid_eval("ux", x_col, y_col, t_col, dx=1, dt_ord=1)
        + solid_eval("uy", x_col, y_col, t_col, dy=1, dt_ord=1),
        dtype=float,
    )
    rhs = np.asarray(
        (
            solid_eval("p_p", x_col, y_col, t_col, dx=2)
            + solid_eval("p_p", x_col, y_col, t_col, dy=2)
        )
        / float(kappa_inv),
        dtype=float,
    )
    operator = A_t + vS_x[:, None] * A_x + vS_y[:, None] * A_y + div_vS[:, None] * A

    Xi, Yi = np.meshgrid(x_line, y_line, indexing="ij")
    x_ic = np.asarray(Xi, dtype=float).ravel()
    y_ic = np.asarray(Yi, dtype=float).ravel()
    t_ic = np.full_like(x_ic, float(t_n), dtype=float)
    x_ic_hat = _normalize(x_ic, center=scales["x"]["center"], halfspan=scales["x"]["halfspan"])
    y_ic_hat = _normalize(y_ic, center=scales["y"]["center"], halfspan=scales["y"]["halfspan"])
    t_ic_hat = _normalize(t_ic, center=scales["t"]["center"], halfspan=scales["t"]["halfspan"])
    A_ic = _tensor_design(terms=terms, x_hat=x_ic_hat, y_hat=y_ic_hat, t_hat=t_ic_hat, scales=scales)

    ux_x = np.asarray(solid_eval("ux", x_ic, y_ic, t_ic, dx=1), dtype=float)
    ux_y = np.asarray(solid_eval("ux", x_ic, y_ic, t_ic, dy=1), dtype=float)
    uy_x = np.asarray(solid_eval("uy", x_ic, y_ic, t_ic, dx=1), dtype=float)
    uy_y = np.asarray(solid_eval("uy", x_ic, y_ic, t_ic, dy=1), dtype=float)
    det_F_inv = (1.0 - ux_x) * (1.0 - uy_y) - ux_y * uy_x
    if np.any(np.abs(det_F_inv) < 1.0e-10):
        raise ValueError("Reference-map Jacobian became singular while reconstructing phi.")
    J_n = 1.0 / det_F_inv
    phi_ic = 1.0 - (1.0 - float(phi_b)) / J_n

    system = np.vstack([operator, float(ic_weight) * A_ic])
    rhs_full = np.concatenate([rhs, float(ic_weight) * phi_ic])
    coeff_vec, residuals, rank, singular_values = np.linalg.lstsq(system, rhs_full, rcond=None)
    coeff_tensor = np.asarray(coeff_vec, dtype=float).reshape((int(deg_x) + 1, int(deg_y) + 1, int(deg_t) + 1))
    pde_misfit = operator @ coeff_vec - rhs
    ic_misfit = A_ic @ coeff_vec - phi_ic
    fit = {
        "coefficients": coeff_tensor,
        "scales": scales,
        "degrees": {"x": int(deg_x), "y": int(deg_y), "t": int(deg_t)},
        "rank": int(rank),
        "singular_values": np.asarray(singular_values, dtype=float),
        "residual_sum_squares": float(residuals[0]) if residuals.size else float(np.sum((system @ coeff_vec - rhs_full) ** 2)),
        "pde_relative_l2": float(np.linalg.norm(pde_misfit) / max(np.linalg.norm(rhs), 1.0e-16)),
        "ic_rmse": float(np.sqrt(np.mean(ic_misfit * ic_misfit))),
    }
    _LOCAL_PHI_RECON_CACHE[cache_key] = fit
    return fit


def build_seboldt_quasistatic_mms_step(
    *,
    fit_module_path: str | Path = DEFAULT_FIT_MODULE_PATH,
    dt_val: float = 5.0e-4,
    t0: float = 1.0,
    theta: float = 1.0,
    y_interface: float = 1.0,
    eps_alpha: float = 0.12,
    interface_beta: float = 40.0,
    support_indicator_beta: float = 4.0,
    field_scale: float = 1.0,
    rho_f: float = 1.0,
    rho_s: float = 1.0,
    mu_f: float = 3.5e-2,
    kappa_inv: float = 1.0e3,
    mu_s: float = 1.67785e5,
    lambda_s: float = 8.22148e6,
    phi_b: float = 0.30,
    fluid_convection: str = "full",
    gamma_v: float = 1.0,
    gamma_p: float = 1.0,
    gamma_vP: float = 1.0,
    gamma_p_pore: float = 1.0,
    gamma_u: float = 1.0,
    disable_pore_momentum: bool = False,
    disable_solid_momentum: bool = True,
    combined_porous_momentum: bool = True,
) -> SeboldtQuasiStaticMMSStep:
    dt = float(dt_val)
    if not (dt > 0.0):
        raise ValueError("dt_val must be positive.")
    supported_layouts = {
        (False, False, False),
        (False, True, True),
        (True, True, True),
    }
    layout_key = (
        bool(disable_pore_momentum),
        bool(disable_solid_momentum),
        bool(combined_porous_momentum),
    )
    if layout_key not in supported_layouts:
        raise NotImplementedError(
            "This MMS builder currently supports only the combined porous "
            "quasi-static layouts: "
            "(disable_pore_momentum=False, disable_solid_momentum=False, combined_porous_momentum=False), "
            "(disable_pore_momentum=False, disable_solid_momentum=True, combined_porous_momentum=True) "
            "or (disable_pore_momentum=True, disable_solid_momentum=True, combined_porous_momentum=True)."
        )
    if float(support_indicator_beta) < 0.0:
        raise ValueError("support_indicator_beta must be non-negative.")

    t_n = float(t0)
    t_k = float(t0) + dt
    theta_val = float(theta)
    if not (0.0 <= theta_val <= 1.0):
        raise ValueError("theta must lie in [0, 1].")
    fluid_conv_key = str(fluid_convection or "full").strip().lower().replace("-", "_")
    if fluid_conv_key not in {"full", "off"}:
        raise ValueError("fluid_convection must be 'full' or 'off'.")
    field_scale_val = float(field_scale)
    if not (field_scale_val > 0.0):
        raise ValueError("field_scale must be positive.")

    fit = _load_fit_module(fit_module_path)
    fit_report = getattr(fit, "REPORT", {}) if hasattr(fit, "REPORT") else {}
    fit_degrees = fit_report.get("degrees", {}) if isinstance(fit_report, dict) else {}
    phi_deg_x = max(1, int(fit_degrees.get("x", 4)))
    phi_deg_y = max(1, int(fit_degrees.get("y", 4)))
    phi_deg_t = max(1, int(fit_degrees.get("t", 4)))

    def _fluid_raw(field: str, x, y, t_abs: float, *, dx: int = 0, dy: int = 0, dt_ord: int = 0):
        return field_scale_val * np.asarray(
            fit.evaluate_derivative("fluid", field, x, y, t_abs, dx=int(dx), dy=int(dy), dt=int(dt_ord)),
            dtype=float,
        )

    def _solid_raw(field: str, x, y, t_abs: float, *, dx: int = 0, dy: int = 0, dt_ord: int = 0):
        return field_scale_val * np.asarray(
            fit.evaluate_derivative("solid", field, x, y, t_abs, dx=int(dx), dy=int(dy), dt=int(dt_ord)),
            dtype=float,
        )

    phi_local_fit = _fit_local_phi_transport(
        fit_module_path=fit_module_path,
        fit_module=fit,
        solid_eval=_solid_raw,
        t_n=float(t_n),
        t_k=float(t_k),
        phi_b=float(phi_b),
        kappa_inv=float(kappa_inv),
        field_scale=float(field_scale_val),
        deg_x=int(phi_deg_x),
        deg_y=int(phi_deg_y),
        deg_t=int(phi_deg_t),
        num_x=max(12, (4 * int(phi_deg_x)) + 2),
        num_y=max(10, (4 * int(phi_deg_y)) + 2),
        num_t=max(5, (2 * int(phi_deg_t)) + 3),
    )

    def _phi_porous_raw(x, y, t_abs: float, *, dx: int = 0, dy: int = 0, dt_ord: int = 0):
        return _evaluate_tensor(
            phi_local_fit["coefficients"],
            x,
            y,
            float(t_abs),
            scales=phi_local_fit["scales"],
            dx=int(dx),
            dy=int(dy),
            dt=int(dt_ord),
        )

    def _exact_state(x, y, t_abs: float):
        xv = np.asarray(x, dtype=float)
        yv = np.asarray(y, dtype=float)
        if float(eps_alpha) > 0.0:
            alpha, alpha_y, alpha_yy, alpha_yyy = _alpha_from_eps(
                yv,
                y_interface=float(y_interface),
                eps_alpha=float(eps_alpha),
            )
        else:
            alpha, alpha_y, alpha_yy, alpha_yyy = _alpha_from_beta(
                yv,
                y_interface=float(y_interface),
                beta=float(interface_beta),
        )
        alpha_bulk, alpha_bulk_y, _ = _support_indicator_raw(
            alpha,
            alpha_y,
            alpha_yy,
            beta=float(support_indicator_beta),
        )
        F_bulk = 1.0 - alpha_bulk
        F_bulk_y = -alpha_bulk_y

        vx = _field_pack(_fluid_raw, "vx", xv, yv, t_abs)
        vy = _field_pack(_fluid_raw, "vy", xv, yv, t_abs)
        p = _field_pack(_fluid_raw, "p", xv, yv, t_abs)
        ux = _field_pack(_solid_raw, "ux", xv, yv, t_abs)
        uy = _field_pack(_solid_raw, "uy", xv, yv, t_abs)
        p_pore = _field_pack(_solid_raw, "p_p", xv, yv, t_abs)
        phi_porous = _scalar_pack(_phi_porous_raw, xv, yv, t_abs)
        phi_minus_one = {key: np.asarray(value, dtype=float).copy() for key, value in phi_porous.items()}
        phi_minus_one["val"] = phi_minus_one["val"] - 1.0
        phi_pack = _weighted_scalar(
            phi_minus_one,
            weight=alpha,
            weight_y=alpha_y,
            weight_yy=alpha_yy,
            weight_yyy=alpha_yyy,
        )
        phi_pack["val"] = phi_pack["val"] + 1.0

        v = np.stack([vx["val"], vy["val"]], axis=-1)
        u = np.stack([ux["val"], uy["val"]], axis=-1)
        grad_u = np.stack(
            [
                np.stack([ux["x"], ux["y"]], axis=-1),
                np.stack([uy["x"], uy["y"]], axis=-1),
            ],
            axis=-2,
        )
        vS = np.stack([ux["t"], uy["t"]], axis=-1)
        div_vS = ux["tx"] + uy["ty"]

        grad_p_pore = np.stack([p_pore["x"], p_pore["y"]], axis=-1)
        lap_p_pore = p_pore["xx"] + p_pore["yy"]
        phi = phi_pack["val"]
        inv_phi_kappa = 1.0 / (phi * float(kappa_inv))
        inv_phi_kappa_x = -(phi_pack["x"]) / (float(kappa_inv) * phi * phi)
        inv_phi_kappa_y = -(phi_pack["y"]) / (float(kappa_inv) * phi * phi)
        inv_phi_kappa_xx = (
            -(phi_pack["xx"]) / (float(kappa_inv) * phi * phi)
            + (2.0 * phi_pack["x"] * phi_pack["x"]) / (float(kappa_inv) * phi * phi * phi)
        )
        inv_phi_kappa_yy = (
            -(phi_pack["yy"]) / (float(kappa_inv) * phi * phi)
            + (2.0 * phi_pack["y"] * phi_pack["y"]) / (float(kappa_inv) * phi * phi * phi)
        )
        rel_x = p_pore["x"]
        rel_y = p_pore["y"]

        vP_x = ux["t"] - inv_phi_kappa * p_pore["x"]
        vP_y = uy["t"] - inv_phi_kappa * p_pore["y"]
        vP_x_x = ux["tx"] - (inv_phi_kappa_x * rel_x + inv_phi_kappa * p_pore["xx"])
        vP_x_y = ux["ty"] - (inv_phi_kappa_y * rel_x + inv_phi_kappa * p_pore["xy"])
        vP_y_x = uy["tx"] - (inv_phi_kappa_x * rel_y + inv_phi_kappa * p_pore["xy"])
        vP_y_y = uy["ty"] - (inv_phi_kappa_y * rel_y + inv_phi_kappa * p_pore["yy"])
        vP_x_xx = ux["txx"] - (
            inv_phi_kappa_xx * rel_x
            + 2.0 * inv_phi_kappa_x * p_pore["xx"]
            + inv_phi_kappa * p_pore["xxx"]
        )
        vP_x_yy = ux["tyy"] - (
            inv_phi_kappa_yy * p_pore["x"]
            + 2.0 * inv_phi_kappa_y * p_pore["xy"]
            + inv_phi_kappa * p_pore["xyy"]
        )
        vP_y_xx = uy["txx"] - (
            inv_phi_kappa_xx * rel_y
            + 2.0 * inv_phi_kappa_x * p_pore["xy"]
            + inv_phi_kappa * p_pore["xxy"]
        )
        vP_y_yy = uy["tyy"] - (
            inv_phi_kappa_yy * p_pore["y"]
            + 2.0 * inv_phi_kappa_y * p_pore["yy"]
            + inv_phi_kappa * p_pore["yyy"]
        )
        vP = np.stack([vP_x, vP_y], axis=-1)

        flux = vS - (grad_p_pore / float(kappa_inv))
        div_flux = div_vS - (lap_p_pore / float(kappa_inv))

        return {
            "alpha": alpha,
            "alpha_y": alpha_y,
            "alpha_bulk": alpha_bulk,
            "alpha_bulk_y": alpha_bulk_y,
            "F_bulk": F_bulk,
            "F_bulk_y": F_bulk_y,
            "phi": phi,
            "phi_pack": phi_pack,
            "v": v,
            "vx_pack": vx,
            "vy_pack": vy,
            "p_pack": p,
            "u": u,
            "ux_pack": ux,
            "uy_pack": uy,
            "grad_u": grad_u,
            "vS": vS,
            "div_vS": div_vS,
            "vP": vP,
            "vP_x_x": vP_x_x,
            "vP_x_y": vP_x_y,
            "vP_y_x": vP_y_x,
            "vP_y_y": vP_y_y,
            "vP_x_xx": vP_x_xx,
            "vP_x_yy": vP_x_yy,
            "vP_y_xx": vP_y_xx,
            "vP_y_yy": vP_y_yy,
            "p_pore_pack": p_pore,
            "grad_p_pore": grad_p_pore,
            "lap_p_pore": lap_p_pore,
            "flux": flux,
            "div_flux": div_flux,
        }

    def _exact_values(x, y, t_abs: float):
        xv = np.asarray(x, dtype=float)
        yv = np.asarray(y, dtype=float)
        if float(eps_alpha) > 0.0:
            alpha, _, _, _ = _alpha_from_eps(
                yv,
                y_interface=float(y_interface),
                eps_alpha=float(eps_alpha),
            )
        else:
            alpha, _, _, _ = _alpha_from_beta(
                yv,
                y_interface=float(y_interface),
                beta=float(interface_beta),
            )
        phi_porous = _phi_porous_raw(xv, yv, t_abs)
        phi = 1.0 + alpha * (phi_porous - 1.0)

        vx_val = _fluid_raw("vx", xv, yv, t_abs)
        vy_val = _fluid_raw("vy", xv, yv, t_abs)
        p_val = _fluid_raw("p", xv, yv, t_abs)
        ux_val = _solid_raw("ux", xv, yv, t_abs)
        uy_val = _solid_raw("uy", xv, yv, t_abs)
        p_p_raw = _solid_raw("p_p", xv, yv, t_abs)
        p_p_x = _solid_raw("p_p", xv, yv, t_abs, dx=1)
        p_p_y = _solid_raw("p_p", xv, yv, t_abs, dy=1)
        vS_x = _solid_raw("ux", xv, yv, t_abs, dt_ord=1)
        vS_y = _solid_raw("uy", xv, yv, t_abs, dt_ord=1)

        return {
            "v": np.stack([vx_val, vy_val], axis=-1),
            "p": p_val,
            "u": np.stack([ux_val, uy_val], axis=-1),
            "p_pore": p_p_raw,
            "vS": np.stack([vS_x, vS_y], axis=-1),
            "vP": np.stack(
                [
                    vS_x - (p_p_x / (phi * float(kappa_inv))),
                    vS_y - (p_p_y / (phi * float(kappa_inv))),
                ],
                axis=-1,
            ),
            "alpha": alpha,
            "phi": phi,
        }

    def _fluid_extension_source(cur):
        return float(gamma_v) * np.stack(
            [
                -(
                    cur["alpha_bulk_y"] * cur["vx_pack"]["y"]
                    + cur["alpha_bulk"] * (cur["vx_pack"]["xx"] + cur["vx_pack"]["yy"])
                ),
                -(
                    cur["alpha_bulk_y"] * cur["vy_pack"]["y"]
                    + cur["alpha_bulk"] * (cur["vy_pack"]["xx"] + cur["vy_pack"]["yy"])
                ),
            ],
            axis=-1,
        )

    def _pressure_extension_source(cur):
        return float(gamma_p) * (
            -(
                cur["alpha_bulk_y"] * cur["p_pack"]["y"]
                + cur["alpha_bulk"] * (cur["p_pack"]["xx"] + cur["p_pack"]["yy"])
            )
        )

    def _vP_extension_source(cur):
        return float(gamma_vP) * np.stack(
            [
                -(
                    cur["F_bulk_y"] * cur["vP_x_y"]
                    + cur["F_bulk"] * (cur["vP_x_xx"] + cur["vP_x_yy"])
                ),
                -(
                    cur["F_bulk_y"] * cur["vP_y_y"]
                    + cur["F_bulk"] * (cur["vP_y_xx"] + cur["vP_y_yy"])
                ),
            ],
            axis=-1,
        )

    def _p_pore_extension_source(cur):
        return float(gamma_p_pore) * (
            -(
                cur["F_bulk_y"] * cur["p_pore_pack"]["y"]
                + cur["F_bulk"] * (cur["p_pore_pack"]["xx"] + cur["p_pore_pack"]["yy"])
            )
        )

    def _u_extension_source(cur):
        return float(gamma_u) * np.stack(
            [
                -(
                    cur["F_bulk_y"] * cur["ux_pack"]["y"]
                    + cur["F_bulk"] * (cur["ux_pack"]["xx"] + cur["ux_pack"]["yy"])
                ),
                -(
                    cur["F_bulk_y"] * cur["uy_pack"]["y"]
                    + cur["F_bulk"] * (cur["uy_pack"]["xx"] + cur["uy_pack"]["yy"])
                ),
            ],
            axis=-1,
        )

    def _fluid_source(x, y):
        cur = _exact_state(x, y, t_k)
        prev = _exact_state(x, y, t_n)
        vx = cur["vx_pack"]
        vy = cur["vy_pack"]
        p = cur["p_pack"]
        conv = np.zeros_like(cur["v"])
        if fluid_conv_key != "off":
            conv = np.stack(
                [
                    cur["v"][..., 0] * vx["x"] + cur["v"][..., 1] * vx["y"],
                    cur["v"][..., 0] * vy["x"] + cur["v"][..., 1] * vy["y"],
                ],
                axis=-1,
            )
        sigma_xx = (2.0 * float(mu_f) * vx["x"]) - p["val"]
        sigma_xy = float(mu_f) * (vx["y"] + vy["x"])
        sigma_yy = (2.0 * float(mu_f) * vy["y"]) - p["val"]
        sigma_xx_x = (2.0 * float(mu_f) * vx["xx"]) - p["x"]
        sigma_xy_x = float(mu_f) * (vx["xy"] + vy["xx"])
        sigma_xy_y = float(mu_f) * (vx["yy"] + vy["xy"])
        sigma_yy_y = (2.0 * float(mu_f) * vy["yy"]) - p["y"]
        div_F_sigma = np.stack(
            [
                cur["F_bulk"] * sigma_xx_x + cur["F_bulk_y"] * sigma_xy + cur["F_bulk"] * sigma_xy_y,
                cur["F_bulk"] * sigma_xy_x + cur["F_bulk_y"] * sigma_yy + cur["F_bulk"] * sigma_yy_y,
            ],
            axis=-1,
        )
        return (
            cur["F_bulk"][..., None]
            * float(rho_f)
            * ((cur["v"] - prev["v"]) / float(dt) + conv)
        ) - div_F_sigma + _fluid_extension_source(cur)

    def _free_mass_source(x, y):
        cur = _exact_state(x, y, t_k)
        return cur["F_bulk"] * (cur["vx_pack"]["x"] + cur["vy_pack"]["y"]) + _pressure_extension_source(cur)

    def _pore_momentum_source(x, y):
        cur = _exact_state(x, y, t_k)
        rel_p = cur["vP"] - cur["vS"]
        physical = np.stack(
            [
                cur["alpha_bulk"] * (cur["p_pore_pack"]["x"] + float(kappa_inv) * cur["phi"] * rel_p[..., 0]),
                cur["alpha_bulk_y"] * cur["p_pore_pack"]["val"]
                + cur["alpha_bulk"] * (cur["p_pore_pack"]["y"] + float(kappa_inv) * cur["phi"] * rel_p[..., 1]),
            ],
            axis=-1,
        )
        return physical + _vP_extension_source(cur)

    def _porous_mass_source(x, y):
        cur = _exact_state(x, y, t_k)
        jump_y = (float(rho_f) * cur["v"][..., 1]) - (float(rho_s) * cur["flux"][..., 1])
        return (
            cur["alpha_bulk"] * cur["div_flux"]
            + cur["alpha_bulk_y"] * cur["flux"][..., 1]
            - cur["alpha_y"] * jump_y
            + _p_pore_extension_source(cur)
        )

    def _porous_body_source(x, y):
        cur = _exact_state(x, y, t_k)
        ux = cur["ux_pack"]
        uy = cur["uy_pack"]
        p_pore = cur["p_pore_pack"]
        div_u = ux["x"] + uy["y"]
        div_u_x = ux["xx"] + uy["xy"]
        div_u_y = ux["xy"] + uy["yy"]

        sigma_s_xx = (2.0 * float(mu_s) * ux["x"]) + float(lambda_s) * div_u
        sigma_s_yy = (2.0 * float(mu_s) * uy["y"]) + float(lambda_s) * div_u
        sigma_s_xy = float(mu_s) * (ux["y"] + uy["x"])
        sigma_s_xx_x = (2.0 * float(mu_s) * ux["xx"]) + float(lambda_s) * div_u_x
        sigma_s_yy_y = (2.0 * float(mu_s) * uy["yy"]) + float(lambda_s) * div_u_y
        sigma_s_xy_x = float(mu_s) * (ux["xy"] + uy["xx"])
        sigma_s_xy_y = float(mu_s) * (ux["yy"] + uy["xy"])

        phi = cur["phi"]
        phi_pack = cur["phi_pack"]
        one_m_phi = 1.0 - phi

        porous_xx = -(phi * p_pore["val"]) + one_m_phi * sigma_s_xx
        porous_xy = one_m_phi * sigma_s_xy
        porous_yy = -(phi * p_pore["val"]) + one_m_phi * sigma_s_yy
        porous_xx_x = (
            -(phi_pack["x"] * p_pore["val"] + phi * p_pore["x"])
            - phi_pack["x"] * sigma_s_xx
            + one_m_phi * sigma_s_xx_x
        )
        porous_xy_x = -(phi_pack["x"] * sigma_s_xy) + one_m_phi * sigma_s_xy_x
        porous_xy_y = -(phi_pack["y"] * sigma_s_xy) + one_m_phi * sigma_s_xy_y
        porous_yy_y = (
            -(phi_pack["y"] * p_pore["val"] + phi * p_pore["y"])
            - (phi_pack["y"] * sigma_s_yy)
            + one_m_phi * sigma_s_yy_y
        )

        div_alpha_sigma = np.stack(
            [
                cur["alpha_bulk"] * porous_xx_x + cur["alpha_bulk_y"] * porous_xy + cur["alpha_bulk"] * porous_xy_y,
                cur["alpha_bulk"] * porous_xy_x + cur["alpha_bulk_y"] * porous_yy + cur["alpha_bulk"] * porous_yy_y,
            ],
            axis=-1,
        )

        sigma_f_xy = float(mu_f) * (cur["vx_pack"]["y"] + cur["vy_pack"]["x"])
        sigma_f_yy = (2.0 * float(mu_f) * cur["vy_pack"]["y"]) - cur["p_pack"]["val"]
        traction_jump = np.stack(
            [
                cur["alpha_y"] * (sigma_f_xy - one_m_phi * sigma_s_xy),
                cur["alpha_y"] * (
                    sigma_f_yy + (phi * p_pore["val"]) - (one_m_phi * sigma_s_yy)
                ),
            ],
            axis=-1,
        )
        return -div_alpha_sigma + traction_jump

    def _kinematics_source(x, y):
        cur = _exact_state(x, y, t_k)
        prev = _exact_state(x, y, t_n)
        grad_u_vS = np.stack(
            [
                cur["ux_pack"]["x"] * cur["vS"][..., 0] + cur["ux_pack"]["y"] * cur["vS"][..., 1],
                cur["uy_pack"]["x"] * cur["vS"][..., 0] + cur["uy_pack"]["y"] * cur["vS"][..., 1],
            ],
            axis=-1,
        )
        return cur["alpha_bulk"][..., None] * (
            ((cur["u"] - prev["u"]) / float(dt)) + grad_u_vS - cur["vS"]
        ) + _u_extension_source(cur)

    def _alpha_source(x, y):
        cur = _exact_state(x, y, t_k)
        return cur["alpha_y"] * cur["vS"][..., 1]

    def _phi_source(x, y):
        cur = _exact_state(x, y, t_k)
        prev = _exact_state(x, y, t_n)
        div_vP = cur["vP_x_x"] + cur["vP_y_y"]
        return cur["alpha_bulk"] * (
            ((cur["phi"] - prev["phi"]) / float(dt))
            + cur["phi"] * div_vP
            + cur["phi_pack"]["x"] * cur["vP"][..., 0]
            + cur["phi_pack"]["y"] * cur["vP"][..., 1]
        )

    def _v_local(x, y, t_local):
        return _exact_values(x, y, t_n + float(t_local))["v"]

    def _p_local(x, y, t_local):
        return _exact_values(x, y, t_n + float(t_local))["p"]

    def _vP_local(x, y, t_local):
        return _exact_values(x, y, t_n + float(t_local))["vP"]

    def _p_pore_local(x, y, t_local):
        return _exact_values(x, y, t_n + float(t_local))["p_pore"]

    def _vS_local(x, y, t_local):
        return _exact_values(x, y, t_n + float(t_local))["vS"]

    def _u_local(x, y, t_local):
        return _exact_values(x, y, t_n + float(t_local))["u"]

    def _alpha_local(x, y, t_local):
        return _exact_values(x, y, t_n + float(t_local))["alpha"]

    def _phi_local(x, y, t_local):
        return _exact_values(x, y, t_n + float(t_local))["phi"]

    v_n = lambda x, y: _exact_values(x, y, t_n)["v"]
    p_n = lambda x, y: _exact_values(x, y, t_n)["p"]
    vP_n = lambda x, y: _exact_values(x, y, t_n)["vP"]
    p_pore_n = lambda x, y: _exact_values(x, y, t_n)["p_pore"]
    vS_n = lambda x, y: _exact_values(x, y, t_n)["vS"]
    u_n = lambda x, y: _exact_values(x, y, t_n)["u"]
    alpha_n = lambda x, y: _exact_values(x, y, t_n)["alpha"]
    phi_n = lambda x, y: _exact_values(x, y, t_n)["phi"]

    v_k = lambda x, y: _exact_values(x, y, t_k)["v"]
    p_k = lambda x, y: _exact_values(x, y, t_k)["p"]
    vP_k = lambda x, y: _exact_values(x, y, t_k)["vP"]
    p_pore_k = lambda x, y: _exact_values(x, y, t_k)["p_pore"]
    vS_k = lambda x, y: _exact_values(x, y, t_k)["vS"]
    u_k = lambda x, y: _exact_values(x, y, t_k)["u"]
    alpha_k = lambda x, y: _exact_values(x, y, t_k)["alpha"]
    phi_k = lambda x, y: _exact_values(x, y, t_k)["phi"]

    return SeboldtQuasiStaticMMSStep(
        fit_module_path=str(Path(fit_module_path).resolve()),
        t_origin=float(t_n),
        t_n=float(t_n),
        t_k=float(t_k),
        dt=float(dt),
        theta=float(theta_val),
        y_interface=float(y_interface),
        eps_alpha=float(eps_alpha),
        interface_beta=float(interface_beta),
        support_indicator_beta=float(support_indicator_beta),
        field_scale=float(field_scale_val),
        rho_f=float(rho_f),
        rho_s=float(rho_s),
        mu_f=float(mu_f),
        kappa_inv=float(kappa_inv),
        mu_s=float(mu_s),
        lambda_s=float(lambda_s),
        phi_b=float(phi_b),
        fluid_convection=str(fluid_conv_key),
        v=_v_local,
        p=_p_local,
        vP=_vP_local,
        p_pore=_p_pore_local,
        vS=_vS_local,
        u=_u_local,
        alpha=_alpha_local,
        phi=_phi_local,
        v_n=v_n,
        p_n=p_n,
        vP_n=vP_n,
        p_pore_n=p_pore_n,
        vS_n=vS_n,
        u_n=u_n,
        alpha_n=alpha_n,
        phi_n=phi_n,
        v_k=v_k,
        p_k=p_k,
        vP_k=vP_k,
        p_pore_k=p_pore_k,
        vS_k=vS_k,
        u_k=u_k,
        alpha_k=alpha_k,
        phi_k=phi_k,
        f_fluid_momentum=_fluid_source,
        s_free_mass=_free_mass_source,
        f_pore_momentum=_pore_momentum_source,
        s_porous_mass=_porous_mass_source,
        f_porous_body=_porous_body_source,
        f_kinematics=_kinematics_source,
        f_alpha=_alpha_source,
        f_phi=_phi_source,
    )
