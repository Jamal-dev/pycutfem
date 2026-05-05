#!/usr/bin/env python3
"""Stoter et al. Section 5.3: 2D channel flow intercepted by a porous medium.

This driver implements the diffuse Navier--Stokes/Darcy formulation from
Stoter et al., CMAME 321 (2017), Section 5.3 on the paper geometry shown in
Fig. 15(a):

  - outer box: 80 mm x 100 mm,
  - lower and upper Stokes channel segments: width 30 mm,
  - lower/upper channel depth: 30 mm,
  - Darcy region: surrounding porous medium.

The active flow model follows the diffuse weak form (28) from the paper:

  - Stokes/Navier--Stokes velocity/pressure `(u_N, p_N)` on the full box,
  - Darcy head `phi_D` on the full box,
  - interface mass transfer via `q_D (u_N · grad(c))`,
  - normal traction transfer via `- g phi_D (w_N · grad(c))`,
  - BJS tangential drag via `(alpha/sqrt(K)) (u_N·tau) (w_N·tau) |grad(c)|`,
  - LSIC stabilization in the Stokes region with `tau_LSIC = Umax h / 2`.

This is a benchmark-local diagnostic driver. It is intentionally independent of
the Seboldt one-domain Stokes--Biot formulation so we can use it as a reference
implementation while debugging our production model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import scipy.sparse.linalg as spla

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    HdivFunctionComponent,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    div,
    dot,
    grad,
    inner,
)
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _eps(v):
    return Constant(0.5) * (grad(v) + grad(v).T)


def _sech2(z: np.ndarray) -> np.ndarray:
    zz = np.asarray(z, dtype=float)
    return 1.0 / np.cosh(zz) ** 2


def _sqrt(expr):
    return expr ** Constant(0.5)


def _smooth_left(x, a, eps):
    xx = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.tanh((xx - float(a)) / float(eps)))


def _smooth_right(x, b, eps):
    xx = np.asarray(x, dtype=float)
    return 0.5 * (1.0 - np.tanh((xx - float(b)) / float(eps)))


def _smooth_band(x, a, b, eps):
    return _smooth_left(x, a, eps) * _smooth_right(x, b, eps)


def _dsmooth_left(x, a, eps):
    xx = np.asarray(x, dtype=float)
    ee = float(eps)
    return 0.5 * _sech2((xx - float(a)) / ee) / ee


def _dsmooth_right(x, b, eps):
    xx = np.asarray(x, dtype=float)
    ee = float(eps)
    return -0.5 * _sech2((xx - float(b)) / ee) / ee


def _dsmooth_band(x, a, b, eps):
    return _dsmooth_left(x, a, eps) * _smooth_right(x, b, eps) + _smooth_left(x, a, eps) * _dsmooth_right(x, b, eps)


@dataclass(frozen=True)
class Geometry:
    Lx: float = 80.0
    Ly: float = 100.0
    channel_x0: float = 25.0
    channel_x1: float = 55.0
    lower_interface_y: float = 30.0
    upper_interface_y: float = 70.0
    r_in: float = 8.5

    @property
    def center_x(self) -> float:
        return 0.5 * (self.channel_x0 + self.channel_x1)

    @property
    def half_channel_width(self) -> float:
        return 0.5 * (self.channel_x1 - self.channel_x0)


def _stokes_phase_fields(geom: Geometry, eps: float):
    x0 = float(geom.channel_x0)
    x1 = float(geom.channel_x1)
    y_lo = float(geom.lower_interface_y)
    y_hi = float(geom.upper_interface_y)
    ee = float(eps)

    def sx(x, y):
        return _smooth_band(x, x0, x1, ee)

    def dsx_dx(x, y):
        return _dsmooth_band(x, x0, x1, ee)

    def below_lo(x, y):
        yy = np.asarray(y, dtype=float)
        return 0.5 * (1.0 - np.tanh((yy - y_lo) / ee))

    def dbelow_lo_dy(x, y):
        yy = np.asarray(y, dtype=float)
        return -0.5 * _sech2((yy - y_lo) / ee) / ee

    def above_hi(x, y):
        yy = np.asarray(y, dtype=float)
        return 0.5 * (1.0 + np.tanh((yy - y_hi) / ee))

    def dabove_hi_dy(x, y):
        yy = np.asarray(y, dtype=float)
        return 0.5 * _sech2((yy - y_hi) / ee) / ee

    def c(x, y):
        return sx(x, y) * (below_lo(x, y) + above_hi(x, y))

    def grad_c(x, y):
        xarr = np.asarray(x, dtype=float)
        yarr = np.asarray(y, dtype=float)
        common_y = below_lo(xarr, yarr) + above_hi(xarr, yarr)
        dc_dx = dsx_dx(xarr, yarr) * common_y
        dc_dy = sx(xarr, yarr) * (dbelow_lo_dy(xarr, yarr) + dabove_hi_dy(xarr, yarr))
        return np.stack([dc_dx, dc_dy], axis=-1)

    def abs_grad_c(x, y):
        gc = grad_c(x, y)
        return np.sqrt(gc[..., 0] ** 2 + gc[..., 1] ** 2)

    def tau_vec(x, y):
        gc = grad_c(x, y)
        mag = np.sqrt(gc[..., 0] ** 2 + gc[..., 1] ** 2)
        denom = np.maximum(mag, 1.0e-14)
        return np.stack([-gc[..., 1] / denom, gc[..., 0] / denom], axis=-1)

    def c_darcy(x, y):
        box_x = _smooth_band(x, 0.0, float(geom.Lx), ee)
        box_y = _smooth_band(y, 0.0, float(geom.Ly), ee)
        return box_x * box_y * (1.0 - c(x, y))

    return c, grad_c, abs_grad_c, tau_vec, c_darcy


def _is_stokes_region(geom: Geometry, x: float, y: float) -> bool:
    in_x = float(geom.channel_x0) <= float(x) <= float(geom.channel_x1)
    in_lower = 0.0 <= float(y) <= float(geom.lower_interface_y)
    in_upper = float(geom.upper_interface_y) <= float(y) <= float(geom.Ly)
    return bool(in_x and (in_lower or in_upper))


def _darcy_outer_distance(geom: Geometry, x: float, y: float) -> float:
    xx = float(x)
    yy = float(y)
    vals = [xx, float(geom.Lx) - xx]
    if xx <= float(geom.channel_x0) or xx >= float(geom.channel_x1):
        vals.extend([yy, float(geom.Ly) - yy])
    return float(min(vals))


def _phase_field_indicator_stokes(geom: Geometry):
    def indicator(x, y):
        xx = np.asarray(x, dtype=float)
        yy = np.asarray(y, dtype=float)
        out = np.zeros_like(xx, dtype=float)
        flat_x = xx.reshape(-1)
        flat_y = yy.reshape(-1)
        flat_o = out.reshape(-1)
        for i in range(flat_o.size):
            flat_o[i] = 1.0 if _is_stokes_region(geom, float(flat_x[i]), float(flat_y[i])) else 0.0
        return out
    return indicator


def _phase_field_indicator_darcy(geom: Geometry, eps: float):
    ee = float(eps)
    def indicator(x, y):
        xx = np.asarray(x, dtype=float)
        yy = np.asarray(y, dtype=float)
        out = np.zeros_like(xx, dtype=float)
        flat_x = xx.reshape(-1)
        flat_y = yy.reshape(-1)
        flat_o = out.reshape(-1)
        for i in range(flat_o.size):
            xi = float(flat_x[i])
            yi = float(flat_y[i])
            if _is_stokes_region(geom, xi, yi):
                flat_o[i] = 0.0
                continue
            flat_o[i] = 1.0 if _darcy_outer_distance(geom, xi, yi) > ee else 0.0
        return out
    return indicator


def _solve_allen_cahn_indicator(
    *,
    mesh: Mesh,
    eps: float,
    indicator_fun,
    dt: float,
    max_steps: int,
    backend: str,
) -> tuple[np.ndarray, dict]:
    me_pf = MixedElement(mesh, field_specs={"c": 1})
    dh_pf = DofHandler(me_pf, method="cg")
    c_trial = TrialFunction("c", dof_handler=dh_pf)
    psi = TestFunction("c", dof_handler=dh_pf)
    c_curr = Function("c_curr", "c", dof_handler=dh_pf)
    c_prev = Function("c_prev", "c", dof_handler=dh_pf)
    c_work = Function("c_work", "c", dof_handler=dh_pf)

    coords = dh_pf.get_dof_coords("c")
    init_vals = np.asarray([float(indicator_fun(x, y)) for x, y in coords], dtype=float)
    c_curr.nodal_values[:] = init_vals
    c_prev.nodal_values[:] = init_vals

    quad_order = 4
    dt_c = Constant(float(dt))
    eps_c = Constant(float(eps))

    def fprime(fun: Function):
        c = fun
        return Constant(4.0) * c * (c - Constant(1.0)) * (Constant(2.0) * c - Constant(1.0))

    a_be = ((c_trial / dt_c) * psi + (eps_c * eps_c) * inner(grad(c_trial), grad(psi))) * dx()
    L_be = ((c_curr / dt_c) - fprime(c_curr)) * psi * dx()
    A_be, b_be = assemble_form(Equation(a_be, L_be), dof_handler=dh_pf, bcs=[], quad_order=quad_order, backend=backend)
    c1 = np.asarray(spla.spsolve(A_be.tocsc(), b_be), dtype=float).ravel()
    c_work.nodal_values[:] = c1
    diff0 = float(np.linalg.norm(c_work.nodal_values - c_curr.nodal_values))
    target = 0.025 * diff0

    c_prev.nodal_values[:] = c_curr.nodal_values[:]
    c_curr.nodal_values[:] = c_work.nodal_values[:]

    step = 1
    while step < int(max_steps):
        a_bdf2 = (((Constant(3.0) / (Constant(2.0) * dt_c)) * c_trial) * psi + (eps_c * eps_c) * inner(grad(c_trial), grad(psi))) * dx()
        rhs_expr = (
            ((Constant(4.0) * c_curr - c_prev) / (Constant(2.0) * dt_c))
            - (Constant(2.0) * fprime(c_curr) - fprime(c_prev))
        )
        L_bdf2 = rhs_expr * psi * dx()
        A, b = assemble_form(Equation(a_bdf2, L_bdf2), dof_handler=dh_pf, bcs=[], quad_order=quad_order, backend=backend)
        c_new = np.asarray(spla.spsolve(A.tocsc(), b), dtype=float).ravel()
        c_work.nodal_values[:] = c_new
        diff = float(np.linalg.norm(c_work.nodal_values - c_curr.nodal_values))
        step += 1
        c_prev.nodal_values[:] = c_curr.nodal_values[:]
        c_curr.nodal_values[:] = c_work.nodal_values[:]
        if diff <= target:
            break

    return np.asarray(c_curr.nodal_values, dtype=float).copy(), {
        "steps": int(step),
        "dt": float(dt),
        "initial_diff_norm": float(diff0),
        "target_diff_norm": float(target),
    }


def _tag_boundaries(mesh: Mesh, geom: Geometry, tol: float = 1.0e-12) -> None:
    cx = float(geom.center_x)
    rin = float(geom.r_in)
    mesh.tag_boundary_edges(
        {
            "bottom_active": lambda x, y: abs(y - 0.0) <= tol and abs(x - cx) <= rin + tol,
            "bottom_rest": lambda x, y: abs(y - 0.0) <= tol and not (abs(x - cx) <= rin + tol),
        }
    )


def _eval_scalar_at_point(dh: DofHandler, mesh: Mesh, field_name: str, func: Function, point: tuple[float, float]) -> float:
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
        if isinstance(func, HdivFunctionComponent):
            parent = func.parent
            comp_idx = int(func.component_index)
            local_vals = np.asarray(
                dh.mixed_element.tabulate_value(parent.field_name, float(xi), float(eta), element_id=int(elem.id)),
                dtype=float,
            )
            gdofs = np.asarray(dh.element_maps[parent.field_name][elem.id], dtype=int)
            vals = np.asarray(parent.get_nodal_values(gdofs), dtype=float).ravel()
            return float(vals @ np.asarray(local_vals[:, comp_idx], dtype=float).ravel())
        phi = dh.mixed_element.basis(field_name, float(xi), float(eta))[dh.mixed_element.slice(field_name)]
        gdofs = np.asarray(dh.element_maps[field_name][elem.id], dtype=int)
        vals = np.asarray(func.get_nodal_values(gdofs), dtype=float)
        return float(np.asarray(phi, dtype=float) @ vals)
    raise RuntimeError(f"Failed to locate point {point} in the Stoter mesh.")


def _eval_scalar_grad_at_point(dh: DofHandler, mesh: Mesh, field_name: str, func: Function, point: tuple[float, float]) -> np.ndarray:
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
        grad_ref = dh.mixed_element.grad_basis(field_name, float(xi), float(eta))[dh.mixed_element.slice(field_name), :]
        grad_phys = np.asarray(transform.map_grad_scalar(mesh, elem.id, grad_ref, (float(xi), float(eta))), dtype=float)
        gdofs = np.asarray(dh.element_maps[field_name][elem.id], dtype=int)
        coeffs = np.asarray(func.get_nodal_values(gdofs), dtype=float)
        return coeffs @ grad_phys
    raise RuntimeError(f"Failed to locate point {point} in the Stoter mesh.")


def _write_centerline_samples(outdir: Path, dh: DofHandler, mesh: Mesh, geom: Geometry, ux, uy, phi_d, c_fun, c_d_fun, K: float) -> Path:
    xs = np.full(201, float(geom.center_x))
    ys = np.linspace(0.0, float(geom.Ly), 201)
    rows = ["y_mm,c,c_D,ux_stokes,uy_stokes,ux_darcy,uy_darcy,ux_combined,uy_combined,phi_D"]
    for x, y in zip(xs, ys):
        ux_h = _eval_scalar_at_point(dh, mesh, "ux", ux, (float(x), float(y)))
        uy_h = _eval_scalar_at_point(dh, mesh, "uy", uy, (float(x), float(y)))
        grad_phi = _eval_scalar_grad_at_point(dh, mesh, "phi_D", phi_d, (float(x), float(y)))
        ux_d = -float(K) * float(grad_phi[0])
        uy_d = -float(K) * float(grad_phi[1])
        cc = float(c_fun(float(x), float(y)))
        cc_d = float(c_d_fun(float(x), float(y)))
        ux_c = cc * ux_h + cc_d * ux_d
        uy_c = cc * uy_h + cc_d * uy_d
        phi_h = _eval_scalar_at_point(dh, mesh, "phi_D", phi_d, (float(x), float(y)))
        rows.append(
            f"{float(y):.12e},{cc:.12e},{cc_d:.12e},{ux_h:.12e},{uy_h:.12e},{ux_d:.12e},{uy_d:.12e},{ux_c:.12e},{uy_c:.12e},{phi_h:.12e}"
        )
    path = outdir / "centerline.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _write_velocity_grid_samples(
    outdir: Path,
    dh: DofHandler,
    mesh: Mesh,
    geom: Geometry,
    ux,
    uy,
    phi_d,
    c_fun,
    c_d_fun,
    K: float,
    nx: int = 41,
    ny: int = 51,
) -> Path:
    xs = np.linspace(0.0, float(geom.Lx), int(nx))
    ys = np.linspace(0.0, float(geom.Ly), int(ny))
    rows = ["x_mm,y_mm,c,c_D,ux_stokes,uy_stokes,ux_darcy,uy_darcy,ux_combined,uy_combined,phi_D"]
    for y in ys:
        for x in xs:
            ux_h = _eval_scalar_at_point(dh, mesh, "ux", ux, (float(x), float(y)))
            uy_h = _eval_scalar_at_point(dh, mesh, "uy", uy, (float(x), float(y)))
            grad_phi = _eval_scalar_grad_at_point(dh, mesh, "phi_D", phi_d, (float(x), float(y)))
            phi_h = _eval_scalar_at_point(dh, mesh, "phi_D", phi_d, (float(x), float(y)))
            ux_d = -float(K) * float(grad_phi[0])
            uy_d = -float(K) * float(grad_phi[1])
            cc = float(c_fun(float(x), float(y)))
            cc_d = float(c_d_fun(float(x), float(y)))
            ux_c = cc * ux_h + cc_d * ux_d
            uy_c = cc * uy_h + cc_d * uy_d
            rows.append(
                f"{float(x):.12e},{float(y):.12e},{cc:.12e},{cc_d:.12e},{ux_h:.12e},{uy_h:.12e},{ux_d:.12e},{uy_d:.12e},{ux_c:.12e},{uy_c:.12e},{phi_h:.12e}"
            )
    path = outdir / "velocity_grid.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _write_velocity_plots(
    outdir: Path,
    velocity_grid_csv: Path,
    geom: Geometry,
    *,
    umax: float,
    clip_threshold: float = 0.5,
) -> dict[str, str]:
    import csv
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(Path(velocity_grid_csv).open("r", encoding="utf-8")))
    xs = sorted({float(row["x_mm"]) for row in rows})
    ys = sorted({float(row["y_mm"]) for row in rows})
    nx = len(xs)
    ny = len(ys)
    x_idx = {x: i for i, x in enumerate(xs)}
    y_idx = {y: j for j, y in enumerate(ys)}
    X = np.asarray(xs, dtype=float)
    Y = np.asarray(ys, dtype=float)
    UX = np.zeros((ny, nx), dtype=float)
    UY = np.zeros((ny, nx), dtype=float)
    MASK = np.zeros((ny, nx), dtype=bool)
    for row in rows:
        i = x_idx[float(row["x_mm"])]
        j = y_idx[float(row["y_mm"])]
        UX[j, i] = float(row["ux_combined"])
        UY[j, i] = float(row["uy_combined"])
        MASK[j, i] = max(float(row.get("c", 0.0)), float(row.get("c_D", 0.0))) >= float(clip_threshold)
    XX, YY = np.meshgrid(X, Y)
    SPEED = np.sqrt(UX * UX + UY * UY)
    SPEED_MASKED = np.where(MASK, SPEED, np.nan)
    UX_MASKED = np.where(MASK, UX, np.nan)
    UY_MASKED = np.where(MASK, UY, np.nan)

    def _draw_geometry(ax):
        x0 = float(geom.channel_x0)
        x1 = float(geom.channel_x1)
        y0 = float(geom.lower_interface_y)
        y1 = float(geom.upper_interface_y)
        ax.plot([x0, x1, x1, x0, x0], [0.0, 0.0, y0, y0, 0.0], color="k", lw=1.0)
        ax.plot([x0, x1, x1, x0, x0], [y1, y1, float(geom.Ly), float(geom.Ly), y1], color="k", lw=1.0)
        ax.plot([0.0, float(geom.Lx), float(geom.Lx), 0.0, 0.0], [0.0, 0.0, float(geom.Ly), float(geom.Ly), 0.0], color="0.3", lw=0.8)

    contour_path = outdir / "speed_contour_clipped.png"
    fig, ax = plt.subplots(figsize=(6.4, 8.0), constrained_layout=True)
    cf = ax.contourf(
        XX,
        YY,
        SPEED_MASKED,
        levels=np.linspace(0.0, float(umax), 21),
        cmap="viridis",
        vmin=0.0,
        vmax=float(umax),
    )
    _draw_geometry(ax)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, float(geom.Lx))
    ax.set_ylim(0.0, float(geom.Ly))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Diffuse Stoter Speed (clipped)")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("|u| [mm/s]")
    fig.savefig(contour_path, dpi=200)
    plt.close(fig)

    quiver_path = outdir / "quiver_clipped.png"
    stride_x = max(1, nx // 20)
    stride_y = max(1, ny // 25)
    fig, ax = plt.subplots(figsize=(6.4, 8.0), constrained_layout=True)
    bg = ax.contourf(
        XX,
        YY,
        SPEED_MASKED,
        levels=np.linspace(0.0, float(umax), 21),
        cmap="viridis",
        vmin=0.0,
        vmax=float(umax),
        alpha=0.9,
    )
    ax.quiver(
        XX[::stride_y, ::stride_x],
        YY[::stride_y, ::stride_x],
        UX_MASKED[::stride_y, ::stride_x],
        UY_MASKED[::stride_y, ::stride_x],
        color="white",
        angles="xy",
        scale_units="xy",
        scale=max(float(umax), 1.0) / 6.0,
        width=0.0035,
        headwidth=3.2,
        headlength=4.2,
    )
    _draw_geometry(ax)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, float(geom.Lx))
    ax.set_ylim(0.0, float(geom.Ly))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Diffuse Stoter Velocity (clipped)")
    cbar = fig.colorbar(bg, ax=ax)
    cbar.set_label("|u| [mm/s]")
    fig.savefig(quiver_path, dpi=200)
    plt.close(fig)

    return {
        "speed_contour_clipped_png": str(contour_path),
        "quiver_clipped_png": str(quiver_path),
    }


def solve_benchmark(
    *,
    nx: int,
    ny: int,
    eps: float,
    Umax: float,
    K: float,
    nu: float,
    friction_alpha: float,
    g: float,
    lsic_scale: float,
    kappa: float,
    backend: str,
    export: bool,
    outdir: Path,
    phase_field_mode: str,
    ac_dt: float,
    ac_max_steps: int,
) -> dict:
    geom = Geometry()

    nodes, elems, _, corners = structured_quad(float(geom.Lx), float(geom.Ly), nx=int(nx), ny=int(ny), poly_order=1, offset=(0.0, 0.0))
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)
    _tag_boundaries(mesh, geom)

    me = MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1, "phi_D": 1, "c": 1, "c_D": 1})
    dh = DofHandler(me, method="cg")

    h = min(float(geom.Lx) / float(nx), float(geom.Ly) / float(ny))
    tol_pin = 0.51 * h
    dh.tag_dof_by_locator(
        "pressure_pin",
        "p",
        lambda x, y: abs(x - float(geom.center_x)) <= tol_pin and abs(y - 0.0) <= tol_pin,
    )

    vel_space = FunctionSpace("velocity", ["ux", "uy"], dim=1)
    du = VectorTrialFunction(space=vel_space, dof_handler=dh)
    w = VectorTestFunction(space=vel_space, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    q = TestFunction("p", dof_handler=dh)
    dphi = TrialFunction("phi_D", dof_handler=dh)
    psi = TestFunction("phi_D", dof_handler=dh)

    u_k = VectorFunction("u_k", ["ux", "uy"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    phi_k = Function("phi_k", "phi_D", dof_handler=dh)
    c_k = Function("c_k", "c", dof_handler=dh)
    c_d_k = Function("c_d_k", "c_D", dof_handler=dh)
    u_n = VectorFunction("u_n", ["ux", "uy"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    phi_n = Function("phi_n", "phi_D", dof_handler=dh)
    for f in (u_k, p_k, phi_k, c_k, c_d_k, u_n, p_n, phi_n):
        f.nodal_values.fill(0.0)

    phase_field_mode_key = str(phase_field_mode).strip().lower().replace("-", "_")
    if phase_field_mode_key not in {"analytic", "allen_cahn", "allen-cahn"}:
        raise ValueError("phase_field_mode must be 'analytic' or 'allen_cahn'.")
    if phase_field_mode_key == "analytic":
        c_fun, grad_c_fun, abs_grad_c_fun, tau_fun, c_d_fun = _stokes_phase_fields(geom, eps=float(eps))
        c_k.set_values_from_function(c_fun)
        c_d_k.set_values_from_function(c_d_fun)
        ac_meta = {"mode": "analytic_surrogate"}
    else:
        c_vals, c_meta = _solve_allen_cahn_indicator(
            mesh=mesh,
            eps=float(eps),
            indicator_fun=_phase_field_indicator_stokes(geom),
            dt=float(ac_dt),
            max_steps=int(ac_max_steps),
            backend=backend,
        )
        c_d_vals, c_d_meta = _solve_allen_cahn_indicator(
            mesh=mesh,
            eps=float(eps),
            indicator_fun=_phase_field_indicator_darcy(geom, float(eps)),
            dt=float(ac_dt),
            max_steps=int(ac_max_steps),
            backend=backend,
        )
        if c_k.nodal_values.shape != c_vals.shape or c_d_k.nodal_values.shape != c_d_vals.shape:
            raise RuntimeError("Allen--Cahn phase-field sizes do not match flow-field coefficient sizes.")
        c_k.nodal_values[:] = c_vals
        c_d_k.nodal_values[:] = c_d_vals
        ac_meta = {
            "mode": "allen_cahn",
            "c": c_meta,
            "c_D": c_d_meta,
        }

    kappa_c = Constant(float(kappa))
    c_mod = (Constant(1.0) - kappa_c) * c_k + kappa_c
    c_d_mod = (Constant(1.0) - kappa_c) * c_d_k + kappa_c
    grad_c_expr = grad(c_k)
    abs_grad_c_expr = _sqrt(inner(grad_c_expr, grad_c_expr) + Constant(1.0e-16))
    tau_x = -grad_c_expr[1] / abs_grad_c_expr
    tau_y = grad_c_expr[0] / abs_grad_c_expr

    conv_factor = Constant(0.0)
    tau_lsic = Constant(float(lsic_scale) * float(Umax) * h / 2.0)

    conv_res = dot(dot(grad(u_k), u_k), w)
    conv_jac = dot(dot(grad(u_k), du), w) + dot(dot(grad(du), u_k), w)
    u_tang = u_k[0] * tau_x + u_k[1] * tau_y
    du_tang = du[0] * tau_x + du[1] * tau_y
    w_tang = w[0] * tau_x + w[1] * tau_y
    bjs_res = Constant(float(friction_alpha / math.sqrt(K))) * u_tang * w_tang * abs_grad_c_expr
    bjs_jac = Constant(float(friction_alpha / math.sqrt(K))) * du_tang * w_tang * abs_grad_c_expr

    residual_form = (
        conv_factor * conv_res * c_mod
        + Constant(2.0 * float(nu)) * inner(_eps(u_k), _eps(w)) * c_mod
        - p_k * div(w) * c_mod
        + q * div(u_k) * c_mod
        + tau_lsic * div(u_k) * div(w) * c_mod
        - Constant(float(g)) * phi_k * dot(w, grad_c_expr)
        + bjs_res
        + Constant(float(K)) * inner(grad(psi), grad(phi_k)) * c_d_mod
        + psi * dot(u_k, grad_c_expr)
    ) * dx()

    jacobian_form = (
        conv_factor * conv_jac * c_mod
        + Constant(2.0 * float(nu)) * inner(_eps(du), _eps(w)) * c_mod
        - dp * div(w) * c_mod
        + q * div(du) * c_mod
        + tau_lsic * div(du) * div(w) * c_mod
        - Constant(float(g)) * dphi * dot(w, grad_c_expr)
        + bjs_jac
        + Constant(float(K)) * inner(grad(psi), grad(dphi)) * c_d_mod
        + psi * dot(du, grad_c_expr)
    ) * dx()

    cx = float(geom.center_x)
    rin = float(geom.r_in)

    def uy_inflow(x, y, t=0.0):
        xx = float(x)
        rr = (xx - cx) / rin
        if abs(xx - cx) > rin:
            return 0.0
        return float(Umax) * (1.0 - rr * rr)

    zero = lambda x, y, t=0.0: 0.0

    bcs = [
        BoundaryCondition("ux", "dirichlet", "bottom_active", zero),
        BoundaryCondition("uy", "dirichlet", "bottom_active", uy_inflow),
        BoundaryCondition("ux", "dirichlet", "bottom_rest", zero),
        BoundaryCondition("uy", "dirichlet", "bottom_rest", zero),
        BoundaryCondition("p", "dirichlet", "pressure_pin", zero),
    ]
    bcs_homog = [
        BoundaryCondition("ux", "dirichlet", "bottom_active", zero),
        BoundaryCondition("uy", "dirichlet", "bottom_active", zero),
        BoundaryCondition("ux", "dirichlet", "bottom_rest", zero),
        BoundaryCondition("uy", "dirichlet", "bottom_rest", zero),
        BoundaryCondition("p", "dirichlet", "pressure_pin", zero),
    ]

    solver = NewtonSolver(
        residual_form=residual_form,
        jacobian_form=jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=1.0e-6, max_newton_iter=40, line_search=True),
        backend=backend,
    )
    bcs_now = NewtonSolver._freeze_bcs(bcs, 0.0)

    functions = [u_k, p_k, phi_k]
    prev_functions = [u_n, p_n, phi_n]
    dh.apply_bcs(bcs_now, *functions)
    dh.apply_bcs(bcs_now, *prev_functions)
    aux_funcs = {"c_k": c_k, "c_d_k": c_d_k}

    conv_factor.value = 0.0
    _, converged_stokes, iters_stokes = solver._newton_loop(functions, prev_functions, aux_funcs, bcs_now)
    if not bool(converged_stokes):
        raise RuntimeError("Stoter Stokes initialization did not converge.")
    u_n.nodal_values[:] = u_k.nodal_values[:]
    p_n.nodal_values[:] = p_k.nodal_values[:]
    phi_n.nodal_values[:] = phi_k.nodal_values[:]

    conv_factor.value = 1.0
    _, converged_ns, iters_ns = solver._newton_loop(functions, prev_functions, aux_funcs, bcs_now)
    if not bool(converged_ns):
        raise RuntimeError("Stoter Navier--Stokes/Darcy solve did not converge.")

    if export:
        export_vtk(
            str(outdir / "final_state.vtu"),
            mesh=mesh,
            dof_handler=dh,
            functions={
                "u_stokes": u_k,
                "p_stokes": p_k,
                "phi_D": phi_k,
            },
        )

    c_eval = lambda x, y: _eval_scalar_at_point(dh, mesh, "c", c_k, (float(x), float(y)))
    c_d_eval = lambda x, y: _eval_scalar_at_point(dh, mesh, "c_D", c_d_k, (float(x), float(y)))

    centerline_path = _write_centerline_samples(outdir, dh, mesh, geom, u_k[0], u_k[1], phi_k, c_eval, c_d_eval, K=float(K))
    velocity_grid_path = _write_velocity_grid_samples(
        outdir,
        dh,
        mesh,
        geom,
        u_k[0],
        u_k[1],
        phi_k,
        c_eval,
        c_d_eval,
        K=float(K),
    )
    plot_paths = _write_velocity_plots(outdir, velocity_grid_path, geom, umax=float(Umax))

    x_probe = float(geom.center_x)
    y_probe = 50.0
    ux_probe = _eval_scalar_at_point(dh, mesh, "ux", u_k[0], (x_probe, y_probe))
    uy_probe = _eval_scalar_at_point(dh, mesh, "uy", u_k[1], (x_probe, y_probe))
    grad_phi_probe = _eval_scalar_grad_at_point(dh, mesh, "phi_D", phi_k, (x_probe, y_probe))
    darcy_probe = -float(K) * np.asarray(grad_phi_probe, dtype=float)
    c_probe = float(c_eval(x_probe, y_probe))
    c_d_probe = float(c_d_eval(x_probe, y_probe))
    combined_probe = c_probe * np.asarray([ux_probe, uy_probe], dtype=float) + c_d_probe * darcy_probe

    summary = {
        "benchmark": "stoter_section_5_3_channel_porous_2d",
        "backend": str(backend),
        "nx": int(nx),
        "ny": int(ny),
        "h": float(h),
        "eps": float(eps),
        "eps_over_h": float(eps) / float(h),
        "Umax_mm_per_s": float(Umax),
        "K_mm_per_s": float(K),
        "nu_mm2_per_s": float(nu),
        "friction_alpha": float(friction_alpha),
        "gravity_g": float(g),
        "lsic_tau": float(tau_lsic.value),
        "kappa": float(kappa),
        "phase_field_mode": ac_meta["mode"],
        "phase_field_meta": ac_meta,
        "stokes_init_iterations": int(iters_stokes),
        "navier_darcy_iterations": int(iters_ns),
        "center_probe_xy_mm": [x_probe, y_probe],
        "phase_weight_probe": {"c": c_probe, "c_D": c_d_probe},
        "stokes_velocity_probe_mm_per_s": [float(ux_probe), float(uy_probe)],
        "darcy_velocity_probe_mm_per_s": [float(darcy_probe[0]), float(darcy_probe[1])],
        "combined_velocity_probe_mm_per_s": [float(combined_probe[0]), float(combined_probe[1])],
        "centerline_csv": str(centerline_path),
        "velocity_grid_csv": str(velocity_grid_path),
        **plot_paths,
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=str, default="out/stoter_channel_porous_2d")
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=120)
    ap.add_argument("--eps", type=float, default=1.0, help="Diffuse-interface length scale in mm.")
    ap.add_argument("--Umax", type=float, default=10.0, help="Maximum inflow speed in mm/s.")
    ap.add_argument("--K", type=float, default=5000.0, help="Hydraulic conductivity in mm/s.")
    ap.add_argument("--nu", type=float, default=2.927, help="Kinematic viscosity in mm^2/s.")
    ap.add_argument("--friction-alpha", type=float, default=1.0, help="BJS friction parameter alpha.")
    ap.add_argument("--g", type=float, default=9.81, help="Gravity in mm/s^2, as used in the paper notation.")
    ap.add_argument("--lsic-scale", type=float, default=1.0, help="Scale factor in tau_LSIC = scale * Umax * h / 2.")
    ap.add_argument("--kappa", type=float, default=1.0e-4, help="Fictitious-domain stabilization weight from Stoter Eq. (39)-(40).")
    ap.add_argument("--phase-field-mode", type=str, default="allen_cahn", choices=("analytic", "allen_cahn"))
    ap.add_argument("--ac-dt", type=float, default=0.05, help="Allen--Cahn preprocessing time step.")
    ap.add_argument("--ac-max-steps", type=int, default=80, help="Maximum Allen--Cahn preprocessing steps.")
    ap.add_argument("--backend", type=str, default="jit", choices=("python", "jit", "cpp"))
    ap.add_argument("--no-export", action="store_true", help="Skip VTK export.")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    summary = solve_benchmark(
        nx=int(args.nx),
        ny=int(args.ny),
        eps=float(args.eps),
        Umax=float(args.Umax),
        K=float(args.K),
        nu=float(args.nu),
        friction_alpha=float(args.friction_alpha),
        g=float(args.g),
        lsic_scale=float(args.lsic_scale),
        kappa=float(args.kappa),
        backend=str(args.backend),
        export=not bool(args.no_export),
        outdir=outdir,
        phase_field_mode=str(args.phase_field_mode),
        ac_dt=float(args.ac_dt),
        ac_max_steps=int(args.ac_max_steps),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
