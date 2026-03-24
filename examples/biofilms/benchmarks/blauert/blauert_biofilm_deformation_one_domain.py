"""
Blauert et al. (2015) biofilm deformation benchmark (one-domain model).

Goal
----
Validate the pure FSI part of the one-domain biofilm model in
`examples/utils/biofilm/one_domain.py` (no growth/detachment/damage) against the
deformation observed in the Blauert et al. supplementary video (mmc1).

Key choices (default)
---------------------
* Indicator alpha0 is initialized from a polygon extracted from the video
  (`exp_frame0_polygon_mm.csv`) and evolved by a conservative transport PDE
  (`--transport-mode pde`) with optional Cahn–Hilliard regularization.
* Porosity phi is solved (no growth sources in this FSI-only benchmark) and
  weakly penalized to phi->1 in the free-fluid region (`--gamma-phi`).
* Skeleton (u,vS) DOFs are restricted to a fixed rectangle around the biofilm so
  the extension penalties do not suppress the deformation when the fluid region
  is large (time-independent active set).

Transport PDE mode
------------------
The driver also supports a refmap mode (`--transport-mode refmap`) where
`alpha(x,t)=alpha0(x-u(x,t))` and `phi` is tied to `alpha`. This can be useful
for pure-deformation debugging, but the default is the PDE transport mode.

Outputs
-------
Writes to `out_dir`:
* `timeseries.csv`: x_front(y,t) and dx_front(y,t) for requested y-levels.
* `vtk/step=XXXX.vtu` (optional): VTK snapshots.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
except Exception:  # pragma: no cover - optional snapshot export
    matplotlib = None
    plt = None
    mtri = None

# Avoid extremely verbose Numba debug dumps if the environment enables them.
for _k in (
    "NUMBA_DEBUG",
    "NUMBA_DUMP_BYTECODE",
    "NUMBA_DUMP_IR",
    "NUMBA_DUMP_SSA",
    "NUMBA_DEBUG_ARRAY_OPT",
):
    os.environ[_k] = "0"

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node
from pycutfem.fem import transform
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import (
    LinearSolverParameters,
    LinearEqualityConstraint,
    NewtonParameters,
    NewtonSolver,
    PetscSnesNewtonSolver,
    PdasNewtonSolver,
    TimeStepperParameters,
    VIParameters,
)
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    HdivFunction,
    HdivTestFunction,
    HdivTrialFunction,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    grad,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dS as dS_measure, ds, dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from examples.utils.biofilm.adhesion import assemble_scalar
from examples.utils.biofilm.deformation_only import build_deformation_only_forms
from examples.utils.biofilm.one_domain import _sqrt, build_biofilm_one_domain_forms


def _tag_channel_boundaries(mesh: Mesh, *, L: float, H: float, tol: float = 1.0e-12) -> None:
    L = float(L)
    H = float(H)
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - L) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - H) <= tol,
        }
    )


def _mark_inactive_fields(dh: DofHandler, *field_names: str) -> None:
    tags = getattr(dh, "dof_tags", None) or {}
    inactive = set(tags.get("inactive", set()))
    for fname in field_names:
        try:
            sl = np.asarray(dh.get_field_slice(fname), dtype=int).ravel()
        except Exception:
            continue
        inactive.update(int(i) for i in sl)
    tags["inactive"] = inactive
    dh.dof_tags = tags


def _mark_inactive_dofs(dh: DofHandler, dof_ids: np.ndarray) -> None:
    tags = getattr(dh, "dof_tags", None) or {}
    inactive = set(tags.get("inactive", set()))
    for i in np.asarray(dof_ids, dtype=int).ravel():
        inactive.add(int(i))
    tags["inactive"] = inactive
    dh.dof_tags = tags


def _deepcopy_dof_tags(dh: DofHandler) -> dict[str, object]:
    tags = getattr(dh, "dof_tags", None) or {}
    out: dict[str, object] = {}
    for key, value in tags.items():
        try:
            out[str(key)] = copy.deepcopy(value)
        except Exception:
            if isinstance(value, set):
                out[str(key)] = set(value)
            else:
                out[str(key)] = value
    return out


def _set_active_fields(
    dh: DofHandler,
    *,
    active_fields: set[str],
    base_tags: dict[str, object] | None = None,
) -> None:
    tags = _deepcopy_dof_tags(dh) if base_tags is None else copy.deepcopy(base_tags)
    inactive = set(tags.get("inactive", set()))
    field_names = list(getattr(dh, "field_names", []) or [])
    for fname in field_names:
        if str(fname) in active_fields:
            continue
        try:
            sl = np.asarray(dh.get_field_slice(str(fname)), dtype=int).ravel()
        except Exception:
            continue
        inactive.update(int(i) for i in sl.tolist())
    tags["inactive"] = inactive
    dh.dof_tags = tags


def _assemble_field_integral_weights(
    problem: dict[str, object],
    *,
    test_function,
    quad_order: int,
    backend: str,
) -> np.ndarray:
    _, vec = assemble_form(
        Equation(None, test_function * dx(metadata={"q": int(quad_order)})),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=int(quad_order),
        backend=str(backend),
    )
    return np.asarray(vec, dtype=float).ravel()


def _function_to_full_vector(dh: DofHandler, func: Function) -> np.ndarray:
    full = np.zeros(int(dh.total_dofs), dtype=float)
    g = np.asarray(getattr(func, "_g_dofs", np.array([], dtype=int)), dtype=int).ravel()
    vals = np.asarray(getattr(func, "nodal_values", np.array([], dtype=float)), dtype=float).ravel()
    if g.size != vals.size:
        raise ValueError(
            f"Function '{getattr(func, 'name', '<unnamed>')}' has nodal_values size {int(vals.size)} "
            f"but _g_dofs size {int(g.size)}."
        )
    if g.size:
        full[g] = vals
    return full


def _find_named_function(funcs, template: Function) -> Function:
    t_name = str(getattr(template, "name", ""))
    t_field = str(getattr(template, "field", ""))
    for func in funcs or []:
        if str(getattr(func, "name", "")) == t_name and str(getattr(func, "field", "")) == t_field:
            return func
    raise KeyError(f"Could not find function matching name={t_name!r}, field={t_field!r}.")


def _build_vi_linear_equalities(
    *,
    support_physics: str,
    backend: str,
    qdeg: int,
    dh: DofHandler,
    alpha_test,
    alpha_k: Function,
    alpha_n: Function,
    phi_test=None,
    phi_k: Function | None = None,
    phi_n: Function | None = None,
    alpha_box_constraints: bool,
    phi_box_constraints: bool,
) -> list[LinearEqualityConstraint]:
    support_key = str(support_physics).strip().lower()
    if support_key != "internal_conversion":
        return []
    if not bool(alpha_box_constraints):
        return []

    problem = {
        "dh": dh,
        "alpha_test": alpha_test,
        "alpha_k": alpha_k,
        "alpha_n": alpha_n,
        "phi_test": phi_test,
        "phi_k": phi_k,
        "phi_n": phi_n,
    }
    alpha_weights_full = _assemble_field_integral_weights(
        problem,
        test_function=alpha_test,
        quad_order=int(qdeg),
        backend=str(backend),
    )

    def _alpha_mass_target(*, prev_funcs, **_kwargs) -> float:
        alpha_prev = _find_named_function(prev_funcs, alpha_n)
        return float(alpha_weights_full @ _function_to_full_vector(dh, alpha_prev))

    equalities: list[LinearEqualityConstraint] = [
        LinearEqualityConstraint(
            name="alpha_mass",
            weights_full=alpha_weights_full,
            target_callback=_alpha_mass_target,
            field_name="alpha",
            project_feasible=True,
        )
    ]

    if phi_test is not None and phi_k is not None and phi_n is not None and bool(phi_box_constraints):
        def _phi_biofilm_fluid_mass_weights(*, funcs, **_kwargs) -> np.ndarray:
            alpha_cur = _find_named_function(funcs, alpha_k)
            _, vec = assemble_form(
                Equation(None, alpha_cur * phi_test * dx(metadata={"q": int(qdeg)})),
                dof_handler=dh,
                bcs=[],
                quad_order=int(qdeg),
                backend=str(backend),
            )
            return np.asarray(vec, dtype=float).ravel()

        def _phi_biofilm_fluid_mass_target(*, prev_funcs, **_kwargs) -> float:
            alpha_prev = _find_named_function(prev_funcs, alpha_n)
            phi_prev = _find_named_function(prev_funcs, phi_n)
            _, weights_prev = assemble_form(
                Equation(None, alpha_prev * phi_test * dx(metadata={"q": int(qdeg)})),
                dof_handler=dh,
                bcs=[],
                quad_order=int(qdeg),
                backend=str(backend),
            )
            return float(np.asarray(weights_prev, dtype=float).ravel() @ _function_to_full_vector(dh, phi_prev))

        equalities.append(
            LinearEqualityConstraint(
                name="phi_biofilm_fluid_mass",
                weights_callback=_phi_biofilm_fluid_mass_weights,
                target_callback=_phi_biofilm_fluid_mass_target,
                field_name="phi",
                project_feasible=True,
            )
        )
    return equalities


def _restrict_skeleton_dofs_to_box(
    dh: DofHandler,
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> tuple[int, int]:
    """
    Restrict (u,vS) unknowns to a fixed axis-aligned rectangle in physical space.

    Outside the box, the DOFs are marked inactive (time-independent active set).
    """
    x0 = float(x0)
    x1 = float(x1)
    y0 = float(y0)
    y1 = float(y1)
    if not (np.isfinite(x0) and np.isfinite(x1) and np.isfinite(y0) and np.isfinite(y1)):
        raise ValueError("Box bounds must be finite.")
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid box: x0={x0}, x1={x1}, y0={y0}, y1={y1}.")

    u_xy = np.asarray(dh.get_dof_coords("u_x"), dtype=float)
    keep = (u_xy[:, 0] >= x0) & (u_xy[:, 0] <= x1) & (u_xy[:, 1] >= y0) & (u_xy[:, 1] <= y1)
    n_keep = int(np.sum(keep))
    n_tot = int(keep.size)
    if n_keep < max(10, int(0.01 * float(n_tot))):
        raise RuntimeError(
            f"restrict-skeleton box keeps too few DOFs: {n_keep}/{n_tot}. Increase the box or disable restriction."
        )

    for fname in ("u_x", "u_y", "vS_x", "vS_y"):
        sl = np.asarray(dh.get_field_slice(fname), dtype=int).ravel()
        if sl.size != keep.size:
            raise RuntimeError(f"Unexpected DOF count mismatch for {fname}: slice={sl.size}, coords={keep.size}")
        _mark_inactive_dofs(dh, sl[~keep])
    return n_keep, n_tot


def _smooth_step(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(np.asarray(z, dtype=float)))


def _alpha_profile_signed_distance(alpha_vals: np.ndarray, eps: float, *, clip: float = 1.0e-6) -> np.ndarray:
    """
    Approximate signed distance from a diffuse alpha profile assuming the
    standard tanh transition alpha ~= 0.5 * (1 - tanh(d / eps)).
    """
    ee = float(eps)
    if not np.isfinite(ee) or ee <= 0.0:
        raise ValueError(f"eps must be positive to recover a signed distance; got {eps!r}.")
    cc = float(clip)
    aa = np.clip(np.asarray(alpha_vals, dtype=float), cc, 1.0 - cc)
    return -ee * np.arctanh(2.0 * aa - 1.0)


def _interface_band_reg_weight(alpha_vals: np.ndarray, *, eps: float, band_factor: float, floor: float) -> np.ndarray:
    bf = float(band_factor)
    ff = float(floor)
    if not np.isfinite(bf) or bf <= 0.0:
        return np.ones_like(np.asarray(alpha_vals, dtype=float))
    if not np.isfinite(ff) or ff < 0.0 or ff > 1.0:
        raise ValueError(f"interface regularization floor must be in [0, 1]; got {floor!r}.")
    band_halfwidth = bf * float(eps)
    dist = _alpha_profile_signed_distance(alpha_vals, float(eps))
    inside = np.abs(dist) <= band_halfwidth
    return np.where(inside, 1.0, ff)


def _as_scalar_expr(val):
    if hasattr(val, "dim"):
        return val
    return Constant(float(val))


def _named_scalar_expr(name: str, val):
    expr = _as_scalar_expr(val)
    try:
        expr._jit_name = str(name)
    except Exception:
        pass
    return expr


def _cosine_ramp_value(t_now: float, ramp_time: float) -> float:
    tt = float(t_now)
    tr = float(ramp_time)
    if not np.isfinite(tr) or tr <= 0.0:
        return 1.0
    if tt <= 0.0:
        return 0.0
    if tt >= tr:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * tt / max(1.0e-12, tr)))


def _cosine_blend_value(
    t_now: float,
    *,
    start_time: float,
    end_time: float,
    start_value: float,
    end_value: float,
) -> float:
    tt = float(t_now)
    t0 = float(start_time)
    t1 = float(end_time)
    v0 = float(start_value)
    v1 = float(end_value)
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return v0
    if tt <= t0:
        return v0
    if tt >= t1:
        return v1
    xi = (tt - t0) / max(1.0e-12, (t1 - t0))
    w = 0.5 * (1.0 - math.cos(math.pi * xi))
    return (1.0 - w) * v0 + w * v1


def _lagged_diffuse_interface_shear_traction(
    *,
    v_lag,
    alpha_lag,
    mu_f: float,
    scale: float = 1.0,
    eta_n: float = 1.0e-12,
    topweight: bool = False,
):
    """
    Return a lagged diffuse-interface tangential traction proxy.

    This approximates the fluid shear traction on the alpha=1/2 contour by
    localizing the tangential projection of 2 mu_f eps(v) n with |grad(alpha)|.
    The result is used as an equal-and-opposite traction pair in the reduced
    one-domain benchmark via the existing traction hook.
    """
    grad_alpha = grad(alpha_lag)
    grad_norm = _sqrt(grad_alpha[0] * grad_alpha[0] + grad_alpha[1] * grad_alpha[1] + Constant(float(eta_n)))
    n_if = (grad_alpha[0] / grad_norm, grad_alpha[1] / grad_norm)
    t_if = (-n_if[1], n_if[0])
    dvx = grad(v_lag[0])
    dvy = grad(v_lag[1])
    eps_xx = dvx[0]
    eps_xy = Constant(0.5) * (dvx[1] + dvy[0])
    eps_yy = dvy[1]
    tr_x = Constant(2.0 * float(mu_f)) * (eps_xx * n_if[0] + eps_xy * n_if[1])
    tr_y = Constant(2.0 * float(mu_f)) * (eps_xy * n_if[0] + eps_yy * n_if[1])
    scale_c = _as_scalar_expr(scale)
    tau_t = scale_c * (tr_x * t_if[0] + tr_y * t_if[1])
    if bool(topweight):
        tau_t = _sqrt(n_if[1] * n_if[1] + Constant(float(eta_n))) * tau_t
    g_t = (tau_t * t_if[0], tau_t * t_if[1])
    return g_t, grad_norm


def _lagged_diffuse_interface_stress_traction(
    *,
    v_lag,
    p_lag,
    alpha_lag,
    mu_f: float,
    scale: float = 1.0,
    eta_n: float = 1.0e-12,
):
    """
    Return the lagged diffuse-interface fluid traction proxy

        t = (-p I + 2 mu_f eps(v)) n_if

    localized by |grad(alpha)|. This carries both the normal compression and the
    tangential shear transmitted by the surrounding fluid and is more appropriate
    than a tangential-only proxy for the Blauert compression benchmark.
    """
    grad_alpha = grad(alpha_lag)
    grad_norm = _sqrt(grad_alpha[0] * grad_alpha[0] + grad_alpha[1] * grad_alpha[1] + Constant(float(eta_n)))
    n_if = (grad_alpha[0] / grad_norm, grad_alpha[1] / grad_norm)
    dvx = grad(v_lag[0])
    dvy = grad(v_lag[1])
    eps_xx = dvx[0]
    eps_xy = Constant(0.5) * (dvx[1] + dvy[0])
    eps_yy = dvy[1]
    tr_x = -p_lag * n_if[0] + Constant(2.0 * float(mu_f)) * (eps_xx * n_if[0] + eps_xy * n_if[1])
    tr_y = -p_lag * n_if[1] + Constant(2.0 * float(mu_f)) * (eps_xy * n_if[0] + eps_yy * n_if[1])
    scale_c = _as_scalar_expr(scale)
    g_t = (scale_c * tr_x, scale_c * tr_y)
    return g_t, grad_norm


def _lagged_diffuse_interface_pressure_traction(
    *,
    p_lag,
    alpha_lag,
    scale: float = 1.0,
    eta_n: float = 1.0e-12,
    xweight: bool = False,
    upstream_only: bool = False,
):
    """
    Return the lagged normal-pressure part of the diffuse-interface traction.

    This isolates the `-p^n n` contribution so the Blauert benchmark can blend
    the paper's Poiseuille tangential proxy with an independently scaled normal
    compression term.
    """
    grad_alpha = grad(alpha_lag)
    grad_norm = _sqrt(grad_alpha[0] * grad_alpha[0] + grad_alpha[1] * grad_alpha[1] + Constant(float(eta_n)))
    n_if = (grad_alpha[0] / grad_norm, grad_alpha[1] / grad_norm)
    scale_c = _as_scalar_expr(scale)
    weight = Constant(1.0)
    if bool(upstream_only):
        nx_pos = Constant(0.5) * (n_if[0] + _sqrt(n_if[0] * n_if[0] + Constant(float(eta_n))))
        weight = nx_pos
    elif bool(xweight):
        weight = _sqrt(n_if[0] * n_if[0] + Constant(float(eta_n)))
    g_t = (-scale_c * weight * p_lag * n_if[0], -scale_c * weight * p_lag * n_if[1])
    return g_t, grad_norm


def _lagged_diffuse_interface_normal_stress_traction(
    *,
    v_lag,
    p_lag,
    alpha_lag,
    mu_f: float,
    scale: float = 1.0,
    eta_n: float = 1.0e-12,
    xweight: bool = False,
    topweight: bool = False,
    topbias: float = 0.0,
    frontbias: float = 0.0,
    bottomskew: float = 0.0,
    channel_height: float | None = None,
    upstream_only: bool = False,
):
    """
    Return the lagged normal-stress part of the diffuse-interface traction.

    This keeps only the normal projection of the full lagged traction

        (-p^n I + 2 mu_f eps(v^n)) n_if

    so the Blauert benchmark can combine the paper's Poiseuille tangential load
    with independently scaled lagged normal compression.
    """
    grad_alpha = grad(alpha_lag)
    grad_norm = _sqrt(grad_alpha[0] * grad_alpha[0] + grad_alpha[1] * grad_alpha[1] + Constant(float(eta_n)))
    n_if = (grad_alpha[0] / grad_norm, grad_alpha[1] / grad_norm)
    dvx = grad(v_lag[0])
    dvy = grad(v_lag[1])
    eps_xx = dvx[0]
    eps_xy = Constant(0.5) * (dvx[1] + dvy[0])
    eps_yy = dvy[1]
    tr_x = -p_lag * n_if[0] + Constant(2.0 * float(mu_f)) * (eps_xx * n_if[0] + eps_xy * n_if[1])
    tr_y = -p_lag * n_if[1] + Constant(2.0 * float(mu_f)) * (eps_xy * n_if[0] + eps_yy * n_if[1])
    tau_n = tr_x * n_if[0] + tr_y * n_if[1]
    weight = Constant(1.0)
    nx_pos = Constant(0.5) * (n_if[0] + _sqrt(n_if[0] * n_if[0] + Constant(float(eta_n))))
    ny_mag = _sqrt(n_if[1] * n_if[1] + Constant(float(eta_n)))
    if bool(upstream_only):
        # Focus the added compression on the inflow-facing side of the patch.
        weight = weight * nx_pos
    else:
        frontbias_c = _as_scalar_expr(frontbias)
        # Blend smoothly between uniform loading and the inflow-facing weight.
        weight = weight * (Constant(1.0) - frontbias_c + frontbias_c * nx_pos)
    if bool(xweight):
        # Streamwise-facing parts of the interface should feel most of the
        # compressive load; nearly horizontal segments should contribute less.
        weight = weight * _sqrt(n_if[0] * n_if[0] + Constant(float(eta_n)))
    if bool(topweight):
        # Upper/lower flanks are under-driven on Benchmark 6; weighting by |n_y|
        # shifts the extra compression away from the near-vertical mid-front.
        weight = weight * ny_mag
    elif abs(float(topbias)) > 0.0:
        # Blend smoothly between uniform loading and the hard |n_y|-weighted
        # flank emphasis instead of switching directly to the corner-heavy mode.
        weight = weight * (Constant(1.0 - float(topbias)) + Constant(float(topbias)) * ny_mag)
    H = float(channel_height) if channel_height is not None else float("nan")
    if not np.isfinite(H) or H <= 0.0:
        raise ValueError("channel_height must be positive for normal-stress height weighting.")
    bottomskew_c = _as_scalar_expr(bottomskew)
    # Redistribute the added compression toward the lower part of the patch
    # while keeping the height-average weight equal to one.
    y_weight = Analytic(
        lambda x, y, H=H: 1.0 - 2.0 * np.asarray(y, dtype=float) / H,
        degree=1,
    )
    weight = weight * (Constant(1.0) + bottomskew_c * y_weight)
    scale_c = _as_scalar_expr(scale)
    g_t = (scale_c * weight * tau_n * n_if[0], scale_c * weight * tau_n * n_if[1])
    return g_t, grad_norm


def _poiseuille_diffuse_interface_shear_traction(
    *,
    alpha_lag,
    H: float,
    u_max: float,
    mu_f: float,
    scale: float = 1.0,
    eta_n: float = 1.0e-12,
    topweight: bool = False,
):
    grad_alpha = grad(alpha_lag)
    grad_norm = _sqrt(grad_alpha[0] * grad_alpha[0] + grad_alpha[1] * grad_alpha[1] + Constant(float(eta_n)))
    n_if = (grad_alpha[0] / grad_norm, grad_alpha[1] / grad_norm)
    t_if = (-n_if[1], n_if[0])
    scale_c = _as_scalar_expr(scale)
    tau_bg = Analytic(
        lambda x, y, _H=float(H), _u=float(u_max), _mu=float(mu_f): _mu
        * (4.0 * _u / _H)
        * (1.0 - 2.0 * np.asarray(y, dtype=float) / _H),
        degree=2,
    )
    tau_t = scale_c * tau_bg
    if bool(topweight):
        tau_t = _sqrt(n_if[1] * n_if[1] + Constant(float(eta_n))) * tau_t
    g_t = (tau_t * t_if[0], tau_t * t_if[1])
    return g_t, grad_norm


def _read_polygon_mm_csv(path: str) -> np.ndarray:
    arr = np.genfromtxt(str(path), delimiter=",", skip_header=1, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    arr = np.asarray(arr, dtype=float)
    if arr.shape[1] < 2:
        raise ValueError(f"Polygon CSV must have at least 2 columns; got shape={arr.shape}")
    pts = arr[:, :2]
    if pts.shape[0] < 3:
        raise ValueError(f"Polygon must have at least 3 points; got {pts.shape[0]}")
    # Ensure closed.
    if not np.allclose(pts[0], pts[-1], rtol=0.0, atol=1.0e-12):
        pts = np.vstack([pts, pts[0]])
    return pts


def _signed_distance_polygon(x: np.ndarray, y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """
    Signed distance to a simple polygon (negative inside).

    Parameters
    ----------
    x, y
        Query point coordinates (broadcastable to same shape).
    poly
        Polygon vertices, shape (M,2), closed (poly[0]==poly[-1]).
    """
    xq = np.asarray(x, dtype=float).ravel()
    yq = np.asarray(y, dtype=float).ravel()
    pts = np.asarray(poly, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] < 4:
        raise ValueError("poly must be closed with shape (M,2), M>=4")

    # Point-in-polygon via ray casting.
    inside = np.zeros_like(xq, dtype=bool)
    xi = pts[:-1, 0]
    yi = pts[:-1, 1]
    xj = pts[1:, 0]
    yj = pts[1:, 1]
    for a_x, a_y, b_x, b_y in zip(xi, yi, xj, yj):
        cond = ((a_y > yq) != (b_y > yq))
        denom = (b_y - a_y) if abs(b_y - a_y) > 1.0e-30 else 1.0e-30
        x_at_y = (b_x - a_x) * (yq - a_y) / denom + a_x
        inside ^= cond & (xq < x_at_y)

    # Distance to segments.
    d2 = np.full_like(xq, float("inf"), dtype=float)
    for a_x, a_y, b_x, b_y in zip(xi, yi, xj, yj):
        abx = float(b_x - a_x)
        aby = float(b_y - a_y)
        ab2 = abx * abx + aby * aby
        if ab2 <= 1.0e-30:
            dx0 = xq - float(a_x)
            dy0 = yq - float(a_y)
            d2 = np.minimum(d2, dx0 * dx0 + dy0 * dy0)
            continue
        apx = xq - float(a_x)
        apy = yq - float(a_y)
        t = (apx * abx + apy * aby) / ab2
        t = np.clip(t, 0.0, 1.0)
        cx = float(a_x) + t * abx
        cy = float(a_y) + t * aby
        dx0 = xq - cx
        dy0 = yq - cy
        d2 = np.minimum(d2, dx0 * dx0 + dy0 * dy0)
    d = np.sqrt(np.maximum(d2, 0.0))
    d[inside] *= -1.0
    return d.reshape(np.asarray(x, dtype=float).shape)


def _build_coord_lookup(coords: np.ndarray, *, ndigits: int = 12) -> dict[tuple[float, float], int]:
    out: dict[tuple[float, float], int] = {}
    for i, xy in enumerate(np.asarray(coords, dtype=float)):
        out[(round(float(xy[0]), ndigits), round(float(xy[1]), ndigits))] = int(i)
    return out


def _nearest_y_levels(y_all: np.ndarray, targets: list[float]) -> list[float]:
    y = np.unique(np.asarray(y_all, dtype=float))
    if y.size == 0:
        return [float(t) for t in targets]
    y = np.sort(np.asarray(y, dtype=float))
    out: list[float] = []
    used: set[int] = set()
    tol = 1.0e-14
    min_allowed = float("-inf")
    for t in [float(v) for v in targets]:
        cand = np.array([i for i in range(y.size) if (i not in used) and (y[i] >= min_allowed - tol)], dtype=int)
        if cand.size == 0:
            cand = np.arange(0, y.size, dtype=int)
        diffs = np.abs(y[cand] - float(t))
        jmin = float(np.min(diffs))
        best = cand[diffs == jmin]
        j = int(best[0])
        out.append(float(y[j]))
        used.add(int(j))
        min_allowed = float(y[j]) + tol
    return out


def _x_alpha_half_on_y_line_mode(
    alpha_xy: np.ndarray,
    alpha_vals: np.ndarray,
    *,
    y_line: float,
    mode: str,
    alpha_half: float = 0.5,
) -> float:
    """
    Intersection x of the alpha=alpha_half contour with the horizontal line y=y_line.

    mode:
      - "rightmost": downstream edge (max x)
      - "leftmost":  upstream edge (min x)
    """
    mode = str(mode).strip().lower()
    if mode not in {"rightmost", "leftmost"}:
        raise ValueError(f"Unknown intersection mode {mode!r}. Use 'rightmost' or 'leftmost'.")

    xy = np.asarray(alpha_xy, dtype=float)
    a = np.asarray(alpha_vals, dtype=float).ravel()
    if xy.ndim != 2 or xy.shape[1] < 2 or a.shape[0] != xy.shape[0]:
        return float("nan")

    mask = np.abs(xy[:, 1] - float(y_line)) <= 1.0e-14
    if not np.any(mask):
        return float("nan")

    x = np.asarray(xy[mask, 0], dtype=float).ravel()
    av = np.asarray(a[mask], dtype=float).ravel()
    if x.size < 2:
        return float("nan")

    order = np.argsort(x)
    x = x[order]
    av = av[order]

    above = av >= float(alpha_half)
    idx = np.nonzero(above)[0]
    if idx.size == 0:
        return float("nan")
    i0 = int(idx[0])
    i1 = int(idx[-1])

    if mode == "rightmost":
        if i1 >= x.size - 1:
            return float(x[-1])
        a0, a1 = float(av[i1]), float(av[i1 + 1])
        x0, x1 = float(x[i1]), float(x[i1 + 1])
    else:
        if i0 <= 0:
            return float(x[0])
        a0, a1 = float(av[i0 - 1]), float(av[i0])
        x0, x1 = float(x[i0 - 1]), float(x[i0])

    da = a1 - a0
    if abs(da) <= 1.0e-16:
        return float(x1 if mode == "leftmost" else x0)
    t = (float(alpha_half) - a0) / da
    return float(x0 + t * (x1 - x0))


def _alpha_half_intervals_on_y_line(
    alpha_xy: np.ndarray,
    alpha_vals: np.ndarray,
    *,
    y_line: float,
    alpha_half: float = 0.5,
) -> list[tuple[float, float]]:
    """
    Return all (x_L, x_R) intervals where alpha >= alpha_half on the line y=y_line.

    This is robust to multiple disconnected segments (e.g. due to pores / detached blobs).
    """
    xy = np.asarray(alpha_xy, dtype=float)
    a = np.asarray(alpha_vals, dtype=float).ravel()
    if xy.ndim != 2 or xy.shape[1] < 2 or a.shape[0] != xy.shape[0]:
        return []

    mask = np.abs(xy[:, 1] - float(y_line)) <= 1.0e-14
    if not np.any(mask):
        return []

    x = np.asarray(xy[mask, 0], dtype=float).ravel()
    av = np.asarray(a[mask], dtype=float).ravel()
    if x.size < 2:
        return []

    order = np.argsort(x)
    x = x[order]
    av = av[order]

    above = av >= float(alpha_half)
    if not np.any(above):
        return []

    # Segment starts/ends in the boolean array.
    starts = np.nonzero(above & np.r_[True, ~above[:-1]])[0]
    ends = np.nonzero(above & np.r_[~above[1:], True])[0]

    out: list[tuple[float, float]] = []
    for s, e in zip(starts, ends):
        s = int(s)
        e = int(e)

        # Left intersection x_L.
        if s <= 0:
            xL = float(x[0])
        else:
            a0, a1 = float(av[s - 1]), float(av[s])
            x0, x1 = float(x[s - 1]), float(x[s])
            da = a1 - a0
            if abs(da) <= 1.0e-16:
                xL = float(x1)
            else:
                t = (float(alpha_half) - a0) / da
                xL = float(x0 + t * (x1 - x0))

        # Right intersection x_R.
        if e >= x.size - 1:
            xR = float(x[-1])
        else:
            a0, a1 = float(av[e]), float(av[e + 1])
            x0, x1 = float(x[e]), float(x[e + 1])
            da = a1 - a0
            if abs(da) <= 1.0e-16:
                xR = float(x0)
            else:
                t = (float(alpha_half) - a0) / da
                xR = float(x0 + t * (x1 - x0))

        if math.isfinite(xL) and math.isfinite(xR) and xR >= xL:
            out.append((float(xL), float(xR)))

    out.sort(key=lambda p: p[0])
    return out


def _x_from_interval(xL: float, xR: float, *, mode: str, q: float) -> float:
    mode = str(mode).strip().lower()
    if mode not in {"leftmost", "rightmost"}:
        raise ValueError(f"Unknown mode {mode!r}. Use 'leftmost' or 'rightmost'.")
    q = float(np.clip(float(q), 0.0, 1.0))
    if mode == "leftmost":
        return float(xL + q * (xR - xL))
    return float(xR - q * (xR - xL))


def _x_alpha_half_track_on_y_line(
    alpha_xy: np.ndarray,
    alpha_vals: np.ndarray,
    *,
    y_line: float,
    mode: str,
    q: float,
    prev_x: float | None,
    alpha_half: float = 0.5,
) -> float:
    intervals = _alpha_half_intervals_on_y_line(alpha_xy, alpha_vals, y_line=float(y_line), alpha_half=float(alpha_half))
    if not intervals:
        return float("nan")
    cands = np.asarray([_x_from_interval(xL, xR, mode=str(mode), q=float(q)) for xL, xR in intervals], dtype=float)
    if cands.size == 0 or not np.any(np.isfinite(cands)):
        return float("nan")
    if prev_x is not None and math.isfinite(float(prev_x)):
        j = int(np.nanargmin(np.abs(cands - float(prev_x))))
        return float(cands[j])
    # Fallback: pick the extreme consistent with the mode.
    return float(np.nanmin(cands) if str(mode).strip().lower() == "leftmost" else np.nanmax(cands))


def _x_alpha_half_quantile_on_y_line(
    alpha_xy: np.ndarray,
    alpha_vals: np.ndarray,
    *,
    y_line: float,
    mode: str,
    q: float,
    alpha_half: float = 0.5,
) -> float:
    """
    Track an interior x-position based on the alpha=alpha_half interval on y=y_line.

    Let (x_L, x_R) be the left/right intersections of the alpha=alpha_half contour.
    Then:
      - mode="leftmost":  x = x_L + q (x_R - x_L)
      - mode="rightmost": x = x_R - q (x_R - x_L)

    With q=0 this reduces to the boundary intersection.
    """
    mode = str(mode).strip().lower()
    q = float(np.clip(float(q), 0.0, 1.0))
    xL = _x_alpha_half_on_y_line_mode(alpha_xy, alpha_vals, y_line=float(y_line), mode="leftmost", alpha_half=float(alpha_half))
    xR = _x_alpha_half_on_y_line_mode(alpha_xy, alpha_vals, y_line=float(y_line), mode="rightmost", alpha_half=float(alpha_half))
    if not (math.isfinite(xL) and math.isfinite(xR)):
        return float("nan")
    if mode == "leftmost":
        return float(xL + q * (xR - xL))
    return float(xR - q * (xR - xL))


def _segment_intersections_x(points: np.ndarray, y_sample: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return np.empty((0,), dtype=float)
    out: list[float] = []
    for i in range(pts.shape[0] - 1):
        x0, y0 = float(pts[i, 0]), float(pts[i, 1])
        x1, y1 = float(pts[i + 1, 0]), float(pts[i + 1, 1])
        if not ((y0 <= y_sample <= y1) or (y1 <= y_sample <= y0)):
            continue
        dy = y1 - y0
        if abs(dy) <= 1.0e-14:
            out.extend([x0, x1])
            continue
        tau = (float(y_sample) - y0) / dy
        if -1.0e-12 <= tau <= 1.0 + 1.0e-12:
            out.append(x0 + tau * (x1 - x0))
    return np.asarray(out, dtype=float)


def _x_front_global_quantile(
    alpha_xy: np.ndarray,
    alpha_vals: np.ndarray,
    *,
    q: float,
    alpha_half: float = 0.5,
) -> float:
    """
    Global upstream front location based on a quantile of row-wise leftmost
    alpha=alpha_half contour intersections.

    The video extractor uses a robust left-quantile over the segmented biofilm
    area, not a quantile over raw contour points. Approximating that from the
    transported alpha=1/2 contour via row-wise leftmost intersections avoids a
    single tiny tail dominating the "global" metric at late times.
    """
    q = float(np.clip(float(q), 0.0, 1.0))
    contours = _alpha_half_contours(alpha_xy, alpha_vals, alpha_half=float(alpha_half))
    contour_pts = [
        np.asarray(pts, dtype=float)
        for pts in contours
        if np.asarray(pts, dtype=float).ndim == 2 and np.asarray(pts, dtype=float).shape[1] >= 2
    ]
    if contour_pts:
        pts_all = np.vstack(contour_pts)
        ys = np.asarray(pts_all[:, 1], dtype=float).ravel()
        ys = ys[np.isfinite(ys)]
        if ys.size > 1:
            y_min = float(np.min(ys))
            y_max = float(np.max(ys))
            if y_max > y_min:
                row_fronts: list[float] = []
                y_samples = np.linspace(y_min, y_max, 400, dtype=float)
                for yy in y_samples:
                    xs_all: list[float] = []
                    for pts in contour_pts:
                        xs = _segment_intersections_x(pts, float(yy))
                        if xs.size:
                            xs_all.extend(float(v) for v in xs if np.isfinite(v))
                    if xs_all:
                        row_fronts.append(float(np.min(np.asarray(xs_all, dtype=float))))
                if row_fronts:
                    return float(np.quantile(np.asarray(row_fronts, dtype=float), q))

        contour_xs = [np.asarray(pts, dtype=float)[:, 0].ravel() for pts in contour_pts]
        xs = np.concatenate(contour_xs)
        xs = xs[np.isfinite(xs)]
        if xs.size > 0:
            return float(np.quantile(xs, q))

    xy = np.asarray(alpha_xy, dtype=float)
    a = np.asarray(alpha_vals, dtype=float).ravel()
    if xy.ndim != 2 or xy.shape[1] < 2 or a.shape[0] != xy.shape[0]:
        return float("nan")
    inside = a >= float(alpha_half)
    if not np.any(inside):
        return float("nan")
    xs = np.asarray(xy[inside, 0], dtype=float).ravel()
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return float("nan")
    return float(np.quantile(xs, q))


def _alpha_half_contours(alpha_xy: np.ndarray, alpha_vals: np.ndarray, *, alpha_half: float = 0.5) -> list[np.ndarray]:
    if plt is None or mtri is None:
        return []
    xy = np.asarray(alpha_xy, dtype=float)
    vals = np.asarray(alpha_vals, dtype=float).ravel()
    if xy.ndim != 2 or xy.shape[1] < 2 or vals.size != xy.shape[0]:
        return []
    # Hanging-node/refined mixed spaces can carry repeated alpha coordinates.
    # Deduplicate before triangulation so matplotlib's contour builder does not
    # bail out on zero-area simplices or coincident sites.
    if xy.shape[0] >= 2:
        xy_key = np.round(xy[:, :2], decimals=14)
        unique_xy, inverse, counts = np.unique(xy_key, axis=0, return_inverse=True, return_counts=True)
        if unique_xy.shape[0] != xy.shape[0]:
            vals_acc = np.zeros(unique_xy.shape[0], dtype=float)
            np.add.at(vals_acc, inverse, vals)
            xy = np.asarray(unique_xy, dtype=float)
            vals = vals_acc / np.maximum(counts.astype(float), 1.0)
    finite = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1]) & np.isfinite(vals)
    if not np.any(finite):
        return []
    xy = xy[finite, :2]
    vals = vals[finite]
    if xy.shape[0] < 3:
        return []

    def _collect_contours(cs) -> list[np.ndarray]:
        contours: list[np.ndarray] = []
        collections = getattr(cs, "collections", None)
        if collections is not None:
            for coll in collections:
                for path in coll.get_paths():
                    verts = np.asarray(path.vertices, dtype=float)
                    if verts.ndim == 2 and verts.shape[0] >= 2:
                        contours.append(verts.copy())
            if contours:
                return contours
        allsegs = getattr(cs, "allsegs", None)
        if allsegs is not None:
            for level_segs in allsegs:
                for seg in level_segs:
                    verts = np.asarray(seg, dtype=float)
                    if verts.ndim == 2 and verts.shape[0] >= 2:
                        contours.append(verts.copy())
        return contours

    try:
        triang = mtri.Triangulation(xy[:, 0], xy[:, 1])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cs = ax.tricontour(triang, vals, levels=[float(alpha_half)])
        contours = _collect_contours(cs)
        plt.close(fig)
        if contours:
            return contours
    except Exception:
        pass

    try:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        x_min = float(np.min(xy[:, 0]))
        x_max = float(np.max(xy[:, 0]))
        y_min = float(np.min(xy[:, 1]))
        y_max = float(np.max(xy[:, 1]))
        if not (np.isfinite(x_min) and np.isfinite(x_max) and np.isfinite(y_min) and np.isfinite(y_max)):
            return []
        if not (x_max > x_min and y_max > y_min):
            return []

        nx = int(np.clip(np.ceil(np.sqrt(float(xy.shape[0]))) * 8.0, 96.0, 512.0))
        aspect = (y_max - y_min) / max(x_max - x_min, 1.0e-16)
        ny = int(np.clip(np.ceil(float(nx) * float(aspect)), 64.0, 512.0))

        gx = np.linspace(x_min, x_max, nx, dtype=float)
        gy = np.linspace(y_min, y_max, ny, dtype=float)
        XX, YY = np.meshgrid(gx, gy)
        interp_lin = LinearNDInterpolator(xy[:, :2], vals, fill_value=np.nan)
        ZZ = np.asarray(interp_lin(XX, YY), dtype=float)
        if np.isnan(ZZ).any():
            interp_nn = NearestNDInterpolator(xy[:, :2], vals)
            ZZ_nn = np.asarray(interp_nn(XX, YY), dtype=float)
            ZZ = np.where(np.isfinite(ZZ), ZZ, ZZ_nn)
        if not np.isfinite(ZZ).any():
            return []

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cs = ax.contour(XX, YY, ZZ, levels=[float(alpha_half)])
        contours = _collect_contours(cs)
        plt.close(fig)
        return contours
    except Exception:
        return []


def _find_element_containing_point(mesh: Mesh, point: np.ndarray) -> int:
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
        if -1.0001 <= xi <= 1.0001 and -1.0001 <= eta <= 1.0001:
            return int(elem.id)
    raise ValueError(f"Point {tuple(point)} not found in mesh.")


def _eval_scalar_with_grad(
    dh: DofHandler,
    mesh: Mesh,
    f_scalar: Function,
    point: tuple[float, float],
) -> tuple[float, np.ndarray]:
    xy = np.asarray(point, dtype=float)
    eid = _find_element_containing_point(mesh, xy)
    xi, eta = transform.inverse_mapping(mesh, eid, xy)
    me = dh.mixed_element
    local_phi = me.basis(f_scalar.field_name, float(xi), float(eta))[me.slice(f_scalar.field_name)]
    local_grad_ref = me.grad_basis(f_scalar.field_name, float(xi), float(eta))[me.slice(f_scalar.field_name)]
    local_grad = transform.map_grad_scalar(mesh, eid, local_grad_ref, (float(xi), float(eta)))
    gdofs = dh.element_maps[f_scalar.field_name][eid]
    vals = f_scalar.get_nodal_values(gdofs)
    return float(local_phi @ vals), np.asarray(vals, dtype=float) @ np.asarray(local_grad, dtype=float)


def _eval_vector_with_grad(
    dh: DofHandler,
    mesh: Mesh,
    f_vec: VectorFunction,
    point: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    vals = []
    grads = []
    for comp in f_vec.components:
        vv, gg = _eval_scalar_with_grad(dh, mesh, comp, point)
        vals.append(vv)
        grads.append(gg)
    return np.asarray(vals, dtype=float), np.vstack(grads)


def _interface_stress_diagnostics(
    *,
    dh: DofHandler,
    mesh: Mesh,
    alpha_xy: np.ndarray,
    alpha_fn: Function,
    v_fn: VectorFunction | HdivFunction,
    p_fn: Function,
    mu_f: float,
    alpha_half: float = 0.5,
    max_points: int = 300,
) -> dict[str, float]:
    nan = float("nan")
    if not hasattr(v_fn, "components"):
        return {
            "n_samples": 0.0,
            "p_mean_pa": nan,
            "p_max_pa": nan,
            "sigma_n_mean_pa": nan,
            "sigma_n_min_pa": nan,
            "sigma_comp_mean_pa": nan,
            "tau_t_abs_mean_pa": nan,
            "tau_t_abs_max_pa": nan,
            "front_samples": 0.0,
            "front_p_mean_pa": nan,
            "front_sigma_n_mean_pa": nan,
            "front_sigma_comp_mean_pa": nan,
            "front_tau_t_abs_mean_pa": nan,
            "top_samples": 0.0,
            "top_p_mean_pa": nan,
            "top_sigma_n_mean_pa": nan,
            "top_tau_t_mean_pa": nan,
            "top_tau_t_abs_mean_pa": nan,
            "top_mu_du_dh_mean_pa": nan,
            "top_tau_over_mu_du_dh_mean": nan,
        }

    contours = _alpha_half_contours(alpha_xy, alpha_fn.nodal_values, alpha_half=float(alpha_half))
    if not contours:
        return {
            "n_samples": 0.0,
            "p_mean_pa": nan,
            "p_max_pa": nan,
            "sigma_n_mean_pa": nan,
            "sigma_n_min_pa": nan,
            "sigma_comp_mean_pa": nan,
            "tau_t_abs_mean_pa": nan,
            "tau_t_abs_max_pa": nan,
            "front_samples": 0.0,
            "front_p_mean_pa": nan,
            "front_sigma_n_mean_pa": nan,
            "front_sigma_comp_mean_pa": nan,
            "front_tau_t_abs_mean_pa": nan,
            "top_samples": 0.0,
            "top_p_mean_pa": nan,
            "top_sigma_n_mean_pa": nan,
            "top_tau_t_mean_pa": nan,
            "top_tau_t_abs_mean_pa": nan,
            "top_mu_du_dh_mean_pa": nan,
            "top_tau_over_mu_du_dh_mean": nan,
        }

    pts_all = []
    for pts in contours:
        arr = np.asarray(pts, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            continue
        pts_all.append(arr[:, :2].copy())
    if not pts_all:
        return {
            "n_samples": 0.0,
            "p_mean_pa": nan,
            "p_max_pa": nan,
            "sigma_n_mean_pa": nan,
            "sigma_n_min_pa": nan,
            "sigma_comp_mean_pa": nan,
            "tau_t_abs_mean_pa": nan,
            "tau_t_abs_max_pa": nan,
            "front_samples": 0.0,
            "front_p_mean_pa": nan,
            "front_sigma_n_mean_pa": nan,
            "front_sigma_comp_mean_pa": nan,
            "front_tau_t_abs_mean_pa": nan,
            "top_samples": 0.0,
            "top_p_mean_pa": nan,
            "top_sigma_n_mean_pa": nan,
            "top_tau_t_mean_pa": nan,
            "top_tau_t_abs_mean_pa": nan,
            "top_mu_du_dh_mean_pa": nan,
            "top_tau_over_mu_du_dh_mean": nan,
        }

    pts = np.vstack(pts_all)
    if pts.shape[0] > 1:
        pts_key = np.round(pts, decimals=12)
        pts = np.unique(pts_key, axis=0)
    if pts.shape[0] > int(max_points):
        idx = np.linspace(0, pts.shape[0] - 1, int(max_points), dtype=int)
        pts = pts[idx]

    p_vals: list[float] = []
    sigma_n_vals: list[float] = []
    tau_t_vals: list[float] = []
    front_p_vals: list[float] = []
    front_sigma_n_vals: list[float] = []
    front_tau_t_vals: list[float] = []
    top_p_vals: list[float] = []
    top_sigma_n_vals: list[float] = []
    top_tau_t_vals: list[float] = []
    top_mu_du_dh_vals: list[float] = []
    top_tau_ratio_vals: list[float] = []

    for xy in np.asarray(pts, dtype=float):
        try:
            p_val, _ = _eval_scalar_with_grad(dh, mesh, p_fn, (float(xy[0]), float(xy[1])))
            _, grad_a = _eval_scalar_with_grad(dh, mesh, alpha_fn, (float(xy[0]), float(xy[1])))
            _, grad_v = _eval_vector_with_grad(dh, mesh, v_fn, (float(xy[0]), float(xy[1])))
        except Exception:
            continue
        grad_a = np.asarray(grad_a, dtype=float).ravel()
        grad_v = np.asarray(grad_v, dtype=float)
        if grad_a.size != 2 or grad_v.shape != (2, 2):
            continue
        grad_norm = float(np.linalg.norm(grad_a))
        if not np.isfinite(grad_norm) or grad_norm <= 1.0e-12:
            continue
        n_out = -grad_a / grad_norm
        t_hat = np.asarray([-n_out[1], n_out[0]], dtype=float)
        sigma = -float(p_val) * np.eye(2, dtype=float) + float(mu_f) * (grad_v + grad_v.T)
        traction = sigma @ n_out
        sigma_n = float(np.dot(traction, n_out))
        tau_t = float(np.dot(traction, t_hat))

        p_vals.append(float(p_val))
        sigma_n_vals.append(sigma_n)
        tau_t_vals.append(tau_t)

        if float(n_out[0]) < -0.5:
            front_p_vals.append(float(p_val))
            front_sigma_n_vals.append(sigma_n)
            front_tau_t_vals.append(tau_t)
        if float(n_out[1]) > 0.5:
            top_p_vals.append(float(p_val))
            top_sigma_n_vals.append(sigma_n)
            top_tau_t_vals.append(tau_t)
            mu_du_dh = float(mu_f) * float(grad_v[0, 1])
            top_mu_du_dh_vals.append(mu_du_dh)
            if abs(mu_du_dh) > 1.0e-12:
                top_tau_ratio_vals.append(float(tau_t / mu_du_dh))

    def _mean_or_nan(vals: list[float]) -> float:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else nan

    def _maxabs_or_nan(vals: list[float]) -> float:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.max(np.abs(arr))) if arr.size else nan

    def _min_or_nan(vals: list[float]) -> float:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.min(arr)) if arr.size else nan

    def _mean_comp_or_nan(vals: list[float]) -> float:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(np.maximum(-arr, 0.0))) if arr.size else nan

    return {
        "n_samples": float(len(p_vals)),
        "p_mean_pa": _mean_or_nan(p_vals),
        "p_max_pa": _maxabs_or_nan(p_vals),
        "sigma_n_mean_pa": _mean_or_nan(sigma_n_vals),
        "sigma_n_min_pa": _min_or_nan(sigma_n_vals),
        "sigma_comp_mean_pa": _mean_comp_or_nan(sigma_n_vals),
        "tau_t_abs_mean_pa": _mean_or_nan([abs(v) for v in tau_t_vals]),
        "tau_t_abs_max_pa": _maxabs_or_nan(tau_t_vals),
        "front_samples": float(len(front_p_vals)),
        "front_p_mean_pa": _mean_or_nan(front_p_vals),
        "front_sigma_n_mean_pa": _mean_or_nan(front_sigma_n_vals),
        "front_sigma_comp_mean_pa": _mean_comp_or_nan(front_sigma_n_vals),
        "front_tau_t_abs_mean_pa": _mean_or_nan([abs(v) for v in front_tau_t_vals]),
        "top_samples": float(len(top_p_vals)),
        "top_p_mean_pa": _mean_or_nan(top_p_vals),
        "top_sigma_n_mean_pa": _mean_or_nan(top_sigma_n_vals),
        "top_tau_t_mean_pa": _mean_or_nan(top_tau_t_vals),
        "top_tau_t_abs_mean_pa": _mean_or_nan([abs(v) for v in top_tau_t_vals]),
        "top_mu_du_dh_mean_pa": _mean_or_nan(top_mu_du_dh_vals),
        "top_tau_over_mu_du_dh_mean": _mean_or_nan(top_tau_ratio_vals),
    }


def _write_contours_csv(path: Path, contours: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("contour_id,point_id,x_m,y_m\n")
        for cid, pts in enumerate(contours):
            arr = np.asarray(pts, dtype=float)
            for pid, (xx, yy) in enumerate(arr):
                f.write(f"{int(cid)},{int(pid)},{float(xx):.12e},{float(yy):.12e}\n")


def _lame_from_E_nu(E: float, nu: float) -> tuple[float, float]:
    E = float(E)
    nu = float(nu)
    if not (E > 0.0):
        raise ValueError("--E must be positive.")
    if not (-1.0 < nu < 0.5):
        raise ValueError("--nu must satisfy -1 < nu < 0.5.")
    mu = E / (2.0 * (1.0 + nu))
    lam = (E * nu) / ((1.0 + nu) * max(1.0e-16, (1.0 - 2.0 * nu)))
    return float(mu), float(lam)


def _phi_init_from_alpha(alpha_vals: np.ndarray, *, phi_b: float, mode: str) -> np.ndarray:
    aa = np.asarray(alpha_vals, dtype=float)
    key = str(mode).strip().lower()
    if key in {"linear_alpha", "linear", "historical"}:
        phi = 1.0 - (1.0 - float(phi_b)) * aa
    elif key in {"constant_phi_b", "constant", "phi_b"}:
        phi = np.full_like(aa, float(phi_b), dtype=float)
    else:
        raise ValueError(
            f"Unknown phi_init_mode {mode!r}. Use 'linear_alpha' or 'constant_phi_b'."
        )
    return np.clip(np.asarray(phi, dtype=float), 0.0, 1.0)


def _restart_peek(path: Path) -> dict[str, float | int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{str(path)}\n"
            "Hint: --restart-from must point to an existing checkpoint (.npz).\n"
            "  - For a fresh run, omit --restart-from and set --restart-write-every N to create checkpoints.\n"
            "  - Then restart from <out_dir>/restart/checkpoint_latest.npz or checkpoint_step=XXXXX.npz."
        )
    with np.load(str(path)) as data:
        fmt = int(np.asarray(data.get("format_version", [0]), dtype=int).ravel()[0])
        if fmt not in (1,):
            raise ValueError(f"Unsupported restart format_version={fmt} in {path}")
        t = float(np.asarray(data["t"], dtype=float).ravel()[0])
        step = int(np.asarray(data["step"], dtype=int).ravel()[0])
        dt = float(np.asarray(data["dt"], dtype=float).ravel()[0])
        theta = float(np.asarray(data.get("theta", [1.0]), dtype=float).ravel()[0])
        gamma_div = float(np.asarray(data.get("gamma_div", [float("nan")]), dtype=float).ravel()[0])
    return {"t": t, "step": step, "dt": dt, "theta": theta, "gamma_div": gamma_div}


def _write_restart_checkpoint(
    path: Path,
    *,
    t: float,
    step: int,
    dt: float,
    theta: float,
    gamma_div: float,
    y_lines: np.ndarray,
    x_ref_global: float,
    x_ref: np.ndarray,
    x_prev: np.ndarray,
    v: VectorFunction,
    p: Function,
    vS: VectorFunction,
    u: VectorFunction,
    alpha: Function,
    phi: Function | None = None,
    S: Function | None = None,
    mu_alpha: Function | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(
        format_version=np.asarray([1], dtype=int),
        t=float(t),
        step=int(step),
        dt=float(dt),
        theta=float(theta),
        gamma_div=float(gamma_div),
        y_lines=np.asarray(y_lines, dtype=float).ravel(),
        x_ref_global=float(x_ref_global),
        x_ref=np.asarray(x_ref, dtype=float).ravel(),
        x_prev=np.asarray(x_prev, dtype=float).ravel(),
        v=np.asarray(v.nodal_values, dtype=float),
        p=np.asarray(p.nodal_values, dtype=float),
        vS=np.asarray(vS.nodal_values, dtype=float),
        u=np.asarray(u.nodal_values, dtype=float),
        alpha=np.asarray(alpha.nodal_values, dtype=float),
    )
    if phi is not None:
        payload["phi"] = np.asarray(phi.nodal_values, dtype=float)
    if S is not None:
        payload["S"] = np.asarray(S.nodal_values, dtype=float)
    if mu_alpha is not None:
        payload["mu_alpha"] = np.asarray(mu_alpha.nodal_values, dtype=float)

    np.savez_compressed(str(path), **payload)


def _load_restart_checkpoint(
    path: Path,
    *,
    v: VectorFunction,
    p: Function,
    vS: VectorFunction,
    u: VectorFunction,
    alpha: Function,
    phi: Function | None = None,
    S: Function | None = None,
    mu_alpha: Function | None = None,
) -> dict[str, object]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    with np.load(str(path)) as data:
        fmt = int(np.asarray(data.get("format_version", [0]), dtype=int).ravel()[0])
        if fmt not in (1,):
            raise ValueError(f"Unsupported restart format_version={fmt} in {path}")

        def _load_into(func, key: str) -> None:
            arr = np.asarray(data[key], dtype=float).ravel()
            if func.nodal_values.size != arr.size:
                raise ValueError(
                    f"Restart field {key!r} size mismatch: file has {int(arr.size)} values, "
                    f"function has {int(func.nodal_values.size)}."
                )
            func.nodal_values[:] = arr

        _load_into(v, "v")
        _load_into(p, "p")
        _load_into(vS, "vS")
        _load_into(u, "u")
        _load_into(alpha, "alpha")
        if phi is not None and "phi" in data:
            _load_into(phi, "phi")
        if S is not None and "S" in data:
            _load_into(S, "S")
        if mu_alpha is not None and "mu_alpha" in data:
            _load_into(mu_alpha, "mu_alpha")

        meta = {
            "t": float(np.asarray(data["t"], dtype=float).ravel()[0]),
            "step": int(np.asarray(data["step"], dtype=int).ravel()[0]),
            "dt": float(np.asarray(data["dt"], dtype=float).ravel()[0]),
            "theta": float(np.asarray(data.get("theta", [1.0]), dtype=float).ravel()[0]),
            "gamma_div": float(np.asarray(data.get("gamma_div", [float("nan")]), dtype=float).ravel()[0]),
            "y_lines": np.asarray(data.get("y_lines", []), dtype=float).ravel(),
            "x_ref_global": float(np.asarray(data.get("x_ref_global", [float("nan")]), dtype=float).ravel()[0]),
            "x_ref": np.asarray(data.get("x_ref", []), dtype=float).ravel(),
            "x_prev": np.asarray(data.get("x_prev", []), dtype=float).ravel(),
        }
    return meta


def _quad_corner_indices(p: int) -> tuple[int, int, int, int]:
    """Return (bl, br, tr, tl) local indices in lattice order (eta outer, xi inner)."""
    n = p + 1
    bl = 0
    br = p
    tr = p * n + p
    tl = p * n
    return bl, br, tr, tl


def _refine_element_quad4(
    mesh: Mesh, eid: int, nodes: list[Node], node_lookup: dict[tuple[float, float], int]
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Split one quad element into 4 children (single-level isotropic refinement).

    Produces a 2:1 nonconforming mesh (hanging nodes) handled by the solver's
    constraint layer. Returns (child_connectivity, child_corners).
    """
    if mesh.element_type != "quad":
        raise ValueError("quad4 refinement only supports quad meshes.")
    p = int(mesh.poly_order)
    t = np.linspace(-1.0, 1.0, p + 1)

    parent_conn = np.asarray(mesh.elements_connectivity[int(eid)], dtype=int).ravel()
    if parent_conn.size != (p + 1) ** 2:
        raise ValueError(f"Unexpected parent connectivity size for p={p}: {int(parent_conn.size)}")

    def _parent_node(xi_p: float, eta_p: float) -> int | None:
        ix = np.where(np.isclose(t, float(xi_p), atol=1e-12))[0]
        iy = np.where(np.isclose(t, float(eta_p), atol=1e-12))[0]
        if ix.size and iy.size:
            idx = int(iy[0] * (p + 1) + ix[0])
            return int(parent_conn[idx])
        return None

    def _get_node(xi_p: float, eta_p: float) -> int:
        # Reuse parent nodes when possible; otherwise create/reuse global by coordinate.
        nid = _parent_node(xi_p, eta_p)
        if nid is not None:
            return nid
        x_phys = transform.x_mapping(mesh, int(eid), (float(xi_p), float(eta_p)))
        key = (float(round(float(x_phys[0]), 14)), float(round(float(x_phys[1]), 14)))
        nid = node_lookup.get(key)
        if nid is not None:
            return int(nid)
        nid = len(nodes)
        node_lookup[key] = int(nid)
        nodes.append(Node(int(nid), float(x_phys[0]), float(x_phys[1])))
        return int(nid)

    def _child(mode: str) -> tuple[list[int], list[int]]:
        # mode: "bl", "br", "tr", "tl"
        conn: list[int] = []
        for eta in t:
            for xi in t:
                if mode == "bl":
                    xi_p = 0.5 * (float(xi) - 1.0)
                    eta_p = 0.5 * (float(eta) - 1.0)
                elif mode == "br":
                    xi_p = 0.5 * (float(xi) + 1.0)
                    eta_p = 0.5 * (float(eta) - 1.0)
                elif mode == "tr":
                    xi_p = 0.5 * (float(xi) + 1.0)
                    eta_p = 0.5 * (float(eta) + 1.0)
                elif mode == "tl":
                    xi_p = 0.5 * (float(xi) - 1.0)
                    eta_p = 0.5 * (float(eta) + 1.0)
                else:
                    raise ValueError(mode)
                conn.append(_get_node(xi_p, eta_p))
        bl, br, tr, tl = _quad_corner_indices(p)
        corners = [conn[bl], conn[br], conn[tr], conn[tl]]
        return conn, corners

    c_bl, cr_bl = _child("bl")
    c_br, cr_br = _child("br")
    c_tr, cr_tr = _child("tr")
    c_tl, cr_tl = _child("tl")
    return [c_bl, c_br, c_tr, c_tl], [cr_bl, cr_br, cr_tr, cr_tl]


def refine_around_biofilm_bbox(
    mesh: Mesh,
    *,
    poly: np.ndarray,
    band: float,
    expand_layers: int = 0,
    L: float,
    H: float,
) -> Mesh:
    """
    Refine quads intersecting the biofilm bbox (expanded by band) in one pass.

    This produces a 2:1 nonconforming mesh (hanging nodes), handled by the
    solver's constraint layer. The refinement is applied only once (no nested
    refinement levels).
    """
    if mesh.element_type != "quad":
        return mesh

    pts = np.asarray(poly, dtype=float)
    x_min = float(np.min(pts[:, 0])) - float(band)
    x_max = float(np.max(pts[:, 0])) + float(band)
    y_min = float(np.min(pts[:, 1])) - float(band)
    y_max = float(np.max(pts[:, 1])) + float(band)
    x_min = max(0.0, x_min)
    x_max = min(float(L), x_max)
    y_min = max(0.0, y_min)
    y_max = min(float(H), y_max)

    marked: set[int] = set()
    for eid, elem in enumerate(mesh.elements_list):
        corners = mesh.nodes_x_y_pos[list(elem.corner_nodes)]
        ex_min, ey_min = corners.min(axis=0)
        ex_max, ey_max = corners.max(axis=0)
        hits_bbox = (ex_min <= x_max) and (ex_max >= x_min) and (ey_min <= y_max) and (ey_max >= y_min)
        if hits_bbox:
            marked.add(int(eid))

    for _ in range(max(0, int(expand_layers))):
        new: set[int] = set()
        for eid in marked:
            for nb in mesh._neighbors[int(eid)]:
                if nb is not None:
                    new.add(int(nb))
        marked |= new

    if not marked:
        return mesh

    corner_conn = getattr(mesh, "corner_connectivity", None)
    if corner_conn is None:
        corner_conn = getattr(mesh, "elements_corner_nodes", None)
    if corner_conn is None:
        raise RuntimeError("Mesh does not expose corner connectivity required for refinement.")

    nodes = list(mesh.nodes_list)
    node_lookup = {(round(float(nd.x), 14), round(float(nd.y), 14)): int(nd.id) for nd in nodes}
    new_elems: list[list[int]] = []
    new_corners: list[list[int]] = []

    for eid in range(len(mesh.elements_list)):
        if eid not in marked:
            new_elems.append(list(np.asarray(mesh.elements_connectivity[int(eid)], dtype=int).ravel()))
            new_corners.append(list(np.asarray(corner_conn[int(eid)], dtype=int).ravel()))
            continue
        conns, corners = _refine_element_quad4(mesh, int(eid), nodes, node_lookup)
        new_elems.extend(conns)
        new_corners.extend(corners)

    new_mesh = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(new_elems, dtype=int),
        elements_corner_nodes=np.asarray(new_corners, dtype=int),
        element_type="quad",
        poly_order=mesh.poly_order,
    )
    _tag_channel_boundaries(new_mesh, L=float(L), H=float(H))
    logging.info(
        f"[refine] bbox band={float(band):.3e}m, layers={int(expand_layers)}: "
        f"marked {len(marked)} elems -> {len(new_elems)} elements, {len(nodes)} nodes"
    )
    return new_mesh


def main() -> None:
    ap = argparse.ArgumentParser(description="Blauert et al. (2015) deformation benchmark (one-domain).")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--nx", type=int, default=220)
    ap.add_argument("--ny", type=int, default=80)
    ap.add_argument("--q", type=int, default=8, help="Quadrature degree.")

    ap.add_argument("--L", type=float, default=5.5e-3, help="Channel length [m]. Default matches the Dian SPH setup (5.5 mm).")
    ap.add_argument("--H", type=float, default=1.0e-3, help="Channel height [m].")

    # Local refinement around the biofilm polygon (single pass; produces hanging nodes)
    ap.add_argument("--refine-biofilm", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--refine-band",
        type=float,
        default=float("nan"),
        help="Refinement band around the biofilm bbox [m]. Default is ~2*max(hx,hy).",
    )
    ap.add_argument("--refine-expand-layers", type=int, default=0, help="Expand marked region by N neighbor layers (still one refinement pass).")

    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--t-final", type=float, default=10.0)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument(
        "--include-skeleton-acceleration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include Eulerian skeleton acceleration in the one-domain skeleton momentum block.",
    )
    ap.add_argument(
        "--rho-s0-tilde",
        type=float,
        default=0.0,
        help="Reference solid density coefficient used by the Eulerian skeleton acceleration term.",
    )
    ap.add_argument(
        "--skeleton-inertia-convection",
        type=str,
        default="lagged",
        choices=("lagged", "full"),
        help="Treatment of the convective part of the Eulerian skeleton inertia.",
    )
    ap.add_argument("--allow-dt-reduction", action="store_true")
    ap.add_argument("--dt-min", type=float, default=0.01)
    ap.add_argument("--dt-reduction-factor", type=float, default=0.5)

    ap.add_argument(
        "--predictor",
        type=str,
        default="delta",
        choices=("prev", "delta"),
        help="Initial guess for each time step: 'prev' uses U^n, 'delta' extrapolates using the previous accepted increment.",
    )
    ap.add_argument(
        "--predictor-damping",
        type=float,
        default=0.5,
        help="Damping factor for the 'delta' predictor (0 disables). Values < 1 often improve robustness during inflow ramping.",
    )
    ap.add_argument(
        "--predictor-clip-01",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip predicted alpha/phi to [0,1] (predictor only; does not change converged solution).",
    )
    ap.add_argument(
        "--startup-staggered-predictor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Before each monolithic Newton/PDAS step, compute a better initial guess by first "
            "solving the fluid block with frozen solid/transport fields and then the solid block "
            "with frozen fluid loading."
        ),
    )
    ap.add_argument(
        "--startup-staggered-max-time",
        type=float,
        default=float("nan"),
        help=(
            "Apply --startup-staggered-predictor only while t_{n+theta} <= this time. "
            "Default: disabled limit (all steps when the predictor is enabled)."
        ),
    )
    ap.add_argument(
        "--startup-fluid-newton-tol",
        type=float,
        default=1.0e-10,
        help="Absolute Newton tolerance used by the restricted fluid startup solve.",
    )
    ap.add_argument(
        "--startup-solid-newton-tol",
        type=float,
        default=1.0e-10,
        help="Absolute Newton tolerance used by the restricted solid startup solve.",
    )
    ap.add_argument(
        "--startup-fluid-max-it",
        type=int,
        default=12,
        help="Maximum Newton iterations for the restricted fluid startup solve.",
    )
    ap.add_argument(
        "--startup-solid-max-it",
        type=int,
        default=12,
        help="Maximum Newton iterations for the restricted solid startup solve.",
    )
    ap.add_argument(
        "--startup-staggered-sweeps",
        type=int,
        default=1,
        help="Number of fluid->solid startup sweeps before the monolithic solve (>=1).",
    )
    ap.add_argument(
        "--startup-staggered-slip-tol",
        type=float,
        default=0.0,
        help="Optional early-stop tolerance on |v-vS|_inf after a startup sweep (<=0 disables).",
    )

    ap.add_argument("--newton-tol", type=float, default=1.0e-6)
    ap.add_argument("--newton-rtol", type=float, default=0.0, help="Relative SNES tolerance (0 disables).")
    ap.add_argument("--max-it", type=int, default=25)
    ap.add_argument(
        "--nonlinear-solver",
        type=str,
        default="pdas",
        choices=("pdas", "newton", "snes"),
        help="Nonlinear solver used for the monolithic step. Benchmark 6 defaults to the internal PDAS path.",
    )
    ap.add_argument(
        "--ls-mode",
        type=str,
        default="dealii",
        choices=("armijo", "dealii"),
        help="Line-search mode used by the internal Newton / PDAS solvers.",
    )
    ap.add_argument("--vi-c", type=float, default=0.0, help="PDAS active-set scaling parameter.")
    ap.add_argument("--vi-enter-tol", type=float, default=0.0, help="Active-set entry threshold for PDAS hysteresis.")
    ap.add_argument("--vi-leave-tol", type=float, default=0.0, help="Active-set release threshold for PDAS hysteresis.")
    ap.add_argument("--vi-persistence", type=int, default=0, help="Iterations a proposed active-set change must persist before it is accepted.")
    ap.add_argument("--vi-lambda0", type=float, default=0.0, help="Initial inactive-block PDAS regularization lambda.")
    ap.add_argument("--vi-lambda-max", type=float, default=1.0e6, help="Maximum inactive-block PDAS regularization lambda.")
    ap.add_argument("--vi-lambda-growth", type=float, default=5.0, help="Multiplicative growth factor for inactive-block PDAS regularization.")
    ap.add_argument("--vi-lambda-decay", type=float, default=0.5, help="Decay factor applied to inactive-block PDAS regularization after easy accepted full steps.")
    ap.add_argument("--vi-active-soft-threshold", type=int, default=0, help="Enable soft damping of marginal active DOFs when DeltaA exceeds this threshold.")
    ap.add_argument("--vi-active-soft-alpha", type=float, default=1.0, help="Step factor used for marginal active DOFs when soft active damping is enabled.")
    ap.add_argument("--vi-active-strong-factor", type=float, default=5.0, help="Indicator multiple used to classify clearly active DOFs that remain hard-updated.")
    ap.add_argument("--vi-filter-max-delta-active", type=int, default=0, help="Reject VI line-search trials whose predicted DeltaA exceeds this threshold (0 disables).")
    ap.add_argument("--vi-unconstrained-lm", action=argparse.BooleanOptionalAction, default=False, help="Enable trust-region / Levenberg-Marquardt globalization when the PDAS active set is empty.")
    ap.add_argument("--vi-lm-lambda0", type=float, default=1.0e-4, help="Initial LM damping parameter for the zero-active-set branch.")
    ap.add_argument("--vi-lm-lambda-max", type=float, default=1.0e6, help="Maximum LM damping parameter for the zero-active-set branch.")
    ap.add_argument("--vi-lm-growth", type=float, default=5.0, help="Multiplicative growth factor for LM damping after rejected steps.")
    ap.add_argument("--vi-lm-decay", type=float, default=0.5, help="Decay factor for LM damping after good accepted steps.")
    ap.add_argument("--vi-lm-accept-ratio", type=float, default=1.0e-3, help="Minimum actual/predicted reduction ratio required to accept an LM step.")
    ap.add_argument("--vi-lm-good-ratio", type=float, default=0.75, help="Ratio above which LM damping is relaxed after an accepted step.")
    ap.add_argument("--vi-lm-max-tries", type=int, default=6, help="Maximum LM trust-region retries per Newton iteration.")
    ap.add_argument("--alpha-box-constraints", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--phi-box-constraints", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--newton-solver",
        type=str,
        default="pdas",
        choices=("pdas", "snes"),
        help="Nonlinear solver: 'pdas' (semismooth Newton for box constraints) or 'snes' (PETSc SNES).",
    )
    ap.add_argument(
        "--accept-nonconverged-atol-factor",
        type=float,
        default=0.0,
        help="Accept SNES best iterate when ‖F‖ <= factor*atol even if SNES reports non-convergence (0 disables).",
    )
    ap.add_argument(
        "--lu-solver",
        type=str,
        default="mumps",
        choices=("mumps", "superlu_dist", "superlu"),
        help="Direct LU factorization backend used by PETSc (SNES only).",
    )
    ap.add_argument(
        "--linear-backend",
        type=str,
        default="petsc",
        choices=("petsc", "scipy", "pardiso"),
        help="Linear algebra backend for internal Newton/PDAS solves (`pardiso` uses pypardiso/MKL).",
    )
    ap.add_argument("--linear-ksp-type", type=str, default="", help="Internal PETSc KSP type for Newton/PDAS/LM linear solves.")
    ap.add_argument("--linear-pc-type", type=str, default="", help="Internal PETSc PC type for Newton/PDAS/LM linear solves.")
    ap.add_argument("--linear-pc-factor-solver-type", type=str, default="", help="Direct factor backend for the internal PETSc linear solve.")
    ap.add_argument("--linear-ksp-rtol", type=float, default=float("nan"), help="Relative tolerance for the internal PETSc KSP.")
    ap.add_argument("--linear-ksp-atol", type=float, default=float("nan"), help="Absolute tolerance for the internal PETSc KSP.")
    ap.add_argument("--linear-ksp-dtol", type=float, default=float("nan"), help="Divergence tolerance for the internal PETSc KSP.")
    ap.add_argument("--linear-ksp-max-it", type=int, default=-1, help="Maximum iterations for the internal PETSc KSP.")
    ap.add_argument("--linear-ksp-trace", action=argparse.BooleanOptionalAction, default=False, help="Print PETSc KSP convergence diagnostics for internal Newton/PDAS solves.")
    ap.add_argument("--linear-schur", action=argparse.BooleanOptionalAction, default=False, help="Use a PETSc Schur-complement fieldsplit on the internal Newton/PDAS linear solve.")
    ap.add_argument("--linear-schur-pressure-field", type=str, default="p", help="Pressure field name used for the Schur pressure split.")
    ap.add_argument("--linear-schur-fact", type=str, default="full", choices=("full", "upper", "lower", "diag"), help="PETSc Schur factorization type.")
    ap.add_argument("--linear-schur-pre", type=str, default="selfp", choices=("selfp", "a11", "user"), help="PETSc Schur preconditioner type.")
    ap.add_argument("--linear-schur-rest-ksp", type=str, default="preonly", help="KSP type for the non-pressure Schur block.")
    ap.add_argument("--linear-schur-rest-pc", type=str, default="ilu", help="PC type for the non-pressure Schur block.")
    ap.add_argument("--linear-schur-rest-factor-solver-type", type=str, default="", help="Optional direct factor backend for the non-pressure Schur block.")
    ap.add_argument("--linear-schur-pressure-ksp", type=str, default="preonly", help="KSP type for the pressure Schur block.")
    ap.add_argument("--linear-schur-pressure-pc", type=str, default="jacobi", help="PC type for the pressure Schur block.")
    ap.add_argument("--linear-schur-pressure-factor-solver-type", type=str, default="", help="Optional direct factor backend for the pressure Schur block.")

    # Flow
    ap.add_argument(
        "--u-avg",
        type=float,
        default=4.56e-2,
        help=(
            "Average inflow velocity [m/s]. For the parabolic channel inflow used here, "
            "the Dian preprocessing value v0=6.84e-2 m/s corresponds to u_avg≈4.56e-2 m/s."
        ),
    )
    ap.add_argument(
        "--re-char-length",
        type=float,
        default=float("nan"),
        help=(
            "Characteristic length [m] used only for Reynolds-number reporting in setup logs. "
            "Default: reuse the channel height H."
        ),
    )
    ap.add_argument(
        "--inflow-profile",
        type=str,
        default="poiseuille",
        choices=("poiseuille", "plug"),
        help=(
            "Left-boundary inflow profile. 'poiseuille' matches the current paper benchmark; "
            "'plug' applies a uniform inlet speed and defaults to the Dian/SPH peak speed "
            "v0 = 1.5 * u_avg unless --inflow-plug-speed is set explicitly."
        ),
    )
    ap.add_argument(
        "--inflow-plug-speed",
        type=float,
        default=float("nan"),
        help=(
            "Uniform inlet speed [m/s] used when --inflow-profile=plug. "
            "Default: 1.5 * --u-avg so the Dian/SPH v0 is recovered from the paper u_avg."
        ),
    )
    ap.add_argument("--t-ramp", type=float, default=0.5, help="Cosine ramp time for inflow [s].")
    ap.add_argument(
        "--flow-shutoff-start",
        type=float,
        default=float("nan"),
        help=(
            "Optional late-time start [s] for a cosine shutoff of the imposed flow drive. "
            "When set together with --flow-shutoff-end, the inflow and all diffuse flow-driven "
            "tractions are multiplied by a smooth factor that transitions from 1 to "
            "--flow-shutoff-end-factor."
        ),
    )
    ap.add_argument(
        "--flow-shutoff-end",
        type=float,
        default=float("nan"),
        help="Optional late-time end [s] for the cosine shutoff of the imposed flow drive.",
    )
    ap.add_argument(
        "--flow-shutoff-end-factor",
        type=float,
        default=0.0,
        help="Residual multiplier after --flow-shutoff-end (default: 0, i.e. full shutoff).",
    )

    # Material / coupling (FSI-only: disable growth/detachment/damage)
    ap.add_argument("--rho-f", type=float, default=1000.0)
    ap.add_argument("--mu-f", type=float, default=1.0e-3)
    ap.add_argument(
        "--mu-b-model",
        type=str,
        default="phi_mu",
        choices=("mu", "phi_mu"),
        help="Effective viscosity model μ(α,φ). 'mu' keeps μ≡μ_f; 'phi_mu' (default) uses Brinkman scaling μ=μ_f((1-α)+αφ).",
    )
    ap.add_argument("--kappa-inv", type=float, default=1.0e12, help="Inverse permeability [1/m^2].")
    ap.add_argument(
        "--kappa-inv-model",
        type=str,
        default="spatial",
        choices=("spatial", "kozeny", "kozeny_carman", "kc", "refmap"),
        help=(
            "Inverse permeability model. 'spatial' keeps kappa_inv constant; "
            "'kozeny_carman' scales it with the transported phi field; "
            "'refmap' pushes a reference inverse permeability through the deformation."
        ),
    )
    ap.add_argument(
        "--kappa-phi-ref",
        type=float,
        default=None,
        help="Reference porosity for Kozeny-Carman normalization. Default: reuse --phi-b.",
    )
    ap.add_argument(
        "--kappa-inv-kc-eps",
        type=float,
        default=1.0e-12,
        help="Regularization epsilon used in the Kozeny-Carman porosity scaling.",
    )
    ap.add_argument("--phi-b", type=float, default=0.47, help="Initial porosity inside the biofilm (0<phi_b<1).")
    ap.add_argument(
        "--phi-init-mode",
        type=str,
        default="linear_alpha",
        choices=("linear_alpha", "constant_phi_b"),
        help=(
            "How the porosity field is initialized from alpha in full PDE mode. "
            "'linear_alpha' uses phi=1-(1-phi_b)alpha; 'constant_phi_b' sets phi=phi_b everywhere initially."
        ),
    )
    ap.add_argument("--E", type=float, default=200.0, help="Young's modulus of the solid phase [Pa] (Dian paper default).")
    ap.add_argument("--nu", type=float, default=0.4, help="Poisson ratio (Dian paper default).")
    ap.add_argument(
        "--alpha-biot",
        type=float,
        default=float("nan"),
        help=(
            "Optional benchmark-local Biot coefficient for an added volumetric skeleton "
            "pressure load. When finite, the skeleton div(vS_test) pressure coefficient "
            "is corrected toward alpha_biot * alpha while the original diffuse-interface "
            "grad(B) term is kept."
        ),
    )
    ap.add_argument(
        "--solid-model",
        type=str,
        default="linear",
        choices=("linear", "neo_hookean", "neo-hookean", "nh", "hencky", "svk"),
        help=(
            "Skeleton constitutive law. 'linear' is the current default; "
            "'neo_hookean', 'hencky', and 'svk' activate the large-deformation "
            "hyperelastic variants already implemented in the one-domain form assembly."
        ),
    )
    ap.add_argument("--solid-visco-eta", type=float, default=0.0, help="Kelvin–Voigt viscosity eta_s [Pa*s] (0 disables).")
    ap.add_argument(
        "--attachment-mode",
        type=str,
        default="clamped",
        choices=("clamped", "adhesion"),
        help=(
            "Bottom attachment model for the skeleton. 'clamped' reproduces the current "
            "Benchmark 6 setup; 'adhesion' replaces the hard bottom clamp with the "
            "existing wall-adhesion spring/dashpot traction on the substratum."
        ),
    )
    ap.add_argument(
        "--adhesion-k-n",
        type=float,
        default=0.0,
        help="Bottom-attachment normal spring stiffness [Pa/m] used when --attachment-mode adhesion.",
    )
    ap.add_argument(
        "--adhesion-k-t",
        type=float,
        default=0.0,
        help="Bottom-attachment tangential spring stiffness [Pa/m] used when --attachment-mode adhesion.",
    )
    ap.add_argument(
        "--adhesion-gamma-n",
        type=float,
        default=0.0,
        help="Bottom-attachment normal dashpot [Pa*s/m] used when --attachment-mode adhesion.",
    )
    ap.add_argument(
        "--adhesion-gamma-t",
        type=float,
        default=0.0,
        help="Bottom-attachment tangential dashpot [Pa*s/m] used when --attachment-mode adhesion.",
    )
    ap.add_argument(
        "--paper1-reduced",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the Paper-1 reduced deformation-only model with fixed phi_b and active "
            "conservative Cahn--Hilliard alpha transport. This disables the full phi/S blocks."
        ),
    )
    ap.add_argument(
        "--diffuse-shear-traction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a lagged diffuse-interface tangential shear-traction transfer term "
            "computed from the current fluid field. This is intended for application-style "
            "channel deformation cases where viscous surface loading is essential."
        ),
    )
    ap.add_argument(
        "--diffuse-shear-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the diffuse-interface tangential shear traction.",
    )
    ap.add_argument(
        "--diffuse-shear-scale-ref",
        type=float,
        default=50.0,
        help=(
            "Reference diffuse-traction scale used when --scale-alpha-ch-eps-with-zeta is enabled. "
            "For |zeta| above this reference, alpha CH eps is increased like (|zeta|/zeta_ref)^2."
        ),
    )
    ap.add_argument(
        "--diffuse-shear-model",
        type=str,
        default="lagged_velocity",
        choices=("lagged_velocity", "lagged_stress", "poiseuille"),
        help=(
            "How the diffuse-interface tangential shear traction is computed: "
            "'lagged_velocity' uses the tangential projection of 2 mu_f eps(v^n) n, "
            "'lagged_stress' uses the full lagged traction (-p^n I + 2 mu_f eps(v^n)) n, "
            "while 'poiseuille' uses the imposed channel Poiseuille shear profile."
        ),
    )
    ap.add_argument(
        "--diffuse-shear-time-scheme",
        type=str,
        default="constant",
        choices=("constant", "imex"),
        help=(
            "Time treatment for the benchmark-local diffuse traction correction: "
            "'constant' keeps the full correction active from t=0, while "
            "'imex' applies the correction with a lagged amplitude continuation "
            "factor based on the accepted previous-step time."
        ),
    )
    ap.add_argument(
        "--diffuse-shear-ramp-time",
        type=float,
        default=float("nan"),
        help=(
            "Ramp time [s] for --diffuse-shear-time-scheme imex. "
            "Default: reuse --t-ramp."
        ),
    )
    ap.add_argument(
        "--diffuse-shear-eta",
        type=float,
        default=1.0e-12,
        help="Regularization added to |grad(alpha)| when building the diffuse-interface normal.",
    )
    ap.add_argument(
        "--diffuse-shear-topweight",
        action="store_true",
        help=(
            "Weight the diffuse-interface tangential shear proxy by |n_y| so the "
            "benchmark-local channel shear acts mainly on the top-facing contour "
            "instead of the nearly vertical inflow/outflow faces."
        ),
    )
    ap.add_argument(
        "--diffuse-normal-pressure-scale",
        type=float,
        default=0.0,
        help=(
            "Additional scale applied to the lagged normal-pressure term -p^n n on the diffuse interface. "
            "Used to augment the Benchmark 6 Poiseuille tangential proxy with independently tuned normal compression."
        ),
    )
    ap.add_argument(
        "--diffuse-normal-pressure-xweight",
        action="store_true",
        help=(
            "Weight the added lagged normal-pressure term by |n_x| so it acts mainly "
            "on streamwise-facing interface segments."
        ),
    )
    ap.add_argument(
        "--diffuse-normal-pressure-upstream-only",
        action="store_true",
        help=(
            "Weight the added lagged normal-pressure term by max(n_x, 0) so the extra "
            "compression acts primarily on the inflow-facing interface."
        ),
    )
    ap.add_argument(
        "--diffuse-normal-stress-scale",
        type=float,
        default=0.0,
        help=(
            "Additional scale applied to the normal projection of the full lagged traction "
            "(-p^n I + 2 mu_f eps(v^n)) n on the diffuse interface. "
            "Used to augment the Benchmark 6 Poiseuille tangential proxy with independently tuned normal stress."
        ),
    )
    ap.add_argument(
        "--diffuse-normal-stress-ramp-time",
        type=float,
        default=float("nan"),
        help=(
            "Optional ramp time [s] applied only to the added diffuse normal-stress "
            "correction. Default: reuse --diffuse-shear-ramp-time / --t-ramp."
        ),
    )
    ap.add_argument(
        "--diffuse-normal-stress-xweight",
        action="store_true",
        help=(
            "Weight the added lagged normal-stress term by |n_x| so it acts mainly on "
            "streamwise-facing interface segments instead of the nearly horizontal top."
        ),
    )
    ap.add_argument(
        "--diffuse-normal-stress-topweight",
        action="store_true",
        help=(
            "Weight the added lagged normal-stress term by |n_y| so it acts mainly on "
            "the top/bottom flanks instead of the near-vertical mid-front."
        ),
    )
    ap.add_argument(
        "--diffuse-normal-stress-topbias",
        type=float,
        default=0.0,
        help=(
            "Smoothly bias the added lagged normal-stress term toward the top/bottom "
            "flanks using (1-b)*1 + b*|n_y|, with b in [0,1]. "
            "Use this to interpolate between uniform loading (0) and hard topweight (1)."
        ),
    )
    ap.add_argument(
        "--diffuse-normal-stress-frontbias",
        type=float,
        default=0.0,
        help=(
            "Smoothly bias the added lagged normal-stress term toward the inflow-facing "
            "interface using (1-b)*1 + b*max(n_x,0), with b in [0,1]. "
            "Use this to interpolate between uniform loading (0) and upstream-only loading (1)."
        ),
    )
    ap.add_argument(
        "--diffuse-normal-stress-frontbias-tail-value",
        type=float,
        default=float("nan"),
        help=(
            "Optional late-time value reached by --diffuse-normal-stress-frontbias over the "
            "same --diffuse-normal-stress-decay-start / --diffuse-normal-stress-decay-end interval "
            "used by the late normal-stress tail. Default keeps frontbias constant."
        ),
    )
    ap.add_argument(
        "--diffuse-normal-stress-bottomskew",
        type=float,
        default=0.0,
        help=(
            "Redistribute the added lagged normal-stress term toward the lower part of "
            "the channel using the height-normalized profile 1 + b*(1-2*y/H), with b in [0,1]. "
            "This raises lower-band compression while reducing upper-band loading at the same mean scale."
        ),
    )
    ap.add_argument(
        "--diffuse-normal-stress-bottomskew-ramp-with-tail",
        action="store_true",
        help=(
            "Ramp --diffuse-normal-stress-bottomskew from zero to its target value over the "
            "same --diffuse-normal-stress-decay-start / --diffuse-normal-stress-decay-end interval "
            "used by the late normal-stress tail."
        ),
    )
    ap.add_argument(
        "--diffuse-normal-stress-upstream-only",
        action="store_true",
        help=(
            "Weight the added lagged normal-stress term by max(n_x, 0) so the extra "
            "compression acts primarily on the inflow-facing interface."
        ),
    )
    ap.add_argument(
        "--diffuse-normal-stress-decay-start",
        type=float,
        default=float("nan"),
        help=(
            "Optional start time [s] for a cosine decay applied only to the added "
            "diffuse normal-stress correction."
        ),
    )
    ap.add_argument(
        "--diffuse-normal-stress-decay-end",
        type=float,
        default=float("nan"),
        help=(
            "Optional end time [s] for a cosine decay applied only to the added "
            "diffuse normal-stress correction."
        ),
    )
    ap.add_argument(
        "--diffuse-normal-stress-tail-factor",
        type=float,
        default=1.0,
        help=(
            "Late-time multiplier reached at --diffuse-normal-stress-decay-end for "
            "the added diffuse normal-stress correction. Default keeps the current "
            "constant-amplitude behavior."
        ),
    )

    ap.add_argument("--gamma-u", type=float, default=5.0, help="u extension penalty outside biofilm.")
    ap.add_argument(
        "--u-extension",
        type=str,
        default="l2",
        choices=("l2", "grad", "h1"),
        help=(
            "Extension regularization used outside the biofilm for u and, by default, vS. "
            "'l2' keeps the current mass-style penalty; 'grad'/'h1' use the H1-seminorm "
            "plus the tiny pinning term."
        ),
    )
    ap.add_argument("--gamma-u-pin", type=float, default=1.0e-4, help="Tiny pinning used only with --u-extension grad.")
    ap.add_argument(
        "--kinematics-scale",
        type=float,
        default=float("nan"),
        help="Scaling applied to the u-kinematics constraint residual (conditioning aid; does not change solution set).",
    )
    ap.add_argument(
        "--v-supg",
        type=float,
        default=0.0,
        help="SUPG-like streamline diffusion for fluid momentum convection (0 disables; typical 0.1–10).",
    )
    ap.add_argument(
        "--v-supg-mode",
        type=str,
        default="streamline",
        choices=("streamline", "residual"),
        help=(
            "Fluid momentum stabilization form for --v-supg: "
            "'streamline' keeps the legacy weak streamline term, "
            "'residual' uses a transient strong-residual SUPG term."
        ),
    )
    ap.add_argument(
        "--v-supg-c-nu",
        type=float,
        default=4.0,
        help=(
            "Viscous constant c_nu in the Green's-function-style elemental tau for "
            "fluid SUPG: tau_K^{-2} contains (c_nu * (mu/rho) / h_K^2)^2."
        ),
    )
    ap.add_argument(
        "--gamma-div",
        type=float,
        default=0.0,
        help=(
            "Consistent augmented-Lagrangian stabilization for the mixture volume constraint "
            "div(C v + B vS)=0. Acts like a grad-div penalty and can improve conditioning "
            "for long transient runs (0 disables; typical ≈mu_f, e.g. 1e-3)."
        ),
    )
    ap.add_argument(
        "--adaptive-gamma-div",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "At dt==dt_min, if a line-search failure is dominated by the momentum/skeleton block, "
            "increase gamma_div and retry the same step from the current best iterate."
        ),
    )
    ap.add_argument(
        "--gamma-div-max",
        type=float,
        default=float("nan"),
        help="Upper cap for adaptive gamma_div. Default: max(--gamma-div, 1e-1) when adaptation is enabled.",
    )
    ap.add_argument(
        "--gamma-div-growth",
        type=float,
        default=2.0,
        help="Multiplicative growth factor used when adaptive gamma_div is triggered.",
    )
    ap.add_argument(
        "--gamma-div-relax-factor",
        type=float,
        default=0.5,
        help="After enough easy accepted steps, relax adaptive gamma_div by this factor toward the baseline value.",
    )
    ap.add_argument(
        "--gamma-div-relax-after",
        type=int,
        default=3,
        help="Number of consecutive easy accepted steps required before relaxing adaptive gamma_div.",
    )
    ap.add_argument(
        "--gamma-div-relax-newton-max",
        type=int,
        default=2,
        help="Accepted steps with nNewton <= this threshold count as easy for adaptive gamma_div relaxation.",
    )
    ap.add_argument(
        "--fluid-convection",
        type=str,
        default="full",
        choices=("full", "lagged", "imex", "off"),
        help=(
            "How to treat fluid convection: 'full' (default) is fully nonlinear, "
            "'lagged' is Picard/Oseen-like (linear in v^k with n-level advector), "
            "'imex' treats convection explicitly at the n-level, "
            "'off' is Stokes/Brinkman."
        ),
    )
    ap.add_argument(
        "--fluid-space",
        type=str,
        default="cg",
        choices=("cg", "hdiv"),
        help="Fluid velocity space. 'hdiv' uses a single RT field for v.",
    )
    ap.add_argument(
        "--fluid-hdiv-order",
        type=int,
        default=0,
        help="RT order used when --fluid-space=hdiv.",
    )
    ap.add_argument(
        "--hdiv-tangential-dirichlet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --fluid-space=hdiv, keep the normal velocity Dirichlet data strong on the RT trace "
            "and impose the tangential Dirichlet data weakly in the momentum equation."
        ),
    )
    ap.add_argument(
        "--hdiv-tangential-gamma",
        type=float,
        default=20.0,
        help="Penalty parameter used for the weak tangential H(div) Dirichlet term.",
    )
    ap.add_argument(
        "--hdiv-tangential-method",
        type=str,
        default="penalty",
        choices=("penalty", "nitsche"),
        help="Weak tangential H(div) boundary formulation: penalty-only or fully consistent symmetric Nitsche.",
    )
    ap.add_argument(
        "--u-supg",
        type=float,
        default=0.0,
        help="SUPG-like streamline diffusion for kinematic u-transport (0 disables; typical 0.1–10).",
    )
    ap.add_argument(
        "--u-cip",
        type=float,
        default=0.0,
        help="CIP stabilization strength for the kinematic u-transport equation (0 disables).",
    )
    ap.add_argument(
        "--u-cip-weight",
        type=str,
        default="biofilm",
        choices=("fluid", "biofilm", "both"),
        help="Localization used by --u-cip: outside biofilm, inside biofilm, or both.",
    )
    ap.add_argument("--v-cip", type=float, default=0.0, help="CIP stabilization strength for fluid velocity (0 disables).")
    ap.add_argument("--vS-cip", type=float, default=0.0, help="CIP stabilization strength for skeleton velocity (0 disables).")

    # Transport (alpha/phi)
    ap.add_argument(
        "--transport-mode",
        type=str,
        default="pde",
        choices=("refmap", "pde"),
        help=(
            "How alpha/phi are evolved: 'refmap' updates alpha(x,t)=alpha0(x-u(x,t)) and ties phi to alpha; "
            "'pde' (default) solves the alpha/phi transport PDEs."
        ),
    )
    ap.add_argument("--D-phi", type=float, default=0.0, help="Porosity diffusion coefficient D_phi (pde mode).")
    ap.add_argument("--gamma-phi", type=float, default=5.0, help="Penalty enforcing phi->1 in free fluid (pde mode).")
    ap.add_argument("--phi-supg", type=float, default=0.0, help="SUPG stabilization strength for phi advection (pde mode).")
    ap.add_argument("--phi-cip", type=float, default=0.0, help="CIP stabilization strength for phi advection (pde mode).")
    ap.add_argument("--D-alpha", type=float, default=0.0, help="Diffusion coefficient D_alpha for alpha PDE (pde mode).")
    ap.add_argument("--alpha-supg", type=float, default=0.0, help="SUPG stabilization strength for alpha advection (pde mode).")
    ap.add_argument("--alpha-cip", type=float, default=0.0, help="CIP stabilization strength for alpha advection (pde mode).")
    ap.add_argument(
        "--alpha-advect-with",
        type=str,
        default="biofilm_volume",
        choices=("vS", "v", "mix", "mix_biofilm", "biofilm_volume", "relative", "interface"),
        help=(
            "Which velocity advects alpha in the alpha PDE (pde mode). "
            "For the support-preserving alpha law, use 'biofilm_volume' so alpha is "
            "transported by the conserved biofilm-support flux."
        ),
    )
    ap.add_argument(
        "--alpha-advection-form",
        type=str,
        default="conservative_weak",
        choices=("advective", "conservative", "conservative_weak"),
        help=(
            "Form of alpha advection by the chosen velocity in the alpha PDE (pde mode). "
            "The recommended physical setup is 'conservative_weak'."
        ),
    )
    ap.add_argument(
        "--support-physics",
        type=str,
        default="internal_conversion",
        choices=("legacy_exchange", "internal_conversion"),
        help=(
            "Biofilm support model. 'internal_conversion' preserves total alpha and evolves phi through "
            "the conservative solid-volume balance B=alpha(1-phi)."
        ),
    )
    ap.add_argument(
        "--alpha-mix-gate-alpha0",
        type=float,
        default=0.1,
        help=(
            "Gate parameter alpha0 used when --alpha-advect-with mix_biofilm. "
            "The fluid part is weighted by g(alpha)=alpha^m/(alpha^m+alpha0^m)."
        ),
    )
    ap.add_argument(
        "--alpha-mix-gate-power",
        type=int,
        default=4,
        help="Gate power m used when --alpha-advect-with mix_biofilm.",
    )
    ap.add_argument("--alpha-ch-M", type=float, default=1.0e-12, help="Cahn–Hilliard mobility M for alpha (0 disables CH).")
    ap.add_argument("--alpha-ch-gamma", type=float, default=2.0e-3, help="Cahn–Hilliard gamma for alpha (0 disables CH).")
    ap.add_argument(
        "--alpha-ch-eps",
        type=float,
        default=float("nan"),
        help="Cahn–Hilliard eps (interface thickness). Default: use --eps.",
    )
    ap.add_argument(
        "--scale-alpha-ch-eps-with-zeta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Increase the effective alpha CH eps with the benchmark-local diffuse traction scale "
            "to keep zeta^2/(gamma_alpha eps_alpha) from growing unchecked."
        ),
    )
    ap.add_argument(
        "--alpha-ch-mobility",
        type=str,
        default="degenerate",
        choices=("constant", "degenerate"),
        help="Mobility model for Cahn–Hilliard alpha regularization.",
    )
    ap.add_argument(
        "--alpha-phi-vi-bounds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use PETSc SNES VI bounds to enforce alpha,phi ∈ [0,1] in --transport-mode pde.",
    )

    # Alpha initialization
    ap.add_argument(
        "--alpha0-file",
        type=str,
        default="examples/biofilms/benchmarks/blauert/exp_frame0_polygon_mm.csv",
        help="Closed polygon CSV (x_mm,y_mm) for alpha0.",
    )
    ap.add_argument("--alpha0-scale", type=float, default=1.0e-3, help="Scale applied to polygon coordinates (mm->m default).")
    ap.add_argument("--alpha0-tx", type=float, default=5.0e-4, help="Translation in x applied to polygon [m].")
    ap.add_argument("--alpha0-ty", type=float, default=0.0, help="Translation in y applied to polygon [m].")
    ap.add_argument("--eps", type=float, default=2.0e-5, help="Diffuse interface thickness eps [m].")

    # Skeleton DOF restriction
    ap.add_argument("--restrict-skeleton-dofs", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--restrict-box-pad", type=float, default=5.0e-4, help="Padding [m] added to biofilm bbox for active (u,vS) box.")
    ap.add_argument("--restrict-box-x0", type=float, default=float("nan"))
    ap.add_argument("--restrict-box-x1", type=float, default=float("nan"))
    ap.add_argument("--restrict-box-y0", type=float, default=float("nan"))
    ap.add_argument("--restrict-box-y1", type=float, default=float("nan"))
    ap.add_argument(
        "--interface-reg-band-factor",
        type=float,
        default=0.0,
        help=(
            "If > 0, localize u/vS extension and CIP regularization to a lagged "
            "interface band |d(alpha)| <= factor * eps, with the outside weight "
            "set by --interface-reg-weight-floor."
        ),
    )
    ap.add_argument(
        "--interface-reg-weight-floor",
        type=float,
        default=1.0,
        help="Background regularization weight outside the interface band when --interface-reg-band-factor > 0.",
    )

    # Tracking output
    ap.add_argument(
        "--track-y-um",
        type=str,
        default="150,250,350",
        help="Comma-separated y-levels [um] at which to track x_front via alpha=0.5 contour.",
    )
    ap.add_argument(
        "--dx-intersection",
        type=str,
        default="leftmost",
        choices=("leftmost", "rightmost"),
        help="Which alpha=0.5 intersection to track on each y-line.",
    )
    ap.add_argument(
        "--dx-quantile",
        type=float,
        default=0.1,
        help="Interior quantile between left/right intersections (0=boundary, 0.1=slightly inside).",
    )
    ap.add_argument(
        "--global-front-quantile",
        type=float,
        default=0.005,
        help="Quantile for global x_front over row-wise leftmost contour fronts (0=min).",
    )
    ap.add_argument(
        "--snapshot-times",
        type=str,
        default="",
        help="Comma-separated times [s] at which to export raw alpha=0.5 contour CSVs.",
    )
    ap.add_argument("--vtk-every", type=int, default=0)
    ap.add_argument(
        "--interface-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write interface_diagnostics.csv with sampled alpha=0.5 contour pressure/traction diagnostics.",
    )
    ap.add_argument(
        "--interface-diagnostics-max-points",
        type=int,
        default=300,
        help="Maximum number of alpha=0.5 contour points sampled per output row for --interface-diagnostics.",
    )
    ap.add_argument("--out-dir", type=str, default="out/_blauert_one_domain")
    ap.add_argument(
        "--restart-from",
        type=str,
        default="",
        help="Path to a restart checkpoint (.npz) written by --restart-write-every. If set, resumes from that state.",
    )
    ap.add_argument(
        "--restart-write-every",
        type=int,
        default=0,
        help="Write restart checkpoint every N accepted time steps (0 disables).",
    )
    ap.add_argument(
        "--restart-dt",
        type=float,
        default=float("nan"),
        help="Override dt when resuming from --restart-from (default: use dt stored in the checkpoint).",
    )
    ap.add_argument(
        "--restart-dir",
        type=str,
        default="",
        help="Directory for restart checkpoints (default: <out_dir>/restart).",
    )
    ap.add_argument("--trace-residual-fields", action="store_true", help="Print per-field |R|_inf during Newton/SNES iterations.")
    ap.add_argument("--trace-residual-fields-n", type=int, default=8, help="How many fields to show with --trace-residual-fields.")
    ap.add_argument("--trace-residual-worst", action="store_true", help="Print worst-DOF residual diagnostics during Newton/SNES.")
    ap.add_argument("--trace-residual-coords", action="store_true", help="With --trace-residual-worst, also print coordinates of worst DOF when available.")

    args = ap.parse_args()

    if bool(args.trace_residual_fields):
        os.environ["PYCUTFEM_NEWTON_TRACE_RES_FIELDS"] = "1"
        os.environ["PYCUTFEM_NEWTON_TRACE_RES_FIELDS_N"] = str(max(1, int(args.trace_residual_fields_n)))
    if bool(args.trace_residual_worst):
        os.environ["PYCUTFEM_RESIDUAL_TRACE"] = "1"
        if bool(args.trace_residual_coords):
            os.environ["PYCUTFEM_RESIDUAL_TRACE_COORDS"] = "1"
    if str(args.linear_ksp_type).strip():
        os.environ["PYCUTFEM_LINEAR_KSP_TYPE"] = str(args.linear_ksp_type).strip()
    if str(args.linear_pc_type).strip():
        os.environ["PYCUTFEM_LINEAR_PC_TYPE"] = str(args.linear_pc_type).strip()
    if str(args.linear_pc_factor_solver_type).strip():
        os.environ["PYCUTFEM_LINEAR_PC_FACTOR_SOLVER_TYPE"] = str(args.linear_pc_factor_solver_type).strip()
    if np.isfinite(float(args.linear_ksp_rtol)):
        os.environ["PYCUTFEM_LINEAR_KSP_RTOL"] = str(float(args.linear_ksp_rtol))
    if np.isfinite(float(args.linear_ksp_atol)):
        os.environ["PYCUTFEM_LINEAR_KSP_ATOL"] = str(float(args.linear_ksp_atol))
    if np.isfinite(float(args.linear_ksp_dtol)):
        os.environ["PYCUTFEM_LINEAR_KSP_DTOL"] = str(float(args.linear_ksp_dtol))
    if int(args.linear_ksp_max_it) > 0:
        os.environ["PYCUTFEM_LINEAR_KSP_MAX_IT"] = str(int(args.linear_ksp_max_it))
    if bool(args.linear_ksp_trace):
        os.environ["PYCUTFEM_LINEAR_KSP_TRACE"] = "1"
    if bool(args.linear_schur):
        os.environ["PYCUTFEM_LINEAR_SCHUR"] = "1"
        os.environ["PYCUTFEM_LINEAR_SCHUR_PRESSURE_FIELD"] = str(args.linear_schur_pressure_field)
        os.environ["PYCUTFEM_LINEAR_SCHUR_FACT_TYPE"] = str(args.linear_schur_fact)
        os.environ["PYCUTFEM_LINEAR_SCHUR_PRECONDITION"] = str(args.linear_schur_pre)
        os.environ["PYCUTFEM_LINEAR_SCHUR_REST_KSP_TYPE"] = str(args.linear_schur_rest_ksp)
        os.environ["PYCUTFEM_LINEAR_SCHUR_REST_PC_TYPE"] = str(args.linear_schur_rest_pc)
        if str(args.linear_schur_rest_factor_solver_type).strip():
            os.environ["PYCUTFEM_LINEAR_SCHUR_REST_FACTOR_SOLVER_TYPE"] = str(args.linear_schur_rest_factor_solver_type)
        os.environ["PYCUTFEM_LINEAR_SCHUR_PRESSURE_KSP_TYPE"] = str(args.linear_schur_pressure_ksp)
        os.environ["PYCUTFEM_LINEAR_SCHUR_PRESSURE_PC_TYPE"] = str(args.linear_schur_pressure_pc)
        if str(args.linear_schur_pressure_factor_solver_type).strip():
            os.environ["PYCUTFEM_LINEAR_SCHUR_PRESSURE_FACTOR_SOLVER_TYPE"] = str(args.linear_schur_pressure_factor_solver_type)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paper1_reduced = bool(getattr(args, "paper1_reduced", False))
    transport_mode = str(getattr(args, "transport_mode", "refmap")).strip().lower()
    if transport_mode not in {"refmap", "pde"}:
        raise ValueError(f"Unknown --transport-mode {transport_mode!r}.")

    alpha_ch_M = float(getattr(args, "alpha_ch_M", 0.0))
    alpha_ch_gamma = float(getattr(args, "alpha_ch_gamma", 0.0))
    ch_requested = bool(alpha_ch_M != 0.0 and alpha_ch_gamma != 0.0)
    if ch_requested and transport_mode != "pde":
        logging.warning("Ignoring --alpha-ch-* because --transport-mode=%s.", transport_mode)
        alpha_ch_M = 0.0
        alpha_ch_gamma = 0.0
    ch_enabled = bool(alpha_ch_M != 0.0 and alpha_ch_gamma != 0.0)
    if paper1_reduced and transport_mode != "pde":
        raise ValueError("--paper1-reduced requires --transport-mode pde.")
    if paper1_reduced and not ch_enabled:
        raise ValueError("--paper1-reduced requires nonzero --alpha-ch-M and --alpha-ch-gamma.")
    alpha_ch_eps_base = float(getattr(args, "alpha_ch_eps", float("nan")))
    if not np.isfinite(alpha_ch_eps_base):
        alpha_ch_eps_base = float(getattr(args, "eps", 1.0))
    scale_alpha_ch_eps_with_zeta = bool(getattr(args, "scale_alpha_ch_eps_with_zeta", False)) and bool(
        getattr(args, "diffuse_shear_traction", False)
    )
    diffuse_shear_scale_ref = float(getattr(args, "diffuse_shear_scale_ref", 50.0))
    alpha_ch_eps_scale_factor = 1.0
    if scale_alpha_ch_eps_with_zeta:
        if (not np.isfinite(diffuse_shear_scale_ref)) or diffuse_shear_scale_ref <= 0.0:
            raise ValueError("--diffuse-shear-scale-ref must be positive when --scale-alpha-ch-eps-with-zeta is enabled.")
        alpha_ch_eps_scale_factor = max(
            1.0,
            (abs(float(getattr(args, "diffuse_shear_scale", 0.0))) / float(diffuse_shear_scale_ref)) ** 2,
        )
    alpha_ch_eps = float(alpha_ch_eps_base) * float(alpha_ch_eps_scale_factor)

    use_alpha_phi_vi_bounds = bool(
        (transport_mode == "pde")
        and (not paper1_reduced)
        and bool(getattr(args, "alpha_phi_vi_bounds", True))
    )

    if (transport_mode == "pde") and (not paper1_reduced):
        D_phi_val = float(getattr(args, "D_phi", 0.0))
        gamma_phi_val = float(getattr(args, "gamma_phi", 0.0))
        if abs(D_phi_val) < 1.0e-30 and abs(gamma_phi_val) < 1.0e-30:
            raise ValueError("--transport-mode pde requires --gamma-phi>0 (recommended) or --D-phi>0 so phi is well-posed in the fluid.")

    L = float(args.L)
    H = float(args.H)
    if not (L > 0.0 and H > 0.0):
        raise ValueError("--L and --H must be positive.")

    restart_from = Path(str(args.restart_from)) if str(args.restart_from).strip() else None
    restart_meta: dict[str, float | int] | None = None
    if restart_from is not None:
        restart_meta = _restart_peek(restart_from)

    gamma_div_base = float(args.gamma_div)
    adaptive_gamma_div = bool(getattr(args, "adaptive_gamma_div", False))
    gamma_div_restart = float(restart_meta.get("gamma_div", float("nan"))) if restart_meta is not None else float("nan")
    gamma_div_init = gamma_div_base
    if adaptive_gamma_div and np.isfinite(gamma_div_restart):
        gamma_div_init = float(gamma_div_restart)
    gamma_div_max = float(getattr(args, "gamma_div_max", float("nan")))
    if not np.isfinite(gamma_div_max):
        gamma_div_max = max(float(gamma_div_base), 1.0e-1) if adaptive_gamma_div else float(gamma_div_base)
    gamma_div_max = max(float(gamma_div_max), float(gamma_div_init), float(gamma_div_base))
    gamma_div_growth = max(1.1, float(getattr(args, "gamma_div_growth", 2.0)))
    gamma_div_relax_factor = min(1.0, max(0.0, float(getattr(args, "gamma_div_relax_factor", 0.5))))
    gamma_div_relax_after = max(1, int(getattr(args, "gamma_div_relax_after", 3)))
    gamma_div_relax_newton_max = max(1, int(getattr(args, "gamma_div_relax_newton_max", 2)))
    gamma_div_expr = _named_scalar_expr("gamma_div", gamma_div_init)
    flow_shutoff_start = float(getattr(args, "flow_shutoff_start", float("nan")))
    flow_shutoff_end = float(getattr(args, "flow_shutoff_end", float("nan")))
    flow_shutoff_end_factor = float(getattr(args, "flow_shutoff_end_factor", 0.0))
    flow_shutoff_start_finite = np.isfinite(flow_shutoff_start)
    flow_shutoff_end_finite = np.isfinite(flow_shutoff_end)
    if flow_shutoff_start_finite != flow_shutoff_end_finite:
        raise ValueError("Use --flow-shutoff-start and --flow-shutoff-end together.")
    if flow_shutoff_start_finite and not (flow_shutoff_end > flow_shutoff_start):
        raise ValueError("--flow-shutoff-end must be greater than --flow-shutoff-start.")
    if not np.isfinite(flow_shutoff_end_factor) or flow_shutoff_end_factor < 0.0:
        raise ValueError("--flow-shutoff-end-factor must be finite and nonnegative.")

    def _late_flow_shutoff_factor(t_now: float) -> float:
        if not flow_shutoff_start_finite:
            return 1.0
        return _cosine_blend_value(
            float(t_now),
            start_time=float(flow_shutoff_start),
            end_time=float(flow_shutoff_end),
            start_value=1.0,
            end_value=float(flow_shutoff_end_factor),
        )

    # ------------------------------------------------------------------
    # Polygon (alpha0)
    # ------------------------------------------------------------------
    poly_mm = _read_polygon_mm_csv(str(args.alpha0_file))
    poly_m = poly_mm * float(args.alpha0_scale)
    poly_m = poly_m + np.array([float(args.alpha0_tx), float(args.alpha0_ty)], dtype=float)
    if float(np.min(poly_m[:, 1])) < -1.0e-12:
        raise ValueError("alpha0 polygon has negative y after translation; expected y>=0.")
    poly_min = np.min(poly_m[:, :2], axis=0)
    poly_max = np.max(poly_m[:, :2], axis=0)
    logging.info(
        "[setup] alpha0 bbox: "
        f"x=[{float(poly_min[0]) * 1.0e3:.3f},{float(poly_max[0]) * 1.0e3:.3f}]mm, "
        f"y=[{float(poly_min[1]) * 1.0e3:.3f},{float(poly_max[1]) * 1.0e3:.3f}]mm "
        f"(clearance_top={float(H - poly_max[1]) * 1.0e6:.1f}um)"
    )
    if float(poly_min[0]) < -1.0e-12 or float(poly_max[0]) > float(L) + 1.0e-12:
        raise ValueError(
            "alpha0 polygon is outside the channel in x after scaling/translation: "
            f"x=[{float(poly_min[0]):.6e},{float(poly_max[0]):.6e}] vs [0,L]=[0,{float(L):.6e}]. "
            "Adjust --alpha0-scale/--alpha0-tx or provide a polygon in the correct coordinate system."
        )
    if float(poly_max[1]) > float(H) + 1.0e-12:
        raise ValueError(
            "alpha0 polygon extends above the top wall after scaling/translation: "
            f"y_max={float(poly_max[1]):.6e} > H={float(H):.6e}. "
            "This typically indicates a wrong pixel-to-length scale or segmentation; "
            "fix the polygon or adjust --alpha0-scale/--alpha0-ty."
        )

    # ------------------------------------------------------------------
    # Mesh + tags
    # ------------------------------------------------------------------
    nodes, elems, _edges, corners = structured_quad(L, H, nx=int(args.nx), ny=int(args.ny), poly_order=2, offset=(0.0, 0.0))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(elems, dtype=int),
        elements_corner_nodes=np.asarray(corners, dtype=int),
        element_type="quad",
        poly_order=2,
    )
    _tag_channel_boundaries(mesh, L=L, H=H)
    if bool(args.refine_biofilm):
        hx = float(L) / float(max(1, int(args.nx)))
        hy = float(H) / float(max(1, int(args.ny)))
        band = float(args.refine_band) if np.isfinite(float(args.refine_band)) else 2.0 * max(hx, hy)
        mesh = refine_around_biofilm_bbox(
            mesh,
            poly=poly_m,
            band=band,
            expand_layers=int(args.refine_expand_layers),
            L=L,
            H=H,
        )

    # ------------------------------------------------------------------
    # Mixed space
    # ------------------------------------------------------------------
    fluid_space_key = str(getattr(args, "fluid_space", "cg")).strip().lower()
    fluid_hdiv_order = int(getattr(args, "fluid_hdiv_order", 0))
    if fluid_space_key not in {"cg", "hdiv"}:
        raise ValueError(f"Unsupported --fluid-space={args.fluid_space!r}.")
    field_specs = {
        "p": 1,
        "vS_x": 2,
        "vS_y": 2,
        "u_x": 2,
        "u_y": 2,
        "alpha": 1,
        "mu_alpha": 1,
    }
    if fluid_space_key == "cg":
        field_specs = {"v_x": 2, "v_y": 2, **field_specs}
    else:
        field_specs = {"v": ("RT", int(fluid_hdiv_order)), **field_specs}
    if not paper1_reduced:
        field_specs["phi"] = 1
        field_specs["S"] = 1
    me = MixedElement(mesh, field_specs=field_specs)
    dh = DofHandler(me, method="cg")

    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    if fluid_space_key == "cg":
        V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
        dv = VectorTrialFunction(space=V, dof_handler=dh)
        v_test = VectorTestFunction(space=V, dof_handler=dh)
        v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
        v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    else:
        dv = HdivTrialFunction("v")
        v_test = HdivTestFunction("v")
        v_k = HdivFunction("v_k", "v", dof_handler=dh)
        v_n = HdivFunction("v_n", "v", dof_handler=dh)
    dvS = VectorTrialFunction(space=VS, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh) if not paper1_reduced else None
    dalpha = TrialFunction("alpha", dof_handler=dh)
    dmu_alpha = TrialFunction("mu_alpha", dof_handler=dh)
    dS = TrialFunction("S", dof_handler=dh) if not paper1_reduced else None

    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh) if not paper1_reduced else None
    alpha_test = TestFunction("alpha", dof_handler=dh)
    mu_alpha_test = TestFunction("mu_alpha", dof_handler=dh)
    S_test = TestFunction("S", dof_handler=dh) if not paper1_reduced else None

    p_k = Function("p_k", "p", dof_handler=dh)
    vS_k = VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh) if not paper1_reduced else None
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    mu_alpha_k = Function("mu_alpha_k", "mu_alpha", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh) if not paper1_reduced else None

    p_n = Function("p_n", "p", dof_handler=dh)
    vS_n = VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh) if not paper1_reduced else None
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    mu_alpha_n = Function("mu_alpha_n", "mu_alpha", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh) if not paper1_reduced else None
    if fluid_space_key == "cg":
        v_trac_n = VectorFunction("v_trac_n", ["v_x", "v_y"], dof_handler=dh)
    else:
        v_trac_n = HdivFunction("v_trac_n", "v", dof_handler=dh)
    p_trac_n = Function("p_trac_n", "p", dof_handler=dh)

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------
    v_n.nodal_values[:] = 0.0
    p_n.nodal_values[:] = 0.0
    v_trac_n.nodal_values[:] = 0.0
    p_trac_n.nodal_values[:] = 0.0
    vS_n.nodal_values[:] = 0.0
    u_n.nodal_values[:] = 0.0
    mu_alpha_n.nodal_values[:] = 0.0
    if S_n is not None:
        S_n.nodal_values[:] = 0.0

    eps0 = float(args.eps)
    if ch_enabled and transport_mode == "pde" and bool(scale_alpha_ch_eps_with_zeta):
        eps0 = max(float(eps0), float(alpha_ch_eps))
    alpha_xy = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    alpha0 = _smooth_step((-_signed_distance_polygon(alpha_xy[:, 0], alpha_xy[:, 1], poly_m)) / max(1.0e-12, eps0))
    alpha_n.nodal_values[:] = np.clip(np.asarray(alpha0, dtype=float), 0.0, 1.0)

    phi_b = float(args.phi_b)
    if not (0.0 < phi_b < 1.0):
        raise ValueError("--phi-b must be in (0,1).")
    phi_init_mode = str(getattr(args, "phi_init_mode", "linear_alpha"))
    if phi_n is not None:
        phi_n.nodal_values[:] = _phi_init_from_alpha(
            np.asarray(alpha_n.nodal_values, dtype=float),
            phi_b=float(phi_b),
            mode=str(phi_init_mode),
        )

    if transport_mode == "refmap":
        # Refmap mode: freeze transport fields and update alpha/phi after each accepted step.
        _mark_inactive_fields(dh, "alpha", "phi", "S", "mu_alpha")
    elif paper1_reduced:
        # Reduced Paper-1 mode: solve alpha/mu_alpha and keep only the reduced unknown set active.
        a0 = np.asarray(alpha_n.nodal_values, dtype=float)
        Wp = 2.0 * a0 * (1.0 - a0) * (1.0 - 2.0 * a0)
        mu_alpha_n.nodal_values[:] = float(alpha_ch_gamma / max(alpha_ch_eps, 1.0e-12)) * Wp
    else:
        # PDE mode: solve alpha/phi. We never solve substrate in this benchmark (no growth/chemistry).
        _mark_inactive_fields(dh, "S")
        if not ch_enabled:
            # CH disabled -> mu_alpha is not part of the residual, so keep it inactive to avoid a singular block.
            _mark_inactive_fields(dh, "mu_alpha")
        else:
            # Initialize a consistent-ish μ guess (drop the -εΔα term to avoid extra projections).
            a0 = np.asarray(alpha_n.nodal_values, dtype=float)
            Wp = 2.0 * a0 * (1.0 - a0) * (1.0 - 2.0 * a0)
            mu_alpha_n.nodal_values[:] = float(alpha_ch_gamma / max(alpha_ch_eps, 1.0e-12)) * Wp

    # Restrict skeleton DOFs to a fixed rectangle around the initial biofilm.
    if bool(args.restrict_skeleton_dofs):
        pad = float(args.restrict_box_pad)
        x0_auto = float(np.min(poly_m[:, 0])) - pad
        x1_auto = float(np.max(poly_m[:, 0])) + pad
        y0_auto = float(np.min(poly_m[:, 1])) - pad
        y1_auto = float(np.max(poly_m[:, 1])) + pad
        x0 = x0_auto if not np.isfinite(float(args.restrict_box_x0)) else float(args.restrict_box_x0)
        x1 = x1_auto if not np.isfinite(float(args.restrict_box_x1)) else float(args.restrict_box_x1)
        y0 = y0_auto if not np.isfinite(float(args.restrict_box_y0)) else float(args.restrict_box_y0)
        y1 = y1_auto if not np.isfinite(float(args.restrict_box_y1)) else float(args.restrict_box_y1)
        # Clamp to domain.
        x0 = max(0.0, x0)
        x1 = min(L, x1)
        y0 = max(0.0, y0)
        y1 = min(H, y1)
        n_keep, n_tot = _restrict_skeleton_dofs_to_box(dh, x0=x0, x1=x1, y0=y0, y1=y1)
        logging.info(f"[setup] restricted (u,vS) DOFs to box: keep {n_keep}/{n_tot} Q2 nodes")

    interface_reg_band_factor = float(getattr(args, "interface_reg_band_factor", 0.0))
    interface_reg_weight_floor = float(getattr(args, "interface_reg_weight_floor", 1.0))
    regularization_weight = None

    def _update_regularization_weight(alpha_vals_now) -> None:
        if regularization_weight is None:
            return
        regularization_weight.nodal_values[:] = _interface_band_reg_weight(
            np.asarray(alpha_vals_now, dtype=float),
            eps=float(alpha_ch_eps),
            band_factor=float(interface_reg_band_factor),
            floor=float(interface_reg_weight_floor),
        )

    if interface_reg_band_factor > 0.0:
        regularization_weight = Function("reg_weight", "alpha", dof_handler=dh)
        _update_regularization_weight(alpha_n.nodal_values)
        logging.info(
            "[setup] interface-band regularization enabled: band_factor=%.3f, eps=%.3e, band_halfwidth=%.3e, floor=%.3f",
            float(interface_reg_band_factor),
            float(alpha_ch_eps),
            float(interface_reg_band_factor) * float(alpha_ch_eps),
            float(interface_reg_weight_floor),
        )

    # ------------------------------------------------------------------
    # Alpha-from-refmap mapping: alpha(x,t) = alpha0(x - u(x,t))
    # Build alpha->u DOF lookup via coordinate matching (Q1 alpha is a subset of Q2 u).
    # ------------------------------------------------------------------
    if transport_mode == "refmap":
        ux_xy = np.asarray(dh.get_dof_coords("u_x"), dtype=float)
        ux_lut = _build_coord_lookup(ux_xy, ndigits=12)
        alpha_to_u = np.full(alpha_xy.shape[0], -1, dtype=int)
        for i, xy in enumerate(alpha_xy):
            key = (round(float(xy[0]), 12), round(float(xy[1]), 12))
            j = ux_lut.get(key, None)
            if j is not None:
                alpha_to_u[i] = int(j)
        if np.any(alpha_to_u < 0):
            missing = int(np.sum(alpha_to_u < 0))
            raise RuntimeError(f"Failed mapping {missing} alpha DOFs to u DOFs; cannot use refmap alpha transport.")

        def _post_step_update() -> None:
            ux = np.asarray(u_k.components[0].nodal_values, dtype=float).ravel()
            uy = np.asarray(u_k.components[1].nodal_values, dtype=float).ravel()
            u_at_alpha = np.column_stack([ux[alpha_to_u], uy[alpha_to_u]])
            chi = alpha_xy - u_at_alpha
            a = _smooth_step((-_signed_distance_polygon(chi[:, 0], chi[:, 1], poly_m)) / max(1.0e-12, eps0))
            alpha_k.nodal_values[:] = np.clip(np.asarray(a, dtype=float), 0.0, 1.0)
            if phi_k is not None:
                phi_k.nodal_values[:] = _phi_init_from_alpha(
                    np.asarray(alpha_k.nodal_values, dtype=float),
                    phi_b=float(phi_b),
                    mode=str(phi_init_mode),
                )
            _update_regularization_weight(alpha_k.nodal_values)

    else:

        def _post_step_update() -> None:
            # Nothing to do in PDE mode; alpha/phi are solved.
            _update_regularization_weight(alpha_k.nodal_values)
            return

    hdiv_tangential_bc_measure = None
    if fluid_space_key == "hdiv" and bool(getattr(args, "hdiv_tangential_dirichlet", True)):
        hdiv_tangential_bc_measure = dS_measure(
            defined_on=(
                mesh.edge_bitset("left")
                | mesh.edge_bitset("bottom")
                | mesh.edge_bitset("top")
            ),
            metadata={"q": int(args.q)},
        )

    # ------------------------------------------------------------------
    # Forms
    # ------------------------------------------------------------------
    if restart_meta is not None:
        dt_val = float(restart_meta["dt"])
        if np.isfinite(float(args.restart_dt)):
            dt_val = float(args.restart_dt)
            logging.info(f"[restart] overriding dt -> {dt_val:.3e} (checkpoint dt was {float(restart_meta['dt']):.3e})")
        t0 = float(restart_meta["t"])
        step0 = int(restart_meta["step"])
        theta0 = float(restart_meta.get("theta", float(args.theta)))
        if abs(theta0 - float(args.theta)) > 1.0e-12:
            logging.warning(f"[restart] checkpoint theta={theta0:.6g} differs from --theta={float(args.theta):.6g}; using --theta.")
    else:
        dt_val = float(args.dt)
        t0 = 0.0
        step0 = 0
    dt_c = _named_scalar_expr("dt", dt_val)
    theta = float(args.theta)
    attachment_mode = str(getattr(args, "attachment_mode", "clamped")).strip().lower()
    use_bottom_adhesion = attachment_mode == "adhesion"
    adhesion_k_n = float(getattr(args, "adhesion_k_n", 0.0))
    adhesion_k_t = float(getattr(args, "adhesion_k_t", 0.0))
    adhesion_gamma_n = float(getattr(args, "adhesion_gamma_n", 0.0))
    adhesion_gamma_t = float(getattr(args, "adhesion_gamma_t", 0.0))
    if use_bottom_adhesion and paper1_reduced:
        raise RuntimeError("--attachment-mode adhesion is only supported in the full one-domain benchmark path; disable --paper1-reduced.")
    if use_bottom_adhesion:
        adhesion_scale = max(abs(adhesion_k_n), abs(adhesion_k_t), abs(adhesion_gamma_n), abs(adhesion_gamma_t))
        if adhesion_scale <= 0.0:
            raise RuntimeError(
                "--attachment-mode adhesion requires at least one nonzero coefficient among "
                "--adhesion-k-n/--adhesion-k-t/--adhesion-gamma-n/--adhesion-gamma-t."
            )
    kappa_inv_model = str(getattr(args, "kappa_inv_model", "spatial")).strip().lower()
    kappa_phi_ref = float(getattr(args, "kappa_phi_ref", None) or args.phi_b)

    mu_s, lambda_s = _lame_from_E_nu(float(args.E), float(args.nu))
    rho_f_c = _named_scalar_expr("rho_f", float(args.rho_f))
    rho_s0_tilde_c = _named_scalar_expr("rho_s0_tilde", float(args.rho_s0_tilde))
    mu_f_c = _named_scalar_expr("mu_f", float(args.mu_f))
    mu_b_c = _named_scalar_expr("mu_b", float(args.mu_f))
    kappa_inv_c = _named_scalar_expr("kappa_inv", float(args.kappa_inv))
    mu_s_c = _named_scalar_expr("mu_s", float(mu_s))
    lambda_s_c = _named_scalar_expr("lambda_s", float(lambda_s))
    solid_visco_eta_c = _named_scalar_expr("solid_visco_eta", float(args.solid_visco_eta))
    gamma_u_c = _named_scalar_expr("gamma_u", float(args.gamma_u))
    gamma_u_pin_c = _named_scalar_expr("gamma_u_pin", float(args.gamma_u_pin))
    v_supg_c = _named_scalar_expr("v_supg", float(args.v_supg))
    u_supg_c = _named_scalar_expr("u_supg", float(args.u_supg))
    u_cip_c = _named_scalar_expr("u_cip", float(args.u_cip))
    v_cip_c = _named_scalar_expr("v_cip", float(args.v_cip))
    vS_cip_c = _named_scalar_expr("vS_cip", float(args.vS_cip))
    D_phi_c = _named_scalar_expr("D_phi", float(args.D_phi))
    gamma_phi_c = _named_scalar_expr("gamma_phi", float(args.gamma_phi))
    D_alpha_c = _named_scalar_expr("D_alpha", float(args.D_alpha))
    alpha_ch_M_c = _named_scalar_expr("alpha_ch_M", float(alpha_ch_M))
    alpha_ch_gamma_c = _named_scalar_expr("alpha_ch_gamma", float(alpha_ch_gamma))
    alpha_ch_eps_c = _named_scalar_expr("alpha_ch_eps", float(alpha_ch_eps))
    diffuse_g_t = None
    diffuse_w_t = None
    diffuse_scale_expr = None
    diffuse_pressure_scale_expr = None
    diffuse_normal_stress_scale_expr = None
    diffuse_scale_update = None
    if bool(getattr(args, "diffuse_shear_traction", False)):
        diffuse_time_scheme = str(getattr(args, "diffuse_shear_time_scheme", "constant")).strip().lower()
        if diffuse_time_scheme not in {"constant", "imex"}:
            raise ValueError(f"Unknown --diffuse-shear-time-scheme {diffuse_time_scheme!r}.")
        diffuse_ramp_time = float(getattr(args, "diffuse_shear_ramp_time", float("nan")))
        if not np.isfinite(diffuse_ramp_time):
            diffuse_ramp_time = float(args.t_ramp)
        target_diffuse_scale = float(args.diffuse_shear_scale)
        target_diffuse_pressure_scale = float(getattr(args, "diffuse_normal_pressure_scale", 0.0))
        target_diffuse_normal_stress_scale = float(getattr(args, "diffuse_normal_stress_scale", 0.0))
        diffuse_shear_topweight = bool(getattr(args, "diffuse_shear_topweight", False))
        diffuse_pressure_xweight = bool(getattr(args, "diffuse_normal_pressure_xweight", False))
        diffuse_pressure_upstream_only = bool(getattr(args, "diffuse_normal_pressure_upstream_only", False))
        normal_stress_ramp_time = float(getattr(args, "diffuse_normal_stress_ramp_time", float("nan")))
        if not np.isfinite(normal_stress_ramp_time):
            normal_stress_ramp_time = float(diffuse_ramp_time)
        normal_stress_decay_start = float(getattr(args, "diffuse_normal_stress_decay_start", float("nan")))
        normal_stress_decay_end = float(getattr(args, "diffuse_normal_stress_decay_end", float("nan")))
        normal_stress_tail_factor = float(getattr(args, "diffuse_normal_stress_tail_factor", 1.0))
        decay_start_finite = np.isfinite(normal_stress_decay_start)
        decay_end_finite = np.isfinite(normal_stress_decay_end)
        normal_stress_topbias = float(getattr(args, "diffuse_normal_stress_topbias", 0.0))
        if not (0.0 <= normal_stress_topbias <= 1.0):
            raise ValueError("--diffuse-normal-stress-topbias must lie in [0, 1].")
        normal_stress_frontbias = float(getattr(args, "diffuse_normal_stress_frontbias", 0.0))
        if not (0.0 <= normal_stress_frontbias <= 1.0):
            raise ValueError("--diffuse-normal-stress-frontbias must lie in [0, 1].")
        normal_stress_frontbias_tail_value = float(
            getattr(args, "diffuse_normal_stress_frontbias_tail_value", float("nan"))
        )
        if np.isfinite(normal_stress_frontbias_tail_value) and not (
            0.0 <= normal_stress_frontbias_tail_value <= 1.0
        ):
            raise ValueError("--diffuse-normal-stress-frontbias-tail-value must lie in [0, 1].")
        normal_stress_bottomskew = float(getattr(args, "diffuse_normal_stress_bottomskew", 0.0))
        if not (0.0 <= normal_stress_bottomskew <= 1.0):
            raise ValueError("--diffuse-normal-stress-bottomskew must lie in [0, 1].")
        normal_stress_bottomskew_ramp_with_tail = bool(
            getattr(args, "diffuse_normal_stress_bottomskew_ramp_with_tail", False)
        )
        if normal_stress_bottomskew_ramp_with_tail and not decay_start_finite:
            raise ValueError(
                "--diffuse-normal-stress-bottomskew-ramp-with-tail requires "
                "--diffuse-normal-stress-decay-start and --diffuse-normal-stress-decay-end."
            )
        if np.isfinite(normal_stress_frontbias_tail_value) and not decay_start_finite:
            raise ValueError(
                "--diffuse-normal-stress-frontbias-tail-value requires "
                "--diffuse-normal-stress-decay-start and --diffuse-normal-stress-decay-end."
            )
        if decay_start_finite != decay_end_finite:
            raise ValueError(
                "Use --diffuse-normal-stress-decay-start and "
                "--diffuse-normal-stress-decay-end together."
            )
        if decay_start_finite and not (normal_stress_decay_end > normal_stress_decay_start):
            raise ValueError(
                "--diffuse-normal-stress-decay-end must be greater than "
                "--diffuse-normal-stress-decay-start."
            )
        diffuse_scale_expr = _named_scalar_expr("diffuse_shear_scale", target_diffuse_scale)
        diffuse_pressure_scale_expr = _named_scalar_expr("diffuse_normal_pressure_scale", target_diffuse_pressure_scale)
        diffuse_normal_stress_scale_expr = _named_scalar_expr(
            "diffuse_normal_stress_scale", target_diffuse_normal_stress_scale
        )
        diffuse_normal_stress_frontbias_expr = _named_scalar_expr(
            "diffuse_normal_stress_frontbias", normal_stress_frontbias
        )
        diffuse_normal_stress_bottomskew_expr = _named_scalar_expr(
            "diffuse_normal_stress_bottomskew", normal_stress_bottomskew
        )

        def _normal_stress_decay_factor(t_now: float) -> float:
            if not decay_start_finite:
                return 1.0
            return _cosine_blend_value(
                float(t_now),
                start_time=float(normal_stress_decay_start),
                end_time=float(normal_stress_decay_end),
                start_value=1.0,
                end_value=float(normal_stress_tail_factor),
            )

        if diffuse_time_scheme == "imex":
            def _update_diffuse_scale(t_now: float) -> None:
                ramp = _cosine_ramp_value(float(t_now), diffuse_ramp_time)
                normal_stress_ramp = _cosine_ramp_value(float(t_now), normal_stress_ramp_time)
                shutoff = _late_flow_shutoff_factor(float(t_now))
                diffuse_scale_expr.value = float(target_diffuse_scale) * ramp * shutoff
                diffuse_pressure_scale_expr.value = float(target_diffuse_pressure_scale) * ramp * shutoff
                diffuse_normal_stress_scale_expr.value = (
                    float(target_diffuse_normal_stress_scale)
                    * normal_stress_ramp
                    * _normal_stress_decay_factor(float(t_now))
                    * shutoff
                )
                if np.isfinite(normal_stress_frontbias_tail_value):
                    diffuse_normal_stress_frontbias_expr.value = _cosine_blend_value(
                        float(t_now),
                        start_time=float(normal_stress_decay_start),
                        end_time=float(normal_stress_decay_end),
                        start_value=float(normal_stress_frontbias),
                        end_value=float(normal_stress_frontbias_tail_value),
                    )
                else:
                    diffuse_normal_stress_frontbias_expr.value = float(normal_stress_frontbias)
                if normal_stress_bottomskew_ramp_with_tail:
                    diffuse_normal_stress_bottomskew_expr.value = (
                        float(normal_stress_bottomskew)
                        * _cosine_blend_value(
                            float(t_now),
                            start_time=float(normal_stress_decay_start),
                            end_time=float(normal_stress_decay_end),
                            start_value=0.0,
                            end_value=1.0,
                        )
                    )
                else:
                    diffuse_normal_stress_bottomskew_expr.value = float(normal_stress_bottomskew)

            diffuse_scale_update = _update_diffuse_scale
            diffuse_scale_update(float(t0))
        else:
            shutoff = _late_flow_shutoff_factor(float(t0))
            diffuse_scale_expr.value = float(target_diffuse_scale) * shutoff
            diffuse_pressure_scale_expr.value = float(target_diffuse_pressure_scale) * shutoff
            diffuse_normal_stress_scale_expr.value = (
                float(target_diffuse_normal_stress_scale)
                * _normal_stress_decay_factor(float(t0))
                * shutoff
            )
            if np.isfinite(normal_stress_frontbias_tail_value):
                diffuse_normal_stress_frontbias_expr.value = _cosine_blend_value(
                    float(t0),
                    start_time=float(normal_stress_decay_start),
                    end_time=float(normal_stress_decay_end),
                    start_value=float(normal_stress_frontbias),
                    end_value=float(normal_stress_frontbias_tail_value),
                )
            else:
                diffuse_normal_stress_frontbias_expr.value = float(normal_stress_frontbias)
            if normal_stress_bottomskew_ramp_with_tail:
                diffuse_normal_stress_bottomskew_expr.value = (
                    float(normal_stress_bottomskew)
                    * _cosine_blend_value(
                        float(t0),
                        start_time=float(normal_stress_decay_start),
                        end_time=float(normal_stress_decay_end),
                        start_value=0.0,
                        end_value=1.0,
                    )
                )
            else:
                diffuse_normal_stress_bottomskew_expr.value = float(normal_stress_bottomskew)

        shear_model = str(getattr(args, "diffuse_shear_model", "lagged_velocity")).strip().lower()
        if shear_model == "poiseuille":
            diffuse_g_t, diffuse_w_t = _poiseuille_diffuse_interface_shear_traction(
                alpha_lag=alpha_n,
                H=float(H),
                u_max=1.5 * float(args.u_avg),
                mu_f=float(args.mu_f),
                scale=diffuse_scale_expr,
                eta_n=float(args.diffuse_shear_eta),
                topweight=diffuse_shear_topweight,
            )
        elif shear_model == "lagged_stress":
            diffuse_g_t, diffuse_w_t = _lagged_diffuse_interface_stress_traction(
                v_lag=v_trac_n,
                p_lag=p_trac_n,
                alpha_lag=alpha_n,
                mu_f=float(args.mu_f),
                scale=diffuse_scale_expr,
                eta_n=float(args.diffuse_shear_eta),
            )
        else:
            diffuse_g_t, diffuse_w_t = _lagged_diffuse_interface_shear_traction(
                v_lag=v_trac_n,
                alpha_lag=alpha_n,
                mu_f=float(args.mu_f),
                scale=diffuse_scale_expr,
                eta_n=float(args.diffuse_shear_eta),
                topweight=diffuse_shear_topweight,
            )
        if abs(float(target_diffuse_pressure_scale)) > 0.0:
            diffuse_g_p, diffuse_w_p = _lagged_diffuse_interface_pressure_traction(
                p_lag=p_trac_n,
                alpha_lag=alpha_n,
                scale=diffuse_pressure_scale_expr,
                eta_n=float(args.diffuse_shear_eta),
                xweight=diffuse_pressure_xweight,
                upstream_only=diffuse_pressure_upstream_only,
            )
            if diffuse_g_t is None:
                diffuse_g_t = diffuse_g_p
                diffuse_w_t = diffuse_w_p
            else:
                diffuse_g_t = (diffuse_g_t[0] + diffuse_g_p[0], diffuse_g_t[1] + diffuse_g_p[1])
        if abs(float(target_diffuse_normal_stress_scale)) > 0.0:
            diffuse_g_n, diffuse_w_n = _lagged_diffuse_interface_normal_stress_traction(
                v_lag=v_trac_n,
                p_lag=p_trac_n,
                alpha_lag=alpha_n,
                mu_f=float(args.mu_f),
                scale=diffuse_normal_stress_scale_expr,
                eta_n=float(args.diffuse_shear_eta),
                xweight=bool(getattr(args, "diffuse_normal_stress_xweight", False)),
                topweight=bool(getattr(args, "diffuse_normal_stress_topweight", False)),
                topbias=normal_stress_topbias,
                frontbias=diffuse_normal_stress_frontbias_expr,
                bottomskew=diffuse_normal_stress_bottomskew_expr,
                channel_height=float(H),
                upstream_only=bool(getattr(args, "diffuse_normal_stress_upstream_only", False)),
            )
            if diffuse_g_t is None:
                diffuse_g_t = diffuse_g_n
                diffuse_w_t = diffuse_w_n
            else:
                diffuse_g_t = (diffuse_g_t[0] + diffuse_g_n[0], diffuse_g_t[1] + diffuse_g_n[1])
        logging.info(
            "[setup] diffuse interface traction enabled: model=%s, scale=%.3e, shear_topweight=%s, normal_pressure_scale=%.3e, normal_pressure_xweight=%s, normal_pressure_upstream_only=%s, normal_stress_scale=%.3e, normal_stress_xweight=%s, normal_stress_topweight=%s, normal_stress_topbias=%.3e, normal_stress_frontbias=%.3e, normal_stress_frontbias_tail_value=%.3e, normal_stress_bottomskew=%.3e, normal_stress_bottomskew_ramp_with_tail=%s, normal_stress_upstream_only=%s, normal_stress_ramp=%.3e, normal_stress_decay_start=%.3e, normal_stress_decay_end=%.3e, normal_stress_tail_factor=%.3e, scheme=%s, ramp=%.3e, eta=%.3e",
            shear_model,
            float(args.diffuse_shear_scale),
            str(diffuse_shear_topweight),
            float(target_diffuse_pressure_scale),
            str(diffuse_pressure_xweight),
            str(diffuse_pressure_upstream_only),
            float(target_diffuse_normal_stress_scale),
            str(bool(getattr(args, "diffuse_normal_stress_xweight", False))),
            str(bool(getattr(args, "diffuse_normal_stress_topweight", False))),
            float(normal_stress_topbias),
            float(normal_stress_frontbias),
            float(normal_stress_frontbias_tail_value),
            float(normal_stress_bottomskew),
            str(normal_stress_bottomskew_ramp_with_tail),
            str(bool(getattr(args, "diffuse_normal_stress_upstream_only", False))),
            float(normal_stress_ramp_time),
            float(normal_stress_decay_start),
            float(normal_stress_decay_end),
            float(normal_stress_tail_factor),
            diffuse_time_scheme,
            float(diffuse_ramp_time),
            float(args.diffuse_shear_eta),
        )
        if flow_shutoff_start_finite:
            logging.info(
                "[setup] late flow shutoff enabled: start=%.3e s end=%.3e s end_factor=%.3e",
                float(flow_shutoff_start),
                float(flow_shutoff_end),
                float(flow_shutoff_end_factor),
            )
    if ch_enabled:
        zeta_ratio = float("nan")
        if abs(float(alpha_ch_gamma)) > 0.0 and abs(float(alpha_ch_eps)) > 0.0:
            zeta_ratio = (float(getattr(args, "diffuse_shear_scale", 0.0)) ** 2) / (float(alpha_ch_gamma) * float(alpha_ch_eps))
        logging.info(
            "[setup] alpha interface thickness: eps0=%.3e alpha_ch_eps_base=%.3e alpha_ch_eps_eff=%.3e scale_with_zeta=%s scale_factor=%.3e zeta_ref=%.3e zeta^2/(gamma_alpha eps_alpha)=%.3e",
            float(eps0),
            float(alpha_ch_eps_base),
            float(alpha_ch_eps),
            str(bool(scale_alpha_ch_eps_with_zeta)).lower(),
            float(alpha_ch_eps_scale_factor),
            float(diffuse_shear_scale_ref),
            float(zeta_ratio),
        )

    if paper1_reduced:
        forms = build_deformation_only_forms(
            v_k=v_k,
            p_k=p_k,
            vS_k=vS_k,
            u_k=u_k,
            alpha_k=alpha_k,
            mu_alpha_k=mu_alpha_k,
            v_n=v_n,
            p_n=p_n,
            vS_n=vS_n,
            u_n=u_n,
            alpha_n=alpha_n,
            mu_alpha_n=mu_alpha_n,
            dv=dv,
            dp=dp,
            dvS=dvS,
            du=du,
            dalpha=dalpha,
            dmu_alpha=dmu_alpha,
            v_test=v_test,
            q_test=q_test,
            vS_test=vS_test,
            u_test=u_test,
            alpha_test=alpha_test,
            mu_alpha_test=mu_alpha_test,
            dx=dx(metadata={"q": int(args.q)}),
            dt=dt_c,
            theta=theta,
            rho_f=rho_f_c,
            mu_f=mu_f_c,
            mu_b=mu_b_c,
            kappa_inv=kappa_inv_c,
            mu_s=mu_s_c,
            lambda_s=lambda_s_c,
            solid_visco_eta=solid_visco_eta_c,
            rho_s0_tilde=rho_s0_tilde_c,
            include_skeleton_acceleration=bool(args.include_skeleton_acceleration),
            skeleton_inertia_convection=str(args.skeleton_inertia_convection),
            gamma_div=gamma_div_expr,
            phi_b=float(phi_b),
            M_alpha=alpha_ch_M_c,
            gamma_alpha=alpha_ch_gamma_c,
            eps_alpha=alpha_ch_eps_c,
            alpha_advect_with=str(args.alpha_advect_with),
            alpha_advection_form=str(args.alpha_advection_form),
            g_t_k=diffuse_g_t,
            g_t_n=diffuse_g_t,
            traction_weight_k=diffuse_w_t,
            traction_weight_n=diffuse_w_t,
            alpha_biot=float(args.alpha_biot) if np.isfinite(float(args.alpha_biot)) else None,
        )
    else:
        forms = build_biofilm_one_domain_forms(
            v_k=v_k,
            p_k=p_k,
            vS_k=vS_k,
            u_k=u_k,
            phi_k=phi_k,
            alpha_k=alpha_k,
            mu_alpha_k=mu_alpha_k,
            S_k=S_k,
            v_n=v_n,
            p_n=p_n,
            vS_n=vS_n,
            u_n=u_n,
            phi_n=phi_n,
            alpha_n=alpha_n,
            mu_alpha_n=mu_alpha_n,
            S_n=S_n,
            dv=dv,
            dp=dp,
            dvS=dvS,
            du=du,
            dphi=dphi,
            dalpha=dalpha,
            dmu_alpha=dmu_alpha,
            dS=dS,
            v_test=v_test,
            q_test=q_test,
            vS_test=vS_test,
            u_test=u_test,
            phi_test=phi_test,
            alpha_test=alpha_test,
            mu_alpha_test=mu_alpha_test,
            S_test=S_test,
            dx=dx(metadata={"q": int(args.q)}),
            ds_cip=ds(metadata={"q": int(args.q)}),
            dt=dt_c,
            theta=theta,
            rho_f=rho_f_c,
            mu_f=mu_f_c,
            mu_b=mu_b_c,
            mu_b_model=str(getattr(args, "mu_b_model", "phi_mu")),
            kappa_inv=kappa_inv_c,
            kappa_inv_model=kappa_inv_model,
            kappa_inv_phi_ref=kappa_phi_ref,
            kappa_inv_kc_eps=float(getattr(args, "kappa_inv_kc_eps", 1.0e-12)),
            mu_s=mu_s_c,
            lambda_s=lambda_s_c,
            solid_model=str(getattr(args, "solid_model", "linear")),
            solid_visco_eta=solid_visco_eta_c,
            rho_s0_tilde=rho_s0_tilde_c,
            include_skeleton_acceleration=bool(args.include_skeleton_acceleration),
            skeleton_inertia_convection=str(args.skeleton_inertia_convection),
            gamma_u=gamma_u_c,
            u_extension_mode=str(args.u_extension),
            gamma_u_pin=gamma_u_pin_c,
            regularization_weight=regularization_weight,
            kinematics_scale=float(args.kinematics_scale) if np.isfinite(float(args.kinematics_scale)) else None,
            v_supg=v_supg_c,
            v_supg_mode=str(getattr(args, "v_supg_mode", "streamline")),
            v_supg_c_nu=float(getattr(args, "v_supg_c_nu", 4.0)),
            fluid_hdiv_order=int(fluid_hdiv_order),
            u_supg=u_supg_c,
            u_cip=u_cip_c,
            u_cip_weight=str(args.u_cip_weight),
            v_cip=v_cip_c,
            vS_cip=vS_cip_c,
            gamma_div=gamma_div_expr,
            fluid_convection=str(getattr(args, "fluid_convection", "full")),
            # Transport/kinetics controls (FSI-only: disable growth/detachment/damage, but may solve alpha/phi in PDE mode).
            D_phi=D_phi_c if transport_mode == "pde" else 0.0,
            gamma_phi=gamma_phi_c if transport_mode == "pde" else 0.0,
            phi_supg=float(args.phi_supg) if transport_mode == "pde" else 0.0,
            phi_cip=float(args.phi_cip) if transport_mode == "pde" else 0.0,
            D_alpha=D_alpha_c if transport_mode == "pde" else 0.0,
            alpha_advect_with=str(args.alpha_advect_with),
            alpha_mix_gate_alpha0=float(getattr(args, "alpha_mix_gate_alpha0", 0.1)),
            alpha_mix_gate_power=int(getattr(args, "alpha_mix_gate_power", 4)),
            alpha_advection_form=str(args.alpha_advection_form) if transport_mode == "pde" else "advective",
            support_physics=str(args.support_physics),
            alpha_ch_M=alpha_ch_M_c if transport_mode == "pde" else 0.0,
            alpha_ch_gamma=alpha_ch_gamma_c if transport_mode == "pde" else 0.0,
            alpha_ch_eps=alpha_ch_eps_c if transport_mode == "pde" else 1.0,
            alpha_ch_mobility=str(args.alpha_ch_mobility),
            alpha_supg=float(args.alpha_supg) if transport_mode == "pde" else 0.0,
            alpha_cip=float(args.alpha_cip) if transport_mode == "pde" else 0.0,
            g_t_k=diffuse_g_t,
            g_t_n=diffuse_g_t,
            traction_weight_k=diffuse_w_t,
            traction_weight_n=diffuse_w_t,
            alpha_biot=float(args.alpha_biot) if np.isfinite(float(args.alpha_biot)) else None,
            ds_hdiv_tangential=hdiv_tangential_bc_measure,
            hdiv_tangential_gamma=float(getattr(args, "hdiv_tangential_gamma", 20.0)),
            hdiv_tangential_method=str(getattr(args, "hdiv_tangential_method", "penalty")),
            ds_adh=dS_measure(defined_on=mesh.edge_bitset("bottom"), metadata={"q": int(args.q)})
            if use_bottom_adhesion
            else None,
            adhesion_k_n=adhesion_k_n if use_bottom_adhesion else 0.0,
            adhesion_k_t=adhesion_k_t if use_bottom_adhesion else 0.0,
            adhesion_gamma_n=adhesion_gamma_n if use_bottom_adhesion else 0.0,
            adhesion_gamma_t=adhesion_gamma_t if use_bottom_adhesion else 0.0,
            # (keep Allen–Cahn disabled in this benchmark)
            alpha_cahn_M=0.0,
            alpha_cahn_gamma=0.0,
            D_S=0.0,
            mu_max=0.0,
            k_g=0.0,
            k_d=0.0,
            k_det=0.0,
            dim=2,
        )
    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------
    u_avg = float(args.u_avg)
    u_max = 1.5 * u_avg
    inflow_profile = str(getattr(args, "inflow_profile", "poiseuille")).strip().lower()
    if inflow_profile not in {"poiseuille", "plug"}:
        raise ValueError(f"Unknown --inflow-profile {inflow_profile!r}.")
    inflow_plug_speed = float(getattr(args, "inflow_plug_speed", float("nan")))
    if not np.isfinite(inflow_plug_speed):
        inflow_plug_speed = u_max
    t_ramp = float(args.t_ramp)
    try:
        mu_f0 = float(args.mu_f)
        rho_f0 = float(args.rho_f)
        tau_w = 6.0 * mu_f0 * u_avg / float(H)  # plane Poiseuille between plates
        inflow_speed_ref = u_avg if inflow_profile == "poiseuille" else inflow_plug_speed
        re_char_length = float(args.re_char_length) if np.isfinite(float(args.re_char_length)) else float(H)
        Re = rho_f0 * inflow_speed_ref * float(re_char_length) / max(mu_f0, 1.0e-30)
        if inflow_profile == "poiseuille":
            logging.info(
                "[setup] inflow: profile=%s u_avg=%.3e m/s (u_max=%.3e), tau_w≈%.3e Pa, Re≈%.1f (L_char=%.3e m)",
                inflow_profile,
                u_avg,
                u_max,
                tau_w,
                Re,
                re_char_length,
            )
        else:
            logging.info(
                "[setup] inflow: profile=%s u_plug=%.3e m/s (u_avg_ref=%.3e, u_max_ref=%.3e), Re≈%.1f (L_char=%.3e m)",
                inflow_profile,
                inflow_plug_speed,
                u_avg,
                u_max,
                Re,
                re_char_length,
            )
        logging.info(
            "[setup] constitutive choices: solid_model=%s mu_b_model=%s alpha_biot=%s",
            str(getattr(args, "solid_model", "linear")),
            str(getattr(args, "mu_b_model", "phi_mu")),
            (
                f"{float(args.alpha_biot):.3e}"
                if np.isfinite(float(getattr(args, "alpha_biot", float('nan'))))
                else "constraint_only"
            ),
        )
        if flow_shutoff_start_finite:
            logging.info(
                "[setup] inflow shutoff schedule: start=%.3e s end=%.3e s end_factor=%.3e",
                float(flow_shutoff_start),
                float(flow_shutoff_end),
                float(flow_shutoff_end_factor),
            )
        logging.info(
            "[setup] permeability: model=%s kappa_inv=%.3e kappa_phi_ref=%.3e kappa_inv_kc_eps=%.3e",
            kappa_inv_model,
            float(args.kappa_inv),
            float(kappa_phi_ref),
            float(getattr(args, "kappa_inv_kc_eps", 1.0e-12)),
        )
        logging.info(
            "[setup] attachment: mode=%s adhesion_k_n=%.3e adhesion_k_t=%.3e adhesion_gamma_n=%.3e adhesion_gamma_t=%.3e",
            attachment_mode,
            float(adhesion_k_n),
            float(adhesion_k_t),
            float(adhesion_gamma_n),
            float(adhesion_gamma_t),
        )
        if float(gamma_div_expr) != 0.0:
            logging.info(f"[setup] mixture grad-div stabilization enabled: gamma_div={float(gamma_div_expr):.3e}")
            if adaptive_gamma_div:
                logging.info(
                    "[setup] adaptive gamma_div enabled: base=%.3e current=%.3e max=%.3e growth=%.2f relax_after=%d relax_factor=%.2f easy_nNewton<=%d",
                    float(gamma_div_base),
                    float(gamma_div_expr),
                    float(gamma_div_max),
                    float(gamma_div_growth),
                    int(gamma_div_relax_after),
                    float(gamma_div_relax_factor),
                    int(gamma_div_relax_newton_max),
                )
    except Exception:
        pass

    def inflow_vx(_x, y, t):
        if inflow_profile == "plug":
            base = float(inflow_plug_speed)
        else:
            yy = float(y) / float(H)
            base = float(u_max * 4.0 * yy * (1.0 - yy))
        if t is None:
            return base
        tt = float(t)
        return base * _cosine_ramp_value(tt, t_ramp) * _late_flow_shutoff_factor(tt)

    if fluid_space_key == "cg":
        bcs = [
            BoundaryCondition("v_x", "dirichlet", "left", inflow_vx),
            BoundaryCondition("v_y", "dirichlet", "left", lambda x, y, t: 0.0),
            BoundaryCondition("v_x", "dirichlet", "bottom", lambda x, y, t: 0.0),
            BoundaryCondition("v_y", "dirichlet", "bottom", lambda x, y, t: 0.0),
            BoundaryCondition("v_x", "dirichlet", "top", lambda x, y, t: 0.0),
            BoundaryCondition("v_y", "dirichlet", "top", lambda x, y, t: 0.0),
            BoundaryCondition("p", "dirichlet", "right", lambda x, y, t: 0.0),
        ]
    else:
        if bool(getattr(args, "hdiv_tangential_dirichlet", True)):
            logging.info(
                "[setup] fluid-space=hdiv uses strong normal-flux Dirichlet data on v and weak tangential Dirichlet data via %s with gamma_t=%.3e.",
                str(getattr(args, "hdiv_tangential_method", "penalty")),
                float(getattr(args, "hdiv_tangential_gamma", 20.0)),
            )
        else:
            logging.info(
                "[setup] fluid-space=hdiv imposes only normal-flux Dirichlet data on v; tangential no-slip remains unconstrained."
            )
        bcs = [
            BoundaryCondition("v", "dirichlet", "left", lambda x, y, t: np.asarray([inflow_vx(x, y, t), 0.0], dtype=float)),
            BoundaryCondition("v", "dirichlet", "bottom", lambda x, y, t: 0.0),
            BoundaryCondition("v", "dirichlet", "top", lambda x, y, t: 0.0),
            BoundaryCondition("p", "dirichlet", "right", lambda x, y, t: 0.0),
        ]
    if not use_bottom_adhesion:
        # Default Benchmark 6 setup: hard clamp on the substratum.
        bcs.extend(
            [
                BoundaryCondition("u_x", "dirichlet", "bottom", lambda x, y, t: 0.0),
                BoundaryCondition("u_y", "dirichlet", "bottom", lambda x, y, t: 0.0),
                BoundaryCondition("vS_x", "dirichlet", "bottom", lambda x, y, t: 0.0),
                BoundaryCondition("vS_y", "dirichlet", "bottom", lambda x, y, t: 0.0),
            ]
        )
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0)) for b in bcs]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    launch_env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("PYCUTFEM_")
        or key in {
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "CONDA_DEFAULT_ENV",
        }
    }
    launch_meta = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "cwd": os.getcwd(),
        "argv": list(sys.argv),
        "command": shlex.join(list(sys.argv)),
        "environment": dict(sorted(launch_env.items())),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(launch_meta, indent=2) + "\n", encoding="utf-8")
    (out_dir / "run_command.sh").write_text("#!/usr/bin/env bash\n" + shlex.join(list(sys.argv)) + "\n", encoding="utf-8")
    gamma_div_history_path = out_dir / "gamma_div_history.csv"
    if adaptive_gamma_div:
        if (not gamma_div_history_path.exists()) or gamma_div_history_path.stat().st_size == 0:
            with gamma_div_history_path.open("w", encoding="utf-8") as f:
                f.write("event,step,t_s,dt_s,gamma_div_old,gamma_div_new,reason\n")
                f.write(
                    f"init,{int(step0) if restart_meta is not None else 0},"
                    f"{float(restart_meta.get('t', 0.0)) if restart_meta is not None else 0.0:.12e},"
                    f"{float(restart_meta.get('dt', dt_val)) if restart_meta is not None else float(dt_val):.12e},"
                    f"{float(gamma_div_expr):.12e},{float(gamma_div_expr):.12e},startup\n"
                )
    vtk_every = int(args.vtk_every)
    vtk_dir = out_dir / "vtk"
    if vtk_every > 0:
        vtk_dir.mkdir(parents=True, exist_ok=True)

    restart_write_every = max(0, int(args.restart_write_every))
    restart_dir = Path(str(args.restart_dir)).expanduser() if str(args.restart_dir).strip() else (out_dir / "restart")
    if restart_write_every > 0 or restart_from is not None:
        restart_dir.mkdir(parents=True, exist_ok=True)

    snapshot_targets = sorted(
        float(v) for v in str(getattr(args, "snapshot_times", "")).split(",") if str(v).strip()
    )
    snapshot_dir = out_dir / "snapshots"
    if snapshot_targets:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    gamma_div_state = {
        "base": float(gamma_div_base),
        "current": float(gamma_div_expr),
        "max": float(gamma_div_max),
        "easy_steps": 0,
    }

    def _record_gamma_div_event(event: str, *, step_no: int, t_s: float, dt_s: float, old: float, new: float, reason: str) -> None:
        if not adaptive_gamma_div:
            return
        with gamma_div_history_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{str(event)},{int(step_no)},{float(t_s):.12e},{float(dt_s):.12e},"
                f"{float(old):.12e},{float(new):.12e},{str(reason)}\n"
            )

    def _set_gamma_div(new_value: float, *, reason: str, step_no: int, t_s: float, dt_s: float) -> None:
        old_value = float(gamma_div_expr)
        new_value = float(new_value)
        if not np.isfinite(new_value):
            return
        if abs(new_value - old_value) <= 1.0e-15:
            return
        gamma_div_expr.value = float(new_value)
        gamma_div_state["current"] = float(new_value)
        gamma_div_state["easy_steps"] = 0
        print(
            f"    [adapt_gamma_div] {str(reason)}; gamma_div {float(old_value):.3e} -> {float(new_value):.3e} "
            f"(step={int(step_no)}, t={float(t_s):.6e}, dt={float(dt_s):.3e})."
        )
        _record_gamma_div_event(
            "update",
            step_no=int(step_no),
            t_s=float(t_s),
            dt_s=float(dt_s),
            old=float(old_value),
            new=float(new_value),
            reason=str(reason),
        )

    restart_state: dict[str, object] | None = None
    if restart_from is not None:
        restart_state = _load_restart_checkpoint(
            restart_from,
            v=v_n,
            p=p_n,
            vS=vS_n,
            u=u_n,
            alpha=alpha_n,
            phi=phi_n,
            S=S_n,
            mu_alpha=mu_alpha_n,
        )
        # Start from the checkpoint state (copy n -> k).
        v_k.nodal_values[:] = v_n.nodal_values
        p_k.nodal_values[:] = p_n.nodal_values
        vS_k.nodal_values[:] = vS_n.nodal_values
        u_k.nodal_values[:] = u_n.nodal_values
        alpha_k.nodal_values[:] = alpha_n.nodal_values
        if phi_k is not None and phi_n is not None:
            phi_k.nodal_values[:] = phi_n.nodal_values
        if S_k is not None and S_n is not None:
            S_k.nodal_values[:] = S_n.nodal_values
        mu_alpha_k.nodal_values[:] = mu_alpha_n.nodal_values
        _update_regularization_weight(alpha_n.nodal_values)
        logging.info(f"[restart] loaded {restart_from} (t={float(t0):.6e}s, step={int(step0)}, dt={float(dt_val):.3e})")
        if adaptive_gamma_div and np.isfinite(float(restart_state.get("gamma_div", float("nan")))):
            logging.info(f"[restart] continuing adaptive gamma_div from checkpoint: {float(gamma_div_expr):.3e}")

    track_y_um = [float(v) for v in str(args.track_y_um).split(",") if str(v).strip()]
    y_targets = [1.0e-6 * float(v) for v in track_y_um]
    alpha_y = np.asarray(alpha_xy[:, 1], dtype=float)
    if restart_state is not None and np.asarray(restart_state.get("y_lines", []), dtype=float).size:
        y_lines = np.asarray(restart_state.get("y_lines", []), dtype=float).ravel()
    else:
        y_lines = np.asarray(_nearest_y_levels(alpha_y, y_targets), dtype=float).ravel()

    x_ref_arr = np.asarray(restart_state.get("x_ref", []), dtype=float).ravel() if restart_state is not None else np.empty((0,), dtype=float)
    if x_ref_arr.size == y_lines.size:
        x_ref = [float(v) for v in x_ref_arr]
    else:
        # Reference x positions at t=0 (use initial alpha_n).
        x_ref = [
            _x_alpha_half_track_on_y_line(
                alpha_xy,
                alpha_n.nodal_values,
                y_line=float(y0),
                mode=str(args.dx_intersection),
                q=float(args.dx_quantile),
                prev_x=None,
                alpha_half=0.5,
            )
            for y0 in y_lines
        ]
    x_ref_global_val = float(restart_state.get("x_ref_global", float("nan"))) if restart_state is not None else float("nan")
    if math.isfinite(x_ref_global_val):
        x_ref_global = float(x_ref_global_val)
    else:
        x_ref_global = _x_front_global_quantile(alpha_xy, alpha_n.nodal_values, q=float(args.global_front_quantile), alpha_half=0.5)

    alpha_vals_ref = np.asarray(alpha_n.nodal_values, dtype=float).ravel()
    alpha_weights_ref = np.clip(alpha_vals_ref, 0.0, 1.0)
    alpha_mask_ref = alpha_vals_ref >= 0.5
    dx_q = dx(metadata={"q": int(args.q)})
    alpha_area_ref = float(assemble_scalar(dh, alpha_n * dx_q, backend=str(args.backend), quad_order=int(args.q)))
    if phi_n is not None:
        phi_vals_ref = np.asarray(phi_n.nodal_values, dtype=float).ravel()
        if np.any(alpha_mask_ref):
            phi_ref_alpha05 = float(np.mean(phi_vals_ref[alpha_mask_ref]))
        else:
            phi_ref_alpha05 = float("nan")
        denom_ref = float(np.sum(alpha_weights_ref))
        if denom_ref > 1.0e-14:
            phi_ref_alpha_weighted = float(np.sum(alpha_weights_ref * phi_vals_ref) / denom_ref)
        else:
            phi_ref_alpha_weighted = float("nan")
        phi_ref_min = float(np.min(phi_vals_ref)) if phi_vals_ref.size else float("nan")
        phi_ref_max = float(np.max(phi_vals_ref)) if phi_vals_ref.size else float("nan")
    else:
        phi_ref_alpha05 = float("nan")
        phi_ref_alpha_weighted = float("nan")
        phi_ref_min = float("nan")
        phi_ref_max = float("nan")

    ts_path = out_dir / "timeseries.csv"
    header = (
        [
            "t_s",
            "alpha_area",
            "alpha_area_rel_drift",
            "x_front_global",
            "dx_front_global",
            "phi_mean_alpha05",
            "phi_drop_alpha05_pp",
            "phi_mean_alpha_weighted",
            "phi_drop_alpha_weighted_pp",
            "phi_min",
            "phi_max",
            "phi_min_delta",
            "phi_max_delta",
        ]
        + [f"x_front_y{int(round(1.0e6 * y))}um" for y in y_lines]
        + [f"dx_front_y{int(round(1.0e6 * y))}um" for y in y_lines]
    )
    header_line = ",".join(header)
    if_diag_enabled = bool(getattr(args, "interface_diagnostics", False))
    if_diag_path = out_dir / "interface_diagnostics.csv"
    if_diag_header = [
        "t_s",
        "n_samples",
        "p_mean_pa",
        "p_max_pa",
        "sigma_n_mean_pa",
        "sigma_n_min_pa",
        "sigma_comp_mean_pa",
        "tau_t_abs_mean_pa",
        "tau_t_abs_max_pa",
        "front_samples",
        "front_p_mean_pa",
        "front_sigma_n_mean_pa",
        "front_sigma_comp_mean_pa",
        "front_tau_t_abs_mean_pa",
        "top_samples",
        "top_p_mean_pa",
        "top_sigma_n_mean_pa",
        "top_tau_t_mean_pa",
        "top_tau_t_abs_mean_pa",
        "top_mu_du_dh_mean_pa",
        "top_tau_over_mu_du_dh_mean",
    ]
    if_diag_header_line = ",".join(if_diag_header)
    append_existing_ts = bool(restart_from is not None and ts_path.exists() and ts_path.stat().st_size > 0)
    if append_existing_ts:
        try:
            with ts_path.open("r", encoding="utf-8") as f:
                first = f.readline().strip()
            if first != header_line:
                raise RuntimeError(
                    f"timeseries.csv header mismatch for restart.\n"
                    f"  file: {first}\n"
                    f"  expected: {header_line}\n"
                    f"Use a fresh --out-dir or match --track-y-um and mesh settings."
                )
        except Exception as exc:
            raise RuntimeError(f"Failed validating existing timeseries.csv for restart: {exc}") from exc
    else:
        with ts_path.open("w", encoding="utf-8") as f:
            f.write(header_line + "\n")
    append_existing_ifdiag = bool(if_diag_enabled and restart_from is not None and if_diag_path.exists() and if_diag_path.stat().st_size > 0)
    if append_existing_ifdiag:
        try:
            with if_diag_path.open("r", encoding="utf-8") as f:
                first = f.readline().strip()
            if first != if_diag_header_line:
                raise RuntimeError(
                    f"interface_diagnostics.csv header mismatch for restart.\n"
                    f"  file: {first}\n"
                    f"  expected: {if_diag_header_line}\n"
                    f"Use a fresh --out-dir."
                )
        except Exception as exc:
            raise RuntimeError(f"Failed validating existing interface_diagnostics.csv for restart: {exc}") from exc
    elif if_diag_enabled:
        with if_diag_path.open("w", encoding="utf-8") as f:
            f.write(if_diag_header_line + "\n")

    x_prev_arr = np.asarray(restart_state.get("x_prev", []), dtype=float).ravel() if restart_state is not None else np.empty((0,), dtype=float)
    if x_prev_arr.size == y_lines.size:
        x_prev = [float(v) for v in x_prev_arr]
    else:
        x_prev = [float(v) for v in x_ref]

    pending_snapshots = list(snapshot_targets)

    def _write_snapshot_contours(t_s: float, step_no: int) -> None:
        if not snapshot_targets:
            return
        contours = _alpha_half_contours(alpha_xy, alpha_k.nodal_values, alpha_half=0.5)
        stem = f"snapshot_step{int(step_no):04d}_t{float(t_s):06.3f}"
        _write_contours_csv(snapshot_dir / f"{stem}_alpha05.csv", contours)

    def _append_timeseries(t_s: float) -> None:
        alpha_area = float(assemble_scalar(dh, alpha_k * dx_q, backend=str(args.backend), quad_order=int(args.q)))
        alpha_area_rel_drift = (
            float((alpha_area - alpha_area_ref) / alpha_area_ref) if abs(alpha_area_ref) > 1.0e-14 else float("nan")
        )
        xg = _x_front_global_quantile(alpha_xy, alpha_k.nodal_values, q=float(args.global_front_quantile), alpha_half=0.5)
        dxg = float(xg - x_ref_global) if (math.isfinite(xg) and math.isfinite(x_ref_global)) else float("nan")
        alpha_vals = np.asarray(alpha_k.nodal_values, dtype=float).ravel()
        alpha_weights = np.clip(alpha_vals, 0.0, 1.0)
        alpha_mask = alpha_vals >= 0.5
        if phi_k is not None:
            phi_vals = np.asarray(phi_k.nodal_values, dtype=float).ravel()
            if np.any(alpha_mask):
                phi_mean_alpha05 = float(np.mean(phi_vals[alpha_mask]))
            else:
                phi_mean_alpha05 = float("nan")
            denom = float(np.sum(alpha_weights))
            if denom > 1.0e-14:
                phi_mean_alpha_weighted = float(np.sum(alpha_weights * phi_vals) / denom)
            else:
                phi_mean_alpha_weighted = float("nan")
            phi_min = float(np.min(phi_vals)) if phi_vals.size else float("nan")
            phi_max = float(np.max(phi_vals)) if phi_vals.size else float("nan")
        else:
            phi_mean_alpha05 = float("nan")
            phi_mean_alpha_weighted = float("nan")
            phi_min = float("nan")
            phi_max = float("nan")
        phi_drop_alpha05_pp = (
            100.0 * float(phi_mean_alpha05 - phi_ref_alpha05)
            if math.isfinite(phi_mean_alpha05) and math.isfinite(phi_ref_alpha05)
            else float("nan")
        )
        phi_drop_alpha_weighted_pp = (
            100.0 * float(phi_mean_alpha_weighted - phi_ref_alpha_weighted)
            if math.isfinite(phi_mean_alpha_weighted) and math.isfinite(phi_ref_alpha_weighted)
            else float("nan")
        )
        phi_min_delta = float(phi_min - phi_ref_min) if math.isfinite(phi_min) and math.isfinite(phi_ref_min) else float("nan")
        phi_max_delta = float(phi_max - phi_ref_max) if math.isfinite(phi_max) and math.isfinite(phi_ref_max) else float("nan")
        nonlocal x_prev
        xs = []
        x_new_prev: list[float] = []
        for i, y0 in enumerate(y_lines):
            xi = _x_alpha_half_track_on_y_line(
                alpha_xy,
                alpha_k.nodal_values,
                y_line=float(y0),
                mode=str(args.dx_intersection),
                q=float(args.dx_quantile),
                prev_x=float(x_prev[i]) if i < len(x_prev) else None,
                alpha_half=0.5,
            )
            xs.append(xi)
            if math.isfinite(xi):
                x_new_prev.append(float(xi))
            else:
                x_new_prev.append(float(x_prev[i]))
        x_prev = x_new_prev
        dxs = [float(xs[i] - x_ref[i]) if (math.isfinite(xs[i]) and math.isfinite(x_ref[i])) else float("nan") for i in range(len(xs))]
        with ts_path.open("a", encoding="utf-8") as f:
            f.write(
                ",".join(
                    [
                        f"{float(t_s):.12e}",
                        f"{alpha_area:.12e}",
                        f"{alpha_area_rel_drift:.12e}",
                        f"{xg:.12e}",
                        f"{dxg:.12e}",
                        f"{phi_mean_alpha05:.12e}",
                        f"{phi_drop_alpha05_pp:.12e}",
                        f"{phi_mean_alpha_weighted:.12e}",
                        f"{phi_drop_alpha_weighted_pp:.12e}",
                        f"{phi_min:.12e}",
                        f"{phi_max:.12e}",
                        f"{phi_min_delta:.12e}",
                        f"{phi_max_delta:.12e}",
                    ]
                    + [f"{z:.12e}" for z in xs]
                    + [f"{z:.12e}" for z in dxs]
                )
                + "\n"
            )

    def _append_interface_diagnostics(t_s: float) -> None:
        if not if_diag_enabled:
            return
        diag = _interface_stress_diagnostics(
            dh=dh,
            mesh=mesh,
            alpha_xy=alpha_xy,
            alpha_fn=alpha_k,
            v_fn=v_k,
            p_fn=p_k,
            mu_f=float(args.mu_f),
            alpha_half=0.5,
            max_points=max(16, int(getattr(args, "interface_diagnostics_max_points", 300))),
        )
        row = {"t_s": float(t_s)}
        row.update(diag)
        with if_diag_path.open("a", encoding="utf-8") as f:
            f.write(",".join(f"{float(row[name]):.12e}" for name in if_diag_header) + "\n")

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    newton_params = NewtonParameters(
        newton_tol=float(args.newton_tol),
        newton_rtol=float(args.newton_rtol),
        max_newton_iter=int(args.max_it),
        accept_nonconverged_atol_factor=float(args.accept_nonconverged_atol_factor),
        line_search=True,
        ls_mode=str(args.ls_mode),
    )
    solver_kind = str(getattr(args, "nonlinear_solver", "pdas")).strip().lower()
    newton_solver_key = str(getattr(args, "newton_solver", "pdas")).strip().lower()
    if newton_solver_key in {"snes", "newton"}:
        solver_kind = newton_solver_key

    common_solver_kwargs = dict(
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=newton_params,
        quad_order=int(args.q),
        backend=str(args.backend),
    )
    if solver_kind == "snes":
        petsc_opts = {
            # For box-constrained runs (alpha/phi bounds), use a VI-capable SNES.
            "snes_type": "vinewtonrsls" if use_alpha_phi_vi_bounds else "newtonls",
            # Backtracking line search; we also apply an infinity-norm based pre-check
            # damping in the solver to reduce DIVERGED_LINE_SEARCH events near tight tol.
            "snes_linesearch_type": "bt",
            "snes_atol": float(args.newton_tol),
            "snes_rtol": float(args.newton_rtol),
            "snes_max_it": int(args.max_it),
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": str(getattr(args, "lu_solver", "mumps")),
        }
        if str(getattr(args, "lu_solver", "mumps")).strip().lower() == "mumps":
            # Improve robustness/accuracy on the non-symmetric transient momentum block.
            # MUMPS ICNTL(10): max iterative refinement steps (0 = default).
            petsc_opts.setdefault("mat_mumps_icntl_10", 5)
        if os.getenv("PYCUTFEM_PETSC_MONITOR", "").strip().lower() in {"1", "true", "yes"}:
            petsc_opts.setdefault("snes_monitor", None)
            petsc_opts.setdefault("snes_converged_reason", None)
            petsc_opts.setdefault("ksp_converged_reason", None)
            petsc_opts.setdefault("ksp_monitor_short", None)
        logging.info("[setup] nonlinear solver: snes")
        solver = PetscSnesNewtonSolver(
            forms.residual_form,
            forms.jacobian_form,
            petsc_options=petsc_opts,
            **common_solver_kwargs,
        )
    elif solver_kind == "newton":
        logging.info("[setup] nonlinear solver: internal-newton")
        solver = NewtonSolver(
            forms.residual_form,
            forms.jacobian_form,
            lin_params=LinearSolverParameters(backend=str(args.linear_backend)),
            **common_solver_kwargs,
        )
    else:
        logging.info("[setup] nonlinear solver: pdas")
        solver = PdasNewtonSolver(
            forms.residual_form,
            forms.jacobian_form,
            vi_params=VIParameters(
                c=float(args.vi_c),
                enter_tol=float(args.vi_enter_tol),
                leave_tol=float(args.vi_leave_tol),
                active_set_persistence=int(args.vi_persistence),
                project_initial_guess=True,
                project_each_iteration=True,
                inactive_reg_lambda0=float(args.vi_lambda0),
                inactive_reg_lambda_max=float(args.vi_lambda_max),
                inactive_reg_growth=float(args.vi_lambda_growth),
                inactive_reg_decay=float(args.vi_lambda_decay),
                active_step_delta_active_trigger=int(args.vi_active_soft_threshold),
                active_step_soft_alpha=float(args.vi_active_soft_alpha),
                active_step_strong_factor=float(args.vi_active_strong_factor),
                filter_max_delta_active=int(args.vi_filter_max_delta_active),
                # Bound-aware globalization is essential once alpha/phi box constraints
                # are active: the first loaded Christan steps can drive many transport
                # DOFs outside [0,1], and projecting a full unconstrained step after the
                # fact leads to pathological active-set jumps.
                bound_step_limit=bool(use_alpha_phi_vi_bounds),
                bound_blocking_activate=bool(use_alpha_phi_vi_bounds),
                unconstrained_lm=bool(args.vi_unconstrained_lm),
                unconstrained_lm_lambda0=float(args.vi_lm_lambda0),
                unconstrained_lm_lambda_max=float(args.vi_lm_lambda_max),
                unconstrained_lm_growth=float(args.vi_lm_growth),
                unconstrained_lm_decay=float(args.vi_lm_decay),
                unconstrained_lm_accept_ratio=float(args.vi_lm_accept_ratio),
                unconstrained_lm_good_ratio=float(args.vi_lm_good_ratio),
                unconstrained_lm_max_tries=int(args.vi_lm_max_tries),
            ),
            lin_params=LinearSolverParameters(backend=str(args.linear_backend)),
            **common_solver_kwargs,
        )
    if bool(getattr(args, "linear_schur", False)) and hasattr(solver, "set_linear_schur_fieldsplit"):
        try:
            solver.set_linear_schur_fieldsplit(
                pressure_field=str(args.linear_schur_pressure_field),
                schur_fact=str(args.linear_schur_fact),
                schur_pre=str(args.linear_schur_pre),
                outer_ksp=(str(args.linear_ksp_type).strip() or None),
                outer_pc="fieldsplit",
                rest_ksp=str(args.linear_schur_rest_ksp),
                rest_pc=str(args.linear_schur_rest_pc),
                rest_factor_solver_type=(str(args.linear_schur_rest_factor_solver_type).strip() or None),
                pressure_ksp=str(args.linear_schur_pressure_ksp),
                pressure_pc=str(args.linear_schur_pressure_pc),
                pressure_factor_solver_type=(str(args.linear_schur_pressure_factor_solver_type).strip() or None),
            )
        except Exception as exc:
            logging.warning(f"[setup] failed to install linear Schur fieldsplit: {exc}")

    bounds_by_field: dict[str, tuple[float | None, float | None]] = {}
    if bool(args.alpha_box_constraints) or use_alpha_phi_vi_bounds:
        bounds_by_field["alpha"] = (0.0, 1.0)
    if (phi_k is not None) and (bool(args.phi_box_constraints) or use_alpha_phi_vi_bounds):
        bounds_by_field["phi"] = (0.0, 1.0)
    if bounds_by_field and hasattr(solver, "set_box_bounds"):
        solver.set_box_bounds(by_field=bounds_by_field)
    vi_linear_equalities = _build_vi_linear_equalities(
        support_physics=str(args.support_physics),
        backend=str(args.backend),
        qdeg=int(args.q),
        dh=dh,
        alpha_test=alpha_test,
        alpha_k=alpha_k,
        alpha_n=alpha_n,
        phi_test=phi_test,
        phi_k=phi_k,
        phi_n=phi_n,
        alpha_box_constraints=bool(args.alpha_box_constraints) or bool(use_alpha_phi_vi_bounds),
        phi_box_constraints=((phi_k is not None) and (bool(args.phi_box_constraints) or bool(use_alpha_phi_vi_bounds))),
    )
    if vi_linear_equalities and hasattr(solver, "set_linear_equalities"):
        solver.set_linear_equalities(vi_linear_equalities)

    startup_predictor_enabled = bool(getattr(args, "startup_staggered_predictor", False))
    startup_predictor_max_time = float(getattr(args, "startup_staggered_max_time", float("nan")))
    startup_base_tags = _deepcopy_dof_tags(dh)
    startup_active_fields_fluid = {"p"}
    if fluid_space_key == "cg":
        startup_active_fields_fluid.update({"v_x", "v_y"})
    else:
        startup_active_fields_fluid.add("v")
    startup_active_fields_solid = {"vS_x", "vS_y", "u_x", "u_y"}

    startup_solver_cache: dict[str, NewtonSolver] = {}
    startup_state = {"last_signature": None, "force_next": True}

    def _make_startup_solver(*, active_fields: set[str], newton_tol: float, max_it: int) -> NewtonSolver:
        _set_active_fields(dh, active_fields=active_fields, base_tags=startup_base_tags)
        try:
            return NewtonSolver(
                forms.residual_form,
                forms.jacobian_form,
                dof_handler=dh,
                mixed_element=me,
                bcs=bcs,
                bcs_homog=bcs_homog,
                newton_params=NewtonParameters(
                    newton_tol=float(newton_tol),
                    newton_rtol=0.0,
                    max_newton_iter=int(max_it),
                    accept_nonconverged_atol_factor=0.0,
                    line_search=True,
                    ls_mode=str(args.ls_mode),
                ),
                lin_params=LinearSolverParameters(backend=str(args.linear_backend)),
                quad_order=int(args.q),
                backend=str(args.backend),
            )
        finally:
            dh.dof_tags = copy.deepcopy(startup_base_tags)

    def _get_startup_solver(name: str) -> NewtonSolver:
        solver_cached = startup_solver_cache.get(str(name))
        if solver_cached is not None:
            return solver_cached
        if str(name) == "fluid":
            solver_cached = _make_startup_solver(
                active_fields=startup_active_fields_fluid,
                newton_tol=float(getattr(args, "startup_fluid_newton_tol", args.newton_tol)),
                max_it=int(getattr(args, "startup_fluid_max_it", args.max_it)),
            )
        elif str(name) == "solid":
            solver_cached = _make_startup_solver(
                active_fields=startup_active_fields_solid,
                newton_tol=float(getattr(args, "startup_solid_newton_tol", args.newton_tol)),
                max_it=int(getattr(args, "startup_solid_max_it", args.max_it)),
            )
        else:
            raise KeyError(f"Unknown startup solver '{name}'.")
        startup_solver_cache[str(name)] = solver_cached
        return solver_cached

    def _snapshot_values(*funcs) -> list[tuple[object, np.ndarray]]:
        out: list[tuple[object, np.ndarray]] = []
        for func in funcs:
            if func is None or getattr(func, "nodal_values", None) is None:
                continue
            out.append((func, np.asarray(func.nodal_values, dtype=float).copy()))
        return out

    def _copy_function_values(src, dst) -> None:
        if src is None or dst is None:
            return
        dst.nodal_values[:] = np.asarray(src.nodal_values, dtype=float)

    def _restore_snapshot(snapshot: list[tuple[object, np.ndarray]]) -> None:
        for func, values in snapshot:
            try:
                func.nodal_values[:] = np.asarray(values, dtype=float)
            except Exception:
                continue

    def _run_restricted_startup_solve(
        *,
        stage_name: str,
        active_fields: set[str],
        funcs,
        prev_funcs,
        aux_funcs,
        bcs_now,
    ) -> tuple[bool, int, float]:
        _set_active_fields(dh, active_fields=active_fields, base_tags=startup_base_tags)
        try:
            startup_solver = _get_startup_solver(stage_name)
            _delta, converged, n_iters = startup_solver._newton_loop(funcs, prev_funcs, aux_funcs, bcs_now)
            norm_last = getattr(startup_solver, "_last_nonlinear_norm", float("nan"))
            return bool(converged), int(n_iters), float(norm_last if norm_last is not None else float("nan"))
        finally:
            dh.dof_tags = copy.deepcopy(startup_base_tags)

    def _startup_interface_summary(label: str) -> None:
        try:
            diag = _interface_stress_diagnostics(
                dh=dh,
                mesh=mesh,
                alpha_xy=alpha_xy,
                alpha_fn=alpha_k,
                v_fn=v_k,
                p_fn=p_k,
                mu_f=float(args.mu_f),
                alpha_half=0.5,
                max_points=120,
            )
        except Exception as exc:
            logging.warning(f"[startup] {label} interface diagnostics failed: {exc}")
            return
        logging.info(
            "[startup] %s interface: p_mean=%.3e Pa p_max=%.3e Pa tau_t_abs_mean=%.3e Pa tau_t_abs_max=%.3e Pa sigma_n_mean=%.3e Pa top_tau_t_abs_mean=%.3e Pa top_mu_du_dh_mean=%.3e Pa top_tau_over_mu_du_dh=%.3e",
            str(label),
            float(diag.get("p_mean_pa", float("nan"))),
            float(diag.get("p_max_pa", float("nan"))),
            float(diag.get("tau_t_abs_mean_pa", float("nan"))),
            float(diag.get("tau_t_abs_max_pa", float("nan"))),
            float(diag.get("sigma_n_mean_pa", float("nan"))),
            float(diag.get("top_tau_t_abs_mean_pa", float("nan"))),
            float(diag.get("top_mu_du_dh_mean_pa", float("nan"))),
            float(diag.get("top_tau_over_mu_du_dh_mean", float("nan"))),
        )

    def _startup_slip_summary(label: str) -> dict[str, float]:
        out = {
            "vx_minus_vSx_inf": float("nan"),
            "vy_minus_vSy_inf": float("nan"),
            "v_minus_vS_inf": float("nan"),
            "v_minus_vS_l2": float("nan"),
        }
        if not hasattr(v_k, "components") or not hasattr(vS_k, "components"):
            logging.info("[startup] %s slip: skipped for non-nodal velocity representation.", str(label))
            return out
        try:
            vx = np.asarray(v_k.components[0].nodal_values, dtype=float).ravel()
            vy = np.asarray(v_k.components[1].nodal_values, dtype=float).ravel()
            vsx = np.asarray(vS_k.components[0].nodal_values, dtype=float).ravel()
            vsy = np.asarray(vS_k.components[1].nodal_values, dtype=float).ravel()
            dxv = vx - vsx
            dyv = vy - vsy
            out = {
                "vx_minus_vSx_inf": float(np.max(np.abs(dxv))),
                "vy_minus_vSy_inf": float(np.max(np.abs(dyv))),
                "v_minus_vS_inf": float(max(np.max(np.abs(dxv)), np.max(np.abs(dyv)))),
                "v_minus_vS_l2": float(np.sqrt(np.dot(dxv, dxv) + np.dot(dyv, dyv))),
            }
            logging.info(
                "[startup] %s slip: |vx-vSx|_inf=%.3e |vy-vSy|_inf=%.3e |v-vS|_inf=%.3e |v-vS|_2=%.3e",
                str(label),
                out["vx_minus_vSx_inf"],
                out["vy_minus_vSy_inf"],
                out["v_minus_vS_inf"],
                out["v_minus_vS_l2"],
            )
        except Exception as exc:
            logging.warning(f"[startup] {label} slip diagnostics failed: {exc}")
        return out

    def _startup_step_initial_guess_callback(**info):  # noqa: ANN001
        if not startup_predictor_enabled:
            return
        t_bc = float(info.get("t_bc", 0.0) or 0.0)
        _copy_function_values(v_n, v_trac_n)
        _copy_function_values(p_n, p_trac_n)
        if np.isfinite(startup_predictor_max_time) and t_bc > float(startup_predictor_max_time) + 1.0e-15:
            return
        step_no = int(info.get("step_no", -1))
        dt_now = float(info.get("dt", 0.0) or 0.0)
        signature = (int(step_no), float(dt_now), float(t_bc))
        if (not bool(startup_state.get("force_next", False))) and startup_state.get("last_signature", None) == signature:
            return

        funcs = list(info.get("functions", []) or [])
        prev_funcs = list(info.get("prev_functions", []) or [])
        aux_funcs = info.get("aux_functions", None)
        bcs_now = info.get("bcs", None)
        if not funcs or not prev_funcs or bcs_now is None:
            return

        startup_state["last_signature"] = signature
        startup_state["force_next"] = False
        startup_sweeps = max(1, int(getattr(args, "startup_staggered_sweeps", 1)))
        startup_slip_tol = float(getattr(args, "startup_staggered_slip_tol", 0.0))

        fluid_backup = _snapshot_values(v_k, p_k)
        solid_backup = _snapshot_values(vS_k, u_k)
        traction_backup = _snapshot_values(v_trac_n, p_trac_n)

        logging.info(
            "[startup] step=%d t_bc=%.6e dt=%.3e: running restricted fluid->solid predictor (%d sweep%s).",
            int(step_no),
            float(t_bc),
            float(dt_now),
            int(startup_sweeps),
            "" if int(startup_sweeps) == 1 else "s",
        )

        for sweep_idx in range(1, startup_sweeps + 1):
            fluid_ok = False
            try:
                fluid_ok, fluid_iters, fluid_norm = _run_restricted_startup_solve(
                    stage_name="fluid",
                    active_fields=startup_active_fields_fluid,
                    funcs=funcs,
                    prev_funcs=prev_funcs,
                    aux_funcs=aux_funcs,
                    bcs_now=bcs_now,
                )
                logging.info(
                    "[startup] fluid predictor sweep=%d/%d: converged=%s iters=%d |F|_inf=%.3e",
                    int(sweep_idx),
                    int(startup_sweeps),
                    str(bool(fluid_ok)).lower(),
                    int(fluid_iters),
                    float(fluid_norm),
                )
                _copy_function_values(v_k, v_trac_n)
                _copy_function_values(p_k, p_trac_n)
                _startup_interface_summary(f"fluid[sweep={int(sweep_idx)}]")
            except Exception as exc:
                _restore_snapshot(fluid_backup)
                _restore_snapshot(solid_backup)
                _restore_snapshot(traction_backup)
                logging.warning(f"[startup] fluid predictor failed on sweep {sweep_idx}: {exc}")
                return

            if not fluid_ok:
                _restore_snapshot(fluid_backup)
                _restore_snapshot(solid_backup)
                _restore_snapshot(traction_backup)
                return

            try:
                solid_ok, solid_iters, solid_norm = _run_restricted_startup_solve(
                    stage_name="solid",
                    active_fields=startup_active_fields_solid,
                    funcs=funcs,
                    prev_funcs=prev_funcs,
                    aux_funcs=aux_funcs,
                    bcs_now=bcs_now,
                )
                logging.info(
                    "[startup] solid predictor sweep=%d/%d: converged=%s iters=%d |F|_inf=%.3e",
                    int(sweep_idx),
                    int(startup_sweeps),
                    str(bool(solid_ok)).lower(),
                    int(solid_iters),
                    float(solid_norm),
                )
                _startup_interface_summary(f"solid[sweep={int(sweep_idx)}]")
                slip_stats = _startup_slip_summary(f"sweep={int(sweep_idx)}")
                if not solid_ok:
                    _restore_snapshot(fluid_backup)
                    _restore_snapshot(solid_backup)
                    _restore_snapshot(traction_backup)
                    return
                if startup_slip_tol > 0.0 and np.isfinite(slip_stats.get("v_minus_vS_inf", float("nan"))):
                    if float(slip_stats["v_minus_vS_inf"]) <= float(startup_slip_tol):
                        logging.info(
                            "[startup] early stop after sweep %d/%d: |v-vS|_inf=%.3e <= %.3e",
                            int(sweep_idx),
                            int(startup_sweeps),
                            float(slip_stats["v_minus_vS_inf"]),
                            float(startup_slip_tol),
                        )
                        break
            except Exception as exc:
                _restore_snapshot(fluid_backup)
                _restore_snapshot(solid_backup)
                _restore_snapshot(traction_backup)
                logging.warning(f"[startup] solid predictor failed on sweep {sweep_idx}: {exc}")
                return

    def _on_dt_change(new_dt: float) -> None:
        dt_c.value = float(new_dt)

    def _masked_residual_inf(vec: np.ndarray, bcs_now) -> float:
        arr = np.asarray(vec, dtype=float).ravel()
        mask = np.ones(arr.size, dtype=bool)
        try:
            dirichlet = dh.get_dirichlet_data(bcs_now or []) or {}
        except Exception:
            dirichlet = {}
        if dirichlet:
            bc_rows = np.fromiter(dirichlet.keys(), dtype=int)
            if bc_rows.size:
                mask[bc_rows] = False
        inactive = set(getattr(dh, "dof_tags", {}).get("inactive", set()))
        if inactive:
            mask[np.fromiter(inactive, dtype=int)] = False
        if not np.any(mask):
            return 0.0
        return float(np.linalg.norm(arr[mask], ord=np.inf))

    def _block_residual_breakdown(bcs_now) -> list[tuple[float, str]]:
        blocks = [
            ("momentum", getattr(forms, "r_momentum", None)),
            ("mass", getattr(forms, "r_mass", None)),
            ("skeleton", getattr(forms, "r_skeleton", None)),
            ("kinematics", getattr(forms, "r_kinematics", None)),
            ("phi", getattr(forms, "r_phi", None)),
            ("alpha", getattr(forms, "r_alpha", None)),
            ("mu_alpha", getattr(forms, "r_mu_alpha", None)),
            ("alpha_lambda", getattr(forms, "r_alpha_lambda", None)),
            ("substrate", getattr(forms, "r_substrate", None)),
            ("damage", getattr(forms, "r_damage", None)),
            ("detached", getattr(forms, "r_detached", None)),
        ]
        out: list[tuple[float, str]] = []
        for name, res_form in blocks:
            if res_form is None:
                continue
            try:
                _, F_blk = assemble_form(
                    Equation(None, res_form),
                    dof_handler=dh,
                    bcs=[],
                    quad_order=int(args.q),
                    backend=str(args.backend),
                )
                out.append((_masked_residual_inf(np.asarray(F_blk, dtype=float), bcs_now), str(name)))
            except Exception as exc:
                out.append((float("inf"), f"{name}(assembly_failed:{exc})"))
        out.sort(reverse=True, key=lambda t: t[0])
        return out

    def _named_residual_breakdown(named_forms, bcs_now) -> list[tuple[float, str]]:
        out: list[tuple[float, str]] = []
        if not isinstance(named_forms, dict):
            return out
        for name, res_form in named_forms.items():
            if res_form is None:
                continue
            try:
                _, F_blk = assemble_form(
                    Equation(None, res_form),
                    dof_handler=dh,
                    bcs=[],
                    quad_order=int(args.q),
                    backend=str(args.backend),
                )
                out.append((_masked_residual_inf(np.asarray(F_blk, dtype=float), bcs_now), str(name)))
            except Exception as exc:
                out.append((float("inf"), f"{name}(assembly_failed:{exc})"))
        out.sort(reverse=True, key=lambda t: t[0])
        return out

    predictor_base = str(args.predictor)
    predictor_damping_base = float(args.predictor_damping)
    predictor_clip_01_base = bool(args.predictor_clip_01)
    if str(args.support_physics).strip().lower() == "internal_conversion" and predictor_clip_01_base:
        logging.info(
            "[setup] disabling predictor_clip_01 for support_physics=internal_conversion; "
            "mass-preserving PDAS equalities provide the bounded predictor path."
        )
        predictor_clip_01_base = False
    retry_state = {
        "step_no": None,
        "predictor_fallbacks": 0,
        "maxit_boosts": 0,
        "linesearch_fallbacks": 0,
        "gamma_div_retries": 0,
        "dtmin_maxit_retries": 0,
        "petsc_resets": 0,
    }

    def on_step_failure(**info):  # noqa: ANN001
        """
        Debug hook for failed/nonconverged time steps.

        Prints per-field residual norms at the best available iterate so we can
        identify which equation block is driving SNES/Newton failures.
        """
        def _startup_retry(action: str | None):
            if startup_predictor_enabled:
                startup_state["force_next"] = str(action).strip().lower() != "retry_keep_guess"
            return action

        try:
            step_no = int(info.get("step_no", info.get("global_step_no", -1)))
        except Exception:
            step_no = -1
        try:
            t_fail = float(info.get("t", 0.0))
        except Exception:
            t_fail = 0.0
        try:
            dt_fail = float(info.get("dt", float(dt_val)))
        except Exception:
            dt_fail = float(dt_val)
        reason = getattr(solver, "_last_nonlinear_reason", None)
        fnorm = getattr(solver, "_last_nonlinear_norm", None)
        exc = info.get("exception", None)
        exc_msg = str(exc) if exc is not None else ""
        print(f"    [fail] step={step_no} t={t_fail:.6e} dt={dt_fail:.3e} reason={reason} norm={fnorm}")

        # For line-search divergence, the most common cause we've observed in this
        # benchmark is an overly aggressive time-step predictor. Try re-solving the same
        # time step with a safer initial guess before reducing Δt.
        #
        # NOTE: This keeps the same tolerances (rtol=0, accept_factor=0) and only changes
        # the initial guess / solve strategy.
        try:
            reason_i = int(reason) if reason is not None else None
        except Exception:
            reason_i = None
        if retry_state.get("step_no", None) != step_no:
            retry_state["step_no"] = int(step_no)
            retry_state["predictor_fallbacks"] = 0
            retry_state["maxit_boosts"] = 0
            retry_state["linesearch_fallbacks"] = 0
            retry_state["gamma_div_retries"] = 0
            retry_state["dtmin_maxit_retries"] = 0
            retry_state["petsc_resets"] = 0

        line_search_failure = bool(
            reason_i == -6 or (isinstance(exc, RuntimeError) and "Line search failed" in exc_msg)
        )

        # Retry once with predictor='prev' on line-search divergence.
        if line_search_failure and retry_state.get("predictor_fallbacks", 0) < 1 and predictor_base.strip().lower() != "prev":
            retry_state["predictor_fallbacks"] = int(retry_state.get("predictor_fallbacks", 0)) + 1
            try:
                time_params.predictor = "prev"
                time_params.predictor_damping = 0.0
                time_params.predictor_clip_01 = predictor_clip_01_base
            except Exception:
                pass
            print("    [retry] line search failed; retrying same Δt with predictor='prev' (no extrapolation).")
            return _startup_retry("retry")

        # If we are very close to the requested absolute tolerance but hit SNES max-it at dt==dt_min,
        # retry the same step with a higher max-it (keeps rtol=0 and avoids dt collapse).
        try:
            atol = float(args.newton_tol)
        except Exception:
            atol = 0.0
        try:
            dt_min_val = float(getattr(args, "dt_min", 0.0))
        except Exception:
            dt_min_val = 0.0
        try:
            max_it_cap = int(os.getenv("PYCUTFEM_MAX_IT_CAP", "80"))
        except Exception:
            max_it_cap = 80
        try:
            max_it_now = int(getattr(solver.np, "max_newton_iter", int(args.max_it)))
        except Exception:
            max_it_now = int(args.max_it)

        close_enough = bool(atol > 0.0 and fnorm is not None and np.isfinite(float(fnorm)) and float(fnorm) <= 20.0 * atol)
        hit_maxit = bool(reason_i == -5)
        at_dt_min = bool(dt_min_val > 0.0 and dt_fail <= dt_min_val + 1.0e-15)
        failure_diag: dict[str, object] | None = None

        def _compute_failure_diagnostics() -> dict[str, object]:
            nonlocal failure_diag
            if failure_diag is not None:
                return failure_diag
            diag: dict[str, object] = {
                "worst_block": "",
                "worst_field": "",
                "field_norms": [],
                "block_norms": [],
                "momentum_terms": [],
                "kinematics_terms": [],
                "slip_norms": {},
            }
            try:
                funcs = list(info.get("functions", []) or [])
                prev_funcs = list(info.get("prev_functions", []) or [])
                bcs_now = info.get("bcs", None)
                if bcs_now is not None and funcs:
                    dh.apply_bcs(bcs_now, *funcs)
                if getattr(solver, "constraints", None) is not None and funcs:
                    solver._enforce_constraints_on_functions(funcs)

                coeffs = {f.name: f for f in funcs}
                coeffs.update({f.name: f for f in prev_funcs})
                aux = info.get("aux_functions", None)
                if isinstance(aux, dict):
                    coeffs.update(aux)

                _, R_red = solver._assemble_system_reduced(coeffs, need_matrix=False)
                R_full = np.asarray(solver.restrictor.expand_vec(np.asarray(R_red, dtype=float)), dtype=float).ravel()

                field_norms: list[tuple[float, str]] = []
                for fld in getattr(dh, "field_names", []):
                    try:
                        sl = np.asarray(dh.get_field_slice(fld), dtype=int).ravel()
                    except Exception:
                        continue
                    if sl.size == 0:
                        continue
                    field_norms.append((float(np.linalg.norm(R_full[sl], ord=np.inf)), str(fld)))
                field_norms.sort(reverse=True, key=lambda t: t[0])
                diag["field_norms"] = field_norms
                if field_norms:
                    diag["worst_field"] = str(field_norms[0][1])
                n_show = min(12, len(field_norms))
                if n_show:
                    items = ", ".join(f"{name}:{val:.2e}" for val, name in field_norms[:n_show])
                    print(f"    [fail_res] {items}")

                block_norms = _block_residual_breakdown(bcs_now)
                diag["block_norms"] = block_norms
                if block_norms:
                    block_items = ", ".join(f"{name}:{val:.2e}" for val, name in block_norms[: min(8, len(block_norms))])
                    print(f"    [fail_blocks] {block_items}")
                    diag["worst_block"] = str(block_norms[0][1])

                momentum_terms = _named_residual_breakdown(getattr(forms, "r_momentum_terms", None), bcs_now)
                diag["momentum_terms"] = momentum_terms
                if momentum_terms:
                    items = ", ".join(f"{name}:{val:.2e}" for val, name in momentum_terms[: min(10, len(momentum_terms))])
                    print(f"    [fail_mom_terms] {items}")

                kinematics_terms = _named_residual_breakdown(getattr(forms, "r_kinematics_terms", None), bcs_now)
                diag["kinematics_terms"] = kinematics_terms
                if kinematics_terms:
                    items = ", ".join(f"{name}:{val:.2e}" for val, name in kinematics_terms[: min(10, len(kinematics_terms))])
                    print(f"    [fail_kin_terms] {items}")

                coeff_map = {f.name: f for f in funcs}
                v_fun = coeff_map.get("v_k")
                vS_fun = coeff_map.get("vS_k")
                if v_fun is not None and vS_fun is not None:
                    if not hasattr(v_fun, "components") or not hasattr(vS_fun, "components"):
                        print("    [fail_slip] skipped for non-nodal velocity representation.")
                    else:
                        try:
                            vx = np.asarray(v_fun.components[0].nodal_values, dtype=float).ravel()
                            vy = np.asarray(v_fun.components[1].nodal_values, dtype=float).ravel()
                            vsx = np.asarray(vS_fun.components[0].nodal_values, dtype=float).ravel()
                            vsy = np.asarray(vS_fun.components[1].nodal_values, dtype=float).ravel()
                            dxv = vx - vsx
                            dyv = vy - vsy
                            slip_l2 = float(np.sqrt(np.dot(dxv, dxv) + np.dot(dyv, dyv)))
                            slip_inf = float(max(np.max(np.abs(dxv)), np.max(np.abs(dyv))))
                            slip_norms = {
                                "vx_minus_vSx_inf": float(np.max(np.abs(dxv))),
                                "vy_minus_vSy_inf": float(np.max(np.abs(dyv))),
                                "v_minus_vS_inf": slip_inf,
                                "v_minus_vS_l2": slip_l2,
                            }
                            diag["slip_norms"] = slip_norms
                            print(
                                "    [fail_slip] "
                                f"|vx-vSx|_inf={slip_norms['vx_minus_vSx_inf']:.2e}, "
                                f"|vy-vSy|_inf={slip_norms['vy_minus_vSy_inf']:.2e}, "
                                f"|v-vS|_inf={slip_norms['v_minus_vS_inf']:.2e}, "
                                f"|v-vS|_2={slip_norms['v_minus_vS_l2']:.2e}"
                            )
                        except Exception as exc_slip:
                            print(f"    [fail_slip] failed to compute slip mismatch: {exc_slip}")

                if at_dt_min:
                    worst_block = str(diag.get("worst_block", ""))
                    worst_field = str(diag.get("worst_field", ""))
                    if worst_block.startswith("momentum") or worst_block.startswith("skeleton") or worst_field.startswith("v"):
                        print(
                            "    [hint] failure occurred at dt==--dt-min and is dominated by the coupled momentum block; "
                            "for Benchmark 6 this usually indicates lost mechanics coercivity/conditioning rather than a tolerance issue. "
                            "Prefer increasing --gamma-div and/or --gamma-u, keep --u-extension l2, and refine around the biofilm before reducing dt_min further."
                        )
            except Exception as exc_inner:
                print(f"    [fail_res] failed to compute residual breakdown: {exc_inner}")
            failure_diag = diag
            return failure_diag

        # At dt==dt_min, if line search fails, do one retry from the current
        # best iterate (keep guess) before giving up. This avoids touching PETSc's
        # internal linesearch objects (API differences across PETSc builds).
        if line_search_failure and at_dt_min and retry_state.get("linesearch_fallbacks", 0) < 1:
            retry_state["linesearch_fallbacks"] = int(retry_state.get("linesearch_fallbacks", 0)) + 1
            print("    [retry] dt==dt_min and line search failed; retrying from best iterate (keep guess).")
            return _startup_retry("retry_keep_guess")

        if adaptive_gamma_div and line_search_failure and at_dt_min:
            diag = _compute_failure_diagnostics()
            worst_block = str(diag.get("worst_block", ""))
            worst_field = str(diag.get("worst_field", ""))
            momentum_dominated = bool(
                worst_block.startswith("momentum") or worst_block.startswith("skeleton") or worst_field.startswith("v")
            )
            cur_gamma = float(gamma_div_expr)
            if momentum_dominated and cur_gamma + 1.0e-15 < float(gamma_div_max):
                retry_state["gamma_div_retries"] = int(retry_state.get("gamma_div_retries", 0)) + 1
                new_gamma = min(float(gamma_div_max), max(float(cur_gamma) * float(gamma_div_growth), float(cur_gamma) + 1.0e-12))
                _set_gamma_div(
                    new_gamma,
                    reason=f"momentum-dominated line-search failure ({worst_block or worst_field})",
                    step_no=int(step_no),
                    t_s=float(t_fail),
                    dt_s=float(dt_fail),
                )
                try:
                    solver._ls_alpha_prev = 1.0
                except Exception:
                    pass
                return _startup_retry("retry_keep_guess")

        if hit_maxit and at_dt_min:
            retry_state["dtmin_maxit_retries"] = int(retry_state.get("dtmin_maxit_retries", 0)) + 1
            # If we keep hitting max-it at dt_min, SNES/KSP can get stuck in a
            # poor internal state. Rebuild the PETSc stack once and retry.
            if retry_state.get("dtmin_maxit_retries", 0) >= 2 and retry_state.get("petsc_resets", 0) < 1:
                retry_state["petsc_resets"] = int(retry_state.get("petsc_resets", 0)) + 1
                if hasattr(solver, "_invalidate_petsc_cache"):
                    try:
                        solver._invalidate_petsc_cache()
                        print("    [retry] repeated max-it at dt_min; rebuilding PETSc SNES/KSP stack and retrying.")
                        return _startup_retry("retry_keep_guess")
                    except Exception as exc:
                        print(f"    [retry] failed to reset PETSc stack: {exc}")

        if hit_maxit and at_dt_min and close_enough and max_it_now < max_it_cap:
            new_max = min(max_it_cap, max(max_it_now + 5, int(1.5 * max_it_now)))
            try:
                solver.np.max_newton_iter = int(new_max)
                if getattr(solver, "_snes", None) is not None:
                    try:
                        _atol, _rtol, _stol, _max_it, _max_funcs = solver._snes.getTolerances()
                        solver._snes.setTolerances(atol=float(_atol), rtol=float(_rtol), stol=float(_stol), max_it=int(new_max))
                    except Exception:
                        pass
                print(
                    f"    [retry] SNES hit max-it at dt_min with ‖F‖_inf={float(fnorm):.3e}; retrying with --max-it {new_max}."
                )
                return _startup_retry("retry_keep_guess")
            except Exception as exc:
                print(f"    [retry] failed to increase max-it: {exc}")

        # If we are close to atol and just hit SNES max-it, do one continuation retry with
        # a larger max-it before reducing Δt. Keep the current best iterate as the initial guess.
        if hit_maxit and close_enough and max_it_now < max_it_cap and retry_state.get("maxit_boosts", 0) < 1:
            retry_state["maxit_boosts"] = int(retry_state.get("maxit_boosts", 0)) + 1
            new_max = min(max_it_cap, max(max_it_now + 5, int(1.5 * max_it_now)))
            try:
                solver.np.max_newton_iter = int(new_max)
                if getattr(solver, "_snes", None) is not None:
                    try:
                        _atol, _rtol, _stol, _max_it, _max_funcs = solver._snes.getTolerances()
                        solver._snes.setTolerances(atol=float(_atol), rtol=float(_rtol), stol=float(_stol), max_it=int(new_max))
                    except Exception:
                        pass
                print(f"    [retry] SNES hit max-it with ‖F‖_inf={float(fnorm):.3e}; retrying with --max-it {new_max}.")
                return _startup_retry("retry_keep_guess")
            except Exception as exc:
                print(f"    [retry] failed to increase max-it: {exc}")

        _compute_failure_diagnostics()
        return None

    def post_step_refiner(step, bcs_now, functions, prev_functions):
        # Restore baseline predictor after successful steps (handles retries where we
        # temporarily fall back to predictor='prev' for robustness).
        try:
            time_params.predictor = str(predictor_base)
            time_params.predictor_damping = float(predictor_damping_base)
            time_params.predictor_clip_01 = bool(predictor_clip_01_base)
        except Exception:
            pass
        if startup_predictor_enabled:
            startup_state["force_next"] = True
        _copy_function_values(v_k, v_trac_n)
        _copy_function_values(p_k, p_trac_n)
        # Line search is configured via `petsc_opts` on solver creation; avoid mutating
        # PETSc options mid-run (can rewire KSP/PC and lead to size-mismatch errors).

        _post_step_update()
        step_no = int(getattr(solver, "_current_step_no", int(step)))
        t_now = float(getattr(solver, "_current_t", 0.0) + getattr(solver, "_current_dt", dt_val))
        dt_now = float(getattr(solver, "_current_dt", dt_val))
        n_newton = int(getattr(solver, "_last_nonlinear_iterations", 0) or 0)
        relaxed_accept = bool(getattr(solver, "_last_nonlinear_relaxed_accept", False))
        if adaptive_gamma_div:
            cur_gamma = float(gamma_div_expr)
            base_gamma = float(gamma_div_base)
            if cur_gamma > base_gamma + 1.0e-15:
                if (not relaxed_accept) and n_newton <= int(gamma_div_relax_newton_max):
                    gamma_div_state["easy_steps"] = int(gamma_div_state.get("easy_steps", 0)) + 1
                else:
                    gamma_div_state["easy_steps"] = 0
                if (
                    float(gamma_div_relax_factor) < 1.0
                    and int(gamma_div_state.get("easy_steps", 0)) >= int(gamma_div_relax_after)
                ):
                    new_gamma = max(float(base_gamma), float(cur_gamma) * float(gamma_div_relax_factor))
                    gamma_div_state["easy_steps"] = 0
                    _set_gamma_div(
                        new_gamma,
                        reason=f"relax after {int(gamma_div_relax_after)} easy accepted steps (nNewton<={int(gamma_div_relax_newton_max)})",
                        step_no=int(step_no),
                        t_s=float(t_now),
                        dt_s=float(dt_now),
                    )
            else:
                gamma_div_state["easy_steps"] = 0
        if diffuse_scale_update is not None:
            diffuse_scale_update(float(t_now))
        _append_timeseries(t_now)
        _append_interface_diagnostics(t_now)
        while pending_snapshots and t_now + 1.0e-12 >= float(pending_snapshots[0]):
            _write_snapshot_contours(float(t_now), int(step_no))
            pending_snapshots.pop(0)
        if vtk_every > 0 and (step_no % vtk_every == 0):
            vtkf = {"v": v_k, "p": p_k, "vS": vS_k, "u": u_k, "alpha": alpha_k}
            if phi_k is not None:
                vtkf["phi"] = phi_k
            if transport_mode == "pde" and ch_enabled:
                vtkf["mu_alpha"] = mu_alpha_k
            export_vtk(str(vtk_dir / f"step={step_no:04d}.vtu"), mesh, dh, vtkf)
        if restart_write_every > 0 and (step_no % restart_write_every == 0):
            dt_step = float(getattr(solver, "_current_dt", dt_val))
            ckpt = restart_dir / f"checkpoint_step={step_no:05d}.npz"
            _write_restart_checkpoint(
                ckpt,
                t=float(t_now),
                step=int(step_no),
                dt=float(dt_step),
                theta=float(theta),
                gamma_div=float(gamma_div_expr),
                y_lines=np.asarray(y_lines, dtype=float),
                x_ref_global=float(x_ref_global),
                x_ref=np.asarray(x_ref, dtype=float),
                x_prev=np.asarray(x_prev, dtype=float),
                v=v_k,
                p=p_k,
                vS=vS_k,
                u=u_k,
                alpha=alpha_k,
                phi=phi_k,
                S=S_k,
                mu_alpha=mu_alpha_k,
            )
            _write_restart_checkpoint(
                restart_dir / "checkpoint_latest.npz",
                t=float(t_now),
                step=int(step_no),
                dt=float(dt_step),
                theta=float(theta),
                gamma_div=float(gamma_div_expr),
                y_lines=np.asarray(y_lines, dtype=float),
                x_ref_global=float(x_ref_global),
                x_ref=np.asarray(x_ref, dtype=float),
                x_prev=np.asarray(x_prev, dtype=float),
                v=v_k,
                p=p_k,
                vS=vS_k,
                u=u_k,
                alpha=alpha_k,
                phi=phi_k,
                S=S_k,
                mu_alpha=mu_alpha_k,
            )

    # Prime alpha_k/phi_k and write the initial row (t=0 for fresh runs, or t=t0 for restart into a fresh out_dir).
    alpha_k.nodal_values[:] = alpha_n.nodal_values
    if phi_k is not None and phi_n is not None:
        phi_k.nodal_values[:] = phi_n.nodal_values
    mu_alpha_k.nodal_values[:] = mu_alpha_n.nodal_values
    if S_k is not None and S_n is not None:
        S_k.nodal_values[:] = S_n.nodal_values
    if restart_from is None:
        _append_timeseries(0.0)
        _append_interface_diagnostics(0.0)
        while pending_snapshots and abs(float(pending_snapshots[0])) <= 1.0e-14:
            _write_snapshot_contours(0.0, 0)
            pending_snapshots.pop(0)
        if restart_write_every > 0:
            _write_restart_checkpoint(
                restart_dir / "checkpoint_step=00000.npz",
                t=0.0,
                step=0,
                dt=float(dt_val),
                theta=float(theta),
                gamma_div=float(gamma_div_expr),
                y_lines=np.asarray(y_lines, dtype=float),
                x_ref_global=float(x_ref_global),
                x_ref=np.asarray(x_ref, dtype=float),
                x_prev=np.asarray(x_prev, dtype=float),
                v=v_k,
                p=p_k,
                vS=vS_k,
                u=u_k,
                alpha=alpha_k,
                phi=phi_k,
                S=S_k,
                mu_alpha=mu_alpha_k,
            )
            _write_restart_checkpoint(
                restart_dir / "checkpoint_latest.npz",
                t=0.0,
                step=0,
                dt=float(dt_val),
                theta=float(theta),
                gamma_div=float(gamma_div_expr),
                y_lines=np.asarray(y_lines, dtype=float),
                x_ref_global=float(x_ref_global),
                x_ref=np.asarray(x_ref, dtype=float),
                x_prev=np.asarray(x_prev, dtype=float),
                v=v_k,
                p=p_k,
                vS=vS_k,
                u=u_k,
                alpha=alpha_k,
                phi=phi_k,
                S=S_k,
                mu_alpha=mu_alpha_k,
            )
    elif not append_existing_ts:
        _append_timeseries(float(t0))
        _append_interface_diagnostics(float(t0))
        while pending_snapshots and float(t0) + 1.0e-12 >= float(pending_snapshots[0]):
            _write_snapshot_contours(float(t0), int(step0))
            pending_snapshots.pop(0)
        if restart_write_every > 0:
            _write_restart_checkpoint(
                restart_dir / f"checkpoint_step={int(step0):05d}.npz",
                t=float(t0),
                step=int(step0),
                dt=float(dt_val),
                theta=float(theta),
                gamma_div=float(gamma_div_expr),
                y_lines=np.asarray(y_lines, dtype=float),
                x_ref_global=float(x_ref_global),
                x_ref=np.asarray(x_ref, dtype=float),
                x_prev=np.asarray(x_prev, dtype=float),
                v=v_k,
                p=p_k,
                vS=vS_k,
                u=u_k,
                alpha=alpha_k,
                phi=phi_k,
                S=S_k,
                mu_alpha=mu_alpha_k,
            )
            _write_restart_checkpoint(
                restart_dir / "checkpoint_latest.npz",
                t=float(t0),
                step=int(step0),
                dt=float(dt_val),
                theta=float(theta),
                gamma_div=float(gamma_div_expr),
                y_lines=np.asarray(y_lines, dtype=float),
                x_ref_global=float(x_ref_global),
                x_ref=np.asarray(x_ref, dtype=float),
                x_prev=np.asarray(x_prev, dtype=float),
                v=v_k,
                p=p_k,
                vS=vS_k,
                u=u_k,
                alpha=alpha_k,
                phi=phi_k,
                S=S_k,
                mu_alpha=mu_alpha_k,
            )

    functions_k = [v_k, p_k, vS_k, u_k, alpha_k]
    functions_n = [v_n, p_n, vS_n, u_n, alpha_n]
    if phi_k is not None and phi_n is not None:
        functions_k.append(phi_k)
        functions_n.append(phi_n)
    if transport_mode == "pde" and ch_enabled:
        functions_k.append(mu_alpha_k)
        functions_n.append(mu_alpha_n)
    if S_k is not None and S_n is not None:
        functions_k.append(S_k)
        functions_n.append(S_n)
    if float(t0) >= float(args.t_final) - 1.0e-14:
        logging.info(f"[done] restart t0={float(t0):.6e}s >= t_final={float(args.t_final):.6e}s; nothing to do.")
        return
    time_params = TimeStepperParameters(
        dt=dt_val,
        final_time=float(args.t_final),
        max_steps=100_000,
        theta=theta,
        t0=float(t0),
        step0=int(step0),
        stop_on_steady=False,
        steady_tol=0.0,
        allow_dt_reduction=bool(args.allow_dt_reduction),
        dt_min=float(args.dt_min),
        dt_reduction_factor=float(args.dt_reduction_factor),
        on_dt_change=_on_dt_change,
        on_step_failure=on_step_failure,
        predictor=str(predictor_base),
        predictor_damping=float(predictor_damping_base),
        predictor_clip_01=bool(predictor_clip_01_base),
        step_initial_guess_callback=_startup_step_initial_guess_callback if startup_predictor_enabled else None,
    )
    aux_solver_functions: dict[str, object] = {"dt": dt_c, "v_trac_n": v_trac_n, "p_trac_n": p_trac_n}
    if regularization_weight is not None:
        aux_solver_functions["reg_weight"] = regularization_weight
    solver.solve_time_interval(
        functions=functions_k,
        prev_functions=functions_n,
        aux_functions=aux_solver_functions,
        time_params=time_params,
        post_step_refiner=post_step_refiner,
    )


if __name__ == "__main__":
    main()
