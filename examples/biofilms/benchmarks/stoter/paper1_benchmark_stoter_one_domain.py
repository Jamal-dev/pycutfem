#!/usr/bin/env python3
"""Rigid one-domain Stokes--Darcy benchmark on the Stoter geometry.

This driver implements two rigid-limit Stokes--Darcy reductions on the Stoter
Section 5.3 geometry. The support is frozen and only the prescribed diffuse
geometry field remains active:

  - porous indicator `alpha` is prescribed from an Allen--Cahn-smoothed channel
    indicator,
  - free-fluid weight is `F = 1 - alpha`,
  - free-fluid momentum uses the weighted stress form `-F p div(w)`,
  - pore-head traction is transferred through the band term `+ g phi (w·grad(alpha))`,
  - interface mass transfer is transferred through `-psi (u·grad(alpha))`,
  - Darcy transport is `alpha K grad(phi) · grad(psi)`,
  - the diagnostic branch can optionally add a diffuse BJS drag term.

The benchmark is local to the Stoter debug workflow. It is intentionally
independent of the moving-support Seboldt production assembly so the rigid
limit can be audited directly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.biofilms.benchmarks.stoter.paper1_benchmark_stoter_channel_porous import (
    Geometry,
    _eval_scalar_at_point,
    _eval_scalar_grad_at_point,
    _phase_field_indicator_stokes,
    _solve_allen_cahn_indicator,
    _stokes_phase_fields,
    _tag_boundaries,
    _write_velocity_plots,
)
from examples.biofilms.benchmarks.stoter.paper1_benchmark_stoter_channel_porous_twodomain import (
    _compare_velocity_grids,
)
from examples.utils.biofilm.final_form import build_biofilm_one_domain_final_form
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, NewtonSolver
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction, div, dot, grad, inner
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _eps(v):
    return Constant(0.5) * (grad(v) + grad(v).T)


def _sqrt(expr):
    return expr ** Constant(0.5)


def _tag_inactive_domains_by_alpha(
    dh: DofHandler,
    mesh: Mesh,
    alpha_h: Function,
    *,
    alpha_low: float | None,
    alpha_high: float | None,
    porous_in_free_fields: tuple[str, ...] = ("vpx", "vpy", "p_pore"),
    fluid_in_porous_fields: tuple[str, ...] = ("ux", "uy", "p"),
) -> dict[str, dict[str, int]]:
    counts = {
        "porous_in_free_fluid": {str(field): 0 for field in porous_in_free_fields},
        "fluid_in_porous": {str(field): 0 for field in fluid_in_porous_fields},
    }
    inactive = set(int(d) for d in list(getattr(dh, "dof_tags", {}).get("inactive", set()) or set()))
    elem_maps = list(getattr(dh, "element_maps", {}).get("alpha", []) or [])
    n_elem = int(getattr(mesh, "n_elements", len(getattr(mesh, "elements_list", []))))
    low_mask = np.zeros((n_elem,), dtype=bool)
    high_mask = np.zeros((n_elem,), dtype=bool)
    use_low = alpha_low is not None and np.isfinite(float(alpha_low))
    use_high = alpha_high is not None and np.isfinite(float(alpha_high))
    low_thr = float(alpha_low) if use_low else float("nan")
    high_thr = float(alpha_high) if use_high else float("nan")
    for eid, gds in enumerate(elem_maps):
        if eid >= n_elem:
            break
        g_arr = np.asarray(gds, dtype=int).ravel()
        if g_arr.size == 0:
            continue
        nodal = np.asarray(alpha_h.get_nodal_values(g_arr), dtype=float)
        if nodal.size == 0 or not np.all(np.isfinite(nodal)):
            continue
        if use_low and np.all(nodal <= low_thr):
            low_mask[eid] = True
        if use_high and np.all(nodal >= high_thr):
            high_mask[eid] = True
    if use_low:
        for field in porous_in_free_fields:
            selected = dh.tag_dofs_from_element_bitset("inactive", field, low_mask, strict=True)
            selected_set = set(int(g) for g in selected)
            inactive.update(selected_set)
            counts["porous_in_free_fluid"][str(field)] = int(len(selected_set))
    if use_high:
        for field in fluid_in_porous_fields:
            selected = dh.tag_dofs_from_element_bitset("inactive", field, high_mask, strict=True)
            selected_set = set(int(g) for g in selected)
            inactive.update(selected_set)
            counts["fluid_in_porous"][str(field)] = int(len(selected_set))
    dh.dof_tags["inactive"] = inactive
    return counts


def _tag_interface_multiplier_band(
    dh: DofHandler,
    mesh: Mesh,
    alpha_h: Function,
    *,
    alpha_low: float | None,
    alpha_high: float | None,
    fields: tuple[str, ...] = ("mu_mass", "mu_normal", "mu_tangent"),
) -> dict[str, int]:
    counts = {str(field): 0 for field in fields}
    use_low = alpha_low is not None and np.isfinite(float(alpha_low))
    use_high = alpha_high is not None and np.isfinite(float(alpha_high))
    if not use_low and not use_high:
        return counts
    elem_maps = list(getattr(dh, "element_maps", {}).get("alpha", []) or [])
    n_elem = int(getattr(mesh, "n_elements", len(getattr(mesh, "elements_list", []))))
    inactive_mask = np.zeros((n_elem,), dtype=bool)
    low_thr = float(alpha_low) if use_low else float("nan")
    high_thr = float(alpha_high) if use_high else float("nan")
    for eid, gds in enumerate(elem_maps):
        if eid >= n_elem:
            break
        g_arr = np.asarray(gds, dtype=int).ravel()
        if g_arr.size == 0:
            continue
        nodal = np.asarray(alpha_h.get_nodal_values(g_arr), dtype=float)
        if nodal.size == 0 or not np.all(np.isfinite(nodal)):
            continue
        if use_low and np.all(nodal <= low_thr):
            inactive_mask[eid] = True
            continue
        if use_high and np.all(nodal >= high_thr):
            inactive_mask[eid] = True
    for field in fields:
        selected = dh.tag_dofs_from_element_bitset("inactive", str(field), inactive_mask, strict=True)
        counts[str(field)] = int(len(set(int(g) for g in selected)))
    return counts


def _write_centerline_samples(
    outdir: Path,
    dh: DofHandler,
    mesh: Mesh,
    geom: Geometry,
    ux,
    uy,
    vpx,
    vpy,
    p_f,
    p_pore,
    alpha_h,
) -> Path:
    xs = np.full(201, float(geom.center_x))
    ys = np.linspace(0.0, float(geom.Ly), 201)
    rows = ["y_mm,alpha,c,c_D,ux_free,uy_free,ux_darcy,uy_darcy,ux_combined,uy_combined,p,p_pore"]
    for x, y in zip(xs, ys):
        alpha = _eval_scalar_at_point(dh, mesh, "alpha", alpha_h, (float(x), float(y)))
        free_w = max(0.0, 1.0 - float(alpha))
        ux_h = _eval_scalar_at_point(dh, mesh, "ux", ux, (float(x), float(y)))
        uy_h = _eval_scalar_at_point(dh, mesh, "uy", uy, (float(x), float(y)))
        ux_d = _eval_scalar_at_point(dh, mesh, "vpx", vpx, (float(x), float(y)))
        uy_d = _eval_scalar_at_point(dh, mesh, "vpy", vpy, (float(x), float(y)))
        ux_c = free_w * ux_h + float(alpha) * ux_d
        uy_c = free_w * uy_h + float(alpha) * uy_d
        p_h = _eval_scalar_at_point(dh, mesh, "p", p_f, (float(x), float(y)))
        phi_h = _eval_scalar_at_point(dh, mesh, "p_pore", p_pore, (float(x), float(y)))
        rows.append(
            f"{float(y):.12e},{float(alpha):.12e},{free_w:.12e},{float(alpha):.12e},{ux_h:.12e},{uy_h:.12e},{ux_d:.12e},{uy_d:.12e},{ux_c:.12e},{uy_c:.12e},{float(p_h):.12e},{float(phi_h):.12e}"
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
    vpx,
    vpy,
    p_f,
    p_pore,
    alpha_h,
    *,
    nx: int = 41,
    ny: int = 51,
) -> Path:
    xs = np.linspace(0.0, float(geom.Lx), int(nx))
    ys = np.linspace(0.0, float(geom.Ly), int(ny))
    rows = ["x_mm,y_mm,alpha,c,c_D,ux_free,uy_free,ux_darcy,uy_darcy,ux_combined,uy_combined,p,p_pore"]
    for y in ys:
        for x in xs:
            alpha = _eval_scalar_at_point(dh, mesh, "alpha", alpha_h, (float(x), float(y)))
            free_w = max(0.0, 1.0 - float(alpha))
            ux_h = _eval_scalar_at_point(dh, mesh, "ux", ux, (float(x), float(y)))
            uy_h = _eval_scalar_at_point(dh, mesh, "uy", uy, (float(x), float(y)))
            p_h = _eval_scalar_at_point(dh, mesh, "p", p_f, (float(x), float(y)))
            phi_h = _eval_scalar_at_point(dh, mesh, "p_pore", p_pore, (float(x), float(y)))
            ux_d = _eval_scalar_at_point(dh, mesh, "vpx", vpx, (float(x), float(y)))
            uy_d = _eval_scalar_at_point(dh, mesh, "vpy", vpy, (float(x), float(y)))
            ux_c = free_w * ux_h + float(alpha) * ux_d
            uy_c = free_w * uy_h + float(alpha) * uy_d
            rows.append(
                f"{float(x):.12e},{float(y):.12e},{float(alpha):.12e},{free_w:.12e},{float(alpha):.12e},{ux_h:.12e},{uy_h:.12e},{ux_d:.12e},{uy_d:.12e},{ux_c:.12e},{uy_c:.12e},{float(p_h):.12e},{float(phi_h):.12e}"
            )
    path = outdir / "velocity_grid.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _sample_contamination_metrics(
    velocity_grid_csv: Path,
    *,
    alpha_low: float | None,
    alpha_high: float | None,
) -> dict[str, float]:
    data = np.genfromtxt(str(velocity_grid_csv), delimiter=",", names=True, dtype=float)
    if getattr(data, "shape", ()) == ():
        data = np.asarray([data], dtype=data.dtype)
    alpha = np.asarray(data["alpha"], dtype=float)
    ux_free = np.asarray(data["ux_free"], dtype=float)
    uy_free = np.asarray(data["uy_free"], dtype=float)
    ux_darcy = np.asarray(data["ux_darcy"], dtype=float)
    uy_darcy = np.asarray(data["uy_darcy"], dtype=float)
    p_free = np.asarray(data["p"], dtype=float)
    p_pore = np.asarray(data["p_pore"], dtype=float)

    free_speed = np.sqrt(ux_free * ux_free + uy_free * uy_free)
    darcy_speed = np.sqrt(ux_darcy * ux_darcy + uy_darcy * uy_darcy)

    metrics: dict[str, float] = {}
    if alpha_low is not None and np.isfinite(float(alpha_low)):
        free_mask = alpha <= float(alpha_low)
        metrics["sample_points_free_region"] = float(np.count_nonzero(free_mask))
        metrics["pore_speed_in_free_max"] = float(np.max(darcy_speed[free_mask])) if np.any(free_mask) else 0.0
        metrics["pore_pressure_in_free_max_abs"] = float(np.max(np.abs(p_pore[free_mask]))) if np.any(free_mask) else 0.0
    if alpha_high is not None and np.isfinite(float(alpha_high)):
        porous_mask = alpha >= float(alpha_high)
        metrics["sample_points_porous_region"] = float(np.count_nonzero(porous_mask))
        metrics["free_speed_in_porous_max"] = float(np.max(free_speed[porous_mask])) if np.any(porous_mask) else 0.0
        metrics["free_pressure_in_porous_max_abs"] = float(np.max(np.abs(p_free[porous_mask]))) if np.any(porous_mask) else 0.0
    return metrics


def _centerline_darcy_fit_metrics(
    centerline_csv: Path,
    *,
    K: float,
    phi0: float,
    y_probe: float,
    darcy_probe: np.ndarray,
) -> dict[str, float | list[float]]:
    data = np.genfromtxt(str(centerline_csv), delimiter=",", names=True, dtype=float)
    if getattr(data, "shape", ()) == ():
        data = np.asarray([data], dtype=data.dtype)
    y_vals = np.asarray(data["y_mm"], dtype=float)
    p_vals = np.asarray(data["p_pore"], dtype=float)
    if y_vals.size < 2 or not np.all(np.isfinite(y_vals)) or not np.all(np.isfinite(p_vals)):
        return {}
    slope, intercept = np.polyfit(y_vals, p_vals, 1)
    expected_vy = -(float(K) / float(phi0)) * float(slope)
    probe_idx = int(np.argmin(np.abs(y_vals - float(y_probe))))
    fit_probe_p = float(slope) * float(y_vals[probe_idx]) + float(intercept)
    return {
        "darcy_centerline_pressure_fit_slope": float(slope),
        "darcy_centerline_pressure_fit_intercept": float(intercept),
        "darcy_centerline_expected_vy_from_fit_mm_per_s": float(expected_vy),
        "darcy_centerline_vy_probe_mm_per_s": float(darcy_probe[1]),
        "darcy_centerline_vy_fit_residual_mm_per_s": float(darcy_probe[1] - expected_vy),
        "darcy_centerline_pressure_fit_probe": float(fit_probe_p),
    }


def _solve_benchmark_final_form_rigid(
    *,
    nx: int,
    ny: int,
    eps: float,
    Umax: float,
    K: float,
    nu: float,
    friction_alpha: float,
    g: float,
    backend: str,
    export: bool,
    outdir: Path,
    phase_field_mode: str,
    alpha_mode: str,
    ac_dt: float,
    ac_max_steps: int,
    compare_sharp_csv: Path | None,
    phi0: float,
    gamma_v_in_porous: float,
    gamma_p_in_porous: float,
    gamma_vp_in_free: float,
    gamma_p_pore_in_free: float,
    inactive_alpha_low: float | None,
    inactive_alpha_high: float | None,
    multiplier_regularization: float,
) -> dict[str, object]:
    geom = Geometry()
    nodes, elems, _, corners = structured_quad(
        float(geom.Lx),
        float(geom.Ly),
        nx=int(nx),
        ny=int(ny),
        poly_order=1,
        offset=(0.0, 0.0),
    )
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)
    _tag_boundaries(mesh, geom)

    field_specs = {
        "ux": 1,
        "uy": 1,
        "p": 1,
        "p_pore": 1,
        "vpx": 1,
        "vpy": 1,
        "vSx": 1,
        "vSy": 1,
        "dx": 1,
        "dy": 1,
        "alpha": 1,
        "phi": 1,
        "rho_s": 1,
        "mu_mass": 1,
        "mu_normal": 1,
        "mu_tangent": 1,
    }
    me = MixedElement(mesh, field_specs=field_specs)
    dh = DofHandler(me, method="cg")

    h = min(float(geom.Lx) / float(nx), float(geom.Ly) / float(ny))
    tol_pin = 0.51 * h
    dh.tag_dof_by_locator(
        "pressure_pin",
        "p",
        lambda x, y: abs(x - float(geom.center_x)) <= tol_pin and abs(y - 0.0) <= tol_pin,
    )
    dh.tag_dof_by_locator(
        "pore_pressure_pin",
        "p_pore",
        lambda x, y: abs(x - float(geom.center_x)) <= tol_pin and abs(y - 0.0) <= tol_pin,
    )

    V = FunctionSpace("velocity", ["ux", "uy"], dim=1)
    VP = FunctionSpace("pore_velocity", ["vpx", "vpy"], dim=1)
    VS = FunctionSpace("solid_velocity", ["vSx", "vSy"], dim=1)
    U = FunctionSpace("solid_disp", ["dx", "dy"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    v_test = VectorTestFunction(space=V, dof_handler=dh)
    dvP = VectorTrialFunction(space=VP, dof_handler=dh)
    vP_test = VectorTestFunction(space=VP, dof_handler=dh)
    dvS = VectorTrialFunction(space=VS, dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)

    dp = TrialFunction("p", dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    dp_pore = TrialFunction("p_pore", dof_handler=dh)
    q_pore_test = TestFunction("p_pore", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    drho_s = TrialFunction("rho_s", dof_handler=dh)
    rho_s_test = TestFunction("rho_s", dof_handler=dh)
    dmu_mass = TrialFunction("mu_mass", dof_handler=dh)
    mu_mass_test = TestFunction("mu_mass", dof_handler=dh)
    dmu_normal = TrialFunction("mu_normal", dof_handler=dh)
    mu_normal_test = TestFunction("mu_normal", dof_handler=dh)
    dmu_tangent = TrialFunction("mu_tangent", dof_handler=dh)
    mu_tangent_test = TestFunction("mu_tangent", dof_handler=dh)

    v_k = VectorFunction("v_k", ["ux", "uy"], dof_handler=dh)
    v_n = VectorFunction("v_n", ["ux", "uy"], dof_handler=dh)
    vP_k = VectorFunction("vP_k", ["vpx", "vpy"], dof_handler=dh)
    vP_n = VectorFunction("vP_n", ["vpx", "vpy"], dof_handler=dh)
    vS_k = VectorFunction("vS_k", ["vSx", "vSy"], dof_handler=dh)
    vS_n = VectorFunction("vS_n", ["vSx", "vSy"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["dx", "dy"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["dx", "dy"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    p_pore_k = Function("p_pore_k", "p_pore", dof_handler=dh)
    p_pore_n = Function("p_pore_n", "p_pore", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    rho_s_k = Function("rho_s_k", "rho_s", dof_handler=dh)
    rho_s_n = Function("rho_s_n", "rho_s", dof_handler=dh)
    mu_mass_k = Function("mu_mass_k", "mu_mass", dof_handler=dh)
    mu_mass_n = Function("mu_mass_n", "mu_mass", dof_handler=dh)
    mu_normal_k = Function("mu_normal_k", "mu_normal", dof_handler=dh)
    mu_normal_n = Function("mu_normal_n", "mu_normal", dof_handler=dh)
    mu_tangent_k = Function("mu_tangent_k", "mu_tangent", dof_handler=dh)
    mu_tangent_n = Function("mu_tangent_n", "mu_tangent", dof_handler=dh)

    for f in (
        v_k,
        v_n,
        vP_k,
        vP_n,
        vS_k,
        vS_n,
        u_k,
        u_n,
        p_k,
        p_n,
        p_pore_k,
        p_pore_n,
        alpha_k,
        alpha_n,
        phi_k,
        phi_n,
        rho_s_k,
        rho_s_n,
        mu_mass_k,
        mu_mass_n,
        mu_normal_k,
        mu_normal_n,
        mu_tangent_k,
        mu_tangent_n,
    ):
        f.nodal_values.fill(0.0)

    alpha_mode_key = str(alpha_mode or "diffuse").strip().lower().replace("-", "_")
    if alpha_mode_key not in {"diffuse", "all_free", "all_porous"}:
        raise ValueError("alpha_mode must be 'diffuse', 'all_free', or 'all_porous'.")
    if alpha_mode_key == "all_free":
        alpha_k.nodal_values.fill(0.0)
        alpha_meta = {"mode": "uniform_all_free"}
    elif alpha_mode_key == "all_porous":
        alpha_k.nodal_values.fill(1.0)
        alpha_meta = {"mode": "uniform_all_porous"}
    else:
        phase_field_mode_key = str(phase_field_mode).strip().lower().replace("-", "_")
        if phase_field_mode_key not in {"analytic", "allen_cahn", "allen-cahn"}:
            raise ValueError("phase_field_mode must be 'analytic' or 'allen_cahn'.")
        if phase_field_mode_key == "analytic":
            c_fun, _, _, _, _ = _stokes_phase_fields(geom, eps=float(eps))
            alpha_fun = lambda x, y: 1.0 - float(c_fun(x, y))
            alpha_k.set_values_from_function(alpha_fun)
            alpha_meta = {"mode": "analytic_surrogate"}
        else:
            c_vals, c_meta = _solve_allen_cahn_indicator(
                mesh=mesh,
                eps=float(eps),
                indicator_fun=_phase_field_indicator_stokes(geom),
                dt=float(ac_dt),
                max_steps=int(ac_max_steps),
                backend=backend,
            )
            alpha_vals = np.clip(1.0 - np.asarray(c_vals, dtype=float), 0.0, 1.0)
            if alpha_k.nodal_values.shape != alpha_vals.shape:
                raise RuntimeError("Allen--Cahn alpha field size mismatch.")
            alpha_k.nodal_values[:] = alpha_vals
            alpha_meta = {"mode": "allen_cahn", "alpha_from_stokes_indicator": c_meta}
    alpha_k.nodal_values[:] = np.clip(np.asarray(alpha_k.nodal_values, dtype=float), 0.0, 1.0)
    alpha_n.nodal_values[:] = alpha_k.nodal_values[:]
    phi_k.nodal_values[:] = float(phi0)
    phi_n.nodal_values[:] = float(phi0)
    rho_s_k.nodal_values[:] = 1.0
    rho_s_n.nodal_values[:] = 1.0

    dh.dof_tags["inactive"] = set()
    full_mask = np.ones((int(getattr(mesh, "n_elements", len(getattr(mesh, "elements_list", [])))),), dtype=bool)
    for field in ("vSx", "vSy", "dx", "dy", "alpha", "phi", "rho_s"):
        dh.tag_dofs_from_element_bitset("inactive", field, full_mask, strict=True)
    if alpha_mode_key == "diffuse":
        # In the mixed diffuse case, keep the pressures globally active and use
        # the weak H1 extensions to control them, but still allow a configurable
        # deep-tail cut for the wrong-side velocities.
        inactive_domain_counts = _tag_inactive_domains_by_alpha(
            dh,
            mesh,
            alpha_k,
            alpha_low=inactive_alpha_low,
            alpha_high=inactive_alpha_high,
            porous_in_free_fields=("vpx", "vpy"),
            fluid_in_porous_fields=("ux", "uy"),
        )
    else:
        inactive_domain_counts = _tag_inactive_domains_by_alpha(
            dh,
            mesh,
            alpha_k,
            alpha_low=inactive_alpha_low,
            alpha_high=inactive_alpha_high,
            porous_in_free_fields=("vpx", "vpy", "p_pore"),
            fluid_in_porous_fields=("ux", "uy", "p"),
        )
    interface_multiplier_inactive = _tag_interface_multiplier_band(
        dh,
        mesh,
        alpha_k,
        alpha_low=inactive_alpha_low,
        alpha_high=inactive_alpha_high,
    )

    forms = build_biofilm_one_domain_final_form(
        v_k=v_k,
        p_k=p_k,
        vP_k=vP_k,
        p_pore_k=p_pore_k,
        vS_k=vS_k,
        u_k=u_k,
        alpha_k=alpha_k,
        phi_k=phi_k,
        rho_s_k=rho_s_k,
        v_n=v_n,
        p_n=p_n,
        vP_n=vP_n,
        p_pore_n=p_pore_n,
        vS_n=vS_n,
        u_n=u_n,
        alpha_n=alpha_n,
        phi_n=phi_n,
        rho_s_n=rho_s_n,
        dv=dv,
        dp=dp,
        dvP=dvP,
        dp_pore=dp_pore,
        dvS=dvS,
        du=du,
        dalpha=dalpha,
        dphi=dphi,
        drho_s=drho_s,
        dmu_mass=dmu_mass,
        dmu_normal=dmu_normal,
        dmu_tangent=dmu_tangent,
        v_test=v_test,
        q_test=q_test,
        q_pore_test=q_pore_test,
        vP_test=vP_test,
        vS_test=vS_test,
        u_test=u_test,
        alpha_test=alpha_test,
        phi_test=phi_test,
        rho_s_test=rho_s_test,
        mu_mass_test=mu_mass_test,
        mu_normal_test=mu_normal_test,
        mu_tangent_test=mu_tangent_test,
        mu_mass_k=mu_mass_k,
        mu_normal_k=mu_normal_k,
        mu_tangent_k=mu_tangent_k,
        dx=dx(),
        dt=Constant(1.0e12),
        rho_f=Constant(1.0),
        mu_f=Constant(float(nu)),
        kappa_inv=Constant(1.0 / float(K)),
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        solid_visco_eta=0.0,
        gamma_phi=0.0,
        gamma_v=float(gamma_v_in_porous),
        v_extension_mode="h1",
        gamma_v_pin=0.0,
        gamma_p=float(gamma_p_in_porous),
        p_extension_mode="h1",
        gamma_p_pin=0.0,
        gamma_vP=float(gamma_vp_in_free),
        vP_extension_mode="h1",
        gamma_vP_pin=0.0,
        gamma_p_pore=float(gamma_p_pore_in_free),
        p_pore_extension_mode="h1",
        gamma_p_pore_pin=0.0,
        gamma_rho_s=0.0,
        rho_s_extension_mode="l2",
        gamma_rho_s_pin=0.0,
        rho_s_ref=1.0,
        constant_rho_s=True,
        gamma_u=0.0,
        u_extension_mode="l2",
        gamma_u_pin=0.0,
        gamma_vS=0.0,
        vS_extension_mode="l2",
        gamma_vS_pin=0.0,
        fluid_convection="off",
        pore_convection="off",
        skeleton_inertia_convection="off",
        solid_model="linear",
        phi_mode="alpha_closure",
        phi_b=float(phi0),
        # The rigid debug audit uses sigma_p = -p_p I directly, so the normal
        # interface law is grad(alpha)·(sigma_f - phi0 sigma_p) = 0 without an
        # extra hydraulic-head scaling factor.
        normal_pressure_scale=1.0,
        normal_constraint_carrier="multiplier",
        rigid_darcy_head_mode=True,
        # The rigid debug reduction follows the simplified audit note exactly:
        # interface transfer is only via the explicit mass and traction laws.
        bjs_coefficient=0.0,
        interface_formulation="decomposed",
    )

    residual_form = forms.residual_form
    jacobian_form = forms.jacobian_form
    multiplier_reg_value = float(multiplier_regularization)
    if multiplier_reg_value != 0.0:
        mu_reg_c = Constant(multiplier_reg_value)
        residual_form += mu_reg_c * (
            mu_mass_k * mu_mass_test + mu_normal_k * mu_normal_test + mu_tangent_k * mu_tangent_test
        ) * dx()
        jacobian_form += mu_reg_c * (
            dmu_mass * mu_mass_test + dmu_normal * mu_normal_test + dmu_tangent * mu_tangent_test
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
    if alpha_mode_key == "all_porous":
        bcs = [
            BoundaryCondition("vpx", "dirichlet", "bottom_active", zero),
            BoundaryCondition("vpy", "dirichlet", "bottom_active", uy_inflow),
            BoundaryCondition("vpx", "dirichlet", "bottom_rest", zero),
            BoundaryCondition("vpy", "dirichlet", "bottom_rest", zero),
            BoundaryCondition("p_pore", "dirichlet", "pore_pressure_pin", zero),
        ]
        bcs_homog = [
            BoundaryCondition("vpx", "dirichlet", "bottom_active", zero),
            BoundaryCondition("vpy", "dirichlet", "bottom_active", zero),
            BoundaryCondition("vpx", "dirichlet", "bottom_rest", zero),
            BoundaryCondition("vpy", "dirichlet", "bottom_rest", zero),
            BoundaryCondition("p_pore", "dirichlet", "pore_pressure_pin", zero),
        ]
    else:
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
        newton_params=NewtonParameters(
            newton_tol=1.0e-4,
            max_newton_iter=60,
            line_search=True,
            ls_fail_hard=False,
            accept_nonconverged_atol_factor=2.0,
        ),
        lin_params=LinearSolverParameters(backend="scipy"),
        backend=backend,
    )
    bcs_now = NewtonSolver._freeze_bcs(bcs, 0.0)
    functions = [
        v_k,
        p_k,
        p_pore_k,
        vP_k,
        vS_k,
        u_k,
        alpha_k,
        phi_k,
        rho_s_k,
        mu_mass_k,
        mu_normal_k,
        mu_tangent_k,
    ]
    prev_functions = [
        v_n,
        p_n,
        p_pore_n,
        vP_n,
        vS_n,
        u_n,
        alpha_n,
        phi_n,
        rho_s_n,
        mu_mass_n,
        mu_normal_n,
        mu_tangent_n,
    ]
    dh.apply_bcs(bcs_now, *functions)
    dh.apply_bcs(bcs_now, *prev_functions)
    aux_funcs = {"alpha_k": alpha_k}
    _, converged, iters_total = solver._newton_loop(functions, prev_functions, aux_funcs, bcs_now)
    if not bool(converged):
        raise RuntimeError("Rigid final_form Stoter solve did not converge.")

    if export:
        export_vtk(
            str(outdir / "final_state.vtu"),
            mesh=mesh,
            dof_handler=dh,
            functions={
                "u_free": v_k,
                "u_pore": vP_k,
                "p_free": p_k,
                "p_pore": p_pore_k,
                "alpha": alpha_k,
                "mu_mass": mu_mass_k,
                "mu_normal": mu_normal_k,
                "mu_tangent": mu_tangent_k,
            },
        )

    centerline_path = _write_centerline_samples(
        outdir,
        dh,
        mesh,
        geom,
        v_k[0],
        v_k[1],
        vP_k[0],
        vP_k[1],
        p_k,
        p_pore_k,
        alpha_k,
    )
    velocity_grid_path = _write_velocity_grid_samples(
        outdir,
        dh,
        mesh,
        geom,
        v_k[0],
        v_k[1],
        vP_k[0],
        vP_k[1],
        p_k,
        p_pore_k,
        alpha_k,
    )
    plot_paths = _write_velocity_plots(outdir, velocity_grid_path, geom, umax=float(Umax))
    contamination_metrics = _sample_contamination_metrics(
        velocity_grid_path,
        alpha_low=inactive_alpha_low,
        alpha_high=inactive_alpha_high,
    )

    x_probe = float(geom.center_x)
    y_probe = 50.0
    alpha_probe = _eval_scalar_at_point(dh, mesh, "alpha", alpha_k, (x_probe, y_probe))
    ux_probe = _eval_scalar_at_point(dh, mesh, "ux", v_k[0], (x_probe, y_probe))
    uy_probe = _eval_scalar_at_point(dh, mesh, "uy", v_k[1], (x_probe, y_probe))
    darcy_probe = np.asarray(
        [
            _eval_scalar_at_point(dh, mesh, "vpx", vP_k[0], (x_probe, y_probe)),
            _eval_scalar_at_point(dh, mesh, "vpy", vP_k[1], (x_probe, y_probe)),
        ],
        dtype=float,
    )
    darcy_grad_probe = np.asarray(
        _eval_scalar_grad_at_point(dh, mesh, "p_pore", p_pore_k, (x_probe, y_probe)),
        dtype=float,
    )
    combined_probe = (1.0 - float(alpha_probe)) * np.asarray([ux_probe, uy_probe], dtype=float) + float(alpha_probe) * darcy_probe

    summary: dict[str, object] = {
        "benchmark": "stoter_section_5_3_one_domain_rigid_limit",
        "formulation": "final_form_rigid",
        "phi0": float(phi0),
        "backend": str(backend),
        "nx": int(nx),
        "ny": int(ny),
        "eps": float(eps),
        "Umax_mm_per_s": float(Umax),
        "K_mm_per_s": float(K),
        "nu_mm2_per_s": float(nu),
        "h": float(h),
        "phase_field_mode": str(alpha_meta.get("mode", phase_field_mode)),
        "alpha_mode": alpha_mode_key,
        "phase_field_meta": alpha_meta,
        "nonlinear_iterations": int(iters_total),
        "friction_alpha_requested": float(friction_alpha),
        "applied_bjs_coefficient": 0.0,
        "multiplier_regularization": multiplier_reg_value,
        "gamma_v_in_porous": float(gamma_v_in_porous),
        "gamma_p_in_porous": float(gamma_p_in_porous),
        "gamma_vp_in_free": float(gamma_vp_in_free),
        "gamma_p_pore_in_free": float(gamma_p_pore_in_free),
        "inactive_alpha_low": (
            None if inactive_alpha_low is None or not np.isfinite(float(inactive_alpha_low)) else float(inactive_alpha_low)
        ),
        "inactive_alpha_high": (
            None if inactive_alpha_high is None or not np.isfinite(float(inactive_alpha_high)) else float(inactive_alpha_high)
        ),
        "inactive_multiplier_counts": interface_multiplier_inactive,
        "inactive_domain_counts": {
            block: {field: int(count) for field, count in counts.items()}
            for block, counts in inactive_domain_counts.items()
        },
        "center_probe_xy_mm": [x_probe, y_probe],
        "alpha_probe": float(alpha_probe),
        "free_velocity_probe_mm_per_s": [float(ux_probe), float(uy_probe)],
        "darcy_velocity_probe_mm_per_s": [float(darcy_probe[0]), float(darcy_probe[1])],
        "darcy_pressure_grad_probe": [float(darcy_grad_probe[0]), float(darcy_grad_probe[1])],
        "combined_velocity_probe_mm_per_s": [float(combined_probe[0]), float(combined_probe[1])],
        "centerline_csv": str(centerline_path),
        "velocity_grid_csv": str(velocity_grid_path),
        **contamination_metrics,
        **plot_paths,
    }
    if alpha_mode_key == "all_porous" and float(phi0) != 0.0:
        summary.update(
            _centerline_darcy_fit_metrics(
                centerline_path,
                K=float(K),
                phi0=float(phi0),
                y_probe=y_probe,
                darcy_probe=darcy_probe,
            )
        )
    if alpha_mode_key == "diffuse" and compare_sharp_csv is not None:
        summary["sharp_comparison"] = _compare_velocity_grids(compare_sharp_csv, velocity_grid_path)
    with (outdir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
    return summary


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
    alpha_mode: str,
    ac_dt: float,
    ac_max_steps: int,
    compare_sharp_csv: Path | None,
    formulation: str,
    phi0: float,
    gamma_v_in_porous: float,
    gamma_p_in_porous: float,
    gamma_vp_in_free: float,
    gamma_p_pore_in_free: float,
    inactive_alpha_low: float | None,
    inactive_alpha_high: float | None,
    multiplier_regularization: float,
) -> dict[str, object]:
    formulation_key = str(formulation or "diagnostic").strip().lower()
    if formulation_key not in {"diagnostic", "final_form_rigid"}:
        raise ValueError("formulation must be 'diagnostic' or 'final_form_rigid'.")
    if formulation_key == "final_form_rigid":
        return _solve_benchmark_final_form_rigid(
            nx=int(nx),
            ny=int(ny),
            eps=float(eps),
            Umax=float(Umax),
            K=float(K),
            nu=float(nu),
            friction_alpha=float(friction_alpha),
            g=float(g),
            backend=str(backend),
            export=bool(export),
            outdir=outdir,
            phase_field_mode=str(phase_field_mode),
            alpha_mode=str(alpha_mode),
            ac_dt=float(ac_dt),
            ac_max_steps=int(ac_max_steps),
            compare_sharp_csv=compare_sharp_csv,
            phi0=float(phi0),
            gamma_v_in_porous=float(gamma_v_in_porous),
            gamma_p_in_porous=float(gamma_p_in_porous),
            gamma_vp_in_free=float(gamma_vp_in_free),
            gamma_p_pore_in_free=float(gamma_p_pore_in_free),
            inactive_alpha_low=inactive_alpha_low,
            inactive_alpha_high=inactive_alpha_high,
            multiplier_regularization=float(multiplier_regularization),
        )

    geom = Geometry()

    nodes, elems, _, corners = structured_quad(float(geom.Lx), float(geom.Ly), nx=int(nx), ny=int(ny), poly_order=1, offset=(0.0, 0.0))
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)
    _tag_boundaries(mesh, geom)

    me = MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "vpx": 1, "vpy": 1, "p": 1, "p_pore": 1, "alpha": 1})
    dh = DofHandler(me, method="cg")

    h = min(float(geom.Lx) / float(nx), float(geom.Ly) / float(ny))
    tol_pin = 0.51 * h
    dh.tag_dof_by_locator(
        "pressure_pin",
        "p",
        lambda x, y: abs(x - float(geom.center_x)) <= tol_pin and abs(y - 0.0) <= tol_pin,
    )

    vel_space = FunctionSpace("velocity", ["ux", "uy"], dim=1)
    pore_space = FunctionSpace("pore_velocity", ["vpx", "vpy"], dim=1)
    du = VectorTrialFunction(space=vel_space, dof_handler=dh)
    w = VectorTestFunction(space=vel_space, dof_handler=dh)
    dvp = VectorTrialFunction(space=pore_space, dof_handler=dh)
    wp = VectorTestFunction(space=pore_space, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    q = TestFunction("p", dof_handler=dh)
    dphi = TrialFunction("p_pore", dof_handler=dh)
    psi = TestFunction("p_pore", dof_handler=dh)

    u_k = VectorFunction("u_k", ["ux", "uy"], dof_handler=dh)
    vp_k = VectorFunction("vp_k", ["vpx", "vpy"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    p_pore_k = Function("p_pore_k", "p_pore", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    u_n = VectorFunction("u_n", ["ux", "uy"], dof_handler=dh)
    vp_n = VectorFunction("vp_n", ["vpx", "vpy"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    p_pore_n = Function("p_pore_n", "p_pore", dof_handler=dh)
    for f in (u_k, vp_k, p_k, p_pore_k, alpha_k, u_n, vp_n, p_n, p_pore_n):
        f.nodal_values.fill(0.0)

    phase_field_mode_key = str(phase_field_mode).strip().lower().replace("-", "_")
    if phase_field_mode_key not in {"analytic", "allen_cahn", "allen-cahn"}:
        raise ValueError("phase_field_mode must be 'analytic' or 'allen_cahn'.")
    if phase_field_mode_key == "analytic":
        c_fun, _, _, _, _ = _stokes_phase_fields(geom, eps=float(eps))
        alpha_fun = lambda x, y: 1.0 - float(c_fun(x, y))
        alpha_k.set_values_from_function(alpha_fun)
        alpha_meta = {"mode": "analytic_surrogate"}
    else:
        c_vals, c_meta = _solve_allen_cahn_indicator(
            mesh=mesh,
            eps=float(eps),
            indicator_fun=_phase_field_indicator_stokes(geom),
            dt=float(ac_dt),
            max_steps=int(ac_max_steps),
            backend=backend,
        )
        alpha_vals = np.clip(1.0 - np.asarray(c_vals, dtype=float), 0.0, 1.0)
        if alpha_k.nodal_values.shape != alpha_vals.shape:
            raise RuntimeError("Allen--Cahn alpha field size mismatch.")
        alpha_k.nodal_values[:] = alpha_vals
        alpha_meta = {"mode": "allen_cahn", "alpha_from_stokes_indicator": c_meta}

    alpha_k.nodal_values[:] = np.clip(np.asarray(alpha_k.nodal_values, dtype=float), 0.0, 1.0)
    inactive_domain_counts = {"porous_in_free_fluid": {}, "fluid_in_porous": {}}

    kappa_c = Constant(float(kappa))
    alpha_mod = (Constant(1.0) - kappa_c) * alpha_k + kappa_c
    F_k = Constant(1.0) - alpha_k
    F_mod = (Constant(1.0) - kappa_c) * F_k + kappa_c
    phi0_c = Constant(float(phi0))
    gamma_v_in_porous_c = Constant(float(gamma_v_in_porous))
    gamma_p_in_porous_c = Constant(float(gamma_p_in_porous))
    gamma_vp_in_free_c = Constant(float(gamma_vp_in_free))
    gamma_p_pore_in_free_c = Constant(float(gamma_p_pore_in_free))
    grad_alpha_expr = grad(alpha_k)
    abs_grad_alpha_expr = _sqrt(inner(grad_alpha_expr, grad_alpha_expr) + Constant(1.0e-16))
    tau_x = -grad_alpha_expr[1] / abs_grad_alpha_expr
    tau_y = grad_alpha_expr[0] / abs_grad_alpha_expr

    conv_factor = Constant(0.0)
    tau_lsic = Constant(float(lsic_scale) * float(Umax) * h / 2.0)
    conv_res = dot(dot(grad(u_k), u_k), w)
    conv_jac = dot(dot(grad(u_k), du), w) + dot(dot(grad(du), u_k), w)
    u_tang = u_k[0] * tau_x + u_k[1] * tau_y
    du_tang = du[0] * tau_x + du[1] * tau_y
    w_tang = w[0] * tau_x + w[1] * tau_y
    bjs_coeff = Constant(float(friction_alpha / math.sqrt(K)))
    vp_tang = vp_k[0] * tau_x + vp_k[1] * tau_y
    dvp_tang = dvp[0] * tau_x + dvp[1] * tau_y
    wp_tang = wp[0] * tau_x + wp[1] * tau_y
    rel_tang = u_tang - vp_tang
    d_rel_tang = du_tang - dvp_tang
    test_rel_tang = w_tang - wp_tang
    bjs_res = bjs_coeff * rel_tang * test_rel_tang * abs_grad_alpha_expr
    bjs_jac = bjs_coeff * d_rel_tang * test_rel_tang * abs_grad_alpha_expr
    bjs_res_rigid = bjs_coeff * u_tang * w_tang * abs_grad_alpha_expr
    bjs_jac_rigid = bjs_coeff * du_tang * w_tang * abs_grad_alpha_expr
    if formulation_key == "diagnostic":
        traction_if_x = Constant(2.0 * float(nu)) * (
            _eps(u_k)[0, 0] * grad_alpha_expr[0] + _eps(u_k)[0, 1] * grad_alpha_expr[1]
        ) + (p_pore_k - p_k) * grad_alpha_expr[0]
        traction_if_y = Constant(2.0 * float(nu)) * (
            _eps(u_k)[1, 0] * grad_alpha_expr[0] + _eps(u_k)[1, 1] * grad_alpha_expr[1]
        ) + (p_pore_k - p_k) * grad_alpha_expr[1]
        dtraction_if_x = Constant(2.0 * float(nu)) * (
            _eps(du)[0, 0] * grad_alpha_expr[0] + _eps(du)[0, 1] * grad_alpha_expr[1]
        ) + (dphi - dp) * grad_alpha_expr[0]
        dtraction_if_y = Constant(2.0 * float(nu)) * (
            _eps(du)[1, 0] * grad_alpha_expr[0] + _eps(du)[1, 1] * grad_alpha_expr[1]
        ) + (dphi - dp) * grad_alpha_expr[1]
        residual_form = (
            conv_factor * Constant(1.0) * conv_res * F_mod
            + Constant(2.0 * float(nu)) * inner(_eps(u_k), _eps(w)) * F_mod
            - (F_mod * p_k) * div(w)
            + q * (F_mod * div(u_k) + dot(grad_alpha_expr, u_k - vp_k))
            + tau_lsic * (F_mod * div(u_k)) * div(w)
            + traction_if_x * w[0]
            + traction_if_y * w[1]
            + bjs_res_rigid
            + alpha_mod * Constant(1.0 / float(K)) * dot(vp_k, wp)
            - alpha_mod * p_pore_k * div(wp)
            + alpha_mod * psi * div(vp_k)
        ) * dx()

        jacobian_form = (
            conv_factor * Constant(1.0) * conv_jac * F_mod
            + Constant(2.0 * float(nu)) * inner(_eps(du), _eps(w)) * F_mod
            - (F_mod * dp) * div(w)
            + q * (F_mod * div(du) + dot(grad_alpha_expr, du - dvp))
            + tau_lsic * (F_mod * div(du)) * div(w)
            + dtraction_if_x * w[0]
            + dtraction_if_y * w[1]
            + bjs_jac_rigid
            + alpha_mod * Constant(1.0 / float(K)) * dot(dvp, wp)
            - alpha_mod * dphi * div(wp)
            + alpha_mod * psi * div(dvp)
        ) * dx()
    else:
        # Rigid-limit reduction of the production final_form with v_s = 0, u = 0,
        # alpha prescribed, rho_s constant, and phi fixed to phi0.
        traction_if = Constant(float(g)) * phi0_c * p_pore_k * dot(w, grad_alpha_expr)
        dtraction_if = Constant(float(g)) * phi0_c * dphi * dot(w, grad_alpha_expr)
        darcy_if = phi0_c * p_pore_k * dot(wp, grad_alpha_expr)
        ddarcy_if = phi0_c * dphi * dot(wp, grad_alpha_expr)
        mass_if = -psi * dot(u_k, grad_alpha_expr)
        dmass_if = -psi * dot(du, grad_alpha_expr)
        residual_form = (
            conv_factor * Constant(1.0) * conv_res * F_mod
            + Constant(2.0 * float(nu)) * inner(_eps(u_k), _eps(w)) * F_mod
            - (F_mod * p_k) * div(w)
            + q * (F_mod * div(u_k))
            + tau_lsic * (F_mod * div(u_k)) * div(w)
            + traction_if
            + bjs_res
            + alpha_mod * phi0_c * Constant(1.0 / float(K)) * dot(vp_k, wp)
            - alpha_mod * phi0_c * p_pore_k * div(wp)
            + darcy_if
            + alpha_mod * phi0_c * psi * div(vp_k)
            + mass_if
            + alpha_mod * gamma_v_in_porous_c * dot(u_k, w)
            + alpha_mod * gamma_p_in_porous_c * p_k * q
            + F_mod * gamma_vp_in_free_c * dot(vp_k, wp)
            + F_mod * gamma_p_pore_in_free_c * p_pore_k * psi
        ) * dx()

        jacobian_form = (
            conv_factor * Constant(1.0) * conv_jac * F_mod
            + Constant(2.0 * float(nu)) * inner(_eps(du), _eps(w)) * F_mod
            - (F_mod * dp) * div(w)
            + q * (F_mod * div(du))
            + tau_lsic * (F_mod * div(du)) * div(w)
            + dtraction_if
            + bjs_jac
            + alpha_mod * phi0_c * Constant(1.0 / float(K)) * dot(dvp, wp)
            - alpha_mod * phi0_c * dphi * div(wp)
            + ddarcy_if
            + alpha_mod * phi0_c * psi * div(dvp)
            + dmass_if
            + alpha_mod * gamma_v_in_porous_c * dot(du, w)
            + alpha_mod * gamma_p_in_porous_c * dp * q
            + F_mod * gamma_vp_in_free_c * dot(dvp, wp)
            + F_mod * gamma_p_pore_in_free_c * dphi * psi
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
    functions = [u_k, vp_k, p_k, p_pore_k]
    prev_functions = [u_n, vp_n, p_n, p_pore_n]
    dh.apply_bcs(bcs_now, *functions)
    dh.apply_bcs(bcs_now, *prev_functions)
    aux_funcs = {"alpha_k": alpha_k}

    conv_history: list[dict[str, float]] = []
    conv_weights = [0.0, 0.25, 0.50, 0.75, 1.0]
    iters_total = 0
    for weight in conv_weights:
        conv_factor.value = float(weight)
        _, converged, iters = solver._newton_loop(functions, prev_functions, aux_funcs, bcs_now)
        conv_history.append(
            {
                "weight": float(weight),
                "converged": float(1.0 if bool(converged) else 0.0),
                "iterations": float(int(iters)),
                "last_norm": float(getattr(solver, "_last_nonlinear_norm", float("nan")) or float("nan")),
            }
        )
        if not bool(converged):
            raise RuntimeError(f"Rigid one-domain convection homotopy failed at weight={float(weight):.2f}.")
        u_n.nodal_values[:] = u_k.nodal_values[:]
        vp_n.nodal_values[:] = vp_k.nodal_values[:]
        p_n.nodal_values[:] = p_k.nodal_values[:]
        p_pore_n.nodal_values[:] = p_pore_k.nodal_values[:]
        iters_total += int(iters)

    if export:
        export_vtk(
            str(outdir / "final_state.vtu"),
            mesh=mesh,
            dof_handler=dh,
            functions={
                "u_free": u_k,
                "u_pore": vp_k,
                "p_free": p_k,
                "p_pore": p_pore_k,
                "alpha": alpha_k,
            },
        )

    centerline_path = _write_centerline_samples(
        outdir,
        dh,
        mesh,
        geom,
        u_k[0],
        u_k[1],
        vp_k[0],
        vp_k[1],
        p_k,
        p_pore_k,
        alpha_k,
    )
    velocity_grid_path = _write_velocity_grid_samples(
        outdir,
        dh,
        mesh,
        geom,
        u_k[0],
        u_k[1],
        vp_k[0],
        vp_k[1],
        p_k,
        p_pore_k,
        alpha_k,
    )
    plot_paths = _write_velocity_plots(outdir, velocity_grid_path, geom, umax=float(Umax))

    x_probe = float(geom.center_x)
    y_probe = 50.0
    alpha_probe = _eval_scalar_at_point(dh, mesh, "alpha", alpha_k, (x_probe, y_probe))
    ux_probe = _eval_scalar_at_point(dh, mesh, "ux", u_k[0], (x_probe, y_probe))
    uy_probe = _eval_scalar_at_point(dh, mesh, "uy", u_k[1], (x_probe, y_probe))
    darcy_probe = np.asarray(
        [
            _eval_scalar_at_point(dh, mesh, "vpx", vp_k[0], (x_probe, y_probe)),
            _eval_scalar_at_point(dh, mesh, "vpy", vp_k[1], (x_probe, y_probe)),
        ],
        dtype=float,
    )
    combined_probe = (1.0 - float(alpha_probe)) * np.asarray([ux_probe, uy_probe], dtype=float) + float(alpha_probe) * darcy_probe

    summary: dict[str, object] = {
        "benchmark": "stoter_section_5_3_one_domain_rigid_limit",
        "formulation": formulation_key,
        "phi0": float(phi0),
        "backend": str(backend),
        "nx": int(nx),
        "ny": int(ny),
        "eps": float(eps),
        "Umax_mm_per_s": float(Umax),
        "K_mm_per_s": float(K),
        "nu_mm2_per_s": float(nu),
        "friction_alpha": float(friction_alpha),
        "gravity_g": float(g),
        "kappa": float(kappa),
        "h": float(h),
        "lsic_tau": float(lsic_scale) * float(Umax) * float(h) / 2.0,
        "phase_field_mode": str(alpha_meta.get("mode", phase_field_mode)),
        "phase_field_meta": alpha_meta,
        "convection_homotopy": conv_history,
        "nonlinear_iterations": int(iters_total),
        "gamma_v_in_porous": float(gamma_v_in_porous),
        "gamma_p_in_porous": float(gamma_p_in_porous),
        "gamma_vp_in_free": float(gamma_vp_in_free),
        "gamma_p_pore_in_free": float(gamma_p_pore_in_free),
        "inactive_alpha_low": (
            None if inactive_alpha_low is None or not np.isfinite(float(inactive_alpha_low)) else float(inactive_alpha_low)
        ),
        "inactive_alpha_high": (
            None if inactive_alpha_high is None or not np.isfinite(float(inactive_alpha_high)) else float(inactive_alpha_high)
        ),
        "inactive_domain_counts": inactive_domain_counts,
        "center_probe_xy_mm": [x_probe, y_probe],
        "alpha_probe": float(alpha_probe),
        "free_velocity_probe_mm_per_s": [float(ux_probe), float(uy_probe)],
        "darcy_velocity_probe_mm_per_s": [float(darcy_probe[0]), float(darcy_probe[1])],
        "combined_velocity_probe_mm_per_s": [float(combined_probe[0]), float(combined_probe[1])],
        "centerline_csv": str(centerline_path),
        "velocity_grid_csv": str(velocity_grid_path),
        **plot_paths,
    }
    if compare_sharp_csv is not None:
        summary["sharp_comparison"] = _compare_velocity_grids(compare_sharp_csv, velocity_grid_path)
    with (outdir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=str, default="out/stoter_channel_one_domain")
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=40)
    ap.add_argument("--eps", type=float, default=2.5)
    ap.add_argument("--Umax", type=float, default=10.0)
    ap.add_argument("--K", type=float, default=5000.0)
    ap.add_argument("--nu", type=float, default=2.927)
    ap.add_argument("--friction-alpha", type=float, default=1.0)
    ap.add_argument("--g", type=float, default=9.81)
    ap.add_argument("--lsic-scale", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=1.0e-4)
    ap.add_argument("--phase-field-mode", type=str, default="allen_cahn", choices=("analytic", "allen_cahn"))
    ap.add_argument("--alpha-mode", type=str, default="diffuse", choices=("diffuse", "all_free", "all_porous"))
    ap.add_argument("--ac-dt", type=float, default=0.05)
    ap.add_argument("--ac-max-steps", type=int, default=80)
    ap.add_argument("--compare-sharp-csv", type=str, default="out/stoter_channel_sharp_32x40_refined_nophi_cpp/velocity_grid.csv")
    ap.add_argument("--formulation", type=str, default="diagnostic", choices=("diagnostic", "final_form_rigid"))
    ap.add_argument("--phi0", type=float, default=1.0)
    ap.add_argument("--gamma-v-in-porous", type=float, default=0.0)
    ap.add_argument("--gamma-p-in-porous", type=float, default=0.0)
    ap.add_argument("--gamma-vp-in-free", type=float, default=0.0)
    ap.add_argument("--gamma-p-pore-in-free", type=float, default=0.0)
    ap.add_argument("--multiplier-regularization", type=float, default=0.0)
    ap.add_argument("--inactive-alpha-low", type=float, default=0.05)
    ap.add_argument("--inactive-alpha-high", type=float, default=0.95)
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    compare_csv = None if not str(args.compare_sharp_csv).strip() else Path(args.compare_sharp_csv).resolve()
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
        alpha_mode=str(args.alpha_mode),
        ac_dt=float(args.ac_dt),
        ac_max_steps=int(args.ac_max_steps),
        compare_sharp_csv=compare_csv,
        formulation=str(args.formulation),
        phi0=float(args.phi0),
        gamma_v_in_porous=float(args.gamma_v_in_porous),
        gamma_p_in_porous=float(args.gamma_p_in_porous),
        gamma_vp_in_free=float(args.gamma_vp_in_free),
        gamma_p_pore_in_free=float(args.gamma_p_pore_in_free),
        multiplier_regularization=float(args.multiplier_regularization),
        inactive_alpha_low=float(args.inactive_alpha_low),
        inactive_alpha_high=float(args.inactive_alpha_high),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
