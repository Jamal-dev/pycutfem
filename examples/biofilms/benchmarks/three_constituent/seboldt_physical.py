"""Physical Seboldt benchmark driver for the three-constituent model.

This file deliberately does not contain a manufactured solution.  It wires the
canonical three-constituent residual into the benchmark-7 geometry and loading:
free liquid below the support, porous skeleton above it, parabolic inflow at
the bottom, drained top pore pressure, and bounded alpha/phi solved by PDAS.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from examples.utils.biofilm.three_constituent_one_domain import (
    _named_c,
    build_three_constituent_one_domain_forms,
    build_three_constituent_pdas_solver,
    one_domain_contents,
)
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import (
    LinearSolverParameters,
    NewtonParameters,
    TimeStepperParameters,
    VIParameters,
)
from pycutfem.ufl.autodiff import linearize_form
from pycutfem.ufl.expressions import (
    FacetNormal,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    dot,
)
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.helpers import analyze_active_dofs
from pycutfem.ufl.measures import dS, dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


@dataclass(frozen=True)
class PhysicalSeboldtResult:
    passed: bool
    outdir: Path
    summary: dict[str, object]
    profile_rows: list[dict[str, float]]


def _tag_rectangle_boundaries(mesh: Mesh, *, Lx: float, Ly: float, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - float(Lx)) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - float(Ly)) <= tol,
        }
    )


def _alpha_profile(x, y, *, y_interface: float, eps_alpha: float) -> float:
    del x
    eps = max(float(eps_alpha), 1.0e-12)
    return float(0.5 * (1.0 + math.tanh((float(y) - float(y_interface)) / (math.sqrt(2.0) * eps))))


def _bottom_inlet_y(x, *, Lx: float, v_in: float, ramp: float = 1.0) -> float:
    xx = float(x) / max(float(Lx), 1.0e-30)
    return float(ramp) * 4.0 * float(v_in) * xx * (1.0 - xx)


def _cosine_ramp_value(t_now: float, ramp_time: float) -> float:
    tr = float(ramp_time)
    if not np.isfinite(tr) or tr <= 0.0:
        return 1.0
    tt = max(0.0, float(t_now))
    if tt >= tr:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * tt / max(1.0e-12, tr)))


def _zero_scalar(x, y, t=0.0) -> float:
    del x, y, t
    return 0.0


def _make_homogeneous_bcs(bcs: list[BoundaryCondition]) -> list[BoundaryCondition]:
    return [BoundaryCondition(bc.field, bc.method, bc.domain_tag, _zero_scalar) for bc in bcs]


def _pore_momentum_outflow_key(value: str | None) -> str:
    key = str(value or "conservative").strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "false": "none",
        "0": "none",
        "full": "conservative",
        "weak_conservative": "conservative",
        "upwind": "outflow_only",
        "positive": "outflow_only",
        "positive_part": "outflow_only",
    }
    key = aliases.get(key, key)
    if key not in {"none", "conservative", "outflow_only"}:
        raise ValueError(
            f"Unsupported pore_momentum_outflow={value!r}. "
            "Use 'none', 'conservative', or 'outflow_only'."
        )
    return key


def _smooth_positive_named(x, *, eta: float, name: str):
    half = _named_c(f"{name}_half", 0.5)
    eta_c = _named_c(f"{name}_eta", float(eta))
    return half * (x + (x * x + eta_c) ** half)


def _flat_interface_band_marker(
    points: np.ndarray,
    *,
    y_interface: float,
    band_halfwidth: float,
    line_spacing: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    y = np.asarray(pts[:, 1], dtype=float)
    band = max(float(band_halfwidth), 0.0)
    spacing = max(float(line_spacing), 1.0e-14)
    if band <= 0.0:
        return y - float(y_interface)
    n_lines = max(int(math.ceil((2.0 * band) / spacing)), 1)
    levels = np.linspace(float(y_interface) - band, float(y_interface) + band, n_lines + 1, dtype=float)
    vals = np.ones_like(y, dtype=float)
    for level in levels.tolist():
        vals *= y - float(level)
    return vals


def _mesh_interface_stats(mesh: Mesh, *, y_interface: float, eps_alpha: float) -> dict[str, float]:
    stats = {
        "mesh_element_count": float(len(getattr(mesh, "elements_list", ()))),
        "mesh_node_count": float(len(getattr(mesh, "nodes_list", ()))),
        "interface_band_element_count": 0.0,
        "interface_band_min_dy": float("nan"),
        "interface_band_max_dy": float("nan"),
        "interface_cells_across_2eps_min": float("nan"),
        "interface_cells_across_2eps_max": float("nan"),
    }
    corners = np.asarray(getattr(mesh, "corner_connectivity", None), dtype=int)
    if corners.ndim != 2 or corners.shape[0] == 0:
        return stats
    coords = np.asarray(mesh.nodes_x_y_pos, dtype=float)[corners]
    ey0 = coords[:, :, 1].min(axis=1)
    ey1 = coords[:, :, 1].max(axis=1)
    eyc = coords[:, :, 1].mean(axis=1)
    dy_vals = ey1 - ey0
    band = max(float(eps_alpha), 1.0e-14)
    mask = np.abs(eyc - float(y_interface)) <= band
    if not np.any(mask):
        return stats
    dy_band = np.asarray(dy_vals[mask], dtype=float)
    dy_min = float(np.min(dy_band))
    dy_max = float(np.max(dy_band))
    two_eps = 2.0 * max(float(eps_alpha), 1.0e-14)
    stats.update(
        {
            "interface_band_element_count": float(np.count_nonzero(mask)),
            "interface_band_min_dy": dy_min,
            "interface_band_max_dy": dy_max,
            "interface_cells_across_2eps_min": float(two_eps / max(dy_max, 1.0e-14)),
            "interface_cells_across_2eps_max": float(two_eps / max(dy_min, 1.0e-14)),
        }
    )
    return stats


def _coords_from_breaks(breaks: np.ndarray, order: int) -> np.ndarray:
    pts: list[float] = []
    for i in range(len(breaks) - 1):
        local = np.linspace(float(breaks[i]), float(breaks[i + 1]), int(order) + 1, dtype=float)
        if i > 0:
            local = local[1:]
        pts.extend(float(v) for v in local)
    return np.asarray(pts, dtype=float)


def _structured_quad_tensor_product(
    *,
    x_breaks: np.ndarray,
    y_breaks: np.ndarray,
    poly_order: int,
) -> tuple[list[Node], np.ndarray, np.ndarray, np.ndarray]:
    order = int(poly_order)
    if order < 1:
        raise ValueError("poly_order must be positive.")
    xb = np.asarray(x_breaks, dtype=float)
    yb = np.asarray(y_breaks, dtype=float)
    nx = int(xb.size - 1)
    ny = int(yb.size - 1)
    x_coords = _coords_from_breaks(xb, order)
    y_coords = _coords_from_breaks(yb, order)
    npx = int(x_coords.size)
    nodes = [
        Node(id=int(j * npx + i), x=float(x), y=float(y))
        for j, y in enumerate(y_coords)
        for i, x in enumerate(x_coords)
    ]

    nloc = (order + 1) ** 2
    elements = np.empty((nx * ny, nloc), dtype=int)
    corners = np.empty((nx * ny, 4), dtype=int)
    edges: set[tuple[int, int]] = set()

    def node_id(ix: int, iy: int) -> int:
        return int(iy * npx + ix)

    for ey in range(ny):
        for ex in range(nx):
            eid = ey * nx + ex
            start_ix = order * ex
            start_iy = order * ey
            local_idx = 0
            for ly in range(order + 1):
                for lx in range(order + 1):
                    elements[eid, local_idx] = node_id(start_ix + lx, start_iy + ly)
                    local_idx += 1
            bl = node_id(start_ix, start_iy)
            br = node_id(start_ix + order, start_iy)
            tr = node_id(start_ix + order, start_iy + order)
            tl = node_id(start_ix, start_iy + order)
            corners[eid, :] = [bl, br, tr, tl]
            edges.add(tuple(sorted((bl, br))))
            edges.add(tuple(sorted((br, tr))))
            edges.add(tuple(sorted((tr, tl))))
            edges.add(tuple(sorted((tl, bl))))
    return nodes, elements, np.asarray(sorted(edges), dtype=int), corners


def _interface_resolved_breaks(
    *,
    length: float,
    n_base: int,
    y_interface: float,
    eps_alpha: float,
    target_cells: float,
) -> np.ndarray:
    n_fine = max(int(round(float(target_cells))), 1)
    y0 = max(0.0, float(y_interface) - float(eps_alpha))
    y1 = min(float(length), float(y_interface) + float(eps_alpha))
    if not (y1 > y0):
        return np.linspace(0.0, float(length), max(int(n_base), 1) + 1, dtype=float)
    h_base = float(length) / max(int(n_base), 1)
    n_bottom = max(int(math.ceil(y0 / max(h_base, 1.0e-14))), 1) if y0 > 0.0 else 0
    n_top = max(int(math.ceil((float(length) - y1) / max(h_base, 1.0e-14))), 1) if y1 < float(length) else 0
    parts: list[np.ndarray] = []
    if n_bottom > 0:
        parts.append(np.linspace(0.0, y0, n_bottom + 1, dtype=float))
    parts.append(np.linspace(y0, y1, n_fine + 1, dtype=float))
    if n_top > 0:
        parts.append(np.linspace(y1, float(length), n_top + 1, dtype=float))
    merged: list[float] = []
    for part in parts:
        for val in part.tolist():
            if not merged or abs(float(val) - merged[-1]) > 1.0e-12:
                merged.append(float(val))
    return np.asarray(merged, dtype=float)


def _build_mesh(
    *,
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    poly_order: int,
    y_interface: float,
    eps_alpha: float,
    adaptive_interface_target_cells: float,
    adaptive_interface_band_halfwidth_factor: float,
    adaptive_interface_max_ref: int,
) -> tuple[Mesh, dict[str, float | str]]:
    meta: dict[str, float | str] = {
        "mode": "structured",
        "adaptive_interface_target_cells": float(adaptive_interface_target_cells),
        "adaptive_interface_used_refine_level": 0.0,
    }
    if float(adaptive_interface_target_cells) > 0.0 and float(eps_alpha) > 0.0:
        target_h = (2.0 * float(eps_alpha)) / max(float(adaptive_interface_target_cells), 1.0e-14)
        x_breaks = np.linspace(0.0, float(Lx), max(int(nx), 1) + 1, dtype=float)
        y_breaks = _interface_resolved_breaks(
            length=float(Ly),
            n_base=int(ny),
            y_interface=float(y_interface),
            eps_alpha=float(eps_alpha),
            target_cells=float(adaptive_interface_target_cells),
        )
        nodes, elems, _, corners = _structured_quad_tensor_product(
            x_breaks=x_breaks,
            y_breaks=y_breaks,
            poly_order=int(poly_order),
        )
        mesh = Mesh(
            nodes=nodes,
            element_connectivity=elems,
            elements_corner_nodes=corners,
            element_type="quad",
            poly_order=int(poly_order),
        )
        meta.update(
            {
                "mode": "interface_resolved_tensor",
                "adaptive_interface_target_h": float(target_h),
                "adaptive_interface_used_refine_level": 0.0,
                "interface_resolved_y_elements": float(len(y_breaks) - 1),
            }
        )
    else:
        nodes, elems, _, corners = structured_quad(
            float(Lx),
            float(Ly),
            nx=int(nx),
            ny=int(ny),
            poly_order=int(poly_order),
        )
        mesh = Mesh(
            nodes=nodes,
            element_connectivity=elems,
            elements_corner_nodes=corners,
            element_type="quad",
            poly_order=int(poly_order),
        )
    _tag_rectangle_boundaries(mesh, Lx=float(Lx), Ly=float(Ly))
    meta.update(_mesh_interface_stats(mesh, y_interface=y_interface, eps_alpha=eps_alpha))
    return mesh, meta


def _make_spaces(dh: DofHandler):
    return {
        "VF": FunctionSpace("VF", ["vf_x", "vf_y"], dim=1),
        "VP": FunctionSpace("VP", ["vp_x", "vp_y"], dim=1),
        "VS": FunctionSpace("VS", ["vs_x", "vs_y"], dim=1),
        "US": FunctionSpace("US", ["us_x", "us_y"], dim=1),
    }


def _make_trial_test(dh: DofHandler, spaces: dict[str, object]):
    trial = {
        "dv_f": VectorTrialFunction(space=spaces["VF"], dof_handler=dh),
        "dp_f": TrialFunction("pf", dof_handler=dh),
        "dv_p": VectorTrialFunction(space=spaces["VP"], dof_handler=dh),
        "dp_p": TrialFunction("pp", dof_handler=dh),
        "dv_s": VectorTrialFunction(space=spaces["VS"], dof_handler=dh),
        "du_s": VectorTrialFunction(space=spaces["US"], dof_handler=dh),
        "dalpha": TrialFunction("alpha", dof_handler=dh),
        "dphi": TrialFunction("phi", dof_handler=dh),
        "dGamma": TrialFunction("Gamma", dof_handler=dh),
    }
    test = {
        "w_f": VectorTestFunction(space=spaces["VF"], dof_handler=dh),
        "q_f": TestFunction("pf", dof_handler=dh),
        "w_p": VectorTestFunction(space=spaces["VP"], dof_handler=dh),
        "q_p": TestFunction("pp", dof_handler=dh),
        "w_s": VectorTestFunction(space=spaces["VS"], dof_handler=dh),
        "z_u": VectorTestFunction(space=spaces["US"], dof_handler=dh),
        "z_alpha": TestFunction("alpha", dof_handler=dh),
        "q_s": TestFunction("phi", dof_handler=dh),
        "z_Gamma": TestFunction("Gamma", dof_handler=dh),
    }
    return trial, test


def _make_state(dh: DofHandler) -> dict[str, object]:
    return {
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


def _set_scalar_values_from_function(dh: DofHandler, function: Function, fn) -> None:
    field = function.field_name
    gdofs = np.asarray(dh.get_field_slice(field), dtype=int)
    coords = np.asarray(dh.get_dof_coords(field), dtype=float)
    vals = np.asarray([fn(float(x), float(y)) for x, y in coords], dtype=float)
    function.set_nodal_values(gdofs, vals)


def _initialize_state(
    dh: DofHandler,
    state: dict[str, object],
    *,
    Lx: float,
    y_interface: float,
    eps_alpha: float,
    phi_b: float,
    v_in: float,
    ramp: float,
) -> None:
    alpha0 = lambda x, y: _alpha_profile(x, y, y_interface=y_interface, eps_alpha=eps_alpha)
    phi0 = lambda x, y: float(phi_b)
    zero_v = lambda x, y: np.asarray([0.0, 0.0], dtype=float)
    inlet_guess = lambda x, y: np.asarray([0.0, _bottom_inlet_y(x, Lx=Lx, v_in=v_in, ramp=ramp)], dtype=float)

    for key in ("v_f_k", "v_f_n"):
        state[key].set_values_from_function(inlet_guess if key == "v_f_k" else zero_v)
    for key in ("v_p_k", "v_p_n", "v_s_k", "v_s_n", "u_s_k", "u_s_n"):
        state[key].set_values_from_function(zero_v)
    for key in ("p_f_k", "p_p_k", "Gamma_k", "Gamma_n"):
        state[key].nodal_values[:] = 0.0
    for key in ("alpha_k", "alpha_n"):
        _set_scalar_values_from_function(dh, state[key], alpha0)
    for key in ("phi_k", "phi_n"):
        _set_scalar_values_from_function(dh, state[key], phi0)


def _tag_inactive_three_constituent_domains(
    dh: DofHandler,
    mesh: Mesh,
    alpha_h: Function,
    *,
    alpha_low: float | None,
    alpha_high: float | None,
    previous_tagged: set[int] | None = None,
) -> tuple[dict[str, dict[str, int]], set[int]]:
    counts = {"free_side": {}, "porous_side": {}}
    inactive = set(int(d) for d in list(getattr(dh, "dof_tags", {}).get("inactive", set()) or set()))
    if previous_tagged:
        inactive.difference_update(set(int(d) for d in previous_tagged))
        dh.dof_tags["inactive"] = inactive
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

    tagged_now: set[int] = set()

    def _tag(mask: np.ndarray, fields: tuple[str, ...], bucket: str) -> None:
        for field in fields:
            if field not in getattr(dh, "field_names", ()):
                counts[bucket][field] = 0
                continue
            selected = dh.tag_dofs_from_element_bitset("inactive", field, mask, strict=True)
            selected_set = set(int(g) for g in selected)
            inactive.update(selected_set)
            tagged_now.update(selected_set)
            counts[bucket][field] = int(len(selected_set))

    if use_low:
        _tag(low_mask, ("vp_x", "vp_y", "pp", "vs_x", "vs_y", "us_x", "us_y", "phi"), "free_side")
    if use_high:
        _tag(high_mask, ("vf_x", "vf_y", "pf"), "porous_side")
    dh.dof_tags["inactive"] = inactive
    counts["free_side"]["elements"] = int(np.count_nonzero(low_mask))
    counts["porous_side"]["elements"] = int(np.count_nonzero(high_mask))
    return counts, tagged_now


def _candidate_active_dofs_from_current_tags(solver, bcs_for_active: list[BoundaryCondition]) -> np.ndarray:
    active_by_restr, has_restriction = analyze_active_dofs(
        solver.equation,
        solver.dh,
        solver.me,
        bcs_for_active,
        verbose=False,
    )
    if has_restriction:
        return np.asarray(sorted(set(int(d) for d in active_by_restr)), dtype=int)
    return np.arange(int(solver.dh.total_dofs), dtype=int)


def _function_objects_by_field(functions: list[object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for obj in functions:
        for field in list(getattr(obj, "field_names", ()) or ()):
            out.setdefault(str(field), obj)
        field_name = getattr(obj, "field_name", None)
        if field_name is not None:
            out.setdefault(str(field_name), obj)
    return out


def _seed_reactivated_inactive_dofs(
    dh: DofHandler,
    reactivated_dofs: set[int],
    *,
    functions: list[object],
    prev_functions: list[object],
    inactive_dofs: set[int],
    fallback_by_field: dict[str, float] | None = None,
    projection: str = "l2_patch",
    quad_order: int = 4,
    patch_layers: int = 2,
) -> dict[str, dict[str, int]]:
    """Seed DOFs that just left the inactive closure from accepted active values.

    The production path uses a constrained patch L2 projection.  Accepted
    active DOFs are held fixed as a donor field; only the reactivated DOFs are
    solved for on the local element patch.  This gives newly active DOFs a best
    approximation from the accepted state instead of a raw copy, clip, or stale
    inactive value.
    """
    targets = set(int(d) for d in reactivated_dofs) - set(int(d) for d in inactive_dofs)
    if not targets:
        return {}
    current_by_field = _function_objects_by_field(functions)
    prev_by_field = _function_objects_by_field(prev_functions)
    fallback_by_field = dict(fallback_by_field or {})
    stats: dict[str, dict[str, int]] = {}

    for field in list(getattr(dh, "field_names", ()) or ()):
        field_gds = np.asarray(dh.get_field_slice(field), dtype=int)
        field_targets = sorted(targets.intersection(set(int(g) for g in field_gds)))
        if not field_targets:
            continue
        current_obj = current_by_field.get(str(field))
        prev_obj = prev_by_field.get(str(field))
        if current_obj is None or prev_obj is None:
            continue

        field_set = set(int(g) for g in field_gds)
        donor_set = field_set - set(int(g) for g in field_targets) - set(int(g) for g in inactive_dofs)
        elem_maps = list(getattr(dh, "element_maps", {}).get(field, []) or [])
        d2e: dict[int, set[int]] = {}
        for eid, gds in enumerate(elem_maps):
            for gd in np.asarray(gds, dtype=int).ravel().tolist():
                d2e.setdefault(int(gd), set()).add(int(eid))

        coords_by_gd: dict[int, np.ndarray] = {}
        donor_coords = np.empty((0, 2), dtype=float)
        donor_gds = np.asarray(sorted(donor_set), dtype=int)
        try:
            coords = np.asarray(dh.get_dof_coords(field), dtype=float)
            coords_by_gd = {int(gd): np.asarray(coords[i], dtype=float) for i, gd in enumerate(field_gds)}
            if donor_gds.size:
                donor_coords = np.asarray([coords_by_gd[int(gd)] for gd in donor_gds], dtype=float)
        except Exception:
            coords_by_gd = {}
            donor_coords = np.empty((0, 2), dtype=float)

        def _fallback_value(gd: int, counts: dict[str, int]) -> float:
            patch_donors: set[int] = set()
            for eid in d2e.get(int(gd), set()):
                if 0 <= int(eid) < len(elem_maps):
                    patch_donors.update(int(d) for d in np.asarray(elem_maps[int(eid)], dtype=int).ravel().tolist())
            patch_donors = patch_donors.intersection(donor_set)
            val = float("nan")
            if patch_donors:
                donor_vals = np.asarray(prev_obj.get_nodal_values(np.asarray(sorted(patch_donors), dtype=int)), dtype=float)
                donor_vals = donor_vals[np.isfinite(donor_vals)]
                if donor_vals.size:
                    val = float(np.mean(donor_vals))
                    counts["patch"] += 1
            if not np.isfinite(val) and donor_gds.size and donor_coords.size and int(gd) in coords_by_gd:
                delta = donor_coords - np.asarray(coords_by_gd[int(gd)], dtype=float)[None, :]
                nearest = int(donor_gds[int(np.argmin(np.einsum("ij,ij->i", delta, delta)))])
                nearest_vals = np.asarray(prev_obj.get_nodal_values(np.asarray([nearest], dtype=int)), dtype=float)
                if nearest_vals.size and np.isfinite(float(nearest_vals[0])):
                    val = float(nearest_vals[0])
                    counts["nearest"] += 1
            if not np.isfinite(val):
                val = float(fallback_by_field.get(str(field), 0.0))
                counts["fallback"] += 1
            return val

        def _expanded_patch(seed_eids: set[int]) -> set[int]:
            patch = set(int(eid) for eid in seed_eids if 0 <= int(eid) < len(elem_maps))
            frontier = set(patch)
            for _ in range(max(int(patch_layers), 0)):
                if not frontier:
                    break
                frontier_dofs: set[int] = set()
                for eid in frontier:
                    frontier_dofs.update(int(d) for d in np.asarray(elem_maps[int(eid)], dtype=int).ravel().tolist())
                next_frontier: set[int] = set()
                for gd in frontier_dofs:
                    next_frontier.update(int(eid) for eid in d2e.get(int(gd), set()))
                next_frontier = {int(eid) for eid in next_frontier if 0 <= int(eid) < len(elem_maps)} - patch
                patch.update(next_frontier)
                frontier = next_frontier
            return patch

        counts = {"reactivated": len(field_targets), "l2": 0, "patch": 0, "nearest": 0, "fallback": 0}
        values = np.full((len(field_targets),), np.nan, dtype=float)
        use_l2 = str(projection or "l2_patch").strip().lower().replace("-", "_") in {
            "l2",
            "l2_patch",
            "patch_l2",
            "projection",
        }

        if use_l2 and donor_set:
            def _poly_extension():
                if donor_gds.size == 0 or donor_coords.size == 0:
                    return None
                donor_vals_all = np.asarray(prev_obj.get_nodal_values(donor_gds), dtype=float).ravel()
                valid = np.isfinite(donor_vals_all) & np.all(np.isfinite(donor_coords), axis=1)
                if not np.any(valid):
                    return None
                xy = np.asarray(donor_coords[valid], dtype=float)
                vals = np.asarray(donor_vals_all[valid], dtype=float)
                order = int(getattr(dh.mixed_element, "_field_orders", {}).get(field, 1))
                x0 = np.mean(xy, axis=0)
                scale = np.maximum(np.ptp(xy, axis=0), 1.0e-12)
                for degree in range(max(order, 0), -1, -1):
                    terms = [(i, j) for j in range(degree + 1) for i in range(degree + 1)]
                    if len(terms) > xy.shape[0]:
                        continue
                    xhat = (xy - x0[None, :]) / scale[None, :]
                    A = np.column_stack([(xhat[:, 0] ** i) * (xhat[:, 1] ** j) for i, j in terms])
                    try:
                        coeff, *_ = np.linalg.lstsq(A, vals, rcond=None)
                    except Exception:
                        continue

                    def _eval(points, *, _terms=terms, _coeff=np.asarray(coeff, dtype=float), _x0=x0, _scale=scale):
                        pts = np.asarray(points, dtype=float)
                        phat = (pts - _x0[None, :]) / _scale[None, :]
                        B = np.column_stack([(phat[:, 0] ** i) * (phat[:, 1] ** j) for i, j in _terms])
                        return B @ _coeff

                    return _eval
                return None

            extension_eval = _poly_extension()
            target_index = {int(gd): i for i, gd in enumerate(field_targets)}
            seed_eids: set[int] = set()
            for gd in field_targets:
                seed_eids.update(int(eid) for eid in d2e.get(int(gd), set()))
            patch_eids = _expanded_patch(seed_eids)
            patch_has_donor = any(
                bool(set(int(d) for d in np.asarray(elem_maps[int(eid)], dtype=int).ravel().tolist()).intersection(donor_set))
                for eid in patch_eids
            )
            if patch_eids and patch_has_donor:
                try:
                    geo = dh.precompute_geometric_factors(int(quad_order), level_set=None, reuse=True)
                    qw_all = np.asarray(geo["qw"], dtype=float)
                    qp_phys_all = np.asarray(geo["qp_phys"], dtype=float)
                    qp_ref = np.asarray(geo["qp_ref"], dtype=float)
                    basis_table = np.asarray(
                        dh.mixed_element._eval_scalar_basis_many(field, qp_ref[:, 0], qp_ref[:, 1]),
                        dtype=float,
                    )
                    rows: list[int] = []
                    cols: list[int] = []
                    data: list[float] = []
                    rhs = np.zeros((len(field_targets),), dtype=float)
                    has_donor_support = np.zeros((len(field_targets),), dtype=bool)
                    for eid in sorted(patch_eids):
                        if not (0 <= int(eid) < len(elem_maps)):
                            continue
                        g_local = np.asarray(elem_maps[int(eid)], dtype=int).ravel()
                        if g_local.size == 0:
                            continue
                        target_lids = [i for i, gd in enumerate(g_local.tolist()) if int(gd) in target_index]
                        if not target_lids:
                            continue
                        donor_lids = [i for i, gd in enumerate(g_local.tolist()) if int(gd) in donor_set]
                        target_ids = [target_index[int(g_local[i])] for i in target_lids]
                        w = np.asarray(qw_all[int(eid), :], dtype=float).ravel()
                        if w.size != basis_table.shape[0]:
                            continue
                        N_t = basis_table[:, target_lids]
                        M_loc = (N_t * w[:, None]).T @ N_t
                        for a, row in enumerate(target_ids):
                            for b, col in enumerate(target_ids):
                                val = float(M_loc[a, b])
                                if val != 0.0 and np.isfinite(val):
                                    rows.append(int(row))
                                    cols.append(int(col))
                                    data.append(val)
                        if donor_lids:
                            donor_gds_local = g_local[donor_lids]
                            donor_vals = np.asarray(prev_obj.get_nodal_values(donor_gds_local), dtype=float).ravel()
                            if donor_vals.size and np.all(np.isfinite(donor_vals)):
                                N_d = basis_table[:, donor_lids]
                                fixed_q = N_d @ donor_vals
                            else:
                                fixed_q = np.zeros((basis_table.shape[0],), dtype=float)
                        else:
                            fixed_q = np.zeros((basis_table.shape[0],), dtype=float)
                        if extension_eval is not None:
                            source_q = np.asarray(extension_eval(qp_phys_all[int(eid), :, :]), dtype=float).ravel()
                            if source_q.shape == fixed_q.shape and np.all(np.isfinite(source_q)):
                                rhs_loc = (N_t * w[:, None]).T @ (source_q - fixed_q)
                                for a, row in enumerate(target_ids):
                                    rhs[int(row)] += float(rhs_loc[a])
                                    has_donor_support[int(row)] = True
                    if rows and np.any(has_donor_support):
                        try:
                            import scipy.sparse as sp
                            import scipy.sparse.linalg as spla

                            M = sp.coo_matrix((data, (rows, cols)), shape=(len(field_targets), len(field_targets))).tocsr()
                            diag_max = float(np.max(np.abs(M.diagonal()))) if M.shape[0] else 0.0
                            reg = max(1.0e-14 * max(diag_max, 1.0), 1.0e-30)
                            M_reg = M + reg * sp.eye(M.shape[0], format="csr")
                            sol = np.asarray(spla.spsolve(M_reg, rhs), dtype=float)
                            bounds = (0.0, 1.0) if str(field) in {"alpha", "phi"} else (None, None)
                            if bounds != (None, None) and sol.shape == rhs.shape and np.all(np.isfinite(sol)):
                                lo, hi = bounds
                                x_box = sol.copy()
                                fixed = np.zeros_like(x_box, dtype=bool)
                                for _ in range(8):
                                    new_fixed = np.zeros_like(fixed)
                                    if lo is not None:
                                        new_fixed |= x_box < float(lo)
                                    if hi is not None:
                                        new_fixed |= x_box > float(hi)
                                    if np.array_equal(new_fixed, fixed):
                                        break
                                    fixed = new_fixed
                                    if lo is not None:
                                        x_box[x_box < float(lo)] = float(lo)
                                    if hi is not None:
                                        x_box[x_box > float(hi)] = float(hi)
                                    free = ~fixed
                                    if not np.any(free):
                                        break
                                    rhs_free = rhs[free] - M_reg[free][:, fixed] @ x_box[fixed]
                                    x_box[free] = np.asarray(spla.spsolve(M_reg[free][:, free], rhs_free), dtype=float)
                                sol = x_box
                        except Exception:
                            M = np.zeros((len(field_targets), len(field_targets)), dtype=float)
                            for row, col, val in zip(rows, cols, data):
                                M[int(row), int(col)] += float(val)
                            diag_max = float(np.max(np.abs(np.diag(M)))) if M.size else 0.0
                            reg = max(1.0e-14 * max(diag_max, 1.0), 1.0e-30)
                            M_reg = M + reg * np.eye(M.shape[0])
                            sol = np.linalg.solve(M_reg, rhs)
                            if str(field) in {"alpha", "phi"} and sol.shape == rhs.shape and np.all(np.isfinite(sol)):
                                lo, hi = 0.0, 1.0
                                x_box = sol.copy()
                                fixed = np.zeros_like(x_box, dtype=bool)
                                for _ in range(8):
                                    new_fixed = (x_box < lo) | (x_box > hi)
                                    if np.array_equal(new_fixed, fixed):
                                        break
                                    fixed = new_fixed
                                    x_box[x_box < lo] = lo
                                    x_box[x_box > hi] = hi
                                    free = ~fixed
                                    if not np.any(free):
                                        break
                                    rhs_free = rhs[free] - M_reg[np.ix_(free, fixed)] @ x_box[fixed]
                                    x_box[free] = np.linalg.solve(M_reg[np.ix_(free, free)], rhs_free)
                                sol = x_box
                        if sol.shape == values.shape and np.all(np.isfinite(sol)):
                            values[has_donor_support] = sol[has_donor_support]
                            counts["l2"] = int(np.count_nonzero(has_donor_support))
                except Exception:
                    values[:] = np.nan
                    counts["l2"] = 0

        for i, gd in enumerate(field_targets):
            if not np.isfinite(values[i]):
                values[i] = _fallback_value(int(gd), counts)

        g_arr = np.asarray(field_targets, dtype=int)
        v_arr = np.asarray(values, dtype=float)
        current_obj.set_nodal_values(g_arr, v_arr)
        prev_obj.set_nodal_values(g_arr, v_arr)
        stats[str(field)] = counts
    return stats


def _eval_scalar_at_point(dh: DofHandler, mesh: Mesh, f_scalar: Function, point: tuple[float, float]) -> float:
    from pycutfem.fem import transform

    xy = np.asarray(point, dtype=float)
    me = dh.mixed_element
    field = f_scalar.field_name
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
        basis = me.basis(field, float(xi), float(eta))[me.slice(field)]
        gdofs = dh.element_maps[field][elem.id]
        vals = f_scalar.get_nodal_values(gdofs)
        return float(np.asarray(basis, dtype=float) @ np.asarray(vals, dtype=float))
    return float("nan")


def _sample_profile(
    *,
    dh: DofHandler,
    mesh: Mesh,
    state: dict[str, object],
    Lx: float,
    y_profile: float,
    n_samples: int,
    rho_f: float = 1.0,
    rho_p: float = 1.0,
    rho_s: float = 1.0,
) -> list[dict[str, float]]:
    xs = np.linspace(0.0, float(Lx), int(n_samples), dtype=float)
    us_x = state["u_s_k"].components[0]
    us_y = state["u_s_k"].components[1]
    vs_x = state["v_s_k"].components[0]
    vs_y = state["v_s_k"].components[1]
    vf_x = state["v_f_k"].components[0]
    vf_y = state["v_f_k"].components[1]
    vp_x = state["v_p_k"].components[0]
    vp_y = state["v_p_k"].components[1]
    rows: list[dict[str, float]] = []
    for x_val in xs:
        pt = (float(x_val), float(y_profile))
        alpha = _eval_scalar_at_point(dh, mesh, state["alpha_k"], pt)
        phi = _eval_scalar_at_point(dh, mesh, state["phi_k"], pt)
        F = 1.0 - alpha
        P = alpha * phi
        B = alpha * (1.0 - phi)
        rho_bar = F * float(rho_f) + P * float(rho_p) + B * float(rho_s)
        vf_x_val = _eval_scalar_at_point(dh, mesh, vf_x, pt)
        vf_y_val = _eval_scalar_at_point(dh, mesh, vf_y, pt)
        vp_x_val = _eval_scalar_at_point(dh, mesh, vp_x, pt)
        vp_y_val = _eval_scalar_at_point(dh, mesh, vp_y, pt)
        vs_x_val = _eval_scalar_at_point(dh, mesh, vs_x, pt)
        vs_y_val = _eval_scalar_at_point(dh, mesh, vs_y, pt)
        if np.isfinite(rho_bar) and abs(rho_bar) > 1.0e-30:
            v_bar_x = (F * float(rho_f) * vf_x_val + P * float(rho_p) * vp_x_val + B * float(rho_s) * vs_x_val) / rho_bar
            v_bar_y = (F * float(rho_f) * vf_y_val + P * float(rho_p) * vp_y_val + B * float(rho_s) * vs_y_val) / rho_bar
        else:
            v_bar_x = float("nan")
            v_bar_y = float("nan")
        rows.append(
            {
                "x": float(x_val),
                "y": float(y_profile),
                "u_x": _eval_scalar_at_point(dh, mesh, us_x, pt),
                "u_y": _eval_scalar_at_point(dh, mesh, us_y, pt),
                "v_s_x": vs_x_val,
                "v_s_y": vs_y_val,
                "v_f_x": vf_x_val,
                "v_f_y": vf_y_val,
                "v_p_x": vp_x_val,
                "v_p_y": vp_y_val,
                "rho_bar": rho_bar,
                "v_bar_x": v_bar_x,
                "v_bar_y": v_bar_y,
                "p_f": _eval_scalar_at_point(dh, mesh, state["p_f_k"], pt),
                "p_p": _eval_scalar_at_point(dh, mesh, state["p_p_k"], pt),
                "alpha": alpha,
                "phi": phi,
                "Gamma": _eval_scalar_at_point(dh, mesh, state["Gamma_k"], pt),
            }
        )
    return rows


def _field_stats(state: dict[str, object]) -> dict[str, float]:
    stats: dict[str, float] = {}
    fields = {
        "vf": state["v_f_k"],
        "vp": state["v_p_k"],
        "vs": state["v_s_k"],
        "us": state["u_s_k"],
        "pf": state["p_f_k"],
        "pp": state["p_p_k"],
        "alpha": state["alpha_k"],
        "phi": state["phi_k"],
        "Gamma": state["Gamma_k"],
    }
    for name, fun in fields.items():
        values = np.asarray(fun.nodal_values, dtype=float).reshape(-1)
        if values.size == 0:
            continue
        stats[f"{name}_min"] = float(np.min(values))
        stats[f"{name}_max"] = float(np.max(values))
        stats[f"{name}_linf"] = float(np.linalg.norm(values, ord=np.inf))
    stats["alpha_lower_violation"] = max(0.0, -stats.get("alpha_min", 0.0))
    stats["alpha_upper_violation"] = max(0.0, stats.get("alpha_max", 0.0) - 1.0)
    stats["phi_lower_violation"] = max(0.0, -stats.get("phi_min", 0.0))
    stats["phi_upper_violation"] = max(0.0, stats.get("phi_max", 0.0) - 1.0)
    return stats


def _element_id_containing_point(mesh: Mesh, xy: np.ndarray) -> int:
    point = np.asarray(xy, dtype=float).ravel()
    if point.size < 2:
        return -1
    corners = np.asarray(getattr(mesh, "corner_connectivity", []), dtype=int)
    coords = np.asarray(getattr(mesh, "nodes_x_y_pos", []), dtype=float)
    if corners.size == 0 or coords.size == 0:
        return -1
    for eid, conn in enumerate(corners):
        verts = coords[np.asarray(conn, dtype=int)]
        if (
            float(np.min(verts[:, 0])) - 1.0e-12 <= float(point[0]) <= float(np.max(verts[:, 0])) + 1.0e-12
            and float(np.min(verts[:, 1])) - 1.0e-12 <= float(point[1]) <= float(np.max(verts[:, 1])) + 1.0e-12
        ):
            return int(eid)
    return -1


def _scalar_extremum_record(
    *,
    dh: DofHandler,
    mesh: Mesh,
    field: str,
    fun: Function,
    prefix: str,
) -> dict[str, float]:
    values = np.asarray(fun.nodal_values, dtype=float).ravel()
    if values.size == 0:
        return {
            f"{prefix}_abs_max": float("nan"),
            f"{prefix}_signed_at_abs_max": float("nan"),
            f"{prefix}_x": float("nan"),
            f"{prefix}_y": float("nan"),
            f"{prefix}_element": -1.0,
        }
    idx = int(np.argmax(np.abs(values)))
    coords = np.asarray(dh.get_dof_coords(field), dtype=float)
    xy = coords[idx] if idx < coords.shape[0] else np.asarray([float("nan"), float("nan")], dtype=float)
    return {
        f"{prefix}_abs_max": float(abs(values[idx])),
        f"{prefix}_signed_at_abs_max": float(values[idx]),
        f"{prefix}_x": float(xy[0]),
        f"{prefix}_y": float(xy[1]),
        f"{prefix}_element": float(_element_id_containing_point(mesh, xy)),
    }


def _vector_extremum_record(
    *,
    dh: DofHandler,
    mesh: Mesh,
    field_x: str,
    field_y: str,
    vec: VectorFunction,
    prefix: str,
) -> dict[str, float]:
    vx = np.asarray(vec.components[0].nodal_values, dtype=float).ravel()
    vy = np.asarray(vec.components[1].nodal_values, dtype=float).ravel()
    n = min(vx.size, vy.size)
    if n == 0:
        return {
            f"{prefix}_mag_max": float("nan"),
            f"{prefix}_x_comp_at_max": float("nan"),
            f"{prefix}_y_comp_at_max": float("nan"),
            f"{prefix}_x": float("nan"),
            f"{prefix}_y": float("nan"),
            f"{prefix}_element": -1.0,
        }
    mag = np.sqrt(vx[:n] ** 2 + vy[:n] ** 2)
    idx = int(np.argmax(mag))
    coords_x = np.asarray(dh.get_dof_coords(field_x), dtype=float)
    coords_y = np.asarray(dh.get_dof_coords(field_y), dtype=float)
    xy = coords_x[idx] if idx < coords_x.shape[0] else coords_y[idx] if idx < coords_y.shape[0] else np.asarray([float("nan"), float("nan")], dtype=float)
    return {
        f"{prefix}_mag_max": float(mag[idx]),
        f"{prefix}_x_comp_at_max": float(vx[idx]),
        f"{prefix}_y_comp_at_max": float(vy[idx]),
        f"{prefix}_x": float(xy[0]),
        f"{prefix}_y": float(xy[1]),
        f"{prefix}_element": float(_element_id_containing_point(mesh, xy)),
    }


def _write_profile(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(path: Path, summary: dict[str, object]) -> None:
    flat = {
        key: value
        for key, value in summary.items()
        if isinstance(value, (int, float, str, bool)) or value is None
    }
    metrics = summary.get("metrics")
    if isinstance(metrics, dict):
        flat.update({str(key): value for key, value in metrics.items() if isinstance(value, (int, float, str, bool))})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)


def _write_plot(path: Path, rows: list[dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    x = np.asarray([r["x"] for r in rows], dtype=float)
    uy = np.asarray([r["u_y"] for r in rows], dtype=float)
    alpha = np.asarray([r["alpha"] for r in rows], dtype=float)
    fig, ax_u = plt.subplots(figsize=(7.0, 4.0))
    ax_u.plot(x, uy, color="tab:blue", label="$u_y$")
    ax_u.set_xlabel("x")
    ax_u.set_ylabel("$u_y$")
    ax_u.grid(True, alpha=0.25)
    ax_a = ax_u.twinx()
    ax_a.plot(x, alpha, color="tab:orange", linestyle="--", label="$\\alpha$")
    ax_a.set_ylabel("$\\alpha$")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_physical_seboldt_three_constituent(
    *,
    outdir: Path,
    Lx: float = 1.0,
    Ly: float = 1.5,
    y_interface: float = 1.0,
    nx: int = 20,
    ny: int = 30,
    poly_order: int = 2,
    pressure_order: int = 1,
    scalar_order: int = 1,
    eps_alpha: float = 0.05,
    phi_b: float = 0.18,
    v_in: float = 5.0,
    t_ramp: float = 0.0,
    dt: float = 1.0e-3,
    final_time: float = 3.0,
    max_steps: int | None = None,
    rho_f: float = 1.0,
    rho_p: float = 1.0,
    rho_s: float = 1.0,
    mu_f: float = 0.035,
    mu_p: float = 0.035,
    mu_s: float = 1.67785e5,
    lambda_s: float = 8.22148e6,
    kappa: float = 1.0e-3,
    R_fp_factor: float = 1.0,
    R_fs_factor: float = 1.0,
    R_ps_factor: float = 1.0,
    ell_gamma_factor: float = 1.0,
    gamma_mobility: str = "interface_delta",
    gamma_delta_epsilon: float = 1.0e-12,
    resistance_model: str = "diagonal",
    theta_fp: float = 0.5,
    transfer_velocity: str = "free",
    lag_alpha_in_constitutive_laws: bool = True,
    inactive_velocity_extension_factor: float = 0.0,
    inactive_pressure_extension_factor: float = 0.0,
    inactive_phi_extension_factor: float = 0.0,
    inactive_displacement_extension_factor: float = 0.0,
    inactive_domain_closure: bool = False,
    inactive_alpha_low: float = 0.02,
    inactive_alpha_high: float = 0.98,
    adaptive_interface_target_cells: float = 0.0,
    adaptive_interface_band_halfwidth_factor: float = 1.0,
    adaptive_interface_max_ref: int = 4,
    backend: str = "cpp",
    linear_backend: str = "scipy",
    quad_order: int = 6,
    newton_tol: float = 1.0e-6,
    newton_rtol: float = 0.0,
    max_newton_iter: int = 12,
    pdas_c: float = 1.0,
    pore_pressure_lower_bound: float | None = 0.0,
    pore_pressure_upper_bound: float | None = None,
    pore_momentum_outflow: str = "conservative",
    pore_momentum_outflow_factor: float = 1.0,
    pore_momentum_outflow_smooth_eta: float = 1.0e-12,
    allow_dt_reduction: bool = True,
    dt_min: float = 1.0e-6,
    dt_reduction_factor: float = 0.5,
    y_profile: float = 1.25,
    profile_samples: int = 201,
    history_stride: int = 10,
    vtk_every: int = 0,
) -> PhysicalSeboldtResult:
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    mesh, mesh_meta = _build_mesh(
        Lx=Lx,
        Ly=Ly,
        nx=nx,
        ny=ny,
        poly_order=poly_order,
        y_interface=y_interface,
        eps_alpha=eps_alpha,
        adaptive_interface_target_cells=adaptive_interface_target_cells,
        adaptive_interface_band_halfwidth_factor=adaptive_interface_band_halfwidth_factor,
        adaptive_interface_max_ref=adaptive_interface_max_ref,
    )
    me = MixedElement(
        mesh,
        field_specs={
            "vf_x": int(poly_order),
            "vf_y": int(poly_order),
            "pf": int(pressure_order),
            "vp_x": int(poly_order),
            "vp_y": int(poly_order),
            "pp": int(pressure_order),
            "vs_x": int(poly_order),
            "vs_y": int(poly_order),
            "us_x": int(poly_order),
            "us_y": int(poly_order),
            "alpha": int(scalar_order),
            "phi": int(scalar_order),
            "Gamma": int(scalar_order),
        },
    )
    dh = DofHandler(me, method="cg")
    spaces = _make_spaces(dh)
    trial, test = _make_trial_test(dh, spaces)
    state = _make_state(dh)

    ramp0 = _cosine_ramp_value(float(dt), float(t_ramp))
    _initialize_state(
        dh,
        state,
        Lx=Lx,
        y_interface=y_interface,
        eps_alpha=eps_alpha,
        phi_b=phi_b,
        v_in=v_in,
        ramp=ramp0,
    )

    dt_c = _named_c("tc_b7_dt", float(dt))
    kappa_eff = max(float(kappa), 1.0e-30)
    R_ps = float(R_ps_factor) * float(mu_f) / kappa_eff
    R_fp = float(R_fp_factor) * R_ps
    R_fs = float(R_fs_factor) * R_ps
    ell_gamma = float(ell_gamma_factor) * R_ps
    R_pair_cholesky = None
    if str(resistance_model).strip().lower() in {"full", "full_cholesky", "cholesky", "spd"}:
        R_pair_cholesky = (
            (math.sqrt(max(R_fp, 0.0)), 0.0, 0.0),
            (0.0, math.sqrt(max(R_fs, 0.0)), 0.0),
            (0.0, 0.0, math.sqrt(max(R_ps, 0.0))),
        )

    forms = build_three_constituent_one_domain_forms(
        **state,
        **trial,
        **test,
        dx=dx(metadata={"q": int(quad_order)}),
        dt=dt_c,
        rho_f=rho_f,
        rho_p=rho_p,
        rho_s=rho_s,
        mu_f=mu_f,
        mu_p=mu_p,
        mu_s=mu_s,
        lambda_s=lambda_s,
        R_fp=R_fp,
        R_fs=R_fs,
        R_ps=R_ps,
        R_pair_cholesky=R_pair_cholesky,
        pair_weight_epsilon=1.0e-12 if R_pair_cholesky is not None else 0.0,
        theta_fp=theta_fp,
        ell_Gamma=ell_gamma,
        gamma_mobility=gamma_mobility,
        gamma_delta_epsilon=gamma_delta_epsilon,
        transfer_velocity=transfer_velocity,
        lag_alpha_in_constitutive_laws=lag_alpha_in_constitutive_laws,
        inactive_velocity_extension_factor=inactive_velocity_extension_factor,
        inactive_pressure_extension_factor=inactive_pressure_extension_factor,
        inactive_phi_extension_factor=inactive_phi_extension_factor,
        inactive_displacement_extension_factor=inactive_displacement_extension_factor,
        phi_extension_value=phi_b,
    )

    pore_momentum_outflow_key = _pore_momentum_outflow_key(pore_momentum_outflow)
    if pore_momentum_outflow_key != "none":
        top_edges = mesh.edge_bitset("top")
        if top_edges.cardinality() == 0:
            raise RuntimeError("Top boundary tag not found for the pore momentum outflow term.")
        d_top = dS(defined_on=top_edges, metadata={"q": int(quad_order)})
        n_top = FacetNormal()
        alpha_pore_out = state["alpha_n"] if bool(lag_alpha_in_constitutive_laws) else state["alpha_k"]
        _, P_pore_out, _ = one_domain_contents(alpha_pore_out, state["phi_k"])
        rho_p_out_c = _named_c("tc_b7_pore_outflow_rho_p", float(rho_p))
        outflow_factor_c = _named_c("tc_b7_pore_momentum_outflow_factor", float(pore_momentum_outflow_factor))
        vn_p = dot(state["v_p_k"], n_top)
        if pore_momentum_outflow_key == "outflow_only":
            boundary_speed = _smooth_positive_named(
                vn_p,
                eta=float(pore_momentum_outflow_smooth_eta),
                name="tc_b7_pore_momentum_outflow_smooth_pos",
            )
        else:
            boundary_speed = vn_p
        r_pore_outflow = (
            outflow_factor_c
            * P_pore_out
            * rho_p_out_c
            * boundary_speed
            * dot(state["v_p_k"], test["w_p"])
            * d_top
        )
        outflow_jac = linearize_form(
            r_pore_outflow,
            [
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
            [
                trial["dv_f"],
                trial["dp_f"],
                trial["dv_p"],
                trial["dp_p"],
                trial["dv_s"],
                trial["du_s"],
                trial["dalpha"],
                trial["dphi"],
                trial["dGamma"],
            ],
        )
        r_momentum_terms = dict(forms.r_momentum_terms)
        r_momentum_terms["pore_outlet_momentum_flux"] = r_pore_outflow
        a_terms = dict(forms.a_terms)
        a_terms["pore_outlet_momentum_flux"] = outflow_jac
        a_terms["total"] = forms.jacobian_form + outflow_jac
        forms = replace(
            forms,
            residual_form=forms.residual_form + r_pore_outflow,
            jacobian_form=forms.jacobian_form + outflow_jac,
            r_momentum_p=forms.r_momentum_p + r_pore_outflow,
            r_total_momentum=forms.r_total_momentum + r_pore_outflow,
            r_momentum_terms=r_momentum_terms,
            a_terms=a_terms,
        )

    inactive_domain_counts: dict[str, dict[str, int]] = {}
    inactive_domain_tagged: set[int] = set()
    if bool(inactive_domain_closure):
        inactive_domain_counts, inactive_domain_tagged = _tag_inactive_three_constituent_domains(
            dh,
            mesh,
            state["alpha_n"],
            alpha_low=float(inactive_alpha_low),
            alpha_high=float(inactive_alpha_high),
            previous_tagged=None,
        )
        print(
            "[setup] three_constituent inactive_domain_closure: "
            f"alpha_low={float(inactive_alpha_low):.6g}, "
            f"alpha_high={float(inactive_alpha_high):.6g}, "
            f"counts={inactive_domain_counts}",
            flush=True,
        )

    alpha_bc = lambda x, y, t: _alpha_profile(x, y, y_interface=y_interface, eps_alpha=eps_alpha)
    phi_bc = lambda x, y, t: float(phi_b)
    inlet_y = lambda x, y, t: _bottom_inlet_y(
        x,
        Lx=Lx,
        v_in=v_in,
        ramp=_cosine_ramp_value(float(t), float(t_ramp)),
    )

    bcs: list[BoundaryCondition] = []
    for tag in ("left", "right"):
        bcs.extend(
            [
                BoundaryCondition("vf_x", "dirichlet", tag, _zero_scalar),
                BoundaryCondition("vf_y", "dirichlet", tag, _zero_scalar),
                BoundaryCondition("vp_x", "dirichlet", tag, _zero_scalar),
                BoundaryCondition("vs_x", "dirichlet", tag, _zero_scalar),
                BoundaryCondition("vs_y", "dirichlet", tag, _zero_scalar),
                BoundaryCondition("us_x", "dirichlet", tag, _zero_scalar),
                BoundaryCondition("us_y", "dirichlet", tag, _zero_scalar),
                BoundaryCondition("alpha", "dirichlet", tag, alpha_bc),
                BoundaryCondition("phi", "dirichlet", tag, phi_bc),
            ]
        )
    bcs.extend(
        [
            BoundaryCondition("vf_x", "dirichlet", "bottom", _zero_scalar),
            BoundaryCondition("vf_y", "dirichlet", "bottom", inlet_y),
            BoundaryCondition("vp_x", "dirichlet", "bottom", _zero_scalar),
            BoundaryCondition("vp_y", "dirichlet", "bottom", _zero_scalar),
            BoundaryCondition("vs_x", "dirichlet", "bottom", _zero_scalar),
            BoundaryCondition("vs_y", "dirichlet", "bottom", _zero_scalar),
            BoundaryCondition("us_x", "dirichlet", "bottom", _zero_scalar),
            BoundaryCondition("us_y", "dirichlet", "bottom", _zero_scalar),
            BoundaryCondition("alpha", "dirichlet", "bottom", alpha_bc),
            BoundaryCondition("phi", "dirichlet", "bottom", phi_bc),
            BoundaryCondition("pf", "dirichlet", "top", _zero_scalar),
            BoundaryCondition("pp", "dirichlet", "top", _zero_scalar),
            BoundaryCondition("alpha", "dirichlet", "top", alpha_bc),
            BoundaryCondition("phi", "dirichlet", "top", phi_bc),
        ]
    )
    bcs_homog = _make_homogeneous_bcs(bcs)
    pp_lo = None if pore_pressure_lower_bound is None else float(pore_pressure_lower_bound)
    pp_hi = None if pore_pressure_upper_bound is None else float(pore_pressure_upper_bound)
    if pp_lo is not None and not np.isfinite(pp_lo):
        pp_lo = None
    if pp_hi is not None and not np.isfinite(pp_hi):
        pp_hi = None
    c_by_field = {"alpha": float(pdas_c), "phi": float(pdas_c)}
    if pp_lo is not None or pp_hi is not None:
        c_by_field["pp"] = float(pdas_c)

    solver = build_three_constituent_pdas_solver(
        forms,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=float(newton_tol),
            newton_rtol=float(newton_rtol),
            max_newton_iter=int(max_newton_iter),
            print_level=1,
            line_search=True,
            ls_max_iter=16,
            ls_reduction=0.5,
        ),
        vi_params=VIParameters(
            c=float(pdas_c),
            c_by_field=c_by_field,
            project_initial_guess=True,
            project_each_iteration=False,
            active_set_persistence=1,
            inactive_reg_lambda0=1.0e-10,
            inactive_reg_lambda_max=1.0e4,
        ),
        lin_params=LinearSolverParameters(backend=str(linear_backend), tol=1.0e-10, maxit=10000),
        backend=str(backend),
        quad_order=int(quad_order),
        alpha_bounds=(0.0, 1.0),
        phi_bounds=(0.0, 1.0),
        pore_pressure_bounds=(pp_lo, pp_hi),
    )

    if hasattr(solver, "configure_identified_manifold_recovery"):
        solver.configure_identified_manifold_recovery(
            proximal_fields=("alpha", "phi"),
            ptc_fields=("vf_x", "vf_y", "vp_x", "vp_y", "vs_x", "vs_y", "us_x", "us_y"),
        )

    functions = [
        state["v_f_k"],
        state["p_f_k"],
        state["v_p_k"],
        state["p_p_k"],
        state["v_s_k"],
        state["u_s_k"],
        state["alpha_k"],
        state["phi_k"],
        state["Gamma_k"],
    ]
    prev_functions = [
        state["v_f_n"],
        state["p_f_k"],
        state["v_p_n"],
        state["p_p_k"],
        state["v_s_n"],
        state["u_s_n"],
        state["alpha_n"],
        state["phi_n"],
        state["Gamma_n"],
    ]

    # Keep previous pressure arrays independent for the time-loop predictor.
    p_f_n = Function("p_f_n", "pf", dof_handler=dh)
    p_p_n = Function("p_p_n", "pp", dof_handler=dh)
    p_f_n.nodal_values[:] = state["p_f_k"].nodal_values[:]
    p_p_n.nodal_values[:] = state["p_p_k"].nodal_values[:]
    prev_functions[1] = p_f_n
    prev_functions[3] = p_p_n

    def _on_dt_change(new_dt: float) -> None:
        dt_c.value = float(new_dt)

    nmax = int(max_steps) if max_steps is not None else max(1, int(math.ceil(float(final_time) / max(float(dt), 1.0e-30))))
    time_params = TimeStepperParameters(
        dt=float(dt),
        max_steps=nmax,
        final_time=float(final_time),
        stop_on_steady=False,
        allow_dt_reduction=bool(allow_dt_reduction),
        dt_reduction_factor=float(dt_reduction_factor),
        dt_decrease_factor_slow=float(dt_reduction_factor),
        dt_min=float(dt_min),
        dt_max=float(dt),
        on_dt_change=_on_dt_change,
        predictor="prev",
    )

    history_path = outdir / "benchmark7_history.csv"
    extrema_path = outdir / "benchmark7_extrema_history.csv"
    vtk_dir = outdir / "vtk"
    history_state = {"step": 0, "time": 0.0, "header": False}
    extrema_state = {"header": False}

    def _profile_metrics_from_rows(rows: list[dict[str, float]]) -> dict[str, float]:
        out: dict[str, float] = {}
        finite_uy = np.asarray([row["u_y"] for row in rows if np.isfinite(row["u_y"])], dtype=float)
        if finite_uy.size:
            out["profile_u_y_min"] = float(np.min(finite_uy))
            out["profile_u_y_max"] = float(np.max(finite_uy))
            out["profile_u_y_linf"] = float(np.linalg.norm(finite_uy, ord=np.inf))
        finite_vbar_y = np.asarray([row["v_bar_y"] for row in rows if np.isfinite(row["v_bar_y"])], dtype=float)
        if finite_vbar_y.size:
            out["profile_v_bar_y_min"] = float(np.min(finite_vbar_y))
            out["profile_v_bar_y_max"] = float(np.max(finite_vbar_y))
            out["profile_v_bar_y_linf"] = float(np.linalg.norm(finite_vbar_y, ord=np.inf))
        finite_rho_bar = np.asarray([row["rho_bar"] for row in rows if np.isfinite(row["rho_bar"])], dtype=float)
        if finite_rho_bar.size:
            out["profile_rho_bar_min"] = float(np.min(finite_rho_bar))
            out["profile_rho_bar_max"] = float(np.max(finite_rho_bar))
        return out

    def _append_history_row() -> None:
        history_state["step"] = int(history_state["step"]) + 1
        history_state["time"] = float(history_state["time"]) + float(getattr(time_params, "dt", dt))
        stride = max(int(history_stride), 1)
        if int(history_state["step"]) != 1 and int(history_state["step"]) % stride != 0:
            return
        metrics_now = _field_stats(state)
        rows_now = _sample_profile(
            dh=dh,
            mesh=mesh,
            state=state,
            Lx=Lx,
            y_profile=y_profile,
            n_samples=profile_samples,
            rho_f=rho_f,
            rho_p=rho_p,
            rho_s=rho_s,
        )
        metrics_now.update(_profile_metrics_from_rows(rows_now))
        selected = {
            "step": int(history_state["step"]),
            "time": float(history_state["time"]),
            "dt": float(getattr(time_params, "dt", dt)),
            "nNewton": int(getattr(solver, "_last_nonlinear_iterations", -1) or -1),
            "G_linf": float(getattr(solver, "_last_nonlinear_norm", float("nan"))),
            "profile_u_y_linf": metrics_now.get("profile_u_y_linf", float("nan")),
            "profile_u_y_max": metrics_now.get("profile_u_y_max", float("nan")),
            "profile_v_bar_y_linf": metrics_now.get("profile_v_bar_y_linf", float("nan")),
            "us_linf": metrics_now.get("us_linf", float("nan")),
            "vs_linf": metrics_now.get("vs_linf", float("nan")),
            "vf_linf": metrics_now.get("vf_linf", float("nan")),
            "vp_linf": metrics_now.get("vp_linf", float("nan")),
            "pf_linf": metrics_now.get("pf_linf", float("nan")),
            "pf_min": metrics_now.get("pf_min", float("nan")),
            "pf_max": metrics_now.get("pf_max", float("nan")),
            "pp_linf": metrics_now.get("pp_linf", float("nan")),
            "pp_min": metrics_now.get("pp_min", float("nan")),
            "pp_max": metrics_now.get("pp_max", float("nan")),
            "Gamma_linf": metrics_now.get("Gamma_linf", float("nan")),
            "Gamma_min": metrics_now.get("Gamma_min", float("nan")),
            "Gamma_max": metrics_now.get("Gamma_max", float("nan")),
            "alpha_min": metrics_now.get("alpha_min", float("nan")),
            "alpha_max": metrics_now.get("alpha_max", float("nan")),
            "phi_min": metrics_now.get("phi_min", float("nan")),
            "phi_max": metrics_now.get("phi_max", float("nan")),
        }
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected.keys()))
            if not bool(history_state["header"]):
                writer.writeheader()
                history_state["header"] = True
            writer.writerow(selected)

    def _append_extrema_row() -> None:
        stride = max(int(history_stride), 1)
        step = int(history_state["step"])
        if step <= 0:
            return
        if step != 1 and step % stride != 0:
            return
        h_ref = float(mesh_meta.get("interface_band_min_dy", float("nan")))
        if not np.isfinite(h_ref) or h_ref <= 0.0:
            h_ref = float(mesh_meta.get("adaptive_interface_target_h", float("nan")))
        if not np.isfinite(h_ref) or h_ref <= 0.0:
            h_ref = min(float(Lx) / max(int(nx), 1), float(Ly) / max(int(ny), 1))
        dt_now = float(getattr(time_params, "dt", dt))
        selected = {
            "step": step,
            "time": float(history_state["time"]),
            "dt": dt_now,
            "h_ref": h_ref,
            "mu_f": float(mu_f),
            "mu_p": float(mu_p),
        }
        selected.update(_scalar_extremum_record(dh=dh, mesh=mesh, field="pp", fun=state["p_p_k"], prefix="pp"))
        selected.update(_scalar_extremum_record(dh=dh, mesh=mesh, field="pf", fun=state["p_f_k"], prefix="pf"))
        selected.update(_scalar_extremum_record(dh=dh, mesh=mesh, field="Gamma", fun=state["Gamma_k"], prefix="Gamma"))
        selected.update(_scalar_extremum_record(dh=dh, mesh=mesh, field="phi", fun=state["phi_k"], prefix="phi"))
        selected.update(_scalar_extremum_record(dh=dh, mesh=mesh, field="alpha", fun=state["alpha_k"], prefix="alpha"))
        selected.update(_vector_extremum_record(dh=dh, mesh=mesh, field_x="vp_x", field_y="vp_y", vec=state["v_p_k"], prefix="vp"))
        selected.update(_vector_extremum_record(dh=dh, mesh=mesh, field_x="vf_x", field_y="vf_y", vec=state["v_f_k"], prefix="vf"))
        selected.update(_vector_extremum_record(dh=dh, mesh=mesh, field_x="vs_x", field_y="vs_y", vec=state["v_s_k"], prefix="vs"))
        selected.update(_vector_extremum_record(dh=dh, mesh=mesh, field_x="us_x", field_y="us_y", vec=state["u_s_k"], prefix="us"))
        for prefix, rho_val, mu_val in (("vp", rho_p, mu_p), ("vf", rho_f, mu_f), ("vs", rho_s, mu_s)):
            vel = float(selected.get(f"{prefix}_mag_max", float("nan")))
            selected[f"{prefix}_courant_h"] = float(vel * dt_now / max(h_ref, 1.0e-30)) if np.isfinite(vel) else float("nan")
            selected[f"{prefix}_peclet_h"] = float(float(rho_val) * vel * h_ref / max(abs(float(mu_val)), 1.0e-30)) if np.isfinite(vel) else float("nan")
        extrema_path.parent.mkdir(parents=True, exist_ok=True)
        with extrema_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected.keys()))
            if not bool(extrema_state["header"]):
                writer.writeheader()
                extrema_state["header"] = True
            writer.writerow(selected)

    def _vtk_functions() -> dict[str, object]:
        return {
            "v_f": state["v_f_k"],
            "p_f": state["p_f_k"],
            "v_p": state["v_p_k"],
            "p_p": state["p_p_k"],
            "v_s": state["v_s_k"],
            "u_s": state["u_s_k"],
            "alpha": state["alpha_k"],
            "phi": state["phi_k"],
            "Gamma": state["Gamma_k"],
        }

    def _write_vtk_snapshot(label: str) -> None:
        if int(vtk_every) <= 0:
            return
        try:
            vtk_dir.mkdir(parents=True, exist_ok=True)
            export_vtk(
                str(vtk_dir / f"{label}.vtu"),
                mesh=mesh,
                dof_handler=dh,
                functions=_vtk_functions(),
            )
        except Exception as exc:  # noqa: PERF203
            print(f"[warn] failed to export VTK snapshot {label!r}: {type(exc).__name__}: {exc}", flush=True)

    inactive_refresh_state = {"count": 0}

    def _refresh_inactive_domain_closure() -> None:
        nonlocal inactive_domain_counts, inactive_domain_tagged
        if not bool(inactive_domain_closure):
            return
        new_counts, new_tagged = _tag_inactive_three_constituent_domains(
            dh,
            mesh,
            state["alpha_n"],
            alpha_low=float(inactive_alpha_low),
            alpha_high=float(inactive_alpha_high),
            previous_tagged=inactive_domain_tagged,
        )
        if new_tagged != inactive_domain_tagged:
            old_tagged = set(inactive_domain_tagged)
            added = len(new_tagged - old_tagged)
            reactivated = old_tagged - new_tagged
            inactive_now = set(int(d) for d in getattr(dh, "dof_tags", {}).get("inactive", set()) or set())
            seed_stats = _seed_reactivated_inactive_dofs(
                dh,
                reactivated,
                functions=functions,
                prev_functions=prev_functions,
                inactive_dofs=inactive_now,
                fallback_by_field={"phi": float(phi_b)},
                projection="l2_patch",
                quad_order=max(int(quad_order), 2 * max(int(poly_order), int(pressure_order), int(scalar_order)) + 2),
            )
            removed = len(reactivated)
            inactive_domain_tagged = new_tagged
            inactive_domain_counts = new_counts
            candidate = _candidate_active_dofs_from_current_tags(solver, bcs_homog)
            solver.set_active_dofs(candidate)
            inactive_refresh_state["count"] = int(inactive_refresh_state["count"]) + 1
            refresh_count = int(inactive_refresh_state["count"])
            if refresh_count <= 3 or refresh_count % 20 == 0:
                print(
                    "[post] three_constituent inactive_domain_closure refreshed: "
                    f"refresh={refresh_count}, added={added}, removed={removed}, "
                    f"seeded={seed_stats}, "
                    f"active_dofs={int(getattr(solver, 'active_dofs', np.asarray([])).size)}, "
                    f"counts={inactive_domain_counts}",
                    flush=True,
                )
        else:
            inactive_domain_counts = new_counts

    def _post_timeloop(_funcs) -> None:
        if int(history_stride) > 0:
            _append_history_row()
            _append_extrema_row()
        elif int(vtk_every) > 0:
            history_state["step"] = int(history_state["step"]) + 1
            history_state["time"] = float(history_state["time"]) + float(getattr(time_params, "dt", dt))
        step = int(history_state["step"])
        if int(vtk_every) > 0 and step > 0 and (step == 1 or step % int(vtk_every) == 0):
            _write_vtk_snapshot(f"step={step:05d}_t={float(history_state['time']):.8e}")
        _refresh_inactive_domain_closure()

    if int(history_stride) > 0 or bool(inactive_domain_closure) or int(vtk_every) > 0:
        solver.post_timeloop_cb = _post_timeloop

    t0 = time.perf_counter()
    passed = False
    error = ""
    n_steps = 0
    elapsed = 0.0
    try:
        _, n_steps, elapsed = solver.solve_time_interval(
            functions=functions,
            prev_functions=prev_functions,
            time_params=time_params,
        )
        passed = True
    except Exception as exc:  # noqa: PERF203
        elapsed = time.perf_counter() - t0
        error = f"{type(exc).__name__}: {exc}"
        n_steps = int(history_state.get("step", 0) or 0)
        # The nonlinear solver leaves `functions` at the failed trial iterate.
        # Export the last accepted state instead, so failed-run plots and
        # summaries are diagnostics of the physical trajectory, not of a
        # rejected line-search state.
        for f, f_prev in zip(functions, prev_functions):
            f.nodal_values[:] = f_prev.nodal_values[:]
        _write_vtk_snapshot("last_accepted_after_failure")

    metrics = _field_stats(state)
    profile_rows = _sample_profile(
        dh=dh,
        mesh=mesh,
        state=state,
        Lx=Lx,
        y_profile=y_profile,
        n_samples=profile_samples,
        rho_f=rho_f,
        rho_p=rho_p,
        rho_s=rho_s,
    )
    metrics.update(_profile_metrics_from_rows(profile_rows))

    summary = {
        "case_id": "seboldt_physical_three_constituent",
        "passed": bool(passed),
        "error": error,
        "n_steps": int(n_steps),
        "elapsed_s": float(elapsed),
        "mesh": {
            "nx": int(nx),
            "ny": int(ny),
            "poly_order": int(poly_order),
            "pressure_order": int(pressure_order),
            "scalar_order": int(scalar_order),
            "total_dofs": int(dh.total_dofs),
            **mesh_meta,
        },
        "parameters": {
            "Lx": float(Lx),
            "Ly": float(Ly),
            "y_interface": float(y_interface),
            "eps_alpha": float(eps_alpha),
            "phi_b": float(phi_b),
            "v_in": float(v_in),
            "dt": float(dt),
            "final_time": float(final_time),
            "kappa": float(kappa),
            "R_fp": float(R_fp),
            "R_fs": float(R_fs),
            "R_ps": float(R_ps),
            "ell_Gamma": float(ell_gamma),
            "gamma_mobility": str(gamma_mobility),
            "gamma_delta_epsilon": float(gamma_delta_epsilon),
            "resistance_model": str(resistance_model),
            "transfer_velocity": str(transfer_velocity),
            "lag_alpha_in_constitutive_laws": bool(lag_alpha_in_constitutive_laws),
            "pore_momentum_outflow": str(pore_momentum_outflow_key),
            "pore_momentum_outflow_factor": float(pore_momentum_outflow_factor),
            "pore_momentum_outflow_smooth_eta": float(pore_momentum_outflow_smooth_eta),
            "inactive_velocity_extension_factor": float(inactive_velocity_extension_factor),
            "inactive_pressure_extension_factor": float(inactive_pressure_extension_factor),
            "inactive_phi_extension_factor": float(inactive_phi_extension_factor),
            "inactive_displacement_extension_factor": float(inactive_displacement_extension_factor),
            "pore_pressure_lower_bound": None if pp_lo is None else float(pp_lo),
            "pore_pressure_upper_bound": None if pp_hi is None else float(pp_hi),
            "inactive_domain_closure": bool(inactive_domain_closure),
            "inactive_alpha_low": float(inactive_alpha_low),
            "inactive_alpha_high": float(inactive_alpha_high),
            "inactive_domain_counts": inactive_domain_counts,
            "exported_state": "last_accepted" if error else "final",
            "backend": str(backend),
            "linear_backend": str(linear_backend),
            "history_stride": int(history_stride),
            "vtk_every": int(vtk_every),
        },
        "metrics": metrics,
    }

    profile_csv = outdir / "profile_final.csv"
    summary_json = outdir / "benchmark7_summary.json"
    summary_csv = outdir / "benchmark7_summary.csv"
    _write_profile(profile_csv, profile_rows)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary_csv(summary_csv, summary)
    _write_plot(outdir / "benchmark7_seboldt_profiles.png", profile_rows)

    return PhysicalSeboldtResult(
        passed=bool(passed),
        outdir=outdir,
        summary=summary,
        profile_rows=profile_rows,
    )


__all__ = [
    "PhysicalSeboldtResult",
    "run_physical_seboldt_three_constituent",
]
