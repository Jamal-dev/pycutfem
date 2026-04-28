#!/usr/bin/env python3
"""Seboldt et al. (2021) Example 2: moving-domain linear partitioned model.

This is a benchmark-local driver for the paper model itself, not the one-domain
Paper 1 surrogate. It implements the Example 2 "structure Darcy--Navier--Stokes"
moving-domain linear-wall case from:

  - Algorithm 3.2,
  - Example 2,
  - Figure 6 "partitioned moving-domain linear".

Scope:
  - moving fluid and porous domains,
  - linear wall model with mu_V = 0,
  - Taylor--Hood P2-P1 fluid discretization,
  - P2 displacement discretization,
  - RT1-P1dc Darcy discretization,
  - dynamic L_3 update from equation (5.3),
  - Example-2 geometry, parameters, and boundary conditions.

The structure step is solved on the reference porous mesh, using Remark 1 to
recast the interface terms onto the reference boundary. The Darcy and fluid
steps are solved on the moved current meshes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import basix.ufl
import gmsh
import numpy as np
import ufl
from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc as fem_petsc
try:
    from dolfinx.io import gmshio
except ImportError:  # dolfinx >= 0.10 exposes the helper as dolfinx.io.gmsh
    from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from pycutfem.solvers.coupling_acceleration import create_coupling_accelerator


REPO_ROOT = Path(__file__).resolve().parents[4]
REFERENCE_CSV = REPO_ROOT / "examples" / "biofilms" / "benchmarks" / "seboldt" / "reference_profiles_fig6.csv"


@dataclass(frozen=True)
class Example2Params:
    rho_f: float = 1.0
    mu_f: float = 0.035
    rho_s: float = 1.1
    c0: float = 1.0e-3
    mu_p: float = 1.67785e5
    lambda_p: float = 8.22148e6
    mu_v: float = 0.0
    alpha: float = 1.0
    gamma: float = 1.0e3
    delta: float = 10.0
    v_in: float = 5.0
    x0: float = 0.0
    x1: float = 1.0
    y_fluid0: float = 0.0
    y_fluid1: float = 1.0
    y_solid0: float = 1.0
    y_solid1: float = 1.5


@dataclass
class BoundaryMarkers:
    meshtags: mesh.MeshTags
    ids: dict[str, int]
    facets: dict[str, np.ndarray]


def _eps(v):
    return ufl.sym(ufl.grad(v))


def _round_key(xy: np.ndarray) -> tuple[float, float]:
    return (round(float(xy[0]), 12), round(float(xy[1]), 12))


def _petsc_options(factor_solver: str | None) -> dict[str, str]:
    opts = {"ksp_type": "preonly", "pc_type": "lu"}
    if factor_solver:
        opts["pc_factor_mat_solver_type"] = str(factor_solver)
    return opts


def _interpolation_points(element) -> np.ndarray:
    points = getattr(element, "interpolation_points")
    return np.asarray(points() if callable(points) else points, dtype=float)


def _linear_problem(
    a,
    L,
    *,
    bcs,
    u,
    petsc_options: dict[str, str],
    petsc_options_prefix: str,
):
    kwargs = {
        "bcs": bcs,
        "u": u,
        "petsc_options": petsc_options,
    }
    try:
        return fem_petsc.LinearProblem(
            a,
            L,
            petsc_options_prefix=str(petsc_options_prefix),
            **kwargs,
        )
    except TypeError:
        return fem_petsc.LinearProblem(a, L, **kwargs)


def _build_rect_mesh(x0: float, y0: float, x1: float, y1: float, nx: int, ny: int) -> mesh.Mesh:
    comm = MPI.COMM_WORLD
    model_rank = 0
    if comm.rank == model_rank:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(f"rect_{x0:.6f}_{y0:.6f}_{x1:.6f}_{y1:.6f}_{nx}_{ny}")
        p1 = gmsh.model.occ.addPoint(float(x0), float(y0), 0.0)
        p2 = gmsh.model.occ.addPoint(float(x1), float(y0), 0.0)
        p3 = gmsh.model.occ.addPoint(float(x1), float(y1), 0.0)
        p4 = gmsh.model.occ.addPoint(float(x0), float(y1), 0.0)
        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p4)
        l4 = gmsh.model.occ.addLine(p4, p1)
        cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        surface = gmsh.model.occ.addPlaneSurface([cl])
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setTransfiniteCurve(l1, int(nx) + 1)
        gmsh.model.mesh.setTransfiniteCurve(l3, int(nx) + 1)
        gmsh.model.mesh.setTransfiniteCurve(l2, int(ny) + 1)
        gmsh.model.mesh.setTransfiniteCurve(l4, int(ny) + 1)
        gmsh.model.mesh.setTransfiniteSurface(surface)
        gmsh.model.addPhysicalGroup(2, [surface], 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
    mesh_data = gmshio.model_to_mesh(gmsh.model, comm, model_rank, gdim=2)
    if hasattr(mesh_data, "mesh"):
        msh = mesh_data.mesh
    else:
        msh = mesh_data[0]
    if comm.rank == model_rank:
        gmsh.finalize()
    return msh


def _build_boundary_markers(msh: mesh.Mesh, domain: str, params: Example2Params) -> BoundaryMarkers:
    fdim = msh.topology.dim - 1
    tol = 1.0e-12
    x0 = float(params.x0)
    x1 = float(params.x1)
    if domain == "fluid":
        y_bottom = float(params.y_fluid0)
        y_top = float(params.y_fluid1)
        named = {
            "left": mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], x0, atol=tol)),
            "right": mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], x1, atol=tol)),
            "bottom": mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], y_bottom, atol=tol)),
            "interface": mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], y_top, atol=tol)),
        }
    elif domain == "solid":
        y_bottom = float(params.y_solid0)
        y_top = float(params.y_solid1)
        named = {
            "left": mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], x0, atol=tol)),
            "right": mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], x1, atol=tol)),
            "interface": mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], y_bottom, atol=tol)),
            "top": mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], y_top, atol=tol)),
        }
    else:
        raise ValueError(f"Unsupported domain kind {domain!r}.")

    ids = {name: i + 1 for i, name in enumerate(named)}
    facet_list = []
    value_list = []
    for name, facets in named.items():
        facets = np.asarray(facets, dtype=np.int32)
        facet_list.append(facets)
        value_list.append(np.full_like(facets, ids[name], dtype=np.int32))
    all_facets = np.concatenate(facet_list) if facet_list else np.zeros((0,), dtype=np.int32)
    all_values = np.concatenate(value_list) if value_list else np.zeros((0,), dtype=np.int32)
    order = np.argsort(all_facets)
    tags = mesh.meshtags(msh, fdim, all_facets[order], all_values[order])
    return BoundaryMarkers(meshtags=tags, ids=ids, facets={k: np.asarray(v, dtype=np.int32) for k, v in named.items()})


def _build_vertex_map(msh: mesh.Mesh, Vgeom_vec) -> tuple[np.ndarray, np.ndarray]:
    ref_coords = np.asarray(msh.geometry.x[:, :2], dtype=float).copy()
    dof_coords = np.asarray(Vgeom_vec.tabulate_dof_coordinates()[:, :2], dtype=float)
    geom_lookup = {_round_key(xy): i for i, xy in enumerate(ref_coords)}
    order = np.asarray([geom_lookup[_round_key(xy)] for xy in dof_coords], dtype=np.int32)
    return ref_coords, order


def _build_pointwise_transfer_order(src_space, dst_space) -> np.ndarray:
    src_coords = np.asarray(src_space.tabulate_dof_coordinates()[:, :2], dtype=float)
    dst_coords = np.asarray(dst_space.tabulate_dof_coordinates()[:, :2], dtype=float)
    lookup = {_round_key(xy): i for i, xy in enumerate(src_coords)}
    return np.asarray([lookup[_round_key(xy)] for xy in dst_coords], dtype=np.int32)


def _update_current_mesh(
    msh_cur: mesh.Mesh,
    ref_coords: np.ndarray,
    disp_ref,
    Vgeom_vec,
    vertex_order: np.ndarray,
    tmp_geom_vec=None,
) -> None:
    src = disp_ref
    if tmp_geom_vec is not None:
        tmp_geom_vec.interpolate(disp_ref)
        src = tmp_geom_vec
    vals = np.asarray(src.x.array, dtype=float).reshape((-1, Vgeom_vec.dofmap.index_map_bs))
    msh_cur.geometry.x[:, :2] = ref_coords + vals[vertex_order, :2]


def _copy_coefficients(dst, src) -> None:
    dst.x.array[:] = src.x.array
    dst.x.scatter_forward()


def _copy_coefficients_by_order(dst, src, src_rows_for_dst: np.ndarray) -> None:
    src_bs = int(src.function_space.dofmap.index_map_bs)
    dst_bs = int(dst.function_space.dofmap.index_map_bs)
    if src_bs != dst_bs:
        raise ValueError("Pointwise coefficient transfer requires matching block sizes.")
    src_vals = np.asarray(src.x.array, dtype=float).reshape((-1, src_bs))
    dst_vals = np.asarray(dst.x.array, dtype=float).reshape((-1, dst_bs))
    dst_vals[:] = src_vals[np.asarray(src_rows_for_dst, dtype=np.int32), :]
    dst.x.scatter_forward()


def _extract_subfunction(parent, subspace, parent_to_sub: np.ndarray, target=None):
    if target is None:
        target = fem.Function(subspace)
    target.x.array[:] = np.asarray(parent.x.array, dtype=float)[np.asarray(parent_to_sub, dtype=np.int32)]
    target.x.scatter_forward()
    return target


def _locate_cells_for_points(msh: mesh.Mesh, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0,), dtype=np.int32)
    if pts.shape[1] == 2:
        pts3 = np.zeros((pts.shape[0], 3), dtype=np.float64)
        pts3[:, :2] = pts
    else:
        pts3 = pts
    bb = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(bb, pts3)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts3)
    cells = np.empty((pts3.shape[0],), dtype=np.int32)
    for i in range(pts3.shape[0]):
        links = colliding.links(i)
        if len(links) == 0:
            raise RuntimeError(f"Failed to locate point {pts[i, :]} in mesh.")
        cells[i] = int(links[0])
    return cells


def _eval_function_at_points(fn, points: np.ndarray, cells: np.ndarray | None = None) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, fn.function_space.dofmap.bs if fn.function_space.dofmap.bs > 0 else 1), dtype=float)
    if pts.shape[1] == 2:
        pts3 = np.zeros((pts.shape[0], 3), dtype=np.float64)
        pts3[:, :2] = pts
    else:
        pts3 = pts
    if cells is None:
        cells_arr = _locate_cells_for_points(fn.function_space.mesh, pts3)
    else:
        cells_arr = np.asarray(cells, dtype=np.int32)
        if cells_arr.shape[0] != pts3.shape[0]:
            raise ValueError("cells and points must have the same length.")
    values = fn.eval(np.asarray(pts3, dtype=np.float64), cells_arr)
    return np.asarray(values, dtype=float)


def _boundary_dof_info(V, fdim: int, facets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dofs = np.asarray(fem.locate_dofs_topological(V, fdim, facets), dtype=np.int32)
    coords = np.asarray(V.tabulate_dof_coordinates()[dofs, :2], dtype=float)
    order = np.argsort(coords[:, 0], kind="mergesort")
    return dofs[order], coords[order]


def _set_boundary_scalar_values(fn, dofs: np.ndarray, values: np.ndarray) -> None:
    arr = np.asarray(fn.x.array, dtype=float)
    arr[:] = 0.0
    arr[np.asarray(dofs, dtype=np.int32)] = np.asarray(values, dtype=float).ravel()
    fn.x.scatter_forward()


def _set_boundary_vector_values(fn, dofs: np.ndarray, values: np.ndarray) -> None:
    data = np.asarray(values, dtype=float)
    arr = np.asarray(fn.x.array, dtype=float).reshape((-1, fn.function_space.dofmap.index_map_bs))
    arr[:] = 0.0
    arr[np.asarray(dofs, dtype=np.int32), : data.shape[1]] = data
    fn.x.scatter_forward()


def _load_reference_curve(kappa: float) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    kappa_target = f"{float(kappa):.12g}"
    with REFERENCE_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["curve_label"] != "partitioned_moving_linear":
                continue
            if f"{float(row['kappa']):.12g}" != kappa_target:
                continue
            xs.append(float(row["x"]))
            ys.append(float(row["eta_y"]))
    if not xs:
        raise RuntimeError(f"Missing partitioned_moving_linear reference curve for kappa={kappa_target}.")
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _sample_interface_geometry(
    eta_ref,
    grad_eta_ref,
    x_ref: np.ndarray,
    y_ref: float,
    cells: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    pts_ref = np.column_stack([x_ref, np.full_like(x_ref, float(y_ref))])
    eta_vals = _eval_function_at_points(eta_ref, pts_ref, cells=cells)
    grad_vals = _eval_function_at_points(grad_eta_ref, pts_ref, cells=cells).reshape((-1, 2, 2))
    tangent = np.column_stack([1.0 + grad_vals[:, 0, 0], grad_vals[:, 1, 0]])
    Jg = np.linalg.norm(tangent, axis=1)
    tau = tangent / Jg[:, None]
    n_p = np.column_stack([tau[:, 1], -tau[:, 0]])
    n_f = -n_p
    pts_phys = np.column_stack([x_ref + eta_vals[:, 0], y_ref + eta_vals[:, 1]])
    return {
        "x_ref": np.asarray(x_ref, dtype=float),
        "pts_ref": pts_ref,
        "pts_phys": pts_phys,
        "eta": eta_vals,
        "grad": grad_vals,
        "Jg": Jg,
        "tau": tau,
        "n_p": n_p,
        "n_f": n_f,
    }


def _interface_norm(values: np.ndarray, Jg: np.ndarray, x_ref: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(Jg, dtype=float).ravel()
    xs = np.asarray(x_ref, dtype=float).ravel()
    return float(math.sqrt(max(np.trapz(vals * vals * weights, xs), 0.0)))


def _profile_metrics(num_x: np.ndarray, num_y: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray) -> dict[str, float]:
    ref_on_num = np.interp(num_x, ref_x, ref_y)
    diff = num_y - ref_on_num
    rmse = float(math.sqrt(float(np.mean(diff * diff))))
    linf = float(np.max(np.abs(diff)))
    num_amp = float(np.max(num_y))
    ref_amp = float(np.max(ref_y))
    num_peak_x = float(num_x[int(np.argmax(num_y))])
    ref_peak_x = float(ref_x[int(np.argmax(ref_y))])
    return {
        "rmse": rmse,
        "linf": linf,
        "num_amplitude": num_amp,
        "ref_amplitude": ref_amp,
        "peak_amplitude_relative_error": abs(num_amp - ref_amp) / max(abs(ref_amp), 1.0e-16),
        "peak_x_error": abs(num_peak_x - ref_peak_x),
    }


def _dynamic_L3_update(
    current_L: float,
    *,
    xi_vals: np.ndarray,
    q_vals: np.ndarray,
    fluid_velocity_vals: np.ndarray,
    fluid_traction_vals: np.ndarray,
    p_pore_vals: np.ndarray,
    n_f: np.ndarray,
    Jg: np.ndarray,
    x_ref_trace: np.ndarray,
    x_ref_samples: np.ndarray | None = None,
) -> tuple[float, float, float]:
    mc_trace_vals = np.einsum("ni,ni->n", np.asarray(xi_vals, dtype=float) + np.asarray(q_vals, dtype=float) - np.asarray(fluid_velocity_vals, dtype=float), np.asarray(n_f, dtype=float))
    pc_trace_vals = np.asarray(fluid_traction_vals, dtype=float).ravel() + np.asarray(p_pore_vals, dtype=float).ravel()
    v_n_trace_vals = np.einsum("ni,ni->n", np.asarray(fluid_velocity_vals, dtype=float), np.asarray(n_f, dtype=float))
    traction_vals = np.asarray(fluid_traction_vals, dtype=float).ravel()
    x_trace = np.asarray(x_ref_trace, dtype=float).ravel()
    Jg_trace = np.asarray(Jg, dtype=float).ravel()
    if x_ref_samples is not None and np.asarray(x_ref_samples, dtype=float).size > x_trace.size:
        x_norm = np.asarray(x_ref_samples, dtype=float).ravel()
        mc_vals = np.interp(x_norm, x_trace, mc_trace_vals)
        pc_vals = np.interp(x_norm, x_trace, pc_trace_vals)
        v_n_vals = np.interp(x_norm, x_trace, v_n_trace_vals)
        traction_norm_vals = np.interp(x_norm, x_trace, traction_vals)
        Jg_vals = np.interp(x_norm, x_trace, Jg_trace)
    else:
        x_norm = x_trace
        mc_vals = mc_trace_vals
        pc_vals = pc_trace_vals
        v_n_vals = v_n_trace_vals
        traction_norm_vals = traction_vals
        Jg_vals = Jg_trace
    e_mc = _interface_norm(mc_vals, Jg_vals, x_norm)
    denom_mc = _interface_norm(v_n_vals, Jg_vals, x_norm)
    e_pc = _interface_norm(pc_vals, Jg_vals, x_norm)
    denom_pc = _interface_norm(traction_norm_vals, Jg_vals, x_norm)
    emc_rel = float(e_mc / max(denom_mc, 1.0e-14))
    epc_rel = float(e_pc / max(denom_pc, 1.0e-14))
    new_L = float(current_L)
    if math.isfinite(emc_rel) and math.isfinite(epc_rel) and emc_rel > 0.0 and epc_rel > 0.0:
        new_L = float(current_L) * (emc_rel / epc_rel) ** (1.0 / 3.0)
    return new_L, emc_rel, epc_rel


def _reduce_robin_l(values: np.ndarray, *, fallback: float) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(fallback)
    return float(np.mean(finite))


def _flatten_interface_state(*, traction: np.ndarray, velocity: np.ndarray, qn: np.ndarray) -> np.ndarray:
    traction_arr = np.asarray(traction, dtype=float).ravel()
    velocity_arr = np.asarray(velocity, dtype=float)
    qn_arr = np.asarray(qn, dtype=float).ravel()
    if velocity_arr.ndim != 2 or velocity_arr.shape[1] != 2:
        raise ValueError("velocity must have shape (n, 2).")
    if traction_arr.size != velocity_arr.shape[0] or qn_arr.size != velocity_arr.shape[0]:
        raise ValueError("Interface state parts must share the same point count.")
    return np.hstack([traction_arr, velocity_arr.reshape((-1,)), qn_arr])


def _unflatten_interface_state(values: np.ndarray, npts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = np.asarray(values, dtype=float).ravel()
    expected = 4 * int(npts)
    if flat.size != expected:
        raise ValueError(f"Expected flattened interface state of size {expected}, got {flat.size}.")
    traction = flat[:npts].copy()
    velocity = flat[npts : 3 * npts].reshape((npts, 2)).copy()
    qn = flat[3 * npts :].copy()
    return traction, velocity, qn


def _interface_state_residual_metrics(current: np.ndarray, updated: np.ndarray) -> tuple[float, float, np.ndarray]:
    cur = np.asarray(current, dtype=float).ravel()
    upd = np.asarray(updated, dtype=float).ravel()
    if cur.size != upd.size:
        raise ValueError("Interface residual requires vectors of equal size.")
    residual = upd - cur
    abs_inf = float(np.max(np.abs(residual))) if residual.size > 0 else 0.0
    scale = max(float(np.max(np.abs(upd))) if upd.size > 0 else 0.0, 1.0e-14)
    rel_inf = float(abs_inf / scale)
    return rel_inf, abs_inf, residual


def _ensure_finite(name: str, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"Non-finite values detected in {name}.")


def _clip_points(points: np.ndarray, *, x0: float, x1: float, y0: float, y1: float, pad: float = 1.0e-11) -> np.ndarray:
    pts = np.asarray(points, dtype=float).copy()
    pts[:, 0] = np.clip(pts[:, 0], float(x0) + float(pad), float(x1) - float(pad))
    pts[:, 1] = np.clip(pts[:, 1], float(y0) + float(pad), float(y1) - float(pad))
    return pts


def _make_reference_tensor_grid(
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    nx: int,
    ny: int,
    pad_fraction: float = 1.0e-6,
) -> np.ndarray:
    if int(nx) < 2 or int(ny) < 2:
        raise ValueError("Reference tensor grid requires at least two points in each direction.")
    x_span = float(x1) - float(x0)
    y_span = float(y1) - float(y0)
    x_pad = max(abs(x_span) * float(pad_fraction), 1.0e-10)
    y_pad = max(abs(y_span) * float(pad_fraction), 1.0e-10)
    xs = np.linspace(float(x0) + x_pad, float(x1) - x_pad, int(nx), dtype=float)
    ys = np.linspace(float(y0) + y_pad, float(y1) - y_pad, int(ny), dtype=float)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack([xx.reshape((-1,)), yy.reshape((-1,))])


def _append_dict_rows_csv(path: Path, *, fieldnames: list[str], rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _space_time_sample_rows(
    *,
    step: int,
    time_value: float,
    pts_ref: np.ndarray,
    pts_phys: np.ndarray,
    field_values: dict[str, np.ndarray],
) -> list[dict[str, float]]:
    ref_pts = np.asarray(pts_ref, dtype=float)
    phys_pts = np.asarray(pts_phys, dtype=float)
    if ref_pts.shape != phys_pts.shape or ref_pts.ndim != 2 or ref_pts.shape[1] != 2:
        raise ValueError("Reference and physical sample points must both have shape (n, 2).")
    normalized_fields: dict[str, np.ndarray] = {}
    for name, values in field_values.items():
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape((-1, 1))
        if arr.ndim != 2 or arr.shape[0] != ref_pts.shape[0]:
            raise ValueError(f"Field {name!r} must have shape (n,) or (n, m) with n matching sample points.")
        normalized_fields[name] = arr
    rows: list[dict[str, float]] = []
    for idx in range(ref_pts.shape[0]):
        row: dict[str, float] = {
            "step": float(step),
            "time": float(time_value),
            "x_ref": float(ref_pts[idx, 0]),
            "y_ref": float(ref_pts[idx, 1]),
            "x_phys": float(phys_pts[idx, 0]),
            "y_phys": float(phys_pts[idx, 1]),
        }
        for name, arr in normalized_fields.items():
            if arr.shape[1] == 1:
                row[name] = float(arr[idx, 0])
            else:
                for comp_idx in range(arr.shape[1]):
                    row[f"{name}{comp_idx}"] = float(arr[idx, comp_idx])
        rows.append(row)
    return rows


def _write_profile_csv(path: Path, x: np.ndarray, eta_y: np.ndarray, ref_x: np.ndarray, ref_eta_y: np.ndarray) -> None:
    ref_on_x = np.interp(x, ref_x, ref_eta_y)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "eta_y", "reference_eta_y", "difference"])
        for xv, yv, rv in zip(x, eta_y, ref_on_x):
            writer.writerow([f"{float(xv):.16e}", f"{float(yv):.16e}", f"{float(rv):.16e}", f"{float(yv - rv):.16e}"])


def _profile_metrics_for_frames(
    *,
    profile_x: np.ndarray,
    profile_pts_ref: np.ndarray,
    profile_pts_phys: np.ndarray,
    eta_ref_fn,
    eta_phys_fn,
    ref_x: np.ndarray,
    ref_eta_y: np.ndarray,
) -> dict[str, object]:
    eta_ref_vals = _eval_function_at_points(eta_ref_fn, profile_pts_ref)
    eta_phys_vals = _eval_function_at_points(eta_phys_fn, profile_pts_phys)
    eta_y_ref = np.asarray(eta_ref_vals[:, 1], dtype=float)
    eta_y_phys = np.asarray(eta_phys_vals[:, 1], dtype=float)
    metrics_ref = _profile_metrics(profile_x, eta_y_ref, ref_x, ref_eta_y)
    metrics_phys = _profile_metrics(profile_x, eta_y_phys, ref_x, ref_eta_y)
    return {
        "eta_y_reference": eta_y_ref,
        "eta_y_physical": eta_y_phys,
        "metrics_reference": metrics_ref,
        "metrics_physical": metrics_phys,
    }


def _ensure_working_jit_compiler() -> dict[str, str]:
    updates: dict[str, str] = {}
    cc_env = os.environ.get("CC")
    cxx_env = os.environ.get("CXX")
    if cc_env and shutil.which(cc_env) is None:
        fallback = shutil.which("cc") or shutil.which("gcc")
        if fallback:
            os.environ["CC"] = fallback
            updates["CC"] = fallback
    if cxx_env and shutil.which(cxx_env) is None:
        fallback = shutil.which("c++") or shutil.which("g++")
        if fallback:
            os.environ["CXX"] = fallback
            updates["CXX"] = fallback
    return updates


def _write_history_csvs(outdir: Path, l_history: list[dict[str, float]], progress_history: list[dict[str, float]]) -> None:
    with (outdir / "L_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "time", "L", "e_mc_rel", "e_pc_rel"])
        writer.writeheader()
        for row in l_history:
            writer.writerow(row)
    with (outdir / "progress_history.csv").open("w", newline="") as handle:
        base_fields = [
            "step",
            "time",
            "L",
            "e_mc_rel",
            "e_pc_rel",
            "outer_sweeps",
            "outer_rel_inf",
            "outer_abs_inf",
            "outer_omega",
            "uy_max",
            "uy_max_physical",
            "rmse",
            "rmse_physical",
            "linf",
            "peak_amplitude_relative_error",
            "peak_amplitude_relative_error_physical",
        ]
        extra_fields: list[str] = []
        for row in progress_history:
            for key in row:
                if key not in base_fields and key not in extra_fields:
                    extra_fields.append(key)
        writer = csv.DictWriter(
            handle,
            fieldnames=base_fields + extra_fields,
        )
        writer.writeheader()
        for row in progress_history:
            writer.writerow(row)


def solve_case(args) -> dict[str, float | int | str]:
    params = Example2Params()
    nx = int(args.nx)
    ny_fluid = int(round((params.y_fluid1 - params.y_fluid0) / ((params.x1 - params.x0) / nx)))
    ny_solid = int(round((params.y_solid1 - params.y_solid0) / ((params.x1 - params.x0) / nx)))
    dt = float(args.dt)
    t_final = float(args.t_final)
    num_steps = int(round(t_final / dt))
    if abs(num_steps * dt - t_final) > 1.0e-12:
        raise ValueError("t_final must be an integer multiple of dt.")
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if MPI.COMM_WORLD.rank == 0:
        print(
            "[seboldt-moving-linear] start "
            f"kappa={float(args.kappa):.6e} nx={nx} dt={dt:.6e} t_final={t_final:.6e} "
            f"factor_solver={args.factor_solver!r} outdir={outdir}",
            flush=True,
        )
    compiler_updates = _ensure_working_jit_compiler()
    if MPI.COMM_WORLD.rank == 0 and compiler_updates:
        print(
            "[seboldt-moving-linear] compiler-fallback "
            + " ".join(f"{k}={v}" for k, v in compiler_updates.items()),
            flush=True,
        )

    msh_s_ref = _build_rect_mesh(params.x0, params.y_solid0, params.x1, params.y_solid1, nx, ny_solid)
    msh_f_ref = _build_rect_mesh(params.x0, params.y_fluid0, params.x1, params.y_fluid1, nx, ny_fluid)
    msh_s_cur = _build_rect_mesh(params.x0, params.y_solid0, params.x1, params.y_solid1, nx, ny_solid)
    msh_f_cur = _build_rect_mesh(params.x0, params.y_fluid0, params.x1, params.y_fluid1, nx, ny_fluid)

    tags_s_ref = _build_boundary_markers(msh_s_ref, "solid", params)
    tags_f_ref = _build_boundary_markers(msh_f_ref, "fluid", params)
    tags_s_cur = _build_boundary_markers(msh_s_cur, "solid", params)
    tags_f_cur = _build_boundary_markers(msh_f_cur, "fluid", params)

    ds_s_ref = ufl.Measure("ds", domain=msh_s_ref, subdomain_data=tags_s_ref.meshtags)
    ds_f_ref = ufl.Measure("ds", domain=msh_f_ref, subdomain_data=tags_f_ref.meshtags)
    ds_s_cur = ufl.Measure("ds", domain=msh_s_cur, subdomain_data=tags_s_cur.meshtags)
    ds_f_cur = ufl.Measure("ds", domain=msh_f_cur, subdomain_data=tags_f_cur.meshtags)
    dx_s_ref = ufl.dx(domain=msh_s_ref)
    dx_f_ref = ufl.dx(domain=msh_f_ref)
    dx_s_cur = ufl.dx(domain=msh_s_cur)
    dx_f_cur = ufl.dx(domain=msh_f_cur)

    V_s_ref = fem.functionspace(msh_s_ref, ("Lagrange", 2, (2,)))
    V_s_ref_scalar = fem.functionspace(msh_s_ref, ("Lagrange", 2))
    Q_s_ref = fem.functionspace(msh_s_ref, ("DG", 1))
    T_s_ref = fem.functionspace(msh_s_ref, ("DG", 1, (2, 2)))

    V_f_ref_geom = fem.functionspace(msh_f_ref, ("Lagrange", 2, (2,)))
    V_f_ref_vel = fem.functionspace(msh_f_ref, ("Lagrange", 2, (2,)))

    V_s_cur_vis = fem.functionspace(msh_s_cur, ("Lagrange", 2, (2,)))
    V_s_cur_scalar = fem.functionspace(msh_s_cur, ("Lagrange", 2))

    cell_s = msh_s_cur.ufl_cell().cellname()
    cell_f = msh_f_cur.ufl_cell().cellname()
    rt_el = basix.ufl.element("RT", cell_s, 1)
    dg1_el = basix.ufl.element("DG", cell_s, 1)
    W_s_cur = fem.functionspace(msh_s_cur, basix.ufl.mixed_element([rt_el, dg1_el]))
    Q_s_cur_flux, map_s_flux = W_s_cur.sub(0).collapse()
    Q_s_cur_pres, map_s_pres = W_s_cur.sub(1).collapse()

    vel_el = basix.ufl.element("Lagrange", cell_f, 2, shape=(2,))
    pres_el = basix.ufl.element("Lagrange", cell_f, 1)
    W_f_cur = fem.functionspace(msh_f_cur, basix.ufl.mixed_element([vel_el, pres_el]))
    V_f_cur_vel, map_f_vel = W_f_cur.sub(0).collapse()
    Q_f_cur_pres, map_f_pres = W_f_cur.sub(1).collapse()
    T_f_cur = fem.functionspace(msh_f_cur, ("DG", 1, (2, 2)))
    V_f_cur_scalar = fem.functionspace(msh_f_cur, ("Lagrange", 2))

    # State on reference meshes.
    eta_s_ref_old = fem.Function(V_s_ref, name="eta_s_ref_old")
    xi_s_ref_old = fem.Function(V_s_ref, name="xi_s_ref_old")
    p_s_ref_old = fem.Function(Q_s_ref, name="p_s_ref_old")
    eta_f_ref_old = fem.Function(V_f_ref_geom, name="eta_f_ref_old")
    w_f_ref = fem.Function(V_f_ref_geom, name="w_f_ref")
    v_f_ref_old = fem.Function(V_f_ref_vel, name="v_f_ref_old")

    # Current-mesh state used for evaluation and explicit data.
    eta_s_cur = fem.Function(V_s_cur_vis, name="eta_s_cur")
    xi_s_cur = fem.Function(V_s_cur_vis, name="xi_s_cur")
    q_s_cur_old = fem.Function(Q_s_cur_flux, name="q_s_cur_old")
    p_s_cur_old = fem.Function(Q_s_cur_pres, name="p_s_cur_old")
    v_f_cur_old = fem.Function(V_f_cur_vel, name="v_f_cur_old")
    p_f_cur_old = fem.Function(Q_f_cur_pres, name="p_f_cur_old")
    v_f_cur_old_on_new = fem.Function(V_f_cur_vel, name="v_f_cur_old_on_new")
    p_s_cur_old_on_new = fem.Function(Q_s_cur_pres, name="p_s_cur_old_on_new")
    w_f_cur = fem.Function(V_f_cur_vel, name="w_f_cur")

    # Boundary data containers.
    traction_ref_old = fem.Function(V_s_ref_scalar)
    vn_ref_old = fem.Function(V_s_ref_scalar)
    vt_ref_old = fem.Function(V_s_ref_scalar)
    qn_ref_old = fem.Function(V_s_ref_scalar)
    eta_top_ref = fem.Function(V_f_ref_geom)
    traction_s_cur = fem.Function(V_s_cur_scalar)
    vn_s_cur = fem.Function(V_s_cur_scalar)
    traction_f_cur = fem.Function(V_f_cur_scalar)
    xi_n_f_cur = fem.Function(V_f_cur_scalar)
    xi_t_f_cur = fem.Function(V_f_cur_scalar)
    q_n_f_cur = fem.Function(V_f_cur_scalar)

    grad_eta_ref = fem.Function(T_s_ref, name="grad_eta_ref")
    grad_v_cur = fem.Function(T_f_cur, name="grad_v_cur")

    eta_s_ref_new = fem.Function(V_s_ref, name="eta_s_ref_new")
    xi_s_ref_new = fem.Function(V_s_ref, name="xi_s_ref_new")
    eta_f_ref_new = fem.Function(V_f_ref_geom, name="eta_f_ref_new")
    w_s_cur = fem.Function(W_s_cur, name="w_s_cur")
    w_f_cur_mixed = fem.Function(W_f_cur, name="w_f_cur")
    q_s_cur_new = fem.Function(Q_s_cur_flux, name="q_s_cur_new")
    p_s_cur_new = fem.Function(Q_s_cur_pres, name="p_s_cur_new")
    v_f_cur_new = fem.Function(V_f_cur_vel, name="v_f_cur_new")
    p_f_cur_new = fem.Function(Q_f_cur_pres, name="p_f_cur_new")
    p_s_ref_new = fem.Function(Q_s_ref, name="p_s_ref_new")
    v_f_ref_new = fem.Function(V_f_ref_vel, name="v_f_ref_new")

    solid_ref_coords, solid_vertex_order = _build_vertex_map(msh_s_ref, V_s_ref)
    fluid_ref_coords, fluid_vertex_order = _build_vertex_map(msh_f_ref, V_f_ref_geom)
    fluid_ref_geom_to_cur_vel = _build_pointwise_transfer_order(V_f_ref_geom, V_f_cur_vel)
    fluid_ref_vel_to_cur_vel = _build_pointwise_transfer_order(V_f_ref_vel, V_f_cur_vel)
    fluid_cur_vel_to_ref_vel = _build_pointwise_transfer_order(V_f_cur_vel, V_f_ref_vel)

    zero_vec_ref = fem.Function(V_s_ref)
    zero_vec_cur_f = fem.Function(V_f_cur_vel)
    zero_flux = fem.Function(Q_s_cur_flux)
    zero_flux.x.array[:] = 0.0
    zero_flux.x.scatter_forward()

    # Dirichlet sets that do not change with time.
    fdim_s_ref = msh_s_ref.topology.dim - 1
    fdim_f_ref = msh_f_ref.topology.dim - 1
    fdim_s_cur = msh_s_cur.topology.dim - 1
    fdim_f_cur = msh_f_cur.topology.dim - 1

    dofs_s_ref_interface, coords_s_ref_interface = _boundary_dof_info(V_s_ref_scalar, fdim_s_ref, tags_s_ref.facets["interface"])
    dofs_f_ref_interface, coords_f_ref_interface = _boundary_dof_info(V_f_ref_geom, fdim_f_ref, tags_f_ref.facets["interface"])
    dofs_s_cur_interface, _ = _boundary_dof_info(V_s_cur_scalar, fdim_s_cur, tags_s_cur.facets["interface"])
    dofs_f_cur_interface, _ = _boundary_dof_info(V_f_cur_scalar, fdim_f_cur, tags_f_cur.facets["interface"])
    x_ref_trace = np.asarray(coords_s_ref_interface[:, 0], dtype=float)
    pts_ref_trace = np.column_stack([x_ref_trace, np.full_like(x_ref_trace, float(params.y_solid0))])
    locate_pad = 1.0e-10
    pts_s_ref_loc = np.column_stack([x_ref_trace, np.full_like(x_ref_trace, float(params.y_solid0) + locate_pad)])
    pts_f_ref_loc = np.column_stack([x_ref_trace, np.full_like(x_ref_trace, float(params.y_fluid1) - locate_pad)])
    cells_s_ref_interface = _locate_cells_for_points(msh_s_ref, pts_s_ref_loc)
    cells_f_ref_interface = _locate_cells_for_points(msh_f_ref, pts_f_ref_loc)
    cells_s_cur_interface = _locate_cells_for_points(msh_s_cur, pts_s_ref_loc)
    cells_f_cur_interface = _locate_cells_for_points(msh_f_cur, pts_f_ref_loc)

    if len(dofs_s_ref_interface) != len(dofs_f_ref_interface):
        raise RuntimeError("Reference interface trace spaces do not share the same number of P2 trace dofs.")
    if len(dofs_s_ref_interface) != len(dofs_s_cur_interface):
        raise RuntimeError("Solid current/reference interface trace spaces do not share the same number of P2 trace dofs.")
    if len(dofs_s_ref_interface) != len(dofs_f_cur_interface):
        raise RuntimeError("Fluid current/reference interface trace spaces do not share the same number of P2 trace dofs.")
    if not np.allclose(coords_f_ref_interface[:, 0], x_ref_trace):
        raise RuntimeError("Reference fluid/solid interface trace coordinates do not align.")

    dofs_s_ref_lr = np.concatenate(
        [
            fem.locate_dofs_topological(V_s_ref, fdim_s_ref, tags_s_ref.facets["left"]),
            fem.locate_dofs_topological(V_s_ref, fdim_s_ref, tags_s_ref.facets["right"]),
        ]
    )
    bc_s_ref_lr = fem.dirichletbc(zero_vec_ref, dofs_s_ref_lr)

    dofs_f_ref_zero = np.concatenate(
        [
            fem.locate_dofs_topological(V_f_ref_geom, fdim_f_ref, tags_f_ref.facets["left"]),
            fem.locate_dofs_topological(V_f_ref_geom, fdim_f_ref, tags_f_ref.facets["right"]),
            fem.locate_dofs_topological(V_f_ref_geom, fdim_f_ref, tags_f_ref.facets["bottom"]),
        ]
    )
    dofs_f_ref_top = np.asarray(fem.locate_dofs_topological(V_f_ref_geom, fdim_f_ref, tags_f_ref.facets["interface"]), dtype=np.int32)
    bc_f_ref_zero = fem.dirichletbc(fem.Function(V_f_ref_geom), dofs_f_ref_zero)

    V_s_sub0, _ = W_s_cur.sub(0).collapse()
    dofs_s_cur_left = fem.locate_dofs_topological((W_s_cur.sub(0), V_s_sub0), fdim_s_cur, tags_s_cur.facets["left"])
    dofs_s_cur_right = fem.locate_dofs_topological((W_s_cur.sub(0), V_s_sub0), fdim_s_cur, tags_s_cur.facets["right"])
    bc_s_cur_flux_left = fem.dirichletbc(zero_flux, dofs_s_cur_left, W_s_cur.sub(0))
    bc_s_cur_flux_right = fem.dirichletbc(zero_flux, dofs_s_cur_right, W_s_cur.sub(0))
    bcs_s_cur = [bc_s_cur_flux_left, bc_s_cur_flux_right]

    V_f_sub0, _ = W_f_cur.sub(0).collapse()
    dofs_f_cur_left = fem.locate_dofs_topological((W_f_cur.sub(0), V_f_sub0), fdim_f_cur, tags_f_cur.facets["left"])
    dofs_f_cur_right = fem.locate_dofs_topological((W_f_cur.sub(0), V_f_sub0), fdim_f_cur, tags_f_cur.facets["right"])
    dofs_f_cur_bottom = fem.locate_dofs_topological((W_f_cur.sub(0), V_f_sub0), fdim_f_cur, tags_f_cur.facets["bottom"])
    Q_f_sub1, _ = W_f_cur.sub(1).collapse()
    dofs_f_cur_p_pin = fem.locate_dofs_geometrical(
        (W_f_cur.sub(1), Q_f_sub1),
        lambda x: np.logical_and(np.isclose(x[0], params.x0), np.isclose(x[1], params.y_fluid0)),
    )

    wall_bc_fun = fem.Function(V_f_cur_vel)
    wall_bc_fun.x.array[:] = 0.0
    wall_bc_fun.x.scatter_forward()

    inflow_fun = fem.Function(V_f_cur_vel)

    def _inflow_cb(x):
        out = np.zeros((2, x.shape[1]), dtype=float)
        out[1, :] = 4.0 * float(params.v_in) * x[0] * (1.0 - x[0])
        return out

    inflow_fun.interpolate(_inflow_cb)
    inflow_fun.x.scatter_forward()
    p_pin = fem.Function(Q_f_sub1)
    p_pin.x.array[:] = 0.0
    p_pin.x.scatter_forward()
    bc_f_cur_left = fem.dirichletbc(wall_bc_fun, dofs_f_cur_left, W_f_cur.sub(0))
    bc_f_cur_right = fem.dirichletbc(wall_bc_fun, dofs_f_cur_right, W_f_cur.sub(0))
    bc_f_cur_bottom = fem.dirichletbc(inflow_fun, dofs_f_cur_bottom, W_f_cur.sub(0))
    bc_f_cur_p_pin = fem.dirichletbc(p_pin, dofs_f_cur_p_pin, W_f_cur.sub(1))
    bcs_f_cur = [bc_f_cur_left, bc_f_cur_right, bc_f_cur_bottom, bc_f_cur_p_pin]

    x_ref_samples = np.linspace(params.x0, params.x1, int(args.interface_samples), dtype=float)
    profile_x = np.linspace(params.x0, params.x1, int(args.profile_samples), dtype=float)
    profile_pts = np.column_stack([profile_x, np.full_like(profile_x, 1.25)])
    profile_pts_phys = np.column_stack([profile_x, np.full_like(profile_x, 1.25)])
    ref_x, ref_eta_y = _load_reference_curve(float(args.kappa))
    petsc_options = _petsc_options(args.factor_solver)
    L_value = float(args.L0)
    l_history: list[dict[str, float]] = []
    progress_history: list[dict[str, float]] = []
    t0 = time.time()
    base_outer_omega = float(
        np.clip(
            float(getattr(args, "outer_relaxation", 0.5)),
            float(getattr(args, "outer_relaxation_min", 1.0e-3)),
            float(getattr(args, "outer_relaxation_max", 1.0)),
        )
    )
    outer_update = str(getattr(args, "outer_update", "aitken")).strip().lower()
    outer_accelerator = create_coupling_accelerator(
        outer_update,
        relaxation=float(base_outer_omega),
        relaxation_min=float(getattr(args, "outer_relaxation_min", 1.0e-3)),
        relaxation_max=float(getattr(args, "outer_relaxation_max", 1.0)),
        history=int(getattr(args, "outer_history", 6)),
        regularization=float(getattr(args, "outer_regularization", 1.0e-10)),
        timestep_horizon=int(getattr(args, "outer_timestep_history", 1)),
    )

    for step in range(num_steps):
        t_np1 = (step + 1) * dt

        grad_eta_ref.interpolate(fem.Expression(ufl.grad(eta_s_ref_old), _interpolation_points(T_s_ref.element)))
        grad_eta_ref.x.scatter_forward()
        geom_old_trace = _sample_interface_geometry(
            eta_s_ref_old, grad_eta_ref, x_ref_trace, params.y_solid0, cells=cells_s_ref_interface
        )
        grad_v_cur.interpolate(fem.Expression(ufl.grad(v_f_cur_old), _interpolation_points(T_f_cur.element)))
        grad_v_cur.x.scatter_forward()
        v_old_vals = _eval_function_at_points(v_f_cur_old, geom_old_trace["pts_phys"], cells=cells_f_cur_interface)
        grad_v_old_vals = _eval_function_at_points(grad_v_cur, geom_old_trace["pts_phys"], cells=cells_f_cur_interface).reshape((-1, 2, 2))
        p_f_old_vals = _eval_function_at_points(p_f_cur_old, geom_old_trace["pts_phys"], cells=cells_f_cur_interface).reshape((-1,))
        q_old_vals = _eval_function_at_points(q_s_cur_old, geom_old_trace["pts_phys"], cells=cells_s_cur_interface)
        Dv_old = 0.5 * (grad_v_old_vals + np.transpose(grad_v_old_vals, axes=(0, 2, 1)))
        traction_old_vals = -p_f_old_vals + 2.0 * float(params.mu_f) * np.einsum(
            "ni,nij,nj->n", geom_old_trace["n_f"], Dv_old, geom_old_trace["n_f"]
        )
        qn_old_vals = np.einsum("ni,ni->n", q_old_vals, geom_old_trace["n_p"])

        interface_guess_vec = _flatten_interface_state(
            traction=traction_old_vals,
            velocity=v_old_vals,
            qn=qn_old_vals,
        )
        outer_accelerator.initialize_solution_step()
        outer_last_omega = float(base_outer_omega)
        outer_abs_tol = float(getattr(args, "outer_abs_tol", 0.0) or 0.0)
        outer_converged = False
        outer_sweeps = 0
        outer_rel_inf = float("inf")
        outer_abs_inf = float("inf")
        emc_rel = float("inf")
        epc_rel = float("inf")

        for outer_it in range(1, int(args.outer_it) + 1):
            outer_sweeps = int(outer_it)
            traction_iter_vals, v_iter_vals, qn_iter_vals = _unflatten_interface_state(interface_guess_vec, len(x_ref_trace))
            vn_iter_old_vals = np.einsum("ni,ni->n", v_iter_vals, geom_old_trace["n_p"])
            vt_iter_old_vals = np.einsum("ni,ni->n", v_iter_vals, geom_old_trace["tau"])

            _set_boundary_scalar_values(traction_ref_old, dofs_s_ref_interface, traction_iter_vals)
            _set_boundary_scalar_values(vn_ref_old, dofs_s_ref_interface, vn_iter_old_vals)
            _set_boundary_scalar_values(vt_ref_old, dofs_s_ref_interface, vt_iter_old_vals)
            _set_boundary_scalar_values(qn_ref_old, dofs_s_ref_interface, qn_iter_vals)

            eta_trial = ufl.TrialFunction(V_s_ref)
            zeta = ufl.TestFunction(V_s_ref)
            I2 = ufl.Identity(2)
            F_old = I2 + ufl.grad(eta_s_ref_old)
            J_old = ufl.det(F_old)
            FinvT_old = ufl.inv(F_old).T
            tangent_old = ufl.as_vector((1.0 + eta_s_ref_old[0].dx(0), eta_s_ref_old[1].dx(0)))
            Jg_old = ufl.sqrt(ufl.dot(tangent_old, tangent_old))
            tau_old = tangent_old / Jg_old
            n_p_old = ufl.as_vector((tau_old[1], -tau_old[0]))
            xi_trial_unknown = eta_trial / dt
            xi_old_from_eta = eta_s_ref_old / dt

            a_s_ref = (float(params.rho_s) / (dt * dt)) * ufl.inner(eta_trial, zeta) * dx_s_ref
            a_s_ref += (
                2.0 * float(params.mu_p) * ufl.inner(_eps(eta_trial), _eps(zeta))
                + float(params.lambda_p) * ufl.div(eta_trial) * ufl.div(zeta)
            ) * dx_s_ref
            if float(params.mu_v) != 0.0:
                a_s_ref += (2.0 * float(params.mu_v) / dt) * ufl.inner(_eps(eta_trial), _eps(zeta)) * dx_s_ref
            a_s_ref += float(L_value) * Jg_old * ufl.dot(xi_trial_unknown, n_p_old) * ufl.dot(zeta, n_p_old) * ds_s_ref(
                tags_s_ref.ids["interface"]
            )
            a_s_ref += float(params.gamma) * Jg_old * ufl.dot(xi_trial_unknown, tau_old) * ufl.dot(zeta, tau_old) * ds_s_ref(
                tags_s_ref.ids["interface"]
            )

            L_s_ref = (float(params.rho_s) / (dt * dt)) * ufl.inner(eta_s_ref_old + dt * xi_s_ref_old, zeta) * dx_s_ref
            if str(args.structure_pressure_coupling) == "linear_divergence":
                L_s_ref += float(params.alpha) * p_s_ref_old * ufl.div(zeta) * dx_s_ref
            else:
                L_s_ref += float(params.alpha) * J_old * p_s_ref_old * ufl.inner(FinvT_old, ufl.grad(zeta)) * dx_s_ref
            if float(params.mu_v) != 0.0:
                L_s_ref += (2.0 * float(params.mu_v) / dt) * ufl.inner(_eps(eta_s_ref_old), _eps(zeta)) * dx_s_ref
            L_s_ref += float(L_value) * Jg_old * ufl.dot(xi_old_from_eta, n_p_old) * ufl.dot(zeta, n_p_old) * ds_s_ref(
                tags_s_ref.ids["interface"]
            )
            L_s_ref += float(params.gamma) * Jg_old * ufl.dot(xi_old_from_eta, tau_old) * ufl.dot(zeta, tau_old) * ds_s_ref(
                tags_s_ref.ids["interface"]
            )
            L_s_ref += Jg_old * (traction_ref_old - float(L_value) * (qn_ref_old - vn_ref_old)) * ufl.dot(zeta, n_p_old) * ds_s_ref(
                tags_s_ref.ids["interface"]
            )
            L_s_ref += float(params.gamma) * Jg_old * vt_ref_old * ufl.dot(zeta, tau_old) * ds_s_ref(tags_s_ref.ids["interface"])

            problem_s = _linear_problem(
                fem.form(a_s_ref),
                fem.form(L_s_ref),
                bcs=[bc_s_ref_lr],
                u=eta_s_ref_new,
                petsc_options=petsc_options,
                petsc_options_prefix=f"b7_struct_step{step:04d}_outer{outer_it:02d}_",
            )
            problem_s.solve()
            eta_s_ref_new.x.scatter_forward()
            _ensure_finite("eta_s_ref_new", eta_s_ref_new.x.array)
            xi_s_ref_new.x.array[:] = (eta_s_ref_new.x.array - eta_s_ref_old.x.array) / dt
            xi_s_ref_new.x.scatter_forward()
            _ensure_finite("xi_s_ref_new", xi_s_ref_new.x.array)

            eta_top_samples = _eval_function_at_points(eta_s_ref_new, pts_ref_trace, cells=cells_s_ref_interface)
            _set_boundary_vector_values(eta_top_ref, dofs_f_ref_interface, eta_top_samples)
            bc_f_ref_top = fem.dirichletbc(eta_top_ref, dofs_f_ref_top)

            eta_f_trial = ufl.TrialFunction(V_f_ref_geom)
            chi = ufl.TestFunction(V_f_ref_geom)
            zero_geom_rhs = fem.Constant(msh_f_ref, np.array((0.0, 0.0), dtype=np.float64))
            problem_geom = _linear_problem(
                fem.form(ufl.inner(ufl.grad(eta_f_trial), ufl.grad(chi)) * dx_f_ref),
                fem.form(ufl.inner(zero_geom_rhs, chi) * dx_f_ref),
                bcs=[bc_f_ref_zero, bc_f_ref_top],
                u=eta_f_ref_new,
                petsc_options=petsc_options,
                petsc_options_prefix=f"b7_geom_step{step:04d}_outer{outer_it:02d}_",
            )
            problem_geom.solve()
            eta_f_ref_new.x.scatter_forward()
            _ensure_finite("eta_f_ref_new", eta_f_ref_new.x.array)
            w_f_ref.x.array[:] = (eta_f_ref_new.x.array - eta_f_ref_old.x.array) / dt
            w_f_ref.x.scatter_forward()
            _ensure_finite("w_f_ref", w_f_ref.x.array)

            _update_current_mesh(msh_s_cur, solid_ref_coords, eta_s_ref_new, V_s_ref, solid_vertex_order)
            _update_current_mesh(msh_f_cur, fluid_ref_coords, eta_f_ref_new, V_f_ref_geom, fluid_vertex_order)

            _copy_coefficients(eta_s_cur, eta_s_ref_new)
            _copy_coefficients(xi_s_cur, xi_s_ref_new)
            _copy_coefficients_by_order(w_f_cur, w_f_ref, fluid_ref_geom_to_cur_vel)
            _copy_coefficients_by_order(v_f_cur_old_on_new, v_f_ref_old, fluid_ref_vel_to_cur_vel)
            _copy_coefficients(p_s_cur_old_on_new, p_s_ref_old)

            grad_eta_ref.interpolate(fem.Expression(ufl.grad(eta_s_ref_new), _interpolation_points(T_s_ref.element)))
            grad_eta_ref.x.scatter_forward()
            geom_new_trace = _sample_interface_geometry(
                eta_s_ref_new, grad_eta_ref, x_ref_trace, params.y_solid0, cells=cells_s_ref_interface
            )
            _ensure_finite("geom_new_trace_pts_phys", geom_new_trace["pts_phys"])

            _set_boundary_scalar_values(traction_s_cur, dofs_s_cur_interface, traction_iter_vals)
            _set_boundary_scalar_values(traction_f_cur, dofs_f_cur_interface, traction_iter_vals)
            vn_iter_new_vals = np.einsum("ni,ni->n", v_iter_vals, geom_new_trace["n_p"])
            _set_boundary_scalar_values(vn_s_cur, dofs_s_cur_interface, vn_iter_new_vals)

            (q_trial, p_trial) = ufl.TrialFunctions(W_s_cur)
            (r_test, phi_test) = ufl.TestFunctions(W_s_cur)
            n_p_cur = ufl.FacetNormal(msh_s_cur)
            xi_n_cur = ufl.dot(xi_s_cur, n_p_cur)

            a_darcy = (
                float(args.kappa) ** (-1.0) * ufl.inner(q_trial, r_test)
                - p_trial * ufl.div(r_test)
                + phi_test * ufl.div(q_trial)
                + (float(params.c0) / dt) * p_trial * phi_test
            ) * dx_s_cur
            a_darcy += (float(params.delta) + float(L_value)) * ufl.dot(q_trial, n_p_cur) * ufl.dot(r_test, n_p_cur) * ds_s_cur(
                tags_s_cur.ids["interface"]
            )

            L_darcy = (float(params.c0) / dt) * p_s_cur_old_on_new * phi_test * dx_s_cur
            L_darcy += -(float(params.alpha) * ufl.div(xi_s_cur) * phi_test) * dx_s_cur
            L_darcy += (traction_s_cur - float(L_value) * (xi_n_cur - vn_s_cur)) * ufl.dot(r_test, n_p_cur) * ds_s_cur(
                tags_s_cur.ids["interface"]
            )

            problem_d = _linear_problem(
                fem.form(a_darcy),
                fem.form(L_darcy),
                bcs=bcs_s_cur,
                u=w_s_cur,
                petsc_options=petsc_options,
                petsc_options_prefix=f"b7_darcy_step{step:04d}_outer{outer_it:02d}_",
            )
            problem_d.solve()
            w_s_cur.x.scatter_forward()
            _ensure_finite("darcy_mixed_solution", w_s_cur.x.array)
            _extract_subfunction(w_s_cur, Q_s_cur_flux, map_s_flux, target=q_s_cur_new)
            _extract_subfunction(w_s_cur, Q_s_cur_pres, map_s_pres, target=p_s_cur_new)
            _ensure_finite("q_s_cur_new", q_s_cur_new.x.array)
            _ensure_finite("p_s_cur_new", p_s_cur_new.x.array)
            _copy_coefficients(p_s_ref_new, p_s_cur_new)

            xi_new_trace_vals = _eval_function_at_points(xi_s_cur, geom_new_trace["pts_phys"], cells=cells_s_cur_interface)
            q_new_trace_vals = _eval_function_at_points(q_s_cur_new, geom_new_trace["pts_phys"], cells=cells_s_cur_interface)
            xi_n_vals = np.einsum("ni,ni->n", xi_new_trace_vals, geom_new_trace["n_f"])
            xi_t_vals = np.einsum("ni,ni->n", xi_new_trace_vals, geom_new_trace["tau"])
            q_n_vals = np.einsum("ni,ni->n", q_new_trace_vals, geom_new_trace["n_f"])
            _set_boundary_scalar_values(xi_n_f_cur, dofs_f_cur_interface, xi_n_vals)
            _set_boundary_scalar_values(xi_t_f_cur, dofs_f_cur_interface, xi_t_vals)
            _set_boundary_scalar_values(q_n_f_cur, dofs_f_cur_interface, q_n_vals)

            (v_trial, pF_trial) = ufl.TrialFunctions(W_f_cur)
            (phi_v, psi_p) = ufl.TestFunctions(W_f_cur)
            n_f_cur = ufl.FacetNormal(msh_f_cur)
            tau_f_cur = ufl.as_vector((n_f_cur[1], -n_f_cur[0]))
            advector = v_f_cur_old_on_new - w_f_cur

            a_fluid = (
                (float(params.rho_f) / dt) * ufl.inner(v_trial, phi_v)
                + float(params.rho_f) * ufl.inner(ufl.grad(v_trial) * advector, phi_v)
                + 2.0 * float(params.mu_f) * ufl.inner(_eps(v_trial), _eps(phi_v))
                - pF_trial * ufl.div(phi_v)
                + psi_p * ufl.div(v_trial)
            ) * dx_f_cur
            a_fluid += float(L_value) * ufl.dot(v_trial, n_f_cur) * ufl.dot(phi_v, n_f_cur) * ds_f_cur(tags_f_cur.ids["interface"])
            a_fluid += float(params.gamma) * ufl.dot(v_trial, tau_f_cur) * ufl.dot(phi_v, tau_f_cur) * ds_f_cur(
                tags_f_cur.ids["interface"]
            )

            L_fluid = (float(params.rho_f) / dt) * ufl.inner(v_f_cur_old_on_new, phi_v) * dx_f_cur
            L_fluid += (traction_f_cur + float(L_value) * (xi_n_f_cur + q_n_f_cur)) * ufl.dot(phi_v, n_f_cur) * ds_f_cur(
                tags_f_cur.ids["interface"]
            )
            L_fluid += float(params.gamma) * xi_t_f_cur * ufl.dot(phi_v, tau_f_cur) * ds_f_cur(tags_f_cur.ids["interface"])

            problem_f = _linear_problem(
                fem.form(a_fluid),
                fem.form(L_fluid),
                bcs=bcs_f_cur,
                u=w_f_cur_mixed,
                petsc_options=petsc_options,
                petsc_options_prefix=f"b7_fluid_step{step:04d}_outer{outer_it:02d}_",
            )
            problem_f.solve()
            w_f_cur_mixed.x.scatter_forward()
            _ensure_finite("fluid_mixed_solution", w_f_cur_mixed.x.array)
            _extract_subfunction(w_f_cur_mixed, V_f_cur_vel, map_f_vel, target=v_f_cur_new)
            _extract_subfunction(w_f_cur_mixed, Q_f_cur_pres, map_f_pres, target=p_f_cur_new)
            _ensure_finite("v_f_cur_new", v_f_cur_new.x.array)
            _ensure_finite("p_f_cur_new", p_f_cur_new.x.array)
            _copy_coefficients_by_order(v_f_ref_new, v_f_cur_new, fluid_cur_vel_to_ref_vel)

            grad_v_cur.interpolate(fem.Expression(ufl.grad(v_f_cur_new), _interpolation_points(T_f_cur.element)))
            grad_v_cur.x.scatter_forward()
            v_new_vals = _eval_function_at_points(v_f_cur_new, geom_new_trace["pts_phys"], cells=cells_f_cur_interface)
            grad_v_new_vals = _eval_function_at_points(grad_v_cur, geom_new_trace["pts_phys"], cells=cells_f_cur_interface).reshape((-1, 2, 2))
            p_f_new_vals = _eval_function_at_points(p_f_cur_new, geom_new_trace["pts_phys"], cells=cells_f_cur_interface).reshape((-1,))
            p_pore_new_vals = _eval_function_at_points(p_s_cur_new, geom_new_trace["pts_phys"], cells=cells_s_cur_interface).reshape((-1,))
            Dv_new = 0.5 * (grad_v_new_vals + np.transpose(grad_v_new_vals, axes=(0, 2, 1)))
            traction_new_vals = -p_f_new_vals + 2.0 * float(params.mu_f) * np.einsum(
                "ni,nij,nj->n", geom_new_trace["n_f"], Dv_new, geom_new_trace["n_f"]
            )
            qn_new_vals = np.einsum("ni,ni->n", q_new_trace_vals, geom_new_trace["n_p"])
            raw_interface_vec = _flatten_interface_state(
                traction=traction_new_vals,
                velocity=v_new_vals,
                qn=qn_new_vals,
            )
            outer_rel_inf, outer_abs_inf, outer_residual_vec = _interface_state_residual_metrics(interface_guess_vec, raw_interface_vec)

            mc_trace_vals = np.einsum("ni,ni->n", xi_new_trace_vals + q_new_trace_vals - v_new_vals, geom_new_trace["n_f"])
            pc_trace_vals = traction_new_vals + p_pore_new_vals
            v_n_trace_vals = np.einsum("ni,ni->n", v_new_vals, geom_new_trace["n_f"])
            if x_ref_samples.size > x_ref_trace.size:
                mc_vals = np.interp(x_ref_samples, x_ref_trace, mc_trace_vals)
                pc_vals = np.interp(x_ref_samples, x_ref_trace, pc_trace_vals)
                v_n_new_vals = np.interp(x_ref_samples, x_ref_trace, v_n_trace_vals)
                traction_sample_vals = np.interp(x_ref_samples, x_ref_trace, traction_new_vals)
                Jg_vals = np.interp(x_ref_samples, x_ref_trace, geom_new_trace["Jg"])
                x_norm = x_ref_samples
            else:
                mc_vals = mc_trace_vals
                pc_vals = pc_trace_vals
                v_n_new_vals = v_n_trace_vals
                traction_sample_vals = traction_new_vals
                Jg_vals = geom_new_trace["Jg"]
                x_norm = geom_new_trace["x_ref"]
            e_mc = _interface_norm(mc_vals, Jg_vals, x_norm)
            denom_mc = _interface_norm(v_n_new_vals, Jg_vals, x_norm)
            e_pc = _interface_norm(pc_vals, Jg_vals, x_norm)
            denom_pc = _interface_norm(traction_sample_vals, Jg_vals, x_norm)
            emc_rel = e_mc / max(denom_mc, 1.0e-14)
            epc_rel = e_pc / max(denom_pc, 1.0e-14)
            if not math.isfinite(emc_rel):
                emc_rel = 0.0
            if not math.isfinite(epc_rel):
                epc_rel = 0.0

            outer_converged = bool(
                np.isfinite(outer_rel_inf)
                and outer_rel_inf <= float(args.outer_tol)
                or (outer_abs_tol > 0.0 and np.isfinite(outer_abs_inf) and outer_abs_inf <= outer_abs_tol)
            )
            if MPI.COMM_WORLD.rank == 0:
                print(
                    f"    [fixed-point] step={step + 1}/{num_steps} "
                    f"iter={outer_it}/{int(args.outer_it)} rel={outer_rel_inf:.3e} abs={outer_abs_inf:.3e} "
                    f"omega={float(outer_last_omega):.3e} converged={int(bool(outer_converged))}",
                    flush=True,
                )
            if outer_converged:
                interface_guess_vec = raw_interface_vec
                break

            update = outer_accelerator.compute_next_iterate(
                x_curr=np.asarray(interface_guess_vec, dtype=float),
                residual_curr=np.asarray(outer_residual_vec, dtype=float),
            )
            interface_guess_vec = np.asarray(update.next_iterate, dtype=float).copy()
            outer_last_omega = float(update.relaxation)

        if not outer_converged:
            outer_accelerator.finalize_solution_step(accepted=False)
            raise RuntimeError(
                f"Outer coupling stalled at step {step + 1} (t={t_np1:.4f}, dt={dt:.3e}) "
                f"after {outer_sweeps} sweeps: rel={outer_rel_inf:.3e}, abs={outer_abs_inf:.3e}. "
                "Reduce dt or increase the outer-iteration budget."
            )
        outer_accelerator.finalize_solution_step(accepted=True)

        if emc_rel > 0.0 and epc_rel > 0.0 and math.isfinite(emc_rel) and math.isfinite(epc_rel):
            L_value *= (emc_rel / epc_rel) ** (1.0 / 3.0)
        l_history.append(
            {
                "step": float(step + 1),
                "time": float(t_np1),
                "L": float(L_value),
                "e_mc_rel": float(emc_rel),
                "e_pc_rel": float(epc_rel),
            }
        )

        _copy_coefficients(eta_s_ref_old, eta_s_ref_new)
        _copy_coefficients(xi_s_ref_old, xi_s_ref_new)
        _copy_coefficients(eta_f_ref_old, eta_f_ref_new)
        _copy_coefficients(p_s_ref_old, p_s_ref_new)
        _copy_coefficients(v_f_ref_old, v_f_ref_new)
        _copy_coefficients(q_s_cur_old, q_s_cur_new)
        _copy_coefficients(p_s_cur_old, p_s_cur_new)
        _copy_coefficients(v_f_cur_old, v_f_cur_new)
        _copy_coefficients(p_f_cur_old, p_f_cur_new)

        if MPI.COMM_WORLD.rank == 0 and ((step + 1) % max(int(args.report_every), 1) == 0 or step + 1 == num_steps):
            report_profiles = _profile_metrics_for_frames(
                profile_x=profile_x,
                profile_pts_ref=profile_pts,
                profile_pts_phys=profile_pts_phys,
                eta_ref_fn=eta_s_ref_old,
                eta_phys_fn=eta_s_cur,
                ref_x=ref_x,
                ref_eta_y=ref_eta_y,
            )
            report_metrics = report_profiles["metrics_reference"]
            report_metrics_phys = report_profiles["metrics_physical"]
            progress_history.append(
                {
                    "step": float(step + 1),
                    "time": float(t_np1),
                    "L": float(L_value),
                    "e_mc_rel": float(emc_rel),
                    "e_pc_rel": float(epc_rel),
                    "outer_sweeps": float(outer_sweeps),
                    "outer_rel_inf": float(outer_rel_inf),
                    "outer_abs_inf": float(outer_abs_inf),
                    "outer_omega": float(outer_last_omega),
                    "uy_max": float(report_metrics["num_amplitude"]),
                    "uy_max_physical": float(report_metrics_phys["num_amplitude"]),
                    "rmse": float(report_metrics["rmse"]),
                    "rmse_physical": float(report_metrics_phys["rmse"]),
                    "linf": float(report_metrics["linf"]),
                    "peak_amplitude_relative_error": float(report_metrics["peak_amplitude_relative_error"]),
                    "peak_amplitude_relative_error_physical": float(report_metrics_phys["peak_amplitude_relative_error"]),
                }
            )
            print(
                f"[seboldt-moving-linear] step={step + 1}/{num_steps} "
                f"t={t_np1:.4f} L={L_value:.6e} e_mc={emc_rel:.3e} e_pc={epc_rel:.3e} "
                f"outer={outer_sweeps} rel={outer_rel_inf:.3e} abs={outer_abs_inf:.3e} "
                f"uy_max={report_metrics['num_amplitude']:.6e} uy_max_phys={report_metrics_phys['num_amplitude']:.6e} "
                f"rmse={report_metrics['rmse']:.3e} rmse_phys={report_metrics_phys['rmse']:.3e}",
                flush=True,
            )
            _write_history_csvs(outdir, l_history, progress_history)

    final_profiles = _profile_metrics_for_frames(
        profile_x=profile_x,
        profile_pts_ref=profile_pts,
        profile_pts_phys=profile_pts_phys,
        eta_ref_fn=eta_s_ref_old,
        eta_phys_fn=eta_s_cur,
        ref_x=ref_x,
        ref_eta_y=ref_eta_y,
    )
    eta_y = final_profiles["eta_y_reference"]
    eta_y_phys = final_profiles["eta_y_physical"]
    metrics = final_profiles["metrics_reference"]
    metrics_phys = final_profiles["metrics_physical"]

    _write_profile_csv(outdir / "uy_profile.csv", profile_x, eta_y, ref_x, ref_eta_y)
    _write_profile_csv(outdir / "uy_profile_physical.csv", profile_x, eta_y_phys, ref_x, ref_eta_y)
    _write_history_csvs(outdir, l_history, progress_history)

    summary = {
        "model": "seboldt_partitioned_moving_linear",
        "paper": "Seboldt et al. 2021 Example 2",
        "kappa": float(args.kappa),
        "nx": int(nx),
        "ny_fluid": int(ny_fluid),
        "ny_solid": int(ny_solid),
        "dt": float(dt),
        "t_final": float(t_final),
        "num_steps": int(num_steps),
        "outer_it": int(args.outer_it),
        "outer_tol": float(args.outer_tol),
        "outer_abs_tol": float(args.outer_abs_tol),
        "outer_update": str(args.outer_update),
        "outer_timestep_history": int(getattr(args, "outer_timestep_history", 1)),
        "final_L": float(L_value),
        "runtime_seconds": float(time.time() - t0),
        **metrics,
        "num_amplitude_physical": float(metrics_phys["num_amplitude"]),
        "peak_amplitude_relative_error_physical": float(metrics_phys["peak_amplitude_relative_error"]),
        "rmse_physical": float(metrics_phys["rmse"]),
        "linf_physical": float(metrics_phys["linf"]),
        "peak_x_error_physical": float(metrics_phys["peak_x_error"]),
    }
    with (outdir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kappa", type=float, default=1.0e-3, help="Hydraulic conductivity kappa from Example 2.")
    ap.add_argument("--nx", type=int, default=20, help="Horizontal cells. h=0.05 corresponds to nx=20.")
    ap.add_argument("--dt", type=float, default=1.0e-3, help="Time step.")
    ap.add_argument("--t-final", type=float, default=3.0, help="Final time.")
    ap.add_argument("--L0", type=float, default=2000.0, help="Initial L used by the paper before the dynamic L_3 update.")
    ap.add_argument("--interface-samples", type=int, default=201, help="Sample points used for interface transfers and L updates.")
    ap.add_argument("--profile-samples", type=int, default=401, help="Samples used for eta_y(x, 1.25).")
    ap.add_argument("--factor-solver", default="mumps", help="PETSc LU backend. The Example 2 driver is screened and run with mumps.")
    ap.add_argument("--report-every", type=int, default=100, help="Progress print frequency in time steps.")
    ap.add_argument("--outer-it", type=int, default=20, help="Maximum outer fixed-point sweeps per time step.")
    ap.add_argument("--outer-tol", type=float, default=1.0e-3, help="Relative outer fixed-point tolerance on the interface-state update.")
    ap.add_argument("--outer-abs-tol", type=float, default=0.0, help="Optional absolute outer fixed-point tolerance. Set <= 0 to disable.")
    ap.add_argument(
        "--outer-update",
        choices=("constant", "aitken", "iqn_ils", "iqln", "mvqn"),
        default="iqn_ils",
        help="Outer interface fixed-point update.",
    )
    ap.add_argument("--outer-history", type=int, default=6, help="History length used by the IQN/IQLN outer update.")
    ap.add_argument(
        "--outer-timestep-history",
        type=int,
        default=1,
        help="Accepted time-step horizon reused by the outer IQN-ILS history; MVQN reuses its Jacobian implicitly.",
    )
    ap.add_argument("--outer-regularization", type=float, default=1.0e-10, help="Tikhonov regularization used in the IQN/IQLN least-squares system.")
    ap.add_argument("--outer-relaxation", type=float, default=0.5, help="Base outer fixed-point relaxation.")
    ap.add_argument("--outer-relaxation-min", type=float, default=1.0e-3, help="Minimum outer relaxation allowed by Aitken/IQN.")
    ap.add_argument("--outer-relaxation-max", type=float, default=1.0, help="Maximum outer relaxation allowed by Aitken/IQN.")
    ap.add_argument(
        "--structure-pressure-coupling",
        choices=("nonlinear_reference", "linear_divergence"),
        default="nonlinear_reference",
        help="Structure-step pore-pressure coupling used in the moving linear-wall case.",
    )
    ap.add_argument(
        "--outdir",
        default=str(REPO_ROOT / "out" / "benchmark7_seboldt_partitioned_moving_linear"),
        help="Output directory.",
    )
    args = ap.parse_args()

    summary = solve_case(args)
    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
