#!/usr/bin/env python3
"""Christan Biofilm I reference model using a body-fitted ALE/FPI split solve.

This driver follows the model described in
`examples/biofilms/benchmarks/christan/christan.tex` and uses the local ALE/FPI
building blocks identified in:

- `examples/biofilms/benchmarks/seboldt/seboldt.tex`
- `examples/fsi_dealii_reference.py`
- `examples/FPI/fpi_mms_example41.py`

Model structure:

- outside the biofilm: stationary ALE Navier-Stokes on the deformed fluid mesh,
- inside the biofilm: Darcy pressure solve and linear poroelastic elasticity,
- mesh motion: harmonic extension from the biofilm displacement to the fluid mesh,
- coupling: explicit/partitioned in time with optional inner fixed-point sweeps.

The goal is not to reproduce COMSOL internals exactly, but to implement the
published Christan/Picioreanu formulation directly in the local pycutfem stack
using a clean, reviewer-defendable reference setup.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np

try:
    import gmsh  # type: ignore
except Exception:
    gmsh = None

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.fem import transform
from pycutfem.solvers.nonlinear_solver import (
    LinearSolverParameters,
    NewtonParameters,
    NewtonSolver,
    TimeStepperParameters,
)
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    Constant,
    FacetNormal,
    Function,
    Identity,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    cof,
    det,
    div,
    dot,
    grad,
    inner,
    inv,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dS, dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.gmsh_loader import mesh_from_gmsh

from examples.biofilms.benchmarks.christan.compare_sim_vs_christan import compare_case
from examples.biofilms.benchmarks.christan.prepare_biofilm_I_geometry import DEFAULT_OUT_DIR as DEFAULT_GEOMETRY_DIR


HERE = Path(__file__).resolve().parent
INITIAL_CONTOUR_CSV = HERE / "biofilm_I_initial_mm.csv"


def _symgrad(u):
    return Constant(0.5) * (grad(u) + grad(u).T)


def _coord_key(x: float, y: float, ndigits: int = 12) -> tuple[float, float]:
    return (round(float(x), ndigits), round(float(y), ndigits))


def _log(args: argparse.Namespace, message: str) -> None:
    if bool(getattr(args, "verbose", False)):
        print(message, flush=True)


def _timed(args: argparse.Namespace, label: str, fn: Callable[[], None]) -> None:
    _log(args, f"[stage] {label}: start")
    t0 = time.perf_counter()
    fn()
    _log(args, f"[stage] {label}: done in {time.perf_counter() - t0:.3f}s")


def _read_contour_m(path: Path) -> np.ndarray:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError(f"Empty contour CSV: {path}")
    pts_mm = np.asarray([(float(row["x_mm"]), float(row["y_mm"])) for row in rows], dtype=float)
    return 1.0e-3 * pts_mm


def _polygon_interface_chain(contour_m: np.ndarray) -> np.ndarray:
    pts = np.asarray(contour_m, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Expected contour array of shape (n,2).")
    if not np.allclose(pts[0], pts[-1], rtol=0.0, atol=1.0e-12):
        raise ValueError("Christan contour must be closed.")
    chain = pts[:-1].copy()
    if chain.shape[0] < 3:
        raise ValueError("Contour has too few points.")
    return chain


def _cosine_ramp_value(t_now: float, ramp_time: float) -> float:
    tr = float(ramp_time)
    if tr <= 0.0:
        return 1.0
    tt = max(0.0, float(t_now))
    if tt >= tr:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * tt / max(1.0e-12, tr)))


@dataclass
class CoordinateLookup:
    coords: np.ndarray
    values: np.ndarray
    dim: int

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must have shape (n,2).")
        if self.dim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[0] != coords.shape[0] or values.shape[1] != int(self.dim):
            raise ValueError("values shape does not match coords / dim.")
        self.coords = coords
        self.values = values
        self._dict = {_coord_key(x, y): values[i].copy() for i, (x, y) in enumerate(coords)}

    def __call__(self, x: float | np.ndarray, y: float | np.ndarray):
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        flat = np.c_[xa.reshape(-1), ya.reshape(-1)]
        out = np.empty((flat.shape[0], self.dim), dtype=float)
        for i, (xx, yy) in enumerate(flat):
            hit = self._dict.get(_coord_key(xx, yy))
            if hit is not None:
                out[i, :] = hit
                continue
            dist2 = np.sum((self.coords - np.asarray([xx, yy], dtype=float)) ** 2, axis=1)
            out[i, :] = self.values[int(np.argmin(dist2)), :]
        if self.dim == 1:
            return out[:, 0].reshape(xa.shape)
        return out.reshape(xa.shape + (self.dim,))

    def component(self, idx: int) -> Callable[[float, float], float]:
        if not (0 <= int(idx) < self.dim):
            raise IndexError(idx)

        def _wrapped(x, y):
            val = np.asarray(self(x, y), dtype=float)
            if val.ndim == 0:
                return float(val.reshape(()))
            return float(val.reshape(-1, self.dim)[0, int(idx)])

        return _wrapped


def _boundary_dof_coords(dh: DofHandler, field: str, tag: str) -> np.ndarray:
    dh._ensure_dof_coords()
    coords: list[np.ndarray] = []
    seen: set[int] = set()
    for eid in dh.mixed_element.mesh.edge_bitset(tag).to_indices():
        edge = dh.mixed_element.mesh.edge(int(eid))
        if edge.right is not None:
            continue
        try:
            gdofs = dh.edge_dofs(field, int(eid))
        except Exception:
            continue
        for gdof in gdofs:
            gd = int(gdof)
            if gd in seen:
                continue
            seen.add(gd)
            coords.append(np.asarray(dh._dof_coords[gd], dtype=float))
    if not coords:
        return np.empty((0, 2), dtype=float)
    return np.vstack(coords)


def _create_snapshots_from_contours(
    *,
    out_dir: Path,
    initial_points_m: np.ndarray,
    final_points_m: np.ndarray,
    final_time: float,
) -> None:
    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    rows0 = [
        {"contour_id": 0, "point_id": i, "x_m": float(pt[0]), "y_m": float(pt[1])}
        for i, pt in enumerate(np.asarray(initial_points_m, dtype=float))
    ]
    rows1 = [
        {"contour_id": 0, "point_id": i, "x_m": float(pt[0]), "y_m": float(pt[1])}
        for i, pt in enumerate(np.asarray(final_points_m, dtype=float))
    ]
    for time_s, rows in ((0.0, rows0), (float(final_time), rows1)):
        path = snap_dir / f"snapshot_step{int(round(time_s * 1000)):04d}_t{float(time_s):06.3f}_alpha05.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["contour_id", "point_id", "x_m", "y_m"])
            writer.writeheader()
            writer.writerows(rows)


def _write_timeseries(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "time_s",
                "ramp",
                "coupling_iter",
                "disp_l2",
                "disp_max_um",
                "darcy_ifc_max_um_s",
                "fluid_ifc_p_mean_pa",
                "fluid_ifc_sigma_n_mean_pa",
                "fluid_ifc_sigma_n_max_pa",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_gmsh_mesh(
    *,
    contour_m: np.ndarray,
    mesh_path: Path,
    length_m: float,
    height_m: float,
    mesh_size_m: float,
) -> Path:
    if gmsh is None:
        raise RuntimeError("gmsh is required to build the Christan ALE mesh.")

    chain = _polygon_interface_chain(contour_m)
    start = chain[0]
    end = chain[-1]

    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.model.add("christan_ale_fpi")
        occ = gmsh.model.occ

        def add_point(x: float, y: float, size: float | None = None) -> int:
            return occ.addPoint(float(x), float(y), 0.0, float(size if size is not None else mesh_size_m))

        p00 = add_point(0.0, 0.0)
        p0H = add_point(0.0, float(height_m))
        pLH = add_point(float(length_m), float(height_m))
        pL0 = add_point(float(length_m), 0.0)

        iface_pts = [add_point(float(pt[0]), float(pt[1]), size=0.5 * mesh_size_m) for pt in chain]

        l_bottom_left = occ.addLine(p00, iface_pts[0])
        l_bottom_right = occ.addLine(iface_pts[-1], pL0)
        l_outlet = occ.addLine(pL0, pLH)
        l_top = occ.addLine(pLH, p0H)
        l_inlet = occ.addLine(p0H, p00)

        iface_lines: list[int] = []
        for a, b in zip(iface_pts[:-1], iface_pts[1:]):
            iface_lines.append(occ.addLine(a, b))
        l_substrate = occ.addLine(iface_pts[0], iface_pts[-1])

        fluid_loop = occ.addCurveLoop([l_bottom_left, *iface_lines, l_bottom_right, l_outlet, l_top, l_inlet])
        solid_loop = occ.addCurveLoop([*[-line for line in iface_lines[::-1]], l_substrate])

        surf_fluid = occ.addPlaneSurface([fluid_loop])
        surf_solid = occ.addPlaneSurface([solid_loop])
        occ.synchronize()

        pg_fluid = gmsh.model.addPhysicalGroup(2, [surf_fluid])
        gmsh.model.setPhysicalName(2, pg_fluid, "fluid")
        pg_solid = gmsh.model.addPhysicalGroup(2, [surf_solid])
        gmsh.model.setPhysicalName(2, pg_solid, "solid")

        pg_inlet = gmsh.model.addPhysicalGroup(1, [l_inlet])
        gmsh.model.setPhysicalName(1, pg_inlet, "inlet")
        pg_outlet = gmsh.model.addPhysicalGroup(1, [l_outlet])
        gmsh.model.setPhysicalName(1, pg_outlet, "outlet")
        pg_walls = gmsh.model.addPhysicalGroup(1, [l_bottom_left, l_bottom_right, l_top])
        gmsh.model.setPhysicalName(1, pg_walls, "walls")
        pg_fsi = gmsh.model.addPhysicalGroup(1, iface_lines)
        gmsh.model.setPhysicalName(1, pg_fsi, "fsi")
        pg_sub = gmsh.model.addPhysicalGroup(1, [l_substrate])
        gmsh.model.setPhysicalName(1, pg_sub, "substrate")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(mesh_size_m))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(mesh_size_m))
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(mesh_path))
    finally:
        gmsh.finalize()
    return mesh_path


def _build_scalar_problem(mesh: Mesh, field: str, order: int) -> tuple[MixedElement, DofHandler, TrialFunction, TestFunction, Function]:
    me = MixedElement(mesh, field_specs={field: int(order)})
    dh = DofHandler(me, method="cg")
    trial = TrialFunction(name=f"d{field}", field_name=field, dof_handler=dh)
    test = TestFunction(name=f"{field}_test", field_name=field, dof_handler=dh)
    fn = Function(name=f"{field}_k", field_name=field, dof_handler=dh)
    fn.nodal_values.fill(0.0)
    return me, dh, trial, test, fn


def _build_vector_problem(
    mesh: Mesh,
    *,
    prefix: str,
    order: int,
) -> tuple[MixedElement, DofHandler, VectorTrialFunction, VectorTestFunction, VectorFunction]:
    field_names = [f"{prefix}x", f"{prefix}y"]
    me = MixedElement(mesh, field_specs={field_names[0]: int(order), field_names[1]: int(order)})
    dh = DofHandler(me, method="cg")
    space = FunctionSpace(name=f"{prefix.upper()}_space", field_names=field_names, dim=1)
    trial = VectorTrialFunction(space=space, dof_handler=dh)
    test = VectorTestFunction(space=space, dof_handler=dh)
    fn = VectorFunction(name=f"{prefix}_k", field_names=field_names, dof_handler=dh)
    fn.nodal_values.fill(0.0)
    return me, dh, trial, test, fn


def _scalar_lookup_from_field(dh: DofHandler, f_scalar: Function) -> CoordinateLookup:
    dh._ensure_dof_coords()
    ids = np.asarray(dh.get_field_slice(f_scalar.field_name), dtype=int)
    coords = np.asarray(dh._dof_coords[ids], dtype=float)
    vals = np.asarray(f_scalar.get_nodal_values(ids), dtype=float).reshape(-1, 1)
    return CoordinateLookup(coords, vals, dim=1)


def _vector_lookup_from_field(dh: DofHandler, f_vec: VectorFunction) -> CoordinateLookup:
    dh._ensure_dof_coords()
    x_ids = np.asarray(dh.get_field_slice(f_vec.components[0].field_name), dtype=int)
    y_ids = np.asarray(dh.get_field_slice(f_vec.components[1].field_name), dtype=int)
    coords = np.asarray(dh._dof_coords[x_ids], dtype=float)
    vals = np.column_stack(
        [
            np.asarray(f_vec.components[0].get_nodal_values(x_ids), dtype=float),
            np.asarray(f_vec.components[1].get_nodal_values(y_ids), dtype=float),
        ]
    )
    return CoordinateLookup(coords, vals, dim=2)


def _solve_linear(
    *,
    eq: Equation,
    dh: DofHandler,
    bcs: list[BoundaryCondition],
    quad_order: int,
    backend: str,
    functions: list[Function | VectorFunction],
) -> None:
    from scipy.sparse.linalg import spsolve

    K, F = assemble_form(eq, dof_handler=dh, bcs=bcs, quad_order=quad_order, backend=backend)
    if hasattr(K, "tocsr"):
        sol = spsolve(K.tocsr(), F)
    else:
        sol = np.linalg.solve(np.asarray(K, dtype=float), np.asarray(F, dtype=float))
    for f in functions:
        f.nodal_values.fill(0.0)
    dh.add_to_functions(np.asarray(sol, dtype=float), functions)
    dh.apply_bcs(bcs, *functions)


def _find_element_containing_point(mesh: Mesh, point: np.ndarray, *, boundary_tag: str | None = None) -> int:
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
            if boundary_tag is None:
                return int(elem.id)
            has_tag = False
            for eid in getattr(elem, "edges", []):
                edge = mesh.edge(int(eid))
                if edge.right is None and getattr(edge, "tag", "") == boundary_tag:
                    has_tag = True
                    break
            if has_tag:
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


def _sample_boundary_normals(mesh: Mesh, tag: str, coords: np.ndarray) -> np.ndarray:
    seg_mid = []
    seg_n = []
    for eid in mesh.edge_bitset(tag).to_indices():
        edge = mesh.edge(int(eid))
        if edge.right is not None:
            continue
        pts = mesh.nodes_x_y_pos[list(edge.nodes)]
        seg_mid.append(np.mean(pts, axis=0))
        seg_n.append(np.asarray(edge.normal, dtype=float))
    if not seg_mid:
        raise ValueError(f"Boundary tag {tag!r} has no boundary edges.")
    mids = np.asarray(seg_mid, dtype=float)
    normals = np.asarray(seg_n, dtype=float)
    out = np.empty((len(coords), 2), dtype=float)
    for i, xy in enumerate(np.asarray(coords, dtype=float)):
        dist2 = np.sum((mids - xy[None, :]) ** 2, axis=1)
        n = normals[int(np.argmin(dist2))]
        nn = np.linalg.norm(n)
        out[i, :] = n / max(nn, 1.0e-14)
    return out


def _fluid_interface_samples(
    *,
    dh: DofHandler,
    mesh: Mesh,
    u: VectorFunction,
    p: Function,
    d_mesh: VectorFunction,
    iface_coords: np.ndarray,
    mu_f: float,
) -> tuple[CoordinateLookup, CoordinateLookup]:
    normals = _sample_boundary_normals(mesh, "fsi", iface_coords)
    p_vals = np.empty((iface_coords.shape[0], 1), dtype=float)
    tr_vals = np.empty((iface_coords.shape[0], 2), dtype=float)
    for i, xy in enumerate(np.asarray(iface_coords, dtype=float)):
        p_val, _ = _eval_scalar_with_grad(dh, mesh, p, tuple(xy))
        _, grad_u = _eval_vector_with_grad(dh, mesh, u, tuple(xy))
        _, grad_m = _eval_vector_with_grad(dh, mesh, d_mesh, tuple(xy))
        F = np.eye(2, dtype=float) + grad_m
        Finv = np.linalg.inv(F)
        sigma = -float(p_val) * np.eye(2) + float(mu_f) * (grad_u @ Finv + Finv.T @ grad_u.T)
        n_solid = -normals[i]
        p_vals[i, 0] = float(p_val)
        tr_vals[i, :] = sigma @ n_solid
    return CoordinateLookup(iface_coords, p_vals, dim=1), CoordinateLookup(iface_coords, tr_vals, dim=2)


def _solid_interface_samples(
    *,
    dh_p: DofHandler,
    mesh_p: Mesh,
    p_b: Function,
    dh_d: DofHandler,
    mesh_d: Mesh,
    d_b: VectorFunction,
    iface_coords: np.ndarray,
    permeability_m2: float,
    mu_f: float,
) -> tuple[CoordinateLookup, CoordinateLookup]:
    disp_vals = np.empty((iface_coords.shape[0], 2), dtype=float)
    darcy_vals = np.empty((iface_coords.shape[0], 2), dtype=float)
    scale = -float(permeability_m2) / float(mu_f)
    for i, xy in enumerate(np.asarray(iface_coords, dtype=float)):
        _, grad_p = _eval_scalar_with_grad(dh_p, mesh_p, p_b, tuple(xy))
        disp, _ = _eval_vector_with_grad(dh_d, mesh_d, d_b, tuple(xy))
        disp_vals[i, :] = disp
        darcy_vals[i, :] = scale * np.asarray(grad_p, dtype=float)
    return CoordinateLookup(iface_coords, disp_vals, dim=2), CoordinateLookup(iface_coords, darcy_vals, dim=2)


def _write_contour_csv(path: Path, points_m: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts_mm = 1.0e3 * np.asarray(points_m, dtype=float)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["x_mm", "y_mm"])
        writer.writeheader()
        for x_mm, y_mm in pts_mm:
            writer.writerow({"x_mm": float(x_mm), "y_mm": float(y_mm)})


def _build_fluid_problem(mesh: Mesh, *, poly_order: int, pressure_order: int):
    me = MixedElement(
        mesh,
        field_specs={
            "ux": int(poly_order),
            "uy": int(poly_order),
            "p": int(pressure_order),
            "mx": int(poly_order),
            "my": int(poly_order),
        },
    )
    dh = DofHandler(me, method="cg")
    v_space = FunctionSpace("Vf", ["ux", "uy"], dim=1)
    m_space = FunctionSpace("Mf", ["mx", "my"], dim=1)
    du = VectorTrialFunction(space=v_space, dof_handler=dh)
    v = VectorTestFunction(space=v_space, dof_handler=dh)
    dp = TrialFunction(name="dp", field_name="p", dof_handler=dh)
    q = TestFunction(name="q", field_name="p", dof_handler=dh)
    dm = VectorTrialFunction(space=m_space, dof_handler=dh)
    z = VectorTestFunction(space=m_space, dof_handler=dh)
    u_k = VectorFunction("u_k", ["ux", "uy"], dof_handler=dh)
    u_prev = VectorFunction("u_prev", ["ux", "uy"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    p_prev = Function("p_prev", "p", dof_handler=dh)
    d_mesh = VectorFunction("d_mesh", ["mx", "my"], dof_handler=dh)
    d_prev = VectorFunction("d_prev", ["mx", "my"], dof_handler=dh)
    for f in (u_k, u_prev, d_mesh, d_prev):
        f.nodal_values.fill(0.0)
    for f in (p_k, p_prev):
        f.nodal_values.fill(0.0)
    return {
        "me": me,
        "dh": dh,
        "du": du,
        "v": v,
        "dp": dp,
        "q": q,
        "dm": dm,
        "z": z,
        "u_k": u_k,
        "u_prev": u_prev,
        "p_k": p_k,
        "p_prev": p_prev,
        "d_mesh": d_mesh,
        "d_prev": d_prev,
    }


def _transfer_vector_field(
    *,
    target_dh: DofHandler,
    target_vec: VectorFunction,
    source_lookup: CoordinateLookup,
) -> None:
    target_dh._ensure_dof_coords()
    for idx, comp in enumerate(target_vec.components):
        ids = np.asarray(target_dh.get_field_slice(comp.field_name), dtype=int)
        xy = np.asarray(target_dh._dof_coords[ids], dtype=float)
        vals = np.asarray(source_lookup(xy[:, 0], xy[:, 1]), dtype=float)
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        comp.set_nodal_values(ids, vals[:, idx])


def _build_mesh_extension_problem(mesh: Mesh, *, poly_order: int):
    me, dh, dm, z, m_k = _build_vector_problem(mesh, prefix="m", order=poly_order)
    return {"me": me, "dh": dh, "dm": dm, "z": z, "m_k": m_k}


def _build_solid_pressure_problem(mesh: Mesh, *, pressure_order: int):
    me, dh, dp, q, p_k = _build_scalar_problem(mesh, field="p", order=pressure_order)
    return {"me": me, "dh": dh, "dp": dp, "q": q, "p_k": p_k}


def _build_solid_displacement_problem(mesh: Mesh, *, poly_order: int):
    me, dh, dd, w, d_k = _build_vector_problem(mesh, prefix="d", order=poly_order)
    return {"me": me, "dh": dh, "dd": dd, "w": w, "d_k": d_k}


def _fluid_residual_and_jacobian(
    *,
    prob: dict[str, object],
    rho_f: float,
    mu_f: float,
    dt: float,
    iface_velocity: CoordinateLookup,
    inlet_lookup: Callable[[float, float], float],
    quad_order: int,
):
    dh: DofHandler = prob["dh"]
    u_k: VectorFunction = prob["u_k"]
    u_prev: VectorFunction = prob["u_prev"]
    p_k: Function = prob["p_k"]
    d_mesh: VectorFunction = prob["d_mesh"]
    d_prev: VectorFunction = prob["d_prev"]
    du = prob["du"]
    v = prob["v"]
    dp = prob["dp"]
    q = prob["q"]

    dx_f = dx(metadata={"q": int(quad_order)})
    F = Identity(2) + grad(d_mesh)
    Finv = inv(F)
    J = det(F)
    w_mesh = (d_mesh - d_prev) / Constant(float(dt))
    pI = p_k * Identity(2)
    sigma = -pI + Constant(float(mu_f)) * (dot(grad(u_k), Finv) + dot(Finv.T, grad(u_k).T))

    conv = Constant(float(rho_f)) * J * dot(dot(grad(u_k), Finv), (u_k - w_mesh))
    r = dot(conv, v) * dx_f
    r += inner(J * dot(sigma, Finv.T), grad(v)) * dx_f
    r += (inner(cof(F), grad(u_k)) * q) * dx_f

    a = Constant(float(rho_f)) * J * dot(dot(dot(grad(du), Finv), (u_k - w_mesh)), v) * dx_f
    a += Constant(float(rho_f)) * J * dot(dot(dot(grad(u_k), Finv), du), v) * dx_f
    sigma_du = -dp * Identity(2) + Constant(float(mu_f)) * (dot(grad(du), Finv) + dot(Finv.T, grad(du).T))
    a += inner(J * dot(sigma_du, Finv.T), grad(v)) * dx_f
    a += (inner(cof(F), grad(du)) * q) * dx_f

    zero = lambda x, y, t=0.0: 0.0
    bcs = [
        BoundaryCondition("ux", "dirichlet", "inlet", inlet_lookup),
        BoundaryCondition("uy", "dirichlet", "inlet", zero),
        BoundaryCondition("ux", "dirichlet", "walls", zero),
        BoundaryCondition("uy", "dirichlet", "walls", zero),
        BoundaryCondition("ux", "dirichlet", "fsi", iface_velocity.component(0)),
        BoundaryCondition("uy", "dirichlet", "fsi", iface_velocity.component(1)),
        BoundaryCondition("p", "dirichlet", "outlet", zero),
    ]
    bcs_homog = [
        BoundaryCondition("ux", "dirichlet", "inlet", zero),
        BoundaryCondition("uy", "dirichlet", "inlet", zero),
        BoundaryCondition("ux", "dirichlet", "walls", zero),
        BoundaryCondition("uy", "dirichlet", "walls", zero),
        BoundaryCondition("ux", "dirichlet", "fsi", zero),
        BoundaryCondition("uy", "dirichlet", "fsi", zero),
        BoundaryCondition("p", "dirichlet", "outlet", zero),
    ]
    return r, a, bcs, bcs_homog


def _run_reference(args: argparse.Namespace) -> dict[str, object]:
    out_dir = (ROOT / str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = out_dir / "mesh" / "christan_ale_fpi.msh"

    initial_contour_m = _read_contour_m(INITIAL_CONTOUR_CSV)
    if (not mesh_path.exists()) or bool(args.rebuild_mesh):
        _build_gmsh_mesh(
            contour_m=initial_contour_m,
            mesh_path=mesh_path,
            length_m=float(args.length_mm) * 1.0e-3,
            height_m=float(args.height_mm) * 1.0e-3,
            mesh_size_m=float(args.mesh_size_um) * 1.0e-6,
        )

    mesh_f = mesh_from_gmsh(mesh_path, surface_physical_names=["fluid"], apply_boundary_tags=True)
    mesh_s = mesh_from_gmsh(mesh_path, surface_physical_names=["solid"], apply_boundary_tags=True)

    poly_order = int(args.poly_order)
    pressure_order = int(args.pressure_order) if args.pressure_order is not None else max(1, poly_order - 1)
    quad_order = int(args.quad_order) if args.quad_order is not None else (2 * poly_order + 4)

    fluid = _build_fluid_problem(mesh_f, poly_order=poly_order, pressure_order=pressure_order)
    mesh_ext = _build_mesh_extension_problem(mesh_f, poly_order=poly_order)
    solid_p = _build_solid_pressure_problem(mesh_s, pressure_order=pressure_order)
    solid_d = _build_solid_displacement_problem(mesh_s, poly_order=poly_order)
    _log(
        args,
        "[setup] dofs "
        f"fluid={fluid['dh'].total_dofs} mesh_ext={mesh_ext['dh'].total_dofs} "
        f"solid_p={solid_p['dh'].total_dofs} solid_d={solid_d['dh'].total_dofs}",
    )

    fluid_iface_coords = _boundary_dof_coords(fluid["dh"], "ux", "fsi")
    solid_iface_p_coords = _boundary_dof_coords(solid_p["dh"], "p", "fsi")
    solid_iface_d_coords = _boundary_dof_coords(solid_d["dh"], "dx", "fsi")
    if fluid_iface_coords.size == 0 or solid_iface_p_coords.size == 0 or solid_iface_d_coords.size == 0:
        raise RuntimeError("Failed to extract interface boundary DOF coordinates from one or more subproblems.")

    zero_scalar = CoordinateLookup(solid_iface_p_coords, np.zeros((solid_iface_p_coords.shape[0], 1), dtype=float), dim=1)
    zero_vector_fluid = CoordinateLookup(fluid_iface_coords, np.zeros((fluid_iface_coords.shape[0], 2), dtype=float), dim=2)

    dt_val = float(args.dt)
    n_steps = int(max(1, math.ceil(float(args.t_final) / dt_val)))
    mu_f = float(args.mu_f)
    rho_f = float(args.rho_f)
    permeability = float(args.kappa_m2)
    alpha_biot = float(args.alpha_biot)
    E = float(args.E)
    nu = float(args.nu)
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu_s = E / (2.0 * (1.0 + nu))

    time_rows: list[dict[str, object]] = []

    fluid_p_lookup = zero_scalar
    fluid_t_lookup = CoordinateLookup(solid_iface_d_coords, np.zeros((solid_iface_d_coords.shape[0], 2), dtype=float), dim=2)
    solid_d_lookup = zero_vector_fluid
    solid_u_lookup = zero_vector_fluid

    for step in range(1, n_steps + 1):
        t_now = min(float(args.t_final), step * dt_val)
        ramp = _cosine_ramp_value(t_now, float(args.t_ramp))
        _log(args, f"[time] step={step}/{n_steps} t={t_now:.3f}s ramp={ramp:.6f}")

        def inlet_profile(x: float, y: float) -> float:
            yy = float(y)
            H = float(args.height_mm) * 1.0e-3
            return float(ramp * 6.0 * float(args.u_avg) * yy * (H - yy) / max(H * H, 1.0e-20))

        for coupling_iter in range(1, int(args.coupling_iters) + 1):
            _log(args, f"[coupling] iter={coupling_iter}/{int(args.coupling_iters)}")
            # Darcy pressure inside the biofilm: harmonic pressure with Dirichlet p_B = p_F on Γ_FSI.
            dp = solid_p["dp"]
            q = solid_p["q"]
            a_p = inner(grad(dp), grad(q)) * dx(metadata={"q": int(quad_order)})
            l_p = Constant(0.0) * q * dx(metadata={"q": int(quad_order)})
            bcs_p = [BoundaryCondition("p", "dirichlet", "fsi", fluid_p_lookup.component(0))]
            _timed(
                args,
                "solid pressure",
                lambda: _solve_linear(
                    eq=Equation(a_p, l_p),
                    dh=solid_p["dh"],
                    bcs=bcs_p,
                    quad_order=quad_order,
                    backend=str(args.backend),
                    functions=[solid_p["p_k"]],
                ),
            )

            # Linear poroelastic displacement.
            dd = solid_d["dd"]
            w = solid_d["w"]
            solid_p_lookup = _scalar_lookup_from_field(solid_p["dh"], solid_p["p_k"])
            pore_coeff = Analytic(lambda x, y: solid_p_lookup(x, y), dim=0)
            traction = Analytic(lambda x, y: fluid_t_lookup(x, y), dim=1)
            dx_s = dx(metadata={"q": int(quad_order)})
            ds_fsi_s = dS(defined_on=mesh_s.edge_bitset("fsi"), metadata={"q": int(quad_order)})
            a_d = (
                Constant(2.0 * mu_s) * inner(_symgrad(dd), _symgrad(w))
                + Constant(lam) * div(dd) * div(w)
            ) * dx_s
            l_d = (Constant(alpha_biot) * pore_coeff * div(w)) * dx_s
            l_d -= dot(traction, w) * ds_fsi_s
            bcs_d = [
                BoundaryCondition("dx", "dirichlet", "substrate", lambda x, y, t=0.0: 0.0),
                BoundaryCondition("dy", "dirichlet", "substrate", lambda x, y, t=0.0: 0.0),
            ]
            _timed(
                args,
                "solid displacement",
                lambda: _solve_linear(
                    eq=Equation(a_d, l_d),
                    dh=solid_d["dh"],
                    bcs=bcs_d,
                    quad_order=quad_order,
                    backend=str(args.backend),
                    functions=[solid_d["d_k"]],
                ),
            )

            # Harmonic extension of the displacement into the fluid mesh.
            solid_d_lookup, solid_u_lookup = _solid_interface_samples(
                dh_p=solid_p["dh"],
                mesh_p=mesh_s,
                p_b=solid_p["p_k"],
                dh_d=solid_d["dh"],
                mesh_d=mesh_s,
                d_b=solid_d["d_k"],
                iface_coords=fluid_iface_coords,
                permeability_m2=permeability,
                mu_f=mu_f,
            )
            dm = mesh_ext["dm"]
            z = mesh_ext["z"]
            a_m = inner(grad(dm), grad(z)) * dx(metadata={"q": int(quad_order)})
            l_m = Constant(0.0) * dot(z, Constant(np.zeros(2), dim=1)) * dx(metadata={"q": int(quad_order)})
            bcs_m = [
                BoundaryCondition("mx", "dirichlet", "inlet", lambda x, y, t=0.0: 0.0),
                BoundaryCondition("my", "dirichlet", "inlet", lambda x, y, t=0.0: 0.0),
                BoundaryCondition("mx", "dirichlet", "outlet", lambda x, y, t=0.0: 0.0),
                BoundaryCondition("my", "dirichlet", "outlet", lambda x, y, t=0.0: 0.0),
                BoundaryCondition("mx", "dirichlet", "walls", lambda x, y, t=0.0: 0.0),
                BoundaryCondition("my", "dirichlet", "walls", lambda x, y, t=0.0: 0.0),
                BoundaryCondition("mx", "dirichlet", "fsi", solid_d_lookup.component(0)),
                BoundaryCondition("my", "dirichlet", "fsi", solid_d_lookup.component(1)),
            ]
            _timed(
                args,
                "mesh extension",
                lambda: _solve_linear(
                    eq=Equation(a_m, l_m),
                    dh=mesh_ext["dh"],
                    bcs=bcs_m,
                    quad_order=quad_order,
                    backend=str(args.backend),
                    functions=[mesh_ext["m_k"]],
                ),
            )

            fluid["d_prev"].nodal_values[:] = fluid["d_mesh"].nodal_values[:]
            mesh_lookup = _vector_lookup_from_field(mesh_ext["dh"], mesh_ext["m_k"])
            _transfer_vector_field(target_dh=fluid["dh"], target_vec=fluid["d_mesh"], source_lookup=mesh_lookup)

            # Nonlinear stationary ALE Navier-Stokes on the deformed fluid mesh.
            r_f, a_f, bcs_f, bcs_f_h = _fluid_residual_and_jacobian(
                prob=fluid,
                rho_f=rho_f,
                mu_f=mu_f,
                dt=dt_val,
                iface_velocity=solid_u_lookup,
                inlet_lookup=inlet_profile,
                quad_order=quad_order,
            )
            solver = NewtonSolver(
                residual_form=r_f,
                jacobian_form=a_f,
                dof_handler=fluid["dh"],
                mixed_element=fluid["me"],
                bcs=bcs_f,
                bcs_homog=bcs_f_h,
                newton_params=NewtonParameters(newton_tol=float(args.newton_tol), max_newton_iter=int(args.max_newton_iter)),
                lin_params=LinearSolverParameters(backend=str(args.linear_backend)),
                quad_order=quad_order,
                backend=str(args.backend),
            )
            fluid["u_prev"].nodal_values[:] = fluid["u_k"].nodal_values[:]
            fluid["p_prev"].nodal_values[:] = fluid["p_k"].nodal_values[:]
            _timed(
                args,
                "fluid Newton",
                lambda: solver.solve_time_interval(
                    functions=[fluid["u_k"], fluid["p_k"]],
                    prev_functions=[fluid["u_prev"], fluid["p_prev"]],
                    aux_functions={
                        "d_mesh": fluid["d_mesh"],
                        "d_prev": fluid["d_prev"],
                    },
                    time_params=TimeStepperParameters(dt=dt_val, final_time=dt_val, max_steps=1),
                ),
            )

            fluid_p_lookup, _ = _fluid_interface_samples(
                dh=fluid["dh"],
                mesh=mesh_f,
                u=fluid["u_k"],
                p=fluid["p_k"],
                d_mesh=fluid["d_mesh"],
                iface_coords=solid_iface_p_coords,
                mu_f=mu_f,
            )
            _, fluid_t_lookup = _fluid_interface_samples(
                dh=fluid["dh"],
                mesh=mesh_f,
                u=fluid["u_k"],
                p=fluid["p_k"],
                d_mesh=fluid["d_mesh"],
                iface_coords=solid_iface_d_coords,
                mu_f=mu_f,
            )

            disp_vals = np.asarray(solid_d_lookup.values, dtype=float)
            darcy_vals = np.asarray(solid_u_lookup.values, dtype=float)
            p_vals = np.asarray(fluid_p_lookup.values[:, 0], dtype=float)
            traction_vals = np.asarray(fluid_t_lookup.values, dtype=float)
            time_rows.append(
                {
                    "step": int(step),
                    "time_s": float(t_now),
                    "ramp": float(ramp),
                    "coupling_iter": int(coupling_iter),
                    "disp_l2": float(np.sqrt(np.mean(np.sum(disp_vals * disp_vals, axis=1)))) if disp_vals.size else 0.0,
                    "disp_max_um": float(1.0e6 * np.max(np.linalg.norm(disp_vals, axis=1))) if disp_vals.size else 0.0,
                    "darcy_ifc_max_um_s": float(1.0e6 * np.max(np.linalg.norm(darcy_vals, axis=1))) if darcy_vals.size else 0.0,
                    "fluid_ifc_p_mean_pa": float(np.mean(p_vals)) if p_vals.size else 0.0,
                    "fluid_ifc_sigma_n_mean_pa": float(np.mean(np.sum(traction_vals * _sample_boundary_normals(mesh_s, "fsi", solid_iface_d_coords), axis=1))) if traction_vals.size else 0.0,
                    "fluid_ifc_sigma_n_max_pa": float(np.max(np.abs(np.sum(traction_vals * _sample_boundary_normals(mesh_s, "fsi", solid_iface_d_coords), axis=1)))) if traction_vals.size else 0.0,
                }
            )

    initial_points_m = np.asarray(initial_contour_m, dtype=float)
    final_points_m = np.empty_like(initial_points_m)
    for i, pt in enumerate(initial_points_m):
        disp, _ = _eval_vector_with_grad(solid_d["dh"], mesh_s, solid_d["d_k"], (float(pt[0]), float(pt[1])))
        final_points_m[i, :] = pt + disp

    _create_snapshots_from_contours(
        out_dir=out_dir,
        initial_points_m=initial_points_m,
        final_points_m=final_points_m,
        final_time=float(args.t_final),
    )
    _write_timeseries(out_dir / "timeseries.csv", time_rows)
    _write_contour_csv(out_dir / "deformed_contour_mm.csv", final_points_m)

    comparison = compare_case(
        out_dir=out_dir,
        target_time=float(args.t_final),
        initial_time=0.0,
        geometry_dir=Path(str(args.geometry_dir)),
        y_levels_um=[int(v) for v in str(args.y_levels_um).split(",") if str(v).strip()],
    )
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

    summary = {
        "out_dir": str(out_dir),
        "mesh_path": str(mesh_path),
        "comparison": comparison,
        "time_rows": len(time_rows),
        "parameters": {
            "E": float(args.E),
            "nu": float(args.nu),
            "alpha_biot": float(args.alpha_biot),
            "kappa_m2": float(args.kappa_m2),
            "rho_f": float(args.rho_f),
            "mu_f": float(args.mu_f),
            "u_avg": float(args.u_avg),
            "t_ramp": float(args.t_ramp),
            "t_final": float(args.t_final),
            "dt": float(args.dt),
            "mesh_size_um": float(args.mesh_size_um),
            "coupling_iters": int(args.coupling_iters),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=str, default="examples/biofilms/results/christan_ale_reference")
    ap.add_argument("--geometry-dir", type=str, default=str(DEFAULT_GEOMETRY_DIR))
    ap.add_argument("--length-mm", type=float, default=3.0)
    ap.add_argument("--height-mm", type=float, default=1.0)
    ap.add_argument("--mesh-size-um", type=float, default=25.0)
    ap.add_argument("--poly-order", type=int, default=2)
    ap.add_argument("--pressure-order", type=int, default=None)
    ap.add_argument("--quad-order", type=int, default=None)
    ap.add_argument("--backend", type=str, default="cpp", choices=("cpp", "jit", "python"))
    ap.add_argument("--linear-backend", type=str, default="scipy")
    ap.add_argument("--rebuild-mesh", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--rho-f", type=float, default=1000.0)
    ap.add_argument("--mu-f", type=float, default=1.0e-3)
    ap.add_argument("--u-avg", type=float, default=0.06825)
    ap.add_argument("--kappa-m2", type=float, default=1.0e-15)
    ap.add_argument("--alpha-biot", type=float, default=1.0)
    ap.add_argument("--E", type=float, default=80.0)
    ap.add_argument("--nu", type=float, default=0.4)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--t-final", type=float, default=1.5)
    ap.add_argument("--t-ramp", type=float, default=1.0)
    ap.add_argument("--coupling-iters", type=int, default=2)
    ap.add_argument("--newton-tol", type=float, default=1.0e-8)
    ap.add_argument("--max-newton-iter", type=int, default=25)
    ap.add_argument("--y-levels-um", type=str, default="150,250,350")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    summary = _run_reference(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
