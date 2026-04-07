from __future__ import annotations

import argparse
import csv
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem import transform
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.mor.interface import build_restriction_matrix
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    CellDiameter,
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

from examples.NIRB.common import dump_json
from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.example2_problem import build_conforming_mesh
from examples.NIRB.example2_problem import _neo_hookean_delta_pk1 as neo_hookean_delta_pk1
from examples.NIRB.example2_problem import _neo_hookean_pk1 as neo_hookean_pk1


def _coord_key(x: float, y: float, ndigits: int = 12) -> tuple[float, float]:
    return (round(float(x), ndigits), round(float(y), ndigits))


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


@dataclass
class CoordinateLookup:
    coords: np.ndarray
    values: np.ndarray
    dim: int

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must have shape (n, 2)")
        if self.dim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[0] != coords.shape[0] or values.shape[1] != int(self.dim):
            raise ValueError("values shape does not match coords / dim")
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


def _solve_linear(
    *,
    eq: Equation,
    dh: DofHandler,
    bcs: list[BoundaryCondition],
    quad_order: int,
    backend: str,
    linear_backend: str,
    functions: list[Function | VectorFunction],
) -> None:
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    K, F = assemble_form(eq, dof_handler=dh, bcs=bcs, quad_order=quad_order, backend=backend)
    lin_backend = str(linear_backend).lower()
    if lin_backend == "petsc":
        try:
            from petsc4py import PETSc  # type: ignore
        except Exception as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("petsc4py is required for linear_backend='petsc'.") from exc

        A = K.tocsr() if hasattr(K, "tocsr") else sp.csr_matrix(np.asarray(K, dtype=float))
        rhs = np.asarray(F, dtype=float).ravel()
        mat = PETSc.Mat().createAIJ(
            size=A.shape,
            csr=(
                A.indptr.astype(PETSc.IntType, copy=False),
                A.indices.astype(PETSc.IntType, copy=False),
                np.asarray(A.data, dtype=float),
            ),
            comm=PETSc.COMM_SELF,
        )
        mat.assemblyBegin()
        mat.assemblyEnd()
        b = PETSc.Vec().createSeq(A.shape[0], comm=PETSc.COMM_SELF)
        x = PETSc.Vec().createSeq(A.shape[0], comm=PETSc.COMM_SELF)
        idx = np.arange(A.shape[0], dtype=PETSc.IntType)
        b.setValues(idx, rhs, addv=PETSc.InsertMode.INSERT_VALUES)
        b.assemblyBegin()
        b.assemblyEnd()
        ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp.setOperators(mat)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solve(b, x)
        sol = x.getArray(readonly=True).copy()
    elif hasattr(K, "tocsr"):
        sol = spsolve(K.tocsr(), F)
    else:
        sol = np.linalg.solve(np.asarray(K, dtype=float), np.asarray(F, dtype=float))
    for function in functions:
        function.nodal_values.fill(0.0)
    dh.add_to_functions(np.asarray(sol, dtype=float), functions)
    dh.apply_bcs(bcs, *functions)


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
    space = FunctionSpace(name=f"{prefix.upper()}Space", field_names=field_names, dim=1)
    trial = VectorTrialFunction(space=space, dof_handler=dh)
    test = VectorTestFunction(space=space, dof_handler=dh)
    fn = VectorFunction(name=f"{prefix}_k", field_names=field_names, dof_handler=dh)
    fn.nodal_values.fill(0.0)
    return me, dh, trial, test, fn


def _build_fluid_problem(mesh: Mesh, *, poly_order: int, pressure_order: int) -> dict[str, object]:
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
    v_space = FunctionSpace("FluidVelocity", ["ux", "uy"], dim=1)
    m_space = FunctionSpace("FluidMesh", ["mx", "my"], dim=1)
    du = VectorTrialFunction(space=v_space, dof_handler=dh)
    v = VectorTestFunction(space=v_space, dof_handler=dh)
    dp = TrialFunction(name="dp", field_name="p", dof_handler=dh)
    q = TestFunction(name="q", field_name="p", dof_handler=dh)
    dm = VectorTrialFunction(space=m_space, dof_handler=dh)
    z = VectorTestFunction(space=m_space, dof_handler=dh)
    u_k = VectorFunction("u_k", ["ux", "uy"], dof_handler=dh)
    u_prev = VectorFunction("u_prev", ["ux", "uy"], dof_handler=dh)
    a_prev = VectorFunction("a_prev", ["ux", "uy"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    p_prev = Function("p_prev", "p", dof_handler=dh)
    d_mesh = VectorFunction("d_mesh", ["mx", "my"], dof_handler=dh)
    d_prev = VectorFunction("d_prev", ["mx", "my"], dof_handler=dh)
    for function in (u_k, u_prev, a_prev, d_mesh, d_prev):
        function.nodal_values.fill(0.0)
    for function in (p_k, p_prev):
        function.nodal_values.fill(0.0)
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
        "a_prev": a_prev,
        "p_k": p_k,
        "p_prev": p_prev,
        "d_mesh": d_mesh,
        "d_prev": d_prev,
    }


def _build_mesh_extension_problem(mesh: Mesh, *, poly_order: int) -> dict[str, object]:
    me, dh, dm, z, m_k = _build_vector_problem(mesh, prefix="m", order=poly_order)
    return {"me": me, "dh": dh, "dm": dm, "z": z, "m_k": m_k}


def _build_solid_problem(mesh: Mesh, *, poly_order: int) -> dict[str, object]:
    me, dh, dd, w, d_k = _build_vector_problem(mesh, prefix="d", order=poly_order)
    d_prev = VectorFunction("d_prev", ["dx", "dy"], dof_handler=dh)
    d_prev.nodal_values.fill(0.0)
    return {
        "me": me,
        "dh": dh,
        "dd": dd,
        "w": w,
        "d_k": d_k,
        "d_prev": d_prev,
    }


def _boundary_field_data(dh: DofHandler, field: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    dh._ensure_dof_coords()
    mesh = dh.mixed_element.mesh
    boundary_points: list[np.ndarray] = []
    boundary_segments: list[tuple[np.ndarray, np.ndarray]] = []
    for eid in mesh.edge_bitset(tag).to_indices():
        edge = mesh.edge(int(eid))
        if edge.right is not None:
            continue
        node_ids = list(getattr(edge, "all_nodes", None) or edge.nodes)
        pts = np.asarray(mesh.nodes_x_y_pos[node_ids], dtype=float)
        for point in pts:
            boundary_points.append(np.asarray(point, dtype=float))
        if pts.shape[0] >= 2:
            boundary_segments.append((pts[0], pts[-1]))
    if not boundary_points:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=int)

    points_arr = np.asarray(boundary_points, dtype=float)
    field_ids = np.asarray(dh.get_field_slice(field), dtype=int)
    field_coords = np.asarray(dh._dof_coords[field_ids], dtype=float)
    span = np.ptp(np.asarray(mesh.nodes_x_y_pos, dtype=float), axis=0)
    span_max = float(np.max(span)) if span.size else 1.0
    tol_sq = (1.0e-8 * max(span_max, 1.0)) ** 2
    keep: list[int] = []
    for idx, xy in enumerate(field_coords):
        dist2_pts = np.sum((points_arr - xy[None, :]) ** 2, axis=1)
        on_boundary = bool(np.min(dist2_pts) <= tol_sq)
        if not on_boundary:
            for a, b in boundary_segments:
                ab = b - a
                denom = float(np.dot(ab, ab))
                if denom <= 1.0e-20:
                    continue
                t = float(np.clip(np.dot(xy - a, ab) / denom, 0.0, 1.0))
                proj = a + t * ab
                if float(np.dot(xy - proj, xy - proj)) <= tol_sq:
                    on_boundary = True
                    break
        if on_boundary:
            keep.append(idx)
    if not keep:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=int)
    return field_coords[np.asarray(keep, dtype=int)], field_ids[np.asarray(keep, dtype=int)]


def _vector_field_matrix(dh: DofHandler, vector: VectorFunction) -> tuple[np.ndarray, np.ndarray]:
    dh._ensure_dof_coords()
    x_ids = np.asarray(dh.get_field_slice(vector.components[0].field_name), dtype=int)
    y_ids = np.asarray(dh.get_field_slice(vector.components[1].field_name), dtype=int)
    coords = np.asarray(dh._dof_coords[x_ids], dtype=float)
    values = np.column_stack(
        [
            np.asarray(vector.components[0].get_nodal_values(x_ids), dtype=float),
            np.asarray(vector.components[1].get_nodal_values(y_ids), dtype=float),
        ]
    )
    return coords, values


def _vector_lookup_from_field(dh: DofHandler, vector: VectorFunction) -> CoordinateLookup:
    coords, values = _vector_field_matrix(dh, vector)
    return CoordinateLookup(coords, values, dim=2)


def _boundary_vector_snapshot(dh: DofHandler, vector: VectorFunction, tag: str) -> tuple[np.ndarray, np.ndarray]:
    coords_x, gdofs_x = _boundary_field_data(dh, vector.components[0].field_name, tag)
    coords_y, gdofs_y = _boundary_field_data(dh, vector.components[1].field_name, tag)
    if coords_x.shape != coords_y.shape:
        raise RuntimeError(f"Mismatched boundary DOF counts for tag {tag!r}: {coords_x.shape} vs {coords_y.shape}")
    if coords_x.size and not np.allclose(coords_x, coords_y):
        raise RuntimeError(f"Mismatched vector boundary coordinate ordering for tag {tag!r}")
    values = np.column_stack(
        [
            np.asarray(vector.components[0].get_nodal_values(gdofs_x), dtype=float),
            np.asarray(vector.components[1].get_nodal_values(gdofs_y), dtype=float),
        ]
    )
    return coords_x, values


def _flatten_vector_snapshot(dh: DofHandler, vector: VectorFunction) -> np.ndarray:
    _, values = _vector_field_matrix(dh, vector)
    return np.asarray(values, dtype=float).reshape(-1)


def _build_interface_restriction_matrix(dh: DofHandler, vector: VectorFunction, tag: str) -> np.ndarray:
    full_x_ids = np.asarray(dh.get_field_slice(vector.components[0].field_name), dtype=int)
    _, boundary_x_ids = _boundary_field_data(dh, vector.components[0].field_name, tag)
    positions = {int(gdof): idx for idx, gdof in enumerate(full_x_ids.tolist())}
    flat_indices: list[int] = []
    for gid in boundary_x_ids:
        pos = positions[int(gid)]
        flat_indices.extend([2 * pos, 2 * pos + 1])
    return build_restriction_matrix(flat_indices, 2 * full_x_ids.size)


def _build_interface_mass_matrix(mesh: Mesh, coords: np.ndarray, tag: str) -> np.ndarray:
    iface_coords = np.asarray(coords, dtype=float)
    if iface_coords.ndim != 2 or iface_coords.shape[1] != 2:
        raise ValueError("coords must have shape (n, 2)")
    if iface_coords.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)

    coord_to_idx = {_coord_key(x, y): i for i, (x, y) in enumerate(iface_coords)}
    mass = np.zeros((iface_coords.shape[0], iface_coords.shape[0]), dtype=float)
    for eid in mesh.edge_bitset(tag).to_indices():
        edge = mesh.edge(int(eid))
        if edge.right is not None:
            continue
        node_ids = list(getattr(edge, "all_nodes", None) or edge.nodes)
        if len(node_ids) != 2:
            raise NotImplementedError("Interface load conversion currently supports linear boundary edges only.")
        pts = np.asarray(mesh.nodes_x_y_pos[node_ids], dtype=float)
        ia = coord_to_idx.get(_coord_key(pts[0, 0], pts[0, 1]))
        ib = coord_to_idx.get(_coord_key(pts[1, 0], pts[1, 1]))
        if ia is None or ib is None:
            continue
        length = float(np.linalg.norm(pts[1] - pts[0]))
        local_mass = (length / 6.0) * np.asarray([[2.0, 1.0], [1.0, 2.0]], dtype=float)
        ids = np.asarray([ia, ib], dtype=int)
        mass[np.ix_(ids, ids)] += local_mass
    return mass


def _interface_load_from_traction(mass_matrix: np.ndarray, traction_values: np.ndarray) -> np.ndarray:
    traction = np.asarray(traction_values, dtype=float)
    if traction.ndim != 2 or traction.shape[1] != 2:
        raise ValueError("traction_values must have shape (n, 2)")
    if traction.shape[0] != mass_matrix.shape[0]:
        raise ValueError("mass_matrix / traction_values size mismatch")
    if traction.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    return np.column_stack([mass_matrix @ traction[:, comp] for comp in range(2)])


def _interface_traction_from_load(
    mass_matrix: np.ndarray,
    load_values: np.ndarray,
    *,
    regularization: float = 1.0e-12,
) -> np.ndarray:
    load = np.asarray(load_values, dtype=float)
    if load.ndim != 2 or load.shape[1] != 2:
        raise ValueError("load_values must have shape (n, 2)")
    if load.shape[0] != mass_matrix.shape[0]:
        raise ValueError("mass_matrix / load_values size mismatch")
    if load.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    reg = float(max(regularization, 0.0))
    operator = np.asarray(mass_matrix, dtype=float)
    if reg > 0.0:
        operator = operator + reg * np.eye(operator.shape[0], dtype=float)
    return np.column_stack([np.linalg.solve(operator, load[:, comp]) for comp in range(2)])


def _boundary_point_load_vector(
    dh: DofHandler,
    *,
    vector: VectorFunction,
    tag: str,
    values: np.ndarray,
) -> np.ndarray:
    coords_x, gdofs_x = _boundary_field_data(dh, vector.components[0].field_name, tag)
    coords_y, gdofs_y = _boundary_field_data(dh, vector.components[1].field_name, tag)
    if coords_x.shape != coords_y.shape:
        raise RuntimeError(f"Mismatched boundary DOF counts for tag {tag!r}: {coords_x.shape} vs {coords_y.shape}")
    if coords_x.size and not np.allclose(coords_x, coords_y):
        raise RuntimeError(f"Mismatched vector boundary coordinate ordering for tag {tag!r}")
    load_vals = np.asarray(values, dtype=float)
    if load_vals.shape != (coords_x.shape[0], 2):
        raise ValueError(f"Expected point-load values with shape {(coords_x.shape[0], 2)}, got {load_vals.shape}")
    rhs = np.zeros(dh.total_dofs, dtype=float)
    rhs[np.asarray(gdofs_x, dtype=int)] = load_vals[:, 0]
    rhs[np.asarray(gdofs_y, dtype=int)] = load_vals[:, 1]
    return rhs


def _boundary_vector_from_global_values(
    dh: DofHandler,
    *,
    vector: VectorFunction,
    tag: str,
    global_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coords_x, gdofs_x = _boundary_field_data(dh, vector.components[0].field_name, tag)
    coords_y, gdofs_y = _boundary_field_data(dh, vector.components[1].field_name, tag)
    if coords_x.shape != coords_y.shape:
        raise RuntimeError(f"Mismatched boundary DOF counts for tag {tag!r}: {coords_x.shape} vs {coords_y.shape}")
    if coords_x.size and not np.allclose(coords_x, coords_y):
        raise RuntimeError(f"Mismatched vector boundary coordinate ordering for tag {tag!r}")
    values = np.column_stack(
        [
            np.asarray(global_values, dtype=float)[np.asarray(gdofs_x, dtype=int)],
            np.asarray(global_values, dtype=float)[np.asarray(gdofs_y, dtype=int)],
        ]
    )
    return coords_x, values


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
    scalar: Function,
    point: tuple[float, float],
) -> tuple[float, np.ndarray]:
    xy = np.asarray(point, dtype=float)
    eid = _find_element_containing_point(mesh, xy)
    xi, eta = transform.inverse_mapping(mesh, eid, xy)
    me = dh.mixed_element
    local_phi = me.basis(scalar.field_name, float(xi), float(eta))[me.slice(scalar.field_name)]
    local_grad_ref = me.grad_basis(scalar.field_name, float(xi), float(eta))[me.slice(scalar.field_name)]
    local_grad = transform.map_grad_scalar(mesh, eid, local_grad_ref, (float(xi), float(eta)))
    gdofs = dh.element_maps[scalar.field_name][eid]
    vals = scalar.get_nodal_values(gdofs)
    return float(local_phi @ vals), np.asarray(vals, dtype=float) @ np.asarray(local_grad, dtype=float)


def _eval_vector_with_grad(
    dh: DofHandler,
    mesh: Mesh,
    vector: VectorFunction,
    point: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    grads = []
    for component in vector.components:
        value, grad_value = _eval_scalar_with_grad(dh, mesh, component, point)
        values.append(value)
        grads.append(grad_value)
    return np.asarray(values, dtype=float), np.vstack(grads)


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
        normal = normals[int(np.argmin(dist2))]
        norm = np.linalg.norm(normal)
        out[i, :] = normal / max(norm, 1.0e-14)
    return out


def _transfer_vector_field(*, target_dh: DofHandler, target_vec: VectorFunction, source_lookup: CoordinateLookup) -> None:
    target_dh._ensure_dof_coords()
    for idx, component in enumerate(target_vec.components):
        ids = np.asarray(target_dh.get_field_slice(component.field_name), dtype=int)
        xy = np.asarray(target_dh._dof_coords[ids], dtype=float)
        vals = np.asarray(source_lookup(xy[:, 0], xy[:, 1]), dtype=float)
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        component.set_nodal_values(ids, vals[:, idx])


def _fluid_residual_and_jacobian(
    *,
    prob: dict[str, object],
    rho_f: float,
    mu_f: float,
    dt: float,
    bossak_alpha: float,
    dynamic_tau: float,
    pressure_gauge: float,
    iface_velocity: CoordinateLookup,
    inlet_lookup: Callable[[float, float], float],
    interface_tag: str,
    outlet_tag: str,
    walls_tag: str,
    cylinder_tag: str,
    quad_order: int,
):
    u_k: VectorFunction = prob["u_k"]
    u_prev: VectorFunction = prob["u_prev"]
    a_prev: VectorFunction = prob["a_prev"]
    p_k: Function = prob["p_k"]
    d_mesh: VectorFunction = prob["d_mesh"]
    d_prev: VectorFunction = prob["d_prev"]
    du = prob["du"]
    v = prob["v"]
    dp = prob["dp"]
    q = prob["q"]

    dx_f = dx(metadata={"q": int(quad_order)})
    h = CellDiameter()
    F = Identity(2) + grad(d_mesh)
    Finv = inv(F)
    J = det(F)
    w_mesh = (d_mesh - d_prev) / Constant(float(dt))
    F_old = Identity(2) + grad(d_prev)
    cof_F = cof(F)
    cof_F_old = cof(F_old)
    grad_u_phys = dot(grad(u_k), Finv)
    grad_du_phys = dot(grad(du), Finv)
    div_u_phys = inner(cof_F, grad(u_k)) / J
    div_du_phys = inner(cof_F, grad(du)) / J
    div_u_old_phys = inner(cof_F_old, grad(u_prev)) / det(F_old)
    div_v_phys = inner(cof_F, grad(v)) / J
    conv_velocity = u_k - w_mesh
    conv_speed = (dot(conv_velocity, conv_velocity) + Constant(1.0e-12)) ** Constant(0.5)
    bossak = _bossak_coefficients(alpha=float(bossak_alpha), dt=float(dt))
    bossak_ma0 = Constant(float(bossak["ma0"]))
    bossak_ma2 = Constant(float(bossak["ma2"]))
    bossak_mass_coeff = Constant(float(bossak["mam"]))
    bossak_alpha_const = Constant(float(bossak["alpha"]))
    a_curr = bossak_ma0 * (u_k - u_prev) + bossak_ma2 * a_prev
    a_relaxed = (Constant(1.0) - bossak_alpha_const) * a_curr + bossak_alpha_const * a_prev
    pI = p_k * Identity(2)
    sigma = -pI + Constant(float(mu_f)) * (grad_u_phys + dot(Finv.T, grad(u_k).T))
    gauge = Constant(float(pressure_gauge))
    rho = Constant(float(rho_f))
    inv_dt = Constant(1.0 / max(float(dt), 1.0e-14))
    tau_c1 = Constant(8.0)
    tau_c2 = Constant(2.0)
    dynamic_tau_const = Constant(float(dynamic_tau))
    tau_one = dynamic_tau_const / (
        tau_c1 * Constant(float(mu_f)) / (h * h)
        + rho * (inv_dt + tau_c2 * conv_speed / h)
    )
    tau_two = Constant(float(mu_f)) + rho * conv_speed * h / Constant(4.0)
    tau_p = rho * h * h * inv_dt / tau_c1
    grad_p_phys = dot(Finv.T, grad(p_k))
    grad_q_phys = dot(Finv.T, grad(q))
    grad_dp_phys = dot(Finv.T, grad(dp))
    tau_test_mass = rho * bossak_mass_coeff * v
    tau_test_conv = rho * dot(dot(grad(v), Finv), conv_velocity)
    tau_test_pres = grad_q_phys
    tau_res_mass = rho * a_relaxed
    tau_res_conv = rho * dot(grad_u_phys, conv_velocity)
    tau_res_pres = grad_p_phys
    tau_dtest_conv = rho * dot(dot(grad(v), Finv), du)
    tau_dres_mass = rho * bossak_mass_coeff * du
    tau_dres_conv_1 = rho * dot(grad_du_phys, conv_velocity)
    tau_dres_conv_2 = rho * dot(grad_u_phys, du)
    tau_dres_pres = grad_dp_phys

    residual = rho * J * dot(a_relaxed, v) * dx_f
    residual += J * rho * dot(dot(grad_u_phys, conv_velocity), v) * dx_f
    residual += inner(J * dot(sigma, Finv.T), grad(v)) * dx_f
    residual += inner(cof_F, grad(u_k)) * q * dx_f
    residual += gauge * p_k * q * dx_f
    residual += J * tau_one * dot(tau_test_mass, tau_res_mass) * dx_f
    residual += J * tau_one * dot(tau_test_mass, tau_res_conv) * dx_f
    residual += J * tau_one * dot(tau_test_mass, tau_res_pres) * dx_f
    residual += J * tau_one * dot(tau_test_conv, tau_res_mass) * dx_f
    residual += J * tau_one * dot(tau_test_conv, tau_res_conv) * dx_f
    residual += J * tau_one * dot(tau_test_conv, tau_res_pres) * dx_f
    residual += J * tau_one * dot(tau_test_pres, tau_res_mass) * dx_f
    residual += J * tau_one * dot(tau_test_pres, tau_res_conv) * dx_f
    residual += J * tau_one * dot(tau_test_pres, tau_res_pres) * dx_f
    residual += J * (((tau_two + tau_p) * div_u_phys - tau_p * div_u_old_phys) * div_v_phys) * dx_f

    jacobian = rho * J * dot(bossak_mass_coeff * du, v) * dx_f
    jacobian += J * rho * dot(dot(grad_du_phys, conv_velocity), v) * dx_f
    jacobian += J * rho * dot(dot(grad_u_phys, du), v) * dx_f
    sigma_du = -dp * Identity(2) + Constant(float(mu_f)) * (grad_du_phys + dot(Finv.T, grad(du).T))
    jacobian += inner(J * dot(sigma_du, Finv.T), grad(v)) * dx_f
    jacobian += inner(cof_F, grad(du)) * q * dx_f
    jacobian += gauge * dp * q * dx_f
    jacobian += J * tau_one * dot(tau_dtest_conv, tau_res_mass) * dx_f
    jacobian += J * tau_one * dot(tau_dtest_conv, tau_res_conv) * dx_f
    jacobian += J * tau_one * dot(tau_dtest_conv, tau_res_pres) * dx_f
    jacobian += J * tau_one * dot(tau_test_mass, tau_dres_mass) * dx_f
    jacobian += J * tau_one * dot(tau_test_mass, tau_dres_conv_1) * dx_f
    jacobian += J * tau_one * dot(tau_test_mass, tau_dres_conv_2) * dx_f
    jacobian += J * tau_one * dot(tau_test_mass, tau_dres_pres) * dx_f
    jacobian += J * tau_one * dot(tau_test_conv, tau_dres_mass) * dx_f
    jacobian += J * tau_one * dot(tau_test_conv, tau_dres_conv_1) * dx_f
    jacobian += J * tau_one * dot(tau_test_conv, tau_dres_conv_2) * dx_f
    jacobian += J * tau_one * dot(tau_test_conv, tau_dres_pres) * dx_f
    jacobian += J * tau_one * dot(tau_test_pres, tau_dres_mass) * dx_f
    jacobian += J * tau_one * dot(tau_test_pres, tau_dres_conv_1) * dx_f
    jacobian += J * tau_one * dot(tau_test_pres, tau_dres_conv_2) * dx_f
    jacobian += J * tau_one * dot(tau_test_pres, tau_dres_pres) * dx_f
    jacobian += J * ((tau_two + tau_p) * div_du_phys * div_v_phys) * dx_f

    zero = lambda x, y, t=0.0: 0.0
    bcs = [
        BoundaryCondition("ux", "dirichlet", "inlet", inlet_lookup),
        BoundaryCondition("uy", "dirichlet", "inlet", zero),
        BoundaryCondition("ux", "dirichlet", walls_tag, zero),
        BoundaryCondition("uy", "dirichlet", walls_tag, zero),
        BoundaryCondition("ux", "dirichlet", cylinder_tag, zero),
        BoundaryCondition("uy", "dirichlet", cylinder_tag, zero),
        BoundaryCondition("ux", "dirichlet", interface_tag, iface_velocity.component(0)),
        BoundaryCondition("uy", "dirichlet", interface_tag, iface_velocity.component(1)),
        BoundaryCondition("p", "dirichlet", outlet_tag, zero),
    ]
    bcs_homog = [
        BoundaryCondition("ux", "dirichlet", "inlet", zero),
        BoundaryCondition("uy", "dirichlet", "inlet", zero),
        BoundaryCondition("ux", "dirichlet", walls_tag, zero),
        BoundaryCondition("uy", "dirichlet", walls_tag, zero),
        BoundaryCondition("ux", "dirichlet", cylinder_tag, zero),
        BoundaryCondition("uy", "dirichlet", cylinder_tag, zero),
        BoundaryCondition("ux", "dirichlet", interface_tag, zero),
        BoundaryCondition("uy", "dirichlet", interface_tag, zero),
        BoundaryCondition("p", "dirichlet", outlet_tag, zero),
    ]
    return residual, jacobian, bcs, bcs_homog


def _solid_residual_and_jacobian(
    *,
    prob: dict[str, object],
    traction_lookup: CoordinateLookup,
    mu_s: float,
    lambda_s: float,
    interface_tag: str,
    clamp_tag: str,
    quad_order: int,
):
    d_k: VectorFunction = prob["d_k"]
    dd = prob["dd"]
    w = prob["w"]

    traction = Analytic(lambda x, y: traction_lookup(x, y), dim=1)
    dx_s = dx(metadata={"q": int(quad_order)})
    ds_iface = dS(defined_on=prob["dh"].mixed_element.mesh.edge_bitset(interface_tag), metadata={"q": int(quad_order)})

    F = Identity(2) + grad(d_k)
    P = neo_hookean_pk1(F, Constant(float(mu_s)), Constant(float(lambda_s)))
    deltaP = neo_hookean_delta_pk1(F, grad(dd), Constant(float(mu_s)), Constant(float(lambda_s)))
    residual = inner(P, grad(w)) * dx_s - dot(traction, w) * ds_iface
    jacobian = inner(deltaP, grad(w)) * dx_s

    zero = lambda x, y, t=0.0: 0.0
    bcs = [
        BoundaryCondition("dx", "dirichlet", clamp_tag, zero),
        BoundaryCondition("dy", "dirichlet", clamp_tag, zero),
    ]
    bcs_homog = [
        BoundaryCondition("dx", "dirichlet", clamp_tag, zero),
        BoundaryCondition("dy", "dirichlet", clamp_tag, zero),
    ]
    return residual, jacobian, bcs, bcs_homog


def _mesh_extension_equation(
    *,
    prob: dict[str, object],
    interface_disp: CoordinateLookup,
    interface_tag: str,
    fixed_tags: tuple[str, ...],
    quad_order: int,
) -> tuple[Equation, list[BoundaryCondition]]:
    dm = prob["dm"]
    z = prob["z"]
    zero_vec = Constant(np.zeros(2), dim=1)
    equation = Equation(
        inner(grad(dm), grad(z)) * dx(metadata={"q": int(quad_order)}),
        Constant(0.0) * dot(z, zero_vec) * dx(metadata={"q": int(quad_order)}),
    )
    zero = lambda x, y, t=0.0: 0.0
    bcs = [
        BoundaryCondition("mx", "dirichlet", tag, zero) for tag in fixed_tags
    ] + [
        BoundaryCondition("my", "dirichlet", tag, zero) for tag in fixed_tags
    ] + [
        BoundaryCondition("mx", "dirichlet", interface_tag, interface_disp.component(0)),
        BoundaryCondition("my", "dirichlet", interface_tag, interface_disp.component(1)),
    ]
    return equation, bcs


def _fluid_interface_samples(
    *,
    dh: DofHandler,
    mesh: Mesh,
    u: VectorFunction,
    p: Function,
    d_mesh: VectorFunction,
    iface_coords: np.ndarray,
    interface_tag: str,
    mu_f: float,
) -> CoordinateLookup:
    normals = _sample_boundary_normals(mesh, interface_tag, iface_coords)
    traction_vals = np.empty((iface_coords.shape[0], 2), dtype=float)
    for i, xy in enumerate(np.asarray(iface_coords, dtype=float)):
        p_val, _ = _eval_scalar_with_grad(dh, mesh, p, tuple(xy))
        _, grad_u = _eval_vector_with_grad(dh, mesh, u, tuple(xy))
        _, grad_m = _eval_vector_with_grad(dh, mesh, d_mesh, tuple(xy))
        F = np.eye(2, dtype=float) + grad_m
        Finv = np.linalg.inv(F)
        J = float(np.linalg.det(F))
        sigma = -float(p_val) * np.eye(2) + float(mu_f) * (grad_u @ Finv + Finv.T @ grad_u.T)
        n_solid = -normals[i]
        traction_vals[i, :] = J * ((sigma @ Finv.T) @ n_solid)
    return CoordinateLookup(iface_coords, traction_vals, dim=2)


def _fluid_interface_point_loads(
    *,
    dh: DofHandler,
    mesh: Mesh,
    u: VectorFunction,
    p: Function,
    d_mesh: VectorFunction,
    interface_tag: str,
    mu_f: float,
    quad_order: int,
) -> CoordinateLookup:
    n_edge_q = max(2, int(math.ceil((max(int(quad_order), 1) + 1) / 2)))
    quad_pts, quad_w = np.polynomial.legendre.leggauss(n_edge_q)
    me = dh.mixed_element

    ux_name = u.components[0].field_name
    uy_name = u.components[1].field_name
    mx_name = d_mesh.components[0].field_name
    my_name = d_mesh.components[1].field_name
    p_name = p.field_name

    rhs = np.zeros(dh.total_dofs, dtype=float)

    for edge_id in mesh.edge_bitset(interface_tag).to_indices():
        edge = mesh.edge(int(edge_id))
        if edge.right is not None:
            continue
        if edge.left is None:
            continue

        elem_id = int(edge.left)
        node_ids = list(getattr(edge, "all_nodes", None) or edge.nodes)
        if len(node_ids) < 2:
            continue

        a = np.asarray(mesh.nodes_x_y_pos[node_ids[0]], dtype=float)
        b = np.asarray(mesh.nodes_x_y_pos[node_ids[-1]], dtype=float)
        jac_line = 0.5 * float(np.linalg.norm(b - a))
        if jac_line <= 1.0e-20:
            continue

        n_solid = -np.asarray(edge.normal, dtype=float)
        n_norm = float(np.linalg.norm(n_solid))
        if n_norm <= 1.0e-20:
            continue
        n_solid /= n_norm

        ux_gdofs = np.asarray(dh.element_maps[ux_name][elem_id], dtype=int)
        uy_gdofs = np.asarray(dh.element_maps[uy_name][elem_id], dtype=int)
        p_gdofs = np.asarray(dh.element_maps[p_name][elem_id], dtype=int)
        mx_gdofs = np.asarray(dh.element_maps[mx_name][elem_id], dtype=int)
        my_gdofs = np.asarray(dh.element_maps[my_name][elem_id], dtype=int)

        ux_vals = np.asarray(u.components[0].get_nodal_values(ux_gdofs), dtype=float)
        uy_vals = np.asarray(u.components[1].get_nodal_values(uy_gdofs), dtype=float)
        p_vals = np.asarray(p.get_nodal_values(p_gdofs), dtype=float)
        mx_vals = np.asarray(d_mesh.components[0].get_nodal_values(mx_gdofs), dtype=float)
        my_vals = np.asarray(d_mesh.components[1].get_nodal_values(my_gdofs), dtype=float)

        for qp, wq in zip(np.asarray(quad_pts, dtype=float), np.asarray(quad_w, dtype=float)):
            xy = 0.5 * ((1.0 - qp) * a + (1.0 + qp) * b)
            xi, eta = transform.inverse_mapping(mesh, elem_id, xy)
            xi_f = float(xi)
            eta_f = float(eta)

            basis_ux = np.asarray(me.basis(ux_name, xi_f, eta_f)[me.slice(ux_name)], dtype=float)
            basis_uy = np.asarray(me.basis(uy_name, xi_f, eta_f)[me.slice(uy_name)], dtype=float)
            basis_p = np.asarray(me.basis(p_name, xi_f, eta_f)[me.slice(p_name)], dtype=float)
            grad_ux_ref = np.asarray(me.grad_basis(ux_name, xi_f, eta_f)[me.slice(ux_name)], dtype=float)
            grad_uy_ref = np.asarray(me.grad_basis(uy_name, xi_f, eta_f)[me.slice(uy_name)], dtype=float)
            grad_mx_ref = np.asarray(me.grad_basis(mx_name, xi_f, eta_f)[me.slice(mx_name)], dtype=float)
            grad_my_ref = np.asarray(me.grad_basis(my_name, xi_f, eta_f)[me.slice(my_name)], dtype=float)

            grad_ux = np.asarray(transform.map_grad_scalar(mesh, elem_id, grad_ux_ref, (xi_f, eta_f)), dtype=float)
            grad_uy = np.asarray(transform.map_grad_scalar(mesh, elem_id, grad_uy_ref, (xi_f, eta_f)), dtype=float)
            grad_mx = np.asarray(transform.map_grad_scalar(mesh, elem_id, grad_mx_ref, (xi_f, eta_f)), dtype=float)
            grad_my = np.asarray(transform.map_grad_scalar(mesh, elem_id, grad_my_ref, (xi_f, eta_f)), dtype=float)

            p_val = float(basis_p @ p_vals)
            grad_u = np.vstack([ux_vals @ grad_ux, uy_vals @ grad_uy])
            grad_m = np.vstack([mx_vals @ grad_mx, my_vals @ grad_my])

            F = np.eye(2, dtype=float) + grad_m
            Finv = np.linalg.inv(F)
            J = float(np.linalg.det(F))
            sigma = -p_val * np.eye(2, dtype=float) + float(mu_f) * (grad_u @ Finv + Finv.T @ grad_u.T)
            traction = J * ((sigma @ Finv.T) @ n_solid)
            weight = jac_line * float(wq)

            rhs[ux_gdofs] += weight * basis_ux * float(traction[0])
            rhs[uy_gdofs] += weight * basis_uy * float(traction[1])

    iface_coords, iface_values = _boundary_vector_from_global_values(
        dh,
        vector=u,
        tag=interface_tag,
        global_values=rhs,
    )
    return CoordinateLookup(iface_coords, iface_values, dim=2)


def _solid_interface_disp_velocity(
    *,
    dh: DofHandler,
    mesh: Mesh,
    d_curr: VectorFunction,
    d_prev: VectorFunction,
    iface_coords: np.ndarray,
    dt: float,
) -> tuple[CoordinateLookup, CoordinateLookup]:
    disp_vals = np.empty((iface_coords.shape[0], 2), dtype=float)
    vel_vals = np.empty((iface_coords.shape[0], 2), dtype=float)
    for i, xy in enumerate(np.asarray(iface_coords, dtype=float)):
        disp_curr, _ = _eval_vector_with_grad(dh, mesh, d_curr, tuple(xy))
        disp_prev_val, _ = _eval_vector_with_grad(dh, mesh, d_prev, tuple(xy))
        disp_vals[i, :] = disp_curr
        vel_vals[i, :] = (disp_curr - disp_prev_val) / max(float(dt), 1.0e-14)
    return CoordinateLookup(iface_coords, disp_vals, dim=2), CoordinateLookup(iface_coords, vel_vals, dim=2)


def _snapshot_function_values(functions: list[Function | VectorFunction]) -> list[np.ndarray]:
    return [np.asarray(function.nodal_values, dtype=float).copy() for function in functions]


def _restore_function_values(functions: list[Function | VectorFunction], snapshots: list[np.ndarray]) -> None:
    if len(functions) != len(snapshots):
        raise ValueError("functions / snapshots length mismatch")
    for function, values in zip(functions, snapshots):
        function.nodal_values[:] = np.asarray(values, dtype=float)


def _guess_callback_from_snapshots(snapshots: list[np.ndarray]):
    def _callback(*, functions, **kwargs) -> None:
        del kwargs
        _restore_function_values(list(functions), snapshots)

    return _callback


def _bossak_coefficients(alpha: float, dt: float) -> dict[str, float]:
    alpha_value = float(alpha)
    dt_value = max(float(dt), 1.0e-14)
    gamma = 0.5 - alpha_value
    if gamma <= 0.0:
        raise ValueError(f"Bossak alpha={alpha_value} yields non-positive gamma={gamma}.")
    beta = 0.25 * (1.0 - alpha_value) ** 2
    ma0 = 1.0 / (gamma * dt_value)
    ma2 = (-1.0 + gamma) / gamma
    mam = (1.0 - alpha_value) * ma0
    return {
        "alpha": alpha_value,
        "dt": dt_value,
        "gamma": float(gamma),
        "beta": float(beta),
        "ma0": float(ma0),
        "ma2": float(ma2),
        "mam": float(mam),
    }


def _relative_change(new: np.ndarray, old: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(new, dtype=float) - np.asarray(old, dtype=float)
    size = max(int(diff.size), 1)
    abs_norm = float(np.linalg.norm(diff.ravel(), ord=2) / math.sqrt(float(size)))
    base = max(float(np.linalg.norm(np.asarray(new, dtype=float).ravel(), ord=2)), 1.0e-14)
    return abs_norm, abs_norm / base


def _relaxed_lookup(
    coords: np.ndarray,
    old_values: np.ndarray,
    new_values: np.ndarray,
    *,
    omega: float,
) -> CoordinateLookup:
    omega_value = float(np.clip(float(omega), 0.0, 1.0))
    values = (1.0 - omega_value) * np.asarray(old_values, dtype=float) + omega_value * np.asarray(new_values, dtype=float)
    return CoordinateLookup(np.asarray(coords, dtype=float), values, dim=2)


def _aitken_relaxation_factor(
    *,
    omega_prev: float,
    residual_prev: np.ndarray | None,
    residual_curr: np.ndarray,
    omega_min: float,
    omega_max: float,
) -> float:
    omega = float(np.clip(float(omega_prev), float(omega_min), float(omega_max)))
    if residual_prev is None:
        return omega
    r_prev = np.asarray(residual_prev, dtype=float).ravel()
    r_curr = np.asarray(residual_curr, dtype=float).ravel()
    delta = r_curr - r_prev
    denom = float(np.dot(delta, delta))
    if denom <= 1.0e-30 or not np.isfinite(denom):
        return omega
    omega_new = -omega * float(np.dot(r_prev, delta)) / denom
    if not np.isfinite(omega_new):
        return omega
    return float(np.clip(omega_new, float(omega_min), float(omega_max)))


def _iqnils_next_iterate(
    *,
    x_curr: np.ndarray,
    g_curr: np.ndarray,
    x_history: list[np.ndarray],
    g_history: list[np.ndarray],
    dr_old_mats: list[np.ndarray] | None = None,
    dg_old_mats: list[np.ndarray] | None = None,
    omega: float,
    horizon: int,
    regularization: float,
) -> np.ndarray:
    x_curr_arr = np.asarray(x_curr, dtype=float)
    g_curr_arr = np.asarray(g_curr, dtype=float)
    x_curr_vec = x_curr_arr.reshape(-1)
    g_curr_vec = g_curr_arr.reshape(-1)
    r_curr = g_curr_vec - x_curr_vec
    omega_value = float(np.clip(float(omega), 0.0, 1.0))
    picard = (x_curr_vec + omega_value * r_curr).reshape(x_curr_arr.shape)
    count = min(max(int(horizon), 1), len(x_history) - 1, len(g_history) - 1)
    current_d_r_blocks: list[np.ndarray] = []
    current_d_g_blocks: list[np.ndarray] = []
    if count >= 1:
        x_seq = [np.asarray(values, dtype=float).reshape(-1) for values in x_history[-(count + 1) :]]
        g_seq = [np.asarray(values, dtype=float).reshape(-1) for values in g_history[-(count + 1) :]]
        r_seq = [g_vec - x_vec for x_vec, g_vec in zip(x_seq, g_seq)]
        if len(r_seq) >= 2:
            current_d_r_blocks.append(np.column_stack([r_seq[i + 1] - r_seq[i] for i in range(len(r_seq) - 1)]))
            current_d_g_blocks.append(np.column_stack([g_seq[i + 1] - g_seq[i] for i in range(len(g_seq) - 1)]))

    d_r_blocks = [block for block in current_d_r_blocks if block.size]
    d_g_blocks = [block for block in current_d_g_blocks if block.size]
    if dr_old_mats:
        d_r_blocks.extend([np.asarray(block, dtype=float) for block in dr_old_mats if np.asarray(block).size])
    if dg_old_mats:
        d_g_blocks.extend([np.asarray(block, dtype=float) for block in dg_old_mats if np.asarray(block).size])
    if not d_r_blocks or not d_g_blocks:
        return picard
    d_r = np.hstack(d_r_blocks)
    d_g = np.hstack(d_g_blocks)

    reg = max(float(regularization), 0.0)
    try:
        if reg > 0.0:
            n_cols = d_r.shape[1]
            d_r_aug = np.vstack([d_r, np.sqrt(reg) * np.eye(n_cols, dtype=float)])
            rhs_aug = np.concatenate([-r_curr, np.zeros(n_cols, dtype=float)])
            gamma = np.linalg.lstsq(d_r_aug, rhs_aug, rcond=None)[0]
        else:
            gamma = np.linalg.lstsq(d_r, -r_curr, rcond=None)[0]
    except np.linalg.LinAlgError:
        return picard

    delta_x = d_g @ gamma + r_curr
    if not np.all(np.isfinite(delta_x)):
        return picard

    return (x_curr_vec + delta_x).reshape(x_curr_arr.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local DoubleFlap Example 2 with a strong staggered fixed-point FSI loop.")
    parser.add_argument("--reference-root", type=Path, default=None, help="Downloaded DoubleFlap reference directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/NIRB/artifacts/example2_local_fom"))
    parser.add_argument("--mesh-size", type=float, default=0.20)
    parser.add_argument("--mesh-order", type=int, default=1)
    parser.add_argument("--poly-order", type=int, default=1)
    parser.add_argument("--pressure-order", type=int, default=None)
    parser.add_argument("--reynolds", type=float, default=250.0)
    parser.add_argument("--reference-velocity", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-coupling-iters", type=int, default=6)
    parser.add_argument("--coupling-rel-tol", type=float, default=1.0e-6)
    parser.add_argument("--coupling-abs-tol", type=float, default=1.0e-6)
    parser.add_argument("--force-update", choices=("constant", "aitken", "iqnils"), default="iqnils")
    parser.add_argument("--force-relaxation", type=float, default=0.5)
    parser.add_argument("--force-relaxation-min", type=float, default=1.0e-3)
    parser.add_argument("--force-relaxation-max", type=float, default=1.0)
    parser.add_argument("--force-history", type=int, default=3)
    parser.add_argument("--force-regularization", type=float, default=1.0e-10)
    parser.add_argument("--newton-tol", type=float, default=1.0e-6)
    parser.add_argument("--max-newton-iter", type=int, default=12)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    parser.add_argument("--dynamic-tau", type=float, default=1.0)
    parser.add_argument("--pressure-gauge", type=float, default=1.0e-8)
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default="python")
    parser.add_argument("--linear-backend", choices=("scipy", "petsc"), default="scipy")
    parser.add_argument("--snapshot-mode", choices=("all", "converged"), default="all")
    parser.add_argument("--reuse-mesh", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_local_example2(
    *,
    output_dir: Path,
    reference_root: Path | None = None,
    mesh_size: float = 0.20,
    mesh_order: int = 1,
    poly_order: int = 1,
    pressure_order: int | None = None,
    reynolds: float = 250.0,
    reference_velocity: float | None = None,
    dt: float | None = None,
    end_time: float | None = None,
    max_steps: int | None = None,
    max_coupling_iters: int = 6,
    coupling_rel_tol: float = 1.0e-6,
    coupling_abs_tol: float = 1.0e-6,
    force_update: str = "iqnils",
    force_relaxation: float = 0.5,
    force_relaxation_min: float = 1.0e-3,
    force_relaxation_max: float = 1.0,
    force_history: int = 3,
    force_regularization: float = 1.0e-10,
    newton_tol: float = 1.0e-6,
    max_newton_iter: int = 12,
    bossak_alpha: float = -0.3,
    dynamic_tau: float = 1.0,
    pressure_gauge: float = 1.0e-8,
    backend: str = "python",
    linear_backend: str = "scipy",
    snapshot_mode: str = "all",
    reuse_mesh: bool = False,
    verbose: bool = False,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    co_sim_dir = output_dir / "coSimData"
    co_sim_dir.mkdir(parents=True, exist_ok=True)

    setup = load_example2_local_setup(
        reference_root=reference_root,
        mesh_size_default=mesh_size,
        mesh_order_default=mesh_order,
    )
    geometry = setup.geometry
    reference_velocity_value = (
        float(reference_velocity)
        if reference_velocity is not None
        else float(setup.material.max_velocity)
    )
    dt_value = float(setup.boundaries.time_step if dt is None else dt)
    end_time_value = float(setup.boundaries.end_time if end_time is None else end_time)
    max_steps_value = int(
        max_steps if max_steps is not None else max(1, math.ceil(end_time_value / max(dt_value, 1.0e-14)))
    )
    step_count = min(max_steps_value, int(max(1, math.ceil(end_time_value / max(dt_value, 1.0e-14)))))
    pressure_order_value = int(pressure_order if pressure_order is not None else max(1, poly_order - 1))
    quad_order = 2 * int(poly_order) + 4

    mesh_path = output_dir / "double_flap_conforming.msh"
    if (not mesh_path.exists()) or (not reuse_mesh):
        _log(verbose, f"[mesh] building {mesh_path} (h={mesh_size:.3f}, order={mesh_order})")
        build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=float(mesh_size), order=int(mesh_order))

    _log(verbose, "[mesh] loading fluid and solid submeshes")
    mesh_f = mesh_from_gmsh(mesh_path, surface_physical_names=["fluid"], apply_boundary_tags=True)
    mesh_s = mesh_from_gmsh(mesh_path, surface_physical_names=["solid"], apply_boundary_tags=True)

    fluid = _build_fluid_problem(mesh_f, poly_order=int(poly_order), pressure_order=pressure_order_value)
    mesh_ext = _build_mesh_extension_problem(mesh_f, poly_order=int(poly_order))
    solid = _build_solid_problem(mesh_s, poly_order=int(poly_order))

    fluid_iface_coords, _ = _boundary_field_data(fluid["dh"], "ux", geometry.interface_tag)
    solid_iface_coords, _ = _boundary_field_data(solid["dh"], "dx", geometry.interface_tag)
    if fluid_iface_coords.size == 0 or solid_iface_coords.size == 0:
        raise RuntimeError("Failed to extract interface DOF coordinates from the local fluid/solid subproblems.")
    map_used = _build_interface_restriction_matrix(solid["dh"], solid["d_k"], geometry.interface_tag)
    np.save(co_sim_dir / "map_used.npy", map_used)
    np.save(co_sim_dir / "coords_interf.npy", solid_iface_coords)

    zero_load_lookup = CoordinateLookup(
        solid_iface_coords,
        np.zeros((solid_iface_coords.shape[0], 2), dtype=float),
        dim=2,
    )
    current_load_lookup = zero_load_lookup
    prev_disp_iter_vals = np.zeros((solid_iface_coords.shape[0], 2), dtype=float)

    mu_f = float(setup.material.density * setup.material.kinematic_viscosity)
    mu_s = float(setup.material.shear_modulus)
    lambda_s = float(setup.material.lame_lambda)

    fixed_mesh_tags = (
        geometry.inlet_tag,
        geometry.outlet_tag,
        geometry.walls_tag,
        geometry.cylinder_tag,
    )

    disp_snapshots: list[np.ndarray] = []
    load_snapshots: list[np.ndarray] = []
    interface_disp_snapshots: list[np.ndarray] = []
    interface_velocity_snapshots: list[np.ndarray] = []
    snapshot_rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = []
    fluid_times: list[float] = []
    structure_times: list[float] = []
    increment_times: list[float] = []
    coupling_iters_per_step: list[int] = []
    converged_steps = 0
    t_total_start = time.perf_counter()
    iqn_old_dr_mats: deque[np.ndarray] = deque(maxlen=max(int(force_history) - 1, 0))
    iqn_old_dg_mats: deque[np.ndarray] = deque(maxlen=max(int(force_history) - 1, 0))

    for step in range(1, step_count + 1):
        t_now = min(end_time_value, step * dt_value)
        _log(verbose, f"[time] step={step}/{step_count} t={t_now:.6f}s")
        increment_start = time.perf_counter()
        solid_prev_step = _snapshot_function_values([solid["d_prev"]])
        fluid_prev_step = _snapshot_function_values([fluid["u_prev"], fluid["p_prev"]])
        fluid_mesh_prev_step = _snapshot_function_values([fluid["d_prev"]])
        if step == 1:
            prev_disp_iter_vals.fill(0.0)
        else:
            _, prev_disp_iter_vals = _boundary_vector_snapshot(solid["dh"], solid["d_prev"], geometry.interface_tag)
        step_converged = False
        last_disp_abs = last_disp_rel = last_load_abs = last_load_rel = float("nan")
        last_force_omega = float(force_relaxation)
        prev_force_residual: np.ndarray | None = None
        load_guess_history: list[np.ndarray] = []
        load_return_history: list[np.ndarray] = []

        def inlet_profile(x: float, y: float) -> float:
            del x
            return geometry.inlet_velocity(y, t_now, reference_velocity=reference_velocity_value)

        for coupling_iter in range(1, int(max_coupling_iters) + 1):
            _log(verbose, f"[coupling] step={step} iter={coupling_iter}/{max_coupling_iters}")
            load_guess_vals = np.asarray(current_load_lookup.values, dtype=float).copy()

            solid_res, solid_jac, solid_bcs, solid_bcs_homog = _solid_residual_and_jacobian(
                prob=solid,
                traction_lookup=zero_load_lookup,
                mu_s=mu_s,
                lambda_s=lambda_s,
                interface_tag=geometry.interface_tag,
                clamp_tag=geometry.clamp_tag,
                quad_order=quad_order,
            )
            solid_solver = NewtonSolver(
                residual_form=solid_res,
                jacobian_form=solid_jac,
                dof_handler=solid["dh"],
                mixed_element=solid["me"],
                bcs=solid_bcs,
                bcs_homog=solid_bcs_homog,
                newton_params=NewtonParameters(
                    newton_tol=float(newton_tol),
                    max_newton_iter=int(max_newton_iter),
                    line_search=True,
                    ls_mode="dealii",
                    globalization="line_search_then_trust",
                ),
                lin_params=LinearSolverParameters(backend=str(linear_backend)),
                quad_order=quad_order,
                backend=str(backend),
            )
            solid_point_load_full = _boundary_point_load_vector(
                solid["dh"],
                vector=solid["d_k"],
                tag=geometry.interface_tag,
                values=load_guess_vals,
            )
            solid_point_load_red = np.asarray(solid_point_load_full[np.asarray(solid_solver.active_dofs, dtype=int)], dtype=float)
            solid_assemble_reduced = solid_solver._assemble_system_reduced

            def _assemble_system_reduced_with_point_load(coeffs, *, need_matrix: bool = True):
                A_red, R_red = solid_assemble_reduced(coeffs, need_matrix=need_matrix)
                return A_red, np.asarray(R_red, dtype=float) - solid_point_load_red

            solid_solver._assemble_system_reduced = _assemble_system_reduced_with_point_load
            t_solid0 = time.perf_counter()
            solid_guess = _snapshot_function_values([solid["d_k"]])
            _restore_function_values([solid["d_prev"]], solid_prev_step)
            solid_solver.solve_time_interval(
                functions=[solid["d_k"]],
                prev_functions=[solid["d_prev"]],
                time_params=TimeStepperParameters(
                    dt=1.0,
                    max_steps=1,
                    final_time=1.0,
                    stop_on_steady=False,
                    step_initial_guess_callback=_guess_callback_from_snapshots(solid_guess),
                ),
            )
            _restore_function_values([solid["d_prev"]], solid_prev_step)
            solid_elapsed = time.perf_counter() - t_solid0
            structure_times.append(float(solid_elapsed))

            solid_disp_solid_lookup, _ = _solid_interface_disp_velocity(
                dh=solid["dh"],
                mesh=mesh_s,
                d_curr=solid["d_k"],
                d_prev=solid["d_prev"],
                iface_coords=solid_iface_coords,
                dt=dt_value,
            )
            solid_disp_fluid_lookup, solid_vel_fluid_lookup = _solid_interface_disp_velocity(
                dh=solid["dh"],
                mesh=mesh_s,
                d_curr=solid["d_k"],
                d_prev=solid["d_prev"],
                iface_coords=fluid_iface_coords,
                dt=dt_value,
            )
            disp_snapshot = _flatten_vector_snapshot(solid["dh"], solid["d_k"])

            mesh_eq, mesh_bcs = _mesh_extension_equation(
                prob=mesh_ext,
                interface_disp=solid_disp_fluid_lookup,
                interface_tag=geometry.interface_tag,
                fixed_tags=fixed_mesh_tags,
                quad_order=quad_order,
            )
            _solve_linear(
                eq=mesh_eq,
                dh=mesh_ext["dh"],
                bcs=mesh_bcs,
                quad_order=quad_order,
                backend=str(backend),
                linear_backend=str(linear_backend),
                functions=[mesh_ext["m_k"]],
            )
            mesh_lookup = _vector_lookup_from_field(mesh_ext["dh"], mesh_ext["m_k"])
            _transfer_vector_field(target_dh=fluid["dh"], target_vec=fluid["d_mesh"], source_lookup=mesh_lookup)

            fluid_res, fluid_jac, fluid_bcs, fluid_bcs_homog = _fluid_residual_and_jacobian(
                prob=fluid,
                rho_f=float(setup.material.density),
                mu_f=mu_f,
                dt=dt_value,
                bossak_alpha=float(bossak_alpha),
                dynamic_tau=float(dynamic_tau),
                pressure_gauge=float(pressure_gauge),
                iface_velocity=solid_vel_fluid_lookup,
                inlet_lookup=inlet_profile,
                interface_tag=geometry.interface_tag,
                outlet_tag=geometry.outlet_tag,
                walls_tag=geometry.walls_tag,
                cylinder_tag=geometry.cylinder_tag,
                quad_order=quad_order,
            )
            fluid_solver = NewtonSolver(
                residual_form=fluid_res,
                jacobian_form=fluid_jac,
                dof_handler=fluid["dh"],
                mixed_element=fluid["me"],
                bcs=fluid_bcs,
                bcs_homog=fluid_bcs_homog,
                newton_params=NewtonParameters(
                    newton_tol=float(newton_tol),
                    max_newton_iter=int(max_newton_iter),
                    line_search=True,
                ),
                lin_params=LinearSolverParameters(backend=str(linear_backend)),
                quad_order=quad_order,
                backend=str(backend),
            )
            t_fluid0 = time.perf_counter()
            fluid_guess = _snapshot_function_values([fluid["u_k"], fluid["p_k"]])
            _restore_function_values([fluid["u_prev"], fluid["p_prev"]], fluid_prev_step)
            _restore_function_values([fluid["d_prev"]], fluid_mesh_prev_step)
            fluid_solver.solve_time_interval(
                functions=[fluid["u_k"], fluid["p_k"]],
                prev_functions=[fluid["u_prev"], fluid["p_prev"]],
                aux_functions={
                    "a_prev": fluid["a_prev"],
                    "d_mesh": fluid["d_mesh"],
                    "d_prev": fluid["d_prev"],
                },
                time_params=TimeStepperParameters(
                    dt=dt_value,
                    max_steps=1,
                    final_time=dt_value,
                    stop_on_steady=False,
                    step_initial_guess_callback=_guess_callback_from_snapshots(fluid_guess),
                ),
            )
            _restore_function_values([fluid["u_prev"], fluid["p_prev"]], fluid_prev_step)
            _restore_function_values([fluid["d_prev"]], fluid_mesh_prev_step)
            fluid_elapsed = time.perf_counter() - t_fluid0
            fluid_times.append(float(fluid_elapsed))

            fluid_point_load_lookup = _fluid_interface_point_loads(
                dh=fluid["dh"],
                mesh=mesh_f,
                u=fluid["u_k"],
                p=fluid["p_k"],
                d_mesh=fluid["d_mesh"],
                interface_tag=geometry.interface_tag,
                mu_f=mu_f,
                quad_order=quad_order,
            )
            returned_load_lookup = CoordinateLookup(
                solid_iface_coords,
                np.asarray(fluid_point_load_lookup(solid_iface_coords[:, 0], solid_iface_coords[:, 1]), dtype=float),
                dim=2,
            )
            interface_velocity_snapshot = np.asarray(
                solid_vel_fluid_lookup(solid_iface_coords[:, 0], solid_iface_coords[:, 1]),
                dtype=float,
            ).reshape(-1)
            interface_disp_snapshot = np.asarray(solid_disp_solid_lookup.values, dtype=float).reshape(-1)
            load_snapshot = np.asarray(returned_load_lookup.values, dtype=float).reshape(-1)
            load_residual = np.asarray(returned_load_lookup.values, dtype=float) - np.asarray(load_guess_vals, dtype=float)
            disp_abs, disp_rel = _relative_change(solid_disp_solid_lookup.values, prev_disp_iter_vals)
            load_abs, load_rel = _relative_change(returned_load_lookup.values, load_guess_vals)
            last_disp_abs = disp_abs
            last_disp_rel = disp_rel
            last_load_abs = load_abs
            last_load_rel = load_rel
            omega_force = float(force_relaxation)
            if str(force_update).lower() == "aitken":
                omega_force = _aitken_relaxation_factor(
                    omega_prev=float(last_force_omega),
                    residual_prev=prev_force_residual,
                    residual_curr=load_residual,
                    omega_min=float(force_relaxation_min),
                    omega_max=float(force_relaxation_max),
                )
            else:
                omega_force = float(np.clip(float(force_relaxation), float(force_relaxation_min), float(force_relaxation_max)))

            disp_max = float(np.max(np.linalg.norm(np.asarray(solid_disp_solid_lookup.values, dtype=float), axis=1)))
            load_guess_max = float(np.max(np.linalg.norm(load_guess_vals, axis=1)))
            load_return_max = float(np.max(np.linalg.norm(np.asarray(returned_load_lookup.values, dtype=float), axis=1)))
            row = {
                "step": int(step),
                "time_s": float(t_now),
                "coupling_iter": int(coupling_iter),
                "disp_abs": float(disp_abs),
                "disp_rel": float(disp_rel),
                "load_abs": float(load_abs),
                "load_rel": float(load_rel),
                "solid_time_s": float(solid_elapsed),
                "fluid_time_s": float(fluid_elapsed),
                "disp_max": disp_max,
                "load_guess_max": load_guess_max,
                "load_return_max": load_return_max,
                "force_omega": float(omega_force),
            }
            step_rows.append(row)

            keep_snapshot = snapshot_mode == "all"
            disp_converged = bool((disp_abs <= coupling_abs_tol) or (disp_rel <= coupling_rel_tol))
            load_converged = bool((load_abs <= coupling_abs_tol) or (load_rel <= coupling_rel_tol))
            step_converged = bool(disp_converged and load_converged)
            if snapshot_mode == "converged" and step_converged:
                keep_snapshot = True
            if keep_snapshot:
                disp_snapshots.append(disp_snapshot)
                load_snapshots.append(load_snapshot)
                interface_disp_snapshots.append(interface_disp_snapshot)
                interface_velocity_snapshots.append(interface_velocity_snapshot)
                snapshot_rows.append(
                    {
                        "step": int(step),
                        "time_s": float(t_now),
                        "coupling_iter": int(coupling_iter),
                        "converged": bool(step_converged),
                    }
                )

            _log(
                verbose,
                "[coupling] "
                f"step={step} iter={coupling_iter} "
                f"disp_abs={disp_abs:.3e} load_abs={load_abs:.3e} "
                f"disp_rel={disp_rel:.3e} load_rel={load_rel:.3e} "
                f"disp_max={disp_max:.3e} load_guess_max={load_guess_max:.3e} "
                f"load_return_max={load_return_max:.3e} omega={omega_force:.3e}",
            )
            load_guess_history.append(np.asarray(load_guess_vals, dtype=float).copy())
            load_return_history.append(np.asarray(returned_load_lookup.values, dtype=float).copy())
            if str(force_update).lower() == "iqnils":
                next_load_values = _iqnils_next_iterate(
                    x_curr=load_guess_vals,
                    g_curr=returned_load_lookup.values,
                    x_history=load_guess_history,
                    g_history=load_return_history,
                    dr_old_mats=list(iqn_old_dr_mats),
                    dg_old_mats=list(iqn_old_dg_mats),
                    omega=float(omega_force),
                    horizon=int(force_history),
                    regularization=float(force_regularization),
                )
                current_load_lookup = CoordinateLookup(solid_iface_coords, next_load_values, dim=2)
            else:
                current_load_lookup = _relaxed_lookup(
                    solid_iface_coords,
                    load_guess_vals,
                    returned_load_lookup.values,
                    omega=float(omega_force),
                )
            prev_disp_iter_vals = np.asarray(solid_disp_solid_lookup.values, dtype=float).copy()
            prev_force_residual = np.asarray(load_residual, dtype=float).copy()
            last_force_omega = float(omega_force)
            if step_converged:
                converged_steps += 1
                break

        coupling_iters_per_step.append(int(coupling_iter))
        increment_elapsed = time.perf_counter() - increment_start
        increment_times.append(float(increment_elapsed))

        bossak = _bossak_coefficients(alpha=float(bossak_alpha), dt=float(dt_value))
        u_prev_old = np.asarray(fluid["u_prev"].nodal_values, dtype=float).copy()
        a_prev_old = np.asarray(fluid["a_prev"].nodal_values, dtype=float).copy()
        solid["d_prev"].nodal_values[:] = solid["d_k"].nodal_values[:]
        fluid["u_prev"].nodal_values[:] = fluid["u_k"].nodal_values[:]
        fluid["p_prev"].nodal_values[:] = fluid["p_k"].nodal_values[:]
        fluid["d_prev"].nodal_values[:] = fluid["d_mesh"].nodal_values[:]
        fluid["a_prev"].nodal_values[:] = (
            float(bossak["ma0"]) * (fluid["u_k"].nodal_values[:] - u_prev_old)
            + float(bossak["ma2"]) * a_prev_old
        )
        if str(force_update).lower() == "iqnils" and len(load_guess_history) >= 2 and len(load_return_history) >= 2:
            x_seq = [np.asarray(values, dtype=float).reshape(-1) for values in load_guess_history]
            g_seq = [np.asarray(values, dtype=float).reshape(-1) for values in load_return_history]
            r_seq = [g_vec - x_vec for x_vec, g_vec in zip(x_seq, g_seq)]
            if len(r_seq) >= 2:
                iqn_old_dr_mats.appendleft(
                    np.column_stack([r_seq[i + 1] - r_seq[i] for i in range(len(r_seq) - 1)])
                )
                iqn_old_dg_mats.appendleft(
                    np.column_stack([g_seq[i + 1] - g_seq[i] for i in range(len(g_seq) - 1)])
                )

        _log(
            verbose,
            "[time] "
            f"step={step} done "
            f"iters={coupling_iter} converged={step_converged} "
            f"disp_abs={last_disp_abs:.3e} load_abs={last_load_abs:.3e} "
            f"disp_rel={last_disp_rel:.3e} load_rel={last_load_rel:.3e} "
            f"wall={increment_elapsed:.3f}s",
        )

    total_elapsed = time.perf_counter() - t_total_start
    disp_matrix = np.column_stack(disp_snapshots) if disp_snapshots else np.zeros((0, 0), dtype=float)
    load_matrix = np.column_stack(load_snapshots) if load_snapshots else np.zeros((0, 0), dtype=float)
    interface_disp_matrix = (
        np.column_stack(interface_disp_snapshots) if interface_disp_snapshots else np.zeros((0, 0), dtype=float)
    )
    interface_velocity_matrix = (
        np.column_stack(interface_velocity_snapshots)
        if interface_velocity_snapshots
        else np.zeros((0, 0), dtype=float)
    )
    np.save(co_sim_dir / "disp_data.npy", disp_matrix)
    np.save(co_sim_dir / "load_data.npy", load_matrix)
    np.save(co_sim_dir / "interface_disp_data.npy", interface_disp_matrix)
    np.save(co_sim_dir / "interface_velocity_data.npy", interface_velocity_matrix)
    np.save(co_sim_dir / "iters.npy", np.asarray(coupling_iters_per_step, dtype=int))
    np.save(co_sim_dir / "fluid_time.npy", np.asarray(fluid_times, dtype=float))
    np.save(co_sim_dir / "structure_time.npy", np.asarray(structure_times, dtype=float))
    np.save(co_sim_dir / "increment_time.npy", np.asarray(increment_times, dtype=float))
    np.save(co_sim_dir / "total_solving_time.npy", np.asarray(float(total_elapsed), dtype=float))

    metadata_path = output_dir / "snapshot_metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "time_s", "coupling_iter", "converged"])
        writer.writeheader()
        writer.writerows(snapshot_rows)

    timeseries_path = output_dir / "timeseries.csv"
    with timeseries_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "time_s",
                "coupling_iter",
                "disp_abs",
                "disp_rel",
                "load_abs",
                "load_rel",
                "solid_time_s",
                "fluid_time_s",
                "disp_max",
                "load_guess_max",
                "load_return_max",
                "force_omega",
            ],
        )
        writer.writeheader()
        writer.writerows(step_rows)

    summary = {
        "output_dir": str(output_dir),
        "mesh_path": str(mesh_path),
        "reynolds": float(reynolds),
        "reference_velocity": float(reference_velocity_value),
        "dt": float(dt_value),
        "end_time": float(end_time_value),
        "pressure_gauge": float(pressure_gauge),
        "bossak_alpha": float(bossak_alpha),
        "dynamic_tau": float(dynamic_tau),
        "steps_requested": int(step_count),
        "steps_converged": int(converged_steps),
        "max_coupling_iters": int(max_coupling_iters),
        "force_update": str(force_update),
        "force_relaxation": float(force_relaxation),
        "force_relaxation_min": float(force_relaxation_min),
        "force_relaxation_max": float(force_relaxation_max),
        "force_history": int(force_history),
        "force_regularization": float(force_regularization),
        "snapshot_mode": str(snapshot_mode),
        "snapshot_count": int(disp_matrix.shape[1]),
        "interface_dofs": int(load_matrix.shape[0]),
        "interface_snapshot_dofs": int(interface_disp_matrix.shape[0]),
        "solid_state_dofs": int(disp_matrix.shape[0]),
        "coupling_iters_per_step": [int(v) for v in coupling_iters_per_step],
        "mean_coupling_iters": float(np.mean(coupling_iters_per_step)) if coupling_iters_per_step else 0.0,
        "total_wall_time_s": float(total_elapsed),
        "mean_increment_time_s": float(np.mean(increment_times)) if increment_times else 0.0,
        "mean_solid_solve_time_s": float(np.mean(structure_times)) if structure_times else 0.0,
        "mean_fluid_solve_time_s": float(np.mean(fluid_times)) if fluid_times else 0.0,
        "co_sim_dir": str(co_sim_dir),
        "timeseries_path": str(timeseries_path),
        "snapshot_metadata_path": str(metadata_path),
    }
    dump_json(summary, output_dir / "summary.json")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_local_example2(
        output_dir=args.output_dir,
        reference_root=args.reference_root,
        mesh_size=float(args.mesh_size),
        mesh_order=int(args.mesh_order),
        poly_order=int(args.poly_order),
        pressure_order=args.pressure_order,
        reynolds=float(args.reynolds),
        reference_velocity=args.reference_velocity,
        dt=args.dt,
        end_time=args.end_time,
        max_steps=args.max_steps,
        max_coupling_iters=int(args.max_coupling_iters),
        coupling_rel_tol=float(args.coupling_rel_tol),
        coupling_abs_tol=float(args.coupling_abs_tol),
        force_update=str(args.force_update),
        force_relaxation=float(args.force_relaxation),
        force_relaxation_min=float(args.force_relaxation_min),
        force_relaxation_max=float(args.force_relaxation_max),
        force_history=int(args.force_history),
        force_regularization=float(args.force_regularization),
        newton_tol=float(args.newton_tol),
        max_newton_iter=int(args.max_newton_iter),
        bossak_alpha=float(args.bossak_alpha),
        dynamic_tau=float(args.dynamic_tau),
        pressure_gauge=float(args.pressure_gauge),
        backend=str(args.backend),
        linear_backend=str(args.linear_backend),
        snapshot_mode=str(args.snapshot_mode),
        reuse_mesh=bool(args.reuse_mesh),
        verbose=bool(args.verbose),
    )
    print(f"mesh: {summary['mesh_path']}")
    print(f"steps_requested: {summary['steps_requested']}")
    print(f"steps_converged: {summary['steps_converged']}")
    print(f"snapshots: {summary['snapshot_count']}")
    print(f"mean_coupling_iters: {summary['mean_coupling_iters']:.3f}")
    print(f"mean_solid_solve_time_s: {summary['mean_solid_solve_time_s']:.6f}")
    print(f"mean_fluid_solve_time_s: {summary['mean_fluid_solve_time_s']:.6f}")
    print(f"summary: {Path(args.output_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
