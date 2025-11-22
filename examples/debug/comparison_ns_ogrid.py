"""
Compare Navier–Stokes forms between PyCutFEM and FEniCSx on the smallest O-grid.

This helper isolates the velocity/pressure block on a 12-element O-grid so that
the assembled Jacobian (K matrix) and residual vectors can be inspected side by
side.  Run with

    PYTHONPATH=/home/bhatti/Documents/pycutfem conda run -n fenicsx \\
        python examples/debug/comparison_ns_ogrid.py
"""
from __future__ import annotations

import functools
import argparse
from pathlib import Path
from typing import Dict
import sympy as sp

import numpy as np
import pandas as pd
from mpi4py import MPI

import dolfinx
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
try:
    from dolfinx.io import gmshio
except ImportError:
    gmshio = None
    from dolfinx.io import gmsh
import ufl
import basix.ufl
import basix.cell
from basix.ufl import mixed_element
import meshio
import tempfile
import os
from petsc4py import PETSc
from scipy.sparse import csr_matrix

from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.gmsh_loader import mesh_from_gmsh
from pycutfem.utils.meshgen import structured_quad
from pycutfem.integration import volume
from pycutfem.fem.reference import get_reference
from pycutfem.fem import transform
from examples.turek_benchmark_volume_only import build_turek_channel_mesh
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    TrialFunction,
    TestFunction,
    VectorTrialFunction,
    VectorTestFunction,
    Function,
    VectorFunction,
    Constant,
    grad,
    inner,
    dot,
    div,
)
from pycutfem.ufl.measures import dx
from pycutfem.ufl.forms import Equation, assemble_form


def debug_interpolate(self, f):
    """
    Evaluate callable ``f`` at all DoF points of ``self`` without relying on a
    VTK backend.  Added as ``Function.debug_interpolate``.
    """
    fs = self.function_space
    try:
        coords = fs.tabulate_dof_coordinates()
    except RuntimeError:
        subspace, _ = fs.collapse()
        coords = subspace.tabulate_dof_coordinates()
    if coords.shape[0] == 0:
        return np.array([], dtype=np.float64)
    values = np.asarray(f(coords.T))
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values.T.flatten()


dolfinx.fem.Function.debug_interpolate = debug_interpolate

OUTPUT_DIR = Path("garbage/ns_ogrid")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Geometry parameters for the minimal O-grid (12 quadratic elements).
OGRID_PARAMS = dict(
    Lx=1.0,
    Ly=1.0,
    circle_center=(0.5, 0.5),
    circle_radius=0.2,
    ring_thickness=0.1,
    n_radial_layers=1,
    nx_outer=(1, 1, 1),
    ny_outer=(1, 1, 1),
)
POLY_ORDER = 2


def build_minimal_ogrid_mesh(mesh_size: float = 0.05) -> tuple[Mesh, np.ndarray, dolfinx.mesh.Mesh]:
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        build_turek_channel_mesh(tmp_path, mesh_size, cell_type="quad", view_mesh=False)
        mesh_pc = mesh_from_gmsh(tmp_path)
        if gmshio is not None:
            mesh_fx, _, _ = gmshio.read_from_msh(str(tmp_path), MPI.COMM_WORLD, gdim=2)
        else:
            meshdata_fx = gmsh.read_from_msh(str(tmp_path), MPI.COMM_WORLD, gdim=2)
            mesh_fx = meshdata_fx.mesh
    finally:
        os.remove(tmp_path)
    coords = np.array([[node.x, node.y] for node in mesh_pc.nodes_list], dtype=np.float64)
    mesh_pc.tag_boundary_edges({"all": lambda _x, _y: True})
    return mesh_pc, coords, mesh_fx


def _warp_nonaffine(xy: np.ndarray) -> np.ndarray:
    """Small smooth warp to make the mapping curved (not affine)."""
    x, y = xy[..., 0], xy[..., 1]
    f1 = 0.2 * (x * (1.0 - x)) * (y + 0.5 * y * y)
    f2 = 0.15 * (y * (1.0 - y)) * (x + 0.25 * x * x)
    out = np.empty_like(xy)
    out[..., 0] = x + f1
    out[..., 1] = y + f2
    return out


def build_single_structured_mesh(poly_order: int, *, curved: bool, seed: int | None = None, coords_override: np.ndarray | None = None) -> tuple[Mesh, dolfinx.mesh.Mesh]:
    # Start from unit quad then optionally random affine and/or warp
    nodes, elements, edges, corners = structured_quad(
        1.0,
        1.0,
        nx=1,
        ny=1,
        poly_order=poly_order,
    )
    coords = np.array([[n.x, n.y] for n in nodes], dtype=np.float64)

    if seed is not None:
        rng = np.random.default_rng(seed)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        rot = np.array(
            ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))),
            dtype=np.float64,
        )
        shear = np.array(((1.0, rng.uniform(-0.2, 0.2)), (0.0, 1.0)), dtype=np.float64)
        shift = rng.uniform(-0.25, 0.25, size=2)
        coords = (coords - 0.5) @ (rot @ shear).T + shift

    if coords_override is not None:
        coords = np.asarray(coords_override, dtype=float)
    elif curved:
        coords = _warp_nonaffine(coords)

    for node, xy in zip(nodes, coords):
        node.x = float(xy[0])
        node.y = float(xy[1])

    mesh_pc = Mesh(nodes, elements, edges, corners, element_type="quad", poly_order=poly_order)
    mesh_pc.tag_boundary_edges({"all": lambda _x, _y: True})

    # Basix Q2 geometry node ordering: vertices (bl, br, tl, tr), edge mids (bottom, right, top, left), center.
    basix_perm = np.array([0, 2, 6, 8, 1, 5, 7, 3, 4], dtype=int)
    coords_fx = coords[basix_perm]
    cells_fx = np.arange(coords_fx.shape[0], dtype=np.int64).reshape(1, -1)
    geom_el = basix.ufl.element("Lagrange", "quadrilateral", poly_order, shape=(2,))
    mesh_fx = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells_fx, geom_el, coords_fx)
    return mesh_pc, mesh_fx


def random_q2_coords(scale: float = 0.25, seed: int | None = None) -> np.ndarray:
    """
    Start from the unit Q2 grid and jitter mid-edge and center nodes by `scale`.
    Corner nodes remain fixed to preserve orientation.
    """
    rng = np.random.default_rng(seed)
    base = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0, 0.5],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    jitter = np.zeros_like(base)
    jitter[1:] = scale * rng.uniform(-1.0, 1.0, size=jitter[1:].shape)
    return base + jitter


def setup_ns_problems(mesh_kind: str = "ogrid", seed: int | None = None):
    if mesh_kind == "ogrid":
        mesh_pc, coords_unused, mesh_fx = build_minimal_ogrid_mesh()
    elif mesh_kind == "single":
        mesh_pc, mesh_fx = build_single_structured_mesh(POLY_ORDER, curved=False, seed=seed)
    elif mesh_kind == "curved":
        mesh_pc, mesh_fx = build_single_structured_mesh(POLY_ORDER, curved=True, seed=None)
    else:
        raise ValueError(f"Unsupported mesh kind '{mesh_kind}'")
    mixed_element_pc = MixedElement(mesh_pc, field_specs={"ux": 2, "uy": 2, "p": 1})
    dof_handler = DofHandler(mixed_element_pc, method="cg")

    velocity_fs = FunctionSpace("velocity", ["ux", "uy"], dim=1)
    pressure_fs = FunctionSpace("pressure", ["p"], dim=0)
    pc = {
        "du": VectorTrialFunction(velocity_fs, dof_handler=dof_handler),
        "dp": TrialFunction(pressure_fs, dof_handler=dof_handler),
        "v": VectorTestFunction(velocity_fs, dof_handler=dof_handler),
        "q": TestFunction(pressure_fs, dof_handler=dof_handler),
        "u_k": VectorFunction("u_k", ["ux", "uy"], dof_handler),
        "p_k": Function("p_k", "p", dof_handler),
        "u_n": VectorFunction("u_n", ["ux", "uy"], dof_handler),
        "rho": Constant(1.0, dim=0),
        "dt": Constant(0.1, dim=0),
        "theta": Constant(0.5, dim=0),
        "mu": Constant(1.0e-2, dim=0),
    }

    tdim = mesh_fx.topology.dim
    P2_el = basix.ufl.element("Lagrange", "quadrilateral", 2, shape=(tdim,))
    P1_el = basix.ufl.element("Lagrange", "quadrilateral", 1)
    W_el = mixed_element([P2_el, P1_el])
    W = dolfinx.fem.functionspace(mesh_fx, W_el)

    fenicsx = {
        "mesh": mesh_fx,
        "W": W,
        "rho": dolfinx.fem.Constant(mesh_fx, 1.0),
        "dt": dolfinx.fem.Constant(mesh_fx, 0.1),
        "theta": dolfinx.fem.Constant(mesh_fx, 0.5),
        "mu": dolfinx.fem.Constant(mesh_fx, 1.0e-2),
        "u_k_p_k": dolfinx.fem.Function(W, name="u_k_p_k"),
        "normal": ufl.FacetNormal(mesh_fx),
    }
    V, _ = W.sub(0).collapse()
    fenicsx["u_n"] = dolfinx.fem.Function(V, name="u_n")
    return pc, dof_handler, fenicsx


def setup_from_meshes(mesh_pc: Mesh, mesh_fx: dolfinx.mesh.Mesh):
    mixed_element_pc = MixedElement(mesh_pc, field_specs={"ux": 2, "uy": 2, "p": 1})
    dof_handler = DofHandler(mixed_element_pc, method="cg")

    velocity_fs = FunctionSpace("velocity", ["ux", "uy"], dim=1)
    pressure_fs = FunctionSpace("pressure", ["p"], dim=0)
    pc = {
        "du": VectorTrialFunction(velocity_fs, dof_handler=dof_handler),
        "dp": TrialFunction(pressure_fs, dof_handler=dof_handler),
        "v": VectorTestFunction(velocity_fs, dof_handler=dof_handler),
        "q": TestFunction(pressure_fs, dof_handler=dof_handler),
        "u_k": VectorFunction("u_k", ["ux", "uy"], dof_handler),
        "p_k": Function("p_k", "p", dof_handler),
        "u_n": VectorFunction("u_n", ["ux", "uy"], dof_handler),
        "rho": Constant(1.0, dim=0),
        "dt": Constant(0.1, dim=0),
        "theta": Constant(0.5, dim=0),
        "mu": Constant(1.0e-2, dim=0),
    }

    tdim = mesh_fx.topology.dim
    P2_el = basix.ufl.element("Lagrange", "quadrilateral", 2, shape=(tdim,))
    P1_el = basix.ufl.element("Lagrange", "quadrilateral", 1)
    W_el = mixed_element([P2_el, P1_el])
    W = dolfinx.fem.functionspace(mesh_fx, W_el)

    fenicsx = {
        "mesh": mesh_fx,
        "W": W,
        "rho": dolfinx.fem.Constant(mesh_fx, 1.0),
        "dt": dolfinx.fem.Constant(mesh_fx, 0.1),
        "theta": dolfinx.fem.Constant(mesh_fx, 0.5),
        "mu": dolfinx.fem.Constant(mesh_fx, 1.0e-2),
        "u_k_p_k": dolfinx.fem.Function(W, name="u_k_p_k"),
        "normal": ufl.FacetNormal(mesh_fx),
    }
    V, _ = W.sub(0).collapse()
    fenicsx["u_n"] = dolfinx.fem.Function(V, name="u_n")
    return pc, dof_handler, fenicsx


def initialize_state(pc: Dict, fenicsx: Dict):
    def u_k_init(x):
        return np.array([11.0 + x[0] * x[1], 33.0 + x[1]])

    def p_k_init(x):
        return np.sin(2.0 * np.pi * x[0] * x[1])

    def u_n_init(x):
        vals = u_k_init(x)
        return 0.5 * vals

    pc["u_k"].set_values_from_function(lambda x, y: u_k_init([x, y]))
    pc["p_k"].set_values_from_function(lambda x, y: p_k_init([x, y]))
    pc["u_n"].set_values_from_function(lambda x, y: u_n_init([x, y]))

    W = fenicsx["W"]
    u_k_p_k_fx = fenicsx["u_k_p_k"]
    u_n_fx = fenicsx["u_n"]

    u_k_fx = u_k_p_k_fx.sub(0)
    p_k_fx = u_k_p_k_fx.sub(1)
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()

    u_k_values = u_k_fx.debug_interpolate(u_k_init)
    u_k_p_k_fx.x.array[V_to_W] = u_k_values
    p_k_values = p_k_fx.debug_interpolate(p_k_init)
    u_k_p_k_fx.x.array[Q_to_W] = p_k_values
    u_k_p_k_fx.x.scatter_forward()

    u_n_values = u_n_fx.debug_interpolate(u_n_init)
    u_n_fx.x.array[:] = u_n_values
    u_n_fx.x.scatter_forward()

    np.testing.assert_allclose(
        np.sort(pc["u_k"].nodal_values),
        np.sort(u_k_values),
        rtol=1e-8,
        atol=1e-12,
    )


def get_pycutfem_dof_coords(dof_handler: DofHandler, field: str) -> np.ndarray:
    if field not in dof_handler.field_names:
        raise KeyError(f"Field '{field}' not present")
    return dof_handler.get_dof_coords(field)


def get_all_pycutfem_dof_coords(dof_handler: DofHandler) -> np.ndarray:
    coords = np.zeros((dof_handler.total_dofs, 2))
    for field in ["ux", "uy", "p"]:
        field_slice = dof_handler.get_field_slice(field)
        coords[field_slice] = get_pycutfem_dof_coords(dof_handler, field)
    return coords


def get_all_fenicsx_dof_coords(W):
    total = W.dofmap.index_map.size_global
    coords = np.zeros((total, 2))

    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()
    V0, V0_map = V.sub(0).collapse()
    V1, V1_map = V.sub(1).collapse()

    V_map = np.asarray(V_map, dtype=int)
    V0_map = np.asarray(V0_map, dtype=int)
    V1_map = np.asarray(V1_map, dtype=int)
    Q_map = np.asarray(Q_map, dtype=int)

    coords[V_map[V0_map]] = V0.tabulate_dof_coordinates()[:, :2]
    coords[V_map[V1_map]] = V1.tabulate_dof_coordinates()[:, :2]
    coords[Q_map] = Q.tabulate_dof_coordinates()[:, :2]
    return coords


def one_to_one_map_coords(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(coords2[:, None, :] - coords1[None, :, :], axis=2)
    rows, cols = scipy_linear_assignment(distances)
    return rows[np.argsort(cols)]


def scipy_linear_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.optimize import linear_sum_assignment

    return linear_sum_assignment(cost)


def create_true_dof_map(dof_handler: DofHandler, W) -> np.ndarray:
    pc_coords = {f: get_pycutfem_dof_coords(dof_handler, f) for f in ["ux", "uy", "p"]}
    pc_slices = {f: dof_handler.get_field_slice(f) for f in ["ux", "uy", "p"]}

    W0, V_map = W.sub(0).collapse()
    W1, Q_map = W.sub(1).collapse()
    W00, V0_map = W0.sub(0).collapse()
    W01, V1_map = W0.sub(1).collapse()

    coords_fx = {
        "ux": W00.tabulate_dof_coordinates()[:, :2],
        "uy": W01.tabulate_dof_coordinates()[:, :2],
        "p": W1.tabulate_dof_coordinates()[:, :2],
    }
    V_map = np.asarray(V_map, dtype=int)
    fx_dofs = {
        "ux": V_map[np.asarray(V0_map, dtype=int)],
        "uy": V_map[np.asarray(V1_map, dtype=int)],
        "p": np.asarray(Q_map, dtype=int),
    }

    mapping = np.zeros(dof_handler.total_dofs, dtype=int)
    for field in ["ux", "uy"]:
        perm = one_to_one_map_coords(pc_coords[field], coords_fx[field])
        mapping[pc_slices[field]] = fx_dofs[field][perm]
    perm_p = one_to_one_map_coords(pc_coords["p"], coords_fx["p"])
    mapping[pc_slices["p"]] = fx_dofs["p"][perm_p]
    return mapping


def element_area_pycutfem(mesh: Mesh, quad: int = 8) -> float:
    """Integrate |detJ| over the element using high quadrature."""
    ref = get_reference(mesh.element_type, mesh.poly_order)
    pts, wts = volume(mesh.element_type, quad)
    area = 0.0
    for eid in range(mesh.n_elements):
        for (xi, eta), w in zip(pts, wts):
            J = transform.jacobian(mesh, eid, (xi, eta))
            area += float(w) * abs(float(np.linalg.det(J)))
    return area


def area_from_mass_block(J: csr_matrix, rho: float, dt: float, n_comp: int = 2) -> float:
    """Recover area from the consistent mass matrix of an n-comp vector field."""
    total = J.toarray().sum()
    return total / (n_comp * rho / dt)


def exact_q2_area_from_coords(coords: np.ndarray) -> float:
    """
    Compute the exact area of a Q2 quad with node coordinates `coords`
    ordered row-major on the reference grid (-1,0,1)x(-1,0,1).
    """
    xi, eta = sp.symbols("xi eta")

    def lagrange_1d(u):
        return [u * (u - 1) / 2, 1 - u**2, u * (u + 1) / 2]

    Lx = lagrange_1d(xi)
    Ly = lagrange_1d(eta)
    N = []
    for j in range(3):
        for i in range(3):
            N.append(Lx[i] * Ly[j])

    coords = np.asarray(coords, dtype=float).reshape(9, 2)
    x_map = sum(N[k] * coords[k, 0] for k in range(9))
    y_map = sum(N[k] * coords[k, 1] for k in range(9))

    J = sp.Matrix(
        [
            [sp.diff(x_map, xi), sp.diff(y_map, xi)],
            [sp.diff(x_map, eta), sp.diff(y_map, eta)],
        ]
    )
    detJ = sp.simplify(J.det())
    area = sp.integrate(detJ, (xi, -1, 1), (eta, -1, 1))
    return float(area.evalf())


def navier_stokes_forms(pc: Dict, fenicsx: Dict):
    rho = pc["rho"]
    dt = pc["dt"]
    theta = pc["theta"]
    mu = pc["mu"]

    jacobian_pc = (
        rho * dot(pc["du"], pc["v"]) / dt
        + theta * rho * dot(dot(grad(pc["u_k"]), pc["du"]), pc["v"])
        + theta * rho * dot(dot(grad(pc["du"]), pc["u_k"]), pc["v"])
        + theta * mu * inner(grad(pc["du"]), grad(pc["v"]))
        - pc["dp"] * div(pc["v"])
        + pc["q"] * div(pc["du"])
    ) * dx()

    residual_pc = (
        rho * dot(pc["u_k"] - pc["u_n"], pc["v"]) / dt
        + theta * rho * dot(dot(grad(pc["u_k"]), pc["u_k"]), pc["v"])
        + (1.0 - theta) * rho * dot(dot(grad(pc["u_n"]), pc["u_n"]), pc["v"])
        + theta * mu * inner(grad(pc["u_k"]), grad(pc["v"]))
        + (1.0 - theta) * mu * inner(grad(pc["u_n"]), grad(pc["v"]))
        - pc["p_k"] * div(pc["v"])
        + pc["q"] * div(pc["u_k"])
    ) * dx()

    W = fenicsx["W"]
    u_k_fx, p_k_fx = ufl.split(fenicsx["u_k_p_k"])
    u_n_fx = fenicsx["u_n"]

    def jacobian_fx(deg):
        dup_fx, vq_fx = ufl.TrialFunction(W), ufl.TestFunction(W)
        du_fx, dp_fx = ufl.split(dup_fx)
        v_fx, q_fx = ufl.split(vq_fx)
        return (
            fenicsx["rho"] * ufl.dot(du_fx, v_fx) / fenicsx["dt"]
            + fenicsx["theta"] * fenicsx["rho"] * ufl.dot(ufl.dot(ufl.grad(u_k_fx), du_fx), v_fx)
            + fenicsx["theta"] * fenicsx["rho"] * ufl.dot(ufl.dot(ufl.grad(du_fx), u_k_fx), v_fx)
            + fenicsx["theta"] * fenicsx["mu"] * ufl.inner(ufl.grad(du_fx), ufl.grad(v_fx))
            - dp_fx * ufl.div(v_fx)
            + q_fx * ufl.div(du_fx)
        ) * ufl.dx(metadata={"quadrature_degree": deg})

    def residual_fx(deg):
        vq_fx = ufl.TestFunction(W)
        v_fx, q_fx = ufl.split(vq_fx)
        return (
            fenicsx["rho"] * ufl.dot(u_k_fx - u_n_fx, v_fx) / fenicsx["dt"]
            + fenicsx["theta"] * fenicsx["rho"] * ufl.dot(ufl.dot(ufl.grad(u_k_fx), u_k_fx), v_fx)
            + (1.0 - fenicsx["theta"]) * fenicsx["rho"] * ufl.dot(ufl.dot(ufl.grad(u_n_fx), u_n_fx), v_fx)
            + fenicsx["theta"] * fenicsx["mu"] * ufl.inner(ufl.grad(u_k_fx), ufl.grad(v_fx))
            + (1.0 - fenicsx["theta"]) * fenicsx["mu"] * ufl.inner(ufl.grad(u_n_fx), ufl.grad(v_fx))
            - p_k_fx * ufl.div(v_fx)
            + q_fx * ufl.div(u_k_fx)
        ) * ufl.dx(metadata={"quadrature_degree": deg})

    return jacobian_pc, residual_pc, jacobian_fx, residual_fx


def assemble_pycutfem(form, dof_handler: DofHandler, quad: int, matrix: bool, backend: str):
    if matrix:
        J_pc, _ = assemble_form(Equation(form, None), dof_handler, quad_degree=quad, backend=backend)
        return J_pc
    _, R_pc = assemble_form(Equation(None, form), dof_handler, quad_degree=quad, backend=backend)
    return R_pc


def assemble_fenicsx(f_form, matrix: bool):
    compiled = dolfinx.fem.form(f_form)
    if matrix:
        mat = dolfinx.fem.petsc.assemble_matrix(compiled)
        mat.assemble()
        indptr, indices, data = mat.getValuesCSR()
        return csr_matrix((data, indices, indptr), shape=mat.getSize())
    vec = dolfinx.fem.petsc.assemble_vector(compiled)
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.FORWARD)
    return vec.array


def compare_vectors(name: str, pc_vec: np.ndarray, fx_vec: np.ndarray, P: np.ndarray, dof_handler: DofHandler, W):
    pc_flat = pc_vec.flatten()
    fx_flat = fx_vec[P]
    diff = pc_flat - fx_flat
    pc_coords = get_all_pycutfem_dof_coords(dof_handler)
    fx_coords = get_all_fenicsx_dof_coords(W)[P]
    max_idx = int(np.argmax(np.abs(diff)))

    data = pd.DataFrame(
        {
            "pc_index": np.arange(pc_flat.size),
            "pc_x": pc_coords[:, 0],
            "pc_y": pc_coords[:, 1],
            "pc_value": pc_flat,
            "fx_x": fx_coords[:, 0],
            "fx_y": fx_coords[:, 1],
            "fx_value": fx_flat,
            "difference": diff,
        }
    )
    data.to_excel(OUTPUT_DIR / f"{name}_residual.xlsx", index=False)

    print(f"[{name}] residual L_inf={np.max(np.abs(diff)):.3e}, L2={np.linalg.norm(diff):.3e}")
    print(
        f"    max difference at dof {max_idx} (pc=({pc_coords[max_idx,0]:.4f},{pc_coords[max_idx,1]:.4f}), "
        f"fx=({fx_coords[max_idx,0]:.4f},{fx_coords[max_idx,1]:.4f}))"
    )


def compare_matrices(name: str, pc_mat: csr_matrix, fx_mat: csr_matrix, P: np.ndarray):
    J_pc = pc_mat.toarray()
    J_fx = fx_mat.toarray()[P][:, P]
    diff = J_pc - J_fx
    max_idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    np.savez(OUTPUT_DIR / f"{name}_jacobian.npz", pc=J_pc, fx=J_fx, diff=diff)
    print(
        f"[{name}] jacobian L_inf={np.max(np.abs(diff)):.3e}, Fro={np.linalg.norm(diff, 'fro'):.3e}, "
        f"max entry at {max_idx}"
    )


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", choices=("ogrid", "single", "curved"), default="ogrid", help="Mesh topology to compare.")
    p.add_argument("--seed", type=int, default=0, help="Seed for affine jitter on single/curved mesh.")
    p.add_argument("--quad-jac", type=int, default=5, help="Quadrature degree for Jacobian assembly.")
    p.add_argument("--quad-res", type=int, default=6, help="Quadrature degree for residual assembly.")
    p.add_argument("--backend", choices=("jit", "python"), default="jit", help="PyCutFEM assembly backend.")
    p.add_argument("--area-sweep", type=int, default=0, help="If >0, run this many random curved-element area diagnostics and exit.")
    p.add_argument("--perturb-scale", type=float, default=0.25, help="Max jitter for non-corner Q2 nodes in area sweep.")
    return p.parse_args()


def main():
    args = _parse_args()
    print(f"Running comparison on mesh='{args.mesh}' (seed={args.seed}, backend={args.backend})")
    if args.area_sweep > 0:
        rng = np.random.default_rng(args.seed)
        rows = []
        for i in range(args.area_sweep):
            coords = random_q2_coords(scale=args.perturb_scale, seed=int(rng.integers(0, 1_000_000)))
            area_exact = exact_q2_area_from_coords(coords)
            mesh_pc, mesh_fx = build_single_structured_mesh(POLY_ORDER, curved=True, coords_override=coords)
            pc, dof_handler, fenicsx = setup_from_meshes(mesh_pc, mesh_fx)
            initialize_state(pc, fenicsx)
            P_map = create_true_dof_map(dof_handler, fenicsx["W"])

            mass_form_pc = (pc["rho"] * dot(pc["du"], pc["v"]) / pc["dt"]) * dx()
            Dup_fx = ufl.TrialFunction(fenicsx["W"])
            Vq_fx = ufl.TestFunction(fenicsx["W"])
            du_fx, _ = ufl.split(Dup_fx)
            v_fx, _ = ufl.split(Vq_fx)
            mass_form_fx = (fenicsx["rho"] * ufl.dot(du_fx, v_fx) / fenicsx["dt"]) * ufl.dx(metadata={"quadrature_degree": args.quad_jac})
            J_pc = assemble_pycutfem(mass_form_pc, dof_handler, quad=args.quad_jac, matrix=True, backend="python")
            J_fx = assemble_fenicsx(mass_form_fx, matrix=True)
            area_pc = area_from_mass_block(J_pc, rho=float(pc["rho"].value), dt=float(pc["dt"].value), n_comp=2)
            area_fx = area_from_mass_block(J_fx, rho=1.0, dt=0.1, n_comp=2)
            rows.append((i, area_exact, area_pc, area_fx))
            print(f"[trial {i}] exact={area_exact:.12f}, mass_pc={area_pc:.12f}, mass_fx={area_fx:.12f}")
        return
    pc, dof_handler, fenicsx = setup_ns_problems(mesh_kind=args.mesh, seed=args.seed)
    initialize_state(pc, fenicsx)
    P_map = create_true_dof_map(dof_handler, fenicsx["W"])

    jacobian_pc, residual_pc, jacobian_fx, residual_fx = navier_stokes_forms(pc, fenicsx)

    if args.mesh == "curved":
        area_geom = element_area_pycutfem(dof_handler.mixed_element.mesh, quad=20)
        print(f"Geometric area (pycutfem, q=20): {area_geom:.12f}")

    print("Assembling Navier–Stokes Jacobian...")
    J_pc = assemble_pycutfem(jacobian_pc, dof_handler, quad=args.quad_jac, matrix=True, backend=args.backend)
    J_fx = assemble_fenicsx(jacobian_fx(args.quad_jac), matrix=True)
    compare_matrices("navier_stokes", J_pc, J_fx, P_map)
    if args.mesh == "curved":
        rho = float(pc["rho"].value)
        dt = float(pc["dt"].value)
        area_mass_pc = area_from_mass_block(J_pc, rho=rho, dt=dt, n_comp=2)
        area_mass_fx = area_from_mass_block(J_fx, rho=rho, dt=dt, n_comp=2)
        print(f"Area from mass (pc): {area_mass_pc:.12f}, (fenicsx): {area_mass_fx:.12f}")

    print("Assembling Navier–Stokes residual...")
    R_pc = assemble_pycutfem(residual_pc, dof_handler, quad=args.quad_res, matrix=False, backend=args.backend)
    R_fx = assemble_fenicsx(residual_fx(args.quad_res), matrix=False)
    compare_vectors("navier_stokes", R_pc, R_fx, P_map, dof_handler, fenicsx["W"])


if __name__ == "__main__":
    main()
