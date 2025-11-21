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
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from mpi4py import MPI

import dolfinx
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
from dolfinx.io import gmshio
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
        mesh_fx, _, _ = gmshio.read_from_msh(str(tmp_path), MPI.COMM_WORLD, gdim=2)
    finally:
        os.remove(tmp_path)
    coords = np.array([[node.x, node.y] for node in mesh_pc.nodes_list], dtype=np.float64)
    mesh_pc.tag_boundary_edges({"all": lambda _x, _y: True})
    return mesh_pc, coords, mesh_fx


def setup_ns_problems():
    mesh_pc, coords, mesh_fx = build_minimal_ogrid_mesh()
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


def assemble_pycutfem(form, dof_handler: DofHandler, quad: int, matrix: bool):
    if matrix:
        J_pc, _ = assemble_form(Equation(form, None), dof_handler, quad_degree=quad, backend="jit")
        return J_pc
    _, R_pc = assemble_form(Equation(None, form), dof_handler, quad_degree=quad, backend="jit")
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


def main():
    pc, dof_handler, fenicsx = setup_ns_problems()
    initialize_state(pc, fenicsx)
    P_map = create_true_dof_map(dof_handler, fenicsx["W"])

    jacobian_pc, residual_pc, jacobian_fx, residual_fx = navier_stokes_forms(pc, fenicsx)

    print("Assembling Navier–Stokes Jacobian...")
    J_pc = assemble_pycutfem(jacobian_pc, dof_handler, quad=5, matrix=True)
    J_fx = assemble_fenicsx(jacobian_fx(5), matrix=True)
    compare_matrices("navier_stokes", J_pc, J_fx, P_map)

    print("Assembling Navier–Stokes residual...")
    R_pc = assemble_pycutfem(residual_pc, dof_handler, quad=6, matrix=False)
    R_fx = assemble_fenicsx(residual_fx(6), matrix=False)
    compare_vectors("navier_stokes", R_pc, R_fx, P_map, dof_handler, fenicsx["W"])


if __name__ == "__main__":
    main()
