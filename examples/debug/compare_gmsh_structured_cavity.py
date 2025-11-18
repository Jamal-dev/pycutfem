#!/usr/bin/env python
"""
Diagnostic script that assembles the cavity Jacobian on two meshes:
    1. Structured Q2 quad mesh generated internally.
    2. Gmsh-generated quad mesh imported via mesh_from_gmsh.

It prints norms of the difference between the resulting global matrices to spot
geometric translation issues.
"""
from __future__ import annotations

import numpy as np
from collections import Counter
from pathlib import Path

from examples.gmsh_cavity_mesh import build_cavity_quad_mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import (
    Constant,
    Function,
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
from pycutfem.ufl.forms import BoundaryCondition, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dx
from pycutfem.utils.gmsh_loader import mesh_from_gmsh
from pycutfem.utils.meshgen import structured_quad


def _build_structured_mesh(nx: int, ny: int) -> Mesh:
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=2)
    return Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=2)


def _sort_coords(coords: np.ndarray) -> np.ndarray:
    view = coords.view([("x", coords.dtype), ("y", coords.dtype)])
    order = np.argsort(view, order=("x", "y"))
    return coords[order]


def _coord_key(pt: np.ndarray, ndigits: int = 12) -> tuple[float, float]:
    return (round(float(pt[0]), ndigits), round(float(pt[1]), ndigits))


def _assemble_cavity_matrix(mesh: Mesh):
    mixed_element = MixedElement(mesh, field_specs={"ux": 2, "uy": 2, "p": 1})
    dof_handler = DofHandler(mixed_element, method="cg")

    L = H = 1.0
    geom_tol = 1e-6
    bc_tags = {
        "bottom_wall": lambda x, y, tol=geom_tol: np.isclose(y, 0.0, atol=tol),
        "left_wall": lambda x, y, tol=geom_tol: np.isclose(x, 0.0, atol=tol),
        "right_wall": lambda x, y, tol=geom_tol: np.isclose(x, L, atol=tol),
        "top_lid": lambda x, y, tol=geom_tol: np.isclose(y, H, atol=tol),
    }
    mesh.tag_boundary_edges(bc_tags)
    counter = Counter(edge.tag for edge in mesh.edges_list if edge.right is None)
    dof_handler.tag_dof_by_locator(
        tag="pressure_pin_point",
        field="p",
        locator=lambda x, y, tol=geom_tol: np.isclose(x, 0.0, atol=tol) and np.isclose(y, 0.0, atol=tol),
        find_first=True,
    )

    bcs = [
        BoundaryCondition("ux", "dirichlet", "bottom_wall", lambda x, y: 0.0),
        BoundaryCondition("uy", "dirichlet", "bottom_wall", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "left_wall", lambda x, y: 0.0),
        BoundaryCondition("uy", "dirichlet", "left_wall", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "right_wall", lambda x, y: 0.0),
        BoundaryCondition("uy", "dirichlet", "right_wall", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "top_lid", lambda x, y: 1.0),
        BoundaryCondition("uy", "dirichlet", "top_lid", lambda x, y: 0.0),
    ]
    bcs_homog = [
        BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs
    ]

    velocity_space = FunctionSpace("velocity", ["ux", "uy"])
    pressure_space = FunctionSpace("pressure", ["p"])

    du = VectorTrialFunction(velocity_space, dof_handler=dof_handler)
    dp = TrialFunction(pressure_space, dof_handler=dof_handler)
    v = VectorTestFunction(velocity_space, dof_handler=dof_handler)
    q = TestFunction(pressure_space, dof_handler=dof_handler)

    u_k = VectorFunction(name="u_k", field_names=["ux", "uy"], dof_handler=dof_handler)
    u_n = VectorFunction(name="u_n", field_names=["ux", "uy"], dof_handler=dof_handler)
    p_k = Function(name="p_k", field_name="p", dof_handler=dof_handler)
    p_n = Function(name="p_n", field_name="p", dof_handler=dof_handler)

    for func in (u_k, u_n):
        func.nodal_values.fill(0.0)
    p_k.nodal_values.fill(0.0)
    p_n.nodal_values.fill(0.0)
    dof_handler.apply_bcs(bcs, u_n, p_n)

    rho = Constant(1.0)
    dt = Constant(0.1)
    theta = Constant(1.0)
    mu = Constant(0.01)

    jacobian_form = (
        rho * dot(du, v) / dt
        + theta * rho * dot(dot(grad(u_k), du), v)
        + theta * rho * dot(dot(grad(du), u_k), v)
        + theta * mu * inner(grad(du), grad(v))
        - dp * div(v)
        + q * div(du)
    ) * dx()

    residual_form = (
        rho * dot(u_k - u_n, v) / dt
        + theta * rho * dot(dot(grad(u_k), u_k), v)
        + (1.0 - theta) * rho * dot(dot(grad(u_n), u_n), v)
        + theta * mu * inner(grad(u_k), grad(v))
        + (1.0 - theta) * mu * inner(grad(u_n), grad(v))
        - p_k * div(v)
        + q * div(u_k)
    ) * dx()

    dirichlet_data = dof_handler.get_dirichlet_data(bcs)
    K, _ = assemble_form(
        jacobian_form == residual_form,
        dof_handler=dof_handler,
        mixed_element=mixed_element,
        bcs=bcs,
        bcs_homog=bcs_homog,
    )
    return K.tocsr(), dof_handler, len(dirichlet_data), counter


def main():
    nx = ny = 2  # => four elements
    gmsh_path = Path("examples/meshes/cavity_quad_test.msh")
    build_cavity_quad_mesh(gmsh_path, nx=nx, ny=ny, element_order=2)

    mesh_struct = _build_structured_mesh(nx, ny)
    mesh_gmsh = mesh_from_gmsh(gmsh_path)

    K_struct, dh_struct, n_dir_struct, counter_struct = _assemble_cavity_matrix(mesh_struct)
    K_gmsh, dh_gmsh, n_dir_gmsh, counter_gmsh = _assemble_cavity_matrix(mesh_gmsh)

    mapping = {
        (field, _coord_key(dh_gmsh._dof_coords[gid])): gid
        for gid, (field, _) in dh_gmsh._dof_to_node_map.items()
    }
    if len(mapping) != dh_gmsh.total_dofs:
        raise RuntimeError("Failed to build a one-to-one DOF permutation between meshes.")
    perm = np.empty(dh_struct.total_dofs, dtype=int)
    for gid in range(dh_struct.total_dofs):
        field = dh_struct._dof_to_node_map[gid][0]
        key = (field, _coord_key(dh_struct._dof_coords[gid]))
        perm[gid] = mapping[key]
    K_gmsh_aligned = K_gmsh[perm][:, perm]
    diff = K_struct - K_gmsh_aligned
    diff_norm = np.linalg.norm(diff.data, ord=np.inf) if diff.nnz else 0.0

    coords_struct = dh_struct.get_dof_coords("ux")
    coords_gmsh = dh_gmsh.get_dof_coords("ux")
    delta_coords = _sort_coords(coords_struct) - _sort_coords(coords_gmsh)
    coord_diff = float(np.max(np.abs(delta_coords)))
    order_mismatch = float(np.max(np.abs(coords_struct - coords_gmsh)))

    print("Structured mesh size:", K_struct.shape, "nnz:", K_struct.nnz, "Dirichlet DOFs:", n_dir_struct)
    print("Gmsh mesh size      :", K_gmsh.shape, "nnz:", K_gmsh.nnz, "Dirichlet DOFs:", n_dir_gmsh)
    print(f"Matrix difference   : max |Δ| = {diff_norm:.3e}")
    print(f"DOF coordinate diff : {coord_diff:.3e}")
    print(f"DOF order difference: {order_mismatch:.3e}")
    print("Structured boundary edges:", counter_struct)
    print("Gmsh boundary edges      :", counter_gmsh)


if __name__ == "__main__":
    main()
