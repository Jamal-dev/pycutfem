#!/usr/bin/env python
"""
Poisson MMS with real hanging nodes produced by anisotropic refinement.

- Domain: [0, 1] x [0, 1]
- Exact: u = sin(pi x) sin(pi y), f = 2 pi^2 sin(pi x) sin(pi y)
- BC: Dirichlet from the exact solution on all boundaries

We start from a coarse Q2 mesh, solve, mark elements with the largest elementwise
L2 error, and refine them with orientation-dependent 1-to-2 splits. This
creates mismatched edge node sets across refined/unrefined neighbors, so
`DofHandler.build_hanging_node_constraints()` detects real hanging nodes and the
solver condenses them (E^T K E, E^T F). Each cycle prints the number of slaves,
masters, L2 error, and a continuity diagnostic (max nodal mismatch at identical
coordinates).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import scipy.sparse.linalg as spla

from pycutfem.core.topology import Node
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem import transform
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.integration.quadrature import volume as vol_rule
from pycutfem.io.visualization import plot_mesh_2
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    grad,
    inner,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad

# -----------------------------------------------------------------------------
# MMS helpers
# -----------------------------------------------------------------------------


def u_exact(x: float, y: float) -> float:
    return math.sin(math.pi * x) * math.sin(math.pi * y)


def f_exact(x: float, y: float) -> float:
    return 2.0 * (math.pi**2) * math.sin(math.pi * x) * math.sin(math.pi * y)


# -----------------------------------------------------------------------------
# Boundaries
# -----------------------------------------------------------------------------


def tag_rect_boundaries(mesh: Mesh, tol: float = 1e-12) -> None:
    xmin, ymin = mesh.nodes_x_y_pos.min(axis=0)
    xmax, ymax = mesh.nodes_x_y_pos.max(axis=0)
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - xmin) <= tol,
            "right": lambda x, y: abs(x - xmax) <= tol,
            "bottom": lambda x, y: abs(y - ymin) <= tol,
            "top": lambda x, y: abs(y - ymax) <= tol,
        }
    )


# -----------------------------------------------------------------------------
# Hanging-producing refinement: coarse vs fine connectivity
# -----------------------------------------------------------------------------


def collapse_for_coarse(conn: np.ndarray, *, collapse_right: bool = True, collapse_top: bool = False) -> np.ndarray:
    """
    Return a copy of a Q2 connectivity where selected mid-edge nodes are
    collapsed to corners to act as a 'coarse' edge (no midpoint DOF).
    """
    c = list(conn)
    if collapse_right:
        c[5] = c[2]  # right mid -> bottom-right corner
    if collapse_top:
        c[7] = c[8]  # top mid -> top-right corner
    return np.asarray(c, dtype=int)


def rebuild_mesh_with_hanging(nodes: list[Node], base_conn: np.ndarray, corners: np.ndarray, fine_set: set[int], poly_order: int) -> Mesh:
    """
    Build a mesh where elements in fine_set keep the original Q2 connectivity,
    and elements outside fine_set have selected mid-edge nodes collapsed.
    Hanging nodes arise where a fine edge (with a midpoint) meets a coarse edge
    (midpoint collapsed).
    """
    new_conn: list[np.ndarray] = []
    for eid, conn in enumerate(base_conn):
        if eid in fine_set:
            new_conn.append(np.asarray(conn, dtype=int))
        else:
            # Collapse the right edge midpoint; alternate by row to mix patterns.
            collapse_top = (eid % 2 == 0)
            new_conn.append(collapse_for_coarse(conn, collapse_right=True, collapse_top=collapse_top))

    mesh = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(new_conn, dtype=int),
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )
    tag_rect_boundaries(mesh)
    return mesh


# -----------------------------------------------------------------------------
# Solve utilities
# -----------------------------------------------------------------------------


@dataclass
class SolveResult:
    mesh: Mesh
    dof_handler: DofHandler
    solution: Function
    l2_error: float
    continuity_jump: float
    constraints_info: str
    elem_errors: np.ndarray


def _build_problem(mesh: Mesh, poly_order: int) -> tuple[DofHandler, Function, list[BoundaryCondition], list[BoundaryCondition]]:
    tag_rect_boundaries(mesh)
    me = MixedElement(mesh, {"u": poly_order})
    dh = DofHandler(me, method="cg")

    f_fun = Function(name="f_rhs", field_name="u", dof_handler=dh)
    gdofs = np.asarray(dh.get_field_slice("u"), int)
    xy = dh.get_dof_coords("u")
    rhs_vals = np.array([f_exact(float(x), float(y)) for x, y in xy], float)
    f_fun.set_nodal_values(gdofs, rhs_vals)

    bcs = [
        BoundaryCondition("u", "dirichlet", "left", lambda x, y: u_exact(x, y)),
        BoundaryCondition("u", "dirichlet", "right", lambda x, y: u_exact(x, y)),
        BoundaryCondition("u", "dirichlet", "bottom", lambda x, y: u_exact(x, y)),
        BoundaryCondition("u", "dirichlet", "top", lambda x, y: u_exact(x, y)),
    ]
    bcs_homog = [BoundaryCondition("u", "dirichlet", bc.domain_tag, lambda x, y: 0.0) for bc in bcs]
    return dh, f_fun, bcs, bcs_homog


def solve_once(mesh: Mesh, poly_order: int) -> SolveResult:
    dh, f_fun, bcs, _ = _build_problem(mesh, poly_order)

    u = Function(name="u_k", field_name="u", dof_handler=dh)
    du = TrialFunction("u", dof_handler=dh)
    v = TestFunction("u", dof_handler=dh)

    dx_all = dx(metadata={"q": 4 * poly_order})
    a = inner(grad(du), grad(v)) * dx_all
    r = (inner(grad(u), grad(v)) - f_fun * v) * dx_all

    eq = Equation(a, r)
    K, F = assemble_form(eq, dof_handler=dh, bcs=bcs, backend=os.getenv("BACKEND", "jit"))

    constraints = dh.build_hanging_node_constraints()
    constraint_msg = "[constraints] none"
    sol_full = None
    if constraints is not None:
        constraint_msg = f"[constraints] slaves={len(constraints.slave_to_master)} masters={constraints.n_master}"
        print(constraint_msg)
        K_red = constraints.E_T @ (K @ constraints.E)
        F_red = constraints.E_T @ F
        sol_red = spla.spsolve(K_red.tocsc(), F_red)
        sol_full = constraints.prolong(sol_red)
    else:
        sol_full = spla.spsolve(K.tocsc(), F)

    u.set_nodal_values(np.arange(len(sol_full), dtype=int), np.asarray(sol_full, float))

    # Elementwise L2 error
    qp, qw = vol_rule(mesh.element_type, max(4, 2 * poly_order + 2))
    elem_err = []
    for elem in mesh.elements_list:
        gd = dh.element_dofs("u", elem.id)
        coeffs = u.get_nodal_values(gd)
        loc = 0.0
        for (xi, eta), w in zip(qp, qw):
            J = transform.jacobian(mesh, elem.id, (float(xi), float(eta)))
            detJ = abs(float(np.linalg.det(J)))
            phi = dh.mixed_element.basis("u", float(xi), float(eta))[dh.mixed_element.slice("u")]
            uh = float(coeffs @ phi)
            xg = transform.x_mapping(mesh, elem.id, (float(xi), float(eta)))
            ue = u_exact(float(xg[0]), float(xg[1]))
            loc += w * detJ * (uh - ue) ** 2
        elem_err.append(loc)
    elem_err = np.asarray(elem_err, float)
    l2_err = math.sqrt(elem_err.sum())

    # Continuity check: max jump at identical coordinates across elements
    coord_to_vals = {}
    node_coords = mesh.nodes_x_y_pos
    for gdof, (fld, nid) in dh._dof_to_node_map.items():
        if fld != "u":
            continue
        key = (round(float(node_coords[int(nid), 0]), 12), round(float(node_coords[int(nid), 1]), 12))
        val = float(u.get_nodal_values(np.array([gdof], dtype=int))[0])
        coord_to_vals.setdefault(key, []).append(val)
    jumps = [abs(max(vs) - min(vs)) for vs in coord_to_vals.values() if len(vs) > 1]
    continuity_jump = max(jumps) if jumps else 0.0

    return SolveResult(
        mesh=mesh,
        dof_handler=dh,
        solution=u,
        l2_error=l2_err,
        continuity_jump=continuity_jump,
        constraints_info=constraint_msg,
        elem_errors=elem_err,
    )


# -----------------------------------------------------------------------------
# Adaptive loop
# -----------------------------------------------------------------------------


def adaptive_poisson(poly_order: int = 2, n_cycles: int = 3, mark_fraction: float = 0.3) -> None:
    nodes, elems, edges, corners = structured_quad(Lx=1.0, Ly=1.0, nx=3, ny=3, poly_order=poly_order)
    nodes_list = nodes
    base_conn = np.asarray(elems, dtype=int)
    fine_set: set[int] = set()
    mesh = rebuild_mesh_with_hanging(nodes_list, base_conn, np.asarray(corners, dtype=int), fine_set, poly_order)

    for cycle in range(n_cycles):
        print(f"\n=== Adaptive cycle {cycle} ===")
        result = solve_once(mesh, poly_order)
        print(f"L2 error = {result.l2_error:.3e}, continuity jump = {result.continuity_jump:.3e}")
        if constraints_info := result.constraints_info:
            print(constraints_info)

        if cycle == n_cycles - 1:
            break

        if cycle == 0:
            centroids = np.array([elem.centroid() for elem in mesh.elements_list])
            center = np.array([0.5, 0.5])
            marked_ids = {int(np.argmin(np.linalg.norm(centroids - center, axis=1)))}
        else:
            n_mark = max(1, int(mark_fraction * len(result.elem_errors)))
            marked_ids = set(np.argsort(result.elem_errors)[-n_mark:])

        fine_set |= marked_ids
        mesh = rebuild_mesh_with_hanging(nodes_list, base_conn, np.asarray(corners, dtype=int), fine_set, poly_order)

    if os.getenv("PLOT_FINAL", "0") != "0":
        plot_mesh_2(mesh, show=True, plot_nodes=False, elem_tags=False, edge_colors=True)


if __name__ == "__main__":
    adaptive_poisson()
