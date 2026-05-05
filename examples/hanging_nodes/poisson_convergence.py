#!/usr/bin/env python
from __future__ import annotations

import math

import numpy as np
import scipy.sparse.linalg as spla

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import TestFunction, TrialFunction, grad, inner
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad
from pycutfem.utils.refinement import TensorRefiner


def _tag_unit_square_boundaries(mesh: Mesh, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - 1.0) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - 1.0) <= tol,
        }
    )


def make_left_refined_hanging_mesh(nx: int, ny: int, *, poly_order: int) -> Mesh:
    nodes, elems, _edges, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=poly_order)
    mesh0 = Mesh(
        nodes,
        element_connectivity=elems,
        edges_connectivity=None,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    # Refine the left half by one extra level -> hanging interface at x=0.5
    cx = np.asarray([mesh0.nodes_x_y_pos[list(el.corner_nodes)][:, 0].mean() for el in mesh0.elements_list], dtype=float)
    marked = cx < 0.5
    rx = np.zeros((len(mesh0.elements_list),), dtype=int)
    ry = np.zeros_like(rx)
    rx[marked] = 1
    ry[marked] = 1

    refiner = TensorRefiner(max_ref=2)
    rx, ry = refiner.balance_levels(mesh0, rx, ry)
    mesh = refiner.refine(mesh0, rx, ry)
    _tag_unit_square_boundaries(mesh)
    return mesh


def solve_poisson_mms(mesh: Mesh, *, poly_order: int) -> float:
    me = MixedElement(mesh, field_specs={"u": poly_order})
    dh = DofHandler(me, method="cg")

    u = TrialFunction(field_name="u", dof_handler=dh)
    v = TestFunction(field_name="u", dof_handler=dh)

    def u_exact(x, y):
        return np.sin(math.pi * x) * np.sin(math.pi * y)

    def f_exact(x, y):
        return 2.0 * (math.pi**2) * np.sin(math.pi * x) * np.sin(math.pi * y)

    q = max(2 * poly_order + 4, 8)
    a = inner(grad(u), grad(v)) * dx(metadata={"q": q})
    L = Analytic(lambda x, y: f_exact(x, y), degree=6) * v * dx(metadata={"q": q})
    eq = Equation(a, L)

    bcs = [
        BoundaryCondition("u", "dirichlet", "left", lambda x, y: float(u_exact(x, y))),
        BoundaryCondition("u", "dirichlet", "right", lambda x, y: float(u_exact(x, y))),
        BoundaryCondition("u", "dirichlet", "bottom", lambda x, y: float(u_exact(x, y))),
        BoundaryCondition("u", "dirichlet", "top", lambda x, y: float(u_exact(x, y))),
    ]

    K, F = assemble_form(eq, dof_handler=dh, bcs=bcs, quad_order=q, backend="python")

    constraints = dh.build_hanging_node_constraints()
    if constraints is None or constraints.slaves.size == 0:
        raise RuntimeError("expected hanging-node constraints on this mesh")

    K_red = constraints.E_T @ (K @ constraints.E)
    F_red = constraints.E_T @ F
    u_red = spla.spsolve(K_red.tocsc(), np.asarray(F_red, dtype=float))
    u_full = constraints.prolong(u_red)

    return float(dh.l2_error(u_full, exact={"u": u_exact}, quad_order=max(2 * poly_order + 6, 10), relative=False))


if __name__ == "__main__":
    poly_order = 2
    Ns = (4, 8, 16)
    errs = []
    print(f"{'N':>5}  {'L2 error':>12}  {'rate':>6}")
    prev = None
    for N in Ns:
        mesh = make_left_refined_hanging_mesh(N, N, poly_order=poly_order)
        err = solve_poisson_mms(mesh, poly_order=poly_order)
        rate = ""
        if prev is not None:
            rate_val = math.log(prev / err) / math.log(2.0)
            rate = f"{rate_val:6.3f}"
        print(f"{N:5d}  {err:12.4e}  {rate}")
        errs.append(err)
        prev = err
