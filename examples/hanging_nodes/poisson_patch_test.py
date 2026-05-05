#!/usr/bin/env python
from __future__ import annotations

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import Function, TestFunction, TrialFunction, grad, inner
from pycutfem.ufl.forms import BoundaryCondition
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


def _make_single_interface_hanging_mesh(poly_order: int) -> Mesh:
    nodes, elems, _edges, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=poly_order)
    mesh0 = Mesh(
        nodes,
        element_connectivity=elems,
        edges_connectivity=None,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    # Refine only the left element in y -> hanging nodes on the interface at x=0.5
    refiner = TensorRefiner(max_ref=3)
    rx = np.zeros((len(mesh0.elements_list),), dtype=int)
    ry = np.zeros_like(rx)
    ry[0] = 1
    mesh = refiner.refine(mesh0, rx, ry)
    _tag_unit_square_boundaries(mesh)
    return mesh


def run(poly_order: int) -> None:
    mesh = _make_single_interface_hanging_mesh(poly_order)
    me = MixedElement(mesh, field_specs={"u": poly_order})
    dh = DofHandler(me, method="cg")

    p = int(poly_order)

    def u_exact(x, y):
        return x**p + y**p

    def f_exact(x, y):
        if p < 2:
            return 0.0 * x
        c = -float(p * (p - 1))
        return c * (x ** (p - 2) + y ** (p - 2))

    q = max(2 * p + 4, 8)

    u_k = Function(name="u_k", field_name="u", dof_handler=dh)
    u_n = Function(name="u_n", field_name="u", dof_handler=dh)
    du = TrialFunction(field_name="u", dof_handler=dh)
    v = TestFunction(field_name="u", dof_handler=dh)

    f_ana = Analytic(lambda x, y: f_exact(x, y), degree=max(1, p - 2))
    residual_form = (inner(grad(u_k), grad(v)) - f_ana * v) * dx(metadata={"q": q})
    jacobian_form = inner(grad(du), grad(v)) * dx(metadata={"q": q})

    bcs = [
        BoundaryCondition("u", "dirichlet", "left", lambda x, y: float(u_exact(x, y))),
        BoundaryCondition("u", "dirichlet", "right", lambda x, y: float(u_exact(x, y))),
        BoundaryCondition("u", "dirichlet", "bottom", lambda x, y: float(u_exact(x, y))),
        BoundaryCondition("u", "dirichlet", "top", lambda x, y: float(u_exact(x, y))),
    ]
    bcs_homog = [BoundaryCondition("u", "dirichlet", bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=1.0e-12, max_newton_iter=10, line_search=False),
        quad_order=q,
        backend="jit",
    )
    constraints = solver.constraints
    slaves = int(constraints.slaves.size) if constraints is not None else 0

    u_n.nodal_values[:] = 0.0
    u_k.nodal_values[:] = 0.0
    solver.solve_time_interval(
        functions=[u_k],
        prev_functions=[u_n],
        time_params=TimeStepperParameters(dt=1.0, max_steps=1, stop_on_steady=True),
    )

    xy = dh.get_dof_coords("u")
    u_exact_vals = u_exact(xy[:, 0], xy[:, 1])
    max_err = float(np.max(np.abs(u_k.nodal_values - u_exact_vals)))
    print(f"p={p}  slaves={slaves}  max|u_h-u|={max_err:.3e}")


if __name__ == "__main__":
    for p in (1, 2, 3):
        run(p)
