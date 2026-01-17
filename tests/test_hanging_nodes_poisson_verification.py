import math

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import Function, TestFunction as UFLTestFunction, TrialFunction, grad, inner
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


def _make_single_interface_hanging_mesh(poly_order: int, *, axis: str = "y") -> Mesh:
    """
    Small mesh with a single hanging interface.

    axis="y": refine left element in y → hanging nodes along a vertical interface
    axis="x": refine bottom element in x → hanging nodes along a horizontal interface
    """
    if axis == "y":
        nodes, elems, _edges, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=poly_order)
    elif axis == "x":
        nodes, elems, _edges, corners = structured_quad(1.0, 1.0, nx=1, ny=2, poly_order=poly_order)
    else:
        raise ValueError(f"unknown axis={axis!r}")

    mesh0 = Mesh(
        nodes,
        element_connectivity=elems,
        edges_connectivity=None,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    refiner = TensorRefiner(max_ref=3)
    rx = np.zeros((len(mesh0.elements_list),), dtype=int)
    ry = np.zeros_like(rx)
    if axis == "y":
        ry[0] = 1
    else:
        rx[0] = 1
    mesh = refiner.refine(mesh0, rx, ry)
    _tag_unit_square_boundaries(mesh)
    return mesh


@pytest.mark.parametrize("jit_backend", ("jit", "cpp"))
@pytest.mark.parametrize("poly_order", (1, 2, 3))
def test_newton_poisson_with_hanging_nodes_reproduces_polynomial(jit_backend, poly_order, monkeypatch, tmp_path):
    if jit_backend == "cpp":
        monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")
    else:
        monkeypatch.delenv("PYCUTFEM_JIT_BACKEND", raising=False)
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_{jit_backend}_p{poly_order}"))

    mesh = _make_single_interface_hanging_mesh(poly_order, axis="y")
    me = MixedElement(mesh, field_specs={"u": poly_order})
    dh = DofHandler(me, method="cg")

    # Exact solution u = x^p + y^p, f = -Δu
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
    v = UFLTestFunction(field_name="u", dof_handler=dh)

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
    assert solver.constraints is not None, "expected hanging-node constraints to be detected"
    assert solver.constraints.slaves.size > 0, "expected at least one slave DOF"

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
    assert max_err < 2.0e-8

    # Postcondition: the stored full-space vector satisfies the constraint relations.
    u_full = u_k.nodal_values
    for sdof, combo in solver.constraints.slave_to_master.items():
        approx = 0.0
        for mdof, w in combo:
            approx += float(w) * float(u_full[int(mdof)])
        assert abs(float(u_full[int(sdof)]) - approx) < 2.0e-9


def _make_left_refined_hanging_mesh(nx: int, ny: int, *, poly_order: int) -> Mesh:
    nodes, elems, _edges, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=poly_order)
    mesh0 = Mesh(
        nodes,
        element_connectivity=elems,
        edges_connectivity=None,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    # Refine the left half by one extra level to create a persistent hanging interface.
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


def _solve_poisson_mms(mesh: Mesh, *, poly_order: int) -> float:
    me = MixedElement(mesh, field_specs={"u": poly_order})
    dh = DofHandler(me, method="cg")

    u = TrialFunction(field_name="u", dof_handler=dh)
    v = UFLTestFunction(field_name="u", dof_handler=dh)

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
    assert constraints is not None, "expected hanging-node constraints in this benchmark mesh"
    assert constraints.slaves.size > 0

    K_red = constraints.E_T @ (K @ constraints.E)
    F_red = constraints.E_T @ F
    u_red = spla.spsolve(K_red.tocsc(), np.asarray(F_red, dtype=float))
    u_full = constraints.prolong(u_red)

    return float(dh.l2_error(u_full, exact={"u": u_exact}, quad_order=max(2 * poly_order + 6, 10), relative=False))


def test_smooth_poisson_converges_on_hanging_mesh():
    # Keep this small (yet meaningful): check that refining reduces the L2 error.
    poly_order = 2
    err = []
    for n in (4, 8):
        mesh = _make_left_refined_hanging_mesh(n, n, poly_order=poly_order)
        err.append(_solve_poisson_mms(mesh, poly_order=poly_order))

    # Expect a clear reduction when doubling resolution.
    assert err[1] < 0.25 * err[0]
