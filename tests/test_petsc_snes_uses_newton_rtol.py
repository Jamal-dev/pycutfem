from __future__ import annotations

import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers import nonlinear_solver as nls
from pycutfem.solvers.nonlinear_solver import NewtonParameters, PetscSnesNewtonSolver, TimeStepperParameters
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad

pytestmark = pytest.mark.skipif(
    not nls.HAS_PETSC, reason="petsc4py not available"
)


def test_petsc_snes_uses_newton_rtol() -> None:
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        poly_order=1,
        element_type="quad",
    )

    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    u_k = Function(name="u_k", field_name="u", dof_handler=dh)
    u_n = Function(name="u_n", field_name="u", dof_handler=dh)
    v = TestFunction(field_name="u", dof_handler=dh)
    u = TrialFunction(field_name="u", dof_handler=dh)

    residual_form = (u_k + u_k * u_k * u_k - Constant(1.0)) * v * dx()
    jacobian_form = ((Constant(1.0) + 3.0 * u_k * u_k) * u * v) * dx()

    newton_rtol = 1.0e-3
    solver = PetscSnesNewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(
            newton_tol=1.0e-10,
            newton_rtol=float(newton_rtol),
            max_newton_iter=5,
        ),
        backend="jit",
    )

    u_n.nodal_values[:] = 0.0
    u_k.nodal_values[:] = 0.0
    solver.solve_time_interval(
        functions=[u_k],
        prev_functions=[u_n],
        time_params=TimeStepperParameters(dt=1.0, max_steps=1, stop_on_steady=True),
    )

    snes = getattr(solver, "_snes", None)
    assert snes is not None
    _atol, rtol, _stol, _max_it, _max_funcs = snes.getTolerances()
    assert float(rtol) == pytest.approx(float(newton_rtol))

