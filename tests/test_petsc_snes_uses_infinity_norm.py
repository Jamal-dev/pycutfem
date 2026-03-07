from __future__ import annotations

import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers import nonlinear_solver as nls
from pycutfem.solvers.nonlinear_solver import NewtonParameters, PetscSnesNewtonSolver, TimeStepperParameters
from pycutfem.ufl.expressions import Function, TestFunction, TrialFunction
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad

pytestmark = pytest.mark.skipif(
    not nls.HAS_PETSC, reason="petsc4py not available"
)


def test_petsc_snes_converges_on_infinity_norm() -> None:
    """
    Regression test: SNES should interpret `newton_tol` like the Python Newton path,
    i.e. as a max-entry (infinity-norm) threshold on the residual.

    We construct a linear problem with many DOFs so that:
      ‖R‖_inf < atol  but  ‖R‖_2 > atol

    Under an infinity-norm convergence check, SNES should converge immediately
    (0 iterations) and leave the initial guess unchanged.
    """
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=10, ny=10, poly_order=1)
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

    residual_form = u_k * v * dx()
    jacobian_form = u * v * dx()

    u0 = 1.0e-4
    atol = 5.0e-6

    solver = PetscSnesNewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(
            newton_tol=float(atol),
            newton_rtol=0.0,
            max_newton_iter=5,
            line_search=True,
        ),
        backend="jit",
    )

    u_n.nodal_values[:] = float(u0)
    u_k.nodal_values[:] = float(u0)
    solver.solve_time_interval(
        functions=[u_k],
        prev_functions=[u_n],
        time_params=TimeStepperParameters(dt=1.0, max_steps=1, stop_on_steady=True),
    )

    # If convergence used the 2-norm, the linear solve would drive u -> 0 in one iteration.
    # With the infinity-norm check, we converge at the initial guess (iteration 0).
    assert np.max(np.abs(np.asarray(u_k.nodal_values, dtype=float))) == pytest.approx(float(u0))

    snes = getattr(solver, "_snes", None)
    assert snes is not None
    assert int(snes.getIterationNumber()) == 0

