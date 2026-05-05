import numpy as np


def test_solve_time_interval_caps_last_dt_to_hit_final_time():
    """
    Regression test: solve_time_interval must not overshoot final_time when dt does not
    divide the interval exactly.
    """
    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.core.mesh import Mesh
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
    from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction
    from pycutfem.ufl.measures import dx
    from pycutfem.utils.meshgen import structured_quad

    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    u_k = Function(name="u_k", field_name="u", dof_handler=dh)
    u_n = Function(name="u_n", field_name="u", dof_handler=dh)
    v = TestFunction(field_name="u", dof_handler=dh)
    u = TrialFunction(field_name="u", dof_handler=dh)

    residual_form = (u_k - Constant(1.0)) * v * dx()
    jacobian_form = (u * v) * dx()

    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-12, max_newton_iter=5, line_search=False),
        backend="python",
    )

    u_n.nodal_values[:] = 0.0
    u_k.nodal_values[:] = 0.0

    times: list[float] = []

    def _post_step(_funcs):
        times.append(float(getattr(solver, "_current_t", 0.0)) + float(getattr(solver, "_current_dt", 0.0)))

    solver.post_timeloop_cb = _post_step
    solver.solve_time_interval(
        functions=[u_k],
        prev_functions=[u_n],
        time_params=TimeStepperParameters(dt=0.3, max_steps=100, stop_on_steady=False, final_time=1.0),
    )

    assert len(times) >= 2
    assert float(np.max(times)) <= 1.0 + 1.0e-12
    assert abs(float(times[-1]) - 1.0) <= 1.0e-12
    assert abs(float(times[-1] - times[-2]) - 0.1) <= 1.0e-12


def test_solve_time_interval_does_not_take_tiny_last_step_due_to_fp_drift():
    """
    Regression test: when dt should hit final_time exactly in real arithmetic,
    floating-point drift (e.g. sum([0.05]*10) = 0.49999999999999994) must NOT
    trigger an extra "tiny" step of size ~1e-16.
    """
    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.core.mesh import Mesh
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
    from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction
    from pycutfem.ufl.measures import dx
    from pycutfem.utils.meshgen import structured_quad

    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    u_k = Function(name="u_k", field_name="u", dof_handler=dh)
    u_n = Function(name="u_n", field_name="u", dof_handler=dh)
    v = TestFunction(field_name="u", dof_handler=dh)
    u = TrialFunction(field_name="u", dof_handler=dh)

    residual_form = (u_k - Constant(1.0)) * v * dx()
    jacobian_form = (u * v) * dx()

    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-12, max_newton_iter=5, line_search=False),
        backend="python",
    )

    u_n.nodal_values[:] = 0.0
    u_k.nodal_values[:] = 0.0

    dt = 0.05
    final_time = 0.5
    expected_steps = int(round(final_time / dt))

    times: list[float] = []

    def _post_step(_funcs):
        times.append(float(getattr(solver, "_current_t", 0.0)) + float(getattr(solver, "_current_dt", 0.0)))

    solver.post_timeloop_cb = _post_step
    solver.solve_time_interval(
        functions=[u_k],
        prev_functions=[u_n],
        time_params=TimeStepperParameters(dt=dt, max_steps=100, stop_on_steady=False, final_time=final_time),
    )

    assert len(times) == expected_steps
    assert float(np.max(times)) <= final_time + 1.0e-12
    assert abs(float(times[-1]) - final_time) <= 1.0e-12
    assert abs(float(times[-1] - times[-2]) - dt) <= 1.0e-12


def test_solve_time_interval_retry_keep_guess_does_not_reapply_step_initial_guess(monkeypatch):
    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.core.mesh import Mesh
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
    from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction
    from pycutfem.ufl.measures import dx
    from pycutfem.utils.meshgen import structured_quad

    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    u_k = Function(name="u_k", field_name="u", dof_handler=dh)
    u_n = Function(name="u_n", field_name="u", dof_handler=dh)
    v = TestFunction(field_name="u", dof_handler=dh)
    u = TrialFunction(field_name="u", dof_handler=dh)

    residual_form = (u_k - Constant(1.0)) * v * dx()
    jacobian_form = (u * v) * dx()

    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-12, max_newton_iter=5, line_search=False),
        backend="python",
    )

    u_n.nodal_values[:] = 0.0
    u_k.nodal_values[:] = 0.0

    state = {
        "guess_calls": 0,
        "newton_calls": 0,
        "seen": [],
    }

    def _step_initial_guess_callback(**kwargs):
        state["guess_calls"] += 1
        funcs = list(kwargs["functions"])
        funcs[0].nodal_values[:] = np.asarray(funcs[0].nodal_values, dtype=float) + 1.0

    def _newton_loop(funcs, prev_funcs, aux_funcs, bcs_now):
        del prev_funcs, aux_funcs, bcs_now
        state["newton_calls"] += 1
        vals = np.asarray(funcs[0].nodal_values, dtype=float).copy()
        state["seen"].append(vals.copy())
        assert np.allclose(vals, 1.0)
        if state["newton_calls"] == 1:
            raise RuntimeError("fail once")
        return np.zeros((vals.size,), dtype=float), True, 1

    def _on_step_failure(**kwargs):
        del kwargs
        return "retry_keep_guess"

    monkeypatch.setattr(solver, "_newton_loop", _newton_loop)

    solver.solve_time_interval(
        functions=[u_k],
        prev_functions=[u_n],
        time_params=TimeStepperParameters(
            dt=0.1,
            max_steps=4,
            stop_on_steady=False,
            final_time=0.1,
            step_initial_guess_callback=_step_initial_guess_callback,
            on_step_failure=_on_step_failure,
        ),
    )

    assert state["guess_calls"] == 1
    assert state["newton_calls"] == 2
    assert len(state["seen"]) == 2
    assert np.allclose(state["seen"][0], 1.0)
    assert np.allclose(state["seen"][1], 1.0)
