import numpy as np


def test_newton_trace_with_hanging_constraints(monkeypatch, tmp_path):
    """
    Regression test: enabling residual/newton tracing must not crash when the solver
    runs in master DOF space (hanging-node constraints present).
    """
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))
    monkeypatch.setenv("PYCUTFEM_NEWTON_TRACE_WORST", "1")
    monkeypatch.setenv("PYCUTFEM_RESIDUAL_TRACE", "1")
    monkeypatch.setenv("PYCUTFEM_RESIDUAL_TRACE_CLASSIFY", "1")

    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.core.mesh import Mesh
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
    from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction
    from pycutfem.ufl.measures import dx
    from pycutfem.utils.meshgen import structured_quad
    from pycutfem.utils.refinement import TensorRefiner

    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=1)
    mesh0 = Mesh(
        nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    # Refine only the left element in y (creates a hanging node on the shared interface).
    refiner = TensorRefiner(max_ref=3)
    rx = np.zeros((len(mesh0.elements_list),), dtype=int)
    ry = np.zeros_like(rx)
    ry[0] = 1
    mesh = refiner.refine(mesh0, rx, ry)

    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    u_k = Function(name="u_k", field_name="u", dof_handler=dh)
    u_n = Function(name="u_n", field_name="u", dof_handler=dh)
    v = TestFunction(field_name="u", dof_handler=dh)
    u = TrialFunction(field_name="u", dof_handler=dh)

    residual_form = (u_k + u_k * u_k * u_k - Constant(1.0)) * v * dx()
    jacobian_form = ((Constant(1.0) + 3.0 * u_k * u_k) * u * v) * dx()

    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-10, max_newton_iter=25, line_search=True),
        backend="jit",
    )
    assert solver.constraints is not None, "expected hanging-node constraints to be detected"

    u_n.nodal_values[:] = 0.0
    u_k.nodal_values[:] = 0.0
    solver.solve_time_interval(
        functions=[u_k],
        prev_functions=[u_n],
        time_params=TimeStepperParameters(dt=1.0, max_steps=1, stop_on_steady=True),
    )

    _, R = solver._assemble_system({"u_k": u_k, "u_n": u_n}, need_matrix=False)
    assert float(np.linalg.norm(np.asarray(R, dtype=float), ord=np.inf)) < 1.0e-8
