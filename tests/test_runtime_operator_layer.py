from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from examples.NIRB import run_example2_local as example2_local
from pycutfem.operators import (
    CallbackFusedLocalAssemblyOperator,
    LocalAssemblyResult,
    LocalStateUpdate,
    OperatorManager,
    PointwiseQuadratureOperator,
    PointwiseQuadratureResult,
    PointwiseQuadratureWorkset,
    RuntimeOperator,
    SymbolicFusedLocalAssemblyOperator,
    SymbolicQuadratureStateUpdateSpec,
    SymbolicPointwiseNewtonOperator,
)
from pycutfem.solvers.nonlinear_solver import NewtonSolver
from pycutfem.state import QuadratureLayout, StateRegistry


class _LogOperator(RuntimeOperator):
    def __init__(self, label: str, log: list[tuple], *, residual_shift: float = 0.0) -> None:
        self.label = label
        self.log = log
        self.residual_shift = float(residual_shift)

    def bind(self, solver) -> None:
        self.log.append(("bind", self.label, solver))

    def on_step_begin(self, *, solver, functions, prev_functions, aux_functions, step: int, step_no: int, t: float, dt: float, bcs) -> None:
        del solver, functions, prev_functions, aux_functions, bcs
        self.log.append(("step_begin", self.label, int(step), int(step_no), float(t), float(dt)))

    def before_assembly(self, *, solver, coeffs, need_matrix: bool) -> None:
        del solver
        self.log.append(("before_assembly", self.label, bool(need_matrix), float(coeffs["marker"])))

    def after_assembly(self, *, solver, coeffs, A_red, R_red, need_matrix: bool):
        del solver, coeffs
        self.log.append(("after_assembly", self.label, bool(need_matrix)))
        return A_red, np.asarray(R_red, dtype=float) + self.residual_shift

    def on_nonlinear_iteration_begin(self, *, solver, functions, prev_functions, aux_functions, iteration: int, coeffs, bcs, metrics=None) -> None:
        del solver, functions, prev_functions, aux_functions, bcs, metrics
        self.log.append(("nonlinear_begin", self.label, int(iteration), float(coeffs["marker"])))

    def on_nonlinear_update(
        self,
        *,
        solver,
        functions,
        prev_functions,
        aux_functions,
        iteration: int,
        coeffs,
        delta_red=None,
        delta_full=None,
        bcs=None,
        metrics=None,
    ) -> None:
        del solver, functions, prev_functions, aux_functions, coeffs, delta_red, delta_full, bcs
        update_inf = 0.0 if metrics is None else float(metrics.get("update_inf", 0.0))
        self.log.append(("nonlinear_update", self.label, int(iteration), update_inf))

    def on_nonlinear_iteration_end(
        self,
        *,
        solver,
        functions,
        prev_functions,
        aux_functions,
        iteration: int,
        coeffs,
        converged: bool,
        bcs,
        metrics=None,
    ) -> None:
        del solver, functions, prev_functions, aux_functions, coeffs, bcs
        reason = None if metrics is None else metrics.get("reason")
        self.log.append(("nonlinear_end", self.label, int(iteration), bool(converged), reason))

    def on_step_accept(self, *, solver, functions, prev_functions, aux_functions, step: int, step_no: int, t: float, dt: float, bcs) -> None:
        del solver, functions, prev_functions, aux_functions, bcs
        self.log.append(("step_accept", self.label, int(step), int(step_no), float(t), float(dt)))

    def on_step_reject(
        self,
        *,
        solver,
        functions,
        prev_functions,
        aux_functions,
        step: int,
        step_no: int,
        t: float,
        dt: float,
        bcs,
        exception,
        reason: str | None,
    ) -> None:
        del solver, functions, prev_functions, aux_functions, bcs, exception
        self.log.append(("step_reject", self.label, int(step), int(step_no), float(t), float(dt), reason))


class _PointwiseProbeOperator(PointwiseQuadratureOperator):
    def __init__(self) -> None:
        self.applied_result = None

    def build_pointwise_workset(self, *, solver, coeffs, need_matrix: bool):
        return PointwiseQuadratureWorkset(
            solver=solver,
            coeffs=coeffs,
            need_matrix=bool(need_matrix),
            backend=str(getattr(solver, "backend", "python")),
            layout=QuadratureLayout(
                entity_kind="volume_cell",
                cell_type="tri",
                quadrature_order=1,
                reference_points=np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float),
                reference_weights=np.array([0.5], dtype=float),
            ),
            entity_ids=np.array([0, 2], dtype=int),
            payload={"marker": float(coeffs["marker"])},
        )

    def update_pointwise_python(self, workset: PointwiseQuadratureWorkset):
        marker = float(workset.payload["marker"])
        return np.full((workset.entity_ids.shape[0], workset.layout.n_qp), marker, dtype=float)

    def update_pointwise_cpp(self, workset: PointwiseQuadratureWorkset):
        marker = float(workset.payload["marker"])
        return PointwiseQuadratureResult(
            values=np.full((workset.entity_ids.shape[0], workset.layout.n_qp), -marker, dtype=float),
            metadata={"backend": "cpp"},
        )

    def apply_pointwise_result(self, *, solver, workset: PointwiseQuadratureWorkset, result: PointwiseQuadratureResult) -> None:
        del solver
        self.applied_result = (workset, result)


class _ReducedScatterStub:
    def __init__(self, *, backend: str = "python") -> None:
        self.backend = str(backend)

    def scatter_element_contribs_reduced(
        self,
        *,
        K_elem,
        F_elem,
        element_ids,
        gdofs_map,
        A_red,
        R_red,
        hook=None,
    ):
        del element_ids, hook
        if K_elem is not None and A_red is not None:
            A_work = A_red.copy()
            for e in range(int(np.asarray(gdofs_map).shape[0])):
                gd = np.asarray(gdofs_map[e], dtype=int)
                A_work[np.ix_(gd, gd)] += np.asarray(K_elem[e], dtype=float)
            A_red = A_work
        if F_elem is not None and R_red is not None:
            R_work = np.asarray(R_red, dtype=float).copy()
            for e in range(int(np.asarray(gdofs_map).shape[0])):
                gd = np.asarray(gdofs_map[e], dtype=int)
                R_work[gd] += np.asarray(F_elem[e], dtype=float)
            R_red = R_work
        return A_red, R_red


def test_operator_manager_orders_runtime_hooks() -> None:
    log: list[tuple] = []
    solver = object()
    manager = OperatorManager(
        [
            _LogOperator("a", log, residual_shift=1.5),
            _LogOperator("b", log, residual_shift=-0.5),
        ]
    )

    manager.bind(solver)
    manager.on_step_begin(
        solver=solver,
        functions=["u"],
        prev_functions=["u_prev"],
        aux_functions={"dt": 0.1},
        step=2,
        step_no=3,
        t=1.25,
        dt=0.1,
        bcs=["bc"],
    )
    manager.before_assembly(solver=solver, coeffs={"marker": 4.0}, need_matrix=False)
    _, residual = manager.after_assembly(
        solver=solver,
        coeffs={"marker": 4.0},
        A_red=sp.eye(2, format="csr"),
        R_red=np.array([2.0, 5.0]),
        need_matrix=False,
    )
    manager.on_nonlinear_iteration_begin(
        solver=solver,
        functions=["u"],
        prev_functions=["u_prev"],
        aux_functions=None,
        iteration=1,
        coeffs={"marker": 4.0},
        bcs=["bc"],
        metrics={"dt": 0.1},
    )
    manager.on_nonlinear_update(
        solver=solver,
        functions=["u"],
        prev_functions=["u_prev"],
        aux_functions=None,
        iteration=1,
        coeffs={"marker": 4.0},
        delta_red=np.array([1.0]),
        delta_full=np.array([1.0, 0.0]),
        bcs=["bc"],
        metrics={"update_inf": 1.0},
    )
    manager.on_nonlinear_iteration_end(
        solver=solver,
        functions=["u"],
        prev_functions=["u_prev"],
        aux_functions=None,
        iteration=1,
        coeffs={"marker": 4.0},
        converged=False,
        bcs=["bc"],
        metrics={"reason": "continue"},
    )
    manager.on_step_accept(
        solver=solver,
        functions=["u"],
        prev_functions=["u_prev"],
        aux_functions=None,
        step=2,
        step_no=3,
        t=1.25,
        dt=0.1,
        bcs=["bc"],
    )
    manager.on_step_reject(
        solver=solver,
        functions=["u"],
        prev_functions=["u_prev"],
        aux_functions=None,
        step=2,
        step_no=3,
        t=1.25,
        dt=0.1,
        bcs=["bc"],
        exception=None,
        reason="retry",
    )

    assert np.allclose(residual, np.array([3.0, 6.0]))
    assert [entry[:2] for entry in log] == [
        ("bind", "a"),
        ("bind", "b"),
        ("step_begin", "a"),
        ("step_begin", "b"),
        ("before_assembly", "a"),
        ("before_assembly", "b"),
        ("after_assembly", "a"),
        ("after_assembly", "b"),
        ("nonlinear_begin", "a"),
        ("nonlinear_begin", "b"),
        ("nonlinear_update", "a"),
        ("nonlinear_update", "b"),
        ("nonlinear_end", "a"),
        ("nonlinear_end", "b"),
        ("step_accept", "a"),
        ("step_accept", "b"),
        ("step_reject", "a"),
        ("step_reject", "b"),
    ]


def test_newton_solver_reduced_assembly_applies_runtime_operators() -> None:
    log: list[tuple] = []
    solver = NewtonSolver.__new__(NewtonSolver)
    solver.backend = "python"
    solver.preassemble_cb = lambda coeffs: log.append(("preassemble", float(coeffs["marker"])))
    solver._assemble_system_reduced_python = (
        lambda coeffs, need_matrix=True: (
            sp.eye(2, format="csr") if need_matrix else None,
            np.array([3.0, 4.0]),
        )
    )
    solver.set_runtime_operators([_LogOperator("runtime", log, residual_shift=-1.0)])

    _, residual = NewtonSolver._assemble_system_reduced(solver, {"marker": 2.0}, need_matrix=False)

    assert np.allclose(residual, np.array([2.0, 3.0]))
    assert [entry[:2] for entry in log] == [
        ("bind", "runtime"),
        ("preassemble", 2.0),
        ("before_assembly", "runtime"),
        ("after_assembly", "runtime"),
    ]


def test_newton_solver_forwards_nonlinear_iteration_hooks() -> None:
    log: list[tuple] = []
    solver = NewtonSolver.__new__(NewtonSolver)
    solver.set_runtime_operators([_LogOperator("runtime", log)])

    coeffs = {"marker": 9.0}
    NewtonSolver._notify_operator_nonlinear_iteration_begin(
        solver,
        functions=["u"],
        prev_functions=["u_prev"],
        aux_functions=None,
        iteration=3,
        coeffs=coeffs,
        bcs=[],
        metrics={"dt": 0.1},
    )
    NewtonSolver._notify_operator_nonlinear_update(
        solver,
        functions=["u"],
        prev_functions=["u_prev"],
        aux_functions=None,
        iteration=3,
        coeffs=coeffs,
        delta_red=np.array([1.0]),
        delta_full=np.array([1.0, 0.0]),
        bcs=[],
        metrics={"update_inf": 1.0},
    )
    NewtonSolver._notify_operator_nonlinear_iteration_end(
        solver,
        functions=["u"],
        prev_functions=["u_prev"],
        aux_functions=None,
        iteration=3,
        coeffs=coeffs,
        converged=True,
        bcs=[],
        metrics={"reason": "unit"},
    )

    assert [entry[:2] for entry in log] == [
        ("bind", "runtime"),
        ("nonlinear_begin", "runtime"),
        ("nonlinear_update", "runtime"),
        ("nonlinear_end", "runtime"),
    ]
    assert log[-1] == ("nonlinear_end", "runtime", 3, True, "unit")


def test_pointwise_operator_dispatches_backend_and_applies_result() -> None:
    solver = type("SolverStub", (), {"backend": "cpp"})()
    op = _PointwiseProbeOperator()

    result = op.run_pointwise(
        solver=solver,
        coeffs={"marker": 2.5},
        need_matrix=False,
    )

    assert result is not None
    assert np.allclose(result.values, -2.5)
    assert op.applied_result is not None
    workset, applied = op.applied_result
    assert workset.backend == "cpp"
    assert np.array_equal(applied.entity_ids, np.array([0, 2], dtype=int))
    assert applied.metadata["backend"] == "cpp"


def test_fluid_dvms_solver_operator_is_symbolic_pointwise_operator() -> None:
    assert issubclass(example2_local._FluidDVMSSolverOperator, SymbolicPointwiseNewtonOperator)


def test_symbolic_pointwise_newton_operator_solves_batched_vector_rhs() -> None:
    matrices = np.asarray(
        [
            [[2.0, 0.0], [0.0, 4.0]],
            [[1.0, 1.0], [0.0, 3.0]],
        ],
        dtype=float,
    )
    rhs = np.asarray(
        [
            [6.0, 8.0],
            [5.0, 9.0],
        ],
        dtype=float,
    )

    delta, solved = SymbolicPointwiseNewtonOperator._solve_active_systems(matrices, rhs)

    assert np.array_equal(solved, np.array([True, True], dtype=bool))
    assert np.allclose(delta[0], np.array([3.0, 2.0]))
    assert np.allclose(delta[1], np.array([2.0, 3.0]))


def test_reduced_residual_shift_operator_subtracts_shift() -> None:
    op = example2_local._ReducedResidualShiftOperator(np.array([1.0, -2.0]))

    _, residual = op.after_assembly(
        solver=None,
        coeffs={},
        A_red=None,
        R_red=np.array([4.0, 3.0]),
        need_matrix=False,
    )

    assert np.allclose(residual, np.array([3.0, 5.0]))


def test_fused_local_operator_stages_and_commits_state() -> None:
    registry = StateRegistry()
    layout = QuadratureLayout(
        entity_kind="volume_cell",
        cell_type="tri",
        quadrature_order=1,
        reference_points=np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float),
        reference_weights=np.array([0.5], dtype=float),
    )
    qfield = registry.register_quadrature(
        "subscale",
        layout=layout,
        n_entities=2,
        tensor_shape=(2,),
        persistence="step",
    )
    qfield.assign(np.ones((2, 1, 2), dtype=float))

    def _workset_builder(*, solver, coeffs, need_matrix: bool):
        del solver, coeffs
        return {
            "need_matrix": bool(need_matrix),
            "element_ids": np.asarray([0, 1], dtype=int),
            "gdofs_map": np.asarray([[0, 1], [1, 2]], dtype=int),
        }

    def _kernel(workset):
        del workset
        return LocalAssemblyResult(
            K_elem=np.asarray(
                [
                    [[2.0, 0.0], [0.0, 2.0]],
                    [[3.0, 0.0], [0.0, 3.0]],
                ],
                dtype=float,
            ),
            F_elem=np.asarray(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                dtype=float,
            ),
            state_updates=(
                LocalStateUpdate(
                    field=qfield,
                    values=7.0 * np.ones((2, 1, 2), dtype=float),
                    entity_ids=np.asarray([0, 1], dtype=int),
                    staged=True,
                ),
            ),
        )

    op = CallbackFusedLocalAssemblyOperator(
        workset_builder=_workset_builder,
        python_kernel=_kernel,
        state_registries=(registry,),
    )
    solver = _ReducedScatterStub(backend="python")
    A_red = np.zeros((3, 3), dtype=float)
    R_red = np.zeros((3,), dtype=float)

    op.on_step_begin(
        solver=solver,
        functions=[],
        prev_functions=[],
        aux_functions=None,
        step=0,
        step_no=1,
        t=0.0,
        dt=1.0,
        bcs=[],
    )
    A_red, R_red = op.after_assembly(
        solver=solver,
        coeffs={},
        A_red=A_red,
        R_red=R_red,
        need_matrix=True,
    )
    assert np.allclose(qfield.values, 1.0)
    assert np.allclose(qfield.staged_values, 7.0)
    op.on_step_accept(
        solver=solver,
        functions=[],
        prev_functions=[],
        aux_functions=None,
        step=0,
        step_no=1,
        t=0.0,
        dt=1.0,
        bcs=[],
    )
    assert np.allclose(qfield.values, 7.0)
    assert np.allclose(qfield.staged_values, 7.0)
    assert np.allclose(
        A_red,
        np.asarray(
            [
                [2.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=float,
        ),
    )
    assert np.allclose(R_red, np.array([1.0, 5.0, 4.0], dtype=float))


def test_fused_local_operator_rollback_restores_step_state() -> None:
    registry = StateRegistry()
    layout = QuadratureLayout(
        entity_kind="volume_cell",
        cell_type="tri",
        quadrature_order=1,
        reference_points=np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float),
        reference_weights=np.array([0.5], dtype=float),
    )
    qfield = registry.register_quadrature(
        "subscale",
        layout=layout,
        n_entities=1,
        tensor_shape=(2,),
        persistence="step",
    )
    qfield.assign(np.ones((1, 1, 2), dtype=float))
    qfield.stage(5.0 * np.ones((1, 1, 2), dtype=float))
    op = CallbackFusedLocalAssemblyOperator(
        workset_builder=lambda **kwargs: None,
        state_registries=(registry,),
    )
    op.on_step_reject(
        solver=None,
        functions=[],
        prev_functions=[],
        aux_functions=None,
        step=0,
        step_no=1,
        t=0.0,
        dt=1.0,
        bcs=[],
        exception=None,
        reason="retry",
    )
    assert np.allclose(qfield.values, 1.0)
    assert np.allclose(qfield.staged_values, 1.0)


def test_symbolic_fused_local_operator_applies_quadrature_state_updates(monkeypatch) -> None:
    class _FakeDH:
        def __init__(self) -> None:
            self.mixed_element = type("ME", (), {"mesh": type("MeshStub", (), {"n_elements": 2})()})()

        def get_elemental_dofs(self, eid: int):
            return np.asarray([eid, eid + 1], dtype=int)

    class _FakeCompiler:
        def assemble_local_contributions(self, form_or_equation, *, entity_ids=None, need_matrix=True, need_vector=True):
            del form_or_equation
            del need_matrix, need_vector
            eids = np.asarray(entity_ids, dtype=int)
            return type(
                "Batch",
                (),
                {
                    "K_elem": np.asarray(
                        [
                            [[2.0, 0.0], [0.0, 2.0]],
                            [[3.0, 0.0], [0.0, 3.0]],
                        ],
                        dtype=float,
                    ),
                    "F_elem": np.asarray(
                        [
                            [1.0, 2.0],
                            [3.0, 4.0],
                        ],
                        dtype=float,
                    ),
                    "element_ids": eids,
                    "gdofs_map": np.asarray([[0, 1], [1, 2]], dtype=int),
                },
            )()

        def assemble_and_scatter_local_contributions_reduced(
            self,
            form_or_equation,
            *,
            solver,
            A_red,
            R_red,
            need_matrix: bool,
            entity_ids=None,
        ):
            batch = self.assemble_local_contributions(form_or_equation, entity_ids=entity_ids)
            return solver.scatter_element_contribs_reduced(
                K_elem=batch.K_elem if bool(need_matrix) else None,
                F_elem=batch.F_elem,
                element_ids=batch.element_ids,
                gdofs_map=batch.gdofs_map,
                A_red=A_red,
                R_red=R_red,
            )

        def evaluate_volume_expressions_on_quadrature(self, exprs, *, layout, element_ids=None):
            del exprs
            eids = np.asarray(element_ids, dtype=int)
            return {
                "update_subscale": 9.0 * np.ones((int(eids.shape[0]), int(layout.n_qp), 2), dtype=float)
            }

    registry = StateRegistry()
    layout = QuadratureLayout(
        entity_kind="volume_cell",
        cell_type="tri",
        quadrature_order=1,
        reference_points=np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float),
        reference_weights=np.array([0.5], dtype=float),
    )
    qfield = registry.register_quadrature(
        "subscale",
        layout=layout,
        n_entities=2,
        tensor_shape=(2,),
        persistence="step",
    )
    qfield.assign(np.ones((2, 1, 2), dtype=float))

    op = SymbolicFusedLocalAssemblyOperator(
        dof_handler=_FakeDH(),
        form_or_equation=object(),
        quadrature_state_updates=(
            SymbolicQuadratureStateUpdateSpec(
                field=qfield,
                expr=object(),
                staged=True,
                name="update_subscale",
            ),
        ),
        state_registries=(registry,),
        element_ids=np.asarray([0, 1], dtype=int),
    )
    monkeypatch.setattr(op, "_compiler", lambda backend: _FakeCompiler())
    monkeypatch.setattr(op, "_local_domain_type", lambda: "volume")
    solver = _ReducedScatterStub(backend="cpp")
    A_red = np.zeros((3, 3), dtype=float)
    R_red = np.zeros((3,), dtype=float)

    A_red, R_red = op.after_assembly(
        solver=solver,
        coeffs={},
        A_red=A_red,
        R_red=R_red,
        need_matrix=True,
    )

    assert np.allclose(qfield.values, 1.0)
    assert np.allclose(qfield.staged_values, 9.0)
    assert np.allclose(A_red[0, 0], 2.0)
    assert np.allclose(A_red[1, 1], 5.0)
    assert np.allclose(A_red[2, 2], 3.0)
    assert np.allclose(R_red, np.array([1.0, 5.0, 4.0], dtype=float))


def test_symbolic_fused_local_operator_applies_nonmatching_quadrature_state_updates(monkeypatch) -> None:
    class _FakeDH:
        def __init__(self) -> None:
            self.mixed_element = type("ME", (), {"mesh": type("MeshStub", (), {"n_elements": 1})()})()

        def get_elemental_dofs(self, eid: int):
            del eid
            return np.asarray([0, 1], dtype=int)

    class _FakeCompiler:
        def __init__(self) -> None:
            self.calls = []

        def assemble_and_scatter_local_contributions_reduced(
            self,
            form_or_equation,
            *,
            solver,
            A_red,
            R_red,
            need_matrix: bool,
            entity_ids=None,
        ):
            del form_or_equation, solver, need_matrix
            if entity_ids is not None:
                raise AssertionError("nonmatching interface updates must not receive volume entity_ids")
            return A_red, R_red

        def evaluate_nonmatching_interface_expressions_on_quadrature(self, exprs, *, layout, interface, quadrature):
            self.calls.append((set(exprs), layout.entity_kind, interface, quadrature))
            return {
                "update_damage": np.asarray(
                    [
                        [0.2, 0.3],
                        [0.4, 0.5],
                    ],
                    dtype=float,
                )
            }

    registry = StateRegistry()
    layout = QuadratureLayout(
        entity_kind="nonmatching_interface",
        cell_type="line",
        quadrature_order=2,
        reference_points=np.array([[-1.0], [1.0]], dtype=float),
        reference_weights=np.array([1.0, 1.0], dtype=float),
    )
    qfield = registry.register_quadrature(
        "damage",
        layout=layout,
        n_entities=2,
        persistence="step",
    )
    qfield.assign(np.zeros((2, 2), dtype=float))

    fake_compiler = _FakeCompiler()
    fake_interface = object()
    op = SymbolicFusedLocalAssemblyOperator(
        dof_handler=_FakeDH(),
        form_or_equation=object(),
        quadrature_state_updates=(
            SymbolicQuadratureStateUpdateSpec(
                field=qfield,
                expr=object(),
                staged=True,
                name="update_damage",
            ),
        ),
        state_registries=(registry,),
    )
    monkeypatch.setattr(op, "_compiler", lambda backend: fake_compiler)
    monkeypatch.setattr(op, "_local_domain_type", lambda: "nonmatching_interface")
    monkeypatch.setattr(op, "_nonmatching_interface_for_updates", lambda: fake_interface)
    monkeypatch.setattr(op, "_nonmatching_quadrature_rule_for_updates", lambda: "gauss_lobatto")

    A_red = np.zeros((2, 2), dtype=float)
    R_red = np.zeros((2,), dtype=float)
    A_out, R_out = op.after_assembly(
        solver=_ReducedScatterStub(backend="cpp"),
        coeffs={},
        A_red=A_red,
        R_red=R_red,
        need_matrix=True,
    )

    assert A_out is A_red
    assert R_out is R_red
    assert len(fake_compiler.calls) == 1
    names, entity_kind, interface, quadrature = fake_compiler.calls[0]
    assert names == {"update_damage"}
    assert entity_kind == "nonmatching_interface"
    assert interface is fake_interface
    assert quadrature == "gauss_lobatto"
    np.testing.assert_allclose(qfield.values, 0.0)
    np.testing.assert_allclose(qfield.staged_values, np.asarray([[0.2, 0.3], [0.4, 0.5]], dtype=float))


def test_symbolic_fused_local_operator_applies_trace_link_quadrature_state_updates(monkeypatch) -> None:
    class _FakeDH:
        def __init__(self) -> None:
            self.mixed_element = type("ME", (), {"mesh": type("MeshStub", (), {"n_elements": 1})()})()

        def get_elemental_dofs(self, eid: int):
            del eid
            return np.asarray([0, 1], dtype=int)

    class _FakeCompiler:
        def __init__(self) -> None:
            self.calls = []

        def assemble_and_scatter_local_contributions_reduced(
            self,
            form_or_equation,
            *,
            solver,
            A_red,
            R_red,
            need_matrix: bool,
            entity_ids=None,
        ):
            del form_or_equation, solver, need_matrix
            if entity_ids is not None:
                raise AssertionError("trace-link updates must not receive volume entity_ids")
            return A_red, R_red

        def evaluate_trace_link_expressions_on_quadrature(self, exprs, *, layout, trace, quadrature):
            self.calls.append((set(exprs), layout.entity_kind, trace, quadrature))
            return {"update_damage": np.asarray([[0.6, 0.7]], dtype=float)}

    registry = StateRegistry()
    layout = QuadratureLayout(
        entity_kind="trace_link",
        cell_type="line",
        quadrature_order=2,
        reference_points=np.array([[-1.0], [1.0]], dtype=float),
        reference_weights=np.array([1.0, 1.0], dtype=float),
    )
    qfield = registry.register_quadrature(
        "damage",
        layout=layout,
        n_entities=1,
        persistence="step",
    )
    qfield.assign(np.zeros((1, 2), dtype=float))

    fake_compiler = _FakeCompiler()
    fake_trace = object()
    op = SymbolicFusedLocalAssemblyOperator(
        dof_handler=_FakeDH(),
        form_or_equation=object(),
        quadrature_state_updates=(
            SymbolicQuadratureStateUpdateSpec(
                field=qfield,
                expr=object(),
                staged=True,
                name="update_damage",
            ),
        ),
        state_registries=(registry,),
    )
    monkeypatch.setattr(op, "_compiler", lambda backend: fake_compiler)
    monkeypatch.setattr(op, "_local_domain_type", lambda: "trace_link")
    monkeypatch.setattr(op, "_trace_link_for_updates", lambda: fake_trace)
    monkeypatch.setattr(op, "_trace_link_quadrature_rule_for_updates", lambda: "gauss_lobatto")

    A_red = np.zeros((2, 2), dtype=float)
    R_red = np.zeros((2,), dtype=float)
    A_out, R_out = op.after_assembly(
        solver=_ReducedScatterStub(backend="cpp"),
        coeffs={},
        A_red=A_red,
        R_red=R_red,
        need_matrix=True,
    )

    assert A_out is A_red
    assert R_out is R_red
    assert len(fake_compiler.calls) == 1
    names, entity_kind, trace, quadrature = fake_compiler.calls[0]
    assert names == {"update_damage"}
    assert entity_kind == "trace_link"
    assert trace is fake_trace
    assert quadrature == "gauss_lobatto"
    np.testing.assert_allclose(qfield.values, 0.0)
    np.testing.assert_allclose(qfield.staged_values, np.asarray([[0.6, 0.7]], dtype=float))
