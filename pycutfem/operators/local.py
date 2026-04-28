from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .base import RuntimeOperator


def _normalize_backend_name(backend: str | None) -> str:
    name = str(backend or "python").strip().lower()
    if name not in {"python", "jit", "cpp"}:
        raise ValueError(f"Unsupported local-operator backend {backend!r}.")
    return name


@dataclass(slots=True)
class LocalAssemblyWorkset:
    """Normalized element batch passed from the solver hook into a local kernel."""

    solver: Any
    coeffs: Any
    need_matrix: bool
    backend: str
    element_ids: np.ndarray
    gdofs_map: np.ndarray
    payload: Mapping[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.backend = _normalize_backend_name(self.backend)
        self.element_ids = np.asarray(self.element_ids, dtype=int).reshape(-1)
        self.gdofs_map = np.asarray(self.gdofs_map, dtype=int)
        if self.gdofs_map.ndim != 2:
            raise ValueError("gdofs_map must have shape (n_entities, n_local_dofs).")
        if self.gdofs_map.shape[0] != self.element_ids.shape[0]:
            raise ValueError(
                "element_ids and gdofs_map must agree on the leading batch dimension "
                f"(got {self.element_ids.shape[0]} and {self.gdofs_map.shape[0]})."
            )
        self.need_matrix = bool(self.need_matrix)
        self.payload = dict(self.payload or {})
        self.metadata = dict(self.metadata or {})


@dataclass(slots=True)
class LocalAssemblyResult:
    """
    Element-local contributions returned by a runtime local-assembly kernel.

    `element_ids` and `gdofs_map` default to the workset batch when omitted.
    """

    K_elem: np.ndarray | None = None
    F_elem: np.ndarray | None = None
    element_ids: np.ndarray | None = None
    gdofs_map: np.ndarray | None = None
    hook: Any = None
    state_updates: tuple["LocalStateUpdate", ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LocalStateUpdate:
    """State-field update produced alongside a local assembly batch."""

    field: Any
    values: np.ndarray
    entity_ids: np.ndarray | None = None
    staged: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SymbolicQuadratureStateUpdateSpec:
    """Symbolic quadrature-state update bundled with a local operator."""

    field: Any
    expr: Any
    staged: bool = True
    name: str | None = None


def _coerce_local_workset(
    raw: LocalAssemblyWorkset | Mapping[str, Any] | None,
    *,
    solver: Any,
    coeffs,
    need_matrix: bool,
) -> LocalAssemblyWorkset | None:
    if raw is None:
        return None
    if isinstance(raw, LocalAssemblyWorkset):
        return raw
    if isinstance(raw, Mapping):
        return LocalAssemblyWorkset(
            solver=solver,
            coeffs=coeffs,
            need_matrix=bool(raw.get("need_matrix", need_matrix)),
            backend=str(raw.get("backend", getattr(solver, "backend", "python"))),
            element_ids=raw["element_ids"],
            gdofs_map=raw["gdofs_map"],
            payload=raw.get("payload", {}),
            metadata=raw.get("metadata", {}),
        )
    raise TypeError(
        "Local assembly worksets must be returned as LocalAssemblyWorkset, mapping, or None."
    )


def _normalize_local_result(
    raw: LocalAssemblyResult | tuple | None,
    *,
    workset: LocalAssemblyWorkset,
) -> LocalAssemblyResult | None:
    if raw is None:
        return None
    if not isinstance(raw, LocalAssemblyResult):
        if not isinstance(raw, tuple):
            raise TypeError(
                "Local assembly kernels must return LocalAssemblyResult, tuple, or None."
            )
        if len(raw) == 2:
            raw = LocalAssemblyResult(K_elem=raw[0], F_elem=raw[1])
        elif len(raw) == 3:
            raw = LocalAssemblyResult(K_elem=raw[0], F_elem=raw[1], state_updates=raw[2])
        elif len(raw) == 4:
            raw = LocalAssemblyResult(
                K_elem=raw[0],
                F_elem=raw[1],
                element_ids=raw[2],
                gdofs_map=raw[3],
            )
        elif len(raw) == 5:
            raw = LocalAssemblyResult(
                K_elem=raw[0],
                F_elem=raw[1],
                element_ids=raw[2],
                gdofs_map=raw[3],
                hook=raw[4],
            )
        elif len(raw) == 6:
            raw = LocalAssemblyResult(
                K_elem=raw[0],
                F_elem=raw[1],
                element_ids=raw[2],
                gdofs_map=raw[3],
                hook=raw[4],
                state_updates=raw[5],
            )
        else:
            raise ValueError(
                "Tuple local assembly results must have length 2, 3, 4, 5, or 6 "
                f"(got {len(raw)})."
            )

    element_ids = workset.element_ids if raw.element_ids is None else np.asarray(raw.element_ids, dtype=int).reshape(-1)
    gdofs_map = workset.gdofs_map if raw.gdofs_map is None else np.asarray(raw.gdofs_map, dtype=int)
    if gdofs_map.ndim != 2:
        raise ValueError("LocalAssemblyResult.gdofs_map must have shape (n_entities, n_local_dofs).")
    if gdofs_map.shape[0] != element_ids.shape[0]:
        raise ValueError(
            "LocalAssemblyResult element_ids and gdofs_map must agree on the leading batch dimension."
        )

    K_elem = None if raw.K_elem is None else np.asarray(raw.K_elem, dtype=float)
    F_elem = None if raw.F_elem is None else np.asarray(raw.F_elem, dtype=float)
    if K_elem is not None:
        if K_elem.ndim != 3:
            raise ValueError("K_elem must have shape (n_entities, n_local_dofs, n_local_dofs).")
        if K_elem.shape[0] != element_ids.shape[0]:
            raise ValueError("K_elem leading dimension must match element_ids.")
    if F_elem is not None:
        if F_elem.ndim != 2:
            raise ValueError("F_elem must have shape (n_entities, n_local_dofs).")
        if F_elem.shape[0] != element_ids.shape[0]:
            raise ValueError("F_elem leading dimension must match element_ids.")

    state_updates = _normalize_local_state_updates(raw.state_updates)

    return LocalAssemblyResult(
        K_elem=K_elem,
        F_elem=F_elem,
        element_ids=element_ids,
        gdofs_map=gdofs_map,
        hook=raw.hook,
        state_updates=state_updates,
        metadata=dict(raw.metadata or {}),
    )


def _normalize_local_state_updates(raw_updates) -> tuple[LocalStateUpdate, ...]:
    if raw_updates is None:
        return ()
    if isinstance(raw_updates, LocalStateUpdate):
        raw_updates = (raw_updates,)
    if not isinstance(raw_updates, (tuple, list)):
        raise TypeError("state_updates must be a LocalStateUpdate or a sequence of LocalStateUpdate objects.")
    normalized: list[LocalStateUpdate] = []
    for item in raw_updates:
        if isinstance(item, LocalStateUpdate):
            normalized.append(
                LocalStateUpdate(
                    field=item.field,
                    values=np.asarray(item.values, dtype=float),
                    entity_ids=None if item.entity_ids is None else np.asarray(item.entity_ids, dtype=int).reshape(-1),
                    staged=bool(item.staged),
                    metadata=dict(item.metadata or {}),
                )
            )
            continue
        if isinstance(item, Mapping):
            normalized.append(
                LocalStateUpdate(
                    field=item["field"],
                    values=np.asarray(item["values"], dtype=float),
                    entity_ids=None if item.get("entity_ids") is None else np.asarray(item.get("entity_ids"), dtype=int).reshape(-1),
                    staged=bool(item.get("staged", True)),
                    metadata=dict(item.get("metadata", {}) or {}),
                )
            )
            continue
        raise TypeError("Each state update must be a LocalStateUpdate or mapping.")
    return tuple(normalized)


def _apply_local_state_updates(state_updates: Sequence[LocalStateUpdate]) -> None:
    for update in state_updates:
        field = update.field
        values = np.asarray(update.values, dtype=float)
        entity_ids = None if update.entity_ids is None else np.asarray(update.entity_ids, dtype=int).reshape(-1)
        target_arr = getattr(field, "staged_values", None) if bool(update.staged) else getattr(field, "values", None)
        if target_arr is None:
            raise TypeError("LocalStateUpdate field must expose either 'values' or 'staged_values'.")
        target_arr = np.asarray(target_arr, dtype=float)
        if entity_ids is None:
            if values.shape != target_arr.shape:
                raise ValueError(
                    f"LocalStateUpdate expected full-field shape {target_arr.shape}, got {values.shape}."
                )
            target_arr[...] = values
            continue
        if values.shape[0] != int(entity_ids.shape[0]):
            raise ValueError(
                "LocalStateUpdate values and entity_ids must agree on the leading entity dimension."
            )
        expected_tail = tuple(int(v) for v in target_arr.shape[1:])
        if tuple(int(v) for v in values.shape[1:]) != expected_tail:
            raise ValueError(
                f"LocalStateUpdate expected trailing shape {expected_tail}, got {values.shape[1:]}."
            )
        target_arr[entity_ids] = values


class LocalAssemblyOperator(RuntimeOperator):
    """
    Runtime operator for explicit element-local discrete contributions.

    Subclasses build a workset, assemble a backend-specific local contribution,
    and the base class scatters it into the reduced system after the continuous
    UFL assembly has finished.
    """

    def build_local_workset(
        self,
        *,
        solver: Any,
        coeffs,
        need_matrix: bool,
    ) -> LocalAssemblyWorkset | Mapping[str, Any] | None:
        return None

    def assemble_local(self, workset: LocalAssemblyWorkset):
        return None

    def assemble_local_python(self, workset: LocalAssemblyWorkset):
        return self.assemble_local(workset)

    def assemble_local_jit(self, workset: LocalAssemblyWorkset):
        return self.assemble_local(workset)

    def assemble_local_cpp(self, workset: LocalAssemblyWorkset):
        return self.assemble_local(workset)

    def _run_local_backend(self, workset: LocalAssemblyWorkset) -> LocalAssemblyResult | None:
        backend = _normalize_backend_name(workset.backend)
        if backend == "python":
            raw = self.assemble_local_python(workset)
        elif backend == "jit":
            raw = self.assemble_local_jit(workset)
        elif backend == "cpp":
            raw = self.assemble_local_cpp(workset)
        else:  # pragma: no cover - guarded by _normalize_backend_name
            raise ValueError(f"Unsupported local-operator backend {backend!r}.")
        return _normalize_local_result(raw, workset=workset)

    def scatter_local_result(
        self,
        *,
        solver: Any,
        result: LocalAssemblyResult,
        A_red,
        R_red,
    ):
        record_pattern = getattr(solver, "record_amgcl_kratos_dirichlet_pattern", None)
        if callable(record_pattern):
            record_pattern(K_elem=result.K_elem, gdofs_map=result.gdofs_map)
        return solver.scatter_element_contribs_reduced(
            K_elem=result.K_elem,
            F_elem=result.F_elem,
            element_ids=result.element_ids,
            gdofs_map=result.gdofs_map,
            A_red=A_red,
            R_red=R_red,
            hook=result.hook,
        )

    def after_assembly(self, *, solver: Any, coeffs, A_red, R_red, need_matrix: bool):
        raw_workset = self.build_local_workset(
            solver=solver,
            coeffs=coeffs,
            need_matrix=bool(need_matrix),
        )
        workset = _coerce_local_workset(
            raw_workset,
            solver=solver,
            coeffs=coeffs,
            need_matrix=bool(need_matrix),
        )
        if workset is None:
            return A_red, R_red
        result = self._run_local_backend(workset)
        if result is None:
            return A_red, R_red
        return self.scatter_local_result(
            solver=solver,
            result=result,
            A_red=A_red,
            R_red=R_red,
        )


class FusedLocalAssemblyOperator(LocalAssemblyOperator):
    """
    Local operator that can update hidden runtime state and assemble local blocks
    under the same backend-dispatched contract.

    The operator owns step/iteration transaction handling for any attached
    `StateRegistry` instances:

    - `on_step_begin` resets iteration-persistent staged state,
    - `on_step_accept` commits step-persistent staged state,
    - `on_step_reject` rolls back step-persistent staged state.
    """

    def __init__(self, *, state_registries: Sequence[Any] | None = None) -> None:
        self.state_registries = tuple(state_registries or ())

    def on_step_begin(
        self,
        *,
        solver: Any,
        functions,
        prev_functions,
        aux_functions,
        step: int,
        step_no: int,
        t: float,
        dt: float,
        bcs,
    ) -> None:
        del solver, functions, prev_functions, aux_functions, step, step_no, t, dt, bcs
        for registry in self.state_registries:
            if hasattr(registry, "reset_iteration"):
                registry.reset_iteration()

    def on_step_accept(
        self,
        *,
        solver: Any,
        functions,
        prev_functions,
        aux_functions,
        step: int,
        step_no: int,
        t: float,
        dt: float,
        bcs,
    ) -> None:
        del solver, functions, prev_functions, aux_functions, step, step_no, t, dt, bcs
        for registry in self.state_registries:
            if hasattr(registry, "commit_step"):
                registry.commit_step()

    def on_step_reject(
        self,
        *,
        solver: Any,
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
        del solver, functions, prev_functions, aux_functions, step, step_no, t, dt, bcs, exception, reason
        for registry in self.state_registries:
            if hasattr(registry, "rollback_step"):
                registry.rollback_step()

    def after_assembly(self, *, solver: Any, coeffs, A_red, R_red, need_matrix: bool):
        raw_workset = self.build_local_workset(
            solver=solver,
            coeffs=coeffs,
            need_matrix=bool(need_matrix),
        )
        workset = _coerce_local_workset(
            raw_workset,
            solver=solver,
            coeffs=coeffs,
            need_matrix=bool(need_matrix),
        )
        if workset is None:
            return A_red, R_red
        result = self._run_local_backend(workset)
        if result is None:
            return A_red, R_red
        if result.state_updates:
            _apply_local_state_updates(result.state_updates)
        return self.scatter_local_result(
            solver=solver,
            result=result,
            A_red=A_red,
            R_red=R_red,
        )


class CallbackFusedLocalAssemblyOperator(FusedLocalAssemblyOperator):
    """Convenience fused local operator defined by backend-specific callables."""

    def __init__(
        self,
        *,
        workset_builder,
        python_kernel=None,
        jit_kernel=None,
        cpp_kernel=None,
        fallback_kernel=None,
        state_registries: Sequence[Any] | None = None,
    ) -> None:
        super().__init__(state_registries=state_registries)
        self._workset_builder = workset_builder
        self._python_kernel = python_kernel
        self._jit_kernel = jit_kernel
        self._cpp_kernel = cpp_kernel
        self._fallback_kernel = fallback_kernel

    def build_local_workset(self, *, solver: Any, coeffs, need_matrix: bool):
        return self._workset_builder(
            solver=solver,
            coeffs=coeffs,
            need_matrix=bool(need_matrix),
        )

    def assemble_local(self, workset: LocalAssemblyWorkset):
        if self._fallback_kernel is None:
            return None
        return self._fallback_kernel(workset)

    def assemble_local_python(self, workset: LocalAssemblyWorkset):
        if self._python_kernel is not None:
            return self._python_kernel(workset)
        return super().assemble_local_python(workset)

    def assemble_local_jit(self, workset: LocalAssemblyWorkset):
        if self._jit_kernel is not None:
            return self._jit_kernel(workset)
        return super().assemble_local_jit(workset)

    def assemble_local_cpp(self, workset: LocalAssemblyWorkset):
        if self._cpp_kernel is not None:
            return self._cpp_kernel(workset)
        return super().assemble_local_cpp(workset)


class CallbackLocalAssemblyOperator(LocalAssemblyOperator):
    """
    Small convenience wrapper for local operators defined by callables.

    This keeps example-local operator code compact while the repo still owns the
    generic backend-dispatch and scatter semantics.
    """

    def __init__(
        self,
        *,
        workset_builder,
        python_kernel=None,
        jit_kernel=None,
        cpp_kernel=None,
        fallback_kernel=None,
    ) -> None:
        self._workset_builder = workset_builder
        self._python_kernel = python_kernel
        self._jit_kernel = jit_kernel
        self._cpp_kernel = cpp_kernel
        self._fallback_kernel = fallback_kernel

    def build_local_workset(self, *, solver: Any, coeffs, need_matrix: bool):
        return self._workset_builder(
            solver=solver,
            coeffs=coeffs,
            need_matrix=bool(need_matrix),
        )

    def assemble_local(self, workset: LocalAssemblyWorkset):
        if self._fallback_kernel is None:
            return None
        return self._fallback_kernel(workset)

    def assemble_local_python(self, workset: LocalAssemblyWorkset):
        if self._python_kernel is not None:
            return self._python_kernel(workset)
        return super().assemble_local_python(workset)

    def assemble_local_jit(self, workset: LocalAssemblyWorkset):
        if self._jit_kernel is not None:
            return self._jit_kernel(workset)
        return super().assemble_local_jit(workset)

    def assemble_local_cpp(self, workset: LocalAssemblyWorkset):
        if self._cpp_kernel is not None:
            return self._cpp_kernel(workset)
        return super().assemble_local_cpp(workset)


class SymbolicLocalAssemblyOperator(LocalAssemblyOperator):
    """
    Local operator backed directly by the existing UFL/FormCompiler stack.

    This lets explicit discrete blocks be written once as UFL forms and then
    assembled as element-local matrix/vector batches under python, jit, or cpp.
    """

    def __init__(
        self,
        *,
        dof_handler,
        form_or_equation,
        element_ids: np.ndarray | None = None,
        entity_ids: np.ndarray | None = None,
        quadrature_order: int | None = None,
    ) -> None:
        self.dh = dof_handler
        self.me = dof_handler.mixed_element
        self.form_or_equation = form_or_equation
        self.element_ids = None if element_ids is None else np.asarray(element_ids, dtype=int).reshape(-1)
        self.entity_ids = (
            None
            if entity_ids is None
            else np.asarray(entity_ids, dtype=int).reshape(-1)
        )
        self.quadrature_order = None if quadrature_order is None else int(quadrature_order)
        self._compiler_cache: dict[tuple[int, str], Any] = {}

    def _local_domain_type(self) -> str:
        from pycutfem.ufl.expressions import Integral
        from pycutfem.ufl.forms import CondensedQuadratureLocalSystem, Equation, Form

        if isinstance(self.form_or_equation, CondensedQuadratureLocalSystem):
            return "volume"
        forms = []
        if isinstance(self.form_or_equation, Equation):
            forms = [frm for frm in (self.form_or_equation.a, self.form_or_equation.L) if frm is not None]
        elif isinstance(self.form_or_equation, (Form, Integral)):
            forms = [self.form_or_equation]
        else:
            raise TypeError(
                "SymbolicLocalAssemblyOperator expects an Equation, Form, or Integral."
            )

        domain_types = {
            str(getattr(integral.measure, "domain_type", "")).strip().lower()
            for form in forms
            for integral in ([form] if isinstance(form, Integral) else list(getattr(form, "integrals", ())))
            if isinstance(integral, Integral)
        }
        if not domain_types:
            raise ValueError("No integrals found for symbolic local assembly.")
        if len(domain_types) != 1:
            raise NotImplementedError(
                "SymbolicLocalAssemblyOperator currently expects all integrals to share one measure/domain type."
            )
        return next(iter(domain_types))

    def _compiler(self, backend: str):
        from pycutfem.ufl.compilers import FormCompiler

        key = (int(id(self.dh)), _normalize_backend_name(backend))
        cached = self._compiler_cache.get(key)
        if cached is not None:
            return cached
        compiler = FormCompiler(
            self.dh,
            quadrature_order=self.quadrature_order,
            backend=_normalize_backend_name(backend),
        )
        self._compiler_cache[key] = compiler
        return compiler

    def build_local_workset(self, *, solver: Any, coeffs, need_matrix: bool):
        del coeffs
        if self.entity_ids is not None:
            eids = np.asarray(self.entity_ids, dtype=int).reshape(-1)
            gdofs_map = np.zeros((int(eids.shape[0]), 1), dtype=int)
        elif self.element_ids is None:
            eids = np.arange(int(self.me.mesh.n_elements), dtype=int)
            gdofs_map = np.vstack([np.asarray(self.dh.get_elemental_dofs(int(eid)), dtype=int) for eid in eids]).astype(int)
        else:
            eids = np.asarray(self.element_ids, dtype=int).reshape(-1)
            gdofs_map = np.vstack([np.asarray(self.dh.get_elemental_dofs(int(eid)), dtype=int) for eid in eids]).astype(int)
        return LocalAssemblyWorkset(
            solver=solver,
            coeffs=None,
            need_matrix=bool(need_matrix),
            backend=str(getattr(solver, "backend", "python")),
            element_ids=eids,
            gdofs_map=gdofs_map,
            payload={},
        )

    def assemble_local(self, workset: LocalAssemblyWorkset):
        compiler = self._compiler(workset.backend)
        domain_type = self._local_domain_type()
        entity_ids = None
        if self.entity_ids is not None:
            entity_ids = self.entity_ids
        elif domain_type in {"volume", "interface"}:
            entity_ids = workset.element_ids
        batch = compiler.assemble_local_contributions(
            self.form_or_equation,
            entity_ids=entity_ids,
            need_matrix=bool(workset.need_matrix),
        )
        return LocalAssemblyResult(
            K_elem=batch.K_elem if workset.need_matrix else None,
            F_elem=batch.F_elem,
            element_ids=batch.element_ids,
            gdofs_map=batch.gdofs_map,
        )

    def after_assembly(self, *, solver: Any, coeffs, A_red, R_red, need_matrix: bool):
        del coeffs
        raw_workset = self.build_local_workset(
            solver=solver,
            coeffs=None,
            need_matrix=bool(need_matrix),
        )
        workset = _coerce_local_workset(
            raw_workset,
            solver=solver,
            coeffs=None,
            need_matrix=bool(need_matrix),
        )
        if workset is None:
            return A_red, R_red
        compiler = self._compiler(workset.backend)
        domain_type = self._local_domain_type()
        entity_ids = None
        if self.entity_ids is not None:
            entity_ids = self.entity_ids
        elif domain_type in {"volume", "interface"}:
            entity_ids = workset.element_ids
        return compiler.assemble_and_scatter_local_contributions_reduced(
            self.form_or_equation,
            solver=solver,
            A_red=A_red,
            R_red=R_red,
            need_matrix=bool(need_matrix),
            entity_ids=entity_ids,
        )


def _normalize_symbolic_quadrature_state_updates(
    raw_specs,
) -> tuple[SymbolicQuadratureStateUpdateSpec, ...]:
    if raw_specs is None:
        return ()
    if isinstance(raw_specs, Mapping):
        raw_specs = [
            SymbolicQuadratureStateUpdateSpec(field=field, expr=expr)
            for field, expr in raw_specs.items()
        ]
    elif isinstance(raw_specs, SymbolicQuadratureStateUpdateSpec):
        raw_specs = (raw_specs,)
    elif not isinstance(raw_specs, (tuple, list)):
        raise TypeError(
            "quadrature_state_updates must be a mapping, SymbolicQuadratureStateUpdateSpec, or sequence."
        )
    out: list[SymbolicQuadratureStateUpdateSpec] = []
    for idx, spec in enumerate(raw_specs):
        if isinstance(spec, SymbolicQuadratureStateUpdateSpec):
            out.append(spec)
            continue
        if isinstance(spec, Mapping):
            out.append(
                SymbolicQuadratureStateUpdateSpec(
                    field=spec["field"],
                    expr=spec["expr"],
                    staged=bool(spec.get("staged", True)),
                    name=spec.get("name"),
                )
            )
            continue
        if isinstance(spec, tuple) and len(spec) in {2, 3}:
            out.append(
                SymbolicQuadratureStateUpdateSpec(
                    field=spec[0],
                    expr=spec[1],
                    staged=True if len(spec) == 2 else bool(spec[2]),
                    name=f"state_update_{idx}",
                )
            )
            continue
        raise TypeError("Invalid quadrature-state update specification.")
    return tuple(out)


class SymbolicFusedLocalAssemblyOperator(FusedLocalAssemblyOperator):
    """
    First-class fused runtime operator for:

    - symbolic local matrix/vector assembly through the existing FormCompiler
    - symbolic quadrature-state updates evaluated on the same backend
    - solver-owned hidden-state transaction handling through attached registries
    """

    def __init__(
        self,
        *,
        dof_handler,
        form_or_equation,
        quadrature_state_updates=None,
        state_registries: Sequence[Any] | None = None,
        element_ids: np.ndarray | None = None,
        entity_ids: np.ndarray | None = None,
        quadrature_order: int | None = None,
    ) -> None:
        super().__init__(state_registries=state_registries)
        self.dh = dof_handler
        self.me = dof_handler.mixed_element
        self.form_or_equation = form_or_equation
        self.element_ids = None if element_ids is None else np.asarray(element_ids, dtype=int).reshape(-1)
        self.entity_ids = None if entity_ids is None else np.asarray(entity_ids, dtype=int).reshape(-1)
        self.quadrature_order = None if quadrature_order is None else int(quadrature_order)
        self.quadrature_state_updates = _normalize_symbolic_quadrature_state_updates(quadrature_state_updates)
        self._compiler_cache: dict[tuple[int, str], Any] = {}

    def _local_domain_type(self) -> str:
        from pycutfem.ufl.expressions import Integral
        from pycutfem.ufl.forms import CondensedQuadratureLocalSystem, Equation, Form

        if isinstance(self.form_or_equation, CondensedQuadratureLocalSystem):
            return "volume"
        forms = []
        if isinstance(self.form_or_equation, Equation):
            forms = [frm for frm in (self.form_or_equation.a, self.form_or_equation.L) if frm is not None]
        elif isinstance(self.form_or_equation, (Form, Integral)):
            forms = [self.form_or_equation]
        else:
            raise TypeError(
                "SymbolicFusedLocalAssemblyOperator expects an Equation, Form, or Integral."
            )

        domain_types = {
            str(getattr(integral.measure, "domain_type", "")).strip().lower()
            for form in forms
            for integral in ([form] if isinstance(form, Integral) else list(getattr(form, "integrals", ())))
            if isinstance(integral, Integral)
        }
        if not domain_types:
            raise ValueError("No integrals found for symbolic fused local assembly.")
        if len(domain_types) != 1:
            raise NotImplementedError(
                "SymbolicFusedLocalAssemblyOperator currently expects all integrals to share one measure/domain type."
            )
        return next(iter(domain_types))

    def _compiler(self, backend: str):
        from pycutfem.ufl.compilers import FormCompiler

        key = (int(id(self.dh)), _normalize_backend_name(backend))
        cached = self._compiler_cache.get(key)
        if cached is not None:
            return cached
        compiler = FormCompiler(
            self.dh,
            quadrature_order=self.quadrature_order,
            backend=_normalize_backend_name(backend),
        )
        self._compiler_cache[key] = compiler
        return compiler

    def build_local_workset(self, *, solver: Any, coeffs, need_matrix: bool):
        del coeffs
        if self.entity_ids is not None:
            eids = np.asarray(self.entity_ids, dtype=int).reshape(-1)
            gdofs_map = np.zeros((int(eids.shape[0]), 1), dtype=int)
        elif self.element_ids is None:
            eids = np.arange(int(self.me.mesh.n_elements), dtype=int)
            gdofs_map = np.vstack([np.asarray(self.dh.get_elemental_dofs(int(eid)), dtype=int) for eid in eids]).astype(int)
        else:
            eids = np.asarray(self.element_ids, dtype=int).reshape(-1)
            gdofs_map = np.vstack([np.asarray(self.dh.get_elemental_dofs(int(eid)), dtype=int) for eid in eids]).astype(int)
        return LocalAssemblyWorkset(
            solver=solver,
            coeffs=None,
            need_matrix=bool(need_matrix),
            backend=str(getattr(solver, "backend", "python")),
            element_ids=eids,
            gdofs_map=gdofs_map,
            payload={},
        )

    def assemble_local(self, workset: LocalAssemblyWorkset):
        compiler = self._compiler(workset.backend)
        domain_type = self._local_domain_type()
        entity_ids = None
        if self.entity_ids is not None:
            entity_ids = self.entity_ids
        elif domain_type in {"volume", "interface"}:
            entity_ids = workset.element_ids
        batch = compiler.assemble_local_contributions(
            self.form_or_equation,
            entity_ids=entity_ids,
            need_matrix=bool(workset.need_matrix),
        )
        state_updates: list[LocalStateUpdate] = []
        if self.quadrature_state_updates:
            grouped: dict[tuple[str, int, str], list[tuple[str, SymbolicQuadratureStateUpdateSpec]]] = {}
            for idx, spec in enumerate(self.quadrature_state_updates):
                layout = getattr(spec.field, "layout", None)
                if layout is None:
                    raise TypeError("SymbolicFusedLocalAssemblyOperator quadrature_state_updates require fields with a 'layout'.")
                key = (layout.signature, int(layout.quadrature_order), str(layout.cell_type))
                name = str(spec.name or getattr(spec.field, "name", f"state_update_{idx}"))
                grouped.setdefault(key, []).append((name, spec))
            for _, specs in grouped.items():
                layout = specs[0][1].field.layout
                exprs = {name: spec.expr for name, spec in specs}
                qp_values = compiler.evaluate_volume_expressions_on_quadrature(
                    exprs,
                    layout=layout,
                    element_ids=np.asarray(workset.element_ids, dtype=int),
                )
                for name, spec in specs:
                    state_updates.append(
                        LocalStateUpdate(
                            field=spec.field,
                            values=np.asarray(qp_values[name], dtype=float),
                            entity_ids=np.asarray(workset.element_ids, dtype=int),
                            staged=bool(spec.staged),
                        )
                    )
        return LocalAssemblyResult(
            K_elem=batch.K_elem if workset.need_matrix else None,
            F_elem=batch.F_elem,
            element_ids=batch.element_ids,
            gdofs_map=batch.gdofs_map,
            state_updates=tuple(state_updates),
        )

    def after_assembly(self, *, solver: Any, coeffs, A_red, R_red, need_matrix: bool):
        del coeffs
        raw_workset = self.build_local_workset(
            solver=solver,
            coeffs=None,
            need_matrix=bool(need_matrix),
        )
        workset = _coerce_local_workset(
            raw_workset,
            solver=solver,
            coeffs=None,
            need_matrix=bool(need_matrix),
        )
        if workset is None:
            return A_red, R_red
        compiler = self._compiler(workset.backend)
        domain_type = self._local_domain_type()
        entity_ids = None
        if self.entity_ids is not None:
            entity_ids = self.entity_ids
        elif domain_type in {"volume", "interface"}:
            entity_ids = workset.element_ids
        if self.quadrature_state_updates:
            grouped: dict[tuple[str, int, str], list[tuple[str, SymbolicQuadratureStateUpdateSpec]]] = {}
            for idx, spec in enumerate(self.quadrature_state_updates):
                layout = getattr(spec.field, "layout", None)
                if layout is None:
                    raise TypeError("SymbolicFusedLocalAssemblyOperator quadrature_state_updates require fields with a 'layout'.")
                key = (layout.signature, int(layout.quadrature_order), str(layout.cell_type))
                name = str(spec.name or getattr(spec.field, "name", f"state_update_{idx}"))
                grouped.setdefault(key, []).append((name, spec))
            state_updates: list[LocalStateUpdate] = []
            for _, specs in grouped.items():
                layout = specs[0][1].field.layout
                exprs = {name: spec.expr for name, spec in specs}
                qp_values = compiler.evaluate_volume_expressions_on_quadrature(
                    exprs,
                    layout=layout,
                    element_ids=np.asarray(workset.element_ids, dtype=int),
                )
                for name, spec in specs:
                    state_updates.append(
                        LocalStateUpdate(
                            field=spec.field,
                            values=np.asarray(qp_values[name], dtype=float),
                            entity_ids=np.asarray(workset.element_ids, dtype=int),
                            staged=bool(spec.staged),
                        )
                    )
            if state_updates:
                _apply_local_state_updates(tuple(state_updates))
        return compiler.assemble_and_scatter_local_contributions_reduced(
            self.form_or_equation,
            solver=solver,
            A_red=A_red,
            R_red=R_red,
            need_matrix=bool(need_matrix),
            entity_ids=entity_ids,
        )


__all__ = [
    "CallbackFusedLocalAssemblyOperator",
    "CallbackLocalAssemblyOperator",
    "FusedLocalAssemblyOperator",
    "LocalAssemblyOperator",
    "LocalAssemblyResult",
    "LocalStateUpdate",
    "LocalAssemblyWorkset",
    "SymbolicFusedLocalAssemblyOperator",
    "SymbolicLocalAssemblyOperator",
    "SymbolicQuadratureStateUpdateSpec",
]
