from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from pycutfem.state import QuadratureLayout, QuadratureStateField

from .base import RuntimeOperator


def _normalize_backend_name(backend: str | None) -> str:
    name = str(backend or "python").strip().lower()
    if name not in {"python", "jit", "cpp"}:
        raise ValueError(f"Unsupported pointwise-operator backend {backend!r}.")
    return name


@dataclass(slots=True)
class PointwiseQuadratureWorkset:
    """Normalized quadrature-batch payload passed into a pointwise update kernel."""

    solver: Any
    coeffs: Any
    need_matrix: bool
    backend: str
    layout: QuadratureLayout
    entity_ids: np.ndarray
    payload: Mapping[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.backend = _normalize_backend_name(self.backend)
        if not isinstance(self.layout, QuadratureLayout):
            raise TypeError("layout must be a QuadratureLayout.")
        self.entity_ids = np.asarray(self.entity_ids, dtype=int).reshape(-1)
        self.need_matrix = bool(self.need_matrix)
        self.payload = dict(self.payload or {})
        self.metadata = dict(self.metadata or {})


@dataclass(slots=True)
class PointwiseQuadratureResult:
    """Quadrature-state update returned by a pointwise runtime operator."""

    values: np.ndarray
    entity_ids: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _coerce_pointwise_workset(
    raw: PointwiseQuadratureWorkset | Mapping[str, Any] | None,
    *,
    solver: Any,
    coeffs,
    need_matrix: bool,
) -> PointwiseQuadratureWorkset | None:
    if raw is None:
        return None
    if isinstance(raw, PointwiseQuadratureWorkset):
        return raw
    if isinstance(raw, Mapping):
        return PointwiseQuadratureWorkset(
            solver=solver,
            coeffs=coeffs,
            need_matrix=bool(raw.get("need_matrix", need_matrix)),
            backend=str(raw.get("backend", getattr(solver, "backend", "python"))),
            layout=raw["layout"],
            entity_ids=raw["entity_ids"],
            payload=raw.get("payload", {}),
            metadata=raw.get("metadata", {}),
        )
    raise TypeError(
        "Pointwise worksets must be returned as PointwiseQuadratureWorkset, mapping, or None."
    )


def _normalize_pointwise_result(
    raw: PointwiseQuadratureResult | Mapping[str, Any] | np.ndarray | tuple | None,
    *,
    workset: PointwiseQuadratureWorkset,
) -> PointwiseQuadratureResult | None:
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        raw = PointwiseQuadratureResult(values=raw)
    elif isinstance(raw, Mapping):
        raw = PointwiseQuadratureResult(
            values=raw["values"],
            entity_ids=raw.get("entity_ids"),
            metadata=raw.get("metadata", {}),
        )
    elif not isinstance(raw, PointwiseQuadratureResult):
        if not isinstance(raw, tuple):
            raise TypeError(
                "Pointwise update kernels must return PointwiseQuadratureResult, mapping, ndarray, tuple, or None."
            )
        if len(raw) == 1:
            raw = PointwiseQuadratureResult(values=raw[0])
        elif len(raw) == 2:
            raw = PointwiseQuadratureResult(values=raw[0], entity_ids=raw[1])
        else:
            raise ValueError(
                "Tuple pointwise results must have length 1 or 2 "
                f"(got {len(raw)})."
            )

    entity_ids = workset.entity_ids if raw.entity_ids is None else np.asarray(raw.entity_ids, dtype=int).reshape(-1)
    values = np.asarray(raw.values, dtype=float)
    if values.shape[0] != entity_ids.shape[0]:
        raise ValueError(
            "PointwiseQuadratureResult values and entity_ids must agree on the leading entity dimension."
        )
    if values.ndim < 2 or int(values.shape[1]) != int(workset.layout.n_qp):
        raise ValueError(
            "PointwiseQuadratureResult values must have shape (n_entities, n_qp, ...)."
        )
    return PointwiseQuadratureResult(
        values=values,
        entity_ids=entity_ids,
        metadata=dict(raw.metadata or {}),
    )


class PointwiseQuadratureOperator(RuntimeOperator):
    """
    Runtime operator for explicit quadrature-state updates before assembly.

    Subclasses build a quadrature workset, run a backend-specific pointwise
    update kernel, and apply the resulting state update.
    """

    def build_pointwise_workset(
        self,
        *,
        solver: Any,
        coeffs,
        need_matrix: bool,
    ) -> PointwiseQuadratureWorkset | Mapping[str, Any] | None:
        return None

    def update_pointwise(self, workset: PointwiseQuadratureWorkset):
        return None

    def update_pointwise_python(self, workset: PointwiseQuadratureWorkset):
        return self.update_pointwise(workset)

    def update_pointwise_jit(self, workset: PointwiseQuadratureWorkset):
        return self.update_pointwise(workset)

    def update_pointwise_cpp(self, workset: PointwiseQuadratureWorkset):
        return self.update_pointwise(workset)

    def apply_pointwise_result(
        self,
        *,
        solver: Any,
        workset: PointwiseQuadratureWorkset,
        result: PointwiseQuadratureResult,
    ) -> None:
        del solver, workset, result

    def _run_pointwise_backend(self, workset: PointwiseQuadratureWorkset) -> PointwiseQuadratureResult | None:
        backend = _normalize_backend_name(workset.backend)
        if backend == "python":
            raw = self.update_pointwise_python(workset)
        elif backend == "jit":
            raw = self.update_pointwise_jit(workset)
        elif backend == "cpp":
            raw = self.update_pointwise_cpp(workset)
        else:  # pragma: no cover - guarded by _normalize_backend_name
            raise ValueError(f"Unsupported pointwise backend {backend!r}.")
        return _normalize_pointwise_result(raw, workset=workset)

    def run_pointwise(
        self,
        *,
        solver: Any,
        coeffs,
        need_matrix: bool,
        backend: str | None = None,
    ) -> PointwiseQuadratureResult | None:
        raw_workset = self.build_pointwise_workset(
            solver=solver,
            coeffs=coeffs,
            need_matrix=bool(need_matrix),
        )
        workset = _coerce_pointwise_workset(
            raw_workset,
            solver=solver,
            coeffs=coeffs,
            need_matrix=bool(need_matrix),
        )
        if workset is None:
            return None
        if backend is not None:
            workset.backend = _normalize_backend_name(backend)
        result = self._run_pointwise_backend(workset)
        if result is None:
            return None
        self.apply_pointwise_result(
            solver=solver,
            workset=workset,
            result=result,
        )
        return result

    def before_assembly(self, *, solver: Any, coeffs, need_matrix: bool) -> None:
        self.run_pointwise(
            solver=solver,
            coeffs=coeffs,
            need_matrix=bool(need_matrix),
        )


class CallbackPointwiseQuadratureOperator(PointwiseQuadratureOperator):
    """Convenience wrapper for pointwise operators defined by callables."""

    def __init__(
        self,
        *,
        workset_builder,
        python_kernel=None,
        jit_kernel=None,
        cpp_kernel=None,
        fallback_kernel=None,
        result_applier=None,
    ) -> None:
        self._workset_builder = workset_builder
        self._python_kernel = python_kernel
        self._jit_kernel = jit_kernel
        self._cpp_kernel = cpp_kernel
        self._fallback_kernel = fallback_kernel
        self._result_applier = result_applier

    def build_pointwise_workset(self, *, solver: Any, coeffs, need_matrix: bool):
        return self._workset_builder(
            solver=solver,
            coeffs=coeffs,
            need_matrix=bool(need_matrix),
        )

    def update_pointwise(self, workset: PointwiseQuadratureWorkset):
        if self._fallback_kernel is None:
            return None
        return self._fallback_kernel(workset)

    def update_pointwise_python(self, workset: PointwiseQuadratureWorkset):
        if self._python_kernel is not None:
            return self._python_kernel(workset)
        return super().update_pointwise_python(workset)

    def update_pointwise_jit(self, workset: PointwiseQuadratureWorkset):
        if self._jit_kernel is not None:
            return self._jit_kernel(workset)
        return super().update_pointwise_jit(workset)

    def update_pointwise_cpp(self, workset: PointwiseQuadratureWorkset):
        if self._cpp_kernel is not None:
            return self._cpp_kernel(workset)
        return super().update_pointwise_cpp(workset)

    def apply_pointwise_result(
        self,
        *,
        solver: Any,
        workset: PointwiseQuadratureWorkset,
        result: PointwiseQuadratureResult,
    ) -> None:
        if self._result_applier is None:
            return
        self._result_applier(
            solver=solver,
            workset=workset,
            result=result,
        )


class SymbolicPointwiseNewtonOperator(PointwiseQuadratureOperator):
    """
    Generic symbolic pointwise Newton update for quadrature-state unknowns.

    The nonlinear residual and linearization are defined once as UFL
    expressions. The operator evaluates them through the existing backend-aware
    quadrature compiler and applies a batched Newton update on the target
    quadrature field.
    """

    def __init__(
        self,
        *,
        dof_handler,
        unknown_field: QuadratureStateField,
        residual_expr,
        jacobian_expr,
        entity_ids: np.ndarray | None = None,
        quadrature_order: int | None = None,
        max_iterations: int = 8,
        rel_tol: float = 1.0e-10,
        abs_tol: float = 1.0e-12,
        failure_mode: str = "zero",
        result_callback=None,
    ) -> None:
        self.dh = dof_handler
        self.unknown_field = unknown_field
        self.residual_expr = residual_expr
        self.jacobian_expr = jacobian_expr
        self.entity_ids = None if entity_ids is None else np.asarray(entity_ids, dtype=int).reshape(-1)
        self.quadrature_order = None if quadrature_order is None else int(quadrature_order)
        self.max_iterations = max(int(max_iterations), 1)
        self.rel_tol = float(rel_tol)
        self.abs_tol = float(abs_tol)
        mode = str(failure_mode).strip().lower()
        if mode not in {"zero", "keep_last"}:
            raise ValueError(f"Unsupported pointwise failure_mode {failure_mode!r}.")
        self.failure_mode = mode
        self._result_callback = result_callback
        self._compiler_cache: dict[tuple[int, str], Any] = {}

        if str(self.unknown_field.layout.entity_kind) != "volume_cell":
            raise NotImplementedError(
                "SymbolicPointwiseNewtonOperator currently supports only volume-cell quadrature layouts."
            )
        if self.unknown_field.layout.cell_type != str(self.dh.mixed_element.mesh.element_type).strip().lower():
            raise ValueError(
                "Quadrature layout cell type does not match the active mesh for pointwise symbolic updates."
            )

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

    def _component_count(self) -> int:
        tensor_shape = tuple(int(v) for v in self.unknown_field.tensor_shape)
        if tensor_shape == ():
            return 1
        if len(tensor_shape) == 1:
            return int(tensor_shape[0])
        raise NotImplementedError(
            "SymbolicPointwiseNewtonOperator currently supports only scalar or vector quadrature fields."
        )

    def _unknown_view(self, values) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        ncomp = self._component_count()
        if ncomp == 1:
            if arr.ndim != 2:
                raise ValueError("Scalar pointwise unknowns must have shape (n_entities, n_qp).")
            return arr.reshape(arr.shape[0], arr.shape[1], 1)
        if arr.ndim != 3 or int(arr.shape[2]) != ncomp:
            raise ValueError(
                f"Vector pointwise unknowns must have shape (n_entities, n_qp, {ncomp})."
            )
        return arr

    def _restore_unknown_shape(self, values_view: np.ndarray) -> np.ndarray:
        if self._component_count() == 1:
            return np.asarray(values_view[..., 0], dtype=float)
        return np.asarray(values_view, dtype=float)

    def _normalize_residual(self, residual_raw, *, entity_ids: np.ndarray) -> np.ndarray:
        arr = np.asarray(residual_raw, dtype=float)
        ncomp = self._component_count()
        n_entities = int(entity_ids.shape[0])
        n_qp = int(self.unknown_field.layout.n_qp)
        if ncomp == 1:
            if arr.shape != (n_entities, n_qp):
                raise ValueError(
                    f"Scalar pointwise residual expected shape {(n_entities, n_qp)}, got {arr.shape}."
                )
            return arr.reshape(n_entities, n_qp, 1)
        if arr.shape != (n_entities, n_qp, ncomp):
            raise ValueError(
                f"Vector pointwise residual expected shape {(n_entities, n_qp, ncomp)}, got {arr.shape}."
            )
        return arr

    def _normalize_jacobian(self, jacobian_raw, *, entity_ids: np.ndarray) -> np.ndarray:
        arr = np.asarray(jacobian_raw, dtype=float)
        ncomp = self._component_count()
        n_entities = int(entity_ids.shape[0])
        n_qp = int(self.unknown_field.layout.n_qp)
        if ncomp == 1:
            if arr.shape != (n_entities, n_qp):
                raise ValueError(
                    f"Scalar pointwise Jacobian expected shape {(n_entities, n_qp)}, got {arr.shape}."
                )
            return arr.reshape(n_entities, n_qp, 1, 1)
        if arr.shape != (n_entities, n_qp, ncomp, ncomp):
            raise ValueError(
                f"Vector pointwise Jacobian expected shape {(n_entities, n_qp, ncomp, ncomp)}, got {arr.shape}."
            )
        return arr

    @staticmethod
    def _solve_active_systems(matrices: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if int(matrices.shape[0]) == 0:
            return np.zeros_like(rhs, dtype=float), np.zeros((0,), dtype=bool)
        rhs_arr = np.asarray(rhs, dtype=float)
        try:
            if rhs_arr.ndim == matrices.ndim - 1 and rhs_arr.shape[-1] == matrices.shape[-1]:
                solved = np.asarray(np.linalg.solve(matrices, rhs_arr[..., None]), dtype=float)[..., 0]
            else:
                solved = np.asarray(np.linalg.solve(matrices, rhs_arr), dtype=float)
            return solved, np.ones((matrices.shape[0],), dtype=bool)
        except np.linalg.LinAlgError:
            delta = np.zeros_like(rhs_arr, dtype=float)
            solved = np.zeros((matrices.shape[0],), dtype=bool)
            for idx in range(int(matrices.shape[0])):
                try:
                    delta[idx, :] = np.asarray(np.linalg.solve(matrices[idx], rhs_arr[idx]), dtype=float)
                    solved[idx] = True
                except np.linalg.LinAlgError:
                    solved[idx] = False
            return delta, solved

    def build_pointwise_workset(self, *, solver: Any, coeffs, need_matrix: bool):
        del coeffs
        if self.entity_ids is None:
            entity_ids = np.arange(int(self.unknown_field.n_entities), dtype=int)
        else:
            entity_ids = np.asarray(self.entity_ids, dtype=int).reshape(-1)
        return PointwiseQuadratureWorkset(
            solver=solver,
            coeffs=None,
            need_matrix=bool(need_matrix),
            backend=str(getattr(solver, "backend", "python")),
            layout=self.unknown_field.layout,
            entity_ids=entity_ids,
            payload={},
        )

    def update_pointwise(self, workset: PointwiseQuadratureWorkset):
        compiler = self._compiler(workset.backend)
        entity_ids = np.asarray(workset.entity_ids, dtype=int).reshape(-1)
        values_all = np.asarray(self.unknown_field.values, dtype=float)
        original = np.asarray(values_all[entity_ids], dtype=float).copy()
        current = np.asarray(original, dtype=float).copy()
        current_view = self._unknown_view(current)
        active = np.ones(current_view.shape[:2], dtype=bool)
        converged = np.zeros_like(active)
        failed = np.zeros_like(active)

        try:
            for _ in range(int(self.max_iterations)):
                values_all[entity_ids] = self._restore_unknown_shape(current_view)
                evaluated = compiler.evaluate_volume_expressions_on_quadrature(
                    {
                        "pointwise_residual": self.residual_expr,
                        "pointwise_jacobian": self.jacobian_expr,
                    },
                    layout=workset.layout,
                    element_ids=entity_ids,
                )
                residual = self._normalize_residual(
                    evaluated["pointwise_residual"],
                    entity_ids=entity_ids,
                )
                jacobian = self._normalize_jacobian(
                    evaluated["pointwise_jacobian"],
                    entity_ids=entity_ids,
                )
                if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jacobian)):
                    raise RuntimeError("Pointwise symbolic Newton evaluation produced non-finite residual/Jacobian values.")

                flat_active = active.reshape(-1)
                if not np.any(flat_active):
                    break

                residual_flat = residual.reshape(-1, residual.shape[-1])
                jacobian_flat = jacobian.reshape(-1, jacobian.shape[-2], jacobian.shape[-1])
                current_flat = current_view.reshape(-1, current_view.shape[-1])
                active_idx = np.flatnonzero(flat_active)
                delta_active, solved_active = self._solve_active_systems(
                    jacobian_flat[active_idx],
                    residual_flat[active_idx],
                )
                valid_active = solved_active & np.all(np.isfinite(delta_active), axis=1)
                invalid_idx = active_idx[~valid_active]
                if invalid_idx.size:
                    failed.reshape(-1)[invalid_idx] = True
                    active.reshape(-1)[invalid_idx] = False
                    current_flat[invalid_idx, :] = 0.0

                if not np.any(valid_active):
                    continue

                solved_idx = active_idx[valid_active]
                current_flat[solved_idx, :] += delta_active[valid_active, :]
                err = np.linalg.norm(delta_active[valid_active, :], axis=1)
                norm_u = np.linalg.norm(current_flat[solved_idx, :], axis=1)
                converged_now = (err <= self.abs_tol) | (
                    (norm_u > self.rel_tol) & ((err / norm_u) <= self.rel_tol)
                )
                converged.reshape(-1)[solved_idx[converged_now]] = True
                active.reshape(-1)[solved_idx[converged_now]] = False

            if np.any(active):
                failed |= active
                if self.failure_mode == "zero":
                    current_view[active, :] = 0.0
        except Exception:
            values_all[entity_ids] = original
            raise

        values = self._restore_unknown_shape(current_view)
        values_all[entity_ids] = values
        return PointwiseQuadratureResult(
            values=values,
            entity_ids=entity_ids,
            metadata={
                "converged_count": int(np.count_nonzero(converged)),
                "failed_count": int(np.count_nonzero(failed)),
                "n_entities": int(entity_ids.shape[0]),
            },
        )

    def apply_pointwise_result(
        self,
        *,
        solver: Any,
        workset: PointwiseQuadratureWorkset,
        result: PointwiseQuadratureResult,
    ) -> None:
        entity_ids = np.asarray(result.entity_ids, dtype=int).reshape(-1)
        self.unknown_field.values[entity_ids] = np.asarray(result.values, dtype=float)
        if self._result_callback is not None:
            self._result_callback(
                solver=solver,
                workset=workset,
                result=result,
            )


__all__ = [
    "CallbackPointwiseQuadratureOperator",
    "PointwiseQuadratureOperator",
    "PointwiseQuadratureResult",
    "PointwiseQuadratureWorkset",
    "SymbolicPointwiseNewtonOperator",
]
