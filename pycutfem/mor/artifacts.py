"""Native reduced-problem artifact schema.

Artifacts are stored as an ``.npz`` containing a JSON manifest plus typed array
payloads.  The schema is intentionally MOR-level: it records kernels,
hyper-reduction targets, decomposition metadata, state updates, and solver
options without depending on a specific example problem.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .decomposition import EmpiricalCubatureRule, InterpolationRule, NativeReducedEvaluationGraph
from .constraints import BoundConstraintSpec
from .reference import ReferencePolicy
from .sparse import NativeSparseMatrix, is_sparse_matrix_like
from .state_updates import AffineStateUpdateSpec, StateTransactionSpec, SymbolicStateUpdateKernelSpec


NATIVE_REDUCED_ARTIFACT_SCHEMA_VERSION = 1


def _finite_matrix(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2-D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _finite_vector(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _encode_jsonable(value: Any, arrays: dict[str, np.ndarray], prefix: str) -> Any:
    if isinstance(value, np.ndarray):
        key = f"{prefix}_{len(arrays)}"
        arrays[key] = np.ascontiguousarray(value)
        return {"__ndarray__": key}
    if isinstance(value, NativeSparseMatrix):
        return {"__native_sparse__": _encode_jsonable(value.to_native_dict(), arrays, f"{prefix}_sparse")}
    if isinstance(value, InterpolationRule):
        return {"__interpolation_rule__": _encode_jsonable(value.to_native_dict(), arrays, f"{prefix}_interp")}
    if isinstance(value, EmpiricalCubatureRule):
        return {"__cubature_rule__": _encode_jsonable(value.to_native_dict(), arrays, f"{prefix}_cubature")}
    if isinstance(value, NativeReducedEvaluationGraph):
        return {"__evaluation_graph__": _encode_jsonable(value.to_native_dict(), arrays, f"{prefix}_graph")}
    if isinstance(value, AffineStateUpdateSpec):
        return {"__affine_update__": _encode_jsonable(value.to_native_dict(), arrays, f"{prefix}_affine")}
    if isinstance(value, SymbolicStateUpdateKernelSpec):
        return {"__symbolic_update__": _encode_jsonable(value.to_native_dict(), arrays, f"{prefix}_symbolic")}
    if isinstance(value, StateTransactionSpec):
        return {"__state_transaction__": _encode_jsonable(value.to_native_dict(), arrays, f"{prefix}_state")}
    if isinstance(value, Mapping):
        return {str(k): _encode_jsonable(v, arrays, f"{prefix}_{k}") for k, v in value.items()}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_jsonable(v, arrays, f"{prefix}_{i}") for i, v in enumerate(value)]}
    if isinstance(value, list):
        return [_encode_jsonable(v, arrays, f"{prefix}_{i}") for i, v in enumerate(value)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _decode_jsonable(value: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    if isinstance(value, list):
        return [_decode_jsonable(v, arrays) for v in value]
    if not isinstance(value, Mapping):
        return value
    if "__ndarray__" in value:
        return np.asarray(arrays[str(value["__ndarray__"])])
    if "__tuple__" in value:
        return tuple(_decode_jsonable(v, arrays) for v in value["__tuple__"])
    if "__native_sparse__" in value:
        return NativeSparseMatrix.from_native_dict(_decode_jsonable(value["__native_sparse__"], arrays))
    if "__interpolation_rule__" in value:
        return InterpolationRule.from_native_dict(_decode_jsonable(value["__interpolation_rule__"], arrays))
    if "__cubature_rule__" in value:
        raw = _decode_jsonable(value["__cubature_rule__"], arrays)
        return EmpiricalCubatureRule(
            entity_ids=raw["entity_ids"],
            weights=raw["weights"],
            entity_kind=str(raw.get("entity_kind", "cell")),
            metadata=raw.get("metadata", {}),
        )
    if "__evaluation_graph__" in value:
        return _decode_jsonable(value["__evaluation_graph__"], arrays)
    if "__affine_update__" in value:
        return AffineStateUpdateSpec.from_native_dict(_decode_jsonable(value["__affine_update__"], arrays))
    if "__symbolic_update__" in value:
        raw = _decode_jsonable(value["__symbolic_update__"], arrays)
        return SymbolicStateUpdateKernelSpec(
            name=raw["name"],
            kernel_id=raw["kernel_id"],
            abi=raw["abi"],
            param_order=tuple(raw["param_order"]),
            target_names=tuple(raw["target_names"]),
            argument_map=raw.get("argument_map", {}),
            stage=raw.get("stage", "pre_residual"),
            metadata=raw.get("metadata", {}),
        )
    if "__state_transaction__" in value:
        return _decode_jsonable(value["__state_transaction__"], arrays)
    return {str(k): _decode_jsonable(v, arrays) for k, v in value.items()}


@dataclass(frozen=True)
class NativeKernelReference:
    """Serializable reference to a generated native UFL/state kernel."""

    kernel_id: str
    abi: str
    param_order: tuple[str, ...]
    source: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        param_order = tuple(str(v) for v in self.param_order)
        if not param_order:
            raise ValueError("kernel reference param_order must not be empty.")
        object.__setattr__(self, "kernel_id", str(self.kernel_id))
        object.__setattr__(self, "abi", str(self.abi))
        object.__setattr__(self, "param_order", param_order)
        object.__setattr__(self, "source", None if self.source is None else str(self.source))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_native_dict(self) -> dict[str, Any]:
        return {
            "kernel_id": self.kernel_id,
            "abi": self.abi,
            "param_order": tuple(self.param_order),
            "source": self.source,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_native_dict(cls, payload: Mapping[str, Any]) -> "NativeKernelReference":
        return cls(
            kernel_id=str(payload["kernel_id"]),
            abi=str(payload["abi"]),
            param_order=tuple(payload["param_order"]),
            source=payload.get("source"),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class NativeGnatTargetSpec:
    """Sampled LSPG/GNAT target data for a native reduced solve."""

    row_dofs: np.ndarray
    element_ids: np.ndarray | None = None
    element_weights: np.ndarray | None = None
    row_weights: np.ndarray | None = None
    lift: np.ndarray | NativeSparseMatrix | Mapping[str, Any] | None = None
    selected_basis: np.ndarray | None = None
    residual_terms: np.ndarray | None = None
    objective: str = "sampled_lspg"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        row_dofs = np.asarray(self.row_dofs, dtype=np.int64).reshape(-1)
        if row_dofs.size == 0 or np.any(row_dofs < 0) or np.unique(row_dofs).size != row_dofs.size:
            raise ValueError("target row_dofs must be nonempty, nonnegative, and unique.")
        element_ids = None if self.element_ids is None else np.asarray(self.element_ids, dtype=np.int64).reshape(-1)
        if element_ids is not None and np.any(element_ids < 0):
            raise ValueError("target element_ids must be nonnegative.")
        element_weights = None if self.element_weights is None else _finite_vector(self.element_weights, "target element weights")
        row_weights = None if self.row_weights is None else _finite_vector(self.row_weights, "target row weights")
        if row_weights is not None and row_weights.size != row_dofs.size:
            raise ValueError("target row_weights size must match row_dofs.")
        selected_basis = None if self.selected_basis is None else _finite_matrix(self.selected_basis, "target selected_basis")
        residual_terms = None if self.residual_terms is None else _finite_matrix(self.residual_terms, "target residual_terms")
        if (selected_basis is None) != (residual_terms is None):
            raise ValueError("selected_basis and residual_terms must be provided together for DEIM/QDEIM targets.")
        if selected_basis is not None:
            if selected_basis.shape[0] != row_dofs.size:
                raise ValueError("target selected_basis row count must match row_dofs.")
            if residual_terms.shape[0] != selected_basis.shape[1]:
                raise ValueError("target residual_terms row count must match selected_basis column count.")
            if residual_terms.shape[1] == 0:
                raise ValueError("target residual_terms must contain at least one residual row.")
        target_lift_cols = int(residual_terms.shape[1]) if residual_terms is not None else int(row_dofs.size)
        lift = self.lift
        if lift is not None:
            lift = NativeSparseMatrix.coerce(lift) if is_sparse_matrix_like(lift) else _finite_matrix(lift, "target dense GNAT lift")
            lift_cols = lift.shape[1] if isinstance(lift, NativeSparseMatrix) else lift.shape[1]
            if int(lift_cols) != target_lift_cols:
                raise ValueError("target GNAT lift columns must match the native target residual size.")
        object.__setattr__(self, "row_dofs", np.ascontiguousarray(row_dofs, dtype=np.int64))
        object.__setattr__(self, "element_ids", None if element_ids is None else np.ascontiguousarray(element_ids, dtype=np.int64))
        object.__setattr__(self, "element_weights", element_weights)
        object.__setattr__(self, "row_weights", row_weights)
        object.__setattr__(self, "lift", lift)
        object.__setattr__(self, "selected_basis", selected_basis)
        object.__setattr__(self, "residual_terms", residual_terms)
        object.__setattr__(self, "objective", str(self.objective).lower())
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_native_dict(self) -> dict[str, Any]:
        lift = self.lift
        if isinstance(lift, NativeSparseMatrix):
            lift_payload: Any = lift.to_native_dict()
        else:
            lift_payload = lift
        return {
            "row_dofs": self.row_dofs,
            "element_ids": np.zeros(0, dtype=np.int64) if self.element_ids is None else self.element_ids,
            "element_weights": np.zeros(0, dtype=float) if self.element_weights is None else self.element_weights,
            "row_weights": np.zeros(0, dtype=float) if self.row_weights is None else self.row_weights,
            "lift": lift_payload,
            "selected_basis": self.selected_basis,
            "residual_terms": self.residual_terms,
            "objective": self.objective,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class NativeAdjointDWRSpec:
    """Artifact metadata for native adjoint and DWR certification."""

    qoi_name: str
    qoi_kernel: NativeKernelReference | Mapping[str, Any] | None = None
    qoi_current_gradient_kernel: NativeKernelReference | Mapping[str, Any] | None = None
    qoi_previous_gradient_kernel: NativeKernelReference | Mapping[str, Any] | None = None
    qoi_element_kernel: NativeKernelReference | Mapping[str, Any] | None = None
    adjoint_basis: np.ndarray | None = None
    transient_dependency: Mapping[str, Any] | None = None
    checkpoint_policy: Mapping[str, Any] | None = None
    field_layout_signature: Mapping[str, Any] | None = None
    pressure_gauge: Mapping[str, Any] | None = None
    norm_equivalence_certificate: Mapping[str, Any] | None = None
    solver_options: Mapping[str, Any] | None = None
    estimator_options: Mapping[str, Any] | None = None
    certification: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        def _kernel(value: NativeKernelReference | Mapping[str, Any] | None) -> NativeKernelReference | None:
            if value is None:
                return None
            if isinstance(value, NativeKernelReference):
                return value
            if isinstance(value, Mapping):
                return NativeKernelReference.from_native_dict(value)
            raise TypeError("adjoint/DWR kernel entries must be NativeKernelReference, mapping, or None.")

        basis = None if self.adjoint_basis is None else _finite_matrix(self.adjoint_basis, "adjoint basis")
        object.__setattr__(self, "qoi_name", str(self.qoi_name))
        object.__setattr__(self, "qoi_kernel", _kernel(self.qoi_kernel))
        object.__setattr__(self, "qoi_current_gradient_kernel", _kernel(self.qoi_current_gradient_kernel))
        object.__setattr__(self, "qoi_previous_gradient_kernel", _kernel(self.qoi_previous_gradient_kernel))
        object.__setattr__(self, "qoi_element_kernel", _kernel(self.qoi_element_kernel))
        object.__setattr__(self, "adjoint_basis", basis)
        object.__setattr__(self, "transient_dependency", dict(self.transient_dependency or {}))
        object.__setattr__(self, "checkpoint_policy", dict(self.checkpoint_policy or {}))
        object.__setattr__(self, "field_layout_signature", dict(self.field_layout_signature or {}))
        object.__setattr__(self, "pressure_gauge", dict(self.pressure_gauge or {}))
        object.__setattr__(self, "norm_equivalence_certificate", dict(self.norm_equivalence_certificate or {}))
        object.__setattr__(self, "solver_options", dict(self.solver_options or {}))
        object.__setattr__(self, "estimator_options", dict(self.estimator_options or {}))
        object.__setattr__(self, "certification", dict(self.certification or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_native_dict(self) -> dict[str, Any]:
        return {
            "qoi_name": self.qoi_name,
            "qoi_kernel": None if self.qoi_kernel is None else self.qoi_kernel.to_native_dict(),
            "qoi_current_gradient_kernel": (
                None if self.qoi_current_gradient_kernel is None else self.qoi_current_gradient_kernel.to_native_dict()
            ),
            "qoi_previous_gradient_kernel": (
                None if self.qoi_previous_gradient_kernel is None else self.qoi_previous_gradient_kernel.to_native_dict()
            ),
            "qoi_element_kernel": None if self.qoi_element_kernel is None else self.qoi_element_kernel.to_native_dict(),
            "adjoint_basis": self.adjoint_basis,
            "transient_dependency": dict(self.transient_dependency or {}),
            "checkpoint_policy": dict(self.checkpoint_policy or {}),
            "field_layout_signature": dict(self.field_layout_signature or {}),
            "pressure_gauge": dict(self.pressure_gauge or {}),
            "norm_equivalence_certificate": dict(self.norm_equivalence_certificate or {}),
            "solver_options": dict(self.solver_options or {}),
            "estimator_options": dict(self.estimator_options or {}),
            "certification": dict(self.certification or {}),
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_native_dict(cls, payload: Mapping[str, Any]) -> "NativeAdjointDWRSpec":
        return cls(
            qoi_name=str(payload["qoi_name"]),
            qoi_kernel=payload.get("qoi_kernel"),
            qoi_current_gradient_kernel=payload.get("qoi_current_gradient_kernel"),
            qoi_previous_gradient_kernel=payload.get("qoi_previous_gradient_kernel"),
            qoi_element_kernel=payload.get("qoi_element_kernel"),
            adjoint_basis=payload.get("adjoint_basis"),
            transient_dependency=payload.get("transient_dependency", {}),
            checkpoint_policy=payload.get("checkpoint_policy", {}),
            field_layout_signature=payload.get("field_layout_signature", {}),
            pressure_gauge=payload.get("pressure_gauge", {}),
            norm_equivalence_certificate=payload.get("norm_equivalence_certificate", {}),
            solver_options=payload.get("solver_options", {}),
            estimator_options=payload.get("estimator_options", {}),
            certification=payload.get("certification", {}),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class NativeReducedArtifact:
    """Complete problem-generic native reduced model artifact."""

    problem_id: str
    trial_basis: np.ndarray
    offset: np.ndarray
    residual_kernel: NativeKernelReference
    tangent_kernel: NativeKernelReference | None = None
    target: NativeGnatTargetSpec | None = None
    bound_constraints: BoundConstraintSpec | Mapping[str, Any] | None = None
    evaluation_graph: NativeReducedEvaluationGraph | Mapping[str, Any] | None = None
    state_transaction: StateTransactionSpec | Mapping[str, Any] | None = None
    adjoint_dwr: NativeAdjointDWRSpec | Mapping[str, Any] | None = None
    reference_policy: ReferencePolicy | Mapping[str, Any] | None = None
    solver_options: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None
    schema_version: int = NATIVE_REDUCED_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        basis = _finite_matrix(self.trial_basis, "artifact trial basis")
        offset = _finite_vector(self.offset, "artifact offset")
        if basis.shape[0] != offset.size:
            raise ValueError("artifact trial_basis rows must match offset size.")
        if int(self.schema_version) != NATIVE_REDUCED_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported native reduced artifact schema version {self.schema_version}.")
        bound_constraints = self.bound_constraints
        if bound_constraints is not None and not isinstance(bound_constraints, BoundConstraintSpec):
            if not isinstance(bound_constraints, Mapping):
                raise TypeError("artifact bound_constraints must be a BoundConstraintSpec or mapping.")
            bound_constraints = BoundConstraintSpec.from_native_dict(bound_constraints)
        adjoint_dwr = self.adjoint_dwr
        if adjoint_dwr is not None and not isinstance(adjoint_dwr, NativeAdjointDWRSpec):
            if not isinstance(adjoint_dwr, Mapping):
                raise TypeError("artifact adjoint_dwr must be a NativeAdjointDWRSpec or mapping.")
            adjoint_dwr = NativeAdjointDWRSpec.from_native_dict(adjoint_dwr)
        reference_policy = self.reference_policy
        if reference_policy is not None and not isinstance(reference_policy, ReferencePolicy):
            if not isinstance(reference_policy, Mapping):
                raise TypeError("artifact reference_policy must be a ReferencePolicy or mapping.")
            reference_policy = ReferencePolicy.from_native_dict(reference_policy)
        object.__setattr__(self, "problem_id", str(self.problem_id))
        object.__setattr__(self, "trial_basis", basis)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "bound_constraints", bound_constraints)
        object.__setattr__(self, "adjoint_dwr", adjoint_dwr)
        object.__setattr__(self, "reference_policy", reference_policy)
        object.__setattr__(self, "solver_options", dict(self.solver_options or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "problem_id": self.problem_id,
            "trial_basis": self.trial_basis,
            "offset": self.offset,
            "residual_kernel": self.residual_kernel.to_native_dict(),
            "tangent_kernel": None if self.tangent_kernel is None else self.tangent_kernel.to_native_dict(),
            "target": None if self.target is None else self.target.to_native_dict(),
            "bound_constraints": None if self.bound_constraints is None else self.bound_constraints.to_native_dict(),
            "evaluation_graph": self.evaluation_graph,
            "state_transaction": self.state_transaction,
            "adjoint_dwr": None if self.adjoint_dwr is None else self.adjoint_dwr.to_native_dict(),
            "reference_policy": None if self.reference_policy is None else self.reference_policy.to_native_dict(),
            "solver_options": dict(self.solver_options or {}),
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NativeReducedArtifact":
        target_payload = payload.get("target")
        target = None
        if target_payload is not None:
            lift_payload = target_payload.get("lift")
            lift = None
            if lift_payload is not None:
                lift = NativeSparseMatrix.from_native_dict(lift_payload) if isinstance(lift_payload, Mapping) and lift_payload.get("layout") == "csr" else lift_payload
            target = NativeGnatTargetSpec(
                row_dofs=target_payload["row_dofs"],
                element_ids=None if np.asarray(target_payload.get("element_ids", [])).size == 0 else target_payload["element_ids"],
                element_weights=None if np.asarray(target_payload.get("element_weights", [])).size == 0 else target_payload["element_weights"],
                row_weights=None if np.asarray(target_payload.get("row_weights", [])).size == 0 else target_payload["row_weights"],
                lift=lift,
                selected_basis=target_payload.get("selected_basis"),
                residual_terms=target_payload.get("residual_terms"),
                objective=str(target_payload.get("objective", "sampled_lspg")),
                metadata=target_payload.get("metadata", {}),
            )
        tangent_payload = payload.get("tangent_kernel")
        bound_payload = payload.get("bound_constraints")
        adjoint_payload = payload.get("adjoint_dwr")
        reference_payload = payload.get("reference_policy")
        return cls(
            problem_id=str(payload["problem_id"]),
            trial_basis=payload["trial_basis"],
            offset=payload["offset"],
            residual_kernel=NativeKernelReference.from_native_dict(payload["residual_kernel"]),
            tangent_kernel=None if tangent_payload is None else NativeKernelReference.from_native_dict(tangent_payload),
            target=target,
            bound_constraints=None if bound_payload is None else BoundConstraintSpec.from_native_dict(bound_payload),
            evaluation_graph=payload.get("evaluation_graph"),
            state_transaction=payload.get("state_transaction"),
            adjoint_dwr=None if adjoint_payload is None else NativeAdjointDWRSpec.from_native_dict(adjoint_payload),
            reference_policy=None if reference_payload is None else ReferencePolicy.from_native_dict(reference_payload),
            solver_options=payload.get("solver_options", {}),
            metadata=payload.get("metadata", {}),
            schema_version=int(payload.get("schema_version", NATIVE_REDUCED_ARTIFACT_SCHEMA_VERSION)),
        )

    def save(self, path: str | Path) -> None:
        save_native_reduced_artifact(self, path)

    def instantiate(
        self,
        *,
        residual_metadata_capsule: Any,
        residual_static_args: Mapping[str, Any],
        tangent_metadata_capsule: Any | None = None,
        tangent_static_args: Mapping[str, Any] | None = None,
        coefficient_arg_names: tuple[str, ...] | None = None,
        residual_state_updates: Any = None,
        tangent_state_updates: Any = None,
        residual_symbolic_state_updates: Any = None,
        tangent_symbolic_state_updates: Any = None,
    ) -> "NativeReducedRuntimeProblem":
        return NativeReducedRuntimeProblem(
            artifact=self,
            residual_metadata_capsule=residual_metadata_capsule,
            residual_static_args=dict(residual_static_args),
            tangent_metadata_capsule=tangent_metadata_capsule,
            tangent_static_args=None if tangent_static_args is None else dict(tangent_static_args),
            coefficient_arg_names=coefficient_arg_names,
            residual_state_updates=residual_state_updates,
            tangent_state_updates=tangent_state_updates,
            residual_symbolic_state_updates=residual_symbolic_state_updates,
            tangent_symbolic_state_updates=tangent_symbolic_state_updates,
        )


@dataclass(frozen=True)
class NativeReducedRuntimeProblem:
    """Live native problem assembled from an artifact and compiled kernels."""

    artifact: NativeReducedArtifact
    residual_metadata_capsule: Any
    residual_static_args: Mapping[str, Any]
    tangent_metadata_capsule: Any | None = None
    tangent_static_args: Mapping[str, Any] | None = None
    coefficient_arg_names: tuple[str, ...] | None = None
    residual_state_updates: Any = None
    tangent_state_updates: Any = None
    residual_symbolic_state_updates: Any = None
    tangent_symbolic_state_updates: Any = None

    def solve(self, initial_coefficients: Any, **solver_options: Any) -> Any:
        if self.artifact.target is None:
            raise ValueError("native reduced runtime solve requires an artifact target.")
        if self.artifact.tangent_kernel is None or self.tangent_metadata_capsule is None or self.tangent_static_args is None:
            raise ValueError("native reduced runtime solve requires a tangent kernel and tangent static args.")

        target = self.artifact.target
        options = dict(self.artifact.solver_options or {})
        options.update(solver_options)
        constraint_method = options.pop("constraint_method", None)
        bound_constrained = bool(options.pop("bound_constrained", False))
        coeff_names = self.coefficient_arg_names
        if coeff_names is None:
            raw_names = self.artifact.metadata.get("coefficient_arg_names", ())
            coeff_names = tuple(str(name) for name in raw_names)
        if coeff_names is not None and len(coeff_names) == 0:
            coeff_names = None
        if constraint_method is not None or bound_constrained:
            if self.artifact.bound_constraints is None:
                raise ValueError("bound-constrained runtime solve requires artifact bound_constraints.")
            if target.selected_basis is not None and target.residual_terms is not None:
                from .online_gauss_newton import solve_native_bound_constrained_deim_online_gauss_newton

                return solve_native_bound_constrained_deim_online_gauss_newton(
                    residual_metadata_capsule=self.residual_metadata_capsule,
                    residual_param_order=self.artifact.residual_kernel.param_order,
                    residual_static_args=self.residual_static_args,
                    tangent_metadata_capsule=self.tangent_metadata_capsule,
                    tangent_param_order=self.artifact.tangent_kernel.param_order,
                    tangent_static_args=self.tangent_static_args,
                    trial_basis=self.artifact.trial_basis,
                    offset=self.artifact.offset,
                    initial_coefficients=initial_coefficients,
                    row_dofs=target.row_dofs,
                    selected_basis=target.selected_basis,
                    residual_terms=target.residual_terms,
                    bound_constraints=self.artifact.bound_constraints,
                    constraint_method="pdas" if constraint_method is None else str(constraint_method),
                    coefficient_arg_names=coeff_names,
                    element_weights=target.element_weights,
                    row_weights=target.row_weights,
                    gnat_lift=target.lift,
                    residual_state_updates=self.residual_state_updates,
                    tangent_state_updates=self.tangent_state_updates,
                    residual_symbolic_state_updates=self.residual_symbolic_state_updates,
                    tangent_symbolic_state_updates=self.tangent_symbolic_state_updates,
                    **options,
                )

            from .online_gauss_newton import solve_native_bound_constrained_online_gauss_newton

            return solve_native_bound_constrained_online_gauss_newton(
                residual_metadata_capsule=self.residual_metadata_capsule,
                residual_param_order=self.artifact.residual_kernel.param_order,
                residual_static_args=self.residual_static_args,
                tangent_metadata_capsule=self.tangent_metadata_capsule,
                tangent_param_order=self.artifact.tangent_kernel.param_order,
                tangent_static_args=self.tangent_static_args,
                trial_basis=self.artifact.trial_basis,
                offset=self.artifact.offset,
                initial_coefficients=initial_coefficients,
                row_dofs=target.row_dofs,
                bound_constraints=self.artifact.bound_constraints,
                constraint_method="pdas" if constraint_method is None else str(constraint_method),
                coefficient_arg_names=coeff_names,
                element_weights=target.element_weights,
                row_weights=target.row_weights,
                gnat_lift=target.lift,
                residual_state_updates=self.residual_state_updates,
                tangent_state_updates=self.tangent_state_updates,
                residual_symbolic_state_updates=self.residual_symbolic_state_updates,
                tangent_symbolic_state_updates=self.tangent_symbolic_state_updates,
                **options,
            )

        if target.selected_basis is not None and target.residual_terms is not None:
            from .online_gauss_newton import solve_native_deim_online_gauss_newton

            return solve_native_deim_online_gauss_newton(
                residual_metadata_capsule=self.residual_metadata_capsule,
                residual_param_order=self.artifact.residual_kernel.param_order,
                residual_static_args=self.residual_static_args,
                tangent_metadata_capsule=self.tangent_metadata_capsule,
                tangent_param_order=self.artifact.tangent_kernel.param_order,
                tangent_static_args=self.tangent_static_args,
                trial_basis=self.artifact.trial_basis,
                offset=self.artifact.offset,
                initial_coefficients=initial_coefficients,
                row_dofs=target.row_dofs,
                selected_basis=target.selected_basis,
                residual_terms=target.residual_terms,
                coefficient_arg_names=coeff_names,
                element_weights=target.element_weights,
                row_weights=target.row_weights,
                gnat_lift=target.lift,
                residual_state_updates=self.residual_state_updates,
                tangent_state_updates=self.tangent_state_updates,
                residual_symbolic_state_updates=self.residual_symbolic_state_updates,
                tangent_symbolic_state_updates=self.tangent_symbolic_state_updates,
                **options,
            )

        from .online_gauss_newton import solve_native_online_gauss_newton

        return solve_native_online_gauss_newton(
            residual_metadata_capsule=self.residual_metadata_capsule,
            residual_param_order=self.artifact.residual_kernel.param_order,
            residual_static_args=self.residual_static_args,
            tangent_metadata_capsule=self.tangent_metadata_capsule,
            tangent_param_order=self.artifact.tangent_kernel.param_order,
            tangent_static_args=self.tangent_static_args,
            trial_basis=self.artifact.trial_basis,
            offset=self.artifact.offset,
            initial_coefficients=initial_coefficients,
            row_dofs=target.row_dofs,
            coefficient_arg_names=coeff_names,
            element_weights=target.element_weights,
            row_weights=target.row_weights,
            gnat_lift=target.lift,
            residual_state_updates=self.residual_state_updates,
            tangent_state_updates=self.tangent_state_updates,
            residual_symbolic_state_updates=self.residual_symbolic_state_updates,
            tangent_symbolic_state_updates=self.tangent_symbolic_state_updates,
            **options,
        )


def save_native_reduced_artifact(artifact: NativeReducedArtifact, path: str | Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    manifest = _encode_jsonable(artifact.to_dict(), arrays, "artifact")
    arrays["manifest_json"] = np.asarray(json.dumps(manifest, sort_keys=True))
    np.savez_compressed(Path(path), **arrays)


def load_native_reduced_artifact(path: str | Path) -> NativeReducedArtifact:
    with np.load(Path(path), allow_pickle=False) as data:
        manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
        arrays = {key: np.asarray(data[key]) for key in data.files if key != "manifest_json"}
        payload = _decode_jsonable(manifest, arrays)
    return NativeReducedArtifact.from_dict(payload)


__all__ = [
    "NATIVE_REDUCED_ARTIFACT_SCHEMA_VERSION",
    "NativeAdjointDWRSpec",
    "NativeGnatTargetSpec",
    "NativeKernelReference",
    "NativeReducedArtifact",
    "NativeReducedRuntimeProblem",
    "load_native_reduced_artifact",
    "save_native_reduced_artifact",
]
