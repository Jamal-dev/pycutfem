"""Discrete adjoint and dual-weighted residual helpers for certified MOR.

The helpers in this module are algebraic on purpose.  UFL/generated kernels
provide residuals, Jacobians, previous-state Jacobians, and QoI derivatives;
this layer owns the backend-agnostic discrete adjoint recursion and DWR
estimator bookkeeping that can be reused by fluid, FSI, and multi-constituent
models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np


AdjointBackend = Literal["python", "cpp", "c++"]
QoiTimeAggregation = Literal["final", "sum", "time_integral", "checkpoint_sum"]


@dataclass(frozen=True)
class QoIFunctionalSpec:
    """Problem-level metadata for a scalar quantity of interest."""

    name: str
    aggregation: QoiTimeAggregation = "final"
    fields: tuple[str, ...] = ()
    tolerance: float | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class QoIKernelSpec:
    """Native UFL QoI kernel references used by future artifact/runtime code."""

    value_kernel: Any | None = None
    current_gradient_kernel: Any | None = None
    previous_gradient_kernel: Any | None = None
    element_contribution_kernel: Any | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TransientResidualDependencySpec:
    """Discrete-time dependency metadata for adjoint/DWR reconstruction."""

    current_state: bool = True
    previous_state: bool = False
    history_width: int = 1
    parameter_names: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        width = int(self.history_width)
        if width < 0:
            raise ValueError("history_width must be nonnegative.")
        object.__setattr__(self, "current_state", bool(self.current_state))
        object.__setattr__(self, "previous_state", bool(self.previous_state))
        object.__setattr__(self, "history_width", width)
        object.__setattr__(self, "parameter_names", tuple(str(name) for name in self.parameter_names))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_state": bool(self.current_state),
            "previous_state": bool(self.previous_state),
            "history_width": int(self.history_width),
            "parameter_names": tuple(self.parameter_names),
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransientResidualDependencySpec":
        return cls(
            current_state=bool(payload.get("current_state", True)),
            previous_state=bool(payload.get("previous_state", False)),
            history_width=int(payload.get("history_width", 1)),
            parameter_names=tuple(str(name) for name in payload.get("parameter_names", ())),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class QoIStatePolicy:
    """Pressure-gauge and lifting metadata for QoI consistency checks."""

    gauge_blocks: tuple[Mapping[str, Any], ...] = ()
    lifted_rows: tuple[int, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gauge_blocks", tuple(dict(block) for block in self.gauge_blocks))
        object.__setattr__(self, "lifted_rows", tuple(int(row) for row in self.lifted_rows))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "gauge_blocks": tuple(dict(block) for block in self.gauge_blocks),
            "lifted_rows": tuple(int(row) for row in self.lifted_rows),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class QoIGradientCheck:
    """Finite-difference check result for a QoI derivative."""

    passed: bool
    relative_error: float
    absolute_error: float
    tolerance: float
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class DWRGuardResult:
    """Acceptance gates around a DWR estimate."""

    passed: bool
    reasons: tuple[str, ...]
    safety_factor: float
    certified_bound: float
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "reasons": tuple(self.reasons),
            "safety_factor": float(self.safety_factor),
            "certified_bound": float(self.certified_bound),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DiscreteAdjointResult:
    """Backward discrete adjoint sweep result."""

    adjoints: tuple[np.ndarray, ...]
    rank_history: np.ndarray
    residual_norm_history: np.ndarray
    condition_estimate_history: np.ndarray
    backend: str


@dataclass(frozen=True)
class DWREstimate:
    """Dual-weighted residual estimate and localization data."""

    estimate: float
    absolute_estimate: float
    step_contributions: np.ndarray
    block_contributions: Mapping[str, float]
    effectivity: float | None
    passed: bool
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": float(self.estimate),
            "absolute_estimate": float(self.absolute_estimate),
            "step_contributions": np.asarray(self.step_contributions, dtype=float),
            "block_contributions": dict(self.block_contributions),
            "effectivity": None if self.effectivity is None else float(self.effectivity),
            "passed": bool(self.passed),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DWRCertificationResult:
    """Adjoint solve plus DWR estimator result for one certified QoI."""

    qoi_name: str
    adjoint: DiscreteAdjointResult
    estimate: DWREstimate
    passed: bool
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "qoi_name": self.qoi_name,
            "adjoint_backend": self.adjoint.backend,
            "adjoint_rank_history": np.asarray(self.adjoint.rank_history, dtype=int),
            "adjoint_residual_norm_history": np.asarray(self.adjoint.residual_norm_history, dtype=float),
            "adjoint_condition_estimate_history": np.asarray(
                self.adjoint.condition_estimate_history,
                dtype=float,
            ),
            "estimate": self.estimate.to_dict(),
            "passed": bool(self.passed),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DWRReducedTrajectory:
    """Dense algebraic histories needed for reduced DWR certification."""

    residuals: np.ndarray
    jacobians: np.ndarray
    qoi_gradients: np.ndarray
    previous_state_jacobians: np.ndarray | None = None
    row_weights: np.ndarray | None = None
    time_weights: np.ndarray | None = None
    reference_qoi_error: float | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        residuals = np.asarray(self.residuals, dtype=float)
        jacobians = np.asarray(self.jacobians, dtype=float)
        gradients = np.asarray(self.qoi_gradients, dtype=float)
        if residuals.ndim != 2:
            raise ValueError("DWR trajectory residuals must have shape (n_steps, n_rows).")
        if jacobians.ndim != 3:
            raise ValueError("DWR trajectory jacobians must have shape (n_steps, n_rows, n_cols).")
        if gradients.ndim != 2:
            raise ValueError("DWR trajectory qoi_gradients must have shape (n_steps, n_cols).")
        if jacobians.shape[0] != residuals.shape[0] or gradients.shape[0] != residuals.shape[0]:
            raise ValueError("DWR trajectory arrays must have the same number of time steps.")
        if jacobians.shape[1] != residuals.shape[1]:
            raise ValueError("DWR trajectory jacobian rows must match residual rows.")
        if jacobians.shape[2] != gradients.shape[1]:
            raise ValueError("DWR trajectory jacobian columns must match QoI-gradient size.")
        previous = None if self.previous_state_jacobians is None else np.asarray(self.previous_state_jacobians, dtype=float)
        if previous is not None and previous.shape != jacobians.shape:
            raise ValueError("previous_state_jacobians must have the same shape as jacobians.")
        row_weights = None if self.row_weights is None else np.asarray(self.row_weights, dtype=float)
        if row_weights is not None and row_weights.shape != residuals.shape:
            raise ValueError("row_weights must have the same shape as residuals.")
        time_weights = None if self.time_weights is None else np.asarray(self.time_weights, dtype=float).reshape(-1)
        if time_weights is not None and time_weights.size != residuals.shape[0]:
            raise ValueError("time_weights must have one entry per time step.")
        for label, arr in (
            ("residuals", residuals),
            ("jacobians", jacobians),
            ("qoi_gradients", gradients),
            ("previous_state_jacobians", previous),
            ("row_weights", row_weights),
            ("time_weights", time_weights),
        ):
            if arr is not None and not np.all(np.isfinite(arr)):
                raise ValueError(f"DWR trajectory {label} must contain only finite values.")
        qerr = None if self.reference_qoi_error is None else float(self.reference_qoi_error)
        if qerr is not None and not np.isfinite(qerr):
            raise ValueError("reference_qoi_error must be finite or None.")
        object.__setattr__(self, "residuals", np.ascontiguousarray(residuals, dtype=np.float64))
        object.__setattr__(self, "jacobians", np.ascontiguousarray(jacobians, dtype=np.float64))
        object.__setattr__(self, "qoi_gradients", np.ascontiguousarray(gradients, dtype=np.float64))
        object.__setattr__(self, "previous_state_jacobians", None if previous is None else np.ascontiguousarray(previous, dtype=np.float64))
        object.__setattr__(self, "row_weights", None if row_weights is None else np.ascontiguousarray(row_weights, dtype=np.float64))
        object.__setattr__(self, "time_weights", None if time_weights is None else np.ascontiguousarray(time_weights, dtype=np.float64))
        object.__setattr__(self, "reference_qoi_error", qerr)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_steps(self) -> int:
        return int(self.residuals.shape[0])

    def to_npz_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "residuals": self.residuals,
            "jacobians": self.jacobians,
            "qoi_gradients": self.qoi_gradients,
            "metadata_json": np.asarray(__import__("json").dumps(dict(self.metadata or {}))),
        }
        if self.previous_state_jacobians is not None:
            payload["previous_state_jacobians"] = self.previous_state_jacobians
        if self.row_weights is not None:
            payload["row_weights"] = self.row_weights
        if self.time_weights is not None:
            payload["time_weights"] = self.time_weights
        if self.reference_qoi_error is not None:
            payload["reference_qoi_error"] = np.asarray([float(self.reference_qoi_error)], dtype=float)
        return payload

    def save(self, path: str | Path) -> None:
        save_dwr_reduced_trajectory(self, path)


def _matrix_sequence(values: Sequence[Any], label: str) -> tuple[np.ndarray, ...]:
    out: list[np.ndarray] = []
    for value in values:
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{label} entries must be rank-2 arrays.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{label} entries must contain only finite values.")
        out.append(np.ascontiguousarray(arr, dtype=np.float64))
    return tuple(out)


def _vector_sequence(values: Sequence[Any], label: str) -> tuple[np.ndarray, ...]:
    out: list[np.ndarray] = []
    for value in values:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{label} entries must contain only finite values.")
        out.append(np.ascontiguousarray(arr, dtype=np.float64))
    return tuple(out)


def _normalize_backend(backend: str) -> str:
    name = str(backend).strip().lower()
    if name == "c++":
        name = "cpp"
    if name not in {"python", "cpp"}:
        raise ValueError("backend must be 'python' or 'cpp'.")
    return name


def _solve_transpose_python(jacobian: np.ndarray, rhs: np.ndarray, rcond: float | None) -> tuple[np.ndarray, int, float, float]:
    if jacobian.ndim != 2:
        raise ValueError("jacobian must be a rank-2 array.")
    if rhs.size != jacobian.shape[1]:
        raise ValueError("rhs size must match jacobian column count for J.T z = rhs.")
    A = np.ascontiguousarray(jacobian.T, dtype=np.float64)
    z, _residuals, rank, singular_values = np.linalg.lstsq(
        A,
        rhs,
        rcond=None if rcond is None or rcond <= 0.0 else float(rcond),
    )
    if singular_values.size:
        positive = singular_values[singular_values > 0.0]
        condition = float(singular_values[0] / positive[-1]) if positive.size else float("inf")
    else:
        condition = float("nan")
    return (
        np.ascontiguousarray(z, dtype=np.float64),
        int(rank),
        float(np.linalg.norm(A @ z - rhs)),
        condition,
    )


def solve_transpose_system(
    jacobian: Any,
    rhs: Any,
    *,
    rcond: float | None = None,
    backend: AdjointBackend = "python",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve ``jacobian.T @ z = rhs`` using the requested backend."""

    J = np.ascontiguousarray(np.asarray(jacobian, dtype=float))
    b = np.ascontiguousarray(np.asarray(rhs, dtype=float).reshape(-1))
    backend_name = _normalize_backend(backend)
    if backend_name == "cpp":
        from .cpp_backend.adjoint import module as _adjoint_module

        raw = _adjoint_module().solve_transpose_system(J, b, -1.0 if rcond is None else float(rcond))
        return (
            np.asarray(raw["solution"], dtype=float).reshape(-1),
            {
                "rank": int(raw["rank"]),
                "residual_norm": float(raw["residual_norm"]),
                "condition_estimate": float(raw["condition_estimate"]),
                "backend": "cpp_native_adjoint",
            },
        )
    z, rank, residual_norm, condition = _solve_transpose_python(J, b, rcond)
    return (
        z,
        {
            "rank": rank,
            "residual_norm": residual_norm,
            "condition_estimate": condition,
            "backend": "python_adjoint",
        },
    )


def solve_discrete_adjoint(
    jacobians: Sequence[Any],
    qoi_gradients: Sequence[Any],
    *,
    previous_state_jacobians: Sequence[Any] | None = None,
    rcond: float | None = None,
    backend: AdjointBackend = "python",
) -> DiscreteAdjointResult:
    """Solve the fully discrete backward adjoint recursion.

    ``jacobians[n]`` is ``dR_n/dx_n`` and ``previous_state_jacobians[n]`` is
    ``dR_n/dx_{n-1}``.  The recursion is

    ``J_n.T z_n = dQ/dx_n - M_{n+1}.T z_{n+1}``.
    """

    J = _matrix_sequence(jacobians, "jacobians")
    gradients = _vector_sequence(qoi_gradients, "qoi_gradients")
    if len(J) != len(gradients):
        raise ValueError("jacobians and qoi_gradients must have the same length.")
    M = None if previous_state_jacobians is None else _matrix_sequence(previous_state_jacobians, "previous_state_jacobians")
    if M is not None and len(M) != len(J):
        raise ValueError("previous_state_jacobians must be None or have the same length as jacobians.")

    backend_name = _normalize_backend(backend)
    if backend_name == "cpp":
        from .cpp_backend.adjoint import module as _adjoint_module

        raw = _adjoint_module().solve_discrete_adjoint(
            J,
            gradients,
            None if M is None else M,
            -1.0 if rcond is None else float(rcond),
        )
        return DiscreteAdjointResult(
            adjoints=tuple(np.asarray(z, dtype=float).reshape(-1) for z in raw["adjoints"]),
            rank_history=np.asarray(raw["rank_history"], dtype=float).astype(int),
            residual_norm_history=np.asarray(raw["residual_norm_history"], dtype=float),
            condition_estimate_history=np.asarray(raw["condition_estimate_history"], dtype=float),
            backend=str(raw["backend"]),
        )

    n_steps = len(J)
    adjoints: list[np.ndarray] = [np.zeros(J[i].shape[0], dtype=np.float64) for i in range(n_steps)]
    ranks = np.zeros(n_steps, dtype=int)
    residuals = np.zeros(n_steps, dtype=float)
    conditions = np.zeros(n_steps, dtype=float)
    for i in range(n_steps - 1, -1, -1):
        rhs = gradients[i].copy()
        if i + 1 < n_steps and M is not None:
            if M[i + 1].shape[0] != adjoints[i + 1].size or M[i + 1].shape[1] != rhs.size:
                raise ValueError("previous-state jacobian has incompatible shape for adjoint coupling.")
            rhs -= M[i + 1].T @ adjoints[i + 1]
        z, rank, residual_norm, condition = _solve_transpose_python(J[i], rhs, rcond)
        adjoints[i] = z
        ranks[i] = rank
        residuals[i] = residual_norm
        conditions[i] = condition
    return DiscreteAdjointResult(
        adjoints=tuple(np.ascontiguousarray(z, dtype=np.float64) for z in adjoints),
        rank_history=ranks,
        residual_norm_history=residuals,
        condition_estimate_history=conditions,
        backend="python_adjoint",
    )


def solve_reduced_discrete_adjoint(
    jacobians: Sequence[Any],
    qoi_gradients: Sequence[Any],
    adjoint_basis: Any,
    *,
    previous_state_jacobians: Sequence[Any] | None = None,
    rcond: float | None = None,
    backend: AdjointBackend = "python",
) -> DiscreteAdjointResult:
    """Solve a reduced adjoint and lift it back to the full residual space.

    The adjoint is approximated as ``z_n = W_z a_n``.  Each backward step solves
    ``J_n.T W_z a_n = rhs`` in the least-squares/SVD sense, so the function is
    compatible with full-row, reduced, and sampled algebraic tangent targets.
    """

    W = np.ascontiguousarray(np.asarray(adjoint_basis, dtype=float))
    if W.ndim != 2 or W.shape[1] == 0:
        raise ValueError("adjoint_basis must be a rank-2 matrix with at least one column.")
    J = _matrix_sequence(jacobians, "jacobians")
    gradients = _vector_sequence(qoi_gradients, "qoi_gradients")
    if len(J) != len(gradients):
        raise ValueError("jacobians and qoi_gradients must have the same length.")
    if any(mat.shape[0] != W.shape[0] for mat in J):
        raise ValueError("adjoint_basis rows must match jacobian residual rows.")
    M = None if previous_state_jacobians is None else _matrix_sequence(previous_state_jacobians, "previous_state_jacobians")
    if M is not None and len(M) != len(J):
        raise ValueError("previous_state_jacobians must be None or have the same length as jacobians.")

    backend_name = _normalize_backend(backend)
    n_steps = len(J)
    adjoints: list[np.ndarray] = [np.zeros(W.shape[0], dtype=np.float64) for _ in range(n_steps)]
    ranks = np.zeros(n_steps, dtype=int)
    residuals = np.zeros(n_steps, dtype=float)
    conditions = np.zeros(n_steps, dtype=float)
    for i in range(n_steps - 1, -1, -1):
        rhs = gradients[i].copy()
        if i + 1 < n_steps and M is not None:
            rhs -= M[i + 1].T @ adjoints[i + 1]
        reduced_jacobian = np.ascontiguousarray(W.T @ J[i], dtype=np.float64)
        coeffs, meta = solve_transpose_system(
            reduced_jacobian,
            rhs,
            rcond=rcond,
            backend=backend_name,  # type: ignore[arg-type]
        )
        adjoints[i] = np.ascontiguousarray(W @ coeffs, dtype=np.float64)
        ranks[i] = int(meta["rank"])
        residuals[i] = float(meta["residual_norm"])
        conditions[i] = float(meta["condition_estimate"])
    return DiscreteAdjointResult(
        adjoints=tuple(adjoints),
        rank_history=ranks,
        residual_norm_history=residuals,
        condition_estimate_history=conditions,
        backend=f"{backend_name}_reduced_adjoint",
    )


def finite_difference_gradient(
    functional: Callable[[np.ndarray], float],
    state: Any,
    *,
    step: float = 1.0e-6,
) -> np.ndarray:
    """Central finite-difference gradient for QoI derivative tests."""

    x = np.asarray(state, dtype=float).reshape(-1)
    if not np.all(np.isfinite(x)):
        raise ValueError("state must contain only finite values.")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive.")
    grad = np.zeros_like(x, dtype=np.float64)
    for i in range(x.size):
        perturb = np.zeros_like(x)
        perturb[i] = float(step)
        grad[i] = (float(functional(x + perturb)) - float(functional(x - perturb))) / (2.0 * float(step))
    return np.ascontiguousarray(grad, dtype=np.float64)


def check_qoi_gradient(
    functional: Callable[[np.ndarray], float],
    state: Any,
    analytic_gradient: Any,
    *,
    step: float = 1.0e-6,
    tolerance: float = 1.0e-6,
    metadata: Mapping[str, Any] | None = None,
) -> QoIGradientCheck:
    """Compare an assembled/generated QoI gradient against finite differences."""

    analytic = np.asarray(analytic_gradient, dtype=float).reshape(-1)
    fd = finite_difference_gradient(functional, state, step=step)
    if analytic.size != fd.size:
        raise ValueError("analytic_gradient size must match state size.")
    diff = analytic - fd
    abs_error = float(np.linalg.norm(diff))
    denom = max(float(np.linalg.norm(fd)), 1.0e-14)
    rel_error = abs_error / denom
    return QoIGradientCheck(
        passed=bool(rel_error <= float(tolerance) or abs_error <= float(tolerance)),
        relative_error=float(rel_error),
        absolute_error=abs_error,
        tolerance=float(tolerance),
        metadata={"step": float(step), **dict(metadata or {})},
    )


def linearize_qoi_functional(
    qoi_form: Any,
    coefficients: Any,
    directions: Any,
    *,
    strict: bool = True,
) -> Any:
    """Return the Gateaux derivative of a scalar QoI functional.

    This is the MOR-facing wrapper around the repo UFL autodiff layer.  The
    returned form is linear in the supplied trial directions and can be assembled
    as a global QoI-gradient vector with ``assemble_qoi_gradient``.
    """

    from pycutfem.ufl.autodiff import gateaux_derivative

    return gateaux_derivative(qoi_form, coefficients, directions, strict=strict)


def evaluate_qoi_functional(
    qoi_form: Any,
    *,
    dof_handler: Any,
    bcs: Sequence[Any] | None = None,
    backend: str = "cpp",
    quad_order: int | None = None,
    name: str = "qoi",
) -> float:
    """Assemble a scalar UFL QoI functional with the requested backend."""

    from pycutfem.ufl.forms import Equation, assemble_form

    integrals = tuple(getattr(qoi_form, "integrals", (qoi_form,)))
    hooks = {integral.integrand: {"name": str(name)} for integral in integrals}
    result = assemble_form(
        Equation(None, qoi_form),
        dof_handler=dof_handler,
        bcs=list(bcs or ()),
        assembler_hooks=hooks,
        backend=backend,
        quad_order=quad_order,
    )
    return float(np.asarray(result[str(name)], dtype=float).reshape(-1).sum())


def assemble_qoi_gradient(
    qoi_form: Any,
    coefficients: Any,
    directions: Any,
    *,
    dof_handler: Any,
    bcs: Sequence[Any] | None = None,
    backend: str = "cpp",
    quad_order: int | None = None,
    active_rows: Any | None = None,
    strict: bool = True,
) -> np.ndarray:
    """Assemble the full-space gradient of a scalar UFL QoI functional."""

    from pycutfem.ufl.forms import Equation, assemble_form

    derivative_form = linearize_qoi_functional(qoi_form, coefficients, directions, strict=strict)
    _matrix, vector = assemble_form(
        Equation(None, derivative_form),
        dof_handler=dof_handler,
        bcs=list(bcs or ()),
        backend=backend,
        quad_order=quad_order,
    )
    grad = np.asarray(vector, dtype=float).reshape(-1)
    if active_rows is not None:
        rows = np.asarray(active_rows, dtype=np.int64).reshape(-1)
        if np.any(rows < 0) or (rows.size and int(np.max(rows)) >= grad.size):
            raise ValueError("active_rows contains entries outside the assembled QoI gradient.")
        return np.ascontiguousarray(grad[rows], dtype=np.float64)
    return np.ascontiguousarray(grad, dtype=np.float64)


def reduced_qoi_gradient_from_full(
    full_gradient: Any,
    trial_basis: Any,
    *,
    row_weights: Any | None = None,
) -> np.ndarray:
    """Project a full-space QoI gradient into reduced coordinates."""

    grad = np.asarray(full_gradient, dtype=float).reshape(-1)
    basis = np.asarray(trial_basis, dtype=float)
    if basis.ndim != 2 or basis.shape[0] != grad.size:
        raise ValueError("trial_basis must be rank-2 with one row per QoI-gradient entry.")
    if row_weights is not None:
        weights = np.asarray(row_weights, dtype=float).reshape(-1)
        if weights.size != grad.size:
            raise ValueError("row_weights must match full_gradient size.")
        grad = weights * grad
    return np.ascontiguousarray(basis.T @ grad, dtype=np.float64)


def _coerce_block_rows(block: Any) -> tuple[str, np.ndarray]:
    if isinstance(block, Mapping):
        return str(block.get("name", "")), np.asarray(block["rows"], dtype=np.int64).reshape(-1)
    if isinstance(block, (tuple, list)) and len(block) == 2:
        return str(block[1]), np.asarray(block[0], dtype=np.int64).reshape(-1)
    return "", np.asarray(block, dtype=np.int64).reshape(-1)


def dual_weighted_residual_estimate(
    residuals: Sequence[Any],
    adjoints: Sequence[Any],
    *,
    row_weights: Sequence[Any] | None = None,
    time_weights: Sequence[float] | None = None,
    row_blocks: Sequence[Any] | None = None,
    reference_qoi_error: float | None = None,
    effectivity_bounds: tuple[float, float] | None = None,
    sign: float = -1.0,
    metadata: Mapping[str, Any] | None = None,
) -> DWREstimate:
    """Compute a DWR estimate ``sign * sum_n z_n.T R_n``.

    With residual convention ``R(x)=0`` and an approximate state ``x_r``, the
    default sign gives the linear error identity
    ``Q(x_exact)-Q(x_r) = - z.T R(x_r)``.
    """

    R = _vector_sequence(residuals, "residuals")
    Z = _vector_sequence(adjoints, "adjoints")
    if len(R) != len(Z):
        raise ValueError("residuals and adjoints must have the same length.")
    weights = None if row_weights is None else _vector_sequence(row_weights, "row_weights")
    if weights is not None and len(weights) != len(R):
        raise ValueError("row_weights must be None or have one entry per residual.")
    if time_weights is None:
        tw = np.ones(len(R), dtype=np.float64)
    else:
        tw = np.asarray(time_weights, dtype=float).reshape(-1)
        if tw.size != len(R):
            raise ValueError("time_weights must have one entry per residual.")
        if not np.all(np.isfinite(tw)):
            raise ValueError("time_weights must contain only finite values.")

    step_contrib = np.zeros(len(R), dtype=np.float64)
    block_contrib: dict[str, float] = {}
    for i, (r, z) in enumerate(zip(R, Z, strict=True)):
        if r.size != z.size:
            raise ValueError("each residual and adjoint vector must have the same size.")
        wr = r if weights is None else np.asarray(weights[i], dtype=float).reshape(-1) * r
        if wr.size != r.size:
            raise ValueError("each row_weights vector must match the residual size.")
        step_contrib[i] = float(sign) * float(tw[i]) * float(np.dot(z, wr))
        if row_blocks is not None:
            for block_idx, block in enumerate(row_blocks):
                name, rows = _coerce_block_rows(block)
                block_name = name or f"block_{block_idx}"
                rows = rows[(rows >= 0) & (rows < r.size)]
                if rows.size == 0:
                    block_contrib.setdefault(block_name, 0.0)
                    continue
                block_contrib[block_name] = block_contrib.get(block_name, 0.0) + float(sign) * float(tw[i]) * float(
                    np.dot(z[rows], wr[rows])
                )

    estimate = float(np.sum(step_contrib))
    effectivity: float | None = None
    passed = True
    if reference_qoi_error is not None:
        err = abs(float(reference_qoi_error))
        effectivity = float(abs(estimate) / err) if err > 0.0 else (0.0 if abs(estimate) == 0.0 else float("inf"))
        if effectivity_bounds is not None:
            lo, hi = float(effectivity_bounds[0]), float(effectivity_bounds[1])
            passed = bool(lo <= effectivity <= hi)
    return DWREstimate(
        estimate=estimate,
        absolute_estimate=abs(estimate),
        step_contributions=np.ascontiguousarray(step_contrib, dtype=np.float64),
        block_contributions=block_contrib,
        effectivity=effectivity,
        passed=passed,
        metadata=dict(metadata or {}),
    )


def dominant_dwr_contributions(
    estimate: DWREstimate,
    *,
    max_entries: int = 5,
) -> dict[str, Any]:
    """Return dominant time and block DWR contributions by absolute value."""

    steps = np.asarray(estimate.step_contributions, dtype=float).reshape(-1)
    k = max(0, min(int(max_entries), steps.size))
    if k:
        order = np.argsort(np.abs(steps))[::-1][:k]
        dominant_steps = tuple(
            {"index": int(idx), "value": float(steps[idx]), "absolute": float(abs(steps[idx]))}
            for idx in order
        )
    else:
        dominant_steps = ()
    block_items = sorted(
        (
            {"name": str(name), "value": float(value), "absolute": float(abs(value))}
            for name, value in estimate.block_contributions.items()
        ),
        key=lambda item: item["absolute"],
        reverse=True,
    )[: int(max_entries)]
    return {
        "steps": dominant_steps,
        "blocks": tuple(block_items),
    }


def dwr_certification_guard(
    estimate: DWREstimate,
    *,
    branch_certificate: Mapping[str, Any] | None = None,
    norm_equivalence_certificate: Any | Mapping[str, Any] | None = None,
    gauge_certificate: Mapping[str, Any] | None = None,
    require_branch: bool = True,
    require_norm_equivalence: bool = False,
    require_gauge: bool = False,
    safety_factor: float = 1.0,
    max_effectivity: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DWRGuardResult:
    """Apply branch, GNAT norm-equivalence, gauge, and effectivity guards."""

    reasons: list[str] = []
    if not estimate.passed:
        reasons.append("dwr_effectivity_gate_failed")
    if require_branch:
        ok = bool((branch_certificate or {}).get("passed", False))
        if not ok:
            reasons.append("branch_certificate_failed")
    if require_norm_equivalence:
        norm_passed = False
        if norm_equivalence_certificate is not None:
            if isinstance(norm_equivalence_certificate, Mapping):
                norm_passed = bool(norm_equivalence_certificate.get("passed", False))
            else:
                norm_passed = bool(getattr(norm_equivalence_certificate, "passed", False))
        if not norm_passed:
            reasons.append("norm_equivalence_failed")
    if require_gauge:
        ok = bool((gauge_certificate or {}).get("passed", False))
        if not ok:
            reasons.append("pressure_gauge_failed")
    if max_effectivity is not None and estimate.effectivity is not None and float(estimate.effectivity) > float(max_effectivity):
        reasons.append("dwr_overconservative")
    factor = float(safety_factor)
    if not np.isfinite(factor) or factor < 1.0:
        raise ValueError("safety_factor must be finite and at least one.")
    return DWRGuardResult(
        passed=not reasons,
        reasons=tuple(reasons),
        safety_factor=factor,
        certified_bound=factor * float(estimate.absolute_estimate),
        metadata={
            "estimate_passed": bool(estimate.passed),
            "effectivity": estimate.effectivity,
            **dict(metadata or {}),
        },
    )


def _artifact_adjoint_spec(artifact: Any | None) -> Any | None:
    if artifact is None:
        return None
    return getattr(artifact, "adjoint_dwr", None)


def _trajectory_mapping(trajectory: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trajectory, (str, Path)):
        with np.load(Path(trajectory), allow_pickle=False) as data:
            return {str(key): np.asarray(data[key]) for key in data.files}
    return dict(trajectory)


def save_dwr_reduced_trajectory(trajectory: DWRReducedTrajectory, path: str | Path) -> None:
    """Persist dense reduced DWR trajectory histories as an ``.npz`` artifact."""

    np.savez_compressed(Path(path), **trajectory.to_npz_payload())


def load_dwr_reduced_trajectory(path: str | Path) -> DWRReducedTrajectory:
    """Load a dense reduced DWR trajectory saved by ``save_dwr_reduced_trajectory``."""

    import json

    payload = _trajectory_mapping(path)
    metadata: dict[str, Any] = {}
    if "metadata_json" in payload:
        raw = np.asarray(payload["metadata_json"])
        metadata = json.loads(str(raw.tolist() if hasattr(raw, "tolist") else raw))
    qerr = None
    if "reference_qoi_error" in payload:
        qerr = float(np.asarray(payload["reference_qoi_error"], dtype=float).reshape(-1)[0])
    return DWRReducedTrajectory(
        residuals=payload["residuals"],
        jacobians=payload["jacobians"],
        qoi_gradients=payload["qoi_gradients"],
        previous_state_jacobians=payload.get("previous_state_jacobians"),
        row_weights=payload.get("row_weights"),
        time_weights=payload.get("time_weights"),
        reference_qoi_error=qerr,
        metadata=metadata,
    )


def _stacked_vector_sequence(payload: Mapping[str, Any], key: str, *, required: bool) -> tuple[np.ndarray, ...] | None:
    if key not in payload:
        if required:
            raise ValueError(f"trajectory payload is missing required array {key!r}.")
        return None
    arr = np.asarray(payload[key], dtype=float)
    if arr.ndim == 1:
        return (np.ascontiguousarray(arr, dtype=np.float64),)
    if arr.ndim != 2:
        raise ValueError(f"trajectory array {key!r} must be rank-1 or rank-2 for vector histories.")
    return tuple(np.ascontiguousarray(arr[i, :], dtype=np.float64) for i in range(arr.shape[0]))


def _stacked_matrix_sequence(payload: Mapping[str, Any], key: str, *, required: bool) -> tuple[np.ndarray, ...] | None:
    if key not in payload:
        if required:
            raise ValueError(f"trajectory payload is missing required array {key!r}.")
        return None
    arr = np.asarray(payload[key], dtype=float)
    if arr.ndim == 2:
        return (np.ascontiguousarray(arr, dtype=np.float64),)
    if arr.ndim != 3:
        raise ValueError(f"trajectory array {key!r} must be rank-2 or rank-3 for matrix histories.")
    return tuple(np.ascontiguousarray(arr[i, :, :], dtype=np.float64) for i in range(arr.shape[0]))


def certify_dual_weighted_residual(
    residuals: Sequence[Any],
    jacobians: Sequence[Any],
    qoi_gradients: Sequence[Any],
    *,
    previous_state_jacobians: Sequence[Any] | None = None,
    artifact: Any | None = None,
    adjoint_dwr: Any | None = None,
    row_weights: Sequence[Any] | None = None,
    time_weights: Sequence[float] | None = None,
    row_blocks: Sequence[Any] | None = None,
    reference_qoi_error: float | None = None,
    effectivity_bounds: tuple[float, float] | None = None,
    sign: float | None = None,
    backend: AdjointBackend | None = None,
    rcond: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DWRCertificationResult:
    """Solve the discrete adjoint and evaluate a DWR certificate.

    The routine is intentionally algebraic: generated UFL kernels or native
    trajectory replay supply ``residuals``, ``jacobians`` and ``qoi_gradients``.
    If a native reduced artifact with ``adjoint_dwr`` metadata is supplied, its
    solver and estimator options become defaults for this certification run.
    """

    spec = adjoint_dwr if adjoint_dwr is not None else _artifact_adjoint_spec(artifact)
    solver_options = dict(getattr(spec, "solver_options", {}) or {})
    estimator_options = dict(getattr(spec, "estimator_options", {}) or {})
    qoi_name = str(getattr(spec, "qoi_name", "") or "qoi")

    backend_name = backend if backend is not None else solver_options.get("backend", "python")
    rcond_value = rcond if rcond is not None else solver_options.get("rcond")
    bounds_value = effectivity_bounds
    if bounds_value is None and "effectivity_bounds" in estimator_options:
        raw_bounds = estimator_options["effectivity_bounds"]
        bounds_value = (float(raw_bounds[0]), float(raw_bounds[1]))
    sign_value = float(sign if sign is not None else estimator_options.get("sign", -1.0))

    adjoint = solve_discrete_adjoint(
        jacobians,
        qoi_gradients,
        previous_state_jacobians=previous_state_jacobians,
        rcond=None if rcond_value is None else float(rcond_value),
        backend=backend_name,  # type: ignore[arg-type]
    )
    estimate = dual_weighted_residual_estimate(
        residuals,
        adjoint.adjoints,
        row_weights=row_weights,
        time_weights=time_weights,
        row_blocks=row_blocks,
        reference_qoi_error=reference_qoi_error,
        effectivity_bounds=bounds_value,
        sign=sign_value,
        metadata={
            "qoi_name": qoi_name,
            "adjoint_backend": adjoint.backend,
            **dict(metadata or {}),
        },
    )

    rank_ok = bool(np.all(np.asarray(adjoint.rank_history, dtype=int) > 0))
    residuals_ok = bool(np.all(np.isfinite(np.asarray(adjoint.residual_norm_history, dtype=float))))
    passed = bool(estimate.passed and rank_ok and residuals_ok)
    result_metadata = {
        "artifact_problem_id": None if artifact is None else getattr(artifact, "problem_id", None),
        "solver_options": solver_options,
        "estimator_options": estimator_options,
        "rank_ok": rank_ok,
        "adjoint_residuals_finite": residuals_ok,
        **dict(metadata or {}),
    }
    return DWRCertificationResult(
        qoi_name=qoi_name,
        adjoint=adjoint,
        estimate=estimate,
        passed=passed,
        metadata=result_metadata,
    )


def certify_dual_weighted_residual_from_artifact_trajectory(
    artifact: Any | str | Path,
    trajectory: str | Path | Mapping[str, Any] | DWRReducedTrajectory,
    *,
    reference_qoi_error: float | None = None,
    effectivity_bounds: tuple[float, float] | None = None,
    sign: float | None = None,
    backend: AdjointBackend | None = None,
    rcond: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DWRCertificationResult:
    """Certify a saved reduced trajectory using artifact adjoint/DWR metadata.

    ``trajectory`` may be a mapping or an ``.npz`` with dense stacked arrays:
    ``residuals`` with shape ``(n_steps, n_rows)``, ``jacobians`` with shape
    ``(n_steps, n_rows, n_cols)``, ``qoi_gradients`` with shape
    ``(n_steps, n_cols)``, and optional ``previous_state_jacobians``,
    ``row_weights`` and ``time_weights``.  This keeps trajectory replay
    problem-generic; generated UFL/native drivers only need to persist these
    algebraic histories.
    """

    artifact_obj = artifact
    if isinstance(artifact, (str, Path)):
        from .artifacts import load_native_reduced_artifact

        artifact_obj = load_native_reduced_artifact(Path(artifact))
    if isinstance(trajectory, DWRReducedTrajectory):
        traj = trajectory
        payload_metadata = dict(traj.metadata or {})
        residuals = tuple(np.asarray(traj.residuals[i, :], dtype=float) for i in range(traj.n_steps))
        jacobians = tuple(np.asarray(traj.jacobians[i, :, :], dtype=float) for i in range(traj.n_steps))
        qoi_gradients = tuple(np.asarray(traj.qoi_gradients[i, :], dtype=float) for i in range(traj.n_steps))
        previous = (
            None
            if traj.previous_state_jacobians is None
            else tuple(np.asarray(traj.previous_state_jacobians[i, :, :], dtype=float) for i in range(traj.n_steps))
        )
        row_weights = (
            None
            if traj.row_weights is None
            else tuple(np.asarray(traj.row_weights[i, :], dtype=float) for i in range(traj.n_steps))
        )
        time_weights = traj.time_weights
        qoi_error = reference_qoi_error if reference_qoi_error is not None else traj.reference_qoi_error
    else:
        payload = _trajectory_mapping(trajectory)
        payload_metadata = {}
        residuals = _stacked_vector_sequence(payload, "residuals", required=True)
        jacobians = _stacked_matrix_sequence(payload, "jacobians", required=True)
        qoi_gradients = _stacked_vector_sequence(payload, "qoi_gradients", required=True)
        previous = _stacked_matrix_sequence(payload, "previous_state_jacobians", required=False)
        row_weights = _stacked_vector_sequence(payload, "row_weights", required=False)
        time_weights = None
        if "time_weights" in payload:
            time_weights = np.asarray(payload["time_weights"], dtype=float).reshape(-1)
        qoi_error = reference_qoi_error
        if qoi_error is None and "reference_qoi_error" in payload:
            qoi_error = float(np.asarray(payload["reference_qoi_error"], dtype=float).reshape(-1)[0])
    return certify_dual_weighted_residual(
        residuals or (),
        jacobians or (),
        qoi_gradients or (),
        previous_state_jacobians=previous,
        artifact=artifact_obj,
        row_weights=row_weights,
        time_weights=time_weights,
        reference_qoi_error=qoi_error,
        effectivity_bounds=effectivity_bounds,
        sign=sign,
        backend=backend,
        rcond=rcond,
        metadata={
            "trajectory_source": str(trajectory) if isinstance(trajectory, (str, Path)) else "mapping",
            **payload_metadata,
            **dict(metadata or {}),
        },
    )


__all__ = [
    "AdjointBackend",
    "DWRReducedTrajectory",
    "DWRCertificationResult",
    "DWREstimate",
    "DWRGuardResult",
    "DiscreteAdjointResult",
    "QoIGradientCheck",
    "QoIFunctionalSpec",
    "QoIKernelSpec",
    "QoIStatePolicy",
    "TransientResidualDependencySpec",
    "assemble_qoi_gradient",
    "certify_dual_weighted_residual",
    "certify_dual_weighted_residual_from_artifact_trajectory",
    "check_qoi_gradient",
    "dominant_dwr_contributions",
    "dual_weighted_residual_estimate",
    "dwr_certification_guard",
    "evaluate_qoi_functional",
    "finite_difference_gradient",
    "linearize_qoi_functional",
    "load_dwr_reduced_trajectory",
    "reduced_qoi_gradient_from_full",
    "save_dwr_reduced_trajectory",
    "solve_discrete_adjoint",
    "solve_reduced_discrete_adjoint",
    "solve_transpose_system",
]
