"""Generic mixed-field MOR reduction helpers.

The routines here are intentionally problem-generic.  They cover the algebraic
offline pieces that mixed-field ROMs need before entering a native online solve:
Dirichlet lifting, homogeneous snapshots, coupled-field lift enrichment
(supremizer-style enrichment is one use case), mixed trial-basis assembly, and
non-affine DEIM/QDEIM reduced terms.

Offline basis construction is not part of the online nonlinear loop.  The
implementations below avoid Python-level inner loops on large arrays and rely on
native BLAS/LAPACK/SuperLU or the repo's native Eigen SparseQR wrapper where
requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .decomposition import (
    CollateralBasis,
    InterpolationRule,
    build_deim_interpolation_rule,
    build_qdeim_interpolation_rule,
    fit_collateral_basis,
)
from .pod import fit_pod


def _as_matrix(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 1-D or 2-D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _as_vector(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _as_positive_weights(value: Any, n_rows: int, label: str) -> np.ndarray:
    weights = np.asarray(value, dtype=float).reshape(-1)
    if weights.size != int(n_rows):
        raise ValueError(f"{label} must have one entry per residual row.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError(f"{label} must contain finite positive values.")
    return np.ascontiguousarray(weights, dtype=np.float64)


def field_dof_indices(dof_handler: Any, fields: Sequence[str]) -> np.ndarray:
    """Return the concatenated global DOF ids for a field list."""

    ids: list[int] = []
    for field in fields:
        ids.extend(int(v) for v in np.asarray(dof_handler.get_field_slice(str(field)), dtype=int).reshape(-1))
    out = np.asarray(ids, dtype=np.int64)
    if np.unique(out).size != out.size:
        raise ValueError("field DOF ids must be unique across the requested fields.")
    return out


def build_dirichlet_lifting_vector(
    dof_handler: Any,
    bcs: Any,
    *,
    total_dofs: int | None = None,
) -> np.ndarray:
    """Build a full-state lifting vector from Dirichlet boundary data.

    Only constrained DOFs are populated.  All unconstrained entries are zero, so
    subtracting this vector from a snapshot produces a homogeneous-BC snapshot
    suitable for POD.
    """

    n = int(dof_handler.total_dofs if total_dofs is None else total_dofs)
    if n <= 0:
        raise ValueError("total_dofs must be positive.")
    lifting = np.zeros(n, dtype=np.float64)
    for dof, value in dof_handler.get_dirichlet_data(bcs).items():
        idx = int(dof)
        if idx < 0 or idx >= n:
            raise ValueError(f"Dirichlet DOF {idx} is outside the full state size {n}.")
        lifting[idx] = float(value)
    return lifting


def remove_lifting_from_snapshots(snapshots: Any, lifting: Any) -> np.ndarray:
    """Subtract a constant or snapshot-dependent lifting matrix."""

    matrix = _as_matrix(snapshots, "snapshots")
    lift = np.asarray(lifting, dtype=float)
    if lift.ndim == 1:
        lift = lift[:, None]
    if lift.ndim != 2:
        raise ValueError("lifting must be a 1-D vector or 2-D matrix.")
    if lift.shape[0] != matrix.shape[0]:
        raise ValueError("lifting row count must match snapshots.")
    if lift.shape[1] not in {1, matrix.shape[1]}:
        raise ValueError("lifting must have one column or one column per snapshot.")
    return np.ascontiguousarray(matrix - lift, dtype=np.float64)


def restore_lifting_to_snapshots(homogeneous_snapshots: Any, lifting: Any) -> np.ndarray:
    """Add a constant or snapshot-dependent lifting matrix back to snapshots."""

    matrix = _as_matrix(homogeneous_snapshots, "homogeneous snapshots")
    lift = np.asarray(lifting, dtype=float)
    if lift.ndim == 1:
        lift = lift[:, None]
    if lift.ndim != 2:
        raise ValueError("lifting must be a 1-D vector or 2-D matrix.")
    if lift.shape[0] != matrix.shape[0]:
        raise ValueError("lifting row count must match snapshots.")
    if lift.shape[1] not in {1, matrix.shape[1]}:
        raise ValueError("lifting must have one column or one column per snapshot.")
    return np.ascontiguousarray(matrix + lift, dtype=np.float64)


def _is_sparse(value: Any) -> bool:
    try:
        import scipy.sparse as sp
    except Exception:  # pragma: no cover - scipy is normally present in this repo
        return False
    return bool(sp.issparse(value))


def _operator_matmul(operator: Any, dense: np.ndarray) -> np.ndarray:
    out = operator @ dense
    return np.ascontiguousarray(np.asarray(out, dtype=float), dtype=np.float64)


def _solve_matrix_rhs(
    operator: Any,
    rhs: np.ndarray,
    regularization: float = 0.0,
    *,
    solver_backend: str = "auto",
    solver: Any | None = None,
) -> np.ndarray:
    """Solve a square operator against a dense multi-RHS matrix.

    Sparse solves use one factorization for all right-hand sides when possible.
    ``solver_backend='eigen_sparseqr'`` dispatches each RHS through the native
    Eigen SparseQR handle; this is useful for rank-deficient or least-squares
    lift problems.
    """

    B = _as_matrix(rhs, "right-hand side")
    if solver is not None:
        if hasattr(solver, "solve") and callable(getattr(solver, "solve")):
            raw = solver.solve(B)
        else:
            raw = solver(operator, B)
        out = _as_matrix(raw, "custom solver result")
        if out.shape != B.shape:
            raise ValueError("custom solver returned an incompatible shape.")
        return out

    backend = str(solver_backend or "auto").strip().lower().replace("-", "_")

    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except Exception:  # pragma: no cover - scipy is normally present in this repo
        sp = None
        spla = None

    if sp is not None and sp.issparse(operator):
        mat = operator.tocsc()
        if regularization:
            mat = mat + float(regularization) * sp.eye(mat.shape[0], format="csc")
        if backend in {"eigen_sparseqr", "eigen_qr", "native_sparseqr"}:
            from pycutfem.linalg import EigenSparseQRSubsolver

            qr = EigenSparseQRSubsolver(mat)
            return np.ascontiguousarray(
                np.column_stack([qr.solve(B[:, j]) for j in range(B.shape[1])]),
                dtype=np.float64,
            )
        if backend in {"auto", "direct", "lu", "superlu"}:
            try:
                lu = spla.splu(mat)
                return np.ascontiguousarray(np.asarray(lu.solve(B), dtype=float), dtype=np.float64)
            except Exception:
                if backend != "auto":
                    raise
        if backend in {"auto", "lsqr", "least_squares"}:
            return np.ascontiguousarray(
                np.column_stack([spla.lsqr(mat, B[:, j])[0] for j in range(B.shape[1])]),
                dtype=np.float64,
            )
        raise ValueError(f"Unsupported sparse lift solver backend {solver_backend!r}.")

    mat = np.asarray(operator, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("operator must be a square matrix.")
    if regularization:
        mat = mat + float(regularization) * np.eye(mat.shape[0], dtype=float)
    if backend in {"auto", "direct", "lu"}:
        try:
            return np.ascontiguousarray(np.linalg.solve(mat, B), dtype=np.float64)
        except np.linalg.LinAlgError:
            if backend != "auto":
                raise
    if backend in {"auto", "svd", "least_squares", "lstsq"}:
        return np.ascontiguousarray(np.linalg.lstsq(mat, B, rcond=None)[0], dtype=np.float64)
    raise ValueError(f"Unsupported dense lift solver backend {solver_backend!r}.")


def solve_coupled_lift_snapshots(
    primary_operator: Any,
    coupling_operator: Any,
    coupled_basis: Any,
    *,
    regularization: float = 0.0,
    solver_backend: str = "auto",
    solver: Any | None = None,
) -> np.ndarray:
    """Compute lift snapshots ``A_primary X = C.T @ Phi_coupled``.

    This is the generic algebra behind pressure supremizers, Lagrange-multiplier
    lifts, constraint lifts, and other inf-sup/coupled-field enrichments.
    ``coupling_operator`` uses coupled-field rows and primary-field columns.
    """

    try:
        import scipy.sparse as sp
    except Exception:  # pragma: no cover - scipy is normally present in this repo
        sp = None

    if sp is not None and sp.issparse(coupling_operator):
        C = coupling_operator.tocsr()
        if not np.all(np.isfinite(C.data)):
            raise ValueError("coupling operator must contain only finite values.")
    else:
        C = _as_matrix(coupling_operator, "coupling operator")
    Phi = _as_matrix(coupled_basis, "coupled basis")
    if C.shape[0] != Phi.shape[0]:
        raise ValueError("coupled_basis rows must match coupling_operator rows.")
    rhs = _operator_matmul(C.T, Phi)
    out = _solve_matrix_rhs(
        primary_operator,
        rhs,
        regularization=regularization,
        solver_backend=solver_backend,
        solver=solver,
    )
    if out.shape != rhs.shape:
        raise ValueError("lift solver returned an incompatible shape.")
    if not np.all(np.isfinite(out)):
        raise ValueError("lift snapshots must contain only finite values.")
    return np.ascontiguousarray(out, dtype=np.float64)


def compute_supremizer_snapshots(
    velocity_operator: Any,
    divergence_coupling: Any,
    pressure_basis: Any,
    *,
    regularization: float = 0.0,
    solver_backend: str = "auto",
    solver: Any | None = None,
) -> np.ndarray:
    """Compatibility wrapper for incompressible pressure supremizers."""

    return solve_coupled_lift_snapshots(
        velocity_operator,
        divergence_coupling,
        pressure_basis,
        regularization=regularization,
        solver_backend=solver_backend,
        solver=solver,
    )


def orthonormalize_columns(matrix: Any, *, inner_product: Any | None = None, tol: float = 1.0e-12) -> np.ndarray:
    """Orthonormalize columns using native factorization kernels.

    With no inner product this uses SVD (`U` columns), which is robust for
    rank-deficient snapshot blocks.  With a supplied SPD inner product ``M`` it
    diagonalizes the small Gram matrix ``A.T @ M @ A`` and forms
    ``A W Lambda^{-1/2}``, avoiding Python-level Gram-Schmidt loops.
    """

    A = _as_matrix(matrix, "basis matrix")
    if A.shape[1] == 0:
        return np.zeros((A.shape[0], 0), dtype=np.float64)

    if inner_product is None:
        U, singular_values, _ = np.linalg.svd(A, full_matrices=False)
        scale = float(singular_values[0]) if singular_values.size else 1.0
        keep = singular_values > float(tol) * max(scale, 1.0)
        return np.ascontiguousarray(U[:, keep], dtype=np.float64)

    if _is_sparse(inner_product):
        if inner_product.shape != (A.shape[0], A.shape[0]):
            raise ValueError("inner_product shape must match basis rows.")
        MA = _operator_matmul(inner_product, A)
    else:
        M = _as_matrix(inner_product, "inner product")
        if M.shape != (A.shape[0], A.shape[0]):
            raise ValueError("inner_product shape must match basis rows.")
        MA = M @ A
    gram = np.ascontiguousarray(A.T @ MA, dtype=np.float64)
    gram = 0.5 * (gram + gram.T)
    eigvals, eigvecs = np.linalg.eigh(gram)
    if eigvals.size == 0:
        return np.zeros((A.shape[0], 0), dtype=np.float64)
    scale = float(np.max(np.abs(eigvals))) if eigvals.size else 1.0
    keep = eigvals > float(tol) * max(scale, 1.0)
    if not np.any(keep):
        return np.zeros((A.shape[0], 0), dtype=np.float64)
    transform = eigvecs[:, keep] / np.sqrt(eigvals[keep])[None, :]
    return np.ascontiguousarray(A @ transform, dtype=np.float64)


@dataclass(frozen=True)
class LiftEnrichment:
    """Primary-field basis enriched with coupled-field lift modes."""

    primary_basis: np.ndarray
    coupled_basis: np.ndarray
    lift_snapshots: np.ndarray
    lift_basis: np.ndarray
    enriched_primary_basis: np.ndarray
    metadata: Mapping[str, Any]


def fit_lift_enriched_basis(
    primary_basis: Any,
    coupled_basis: Any,
    lift_snapshots: Any,
    *,
    n_lift_modes: int | None = None,
    lift_energy: float | None = None,
    inner_product: Any | None = None,
) -> LiftEnrichment:
    """Fit coupled-field lift modes and append them to a primary basis."""

    primary = _as_matrix(primary_basis, "primary basis")
    coupled = _as_matrix(coupled_basis, "coupled basis")
    lifts = _as_matrix(lift_snapshots, "lift snapshots")
    if lifts.shape[0] != primary.shape[0]:
        raise ValueError("lift snapshot rows must match primary basis rows.")
    pod = fit_pod(lifts, n_modes=n_lift_modes, energy=lift_energy, center=False)
    lift_basis = pod.basis
    enriched = orthonormalize_columns(np.column_stack([primary, lift_basis]), inner_product=inner_product)
    return LiftEnrichment(
        primary_basis=primary,
        coupled_basis=coupled,
        lift_snapshots=lifts,
        lift_basis=np.ascontiguousarray(lift_basis, dtype=np.float64),
        enriched_primary_basis=enriched,
        metadata={
            "primary_modes": int(primary.shape[1]),
            "coupled_modes": int(coupled.shape[1]),
            "lift_modes": int(lift_basis.shape[1]),
            "enriched_primary_modes": int(enriched.shape[1]),
        },
    )


@dataclass(frozen=True)
class SupremizerEnrichment:
    """Compatibility view for velocity-pressure supremizer enrichment."""

    velocity_basis: np.ndarray
    pressure_basis: np.ndarray
    supremizer_snapshots: np.ndarray
    supremizer_basis: np.ndarray
    enriched_velocity_basis: np.ndarray
    metadata: Mapping[str, Any]


def fit_supremizer_enriched_velocity_basis(
    velocity_basis: Any,
    pressure_basis: Any,
    supremizer_snapshots: Any,
    *,
    n_supremizer_modes: int | None = None,
    supremizer_energy: float | None = None,
    inner_product: Any | None = None,
) -> SupremizerEnrichment:
    """Compatibility wrapper for incompressible velocity-pressure ROMs."""

    generic = fit_lift_enriched_basis(
        velocity_basis,
        pressure_basis,
        supremizer_snapshots,
        n_lift_modes=n_supremizer_modes,
        lift_energy=supremizer_energy,
        inner_product=inner_product,
    )
    return SupremizerEnrichment(
        velocity_basis=generic.primary_basis,
        pressure_basis=generic.coupled_basis,
        supremizer_snapshots=generic.lift_snapshots,
        supremizer_basis=generic.lift_basis,
        enriched_velocity_basis=generic.enriched_primary_basis,
        metadata={
            "velocity_modes": int(generic.primary_basis.shape[1]),
            "pressure_modes": int(generic.coupled_basis.shape[1]),
            "supremizer_modes": int(generic.lift_basis.shape[1]),
            "enriched_velocity_modes": int(generic.enriched_primary_basis.shape[1]),
        },
    )


@dataclass(frozen=True)
class MixedBasisBlock:
    """One field block to embed into a mixed global basis."""

    rows: np.ndarray
    basis: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        rows = np.asarray(self.rows, dtype=np.int64).reshape(-1)
        basis = _as_matrix(self.basis, f"{self.name or 'field'} basis")
        if rows.size != basis.shape[0]:
            raise ValueError("mixed basis block rows must match basis rows.")
        object.__setattr__(self, "rows", np.ascontiguousarray(rows, dtype=np.int64))
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "name", str(self.name))


def _coerce_basis_block(block: Any) -> MixedBasisBlock:
    if isinstance(block, MixedBasisBlock):
        return block
    if isinstance(block, Mapping):
        return MixedBasisBlock(
            rows=block["rows"],
            basis=block["basis"],
            name=str(block.get("name", "")),
        )
    if isinstance(block, (tuple, list)) and len(block) in {2, 3}:
        name = "" if len(block) == 2 else str(block[2])
        return MixedBasisBlock(rows=block[0], basis=block[1], name=name)
    raise TypeError("field blocks must be MixedBasisBlock, mapping, or (rows, basis[, name]) tuple.")


@dataclass(frozen=True)
class FieldwisePODBasis:
    """Block-diagonal mixed POD basis fitted independently per field group."""

    basis: np.ndarray
    offset: np.ndarray
    blocks: tuple[MixedBasisBlock, ...]
    singular_values: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "basis", _as_matrix(self.basis, "fieldwise basis"))
        object.__setattr__(self, "offset", _as_vector(self.offset, "fieldwise offset"))
        object.__setattr__(self, "blocks", tuple(_coerce_basis_block(block) for block in self.blocks))
        object.__setattr__(
            self,
            "singular_values",
            {str(key): _as_vector(value, f"{key} singular values") for key, value in dict(self.singular_values).items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class PressureGaugeBlock:
    """Rows and weights defining one pressure/null-mode gauge."""

    rows: np.ndarray
    name: str = ""
    weights: np.ndarray | None = None
    target: float = 0.0

    def __post_init__(self) -> None:
        rows = np.asarray(self.rows, dtype=np.int64).reshape(-1)
        if rows.size == 0 or np.any(rows < 0) or np.unique(rows).size != rows.size:
            raise ValueError("pressure gauge rows must be nonempty, nonnegative, and unique.")
        weights = None if self.weights is None else np.asarray(self.weights, dtype=float).reshape(-1)
        if weights is not None:
            if weights.size != rows.size:
                raise ValueError("pressure gauge weights must match rows.")
            if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
                raise ValueError("pressure gauge weights must be finite and positive.")
            weights = np.ascontiguousarray(weights / float(np.sum(weights)), dtype=np.float64)
        object.__setattr__(self, "rows", np.ascontiguousarray(rows, dtype=np.int64))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "target", float(self.target))


@dataclass(frozen=True)
class GaugeCorrectionResult:
    """Gauge-corrected snapshot matrix and removed gauge histories."""

    corrected_snapshots: np.ndarray
    gauge_histories: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class FieldProjectionError:
    """Per-field projection-error summary."""

    name: str
    max_relative_error: float
    rms_relative_error: float
    max_absolute_error: float
    sample_count: int
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_relative_error": float(self.max_relative_error),
            "rms_relative_error": float(self.rms_relative_error),
            "max_absolute_error": float(self.max_absolute_error),
            "sample_count": int(self.sample_count),
            "passed": bool(self.passed),
        }


@dataclass(frozen=True)
class CouplingRankCertificate:
    """Reduced mixed-coupling rank diagnostic."""

    name: str
    rank: int
    required_rank: int
    singular_values: np.ndarray
    condition_estimate: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rank": int(self.rank),
            "required_rank": int(self.required_rank),
            "singular_values": np.asarray(self.singular_values, dtype=float),
            "condition_estimate": float(self.condition_estimate),
            "passed": bool(self.passed),
        }


@dataclass(frozen=True)
class MixedStabilityCertificate:
    """Projection, gauge, and reduced coupling checks for mixed ROM bases."""

    passed: bool
    field_errors: Mapping[str, FieldProjectionError]
    gauge_max_abs: Mapping[str, float]
    coupling_ranks: Mapping[str, CouplingRankCertificate]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "field_errors": {name: value.to_dict() for name, value in self.field_errors.items()},
            "gauge_max_abs": {name: float(value) for name, value in self.gauge_max_abs.items()},
            "coupling_ranks": {name: value.to_dict() for name, value in self.coupling_ranks.items()},
            "metadata": dict(self.metadata),
        }


def _modes_for_block(n_modes_per_block: int | Mapping[str, int] | None, name: str) -> int | None:
    if n_modes_per_block is None:
        return None
    if isinstance(n_modes_per_block, Mapping):
        if name in n_modes_per_block:
            return max(1, int(n_modes_per_block[name]))
        if "*" in n_modes_per_block:
            return max(1, int(n_modes_per_block["*"]))
        return None
    return max(1, int(n_modes_per_block))


def fit_fieldwise_pod_basis(
    snapshots: Any,
    row_blocks: Sequence[Any],
    *,
    total_dofs: int | None = None,
    n_modes_per_block: int | Mapping[str, int] | None = None,
    energy: float | None = None,
    center: bool = True,
) -> FieldwisePODBasis:
    """Fit an independent POD basis per disjoint mixed-field row block.

    This is the generic basis constructor for saddle-point and multi-physics
    ROMs where a single global Euclidean POD can hide small-but-essential
    fields behind high-magnitude variables.  Each block is centered and reduced
    independently, then embedded into the global mixed DOF ordering.
    """

    matrix = _as_matrix(snapshots, "snapshots")
    n_total = int(matrix.shape[0] if total_dofs is None else total_dofs)
    if n_total != matrix.shape[0]:
        raise ValueError("total_dofs must match the snapshot row count.")
    coerced = tuple(_coerce_row_block(block) for block in row_blocks)
    if not coerced:
        raise ValueError("row_blocks must contain at least one block.")

    offset = np.zeros(n_total, dtype=np.float64)
    basis_blocks: list[MixedBasisBlock] = []
    singular_values: dict[str, np.ndarray] = {}
    metadata_blocks: list[dict[str, Any]] = []
    seen: set[int] = set()
    for rows_raw, raw_name in coerced:
        name = str(raw_name or f"block_{len(metadata_blocks)}")
        rows = np.asarray(rows_raw, dtype=np.int64).reshape(-1)
        if rows.size == 0:
            continue
        if np.any(rows < 0) or np.any(rows >= n_total):
            raise ValueError(f"row block {name!r} contains out-of-range rows.")
        overlap = seen.intersection(int(v) for v in rows)
        if overlap:
            raise ValueError(f"row block {name!r} overlaps previous blocks.")
        seen.update(int(v) for v in rows)
        requested_modes = _modes_for_block(n_modes_per_block, name)
        local = matrix[rows, :]
        pod = fit_pod(local, n_modes=requested_modes, energy=energy, center=bool(center))
        block_basis = np.ascontiguousarray(np.asarray(pod.basis, dtype=float), dtype=np.float64)
        if block_basis.shape[1] == 0:
            continue
        basis_blocks.append(MixedBasisBlock(rows=rows, basis=block_basis, name=name))
        if center:
            offset[rows] = np.asarray(pod.mean, dtype=float).reshape(-1)
        singular_values[name] = np.ascontiguousarray(np.asarray(pod.singular_values, dtype=float).reshape(-1), dtype=np.float64)
        metadata_blocks.append(
            {
                "name": name,
                "rows": int(rows.size),
                "modes": int(block_basis.shape[1]),
                "centered": bool(center),
                "snapshot_count": int(matrix.shape[1]),
            }
        )
    if not basis_blocks:
        raise ValueError("fieldwise POD produced no basis columns.")
    basis = build_mixed_field_basis(total_dofs=n_total, field_blocks=tuple(basis_blocks))
    return FieldwisePODBasis(
        basis=basis,
        offset=np.ascontiguousarray(offset, dtype=np.float64),
        blocks=tuple(basis_blocks),
        singular_values=singular_values,
        metadata={
            "strategy": "fieldwise_pod",
            "total_dofs": int(n_total),
            "snapshot_count": int(matrix.shape[1]),
            "total_modes": int(basis.shape[1]),
            "blocks": tuple(metadata_blocks),
        },
    )


def build_mixed_field_basis(*, total_dofs: int, field_blocks: Sequence[Any]) -> np.ndarray:
    """Embed arbitrary field bases into a full mixed DOF ordering."""

    n = int(total_dofs)
    if n <= 0:
        raise ValueError("total_dofs must be positive.")
    blocks = tuple(_coerce_basis_block(block) for block in field_blocks)
    if not blocks:
        raise ValueError("field_blocks must contain at least one block.")
    seen: set[int] = set()
    n_cols = int(sum(block.basis.shape[1] for block in blocks))
    out = np.zeros((n, n_cols), dtype=np.float64)
    col0 = 0
    for block in blocks:
        rows = block.rows
        if np.any(rows < 0) or np.any(rows >= n):
            raise ValueError(f"mixed basis block {block.name!r} rows are out of range.")
        overlap = seen.intersection(int(v) for v in rows)
        if overlap:
            raise ValueError(f"mixed basis block {block.name!r} overlaps previous rows.")
        seen.update(int(v) for v in rows)
        col1 = col0 + int(block.basis.shape[1])
        out[rows, col0:col1] = block.basis
        col0 = col1
    return np.ascontiguousarray(out, dtype=np.float64)


def build_mixed_velocity_pressure_basis(
    *,
    total_dofs: int,
    velocity_rows: Any,
    velocity_basis: Any,
    pressure_rows: Any,
    pressure_basis: Any,
) -> np.ndarray:
    """Compatibility wrapper for velocity-pressure mixed bases."""

    return build_mixed_field_basis(
        total_dofs=total_dofs,
        field_blocks=(
            MixedBasisBlock(rows=velocity_rows, basis=velocity_basis, name="velocity"),
            MixedBasisBlock(rows=pressure_rows, basis=pressure_basis, name="pressure"),
        ),
    )


def _coerce_row_block(block: Any) -> tuple[np.ndarray, str]:
    if hasattr(block, "rows"):
        return (
            np.asarray(getattr(block, "rows"), dtype=np.int64).reshape(-1),
            str(getattr(block, "name", "")),
        )
    if isinstance(block, Mapping):
        return np.asarray(block["rows"], dtype=np.int64).reshape(-1), str(block.get("name", ""))
    if isinstance(block, (tuple, list)) and len(block) in {1, 2}:
        name = "" if len(block) == 1 else str(block[1])
        return np.asarray(block[0], dtype=np.int64).reshape(-1), name
    return np.asarray(block, dtype=np.int64).reshape(-1), ""


def _coerce_gauge_block(block: Any) -> PressureGaugeBlock:
    if isinstance(block, PressureGaugeBlock):
        return block
    if isinstance(block, Mapping):
        return PressureGaugeBlock(
            rows=block["rows"],
            name=str(block.get("name", "")),
            weights=block.get("weights"),
            target=float(block.get("target", 0.0)),
        )
    if isinstance(block, (tuple, list)) and len(block) in {1, 2, 3}:
        name = "" if len(block) < 2 else str(block[1])
        weights = None if len(block) < 3 else block[2]
        return PressureGaugeBlock(rows=block[0], name=name, weights=weights)
    return PressureGaugeBlock(rows=block)


def pressure_gauge_history(snapshots: Any, gauge_block: Any) -> np.ndarray:
    """Return the weighted pressure/null-mode gauge value per snapshot."""

    matrix = _as_matrix(snapshots, "snapshots")
    gauge = _coerce_gauge_block(gauge_block)
    if np.any(gauge.rows >= matrix.shape[0]):
        raise ValueError("pressure gauge rows are outside the snapshot matrix.")
    weights = gauge.weights
    if weights is None:
        weights = np.full(gauge.rows.size, 1.0 / float(gauge.rows.size), dtype=np.float64)
    history = weights @ matrix[gauge.rows, :]
    return np.ascontiguousarray(history, dtype=np.float64)


def gauge_correct_snapshots(
    snapshots: Any,
    gauge_blocks: Sequence[Any],
) -> GaugeCorrectionResult:
    """Subtract pressure/null-mode gauges from snapshot rows.

    Each gauge block is corrected independently by subtracting its weighted mean
    history from all rows in the block and adding the requested target value.
    This makes POD modes homogeneous with respect to arbitrary pressure offsets
    before mixed-basis construction.
    """

    matrix = _as_matrix(snapshots, "snapshots")
    corrected = matrix.copy()
    histories: dict[str, np.ndarray] = {}
    blocks = tuple(_coerce_gauge_block(block) for block in gauge_blocks)
    if not blocks:
        return GaugeCorrectionResult(
            corrected_snapshots=corrected,
            gauge_histories={},
            metadata={"gauge_blocks": 0},
        )
    for idx, gauge in enumerate(blocks):
        if np.any(gauge.rows >= matrix.shape[0]):
            raise ValueError(f"pressure gauge block {gauge.name!r} contains out-of-range rows.")
        name = gauge.name or f"gauge_{idx}"
        history = pressure_gauge_history(corrected, gauge)
        corrected[gauge.rows, :] -= history[None, :]
        if gauge.target:
            corrected[gauge.rows, :] += float(gauge.target)
        histories[name] = np.ascontiguousarray(history - float(gauge.target), dtype=np.float64)
    return GaugeCorrectionResult(
        corrected_snapshots=np.ascontiguousarray(corrected, dtype=np.float64),
        gauge_histories=histories,
        metadata={
            "gauge_blocks": len(blocks),
            "rows": {gauge.name or f"gauge_{idx}": int(gauge.rows.size) for idx, gauge in enumerate(blocks)},
        },
    )


def field_projection_errors(
    snapshots: Any,
    basis: Any,
    *,
    offset: Any | None = None,
    row_blocks: Sequence[Any],
    tolerance: float = 5.0e-2,
    absolute_tolerance: float | None = None,
    norm_floor: float = 1.0e-14,
) -> dict[str, FieldProjectionError]:
    """Compute per-field projection errors for a mixed reduced basis."""

    matrix = _as_matrix(snapshots, "snapshots")
    V = _as_matrix(basis, "basis")
    if V.shape[0] != matrix.shape[0]:
        raise ValueError("basis rows must match snapshots.")
    x0 = np.zeros(matrix.shape[0], dtype=np.float64) if offset is None else _as_vector(offset, "offset")
    if x0.size != matrix.shape[0]:
        raise ValueError("offset size must match snapshots.")
    coefficients, *_ = np.linalg.lstsq(V, matrix - x0[:, None], rcond=None)
    reconstructed = x0[:, None] + V @ coefficients
    errors: dict[str, FieldProjectionError] = {}
    for block_idx, block in enumerate(row_blocks):
        rows, name_raw = _coerce_row_block(block)
        rows = rows[(rows >= 0) & (rows < matrix.shape[0])]
        name = str(name_raw or f"block_{block_idx}")
        if rows.size == 0:
            errors[name] = FieldProjectionError(
                name=name,
                max_relative_error=0.0,
                rms_relative_error=0.0,
                max_absolute_error=0.0,
                sample_count=int(matrix.shape[1]),
                passed=True,
            )
            continue
        diff = reconstructed[rows, :] - matrix[rows, :]
        abs_norms = np.linalg.norm(diff, axis=0)
        ref_norms = np.maximum(np.linalg.norm(matrix[rows, :], axis=0), float(norm_floor))
        rel = abs_norms / ref_norms
        max_rel = float(np.max(rel)) if rel.size else 0.0
        max_abs = float(np.max(abs_norms)) if abs_norms.size else 0.0
        abs_tol = float(tolerance if absolute_tolerance is None else absolute_tolerance)
        errors[name] = FieldProjectionError(
            name=name,
            max_relative_error=max_rel,
            rms_relative_error=float(np.sqrt(np.mean(rel * rel))) if rel.size else 0.0,
            max_absolute_error=max_abs,
            sample_count=int(matrix.shape[1]),
            passed=bool(max_rel <= float(tolerance) or max_abs <= abs_tol),
        )
    return errors


def reduced_coupling_rank_certificate(
    coupling_operator: Any,
    primary_basis: Any,
    coupled_basis: Any,
    *,
    name: str = "coupling",
    required_rank: int | None = None,
    tolerance: float = 1.0e-10,
) -> CouplingRankCertificate:
    """Check rank of a reduced mixed coupling such as pressure/divergence."""

    C = coupling_operator
    if _is_sparse(C):
        Cmat = C
    else:
        Cmat = _as_matrix(C, "coupling operator")
    Vp = _as_matrix(primary_basis, "primary basis")
    Wc = _as_matrix(coupled_basis, "coupled basis")
    if Cmat.shape[1] != Vp.shape[0] or Cmat.shape[0] != Wc.shape[0]:
        raise ValueError("coupling operator shape must be (coupled rows, primary rows).")
    reduced = np.ascontiguousarray(Wc.T @ _operator_matmul(Cmat, Vp), dtype=np.float64)
    singular_values = np.linalg.svd(reduced, compute_uv=False)
    scale = float(singular_values[0]) if singular_values.size else 0.0
    threshold = float(tolerance) * max(scale, 1.0)
    rank = int(np.count_nonzero(singular_values > threshold))
    required = int(min(reduced.shape) if required_rank is None else required_rank)
    positive = singular_values[singular_values > threshold]
    condition = float(singular_values[0] / positive[-1]) if positive.size else float("inf")
    return CouplingRankCertificate(
        name=str(name),
        rank=rank,
        required_rank=required,
        singular_values=np.ascontiguousarray(singular_values, dtype=np.float64),
        condition_estimate=condition,
        passed=bool(rank >= required),
    )


def certify_mixed_stability_basis(
    snapshots: Any,
    basis: Any,
    *,
    offset: Any | None = None,
    row_blocks: Sequence[Any],
    pressure_gauge_blocks: Sequence[Any] | None = None,
    projection_tolerance: float = 5.0e-2,
    gauge_tolerance: float = 1.0e-10,
    coupling_certificates: Sequence[CouplingRankCertificate | Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MixedStabilityCertificate:
    """Build a mixed-basis acceptance certificate.

    This combines the gates needed before using a mixed ROM in a nonlinear
    native solve: per-field projection accuracy, pressure/null-mode gauges, and
    supplied reduced coupling rank certificates.
    """

    matrix = _as_matrix(snapshots, "snapshots")
    gauges: dict[str, float] = {}
    if pressure_gauge_blocks:
        corrected = gauge_correct_snapshots(matrix, pressure_gauge_blocks)
        for idx, gauge_raw in enumerate(pressure_gauge_blocks):
            gauge = _coerce_gauge_block(gauge_raw)
            name = gauge.name or f"gauge_{idx}"
            residual_history = pressure_gauge_history(corrected.corrected_snapshots, gauge) - float(gauge.target)
            gauges[name] = float(np.max(np.abs(residual_history))) if residual_history.size else 0.0
        matrix_for_projection = corrected.corrected_snapshots
    else:
        matrix_for_projection = matrix
    errors = field_projection_errors(
        matrix_for_projection,
        basis,
        offset=offset,
        row_blocks=row_blocks,
        tolerance=projection_tolerance,
    )
    ranks: dict[str, CouplingRankCertificate] = {}
    for idx, raw in enumerate(coupling_certificates or ()):
        if isinstance(raw, CouplingRankCertificate):
            cert = raw
        else:
            cert = CouplingRankCertificate(
                name=str(raw.get("name", f"coupling_{idx}")),
                rank=int(raw["rank"]),
                required_rank=int(raw["required_rank"]),
                singular_values=np.asarray(raw.get("singular_values", []), dtype=float),
                condition_estimate=float(raw.get("condition_estimate", float("nan"))),
                passed=bool(raw["passed"]),
            )
        ranks[cert.name] = cert
    passed = bool(
        all(error.passed for error in errors.values())
        and all(value <= float(gauge_tolerance) for value in gauges.values())
        and all(cert.passed for cert in ranks.values())
    )
    return MixedStabilityCertificate(
        passed=passed,
        field_errors=errors,
        gauge_max_abs=gauges,
        coupling_ranks=ranks,
        metadata={
            "projection_tolerance": float(projection_tolerance),
            "gauge_tolerance": float(gauge_tolerance),
            "snapshot_count": int(matrix.shape[1]),
            **dict(metadata or {}),
        },
    )


def build_block_row_weights(
    reference_matrix: Any,
    row_blocks: Sequence[Any],
    *,
    min_rms: float = 1.0e-14,
    min_weight: float = 1.0e-8,
    max_weight: float = 1.0e8,
) -> np.ndarray:
    """Build positive row weights that balance residual blocks by RMS scale.

    The returned weights are intended for least-squares objectives where rows
    are scaled by ``sqrt(weight)``.  Each supplied block is scaled so that its
    RMS magnitude matches the largest nonzero block RMS in ``reference_matrix``.
    This is useful for mixed saddle-point systems where one residual block can
    otherwise dominate DEIM/QDEIM row selection and the online reduced target.
    """

    matrix = _as_matrix(reference_matrix, "reference matrix")
    if not row_blocks:
        raise ValueError("row_blocks must contain at least one block.")
    if min_rms <= 0.0 or min_weight <= 0.0 or max_weight <= 0.0 or min_weight > max_weight:
        raise ValueError("weight bounds must be positive and ordered.")

    coerced = tuple(_coerce_row_block(block) for block in row_blocks)
    block_rms: list[float] = []
    for rows, name in coerced:
        del name
        if rows.size == 0:
            block_rms.append(0.0)
            continue
        if np.any(rows < 0) or np.any(rows >= matrix.shape[0]):
            raise ValueError("row block contains out-of-range rows.")
        block = matrix[rows, :]
        block_rms.append(float(np.sqrt(np.mean(block * block))))
    nonzero = [value for value in block_rms if value > float(min_rms)]
    target = max(nonzero) if nonzero else 1.0

    weights = np.ones(matrix.shape[0], dtype=np.float64)
    for (rows, _name), rms in zip(coerced, block_rms):
        if rows.size == 0:
            continue
        effective = max(float(rms), float(min_rms))
        weight = (target / effective) ** 2
        weights[rows] = np.clip(weight, float(min_weight), float(max_weight))
    return np.ascontiguousarray(weights, dtype=np.float64)


@dataclass(frozen=True)
class NonAffineReducedDecomposition:
    """Collateral basis, interpolation rule, and reduced DEIM terms."""

    collateral_basis: CollateralBasis
    interpolation_rule: InterpolationRule
    residual_terms: np.ndarray
    metadata: Mapping[str, Any]
    row_weights: np.ndarray | None = None
    sampled_row_weights: np.ndarray | None = None


def build_nonaffine_reduced_decomposition(
    residual_snapshots: Any,
    trial_basis: Any,
    *,
    n_modes: int | None = None,
    energy: float | None = None,
    method: str = "qdeim",
    center: bool = False,
    rcond: float | None = None,
    row_weights: Any | None = None,
) -> NonAffineReducedDecomposition:
    """Build DEIM/QDEIM data for a non-affine residual contribution."""

    R = _as_matrix(residual_snapshots, "residual snapshots")
    V = _as_matrix(trial_basis, "trial basis")
    if R.shape[0] != V.shape[0]:
        raise ValueError("residual snapshot rows must match trial basis rows.")
    weights = None if row_weights is None else _as_positive_weights(row_weights, R.shape[0], "row_weights")
    if weights is None:
        fit_matrix = R
    else:
        fit_matrix = np.ascontiguousarray(np.sqrt(weights)[:, None] * R, dtype=np.float64)
    collateral = fit_collateral_basis(fit_matrix, n_modes=n_modes, energy=energy, center=center)
    name = str(method).strip().lower()
    if name == "deim":
        rule = build_deim_interpolation_rule(collateral, rcond=rcond)
    elif name == "qdeim":
        rule = build_qdeim_interpolation_rule(collateral, rcond=rcond)
    else:
        raise ValueError("method must be 'deim' or 'qdeim'.")
    if weights is None:
        reduced_collateral = collateral.basis
        sampled_weights = None
    else:
        reduced_collateral = collateral.basis / np.sqrt(weights)[:, None]
        sampled_weights = np.ascontiguousarray(weights[rule.rows], dtype=np.float64)
    residual_terms = np.ascontiguousarray(reduced_collateral.T @ V, dtype=np.float64)
    return NonAffineReducedDecomposition(
        collateral_basis=collateral,
        interpolation_rule=rule,
        residual_terms=residual_terms,
        metadata={
            "method": name,
            "residual_features": int(R.shape[0]),
            "snapshot_count": int(R.shape[1]),
            "trial_modes": int(V.shape[1]),
            "collateral_modes": int(collateral.n_modes),
            "row_weighted": bool(weights is not None),
        },
        row_weights=weights,
        sampled_row_weights=sampled_weights,
    )


__all__ = [
    "FieldwisePODBasis",
    "CouplingRankCertificate",
    "FieldProjectionError",
    "GaugeCorrectionResult",
    "LiftEnrichment",
    "MixedBasisBlock",
    "MixedStabilityCertificate",
    "NonAffineReducedDecomposition",
    "PressureGaugeBlock",
    "SupremizerEnrichment",
    "build_dirichlet_lifting_vector",
    "build_block_row_weights",
    "build_mixed_field_basis",
    "build_mixed_velocity_pressure_basis",
    "build_nonaffine_reduced_decomposition",
    "compute_supremizer_snapshots",
    "certify_mixed_stability_basis",
    "field_dof_indices",
    "field_projection_errors",
    "fit_lift_enriched_basis",
    "fit_fieldwise_pod_basis",
    "fit_supremizer_enriched_velocity_basis",
    "gauge_correct_snapshots",
    "orthonormalize_columns",
    "pressure_gauge_history",
    "remove_lifting_from_snapshots",
    "reduced_coupling_rank_certificate",
    "restore_lifting_to_snapshots",
    "solve_coupled_lift_snapshots",
]
