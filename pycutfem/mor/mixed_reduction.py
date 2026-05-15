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
    if isinstance(block, Mapping):
        return np.asarray(block["rows"], dtype=np.int64).reshape(-1), str(block.get("name", ""))
    if isinstance(block, (tuple, list)) and len(block) in {1, 2}:
        name = "" if len(block) == 1 else str(block[1])
        return np.asarray(block[0], dtype=np.int64).reshape(-1), name
    return np.asarray(block, dtype=np.int64).reshape(-1), ""


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
    "LiftEnrichment",
    "MixedBasisBlock",
    "NonAffineReducedDecomposition",
    "SupremizerEnrichment",
    "build_dirichlet_lifting_vector",
    "build_block_row_weights",
    "build_mixed_field_basis",
    "build_mixed_velocity_pressure_basis",
    "build_nonaffine_reduced_decomposition",
    "compute_supremizer_snapshots",
    "field_dof_indices",
    "fit_lift_enriched_basis",
    "fit_supremizer_enriched_velocity_basis",
    "orthonormalize_columns",
    "remove_lifting_from_snapshots",
    "restore_lifting_to_snapshots",
    "solve_coupled_lift_snapshots",
]
