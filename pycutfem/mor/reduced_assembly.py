"""Reduced local assembly and projection utilities.

This module owns backend-neutral MOR algebra that used to live in example
drivers: affine reduced-state decoding, sampled LSPG row projection, reduced
Galerkin projection, constrained-row reaction extraction, and a small UFL-backed
assembler wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _normalize_backend_name(backend: str | None) -> str:
    name = str(backend or "python").strip().lower()
    if name in {"c++", "cpp"}:
        return "cpp"
    if name not in {"python", "jit", "cpp"}:
        raise ValueError(f"Unsupported reduced assembly backend {backend!r}.")
    return name


def _normalize_projection_backend_name(backend: str | None) -> str:
    name = str(backend or "python").strip().lower()
    if name in {"c++", "cpp"}:
        return "cpp"
    if name in {"python", "jit"}:
        return "python"
    raise ValueError(f"Unsupported reduced projection backend {backend!r}.")


def _cpp_projection_module():
    from .cpp_backend.reduced_projection import module as _reduced_projection_cpp_module

    return _reduced_projection_cpp_module()


def _as_vector(values: np.ndarray, *, name: str, dtype=float) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if not np.all(np.isfinite(arr)) and np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def validate_element_weights(
    element_count: int,
    element_weights: np.ndarray | None,
    *,
    context: str = "reduced local assembly",
) -> np.ndarray | None:
    """Validate optional empirical-cubature weights for element-local blocks."""

    if element_weights is None:
        return None
    weights = np.asarray(element_weights, dtype=float).reshape(-1)
    if int(weights.size) != int(element_count):
        raise ValueError(f"{context} element_weights size must match the sampled element count.")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError(f"{context} element_weights must be finite and nonnegative.")
    return weights


def validate_local_blocks(
    *,
    K_elem: np.ndarray,
    vector_elem: np.ndarray,
    gdofs_map: np.ndarray,
    trial_basis: np.ndarray,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize element-local blocks used by reduced projections."""

    K = np.asarray(K_elem, dtype=float)
    vec = np.asarray(vector_elem, dtype=float)
    gdofs = np.asarray(gdofs_map, dtype=int)
    basis = np.asarray(trial_basis, dtype=float)
    if K.ndim != 3 or vec.ndim != 2 or gdofs.ndim != 2:
        raise ValueError(f"{context} local blocks must have shapes K=(e,l,l), vector=(e,l), gdofs=(e,l).")
    if int(K.shape[0]) != int(vec.shape[0]) or int(K.shape[0]) != int(gdofs.shape[0]):
        raise ValueError(f"{context} local block element counts are inconsistent.")
    if int(K.shape[1]) != int(K.shape[2]) or int(K.shape[1]) != int(vec.shape[1]):
        raise ValueError(f"{context} local block row/column sizes are inconsistent.")
    if int(gdofs.shape[1]) != int(K.shape[1]):
        raise ValueError(f"{context} gdofs_map local size is incompatible with K_elem.")
    if basis.ndim != 2:
        raise ValueError(f"{context} trial_basis must be a 2-D array.")
    if gdofs.size:
        if np.any(gdofs < 0) or np.any(gdofs >= int(basis.shape[0])):
            raise ValueError(f"{context} gdofs_map contains dofs outside trial_basis.")
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(vec)) or not np.all(np.isfinite(basis)):
        raise ValueError(f"{context} local blocks and trial_basis must be finite.")
    return K, vec, gdofs, basis


@dataclass(frozen=True)
class AffineReducedState:
    """Affine reduced state ``u = offset + basis @ coefficients``."""

    basis: np.ndarray
    offset: np.ndarray

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis, dtype=float)
        offset = np.asarray(self.offset, dtype=float).reshape(-1)
        if basis.ndim != 2:
            raise ValueError("reduced basis must be a 2-D array.")
        if int(basis.shape[0]) != int(offset.size):
            raise ValueError("reduced basis and offset have incompatible row counts.")
        if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(offset)):
            raise ValueError("reduced basis and offset must be finite.")
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "offset", offset)

    @property
    def n_dofs(self) -> int:
        return int(self.basis.shape[0])

    @property
    def n_modes(self) -> int:
        return int(self.basis.shape[1])

    def validate_coefficients(self, coefficients: np.ndarray) -> np.ndarray:
        coeffs = _as_vector(coefficients, name="reduced coefficients", dtype=float)
        if int(coeffs.size) != self.n_modes:
            raise ValueError(f"expected {self.n_modes} reduced coefficients, got {coeffs.size}.")
        return coeffs

    def values_on_dofs(self, dofs: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        return decode_values_on_dofs(
            offset=self.offset,
            basis=self.basis,
            dofs=dofs,
            coefficients=coefficients,
        )

    def element_values(self, local_map: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        return decode_element_values(
            offset=self.offset,
            basis=self.basis,
            local_map=local_map,
            coefficients=coefficients,
        )

    def full_values(self, coefficients: np.ndarray) -> np.ndarray:
        coeffs = self.validate_coefficients(coefficients)
        return np.asarray(self.offset + self.basis @ coeffs, dtype=float).reshape(-1)


def decode_values_on_dofs(
    *,
    offset: np.ndarray,
    basis: np.ndarray,
    dofs: np.ndarray,
    coefficients: np.ndarray,
    backend: str = "python",
) -> np.ndarray:
    """Decode an affine reduced state on a selected global DOF set."""

    if _normalize_projection_backend_name(backend) == "cpp":
        return np.asarray(
            _cpp_projection_module().decode_values_on_dofs(offset, basis, dofs, coefficients),
            dtype=float,
        ).reshape(-1)
    state = AffineReducedState(basis=basis, offset=offset)
    coeffs = state.validate_coefficients(coefficients)
    ids = np.asarray(dofs, dtype=int).reshape(-1)
    if ids.size == 0:
        return np.zeros(0, dtype=float)
    if np.any(ids < 0) or np.any(ids >= state.n_dofs):
        raise ValueError("sampled reduced dofs contain out-of-range entries.")
    return np.asarray(state.offset[ids] + state.basis[ids, :] @ coeffs, dtype=float).reshape(-1)


def decode_element_values(
    *,
    offset: np.ndarray,
    basis: np.ndarray,
    local_map: np.ndarray,
    coefficients: np.ndarray,
    backend: str = "python",
) -> np.ndarray:
    """Decode an affine reduced state on an element-local DOF map."""

    mapping = np.asarray(local_map, dtype=int)
    if mapping.ndim != 2:
        raise ValueError("local_map must have shape (n_elements, n_local_dofs).")
    if _normalize_projection_backend_name(backend) == "cpp":
        return np.asarray(
            _cpp_projection_module().decode_element_values(offset, basis, mapping, coefficients),
            dtype=float,
        )
    if mapping.size == 0:
        return np.zeros(mapping.shape, dtype=float)
    values = decode_values_on_dofs(
        offset=offset,
        basis=basis,
        dofs=mapping.reshape(-1),
        coefficients=coefficients,
        backend=backend,
    )
    return values.reshape(mapping.shape)


def sampled_lspg_element_contributions_from_local_blocks(
    *,
    K_elem: np.ndarray,
    raw_rhs_elem: np.ndarray,
    gdofs_map: np.ndarray,
    row_dofs: np.ndarray,
    trial_basis: np.ndarray,
    backend: str = "python",
) -> tuple[np.ndarray, np.ndarray]:
    """Project element-local blocks into sampled LSPG rows.

    ``raw_rhs_elem`` follows a local-system RHS convention. The returned
    residual uses Newton residual sign, i.e. ``residual = -raw_rhs`` on sampled
    rows.
    """

    if _normalize_projection_backend_name(backend) == "cpp":
        residual, trial = _cpp_projection_module().sampled_lspg_element_contributions_from_local_blocks(
            K_elem,
            raw_rhs_elem,
            gdofs_map,
            row_dofs,
            trial_basis,
        )
        return np.asarray(residual, dtype=float), np.asarray(trial, dtype=float)

    K, raw_rhs, gdofs, basis = validate_local_blocks(
        K_elem=K_elem,
        vector_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        trial_basis=trial_basis,
        context="sampled LSPG",
    )
    rows = np.asarray(row_dofs, dtype=int).reshape(-1)
    if np.any(rows < 0) or np.any(rows >= int(basis.shape[0])):
        raise ValueError("sampled LSPG row_dofs contain out-of-range entries.")
    if np.unique(rows).size != rows.size:
        raise ValueError("sampled LSPG row_dofs must be unique.")

    row_positions = np.full(int(basis.shape[0]), -1, dtype=int)
    row_positions[rows] = np.arange(int(rows.size), dtype=int)
    local_row_positions = row_positions[gdofs]
    elem_idx, local_i = np.nonzero(local_row_positions >= 0)

    residual_by_element = np.zeros((int(K.shape[0]), int(rows.size)), dtype=float)
    trial_by_element = np.zeros((int(K.shape[0]), int(rows.size), int(basis.shape[1])), dtype=float)
    if elem_idx.size:
        sample_idx = local_row_positions[elem_idx, local_i]
        np.add.at(residual_by_element, (elem_idx, sample_idx), raw_rhs[elem_idx, local_i])
        local_basis = basis[gdofs[elem_idx], :]
        local_k_rows = K[elem_idx, local_i, :]
        trial_contrib = np.einsum("sl,slm->sm", local_k_rows, local_basis, optimize=True)
        np.add.at(trial_by_element, (elem_idx, sample_idx), trial_contrib)
    return -residual_by_element, trial_by_element


def sampled_lspg_rows_from_local_blocks(
    *,
    K_elem: np.ndarray,
    raw_rhs_elem: np.ndarray,
    gdofs_map: np.ndarray,
    row_dofs: np.ndarray,
    trial_basis: np.ndarray,
    element_weights: np.ndarray | None = None,
    backend: str = "python",
) -> tuple[np.ndarray, np.ndarray]:
    """Sum weighted element-local sampled LSPG rows without full-space scatter."""

    if _normalize_projection_backend_name(backend) == "cpp":
        residual, trial = _cpp_projection_module().sampled_lspg_rows_from_local_blocks(
            K_elem,
            raw_rhs_elem,
            gdofs_map,
            row_dofs,
            trial_basis,
            element_weights,
        )
        return np.asarray(residual, dtype=float).reshape(-1), np.asarray(trial, dtype=float)

    residual_by_element, trial_by_element = sampled_lspg_element_contributions_from_local_blocks(
        K_elem=K_elem,
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        row_dofs=row_dofs,
        trial_basis=trial_basis,
        backend=backend,
    )
    weights = validate_element_weights(
        int(residual_by_element.shape[0]),
        element_weights,
        context="sampled LSPG",
    )
    if weights is not None:
        residual_by_element = residual_by_element * weights[:, None]
        trial_by_element = trial_by_element * weights[:, None, None]
    return np.sum(residual_by_element, axis=0).reshape(-1), np.sum(trial_by_element, axis=0)


def sampled_galerkin_element_contributions_from_local_blocks(
    *,
    K_elem: np.ndarray,
    residual_elem: np.ndarray,
    gdofs_map: np.ndarray,
    trial_basis: np.ndarray,
    backend: str = "python",
) -> tuple[np.ndarray, np.ndarray]:
    """Project element-local residual/tangent blocks into a reduced space."""

    if _normalize_projection_backend_name(backend) == "cpp":
        residual, tangent = _cpp_projection_module().sampled_galerkin_element_contributions_from_local_blocks(
            K_elem,
            residual_elem,
            gdofs_map,
            trial_basis,
        )
        return np.asarray(residual, dtype=float), np.asarray(tangent, dtype=float)

    K, residual, gdofs, basis = validate_local_blocks(
        K_elem=K_elem,
        vector_elem=residual_elem,
        gdofs_map=gdofs_map,
        trial_basis=trial_basis,
        context="sampled Galerkin",
    )
    local_basis = basis[gdofs, :]
    reduced_residual_by_element = np.einsum("elm,el->em", local_basis, residual, optimize=True)
    reduced_tangent_by_element = np.einsum("eim,eij,ejn->emn", local_basis, K, local_basis, optimize=True)
    return reduced_residual_by_element, reduced_tangent_by_element


def sampled_galerkin_reduced_system_from_local_blocks(
    *,
    K_elem: np.ndarray,
    residual_elem: np.ndarray,
    gdofs_map: np.ndarray,
    trial_basis: np.ndarray,
    element_weights: np.ndarray | None = None,
    backend: str = "python",
) -> tuple[np.ndarray, np.ndarray]:
    """Sum weighted element-local Galerkin residual/tangent contributions."""

    if _normalize_projection_backend_name(backend) == "cpp":
        residual, tangent = _cpp_projection_module().sampled_galerkin_reduced_system_from_local_blocks(
            K_elem,
            residual_elem,
            gdofs_map,
            trial_basis,
            element_weights,
        )
        return np.asarray(residual, dtype=float).reshape(-1), np.asarray(tangent, dtype=float)

    residual_by_element, tangent_by_element = sampled_galerkin_element_contributions_from_local_blocks(
        K_elem=K_elem,
        residual_elem=residual_elem,
        gdofs_map=gdofs_map,
        trial_basis=trial_basis,
        backend=backend,
    )
    weights = validate_element_weights(
        int(residual_by_element.shape[0]),
        element_weights,
        context="sampled Galerkin",
    )
    if weights is not None:
        residual_by_element = residual_by_element * weights[:, None]
        tangent_by_element = tangent_by_element * weights[:, None, None]
    return np.sum(residual_by_element, axis=0).reshape(-1), np.sum(tangent_by_element, axis=0)


def constrained_reaction_rows_from_local_blocks(
    *,
    raw_rhs_elem: np.ndarray,
    gdofs_map: np.ndarray,
    constrained_row_dofs: np.ndarray,
    element_weights: np.ndarray | None = None,
    backend: str = "python",
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate constrained-row reactions without building a full vector."""

    if _normalize_projection_backend_name(backend) == "cpp":
        rows, reaction = _cpp_projection_module().constrained_reaction_rows_from_local_blocks(
            raw_rhs_elem,
            gdofs_map,
            constrained_row_dofs,
            element_weights,
        )
        return np.asarray(rows, dtype=int).reshape(-1), np.asarray(reaction, dtype=float).reshape(-1)

    raw_rhs = np.asarray(raw_rhs_elem, dtype=float)
    gdofs = np.asarray(gdofs_map, dtype=int)
    if raw_rhs.ndim != 2 or gdofs.ndim != 2:
        raise ValueError("reaction local blocks must have shapes raw_rhs=(e,l), gdofs=(e,l).")
    if raw_rhs.shape != gdofs.shape:
        raise ValueError("reaction raw_rhs_elem and gdofs_map must have the same shape.")
    if not np.all(np.isfinite(raw_rhs)):
        raise ValueError("reaction raw_rhs_elem must be finite.")
    if gdofs.size and np.any(gdofs < 0):
        raise ValueError("reaction gdofs_map contains negative dofs.")
    rows = np.asarray(constrained_row_dofs, dtype=int).reshape(-1)
    if rows.size == 0:
        return rows.astype(int, copy=False), np.zeros(0, dtype=float)
    if np.any(rows < 0):
        raise ValueError("constrained_row_dofs contains negative dofs.")
    if np.unique(rows).size != rows.size:
        raise ValueError("constrained_row_dofs must be unique.")

    weights = validate_element_weights(
        int(raw_rhs.shape[0]),
        element_weights,
        context="constrained reaction",
    )
    if weights is not None:
        raw_rhs = raw_rhs * weights[:, None]

    row_positions = {int(row): idx for idx, row in enumerate(rows)}
    values = np.zeros(int(rows.size), dtype=float)
    for elem_idx in range(int(gdofs.shape[0])):
        for local_idx, gdof in enumerate(gdofs[elem_idx]):
            pos = row_positions.get(int(gdof))
            if pos is not None:
                values[pos] -= float(raw_rhs[elem_idx, local_idx])
    return rows.astype(int, copy=False), values


def reduced_reaction_from_local_blocks(
    *,
    raw_rhs_elem: np.ndarray,
    gdofs_map: np.ndarray,
    constrained_row_dofs: np.ndarray,
    row_to_reduced_load: np.ndarray,
    element_weights: np.ndarray | None = None,
    backend: str = "python",
) -> np.ndarray:
    """Map constrained-row local reaction contributions to load coordinates."""

    if _normalize_projection_backend_name(backend) == "cpp":
        return np.asarray(
            _cpp_projection_module().reduced_reaction_from_local_blocks(
                raw_rhs_elem,
                gdofs_map,
                constrained_row_dofs,
                row_to_reduced_load,
                element_weights,
            ),
            dtype=float,
        ).reshape(-1)

    _rows, reaction_rows = constrained_reaction_rows_from_local_blocks(
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        constrained_row_dofs=constrained_row_dofs,
        element_weights=element_weights,
        backend=backend,
    )
    transfer = np.asarray(row_to_reduced_load, dtype=float)
    if transfer.ndim != 2:
        raise ValueError("row_to_reduced_load must be a 2-D matrix.")
    if int(transfer.shape[1]) != int(reaction_rows.size):
        raise ValueError("row_to_reduced_load columns must match constrained reaction rows.")
    if not np.all(np.isfinite(transfer)):
        raise ValueError("row_to_reduced_load must be finite.")
    return np.asarray(transfer @ reaction_rows, dtype=float).reshape(-1)


def apply_gnat_lift(
    *,
    sample_to_residual_coefficients: np.ndarray,
    sampled_residual: np.ndarray,
    sampled_trial_jacobian: np.ndarray,
    backend: str = "python",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a GNAT residual lift to sampled residual/Jacobian rows."""

    from .sparse import NativeSparseMatrix, apply_sparse_gnat_lift, is_sparse_matrix_like

    if is_sparse_matrix_like(sample_to_residual_coefficients):
        lift = NativeSparseMatrix.coerce(sample_to_residual_coefficients)
        return apply_sparse_gnat_lift(
            lift,
            sampled_residual,
            sampled_trial_jacobian,
            backend=_normalize_projection_backend_name(backend),
        )

    if _normalize_projection_backend_name(backend) == "cpp":
        residual, trial = _cpp_projection_module().apply_gnat_lift(
            sample_to_residual_coefficients,
            sampled_residual,
            sampled_trial_jacobian,
        )
        return np.asarray(residual, dtype=float).reshape(-1), np.asarray(trial, dtype=float)
    lift = np.asarray(sample_to_residual_coefficients, dtype=float)
    residual = np.asarray(sampled_residual, dtype=float).reshape(-1)
    trial = np.asarray(sampled_trial_jacobian, dtype=float)
    if lift.ndim != 2 or trial.ndim != 2:
        raise ValueError("GNAT lift and sampled_trial_jacobian must be 2-D matrices.")
    if int(lift.shape[1]) != int(residual.size) or int(lift.shape[1]) != int(trial.shape[0]):
        raise ValueError("GNAT lift columns must match sampled residual/Jacobian rows.")
    if not np.all(np.isfinite(lift)) or not np.all(np.isfinite(residual)) or not np.all(np.isfinite(trial)):
        raise ValueError("GNAT lift inputs must be finite.")
    return np.asarray(lift @ residual, dtype=float).reshape(-1), np.asarray(lift @ trial, dtype=float)


@dataclass(frozen=True)
class ReducedLocalAssembler:
    """UFL-backed local assembler with MOR projection targets.

    The UFL compiler owns local block assembly for ``python``, ``jit``, and
    generated ``cpp`` backends. This wrapper then applies generic reduced-space
    projections without requiring problem-specific C++ modules.
    """

    dof_handler: Any
    form_or_equation: Any
    trial_basis: np.ndarray
    quadrature_order: int | None = None
    backend: str = "python"
    element_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        basis = np.asarray(self.trial_basis, dtype=float)
        if basis.ndim != 2:
            raise ValueError("trial_basis must be a 2-D array.")
        total_dofs = int(getattr(self.dof_handler, "total_dofs"))
        if int(basis.shape[0]) != total_dofs:
            raise ValueError("trial_basis row count must match dof_handler.total_dofs.")
        if not np.all(np.isfinite(basis)):
            raise ValueError("trial_basis must contain only finite values.")
        object.__setattr__(self, "trial_basis", basis)
        object.__setattr__(self, "backend", _normalize_backend_name(self.backend))
        if self.element_ids is not None:
            object.__setattr__(self, "element_ids", np.asarray(self.element_ids, dtype=int).reshape(-1))

    def local_blocks(self, *, need_matrix: bool = True, need_vector: bool = True):
        """Assemble UFL local blocks through ``FormCompiler``."""

        from pycutfem.ufl.compilers import FormCompiler

        compiler = FormCompiler(
            self.dof_handler,
            quadrature_order=self.quadrature_order,
            backend=self.backend,
        )
        return compiler.assemble_local_contributions(
            self.form_or_equation,
            entity_ids=self.element_ids,
            need_matrix=bool(need_matrix),
            need_vector=bool(need_vector),
        )

    def galerkin_system(self, *, element_weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Assemble ``V.T @ R`` and ``V.T @ J @ V`` from UFL local blocks."""

        batch = self.local_blocks(need_matrix=True, need_vector=True)
        if batch.K_elem is None or batch.F_elem is None:
            raise RuntimeError("Reduced Galerkin projection requires both local K_elem and F_elem.")
        return sampled_galerkin_reduced_system_from_local_blocks(
            K_elem=batch.K_elem,
            residual_elem=batch.F_elem,
            gdofs_map=batch.gdofs_map,
            trial_basis=self.trial_basis,
            element_weights=element_weights,
            backend="cpp" if self.backend == "cpp" else "python",
        )

    def sampled_lspg_rows(
        self,
        *,
        row_dofs: np.ndarray,
        element_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assemble sampled residual rows and ``J @ V`` rows from UFL local blocks."""

        batch = self.local_blocks(need_matrix=True, need_vector=True)
        if batch.K_elem is None or batch.F_elem is None:
            raise RuntimeError("Sampled LSPG projection requires both local K_elem and F_elem.")
        # The low-level LSPG helper uses local-system RHS sign, so negate the
        # Newton residual returned by FormCompiler.
        return sampled_lspg_rows_from_local_blocks(
            K_elem=batch.K_elem,
            raw_rhs_elem=-np.asarray(batch.F_elem, dtype=float),
            gdofs_map=batch.gdofs_map,
            row_dofs=row_dofs,
            trial_basis=self.trial_basis,
            element_weights=element_weights,
            backend="cpp" if self.backend == "cpp" else "python",
        )


__all__ = [
    "AffineReducedState",
    "ReducedLocalAssembler",
    "apply_gnat_lift",
    "constrained_reaction_rows_from_local_blocks",
    "decode_element_values",
    "decode_values_on_dofs",
    "reduced_reaction_from_local_blocks",
    "sampled_galerkin_element_contributions_from_local_blocks",
    "sampled_galerkin_reduced_system_from_local_blocks",
    "sampled_lspg_element_contributions_from_local_blocks",
    "sampled_lspg_rows_from_local_blocks",
    "validate_element_weights",
    "validate_local_blocks",
]
