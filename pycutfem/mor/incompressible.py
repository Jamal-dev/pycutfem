"""Compatibility exports for incompressible mixed ROM helpers."""

from __future__ import annotations

from .mixed_reduction import (
    NonAffineReducedDecomposition,
    SupremizerEnrichment,
    build_dirichlet_lifting_vector,
    build_mixed_velocity_pressure_basis,
    build_nonaffine_reduced_decomposition,
    compute_supremizer_snapshots,
    field_dof_indices,
    fit_supremizer_enriched_velocity_basis,
    orthonormalize_columns,
    remove_lifting_from_snapshots,
    restore_lifting_to_snapshots,
)

__all__ = [
    "NonAffineReducedDecomposition",
    "SupremizerEnrichment",
    "build_dirichlet_lifting_vector",
    "build_mixed_velocity_pressure_basis",
    "build_nonaffine_reduced_decomposition",
    "compute_supremizer_snapshots",
    "field_dof_indices",
    "fit_supremizer_enriched_velocity_basis",
    "orthonormalize_columns",
    "remove_lifting_from_snapshots",
    "restore_lifting_to_snapshots",
]
