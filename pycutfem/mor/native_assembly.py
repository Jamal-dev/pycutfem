"""Native C++ reduced assembly helpers for generated UFL kernels."""

from __future__ import annotations

import sys
from typing import Any, Mapping, Sequence

import numpy as np


def native_kernel_metadata_from_runner(runner: Any) -> Any:
    """Return the native metadata capsule attached to a C++ JIT kernel runner."""

    metadata = getattr(runner, "native_kernel_metadata", None)
    if metadata is not None:
        return metadata
    module = sys.modules.get(getattr(getattr(runner, "kernel", None), "__module__", ""))
    if module is None:
        raise ValueError("runner does not reference an imported C++ kernel module.")
    metadata = getattr(module, "NATIVE_KERNEL_METADATA", None)
    if metadata is None:
        raise ValueError("runner kernel module does not expose NATIVE_KERNEL_METADATA.")
    return metadata


def call_native_kernel(
    *,
    metadata_capsule: Any,
    param_order: Sequence[str],
    static_args: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Call a generated UFL native kernel entrypoint and return local ``K,F,J`` blocks."""

    from .cpp_backend.native_reduced_assembler import module as _native_assembler_module

    K, F, J = _native_assembler_module().call_native_kernel(
        metadata_capsule,
        list(param_order),
        dict(static_args),
    )
    return np.asarray(K, dtype=float), np.asarray(F, dtype=float), np.asarray(J, dtype=float)


def sampled_lspg_rows_from_native_kernel(
    *,
    metadata_capsule: Any,
    param_order: Sequence[str],
    static_args: Mapping[str, Any],
    row_dofs: np.ndarray,
    trial_basis: np.ndarray,
    element_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Call a native UFL kernel and project its local blocks into sampled LSPG rows."""

    from .cpp_backend.native_reduced_assembler import module as _native_assembler_module

    residual, trial = _native_assembler_module().sampled_lspg_rows_from_native_kernel(
        metadata_capsule,
        list(param_order),
        dict(static_args),
        row_dofs,
        trial_basis,
        element_weights,
    )
    return np.asarray(residual, dtype=float).reshape(-1), np.asarray(trial, dtype=float)


def sampled_galerkin_reduced_system_from_native_kernel(
    *,
    metadata_capsule: Any,
    param_order: Sequence[str],
    static_args: Mapping[str, Any],
    trial_basis: np.ndarray,
    element_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Call a native UFL kernel and project its local blocks into a reduced Galerkin system."""

    from .cpp_backend.native_reduced_assembler import module as _native_assembler_module

    residual, tangent = _native_assembler_module().sampled_galerkin_reduced_system_from_native_kernel(
        metadata_capsule,
        list(param_order),
        dict(static_args),
        trial_basis,
        element_weights,
    )
    return np.asarray(residual, dtype=float).reshape(-1), np.asarray(tangent, dtype=float)


def gnat_system_from_native_kernel(
    *,
    metadata_capsule: Any,
    param_order: Sequence[str],
    static_args: Mapping[str, Any],
    row_dofs: np.ndarray,
    trial_basis: np.ndarray,
    sample_to_residual_coefficients: np.ndarray,
    element_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Call a native UFL kernel and build the GNAT residual/Jacobian target."""

    from .sparse import NativeSparseMatrix, apply_sparse_gnat_lift, is_sparse_matrix_like

    if is_sparse_matrix_like(sample_to_residual_coefficients):
        sampled_residual, sampled_trial = sampled_lspg_rows_from_native_kernel(
            metadata_capsule=metadata_capsule,
            param_order=param_order,
            static_args=static_args,
            row_dofs=row_dofs,
            trial_basis=trial_basis,
            element_weights=element_weights,
        )
        return apply_sparse_gnat_lift(
            NativeSparseMatrix.coerce(sample_to_residual_coefficients),
            sampled_residual,
            sampled_trial,
            backend="cpp",
        )

    from .cpp_backend.native_reduced_assembler import module as _native_assembler_module

    residual, trial = _native_assembler_module().gnat_system_from_native_kernel(
        metadata_capsule,
        list(param_order),
        dict(static_args),
        row_dofs,
        trial_basis,
        sample_to_residual_coefficients,
        element_weights,
    )
    return np.asarray(residual, dtype=float).reshape(-1), np.asarray(trial, dtype=float)


__all__ = [
    "call_native_kernel",
    "gnat_system_from_native_kernel",
    "native_kernel_metadata_from_runner",
    "sampled_galerkin_reduced_system_from_native_kernel",
    "sampled_lspg_rows_from_native_kernel",
]
