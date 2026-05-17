"""Native C++ reduced assembly helpers for generated UFL kernels."""

from __future__ import annotations

import sys
from typing import Any, Mapping, Sequence

import numpy as np


def apply_affine_updates_to_static_args(
    static_args: Mapping[str, Any],
    updates: Sequence[Mapping[str, Any]] | None,
    coefficients: Any,
) -> tuple[str, ...]:
    """Evaluate affine state updates and copy them into native static args.

    This is intended for offline/checkpoint reconstruction around native online
    solves.  The C++ online loop owns these updates during Newton iterations;
    this helper mirrors the accepted-state update once a trajectory checkpoint
    needs to be persisted.
    """

    if not updates:
        return ()
    from .state_updates import apply_affine_state_updates

    args = dict(static_args)
    values = apply_affine_state_updates(updates, coefficients)
    changed: list[str] = []
    for name, flat in values.items():
        target = args.get(name)
        if not isinstance(target, np.ndarray):
            continue
        arr = np.asarray(flat, dtype=float).reshape(target.shape)
        np.copyto(target, arr, casting="unsafe")
        changed.append(str(name))
    return tuple(changed)


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


def _sampled_rows_from_local_pair_online_convention(
    *,
    K_elem: np.ndarray,
    residual_elem: np.ndarray,
    gdofs_map: np.ndarray,
    row_dofs: np.ndarray,
    trial_basis: np.ndarray,
    element_weights: np.ndarray | None = None,
    row_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(K_elem, dtype=float)
    F = np.asarray(residual_elem, dtype=float)
    gdofs = np.asarray(gdofs_map, dtype=np.int64)
    basis = np.asarray(trial_basis, dtype=float)
    rows = np.asarray(row_dofs, dtype=np.int64).reshape(-1)
    if K.ndim != 3 or F.ndim != 2 or gdofs.ndim != 2:
        raise ValueError("local target blocks must have K=(e,l,l), F=(e,l), gdofs=(e,l).")
    if K.shape[0] != F.shape[0] or K.shape[:2] != gdofs.shape or F.shape != gdofs.shape:
        raise ValueError("local target blocks have incompatible shapes.")
    if basis.ndim != 2:
        raise ValueError("trial_basis must be rank-2.")
    if np.any(rows < 0) or (rows.size and int(np.max(rows)) >= basis.shape[0]):
        raise ValueError("row_dofs contains rows outside trial_basis.")
    if np.unique(rows).size != rows.size:
        raise ValueError("row_dofs must be unique.")
    weights = np.ones(int(K.shape[0]), dtype=float)
    if element_weights is not None:
        weights = np.asarray(element_weights, dtype=float).reshape(-1)
        if weights.size != K.shape[0]:
            raise ValueError("element_weights must match the local element count.")
    row_scale = np.ones(int(rows.size), dtype=float)
    if row_weights is not None:
        raw = np.asarray(row_weights, dtype=float).reshape(-1)
        if raw.size != rows.size:
            raise ValueError("row_weights must match row_dofs.")
        if np.any(raw < 0.0):
            raise ValueError("row_weights must be nonnegative.")
        row_scale = np.sqrt(raw)

    row_positions = np.full(int(basis.shape[0]), -1, dtype=np.int64)
    row_positions[rows] = np.arange(int(rows.size), dtype=np.int64)
    safe_gdofs = np.where(gdofs >= 0, gdofs, 0)
    local_row_positions = np.where(gdofs >= 0, row_positions[safe_gdofs], -1)
    elem_idx, local_i = np.nonzero(local_row_positions >= 0)
    residual = np.zeros(int(rows.size), dtype=float)
    jacobian = np.zeros((int(rows.size), int(basis.shape[1])), dtype=float)
    if elem_idx.size:
        sample_idx = local_row_positions[elem_idx, local_i]
        weighted_residual = weights[elem_idx] * F[elem_idx, local_i] * row_scale[sample_idx]
        np.add.at(residual, sample_idx, weighted_residual)
        safe_cols = np.where(gdofs[elem_idx, :] >= 0, gdofs[elem_idx, :], 0)
        local_basis = basis[safe_cols, :]
        local_basis = np.where((gdofs[elem_idx, :] >= 0)[:, :, None], local_basis, 0.0)
        weighted_k_rows = (weights[elem_idx] * row_scale[sample_idx])[:, None] * K[elem_idx, local_i, :]
        trial_contrib = np.einsum("sl,slm->sm", weighted_k_rows, local_basis, optimize=True)
        np.add.at(jacobian, sample_idx, trial_contrib)
    return np.ascontiguousarray(residual, dtype=np.float64), np.ascontiguousarray(jacobian, dtype=np.float64)


def reduced_target_from_native_kernel_pair(
    *,
    residual_metadata_capsule: Any,
    residual_param_order: Sequence[str],
    residual_static_args: Mapping[str, Any],
    tangent_metadata_capsule: Any,
    tangent_param_order: Sequence[str],
    tangent_static_args: Mapping[str, Any],
    trial_basis: np.ndarray,
    row_dofs: np.ndarray,
    target: str = "sampled_lspg",
    selected_basis: np.ndarray | None = None,
    residual_terms: np.ndarray | None = None,
    element_weights: np.ndarray | None = None,
    row_weights: np.ndarray | None = None,
    gnat_lift: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the final reduced target from separate residual/tangent kernels."""

    _Kr, F_elem, _Jr = call_native_kernel(
        metadata_capsule=residual_metadata_capsule,
        param_order=residual_param_order,
        static_args=residual_static_args,
    )
    K_elem, _Ft, _Jt = call_native_kernel(
        metadata_capsule=tangent_metadata_capsule,
        param_order=tangent_param_order,
        static_args=tangent_static_args,
    )
    gdofs_r = np.asarray(residual_static_args["gdofs_map"], dtype=np.int64)
    gdofs_t = np.asarray(tangent_static_args["gdofs_map"], dtype=np.int64)
    if gdofs_r.shape != gdofs_t.shape or not np.array_equal(gdofs_r, gdofs_t):
        raise ValueError("residual and tangent native kernels must use the same gdofs_map for checkpoint assembly.")
    sampled_residual, sampled_jacobian = _sampled_rows_from_local_pair_online_convention(
        K_elem=K_elem,
        residual_elem=F_elem,
        gdofs_map=gdofs_t,
        row_dofs=row_dofs,
        trial_basis=trial_basis,
        element_weights=element_weights,
        row_weights=row_weights,
    )
    target_name = str(target).strip().lower()
    if target_name in {"sampled_lspg", "full_row_lspg", "lspg"}:
        residual, jacobian = sampled_residual, sampled_jacobian
    elif target_name in {"true_galerkin", "galerkin", "full_galerkin"}:
        rows = np.asarray(row_dofs, dtype=np.int64).reshape(-1)
        test_basis = np.asarray(trial_basis, dtype=float)[rows, :]
        if row_weights is not None:
            test_basis = np.sqrt(np.asarray(row_weights, dtype=float).reshape(-1))[:, None] * test_basis
        residual = test_basis.T @ sampled_residual
        jacobian = test_basis.T @ sampled_jacobian
    elif target_name in {"sampled_gnat", "gnat", "qdeim", "deim"}:
        if selected_basis is None or residual_terms is None:
            raise ValueError("selected_basis and residual_terms are required for sampled GNAT/DEIM targets.")
        selected = np.asarray(selected_basis, dtype=float)
        terms = np.asarray(residual_terms, dtype=float)
        if selected.ndim != 2 or terms.ndim != 2:
            raise ValueError("selected_basis and residual_terms must be rank-2.")
        if selected.shape[0] != sampled_residual.size or terms.shape[0] != selected.shape[1]:
            raise ValueError("selected_basis/residual_terms shapes are incompatible with sampled target rows.")
        if selected.shape[0] == selected.shape[1]:
            try:
                interp = np.linalg.solve(selected, np.eye(selected.shape[0], dtype=float))
            except np.linalg.LinAlgError:
                interp = np.linalg.pinv(selected, rcond=1.0e-12)
        else:
            interp = np.linalg.pinv(selected, rcond=1.0e-12)
        coeffs = interp @ sampled_residual
        coeff_jac = interp @ sampled_jacobian
        residual = terms.T @ coeffs
        jacobian = terms.T @ coeff_jac
    else:
        raise ValueError(f"unsupported native reduced target {target!r}.")

    if gnat_lift is not None:
        from .sparse import NativeSparseMatrix, apply_sparse_gnat_lift, is_sparse_matrix_like

        if is_sparse_matrix_like(gnat_lift):
            residual, jacobian = apply_sparse_gnat_lift(NativeSparseMatrix.coerce(gnat_lift), residual, jacobian, backend="cpp")
        else:
            lift = np.asarray(gnat_lift, dtype=float)
            residual = lift @ residual
            jacobian = lift @ jacobian
    return np.ascontiguousarray(residual, dtype=np.float64), np.ascontiguousarray(jacobian, dtype=np.float64)


__all__ = [
    "apply_affine_updates_to_static_args",
    "call_native_kernel",
    "gnat_system_from_native_kernel",
    "native_kernel_metadata_from_runner",
    "reduced_target_from_native_kernel_pair",
    "sampled_galerkin_reduced_system_from_native_kernel",
    "sampled_lspg_rows_from_native_kernel",
]
