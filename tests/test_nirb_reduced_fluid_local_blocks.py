import numpy as np
import pytest

from examples.NIRB.reduced_fluid import (
    constrained_reaction_rows_from_local_blocks,
    reduced_reaction_from_local_blocks,
    sampled_galerkin_element_contributions_from_local_blocks,
    sampled_galerkin_reduced_system_from_local_blocks,
    sampled_lspg_element_contributions_from_local_blocks,
    sampled_lspg_rows_from_local_blocks,
)


def _scatter_local_blocks(
    K_elem: np.ndarray,
    vector_elem: np.ndarray,
    gdofs_map: np.ndarray,
    total_dofs: int,
) -> tuple[np.ndarray, np.ndarray]:
    K_full = np.zeros((total_dofs, total_dofs), dtype=float)
    vector_full = np.zeros(total_dofs, dtype=float)
    for elem, gdofs in enumerate(gdofs_map):
        vector_full[gdofs] += vector_elem[elem]
        for local_i, global_i in enumerate(gdofs):
            K_full[global_i, gdofs] += K_elem[elem, local_i, :]
    return K_full, vector_full


def test_sampled_lspg_rows_match_full_scatter_with_element_weights() -> None:
    K_elem = np.array(
        [
            [[2.0, -1.0, 0.5], [0.0, 3.0, -0.25], [1.5, 0.0, 1.0]],
            [[-1.0, 0.2, 0.0], [0.75, 1.25, -0.5], [0.0, -2.0, 4.0]],
        ]
    )
    raw_rhs_elem = np.array([[1.0, -2.0, 0.5], [3.0, -1.0, 2.0]])
    gdofs_map = np.array([[0, 2, 4], [2, 3, 5]], dtype=int)
    basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.5, -1.0],
            [1.5, 0.25],
            [-0.5, 2.0],
            [0.25, -0.75],
        ]
    )
    rows = np.array([2, 4, 5], dtype=int)
    weights = np.array([0.25, 2.0])

    residual, trial = sampled_lspg_rows_from_local_blocks(
        K_elem=K_elem,
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        row_dofs=rows,
        trial_basis=basis,
        element_weights=weights,
    )

    weighted_K = K_elem * weights[:, None, None]
    weighted_rhs = raw_rhs_elem * weights[:, None]
    K_full, raw_rhs_full = _scatter_local_blocks(weighted_K, weighted_rhs, gdofs_map, total_dofs=basis.shape[0])
    np.testing.assert_allclose(residual, -raw_rhs_full[rows])
    np.testing.assert_allclose(trial, K_full[rows, :] @ basis)


def test_sampled_lspg_element_contributions_keep_element_axis() -> None:
    K_elem = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[-1.0, 0.5], [2.5, -0.25]],
        ]
    )
    raw_rhs_elem = np.array([[5.0, 6.0], [-2.0, 7.0]])
    gdofs_map = np.array([[0, 1], [1, 2]], dtype=int)
    basis = np.array([[1.0], [2.0], [-1.0]])
    rows = np.array([1], dtype=int)

    residual_by_element, trial_by_element = sampled_lspg_element_contributions_from_local_blocks(
        K_elem=K_elem,
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        row_dofs=rows,
        trial_basis=basis,
    )

    np.testing.assert_allclose(residual_by_element[:, 0], [-6.0, 2.0])
    np.testing.assert_allclose(trial_by_element[:, 0, 0], [11.0, -2.5])


def test_sampled_galerkin_reduced_system_matches_full_scatter() -> None:
    K_elem = np.array(
        [
            [[4.0, 1.0], [2.0, 3.0]],
            [[1.0, -1.0], [0.5, 2.0]],
        ]
    )
    residual_elem = np.array([[1.5, -0.5], [2.0, 3.0]])
    gdofs_map = np.array([[0, 2], [1, 2]], dtype=int)
    basis = np.array([[1.0, 0.25], [-0.5, 2.0], [0.75, -1.0]])
    weights = np.array([1.5, 0.25])

    reduced_residual, reduced_tangent = sampled_galerkin_reduced_system_from_local_blocks(
        K_elem=K_elem,
        residual_elem=residual_elem,
        gdofs_map=gdofs_map,
        trial_basis=basis,
        element_weights=weights,
    )

    weighted_K = K_elem * weights[:, None, None]
    weighted_residual = residual_elem * weights[:, None]
    K_full, residual_full = _scatter_local_blocks(weighted_K, weighted_residual, gdofs_map, total_dofs=basis.shape[0])
    np.testing.assert_allclose(reduced_residual, basis.T @ residual_full)
    np.testing.assert_allclose(reduced_tangent, basis.T @ K_full @ basis)


def test_sampled_galerkin_element_contributions_keep_element_axis() -> None:
    K_elem = np.array([[[2.0, -1.0], [0.5, 3.0]]])
    residual_elem = np.array([[4.0, -2.0]])
    gdofs_map = np.array([[0, 1]], dtype=int)
    basis = np.array([[1.0, 2.0], [0.5, -1.0]])

    residual_by_element, tangent_by_element = sampled_galerkin_element_contributions_from_local_blocks(
        K_elem=K_elem,
        residual_elem=residual_elem,
        gdofs_map=gdofs_map,
        trial_basis=basis,
    )

    local_basis = basis[gdofs_map[0], :]
    np.testing.assert_allclose(residual_by_element[0], local_basis.T @ residual_elem[0])
    np.testing.assert_allclose(tangent_by_element[0], local_basis.T @ K_elem[0] @ local_basis)


def test_constrained_reaction_rows_match_full_scatter_without_global_vector() -> None:
    raw_rhs_elem = np.array([[1.0, -2.0, 0.5], [3.0, -1.0, 2.0]])
    gdofs_map = np.array([[0, 2, 4], [2, 3, 5]], dtype=int)
    constrained_rows = np.array([2, 4, 5], dtype=int)
    weights = np.array([0.25, 2.0])

    rows, reaction = constrained_reaction_rows_from_local_blocks(
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        constrained_row_dofs=constrained_rows,
        element_weights=weights,
    )

    _K_full, rhs_full = _scatter_local_blocks(
        np.zeros((2, 3, 3), dtype=float),
        raw_rhs_elem * weights[:, None],
        gdofs_map,
        total_dofs=6,
    )
    np.testing.assert_array_equal(rows, constrained_rows)
    np.testing.assert_allclose(reaction, -rhs_full[constrained_rows])


def test_reduced_reaction_from_local_blocks_applies_reduced_load_map() -> None:
    raw_rhs_elem = np.array([[1.0, -2.0], [0.5, 4.0]])
    gdofs_map = np.array([[0, 3], [3, 5]], dtype=int)
    constrained_rows = np.array([3, 5], dtype=int)
    row_to_load = np.array([[2.0, -1.0], [0.5, 0.25]])

    reduced = reduced_reaction_from_local_blocks(
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        constrained_row_dofs=constrained_rows,
        row_to_reduced_load=row_to_load,
    )

    _rows, reaction = constrained_reaction_rows_from_local_blocks(
        raw_rhs_elem=raw_rhs_elem,
        gdofs_map=gdofs_map,
        constrained_row_dofs=constrained_rows,
    )
    np.testing.assert_allclose(reduced, row_to_load @ reaction)


def test_sampled_local_block_helpers_reject_invalid_weights_and_duplicate_rows() -> None:
    K_elem = np.eye(2, dtype=float).reshape(1, 2, 2)
    raw_rhs_elem = np.ones((1, 2), dtype=float)
    gdofs_map = np.array([[0, 1]], dtype=int)
    basis = np.eye(2, dtype=float)

    with pytest.raises(ValueError, match="row_dofs must be unique"):
        sampled_lspg_rows_from_local_blocks(
            K_elem=K_elem,
            raw_rhs_elem=raw_rhs_elem,
            gdofs_map=gdofs_map,
            row_dofs=np.array([0, 0], dtype=int),
            trial_basis=basis,
        )

    with pytest.raises(ValueError, match="element_weights"):
        sampled_galerkin_reduced_system_from_local_blocks(
            K_elem=K_elem,
            residual_elem=raw_rhs_elem,
            gdofs_map=gdofs_map,
            trial_basis=basis,
            element_weights=np.array([-1.0]),
        )

    with pytest.raises(ValueError, match="constrained_row_dofs must be unique"):
        constrained_reaction_rows_from_local_blocks(
            raw_rhs_elem=raw_rhs_elem,
            gdofs_map=gdofs_map,
            constrained_row_dofs=np.array([0, 0], dtype=int),
        )
