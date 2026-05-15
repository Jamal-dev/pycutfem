from __future__ import annotations

import numpy as np
import pytest

from pycutfem.mor import (
    MixedBasisBlock,
    build_block_row_weights,
    build_mixed_field_basis,
    build_mixed_velocity_pressure_basis,
    build_nonaffine_reduced_decomposition,
    compute_supremizer_snapshots,
    fit_lift_enriched_basis,
    fit_supremizer_enriched_velocity_basis,
    orthonormalize_columns,
    remove_lifting_from_snapshots,
    restore_lifting_to_snapshots,
    solve_coupled_lift_snapshots,
)


def test_snapshot_lifting_roundtrip_supports_time_dependent_lifts() -> None:
    snapshots = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=float,
    )
    lifting = np.array(
        [
            [0.5, 1.0, 1.5],
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=float,
    )

    homogeneous = remove_lifting_from_snapshots(snapshots, lifting)
    restored = restore_lifting_to_snapshots(homogeneous, lifting)

    np.testing.assert_allclose(restored, snapshots)
    np.testing.assert_allclose(homogeneous[0], np.array([0.5, 1.0, 1.5]))


def test_supremizer_enrichment_and_mixed_basis_embedding() -> None:
    velocity_basis = np.eye(4, 2)
    pressure_basis = np.eye(3, 2)
    velocity_operator = np.diag([2.0, 3.0, 4.0, 5.0])
    divergence_coupling = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 5.0],
        ],
        dtype=float,
    )

    supremizers = compute_supremizer_snapshots(velocity_operator, divergence_coupling, pressure_basis)
    expected_rhs = divergence_coupling.T @ pressure_basis
    np.testing.assert_allclose(velocity_operator @ supremizers, expected_rhs)

    enriched = fit_supremizer_enriched_velocity_basis(
        velocity_basis,
        pressure_basis,
        supremizers,
        n_supremizer_modes=2,
    )
    assert enriched.enriched_velocity_basis.shape[0] == 4
    assert enriched.enriched_velocity_basis.shape[1] >= velocity_basis.shape[1]

    mixed = build_mixed_velocity_pressure_basis(
        total_dofs=7,
        velocity_rows=np.array([0, 1, 2, 3]),
        velocity_basis=enriched.enriched_velocity_basis,
        pressure_rows=np.array([4, 5, 6]),
        pressure_basis=pressure_basis,
    )
    assert mixed.shape == (7, enriched.enriched_velocity_basis.shape[1] + pressure_basis.shape[1])
    np.testing.assert_allclose(mixed[4:, -pressure_basis.shape[1] :], pressure_basis)


def test_generic_sparse_lift_enrichment_and_three_field_embedding() -> None:
    sp = pytest.importorskip("scipy.sparse")

    primary_operator = sp.diags([2.0, 3.0, 5.0, 7.0], format="csc")
    coupling_operator = sp.csr_matrix(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 7.0],
        ],
        dtype=float,
    )
    coupled_basis = np.eye(3, 2)

    lift_snapshots = solve_coupled_lift_snapshots(primary_operator, coupling_operator, coupled_basis)
    np.testing.assert_allclose(primary_operator @ lift_snapshots, coupling_operator.T @ coupled_basis)

    primary_basis = np.eye(4, 2)
    enrichment = fit_lift_enriched_basis(primary_basis, coupled_basis, lift_snapshots, n_lift_modes=2)
    assert enrichment.enriched_primary_basis.shape[0] == primary_basis.shape[0]
    assert enrichment.enriched_primary_basis.shape[1] >= primary_basis.shape[1]

    scalar_basis = np.array([[1.0], [0.5]])
    multiplier_basis = np.eye(3, 2)
    mixed = build_mixed_field_basis(
        total_dofs=9,
        field_blocks=(
            MixedBasisBlock(rows=np.array([0, 3, 6, 8]), basis=enrichment.enriched_primary_basis, name="primary"),
            {"rows": np.array([1, 4, 7]), "basis": multiplier_basis, "name": "coupled"},
            (np.array([2, 5]), scalar_basis, "scalar"),
        ),
    )
    expected_cols = enrichment.enriched_primary_basis.shape[1] + multiplier_basis.shape[1] + scalar_basis.shape[1]
    assert mixed.shape == (9, expected_cols)
    np.testing.assert_allclose(mixed[np.array([2, 5]), -1:], scalar_basis)


def test_orthonormalize_columns_uses_mass_inner_product_factorization() -> None:
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(6, 4))
    mass = np.diag([1.0, 2.0, 3.0, 5.0, 7.0, 11.0])

    basis = orthonormalize_columns(matrix, inner_product=mass)

    np.testing.assert_allclose(basis.T @ mass @ basis, np.eye(basis.shape[1]), atol=1.0e-12)


def test_nonaffine_qdeim_terms_project_collateral_modes() -> None:
    rng = np.random.default_rng(11)
    trial_basis, _ = np.linalg.qr(rng.normal(size=(8, 3)))
    coeffs = rng.normal(size=(3, 6))
    residual_snapshots = trial_basis @ coeffs + 0.01 * rng.normal(size=(8, 6))

    decomp = build_nonaffine_reduced_decomposition(
        residual_snapshots,
        trial_basis,
        n_modes=3,
        method="qdeim",
    )

    assert decomp.interpolation_rule.rows.shape == (3,)
    assert decomp.residual_terms.shape == (3, 3)
    np.testing.assert_allclose(decomp.residual_terms, decomp.collateral_basis.basis.T @ trial_basis)


def test_weighted_nonaffine_decomposition_reconstructs_unweighted_reduced_target() -> None:
    rng = np.random.default_rng(123)
    trial_basis, _ = np.linalg.qr(rng.normal(size=(7, 3)))
    collateral, _ = np.linalg.qr(rng.normal(size=(7, 3)))
    coeffs = rng.normal(size=(3, 8))
    residual_snapshots = collateral @ coeffs
    weights = np.array([1.0, 8.0, 2.0, 4.0, 1.5, 3.0, 6.0])

    decomp = build_nonaffine_reduced_decomposition(
        residual_snapshots,
        trial_basis,
        n_modes=3,
        method="qdeim",
        row_weights=weights,
    )

    sample = residual_snapshots[:, 2]
    selected_values = np.sqrt(decomp.sampled_row_weights) * sample[decomp.interpolation_rule.rows]
    alpha = np.linalg.solve(decomp.interpolation_rule.selected_basis, selected_values)
    reduced = decomp.residual_terms.T @ alpha

    np.testing.assert_allclose(reduced, trial_basis.T @ sample, atol=1.0e-11)


def test_block_row_weights_balance_residual_blocks() -> None:
    reference = np.zeros((6, 4), dtype=float)
    reference[:3, :] = 10.0
    reference[3:, :] = 0.5

    weights = build_block_row_weights(
        reference,
        (
            {"rows": np.array([0, 1, 2]), "name": "large"},
            {"rows": np.array([3, 4, 5]), "name": "small"},
        ),
    )

    assert np.allclose(weights[:3], 1.0)
    assert np.all(weights[3:] > weights[:3].max())
    scaled = np.sqrt(weights)[:, None] * reference
    np.testing.assert_allclose(np.sqrt(np.mean(scaled[:3] ** 2)), np.sqrt(np.mean(scaled[3:] ** 2)))
