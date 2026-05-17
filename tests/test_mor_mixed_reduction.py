from __future__ import annotations

import numpy as np
import pytest

from pycutfem.mor import (
    MixedBasisBlock,
    build_block_row_weights,
    build_mixed_field_basis,
    build_mixed_velocity_pressure_basis,
    build_nonaffine_reduced_decomposition,
    certify_mixed_stability_basis,
    compute_supremizer_snapshots,
    gauge_correct_snapshots,
    fit_fieldwise_pod_basis,
    fit_lift_enriched_basis,
    fit_supremizer_enriched_velocity_basis,
    orthonormalize_columns,
    pressure_gauge_history,
    reduced_coupling_rank_certificate,
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


def test_fieldwise_pod_basis_preserves_small_magnitude_blocks() -> None:
    t = np.linspace(0.0, 1.0, 5)
    large = np.vstack([1000.0 * t, 500.0 * t * t])
    small = np.vstack([1.0e-3 * np.sin(t), 2.0e-3 * np.cos(t), 3.0e-3 * t])
    snapshots = np.vstack([large, small])

    fitted = fit_fieldwise_pod_basis(
        snapshots,
        (
            {"rows": np.array([0, 1]), "name": "pressure_like"},
            {"rows": np.array([2, 3, 4]), "name": "scalar_like"},
        ),
        n_modes_per_block={"pressure_like": 2, "scalar_like": 2},
    )

    assert fitted.basis.shape == (5, 4)
    assert fitted.metadata["total_modes"] == 4
    np.testing.assert_allclose(fitted.basis[:2, 2:], 0.0)
    np.testing.assert_allclose(fitted.basis[2:, :2], 0.0)
    assert fitted.singular_values["scalar_like"].size >= 2
    coeffs, *_ = np.linalg.lstsq(fitted.basis, snapshots - fitted.offset[:, None], rcond=None)
    reconstructed = fitted.offset[:, None] + fitted.basis @ coeffs
    assert np.linalg.norm(reconstructed[2:] - snapshots[2:]) < 2.0e-5


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


def test_pressure_gauge_and_mixed_stability_certificate() -> None:
    t = np.linspace(0.0, 1.0, 6)
    velocity = np.vstack([t, t * t])
    pressure = np.vstack([1.0 + t, 2.0 + t, 3.0 + t])
    snapshots = np.vstack([velocity, pressure])
    pressure_rows = np.array([2, 3, 4], dtype=np.int64)
    gauge = {"rows": pressure_rows, "name": "pressure"}

    corrected = gauge_correct_snapshots(snapshots, [gauge])
    np.testing.assert_allclose(pressure_gauge_history(corrected.corrected_snapshots, gauge), 0.0, atol=1.0e-14)
    assert corrected.gauge_histories["pressure"].shape == (snapshots.shape[1],)

    fitted = fit_fieldwise_pod_basis(
        corrected.corrected_snapshots,
        (
            {"rows": np.array([0, 1], dtype=np.int64), "name": "velocity"},
            {"rows": pressure_rows, "name": "pressure"},
        ),
        n_modes_per_block={"velocity": 2, "pressure": 2},
        center=True,
    )
    coupling = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    rank_cert = reduced_coupling_rank_certificate(
        coupling,
        fitted.basis[np.array([0, 1]), :2],
        fitted.basis[pressure_rows, 2:],
        name="divergence_pressure",
        required_rank=2,
    )
    assert rank_cert.passed

    cert = certify_mixed_stability_basis(
        snapshots,
        fitted.basis,
        offset=fitted.offset,
        row_blocks=(
            {"rows": np.array([0, 1], dtype=np.int64), "name": "velocity"},
            {"rows": pressure_rows, "name": "pressure"},
        ),
        pressure_gauge_blocks=[gauge],
        projection_tolerance=1.0e-10,
        coupling_certificates=[rank_cert],
    )
    assert cert.passed
    assert cert.gauge_max_abs["pressure"] < 1.0e-12
    assert cert.field_errors["pressure"].max_relative_error < 1.0e-10
