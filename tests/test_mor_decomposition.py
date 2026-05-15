from __future__ import annotations

import numpy as np
import pytest

from pycutfem.mor import (
    CollateralBasis,
    EmpiricalCubatureRule,
    InterpolationRule,
    NativeReducedEvaluationGraph,
    ReducedOperatorTerm,
    build_deim_interpolation_rule,
    build_qdeim_interpolation_rule,
    compose_reduced_operator,
    fit_collateral_basis,
    interpolation_coefficients,
    reconstruct_from_interpolation,
    select_deim_rows,
    select_qdeim_rows,
)


def _have_cpp_backend() -> bool:
    try:
        import pybind11  # noqa: F401

        return True
    except Exception:
        return False


def _basis() -> np.ndarray:
    q, _ = np.linalg.qr(
        np.array(
            [
                [1.0, 0.2, -0.3],
                [0.4, 1.2, 0.5],
                [-0.8, 0.1, 1.1],
                [1.5, -0.7, 0.2],
                [0.3, 0.9, -1.0],
            ],
            dtype=float,
        )
    )
    return q[:, :3]


def test_fit_collateral_basis_and_deim_reconstruct_span_vector() -> None:
    basis = _basis()
    coeffs = np.array([1.25, -0.5, 0.75], dtype=float)
    snapshots = basis @ np.array(
        [
            [1.0, 0.0, 2.0, -1.0],
            [0.0, 1.5, -0.5, 0.2],
            [2.0, -1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    collateral = fit_collateral_basis(snapshots, n_modes=3)
    rule = build_deim_interpolation_rule(collateral)
    values = collateral.basis @ coeffs
    selected_values = values[rule.rows]

    reconstructed = reconstruct_from_interpolation(rule, selected_values)

    np.testing.assert_allclose(reconstructed, values, rtol=1.0e-12, atol=1.0e-12)
    assert rule.method == "deim"
    assert rule.rows.size == collateral.n_modes
    assert np.unique(rule.rows).size == rule.rows.size


def test_deim_and_qdeim_row_selection_are_valid() -> None:
    basis = _basis()
    deim_rows = select_deim_rows(basis)
    qdeim_rows = select_qdeim_rows(basis)

    for rows in (deim_rows, qdeim_rows):
        assert rows.shape == (basis.shape[1],)
        assert np.unique(rows).size == rows.size
        assert np.all(rows >= 0)
        assert np.all(rows < basis.shape[0])
        assert np.linalg.matrix_rank(basis[rows, :]) == basis.shape[1]


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_cpp_interpolation_and_composition_match_numpy(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "deim_online"))
    basis = _basis()
    rule = build_qdeim_interpolation_rule(CollateralBasis(basis=basis))
    true_coeffs = np.array([0.5, -1.25, 2.0], dtype=float)
    selected_values = (basis @ true_coeffs)[rule.rows]

    coeffs_py = interpolation_coefficients(rule, selected_values, backend="python")
    coeffs_cpp = interpolation_coefficients(rule, selected_values, backend="cpp")

    np.testing.assert_allclose(coeffs_cpp, coeffs_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(coeffs_cpp, true_coeffs, rtol=1.0e-12, atol=1.0e-12)

    residual_terms = np.array([[1.0, 0.0], [-0.5, 2.0], [3.0, 1.0]], dtype=float)
    jacobian_terms = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, -1.0], [2.0, 0.25]],
            [[-0.25, 0.75], [1.5, -0.5]],
        ],
        dtype=float,
    )
    out_py = compose_reduced_operator(coeffs_cpp, residual_terms, jacobian_terms, backend="python")
    out_cpp = compose_reduced_operator(coeffs_cpp, residual_terms, jacobian_terms, backend="cpp")

    np.testing.assert_allclose(out_cpp["residual"], out_py["residual"], rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(out_cpp["jacobian"], out_py["jacobian"], rtol=1.0e-13, atol=1.0e-13)


def test_reduced_evaluation_graph_serializes_problem_generic_metadata() -> None:
    basis = _basis()
    rule = InterpolationRule(method="custom", rows=np.array([0, 2, 3]), basis=basis)
    cubature = EmpiricalCubatureRule(entity_ids=np.array([1, 4]), weights=np.array([0.25, 1.75]))
    term = ReducedOperatorTerm(
        term_id="nonlinear_flux",
        residual_block=np.array([1.0, -2.0]),
        jacobian_block=np.eye(2),
        role="residual+tangent",
    )
    graph = NativeReducedEvaluationGraph(
        interpolation_rule=rule,
        cubature_rule=cubature,
        operator_terms=(term,),
        metadata={"system": "generic"},
    )

    payload = graph.to_native_dict()

    assert payload["interpolation_rule"]["method"] == "custom"
    assert payload["cubature_rule"]["entity_kind"] == "cell"
    assert payload["operator_term_ids"] == ("nonlinear_flux",)
    assert payload["metadata"]["system"] == "generic"
