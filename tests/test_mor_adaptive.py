from __future__ import annotations

import numpy as np

from pycutfem.mor import (
    augment_rows_from_dwr_localization,
    dual_weighted_residual_estimate,
    select_certified_adaptive_enrichment_actions,
)


def test_certified_adaptive_decision_selects_failed_gates() -> None:
    estimate = dual_weighted_residual_estimate(
        [np.array([0.1, 2.0], dtype=float)],
        [np.array([1.0, 0.5], dtype=float)],
    )
    decision = select_certified_adaptive_enrichment_actions(
        field_errors={"pressure": {"max_relative_error": 0.25, "passed": False}},
        dwr_estimate=estimate,
        norm_equivalence_certificate={"passed": False, "lower_constant": 1.0e-5},
        branch_certificate={"passed": False, "max_branch_distance": 0.4},
        projection_tolerance=0.05,
    )

    assert not decision.accepted
    kinds = {action.kind for action in decision.actions}
    assert {"primal_basis", "adjoint_basis", "gnat_rows", "branch_reference"}.issubset(kinds)
    assert decision.actions[0].priority >= decision.actions[-1].priority


def test_dwr_localization_row_augmentation_keeps_mandatory_rows() -> None:
    residuals = np.array([[0.1, 10.0, 0.2, 5.0]], dtype=float)
    adjoints = np.array([[1.0, 0.5, 1.0, 2.0]], dtype=float)

    rows = augment_rows_from_dwr_localization(
        residuals,
        adjoints,
        row_dofs=np.array([0], dtype=np.int64),
        mandatory_rows=np.array([2], dtype=np.int64),
        max_new_rows=1,
    )

    assert {0, 2}.issubset(set(int(row) for row in rows))
    assert 3 in set(int(row) for row in rows)
