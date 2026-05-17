from __future__ import annotations

import numpy as np
import pytest

from pycutfem.mor import (
    BoundConstraintSpec,
    NativeSparseMatrix,
    bound_constraints_from_fields,
    project_reduced_coefficients_to_bounds,
)


class _FakeDofHandler:
    field_names = ("alpha", "phi", "u")

    def __init__(self) -> None:
        self._rows = {
            "alpha": [0, 3, 6],
            "phi": [1, 4],
            "u": [2, 5, 7],
        }

    def get_field_slice(self, field: str):
        return list(self._rows[field])


def test_bound_constraint_spec_reduces_decoded_full_state_bounds() -> None:
    trial_basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, -0.25],
            [-1.0, 2.0],
        ],
        dtype=float,
    )
    offset = np.array([0.25, 0.5, -1.0, 0.75], dtype=float)
    spec = BoundConstraintSpec(
        rows=np.array([0, 3]),
        lower=np.array([0.0, -0.5]),
        upper=np.array([1.0, 0.9]),
        row_scaling=np.array([2.0, 0.5]),
        metadata={"fields": ("alpha",)},
    )

    reduced = spec.reduce(trial_basis=trial_basis, offset=offset)
    q = np.array([0.2, -0.1], dtype=float)

    np.testing.assert_allclose(reduced.decoded_values(q), offset[[0, 3]] + trial_basis[[0, 3], :] @ q)
    assert reduced.max_violation(q) == 0.0
    equations = reduced.active_equations(np.array([-0.25, -0.1]), active_tol=1.0e-12)
    assert equations.constraint_matrix.shape == (1, 2)
    np.testing.assert_allclose(equations.rows, np.array([0]))
    np.testing.assert_allclose(equations.rhs, np.array([0.0]))
    np.testing.assert_allclose(equations.constraint_matrix, np.array([[2.0, 0.0]]))


def test_bound_constraints_from_fields_builds_field_metadata() -> None:
    dh = _FakeDofHandler()

    spec = bound_constraints_from_fields(
        dh,
        {
            "alpha": (0.0, 1.0),
            "phi": (0.1, 0.9),
        },
        row_scaling={"alpha": 2.0, "phi": 3.0},
    )

    np.testing.assert_array_equal(spec.rows, np.array([0, 3, 6, 1, 4]))
    np.testing.assert_allclose(spec.lower, np.array([0.0, 0.0, 0.0, 0.1, 0.1]))
    np.testing.assert_allclose(spec.upper, np.array([1.0, 1.0, 1.0, 0.9, 0.9]))
    np.testing.assert_allclose(spec.row_scaling, np.array([2.0, 2.0, 2.0, 3.0, 3.0]))
    assert spec.metadata["fields"] == ("alpha", "phi")


def test_reduced_bound_constraints_report_activity_and_violations() -> None:
    spec = BoundConstraintSpec(
        rows=np.array([0, 1, 2]),
        lower=np.array([0.0, -np.inf, -1.0]),
        upper=np.array([1.0, 0.5, np.inf]),
    )
    reduced = spec.reduce(trial_basis=np.eye(3), offset=np.zeros(3))
    q = np.array([-0.2, 0.6, -1.0], dtype=float)

    activity = reduced.activity(q, active_tol=1.0e-12)

    np.testing.assert_array_equal(activity.lower_active, np.array([0, 2]))
    np.testing.assert_array_equal(activity.upper_active, np.array([1]))
    np.testing.assert_allclose(activity.lower_violation, np.array([0.2, 0.0, 0.0]))
    np.testing.assert_allclose(activity.upper_violation, np.array([0.0, 0.1, 0.0]))
    assert activity.max_violation == pytest.approx(0.2)


def test_bound_constraint_reduction_can_keep_sparse_rows() -> None:
    trial_basis = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.5, 0.0, 0.25],
        ],
        dtype=float,
    )
    spec = BoundConstraintSpec(rows=np.array([0, 2, 3]), lower=-1.0, upper=1.0, row_scaling=2.0)

    reduced = spec.reduce(trial_basis=trial_basis, offset=np.zeros(4), sparse=True)

    assert isinstance(reduced.constraint_matrix, NativeSparseMatrix)
    assert reduced.constraint_matrix.nnz == 4
    q = np.array([0.25, 0.0, -1.0], dtype=float)
    np.testing.assert_allclose(reduced.decoded_values(q), trial_basis[[0, 2, 3], :] @ q)
    equations = reduced.active_equations(np.array([2.0, 0.0, -0.5]), active_tol=1.0e-12)
    np.testing.assert_allclose(equations.constraint_matrix, np.array([[2.0, 0.0, 0.0]], dtype=float))
    assert reduced.to_native_dict()["constraint_matrix"]["layout"] == "csr"


def test_project_reduced_coefficients_to_bounds_repairs_decoded_violations() -> None:
    trial_basis = np.array(
        [
            [1.0, 0.0],
            [0.5, 1.0],
            [-1.0, 0.25],
        ],
        dtype=float,
    )
    spec = BoundConstraintSpec(rows=np.array([0, 1, 2]), lower=0.0, upper=1.0)
    reduced = spec.reduce(trial_basis=trial_basis, offset=np.zeros(3))
    q0 = np.array([-0.25, 1.5], dtype=float)

    projected = project_reduced_coefficients_to_bounds(reduced, q0, tolerance=1.0e-12)

    assert reduced.max_violation(projected) <= 1.0e-12
    assert np.linalg.norm(projected - q0) > 0.0


def test_bound_constraint_validation_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        BoundConstraintSpec(rows=np.array([0, 0]), lower=0.0, upper=1.0)
    with pytest.raises(ValueError, match="less than"):
        BoundConstraintSpec(rows=np.array([0]), lower=2.0, upper=1.0)
    with pytest.raises(ValueError, match="outside"):
        BoundConstraintSpec(rows=np.array([5]), lower=0.0, upper=1.0).reduce(
            trial_basis=np.eye(2),
            offset=np.zeros(2),
        )
