from __future__ import annotations

import numpy as np
import pytest

from pycutfem.state import QuadratureLayout, StateRegistry


def test_state_registry_commit_and_rollback_preserve_shapes() -> None:
    registry = StateRegistry()
    field = registry.register_cell(
        "velocity_subscale",
        n_cells=3,
        tensor_shape=(2,),
        persistence="step",
    )

    assert field.shape == (3, 2)
    field.assign(np.ones((3, 2), dtype=float))
    field.stage(2.0 * np.ones((3, 2), dtype=float))
    registry.rollback_step()
    assert np.allclose(field.values, 1.0)
    assert np.allclose(field.staged_values, 1.0)

    field.stage(3.0 * np.ones((3, 2), dtype=float))
    registry.commit_step()
    assert np.allclose(field.values, 3.0)
    assert np.allclose(field.staged_values, 3.0)


def test_state_registry_iteration_reset_only_touches_iteration_fields() -> None:
    registry = StateRegistry()
    step_field = registry.register_cell(
        "step_state",
        values=np.arange(4, dtype=float).reshape(4, 1),
        tensor_shape=(),
        persistence="step",
        copy=False,
    )
    iter_field = registry.register_cell(
        "iter_state",
        values=np.arange(8, dtype=float).reshape(4, 2),
        persistence="iteration",
        copy=False,
    )

    iter_field.stage(np.full((4, 2), 7.0, dtype=float))
    step_field.stage(np.full((4,), 9.0, dtype=float))
    registry.reset_iteration()

    assert np.allclose(iter_field.staged_values, iter_field.values)
    assert not np.allclose(step_field.staged_values, step_field.values)


def test_state_registry_rejects_shape_mismatch() -> None:
    registry = StateRegistry()
    field = registry.register_cell("mass_residual", n_cells=2, tensor_shape=())

    with pytest.raises(ValueError, match="expected shape"):
        field.assign(np.zeros((3, 1), dtype=float))


def test_state_registry_normalizes_explicit_scalar_column_vectors() -> None:
    registry = StateRegistry()
    field = registry.register_cell(
        "scalar_column",
        values=np.arange(3, dtype=float).reshape(3, 1),
        tensor_shape=(),
        copy=False,
    )

    assert field.shape == (3,)
    assert field.tensor_shape == ()
    assert np.allclose(field.values, np.array([0.0, 1.0, 2.0]))


def test_quadrature_state_registry_commit_and_layout_validation() -> None:
    registry = StateRegistry()
    layout = QuadratureLayout(
        entity_kind="volume_cell",
        cell_type="tri",
        quadrature_order=2,
        reference_points=np.array(
            [
                [1.0 / 6.0, 1.0 / 6.0],
                [2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0],
            ],
            dtype=float,
        ),
        reference_weights=np.full((3,), 1.0 / 6.0, dtype=float),
    )
    field = registry.register_quadrature(
        "predicted_subscale",
        layout=layout,
        n_entities=2,
        tensor_shape=(2,),
        persistence="step",
    )

    assert field.shape == (2, 3, 2)
    field.assign(np.ones((2, 3, 2), dtype=float))
    field.stage(4.0 * np.ones((2, 3, 2), dtype=float))
    registry.commit_step()
    assert np.allclose(field.values, 4.0)
    assert np.allclose(field.staged_values, 4.0)

    field.stage(np.zeros((2, 3, 2), dtype=float))
    registry.rollback_step()
    assert np.allclose(field.values, 4.0)
    assert np.allclose(field.staged_values, 4.0)
    field.layout.validate_against(
        reference_points=layout.reference_points,
        reference_weights=layout.reference_weights,
        context="state-registry-test",
    )
