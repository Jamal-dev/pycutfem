from __future__ import annotations

import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.state import (
    build_gaussian_nonlocal_quadrature_map,
    build_volume_nonlocal_quadrature_map,
)

from examples.utils.poromechanics.kratos_parity import build_kratos_consolidation_2d_mesh


def test_gaussian_nonlocal_quadrature_map_matches_manual_kernel() -> None:
    points = np.asarray(
        [
            [[0.0, 0.0], [0.25, 0.0]],
            [[1.0, 0.0], [1.5, 0.0]],
        ],
        dtype=float,
    )
    weights = np.asarray([[2.0, 3.0], [5.0, 7.0]], dtype=float)
    length = 0.6
    qmap = build_gaussian_nonlocal_quadrature_map(
        points,
        weights,
        characteristic_length=length,
    )
    values = np.asarray([[1.0, 2.0], [10.0, 20.0]], dtype=float)

    flat_points = points.reshape(-1, 2)
    flat_weights = weights.reshape(-1)
    flat_values = values.reshape(-1)
    expected = np.empty_like(flat_values)
    for i, point in enumerate(flat_points):
        dist = np.linalg.norm(flat_points - point, axis=1)
        active = dist <= length
        raw = flat_weights[active] * np.exp(-4.0 * dist[active] ** 2 / length**2)
        expected[i] = raw @ flat_values[active] / np.sum(raw)

    assert np.allclose(qmap.apply(values).reshape(-1), expected)
    assert np.allclose(qmap.operator @ np.ones(qmap.n_points), 1.0)


def test_gaussian_nonlocal_quadrature_map_rejects_invalid_inputs() -> None:
    points = np.zeros((1, 2, 2), dtype=float)
    weights = np.ones((1, 2), dtype=float)

    with pytest.raises(ValueError, match="positive finite"):
        build_gaussian_nonlocal_quadrature_map(points, weights, characteristic_length=0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        build_gaussian_nonlocal_quadrature_map(points, np.zeros_like(weights), characteristic_length=1.0)


def test_volume_nonlocal_quadrature_map_layout_matches_dofhandler_volume_rule() -> None:
    mesh = build_kratos_consolidation_2d_mesh()
    mixed = MixedElement(mesh, field_specs={"ux": 1, "uy": 1, "p": 1})
    dh = DofHandler(mixed, method="cg")

    qmap, layout = build_volume_nonlocal_quadrature_map(
        dh,
        quadrature_order=2,
        characteristic_length=2.0,
    )

    assert qmap.points.shape == (mesh.n_elements, layout.n_qp, 2)
    assert qmap.weights.shape == (mesh.n_elements, layout.n_qp)
    assert layout.entity_kind == "volume_cell"
    assert layout.cell_type == "quad"
    assert np.allclose(qmap.operator @ np.ones(qmap.n_points), 1.0)
