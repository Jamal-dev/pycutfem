from __future__ import annotations

import numpy as np

from examples.utils.biofilm.interface_transport_cases import build_interface_transport_case


def test_interface_transport_cases_are_divergence_free_and_pure_transport() -> None:
    sample_points = np.array(
        [
            [0.23, 0.41],
            [0.51, 0.72],
            [0.77, 0.35],
        ],
        dtype=float,
    )
    for key in ("translation", "rotation", "shear_return"):
        case = build_interface_transport_case(key)
        assert case.M_alpha == 0.0
        assert case.gamma_alpha > 0.0
        centroid0 = np.asarray(case.centroid_exact(0.0), dtype=float)
        assert np.allclose(centroid0, np.asarray(case.centroid0, dtype=float))

        div_vals = np.asarray(
            case.div_velocity(sample_points[:, 0], sample_points[:, 1], 0.37 * case.t_final),
            dtype=float,
        )
        assert np.max(np.abs(div_vals)) <= 1.0e-12
