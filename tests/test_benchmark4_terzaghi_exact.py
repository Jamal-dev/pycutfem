from __future__ import annotations

import numpy as np

from examples.biofilms.benchmarks.poroelastic.paper1_benchmark4_terzaghi_consolidation import (
    TerzaghiParameters,
    terzaghi_pressure_bar_exact,
    terzaghi_settlement_bar_exact,
)


def test_terzaghi_exact_pressure_respects_mixed_boundary_conditions() -> None:
    params = TerzaghiParameters()
    t = 0.2 * (params.H**2) / params.consolidation_coefficient

    y = np.linspace(0.0, params.H, 4001)
    p_bar = terzaghi_pressure_bar_exact(y, t, params=params, n_terms=600)

    # Drained top boundary.
    assert abs(float(p_bar[-1])) < 1.0e-10

    # Impermeable base: dp/dy = 0 at y = 0.
    dpdy0 = float((p_bar[1] - p_bar[0]) / (y[1] - y[0]))
    assert abs(dpdy0) < 5.0e-3

    # Pressure must decay from the impermeable base to the drained top.
    assert float(p_bar[0]) > float(p_bar[len(p_bar) // 2]) > float(p_bar[-1])


def test_terzaghi_exact_settlement_bar_relaxes_to_one() -> None:
    params = TerzaghiParameters()
    t_small = 0.1 * (params.H**2) / params.consolidation_coefficient
    t_large = 5.0 * (params.H**2) / params.consolidation_coefficient

    s_small = float(terzaghi_settlement_bar_exact(t_small, params=params, n_terms=600))
    s_large = float(terzaghi_settlement_bar_exact(t_large, params=params, n_terms=600))

    assert 0.0 < s_small < 1.0
    assert 0.99 < s_large < 1.000001
    assert s_large > s_small
