import math

import numpy as np

from examples.NIRB.run_example2_local import (
    _SampledLSPGHybridModel,
    _certified_hrom_relaxation_factor,
    _fluid_hrom_interface_trust_region,
)


def _minimal_model(**kwargs):
    defaults = dict(
        basis=np.eye(3, 2),
        free_dofs=np.array([0, 1, 2]),
        sample_row_dofs=np.array([0, 2]),
        sample_element_ids=np.array([0]),
        sample_weights=np.ones(2),
        sample_element_weights=np.ones(1),
        objective="sampled_lspg",
        max_iterations=4,
        residual_tol=1.0e-8,
        line_search=False,
        lspg_block_scale=False,
        lspg_block_scale_relative_floor=0.0,
        incompressibility_stabilization_scale=1.0,
        recommended_switch_iter=2,
        source_path=__file__,
    )
    defaults.update(kwargs)
    return _SampledLSPGHybridModel(**defaults)


def test_coefficient_manifold_distance_uses_training_statistics():
    model = _minimal_model(
        training_coefficient_mean=np.array([1.0, -1.0]),
        training_coefficient_scale=np.array([2.0, 0.5]),
    )

    distance = model.coefficient_manifold_distance(np.array([3.0, 0.0]))

    assert math.isclose(distance, math.sqrt(5.0))


def test_missing_training_statistics_disables_manifold_distance():
    model = _minimal_model()

    assert math.isnan(model.coefficient_manifold_distance(np.array([0.0, 0.0])))


def test_dual_weighted_residual_error_matches_dual_pairing():
    model = _minimal_model(dwr_dual=np.array([2.0, -3.0]))

    assert math.isclose(model.dual_weighted_residual_error(np.array([4.0, 1.0])), 5.0)


def test_interface_trust_region_rejects_large_uncorrected_update():
    result = _fluid_hrom_interface_trust_region(
        current_values=np.zeros((2, 2)),
        proposed_values=np.ones((2, 2)),
        previous_load_abs=0.1,
        mode="fallback",
        max_step_ratio=1.0,
        max_load_rel=float("inf"),
        min_correction_alpha=0.0,
    )

    assert not result.accepted
    assert result.reason == "outside_interface_trust_region"


def test_certified_hrom_relaxation_grows_only_when_gates_pass():
    omega, reason = _certified_hrom_relaxation_factor(
        base_omega=0.1,
        previous_omega=0.1,
        hrom_used=True,
        eta_gamma=1.0e-3,
        eta_gamma_tol=1.0e-2,
        manifold_distance=2.0,
        manifold_distance_max=5.0,
        contraction_ratio=0.5,
        contraction_ratio_max=0.9,
        growth=1.5,
        shrink=0.5,
        omega_min=1.0e-3,
        omega_max=1.0,
    )

    assert math.isclose(omega, 0.15)
    assert reason == "certified_growth"


def test_certified_hrom_relaxation_shrinks_on_bad_estimator():
    omega, reason = _certified_hrom_relaxation_factor(
        base_omega=0.2,
        previous_omega=0.2,
        hrom_used=True,
        eta_gamma=2.0e-2,
        eta_gamma_tol=1.0e-2,
        manifold_distance=2.0,
        manifold_distance_max=5.0,
        contraction_ratio=0.5,
        contraction_ratio_max=0.9,
        growth=1.5,
        shrink=0.25,
        omega_min=1.0e-3,
        omega_max=1.0,
    )

    assert math.isclose(omega, 0.05)
    assert reason == "certified_shrink"
