from examples.biofilms.deformation_only_mms_convergence import _add_eocs, _prune_report_row, _reported_error_metrics


def test_translation_reporting_prunes_auxiliary_mechanics_metrics():
    row = {
        "case": "translation",
        "nx": 8,
        "h": 0.125,
        "dt": 0.05,
        "theta": 1.0,
        "newton_iters": 2,
        "solve_seconds": 1.0,
        "alpha_transport_velocity": "biofilm_volume",
        "alpha_transport_form": "conservative_weak",
        "v_l2": 1.0e-4,
        "u_h1": 2.0e-4,
        "alpha_l2": 3.0e-2,
        "B_l2": 1.0e-2,
        "mu_alpha_l2": 2.0e-1,
        "alpha_h1": 8.0e-1,
        "B_h1": 3.0e-1,
        "mu_alpha_h1": 4.0,
        "param_support_physics": "internal_conversion",
    }

    pruned = _prune_report_row("translation", row)

    assert "v_l2" not in pruned
    assert "u_h1" not in pruned
    assert pruned["alpha_l2"] == row["alpha_l2"]
    assert pruned["B_h1"] == row["B_h1"]


def test_translation_eocs_only_cover_transport_target_metrics():
    rows = [
        _prune_report_row(
            "translation",
            {
                "case": "translation",
                "nx": 8,
                "h": 0.125,
                "dt": 0.05,
                "theta": 1.0,
                "newton_iters": 3,
                "solve_seconds": 1.0,
                "alpha_transport_velocity": "biofilm_volume",
                "alpha_transport_form": "conservative_weak",
                "alpha_l2": 2.0e-2,
                "B_l2": 1.0e-2,
                "mu_alpha_l2": 1.0e-1,
                "alpha_h1": 8.0e-1,
                "B_h1": 3.0e-1,
                "mu_alpha_h1": 4.0,
            },
        ),
        _prune_report_row(
            "translation",
            {
                "case": "translation",
                "nx": 16,
                "h": 0.0625,
                "dt": 0.05,
                "theta": 1.0,
                "newton_iters": 2,
                "solve_seconds": 1.0,
                "alpha_transport_velocity": "biofilm_volume",
                "alpha_transport_form": "conservative_weak",
                "alpha_l2": 5.0e-3,
                "B_l2": 2.5e-3,
                "mu_alpha_l2": 2.5e-2,
                "alpha_h1": 4.0e-1,
                "B_h1": 1.5e-1,
                "mu_alpha_h1": 2.0,
            },
        ),
    ]

    out = _add_eocs(rows, case="translation")

    for metric in _reported_error_metrics("translation"):
        assert f"eoc_{metric}" in out[1]
    assert "eoc_u_h1" not in out[1]
    assert "eoc_v_l2" not in out[1]
